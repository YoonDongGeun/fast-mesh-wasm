/**
 * Zero-dependency ZIP reader and writer (`STORE` and `DEFLATE`) with hostile-input guards.
 *
 * Uses Web-standard `DecompressionStream("deflate-raw")` and
 * `CompressionStream("deflate-raw")` with IEEE CRC-32 verification and strict
 * bounds checks to prevent zip bombs, OOM allocations, and Zip Slip traversal.
 */

export interface ZipEntryInput {
  /** Archive-relative POSIX path (e.g. `"3D/3dmodel.model"`). */
  name: string;
  /** Uncompressed entry payload. */
  data: Uint8Array;
  /** Override per-entry compression (`true` = deflate when smaller, `false` = store). */
  compress?: boolean;
}

export interface ZipReadOptions {
  /** Maximum accepted ZIP archive size in bytes. Default: 512 MiB. */
  maxFileBytes?: number;
  /** Maximum accepted number of ZIP entries. Default: 4096. */
  maxEntries?: number;
  /** Maximum uncompressed bytes allowed for any single entry. Default: `maxFileBytes`. */
  maxEntryBytes?: number;
  /** Maximum cumulative uncompressed bytes across all entries. Default: `maxFileBytes`. */
  maxTotalUncompressedBytes?: number;
  /** Verify IEEE CRC-32 checksums on read entries. Default: `true`. */
  verifyCrc?: boolean;
  /** Optional abort signal to cancel extraction mid-stream. */
  signal?: AbortSignal;
}

export interface ZipWriteOptions {
  /** Compress entries using raw DEFLATE (`CompressionStream("deflate-raw")`). Default: `true`. */
  compress?: boolean;
}

export interface ZipEntry {
  name: string;
  data: Uint8Array;
  method: number;
  crc32: number;
  compressedSize: number;
  uncompressedSize: number;
}

export const DEFAULT_MAX_ZIP_FILE_BYTES = 512 * 1024 * 1024;
export const DEFAULT_MAX_ZIP_ENTRIES = 4096;

const LOCAL_FILE_HEADER_SIG = 0x04034b50;
const CENTRAL_DIR_HEADER_SIG = 0x02014b50;
const EOCD_SIG = 0x06054b50;
const ZIP64_EOCD_SIG = 0x06064b50;
const ZIP64_EOCD_LOCATOR_SIG = 0x07064b50;

const METHOD_STORE = 0;
const METHOD_DEFLATE = 8;

const FLAG_ENCRYPTED = 0x0001;
const FLAG_DATA_DESCRIPTOR = 0x0008;
const FLAG_UTF8 = 0x0800;

const ZIP64_SENTINEL_U16 = 0xffff;
const ZIP64_SENTINEL_U32 = 0xffffffff;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder("utf-8", { fatal: false });

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) !== 0 ? (0xedb88320 ^ (c >>> 1)) >>> 0 : c >>> 1;
    }
    table[i] = c;
  }
  return table;
})();

/**
 * Compute standard IEEE 802.3 CRC-32 checksum.
 */
export function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i++) {
    crc = (CRC32_TABLE[(crc ^ data[i]) & 0xff] ^ (crc >>> 8)) >>> 0;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

interface RawCompressionStreamCtor {
  new (format: "deflate-raw" | "deflate" | "gzip"): {
    readable: ReadableStream<Uint8Array>;
    writable: WritableStream<Uint8Array>;
  };
}

function getCompressionCtor(
  kind: "CompressionStream" | "DecompressionStream",
): RawCompressionStreamCtor | undefined {
  const g = globalThis as unknown as Record<string, unknown>;
  const ctor = g[kind];
  return typeof ctor === "function" ? (ctor as RawCompressionStreamCtor) : undefined;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    const reason = signal.reason;
    if (reason instanceof Error) {
      throw reason;
    }
    throw new DOMException("The operation was aborted.", "AbortError");
  }
}

/**
 * Reject absolute paths, drive letters, NUL bytes, backslashes, and `..` traversal.
 */
export function validateZipEntryName(name: string): string {
  if (name.length === 0) {
    throw new Error("ZIP archive contains an entry with an empty filename");
  }
  if (name.includes("\0")) {
    throw new Error(`ZIP entry name contains NUL byte: ${JSON.stringify(name)}`);
  }
  if (name.startsWith("/") || name.startsWith("\\")) {
    throw new Error(`ZIP entry name must be relative (unsafe path "${name}")`);
  }
  if (/^[a-zA-Z]:/.test(name)) {
    throw new Error(`ZIP entry name must not contain a drive prefix ("${name}")`);
  }
  if (name.includes("\\")) {
    throw new Error(`ZIP entry name contains backslash ("${name}")`);
  }
  const segments = name.split("/");
  for (let i = 0; i < segments.length; i++) {
    if (segments[i] === "..") {
      throw new Error(`ZIP entry name attempts path traversal ("${name}")`);
    }
  }
  return name;
}

interface EocdInfo {
  totalEntries: number;
  cdSize: number;
  cdOffset: number;
}

function readU64AsSafeNumber(view: DataView, offset: number, label: string): number {
  const lo = view.getUint32(offset, true);
  const hi = view.getUint32(offset + 4, true);
  const val = hi * 4294967296 + lo;
  if (!Number.isSafeInteger(val)) {
    throw new Error(`ZIP64 ${label} exceeds safe integer range`);
  }
  return val;
}

function findEocd(bytes: Uint8Array, view: DataView): EocdInfo {
  const len = bytes.byteLength;
  if (len < 22) {
    throw new Error("Invalid ZIP archive: file is smaller than End of Central Directory record");
  }
  const minOffset = Math.max(0, len - 22 - 0xffff);
  let eocdOffset = -1;
  for (let i = len - 22; i >= minOffset; i--) {
    if (
      bytes[i] === 0x50 &&
      bytes[i + 1] === 0x4b &&
      bytes[i + 2] === 0x05 &&
      bytes[i + 3] === 0x06
    ) {
      const commentLen = view.getUint16(i + 20, true);
      if (i + 22 + commentLen === len) {
        eocdOffset = i;
        break;
      }
    }
  }
  if (eocdOffset < 0) {
    throw new Error("Invalid ZIP archive: End of Central Directory record not found");
  }

  const diskNumber = view.getUint16(eocdOffset + 4, true);
  const cdDisk = view.getUint16(eocdOffset + 6, true);
  let diskEntries = view.getUint16(eocdOffset + 8, true);
  let totalEntries = view.getUint16(eocdOffset + 10, true);
  let cdSize = view.getUint32(eocdOffset + 12, true);
  let cdOffset = view.getUint32(eocdOffset + 16, true);

  if (
    diskNumber === ZIP64_SENTINEL_U16 ||
    cdDisk === ZIP64_SENTINEL_U16 ||
    diskEntries === ZIP64_SENTINEL_U16 ||
    totalEntries === ZIP64_SENTINEL_U16 ||
    cdSize === ZIP64_SENTINEL_U32 ||
    cdOffset === ZIP64_SENTINEL_U32
  ) {
    const locOffset = eocdOffset - 20;
    if (locOffset < 0 || view.getUint32(locOffset, true) !== ZIP64_EOCD_LOCATOR_SIG) {
      throw new Error("Invalid ZIP64 archive: missing ZIP64 End of Central Directory Locator");
    }
    const zip64EocdOffset = readU64AsSafeNumber(view, locOffset + 8, "EOCD offset");
    if (zip64EocdOffset + 56 > locOffset || view.getUint32(zip64EocdOffset, true) !== ZIP64_EOCD_SIG) {
      throw new Error("Invalid ZIP64 archive: End of Central Directory record is corrupt");
    }
    diskEntries = readU64AsSafeNumber(view, zip64EocdOffset + 24, "disk entry count");
    totalEntries = readU64AsSafeNumber(view, zip64EocdOffset + 32, "total entry count");
    cdSize = readU64AsSafeNumber(view, zip64EocdOffset + 40, "central directory size");
    cdOffset = readU64AsSafeNumber(view, zip64EocdOffset + 48, "central directory offset");
  } else if (diskNumber !== 0 || cdDisk !== 0 || diskEntries !== totalEntries) {
    throw new Error("Multi-disk ZIP archives are not supported");
  }

  if (cdOffset > eocdOffset || cdOffset + cdSize > eocdOffset) {
    throw new Error("Invalid ZIP archive: Central Directory extends outside archive bounds");
  }

  return { totalEntries, cdSize, cdOffset };
}

interface CentralDirEntry {
  name: string;
  flags: number;
  method: number;
  crc32: number;
  compressedSize: number;
  uncompressedSize: number;
  localHeaderOffset: number;
}

function parseZip64ExtraField(
  bytes: Uint8Array,
  view: DataView,
  extraStart: number,
  extraLen: number,
  needUncompressed: boolean,
  needCompressed: boolean,
  needOffset: boolean,
): { uncompressed?: number; compressed?: number; offset?: number } {
  let pos = extraStart;
  const end = extraStart + extraLen;
  while (pos + 4 <= end) {
    const headerId = view.getUint16(pos, true);
    const dataSize = view.getUint16(pos + 2, true);
    const dataStart = pos + 4;
    if (dataStart + dataSize > end) {
      break;
    }
    if (headerId === 0x0001) {
      let cur = dataStart;
      const out: { uncompressed?: number; compressed?: number; offset?: number } = {};
      if (needUncompressed) {
        if (cur + 8 > dataStart + dataSize) {
          throw new Error("Invalid ZIP64 extra field: truncated uncompressed size");
        }
        out.uncompressed = readU64AsSafeNumber(view, cur, "uncompressed size");
        cur += 8;
      }
      if (needCompressed) {
        if (cur + 8 > dataStart + dataSize) {
          throw new Error("Invalid ZIP64 extra field: truncated compressed size");
        }
        out.compressed = readU64AsSafeNumber(view, cur, "compressed size");
        cur += 8;
      }
      if (needOffset) {
        if (cur + 8 > dataStart + dataSize) {
          throw new Error("Invalid ZIP64 extra field: truncated local header offset");
        }
        out.offset = readU64AsSafeNumber(view, cur, "local header offset");
      }
      return out;
    }
    pos = dataStart + dataSize;
  }
  void bytes;
  throw new Error("Invalid ZIP64 entry: required ZIP64 extra field missing");
}

function parseCentralDirectory(
  bytes: Uint8Array,
  view: DataView,
  cdOffset: number,
  cdSize: number,
  expectedEntries: number,
  maxEntries: number,
): CentralDirEntry[] {
  if (expectedEntries > maxEntries) {
    throw new Error(
      `ZIP archive has ${expectedEntries} entries, exceeding maximum of ${maxEntries}`,
    );
  }

  const entries: CentralDirEntry[] = [];
  let pos = cdOffset;
  const end = cdOffset + cdSize;

  while (pos < end) {
    if (entries.length >= maxEntries) {
      throw new Error(`ZIP archive entry count exceeds maximum of ${maxEntries}`);
    }
    if (pos + 46 > end) {
      throw new Error("Invalid ZIP archive: truncated Central Directory entry");
    }
    if (view.getUint32(pos, true) !== CENTRAL_DIR_HEADER_SIG) {
      throw new Error("Invalid ZIP archive: bad Central Directory entry signature");
    }

    const flags = view.getUint16(pos + 8, true);
    const method = view.getUint16(pos + 10, true);
    const entryCrc = view.getUint32(pos + 16, true);
    let compressedSize = view.getUint32(pos + 20, true);
    let uncompressedSize = view.getUint32(pos + 24, true);
    const nameLen = view.getUint16(pos + 28, true);
    const extraLen = view.getUint16(pos + 30, true);
    const commentLen = view.getUint16(pos + 32, true);
    let localHeaderOffset = view.getUint32(pos + 42, true);

    const recordEnd = pos + 46 + nameLen + extraLen + commentLen;
    if (recordEnd > end) {
      throw new Error("Invalid ZIP archive: Central Directory entry extends past directory end");
    }

    const name = validateZipEntryName(
      textDecoder.decode(bytes.subarray(pos + 46, pos + 46 + nameLen)),
    );

    if (
      uncompressedSize === ZIP64_SENTINEL_U32 ||
      compressedSize === ZIP64_SENTINEL_U32 ||
      localHeaderOffset === ZIP64_SENTINEL_U32
    ) {
      const z64 = parseZip64ExtraField(
        bytes,
        view,
        pos + 46 + nameLen,
        extraLen,
        uncompressedSize === ZIP64_SENTINEL_U32,
        compressedSize === ZIP64_SENTINEL_U32,
        localHeaderOffset === ZIP64_SENTINEL_U32,
      );
      if (z64.uncompressed !== undefined) uncompressedSize = z64.uncompressed;
      if (z64.compressed !== undefined) compressedSize = z64.compressed;
      if (z64.offset !== undefined) localHeaderOffset = z64.offset;
    }

    entries.push({
      name,
      flags,
      method,
      crc32: entryCrc,
      compressedSize,
      uncompressedSize,
      localHeaderOffset,
    });

    pos = recordEnd;
  }

  if (entries.length !== expectedEntries) {
    throw new Error(
      `Invalid ZIP archive: Central Directory declares ${expectedEntries} entries, found ${entries.length}`,
    );
  }
  return entries;
}

/**
 * Inflate raw DEFLATE bytes with hard output-byte limits and abort support.
 * Reads output chunks concurrently while feeding bounded input chunks so a zip
 * bomb fails immediately instead of buffering GBs in memory.
 */
async function inflateRawBounded(
  compressed: Uint8Array,
  maxBytes: number,
  entryName: string,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  throwIfAborted(signal);
  const Ctor = getCompressionCtor("DecompressionStream");
  if (!Ctor) {
    throw new Error(
      "DecompressionStream('deflate-raw') is not available in this runtime",
    );
  }

  const stream = new Ctor("deflate-raw");
  const writer = stream.writable.getWriter();
  const reader = stream.readable.getReader();

  const chunks: Uint8Array[] = [];
  let totalOut = 0;
  let failure: unknown = null;
  let aborted = false;

  const cancelAll = async (err: unknown): Promise<void> => {
    if (aborted) return;
    aborted = true;
    failure = err;
    await Promise.allSettled([reader.cancel(err), writer.abort(err)]);
  };

  const onAbort = () => {
    const err =
      signal?.reason instanceof Error
        ? signal.reason
        : new DOMException("The operation was aborted.", "AbortError");
    void cancelAll(err);
  };
  signal?.addEventListener("abort", onAbort, { once: true });

  const writePump = (async () => {
    const CHUNK = 64 * 1024;
    try {
      for (let off = 0; off < compressed.byteLength; off += CHUNK) {
        if (aborted) return;
        throwIfAborted(signal);
        const slice = compressed.subarray(off, Math.min(off + CHUNK, compressed.byteLength));
        await writer.write(slice);
      }
      if (!aborted) {
        await writer.close();
      }
    } catch (err) {
      if (!aborted) {
        await cancelAll(err);
      }
    }
  })();

  const readPump = (async () => {
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (aborted) break;
        throwIfAborted(signal);
        if (value && value.byteLength > 0) {
          if (value.byteLength > maxBytes - totalOut) {
            const err = new Error(
              `ZIP entry "${entryName}" exceeds uncompressed limit of ${maxBytes} bytes (zip bomb guard)`,
            );
            await cancelAll(err);
            return;
          }
          totalOut += value.byteLength;
          // Copy slice in case the underlying stream recycles its chunk buffer
          chunks.push(new Uint8Array(value));
        }
      }
    } catch (err) {
      if (!aborted) {
        await cancelAll(err);
      }
    }
  })();

  try {
    await Promise.all([writePump, readPump]);
  } finally {
    signal?.removeEventListener("abort", onAbort);
  }

  if (failure !== null) {
    throw failure instanceof Error ? failure : new Error(String(failure));
  }
  throwIfAborted(signal);

  const out = new Uint8Array(totalOut);
  let pos = 0;
  for (let i = 0; i < chunks.length; i++) {
    out.set(chunks[i], pos);
    pos += chunks[i].byteLength;
  }
  return out;
}

async function deflateRaw(data: Uint8Array): Promise<Uint8Array> {
  const Ctor = getCompressionCtor("CompressionStream");
  if (!Ctor) {
    throw new Error(
      "CompressionStream('deflate-raw') is not available in this runtime",
    );
  }

  const stream = new Ctor("deflate-raw");
  const writer = stream.writable.getWriter();
  const reader = stream.readable.getReader();

  const chunks: Uint8Array[] = [];
  let totalOut = 0;

  const writePump = (async () => {
    const CHUNK = 64 * 1024;
    for (let off = 0; off < data.byteLength; off += CHUNK) {
      await writer.write(data.subarray(off, Math.min(off + CHUNK, data.byteLength)));
    }
    await writer.close();
  })();

  const readPump = (async () => {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value && value.byteLength > 0) {
        totalOut += value.byteLength;
        chunks.push(new Uint8Array(value));
      }
    }
  })();

  await Promise.all([writePump, readPump]);
  const out = new Uint8Array(totalOut);
  let pos = 0;
  for (let i = 0; i < chunks.length; i++) {
    out.set(chunks[i], pos);
    pos += chunks[i].byteLength;
  }
  return out;
}

function locateEntryPayload(
  bytes: Uint8Array,
  view: DataView,
  cdOffset: number,
  cdEntry: CentralDirEntry,
): Uint8Array {
  const lh = cdEntry.localHeaderOffset;
  if (lh + 30 > cdOffset) {
    throw new Error(
      `Invalid ZIP entry "${cdEntry.name}": local header offset ${lh} is outside archive data`,
    );
  }
  if (view.getUint32(lh, true) !== LOCAL_FILE_HEADER_SIG) {
    throw new Error(
      `Invalid ZIP entry "${cdEntry.name}": bad local file header signature`,
    );
  }

  const nameLen = view.getUint16(lh + 26, true);
  const extraLen = view.getUint16(lh + 28, true);
  const dataStart = lh + 30 + nameLen + extraLen;
  if (dataStart > cdOffset) {
    throw new Error(
      `Invalid ZIP entry "${cdEntry.name}": local file header extends into Central Directory`,
    );
  }
  if (cdEntry.compressedSize > cdOffset - dataStart) {
    throw new Error(
      `Invalid ZIP entry "${cdEntry.name}": compressed size (${cdEntry.compressedSize}) exceeds available bytes`,
    );
  }

  return bytes.subarray(dataStart, dataStart + cdEntry.compressedSize);
}

/**
 * Parse and decompress all entries in a ZIP archive with strict bounds checks.
 */
export async function readZip(
  bytes: Uint8Array,
  options?: ZipReadOptions,
): Promise<Map<string, Uint8Array>> {
  const maxFileBytes = options?.maxFileBytes ?? DEFAULT_MAX_ZIP_FILE_BYTES;
  const maxEntries = options?.maxEntries ?? DEFAULT_MAX_ZIP_ENTRIES;
  const maxEntryBytes = options?.maxEntryBytes ?? maxFileBytes;
  const maxTotalBytes = options?.maxTotalUncompressedBytes ?? maxFileBytes;
  const verifyCrc = options?.verifyCrc ?? true;
  const signal = options?.signal;

  throwIfAborted(signal);
  if (bytes.byteLength > maxFileBytes) {
    throw new Error(
      `ZIP archive size (${bytes.byteLength} bytes) exceeds limit of ${maxFileBytes} bytes`,
    );
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const eocd = findEocd(bytes, view);
  const cdEntries = parseCentralDirectory(
    bytes,
    view,
    eocd.cdOffset,
    eocd.cdSize,
    eocd.totalEntries,
    maxEntries,
  );

  const result = new Map<string, Uint8Array>();
  let totalUncompressed = 0;

  for (let i = 0; i < cdEntries.length; i++) {
    throwIfAborted(signal);
    const entry = cdEntries[i];

    if ((entry.flags & FLAG_ENCRYPTED) !== 0) {
      throw new Error(`ZIP entry "${entry.name}" is encrypted, which is not supported`);
    }
    if (entry.method !== METHOD_STORE && entry.method !== METHOD_DEFLATE) {
      throw new Error(
        `ZIP entry "${entry.name}" uses unsupported compression method ${entry.method}`,
      );
    }

    const remainingTotal = maxTotalBytes - totalUncompressed;
    const entryBudget = Math.min(maxEntryBytes, remainingTotal);

    // Reject absurd claimed uncompressed sizes before decompressing, but NEVER trust
    // a small claim to preallocate or raise the streaming limit.
    if (entry.uncompressedSize > maxEntryBytes) {
      throw new Error(
        `ZIP entry "${entry.name}" declares ${entry.uncompressedSize} uncompressed bytes, exceeding single-entry limit of ${maxEntryBytes}`,
      );
    }
    if (entry.uncompressedSize > remainingTotal) {
      throw new Error(
        `ZIP archive uncompressed total exceeds limit of ${maxTotalBytes} bytes`,
      );
    }

    const compressedSlice = locateEntryPayload(bytes, view, eocd.cdOffset, entry);

    let payload: Uint8Array;
    if (entry.method === METHOD_STORE) {
      if (
        (entry.flags & FLAG_DATA_DESCRIPTOR) === 0 &&
        entry.compressedSize !== entry.uncompressedSize
      ) {
        throw new Error(
          `ZIP STORE entry "${entry.name}" has mismatched sizes (${entry.compressedSize} vs ${entry.uncompressedSize})`,
        );
      }
      if (compressedSlice.byteLength > entryBudget) {
        throw new Error(
          `ZIP entry "${entry.name}" exceeds uncompressed limit of ${entryBudget} bytes`,
        );
      }
      payload = compressedSlice.slice();
    } else {
      payload = await inflateRawBounded(
        compressedSlice,
        entryBudget,
        entry.name,
        signal,
      );
    }

    if (
      (entry.flags & FLAG_DATA_DESCRIPTOR) === 0 &&
      payload.byteLength !== entry.uncompressedSize
    ) {
      throw new Error(
        `ZIP entry "${entry.name}" decompressed to ${payload.byteLength} bytes, expected ${entry.uncompressedSize}`,
      );
    }

    if (verifyCrc) {
      const actualCrc = crc32(payload);
      if (actualCrc !== entry.crc32) {
        throw new Error(
          `ZIP entry "${entry.name}" failed CRC-32 check (expected 0x${entry.crc32.toString(16)}, got 0x${actualCrc.toString(16)})`,
        );
      }
    }

    totalUncompressed += payload.byteLength;
    if (totalUncompressed > maxTotalBytes) {
      throw new Error(
        `ZIP archive uncompressed total (${totalUncompressed} bytes) exceeds limit of ${maxTotalBytes} bytes`,
      );
    }

    // Directories end with '/'; still validate name above and store if needed
    result.set(entry.name, payload);
  }

  return result;
}

interface PreparedEntry {
  nameBytes: Uint8Array;
  payload: Uint8Array;
  method: number;
  crc: number;
  uncompressedSize: number;
  compressedSize: number;
  localOffset: number;
}

function assembleZip(prepared: PreparedEntry[]): Uint8Array {
  let localTotal = 0;
  let cdTotal = 0;
  for (let i = 0; i < prepared.length; i++) {
    const e = prepared[i];
    e.localOffset = localTotal;
    localTotal += 30 + e.nameBytes.byteLength + e.compressedSize;
    cdTotal += 46 + e.nameBytes.byteLength;
  }

  const totalSize = localTotal + cdTotal + 22;
  const out = new Uint8Array(totalSize);
  const view = new DataView(out.buffer);

  let pos = 0;
  for (let i = 0; i < prepared.length; i++) {
    const e = prepared[i];
    view.setUint32(pos, LOCAL_FILE_HEADER_SIG, true);
    view.setUint16(pos + 4, 20, true); // version needed 2.0
    view.setUint16(pos + 6, FLAG_UTF8, true);
    view.setUint16(pos + 8, e.method, true);
    view.setUint16(pos + 10, 0, true); // mod time
    view.setUint16(pos + 12, 0x5421, true); // fixed deterministic DOS date
    view.setUint32(pos + 14, e.crc, true);
    view.setUint32(pos + 18, e.compressedSize, true);
    view.setUint32(pos + 22, e.uncompressedSize, true);
    view.setUint16(pos + 26, e.nameBytes.byteLength, true);
    view.setUint16(pos + 28, 0, true); // extra length
    out.set(e.nameBytes, pos + 30);
    pos += 30 + e.nameBytes.byteLength;
    out.set(e.payload, pos);
    pos += e.compressedSize;
  }

  const cdOffset = pos;
  for (let i = 0; i < prepared.length; i++) {
    const e = prepared[i];
    view.setUint32(pos, CENTRAL_DIR_HEADER_SIG, true);
    view.setUint16(pos + 4, 20, true); // version made by
    view.setUint16(pos + 6, 20, true); // version needed
    view.setUint16(pos + 8, FLAG_UTF8, true);
    view.setUint16(pos + 10, e.method, true);
    view.setUint16(pos + 12, 0, true);
    view.setUint16(pos + 14, 0x5421, true);
    view.setUint32(pos + 16, e.crc, true);
    view.setUint32(pos + 20, e.compressedSize, true);
    view.setUint32(pos + 24, e.uncompressedSize, true);
    view.setUint16(pos + 28, e.nameBytes.byteLength, true);
    view.setUint16(pos + 30, 0, true); // extra
    view.setUint16(pos + 32, 0, true); // comment
    view.setUint16(pos + 34, 0, true); // disk number
    view.setUint16(pos + 36, 0, true); // internal attrs
    view.setUint32(pos + 38, 0, true); // external attrs
    view.setUint32(pos + 42, e.localOffset, true);
    out.set(e.nameBytes, pos + 46);
    pos += 46 + e.nameBytes.byteLength;
  }

  view.setUint32(pos, EOCD_SIG, true);
  view.setUint16(pos + 4, 0, true);
  view.setUint16(pos + 6, 0, true);
  view.setUint16(pos + 8, prepared.length, true);
  view.setUint16(pos + 10, prepared.length, true);
  view.setUint32(pos + 12, cdTotal, true);
  view.setUint32(pos + 16, cdOffset, true);
  view.setUint16(pos + 20, 0, true);

  return out;
}

/**
 * Create a ZIP archive synchronously using `STORE` (method 0).
 */
export function writeZipSync(entries: readonly ZipEntryInput[]): Uint8Array {
  if (entries.length > 0xffff) {
    throw new Error(`writeZipSync supports at most 65535 entries (got ${entries.length})`);
  }
  const seen = new Set<string>();
  const prepared: PreparedEntry[] = [];
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    const name = validateZipEntryName(entry.name);
    if (seen.has(name)) {
      throw new Error(`Duplicate ZIP entry name "${name}"`);
    }
    seen.add(name);
    const nameBytes = textEncoder.encode(name);
    prepared.push({
      nameBytes,
      payload: entry.data,
      method: METHOD_STORE,
      crc: crc32(entry.data),
      uncompressedSize: entry.data.byteLength,
      compressedSize: entry.data.byteLength,
      localOffset: 0,
    });
  }
  return assembleZip(prepared);
}

/**
 * Create a ZIP archive, compressing entries with raw DEFLATE when smaller.
 */
export async function writeZip(
  entries: readonly ZipEntryInput[],
  options?: ZipWriteOptions,
): Promise<Uint8Array> {
  if (entries.length > 0xffff) {
    throw new Error(`writeZip supports at most 65535 entries (got ${entries.length})`);
  }
  const canDeflate =
    (options?.compress ?? true) && getCompressionCtor("CompressionStream") !== undefined;
  const seen = new Set<string>();
  const prepared: PreparedEntry[] = [];

  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    const name = validateZipEntryName(entry.name);
    if (seen.has(name)) {
      throw new Error(`Duplicate ZIP entry name "${name}"`);
    }
    seen.add(name);
    const nameBytes = textEncoder.encode(name);
    const shouldCompress = entry.compress ?? canDeflate;
    let payload = entry.data;
    let method = METHOD_STORE;
    if (shouldCompress && canDeflate && entry.data.byteLength > 0) {
      const deflated = await deflateRaw(entry.data);
      if (deflated.byteLength < entry.data.byteLength) {
        payload = deflated;
        method = METHOD_DEFLATE;
      }
    }
    prepared.push({
      nameBytes,
      payload,
      method,
      crc: crc32(entry.data),
      uncompressedSize: entry.data.byteLength,
      compressedSize: payload.byteLength,
      localOffset: 0,
    });
  }

  return assembleZip(prepared);
}
