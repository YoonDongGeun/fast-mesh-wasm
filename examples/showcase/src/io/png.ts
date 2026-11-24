// 16-bit PNG, without a dependency: the encoder for a height map (one 16-bit
// grey channel, which a canvas cannot write) and the decoder for PNGs with
// full 8 or 16-bit channels (a canvas reads only 8 bits of 16). Deflate is the
// platform's `CompressionStream`, and the checksum the library's CRC-32.
//
// Other images (interlaced, palette and low-depth PNGs, JPEG, WebP, ...) are
// left to the browser's own decoder: see `height-image.ts`.
import { crc32 } from "fast-mesh-wasm/io";

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/** The most deflated bytes to one IDAT chunk. */
const IDAT_BYTES = 1 << 20;

/** Channels of each colour type (grey, RGB, palette, grey + alpha, RGBA). */
const CHANNELS: Readonly<Record<number, number>> = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

/** A PNG's samples as stored: row by row, `channels` to a pixel, at `depth` bits. */
export interface PngImage {
  width: number;
  height: number;
  /** 1 grey, 2 grey and alpha, 3 RGB, 4 RGBA. */
  channels: number;
  /** 8 or 16 bits per sample. */
  depth: number;
  samples: Uint8Array | Uint16Array;
}

/** A PNG's size and format, from its header. */
export interface PngHeader {
  width: number;
  height: number;
  depth: number;
  colorType: number;
  interlaced: boolean;
}

/** One chunk: length, type, data, CRC of the type and the data. */
function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) {
    out[4 + i] = type.charCodeAt(i);
  }
  out.set(data, 8);
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

function join(parts: readonly Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

/** Everything `stream` produces from the pieces of `input`. */
async function pump(
  input: Iterable<Uint8Array>,
  stream: { readable: ReadableStream<Uint8Array>; writable: WritableStream<BufferSource> },
): Promise<Uint8Array[]> {
  const writer = stream.writable.getWriter();
  const write = (async () => {
    for (const piece of input) {
      await writer.ready;
      await writer.write(piece as Uint8Array<ArrayBuffer>);
    }
    await writer.close();
  })();
  // A failure shows on the reading side; this only keeps it from being unhandled.
  write.catch(() => {});
  const out: Uint8Array[] = [];
  const reader = stream.readable.getReader();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    out.push(value);
  }
  await write;
  return out;
}

/** What a byte counts as when comparing filters: its distance from 0, as a signed byte. */
const cost = (byte: number): number => (byte < 128 ? byte : 256 - byte);

/** A 16-bit grey image's filtered scanlines, `rows` rows at a time. */
function* scanlines(codes: Uint16Array, width: number, height: number): Generator<Uint8Array> {
  const stride = 2 * width;
  const rows = Math.max(1, Math.floor(IDAT_BYTES / (stride + 1)));
  let prev = new Uint8Array(stride);
  let cur = new Uint8Array(stride);
  const sub = new Uint8Array(stride);
  const up = new Uint8Array(stride);
  for (let first = 0; first < height; first += rows) {
    const end = Math.min(first + rows, height);
    const out = new Uint8Array((end - first) * (stride + 1));
    for (let row = first; row < end; row++) {
      let subCost = 0;
      let upCost = 0;
      for (let col = 0; col < width; col++) {
        const code = codes[row * width + col];
        cur[2 * col] = code >> 8;
        cur[2 * col + 1] = code & 255;
      }
      for (let i = 0; i < stride; i++) {
        sub[i] = cur[i] - (i < 2 ? 0 : cur[i - 2]);
        up[i] = cur[i] - prev[i];
        subCost += cost(sub[i]);
        upCost += cost(up[i]);
      }
      const at = (row - first) * (stride + 1);
      out[at] = upCost < subCost ? 2 : 1;
      out.set(upCost < subCost ? up : sub, at + 1);
      [prev, cur] = [cur, prev];
    }
    yield out;
  }
}

/** Encodes `codes` (row by row from the top left) as a 16-bit grey PNG. */
export async function encodePng16(codes: Uint16Array, width: number, height: number): Promise<Uint8Array> {
  if (!(Number.isInteger(width) && Number.isInteger(height) && width > 0 && height > 0)) {
    throw new RangeError("a PNG needs a width and a height");
  }
  if (codes.length !== width * height) {
    throw new RangeError("codes must hold width times height values");
  }
  const header = new Uint8Array(13);
  const view = new DataView(header.buffer);
  view.setUint32(0, width);
  view.setUint32(4, height);
  header[8] = 16;
  const deflated = await pump(scanlines(codes, width, height), new CompressionStream("deflate") as never);
  const parts: Uint8Array[] = [Uint8Array.from(SIGNATURE), chunk("IHDR", header)];
  for (const piece of deflated) {
    for (let at = 0; at < piece.length; at += IDAT_BYTES) {
      parts.push(chunk("IDAT", piece.subarray(at, at + IDAT_BYTES)));
    }
  }
  parts.push(chunk("IEND", new Uint8Array(0)));
  return join(parts);
}

/** Whether `bytes` start as a PNG does. */
export function isPng(bytes: Uint8Array): boolean {
  return bytes.length >= 8 && SIGNATURE.every((b, i) => bytes[i] === b);
}

/** The header of a PNG, or `null` for anything else. */
export function readPngHeader(bytes: Uint8Array): PngHeader | null {
  if (!isPng(bytes) || bytes.length < 33) {
    return null;
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(8) !== 13 || view.getUint32(12) !== 0x49484452) {
    return null;
  }
  return {
    width: view.getUint32(16),
    height: view.getUint32(20),
    depth: bytes[24],
    colorType: bytes[25],
    interlaced: bytes[28] !== 0,
  };
}

/** Whether {@link decodePng} reads this PNG: whole 8 or 16-bit channels, no palette, not interlaced. */
export function canDecodePng(h: PngHeader): boolean {
  return (h.depth === 8 || h.depth === 16) && !h.interlaced && h.colorType !== 3 && CHANNELS[h.colorType] !== undefined;
}

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

/** Undoes the filter of each scanline of `raw`, in place. */
function unfilter(raw: Uint8Array, stride: number, height: number, bpp: number): void {
  for (let row = 0; row < height; row++) {
    const line = row * (stride + 1);
    const filter = raw[line];
    const at = line + 1;
    const above = at - (stride + 1);
    if (filter > 4) {
      throw new Error("the PNG has an unknown scanline filter");
    }
    if (filter === 0) {
      continue;
    }
    for (let i = 0; i < stride; i++) {
      const a = i < bpp ? 0 : raw[at + i - bpp];
      const b = row === 0 ? 0 : raw[above + i];
      const c = i < bpp || row === 0 ? 0 : raw[above + i - bpp];
      const add = filter === 1 ? a : filter === 2 ? b : filter === 3 ? (a + b) >> 1 : paeth(a, b, c);
      raw[at + i] += add;
    }
  }
}

/**
 * Decodes a PNG that {@link canDecodePng} accepts to its stored samples (no
 * gamma or colour profile applied).
 *
 * @throws on a damaged file, or one it does not read.
 */
export async function decodePng(bytes: Uint8Array): Promise<PngImage> {
  const header = readPngHeader(bytes);
  if (header === null) {
    throw new Error("not a PNG file");
  }
  if (!canDecodePng(header)) {
    throw new Error("this kind of PNG is not supported");
  }
  const { width, height, depth } = header;
  if (width === 0 || height === 0) {
    throw new Error("the PNG has no pixels");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const idat: Uint8Array[] = [];
  for (let at = 8; at + 12 <= bytes.length; ) {
    const length = view.getUint32(at);
    const type = view.getUint32(at + 4);
    if (at + 12 + length > bytes.length) {
      throw new Error("the PNG file is cut short");
    }
    if (type === 0x49444154) {
      idat.push(bytes.subarray(at + 8, at + 8 + length));
    } else if (type === 0x49454e44) {
      break;
    }
    at += 12 + length;
  }
  const channels = CHANNELS[header.colorType];
  const bpp = (channels * depth) / 8;
  const stride = width * bpp;
  const size = (stride + 1) * height;
  let raw: Uint8Array;
  try {
    raw = join(await pump(idat, new DecompressionStream("deflate") as never));
  } catch {
    throw new Error("the PNG's image data is damaged");
  }
  if (raw.length < size) {
    throw new Error("the PNG file is cut short");
  }
  unfilter(raw, stride, height, bpp);
  const count = width * channels;
  const samples = depth === 8 ? new Uint8Array(count * height) : new Uint16Array(count * height);
  for (let row = 0; row < height; row++) {
    const at = row * (stride + 1) + 1;
    if (depth === 8) {
      samples.set(raw.subarray(at, at + stride), row * count);
    } else {
      for (let i = 0; i < count; i++) {
        samples[row * count + i] = (raw[at + 2 * i] << 8) | raw[at + 2 * i + 1];
      }
    }
  }
  return { width, height, channels, depth, samples };
}
