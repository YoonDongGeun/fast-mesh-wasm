/**
 * Pure-TS binary & ASCII STL importer with optional spatial-hash vertex welding.
 */
import type { ImportedMesh } from "./index.js";

/** Options for {@link importSTL}. */
export interface StlImportOptions {
  /** Override mesh name. Defaults to ASCII `solid` name or `"mesh"`. */
  name?: string;
  /** Weld coincident vertices (`-0` normalized to `0`) via spatial hashing. Default: `false`. */
  deduplicateVertices?: boolean;
  /** Maximum accepted file size in bytes. Default: 512 MiB. */
  maxFileBytes?: number;
  /** Optional abort signal. */
  signal?: AbortSignal;
}

const DEFAULT_MAX_BYTES = 512 * 1024 * 1024;
const textDecoder = new TextDecoder("utf-8");

function toUint8(bytes: Uint8Array | ArrayBuffer): Uint8Array {
  return bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
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

function isProbablyAsciiStl(bytes: Uint8Array): boolean {
  const probeLen = Math.min(bytes.byteLength, 512);
  let i = 0;
  // Skip UTF-8 BOM if present
  if (
    probeLen >= 3 &&
    bytes[0] === 0xef &&
    bytes[1] === 0xbb &&
    bytes[2] === 0xbf
  ) {
    i = 3;
  }
  while (i < probeLen && (bytes[i] === 0x20 || bytes[i] === 0x09 || bytes[i] === 0x0a || bytes[i] === 0x0d)) {
    i++;
  }
  // Check "solid"
  if (i + 5 > probeLen) return false;
  if (
    bytes[i] !== 0x73 || // s
    bytes[i + 1] !== 0x6f || // o
    bytes[i + 2] !== 0x6c || // l
    bytes[i + 3] !== 0x69 || // i
    bytes[i + 4] !== 0x64 // d
  ) {
    return false;
  }
  if (i + 5 < probeLen) {
    const after = bytes[i + 5];
    if (after !== 0x20 && after !== 0x09 && after !== 0x0a && after !== 0x0d) {
      return false;
    }
  }
  // NUL or control bytes in the header indicate a binary STL whose 80-byte comment happens to start with "solid".
  for (let k = i; k < probeLen; k++) {
    const c = bytes[k];
    if (c === 0 || (c < 0x20 && c !== 0x09 && c !== 0x0a && c !== 0x0d)) {
      return false;
    }
  }
  return true;
}

class VertexWelder {
  private positions: Float64Array;
  private count = 0;
  private readonly head: Int32Array;
  private readonly next: Int32Array;
  private readonly mask: number;
  private readonly scratchBuf = new ArrayBuffer(12);
  private readonly scratchF32 = new Float32Array(this.scratchBuf);
  private readonly scratchI32 = new Int32Array(this.scratchBuf);

  constructor(maxVertices: number) {
    this.positions = new Float64Array(maxVertices * 3);
    let cap = 16;
    const target = Math.max(16, maxVertices);
    while (cap < target) cap <<= 1;
    this.mask = cap - 1;
    this.head = new Int32Array(cap).fill(-1);
    this.next = new Int32Array(maxVertices).fill(-1);
  }

  intern(xRaw: number, yRaw: number, zRaw: number): number {
    const x = xRaw === 0 ? 0 : xRaw;
    const y = yRaw === 0 ? 0 : yRaw;
    const z = zRaw === 0 ? 0 : zRaw;

    this.scratchF32[0] = x;
    this.scratchF32[1] = y;
    this.scratchF32[2] = z;
    // Use bit pattern of quantized Float32 for fast spatial hash, but verify exact Float64 match.
    const h =
      ((Math.imul(this.scratchI32[0], 73856093) ^
        Math.imul(this.scratchI32[1], 19349663) ^
        Math.imul(this.scratchI32[2], 83492791)) >>>
        0) &
      this.mask;

    let idx = this.head[h];
    while (idx !== -1) {
      const base = idx * 3;
      if (
        this.positions[base] === x &&
        this.positions[base + 1] === y &&
        this.positions[base + 2] === z
      ) {
        return idx;
      }
      idx = this.next[idx];
    }

    const newIdx = this.count++;
    const base = newIdx * 3;
    this.positions[base] = x;
    this.positions[base + 1] = y;
    this.positions[base + 2] = z;
    this.next[newIdx] = this.head[h];
    this.head[h] = newIdx;
    return newIdx;
  }

  finish(): Float64Array {
    return this.positions.subarray(0, this.count * 3).slice();
  }
}

function parseBinaryStl(
  bytes: Uint8Array,
  view: DataView,
  triCount: number,
  options?: StlImportOptions,
): ImportedMesh {
  const signal = options?.signal;
  const dedup = options?.deduplicateVertices === true;
  const maxVerts = triCount * 3;

  const indices = new Uint32Array(maxVerts);

  if (!dedup) {
    const positions = new Float64Array(maxVerts * 3);
    let offset = 84;
    for (let t = 0; t < triCount; t++) {
      if ((t & 0x3fff) === 0) throwIfAborted(signal);
      // Skip facet normal at +0..+11; read 3 vertices at +12..+47
      for (let v = 0; v < 3; v++) {
        const vo = offset + 12 + v * 12;
        const x = view.getFloat32(vo, true);
        const y = view.getFloat32(vo + 4, true);
        const z = view.getFloat32(vo + 8, true);
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
          throw new Error(
            `Invalid binary STL vertex at triangle ${t}: non-finite coordinate`,
          );
        }
        const vi = t * 3 + v;
        positions[vi * 3] = x;
        positions[vi * 3 + 1] = y;
        positions[vi * 3 + 2] = z;
        indices[vi] = vi;
      }
      offset += 50;
    }

    return {
      name: options?.name ?? "mesh",
      positions,
      indices,
      placement: new Float64Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ]),
      colors: null,
      uvs: null,
    };
  }

  const welder = new VertexWelder(maxVerts);
  let offset = 84;
  for (let t = 0; t < triCount; t++) {
    if ((t & 0x3fff) === 0) throwIfAborted(signal);
    for (let v = 0; v < 3; v++) {
      const vo = offset + 12 + v * 12;
      const x = view.getFloat32(vo, true);
      const y = view.getFloat32(vo + 4, true);
      const z = view.getFloat32(vo + 8, true);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        throw new Error(
          `Invalid binary STL vertex at triangle ${t}: non-finite coordinate`,
        );
      }
      indices[t * 3 + v] = welder.intern(x, y, z);
    }
    offset += 50;
  }

  return {
    name: options?.name ?? "mesh",
    positions: welder.finish(),
    indices,
    placement: new Float64Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]),
    colors: null,
    uvs: null,
  };
}

function parseAsciiStl(
  bytes: Uint8Array,
  options?: StlImportOptions,
): ImportedMesh {
  const signal = options?.signal;
  let text = textDecoder.decode(bytes);
  if (text.charCodeAt(0) === 0xfeff) {
    text = text.slice(1);
  }

  let solidName = "mesh";
  const coords: number[] = [];
  let len = text.length;
  let pos = 0;
  let sawSolid = false;
  let sawEndSolid = false;
  let iter = 0;

  // Vertex buffer for current facet (must be 3 vertices per facet)
  let facetVertCount = 0;
  let inFacet = false;
  let inLoop = false;

  while (pos < len) {
    if ((iter++ & 0x3fff) === 0) throwIfAborted(signal);
    let nl = text.indexOf("\n", pos);
    if (nl < 0) nl = len;
    const line = text.slice(pos, nl).trim();
    pos = nl < len ? nl + 1 : len;
    if (line === "") continue;

    if (!sawSolid) {
      if (line.startsWith("solid") && (line.length === 5 || line.charCodeAt(5) <= 32)) {
        sawSolid = true;
        const n = line.slice(5).trim();
        if (n.length > 0) solidName = n;
        continue;
      }
      throw new Error("Invalid ASCII STL: missing leading 'solid' header");
    }

    if (line.startsWith("vertex") && line.length > 6 && line.charCodeAt(6) <= 32) {
      if (!inLoop) {
        throw new Error("Invalid ASCII STL: 'vertex' outside 'outer loop'");
      }
      const parts = line.slice(6).trim().split(/\s+/);
      if (parts.length < 3) {
        throw new Error(`Invalid ASCII STL vertex line: "${line}"`);
      }
      const x = Number(parts[0]);
      const y = Number(parts[1]);
      const z = Number(parts[2]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        throw new Error(`Invalid ASCII STL vertex coordinates: "${line}"`);
      }
      coords.push(x, y, z);
      facetVertCount++;
      continue;
    }

    if (line.startsWith("facet") && (line.length === 5 || line.charCodeAt(5) <= 32)) {
      inFacet = true;
      facetVertCount = 0;
      continue;
    }
    if (line.startsWith("outer loop") || line === "outer") {
      inLoop = true;
      continue;
    }
    if (line === "endloop") {
      inLoop = false;
      continue;
    }
    if (line === "endfacet") {
      if (!inFacet || facetVertCount !== 3) {
        throw new Error(
          `Invalid ASCII STL facet: expected 3 vertices, got ${facetVertCount}`,
        );
      }
      inFacet = false;
      inLoop = false;
      continue;
    }
    if (line.startsWith("endsolid")) {
      sawEndSolid = true;
      break;
    }
  }

  if (!sawSolid) {
    throw new Error("Invalid ASCII STL: missing 'solid' header");
  }
  if (inFacet || inLoop || !sawEndSolid) {
    throw new Error("Invalid ASCII STL: file ends early before 'endsolid'");
  }

  const rawVertCount = coords.length / 3;
  const triCount = rawVertCount / 3;
  const dedup = options?.deduplicateVertices === true;
  const meshName = options?.name ?? solidName;

  if (!dedup) {
    const positions = Float64Array.from(coords);
    const indices = new Uint32Array(rawVertCount);
    for (let i = 0; i < rawVertCount; i++) {
      indices[i] = i;
    }
    return {
      name: meshName,
      positions,
      indices,
      placement: new Float64Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ]),
      colors: null,
      uvs: null,
    };
  }

  const welder = new VertexWelder(rawVertCount);
  const indices = new Uint32Array(triCount * 3);
  for (let i = 0; i < rawVertCount; i++) {
    if ((i & 0x3fff) === 0) throwIfAborted(signal);
    indices[i] = welder.intern(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]);
  }
  return {
    name: meshName,
    positions: welder.finish(),
    indices,
    placement: new Float64Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]),
    colors: null,
    uvs: null,
  };
}

/**
 * Parse a binary or ASCII STL payload into an {@link ImportedMesh}.
 */
export function importSTL(
  bytes: Uint8Array | ArrayBuffer,
  options?: StlImportOptions,
): ImportedMesh {
  throwIfAborted(options?.signal);
  const u8 = toUint8(bytes);
  const maxBytes = options?.maxFileBytes ?? DEFAULT_MAX_BYTES;
  if (u8.byteLength > maxBytes) {
    throw new Error(
      `STL file size (${u8.byteLength} bytes) exceeds limit of ${maxBytes} bytes`,
    );
  }

  // Exact binary STL size match wins unconditionally even if the 80-byte header begins with "solid".
  if (u8.byteLength >= 84) {
    const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const triCount = view.getUint32(80, true);
    const expected = 84 + 50 * triCount;
    if (expected === u8.byteLength) {
      return parseBinaryStl(u8, view, triCount, options);
    }
    if (isProbablyAsciiStl(u8)) {
      return parseAsciiStl(u8, options);
    }
    throw new Error(
      `binary STL: the header says ${triCount} triangles (${expected} bytes), but the file has ${u8.byteLength} bytes`,
    );
  }

  if (isProbablyAsciiStl(u8)) {
    return parseAsciiStl(u8, options);
  }
  throw new Error(
    `Invalid STL file: expected ASCII STL or at least 84-byte binary STL header (got ${u8.byteLength} bytes)`,
  );
}
