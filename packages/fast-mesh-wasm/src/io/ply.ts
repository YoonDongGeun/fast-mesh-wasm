/**
 * Pure-TS PLY (`ascii 1.0`, `binary_little_endian 1.0`, `binary_big_endian 1.0`) importer.
 *
 * Supports all standard PLY scalar types, arbitrary custom property skipping,
 * vertex positions, normals (`nx/ny/nz`), UVs (`s/t`, `u/v`, `texture_u/v`),
 * RGBA vertex colors (scaling float `0..1` -> `0..255`), polygon fan
 * triangulation, and strict validation against truncated payloads, NaN/Inf,
 * invalid face vertex counts (< 3), and out-of-range indices.
 */
import type { ImportedMesh } from "./index.js";

/** Options for {@link importPLY}. */
export interface PlyImportOptions {
  /** Mesh name assigned to the imported record. Defaults to `"mesh"`. */
  name?: string;
  /** Maximum accepted file size in bytes. Default: 512 MiB. */
  maxFileBytes?: number;
  /** Optional abort signal. */
  signal?: AbortSignal;
}

type PlyFormat = "ascii" | "binary_little_endian" | "binary_big_endian";

type PlyScalarType =
  | "char"
  | "uchar"
  | "short"
  | "ushort"
  | "int"
  | "uint"
  | "float"
  | "double";

interface PlyScalarProp {
  kind: "scalar";
  name: string;
  type: PlyScalarType;
  bytes: number;
}

interface PlyListProp {
  kind: "list";
  name: string;
  countType: PlyScalarType;
  countBytes: number;
  itemType: PlyScalarType;
  itemBytes: number;
}

type PlyProperty = PlyScalarProp | PlyListProp;

interface PlyElement {
  name: string;
  count: number;
  properties: PlyProperty[];
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

function normalizeScalarType(raw: string): { type: PlyScalarType; bytes: number } {
  switch (raw) {
    case "char":
    case "int8":
      return { type: "char", bytes: 1 };
    case "uchar":
    case "uint8":
      return { type: "uchar", bytes: 1 };
    case "short":
    case "int16":
      return { type: "short", bytes: 2 };
    case "ushort":
    case "uint16":
      return { type: "ushort", bytes: 2 };
    case "int":
    case "int32":
      return { type: "int", bytes: 4 };
    case "uint":
    case "uint32":
      return { type: "uint", bytes: 4 };
    case "float":
    case "float32":
      return { type: "float", bytes: 4 };
    case "double":
    case "float64":
      return { type: "double", bytes: 8 };
    default:
      throw new Error(`Invalid PLY property type "${raw}"`);
  }
}

function readScalar(
  view: DataView,
  offset: number,
  type: PlyScalarType,
  littleEndian: boolean,
): number {
  switch (type) {
    case "char":
      return view.getInt8(offset);
    case "uchar":
      return view.getUint8(offset);
    case "short":
      return view.getInt16(offset, littleEndian);
    case "ushort":
      return view.getUint16(offset, littleEndian);
    case "int":
      return view.getInt32(offset, littleEndian);
    case "uint":
      return view.getUint32(offset, littleEndian);
    case "float":
      return view.getFloat32(offset, littleEndian);
    case "double":
      return view.getFloat64(offset, littleEndian);
  }
}

function isFloatType(type: PlyScalarType): boolean {
  return type === "float" || type === "double";
}

function colorByteFromScalar(value: number, type: PlyScalarType): number {
  if (!Number.isFinite(value)) return 255;
  if (isFloatType(type)) {
    const clamped = value < 0 ? 0 : value > 1 ? 1 : value;
    return Math.round(clamped * 255);
  }
  if (value <= 0) return 0;
  if (value >= 255) return 255;
  return Math.round(value);
}

interface ParsedHeader {
  format: PlyFormat;
  elements: PlyElement[];
  bodyOffset: number;
}

function findHeaderEnd(bytes: Uint8Array): { headerText: string; bodyOffset: number } {
  // Scan line-by-line on raw bytes so binary payloads after `end_header` are untouched.
  const maxHeaderScan = Math.min(bytes.byteLength, 1024 * 1024);
  let lineStart = 0;
  let pos = 0;

  while (pos < maxHeaderScan) {
    const b = bytes[pos];
    if (b === 0x0a /* \n */ || b === 0x0d /* \r */) {
      const lineEnd = pos;
      let nextPos = pos + 1;
      if (b === 0x0d && nextPos < maxHeaderScan && bytes[nextPos] === 0x0a) {
        nextPos++;
      }
      const line = textDecoder.decode(bytes.subarray(lineStart, lineEnd)).trim();
      if (line === "end_header") {
        return {
          headerText: textDecoder.decode(bytes.subarray(0, lineStart)),
          bodyOffset: nextPos,
        };
      }
      lineStart = nextPos;
      pos = nextPos;
      continue;
    }
    pos++;
  }

  // Handle degenerate header-only empty file ending right at "end_header"
  if (bytes.byteLength <= maxHeaderScan) {
    const lastLine = textDecoder.decode(bytes.subarray(lineStart)).trim();
    if (lastLine === "end_header") {
      return {
        headerText: textDecoder.decode(bytes.subarray(0, lineStart)),
        bodyOffset: bytes.byteLength,
      };
    }
  }

  throw new Error("Invalid PLY file: missing end_header marker");
}

function parseHeader(bytes: Uint8Array): ParsedHeader {
  const { headerText, bodyOffset } = findHeaderEnd(bytes);
  const lines = headerText.split(/\r?\n/);

  let idx = 0;
  while (idx < lines.length && lines[idx].trim() === "") idx++;
  if (idx >= lines.length || lines[idx].trim() !== "ply") {
    throw new Error("Invalid PLY file: missing leading 'ply' magic line");
  }
  idx++;

  let format: PlyFormat | null = null;
  const elements: PlyElement[] = [];
  let currentElement: PlyElement | null = null;

  for (; idx < lines.length; idx++) {
    const trimmed = lines[idx].trim();
    if (trimmed === "") continue;
    const tokens = trimmed.split(/\s+/);
    const kw = tokens[0];

    if (kw === "comment" || kw === "obj_info") {
      continue;
    }
    if (kw === "format") {
      if (tokens.length < 3) {
        throw new Error(`Invalid PLY format line: "${trimmed}"`);
      }
      if (
        tokens[1] !== "ascii" &&
        tokens[1] !== "binary_little_endian" &&
        tokens[1] !== "binary_big_endian"
      ) {
        throw new Error(`Unsupported PLY format "${tokens[1]}"`);
      }
      if (tokens[2] !== "1.0") {
        throw new Error(`Unsupported PLY format version "${tokens[2]}"`);
      }
      format = tokens[1];
      continue;
    }
    if (kw === "element") {
      if (tokens.length !== 3) {
        throw new Error(`Invalid PLY element line: "${trimmed}"`);
      }
      const count = Number(tokens[2]);
      if (!Number.isInteger(count) || count < 0) {
        throw new Error(`Invalid PLY element count "${tokens[2]}" for ${tokens[1]}`);
      }
      currentElement = { name: tokens[1], count, properties: [] };
      elements.push(currentElement);
      continue;
    }
    if (kw === "property") {
      if (!currentElement) {
        throw new Error(`Invalid PLY property outside element: "${trimmed}"`);
      }
      if (tokens[1] === "list") {
        if (tokens.length !== 5) {
          throw new Error(`Invalid PLY list property: "${trimmed}"`);
        }
        const ct = normalizeScalarType(tokens[2]);
        const it = normalizeScalarType(tokens[3]);
        currentElement.properties.push({
          kind: "list",
          name: tokens[4],
          countType: ct.type,
          countBytes: ct.bytes,
          itemType: it.type,
          itemBytes: it.bytes,
        });
      } else {
        if (tokens.length !== 3) {
          throw new Error(`Invalid PLY scalar property: "${trimmed}"`);
        }
        const st = normalizeScalarType(tokens[1]);
        currentElement.properties.push({
          kind: "scalar",
          name: tokens[2],
          type: st.type,
          bytes: st.bytes,
        });
      }
      continue;
    }
    throw new Error(`Unknown PLY header directive "${kw}"`);
  }

  if (format === null) {
    throw new Error("Invalid PLY header: missing 'format' line");
  }

  return { format, elements, bodyOffset };
}

interface VertexPropertyLayout {
  xIdx: number;
  yIdx: number;
  zIdx: number;
  nxIdx: number;
  nyIdx: number;
  nzIdx: number;
  uIdx: number;
  vIdx: number;
  rIdx: number;
  gIdx: number;
  bIdx: number;
  aIdx: number;
  hasNormals: boolean;
  hasUvs: boolean;
  hasColors: boolean;
  hasAlpha: boolean;
}

function analyzeVertexProperties(props: PlyProperty[]): VertexPropertyLayout {
  const indexOfScalar = (...names: string[]): number => {
    for (const candidate of names) {
      for (let i = 0; i < props.length; i++) {
        if (props[i].kind === "scalar" && props[i].name === candidate) {
          return i;
        }
      }
    }
    return -1;
  };

  const xIdx = indexOfScalar("x");
  const yIdx = indexOfScalar("y");
  const zIdx = indexOfScalar("z");
  if (xIdx < 0 || yIdx < 0 || zIdx < 0) {
    throw new Error("PLY vertex element must define scalar properties x, y, and z");
  }

  const nxIdx = indexOfScalar("nx");
  const nyIdx = indexOfScalar("ny");
  const nzIdx = indexOfScalar("nz");
  const hasNormals = nxIdx >= 0 && nyIdx >= 0 && nzIdx >= 0;

  let uIdx = indexOfScalar("u");
  let vIdx = indexOfScalar("v");
  if (uIdx < 0 || vIdx < 0) {
    uIdx = indexOfScalar("s");
    vIdx = indexOfScalar("t");
  }
  if (uIdx < 0 || vIdx < 0) {
    uIdx = indexOfScalar("texture_u");
    vIdx = indexOfScalar("texture_v");
  }
  const hasUvs = uIdx >= 0 && vIdx >= 0;

  const rIdx = indexOfScalar("red", "r");
  const gIdx = indexOfScalar("green", "g");
  const bIdx = indexOfScalar("blue", "b");
  const aIdx = indexOfScalar("alpha", "a");
  const hasColors = rIdx >= 0 && gIdx >= 0 && bIdx >= 0;
  const hasAlpha = hasColors && aIdx >= 0;

  return {
    xIdx,
    yIdx,
    zIdx,
    nxIdx,
    nyIdx,
    nzIdx,
    uIdx,
    vIdx,
    rIdx,
    gIdx,
    bIdx,
    aIdx,
    hasNormals,
    hasUvs,
    hasColors,
    hasAlpha,
  };
}

function findFaceVertexIndicesProp(props: PlyProperty[]): number {
  for (let i = 0; i < props.length; i++) {
    const p = props[i];
    if (
      p.kind === "list" &&
      (p.name === "vertex_indices" || p.name === "vertex_index")
    ) {
      return i;
    }
  }
  return -1;
}

function checkVertexFinite(vIdx: number, x: number, y: number, z: number): void {
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    throw new Error(`Invalid PLY vertex ${vIdx}: position contains non-finite coordinate`);
  }
}

function checkNormalFinite(vIdx: number, nx: number, ny: number, nz: number): void {
  if (!Number.isFinite(nx) || !Number.isFinite(ny) || !Number.isFinite(nz)) {
    throw new Error(`Invalid PLY vertex ${vIdx}: normal contains non-finite value`);
  }
}

function checkUvFinite(vIdx: number, u: number, v: number): void {
  if (!Number.isFinite(u) || !Number.isFinite(v)) {
    throw new Error(`Invalid PLY vertex ${vIdx}: UV contains non-finite value`);
  }
}

function parseAsciiBody(
  bodyBytes: Uint8Array,
  elements: PlyElement[],
  signal?: AbortSignal,
): {
  positions: Float64Array;
  normals: Float32Array | null;
  uvs: Float32Array | null;
  colors: Uint8Array | null;
  indices: Uint32Array;
} {
  const body = textDecoder.decode(bodyBytes);
  const len = body.length;
  let pos = 0;

  // Token scanner that tracks line boundaries cleanly and handles CRLF/comments
  const readLineTokens = (elementName: string, rowIdx: number): string[] => {
    while (pos < len) {
      let lineEnd = body.indexOf("\n", pos);
      if (lineEnd < 0) lineEnd = len;
      const line = body.slice(pos, lineEnd).trim();
      pos = lineEnd < len ? lineEnd + 1 : len;
      if (line === "") continue;
      return line.split(/\s+/);
    }
    throw new Error(
      `PLY body ends early while reading ${elementName} row ${rowIdx}`,
    );
  };

  let positions = new Float64Array(0);
  let normals: Float32Array | null = null;
  let uvs: Float32Array | null = null;
  let colors: Uint8Array | null = null;
  const indicesList: number[] = [];
  let vertexCount = 0;
  let hasVertexElement = false;

  for (let eIdx = 0; eIdx < elements.length; eIdx++) {
    const el = elements[eIdx];
    throwIfAborted(signal);

    if (el.name === "vertex") {
      hasVertexElement = true;
      vertexCount = el.count;
      const layout = analyzeVertexProperties(el.properties);
      positions = new Float64Array(vertexCount * 3);
      normals = layout.hasNormals ? new Float32Array(vertexCount * 3) : null;
      uvs = layout.hasUvs ? new Float32Array(vertexCount * 2) : null;
      colors = layout.hasColors ? new Uint8Array(vertexCount * 4) : null;

      // Check fixed scalar indices (ASCII vertex rows could also hold list props to skip)
      const hasListProps = el.properties.some((p) => p.kind === "list");

      for (let v = 0; v < vertexCount; v++) {
        if ((v & 0x3fff) === 0) throwIfAborted(signal);
        const toks = readLineTokens("vertex", v);

        // Map property index -> token offset
        let tokOffsets: number[] | null = null;
        if (hasListProps) {
          tokOffsets = new Array<number>(el.properties.length);
          let cur = 0;
          for (let p = 0; p < el.properties.length; p++) {
            tokOffsets[p] = cur;
            const prop = el.properties[p];
            if (prop.kind === "list") {
              if (cur >= toks.length) {
                throw new Error(`PLY vertex ${v} line ends early`);
              }
              const cnt = Number(toks[cur]);
              if (!Number.isInteger(cnt) || cnt < 0) {
                throw new Error(`Invalid list length in PLY vertex ${v}`);
              }
              cur += 1 + cnt;
            } else {
              cur += 1;
            }
          }
          if (cur > toks.length) {
            throw new Error(`PLY vertex ${v} line ends early`);
          }
        } else if (toks.length < el.properties.length) {
          throw new Error(
            `PLY vertex ${v} has ${toks.length} values, expected at least ${el.properties.length}`,
          );
        }

        const valAt = (propIdx: number): number => {
          const tIdx = tokOffsets ? tokOffsets[propIdx] : propIdx;
          return Number(toks[tIdx]);
        };

        const x = valAt(layout.xIdx);
        const y = valAt(layout.yIdx);
        const z = valAt(layout.zIdx);
        checkVertexFinite(v, x, y, z);
        positions[v * 3] = x;
        positions[v * 3 + 1] = y;
        positions[v * 3 + 2] = z;

        if (normals !== null) {
          const nx = valAt(layout.nxIdx);
          const ny = valAt(layout.nyIdx);
          const nz = valAt(layout.nzIdx);
          checkNormalFinite(v, nx, ny, nz);
          normals[v * 3] = nx;
          normals[v * 3 + 1] = ny;
          normals[v * 3 + 2] = nz;
        }
        if (uvs !== null) {
          const uCoord = valAt(layout.uIdx);
          const vCoord = valAt(layout.vIdx);
          checkUvFinite(v, uCoord, vCoord);
          uvs[v * 2] = uCoord;
          uvs[v * 2 + 1] = vCoord;
        }
        if (colors !== null) {
          const rProp = el.properties[layout.rIdx] as PlyScalarProp;
          const gProp = el.properties[layout.gIdx] as PlyScalarProp;
          const bProp = el.properties[layout.bIdx] as PlyScalarProp;
          colors[v * 4] = colorByteFromScalar(valAt(layout.rIdx), rProp.type);
          colors[v * 4 + 1] = colorByteFromScalar(valAt(layout.gIdx), gProp.type);
          colors[v * 4 + 2] = colorByteFromScalar(valAt(layout.bIdx), bProp.type);
          colors[v * 4 + 3] = layout.hasAlpha
            ? colorByteFromScalar(
                valAt(layout.aIdx),
                (el.properties[layout.aIdx] as PlyScalarProp).type,
              )
            : 255;
        }
      }
      continue;
    }

    if (el.name === "face") {
      const listPropIdx = findFaceVertexIndicesProp(el.properties);
      if (el.count > 0 && listPropIdx < 0) {
        throw new Error(
          "PLY face element is missing 'vertex_indices' or 'vertex_index' list property",
        );
      }

      for (let f = 0; f < el.count; f++) {
        if ((f & 0x3fff) === 0) throwIfAborted(signal);
        const toks = readLineTokens("face", f);
        let cur = 0;
        let faceVerts: number[] | null = null;

        for (let p = 0; p < el.properties.length; p++) {
          const prop = el.properties[p];
          if (prop.kind === "scalar") {
            if (cur >= toks.length) {
              throw new Error(`PLY face ${f} ends early`);
            }
            cur++;
          } else {
            if (cur >= toks.length) {
              throw new Error(`PLY face ${f} ends early`);
            }
            const k = Number(toks[cur++]);
            if (!Number.isInteger(k) || cur + k > toks.length) {
              throw new Error(`PLY face ${f} list length (${k}) is truncated or invalid`);
            }
            if (p === listPropIdx) {
              if (k < 3) {
                throw new Error(
                  `Invalid PLY face ${f}: vertex count ${k} is less than 3`,
                );
              }
              faceVerts = new Array<number>(k);
              for (let j = 0; j < k; j++) {
                const vi = Number(toks[cur + j]);
                if (!Number.isInteger(vi) || vi < 0 || vi >= vertexCount) {
                  throw new Error(
                    `Invalid PLY face ${f} index ${vi} is out of range (0..${vertexCount - 1})`,
                  );
                }
                faceVerts[j] = vi;
              }
            }
            cur += k;
          }
        }

        if (faceVerts !== null) {
          const v0 = faceVerts[0];
          for (let j = 1; j < faceVerts.length - 1; j++) {
            indicesList.push(v0, faceVerts[j], faceVerts[j + 1]);
          }
        }
      }
      continue;
    }

    // Custom unrelated elements: skip exact row count
    for (let r = 0; r < el.count; r++) {
      if ((r & 0x3fff) === 0) throwIfAborted(signal);
      readLineTokens(el.name, r);
    }
  }

  if (!hasVertexElement) {
    throw new Error("PLY file does not declare a 'vertex' element");
  }

  return {
    positions,
    normals,
    uvs,
    colors,
    indices: Uint32Array.from(indicesList),
  };
}

function parseBinaryBody(
  bytes: Uint8Array,
  bodyOffset: number,
  elements: PlyElement[],
  littleEndian: boolean,
  signal?: AbortSignal,
): {
  positions: Float64Array;
  normals: Float32Array | null;
  uvs: Float32Array | null;
  colors: Uint8Array | null;
  indices: Uint32Array;
} {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let at = bodyOffset;
  const totalLen = bytes.byteLength;

  let positions = new Float64Array(0);
  let normals: Float32Array | null = null;
  let uvs: Float32Array | null = null;
  let colors: Uint8Array | null = null;
  const indicesList: number[] = [];
  let vertexCount = 0;
  let hasVertexElement = false;

  const skipElement = (el: PlyElement) => {
    const fixedStride = el.properties.every((p) => p.kind === "scalar")
      ? el.properties.reduce(
          (acc, p) => acc + (p as PlyScalarProp).bytes,
          0,
        )
      : -1;
    if (fixedStride >= 0) {
      const needed = el.count * fixedStride;
      if (at + needed > totalLen) {
        throw new Error(
          `PLY binary payload ends early in element "${el.name}"`,
        );
      }
      at += needed;
      return;
    }
    for (let r = 0; r < el.count; r++) {
      if ((r & 0x3fff) === 0) throwIfAborted(signal);
      for (let p = 0; p < el.properties.length; p++) {
        const prop = el.properties[p];
        if (prop.kind === "scalar") {
          if (at + prop.bytes > totalLen) {
            throw new Error(`PLY binary payload ends early in "${el.name}"`);
          }
          at += prop.bytes;
        } else {
          if (at + prop.countBytes > totalLen) {
            throw new Error(`PLY binary payload ends early in "${el.name}"`);
          }
          const k = readScalar(view, at, prop.countType, littleEndian);
          at += prop.countBytes;
          if (!Number.isInteger(k) || k < 0 || at + k * prop.itemBytes > totalLen) {
            throw new Error(`PLY binary list ends early or invalid length in "${el.name}"`);
          }
          at += k * prop.itemBytes;
        }
      }
    }
  };

  for (let eIdx = 0; eIdx < elements.length; eIdx++) {
    const el = elements[eIdx];
    throwIfAborted(signal);

    if (el.name === "vertex") {
      hasVertexElement = true;
      vertexCount = el.count;
      const layout = analyzeVertexProperties(el.properties);
      positions = new Float64Array(vertexCount * 3);
      normals = layout.hasNormals ? new Float32Array(vertexCount * 3) : null;
      uvs = layout.hasUvs ? new Float32Array(vertexCount * 2) : null;
      colors = layout.hasColors ? new Uint8Array(vertexCount * 4) : null;

      const allScalars = el.properties.every((p) => p.kind === "scalar");
      if (allScalars) {
        const propOffsets = new Int32Array(el.properties.length);
        const propTypes: PlyScalarType[] = [];
        let stride = 0;
        for (let p = 0; p < el.properties.length; p++) {
          const sp = el.properties[p] as PlyScalarProp;
          propOffsets[p] = stride;
          propTypes.push(sp.type);
          stride += sp.bytes;
        }
        if (at + vertexCount * stride > totalLen) {
          throw new Error(
            `PLY binary vertex section ends early (need ${vertexCount * stride} bytes, have ${totalLen - at})`,
          );
        }

        for (let v = 0; v < vertexCount; v++) {
          if ((v & 0x3fff) === 0) throwIfAborted(signal);
          const x = readScalar(view, at + propOffsets[layout.xIdx], propTypes[layout.xIdx], littleEndian);
          const y = readScalar(view, at + propOffsets[layout.yIdx], propTypes[layout.yIdx], littleEndian);
          const z = readScalar(view, at + propOffsets[layout.zIdx], propTypes[layout.zIdx], littleEndian);
          checkVertexFinite(v, x, y, z);
          positions[v * 3] = x;
          positions[v * 3 + 1] = y;
          positions[v * 3 + 2] = z;

          if (normals !== null) {
            const nx = readScalar(view, at + propOffsets[layout.nxIdx], propTypes[layout.nxIdx], littleEndian);
            const ny = readScalar(view, at + propOffsets[layout.nyIdx], propTypes[layout.nyIdx], littleEndian);
            const nz = readScalar(view, at + propOffsets[layout.nzIdx], propTypes[layout.nzIdx], littleEndian);
            checkNormalFinite(v, nx, ny, nz);
            normals[v * 3] = nx;
            normals[v * 3 + 1] = ny;
            normals[v * 3 + 2] = nz;
          }
          if (uvs !== null) {
            const uCoord = readScalar(view, at + propOffsets[layout.uIdx], propTypes[layout.uIdx], littleEndian);
            const vCoord = readScalar(view, at + propOffsets[layout.vIdx], propTypes[layout.vIdx], littleEndian);
            checkUvFinite(v, uCoord, vCoord);
            uvs[v * 2] = uCoord;
            uvs[v * 2 + 1] = vCoord;
          }
          if (colors !== null) {
            const rT = propTypes[layout.rIdx];
            const gT = propTypes[layout.gIdx];
            const bT = propTypes[layout.bIdx];
            colors[v * 4] = colorByteFromScalar(
              readScalar(view, at + propOffsets[layout.rIdx], rT, littleEndian),
              rT,
            );
            colors[v * 4 + 1] = colorByteFromScalar(
              readScalar(view, at + propOffsets[layout.gIdx], gT, littleEndian),
              gT,
            );
            colors[v * 4 + 2] = colorByteFromScalar(
              readScalar(view, at + propOffsets[layout.bIdx], bT, littleEndian),
              bT,
            );
            colors[v * 4 + 3] = layout.hasAlpha
              ? colorByteFromScalar(
                  readScalar(view, at + propOffsets[layout.aIdx], propTypes[layout.aIdx], littleEndian),
                  propTypes[layout.aIdx],
                )
              : 255;
          }
          at += stride;
        }
      } else {
        // Mixed scalar/list properties on vertex
        const rowValues = new Float64Array(el.properties.length);
        for (let v = 0; v < vertexCount; v++) {
          if ((v & 0x3fff) === 0) throwIfAborted(signal);
          for (let p = 0; p < el.properties.length; p++) {
            const prop = el.properties[p];
            if (prop.kind === "scalar") {
              if (at + prop.bytes > totalLen) {
                throw new Error("PLY binary vertex section ends early");
              }
              rowValues[p] = readScalar(view, at, prop.type, littleEndian);
              at += prop.bytes;
            } else {
              if (at + prop.countBytes > totalLen) {
                throw new Error("PLY binary vertex list count ends early");
              }
              const k = readScalar(view, at, prop.countType, littleEndian);
              at += prop.countBytes;
              if (!Number.isInteger(k) || k < 0 || at + k * prop.itemBytes > totalLen) {
                throw new Error("PLY binary vertex list payload ends early");
              }
              at += k * prop.itemBytes;
            }
          }
          const x = rowValues[layout.xIdx];
          const y = rowValues[layout.yIdx];
          const z = rowValues[layout.zIdx];
          checkVertexFinite(v, x, y, z);
          positions[v * 3] = x;
          positions[v * 3 + 1] = y;
          positions[v * 3 + 2] = z;

          if (normals !== null) {
            const nx = rowValues[layout.nxIdx];
            const ny = rowValues[layout.nyIdx];
            const nz = rowValues[layout.nzIdx];
            checkNormalFinite(v, nx, ny, nz);
            normals[v * 3] = nx;
            normals[v * 3 + 1] = ny;
            normals[v * 3 + 2] = nz;
          }
          if (uvs !== null) {
            const uCoord = rowValues[layout.uIdx];
            const vCoord = rowValues[layout.vIdx];
            checkUvFinite(v, uCoord, vCoord);
            uvs[v * 2] = uCoord;
            uvs[v * 2 + 1] = vCoord;
          }
          if (colors !== null) {
            colors[v * 4] = colorByteFromScalar(
              rowValues[layout.rIdx],
              (el.properties[layout.rIdx] as PlyScalarProp).type,
            );
            colors[v * 4 + 1] = colorByteFromScalar(
              rowValues[layout.gIdx],
              (el.properties[layout.gIdx] as PlyScalarProp).type,
            );
            colors[v * 4 + 2] = colorByteFromScalar(
              rowValues[layout.bIdx],
              (el.properties[layout.bIdx] as PlyScalarProp).type,
            );
            colors[v * 4 + 3] = layout.hasAlpha
              ? colorByteFromScalar(
                  rowValues[layout.aIdx],
                  (el.properties[layout.aIdx] as PlyScalarProp).type,
                )
              : 255;
          }
        }
      }
      continue;
    }

    if (el.name === "face") {
      const listPropIdx = findFaceVertexIndicesProp(el.properties);
      if (el.count > 0 && listPropIdx < 0) {
        throw new Error(
          "PLY face element is missing 'vertex_indices' or 'vertex_index' list property",
        );
      }

      for (let f = 0; f < el.count; f++) {
        if ((f & 0x3fff) === 0) throwIfAborted(signal);
        for (let p = 0; p < el.properties.length; p++) {
          const prop = el.properties[p];
          if (prop.kind === "scalar") {
            if (at + prop.bytes > totalLen) {
              throw new Error("PLY binary face section ends early");
            }
            at += prop.bytes;
          } else {
            if (at + prop.countBytes > totalLen) {
              throw new Error("PLY binary face count ends early");
            }
            const k = readScalar(view, at, prop.countType, littleEndian);
            at += prop.countBytes;
            if (!Number.isInteger(k) || k < 0) {
              throw new Error(`Invalid PLY face ${f} list count ${k}`);
            }
            if (at + k * prop.itemBytes > totalLen) {
              throw new Error(
                `PLY binary face ${f} ends early (declared ${k} items, ${totalLen - at} bytes remaining)`,
              );
            }
            if (p === listPropIdx) {
              if (k < 3) {
                throw new Error(
                  `Invalid PLY face ${f}: vertex count ${k} is less than 3`,
                );
              }
              const v0 = readScalar(view, at, prop.itemType, littleEndian);
              if (!Number.isInteger(v0) || v0 < 0 || v0 >= vertexCount) {
                throw new Error(
                  `Invalid PLY face ${f} index ${v0} is out of range (0..${vertexCount - 1})`,
                );
              }
              let prev = readScalar(
                view,
                at + prop.itemBytes,
                prop.itemType,
                littleEndian,
              );
              if (!Number.isInteger(prev) || prev < 0 || prev >= vertexCount) {
                throw new Error(
                  `Invalid PLY face ${f} index ${prev} is out of range (0..${vertexCount - 1})`,
                );
              }
              for (let j = 2; j < k; j++) {
                const curr = readScalar(
                  view,
                  at + j * prop.itemBytes,
                  prop.itemType,
                  littleEndian,
                );
                if (!Number.isInteger(curr) || curr < 0 || curr >= vertexCount) {
                  throw new Error(
                    `Invalid PLY face ${f} index ${curr} is out of range (0..${vertexCount - 1})`,
                  );
                }
                indicesList.push(v0, prev, curr);
                prev = curr;
              }
            }
            at += k * prop.itemBytes;
          }
        }
      }
      continue;
    }

    skipElement(el);
  }

  if (!hasVertexElement) {
    throw new Error("PLY file does not declare a 'vertex' element");
  }

  return {
    positions,
    normals,
    uvs,
    colors,
    indices: Uint32Array.from(indicesList),
  };
}

/**
 * Parse an ASCII or binary (little- or big-endian) PLY file into an {@link ImportedMesh}.
 */
export function importPLY(
  bytes: Uint8Array | ArrayBuffer,
  options?: PlyImportOptions,
): ImportedMesh {
  throwIfAborted(options?.signal);
  const u8 = toUint8(bytes);
  const maxBytes = options?.maxFileBytes ?? DEFAULT_MAX_BYTES;
  if (u8.byteLength > maxBytes) {
    throw new Error(
      `PLY file size (${u8.byteLength} bytes) exceeds limit of ${maxBytes} bytes`,
    );
  }

  const header = parseHeader(u8);
  const parsed =
    header.format === "ascii"
      ? parseAsciiBody(u8.subarray(header.bodyOffset), header.elements, options?.signal)
      : parseBinaryBody(
          u8,
          header.bodyOffset,
          header.elements,
          header.format === "binary_little_endian",
          options?.signal,
        );

  return {
    name: options?.name ?? "mesh",
    positions: parsed.positions,
    indices: parsed.indices,
    placement: new Float64Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]),
    colors: parsed.colors,
    uvs: parsed.uvs,
    normals: parsed.normals,
  };
}
