/**
 * Pure-TS Wavefront OBJ importer (`v` with optional `r g b [a]`, `vn`, `vt`,
 * `f` k-gon fan triangulation, relative negative indices, `o`/`g` groups).
 */
import type { ImportedMesh } from "./index.js";

/** Options for {@link importOBJ}. */
export interface ObjImportOptions {
  /** Override base mesh name when no `o`/`g` header is present or when merging groups. */
  name?: string;
  /** Merge all `o`/`g` groups into a single {@link ImportedMesh}. Default: `false`. */
  mergeGroups?: boolean;
  /**
   * Ignore normals/UVs/colors and keep vertices in original file order without
   * splitting attribute seams. Default: `false`.
   */
  positionsOnly?: boolean;
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

interface Corner {
  v: number;
  vt: number;
  vn: number;
}

interface RawGroup {
  name: string;
  triCorners: Corner[];
}

function resolveObjIndex(
  rawStr: string,
  countSoFar: number,
  kind: "vertex" | "texcoord" | "normal",
): number {
  const n = Number(rawStr);
  if (!Number.isInteger(n)) {
    throw new Error(`Invalid OBJ ${kind} index "${rawStr}"`);
  }
  if (n === 0) {
    throw new Error(`Invalid OBJ index "0" (OBJ indices are 1-based)`);
  }
  if (kind === "vertex" && countSoFar === 0) {
    throw new Error(`OBJ vertex index "${rawStr}" appears before the first vertex`);
  }
  const resolved = n > 0 ? n - 1 : countSoFar + n;
  if (resolved < 0) {
    throw new Error(
      `OBJ negative ${kind} index "${rawStr}" points before the first ${kind} (available ${countSoFar})`,
    );
  }
  if (resolved >= countSoFar) {
    throw new Error(
      `OBJ ${kind} index "${rawStr}" is out of range (resolved ${resolved}, available ${countSoFar})`,
    );
  }
  return resolved;
}

function clampByte(n: number): number {
  if (n <= 0) return 0;
  if (n >= 255) return 255;
  return Math.round(n);
}

/**
 * Parse a Wavefront OBJ file into one or more {@link ImportedMesh} records.
 */
export function importOBJ(
  bytes: Uint8Array | ArrayBuffer,
  options?: ObjImportOptions,
): ImportedMesh[] {
  throwIfAborted(options?.signal);
  const u8 = toUint8(bytes);
  const maxBytes = options?.maxFileBytes ?? DEFAULT_MAX_BYTES;
  if (u8.byteLength > maxBytes) {
    throw new Error(
      `OBJ file size (${u8.byteLength} bytes) exceeds limit of ${maxBytes} bytes`,
    );
  }

  let text = textDecoder.decode(u8);
  if (text.charCodeAt(0) === 0xfeff) {
    text = text.slice(1);
  }

  const positions: number[] = [];
  const rawColors: number[] = []; // aligned 4 numbers per vertex (NaN when uncolored)
  const vertexHasColor: number[] = [];
  let maxColorVal = 0;
  let anyVertexHasColor = false;

  const uvs: number[] = [];
  const normals: number[] = [];

  const mergeGroups = options?.mergeGroups === true;
  const groups: RawGroup[] = [];
  let currentGroup: RawGroup = {
    name: options?.name ?? "mesh",
    triCorners: [],
  };
  let hasExplicitName = false;

  const flushGroup = () => {
    if (currentGroup.triCorners.length > 0) {
      groups.push(currentGroup);
    }
  };

  // Line scanner handling CRLF, LF, and trailing backslash line continuation
  const len = text.length;
  let pos = 0;
  let iter = 0;

  while (pos < len) {
    if ((iter++ & 0x3fff) === 0) throwIfAborted(options?.signal);

    let line = "";
    while (pos < len) {
      let nl = text.indexOf("\n", pos);
      if (nl < 0) nl = len;
      let chunk = text.slice(pos, nl);
      if (chunk.endsWith("\r")) {
        chunk = chunk.slice(0, -1);
      }
      pos = nl < len ? nl + 1 : len;
      if (chunk.endsWith("\\")) {
        line += chunk.slice(0, -1) + " ";
      } else {
        line += chunk;
        break;
      }
    }

    const hashIdx = line.indexOf("#");
    const stripped = (hashIdx >= 0 ? line.slice(0, hashIdx) : line).trim();
    if (stripped === "") continue;

    const tokens = stripped.split(/\s+/);
    const cmd = tokens[0];

    if (cmd === "v") {
      if (tokens.length < 4) {
        throw new Error(`Invalid OBJ vertex line: "${stripped}"`);
      }
      const x = Number(tokens[1]);
      const y = Number(tokens[2]);
      const z = Number(tokens[3]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        throw new Error(`Invalid OBJ vertex coordinates: "${stripped}"`);
      }
      positions.push(x, y, z);

      // Only 6+ values (x y z r g b [a]) denote vertex colors; 4 values (x y z w) is homogeneous w.
      if (tokens.length >= 7) {
        const r = Number(tokens[4]);
        const g = Number(tokens[5]);
        const b = Number(tokens[6]);
        const a = tokens.length >= 8 ? Number(tokens[7]) : Number.NaN;
        if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b) || (tokens.length >= 8 && !Number.isFinite(a))) {
          throw new Error(`Invalid OBJ vertex color: "${stripped}"`);
        }
        anyVertexHasColor = true;
        vertexHasColor.push(1);
        rawColors.push(r, g, b, tokens.length >= 8 ? a : Number.NaN);
        if (r > maxColorVal) maxColorVal = r;
        if (g > maxColorVal) maxColorVal = g;
        if (b > maxColorVal) maxColorVal = b;
        if (Number.isFinite(a) && a > maxColorVal) maxColorVal = a;
      } else {
        vertexHasColor.push(0);
        rawColors.push(0, 0, 0, 0);
      }
      continue;
    }

    if (cmd === "vt") {
      if (tokens.length < 3) {
        throw new Error(`Invalid OBJ texture coordinate line: "${stripped}"`);
      }
      const u = Number(tokens[1]);
      const v = Number(tokens[2]);
      if (!Number.isFinite(u) || !Number.isFinite(v)) {
        throw new Error(`Invalid OBJ texture coordinate: "${stripped}"`);
      }
      uvs.push(u, v);
      continue;
    }

    if (cmd === "vn") {
      if (tokens.length < 4) {
        throw new Error(`Invalid OBJ normal line: "${stripped}"`);
      }
      const nx = Number(tokens[1]);
      const ny = Number(tokens[2]);
      const nz = Number(tokens[3]);
      if (!Number.isFinite(nx) || !Number.isFinite(ny) || !Number.isFinite(nz)) {
        throw new Error(`Invalid OBJ normal values: "${stripped}"`);
      }
      normals.push(nx, ny, nz);
      continue;
    }

    if (cmd === "f") {
      const faceVertsCount = tokens.length - 1;
      if (faceVertsCount < 3) {
        throw new Error(`Invalid OBJ face with fewer than 3 vertices: "${stripped}"`);
      }
      const vCount = positions.length / 3;
      const vtCount = uvs.length / 2;
      const vnCount = normals.length / 3;

      const corners = new Array<Corner>(faceVertsCount);
      for (let i = 0; i < faceVertsCount; i++) {
        const tok = tokens[i + 1];
        const slash1 = tok.indexOf("/");
        if (slash1 < 0) {
          corners[i] = {
            v: resolveObjIndex(tok, vCount, "vertex"),
            vt: -1,
            vn: -1,
          };
        } else {
          const vStr = tok.slice(0, slash1);
          const vIdx = resolveObjIndex(vStr, vCount, "vertex");
          const slash2 = tok.indexOf("/", slash1 + 1);
          if (slash2 < 0) {
            const vtStr = tok.slice(slash1 + 1);
            const vtIdx =
              vtStr.length > 0 ? resolveObjIndex(vtStr, vtCount, "texcoord") : -1;
            corners[i] = { v: vIdx, vt: vtIdx, vn: -1 };
          } else {
            const vtStr = tok.slice(slash1 + 1, slash2);
            const vnStr = tok.slice(slash2 + 1);
            if (tok.indexOf("/", slash2 + 1) >= 0) {
              throw new Error(`Invalid OBJ face token "${tok}"`);
            }
            const vtIdx =
              vtStr.length > 0 ? resolveObjIndex(vtStr, vtCount, "texcoord") : -1;
            const vnIdx =
              vnStr.length > 0 ? resolveObjIndex(vnStr, vnCount, "normal") : -1;
            corners[i] = { v: vIdx, vt: vtIdx, vn: vnIdx };
          }
        }
      }

      const c0 = corners[0];
      for (let i = 1; i < faceVertsCount - 1; i++) {
        currentGroup.triCorners.push(c0, corners[i], corners[i + 1]);
      }
      continue;
    }

    if (!mergeGroups && (cmd === "o" || cmd === "g")) {
      const nextName = stripped.slice(1).trim();
      if (currentGroup.triCorners.length > 0) {
        flushGroup();
        currentGroup = {
          name: nextName || options?.name || `mesh-${groups.length + 1}`,
          triCorners: [],
        };
        hasExplicitName = nextName.length > 0;
      } else if (nextName.length > 0 && (!hasExplicitName || cmd === "o")) {
        currentGroup.name = nextName;
        hasExplicitName = true;
      }
    }
  }

  flushGroup();
  if (groups.length === 0) {
    throw new Error("OBJ file contains no faces");
  }

  // Convert parsed vertex colors consistently across the whole file:
  // if any channel exceeds 1.0, treat the file as 0..255 bytes; else scale 0..1 -> 0..255.
  let vertColorBytes: Uint8Array | null = null;
  if (anyVertexHasColor) {
    const scale = maxColorVal > 1 ? 1 : 255;
    const totalV = positions.length / 3;
    vertColorBytes = new Uint8Array(totalV * 4);
    for (let v = 0; v < totalV; v++) {
      if (vertexHasColor[v] !== 0) {
        const r = rawColors[v * 4];
        const g = rawColors[v * 4 + 1];
        const b = rawColors[v * 4 + 2];
        const aRaw = rawColors[v * 4 + 3];
        vertColorBytes[v * 4] = clampByte(r * scale);
        vertColorBytes[v * 4 + 1] = clampByte(g * scale);
        vertColorBytes[v * 4 + 2] = clampByte(b * scale);
        vertColorBytes[v * 4 + 3] = Number.isFinite(aRaw) ? clampByte(aRaw * scale) : 255;
      } else {
        vertColorBytes[v * 4] = 255;
        vertColorBytes[v * 4 + 1] = 255;
        vertColorBytes[v * 4 + 2] = 255;
        vertColorBytes[v * 4 + 3] = 255;
      }
    }
  }

  const results: ImportedMesh[] = [];
  if (options?.positionsOnly === true) {
    const outPos = Float64Array.from(positions);
    const identity = new Float64Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
    for (let gIdx = 0; gIdx < groups.length; gIdx++) {
      throwIfAborted(options?.signal);
      const group = groups[gIdx];
      const cornerCount = group.triCorners.length;
      const outIndices = new Uint32Array(cornerCount);
      for (let i = 0; i < cornerCount; i++) {
        outIndices[i] = group.triCorners[i].v;
      }
      results.push({
        name: group.name,
        positions: gIdx === 0 ? outPos : outPos.slice(),
        indices: outIndices,
        placement: identity.slice(),
        colors: null,
        uvs: null,
      });
    }
    return results;
  }

  for (let gIdx = 0; gIdx < groups.length; gIdx++) {
    throwIfAborted(options?.signal);
    const group = groups[gIdx];
    const cornerCount = group.triCorners.length;
    const outIndices = new Uint32Array(cornerCount);

    let hasGroupUvs = false;
    let hasGroupNormals = false;
    let hasGroupColors = false;
    for (let i = 0; i < cornerCount; i++) {
      const c = group.triCorners[i];
      if (c.vt >= 0) hasGroupUvs = true;
      if (c.vn >= 0) hasGroupNormals = true;
      if (vertColorBytes !== null && vertexHasColor[c.v] !== 0) {
        hasGroupColors = true;
      }
    }

    // Per-vertex bucket list deduplicating unique `(v, vt, vn)` tuples without allocating map keys
    interface InternedCorner {
      vt: number;
      vn: number;
      outIdx: number;
    }
    const buckets = new Map<number, InternedCorner[]>();
    const outVertCorners: Corner[] = [];

    for (let i = 0; i < cornerCount; i++) {
      const c = group.triCorners[i];
      const normVt = hasGroupUvs ? c.vt : -1;
      const normVn = hasGroupNormals ? c.vn : -1;
      let list = buckets.get(c.v);
      let found = -1;
      if (list) {
        for (let k = 0; k < list.length; k++) {
          if (list[k].vt === normVt && list[k].vn === normVn) {
            found = list[k].outIdx;
            break;
          }
        }
      } else {
        list = [];
        buckets.set(c.v, list);
      }
      if (found < 0) {
        found = outVertCorners.length;
        outVertCorners.push({ v: c.v, vt: normVt, vn: normVn });
        list.push({ vt: normVt, vn: normVn, outIdx: found });
      }
      outIndices[i] = found;
    }

    const vertCount = outVertCorners.length;
    const outPos = new Float64Array(vertCount * 3);
    const outUvs = hasGroupUvs ? new Float32Array(vertCount * 2) : null;
    const outNormals = hasGroupNormals ? new Float32Array(vertCount * 3) : null;
    const outColors =
      hasGroupColors && vertColorBytes !== null ? new Uint8Array(vertCount * 4) : null;

    for (let i = 0; i < vertCount; i++) {
      const c = outVertCorners[i];
      outPos[i * 3] = positions[c.v * 3];
      outPos[i * 3 + 1] = positions[c.v * 3 + 1];
      outPos[i * 3 + 2] = positions[c.v * 3 + 2];

      if (outUvs !== null) {
        if (c.vt >= 0) {
          outUvs[i * 2] = uvs[c.vt * 2];
          outUvs[i * 2 + 1] = uvs[c.vt * 2 + 1];
        }
      }
      if (outNormals !== null) {
        if (c.vn >= 0) {
          outNormals[i * 3] = normals[c.vn * 3];
          outNormals[i * 3 + 1] = normals[c.vn * 3 + 1];
          outNormals[i * 3 + 2] = normals[c.vn * 3 + 2];
        }
      }
      if (outColors !== null && vertColorBytes !== null) {
        outColors[i * 4] = vertColorBytes[c.v * 4];
        outColors[i * 4 + 1] = vertColorBytes[c.v * 4 + 1];
        outColors[i * 4 + 2] = vertColorBytes[c.v * 4 + 2];
        outColors[i * 4 + 3] = vertColorBytes[c.v * 4 + 3];
      }
    }

    results.push({
      name: group.name,
      positions: outPos,
      indices: outIndices,
      placement: new Float64Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ]),
      colors: outColors,
      uvs: outUvs,
      ...(outNormals !== null ? { normals: outNormals } : {}),
    });
  }

  return results;
}
