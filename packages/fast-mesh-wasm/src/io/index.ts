// Mesh exporters: the session's meshes out to STL, OBJ, PLY and glTF.
// Pure TypeScript over a {@link SceneLike}: they read `scene.views(mesh)`
// (positions, indices, normals, colors), so meshes edited, repaired or
// colored in the session export as they are. No WebAssembly, no Worker
// wiring: `store.scene` is a `SceneLike` with synchronous views for both
// local and Worker scenes.
//
// Every exporter returns fresh bytes (`Uint8Array`; ASCII formats UTF-8
// encoded), compacting dead slots away: only the vertices the live
// triangles use are written. A triangle slot reading `0, 0, 0` is dead (a
// genuinely degenerate live `(0, 0, 0)` triangle exports the same way, as
// in the three.js adapter).
//
// The writers are streaming-friendly for 1M+ triangles: binary formats go
// into one preallocated buffer in a single pass, ASCII ones accumulate
// per-triangle chunks joined once.
import type { MeshHandle } from "../handles.js";
import type { SceneLike } from "../scene-like.js";

/** Shared by every exporter: whether local render coordinates are mapped to the world. */
export interface ExportOptions {
  /**
   * Map the mesh through `scene.matrix4(mesh)` (placement and origin) before
   * writing, so the file holds world coordinates. Defaults to `true`; pass
   * `false` for the local render coordinates in `views(mesh).positions()`.
   * Normals follow through the inverse-transpose rotation.
   */
  applyPlacement?: boolean;
}

/** Options of {@link exportSTL}. */
export interface StlExportOptions extends ExportOptions {
  /**
   * Write binary STL (one 50-byte record per triangle). Defaults to `true`;
   * `false` writes ASCII STL (`solid name ... endsolid name`).
   */
  binary?: boolean;
  /** The `solid` name (ASCII) and 80-byte header text (binary). Defaults to `"mesh"`. */
  name?: string;
}

/** Options of {@link exportOBJ}. */
export interface ObjExportOptions extends ExportOptions {
  /** The `o` object name. Defaults to `"mesh"`. */
  name?: string;
}

/** Options of {@link exportPLY}. */
export interface PlyExportOptions extends ExportOptions {
  /**
   * Write binary little-endian PLY (`binary_little_endian 1.0`). Defaults to
   * `true`; `false` writes ASCII (`ascii`) PLY.
   */
  binary?: boolean;
}

/** Options of {@link exportGLTF}. */
export interface GltfExportOptions extends ExportOptions {
  /**
   * Write a binary `.glb` (JSON + BIN chunks). Defaults to `true`; `false`
   * writes a `.gltf` JSON whose buffer is an embedded base64 data URI, so the
   * result stays a single value either way.
   */
  binary?: boolean;
  /** The glTF asset's `generator` string. Defaults to `"fast-mesh-wasm"`. */
  generator?: string;
  /** The mesh and node names. Defaults to `"mesh"`. */
  name?: string;
}

/** Compacted mesh data: only what live triangles use, placement applied. */
interface Snapshot {
  /** 3 floats per vertex. */
  positions: Float32Array;
  /** 3 floats per vertex, or `null` without usable normals. */
  normals: Float32Array | null;
  /** 4 bytes (RGBA) per vertex, or `null` without usable colors. */
  colors: Uint8Array | null;
  /** 3 compacted vertex ids per live triangle. */
  indices: Uint32Array;
  /** Live triangles. */
  triangles: number;
}

/**
 * Reads the mesh's views, copies them (views borrow wasm memory and die on
 * the next scene call), drops dead slots and optionally applies the
 * placement. Throws when the handle is unknown, like `views` does.
 */
function snapshot(scene: SceneLike, mesh: MeshHandle, options?: ExportOptions): Snapshot {
  const views = scene.views(mesh);
  // Copy first: a later `matrix4` call is read-only on `Scene`, but other
  // `SceneLike` hosts are free to move buffers on any call.
  const positions = views.positions().slice();
  const indices = views.indices().slice();
  const rawNormals = views.normals();
  const rawColors = views.colors();
  const vertexSlots = positions.length / 3;
  const triangleSlots = indices.length / 3;
  // `Scene` normals can lag the slots on `"lazy"`/`"manual"` meshes until
  // `update()`: only use a full-length buffer.
  const normals =
    rawNormals !== null && rawNormals !== undefined && rawNormals.length >= positions.length
      ? rawNormals.slice()
      : null;
  const colors =
    rawColors !== null && rawColors !== undefined && rawColors.length >= (4 / 3) * positions.length
      ? rawColors.slice()
      : null;

  // Live triangles, compacting vertices to the ones they use. Dead slots
  // read `0, 0, 0`; the remap drops unreferenced (dead) vertex slots.
  const remap = new Int32Array(vertexSlots).fill(-1);
  const kept = new Uint32Array(indices.length);
  let triangles = 0;
  let vertices = 0;
  for (let t = 0; t < triangleSlots; t++) {
    const a = indices[3 * t];
    const b = indices[3 * t + 1];
    const c = indices[3 * t + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    for (const v of [a, b, c]) {
      if (v >= vertexSlots) {
        throw new Error(`fast-mesh-wasm/io: triangle ${t} indexes missing vertex ${v}`);
      }
      if (remap[v] === -1) {
        remap[v] = vertices++;
      }
    }
    kept[3 * triangles] = remap[a];
    kept[3 * triangles + 1] = remap[b];
    kept[3 * triangles + 2] = remap[c];
    triangles++;
  }
  const outPositions = new Float32Array(3 * vertices);
  const outNormals = normals !== null ? new Float32Array(3 * vertices) : null;
  const outColors = colors !== null ? new Uint8Array(4 * vertices) : null;
  for (let v = 0; v < vertexSlots; v++) {
    const to = remap[v];
    if (to === -1) {
      continue;
    }
    outPositions[3 * to] = positions[3 * v];
    outPositions[3 * to + 1] = positions[3 * v + 1];
    outPositions[3 * to + 2] = positions[3 * v + 2];
    if (outNormals !== null && normals !== null) {
      outNormals[3 * to] = normals[3 * v];
      outNormals[3 * to + 1] = normals[3 * v + 1];
      outNormals[3 * to + 2] = normals[3 * v + 2];
    }
    if (outColors !== null && colors !== null) {
      outColors[4 * to] = colors[4 * v];
      outColors[4 * to + 1] = colors[4 * v + 1];
      outColors[4 * to + 2] = colors[4 * v + 2];
      outColors[4 * to + 3] = colors[4 * v + 3];
    }
  }

  if (options?.applyPlacement !== false) {
    applyMatrix(scene.matrix4(mesh), outPositions, outNormals);
  }
  return {
    positions: outPositions,
    normals: outNormals,
    colors: outColors,
    indices: kept.subarray(0, 3 * triangles),
    triangles,
  };
}

/**
 * Maps positions through the column-major world matrix and normals through
 * its inverse-transpose (renormalized), in place.
 */
function applyMatrix(m: ArrayLike<number>, positions: Float32Array, normals: Float32Array | null): void {
  const m00 = m[0];
  const m01 = m[1];
  const m02 = m[2];
  const m10 = m[4];
  const m11 = m[5];
  const m12 = m[6];
  const m20 = m[8];
  const m21 = m[9];
  const m22 = m[10];
  const tx = m[12];
  const ty = m[13];
  const tz = m[14];
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    positions[i] = m00 * x + m10 * y + m20 * z + tx;
    positions[i + 1] = m01 * x + m11 * y + m21 * z + ty;
    positions[i + 2] = m02 * x + m12 * y + m22 * z + tz;
  }
  if (normals === null) {
    return;
  }
  // Inverse-transpose of the 3x3, so non-uniform placements keep normals
  // perpendicular. Falls back to the untransformed normal when singular.
  const a00 = m11 * m22 - m12 * m21;
  const a01 = m12 * m20 - m10 * m22;
  const a02 = m10 * m21 - m11 * m20;
  const det = m00 * a00 + m01 * a01 + m02 * a02;
  if (!(Math.abs(det) > 1e-30)) {
    return;
  }
  const s = 1 / det;
  // Transposed cofactor matrix (already the inverse-transpose), row-major.
  const n00 = a00 * s;
  const n01 = (m02 * m21 - m01 * m22) * s;
  const n02 = (m01 * m12 - m02 * m11) * s;
  const n10 = a01 * s;
  const n11 = (m00 * m22 - m02 * m20) * s;
  const n12 = (m02 * m10 - m00 * m12) * s;
  const n20 = a02 * s;
  const n21 = (m01 * m20 - m00 * m21) * s;
  const n22 = (m00 * m11 - m01 * m10) * s;
  for (let i = 0; i < normals.length; i += 3) {
    const x = normals[i];
    const y = normals[i + 1];
    const z = normals[i + 2];
    const rx = n00 * x + n10 * y + n20 * z;
    const ry = n01 * x + n11 * y + n21 * z;
    const rz = n02 * x + n12 * y + n22 * z;
    const len = Math.hypot(rx, ry, rz);
    if (len > 1e-30) {
      normals[i] = rx / len;
      normals[i + 1] = ry / len;
      normals[i + 2] = rz / len;
    }
  }
}

/** The geometric normal of triangle `t`, or zeros when degenerate. */
function facetNormal(positions: Float32Array, indices: ArrayLike<number>, t: number, out: Float32Array): void {
  const a = 3 * indices[3 * t];
  const b = 3 * indices[3 * t + 1];
  const c = 3 * indices[3 * t + 2];
  const ux = positions[b] - positions[a];
  const uy = positions[b + 1] - positions[a + 1];
  const uz = positions[b + 2] - positions[a + 2];
  const vx = positions[c] - positions[a];
  const vy = positions[c + 1] - positions[a + 1];
  const vz = positions[c + 2] - positions[a + 2];
  const nx = uy * vz - uz * vy;
  const ny = uz * vx - ux * vz;
  const nz = ux * vy - uy * vx;
  const len = Math.hypot(nx, ny, nz);
  out[0] = len > 0 ? nx / len : 0;
  out[1] = len > 0 ? ny / len : 0;
  out[2] = len > 0 ? nz / len : 0;
}

/**
 * Binary STL: an 80-byte header, the triangle count, then 50 bytes per
 * triangle (normal, three corners as f32, a zero attribute word).
 */
export function exportSTL(scene: SceneLike, mesh: MeshHandle, options?: StlExportOptions): Uint8Array {
  const snap = snapshot(scene, mesh, options);
  if (options?.binary === false) {
    return new TextEncoder().encode(exportSTLAscii(snap, options?.name ?? "mesh"));
  }
  const bytes = new Uint8Array(84 + 50 * snap.triangles);
  const header = `exported by fast-mesh-wasm${options?.name !== undefined ? ` (${options.name})` : ""}`;
  bytes.set(new TextEncoder().encode(header).subarray(0, 80), 0);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  view.setUint32(80, snap.triangles, true);
  const n = new Float32Array(3);
  const useNormals = snap.normals !== null;
  for (let t = 0; t < snap.triangles; t++) {
    if (useNormals) {
      // Per-vertex normals collapse to the facet average, renormalized.
      const nx =
        (snap.normals as Float32Array)[3 * snap.indices[3 * t]] +
        (snap.normals as Float32Array)[3 * snap.indices[3 * t + 1]] +
        (snap.normals as Float32Array)[3 * snap.indices[3 * t + 2]];
      const ny =
        (snap.normals as Float32Array)[3 * snap.indices[3 * t] + 1] +
        (snap.normals as Float32Array)[3 * snap.indices[3 * t + 1] + 1] +
        (snap.normals as Float32Array)[3 * snap.indices[3 * t + 2] + 1];
      const nz =
        (snap.normals as Float32Array)[3 * snap.indices[3 * t] + 2] +
        (snap.normals as Float32Array)[3 * snap.indices[3 * t + 1] + 2] +
        (snap.normals as Float32Array)[3 * snap.indices[3 * t + 2] + 2];
      const len = Math.hypot(nx, ny, nz);
      n[0] = len > 0 ? nx / len : 0;
      n[1] = len > 0 ? ny / len : 0;
      n[2] = len > 0 ? nz / len : 0;
    } else {
      facetNormal(snap.positions, snap.indices, t, n);
    }
    let at = 84 + 50 * t;
    view.setFloat32(at, n[0], true);
    view.setFloat32(at + 4, n[1], true);
    view.setFloat32(at + 8, n[2], true);
    at += 12;
    for (let k = 0; k < 3; k++) {
      const v = 3 * snap.indices[3 * t + k];
      view.setFloat32(at, snap.positions[v], true);
      view.setFloat32(at + 4, snap.positions[v + 1], true);
      view.setFloat32(at + 8, snap.positions[v + 2], true);
      at += 12;
    }
    view.setUint16(at, 0, true);
  }
  return bytes;
}

function exportSTLAscii(snap: Snapshot, name: string): string {
  const parts: string[] = [`solid ${name}\n`];
  const n = new Float32Array(3);
  for (let t = 0; t < snap.triangles; t++) {
    facetNormal(snap.positions, snap.indices, t, n);
    parts.push(`facet normal ${n[0]} ${n[1]} ${n[2]}\n outer loop\n`);
    for (let k = 0; k < 3; k++) {
      const v = 3 * snap.indices[3 * t + k];
      parts.push(`  vertex ${snap.positions[v]} ${snap.positions[v + 1]} ${snap.positions[v + 2]}\n`);
    }
    parts.push(" endloop\nendfacet\n");
  }
  parts.push(`endsolid ${name}\n`);
  return parts.join("");
}

/**
 * Wavefront OBJ: `v` positions (with `r g b` 0..1 suffixes where the mesh
 * is colored, a widely read extension), `vn` normals where present, and
 * `f` faces (`v//vn` with normals, else `v`). 1-based.
 */
export function exportOBJ(scene: SceneLike, mesh: MeshHandle, options?: ObjExportOptions): Uint8Array {
  const snap = snapshot(scene, mesh, options);
  const name = options?.name ?? "mesh";
  const vertices = snap.positions.length / 3;
  const parts: string[] = [`o ${name}\n`];
  for (let v = 0; v < vertices; v++) {
    if (snap.colors !== null) {
      const r = (snap.colors[4 * v] / 255).toPrecision(8);
      const g = (snap.colors[4 * v + 1] / 255).toPrecision(8);
      const b = (snap.colors[4 * v + 2] / 255).toPrecision(8);
      parts.push(`v ${snap.positions[3 * v]} ${snap.positions[3 * v + 1]} ${snap.positions[3 * v + 2]} ${r} ${g} ${b}\n`);
    } else {
      parts.push(`v ${snap.positions[3 * v]} ${snap.positions[3 * v + 1]} ${snap.positions[3 * v + 2]}\n`);
    }
  }
  const useNormals = snap.normals !== null;
  if (useNormals) {
    for (let v = 0; v < vertices; v++) {
      parts.push(
        `vn ${(snap.normals as Float32Array)[3 * v]} ${(snap.normals as Float32Array)[3 * v + 1]} ${(snap.normals as Float32Array)[3 * v + 2]}\n`,
      );
    }
  }
  for (let t = 0; t < snap.triangles; t++) {
    const a = snap.indices[3 * t] + 1;
    const b = snap.indices[3 * t + 1] + 1;
    const c = snap.indices[3 * t + 2] + 1;
    parts.push(useNormals ? `f ${a}//${a} ${b}//${b} ${c}//${c}\n` : `f ${a} ${b} ${c}\n`);
  }
  return new TextEncoder().encode(parts.join(""));
}

/**
 * Stanford PLY: `x y z` floats, `nx ny nz` where normals exist, `red green
 * blue alpha` uchar where colors exist, and `vertex_indices` faces.
 */
export function exportPLY(scene: SceneLike, mesh: MeshHandle, options?: PlyExportOptions): Uint8Array {
  const snap = snapshot(scene, mesh, options);
  const vertices = snap.positions.length / 3;
  const useNormals = snap.normals !== null;
  const useColors = snap.colors !== null;
  const header = [
    "ply",
    `format ${options?.binary === false ? "ascii 1.0" : "binary_little_endian 1.0"}`,
    `element vertex ${vertices}`,
    "property float x",
    "property float y",
    "property float z",
    ...(useNormals ? ["property float nx", "property float ny", "property float nz"] : []),
    ...(useColors ? ["property uchar red", "property uchar green", "property uchar blue", "property uchar alpha"] : []),
    `element face ${snap.triangles}`,
    "property list uchar uint vertex_indices",
    "end_header",
    "",
  ].join("\n");
  if (options?.binary === false) {
    const parts = [header];
    for (let v = 0; v < vertices; v++) {
      let line = `${snap.positions[3 * v]} ${snap.positions[3 * v + 1]} ${snap.positions[3 * v + 2]}`;
      if (useNormals) {
        line += ` ${(snap.normals as Float32Array)[3 * v]} ${(snap.normals as Float32Array)[3 * v + 1]} ${(snap.normals as Float32Array)[3 * v + 2]}`;
      }
      if (useColors) {
        line += ` ${snap.colors?.[4 * v]} ${snap.colors?.[4 * v + 1]} ${snap.colors?.[4 * v + 2]} ${snap.colors?.[4 * v + 3]}`;
      }
      parts.push(`${line}\n`);
    }
    for (let t = 0; t < snap.triangles; t++) {
      parts.push(`3 ${snap.indices[3 * t]} ${snap.indices[3 * t + 1]} ${snap.indices[3 * t + 2]}\n`);
    }
    return new TextEncoder().encode(parts.join(""));
  }
  const head = new TextEncoder().encode(header);
  const floatsPerVertex = 3 + (useNormals ? 3 : 0);
  const body = new Uint8Array(vertices * (4 * floatsPerVertex + (useColors ? 4 : 0)) + snap.triangles * 13);
  const view = new DataView(body.buffer, body.byteOffset, body.byteLength);
  let at = 0;
  for (let v = 0; v < vertices; v++) {
    view.setFloat32(at, snap.positions[3 * v], true);
    view.setFloat32(at + 4, snap.positions[3 * v + 1], true);
    view.setFloat32(at + 8, snap.positions[3 * v + 2], true);
    at += 12;
    if (useNormals) {
      view.setFloat32(at, (snap.normals as Float32Array)[3 * v], true);
      view.setFloat32(at + 4, (snap.normals as Float32Array)[3 * v + 1], true);
      view.setFloat32(at + 8, (snap.normals as Float32Array)[3 * v + 2], true);
      at += 12;
    }
    if (useColors) {
      body[at] = snap.colors?.[4 * v] ?? 0;
      body[at + 1] = snap.colors?.[4 * v + 1] ?? 0;
      body[at + 2] = snap.colors?.[4 * v + 2] ?? 0;
      body[at + 3] = snap.colors?.[4 * v + 3] ?? 0;
      at += 4;
    }
  }
  for (let t = 0; t < snap.triangles; t++) {
    view.setUint8(at, 3);
    view.setUint32(at + 1, snap.indices[3 * t], true);
    view.setUint32(at + 5, snap.indices[3 * t + 1], true);
    view.setUint32(at + 9, snap.indices[3 * t + 2], true);
    at += 13;
  }
  const out = new Uint8Array(head.length + body.length);
  out.set(head, 0);
  out.set(body, head.length);
  return out;
}

/**
 * glTF 2.0: one node, one mesh, one `TRIANGLES` primitive with `POSITION`
 * (plus `NORMAL` where present and `COLOR_0` as normalized RGBA bytes where
 * the mesh is colored). Binary (default) returns a `.glb`; `binary: false`
 * a `.gltf` JSON with the buffer embedded as a base64 data URI.
 *
 * Draco compression is not applied: it rides on the Draco issue.
 */
export function exportGLTF(scene: SceneLike, mesh: MeshHandle, options?: GltfExportOptions): Uint8Array {
  const snap = snapshot(scene, mesh, options);
  const name = options?.name ?? "mesh";
  const vertices = snap.positions.length / 3;
  const useNormals = snap.normals !== null;
  const useColors = snap.colors !== null;

  const positionBytes = new Uint8Array(snap.positions.buffer, snap.positions.byteOffset, snap.positions.byteLength);
  const normalBytes =
    snap.normals !== null
      ? new Uint8Array(snap.normals.buffer, snap.normals.byteOffset, snap.normals.byteLength)
      : null;

  // One BIN buffer: positions, normals, colors (RGBA bytes), indices.
  const chunks: Uint8Array[] = [positionBytes];
  if (normalBytes !== null) {
    chunks.push(normalBytes);
  }
  let colorBytes: Uint8Array | null = null;
  if (snap.colors !== null) {
    colorBytes = new Uint8Array(snap.colors.buffer, snap.colors.byteOffset, snap.colors.byteLength);
    chunks.push(colorBytes);
  }
  const indicesCopy = new Uint32Array(snap.indices);
  const indicesChunk = new Uint8Array(indicesCopy.buffer, indicesCopy.byteOffset, indicesCopy.byteLength);
  chunks.push(indicesChunk);

  let offset = 0;
  const bufferViews: { byteOffset: number; byteLength: number; target: number }[] = [];
  for (const chunk of chunks) {
    bufferViews.push({ byteOffset: offset, byteLength: chunk.length, target: 34962 });
    offset += chunk.length;
  }
  bufferViews[bufferViews.length - 1].target = 34963; // ELEMENT_ARRAY_BUFFER
  const binLength = offset;

  let min: [number, number, number] = [0, 0, 0];
  let max: [number, number, number] = [0, 0, 0];
  if (vertices > 0) {
    min = [snap.positions[0], snap.positions[1], snap.positions[2]];
    max = [min[0], min[1], min[2]];
    for (let v = 1; v < vertices; v++) {
      for (let c = 0; c < 3; c++) {
        const x = snap.positions[3 * v + c];
        if (x < min[c]) {
          min[c] = x;
        }
        if (x > max[c]) {
          max[c] = x;
        }
      }
    }
  }

  let nextView = 1;
  const positionView = 0;
  const normalView = useNormals ? nextView++ : -1;
  const colorView = useColors ? nextView++ : -1;
  // The indices live in the last buffer view.
  const attributes: Record<string, number> = { POSITION: 0 };
  let accessorIndex = 1;
  if (useNormals) {
    attributes["NORMAL"] = accessorIndex++;
  }
  if (useColors) {
    attributes["COLOR_0"] = accessorIndex++;
  }

  const accessors: unknown[] = [
    { bufferView: positionView, componentType: 5126, count: vertices, type: "VEC3", min, max },
  ];
  if (useNormals) {
    accessors.push({ bufferView: normalView, componentType: 5126, count: vertices, type: "VEC3" });
  }
  if (useColors) {
    accessors.push({ bufferView: colorView, componentType: 5121, count: vertices, type: "VEC4", normalized: true });
  }
  accessors.push({ bufferView: bufferViews.length - 1, componentType: 5125, count: snap.indices.length, type: "SCALAR" });

  const json = {
    asset: { version: "2.0", generator: options?.generator ?? "fast-mesh-wasm" },
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name }],
    meshes: [{ name, primitives: [{ attributes, indices: accessors.length - 1, mode: 4 }] }],
    accessors,
    bufferViews: bufferViews.map((b) => ({ buffer: 0, ...b })),
    buffers: [{ byteLength: binLength }],
  };

  if (options?.binary === false) {
    const bin = concat(chunks, binLength);
    (json.buffers[0] as { uri?: string }).uri = `data:application/octet-stream;base64,${toBase64(bin)}`;
    return new TextEncoder().encode(JSON.stringify(json));
  }
  // `.glb`: 12-byte header, a padded JSON chunk, a padded BIN chunk.
  const jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jsonPadded = padLength(jsonBytes.length);
  const binPadded = (binLength + 3) & ~3;
  const total = 12 + 8 + jsonPadded + 8 + binPadded;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
  view.setUint32(0, 0x46546c67, true); // "glTF"
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, jsonPadded, true);
  view.setUint32(16, 0x4e4f534a, true); // "JSON"
  out.set(jsonBytes, 20);
  out.fill(0x20, 20 + jsonBytes.length, 20 + jsonPadded);
  const binAt = 20 + jsonPadded;
  view.setUint32(binAt, binPadded, true);
  view.setUint32(binAt + 4, 0x004e4942, true); // "BIN\0"
  let at = binAt + 8;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.length;
  }
  return out;
}

function concat(chunks: Uint8Array[], total: number): Uint8Array {
  const out = new Uint8Array(total);
  let at = 0;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.length;
  }
  return out;
}

function padLength(length: number): number {
  return (length + 3) & ~3;
}

function toBase64(bytes: Uint8Array): string {
  // Node/Bun have `Buffer`; browsers have `btoa`. Chunked so a 1M-triangle
  // buffer never blows the argument limit.
  const g = globalThis as { Buffer?: { from(b: Uint8Array): { toString(e: string): string } } };
  if (g.Buffer !== undefined) {
    return g.Buffer.from(bytes).toString("base64");
  }
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(text);
}
