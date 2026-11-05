// Mesh exporters and the glTF importer: the session's meshes out to STL, OBJ,
// PLY and glTF, and glTF 2.0 assets (binary `.glb` or JSON `.gltf` with
// embedded buffers) back in as one mesh per (node, primitive).
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
import type { Attribute, Decoder, DecoderModule, EncoderModule, Mesh, MeshBuilder } from "draco3d";

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
  /**
   * Compress the primitive with Draco (`KHR_draco_mesh_compression`) instead
   * of raw buffer views. Defaults to `false`; with `true` the call is async
   * (a `Promise<Uint8Array>`), encoding through the draco3d encoder (WASM),
   * which lazy-loads like the decoder. Reads back through
   * {@link importGLTFAsync}.
   */
  draco?: boolean;
  /**
   * The Draco encoder settings (quantization, compression level, face-order
   * preservation, encoder module). Only read when `draco` is `true`; the
   * attributes themselves always come from the mesh.
   */
  dracoOptions?: DracoCompressionOptions;
}

/** Compacted mesh data: only what live triangles use, placement applied. */
interface Snapshot {
  /** 3 floats per vertex. */
  positions: Float32Array;
  /** 3 floats per vertex, or `null` without usable normals. */
  normals: Float32Array | null;
  /** 4 bytes (RGBA) per vertex, or `null` without usable colors. */
  colors: Uint8Array | null;
  /** 2 floats per vertex, or `null` without UVs. */
  uvs: Float32Array | null;
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
  const rawUvs = views.uvs?.() ?? null;
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
  const uvs = rawUvs !== null && rawUvs.length >= (2 / 3) * positions.length ? rawUvs.slice() : null;

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
  const outUvs = uvs !== null ? new Float32Array(2 * vertices) : null;
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
    if (outUvs !== null && uvs !== null) {
      outUvs[2 * to] = uvs[2 * v];
      outUvs[2 * to + 1] = uvs[2 * v + 1];
    }
  }

  if (options?.applyPlacement !== false) {
    applyMatrix(scene.matrix4(mesh), outPositions, outNormals);
  }
  return {
    positions: outPositions,
    normals: outNormals,
    colors: outColors,
    uvs: outUvs,
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
 * (plus `TEXCOORD_0` as float pairs where the mesh has UVs, `NORMAL` where present and `COLOR_0` as normalized RGBA bytes where
 * the mesh is colored). Binary (default) returns a `.glb`; `binary: false`
 * a `.gltf` JSON with the buffer embedded as a base64 data URI.
 *
 * Pass `draco: true` for `KHR_draco_mesh_compression` instead: the call
 * becomes async (a `Promise<Uint8Array>`), encoding through the draco3d
 * encoder (WASM), which lazy-loads like the decoder.
 */
export function exportGLTF(
  scene: SceneLike,
  mesh: MeshHandle,
  options?: GltfExportOptions & { draco?: false },
): Uint8Array;
/**
 * glTF 2.0 with `draco: true`: the same node, mesh and `TRIANGLES`
 * primitive, but the positions (plus normals, colors and UVs where present) ride
 * a Draco bitstream in one buffer view instead of raw buffer views. The
 * primitive's accessors describe the decoded shape (POSITION keeps its
 * min/max); `extensionsUsed`/`extensionsRequired` name the extension.
 *
 * The draco3d encoder (WASM) initializes on first use; its module is shared
 * across calls. Needs the `draco3d` package installed unless `dracoOptions`
 * supplies the module or its `.wasm`. Reads back through
 * {@link importGLTFAsync} (compressed attributes beyond POSITION stay
 * unmapped there); the sync {@link importGLTF} still throws on it.
 */
export function exportGLTF(
  scene: SceneLike,
  mesh: MeshHandle,
  options: GltfExportOptions & { draco: true },
): Promise<Uint8Array>;
export function exportGLTF(
  scene: SceneLike,
  mesh: MeshHandle,
  options?: GltfExportOptions,
): Uint8Array | Promise<Uint8Array>;
export function exportGLTF(
  scene: SceneLike,
  mesh: MeshHandle,
  options?: GltfExportOptions,
): Uint8Array | Promise<Uint8Array> {
  if (options?.draco === true) {
    return exportGLTFDraco(scene, mesh, options);
  }
  const snap = snapshot(scene, mesh, options);
  const name = options?.name ?? "mesh";
  const vertices = snap.positions.length / 3;
  const useNormals = snap.normals !== null;
  const useColors = snap.colors !== null;
  const useUvs = snap.uvs !== null;

  const positionBytes = new Uint8Array(snap.positions.buffer, snap.positions.byteOffset, snap.positions.byteLength);
  const normalBytes =
    snap.normals !== null
      ? new Uint8Array(snap.normals.buffer, snap.normals.byteOffset, snap.normals.byteLength)
      : null;

  // One BIN buffer: positions, normals, colors (RGBA bytes), UVs, indices.
  const chunks: Uint8Array[] = [positionBytes];
  if (normalBytes !== null) {
    chunks.push(normalBytes);
  }
  let colorBytes: Uint8Array | null = null;
  if (snap.colors !== null) {
    colorBytes = new Uint8Array(snap.colors.buffer, snap.colors.byteOffset, snap.colors.byteLength);
    chunks.push(colorBytes);
  }
  if (snap.uvs !== null) {
    chunks.push(new Uint8Array(snap.uvs.buffer, snap.uvs.byteOffset, snap.uvs.byteLength));
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

  const { min, max } = positionBounds(snap.positions);

  let nextView = 1;
  const positionView = 0;
  const normalView = useNormals ? nextView++ : -1;
  const colorView = useColors ? nextView++ : -1;
  const uvView = useUvs ? nextView++ : -1;
  // The indices live in the last buffer view.
  const attributes: Record<string, number> = { POSITION: 0 };
  let accessorIndex = 1;
  if (useNormals) {
    attributes["NORMAL"] = accessorIndex++;
  }
  if (useColors) {
    attributes["COLOR_0"] = accessorIndex++;
  }
  if (useUvs) {
    attributes["TEXCOORD_0"] = accessorIndex++;
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
  if (useUvs) {
    accessors.push({ bufferView: uvView, componentType: 5126, count: vertices, type: "VEC2" });
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

  return packGLTF(json, chunks, options?.binary !== false);
}

/** The POSITION min/max over compacted vertices (zeros without vertices). */
function positionBounds(positions: Float32Array): { min: [number, number, number]; max: [number, number, number] } {
  const min: [number, number, number] = [0, 0, 0];
  const max: [number, number, number] = [0, 0, 0];
  const vertices = positions.length / 3;
  if (vertices > 0) {
    min[0] = positions[0];
    min[1] = positions[1];
    min[2] = positions[2];
    max[0] = min[0];
    max[1] = min[1];
    max[2] = min[2];
    for (let v = 1; v < vertices; v++) {
      for (let c = 0; c < 3; c++) {
        const x = positions[3 * v + c];
        if (x < min[c]) {
          min[c] = x;
        }
        if (x > max[c]) {
          max[c] = x;
        }
      }
    }
  }
  return { min, max };
}

/**
 * Packs the asset JSON plus its buffer chunks into one value: a binary
 * `.glb`, or `.gltf` JSON with the buffer embedded as a base64 data URI.
 */
function packGLTF(
  json: { buffers: { byteLength: number; uri?: string }[] },
  chunks: Uint8Array[],
  binary: boolean,
): Uint8Array {
  const binLength = json.buffers[0].byteLength;
  if (!binary) {
    const bin = concat(chunks, binLength);
    json.buffers[0].uri = `data:application/octet-stream;base64,${toBase64(bin)}`;
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

/** Per-attribute Draco quantization bits: fewer bits compress smaller, at coarser values. */
export interface DracoQuantizationBits {
  /** POSITION bits. Defaults to `14`. */
  position?: number;
  /** NORMAL bits. Defaults to `10`. */
  normal?: number;
  /** COLOR bits. Defaults to `8` (lossless for byte colors). */
  color?: number;
  /** TEX_COORD bits, over the UVs' range. Defaults to `16`. */
  uv?: number;
  /**
   * TEX_COORD bits. Defaults to `12`. Validated but currently unused: UVs
   * ride issue #137 (there is no UV channel to encode yet).
   */
  texCoord?: number;
}

/** The Draco encoder settings shared by {@link encodeDraco} and the `draco: true` {@link exportGLTF} path. */
export interface DracoCompressionOptions {
  /**
   * A ready draco3d encoder module, as `createEncoderModule` resolves. When
   * omitted, one is created from the `draco3d` package and shared across
   * calls. Pass one in to reuse a module (or to supply the `.wasm` in an
   * environment the default lookup cannot reach, like a bundled Worker).
   */
  encoderModule?: unknown;
  /**
   * The encoder `.wasm` bytes, forwarded to `createEncoderModule` when the
   * shared module is created. Bundlers cannot always locate draco3d's own
   * `.wasm` next to its glue (Vite workers, inlined builds): fetch the
   * encoder asset (a `?url` import of `draco3d/draco_encoder.wasm`) and pass
   * its bytes here. Ignored when `encoderModule` is given.
   */
  wasmBinary?: Uint8Array | ArrayBuffer;
  /** Per-attribute quantization bits (each an integer 1..30). */
  quantizationBits?: DracoQuantizationBits;
  /**
   * How hard the encoder squeezes: an integer 0..10, defaulting to `5`.
   * Higher levels encode slower into fewer bytes (the encode/decode speeds
   * run at `10 - level`); `0` is the fastest encode.
   */
  compressionLevel?: number;
  /**
   * Keep the decoded faces (and vertices) in input order via sequential
   * encoding. Defaults to `false`: the edgebreaker reorders faces while
   * compressing better. Either way the same triangles decode.
   */
  preserveFaceOrder?: boolean;
}

/** Options of {@link encodeDraco}: the encoder settings plus the extra attributes. */
export interface DracoEncodeOptions extends DracoCompressionOptions {
  /**
   * Per-vertex normals: 3 numbers per vertex, encoded as the NORMAL
   * attribute. Must match `positions` in length.
   */
  normals?: ArrayLike<number>;
  /**
   * Per-vertex colors: 4 bytes (RGBA) per vertex, encoded as a normalized
   * COLOR attribute. Each value must be an integer 0..255.
   */
  colors?: ArrayLike<number>;
  /** Texture coordinates: 2 finite numbers per vertex. */
  uvs?: ArrayLike<number>;
}

/**
 * Encodes a triangle mesh into standalone Draco bytes (a `.drc` file):
 * positions always, plus the `normals`/`colors`/`uvs` options when given.
 * Positions encode as 32-bit floats (wider inputs narrow).
 *
 * The draco3d encoder (WASM) initializes on first use; its module is shared
 * across calls. Needs the `draco3d` package installed unless `options`
 * supplies the module or its `.wasm`.
 *
 * Decodes through {@link decodeDraco} (positions, faces and UVs: normals and
 * colors need draco3d's own decoder).
 */
export async function encodeDraco(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  options?: DracoEncodeOptions,
): Promise<Uint8Array> {
  const checkedPositions = checkedEncodePositions(positions);
  const checkedIndices = checkedEncodeIndices(indices, checkedPositions.length / 3);
  const checkedNormals = checkedEncodeNormals(options?.normals, checkedPositions.length);
  const checkedColors = checkedEncodeColors(options?.colors, checkedPositions.length / 3);
  const settings = encodeSettings(options);
  const module = await encoderOf(options);
  const checkedUvs = checkedEncodeUvs(options?.uvs, checkedPositions.length / 3);
  return encodeMeshDraco(module, checkedPositions, checkedIndices, checkedNormals, checkedColors, checkedUvs, settings)
    .bytes;
}

/**
 * glTF 2.0 with `draco: true`: one node, one mesh, one `TRIANGLES`
 * primitive whose attributes ride a Draco bitstream in a single buffer view.
 * The primitive's accessors describe the decoded shape (POSITION keeps its
 * min/max); the `KHR_draco_mesh_compression` extension maps them to the
 * bitstream's attribute ids.
 */
async function exportGLTFDraco(
  scene: SceneLike,
  mesh: MeshHandle,
  options?: GltfExportOptions,
): Promise<Uint8Array> {
  const snap = snapshot(scene, mesh, options);
  const name = options?.name ?? "mesh";
  const vertices = snap.positions.length / 3;
  const settings = encodeSettings(options?.dracoOptions);
  // The snapshot is already compacted; the checks retype it (f64 views
  // narrow to f32) and reject the unencodable (Draco needs triangles).
  const positions = checkedEncodePositions(snap.positions);
  const indices = checkedEncodeIndices(snap.indices, vertices);
  const normals = checkedEncodeNormals(snap.normals, positions.length);
  const colors = checkedEncodeColors(snap.colors, vertices);
  const uvs = checkedEncodeUvs(snap.uvs, vertices);
  const module = await encoderOf(options?.dracoOptions);
  const { bytes, positionId, normalId, colorId, uvId } = encodeMeshDraco(
    module,
    positions,
    indices,
    normals,
    colors,
    uvs,
    settings,
  );

  const { min, max } = positionBounds(snap.positions);
  const attributes: Record<string, number> = { POSITION: 0 };
  const accessors: unknown[] = [{ componentType: 5126, count: vertices, type: "VEC3", min, max }];
  const dracoAttributes: Record<string, number> = { POSITION: positionId };
  let accessorIndex = 1;
  if (normals !== null) {
    attributes["NORMAL"] = accessorIndex++;
    accessors.push({ componentType: 5126, count: vertices, type: "VEC3" });
    dracoAttributes["NORMAL"] = normalId as number;
  }
  if (colors !== null) {
    attributes["COLOR_0"] = accessorIndex++;
    accessors.push({ componentType: 5121, count: vertices, type: "VEC4", normalized: true });
    dracoAttributes["COLOR_0"] = colorId as number;
  }
  if (uvs !== null) {
    attributes["TEXCOORD_0"] = accessorIndex++;
    accessors.push({ componentType: 5126, count: vertices, type: "VEC2" });
    dracoAttributes["TEXCOORD_0"] = uvId as number;
  }
  accessors.push({ componentType: 5125, count: indices.length, type: "SCALAR" });

  const json = {
    asset: { version: "2.0", generator: options?.generator ?? "fast-mesh-wasm" },
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name }],
    meshes: [
      {
        name,
        primitives: [
          {
            attributes,
            indices: accessors.length - 1,
            mode: 4,
            extensions: { KHR_draco_mesh_compression: { bufferView: 0, attributes: dracoAttributes } },
          },
        ],
      },
    ],
    accessors,
    // The bitstream's buffer view carries no target: it is not a GPU buffer.
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: bytes.length }],
    buffers: [{ byteLength: bytes.length }],
    extensionsUsed: ["KHR_draco_mesh_compression"],
    extensionsRequired: ["KHR_draco_mesh_compression"],
  };
  return packGLTF(json, [bytes], options?.binary !== false);
}

/** The encoder knobs, validated with defaults filled in. */
interface ResolvedEncodeSettings {
  positionBits: number;
  normalBits: number;
  colorBits: number;
  uvBits: number;
  level: number;
  preserveFaceOrder: boolean;
}

/** Validates the quantization/level/order options before the module loads. */
function encodeSettings(options?: DracoCompressionOptions): ResolvedEncodeSettings {
  const bits = options?.quantizationBits;
  // Validated for #137; nothing encodes UVs yet.
  checkedQuantizationBits(bits?.texCoord, "texCoord", 12);
  const level = options?.compressionLevel ?? 5;
  if (!Number.isInteger(level) || level < 0 || level > 10) {
    throw new Error(`Draco: the compression level must be an integer from 0 to 10 (got ${options?.compressionLevel})`);
  }
  return {
    positionBits: checkedQuantizationBits(bits?.position, "position", 14),
    normalBits: checkedQuantizationBits(bits?.normal, "normal", 10),
    colorBits: checkedQuantizationBits(bits?.color, "color", 8),
    uvBits: checkedQuantizationBits(bits?.uv, "uv", 16),
    level,
    preserveFaceOrder: options?.preserveFaceOrder === true,
  };
}

/**
 * One attribute's quantization bits: the default without an override, else
 * an integer 1..30. Checked here because the encoder accepts any int.
 */
function checkedQuantizationBits(value: number | undefined, name: string, fallback: number): number {
  if (value === undefined) {
    return fallback;
  }
  if (!Number.isInteger(value) || value < 1 || value > 30) {
    throw new Error(`Draco: ${name} quantization bits must be an integer from 1 to 30 (got ${value})`);
  }
  return value;
}

/** Copies positions to f32, checking length (3 per vertex) and finiteness. */
function checkedEncodePositions(positions: ArrayLike<number>): Float32Array {
  if (positions.length === 0) {
    throw new Error("Draco: cannot encode a mesh with no vertices");
  }
  if (positions.length % 3 !== 0) {
    throw new Error(`Draco: positions need 3 numbers per vertex (got length ${positions.length})`);
  }
  const out = Float32Array.from(positions);
  for (let i = 0; i < out.length; i++) {
    if (!Number.isFinite(out[i])) {
      throw new Error(`Draco: position ${Math.floor(i / 3)} is not finite`);
    }
  }
  return out;
}

/** Copies indices to u32, checking length (3 per triangle) and range. */
function checkedEncodeIndices(indices: ArrayLike<number>, vertices: number): Uint32Array {
  if (indices.length === 0) {
    throw new Error("Draco: cannot encode a mesh with no triangles");
  }
  if (indices.length % 3 !== 0) {
    throw new Error(`Draco: indices need 3 corners per triangle (got length ${indices.length})`);
  }
  const out = new Uint32Array(indices.length);
  for (let t = 0; t < indices.length / 3; t++) {
    for (let k = 0; k < 3; k++) {
      const c = indices[3 * t + k];
      if (!Number.isInteger(c) || c < 0 || c >= vertices) {
        throw new Error(`Draco: triangle ${t} indexes missing vertex ${c}`);
      }
      out[3 * t + k] = c;
    }
  }
  return out;
}

/** Copies normals to f32 (`null` without), checking length and finiteness. */
function checkedEncodeNormals(
  normals: ArrayLike<number> | null | undefined,
  positionsLength: number,
): Float32Array | null {
  if (normals === null || normals === undefined) {
    return null;
  }
  if (normals.length !== positionsLength) {
    throw new Error(
      `Draco: normals need 3 numbers per vertex (got length ${normals.length} for ${positionsLength / 3} vertices)`,
    );
  }
  const out = Float32Array.from(normals);
  for (let i = 0; i < out.length; i++) {
    if (!Number.isFinite(out[i])) {
      throw new Error(`Draco: normal ${Math.floor(i / 3)} is not finite`);
    }
  }
  return out;
}

/** Copies colors to RGBA bytes (`null` without), checking length and byte range. */
function checkedEncodeColors(
  colors: ArrayLike<number> | null | undefined,
  vertices: number,
): Uint8Array | null {
  if (colors === null || colors === undefined) {
    return null;
  }
  if (colors.length !== 4 * vertices) {
    throw new Error(`Draco: colors need 4 bytes per vertex (got length ${colors.length} for ${vertices} vertices)`);
  }
  const out = new Uint8Array(4 * vertices);
  for (let i = 0; i < out.length; i++) {
    const c = colors[i];
    if (!Number.isInteger(c) || c < 0 || c > 255) {
      throw new Error(`Draco: colors[${i}] is not a byte 0..255 (got ${c})`);
    }
    out[i] = c;
  }
  return out;
}

/** Copies UVs to floats (`null` without), checking length and finiteness. */
function checkedEncodeUvs(uvs: ArrayLike<number> | null | undefined, vertices: number): Float32Array | null {
  if (uvs === null || uvs === undefined) {
    return null;
  }
  if (uvs.length !== 2 * vertices) {
    throw new Error(`Draco: uvs need 2 numbers per vertex (got length ${uvs.length} for ${vertices} vertices)`);
  }
  const out = Float32Array.from(uvs);
  for (let i = 0; i < out.length; i++) {
    if (!Number.isFinite(out[i])) {
      throw new Error(`Draco: uv ${Math.floor(i / 2)} is not finite`);
    }
  }
  return out;
}

/** draco3d@1.5's float-attribute mesh writer: `@types/draco3d@1.4` predates the `*ToMesh` names. */
interface DracoMeshBuilder {
  AddFloatAttributeToMesh(mesh: Mesh, attribute: number, count: number, size: number, values: Float32Array): number;
}

/** Encoded bytes plus the attribute ids the KHR extension maps. */
interface EncodedDracoMesh {
  bytes: Uint8Array;
  positionId: number;
  normalId: number | null;
  colorId: number | null;
  uvId: number | null;
}

/**
 * Builds the mesh (faces, POSITION, NORMAL/COLOR/TEX_COORD when present) and encodes
 * it with the resolved settings. Colors ride a normalized uint8 attribute,
 * like glTF's `COLOR_0`. Throws when the encoder reports failure.
 */
function encodeMeshDraco(
  module: EncoderModule,
  positions: Float32Array,
  indices: Uint32Array,
  normals: Float32Array | null,
  colors: Uint8Array | null,
  uvs: Float32Array | null,
  settings: ResolvedEncodeSettings,
): EncodedDracoMesh {
  const mesh = new module.Mesh();
  const builder = new module.MeshBuilder() as unknown as MeshBuilder & DracoMeshBuilder;
  try {
    builder.AddFacesToMesh(mesh, indices.length / 3, indices);
    const positionId = builder.AddFloatAttributeToMesh(mesh, module.POSITION, positions.length / 3, 3, positions);
    let normalId: number | null = null;
    let colorId: number | null = null;
    let uvId: number | null = null;
    if (normals !== null) {
      normalId = builder.AddFloatAttributeToMesh(mesh, module.NORMAL, normals.length / 3, 3, normals);
    }
    if (colors !== null) {
      colorId = builder.AddUInt8Attribute(mesh, module.COLOR, colors.length / 4, 4, colors);
      builder.SetNormalizedFlagForAttribute(mesh, colorId, true);
    }
    if (uvs !== null) {
      uvId = builder.AddFloatAttributeToMesh(mesh, module.TEX_COORD, uvs.length / 2, 2, uvs);
    }
    const encoder = new module.Encoder();
    try {
      encoder.SetSpeedOptions(10 - settings.level, 10 - settings.level);
      encoder.SetAttributeQuantization(module.POSITION, settings.positionBits);
      if (normals !== null) {
        encoder.SetAttributeQuantization(module.NORMAL, settings.normalBits);
      }
      if (colors !== null) {
        encoder.SetAttributeQuantization(module.COLOR, settings.colorBits);
      }
      if (uvs !== null) {
        encoder.SetAttributeQuantization(module.TEX_COORD, settings.uvBits);
      }
      encoder.SetEncodingMethod(
        settings.preserveFaceOrder ? module.MESH_SEQUENTIAL_ENCODING : module.MESH_EDGEBREAKER_ENCODING,
      );
      const out = new module.DracoInt8Array();
      try {
        const length = encoder.EncodeMeshToDracoBuffer(mesh, out);
        if (!(length > 0)) {
          throw new Error("Draco: encoding failed (the encoder returned no bytes)");
        }
        const bytes = new Uint8Array(length);
        for (let i = 0; i < length; i++) {
          bytes[i] = out.GetValue(i);
        }
        return { bytes, positionId, normalId, colorId, uvId };
      } finally {
        module.destroy(out);
      }
    } finally {
      module.destroy(encoder);
    }
  } finally {
    module.destroy(builder);
    module.destroy(mesh);
  }
}

/** The shared draco3d encoder module: `options`'s, or a lazily created one. */
async function encoderOf(options?: DracoCompressionOptions): Promise<EncoderModule> {
  if (options?.encoderModule !== undefined) {
    const module = options.encoderModule as EncoderModule;
    if (typeof module?.Encoder !== "function" || typeof module?.MeshBuilder !== "function") {
      throw new Error("Draco: the encoder module is not a draco3d encoder module (from createEncoderModule)");
    }
    return module;
  }
  if (options?.wasmBinary === undefined && sharedEncoderModule !== null) {
    return sharedEncoderModule;
  }
  const created = await createEncoderModule(options?.wasmBinary);
  if (options?.wasmBinary === undefined) {
    sharedEncoderModule = created;
  }
  return created;
}

/** The process-wide encoder module, created on the first default encode. */
let sharedEncoderModule: EncoderModule | null = null;

/**
 * Creates a draco3d encoder module, optionally seeded with the `.wasm` bytes.
 * Imported lazily so bundles that never touch Draco never load the encoder.
 */
async function createEncoderModule(wasmBinary?: Uint8Array | ArrayBuffer): Promise<EncoderModule> {
  let create: (options?: Record<string, unknown>) => Promise<EncoderModule>;
  try {
    ({ createEncoderModule: create } = await import("draco3d"));
  } catch {
    throw new Error("Draco: encoding needs the draco3d package (npm install draco3d), or pass an encoder module");
  }
  const init: Record<string, unknown> = {};
  if (wasmBinary !== undefined) {
    init["wasmBinary"] = wasmBinary;
  }
  return create(init);
}

/**
 * An imported glTF mesh: one scene mesh per glTF (node, primitive). The
 * node's world placement is baked into `positions` (world coordinates, like
 * the exporters write with the default `applyPlacement`) and also returned
 * as `placement` (column-major, like `scene.matrix4`).
 *
 * Only uncompressed buffers are read. `KHR_draco_mesh_compression`
 * primitives throw (decode them with {@link importGLTFAsync}), as do sparse
 * accessors. External (non-embedded) `.bin` buffers load through
 * {@link importGLTFAsync}'s `resolveBuffer`. Animations, skins, materials
 * beyond per-vertex colors and `TEXCOORD_1` and up are skipped: the documented
 * subset.
 */
export interface ImportedMesh {
  /** The node name, else the mesh name, else `"mesh"` (deduplicated). */
  name: string;
  /** World positions: 3 f64 per vertex, node placements baked in. */
  positions: Float64Array;
  /** 3 vertex ids per triangle. */
  indices: Uint32Array;
  /** The node's world matrix: 16 numbers, column-major. */
  placement: Float64Array;
  /** RGBA bytes per vertex from `COLOR_0`, or `null` without colors. */
  colors: Uint8Array | null;
  /**
   * 2 floats per vertex from `TEXCOORD_0` (`scene.addMesh`'s `uvs`), or `null`
   * without UVs. In glTF's convention: `v` points down from the image's top.
   */
  uvs: Float32Array | null;
}

/**
 * Reads a glTF 2.0 asset: a binary `.glb`, or a `.gltf` JSON whose buffers
 * are embedded base64 data URIs. Every triangle primitive of every mesh
 * instanced by the default scene becomes one mesh (node transforms composed
 * down the hierarchy, so placements are preserved); other primitive modes
 * throw, except `TRIANGLE_STRIP` and `TRIANGLE_FAN`, which are expanded.
 *
 * @throws When the container is malformed, a buffer is external (load it
 * with {@link importGLTFAsync} and a `resolveBuffer` callback), a primitive
 * is Draco-compressed (decode it with {@link importGLTFAsync}) or not a
 * triangle mesh, or the file holds no triangle meshes.
 */
export function importGLTF(bytes: Uint8Array | ArrayBuffer): ImportedMesh[] {
  const raw = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const { json, bin } = splitContainer(raw);
  const doc = checkedDocument(json);
  const buffers = resolveBuffers(doc, bin);
  const usedNames = new Map<string, number>();
  const parts = collectPrimitiveSpecs(doc).map((spec) =>
    readPrimitive(doc, buffers, spec.node, spec.mesh, spec.primitive, spec.world, usedNames),
  );
  // Primitives with no triangles (empty index lists) contribute nothing.
  const kept = parts.filter((p) => p.indices.length > 0);
  if (kept.length === 0) {
    throw new Error("glTF: the file has no triangle meshes");
  }
  return kept;
}

/** A glTF primitive with its ancestry resolved: the mesh and node that own it and its world matrix. */
interface PrimitiveSpec {
  node: GltfNode;
  mesh: { name?: string; primitives: GltfPrimitive[] };
  primitive: GltfPrimitive;
  /** The node's world matrix: 16 numbers, column-major. */
  world: Float64Array;
}

/**
 * Walks the default scene's node hierarchy once, resolving every instanced
 * primitive with the world matrix its node placements compose. Shared by the
 * sync and async importers, so both see the same nodes in the same order and
 * fail on the same malformed hierarchies.
 */
function collectPrimitiveSpecs(doc: GltfDocument): PrimitiveSpec[] {
  const roots = rootNodes(doc);
  const specs: PrimitiveSpec[] = [];
  const identity = Float64Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  const trail = new Set<number>();
  const visit = (nodeIndex: number, parent: Float64Array): void => {
    const nodes = doc.nodes ?? [];
    const node = nodes[nodeIndex];
    if (node === undefined) {
      throw new Error(`glTF: node ${nodeIndex} is missing`);
    }
    if (trail.has(nodeIndex)) {
      throw new Error(`glTF: the node hierarchy cycles through node ${nodeIndex}`);
    }
    trail.add(nodeIndex);
    try {
      const world = multiplyAffine(parent, localMatrix(node, nodeIndex));
      if (node.mesh !== undefined) {
        const mesh = doc.meshes?.[node.mesh];
        if (mesh === undefined) {
          throw new Error(`glTF: node ${nodeIndex} points at missing mesh ${node.mesh}`);
        }
        for (const primitive of mesh.primitives) {
          specs.push({ node, mesh, primitive, world });
        }
      }
      for (const child of node.children ?? []) {
        if (!Number.isInteger(child) || child < 0 || child >= nodes.length) {
          throw new Error(`glTF: node ${nodeIndex} has child ${child} past its ${nodes.length} nodes`);
        }
        visit(child, world);
      }
    } finally {
      trail.delete(nodeIndex);
    }
  };
  for (const root of roots) {
    visit(root, identity);
  }
  return specs;
}

/**
 * A Draco-decoded triangle mesh: 3 f64 positions per vertex and 3 Uint32
 * vertex ids per triangle, ready for `scene.addMesh`.
 */
export interface DecodedDracoMesh {
  /** 3 f64 per vertex. */
  positions: Float64Array;
  /** 3 vertex ids per triangle. */
  indices: Uint32Array;
  /** 2 floats per vertex from the texture-coordinate attribute, or `null` without one. */
  uvs: Float32Array | null;
}

/**
 * Supplies an external glTF buffer's bytes, by its `uri` as declared
 * (`"model.bin"`, a relative path, an absolute URL: passed through
 * verbatim). Called at most once per external buffer, in buffer order.
 *
 * The importer never fetches: this callback is the only way external bytes
 * enter (fetch / FileReader / fs on the caller's side). Cap sizes here —
 * reject URIs outside the asset's directory and buffers past a byte budget
 * before returning them; oversized or hostile buffers otherwise land in
 * memory unchecked.
 */
export type GltfBufferResolver = (
  uri: string,
) => Uint8Array | ArrayBuffer | Promise<Uint8Array | ArrayBuffer>;

/** Options of {@link decodeDraco} and {@link importGLTFAsync}. */
export interface DracoDecodeOptions {
  /**
   * A ready draco3d decoder module, as `createDecoderModule` resolves. When
   * omitted, one is created from the `draco3d` package and shared across
   * calls. Pass one in to reuse a module (or to supply the `.wasm` in an
   * environment the default lookup cannot reach, like a bundled Worker).
   */
  decoderModule?: unknown;
  /**
   * The decoder `.wasm` bytes, forwarded to `createDecoderModule` when the
   * shared module is created. Bundlers cannot always locate draco3d's own
   * `.wasm` next to its glue (Vite workers, inlined builds): fetch the
   * decoder asset (a `?url` import of `draco3d/draco_decoder.wasm`) and pass
   * its bytes here. Ignored when `decoderModule` is given.
   */
  wasmBinary?: Uint8Array | ArrayBuffer;
}

/** Options of {@link importGLTFAsync}: Draco decoding plus external buffers. */
export interface GltfImportOptions extends DracoDecodeOptions {
  /**
   * Loads external (non-embedded) `.bin` buffers, by URI. Without one, an
   * external buffer throws. Embedded data URIs and the GLB body chunk never
   * consult it.
   */
  resolveBuffer?: GltfBufferResolver;
}

/**
 * Decodes a standalone Draco file (`.drc`): a compressed triangle mesh. Only
 * triangle meshes decode; point clouds throw.
 *
 * The draco3d decoder (WASM) initializes on first use; its module is shared
 * across calls. Needs the `draco3d` package installed unless `options`
 * supplies the module or its `.wasm`.
 *
 * @throws When the bytes are not a Draco triangle mesh.
 */
export async function decodeDraco(
  bytes: Uint8Array | ArrayBuffer,
  options?: DracoDecodeOptions,
): Promise<DecodedDracoMesh> {
  const raw = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const module = await decoderOf(options);
  return decodeDracoMesh(module, raw, "Draco");
}

/**
 * Reads a glTF 2.0 asset like {@link importGLTF}: `KHR_draco_mesh_compression`
 * primitives decode through the draco3d decoder (WASM), and external
 * (non-embedded) `.bin` buffers load through `options.resolveBuffer`. A Draco
 * primitive's positions and faces come from its compressed buffer view
 * (attribute ids from the extension); its node placement still bakes into
 * world positions and its `TEXCOORD_0` UVs, while colors read as `null` (compressed attributes beyond
 * POSITION are not mapped).
 *
 * Primitives without the extension decode synchronously in place, so assets
 * without Draco never initialize the decoder.
 *
 * @throws When the container is malformed, a buffer is external without a
 * `resolveBuffer` (or its bytes cannot be resolved), a Draco primitive fails
 * to decode, a primitive is not a triangle mesh, or the file holds no
 * triangle meshes.
 */
export async function importGLTFAsync(
  bytes: Uint8Array | ArrayBuffer,
  options?: GltfImportOptions,
): Promise<ImportedMesh[]> {
  const raw = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const { json, bin } = splitContainer(raw);
  const doc = checkedDocument(json);
  const buffers = await resolveBuffersAsync(doc, bin, options?.resolveBuffer);
  const specs = collectPrimitiveSpecs(doc);
  const usedNames = new Map<string, number>();
  let shared: DecoderModule | null = null;
  const decoderOfOptions = async (): Promise<DecoderModule> => {
    if (shared === null) {
      shared = await decoderOf(options);
    }
    return shared;
  };
  const parts: ImportedMesh[] = [];
  for (const spec of specs) {
    if (spec.primitive.extensions?.["KHR_draco_mesh_compression"] === undefined) {
      parts.push(readPrimitive(doc, buffers, spec.node, spec.mesh, spec.primitive, spec.world, usedNames));
    } else {
      parts.push(await readPrimitiveDraco(doc, buffers, spec, usedNames, decoderOfOptions));
    }
  }
  // Primitives with no triangles (empty index lists) contribute nothing.
  const kept = parts.filter((p) => p.indices.length > 0);
  if (kept.length === 0) {
    throw new Error("glTF: the file has no triangle meshes");
  }
  return kept;
}

/** The shared draco3d decoder module: `options`'s, or a lazily created one. */
async function decoderOf(options?: DracoDecodeOptions): Promise<DecoderModule> {
  if (options?.decoderModule !== undefined) {
    const module = options.decoderModule as DecoderModule;
    if (typeof module?.Decoder !== "function" || typeof module?.DecoderBuffer !== "function") {
      throw new Error("Draco: the decoder module is not a draco3d decoder module (from createDecoderModule)");
    }
    return module;
  }
  if (options?.wasmBinary === undefined && sharedDecoderModule !== null) {
    return sharedDecoderModule;
  }
  const created = await createDecoderModule(options?.wasmBinary);
  if (options?.wasmBinary === undefined) {
    sharedDecoderModule = created;
  }
  return created;
}

/** The process-wide decoder module, created on the first default decode. */
let sharedDecoderModule: DecoderModule | null = null;

/**
 * Creates a draco3d decoder module, optionally seeded with the `.wasm` bytes.
 * Imported lazily so bundles that never touch Draco never load the decoder.
 */
async function createDecoderModule(wasmBinary?: Uint8Array | ArrayBuffer): Promise<DecoderModule> {
  let create: (options?: Record<string, unknown>) => Promise<DecoderModule>;
  try {
    ({ createDecoderModule: create } = await import("draco3d"));
  } catch {
    throw new Error("Draco: decoding needs the draco3d package (npm install draco3d), or pass a decoder module");
  }
  const init: Record<string, unknown> = {};
  if (wasmBinary !== undefined) {
    init["wasmBinary"] = wasmBinary;
  }
  return create(init);
}

/**
 * Decodes Draco-compressed bytes into positions and faces. `what` prefixes
 * every error (`"Draco"` for files, `glTF: "name"` for primitives). Reads
 * POSITION by type, or the extension's attribute id when one is given (the
 * ids the encoder wrote need not match the decoder's lookup), and likewise the
 * UVs: by type when `uvId` is undefined (a file), by id for a primitive that
 * maps them, not at all for `null`.
 */
function decodeDracoMesh(
  module: DecoderModule,
  bytes: Uint8Array,
  what: string,
  positionId?: number,
  uvId?: number | null,
): DecodedDracoMesh {
  const decoder = new module.Decoder();
  const buffer = new module.DecoderBuffer();
  // The decoder reads an Int8Array view: 1-byte elements, so any offset aligns.
  const view = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  try {
    let geometryType: number;
    try {
      buffer.Init(view, view.length);
      geometryType = decoder.GetEncodedGeometryType(buffer) as number;
    } catch {
      throw new Error(`${what}: the bytes are not Draco-compressed geometry`);
    }
    if (geometryType !== (module.TRIANGULAR_MESH as number)) {
      throw new Error(`${what}: only Draco triangle meshes decode (not point clouds)`);
    }
    const mesh = new module.Mesh();
    try {
      const status = decoder.DecodeBufferToMesh(buffer, mesh);
      if (!status.ok()) {
        throw new Error(`${what}: Draco decode failed (${status.error_msg()})`);
      }
      const vertices = mesh.num_points();
      const attribute =
        positionId === undefined
          ? decoder.GetAttribute(mesh, checkedPositionId(decoder, mesh, module, what))
          : checkedUniqueAttribute(decoder, mesh, positionId, what);
      const positions = new Float64Array(3 * vertices);
      const floats = new module.DracoFloat32Array();
      try {
        decoder.GetAttributeFloatForAllPoints(mesh, attribute, floats);
        for (let v = 0; v < vertices; v++) {
          positions[3 * v] = floats.GetValue(3 * v);
          positions[3 * v + 1] = floats.GetValue(3 * v + 1);
          positions[3 * v + 2] = floats.GetValue(3 * v + 2);
        }
      } finally {
        module.destroy(floats);
      }
      let uvAttribute: Attribute | undefined;
      if (typeof uvId === "number") {
        uvAttribute = checkedUniqueAttribute(decoder, mesh, uvId, what, "TEXCOORD_0");
      } else if (uvId === undefined) {
        const id = decoder.GetAttributeId(mesh, module.TEX_COORD);
        uvAttribute = id < 0 ? undefined : decoder.GetAttribute(mesh, id);
      }
      let uvs: Float32Array | null = null;
      if (uvAttribute !== undefined) {
        if (uvAttribute.num_components() !== 2) {
          throw new Error(`${what}: the Draco UV attribute has ${uvAttribute.num_components()} components (2 expected)`);
        }
        uvs = new Float32Array(2 * vertices);
        const values = new module.DracoFloat32Array();
        try {
          decoder.GetAttributeFloatForAllPoints(mesh, uvAttribute, values);
          for (let i = 0; i < uvs.length; i++) {
            uvs[i] = values.GetValue(i);
          }
        } finally {
          module.destroy(values);
        }
      }
      const faces = mesh.num_faces();
      const indices = new Uint32Array(3 * faces);
      const face = new module.DracoInt32Array();
      try {
        for (let f = 0; f < faces; f++) {
          decoder.GetFaceFromMesh(mesh, f, face);
          indices[3 * f] = face.GetValue(0);
          indices[3 * f + 1] = face.GetValue(1);
          indices[3 * f + 2] = face.GetValue(2);
        }
      } finally {
        module.destroy(face);
      }
      return { positions, indices, uvs };
    } finally {
      module.destroy(mesh);
    }
  } finally {
    module.destroy(buffer);
    module.destroy(decoder);
  }
}

/** POSITION by decoder lookup (standalone `.drc`): throws without one. */
function checkedPositionId(decoder: Decoder, mesh: Mesh, module: DecoderModule, what: string): number {
  const id = decoder.GetAttributeId(mesh, module.POSITION);
  if (id < 0) {
    throw new Error(`${what}: the Draco mesh has no POSITION attribute`);
  }
  return id;
}

/** POSITION by the KHR extension's attribute id: throws on a dangling one. */
function checkedUniqueAttribute(
  decoder: Decoder,
  mesh: Mesh,
  id: number,
  what: string,
  name = "POSITION",
): Attribute {
  const attribute = decoder.GetAttributeByUniqueId(mesh, id);
  if (attribute === undefined || attribute === null) {
    throw new Error(`${what} points at missing Draco ${name} attribute ${id}`);
  }
  return attribute;
}

/** Reads one Draco-compressed primitive: the extension's buffer view decoded, placement baked. */
async function readPrimitiveDraco(
  doc: GltfDocument,
  buffers: Uint8Array[],
  spec: PrimitiveSpec,
  usedNames: Map<string, number>,
  decoderOfOptions: () => Promise<DecoderModule>,
): Promise<ImportedMesh> {
  const name = dedupName(spec.node, spec.mesh, usedNames);
  const what = `glTF: "${name}"`;
  const extension = spec.primitive.extensions?.["KHR_draco_mesh_compression"] as
    | { bufferView?: unknown; attributes?: unknown }
    | undefined;
  const viewIndex = extension?.bufferView;
  const attributes = extension?.attributes;
  if (
    typeof extension !== "object" ||
    extension === null ||
    !Number.isInteger(viewIndex) ||
    typeof attributes !== "object" ||
    attributes === null
  ) {
    throw new Error(`${what} has a bad KHR_draco_mesh_compression extension (a buffer view and attributes)`);
  }
  if ((spec.primitive.mode ?? 4) !== 4) {
    throw new Error(
      `${what} has primitive mode ${spec.primitive.mode} (only TRIANGLES Draco primitives are supported)`,
    );
  }
  const bufferView = doc.bufferViews?.[viewIndex as number];
  if (bufferView === undefined) {
    throw new Error(`${what} points at missing buffer view ${viewIndex}`);
  }
  const buffer = buffers[bufferView.buffer ?? 0];
  if (buffer === undefined) {
    throw new Error(`${what} points at missing buffer ${bufferView.buffer ?? 0}`);
  }
  const base = bufferView.byteOffset ?? 0;
  const end = base + (bufferView.byteLength ?? buffer.length - base);
  if (!(base >= 0) || end > buffer.length) {
    throw new Error(`${what} Draco buffer view runs past the end of its buffer`);
  }
  const module = await decoderOfOptions();
  // POSITION rides the extension's attribute ids (the ids the encoder wrote
  // need not match the decoder's own lookup); only POSITION maps, like the
  // sync path only POSITION feeds positions.
  const positionAttr = (attributes as Record<string, unknown>)["POSITION"];
  let positionId: number | undefined;
  if (positionAttr !== undefined) {
    if (!Number.isInteger(positionAttr) || (positionAttr as number) < 0) {
      throw new Error(`${what} has a bad KHR_draco_mesh_compression POSITION attribute id`);
    }
    positionId = positionAttr as number;
  }
  const uvAttr = (attributes as Record<string, unknown>)["TEXCOORD_0"];
  if (uvAttr !== undefined && (!Number.isInteger(uvAttr) || (uvAttr as number) < 0)) {
    throw new Error(`${what} has a bad KHR_draco_mesh_compression TEXCOORD_0 attribute id`);
  }
  const decoded = decodeDracoMesh(
    module,
    buffer.subarray(base, end),
    what,
    positionId,
    uvAttr === undefined ? null : (uvAttr as number),
  );
  // The node placement, baked: world positions, like the exporters write.
  const { positions, indices } = decoded;
  // UVs left out of the bitstream may still ride a plain accessor.
  const plainUvs = spec.primitive.attributes["TEXCOORD_0"];
  const uvs =
    decoded.uvs ??
    (typeof plainUvs === "number" && doc.accessors?.[plainUvs]?.bufferView !== undefined
      ? readUvs(doc, buffers, plainUvs, positions.length / 3, name)
      : null);
  const world = spec.world;
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    positions[i] = world[0] * x + world[4] * y + world[8] * z + world[12];
    positions[i + 1] = world[1] * x + world[5] * y + world[9] * z + world[13];
    positions[i + 2] = world[2] * x + world[6] * y + world[10] * z + world[14];
  }
  return { name, positions, indices, placement: world.slice(), colors: null, uvs };
}

/** The asset JSON, checked to be an object before the importers read it. */
function checkedDocument(json: unknown): GltfDocument {
  const doc = json as GltfDocument;
  if (typeof doc !== "object" || doc === null || Array.isArray(doc)) {
    throw new Error("glTF: the asset is not a JSON object");
  }
  return doc;
}

interface GltfDocument {
  scene?: number;
  scenes?: { nodes?: number[] }[];
  nodes?: GltfNode[];
  meshes?: { name?: string; primitives: GltfPrimitive[] }[];
  accessors?: GltfAccessor[];
  bufferViews?: GltfBufferView[];
  buffers?: { byteLength: number; uri?: string }[];
}

interface GltfNode {
  name?: string;
  mesh?: number;
  children?: number[];
  matrix?: number[];
  translation?: number[];
  rotation?: number[];
  scale?: number[];
}

interface GltfPrimitive {
  attributes: Record<string, number>;
  indices?: number;
  mode?: number;
  extensions?: Record<string, unknown>;
}

interface GltfAccessor {
  bufferView?: number;
  byteOffset?: number;
  componentType: number;
  count: number;
  type: string;
  normalized?: boolean;
  sparse?: unknown;
}

interface GltfBufferView {
  buffer?: number;
  byteOffset?: number;
  byteLength?: number;
  byteStride?: number;
}

/** Splits the input into the asset JSON and the GLB body chunk (if any). */
function splitContainer(raw: Uint8Array): { json: unknown; bin: Uint8Array | null } {
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  if (raw.length >= 12 && view.getUint32(0, true) === 0x46546c67) {
    const version = view.getUint32(4, true);
    if (version !== 2) {
      throw new Error(`glTF: unsupported GLB version ${version} (only 2.0)`);
    }
    const length = view.getUint32(8, true);
    if (length > raw.length) {
      throw new Error("glTF: the GLB length runs past the end of the file");
    }
    let json: Uint8Array | null = null;
    let bin: Uint8Array | null = null;
    let at = 12;
    while (at + 8 <= length) {
      const chunkLength = view.getUint32(at, true);
      const chunkType = view.getUint32(at + 4, true);
      const start = at + 8;
      const end = start + chunkLength;
      if (end > raw.length) {
        throw new Error("glTF: a GLB chunk runs past the end of the file");
      }
      if (chunkType === 0x4e4f534a) {
        json = raw.subarray(start, end);
      } else if (chunkType === 0x004e4942) {
        bin = raw.subarray(start, end);
      }
      at = end;
    }
    if (json === null) {
      throw new Error("glTF: the GLB has no JSON chunk");
    }
    return { json: parseAssetJson(json), bin };
  }
  return { json: parseAssetJson(raw), bin: null };
}

function parseAssetJson(bytes: Uint8Array): unknown {
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch (error) {
    throw new Error(`glTF: the asset JSON does not parse (${error instanceof Error ? error.message : String(error)})`);
  }
}

/** Resolves every declared buffer to bytes: the GLB body chunk or embedded base64 data URIs. */
function resolveBuffers(doc: GltfDocument, bin: Uint8Array | null): Uint8Array[] {
  const declared = doc.buffers ?? [];
  if (bin !== null) {
    if (declared.length === 0) {
      return [bin];
    }
    return declared.map((b, i) => {
      if (i === 0 && b.uri === undefined) {
        return bin.subarray(0, Math.min(b.byteLength, bin.length));
      }
      if (b.uri === undefined) {
        throw new Error(`glTF: buffer ${i} has no URI and is not the GLB body (only the first GLB buffer may omit its URI)`);
      }
      return decodeDataUri(b.uri, i);
    });
  }
  if (declared.length === 0) {
    // No buffers declared: files without accessors (no meshes, or scenes
    // alone) report "no triangle meshes" below; anything referencing buffer
    // 0 fails at the accessor with a missing-buffer error.
    return [];
  }
  return declared.map((b, i) => {
    if (b.uri === undefined) {
      throw new Error(
        `glTF: buffer ${i} is external (only embedded base64 buffers are supported; load it with importGLTFAsync and a resolveBuffer callback)`,
      );
    }
    return decodeDataUri(b.uri, i);
  });
}

/**
 * Resolves every declared buffer to bytes like {@link resolveBuffers}, and
 * additionally loads external (non-embedded) `.bin` buffers through
 * `resolveBuffer`. Embedded data URIs and the GLB body chunk decode inline,
 * exactly as the sync path does; the importer itself never fetches.
 */
async function resolveBuffersAsync(
  doc: GltfDocument,
  bin: Uint8Array | null,
  resolveBuffer?: GltfBufferResolver,
): Promise<Uint8Array[]> {
  const declared = doc.buffers ?? [];
  if (bin !== null && declared.length === 0) {
    return [bin];
  }
  if (bin === null && declared.length === 0) {
    // No buffers declared: files without accessors (no meshes, or scenes
    // alone) report "no triangle meshes" below; anything referencing buffer
    // 0 fails at the accessor with a missing-buffer error.
    return [];
  }
  const out: Uint8Array[] = [];
  for (const [i, b] of declared.entries()) {
    if (bin !== null && i === 0 && b.uri === undefined) {
      out.push(bin.subarray(0, Math.min(b.byteLength, bin.length)));
    } else if (b.uri === undefined) {
      throw new Error(`glTF: buffer ${i} has no URI and is not the GLB body (only the first GLB buffer may omit its URI)`);
    } else if (b.uri.startsWith("data:")) {
      out.push(decodeDataUri(b.uri, i));
    } else {
      out.push(await resolveExternal(b.uri, i, resolveBuffer));
    }
  }
  return out;
}

/**
 * Loads one external buffer through the caller's resolver. Throws without a
 * resolver, wraps a resolver failure with the buffer's index and URI, and
 * rejects anything but bytes.
 */
async function resolveExternal(
  uri: string,
  index: number,
  resolveBuffer?: GltfBufferResolver,
): Promise<Uint8Array> {
  const what = `glTF: buffer ${index} ("${shortUri(uri)}")`;
  if (resolveBuffer === undefined) {
    throw new Error(`${what} is external; pass a resolveBuffer callback to load it`);
  }
  let bytes: Uint8Array | ArrayBuffer;
  try {
    bytes = await resolveBuffer(uri);
  } catch (error) {
    throw new Error(`${what} could not be resolved (${error instanceof Error ? error.message : String(error)})`);
  }
  if (bytes instanceof Uint8Array) {
    return bytes;
  }
  if (bytes instanceof ArrayBuffer) {
    return new Uint8Array(bytes);
  }
  throw new Error(`${what} resolved to no bytes (the resolveBuffer callback must return a Uint8Array or ArrayBuffer)`);
}

/** A URI shortened for error messages: hostile files can declare megabytes of URI. */
function shortUri(uri: string): string {
  return uri.length > 120 ? `${uri.slice(0, 117)}...` : uri;
}

function decodeDataUri(uri: string, buffer: number): Uint8Array {
  const marker = ";base64,";
  const at = uri.indexOf(marker);
  if (!uri.startsWith("data:") || at === -1) {
    throw new Error(
      `glTF: buffer ${buffer} is external (only embedded base64 buffers are supported; load it with importGLTFAsync and a resolveBuffer callback)`,
    );
  }
  return decodeBase64(uri.slice(at + marker.length), buffer);
}

const BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

function decodeBase64(text: string, buffer: number): Uint8Array {
  const fail = (): Error => new Error(`glTF: buffer ${buffer} is not valid base64`);
  const clean = text.replace(/\s/g, "");
  if (clean.length % 4 !== 0) {
    throw fail();
  }
  let padding = 0;
  if (clean.endsWith("==")) {
    padding = 2;
  } else if (clean.endsWith("=")) {
    padding = 1;
  }
  const out = new Uint8Array((clean.length / 4) * 3 - padding);
  let at = 0;
  for (let i = 0; i < clean.length; i += 4) {
    let n = 0;
    let pad = 0;
    for (let k = 0; k < 4; k++) {
      const c = clean[i + k];
      if (c === "=") {
        if (i + 4 !== clean.length || k < 2) {
          throw fail();
        }
        pad++;
        n <<= 6;
      } else {
        const v = BASE64.indexOf(c);
        if (v === -1 || pad > 0) {
          throw fail();
        }
        n = (n << 6) | v;
      }
    }
    out[at++] = (n >>> 16) & 0xff;
    if (pad < 2) {
      out[at++] = (n >>> 8) & 0xff;
    }
    if (pad < 1) {
      out[at++] = n & 0xff;
    }
  }
  return out;
}

/** The root node indices: the default scene's, or every unparented node without scenes. */
function rootNodes(doc: GltfDocument): number[] {
  const nodes = doc.nodes ?? [];
  const scenes = doc.scenes ?? [];
  if (scenes.length > 0) {
    const index = doc.scene ?? 0;
    const scene = scenes[index];
    if (scene === undefined) {
      throw new Error(`glTF: scene ${index} is missing (the file has ${scenes.length})`);
    }
    const roots = scene.nodes ?? [];
    for (const root of roots) {
      if (!Number.isInteger(root) || root < 0 || root >= nodes.length) {
        throw new Error(`glTF: scene ${index} points at missing node ${root}`);
      }
    }
    return roots;
  }
  const parented = new Set<number>();
  for (const node of nodes) {
    for (const child of node.children ?? []) {
      parented.add(child);
    }
  }
  return nodes.map((_, i) => i).filter((i) => !parented.has(i));
}

/** The node's local matrix (column-major): `matrix`, else TRS composed. */
function localMatrix(node: GltfNode, nodeIndex: number): Float64Array {
  const what = `node ${nodeIndex}${node.name !== undefined ? ` "${node.name}"` : ""}`;
  if (node.matrix !== undefined) {
    if (node.matrix.length !== 16 || node.matrix.some((x) => !Number.isFinite(x))) {
      throw new Error(`glTF: ${what} has a bad matrix (16 finite numbers)`);
    }
    return Float64Array.from(node.matrix);
  }
  const t = node.translation ?? [0, 0, 0];
  const r = node.rotation ?? [0, 0, 0, 1];
  const s = node.scale ?? [1, 1, 1];
  if (t.length !== 3 || r.length !== 4 || s.length !== 3 || [...t, ...r, ...s].some((x) => !Number.isFinite(x))) {
    throw new Error(`glTF: ${what} has a bad TRS transform`);
  }
  const len = Math.hypot(r[0], r[1], r[2], r[3]);
  if (!(len > 1e-30)) {
    throw new Error(`glTF: ${what} has a zero rotation quaternion`);
  }
  const x = r[0] / len;
  const y = r[1] / len;
  const z = r[2] / len;
  const w = r[3] / len;
  // Row-major rotation from the unit quaternion, scaled into column-major.
  const r00 = 1 - 2 * (y * y + z * z);
  const r01 = 2 * (x * y - z * w);
  const r02 = 2 * (x * z + y * w);
  const r10 = 2 * (x * y + z * w);
  const r11 = 1 - 2 * (x * x + z * z);
  const r12 = 2 * (y * z - x * w);
  const r20 = 2 * (x * z - y * w);
  const r21 = 2 * (y * z + x * w);
  const r22 = 1 - 2 * (x * x + y * y);
  return Float64Array.from([
    r00 * s[0], r10 * s[0], r20 * s[0], 0,
    r01 * s[1], r11 * s[1], r21 * s[1], 0,
    r02 * s[2], r12 * s[2], r22 * s[2], 0,
    t[0], t[1], t[2], 1,
  ]);
}

/** Column-major product of two affine matrices. */
function multiplyAffine(a: Float64Array, b: Float64Array): Float64Array {
  const out = new Float64Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

const COMPONENT_SIZE: Record<number, number> = {
  5120: 1, // byte
  5121: 1, // unsigned byte
  5122: 2, // short
  5123: 2, // unsigned short
  5125: 4, // unsigned int
  5126: 4, // float
};

const TYPE_COMPONENTS: Record<string, number> = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
};

interface AccessorData {
  buffer: Uint8Array;
  view: DataView;
  base: number;
  count: number;
  componentType: number;
  components: number;
  size: number;
  stride: number;
}

/** Resolves an accessor to its buffer bytes, honoring view/element offsets and interleaved strides. */
function accessorData(doc: GltfDocument, buffers: Uint8Array[], index: number, what: string): AccessorData {
  const accessor = doc.accessors?.[index];
  if (accessor === undefined) {
    throw new Error(`glTF: the ${what} accessor ${index} is missing`);
  }
  const size = COMPONENT_SIZE[accessor.componentType];
  if (size === undefined) {
    throw new Error(`glTF: the ${what} accessor has component type ${accessor.componentType}`);
  }
  const components = TYPE_COMPONENTS[accessor.type];
  if (components === undefined) {
    throw new Error(`glTF: the ${what} accessor has type "${accessor.type}"`);
  }
  if (!Number.isInteger(accessor.count) || accessor.count < 0) {
    throw new Error(`glTF: the ${what} accessor has count ${accessor.count}`);
  }
  if (accessor.sparse !== undefined) {
    throw new Error(`glTF: sparse accessors are not supported (${what})`);
  }
  if (accessor.bufferView === undefined) {
    // No buffer view: every element is zero.
    return {
      buffer: new Uint8Array(0),
      view: new DataView(new ArrayBuffer(0)),
      base: 0,
      count: accessor.count,
      componentType: accessor.componentType,
      components,
      size,
      stride: components * size,
    };
  }
  const bufferView = doc.bufferViews?.[accessor.bufferView];
  if (bufferView === undefined) {
    throw new Error(`glTF: the ${what} buffer view ${accessor.bufferView} is missing`);
  }
  const buffer = buffers[bufferView.buffer ?? 0];
  if (buffer === undefined) {
    throw new Error(`glTF: the ${what} buffer ${bufferView.buffer ?? 0} is missing`);
  }
  const base = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const stride = bufferView.byteStride ?? components * size;
  if (!(stride >= components * size)) {
    throw new Error(`glTF: the ${what} stride ${stride} is narrower than its elements`);
  }
  if (accessor.count > 0) {
    const end = base + (accessor.count - 1) * stride + components * size;
    if (!(base >= 0) || end > buffer.length) {
      throw new Error(`glTF: the ${what} accessor runs past the end of its buffer`);
    }
  }
  return {
    buffer,
    view: new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength),
    base,
    count: accessor.count,
    componentType: accessor.componentType,
    components,
    size,
    stride,
  };
}

function readFloat(data: AccessorData, element: number, component: number): number {
  return data.view.getFloat32(data.base + element * data.stride + component * data.size, true);
}

function readPrimitive(
  doc: GltfDocument,
  buffers: Uint8Array[],
  node: GltfNode,
  mesh: { name?: string; primitives: GltfPrimitive[] },
  primitive: GltfPrimitive,
  world: Float64Array,
  usedNames: Map<string, number>,
): ImportedMesh {
  const name = dedupName(node, mesh, usedNames);
  if (primitive.extensions?.["KHR_draco_mesh_compression"] !== undefined) {
    throw new Error(`glTF: "${name}" uses KHR_draco_mesh_compression (decode it with importGLTFAsync)`);
  }
  if (typeof primitive.attributes["POSITION"] !== "number") {
    throw new Error(`glTF: "${name}" has a primitive without POSITION`);
  }
  const positions = readPositions(doc, buffers, primitive.attributes["POSITION"], name);
  const indices = readPrimitiveIndices(doc, buffers, primitive, positions.length / 3, name);
  // The node placement, baked: world positions, like the exporters write.
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    positions[i] = world[0] * x + world[4] * y + world[8] * z + world[12];
    positions[i + 1] = world[1] * x + world[5] * y + world[9] * z + world[13];
    positions[i + 2] = world[2] * x + world[6] * y + world[10] * z + world[14];
  }
  const colors =
    typeof primitive.attributes["COLOR_0"] === "number"
      ? readColors(doc, buffers, primitive.attributes["COLOR_0"], positions.length / 3, name)
      : null;
  const uvs =
    typeof primitive.attributes["TEXCOORD_0"] === "number"
      ? readUvs(doc, buffers, primitive.attributes["TEXCOORD_0"], positions.length / 3, name)
      : null;
  return { name, positions, indices, placement: world.slice(), colors, uvs };
}

/** The deduplicated mesh name: the node name, else the mesh name, else `"mesh"`. */
function dedupName(
  node: GltfNode,
  mesh: { name?: string; primitives: GltfPrimitive[] },
  usedNames: Map<string, number>,
): string {
  const base = node.name ?? mesh.name ?? "mesh";
  const seen = usedNames.get(base) ?? 0;
  usedNames.set(base, seen + 1);
  return seen === 0 ? base : `${base} (${seen + 1})`;
}

function readPositions(doc: GltfDocument, buffers: Uint8Array[], index: number, name: string): Float64Array {
  const data = accessorData(doc, buffers, index, "POSITION");
  if (data.components !== 3 || data.componentType !== 5126) {
    throw new Error(`glTF: "${name}" has POSITION with ${data.components} ${data.size}-byte components (VEC3 float)`);
  }
  const out = new Float64Array(3 * data.count);
  for (let i = 0; i < data.count; i++) {
    out[3 * i] = readFloat(data, i, 0);
    out[3 * i + 1] = readFloat(data, i, 1);
    out[3 * i + 2] = readFloat(data, i, 2);
    if (!Number.isFinite(out[3 * i]) || !Number.isFinite(out[3 * i + 1]) || !Number.isFinite(out[3 * i + 2])) {
      throw new Error(`glTF: "${name}" has a non-finite position`);
    }
  }
  return out;
}

function readPrimitiveIndices(
  doc: GltfDocument,
  buffers: Uint8Array[],
  primitive: GltfPrimitive,
  vertices: number,
  name: string,
): Uint32Array {
  const mode = primitive.mode ?? 4;
  let corners: Uint32Array;
  if (primitive.indices === undefined) {
    corners = Uint32Array.from({ length: accessorCount(doc, primitive, name) }, (_, i) => i);
  } else {
    const data = accessorData(doc, buffers, primitive.indices, "index");
    if (data.components !== 1 || (data.componentType !== 5121 && data.componentType !== 5123 && data.componentType !== 5125)) {
      throw new Error(`glTF: "${name}" has non-scalar 8/16/32-bit unsigned indices`);
    }
    corners = new Uint32Array(data.count);
    for (let i = 0; i < data.count; i++) {
      const at = data.base + i * data.stride;
      corners[i] =
        data.componentType === 5121
          ? data.view.getUint8(at)
          : data.componentType === 5123
            ? data.view.getUint16(at, true)
            : data.view.getUint32(at, true);
    }
  }
  for (const c of corners) {
    if (c >= vertices) {
      throw new Error(`glTF: "${name}" indexes vertex ${c} past its ${vertices} vertices`);
    }
  }
  if (mode === 4) {
    if (corners.length % 3 !== 0) {
      throw new Error(`glTF: "${name}" has ${corners.length} TRIANGLES corners (a multiple of 3)`);
    }
    return corners;
  }
  if (mode !== 5 && mode !== 6) {
    throw new Error(
      `glTF: "${name}" has primitive mode ${mode} (only TRIANGLES, TRIANGLE_STRIP and TRIANGLE_FAN are supported)`,
    );
  }
  if (corners.length < 3) {
    return new Uint32Array(0);
  }
  if (mode === 5) {
    // Triangle strip: alternate the winding so neighbours face the same way.
    const out = new Uint32Array((corners.length - 2) * 3);
    for (let i = 0; i + 2 < corners.length; i++) {
      if (i % 2 === 0) {
        out.set([corners[i], corners[i + 1], corners[i + 2]], 3 * i);
      } else {
        out.set([corners[i], corners[i + 2], corners[i + 1]], 3 * i);
      }
    }
    return out;
  }
  if (mode === 6) {
    const out = new Uint32Array((corners.length - 2) * 3);
    for (let i = 0; i + 2 < corners.length; i++) {
      out.set([corners[0], corners[i + 1], corners[i + 2]], 3 * i);
    }
    return out;
  }
  throw new Error(`glTF: unreachable primitive mode ${mode}`);
}

/** The corner count of a non-indexed primitive: its POSITION count. */
function accessorCount(doc: GltfDocument, primitive: GltfPrimitive, name: string): number {
  const accessor = doc.accessors?.[primitive.attributes["POSITION"]];
  if (accessor === undefined) {
    throw new Error(`glTF: "${name}" has a primitive without POSITION`);
  }
  return accessor.count;
}

function readUvs(
  doc: GltfDocument,
  buffers: Uint8Array[],
  index: number,
  vertices: number,
  name: string,
): Float32Array {
  const data = accessorData(doc, buffers, index, "TEXCOORD_0");
  if (data.components !== 2 || (data.componentType !== 5126 && data.componentType !== 5121 && data.componentType !== 5123)) {
    throw new Error(`glTF: "${name}" has TEXCOORD_0 with ${data.components} ${data.size}-byte components (float, u8 or u16 VEC2)`);
  }
  if (data.count !== vertices) {
    throw new Error(`glTF: "${name}" has ${data.count} UVs for ${vertices} vertices`);
  }
  const out = new Float32Array(2 * vertices);
  if (data.buffer.length === 0) {
    return out; // No buffer view: every element is zero.
  }
  for (let i = 0; i < vertices; i++) {
    for (let c = 0; c < 2; c++) {
      const at = data.base + i * data.stride + c * data.size;
      // The integer types are normalized, as the spec requires of them.
      const v =
        data.componentType === 5126
          ? data.view.getFloat32(at, true)
          : data.componentType === 5121
            ? data.view.getUint8(at) / 255
            : data.view.getUint16(at, true) / 65535;
      if (!Number.isFinite(v)) {
        throw new Error(`glTF: "${name}" has a non-finite UV at vertex ${i}`);
      }
      out[2 * i + c] = v;
    }
  }
  return out;
}

function readColors(
  doc: GltfDocument,
  buffers: Uint8Array[],
  index: number,
  vertices: number,
  name: string,
): Uint8Array {
  const data = accessorData(doc, buffers, index, "COLOR_0");
  if (
    (data.components !== 3 && data.components !== 4) ||
    (data.componentType !== 5121 && data.componentType !== 5126)
  ) {
    throw new Error(`glTF: "${name}" has COLOR_0 with ${data.components} ${data.size}-byte components (u8 or float VEC3/VEC4)`);
  }
  if (data.count !== vertices) {
    throw new Error(`glTF: "${name}" has ${data.count} colors for ${vertices} vertices`);
  }
  if (data.size === 1 && data.components === 4) {
    // RGBA bytes, packed or strided: copy whole quads.
    const out = new Uint8Array(4 * vertices);
    for (let i = 0; i < vertices; i++) {
      const at = data.base + i * data.stride;
      out[4 * i] = data.view.getUint8(at);
      out[4 * i + 1] = data.view.getUint8(at + 1);
      out[4 * i + 2] = data.view.getUint8(at + 2);
      out[4 * i + 3] = data.view.getUint8(at + 3);
    }
    return out;
  }
  const out = new Uint8Array(4 * vertices);
  for (let i = 0; i < vertices; i++) {
    for (let c = 0; c < 4; c++) {
      let v: number;
      if (c < data.components) {
        const at = data.base + i * data.stride + c * data.size;
        v = data.size === 1 ? data.view.getUint8(at) / 255 : data.view.getFloat32(at, true);
      } else {
        v = 1; // A VEC3 color is opaque.
      }
      out[4 * i + c] = Math.round(Math.min(1, Math.max(0, v)) * 255);
    }
  }
  return out;
}
