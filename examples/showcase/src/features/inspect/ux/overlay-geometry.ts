// The object overlays' geometry (#257): the bounding box's twelve edges and
// its dimension label anchors, and the normals as short line segments. All
// of it is built once per change (the mesh's geometry, the box, the length),
// never per frame, and the normals are capped so a million-triangle mesh
// draws a sample of them. Pure (no three), so vitest covers it.
import type { Vec3 } from "../../../io/mesh-data";
import type { Box } from "./object-info";

/** The most normal lines drawn: past it, every k-th face or vertex shows. */
export const MAX_NORMAL_LINES = 20_000;

/** The box's twelve edges as line segment endpoints, flat `x, y, z` (72 floats). */
export function boxEdges(box: Box): Float32Array {
  const { min: a, max: b } = box;
  const corner = (i: number): Vec3 => [i & 1 ? b[0] : a[0], i & 2 ? b[1] : a[1], i & 4 ? b[2] : a[2]];
  // Corner i has bit 0 = x max, bit 1 = y max, bit 2 = z max; an edge flips one bit.
  const pairs: [number, number][] = [];
  for (let i = 0; i < 8; i++) {
    for (const bit of [1, 2, 4]) {
      if ((i & bit) === 0) {
        pairs.push([i, i | bit]);
      }
    }
  }
  const out = new Float32Array(pairs.length * 6);
  pairs.forEach(([i, j], k) => {
    out.set(corner(i), 6 * k);
    out.set(corner(j), 6 * k + 3);
  });
  return out;
}

/**
 * Where the box's three dimension labels sit: the middle of one edge along
 * each axis, all three on edges through the minimum corner.
 */
export function boxLabelAnchors(box: Box): [x: Vec3, y: Vec3, z: Vec3] {
  const { min: a, max: b } = box;
  const mid = (i: number) => (a[i] + b[i]) / 2;
  return [
    [mid(0), a[1], a[2]],
    [a[0], mid(1), a[2]],
    [a[0], a[1], mid(2)],
  ];
}

/**
 * Face normals as line segments from each face's centroid, `length` long,
 * flat `x, y, z` (6 floats a line), in the buffers' local coordinates. At
 * most `cap` lines: every k-th live face past it. Faces without area draw
 * none.
 */
export function faceNormalLines(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  length: number,
  cap: number = MAX_NORMAL_LINES,
): Float32Array {
  const triSlots = Math.floor(indices.length / 3);
  let live = 0;
  for (let t = 0; t < triSlots; t++) {
    if (indices[3 * t] !== 0 || indices[3 * t + 1] !== 0 || indices[3 * t + 2] !== 0) {
      live++;
    }
  }
  const stride = Math.max(1, Math.ceil(live / Math.max(1, cap)));
  const out = new Float32Array(6 * Math.min(live, Math.ceil(live / stride)));
  let k = 0;
  let seen = 0;
  for (let t = 0; t < triSlots; t++) {
    const a = 3 * indices[3 * t];
    const b = 3 * indices[3 * t + 1];
    const c = 3 * indices[3 * t + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    if (seen++ % stride !== 0) {
      continue;
    }
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (!(len > 0) || k + 6 > out.length) {
      continue;
    }
    const cx = (positions[a] + positions[b] + positions[c]) / 3;
    const cy = (positions[a + 1] + positions[b + 1] + positions[c + 1]) / 3;
    const cz = (positions[a + 2] + positions[b + 2] + positions[c + 2]) / 3;
    const s = length / len;
    out[k++] = cx;
    out[k++] = cy;
    out[k++] = cz;
    out[k++] = cx + nx * s;
    out[k++] = cy + ny * s;
    out[k++] = cz + nz * s;
  }
  return k === out.length ? out : out.slice(0, k);
}

/**
 * Area-weighted vertex normals of the live triangles, 3 floats per vertex
 * slot, not normalised (for a mesh whose normals are off).
 */
export function accumulateVertexNormals(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
): Float32Array {
  const out = new Float32Array(positions.length);
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = 3 * indices[t];
    const b = 3 * indices[t + 1];
    const c = 3 * indices[t + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    out[a] += nx;
    out[a + 1] += ny;
    out[a + 2] += nz;
    out[b] += nx;
    out[b + 1] += ny;
    out[b + 2] += nz;
    out[c] += nx;
    out[c + 1] += ny;
    out[c + 2] += nz;
  }
  return out;
}

/**
 * Vertex normals as line segments from each used vertex, `length` long,
 * flat `x, y, z`, in local coordinates: the mesh's own normals when it has
 * them (`normals`), else area-weighted ones. At most `cap` lines (every k-th
 * used vertex past it); zero normals draw none.
 */
export function vertexNormalLines(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  used: ArrayLike<number>,
  normals: ArrayLike<number> | null,
  length: number,
  cap: number = MAX_NORMAL_LINES,
): Float32Array {
  const n = normals !== null && normals.length >= positions.length ? normals : accumulateVertexNormals(positions, indices);
  const stride = Math.max(1, Math.ceil(used.length / Math.max(1, cap)));
  const out = new Float32Array(6 * Math.ceil(used.length / stride));
  let k = 0;
  for (let i = 0; i < used.length; i += stride) {
    const v = 3 * used[i];
    const nx = n[v];
    const ny = n[v + 1];
    const nz = n[v + 2];
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (!(len > 0)) {
      continue;
    }
    const s = length / len;
    out[k++] = positions[v];
    out[k++] = positions[v + 1];
    out[k++] = positions[v + 2];
    out[k++] = positions[v] + nx * s;
    out[k++] = positions[v + 1] + ny * s;
    out[k++] = positions[v + 2] + nz * s;
  }
  return k === out.length ? out : out.slice(0, k);
}

/**
 * The vertex slots the live triangles use, ascending: what the box and the
 * vertex normals walk (a dead slot keeps stale positions).
 */
export function usedVertices(indices: ArrayLike<number>, vertexSlots: number): Uint32Array {
  const mark = new Uint8Array(vertexSlots);
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = indices[t];
    const b = indices[t + 1];
    const c = indices[t + 2];
    if ((a === 0 && b === 0 && c === 0) || a >= vertexSlots || b >= vertexSlots || c >= vertexSlots) {
      continue;
    }
    mark[a] = 1;
    mark[b] = 1;
    mark[c] = 1;
  }
  let count = 0;
  for (let v = 0; v < vertexSlots; v++) {
    count += mark[v];
  }
  const out = new Uint32Array(count);
  for (let v = 0, j = 0; v < vertexSlots; v++) {
    if (mark[v] === 1) {
      out[j++] = v;
    }
  }
  return out;
}
