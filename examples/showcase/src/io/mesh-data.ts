// What the parsers and the demo generators produce: world positions in f64
// and 3 Uint32 indices per triangle, ready for `store.addMesh` (both arrays
// transferable). Plus the bounding box and the exact-duplicate weld STL needs.

export type Vec3 = [x: number, y: number, z: number];

export interface MeshData {
  positions: Float64Array;
  indices: Uint32Array;
}

/** The centre and radius (half the diagonal) of the bounding box of `positions`. */
export function bounds(positions: Float64Array): { center: Vec3; radius: number } {
  let x0 = Infinity;
  let y0 = Infinity;
  let z0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  let z1 = -Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
    if (z < z0) z0 = z;
    if (z > z1) z1 = z;
  }
  if (!(x0 <= x1)) {
    return { center: [0, 0, 0], radius: 0 };
  }
  return {
    center: [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2],
    radius: Math.hypot(x1 - x0, y1 - y0, z1 - z0) / 2,
  };
}

/**
 * A triangle soup (3 corners of 3 coordinates per triangle) as an indexed
 * mesh: corners whose positions are bit-identical share a vertex (after
 * -0 is made +0, in place). Nearly equal ones stay apart. An open-addressing hash on the 6 words of each
 * corner's f64 coordinates, so a million triangles weld in well under a
 * second.
 */
export function weldCorners(corners: Float64Array): MeshData {
  const n = corners.length / 3;
  const words = new Uint32Array(corners.buffer, corners.byteOffset, corners.length * 2);
  let size = 16;
  while (size < 2 * n) {
    size *= 2;
  }
  const mask = size - 1;
  // Each slot holds a vertex id + 1 (0 is empty); `first` maps a vertex to its first corner.
  const table = new Uint32Array(size);
  const first = new Uint32Array(n);
  const indices = new Uint32Array(n);
  let vertices = 0;
  // -0 and +0 are the same position with different bits: make every zero +0.
  for (let i = 0; i < corners.length; i++) {
    if (corners[i] === 0) {
      corners[i] = 0;
    }
  }
  for (let c = 0; c < n; c++) {
    const w = 6 * c;
    let h = 0x811c9dc5;
    for (let k = 0; k < 6; k++) {
      h = Math.imul(h ^ words[w + k], 0x01000193);
    }
    h ^= h >>> 15;
    let slot = h & mask;
    for (;;) {
      const at = table[slot];
      if (at === 0) {
        table[slot] = vertices + 1;
        first[vertices] = c;
        indices[c] = vertices++;
        break;
      }
      const o = 6 * first[at - 1];
      if (
        words[o] === words[w] &&
        words[o + 1] === words[w + 1] &&
        words[o + 2] === words[w + 2] &&
        words[o + 3] === words[w + 3] &&
        words[o + 4] === words[w + 4] &&
        words[o + 5] === words[w + 5]
      ) {
        indices[c] = at - 1;
        break;
      }
      slot = (slot + 1) & mask;
    }
  }
  const positions = new Float64Array(3 * vertices);
  for (let v = 0; v < vertices; v++) {
    const c = 3 * first[v];
    positions[3 * v] = corners[c];
    positions[3 * v + 1] = corners[c + 1];
    positions[3 * v + 2] = corners[c + 2];
  }
  return { positions, indices };
}

/** Throws unless every index is below the vertex count and there is at least one triangle. */
export function checkMesh(data: MeshData, format: string): MeshData {
  if (data.indices.length === 0) {
    throw new Error(`${format}: the file has no faces`);
  }
  const vertices = data.positions.length / 3;
  for (let i = 0; i < data.indices.length; i++) {
    if (data.indices[i] >= vertices) {
      throw new Error(`${format}: face index ${data.indices[i]} is past the ${vertices} vertices`);
    }
  }
  for (let i = 0; i < data.positions.length; i++) {
    if (!Number.isFinite(data.positions[i])) {
      throw new Error(`${format}: a vertex coordinate is not a finite number`);
    }
  }
  return data;
}
