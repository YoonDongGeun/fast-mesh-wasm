// Mesh generators (world positions in f64, Uint32 indices, counter-clockwise
// outward triangles) and a tiny binary STL reader for drag-and-drop.

export type Vec3 = [x: number, y: number, z: number];

export interface MeshData {
  positions: Float64Array;
  indices: Uint32Array;
}

/** A closed UV sphere around `center` (y up), poles shared. */
export function sphere(center: Vec3, radius: number, segments = 48, rings = 32): MeshData {
  const positions: number[] = [];
  const indices: number[] = [];
  const [cx, cy, cz] = center;
  positions.push(cx, cy + radius, cz); // top pole: 0
  for (let i = 1; i < rings; i++) {
    const theta = (Math.PI * i) / rings;
    for (let j = 0; j < segments; j++) {
      const phi = (2 * Math.PI * j) / segments;
      positions.push(
        cx + radius * Math.sin(theta) * Math.cos(phi),
        cy + radius * Math.cos(theta),
        cz + radius * Math.sin(theta) * Math.sin(phi),
      );
    }
  }
  const bottom = positions.length / 3;
  positions.push(cx, cy - radius, cz);
  const ring = (i: number, j: number) => 1 + (i - 1) * segments + (j % segments);
  for (let j = 0; j < segments; j++) {
    indices.push(0, ring(1, j + 1), ring(1, j));
    indices.push(bottom, ring(rings - 1, j), ring(rings - 1, j + 1));
  }
  for (let i = 1; i < rings - 1; i++) {
    for (let j = 0; j < segments; j++) {
      const a = ring(i, j);
      const b = ring(i, j + 1);
      const c = ring(i + 1, j);
      const d = ring(i + 1, j + 1);
      indices.push(a, b, d, a, d, c);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

/** A closed torus around `center`, lying in the XZ plane. */
export function torus(center: Vec3, major: number, minor: number, segments = 64, sides = 24): MeshData {
  const positions: number[] = [];
  const indices: number[] = [];
  const [cx, cy, cz] = center;
  for (let i = 0; i < segments; i++) {
    const u = (2 * Math.PI * i) / segments;
    for (let j = 0; j < sides; j++) {
      const v = (2 * Math.PI * j) / sides;
      const r = major + minor * Math.cos(v);
      positions.push(cx + r * Math.cos(u), cy + minor * Math.sin(v), cz + r * Math.sin(u));
    }
  }
  const at = (i: number, j: number) => (i % segments) * sides + (j % sides);
  for (let i = 0; i < segments; i++) {
    for (let j = 0; j < sides; j++) {
      const a = at(i, j);
      const b = at(i + 1, j);
      const c = at(i, j + 1);
      const d = at(i + 1, j + 1);
      indices.push(a, c, d, a, d, b);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

/** An open square plate in the XZ plane with a sine wave in y. */
export function wavyPlate(center: Vec3, size: number, amplitude: number, n = 64): MeshData {
  const positions: number[] = [];
  const indices: number[] = [];
  const [cx, cy, cz] = center;
  for (let i = 0; i <= n; i++) {
    const z = size * (i / n - 0.5);
    for (let j = 0; j <= n; j++) {
      const x = size * (j / n - 0.5);
      const y = amplitude * Math.sin((3 * Math.PI * x) / size) * Math.cos((2 * Math.PI * z) / size);
      positions.push(cx + x, cy + y, cz + z);
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const a = i * (n + 1) + j;
      const b = a + 1;
      const c = a + n + 1;
      const d = c + 1;
      indices.push(a, c, d, a, d, b);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

/**
 * A binary STL file as a mesh, exact duplicate corners welded so normals are
 * smooth. `null` when the file is not binary STL (ASCII STL is not read).
 */
export function parseBinaryStl(buffer: ArrayBuffer): MeshData | null {
  if (buffer.byteLength < 84) {
    return null;
  }
  const data = new DataView(buffer);
  const count = data.getUint32(80, true);
  if (count === 0 || buffer.byteLength !== 84 + 50 * count) {
    return null;
  }
  const positions: number[] = [];
  const indices = new Uint32Array(3 * count);
  const ids = new Map<string, number>();
  for (let t = 0; t < count; t++) {
    for (let k = 0; k < 3; k++) {
      // 50 bytes per triangle: normal (12), three corners (36), attribute (2).
      const at = 84 + 50 * t + 12 + 12 * k;
      const x = data.getFloat32(at, true);
      const y = data.getFloat32(at + 4, true);
      const z = data.getFloat32(at + 8, true);
      const key = `${x},${y},${z}`;
      let id = ids.get(key);
      if (id === undefined) {
        id = positions.length / 3;
        ids.set(key, id);
        positions.push(x, y, z);
      }
      indices[3 * t + k] = id;
    }
  }
  return { positions: Float64Array.from(positions), indices };
}
