// Mesh generators and the binary STL reader, as in the WebGPU example
// (`examples/webgpu/src/meshes.ts`): world positions in f64, Uint32 indices,
// counter-clockwise outward triangles. Plus `fit`, which scales a dropped
// STL to the size and place of the mesh it replaces.

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

/** An open square plate in the XZ plane, displaced in y by a sine wave. */
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

/** The centre and radius (half the diagonal) of the bounding box of `positions`. */
export function bounds(positions: Float64Array): { center: Vec3; radius: number } {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i++) {
    min[i % 3] = Math.min(min[i % 3], positions[i]);
    max[i % 3] = Math.max(max[i % 3], positions[i]);
  }
  return {
    center: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2],
    radius: Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2,
  };
}

/** `data` scaled and moved so its bounding box has `center` and `radius`. */
export function fit(data: MeshData, center: Vec3, radius: number): MeshData {
  const from = bounds(data.positions);
  const s = from.radius > 0 ? radius / from.radius : 1;
  const positions = new Float64Array(data.positions.length);
  for (let i = 0; i < positions.length; i++) {
    const k = i % 3;
    positions[i] = center[k] + s * (data.positions[i] - from.center[k]);
  }
  return { positions, indices: data.indices };
}
