// The demo meshes loaded when the app opens (spec §6.8), generated in the
// loader Worker so a `?big` scene never blocks the page:
// - A: a closed blobby sphere;
// - B: a closed rounded box that penetrates A;
// - C: a sphere shell with two holes and a floating piece, for the repair panel.
// `?big` makes A and B about a million triangles each.
import type { MeshData, Vec3 } from "../load/mesh-data";

export interface DemoMesh {
  name: string;
  data: MeshData;
}

/** A closed UV sphere around `center` (y up, poles shared), its radius varied smoothly with direction. */
export function blobbySphere(center: Vec3, radius: number, segments: number, rings: number, bump = 0.12): MeshData {
  const r = (x: number, y: number, z: number) =>
    radius * (1 + bump * (Math.sin(4 * x + 0.5) * Math.cos(3 * y) + 0.7 * Math.sin(5 * z + 2 * x)));
  return uvSphere(center, segments, rings, r);
}

/**
 * A UV sphere: `radiusAt(dx, dy, dz)` gives the radius along each unit
 * direction. Counter-clockwise outward triangles.
 */
function uvSphere(
  center: Vec3,
  segments: number,
  rings: number,
  radiusAt: (x: number, y: number, z: number) => number,
): MeshData {
  const [cx, cy, cz] = center;
  const vertices = 2 + segments * (rings - 1);
  const positions = new Float64Array(3 * vertices);
  const put = (v: number, x: number, y: number, z: number) => {
    const s = radiusAt(x, y, z);
    positions[3 * v] = cx + s * x;
    positions[3 * v + 1] = cy + s * y;
    positions[3 * v + 2] = cz + s * z;
  };
  put(0, 0, 1, 0); // top pole
  for (let i = 1; i < rings; i++) {
    const theta = (Math.PI * i) / rings;
    for (let j = 0; j < segments; j++) {
      const phi = (2 * Math.PI * j) / segments;
      put(1 + (i - 1) * segments + j, Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi));
    }
  }
  const bottom = vertices - 1;
  put(bottom, 0, -1, 0);
  const ring = (i: number, j: number) => 1 + (i - 1) * segments + (j % segments);
  const indices = new Uint32Array(3 * 2 * segments * (rings - 1));
  let t = 0;
  const tri = (a: number, b: number, c: number) => {
    indices[t++] = a;
    indices[t++] = b;
    indices[t++] = c;
  };
  for (let j = 0; j < segments; j++) {
    tri(0, ring(1, j + 1), ring(1, j));
    tri(bottom, ring(rings - 1, j), ring(rings - 1, j + 1));
  }
  for (let i = 1; i < rings - 1; i++) {
    for (let j = 0; j < segments; j++) {
      const a = ring(i, j);
      const b = ring(i, j + 1);
      const c = ring(i + 1, j);
      const d = ring(i + 1, j + 1);
      tri(a, b, d);
      tri(a, d, c);
    }
  }
  return { positions, indices };
}

/**
 * A closed box with rounded edges and corners: half-extents `half`, edge
 * radius `round` (below every half-extent), from a cube whose faces are
 * `n` x `n` grids. Grid points are pushed onto the rounded surface.
 */
export function roundedBox(center: Vec3, half: Vec3, round: number, n: number): MeshData {
  const side = n + 1;
  const ids = new Map<number, number>();
  const positions: number[] = [];
  const inner = half.map((h) => h - round);
  const vertex = (l: Vec3): number => {
    const key = (l[0] * side + l[1]) * side + l[2];
    let id = ids.get(key);
    if (id === undefined) {
      id = positions.length / 3;
      ids.set(key, id);
      const p = l.map((x, a) => (2 * x / n - 1) * half[a]);
      const q = p.map((x, a) => Math.min(Math.max(x, -inner[a]), inner[a]));
      const d = p.map((x, a) => x - q[a]);
      const len = Math.hypot(d[0], d[1], d[2]);
      for (let a = 0; a < 3; a++) {
        positions.push(center[a] + (len > 0 ? q[a] + (d[a] / len) * round : p[a]));
      }
    }
    return id;
  };
  const indices: number[] = [];
  for (let a = 0; a < 3; a++) {
    const b = (a + 1) % 3;
    const c = (a + 2) % 3;
    for (const s of [0, n]) {
      for (let u = 0; u < n; u++) {
        for (let v = 0; v < n; v++) {
          const at = (du: number, dv: number) => {
            const l: Vec3 = [0, 0, 0];
            l[a] = s;
            l[b] = u + du;
            l[c] = v + dv;
            return vertex(l);
          };
          const p00 = at(0, 0);
          const p10 = at(1, 0);
          const p11 = at(1, 1);
          const p01 = at(0, 1);
          // e_b x e_c = e_a: counter-clockwise on the +a side, reversed on the -a side.
          if (s === n) {
            indices.push(p00, p10, p11, p00, p11, p01);
          } else {
            indices.push(p00, p11, p10, p00, p01, p11);
          }
        }
      }
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

/**
 * A sphere shell with a large hole (its top cap removed) and a small one (one
 * quad removed on its side), plus a small closed sphere floating above it.
 */
export function openShellWithPiece(center: Vec3, radius: number, segments = 64, rings = 32): MeshData {
  const shell = uvSphere(center, segments, rings, () => radius);
  // Drop the triangles touching the top pole or the first ring (a cap), and
  // the quad (two triangles) at ring rings/2, segment 0: `uvSphere` writes
  // the 2 pole fans first (2 * segments triangles), then 2 per quad, ring by ring.
  const quad = 2 * segments + 2 * segments * (Math.floor(rings / 2) - 1);
  const kept: number[] = [];
  for (let t = 0; t < shell.indices.length / 3; t++) {
    const tri = [shell.indices[3 * t], shell.indices[3 * t + 1], shell.indices[3 * t + 2]];
    if (t !== quad && t !== quad + 1 && tri.every((i) => i > segments)) {
      kept.push(...tri);
    }
  }
  const holed = compact(shell.positions, kept);
  const piece = uvSphere([center[0], center[1] + 1.6 * radius, center[2]], 16, 8, () => 0.2 * radius);
  return merge(holed, piece);
}

/** Drops the vertices no triangle uses. */
function compact(positions: Float64Array, indices: number[]): MeshData {
  const map = new Int32Array(positions.length / 3).fill(-1);
  const out: number[] = [];
  const idx = new Uint32Array(indices.length);
  indices.forEach((i, k) => {
    if (map[i] === -1) {
      map[i] = out.length / 3;
      out.push(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);
    }
    idx[k] = map[i];
  });
  return { positions: Float64Array.from(out), indices: idx };
}

function merge(a: MeshData, b: MeshData): MeshData {
  const positions = new Float64Array(a.positions.length + b.positions.length);
  positions.set(a.positions);
  positions.set(b.positions, a.positions.length);
  const indices = new Uint32Array(a.indices.length + b.indices.length);
  indices.set(a.indices);
  const offset = a.positions.length / 3;
  for (let i = 0; i < b.indices.length; i++) {
    indices[a.indices.length + i] = b.indices[i] + offset;
  }
  return { positions, indices };
}

/** A, B and C; `big` makes A and B about a million triangles each. */
export function demoMeshes(big: boolean): DemoMesh[] {
  return [
    { name: "A · blobby sphere", data: blobbySphere([0, 1, 0], 1, big ? 1024 : 128, big ? 512 : 64) },
    { name: "B · rounded box", data: roundedBox([1.05, 1, 0.1], [0.55, 0.55, 0.55], 0.15, big ? 290 : 32) },
    { name: "C · open shell + piece", data: openShellWithPiece([-2.8, 1, 0], 0.8) },
  ];
}
