// The isosurface: marching tetrahedra over a regular volume. Each cube
// splits into the same 6 Freudenthal tetrahedra, so shared cube faces always
// agree (no cracks); crossings dedupe by lattice edge, and each triangle
// orients itself from inside to outside. Plus Otsu's threshold, a
// modality-agnostic default isovalue (CT and MR alike).
import type { MeshData } from "./mesh-data";

export interface LatticeGeometry {
  columns: number;
  rows: number;
  slices: number;
  origin: readonly [number, number, number];
  xStep?: readonly [number, number, number];
  yStep?: readonly [number, number, number];
  sliceStep?: readonly [number, number, number];
}

/** The 6 Freudenthal tetrahedra: cube nodes as x + 2y + 4z bits. */
const TETS: readonly (readonly [number, number, number, number])[] = [
  [0, 1, 3, 7],
  [0, 1, 5, 7],
  [0, 2, 3, 7],
  [0, 2, 6, 7],
  [0, 4, 5, 7],
  [0, 4, 6, 7],
];

/** A tetrahedron's 6 edges over its local vertices. */
const EDGES: readonly (readonly [number, number])[] = [
  [0, 1],
  [0, 2],
  [0, 3],
  [1, 2],
  [1, 3],
  [2, 3],
];

/**
 * Inside-mask (bit per local vertex) to triangles over edge crossings.
 * Topology only: each triangle orients itself in `emit`.
 */
const CASES: readonly (readonly (readonly [number, number, number])[])[] = [
  [],
  [[0, 1, 2]],
  [[0, 3, 4]],
  [
    [1, 3, 4],
    [1, 4, 2],
  ],
  [[1, 3, 5]],
  [
    [0, 3, 5],
    [0, 5, 2],
  ],
  [
    [0, 1, 5],
    [0, 5, 4],
  ],
  [[2, 5, 4]],
  [[2, 4, 5]],
  [
    [0, 4, 5],
    [0, 5, 1],
  ],
  [
    [0, 2, 5],
    [0, 5, 3],
  ],
  [[1, 5, 3]],
  [
    [1, 2, 4],
    [1, 4, 3],
  ],
  [[0, 4, 3]],
  [[0, 2, 1]],
  [],
];

/**
 * The `threshold` isosurface of slice-major `values` as an indexed mesh.
 * Inside is `value >= threshold` (`<=` when `flip`, for MONOCHROME1).
 */
export function isosurface(
  values: Float32Array,
  geometry: LatticeGeometry,
  threshold: number,
  flip: boolean,
): MeshData {
  const { columns: nx, rows: ny, slices: nz, origin } = geometry;
  const xs = geometry.xStep ?? [1, 0, 0];
  const ys = geometry.yStep ?? [0, 1, 0];
  const zs = geometry.sliceStep ?? [0, 0, 1];
  const inside = (v: number): boolean => (flip ? v <= threshold : v >= threshold);
  const positions: number[] = [];
  const indices: number[] = [];
  /** Crossing vertices by lattice edge (nested: ids overflow a packed key at CT scale). */
  const verts = new Map<number, Map<number, number>>();
  const node = (x: number, y: number, z: number): number => z * ny * nx + y * nx + x;
  const point = (id: number, out: [number, number, number]): void => {
    const x = id % nx;
    const y = Math.floor(id / nx) % ny;
    const z = Math.floor(id / (nx * ny));
    out[0] = origin[0] + x * xs[0] + y * ys[0] + z * zs[0];
    out[1] = origin[1] + x * xs[1] + y * ys[1] + z * zs[1];
    out[2] = origin[2] + x * xs[2] + y * ys[2] + z * zs[2];
  };
  const pa: [number, number, number] = [0, 0, 0];
  const pb: [number, number, number] = [0, 0, 0];
  const cross = (a: number, b: number): number => {
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    let inner = verts.get(lo);
    if (inner === undefined) {
      inner = new Map();
      verts.set(lo, inner);
    }
    const known = inner.get(hi);
    if (known !== undefined) {
      return known;
    }
    const va = values[a];
    const vb = values[b];
    point(a, pa);
    point(b, pb);
    let t = (threshold - va) / (vb - va);
    if (!(t > 0)) {
      t = 0;
    } else if (!(t < 1)) {
      t = 1;
    }
    const id = positions.length / 3;
    positions.push(pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]), pa[2] + t * (pb[2] - pa[2]));
    inner.set(hi, id);
    return id;
  };
  const emit = (corners: readonly [number, number, number, number], tri: readonly [number, number, number]): void => {
    const [ia, ib, ic] = [cross(corners[EDGES[tri[0]][0]], corners[EDGES[tri[0]][1]]), cross(corners[EDGES[tri[1]][0]], corners[EDGES[tri[1]][1]]), cross(corners[EDGES[tri[2]][0]], corners[EDGES[tri[2]][1]])];
    if (ia === ib || ib === ic || ic === ia) {
      return; // snapped onto shared nodes (values exactly at the threshold)
    }
    // Orient from inside to outside: a known-inside reference must lie
    // behind the triangle.
    let rx = 0;
    let ry = 0;
    let rz = 0;
    let n = 0;
    for (const c of corners) {
      if (inside(values[c])) {
        point(c, pa);
        rx += pa[0];
        ry += pa[1];
        rz += pa[2];
        n++;
      }
    }
    const ax = positions[3 * ia];
    const ay = positions[3 * ia + 1];
    const az = positions[3 * ia + 2];
    const abx = positions[3 * ib] - ax;
    const aby = positions[3 * ib + 1] - ay;
    const abz = positions[3 * ib + 2] - az;
    const acx = positions[3 * ic] - ax;
    const acy = positions[3 * ic + 1] - ay;
    const acz = positions[3 * ic + 2] - az;
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    const cx = (ax + positions[3 * ib] + positions[3 * ic]) / 3 - rx / n;
    const cy = (ay + positions[3 * ib + 1] + positions[3 * ic + 1]) / 3 - ry / n;
    const cz = (az + positions[3 * ib + 2] + positions[3 * ic + 2]) / 3 - rz / n;
    if (nx * cx + ny * cy + nz * cz < 0) {
      indices.push(ia, ic, ib);
    } else {
      indices.push(ia, ib, ic);
    }
  };
  for (let z = 0; z + 1 < nz; z++) {
    for (let y = 0; y + 1 < ny; y++) {
      for (let x = 0; x + 1 < nx; x++) {
        const cube = [0, 0, 0, 0, 0, 0, 0, 0];
        for (let b = 0; b < 8; b++) {
          cube[b] = node(x + (b & 1), y + ((b >> 1) & 1), z + ((b >> 2) & 1));
        }
        for (const tet of TETS) {
          const corners = [cube[tet[0]], cube[tet[1]], cube[tet[2]], cube[tet[3]]] as const;
          let mask = 0;
          for (let v = 0; v < 4; v++) {
            if (inside(values[corners[v]])) {
              mask |= 1 << v;
            }
          }
          for (const tri of CASES[mask]) {
            emit(corners, tri);
          }
        }
      }
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

/** Otsu's threshold: the bin splitting the histogram into the two tightest classes. */
export function otsu(values: Float32Array, bins = 256): number {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) {
      min = v;
    }
    if (v > max) {
      max = v;
    }
  }
  if (!(min < max)) {
    return min === Infinity ? 0 : min;
  }
  const hist = new Float64Array(bins);
  const width = (max - min) / bins;
  for (const v of values) {
    hist[Math.min(bins - 1, Math.floor((v - min) / width))]++;
  }
  let best = 0;
  let bestVar = -1;
  let w0 = 0;
  let sum0 = 0;
  const total = values.length;
  const sum = hist.reduce((a, h, i) => a + h * (i + 0.5), 0);
  for (let t = 0; t < bins; t++) {
    w0 += hist[t];
    sum0 += hist[t] * (t + 0.5);
    const w1 = total - w0;
    if (w0 === 0 || w1 === 0) {
      continue;
    }
    const gap = sum0 / w0 - (sum - sum0) / w1;
    const between = w0 * w1 * gap * gap;
    if (between > bestVar) {
      bestVar = between;
      best = t;
    }
  }
  return min + (best + 1) * width;
}
