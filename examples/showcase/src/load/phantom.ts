// The DICOM phantoms: vessel tree, brain and heart SDFs sampled into
// slice stacks. They mesh through the same extraction pipeline as dropped
// files, so the demos prove the path. All deterministic (seeded), millimetre
// voxels, signed 16-bit samples of the SDF (isovalue 0).
import type { DicomSlice } from "./dicom";

export type PhantomKind = "vessels" | "brain" | "heart";

/** Display names for the demo buttons and job. */
export const PHANTOM_NAMES: Record<PhantomKind, string> = {
  vessels: "Vessel tree",
  brain: "Brain",
  heart: "Heart",
};

type V3 = [number, number, number];

/** Deterministic RNG (mulberry32). */
function rng(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Value noise in [0, 1]: hashed lattice, smootherstep, trilinear. */
function makeNoise(seed: number): (x: number, y: number, z: number) => number {
  const hash = (x: number, y: number, z: number): number => {
    let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(z, 1440662683) + Math.imul(seed, 974634211)) | 0;
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  };
  return (x, y, z) => {
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const zi = Math.floor(z);
    const xf = x - xi;
    const yf = y - yi;
    const zf = z - zi;
    const s = (t: number): number => t * t * t * (t * (t * 6 - 15) + 10);
    const u = s(xf);
    const v = s(yf);
    const w = s(zf);
    let acc = 0;
    for (let dz = 0; dz < 2; dz++) {
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const weight = (dx === 0 ? 1 - u : u) * (dy === 0 ? 1 - v : v) * (dz === 0 ? 1 - w : w);
          acc += weight * hash(xi + dx, yi + dy, zi + dz);
        }
      }
    }
    return acc;
  };
}

/** A millimetre SDF grid accumulated by primitives, each over its own box. */
class Grid {
  readonly data: Float32Array;
  constructor(
    readonly nx: number,
    readonly ny: number,
    readonly nz: number,
  ) {
    this.data = new Float32Array(nx * ny * nz).fill(Infinity);
  }

  /** Merges `sdf` over `box` ([[x0,x1],[y0,y1],[z0,z1]], millimetres, clamped). */
  add(box: V3, half: V3, sdf: (x: number, y: number, z: number) => number): void {
    const x0 = Math.max(0, Math.floor(box[0] - half[0]));
    const x1 = Math.min(this.nx - 1, Math.ceil(box[0] + half[0]));
    const y0 = Math.max(0, Math.floor(box[1] - half[1]));
    const y1 = Math.min(this.ny - 1, Math.ceil(box[1] + half[1]));
    const z0 = Math.max(0, Math.floor(box[2] - half[2]));
    const z1 = Math.min(this.nz - 1, Math.ceil(box[2] + half[2]));
    for (let z = z0; z <= z1; z++) {
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const at = z * this.ny * this.nx + y * this.nx + x;
          const d = sdf(x, y, z);
          if (d < this.data[at]) {
            this.data[at] = d;
          }
        }
      }
    }
  }
}

function capsule(ax: number, ay: number, az: number, bx: number, by: number, bz: number, r: number): {
  box: V3;
  half: V3;
  sdf: (x: number, y: number, z: number) => number;
} {
  const abx = bx - ax;
  const aby = by - ay;
  const abz = bz - az;
  const len2 = abx * abx + aby * aby + abz * abz;
  return {
    box: [(ax + bx) / 2, (ay + by) / 2, (az + bz) / 2],
    half: [Math.abs(abx) / 2 + r + 1, Math.abs(aby) / 2 + r + 1, Math.abs(abz) / 2 + r + 1],
    sdf: (x, y, z) => {
      const t = len2 === 0 ? 0 : Math.min(1, Math.max(0, ((x - ax) * abx + (y - ay) * aby + (z - az) * abz) / len2));
      return Math.hypot(x - (ax + t * abx), y - (ay + t * aby), z - (az + t * abz)) - r;
    },
  };
}

function ellipsoid(cx: number, cy: number, cz: number, rx: number, ry: number, rz: number): {
  box: V3;
  half: V3;
  sdf: (x: number, y: number, z: number) => number;
} {
  return {
    box: [cx, cy, cz],
    half: [rx + 1, ry + 1, rz + 1],
    sdf: (x, y, z) => {
      const qx = (x - cx) / rx;
      const qy = (y - cy) / ry;
      const qz = (z - cz) / rz;
      const k0 = Math.hypot(qx, qy, qz);
      if (k0 === 0) {
        return -Math.min(rx, ry, rz);
      }
      const k1 = Math.hypot(qx / rx, qy / ry, qz / rz);
      return (k0 * (k0 - 1)) / k1;
    },
  };
}

interface Segment {
  a: V3;
  b: V3;
  r: number;
}

/** A deterministic bifurcating tree (Murray-ish radii), millimetres in a 96^3 box. */
function vesselTree(): Segment[] {
  const rand = rng(20261027);
  const segs: Segment[] = [];
  const grow = (p: V3, dir: V3, len: number, r: number, depth: number): void => {
    const b: V3 = [p[0] + dir[0] * len, p[1] + dir[1] * len, p[2] + dir[2] * len];
    segs.push({ a: p, b, r });
    if (depth === 0 || r < 0.9) {
      return;
    }
    // Two children rotated apart around a random perpendicular axis pair.
    const up: V3 = Math.abs(dir[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
    const u: V3 = [
      dir[1] * up[2] - dir[2] * up[1],
      dir[2] * up[0] - dir[0] * up[2],
      dir[0] * up[1] - dir[1] * up[0],
    ];
    const ul = Math.hypot(u[0], u[1], u[2]);
    u[0] /= ul;
    u[1] /= ul;
    u[2] /= ul;
    const v: V3 = [dir[1] * u[2] - dir[2] * u[1], dir[2] * u[0] - dir[0] * u[2], dir[0] * u[1] - dir[1] * u[0]];
    const spread = 0.45 + rand() * 0.25;
    const roll = rand() * Math.PI * 2;
    const c = Math.cos(roll);
    const s = Math.sin(roll);
    const w1: V3 = [u[0] * c + v[0] * s, u[1] * c + v[1] * s, u[2] * c + v[2] * s];
    const tilt = (w: V3, sign: number): V3 => {
      const d: V3 = [dir[0] + sign * spread * w[0], dir[1] + sign * spread * w[1], dir[2] + sign * spread * w[2]];
      const l = Math.hypot(d[0], d[1], d[2]);
      return [d[0] / l, d[1] / l, d[2] / l];
    };
    const j = () => 0.94 + rand() * 0.12;
    grow(b, tilt(w1, 1), len * 0.76 * j(), r * 0.68 * j(), depth - 1);
    grow(b, tilt(w1, -1), len * 0.76 * j(), r * 0.6 * j(), depth - 1);
  };
  grow([48, 48, 6], [0.06, -0.04, 1], 18, 5.5, 7);
  return segs;
}

const PHANTOMS: Record<PhantomKind, { name: string; uid: string; dims: [number, number, number]; build: (g: Grid) => void }> = {
  vessels: {
    name: "Vessel tree",
    uid: "1.9.9.1",
    dims: [96, 96, 96],
    build(g) {
      for (const s of vesselTree()) {
        const c = capsule(s.a[0], s.a[1], s.a[2], s.b[0], s.b[1], s.b[2], s.r);
        g.add(c.box, c.half, c.sdf);
      }
    },
  },
  brain: {
    name: "Brain",
    uid: "1.9.9.2",
    dims: [96, 112, 80],
    build(g) {
      // Domain-warped hemispheres: the warp reads as gyri and sulci.
      const n1 = makeNoise(11);
      const n2 = makeNoise(23);
      const n3 = makeNoise(37);
      const warp = (x: number, y: number, z: number): V3 => {
        const s = 1 / 14;
        return [
          x - 5 * (n1(x * s, y * s, z * s) - 0.5),
          y - 5 * (n2(x * s, y * s, z * s) - 0.5),
          z - 5 * (n3(x * s, y * s, z * s) - 0.5),
        ];
      };
      // Half-space-clipped ellipsoids: a 2 mm longitudinal fissure.
      const left = ellipsoid(36, 58, 42, 20, 27, 25);
      const right = ellipsoid(60, 58, 42, 20, 27, 25);
      g.add([48, 58, 42], [30, 32, 30], (x, y, z) => {
        const [qx, qy, qz] = warp(x, y, z);
        const dl = Math.max(left.sdf(qx, qy, qz), qx - 47);
        const dr = Math.max(right.sdf(qx, qy, qz), 49 - qx);
        return Math.min(dl, dr);
      });
      const cere = ellipsoid(48, 24, 24, 15, 13, 11);
      g.add(cere.box, cere.half, (x, y, z) => {
        const [qx, qy, qz] = warp(x, y, z);
        return cere.sdf(qx, qy, qz);
      });
      const stem = capsule(48, 32, 20, 48, 28, 4, 5);
      g.add(stem.box, stem.half, stem.sdf);
    },
  },
  heart: {
    name: "Heart",
    uid: "1.9.9.3",
    dims: [88, 88, 88],
    build(g) {
      const parts = [
        ellipsoid(38, 38, 44, 13, 17, 15), // LV
        ellipsoid(53, 34, 50, 9, 13, 11), // RV
        ellipsoid(36, 52, 36, 9, 9, 9), // LA
        ellipsoid(55, 50, 38, 8.5, 8.5, 8.5), // RA
        capsule(40, 42, 56, 40, 42, 70, 6.5), // ascending aorta
        capsule(40, 42, 70, 33, 37, 71, 6), // arch
        capsule(33, 37, 71, 28, 33, 68, 5.5), // arch
        capsule(28, 33, 68, 28, 30, 52, 5.5), // descending aorta
        capsule(49, 40, 60, 49, 40, 70, 5.5), // pulmonary trunk
        capsule(49, 40, 70, 40, 38, 74, 3.5), // left pulmonary
        capsule(49, 40, 70, 58, 38, 74, 3.5), // right pulmonary
        capsule(55, 48, 52, 55, 48, 64, 4.5), // SVC stub
      ];
      // One box over the union: smooth-min blending needs every part per voxel.
      const k = 2.5;
      const smin = (a: number, b: number): number => {
        const h = Math.min(1, Math.max(0, 0.5 + ((b - a) / k) * 0.5));
        return b * (1 - h) + a * h - k * h * (1 - h);
      };
      g.add([44, 44, 44], [34, 34, 34], (x, y, z) => {
        // From the first part (Infinity would NaN the blend: Inf * 0).
        let d = parts[0].sdf(x, y, z);
        for (let i = 1; i < parts.length; i++) {
          d = smin(d, parts[i].sdf(x, y, z));
        }
        return d;
      });
    },
  },
};

/** The phantom's slices (axial, 1 mm, signed SDF samples × 100). */
export function phantomSlices(kind: PhantomKind): DicomSlice[] {
  const def = PHANTOMS[kind];
  const [nx, ny, nz] = def.dims;
  const g = new Grid(nx, ny, nz);
  def.build(g);
  const out: DicomSlice[] = [];
  for (let z = 0; z < nz; z++) {
    const pixels = new Int16Array(nx * ny);
    for (let i = 0; i < nx * ny; i++) {
      const sdf = g.data[z * ny * nx + i];
      pixels[i] = Math.min(32767, Math.max(-32768, Math.round((Number.isFinite(sdf) ? sdf : 1000) * 100)));
    }
    out.push({
      seriesUid: def.uid,
      instanceUid: `${def.uid}.${z}`,
      description: `${def.name} (DICOM demo)`,
      rows: ny,
      columns: nx,
      bitsAllocated: 16,
      pixelRepresentation: 1,
      rescaleSlope: 0.01,
      rescaleIntercept: 0,
      pixelSpacing: [1, 1],
      position: [0, 0, z],
      orientation: [1, 0, 0, 0, 1, 0],
      pixels,
      inverted: false,
    });
  }
  return out;
}
