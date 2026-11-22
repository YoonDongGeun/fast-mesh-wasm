// The FFD lattice math behind the deform cage (#193): control point indexing,
// the rest and displaced control point positions (matching the engine's
// `axis_param`), the cage wireframe, and immutable control point edits.
import type { FfdBasis, FfdFrameData, FfdLatticeData } from "fast-mesh-wasm";

export type Vec3Tuple = [number, number, number];

/** Returns the row-major flat index `(i * ny + j) * nz + k` for FFD control point `(i, j, k)`. */
export function ffdControlPointIndex(
  dims: readonly [number, number, number],
  i: number,
  j: number,
  k: number,
): number {
  const [nx, ny, nz] = dims;
  if (
    !Number.isInteger(nx) ||
    !Number.isInteger(ny) ||
    !Number.isInteger(nz) ||
    nx < 2 ||
    ny < 2 ||
    nz < 2
  ) {
    throw new Error(`Invalid FFD lattice dimensions [${nx}, ${ny}, ${nz}]; each axis must be an integer >= 2.`);
  }
  if (
    !Number.isInteger(i) ||
    !Number.isInteger(j) ||
    !Number.isInteger(k) ||
    i < 0 ||
    i >= nx ||
    j < 0 ||
    j >= ny ||
    k < 0 ||
    k >= nz
  ) {
    throw new Error(`FFD control point (${i}, ${j}, ${k}) out of bounds for dims [${nx}, ${ny}, ${nz}].`);
  }
  return (i * ny + j) * nz + k;
}

function bsplineKnot(n: number, p: number, idx: number): number {
  if (idx <= p) return 0;
  if (idx >= n) return 1;
  return (idx - p) / (n - p);
}

/** Normalized lattice parameter in `[0, 1]` matching Rust `axis_param` for Bernstein and Cubic B-Spline. */
export function axisParam(i: number, n: number, basis: FfdBasis): number {
  if (basis === "cubicBSpline") {
    const p = Math.min(n - 1, 3);
    let sum = 0;
    for (let offset = 1; offset <= p; offset++) {
      sum += bsplineKnot(n, p, i + offset);
    }
    return sum / p;
  }
  return i / (n - 1);
}

function paramToLocal(frame: FfdFrameData, uvw: readonly [number, number, number]): Vec3Tuple {
  let x = frame.center[0];
  let y = frame.center[1];
  let z = frame.center[2];
  for (let a = 0; a < 3; a++) {
    const s = (2 * uvw[a] - 1) * frame.halfExtents[a];
    const axis = frame.axes[a];
    x += s * axis[0];
    y += s * axis[1];
    z += s * axis[2];
  }
  return [x, y, z];
}

/** Computes displaced control point positions in mesh-local space in row-major order. */
export function computeLatticeControlPoints(lattice: FfdLatticeData): Vec3Tuple[] {
  const [nx, ny, nz] = lattice.dims;
  const total = nx * ny * nz;
  if (lattice.offsets.length < total * 3) {
    throw new Error(
      `FFD lattice offsets length ${lattice.offsets.length} is shorter than required ${total * 3}.`,
    );
  }
  const points: Vec3Tuple[] = new Array(total);
  for (let i = 0; i < nx; i++) {
    const u = axisParam(i, nx, lattice.basis);
    for (let j = 0; j < ny; j++) {
      const v = axisParam(j, ny, lattice.basis);
      for (let k = 0; k < nz; k++) {
        const w = axisParam(k, nz, lattice.basis);
        const idx = (i * ny + j) * nz + k;
        const rest = paramToLocal(lattice.frame, [u, v, w]);
        const base = idx * 3;
        points[idx] = [
          rest[0] + lattice.offsets[base],
          rest[1] + lattice.offsets[base + 1],
          rest[2] + lattice.offsets[base + 2],
        ];
      }
    }
  }
  return points;
}

/** Builds flat line-segment coordinates (`[x0,y0,z0, x1,y1,z1, ...]`) for the FFD cage wireframe. */
export function buildLatticeEdgeSegments(lattice: FfdLatticeData): Float32Array {
  const [nx, ny, nz] = lattice.dims;
  const points = computeLatticeControlPoints(lattice);
  const segCount = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1);
  const out = new Float32Array(segCount * 6);
  let cursor = 0;

  const pushSeg = (a: Vec3Tuple, b: Vec3Tuple) => {
    out[cursor++] = a[0];
    out[cursor++] = a[1];
    out[cursor++] = a[2];
    out[cursor++] = b[0];
    out[cursor++] = b[1];
    out[cursor++] = b[2];
  };

  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      for (let k = 0; k < nz; k++) {
        const idx = (i * ny + j) * nz + k;
        const p = points[idx];
        if (i + 1 < nx) {
          pushSeg(p, points[((i + 1) * ny + j) * nz + k]);
        }
        if (j + 1 < ny) {
          pushSeg(p, points[(i * ny + (j + 1)) * nz + k]);
        }
        if (k + 1 < nz) {
          pushSeg(p, points[(i * ny + j) * nz + (k + 1)]);
        }
      }
    }
  }
  return out;
}

/** Clones an `FfdLatticeData` object with a fresh `Float64Array` for offsets. */
export function cloneLattice(lattice: FfdLatticeData): FfdLatticeData {
  return {
    dims: [lattice.dims[0], lattice.dims[1], lattice.dims[2]],
    basis: lattice.basis,
    frame: {
      center: [lattice.frame.center[0], lattice.frame.center[1], lattice.frame.center[2]],
      axes: [
        [lattice.frame.axes[0][0], lattice.frame.axes[0][1], lattice.frame.axes[0][2]],
        [lattice.frame.axes[1][0], lattice.frame.axes[1][1], lattice.frame.axes[1][2]],
        [lattice.frame.axes[2][0], lattice.frame.axes[2][1], lattice.frame.axes[2][2]],
      ],
      halfExtents: [
        lattice.frame.halfExtents[0],
        lattice.frame.halfExtents[1],
        lattice.frame.halfExtents[2],
      ],
    },
    offsets: new Float64Array(lattice.offsets),
  };
}

/**
 * Pure immutable update of one control point offset by flat index or `(i, j, k)`.
 * `"delta"` adds `vector` to the existing offset; `"set"` replaces it.
 */
export function applyDisplacementToLattice(
  lattice: FfdLatticeData,
  indexOrIjk: number | readonly [number, number, number],
  vector: readonly [number, number, number],
  mode: "delta" | "set" = "delta",
): FfdLatticeData {
  if (!Number.isFinite(vector[0]) || !Number.isFinite(vector[1]) || !Number.isFinite(vector[2])) {
    throw new Error("FFD control point displacement vector must contain finite numbers.");
  }
  const [nx, ny, nz] = lattice.dims;
  const total = nx * ny * nz;
  const idx =
    typeof indexOrIjk === "number"
      ? indexOrIjk
      : ffdControlPointIndex(lattice.dims, indexOrIjk[0], indexOrIjk[1], indexOrIjk[2]);
  if (!Number.isInteger(idx) || idx < 0 || idx >= total) {
    throw new Error(`FFD control point flat index ${idx} out of bounds [0, ${total}).`);
  }
  const next = cloneLattice(lattice);
  const base = idx * 3;
  if (mode === "set") {
    next.offsets[base] = vector[0];
    next.offsets[base + 1] = vector[1];
    next.offsets[base + 2] = vector[2];
  } else {
    next.offsets[base] += vector[0];
    next.offsets[base + 1] += vector[1];
    next.offsets[base + 2] += vector[2];
  }
  return next;
}

/** Pure helper moving FFD control point `(i, j, k)` by delta or absolute offset. */
export function moveFfdControlPoint(
  lattice: FfdLatticeData,
  i: number,
  j: number,
  k: number,
  offsetOrDelta: readonly [number, number, number],
  mode: "delta" | "set" = "delta",
): FfdLatticeData {
  const idx = ffdControlPointIndex(lattice.dims, i, j, k);
  return applyDisplacementToLattice(lattice, idx, offsetOrDelta, mode);
}

/** Returns a clone of `lattice` with all control point offsets cleared to zero. */
export function resetLatticeOffsets(lattice: FfdLatticeData): FfdLatticeData {
  const next = cloneLattice(lattice);
  next.offsets.fill(0);
  return next;
}
