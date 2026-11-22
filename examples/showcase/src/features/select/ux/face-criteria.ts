// Face selection by criteria (#258): the per-triangle tests that run on the
// main thread. Each walks a snapshot of the mesh's buffers in slices and
// yields between them once a slice has used its time budget, so a mesh of a
// million triangles stays interactive while it runs. The edge tests share
// one adjacency: per triangle edge the other triangle on it, built from the
// vertex fans in linear time. Results are ascending triangle slots. Pure,
// for tests (the pause is injectable).
import type { Vec3 } from "fast-mesh-wasm";

/** Waits for the event loop, between slices. */
export type Pause = () => Promise<void>;

/** A macrotask: input and rendering get their turn before the next slice. */
export const yieldToEventLoop: Pause = () => new Promise((resolve) => setTimeout(resolve, 0));

/** How long one slice runs before it yields, in milliseconds. */
const SLICE_BUDGET_MS = 8;
/** The clock is read once per this many steps. */
const STEPS_PER_CHECK = 4096;

/** Counts work and says when the slice's budget is used. */
class Slicer {
  readonly #pause: Pause;
  #start = performance.now();
  #steps = 0;

  constructor(pause: Pause) {
    this.#pause = pause;
  }

  /** Counts one step; true when the slice should yield now. */
  due(): boolean {
    if (++this.#steps < STEPS_PER_CHECK) {
      return false;
    }
    this.#steps = 0;
    return performance.now() - this.#start >= SLICE_BUDGET_MS;
  }

  async pause(): Promise<void> {
    await this.#pause();
    this.#start = performance.now();
  }
}

/** A snapshot of a mesh's render buffers and its world matrix. */
export interface MeshGeometry {
  /** Local render coordinates, 3 per vertex slot. */
  positions: Float32Array;
  /** 3 per triangle slot; a dead slot reads `[0, 0, 0]`. */
  indices: Uint32Array;
  /** The column-major 4x4 world matrix of the local render coordinates. */
  matrix: ArrayLike<number>;
}

/** The identity world matrix, column-major. */
export const IDENTITY_MATRIX: readonly number[] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function triangleSlots(geom: MeshGeometry): number {
  return Math.floor(geom.indices.length / 3);
}

/** Whether triangle `t` is live with three distinct vertex ids that exist (the engine's "proper" triangles). */
function isProper(indices: Uint32Array, vertexSlots: number, t: number): boolean {
  const a = indices[3 * t];
  const b = indices[3 * t + 1];
  const c = indices[3 * t + 2];
  return a !== b && b !== c && a !== c && a < vertexSlots && b < vertexSlots && c < vertexSlots;
}

/** The ascending slots whose flag is set. */
function maskToSlots(mask: Uint8Array): Uint32Array {
  let n = 0;
  for (let t = 0; t < mask.length; t++) {
    n += mask[t];
  }
  const out = new Uint32Array(n);
  let k = 0;
  for (let t = 0; t < mask.length && k < n; t++) {
    if (mask[t] !== 0) {
      out[k++] = t;
    }
  }
  return out;
}

/**
 * Reads the world corners of every proper triangle into `scratch` (9
 * numbers) and calls `visit(t, scratch)`. Improper and dead slots are
 * skipped.
 */
async function forEachWorldTriangle(
  geom: MeshGeometry,
  pause: Pause,
  visit: (t: number, corners: Float64Array) => void,
): Promise<void> {
  const { positions, indices, matrix: m } = geom;
  const vertexSlots = Math.floor(positions.length / 3);
  const total = triangleSlots(geom);
  const corners = new Float64Array(9);
  const m0 = m[0], m1 = m[1], m2 = m[2];
  const m4 = m[4], m5 = m[5], m6 = m[6];
  const m8 = m[8], m9 = m[9], m10 = m[10];
  const m12 = m[12], m13 = m[13], m14 = m[14];
  const slicer = new Slicer(pause);
  for (let t = 0; t < total; t++) {
    if (isProper(indices, vertexSlots, t)) {
      for (let k = 0; k < 3; k++) {
        const o = indices[3 * t + k] * 3;
        const x = positions[o];
        const y = positions[o + 1];
        const z = positions[o + 2];
        corners[3 * k] = m0 * x + m4 * y + m8 * z + m12;
        corners[3 * k + 1] = m1 * x + m5 * y + m9 * z + m13;
        corners[3 * k + 2] = m2 * x + m6 * y + m10 * z + m14;
      }
      visit(t, corners);
    }
    if (slicer.due()) {
      await slicer.pause();
    }
  }
}

/** The (unnormalised) normal of the corners in `c` into `out`; returns its length (twice the area). */
function crossInto(c: Float64Array, out: Float64Array): number {
  const ux = c[3] - c[0];
  const uy = c[4] - c[1];
  const uz = c[5] - c[2];
  const vx = c[6] - c[0];
  const vy = c[7] - c[1];
  const vz = c[8] - c[2];
  out[0] = uy * vz - uz * vy;
  out[1] = uz * vx - ux * vz;
  out[2] = ux * vy - uy * vx;
  return Math.hypot(out[0], out[1], out[2]);
}

function unit(v: Readonly<Vec3>): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (!(len > 0) || !Number.isFinite(len)) {
    throw new Error("direction must be finite and non-zero");
  }
  return [v[0] / len, v[1] / len, v[2] / len];
}

/** Options of {@link selectByArea}. */
export interface AreaOptions {
  /** `"below"` selects faces smaller than `threshold`, `"above"` larger ones. */
  mode: "below" | "above";
  /** In world units squared. */
  threshold: number;
}

/** Faces whose world area is below or above a threshold (strictly). */
export async function selectByArea(
  geom: MeshGeometry,
  options: AreaOptions,
  pause: Pause = yieldToEventLoop,
): Promise<Uint32Array> {
  const mask = new Uint8Array(triangleSlots(geom));
  const n = new Float64Array(3);
  const below = options.mode === "below";
  const threshold = options.threshold;
  await forEachWorldTriangle(geom, pause, (t, c) => {
    const area = 0.5 * crossInto(c, n);
    if (below ? area < threshold : area > threshold) {
      mask[t] = 1;
    }
  });
  return maskToSlots(mask);
}

/** Where the camera is and where it looks, in world coordinates. */
export interface CameraPose {
  eye: Vec3;
  /** The view direction (need not be unit length). */
  forward: Vec3;
  /** An orthographic camera looks along `forward` everywhere; a perspective one from `eye`. */
  orthographic: boolean;
}

/**
 * Faces turned towards the camera: the angle between the face's normal and
 * the direction to the camera is at most `maxAngleDeg` (from the face's
 * centroid to the eye for a perspective camera, against `forward` for an
 * orthographic one). Occlusion is not considered; zero-area faces have no
 * normal and never match.
 */
export async function selectFacingCamera(
  geom: MeshGeometry,
  camera: CameraPose,
  maxAngleDeg: number,
  pause: Pause = yieldToEventLoop,
): Promise<Uint32Array> {
  const mask = new Uint8Array(triangleSlots(geom));
  const n = new Float64Array(3);
  const cosLimit = Math.cos((Math.min(Math.max(maxAngleDeg, 0), 180) * Math.PI) / 180);
  const look = unit(camera.forward);
  const [ex, ey, ez] = camera.eye;
  const ortho = camera.orthographic;
  await forEachWorldTriangle(geom, pause, (t, c) => {
    const len = crossInto(c, n);
    if (!(len > 0)) {
      return;
    }
    let dx: number;
    let dy: number;
    let dz: number;
    if (ortho) {
      dx = -look[0];
      dy = -look[1];
      dz = -look[2];
    } else {
      dx = ex - (c[0] + c[3] + c[6]) / 3;
      dy = ey - (c[1] + c[4] + c[7]) / 3;
      dz = ez - (c[2] + c[5] + c[8]) / 3;
      const d = Math.hypot(dx, dy, dz);
      if (!(d > 0)) {
        return;
      }
      dx /= d;
      dy /= d;
      dz /= d;
    }
    if ((n[0] * dx + n[1] * dy + n[2] * dz) / len >= cosLimit) {
      mask[t] = 1;
    }
  });
  return maskToSlots(mask);
}

/**
 * Overhanging faces, as `inspect.overhangs` counts them: a face's overhang
 * angle is how far its normal tilts below horizontal relative to the build
 * direction (`max(angle from up - 90°, 0)`, so walls and upward faces read 0
 * and a flat ceiling 90), and faces past `criticalAngleDeg` match.
 */
export async function selectOverhangs(
  geom: MeshGeometry,
  buildDirection: Readonly<Vec3>,
  criticalAngleDeg: number,
  pause: Pause = yieldToEventLoop,
): Promise<Uint32Array> {
  if (!Number.isFinite(criticalAngleDeg) || criticalAngleDeg < 0 || criticalAngleDeg > 180) {
    throw new Error("the critical angle must be between 0 and 180 degrees");
  }
  const up = unit(buildDirection);
  const mask = new Uint8Array(triangleSlots(geom));
  const n = new Float64Array(3);
  await forEachWorldTriangle(geom, pause, (t, c) => {
    const len = crossInto(c, n);
    if (!(len > 0) || !Number.isFinite(len)) {
      return;
    }
    const cosUp = Math.min(1, Math.max(-1, (n[0] * up[0] + n[1] * up[1] + n[2] * up[2]) / len));
    const overhang = Math.max((Math.acos(cosUp) * 180) / Math.PI - 90, 0);
    if (overhang > criticalAngleDeg) {
      mask[t] = 1;
    }
  });
  return maskToSlots(mask);
}

/**
 * Sliver faces: the ratio of the circumradius to twice the inradius (1 for
 * an equilateral triangle) is at least `minAspect`, as the mesh healer's
 * sliver overlay reads it. Zero-area faces are left to the degenerate test.
 */
export async function selectSlivers(
  geom: MeshGeometry,
  minAspect: number,
  pause: Pause = yieldToEventLoop,
): Promise<Uint32Array> {
  const mask = new Uint8Array(triangleSlots(geom));
  const n = new Float64Array(3);
  await forEachWorldTriangle(geom, pause, (t, c) => {
    const area = 0.5 * crossInto(c, n);
    if (!(area > 1e-18) || !Number.isFinite(area)) {
      return;
    }
    const ab = Math.hypot(c[3] - c[0], c[4] - c[1], c[5] - c[2]);
    const bc = Math.hypot(c[6] - c[3], c[7] - c[4], c[8] - c[5]);
    const ca = Math.hypot(c[0] - c[6], c[1] - c[7], c[2] - c[8]);
    const s = (ab + bc + ca) * 0.5;
    if ((ab * bc * ca * s) / (8 * area * area) >= minAspect) {
      mask[t] = 1;
    }
  });
  return maskToSlots(mask);
}

/** A face is degenerate by area at most this times its longest edge squared (the engine's `removeDegenerate` default). */
export const DEGENERATE_AREA_EPSILON = 1e-12;

/**
 * Degenerate faces, as `analyze` and `removeDegenerate` read them: a
 * repeated vertex id, or an area at most {@link DEGENERATE_AREA_EPSILON}
 * times the longest edge squared, in local coordinates.
 */
export async function selectDegenerate(geom: MeshGeometry, pause: Pause = yieldToEventLoop): Promise<Uint32Array> {
  const { positions, indices } = geom;
  const total = triangleSlots(geom);
  const mask = new Uint8Array(total);
  const slicer = new Slicer(pause);
  for (let t = 0; t < total; t++) {
    const a = indices[3 * t];
    const b = indices[3 * t + 1];
    const c = indices[3 * t + 2];
    // A dead slot reads [0, 0, 0].
    if (a !== 0 || b !== 0 || c !== 0) {
      if (a === b || b === c || a === c) {
        mask[t] = 1;
      } else {
        const pa = a * 3;
        const pb = b * 3;
        const pc = c * 3;
        const ux = positions[pb] - positions[pa];
        const uy = positions[pb + 1] - positions[pa + 1];
        const uz = positions[pb + 2] - positions[pa + 2];
        const vx = positions[pc] - positions[pa];
        const vy = positions[pc + 1] - positions[pa + 1];
        const vz = positions[pc + 2] - positions[pa + 2];
        const wx = positions[pc] - positions[pb];
        const wy = positions[pc + 1] - positions[pb + 1];
        const wz = positions[pc + 2] - positions[pb + 2];
        const longest = Math.max(ux * ux + uy * uy + uz * uz, wx * wx + wy * wy + wz * wz, vx * vx + vy * vy + vz * vz);
        const cx = uy * vz - uz * vy;
        const cy = uz * vx - ux * vz;
        const cz = ux * vy - uy * vx;
        if (0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz) <= DEGENERATE_AREA_EPSILON * longest) {
          mask[t] = 1;
        }
      }
    }
    if (slicer.due()) {
      await slicer.pause();
    }
  }
  return maskToSlots(mask);
}

/** {@link EdgeAdjacency.across} of an edge no other triangle uses. */
export const BOUNDARY_EDGE = -1;
/** {@link EdgeAdjacency.across} of an edge three or more triangles use. */
export const NON_MANIFOLD_EDGE = -2;

/**
 * The edge neighbours of a mesh's proper triangles (live, three distinct
 * vertex ids), as the engine's edge map sees them.
 */
export interface EdgeAdjacency {
  /** Per triangle slot: 1 for a proper triangle. */
  proper: Uint8Array;
  /**
   * Per triangle corner `3t + k`, for the edge from corner `k` to corner
   * `k + 1`: the one other proper triangle on that edge, else
   * {@link BOUNDARY_EDGE} or {@link NON_MANIFOLD_EDGE}. Unused for improper slots.
   */
  across: Int32Array;
  /** Per proper triangle: its component (triangles linked through shared edges), named by its smallest triangle slot. */
  component: Uint32Array;
  /** Per component (by its smallest slot): 1 when it has a boundary or non-manifold edge. */
  open: Uint8Array;
}

/** The root of `t`, halving the path on the way. */
function findRoot(parent: Uint32Array, t: number): number {
  let x = t;
  while (parent[x] !== x) {
    const up = parent[parent[x]];
    parent[x] = up;
    x = up;
  }
  return x;
}

/**
 * Builds the {@link EdgeAdjacency} of `indices` over `vertexSlots` vertex
 * slots: the vertex fans in counting-sort passes, then each triangle edge
 * looked up in its first vertex's fan. Components are joined through every
 * shared edge, non-manifold ones included, and named by their smallest slot.
 */
export async function buildEdgeAdjacency(
  indices: Uint32Array,
  vertexSlots: number,
  pause: Pause = yieldToEventLoop,
): Promise<EdgeAdjacency> {
  const total = Math.floor(indices.length / 3);
  const slicer = new Slicer(pause);
  const proper = new Uint8Array(total);
  const fanStart = new Uint32Array(vertexSlots + 1);
  for (let t = 0; t < total; t++) {
    if (isProper(indices, vertexSlots, t)) {
      proper[t] = 1;
      fanStart[indices[3 * t] + 1]++;
      fanStart[indices[3 * t + 1] + 1]++;
      fanStart[indices[3 * t + 2] + 1]++;
    }
    if (slicer.due()) {
      await slicer.pause();
    }
  }
  for (let v = 0; v < vertexSlots; v++) {
    fanStart[v + 1] += fanStart[v];
  }
  // Each fan in ascending triangle order.
  const fan = new Uint32Array(fanStart[vertexSlots]);
  const cursor = fanStart.slice(0, vertexSlots);
  for (let t = 0; t < total; t++) {
    if (proper[t] !== 0) {
      fan[cursor[indices[3 * t]]++] = t;
      fan[cursor[indices[3 * t + 1]]++] = t;
      fan[cursor[indices[3 * t + 2]]++] = t;
    }
    if (slicer.due()) {
      await slicer.pause();
    }
  }

  const across = new Int32Array(3 * total).fill(BOUNDARY_EDGE);
  const parent = new Uint32Array(total);
  for (let t = 0; t < total; t++) {
    parent[t] = t;
  }
  for (let t = 0; t < total; t++) {
    if (proper[t] !== 0) {
      for (let k = 0; k < 3; k++) {
        const u = indices[3 * t + k];
        const v = indices[3 * t + (k === 2 ? 0 : k + 1)];
        let other = BOUNDARY_EDGE;
        let found = 0;
        for (let j = fanStart[u], end = fanStart[u + 1]; j < end; j++) {
          const w = fan[j];
          if (w === t || (indices[3 * w] !== v && indices[3 * w + 1] !== v && indices[3 * w + 2] !== v)) {
            continue;
          }
          found++;
          other = w;
          // Join the two components, the smaller slot naming them.
          const ra = findRoot(parent, t);
          const rb = findRoot(parent, w);
          if (ra < rb) {
            parent[rb] = ra;
          } else if (rb < ra) {
            parent[ra] = rb;
          }
        }
        across[3 * t + k] = found <= 1 ? other : NON_MANIFOLD_EDGE;
      }
    }
    if (slicer.due()) {
      await slicer.pause();
    }
  }

  const component = new Uint32Array(total);
  const open = new Uint8Array(total);
  for (let t = 0; t < total; t++) {
    if (proper[t] !== 0) {
      const root = findRoot(parent, t);
      component[t] = root;
      if (across[3 * t] < 0 || across[3 * t + 1] < 0 || across[3 * t + 2] < 0) {
        open[root] = 1;
      }
    }
    if (slicer.due()) {
      await slicer.pause();
    }
  }
  return { proper, across, component, open };
}

/**
 * Faces on an open boundary: with `"edge"` adjacency those with a boundary
 * edge (used by no other triangle), with `"vertex"` those touching a vertex
 * of one. Non-manifold edges are not boundary.
 */
export async function selectBoundaryFaces(
  indices: Uint32Array,
  vertexSlots: number,
  adjacency: EdgeAdjacency,
  kind: "edge" | "vertex",
  pause: Pause = yieldToEventLoop,
): Promise<Uint32Array> {
  const { proper, across } = adjacency;
  const total = proper.length;
  const mask = new Uint8Array(total);
  const slicer = new Slicer(pause);
  const onBoundary = kind === "vertex" ? new Uint8Array(vertexSlots) : null;
  for (let t = 0; t < total; t++) {
    if (proper[t] !== 0) {
      for (let k = 0; k < 3; k++) {
        if (across[3 * t + k] === BOUNDARY_EDGE) {
          mask[t] = 1;
          if (onBoundary !== null) {
            onBoundary[indices[3 * t + k]] = 1;
            onBoundary[indices[3 * t + (k === 2 ? 0 : k + 1)]] = 1;
          }
        }
      }
    }
    if (slicer.due()) {
      await slicer.pause();
    }
  }
  if (onBoundary !== null) {
    for (let t = 0; t < total; t++) {
      if (
        proper[t] !== 0 &&
        (onBoundary[indices[3 * t]] !== 0 || onBoundary[indices[3 * t + 1]] !== 0 || onBoundary[indices[3 * t + 2]] !== 0)
      ) {
        mask[t] = 1;
      }
      if (slicer.due()) {
        await slicer.pause();
      }
    }
  }
  return maskToSlots(mask);
}

/**
 * Faces along a sharp edge: the angle between the normals of the two faces
 * on an edge (0 for a flat continuation) is at least `minAngleDeg`; both
 * faces match. Edges on the boundary or shared by three or more faces, and
 * zero-area faces, are not measured.
 */
export async function selectSharpFaces(
  geom: MeshGeometry,
  adjacency: EdgeAdjacency,
  minAngleDeg: number,
  pause: Pause = yieldToEventLoop,
): Promise<Uint32Array> {
  const total = triangleSlots(geom);
  const normals = new Float64Array(3 * total);
  const scratch = new Float64Array(3);
  await forEachWorldTriangle(geom, pause, (t, c) => {
    const len = crossInto(c, scratch);
    if (len > 0 && Number.isFinite(len)) {
      normals[3 * t] = scratch[0] / len;
      normals[3 * t + 1] = scratch[1] / len;
      normals[3 * t + 2] = scratch[2] / len;
    }
  });
  const cosLimit = Math.cos((Math.min(Math.max(minAngleDeg, 0), 180) * Math.PI) / 180);
  const { proper, across } = adjacency;
  const mask = new Uint8Array(total);
  const slicer = new Slicer(pause);
  for (let t = 0; t < total; t++) {
    if (proper[t] !== 0) {
      const nx = normals[3 * t];
      const ny = normals[3 * t + 1];
      const nz = normals[3 * t + 2];
      if (nx !== 0 || ny !== 0 || nz !== 0) {
        for (let k = 0; k < 3; k++) {
          const w = across[3 * t + k];
          // Each manifold edge once, from its smaller slot.
          if (w <= t) {
            continue;
          }
          const mx = normals[3 * w];
          const my = normals[3 * w + 1];
          const mz = normals[3 * w + 2];
          if ((mx !== 0 || my !== 0 || mz !== 0) && nx * mx + ny * my + nz * mz <= cosLimit) {
            mask[t] = 1;
            mask[w] = 1;
          }
        }
      }
    }
    if (slicer.due()) {
      await slicer.pause();
    }
  }
  return maskToSlots(mask);
}

/** Whether triangle `w` runs along its edge `a`-`b` from `a` to `b`. */
function runsForward(indices: Uint32Array, w: number, a: number, b: number): boolean {
  const x = indices[3 * w];
  const y = indices[3 * w + 1];
  const z = indices[3 * w + 2];
  return (x === a && y === b) || (y === a && z === b) || (z === a && x === b);
}

/**
 * Inconsistently oriented faces: the ones `repair.orient` would reverse,
 * planned the same way on the local positions (the placement neither adds
 * nor removes one).
 *
 * - Patches grow from the smallest unvisited slot across edges shared by
 *   exactly two faces; a neighbour running along the shared edge in the same
 *   direction has the opposite winding. A patch that needs a face both ways
 *   (a strip with a half twist) is left alone.
 * - In each patch the winding with the smaller area is wrong; on a tie, the
 *   one the patch's smallest slot does not have.
 * - A closed component whose signed volume, with those faces reversed, is
 *   negative faces inward: then every one of its faces flips.
 */
export async function selectInconsistentOrientation(
  geom: MeshGeometry,
  adjacency: EdgeAdjacency,
  pause: Pause = yieldToEventLoop,
): Promise<Uint32Array> {
  const { positions, indices } = geom;
  const { proper, across, component, open } = adjacency;
  const total = proper.length;
  const slicer = new Slicer(pause);
  const patchOf = new Int32Array(total).fill(-1);
  const parity = new Uint8Array(total);
  const order = new Uint32Array(total);
  // Per patch, indexed by patch number: run in `order`, orientable, areas.
  const patchStart: number[] = [];
  const patchEnd: number[] = [];
  const patchOrientable: boolean[] = [];
  const patchArea0: number[] = [];
  const patchArea1: number[] = [];
  let filled = 0;

  const areaOf = (t: number): number => {
    const a = indices[3 * t] * 3;
    const b = indices[3 * t + 1] * 3;
    const c = indices[3 * t + 2] * 3;
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    return 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
  };

  for (let seed = 0; seed < total; seed++) {
    if (patchOf[seed] !== -1 || proper[seed] === 0) {
      continue;
    }
    const index = patchStart.length;
    const start = filled;
    let orientable = true;
    patchOf[seed] = index;
    order[filled++] = seed;
    for (let head = start; head < filled; head++) {
      const u = order[head];
      for (let k = 0; k < 3; k++) {
        const w = across[3 * u + k];
        if (w < 0) {
          continue;
        }
        const a = indices[3 * u + k];
        const b = indices[3 * u + (k === 2 ? 0 : k + 1)];
        const want = parity[u] ^ (runsForward(indices, w, a, b) ? 1 : 0);
        if (patchOf[w] === -1) {
          patchOf[w] = index;
          parity[w] = want;
          order[filled++] = w;
        } else if (parity[w] !== want) {
          orientable = false;
        }
      }
      if (slicer.due()) {
        await slicer.pause();
      }
    }
    let area0 = 0;
    let area1 = 0;
    for (let i = start; i < filled; i++) {
      const t = order[i];
      if (parity[t] === 0) {
        area0 += areaOf(t);
      } else {
        area1 += areaOf(t);
      }
      if (slicer.due()) {
        await slicer.pause();
      }
    }
    patchStart.push(start);
    patchEnd.push(filled);
    patchOrientable.push(orientable);
    patchArea0.push(area0);
    patchArea1.push(area1);
  }

  const flip = new Uint8Array(total);
  for (let p = 0; p < patchStart.length; p++) {
    if (!patchOrientable[p]) {
      continue;
    }
    // The seed has parity 0, so 0 keeps a tie.
    const oddWins = patchArea1[p] > patchArea0[p] ? 1 : 0;
    for (let i = patchStart[p]; i < patchEnd[p]; i++) {
      const t = order[i];
      flip[t] = parity[t] !== oddWins ? 1 : 0;
      if (slicer.due()) {
        await slicer.pause();
      }
    }
  }

  // Six times each closed component's signed volume with the flips, about
  // its smallest triangle's first vertex, summed in ascending slot order.
  const volume = new Float64Array(total);
  const blocked = new Uint8Array(total);
  for (let t = 0; t < total; t++) {
    if (proper[t] !== 0) {
      const root = component[t];
      if (!patchOrientable[patchOf[t]]) {
        blocked[root] = 1;
      }
      if (open[root] === 0) {
        const o = indices[3 * root] * 3;
        const ox = positions[o];
        const oy = positions[o + 1];
        const oz = positions[o + 2];
        const a = indices[3 * t] * 3;
        let b = indices[3 * t + 1] * 3;
        let c = indices[3 * t + 2] * 3;
        if (flip[t] !== 0) {
          const s = b;
          b = c;
          c = s;
        }
        const ax = positions[a] - ox;
        const ay = positions[a + 1] - oy;
        const az = positions[a + 2] - oz;
        const bx = positions[b] - ox;
        const by = positions[b + 1] - oy;
        const bz = positions[b + 2] - oz;
        const cx = positions[c] - ox;
        const cy = positions[c + 1] - oy;
        const cz = positions[c + 2] - oz;
        volume[root] += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);
      }
    }
    if (slicer.due()) {
      await slicer.pause();
    }
  }
  for (let t = 0; t < total; t++) {
    if (proper[t] !== 0) {
      const root = component[t];
      if (open[root] === 0 && blocked[root] === 0 && volume[root] < 0) {
        flip[t] ^= 1;
      }
    }
    if (slicer.due()) {
      await slicer.pause();
    }
  }
  return maskToSlots(flip);
}

/**
 * Grows `selected` to whole components: every component (a list of
 * triangle slots) with a selected face joins the result.
 */
export function expandToComponents(
  components: readonly { readonly triangles: ArrayLike<number> }[],
  selected: ArrayLike<number>,
  triangleSlotCount: number,
): Uint32Array {
  const mask = new Uint8Array(triangleSlotCount);
  for (let i = 0; i < selected.length; i++) {
    const t = selected[i];
    if (t >= 0 && t < triangleSlotCount) {
      mask[t] = 1;
    }
  }
  const out = new Uint8Array(triangleSlotCount);
  for (const comp of components) {
    const tris = comp.triangles;
    let touched = false;
    for (let i = 0; i < tris.length; i++) {
      if (mask[tris[i]] !== 0) {
        touched = true;
        break;
      }
    }
    if (touched) {
      for (let i = 0; i < tris.length; i++) {
        if (tris[i] < triangleSlotCount) {
          out[tris[i]] = 1;
        }
      }
    }
  }
  return maskToSlots(out);
}

/** The ascending union of triangle slot lists, each below `triangleSlotCount`. */
export function unionSlots(lists: readonly ArrayLike<number>[], triangleSlotCount: number): Uint32Array {
  const mask = new Uint8Array(triangleSlotCount);
  for (const list of lists) {
    for (let i = 0; i < list.length; i++) {
      const t = list[i];
      if (t >= 0 && t < triangleSlotCount) {
        mask[t] = 1;
      }
    }
  }
  return maskToSlots(mask);
}
