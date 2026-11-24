// Planar curves (#268): the plane a curve lies in, the curve in that plane's
// 2D coordinates and back, and a box tree over 2D segments for what the
// planar tools ask: does a curve cross itself, what lies near this box.
import type { Vec3 } from "../../../io/mesh-data";
import { planeBasis } from "../../../io/report";
import { curvesExtent, type PolylineCurve } from "./polyline";

/** A plane with 2D axes: `u × v = normal`. */
export interface Plane {
  origin: Vec3;
  normal: Vec3;
  u: Vec3;
  v: Vec3;
}

/** The plane through `origin` facing `normal` (unit), with the contour exporter's 2D axes. */
export function planeAt(origin: Readonly<Vec3>, normal: Readonly<Vec3>): Plane {
  const { u, v } = planeBasis(normal);
  return { origin: [...origin], normal: [...normal], u: [...u], v: [...v] };
}

/** What fitting a plane to points found. */
export interface PlaneFit {
  plane: Plane;
  /** The largest distance of a point from the plane. */
  deviation: number;
  /** The points are in a line (or one point): no one plane is theirs. */
  straight: boolean;
}

/**
 * The plane that fits the curves' points best (least squares), facing the
 * side its normal's largest component is positive on: up for a curve that
 * lies flat. `null` without points.
 */
export function fitPlane(curves: readonly PolylineCurve[]): PlaneFit | null {
  let count = 0;
  const c: Vec3 = [0, 0, 0];
  for (const { points } of curves) {
    for (let i = 0; i + 2 < points.length; i += 3) {
      count++;
      for (let a = 0; a < 3; a++) {
        c[a] += (points[i + a] - c[a]) / count;
      }
    }
  }
  if (count === 0) {
    return null;
  }
  // The covariance's smallest axis is the normal.
  let [xx, xy, xz, yy, yz, zz] = [0, 0, 0, 0, 0, 0];
  for (const { points } of curves) {
    for (let i = 0; i + 2 < points.length; i += 3) {
      const x = points[i] - c[0];
      const y = points[i + 1] - c[1];
      const z = points[i + 2] - c[2];
      xx += x * x;
      xy += x * y;
      xz += x * z;
      yy += y * y;
      yz += y * z;
      zz += z * z;
    }
  }
  const axes = eigenSymmetric([
    [xx, xy, xz],
    [xy, yy, yz],
    [xz, yz, zz],
  ]);
  const normal = facingUp(axes[0].vector);
  let deviation = 0;
  for (const { points } of curves) {
    for (let i = 0; i + 2 < points.length; i += 3) {
      const d = (points[i] - c[0]) * normal[0] + (points[i + 1] - c[1]) * normal[1] + (points[i + 2] - c[2]) * normal[2];
      deviation = Math.max(deviation, Math.abs(d));
    }
  }
  // A second axis a thousand times shorter than the first: a line.
  const straight = !(Math.max(0, axes[1].value) > 1e-6 * axes[2].value);
  return { plane: planeAt(c, normal), deviation, straight };
}

/** `n` as a unit normal whose largest component is positive; one along an axis up to rounding is exactly so. */
function facingUp(n: Readonly<Vec3>): Vec3 {
  const len = Math.hypot(n[0], n[1], n[2]) || 1;
  const out: Vec3 = [n[0] / len, n[1] / len, n[2] / len];
  let k = 2;
  for (const a of [1, 0]) {
    if (Math.abs(out[a]) > Math.abs(out[k]) + 1e-9) {
      k = a;
    }
  }
  if (out[k] < 0) {
    for (let a = 0; a < 3; a++) {
      out[a] = -out[a];
    }
  }
  if (out.every((x, a) => a === k || Math.abs(x) < 1e-7)) {
    return [k === 0 ? 1 : 0, k === 1 ? 1 : 0, k === 2 ? 1 : 0];
  }
  return out;
}

/** The eigenvalues of a symmetric 3x3 matrix with their unit vectors, smallest first (Jacobi). */
function eigenSymmetric(a: number[][]): { value: number; vector: Vec3 }[] {
  const v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let sweep = 0; sweep < 32; sweep++) {
    const off = a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2;
    if (off <= 1e-30 * (a[0][0] ** 2 + a[1][1] ** 2 + a[2][2] ** 2)) {
      break;
    }
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ]) {
      if (a[p][q] === 0) {
        continue;
      }
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (const m of [a, v]) {
        for (let k = 0; k < 3; k++) {
          [m[k][p], m[k][q]] = [c * m[k][p] - s * m[k][q], s * m[k][p] + c * m[k][q]];
        }
      }
      for (let k = 0; k < 3; k++) {
        [a[p][k], a[q][k]] = [c * a[p][k] - s * a[q][k], s * a[p][k] + c * a[q][k]];
      }
    }
  }
  return [0, 1, 2]
    .map((k) => ({ value: a[k][k], vector: [v[0][k], v[1][k], v[2][k]] as Vec3 }))
    .sort((p, q) => p.value - q.value);
}

/** How far a planar curve may stray from its plane, as a share of its size. */
const PLANAR_SHARE = 1e-4;
/** And as a share of its largest coordinate: section points are `f32`. */
const PLANAR_F32 = 1e-6;

/** The largest distance from their plane at which `curves` still count as planar. */
export function planarTolerance(curves: readonly PolylineCurve[]): number {
  let far = 0;
  for (const { points } of curves) {
    for (let i = 0; i < points.length; i++) {
      far = Math.max(far, Math.abs(points[i]));
    }
  }
  return PLANAR_SHARE * curvesExtent(curves) + PLANAR_F32 * far;
}

/** The plane the curves lie in, or why there is none: they are in a line, or stray from their best plane. */
export type PlanarFit = { plane: Plane } | { problem: "straight" } | { problem: "bent"; deviation: number };

/** The plane all of `curves` lie in, if they do. */
export function planarFit(curves: readonly PolylineCurve[]): PlanarFit {
  const fit = fitPlane(curves);
  if (fit === null || fit.straight) {
    return { problem: "straight" };
  }
  if (fit.deviation > planarTolerance(curves)) {
    return { problem: "bent", deviation: fit.deviation };
  }
  return { plane: fit.plane };
}

/** The curve in the plane's 2D coordinates, `x, y` each. */
export function toPlane(curve: PolylineCurve, plane: Plane): Float64Array {
  const { origin: o, u, v } = plane;
  const p = curve.points;
  const out = new Float64Array((p.length / 3) * 2);
  for (let i = 0, j = 0; i + 2 < p.length; i += 3, j += 2) {
    const x = p[i] - o[0];
    const y = p[i + 1] - o[1];
    const z = p[i + 2] - o[2];
    out[j] = x * u[0] + y * u[1] + z * u[2];
    out[j + 1] = x * v[0] + y * v[1] + z * v[2];
  }
  return out;
}

/** Points in the plane's 2D coordinates (`x, y` each) as world points on it. */
export function fromPlane(xy: ArrayLike<number>, plane: Plane): Float64Array {
  const { origin: o, u, v } = plane;
  const out = new Float64Array((xy.length / 2) * 3);
  for (let i = 0, j = 0; i + 1 < xy.length; i += 2, j += 3) {
    for (let a = 0; a < 3; a++) {
      out[j + a] = o[a] + xy[i] * u[a] + xy[i + 1] * v[a];
    }
  }
  return out;
}

/** The signed area of a closed 2D run (positive counter-clockwise). */
export function signedArea2(xy: ArrayLike<number>): number {
  let a = 0;
  const n = xy.length;
  for (let i = 0; i + 1 < n; i += 2) {
    const j = (i + 2) % n;
    a += xy[i] * xy[j + 1] - xy[j] * xy[i + 1];
  }
  return a / 2;
}

/** Whether `(x, y)` is inside the closed 2D run (even-odd). */
export function inside2(xy: ArrayLike<number>, x: number, y: number): boolean {
  let hit = false;
  const n = xy.length;
  for (let i = 0, j = n - 2; i < n; j = i, i += 2) {
    const yi = xy[i + 1];
    const yj = xy[j + 1];
    if (yi > y !== yj > y && x < ((xy[j] - xy[i]) * (y - yi)) / (yj - yi) + xy[i]) {
      hit = !hit;
    }
  }
  return hit;
}

/** The 2D segments of a run of points, `x0, y0, x1, y1` each. */
export function segments2(xy: ArrayLike<number>, closed: boolean): Float64Array {
  const n = xy.length / 2;
  const m = closed ? n : Math.max(0, n - 1);
  const out = new Float64Array(4 * m);
  for (let i = 0; i < m; i++) {
    const j = (i + 1) % n;
    out[4 * i] = xy[2 * i];
    out[4 * i + 1] = xy[2 * i + 1];
    out[4 * i + 2] = xy[2 * j];
    out[4 * i + 3] = xy[2 * j + 1];
  }
  return out;
}

/** Segments per leaf of a {@link SegmentTree}. */
const LEAF = 4;

/**
 * A box tree over 2D segments (`x0, y0, x1, y1` each), kept in their order:
 * the segments of a curve follow one another in space, so runs of them make
 * tight boxes.
 */
export class SegmentTree {
  readonly #count: number;
  /** The first leaf's node; nodes are numbered from 1, children `2i` and `2i + 1`. */
  readonly #first: number;
  /** `x0, y0, x1, y1` per node. */
  readonly #box: Float64Array;

  constructor(segments: Float64Array) {
    this.#count = segments.length / 4;
    const leaves = Math.max(1, Math.ceil(this.#count / LEAF));
    let first = 1;
    while (first < leaves) {
      first *= 2;
    }
    this.#first = first;
    const box = new Float64Array(8 * first);
    for (let i = 0; i < box.length; i += 4) {
      box[i] = box[i + 1] = Infinity;
      box[i + 2] = box[i + 3] = -Infinity;
    }
    for (let s = 0; s < this.#count; s++) {
      const b = 4 * (first + Math.floor(s / LEAF));
      for (let e = 0; e < 4; e += 2) {
        const x = segments[4 * s + e];
        const y = segments[4 * s + e + 1];
        if (x < box[b]) box[b] = x;
        if (y < box[b + 1]) box[b + 1] = y;
        if (x > box[b + 2]) box[b + 2] = x;
        if (y > box[b + 3]) box[b + 3] = y;
      }
    }
    for (let i = first - 1; i >= 1; i--) {
      const l = 8 * i;
      const r = l + 4;
      box[4 * i] = Math.min(box[l], box[r]);
      box[4 * i + 1] = Math.min(box[l + 1], box[r + 1]);
      box[4 * i + 2] = Math.max(box[l + 2], box[r + 2]);
      box[4 * i + 3] = Math.max(box[l + 3], box[r + 3]);
    }
    this.#box = box;
  }

  /**
   * Calls `visit` with the segments whose leaf's box meets the given one (or
   * is within `reach` of it), until it returns true. Returns whether it did.
   * Nearer boxes come first, so a visitor that stops at what is near does.
   */
  query(x0: number, y0: number, x1: number, y1: number, visit: (segment: number) => boolean, reach = 0): boolean {
    const box = this.#box;
    const reach2 = reach * reach;
    const mx = (x0 + x1) / 2;
    const my = (y0 + y1) / 2;
    /** How far node `i`'s box is from the middle of the given one (squared). */
    const away = (i: number): number => {
      const dx = Math.max(0, box[4 * i] - mx, mx - box[4 * i + 2]);
      const dy = Math.max(0, box[4 * i + 1] - my, my - box[4 * i + 3]);
      return dx * dx + dy * dy;
    };
    const stack = [1];
    while (stack.length > 0) {
      const i = stack.pop() as number;
      const b = 4 * i;
      const dx = Math.max(0, box[b] - x1, x0 - box[b + 2]);
      const dy = Math.max(0, box[b + 1] - y1, y0 - box[b + 3]);
      if (!(dx * dx + dy * dy <= reach2)) {
        continue;
      }
      if (i < this.#first) {
        if (away(2 * i) <= away(2 * i + 1)) {
          stack.push(2 * i + 1, 2 * i);
        } else {
          stack.push(2 * i, 2 * i + 1);
        }
        continue;
      }
      const start = LEAF * (i - this.#first);
      const end = Math.min(start + LEAF, this.#count);
      for (let s = start; s < end; s++) {
        if (visit(s)) {
          return true;
        }
      }
    }
    return false;
  }
}

/** Twice the signed area of the triangle `a, b, c`. */
function orient(ax: number, ay: number, bx: number, by: number, cx: number, cy: number): number {
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

/** Whether segments `i` and `j` meet, touching included. */
export function segmentsMeet(s: Float64Array, i: number, j: number): boolean {
  const ax = s[4 * i];
  const ay = s[4 * i + 1];
  const bx = s[4 * i + 2];
  const by = s[4 * i + 3];
  const cx = s[4 * j];
  const cy = s[4 * j + 1];
  const dx = s[4 * j + 2];
  const dy = s[4 * j + 3];
  if (
    Math.max(ax, bx) < Math.min(cx, dx) ||
    Math.max(cx, dx) < Math.min(ax, bx) ||
    Math.max(ay, by) < Math.min(cy, dy) ||
    Math.max(cy, dy) < Math.min(ay, by)
  ) {
    return false;
  }
  return (
    orient(ax, ay, bx, by, cx, cy) * orient(ax, ay, bx, by, dx, dy) <= 0 &&
    orient(cx, cy, dx, dy, ax, ay) * orient(cx, cy, dx, dy, bx, by) <= 0
  );
}

/** Whether segment `j` runs straight back over segment `i`, which ends where it starts. */
function foldsBack(s: Float64Array, i: number, j: number): boolean {
  const dx = s[4 * i + 2] - s[4 * i];
  const dy = s[4 * i + 3] - s[4 * i + 1];
  const ex = s[4 * j + 2] - s[4 * j];
  const ey = s[4 * j + 3] - s[4 * j + 1];
  return dx * ey - dy * ex === 0 && dx * ex + dy * ey < 0;
}

/** Whether any of the 2D runs meets itself or another one (a touch counts). */
export function crosses2(runs: readonly { xy: ArrayLike<number>; closed: boolean }[]): boolean {
  const parts = runs.map((r) => segments2(r.xy, r.closed));
  const all = new Float64Array(parts.reduce((n, p) => n + p.length, 0));
  /** The run of each segment, and where that run's segments start and end. */
  const total = all.length / 4;
  const first = new Int32Array(total);
  const last = new Int32Array(total);
  const closedRun = new Uint8Array(total);
  let at = 0;
  parts.forEach((p, r) => {
    all.set(p, 4 * at);
    const end = at + p.length / 4;
    first.fill(at, at, end);
    last.fill(end - 1, at, end);
    closedRun.fill(runs[r].closed ? 1 : 0, at, end);
    at = end;
  });
  const tree = new SegmentTree(all);
  for (let i = 0; i < total; i++) {
    const hit = tree.query(
      Math.min(all[4 * i], all[4 * i + 2]),
      Math.min(all[4 * i + 1], all[4 * i + 3]),
      Math.max(all[4 * i], all[4 * i + 2]),
      Math.max(all[4 * i + 1], all[4 * i + 3]),
      (j) => {
        if (j <= i) {
          return false;
        }
        // Neighbours share a point: they cross only by running back over each other.
        if (j === i + 1 && j <= last[i]) {
          return foldsBack(all, i, j);
        }
        if (closedRun[i] === 1 && i === first[i] && j === last[i]) {
          return foldsBack(all, j, i);
        }
        return segmentsMeet(all, i, j);
      },
    );
    if (hit) {
      return true;
    }
  }
  return false;
}
