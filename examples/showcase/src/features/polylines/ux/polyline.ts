// Polylines as scene objects (#268): what one is, and the basic measures of
// a curve. A curve is a run of world points in millimetres, open or closed;
// a closed one does not repeat its first point. Curves are never written to
// after they are made: an edit makes a new one, so the undo history can hold
// on to the old. No React, no three and no engine here.
import type { Vec3 } from "../../../io/mesh-data";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";

/** A run of points, joined in order (and back to the first when closed). */
export interface PolylineCurve {
  readonly closed: boolean;
  /** World millimetres, `x, y, z` each. A closed curve's first point is not repeated at the end. */
  readonly points: Float64Array;
}

/** A polyline kept as an object of the scene. */
export interface Polyline extends PolylineCurve {
  /** Stable: edits keep it, and so do undo and redo. */
  readonly id: number;
  readonly name: string;
  /** `0xrrggbb`. */
  readonly color: number;
  readonly visible: boolean;
}

/** The parts of a polyline that are display settings: they change at once and are not undo steps. */
export type PolylineDisplay = Pick<Polyline, "name" | "color" | "visible">;

/** The colours of kept polylines, one per Keep press. */
export const POLYLINE_PALETTE: readonly number[] = [0xffd166, 0x4cc9f0, 0xf472b6, 0x80ed99, 0xff9f1c, 0xb8a1ff];

/** The most polylines one action makes: each is a row in the list. */
export const MAX_POLYLINES_PER_ACTION = 1000;

/** The number of points of a curve. */
export function pointCount(curve: PolylineCurve): number {
  return curve.points.length / 3;
}

/** The number of straight pieces of a curve (the closing one included). */
export function segmentCount(curve: PolylineCurve): number {
  const n = pointCount(curve);
  return curve.closed ? n : Math.max(0, n - 1);
}

/** The centre and half diagonal of the box around every curve, or `null` without points. */
export function curvesSphere(curves: readonly PolylineCurve[]): { center: Vec3; radius: number } | null {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const { points } of curves) {
    for (let i = 0; i + 2 < points.length; i += 3) {
      for (let a = 0; a < 3; a++) {
        const v = points[i + a];
        if (v < min[a]) min[a] = v;
        if (v > max[a]) max[a] = v;
      }
    }
  }
  if (!(min[0] <= max[0])) {
    return null;
  }
  return {
    center: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2],
    radius: Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2,
  };
}

/** The diagonal of the box around every curve: their size, for defaults and tolerances. */
export function curvesExtent(curves: readonly PolylineCurve[]): number {
  return 2 * (curvesSphere(curves)?.radius ?? 0);
}

/** Points nearer together than this share of the curve's size are one point. */
const SAME_POINT = 1e-9;

/**
 * `points` as a curve: repeated points are dropped (a section repeats a
 * point where its plane passes through a vertex), and so is a closing point
 * that repeats the first. A closed run with fewer than three points left is
 * an open one. `null` with fewer than two, or with a coordinate that is not
 * finite.
 */
export function makeCurve(points: ArrayLike<number>, closed: boolean): PolylineCurve | null {
  const n = Math.floor(points.length / 3);
  const out = new Float64Array(3 * n);
  for (let i = 0; i < 3 * n; i++) {
    if (!Number.isFinite(points[i])) {
      return null;
    }
    out[i] = points[i];
  }
  const eps = SAME_POINT * curvesExtent([{ closed, points: out }]);
  let m = 0;
  for (let i = 0; i < n; i++) {
    const x = out[3 * i];
    const y = out[3 * i + 1];
    const z = out[3 * i + 2];
    if (m > 0 && Math.hypot(x - out[3 * m - 3], y - out[3 * m - 2], z - out[3 * m - 1]) <= eps) {
      continue;
    }
    out[3 * m] = x;
    out[3 * m + 1] = y;
    out[3 * m + 2] = z;
    m++;
  }
  if (closed) {
    while (m > 1 && Math.hypot(out[0] - out[3 * m - 3], out[1] - out[3 * m - 2], out[2] - out[3 * m - 1]) <= eps) {
      m--;
    }
  }
  if (m < 2) {
    return null;
  }
  return { closed: closed && m >= 3, points: out.slice(0, 3 * m) };
}

/** A curve's length in millimetres, its closing piece included. */
export function curveLength(curve: PolylineCurve): number {
  const p = curve.points;
  const n = pointCount(curve);
  let sum = 0;
  for (let i = 0; i < segmentCount(curve); i++) {
    const a = 3 * i;
    const b = 3 * ((i + 1) % n);
    sum += Math.hypot(p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]);
  }
  return sum;
}

/**
 * A curve's straight pieces for drawing: two endpoints per piece, as `f32`
 * offsets from `center` (so a curve far from the origin still draws finely).
 */
export function curveSegments(curve: PolylineCurve, center: Readonly<Vec3>): Float32Array {
  const p = curve.points;
  const n = pointCount(curve);
  const m = segmentCount(curve);
  const out = new Float32Array(6 * m);
  for (let i = 0; i < m; i++) {
    const a = 3 * i;
    const b = 3 * ((i + 1) % n);
    for (let k = 0; k < 3; k++) {
      out[6 * i + k] = p[a + k] - center[k];
      out[6 * i + 3 + k] = p[b + k] - center[k];
    }
  }
  return out;
}

/** A polyline's readout: its length, points and whether it is closed. */
export function curveReadout(curve: PolylineCurve, unit: LengthUnit): string {
  const n = pointCount(curve).toLocaleString("en");
  return `${formatLength(curveLength(curve), unit)} · ${n} points · ${curve.closed ? "closed" : "open"}`;
}

/**
 * `count` names after `base`, none among `taken`: for one, `base` itself
 * while it is free, else `base 2`, ...; for several, `base 1`, `base 2`, ...
 */
export function freshNames(base: string, count: number, taken: Iterable<string>): string[] {
  const used = new Set(taken);
  if (count === 1 && !used.has(base)) {
    return [base];
  }
  const out: string[] = [];
  for (let n = count === 1 ? 2 : 1; out.length < count; n++) {
    const name = `${base} ${n}`;
    if (!used.has(name)) {
      out.push(name);
    }
  }
  return out;
}
