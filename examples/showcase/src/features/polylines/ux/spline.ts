// Smooth and resample (#268): a curve replaced by evenly spaced points of a
// spline. The spline is a centripetal Catmull-Rom one, which passes through
// every point and does not overshoot or loop between them. With smoothing it
// goes through a smoothed curve instead: detail shorter than the smoothing
// length goes, what is longer stays in place (the discrete form of a
// smoothing spline: the points' distances from the curve plus a share of its
// bending are least). A wave twice as long as the smoothing length keeps 94%
// of its height, one of its length half, one half as long 6%, so a loop does
// not shrink as it does when points are averaged.
import { curveLength, makeCurve, pointCount, type PolylineCurve, segmentCount } from "./polyline";

/** How {@link resampleCurve} samples, in millimetres. */
export interface ResampleOptions {
  /** The distance between neighbouring points along the curve. */
  spacing: number;
  /** The smoothing length; 0 keeps every point on the spline. */
  smoothing: number;
}

/** The most points a resampled curve may have. */
export const MAX_RESAMPLED_POINTS = 200_000;
/** The most samples smoothing works on. */
const MAX_FINE = 200_000;
/** Smoothing samples at least this many points per smoothing length. */
const FINE_PER_LENGTH = 16;
/** And at most this many: more lose digits in the solve and add nothing. */
const MAX_FINE_PER_LENGTH = 256;
/** A closed curve is continued this many smoothing lengths past its seam: the rest has no say. */
const WRAP_LENGTHS = 6;

/** The pieces an evenly spaced run of `length` is cut into at about `spacing`. */
export function spacedPieces(length: number, spacing: number, closed: boolean): number {
  return Math.max(closed ? 3 : 1, Math.round(length / spacing));
}

/** Why a curve cannot be resampled so, in words, or `null` when it can. */
export function resampleProblem(curve: PolylineCurve, options: ResampleOptions): string | null {
  const { spacing, smoothing } = options;
  if (!(spacing > 0) || !Number.isFinite(spacing)) {
    return "The point spacing must be above zero.";
  }
  if (!(smoothing >= 0) || !Number.isFinite(smoothing)) {
    return "The smoothing length must be zero or more.";
  }
  const length = curveLength(curve);
  const points = spacedPieces(length, spacing, curve.closed) + (curve.closed ? 0 : 1);
  if (points > MAX_RESAMPLED_POINTS) {
    return `That spacing makes ${points.toLocaleString("en")} points: at most ${MAX_RESAMPLED_POINTS.toLocaleString("en")}.`;
  }
  if (smoothing > length / 2) {
    return "The smoothing length must be at most half the curve's length: longer ones would flatten it.";
  }
  return null;
}

/**
 * The curve as evenly spaced points of its spline (see the file comment).
 * An open curve keeps its ends. The spacing is rounded so that a whole
 * number of pieces fits.
 *
 * @throws with {@link resampleProblem}'s words when it cannot be.
 */
export function resampleCurve(curve: PolylineCurve, options: ResampleOptions): PolylineCurve {
  const problem = resampleProblem(curve, options);
  if (problem !== null) {
    throw new Error(problem);
  }
  const base = options.smoothing > 0 ? smoothed(curve, options.smoothing) : curve;
  // Without repeated points, which the spline could not pass through.
  const clean = makeCurve(base.points, base.closed) ?? curve;
  return makeCurve(splineSamples(clean, options.spacing), clean.closed) ?? curve;
}

/** A curve's spline as one cubic per piece: its ends and the tangents there (per unit of the piece's parameter). */
class Spline {
  readonly pieces: number;
  readonly #p: Float64Array;
  readonly #n: number;
  /** Start and end tangents, 6 numbers per piece. */
  readonly #m: Float64Array;

  constructor(curve: PolylineCurve) {
    const p = curve.points;
    const n = pointCount(curve);
    this.#p = p;
    this.#n = n;
    this.pieces = segmentCount(curve);
    this.#m = new Float64Array(6 * this.pieces);
    const at = new Float64Array(12);
    for (let s = 0; s < this.pieces; s++) {
      // The four points around the piece; past an open end, the neighbour mirrored.
      for (let k = 0; k < 4; k++) {
        const i = s + k - 1;
        for (let a = 0; a < 3; a++) {
          if (curve.closed) {
            at[3 * k + a] = p[3 * ((i + n) % n) + a];
          } else if (i < 0) {
            at[a] = 2 * p[a] - p[3 + a];
          } else if (i >= n) {
            at[9 + a] = 2 * p[3 * (n - 1) + a] - p[3 * (n - 2) + a];
          } else {
            at[3 * k + a] = p[3 * i + a];
          }
        }
      }
      const gap = (k: number) =>
        Math.sqrt(Math.hypot(at[3 * k + 3] - at[3 * k], at[3 * k + 4] - at[3 * k + 1], at[3 * k + 5] - at[3 * k + 2]));
      const t0 = gap(0);
      const t1 = gap(1);
      const t2 = gap(2);
      for (let a = 0; a < 3; a++) {
        const d0 = (at[3 + a] - at[a]) / t0;
        const d1 = (at[6 + a] - at[3 + a]) / t1;
        const d2 = (at[9 + a] - at[6 + a]) / t2;
        this.#m[6 * s + a] = t1 * (d0 - (at[6 + a] - at[a]) / (t0 + t1) + d1);
        this.#m[6 * s + 3 + a] = t1 * (d1 - (at[9 + a] - at[3 + a]) / (t1 + t2) + d2);
      }
    }
  }

  /** The point at `t` (0..1) of piece `s`, into `out` at `o`. */
  at(s: number, t: number, out: Float64Array, o: number): void {
    const p = this.#p;
    const a = 3 * s;
    const b = 3 * ((s + 1) % this.#n);
    const t2 = t * t;
    const t3 = t2 * t;
    const h0 = 2 * t3 - 3 * t2 + 1;
    const h1 = t3 - 2 * t2 + t;
    const h2 = 3 * t2 - 2 * t3;
    const h3 = t3 - t2;
    for (let k = 0; k < 3; k++) {
      out[o + k] = h0 * p[a + k] + h1 * this.#m[6 * s + k] + h2 * p[b + k] + h3 * this.#m[6 * s + 3 + k];
    }
  }
}

/** Evenly spaced points of the curve's spline, at about `spacing` (an open curve's last point included). */
function splineSamples(curve: PolylineCurve, spacing: number): Float64Array {
  const spline = new Spline(curve);
  // The length along the spline at steps of each piece.
  const p = curve.points;
  const n = pointCount(curve);
  const steps = new Int32Array(spline.pieces);
  let total = 0;
  for (let s = 0; s < spline.pieces; s++) {
    const a = 3 * s;
    const b = 3 * ((s + 1) % n);
    const chord = Math.hypot(p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]);
    steps[s] = Math.min(64, Math.max(8, Math.ceil((4 * chord) / spacing)));
    total += steps[s];
  }
  const along = new Float64Array(total + 1);
  const here = new Float64Array(3);
  const last = Float64Array.from(p.subarray(0, 3));
  for (let s = 0, k = 1; s < spline.pieces; s++) {
    for (let j = 1; j <= steps[s]; j++, k++) {
      spline.at(s, j / steps[s], here, 0);
      along[k] = along[k - 1] + Math.hypot(here[0] - last[0], here[1] - last[1], here[2] - last[2]);
      last.set(here);
    }
  }
  const length = along[total];
  const pieces = Math.min(spacedPieces(length, spacing, curve.closed), MAX_RESAMPLED_POINTS);
  const count = curve.closed ? pieces : pieces + 1;
  const out = new Float64Array(3 * count);
  let s = 0;
  let base = 0; // the first step of piece `s`
  let k = 1;
  for (let i = 0; i < count; i++) {
    const want = (length * i) / pieces;
    while (k < total && along[k] < want) {
      k++;
    }
    while (k > base + steps[s]) {
      base += steps[s];
      s++;
    }
    const gap = along[k] - along[k - 1];
    const f = gap > 0 ? Math.min(1, Math.max(0, (want - along[k - 1]) / gap)) : 0;
    spline.at(s, (k - 1 - base + f) / steps[s], out, 3 * i);
  }
  if (!curve.closed) {
    // The end exactly, not up to rounding.
    out.set(p.subarray(p.length - 3), out.length - 3);
  }
  return out;
}

/** Evenly spaced points along the straight pieces of the curve, `pieces` of them (an open one's last point included). */
function linearSamples(curve: PolylineCurve, length: number, pieces: number): Float64Array {
  const p = curve.points;
  const n = pointCount(curve);
  const m = segmentCount(curve);
  const count = curve.closed ? pieces : pieces + 1;
  const out = new Float64Array(3 * count);
  let s = 0;
  let from = 0; // the length at the start of piece `s`
  const pieceLength = (i: number) => {
    const a = 3 * i;
    const b = 3 * ((i + 1) % n);
    return Math.hypot(p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]);
  };
  let len = pieceLength(0);
  for (let i = 0; i < count; i++) {
    const want = (length * i) / pieces;
    while (s < m - 1 && from + len < want) {
      from += len;
      s++;
      len = pieceLength(s);
    }
    const f = len > 0 ? Math.min(1, Math.max(0, (want - from) / len)) : 0;
    const a = 3 * s;
    const b = 3 * ((s + 1) % n);
    for (let k = 0; k < 3; k++) {
      out[3 * i + k] = p[a + k] + f * (p[b + k] - p[a + k]);
    }
  }
  return out;
}

/** The curve smoothed by `smoothing` (above zero), as evenly spaced points. */
function smoothed(curve: PolylineCurve, smoothing: number): PolylineCurve {
  const length = curveLength(curve);
  const mean = length / segmentCount(curve);
  const step = Math.max(
    length / MAX_FINE,
    smoothing / MAX_FINE_PER_LENGTH,
    Math.min(smoothing / FINE_PER_LENGTH, mean),
  );
  const pieces = Math.max(curve.closed ? 3 : 2, Math.ceil(length / step));
  const h = length / pieces;
  const fine = linearSamples(curve, length, pieces);
  const n = fine.length / 3;
  const weight = (smoothing / (2 * Math.PI * h)) ** 4;
  // A closed curve is continued around itself on both sides, so that the seam is like any other place.
  const wrap = curve.closed ? Math.ceil((WRAP_LENGTHS * smoothing) / h) + 2 : 0;
  const m = n + 2 * wrap;
  const out = new Float64Array(3 * n);
  for (let a = 0; a < 3; a++) {
    // About the first point: the solve keeps more digits.
    const shift = fine[a];
    const values = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      values[i] = fine[3 * ((((i - wrap) % n) + n) % n) + a] - shift;
    }
    const solved = stiffen(values, weight);
    for (let i = 0; i < n; i++) {
      out[3 * i + a] = solved[wrap + i] + shift;
    }
  }
  return { closed: curve.closed, points: out };
}

/**
 * The `q` with the least `sum (q - values)^2 + weight * sum (second
 * differences of q)^2`, its ends held: a symmetric five-band system, solved
 * directly (LDLᵀ) in linear time.
 */
export function stiffen(values: Float64Array, weight: number): Float64Array {
  const n = values.length;
  const q = Float64Array.from(values);
  const m = n - 2;
  if (m < 1) {
    return q;
  }
  // Row `i` is sample `i + 1`: the matrix's diagonal and its two upper bands.
  const d0 = new Float64Array(m);
  const d1 = new Float64Array(m);
  const d2 = new Float64Array(m);
  const rhs = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    d0[i] = 1 + weight * (4 + (i > 0 ? 1 : 0) + (i < m - 1 ? 1 : 0));
    d1[i] = -4 * weight;
    d2[i] = weight;
    rhs[i] = values[i + 1];
  }
  // The held ends: the second difference at the first unknown has them as a neighbour, as has the one after it.
  rhs[0] += 2 * weight * values[0];
  rhs[m - 1] += 2 * weight * values[n - 1];
  if (m >= 2) {
    rhs[1] -= weight * values[0];
    rhs[m - 2] -= weight * values[n - 1];
  }
  // L: the two lower bands; D in d0.
  const l1 = new Float64Array(m);
  const l2 = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    if (i >= 2) {
      l2[i] = d2[i - 2] / d0[i - 2];
    }
    if (i >= 1) {
      l1[i] = (d1[i - 1] - (i >= 2 ? l2[i] * d0[i - 2] * l1[i - 1] : 0)) / d0[i - 1];
      d0[i] -= l1[i] * l1[i] * d0[i - 1];
      rhs[i] -= l1[i] * rhs[i - 1];
    }
    if (i >= 2) {
      d0[i] -= l2[i] * l2[i] * d0[i - 2];
      rhs[i] -= l2[i] * rhs[i - 2];
    }
  }
  for (let i = m - 1; i >= 0; i--) {
    let x = rhs[i] / d0[i];
    if (i + 1 < m) {
      x -= l1[i + 1] * q[i + 2];
    }
    if (i + 2 < m) {
      x -= l2[i + 2] * q[i + 3];
    }
    q[i + 1] = x;
  }
  return q;
}
