// Simplify (#268): fewer points within a tolerance (Ramer-Douglas-Peucker).
// Every point dropped stays within the tolerance of the piece that replaced
// its run, and the points kept are points of the curve. An open curve keeps
// its ends; a closed one is split at two points far apart first, so where it
// happens to start does not matter much.
import { pointCount, type PolylineCurve } from "./polyline";

/** The distance of point `i` from the segment `a b`, over flat 3D points. */
function segmentDistance(p: Float64Array, i: number, a: number, b: number): number {
  const ax = p[3 * a];
  const ay = p[3 * a + 1];
  const az = p[3 * a + 2];
  const dx = p[3 * b] - ax;
  const dy = p[3 * b + 1] - ay;
  const dz = p[3 * b + 2] - az;
  const ex = p[3 * i] - ax;
  const ey = p[3 * i + 1] - ay;
  const ez = p[3 * i + 2] - az;
  const len2 = dx * dx + dy * dy + dz * dz;
  const t = len2 > 0 ? Math.min(1, Math.max(0, (ex * dx + ey * dy + ez * dz) / len2)) : 0;
  return Math.hypot(ex - t * dx, ey - t * dy, ez - t * dz);
}

/** The index of the point furthest from point `from` (the first of equals). */
function furthest(p: Float64Array, from: number): number {
  let best = 0;
  let at = -1;
  for (let i = 0; i < p.length / 3; i++) {
    const d = Math.hypot(p[3 * i] - p[3 * from], p[3 * i + 1] - p[3 * from + 1], p[3 * i + 2] - p[3 * from + 2]);
    if (d > at) {
      at = d;
      best = i;
    }
  }
  return best;
}

/** Marks in `keep` what the run `a..b` (indices modulo `n`) needs within `tolerance`. */
function simplifyRun(p: Float64Array, n: number, a: number, b: number, tolerance: number, keep: Uint8Array): void {
  const stack: [number, number][] = [[a, b]];
  while (stack.length > 0) {
    const [s, e] = stack.pop() as [number, number];
    let worst = tolerance;
    let at = -1;
    for (let i = s + 1; i < e; i++) {
      const d = segmentDistance(p, i % n, s % n, e % n);
      if (d > worst) {
        worst = d;
        at = i;
      }
    }
    if (at >= 0) {
      keep[at % n] = 1;
      stack.push([s, at], [at, e]);
    }
  }
}

/**
 * The curve with the points it needs to stay within `tolerance`
 * (millimetres) of the original. A closed curve keeps at least three. The
 * same curve when there is nothing to drop.
 */
export function simplifyCurve(curve: PolylineCurve, tolerance: number): PolylineCurve {
  const p = curve.points;
  const n = pointCount(curve);
  const keep = new Uint8Array(n);
  if (curve.closed) {
    const a = furthest(p, 0);
    const b = furthest(p, a);
    const [s, e] = a < b ? [a, b] : [b, a];
    keep[s] = 1;
    keep[e] = 1;
    simplifyRun(p, n, s, e, tolerance, keep);
    simplifyRun(p, n, e, s + n, tolerance, keep);
    let kept = 0;
    for (let i = 0; i < n; i++) {
      kept += keep[i];
    }
    if (kept < 3) {
      // The widest point off the line between the two: the curve stays a loop.
      let widest = -1;
      let at = -1;
      for (let i = 0; i < n; i++) {
        if (keep[i] === 1) {
          continue;
        }
        const d = segmentDistance(p, i, s, e);
        if (d > widest) {
          widest = d;
          at = i;
        }
      }
      if (at >= 0) {
        keep[at] = 1;
      }
    }
  } else {
    keep[0] = 1;
    keep[n - 1] = 1;
    simplifyRun(p, n, 0, n - 1, tolerance, keep);
  }
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    if (keep[i] === 1) {
      out.push(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
    }
  }
  return out.length === p.length ? curve : { closed: curve.closed, points: Float64Array.from(out) };
}
