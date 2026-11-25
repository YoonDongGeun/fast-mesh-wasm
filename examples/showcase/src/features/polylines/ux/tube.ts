// Tube (#268): a circle swept along a curve into a closed mesh. Each point of
// the curve gets a ring in the plane that halves its corner, so the tube is
// as thick along every straight piece as it should be. The rings are turned
// as little as possible from one to the next; around a closed curve the turn
// that is left over is spread along it, so the last ring meets the first. An
// open curve's ends are capped. Built directly in linear time, so a curve of
// a hundred thousand points needs no engine work.
import type { Vec3 } from "../../../io/mesh-data";
import { makeCurve, pointCount, type PolylineCurve, segmentCount } from "./polyline";
import { centredSolid, type SolidMesh } from "./solid";

/** The fewest and most sides of a tube. */
export const TUBE_SIDES = { min: 3, max: 256 } as const;
/** A corner's ring is stretched at most this much: past it, the outside of the bend is cut off. */
const MITER_LIMIT = 2;
/** Points nearer together than this share of the radius are one: their direction says nothing. */
const MERGE = 1e-2;

/** The triangles of a tube along `curve` (points that are merged, or none, change it a little). */
export function tubeTriangles(curve: PolylineCurve, sides: number): number {
  return 2 * sides * segmentCount(curve) + (curve.closed ? 0 : 2 * sides);
}

/** Why a tube cannot be built, in words, or `null` when it can. */
export function tubeProblem(curve: PolylineCurve, radius: number, sides: number, maxTriangles: number): string | null {
  if (!(radius > 0) || !Number.isFinite(radius)) {
    return "The tube radius must be above zero.";
  }
  if (!Number.isInteger(sides) || sides < TUBE_SIDES.min || sides > TUBE_SIDES.max) {
    return `The tube's sides must be a whole number from ${TUBE_SIDES.min} to ${TUBE_SIDES.max}.`;
  }
  const triangles = tubeTriangles(curve, sides);
  if (triangles > maxTriangles) {
    return `${triangles.toLocaleString("en")} triangles: at most ${maxTriangles.toLocaleString("en")}. Resample the curve or use fewer sides.`;
  }
  return null;
}

/** The curve without points nearer than `gap` to the one before (an open curve keeps its ends). */
function merged(curve: PolylineCurve, gap: number): PolylineCurve {
  const p = curve.points;
  const n = pointCount(curve);
  const out: number[] = [p[0], p[1], p[2]];
  const far = (i: number) => {
    const k = out.length - 3;
    return Math.hypot(p[3 * i] - out[k], p[3 * i + 1] - out[k + 1], p[3 * i + 2] - out[k + 2]) > gap;
  };
  for (let i = 1; i < n; i++) {
    if (far(i)) {
      out.push(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
    }
  }
  if (curve.closed) {
    while (out.length > 9 && Math.hypot(out[0] - out[out.length - 3], out[1] - out[out.length - 2], out[2] - out[out.length - 1]) <= gap) {
      out.length -= 3;
    }
  } else if (!far(n - 1)) {
    // The end itself, in place of the last point near it.
    if (out.length > 3) {
      out.length -= 3;
    }
    out.push(p[3 * n - 3], p[3 * n - 2], p[3 * n - 1]);
  }
  return makeCurve(out, curve.closed) ?? curve;
}

/** What a sweep made, and where the curve bends tighter than the tube is thick. */
export interface TubeMesh extends SolidMesh {
  /** Pieces of the curve along which the tube's inside folds over: it overlaps itself there. */
  folds: number;
}

/** The unit vector along `(x, y, z)`, or `fallback` for none. */
function unit(x: number, y: number, z: number, fallback: Vec3): Vec3 {
  const len = Math.hypot(x, y, z);
  return len > 0 ? [x / len, y / len, z / len] : fallback;
}

function dot(a: Readonly<Vec3>, b: Readonly<Vec3>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** `x` (across direction `a`) turned with the least turn that takes `a` to `b` (unit): across `b`. */
function carried(x: Readonly<Vec3>, a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 {
  const c = dot(a, b);
  if (c < -1 + 1e-12) {
    // Straight back: `x` is across both.
    return [x[0], x[1], x[2]];
  }
  // Rodrigues about a × b, for the angle between a and b.
  const w = cross(a, b);
  const wx = cross(w, x);
  const k = dot(w, x) / (1 + c);
  const out: Vec3 = [c * x[0] + wx[0] + k * w[0], c * x[1] + wx[1] + k * w[1], c * x[2] + wx[2] + k * w[2]];
  // Kept across b, and unit, against drift over many points.
  const d = dot(out, b);
  return unit(out[0] - d * b[0], out[1] - d * b[1], out[2] - d * b[2], [x[0], x[1], x[2]]);
}

/** A unit vector across `d`. */
function across(d: Readonly<Vec3>): Vec3 {
  const ax = Math.abs(d[0]) <= Math.abs(d[1]) && Math.abs(d[0]) <= Math.abs(d[2]) ? 0 : Math.abs(d[1]) <= Math.abs(d[2]) ? 1 : 2;
  const h: Vec3 = [ax === 0 ? 1 : 0, ax === 1 ? 1 : 0, ax === 2 ? 1 : 0];
  const k = dot(h, d);
  return unit(h[0] - k * d[0], h[1] - k * d[1], h[2] - k * d[2], [1, 0, 0]);
}

/**
 * The closed tube of `radius` with `sides` sides along `curve`. Every edge
 * has two triangles, wound outward, whatever the curve does; where the curve
 * bends tighter than the radius the surface overlaps itself (`folds`).
 */
export function buildTube(curve: PolylineCurve, radius: number, sides: number): TubeMesh {
  const path = merged(curve, MERGE * radius);
  const p = path.points;
  const n = pointCount(path);
  const m = segmentCount(path);
  const dirs: Vec3[] = [];
  const lens = new Float64Array(m);
  let length = 0;
  for (let i = 0; i < m; i++) {
    const a = 3 * i;
    const b = 3 * ((i + 1) % n);
    const d: Vec3 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]];
    lens[i] = Math.hypot(...d);
    length += lens[i];
    dirs.push(unit(...d, [1, 0, 0]));
  }
  // The frame's first axis along each piece, carried around every corner.
  const axes: Vec3[] = [across(dirs[0])];
  for (let i = 1; i < m; i++) {
    axes.push(carried(axes[i - 1], dirs[i - 1], dirs[i]));
  }
  // A closed curve: what the frame has turned by once it is back, undone evenly along the way.
  let twist = 0;
  if (path.closed) {
    const back = carried(axes[m - 1], dirs[m - 1], dirs[0]);
    twist = -Math.atan2(dot(cross(axes[0], back), dirs[0]), dot(axes[0], back));
  }
  const ringCount = n;
  const caps = path.closed ? 0 : 2;
  const positions = new Float64Array(3 * (ringCount * sides + caps));
  let gone = 0; // the length up to ring `i`
  for (let i = 0; i < n; i++) {
    // The pieces that end and start here (an open end has the one).
    const before = path.closed ? (i + m - 1) % m : Math.max(0, i - 1);
    const after = path.closed ? i : Math.min(m - 1, i);
    const a = dirs[before];
    const b = dirs[after];
    const x = axes[before];
    // Turned by the twist's share so far; the first ring by none, as the last piece brings it all the way.
    const angle0 = path.closed ? (twist * (i === 0 ? length : gone)) / length : 0;
    const y = cross(a, x);
    // Along the piece before to the plane that halves the corner.
    const tangent = unit(a[0] + b[0], a[1] + b[1], a[2] + b[2], a);
    const cosHalf = dot(a, tangent);
    const bend: Vec3 = [tangent[0] - cosHalf * a[0], tangent[1] - cosHalf * a[1], tangent[2] - cosHalf * a[2]];
    const limited = Math.max(cosHalf, 1 / MITER_LIMIT);
    for (let s = 0; s < sides; s++) {
      const angle = angle0 + (2 * Math.PI * s) / sides;
      const c = radius * Math.cos(angle);
      const sn = radius * Math.sin(angle);
      const o: Vec3 = [c * x[0] + sn * y[0], c * x[1] + sn * y[1], c * x[2] + sn * y[2]];
      const slide = -dot(o, bend) / limited;
      const k = 3 * (i * sides + s);
      for (let t = 0; t < 3; t++) {
        positions[k + t] = p[3 * i + t] + o[t] + slide * a[t];
      }
    }
    if (i < m) {
      gone += lens[i];
    }
  }
  const indices = new Uint32Array(3 * (2 * sides * m + caps * sides));
  let t = 0;
  let folds = 0;
  const tri = (a: number, b: number, c: number) => {
    indices[t++] = a;
    indices[t++] = b;
    indices[t++] = c;
  };
  for (let i = 0; i < m; i++) {
    const r0 = i * sides;
    const r1 = ((i + 1) % n) * sides;
    let folded = false;
    for (let s = 0; s < sides; s++) {
      const s1 = (s + 1) % sides;
      tri(r0 + s, r0 + s1, r1 + s1);
      tri(r0 + s, r1 + s1, r1 + s);
      // Along the piece, the tube's side runs backwards: its two rings cross.
      let along = 0;
      for (let k = 0; k < 3; k++) {
        along += (positions[3 * (r1 + s) + k] - positions[3 * (r0 + s) + k]) * dirs[i][k];
      }
      folded ||= along < -1e-6 * radius;
    }
    folds += folded ? 1 : 0;
  }
  if (!path.closed) {
    const first = ringCount * sides;
    const last = first + 1;
    for (let k = 0; k < 3; k++) {
      positions[3 * first + k] = p[k];
      positions[3 * last + k] = p[3 * (n - 1) + k];
    }
    const end = (n - 1) * sides;
    for (let s = 0; s < sides; s++) {
      const s1 = (s + 1) % sides;
      tri(first, s1, s);
      tri(last, end + s, end + s1);
    }
  }
  return { ...centredSolid(positions, indices), folds };
}
