// Offset (#268): the curve at a distance to one side of a planar curve.
//
// Each piece of the curve is moved sideways. At a corner the moved pieces
// either part, and are joined around the corner by an arc, or meet, and are
// cut where they do. What is left may still come nearer to some other part
// of the curve than the distance (a narrow place, a bend tighter than the
// distance), so every stretch of it within the distance of any piece is
// taken out: the points within a distance of a straight piece are a convex
// region, so each piece takes one interval out of each moved piece, found
// exactly. The ends that leaves are joined up where they meet. So an offset
// may come apart into several curves, close a gap (the hole it leaves is a
// curve of its own), or vanish. Sharp corners come last: an arc that is
// whole is replaced by its point, unless that would run into something.
import { makeCurve, type PolylineCurve } from "./polyline";
import {
  crosses2,
  fromPlane,
  inside2,
  planarFit,
  type Plane,
  segments2,
  segmentsMeet,
  SegmentTree,
  signedArea2,
  toPlane,
} from "./planar";
import { simplifyCurve } from "./simplify";

/** What happens outside a corner: an arc about it, or both sides extended to a point. */
export type OffsetCorners = "round" | "sharp";

/** A 2D run: `x, y` each. */
export interface Run2 {
  xy: Float64Array;
  closed: boolean;
}

/** A rounded corner's straight pieces stay within this share of the distance from the arc. */
const ARC_ERROR = 1e-3;
/** A sharp corner reaches at most this many distances out; a sharper one is cut off across. */
const MITER_LIMIT = 4;
/** Stretches nearer than the distance by less than this share are at the distance. */
const NEAR = 1e-9;
/** Leftover bits shorter than this share of the distance are rounding. */
const SCRAP = 1e-6;
/** Ends within this share of the distance are joined; then, for what is left, within the next. */
const JOIN = [1e-4, 2e-2];

/** The parameter interval on which a moving point is within `r` of a segment, as `[lo, hi]` in `out`; false for none. */
function within(
  ax: number,
  ay: number,
  ex: number,
  ey: number,
  seg: Float64Array,
  k: number,
  r: number,
  out: Float64Array,
): boolean {
  const px = seg[4 * k];
  const py = seg[4 * k + 1];
  const dx = seg[4 * k + 2] - px;
  const dy = seg[4 * k + 3] - py;
  const len = Math.hypot(dx, dy);
  let lo = Infinity;
  let hi = -Infinity;
  // The discs about the segment's ends.
  const ee = ex * ex + ey * ey;
  for (let end = 0; end < 2; end++) {
    const wx = ax - (end === 0 ? px : seg[4 * k + 2]);
    const wy = ay - (end === 0 ? py : seg[4 * k + 3]);
    const b = wx * ex + wy * ey;
    const c = wx * wx + wy * wy - r * r;
    const disc = b * b - ee * c;
    if (disc > 0 && ee > 0) {
      const root = Math.sqrt(disc);
      // The root that loses no digits, and the other from their product.
      const far = b >= 0 ? -(b + root) : root - b;
      const t0 = far / ee;
      const t1 = far === 0 ? 0 : c / far;
      lo = Math.min(lo, t0, t1);
      hi = Math.max(hi, t0, t1);
    } else if (ee === 0 && c < 0) {
      lo = 0;
      hi = 1;
    }
  }
  // The rectangle along it: two slabs.
  if (len > 0) {
    const ux = dx / len;
    const uy = dy / len;
    const wx = ax - px;
    const wy = ay - py;
    let t0 = 0;
    let t1 = 1;
    const slab = (at: number, rate: number, min: number, max: number): void => {
      if (rate === 0) {
        if (at <= min || at >= max) {
          t0 = 1;
          t1 = 0;
        }
        return;
      }
      const p = (min - at) / rate;
      const q = (max - at) / rate;
      t0 = Math.max(t0, Math.min(p, q));
      t1 = Math.min(t1, Math.max(p, q));
    };
    slab(wx * ux + wy * uy, ex * ux + ey * uy, 0, len);
    slab(wy * ux - wx * uy, ey * ux - ex * uy, -r, r);
    if (t0 < t1) {
      lo = Math.min(lo, t0);
      hi = Math.max(hi, t1);
    }
  }
  lo = Math.max(lo, 0);
  hi = Math.min(hi, 1);
  if (!(lo < hi)) {
    return false;
  }
  out[0] = lo;
  out[1] = hi;
  return true;
}

/** A stretch of a moved run: its points, and per point the corner (point of the curve) whose arc it is on, else -1. */
interface Chain {
  xy: number[];
  arc: number[];
}

/** Per corner with an arc: its number of points, and the points of its sharp form. */
type Arcs = Map<number, { count: number; sharp: number[] }>;

/** The curve's pieces moved to the left by `distance` (right when negative), joined around their corners by arcs. */
function movedRun(xy: Float64Array, closed: boolean, distance: number): Chain & { arcs: Arcs } {
  const n = xy.length / 2;
  const m = closed ? n : n - 1;
  const r = Math.abs(distance);
  const side = Math.sign(distance);
  const dirX = new Float64Array(m);
  const dirY = new Float64Array(m);
  const len = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    const j = (i + 1) % n;
    const dx = xy[2 * j] - xy[2 * i];
    const dy = xy[2 * j + 1] - xy[2 * i + 1];
    len[i] = Math.hypot(dx, dy);
    dirX[i] = dx / len[i];
    dirY[i] = dy / len[i];
  }
  // The side's normal of piece `i`.
  const nx = (i: number) => -side * dirY[i];
  const ny = (i: number) => side * dirX[i];
  const out: Chain & { arcs: Arcs } = { xy: [], arc: [], arcs: new Map() };
  const push = (x: number, y: number, arc = -1) => {
    out.xy.push(x, y);
    out.arc.push(arc);
  };
  const arcStep = 2 * Math.acos(1 / (1 + ARC_ERROR));
  const miterAngle = 2 * Math.acos(1 / MITER_LIMIT);
  const corner = (i: number, k: number) => {
    const px = xy[2 * k];
    const py = xy[2 * k + 1];
    const turn = Math.atan2(dirX[i] * dirY[k] - dirY[i] * dirX[k], dirX[i] * dirX[k] + dirY[i] * dirY[k]);
    const phi = Math.abs(turn);
    /** Where the tangents of the circle about the corner meet, over `count` even turns. */
    const tangents = (count: number): number[] => {
      const points: number[] = [];
      const radius = r / Math.cos(phi / (2 * count));
      for (let j = 0; j < count; j++) {
        const angle = (turn * (2 * j + 1)) / (2 * count);
        const c = radius * Math.cos(angle);
        const s = radius * Math.sin(angle);
        points.push(px + c * nx(i) - s * ny(i), py + s * nx(i) + c * ny(i));
      }
      return points;
    };
    if (side * turn < 0 || phi > Math.PI - 1e-9) {
      // The pieces part.
      const count = Math.max(1, Math.ceil(phi / arcStep));
      const points = tangents(count);
      for (let j = 0; j < points.length; j += 2) {
        push(points[j], points[j + 1], k);
      }
      if (count > 1) {
        out.arcs.set(k, { count, sharp: tangents(phi <= miterAngle ? 1 : 2) });
      }
    } else if (r * Math.tan(phi / 2) <= Math.min(len[i], len[k])) {
      // They meet within both.
      const [x, y] = tangents(1);
      push(x, y);
    } else {
      // They do not: joined straight across, which is too near the corner and will be taken out.
      push(px + r * nx(i), py + r * ny(i));
      push(px + r * nx(k), py + r * ny(k));
    }
  };
  if (closed) {
    for (let k = 0; k < n; k++) {
      corner((k + n - 1) % n, k);
    }
  } else {
    push(xy[0] + r * nx(0), xy[1] + r * ny(0));
    for (let k = 1; k < n - 1; k++) {
      corner(k - 1, k);
    }
    push(xy[2 * n - 2] + r * nx(m - 1), xy[2 * n - 1] + r * ny(m - 1));
  }
  return out;
}

/** The stretches of the moved run that keep the distance `r` from the curve; `whole` when that is all of it. */
function clearChains(xy: Float64Array, closed: boolean, moved: Chain, r: number): { chains: Chain[]; whole: boolean } {
  const source = segments2(xy, closed);
  const tree = new SegmentTree(source);
  const at = moved.xy;
  const n = at.length / 2;
  const m = closed ? n : n - 1;
  let extent = 0;
  for (let i = 0; i < xy.length; i++) {
    extent = Math.max(extent, Math.abs(xy[i]));
  }
  const near = Math.max(0, r - NEAR * r - 1e-13 * extent);
  const chains: Chain[] = [];
  let chain: Chain | null = null;
  const span = new Float64Array(2);
  let killer = -1;
  let whole = true;
  for (let i = 0; i < m; i++) {
    const j = (i + 1) % n;
    const ax = at[2 * i];
    const ay = at[2 * i + 1];
    const ex = at[2 * j] - ax;
    const ey = at[2 * j + 1] - ay;
    const cuts: [number, number][] = [];
    // The piece that took the whole of the last one often takes the whole of this one.
    let gone = killer >= 0 && within(ax, ay, ex, ey, source, killer, near, span) && span[0] === 0 && span[1] === 1;
    if (!gone) {
      killer = -1;
      gone = tree.query(
        Math.min(ax, ax + ex) - r,
        Math.min(ay, ay + ey) - r,
        Math.max(ax, ax + ex) + r,
        Math.max(ay, ay + ey) + r,
        (k) => {
          if (!within(ax, ay, ex, ey, source, k, near, span)) {
            return false;
          }
          if (span[0] === 0 && span[1] === 1) {
            killer = k;
            return true;
          }
          cuts.push([span[0], span[1]]);
          return false;
        },
      );
    }
    if (gone) {
      whole = false;
      chain = null;
      continue;
    }
    cuts.sort((p, q) => p[0] - q[0]);
    // What is left between the cuts: the first stretch goes on from the piece before.
    let t = 0;
    for (let c = 0; c <= cuts.length; c++) {
      const end = c < cuts.length ? cuts[c][0] : 1;
      if (end > t) {
        if (chain === null || t > 0) {
          chain = { xy: [ax + t * ex, ay + t * ey], arc: [t > 0 ? -1 : moved.arc[i]] };
          chains.push(chain);
        }
        if (end < 1) {
          chain.xy.push(ax + end * ex, ay + end * ey);
          chain.arc.push(-1);
        } else {
          chain.xy.push(at[2 * j], at[2 * j + 1]);
          chain.arc.push(moved.arc[j]);
        }
      }
      if (c < cuts.length) {
        t = Math.max(t, cuts[c][1]);
        chain = null;
        whole = false;
      }
    }
  }
  // A closed run's last stretch goes on into its first.
  const first = chains[0];
  if (closed && !whole && chain !== null && chains.length > 1 && first.xy[0] === at[0] && first.xy[1] === at[1]) {
    const last = chains.pop() as Chain;
    chains[0] = { xy: [...last.xy, ...first.xy.slice(2)], arc: [...last.arc, ...first.arc.slice(1)] };
  }
  return { chains, whole };
}

/** A run with the arc each of its points is on. */
interface JoinedRun extends Chain {
  closed: boolean;
}

/**
 * The runs at `distance` to the left of a 2D run (to the right when
 * negative), each in the direction of the run. Empty when there is nothing
 * at that distance. Rounded corners are within a thousandth of the distance
 * from true ones, on the far side.
 */
export function offsetRun(run: Run2, distance: number, corners: OffsetCorners): Run2[] {
  const { closed, xy } = run;
  const r = Math.abs(distance);
  if (r === 0) {
    return [{ xy: Float64Array.from(xy), closed }];
  }
  const moved = movedRun(xy, closed, distance);
  const { chains, whole } = clearChains(xy, closed, moved, r);
  let joined: JoinedRun[];
  if (whole) {
    joined = [{ xy: moved.xy, arc: moved.arc, closed }];
  } else {
    const inward = closed && Math.sign(signedArea2(xy)) === Math.sign(distance);
    const kept = chains.filter(({ xy: c }) => {
      let length = 0;
      let longest = 0;
      let mid = [c[0], c[1]];
      for (let i = 2; i < c.length; i += 2) {
        const l = Math.hypot(c[i] - c[i - 2], c[i + 1] - c[i - 1]);
        length += l;
        if (l > longest) {
          longest = l;
          mid = [(c[i] + c[i - 2]) / 2, (c[i + 1] + c[i - 1]) / 2];
        }
      }
      return length > SCRAP * r && (!closed || inside2(xy, mid[0], mid[1]) === inward);
    });
    joined = joinChains(kept, r);
  }
  return corners === "sharp"
    ? sharpened(joined, moved.arcs)
    : joined.map((j) => ({ xy: Float64Array.from(j.xy), closed: j.closed }));
}

/**
 * Where the last piece of `a` and the first of `b` meet: their ends are one
 * point up to the rounding of arcs, so the crossing of their lines when it is
 * as near to both, else the middle.
 */
function meeting(a: readonly number[], b: readonly number[]): [number, number] {
  const n = a.length;
  const ex = a[n - 2];
  const ey = a[n - 1];
  const gap = Math.hypot(b[0] - ex, b[1] - ey);
  const mid: [number, number] = [(ex + b[0]) / 2, (ey + b[1]) / 2];
  const ux = ex - a[n - 4];
  const uy = ey - a[n - 3];
  const vx = b[2] - b[0];
  const vy = b[3] - b[1];
  const det = ux * vy - uy * vx;
  if (det !== 0) {
    const t = ((b[0] - ex) * vy - (b[1] - ey) * vx) / det;
    const x = ex + t * ux;
    const y = ey + t * uy;
    if (Math.hypot(x - mid[0], y - mid[1]) <= 2 * gap) {
      return [x, y];
    }
  }
  return mid;
}

/** Chains joined end to start where they meet: closed runs where they come round, open ones else. */
function joinChains(chains: readonly Chain[], r: number): JoinedRun[] {
  const n = chains.length;
  const next = new Int32Array(n).fill(-1);
  const taken = new Uint8Array(n);
  for (const tolerance of JOIN) {
    const pairs: { from: number; to: number; d: number }[] = [];
    for (let i = 0; i < n; i++) {
      if (next[i] >= 0) {
        continue;
      }
      const c = chains[i].xy;
      for (let j = 0; j < n; j++) {
        if (taken[j] === 1) {
          continue;
        }
        const d = Math.hypot(chains[j].xy[0] - c[c.length - 2], chains[j].xy[1] - c[c.length - 1]);
        if (d <= tolerance * r) {
          pairs.push({ from: i, to: j, d });
        }
      }
    }
    pairs.sort((p, q) => p.d - q.d);
    for (const { from, to } of pairs) {
      if (next[from] < 0 && taken[to] === 0) {
        next[from] = to;
        taken[to] = 1;
      }
    }
  }
  const out: JoinedRun[] = [];
  const used = new Uint8Array(n);
  const follow = (start: number): JoinedRun => {
    const xy = [...chains[start].xy];
    const arc = [...chains[start].arc];
    used[start] = 1;
    let i = next[start];
    while (i >= 0 && i !== start && used[i] === 0) {
      used[i] = 1;
      const c = chains[i];
      const [x, y] = meeting(xy, c.xy);
      xy[xy.length - 2] = x;
      xy[xy.length - 1] = y;
      for (let k = 2; k < c.xy.length; k++) {
        xy.push(c.xy[k]);
      }
      for (let k = 1; k < c.arc.length; k++) {
        arc.push(c.arc[k]);
      }
      i = next[i];
    }
    const closed = i === start;
    if (closed) {
      [xy[0], xy[1]] = meeting(xy, xy);
      xy.length -= 2;
      arc.length -= 1;
    }
    return { xy, arc, closed };
  };
  for (let i = 0; i < n; i++) {
    if (used[i] === 0 && taken[i] === 0) {
      out.push(follow(i)); // an open run's start
    }
  }
  for (let i = 0; i < n; i++) {
    if (used[i] === 0) {
      out.push(follow(i));
    }
  }
  return out.filter((run) => run.xy.length >= (run.closed ? 6 : 4));
}

/** The runs with every whole arc as its sharp corner, but for the ones that would meet something. */
function sharpened(runs: readonly JoinedRun[], arcs: Arcs): Run2[] {
  /** The arcs that stay round. */
  const rounded = new Set<number>();
  for (;;) {
    /** Per segment, the arc whose sharp corner it is at, else -1. */
    const owner: number[] = [];
    const out = runs.map((run): Run2 => {
      const n = run.arc.length;
      const points: number[] = [];
      const tag: number[] = [];
      // A closed run starts where no arc runs across the seam.
      let start = 0;
      while (run.closed && start < n && run.arc[start] >= 0 && run.arc[start] === run.arc[(start + n - 1) % n]) {
        start++;
      }
      for (let k = 0; k < n; ) {
        const arc = run.arc[(start + k) % n];
        const whole = arcs.get(arc);
        let count = 1;
        while (arc >= 0 && k + count < n && run.arc[(start + k + count) % n] === arc) {
          count++;
        }
        if (whole !== undefined && count === whole.count && !rounded.has(arc)) {
          for (let j = 0; j < whole.sharp.length; j += 2) {
            points.push(whole.sharp[j], whole.sharp[j + 1]);
            tag.push(arc);
          }
        } else {
          for (let j = 0; j < count; j++) {
            const at = (start + k + j) % n;
            points.push(run.xy[2 * at], run.xy[2 * at + 1]);
            tag.push(-1);
          }
        }
        k += count;
      }
      const m = run.closed ? tag.length : tag.length - 1;
      for (let i = 0; i < m; i++) {
        owner.push(Math.max(tag[i], tag[(i + 1) % tag.length]));
      }
      return { xy: Float64Array.from(points), closed: run.closed };
    });
    const met = meetingOwners(out, owner);
    if (met.size === 0) {
      return out;
    }
    for (const arc of met) {
      rounded.add(arc);
    }
  }
}

/** The owners of the runs' segments that meet a segment that is not their neighbour. */
function meetingOwners(runs: readonly Run2[], owner: readonly number[]): Set<number> {
  const all = new Float64Array(4 * owner.length);
  /** Per segment, the ones before and after it. */
  const beside = new Int32Array(2 * owner.length).fill(-1);
  let first = 0;
  for (const run of runs) {
    const part = segments2(run.xy, run.closed);
    const m = part.length / 4;
    all.set(part, 4 * first);
    for (let i = 0; i < m; i++) {
      if (run.closed || i > 0) {
        beside[2 * (first + i)] = first + ((i + m - 1) % m);
      }
      if (run.closed || i < m - 1) {
        beside[2 * (first + i) + 1] = first + ((i + 1) % m);
      }
    }
    first += m;
  }
  const tree = new SegmentTree(all);
  const met = new Set<number>();
  for (let i = 0; i < owner.length; i++) {
    if (owner[i] < 0) {
      continue;
    }
    tree.query(
      Math.min(all[4 * i], all[4 * i + 2]),
      Math.min(all[4 * i + 1], all[4 * i + 3]),
      Math.max(all[4 * i], all[4 * i + 2]),
      Math.max(all[4 * i + 1], all[4 * i + 3]),
      (j) => {
        if (j !== i && j !== beside[2 * i] && j !== beside[2 * i + 1] && segmentsMeet(all, i, j)) {
          met.add(owner[i]);
          if (owner[j] >= 0) {
            met.add(owner[j]);
          }
        }
        return false;
      },
    );
  }
  return met;
}

/** Points that a straight run passes through within this share of the distance are dropped. */
const STRAIGHT = 1e-6;

/** Why a curve cannot be offset (a bent one's distance from its plane is in millimetres). */
export type OffsetProblem = { reason: "straight" | "crossing" } | { reason: "bent"; deviation: number };

/** The plane a curve can be offset in and the curve in it, or why it cannot be. */
function offsetPlane(curve: PolylineCurve): { plane: Plane; xy: Float64Array } | OffsetProblem {
  const fit = planarFit([curve]);
  if ("problem" in fit) {
    return fit.problem === "bent" ? { reason: "bent", deviation: fit.deviation } : { reason: "straight" };
  }
  const xy = toPlane(curve, fit.plane);
  if (curve.closed && crosses2([{ xy, closed: true }])) {
    return { reason: "crossing" };
  }
  return { plane: fit.plane, xy };
}

/**
 * The curves at `distance` from a planar curve: outward of a closed one
 * (inward when negative), to the left of an open one as its plane's normal
 * is faced (to the right when negative). Empty when a loop shrinks away.
 */
export function offsetCurve(
  curve: PolylineCurve,
  distance: number,
  corners: OffsetCorners,
): PolylineCurve[] | OffsetProblem {
  const at = offsetPlane(curve);
  if ("reason" in at) {
    return at;
  }
  // Left of a counter-clockwise loop is its inside.
  const left = curve.closed && signedArea2(at.xy) > 0 ? -distance : distance;
  const out: PolylineCurve[] = [];
  for (const run of offsetRun({ xy: at.xy, closed: curve.closed }, left, corners)) {
    const made = makeCurve(fromPlane(run.xy, at.plane), run.closed);
    if (made !== null) {
      out.push(simplifyCurve(made, STRAIGHT * Math.abs(distance)));
    }
  }
  return out;
}
