// Offset (#268): the curve at a distance to one side of a planar curve.
//
// Each piece of the curve is moved sideways, and where two moved pieces part
// at a corner an arc about it joins them. What this makes may still come
// nearer to some other part of the curve than the distance (a narrow place,
// a bend tighter than the distance), so every stretch of it within the
// distance of any piece is taken out: the points within a distance of a
// straight piece are a rectangle and two discs, and where a moved piece or an
// arc enters and leaves them is found exactly. That matters on a fine
// outline, whose neighbouring arcs cross at angles of a thousandth of a
// radian: arcs made of straight pieces would disagree on where. The ends left
// over are then joined where they meet, and arcs become straight pieces last.
// So an offset may come apart into several curves, close a gap (the hole it
// leaves is a curve of its own), or vanish. With sharp corners, an arc that
// is whole is replaced by its point, unless that would run into something.
import { makeCurve, type PolylineCurve } from "./polyline";
import { crosses2, fromPlane, planarFit, type Plane, segments2, segmentsMeet, SegmentTree, signedArea2, toPlane } from "./planar";
import { simplifyCurve } from "./simplify";

/** What happens outside a corner: an arc about it, or both sides extended to a point. */
export type OffsetCorners = "round" | "sharp";

/** A 2D run: `x, y` each. */
export interface Run2 {
  xy: Float64Array;
  closed: boolean;
}

/** An arc's straight pieces stay within this share of the distance from it. */
const ARC_ERROR = 1e-3;
/** An arc that turns by less than this (radians) is its chord. */
const CHORD = 2e-3;
/** A sharp corner reaches at most this many distances out; a sharper one is cut off across. */
const MITER_LIMIT = 4;
/** Stretches nearer than the distance by less than this share are at the distance. */
const NEAR = 1e-9;
/** The curve is first simplified within this share of the distance: shorter detail has no say at it. */
const FINE = 1e-6;
/** Leftover bits shorter than this share of the distance are rounding, and points nearer together are one. */
const SCRAP = 1e-6;
/** Ends within this share of the distance are joined; then, for what is left, within the next. */
const JOIN = [1e-4, 2e-2];
/** The neighbours of a piece that are tried first, at doubling distances: what takes a part of it usually is one of them. */
const NEIGHBOURS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
/** Rounds of giving up sharp corners that run into something before all stay round. */
const MAX_ROUNDS = 12;

/** An arc about a point of the curve: from angle `from`, turning by `sweep` (signed). */
interface Arc {
  /** The point of the curve it is about. */
  corner: number;
  cx: number;
  cy: number;
  from: number;
  sweep: number;
}

/** What of an arc is still to be looked at: its middle, and how far the rest of it is from there at most. */
interface Focus {
  t0: number;
  t1: number;
  mx: number;
  my: number;
  half: number;
}

/** A part of the moved curve: a straight piece, or an arc, from `(x0, y0)` to `(x1, y1)`. */
interface Part {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  arc: Arc | null;
  /** The piece of the curve it belongs to, for where to look first. */
  piece: number;
  /** It starts where the part before it ends. */
  follows: boolean;
}

/** A stretch of a part: from `t0` to `t1` along it (0 to 1). */
interface Stretch {
  part: Part;
  t0: number;
  t1: number;
}

/** Stretches that follow one another without a break. */
type Chain = Stretch[];

/** The point at `t` along a part of an offset by `r`. */
function pointAt(part: Part, t: number, r: number): [number, number] {
  const { arc } = part;
  if (t === 0) {
    return [part.x0, part.y0];
  }
  if (t === 1) {
    return [part.x1, part.y1];
  }
  if (arc === null) {
    return [part.x0 + t * (part.x1 - part.x0), part.y0 + t * (part.y1 - part.y0)];
  }
  const a = arc.from + t * arc.sweep;
  return [arc.cx + r * Math.cos(a), arc.cy + r * Math.sin(a)];
}

/** The parts of the curve moved to the left by `distance` (right when negative). */
function movedParts(xy: Float64Array, closed: boolean, distance: number): Part[] {
  const n = xy.length / 2;
  const m = closed ? n : n - 1;
  const r = Math.abs(distance);
  const side = Math.sign(distance);
  const lines: Part[] = [];
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
    const ox = -distance * dirY[i];
    const oy = distance * dirX[i];
    lines.push({
      x0: xy[2 * i] + ox,
      y0: xy[2 * i + 1] + oy,
      x1: xy[2 * j] + ox,
      y1: xy[2 * j + 1] + oy,
      arc: null,
      piece: i,
      follows: false,
    });
  }
  const parts: Part[] = [];
  for (let k = 0; k < m; k++) {
    const i = k > 0 ? k - 1 : m - 1;
    if (k > 0 || closed) {
      const px = xy[2 * k];
      const py = xy[2 * k + 1];
      const before = lines[i];
      const after = lines[k];
      const turn = Math.atan2(dirX[i] * dirY[k] - dirY[i] * dirX[k], dirX[i] * dirX[k] + dirY[i] * dirY[k]);
      const phi = Math.abs(turn);
      if (phi < 1e-12) {
        after.follows = true;
      } else if (side * turn < 0 || phi > Math.PI - 1e-9) {
        // They part: an arc from one to the other.
        const from = Math.atan2(side * dirX[i], -side * dirY[i]);
        parts.push({
          x0: before.x1,
          y0: before.y1,
          x1: after.x0,
          y1: after.y0,
          arc: { corner: k, cx: px, cy: py, from, sweep: -side * phi },
          piece: i,
          follows: true,
        });
        after.follows = true;
      } else if (r * Math.tan(phi / 2) <= Math.min(len[i], len[k])) {
        // They meet within both: cut there. (What else is too near goes with the rest.)
        const bx = dirX[i] + dirX[k];
        const by = dirY[i] + dirY[k];
        const b = Math.hypot(bx, by);
        const reach = distance / Math.cos(phi / 2);
        const x = px - (reach * by) / b;
        const y = py + (reach * bx) / b;
        before.x1 = x;
        before.y1 = y;
        after.x0 = x;
        after.y0 = y;
        after.follows = true;
      }
    }
    parts.push(lines[k]);
  }
  return parts;
}

/** Adds `[lo, hi]` to the sorted, disjoint intervals of `cuts` (`lo, hi` each). */
function addCut(cuts: number[], lo: number, hi: number): void {
  let at = 0;
  while (at < cuts.length && cuts[at + 1] < lo) {
    at += 2;
  }
  let end = at;
  while (end < cuts.length && cuts[end] <= hi) {
    lo = Math.min(lo, cuts[end]);
    hi = Math.max(hi, cuts[end + 1]);
    end += 2;
  }
  cuts.splice(at, end - at, lo, hi);
}

/** Whether `cuts` leaves nothing of 0 to 1. */
function allCut(cuts: readonly number[]): boolean {
  return cuts.length === 2 && cuts[0] <= 0 && cuts[1] >= 1;
}

/** Adds where a straight part is within `r` of segment `k` to `cuts`. */
function cutLine(part: Part, seg: Float64Array, k: number, r: number, cuts: number[]): void {
  const ax = part.x0;
  const ay = part.y0;
  const ex = part.x1 - ax;
  const ey = part.y1 - ay;
  const px = seg[4 * k];
  const py = seg[4 * k + 1];
  const dx = seg[4 * k + 2] - px;
  const dy = seg[4 * k + 3] - py;
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
  const len = Math.hypot(dx, dy);
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
  // All of it is convex: one interval.
  lo = Math.max(lo, 0);
  hi = Math.min(hi, 1);
  if (lo < hi) {
    addCut(cuts, lo, hi);
  }
}

const TURN = 2 * Math.PI;

/** Adds the angles `[lo, hi]` of an arc's circle to `cuts`, as far as the arc goes. */
function cutAngles(arc: Arc, lo: number, hi: number, cuts: number[]): void {
  const size = Math.abs(arc.sweep);
  // Angles past the arc's start, in its direction.
  let u0 = arc.sweep > 0 ? lo - arc.from : arc.from - hi;
  const width = hi - lo;
  u0 -= TURN * Math.floor(u0 / TURN);
  for (const start of [u0, u0 - TURN]) {
    const a = Math.max(0, start);
    const b = Math.min(size, start + width);
    if (a < b) {
      addCut(cuts, a / size, b === size ? 1 : b / size);
    }
  }
}

/** The stretch of an arc from `t0` to `t1`. */
function focusOn(arc: Arc, t0: number, t1: number, radius: number): Focus {
  const middle = arc.from + ((t0 + t1) / 2) * arc.sweep;
  return {
    t0,
    t1,
    mx: arc.cx + radius * Math.cos(middle),
    my: arc.cy + radius * Math.sin(middle),
    half: (radius * (t1 - t0) * Math.abs(arc.sweep)) / 2,
  };
}

/** Adds where an arc of radius `radius` is within `r` (less than it) of segment `k` to `cuts`, as far as `focus` needs it. */
function cutArc(arc: Arc, focus: Focus, radius: number, seg: Float64Array, k: number, r: number, cuts: number[]): void {
  const px = seg[4 * k];
  const py = seg[4 * k + 1];
  const ex = seg[4 * k + 2] - px;
  const ey = seg[4 * k + 3] - py;
  // Most are all in or all out, as the distance of the middle tells.
  const ll = ex * ex + ey * ey;
  const t = ll > 0 ? Math.min(1, Math.max(0, ((focus.mx - px) * ex + (focus.my - py) * ey) / ll)) : 0;
  const away = Math.hypot(focus.mx - px - t * ex, focus.my - py - t * ey);
  if (away - focus.half >= r) {
    return;
  }
  if (away + focus.half < r) {
    addCut(cuts, focus.t0, focus.t1);
    return;
  }
  // The discs about the segment's ends: the angles about their direction.
  for (let end = 0; end < 2; end++) {
    const dx = (end === 0 ? px : seg[4 * k + 2]) - arc.cx;
    const dy = (end === 0 ? py : seg[4 * k + 3]) - arc.cy;
    const d = Math.hypot(dx, dy);
    if (d === 0) {
      continue;
    }
    const c = (d * d + radius * radius - r * r) / (2 * d * radius);
    if (c < 1) {
      const half = Math.acos(Math.max(-1, c));
      const at = Math.atan2(dy, dx);
      cutAngles(arc, at - half, at + half, cuts);
    }
  }
  // The rectangle along it: the angles at which the arc is beside the segment, and near its line.
  const len = Math.sqrt(ll);
  if (len === 0) {
    return;
  }
  const ux = ex / len;
  const uy = ey / len;
  const wx = arc.cx - px;
  const wy = arc.cy - py;
  const along = wx * ux + wy * uy;
  const across = wy * ux - wx * uy;
  const c0 = -along / radius;
  const c1 = (len - along) / radius;
  const s0 = (-r - across) / radius;
  const s1 = (r - across) / radius;
  if (c1 <= -1 || c0 >= 1 || s1 <= -1 || s0 >= 1) {
    return;
  }
  const p = Math.acos(Math.min(1, c1));
  const q = Math.acos(Math.max(-1, c0));
  const x = Math.asin(Math.max(-1, s0));
  const y = Math.asin(Math.min(1, s1));
  const turned = Math.atan2(uy, ux);
  for (const [a0, a1] of [
    [p, q],
    [-q, -p],
    [TURN - q, TURN - p],
  ]) {
    for (const [b0, b1] of [
      [x, y],
      [Math.PI - y, Math.PI - x],
    ]) {
      const lo = Math.max(a0, b0);
      const hi = Math.min(a1, b1);
      if (lo < hi) {
        cutAngles(arc, turned + lo, turned + hi, cuts);
      }
    }
  }
}

/** The box of a part from `t0` to `t1`, into `out`: `x0, y0, x1, y1`. */
function partBox(part: Part, t0: number, t1: number, r: number, out: Float64Array): void {
  const [x0, y0] = pointAt(part, t0, r);
  const [x1, y1] = pointAt(part, t1, r);
  out[0] = Math.min(x0, x1);
  out[1] = Math.min(y0, y1);
  out[2] = Math.max(x0, x1);
  out[3] = Math.max(y0, y1);
  const { arc } = part;
  if (arc === null) {
    return;
  }
  const from = arc.from + t0 * arc.sweep;
  const to = arc.from + t1 * arc.sweep;
  const a = Math.min(from, to);
  const b = Math.max(from, to);
  for (let q = Math.ceil(a / (Math.PI / 2)); q * (Math.PI / 2) <= b; q++) {
    const way = ((q % 4) + 4) % 4;
    if (way === 0) out[2] = arc.cx + r;
    if (way === 1) out[3] = arc.cy + r;
    if (way === 2) out[0] = arc.cx - r;
    if (way === 3) out[1] = arc.cy - r;
  }
}

/** The stretches of the moved curve that keep the distance `r` from the curve, as chains; `whole` when that is all of it. */
function clearChains(xy: Float64Array, closed: boolean, parts: readonly Part[], r: number): { chains: Chain[]; whole: boolean } {
  const source = segments2(xy, closed);
  const tree = new SegmentTree(source);
  const pieces = source.length / 4;
  let extent = 0;
  for (let i = 0; i < xy.length; i++) {
    extent = Math.max(extent, Math.abs(xy[i]));
  }
  const near = Math.max(0, r - NEAR * r - 1e-13 * extent);
  const chains: Chain[] = [];
  let chain: Chain | null = null;
  const box = new Float64Array(4);
  let killer = -1;
  let whole = true;
  for (const part of parts) {
    const { arc } = part;
    const cuts: number[] = [];
    let focus = arc === null ? null : focusOn(arc, 0, 1, r);
    const cut = (k: number): boolean => {
      if (arc === null || focus === null) {
        cutLine(part, source, k, near, cuts);
      } else {
        cutArc(arc, focus, r, source, k, near, cuts);
      }
      if (allCut(cuts)) {
        killer = k;
        return true;
      }
      return false;
    };
    // What took the last one often takes this one too; then its neighbours.
    let gone = killer >= 0 && cut(killer);
    for (let s = 0; s < NEIGHBOURS.length && NEIGHBOURS[s] < pieces && !gone; s++) {
      for (const k of [part.piece - NEIGHBOURS[s], part.piece + NEIGHBOURS[s]]) {
        if (!gone && (closed || (k >= 0 && k < pieces))) {
          gone = cut(((k % pieces) + pieces) % pieces);
        }
      }
    }
    if (!gone) {
      // The rest of the curve, as far as it matters to what is left.
      const t0 = cuts.length > 0 && cuts[0] <= 0 ? cuts[1] : 0;
      const t1 = cuts.length > 0 && cuts[cuts.length - 1] >= 1 ? cuts[cuts.length - 2] : 1;
      if (arc !== null) {
        focus = focusOn(arc, t0, t1, r);
      }
      partBox(part, t0, t1, r, box);
      gone = tree.query(box[0], box[1], box[2], box[3], cut, near);
    }
    if (gone) {
      whole = false;
      chain = null;
      continue;
    }
    if (!part.follows) {
      chain = null;
      whole = whole && !closed && part === parts[0];
    }
    // What is left between the cuts: the first stretch goes on from the part before.
    let t = 0;
    for (let c = 0; c <= cuts.length; c += 2) {
      const end = c < cuts.length ? cuts[c] : 1;
      if (end > t) {
        if (chain === null || t > 0) {
          chain = [];
          chains.push(chain);
        }
        chain.push({ part, t0: t, t1: end });
      }
      if (c < cuts.length) {
        t = cuts[c + 1];
        chain = null;
        whole = false;
      }
    }
  }
  // A closed run's last stretch goes on into its first.
  if (closed && !whole && chain !== null && chains.length > 1 && parts[0].follows) {
    const first = chains[0][0];
    if (first.part === parts[0] && first.t0 === 0) {
      const last = chains.pop() as Chain;
      chains[0] = last.concat(chains[0]);
    }
  }
  return { chains, whole };
}

/** Chains joined end to start, back to the first when closed. */
interface JoinedRun {
  chains: Chain[];
  closed: boolean;
}

/** Chains joined end to start where they meet: closed runs where they come round, open ones else. */
function joinChains(chains: readonly Chain[], r: number): JoinedRun[] {
  const n = chains.length;
  const starts = chains.map((c) => pointAt(c[0].part, c[0].t0, r));
  const ends = chains.map((c) => pointAt(c[c.length - 1].part, c[c.length - 1].t1, r));
  const next = new Int32Array(n).fill(-1);
  const taken = new Uint8Array(n);
  for (const tolerance of JOIN) {
    const gap = tolerance * r;
    // The free starts by grid cell, so that an end looks only at the ones near it.
    const cells = new Map<string, number[]>();
    for (let j = 0; j < n; j++) {
      if (taken[j] === 0) {
        const key = `${Math.floor(starts[j][0] / gap)} ${Math.floor(starts[j][1] / gap)}`;
        const cell = cells.get(key);
        if (cell === undefined) {
          cells.set(key, [j]);
        } else {
          cell.push(j);
        }
      }
    }
    const pairs: { from: number; to: number; d: number }[] = [];
    for (let i = 0; i < n; i++) {
      if (next[i] >= 0) {
        continue;
      }
      const [x, y] = ends[i];
      const cx = Math.floor(x / gap);
      const cy = Math.floor(y / gap);
      for (let k = 0; k < 9; k++) {
        for (const j of cells.get(`${cx + (k % 3) - 1} ${cy + Math.floor(k / 3) - 1}`) ?? []) {
          const d = Math.hypot(starts[j][0] - x, starts[j][1] - y);
          if (d <= gap) {
            pairs.push({ from: i, to: j, d });
          }
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
  const follow = (start: number): void => {
    const joined: Chain[] = [];
    let i = start;
    do {
      used[i] = 1;
      joined.push(chains[i]);
      i = next[i];
    } while (i >= 0 && used[i] === 0);
    out.push({ chains: joined, closed: i === start });
  };
  for (let i = 0; i < n; i++) {
    if (used[i] === 0 && taken[i] === 0) {
      follow(i); // an open run's start
    }
  }
  for (let i = 0; i < n; i++) {
    if (used[i] === 0) {
      follow(i);
    }
  }
  return out;
}

/** A run of points, and per point the corner it is the sharp point of, else -1. */
interface TaggedRun {
  run: Run2;
  tag: number[];
}

/** A joined run as points: arcs as tangents, and the whole ones that `sharp` takes as their sharp corners. */
function drawn(joined: JoinedRun, r: number, whole: boolean, sharp: (corner: number) => boolean): TaggedRun {
  const xy: number[] = [];
  const tag: number[] = [];
  const gap = SCRAP * r;
  const push = (x: number, y: number, corner = -1) => {
    const n = xy.length;
    if (n > 0 && Math.hypot(x - xy[n - 2], y - xy[n - 1]) <= gap) {
      return;
    }
    xy.push(x, y);
    tag.push(corner);
  };
  const arcStep = 2 * Math.acos(1 / (1 + ARC_ERROR));
  const miterAngle = 2 * Math.acos(1 / MITER_LIMIT);
  const tangents = (arc: Arc, t0: number, t1: number, count: number, corner: number) => {
    const size = (t1 - t0) * arc.sweep;
    const radius = r / Math.cos(size / (2 * count));
    for (let j = 0; j < count; j++) {
      const a = arc.from + t0 * arc.sweep + (size * (2 * j + 1)) / (2 * count);
      push(arc.cx + radius * Math.cos(a), arc.cy + radius * Math.sin(a), corner);
    }
  };
  joined.chains.forEach((chain, c) => {
    chain.forEach(({ part, t0, t1 }, s) => {
      const { arc } = part;
      const size = arc === null ? 0 : Math.abs((t1 - t0) * arc.sweep);
      // A whole arc between the pieces it joins can be a point.
      const between = whole ? joined.closed || (s > 0 && s < chain.length - 1) : s > 0 && s < chain.length - 1;
      if (arc !== null && between && t0 === 0 && t1 === 1 && size > arcStep && sharp(arc.corner)) {
        tangents(arc, 0, 1, size <= miterAngle ? 1 : 2, arc.corner);
        return;
      }
      const [x0, y0] = pointAt(part, t0, r);
      const n = xy.length;
      if (s === 0 && c > 0 && n > 0) {
        // Two chains meet here: one point, up to rounding.
        xy[n - 2] = (xy[n - 2] + x0) / 2;
        xy[n - 1] = (xy[n - 1] + y0) / 2;
      } else {
        push(x0, y0);
      }
      if (arc !== null && size > CHORD) {
        tangents(arc, t0, t1, Math.max(1, Math.ceil(size / arcStep)), -1);
      }
      const [x1, y1] = pointAt(part, t1, r);
      push(x1, y1);
    });
  });
  if (joined.closed && tag.length > 1) {
    const n = xy.length;
    if (!whole && tag[0] < 0 && tag[tag.length - 1] < 0) {
      // The last chain meets the first.
      xy[0] = (xy[0] + xy[n - 2]) / 2;
      xy[1] = (xy[1] + xy[n - 1]) / 2;
      xy.length -= 2;
      tag.length -= 1;
    }
    while (tag.length > 1 && Math.hypot(xy[0] - xy[xy.length - 2], xy[1] - xy[xy.length - 1]) <= gap) {
      xy.length -= 2;
      tag.length -= 1;
    }
  }
  return { run: { xy: Float64Array.from(xy), closed: joined.closed }, tag };
}

/** The corners whose sharp points' segments meet a segment that is not their neighbour. */
function meetingCorners(runs: readonly TaggedRun[]): Set<number> {
  const parts = runs.map(({ run }) => segments2(run.xy, run.closed));
  const total = parts.reduce((sum, p) => sum + p.length / 4, 0);
  const all = new Float64Array(4 * total);
  /** Per segment: the ones before and after it, and the corners at its ends. */
  const beside = new Int32Array(2 * total).fill(-1);
  const owner = new Int32Array(2 * total).fill(-1);
  let first = 0;
  runs.forEach(({ run, tag }, k) => {
    const m = parts[k].length / 4;
    all.set(parts[k], 4 * first);
    for (let i = 0; i < m; i++) {
      if (run.closed || i > 0) {
        beside[2 * (first + i)] = first + ((i + m - 1) % m);
      }
      if (run.closed || i < m - 1) {
        beside[2 * (first + i) + 1] = first + ((i + 1) % m);
      }
      owner[2 * (first + i)] = tag[i];
      owner[2 * (first + i) + 1] = tag[(i + 1) % tag.length];
    }
    first += m;
  });
  const tree = new SegmentTree(all);
  const met = new Set<number>();
  for (let i = 0; i < total; i++) {
    if (owner[2 * i] < 0 && owner[2 * i + 1] < 0) {
      continue;
    }
    tree.query(
      Math.min(all[4 * i], all[4 * i + 2]),
      Math.min(all[4 * i + 1], all[4 * i + 3]),
      Math.max(all[4 * i], all[4 * i + 2]),
      Math.max(all[4 * i + 1], all[4 * i + 3]),
      (j) => {
        if (j !== i && j !== beside[2 * i] && j !== beside[2 * i + 1] && segmentsMeet(all, i, j)) {
          for (const c of [owner[2 * i], owner[2 * i + 1], owner[2 * j], owner[2 * j + 1]]) {
            if (c >= 0) {
              met.add(c);
            }
          }
        }
        return false;
      },
    );
  }
  return met;
}

/** The run without points that repeat the one before (within `gap`). */
function distinct(xy: Float64Array, closed: boolean, gap: number): Float64Array {
  const out: number[] = [];
  for (let i = 0; i < xy.length; i += 2) {
    const n = out.length;
    if (n === 0 || Math.hypot(xy[i] - out[n - 2], xy[i + 1] - out[n - 1]) > gap) {
      out.push(xy[i], xy[i + 1]);
    }
  }
  while (closed && out.length > 2 && Math.hypot(out[0] - out[out.length - 2], out[1] - out[out.length - 1]) <= gap) {
    out.length -= 2;
  }
  return Float64Array.from(out);
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
  const parts = movedParts(xy, closed, distance);
  const { chains, whole } = clearChains(xy, closed, parts, r);
  let joined: JoinedRun[];
  if (whole) {
    joined = [{ chains: [parts.map((part) => ({ part, t0: 0, t1: 1 }))], closed }];
  } else {
    const kept = chains.filter((c) => {
      let length = 0;
      for (const { part, t0, t1 } of c) {
        length +=
          (t1 - t0) * (part.arc === null ? Math.hypot(part.x1 - part.x0, part.y1 - part.y0) : r * Math.abs(part.arc.sweep));
      }
      return length > SCRAP * r;
    });
    joined = joinChains(kept, r);
  }
  const usable = (runs: readonly TaggedRun[]) =>
    runs.map((t) => t.run).filter((q) => q.xy.length >= (q.closed ? 6 : 4));
  if (corners === "sharp") {
    const round = new Set<number>();
    for (let k = 0; k < MAX_ROUNDS; k++) {
      const runs = joined.map((j) => drawn(j, r, whole, (corner) => !round.has(corner)));
      const met = meetingCorners(runs);
      if (met.size === 0) {
        return usable(runs);
      }
      for (const corner of met) {
        round.add(corner);
      }
    }
  }
  return usable(joined.map((j) => drawn(j, r, whole, () => false)));
}

/** Why a curve cannot be offset (a bent one's distance from its plane is in millimetres). */
export type OffsetProblem = { reason: "straight" | "crossing" } | { reason: "bent"; deviation: number };

/** The plane a curve can be offset in and the curve in it, or why it cannot be. */
function offsetPlane(curve: PolylineCurve, gap: number): { plane: Plane; xy: Float64Array } | OffsetProblem {
  const fit = planarFit([curve]);
  if ("problem" in fit) {
    return fit.problem === "bent" ? { reason: "bent", deviation: fit.deviation } : { reason: "straight" };
  }
  const xy = distinct(toPlane(curve, fit.plane), curve.closed, gap);
  if (xy.length < (curve.closed ? 6 : 4)) {
    return { reason: "straight" };
  }
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
  const slack = FINE * Math.abs(distance);
  const at = offsetPlane(simplifyCurve(curve, slack), slack);
  if ("reason" in at) {
    return at;
  }
  // Left of a counter-clockwise loop is its inside.
  const left = curve.closed && signedArea2(at.xy) > 0 ? -distance : distance;
  const out: PolylineCurve[] = [];
  for (const run of offsetRun({ xy: at.xy, closed: curve.closed }, left, corners)) {
    const made = makeCurve(fromPlane(run.xy, at.plane), run.closed);
    if (made !== null && made.closed === run.closed) {
      out.push(simplifyCurve(made, slack));
    }
  }
  return out;
}
