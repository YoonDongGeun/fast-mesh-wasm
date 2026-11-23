// Rectangle packing for the build plate (#266): places as many rectangles as
// it can on a plate, with a gap between them and 90° turns, off the given
// obstacles. The free space is kept as its maximal free rectangles; each
// rectangle takes the free corner with the best score, largest first. The
// first strategy keeps the arrangement compact; when it leaves something
// out, other orders and scores are tried and the one that places the most
// wins. Pure and deterministic (ties go to the first in the list).
//
// Sizes are rounded up to a millionth of the plate's longer side, and the
// packing itself runs on those whole numbers, so it is exact: two placed
// rectangles never overlap and the gap is never short. The rounding lets a
// quarter of a unit through, for sizes that carry `f32` noise: four 50 mm
// boxes fill a 100 mm plate. That quarter between two rectangles and at most
// one unit at the plate's far edges are all a result can be off by.

/** A rectangle's size, in any one unit (the plate's). */
export interface PackSize {
  w: number;
  h: number;
}

/** A rectangle on the plate; `x` and `y` from the plate's corner. */
export interface PackBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface PackOptions {
  /** The plate's size. */
  width: number;
  depth: number;
  /** The least gap between two rectangles (and to an obstacle). Default 0. */
  spacing?: number;
  /** Rectangles may be turned 90°. Default `true`. */
  turn?: boolean;
  /** What to keep off, in plate coordinates; parts off the plate do not matter. */
  obstacles?: readonly PackBox[];
}

/** A placed rectangle: `x`, `y` its lowest corner; turned, its sides are swapped. */
export interface PackedRect {
  index: number;
  x: number;
  y: number;
  turned: boolean;
}

export interface PackResult {
  /** The placed rectangles, by index. */
  placed: PackedRect[];
  /** Those larger than the whole plate, by index. */
  oversized: number[];
  /** Those that would fit on an empty plate, but for which no room was left, by index. */
  noRoom: number[];
}

/** Units along the plate's longer side. */
const UNITS = 1_000_000;
/** What a length may exceed its whole units by, in units. */
const SLACK = 0.25;
/** Past this many scored positions no further strategy is tried. */
const MAX_SCORED = 1_000_000;

/** A rectangle in whole units, with its sides as given. */
interface Part {
  index: number;
  w: number;
  h: number;
}

interface Strategy {
  order: (a: Part, b: Part) => number;
  /** The score of a `w` x `h` rectangle at the lowest corner of `free` (lower is better, compared in turn). */
  score: (free: PackBox, w: number, h: number, plateW: number, plateH: number) => [number, number];
}

const area = (p: Part) => p.w * p.h;
const long = (p: Part) => Math.max(p.w, p.h);
const short = (p: Part) => Math.min(p.w, p.h);

const byArea = (a: Part, b: Part) => area(b) - area(a) || long(b) - long(a) || a.index - b.index;
const byLongSide = (a: Part, b: Part) => long(b) - long(a) || short(b) - short(a) || a.index - b.index;
const byShortSide = (a: Part, b: Part) => short(b) - short(a) || long(b) - long(a) || a.index - b.index;

/** Keeps the used part of the plate small, in the plate's own proportions. */
const compact: Strategy["score"] = (f, w, h, plateW, plateH) => {
  const along = (f.x0 + w) * plateH;
  const across = (f.y0 + h) * plateW;
  return [Math.max(along, across), Math.min(along, across)];
};
/** Leaves the least on the side that fits best. */
const bestShortSide: Strategy["score"] = (f, w, h) => {
  const dw = f.x1 - f.x0 - w;
  const dh = f.y1 - f.y0 - h;
  return [Math.min(dw, dh), Math.max(dw, dh)];
};
/** Leaves the least on the side that fits worst. */
const bestLongSide: Strategy["score"] = (f, w, h) => {
  const dw = f.x1 - f.x0 - w;
  const dh = f.y1 - f.y0 - h;
  return [Math.max(dw, dh), Math.min(dw, dh)];
};
/** Takes the smallest free rectangle. */
const bestArea: Strategy["score"] = (f, w, h) => [
  (f.x1 - f.x0) * (f.y1 - f.y0) - w * h,
  Math.min(f.x1 - f.x0 - w, f.y1 - f.y0 - h),
];
/** As low as it goes, then as far left. */
const bottomLeft: Strategy["score"] = (f, _w, h) => [f.y0 + h, f.x0];

const STRATEGIES: readonly Strategy[] = [
  { order: byArea, score: compact },
  ...[bestShortSide, bestArea, bestLongSide, bottomLeft, compact].flatMap((score) =>
    [byLongSide, byArea, byShortSide].map((order) => ({ order, score })),
  ),
];

function contains(a: PackBox, b: PackBox): boolean {
  return a.x0 <= b.x0 && a.y0 <= b.y0 && a.x1 >= b.x1 && a.y1 >= b.y1;
}

/** The maximal free rectangles once `box` is taken out of them. */
function without(free: readonly PackBox[], box: PackBox): PackBox[] {
  const kept: PackBox[] = [];
  const cut: PackBox[] = [];
  for (const f of free) {
    if (box.x1 <= f.x0 || box.x0 >= f.x1 || box.y1 <= f.y0 || box.y0 >= f.y1) {
      kept.push(f);
      continue;
    }
    if (box.x0 > f.x0) cut.push({ ...f, x1: box.x0 });
    if (box.x1 < f.x1) cut.push({ ...f, x0: box.x1 });
    if (box.y0 > f.y0) cut.push({ ...f, y1: box.y0 });
    if (box.y1 < f.y1) cut.push({ ...f, y0: box.y1 });
  }
  // A piece is a part of what it was cut from, so it can only lie inside a
  // kept rectangle or another piece, never the other way round.
  const maximal = cut.filter(
    (c, i) =>
      !kept.some((k) => contains(k, c)) &&
      !cut.some((d, j) => j !== i && contains(d, c) && (j < i || !contains(c, d))),
  );
  return kept.concat(maximal);
}

/** Whether `a` comes before `b`, entry by entry. */
function less(a: readonly number[], b: readonly number[]): boolean {
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return a[i] < b[i];
    }
  }
  return false;
}

interface Attempt {
  spots: { part: Part; x: number; y: number; turned: boolean }[];
  left: Part[];
  scored: number;
}

/** Packs `parts` into `W` x `H` units, each with `gap` kept free on its far sides. */
function attempt(
  parts: readonly Part[],
  strategy: Strategy,
  W: number,
  H: number,
  gap: number,
  turn: boolean,
  obstacles: readonly PackBox[],
): Attempt {
  // The gap is part of each rectangle, and the plate one gap larger: the gap may hang over its far edges.
  let free: PackBox[] = [{ x0: 0, y0: 0, x1: W + gap, y1: H + gap }];
  for (const o of obstacles) {
    free = without(free, o);
  }
  const out: Attempt = { spots: [], left: [], scored: 0 };
  for (const part of [...parts].sort(strategy.order)) {
    let best: { x: number; y: number; turned: boolean; key: number[] } | null = null;
    for (const turned of turn && part.w !== part.h ? [false, true] : [false]) {
      const w = turned ? part.h : part.w;
      const h = turned ? part.w : part.h;
      for (const f of free) {
        if (f.x0 + w + gap > f.x1 || f.y0 + h + gap > f.y1) {
          continue;
        }
        out.scored++;
        const key = [...strategy.score(f, w, h, W, H), f.y0, f.x0];
        if (best === null || less(key, best.key)) {
          best = { x: f.x0, y: f.y0, turned, key };
        }
      }
    }
    if (best === null) {
      out.left.push(part);
      continue;
    }
    const w = best.turned ? part.h : part.w;
    const h = best.turned ? part.w : part.h;
    out.spots.push({ part, x: best.x, y: best.y, turned: best.turned });
    free = without(free, { x0: best.x, y0: best.y, x1: best.x + w + gap, y1: best.y + h + gap });
  }
  return out;
}

/** Whether attempt `a` is better than `b`: more rectangles placed, then more area. */
function better(a: Attempt, b: Attempt): boolean {
  if (a.spots.length !== b.spots.length) {
    return a.spots.length > b.spots.length;
  }
  const used = (t: Attempt) => t.spots.reduce((s, p) => s + area(p.part), 0);
  return used(a) > used(b);
}

/**
 * Packs `sizes` on the plate (see the file comment). With no obstacles on
 * the plate the arrangement is centred on it; with some it grows from the
 * plate's lowest corner.
 */
export function packRects(sizes: readonly PackSize[], options: PackOptions): PackResult {
  const { width, depth, spacing = 0, turn = true, obstacles = [] } = options;
  if (!(width > 0 && depth > 0 && Number.isFinite(width) && Number.isFinite(depth) && spacing >= 0)) {
    throw new Error("The plate needs a size above zero and a spacing of zero or more.");
  }
  const unit = Math.max(width, depth) / UNITS;
  // The same rounding for the plate and the rectangles, so one exactly as large as the plate fits.
  const up = (x: number) => Math.max(1, Math.ceil(x / unit - SLACK));
  const W = up(width);
  const H = up(depth);
  // A gap as wide as the plate leaves room for one rectangle, whatever it is.
  const gap = spacing > 0 ? Math.min(up(spacing), W + H) : 0;

  const result: PackResult = { placed: [], oversized: [], noRoom: [] };
  const parts: Part[] = [];
  sizes.forEach((size, index) => {
    if (!(size.w >= 0 && size.h >= 0 && Number.isFinite(size.w) && Number.isFinite(size.h))) {
      throw new Error("A rectangle's sides must be numbers, zero or more.");
    }
    const part: Part = { index, w: up(size.w), h: up(size.h) };
    if ((part.w <= W && part.h <= H) || (turn && part.h <= W && part.w <= H)) {
      parts.push(part);
    } else {
      result.oversized.push(index);
    }
  });

  // Rounded outwards, with the gap on their far sides as the rectangles have it.
  const blocked = obstacles
    .map((o) => ({
      x0: Math.floor(o.x0 / unit + SLACK),
      y0: Math.floor(o.y0 / unit + SLACK),
      x1: Math.ceil(o.x1 / unit - SLACK) + gap,
      y1: Math.ceil(o.y1 / unit - SLACK) + gap,
    }))
    .filter((o) => o.x0 < W + gap && o.y0 < H + gap && o.x1 > 0 && o.y1 > 0);

  let best: Attempt = { spots: [], left: parts, scored: 0 };
  let scored = 0;
  for (const strategy of STRATEGIES) {
    if (best.left.length === 0 || scored > MAX_SCORED) {
      break;
    }
    const t = attempt(parts, strategy, W, H, gap, turn, blocked);
    scored += t.scored;
    if (better(t, best)) {
      best = t;
    }
  }

  let dx = 0;
  let dy = 0;
  if (blocked.length === 0) {
    const usedW = Math.max(0, ...best.spots.map((s) => s.x + (s.turned ? s.part.h : s.part.w)));
    const usedH = Math.max(0, ...best.spots.map((s) => s.y + (s.turned ? s.part.w : s.part.h)));
    dx = Math.floor((W - usedW) / 2);
    dy = Math.floor((H - usedH) / 2);
  }
  result.placed = best.spots
    .map((s) => ({ index: s.part.index, x: (s.x + dx) * unit, y: (s.y + dy) * unit, turned: s.turned }))
    .sort((a, b) => a.index - b.index);
  result.noRoom = best.left.map((p) => p.index).sort((a, b) => a - b);
  return result;
}
