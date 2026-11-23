// The build plate's packing (#266): nothing overlaps or leaves the plate,
// the spacing holds, every rectangle is placed or reported, results are
// deterministic, and 90° turns, exact fits and obstacles work.
import { describe, expect, it } from "vitest";
import { type PackBox, packRects, type PackOptions, type PackResult, type PackSize } from "../src/features/build-plate/ux/pack";

/** A seeded generator in [0, 1). */
function random(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s ^ (s >>> 15), 0x2c1b3c6d) + 0x9e3779b9) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 12), 0x297a2d39);
    return ((t ^ (t >>> 15)) >>> 0) / 4294967296;
  };
}

function boxes(sizes: readonly PackSize[], r: PackResult): PackBox[] {
  return r.placed.map((p) => {
    const { w, h } = sizes[p.index];
    return { x0: p.x, y0: p.y, x1: p.x + (p.turned ? h : w), y1: p.y + (p.turned ? w : h) };
  });
}

/** The gap between two boxes along the axis that separates them most (negative: they overlap). */
function gap(a: PackBox, b: PackBox): number {
  return Math.max(a.x0 - b.x1, b.x0 - a.x1, a.y0 - b.y1, b.y0 - a.y1);
}

/** Checks the invariants; what a result may be off by is a millionth of the plate's longer side. */
function check(sizes: readonly PackSize[], options: PackOptions): PackResult {
  const r = packRects(sizes, options);
  const { width, depth, spacing = 0, turn = true, obstacles = [] } = options;
  const eps = Math.max(width, depth) * 1e-6;
  const placed = boxes(sizes, r);
  for (const b of placed) {
    expect(b.x0).toBeGreaterThanOrEqual(0);
    expect(b.y0).toBeGreaterThanOrEqual(0);
    expect(b.x1).toBeLessThanOrEqual(width + eps);
    expect(b.y1).toBeLessThanOrEqual(depth + eps);
  }
  let least = Infinity;
  for (let i = 0; i < placed.length; i++) {
    for (let j = 0; j < i; j++) {
      least = Math.min(least, gap(placed[i], placed[j]));
    }
    for (const o of obstacles) {
      least = Math.min(least, gap(placed[i], o));
    }
  }
  expect(least).toBeGreaterThanOrEqual(spacing - eps / 2);
  // Each rectangle is placed or reported, once.
  const all = [...r.placed.map((p) => p.index), ...r.oversized, ...r.noRoom].sort((a, b) => a - b);
  expect(all).toEqual(sizes.map((_s, i) => i));
  if (!turn) {
    expect(r.placed.every((p) => !p.turned)).toBe(true);
  }
  for (const i of r.oversized) {
    const { w, h } = sizes[i];
    const fits = (a: number, b: number) => a <= width && b <= depth;
    expect(fits(w, h) || (turn && fits(h, w))).toBe(false);
  }
  return r;
}

describe("packRects", () => {
  it("packs 20 demo-like footprints without overlaps, on the plate, around its centre", () => {
    const rnd = random(7);
    const sizes = Array.from({ length: 20 }, () => ({ w: 1 + 2 * rnd(), h: 1 + 2 * rnd() }));
    const r = check(sizes, { width: 200, depth: 200, spacing: 2 });
    expect(r.placed).toHaveLength(20);
    const bs = boxes(sizes, r);
    const x0 = Math.min(...bs.map((b) => b.x0));
    const x1 = Math.max(...bs.map((b) => b.x1));
    const y0 = Math.min(...bs.map((b) => b.y0));
    const y1 = Math.max(...bs.map((b) => b.y1));
    expect((x0 + x1) / 2).toBeCloseTo(100, 2);
    expect((y0 + y1) / 2).toBeCloseTo(100, 2);
    // A cluster, not a line along an edge of the plate: 20 cells of at most 5 mm.
    expect(Math.max(x1 - x0, y1 - y0)).toBeLessThan(6 * 5);
  });

  it("keeps its invariants over many random sets, spacings, plates and obstacles", () => {
    for (let seed = 0; seed < 120; seed++) {
      const rnd = random(seed);
      const width = 20 + 280 * rnd();
      const depth = 20 + 280 * rnd();
      const n = 1 + Math.floor(40 * rnd());
      // Some longer than the plate, some flat.
      const sizes = Array.from({ length: n }, () => ({
        w: rnd() < 0.05 ? 0 : 60 * rnd() ** 2 + (rnd() < 0.05 ? width : 0),
        h: 60 * rnd() ** 2 + (rnd() < 0.05 ? depth : 0),
      }));
      const obstacles = Array.from({ length: seed % 4 }, () => {
        const x0 = width * (1.4 * rnd() - 0.2);
        const y0 = depth * (1.4 * rnd() - 0.2);
        return { x0, y0, x1: x0 + 30 * rnd(), y1: y0 + 30 * rnd() };
      });
      check(sizes, { width, depth, spacing: seed % 3 === 0 ? 0 : 5 * rnd(), turn: seed % 5 !== 0, obstacles });
    }
  });

  it("is deterministic, and does not change its input", () => {
    const rnd = random(3);
    const sizes = Array.from({ length: 60 }, () => ({ w: 5 + 30 * rnd(), h: 5 + 30 * rnd() }));
    const frozen = Object.freeze(sizes.map((s) => Object.freeze({ ...s })));
    const options = { width: 180, depth: 120, spacing: 1.5 };
    const a = packRects(frozen, options);
    expect(packRects(frozen, options)).toEqual(a);
    // Too much for the plate: it fills most of it, and reports the rest.
    expect(a.noRoom.length).toBeGreaterThan(0);
    const used = a.placed.reduce((s, p) => s + (sizes[p.index].w + 1.5) * (sizes[p.index].h + 1.5), 0);
    expect(used / (181.5 * 121.5)).toBeGreaterThan(0.75);
  });

  it("fills a plate exactly, float noise in the sizes included", () => {
    const four = Array.from({ length: 4 }, () => ({ w: 50.000002, h: 49.999999 }));
    expect(check(four, { width: 100, depth: 100 }).placed).toHaveLength(4);
    expect(packRects([{ w: 200, h: 200 }], { width: 200, depth: 200, spacing: 10 }).placed).toEqual([
      { index: 0, x: 0, y: 0, turned: false },
    ]);
    // A side that is not a whole number of packing units.
    expect(packRects([{ w: 300, h: 100.1 }], { width: 300, depth: 100.1 }).placed).toHaveLength(1);
    // Three parts and two gaps.
    const three = Array.from({ length: 3 }, () => ({ w: 32, h: 100 }));
    expect(check(three, { width: 100, depth: 100, spacing: 2 }).placed).toHaveLength(3);
    expect(check(three, { width: 99.9, depth: 100, spacing: 2 }).noRoom).toHaveLength(1);
  });

  it("turns rectangles to fit, only when it may", () => {
    const tall = [{ w: 20, h: 90 }];
    const plate = { width: 100, depth: 30 };
    expect(check(tall, plate).placed).toEqual([{ index: 0, x: 5, y: 5, turned: true }]);
    expect(check(tall, { ...plate, turn: false })).toEqual({ placed: [], oversized: [0], noRoom: [] });
    // Four bars of 90 x 20 on a 100 x 100 plate, however they are given.
    const bars = [0, 1, 2, 3].map((i) => (i % 2 === 0 ? { w: 90, h: 20 } : { w: 20, h: 90 }));
    expect(check(bars, { width: 100, depth: 100, spacing: 5 }).placed).toHaveLength(4);
  });

  it("tells the oversized from those without room", () => {
    const sizes = [
      { w: 120, h: 10 },
      { w: 70, h: 70 },
      { w: 60, h: 60 },
      { w: 20, h: 20 },
    ];
    const r = check(sizes, { width: 100, depth: 100 });
    expect(r.oversized).toEqual([0]);
    expect(r.noRoom).toEqual([2]);
    expect(r.placed.map((p) => p.index)).toEqual([1, 3]);
  });

  it("places more by another strategy when the compact one leaves something out", () => {
    // Eight parts cut from a 100 mm square, on a 104 mm plate: the compact strategy alone places all of them 4 times in 60.
    let whole = 0;
    for (let seed = 0; seed < 60; seed++) {
      const rnd = random(1000 + seed);
      let parts: PackSize[] = [{ w: 100, h: 100 }];
      for (let k = 0; k < 7; k++) {
        parts.sort((a, b) => b.w * b.h - a.w * a.h);
        const [first, ...rest] = parts;
        const at = Math.round(Math.max(first.w, first.h) * (0.3 + 0.4 * rnd()));
        parts =
          first.w > first.h
            ? [...rest, { w: at, h: first.h }, { w: first.w - at, h: first.h }]
            : [...rest, { w: first.w, h: at }, { w: first.w, h: first.h - at }];
      }
      whole += check(parts, { width: 104, depth: 104 }).noRoom.length === 0 ? 1 : 0;
    }
    expect(whole).toBeGreaterThanOrEqual(25);
  });

  it("keeps clear of obstacles, off the plate's edge too", () => {
    // The middle column is taken.
    const obstacles = [{ x0: 40, y0: -10, x1: 60, y1: 200 }];
    const sizes = Array.from({ length: 6 }, () => ({ w: 35, h: 30 }));
    const r = check(sizes, { width: 100, depth: 100, spacing: 5, obstacles });
    expect(r.placed).toHaveLength(6);
    // One that stands just off the plate keeps its gap on it.
    const beside = check([{ w: 100, h: 98 }], {
      width: 100,
      depth: 100,
      spacing: 3,
      obstacles: [{ x0: 0, y0: -5, x1: 100, y1: -1 }],
    });
    expect(beside.placed).toEqual([{ index: 0, x: 0, y: 2, turned: false }]);
    expect(
      check([{ w: 100, h: 98.5 }], { width: 100, depth: 100, spacing: 3, obstacles: [{ x0: 0, y0: -5, x1: 100, y1: -1 }] })
        .noRoom,
    ).toEqual([0]);
    // Far away ones do not matter.
    expect(
      packRects(sizes, { width: 100, depth: 200, obstacles: [{ x0: 500, y0: 0, x1: 600, y1: 10 }] }),
    ).toEqual(packRects(sizes, { width: 100, depth: 200 }));
  });

  it("takes flat and empty input, and refuses a plate without a size", () => {
    expect(packRects([], { width: 10, depth: 10 })).toEqual({ placed: [], oversized: [], noRoom: [] });
    check(Array.from({ length: 5 }, () => ({ w: 0, h: 4 })), { width: 10, depth: 10, spacing: 0 });
    expect(() => packRects([], { width: 0, depth: 10 })).toThrow(/size/);
    expect(() => packRects([{ w: Number.NaN, h: 1 }], { width: 10, depth: 10 })).toThrow(/sides/);
    // A spacing wider than the plate leaves room for one.
    expect(packRects([{ w: 1, h: 1 }, { w: 1, h: 1 }], { width: 10, depth: 10, spacing: 1e12 })).toMatchObject({
      placed: [{ index: 0 }],
      noRoom: [1],
    });
  });

  it("packs a thousand parts quickly, and gives up in bounded time when they cannot fit", () => {
    const rnd = random(11);
    const sizes = Array.from({ length: 1000 }, () => ({ w: 2 + 10 * rnd(), h: 2 + 10 * rnd() }));
    const start = performance.now();
    const fits = check(sizes, { width: 400, depth: 400, spacing: 1 });
    expect(fits.noRoom).toEqual([]);
    const full = check(sizes, { width: 150, depth: 150, spacing: 1 });
    expect(full.noRoom.length).toBeGreaterThan(0);
    // Seconds only on a loaded machine: both together take a fraction of one.
    expect(performance.now() - start).toBeLessThan(20_000);
  }, 30_000);
});
