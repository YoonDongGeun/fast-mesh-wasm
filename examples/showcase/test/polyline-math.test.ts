// The polyline math (#268): curves and their measures, the plane a curve
// lies in, simplify, smooth and resample, and offset. The offset is held to
// its definition on random shapes: every point of it is the distance away,
// and a raster of the true distance finds nothing missing.
import { describe, expect, it } from "vitest";
import { offsetCurve, offsetRun, type Run2 } from "../src/features/polylines/ux/offset";
import {
  crosses2,
  fitPlane,
  fromPlane,
  inside2,
  planarFit,
  planarTolerance,
  planeAt,
  SegmentTree,
  segments2,
  signedArea2,
  toPlane,
} from "../src/features/polylines/ux/planar";
import {
  curveLength,
  curveReadout,
  curveSegments,
  curvesExtent,
  curvesSphere,
  freshNames,
  makeCurve,
  pointCount,
  type PolylineCurve,
  segmentCount,
} from "../src/features/polylines/ux/polyline";
import { simplifyCurve } from "../src/features/polylines/ux/simplify";
import { resampleCurve, resampleProblem, spacedPieces, stiffen } from "../src/features/polylines/ux/spline";

type V2 = readonly [number, number];

/** A closed or open curve in the plane z = `z`. */
function flat(points: readonly V2[], closed: boolean, z = 0): PolylineCurve {
  return makeCurve(points.flatMap(([x, y]) => [x, y, z]), closed)!;
}

function circle(n: number, r: number, z = 0, phase = 0): PolylineCurve {
  return flat(
    Array.from({ length: n }, (_, i): V2 => {
      const a = phase + (2 * Math.PI * i) / n;
      return [r * Math.cos(a), r * Math.sin(a)];
    }),
    true,
    z,
  );
}

/** A pseudo-random number generator (mulberry32). */
function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SEGMENTS = new WeakMap<Run2, Float64Array>();

/** The distance of `(x, y)` from a 2D run. */
function runDistance(run: Run2, x: number, y: number): number {
  let s = SEGMENTS.get(run);
  if (s === undefined) {
    s = segments2(run.xy, run.closed);
    SEGMENTS.set(run, s);
  }
  let best = Infinity;
  for (let i = 0; i < s.length; i += 4) {
    const dx = s[i + 2] - s[i];
    const dy = s[i + 3] - s[i + 1];
    const l2 = dx * dx + dy * dy;
    const t = l2 > 0 ? Math.min(1, Math.max(0, ((x - s[i]) * dx + (y - s[i + 1]) * dy) / l2)) : 0;
    best = Math.min(best, Math.hypot(x - s[i] - t * dx, y - s[i + 1] - t * dy));
  }
  return best;
}

/** `curve` turned and moved in space. */
function moved(curve: PolylineCurve): PolylineCurve {
  const plane = planeAt([10, -20, 30], [1 / Math.sqrt(14), 2 / Math.sqrt(14), 3 / Math.sqrt(14)]);
  return makeCurve(fromPlane(toPlane(curve, planeAt([0, 0, 0], [0, 0, 1])), plane), curve.closed)!;
}

describe("curves", () => {
  it("drops repeated points and a closing point that repeats the first", () => {
    const c = makeCurve([0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0], true)!;
    expect([...c.points]).toEqual([0, 0, 0, 1, 0, 0, 1, 1, 0]);
    expect(c.closed).toBe(true);
    expect(pointCount(c)).toBe(3);
    expect(segmentCount(c)).toBe(3);
  });

  it("is open with two points, and nothing with one or with a coordinate that is not finite", () => {
    expect(makeCurve([0, 0, 0, 1, 0, 0, 0, 0, 0], true)).toMatchObject({ closed: false });
    expect(makeCurve([0, 0, 0, 0, 0, 0], false)).toBeNull();
    expect(makeCurve([], false)).toBeNull();
    expect(makeCurve([0, 0, 0, 1, Number.NaN, 0], false)).toBeNull();
  });

  it("measures the closing piece too", () => {
    const square = flat([[0, 0], [2, 0], [2, 2], [0, 2]], true);
    expect(curveLength(square)).toBe(8);
    expect(curveLength({ ...square, closed: false })).toBe(6);
    expect(curvesExtent([square])).toBeCloseTo(2 * Math.SQRT2, 12);
    expect(curvesSphere([square])!.center).toEqual([1, 1, 0]);
    expect(curvesSphere([])).toBeNull();
  });

  it("reads in the display unit", () => {
    const line = flat([[0, 0], [25.4, 0]], false);
    expect(curveReadout(line, "mm")).toBe("25.4 mm · 2 points · open");
    expect(curveReadout(line, "in")).toBe("1 in · 2 points · open");
    expect(curveReadout(flat([[0, 0], [1, 0], [0, 1]], true), "mm")).toContain("3 points · closed");
  });

  it("draws as segments about a centre, so far curves keep their detail", () => {
    const c = makeCurve([1e7, 0, 0, 1e7 + 0.25, 0, 0, 1e7 + 0.25, 0.5, 0], true)!;
    expect([...curveSegments(c, [1e7, 0, 0])]).toEqual([0, 0, 0, 0.25, 0, 0, 0.25, 0, 0, 0.25, 0.5, 0, 0.25, 0.5, 0, 0, 0, 0]);
  });

  it("names one after its base and several with numbers, none taken", () => {
    expect(freshNames("A path", 1, [])).toEqual(["A path"]);
    expect(freshNames("A path", 1, ["A path"])).toEqual(["A path 2"]);
    expect(freshNames("A cut", 3, ["A cut 2"])).toEqual(["A cut 1", "A cut 3", "A cut 4"]);
  });
});

describe("planes", () => {
  it("fits a flat curve's plane facing up, and takes a curve in it there and back", () => {
    const c = circle(40, 3, 7);
    const fit = fitPlane([c])!;
    expect(fit.plane.normal).toEqual([0, 0, 1]);
    expect(fit.plane.origin[2]).toBeCloseTo(7, 12);
    expect(fit.deviation).toBeLessThan(1e-12);
    expect(fit.straight).toBe(false);
    const tilted = moved(c);
    const at = planarFit([tilted]);
    if (!("plane" in at)) {
      throw new Error("not planar");
    }
    expect(at.plane.normal[2]).toBeCloseTo(3 / Math.sqrt(14), 9);
    const back = fromPlane(toPlane(tilted, at.plane), at.plane);
    back.forEach((v, i) => expect(v).toBeCloseTo(tilted.points[i], 9));
    // The 2D axes are right-handed about the normal, and lengths stay.
    const { u, v, normal } = at.plane;
    expect(u[1] * v[2] - u[2] * v[1]).toBeCloseTo(normal[0], 12);
    expect(signedArea2(toPlane(tilted, at.plane))).toBeCloseTo(Math.abs(signedArea2(toPlane(c, fit.plane))), 6);
  });

  it("tells a straight line and a bent curve from a planar one, and allows f32 rounding", () => {
    expect(planarFit([flat([[0, 0], [1, 0], [5, 0]], false)])).toEqual({ problem: "straight" });
    const helix = makeCurve(
      Array.from({ length: 50 }, (_, i) => [Math.cos(i / 5), Math.sin(i / 5), i / 50]).flat(),
      false,
    )!;
    expect(planarFit([helix])).toMatchObject({ problem: "bent" });
    const rounded = makeCurve(Float32Array.from(moved(circle(200, 2)).points, (x) => x + 1000), true)!;
    expect("plane" in planarFit([rounded])).toBe(true);
    expect(planarTolerance([rounded])).toBeGreaterThan(1e-3);
  });

  it("finds what meets a box, and nothing else, as a scan does", () => {
    const rnd = random(5);
    const s = Float64Array.from({ length: 4 * 333 }, () => rnd());
    const tree = new SegmentTree(s);
    for (let k = 0; k < 50; k++) {
      const [x, y] = [rnd(), rnd()];
      const [w, h] = [0.2 * rnd(), 0.2 * rnd()];
      const hits = new Set<number>();
      tree.query(x, y, x + w, y + h, (i) => {
        hits.add(i);
        return false;
      });
      for (let i = 0; i < 333; i++) {
        const meets =
          Math.min(s[4 * i], s[4 * i + 2]) <= x + w &&
          Math.max(s[4 * i], s[4 * i + 2]) >= x &&
          Math.min(s[4 * i + 1], s[4 * i + 3]) <= y + h &&
          Math.max(s[4 * i + 1], s[4 * i + 3]) >= y;
        if (meets) {
          expect(hits.has(i)).toBe(true);
        }
      }
    }
    // Stops where the visitor says.
    let seen = 0;
    expect(tree.query(-1, -1, 2, 2, () => ++seen === 3)).toBe(true);
    expect(seen).toBe(3);
  });

  it("finds curves that cross or touch themselves or each other", () => {
    const run = (points: readonly V2[], closed: boolean) => ({ xy: points.flat(), closed });
    const square = run([[0, 0], [4, 0], [4, 4], [0, 4]], true);
    expect(crosses2([square])).toBe(false);
    expect(crosses2([run([[0, 0], [4, 4], [4, 0], [0, 4]], true)])).toBe(true); // a bow tie
    expect(crosses2([run([[0, 0], [4, 0], [4, 4], [2, 0], [0, 4]], true)])).toBe(true); // a touch
    expect(crosses2([run([[0, 0], [4, 0], [2, 0], [2, 3]], false)])).toBe(true); // back over itself
    expect(crosses2([square, run([[1, 1], [3, 1], [3, 3], [1, 3]], true)])).toBe(false); // nested
    expect(crosses2([square, run([[3, 1], [6, 1], [6, 3], [3, 3]], true)])).toBe(true);
    expect(crosses2([run([[0, 0], [1, 0], [0, 1]], true)])).toBe(false);
  });
});

describe("simplify", () => {
  it("keeps an open curve's ends and stays within the tolerance", () => {
    const rnd = random(1);
    const points = Array.from({ length: 400 }, (_, i): V2 => [i / 10, Math.sin(i / 20) + 0.01 * (rnd() - 0.5)]);
    const c = flat(points, false);
    let before = pointCount(c);
    for (const tolerance of [0.001, 0.01, 0.1, 1]) {
      const s = simplifyCurve(c, tolerance);
      expect(pointCount(s)).toBeLessThanOrEqual(before);
      before = pointCount(s);
      expect([...s.points.subarray(0, 3)]).toEqual([...c.points.subarray(0, 3)]);
      expect([...s.points.subarray(-3)]).toEqual([...c.points.subarray(-3)]);
      const run = { xy: toPlane(s, planeAt([0, 0, 0], [0, 0, 1])), closed: false };
      for (const [x, y] of points) {
        // The plane's axes are turned: compare as the run is.
        const q = toPlane(flat([[x, y], [x + 1, y]], false), planeAt([0, 0, 0], [0, 0, 1]));
        expect(runDistance(run, q[0], q[1])).toBeLessThanOrEqual(tolerance + 1e-12);
      }
    }
    expect(before).toBeLessThan(10);
  });

  it("drops the points of straight runs on a closed curve, wherever it starts", () => {
    const side = 25;
    const points: V2[] = [];
    for (const [x, y, dx, dy] of [[0, 0, 1, 0], [1, 0, 0, 1], [1, 1, -1, 0], [0, 1, 0, -1]]) {
      for (let i = 0; i < side; i++) {
        points.push([4 * (x + (dx * i) / side), 4 * (y + (dy * i) / side)]);
      }
    }
    for (const start of [0, 7, 60]) {
      const c = flat([...points.slice(start), ...points.slice(0, start)], true);
      const s = simplifyCurve(c, 1e-9);
      // The corners, and at most the point it happened to start from.
      expect(pointCount(s)).toBeLessThanOrEqual(5);
      expect(curveLength(s)).toBeCloseTo(16, 9);
      expect(s.closed).toBe(true);
    }
  });

  it("leaves a loop a loop however large the tolerance, and a curve with nothing to drop as it is", () => {
    const loop = simplifyCurve(circle(100, 1), 50);
    expect(pointCount(loop)).toBe(3);
    expect(Math.abs(signedArea2(toPlane(loop, planeAt([0, 0, 0], [0, 0, 1]))))).toBeGreaterThan(0.5);
    const square = flat([[0, 0], [2, 0], [2, 2], [0, 2]], true);
    expect(simplifyCurve(square, 0.1)).toBe(square);
  });

  it("works in space", () => {
    const helix = makeCurve(
      Array.from({ length: 2000 }, (_, i) => [5 * Math.cos(i / 50), 5 * Math.sin(i / 50), i / 100]).flat(),
      false,
    )!;
    const s = simplifyCurve(helix, 0.02);
    expect(pointCount(s)).toBeLessThan(400);
    expect(curveLength(s)).toBeCloseTo(curveLength(helix), 0);
  });
});

describe("smooth and resample", () => {
  it("spaces points evenly along a closed curve, on the spline through its points", () => {
    const c = circle(24, 10);
    const r = resampleCurve(c, { spacing: 0.5, smoothing: 0 });
    expect(r.closed).toBe(true);
    expect(pointCount(r)).toBe(spacedPieces(2 * Math.PI * 10, 0.5, true));
    let min = Infinity;
    let max = 0;
    const n = pointCount(r);
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const d = Math.hypot(r.points[3 * j] - r.points[3 * i], r.points[3 * j + 1] - r.points[3 * i + 1]);
      min = Math.min(min, d);
      max = Math.max(max, d);
      // A spline through 24 points of a circle is the circle to a few microns.
      expect(Math.hypot(r.points[3 * i], r.points[3 * i + 1])).toBeCloseTo(10, 2);
    }
    expect(max / min).toBeLessThan(1.02);
    expect(curveLength(r)).toBeCloseTo(2 * Math.PI * 10, 1);
  });

  it("keeps an open curve's ends exactly, and a straight line straight", () => {
    const c = makeCurve([0, 0, 0, 1, 2, 0, 4, 1, 3, 7, 7, 7], false)!;
    const r = resampleCurve(c, { spacing: 0.1, smoothing: 0 });
    expect([...r.points.subarray(0, 3)]).toEqual([0, 0, 0]);
    expect([...r.points.subarray(-3)]).toEqual([7, 7, 7]);
    // Through every point of the curve.
    for (let i = 0; i < 4; i++) {
      let best = Infinity;
      for (let k = 0; k < r.points.length; k += 3) {
        best = Math.min(best, Math.hypot(r.points[k] - c.points[3 * i], r.points[k + 1] - c.points[3 * i + 1], r.points[k + 2] - c.points[3 * i + 2]));
      }
      expect(best).toBeLessThan(0.06);
    }
    const line = resampleCurve(flat([[0, 0], [3, 0]], false), { spacing: 1, smoothing: 0 });
    expect(pointCount(line)).toBe(4);
    [0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0].forEach((v, i) => expect(line.points[i]).toBeCloseTo(v, 12));
  });

  it("smooths away detail shorter than the smoothing length and leaves longer shapes their size", () => {
    const rnd = random(3);
    const noisy = flat(
      Array.from({ length: 2000 }, (_, i): V2 => {
        const a = (2 * Math.PI * i) / 2000;
        const r = 20 + 0.2 * (rnd() - 0.5);
        return [r * Math.cos(a), r * Math.sin(a)];
      }),
      true,
    );
    const off = (c: PolylineCurve) => {
      let worst = 0;
      for (let k = 0; k < c.points.length; k += 3) {
        worst = Math.max(worst, Math.abs(Math.hypot(c.points[k], c.points[k + 1]) - 20));
      }
      return worst;
    };
    expect(off(noisy)).toBeGreaterThan(0.09);
    const smooth = resampleCurve(noisy, { spacing: 1, smoothing: 10 });
    expect(off(smooth)).toBeLessThan(0.03);
    // A loop of 126 mm smoothed by 10 mm keeps its radius.
    expect(curveLength(smooth)).toBeCloseTo(2 * Math.PI * 20, 0);
    const corner = resampleCurve(flat([[0, 0], [10, 0], [10, 10]], false), { spacing: 0.2, smoothing: 4 });
    expect([...corner.points.subarray(0, 3)]).toEqual([0, 0, 0]);
    expect([...corner.points.subarray(-3)]).toEqual([10, 10, 0]);
    // The corner is rounded off.
    let nearest = Infinity;
    for (let k = 0; k < corner.points.length; k += 3) {
      nearest = Math.min(nearest, Math.hypot(corner.points[k] - 10, corner.points[k + 1]));
    }
    expect(nearest).toBeGreaterThan(0.1);
  });

  it("smooths a loop alike wherever it starts", () => {
    const shape = (start: number) => {
      const pts = Array.from({ length: 80 }, (_, k): V2 => {
        const a = (2 * Math.PI * ((k + start) % 80)) / 80;
        const r = 10 + 2 * Math.sin(5 * a) + (k + start) % 2;
        return [r * Math.cos(a), r * Math.sin(a)];
      });
      return flat(pts, true);
    };
    const a = resampleCurve(shape(0), { spacing: 0.5, smoothing: 3 });
    const b = resampleCurve(shape(13), { spacing: 0.5, smoothing: 3 });
    expect(curveLength(a)).toBeCloseTo(curveLength(b), 1);
    const plane = planeAt([0, 0, 0], [0, 0, 1]);
    expect(signedArea2(toPlane(a, plane))).toBeCloseTo(signedArea2(toPlane(b, plane)), 1);
  });

  it("solves its five-band system: the result is a least one", () => {
    const rnd = random(9);
    const values = Float64Array.from({ length: 40 }, () => rnd());
    const weight = 3;
    const energy = (q: Float64Array) => {
      let e = 0;
      for (let i = 0; i < q.length; i++) {
        e += (q[i] - values[i]) ** 2;
        if (i > 0 && i < q.length - 1) {
          e += weight * (q[i - 1] - 2 * q[i] + q[i + 1]) ** 2;
        }
      }
      return e;
    };
    const q = stiffen(values, weight);
    expect(q[0]).toBe(values[0]);
    expect(q[39]).toBe(values[39]);
    const least = energy(q);
    for (let i = 1; i < 39; i++) {
      for (const d of [1e-3, -1e-3]) {
        const p = Float64Array.from(q);
        p[i] += d;
        expect(energy(p)).toBeGreaterThan(least);
      }
    }
    expect([...stiffen(Float64Array.from([1, 2]), 5)]).toEqual([1, 2]);
    expect(stiffen(Float64Array.from([0, 3, 0]), 1)[1]).toBeCloseTo(3 / 5, 12);
  });

  it("says what it cannot do", () => {
    const c = circle(100, 10);
    expect(resampleProblem(c, { spacing: 0, smoothing: 0 })).toContain("spacing");
    expect(resampleProblem(c, { spacing: 1e-6, smoothing: 0 })).toContain("points");
    expect(resampleProblem(c, { spacing: 1, smoothing: 100 })).toContain("half");
    expect(resampleProblem(c, { spacing: 1, smoothing: -1 })).toContain("zero or more");
    expect(() => resampleCurve(c, { spacing: -1, smoothing: 0 })).toThrow("spacing");
  });
});

describe("offset", () => {
  const plane = planeAt([0, 0, 0], [0, 0, 1]);
  const area = (c: PolylineCurve) => Math.abs(signedArea2(toPlane(c, plane)));
  const many = (r: ReturnType<typeof offsetCurve>): PolylineCurve[] => {
    if (!Array.isArray(r)) {
      throw new Error(r.reason);
    }
    return r;
  };
  const square = flat([[0, 0], [10, 0], [10, 10], [0, 10]], true);

  it("grows and shrinks a square, with sharp or rounded corners", () => {
    const [out] = many(offsetCurve(square, 1, "sharp"));
    expect(pointCount(out)).toBe(4);
    expect(area(out)).toBeCloseTo(144, 9);
    const [round] = many(offsetCurve(square, 1, "round"));
    expect(area(round)).toBeCloseTo(100 + 40 + Math.PI, 2);
    const [inner] = many(offsetCurve(square, -1, "round"));
    expect(pointCount(inner)).toBe(4);
    expect(area(inner)).toBeCloseTo(64, 9);
    // Outward is outward whichever way the loop runs.
    const back = flat([[0, 10], [10, 10], [10, 0], [0, 0]], true);
    expect(area(many(offsetCurve(back, 1, "sharp"))[0])).toBeCloseTo(144, 9);
    expect(many(offsetCurve(square, -6, "sharp"))).toEqual([]);
  });

  it("works in any plane", () => {
    const [out] = many(offsetCurve(moved(square), 2, "sharp"));
    expect(curveLength(out)).toBeCloseTo(56, 6);
    const fit = planarFit([out]);
    expect("plane" in fit && fit.plane.normal[2]).toBeCloseTo(3 / Math.sqrt(14), 9);
  });

  it("offsets a circle to circles, and a too deep one to nothing", () => {
    for (const d of [1, 5, -1, -4, -9]) {
      const [c] = many(offsetCurve(circle(180, 10), d, "round"));
      for (let k = 0; k < c.points.length; k += 3) {
        expect(Math.hypot(c.points[k], c.points[k + 1])).toBeCloseTo(10 + d, 1);
      }
    }
    expect(many(offsetCurve(circle(180, 10), -10.5, "round"))).toEqual([]);
  });

  it("splits a shape at a narrow place, and closes a gap into a hole", () => {
    // Two squares of 10 joined by a bar 2 wide: shrunk by 1.5 they come apart.
    const bones = flat(
      [[0, 0], [10, 0], [10, 4], [20, 4], [20, 0], [30, 0], [30, 10], [20, 10], [20, 6], [10, 6], [10, 10], [0, 10]],
      true,
    );
    const parts = many(offsetCurve(bones, -1.5, "round"));
    expect(parts).toHaveLength(2);
    for (const p of parts) {
      expect(p.closed).toBe(true);
      // 7 x 7, and the bulge toward the bar that its two corners leave.
      expect(area(p)).toBeCloseTo(49 + 2 * (1.5 - (Math.sqrt(1.25) / 2 + 1.125 * Math.asin(1 / 1.5))), 2);
    }
    expect(many(offsetCurve(bones, -0.5, "round"))).toHaveLength(1);
    // A ring with a slot 2 wide: grown by 1.5 the slot closes.
    const slotted = flat(
      [[0, 0], [20, 0], [20, 9], [16, 9], [16, 4], [4, 4], [4, 16], [16, 16], [16, 11], [20, 11], [20, 20], [0, 20]],
      true,
    );
    const grown = many(offsetCurve(slotted, 1.5, "sharp"));
    expect(grown).toHaveLength(2);
    const areas = grown.map(area).sort((a, b) => a - b);
    const bulge = 2 * (1.5 - (Math.sqrt(1.25) / 2 + 1.125 * Math.asin(1 / 1.5)));
    expect(areas[0]).toBeCloseTo(9 * 9 + bulge, 2);
    expect(areas[1]).toBeCloseTo(23 * 23 - bulge, 2);
  });

  it("keeps a corner round where its point would run into something", () => {
    // Random spiky loops: whatever the corners do, the offset never crosses itself.
    let sharp = 0;
    for (let seed = 1; seed <= 30; seed++) {
      const rnd = random(seed);
      const spikes = 5 + Math.floor(rnd() * 12);
      const star = flat(
        Array.from({ length: 2 * spikes }, (_, i): V2 => {
          const a = (Math.PI * (i + 0.6 * rnd())) / spikes;
          const r = i % 2 === 0 ? 8 + 4 * rnd() : 2 + 3 * rnd();
          return [r * Math.cos(a), r * Math.sin(a)];
        }),
        true,
      );
      if (crosses2([{ xy: toPlane(star, plane), closed: true }])) {
        continue;
      }
      const source: Run2 = { xy: toPlane(star, plane), closed: true };
      for (const d of [0.3, 1, 3, -0.3, -0.8]) {
        const runs = many(offsetCurve(star, d, "sharp")).map((c) => ({ xy: toPlane(c, plane), closed: c.closed }));
        expect(crosses2(runs), `seed ${seed} d ${d}`).toBe(false);
        const round = many(offsetCurve(star, d, "round")).map((c) => ({ xy: toPlane(c, plane), closed: c.closed }));
        expect(crosses2(round), `seed ${seed} d ${d} round`).toBe(false);
        expect(runs.every((r) => r.closed) && round.every((r) => r.closed)).toBe(true);
        for (const { xy } of runs) {
          for (let i = 0; i < xy.length; i += 2) {
            const away = runDistance(source, xy[i], xy[i + 1]);
            expect(away).toBeGreaterThan(Math.abs(d) * (1 - 1e-5));
            expect(away).toBeLessThan(Math.abs(d) * 4.01);
            sharp += away > Math.abs(d) * 1.1 ? 1 : 0;
          }
        }
      }
    }
    expect(sharp).toBeGreaterThan(100);
  });

  it("offsets an open curve to either side, and stays clear of its other end", () => {
    const line = flat([[0, 0], [10, 0], [10, 10]], false);
    const [left] = many(offsetCurve(line, 1, "sharp"));
    const [right] = many(offsetCurve(line, -1, "sharp"));
    expect(left.closed).toBe(false);
    expect(curveLength(left)).toBeCloseTo(18, 9);
    expect(curveLength(right)).toBeCloseTo(22, 9);
    const spiral = flat(
      Array.from({ length: 300 }, (_, i): V2 => {
        const a = i / 20;
        return [(1 + 0.3 * a) * Math.cos(a), (1 + 0.3 * a) * Math.sin(a)];
      }),
      false,
    );
    const run: Run2 = { xy: toPlane(spiral, plane), closed: false };
    for (const d of [0.4, -0.4, 1.2, -1.2]) {
      for (const o of many(offsetCurve(spiral, d, "round"))) {
        const p = toPlane(o, plane);
        for (let i = 0; i < p.length; i += 2) {
          expect(runDistance(run, p[i], p[i + 1])).toBeGreaterThan(Math.abs(d) * (1 - 1e-5));
        }
      }
    }
  });

  it("says why a curve cannot be offset", () => {
    expect(offsetCurve(flat([[0, 0], [5, 0]], false), 1, "round")).toEqual({ reason: "straight" });
    expect(offsetCurve(flat([[0, 0], [4, 4], [4, 0], [0, 4]], true), 1, "round")).toEqual({ reason: "crossing" });
    const bent = makeCurve([0, 0, 0, 5, 0, 0, 5, 5, 2, 0, 5, 0], true)!;
    expect(offsetCurve(bent, 1, "round")).toMatchObject({ reason: "bent" });
  });

  /** A star-shaped blob with bumps. */
  function blob(seed: number, n: number, bumps: number): Run2 {
    const rnd = random(seed);
    const waves = Array.from({ length: bumps }, () => [2 + Math.floor(rnd() * 9), rnd() * 1.6, rnd() * 6.28] as const);
    const xy = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      const a = (2 * Math.PI * i) / n;
      let r = 10;
      for (const [k, h, p] of waves) {
        r += (h / Math.sqrt(k)) * 2.5 * Math.sin(k * a + p);
      }
      r = Math.max(1.5, r);
      xy[2 * i] = r * Math.cos(a);
      xy[2 * i + 1] = r * Math.sin(a);
    }
    return { xy, closed: true };
  }

  it("is the distance away everywhere, and misses nothing, on random shapes", () => {
    let loops = 0;
    for (let seed = 1; seed <= 12; seed++) {
      const n = [40, 160, 600][seed % 3];
      const source = blob(seed, n, 4);
      expect(crosses2([source])).toBe(false);
      for (const d of [0.7, 2.5, 5]) {
        for (const inward of [true, false]) {
          const runs = offsetRun(source, inward ? d : -d, "round");
          for (const run of runs) {
            expect(run.closed, `seed ${seed} d ${d} inward ${inward}`).toBe(true);
            loops++;
            const s = segments2(run.xy, true);
            for (let i = 0; i < s.length; i += 4) {
              for (const t of [0, 0.5]) {
                const x = s[i] + t * (s[i + 2] - s[i]);
                const y = s[i + 1] + t * (s[i + 3] - s[i + 1]);
                const away = runDistance(source, x, y);
                expect(away).toBeGreaterThan(d * (1 - 1e-6));
                expect(away).toBeLessThan(d * (1 + 2e-3));
                expect(inside2(source.xy, x, y)).toBe(inward);
              }
            }
          }
          // The region further than `d` on that side, counted on a grid, has the loops' area.
          const step = 0.5;
          let cells = 0;
          const range = 25;
          for (let x = -range; x < range; x += step) {
            for (let y = -range; y < range; y += step) {
              const inside = inside2(source.xy, x, y);
              const away = runDistance(source, x, y);
              if (inward ? inside && away > d : inside || away < d) {
                cells++;
              }
            }
          }
          const enclosed = Math.abs(runs.reduce((sum, r) => sum + signedArea2(r.xy), 0));
          const perimeter = runs.reduce((sum, r) => sum + curveLength(makeCurve(fromPlane(r.xy, plane), true)!), 0);
          expect(Math.abs(enclosed - cells * step * step)).toBeLessThan(0.06 * perimeter + 1);
        }
      }
    }
    expect(loops).toBeGreaterThan(50);
  }, 60_000);

  /** A wavy outline with f32 points that zigzag by up to `noise`, as a section of a dense mesh has. */
  function outline(n: number, noise: number, seed: number): PolylineCurve {
    const rnd = random(seed);
    return makeCurve(
      Array.from({ length: n }, (_, i) => {
        const a = (2 * Math.PI * i) / n;
        const r = 50 + 6 * Math.sin(7 * a) + 3 * Math.sin(19 * a) + noise * rnd();
        return [Math.fround(r * Math.cos(a)), Math.fround(r * Math.sin(a)), 5];
      }).flat(),
      true,
    )!;
  }

  it("stays one clean loop on fine, noisy outlines", () => {
    // Neighbouring arcs then cross at angles of a thousandth of a radian.
    for (const noise of [0, 1e-4, 2e-3, 5e-2]) {
      const source = outline(2000, noise, 11);
      const fit = fitPlane([source])!;
      const run: Run2 = { xy: toPlane(source, fit.plane), closed: true };
      for (const d of [0.5, 5, -5, 40]) {
        for (const corners of ["round", "sharp"] as const) {
          const loops = many(offsetCurve(source, d, corners));
          const label = `noise ${noise} d ${d} ${corners}`;
          expect(loops, label).toHaveLength(1);
          expect(loops[0].closed, label).toBe(true);
          const out: Run2 = { xy: toPlane(loops[0], fit.plane), closed: true };
          expect(crosses2([out]), label).toBe(false);
          const s = segments2(out.xy, true);
          for (let i = 0; i < s.length; i += 4 * 5) {
            const away = runDistance(run, (s[i] + s[i + 2]) / 2, (s[i + 1] + s[i + 3]) / 2);
            expect(away, label).toBeGreaterThan(Math.abs(d) * (1 - 1e-5));
            if (corners === "round") {
              expect(away, label).toBeLessThan(Math.abs(d) * (1 + 2e-3));
            }
          }
          // What is made can be worked on.
          expect(many(offsetCurve(loops[0], -d / 2, "round")).every((l) => l.closed), label).toBe(true);
        }
      }
    }
  }, 60_000);

  it("takes thousands of points in a moment, noisy ones too", () => {
    for (const noise of [0, 2e-3]) {
      const source = outline(30_000, noise, 2);
      for (const d of [0.5, 5, -5]) {
        const start = performance.now();
        const loops = many(offsetCurve(source, d, "round"));
        // A third of a second on an idle machine.
        expect(performance.now() - start).toBeLessThan(6000);
        expect(loops).toHaveLength(1);
        expect(loops[0].closed).toBe(true);
      }
    }
  });
});
