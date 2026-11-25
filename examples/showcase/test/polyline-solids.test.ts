// The meshes the polyline tools make (#268), the tube and the extrusion:
// closed, wound outward, of the volume they should have, and accepted by the
// engine as closed (no holes, no non-manifold edges, no inconsistent
// orientation).
import { Scene } from "fast-mesh-wasm";
import { afterEach, describe, expect, it } from "vitest";
import { isClosed } from "../src/features/mesh-edit/ux/text-mesh";
import { extrudeGroup, planarGroups, type PlanarGroup } from "../src/features/polylines/ux/extrude-curves";
import { makeCurve, type PolylineCurve } from "../src/features/polylines/ux/polyline";
import type { SolidMesh } from "../src/features/polylines/ux/solid";
import { buildTube, tubeProblem, tubeTriangles } from "../src/features/polylines/ux/tube";

const scenes: Scene[] = [];
afterEach(() => {
  for (const s of scenes.splice(0)) {
    s.free();
  }
});

function curve(points: readonly (readonly number[])[], closed: boolean): PolylineCurve {
  return makeCurve(points.flat(), closed)!;
}

function circle(n: number, r: number, z = 0, cx = 0): PolylineCurve {
  return curve(
    Array.from({ length: n }, (_, i) => [cx + r * Math.cos((2 * Math.PI * i) / n), r * Math.sin((2 * Math.PI * i) / n), z]),
    true,
  );
}

function square(h: number, z = 0, cx = 0): PolylineCurve {
  return curve([[cx - h, -h, z], [cx + h, -h, z], [cx + h, h, z], [cx - h, h, z]], true);
}

/** Six times the signed volume over six. */
function volume({ positions: p, indices: t }: SolidMesh): number {
  let six = 0;
  for (let k = 0; k < t.length; k += 3) {
    const [a, b, c] = [3 * t[k], 3 * t[k + 1], 3 * t[k + 2]];
    six +=
      p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) +
      p[a + 1] * (p[b + 2] * p[c] - p[b] * p[c + 2]) +
      p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c]);
  }
  return six / 6;
}

/** The engine's own look at a mesh: closed, manifold and oriented. */
function engineAgrees(mesh: SolidMesh, components = 1): void {
  const scene = new Scene();
  scenes.push(scene);
  const world = Float64Array.from(mesh.positions, (v, i) => v + mesh.center[i % 3]);
  const handle = scene.addMesh(world, mesh.indices, { origin: mesh.center });
  const a = scene.analyze(handle);
  expect(a.loops).toHaveLength(0);
  expect(a.nonManifoldEdges).toBe(0);
  expect(a.pinchVertices).toHaveLength(0);
  expect(a.orientationConflicts).toBe(0);
  expect(a.components).toHaveLength(components);
  expect(scene.inspect.measure(handle).volume).toBeCloseTo(volume(mesh), 1);
}

function groups(curves: readonly PolylineCurve[]): PlanarGroup[] {
  const g = planarGroups(curves);
  if (!Array.isArray(g)) {
    throw new Error(g.reason);
  }
  return g;
}

function solid(curves: readonly PolylineCurve[], depth: number, side: "one" | "both" = "one"): SolidMesh {
  const s = extrudeGroup(groups(curves)[0], depth, side);
  if ("reason" in s) {
    throw new Error(s.reason);
  }
  return s;
}

/** The area of a regular polygon with `n` sides about a circle of radius `r`. */
const ngon = (n: number, r: number) => (n / 2) * r * r * Math.sin((2 * Math.PI) / n);

describe("tube", () => {
  it("is a capped cylinder along a straight line", () => {
    const t = buildTube(curve([[1, 2, 3], [1, 2, 13]], false), 2, 16);
    expect(t.indices.length / 3).toBe(tubeTriangles(curve([[1, 2, 3], [1, 2, 13]], false), 16));
    expect(isClosed(t.indices)).toBe(true);
    expect(volume(t)).toBeCloseTo(10 * ngon(16, 2), 3);
    [1, 2, 8].forEach((v, i) => expect(t.center[i]).toBeCloseTo(v, 6));
    expect(t.folds).toBe(0);
    engineAgrees(t);
  });

  it("keeps its thickness around corners", () => {
    const path = curve([[0, 0, 0], [20, 0, 0], [20, 15, 0], [20, 15, 10]], false);
    const t = buildTube(path, 1, 24);
    // A mitred corner adds what it takes: the cross section times the length.
    expect(volume(t)).toBeCloseTo(45 * ngon(24, 1), 2);
    expect(isClosed(t.indices)).toBe(true);
    // Every ring point is a radius away from one of the pieces it is between.
    for (let k = 0; k < 4 * 24; k++) {
      const [x, y, z] = [0, 1, 2].map((a) => t.positions[3 * k + a] + t.center[a]);
      const along = [Math.hypot(y, z), Math.hypot(x - 20, z), Math.hypot(x - 20, y - 15)];
      expect(Math.min(...along.map((d) => Math.abs(d - 1)))).toBeLessThan(1e-5);
    }
    engineAgrees(t);
  });

  it("closes around a closed curve, twisted in space or not", () => {
    const flat = buildTube(circle(100, 10), 1, 20);
    expect(flat.indices.length / 3).toBe(2 * 100 * 20);
    expect(isClosed(flat.indices)).toBe(true);
    expect(volume(flat)).toBeCloseTo(ngon(20, 1) * 2 * Math.PI * 10, 0);
    engineAgrees(flat);
    // A knot: the frame comes back turned, and the rings still meet without a twist.
    const n = 400;
    const knot = curve(
      Array.from({ length: n }, (_, i) => {
        const a = (2 * Math.PI * i) / n;
        return [(6 + 2 * Math.cos(3 * a)) * Math.cos(2 * a), (6 + 2 * Math.cos(3 * a)) * Math.sin(2 * a), 2 * Math.sin(3 * a)];
      }),
      true,
    );
    const sides = 12;
    const t = buildTube(knot, 0.4, sides);
    expect(isClosed(t.indices)).toBe(true);
    expect(t.folds).toBe(0);
    let longest = 0;
    const ring = (i: number, s: number, a: number) => t.positions[3 * (((i % n) + n) % n) * sides + 3 * s + a];
    for (let i = 0; i < n; i++) {
      for (let s = 0; s < sides; s++) {
        longest = Math.max(
          longest,
          Math.hypot(ring(i + 1, s, 0) - ring(i, s, 0), ring(i + 1, s, 1) - ring(i, s, 1), ring(i + 1, s, 2) - ring(i, s, 2)),
        );
      }
    }
    // No jump at the seam or elsewhere: at most a step of the curve, and a bit on the outside of a bend.
    let step = 0;
    for (let i = 0; i < n; i++) {
      const p = knot.points;
      const j = (i + 1) % n;
      step = Math.max(step, Math.hypot(p[3 * j] - p[3 * i], p[3 * j + 1] - p[3 * i + 1], p[3 * j + 2] - p[3 * i + 2]));
    }
    expect(longest).toBeLessThan(1.2 * step);
    engineAgrees(t);
  });

  it("stays closed whatever the curve does, and says where it folds", () => {
    const tight = circle(60, 1);
    const t = buildTube(tight, 3, 16);
    expect(isClosed(t.indices)).toBe(true);
    expect(t.folds).toBeGreaterThan(0);
    const back = buildTube(curve([[0, 0, 0], [5, 0, 0], [0, 0, 0.0001]], false), 1, 8);
    expect(isClosed(back.indices)).toBe(true);
    expect([...back.positions].every(Number.isFinite)).toBe(true);
    const straightBack = buildTube(curve([[0, 0, 0], [5, 0, 0], [1, 0, 0]], false), 1, 8);
    expect([...straightBack.positions].every(Number.isFinite)).toBe(true);
    // A sharp corner is cut off rather than a spike.
    const spike = buildTube(curve([[0, 0, 0], [20, 0, 0], [0, 1, 0]], false), 1, 16);
    for (let k = 0; k < spike.positions.length; k += 3) {
      expect(spike.positions[k] + spike.center[0]).toBeLessThan(20 + 2.5);
    }
    expect(isClosed(spike.indices)).toBe(true);
  });

  it("merges points too close together to have a direction, keeping the ends", () => {
    const rough: number[][] = [];
    for (let i = 0; i <= 100; i++) {
      rough.push([i / 10, 0, 0], [i / 10 + 1e-5, (i % 2) * 2e-5, 0]);
    }
    const t = buildTube(curve(rough, false), 1, 16);
    expect(isClosed(t.indices)).toBe(true);
    expect(volume(t)).toBeCloseTo(10 * ngon(16, 1), 1);
    let min = Infinity;
    let max = -Infinity;
    for (let k = 0; k < t.positions.length; k += 3) {
      min = Math.min(min, t.positions[k] + t.center[0]);
      max = Math.max(max, t.positions[k] + t.center[0]);
    }
    expect(min).toBeCloseTo(0, 2);
    expect(max).toBeCloseTo(10, 2);
  });

  it("says what it cannot build", () => {
    const c = circle(100, 1);
    expect(tubeProblem(c, 1, 16, 1e6)).toBeNull();
    expect(tubeProblem(c, 0, 16, 1e6)).toContain("radius");
    expect(tubeProblem(c, 1, 2, 1e6)).toContain("sides");
    expect(tubeProblem(c, 1, 16.5, 1e6)).toContain("whole number");
    expect(tubeProblem(c, 1, 16, 1000)).toContain("at most 1,000");
  });

  it("takes a long curve in a moment", () => {
    const n = 50_000;
    const helix = curve(
      Array.from({ length: n }, (_, i) => [20 * Math.cos(i / 200), 20 * Math.sin(i / 200), i / 500]),
      false,
    );
    const start = performance.now();
    const t = buildTube(helix, 0.5, 16);
    expect(performance.now() - start).toBeLessThan(3000);
    expect(t.indices.length / 3).toBeGreaterThan(1_500_000);
    expect(t.folds).toBe(0);
  });
});

describe("extrude", () => {
  it("makes a box of a square, up its plane's normal or down, or both ways", () => {
    const up = solid([square(5, 2)], 4);
    expect(up.indices.length / 3).toBe(12);
    expect(isClosed(up.indices)).toBe(true);
    expect(volume(up)).toBeCloseTo(400, 4);
    expect(up.center[2]).toBeCloseTo(4, 6);
    const down = solid([square(5, 2)], -4);
    expect(volume(down)).toBeCloseTo(400, 4);
    expect(down.center[2]).toBeCloseTo(0, 6);
    expect(solid([square(5, 2)], 4, "both").center[2]).toBeCloseTo(2, 6);
    engineAgrees(up);
    engineAgrees(down);
  });

  it("makes curves inside curves holes, and islands inside those", () => {
    const ring = solid([circle(64, 10), circle(32, 6), square(2)], 3);
    expect(isClosed(ring.indices)).toBe(true);
    expect(volume(ring)).toBeCloseTo(3 * (ngon(64, 10) - ngon(32, 6) + 16), 2);
    engineAgrees(ring, 2);
    // In any order.
    expect(volume(solid([square(2), circle(32, 6), circle(64, 10)], 3))).toBeCloseTo(volume(ring), 3);
  });

  it("keeps curves of different planes apart", () => {
    const g = groups([square(5, 0), square(2, 0), square(5, 10), curve([[0, 0, 0], [0, 4, 0], [0, 4, 4], [0, 0, 4]], true)]);
    expect(g.map((x) => x.curves.length)).toEqual([2, 1, 1]);
    const wall = extrudeGroup(g[2], 2, "one");
    expect("reason" in wall).toBe(false);
    if (!("reason" in wall)) {
      expect(volume(wall)).toBeCloseTo(32, 4);
      [1, 2, 2].forEach((v, i) => expect(wall.center[i]).toBeCloseTo(v, 6));
    }
  });

  it("extrudes in a tilted plane, and drops the points of straight runs", () => {
    const n = [1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)];
    const u = [1 / Math.SQRT2, -1 / Math.SQRT2, 0];
    const v = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
    const pts: number[][] = [];
    for (let s = 0; s < 4; s++) {
      for (let i = 0; i < 50; i++) {
        const t = i / 50;
        const [x, y] = [
          [t, 0],
          [1, t],
          [1 - t, 1],
          [0, 1 - t],
        ][s];
        pts.push([0, 1, 2].map((a) => Math.fround(100 + 6 * x * u[a] + 6 * y * v[a])));
      }
    }
    const s = solid([curve(pts, true)], 5);
    expect(isClosed(s.indices)).toBe(true);
    // Sections are f32: straight runs are straight to rounding, and go.
    expect(s.indices.length / 3).toBeLessThan(60);
    expect(volume(s)).toBeCloseTo(180, 1);
    engineAgrees(s);
  });

  it("closes on a thousand random loops, nested ones too", () => {
    let seed = 7;
    const rnd = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    for (let k = 0; k < 1000; k++) {
      const n = 3 + Math.floor(rnd() * 60);
      const loop = (r0: number, r1: number) =>
        curve(
          Array.from({ length: n }, (_, i) => {
            const a = (2 * Math.PI * (i + 0.5 * rnd())) / n;
            const r = r0 + (r1 - r0) * rnd();
            return [Math.fround(r * Math.cos(a)), Math.fround(r * Math.sin(a)), 1];
          }),
          true,
        );
      const curves = n >= 6 ? [loop(8, 12), loop(3, 5)] : [loop(8, 12)];
      const s = extrudeGroup(groups(curves)[0], 1 + rnd(), "one");
      if ("reason" in s) {
        expect(s.reason, `loop ${k}`).toBe("crossing"); // a triangle's sides may pass through the inner loop
        continue;
      }
      expect(isClosed(s.indices), `loop ${k}`).toBe(true);
      expect(volume(s), `loop ${k}`).toBeGreaterThan(0);
    }
  });

  it("says why curves cannot be extruded", () => {
    expect(planarGroups([curve([[0, 0, 0], [1, 0, 0], [1, 1, 0]], false)])).toEqual({ reason: "open" });
    expect(planarGroups([curve([[0, 0, 0], [5, 0, 0], [5, 5, 2], [0, 5, 0]], true)])).toMatchObject({ reason: "bent" });
    expect(planarGroups([curve([[0, 0, 0], [5, 0, 0], [9, 0, 0]], true)])).toEqual({ reason: "straight" });
    const bow = groups([curve([[0, 0, 0], [4, 4, 0], [4, 0, 0], [0, 4, 0]], true)]);
    expect(extrudeGroup(bow[0], 1, "one")).toEqual({ reason: "crossing" });
    const overlap = groups([square(5), square(5, 0, 4)]);
    expect(extrudeGroup(overlap[0], 1, "one")).toEqual({ reason: "crossing" });
    const rough = groups([
      curve(
        Array.from({ length: 60_000 }, (_, i) => {
          const a = (2 * Math.PI * i) / 60_000;
          const r = 1000 + (i % 2);
          return [r * Math.cos(a), r * Math.sin(a), 0];
        }),
        true,
      ),
    ]);
    expect(extrudeGroup(rough[0], 1, "one")).toEqual({ reason: "points", points: 60_000 });
  });

  it("takes a detailed outline in a moment", () => {
    const n = 20_000;
    const fine = curve(
      Array.from({ length: n }, (_, i) => {
        const a = (2 * Math.PI * i) / n;
        const r = 50 + 3 * Math.sin(40 * a);
        return [r * Math.cos(a), r * Math.sin(a), 0];
      }),
      true,
    );
    const start = performance.now();
    const s = solid([fine], 10);
    expect(performance.now() - start).toBeLessThan(5000);
    expect(isClosed(s.indices)).toBe(true);
  });
});
