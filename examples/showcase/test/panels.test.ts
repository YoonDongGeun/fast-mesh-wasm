// The pure parts of the repair and collision panels: slider scales and
// limits, the deepest-penetration readout, and the analysis and repair
// report summaries.
import type { Analysis } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import {
  clampTolerance,
  deepestPenetration,
  formatDistance,
  niceScale,
  toleranceLimit,
} from "../src/panels/collision-math";
import { describeReport, summarize } from "../src/panels/repair-summary";

describe("collision math", () => {
  it("nice_scale", () => {
    expect(niceScale(1.9)).toBe(2);
    expect(niceScale(2)).toBe(2);
    expect(niceScale(2.1)).toBe(5);
    expect(niceScale(7)).toBe(10);
    expect(niceScale(0.03)).toBeCloseTo(0.05, 12);
    expect(niceScale(140)).toBe(200);
    expect(niceScale(0)).toBe(1);
    expect(niceScale(Number.NaN)).toBe(1);
  });

  it("tolerance_limits", () => {
    expect(toleranceLimit(1, 0.5)).toBe(0.5);
    expect(toleranceLimit(0.3, 0.5)).toBe(0.3);
    // "Colour from" 0: only the depth bounds the band.
    expect(toleranceLimit(1, 0)).toBe(1);
    // Strictly below the limit, by at least a step.
    expect(clampTolerance(0.2, 1, 0.5, 0.01)).toBe(0.2);
    expect(clampTolerance(0.7, 1, 0.5, 0.01)).toBeCloseTo(0.49, 12);
    expect(clampTolerance(0.7, 1, 0, 0.01)).toBe(0.7);
    expect(clampTolerance(2, 1, 0, 0.01)).toBeCloseTo(0.99, 12);
    expect(clampTolerance(0.5, 1, 0.005, 0.01)).toBe(0);
    expect(clampTolerance(-1, 1, 1, 0.01)).toBe(0);
  });

  it("deepest_penetration", () => {
    expect(deepestPenetration([null, null], 1)).toBeNull();
    // The deepest negative value over both meshes, ignoring far (Infinity) and NaN.
    const a = Float32Array.from([0.5, -0.25, Infinity, Number.NaN]);
    const b = Float32Array.from([-0.125, 2, Infinity]);
    expect(deepestPenetration([a, b], Infinity)).toEqual({ depth: 0.25, atBound: false });
    // No penetration: 0.
    expect(deepestPenetration([Float32Array.from([0.5, Infinity])], 1)).toEqual({ depth: 0, atBound: false });
    // A bounded field: a value at the bound means "at least".
    expect(deepestPenetration([Float32Array.from([Math.fround(-0.3), 1])], 0.3)).toEqual({
      depth: Math.fround(0.3),
      atBound: true,
    });
    expect(deepestPenetration([Float32Array.from([-0.1])], 0.3)?.atBound).toBe(false);
  });

  it("format_distance", () => {
    expect(formatDistance(0)).toBe("0");
    expect(formatDistance(0.123456)).toBe("0.123");
    expect(formatDistance(12.3456)).toBe("12.3");
    expect(formatDistance(1234.5)).toBe("1235");
    expect(formatDistance(0.000012345)).toBe("1.23e-5");
  });
});

/** An analysis with nothing to report, then the given fields. */
function analysis(patch: Partial<Analysis>): Analysis {
  return {
    loops: [],
    coincidentGroups: 0,
    coincident: [],
    degenerate: 0,
    degenerateList: new Uint32Array(),
    duplicate: 0,
    duplicateList: new Uint32Array(),
    nonManifoldEdges: 0,
    nonManifoldEdgeList: [],
    nonManifoldVertices: new Uint32Array(),
    pinchVertices: new Uint32Array(),
    components: [{ triangles: new Uint32Array(12), area: 6, closed: true }],
    orientationConflicts: 0,
    selfIntersects: null,
    ...patch,
  };
}

const loop = (len: number, perimeter: number, closed = true) => ({
  id: { minVertex: 0, len, hash: 0n },
  vertices: new Uint32Array(len),
  perimeter,
  area: 0,
  closed,
  clean: true,
});

describe("repair summaries", () => {
  it("summarize_clean", () => {
    const s = summarize(analysis({}));
    expect(s.clean).toBe(true);
    expect(s.holes).toEqual([]);
    expect(s.components).toEqual([{ triangles: 12, closed: true }]);
  });

  it("summarize_defects", () => {
    const s = summarize(
      analysis({
        loops: [loop(4, 0.4), loop(64, 5), loop(3, 0.1, false)],
        components: [
          { triangles: new Uint32Array(224), area: 0.5, closed: true },
          { triangles: new Uint32Array(3836), area: 7, closed: false },
        ],
        orientationConflicts: 2,
        nonManifoldEdges: 1,
        nonManifoldVertices: new Uint32Array(3),
        degenerate: 5,
      }),
    );
    expect(s.clean).toBe(false);
    // Holes are the closed loops, largest first; open chains are counted apart.
    expect(s.holes).toEqual([
      { edges: 64, perimeter: 5 },
      { edges: 4, perimeter: 0.4 },
    ]);
    expect(s.openChains).toBe(1);
    // Components largest first.
    expect(s.components.map((c) => c.triangles)).toEqual([3836, 224]);
    expect(s.orientationConflicts).toBe(2);
    expect(s.nonManifoldEdges).toBe(1);
    expect(s.nonManifoldVertices).toBe(3);
    expect(s.degenerate).toBe(5);
  });

  it("describe_reports", () => {
    expect(describeReport("weld", { mergedVertices: 3, removedTriangles: 1 })).toBe(
      "Weld: merged 3 vertices, removed 1 triangle",
    );
    expect(describeReport("removeDegenerate", { removedTriangles: 0, removedVertices: 0 })).toBe(
      "Remove degenerate: removed 0 triangles, 0 vertices",
    );
    expect(
      describeReport("orient", {
        flippedTriangles: 10,
        patches: 2,
        nonOrientablePatches: 0,
        flippedOutwardComponents: 1,
      }),
    ).toBe("Orient: flipped 10 triangles in 2 patches (1 component turned outward)");
    expect(
      describeReport("removeSmallComponents", { removedComponents: 1, removedTriangles: 224, removedVertices: 114 }),
    ).toBe("Remove small components: removed 1 component (224 triangles)");
    expect(
      describeReport("fillHoles", {
        filled: [
          {
            id: { minVertex: 0, len: 4, hash: 0n },
            trianglesAdded: 2,
            verticesAdded: 0,
            converged: true,
            weights: null,
          },
          {
            id: { minVertex: 9, len: 64, hash: 1n },
            trianglesAdded: 90,
            verticesAdded: 30,
            converged: true,
            weights: "cotangent",
          },
        ],
        skipped: [{ id: { minVertex: 5, len: 300, hash: 2n }, reason: "bounds" }],
        failed: [],
      }),
    ).toBe("Fill holes: filled 2 holes (92 triangles, 30 vertices added); skipped 1 (1 beyond the size limit)");
  });
});
