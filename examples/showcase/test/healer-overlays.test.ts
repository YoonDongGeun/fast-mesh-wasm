import type { Analysis } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import { buildSelectiveHealOps, formatHealerDelta } from "../src/panels/MeshHealerSection";
import type { AnalysisSummary } from "../src/panels/repair-summary";
import {
  buildHealerOverlayGroup,
  countHealerOverlays,
  detectInvertedClosedComponents,
  detectOrientationConflictEdges,
  detectSelfIntersectingTriangles,
  detectSliverTriangles,
  disposeHealerOverlayGroup,
  extractDefectHotspots,
} from "../src/viewport/healer-overlays";

function makeAnalysis(partial: Partial<Analysis> = {}): Analysis {
  return {
    loops: [],
    coincidentGroups: 0,
    coincident: [],
    degenerate: 0,
    degenerateList: new Uint32Array(0),
    duplicate: 0,
    duplicateList: new Uint32Array(0),
    nonManifoldEdges: 0,
    nonManifoldEdgeList: [],
    nonManifoldVertices: new Uint32Array(0),
    pinchVertices: new Uint32Array(0),
    components: [],
    orientationConflicts: 0,
    selfIntersects: false,
    ...partial,
  };
}

describe("healer-overlays (#235)", () => {
  it("detects extreme sliver triangles while ignoring equilateral ones", () => {
    // Tri 0: equilateral (aspect ~ 1). Tri 1: long thin needle (aspect >> 20).
    const positions = new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0.5, Math.sqrt(3) / 2, 0,
      0, 2, 0,
      100, 2, 0,
      50, 2.01, 0,
    ]);
    const indices = new Uint32Array([0, 1, 2, 3, 4, 5]);
    const slivers = detectSliverTriangles(positions, indices);
    expect(Array.from(slivers)).toEqual([1]);
  });

  it("detects orientation conflict edges vs consistently oriented interior edges", () => {
    // Consistent pair: (0,1,2) and (2,1,3) traverse shared edge 1-2 in opposite directions.
    const consistent = new Uint32Array([0, 1, 2, 2, 1, 3]);
    expect(detectOrientationConflictEdges(consistent).length).toBe(0);

    // Conflicting pair: (0,1,2) and (1,2,3) both traverse 1->2 forward!
    const conflicting = new Uint32Array([0, 1, 2, 1, 2, 3]);
    expect(Array.from(detectOrientationConflictEdges(conflicting))).toEqual([1, 2]);
  });

  it("detects inward-oriented closed tetrahedron shells via negative signed volume", () => {
    const positions = new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    // Inward-facing winding for unit corner tetrahedron.
    const inward = new Uint32Array([
      0, 1, 2,
      0, 3, 1,
      0, 2, 3,
      1, 3, 2,
    ]);
    const analysis = makeAnalysis({
      components: [{ triangles: new Uint32Array([0, 1, 2, 3]), area: 1, closed: true }],
    });
    const inv = detectInvertedClosedComponents(analysis, positions, inward);
    expect(Array.from(inv)).toEqual([0, 1, 2, 3]);

    // Flip winding -> outward.
    const outward = new Uint32Array([
      0, 2, 1,
      0, 1, 3,
      0, 3, 2,
      1, 2, 3,
    ]);
    expect(detectInvertedClosedComponents(analysis, positions, outward).length).toBe(0);
  });

  it("detects vertex-disjoint crossing triangles and builds + disposes overlay group", () => {
    // Two crossing triangles centered around origin sharing no vertices.
    const positions = new Float32Array([
      -1, -1, 0,
      1, -1, 0,
      0, 1, 0,
      0, 0, -1,
      0, 0, 1,
      0.2, 0.2, 0,
    ]);
    // Wait: tri 1 (3,4,5) intersects tri 0 (0,1,2) along y=[0,0.2], z=0 inside tri 0.
    // Let's push v5 above/across plane z=0 cleanly so it pierces z=0:
    positions[15] = 0.5;
    positions[16] = 0.5;
    positions[17] = 1;
    positions[12] = 0.1;
    positions[13] = 0.0;
    positions[14] = -1;
    positions[9] = -0.2;
    positions[10] = -0.2;
    positions[11] = -1;

    const indices = new Uint32Array([0, 1, 2, 3, 4, 5]);
    const si = detectSelfIntersectingTriangles(positions, indices);
    expect(si.length).toBe(2);

    const analysis = makeAnalysis({
      selfIntersects: true,
      orientationConflicts: 0,
      nonManifoldEdges: 1,
      nonManifoldEdgeList: [[0, 1]],
    });
    const counts = countHealerOverlays(analysis, positions, indices);
    expect(counts.selfIntersections).toBe(2);

    const group = buildHealerOverlayGroup(
      analysis,
      positions,
      indices,
      { holes: true, nonManifold: true, degenerate: true, components: true },
      { selfIntersections: true, inverted: true, slivers: true },
    );
    expect(group).not.toBeNull();
    const names = group!.children.map((c) => c.name);
    expect(names).toContain("overlay-selfIntersections");

    const hotspots = extractDefectHotspots(analysis, positions, indices);
    expect(hotspots.some((h) => h.kind === "selfIntersections")).toBe(true);
    expect(hotspots.some((h) => h.kind === "nonManifold")).toBe(true);
    for (const h of hotspots) {
      expect(h.radius).toBeGreaterThan(0);
    }

    expect(() => disposeHealerOverlayGroup(group!)).not.toThrow();
  });

  it("builds selective repair pipeline ops for checked categories and formats delta readout", () => {
    const ops = buildSelectiveHealOps(
      {
        weld: true,
        removeDegenerate: false,
        orient: true,
        makeManifold: true,
        removeSmallComponents: false,
        fillHoles: true,
        cleanupSlivers: true,
        resolveSelfIntersections: false,
      },
      {
        weldTolerance: 1e-4,
        minTriangles: 100,
        maxEdges: 64,
        fair: true,
        anySize: 1e9,
        selfIntersectionMode: "deleteAndFill",
        siVoxelSize: "",
      },
    );
    expect(ops.map((o) => o.op)).toEqual(["weld", "orient", "makeManifold", "fillHoles", "cleanupSlivers"]);

    const summaryBefore: AnalysisSummary = {
      holes: [{ id: { handle: 1, len: 6 } as never, edges: 6, perimeter: 3 }],
      openChains: 0,
      orientationConflicts: 4,
      components: [
        { triangles: 100, closed: false },
        { triangles: 5, closed: true },
      ],
      nonManifoldEdges: 2,
      nonManifoldVertices: 0,
      pinchVertices: 0,
      degenerate: 3,
      duplicate: 1,
      clean: false,
    };
    const summaryAfter: AnalysisSummary = {
      holes: [],
      openChains: 0,
      orientationConflicts: 0,
      components: [{ triangles: 102, closed: true }],
      nonManifoldEdges: 0,
      nonManifoldVertices: 0,
      pinchVertices: 0,
      degenerate: 0,
      duplicate: 0,
      clean: true,
    };
    const delta = formatHealerDelta(
      summaryBefore,
      summaryAfter,
      { slivers: 5, selfIntersections: 2 },
      { slivers: 0, selfIntersections: 0 },
    );
    expect(delta).toContain("Holes: 1 → 0");
    expect(delta).toContain("Non-manifold: 2 → 0");
    expect(delta).toContain("Slivers: 5 → 0");
    expect(delta).toContain("Self-intersect tris: 2 → 0");
  });
});
