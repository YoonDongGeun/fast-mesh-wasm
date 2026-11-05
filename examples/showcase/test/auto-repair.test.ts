// The one-click auto-repair button: the library's auto request with the
// panel's live settings, a per-stage report plus a before/after summary, one
// repair call (one grouped undo step), and an analyse-after that feeds the
// #124 overlay refresh.
import { type Analysis, type AutoRepairReport, type AutoRepairRequest, type RepairOp, Scene } from "fast-mesh-wasm";
import { describe, expect, it, vi } from "vitest";
import {
  buildAutoRepairRequest,
  describeAutoRepairStage,
  describeAutoRepairSummary,
  formatAutoRepairReport,
  runAutoRepairPipeline,
} from "../src/panels/auto-repair";
import { summarize, type AnalysisSummary } from "../src/panels/repair-summary";
import { openShellWithPiece } from "../src/scene/demo";
import { countRepairOverlays } from "../src/viewport/repair-overlays";

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

const settings = { weldTolerance: 1e-5, minTriangles: 500, maxEdges: 64, fair: true };

/** One fake report per stage, in pipeline order. */
function stageReports(): AutoRepairReport {
  return {
    weld: { mergedVertices: 3, removedTriangles: 1 },
    removeDegenerate: { removedTriangles: 2, removedVertices: 4 },
    orient: { flippedTriangles: 10, patches: 2, nonOrientablePatches: 0, flippedOutwardComponents: 1 },
    removeSmallComponents: { removedComponents: 1, removedTriangles: 224, removedVertices: 114 },
    fillHoles: {
      filled: [
        {
          id: { minVertex: 0, len: 4, hash: 0n },
          trianglesAdded: 2,
          verticesAdded: 0,
          converged: true,
          weights: null,
          refineCapped: false,
        },
      ],
      skipped: [],
      failed: [],
    },
  };
}

describe("buildAutoRepairRequest", () => {
  it("shares_the_panel_settings", () => {
    expect(buildAutoRepairRequest(settings)).toEqual({
      kind: "auto",
      weldTolerance: 1e-5,
      minTriangles: 500,
      maxEdges: 64,
      fair: true,
    });
    // Another panel state, another pipeline: tolerance, cutoff, limit, fairing all follow.
    expect(buildAutoRepairRequest({ weldTolerance: 0.01, minTriangles: 10, maxEdges: 8, fair: false })).toEqual({
      kind: "auto",
      weldTolerance: 0.01,
      minTriangles: 10,
      maxEdges: 8,
      fair: false,
    });
  });

  it("empty_max_means_any_size", () => {
    expect(buildAutoRepairRequest({ ...settings, maxEdges: 0 }).maxEdges).toBe(Number.POSITIVE_INFINITY);
  });
});

describe("auto-repair report", () => {
  it("describes_each_stage", () => {
    const lines = Object.entries(stageReports()).map(([op, r]) => describeAutoRepairStage(op as RepairOp["op"], r));
    expect(lines[0]).toBe("Weld: merged 3 vertices, removed 1 triangle");
    expect(lines[1]).toBe("Remove degenerate: removed 2 triangles, 4 vertices");
    expect(lines[2]).toBe("Orient: flipped 10 triangles in 2 patches (1 component turned outward)");
    expect(lines[3]).toBe("Remove small components: removed 1 component (224 triangles)");
    expect(lines[4]).toBe("Fill holes: filled 1 hole (2 triangles, 0 vertices added)");
  });

  it("summarizes_before_and_after", () => {
    const before = summarize(
      analysis({
        loops: [loop(4, 0.4), loop(64, 5)],
        components: [
          { triangles: new Uint32Array(224), area: 0.5, closed: true },
          { triangles: new Uint32Array(3836), area: 7, closed: false },
        ],
        orientationConflicts: 3,
        degenerate: 5,
      }),
    );
    const after = summarize(analysis({}));
    expect(describeAutoRepairSummary(before, after)).toBe(
      "2 holes → none, 2 components → 1 component, 5 degenerate triangles → none, 3 orientation conflicts → none",
    );
  });

  it("keeps_counts_when_nothing_changed", () => {
    const clean = summarize(analysis({}));
    expect(describeAutoRepairSummary(clean, clean)).toBe(
      "0 holes → none, 1 component → 1 component, 0 degenerate triangles → none, 0 orientation conflicts → none",
    );
  });

  it("formats_the_button_report", () => {
    expect(formatAutoRepairReport(["Weld: merged 3 vertices, removed 1 triangle"], "2 holes → none")).toBe(
      "Auto repair (one undo step): 2 holes → none. Weld: merged 3 vertices, removed 1 triangle",
    );
  });
});

describe("runAutoRepairPipeline", () => {
  const beforeSummary: AnalysisSummary = summarize(
    analysis({
      loops: [loop(4, 0.4)],
      components: [
        { triangles: new Uint32Array(224), area: 0.5, closed: true },
        { triangles: new Uint32Array(3836), area: 7, closed: false },
      ],
      degenerate: 5,
    }),
  );
  const afterSummary: AnalysisSummary = summarize(analysis({}));

  it("is_one_repair_call_between_two_analyses", async () => {
    const repair = vi.fn(async (_request: AutoRepairRequest) => stageReports());
    let analyses = 0;
    const analyze = vi.fn(async () => (analyses++ === 0 ? beforeSummary : afterSummary));
    const result = await runAutoRepairPipeline({ repair, analyze }, settings);
    // Grouped undo: exactly one repair call with the auto request.
    expect(repair).toHaveBeenCalledTimes(1);
    expect(repair.mock.calls[0][0]).toEqual(buildAutoRepairRequest(settings));
    // Before and after: the second analysis feeds the overlay refresh.
    expect(analyze).toHaveBeenCalledTimes(2);
    expect(result.before).toBe(beforeSummary);
    expect(result.after).toBe(afterSummary);
    expect(result.stageLines).toHaveLength(5);
    expect(result.line).toContain("Auto repair (one undo step): 1 hole → none");
    expect(result.line).toContain("Weld: merged 3 vertices, removed 1 triangle");
  });

  it("a_failure_leaves_one_call_and_no_report", async () => {
    const repair = vi.fn(async (_request: AutoRepairRequest): Promise<AutoRepairReport> => {
      throw new Error("STALE_LOOPS");
    });
    const analyze = vi.fn(async () => beforeSummary);
    await expect(runAutoRepairPipeline({ repair, analyze }, settings)).rejects.toThrow("STALE_LOOPS");
    // Still one call (the engine runs the request all-or-nothing): nothing half-repaired.
    expect(repair).toHaveBeenCalledTimes(1);
    // Only the before analysis ran: no after-analysis on failure.
    expect(analyze).toHaveBeenCalledTimes(1);
  });
});

describe("auto-repair on demo mesh C", () => {
  // The button's report before the pipeline moved into the library (#140),
  // from the TS op list: weld, degenerate, orient, components, fill.
  const BEFORE_140 =
    "Auto repair (one undo step): 2 holes → none, 2 components → 1 component, 0 degenerate triangles → none, " +
    "0 orientation conflicts → none. Weld: merged 0 vertices, removed 0 triangles; " +
    "Remove degenerate: removed 0 triangles, 0 vertices; Orient: flipped 0 triangles in 2 patches; " +
    "Remove small components: removed 1 component (224 triangles); " +
    "Fill holes: filled 2 holes (186 triangles, 61 vertices added)";

  it.each([
    ["the_panel_defaults", settings],
    ["any_size_unfaired", { ...settings, maxEdges: 0, fair: false }],
  ])("reports_the_same_as_the_op_list_with_%s", async (_name, panel) => {
    const data = openShellWithPiece([-2.8, 1, 0], 0.8);
    const scene = new Scene();
    try {
      const mesh = scene.addMesh(data.positions, data.indices);
      const triangles = scene.triangleCount(mesh);
      const result = await runAutoRepairPipeline(
        {
          analyze: async () => summarize(scene.analyze(mesh, { weldTolerance: panel.weldTolerance, maxListed: 64 })),
          repair: async (request) => scene.repair(mesh, request),
        },
        panel,
      );
      expect(result.line).toBe(BEFORE_140);
      expect(scene.triangleCount(mesh)).toBe(3960);
      // One undo step back to the loaded mesh.
      expect(scene.undo()).toBe(true);
      expect(scene.canUndo()).toBe(false);
      expect(scene.triangleCount(mesh)).toBe(triangles);
    } finally {
      scene.free();
    }
  });
});

describe("auto-repair overlay refresh", () => {
  it("counts_drop_after_the_pipeline", () => {
    // What the panel's post-pipeline analyse observes: the issue primitives
    // the #124 overlays draw clear once the mesh is repaired.
    const before = analysis({
      loops: [loop(4, 0.4)],
      nonManifoldEdges: 2,
      nonManifoldEdgeList: [
        [0, 1],
        [2, 3],
      ],
      degenerate: 1,
      degenerateList: Uint32Array.from([4]),
      duplicate: 1,
      duplicateList: Uint32Array.from([5]),
      components: [
        { triangles: new Uint32Array(12), area: 6, closed: true },
        { triangles: new Uint32Array(3), area: 1, closed: false },
      ],
    });
    const after = analysis({});
    expect(countRepairOverlays(before)).toEqual({ holes: 4, nonManifold: 2, degenerate: 2, components: 3 });
    expect(countRepairOverlays(after)).toEqual({ holes: 0, nonManifold: 0, degenerate: 0, components: 0 });
  });
});
