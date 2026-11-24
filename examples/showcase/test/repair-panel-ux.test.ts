// The repair panel's pure behaviour: how its number fields parse, the
// summary holes matched to their raw analysis loops, the overlay issue
// counts, the Auto repair note, the continuity grades, and the healer's
// defect reading and click-to-focus.
import type { Analysis, BoundaryLoop, LoopId, MeshHandle } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import { AppState } from "../src/app/state/app-state";
import { HEALER_OVERLAY_KINDS, isHealerOverlayKind } from "../src/features/mesh-repair/ux/healer-defects";
import { focusDefectHotspot, healerCountsOf, readHealerDefects } from "../src/features/mesh-repair/ux/mesh-healer";
import { continuityGrade } from "../src/features/mesh-repair/ux/repair-actions";
import {
  type AnalysisSummary,
  AUTO_REPAIR_PRESET_LABELS,
  autoRepairNote,
  overlayIssueCounts,
  summarize,
} from "../src/features/mesh-repair/ux/repair-summary";
import { holeTargetsOf } from "../src/features/mesh-repair/ux/use-repair-loop-picks";
import { voxelSizeField, wholeAtLeast, wholeNumber } from "../src/features/mesh-repair/ux/use-repair-options";

function loopId(minVertex: number, len: number): LoopId {
  return { minVertex, len, hash: 0n } as unknown as LoopId;
}

function loop(minVertex: number, len: number, closed = true): BoundaryLoop {
  return {
    id: loopId(minVertex, len),
    vertices: Uint32Array.from({ length: len }, (_, k) => minVertex + k),
    perimeter: len,
    area: 0,
    closed,
    clean: true,
  };
}

function analysis(patch: Partial<Analysis> = {}): Analysis {
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

describe("repair option fields", () => {
  it("whole numbers at least a minimum, invalid entries reading as the minimum", () => {
    expect(wholeAtLeast("7.6", 1)).toBe(8);
    expect(wholeAtLeast("0", 1)).toBe(1);
    expect(wholeAtLeast("-4", 0)).toBe(0);
    expect(wholeAtLeast("", 0)).toBe(0);
    expect(wholeAtLeast("abc", 1)).toBe(1);
    expect(wholeAtLeast("128", 0)).toBe(128);
  });

  it("whole numbers of either sign", () => {
    expect(wholeNumber("-2.6")).toBe(-3);
    expect(wholeNumber("4.4")).toBe(4);
    expect(wholeNumber("")).toBe(0);
    expect(wholeNumber("x")).toBe(0);
  });

  it("an optional voxel size only when positive", () => {
    expect(voxelSizeField("0.5")).toEqual({ voxelSize: 0.5 });
    expect(voxelSizeField("")).toEqual({});
    expect(voxelSizeField("0")).toEqual({});
    expect(voxelSizeField("-1")).toEqual({});
    expect(voxelSizeField("auto")).toEqual({});
  });
});

describe("hole targets", () => {
  it("matches summary holes (largest first) to their raw loop indices and vertices", () => {
    const raw = [loop(10, 3), loop(0, 2, false), loop(20, 5)];
    const s = summarize(analysis({ loops: raw }));
    const targets = holeTargetsOf(s, raw);
    expect(targets.map((t) => [t.displayIndex, t.loopIndex, t.edges])).toEqual([
      [0, 2, 5],
      [1, 0, 3],
    ]);
    expect(Array.from(targets[0]!.vertices ?? [])).toEqual([20, 21, 22, 23, 24]);
  });

  it("falls back to the summary's place when the raw analysis lacks the loop", () => {
    const s = summarize(analysis({ loops: [loop(10, 3)] }));
    const targets = holeTargetsOf(s, []);
    expect(targets).toHaveLength(1);
    expect(targets[0]!.loopIndex).toBe(0);
    expect(targets[0]!.vertices).toBeUndefined();
    expect(holeTargetsOf(null, [])).toEqual([]);
  });
});

describe("repair summary readouts", () => {
  it("counts each overlay kind's issues, beyond the largest component", () => {
    const s: AnalysisSummary = {
      ...summarize(analysis()),
      holes: [{ id: loopId(0, 3), edges: 3, perimeter: 3 }],
      nonManifoldEdges: 2,
      degenerate: 1,
      duplicate: 4,
      components: [
        { triangles: 10, closed: true },
        { triangles: 2, closed: false },
      ],
    };
    expect(overlayIssueCounts(s)).toEqual({ holes: 1, nonManifold: 2, degenerate: 5, components: 1 });
    expect(overlayIssueCounts({ ...s, components: [] }).components).toBe(0);
  });

  it("notes the default pipeline's order, or what a preset tailors", () => {
    expect(autoRepairNote("general", "auto")).toMatch(/^Order: weld → remove degenerate/);
    expect(autoRepairNote("print3d", "auto")).toContain(`Preset ${AUTO_REPAIR_PRESET_LABELS.print3d}:`);
    expect(autoRepairNote("print3d", "auto")).toContain("volumetric fallback");
    expect(autoRepairNote("scan3d", "auto")).toContain("outlier pruning");
    expect(autoRepairNote("cadImport", "auto")).toContain("sharp CAD boundaries");
    // An intersection override leaves the default pipeline.
    expect(autoRepairNote("general", "voxelHeal")).toContain("standard 5-stage cleanup");
  });

  it("grades continuity G0, G1, G2", () => {
    expect(continuityGrade("flat")).toBe("G0");
    expect(continuityGrade("tangent")).toBe("G1");
    expect(continuityGrade("curvature")).toBe("G2");
  });
});

describe("mesh healer reading", () => {
  const positions = Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const indices = Uint32Array.from([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]);
  const mesh = { __mesh: 1 } as unknown as MeshHandle;
  const scene = { views: () => ({ positions: () => positions, indices: () => indices }) };
  const gone = {
    views: (): never => {
      throw new Error("removed");
    },
  };

  it("tells the healer's own overlay kinds from the repair overlays'", () => {
    for (const kind of HEALER_OVERLAY_KINDS) {
      expect(isHealerOverlayKind(kind)).toBe(true);
    }
    expect(isHealerOverlayKind("holes")).toBe(false);
    expect(isHealerOverlayKind("components")).toBe(false);
  });

  it("reads nothing without an analysis or once the mesh is gone", () => {
    const empty = { hotspots: [], extraCounts: { selfIntersections: 0, inverted: 0, slivers: 0 } };
    expect(readHealerDefects(null, scene, mesh)).toEqual(empty);
    expect(readHealerDefects(analysis(), gone, mesh)).toEqual(empty);
    expect(healerCountsOf(gone, mesh, analysis())).toBeNull();
  });

  it("reads hotspots and counts from the mesh buffers", () => {
    const a = analysis({ loops: [loop(0, 3)] });
    const read = readHealerDefects(a, scene, mesh);
    expect(read.hotspots.map((h) => h.kind)).toContain("holes");
    expect(read.extraCounts).toEqual(healerCountsOf(scene, mesh, a));
  });

  it("focus turns on the overlay drawing the spot's kind", () => {
    const app = new AppState();
    const calls: string[] = [];
    app.setHealerOverlay = (kind, on) => void calls.push(`healer:${kind}:${on}`);
    app.setRepairOverlay = (kind, on) => void calls.push(`repair:${kind}:${on}`);
    app.focusDefect = (_mesh, center, radius) => void calls.push(`focus:${center.join(",")}:${radius}`);
    focusDefectHotspot(app, mesh, { id: "s", kind: "slivers", label: "", count: 1, center: [1, 2, 3], radius: 4 });
    focusDefectHotspot(app, mesh, { id: "h", kind: "holes", label: "", count: 1, center: [0, 0, 0], radius: 1 });
    expect(calls).toEqual(["healer:slivers:true", "focus:1,2,3:4", "repair:holes:true", "focus:0,0,0:1"]);
  });
});
