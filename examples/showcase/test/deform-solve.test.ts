// The free-form solvers' inputs and outputs (#193): the deform domain guard,
// the ARAP and Kelvinlet options from the deform state, the Kelvinlet loads
// from the panel and from a viewport pull, the vertex picks the panel seeds
// anchors, handles, pins and centers with, the report line, and the mask
// protection a viewport drag solves under.
import type { FreeformReport, MeshHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { describe, expect, it } from "vitest";
import { createDefaultWave6State } from "../src/app/state/wave6-state";
import { dragProtection } from "../src/features/sculpt/ux/deform-protection";
import {
  anchorPins,
  arapOptions,
  bottomBandAnchors,
  boundsCenter,
  buildKelvinletLoad,
  buildKelvinletLoadFromDrag,
  formatError,
  formatFreeformReport,
  handleTargetFromDelta,
  kelvinletOptions,
  requireDeformDomain,
} from "../src/features/sculpt/ux/deform-solve";

/** A 5 x 5 grid in the XY plane over [-1, 1], row by row from y = -1. */
function grid(): Float32Array {
  const out = new Float32Array(25 * 3);
  for (let iy = 0; iy < 5; iy++) {
    for (let ix = 0; ix < 5; ix++) {
      const v = iy * 5 + ix;
      out[v * 3] = -1 + ix * 0.5;
      out[v * 3 + 1] = -1 + iy * 0.5;
    }
  }
  return out;
}

const report = (contacts: number): FreeformReport => ({
  verticesEvaluated: 25,
  verticesMoved: 7,
  maxDisplacement: 0.123456,
  iterations: 3,
  converged: true,
  selfCollisionContactsResolved: contacts,
});

describe("deform solver plumbing", () => {
  it("requires_the_deform_domain", () => {
    const bare = { scene: {} } as unknown as SceneStore;
    expect(() => requireDeformDomain(bare)).toThrow("Deform domain is unavailable on this scene.");
    const domain = {};
    const withDomain = { scene: { deform: domain } } as unknown as SceneStore;
    expect(requireDeformDomain(withDomain)).toBe(domain);
  });

  it("formats_the_report_and_errors", () => {
    expect(formatFreeformReport("FFD", report(0))).toBe("FFD: moved 7/25 verts, max disp 0.1235");
    expect(formatFreeformReport("XPBD", report(4))).toBe(
      "XPBD: moved 7/25 verts, max disp 0.1235, 4 self-collisions resolved",
    );
    expect(formatError(new Error("boom"))).toBe("boom");
    expect(formatError("plain")).toBe("plain");
  });

  it("builds_arap_and_kelvinlet_options_from_the_state", () => {
    const deform = {
      ...createDefaultWave6State().deform,
      arapAnchors: [0, 3],
      arapHandles: [{ vertex: 7, target: [1, 2, 3] as [number, number, number] }],
      kelvinletSymY: true,
    };
    const protect = { vertices: Uint32Array.from([2]) };
    expect(arapOptions(deform, 15, protect)).toEqual({
      anchors: [{ vertex: 0 }, { vertex: 3 }],
      handles: deform.arapHandles,
      weights: "cotangent",
      iterations: 15,
      protect,
    });
    const load = { kind: "grab" as const, force: [0, 1, 0] as [number, number, number] };
    expect(kelvinletOptions(deform, [1, 0, 0], load, undefined)).toEqual({
      center: [1, 0, 0],
      load,
      epsilon: 0.4,
      nu: 0.35,
      scale: "bi",
      symmetry: { x: false, y: true, z: false },
      protect: undefined,
    });
  });
});

describe("kelvinlet loads", () => {
  it("maps_the_panel_inputs_per_kind", () => {
    const force: [number, number, number] = [0.1, 0.2, 0.3];
    expect(buildKelvinletLoad("grab", force, 0.5)).toEqual({ kind: "grab", force });
    expect(buildKelvinletLoad("twist", force, 0.5)).toEqual({ kind: "twist", torque: force });
    expect(buildKelvinletLoad("scale", force, 0.5)).toEqual({ kind: "scale", magnitude: 0.5 });
    expect(buildKelvinletLoad("pinch", force, 0.5)).toEqual({
      kind: "pinch",
      forceDipole: [
        [0.1, 0, 0],
        [0, 0.2, 0],
        [0, 0, 0.3],
      ],
    });
  });

  it("maps_a_pull_per_kind", () => {
    expect(buildKelvinletLoadFromDrag("grab", [0.3, 0.4, 0])).toEqual({ kind: "grab", force: [0.3, 0.4, 0] });
    expect(buildKelvinletLoadFromDrag("twist", [0.3, 0.4, 0])).toEqual({ kind: "twist", torque: [0.6, 0.8, 0] });
    expect(buildKelvinletLoadFromDrag("pinch", [1, 2, 3])).toEqual({
      kind: "pinch",
      forceDipole: [
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3],
      ],
    });
  });

  it("signs_a_scale_pull_by_its_x_plus_y", () => {
    const magnitude = (delta: [number, number, number]) => {
      const load = buildKelvinletLoadFromDrag("scale", delta);
      return load.kind === "scale" ? load.magnitude : NaN;
    };
    expect(magnitude([0.3, 0.4, 0])).toBeCloseTo(0.5, 9);
    expect(magnitude([-0.3, -0.4, 0])).toBeCloseTo(-0.5, 9);
    expect(magnitude([-0.3, 0.3, 0])).toBeCloseTo(Math.hypot(0.3, 0.3), 9);
  });
});

describe("deform vertex picks", () => {
  it("pins_the_lowest_fifth_and_keeps_the_top_vertex", () => {
    expect(bottomBandAnchors(grid())).toEqual({ pinned: [0, 1, 2, 3, 4], topVertex: 24 });
    expect(bottomBandAnchors(new Float32Array(0))).toBeNull();
  });

  it("targets_a_handle_by_delta_and_rejects_out_of_range", () => {
    const positions = Float32Array.from([1, 2, 3, 4, 5, 6]);
    expect(handleTargetFromDelta(positions, 1, [0.5, 0, -1])).toEqual([4.5, 5, 5]);
    expect(() => handleTargetFromDelta(positions, 2, [0, 0, 0])).toThrow("Vertex 2 is out of range.");
    expect(() => handleTargetFromDelta(positions, -1, [0, 0, 0])).toThrow("Vertex -1 is out of range.");
  });

  it("pins_in_range_anchors_where_they_are", () => {
    const positions = Float32Array.from([1, 2, 3, 4, 5, 6]);
    expect(anchorPins(positions, [1, 5, 0])).toEqual([
      { vertex: 1, target: [4, 5, 6] },
      { vertex: 0, target: [1, 2, 3] },
    ]);
  });

  it("centers_on_the_bounds_rounded_for_the_inputs", () => {
    expect(boundsCenter(Float32Array.from([0, 0, 0, 2, 4, -2]))).toEqual([1, 2, -1]);
    expect(boundsCenter([0, 0, 0, 0.123456, 0, 0])).toEqual([0.0617, 0, 0]);
    expect(boundsCenter(new Float32Array(0))).toBeNull();
  });
});

describe("drag protection", () => {
  it("freezes_the_mask_on_the_mesh_while_protected", () => {
    const mesh = {} as MeshHandle;
    const other = {} as MeshHandle;
    const mask = { mesh, vertices: Uint32Array.from([2, 5]), slots: 25 };
    expect(dragProtection(false, mask, mesh)).toBeUndefined();
    expect(dragProtection(true, null, mesh)).toBeUndefined();
    expect(dragProtection(true, mask, other)).toBeUndefined();
    const protect = dragProtection(true, mask, mesh)!;
    expect(Array.from(protect.vertices)).toEqual([2, 5]);
    expect(Array.from(protect.weights!)).toEqual([0, 0]);
  });
});
