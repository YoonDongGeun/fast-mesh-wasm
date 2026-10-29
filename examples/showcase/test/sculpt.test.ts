// The sculpt stroke: dab spacing and ops, and one press-and-hold stroke
// through a fake applier (lazy step, busy and spacing guards, one undo
// step per stroke where the scene nests steps).
import type { MeshHandle } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import {
  GrabAnchor,
  SculptStroke,
  type SculptBrush,
  dabOp,
  dabSpacing,
  faceNormal,
  farEnough,
  grabLocalDelta,
  interpolateHit,
  meanEdgeLength,
  remeshTooBig,
  worldPerPixel,
} from "../src/viewport/sculpt";

const brush: SculptBrush = {
  mesh: 7 as unknown as MeshHandle,
  op: "inflate",
  radius: 1,
  strength: 0.1,
  falloff: "smooth",
  cycles: 3,
  direction: [0, 0, 0],
};

function deferred() {
  let resolve!: () => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<void>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Lets a dab's promise (and the stroke's handlers) settle mid-stroke. */
function tick(): Promise<void> {
  return new Promise((r) => setTimeout(r, 0));
}

describe("sculpt dabs", () => {
  it("spaces_dabs_tightly", () => {
    expect(dabSpacing(1)).toBe(0.15);
    expect(dabSpacing(0.4)).toBeCloseTo(0.06, 10);
    expect(dabSpacing(0)).toBeGreaterThan(0);
    expect(dabSpacing(-2)).toBeGreaterThan(0);
  });

  it("far_enough_guards_spacing", () => {
    expect(farEnough([0, 0, 0], [0.25, 0, 0], 0.25)).toBe(true);
    expect(farEnough([0, 0, 0], [0.24, 0, 0], 0.25)).toBe(false);
    expect(farEnough([1, 1, 1], [1, 1, 1], 0.25)).toBe(false);
  });

  it("dab_op_builds_inflate_relax_and_grab", () => {
    expect(dabOp(brush, [1, 2, 3])).toEqual({
      op: "inflate",
      center: [1, 2, 3],
      radius: 1,
      falloff: "smooth",
      strength: 0.1,
    });
    expect(dabOp({ ...brush, op: "relax" }, [1, 2, 3])).toEqual({
      op: "relax",
      center: [1, 2, 3],
      radius: 1,
      falloff: "smooth",
      strength: 0.1,
      cycles: 3,
    });
    expect(dabOp({ ...brush, op: "grab", direction: [1, 2, 3] }, [1, 2, 3])).toEqual({
      op: "grab",
      center: [1, 2, 3],
      radius: 1,
      falloff: "smooth",
      strength: 0.1,
      direction: [1, 2, 3],
    });
  });

  it("world_per_pixel_scales_with_depth", () => {
    const wpp = worldPerPixel(10, 45, 800);
    expect(wpp).toBeCloseTo((2 * 10 * Math.tan(Math.PI / 8)) / 800, 12);
    expect(worldPerPixel(5, 45, 800)).toBeCloseTo(wpp / 2, 12);
  });

  it("grab_delta_follows_the_screen", () => {
    // Identity orientation: right is +x, up is +y, screen y grows down.
    const d = grabLocalDelta(10, -5, 0.1, [1, 0, 0], [0, 1, 0], [0, 0, 0, 1]);
    expect(d[0]).toBeCloseTo(1, 12);
    expect(d[1]).toBeCloseTo(0.5, 12);
    expect(d[2]).toBeCloseTo(0, 12);
  });

  it("grab_delta_lands_in_the_mesh_frame", () => {
    // The mesh is turned 90° about z (world +x is its -y): the conjugate
    // quaternion [0, 0, -√½, √½] carries the world delta back.
    const q = Math.SQRT1_2;
    const d = grabLocalDelta(10, 0, 0.1, [1, 0, 0], [0, 1, 0], [0, 0, -q, q]);
    expect(d[0]).toBeCloseTo(0, 10);
    expect(d[1]).toBeCloseTo(-1, 10);
    expect(d[2]).toBeCloseTo(0, 10);
  });

  it("mean_edge_counts_each_edge_once", () => {
    // Two triangles sharing the (1, 2) edge: 5 edges (4 of length 1, one √2).
    const positions = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0];
    expect(meanEdgeLength(positions, [0, 1, 2, 1, 3, 2])).toBeCloseTo((4 + Math.SQRT2) / 5, 12);
    // Direction is insignificant: the shared edge counts once either way.
    expect(meanEdgeLength(positions, [0, 1, 2, 2, 3, 1])).toBeCloseTo((4 + Math.SQRT2) / 5, 12);
  });

  it("mean_edge_skips_dead_and_degenerate", () => {
    // A dead slot (0, 0, 0) and a degenerate triangle add no edges.
    const positions = [0, 0, 0, 3, 0, 0, 0, 4, 0];
    expect(meanEdgeLength(positions, [0, 0, 0, 0, 1, 2, 2, 2, 2])).toBeCloseTo((3 + 4 + 5) / 3, 12);
    expect(meanEdgeLength(positions, [0, 0, 0])).toBeNull();
    expect(meanEdgeLength([], [])).toBeNull();
  });

  it("face_normal_and_hit_interpolation", () => {
    expect(faceNormal([0, 0, 0, 1, 0, 0, 0, 1, 0], 0, 1, 2)).toEqual([0, 0, 1]);
    const at = interpolateHit([0, 0, 0, 2, 0, 0, 0, 2, 0], [0, 1, 2], [0.5, 0.25, 0.25]);
    expect(at[0]).toBeCloseTo(0.5, 10);
    expect(at[1]).toBeCloseTo(0.5, 10);
    expect(at[2]).toBe(0);
  });
});

describe("SculptStroke", () => {
  it("one_step_per_stroke", async () => {
    const calls: string[] = [];
    const stroke = new SculptStroke(
      brush,
      {
        beginStep: () => void calls.push("begin"),
        endStep: () => void calls.push("end"),
        cancelStep: () => void calls.push("cancel"),
      },
      () => {
        calls.push("apply");
        return Promise.resolve();
      },
    );
    expect(stroke.dab([0, 0, 0])).toBe("stamped");
    expect(stroke.dab([1, 0, 0])).toBe("busy");
    stroke.end();
    await stroke.done;
    expect(stroke.dabs).toBe(1);
    expect(calls).toEqual(["begin", "apply", "end"]);
    stroke.end();
    await stroke.done;
    expect(calls).toEqual(["begin", "apply", "end"]);
  });

  it("skips_dabs_too_close", async () => {
    let applies = 0;
    const stroke = new SculptStroke(brush, {}, () => {
      applies++;
      return Promise.resolve();
    });
    expect(stroke.dab([0, 0, 0])).toBe("stamped");
    await tick();
    expect(stroke.dab([0.1, 0, 0])).toBe("skipped");
    expect(stroke.dab([0.1, 0, 0], true)).toBe("stamped");
    expect(applies).toBe(2);
    stroke.end();
    await stroke.done;
  });

  it("closes_nothing_without_a_dab", async () => {
    const calls: string[] = [];
    const stroke = new SculptStroke(
      brush,
      {
        beginStep: () => void calls.push("begin"),
        endStep: () => void calls.push("end"),
        cancelStep: () => void calls.push("cancel"),
      },
      () => Promise.resolve(),
    );
    stroke.end();
    await stroke.done;
    expect(calls).toEqual([]);
    expect(stroke.dab([0, 0, 0])).toBe("ended");
  });

  it("cancels_when_every_dab_fails", async () => {
    const calls: string[] = [];
    const stroke = new SculptStroke(
      brush,
      {
        beginStep: () => void calls.push("begin"),
        endStep: () => void calls.push("end"),
        cancelStep: () => void calls.push("cancel"),
      },
      () => Promise.reject(new Error("removed")),
    );
    expect(stroke.dab([0, 0, 0])).toBe("stamped");
    stroke.end();
    await stroke.done;
    expect(calls).toEqual(["begin", "cancel"]);
  });

  it("serializes_dabs_behind_a_slow_apply", async () => {
    const gate = deferred();
    const seen: string[] = [];
    const stroke = new SculptStroke(
      brush,
      {},
      () => {
        seen.push("apply");
        return gate.promise;
      },
    );
    expect(stroke.dab([0, 0, 0])).toBe("stamped");
    expect(stroke.dab([5, 0, 0])).toBe("busy");
    expect(seen).toEqual(["apply"]);
    gate.resolve();
    await tick();
    expect(stroke.dab([5, 0, 0])).toBe("stamped");
    stroke.end();
    await stroke.done;
    expect(seen).toEqual(["apply", "apply"]);
  });

  it("creeps_toward_a_distant_pointer", async () => {
    const centers: (readonly [number, number, number])[] = [];
    const stroke = new SculptStroke(
      brush,
      {},
      (op) => {
        centers.push((op as { center: readonly [number, number, number] }).center);
        return Promise.resolve();
      },
    );
    expect(stroke.dab([0, 0, 0])).toBe("stamped");
    await tick();
    // One spacing step toward the pointer, not the whole jump.
    expect(stroke.dab([0.3, 0, 0])).toBe("stamped");
    await tick();
    expect(centers.length).toBe(2);
    expect(centers[1][0]).toBeCloseTo(0.15, 10);
    // The next dab lands on the pointer itself.
    expect(stroke.dab([0.3, 0, 0])).toBe("stamped");
    await tick();
    expect(centers[2][0]).toBeCloseTo(0.3, 10);
    stroke.end();
    await stroke.done;
  });

  it("jumps_when_the_pointer_teleports", async () => {
    const centers: (readonly [number, number, number])[] = [];
    const stroke = new SculptStroke(
      brush,
      {},
      (op) => {
        centers.push((op as { center: readonly [number, number, number] }).center);
        return Promise.resolve();
      },
    );
    expect(stroke.dab([0, 0, 0])).toBe("stamped");
    await tick();
    expect(stroke.dab([10, 0, 0])).toBe("stamped");
    stroke.end();
    await stroke.done;
    expect(centers[1]).toEqual([10, 0, 0]);
  });

  it("tracks_bounds_covering_the_stroke", async () => {
    const stroke = new SculptStroke(brush, {}, () => Promise.resolve());
    expect(stroke.bounds()).toBeNull();
    expect(stroke.dab([0, 0, 0])).toBe("stamped");
    await tick();
    expect(stroke.dab([10, 0, 0])).toBe("stamped");
    const b = stroke.bounds();
    expect(b?.center).toEqual([5, 0, 0]);
    // Half the span, plus twice the brush radius (cover + growth margin).
    expect(b?.radius).toBeCloseTo(7, 10);
    stroke.end();
    await stroke.done;
  });

  it("grab_anchor_holds_the_press_point", () => {
    const anchor = new GrabAnchor([1, 2, 3]);
    expect(anchor.center).toEqual([1, 2, 3]);
    // Later raycast hits never move it: only the consumed drag does.
    expect(anchor.center).toEqual([1, 2, 3]);
    // A copy: mutating it leaves the anchor alone.
    const c = anchor.center;
    c[0] = 99;
    expect(anchor.center).toEqual([1, 2, 3]);
  });

  it("grab_anchor_advances_with_the_drag", () => {
    const anchor = new GrabAnchor([1, 2, 3]);
    anchor.advance([0.5, 0, 0]);
    anchor.advance([0, -1, 2]);
    expect(anchor.center).toEqual([1.5, 1, 5]);
  });

  it("reads_a_getter_brush_live", async () => {
    let radius = 1;
    const seen: number[] = [];
    const stroke = new SculptStroke(
      () => ({ ...brush, radius }),
      {},
      (op) => {
        seen.push((op as { radius: number }).radius);
        return Promise.resolve();
      },
    );
    expect(stroke.dab([0, 0, 0])).toBe("stamped");
    await tick();
    radius = 2;
    expect(stroke.dab([1, 0, 0])).toBe("stamped");
    stroke.end();
    await stroke.done;
    expect(seen).toEqual([1, 2]);
  });
});

describe("remesh gate", () => {
  it("stands remeshes down past the triangle limit", () => {
    expect(remeshTooBig(27_584)).toBe(false);
    expect(remeshTooBig(100_000)).toBe(false);
    expect(remeshTooBig(100_001)).toBe(true);
    expect(remeshTooBig(302_174)).toBe(true);
  });
});
