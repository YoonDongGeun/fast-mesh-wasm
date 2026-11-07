// The sculpt stroke: dab spacing and ops, one press-and-hold stroke
// through a fake applier (lazy step, queueing and spacing guards, one undo
// step per stroke where the scene nests steps), and the brush picker's
// tool-to-brush mapping (what the toolbox arms).
//
// Headless covers the stroke math and the picker mapping here. The rest is
// manual, through the canvas (the e2e `sculpt-surface` path arms
// `[data-action="sculpt-tool-inflate"]` and checks the surface returns):
// the hover ball follows the surface point, press-and-hold paints dabs
// (holding still builds up, fast moves leave no dotted trails),
// grab drags the surface along with the pointer, and Esc disarms.
import type { MeshHandle } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import {
  DEFAULT_STRENGTHS,
  LIVE_REMESH_MIN_DABS,
  LIVE_REMESH_MIN_MS,
  RELAX_PASSES,
  SculptStroke,
  checkMask,
  invertMaskIds,
  liveRemeshDue,
  maskBlocksDab,
  maskUsable,
  MaskError,
  paintMaskIds,
  sculptBrush,
  type SculptBrush,
  type SculptMask,
  type SculptTool,
  dabOp,
  dabSpacing,
  faceNormal,
  farEnough,
  grabLocalDelta,
  interpolateHit,
  meanEdgeLength,
  remeshTooBig,
  sculptOpForTool,
  unionMaskIds,
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
    expect(stroke.dab([0, 0, 0], true)).toBe("busy");
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

  it("queues_dabs_behind_a_slow_apply_and_sends_them_together", async () => {
    const gate = deferred();
    const seen: number[] = [];
    const stroke = new SculptStroke(
      brush,
      {},
      (ops) => {
        seen.push(ops.length);
        return gate.promise;
      },
    );
    expect(stroke.dab([0, 0, 0])).toBe("stamped");
    // Nothing is dropped: one spacing is 0.15, so these are three more dabs.
    expect(stroke.dab([0.15, 0, 0])).toBe("stamped");
    expect(stroke.dab([0.45, 0, 0])).toBe("stamped");
    expect(stroke.dabs).toBe(4);
    expect(seen).toEqual([1]);
    gate.resolve();
    await tick();
    expect(seen).toEqual([1, 3]);
    stroke.end();
    await stroke.done;
  });

  it("flush_sends_what_is_due_with_a_dab_in_flight", async () => {
    const gate = deferred();
    const seen: number[] = [];
    const stroke = new SculptStroke(
      brush,
      {},
      (ops) => {
        seen.push(ops.length);
        return gate.promise;
      },
    );
    stroke.dab([0, 0, 0]);
    stroke.dab([0.3, 0, 0]);
    expect(seen).toEqual([1]);
    stroke.flush();
    expect(seen).toEqual([1, 2]);
    gate.resolve();
    stroke.end();
    await stroke.done;
  });

  it("a_pass_runs_in_order_and_holds_the_end", async () => {
    const calls: string[] = [];
    const gate = deferred();
    const stroke = new SculptStroke(
      brush,
      { beginStep: () => void calls.push("begin"), endStep: () => void calls.push("end") },
      (ops) => {
        calls.push(`apply ${ops.length}`);
        return Promise.resolve();
      },
    );
    stroke.dab([0, 0, 0]);
    stroke.dab([0.3, 0, 0]);
    const passed = stroke.pass(() => {
      calls.push("pass");
      return gate.promise;
    });
    stroke.dab([0.6, 0, 0]);
    stroke.end();
    expect(calls).toEqual(["begin", "apply 1", "apply 2", "pass"]);
    gate.resolve();
    await passed;
    await stroke.done;
    expect(calls).toEqual(["begin", "apply 1", "apply 2", "pass", "apply 2", "end"]);
  });

  it("a_pass_holds_a_waiting_move_through_a_flush_and_the_end", async () => {
    const calls: string[] = [];
    const gate = deferred();
    const stroke = new SculptStroke(
      brush,
      { beginStep: () => void calls.push("begin"), endStep: () => void calls.push("end") },
      () => Promise.resolve(),
    );
    const move = (name: string) =>
      stroke.move(
        () => {
          calls.push(name);
          return Promise.resolve();
        },
        [0, 0, 0],
      );
    move("before");
    await tick();
    const passed = stroke.pass(() => {
      calls.push("pass");
      return gate.promise;
    });
    move("during");
    // A release: what is waiting must still land after the pass.
    stroke.flush();
    stroke.end();
    expect(calls).toEqual(["begin", "before", "pass"]);
    gate.resolve();
    await passed;
    await stroke.done;
    expect(calls).toEqual(["begin", "before", "pass", "during", "end"]);
  });

  it("a_second_pass_waits_for_the_first_and_what_was_due_before_it", async () => {
    const calls: string[] = [];
    const first = deferred();
    const stroke = new SculptStroke(brush, {}, () => Promise.resolve());
    stroke.pass(() => {
      calls.push("live");
      return first.promise;
    });
    stroke.move(
      () => {
        calls.push("move");
        return Promise.resolve();
      },
      [0, 0, 0],
    );
    void stroke.pass(() => {
      calls.push("end remesh");
      return Promise.resolve();
    });
    stroke.end();
    expect(calls).toEqual(["live"]);
    first.resolve();
    await stroke.done;
    expect(calls).toEqual(["live", "move", "end remesh"]);
  });

  it("the_end_waits_for_every_batch", async () => {
    const calls: string[] = [];
    const gate = deferred();
    const stroke = new SculptStroke(
      brush,
      { beginStep: () => void calls.push("begin"), endStep: () => void calls.push("end") },
      (ops) => {
        calls.push(`apply ${ops.length}`);
        return gate.promise;
      },
    );
    stroke.dab([0, 0, 0]);
    stroke.dab([0.3, 0, 0]);
    stroke.end();
    expect(calls).toEqual(["begin", "apply 1", "apply 2"]);
    gate.resolve();
    await stroke.done;
    expect(calls).toEqual(["begin", "apply 1", "apply 2", "end"]);
  });

  it("lays_a_dab_every_spacing_along_a_fast_move", async () => {
    const centers: (readonly [number, number, number])[] = [];
    const stroke = new SculptStroke(
      brush,
      {},
      (ops) => {
        centers.push(...ops.map((op) => (op as { center: readonly [number, number, number] }).center));
        return Promise.resolve();
      },
    );
    expect(stroke.dab([0, 0, 0])).toBe("stamped");
    await tick();
    expect(stroke.dab([0.5, 0, 0])).toBe("stamped");
    await tick();
    expect(centers.map((c) => Number(c[0].toFixed(6)))).toEqual([0, 0.15, 0.3, 0.45]);
    // The remainder carries over: the next sample goes on from the last dab.
    expect(stroke.dab([0.55, 0, 0])).toBe("skipped");
    expect(stroke.dab([0.65, 0, 0])).toBe("stamped");
    await tick();
    expect(centers.at(-1)?.[0]).toBeCloseTo(0.6, 10);
    stroke.end();
    await stroke.done;
  });

  it("starts_over_after_a_lift", async () => {
    const centers: (readonly [number, number, number])[] = [];
    const stroke = new SculptStroke(
      brush,
      {},
      (ops) => {
        centers.push(...ops.map((op) => (op as { center: readonly [number, number, number] }).center));
        return Promise.resolve();
      },
    );
    stroke.dab([0, 0, 0]);
    await tick();
    stroke.lift();
    expect(stroke.dab([1, 0, 0])).toBe("stamped");
    stroke.end();
    await stroke.done;
    expect(centers).toEqual([
      [0, 0, 0],
      [1, 0, 0],
    ]);
  });

  it("leaves_out_blocked_dabs_and_goes_on_past_them", async () => {
    const centers: number[] = [];
    const stroke = new SculptStroke(
      brush,
      {},
      (ops) => {
        centers.push(...ops.map((op) => (op as { center: readonly number[] }).center[0]));
        return Promise.resolve();
      },
      (at) => at[0] > 0.2 && at[0] < 0.4,
    );
    stroke.dab([0, 0, 0]);
    await tick();
    stroke.dab([0.65, 0, 0]);
    stroke.end();
    await stroke.done;
    expect(centers.map((x) => Number(x.toFixed(6)))).toEqual([0, 0.15, 0.45, 0.6]);
  });

  it("a_newer_move_replaces_one_that_has_not_started", async () => {
    const ran: string[] = [];
    let release = () => {};
    const stroke = new SculptStroke(brush, {}, () => Promise.resolve());
    stroke.move(
      () => {
        ran.push("first");
        return new Promise<void>((r) => {
          release = r;
        });
      },
      [0, 0, 0],
    );
    for (const name of ["second", "third"]) {
      stroke.move(
        () => {
          ran.push(name);
          return Promise.resolve();
        },
        [1, 0, 0],
      );
    }
    expect(ran).toEqual(["first"]);
    release();
    await tick();
    expect(ran).toEqual(["first", "third"]);
    stroke.end();
    await stroke.done;
    expect(stroke.bounds()?.center).toEqual([0.5, 0, 0]);
  });

  it("jumps_when_the_pointer_teleports", async () => {
    const centers: (readonly [number, number, number])[] = [];
    const stroke = new SculptStroke(
      brush,
      {},
      (ops) => {
        centers.push(...ops.map((op) => (op as { center: readonly [number, number, number] }).center));
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

  it("reads_a_getter_brush_live", async () => {
    let radius = 1;
    const seen: number[] = [];
    const stroke = new SculptStroke(
      () => ({ ...brush, radius }),
      {},
      (ops) => {
        seen.push(...ops.map((op) => (op as { radius: number }).radius));
        return Promise.resolve();
      },
    );
    expect(stroke.dab([0, 0, 0])).toBe("stamped");
    await tick();
    radius = 2;
    expect(stroke.dab([1, 0, 0])).toBe("stamped");
    stroke.end();
    await stroke.done;
    // The second radius applies from the second sample on, its spacing 0.3 over the distance 1.
    expect(seen).toEqual([1, 2, 2, 2]);
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

describe("brush picker", () => {
  const mesh = 7 as unknown as MeshHandle;

  it("maps_each_tool_to_its_repair_op", () => {
    const ops: Record<SculptTool, "inflate" | "relax" | "grab"> = {
      inflate: "inflate",
      deflate: "inflate",
      smooth: "relax",
      grab: "grab",
    };
    for (const [tool, op] of Object.entries(ops)) {
      expect(sculptOpForTool(tool as SculptTool)).toBe(op);
    }
  });

  it("starts_each_tool_at_its_own_strength", () => {
    expect(DEFAULT_STRENGTHS).toEqual({ inflate: 0.1, deflate: 0.1, smooth: 0.5, grab: 1 });
  });

  it("arms_inflate_up_and_deflate_down", () => {
    expect(sculptBrush("inflate", mesh, 0.25, 0.05, "smooth").strength).toBeCloseTo(0.05, 12);
    expect(sculptBrush("deflate", mesh, 0.25, 0.05, "smooth").strength).toBeCloseTo(-0.05, 12);
    // A stray sign never leaks through: deflate always pushes back in.
    expect(sculptBrush("deflate", mesh, 0.25, -0.05, "smooth").strength).toBeCloseTo(-0.05, 12);
    expect(sculptBrush("deflate", mesh, 0.25, 0.05, "smooth").op).toBe("inflate");
  });

  it("arms_smooth_as_relax_with_passes", () => {
    const brush = sculptBrush("smooth", mesh, 0.25, 0.5, "smooth");
    expect(brush.op).toBe("relax");
    expect(brush.cycles).toBe(RELAX_PASSES);
    expect(dabOp(brush, [1, 2, 3])).toEqual({
      op: "relax",
      center: [1, 2, 3],
      radius: 0.25,
      falloff: "smooth",
      strength: 0.5,
      cycles: RELAX_PASSES,
    });
  });

  it("arms_grab_follow_at_full_strength", () => {
    const brush = sculptBrush("grab", mesh, 0.25, 1, "smooth");
    expect(brush.op).toBe("grab");
    expect(brush.direction).toEqual([0, 0, 0]);
    expect(dabOp(brush, [1, 2, 3])).toEqual({
      op: "grab",
      center: [1, 2, 3],
      radius: 0.25,
      falloff: "smooth",
      strength: 1,
      direction: [0, 0, 0],
    });
  });

  it("carries_radius_and_falloff_through", () => {
    const brush = sculptBrush("inflate", mesh, 0.4, 0.05, "linear");
    expect(brush.mesh).toBe(mesh);
    expect(brush.radius).toBe(0.4);
    expect(brush.falloff).toBe("linear");
  });
});

describe("live remesh gate", () => {
  it("fires_only_past_both_budgets_and_never_while_running", () => {
    expect(liveRemeshDue(LIVE_REMESH_MIN_DABS, LIVE_REMESH_MIN_MS, false)).toBe(true);
    // Not enough new dabs since the last pass.
    expect(liveRemeshDue(LIVE_REMESH_MIN_DABS - 1, LIVE_REMESH_MIN_MS + 1000, false)).toBe(false);
    // Too soon after the last pass.
    expect(liveRemeshDue(LIVE_REMESH_MIN_DABS + 10, LIVE_REMESH_MIN_MS - 1, false)).toBe(false);
    // A pass still in flight: never two at once.
    expect(liveRemeshDue(LIVE_REMESH_MIN_DABS + 10, LIVE_REMESH_MIN_MS + 1000, true)).toBe(false);
  });

  it("keeps_the_per_frame_budget_usable_mid_stroke", () => {
    // A dab a frame at 60 fps clears the dab budget well inside the time
    // budget, so long strokes still even out while painting.
    expect(LIVE_REMESH_MIN_DABS).toBeLessThanOrEqual(8);
    expect(LIVE_REMESH_MIN_MS).toBeLessThanOrEqual(1000);
  });
});

describe("sculpt mask", () => {
  // A unit quad: 4 slots, two live triangles.
  const mesh = 7 as unknown as MeshHandle;
  const other = 9 as unknown as MeshHandle;
  const positions = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0];
  const indices = [0, 1, 2, 1, 3, 2];
  const mask = (vertices: number[], slots = 4): SculptMask => ({
    mesh,
    vertices: Uint32Array.from(vertices),
    slots,
  });

  it("paints_a_sphere_brush_over_live_vertices", () => {
    expect([...paintMaskIds(positions, indices, 4, [0, 0, 0], 0.1)]).toEqual([0]);
    // Within 1 of the origin: slots 0, 1 and 2 (slot 3 sits at √2).
    expect([...paintMaskIds(positions, indices, 4, [0, 0, 0], 1)]).toEqual([0, 1, 2]);
    expect([...paintMaskIds(positions, indices, 4, [0, 0, 0], 2)]).toEqual([0, 1, 2, 3]);
  });

  it("paint_skips_dead_and_out_of_range_slots", () => {
    // A removed triple paints nothing, even at its own slots.
    expect([...paintMaskIds(positions, [0, 0, 0], 4, [0, 0, 0], 2)]).toEqual([]);
    // An out-of-range id in a live triple never paints.
    expect([...paintMaskIds(positions, [0, 1, 9], 4, [0, 0, 0], 2)]).toEqual([0, 1]);
  });

  it("union_accumulates_paint_sorted", () => {
    expect([...unionMaskIds([0, 2], [1, 2, 3])]).toEqual([0, 1, 2, 3]);
    expect([...unionMaskIds([], [1])]).toEqual([1]);
    expect([...unionMaskIds([0, 1], [2, 3])]).toEqual([0, 1, 2, 3]);
  });

  it("invert_covers_the_live_universe", () => {
    expect([...invertMaskIds([0], indices, 4)]).toEqual([1, 2, 3]);
    expect([...invertMaskIds([], indices, 4)]).toEqual([0, 1, 2, 3]);
    // Dead masked ids drop out instead of reappearing.
    expect([...invertMaskIds([0, 7], [0, 0, 0], 4)]).toEqual([]);
  });

  it("blocks_dabs_whose_ball_touches_the_mask", () => {
    expect(maskBlocksDab([0], positions, [0.05, 0, 0], 0.1)).toBe(true);
    // The ball's edge counts, like the paint's own measure.
    expect(maskBlocksDab([1], positions, [0, 0, 0], 1)).toBe(true);
    expect(maskBlocksDab([3], positions, [0, 0, 0], 1)).toBe(false);
    expect(maskBlocksDab([], positions, [0, 0, 0], 1)).toBe(false);
  });

  it("usable_while_every_masked_slot_is_live", () => {
    expect(maskUsable(mask([0, 3]), indices, 4)).toBe(true);
    expect(maskUsable(mask([]), indices, 4)).toBe(true);
    // Out of range, or no live triangle left using it.
    expect(maskUsable(mask([7]), indices, 4)).toBe(false);
    expect(maskUsable(mask([3]), [0, 1, 2], 4)).toBe(false);
    // A removed triple's slots are dead, even at their own corners.
    expect(maskUsable(mask([0]), [0, 0, 0], 4)).toBe(false);
  });

  it("survives_position_only_stroke_edits", () => {
    // A dab moves vertices without touching the slots: the mask stands,
    // and the gate follows the moved positions.
    const moved = [0.05, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0];
    expect(maskUsable(mask([0, 3]), indices, 4)).toBe(true);
    expect(maskBlocksDab([0], moved, [0.05, 0, 0], 0.1)).toBe(true);
  });

  it("survives_splits_that_keep_old_vertices_alive", () => {
    // A remesh split appends a slot and a triangle: the painted ids still
    // name live vertices, so the mask carries over (the newcomer unmasked).
    const grown = [...indices, 0, 1, 4];
    expect(maskUsable(mask([0, 3], 4), grown, 5)).toBe(true);
  });

  it("fails_loudly_once_a_topology_edit_takes_a_masked_vertex", () => {
    // A collapse removes every triangle using slot 3: the mask dies.
    const collapsed = [0, 1, 2];
    expect(maskUsable(mask([0, 3]), collapsed, 4)).toBe(false);
    try {
      checkMask(mask([0, 3]), mesh, collapsed, 4);
      expect.unreachable();
    } catch (e) {
      // The same machine-readable code as the engine's stale selections.
      expect(e).toBeInstanceOf(MaskError);
      expect((e as MaskError).code).toBe("STALE_SELECTION");
    }
  });

  it("checkMask_rejects_the_wrong_mesh", () => {
    try {
      checkMask(mask([0]), other, indices, 4);
      expect.unreachable();
    } catch (e) {
      expect((e as MaskError).code).toBe("INVALID_ARGUMENT");
    }
    expect(() => checkMask(mask([0]), mesh, indices, 4)).not.toThrow();
  });
});
