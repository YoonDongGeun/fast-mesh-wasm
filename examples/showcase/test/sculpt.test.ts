// The sculpt stroke: dab spacing and ops, and one press-and-hold stroke
// through a fake applier (lazy step, busy and spacing guards, one undo
// step per stroke where the scene nests steps).
import type { MeshHandle } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import {
  SculptStroke,
  type SculptBrush,
  dabOp,
  dabSpacing,
  faceNormal,
  farEnough,
  interpolateHit,
} from "../src/viewport/sculpt";

const brush: SculptBrush = {
  mesh: 7 as unknown as MeshHandle,
  op: "inflate",
  radius: 1,
  strength: 0.1,
  falloff: "smooth",
  cycles: 3,
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
  it("spaces_dabs_by_a_quarter_radius", () => {
    expect(dabSpacing(1)).toBe(0.25);
    expect(dabSpacing(0.4)).toBeCloseTo(0.1, 10);
    expect(dabSpacing(0)).toBeGreaterThan(0);
    expect(dabSpacing(-2)).toBeGreaterThan(0);
  });

  it("far_enough_guards_spacing", () => {
    expect(farEnough([0, 0, 0], [0.25, 0, 0], 0.25)).toBe(true);
    expect(farEnough([0, 0, 0], [0.24, 0, 0], 0.25)).toBe(false);
    expect(farEnough([1, 1, 1], [1, 1, 1], 0.25)).toBe(false);
  });

  it("dab_op_builds_inflate_and_smooth", () => {
    expect(dabOp(brush, [1, 2, 3])).toEqual({
      op: "inflate",
      center: [1, 2, 3],
      radius: 1,
      falloff: "smooth",
      strength: 0.1,
    });
    expect(dabOp({ ...brush, op: "smooth" }, [1, 2, 3])).toEqual({
      op: "smooth",
      center: [1, 2, 3],
      radius: 1,
      falloff: "smooth",
      strength: 0.1,
      cycles: 3,
    });
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
