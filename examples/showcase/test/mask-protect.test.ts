// Per-vertex mask protection: a dab near the mask moves the free side.
// Bytes on a real engine: `scripts/showcase-mask-check.mjs`.
import type { RepairOp } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import { maskCoversDab, maskEdgeField, maskTouchesRegion, protectDab } from "../src/viewport/mask-protect";

/** A strip along x: vertices at (i, 0, 0) and (i, 1, 0), `2 * (n + 1)` slots. */
function strip(n: number): { positions: number[]; indices: number[]; slots: number } {
  const positions: number[] = [];
  for (let i = 0; i <= n; i++) {
    positions.push(i, 0, 0, i, 1, 0);
  }
  const indices: number[] = [];
  for (let i = 0; i < n; i++) {
    indices.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
  }
  return { positions, indices, slots: 2 * (n + 1) };
}

/** The strip's slots at x < `x`. */
const leftOf = (x: number) => Array.from({ length: 2 * x }, (_, i) => i);

const inflateAt = (center: [number, number, number], radius: number): RepairOp => ({
  op: "inflate",
  center,
  radius,
  falloff: "smooth",
  strength: 0.1,
});

describe("maskEdgeField", () => {
  it("weights_masked_slots_zero_and_rises_across_the_band", () => {
    const { positions, indices, slots } = strip(6);
    const f = maskEdgeField(leftOf(2), positions, indices, slots, 2.5);
    const weightOf = (v: number) => f.weights[Array.from(f.ids).indexOf(v)];
    expect(weightOf(0)).toBe(0);
    expect(weightOf(3)).toBe(0);
    const one = weightOf(4); // x = 2, one edge from the mask.
    const two = weightOf(6); // x = 3, two edges from it.
    expect(one).toBeGreaterThan(0);
    expect(two).toBeGreaterThan(one);
    expect(two).toBeLessThan(1);
    // Past the band: not listed, so it moves freely.
    expect(Array.from(f.ids)).not.toContain(8);
  });

  it("flags_exactly_the_masked_slots", () => {
    const { positions, indices, slots } = strip(3);
    const f = maskEdgeField([1, 2, 99], positions, indices, slots, 0);
    expect(Array.from(f.flags)).toEqual([0, 1, 1, 0, 0, 0, 0, 0]);
    expect(Array.from(f.ids)).toEqual([1, 2]);
  });

  it("does_not_walk_through_dead_triangles", () => {
    const f = maskEdgeField([0], [0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0], 3, 5);
    expect(Array.from(f.ids)).toEqual([0]);
  });

  it("measures_along_edges_not_through_space", () => {
    // A hairpin: slots 0 and 3 are close in space, several long edges apart.
    const positions = [0, 0, 0, 5, 0, 0, 5, 0.1, 0, 0, 0.1, 0, 5, 0.2, 0];
    const f = maskEdgeField([0], positions, [0, 1, 4, 1, 2, 4, 2, 3, 4], 5, 1);
    expect(Array.from(f.ids)).toEqual([0]);
  });
});

describe("a dab touching the mask's edge", () => {
  const { positions, indices, slots } = strip(8);
  const live = new Uint8Array(slots).fill(1);
  const field = maskEdgeField(leftOf(3), positions, indices, slots, 1.5);

  it("is_not_dropped_while_its_ball_holds_free_vertices", () => {
    expect(maskCoversDab(field.flags, live, positions, [3.5, 0, 0], 2)).toBe(false);
  });

  it("is_dropped_when_every_vertex_in_its_ball_is_masked", () => {
    expect(maskCoversDab(field.flags, live, positions, [1, 0, 0], 0.5)).toBe(true);
  });

  it("is_not_reported_covered_with_nothing_in_its_ball", () => {
    expect(maskCoversDab(field.flags, live, positions, [1, 50, 0], 0.5)).toBe(false);
  });

  it("ignores_dead_slots_in_its_ball", () => {
    const onlyMaskedLive = new Uint8Array(slots);
    onlyMaskedLive[0] = 1;
    expect(maskCoversDab(field.flags, onlyMaskedLive, positions, [1, 0, 0], 8)).toBe(true);
  });

  it("protects_the_masked_vertices_it_reaches_at_weight_zero", () => {
    const op = protectDab(inflateAt([3.5, 0, 0], 2), field, positions) as RepairOp & {
      protect: { vertices: Uint32Array; weights: Float32Array };
    };
    const { vertices, weights } = op.protect;
    // Masked slots 4 and 5 (x = 2) are in the ball; slots 0-1 (x = 0) are past its reach.
    expect(Array.from(vertices)).toContain(4);
    expect(Array.from(vertices)).toContain(5);
    expect(Array.from(vertices)).not.toContain(0);
    expect(weights[Array.from(vertices).indexOf(4)]).toBe(0);
    // Free slots in the band are listed softly; those past it are not listed.
    const soft = Array.from(vertices).indexOf(6);
    expect(weights[soft]).toBeGreaterThan(0);
    expect(weights[soft]).toBeLessThan(1);
    expect(Array.from(vertices)).not.toContain(10);
  });

  it("keeps_the_brush_itself", () => {
    const op = inflateAt([3.5, 0, 0], 2);
    expect(protectDab(op, field, positions)).toMatchObject(op);
  });

  it("returns_the_op_itself_away_from_the_mask", () => {
    const op = inflateAt([8, 0, 0], 1);
    expect(protectDab(op, field, positions)).toBe(op);
  });

  it("leaves_a_non_brush_op_alone", () => {
    const op: RepairOp = { op: "remesh", center: [1, 0, 0], radius: 5 };
    expect(protectDab(op, field, positions)).toBe(op);
  });
});

describe("maskTouchesRegion", () => {
  it("reports_a_masked_slot_inside_the_region", () => {
    const { positions } = strip(4);
    expect(maskTouchesRegion([0], positions, [1, 0, 0], 1)).toBe(true);
    expect(maskTouchesRegion([0], positions, [3, 0, 0], 1)).toBe(false);
    expect(maskTouchesRegion([], positions, [0, 0, 0], 9)).toBe(false);
  });
});
