// The collision panel's settings arithmetic: the sliders' scale from the
// meshes coloured, the heatmap values clamped at that scale, and the
// collision probe's nudge step.
import { describe, expect, it } from "vitest";
import { heatmapLimits, meshScale, nudgeOffset } from "../src/features/collision/ux/collision-math";

describe("collision settings", () => {
  it("mesh_scale_follows_the_largest_listed_radius", () => {
    const items = [
      { mesh: "a", radius: 1.9 },
      { mesh: "b", radius: 140 },
      { mesh: "c", radius: 0.03 },
    ];
    expect(meshScale(items, ["a", "c"])).toBe(2);
    expect(meshScale(items, ["a", "b"])).toBe(200);
    // Nothing picked: niceScale(0).
    expect(meshScale(items, [null, null])).toBe(1);
    expect(meshScale(items, [])).toBe(1);
  });

  it("heatmap_limits_default_to_the_scale", () => {
    const l = heatmapLimits(null, 2);
    expect(l.step).toBe(0.002);
    expect(l.depth).toBe(1);
    expect(l.gap).toBe(1);
    expect(l.tolerance).toBe(0.04);
    expect(l.toleranceMax).toBeCloseTo(0.998, 12);
  });

  it("heatmap_limits_clamp_set_values", () => {
    // Depth at least a step, both at most the scale, the gap never negative.
    const l = heatmapLimits({ depth: 0, gap: -1, tolerance: 0.5 }, 1);
    expect(l.depth).toBe(0.001);
    expect(l.gap).toBe(0);
    // "Colour from" 0: the depth bounds the tolerance, a step below it.
    expect(l.tolerance).toBe(0);
    expect(l.toleranceMax).toBe(0);
    const m = heatmapLimits({ depth: 5, gap: 0.5, tolerance: 0.7 }, 1);
    expect(m.depth).toBe(1);
    expect(m.gap).toBe(0.5);
    expect(m.tolerance).toBeCloseTo(0.499, 12);
    expect(m.toleranceMax).toBeCloseTo(0.499, 12);
  });

  it("nudge_moves_b_toward_a", () => {
    // Half the way when far.
    expect(nudgeOffset([4, 0, 0], [0, 0, 0])).toEqual([2, 0, 0]);
    // At least 0.5 when that is under the distance.
    const near = nudgeOffset([0, 0.6, 0], [0, 0, 0]);
    expect(near[0]).toBe(0);
    expect(near[1]).toBeCloseTo(0.5, 12);
    expect(near[2]).toBe(0);
    // The whole way when closer than 0.5.
    const close = nudgeOffset([0, 0, 0], [0, 0, 0.3]);
    expect(close[2]).toBeCloseTo(-0.3, 12);
    // Coincident centres: a fixed step along x.
    expect(nudgeOffset([1, 1, 1], [1, 1, 1])).toEqual([0.25, 0, 0]);
  });
});
