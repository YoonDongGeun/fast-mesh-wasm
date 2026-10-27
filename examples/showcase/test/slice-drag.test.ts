// Dragging an MPR slice quad through the main scene: the pointer ray
// meets a drag plane (through the grab point, facing the camera while it
// contains the quad normal), and the hit's travel along the normal
// becomes slice steps.
import { describe, expect, it } from "vitest";
import { clampSlice, dragPlaneNormal, planeHit, sliceSteps } from "../src/viewport/slice-drag";

const close = (got: readonly number[] | null, want: readonly number[]): void => {
  expect(got === null ? null : [...got]).not.toBe(null);
  (got as readonly number[]).forEach((v, i) => expect(v).toBeCloseTo(want[i], 9));
};

describe("dragPlaneNormal", () => {
  it("faces_the_camera_around_the_quad_normal", () => {
    close(dragPlaneNormal([0, 0, 1], [0, 1, 0]), [0, 1, 0]);
    close(dragPlaneNormal([0, 0, 1], [0, 0.6, 0.8]), [0, 1, 0]);
    close(dragPlaneNormal([0, 1, 0], [1, 1, 0]), [1, 0, 0]);
  });

  it("gives_up_looking_straight_down_the_normal", () => {
    expect(dragPlaneNormal([0, 0, 1], [0, 0, 1])).toBe(null);
    expect(dragPlaneNormal([0, 0, 1], [0, 0, -1])).toBe(null);
    expect(dragPlaneNormal([1, 0, 0], [1, 1e-9, 0])).toBe(null);
  });
});

describe("planeHit", () => {
  it("meets_the_ray_with_the_plane", () => {
    close(planeHit([0, 0, 5], [0, 0, -1], [1, 2, 0], [0, 0, 1]), [0, 0, 0]);
    close(planeHit([1, 1, 1], [0, -1, 0], [0, 0, 0], [0, 1, 0]), [1, 0, 1]);
  });

  it("misses_a_parallel_ray_or_a_plane_behind", () => {
    expect(planeHit([0, 0, 5], [1, 0, 0], [0, 0, 0], [0, 0, 1])).toBe(null);
    expect(planeHit([0, 0, -5], [0, 0, -1], [0, 0, 0], [0, 0, 1])).toBe(null);
  });
});

describe("sliceSteps", () => {
  it("counts_travel_along_the_normal_in_spacing", () => {
    expect(sliceSteps([0, 0, 0], [0, 0, 10], [0, 0, 1], 2)).toBeCloseTo(5, 9);
    expect(sliceSteps([0, 0, 0], [0, 0, -5], [0, 0, 1], 2)).toBeCloseTo(-2.5, 9);
    expect(sliceSteps([1, 2, 3], [9, 2, 3], [0, 0, 1], 2)).toBeCloseTo(0, 9);
  });

  it("stands_still_without_a_spacing", () => {
    expect(sliceSteps([0, 0, 0], [0, 0, 10], [0, 0, 1], 0)).toBe(0);
  });
});

describe("clampSlice", () => {
  it("rounds_and_clamps_to_the_stack", () => {
    expect(clampSlice(3.4, 10)).toBe(3);
    expect(clampSlice(3.6, 10)).toBe(4);
    expect(clampSlice(-2, 10)).toBe(0);
    expect(clampSlice(99, 10)).toBe(9);
  });
});
