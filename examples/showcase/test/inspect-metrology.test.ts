// The inspect panel's metrology helpers: the fitted primitive's readout
// lines, the custom contour normal and the contour totals.
import type { ContourSlice, FittedPrimitive } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import { contourSliceTotals, normalizeNormalVec } from "../src/features/inspect/ux/use-contour-slices";
import { formatVec3Unitless, summarizePrimitiveGeometry } from "../src/features/inspect/ux/use-primitive-fit";

describe("formatVec3Unitless", () => {
  it("prints_three_decimals", () => {
    expect(formatVec3Unitless([0, -0.5, 1])).toBe("(0.000, -0.500, 1.000)");
  });
});

describe("summarizePrimitiveGeometry", () => {
  it("reads_a_plane", () => {
    const plane = { kind: "plane", origin: [1, 2, 3], normal: [0, 0, 1] } as unknown as FittedPrimitive;
    expect(summarizePrimitiveGeometry(plane, "mm")).toEqual(["Origin: (1, 2, 3) mm", "Normal: (0.000, 0.000, 1.000)"]);
  });

  it("reads_a_sphere", () => {
    const sphere = { kind: "sphere", center: [0, 0, 0], radius: 2 } as unknown as FittedPrimitive;
    expect(summarizePrimitiveGeometry(sphere, "mm")).toEqual(["Center: (0, 0, 0) mm", "Radius: 2 mm"]);
  });

  it("reads_a_cone_angle_in_degrees", () => {
    const cone = {
      kind: "cone",
      apex: [0, 0, 0],
      axisDir: [0, 1, 0],
      halfAngleRad: Math.PI / 6,
      height: 3,
    } as unknown as FittedPrimitive;
    expect(summarizePrimitiveGeometry(cone, "mm")).toEqual([
      "Apex: (0, 0, 0) mm",
      "Axis: (0.000, 1.000, 0.000)",
      "Half-angle: 30.00°",
      "Height: 3 mm",
    ]);
  });
});

describe("normalizeNormalVec", () => {
  it("scales_to_unit_length", () => {
    expect(normalizeNormalVec([0, 3, 4])).toEqual([0, 0.6, 0.8]);
  });

  it("falls_back_to_plus_z_for_zero_or_non_finite", () => {
    expect(normalizeNormalVec([0, 0, 0])).toEqual([0, 0, 1]);
    expect(normalizeNormalVec([Number.NaN, 1, 0])).toEqual([0, 0, 1]);
  });
});

describe("contourSliceTotals", () => {
  it("is_zero_without_slices", () => {
    expect(contourSliceTotals(null)).toEqual({ loops: 0, perimeter: 0, area: 0 });
  });

  it("sums_loops_perimeter_and_area", () => {
    const slices = [
      { loops: [{}, {}], totalPerimeter: 3, totalArea: 1 },
      { loops: [{}], totalPerimeter: 4, totalArea: 2.5 },
    ] as unknown as ContourSlice[];
    expect(contourSliceTotals(slices)).toEqual({ loops: 3, perimeter: 7, area: 3.5 });
  });
});
