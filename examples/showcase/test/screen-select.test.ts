// Screen-space face selection shapes: canvas pixels to NDC, the clamped
// local pointer, the box (with its click pad) and the lasso (with its
// single-point fallback).
import { describe, expect, it } from "vitest";
import {
  boxShapeFromPixels,
  clientToLocalPixels,
  lassoShapeFromPixels,
  localPointToNdc,
  rectShapeFromPixels,
} from "../src/features/select/ux/screen-select";

const rect = { left: 10, top: 20, width: 200, height: 100 };

describe("localPointToNdc", () => {
  it("maps_corners_and_center", () => {
    expect(localPointToNdc(0, 0, 200, 100)).toEqual([-1, 1]);
    expect(localPointToNdc(200, 100, 200, 100)).toEqual([1, -1]);
    expect(localPointToNdc(100, 50, 200, 100)).toEqual([0, 0]);
  });

  it("guards_a_zero_sized_canvas", () => {
    expect(localPointToNdc(0, 0, 0, 0)).toEqual([-1, 1]);
  });
});

describe("clientToLocalPixels", () => {
  it("offsets_by_the_canvas_corner", () => {
    expect(clientToLocalPixels(60, 70, rect)).toEqual([50, 50]);
  });

  it("clamps_into_the_canvas", () => {
    expect(clientToLocalPixels(0, 500, rect)).toEqual([0, 100]);
  });
});

describe("rectShapeFromPixels", () => {
  it("orders_the_corners_either_way", () => {
    const shape = { kind: "rect", minNdc: [-1, -1], maxNdc: [0, 1] };
    expect(rectShapeFromPixels(0, 0, 100, 100, rect, 0)).toEqual(shape);
    expect(rectShapeFromPixels(100, 100, 0, 0, rect, 0)).toEqual(shape);
  });

  it("grows_by_the_pad", () => {
    const shape = rectShapeFromPixels(100, 50, 100, 50, rect, 10);
    if (shape.kind !== "rect") {
      throw new Error("expected a rect");
    }
    expect(shape.minNdc[0]).toBeCloseTo(-0.1, 12);
    expect(shape.minNdc[1]).toBeCloseTo(-0.2, 12);
    expect(shape.maxNdc[0]).toBeCloseTo(0.1, 12);
    expect(shape.maxNdc[1]).toBeCloseTo(0.2, 12);
  });
});

describe("boxShapeFromPixels", () => {
  it("pads_a_click", () => {
    expect(boxShapeFromPixels(100, 50, 101, 51, rect)).toEqual(rectShapeFromPixels(100, 50, 101, 51, rect, 4));
  });

  it("keeps_a_drag_exact", () => {
    expect(boxShapeFromPixels(0, 0, 100, 100, rect)).toEqual(rectShapeFromPixels(0, 0, 100, 100, rect, 0));
  });
});

describe("lassoShapeFromPixels", () => {
  it("projects_the_polygon", () => {
    expect(lassoShapeFromPixels([[0, 0], [200, 0], [100, 100]], rect)).toEqual({
      kind: "lasso",
      polygonNdc: [[-1, 1], [1, 1], [0, -1]],
    });
  });

  it("falls_back_to_a_padded_point", () => {
    expect(lassoShapeFromPixels([[100, 50], [104, 50]], rect)).toEqual(rectShapeFromPixels(100, 50, 100, 50, rect, 4));
    expect(lassoShapeFromPixels([], rect)).toEqual(rectShapeFromPixels(0, 0, 0, 0, rect, 4));
  });
});
