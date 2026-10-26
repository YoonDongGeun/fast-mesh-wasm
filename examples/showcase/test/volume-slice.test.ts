// The MPR slices: plane extraction with a fixed orientation (up is +Y
// for axial, +Z otherwise) and window/level to gray.
import { describe, expect, it } from "vitest";
import { extractSlice, windowToGray } from "../src/panels/volume-slice";

// 3 x 2 x 2 voxels, x fastest: value = 100 * k + 10 * j + i.
const COLUMNS = 3;
const ROWS = 2;
const SLICES = 2;
const VALUES = new Float32Array([
  0, 1, 2, 10, 11, 12, 100, 101, 102, 110, 111, 112,
]);

describe("extractSlice", () => {
  it("extracts_axial_with_y_up", () => {
    const s = extractSlice(VALUES, COLUMNS, ROWS, SLICES, "axial", 1);
    expect([s.width, s.height]).toEqual([3, 2]);
    expect(Array.from(s.data)).toEqual([110, 111, 112, 100, 101, 102]);
  });

  it("extracts_coronal_with_z_up", () => {
    const s = extractSlice(VALUES, COLUMNS, ROWS, SLICES, "coronal", 1);
    expect([s.width, s.height]).toEqual([3, 2]);
    expect(Array.from(s.data)).toEqual([110, 111, 112, 10, 11, 12]);
  });

  it("extracts_sagittal_with_z_up", () => {
    const s = extractSlice(VALUES, COLUMNS, ROWS, SLICES, "sagittal", 2);
    expect([s.width, s.height]).toEqual([2, 2]);
    expect(Array.from(s.data)).toEqual([102, 112, 2, 12]);
  });
});

describe("windowToGray", () => {
  it("maps_window_level_to_gray_and_clamps", () => {
    const out = windowToGray(new Float32Array([-100, 0, 50, 100, 200]), 100, 50);
    // Window [0, 100]: below black, above white, mid gray, all opaque.
    expect(Array.from(out)).toEqual([0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255, 255, 255, 255, 255]);
  });
});
