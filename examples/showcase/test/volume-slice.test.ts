// The MPR slices: plane extraction with a fixed orientation (up is +Y
// for axial, +Z otherwise) and window/level to gray.
import { describe, expect, it } from "vitest";
import { extractSlice, pointerSlice, slicePlaneImages, windowToGray } from "../src/panels/volume-slice";

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

describe("pointerSlice", () => {
  const dims = { columns: 3, rows: 2, slices: 2 };

  it("maps_an_axial_click_to_sagittal_and_coronal", () => {
    // The 3x2 canvas drawn 10x: right side, top row.
    const at = pointerSlice("axial", dims, { width: 3, height: 2 }, { width: 30, height: 20 }, 25, 5);
    expect(at).toEqual({ sagittal: 2, coronal: 1 });
  });

  it("maps_coronal_and_sagittal_clicks", () => {
    const c = pointerSlice("coronal", dims, { width: 3, height: 2 }, { width: 30, height: 20 }, 5, 15);
    expect(c).toEqual({ sagittal: 0, axial: 0 });
    const s = pointerSlice("sagittal", dims, { width: 2, height: 2 }, { width: 20, height: 20 }, 15, 5);
    expect(s).toEqual({ coronal: 1, axial: 1 });
  });

  it("accounts_for_letterboxing_and_clamps", () => {
    // The 3x2 image centered in a 40x20 box: 5 px bars left and right.
    const dims2 = { columns: 3, rows: 2, slices: 2 };
    const at = pointerSlice("axial", dims2, { width: 3, height: 2 }, { width: 40, height: 20 }, 5, 10);
    expect(at).toEqual({ sagittal: 0, coronal: 0 });
    const bar = pointerSlice("axial", dims2, { width: 3, height: 2 }, { width: 40, height: 20 }, 2, 10);
    expect(bar).toEqual({ sagittal: 0, coronal: 0 });
  });
});

describe("slicePlaneImages", () => {
  it("windows_all_three_planes_at_the_slices", () => {
    const dims = { columns: COLUMNS, rows: ROWS, slices: SLICES };
    const planes = slicePlaneImages(VALUES, dims, { axial: 1, coronal: 1, sagittal: 2 }, 100, 50);
    expect([planes.axial.width, planes.axial.height, planes.axial.index, planes.axial.count]).toEqual([3, 2, 1, 2]);
    expect([planes.coronal.width, planes.coronal.height, planes.coronal.index, planes.coronal.count]).toEqual([3, 2, 1, 2]);
    expect([planes.sagittal.width, planes.sagittal.height, planes.sagittal.index, planes.sagittal.count]).toEqual([2, 2, 2, 3]);
    // Axial slice 1 starts at voxel 110: above the window, white.
    expect(Array.from(planes.axial.data.slice(0, 4))).toEqual([255, 255, 255, 255]);
  });
});

describe("windowToGray", () => {
  it("maps_window_level_to_gray_and_clamps", () => {
    const out = windowToGray(new Float32Array([-100, 0, 50, 100, 200]), 100, 50);
    // Window [0, 100]: below black, above white, mid gray, all opaque.
    expect(Array.from(out)).toEqual([0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255, 255, 255, 255, 255]);
  });
});
