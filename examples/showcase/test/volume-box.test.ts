// The main scene's volume box: which volume it shows (the selected
// volume mesh's, while the linked MPR state points at it), where the box
// sits in mesh-local coordinates, and where the cutaway clips it.
import { describe, expect, it } from "vitest";
import { cutFractions, matchVolume, volumeBoxLayout } from "../src/viewport/volume-box";

describe("matchVolume", () => {
  const items = [
    { mesh: 1, volumeId: "vol-a" },
    { mesh: 2, volumeId: null },
  ];
  it("matches the selected volume mesh while the MPR state points at it", () => {
    expect(matchVolume(items, 1, { volumeId: "vol-a" })).toEqual({
      item: items[0],
      mpr: { volumeId: "vol-a" },
      volumeId: "vol-a",
    });
  });
  it("misses without a selection, a volume, or a matching MPR series", () => {
    expect(matchVolume(items, null, { volumeId: "vol-a" })).toBe(null);
    expect(matchVolume(items, 2, { volumeId: "vol-a" })).toBe(null);
    expect(matchVolume(items, 9, { volumeId: "vol-a" })).toBe(null);
    expect(matchVolume(items, 1, null)).toBe(null);
    expect(matchVolume(items, 1, { volumeId: "vol-b" })).toBe(null);
  });
});

describe("volumeBoxLayout", () => {
  it("centres the box on the volume in mesh-local coordinates", () => {
    // 4 x 6 x 8 voxels at 2 mm, from an origin, against a mesh centre.
    const layout = volumeBoxLayout(
      { origin: [10, 20, 30], spacing: [2, 2, 2], columns: 4, rows: 6, slices: 8 },
      [12, 24, 36],
    );
    expect(layout.size).toEqual([8, 12, 16]);
    // The volume centre (14, 26, 38) minus the mesh centre.
    expect(layout.center).toEqual([2, 2, 2]);
  });
});

describe("cutFractions", () => {
  it("keeps the voxels at and below each slice, in box fraction", () => {
    expect(cutFractions({ axial: 9, coronal: 4, sagittal: 1 }, { columns: 4, rows: 10, slices: 20 })).toEqual([
      (1 + 0.5) / 4,
      (4 + 0.5) / 10,
      (9 + 0.5) / 20,
    ]);
  });
});
