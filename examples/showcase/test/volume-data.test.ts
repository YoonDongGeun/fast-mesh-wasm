// The volume display values: real volumes pass through (Hounsfield units),
// phantom SDF maps to pseudo-HU for the CT presets (air past 5 mm out).
import { describe, expect, it } from "vitest";
import { displayValues } from "../src/panels/volume-data";

describe("displayValues", () => {
  it("passes_real_volumes_through", () => {
    const v = new Float32Array([-1000, 0, 500]);
    expect(displayValues(v, false)).toBe(v);
  });

  it("maps_phantom_sdf_to_pseudo_hu", () => {
    const out = displayValues(new Float32Array([-5, 0, 2, 5, 30]), true);
    expect(Array.from(out)).toEqual([500, 300, 220, 100, -1000]);
  });
});
