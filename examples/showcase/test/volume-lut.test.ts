// The volume LUT: a transfer preset sampled into 256 RGBA entries over
// the data range, plus the volume normalized to 8-bit indices.
import { describe, expect, it } from "vitest";
import { VOLUME_PRESETS } from "../src/panels/volume-presets";
import { dataRange, normalizeVolume, presetLut } from "../src/panels/volume-lut";

function alphaAt(lut: Uint8Array, v: number, min: number, max: number): number {
  const i = Math.round(((v - min) / (max - min)) * 255);
  return lut[4 * i + 3];
}

describe("presetLut", () => {
  it("keeps_air_transparent_and_bone_opaque", () => {
    const bone = VOLUME_PRESETS[0];
    const lut = presetLut(bone, -1000, 2500);
    expect(lut.length).toBe(256 * 4);
    expect(alphaAt(lut, -1000, -1000, 2500)).toBe(0);
    expect(alphaAt(lut, 200, -1000, 2500)).toBe(0);
    expect(alphaAt(lut, 800, -1000, 2500)).toBeGreaterThan(100);
    expect(alphaAt(lut, 2500, -1000, 2500)).toBeGreaterThan(200);
  });

  it("interpolates_color_between_points", () => {
    const bone = VOLUME_PRESETS[0];
    const lut = presetLut(bone, -1000, 2500);
    const i = Math.round(((850 - -1000) / 3500) * 255);
    // Between [800-ish] and [1200]: warm gray, brighter than at 500.
    const j = Math.round(((500 - -1000) / 3500) * 255);
    expect(lut[4 * i]).toBeGreaterThan(lut[4 * j]);
  });
});

describe("normalizeVolume", () => {
  it("maps_the_range_to_8_bit_and_clamps", () => {
    expect(Array.from(normalizeVolume(new Float32Array([-2000, -1000, 750, 2500, 5000]), -1000, 2500))).toEqual([
      0, 0, 128, 255, 255,
    ]);
  });
});

describe("dataRange", () => {
  it("finds_min_and_max", () => {
    expect(dataRange(new Float32Array([3, -2, 7, 0]))).toEqual([-2, 7]);
  });
});
