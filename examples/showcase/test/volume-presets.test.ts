// The volume presets: HU-tuned color/opacity ramps (bone, soft tissue,
// vessels), applied to the vtk.js transfer functions, and the window
// presets resolved over the data range.
import { describe, expect, it } from "vitest";
import { presetWindowLevel, VOLUME_PRESETS, WINDOW_PRESETS } from "../src/features/ct/ux/volume-presets";

describe("VOLUME_PRESETS", () => {
  it("covers_bone_tissue_and_vessels", () => {
    expect(VOLUME_PRESETS.map((p) => p.name)).toEqual(["Bone", "Soft tissue", "Vessels"]);
  });

  it("has_sorted_valid_ramps", () => {
    for (const p of VOLUME_PRESETS) {
      expect(p.color.length).toBeGreaterThanOrEqual(2);
      expect(p.opacity.length).toBeGreaterThanOrEqual(2);
      for (let i = 1; i < p.color.length; i++) {
        expect(p.color[i][0]).toBeGreaterThan(p.color[i - 1][0]);
      }
      for (let i = 1; i < p.opacity.length; i++) {
        expect(p.opacity[i][0]).toBeGreaterThan(p.opacity[i - 1][0]);
      }
      for (const [, r, g, b] of p.color) {
        for (const c of [r, g, b]) {
          expect(c).toBeGreaterThanOrEqual(0);
          expect(c).toBeLessThanOrEqual(1);
        }
      }
      for (const [, o] of p.opacity) {
        expect(o).toBeGreaterThanOrEqual(0);
        expect(o).toBeLessThanOrEqual(1);
      }
      // Air stays invisible: zero opacity at -1000 HU and below.
      expect(p.opacity[0][0]).toBeLessThanOrEqual(-1000);
      expect(p.opacity[0][1]).toBe(0);
    }
  });
});

describe("WINDOW_PRESETS", () => {
  it("has_a_window_preset_for_bone_soft_lung_and_full", () => {
    expect(WINDOW_PRESETS.map((p) => p.name)).toEqual(["Bone", "Soft tissue", "Lung", "Full range"]);
    for (const p of WINDOW_PRESETS) {
      expect(p.window === "auto" || p.window > 0).toBe(true);
      expect(p.level === "auto" || Number.isFinite(p.level)).toBe(true);
    }
  });
});

describe("presetWindowLevel", () => {
  it("keeps_a_fixed_window_and_level", () => {
    expect(presetWindowLevel({ name: "Bone", window: 2000, level: 400 }, [-1000, 3000])).toEqual({ window: 2000, level: 400 });
  });

  it("spans_the_data_range_when_auto", () => {
    expect(presetWindowLevel({ name: "Full range", window: "auto", level: "auto" }, [-1000, 3000])).toEqual({
      window: 4000,
      level: 1000,
    });
    expect(presetWindowLevel({ name: "Mixed", window: "auto", level: 40 }, [0, 10])).toEqual({ window: 10, level: 40 });
  });
});
