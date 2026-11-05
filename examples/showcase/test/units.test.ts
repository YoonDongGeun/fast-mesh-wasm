// Units (issue #145): the world unit is the millimetre. A display unit
// changes text only; an import scale changes the stored geometry, once.
// Expected values are written out independently of the code's constants:
// 1 in = 25.4 mm, 1 in² = 645.16 mm², 1 in³ = 16387.064 mm³.
import { describe, expect, it } from "vitest";
import {
  formatArea,
  formatLength,
  formatPoint,
  formatVolume,
  fromDisplayLength,
  importScaleFromFactor,
  importScaleOf,
  lengthInputValue,
  NO_IMPORT_SCALE,
  parseImportScale,
  rangeToDisplay,
  scalePositions,
  toDisplayArea,
  toDisplayLength,
  toDisplayVolume,
} from "../src/panels/units";
import { defaultSettings } from "../src/panels/settings";
import { AppState } from "../src/scene/app-state";

describe("conversion factors", () => {
  it("length: 1 in is 25.4 mm", () => {
    expect(toDisplayLength(25.4, "in")).toBeCloseTo(1, 12);
    expect(toDisplayLength(254, "in")).toBeCloseTo(10, 12);
    expect(fromDisplayLength(1, "in")).toBeCloseTo(25.4, 12);
  });

  it("area: 1 in² is 645.16 mm², the squared factor", () => {
    expect(toDisplayArea(645.16, "in")).toBeCloseTo(1, 12);
    // A 1 in cube's six faces.
    expect(toDisplayArea(6 * 645.16, "in")).toBeCloseTo(6, 12);
  });

  it("volume: 1 in³ is 16387.064 mm³, the cubed factor", () => {
    expect(toDisplayVolume(16387.064, "in")).toBeCloseTo(1, 12);
    expect(toDisplayVolume(8 * 16387.064, "in")).toBeCloseTo(8, 12);
  });

  it("mm is the world unit: every value passes through bit-identically", () => {
    for (const x of [0, 1e-5, 0.1, 1 / 3, 25.4, 1e12]) {
      expect(toDisplayLength(x, "mm")).toBe(x);
      expect(toDisplayArea(x, "mm")).toBe(x);
      expect(toDisplayVolume(x, "mm")).toBe(x);
      expect(fromDisplayLength(x, "mm")).toBe(x);
    }
  });
});

describe("formatting", () => {
  it("names the unit, with its power", () => {
    expect(formatLength(25.4, "mm")).toBe("25.4 mm");
    expect(formatLength(25.4, "in")).toBe("1 in");
    expect(formatArea(645.16, "in")).toBe("1 in²");
    expect(formatArea(2.5, "mm")).toBe("2.5 mm²");
    expect(formatVolume(16387.064, "in")).toBe("1 in³");
    expect(formatVolume(2.5, "mm")).toBe("2.5 mm³");
  });

  it("zero keeps its unit, and small inches stay readable", () => {
    expect(formatLength(0, "in")).toBe("0 in");
    expect(formatLength(0.0254, "in")).toBe("0.001 in");
    expect(formatLength(0.00254, "in")).toBe("1.00e-4 in");
  });

  it("converts a point per coordinate, naming the unit once", () => {
    expect(formatPoint([25.4, 0, -50.8], "in")).toBe("(1, 0, -2) in");
    expect(formatPoint([1, 2, 3], "mm")).toBe("(1, 2, 3) mm");
  });
});

describe("length inputs, both directions", () => {
  it("a range's bounds and value convert, and a dragged value converts back", () => {
    const r = rangeToDisplay({ min: 0, max: 50.8, step: 0.0508, value: 25.4 }, "in");
    expect(r.min).toBe(0);
    expect(r.max).toBeCloseTo(2, 12);
    expect(r.step).toBeCloseTo(0.002, 12);
    expect(r.value).toBeCloseTo(1, 12);
    // Dragging "Colour from" to 1 on the inch scale stores 25.4 mm.
    expect(fromDisplayLength(1, "in")).toBeCloseTo(25.4, 12);
  });

  it("leaves a range in mm untouched", () => {
    const r = { min: 0.001, max: 1, step: 0.001, value: 0.5 };
    expect(rangeToDisplay(r, "mm")).toEqual(r);
  });

  it("a typed value reads back as typed, not through float noise", () => {
    for (const typed of [0.1, 0.2, 0.3, 1, 1.5, 0.007, 12.345]) {
      expect(lengthInputValue(fromDisplayLength(typed, "in"), "in")).toBe(typed);
    }
    expect(lengthInputValue(1 / 3, "mm")).toBe(1 / 3);
  });
});

describe("import scale", () => {
  it("scales the positions in place, by the factor", () => {
    const p = Float64Array.from([1, 2, -1, 0, 0.5, 10]);
    const out = scalePositions(p, 25.4);
    expect(out).toBe(p);
    expect([...p]).toEqual([25.4, 50.8, -25.4, 0, 12.7, 254]);
  });

  it("factor 1 leaves the bits alone", () => {
    const original = [0.1, 1 / 3, 1e-300, -7];
    const p = Float64Array.from(original);
    scalePositions(p, 1);
    expect([...p]).toEqual(original);
  });

  it("the assumed units map to their factor in mm", () => {
    expect(importScaleOf("mm")).toEqual({ factor: 1, units: "mm" });
    expect(importScaleOf("cm")).toEqual({ factor: 10, units: "cm" });
    expect(importScaleOf("m")).toEqual({ factor: 1000, units: "m" });
    expect(importScaleOf("in")).toEqual({ factor: 25.4, units: "in" });
    expect(NO_IMPORT_SCALE).toEqual({ factor: 1, units: "mm" });
  });

  it("a typed factor names its unit when one matches, else is custom", () => {
    expect(importScaleFromFactor(25.4)).toEqual({ factor: 25.4, units: "in" });
    expect(importScaleFromFactor(1)).toEqual({ factor: 1, units: "mm" });
    expect(importScaleFromFactor(7)).toEqual({ factor: 7, units: null });
  });

  it("rejects factors that would destroy the geometry", () => {
    expect(parseImportScale("25.4")).toBe(25.4);
    expect(parseImportScale(" 2 ")).toBe(2);
    for (const bad of ["", "0", "-1", "abc", "Infinity", "NaN"]) {
      expect(parseImportScale(bad)).toBeNull();
    }
  });
});

describe("the per-project setting and the global default", () => {
  it("starts on the stored default", () => {
    const snap = new AppState({ ...defaultSettings(), defaultUnits: "in" }).getSnapshot();
    expect(snap.units).toBe("in");
    expect(snap.defaultUnits).toBe("in");
  });

  it("a pick by hand changes the project's units and the default", () => {
    const app = new AppState();
    expect(app.getSnapshot().units).toBe("mm");
    app.setUnits("in");
    expect(app.getSnapshot().units).toBe("in");
    expect(app.getSnapshot().defaultUnits).toBe("in");
  });

  it("opening a project changes its units only, never the default", () => {
    const app = new AppState();
    app.setProjectUnits("in");
    expect(app.getSnapshot().units).toBe("in");
    expect(app.getSnapshot().defaultUnits).toBe("mm");
    // An older file names none: back to the default.
    app.setProjectUnits(undefined);
    expect(app.getSnapshot().units).toBe("mm");
  });

  it("reset restores mm and the import scale", () => {
    const app = new AppState();
    app.setUnits("in");
    app.setImportScale(importScaleOf("in"));
    app.resetDisplaySettings();
    const s = app.getSnapshot();
    expect([s.units, s.defaultUnits]).toEqual(["mm", "mm"]);
    expect(s.importScale).toEqual({ factor: 1, units: "mm" });
  });

  it("the display unit moves no canonical value", () => {
    const app = new AppState();
    app.setHeatmapParams({ depth: 25.4, gap: 12.7, tolerance: 1 });
    app.setEdgeSoftness(0.5);
    app.setUnits("in");
    expect(app.getSnapshot().heatmapParams).toEqual({ depth: 25.4, gap: 12.7, tolerance: 1 });
    expect(app.getSnapshot().edgeSoftness).toBe(0.5);
  });
});
