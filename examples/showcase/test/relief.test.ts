// Image to relief (#267): its settings and their checks, the grid they sample
// with, what a height map's JSON sets them to, and which dropped files it takes.
import { describe, expect, it } from "vitest";
import {
  DEFAULT_SAMPLES,
  defaultRelief,
  describeRelief,
  MAX_RELIEF_TRIANGLES,
  maxSamples,
  reliefDepth,
  reliefGrid,
  reliefName,
  reliefOfFrame,
  reliefProblem,
  reliefTriangles,
  splitReliefFiles,
} from "../src/features/mesh-edit/ux/relief";
import { eulerDegToMat3 } from "../src/features/transform/ux/transform-math";
import { gridSize } from "../src/io/height-image";

const IMAGE = { width: 1001, height: 501 };

describe("relief settings", () => {
  it("starts at the scene's size, its longer side across", () => {
    const p = defaultRelief(IMAGE, 100);
    expect(p).toEqual({ width: 100, maxHeight: 10, base: 5, samples: DEFAULT_SAMPLES, invert: false });
    expect(reliefDepth(p.width, IMAGE)).toBe(50);
    const tall = defaultRelief({ width: 501, height: 1001 }, 100);
    expect(tall.width).toBe(50);
    expect(reliefDepth(tall.width, { width: 501, height: 1001 })).toBe(100);
    // A small image keeps its own pixels.
    expect(defaultRelief({ width: 20, height: 10 }, 1).samples).toBe(20);
  });

  it("counts the triangles the engine builds", () => {
    // 2 x 2: 2 for the top, 8 for the sides, 2 for the base.
    expect(reliefTriangles(2, 2)).toBe(12);
    expect(reliefTriangles(1024, 1024)).toBe(2 * 1023 * 1023 + 3 * 4092 - 2);
  });

  it("checks its values and keeps the mesh within its limit", () => {
    const p = defaultRelief(IMAGE, 100);
    expect(reliefProblem(p, IMAGE)).toBeNull();
    for (const key of ["width", "maxHeight", "base"] as const) {
      for (const v of [0, -1, Number.NaN, Infinity]) {
        expect(reliefProblem({ ...p, [key]: v }, IMAGE)).toMatch(/must be above zero/);
      }
    }
    for (const samples of [1, 2.5, 1002, Number.NaN]) {
      expect(reliefProblem({ ...p, samples }, IMAGE)).toMatch(/whole number from 2 to 1001/);
    }
    const big = { width: 5000, height: 5000 };
    const most = maxSamples(big);
    expect(reliefTriangles(...gridSize(5000, 5000, most))).toBeLessThanOrEqual(MAX_RELIEF_TRIANGLES);
    expect(reliefTriangles(...gridSize(5000, 5000, most + 1))).toBeGreaterThan(MAX_RELIEF_TRIANGLES);
    expect(reliefProblem({ ...p, samples: most }, big)).toBeNull();
    expect(reliefProblem({ ...p, samples: most + 1 }, big)).toMatch(new RegExp(`at most .* \\(${most} samples\\)`));
    expect(maxSamples(IMAGE)).toBe(1001);
    expect(maxSamples({ width: 2, height: 2 })).toBe(2);
  });

  it("spans the same area at every sample count", () => {
    const p = { ...defaultRelief(IMAGE, 100), samples: 1001 };
    expect(reliefGrid(p, IMAGE)).toEqual({ w: 1001, h: 501, pixelSize: [0.1, 0.1] });
    for (const samples of [2, 3, 40, 700]) {
      const { w, h, pixelSize } = reliefGrid({ ...p, samples }, IMAGE);
      expect(pixelSize[0] * (w - 1)).toBeCloseTo(100, 9);
      expect(pixelSize[1] * (h - 1)).toBeCloseTo(50, 9);
    }
    // The preview never takes more than the relief itself.
    expect(reliefGrid({ ...p, samples: 50 }, IMAGE, 160).w).toBe(50);
    expect(reliefGrid(p, IMAGE, 160).w).toBe(160);
  });

  it("describes itself in the display unit", () => {
    const p = { width: 25.4, maxHeight: 2.54, base: 2.54, samples: 3, invert: false };
    expect(describeRelief(p, { width: 3, height: 3 }, "in")).toBe("30 triangles · 3 × 3 samples · 1 in × 1 in × 0.2 in");
  });
});

describe("a height map's JSON", () => {
  const frame = {
    width: 101,
    height: 51,
    origin: [1, 20, -3] as [number, number, number],
    right: [1, 0, 0] as [number, number, number],
    up: [0, 0, -1] as [number, number, number],
    normal: [0, 1, 0] as [number, number, number],
    pixelSize: 0.5,
    range: 7,
  };
  const image = { width: 101, height: 51 };

  it("sets the size, the height and the place the map saw", () => {
    const r = reliefOfFrame(frame, image, 2)!;
    expect(r.params).toEqual({ width: 50, maxHeight: 7, base: 2, samples: 101, invert: false });
    expect(reliefDepth(r.params.width, image)).toBe(25);
    // The middle of the image, on its zero plane.
    expect(r.place.centre).toEqual([1 + 25, 20, -3 + 12.5]);
    // The rotation's columns are the image's axes.
    const m = eulerDegToMat3(r.place.rotationDeg);
    const column = (k: number) => [m[k], m[3 + k], m[6 + k]];
    [frame.right, frame.up, frame.normal].forEach((axis, k) =>
      axis.forEach((c, i) => expect(column(k)[i]).toBeCloseTo(c, 12)),
    );
    // The first sample is where the map's first pixel was.
    const local = [-25, 12.5, 0];
    const at = [0, 1, 2].map((k) => r.place.centre[k] + m[3 * k] * local[0] + m[3 * k + 1] * local[1]);
    at.forEach((c, k) => expect(c).toBeCloseTo(frame.origin[k], 12));
  });

  it("is left out for another image, or for one that shows no height", () => {
    expect(reliefOfFrame(frame, { width: 100, height: 51 }, 1)).toBeNull();
    expect(reliefOfFrame({ ...frame, range: 0 }, image, 1)).toBeNull();
  });
});

describe("dropped files", () => {
  const file = (name: string, type = "") => new File(["x"], name, { type });

  it("takes an image and the JSON of its name, and leaves the rest to load", () => {
    const image = file("part-height-map.png", "image/png");
    const json = file("part-height-map.json");
    const stray = file("other.json");
    const stl = file("a.stl");
    const { relief, rest } = splitReliefFiles([stray, stl, json, image]);
    expect(relief).toEqual({ image, json });
    expect(rest).toEqual([stray, stl]);
    // Without one of its name, the first JSON (the dialog says when it is not one).
    expect(splitReliefFiles([image, stray]).relief).toEqual({ image, json: stray });
    expect(splitReliefFiles([image]).relief).toEqual({ image, json: null });
  });

  it("loads everything as before without an image", () => {
    const list = [file("a.stl"), file("scene.gltf"), file("scene.json")];
    expect(splitReliefFiles(list)).toEqual({ relief: null, rest: list });
  });

  it("names a relief after its file", () => {
    expect(reliefName("photo.of.hills.png")).toBe("photo.of.hills");
    expect(reliefName(".png")).toBe("Relief");
  });
});
