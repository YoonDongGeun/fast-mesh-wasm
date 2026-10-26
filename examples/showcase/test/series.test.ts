// The series stack: slices grouped by series, sorted along the slice
// normal, rescaled into a regular float volume with its patient geometry.
// Lone, inconsistent or tilted series fail with a clear message.
import { describe, expect, it } from "vitest";
import type { DicomSlice } from "../src/load/dicom";
import { parseDicom } from "../src/load/dicom";
import { stackSeries } from "../src/load/series";
import { dicomFile } from "./dicom-fixture";

function slice(z: number, pixels: number[], extra = {}): DicomSlice {
  return parseDicom(
    `s${z}.dcm`,
    dicomFile({ rows: 2, columns: 2, position: [0, 0, z], pixels, ...extra }),
  );
}

describe("stackSeries", () => {
  it("stacks_slices_in_order", () => {
    const [v] = stackSeries([slice(1, [4, 5, 6, 7]), slice(0, [0, 1, 2, 3])]);
    expect([v.columns, v.rows, v.slices]).toEqual([2, 2, 2]);
    expect(Array.from(v.values)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(v.spacing).toEqual([1, 1, 1]);
    expect(v.origin).toEqual([0, 0, 0]);
    expect(v.seriesUid).toBe("1.2.3");
  });

  it("keeps_patient_geometry", () => {
    const [v] = stackSeries([
      slice(0, [0, 0, 0, 0], { pixelSpacing: [0.5, 0.25], position: [10, 20, 30] }),
      slice(2, [0, 0, 0, 0], { pixelSpacing: [0.5, 0.25], position: [10, 20, 32] }),
    ]);
    expect(v.spacing).toEqual([0.25, 0.5, 2]);
    expect(v.origin).toEqual([10, 20, 30]);
    expect(v.xStep).toEqual([0.25, 0, 0]);
    expect(v.yStep).toEqual([0, 0.5, 0]);
    expect(v.sliceStep).toEqual([0, 0, 2]);
  });

  it("rescales_per_slice", () => {
    const [v] = stackSeries([
      slice(0, [100, 100, 100, 100], { slope: 1, intercept: -100 }),
      slice(1, [100, 100, 100, 100], { slope: 2, intercept: 0 }),
    ]);
    expect(Array.from(v.values)).toEqual([0, 0, 0, 0, 200, 200, 200, 200]);
  });

  it("groups_series", () => {
    const vols = stackSeries([
      slice(0, [1, 1, 1, 1], { seriesUid: "a" }),
      slice(0, [2, 2, 2, 2], { seriesUid: "b" }),
      slice(1, [3, 3, 3, 3], { seriesUid: "a" }),
      slice(1, [4, 4, 4, 4], { seriesUid: "b" }),
    ]);
    expect(vols.map((v) => v.seriesUid).sort()).toEqual(["a", "b"]);
    expect(vols.find((v) => v.seriesUid === "b")?.values[0]).toBe(2);
  });

  it("rejects_a_lone_slice", () => {
    expect(() => stackSeries([slice(0, [0, 0, 0, 0])])).toThrow(/whole series/i);
  });

  it("rejects_mixed_frames", () => {
    const other = parseDicom("x.dcm", dicomFile({ rows: 3, columns: 2, position: [0, 0, 1] }));
    expect(() => stackSeries([slice(0, [0, 0, 0, 0]), other])).toThrow(/rows|columns|frame/i);
  });

  it("rejects_tilted_series", () => {
    expect(() =>
      stackSeries([slice(0, [0, 0, 0, 0]), slice(1, [0, 0, 0, 0], { position: [0.5, 0, 1] })]),
    ).toThrow(/tilt/i);
  });

  it("rejects_irregular_gaps", () => {
    expect(() =>
      stackSeries([slice(0, [0, 0, 0, 0]), slice(1, [0, 0, 0, 0]), slice(3, [0, 0, 0, 0])]),
    ).toThrow(/spacing/i);
  });
});
