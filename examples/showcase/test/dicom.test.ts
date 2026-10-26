// The DICOM slice parser: Part-10 explicit and implicit VR little endian,
// the geometry and rescale tags a series stack needs, and raw pixels.
// Encapsulated (compressed) transfer syntaxes fail with a clear message.
import { describe, expect, it } from "vitest";
import { isDicom, parseDicom } from "../src/load/dicom";
import { dicomFile } from "./dicom-fixture";

describe("parseDicom", () => {
  it("reads_an_explicit_slice", () => {
    const s = parseDicom(
      "slice.dcm",
      dicomFile({
        seriesUid: "1.2.3",
        instanceUid: "1.2.3.1",
        rows: 2,
        columns: 2,
        pixels: [0, 100, 200, 300],
        slope: 2,
        intercept: -100,
        pixelSpacing: [0.5, 0.5],
        position: [10, 20, 30],
      }),
    );
    expect(s.seriesUid).toBe("1.2.3");
    expect(s.instanceUid).toBe("1.2.3.1");
    expect([s.rows, s.columns]).toEqual([2, 2]);
    expect([s.rescaleSlope, s.rescaleIntercept]).toEqual([2, -100]);
    expect(s.pixelSpacing).toEqual([0.5, 0.5]);
    expect(s.position).toEqual([10, 20, 30]);
    expect(s.orientation).toEqual([1, 0, 0, 0, 1, 0]);
    expect(Array.from(s.pixels as Uint16Array)).toEqual([0, 100, 200, 300]);
  });

  it("reads_an_implicit_slice", () => {
    const s = parseDicom("slice", dicomFile({ implicit: true, pixels: [7, 8, 9, 10] }));
    expect([s.rows, s.columns]).toEqual([2, 2]);
    expect(Array.from(s.pixels as Uint16Array)).toEqual([7, 8, 9, 10]);
  });

  it("reads_signed_8bit", () => {
    const s = parseDicom(
      "s.dcm",
      dicomFile({ bitsAllocated: 8, pixelRepresentation: 0, pixels: [1, 2, 3, 4] }),
    );
    expect(Array.from(s.pixels as Uint8Array)).toEqual([1, 2, 3, 4]);
    const t = parseDicom(
      "t.dcm",
      dicomFile({ pixelRepresentation: 1, pixels: [0xffff, 1, 2, 3] }),
    );
    expect(Array.from(t.pixels as Int16Array)).toEqual([-1, 1, 2, 3]);
  });

  it("skips_unknown_elements", () => {
    const s = parseDicom(
      "s.dcm",
      dicomFile({
        pixels: [5, 6, 7, 8],
        extra: [
          { group: 0x0010, element: 0x0010, vr: "PN", value: new TextEncoder().encode("Anon^Test ") },
          { group: 0x0029, element: 0x0010, vr: "LO", value: new TextEncoder().encode("PRIVATE!! ") },
        ],
      }),
    );
    expect(Array.from(s.pixels as Uint16Array)).toEqual([5, 6, 7, 8]);
  });

  it("reads_series_description", () => {
    expect(parseDicom("a.dcm", dicomFile({})).description).toBe("");
    expect(parseDicom("b.dcm", dicomFile({ description: "CT Chest" })).description).toBe("CT Chest");
  });

  it("flags_inverted_monochrome", () => {
    expect(parseDicom("a.dcm", dicomFile({})).inverted).toBe(false);
    expect(parseDicom("b.dcm", dicomFile({ photometric: "MONOCHROME1" })).inverted).toBe(true);
  });

  it("rejects_non_dicom", () => {
    expect(() => parseDicom("a.stl", new Uint8Array([1, 2, 3, 4]).buffer as ArrayBuffer)).toThrow(
      /not a DICOM/i,
    );
    expect(isDicom(new Uint8Array(200).buffer as ArrayBuffer)).toBe(false);
  });

  it("rejects_compressed_syntaxes", () => {
    const buf = dicomFile({ transferSyntax: "1.2.840.10008.1.2.4.70" });
    expect(() => parseDicom("j.dcm", buf)).toThrow(/transfer syntax/i);
  });

  it("rejects_rgb", () => {
    const raw = new Uint8Array(dicomFile({}));
    // SamplesPerPixel lives in the dataset; flip its single US value to 3.
    // The fixture writes it right after SeriesInstanceUID; find the tag.
    const at = findTag(raw, 0x0028, 0x0002);
    const view = new DataView(raw.buffer);
    view.setUint16(at, 3, true);
    expect(() => parseDicom("rgb.dcm", raw.buffer as ArrayBuffer)).toThrow(/samples/i);
  });
});

/** The value offset of an explicit-VR US element's first value. */
function findTag(raw: Uint8Array, group: number, element: number): number {
  for (let i = 132; i + 8 < raw.length; i++) {
    if (raw[i] === (group & 0xff) && raw[i + 1] === (group >> 8) && raw[i + 2] === (element & 0xff) && raw[i + 3] === (element >> 8)) {
      return i + 8;
    }
  }
  throw new Error("tag not found");
}
