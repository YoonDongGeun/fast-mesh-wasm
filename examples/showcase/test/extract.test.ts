// The DICOM extraction pipeline: files parsed, series stacked, each
// isosurface meshed (Otsu's threshold unless one is given), named for the
// list. Failures name the file or series and the reason.
import { describe, expect, it } from "vitest";
import { extractSeries } from "../src/load/extract";
import { dicomFile } from "./dicom-fixture";

function sphereSeries(uid = "9.9.9", description?: string): { name: string; buffer: ArrayBuffer }[] {
  const out: { name: string; buffer: ArrayBuffer }[] = [];
  for (let z = 0; z < 8; z++) {
    const pixels: number[] = [];
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const d = Math.hypot(x - 3.5, y - 3.5, z - 3.5);
        pixels.push(d <= 2.5 ? 100 : 0);
      }
    }
    out.push({
      name: `s${z}.dcm`,
      buffer: dicomFile({
        seriesUid: uid,
        instanceUid: `${uid}.${z}`,
        rows: 8,
        columns: 8,
        position: [0, 0, z],
        pixels,
        description,
      }),
    });
  }
  return out;
}

describe("extractSeries", () => {
  it("meshes_a_sphere", () => {
    const [m] = extractSeries(sphereSeries("9.9.9", "CT Ball"), null);
    expect(m.threshold).toBeGreaterThan(0);
    expect(m.threshold).toBeLessThan(100);
    expect(m.auto).toBe(true);
    expect(m.name).toBe(`CT Ball (iso ${Math.round(m.threshold * 100) / 100})`);
    expect(m.data.indices.length).toBeGreaterThan(0);
    // A closed ball: every directed edge has its reverse.
    const seen = new Set<number>();
    const idx = m.data.indices;
    for (let t = 0; t < idx.length; t += 3) {
      for (const [a, b] of [
        [idx[t], idx[t + 1]],
        [idx[t + 1], idx[t + 2]],
        [idx[t + 2], idx[t]],
      ]) {
        seen.add(a * 4294967296 + b);
      }
    }
    for (const k of seen) {
      expect(seen.has((k % 4294967296) * 4294967296 + Math.floor(k / 4294967296))).toBe(true);
    }
  });

  it("honors_an_explicit_threshold", () => {
    const [m] = extractSeries(sphereSeries(), 50);
    expect(m.threshold).toBe(50);
    expect(m.auto).toBe(false);
    expect(m.name).toContain("iso 50");
    expect(m.data.indices.length).toBeGreaterThan(0);
  });

  it("names_an_undescribed_series", () => {
    const [m] = extractSeries(sphereSeries("1.2.3.99"), null);
    expect(m.name).toMatch(/^DICOM .* \(iso .*\)$/);
  });

  it("fails_lone_and_foreign_files", () => {
    expect(() => extractSeries(sphereSeries().slice(0, 1), null)).toThrow(/whole series/i);
    expect(() => extractSeries([{ name: "a.stl", buffer: new Uint8Array([1, 2]).buffer as ArrayBuffer }], null)).toThrow(
      /a\.stl.*DICOM/i,
    );
  });
});
