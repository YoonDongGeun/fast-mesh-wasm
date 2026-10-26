// The DICOM phantoms: vessel tree, brain and heart SDFs sampled into
// slice stacks, meshed through the same extraction pipeline as dropped
// files (so the demos prove the path, not a second one).
import { describe, expect, it } from "vitest";
import { meshSlices } from "../src/load/extract";
import { phantomSlices, type PhantomKind } from "../src/load/phantom";
import { stackSeries } from "../src/load/series";

const KINDS: PhantomKind[] = ["vessels", "brain", "heart"];

describe("phantomSlices", () => {
  it.each(KINDS)("%s stacks into one series", (kind) => {
    const slices = phantomSlices(kind);
    expect(slices.length).toBeGreaterThan(8);
    const [v] = stackSeries(slices);
    expect(v.slices).toBe(slices.length);
    // Inside exists (negative SDF) and so does air.
    let inside = 0;
    let outside = 0;
    for (const s of slices) {
      for (let i = 0; i < s.pixels.length; i++) {
        if (s.pixels[i] < 0) {
          inside++;
        } else {
          outside++;
        }
      }
    }
    expect(inside).toBeGreaterThan(100);
    expect(outside).toBeGreaterThan(100);
  });

  it("is_deterministic", () => {
    const a = phantomSlices("vessels");
    const b = phantomSlices("vessels");
    expect(a.length).toBe(b.length);
    for (let i = 0; i < a.length; i++) {
      expect(Array.from(a[i].pixels as Int16Array)).toEqual(Array.from(b[i].pixels as Int16Array));
    }
  });

  it.each(KINDS)("%s meshes a non-empty surface at zero", (kind) => {
    const [m] = meshSlices(phantomSlices(kind), 0);
    expect(m.threshold).toBe(0);
    expect(m.data.indices.length).toBeGreaterThan(100);
    expect(m.name).toContain("(iso 0)");
    // Every vertex sits inside the volume's millimetre box.
    const [v] = stackSeries(phantomSlices(kind));
    const p = m.data.positions;
    for (let i = 0; i < p.length; i += 3) {
      expect(p[i]).toBeGreaterThanOrEqual(v.origin[0] - 1e-6);
      expect(p[i]).toBeLessThanOrEqual(v.origin[0] + v.columns * v.spacing[0] + 1e-6);
    }
  });
});
