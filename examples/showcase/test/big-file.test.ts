// The pure large-file UX core (issue #113): URL thresholds, weight plans,
// memory estimates and the post-load summary.
import { describe, expect, it } from "vitest";
import { AppState } from "../src/scene/app-state";
import {
  BIG_DECIMATE_TARGET,
  BIG_DECIMATE_TRIANGLES,
  BIG_FILE_DEFAULTS,
  BIG_WARN_TRIANGLES,
  bigFilePlan,
  decimateStepTargets,
  decimateTargetFor,
  estimateMeshBytes,
  formatLoadMs,
  formatMeshBytes,
  formatTriangles,
  parseBigFileParams,
  summarizeLoad,
} from "../src/load/big-file";

describe("big-file params", () => {
  it("defaults", () => {
    expect(parseBigFileParams("")).toEqual({
      warnTriangles: BIG_WARN_TRIANGLES,
      decimateTriangles: BIG_DECIMATE_TRIANGLES,
      decimateTarget: BIG_DECIMATE_TARGET,
      decimateOnLoad: true,
      demoBig: false,
    });
  });

  it("overrides", () => {
    const p = parseBigFileParams("?warn-tris=10&decimate-tris=20&decimate-target=5&decimate=0&big");
    expect(p).toEqual({ warnTriangles: 10, decimateTriangles: 20, decimateTarget: 5, decimateOnLoad: false, demoBig: true });
  });

  it("ignores_garbage", () => {
    const p = parseBigFileParams("?warn-tris=many&decimate-tris=-3&decimate-target=0");
    expect(p.warnTriangles).toBe(BIG_FILE_DEFAULTS.warnTriangles);
    expect(p.decimateTriangles).toBe(BIG_FILE_DEFAULTS.decimateTriangles);
    expect(p.decimateTarget).toBe(BIG_FILE_DEFAULTS.decimateTarget);
    // Garbage never opts out of decimation or into the big demo.
    expect(p.decimateOnLoad).toBe(true);
    expect(p.demoBig).toBe(false);
  });
});

describe("big-file plan", () => {
  const opts = { warnTriangles: 500_000, decimateTriangles: 1_000_000 };
  it("boundaries", () => {
    expect(bigFilePlan(499_999, opts)).toEqual({ warn: false, offerDecimate: false });
    expect(bigFilePlan(500_000, opts)).toEqual({ warn: true, offerDecimate: false });
    expect(bigFilePlan(1_000_000, opts)).toEqual({ warn: true, offerDecimate: true });
  });

  it("decimate_target", () => {
    expect(decimateTargetFor(1_000_000, 200_000)).toBe(200_000);
    expect(decimateTargetFor(200_000, 200_000)).toBeNull();
    expect(decimateTargetFor(10, 200_000)).toBeNull();
  });

  it("decimate_step_targets", () => {
    const steps = decimateStepTargets(1_000_000, 200_000, 5);
    expect(steps).toHaveLength(5);
    expect(steps.at(-1)).toBe(200_000);
    for (let i = 0; i < steps.length; i++) {
      expect(steps[i]).toBeLessThan(i === 0 ? 1_000_000 : steps[i - 1]);
    }
    expect(decimateStepTargets(103, 100, 5)).toEqual([102, 101, 100]);
    expect(decimateStepTargets(1_000, 200, 1)).toEqual([200]);
    expect(decimateStepTargets(200, 200, 5)).toEqual([]);
    expect(decimateStepTargets(50, 200, 5)).toEqual([]);
  });
});

describe("big-file estimates", () => {
  it("memory_grows_with_triangles", () => {
    const small = estimateMeshBytes(1_000);
    const big = estimateMeshBytes(1_000_000);
    expect(big).toBeGreaterThan(small);
    // ~24 B per triangle (12 B indices + ~12 B positions for a closed mesh).
    expect(big).toBeGreaterThan(20_000_000);
    expect(big).toBeLessThan(30_000_000);
  });

  it("explicit_vertices", () => {
    expect(estimateMeshBytes(2, 4)).toBe(4 * 24 + 2 * 12);
  });

  it("formats", () => {
    expect(formatTriangles(12_288)).toBe("12,288");
    expect(formatTriangles(1_000_000)).toBe("1.00 M");
    expect(formatLoadMs(850)).toBe("850 ms");
    expect(formatLoadMs(2300)).toBe("2.3 s");
    expect(formatMeshBytes(24_000_000)).toBe("≈ 24.0 MB");
  });
});

describe("load summary", () => {
  it("plain", () => {
    expect(summarizeLoad({ name: "a.stl", triangles: 12_288, loadMs: 850, decimated: null })).toBe(
      "a.stl: 12,288 in 850 ms",
    );
  });

  it("decimated", () => {
    expect(
      summarizeLoad({ name: "big.stl", triangles: 1_048_576, loadMs: 2300, decimated: { from: 1_048_576, to: 200_000 } }),
    ).toBe("big.stl: 1.05 M in 2.3 s (decimated 1.05 M → 200,000)");
  });
});

describe("app-state large-file jobs", () => {
  it("decimate_defaults_on_and_toggles", () => {
    const app = new AppState();
    expect(app.getSnapshot().decimateOnLoad).toBe(true);
    app.setDecimateOnLoad(false);
    expect(app.getSnapshot().decimateOnLoad).toBe(false);
  });

  it("job_progress_and_summary_round_trip", () => {
    const app = new AppState();
    app.setJob({ id: 7, name: "big.stl", phase: "parsing", progress: 0.5 });
    expect(app.getSnapshot().jobs).toEqual([{ id: 7, name: "big.stl", phase: "parsing", progress: 0.5 }]);
    app.setJob({ id: 7, name: "big.stl", phase: "decimating", progress: 0.8 });
    expect(app.getSnapshot().jobs).toEqual([{ id: 7, name: "big.stl", phase: "decimating", progress: 0.8 }]);
    app.setJob({ id: 7, name: "big.stl", phase: "done", progress: 1, summary: "big.stl: 1.00 M in 2.3 s" });
    expect(app.getSnapshot().jobs[0]?.summary).toBe("big.stl: 1.00 M in 2.3 s");
  });
});
