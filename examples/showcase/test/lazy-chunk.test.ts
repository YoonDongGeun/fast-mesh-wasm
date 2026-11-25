// Chunks on demand (#285): one import per chunk, again after a failure, and the
// count of the parts that wait for one; when the idle load may start; and what
// the page imports statically, which must not bring a feature chunk back in.
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppState } from "../src/app/state/app-state";
import { whenFetched } from "../src/app/ux/feature-chunks";
import { LazyChunk, loadWaitingParts, partsArriving, waitForChunk } from "../src/shared/ux/lazy-chunk";

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

/** An import the test settles. */
function pending<M>() {
  let resolveIt!: (m: M) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<M>((a, b) => {
    resolveIt = a;
    reject = b;
  });
  return { promise, resolve: resolveIt, reject };
}

describe("LazyChunk", () => {
  it("imports once, however often it is asked, and tells its listeners", async () => {
    const gate = pending<{ x: number }>();
    const importer = vi.fn(() => gate.promise);
    const chunk = new LazyChunk("x", importer);
    const seen: string[] = [];
    chunk.subscribe(() => seen.push(chunk.getSnapshot().status));
    expect(chunk.getSnapshot().status).toBe("idle");

    const first = chunk.load();
    expect(chunk.load()).toBe(first);
    gate.resolve({ x: 1 });
    expect(await first).toBe(true);
    expect(await chunk.load()).toBe(true);

    expect(importer).toHaveBeenCalledTimes(1);
    expect(seen).toEqual(["loading", "ready"]);
    expect(chunk.getSnapshot()).toEqual({ status: "ready", module: { x: 1 } });
  });

  it("reports a failure without rejecting, and imports again when asked again", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const error = new Error("offline");
    const importer = vi.fn<() => Promise<{ x: number }>>().mockRejectedValueOnce(error).mockResolvedValueOnce({ x: 2 });
    const chunk = new LazyChunk("x", importer);

    expect(await chunk.load()).toBe(false);
    expect(chunk.getSnapshot()).toEqual({ status: "failed", error });
    expect(warn).toHaveBeenCalledTimes(1);

    expect(await chunk.load()).toBe(true);
    expect(chunk.getSnapshot()).toEqual({ status: "ready", module: { x: 2 } });
    expect(importer).toHaveBeenCalledTimes(2);
  });

  it("stops telling a listener that left", async () => {
    const chunk = new LazyChunk("x", () => Promise.resolve({}));
    const listener = vi.fn();
    chunk.subscribe(listener)();
    await chunk.load();
    expect(listener).not.toHaveBeenCalled();
  });
});

describe("waiting parts", () => {
  it("count per chunk until each is done waiting, and load together", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const gate = pending<object>();
    const good = new LazyChunk("good", () => gate.promise);
    const bad = new LazyChunk("bad", () => Promise.reject(new Error("gone")));
    expect(loadWaitingParts()).toBe(false);
    expect(partsArriving()).toBe(false);

    const doneA = waitForChunk(good);
    const doneB = waitForChunk(good);
    const doneC = waitForChunk(bad);
    expect(loadWaitingParts()).toBe(true);
    expect(good.getSnapshot().status).toBe("loading");
    await vi.waitFor(() => expect(bad.getSnapshot().status).toBe("failed"));
    expect(partsArriving()).toBe(true);

    doneA();
    doneA(); // Twice is once.
    expect(partsArriving()).toBe(true);
    doneB();
    // Only the failed one is left: nothing is arriving, but it can be tried again.
    expect(partsArriving()).toBe(false);
    expect(loadWaitingParts()).toBe(true);
    doneC();
    expect(loadWaitingParts()).toBe(false);
  });
});

describe("whenFetched", () => {
  const job = (over: object) => ({ id: 1, name: "head", phase: "reading" as const, progress: 0, ...over });

  it("runs at once with nothing on its way", () => {
    const run = vi.fn();
    whenFetched(new AppState(), run);
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("waits for the mesh that is being fetched, not for its parsing", () => {
    const app = new AppState();
    app.setJob(job({}));
    const run = vi.fn();
    whenFetched(app, run);
    app.setJob(job({ phase: "parsing" }));
    expect(run).not.toHaveBeenCalled();
    app.setJob(job({ phase: "parsing", progress: 0.4 }));
    expect(run).toHaveBeenCalledTimes(1);
    app.setJob(job({ phase: "done", progress: 1 }));
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("does not wait for what was asked for after it", () => {
    const app = new AppState();
    app.setJob(job({}));
    const run = vi.fn();
    whenFetched(app, run);
    // Queued behind the first in the parser's Worker.
    app.setJob(job({ id: 2, name: "demo" }));
    expect(run).not.toHaveBeenCalled();
    app.setJob(job({ phase: "parsing", progress: 0.4 }));
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("does not wait for a failed or finished load, nor for ever", () => {
    vi.useFakeTimers();
    const app = new AppState();
    app.setJob(job({ phase: "error" }));
    const now = vi.fn();
    whenFetched(app, now);
    expect(now).toHaveBeenCalledTimes(1);

    const slow = new AppState();
    slow.setJob(job({ id: 2 }));
    const later = vi.fn();
    whenFetched(slow, later);
    vi.advanceTimersByTime(4900);
    expect(later).not.toHaveBeenCalled();
    vi.advanceTimersByTime(200);
    expect(later).toHaveBeenCalledTimes(1);
  });

  it("can be cancelled", () => {
    vi.useFakeTimers();
    const app = new AppState();
    app.setJob(job({}));
    const run = vi.fn();
    whenFetched(app, run)();
    app.setJob(job({ progress: 0.5 }));
    vi.advanceTimersByTime(10_000);
    expect(run).not.toHaveBeenCalled();
  });
});

describe("the page's static imports", () => {
  const src = resolve(import.meta.dirname, "..", "src");

  /** The file a relative import names, if it is one of the app's modules. */
  function moduleFile(from: string, spec: string): string | null {
    if (!spec.startsWith(".")) {
      return null;
    }
    const base = join(dirname(from), spec);
    return [base, `${base}.ts`, `${base}.tsx`].find((f) => /\.tsx?$/.test(f) && existsSync(f)) ?? null;
  }

  /** Every module `entry` reaches through static imports (type-only ones are dropped by the build). */
  function reached(entry: string): Set<string> {
    const seen = new Set<string>();
    const queue = [entry];
    for (let file = queue.pop(); file !== undefined; file = queue.pop()) {
      if (seen.has(file)) {
        continue;
      }
      seen.add(file);
      const code = readFileSync(file, "utf8");
      const statements = code.matchAll(/^(?:import|export)\s+(?!type\b)(?:[^"';]*?\sfrom\s+)?"([^"]+)"/gm);
      for (const [, spec] of statements) {
        const next = moduleFile(file, spec);
        if (next !== null) {
          queue.push(next);
        }
      }
    }
    return seen;
  }

  const page = new Set([...reached(join(src, "main.tsx"))].map((f) => relative(src, f).replaceAll("\\", "/")));

  it("are followed", () => {
    expect(page.has("app/App.tsx")).toBe(true);
    expect(page.has("viewport/ui/Viewport.tsx")).toBe(true);
    expect(page.has("features/home/ui/MeshList.tsx")).toBe(true);
  });

  it("leave the feature chunks and the test hook out", () => {
    const lazy = [...page].filter((f) => /-chunk\.ts$/.test(f) && f.startsWith("features/"));
    expect(lazy).toEqual([]);
    for (const file of [
      "app/testing/test-hook.ts",
      "features/build-plate/ui/BuildPlatePanel.tsx",
      "features/collision/ui/CollisionPanel.tsx",
      "features/ct/ui/VolumePanel.tsx",
      "features/home/ui/CreatePrimitiveDialog.tsx",
      "features/home/ui/ExportPanel.tsx",
      "features/home/ux/create-primitive.ts",
      "features/inspect/ui/InspectLayers.tsx",
      "features/inspect/ui/InspectPanel.tsx",
      "features/mesh-edit/ui/ModelingPanel.tsx",
      "features/mesh-edit/ux/text-mesh.ts",
      "features/mesh-edit/ux/relief.ts",
      "features/mesh-repair/ui/RepairPanel.tsx",
      "features/points/ui/PointsPanel.tsx",
      "features/polylines/ux/polyline-actions.ts",
      "features/sculpt/ui/BrushLayers.tsx",
      "features/sculpt/ui/DeformPanel.tsx",
      "features/select/ui/SelectPanel.tsx",
      "features/transform/ux/align-tools.ts",
      "io/height-image.ts",
      "io/parse.ts",
      "viewport/ux/use-sculpt-stroke.ts",
    ]) {
      expect(existsSync(join(src, file)), `${file} is there`).toBe(true);
      expect(page.has(file), `${file} is not in the page's chunk`).toBe(false);
    }
  });
});
