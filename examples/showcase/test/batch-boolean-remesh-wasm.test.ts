// Real-wasm tests for #254: the multi-mesh boolean over three or more inputs
// (union all, primary minus others, intersect all; inputs kept or not) and
// the isotropic remesh, each one undo step that restores the inputs, on the
// sync `Scene`, on the demo meshes, and on a `WorkerScene` run in a Node
// worker thread.
import { Worker } from "node:worker_threads";
import { type MeshHandle, Scene, type SceneLike } from "fast-mesh-wasm";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { createWorkerScene, type WorkerLike } from "fast-mesh-wasm/worker";
import { afterEach, describe, expect, it } from "vitest";
import { AppState, type Item } from "../src/app/state/app-state";
import { duplicateMesh, runBatchBoolean, runIsotropicRemesh } from "../src/engine/workbench-ops";
import { selectMesh } from "../src/features/select/ux/selection";
import { batchInputs } from "../src/features/mesh-edit/ux/batch-boolean";
import { defaultTargetEdge, edgeStats } from "../src/features/mesh-edit/ux/isotropic-remesh";
import { demoMeshes } from "../src/io/demo";
import { bounds } from "../src/io/mesh-data";

function boxGeometry(h: number, c: readonly [number, number, number]) {
  const [cx, cy, cz] = c;
  const [x0, x1, y0, y1, z0, z1] = [cx - h, cx + h, cy - h, cy + h, cz - h, cz + h];
  return {
    positions: Float64Array.from([x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1]),
    indices: Uint32Array.from([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5]),
  };
}

function addBox(
  store: SceneStore,
  app: AppState,
  name: string,
  center: readonly [number, number, number],
  translation?: readonly [number, number, number],
): Item {
  const g = boxGeometry(1, center);
  const item = app.addMesh(store, { name, ...g, center: [...center], radius: Math.sqrt(3), volumeId: null, preset: null });
  if (translation !== undefined) {
    store.setPlacement(item.mesh, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [...translation] });
  }
  return item;
}

/** The demo meshes as the loader lists them (origin at each bounding-box centre). */
function addDemos(store: SceneStore, app: AppState): Item[] {
  return demoMeshes(false).map((d) =>
    app.addMesh(store, { name: d.name, ...d.data, ...bounds(d.data.positions), volumeId: null, preset: null }),
  );
}

function liveTriangles(scene: SceneLike, mesh: MeshHandle): number {
  const idx = scene.views(mesh).indices();
  let n = 0;
  for (let k = 0; k + 2 < idx.length; k += 3) {
    if (idx[k] !== 0 || idx[k + 1] !== 0 || idx[k + 2] !== 0) {
      n++;
    }
  }
  return n;
}

/** The mesh's enclosed world-space volume (its placement applied). */
function worldVolume(scene: SceneLike, mesh: MeshHandle): number {
  const v = scene.views(mesh);
  const p = v.positions();
  const idx = v.indices();
  const m = scene.matrix4(mesh);
  const w = (i: number): [number, number, number] => [
    m[0] * p[3 * i] + m[4] * p[3 * i + 1] + m[8] * p[3 * i + 2] + m[12],
    m[1] * p[3 * i] + m[5] * p[3 * i + 1] + m[9] * p[3 * i + 2] + m[13],
    m[2] * p[3 * i] + m[6] * p[3 * i + 1] + m[10] * p[3 * i + 2] + m[14],
  ];
  let vol6 = 0;
  for (let k = 0; k + 2 < idx.length; k += 3) {
    if (idx[k] === 0 && idx[k + 1] === 0 && idx[k + 2] === 0) {
      continue;
    }
    const [ax, ay, az] = w(idx[k]);
    const [bx, by, bz] = w(idx[k + 1]);
    const [cx, cy, cz] = w(idx[k + 2]);
    vol6 += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
  }
  return vol6 / 6;
}

/** Every input's world volume and triangle count, to compare after undo. */
function shapes(scene: SceneLike, items: readonly Item[]): [number, number][] {
  return items.map((i) => [Number(worldVolume(scene, i.mesh).toFixed(6)), liveTriangles(scene, i.mesh)]);
}

/** The demo meshes take about a second each run; allow for a loaded machine. */
const DEMO_TIMEOUT_MS = 60_000;

const cleanups: (() => void | Promise<void>)[] = [];

afterEach(async () => {
  while (cleanups.length > 0) {
    await cleanups.pop()?.();
  }
});

function syncSetup(): { scene: Scene; store: SceneStore; app: AppState } {
  const scene = new Scene();
  const store = createSceneStore(scene);
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return { scene, store, app: new AppState() };
}

/**
 * A, B and C: unit-half boxes at the origin, at +x and (by placement) at +y.
 * Union 16, A minus the others 2, intersection 2.
 */
function threeBoxes(store: SceneStore, app: AppState): Item[] {
  return [
    addBox(store, app, "A", [0, 0, 0]),
    addBox(store, app, "B", [1, 0, 0]),
    addBox(store, app, "C", [0, 0, 0], [0, 1, 0]),
  ];
}

/** Shift-clicks `items` in order, so the first is the primary. */
function shiftSelect(store: SceneStore, app: AppState, items: readonly Item[]): void {
  items.forEach((it, i) => selectMesh(app, (m) => store.select(m), store.getSnapshot().selection, it.mesh, i > 0));
}

describe("multi-mesh boolean (sync Scene)", () => {
  it("reads three shift-clicked meshes as its inputs, primary first", () => {
    const { store, app } = syncSetup();
    const [a, b, c] = threeBoxes(store, app);
    shiftSelect(store, app, [b, c, a]);
    expect(batchInputs(app.getSnapshot().multi, store.getSnapshot().selection, app.getSnapshot().items)).toEqual([b, c, a]);
  });

  it("keeps the inputs: the result fills a new mesh in one undo step", async () => {
    const { scene, store, app } = syncSetup();
    const inputs = threeBoxes(store, app);
    const before = shapes(scene, inputs);
    for (const [op, volume] of [
      ["union", 16],
      ["subtract", 2],
      ["intersect", 2],
    ] as const) {
      const count = app.getSnapshot().items.length;
      const r = await runBatchBoolean({ store, app, inputs, op, keepInputs: true });
      expect(app.getSnapshot().items.length).toBe(count + 1);
      expect(app.getSnapshot().items.at(-1)?.mesh).toBe(r.mesh);
      expect(r.name).toBe(`A ${op === "union" ? "union" : op === "subtract" ? "difference" : "intersection"} +2`);
      expect(store.getSnapshot().selection).toBe(r.mesh);
      expect(app.getSnapshot().multi).toEqual([r.mesh]);
      expect(worldVolume(scene, r.mesh)).toBeCloseTo(volume, 4);
      expect(liveTriangles(scene, r.mesh)).toBe(r.triangles);
      expect(shapes(scene, inputs)).toEqual(before);
    }
    // One undo empties the last result only; redo fills it again.
    const last = app.getSnapshot().items.at(-1)!;
    store.undo();
    expect(liveTriangles(scene, last.mesh)).toBe(0);
    expect(liveTriangles(scene, app.getSnapshot().items.at(-2)!.mesh)).toBeGreaterThan(0);
    store.redo();
    expect(worldVolume(scene, last.mesh)).toBeCloseTo(2, 4);
    expect(shapes(scene, inputs)).toEqual(before);
  });

  it("removes the inputs: the primary takes the result, the others empty, and one undo restores all three", async () => {
    const { scene, store, app } = syncSetup();
    // The primary is placed, so the result maps into its local frame.
    const a = addBox(store, app, "A", [5, 0, 0], [-5, 0, 0]);
    const b = addBox(store, app, "B", [1, 0, 0]);
    const c = addBox(store, app, "C", [0, 0, 0], [0, 1, 0]);
    const inputs = [a, b, c];
    const before = shapes(scene, inputs);
    const placementBefore = scene.placement(a.mesh);

    const r = await runBatchBoolean({ store, app, inputs, op: "subtract", keepInputs: false });
    expect(r.mesh).toBe(a.mesh);
    expect(r.name).toBe("A");
    expect(app.getSnapshot().items.length).toBe(3);
    expect(worldVolume(scene, a.mesh)).toBeCloseTo(2, 4);
    expect(scene.placement(a.mesh)).toEqual(placementBefore);
    expect(liveTriangles(scene, b.mesh)).toBe(0);
    expect(liveTriangles(scene, c.mesh)).toBe(0);
    expect(store.getSnapshot().selection).toBe(a.mesh);

    store.undo();
    expect(shapes(scene, inputs)).toEqual(before);
    // Only the boolean was undone: the placements before it still stand.
    expect(scene.placement(c.mesh).translation).toEqual([0, 1, 0]);
    store.redo();
    expect(worldVolume(scene, a.mesh)).toBeCloseTo(2, 4);
    expect(liveTriangles(scene, b.mesh) + liveTriangles(scene, c.mesh)).toBe(0);
  });

  it("changes nothing when the result is empty", async () => {
    const { scene, store, app } = syncSetup();
    const inputs = [addBox(store, app, "A", [0, 0, 0]), addBox(store, app, "B", [1, 0, 0]), addBox(store, app, "Far", [9, 0, 0])];
    const before = shapes(scene, inputs);
    await expect(runBatchBoolean({ store, app, inputs, op: "intersect", keepInputs: false })).rejects.toThrow(/do not all overlap/);
    expect(shapes(scene, inputs)).toEqual(before);
    expect(app.getSnapshot().items.length).toBe(3);
  });

  it("combines the demo meshes (A, B and a moved copy of A) and undo restores them", async () => {
    const { scene, store, app } = syncSetup();
    const [a, b] = addDemos(store, app);
    const copy = duplicateMesh(store, app, a);
    store.setPlacement(copy.mesh, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0.6, 0.5, 0.3] });
    const inputs = [a, b, copy];
    const before = shapes(scene, inputs);
    const volumes = before.map(([v]) => v);

    const union = await runBatchBoolean({ store, app, inputs, op: "union", keepInputs: false });
    const unionVolume = worldVolume(scene, union.mesh);
    expect(unionVolume).toBeGreaterThan(Math.max(...volumes));
    expect(unionVolume).toBeLessThan(volumes.reduce((s, v) => s + v, 0));
    store.undo();
    expect(shapes(scene, inputs)).toEqual(before);

    const diff = await runBatchBoolean({ store, app, inputs, op: "subtract", keepInputs: true });
    expect(worldVolume(scene, diff.mesh)).toBeLessThan(volumes[0]);
    const inter = await runBatchBoolean({ store, app, inputs, op: "intersect", keepInputs: true });
    expect(worldVolume(scene, inter.mesh)).toBeGreaterThan(0);
    expect(worldVolume(scene, inter.mesh)).toBeLessThan(Math.min(...volumes));
    expect(shapes(scene, inputs)).toEqual(before);
  }, DEMO_TIMEOUT_MS);

  it("takes an open demo mesh through a voxel backend", async () => {
    const { scene, store, app } = syncSetup();
    const inputs = addDemos(store, app);
    // The exact boolean needs closed inputs; C is an open shell.
    await expect(runBatchBoolean({ store, app, inputs, op: "union", keepInputs: true })).rejects.toThrow(/closed/);
    expect(app.getSnapshot().items.length).toBe(3);
    const r = await runBatchBoolean({ store, app, inputs, op: "union", keepInputs: true, method: "bvhSdf" });
    expect(r.triangles).toBeGreaterThan(0);
    expect(liveTriangles(scene, r.mesh)).toBe(r.triangles);
  }, DEMO_TIMEOUT_MS);
});

describe("isotropic remesh (sync Scene)", () => {
  it("remeshes a demo mesh at its mean edge length in one undo step", async () => {
    const { scene, store, app } = syncSetup();
    const [, b] = addDemos(store, app);
    const v = scene.views(b.mesh);
    const stats = edgeStats(v.positions(), v.indices());
    expect(stats.triangles).toBe(liveTriangles(scene, b.mesh));
    const volume = worldVolume(scene, b.mesh);

    const target = defaultTargetEdge(stats.meanEdge) * 0.75;
    const r = await runIsotropicRemesh({
      store,
      mesh: b.mesh,
      options: { targetEdgeLength: target, iterations: 4, lockBoundaries: true, lockCreases: true, creaseAngleDeg: 35 },
    });
    expect(r.initialTriangles).toBe(stats.triangles);
    expect(r.finalTriangles).toBe(liveTriangles(scene, b.mesh));
    expect(r.finalTriangles).toBeGreaterThan(stats.triangles);
    expect(r.iterationsRun).toBe(4);
    const after = scene.views(b.mesh);
    // Isotropic: the mean edge moves toward the target.
    expect(Math.abs(edgeStats(after.positions(), after.indices()).meanEdge - target)).toBeLessThan(
      Math.abs(stats.meanEdge - target),
    );
    expect(worldVolume(scene, b.mesh)).toBeCloseTo(volume, 2);

    store.undo();
    expect(liveTriangles(scene, b.mesh)).toBe(stats.triangles);
    expect(worldVolume(scene, b.mesh)).toBeCloseTo(volume, 6);
  }, DEMO_TIMEOUT_MS);
});

/** The SDK's Node worker-thread stand-in for the browser engine Worker. */
class NodeWorker implements WorkerLike {
  readonly worker = new Worker(new URL("../../../packages/fast-mesh-wasm/test/harness/node-worker.mjs", import.meta.url), {
    workerData: { serve: true },
  });
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  readonly #error: ((e: unknown) => void)[] = [];

  constructor() {
    this.worker.on("message", (data: unknown) => {
      for (const cb of this.#message) {
        cb({ data });
      }
    });
    this.worker.on("error", (e) => {
      for (const cb of this.#error) {
        cb(e);
      }
    });
  }

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", cb: (e: never) => void): void {
    (type === "message" ? this.#message : this.#error).push(cb as (e: unknown) => void);
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

/** Frames until the Worker has answered everything sent (at once on a local scene). */
async function settle(store: SceneStore): Promise<void> {
  for (let k = 0; k < 2000; k++) {
    store.commitFrame();
    if (store.mode === "local") {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 2));
    if (store.getSnapshot().status !== "computing") {
      store.commitFrame();
      return;
    }
  }
  throw new Error("the store did not settle");
}

describe("multi-mesh boolean and isotropic remesh (WorkerScene)", () => {
  async function workerSetup(): Promise<{ store: SceneStore; app: AppState }> {
    const scene = await createWorkerScene({ createWorker: () => new NodeWorker() });
    const store = createSceneStore(scene);
    cleanups.push(() => {
      store.dispose();
      scene.terminate();
    });
    return { store, app: new AppState() };
  }

  it("runs both off the main thread, each one undo step", async () => {
    const { store, app } = await workerSetup();
    expect(store.mode).toBe("worker");
    const scene = store.scene;
    const inputs = threeBoxes(store, app);
    await settle(store);
    const before = shapes(scene, inputs);

    const kept = await runBatchBoolean({ store, app, inputs, op: "union", keepInputs: true });
    await settle(store);
    expect(worldVolume(scene, kept.mesh)).toBeCloseTo(16, 4);
    expect(app.getSnapshot().items.find((i) => i.mesh === kept.mesh)?.triangles).toBe(kept.triangles);
    store.undo();
    await settle(store);
    expect(liveTriangles(scene, kept.mesh)).toBe(0);

    const r = await runBatchBoolean({ store, app, inputs, op: "intersect", keepInputs: false });
    await settle(store);
    expect(worldVolume(scene, r.mesh)).toBeCloseTo(2, 4);
    expect(liveTriangles(scene, inputs[1].mesh) + liveTriangles(scene, inputs[2].mesh)).toBe(0);
    store.undo();
    await settle(store);
    expect(shapes(scene, inputs)).toEqual(before);

    const remesh = await runIsotropicRemesh({ store, mesh: inputs[0].mesh, options: { targetEdgeLength: 0.5 } });
    await settle(store);
    expect(remesh.initialTriangles).toBe(12);
    expect(liveTriangles(scene, inputs[0].mesh)).toBe(remesh.finalTriangles);
    expect(remesh.finalTriangles).toBeGreaterThan(12);
    store.undo();
    await settle(store);
    expect(shapes(scene, inputs)).toEqual(before);
  }, 60_000);
});
