// Real-wasm tests for #253 on the demo meshes: a moved copy of a demo mesh
// comes back by landmark pairs and by coarse + ICP, the rounded box's
// symmetry plane is found and mirrored across, each one undo step, on the
// sync `Scene` and on a `WorkerScene` run in a Node worker thread.
import { Worker } from "node:worker_threads";
import { Scene, type SceneLike } from "fast-mesh-wasm";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { createWorkerScene, type WorkerLike } from "fast-mesh-wasm/worker";
import { afterEach, describe, expect, it } from "vitest";
import { AppState, type Item } from "../src/app/state/app-state";
import {
  duplicateMesh,
  mirrorMesh,
  runDetectSymmetryPlane,
  runLandmarkAlign,
  runMultistageAlign,
} from "../src/engine/workbench-ops";
import {
  addLandmark,
  landmarkPairs,
  landmarkResiduals,
  multistageWarning,
  planeToLocal,
  planeToWorld,
  startLandmarks,
} from "../src/features/transform/ux/align-tools";
import { demoMeshes } from "../src/io/demo";
import { bounds } from "../src/io/mesh-data";
import { localHitPoint } from "../src/viewport/ux/surface-raycast";

const DEMO_TIMEOUT_MS = 60_000;
const cleanups: (() => void)[] = [];

afterEach(() => {
  for (const fn of cleanups.splice(0).reverse()) {
    fn();
  }
});

/** The demo meshes as the loader lists them (origin at each bounding-box centre). */
function addDemos(store: SceneStore, app: AppState): Item[] {
  return demoMeshes(false).map((d) =>
    app.addMesh(store, { name: d.name, ...d.data, ...bounds(d.data.positions), volumeId: null, preset: null }),
  );
}

const rotZ = (a: number): number[] => [Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a), 0, 0, 0, 1];
const rotX = (a: number): number[] => [1, 0, 0, 0, Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a)];

/** How far apart the two meshes' matching vertices are, at most (world). */
function worstGap(scene: SceneLike, a: Item, b: Item): number {
  const pa = scene.views(a.mesh).positions();
  const pb = scene.views(b.mesh).positions();
  const ma = scene.matrix4(a.mesh);
  const mb = scene.matrix4(b.mesh);
  let worst = 0;
  for (let i = 0; i < pa.length; i += 3) {
    let d2 = 0;
    for (let k = 0; k < 3; k++) {
      const wa = ma[k] * pa[i] + ma[4 + k] * pa[i + 1] + ma[8 + k] * pa[i + 2] + ma[12 + k];
      const wb = mb[k] * pb[i] + mb[4 + k] * pb[i + 1] + mb[8 + k] * pb[i + 2] + mb[12 + k];
      d2 += (wa - wb) ** 2;
    }
    worst = Math.max(worst, Math.sqrt(d2));
  }
  return worst;
}

/** A copy of `item`, turned and moved away. */
function movedCopy(store: SceneStore, app: AppState, item: Item, rotation: number[], translation: [number, number, number]): Item {
  const copy = duplicateMesh(store, app, item);
  store.setPlacement(copy.mesh, { rotation, translation });
  return copy;
}

/** Landmarks at the same triangle centroids on `moving` and `fixed`. */
function pairsAt(scene: SceneLike, moving: Item, fixed: Item, triangles: readonly number[]) {
  let set = startLandmarks(moving.mesh, fixed.mesh);
  const third = [1 / 3, 1 / 3, 1 / 3] as [number, number, number];
  for (const triangle of triangles) {
    set = addLandmark(set, localHitPoint(scene as Scene, { mesh: moving.mesh, triangle, barycentric: third }));
    set = addLandmark(set, localHitPoint(scene as Scene, { mesh: fixed.mesh, triangle, barycentric: third }));
  }
  return set;
}

describe("align tools on the demo meshes (Scene)", () => {
  function setup() {
    const scene = new Scene();
    const store = createSceneStore(scene);
    const app = new AppState();
    cleanups.push(() => {
      store.dispose();
      scene.free();
    });
    const demos = addDemos(store, app);
    return { scene, store, app, demos };
  }

  it("brings a moved copy of the blobby sphere back by landmark pairs, one undo step", async () => {
    const { scene, store, app, demos } = setup();
    const copy = movedCopy(store, app, demos[0], rotZ(0.6), [3, -2, 1]);
    scene.clearHistory();
    const set = pairsAt(scene, copy, demos[0], [10, 4000, 9000, 15000]);
    const r = await runLandmarkAlign({
      store,
      moving: copy.mesh,
      fixed: demos[0].mesh,
      pairs: landmarkPairs(set, scene.matrix4(copy.mesh), scene.matrix4(demos[0].mesh)),
      allowScaling: false,
    });
    expect(r.rmse).toBeLessThan(1e-9);
    expect(r.scale).toBe(1);
    expect(worstGap(scene, copy, demos[0])).toBeLessThan(1e-6);
    for (const d of landmarkResiduals(set, scene.matrix4(copy.mesh), scene.matrix4(demos[0].mesh))) {
      expect(d).toBeLessThan(1e-9);
    }
    store.undo();
    expect(scene.canUndo()).toBe(false);
    expect(scene.placement(copy.mesh).translation).toEqual([3, -2, 1]);
  }, DEMO_TIMEOUT_MS);

  it("brings a turned copy of the rounded box and the open shell back by coarse + ICP, one undo step each", async () => {
    const { scene, store, app, demos } = setup();
    for (const [source, rotation] of [
      [demos[1], rotX(0.5)],
      [demos[2], rotZ(Math.PI / 2)],
    ] as const) {
      const copy = movedCopy(store, app, source, [...rotation], [2.5, 1.5, -1]);
      scene.clearHistory();
      const { coarse, fine } = await runMultistageAlign({
        store,
        moving: copy.mesh,
        fixed: source.mesh,
        coarseMethod: "orientedBoundingBox",
        objective: "pointToPlane",
        maxIterations: 50,
        overlap: 1,
      });
      expect(coarse.iterations).toBe(0);
      expect(fine.finalRmse).toBeLessThanOrEqual(coarse.finalRmse + 1e-12);
      expect(fine.finalRmse).toBeLessThan(1e-3);
      expect(worstGap(scene, copy, source)).toBeLessThan(1e-2);
      store.undo();
      expect(scene.canUndo()).toBe(false);
      expect(scene.placement(copy.mesh).translation).toEqual([2.5, 1.5, -1]);
    }
  }, DEMO_TIMEOUT_MS);

  it("moves nothing and records no undo step when no point is within reach", async () => {
    const { scene, store, demos } = setup();
    const moving = demos[1];
    store.setPlacement(moving.mesh, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [50, 0, 0] });
    scene.clearHistory();
    const { coarse, fine } = await runMultistageAlign({
      store,
      moving: moving.mesh,
      fixed: demos[0].mesh,
      coarseMethod: "pcaCentroid",
      objective: "pointToPlane",
      maxIterations: 20,
      overlap: 1,
      maxDistance: 1e-9,
    });
    expect(coarse.finalRmse).toBe(Infinity);
    expect(multistageWarning(coarse, fine)).toContain("nothing moved");
    expect(scene.placement(moving.mesh).translation).toEqual([50, 0, 0]);
    expect(scene.canUndo()).toBe(false);
  }, DEMO_TIMEOUT_MS);

  it("finds the rounded box's symmetry plane through its centre and mirrors across it in one undo step", async () => {
    const { scene, store, demos } = setup();
    const box = demos[1];
    store.setPlacement(box.mesh, { rotation: rotZ(0.4), translation: [1, 2, 3] });
    scene.clearHistory();
    const r = await runDetectSymmetryPlane(store, box.mesh);
    expect(r.symmetryScore).toBeGreaterThan(0.99);
    // Local to the box, the plane passes its centre (the origin there) and is one of a
    // cube's mirror planes: across a face (an axis) or a diagonal (two axes at 45 degrees).
    const local = planeToLocal(scene.matrix4(box.mesh), r.planePoint, r.planeNormal);
    const offset = local.point[0] * local.normal[0] + local.point[1] * local.normal[1] + local.point[2] * local.normal[2];
    expect(Math.abs(offset)).toBeLessThan(1e-3 * box.radius);
    const n = local.normal.map(Math.abs).sort((a, b) => a - b);
    expect(n[0]).toBeCloseTo(0, 3);
    expect([0, Math.SQRT1_2].some((v) => Math.abs(n[1] - v) < 1e-3)).toBe(true);

    const before = Float64Array.from(scene.views(box.mesh).positions());
    await mirrorMesh(store, box.mesh, planeToWorld(scene.matrix4(box.mesh), local.point, local.normal));
    store.undo();
    expect(scene.canUndo()).toBe(false);
    expect(Float64Array.from(scene.views(box.mesh).positions())).toEqual(before);
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

/** Frames until the Worker has answered everything sent. */
async function settle(store: SceneStore): Promise<void> {
  for (let k = 0; k < 2000; k++) {
    store.commitFrame();
    await new Promise((resolve) => setTimeout(resolve, 2));
    if (store.getSnapshot().status !== "computing") {
      store.commitFrame();
      return;
    }
  }
  throw new Error("the store did not settle");
}

describe("align tools on the demo meshes (WorkerScene)", () => {
  it("runs landmarks, coarse + ICP and the symmetry plane off the main thread, one undo step each", async () => {
    const scene = await createWorkerScene({ createWorker: () => new NodeWorker() });
    const store = createSceneStore(scene);
    cleanups.push(() => {
      store.dispose();
      scene.terminate();
    });
    expect(store.mode).toBe("worker");
    const app = new AppState();
    const demos = addDemos(store, app);
    await settle(store);
    const copy = movedCopy(store, app, demos[1], rotZ(0.8), [2, 0, 1]);
    await settle(store);

    const set = pairsAt(scene, copy, demos[1], [5, 3000, 7000, 11000]);
    await runLandmarkAlign({
      store,
      moving: copy.mesh,
      fixed: demos[1].mesh,
      pairs: landmarkPairs(set, scene.matrix4(copy.mesh), scene.matrix4(demos[1].mesh)),
      allowScaling: false,
    });
    await settle(store);
    expect(worstGap(scene, copy, demos[1])).toBeLessThan(1e-6);
    store.undo();
    await settle(store);
    expect(scene.placement(copy.mesh).translation).toEqual([2, 0, 1]);

    const { fine } = await runMultistageAlign({
      store,
      moving: copy.mesh,
      fixed: demos[1].mesh,
      coarseMethod: "pcaCentroid",
      objective: "pointToPlane",
      maxIterations: 50,
      overlap: 1,
    });
    await settle(store);
    expect(fine.finalRmse).toBeLessThan(1e-3);
    expect(worstGap(scene, copy, demos[1])).toBeLessThan(1e-2);
    store.undo();
    await settle(store);
    expect(scene.placement(copy.mesh).translation).toEqual([2, 0, 1]);

    const plane = await runDetectSymmetryPlane(store, demos[1].mesh);
    expect(plane.symmetryScore).toBeGreaterThan(0.99);
  }, DEMO_TIMEOUT_MS);
});
