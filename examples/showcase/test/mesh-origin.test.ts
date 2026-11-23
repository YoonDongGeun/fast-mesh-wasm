// Meshes whose f64 origin is far from the world origin (#272): Duplicate and
// the other tools that rebuild a mesh from another's buffers keep its world
// shape, with any origin, rotation and scale, on the sync `Scene` and on a
// Worker.
import { Worker } from "node:worker_threads";
import { type MeshHandle, type Placement, Scene, type SceneLike } from "fast-mesh-wasm";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { createWorkerScene, type WorkerLike } from "fast-mesh-wasm/worker";
import { afterEach, describe, expect, it } from "vitest";
import { AppState, type Item } from "../src/app/state/app-state";
import {
  applySurfaceField,
  duplicateMesh,
  extractLiveMeshData,
  runBatchBoolean,
  runBooleanCsg,
  runPlaneCut,
} from "../src/engine/workbench-ops";
import { blobbySphere } from "../src/io/demo";

const cleanups: (() => void | Promise<void>)[] = [];
afterEach(async () => {
  while (cleanups.length > 0) {
    await cleanups.pop()?.();
  }
});

/** The SDK's Node worker-thread stand-in for the browser engine Worker. */
class NodeWorker implements WorkerLike {
  readonly worker = new Worker(new URL("../../../packages/fast-mesh-wasm/test/harness/node-worker.mjs", import.meta.url), {
    workerData: { serve: true },
  });

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", cb: (e: never) => void): void {
    this.worker.on(type, (data: unknown) => cb((type === "message" ? { data } : data) as never));
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

interface Harness {
  store: SceneStore;
  app: AppState;
  /** Commits frames until `work` (or what was asked so far) is done. */
  settle: <T>(work?: Promise<T>) => Promise<T>;
}

async function local(): Promise<Harness> {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(app.follow(store));
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return {
    store,
    app,
    settle: async (work) => {
      const value = await work;
      store.commitFrame();
      return value as never;
    },
  };
}

async function worker(): Promise<Harness> {
  const scene = await createWorkerScene({ createWorker: () => new NodeWorker() });
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(app.follow(store));
  cleanups.push(() => {
    store.dispose();
    scene.terminate();
  });
  return {
    store,
    app,
    settle: async (work) => {
      let done = false;
      const finished = Promise.resolve(work).finally(() => {
        done = true;
      });
      finished.catch(() => {});
      for (let k = 0; k < 2 || !done; k++) {
        const barrier = scene.raycast([0, 0, 1e6], [0, 0, 1]);
        store.commitFrame();
        await barrier;
        store.commitFrame();
      }
      return (await finished) as never;
    },
  };
}

type V3 = [number, number, number];

/** A box of half size `h` around `c` (world positions). */
function boxGeometry(h: number, c: Readonly<V3>) {
  const [cx, cy, cz] = c;
  const [x0, x1, y0, y1, z0, z1] = [cx - h, cx + h, cy - h, cy + h, cz - h, cz + h];
  return {
    positions: Float64Array.from([x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1]),
    indices: Uint32Array.from([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5]),
  };
}

/** A rotation about z, then about x. */
function rotation(z: number, x: number): number[] {
  const [cz, sz, cx, sx] = [Math.cos(z), Math.sin(z), Math.cos(x), Math.sin(x)];
  return [cz, -sz, 0, cx * sz, cx * cz, -sx, sx * sz, sx * cz, cx];
}

const ORIGIN: V3 = [125_000.5, -80_000.25, 4_000.125];
const PLACEMENT: Placement = {
  rotation: rotation(0.6, -0.35),
  translation: [-7, 42.5, 300],
  scale: [1.5, 0.5, 2],
};

/** A box listed with its centre as its origin, as the loader adds meshes. */
function addBox(h: Harness, name: string, half: number, centre: Readonly<V3>, placement?: Placement): Item {
  const item = h.app.addMesh(h.store, {
    name,
    ...boxGeometry(half, centre),
    center: [...centre],
    radius: half * Math.sqrt(3),
    volumeId: null,
    preset: null,
  });
  if (placement !== undefined) {
    h.store.setPlacement(item.mesh, placement);
  }
  return item;
}

/** The world bounds of the vertices that live triangles use. */
function worldBounds(scene: SceneLike, mesh: MeshHandle): { min: V3; max: V3 } {
  const views = scene.views(mesh);
  const positions = views.positions();
  const indices = views.indices();
  const m = scene.matrix4(mesh);
  const min: V3 = [Infinity, Infinity, Infinity];
  const max: V3 = [-Infinity, -Infinity, -Infinity];
  for (let k = 0; k + 2 < indices.length; k += 3) {
    if (indices[k] === 0 && indices[k + 1] === 0 && indices[k + 2] === 0) {
      continue;
    }
    for (let c = 0; c < 3; c++) {
      const v = 3 * indices[k + c];
      for (let a = 0; a < 3; a++) {
        const w = m[a] * positions[v] + m[4 + a] * positions[v + 1] + m[8 + a] * positions[v + 2] + m[12 + a];
        min[a] = Math.min(min[a], w);
        max[a] = Math.max(max[a], w);
      }
    }
  }
  return { min, max };
}

function expectBounds(got: { min: V3; max: V3 }, want: { min: V3; max: V3 }, digits = 4): void {
  for (let a = 0; a < 3; a++) {
    expect(got.min[a]).toBeCloseTo(want.min[a], digits);
    expect(got.max[a]).toBeCloseTo(want.max[a], digits);
  }
}

for (const [mode, open] of [
  ["sync Scene", local],
  ["WorkerScene", worker],
] as const) {
  describe(`a mesh with a far origin (${mode})`, () => {
    it.each([
      ["no placement", undefined],
      ["a rotated, scaled placement", PLACEMENT],
    ] as const)("is duplicated in place, with %s", async (_, placement) => {
      const h = await open();
      const { store, app } = h;
      const source = addBox(h, "Far", 2, ORIGIN, placement);
      await h.settle();
      const before = worldBounds(store.scene, source.mesh);
      const copy = duplicateMesh(store, app, source);
      await h.settle();
      expect(worldBounds(store.scene, copy.mesh)).toEqual(before);
      expect(store.scene.placement(copy.mesh)).toEqual(store.scene.placement(source.mesh));
      expect((store.scene as Scene).origin(copy.mesh)).toEqual(ORIGIN);
      expect(Array.from(store.scene.views(copy.mesh).positions())).toEqual(
        Array.from(store.scene.views(source.mesh).positions()),
      );
      expect(copy.center).toEqual(source.center);
    });

    it("is duplicated in place from coordinates that f64 rounds", async () => {
      const h = await open();
      const { store, app } = h;
      const ball = blobbySphere([...ORIGIN], 1, 24, 12, 0.2);
      const source = app.addMesh(store, { name: "Ball", ...ball, center: [...ORIGIN], radius: 1, volumeId: null, preset: null });
      store.setPlacement(source.mesh, PLACEMENT);
      await h.settle();
      const copy = duplicateMesh(store, app, source);
      await h.settle();
      expectBounds(worldBounds(store.scene, copy.mesh), worldBounds(store.scene, source.mesh), 8);
    });

    it("is copied in place when vertices it no longer uses have gone", async () => {
      const h = await open();
      const { store, app } = h;
      const source = addBox(h, "Far", 2, ORIGIN, PLACEMENT);
      await h.settle();
      await h.settle(store.edit(source.mesh, [{ op: "removeTriangles", triangles: [0, 1, 6, 7] }]));
      const copy = duplicateMesh(store, app, source);
      await h.settle();
      expect(worldBounds(store.scene, copy.mesh)).toEqual(worldBounds(store.scene, source.mesh));
      expect(extractLiveMeshData(store.scene, copy.mesh).indices.length).toBe(3 * 8);
    });

    it("takes part in a boolean where it is, kept or replaced", async () => {
      const h = await open();
      const { store, app } = h;
      // The same world boxes whatever their origins and placements.
      const a = addBox(h, "A", 2, ORIGIN, { ...PLACEMENT, scale: [1, 1, 1] });
      await h.settle();
      const pose = store.scene.placement(a.mesh);
      const b = addBox(h, "B", 1, [ORIGIN[0] + 2, ORIGIN[1], ORIGIN[2]], pose);
      await h.settle();
      const wantA = worldBounds(store.scene, a.mesh);
      const wantB = worldBounds(store.scene, b.mesh);
      const union = {
        min: wantA.min.map((v, k) => Math.min(v, wantB.min[k])) as V3,
        max: wantA.max.map((v, k) => Math.max(v, wantB.max[k])) as V3,
      };
      const kept = await h.settle(
        runBooleanCsg({ store, app, target: a, source: b, op: "union", keepOriginals: true }),
      );
      await h.settle();
      expectBounds(worldBounds(store.scene, kept.mesh), union);
      expectBounds(worldBounds(store.scene, a.mesh), wantA);
      const replaced = await h.settle(
        runBooleanCsg({ store, app, target: a, source: b, op: "union", keepOriginals: false }),
      );
      await h.settle();
      expect(replaced.mesh).toBe(a.mesh);
      expectBounds(worldBounds(store.scene, a.mesh), union);
      store.undo();
      await h.settle();
      expectBounds(worldBounds(store.scene, a.mesh), wantA);
      const batch = await h.settle(runBatchBoolean({ store, app, inputs: [a, b], op: "union", keepInputs: true }));
      await h.settle();
      // The engine's N-way boolean places vertices by f32 world coordinates.
      expectBounds(worldBounds(store.scene, batch.mesh), union, 1);
    });

    it("takes part in a boolean at its scale", async () => {
      const h = await open();
      const { store, app } = h;
      // World boxes [-2, 2] x [-1, 1]^2 and [1, 3] x [-1, 1]^2, the first by a scale.
      const identity = rotation(0, 0);
      const a = addBox(h, "A", 1, [0, 0, 0], { rotation: identity, translation: [0, 0, 0], scale: [2, 1, 1] });
      const b = addBox(h, "B", 0.5, [1, 0, 0], { rotation: identity, translation: [0, 0, 0], scale: [2, 2, 2] });
      await h.settle();
      const kept = await h.settle(
        runBooleanCsg({ store, app, target: a, source: b, op: "subtract", keepOriginals: true }),
      );
      await h.settle();
      expectBounds(worldBounds(store.scene, kept.mesh), { min: [-2, -1, -1], max: [1, 1, 1] });
    });

    it("is cut by a world plane where it is", async () => {
      const h = await open();
      const { store } = h;
      const a = addBox(h, "A", 2, [10, 20, 30], { rotation: rotation(0, 0), translation: [0, 0, 5], scale: [1, 1, 2] });
      await h.settle();
      expectBounds(worldBounds(store.scene, a.mesh), { min: [8, 18, 61], max: [12, 22, 69] });
      const cut = await h.settle(
        runPlaneCut({ store, mesh: a.mesh, axis: "x", offset: 11, cap: true, keepSide: "+" }),
      );
      await h.settle();
      expect(cut.kept).toBe(a.mesh);
      expectBounds(worldBounds(store.scene, a.mesh), { min: [11, 18, 61], max: [12, 22, 69] });
    });

    it("is measured for thickness at its scale", async () => {
      const h = await open();
      const { store, app } = h;
      const a = addBox(h, "A", 2, ORIGIN, { rotation: rotation(0, 0), translation: [0, 0, 0], scale: [0.5, 0.5, 0.5] });
      await h.settle();
      const state = { ...app.getSnapshot().surfaceField, mode: "thickness" as const, mesh: a.mesh, maxDistance: 100 };
      const r = await h.settle(applySurfaceField({ store, app, state }));
      // Across the box's corners, along their normals: the world box is 2 wide.
      expect(r.summary!.max).toBeLessThan(2 * Math.sqrt(3) + 0.01);
      expect(r.summary!.max).toBeGreaterThan(1);
    });
  });
}
