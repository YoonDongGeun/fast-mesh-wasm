// Undoable adds and removals in the showcase (#269): the tools that add or
// replace meshes, and the app state following undo and redo (the list, the
// selection, the collision pair, the groups), on the sync `Scene` and on a
// Worker. The mesh list's Remove button: undo-remove-panel.test.tsx.
import { Worker } from "node:worker_threads";
import { type MeshHandle, Scene } from "fast-mesh-wasm";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { createWorkerScene, type WorkerLike } from "fast-mesh-wasm/worker";
import { afterEach, describe, expect, it } from "vitest";
import { AppState, type Item } from "../src/app/state/app-state";
import { duplicateMesh, runBooleanCsg, runConvexHull, runPlaneCut } from "../src/engine/workbench-ops";
import { restoreMember } from "../src/features/collision/ux/groups";
import { collectProject, restoreProject } from "../src/io/project";
import { selectMesh } from "../src/features/select/ux/selection";

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
  settle: () => Promise<void>;
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
    settle: async () => {
      store.commitFrame();
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
    settle: async () => {
      for (let k = 0; k < 2; k++) {
        const barrier = scene.raycast([0, 0, 1e6], [0, 0, 1]);
        store.commitFrame();
        await barrier;
        store.commitFrame();
      }
    },
  };
}

function boxGeometry(h: number, c: readonly [number, number, number]) {
  const [cx, cy, cz] = c;
  const [x0, x1, y0, y1, z0, z1] = [cx - h, cx + h, cy - h, cy + h, cz - h, cz + h];
  return {
    positions: Float64Array.from([x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1]),
    indices: Uint32Array.from([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5]),
  };
}

function addBox(h: Harness, name: string, center: readonly [number, number, number] = [0, 0, 0]): Item {
  return h.app.addMesh(h.store, {
    name,
    ...boxGeometry(1, center),
    center: [...center],
    radius: Math.sqrt(3),
    volumeId: null,
    preset: null,
  });
}

function liveTriangles(store: SceneStore, mesh: MeshHandle): number {
  const idx = store.scene.views(mesh).indices();
  let n = 0;
  for (let k = 0; k + 2 < idx.length; k += 3) {
    if (idx[k] !== 0 || idx[k + 1] !== 0 || idx[k + 2] !== 0) {
      n++;
    }
  }
  return n;
}

const names = (app: AppState) => app.getSnapshot().items.map((i) => i.name);
const select = (h: Harness, mesh: MeshHandle | null, additive = false) =>
  selectMesh(h.app, (m) => h.store.select(m), h.store.getSnapshot().selection, mesh, additive);

describe("the app state", () => {
  it("restores several removals in their order, whatever was selected since", async () => {
    const h = await local();
    const { store, app } = h;
    const [a, b, c, d] = ["A", "B", "C", "D"].map((n, k) => addBox(h, n, [3 * k, 0, 0]));
    app.removeMesh(store, b);
    app.removeMesh(store, d);
    app.removeMesh(store, a);
    expect(names(app)).toEqual(["C"]);
    select(h, c.mesh);
    store.undo();
    expect(names(app)).toEqual(["A", "C"]);
    expect(store.getSnapshot().selection).toBe(c.mesh);
    store.undo();
    store.undo();
    expect(names(app)).toEqual(["A", "B", "C", "D"]);
    // Removed in one step: one undo.
    store.beginStep();
    for (const i of [a, b, d]) {
      app.removeMesh(store, i);
    }
    store.endStep();
    expect(names(app)).toEqual(["C"]);
    store.undo();
    expect(names(app)).toEqual(["A", "B", "C", "D"]);
  });

  it("puts a restored mesh back into the collision pair, the contact group and its named groups", async () => {
    const h = await local();
    const { store, app } = h;
    const [a, b, c] = ["A", "B", "C"].map((n, k) => addBox(h, n, [1.5 * k, 0, 0]));
    // The first meshes loaded are the default pair and group.
    expect(app.getSnapshot().collision).toEqual({ a: a.mesh, b: b.mesh });
    expect(app.getSnapshot().group.members).toEqual([a.mesh, b.mesh, c.mesh]);
    app.setGroupPairs([
      [a.mesh, b.mesh],
      [b.mesh, c.mesh],
    ]);
    app.toggleCheck(a.mesh, true);
    app.toggleCheck(b.mesh, true);
    app.createGroup("Pair");
    const before = app.getSnapshot();
    expect(before.meshGroups[0].members).toEqual([a.mesh, b.mesh]);
    app.removeMesh(store, b);
    expect(app.getSnapshot().collision).toEqual({ a: a.mesh, b: null });
    expect(app.getSnapshot().group).toEqual({ members: [a.mesh, c.mesh], pairs: [] });
    expect(app.getSnapshot().meshGroups[0].members).toEqual([a.mesh]);
    store.undo();
    expect(app.getSnapshot().collision).toEqual(before.collision);
    expect(new Set(app.getSnapshot().group.members)).toEqual(new Set(before.group.members));
    expect(app.getSnapshot().group.pairs).toEqual(before.group.pairs);
    expect(app.getSnapshot().meshGroups[0].members).toEqual([a.mesh, b.mesh]);
    // A side picked meanwhile keeps its pick.
    store.redo();
    app.pick("b", c.mesh);
    store.undo();
    expect(app.getSnapshot().collision).toEqual({ a: a.mesh, b: c.mesh });
  });

  it("in pair mode, meshes that leave and return together keep their place, pair and selection", async () => {
    const h = await local();
    const { store, app } = h;
    const [a, b, c] = ["A", "B", "C"].map((n, k) => addBox(h, n, [1.5 * k, 0, 0]));
    app.setCollisionMode("pair");
    store.setDisplay(a.mesh, { coloring: { distanceTo: b.mesh, maxDistance: 5, range: [0, 1], map: "turbo" } });
    store.setDisplay(b.mesh, { coloring: { distanceTo: a.mesh, maxDistance: 5, range: [0, 1], map: "turbo" } });
    await h.settle();
    const { collision } = app.getSnapshot();
    expect(collision).toEqual({ a: a.mesh, b: b.mesh });
    select(h, b.mesh);
    // Undone adds, not removals: the app only learns of them from the store.
    store.beginStep();
    app.removeMesh(store, a);
    app.removeMesh(store, b);
    store.endStep();
    for (let k = 0; k < 2; k++) {
      expect(app.getSnapshot().items).toEqual([c]);
      expect(app.getSnapshot().collision).toEqual({ a: null, b: null });
      store.undo();
      expect(app.getSnapshot().items).toEqual([a, b, c]);
      expect(app.getSnapshot().collision).toEqual(collision);
      expect(store.getSnapshot().selection).toBe(b.mesh);
      expect(app.getSnapshot().multi).toEqual([b.mesh]);
      store.redo();
    }
    // One of the pair only: the other's colouring stops, and the item keeps its place.
    store.undo();
    store.setDisplay(a.mesh, { coloring: { distanceTo: b.mesh, maxDistance: 5, range: [0, 1], map: "turbo" } });
    await h.settle();
    app.removeMesh(store, b);
    expect(store.getSnapshot().meshes.get(a.mesh)!.desired.coloring).toBeNull();
    store.undo();
    expect(app.getSnapshot().items).toEqual([a, b, c]);
    expect(app.getSnapshot().collision).toEqual(collision);
    store.redo();
    expect(app.getSnapshot().items).toEqual([a, c]);
    store.undo();
    expect(app.getSnapshot().items).toEqual([a, b, c]);
  });

  it("keeps a removed mesh's volume while undo can bring it back", async () => {
    const h = await local();
    const { store, app } = h;
    const item = app.addMesh(store, {
      name: "Iso",
      ...boxGeometry(1, [0, 0, 0]),
      center: [0, 0, 0],
      radius: 1,
      volumeId: "v1",
      preset: null,
    });
    app.removeMesh(store, item);
    expect(app.parkedVolumes()).toEqual(["v1"]);
    store.undo();
    expect(app.parkedVolumes()).toEqual([]);
    app.removeMesh(store, item, { undoable: false });
    expect(app.parkedVolumes()).toEqual([]);
    expect(store.getSnapshot().history.canUndo).toBe(false);
  });

  it("restores members only to groups that are still there", () => {
    const groups = [
      { id: 1, name: "One", members: ["a"], colored: false },
      { id: 3, name: "Three", members: ["a", "m"], colored: true },
    ];
    expect(restoreMember(groups, [1, 2, 3], "m")).toEqual([
      { id: 1, name: "One", members: ["a", "m"], colored: false },
      { id: 3, name: "Three", members: ["a", "m"], colored: true },
    ]);
  });
});

describe("projects", () => {
  it("opening one is not undone, and nothing from before comes back", async () => {
    const h = await local();
    const { store, app } = h;
    const [a] = [addBox(h, "A"), addBox(h, "B", [3, 0, 0])];
    const doc = collectProject(store, app);
    app.removeMesh(store, a);
    expect(store.getSnapshot().history.canUndo).toBe(true);
    restoreProject(store, app, doc);
    expect(names(app)).toEqual(["A", "B"]);
    for (let k = 0; k < 10 && store.getSnapshot().history.canUndo; k++) {
      store.undo();
      expect(names(app)).toEqual(["A", "B"]);
    }
  });
});

for (const [mode, open] of [
  ["sync Scene", local],
  ["WorkerScene", worker],
] as const) {
  describe(`tools that add or replace meshes (${mode})`, () => {
    it("the old pattern still works: an empty mesh, listed, then filled as one edit", async () => {
      const h = await open();
      const { store, app } = h;
      const g = boxGeometry(1, [0, 0, 0]);
      const mesh = store.addMesh(new Float64Array(0), new Uint32Array(0), { origin: [0, 0, 0] });
      const item = app.listMesh(mesh, { name: "Filled", triangles: 12, center: [0, 0, 0], radius: 2 });
      const edit = store.edit(mesh, [
        { op: "addVertices", positions: g.positions },
        { op: "addTriangles", triangles: g.indices },
      ]);
      await h.settle();
      await edit;
      expect(liveTriangles(store, mesh)).toBe(12);
      store.undo();
      await h.settle();
      // Undo empties it; its row stays.
      expect(liveTriangles(store, mesh)).toBe(0);
      expect(app.getSnapshot().items).toEqual([item]);
      store.redo();
      await h.settle();
      expect(liveTriangles(store, mesh)).toBe(12);
      // And its list entry is removed and restored like any other.
      app.removeMesh(store, item);
      await h.settle();
      expect(app.getSnapshot().items).toEqual([]);
      store.undo();
      await h.settle();
      expect(app.getSnapshot().items).toEqual([item]);
      expect(liveTriangles(store, mesh)).toBe(12);
    });

    it("a boolean that keeps its inputs, and one that replaces them, are one undo step each", async () => {
      const h = await open();
      const { store, app } = h;
      const a = addBox(h, "A");
      const b = addBox(h, "B", [1, 0, 0]);
      await h.settle();

      const kept = runBooleanCsg({ store, app, target: a, source: b, op: "union", keepOriginals: true });
      await h.settle();
      const r = await kept;
      await h.settle();
      expect(names(app)).toEqual(["A", "B", "A union B"]);
      expect(liveTriangles(store, r.mesh)).toBe(r.triangles);
      store.undo();
      await h.settle();
      expect(app.getSnapshot().items).toEqual([a, b]);
      expect(store.getSnapshot().history.canUndo).toBe(false);
      store.redo();
      await h.settle();
      expect(names(app)).toEqual(["A", "B", "A union B"]);
      expect(liveTriangles(store, r.mesh)).toBe(r.triangles);
      store.undo();
      await h.settle();

      const replaced = runBooleanCsg({ store, app, target: a, source: b, op: "subtract", keepOriginals: false });
      for (let k = 0; k < 4; k++) {
        await h.settle();
      }
      const cut = await replaced;
      await h.settle();
      expect(app.getSnapshot().items).toEqual([a]);
      expect(liveTriangles(store, a.mesh)).toBe(cut.triangles);
      store.undo();
      await h.settle();
      expect(app.getSnapshot().items).toEqual([a, b]);
      expect(liveTriangles(store, a.mesh)).toBe(12);
      expect(liveTriangles(store, b.mesh)).toBe(12);
      expect(store.getSnapshot().history.canUndo).toBe(false);
      store.redo();
      await h.settle();
      expect(app.getSnapshot().items).toEqual([a]);
      expect(liveTriangles(store, a.mesh)).toBe(cut.triangles);
    });

    it("a hull and a duplicate undo to nothing, a plane cut to the whole mesh", async () => {
      const h = await open();
      const { store, app } = h;
      const a = addBox(h, "A");
      await h.settle();

      const hull = runConvexHull({ store, app, name: "A hull", sources: [a] });
      await h.settle();
      const made = (await hull).item;
      await h.settle();
      store.undo();
      await h.settle();
      expect(app.getSnapshot().items).toEqual([a]);
      store.redo();
      await h.settle();
      expect(app.getSnapshot().items).toEqual([a, made]);
      store.undo();
      await h.settle();

      const copy = duplicateMesh(store, app, a);
      await h.settle();
      expect(names(app)).toEqual(["A", "A copy"]);
      store.undo();
      await h.settle();
      expect(app.getSnapshot().items).toEqual([a]);
      expect(store.getSnapshot().history.canUndo).toBe(false);
      store.redo();
      await h.settle();
      expect(app.getSnapshot().items).toEqual([a, copy]);
      expect(liveTriangles(store, copy.mesh)).toBe(12);
      store.undo();
      await h.settle();

      const cut = runPlaneCut({ store, mesh: a.mesh, axis: "x", offset: 0, cap: true, keepSide: "+" });
      for (let k = 0; k < 4; k++) {
        await h.settle();
      }
      await cut;
      await h.settle();
      const count = liveTriangles(store, a.mesh);
      expect(count).not.toBe(12);
      store.undo();
      await h.settle();
      expect(liveTriangles(store, a.mesh)).toBe(12);
      expect(store.getSnapshot().history.canUndo).toBe(false);
      store.redo();
      await h.settle();
      expect(liveTriangles(store, a.mesh)).toBe(count);
      // Only the kept side is there.
      expect(app.getSnapshot().items).toEqual([a]);
      expect(store.getSnapshot().meshes.size).toBe(1);
    });

    it("a boolean that fails leaves the list and the history as they were", async () => {
      const h = await open();
      const { store, app } = h;
      const a = addBox(h, "A");
      const open1 = app.addMesh(store, {
        name: "Open",
        positions: Float64Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        indices: Uint32Array.from([0, 1, 2]),
        center: [0, 0, 0],
        radius: 1,
        volumeId: null,
        preset: null,
      });
      await h.settle();
      const failed = runBooleanCsg({ store, app, target: a, source: open1, op: "union", keepOriginals: true });
      failed.catch(() => {});
      await h.settle();
      await expect(failed).rejects.toThrow();
      await h.settle();
      expect(app.getSnapshot().items).toEqual([a, open1]);
      expect(store.getSnapshot().meshes.size).toBe(2);
      expect(store.getSnapshot().history).toEqual({ canUndo: false, canRedo: false });
    });
  });
}
