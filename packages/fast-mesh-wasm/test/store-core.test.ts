// The `SceneStore` core (`fast-mesh-wasm/store`) over both hosts: the
// synchronous `Scene` (Node build) and `WorkerScene` against the Worker
// engine in the Node `worker_threads` harness (built dist/, `serve` mode),
// through the `createWorker` option. Snapshots, structural sharing, history,
// selection, removed meshes, the flush schedule, subscriptions and the
// package layout of the subpath.
import { existsSync, readFileSync } from "node:fs";
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FastMeshError, Scene } from "../src/node.js";
import {
  createSceneStore,
  EMPTY_SNAPSHOT,
  meshState,
  REMOVED_MESH_STATE,
  type SceneStore,
} from "../src/store/index.js";
import { listenerCount } from "../src/store/store.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/** A closed tetrahedron offset by `dx` along x. */
function tetra(dx = 0) {
  const positions = Float64Array.from([dx, 0, 0, dx + 1, 0, 0, dx, 1, 0, dx, 0, 1]);
  const indices = Uint32Array.from([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]);
  return { positions, indices };
}

/** The harness Worker as a `WorkerLike`. */
class NodeWorker implements WorkerLike {
  readonly worker: Worker;
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  readonly #error: ((e: unknown) => void)[] = [];

  constructor() {
    this.worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), {
      workerData: { serve: true },
    });
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

  addEventListener(type: "message" | "error", cb: (e: any) => void): void {
    (type === "message" ? this.#message : this.#error).push(cb);
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

const cleanup: (() => void)[] = [];
afterEach(() => {
  for (const f of cleanup.splice(0)) {
    f();
  }
});

async function makeLocal(): Promise<Scene> {
  const scene = new Scene();
  cleanup.push(() => scene.free());
  return scene;
}

async function makeWorker(): Promise<WorkerScene> {
  const scene = await createWorkerScene({ createWorker: () => new NodeWorker() });
  cleanup.push(() => scene.terminate());
  return scene;
}

/**
 * Commits frames until the store's keys settle: no mesh, section or history
 * status reads `"computing"` (results applied, statuses delivered), then one
 * more frame delivers the change records.
 */
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

function withTwo(store: SceneStore) {
  const a = store.addMesh(tetra(0).positions, tetra(0).indices);
  const b = store.addMesh(tetra(3).positions, tetra(3).indices);
  return { a, b };
}

describe.each([
  ["local", makeLocal],
  ["worker", makeWorker],
] as const)("SceneStore (%s)", (mode, make) => {
  async function start(): Promise<SceneStore> {
    const store = createSceneStore(await make());
    cleanup.unshift(() => store.dispose());
    expect(store.mode).toBe(mode);
    return store;
  }

  it("snapshot_is_immutable_and_shared", async () => {
    const store = await start();
    const { a, b } = withTwo(store);
    await settle(store);
    const before = store.getSnapshot();
    store.setDisplay(a, { opacity: 0.5 });
    await settle(store);
    const snap = store.getSnapshot();
    expect(snap).not.toBe(before);
    expect(snap.revision).toBeGreaterThan(before.revision);
    expect(snap.meshes.get(b)).toBe(before.meshes.get(b));
    expect(Object.isFrozen(snap)).toBe(true);
    expect(Object.isFrozen(snap.meshes.get(a)!)).toBe(true);
    expect(Object.isFrozen(snap.meshes.get(a)!.display)).toBe(true);
    expect(Object.isFrozen(snap.meshes.get(a)!.desired)).toBe(true);
    expect(snap.meshes.get(a)!.display.opacity).toBe(0.5);
    expect(snap.meshes.get(a)!.desired.opacity).toBe(0.5);
    expect(snap.meshes.get(a)!.status).toBe("ready");
  });

  it("no_change_no_new_snapshot", async () => {
    const store = await start();
    const { a } = withTwo(store);
    await settle(store);
    const before = store.getSnapshot();
    const listener = vi.fn();
    store.subscribe(listener);
    store.setDisplay(a, { opacity: 1 });
    store.setDisplay(a, { visible: true, coloring: null });
    await Promise.resolve();
    store.commitFrame();
    await new Promise((resolve) => setTimeout(resolve, 5));
    store.commitFrame();
    expect(store.getSnapshot()).toBe(before);
    expect(listener).not.toHaveBeenCalled();
  });

  it("history_follows_actions", async () => {
    const store = await start();
    const { a } = withTwo(store);
    await settle(store);
    expect(store.getSnapshot().history).toEqual({ canUndo: false, canRedo: false });
    store.setPlacement(a, { rotation: IDENTITY, translation: [1, 0, 0] });
    await settle(store);
    expect(store.getSnapshot().history).toEqual({ canUndo: true, canRedo: false });
    store.undo();
    await settle(store);
    expect(store.getSnapshot().history).toEqual({ canUndo: false, canRedo: true });
    store.redo();
    await settle(store);
    const history = store.getSnapshot().history;
    expect(history).toEqual({ canUndo: true, canRedo: false });

    // Display state never enters history.
    store.setDisplay(a, { visible: false, opacity: 0.25 });
    await settle(store);
    expect(store.getSnapshot().meshes.get(a)!.display.visible).toBe(false);
    expect(store.getSnapshot().history).toBe(history);

    // A drag through the store records one step.
    store.undo();
    await settle(store);
    expect(store.getSnapshot().history).toEqual({ canUndo: false, canRedo: true });
    const drag = store.drag(a);
    drag.move({ rotation: IDENTITY, translation: [0, 2, 0] });
    expect(drag.active).toBe(true);
    drag.end();
    expect(drag.active).toBe(false);
    await settle(store);
    expect(store.getSnapshot().history).toEqual({ canUndo: true, canRedo: false });
  });

  it("selection_and_removed_mesh", async () => {
    const store = await start();
    const { a, b } = withTwo(store);
    await settle(store);
    store.select(a);
    expect(store.getSnapshot().selection).toBe(a);
    store.removeMesh(a);
    await settle(store);
    const snap = store.getSnapshot();
    expect(snap.selection).toBeNull();
    expect(snap.meshes.has(a)).toBe(false);
    expect(snap.meshes.has(b)).toBe(true);
    for (const call of [() => store.setDisplay(a, { opacity: 0.2 }), () => store.select(a)]) {
      let error: unknown;
      try {
        call();
      } catch (e) {
        error = e;
      }
      expect(error).toBeInstanceOf(FastMeshError);
      expect((error as FastMeshError).code).toBe("UNKNOWN_MESH");
    }
    // Selecting another mesh, then removing one not selected, keeps it.
    store.select(b);
    const c = store.addMesh(tetra(6).positions, tetra(6).indices);
    store.removeMesh(c);
    await settle(store);
    expect(store.getSnapshot().selection).toBe(b);
    store.select(null);
    expect(store.getSnapshot().selection).toBeNull();
  });

  it("removed_mesh_state", async () => {
    const store = await start();
    const { a } = withTwo(store);
    await settle(store);
    store.setDisplay(a, { opacity: 0.3 });
    store.removeMesh(a);
    await settle(store);
    const snap = store.getSnapshot();
    expect(snap.meshes.get(a)).toBeUndefined();
    const state = meshState(snap, a);
    expect(state).toBe(REMOVED_MESH_STATE);
    expect(Object.isFrozen(state)).toBe(true);
    expect(Object.isFrozen(state.display)).toBe(true);
    expect(Object.isFrozen(state.desired)).toBe(true);
    expect(state.status).toBe("idle");
    expect(state.display).toEqual({ coloring: null, opacity: 1, visible: true });
    expect(state.desired).toEqual({ coloring: null, opacity: 1, visible: true });
    expect(meshState(EMPTY_SNAPSHOT, a)).toBe(REMOVED_MESH_STATE);
    expect(Object.isFrozen(EMPTY_SNAPSHOT)).toBe(true);
    expect(EMPTY_SNAPSHOT.meshes.size).toBe(0);
    expect(EMPTY_SNAPSHOT.status).toBe("idle");
  });

  it("flush_on_microtask_without_frames", async () => {
    const store = await start();
    // No frame yet: declarative changes flush on a microtask.
    const { a } = withTwo(store);
    store.setDisplay(a, { visible: false });
    expect(store.getSnapshot().meshes.get(a)!.desired.visible).toBe(false);
    expect(store.getSnapshot().meshes.get(a)!.display.visible).toBe(true);
    await Promise.resolve();
    expect(store.getSnapshot().meshes.get(a)!.display.visible).toBe(false);

    // With a frame loop, they apply at the next commitFrame(), last value wins.
    store.commitFrame();
    store.setDisplay(a, { visible: true, opacity: 0.1 });
    store.setDisplay(a, { opacity: 0.7 });
    expect(store.getSnapshot().meshes.get(a)!.desired).toEqual({ coloring: null, opacity: 0.7, visible: true });
    await Promise.resolve();
    await new Promise((resolve) => setTimeout(resolve, 5));
    expect(store.getSnapshot().meshes.get(a)!.display.visible).toBe(false);
    expect(store.getSnapshot().meshes.get(a)!.display.opacity).toBe(1);
    store.commitFrame();
    expect(store.getSnapshot().meshes.get(a)!.display).toEqual({ coloring: null, opacity: 0.7, visible: true });
  });

  it("set_display_validates_and_stores_coloring", async () => {
    const store = await start();
    const { a, b } = withTwo(store);
    for (const bad of [{ opacity: 1.5 }, { opacity: Number.NaN }, { visible: 1 as unknown as boolean }]) {
      expect(() => store.setDisplay(a, bad)).toThrow(FastMeshError);
    }
    // Colouring is stored as desired here; applying it comes with Task 2.
    const rule = { distanceTo: b, maxDistance: 2, map: "turbo" as const, range: [0, 1] as const };
    store.setDisplay(a, { coloring: rule });
    const desired = store.getSnapshot().meshes.get(a)!.desired;
    expect(desired.coloring).toEqual(rule);
    expect(desired.coloring).not.toBe(rule);
    expect(Object.isFrozen(desired.coloring)).toBe(true);
    // An equal (not identical) rule changes nothing.
    const snap = store.getSnapshot();
    store.setDisplay(a, { coloring: { ...rule, range: [0, 1] } });
    expect(store.getSnapshot()).toBe(snap);
  });

  it("subscribe_unsubscribe", async () => {
    const scene = await make();
    const removed: string[] = [];
    const onChanges = scene.onChanges.bind(scene);
    vi.spyOn(scene, "onChanges").mockImplementation((l) => {
      const off = onChanges(l);
      return () => {
        removed.push("changes");
        off();
      };
    });
    if (mode === "worker") {
      const ws = scene as WorkerScene;
      const onStatus = ws.onStatus.bind(ws);
      vi.spyOn(ws, "onStatus").mockImplementation((l) => {
        const off = onStatus(l);
        return () => {
          removed.push("status");
          off();
        };
      });
    }
    const store = createSceneStore(scene);
    const listener = vi.fn();
    const off = store.subscribe(listener);
    const off2 = store.subscribe(() => {});
    expect(listenerCount(store)).toBe(2);
    const a = store.addMesh(tetra().positions, tetra().indices);
    expect(listener).toHaveBeenCalled();
    off();
    off();
    off2();
    expect(listenerCount(store)).toBe(0);
    listener.mockClear();
    store.select(a);
    expect(listener).not.toHaveBeenCalled();

    store.dispose();
    store.dispose();
    expect(removed.sort()).toEqual(mode === "worker" ? ["changes", "status"] : ["changes"]);
    expect(() => store.select(null)).toThrow(FastMeshError);
  });

  it("status_follows_keys", async () => {
    const store = await start();
    const { a, b } = withTwo(store);
    // Local calls apply at once; Worker ones are pending until their results.
    const pending = mode === "local" ? "ready" : "computing";
    expect(store.getSnapshot().meshes.get(a)!.status).toBe(pending);
    expect(store.getSnapshot().status).toBe(mode === "local" ? "idle" : "computing");
    await settle(store);
    expect(store.getSnapshot().meshes.get(a)!.status).toBe("ready");
    expect(store.getSnapshot().status).toBe("idle");
    const stateB = store.getSnapshot().meshes.get(b);
    store.setPlacement(a, { rotation: IDENTITY, translation: [0, 0, 1] });
    expect(store.getSnapshot().meshes.get(a)!.status).toBe(pending);
    await settle(store);
    expect(store.getSnapshot().meshes.get(a)!.status).toBe("ready");
    expect(store.getSnapshot().meshes.get(b)).toBe(stateB);
  });

  it("restart_by_host", async () => {
    const store = await start();
    withTwo(store);
    if (mode === "local") {
      await expect(store.restart()).rejects.toMatchObject({
        code: "INVALID_ARGUMENT",
        message: "restart needs a Worker scene",
      });
    } else {
      await settle(store);
      await store.restart();
      await settle(store);
      expect(store.getSnapshot().status).toBe("idle");
      expect(store.getSnapshot().meshes.size).toBe(2);
    }
  });

  it("edit_and_repair_return_promises", async () => {
    const store = await start();
    const { a } = withTwo(store);
    await settle(store);
    const edit = store.edit(a, []);
    expect(edit).toBeInstanceOf(Promise);
    const repair = store.repair(a, { op: "orient" });
    expect(repair).toBeInstanceOf(Promise);
    store.commitFrame();
    await edit;
    await repair;
    const bad = store.edit(a, [{ op: "nope" } as never]);
    expect(bad).toBeInstanceOf(Promise);
    store.commitFrame();
    await expect(bad).rejects.toBeInstanceOf(FastMeshError);
  });
});

describe("fast-mesh-wasm/store package", () => {
  it("package_store_export", () => {
    const dist = new URL("../dist/", import.meta.url);
    expect(existsSync(new URL("store.js", dist))).toBe(true);
    expect(existsSync(new URL("store.d.ts", dist))).toBe(true);
    // Test-only helpers stay out of the public types.
    expect(readFileSync(new URL("store.d.ts", dist), "utf8")).not.toMatch(/listenerCount/);
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
    expect(pkg.exports["./store"]).toEqual({ types: "./dist/store.d.ts", default: "./dist/store.js" });

    /** Every dist file a file imports, transitively. */
    const graph = (file: string, seen = new Set<string>()): Set<string> => {
      if (seen.has(file)) {
        return seen;
      }
      seen.add(file);
      const src = readFileSync(new URL(file, dist), "utf8");
      for (const m of src.matchAll(/(?:import|export)\s[^"']*?from\s*["']([^"']+)["']|import\s*["']([^"']+)["']/g)) {
        const spec = m[1] ?? m[2];
        if (spec.startsWith("./")) {
          graph(new URL(spec, new URL(file, dist)).href.slice(dist.href.length), seen);
        } else {
          seen.add(spec);
        }
      }
      return seen;
    };
    const files = [...graph("store.js")];
    expect(files.filter((f) => /react|three|fast_mesh_wasm\.js|^wasm\//.test(f))).toEqual([]);
    // Nor does any of them import a module dynamically.
    for (const f of files.filter((f) => f.endsWith(".js"))) {
      expect(readFileSync(new URL(f, dist), "utf8"), f).not.toMatch(/\bimport\(/);
    }
  });
});
