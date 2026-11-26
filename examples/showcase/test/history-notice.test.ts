// Saying that an action is permanent (#281): removals, joins and undone adds
// larger than the undo memory, on the sync `Scene` and on a Worker, with a
// small undo memory instead of huge meshes. The status bar: history-notice-bar.test.tsx.
import { Worker } from "node:worker_threads";
import { type MeshHandle, Scene } from "fast-mesh-wasm";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "fast-mesh-wasm/worker";
import { afterEach, describe, expect, it } from "vitest";
import { AppState, type Item } from "../src/app/state/app-state";
import { describePermanent, historyBudgetParam } from "../src/app/ux/history-notice";
import { duplicateMesh } from "../src/engine/workbench-ops";
import { joinMeshes } from "../src/features/mesh-repair/ux/join";
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
  scene: Scene | WorkerScene;
  store: SceneStore;
  app: AppState;
  /** Frames until every result is in; `p` settles meanwhile. */
  settle: <T>(p?: Promise<T>) => Promise<T | undefined>;
}

function harness(scene: Scene | WorkerScene, free: () => void): Harness {
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(app.follow(store));
  cleanups.push(() => {
    store.dispose();
    free();
  });
  return {
    scene,
    store,
    app,
    settle: async (p) => {
      for (let k = 0; k < 2; k++) {
        const barrier = scene.raycast([0, 0, 1e6], [0, 0, 1]);
        store.commitFrame();
        await barrier;
        store.commitFrame();
      }
      const out = await p;
      store.commitFrame();
      return out;
    },
  };
}

const local = async (historyBudget?: number): Promise<Harness> => {
  const scene = new Scene({ historyBudget });
  return harness(scene, () => scene.free());
};

const worker = async (historyBudget?: number): Promise<Harness> => {
  const scene = await createWorkerScene({ createWorker: () => new NodeWorker(), historyBudget });
  return harness(scene, () => scene.terminate());
};

function addBall(h: Harness, name: string, x = 0): Item {
  const d = blobbySphere([x, 0, 0], 1, 48, 32, 0);
  return h.app.addMesh(h.store, { name, ...d, center: [x, 0, 0], radius: 1, volumeId: null, preset: null });
}

const notice = (h: Harness) => h.app.getSnapshot().historyNotice?.text ?? null;
const items = (h: Harness) => h.app.getSnapshot().items.map((i) => i.name);

describe("the words", () => {
  const [a, b, c, d, scratch] = [0, 1, 2, 3, 4].map(() => ({}) as MeshHandle);
  const names = new Map([
    [a, "A"],
    [b, "B"],
    [c, "C"],
    [d, "D"],
  ]);
  const say = (lost: "undo" | "redo", meshes: MeshHandle[], gone: MeshHandle[]) =>
    describePermanent({ lost, meshes, gone, bytes: 300 * 2 ** 20 }, (m) => names.get(m), 256 * 2 ** 20);
  const why = ": it takes 300 MB of undo memory, over the limit of 256 MB.";

  it("name what is gone and what changed", () => {
    expect(say("undo", [a], [a])).toBe(`Removing "A" cannot be undone${why}`);
    expect(say("undo", [a, b], [a, b])).toBe(`Removing "A" and "B" cannot be undone${why}`);
    expect(say("undo", [a, b, c, d], [a, b, c, d])).toBe(`Removing "A", "B" and 2 more cannot be undone${why}`);
    expect(say("undo", [a, b], [b])).toBe(`Changing "A" and removing "B" cannot be undone${why}`);
    expect(say("undo", [a], [])).toBe(`The last change to "A" cannot be undone${why}`);
    expect(say("redo", [a], [a])).toBe(`Redo cannot bring back "A"${why}`);
    expect(say("redo", [a], [])).toBe(`The undone change cannot be redone${why}`);
    // A mesh never listed has no name.
    expect(say("undo", [scratch], [scratch])).toBe(`The last change cannot be undone${why}`);
    expect(say("undo", [], [])).toBe(`The last change cannot be undone${why}`);
  });

  it("read the URL's undo memory", () => {
    const of = (q: string) => historyBudgetParam(new URLSearchParams(q));
    expect(of("")).toBeUndefined();
    expect(of("history=64")).toBe(64 * 2 ** 20);
    expect(of("history=0.5")).toBe(2 ** 19);
    expect(of("history=0")).toBe(0);
    expect(["history=", "history=x", "history=-1"].map(of)).toEqual([undefined, undefined, undefined]);
  });
});

describe.each([
  ["local", local],
  ["worker", worker],
] as const)("the app (%s)", (_mode, make) => {
  it("says that a removal too large to undo is permanent, and nothing of one that fits", async () => {
    const h = await make(1_000_000);
    const [a, b] = [addBall(h, "Small"), addBall(h, "Big", 5)];
    await h.settle();
    h.app.removeMesh(h.store, a);
    await h.settle();
    expect(notice(h)).toBeNull();
    h.store.undo();
    await h.settle();
    expect(items(h)).toEqual(["Small", "Big"]);

    h.store.setHistoryBudget(10_000);
    h.app.removeMesh(h.store, b);
    await h.settle();
    expect(notice(h)).toMatch(/^Removing "Big" cannot be undone: it takes 1 MB of undo memory/);
    expect(h.store.isRestorable(b.mesh)).toBe(false);
    expect(h.store.getSnapshot().history.canUndo).toBe(false);
    expect(items(h)).toEqual(["Small"]);
    h.store.undo();
    await h.settle();
    expect(items(h)).toEqual(["Small"]);

    // Dismissed, and said again for the next.
    h.app.setHistoryNotice(null);
    expect(notice(h)).toBeNull();
    h.app.removeMesh(h.store, a);
    await h.settle();
    expect(notice(h)).toContain('Removing "Small" cannot be undone');
    // Opening a project or file removes for good on purpose.
    h.app.setHistoryNotice(null);
    const c = addBall(h, "Third");
    await h.settle();
    h.app.removeMesh(h.store, c, { undoable: false });
    await h.settle();
    expect(notice(h)).toBeNull();
  });

  it("says it once for removals of one step", async () => {
    const h = await make(20_000);
    const balls = ["A", "B", "C"].map((n, k) => addBall(h, n, 3 * k));
    await h.settle();
    h.store.beginStep();
    for (const ball of balls) {
      h.app.removeMesh(h.store, ball);
    }
    h.store.endStep();
    await h.settle();
    expect(notice(h)).toMatch(/^Removing "A", "B" and 1 more cannot be undone/);
  });

  it("says that a join too large to undo is permanent", async () => {
    const run = async (budget?: number) => {
      const h = await make(budget);
      const quad = (name: string, z: number, indices: number[]) =>
        h.app.addMesh(h.store, {
          name,
          positions: Float64Array.from([0, 0, z, 1, 0, z, 1, 1, z, 0, 1, z]),
          indices: Uint32Array.from(indices),
          center: [0.5, 0.5, z],
          radius: 1,
          volumeId: null,
          preset: null,
        });
      const a = quad("A", 0, [0, 2, 1, 0, 3, 2]);
      const b = quad("B", 1, [0, 1, 2, 0, 2, 3]);
      await h.settle();
      const loops = await Promise.all([a, b].map((i) => h.settle(Promise.resolve(h.scene.analyze(i.mesh)))));
      await h.settle(joinMeshes(h.store, h.app, a.mesh, b, { loopA: loops[0]!.loops[0].id, loopB: loops[1]!.loops[0].id }));
      await h.settle();
      expect(items(h)).toEqual(["A"]);
      return { h, b };
    };
    const kept = await run();
    expect([notice(kept.h), kept.h.store.isRestorable(kept.b.mesh)]).toEqual([null, true]);
    const { h, b } = await run(100);
    expect(notice(h)).toMatch(/^Changing "A" and removing "B" cannot be undone/);
    expect(h.store.isRestorable(b.mesh)).toBe(false);
  });

  it("says that an undone copy too large to redo is gone", async () => {
    const h = await make(1000);
    const a = addBall(h, "Ball");
    await h.settle();
    duplicateMesh(h.store, h.app, a);
    await h.settle();
    expect([items(h), notice(h)]).toEqual([["Ball", "Ball copy"], null]);
    h.store.undo();
    await h.settle();
    expect(items(h)).toEqual(["Ball"]);
    expect(notice(h)).toMatch(/^Redo cannot bring back "Ball copy"/);
    expect(h.store.getSnapshot().history.canRedo).toBe(false);
  });
});
