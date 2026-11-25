// The history budget (#281): set at construction or later, on `Scene`, on
// `WorkerScene` (against the Worker engine in the Node `worker_threads`
// harness) and through the store; what it keeps undoable, and the report of
// steps too large to keep.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_HISTORY_BUDGET, type MeshHandle, type PermanentStep, Scene } from "../src/node.js";
import { createSceneStore } from "../src/store/index.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

class NodeWorker implements WorkerLike {
  readonly worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), { workerData: { serve: true } });

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", cb: (e: any) => void): void {
    this.worker.on(type, (data: unknown) => cb(type === "message" ? { data } : data));
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

type AnyScene = Scene | WorkerScene;

async function makeLocal(historyBudget?: number): Promise<Scene> {
  const scene = new Scene({ historyBudget });
  cleanup.push(() => scene.free());
  return scene;
}

async function makeWorker(historyBudget?: number): Promise<WorkerScene> {
  const scene = await createWorkerScene({ createWorker: () => new NodeWorker(), historyBudget });
  cleanup.push(() => scene.terminate());
  return scene;
}

/** A flat grid of `2 n²` triangles at height `z`. */
function grid(n: number, z = 0) {
  const positions: number[] = [];
  const indices: number[] = [];
  for (let y = 0; y <= n; y++) {
    for (let x = 0; x <= n; x++) {
      positions.push(x, y, z);
    }
  }
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const a = y * (n + 1) + x;
      indices.push(a, a + 1, a + n + 1, a + 1, a + n + 2, a + n + 1);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

function add(s: AnyScene, n: number, options: { undoable?: boolean } = {}): MeshHandle {
  const g = grid(n);
  return s.addMesh(g.positions, g.indices, options);
}

/** Sends the frame, waits for every result so far and delivers it. */
async function settle(s: AnyScene): Promise<void> {
  if (s instanceof Scene) {
    s.commitFrame();
    return;
  }
  const barrier = s.raycast([0, 0, 1e6], [0, 0, 1]);
  s.commitFrame();
  await barrier;
  s.commitFrame();
}

const move = (s: AnyScene, m: MeshHandle, x: number) => s.setPlacement(m, { rotation: IDENTITY, translation: [x, 0, 0] });
const canUndo = (s: AnyScene) => (s instanceof Scene ? s.canUndo() : s.canUndo);
const canRedo = (s: AnyScene) => (s instanceof Scene ? s.canRedo() : s.canRedo);
const budgetOf = (s: AnyScene) => (s instanceof Scene ? s.historyBudget() : s.historyBudget);
const bytesOf = (s: AnyScene) => (s instanceof Scene ? s.historyBytes() : s.historyBytes);

/** What `onPermanent` reports, as it comes. */
function reports(s: { onPermanent(l: (p: PermanentStep) => void): () => void }): PermanentStep[] {
  const seen: PermanentStep[] = [];
  s.onPermanent((p) => seen.push(p));
  return seen;
}

function code(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    return (e as { code?: string }).code ?? String(e);
  }
  return "no error";
}

/** How many of `meshes`, removed in order, undo brings back, newest first. */
async function undoAll(s: AnyScene): Promise<number> {
  let n = 0;
  while (canUndo(s)) {
    s.undo();
    await settle(s);
    n++;
  }
  return n;
}

/** Two unit quads facing away from each other, for joins. */
const FACING = [
  { positions: Float64Array.from([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]), indices: Uint32Array.from([0, 2, 1, 0, 3, 2]) },
  { positions: Float64Array.from([0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]), indices: Uint32Array.from([0, 1, 2, 0, 2, 3]) },
] as const;

for (const [name, make] of [
  ["Scene", makeLocal],
  ["WorkerScene", makeWorker],
] as const) {
  describe(`history budget: ${name}`, () => {
    it("defaults to 256 MiB and takes a number of bytes, 0 or more", async () => {
      const s = await make();
      expect(DEFAULT_HISTORY_BUDGET).toBe(256 * 1024 * 1024);
      expect(budgetOf(s)).toBe(DEFAULT_HISTORY_BUDGET);
      for (const [set, got] of [
        [0, 0],
        [1234.9, 1234],
        [2 ** 32 - 1, 2 ** 32 - 1],
        [2 ** 40, 2 ** 32 - 1],
        [Infinity, 2 ** 32 - 1],
      ]) {
        s.setHistoryBudget(set);
        expect(budgetOf(s)).toBe(got);
      }
      for (const bad of [-1, NaN, -Infinity, "64" as never, undefined as never, null as never]) {
        expect(code(() => s.setHistoryBudget(bad))).toBe("INVALID_ARGUMENT");
      }
      expect(budgetOf(s)).toBe(2 ** 32 - 1);
      await expect(make(-1)).rejects.toMatchObject({ code: "INVALID_ARGUMENT" });
      await expect(make(NaN)).rejects.toMatchObject({ code: "INVALID_ARGUMENT" });
    });

    it("decides at construction what stays undoable", async () => {
      const counts: number[] = [];
      for (const budget of [undefined, 1, 2.5]) {
        const probe = await make();
        const one = await probe.removalBytes(add(probe, 30));
        const s = await make(budget === undefined ? undefined : budget * one + 1000);
        const meshes = [0, 1, 2, 3].map(() => add(s, 30));
        await settle(s);
        for (const m of meshes) {
          s.removeMesh(m, { undoable: true });
        }
        await settle(s);
        expect(bytesOf(s)).toBeLessThanOrEqual(budgetOf(s));
        counts.push(await undoAll(s));
        // The newest removals are the ones undone, and their meshes are whole.
        const back = meshes.map((m) => s.has(m));
        expect(back).toEqual([0, 1, 2, 3].map((k) => k >= 4 - (counts.at(-1) as number)));
        expect(meshes.filter((m) => s.has(m)).every((m) => s.views(m).indices().length === 3 * 1800)).toBe(true);
      }
      expect(counts).toEqual([4, 1, 2]);
    });

    it("evicts the oldest steps when it shrinks, and the rest still undo and redo", async () => {
      const s = await make();
      const m = add(s, 20);
      await settle(s);
      const lift = async (z: number) => {
        const { positions } = grid(20, z);
        await frame(s, s.edit(m, [{ op: "setPositions", vertices: Array.from({ length: 441 }, (_, v) => v), positions }]));
      };
      const z = () => s.views(m).positions()[2];
      const zs: number[] = [];
      for (let k = 1; k <= 5; k++) {
        await lift(k);
        zs.push(z());
      }
      await settle(s);
      const step = bytesOf(s) / 5;
      s.setHistoryBudget(2.5 * step);
      await settle(s);
      expect(bytesOf(s)).toBeCloseTo(2 * step, 6);
      expect(z()).toBe(zs[4]);
      s.undo();
      await settle(s);
      expect(z()).toBe(zs[3]);
      s.undo();
      await settle(s);
      expect(z()).toBe(zs[2]);
      expect(canUndo(s)).toBe(false);
      s.undo();
      await settle(s);
      expect(z()).toBe(zs[2]);
      // Undone steps go last, the furthest first.
      s.setHistoryBudget(1.5 * step);
      await settle(s);
      s.redo();
      await settle(s);
      expect(z()).toBe(zs[3]);
      expect(canRedo(s)).toBe(false);
      // A larger budget keeps more again.
      s.setHistoryBudget(Infinity);
      for (let k = 6; k <= 9; k++) {
        await lift(k);
      }
      await settle(s);
      expect(await undoAll(s)).toBe(5);
      expect(z()).toBe(zs[2]);
      s.clearHistory();
      await settle(s);
      expect(bytesOf(s)).toBe(0);
    });

    it("frees the meshes of evicted removals, and their handles stay out", async () => {
      const s = await make();
      const [a, b] = [add(s, 30), add(s, 30)];
      await settle(s);
      s.removeMesh(a, { undoable: true });
      s.removeMesh(b, { undoable: true });
      await settle(s);
      const one = bytesOf(s) / 2;
      s.setHistoryBudget(1.5 * one);
      await settle(s);
      expect([s.isRestorable(a), s.isRestorable(b)]).toEqual([false, true]);
      s.undo();
      await settle(s);
      expect([s.has(a), s.has(b), canUndo(s)]).toEqual([false, true, false]);
      // Undone, it holds nothing; redone, it does not fit any more.
      const seen = reports(s);
      s.setHistoryBudget(0);
      await settle(s);
      expect([s.has(b), canRedo(s)]).toEqual([true, true]);
      s.redo();
      await settle(s);
      expect([s.has(b), s.isRestorable(b), canUndo(s)]).toEqual([false, false, false]);
      expect(seen.map((p) => [p.lost, p.gone])).toEqual([["undo", [b]]]);
    });

    it("reports a removal too large to undo", async () => {
      const s = await make(20_000);
      const seen = reports(s);
      const [small, big] = [add(s, 2), add(s, 40)];
      await settle(s);
      const bytes = await frame(s, Promise.resolve(s.removalBytes(big)));
      expect(bytes).toBeGreaterThan(20_000);
      expect(await frame(s, Promise.resolve(s.removalBytes(small)))).toBeLessThan(20_000);
      // About what the docs say a triangle takes.
      expect(bytes / 3200).toBeGreaterThan(40);
      expect(bytes / 3200).toBeLessThan(100);
      s.removeMesh(small, { undoable: true });
      await settle(s);
      expect(seen).toEqual([]);
      expect(s.isRestorable(small)).toBe(true);
      s.removeMesh(big, { undoable: true });
      await settle(s);
      expect(seen).toEqual([{ lost: "undo", meshes: [big], gone: [big], bytes: seen[0].bytes }]);
      expect(Math.abs(seen[0].bytes - bytes)).toBeLessThan(512);
      expect([s.has(big), s.isRestorable(big), canUndo(s)]).toEqual([false, false, false]);
      // Removed for good on purpose: nothing to report.
      const other = add(s, 40);
      s.removeMesh(other);
      await settle(s);
      expect(seen).toHaveLength(1);
    });

    it("reports a join too large to undo, before its promise settles", async () => {
      const run = async (budget?: number) => {
        const s = await make(budget);
        const seen = reports(s);
        const [a, b] = FACING.map((f) => s.addMesh(f.positions, f.indices));
        await settle(s);
        const loops = await Promise.all([a, b].map((m) => frame(s, Promise.resolve(s.analyze(m)))));
        const told = await frame(
          s,
          Promise.resolve(s.join(a, b, { loopA: loops[0].loops[0].id, loopB: loops[1].loops[0].id })).then(() => seen.length),
        );
        await settle(s);
        return { s, seen, a, b, told };
      };
      const kept = await run();
      expect([kept.told, kept.s.isRestorable(kept.b), canUndo(kept.s)]).toEqual([0, true, true]);
      const { s, seen, a, b, told } = await run(64);
      expect([told, s.isRestorable(b), canUndo(s)]).toEqual([1, false, false]);
      expect(seen).toEqual([{ lost: "undo", meshes: [a, b], gone: [b], bytes: seen[0].bytes }]);
      expect(s.has(a) && !s.has(b)).toBe(true);
      expect(s.views(a).indices().length).toBeGreaterThan(6);
    });

    it("reports an undone add too large to redo, and keeps the steps before it", async () => {
      const s = await make(20_000);
      const seen = reports(s);
      const m = add(s, 2);
      await settle(s);
      move(s, m, 1);
      const big = add(s, 40, { undoable: true });
      await settle(s);
      expect(seen).toEqual([]);
      s.undo();
      await settle(s);
      expect(seen).toEqual([{ lost: "redo", meshes: [big], gone: [big], bytes: seen[0].bytes }]);
      expect([s.has(big), s.isRestorable(big), canRedo(s), canUndo(s)]).toEqual([false, false, false, true]);
      s.undo();
      await settle(s);
      expect(s.placement(m).translation).toEqual([0, 0, 0]);
    });

    it("reports a whole step, and nothing of steps that only make room", async () => {
      const s = await make(4096);
      const seen = reports(s);
      const m = add(s, 20);
      const n = add(s, 2);
      await settle(s);
      for (let k = 1; k <= 100; k++) {
        move(s, n, k);
      }
      await settle(s);
      expect(seen).toEqual([]);
      const { positions } = grid(20, 1);
      const vertices = Array.from({ length: 441 }, (_, v) => v);
      s.beginStep();
      move(s, n, 0.5);
      await frame(s, s.edit(m, [{ op: "setPositions", vertices, positions }]));
      s.endStep();
      await settle(s);
      expect(seen).toEqual([{ lost: "undo", meshes: [n, m], gone: [], bytes: seen[0].bytes }]);
      // Reverted, or dropped on purpose: no report.
      s.beginStep();
      await frame(s, s.edit(m, [{ op: "setPositions", vertices, positions: grid(20, 2).positions }]));
      s.cancelStep();
      move(s, n, 7);
      s.clearHistory();
      move(s, n, 8);
      s.setHistoryBudget(0);
      await settle(s);
      expect(seen).toHaveLength(1);
      // An edit that is a step of its own.
      await frame(s, s.edit(m, [{ op: "setPositions", vertices, positions: grid(20, 3).positions }]));
      await settle(s);
      expect(seen.map((p) => p.meshes)).toEqual([[n, m], [m]]);
    });

    it("stops telling a removed listener, and survives one that throws", async () => {
      const errors: unknown[] = [];
      vi.stubGlobal("reportError", (e: unknown) => errors.push(e));
      try {
        const s = await make(16);
        const m = add(s, 2);
        await settle(s);
        move(s, m, 1);
        await settle(s);
        // Registered late: told of what comes, not of what was.
        let calls = 0;
        const stop = s.onPermanent(() => {
          calls++;
          throw new Error("listener");
        });
        const seen = reports(s);
        move(s, m, 2);
        await settle(s);
        expect([calls, seen.length, errors.length]).toEqual([1, 1, 1]);
        stop();
        move(s, m, 3);
        await settle(s);
        expect([calls, seen.length]).toEqual([1, 2]);
        expect(s.placement(m).translation).toEqual([3, 0, 0]);
      } finally {
        vi.unstubAllGlobals();
      }
    });

    it("is an option and an action of the store, which tells too", async () => {
      const s = await make();
      const store = createSceneStore(s, { historyBudget: 20_000 });
      cleanup.push(() => store.dispose());
      expect([store.historyBudget, budgetOf(s)]).toEqual([20_000, 20_000]);
      const seen = reports(store);
      const g = grid(40);
      const big = store.addMesh(g.positions, g.indices);
      const more = store.addMesh(g.positions, g.indices);
      await settle(s);
      store.removeMesh(big, { undoable: true });
      await settle(s);
      expect(seen.map((p) => [p.lost, p.gone])).toEqual([["undo", [big]]]);
      expect(store.isRestorable(big)).toBe(false);
      expect(store.getSnapshot().history.canUndo).toBe(false);
      store.setHistoryBudget(1e9);
      store.removeMesh(more, { undoable: true });
      await settle(s);
      expect([store.historyBudget, seen.length, store.isRestorable(more)]).toEqual([1e9, 1, true]);
      store.setHistoryBudget(0);
      await settle(s);
      expect([store.isRestorable(more), store.getSnapshot().history.canUndo]).toEqual([false, false]);
      store.undo();
      await settle(s);
      expect(store.getSnapshot().meshes.size).toBe(0);
      expect(code(() => store.setHistoryBudget(-1))).toBe("INVALID_ARGUMENT");
      // Without the option, the scene's own.
      const other = await make(777);
      const plain = createSceneStore(other);
      cleanup.push(() => plain.dispose());
      expect(plain.historyBudget).toBe(777);
    });
  });
}

/** Waits for a promise call, with the frame that sends it. */
async function frame<T>(s: AnyScene, p: Promise<T> | T): Promise<T> {
  s.commitFrame();
  const out = await p;
  s.commitFrame();
  return out;
}

describe("history budget: Scene only", () => {
  it("tells of a step recorded by any call within a microtask", async () => {
    const s = await makeLocal(16);
    const seen = reports(s);
    const m = add(s, 2);
    move(s, m, 1);
    expect(seen).toEqual([]);
    await Promise.resolve();
    expect(seen).toHaveLength(1);
    s.removeMesh(m, { undoable: true });
    expect(seen).toHaveLength(2);
    // A scene freed meanwhile has nothing to tell.
    move(s, add(s, 2), 1);
    s.free();
    await Promise.resolve();
    expect(seen).toHaveLength(2);
  });
});

describe("history budget: WorkerScene only", () => {
  it("applies in call order, with calls in flight", async () => {
    const s = await makeWorker();
    const seen = reports(s);
    const [a, b, c] = [add(s, 30), add(s, 30), add(s, 30)];
    await settle(s);
    s.removeMesh(a, { undoable: true });
    s.setHistoryBudget(1000);
    s.removeMesh(b, { undoable: true });
    s.setHistoryBudget(DEFAULT_HISTORY_BUDGET);
    s.removeMesh(c, { undoable: true });
    expect(s.historyBudget).toBe(DEFAULT_HISTORY_BUDGET);
    await settle(s);
    // `a` made room on request; `b` did not fit; `c` does.
    expect([a, b, c].map((m) => s.isRestorable(m))).toEqual([false, false, true]);
    expect(seen.map((p) => p.gone)).toEqual([[b]]);
    expect(s.status("history").status).toBe("ready");
  });

  it("outlives a restart, also one that a change of it was waiting for", async () => {
    const s = await makeWorker(20_000);
    const seen = reports(s);
    const m = add(s, 40);
    await settle(s);
    await s.restart();
    s.commitFrame();
    expect([s.historyBudget, s.historyBytes]).toEqual([20_000, 0]);
    s.removeMesh(m, { undoable: true });
    await settle(s);
    expect(seen).toHaveLength(1);

    const n = add(s, 40);
    await settle(s);
    s.setHistoryBudget(1e8);
    await s.restart();
    s.commitFrame();
    expect(s.status("history").status).toBe("ready");
    s.removeMesh(n, { undoable: true });
    await settle(s);
    expect([seen.length, s.isRestorable(n), s.historyBudget]).toEqual([1, true, 1e8]);
  });
});
