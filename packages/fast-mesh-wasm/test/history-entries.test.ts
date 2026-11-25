// Entries of the application in the undo history (#282), on `Scene`, on
// `WorkerScene` (against the Worker engine in the Node `worker_threads`
// harness) and through the store: in order with mesh edits, in their step,
// coalesced, cancelled, budgeted, and dropped once; and on a Worker with undo
// while other calls are in flight.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it, vi } from "vitest";
import { type HistoryEntry, type MeshHandle, type PermanentStep, Scene } from "../src/node.js";
import { createSceneStore } from "../src/store/index.js";
import { debugSizes } from "../src/worker/client.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";
import type { Request, Result } from "../src/worker/protocol.js";

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

class NodeWorker implements WorkerLike {
  readonly worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), { workerData: { serve: true } });
  /** The next batch cannot be posted. */
  refuse = false;
  #error: ((e: unknown) => void) | undefined;

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    if (this.refuse) {
      this.refuse = false;
      throw new DOMException("no", "DataCloneError");
    }
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", cb: (e: any) => void): void {
    if (type === "error") {
      this.#error = cb;
    }
    this.worker.on(type, (data: unknown) => cb(type === "message" ? { data } : data));
  }

  terminate(): void {
    void this.worker.terminate();
  }

  crash(): void {
    this.#error?.({ message: "crashed" });
  }
}

const cleanup: (() => void)[] = [];
afterEach(() => {
  vi.unstubAllGlobals();
  for (const f of cleanup.splice(0)) {
    f();
  }
});

type AnyScene = Scene | WorkerScene;

async function makeLocal(): Promise<Scene> {
  const scene = new Scene();
  cleanup.push(() => scene.free());
  return scene;
}

const workers: NodeWorker[] = [];
async function makeWorker(): Promise<WorkerScene> {
  workers.length = 0;
  const createWorker = () => {
    const w = new NodeWorker();
    workers.push(w);
    return w;
  };
  const scene = await createWorkerScene({ createWorker });
  cleanup.push(() => scene.terminate());
  return scene;
}

/** A flat grid of `2 n²` triangles. */
function grid(n: number) {
  const positions: number[] = [];
  const indices: number[] = [];
  for (let y = 0; y <= n; y++) {
    for (let x = 0; x <= n; x++) {
      positions.push(x, y, 0);
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
const xOf = (s: AnyScene, m: MeshHandle) => s.placement(m).translation[0];
const canUndo = (s: AnyScene) => (s instanceof Scene ? s.canUndo() : s.canUndo);
const canRedo = (s: AnyScene) => (s instanceof Scene ? s.canRedo() : s.canRedo);
const bytesOf = (s: AnyScene) => (s instanceof Scene ? s.historyBytes() : s.historyBytes);

/** Raises a vertex: a mesh edit, not waited for. */
const lift = (s: AnyScene, m: MeshHandle, z: number): Promise<unknown> =>
  Promise.resolve(s.edit(m, [{ op: "setPositions", vertices: [0], positions: [0, 0, z] }]));
const zOf = (s: AnyScene, m: MeshHandle) => s.views(m).positions()[2];

/**
 * The application: a stack of names. Each entry pushes one; its undo must
 * find it on top and its redo must find what it was pushed on, so a callback
 * run in any other order than the history's throws.
 */
class Stack {
  readonly names: string[] = [];
  readonly log: string[] = [];
  readonly dropped: string[] = [];

  /** Pushes `name` and returns the entry for it. */
  push(name: string, more: Partial<HistoryEntry> = {}): HistoryEntry {
    const below = this.names.join();
    this.names.push(name);
    let at = true;
    return {
      undo: () => {
        this.log.push(`undo ${name}`);
        expect([name, at, this.names.at(-1)]).toEqual([name, true, name]);
        this.names.pop();
        at = false;
      },
      redo: () => {
        this.log.push(`redo ${name}`);
        expect([name, at]).toEqual([name, false]);
        if (!this.loose) {
          expect(this.names.join()).toBe(below);
        }
        this.names.push(name);
        at = true;
      },
      drop: () => {
        expect(this.dropped).not.toContain(name);
        this.dropped.push(name);
      },
      ...more,
    };
  }

  /** Entries may be redone on something else than they were made on (a step before them was undone meanwhile). */
  loose = false;

  take(): string[] {
    return this.log.splice(0);
  }
}

/** Collects the errors callbacks throw, which are reported and never thrown. */
function reported(): unknown[] {
  const errors: unknown[] = [];
  vi.stubGlobal("reportError", (e: unknown) => errors.push(e));
  return errors;
}

function code(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    return (e as { code?: string }).code ?? String(e);
  }
  return "no error";
}

for (const [name, make] of [
  ["Scene", makeLocal],
  ["WorkerScene", makeWorker],
] as const) {
  describe(`application entries: ${name}`, () => {
    it("interleave with edits, moves, adds and removals, in order", async () => {
      const errors = reported();
      const s: AnyScene = await make();
      const app = new Stack();
      const m = add(s, 2);
      await settle(s);
      s.recordEntry(app.push("a"));
      await frame(s, lift(s, m, 1));
      s.recordEntry(app.push("b"));
      move(s, m, 5);
      const n = add(s, 1, { undoable: true });
      s.removeMesh(m, { undoable: true });
      s.recordEntry(app.push("c"));
      await settle(s);
      const state = () => [
        app.names.join(""),
        s.has(m) ? zOf(s, m) > 0.5 : "out",
        s.has(m) ? xOf(s, m) : "out",
        s.has(n),
      ];
      const states = [
        ["abc", "out", "out", true],
        ["ab", "out", "out", true],
        ["ab", true, 5, true],
        ["ab", true, 5, false],
        ["ab", true, 0, false],
        ["a", true, 0, false],
        ["a", false, 0, false],
        ["", false, 0, false],
      ];
      for (let round = 0; round < 2; round++) {
        expect(state()).toEqual(states[0]);
        for (const want of states.slice(1)) {
          expect(canUndo(s)).toBe(true);
          s.undo();
          await settle(s);
          expect(state()).toEqual(want);
        }
        expect(canUndo(s)).toBe(false);
        s.undo();
        await settle(s);
        for (const want of [...states].reverse().slice(1)) {
          expect(canRedo(s)).toBe(true);
          s.redo();
          await settle(s);
          expect(state()).toEqual(want);
        }
        expect(canRedo(s)).toBe(false);
      }
      const round = ["undo c", "undo b", "undo a", "redo a", "redo b", "redo c"];
      expect(app.take()).toEqual([...round, ...round]);
      expect([app.dropped, errors]).toEqual([[], []]);
    });

    it("join the open step: one undo, the reverse order, and nested steps", async () => {
      const errors = reported();
      const s: AnyScene = await make();
      const app = new Stack();
      const m = add(s, 2);
      await settle(s);
      s.beginStep();
      s.recordEntry(app.push("a"));
      await frame(s, lift(s, m, 1));
      s.beginStep();
      s.recordEntry(app.push("b"));
      move(s, m, 1);
      s.endStep();
      s.recordEntry(app.push("c"));
      s.endStep();
      await settle(s);
      s.undo();
      await settle(s);
      expect([app.take(), zOf(s, m), xOf(s, m), canUndo(s)]).toEqual([["undo c", "undo b", "undo a"], 0, 0, false]);
      s.redo();
      await settle(s);
      expect([app.take(), xOf(s, m), canRedo(s)]).toEqual([["redo a", "redo b", "redo c"], 1, false]);
      // Undo with the step open undoes it.
      s.beginStep();
      s.recordEntry(app.push("d"));
      s.undo();
      await settle(s);
      expect([app.take(), app.names]).toEqual([["undo d"], ["a", "b", "c"]]);
      expect(code(() => s.endStep())).toBe("STEP_NOT_OPEN");
      expect([app.dropped, errors]).toEqual([[], []]);
    });

    it("cancel with their step: undone, then dropped", async () => {
      const errors = reported();
      const s: AnyScene = await make();
      const app = new Stack();
      const m = add(s, 2);
      await settle(s);
      s.recordEntry(app.push("kept"));
      s.beginStep();
      s.recordEntry(app.push("a"));
      move(s, m, 1);
      s.recordEntry(app.push("b"));
      s.cancelStep();
      await settle(s);
      expect([app.take(), app.names, xOf(s, m)]).toEqual([["undo b", "undo a"], ["kept"], 0]);
      expect([...app.dropped].sort()).toEqual(["a", "b"]);
      s.undo();
      await settle(s);
      expect([app.take(), app.names, canUndo(s)]).toEqual([["undo kept"], [], false]);
      expect(errors).toEqual([]);
    });

    it("coalesce in an open step: the first is undone, the last redone", async () => {
      const s: AnyScene = await make();
      const m = add(s, 2);
      await settle(s);
      let value = 0;
      const log: string[] = [];
      const set = (to: number, coalesce?: string): void => {
        const from = value;
        value = to;
        const undo = () => {
          log.push(`undo ${to}`);
          value = from;
        };
        const redo = () => {
          log.push(`redo ${to}`);
          value = to;
        };
        s.recordEntry({ undo, redo, drop: () => log.push(`drop ${to}`), bytes: 1000, coalesce });
      };
      s.beginStep();
      for (let k = 1; k <= 50; k++) {
        set(k, "slider");
      }
      await settle(s);
      expect(log.splice(0)).toEqual(Array.from({ length: 48 }, (_, k) => `drop ${k + 2}`));
      // Two of them are held.
      expect(bytesOf(s)).toBeGreaterThan(2000);
      expect(bytesOf(s)).toBeLessThan(2400);
      set(51, "other");
      set(52, "slider");
      set(53);
      set(54);
      move(s, m, 1);
      set(55, "slider");
      set(56, "slider");
      s.endStep();
      await settle(s);
      expect(log).toEqual([]);
      for (let k = 0; k < 2; k++) {
        s.undo();
        await settle(s);
        expect(log.splice(0)).toEqual(["undo 55", "undo 54", "undo 53", "undo 52", "undo 51", "undo 1"]);
        expect(value).toBe(0);
        s.redo();
        await settle(s);
        expect(log.splice(0)).toEqual(["redo 50", "redo 51", "redo 52", "redo 53", "redo 54", "redo 56"]);
        expect(value).toBe(56);
      }
      // Steps of their own stay apart.
      set(57, "slider");
      set(58, "slider");
      await settle(s);
      s.undo();
      await settle(s);
      expect([log.splice(0), value]).toEqual([["undo 58"], 57]);
      s.clearHistory();
      await settle(s);
      const numbers = [1, 50, 51, 52, 53, 54, 55, 56, 57, 58];
      expect(log.sort()).toEqual(numbers.map((n) => `drop ${n}`).sort());
      expect(value).toBe(57);
    });

    it("count against the budget, and are dropped by it", async () => {
      const errors = reported();
      const s: AnyScene = await make();
      const app = new Stack();
      const m = add(s, 2);
      await settle(s);
      const permanent: PermanentStep[] = [];
      s.onPermanent((p) => permanent.push(p));
      s.recordEntry(app.push("free"));
      await settle(s);
      const overhead = bytesOf(s);
      expect(overhead).toBeGreaterThan(0);
      expect(overhead).toBeLessThan(200);
      s.recordEntry(app.push("a", { bytes: 10_000 }));
      await settle(s);
      expect(bytesOf(s)).toBe(10_000 + 2 * overhead);
      s.setHistoryBudget(25_000);
      s.recordEntry(app.push("b", { bytes: 10_000 }));
      move(s, m, 1);
      await settle(s);
      expect(app.dropped).toEqual([]);
      // The oldest steps make room, entries and mesh steps alike.
      s.recordEntry(app.push("c", { bytes: 10_000 }));
      await settle(s);
      expect(app.dropped).toEqual(["free", "a"]);
      expect(bytesOf(s)).toBeLessThanOrEqual(25_000);
      // Undone, an entry still counts: its redo is kept.
      const held = bytesOf(s);
      s.undo();
      await settle(s);
      expect([bytesOf(s), app.names.join()]).toEqual([held, "free,a,b"]);
      s.redo();
      await settle(s);
      expect(permanent).toEqual([]);

      // Larger than the whole budget: dropped at once, and the change stays.
      s.recordEntry(app.push("huge", { bytes: 30_000 }));
      await settle(s);
      expect(app.dropped).toEqual(["free", "a", "b", "c", "huge"]);
      expect([canUndo(s), bytesOf(s)]).toEqual([false, 0]);
      expect(permanent.map((p) => [p.lost, p.meshes, p.gone, p.bytes])).toEqual([["undo", [], [], 30_000 + overhead]]);
      s.undo();
      await settle(s);
      expect(app.names.at(-1)).toBe("huge");

      // A smaller budget drops them too.
      s.recordEntry(app.push("d", { bytes: 5000 }));
      s.recordEntry(app.push("e", { bytes: 5000 }));
      await settle(s);
      s.setHistoryBudget(6000);
      await settle(s);
      expect(app.dropped.slice(5)).toEqual(["d"]);
      s.undo();
      await settle(s);
      s.setHistoryBudget(0);
      await settle(s);
      expect(app.dropped.slice(5)).toEqual(["d", "e"]);
      expect([canUndo(s), canRedo(s), app.names.at(-1)]).toEqual([false, false, "d"]);
      // Whatever an entry says it holds.
      s.setHistoryBudget(1e6);
      s.beginStep();
      s.recordEntry(app.push("f", { bytes: Infinity }));
      s.recordEntry(app.push("g", { bytes: Infinity }));
      s.cancelStep();
      await settle(s);
      expect(bytesOf(s)).toBe(0);
      expect(errors).toEqual([]);
    });

    it("are dropped with the redo steps a new step replaces, whatever it is", async () => {
      const errors = reported();
      const s: AnyScene = await make();
      const app = new Stack();
      const m = add(s, 2);
      await settle(s);
      const recorders: (() => unknown)[] = [
        () => move(s, m, Math.random()),
        () => frame(s, lift(s, m, Math.random())),
        () => add(s, 1, { undoable: true }),
        () => s.removeMesh(add(s, 1), { undoable: true }),
        () => s.recordEntry(app.push("last")),
      ];
      for (const [k, record] of recorders.entries()) {
        s.recordEntry(app.push(`x${k}`));
        s.recordEntry(app.push(`y${k}`));
        s.undo();
        s.undo();
        await settle(s);
        expect(app.dropped).toHaveLength(2 * k);
        await record();
        await settle(s);
        expect(app.dropped.slice(2 * k).sort()).toEqual([`x${k}`, `y${k}`]);
      }
      expect(app.names).toEqual(["last"]);
      s.clearHistory();
      await settle(s);
      expect(app.dropped.at(-1)).toBe("last");
      expect([app.names, errors]).toEqual([["last"], []]);
    });

    it("need undo and redo, and leave the scene alone when they are refused", async () => {
      const s: AnyScene = await make();
      const fn = (): void => {
        throw new Error("called");
      };
      const bad = [
        null,
        undefined,
        {},
        { undo: fn },
        { redo: fn },
        { undo: fn, redo: 1 },
        { undo: fn, redo: fn, drop: 1 },
        { undo: fn, redo: fn, bytes: -1 },
        { undo: fn, redo: fn, bytes: NaN },
        { undo: fn, redo: fn, bytes: "1" },
        { undo: fn, redo: fn, coalesce: 1 },
      ];
      for (const entry of bad) {
        expect(code(() => s.recordEntry(entry as never))).toBe("INVALID_ARGUMENT");
      }
      const store = createSceneStore(s);
      cleanup.push(() => store.dispose());
      expect(code(() => store.recordEntry({ undo: fn } as never))).toBe("INVALID_ARGUMENT");
      await settle(s);
      expect(canUndo(s)).toBe(false);
    });

    it("are called as methods, and what they throw is reported while the undo stands", async () => {
      const errors = reported();
      const s: AnyScene = await make();
      const m = add(s, 2);
      await settle(s);
      class Payload implements HistoryEntry {
        done = true;
        dropped = 0;
        undo(): void {
          this.done = false;
          throw new Error("undo");
        }
        redo(): void {
          this.done = true;
          throw new Error("redo");
        }
        drop(): void {
          this.dropped++;
          throw new Error("drop");
        }
      }
      const [a, b] = [new Payload(), new Payload()];
      s.beginStep();
      s.recordEntry(a);
      move(s, m, 1);
      s.recordEntry(b);
      s.endStep();
      s.undo();
      await settle(s);
      expect([a.done, b.done, xOf(s, m)]).toEqual([false, false, 0]);
      s.redo();
      await settle(s);
      expect([a.done, b.done, xOf(s, m)]).toEqual([true, true, 1]);
      s.clearHistory();
      await settle(s);
      expect([a.dropped, b.dropped]).toEqual([1, 1]);
      const said = ["undo", "undo", "redo", "redo", "drop", "drop"];
      expect(errors.map((e) => (e as Error).message)).toEqual(said);
    });

    it("go through the store, whose snapshot is the step's in the callbacks", async () => {
      const errors = reported();
      const s: AnyScene = await make();
      const store = createSceneStore(s);
      cleanup.push(() => store.dispose());
      const app = new Stack();
      const g = grid(2);
      const m = store.addMesh(g.positions, g.indices);
      await settle(s);
      const seen: [string, boolean][] = [];
      const entry = app.push("a");
      store.beginStep();
      store.recordEntry({
        undo: () => {
          seen.push(["undo", store.getSnapshot().meshes.has(m)]);
          entry.undo();
        },
        redo: () => {
          seen.push(["redo", store.getSnapshot().meshes.has(m)]);
          entry.redo();
        },
        drop: () => entry.drop?.(),
        bytes: 100,
      });
      store.removeMesh(m, { undoable: true });
      store.endStep();
      await settle(s);
      expect(store.getSnapshot().history).toEqual({ canUndo: true, canRedo: false });
      store.undo();
      await settle(s);
      expect([app.names, store.getSnapshot().history]).toEqual([[], { canUndo: false, canRedo: true }]);
      store.redo();
      await settle(s);
      expect([app.names, store.getSnapshot().meshes.size]).toEqual([["a"], 0]);
      expect(seen).toEqual([
        ["undo", true],
        ["redo", false],
      ]);
      // It goes on when the store does not.
      store.dispose();
      s.undo();
      await settle(s);
      expect([app.names, app.dropped, errors]).toEqual([[], [], []]);
      expect(code(() => store.recordEntry(app.push("late")))).toBe("INVALID_ARGUMENT");
    });
  });
}

/** Waits for a promise call, with the frame that sends it. */
async function frame<T>(s: AnyScene, p: Promise<T>): Promise<T> {
  s.commitFrame();
  const out = await p;
  s.commitFrame();
  return out;
}

describe("application entries: Scene only", () => {
  it("are called before undo, redo and cancelStep return", async () => {
    const s = await makeLocal();
    const app = new Stack();
    const m = add(s, 2);
    s.recordEntry(app.push("a"));
    s.undo();
    expect(app.take()).toEqual(["undo a"]);
    s.redo();
    expect(app.take()).toEqual(["redo a"]);
    s.beginStep();
    s.recordEntry(app.push("b"));
    s.cancelStep();
    expect([app.take(), app.dropped]).toEqual([["undo b"], ["b"]]);
    // A coalesced drag's pending pose lands before the drag's undo.
    const drag = s.drag(m);
    drag.move({ rotation: IDENTITY, translation: [1, 0, 0] });
    s.recordEntry(app.push("in the drag"));
    s.undo();
    expect([app.take(), xOf(s, m)]).toEqual([["undo in the drag"], 0]);
  });

  it("may use the scene in their callbacks", async () => {
    const errors = reported();
    const s = await makeLocal();
    const m = add(s, 2);
    const log: string[] = [];
    move(s, m, 1);
    s.recordEntry({
      undo: () => log.push(`undo ${xOf(s, m)}`, `${s.undo()}`),
      redo: () => log.push("redo"),
    });
    s.undo();
    expect([log.splice(0), xOf(s, m)]).toEqual([["undo 1", "true"], 0]);
    // Even free it.
    s.clearHistory();
    s.beginStep();
    s.recordEntry({ undo: () => log.push("not reached"), redo: () => {} });
    s.recordEntry({ undo: () => s.free(), redo: () => {} });
    s.endStep();
    s.undo();
    expect([log, errors]).toEqual([[], []]);
  });

  it("are dropped soon after any call, and by free", async () => {
    const errors = reported();
    const s = await makeLocal();
    const app = new Stack();
    const m = add(s, 2);
    s.recordEntry(app.push("a"));
    s.undo();
    // Not a history call: it drops the redo branch all the same.
    move(s, m, 1);
    await Promise.resolve();
    expect(app.dropped).toEqual(["a"]);
    s.recordEntry(app.push("b"));
    s.recordEntry(app.push("c"));
    s.undo();
    s.free();
    await Promise.resolve();
    expect(app.dropped.sort()).toEqual(["a", "b", "c"]);
    expect([app.names, errors]).toEqual([["b"], []]);
  });
});

describe("application entries: WorkerScene only", () => {
  it("undo what was recorded last when undo was called, with calls in flight", async () => {
    const errors = reported();
    const s = await makeWorker();
    const app = new Stack();
    const m = add(s, 30);
    await settle(s);
    // Edits (long, unanswered) around an entry, and undos at once.
    const p = [lift(s, m, 1)];
    s.recordEntry(app.push("a"));
    p.push(lift(s, m, 2));
    s.undo(); // the second edit
    s.undo(); // "a"
    expect(app.names).toEqual(["a"]); // not heard of yet
    s.commitFrame();
    await Promise.all(p);
    // The callbacks ran before the promises of the same result.
    expect(app.take()).toEqual(["undo a"]);
    await settle(s);
    expect([app.names, zOf(s, m)]).toEqual([[], 1]);
    s.redo();
    s.redo();
    await settle(s);
    expect([app.take(), app.names, zOf(s, m)]).toEqual([["redo a"], ["a"], 2]);
    expect([app.dropped, errors]).toEqual([[], []]);
  });

  it("run in the history's order around entries recorded while an undo is in flight", async () => {
    const errors = reported();
    const s = await makeWorker();
    const app = new Stack();
    app.loose = true;
    s.recordEntry(app.push("a"));
    s.recordEntry(app.push("b"));
    s.undo(); // "b", not heard of yet
    s.recordEntry(app.push("c"));
    s.beginStep();
    s.recordEntry(app.push("d"));
    s.endStep();
    expect(app.names.join("")).toBe("abcd");
    await settle(s);
    // "c" and "d" step back for "b" to be undone, and forward again.
    expect(app.take()).toEqual(["undo d", "undo c", "undo b", "redo c", "redo d"]);
    expect([app.names.join(""), app.dropped]).toEqual(["acd", ["b"]]);
    for (const want of ["ac", "a", ""]) {
      s.undo();
      await settle(s);
      expect(app.names.join("")).toBe(want);
    }
    s.redo();
    s.redo();
    s.redo();
    await settle(s);
    expect(app.names.join("")).toBe("acd");
    app.take();

    // The same for a cancelled step, and for a redo.
    s.beginStep();
    s.recordEntry(app.push("e"));
    s.cancelStep();
    s.recordEntry(app.push("f"));
    await settle(s);
    expect(app.take()).toEqual(["undo f", "undo e", "redo f"]);
    s.undo();
    await settle(s);
    app.take();
    s.redo();
    s.recordEntry(app.push("g"));
    await settle(s);
    expect(app.take()).toEqual(["undo g", "redo f", "redo g"]);
    expect([app.names.join(""), errors]).toEqual(["acdfg", []]);
  });

  it("match a sync scene over random calls, whenever the results come", async () => {
    const errors = reported();
    // A small deterministic generator (mulberry32).
    let seed = 0x282;
    const random = (): number => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
    };
    for (let round = 0; round < 12; round++) {
      const worker = await makeWorker();
      const sync = await makeLocal();
      const apps = new Map<AnyScene, Stack>([
        [worker, new Stack()],
        [sync, new Stack()],
      ]);
      const meshes = new Map<AnyScene, MeshHandle>();
      for (const [s, app] of apps) {
        app.loose = true;
        meshes.set(s, add(s, 4));
        // Room for some of it only, in some rounds.
        s.setHistoryBudget(round % 3 === 0 ? 1500 : 1e9);
      }
      await settle(worker);
      let depth = 0;
      for (let k = 0; k < 60; k++) {
        const op = random();
        const x = Math.floor(random() * 1000);
        const bytes = Math.floor(random() * 300);
        for (const [s, app] of apps) {
          const m = meshes.get(s) as MeshHandle;
          if (op < 0.3) {
            s.recordEntry(app.push(`e${k}`, { bytes }));
          } else if (op < 0.45) {
            move(s, m, x);
          } else if (op < 0.5) {
            void lift(s, m, x).catch(() => {});
          } else if (op < 0.72) {
            s.undo();
          } else if (op < 0.86) {
            s.redo();
          } else if (op < 0.92) {
            s.beginStep();
          } else if (depth > 0 && op < 0.97) {
            s.endStep();
          } else if (depth > 0) {
            s.cancelStep();
          }
        }
        if (op >= 0.5 && op < 0.86) {
          depth = 0;
        } else if (op >= 0.86 && op < 0.92) {
          depth++;
        } else if (depth > 0 && op >= 0.92) {
          depth = op < 0.97 ? depth - 1 : 0;
        }
        // Results come now and then, at any point.
        if (random() < 0.15) {
          worker.commitFrame();
          await new Promise((r) => setTimeout(r, random() * 4));
        }
      }
      await settle(worker);
      sync.commitFrame();
      const [a, b] = [apps.get(worker) as Stack, apps.get(sync) as Stack];
      expect(a.names).toEqual(b.names);
      expect(xOf(worker, meshes.get(worker) as MeshHandle)).toBe(xOf(sync, meshes.get(sync) as MeshHandle));
      expect(zOf(worker, meshes.get(worker) as MeshHandle)).toBe(zOf(sync, meshes.get(sync) as MeshHandle));
      expect([canUndo(worker), canRedo(worker), bytesOf(worker)]).toEqual([canUndo(sync), canRedo(sync), bytesOf(sync)]);
      expect([...a.dropped].sort()).toEqual([...b.dropped].sort());
      // All the way back, then nothing is left behind.
      while (sync.canUndo()) {
        sync.undo();
        worker.undo();
      }
      await settle(worker);
      expect(a.names).toEqual(b.names);
      worker.clearHistory();
      await settle(worker);
      expect(debugSizes(worker).entries).toBe(0);
      expect(errors).toEqual([]);
      for (const f of cleanup.splice(0)) {
        f();
      }
    }
  });

  it("are dropped by a restart, a termination and a crash, each once", async () => {
    const errors = reported();
    const s = await makeWorker();
    const app = new Stack();
    s.recordEntry(app.push("a"));
    s.recordEntry(app.push("b"));
    s.undo();
    await settle(s);
    s.recordEntry(app.push("in flight")); // drops "b" in the old engine, unheard
    const restarted = s.restart();
    await Promise.resolve();
    expect(app.dropped.sort()).toEqual(["a", "b", "in flight"]);
    expect(debugSizes(s).entries).toBe(0);
    await restarted;
    s.commitFrame();
    expect([app.names, canUndo(s)]).toEqual([["a", "in flight"], false]);
    // New entries work in the new engine, whose ids start over.
    s.recordEntry(app.push("c"));
    s.undo();
    await settle(s);
    expect([app.names, app.take().at(-1)]).toEqual([["a", "in flight"], "undo c"]);

    s.recordEntry(app.push("d"));
    await settle(s);
    workers.at(-1)?.crash();
    s.recordEntry(app.push("e")); // the scene is poisoned: it fails when it is sent
    s.commitFrame();
    await Promise.resolve();
    expect(app.dropped.slice(3)).toEqual(expect.arrayContaining(["c", "d", "e"]));
    expect(debugSizes(s).entries).toBe(0);
    await s.restart();
    s.commitFrame();

    s.recordEntry(app.push("f"));
    s.terminate();
    await Promise.resolve();
    expect(app.dropped.slice(3).sort()).toEqual(["c", "d", "e", "f"]);
    expect(code(() => s.recordEntry(app.push("g")))).toBe("INVALID_ARGUMENT");
    expect(errors).toEqual([]);
  });

  it("are dropped when the batch they are in cannot be sent", async () => {
    const s = await makeWorker();
    const log: string[] = [];
    const entry = (name: string): HistoryEntry => ({
      undo: () => log.push(`undo ${name}`),
      redo: () => log.push(`redo ${name}`),
      drop: () => log.push(`drop ${name}`),
    });
    s.recordEntry(entry("a"));
    await settle(s);
    s.recordEntry(entry("b"));
    (workers.at(-1) as NodeWorker).refuse = true;
    s.commitFrame();
    expect(log.splice(0)).toEqual(["drop b"]);
    // What "b" changed stays; "a" was recorded last.
    s.undo();
    await settle(s);
    expect(log).toEqual(["undo a"]);
    expect(debugSizes(s).entries).toBe(1);
  });

  it("are reported by the engine per request, and dropped when it cannot record them", async () => {
    const worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), { workerData: {} });
    cleanup.push(() => void worker.terminate());
    const send = (batch: Request[]) =>
      new Promise<Result>((resolve) => {
        worker.once("message", resolve);
        worker.postMessage(batch);
      });
    const r = await send([
      { kind: "entry", rev: 1, id: 7, bytes: 10 },
      { kind: "entry", rev: 2, id: 8, bytes: "x" as never },
      { kind: "undo", rev: 3 },
      { kind: "redo", rev: 4 },
      { kind: "step", rev: 5, phase: "begin" },
      { kind: "entry", rev: 6, id: 9, bytes: 0, coalesce: "k" },
      { kind: "entry", rev: 7, id: 10, bytes: 0, coalesce: "k" },
      { kind: "step", rev: 8, phase: "cancel" },
    ]);
    expect(r.entries).toEqual([
      { id: 8, what: "drop", rev: 2 },
      { id: 7, what: "undo", rev: 3 },
      { id: 7, what: "redo", rev: 4 },
      { id: 9, what: "undo", rev: 8 },
      { id: 9, what: "drop", rev: 8 },
      { id: 10, what: "drop", rev: 8 },
    ]);
    const quiet = await send([{ kind: "step", rev: 9, phase: "begin" }]);
    expect("entries" in quiet).toBe(false);
  });
});
