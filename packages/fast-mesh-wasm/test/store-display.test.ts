// `SceneStore` declarative display (`fast-mesh-wasm/store`): distance
// colouring with store-owned fields, declarative sections, dependent and
// overall status, and `restart()`, over both hosts: the synchronous `Scene`
// (Node build) and `WorkerScene` against the Worker engine in the Node
// `worker_threads` harness (built dist/, `serve` mode). The cancellation test
// runs the engine in-process (sources) with its test-only `onChunk` hook.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it, vi } from "vitest";
import { type MeshHandle, Scene, type SectionPlane } from "../src/node.js";
import { createSceneStore, type DistanceColoring, type SceneStore } from "../src/store/index.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";
import { serveEngine } from "../src/worker/engine.js";
import type { Request } from "../src/worker/protocol.js";

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/** A closed tetrahedron offset by `dx` along x. */
function tetra(dx = 0) {
  const positions = Float64Array.from([dx, 0, 0, dx + 1, 0, 0, dx, 1, 0, dx, 0, 1]);
  const indices = Uint32Array.from([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]);
  return { positions, indices };
}

/**
 * The harness Worker as a `WorkerLike`. It records every batch sent and,
 * while `hold` is set, keeps results back until `release()`.
 */
class NodeWorker implements WorkerLike {
  readonly worker: Worker;
  readonly sent: Request[][] = [];
  hold = false;
  readonly held: unknown[] = [];
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  readonly #error: ((e: unknown) => void)[] = [];
  #arrived: (() => void)[] = [];

  constructor(workerData: { testPanic?: boolean } = {}) {
    this.worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), {
      workerData: { serve: true, ...workerData },
    });
    this.worker.on("message", (data: { kind?: string }) => {
      if (this.hold && data.kind === undefined) {
        this.held.push(data);
        for (const f of this.#arrived.splice(0)) {
          f();
        }
        return;
      }
      this.#dispatch(data);
    });
    this.worker.on("error", (e) => {
      for (const cb of this.#error) {
        cb(e);
      }
    });
  }

  #dispatch(data: unknown): void {
    for (const cb of this.#message) {
      cb({ data });
    }
  }

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.sent.push(msg as Request[]);
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", cb: (e: any) => void): void {
    (type === "message" ? this.#message : this.#error).push(cb);
  }

  terminate(): void {
    void this.worker.terminate();
  }

  /** Delivers the first `n` held results. */
  release(n = this.held.length): void {
    for (const m of this.held.splice(0, n)) {
      this.#dispatch(m);
    }
  }

  async waitHeld(n: number): Promise<void> {
    while (this.held.length < n) {
      await new Promise<void>((resolve) => this.#arrived.push(resolve));
    }
  }
}

/**
 * The engine in-process (sources) as a `WorkerLike`, with its `onChunk`
 * hook. Messages are cloned and delivered on a task, as a Worker's are;
 * while `holdIn` is set, batches sent wait for `releaseIn()`, which delivers
 * them all at once (so later ones are queued before the engine resumes).
 */
class InProcessWorker implements WorkerLike {
  readonly sent: Request[][] = [];
  holdIn = false;
  readonly #heldIn: unknown[] = [];
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  #deliver: ((data: unknown) => void) | undefined;
  #terminated = false;

  constructor(onChunk: (chunk: { kind: string; id?: number }) => void) {
    serveEngine(
      {
        post: (msg) => {
          const copy = structuredClone(msg);
          setTimeout(() => {
            if (!this.#terminated) {
              for (const cb of this.#message) {
                cb({ data: copy });
              }
            }
          });
        },
        onMessage: (cb) => {
          this.#deliver = cb;
        },
      },
      Promise.resolve(),
      { onChunk },
    );
  }

  postMessage(msg: unknown): void {
    this.sent.push(msg as Request[]);
    const copy = structuredClone(msg);
    if (this.holdIn) {
      this.#heldIn.push(copy);
      return;
    }
    setTimeout(() => this.#deliver?.(copy));
  }

  releaseIn(): void {
    this.holdIn = false;
    for (const m of this.#heldIn.splice(0)) {
      this.#deliver?.(m);
    }
  }

  addEventListener(type: "message" | "error", cb: (e: any) => void): void {
    if (type === "message") {
      this.#message.push(cb);
    }
  }

  terminate(): void {
    this.#terminated = true;
  }
}

const cleanup: (() => void)[] = [];
afterEach(() => {
  for (const f of cleanup.splice(0)) {
    f();
  }
  vi.restoreAllMocks();
});

/** Commits frames until nothing reads `"computing"`, then one more frame. */
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

/** A store with the counts of the scene calls the tests check, by host. */
interface Fixture {
  store: SceneStore;
  scene: Scene | WorkerScene;
  worker?: NodeWorker;
  fieldAdds(): number;
  fieldRemoves(): number;
  colorings(): number;
  planeSets(): number;
  /** Section `set`s of meshes or fill. */
  otherSectionSets(): number;
  /** Reads of a field's values (never needed by the store). */
  valueReads(): number;
}

async function makeLocal(): Promise<Fixture> {
  const scene = new Scene();
  const store = createSceneStore(scene);
  cleanup.push(() => {
    store.dispose();
    scene.free();
  });
  const add = vi.spyOn(Scene.prototype, "addDistanceField");
  const remove = vi.spyOn(Scene.prototype, "removeField");
  const color = vi.spyOn(Scene.prototype, "setColoring");
  const plane = vi.spyOn(Scene.prototype, "setSectionPlane");
  const meshes = vi.spyOn(Scene.prototype, "setSectionMeshes");
  const fill = vi.spyOn(Scene.prototype, "setSectionFill");
  const values = vi.spyOn(Scene.prototype, "fieldValues");
  return {
    store,
    scene,
    fieldAdds: () => add.mock.calls.length,
    fieldRemoves: () => remove.mock.calls.length,
    colorings: () => color.mock.calls.length,
    planeSets: () => plane.mock.calls.length,
    otherSectionSets: () => meshes.mock.calls.length + fill.mock.calls.length,
    valueReads: () => values.mock.calls.length,
  };
}

async function makeWorker(workerData?: { testPanic?: boolean }): Promise<Fixture> {
  let worker!: NodeWorker;
  const scene = await createWorkerScene({
    createWorker: () => {
      worker = new NodeWorker(workerData);
      return worker;
    },
  });
  const store = createSceneStore(scene);
  cleanup.push(() => {
    store.dispose();
    scene.terminate();
  });
  const reqs = () => worker.sent.flat();
  return {
    store,
    scene,
    get worker() {
      return worker;
    },
    fieldAdds: () => reqs().filter((r) => r.kind === "field" && r.op === "add").length,
    fieldRemoves: () => reqs().filter((r) => r.kind === "field" && r.op === "remove").length,
    colorings: () => reqs().filter((r) => r.kind === "coloring").length,
    planeSets: () => reqs().filter((r) => r.kind === "section" && r.op === "set" && r.desc?.plane !== undefined).length,
    otherSectionSets: () =>
      reqs().filter(
        (r) => r.kind === "section" && r.op === "set" && (r.desc?.meshes !== undefined || r.desc?.fill !== undefined),
      ).length,
    valueReads: () =>
      reqs().filter((r) => r.kind === "fetchFieldValues" || (r.kind === "query" && r.name === "fieldStatus")).length,
  };
}

function withTwo(store: SceneStore) {
  const a = store.addMesh(tetra(0).positions, tetra(0).indices);
  const b = store.addMesh(tetra(3).positions, tetra(3).indices);
  return { a, b };
}

function ruleTo(to: MeshHandle, maxDistance = 5): DistanceColoring {
  return { distanceTo: to, maxDistance, map: "turbo", range: [0, 2] };
}

const P1: SectionPlane = { origin: [0, 0, 0.2], normal: [0, 0, 1] };
const P2: SectionPlane = { origin: [0, 0, 0.3], normal: [0, 0, 1] };
const P3: SectionPlane = { origin: [0, 0, 0.4], normal: [0, 0, 1] };

describe.each([
  ["local", makeLocal],
  ["worker", () => makeWorker()],
] as const)("SceneStore display (%s)", (mode, make) => {
  it("toggle_storm_one_build", async () => {
    const f = await make();
    const { store } = f;
    expect(store.mode).toBe(mode);
    const { a, b } = withTwo(store);
    await settle(store);
    const rule = ruleTo(b);
    store.setDisplay(a, { coloring: rule });
    store.setDisplay(a, { coloring: null });
    store.setDisplay(a, { coloring: rule });
    await settle(store);
    expect(f.fieldAdds()).toBe(1);
    expect(f.colorings()).toBe(1);
    const st = store.getSnapshot().meshes.get(a)!;
    expect(st.display.coloring).toEqual(rule);
    expect(st.status).toBe("ready");
  });

  it("toggle_reuses_field", async () => {
    const f = await make();
    const { store } = f;
    const { a, b } = withTwo(store);
    await settle(store);
    store.setDisplay(a, { coloring: ruleTo(b) });
    await settle(store);
    expect(store.getSnapshot().meshes.get(a)!.display.coloring).toEqual(ruleTo(b));
    store.setDisplay(a, { coloring: null });
    await settle(store);
    expect(store.getSnapshot().meshes.get(a)!.display.coloring).toBe(null);
    store.setDisplay(a, { coloring: ruleTo(b) });
    await settle(store);
    expect(store.getSnapshot().meshes.get(a)!.display.coloring).toEqual(ruleTo(b));
    expect(f.fieldAdds()).toBe(1);
    expect(f.fieldRemoves()).toBe(0);
    expect(f.colorings()).toBe(3);
    expect(f.valueReads()).toBe(0);
  });

  it("field_replaced_on_key_change", async () => {
    const f = await make();
    const { store } = f;
    const { a, b } = withTwo(store);
    await settle(store);
    store.setDisplay(a, { coloring: ruleTo(b, 5) });
    await settle(store);
    store.setDisplay(a, { coloring: ruleTo(b, 6) });
    await settle(store);
    expect(f.fieldRemoves()).toBe(1);
    expect(f.fieldAdds()).toBe(2);
    const st = store.getSnapshot().meshes.get(a)!;
    expect(st.display.coloring).toEqual(ruleTo(b, 6));
    expect(st.status).toBe("ready");
    // A colour-only change keeps the field.
    store.setDisplay(a, { coloring: { ...ruleTo(b, 6), map: "viridis" } });
    await settle(store);
    expect(f.fieldAdds()).toBe(2);
    expect(store.getSnapshot().meshes.get(a)!.display.coloring?.map).toBe("viridis");
  });

  it("equal_rules_build_one_field", async () => {
    const f = await make();
    const { store } = f;
    const { a, b } = withTwo(store);
    await settle(store);
    // As StrictMode's double effects do: equal, not identical, rules.
    store.setDisplay(a, { coloring: { distanceTo: b, maxDistance: 5, map: "turbo", range: [0, 2] } });
    store.setDisplay(a, { coloring: { distanceTo: b, maxDistance: 5, map: "turbo", range: [0, 2] } });
    await settle(store);
    expect(f.fieldAdds()).toBe(1);
    expect(f.colorings()).toBe(1);
  });

  it("colouring_target_removed", async () => {
    const f = await make();
    const { store } = f;
    const { a, b } = withTwo(store);
    await settle(store);
    const rule = ruleTo(b);
    store.setDisplay(a, { coloring: rule });
    await settle(store);
    expect(() => store.removeMesh(b)).not.toThrow();
    await settle(store);
    const st = store.getSnapshot().meshes.get(a)!;
    expect(st.display.coloring).toBe(null);
    expect(st.desired.coloring).toEqual(rule);
    expect(st.status).toBe("error");
    expect(st.error?.code).toBe("UNKNOWN_MESH");
    // Turning the colouring off clears the error.
    store.setDisplay(a, { coloring: null });
    await settle(store);
    expect(store.getSnapshot().meshes.get(a)!.status).toBe("ready");
    expect(store.getSnapshot().status).toBe("idle");
  });

  it("section_declarative", async () => {
    const f = await make();
    const { store } = f;
    const { a, b } = withTwo(store);
    await settle(store);
    const s = store.addSection({ plane: P1, meshes: [a, b], fill: true });
    await settle(store);
    let st = store.getSnapshot().sections.get(s)!;
    expect(st.plane).toEqual(P1);
    expect(st.meshes).toEqual([a, b]);
    expect(st.fill).toBe(true);
    expect(st.desired).toEqual({ plane: P1, meshes: [a, b], fill: true });
    expect(st.status).toBe("ready");

    const planes = f.planeSets();
    store.setSection(s, { plane: P3 });
    store.setSection(s, { plane: P2 });
    expect(store.getSnapshot().sections.get(s)!.desired.plane).toEqual(P2);
    await settle(store);
    expect(f.planeSets()).toBe(planes + 1);
    expect(f.otherSectionSets()).toBe(0);
    st = store.getSnapshot().sections.get(s)!;
    expect(st.plane).toEqual(P2);
    expect(st.status).toBe("ready");

    store.setSection(s, { fill: false });
    await settle(store);
    expect(f.otherSectionSets()).toBe(1);
    expect(f.planeSets()).toBe(planes + 1);
    expect(store.getSnapshot().sections.get(s)!.fill).toBe(false);

    store.removeSection(s);
    expect(store.getSnapshot().sections.has(s)).toBe(false);
    await settle(store);
    expect(store.getSnapshot().sections.size).toBe(0);
  });

  it("overall_status", async () => {
    const f = await make();
    const { store } = f;
    const { a, b } = withTwo(store);
    await settle(store);
    expect(store.getSnapshot().status).toBe("idle");
    if (f.worker !== undefined) {
      f.worker.hold = true;
      store.setDisplay(a, { coloring: ruleTo(b) });
      store.commitFrame();
      await f.worker.waitHeld(1);
      expect(store.getSnapshot().status).toBe("computing");
      f.worker.hold = false;
      f.worker.release();
      await settle(store);
      expect(store.getSnapshot().status).toBe("idle");
    }
    // A failed action: an empty colour range.
    store.setDisplay(a, { coloring: { ...ruleTo(b), range: [2, 0] } });
    await settle(store);
    let snap = store.getSnapshot();
    expect(snap.status).toBe("error");
    expect(snap.error).toBeDefined();
    expect(snap.meshes.get(a)!.status).toBe("error");
    store.setDisplay(a, { coloring: ruleTo(b) });
    await settle(store);
    snap = store.getSnapshot();
    expect(snap.status).toBe("idle");
    expect(snap.error).toBeUndefined();
    expect(snap.meshes.get(a)!.display.coloring).toEqual(ruleTo(b));
  });
});

describe("SceneStore display (worker only)", () => {
  it("dependent_status_during_drag", async () => {
    const f = await makeWorker();
    const { store } = f;
    const { a, b } = withTwo(store);
    await settle(store);
    store.setDisplay(a, { coloring: ruleTo(b) });
    await settle(store);
    expect(store.getSnapshot().meshes.get(a)!.status).toBe("ready");
    const w = f.worker!;
    w.hold = true;
    const drag = store.drag(b);
    drag.move({ rotation: IDENTITY, translation: [0.5, 0, 0] });
    store.commitFrame();
    await w.waitHeld(1);
    expect(store.getSnapshot().meshes.get(a)!.status).toBe("computing");
    w.hold = false;
    w.release();
    await settle(store);
    expect(store.getSnapshot().meshes.get(a)!.status).toBe("ready");
    drag.end();
    await settle(store);
    expect(store.getSnapshot().meshes.get(a)!.status).toBe("ready");
  });

  it("superseded_field_build_is_cancelled", async () => {
    const builds: number[] = [];
    let w!: InProcessWorker;
    const scene = await createWorkerScene({
      createWorker: () => {
        w = new InProcessWorker((chunk) => {
          if (chunk.kind === "fieldBuild") {
            builds.push(chunk.id as number);
          }
        });
        return w;
      },
    });
    const store = createSceneStore(scene);
    cleanup.push(() => {
      store.dispose();
      scene.terminate();
    });
    const target = store.addMesh(tetra(10).positions, tetra(10).indices);
    const meshes = [0, 1, 2, 3, 4].map((k) => store.addMesh(tetra(2 * k).positions, tetra(2 * k).indices));
    await settle(store);

    w.holdIn = true;
    for (const m of meshes) {
      store.setDisplay(m, { coloring: ruleTo(target, Infinity) });
    }
    store.commitFrame();
    for (const m of meshes) {
      store.setDisplay(m, { coloring: null });
    }
    store.commitFrame();
    const adds = w.sent.flat().filter((r) => r.kind === "field" && r.op === "add");
    expect(adds).toHaveLength(5);
    expect(adds.map((r) => (r as { key?: string }).key)).toEqual(meshes.map((m) => `display:${scene.idOf(m)}`));
    w.releaseIn();
    await settle(store);

    expect(builds.length).toBeLessThan(5);
    for (const m of meshes) {
      const st = store.getSnapshot().meshes.get(m)!;
      expect(st.display.coloring).toBe(null);
      expect(st.status).toBe("ready");
    }
  });

  it("restart_keeps_declarative_state", async () => {
    const f = await makeWorker({ testPanic: true });
    const { store } = f;
    const scene = f.scene as WorkerScene;
    const { a, b } = withTwo(store);
    store.setDisplay(a, { coloring: ruleTo(b) });
    const s = store.addSection({ plane: P1, meshes: [a, b], fill: true });
    await settle(store);
    // The harness makes `analyze` trap: the engine is poisoned.
    const analysis = scene.analyze(a);
    store.commitFrame();
    await expect(analysis).rejects.toMatchObject({ code: "INSTANCE_POISONED" });

    await store.restart();
    await settle(store);
    const snap = store.getSnapshot();
    for (const st of snap.meshes.values()) {
      expect(st.status).toBe("ready");
    }
    expect(snap.sections.get(s)!.status).toBe("ready");
    expect(snap.sections.get(s)!.plane).toEqual(P1);
    const sa = snap.meshes.get(a)!;
    expect(sa.display.coloring).toEqual(sa.desired.coloring);
    expect(sa.display.coloring).toEqual(ruleTo(b));
    expect(snap.history).toEqual({ canUndo: false, canRedo: false });
    expect(snap.status).toBe("idle");
  });
});

describe("SceneStore display (worker only, review fixes)", () => {
  it("restart_rearms_unapplied_changes", async () => {
    const f = await makeWorker({ testPanic: true });
    const { store } = f;
    const scene = f.scene as WorkerScene;
    const { a, b } = withTwo(store);
    store.setDisplay(a, { coloring: ruleTo(b) });
    const s = store.addSection({ plane: P1, meshes: [a, b], fill: true });
    await settle(store);
    // The trapping query goes first in the batch, so the colouring and the
    // plane sent with it fail (poisoned) instead of applying.
    const analysis = scene.analyze(a);
    store.setDisplay(a, { coloring: null });
    store.setSection(s, { plane: P2 });
    store.commitFrame();
    await expect(analysis).rejects.toMatchObject({ code: "INSTANCE_POISONED" });
    for (let k = 0; k < 100 && store.getSnapshot().meshes.get(a)!.status !== "error"; k++) {
      await new Promise((resolve) => setTimeout(resolve, 2));
    }
    let snap = store.getSnapshot();
    expect(snap.meshes.get(a)!.status).toBe("error");
    expect(snap.meshes.get(a)!.display.coloring).toEqual(ruleTo(b));
    expect(snap.sections.get(s)!.plane).toEqual(P1);

    await store.restart();
    await settle(store);
    snap = store.getSnapshot();
    const sa = snap.meshes.get(a)!;
    expect(sa.display.coloring).toBe(null);
    expect(sa.display.coloring).toEqual(sa.desired.coloring);
    expect(sa.status).toBe("ready");
    const ss = snap.sections.get(s)!;
    expect(ss.plane).toEqual(P2);
    expect(ss.plane).toEqual(ss.desired.plane);
    expect(ss.status).toBe("ready");
    expect(snap.status).toBe("idle");
  });

  it("restart_clears_sync_errors", async () => {
    const f = await makeWorker();
    const { store } = f;
    const { a, b } = withTwo(store);
    store.setDisplay(a, { coloring: ruleTo(b) });
    await settle(store);
    store.removeMesh(b);
    await settle(store);
    expect(store.getSnapshot().meshes.get(a)!.status).toBe("error");
    await store.restart();
    await settle(store);
    // Statuses are the replay's once it is applied.
    expect(store.getSnapshot().meshes.get(a)!.status).toBe("ready");
  });

  it("dependent_status_target_outside_store", async () => {
    const f = await makeWorker();
    const { store } = f;
    const scene = f.scene as WorkerScene;
    const a = store.addMesh(tetra(0).positions, tetra(0).indices);
    // The target is added straight to the scene, not through the store.
    const b = scene.addMesh(tetra(3).positions, tetra(3).indices);
    await settle(store);
    store.setDisplay(a, { coloring: ruleTo(b) });
    await settle(store);
    expect(store.getSnapshot().meshes.get(a)!.status).toBe("ready");
    const w = f.worker!;
    w.hold = true;
    const drag = scene.drag(b);
    drag.move({ rotation: IDENTITY, translation: [0.5, 0, 0] });
    store.commitFrame();
    await w.waitHeld(1);
    expect(store.getSnapshot().meshes.get(a)!.status).toBe("computing");
    w.hold = false;
    w.release();
    drag.end();
    await settle(store);
    expect(store.getSnapshot().meshes.get(a)!.status).toBe("ready");
  });
});

describe("SceneStore display (local only)", () => {
  it("local_restart_rejects", async () => {
    const { store } = await makeLocal();
    await expect(store.restart()).rejects.toMatchObject({ code: "INVALID_ARGUMENT" });
  });
});
