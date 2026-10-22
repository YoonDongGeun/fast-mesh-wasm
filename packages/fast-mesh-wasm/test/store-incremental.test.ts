// The store's incremental refresh (issue #50) against a full rebuild: random
// sequences of actions, frames and waits (status events) over both hosts. The
// synchronous `Scene` (Node build) and `WorkerScene` against the Worker engine
// in the Node `worker_threads` harness (built dist/, `serve` mode). After every
// step, the snapshot the store holds must be the one a full rebuild (the
// test-only `setFullRefresh` switch) gives: deep-equal, and the same object
// (a full rebuild that finds nothing changed publishes nothing).
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import { type MeshHandle, Scene, type SectionHandle } from "../src/node.js";
import { createSceneStore, type DistanceColoring, type SceneStore } from "../src/store/index.js";
import { setFullRefresh } from "../src/store/store.js";
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

/** A seeded PRNG (mulberry32) in [0, 1). */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Asserts that the store's snapshot is the one a full rebuild gives: the
 * switch rebuilds every state at once, and a rebuild that finds nothing
 * changed keeps the snapshot (the same object).
 */
function expectFull(store: SceneStore, step: string): void {
  const incremental = store.getSnapshot();
  setFullRefresh(store, true);
  const full = store.getSnapshot();
  setFullRefresh(store, false);
  expect(full, step).toEqual(incremental);
  expect(full, step).toBe(incremental);
}

/** Commits frames until nothing is "computing" (a failed state may stay "error"), checking on the way. */
async function settle(store: SceneStore, step: string): Promise<void> {
  for (let k = 0; k < 200 && store.getSnapshot().status === "computing"; k++) {
    store.commitFrame();
    await new Promise((resolve) => setTimeout(resolve, 2));
    expectFull(store, `${step} ${k}`);
  }
  store.commitFrame();
  expectFull(store, `${step}, settled`);
}

describe.each([
  ["local", makeLocal, 400],
  ["worker", makeWorker, 300],
] as const)("SceneStore incremental refresh (%s)", (mode, make, steps) => {
  it.each([1, 2, 3, 4, 5, 6])("matches_full_rebuild_seed_%i", async (seed) => {
    const scene = await make();
    const store = createSceneStore(scene);
    cleanup.unshift(() => store.dispose());
    const random = rng(seed * 7919 + (mode === "local" ? 0 : 1));
    const pick = <T>(xs: readonly T[]): T | undefined =>
      xs.length === 0 ? undefined : xs[Math.floor(random() * xs.length)];
    const meshes: MeshHandle[] = [];
    const gone: MeshHandle[] = [];
    const sections: SectionHandle[] = [];
    let dx = 0;
    const add = () => {
      const { positions, indices } = tetra((dx += 2));
      meshes.push(store.addMesh(positions, indices));
    };
    // A mesh added straight to the scene: the store does not track it, but
    // colourings and sections may name it.
    const { positions, indices } = tetra(-4);
    const outside = scene.addMesh(positions, indices);
    for (let k = 0; k < 4; k++) {
      add();
    }
    expectFull(store, "start");

    const coloring = (): DistanceColoring | null => {
      const r = random();
      if (r < 0.25) {
        return null;
      }
      const to = r < 0.35 ? outside : r < 0.45 ? (pick(gone) ?? outside) : (pick(meshes) ?? outside);
      return {
        distanceTo: to,
        maxDistance: pick([5, 10, Infinity])!,
        map: pick(["viridis", "turbo"] as const)!,
        // An empty range is invalid: the flush puts the mesh in "error".
        range: random() < 0.1 ? [1, 1] : [0, pick([1, 2, 4])!],
      };
    };
    const quiet = (p: Promise<unknown>) => void p.catch(() => {});

    for (let i = 0; i < steps; i++) {
      const r = random();
      const m = pick(meshes);
      let step: string;
      if (r < 0.1 || m === undefined) {
        step = "addMesh";
        add();
      } else if (r < 0.15) {
        step = "removeMesh";
        store.removeMesh(m);
        meshes.splice(meshes.indexOf(m), 1);
        gone.push(m);
      } else if (r < 0.25) {
        step = "setPlacement";
        store.setPlacement(m, { rotation: IDENTITY, translation: [0, 0, random()] });
      } else if (r < 0.4) {
        step = "setDisplay";
        const q = random();
        store.setDisplay(
          m,
          q < 0.3 ? { opacity: pick([0.25, 0.5, 1])! } : q < 0.45 ? { visible: random() < 0.5 } : { coloring: coloring() },
        );
      } else if (r < 0.45) {
        step = "addSection";
        sections.push(
          store.addSection({
            plane: { origin: [0, 0, 0.5], normal: [0, 0, 1] },
            meshes: meshes.filter(() => random() < 0.5),
          }),
        );
      } else if (r < 0.55 && sections.length > 0) {
        step = "setSection";
        const q = random();
        const pool = [...meshes, outside];
        store.setSection(
          pick(sections)!,
          q < 0.4
            ? { plane: { origin: [0, 0, random()], normal: [0, 0, 1] } }
            : q < 0.8
              ? { meshes: pool.filter(() => random() < 0.5) }
              : { fill: random() < 0.5 },
        );
      } else if (r < 0.57 && sections.length > 0) {
        step = "removeSection";
        const s = pick(sections)!;
        store.removeSection(s);
        sections.splice(sections.indexOf(s), 1);
      } else if (r < 0.6) {
        step = "select";
        store.select(random() < 0.2 ? null : m);
      } else if (r < 0.65) {
        step = "undo/redo";
        if (random() < 0.6) {
          store.undo();
        } else {
          store.redo();
        }
      } else if (r < 0.7) {
        step = "edit";
        // An unknown op fails: its promise rejects, the mesh stays as it was.
        quiet(store.edit(m, random() < 0.3 ? [{ op: "nope" } as never] : []));
      } else if (r < 0.72) {
        step = "drag";
        const drag = store.drag(m);
        drag.move({ rotation: IDENTITY, translation: [random(), 0, 0] });
        if (random() < 0.5) {
          drag.end();
        } else {
          drag.cancel();
        }
      } else if (r < 0.85) {
        step = "commitFrame";
        store.commitFrame();
      } else if (r < 0.89) {
        step = "microtask";
        await Promise.resolve();
      } else if (r < 0.93) {
        step = "wait";
        await new Promise((resolve) => setTimeout(resolve, mode === "local" ? 0 : Math.floor(random() * 6)));
      } else if (r < 0.95) {
        // Straight through the scene: a colouring target the store does not track moves.
        step = "outside setPlacement";
        scene.setPlacement(outside, { rotation: IDENTITY, translation: [0, random(), 0] });
        // The store hears of it (the undo state, on the local host) through the scene's events.
        scene.commitFrame();
      } else if (r < 0.995 || mode === "local") {
        // Settled states show which ones a later change makes "computing".
        step = "settle";
        await settle(store, step);
      } else {
        step = "restart";
        quiet(store.restart());
      }
      expectFull(store, `step ${i}: ${step}`);
    }
    await settle(store, "end");
  }, 60_000);
});
