// Issue #226: WorkerScene & SceneStore pointerdown-to-pointerup step nesting and stroke undo coalescing.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FastMeshError, type MeshHandle, type Vec3 } from "../src/node.js";
import { createSceneStore, type SceneStore } from "../src/store/index.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

function sphereMesh(center: Vec3 = [0, 0, 0], radius = 1, stacks = 10, slices = 20) {
  const positions: number[] = [center[0], center[1], center[2] + radius];
  for (let i = 1; i < stacks; i++) {
    const phi = (Math.PI * i) / stacks;
    for (let j = 0; j < slices; j++) {
      const theta = (2 * Math.PI * j) / slices;
      positions.push(
        center[0] + radius * Math.sin(phi) * Math.cos(theta),
        center[1] + radius * Math.sin(phi) * Math.sin(theta),
        center[2] + radius * Math.cos(phi),
      );
    }
  }
  positions.push(center[0], center[1], center[2] - radius);
  const south = positions.length / 3 - 1;
  const ring = (i: number, j: number) => 1 + (i - 1) * slices + (j % slices);
  const indices: number[] = [];
  for (let j = 0; j < slices; j++) {
    indices.push(0, ring(1, j), ring(1, j + 1));
    indices.push(south, ring(stacks - 1, j + 1), ring(stacks - 1, j));
  }
  for (let i = 1; i < stacks - 1; i++) {
    for (let j = 0; j < slices; j++) {
      const a = ring(i, j);
      const b = ring(i, j + 1);
      const c = ring(i + 1, j + 1);
      const d = ring(i + 1, j);
      indices.push(a, d, c, a, c, b);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

class NodeWorker implements WorkerLike {
  readonly worker: Worker;
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  readonly #error: ((e: unknown) => void)[] = [];

  constructor() {
    this.worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), {
      workerData: { serve: true },
    });
    this.worker.on("message", (data) => {
      for (const l of [...this.#message]) l({ data });
    });
    this.worker.on("error", (err) => {
      const ev = { message: err instanceof Error ? err.message : String(err) };
      for (const l of [...this.#error]) l(ev);
    });
  }

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", listener: (e: never) => void): void {
    (type === "message" ? this.#message : this.#error).push(listener as never);
  }

  removeEventListener(type: "message" | "error", listener: (e: never) => void): void {
    const list = type === "message" ? this.#message : this.#error;
    const i = list.indexOf(listener as never);
    if (i >= 0) list.splice(i, 1);
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

async function flushMicrotasks(): Promise<void> {
  await new Promise((r) => setTimeout(r, 30));
}

async function settle(scene: WorkerScene, mesh: MeshHandle): Promise<void> {
  scene.commitFrame();
  await scene.bounds(mesh);
  await flushMicrotasks();
}

function liveIndices(indices: Uint32Array): number[] {
  const out: number[] = [];
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i]!;
    const b = indices[i + 1]!;
    const c = indices[i + 2]!;
    if (a !== 0 || b !== 0 || c !== 0) {
      out.push(a, b, c);
    }
  }
  return out;
}

describe("Issue #226: WorkerScene & SceneStore step nesting + stroke coalescing", () => {
  const cleanups: (() => void)[] = [];

  afterEach(() => {
    for (const c of cleanups.splice(0)) c();
  });

  async function openWorker() {
    const worker = await createWorkerScene({
      createWorker: () => new NodeWorker(),
    });
    cleanups.push(() => worker.terminate());
    return worker;
  }

  it("coalesces multi-dab + splitStretched + remesh stroke on WorkerScene into 1 undo/redo entry", async () => {
    const worker = await openWorker();
    const sphere = sphereMesh([0, 0, 0], 1, 8, 16);
    const mesh = worker.addMesh(sphere.positions, sphere.indices);
    await settle(worker, mesh);

    const beforeVolume = (await worker.inspect.measure(mesh)).volume;
    const beforePositions = worker.views(mesh).positions().slice();
    const beforeIndices = liveIndices(worker.views(mesh).indices());
    expect(worker.canUndo).toBe(false);
    expect(worker.canRedo).toBe(false);

    // Begin stroke -> multi-dab grab -> live splitStretched -> remesh -> end stroke
    await worker.sculpt.beginStroke(mesh, { captureRestSizes: true });
    await worker.sculpt.grab(mesh, {
      center: [0, 0, 1],
      radius: 0.7,
      falloff: "smooth",
      strength: 0.6,
      direction: [0, 0, 0.5],
      fromStroke: true,
    });
    await worker.sculpt.grab(mesh, {
      center: [0, 0, 1.1],
      radius: 0.7,
      falloff: "smooth",
      strength: 0.9,
      direction: [0, 0, 0.4],
      fromStroke: true,
    });
    const stretched = await worker.sculpt.splitStretched(mesh, {
      center: [0, 0, 1],
      radius: 1.5,
      limit: 1.3,
    });
    expect(stretched.splits).toBeGreaterThanOrEqual(0);
    await worker.remesh.remesh(mesh, {
      center: [0, 0, 1],
      radius: 1.5,
      targetLength: 0.35,
      maxRounds: 1,
    });
    worker.sculpt.endStroke();
    worker.commitFrame();

    await vi.waitFor(() => expect(worker.canUndo).toBe(true));
    const afterVolume = (await worker.inspect.measure(mesh)).volume;
    const afterPositions = worker.views(mesh).positions().slice();
    const afterIndices = worker.views(mesh).indices().slice();
    expect(afterPositions).not.toEqual(beforePositions);
    expect(afterVolume).not.toEqual(beforeVolume);

    // A single undo() reverts every dab + splitStretched + remesh at once.
    worker.undo();
    worker.commitFrame();
    await vi.waitFor(() => expect(worker.canUndo).toBe(false));
    expect(worker.canRedo).toBe(true);
    expect((await worker.inspect.measure(mesh)).volume).toEqual(beforeVolume);
    expect(worker.views(mesh).positions().subarray(0, beforePositions.length)).toEqual(beforePositions);
    expect(liveIndices(worker.views(mesh).indices())).toEqual(beforeIndices);

    // A single redo() restores the entire sculpted/remeshed stroke.
    worker.redo();
    worker.commitFrame();
    await vi.waitFor(() => expect(worker.canRedo).toBe(false));
    expect(worker.canUndo).toBe(true);
    expect((await worker.inspect.measure(mesh)).volume).toEqual(afterVolume);
    expect(worker.views(mesh).positions()).toEqual(afterPositions);
    expect(worker.views(mesh).indices()).toEqual(afterIndices);
  });

  it("cancelling or completing empty strokes leaves 0 undo entries and reverts changes", async () => {
    const worker = await openWorker();
    const sphere = sphereMesh([0, 0, 0], 1, 8, 16);
    const mesh = worker.addMesh(sphere.positions, sphere.indices);
    await settle(worker, mesh);

    const baselineVolume = (await worker.inspect.measure(mesh)).volume;
    const baselinePos = worker.views(mesh).positions().slice();
    const baselineIdx = liveIndices(worker.views(mesh).indices());

    // Empty beginStep/endStep -> 0 undo entries
    worker.beginStep();
    worker.endStep();
    worker.commitFrame();
    await flushMicrotasks();
    expect(worker.canUndo).toBe(false);

    // Cancelled stroke after multi-dab + remesh -> 0 undo entries and geometry restored
    await worker.sculpt.beginStroke(mesh, { captureRestSizes: true });
    await worker.sculpt.inflate(mesh, {
      center: [0, 0, 1],
      radius: 0.8,
      falloff: "smooth",
      strength: 0.4,
    });
    await worker.sculpt.splitStretched(mesh, { center: [0, 0, 1], radius: 1.5, limit: 1.25 });
    await worker.remesh.remesh(mesh, {
      center: [0, 0, 1],
      radius: 1.5,
      targetLength: 0.35,
      maxRounds: 1,
    });
    worker.sculpt.cancelStroke();
    worker.commitFrame();

    await vi.waitFor(() => expect(worker.canUndo).toBe(false));
    expect(worker.canRedo).toBe(false);
    expect((await worker.inspect.measure(mesh)).volume).toEqual(baselineVolume);
    expect(worker.views(mesh).positions().subarray(0, baselinePos.length)).toEqual(baselinePos);
    expect(liveIndices(worker.views(mesh).indices())).toEqual(baselineIdx);
  });

  it("enforces STEP_NOT_OPEN, nesting, drag interaction, clearHistory, and restart semantics", async () => {
    const worker = await openWorker();
    const sphere = sphereMesh();
    const mesh = worker.addMesh(sphere.positions, sphere.indices);
    await settle(worker, mesh);

    expect(() => worker.endStep()).toThrowError(FastMeshError);
    try {
      worker.endStep();
    } catch (e) {
      expect((e as FastMeshError).code).toBe("STEP_NOT_OPEN");
    }
    expect(() => worker.cancelStep()).toThrowError(FastMeshError);

    // Nested beginStep + drag + endStep coalesces into 1 step; inner cancelStep unwinds everything.
    worker.beginStep();
    const drag = worker.drag(mesh);
    drag.move({ rotation: IDENTITY, translation: [2, 0, 0] });
    worker.beginStep();
    drag.move({ rotation: IDENTITY, translation: [4, 0, 0] });
    worker.cancelStep();
    worker.commitFrame();
    await vi.waitFor(() => expect(drag.active).toBe(false));
    expect(worker.canUndo).toBe(false);
    expect(worker.placement(mesh).translation).toEqual([0, 0, 0]);
    expect(() => worker.endStep()).toThrowError(FastMeshError);

    // Nested steps followed by outer endStep coalesce and clearHistory() resets open stepDepth.
    worker.beginStep();
    worker.setPlacement(mesh, { rotation: IDENTITY, translation: [1, 0, 0] });
    worker.beginStep();
    await worker.sculpt.inflate(mesh, {
      center: [0, 0, 1],
      radius: 0.7,
      falloff: "smooth",
      strength: 0.2,
    });
    worker.endStep();
    // Still inside outer step -> ending outer commits 1 undo step
    worker.endStep();
    worker.commitFrame();
    await vi.waitFor(() => expect(worker.canUndo).toBe(true));

    worker.beginStep();
    worker.clearHistory();
    worker.commitFrame();
    await vi.waitFor(() => expect(worker.canUndo).toBe(false));
    expect(() => worker.endStep()).toThrowError(FastMeshError);

    // Restart while a step is open resets stepDepth
    worker.beginStep();
    await worker.restart();
    expect(() => worker.endStep()).toThrowError(FastMeshError);
  });

  it("publishes SceneStore history reactivity across beginStep/endStep/cancelStep", async () => {
    const worker = await openWorker();
    const store: SceneStore = createSceneStore(worker);
    cleanups.push(() => store.dispose());

    const sphere = sphereMesh();
    const mesh = store.addMesh(sphere.positions, sphere.indices);
    await settle(worker, mesh);
    const initialPositions = worker.views(mesh).positions().slice();

    const historyEvents: { canUndo: boolean; canRedo: boolean }[] = [];
    const unsub = store.subscribe(() => {
      historyEvents.push({ ...store.getSnapshot().history });
    });

    // Empty open step keeps canUndo false until an edit runs.
    store.beginStep();
    store.commitFrame();
    await flushMicrotasks();
    expect(store.getSnapshot().history.canUndo).toBe(false);

    await store.sculpt(mesh, [
      { op: "inflate", center: [0, 0, 1], radius: 0.7, falloff: "smooth", strength: 0.15 },
    ]);
    store.setPlacement(mesh, { rotation: IDENTITY, translation: [2, 0, 0] });
    store.endStep();
    store.commitFrame();
    await vi.waitFor(() => expect(worker.placement(mesh).translation).toEqual([2, 0, 0]));
    expect(store.getSnapshot().history.canUndo).toBe(true);
    expect(historyEvents.some((h) => h.canUndo)).toBe(true);
    const committedPositions = worker.views(mesh).positions().slice();
    expect(committedPositions).not.toEqual(initialPositions);

    // Cancelling another multi-edit step leaves canRedo/canUndo unchanged and reverts geometry + pose.
    store.beginStep();
    store.setPlacement(mesh, { rotation: IDENTITY, translation: [9, 0, 0] });
    await store.sculpt(mesh, { op: "inflate", center: [0, 0, 1], radius: 0.7, falloff: "smooth", strength: 0.3 });
    await vi.waitFor(() => expect(worker.placement(mesh).translation).toEqual([9, 0, 0]));
    store.cancelStep();
    store.commitFrame();
    await vi.waitFor(() => expect(worker.placement(mesh).translation).toEqual([2, 0, 0]));
    expect(store.getSnapshot().history.canUndo).toBe(true);
    expect(store.getSnapshot().history.canRedo).toBe(false);
    expect(worker.views(mesh).positions()).toEqual(committedPositions);

    // Single undo drops both the committed sculpt and placement at once.
    store.undo();
    store.commitFrame();
    await vi.waitFor(() => expect(store.getSnapshot().history.canUndo).toBe(false));
    expect(store.getSnapshot().history.canRedo).toBe(true);
    expect(worker.placement(mesh).translation).toEqual([0, 0, 0]);
    expect(worker.views(mesh).positions()).toEqual(initialPositions);
    unsub();
  });
});
