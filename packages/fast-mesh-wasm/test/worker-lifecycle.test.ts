// Phase 9a lifecycle: cooperative cancellation of long Worker operations,
// stale results, and `WorkerScene.restart()`. The cancellation test runs the
// engine in-process (sources, same wasm) with its test-only `onChunk` hook;
// the restart tests run `WorkerScene` against the Node `worker_threads`
// harness (built dist/), as test/worker-client.test.ts does.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import { type MeshHandle, Scene, type Vec3 } from "../src/node.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";
import { debugDump } from "../src/worker/client.js";
import { runEngine } from "../src/worker/engine.js";
import type { ClientId, Request, Result } from "../src/worker/protocol.js";

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/** A closed UV sphere (even stack count). */
function sphere(center: Vec3, radius = 1, stacks = 8, slices = 16) {
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

// ---- The engine in-process ----

let revision = 0;
const rev = () => ++revision;

function addMeshReq(id: ClientId, mesh: { positions: Float64Array; indices: Uint32Array }): Request {
  return {
    kind: "addMesh",
    rev: rev(),
    id,
    positions: mesh.positions.slice().buffer,
    f64: true,
    indices: mesh.indices.slice().buffer,
    options: {},
  };
}

describe("cooperative cancellation", () => {
  it("superseded_long_field_build_is_cancelled", async () => {
    const results: Result[] = [];
    let arrived: (() => void) | undefined;
    let deliver!: (batch: Request[]) => void;
    const builds: ClientId[] = [];
    let newer: Request[] | undefined;
    runEngine(
      {
        post: (msg) => {
          results.push(msg);
          arrived?.();
        },
        onMessage: (cb) => {
          deliver = cb;
        },
      },
      {
        onChunk: (chunk) => {
          if (chunk.kind !== "fieldBuild") {
            return;
          }
          builds.push(chunk.id as ClientId);
          if (builds.length === 1 && newer !== undefined) {
            // The newer colouring arrives during the build (after chunk 1),
            // as a message would: the engine sees it only if it yields.
            const batch = newer;
            newer = undefined;
            queueMicrotask(() => deliver(batch));
          }
        },
      },
    );
    const until = async (n: number) => {
      while (results.length < n) {
        await new Promise<void>((resolve) => {
          arrived = resolve;
        });
      }
    };

    const s = sphere([0, 0, 0], 1, 24, 48);
    const t = sphere([2.2, 0, 0], 1, 24, 48);
    deliver([addMeshReq(1, s), addMeshReq(2, t)]);
    await until(1);

    // Five fields built for one display, then the colouring that uses them.
    const display = "display:1";
    const fields: Request[] = [1, 2, 3, 4, 5].map((field) => ({
      kind: "field",
      rev: rev(),
      field,
      op: "add",
      desc: { name: `f${field}`, on: 1, to: 2, maxDistance: Infinity },
      key: display,
    }));
    const first: Request = {
      kind: "coloring",
      rev: rev(),
      id: 1,
      rule: { field: 5, map: "turbo", range: [0, 3] },
      key: display,
    };
    const second: Request = {
      kind: "coloring",
      rev: rev(),
      id: 1,
      rule: { field: 3, map: "viridis", range: [0, 3] },
      key: display,
    };
    newer = [second];
    deliver([...fields, first]);
    await until(3);

    // Chunk 1 ran; the rest of the superseded build was abandoned, and only
    // the field the newer colouring uses was built (on demand).
    expect(builds).toEqual([1, 3]);
    expect(builds.length).toBeLessThan(5);
    // One result per batch, in order.
    const [r1, r2] = results.slice(1);
    expect(r1.rev).toBe(first.rev);
    expect(r2.rev).toBe(second.rev);
    // The abandoned part reports no status: the key stays "computing" on the
    // main side until the newer request's status arrives.
    expect(r1.status.filter((st) => st.key === display)).toEqual([
      { key: display, rev: fields[0].rev, status: "ready" },
    ]);
    expect(r2.status).toEqual([{ key: display, rev: second.rev, status: "ready" }]);
    expect(r2.changes.find((c) => c.id === 1)?.colors).toBe("all");

    // An abandoned field still works: it is built when first needed.
    deliver([{ kind: "fetchFieldValues", rev: rev(), qid: 1, field: 4 }]);
    await until(4);
    expect(builds).toEqual([1, 3, 4]);
    const reply = results[3].replies[0];
    expect(reply.ok).toBe(true);
    const ref = new Scene();
    try {
      const a = ref.addMesh(s.positions, s.indices);
      const b = ref.addMesh(t.positions, t.indices);
      const f = ref.addDistanceField({ name: "f4", on: a, to: b, maxDistance: Infinity });
      expect(Buffer.from((reply.value as Float32Array).buffer).equals(Buffer.from(ref.fieldValues(f).slice().buffer))).toBe(
        true,
      );
    } finally {
      ref.free();
    }

    // Removing an abandoned field never builds it.
    deliver([{ kind: "field", rev: rev(), field: 2, op: "remove" }]);
    await until(5);
    expect(builds).toEqual([1, 3, 4]);
    expect(results[4].status).toEqual([{ key: "field:2", rev: revision, status: "ready" }]);
  });
});

/** The engine in-process, with each batch's result awaited. */
function engine(options: { onChunk?: (chunk: { kind: string; id?: ClientId }) => void } = {}) {
  const results: Result[] = [];
  const waiting: (() => void)[] = [];
  let deliver!: (batch: Request[]) => void;
  runEngine(
    {
      post: (msg) => {
        results.push(msg);
        for (const f of waiting.splice(0)) {
          f();
        }
      },
      onMessage: (cb) => {
        deliver = cb;
      },
    },
    { debug: true, ...options },
  );
  return async (batch: Request[]): Promise<Result> => {
    const n = results.length + 1;
    deliver(batch);
    while (results.length < n) {
      await new Promise<void>((resolve) => waiting.push(resolve));
    }
    return results[n - 1];
  };
}

describe("yields in a Worker", () => {
  /**
   * Five field builds for a display and its colouring, with a batch turning
   * the colouring off sent at once; returns how many of the first batch's
   * display requests ran.
   */
  async function race(yieldMs: number): Promise<number> {
    const worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), { workerData: { yieldMs } });
    try {
      const results: Result[] = [];
      const two = new Promise<void>((resolve) => {
        worker.on("message", (r: Result) => {
          results.push(r);
          if (results.length === 2) {
            resolve();
          }
        });
      });
      const a = addMeshReq(1, sphere([0, 0, 0], 1, 24, 48));
      const b = addMeshReq(2, sphere([2.2, 0, 0], 1, 24, 48));
      const display = "display:1";
      const batch: Request[] = [a, b];
      for (const field of [1, 2, 3, 4, 5]) {
        batch.push({
          kind: "field",
          rev: rev(),
          field,
          op: "add",
          desc: { name: `f${field}`, on: 1, to: 2, maxDistance: Infinity },
          key: display,
        });
      }
      batch.push({ kind: "coloring", rev: rev(), id: 1, rule: { field: 5, map: "turbo", range: [0, 3] }, key: display });
      const off: Request = { kind: "coloring", rev: rev(), id: 1, rule: null, key: display };
      const buffers = [a, b].flatMap((r) => [(r as { positions: ArrayBuffer }).positions, (r as { indices: ArrayBuffer }).indices]);
      // Both batches are sent at once; the second is received only when the
      // engine yields to the message loop, and then cancels the rest.
      worker.postMessage(batch, buffers);
      worker.postMessage([off]);
      await two;
      expect(results[1].status).toEqual([{ key: display, rev: off.rev, status: "ready" }]);
      return results[0].status.filter((st) => st.key === display).length;
    } finally {
      await worker.terminate();
    }
  }

  it("later_batch_arrives_between_chunks", async () => {
    // Yielding after every chunk: the second batch cancels the rest.
    expect(await race(0)).toBeLessThan(6);
    // Within the time budget the engine does not yield: everything runs.
    expect(await race(1e9)).toBe(6);
  });
});

describe("poisoning while batches wait", () => {
  it("poison_mid_drain_posts_one_result_per_batch", async () => {
    // The harness makes `analyze` trap (test build of the wasm only).
    const worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), {
      workerData: { testPanic: true },
    });
    try {
      const results: Result[] = [];
      const three = new Promise<void>((resolve) => {
        worker.on("message", (r: Result) => {
          results.push(r);
          if (results.length === 3) {
            resolve();
          }
        });
      });
      const add = addMeshReq(1, sphere([0, 0, 0]));
      // Sent together: the later batches arrive while the first one runs
      // (the engine yields between its requests), then the trap comes.
      worker.postMessage(
        [
          add,
          { kind: "edit", rev: rev(), id: 1, payload: [{ op: "addVertices", positions: [0, 0, 3] }] },
          { kind: "query", rev: rev(), qid: 1, name: "analyze", args: { mesh: 1 } },
        ] satisfies Request[],
        [(add as { positions: ArrayBuffer }).positions, (add as { indices: ArrayBuffer }).indices],
      );
      const query: Request = {
        kind: "query",
        rev: rev(),
        qid: 2,
        name: "raycast",
        args: { origin: [0, 0, 5], direction: [0, 0, -1] },
      };
      worker.postMessage([query]);
      const compact: Request = { kind: "compact", rev: rev(), id: 1 };
      worker.postMessage([compact]);
      await three;
      // One result per batch, every one poisoned.
      expect(results.map((r) => r.poisoned)).toEqual([true, true, true]);
      expect(results[0].replies).toMatchObject([{ qid: 1, ok: false, error: { code: "INSTANCE_POISONED" } }]);
      expect(results[1].rev).toBe(query.rev);
      expect(results[1].replies).toMatchObject([{ qid: 2, ok: false, error: { code: "INSTANCE_POISONED" } }]);
      expect(results[2].rev).toBe(compact.rev);
      expect(results[2].status).toMatchObject([{ key: "edit:1", status: "error", error: { code: "INSTANCE_POISONED" } }]);
    } finally {
      await worker.terminate();
    }
  });
});

describe("chunked repair", () => {
  it("repair_list_runs_in_steps", async () => {
    const ops: string[] = [];
    const send = engine({ onChunk: (c) => ops.push(c.kind) });
    const s = sphere([0, 0, 0]);
    const edits = [
      { op: "removeTriangles" as const, triangles: [0, 1] },
      { op: "addVertices" as const, positions: [0, 0, 5] },
    ];
    await send([addMeshReq(1, s), { kind: "edit", rev: rev(), id: 1, payload: edits }]);
    const ref = new Scene();
    try {
      const m = ref.addMesh(s.positions, s.indices);
      ref.edit(m, edits);
      const dump = async () => {
        const r = await send([{ kind: "query", rev: rev(), qid: 9, name: "debugDump", args: { mesh: 1 } }]);
        return r.replies[0].value as { positions: Float32Array; indices: Uint32Array };
      };
      const same = async () => {
        const d = await dump();
        const v = ref.views(m);
        expect(Buffer.from(d.positions.buffer).equals(Buffer.from(v.positions().slice().buffer))).toBe(true);
        expect(Buffer.from(d.indices.buffer).equals(Buffer.from(v.indices().slice().buffer))).toBe(true);
      };

      // A failing list changes nothing, as the list call does.
      ops.length = 0;
      const failing = await send([
        { kind: "repair", rev: rev(), id: 1, qid: 1, payload: [{ op: "removeDegenerate", removeUnusedVertices: true }, { op: "fillHoles" }] },
      ]);
      expect(() => ref.repair(m, [{ op: "removeDegenerate", removeUnusedVertices: true }, { op: "fillHoles" }])).toThrow();
      expect(failing.replies[0]).toMatchObject({ ok: false, error: { code: "INVALID_ARGUMENT" } });
      expect(ops.filter((k) => k === "repairOp")).toHaveLength(1);
      expect(failing.history).toEqual({ canUndo: ref.canUndo(), canRedo: ref.canRedo() });
      await same();

      // One operation per chunk; the same reports, and one undo step.
      ops.length = 0;
      const list = [{ op: "removeDegenerate", removeUnusedVertices: true }, { op: "fillHoles", maxEdges: 16 }] as const;
      const r = await send([{ kind: "repair", rev: rev(), id: 1, qid: 2, payload: list }]);
      expect(ops.filter((k) => k === "repairOp")).toHaveLength(2);
      expect(r.replies[0]).toMatchObject({ ok: true });
      expect(r.replies[0].value).toEqual(ref.repair(m, list));
      await same();
      const undone = await send([{ kind: "undo", rev: rev() }]);
      ref.undo();
      expect(undone.history).toEqual({ canUndo: ref.canUndo(), canRedo: ref.canRedo() });
      await same();
    } finally {
      ref.free();
    }
  });
});

// ---- WorkerScene against the harness ----

/**
 * The harness Worker as a `WorkerLike`: records every batch, and holds
 * results back while `hold` is set.
 */
class NodeWorker implements WorkerLike {
  readonly worker: Worker;
  readonly sent: Request[][] = [];
  hold = false;
  terminated = false;
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
      this.inject(data);
    });
    this.worker.on("error", (e) => {
      for (const cb of this.#error) {
        cb(e);
      }
    });
  }

  /** Delivers `data` to the listeners as if the Worker posted it. */
  inject(data: unknown): void {
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
    this.terminated = true;
    void this.worker.terminate();
  }

  release(n = this.held.length): void {
    for (const m of this.held.splice(0, n)) {
      this.inject(m);
    }
  }

  async waitHeld(n: number): Promise<void> {
    while (this.held.length < n) {
      await new Promise<void>((resolve) => this.#arrived.push(resolve));
    }
  }
}

const opened: WorkerScene[] = [];
afterEach(() => {
  for (const s of opened.splice(0)) {
    s.terminate();
  }
});

/** A scene whose every Worker (the first and each restart's) is recorded. */
async function start(
  workerData?: { testPanic?: boolean },
  failCreate?: (n: number) => boolean,
): Promise<{ scene: WorkerScene; workers: NodeWorker[] }> {
  const workers: NodeWorker[] = [];
  let created = 0;
  const scene = await createWorkerScene({
    createWorker: () => {
      if (failCreate?.(++created)) {
        throw new Error("no Worker today");
      }
      const w = new NodeWorker(workerData);
      workers.push(w);
      return w;
    },
  });
  opened.push(scene);
  return { scene, workers };
}

async function settle(scene: WorkerScene): Promise<void> {
  const barrier = scene.raycast([0, 0, 1e6], [0, 0, 1]);
  scene.commitFrame();
  await barrier;
}

function bytes(v: ArrayBufferView | null): Buffer | null {
  return v === null ? null : Buffer.from(v.buffer, v.byteOffset, v.byteLength);
}

async function expectMirrored(scene: WorkerScene, mesh: MeshHandle): Promise<void> {
  const pending = debugDump(scene, mesh);
  scene.commitFrame();
  const dump = await pending;
  const v = scene.views(mesh);
  for (const name of ["positions", "normals", "colors", "indices"] as const) {
    const want = bytes(dump[name]);
    const got = bytes(v[name]());
    if (want === null || got === null) {
      expect(got, name).toBe(want);
    } else {
      expect(got.equals(want), `${name} bytes`).toBe(true);
    }
  }
  expect(Array.from(scene.matrix4(mesh))).toEqual(dump.matrix4);
  expect(scene.placement(mesh)).toEqual(dump.placement);
}

/**
 * The mesh as live triangles: per triangle corner, its local position,
 * normal and colour, in slot order (independent of slot numbering, so a
 * compacted rebuild compares equal).
 */
function soup(scene: WorkerScene, mesh: MeshHandle) {
  const v = scene.views(mesh);
  const p = v.positions();
  const n = v.normals();
  const c = v.colors();
  const idx = v.indices();
  const positions: number[] = [];
  const normals: number[] = [];
  const colors: number[] = [];
  for (let t = 0; t < idx.length; t += 3) {
    if (idx[t] === 0 && idx[t + 1] === 0 && idx[t + 2] === 0) {
      continue;
    }
    for (let k = 0; k < 3; k++) {
      const i = idx[t + k];
      positions.push(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
      if (n !== null) {
        normals.push(n[3 * i], n[3 * i + 1], n[3 * i + 2]);
      }
      if (c !== null) {
        colors.push(c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]);
      }
    }
  }
  return { positions, normals: n === null ? null : normals, colors: c === null ? null : colors };
}

function expectSameSoup(got: ReturnType<typeof soup>, want: ReturnType<typeof soup>): void {
  expect(got.positions.length).toBe(want.positions.length);
  expect(got.positions).toEqual(want.positions);
  expect(got.colors).toEqual(want.colors);
  expect(got.normals === null).toBe(want.normals === null);
  if (got.normals !== null && want.normals !== null) {
    for (let k = 0; k < want.normals.length; k++) {
      expect(Math.abs(got.normals[k] - want.normals[k])).toBeLessThan(1e-5);
    }
  }
}

describe("WorkerScene restart", () => {
  it("stale_status_ignored", async () => {
    const { scene, workers } = await start();
    const s = sphere([0, 0, 0]);
    const t = sphere([2.5, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    const b = scene.addMesh(t.positions, t.indices);
    const field = scene.addDistanceField({ name: "d", on: a, to: b, maxDistance: 3 });
    await settle(scene);
    const key = `display:${scene.idOf(a)}`;

    // A colouring whose result is held back...
    const old = workers[0];
    old.hold = true;
    scene.setColoring(a, { field, map: "turbo", range: [0, 2] });
    scene.commitFrame();
    const r1 = scene.status(key).revision;
    await old.waitHeld(1);

    // ...then a restart, which replays the colouring with a newer revision.
    const restarted = scene.restart();
    expect(old.terminated).toBe(true);
    const replay = scene.status(key);
    expect(replay.status).toBe("computing");
    expect(replay.revision).toBeGreaterThan(r1);

    // The old Worker's late result is ignored entirely.
    const colors = scene.views(a).colors();
    old.release();
    expect(scene.status(key)).toEqual(replay);
    expect(scene.views(a).colors()).toBe(colors);
    await restarted;
    expect(scene.status(key)).toEqual({ status: "ready", revision: replay.revision });

    // A status older than the one applied for its key is ignored.
    workers[1].inject({
      rev: r1,
      changes: [],
      placements: [],
      sections: [],
      history: { canUndo: false, canRedo: false },
      status: [{ key, rev: r1, status: "error", error: { code: "INVALID_ARGUMENT", message: "stale" } }],
      replies: [],
    } satisfies Result);
    expect(scene.status(key)).toEqual({ status: "ready", revision: replay.revision });
    scene.commitFrame();
    expect(scene.views(a).colors()).not.toBeNull();
    await expectMirrored(scene, a);
  });

  it("poison_then_restart_rebuilds", async () => {
    // The harness makes `analyze` trap (test build of the wasm only).
    const { scene, workers } = await start({ testPanic: true });
    const s = sphere([0, 0, 0]);
    const t = sphere([2.5, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    const b = scene.addMesh(t.positions, t.indices, { normals: false });
    // Dead slots, moved vertices and added ones: the rebuild compacts them.
    void scene.edit(a, [{ op: "removeTriangles", triangles: [0, 5, 17] }]);
    void scene.edit(a, [{ op: "setPositions", vertices: [3, 40], positions: [0.1, 0.2, 1.3, -0.4, 0.5, 0.2] }]);
    scene.setPlacement(b, { rotation: IDENTITY, translation: [0.5, 0, 0] });
    const field = scene.addDistanceField({ name: "d", on: a, to: b, maxDistance: 3 });
    scene.setColoring(a, { field, map: "turbo", range: [0, 2] });
    const section = scene.addSection({ plane: { origin: [0, 0, 0.2], normal: [0, 0, 1] }, meshes: [a, b], fill: true });
    await settle(scene);
    scene.commitFrame();
    const before = { a: soup(scene, a), b: soup(scene, b) };
    const areaA = scene.section(section, a).area;
    const areaB = scene.section(section, b).area;
    expect(areaA).not.toBe(0);
    const slotsBefore = scene.views(a).indices().length / 3;

    // A mesh added in the batch that traps (sent: its buffers are gone)...
    const sent = scene.addMesh(sphere([0, 3, 0]).positions, sphere([0, 3, 0]).indices);
    const analysis = scene.analyze(a);
    scene.commitFrame();
    await expect(analysis).rejects.toMatchObject({ code: "INSTANCE_POISONED" });
    // ...one added after it (failed here, never sent), and a drag held
    // after the poisoning.
    const later = scene.addMesh(sphere([0, -3, 0]).positions, sphere([0, -3, 0]).indices);
    const drag = scene.drag(a);
    drag.move({ rotation: IDENTITY, translation: [4, 0, 0] });
    scene.commitFrame();
    expect(scene.status(`mesh:${scene.idOf(later)}`)).toMatchObject({ status: "error" });
    expect(scene.placement(a).translation).toEqual([4, 0, 0]);

    const restarted = scene.restart();
    expect(workers).toHaveLength(2);
    // The holds are gone, history is cleared, every replayed key computes.
    expect(drag.active).toBe(false);
    expect(scene.placement(a).translation).toEqual([0, 0, 0]);
    expect(scene.canUndo).toBe(false);
    expect(scene.canRedo).toBe(false);
    for (const key of [
      `mesh:${scene.idOf(a)}`,
      `mesh:${scene.idOf(b)}`,
      `mesh:${scene.idOf(later)}`,
      `display:${scene.idOf(a)}`,
      `section:${scene.idOf(section)}`,
    ]) {
      expect(scene.status(key).status, key).toBe("computing");
    }
    await restarted;
    scene.commitFrame();

    // Rebuilt from the mirrors: same live geometry, colours and sections,
    // compacted.
    expectSameSoup(soup(scene, a), before.a);
    expectSameSoup(soup(scene, b), before.b);
    expect(scene.views(a).indices().length / 3).toBe(slotsBefore - 3);
    expect(scene.placement(b).translation).toEqual([0.5, 0, 0]);
    expect(scene.section(section, a).area).toBeCloseTo(areaA, 9);
    expect(scene.section(section, b).area).toBeCloseTo(areaB, 9);
    expect(scene.section(section, a).fill).toBeDefined();
    await expectMirrored(scene, a);
    await expectMirrored(scene, b);
    for (const key of [`mesh:${scene.idOf(a)}`, `display:${scene.idOf(a)}`, `section:${scene.idOf(section)}`]) {
      expect(scene.status(key).status, key).toBe("ready");
    }
    expect(scene.canUndo).toBe(false);

    // The never-sent add was replayed; the sent one surfaces as an error.
    expect(scene.status(`mesh:${scene.idOf(later)}`).status).toBe("ready");
    expect(scene.views(later).indices().length).toBeGreaterThan(0);
    await expectMirrored(scene, later);
    expect(scene.status(`mesh:${scene.idOf(sent)}`)).toMatchObject({ status: "error" });
    await expect(scene.intersects(sent, a)).rejects.toMatchObject({ code: "UNKNOWN_MESH" });

    // The handles keep working on the new engine.
    expect(await scene.intersects(a, later)).toBe(false);
    await scene.edit(a, [{ op: "removeTriangles", triangles: [0] }]);
    await settle(scene);
    expect(scene.canUndo).toBe(true);
    await expectMirrored(scene, a);
  });

  it("restart_preserves_handles_and_display", async () => {
    const { scene, workers } = await start();
    const s = sphere([0, 0, 0]);
    const t = sphere([2.5, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    const b = scene.addMesh(t.positions, t.indices);
    const gone = scene.addMesh(sphere([0, 4, 0]).positions, sphere([0, 4, 0]).indices);
    const field = scene.addDistanceField({ name: "d", on: a, to: b, maxDistance: 3 });
    scene.setColoring(a, { field, map: "viridis", range: [0, 2], above: "clamp" });
    const dropped = scene.addDistanceField({ name: "e", on: b, to: a, maxDistance: 3 });
    scene.setColoring(b, { field: dropped, map: "turbo", range: [0, 2] });
    const section = scene.addSection({ plane: { origin: [0, 0, 0], normal: [0, 0, 1] }, meshes: [a, b, gone], fill: true });
    const removedSection = scene.addSection({ plane: { origin: [0, 0, 0.5], normal: [0, 0, 1] }, meshes: [a] });
    scene.setPlacement(a, { rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2], translation: [0, 0, 0.25] });
    await settle(scene);
    scene.removeField(dropped);
    scene.removeSection(removedSection);
    scene.removeMesh(gone);
    await settle(scene);
    scene.commitFrame();
    expect(scene.canUndo).toBe(true);
    const before = { a: soup(scene, a), b: soup(scene, b) };
    expect(before.a.colors).not.toBeNull();
    expect(before.b.colors).toBeNull();
    const placement = scene.placement(a);
    const matrix = Array.from(scene.matrix4(a));
    const area = scene.section(section, a).area;

    // Requested this frame, not sent yet: a new mesh and a placement.
    const fresh = scene.addMesh(sphere([0, -4, 0]).positions, sphere([0, -4, 0]).indices);
    scene.setPlacement(fresh, { rotation: IDENTITY, translation: [0.75, 0, 0] });
    // Added and removed before any result: its removal is reported here.
    const ghost = scene.addMesh(sphere([5, 0, 0]).positions, sphere([5, 0, 0]).indices);
    scene.removeMesh(ghost);

    const restarted = scene.restart();
    expect(scene.status(`display:${scene.idOf(a)}`).status).toBe("computing");
    expect(scene.canUndo).toBe(false);
    // Until the replay's results arrive, the mirrors stay as they were.
    expect(soup(scene, a)).toEqual(before.a);
    expect(scene.section(section, a).area).toBe(area);
    await restarted;
    const changes = scene.commitFrame();
    expect(changes.find((c) => c.mesh === a)?.added).toBe(true);
    expect(changes.find((c) => c.mesh === ghost)?.removed).toBe(true);
    expect(() => scene.idOf(ghost)).toThrow(/removed/);

    // The replay re-created what is live, and nothing removed.
    const replay = workers[1].sent[0];
    const fieldIds = replay.flatMap((r) => (r.kind === "field" ? [r.field] : []));
    expect(fieldIds).toEqual([scene.idOf(field)]);
    const sectionIds = replay.flatMap((r) => (r.kind === "section" ? [r.section] : []));
    expect(sectionIds).toEqual([scene.idOf(section)]);
    expect(replay.filter((r) => r.kind === "addMesh")).toHaveLength(3);

    // Same handles, same display.
    expectSameSoup(soup(scene, a), before.a);
    expectSameSoup(soup(scene, b), before.b);
    expect(scene.placement(a)).toEqual(placement);
    expect(Array.from(scene.matrix4(a))).toEqual(matrix);
    expect(scene.placement(fresh).translation).toEqual([0.75, 0, 0]);
    expect(scene.section(section, a).area).toBeCloseTo(area, 9);
    expect(() => scene.sectionPlane(removedSection)).toThrow(/removed/);
    await expect(scene.fetchFieldValues(dropped)).rejects.toMatchObject({ code: "UNKNOWN_FIELD" });
    expect(await scene.fetchFieldValues(field)).toHaveLength(scene.views(a).positions().length / 3);
    for (const m of [a, b, fresh]) {
      await expectMirrored(scene, m);
    }
    for (const key of [
      `display:${scene.idOf(a)}`,
      `placement:${scene.idOf(a)}`,
      `placement:${scene.idOf(fresh)}`,
      `mesh:${scene.idOf(fresh)}`,
      `section:${scene.idOf(section)}`,
    ]) {
      expect(scene.status(key).status, key).toBe("ready");
    }
    expect(scene.canUndo).toBe(false);
    expect(scene.canRedo).toBe(false);

    // A drag and an undo work on the new engine.
    const d = scene.drag(b);
    d.move({ rotation: IDENTITY, translation: [1, 0, 0] });
    d.end();
    await settle(scene);
    expect(scene.canUndo).toBe(true);
    scene.undo();
    await settle(scene);
    expect(scene.placement(b).translation).toEqual([0, 0, 0]);
    await expectMirrored(scene, b);
  });

  it("undo_drops_ended_drag_hold", async () => {
    const { scene, workers } = await start();
    const s = sphere([0, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    await settle(scene);
    const w = workers[0];
    w.hold = true;
    const d = scene.drag(a);
    d.move({ rotation: IDENTITY, translation: [5, 0, 0] });
    d.end();
    scene.commitFrame();
    expect(scene.placement(a).translation).toEqual([5, 0, 0]);
    // The undo will undo the drag: the held drag pose no longer applies, so
    // the engine's pose shows and a new drag starts from it.
    scene.undo();
    expect(scene.placement(a).translation).toEqual([0, 0, 0]);
    const d2 = scene.drag(a);
    d2.cancel();
    expect(scene.placement(a).translation).toEqual([0, 0, 0]);
    scene.commitFrame();
    await w.waitHeld(2);
    w.hold = false;
    w.release();
    await settle(scene);
    expect(scene.placement(a).translation).toEqual([0, 0, 0]);
    await expectMirrored(scene, a);
  });

  it("pending_declarative_requests_settle_on_restart", async () => {
    const { scene, workers } = await start();
    const s = sphere([0, 0, 0]);
    const t = sphere([2.5, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    const b = scene.addMesh(t.positions, t.indices);
    const field = scene.addDistanceField({ name: "d", on: a, to: b, maxDistance: 3 });
    scene.setColoring(a, { field, map: "turbo", range: [0, 2] });
    scene.setPlacement(a, { rotation: IDENTITY, translation: [0, 0, 0.5] });
    const section = scene.addSection({ plane: { origin: [0, 0, 0], normal: [0, 0, 1] }, meshes: [a, b] });
    await settle(scene);
    const id = (h: Parameters<WorkerScene["idOf"]>[0]) => scene.idOf(h);

    // In flight when the restart comes: a section plane, a placement back to
    // the identity pose, colouring off, and an edit.
    workers[0].hold = true;
    scene.setSectionPlane(section, { origin: [0, 0, 0.3], normal: [0, 0, 1] });
    scene.setPlacement(a, { rotation: IDENTITY, translation: [0, 0, 0] });
    scene.setColoring(a, null);
    const edit = scene.edit(a, [{ op: "removeTriangles", triangles: [0] }]);
    scene.commitFrame();
    await workers[0].waitHeld(1);
    // Queued, not sent: the section's fill.
    scene.setSectionFill(section, true);

    const keys = [
      `section:${id(section)}:plane`,
      `section:${id(section)}:fill`,
      `placement:${id(a)}`,
      `display:${id(a)}`,
    ];
    const restarted = scene.restart();
    for (const key of keys) {
      expect(scene.status(key).status, key).toBe("computing");
    }
    await expect(edit).rejects.toMatchObject({ code: "INSTANCE_POISONED" });
    expect(scene.status(`edit:${id(a)}`)).toMatchObject({ status: "error", error: { code: "INSTANCE_POISONED" } });
    await restarted;
    scene.commitFrame();
    for (const key of keys) {
      expect(scene.status(key).status, key).toBe("ready");
    }
    // The replay applied the newest values.
    expect(scene.placement(a).translation).toEqual([0, 0, 0]);
    expect(scene.views(a).colors()).toBeNull();
    const cut = scene.section(section, a);
    expect(cut.fill).toBeDefined();
    expect(Math.abs(cut.area)).toBeLessThan(Math.PI * (1 - 0.3 * 0.3));
    expect(Math.abs(cut.area)).toBeGreaterThan(0);
    await expectMirrored(scene, a);
  });

  it("poisoned_restart_settles_removals", async () => {
    const { scene } = await start({ testPanic: true });
    const s = sphere([0, 0, 0]);
    const t = sphere([2.5, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    const b = scene.addMesh(t.positions, t.indices);
    const c = scene.addMesh(sphere([0, 4, 0]).positions, sphere([0, 4, 0]).indices);
    const field = scene.addDistanceField({ name: "d", on: a, to: b, maxDistance: 3 });
    const section = scene.addSection({ plane: { origin: [0, 0, 0], normal: [0, 0, 1] }, meshes: [a, b] });
    await settle(scene);
    await expect(scene.analyze(a)).rejects.toMatchObject({ code: "INSTANCE_POISONED" });
    const keys = {
      field: `field:${scene.idOf(field)}`,
      mesh: `mesh:${scene.idOf(c)}`,
      plane: `section:${scene.idOf(section)}:plane`,
      placement: `placement:${scene.idOf(a)}`,
    };
    // After the poisoning: removals, a section plane and a placement fail.
    scene.removeField(field);
    scene.removeMesh(c);
    scene.setSectionPlane(section, { origin: [0, 0, 0.3], normal: [0, 0, 1] });
    scene.setPlacement(a, { rotation: IDENTITY, translation: [1, 0, 0] });
    scene.commitFrame();
    for (const key of Object.values(keys)) {
      expect(scene.status(key).status, key).toBe("error");
    }
    await scene.restart();
    const changes = scene.commitFrame();
    // The removals happened; the plane is replayed as last requested; the
    // placement replays the engine's pose, so its request stays failed.
    expect(scene.status(keys.field).status).toBe("ready");
    expect(scene.status(keys.mesh).status).toBe("ready");
    expect(scene.status(keys.plane).status).toBe("ready");
    expect(scene.status(keys.placement).status).toBe("error");
    expect(scene.placement(a).translation).toEqual([0, 0, 0]);
    expect(changes.find((ch) => ch.mesh === c)?.removed).toBe(true);
    expect(scene.sectionPlane(section).origin).toEqual([0, 0, 0.3]);
    await expectMirrored(scene, a);
  });

  it("empty_mesh_restart", async () => {
    // Poisoned: an empty add sent in the trapping batch is lost; one added
    // after the poisoning (never sent) is replayed.
    {
      const { scene } = await start({ testPanic: true });
      const s = sphere([0, 0, 0]);
      const a = scene.addMesh(s.positions, s.indices);
      await settle(scene);
      const sent = scene.addMesh([], []);
      await expect(scene.analyze(a)).rejects.toMatchObject({ code: "INSTANCE_POISONED" });
      const later = scene.addMesh([], []);
      scene.commitFrame();
      await scene.restart();
      scene.commitFrame();
      expect(scene.status(`mesh:${scene.idOf(sent)}`).status).toBe("error");
      expect(scene.status(`mesh:${scene.idOf(later)}`).status).toBe("ready");
      await expect(scene.meshDistance(sent, a)).rejects.toMatchObject({ code: "UNKNOWN_MESH" });
      await scene.meshDistance(later, a);
      await expectMirrored(scene, later);
    }
    // Healthy: an empty add in flight is lost; a queued one is replayed.
    {
      const { scene, workers } = await start();
      workers[0].hold = true;
      const sent = scene.addMesh([], []);
      scene.commitFrame();
      await workers[0].waitHeld(1);
      const queued = scene.addMesh([], []);
      await scene.restart();
      scene.commitFrame();
      expect(scene.status(`mesh:${scene.idOf(sent)}`)).toMatchObject({
        status: "error",
        error: { code: "INSTANCE_POISONED" },
      });
      expect(scene.status(`mesh:${scene.idOf(queued)}`).status).toBe("ready");
      await expectMirrored(scene, queued);
    }
  });

  it("restart_survives_a_failing_worker_factory", async () => {
    const { scene } = await start(undefined, (n) => n === 2);
    const s = sphere([0, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    await settle(scene);
    const before = soup(scene, a);
    let failed: Promise<void> | undefined;
    expect(() => {
      failed = scene.restart();
    }).not.toThrow();
    await expect(failed).rejects.toMatchObject({ code: "INSTANCE_POISONED" });
    expect(scene.status(`mesh:${scene.idOf(a)}`).status).toBe("error");
    // Poisoned until a restart works; then everything is back.
    await expect(scene.intersects(a, a)).rejects.toMatchObject({ code: "INSTANCE_POISONED" });
    await scene.restart();
    scene.commitFrame();
    expectSameSoup(soup(scene, a), before);
    expect(scene.status(`mesh:${scene.idOf(a)}`).status).toBe("ready");
    await expectMirrored(scene, a);
  });

  it("unsendable_batch_fails_alone", async () => {
    const { scene } = await start();
    const s = sphere([0, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    await settle(scene);
    // A function cannot be cloned: posting the batch throws.
    const bad = scene.raycast([0, 0, 5], [0, 0, -1], { bogus: () => 1 } as never);
    await expect(bad).rejects.toMatchObject({ code: "INVALID_ARGUMENT" });
    // Later results still match their batches.
    const hit = await scene.raycast([0, 0, 5], [0, 0, -1]);
    expect(hit?.mesh).toBe(a);
    scene.setPlacement(a, { rotation: IDENTITY, translation: [1, 0, 0] });
    await settle(scene);
    expect(scene.placement(a).translation).toEqual([1, 0, 0]);
    await expectMirrored(scene, a);
  });
});
