// Drag speed part A in the Worker: the engine defers field updates and steps
// the fields a batch leaves due (`stepPoints` vertices a step), abandoning a
// unit at a step boundary when a queued request makes it stale (spec §6.1)
// and letting every other request wait. The engine runs in-process (sources,
// same wasm) with its test-only `onChunk` hook, which delivers newer batches
// between steps; the status tests drive it through `WorkerScene`.
import { afterEach, describe, expect, it } from "vitest";
import { type MeshHandle, Scene, type Vec3 } from "../src/node.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";
import { debugDump } from "../src/worker/client.js";
import { runEngine, serveEngine } from "../src/worker/engine.js";
import type { ClientId, MeshChangeWire, Request, Result } from "../src/worker/protocol.js";

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
/** Few vertices a step, so a unit spans many steps. */
const STEP_POINTS = 16;
/** Wide: every vertex of the coloured terrain is in range. */
const MAX_DISTANCE = 8;

/** A wavy `n × n` terrain grid at height `z`, its corner at `(x, y)`. */
function terrain(x: number, y: number, z: number, n = 24, size = 4) {
  const positions: number[] = [];
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const px = x + (size * i) / (n - 1);
      const py = y + (size * j) / (n - 1);
      positions.push(px, py, z + 0.3 * Math.sin(1.7 * px) * Math.cos(1.3 * py));
    }
  }
  const indices: number[] = [];
  for (let j = 0; j + 1 < n; j++) {
    for (let i = 0; i + 1 < n; i++) {
      const a = j * n + i;
      indices.push(a, a + 1, a + n + 1, a, a + n + 1, a + n);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

const COLORED = terrain(0, 0, 0);
const TARGET = terrain(0.5, 0.3, 1.2);
const THIRD = terrain(10, 0, 0);

/** The `k`th pose of the coloured mesh. */
function pose(k: number): Vec3 {
  return [0.07 * k, -0.05 * k, 0.03 * k];
}

/** Whether a result changed the colours of mesh `id`. */
function recolors(r: Result, id: ClientId): boolean {
  const c: MeshChangeWire | undefined = r.changes.find((x) => x.id === id);
  return c !== undefined && (c.colors === "all" || c.colors.length > 0);
}

/**
 * A fresh synchronous scene's colours of the coloured mesh at translation
 * `t`, after `edit` (if any) on it.
 */
function referenceColors(t: Vec3, edit?: (s: Scene, mesh: MeshHandle) => void): Uint8Array {
  const s = new Scene();
  try {
    const a = s.addMesh(COLORED.positions, COLORED.indices);
    const b = s.addMesh(TARGET.positions, TARGET.indices);
    s.setPlacement(a, { rotation: IDENTITY, translation: t });
    edit?.(s, a);
    const f = s.addDistanceField({ name: "d", on: a, to: b, maxDistance: MAX_DISTANCE });
    s.setColoring(a, { field: f, map: "turbo", range: [0, MAX_DISTANCE] });
    return (s.views(a).colors() as Uint8Array).slice();
  } finally {
    s.free();
  }
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

function moveReq(k: number): Request {
  return {
    kind: "drag",
    rev: rev(),
    id: 1,
    phase: "move",
    rotation: IDENTITY,
    translation: pose(k),
    key: "placement:1",
  };
}

/**
 * The engine in-process with a small step. `chunks` logs every chunk kind;
 * `onStep` runs on each field step (`"fieldStep"` chunk).
 */
function engine() {
  const results: Result[] = [];
  const chunks: string[] = [];
  const waiting: (() => void)[] = [];
  let deliver!: (batch: Request[]) => void;
  const h = {
    results,
    chunks,
    onStep: undefined as (() => void) | undefined,
    /** Delivers `batch` on a microtask: between two chunks of the engine. */
    later(batch: Request[]) {
      queueMicrotask(() => deliver(batch));
    },
    send(batch: Request[]) {
      deliver(batch);
    },
    async until(n: number) {
      while (results.length < n) {
        await new Promise<void>((resolve) => waiting.push(resolve));
      }
    },
  };
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
    {
      debug: true,
      stepPoints: STEP_POINTS,
      onChunk: (chunk) => {
        chunks.push(chunk.kind);
        if (chunk.kind === "fieldStep") {
          h.onStep?.();
        }
      },
    },
  );
  return h;
}

/** Meshes 1 (coloured by field 1, to mesh 2), 2 and 3 (coloured by field 2, to mesh 2). */
async function setup(h: ReturnType<typeof engine>, third = false): Promise<void> {
  const batch: Request[] = [
    addMeshReq(1, COLORED),
    addMeshReq(2, TARGET),
    { kind: "field", rev: rev(), field: 1, op: "add", desc: { name: "d", on: 1, to: 2, maxDistance: MAX_DISTANCE } },
    { kind: "coloring", rev: rev(), id: 1, rule: { field: 1, map: "turbo", range: [0, MAX_DISTANCE] }, key: "display:1" },
  ];
  if (third) {
    batch.push(addMeshReq(3, THIRD), {
      kind: "field",
      rev: rev(),
      field: 2,
      op: "add",
      desc: { name: "e", on: 3, to: 2, maxDistance: MAX_DISTANCE },
    });
  }
  h.send(batch);
  await h.until(h.results.length + 1);
}

async function dumpColors(h: ReturnType<typeof engine>, mesh: ClientId): Promise<Uint8Array> {
  const n = h.results.length + 1;
  h.send([{ kind: "query", rev: rev(), qid: 1000 + n, name: "debugDump", args: { mesh } }]);
  await h.until(n);
  const reply = h.results[n - 1].replies[0];
  expect(reply.ok).toBe(true);
  return (reply.value as { colors: Uint8Array }).colors;
}

describe("stepped field updates in the Worker", () => {
  it("stale_pose_is_abandoned", async () => {
    const h = engine();
    await setup(h);
    h.send([{ kind: "drag", rev: rev(), id: 1, phase: "begin", key: "placement:1" }]);
    await h.until(2);
    const first = h.results.length;
    const chunksBefore = h.chunks.length;

    // Ten moves, each arriving after the previous one's first step.
    const MOVES = 10;
    let sent = 1;
    let fresh = true;
    h.onStep = () => {
      if (fresh && sent < MOVES) {
        fresh = false;
        sent++;
        const next = moveReq(sent);
        queueMicrotask(() => {
          fresh = true;
          h.send([next]);
        });
      }
    };
    h.send([moveReq(1)]);
    await h.until(first + MOVES);
    h.onStep = undefined;
    const moves = h.results.slice(first, first + MOVES);

    // Exactly one unit completed: the last pose's. No earlier pose's
    // colours were published.
    expect(moves.map((r) => recolors(r, 1))).toEqual([...Array(MOVES - 1).fill(false), true]);
    const chunks = h.chunks.slice(chunksBefore);
    expect(chunks.filter((c) => c === "fieldStepAbandon")).toHaveLength(MOVES - 1);
    // Every abandoned unit had begun: one step each before its abandon.
    expect(chunks.filter((c) => c === "fieldStep").length).toBeGreaterThan(MOVES);

    expect(await dumpColors(h, 1)).toEqual(referenceColors(pose(MOVES)));
  });

  it("unrelated_requests_wait", async () => {
    const h = engine();
    await setup(h, true);
    h.send([{ kind: "drag", rev: rev(), id: 1, phase: "begin", key: "placement:1" }]);
    await h.until(2);
    const chunksBefore = h.chunks.length;

    const raycast: Request = {
      kind: "query",
      rev: rev(),
      qid: 7,
      name: "raycast",
      args: { origin: [12, 2, 5], direction: [0, 0, -1] },
    };
    const coloring: Request = {
      kind: "coloring",
      rev: rev(),
      id: 3,
      rule: { field: 2, map: "viridis", range: [0, MAX_DISTANCE] },
      key: "display:3",
    };
    let once = true;
    h.onStep = () => {
      if (once) {
        once = false;
        h.later([raycast, coloring]);
      }
    };
    h.send([moveReq(1)]);
    await h.until(4);
    h.onStep = undefined;
    const [unit, waited] = h.results.slice(2);

    // The unit completed in its own batch's result, not abandoned.
    expect(recolors(unit, 1)).toBe(true);
    const chunks = h.chunks.slice(chunksBefore);
    expect(chunks).not.toContain("fieldStepAbandon");
    // The drag move, every step of its unit, then the waiting requests in
    // arrival order (and their batch's idle step).
    const steps = chunks.indexOf("query") - 1;
    expect(steps).toBeGreaterThan(2);
    expect(chunks).toEqual(["drag", ...Array(steps).fill("fieldStep"), "query", "coloring", "fieldStep"]);
    expect(waited.replies).toHaveLength(1);
    expect(waited.replies[0].qid).toBe(7);
    expect(waited.replies[0].ok).toBe(true);
    expect((waited.replies[0].value as { mesh: ClientId }).mesh).toBe(3);
    expect(waited.status).toEqual([{ key: "display:3", rev: coloring.rev, status: "ready" }]);
    expect(recolors(waited, 3)).toBe(true);

    expect(await dumpColors(h, 1)).toEqual(referenceColors(pose(1)));
  });

  it("edit_abandons_unit", async () => {
    const h = engine();
    await setup(h);
    const chunksBefore = h.chunks.length;
    const t = pose(3);
    const moved: Vec3 = [0.2, 0.1, 0.9];
    const edit: Request = {
      kind: "edit",
      rev: rev(),
      id: 1,
      payload: [{ op: "setPositions", vertices: [30], positions: moved }],
    };
    let once = true;
    h.onStep = () => {
      if (once) {
        once = false;
        h.later([edit]);
      }
    };
    h.send([{ kind: "setPlacement", rev: rev(), id: 1, rotation: IDENTITY, translation: t, key: "placement:1" }]);
    await h.until(3);
    h.onStep = undefined;
    const [placed, edited] = h.results.slice(1);

    expect(h.chunks.slice(chunksBefore).filter((c) => c === "fieldStepAbandon")).toHaveLength(1);
    expect(recolors(placed, 1)).toBe(false);
    expect(placed.placements.map((p) => p.id)).toEqual([1]);
    expect(recolors(edited, 1)).toBe(true);
    expect(edited.status).toEqual([{ key: "edit:1", rev: edit.rev, status: "ready" }]);

    const expected = referenceColors(t, (s, mesh) => {
      s.edit(mesh, [{ op: "setPositions", vertices: [30], positions: moved }]);
    });
    expect(await dumpColors(h, 1)).toEqual(expected);
  });
});

// ---- Through `WorkerScene` ----

/**
 * The engine in-process (sources) as a `WorkerLike`, with a small step.
 * Batches are delivered on a microtask (so one sent from `onStep` arrives
 * between two steps); results on a task, cloned, as a Worker's are.
 * `afterResult` runs once the scene has applied a result.
 */
class InProcessWorker implements WorkerLike {
  onStep: (() => void) | undefined;
  afterResult: (() => void) | undefined;
  readonly chunks: string[] = [];
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  #deliver: ((data: unknown) => void) | undefined;
  #terminated = false;

  constructor() {
    serveEngine(
      {
        post: (msg) => {
          const copy = structuredClone(msg);
          setTimeout(() => {
            if (this.#terminated) {
              return;
            }
            for (const cb of this.#message) {
              cb({ data: copy });
            }
            if (!("kind" in copy)) {
              this.afterResult?.();
            }
          });
        },
        onMessage: (cb) => {
          this.#deliver = cb;
        },
      },
      Promise.resolve(),
      {
        debug: true,
        stepPoints: STEP_POINTS,
        onChunk: (chunk) => {
          this.chunks.push(chunk.kind);
          if (chunk.kind === "fieldStep") {
            this.onStep?.();
          }
        },
      },
    );
  }

  postMessage(msg: unknown): void {
    const copy = structuredClone(msg);
    queueMicrotask(() => this.#deliver?.(copy));
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
});

/** Commits frames until `done()` holds. */
async function until(scene: WorkerScene, done: () => boolean): Promise<void> {
  for (let k = 0; k < 2000; k++) {
    scene.commitFrame();
    if (done()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 2));
  }
  throw new Error("timed out");
}

async function workerSetup() {
  const worker = new InProcessWorker();
  const scene = await createWorkerScene({ createWorker: () => worker });
  cleanup.push(() => scene.terminate());
  const a = scene.addMesh(COLORED.positions, COLORED.indices);
  const b = scene.addMesh(TARGET.positions, TARGET.indices);
  const field = scene.addDistanceField({ name: "d", on: a, to: b, maxDistance: MAX_DISTANCE });
  scene.setColoring(a, { field, map: "turbo", range: [0, MAX_DISTANCE] });
  const display = `display:${scene.idOf(a)}`;
  await until(scene, () => scene.status(display).status === "ready");
  return { worker, scene, a, field, display };
}

describe("stepped field updates through WorkerScene", () => {
  it("drag_end_completes_final_pose", async () => {
    const { worker, scene, a, field, display } = await workerSetup();
    const before = worker.chunks.length;
    const drag = scene.drag(a);
    drag.move({ rotation: IDENTITY, translation: pose(4) });
    // A new rule in the same frame: the display key computes with the pose.
    scene.setColoring(a, { field, map: "viridis", range: [0, MAX_DISTANCE] });
    let once = true;
    worker.onStep = () => {
      if (once) {
        once = false;
        // The drag ends while the final pose's unit is mid-step.
        drag.end();
        scene.commitFrame();
      }
    };
    scene.commitFrame();
    expect(scene.status(display).status).toBe("computing");
    await until(scene, () => scene.status(display).status === "ready" && scene.status(`placement:${scene.idOf(a)}`).status === "ready");
    worker.onStep = undefined;

    expect(once).toBe(false);
    expect(drag.active).toBe(false);
    const chunks = worker.chunks.slice(before);
    expect(chunks).not.toContain("fieldStepAbandon");
    expect(chunks.filter((c) => c === "fieldStep").length).toBeGreaterThan(2);
    const s = new Scene();
    let expected: Uint8Array;
    try {
      const ra = s.addMesh(COLORED.positions, COLORED.indices);
      const rb = s.addMesh(TARGET.positions, TARGET.indices);
      s.setPlacement(ra, { rotation: IDENTITY, translation: pose(4) });
      const f = s.addDistanceField({ name: "d", on: ra, to: rb, maxDistance: MAX_DISTANCE });
      s.setColoring(ra, { field: f, map: "viridis", range: [0, MAX_DISTANCE] });
      expected = (s.views(ra).colors() as Uint8Array).slice();
    } finally {
      s.free();
    }
    expect((scene.views(a).colors() as Uint8Array).slice()).toEqual(expected);
    expect((await debugDump(scene, a)).colors).toEqual(expected);
    expect(scene.status(display).status).toBe("ready");
  });

  it("statuses_stay_computing_while_abandoned", async () => {
    const { worker, scene, a, field, display } = await workerSetup();
    const before = worker.chunks.length;
    const drag = scene.drag(a);
    drag.move({ rotation: IDENTITY, translation: pose(2) });
    scene.setColoring(a, { field, map: "viridis", range: [0, MAX_DISTANCE] });
    let once = true;
    worker.onStep = () => {
      if (once) {
        once = false;
        // A newer pose arrives mid-unit: the unit is abandoned.
        drag.move({ rotation: IDENTITY, translation: pose(5) });
        scene.commitFrame();
      }
    };
    const seen: string[] = [];
    worker.afterResult = () => seen.push(scene.status(display).status);
    scene.commitFrame();
    await until(scene, () => seen.length >= 2);
    worker.onStep = undefined;
    worker.afterResult = undefined;

    expect(worker.chunks.slice(before).filter((c) => c === "fieldStepAbandon")).toHaveLength(1);
    // Between the abandon and the completing unit, the display computes.
    expect(seen).toEqual(["computing", "ready"]);
    drag.end();
    await until(scene, () => scene.status(`placement:${scene.idOf(a)}`).status === "ready");
    expect(scene.status(display).status).toBe("ready");
    const expected = referenceColors(pose(5));
    // `referenceColors` uses turbo: recolour with it for the comparison.
    scene.setColoring(a, { field, map: "turbo", range: [0, MAX_DISTANCE] });
    await until(scene, () => scene.status(display).status === "ready");
    expect((await debugDump(scene, a)).colors).toEqual(expected);
  });
});
