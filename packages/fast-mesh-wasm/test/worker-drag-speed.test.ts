// Drag speed part A in the Worker: the engine defers field updates and steps
// the fields a batch leaves due (`stepPoints` vertices a step), abandoning a
// unit at a step boundary when a queued request makes it stale (spec §6.1)
// and letting every other request wait. The engine runs in-process (sources,
// same wasm) with its test-only `onChunk` hook, which delivers newer batches
// between steps; the status tests drive it through `WorkerScene`.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import { type MeshHandle, Scene, type Vec3 } from "../src/node.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";
import { debugDump } from "../src/worker/client.js";
import { runEngine, serveEngine } from "../src/worker/engine.js";
import { WasmScene } from "../src/wasm/fast_mesh_wasm.js";
import type { ClientId, MeshChangeWire, Request, Result } from "../src/worker/protocol.js";

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
/** Few vertices a step, so a unit spans many steps. */
const STEP_POINTS = 16;
/**
 * A starvation guard no test reaches: drag moves always abandon (see
 * `starvation_guard_publishes_during_storm` for the guard).
 */
const LONG_GUARD = 60_000;
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
function referenceColors(
  t: Vec3,
  edit?: (s: Scene, mesh: MeshHandle) => void,
  target: Vec3 = [0, 0, 0],
): Uint8Array {
  const s = new Scene();
  try {
    const a = s.addMesh(COLORED.positions, COLORED.indices);
    const b = s.addMesh(TARGET.positions, TARGET.indices);
    s.setPlacement(a, { rotation: IDENTITY, translation: t });
    s.setPlacement(b, { rotation: IDENTITY, translation: target });
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
function engine(stepPoints = STEP_POINTS, staleGuardMs = LONG_GUARD) {
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
      stepPoints,
      staleGuardMs,
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
    { kind: "field", rev: rev(), field: 1, op: "add", desc: { name: "d", on: 1, to: [2], maxDistance: MAX_DISTANCE } },
    { kind: "coloring", rev: rev(), id: 1, rule: { field: 1, map: "turbo", range: [0, MAX_DISTANCE] }, key: "display:1" },
  ];
  if (third) {
    batch.push(addMeshReq(3, THIRD), {
      kind: "field",
      rev: rev(),
      field: 2,
      op: "add",
      desc: { name: "e", on: 3, to: [2], maxDistance: MAX_DISTANCE },
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

/**
 * Places the coloured mesh at `pose(3)` (a unit over meshes 1 and 2), sends
 * `stale` after its first step, and waits for both results. Returns them and
 * the chunks logged since.
 */
async function placeThen(h: ReturnType<typeof engine>, stale: Request[]) {
  const first = h.results.length;
  const chunksBefore = h.chunks.length;
  let once = true;
  h.onStep = () => {
    if (once) {
      once = false;
      h.later(stale);
    }
  };
  h.send([{ kind: "setPlacement", rev: rev(), id: 1, rotation: IDENTITY, translation: pose(3), key: "placement:1" }]);
  await h.until(first + 2);
  h.onStep = undefined;
  const [placed, after] = h.results.slice(first);
  return { placed, after, chunks: h.chunks.slice(chunksBefore) };
}

describe("stale-making requests mid-unit", () => {
  it("cancel_abandons_unit", async () => {
    const h = engine();
    await setup(h);
    h.send([{ kind: "drag", rev: rev(), id: 1, phase: "begin", key: "placement:1" }]);
    await h.until(2);
    const chunksBefore = h.chunks.length;
    let once = true;
    h.onStep = () => {
      if (once) {
        once = false;
        h.later([{ kind: "drag", rev: rev(), id: 1, phase: "cancel", key: "placement:1" }]);
      }
    };
    h.send([moveReq(3)]);
    await h.until(4);
    h.onStep = undefined;
    const [moved, cancelled] = h.results.slice(2);

    expect(h.chunks.slice(chunksBefore).filter((c) => c === "fieldStepAbandon")).toHaveLength(1);
    expect(recolors(moved, 1)).toBe(false);
    expect(cancelled.status).toEqual([{ key: "placement:1", rev: revision, status: "ready" }]);
    // The restored (start) pose.
    expect(await dumpColors(h, 1)).toEqual(referenceColors([0, 0, 0]));
  });

  it("queued_moves_do_not_reopen_units", async () => {
    const h = engine();
    await setup(h);
    h.send([{ kind: "drag", rev: rev(), id: 1, phase: "begin", key: "placement:1" }]);
    await h.until(2);
    const chunksBefore = h.chunks.length;
    let once = true;
    h.onStep = () => {
      if (once) {
        once = false;
        // Three move batches queued at once, mid-unit.
        const batches = [[moveReq(2)], [moveReq(3)], [moveReq(4)]];
        queueMicrotask(() => {
          for (const batch of batches) {
            h.send(batch);
          }
        });
      }
    };
    h.send([moveReq(1)]);
    await h.until(6);
    h.onStep = undefined;
    const moves = h.results.slice(2);

    expect(moves.map((r) => recolors(r, 1))).toEqual([false, false, false, true]);
    // The first move's unit steps once and is abandoned; the batches in
    // between (their moves superseded) open no unit; the last pose's unit
    // runs to completion.
    const chunks = h.chunks.slice(chunksBefore);
    const steps = chunks.length - 4;
    expect(steps).toBeGreaterThan(2);
    expect(chunks).toEqual(["drag", "fieldStep", "fieldStepAbandon", "drag", ...Array(steps).fill("fieldStep")]);
    expect(await dumpColors(h, 1)).toEqual(referenceColors(pose(4)));
  });

  it("undo_abandons_unit", async () => {
    const h = engine();
    await setup(h);
    const undo: Request = { kind: "undo", rev: rev() };
    const { placed, after, chunks } = await placeThen(h, [undo]);
    expect(chunks.filter((c) => c === "fieldStepAbandon")).toHaveLength(1);
    expect(recolors(placed, 1)).toBe(false);
    expect(after.status).toEqual([{ key: "history", rev: undo.rev, status: "ready" }]);
    // The placement undone.
    expect(await dumpColors(h, 1)).toEqual(referenceColors([0, 0, 0]));
  });

  it("remove_mesh_abandons_unit", async () => {
    const h = engine();
    await setup(h);
    const remove: Request = { kind: "removeMesh", rev: rev(), id: 2 };
    const { placed, after, chunks } = await placeThen(h, [remove]);
    expect(chunks.filter((c) => c === "fieldStepAbandon")).toHaveLength(1);
    expect(recolors(placed, 1)).toBe(false);
    expect(after.status).toEqual([{ key: "mesh:2", rev: remove.rev, status: "ready" }]);
    // The removal dropped the field and its colouring: nothing is due (one
    // idle step), and the coloured mesh has no colours left.
    expect(chunks.at(-1)).toBe("fieldStep");
    expect(chunks.slice(chunks.indexOf("fieldStepAbandon") + 1)).toEqual(["removeMesh", "fieldStep"]);
    expect(await dumpColors(h, 1)).toBeNull();
  });

  it("target_move_abandons_unit", async () => {
    const h = engine();
    await setup(h);
    const target: Vec3 = [0.2, 0.1, -0.15];
    const { placed, after, chunks } = await placeThen(h, [
      { kind: "setPlacement", rev: rev(), id: 2, rotation: IDENTITY, translation: target, key: "placement:2" },
    ]);
    expect(chunks.filter((c) => c === "fieldStepAbandon")).toHaveLength(1);
    expect(recolors(placed, 1)).toBe(false);
    expect(recolors(after, 1)).toBe(true);
    expect(await dumpColors(h, 1)).toEqual(referenceColors(pose(3), undefined, target));
  });

  it("step_error_ends_unit", async () => {
    const h = engine();
    await setup(h);
    // The third step fails as a field error would: the unit is dropped (its
    // fields stay due) and the step throws a coded error.
    const proto = WasmScene.prototype as unknown as {
      field_step(points: number): number;
      abandon_field_step(): void;
    };
    const original = proto.field_step;
    let calls = 0;
    proto.field_step = function (this: typeof proto, points: number) {
      if (++calls === 3) {
        this.abandon_field_step();
        throw new Error("INVALID_ARGUMENT: field step failed (test)");
      }
      return original.call(this, points);
    };
    let placed: Result;
    const chunksBefore = h.chunks.length;
    try {
      h.send([{ kind: "setPlacement", rev: rev(), id: 1, rotation: IDENTITY, translation: pose(3), key: "placement:1" }]);
      await h.until(2);
      placed = h.results[1];
    } finally {
      proto.field_step = original;
    }
    // Swallowed like a failed eager update: the batch completes once, with
    // no loop and no colours; its status is ready.
    expect(h.chunks.slice(chunksBefore)).toEqual(["setPlacement", "fieldStep", "fieldStep", "fieldStep"]);
    expect(recolors(placed, 1)).toBe(false);
    expect(placed.status).toEqual([{ key: "placement:1", rev: revision, status: "ready" }]);
    // The fields stayed due: the next read brings them up to date.
    expect(await dumpColors(h, 1)).toEqual(referenceColors(pose(3)));
  });

  it("step_points_clamped", async () => {
    const h = engine(0);
    await setup(h);
    const chunksBefore = h.chunks.length;
    h.send([{ kind: "setPlacement", rev: rev(), id: 1, rotation: IDENTITY, translation: pose(3), key: "placement:1" }]);
    await h.until(2);
    // One vertex a step: many steps, and the unit completes.
    expect(h.chunks.slice(chunksBefore).filter((c) => c === "fieldStep").length).toBeGreaterThan(STEP_POINTS);
    expect(recolors(h.results[1], 1)).toBe(true);
  });
});

/**
 * Mirrors mesh 1's colours and placement from results, as a client does:
 * `apply` returns the translation a recolouring result claims (the latest
 * placement of mesh 1 so far) or `undefined` when it did not recolour.
 */
function colorMirror(start: Uint8Array) {
  const colors = start.slice();
  let translation: Vec3 = [0, 0, 0];
  return {
    colors,
    apply(r: Result): Vec3 | undefined {
      const p = r.placements.find((x) => x.id === 1);
      if (p !== undefined) {
        translation = p.translation as Vec3;
      }
      const c = r.changes.find((x) => x.id === 1);
      if (c === undefined || !recolors(r, 1)) {
        return undefined;
      }
      const bytes = new Uint8Array(c.bytes.colors as ArrayBuffer);
      if (c.colors === "all") {
        colors.set(bytes);
      } else {
        let at = 0;
        for (const [first, count] of c.colors) {
          colors.set(bytes.subarray(at, at + 4 * count), 4 * first);
          at += 4 * count;
        }
      }
      return translation;
    },
  };
}

describe("starvation guard", () => {
  it("starvation_guard_publishes_during_storm", async () => {
    const GUARD = 15;
    const h = engine(STEP_POINTS, GUARD);
    await setup(h);
    const mirror = colorMirror(await dumpColors(h, 1));
    h.send([{ kind: "drag", rev: rev(), id: 1, phase: "begin", key: "placement:1" }]);
    const first = h.results.length + 1;
    await h.until(first);
    const chunksBefore = h.chunks.length;

    // A move on every step, in its own batch, for a while: a storm that
    // never lets a unit finish within the guard.
    let k = 1;
    let pending = false;
    const until = performance.now() + 250;
    h.onStep = () => {
      if (!pending && performance.now() < until) {
        pending = true;
        const next = moveReq(++k);
        queueMicrotask(() => {
          pending = false;
          h.send([next]);
        });
      }
    };
    h.send([moveReq(1)]);
    // The moves stop; the last pose completes.
    while (performance.now() < until || pending) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    h.onStep = undefined;
    await h.until(first + k);
    const results = h.results.slice(first);
    expect(results).toHaveLength(k);
    const chunks = h.chunks.slice(chunksBefore);

    // Units were abandoned and others published mid-storm, each exactly
    // one pose: the colours equal a fresh scene at the pose they claim.
    const published: Vec3[] = [];
    for (const r of results) {
      const t = mirror.apply(r);
      if (t !== undefined) {
        published.push(t);
        expect(mirror.colors).toEqual(referenceColors(t));
      }
    }
    expect(chunks.filter((c) => c === "fieldStepAbandon").length).toBeGreaterThan(0);
    expect(published.length).toBeGreaterThan(1);
    // The final pose completes.
    expect(recolors(results[results.length - 1], 1)).toBe(true);
    expect(published[published.length - 1]).toEqual(pose(k));
    expect(await dumpColors(h, 1)).toEqual(referenceColors(pose(k)));
  });

  it("moves_within_guard_abandon", async () => {
    // A guard of 10 s, just after a published unit: every move abandons.
    const h = engine(STEP_POINTS, 10_000);
    await setup(h);
    h.send([{ kind: "setPlacement", rev: rev(), id: 1, rotation: IDENTITY, translation: [0.5, 0.4, 0], key: "placement:1" }]);
    await h.until(2);
    expect(recolors(h.results[1], 1)).toBe(true);
    h.send([{ kind: "drag", rev: rev(), id: 1, phase: "begin", key: "placement:1" }]);
    await h.until(3);
    const chunksBefore = h.chunks.length;
    const MOVES = 5;
    let sent = 1;
    let fresh = true;
    h.onStep = () => {
      if (fresh && sent < MOVES) {
        fresh = false;
        const next = moveReq(++sent);
        queueMicrotask(() => {
          fresh = true;
          h.send([next]);
        });
      }
    };
    h.send([moveReq(1)]);
    await h.until(3 + MOVES);
    h.onStep = undefined;
    expect(h.results.slice(3).map((r) => recolors(r, 1))).toEqual([false, false, false, false, true]);
    expect(h.chunks.slice(chunksBefore).filter((c) => c === "fieldStepAbandon")).toHaveLength(MOVES - 1);
    expect(await dumpColors(h, 1)).toEqual(referenceColors(pose(MOVES)));
  });

  it("guard_never_spares_an_edit", async () => {
    // A guard of 0: drag moves never abandon, edits still do.
    const h = engine(STEP_POINTS, 0);
    await setup(h);
    const edit: Request = {
      kind: "edit",
      rev: rev(),
      id: 1,
      payload: [{ op: "setPositions", vertices: [30], positions: [0.2, 0.1, 0.9] }],
    };
    const { placed, chunks } = await placeThen(h, [edit]);
    expect(chunks.filter((c) => c === "fieldStepAbandon")).toHaveLength(1);
    expect(recolors(placed, 1)).toBe(false);
  });

  it("drag_end_bypasses_guard", async () => {
    // A guard of 0 (always expired): a newer move alone keeps the unit...
    const h = engine(STEP_POINTS, 0);
    await setup(h);
    h.send([{ kind: "drag", rev: rev(), id: 1, phase: "begin", key: "placement:1" }]);
    await h.until(2);
    let chunksBefore = h.chunks.length;
    let stale: Request[][] = [[moveReq(2)]];
    h.onStep = () => {
      const batches = stale;
      stale = [];
      queueMicrotask(() => {
        for (const batch of batches) {
          h.send(batch);
        }
      });
    };
    h.send([moveReq(1)]);
    await h.until(4);
    expect(h.chunks.slice(chunksBefore)).not.toContain("fieldStepAbandon");
    expect(h.results.slice(2).map((r) => recolors(r, 1))).toEqual([true, true]);

    // ...but with the drag's end queued behind it, the kept unit is dropped
    // and the final pose follows.
    chunksBefore = h.chunks.length;
    stale = [[moveReq(4)], [{ kind: "drag", rev: rev(), id: 1, phase: "end", key: "placement:1" }]];
    h.send([moveReq(3)]);
    await h.until(7);
    h.onStep = undefined;
    const chunks = h.chunks.slice(chunksBefore);
    expect(chunks.filter((c) => c === "fieldStepAbandon")).toHaveLength(1);
    const [moved, last, ended] = h.results.slice(4);
    expect(recolors(moved, 1)).toBe(false);
    expect(recolors(last, 1)).toBe(true);
    expect(recolors(ended, 1)).toBe(false);
    expect(await dumpColors(h, 1)).toEqual(referenceColors(pose(4)));
  });
});

// ---- In a Node Worker (real `worker_threads`, built dist/) ----

describe("in a Node Worker", () => {
  it("message_arrives_mid_unit", async () => {
    // Steps of a few milliseconds each (in Node, a chain of `MessageChannel`
    // ticks after chunks of a millisecond or more lets no `parentPort`
    // message in), a yield after every step: a batch sent while the unit
    // steps must arrive before it ends.
    const worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), {
      workerData: { yieldMs: 0, stepPoints: 4096, staleGuardMs: LONG_GUARD, reportSteps: true },
    });
    cleanup.push(() => void worker.terminate());
    const results: Result[] = [];
    const chunks: string[] = [];
    let onStep: (() => void) | undefined;
    const waiting: (() => void)[] = [];
    worker.on("message", (msg: Result | { kind: "chunk"; chunk: { kind: string } }) => {
      if ("kind" in msg) {
        chunks.push(msg.chunk.kind);
        if (msg.chunk.kind === "fieldStep") {
          onStep?.();
        }
        return;
      }
      results.push(msg);
      for (const f of waiting.splice(0)) {
        f();
      }
    });
    const until = async (n: number) => {
      while (results.length < n) {
        await new Promise<void>((resolve) => waiting.push(resolve));
      }
    };
    const big = terrain(0, 0, 0, 160);
    worker.postMessage([
      addMeshReq(1, big),
      addMeshReq(2, TARGET),
      { kind: "field", rev: rev(), field: 1, op: "add", desc: { name: "d", on: 1, to: [2], maxDistance: MAX_DISTANCE } },
      { kind: "coloring", rev: rev(), id: 1, rule: { field: 1, map: "turbo", range: [0, MAX_DISTANCE] }, key: "display:1" },
    ]);
    await until(1);
    const chunksBefore = chunks.length;

    // The edit is sent from the main thread after the unit's third step: in
    // Node the first tick after a quiet spell still lets a message in, the
    // ticks chained after it did not.
    let steps = 0;
    onStep = () => {
      if (++steps === 3) {
        worker.postMessage([
          { kind: "edit", rev: rev(), id: 1, payload: [{ op: "setPositions", vertices: [30], positions: [0.2, 0.1, 0.9] }] },
        ]);
      }
    };
    worker.postMessage([
      { kind: "setPlacement", rev: rev(), id: 1, rotation: IDENTITY, translation: pose(3), key: "placement:1" },
    ]);
    await until(3);
    onStep = undefined;
    const [placed, edited] = results.slice(1);
    expect(chunks.slice(chunksBefore)).toContain("fieldStepAbandon");
    expect(recolors(placed, 1)).toBe(false);
    expect(recolors(edited, 1)).toBe(true);
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
        staleGuardMs: LONG_GUARD,
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
