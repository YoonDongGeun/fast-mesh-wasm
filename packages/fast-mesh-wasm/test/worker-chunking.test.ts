// Issue #48 in the Worker: a field add builds its field in steps (sign data,
// near search and values), yielding between them, and a build for a
// superseded display is abandoned at a step boundary. The engine runs
// in-process (sources, same wasm) with its test-only `onChunk` hook, which
// delivers newer batches between chunks.
import { describe, expect, it } from "vitest";
import { Scene } from "../src/node.js";
import { WasmScene } from "../src/wasm/fast_mesh_wasm.js";
import { runEngine } from "../src/worker/engine.js";
import type { EditOp } from "../src/node.js";
import type { ClientId, Request, Result } from "../src/worker/protocol.js";

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
/** Few vertices a step, so a build spans many steps. */
const STEP_POINTS = 16;

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

const A = terrain(0, 0, 0);
const B = terrain(0.5, 0.3, 0.4);
const C = terrain(10, 0, 0, 8);

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

function fieldAdd(field: ClientId, on: ClientId, to: ClientId, maxDistance: number, key?: string): Request {
  const req: Request = { kind: "field", rev: rev(), field, op: "add", desc: { name: `f${field}`, on, to: [to], maxDistance } };
  return key === undefined ? req : { ...req, key };
}

/** The engine in-process with a small step; `onChunk` sees every chunk. */
function engine() {
  const results: Result[] = [];
  const chunks: { kind: string; id?: ClientId }[] = [];
  const waiting: (() => void)[] = [];
  let deliver!: (batch: Request[]) => void;
  const h = {
    results,
    chunks,
    onChunk: undefined as ((chunk: { kind: string; id?: ClientId }) => void) | undefined,
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
      yieldMs: 0,
      onChunk: (chunk) => {
        chunks.push(chunk);
        h.onChunk?.(chunk);
      },
    },
  );
  return h;
}

/** A synchronous scene's values of a field on `a` to `b` (meshes as the engine has them). */
function referenceValues(maxDistance: number): Float32Array {
  const s = new Scene();
  try {
    const a = s.addMesh(A.positions, A.indices);
    const b = s.addMesh(B.positions, B.indices);
    s.addMesh(C.positions, C.indices);
    const f = s.addDistanceField({ name: "r", on: a, to: b, maxDistance });
    return s.fieldValues(f).slice();
  } finally {
    s.free();
  }
}

async function fetchValues(h: ReturnType<typeof engine>, field: ClientId): Promise<Float32Array> {
  const n = h.results.length + 1;
  h.send([{ kind: "fetchFieldValues", rev: rev(), qid: 5000 + n, field }]);
  await h.until(n);
  const reply = h.results[n - 1].replies[0];
  expect(reply.ok).toBe(true);
  return reply.value as Float32Array;
}

const bytes = (v: Float32Array) => Buffer.from(v.buffer, v.byteOffset, v.byteLength);

describe("field builds in steps", () => {
  it("field_add_builds_in_steps_with_yields", async () => {
    const h = engine();
    h.send([addMeshReq(1, A), addMeshReq(2, B), addMeshReq(3, C)]);
    await h.until(1);
    // A batch arriving after the build's first step is received before the
    // build finishes, and runs after it, in its own result.
    const toggle: Request = {
      kind: "setPlacement",
      rev: rev(),
      id: 3,
      rotation: IDENTITY,
      translation: [0, 1, 0],
      key: "placement:3",
    };
    let receivedAt = -1;
    h.onChunk = (chunk) => {
      if (chunk.kind === "fieldBuildStep" && receivedAt < 0) {
        receivedAt = h.chunks.length;
        h.later([toggle]);
      }
    };
    h.send([
      fieldAdd(1, 1, 2, Number.POSITIVE_INFINITY),
      { kind: "coloring", rev: rev(), id: 1, rule: { field: 1, map: "turbo", range: [0, 1] }, key: "display:1" },
    ]);
    await h.until(3);
    const kinds = h.chunks.map((c) => c.kind);
    const built = kinds.indexOf("fieldBuild");
    const steps = kinds.filter((k) => k === "fieldBuildStep").length;
    expect(steps).toBeGreaterThan(10);
    expect(receivedAt).toBeGreaterThan(0);
    expect(receivedAt).toBeLessThan(built);
    // The toggle ran after the build: its result is the third.
    expect(kinds.lastIndexOf("setPlacement")).toBeGreaterThan(built);
    expect(h.results[1].status).toContainEqual({ key: "field:1", rev: expect.any(Number), status: "ready" });
    expect(h.results[1].changes.find((c) => c.id === 1)?.fields).toEqual([1]);
    // The same values as a synchronous build, bit for bit.
    expect(bytes(await fetchValues(h, 1)).equals(bytes(referenceValues(Number.POSITIVE_INFINITY)))).toBe(true);
    // A finite reach (near search and sign data both stepped).
    h.onChunk = undefined;
    h.send([fieldAdd(2, 1, 2, 0.3)]);
    await h.until(h.results.length + 1);
    expect(bytes(await fetchValues(h, 2)).equals(bytes(referenceValues(0.3)))).toBe(true);
  });

  it("a_refused_withdraw_reports_the_removal_with_its_id", async () => {
    // Re-review R2: should the scene refuse to withdraw an abandoned build
    // (not reachable today; forced here), the field is removed the usual
    // way, and its removal record names its client id.
    await patched(
      "withdraw_field",
      () => () => false,
      async () => {
        const h = engine();
        h.send([addMeshReq(1, A), addMeshReq(2, B), addMeshReq(3, C)]);
        await h.until(1);
        const display = "display:1";
        const first: Request = { kind: "coloring", rev: rev(), id: 1, rule: { field: 1, map: "turbo", range: [0, 1] }, key: display };
        const second: Request = { kind: "coloring", rev: rev(), id: 1, rule: null, key: display };
        let step = 0;
        h.onChunk = (chunk) => {
          if (chunk.kind === "fieldBuildStep" && ++step === 3) {
            h.later([second]);
          }
        };
        h.send([fieldAdd(1, 1, 2, Number.POSITIVE_INFINITY, display), first]);
        await h.until(3);
        expect(h.chunks.map((c) => c.kind)).toContain("fieldBuildAbandon");
        expect(h.results[1].changes.flatMap((c) => c.fields)).toEqual([1]);
      },
    );
  });

  it("superseded_display_build_is_abandoned_mid_build", async () => {
    const h = engine();
    h.send([addMeshReq(1, A), addMeshReq(2, B), addMeshReq(3, C)]);
    await h.until(1);
    const display = "display:1";
    const first: Request = { kind: "coloring", rev: rev(), id: 1, rule: { field: 1, map: "turbo", range: [0, 1] }, key: display };
    const second: Request = { kind: "coloring", rev: rev(), id: 1, rule: null, key: display };
    let step = 0;
    h.onChunk = (chunk) => {
      if (chunk.kind === "fieldBuildStep" && ++step === 3) {
        h.later([second]);
      }
    };
    h.send([fieldAdd(1, 1, 2, Number.POSITIVE_INFINITY, display), first]);
    await h.until(3);
    const kinds = h.chunks.map((c) => c.kind);
    expect(kinds).toContain("fieldBuildAbandon");
    expect(kinds).not.toContain("fieldBuild");
    // Nothing of the abandoned build is reported: no field change, no status.
    const [r1, r2] = h.results.slice(1);
    expect(r1.changes.flatMap((c) => c.fields)).toEqual([]);
    expect(r1.status.filter((st) => st.key === display)).toEqual([]);
    expect(r2.status).toEqual([{ key: display, rev: second.rev, status: "ready" }]);
    // The abandoned field still works: it is built when first needed.
    h.onChunk = undefined;
    expect(bytes(await fetchValues(h, 1)).equals(bytes(referenceValues(Number.POSITIVE_INFINITY)))).toBe(true);
    expect(h.chunks.map((c) => c.kind)).toContain("fieldBuild");
  });

  it("field_add_errors_leave_no_field", async () => {
    const h = engine();
    h.send([addMeshReq(1, A), addMeshReq(2, B)]);
    await h.until(1);
    // A taken name fails at once; the client id is free again.
    const taken: Request = { kind: "field", rev: rev(), field: 2, op: "add", desc: { name: "f1", on: 1, to: [2], maxDistance: 0.3 } };
    h.send([fieldAdd(1, 1, 2, 0.3), taken]);
    await h.until(2);
    expect(h.results[1].status.find((st) => st.key === "field:2")?.status).toBe("error");
    h.send([fieldAdd(2, 1, 2, 0.5)]);
    await h.until(3);
    expect(h.results[2].status.find((st) => st.key === "field:2")?.status).toBe("ready");
  });
});

// ---- Background adds ----

/** What the engine posts: batch results, and `update` results (no batch). */
type Posted = { update: boolean; result: Result };

/**
 * The engine in-process with background adds for meshes of 2000 triangles or more (BIG, not A or B) in
 * small steps, and an update channel.
 */
function addEngine(
  options: { backgroundAddTriangles?: number; stepPoints?: number; addStepWork?: number; threads?: number } = {},
) {
  const posted: Posted[] = [];
  const chunks: { kind: string; id?: ClientId }[] = [];
  const waiting: (() => void)[] = [];
  let deliver!: (batch: Request[]) => void;
  const wake = () => {
    for (const f of waiting.splice(0)) {
      f();
    }
  };
  const h = {
    posted,
    chunks,
    onChunk: undefined as ((chunk: { kind: string; id?: ClientId }) => void) | undefined,
    later(batch: Request[]) {
      queueMicrotask(() => deliver(batch));
    },
    send(batch: Request[]) {
      deliver(batch);
    },
    /** Batch results only. */
    results: () => posted.filter((p) => !p.update).map((p) => p.result),
    async until(pred: () => boolean) {
      while (!pred()) {
        await new Promise<void>((resolve) => waiting.push(resolve));
      }
    },
  };
  runEngine(
    {
      post: (msg) => {
        posted.push({ update: false, result: msg });
        wake();
      },
      postUpdate: (msg) => {
        posted.push({ update: true, result: msg });
        wake();
      },
      onMessage: (cb) => {
        deliver = cb;
      },
    },
    {
      debug: true,
      yieldMs: 0,
      addStepWork: options.addStepWork ?? 64,
      threads: options.threads,
      stepPoints: options.stepPoints,
      backgroundAddTriangles: options.backgroundAddTriangles ?? 2000,
      onChunk: (chunk) => {
        chunks.push(chunk);
        h.onChunk?.(chunk);
      },
    },
  );
  return h;
}

const BIG = terrain(0, 5, 0, 40);

function placement(id: ClientId, x: number): Request {
  return { kind: "setPlacement", rev: rev(), id, rotation: IDENTITY, translation: [x, 0, 0], key: `placement:${id}` };
}

/** A synchronous scene's buffers of BIG added after A and B, then edited by `ops`. */
function referenceBuffers(ops: EditOp[] = []) {
  const s = new Scene();
  try {
    s.addMesh(A.positions, A.indices);
    s.addMesh(B.positions, B.indices);
    const m = s.addMesh(BIG.positions, BIG.indices);
    if (ops.length > 0) {
      s.edit(m, ops);
    }
    const v = s.views(m);
    return {
      positions: v.positions().slice(),
      normals: v.normals()?.slice() ?? null,
      indices: v.indices().slice(),
      hit: s.raycast([1, 6, 5], [0, 0, -1]),
    };
  } finally {
    s.free();
  }
}

describe("background adds", () => {
  it("toggle_runs_during_background_add", async () => {
    const h = addEngine();
    h.send([addMeshReq(1, A), addMeshReq(2, B)]);
    await h.until(() => h.results().length === 1);
    const toggle = placement(1, 0.5);
    let sent = false;
    h.onChunk = (chunk) => {
      if (chunk.kind === "addMeshStep" && !sent) {
        sent = true;
        h.later([toggle]);
      }
    };
    h.send([addMeshReq(3, BIG)]);
    await h.until(() => h.chunks.some((c) => c.kind === "addMesh" && c.id === 3) && h.posted.length >= 4);
    const kinds = h.chunks.map((c) => c.kind);
    // The toggle ran while the add was being built, and its result came first.
    expect(kinds.indexOf("setPlacement")).toBeLessThan(kinds.lastIndexOf("addMesh"));
    expect(kinds.filter((k) => k === "addMeshStep").length).toBeGreaterThan(5);
    const [, r1, r2] = h.results();
    expect(r1.status).toEqual([]);
    expect(r1.changes.find((c) => c.id === 3)).toBeUndefined();
    expect(r2.status).toEqual([{ key: "placement:1", rev: toggle.rev, status: "ready" }]);
    // The add landed while the Worker was idle: an update carries it.
    const update = h.posted.find((p) => p.update);
    expect(update).toBeDefined();
    const u = update?.result as Result;
    expect(u.rev).toBe(Math.max(r1.rev, r2.rev));
    expect(u.status).toEqual([{ key: "mesh:3", rev: expect.any(Number), status: "ready" }]);
    const added = u.changes.find((c) => c.id === 3);
    expect(added?.added).toBe(true);
    // The same buffers, bit for bit, as a synchronous add.
    const ref = referenceBuffers();
    const bytes = added?.bytes;
    expect(Buffer.from(bytes?.positions as ArrayBuffer).equals(Buffer.from(ref.positions.buffer))).toBe(true);
    expect(Buffer.from(bytes?.indices as ArrayBuffer).equals(Buffer.from(ref.indices.buffer))).toBe(true);
    expect(Buffer.from(bytes?.normals as ArrayBuffer).equals(Buffer.from((ref.normals as Float32Array).buffer))).toBe(true);
    expect(u.placements.map((p) => p.id)).toEqual([3]);
  });

  it("the_landing_is_copied_out_in_steps_before_it_lands", async () => {
    // Issue #64: the normals are computed in the build's steps and the
    // render buffers copied out in steps of their own, after the BVH, so
    // that landing packs nothing large in one piece. The record is the same,
    // bit for bit, as a synchronous add's.
    const h = addEngine();
    h.send([addMeshReq(1, A), addMeshReq(2, B)]);
    await h.until(() => h.results().length === 1);
    h.send([addMeshReq(3, BIG)]);
    await h.until(() => h.chunks.some((c) => c.kind === "addMesh" && c.id === 3) && h.posted.some((p) => p.update));
    const kinds = h.chunks.map((c) => c.kind);
    const packs = kinds.filter((k) => k === "addMeshPack").length;
    expect(packs).toBeGreaterThan(5);
    expect(kinds.lastIndexOf("addMeshStep")).toBeLessThan(kinds.indexOf("addMeshPack"));
    expect(kinds.lastIndexOf("addMeshPack")).toBeLessThan(kinds.lastIndexOf("addMesh"));
    const u = h.posted.find((p) => p.update)?.result as Result;
    const added = u.changes.find((c) => c.id === 3);
    const ref = referenceBuffers();
    const same = (got: ArrayBuffer | null | undefined, want: ArrayBufferView | null) =>
      got != null && want !== null && Buffer.from(got).equals(Buffer.from(want.buffer, want.byteOffset, want.byteLength));
    expect(same(added?.bytes.positions, ref.positions)).toBe(true);
    expect(same(added?.bytes.normals, ref.normals)).toBe(true);
    expect(same(added?.bytes.indices, ref.indices)).toBe(true);
  });

  it("an_edit_in_the_landing_batch_packs_the_edited_mesh", async () => {
    // The buffers copied out before the landing are those of the mesh as
    // built: an edit after it lands, in the same frame, is packed instead.
    const h = addEngine();
    h.send([addMeshReq(1, A), addMeshReq(2, B)]);
    await h.until(() => h.results().length === 1);
    const op: EditOp = { op: "setPositions", vertices: [41], positions: [1.1, 6.2, 0.9] };
    h.send([addMeshReq(3, BIG), { kind: "edit", rev: rev(), id: 3, payload: [op] }]);
    await h.until(() => h.results().length === 2);
    expect(h.chunks.some((c) => c.kind === "addMeshPack")).toBe(true);
    const r = h.results()[1];
    expect(r.status.map((st) => st.key)).toEqual(["mesh:3", "edit:3"]);
    const added = r.changes.find((c) => c.id === 3);
    expect(added?.added).toBe(true);
    const ref = referenceBuffers([op]);
    const same = (got: ArrayBuffer | null | undefined, want: ArrayBufferView | null) =>
      got != null && want !== null && Buffer.from(got).equals(Buffer.from(want.buffer, want.byteOffset, want.byteLength));
    expect(same(added?.bytes.positions, ref.positions)).toBe(true);
    expect(same(added?.bytes.normals, ref.normals)).toBe(true);
    expect(same(added?.bytes.indices, ref.indices)).toBe(true);
    expect(same(added?.bytes.positions, referenceBuffers().positions)).toBe(false);
  });

  it("a_waiting_request_steps_the_add_single_threaded", async () => {
    // Re-review R1: single-threaded, a request waiting on a background add
    // still lets the add go on in steps with yields (the one-call build
    // would not be faster), so newer batches arrive and coalesce meanwhile.
    const h = addEngine();
    h.send([addMeshReq(1, A)]);
    await h.until(() => h.results().length === 1);
    let arrived = -1;
    h.onChunk = (chunk) => {
      if (chunk.kind === "addMeshStep" && arrived < 0) {
        arrived = h.chunks.length;
        h.later([placement(1, 0.25)]);
      }
    };
    h.send([addMeshReq(3, BIG), placement(3, 0.5)]);
    await h.until(() => h.results().length === 3);
    const steps = h.chunks.filter((c) => c.kind === "addMeshStep" && c.id === 3).length;
    expect(steps).toBeGreaterThan(10);
    // The batch sent during the wait was received before the add landed.
    const landed = h.chunks.findIndex((c) => c.kind === "addMesh" && c.id === 3);
    expect(arrived).toBeGreaterThan(0);
    expect(arrived).toBeLessThan(landed);
    expect(h.results()[1].status.map((st) => st.key)).toEqual(["mesh:3", "placement:3"]);
  });

  it("a_waiting_request_finishes_the_add_in_one_call_threaded", async () => {
    // Review I3: on a threaded engine, once a request depends on a
    // background add, the rest of the build runs in one call (the
    // one-call, parallel build), not in small steps with yields.
    const h = addEngine({ threads: 4 });
    h.send([addMeshReq(1, A)]);
    await h.until(() => h.results().length === 1);
    h.send([addMeshReq(3, BIG), placement(3, 0.5)]);
    await h.until(() => h.results().length === 2);
    // At most the one step the engine's yield between the two requests
    // gives a background add; the rest in the call that lands it.
    const steps = h.chunks.filter((c) => c.kind === "addMeshStep" && c.id === 3).length;
    expect(steps).toBeLessThanOrEqual(1);
    expect(h.chunks.some((c) => c.kind === "addMeshPack")).toBe(false);
    expect(h.results()[1].status.map((st) => st.key)).toEqual(["mesh:3", "placement:3"]);
    // Packed at the frame, as before: the same buffers as a synchronous add.
    const bytes = h.results()[1].changes.find((c) => c.id === 3)?.bytes;
    const ref = referenceBuffers();
    expect(Buffer.from(bytes?.positions as ArrayBuffer).equals(Buffer.from(ref.positions.buffer))).toBe(true);
    expect(Buffer.from(bytes?.normals as ArrayBuffer).equals(Buffer.from((ref.normals as Float32Array).buffer))).toBe(true);
  });

  it("add_step_work_is_clamped", async () => {
    // Review M1: an unbounded step (Infinity) builds the whole BVH in one
    // step instead of wrapping to nothing, the normals in a second (#64)
    // and the copy of the buffers in one more; NaN and values below 1 fall
    // back to the default (100,000 units: the same steps for this mesh);
    // 2 units take many steps.
    for (const addStepWork of [Number.POSITIVE_INFINITY, 2 ** 40, Number.NaN, -5, 2]) {
      const h = addEngine({ addStepWork });
      h.send([addMeshReq(1, A)]);
      await h.until(() => h.results().length === 1);
      h.send([addMeshReq(3, BIG)]);
      await h.until(() => h.chunks.some((c) => c.kind === "addMesh" && c.id === 3));
      const steps = h.chunks.filter((c) => c.kind === "addMeshStep").length;
      const packs = h.chunks.filter((c) => c.kind === "addMeshPack").length;
      if (addStepWork === 2) {
        expect(steps).toBeGreaterThan(100);
        expect(packs).toBeGreaterThan(100);
      } else {
        expect([steps, packs], `${addStepWork}`).toEqual([2, 1]);
      }
    }
  });

  it("background_add_threshold_is_validated", async () => {
    // NaN keeps the default; a negative threshold counts as 0 (every add).
    const h = addEngine({ backgroundAddTriangles: Number.NaN });
    h.send([addMeshReq(1, BIG)]);
    await h.until(() => h.results().length === 1);
    expect(h.chunks.some((c) => c.kind === "addMeshBegin")).toBe(false);
    const g = addEngine({ backgroundAddTriangles: -1 });
    g.send([addMeshReq(1, A)]);
    await g.until(() => g.chunks.some((c) => c.kind === "addMesh"));
    expect(g.chunks.some((c) => c.kind === "addMeshBegin")).toBe(true);
  });

  it("dependent_requests_wait_for_the_add", async () => {
    const h = addEngine();
    h.send([addMeshReq(1, A), addMeshReq(2, B)]);
    await h.until(() => h.results().length === 1);
    // A placement of the new mesh waits for it, in the same batch.
    const move = placement(3, 0.25);
    h.send([addMeshReq(3, BIG), move]);
    await h.until(() => h.results().length === 2);
    const r = h.results()[1];
    expect(r.status).toEqual([
      { key: "mesh:3", rev: expect.any(Number), status: "ready" },
      { key: "placement:3", rev: move.rev, status: "ready" },
    ]);
    const c = r.changes.find((x) => x.id === 3);
    expect(c?.added).toBe(true);
    expect(c?.placementChanged).toBe(true);
    expect(h.posted.some((p) => p.update)).toBe(false);
    // A raycast (every mesh) waits too, and sees the new mesh exactly.
    h.send([addMeshReq(4, BIG), { kind: "query", rev: rev(), qid: 7, name: "raycast", args: { origin: [1, 6, 5], direction: [0, 0, -1] } }]);
    await h.until(() => h.results().length === 3);
    const r3 = h.results()[2];
    expect(r3.status[0]).toEqual({ key: "mesh:4", rev: expect.any(Number), status: "ready" });
    const hit = r3.replies[0].value as { mesh: ClientId; distance: number };
    expect([3, 4]).toContain(hit.mesh);
    // Two adds in a row land in arrival order; a removal waits for them.
    h.send([addMeshReq(5, BIG), addMeshReq(6, BIG), { kind: "removeMesh", rev: rev(), id: 5 }]);
    await h.until(() => h.results().length === 4);
    const r4 = h.results()[3];
    expect(r4.status.map((st) => st.key)).toEqual(["mesh:5", "mesh:6", "mesh:5"]);
    const order = h.chunks.filter((c) => c.kind === "addMesh").map((c) => c.id);
    expect(order.slice(-2)).toEqual([5, 6]);
  });

  it("background_add_matches_synchronous_add", async () => {
    const h = addEngine();
    h.send([addMeshReq(1, A), addMeshReq(2, B), addMeshReq(3, BIG)]);
    await h.until(() => h.chunks.some((c) => c.kind === "addMesh" && c.id === 3));
    await h.until(() => h.posted.length >= 2);
    const n = h.results().length + 1;
    h.send([{ kind: "query", rev: rev(), qid: 9, name: "raycast", args: { origin: [1, 6, 5], direction: [0, 0, -1] } }]);
    await h.until(() => h.results().length === n);
    const hit = h.results()[n - 1].replies[0].value as { mesh: ClientId; triangle: number; distance: number };
    const ref = referenceBuffers().hit;
    expect(hit.mesh).toBe(3);
    expect(hit.triangle).toBe(ref?.triangle);
    expect(hit.distance).toBe(ref?.distance);
  });

  it("small_and_inline_adds_stay_in_place", async () => {
    const h = addEngine({ backgroundAddTriangles: 1_000_000 });
    h.send([addMeshReq(1, A), addMeshReq(2, BIG)]);
    await h.until(() => h.results().length === 1);
    expect(h.results()[0].status.map((st) => st.key)).toEqual(["mesh:1", "mesh:2"]);
    expect(h.chunks.some((c) => c.kind === "addMeshBegin")).toBe(false);
    const g = addEngine();
    g.send([{ ...addMeshReq(1, BIG), inline: true } as Request]);
    await g.until(() => g.results().length === 1);
    expect(g.results()[0].changes.find((c) => c.id === 1)?.added).toBe(true);
    expect(g.chunks.some((c) => c.kind === "addMeshBegin")).toBe(false);
  });

  it("deferred_field_on_a_pending_mesh_waits_for_it", async () => {
    // Review C1: a display-keyed field add measured to a mesh still being
    // added, abandoned by the 9a rule (its colouring is superseded at
    // once), then needed: it must be built after the mesh lands, as in
    // arrival order, not fail and be lost.
    const fieldRef = () => {
      const s = new Scene();
      try {
        const a = s.addMesh(A.positions, A.indices);
        s.addMesh(B.positions, B.indices);
        const m = s.addMesh(BIG.positions, BIG.indices);
        const f = s.addDistanceField({ name: "f10", on: a, to: m, maxDistance: 1 });
        return { values: s.fieldValues(f).slice(), status: s.fieldStatus(f) };
      } finally {
        s.free();
      }
    };
    const ref = fieldRef();
    const display = "display:1";
    const add10: Request = {
      kind: "field",
      rev: rev(),
      field: 10,
      op: "add",
      desc: { name: "f10", on: 1, to: [3], maxDistance: 1 },
      key: display,
    };
    const colour10: Request = {
      kind: "coloring",
      rev: rev(),
      id: 1,
      rule: { field: 10, map: "turbo", range: [0, 1] },
      key: display,
    };
    const uncolour: Request = { kind: "coloring", rev: rev(), id: 1, rule: null, key: display };
    const fetch = (qid: number): Request => ({ kind: "fetchFieldValues", rev: rev(), qid, field: 10 });
    const status = (qid: number): Request => ({ kind: "query", rev: rev(), qid, name: "fieldStatus", args: { field: 10 } });

    // fetchFieldValues of the deferred field.
    let h = addEngine();
    h.send([addMeshReq(1, A), addMeshReq(2, B)]);
    await h.until(() => h.results().length === 1);
    h.send([addMeshReq(3, BIG), add10, colour10, uncolour, fetch(5)]);
    await h.until(() => h.results().length === 2);
    let reply = h.results()[1].replies[0];
    expect(reply.ok, JSON.stringify(reply.error)).toBe(true);
    expect(bytes(reply.value as Float32Array).equals(bytes(ref.values))).toBe(true);
    expect(h.results()[1].status.find((st) => st.key === "mesh:3")?.status).toBe("ready");

    // A colouring switching back to the deferred field, then its status.
    h = addEngine();
    h.send([addMeshReq(1, A), addMeshReq(2, B)]);
    await h.until(() => h.results().length === 1);
    const back: Request = { ...colour10, rev: rev() } as Request;
    h.send([addMeshReq(3, BIG), add10, colour10, uncolour, back, status(6)]);
    await h.until(() => h.results().length === 2);
    const r = h.results()[1];
    expect(r.status.find((st) => st.key === display && st.rev === back.rev)?.status).toBe("ready");
    reply = r.replies[0];
    expect(reply.ok, JSON.stringify(reply.error)).toBe(true);
    expect(r.changes.find((c) => c.id === 1)?.colors).not.toEqual([]);
  });

  it("deferred_field_build_failure_keeps_the_field", async () => {
    // A deferred field whose build fails (its name is taken by then) keeps
    // its desc: the error is reported each time it is needed, and the id
    // stays known.
    const h = addEngine();
    h.send([addMeshReq(1, A), addMeshReq(2, B)]);
    await h.until(() => h.results().length === 1);
    const display = "display:1";
    const add10: Request = {
      kind: "field",
      rev: rev(),
      field: 10,
      op: "add",
      desc: { name: "dup", on: 1, to: [2], maxDistance: 1 },
      key: display,
    };
    const colour10: Request = { kind: "coloring", rev: rev(), id: 1, rule: { field: 10, map: "turbo", range: [0, 1] }, key: display };
    const uncolour: Request = { kind: "coloring", rev: rev(), id: 1, rule: null, key: display };
    const taken: Request = { kind: "field", rev: rev(), field: 11, op: "add", desc: { name: "dup", on: 1, to: [2], maxDistance: 1 } };
    const fetch: Request = { kind: "fetchFieldValues", rev: rev(), qid: 9, field: 10 };
    h.send([add10, colour10, uncolour, taken, fetch]);
    await h.until(() => h.results().length === 2);
    expect(h.results()[1].replies[0].error?.code).toBe("INVALID_ARGUMENT");
    // Still known: removing it works, after which it is unknown.
    h.send([{ kind: "field", rev: rev(), field: 10, op: "remove" }, { ...fetch, rev: rev(), qid: 10 } as Request]);
    await h.until(() => h.results().length === 3);
    expect(h.results()[2].status.find((st) => st.key === "field:10")?.status).toBe("ready");
    expect(h.results()[2].replies[0].error?.code).toBe("UNKNOWN_FIELD");
  });

  it("failed_background_add_reports_its_error", async () => {
    const h = addEngine();
    // An index out of range fails when the build starts: at once.
    const bad: Request = { ...addMeshReq(1, A), indices: Uint32Array.from([0, 1, 99999]).buffer } as Request;
    h.send([bad]);
    await h.until(() => h.results().length === 1);
    expect(h.results()[0].status[0].status).toBe("error");
    // The id is free again.
    h.send([addMeshReq(1, A)]);
    await h.until(() => h.posted.length >= 2 && h.chunks.some((c) => c.kind === "addMesh"));
  });
});

// ---- Poisoning with a background add pending (review I1) ----

type Proto = Record<string, (...args: unknown[]) => unknown>;

/** Replaces a `WasmScene` method for the duration of `run`. */
async function patched(name: string, make: (orig: (...args: unknown[]) => unknown) => (...args: unknown[]) => unknown, run: () => Promise<void>) {
  const proto = WasmScene.prototype as unknown as Proto;
  const orig = proto[name];
  proto[name] = make(orig);
  try {
    await run();
  } finally {
    proto[name] = orig;
  }
}

/**
 * A trap as the engine sees it (`INSTANCE_POISONED`), without poisoning
 * this test process's shared instance as a real `RuntimeError` would.
 */
const trap = () => {
  throw new Error("INSTANCE_POISONED: trap (test)");
};

/** Every status posted for `key`, as `status` strings. */
const statusesOf = (h: ReturnType<typeof addEngine>, key: string) =>
  h.posted.flatMap((p) => p.result.status.filter((st) => st.key === key).map((st) => st.status));

describe("poisoning with a background add pending", () => {
  it("an_unrelated_trap_errors_the_pending_add", async () => {
    await patched("analyze", () => trap, async () => {
      const h = addEngine();
      h.send([addMeshReq(1, A)]);
      await h.until(() => h.results().length === 1);
      h.send([addMeshReq(3, BIG)]);
      await h.until(() => h.results().length === 2);
      h.send([{ kind: "query", rev: rev(), qid: 1, name: "analyze", args: { mesh: 1 } }]);
      await h.until(() => h.results().length === 3);
      const r = h.results()[2];
      expect(r.poisoned).toBe(true);
      expect(statusesOf(h, "mesh:3")).toEqual(["error"]);
      expect(r.replies[0].ok).toBe(false);
    });
  });

  it("a_trap_while_finishing_adds_errors_the_waiting_request", async () => {
    // The waiting request lands the add in one call: that call traps.
    await patched(
      "finish_add_mesh",
      () => trap,
      async () => {
        const h = addEngine();
        h.send([addMeshReq(1, A)]);
        await h.until(() => h.results().length === 1);
        const move = placement(3, 1);
        h.send([addMeshReq(3, BIG), move]);
        await h.until(() => h.results().length === 2);
        const r = h.results()[1];
        expect(r.poisoned).toBe(true);
        expect(r.status.find((st) => st.key === "placement:3")?.status).toBe("error");
        expect(r.status.find((st) => st.key === "mesh:3")?.status).toBe("error");
      },
    );
  });

  it("a_trap_while_stepping_fields_keeps_one_result_per_batch", async () => {
    let calls = 0;
    let armed = false;
    await patched(
      "add_mesh_step",
      (orig) =>
        function (this: unknown, ...args: unknown[]) {
          if (armed && ++calls === 2) {
            trap();
          }
          return orig.apply(this, args);
        },
      async () => {
        const h = addEngine({ stepPoints: 4 });
        h.send([
          addMeshReq(1, A),
          addMeshReq(2, B),
          { kind: "field", rev: rev(), field: 1, op: "add", desc: { name: "d", on: 1, to: [2], maxDistance: 8 } },
        ]);
        await h.until(() => h.results().length === 1);
        armed = true;
        // A background add, then a batch whose field recompute steps (the
        // add steps between them and traps), and one more batch behind.
        h.send([addMeshReq(3, BIG)]);
        h.send([placement(1, 0.2)]);
        const last = placement(2, 0.1);
        h.send([last]);
        await h.until(() => h.results().length === 4);
        await new Promise((resolve) => setTimeout(resolve, 20));
        const results = h.results();
        expect(results.length).toBe(4);
        expect(results[3].poisoned).toBe(true);
        expect(results[3].status).toEqual([expect.objectContaining({ key: "placement:2", status: "error" })]);
        expect(statusesOf(h, "mesh:3")).toEqual(["error"]);
        expect(h.posted.filter((p) => p.update)).toEqual([]);
      },
    );
  });

  it("an_idle_trap_is_posted_as_an_update", async () => {
    let calls = 0;
    await patched(
      "add_mesh_step",
      (orig) =>
        function (this: unknown, ...args: unknown[]) {
          if (++calls === 3) {
            trap();
          }
          return orig.apply(this, args);
        },
      async () => {
        const h = addEngine();
        h.send([addMeshReq(1, A)]);
        await h.until(() => h.results().length === 1);
        h.send([addMeshReq(3, BIG)]);
        await h.until(() => h.posted.some((p) => p.update));
        const u = h.posted.find((p) => p.update)?.result as Result;
        expect(u.poisoned).toBe(true);
        expect(u.status).toEqual([expect.objectContaining({ key: "mesh:3", status: "error" })]);
        // One result for each of the two batches, no more.
        expect(h.results().length).toBe(2);
        // Later batches are answered as poisoned.
        h.send([placement(1, 0.3)]);
        await h.until(() => h.results().length === 3);
        expect(h.results()[2].poisoned).toBe(true);
      },
    );
  });
});
