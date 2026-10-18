// The Worker-side engine through the Node `worker_threads` harness (which
// runs the built dist/): request order, coalescing, client ids, results with
// dirty-range bytes, errors and poisoning. A sync `Scene` from the sources
// replays the same requests as the reference: same wasm, same ops, so the
// mirror rebuilt from results must match its views byte for byte.
// `trap_posts_poisoned` needs the test build of the wasm (`__test_panic`).
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import { type MeshHandle, Scene, type SectionResult, type Vec3 } from "../src/node.js";
import type { ClientId, MeshChangeWire, Request, Result } from "../src/worker/protocol.js";

const QUARTER_Z = [0, -1, 0, 1, 0, 0, 0, 0, 1];
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

/** The engine in a worker thread: each batch sent gets the next result. */
class Harness {
  readonly worker: Worker;
  readonly #pending: { resolve: (r: Result) => void; reject: (e: unknown) => void }[] = [];

  constructor(workerData: { testPanic?: boolean } = {}) {
    this.worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), { workerData });
    this.worker.on("message", (r: Result) => this.#pending.shift()?.resolve(r));
    this.worker.on("error", (e) => {
      for (const p of this.#pending.splice(0)) {
        p.reject(e);
      }
    });
  }

  send(batch: Request[]): Promise<Result> {
    const transfer: ArrayBuffer[] = [];
    for (const req of batch) {
      if (req.kind === "addMesh") {
        transfer.push(req.positions, req.indices);
      }
    }
    return new Promise((resolve, reject) => {
      this.#pending.push({ resolve, reject });
      this.worker.postMessage(batch, transfer);
    });
  }
}

const harnesses: Harness[] = [];
const scenes: Scene[] = [];
function harness(workerData?: { testPanic?: boolean }): Harness {
  const h = new Harness(workerData);
  harnesses.push(h);
  return h;
}
function scene(): Scene {
  const s = new Scene();
  scenes.push(s);
  return s;
}
afterEach(async () => {
  for (const h of harnesses.splice(0)) {
    await h.worker.terminate();
  }
  for (const s of scenes.splice(0)) {
    s.free();
  }
});

let revision = 0;
const rev = () => ++revision;

/** An `addMesh` request with copies of the buffers (they are transferred). */
function addMesh(id: ClientId, mesh: { positions: Float64Array; indices: Uint32Array }): Request {
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

/** Main-thread mirrors of each mesh's buffers, rebuilt from results only. */
interface MeshMirror {
  positions: Float32Array | null;
  normals: Float32Array | null;
  colors: Uint8Array | null;
  indices: Uint32Array | null;
}

const STRIDE = { positions: 3, normals: 3, colors: 4, indices: 3 } as const;
const MAKE = {
  positions: (b: ArrayBuffer) => new Float32Array(b),
  normals: (b: ArrayBuffer) => new Float32Array(b),
  colors: (b: ArrayBuffer) => new Uint8Array(b),
  indices: (b: ArrayBuffer) => new Uint32Array(b),
};

function applyChange(mirrors: Map<ClientId, MeshMirror>, c: MeshChangeWire): void {
  if (c.removed) {
    mirrors.delete(c.id);
    return;
  }
  let m = mirrors.get(c.id);
  if (m === undefined) {
    m = { positions: null, normals: null, colors: null, indices: null };
    mirrors.set(c.id, m);
  }
  for (const name of Object.keys(STRIDE) as (keyof typeof STRIDE)[]) {
    const bytes = c.bytes[name];
    const ranges = c[name];
    if (bytes === null) {
      m[name] = null;
      continue;
    }
    const src = MAKE[name](bytes);
    if (ranges === "all") {
      m[name] = src as never;
      continue;
    }
    const dst = m[name];
    if (dst === null) {
      throw new Error(`ranges for ${name} of mesh ${c.id} without a buffer`);
    }
    let at = 0;
    for (const [start, count] of ranges) {
      const n = count * STRIDE[name];
      dst.set(src.subarray(at, at + n), start * STRIDE[name]);
      at += n;
    }
  }
}

function applyResult(mirrors: Map<ClientId, MeshMirror>, r: Result): void {
  for (const c of r.changes) {
    applyChange(mirrors, c);
  }
}

function bytesOf(v: ArrayBufferView | null): Buffer | null {
  return v === null ? null : Buffer.from(v.buffer, v.byteOffset, v.byteLength);
}

function expectMirror(m: MeshMirror | undefined, ref: Scene, mesh: MeshHandle): void {
  expect(m).toBeDefined();
  const v = ref.views(mesh);
  for (const name of Object.keys(STRIDE) as (keyof typeof STRIDE)[]) {
    const want = bytesOf(v[name]());
    const got = bytesOf(m![name]);
    if (want === null || got === null) {
      expect(got, name).toBe(want);
    } else {
      expect(got.equals(want), `${name} bytes`).toBe(true);
    }
  }
}

/** The latest status of `key` in a result. */
function statusOf(r: Result, key: string) {
  return [...r.status].reverse().find((s) => s.key === key);
}

/** Every status of `key` in a result, in order, as `[rev, status, code]`. */
function statusesOf(r: Result, key: string) {
  return r.status.filter((s) => s.key === key).map((s) => [s.rev, s.status, s.error?.code]);
}

function replyOf(r: Result, qid: number) {
  return r.replies.find((x) => x.qid === qid);
}

function expectSection(got: SectionResult | null | undefined, want: SectionResult): void {
  expect(got).toBeTruthy();
  expect(got!.status).toBe(want.status);
  expect(got!.length).toBeCloseTo(want.length, 12);
  expect(got!.area).toBeCloseTo(want.area, 12);
  expect(got!.contours.map((c) => [c.closed, Array.from(c.points)])).toEqual(
    want.contours.map((c) => [c.closed, Array.from(c.points)]),
  );
  expect(got!.fill === undefined).toBe(want.fill === undefined);
  if (want.fill !== undefined) {
    expect(Array.from(got!.fill!.positions)).toEqual(Array.from(want.fill.positions));
    expect(Array.from(got!.fill!.indices)).toEqual(Array.from(want.fill.indices));
  }
}

describe("worker engine", () => {
  it("add_place_round_trip", async () => {
    const h = harness();
    const ref = scene();
    const mirrors = new Map<ClientId, MeshMirror>();
    const a = sphere([0, 0, 0]);
    const b = sphere([3, 0, 0]);

    // Add, then place, in one batch.
    const batch1: Request[] = [
      addMesh(1, a),
      addMesh(2, b),
      { kind: "setPlacement", rev: rev(), id: 1, rotation: QUARTER_Z, translation: [1, 2, 3], key: "placement:1" },
    ];
    const r1 = await h.send(batch1);
    const ra = ref.addMesh(a.positions, a.indices);
    const rb = ref.addMesh(b.positions, b.indices);
    ref.setPlacement(ra, { rotation: QUARTER_Z, translation: [1, 2, 3] });

    expect(r1.rev).toBe(batch1[2].rev);
    const c1 = r1.changes.find((c) => c.id === 1)!;
    expect(c1.added).toBe(true);
    expect(c1.positions).toBe("all");
    expect(c1.vertexSlots).toBe(ref.vertexSlots(ra));
    expect(c1.triangleSlots).toBe(ref.triangleSlots(ra));
    applyResult(mirrors, r1);
    expectMirror(mirrors.get(1), ref, ra);
    expectMirror(mirrors.get(2), ref, rb);
    const p1 = r1.placements.find((p) => p.id === 1)!;
    expect(p1.matrix4).toEqual(Array.from(ref.matrix4(ra)));
    expect(p1.rotation).toEqual(ref.placement(ra).rotation);
    expect(p1.translation).toEqual([1, 2, 3]);
    // The added mesh's matrix (its origin) arrives too.
    expect(r1.placements.find((p) => p.id === 2)!.matrix4).toEqual(Array.from(ref.matrix4(rb)));
    expect(statusOf(r1, "mesh:1")).toMatchObject({ status: "ready", rev: batch1[0].rev });
    expect(statusOf(r1, "placement:1")).toMatchObject({ status: "ready", rev: batch1[2].rev });
    expect(r1.history).toEqual({ canUndo: true, canRedo: false });

    // An edit sends only its dirty ranges.
    const ops = [{ op: "setPositions" as const, vertices: [0, 7], positions: [2, 2, 4, 1.5, 2.5, 3.5] }];
    const r2 = await h.send([{ kind: "edit", rev: rev(), id: 1, payload: ops, qid: 1 }]);
    const refEdit = ref.edit(ra, ops);
    const c2 = r2.changes.find((c) => c.id === 1)!;
    expect(c2.positions).not.toBe("all");
    expect((c2.positions as unknown[]).length).toBeGreaterThan(0);
    expect(r2.placements).toEqual([]);
    expect(replyOf(r2, 1)).toMatchObject({ ok: true, value: refEdit });
    applyResult(mirrors, r2);
    expectMirror(mirrors.get(1), ref, ra);

    // Colouring by a distance field fills the colour mirror; off clears it.
    const rule = { field: 1, map: "turbo" as const, range: [0, 3] as [number, number] };
    const r3 = await h.send([
      { kind: "field", rev: rev(), field: 1, op: "add", desc: { name: "d", on: 1, to: 2, maxDistance: 5 } },
      { kind: "coloring", rev: rev(), id: 1, rule, key: "display:1" },
    ]);
    const f = ref.addDistanceField({ name: "d", on: ra, to: rb, maxDistance: 5 });
    ref.setColoring(ra, { ...rule, field: f });
    applyResult(mirrors, r3);
    expect(mirrors.get(1)!.colors).not.toBeNull();
    expectMirror(mirrors.get(1), ref, ra);
    expect(r3.changes.find((c) => c.id === 1)!.fields).toEqual([1]);

    // Moving the target mesh recolours only what may change.
    const r4 = await h.send([
      { kind: "setPlacement", rev: rev(), id: 2, rotation: IDENTITY, translation: [-0.5, 0, 0], key: "placement:2" },
    ]);
    ref.setPlacement(rb, { rotation: IDENTITY, translation: [-0.5, 0, 0] });
    applyResult(mirrors, r4);
    expectMirror(mirrors.get(1), ref, ra);

    const r5 = await h.send([{ kind: "coloring", rev: rev(), id: 1, rule: null, key: "display:1" }]);
    ref.setColoring(ra, null);
    applyResult(mirrors, r5);
    expect(mirrors.get(1)!.colors).toBeNull();
    expectMirror(mirrors.get(1), ref, ra);

    // Compaction and removal.
    const r6 = await h.send([
      {
        kind: "edit",
        rev: rev(),
        id: 1,
        payload: [{ op: "removeTriangles", triangles: [0, 1, 2] }],
      },
      { kind: "compact", rev: rev(), id: 1 },
      { kind: "removeMesh", rev: rev(), id: 2 },
    ]);
    ref.edit(ra, [{ op: "removeTriangles", triangles: [0, 1, 2] }]);
    ref.compact(ra);
    ref.removeMesh(rb);
    applyResult(mirrors, r6);
    expect(r6.changes.find((c) => c.id === 2)!.removed).toBe(true);
    expect(mirrors.has(2)).toBe(false);
    expectMirror(mirrors.get(1), ref, ra);
  });

  it("mixed_batch_keeps_edit_order", async () => {
    const h = harness();
    const ref = scene();
    const mirrors = new Map<ClientId, MeshMirror>();
    const a = sphere([0, 0, 0]);
    const b = sphere([3, 0, 0]);
    const rule = (hi: number) => ({ field: 1, map: "viridis" as const, range: [0, hi] as [number, number] });
    const move = (z: number) => [{ op: "setPositions" as const, vertices: [0], positions: [0, 0, z] }];

    const r = await h.send([
      addMesh(1, a),
      addMesh(2, b),
      { kind: "field", rev: rev(), field: 1, op: "add", desc: { name: "d", on: 1, to: 2, maxDistance: 5 } },
      { kind: "coloring", rev: rev(), id: 1, rule: rule(2), key: "display:1" },
      { kind: "edit", rev: rev(), id: 1, payload: move(1.5), qid: 1 },
      { kind: "coloring", rev: rev(), id: 1, rule: rule(3), key: "display:1" },
      { kind: "edit", rev: rev(), id: 1, payload: move(1.2), qid: 2 },
      { kind: "coloring", rev: rev(), id: 1, rule: rule(4), key: "display:1" },
    ]);
    const ra = ref.addMesh(a.positions, a.indices);
    const rb = ref.addMesh(b.positions, b.indices);
    const f = ref.addDistanceField({ name: "d", on: ra, to: rb, maxDistance: 5 });
    ref.setColoring(ra, { ...rule(2), field: f });
    ref.edit(ra, move(1.5));
    ref.setColoring(ra, { ...rule(3), field: f });
    ref.edit(ra, move(1.2));
    ref.setColoring(ra, { ...rule(4), field: f });

    // Both edits ran, in order (the second one wins), each replied in order.
    expect(r.replies.map((x) => [x.qid, x.ok])).toEqual([
      [1, true],
      [2, true],
    ]);
    applyResult(mirrors, r);
    expectMirror(mirrors.get(1), ref, ra);
    expectMirror(mirrors.get(2), ref, rb);
    // Undo takes back the second edit only.
    const u = await h.send([{ kind: "undo", rev: rev() }]);
    ref.undo();
    applyResult(mirrors, u);
    expectMirror(mirrors.get(1), ref, ra);
    expect(statusOf(u, "history")).toMatchObject({ status: "ready" });
  });

  it("handle_before_add_processed", async () => {
    const h = harness();
    const a = sphere([0, 0, 0]);

    // A request on a handle queues behind its add and applies.
    const add = addMesh(1, a);
    const place: Request = {
      kind: "setPlacement",
      rev: rev(),
      id: 1,
      rotation: IDENTITY,
      translation: [4, 5, 6],
      key: "placement:1",
    };
    const r1 = await h.send([add, place]);
    expect(statusOf(r1, "mesh:1")).toMatchObject({ status: "ready", rev: add.rev });
    expect(statusOf(r1, "placement:1")).toMatchObject({ status: "ready", rev: place.rev });
    expect(r1.placements.find((p) => p.id === 1)!.translation).toEqual([4, 5, 6]);

    // A failed add: the handle's key reports the add's error, later
    // requests on it give UNKNOWN_MESH.
    const bad = addMesh(2, { positions: a.positions, indices: Uint32Array.from([0, 1, 9999]) });
    const r2 = await h.send([
      bad,
      { kind: "setPlacement", rev: rev(), id: 2, rotation: IDENTITY, translation: [1, 0, 0], key: "placement:2" },
    ]);
    expect(statusOf(r2, "mesh:2")).toMatchObject({ status: "error", rev: bad.rev, error: { code: "INVALID_INDEX" } });
    expect(statusOf(r2, "placement:2")).toMatchObject({ status: "error", error: { code: "UNKNOWN_MESH" } });
    expect(r2.changes).toEqual([]);
    const r3 = await h.send([
      { kind: "edit", rev: rev(), id: 2, payload: [{ op: "removeTriangles", triangles: [0] }], qid: 7 },
    ]);
    expect(replyOf(r3, 7)).toMatchObject({ ok: false, error: { code: "UNKNOWN_MESH" } });
    expect(statusOf(r3, "edit:2")).toMatchObject({ status: "error", error: { code: "UNKNOWN_MESH" } });
  });

  it("coalesces_last_wins_keys_only", async () => {
    const h = harness();
    const a = sphere([0, 0, 0]);
    const b = sphere([3, 0, 0]);
    const plane = (z: number) => ({ origin: [0, 0, z] as Vec3, normal: [0, 0, 1] as Vec3 });
    const at = (x: number) => ({ rotation: IDENTITY, translation: [x, 0, 0] });

    const lastColoring: Request = {
      kind: "coloring",
      rev: 0,
      id: 1,
      rule: { field: 1, map: "turbo", range: [0, 2] },
      key: "display:1",
    };
    const batch: Request[] = [
      addMesh(1, a),
      addMesh(2, b),
      { kind: "field", rev: rev(), field: 1, op: "add", desc: { name: "d", on: 1, to: 2, maxDistance: 5 } },
      // An invalid colouring superseded by a valid one: never run.
      { kind: "coloring", rev: rev(), id: 1, rule: { field: 1, map: "turbo", range: [2, 0] }, key: "display:1" },
      { ...lastColoring, rev: rev() } as Request,
      // A drag whose invalid move is superseded by a valid one.
      { kind: "drag", rev: rev(), id: 1, phase: "begin", key: "placement:1" },
      {
        kind: "drag",
        rev: rev(),
        id: 1,
        phase: "move",
        rotation: [0, 0, 0, 0],
        translation: [9, 9, 9],
        key: "placement:1",
      },
      { kind: "drag", rev: rev(), id: 1, phase: "move", ...at(0.5), key: "placement:1" },
      { kind: "drag", rev: rev(), id: 1, phase: "end", key: "placement:1" },
      // Section plane changes: only the last one runs.
      { kind: "section", rev: rev(), section: 1, op: "add", desc: { plane: plane(0.1), meshes: [1, 2] } },
      {
        kind: "section",
        rev: rev(),
        section: 1,
        op: "set",
        desc: { plane: { origin: [0, 0, 0], normal: [0, 0, 0] } },
        key: "section:1:plane",
      },
      { kind: "section", rev: rev(), section: 1, op: "set", desc: { plane: plane(0.2) }, key: "section:1:plane" },
      // Edits are never coalesced: the invalid one still runs and fails.
      { kind: "edit", rev: rev(), id: 1, payload: [{ op: "removeTriangles", triangles: [99999] }], qid: 10 },
      { kind: "edit", rev: rev(), id: 1, payload: [{ op: "removeTriangles", triangles: [0] }], qid: 11 },
    ];
    const r1 = await h.send(batch);
    expect(r1.rev).toBe(batch[batch.length - 1].rev);
    // Superseded requests never ran: no status (and no error) of theirs.
    expect(statusesOf(r1, "display:1")).toEqual([[batch[4].rev, "ready", undefined]]);
    expect(statusesOf(r1, "placement:1")).toEqual([
      [batch[5].rev, "ready", undefined],
      [batch[7].rev, "ready", undefined],
      [batch[8].rev, "ready", undefined],
    ]);
    expect(statusesOf(r1, "section:1:plane")).toEqual([[batch[11].rev, "ready", undefined]]);
    expect(replyOf(r1, 10)).toMatchObject({ ok: false, error: { code: "INVALID_INDEX" } });
    expect(replyOf(r1, 11)).toMatchObject({ ok: true });
    expect(statusesOf(r1, "edit:1")).toEqual([
      [batch[12].rev, "error", "INVALID_INDEX"],
      [batch[13].rev, "ready", undefined],
    ]);
    expect(r1.placements.find((p) => p.id === 1)!.translation).toEqual([0.5, 0, 0]);

    // Moves of separate drags are not merged: each drag is its own step.
    const r2 = await h.send([
      { kind: "drag", rev: rev(), id: 1, phase: "begin", key: "placement:1" },
      { kind: "drag", rev: rev(), id: 1, phase: "move", ...at(1), key: "placement:1" },
      { kind: "drag", rev: rev(), id: 1, phase: "end", key: "placement:1" },
      { kind: "drag", rev: rev(), id: 1, phase: "begin", key: "placement:1" },
      { kind: "drag", rev: rev(), id: 1, phase: "move", ...at(2), key: "placement:1" },
      { kind: "drag", rev: rev(), id: 1, phase: "end", key: "placement:1" },
    ]);
    expect(r2.placements.find((p) => p.id === 1)!.translation).toEqual([2, 0, 0]);
    const r3 = await h.send([{ kind: "undo", rev: rev() }]);
    expect(r3.placements.find((p) => p.id === 1)!.translation).toEqual([1, 0, 0]);
  });

  it("section_and_placement_mirrors_delivered", async () => {
    const h = harness();
    const ref = scene();
    const a = sphere([0, 0, 0]);
    const b = sphere([3, 0, 0]);
    const plane = { origin: [0, 0, 0.1] as Vec3, normal: [0, 0, 1] as Vec3 };

    const r1 = await h.send([
      addMesh(1, a),
      addMesh(2, b),
      { kind: "section", rev: rev(), section: 5, op: "add", desc: { plane, meshes: [1, 2], fill: true } },
    ]);
    const ra = ref.addMesh(a.positions, a.indices);
    const rb = ref.addMesh(b.positions, b.indices);
    const sec = ref.addSection({ plane, meshes: [ra, rb], fill: true });
    const entry = (r: Result, mesh: ClientId) => r.sections.find((x) => x.section === 5 && x.mesh === mesh);
    expect(entry(r1, 1)!.result!.contours.length).toBeGreaterThan(0);
    expect(entry(r1, 1)!.result!.fill).toBeDefined();
    expectSection(entry(r1, 1)!.result, ref.section(sec, ra));
    expectSection(entry(r1, 2)!.result, ref.section(sec, rb));
    expect(r1.changes.find((c) => c.id === 1)!.sections).toEqual([5]);

    // Moving a mesh delivers its placement, matrix and new section result.
    const r2 = await h.send([
      { kind: "setPlacement", rev: rev(), id: 2, rotation: QUARTER_Z, translation: [0, 0, 0.3], key: "placement:2" },
    ]);
    ref.setPlacement(rb, { rotation: QUARTER_Z, translation: [0, 0, 0.3] });
    const p2 = r2.placements.find((p) => p.id === 2)!;
    expect(p2.matrix4).toEqual(Array.from(ref.matrix4(rb)));
    expect(p2.rotation).toEqual(ref.placement(rb).rotation);
    expect(r2.placements.find((p) => p.id === 1)).toBeUndefined();
    expectSection(entry(r2, 2)!.result, ref.section(sec, rb));

    // A section dropping a mesh, then removed: the gone entries carry null.
    const r3 = await h.send([
      { kind: "section", rev: rev(), section: 5, op: "set", desc: { meshes: [1] }, key: "section:5:meshes" },
    ]);
    expect(entry(r3, 2)?.result ?? null).toBeNull();
    const r4 = await h.send([{ kind: "section", rev: rev(), section: 5, op: "remove" }]);
    for (const x of r4.sections) {
      expect(x.result).toBeNull();
    }
    expect(statusOf(r4, "section:5")).toMatchObject({ status: "ready" });
  });

  it("errors_cross_boundary_with_codes", async () => {
    const h = harness();
    const r = await h.send([
      addMesh(1, sphere([0, 0, 0])),
      { kind: "query", rev: rev(), qid: 1, name: "meshDistance", args: { a: 1, b: 99 } },
      { kind: "coloring", rev: rev(), id: 1, rule: { field: 42, map: "turbo", range: [0, 1] }, key: "display:1" },
      { kind: "section", rev: rev(), section: 7, op: "set", desc: { fill: true }, key: "section:7:fill" },
      { kind: "edit", rev: rev(), id: 1, payload: [{ op: "removeTriangles", triangles: [999] }], qid: 2 },
      { kind: "fetchFieldValues", rev: rev(), qid: 3, field: 5 },
      { kind: "query", rev: rev(), qid: 4, name: "nope", args: {} },
      { kind: "query", rev: rev(), qid: 5, name: "raycast", args: { origin: [-5, 0, 0], direction: [0, 0, 0] } },
      // The queue goes on after errors.
      { kind: "setPlacement", rev: rev(), id: 1, rotation: IDENTITY, translation: [1, 0, 0], key: "placement:1" },
      { kind: "query", rev: rev(), qid: 6, name: "raycast", args: { origin: [-5, 0, 0], direction: [1, 0, 0] } },
    ]);
    expect(replyOf(r, 1)).toMatchObject({ ok: false, error: { code: "UNKNOWN_MESH" } });
    expect(statusOf(r, "display:1")).toMatchObject({ status: "error", error: { code: "UNKNOWN_FIELD" } });
    expect(statusOf(r, "section:7:fill")).toMatchObject({ status: "error", error: { code: "UNKNOWN_SECTION" } });
    expect(replyOf(r, 2)).toMatchObject({ ok: false, error: { code: "INVALID_INDEX" } });
    expect(statusOf(r, "edit:1")).toMatchObject({ status: "error", error: { code: "INVALID_INDEX" } });
    expect(replyOf(r, 3)).toMatchObject({ ok: false, error: { code: "UNKNOWN_FIELD" } });
    expect(replyOf(r, 4)).toMatchObject({ ok: false, error: { code: "INVALID_ARGUMENT" } });
    expect(replyOf(r, 5)).toMatchObject({ ok: false, error: { code: "INVALID_ARGUMENT" } });
    expect(typeof replyOf(r, 5)!.error!.message).toBe("string");
    expect(statusOf(r, "placement:1")).toMatchObject({ status: "ready" });
    expect(r.placements.find((p) => p.id === 1)!.translation).toEqual([1, 0, 0]);
    // A query result names meshes by client id.
    expect(replyOf(r, 6)).toMatchObject({ ok: true, value: { mesh: 1 } });
  });

  it("trap_posts_poisoned", async () => {
    const h = harness({ testPanic: true });
    const r1 = await h.send([
      addMesh(1, sphere([0, 0, 0])),
      { kind: "query", rev: rev(), qid: 1, name: "analyze", args: { mesh: 1 } },
      { kind: "setPlacement", rev: rev(), id: 1, rotation: IDENTITY, translation: [1, 0, 0], key: "placement:1" },
      { kind: "query", rev: rev(), qid: 2, name: "intersects", args: { a: 1, b: 1 } },
    ]);
    expect(r1.poisoned).toBe(true);
    expect(replyOf(r1, 1)).toMatchObject({ ok: false, error: { code: "INSTANCE_POISONED" } });
    expect(replyOf(r1, 2)).toMatchObject({ ok: false, error: { code: "INSTANCE_POISONED" } });
    expect(statusOf(r1, "placement:1")).toMatchObject({ status: "error", error: { code: "INSTANCE_POISONED" } });
    expect(r1.changes).toEqual([]);

    // The engine has stopped: later batches fail without running.
    const last = rev();
    const r2 = await h.send([{ kind: "query", rev: last, qid: 3, name: "intersects", args: { a: 1, b: 1 } }]);
    expect(r2.poisoned).toBe(true);
    expect(r2.rev).toBe(last);
    expect(replyOf(r2, 3)).toMatchObject({ ok: false, error: { code: "INSTANCE_POISONED" } });
  });
});
