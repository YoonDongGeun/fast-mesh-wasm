// The main-thread `WorkerScene` against the Worker engine in the Node
// `worker_threads` harness (built dist/, `serve` mode with the test-only
// `debugDump` query), through the `createWorker` option: mirrors, coalescing,
// optimistic drags, async queries, statuses, the three.js adapter, poisoning
// and the package layout of `fast-mesh-wasm/worker`.
import { execSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { Worker } from "node:worker_threads";
import * as THREE from "three";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FastMeshError, type MeshChange, type MeshHandle, Scene, type Vec3 } from "../src/node.js";
import { bindMesh, bindSection } from "../src/three/index.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";
import { debugDump, FALLBACK_FLUSH_MS } from "../src/worker/client.js";
import type { MeshChangeWire, Request } from "../src/worker/protocol.js";

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
    // Batches only: not the init message.
    if (Array.isArray(msg)) {
      this.sent.push(msg as Request[]);
    }
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

  /** The requests of the last batch sent. */
  last(): Request[] {
    return this.sent[this.sent.length - 1];
  }
}

const opened: WorkerScene[] = [];
afterEach(() => {
  for (const s of opened.splice(0)) {
    s.terminate();
  }
});

async function start(workerData?: { testPanic?: boolean }): Promise<{ scene: WorkerScene; w: NodeWorker }> {
  let w: NodeWorker | undefined;
  const scene = await createWorkerScene({
    createWorker: () => {
      w = new NodeWorker(workerData);
      return w;
    },
  });
  opened.push(scene);
  return { scene, w: w as NodeWorker };
}

/** Sends this frame and waits until every result so far is in. */
async function settle(scene: WorkerScene): Promise<void> {
  // Queued first, so this frame's batch carries it.
  const barrier = scene.raycast([0, 0, 1e6], [0, 0, 1]);
  scene.commitFrame();
  await barrier;
}

function bytes(v: ArrayBufferView | null): Buffer | null {
  return v === null ? null : Buffer.from(v.buffer, v.byteOffset, v.byteLength);
}

/** The mirrors of `mesh` equal the Worker engine's buffers byte for byte, and its matrix. */
async function expectMirrored(scene: WorkerScene, mesh: MeshHandle): Promise<void> {
  // The values mirror is opt-in: turned on (if it was not) in the same batch.
  scene.views(mesh).values?.();
  const pending = debugDump(scene, mesh);
  // Sent now, not by the fallback timer (this also delivers pending records).
  scene.commitFrame();
  const dump = await pending;
  const v = scene.views(mesh);
  for (const name of ["positions", "normals", "colors", "values", "indices"] as const) {
    expect(dump[name], `${name} dumped`).not.toBeUndefined();
    const want = bytes(dump[name]);
    const got = bytes(v[name]?.() ?? null);
    if (want === null || got === null) {
      expect(got, name).toBe(want);
    } else {
      expect(got.equals(want), `${name} bytes`).toBe(true);
    }
  }
  expect(Array.from(scene.matrix4(mesh))).toEqual(dump.matrix4);
  expect(scene.placement(mesh)).toEqual(dump.placement);
}

/** A deterministic PRNG in [0, 1). */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("WorkerScene", () => {
  it("multi_target_field_matches_local_scene", async () => {
    const { scene } = await start();
    const local = new Scene();
    const centres: Vec3[] = [
      [0, 0, 0],
      [1.5, 0, 0],
      [-1.2, 0.3, 0],
    ];
    const [a, b, c] = centres.map((p) => scene.addMesh(sphere(p).positions, sphere(p).indices));
    const [la, lb, lc] = centres.map((p) => local.addMesh(sphere(p).positions, sphere(p).indices));
    const field = scene.addDistanceField({ name: "m", on: a, to: [b, c], maxDistance: 1 });
    const want = local.addDistanceField({ name: "m", on: la, to: [lb, lc], maxDistance: 1 });
    scene.setPlacement(c, { rotation: IDENTITY, translation: [0.2, 0, 0] });
    local.setPlacement(lc, { rotation: IDENTITY, translation: [0.2, 0, 0] });
    await settle(scene);
    expect(Array.from(await scene.fetchFieldValues(field))).toEqual(Array.from(local.fieldValues(want)));
    const nearest = await scene.fetchFieldNearest(field);
    expect(await scene.fieldTargetStatus(field)).toEqual(local.fieldTargetStatus(want));
    expect((await scene.fieldStatus(field)).sign).toEqual(local.fieldStatus(want).sign);
    expect(Array.from(nearest)).toEqual(Array.from(local.fieldNearest(want)));
    expect(Array.from(nearest)).toContain(1);
    // A removed target leaves the field with the others.
    scene.removeMesh(c);
    local.removeMesh(lc);
    await settle(scene);
    expect(Array.from(await scene.fetchFieldValues(field))).toEqual(Array.from(local.fieldValues(want)));
    expect(Array.from(await scene.fetchFieldNearest(field))).toEqual(Array.from(local.fieldNearest(want)));
    local.free();
  });

  it("values_mirror_is_opt_in", async () => {
    const { scene, w } = await start();
    const sent: (ArrayBuffer | null)[] = [];
    w.worker.on("message", (data: { changes?: MeshChangeWire[] }) => {
      for (const c of data.changes ?? []) {
        sent.push(c.bytes.values);
      }
    });
    const a = scene.addMesh(sphere([0, 0, 0]).positions, sphere([0, 0, 0]).indices);
    const b = scene.addMesh(sphere([2.5, 0, 0]).positions, sphere([2.5, 0, 0]).indices);
    const field = scene.addDistanceField({ name: "d", on: a, to: b, maxDistance: 3 });
    scene.setColoring(a, { field, map: "turbo", range: [0, 2] });
    await settle(scene);
    scene.commitFrame();
    // Vertex shading never reads values(): no value bytes cross.
    expect(sent.length).toBeGreaterThan(0);
    expect(sent.every((v) => v === null)).toBe(true);
    // The first read turns the mirror on; the next result fills it whole.
    expect(scene.views(a).values!()).toBeNull();
    const records: MeshChange[] = [];
    scene.onChanges((c) => records.push(...c));
    await settle(scene);
    scene.commitFrame();
    expect(records.find((r) => r.mesh === a)?.colors).toBe("all");
    expect(scene.views(a).values!()).not.toBeNull();
    expect(scene.views(b).values!()).toBeNull();
    await expectMirrored(scene, a);
    // Then it follows the colours' ranges.
    scene.setPlacement(b, { rotation: IDENTITY, translation: [0.3, 0, 0] });
    await settle(scene);
    await expectMirrored(scene, a);
  });

  it("coloring_follows_the_applied_rule", async () => {
    const { scene, w } = await start();
    const a = scene.addMesh(sphere([0, 0, 0]).positions, sphere([0, 0, 0]).indices);
    const b = scene.addMesh(sphere([2.2, 0, 0]).positions, sphere([2.2, 0, 0]).indices);
    // Every value finite, or most of them far: the values tell the fields apart.
    const whole = scene.addDistanceField({ name: "whole", on: a, to: b, maxDistance: Infinity });
    const near = scene.addDistanceField({ name: "near", on: a, to: b, maxDistance: 0.4 });
    const first = { field: whole, map: "turbo" as const, range: [0, 2] as const };
    scene.setColoring(a, first);
    scene.views(a).values!();
    await settle(scene);
    scene.commitFrame();
    expect(scene.coloring(a)?.field).toBe(whole);
    /** The rule shown matches the mirrored values. */
    const consistent = () => {
      const values = scene.views(a).values!();
      if (values === null) {
        return scene.coloring(a) === null;
      }
      const far = values.some((v) => v === Infinity);
      return scene.coloring(a)?.field === (far ? near : whole);
    };
    const seen: boolean[] = [];
    scene.onChanges((changes) => {
      if (changes.some((c) => c.mesh === a)) {
        seen.push(consistent());
      }
    });
    // A drag result in flight, then a rule on the other field.
    w.hold = true;
    const d = scene.drag(a);
    d.move({ rotation: [0, 0, 0, 1], translation: [0.1, 0, 0] });
    scene.commitFrame();
    scene.setColoring(a, { field: near, map: "turbo", range: [0, 0.4] });
    // Requested, not shown yet.
    expect(scene.coloring(a)?.field).toBe(whole);
    scene.commitFrame();
    await w.waitHeld(2);
    w.release(1);
    scene.commitFrame();
    expect(scene.coloring(a)?.field).toBe(whole);
    w.release(1);
    w.hold = false;
    scene.commitFrame();
    expect(scene.coloring(a)?.field).toBe(near);
    d.end();
    await settle(scene);
    scene.commitFrame();
    expect(seen.length).toBeGreaterThanOrEqual(2);
    expect(seen.every((x) => x)).toBe(true);

    // A rejected rule leaves the shown one.
    const before = scene.coloring(a);
    expect(Object.isFrozen(before)).toBe(true);
    expect(Object.isFrozen(before?.range)).toBe(true);
    scene.setColoring(a, { field: whole, map: "turbo", range: [1, 0] });
    await settle(scene);
    scene.commitFrame();
    expect(scene.status(`display:${scene.idOf(a)}`)).toMatchObject({ status: "error" });
    expect(scene.coloring(a)).toBe(before);
    // The colouring of a removed field goes when the removal is applied.
    scene.removeField(near);
    expect(scene.coloring(a)).toBe(before);
    await settle(scene);
    scene.commitFrame();
    expect(scene.coloring(a)).toBeNull();
  });

  it("views_mirror_engine_after_edits", async () => {
    const { scene } = await start();
    const a = scene.addMesh(sphere([0, 0, 0]).positions, sphere([0, 0, 0]).indices);
    const b = scene.addMesh(sphere([2.5, 0, 0]).positions, sphere([2.5, 0, 0]).indices, { normals: false });
    const field = scene.addDistanceField({ name: "d", on: a, to: b, maxDistance: 3 });
    const rule = { field, map: "turbo" as const, range: [0, 2] as const };
    scene.setColoring(a, rule);
    // The values mirror is opt-in: on from the start here.
    scene.views(a).values!();
    await settle(scene);
    await expectMirrored(scene, a);
    await expectMirrored(scene, b);

    const random = rng(7);
    const pick = (n: number) => Math.floor(random() * n);
    const ignore = () => {};
    for (let step = 0; step < 40; step++) {
      const ops = 1 + pick(3);
      for (let k = 0; k < ops; k++) {
        const vs = scene.views(a).positions().length / 3;
        const ts = scene.views(a).indices().length / 3;
        switch (pick(9)) {
          case 0: {
            const vertices = [pick(vs), pick(vs)].filter((x, i, l) => l.indexOf(x) === i);
            const positions = vertices.flatMap(() => [random() - 0.5, random() - 0.5, random() - 0.5]);
            scene.edit(a, [{ op: "setPositions", vertices, positions }]).catch(ignore);
            break;
          }
          case 1:
            scene.edit(a, [{ op: "removeTriangles", triangles: [pick(ts)] }]).catch(ignore);
            break;
          case 2:
            scene
              .edit(a, [
                { op: "addVertices", positions: [0, 0, 1.2, 0.1, 0, 1.2, 0, 0.1, 1.2] },
                { op: "addTriangles", triangles: [{ ref: [0, 0] }, { ref: [0, 1] }, { ref: [0, 2] }] },
              ])
              .catch(ignore);
            break;
          case 3:
            scene.setPlacement(b, { rotation: IDENTITY, translation: [random() - 0.5, random() - 0.5, 0] });
            break;
          case 4: {
            const d = scene.drag(a);
            d.move({ rotation: [0, 0, Math.sin(random()), 1], translation: [random(), 0, 0] });
            if (random() < 0.5) {
              d.end();
            } else {
              d.cancel();
            }
            break;
          }
          case 5:
            scene.undo();
            break;
          case 6:
            scene.redo();
            break;
          case 7:
            scene.compact(a);
            break;
          case 8:
            scene.setColoring(a, random() < 0.3 ? null : rule);
            break;
        }
      }
      await settle(scene);
      scene.commitFrame();
      await expectMirrored(scene, a);
      await expectMirrored(scene, b);
    }
  });

  it("toggle_storm_coalesces", async () => {
    const { scene, w } = await start();
    const s = sphere([0, 0, 0], 1, 32, 64);
    const t = sphere([2.2, 0, 0], 1, 32, 64);
    const a = scene.addMesh(s.positions, s.indices);
    const b = scene.addMesh(t.positions, t.indices);
    await settle(scene);

    // A field build and 50 alternating colouring toggles in one frame.
    const field = scene.addDistanceField({ name: "d", on: a, to: b, maxDistance: Infinity });
    const rule = { field, map: "viridis" as const, range: [0, 3] as const };
    for (let i = 0; i < 50; i++) {
      scene.setColoring(a, i % 2 === 1 ? rule : null);
    }
    const key = `display:${scene.idOf(a)}`;
    expect(scene.status(key).status).toBe("computing");
    const latest = scene.status(key).revision;
    scene.commitFrame();
    const batch = w.last();
    expect(batch.filter((r) => r.kind === "field")).toHaveLength(1);
    const colorings = batch.filter((r) => r.kind === "coloring");
    expect(colorings).toHaveLength(1);
    expect(colorings[0]).toMatchObject({ rev: latest, rule: { field: scene.idOf(field), map: "viridis" } });

    await settle(scene);
    const changes = scene.commitFrame();
    expect(scene.status(key)).toEqual({ status: "ready", revision: latest });
    // At most one recolour reached the mirrors.
    expect(changes.filter((c) => c.mesh === a && c.colors !== "all" && c.colors.length > 0)).toHaveLength(0);
    expect(scene.views(a).colors()).not.toBeNull();
    await expectMirrored(scene, a);

    // The storm ending on "off" leaves colouring off.
    for (let i = 0; i < 50; i++) {
      scene.setColoring(a, i % 2 === 0 ? rule : null);
    }
    await settle(scene);
    expect(w.last().filter((r) => r.kind === "coloring")).toHaveLength(1);
    expect(scene.views(a).colors()).toBeNull();
  });

  it("optimistic_drag_not_snapped_back", async () => {
    const { scene, w } = await start();
    const s = sphere([0, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    await settle(scene);
    scene.commitFrame();
    const at = (x: number) => ({ rotation: IDENTITY, translation: [x, 0, 0] as Vec3 });

    w.hold = true;
    // An older placement in flight...
    scene.setPlacement(a, at(1));
    scene.commitFrame();
    // ...then a drag that moves the mesh before its result arrives.
    const d = scene.drag(a);
    d.move(at(5));
    let changes = scene.commitFrame();
    expect(changes.find((c) => c.mesh === a)?.placementChanged).toBe(true);
    expect(scene.placement(a).translation).toEqual([5, 0, 0]);
    expect(scene.matrix4(a)[12]).toBe(5);
    // Pointer moves within a frame: only the latest is sent.
    d.move(at(5.5));
    d.move(at(6));
    expect(scene.placement(a).translation).toEqual([6, 0, 0]);

    await w.waitHeld(2);
    // The placement result (x = 1) arrives: the pose is not snapped back.
    w.release(1);
    expect(scene.placement(a).translation).toEqual([6, 0, 0]);
    changes = scene.commitFrame();
    const moves = w.last().filter((r) => r.kind === "drag" && r.phase === "move");
    expect(moves).toHaveLength(1);
    expect(moves[0]).toMatchObject({ translation: [6, 0, 0] });
    expect(scene.matrix4(a)[12]).toBe(6);
    w.release();
    expect(scene.placement(a).translation).toEqual([6, 0, 0]);

    d.end();
    expect(d.active).toBe(false);
    scene.commitFrame();
    await w.waitHeld(2);
    // The move result does not snap it back either; the end result hands over.
    w.release();
    w.hold = false;
    await settle(scene);
    expect(scene.placement(a).translation).toEqual([6, 0, 0]);
    expect(scene.canUndo).toBe(true);
    await expectMirrored(scene, a);

    // Cancel puts the start pose back at once and records nothing.
    const d2 = scene.drag(a);
    d2.move(at(9));
    expect(scene.placement(a).translation).toEqual([9, 0, 0]);
    d2.cancel();
    expect(scene.placement(a).translation).toEqual([6, 0, 0]);
    expect(() => d2.move(at(1))).toThrow(FastMeshError);
    await settle(scene);
    expect(scene.placement(a).translation).toEqual([6, 0, 0]);
    await expectMirrored(scene, a);
    scene.undo();
    await settle(scene);
    // The undo undid the ended drag, not the cancelled one.
    expect(scene.placement(a).translation).toEqual([1, 0, 0]);
    await expectMirrored(scene, a);

    // One drag at a time.
    const d3 = scene.drag(a);
    expect(() => scene.drag(a)).toThrow(/already active/);
    d3.end();
  });

  it("async_queries_resolve_and_reject_with_codes", async () => {
    const { scene } = await start();
    const s = sphere([0, 0, 0]);
    const t = sphere([3, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    const b = scene.addMesh(t.positions, t.indices);
    // Queries queue behind the adds, without a commitFrame.
    const hit = await scene.raycast([-5, 0, 0], [1, 0, 0]);
    expect(hit?.mesh).toBe(a);
    expect(hit?.distance).toBeCloseTo(4, 1);
    expect((await scene.closestPoint([5, 0, 0]))?.mesh).toBe(b);
    expect((await scene.closestPoint([5, 0, 0], { mesh: a }))?.mesh).toBe(a);
    const gap = await scene.meshDistance(a, b);
    expect(gap?.distance).toBeCloseTo(1, 1);
    expect(await scene.intersects(a, b)).toBe(false);
    const contacts = await scene.findContacts({ margin: 2 });
    expect(contacts).toHaveLength(1);
    expect(contacts[0].a).toBe(a);
    expect(contacts[0].b).toBe(b);
    expect(contacts[0].trianglesA).toBeInstanceOf(Uint32Array);
    const analysis = await scene.analyze(a);
    expect(analysis.loops).toEqual([]);
    const added = await scene.edit(a, [{ op: "addVertices", positions: [0, 0, 5] }]);
    expect(Array.from(added.added[0])).toEqual([s.positions.length / 3]);
    const report = await scene.repair(a, { op: "removeDegenerate", removeUnusedVertices: true });
    expect(report.removedVertices).toBe(1);
    const field = scene.addDistanceField({ name: "d", on: a, to: b, maxDistance: 5 });
    const values = await scene.fetchFieldValues(field);
    expect(values).toBeInstanceOf(Float32Array);
    expect(values.length).toBe(scene.views(a).positions().length / 3);
    expect((await scene.fieldStatus(field)).current).toBe(true);

    const code = async (p: Promise<unknown>) => {
      const e = await p.then(
        () => undefined,
        (x: unknown) => x,
      );
      expect(e).toBeInstanceOf(FastMeshError);
      return (e as FastMeshError).code;
    };
    // Engine errors keep their codes.
    expect(await code(scene.edit(a, [{ op: "removeTriangles", triangles: [99999] }]))).toBe("INVALID_INDEX");
    expect(await code(scene.raycast([0, 0, 0], [0, 0, 0]))).toBe("INVALID_ARGUMENT");
    // A failed add: its status says why, and later calls give UNKNOWN_MESH.
    const bad = scene.addMesh([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 7]);
    expect(await code(scene.intersects(bad, a))).toBe("UNKNOWN_MESH");
    expect(scene.status(`mesh:${scene.idOf(bad)}`)).toMatchObject({
      status: "error",
      error: { code: "INVALID_INDEX" },
    });
    // Handle checks here: removed, or not of this scene.
    scene.removeMesh(b);
    expect(await code(scene.intersects(a, b))).toBe("UNKNOWN_MESH");
    expect(() => scene.setPlacement(b, { rotation: IDENTITY, translation: [0, 0, 0] })).toThrow(/removed/);
    const other = new Scene();
    const foreign = other.addMesh(s.positions, s.indices);
    expect(await code(scene.analyze(foreign))).toBe("INVALID_ARGUMENT");
    other.free();
    // Terminated: pending and later calls fail.
    const pending = scene.analyze(a);
    scene.terminate();
    expect(await code(pending)).toBe("INVALID_ARGUMENT");
    expect(await code(scene.analyze(a))).toBe("INVALID_ARGUMENT");
    expect(() => scene.commitFrame()).toThrow(/terminated/);
  });

  it("three_adapter_works_on_worker_scene", async () => {
    const { scene } = await start();
    const s = sphere([0, 0, 0]);
    const t = sphere([2.5, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    const b = scene.addMesh(t.positions, t.indices);
    // Bound before the Worker has the mesh: no slots yet.
    const bound = bindMesh(scene, a);
    const geometry = bound.object.geometry;
    expect(geometry.getAttribute("position").array.length).toBe(0);
    const section = scene.addSection({ plane: { origin: [0, 0, 0.1], normal: [0, 0, 2] }, meshes: [a, b] });
    expect(scene.sectionPlane(section).normal).toEqual([0, 0, 1]);
    const boundB = bindMesh(scene, b);
    const drawn = bindSection(scene, section, [bound, boundB]);

    const expectInSync = async () => {
      const dump = await debugDump(scene, a);
      expect(Array.from(geometry.getAttribute("position").array)).toEqual(Array.from(dump.positions));
      expect(Array.from(geometry.getIndex()!.array)).toEqual(Array.from(dump.indices));
      expect(Array.from(geometry.getAttribute("normal").array)).toEqual(Array.from(dump.normals!));
      expect(bound.object.matrix.toArray()).toEqual(dump.matrix4);
    };
    await settle(scene);
    scene.commitFrame();
    await expectInSync();
    expect(geometry.boundingSphere!.radius).toBeCloseTo(1, 3);
    const lines = drawn.group.children[0] as THREE.LineSegments;
    expect(lines.geometry.getAttribute("position").count).toBeGreaterThan(0);

    // Edits, colouring and placements flow through the change stream.
    const field = scene.addDistanceField({ name: "d", on: a, to: b, maxDistance: 3 });
    scene.setColoring(a, { field, map: "turbo", range: [0, 2] });
    scene.edit(a, [{ op: "setPositions", vertices: [0], positions: [0, 0, 1.5] }]);
    scene.setPlacement(a, { rotation: [0, 0, 0.3826834, 0.9238795], translation: [0.2, 0, 0] });
    await settle(scene);
    scene.commitFrame();
    await expectInSync();
    expect(geometry.getAttribute("color")).toBeDefined();
    expect((bound.object.material as THREE.MeshStandardMaterial).vertexColors).toBe(true);

    // An optimistic drag moves the object at once.
    const d = scene.drag(a);
    d.move({ rotation: IDENTITY, translation: [1, 2, 3] });
    scene.commitFrame();
    expect(bound.object.matrix.toArray()).toEqual(Array.from(scene.matrix4(a)));
    expect(bound.object.matrix.elements[13]).toBe(2);
    d.end();
    await settle(scene);
    scene.commitFrame();
    await expectInSync();

    // A growth (slot count change) recreates the attributes.
    await scene.edit(a, [{ op: "addVertices", positions: new Array(300).fill(0.25) }]);
    scene.commitFrame();
    expect(geometry.getAttribute("position").array.length).toBe(scene.views(a).positions().length);
    await expectInSync();

    // Removal detaches the binding; the section stops drawing on removal.
    scene.removeSection(section);
    scene.removeMesh(a);
    await settle(scene);
    const last = scene.commitFrame();
    expect(last.find((c) => c.mesh === a)?.removed).toBe(true);
    expect(() => scene.views(a)).toThrow(/removed/);
    expect(lines.parent).toBe(drawn.group);
    expect((drawn.group.children[0] as THREE.LineSegments).geometry.getAttribute("position")).toBeUndefined();
    bound.dispose();
    boundB.dispose();
    drawn.dispose();
  });

  it("status_computing_then_ready", async () => {
    const { scene, w } = await start();
    const s = sphere([0, 0, 0]);
    const t = sphere([3, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    const b = scene.addMesh(t.positions, t.indices);
    const id = scene.idOf(a);
    expect(scene.status(`mesh:${id}`).status).toBe("computing");
    expect(scene.status("nothing")).toEqual({ status: "idle", revision: 0 });
    let notified = 0;
    const off = scene.onStatus(() => notified++);
    await settle(scene);
    expect(scene.status(`mesh:${id}`).status).toBe("ready");
    expect(notified).toBeGreaterThan(0);

    const field = scene.addDistanceField({ name: "d", on: a, to: b, maxDistance: 5 });
    const key = `display:${id}`;
    w.hold = true;
    scene.setColoring(a, { field, map: "turbo", range: [0, 2] });
    const first = scene.status(key).revision;
    scene.commitFrame();
    scene.setColoring(a, { field, map: "turbo", range: [2, 0] });
    const second = scene.status(key).revision;
    expect(second).toBeGreaterThan(first);
    scene.commitFrame();
    await w.waitHeld(2);
    // The first result: a newer request for the key is still pending.
    w.release(1);
    expect(scene.status(key)).toEqual({ status: "computing", revision: second });
    w.release(1);
    w.hold = false;
    const st = scene.status(key);
    expect(st.status).toBe("error");
    expect(st.revision).toBe(second);
    expect(st.error).toBeInstanceOf(FastMeshError);
    expect(st.error!.code).toBe("INVALID_ARGUMENT");

    scene.setColoring(a, null);
    expect(scene.status(key).status).toBe("computing");
    await settle(scene);
    expect(scene.status(key).status).toBe("ready");

    // History state follows results.
    expect(scene.canUndo).toBe(false);
    const before = notified;
    scene.setPlacement(a, { rotation: IDENTITY, translation: [1, 0, 0] });
    expect(scene.status(`placement:${id}`).status).toBe("computing");
    await settle(scene);
    await Promise.resolve();
    expect(scene.status(`placement:${id}`).status).toBe("ready");
    expect(scene.canUndo).toBe(true);
    expect(notified).toBeGreaterThan(before);
    off();
  });

  it("poisoned_result_rejects_pending", async () => {
    // The harness makes `analyze` trap (test build of the wasm only).
    const { scene } = await start({ testPanic: true });
    const s = sphere([0, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    await settle(scene);
    const views = scene.views(a).positions();
    // A drag held across the trap is let go.
    const d = scene.drag(a);
    d.move({ rotation: IDENTITY, translation: [4, 0, 0] });
    scene.commitFrame();
    scene.setPlacement(a, { rotation: IDENTITY, translation: [1, 0, 0] });
    const analysis = scene.analyze(a);
    scene.commitFrame();
    await expect(analysis).rejects.toMatchObject({ code: "INSTANCE_POISONED" });
    expect(d.active).toBe(false);
    // Nothing of the batch applied; the status says why.
    expect(scene.placement(a).translation).toEqual([4, 0, 0]);
    expect(scene.status(`placement:${scene.idOf(a)}`)).toMatchObject({
      status: "error",
      error: { code: "INSTANCE_POISONED" },
    });
    expect(scene.views(a).positions()).toBe(views);
    // Later calls fail the same way.
    await expect(scene.intersects(a, a)).rejects.toMatchObject({ code: "INSTANCE_POISONED" });
    scene.compact(a);
    scene.commitFrame();
    expect(scene.status(`edit:${scene.idOf(a)}`)).toMatchObject({ status: "error" });
  });


  it("frame_queries_wait_for_commit", async () => {
    const { scene, w } = await start();
    const s = sphere([0, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    await settle(scene);
    const sent = w.sent.length;
    // One frame of pointer events: a drag move and a hover raycast each.
    const d = scene.drag(a);
    const hits: Promise<unknown>[] = [];
    for (let i = 0; i < 20; i++) {
      d.move({ rotation: IDENTITY, translation: [i * 0.1, 0, 0] });
      hits.push(scene.raycast([-5, 0, 0], [1, 0, 0]));
    }
    await Promise.resolve();
    await Promise.resolve();
    // Nothing goes before the frame ends.
    expect(w.sent.length).toBe(sent);
    scene.commitFrame();
    expect(w.sent.length).toBe(sent + 1);
    const batch = w.last();
    expect(batch.filter((r) => r.kind === "drag" && r.phase === "move")).toHaveLength(1);
    expect(batch.filter((r) => r.kind === "query")).toHaveLength(20);
    // Queries keep their place: after the begin, in call order.
    expect(batch[0]).toMatchObject({ kind: "drag", phase: "begin" });
    const results = await Promise.all(hits);
    expect(results.every((h) => (h as { mesh: unknown }).mesh === a)).toBe(true);
    d.end();
    await settle(scene);
    expect(scene.placement(a).translation).toEqual([1.9000000000000001, 0, 0]);
  });

  it("fallback_flush_without_frames", async () => {
    const { scene, w } = await start();
    const s = sphere([0, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    await settle(scene);
    const sent = w.sent.length;
    // Frames stopped: the query is sent by the fallback timer.
    const t0 = performance.now();
    const hit = scene.raycast([-5, 0, 0], [1, 0, 0]);
    await new Promise((r) => setTimeout(r, FALLBACK_FLUSH_MS / 4));
    expect(w.sent.length).toBe(sent);
    expect(((await hit) as { mesh: unknown }).mesh).toBe(a);
    expect(performance.now() - t0).toBeGreaterThanOrEqual(FALLBACK_FLUSH_MS - 5);
    expect(w.sent.length).toBe(sent + 1);
    // A frame clears the timer: the query goes with the frame, once.
    const hit2 = scene.raycast([-5, 0, 0], [1, 0, 0]);
    scene.commitFrame();
    await hit2;
    await new Promise((r) => setTimeout(r, FALLBACK_FLUSH_MS * 1.5));
    expect(w.sent.length).toBe(sent + 2);
  });

  it("no_frame_loop_sends_on_microtask", async () => {
    const { scene, w } = await start();
    const s = sphere([0, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    const hit = scene.raycast([-5, 0, 0], [1, 0, 0]);
    expect(w.sent.length).toBe(0);
    await Promise.resolve();
    // The add and the query, in one batch, without any commitFrame.
    expect(w.sent.length).toBe(1);
    expect(w.last().map((r) => r.kind)).toEqual(["addMesh", "query"]);
    expect(((await hit) as { mesh: unknown }).mesh).toBe(a);
  });

  it("review_fixes_minor", async () => {
    const { scene } = await start();
    const s = sphere([0, 0, 0]);
    const t = sphere([3, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    const b = scene.addMesh(t.positions, t.indices);
    await settle(scene);

    // A drag started while a setPlacement is queued starts from it.
    scene.setPlacement(a, { rotation: IDENTITY, translation: [2, 0, 0] });
    const d = scene.drag(a);
    expect(scene.placement(a).translation).toEqual([2, 0, 0]);
    d.move({ rotation: IDENTITY, translation: [3, 0, 0] });
    d.cancel();
    expect(scene.placement(a).translation).toEqual([2, 0, 0]);
    await settle(scene);
    scene.commitFrame();
    expect(scene.placement(a).translation).toEqual([2, 0, 0]);
    await expectMirrored(scene, a);

    // Invalid planes throw here, as Scene does; the mirror keeps a valid one.
    const section = scene.addSection({ plane: { origin: [0, 0, 0], normal: [0, 0, 1] }, meshes: [a] });
    const code = (f: () => unknown) => {
      try {
        f();
      } catch (e) {
        expect(e).toBeInstanceOf(FastMeshError);
        return (e as FastMeshError).code;
      }
      return undefined;
    };
    expect(code(() => scene.setSectionPlane(section, { origin: [0, 0, 0], normal: [0, 0, 0] }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(code(() => scene.setSectionPlane(section, { origin: [0, NaN, 0], normal: [0, 0, 1] }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(
      code(() => scene.addSection({ plane: { origin: [0, 0, 0], normal: [0, 0, Infinity] }, meshes: [a] })),
    ).toBe("INVALID_ARGUMENT");
    expect(scene.sectionPlane(section)).toEqual({ origin: [0, 0, 0], normal: [0, 0, 1] });

    // Garbage input to addMesh is a FastMeshError, not a TypeError.
    expect(code(() => scene.addMesh(null as never, [0, 1, 2]))).toBe("INVALID_ARGUMENT");
    expect(code(() => scene.addMesh(42 as never, [0, 1, 2]))).toBe("INVALID_ARGUMENT");
    expect(code(() => scene.addMesh([0, 0, 0, 1, 0, 0, 0, 1, 0], { length: 3 } as never))).toBeUndefined();
    expect(code(() => scene.addMesh([0, 0, 0], Symbol() as never))).toBe("INVALID_ARGUMENT");

    // A removed mesh's state is pruned to a tombstone once delivered.
    scene.removeMesh(b);
    await settle(scene);
    scene.commitFrame();
    expect(code(() => scene.views(b))).toBe("UNKNOWN_MESH");
    expect(code(() => scene.idOf(b))).toBe("UNKNOWN_MESH");
    expect(code(() => scene.placement(b))).toBe("UNKNOWN_MESH");
    expect(code(() => scene.removeMesh(b))).toBe("UNKNOWN_MESH");

    // A throwing status listener does not stop the others; its error is reported.
    const reported: unknown[] = [];
    vi.stubGlobal("reportError", (e: unknown) => reported.push(e));
    let after = 0;
    const off1 = scene.onStatus(() => {
      throw new Error("listener boom");
    });
    const off2 = scene.onStatus(() => after++);
    scene.setPlacement(a, { rotation: IDENTITY, translation: [0, 0, 0] });
    await Promise.resolve();
    expect(after).toBe(1);
    expect(reported).toHaveLength(1);
    expect((reported[0] as Error).message).toBe("listener boom");
    off1();
    off2();
    vi.unstubAllGlobals();
  });

  it("drag_start_prefers_newest_pose", async () => {
    const { scene } = await start();
    const s = sphere([0, 0, 0]);
    const a = scene.addMesh(s.positions, s.indices);
    const b = scene.addMesh(s.positions, s.indices);
    await settle(scene);
    const at = (x: number) => ({ rotation: IDENTITY, translation: [x, 0, 0] as Vec3 });

    // A drag's held pose, then a newer setPlacement: the next drag starts from it.
    const d1 = scene.drag(a);
    d1.move(at(1));
    d1.end();
    scene.setPlacement(a, at(7));
    const d2 = scene.drag(a);
    expect(scene.placement(a).translation).toEqual([7, 0, 0]);
    d2.move(at(9));
    d2.cancel();
    expect(scene.placement(a).translation).toEqual([7, 0, 0]);
    await settle(scene);
    scene.commitFrame();
    expect(scene.placement(a).translation).toEqual([7, 0, 0]);
    await expectMirrored(scene, a);

    // An undo queued after a setPlacement makes it stale: the drag starts
    // from the engine's pose, which the undo leaves where it was.
    scene.setPlacement(b, at(3));
    await settle(scene);
    scene.setPlacement(b, at(5));
    scene.undo();
    const d3 = scene.drag(b);
    expect(scene.placement(b).translation).toEqual([3, 0, 0]);
    d3.move(at(4));
    d3.cancel();
    expect(scene.placement(b).translation).toEqual([3, 0, 0]);
    await settle(scene);
    scene.commitFrame();
    expect(scene.placement(b).translation).toEqual([3, 0, 0]);
    await expectMirrored(scene, b);

    // A huge finite normal still normalises.
    const section = scene.addSection({ plane: { origin: [0, 0, 0], normal: [0, 0, 1e308] }, meshes: [a] });
    scene.setSectionPlane(section, { origin: [0, 0, 0], normal: [1e308, 1e308, 0] });
    const n = scene.sectionPlane(section).normal;
    expect(n[0]).toBeCloseTo(Math.SQRT1_2, 12);
    expect(n[1]).toBeCloseTo(Math.SQRT1_2, 12);
    expect(n[2]).toBe(0);

    // terminate() clears the fallback timer: no flush after it.
    const pending = scene.raycast([0, 0, 0], [1, 0, 0]);
    scene.terminate();
    await expect(pending).rejects.toMatchObject({ code: "INVALID_ARGUMENT" });
  });

  it("package_worker_export_resolves", () => {
    const dist = new URL("../dist/", import.meta.url);
    expect(existsSync(new URL("worker.js", dist))).toBe(true);
    expect(existsSync(new URL("worker-entry.js", dist))).toBe(true);
    expect(existsSync(new URL("worker.d.ts", dist))).toBe(true);
    // The test-only dump stays out of the public types.
    expect(readFileSync(new URL("worker.d.ts", dist), "utf8")).not.toMatch(/debugDump/);
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
    expect(pkg.exports["./worker"]).toEqual({ types: "./dist/worker.d.ts", default: "./dist/worker.js" });

    /** Every dist file a file imports, transitively. */
    const graph = (file: string, seen = new Set<string>()): Set<string> => {
      if (seen.has(file)) {
        return seen;
      }
      seen.add(file);
      const src = readFileSync(new URL(file, dist), "utf8");
      for (const m of src.matchAll(/(?:import|export)\s[^"']*?from\s*["']([^"']+)["']|import\s*["']([^"']+)["']/g)) {
        const spec = m[1] ?? m[2];
        if (spec.startsWith("./")) {
          graph(new URL(spec, new URL(file, dist)).href.slice(dist.href.length), seen);
        } else {
          seen.add(spec);
        }
      }
      return seen;
    };
    for (const core of ["web.js", "node.js", "three.js"]) {
      const files = [...graph(core)];
      expect(files.filter((f) => f.startsWith("worker") || f.startsWith("protocol")), core).toEqual([]);
    }
    const entry = [...graph("worker-entry.js")];
    // The single-threaded glue (the scene imports it; the entry also loads it
    // by dynamic import, and the threaded one only that way).
    expect(entry).toContain("wasm/fast_mesh_wasm.js");
    expect(entry.filter((f) => /three|react/.test(f))).toEqual([]);
    expect([...graph("worker.js")].filter((f) => /three|react|worker-engine/.test(f))).toEqual([]);
    // In worker.js or a chunk it imports (the client is shared with
    // fast-mesh-wasm/store); chunks sit next to worker-entry.js in dist/, so
    // the relative URL resolves the same.
    const workerFiles = [...graph("worker.js")].filter((f) => f.endsWith(".js") && !f.includes("/"));
    expect(
      workerFiles.some((f) =>
        readFileSync(new URL(f, dist), "utf8").includes(
          'new Worker(new URL("./worker-entry.js", import.meta.url), { type: "module" })',
        ),
      ),
    ).toBe(true);
    // publint and attw.
    execSync("pnpm run check", { cwd: new URL("..", import.meta.url), stdio: "pipe" });
  }, 120_000);
});
