// Wraps of meshes on Scene and WorkerScene: the plan, the envelope, the
// placement rule, one undo step and the errors.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_WRAP_DETAIL,
  DEFAULT_WRAP_MAX_VOXELS,
  FastMeshError,
  type MeshHandle,
  Scene,
  type WrapOptions,
} from "../src/node.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

/** A closed axis-aligned box from `lo` to `hi`, without its `+z` face when `open`. */
function box(lo: [number, number, number], hi: [number, number, number], open = false) {
  const positions: number[] = [];
  for (let i = 0; i < 8; i++) {
    positions.push(i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]);
  }
  const indices = [
    0, 2, 1, 1, 2, 3, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5,
    ...(open ? [] : [4, 5, 6, 5, 7, 6]),
  ];
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

/** Live triangles in the mesh's render view (dead slots read 0, 0, 0). */
function liveTriangles(s: Scene | WorkerScene, mesh: MeshHandle): number {
  const idx = s.views(mesh).indices();
  let n = 0;
  for (let k = 0; k < idx.length; k += 3) {
    if (idx[k] !== 0 || idx[k + 1] !== 0 || idx[k + 2] !== 0) n++;
  }
  return n;
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

describe("wraps on Scene & WorkerScene", () => {
  const cleanups: (() => void)[] = [];

  afterEach(() => {
    for (const c of cleanups.splice(0)) c();
  });

  function openSync(): Scene {
    const scene = new Scene();
    cleanups.push(() => scene.free());
    return scene;
  }

  async function openWorker(): Promise<WorkerScene> {
    const worker = await createWorkerScene({ createWorker: () => new NodeWorker() });
    cleanups.push(() => worker.terminate());
    return worker;
  }

  /** Sends the frame and waits for every result so far. */
  async function settle(s: Scene | WorkerScene): Promise<void> {
    if (s instanceof Scene) {
      s.commitFrame();
      return;
    }
    const barrier = s.raycast([0, 0, 1e6], [0, 0, 1]);
    s.commitFrame();
    await barrier;
  }

  const rejects = (run: () => unknown) => expect(Promise.resolve().then(run)).rejects;

  async function runChecks(s: Scene | WorkerScene): Promise<void> {
    // A unit box, and one touching it whose far side is open by a 1 x 1 hole.
    const a = box([0, 0, 0], [1, 1, 1]);
    const b = box([0, 0, 1], [1, 1, 2], true);
    const ma = s.addMesh(a.positions, a.indices);
    const mb = s.addMesh(b.positions, b.indices);
    await settle(s);

    // The plan first: 2 / 32 voxels, half the gap is 1.2 / 2, plus three voxels.
    const options: WrapOptions = { gap: 1.2, detail: 32 };
    const plan = await s.wrapPlan([ma, mb], options);
    const side = Math.ceil((1 + 2 * (0.6 + 3 / 16)) * 16) + 1;
    expect(plan.dims).toEqual([side, side, side + 16]);
    expect(plan.voxelSize).toBe(0.0625);
    expect(plan.radius).toBeCloseTo(0.6, 6);
    expect(plan.voxels).toBe(side * side * (side + 16));
    expect(plan.bytes).toBe(plan.voxels * 17);
    expect([plan.maxVoxels, plan.fits]).toEqual([DEFAULT_WRAP_MAX_VOXELS, true]);
    expect((await s.wrapPlan(ma)).voxelSize).toBeCloseTo(1 / DEFAULT_WRAP_DETAIL, 9);

    // Together: one closed envelope over the hole, as a new mesh.
    const r = await s.wrap([ma, mb], options);
    await settle(s);
    expect(r.plan).toEqual(plan);
    expect(r.projected).toBe(0);
    expect(liveTriangles(s, r.mesh)).toBe(r.triangles);
    // Closed and manifold: V - E + F = 2 for one sphere-like shell.
    expect(r.vertices - r.triangles / 2).toBe(2);
    // In the world, the two boxes' box: matrix4 maps the local bounds.
    const bounds = s.bounds(r.mesh);
    const at = s.matrix4(r.mesh).slice(12, 15);
    for (let k = 0; k < 3; k++) {
      expect(Math.abs(bounds!.min[k] + at[k])).toBeLessThan(0.02);
      expect(Math.abs(bounds!.max[k] + at[k] - (k === 2 ? 2 : 1))).toBeLessThan(0.1);
    }
    // The patch over the hole sinks as far as a ball of the radius does: 0.27 in the middle.
    const measured = await s.measure(r.mesh);
    expect(measured.volume).toBeGreaterThan(1.8);
    expect(measured.volume).toBeLessThan(1.9);
    expect(await s.selfIntersectingTriangles(r.mesh)).toHaveLength(0);

    // One undo step empties it; redo fills it again. The sources are untouched.
    s.undo();
    await settle(s);
    expect(liveTriangles(s, r.mesh)).toBe(0);
    expect(liveTriangles(s, ma)).toBe(12);
    s.redo();
    await settle(s);
    expect(liveTriangles(s, r.mesh)).toBe(r.triangles);

    // Projection puts most vertices on the meshes.
    const p = await s.wrap([ma, mb], { ...options, project: true, normals: false });
    await settle(s);
    expect(p.projected).toBeGreaterThan(p.vertices / 2);
    expect(await s.selfIntersectingTriangles(p.mesh)).toHaveLength(0);

    // One placed mesh: its own placement; a scaled one is wrapped in the world.
    const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    s.setPlacement(ma, { rotation: identity, translation: [0, 0, 5] });
    const moved = await s.wrap(ma, { voxelSize: 0.1 });
    await settle(s);
    expect(s.placement(moved.mesh).translation).toEqual([0, 0, 5]);
    expect(moved.plan.voxelSize).toBeCloseTo(0.1, 6);
    s.setPlacement(ma, { rotation: identity, translation: [0, 0, 5], scale: [2, 2, 2] });
    const scaled = await s.wrap(ma, { voxelSize: 0.1 });
    await settle(s);
    expect(s.placement(scaled.mesh).translation).toEqual([0, 0, 0]);
    expect((await s.measure(scaled.mesh)).volume).toBeCloseTo(8, 0);

    // A grid over the limit: the plan says so, and the wrap is refused with the numbers.
    const big = { voxelSize: 0.001 };
    expect((await s.wrapPlan(mb, big)).fits).toBe(false);
    await rejects(() => s.wrap(mb, big)).toThrow(/\d+ x \d+ x \d+ \(\d+ voxels, about \d+ MB\) is over the limit/);
    await rejects(() => s.wrap(mb, { maxVoxels: 1000 })).toBeInstanceOf(FastMeshError);
    // A flat sheet encloses nothing.
    const sheet = s.addMesh([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], [0, 1, 2, 0, 2, 3]);
    await settle(s);
    await rejects(() => s.wrap(sheet, { gap: 0.2, detail: 32 })).toThrow(/no volume/);
    await rejects(() => s.wrap([])).toThrow(/at least one mesh/);
    for (const [bad, message] of [
      [{ gap: -1 }, /gap/],
      [{ gap: Number.NaN }, /gap/],
      [{ voxelSize: 0 }, /voxelSize/],
      [{ detail: 7 }, /detail/],
      [{ detail: 12.5 }, /detail/],
      [{ maxVoxels: 10 }, /maxVoxels/],
      [{ project: 1 }, /project/],
      [{ normals: "yes" }, /normals/],
      [null, /options/],
    ] as [WrapOptions, RegExp][]) {
      await rejects(() => s.wrap(mb, bad)).toThrow(message);
      await rejects(() => s.wrapPlan(mb, bad)).toThrow(message);
    }
    await settle(s);
    // A failed wrap adds nothing: the next one still works.
    const again = await s.wrap(ma, { detail: 16 });
    await settle(s);
    expect(liveTriangles(s, again.mesh)).toBe(again.triangles);
  }

  it("plans and wraps meshes on Scene", async () => {
    await runChecks(openSync());
  });

  it("plans and wraps meshes on WorkerScene", async () => {
    await runChecks(await openWorker());
  });

  it("gives the same wrap on Scene and WorkerScene", async () => {
    const a = box([0, 0, 0], [1, 1, 1]);
    const b = box([0.4, 0.5, 0.6], [1.5, 1.4, 1.3]);
    const wrapped = async (s: Scene | WorkerScene) => {
      const meshes = [s.addMesh(a.positions, a.indices), s.addMesh(b.positions, b.indices)];
      await settle(s);
      const r = await s.wrap(meshes, { gap: 0.1, detail: 24, project: true });
      await settle(s);
      const v = s.views(r.mesh);
      return { counts: [r.vertices, r.triangles, r.projected], p: [...v.positions()], i: [...v.indices()] };
    };
    expect(await wrapped(await openWorker())).toEqual(await wrapped(openSync()));
  });
});
