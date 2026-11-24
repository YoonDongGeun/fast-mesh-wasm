// Lattice infill on Scene and WorkerScene: the plan, the preview, open and
// shelled fills, cavities, one undo step and the errors.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_LATTICE_MAX_VOXELS,
  FastMeshError,
  type LatticeOptions,
  type MeshHandle,
  Scene,
} from "../src/node.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

/** A closed axis-aligned box from `lo` to `hi`, wound inward when `inward`. */
function box(lo: number, hi: number, inward = false) {
  const positions: number[] = [];
  for (let i = 0; i < 8; i++) {
    positions.push(i & 1 ? hi : lo, i & 2 ? hi : lo, i & 4 ? hi : lo);
  }
  const indices = [
    0, 2, 1, 1, 2, 3, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5, 4, 5, 6, 5, 7, 6,
  ];
  if (inward) {
    for (let k = 0; k < indices.length; k += 3) {
      [indices[k + 1], indices[k + 2]] = [indices[k + 2], indices[k + 1]];
    }
  }
  return { positions, indices };
}

/** A box with a concentric cavity. */
function hollowBox() {
  const outer = box(0, 2);
  const inner = box(0.25, 1.75, true);
  return {
    positions: [...outer.positions, ...inner.positions],
    indices: [...outer.indices, ...inner.indices.map((v) => v + 8)],
  };
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

describe("lattice infill on Scene & WorkerScene", () => {
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

  /** Closed, manifold and free of self-intersections. */
  async function checkClean(s: Scene | WorkerScene, mesh: MeshHandle): Promise<void> {
    const [a, crossing] = await Promise.all([s.analyze(mesh), s.selfIntersectingTriangles(mesh)]);
    expect(a.loops).toHaveLength(0);
    expect(a.nonManifoldEdges).toBe(0);
    expect(a.pinchVertices).toHaveLength(0);
    expect(a.orientationConflicts).toBe(0);
    expect(crossing).toHaveLength(0);
  }

  async function runFills(s: Scene | WorkerScene): Promise<void> {
    const cube = box(0, 2);
    const id = s.addMesh(cube.positions, cube.indices);
    await settle(s);

    // The plan first: 2 / 0.1 voxels, plus two on each side.
    const auto = await s.latticePlan(id, { cellSize: 1, wallThickness: 0.2 });
    expect(auto.voxelSize).toBeCloseTo(0.0625, 6);
    const options: LatticeOptions = { cellSize: 1, wallThickness: 0.2, voxelSize: 0.1 };
    const plan = await s.latticePlan(id, options);
    expect(plan.dims).toEqual([25, 25, 25]);
    expect(plan.voxels).toBe(25 ** 3);
    expect(plan.bytes).toBe(5 * plan.voxels + 64 * plan.triangles);
    expect([plan.region, plan.fits, plan.cropped]).toEqual(["inside", true, false]);
    expect(plan.maxVoxels).toBe(DEFAULT_LATTICE_MAX_VOXELS);
    expect(plan.regionVolume).toBeCloseTo(8, 6);
    // 3.0915 x - 4.1888 x^3 at x = 0.2.
    expect(plan.solidFraction).toBeCloseTo(0.5848, 3);

    // The preview changes nothing; here it is the lattice itself.
    const preview = await s.latticePreview(id, options);
    expect(preview.plan.dims).toEqual(plan.dims);
    expect(preview.indices.length).toBeGreaterThan(9000);
    expect(preview.positions.every((c) => c >= -1.001 && c <= 1.001)).toBe(true);
    expect(preview.indices.reduce((m, v) => Math.max(m, v), 0)).toBe(preview.positions.length / 3 - 1);
    const coarse = await s.latticePreview(id, { cellSize: 1, wallThickness: 0.1, maxVoxels: 20_000 });
    expect(coarse.plan.voxels).toBeLessThanOrEqual(20_000);
    expect(coarse.plan.wallThickness).toBeGreaterThan(0.1);
    await settle(s);
    expect(liveTriangles(s, id)).toBe(12);

    // An open lattice, as one undo step.
    for (const kind of ["gyroid", "schwarzP", "diamond"] as const) {
      const planned = await s.latticePlan(id, { ...options, kind });
      const r = await s.latticeFill(id, { ...options, kind });
      await settle(s);
      expect(r.plan).toEqual(planned);
      expect(liveTriangles(s, id)).toBe(r.triangles);
      expect(r.solidVolume / 8).toBeGreaterThan(0.9 * planned.solidFraction);
      expect(r.solidVolume / 8).toBeLessThan(1.1 * planned.solidFraction);
      expect((await s.measure(id)).volume).toBeCloseTo(r.solidVolume, 3);
      expect(r.uvsDropped).toBe(false);
      await checkClean(s, id);
      if (kind === "gyroid") {
        expect([r.vertices, r.triangles]).toEqual([preview.positions.length / 3, preview.indices.length / 3]);
      }
      s.undo();
      await settle(s);
      expect(liveTriangles(s, id)).toBe(12);
    }

    // In a shell: the surface stays and the rest is carved out.
    const shelled = { ...options, shellThickness: 0.2 };
    const r = await s.latticeFill(id, shelled);
    await settle(s);
    expect(liveTriangles(s, id)).toBe(12 + r.triangles);
    expect((await s.measure(id)).volume).toBeCloseTo(r.solidVolume, 3);
    expect(r.solidVolume).toBeGreaterThan(8 - 1.6 ** 3);
    await checkClean(s, id);
    s.undo();
    await settle(s);

    // A cavity, found on its own or asked for.
    const part = hollowBox();
    const hollow = s.addMesh(part.positions, part.indices);
    await settle(s);
    const filled = await s.latticeFill(hollow, { cellSize: 0.75, wallThickness: 0.15, voxelSize: 0.075 });
    await settle(s);
    expect(filled.plan.region).toBe("cavity");
    expect(filled.plan.regionVolume).toBeCloseTo(1.5 ** 3, 5);
    expect(liveTriangles(s, hollow)).toBe(12 + filled.triangles);
    expect((await s.measure(hollow)).volume).toBeCloseTo(8 - 1.5 ** 3 + filled.solidVolume, 2);
    await checkClean(s, hollow);

    // World lengths: the same lattice in the box at half the size, scaled back.
    const half = box(0, 1);
    const scaled = s.addMesh(half.positions, half.indices);
    s.setPlacement(scaled, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0], scale: [2, 2, 2] });
    await settle(s);
    expect((await s.latticePlan(scaled, options)).regionVolume).toBeCloseTo(8, 5);
    const world = await s.latticeFill(scaled, options);
    await settle(s);
    expect(world.triangles).toBe(preview.indices.length / 3);
  }

  async function runErrors(s: Scene | WorkerScene): Promise<void> {
    const cube = box(0, 2);
    const id = s.addMesh(cube.positions, cube.indices);
    const options: LatticeOptions = { cellSize: 1, wallThickness: 0.2, voxelSize: 0.1 };
    await settle(s);
    // A grid over the limit: the plan says so, and the fill is refused with the numbers.
    const big = { ...options, voxelSize: 0.001 };
    expect((await s.latticePlan(id, big)).fits).toBe(false);
    await rejects(() => s.latticeFill(id, big)).toThrow(
      /\d+ x \d+ x \d+ \(\d+ voxels, about \d+ MB\) is over the limit/,
    );
    await rejects(() => s.latticeFill(id, { ...options, maxVoxels: 1000 })).toBeInstanceOf(FastMeshError);
    await rejects(() => s.latticeFill(id, { ...options, region: "cavity" })).toThrow(/no cavity/);
    await rejects(() => s.latticeFill(id, { ...options, voxelSize: 0.12 })).toThrow(/thinner than two voxels/);
    await rejects(() => s.latticeFill(id, { ...options, shellThickness: 1.5 })).toThrow(/no open space/);
    const sheet = s.addMesh([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], [0, 1, 2, 0, 2, 3]);
    await settle(s);
    await rejects(() => s.latticeFill(sheet, options)).toThrow(/closed/);
    for (const [bad, message] of [
      [{ ...options, kind: "honeycomb" }, /kind/],
      [{ ...options, region: "outside" }, /region/],
      [{ ...options, cellSize: 0 }, /cellSize/],
      [{ wallThickness: 0.2 }, /cellSize/],
      [{ ...options, wallThickness: Number.NaN }, /wallThickness/],
      [{ ...options, wallThickness: 0.5 }, /third of the cell/],
      [{ ...options, shellThickness: -1 }, /shellThickness/],
      [{ ...options, voxelSize: 0 }, /voxelSize/],
      [{ ...options, maxVoxels: 10 }, /maxVoxels/],
      [{ ...options, minFragment: -1 }, /minFragment/],
      [null, /options/],
    ] as [LatticeOptions, RegExp][]) {
      await rejects(() => s.latticeFill(id, bad)).toThrow(message);
      await rejects(() => s.latticePlan(id, bad)).toThrow(message);
      await rejects(() => s.latticePreview(id, bad)).toThrow(message);
    }
    await settle(s);
    expect(liveTriangles(s, id)).toBe(12);
  }

  // Each awaited call on a WorkerScene may wait for its fallback flush.
  const TIMEOUT = 30_000;

  it("plans, previews and fills on Scene", async () => {
    await runFills(openSync());
  });

  it("plans, previews and fills on WorkerScene", { timeout: TIMEOUT }, async () => {
    await runFills(await openWorker());
  });

  it("refuses bad input on Scene and WorkerScene", { timeout: TIMEOUT }, async () => {
    await runErrors(openSync());
    await runErrors(await openWorker());
  });

  it("gives the same lattice on Scene and WorkerScene", async () => {
    const cube = box(0, 1.2);
    const filled = async (s: Scene | WorkerScene) => {
      const id = s.addMesh(cube.positions, cube.indices);
      await settle(s);
      const options: LatticeOptions = {
        kind: "diamond",
        cellSize: 0.8,
        wallThickness: 0.16,
        shellThickness: 0.16,
        voxelSize: 0.08,
      };
      const p = await s.latticePreview(id, options);
      const r = await s.latticeFill(id, options);
      await settle(s);
      const v = s.views(id);
      return { r, p: [...v.positions()], i: [...v.indices()], preview: [...p.positions, ...p.indices] };
    };
    expect(await filled(await openWorker())).toEqual(await filled(openSync()));
  });
});
