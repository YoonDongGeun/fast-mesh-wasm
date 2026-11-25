// Denoising and Delaunay flips on Scene and WorkerScene: what they do to a
// noisy box and a badly split grid, regions, one undo step each, and errors.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import { FastMeshError, generatePrimitive, type MeshHandle, Scene } from "../src/node.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

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

/** A cube of side 2 with `n` cells a side, and its vertices moved by up to `noise` along each axis. */
function noisyCube(n: number, noise: number): { positions: Float32Array; indices: Uint32Array } {
  const { positions, indices } = generatePrimitive({ kind: "box", size: [2, 2, 2], segments: [n, n, n] });
  let state = 1;
  const next = () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return (2 * state) / 2147483648 - 1;
  };
  return { positions: Float32Array.from(positions, (p) => p + noise * next()), indices };
}

/** A flat grid of `n` x `n` sheared cells, each split along its long diagonal. */
function shearedGrid(n: number): { positions: Float32Array; indices: Uint32Array } {
  const positions: number[] = [];
  const indices: number[] = [];
  for (let y = 0; y <= n; y++) {
    for (let x = 0; x <= n; x++) positions.push(x + 0.8 * y, y, 0);
  }
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const a = y * (n + 1) + x;
      indices.push(a, a + 1, a + n + 2, a, a + n + 2, a + n + 1);
    }
  }
  return { positions: Float32Array.from(positions), indices: Uint32Array.from(indices) };
}

/** The mesh's local positions and indices, copied out of its views. */
function parts(s: Scene | WorkerScene, mesh: MeshHandle): { positions: number[]; indices: number[] } {
  const v = s.views(mesh);
  return { positions: Array.from(v.positions()), indices: Array.from(v.indices()) };
}

/** The mean angle between each face normal and the nearest axis, in degrees. */
function meanOffAxis({ positions, indices }: { positions: number[]; indices: number[] }): number {
  let sum = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const [a, b, c] = [0, 1, 2].map((k) => {
      const v = 3 * indices[t + k];
      return [positions[v], positions[v + 1], positions[v + 2]];
    });
    const u = [0, 1, 2].map((k) => b[k] - a[k]);
    const w = [0, 1, 2].map((k) => c[k] - a[k]);
    const n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    const len = Math.hypot(n[0], n[1], n[2]);
    const best = Math.max(...n.map((x) => Math.abs(x))) / len;
    sum += (Math.acos(Math.min(1, best)) * 180) / Math.PI;
  }
  return sum / (indices.length / 3);
}

describe("denoise and delaunayFlips on Scene & WorkerScene", () => {
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

  async function runChecks(s: Scene | WorkerScene): Promise<unknown> {
    const n = 8;
    const cube = noisyCube(n, 0.04);
    const mesh = s.addMesh(cube.positions, cube.indices);
    await settle(s);
    const start = parts(s, mesh);
    const off = meanOffAxis(start);
    expect(off).toBeGreaterThan(5);

    // The faces come out flat, and the whole thing is one undo step.
    const r = await s.denoise(mesh, { iterations: 10, strength: 1, featureAngle: 45 });
    await settle(s);
    expect(r.moved).toBeGreaterThan(300);
    expect(r.maxDisplacement).toBeGreaterThan(0);
    expect(r.maxDisplacement).toBeLessThan(0.15);
    expect([r.iterations, r.reverted]).toEqual([10, 0]);
    expect(r.before.triangles).toBe(12 * n * n);
    expect(r.after.meanNormalDeviation).toBeLessThan(r.before.meanNormalDeviation / 1.5);
    const smooth = parts(s, mesh);
    expect(meanOffAxis(smooth)).toBeLessThan(0.6);
    expect(smooth.indices).toEqual(start.indices);
    s.undo();
    await settle(s);
    expect(parts(s, mesh)).toEqual(start);
    s.redo();
    await settle(s);
    expect(parts(s, mesh)).toEqual(smooth);
    // A second run over a region starts from here; the rest stays as it is.
    const region = Array.from({ length: 2 * n * n }, (_, t) => t);
    const part = await s.denoise(mesh, { region });
    await settle(s);
    expect(part.before.triangles).toBe(region.length);
    const touched = parts(s, mesh);
    const inRegion = new Set(region.flatMap((t) => start.indices.slice(3 * t, 3 * t + 3)));
    const outside = new Set(start.indices.slice(6 * n * n));
    for (let v = 0; v < touched.positions.length / 3; v++) {
      if (outside.has(v) || !inRegion.has(v)) {
        expect(touched.positions.slice(3 * v, 3 * v + 3)).toEqual(smooth.positions.slice(3 * v, 3 * v + 3));
      }
    }

    // Flips: the smallest angle rises, no vertex moves.
    const grid = shearedGrid(6);
    const flat = s.addMesh(grid.positions, grid.indices);
    await settle(s);
    const before = parts(s, flat);
    const f = await s.delaunayFlips(flat);
    await settle(s);
    expect(f.flips).toBeGreaterThanOrEqual(36);
    expect(f.converged).toBe(true);
    expect(f.passes).toBeGreaterThan(0);
    expect(f.before.minAngle).toBeCloseTo(22.29, 1);
    expect(f.after.minAngle).toBeGreaterThan(27.3);
    expect(f.after.triangles).toBe(72);
    const after = parts(s, flat);
    expect(after.positions).toEqual(before.positions);
    expect(after.indices).not.toEqual(before.indices);
    s.undo();
    await settle(s);
    expect(parts(s, flat).indices).toEqual(before.indices);
    // Only inside the region: the last row of cells is left alone.
    const some = await s.delaunayFlips(flat, { dihedralLimit: 5, region: Uint32Array.from({ length: 60 }, (_, t) => t) });
    await settle(s);
    expect(some.before.triangles).toBe(60);
    expect(parts(s, flat).indices.slice(180)).toEqual(before.indices.slice(180));
    // An empty region changes nothing.
    const none = await s.delaunayFlips(flat, { region: [] });
    expect([none.flips, none.before.triangles, none.converged]).toEqual([0, 0, true]);

    // Errors.
    const rejects = (f: () => unknown, pattern: RegExp) =>
      expect(Promise.resolve().then(f)).rejects.toThrow(pattern);
    await rejects(() => s.denoise(mesh, { iterations: 101 }), /iterations/);
    await rejects(() => s.denoise(mesh, { iterations: 1.5 }), /iterations/);
    await rejects(() => s.denoise(mesh, { strength: 2 }), /strength/);
    await rejects(() => s.denoise(mesh, { strength: Number.NaN }), /options\.strength must be a finite number/);
    await rejects(() => s.denoise(mesh, { featureAngle: 0 }), /feature angle/);
    await rejects(() => s.denoise(mesh, { region: [1e6] }), /triangle id is dead or out of range/);
    await rejects(() => s.denoise(mesh, { region: [-1] }), /options\.region/);
    await rejects(() => s.denoise(mesh, 5 as never), /options must be an object/);
    await rejects(() => s.delaunayFlips(flat, { dihedralLimit: 91 }), /dihedral limit/);
    await rejects(() => s.delaunayFlips(flat, { dihedralLimit: "x" as never }), /dihedralLimit/);
    await rejects(() => s.delaunayFlips(flat, { region: [1e6] }), /triangle id is dead or out of range/);
    await expect(Promise.resolve().then(() => s.delaunayFlips(flat, { dihedralLimit: 91 }))).rejects.toBeInstanceOf(
      FastMeshError,
    );
    s.removeMesh(flat);
    await settle(s);
    await rejects(() => s.delaunayFlips(flat), /mesh/i);
    // Still usable, and still the same.
    expect(parts(s, mesh)).toEqual(touched);
    return { r, part, f, some, touched, after };
  }

  it("smooths and flips on Scene, and gives the same bits on WorkerScene", async () => {
    const sync = await runChecks(openSync());
    const worker = await runChecks(await openWorker());
    expect(worker).toEqual(sync);
  });
});
