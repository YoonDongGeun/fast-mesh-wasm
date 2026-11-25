// Convex hulls of meshes, several meshes and raw point clouds, on Scene and
// WorkerScene: the counts, the placement rule, one undo step and the errors.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import { FastMeshError, type MeshHandle, packPointCloud, Scene } from "../src/node.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

/** A closed axis-aligned box from `lo` to `hi`. */
function box(lo: [number, number, number], hi: [number, number, number]) {
  const positions: number[] = [];
  for (let i = 0; i < 8; i++) {
    positions.push(i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]);
  }
  const indices = Uint32Array.from([
    0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5,
    3, 7, 5,
  ]);
  return { positions: Float64Array.from(positions), indices };
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

describe("convex hulls on Scene & WorkerScene", () => {
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

  async function runChecks(s: Scene | WorkerScene): Promise<void> {
    const a = box([0, 0, 0], [1, 1, 1]);
    const ma = s.addMesh(a.positions, a.indices);
    const mb = s.addMesh(a.positions, a.indices);
    const placed = { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [3, 0, 0] as [number, number, number] };
    s.setPlacement(mb, placed);
    await settle(s);

    // One mesh: its own placement.
    s.setPlacement(ma, { rotation: placed.rotation, translation: [0, 0, 5] });
    const one = await s.convexHull(ma);
    await settle(s);
    expect([one.inputPoints, one.vertices, one.triangles]).toEqual([8, 8, 12]);
    expect(s.placement(one.mesh).translation).toEqual([0, 0, 5]);
    expect(liveTriangles(s, one.mesh)).toBe(12);
    s.setPlacement(ma, { rotation: placed.rotation, translation: [0, 0, 0] });

    // Several meshes in the world: a 4 x 1 x 1 box at identity.
    const both = await s.convexHull([ma, mb], { normals: false });
    await settle(s);
    expect([both.inputPoints, both.vertices, both.triangles]).toEqual([16, 8, 12]);
    expect(s.placement(both.mesh).translation).toEqual([0, 0, 0]);
    expect(s.bounds(both.mesh)).not.toBeNull();
    expect(liveTriangles(s, both.mesh)).toBe(12);

    // One undo step empties it; redo fills it again. The sources are untouched.
    s.undo();
    await settle(s);
    expect(liveTriangles(s, both.mesh)).toBe(0);
    expect(liveTriangles(s, ma)).toBe(12);
    s.redo();
    await settle(s);
    expect(liveTriangles(s, both.mesh)).toBe(12);

    // Raw points, flat or packed, with duplicates and inside points.
    const flat = Float32Array.from([...a.positions, ...a.positions, 0.5, 0.5, 0.5]);
    const pts = await s.convexHullOfPoints(flat);
    await settle(s);
    expect([pts.inputPoints, pts.vertices, pts.triangles]).toEqual([17, 8, 12]);
    const packed = await s.convexHullOfPoints(packPointCloud(flat));
    expect([packed.inputPoints, packed.vertices, packed.triangles]).toEqual([17, 8, 12]);
    const numbers = await s.convexHullOfPoints([0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2]);
    expect([numbers.vertices, numbers.triangles]).toEqual([4, 4]);
    await settle(s);

    // Errors: no volume, bad arguments.
    const plane = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0.5, 0.5, 0];
    await expect(Promise.resolve().then(() => s.convexHullOfPoints(plane))).rejects.toThrow(/coplanar/);
    await expect(Promise.resolve().then(() => s.convexHullOfPoints(plane))).rejects.toBeInstanceOf(
      FastMeshError,
    );
    await expect(Promise.resolve().then(() => s.convexHullOfPoints([0, 0, 0, 1]))).rejects.toThrow(
      /multiple of 3/,
    );
    await expect(Promise.resolve().then(() => s.convexHullOfPoints([0, 0, Number.NaN]))).rejects.toThrow(
      /finite/,
    );
    await expect(Promise.resolve().then(() => s.convexHull([]))).rejects.toThrow(/at least one mesh/);
    await expect(
      Promise.resolve().then(() => s.convexHull(ma, { normals: "yes" as unknown as boolean })),
    ).rejects.toThrow(/normals/);
    await settle(s);
    // A failed hull adds nothing: the next hull still works.
    const again = await s.convexHull(mb);
    await settle(s);
    expect(liveTriangles(s, again.mesh)).toBe(12);
  }

  it("builds hulls of a mesh, several meshes and raw points on Scene", async () => {
    await runChecks(openSync());
  });

  it("builds hulls of a mesh, several meshes and raw points on WorkerScene", async () => {
    await runChecks(await openWorker());
  });
});
