// Extruding faces and bases on Scene and WorkerScene: the results, world
// units, one undo step, the reported self-intersections and the errors.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import {
  type ExtrudeBaseOptions,
  type ExtrudeFacesOptions,
  FastMeshError,
  type MeshHandle,
  Scene,
} from "../src/node.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

type AnyScene = Scene | WorkerScene;

/** A closed box from 0 to `size`, outward wound; triangles 10 and 11 are its +z face. */
function box(size: [number, number, number]) {
  const positions: number[] = [];
  for (let i = 0; i < 8; i++) {
    positions.push(i & 1 ? size[0] : 0, i & 2 ? size[1] : 0, i & 4 ? size[2] : 0);
  }
  const indices = Uint32Array.from([
    0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0, 2, 3, 0, 3, 1, 4, 5, 7, 4,
    7, 6,
  ]);
  return { positions: Float64Array.from(positions), indices };
}

/** An open `n` by `n` sheet of unit cells in the plane y = 1, facing +y, without the cells in `missing`. */
function sheet(n: number, missing: readonly [number, number][] = []) {
  const positions: number[] = [];
  for (let z = 0; z <= n; z++) {
    for (let x = 0; x <= n; x++) positions.push(x, 1, z);
  }
  const indices: number[] = [];
  for (let z = 0; z < n; z++) {
    for (let x = 0; x < n; x++) {
      if (missing.some(([mx, mz]) => mx === x && mz === z)) continue;
      const a = z * (n + 1) + x;
      indices.push(a, a + n + 2, a + 1, a, a + n + 1, a + n + 2);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
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

describe("extrusions on Scene & WorkerScene", () => {
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
  async function settle(s: AnyScene): Promise<void> {
    if (s instanceof Scene) {
      s.commitFrame();
      return;
    }
    const barrier = s.raycast([0, 0, 1e6], [0, 0, 1]);
    s.commitFrame();
    await barrier;
  }

  async function volume(s: AnyScene, mesh: MeshHandle): Promise<number | null> {
    await settle(s);
    const m = await s.measure(mesh);
    return m.volume;
  }

  async function undo(s: AnyScene): Promise<void> {
    s.undo();
    await settle(s);
  }

  async function code(run: () => unknown): Promise<string> {
    try {
      await run();
    } catch (e) {
      expect(e).toBeInstanceOf(FastMeshError);
      return (e as FastMeshError).code;
    }
    return "none";
  }

  async function checkFaces(s: AnyScene): Promise<void> {
    const b = box([2, 3, 4]);
    const mesh = s.addMesh(b.positions, b.indices);
    await settle(s);
    const up = await s.extrudeFaces(mesh, [10, 11], { distance: 0.5 });
    expect({ ...up, wallTriangles: [...up.wallTriangles], selfIntersections: [...(up.selfIntersections ?? [1])] }).toEqual({
      regions: 1,
      capTriangles: 2,
      movedVertices: 0,
      addedVertices: 4,
      wallTriangles: [12, 13, 14, 15, 16, 17, 18, 19],
      selfIntersections: [],
    });
    expect(up.wallTriangles).toBeInstanceOf(Uint32Array);
    expect(await volume(s, mesh)).toBeCloseTo(27, 5);
    // The same selection again: the moved faces.
    await s.extrudeFaces(mesh, Uint32Array.from([11, 10, 10]), { distance: 1, direction: "vertex" });
    expect(await volume(s, mesh)).toBeCloseTo(33, 5);
    // A direction: only the rise counts.
    const slanted = await s.extrudeFaces(mesh, [10, 11], { distance: 5, direction: [3, 0, 4] });
    expect(slanted.regions).toBe(1);
    expect(await volume(s, mesh)).toBeCloseTo(57, 4);
    // One undo step each.
    await undo(s);
    expect(await volume(s, mesh)).toBeCloseTo(33, 5);
    await undo(s);
    await undo(s);
    expect(await volume(s, mesh)).toBeCloseTo(24, 5);
    s.redo();
    expect(await volume(s, mesh)).toBeCloseTo(27, 5);
    await undo(s);

    // World units under a scaled placement: 1 is half a local unit along z.
    s.setPlacement(mesh, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [9, 0, 0], scale: [1, 1, 2] });
    await settle(s);
    const before = s.bounds(mesh);
    await s.extrudeFaces(mesh, [10, 11], { distance: 1 });
    await settle(s);
    expect((s.bounds(mesh)?.max[2] ?? 0) - (before?.max[2] ?? 0)).toBeCloseTo(0.5, 5);
    await undo(s);

    // Crossings are reported, and the mesh is written all the same.
    const through = await s.extrudeFaces(mesh, [10, 11], { distance: -20 });
    expect(through.selfIntersections?.length).toBeGreaterThan(0);
    await undo(s);
    const unchecked = await s.extrudeFaces(mesh, [10, 11], { distance: -20, checkSelfIntersections: false });
    expect(unchecked.selfIntersections).toBeNull();
    await undo(s);

    // Errors leave the mesh as it was.
    const options = (o: unknown) => o as ExtrudeFacesOptions;
    for (const bad of [
      undefined,
      { distance: 0 },
      { distance: Number.NaN },
      { distance: "1" },
      { distance: 1, direction: "up" },
      { distance: 1, direction: [0, 1] },
      { distance: 1, direction: [0, 0, 0] },
      { distance: 1, checkSelfIntersections: 1 },
    ]) {
      expect(await code(() => s.extrudeFaces(mesh, [10, 11], options(bad)))).toBe("INVALID_ARGUMENT");
    }
    expect(await code(() => s.extrudeFaces(mesh, [], { distance: 1 }))).toBe("INVALID_ARGUMENT");
    expect(await code(() => s.extrudeFaces(mesh, [12], { distance: 1 }))).toBe("INVALID_ARGUMENT");
    expect(await code(() => s.extrudeFaces(mesh, [-1], { distance: 1 }))).toBe("INVALID_ARGUMENT");
    // Every face of a closed box: its normals cancel out.
    const all = Uint32Array.from({ length: 12 }, (_, t) => t);
    await expect(Promise.resolve().then(() => s.extrudeFaces(mesh, all, { distance: 1 }))).rejects.toThrow(/cancel/);
    s.removeMesh(mesh);
    await settle(s);
    expect(await code(() => s.extrudeFaces(mesh, [10], { distance: 1 }))).toBe("UNKNOWN_MESH");
  }

  async function checkBase(s: AnyScene): Promise<void> {
    const open = sheet(4);
    const mesh = s.addMesh(open.positions, open.indices);
    expect(await volume(s, mesh)).toBeNull();
    const below = await s.extrudeBase(mesh, { offset: 2 });
    expect(below).toMatchObject({ height: -1, baseVertices: 16, wallTriangles: 32, capped: false, flipped: false });
    expect(below.loops).toHaveLength(1);
    expect(below.capTriangles).toBe(14 + 2 * below.capVertices);
    expect(below.capVertices).toBeGreaterThan(0);
    expect(below.selfIntersections).toEqual(new Uint32Array(0));
    expect(await volume(s, mesh)).toBeCloseTo(32, 4);
    await undo(s);
    expect(await volume(s, mesh)).toBeNull();

    // To a height, with the rim's vertices only, along another direction.
    const flat = await s.extrudeBase(mesh, { height: 0.25, refine: false, checkSelfIntersections: false });
    expect(flat).toMatchObject({ height: 0.25, capVertices: 0, capTriangles: 14, selfIntersections: null });
    expect(await volume(s, mesh)).toBeCloseTo(12, 4);
    await undo(s);
    // Upward it faces the base, so it is turned around; height is then -y.
    const up = await s.extrudeBase(mesh, { direction: [0, 7, 0], height: -3 });
    expect(up.flipped).toBe(true);
    expect(await volume(s, mesh)).toBeCloseTo(32, 4);
    await undo(s);

    // A hole: left open, made a hole through the solid, or extruded by id.
    const holed = sheet(6, [
      [2, 2],
      [3, 2],
      [2, 3],
      [3, 3],
    ]);
    const ring = s.addMesh(holed.positions, holed.indices);
    await settle(s);
    const outer = await s.extrudeBase(ring, { offset: 1 });
    expect(outer.baseVertices).toBe(24);
    expect(await volume(s, ring)).toBeNull();
    await undo(s);
    const both = await s.extrudeBase(ring, { offset: 1, loops: "all" });
    expect(both.loops).toHaveLength(2);
    expect(await volume(s, ring)).toBeCloseTo(32, 4);
    await undo(s);
    const loops = (await s.analyze(ring)).loops;
    const inner = loops.find((l) => l.id.len === 8);
    const one = await s.extrudeBase(ring, { offset: 1, loops: inner ? [inner.id] : [] });
    expect(one.loops).toEqual([inner?.id]);
    await settle(s);
    const left = (await s.analyze(ring)).loops;
    expect(left.map((l) => l.id.len)).toEqual([24]);
    await undo(s);

    // Errors.
    const options = (o: unknown) => o as ExtrudeBaseOptions;
    for (const bad of [
      undefined,
      {},
      { height: 0, offset: 1 },
      { offset: Number.POSITIVE_INFINITY },
      { offset: 0 },
      { height: 5 },
      { offset: 1, direction: [0, 0, 0] },
      { offset: 1, direction: [0, -1] },
      { offset: 1, loops: "some" },
      { offset: 1, loops: 3 },
      { offset: 1, loops: [] },
      { offset: 1, loops: [{ minVertex: 0, len: 4, hash: 5 }] },
      { offset: 1, refine: "yes" },
    ]) {
      expect(await code(() => s.extrudeBase(mesh, options(bad)))).toBe("INVALID_ARGUMENT");
    }
    const stale = { minVertex: 0, len: 16, hash: 1n };
    expect(await code(() => s.extrudeBase(mesh, { offset: 1, loops: [stale] }))).toBe("STALE_LOOPS");
    const b = box([1, 1, 1]);
    const closed = s.addMesh(b.positions, b.indices);
    await settle(s);
    await expect(Promise.resolve().then(() => s.extrudeBase(closed, { offset: 1 }))).rejects.toThrow(
      /no boundary loop/,
    );
    s.removeMesh(closed);
    await settle(s);
    expect(await code(() => s.extrudeBase(closed, { offset: 1 }))).toBe("UNKNOWN_MESH");
  }

  it("extrudes faces on Scene", async () => {
    await checkFaces(openSync());
  });

  it("extrudes faces on WorkerScene", async () => {
    await checkFaces(await openWorker());
  });

  it("extrudes bases on Scene", async () => {
    await checkBase(openSync());
  });

  it("extrudes bases on WorkerScene", async () => {
    await checkBase(await openWorker());
  });
});
