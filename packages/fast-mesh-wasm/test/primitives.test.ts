// Primitive generators (#259): `generatePrimitive` on the main thread and
// `WorkerScene.generatePrimitive` in the scene's Worker, the shape encoding,
// and the routing through the active `WasmScene` class.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  FastMeshError,
  generatePrimitive,
  PRIMITIVE_SHAPE_KINDS,
  type PrimitiveMesh,
  type PrimitiveShape,
  Scene,
} from "../src/node.js";
import { activeWasmScene, useWasmScene } from "../src/scene.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

/** Every edge crossed once per direction (closed, consistently wound) or once in all (boundary). */
function topology(mesh: PrimitiveMesh): { boundary: number; euler: number } {
  const { positions, indices } = mesh;
  const v = positions.length / 3;
  const directed = new Map<number, number>();
  for (let t = 0; t < indices.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      const a = indices[t + k];
      const b = indices[t + ((k + 1) % 3)];
      const key = a * v + b;
      expect(directed.has(key)).toBe(false);
      directed.set(key, 1);
    }
  }
  let boundary = 0;
  let edges = 0;
  for (const key of directed.keys()) {
    const a = Math.floor(key / v);
    const b = key % v;
    if (!directed.has(b * v + a)) {
      boundary++;
      edges++;
    } else if (a < b) {
      edges++;
    }
  }
  return { boundary, euler: v - edges + indices.length / 3 };
}

/** The divergence-theorem volume: positive for outward winding. */
function signedVolume({ positions: p, indices: i }: PrimitiveMesh): number {
  let sum = 0;
  for (let t = 0; t < i.length; t += 3) {
    const [a, b, c] = [3 * i[t], 3 * i[t + 1], 3 * i[t + 2]];
    sum +=
      p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) -
      p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c]) +
      p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c]);
  }
  return sum / 6;
}

function extent(p: Float32Array, k: number): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = k; i < p.length; i += 3) {
    lo = Math.min(lo, p[i]);
    hi = Math.max(hi, p[i]);
  }
  return [lo, hi];
}

function invalid(fn: () => unknown): void {
  try {
    fn();
  } catch (e) {
    expect(e).toBeInstanceOf(FastMeshError);
    expect((e as FastMeshError).code).toBe("INVALID_ARGUMENT");
    return;
  }
  throw new Error("expected INVALID_ARGUMENT");
}

describe("generatePrimitive", () => {
  it("builds every kind closed, welded and outward with its defaults", () => {
    expect(PRIMITIVE_SHAPE_KINDS).toHaveLength(9);
    for (const kind of PRIMITIVE_SHAPE_KINDS) {
      const mesh = generatePrimitive({ kind } as PrimitiveShape);
      expect(mesh.positions).toBeInstanceOf(Float32Array);
      expect(mesh.indices).toBeInstanceOf(Uint32Array);
      expect(mesh.positions.length % 3).toBe(0);
      expect(mesh.indices.length % 3).toBe(0);
      const t = topology(mesh);
      if (kind === "plane") {
        expect(t.boundary).toBe(4);
        expect(t.euler).toBe(1);
      } else {
        expect(t.boundary, kind).toBe(0);
        expect(t.euler, kind).toBe(kind === "torus" ? 0 : 2);
        expect(signedVolume(mesh), kind).toBeGreaterThan(0);
      }
      // Centred along the default Y axis, and inside the unit cube.
      const [lo, hi] = extent(mesh.positions, 1);
      expect(Math.abs(lo + hi)).toBeLessThan(1e-6);
      for (let k = 0; k < 3; k++) {
        const [a, b] = extent(mesh.positions, k);
        expect(a).toBeGreaterThanOrEqual(-0.5 - 1e-6);
        expect(b).toBeLessThanOrEqual(0.5 + 1e-6);
      }
    }
  });

  it("takes each option", () => {
    const box = generatePrimitive({ kind: "box", size: [2, 4, 6], segments: [1, 2, 3] });
    expect(box.indices.length / 3).toBe(4 * (2 + 6 + 3));
    expect([extent(box.positions, 0), extent(box.positions, 1), extent(box.positions, 2)]).toEqual([
      [-1, 1],
      [-2, 2],
      [-3, 3],
    ]);
    expect(signedVolume(box)).toBeCloseTo(48, 4);
    const cube = generatePrimitive({ kind: "box", size: 2, segments: 4 });
    expect(cube.indices.length / 3).toBe(4 * 3 * 16);
    expect(signedVolume(cube)).toBeCloseTo(8, 4);

    const cyl = generatePrimitive({ kind: "cylinder", radius: 2, height: 3, segments: 256, axis: "z" });
    expect(extent(cyl.positions, 2)).toEqual([-1.5, 1.5]);
    expect(signedVolume(cyl)).toBeCloseTo(Math.PI * 12 * (Math.sin((2 * Math.PI) / 256) / ((2 * Math.PI) / 256)), 3);

    const sphere = generatePrimitive({ kind: "uvSphere", radius: 1, segments: 8, rings: 4 });
    expect(sphere.positions.length / 3).toBe(2 + 8 * 3);
    const ico = generatePrimitive({ kind: "icosphere", radius: 1, subdivisions: 2 });
    expect(ico.indices.length / 3).toBe(20 * 16);
    const torus = generatePrimitive({ kind: "torus", majorRadius: 2, minorRadius: 0.5, segments: 12, tubeSegments: 6 });
    expect(torus.positions.length / 3).toBe(72);
    const arrow = generatePrimitive({ kind: "arrow", length: 4, axis: "x" });
    expect(extent(arrow.positions, 0)).toEqual([-2, 2]);
    const prism = generatePrimitive({ kind: "prism", sides: 5, radius: 1, height: 2 });
    expect(prism.positions.length / 3).toBe(12);
    const cone = generatePrimitive({ kind: "cone", radius: 1, topRadius: 0.5, rings: 2, capRings: 2 });
    expect(topology(cone).boundary).toBe(0);
    const plane = generatePrimitive({ kind: "plane", size: [2, 4], segments: [2, 3], axis: "z" });
    expect(plane.indices.length / 3).toBe(12);
    expect([extent(plane.positions, 0), extent(plane.positions, 1), extent(plane.positions, 2)]).toEqual([
      [-1, 1],
      [-2, 2],
      [0, 0],
    ]);
  });

  it("refuses bad shapes with INVALID_ARGUMENT", () => {
    invalid(() => generatePrimitive({ kind: "pyramid" } as unknown as PrimitiveShape));
    invalid(() => generatePrimitive(null as unknown as PrimitiveShape));
    invalid(() => generatePrimitive({ kind: "box", size: "big" } as unknown as PrimitiveShape));
    invalid(() => generatePrimitive({ kind: "box", size: [1, 2] as unknown as [number, number, number] }));
    invalid(() => generatePrimitive({ kind: "cylinder", segments: 12.5 }));
    invalid(() => generatePrimitive({ kind: "cylinder", radius: Number.NaN }));
    invalid(() => generatePrimitive({ kind: "cylinder", axis: "w" as "x" }));
    // Out of range: checked by the generator.
    invalid(() => generatePrimitive({ kind: "cylinder", segments: 2 }));
    invalid(() => generatePrimitive({ kind: "uvSphere", radius: 0 }));
    invalid(() => generatePrimitive({ kind: "torus", majorRadius: 1, minorRadius: 1 }));
    invalid(() => generatePrimitive({ kind: "arrow", headRadius: 0.01 }));
    invalid(() => generatePrimitive({ kind: "icosphere", subdivisions: 17 }));
    invalid(() => generatePrimitive({ kind: "plane", segments: 0 }));
    invalid(() => generatePrimitive({ kind: "box", segments: [1 << 20, 1 << 20, 1] }));
  });

  it("adds to a scene as a closed mesh", () => {
    const scene = new Scene();
    try {
      const { positions, indices } = generatePrimitive({ kind: "icosphere", radius: 2, subdivisions: 4 });
      const mesh = scene.addMesh(positions, indices);
      const m = scene.inspect.measure(mesh);
      expect(m.closed).toBe(true);
      expect(m.volume).toBeGreaterThan(0.99 * (4 / 3) * Math.PI * 8);
      expect(m.volume).toBeLessThan((4 / 3) * Math.PI * 8);
    } finally {
      scene.free();
    }
  });

  it("goes through the WasmScene class scenes are made from", () => {
    // The Worker entry swaps in the threaded glue's class (`useWasmScene`):
    // the generator must follow it, not the single-threaded glue.
    const real = activeWasmScene();
    const spy = vi.fn((kind: number, params: Float64Array) => real.generatePrimitive(kind, params));
    useWasmScene(Object.assign(Object.create(real), { generatePrimitive: spy }) as typeof real);
    try {
      generatePrimitive({ kind: "prism", sides: 3 });
      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy.mock.calls[0][0]).toBe(7);
      expect(Array.from(spy.mock.calls[0][1])).toEqual([3, Number.NaN, Number.NaN, Number.NaN]);
    } finally {
      useWasmScene(real);
    }
  });
});

/** The engine in a `worker_threads` Worker, served as the browser Worker script is. */
class NodeWorker implements WorkerLike {
  readonly worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), {
    workerData: { serve: true },
  });
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  readonly #error: ((e: unknown) => void)[] = [];

  constructor() {
    this.worker.on("message", (data) => {
      for (const l of [...this.#message]) l({ data });
    });
    this.worker.on("error", (err) => {
      for (const l of [...this.#error]) l({ message: String(err) });
    });
  }

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error" | "messageerror", listener: (e: never) => void): void {
    (type === "message" ? this.#message : this.#error).push(listener as never);
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

describe("WorkerScene.generatePrimitive", () => {
  const cleanups: (() => void)[] = [];
  afterEach(() => {
    for (const c of cleanups.splice(0)) c();
  });

  async function open(): Promise<WorkerScene> {
    const scene = await createWorkerScene({ createWorker: () => new NodeWorker() });
    cleanups.push(() => scene.terminate());
    return scene;
  }

  it("matches the main thread bit for bit and leaves the scene alone", async () => {
    const scene = await open();
    for (const shape of [
      { kind: "box", size: [1, 2, 3], segments: [2, 2, 2] },
      { kind: "torus", segments: 64, tubeSegments: 32, axis: "z" },
      { kind: "icosphere", subdivisions: 5 },
      { kind: "plane", size: 3, segments: 7 },
    ] as PrimitiveShape[]) {
      const there = await scene.generatePrimitive(shape);
      const here = generatePrimitive(shape);
      expect(new Uint32Array(there.positions.buffer)).toEqual(new Uint32Array(here.positions.buffer));
      expect(there.indices).toEqual(here.indices);
    }
    expect(scene.canUndo).toBe(false);
    // Addable as is: the same arrays go straight back to the Worker.
    const { positions, indices } = await scene.generatePrimitive({ kind: "cylinder", segments: 48 });
    const mesh = scene.addMesh(positions, indices);
    const m = await scene.inspect.measure(mesh);
    expect(m.closed).toBe(true);
    expect(m.volume).toBeGreaterThan(0);
  });

  it("fills an empty mesh in one undoable edit, as an app adding a shape as one undo step does", async () => {
    // The engine's history records edits, not added meshes: an empty mesh,
    // then every vertex and triangle in one edit (a fresh mesh numbers its
    // vertices from 0, so the indices stand).
    const scene = await open();
    const { positions, indices } = await scene.generatePrimitive({ kind: "torus", segments: 24, tubeSegments: 12 });
    const mesh = scene.addMesh(new Float64Array(0), new Uint32Array(0), { origin: [5, 0, 0] });
    const world = Float64Array.from(positions, (v, i) => (i % 3 === 0 ? v + 5 : v));
    const r = await scene.edit(mesh, [
      { op: "addVertices", positions: world },
      { op: "addTriangles", triangles: indices },
    ]);
    expect(Array.from(r.added[0])).toEqual(Array.from({ length: positions.length / 3 }, (_, i) => i));
    expect(scene.canUndo).toBe(true);
    const m = await scene.inspect.measure(mesh);
    expect(m.closed).toBe(true);
    expect(m.volume).toBeCloseTo(signedVolume({ positions, indices }), 4);
    expect(scene.views(mesh).indices().length).toBe(indices.length);
    scene.undo();
    expect((await scene.inspect.measure(mesh)).area).toBe(0);
    expect(scene.canUndo).toBe(false);
    scene.redo();
    expect((await scene.inspect.measure(mesh)).closed).toBe(true);
  });

  it("rejects bad shapes with INVALID_ARGUMENT, checked here or in the Worker", async () => {
    const scene = await open();
    for (const shape of [
      { kind: "cube" },
      { kind: "cylinder", segments: 1.5 },
      { kind: "cylinder", segments: 2 },
      { kind: "torus", majorRadius: 1, minorRadius: 2 },
    ]) {
      await expect(scene.generatePrimitive(shape as PrimitiveShape)).rejects.toMatchObject({
        name: "FastMeshError",
        code: "INVALID_ARGUMENT",
      });
    }
    // The scene still works.
    const ok = await scene.generatePrimitive({ kind: "box" });
    expect(ok.indices.length).toBe(36);
  });
});
