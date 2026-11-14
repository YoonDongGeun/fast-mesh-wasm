// Isotropic/adaptive surface remeshing (#208) and extended QEM decimation (#211) tests
// across synchronous Scene and async WorkerScene.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, expectTypeOf, it } from "vitest";
import {
  type AdaptiveRemeshOptions,
  FastMeshError,
  type IsotropicRemeshOptions,
  Scene,
  type SimplifyOptions,
  type SimplifyReport,
  type SurfaceRemeshOptions,
  type SurfaceRemeshReport,
  type Vec3,
} from "../src/node.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

function boxMesh(cx = 0, cy = 0, cz = 0, half = 1) {
  const x0 = cx - half;
  const x1 = cx + half;
  const y0 = cy - half;
  const y1 = cy + half;
  const z0 = cz - half;
  const z1 = cz + half;
  const positions = Float64Array.from([
    x0, y0, z0,
    x1, y0, z0,
    x1, y1, z0,
    x0, y1, z0,
    x0, y0, z1,
    x1, y0, z1,
    x1, y1, z1,
    x0, y1, z1,
  ]);
  const indices = Uint32Array.from([
    0, 2, 1, 0, 3, 2,
    4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4,
    2, 3, 7, 2, 7, 6,
    1, 2, 6, 1, 6, 5,
    3, 0, 4, 3, 4, 7,
  ]);
  return { positions, indices };
}

function subdividedBoxMesh(half = 1, seg = 3) {
  const positions: number[] = [];
  const indices: number[] = [];
  const vertMap = new Map<string, number>();
  const addVert = (x: number, y: number, z: number): number => {
    const k = `${x.toFixed(5)},${y.toFixed(5)},${z.toFixed(5)}`;
    let id = vertMap.get(k);
    if (id === undefined) {
      id = positions.length / 3;
      vertMap.set(k, id);
      positions.push(x, y, z);
    }
    return id;
  };
  const addFace = (flip: boolean, corner: (u: number, v: number) => [number, number, number]) => {
    for (let j = 0; j < seg; j++) {
      const v0 = -half + (2 * half * j) / seg;
      const v1 = -half + (2 * half * (j + 1)) / seg;
      for (let i = 0; i < seg; i++) {
        const u0 = -half + (2 * half * i) / seg;
        const u1 = -half + (2 * half * (i + 1)) / seg;
        const p00 = addVert(...corner(u0, v0));
        const p10 = addVert(...corner(u1, v0));
        const p11 = addVert(...corner(u1, v1));
        const p01 = addVert(...corner(u0, v1));
        if (flip) {
          indices.push(p00, p11, p10, p00, p01, p11);
        } else {
          indices.push(p00, p10, p11, p00, p11, p01);
        }
      }
    }
  };
  addFace(false, (u, v) => [u, v, half]);
  addFace(true, (u, v) => [u, v, -half]);
  addFace(false, (u, v) => [half, u, v]);
  addFace(true, (u, v) => [-half, u, v]);
  addFace(false, (u, v) => [v, half, u]);
  addFace(true, (u, v) => [v, -half, u]);
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

function sphereMesh(center: Vec3 = [0, 0, 0], radius = 1, stacks = 8, slices = 16) {
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

function uvGridMesh(nx = 6, ny = 6) {
  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  for (let j = 0; j <= ny; j++) {
    const v = j / ny;
    for (let i = 0; i <= nx; i++) {
      const u = i / nx;
      positions.push(u * 2 - 1, v * 2 - 1, 0);
      uvs.push(u, v);
    }
  }
  const row = nx + 1;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const a = j * row + i;
      const b = a + 1;
      const c = a + row + 1;
      const d = a + row;
      indices.push(a, b, c, a, c, d);
    }
  }
  return {
    positions: Float64Array.from(positions),
    indices: Uint32Array.from(indices),
    uvs: Float32Array.from(uvs),
  };
}

class NodeWorker implements WorkerLike {
  readonly worker: Worker;
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  readonly #error: ((e: unknown) => void)[] = [];

  constructor() {
    this.worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), {
      workerData: { serve: true },
    });
    this.worker.on("message", (data: unknown) => {
      for (const cb of this.#message) {
        cb({ data });
      }
    });
    this.worker.on("error", (e) => {
      for (const cb of this.#error) {
        cb(e);
      }
    });
  }

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", cb: (e: any) => void): void {
    (type === "message" ? this.#message : this.#error).push(cb);
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

const syncScenes: Scene[] = [];
const workerScenes: WorkerScene[] = [];

function createSyncScene(): Scene {
  const s = new Scene();
  syncScenes.push(s);
  return s;
}

async function createTestWorkerScene(): Promise<WorkerScene> {
  const s = await createWorkerScene({ createWorker: () => new NodeWorker() });
  workerScenes.push(s);
  return s;
}

afterEach(() => {
  for (const s of syncScenes.splice(0)) {
    s.free();
  }
  for (const s of workerScenes.splice(0)) {
    s.terminate();
  }
});

describe("surface remesh and extended simplify (#208 & #211)", () => {
  it("exports typed option and report interfaces on public entrypoints", () => {
    expectTypeOf<SurfaceRemeshOptions>().toExtend<{ targetEdgeLength: number }>();
    expectTypeOf<IsotropicRemeshOptions>().not.toHaveProperty("mode");
    expectTypeOf<AdaptiveRemeshOptions>().not.toHaveProperty("mode");
    expectTypeOf<SurfaceRemeshReport>().toHaveProperty("uvsDropped");
    expectTypeOf<SimplifyOptions>().toHaveProperty("targetRatio");
    expectTypeOf<SimplifyReport>().toHaveProperty("maxDeviationAchieved");
  });

  it("isotropic and adaptive surface remesh keep manifoldness and drop UVs undoably", () => {
    const s = createSyncScene();
    const sphere = sphereMesh([0, 0, 0], 1, 8, 12);
    const mesh = s.addMesh(sphere.positions, sphere.indices);

    const iso = s.remesh.isotropic(mesh, {
      targetEdgeLength: 0.45,
      iterations: 3,
      smoothStrength: 0.5,
      projectToSurface: true,
    });
    expect(iso.initialTriangles).toBe(sphere.indices.length / 3);
    expect(iso.finalTriangles).toBeGreaterThan(0);
    expect(iso.iterationsRun).toBe(3);
    expect(iso.uvsDropped).toBe(false);

    const analysis = s.inspect.analyze(mesh);
    expect(analysis.nonManifoldEdges).toBe(0);
    expect(analysis.orientationConflicts).toBe(0);

    const adapt = s.remesh.adaptive(mesh, {
      targetEdgeLength: 0.4,
      minEdgeLength: 0.18,
      maxEdgeLength: 0.75,
      iterations: 2,
    });
    expect(adapt.iterationsRun).toBe(2);
    expect(adapt.finalTriangles).toBeGreaterThan(0);

    // Box with UV channel: remeshSurface drops UVs and undo restores them.
    const box = boxMesh(0, 0, 0, 0.5);
    const boxUvs = Float32Array.from([
      0, 0, 1, 0, 1, 1, 0, 1,
      0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8,
    ]);
    const textured = s.addMesh(box.positions, box.indices, { uvs: boxUvs });
    s.takeChanges();

    const dropReport = s.repair(textured, {
      op: "remeshSurface",
      mode: "isotropic",
      targetEdgeLength: 0.4,
      iterations: 2,
      lockCreases: true,
      creaseAngleDeg: 30,
    });
    expect(dropReport.uvsDropped).toBe(true);
    expect(dropReport.finalTriangles).toBeGreaterThan(dropReport.initialTriangles);
    expect(s.views(textured).uvs()).toBeNull();
    const changes = s.takeChanges();
    expect(changes.find((c) => c.mesh === textured)?.uvs).toBe("all");

    expect(s.undo()).toBe(true);
    const restoredUvs = s.views(textured).uvs();
    expect(restoredUvs).not.toBeNull();
    expect(Array.from(restoredUvs!.subarray(0, boxUvs.length))).toEqual(Array.from(boxUvs));
  });

  it("extended simplify supports targetRatio, maxDeviation, preserveUvs, and lockCreases", () => {
    const s = createSyncScene();
    const sphere = sphereMesh([0, 0, 0], 1, 10, 16);
    const m1 = s.addMesh(sphere.positions, sphere.indices);
    const initialTris = sphere.indices.length / 3;

    const byRatio = s.remesh.simplify(m1, { targetRatio: 0.5 });
    expect(byRatio.collapsed).toBeGreaterThan(0);
    expect(byRatio.trianglesRemoved).toBeGreaterThan(0);
    expect(byRatio.maxDeviationAchieved).toBeGreaterThanOrEqual(0);
    expect(initialTris - byRatio.trianglesRemoved).toBeLessThanOrEqual(Math.round(initialTris * 0.5) + 4);

    // Tight error bound halts early before exceeding delta.
    const m2 = s.addMesh(sphere.positions, sphere.indices);
    const bounded = s.remesh.simplify(m2, {
      targetRatio: 0.1,
      maxDeviation: 1e-4,
    });
    expect(bounded.maxDeviationAchieved).toBeLessThanOrEqual(1e-4 + 1e-12);

    // UV-preserving decimation keeps the UV channel intact.
    const grid = uvGridMesh(6, 6);
    const mUv = s.addMesh(grid.positions, grid.indices, { uvs: grid.uvs });
    const uvReport = s.remesh.simplify(mUv, {
      targetRatio: 0.5,
      preserveUvs: true,
      preserveBoundary: true,
      preventNormalInversion: true,
    });
    expect(uvReport.collapsed).toBeGreaterThan(0);
    expect(s.views(mUv).uvs()).not.toBeNull();

    // Crease-locked decimation on sharp subdivided box preserves sharp corners.
    const sharpBox = subdividedBoxMesh(1, 3);
    const mBox = s.addMesh(sharpBox.positions, sharpBox.indices);
    const creaseReport = s.remesh.simplify(mBox, {
      targetTriangles: 24,
      lockCreases: true,
      creaseAngleDeg: 30,
    });
    expect(creaseReport.collapsed).toBeGreaterThan(0);
    expect(s.inspect.analyze(mBox).nonManifoldEdges).toBe(0);
    const bounds = s.inspect.bounds(mBox);
    expect(bounds.min).toEqual([-1, -1, -1]);
    expect(bounds.max).toEqual([1, 1, 1]);
  });

  it("rejects invalid surface remesh and simplify arguments cleanly", () => {
    const s = createSyncScene();
    const box = boxMesh();
    const m = s.addMesh(box.positions, box.indices);

    expect(() => s.remesh.isotropic(m, { targetEdgeLength: -0.2 })).toThrow(FastMeshError);
    expect(() =>
      s.remesh.adaptive(m, { targetEdgeLength: 0.3, minEdgeLength: 0.8, maxEdgeLength: 0.2 }),
    ).toThrow(FastMeshError);
    expect(() => s.remesh.simplify(m, {})).toThrow(FastMeshError);
    expect(() => s.remesh.simplify(m, { targetRatio: 1.5 })).toThrow(FastMeshError);
    expect(() => s.remesh.simplify(m, { targetRatio: 0.5, maxDeviation: -1 })).toThrow(FastMeshError);
  });

  it("mirrors surface remesh and extended simplify results across WorkerScene", async () => {
    const worker = await createTestWorkerScene();
    const sphere = sphereMesh([0, 0, 0], 1, 8, 12);
    const m = worker.addMesh(sphere.positions, sphere.indices);

    const rem = await worker.remesh.adaptive(m, {
      targetEdgeLength: 0.45,
      minEdgeLength: 0.2,
      maxEdgeLength: 0.8,
      iterations: 2,
    });
    expect(rem.iterationsRun).toBe(2);
    expect(rem.finalTriangles).toBeGreaterThan(0);

    const dec = await worker.remesh.simplify(m, {
      targetRatio: 0.6,
      maxDeviation: 0.25,
      lockCreases: true,
      creaseAngleDeg: 30,
    });
    expect(dec.collapsed).toBeGreaterThan(0);
    expect(dec.maxDeviationAchieved).toBeLessThanOrEqual(0.25 + 1e-12);
    expect(worker.views(m).positions().length).toBeGreaterThan(0);
  });
});
