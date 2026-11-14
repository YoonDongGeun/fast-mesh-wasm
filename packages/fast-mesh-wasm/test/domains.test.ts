// Domain-separated API tests across synchronous Scene and async WorkerScene.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, expectTypeOf, it, vi } from "vitest";
import {
  FastMeshError,
  type MeshHandle,
  Scene,
  type SceneDomains,
  type SceneLike,
  type SyncSceneDomains,
  type Vec3,
  type WorkerSceneDomains,
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
  vi.restoreAllMocks();
});

describe("domain architecture type parity", () => {
  it("Scene and WorkerScene satisfy domain and SceneLike contracts", () => {
    expectTypeOf<Scene>().toMatchTypeOf<SceneLike>();
    expectTypeOf<Scene>().toMatchTypeOf<SyncSceneDomains>();
    expectTypeOf<Scene>().toMatchTypeOf<SceneDomains<false>>();
    expectTypeOf<WorkerScene>().toMatchTypeOf<SceneLike>();
    expectTypeOf<WorkerScene>().toMatchTypeOf<WorkerSceneDomains>();
    expectTypeOf<WorkerScene>().toMatchTypeOf<SceneDomains<true>>();
  });
});

describe("Scene 7-domain namespaces", () => {
  it("repair callable signatures and namespace methods match legacy repair", () => {
    const warnSpy = vi.spyOn(console, "warn");
    const scene = createSyncScene();
    const { positions, indices } = boxMesh(0, 0, 0, 1);
    const mesh = scene.addMesh(positions, indices);

    // Callable auto request + single op + batched list remain valid
    const autoLegacy = scene.repair(mesh, { kind: "auto", preset: "general" });
    expect(autoLegacy.weld.mergedVertices).toBe(0);

    const singleOp = scene.repair(mesh, { op: "weld", tolerance: 1e-6 });
    expect(singleOp.mergedVertices).toBe(0);

    const [orientRep, sliverRep] = scene.repair(mesh, [
      { op: "orient" },
      { op: "cleanupSlivers" },
    ] as const);
    expect(orientRep.flippedTriangles).toBe(0);
    expect(sliverRep.collapsedNeedles).toBe(0);

    // Domain namespace methods
    const analysis = scene.repair.analyze(mesh);
    expect(analysis.loops.length).toBe(0);
    const autoDom = scene.repair.autoRepair(mesh, { preset: "general" });
    expect(autoDom.orient.nonOrientablePatches).toBe(0);
    const cleaned = scene.repair.clean(mesh);
    expect(cleaned.degenerate.removedTriangles).toBe(0);
    expect(scene.repair.weld(mesh, 1e-6).mergedVertices).toBe(0);
    expect(scene.repair.orient(mesh).flippedTriangles).toBe(0);
    expect(scene.repair.makeManifold(mesh).removedDuplicateTriangles).toBe(0);
    expect(scene.repair.resolveSelfIntersections(mesh).initialPairs).toBe(0);
    expect(scene.repair.cleanupSlivers(mesh).collapsedNeedles).toBe(0);

    // Deprecated legacy wrappers continue to work without runtime console.warn
    expect(scene.analyze(mesh).components.length).toBe(1);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it("sculpt callable signatures, stroke lifecycle, filter, and brush shortcuts work", () => {
    const scene = createSyncScene();
    const { positions, indices } = sphereMesh([0, 0, 0], 1, 10, 20);
    const mesh = scene.addMesh(positions, indices);

    // Stroke lifecycle collapses dabs into a single undoable step
    scene.sculpt.beginStroke(mesh, { captureRestSizes: true });
    const d1 = scene.sculpt.dab(mesh, {
      op: "inflate",
      center: [0, 0, 1],
      radius: 0.6,
      falloff: "smooth",
      strength: 0.1,
    });
    expect(d1.moved).toBeGreaterThan(0);
    const d2 = scene.sculpt.smooth(mesh, {
      center: [0, 0, 1],
      radius: 0.6,
      falloff: "smooth",
      strength: 0.2,
      cycles: 1,
    });
    expect(d2.moved).toBeGreaterThan(0);
    scene.sculpt.endStroke();

    expect(scene.canUndo()).toBe(true);
    scene.undo();
    expect(scene.canUndo()).toBe(false);

    // Cancel stroke rolls back completely
    scene.sculpt.beginStroke(mesh);
    scene.sculpt.clay(mesh, {
      center: [0, 0, 1],
      radius: 0.5,
      falloff: "smooth",
      strength: 0.15,
    });
    scene.sculpt.cancelStroke();
    expect(scene.canUndo()).toBe(false);

    // Filter builder + immediate filtered execution + callable legacy signature
    const filteredOp = scene.sculpt.filter(
      { op: "inflate", center: [0, 1, 0], radius: 0.6, falloff: "smooth", strength: 0.1 },
      { kind: "peaksOnly" },
    );
    expect(filteredOp.filter?.kind).toBe("peaksOnly");
    const callReport = scene.sculpt(mesh, filteredOp);
    expect(callReport.moved).toBeGreaterThan(0);
    const nearVerts = scene.sculpt.verticesInSphere(mesh, [0, 0, 1], 0.5);
    expect(nearVerts.length).toBeGreaterThan(0);
  });

  it("remesh domain exposes subdivide, simplify, region remesh, and voxel remesh", () => {
    const scene = createSyncScene();
    const { positions, indices } = sphereMesh([0, 0, 0], 1, 8, 12);
    const mesh = scene.addMesh(positions, indices);

    const sub = scene.remesh.subdivide(mesh, { levels: 1 });
    expect(sub.trianglesAdded).toBeGreaterThan(0);

    const reg = scene.remesh.region(mesh, { center: [0, 0, 1], radius: 0.7, maxRounds: 2 });
    expect(reg.rounds).toBeLessThanOrEqual(2);

    const simp = scene.remesh.simplify(mesh, { targetTriangles: 120, preserveBoundary: true });
    expect(simp.collapsed).toBeGreaterThan(0);

    const vox = scene.remesh.voxel(mesh, { algorithm: "fhEdt", voxelSize: 0.2 });
    expect(vox.trianglesAdded).toBeGreaterThan(0);
  });

  it("boolean callable signatures and union, difference, intersection, cut, offset, shell work", () => {
    const scene = createSyncScene();
    const boxA = boxMesh(0, 0, 0, 1);
    const boxB = boxMesh(0.8, 0, 0, 1);
    const a = scene.addMesh(boxA.positions, boxA.indices);
    const b = scene.addMesh(boxB.positions, boxB.indices);

    // Callable form
    scene.boolean(a, b, "union");
    const volAfterUnion = scene.inspect.measure(a).volume;
    expect(volAfterUnion).not.toBeNull();
    expect(volAfterUnion!).toBeGreaterThan(8);

    // Difference & intersection helpers
    const c = scene.addMesh(boxMesh(3, 0, 0, 1).positions, boxMesh(3, 0, 0, 1).indices);
    const d = scene.addMesh(boxMesh(3.5, 0, 0, 1).positions, boxMesh(3.5, 0, 0, 1).indices);
    scene.boolean.difference(c, d);
    expect(scene.inspect.measure(c).volume!).toBeLessThan(8);

    const cutReport = scene.boolean.cut(c, { origin: [2.25, 0, 0], normal: [1, 0, 0] });
    expect(cutReport.keptTriangles).toBeGreaterThan(0);
    expect(cutReport.removedTriangles).toBeGreaterThan(0);

    const e = scene.addMesh(boxMesh(0, 5, 0, 1).positions, boxMesh(0, 5, 0, 1).indices);
    const off = scene.boolean.offset(e, { distance: 0.1, algorithm: "normal" });
    expect(off.moved).toBeGreaterThan(0);

    // Open triangle sheet -> shell solidification
    const sheet = scene.addMesh(
      Float64Array.from([0, 0, 0, 2, 0, 0, 0, 2, 0]),
      Uint32Array.from([0, 1, 2]),
    );
    const sh = scene.boolean.shell(sheet, 0.25);
    expect(sh.loopsClosed).toBeGreaterThan(0);
  });

  it("deform domain exposes brush deformation and vertex protection", () => {
    const scene = createSyncScene();
    const { positions, indices } = sphereMesh([0, 0, 0], 1, 8, 16);
    const mesh = scene.addMesh(positions, indices);

    const protectedVerts = Uint32Array.from([0]);
    const op = scene.deform.protect(
      { op: "inflate", center: [0, 0, 1], radius: 0.8, falloff: "smooth", strength: 0.2 },
      { vertices: protectedVerts },
    );
    expect(op.protect?.vertices.length).toBe(1);

    const p0Before = scene.views(mesh).positions().slice(0, 3);
    const report = scene.deform.protect(
      mesh,
      { op: "grab", center: [0, 0, 1], radius: 0.8, falloff: "smooth", strength: 0.5, direction: [0, 0, 1] },
      { vertices: protectedVerts },
    );
    expect(report.moved).toBeGreaterThan(0);
    const p0After = scene.views(mesh).positions().slice(0, 3);
    expect(Array.from(p0After)).toEqual(Array.from(p0Before));

    expect(
      scene.deform.relaxed(mesh, { center: [0, 0, -1], radius: 0.8, falloff: "smooth", strength: 0.3, cycles: 1 })
        .moved,
    ).toBeGreaterThan(0);
  });

  it("inspect and align domains expose queries, fields, sections, measure, and ICP registration", () => {
    const scene = createSyncScene();
    const target = scene.addMesh(boxMesh(0, 0, 0, 1).positions, boxMesh(0, 0, 0, 1).indices);
    const probe = scene.addMesh(boxMesh(0, 0, 0, 0.5).positions, boxMesh(0, 0, 0, 0.5).indices);
    scene.setPlacement(probe, {
      rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      translation: [3, 0, 0],
    });

    const m = scene.inspect.measure(target);
    expect(m.closed).toBe(true);
    expect(m.volume).toBeCloseTo(8, 3);
    expect(scene.inspect.bounds(target)).not.toBeNull();

    const hit = scene.inspect.raycast([0, 0, 5], [0, 0, -1]);
    expect(hit?.mesh).toBe(target);
    expect(scene.inspect.intersects(target, probe)).toBe(false);

    const df = scene.inspect.addDistanceField({ name: "dist", on: target, to: probe, maxDistance: 10 });
    expect(scene.inspect.fieldValues(df).length).toBe(8);
    const tf = scene.inspect.addThicknessField({ name: "thick", on: target, maxDistance: 10 });
    expect(scene.inspect.fieldStatus(tf).current).toBe(true);
    const cf = scene.inspect.addCurvatureField({ name: "curv", on: target });
    expect(scene.inspect.fieldValues(cf).length).toBe(8);

    const sec = scene.inspect.addSection({
      plane: { origin: [0, 0, 0], normal: [0, 0, 1] },
      meshes: [target],
    });
    scene.inspect.updateSections();
    expect(scene.inspect.section(sec, target).contours.length).toBeGreaterThan(0);

    // Align two spheres slightly displaced
    const fixed = scene.addMesh(sphereMesh([10, 0, 0], 1, 12, 24).positions, sphereMesh([10, 0, 0], 1, 12, 24).indices);
    const moving = scene.addMesh(sphereMesh([10, 0, 0], 1, 12, 24).positions, sphereMesh([10, 0, 0], 1, 12, 24).indices);
    scene.setPlacement(moving, {
      rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      translation: [0.05, 0, 0],
    });
    const reg = scene.align.register(moving, fixed, { maxIterations: 20 });
    expect(reg.rmse).toBeLessThan(0.1);
  });
});

describe("WorkerScene 7-domain async namespaces", () => {
  it("delegates all 7 domains cleanly over the Worker client", async () => {
    const worker = await createTestWorkerScene();
    const { positions, indices } = sphereMesh([0, 0, 0], 1, 8, 16);
    const mesh: MeshHandle = worker.addMesh(positions, indices);
    const other: MeshHandle = worker.addMesh(positions, indices);
    worker.setPlacement(other, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0.05, 0, 0] });

    // Repair (callable + namespace)
    const legacyRep = await worker.repair(mesh, { op: "weld", tolerance: 1e-6 });
    expect(legacyRep.mergedVertices).toBe(0);
    const autoRep = await worker.repair.autoRepair(mesh, { preset: "general" });
    expect(autoRep.orient.nonOrientablePatches).toBe(0);

    // Inspect
    const measured = await worker.inspect.measure(mesh);
    expect(measured.closed).toBe(true);
    const analyzed = await worker.inspect.analyze(mesh);
    expect(analyzed.components.length).toBe(1);
    expect(worker.inspect.bounds(mesh)?.center).toEqual([0, 0, 0]);

    // Sculpt + Deform
    await worker.sculpt.beginStroke(mesh);
    const dab = await worker.sculpt.inflate(mesh, {
      center: [0, 0, 1],
      radius: 0.7,
      falloff: "smooth",
      strength: 0.05,
    });
    expect(dab.moved).toBeGreaterThan(0);
    worker.sculpt.endStroke();

    const def = await worker.deform.smooth(mesh, {
      center: [0, 0, 1],
      radius: 0.7,
      falloff: "smooth",
      strength: 0.2,
      cycles: 1,
    });
    expect(def.moved).toBeGreaterThan(0);

    // Remesh
    const sub = await worker.remesh.subdivide(mesh, { levels: 1 });
    expect(sub.trianglesAdded).toBeGreaterThan(0);
    const simp = await worker.remesh.simplify(mesh, { targetTriangles: 100 });
    expect(simp.collapsed).toBeGreaterThan(0);

    // Boolean (async offset works; CSG boolean cleanly rejects with FastMeshError)
    const off = await worker.boolean.offset(mesh, { distance: 0.05 });
    expect(off.moved).toBeGreaterThan(0);
    await expect(worker.boolean.union(mesh, other)).rejects.toBeInstanceOf(FastMeshError);

    // Align
    const aligned = await worker.align.register(other, mesh, { apply: false });
    expect(aligned.rmse).toBeLessThan(0.1);
  });
});
