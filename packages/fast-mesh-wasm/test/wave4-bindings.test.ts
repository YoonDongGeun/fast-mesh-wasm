// Wave 4 SDK & Worker RPC tests for Issues #206, #209, #213, and #214.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import {
  type BooleanOptions,
  type MeshHandle,
  Scene,
  type Vec3,
} from "../src/node.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

function boxMesh(cx = 0, cy = 0, cz = 0, hx = 1, hy = 1, hz = 1) {
  const x0 = cx - hx;
  const x1 = cx + hx;
  const y0 = cy - hy;
  const y1 = cy + hy;
  const z0 = cz - hz;
  const z1 = cz + hz;
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

function sphereMesh(center: Vec3 = [0, 0, 0], radius = 1, stacks = 12, slices = 24) {
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

function gridMesh(nx = 6, ny = 6) {
  const positions: number[] = [];
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      positions.push(x / (nx - 1) - 0.5, y / (ny - 1) - 0.5, 0);
    }
  }
  const indices: number[] = [];
  for (let y = 0; y < ny - 1; y++) {
    for (let x = 0; x < nx - 1; x++) {
      const a = y * nx + x;
      const b = a + 1;
      const c = a + nx + 1;
      const d = a + nx;
      indices.push(a, b, c, a, c, d);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

function rotZ(theta: number): number[] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [c, -s, 0, s, c, 0, 0, 0, 1];
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

describe("Wave 4 bindings (#206, #209, #213, #214) on Scene & WorkerScene", () => {
  const cleanups: (() => void)[] = [];

  afterEach(() => {
    for (const c of cleanups.splice(0)) c();
  });

  function openSync() {
    const scene = new Scene();
    cleanups.push(() => scene.free());
    return scene;
  }

  async function openWorker(): Promise<WorkerScene> {
    const worker = await createWorkerScene({
      createWorker: () => new NodeWorker(),
    });
    cleanups.push(() => worker.terminate());
    return worker;
  }

  it("#206: boolean options (manifold, bvhSdf, fhEdt), booleanBatch, and cutByPlane work on Scene & WorkerScene", async () => {
    // Sync scene tests
    const s = openSync();
    const aData = boxMesh(0, 0, 0, 1, 1, 1);
    const bData = boxMesh(0.8, 0, 0, 1, 1, 1);
    const cData = boxMesh(0, 0, 0.8, 0.7, 0.7, 0.7);
    const a = s.addMesh(aData.positions, aData.indices);
    const b = s.addMesh(bData.positions, bData.indices);
    const c = s.addMesh(cData.positions, cData.indices);

    const opts: BooleanOptions = {
      method: "manifold",
      coplanarTolerance: 1e-5,
      snapSlivers: true,
    };
    const merged = s.boolean.batch(
      [a, b, c],
      {
        kind: "op",
        op: "subtract",
        left: {
          kind: "op",
          op: "union",
          left: { kind: "leaf", meshIndex: 0 },
          right: { kind: "leaf", meshIndex: 1 },
        },
        right: { kind: "leaf", meshIndex: 2 },
      },
      opts,
    );
    expect(s.measure(merged).closed).toBe(true);

    // Volumetric fallbacks + handle-leaf CsgNode tree
    const sdf = s.booleanBatch(
      {
        kind: "op",
        op: "union",
        left: { kind: "leaf", mesh: a },
        right: { kind: "leaf", mesh: b },
      },
      { method: "bvhSdf", voxelPitch: 0.25 },
    );
    expect(s.inspect.triangleQuality(sdf).liveTriangleCount).toBeGreaterThan(20);

    const edt = s.boolean.union(a, b, { method: "fhEdt", voxelPitch: 0.25, maxTriangles: 120 });
    expect(edt).toBe(a);
    expect(s.inspect.triangleQuality(a).liveTriangleCount).toBeLessThanOrEqual(200);

    // Worker scene tests (#206 async RPCs)
    const worker = await openWorker();
    const wa = worker.addMesh(aData.positions, aData.indices);
    const wb = worker.addMesh(bData.positions, bData.indices);
    const wc = worker.addMesh(cData.positions, cData.indices);
    worker.commitFrame();

    const wBatch: MeshHandle = await worker.boolean.batch(
      {
        kind: "op",
        op: "subtract",
        left: {
          kind: "op",
          op: "union",
          left: { kind: "leaf", mesh: wa },
          right: { kind: "leaf", mesh: wb },
        },
        right: { kind: "leaf", mesh: wc },
      },
      { method: "manifold", snapSlivers: true },
    );
    worker.commitFrame();
    expect(worker.views(wBatch).positions().length).toBeGreaterThan(0);
    const wMeasured = await worker.inspect.measure(wBatch);
    expect(wMeasured.closed).toBe(true);

    const wUnion = await worker.boolean.union(wa, wb, { method: "bvhSdf", voxelPitch: 0.25 });
    expect(wUnion).toBe(wa);

    const wCut = await worker.boolean.cut(wBatch, { origin: [0, 0, 0], normal: [1, 0, 0] }, { cap: true });
    expect(wCut.kept).toBe(wBatch);
    expect(wCut.keptTriangles).toBeGreaterThan(0);
    expect(wCut.removedTriangles).toBeGreaterThan(0);
    expect(wCut.capTriangles).toBeGreaterThan(0);
    worker.commitFrame();
    expect(worker.views(wCut.removed).positions().length).toBeGreaterThan(0);
  });

  it("#209 & #213: overhangs, triangleQuality, meshDeviation, geodesics, shortestSurfacePath, and trimByCurve", async () => {
    const sph = sphereMesh([0, 0, 0], 1, 10, 20);
    const sphShifted = sphereMesh([0.2, 0, 0], 1, 10, 20);
    const grid = gridMesh(6, 6);

    // Sync scene
    const s = openSync();
    const sm = s.addMesh(sph.positions, sph.indices);
    const ss = s.addMesh(sphShifted.positions, sphShifted.indices);
    const sg = s.addMesh(grid.positions, grid.indices);

    const ov = s.inspect.overhangs(sm, { buildDirection: [0, 0, 1], criticalAngleDeg: 45 });
    expect(ov.overhangFaceCount).toBeGreaterThan(0);
    expect(ov.totalSupportArea).toBeGreaterThan(0);
    expect(ov.vertexAnglesDeg.length).toBe(s.vertexSlots(sm));

    const qual = s.inspect.triangleQuality(sm, { metric: "aspectRatio", sliverThreshold: 12 });
    expect(qual.liveTriangleCount).toBe(s.views(sm).indices().length / 3);
    expect(qual.max).toBeGreaterThanOrEqual(qual.min);
    expect(qual.vertexQuality.length).toBe(s.vertexSlots(sm));

    const dev = s.inspect.meshDeviation(sm, ss, { signed: false });
    expect(dev.max).toBeGreaterThan(0.15);
    expect(dev.rms).toBeGreaterThan(0);
    expect(dev.vertexDeviations.length).toBe(s.vertexSlots(sm));

    const geoDijkstra = s.inspect.geodesicDistances(sg, 0, { method: "dijkstra" });
    const geoHeat = s.inspect.geodesicDistances(sg, [0], { method: "heatMethod" });
    expect(geoDijkstra[0]).toBeCloseTo(0, 5);
    expect(geoDijkstra[geoDijkstra.length - 1]).toBeGreaterThan(0.8);
    expect(geoHeat[geoHeat.length - 1]).toBeGreaterThan(0.5);

    const path = s.inspect.shortestSurfacePath(sg, 0, 35, { method: "dijkstra", distances: geoDijkstra });
    expect(path.length).toBeGreaterThan(1);
    expect(path[0].triangle).toBeGreaterThanOrEqual(0);

    // Closed square loop on flat unit grid -> trims interior triangles undoably
    const trim = s.boolean.trimByCurve(
      sg,
      [
        [-0.2, -0.2, 0],
        [0.2, -0.2, 0],
        [0.2, 0.2, 0],
        [-0.2, 0.2, 0],
      ],
      { closed: true, deleteInside: true },
    );
    expect(trim.kept).toBe(sg);
    expect(trim.seamEdges).toBeGreaterThan(0);
    expect(trim.trianglesRemoved).toBeGreaterThan(0);

    // Worker RPC parity (#209 + #213)
    const worker = await openWorker();
    const wm = worker.addMesh(sph.positions, sph.indices);
    const ws = worker.addMesh(sphShifted.positions, sphShifted.indices);
    const wg = worker.addMesh(grid.positions, grid.indices);
    worker.commitFrame();

    const [wOv, wQual, wDev, wGeo, wPath] = await Promise.all([
      worker.inspect.overhangs(wm, { buildDirection: [0, 0, 1], criticalAngleDeg: 45 }),
      worker.inspect.triangleQuality(wm, "aspectRatio", 12),
      worker.inspect.meshDeviation(wm, ws, { signed: false }),
      worker.inspect.geodesicDistances(wg, 0, { method: "dijkstra" }),
      worker.inspect.shortestSurfacePath(wg, 0, 35, { method: "dijkstra" }),
    ]);
    expect(wOv.overhangFaceCount).toBe(ov.overhangFaceCount);
    expect(wQual.liveTriangleCount).toBe(qual.liveTriangleCount);
    expect(wDev.max).toBeCloseTo(dev.max, 4);
    expect(wGeo[wGeo.length - 1]).toBeCloseTo(geoDijkstra[geoDijkstra.length - 1], 4);
    expect(wPath.length).toBe(path.length);

    const wTrim = await worker.boolean.trimByCurve(
      wg,
      [
        [-0.2, -0.2, 0],
        [0.2, -0.2, 0],
        [0.2, 0.2, 0],
        [-0.2, 0.2, 0],
      ],
      { closed: true, deleteInside: true },
    );
    expect(wTrim.kept).toBe(wg);
    expect(wTrim.trianglesRemoved).toBe(trim.trianglesRemoved);
    expect(worker.canUndo).toBe(true);
  });

  it("#214: alignByLandmarks, registerMultistage, and detectSymmetryPlane on Scene & WorkerScene", async () => {
    const fixedBox = boxMesh(0, 0, 0, 2, 1, 0.5);
    const movingBox = boxMesh(0, 0, 0, 2, 1, 0.5);

    const s = openSync();
    const sf = s.addMesh(fixedBox.positions, fixedBox.indices);
    const sm = s.addMesh(movingBox.positions, movingBox.indices);
    const theta = 0.45;
    const shift: Vec3 = [3, -2, 1];
    const xform = (p: Vec3): Vec3 => {
      const c = Math.cos(theta);
      const sn = Math.sin(theta);
      return [c * p[0] - sn * p[1] + shift[0], sn * p[0] + c * p[1] + shift[1], p[2] + shift[2]];
    };
    const targets: Vec3[] = [
      [2, 1, 0.5],
      [-2, 1, 0.5],
      [-2, -1, -0.5],
    ];
    const pairs = targets.map((t) => ({ source: xform(t), target: t }));

    s.setPlacement(sm, { rotation: rotZ(theta), translation: shift });

    // Umeyama landmarks from rotated+shifted world corners recover target placement
    const lm = s.align.alignByLandmarks(sm, sf, pairs, false);
    expect(lm.rmse).toBeLessThan(1e-5);
    expect(s.placement(sm).translation[0]).toBeCloseTo(0, 4);

    // Perturb mesh and run multi-stage OBB coarse + symmetricNormal ICP
    s.setPlacement(sm, { rotation: rotZ(0.35), translation: [1.5, -1.0, 0.4] });
    const reg = s.align.registerMultistage(sm, sf, {
      coarseMethod: "orientedBoundingBox",
      objective: "symmetricNormal",
      maxIterations: 40,
    });
    expect(reg.transform.length).toBe(16);
    expect(reg.finalRmse).toBeLessThan(0.05);

    const sym = s.align.detectSymmetryPlane(sf);
    const symAlias = s.align.detectSymmetry(sf);
    expect(sym.symmetryScore).toBeGreaterThan(0.9);
    expect(sym.symmetryErrorRms).toBeLessThan(1e-4);
    expect(symAlias.symmetryScore).toBeCloseTo(sym.symmetryScore, 5);

    // Worker RPC parity
    const worker = await openWorker();
    const wf = worker.addMesh(fixedBox.positions, fixedBox.indices);
    const wm = worker.addMesh(movingBox.positions, movingBox.indices);
    worker.commitFrame();

    worker.setPlacement(wm, { rotation: rotZ(theta), translation: shift });
    const wLm = await worker.align.alignByLandmarks(wm, wf, pairs, false);
    expect(wLm.rmse).toBeLessThan(1e-5);
    worker.commitFrame();
    expect(worker.placement(wm).translation[0]).toBeCloseTo(0, 4);

    worker.setPlacement(wm, { rotation: rotZ(0.35), translation: [1.5, -1.0, 0.4] });
    const wReg = await worker.align.registerMultistage(wm, wf, {
      coarseMethod: "pcaCentroid",
      objective: "pointToPlane",
      maxIterations: 40,
    });
    expect(wReg.finalRmse).toBeLessThan(0.1);
    worker.commitFrame();
    expect(worker.placement(wm).translation[0]).toBeCloseTo(wReg.placement.translation[0], 4);

    const wSym = await worker.align.detectSymmetryPlane(wf);
    expect(wSym.symmetryScore).toBeGreaterThan(0.9);
  });
});
