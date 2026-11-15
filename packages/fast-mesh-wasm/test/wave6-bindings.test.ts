// Wave 6 SDK & Worker RPC tests for Issues #233, #193, #194, #234, and #236.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_COLLISION_RULE,
  type MeshHandle,
  packPointCloud,
  parsePointCloudBytes,
  parsePointCloudText,
  pointCloudEstimateNormals,
  pointCloudRor,
  pointCloudSor,
  pointCloudVoxelDownsample,
  Scene,
  sliceContoursExport,
  unpackPointCloud,
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

function gridMesh(nx = 6, ny = 6, size = 2) {
  const positions: number[] = [];
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      positions.push(((x / (nx - 1)) - 0.5) * size, ((y / (ny - 1)) - 0.5) * size, 0);
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

function cylinderMesh(radius = 1, height = 2, slices = 24) {
  const positions: number[] = [];
  const indices: number[] = [];
  const h = height * 0.5;
  // Tilted + rotated cylinder or axis-aligned along Y
  const bottomCenter = 0;
  positions.push(0, -h, 0);
  const topCenter = 1;
  positions.push(0, h, 0);
  for (let i = 0; i < slices; i++) {
    const t = (2 * Math.PI * i) / slices;
    positions.push(radius * Math.cos(t), -h, radius * Math.sin(t));
  }
  for (let i = 0; i < slices; i++) {
    const t = (2 * Math.PI * i) / slices;
    positions.push(radius * Math.cos(t), h, radius * Math.sin(t));
  }
  for (let i = 0; i < slices; i++) {
    const j = (i + 1) % slices;
    const b0 = 2 + i;
    const b1 = 2 + j;
    const t0 = 2 + slices + i;
    const t1 = 2 + slices + j;
    indices.push(bottomCenter, b0, b1);
    indices.push(topCenter, t1, t0);
    indices.push(b0, t0, t1, b0, t1, b1);
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

function rotX(theta: number): number[] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [1, 0, 0, 0, c, -s, 0, s, c];
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

describe("Wave 6 bindings (#233, #193, #194, #234, #236) on Scene & WorkerScene", () => {
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

  it("#233: screen, sphere, crease, shell, islands, topology ops, and extract/split/delete on Scene & WorkerScene", async () => {
    // Column-major ortho projection scaling world [-5..5] into [-1..1], z [-10..10] -> [-1..1]
    const orthoViewProj = [
      0.2, 0, 0, 0,
      0, 0.2, 0, 0,
      0, 0, -0.1, 0,
      0, 0, 0, 1,
    ];

    const runChecks = async (
      s: Scene | WorkerScene,
      flush: () => void,
    ) => {
      const b1 = boxMesh(0, 0, 0, 1, 1, 1);
      const b2 = boxMesh(10, 0, 0, 0.4, 0.4, 0.4);
      // Merged mesh with two disconnected shells so connect/islands work cleanly
      const mergedPos = new Float64Array(b1.positions.length + b2.positions.length);
      mergedPos.set(b1.positions, 0);
      mergedPos.set(b2.positions, b1.positions.length);
      const mergedIdx = new Uint32Array(b1.indices.length + b2.indices.length);
      mergedIdx.set(b1.indices, 0);
      for (let i = 0; i < b2.indices.length; i++) mergedIdx[b1.indices.length + i] = b2.indices[i] + 8;

      const mesh = s.addMesh(mergedPos, mergedIdx, { origin: [0, 0, 0] });
      s.setPlacement(mesh, { rotation: IDENTITY, translation: [2, 0, 0] });
      flush();

      // Screen rect covering placed big box (world x in [1,3] -> ndc [0.2, 0.6])
      const screenRect = await s.repair.selectFacesScreen(
        mesh,
        { kind: "rect", minNdc: [0.15, -0.5], maxNdc: [0.65, 0.5] },
        {
          viewProj: orthoViewProj,
          cameraOrigin: [0, 0, 10],
          cameraForward: [0, 0, -1],
          isOrthographic: true,
          visibleOnly: true,
          rejectBackfaces: true,
        },
      );
      expect(screenRect.length).toBeGreaterThan(0);

      // Screen lasso missing world box
      const screenLassoMiss = await s.repair.selectFacesScreen(
        mesh,
        { kind: "lasso", polygonNdc: [[-0.95, -0.95], [-0.8, -0.95], [-0.8, -0.8]] },
        {
          viewProj: orthoViewProj,
          cameraOrigin: [0, 0, 10],
          cameraForward: [0, 0, -1],
          isOrthographic: true,
        },
      );
      expect(screenLassoMiss.length).toBe(0);

      // World sphere near placed center [2, 0, 0] with normal cone along +Z
      const sphereSel = await s.repair.selectFacesInSphere(
        mesh,
        [2, 0, 1],
        1.6,
        { normal: [0, 0, 1], angleRad: 0.4 },
      );
      expect(sphereSel.length).toBe(2);

      // Crease angle selects 2 triangles of one flat face
      const crease = await s.repair.selectByCreaseAngle(mesh, 0, 0.1);
      expect(crease.length).toBe(2);

      // Connected shell on seed tri 0 -> 12 faces of first box
      const shell = await s.repair.selectConnectedShell(mesh, 0);
      expect(shell.length).toBe(12);

      // Small face islands (maxFaceCount = 12 selects only the 12-tri island next to a connected 168-tri sphere)
      const bigShell = sphereMesh([0, 0, 0], 1, 8, 12);
      const subTiny = boxMesh(8, 0, 0, 0.5, 0.5, 0.5); // 12 tris
      const twoIslandsPos = new Float64Array(bigShell.positions.length + subTiny.positions.length);
      twoIslandsPos.set(bigShell.positions, 0);
      twoIslandsPos.set(subTiny.positions, bigShell.positions.length);
      const twoIslandsIdx = new Uint32Array(bigShell.indices.length + subTiny.indices.length);
      twoIslandsIdx.set(bigShell.indices, 0);
      const bigVerts = bigShell.positions.length / 3;
      for (let i = 0; i < subTiny.indices.length; i++) {
        twoIslandsIdx[bigShell.indices.length + i] = subTiny.indices[i] + bigVerts;
      }
      const islandMesh = s.addMesh(twoIslandsPos, twoIslandsIdx);
      flush();
      const smallIslands = await s.repair.selectSmallFaceIslands(islandMesh, 12);
      expect(smallIslands.length).toBe(12);

      // Grow, shrink, hole fill, boundary smooth, vertex mask roundtrip
      const grown = await s.repair.growFaceSelection(mesh, [0], 1, "edge");
      expect(grown.length).toBeGreaterThan(1);
      const shrunk = await s.repair.shrinkFaceSelection(mesh, grown, 1, "edge");
      expect(shrunk.length).toBeLessThan(grown.length);
      const filledHoles = await s.repair.fillSelectionHoles(mesh, [0], 2);
      expect(filledHoles.length).toBeGreaterThanOrEqual(1);
      const smoothed = await s.repair.smoothSelectionBoundary(mesh, grown, 1);
      expect(smoothed.length).toBeGreaterThan(0);

      const vMask = await s.repair.faceSelectionToVertexMask(mesh, shell);
      const rtFaces = await s.repair.vertexMaskToFaceSelection(mesh, vMask, 0.5, true);
      expect(Array.from(rtFaces)).toEqual(Array.from(shell));

      // Extract preserves world placement/origin
      const extracted = await s.repair.extractSelectedFaces(mesh, shell);
      flush();
      const bExt = s.bounds(extracted)!;
      const pExt = s.placement(extracted);
      expect(bExt).not.toBeNull();
      // World center of box1 was placement.translation + 0 = [2,0,0]
      expect(pExt.translation[0] + bExt.center[0]).toBeCloseTo(2, 4);

      // Split into detached + delete on remaining
      const split = await s.repair.splitSelectedFaces(mesh, [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]);
      flush();
      expect(split.removedTriangles).toBe(12);
      const bDet = s.bounds(split.detached)!;
      const pDet = s.placement(split.detached);
      expect(pDet.translation[0] + bDet.center[0]).toBeCloseTo(12, 4);

      const del = await s.repair.deleteSelectedFaces(mesh, [0, 1]);
      flush();
      expect(del.removedTriangles).toBe(2);
    };

    const sync = openSync();
    await runChecks(sync, () => sync.commitFrame());

    const worker = await openWorker();
    await runChecks(worker, () => worker.commitFrame());
  });

  it("#193: FFD, ARAP, Kelvinlet, and XPBD (incl. selfCollision true/false & protection) on Scene & WorkerScene", async () => {
    const runChecks = async (s: Scene | WorkerScene, flush: () => void) => {
      const grid = gridMesh(6, 6, 2);
      const m = s.addMesh(grid.positions, grid.indices);
      flush();

      // FFD (bernstein)
      const lattice = await s.deform.buildFfdLattice(m, [3, 3, 2], "bernstein", 0.1);
      expect(lattice.dims).toEqual([3, 3, 2]);
      expect(lattice.offsets.length).toBe(3 * 3 * 2 * 3);
      lattice.offsets[2] = 0.4; // displace +Z at corner 0
      const ffdRep = await s.deform.applyFfd(m, { lattice });
      flush();
      expect(ffdRep.verticesMoved).toBeGreaterThan(0);

      // FFD (bspline alias + protection: weight 0 freezes, 0.5 halves move)
      const zBeforeProt = s.views(m).positions()[2];
      const z1Before = s.views(m).positions()[1 * 3 + 2];
      const zUnprotBefore = s.views(m).positions()[35 * 3 + 2];
      const bsplineLat = await s.deform.buildFfdLattice(m, [4, 4, 2], "bspline", 0.05);
      bsplineLat.offsets.fill(0.15);
      const protRep = await s.deform.applyFfd(m, {
        lattice: bsplineLat,
        protect: { vertices: [0, 1], weights: [0, 0.5] },
      });
      flush();
      expect(protRep.verticesEvaluated).toBe(36);
      expect(s.views(m).positions()[2]).toBeCloseTo(zBeforeProt, 4); // weight 0 -> frozen
      expect(s.views(m).positions()[1 * 3 + 2] - z1Before).toBeCloseTo(0.075, 2); // weight 0.5 -> halved
      expect(s.views(m).positions()[35 * 3 + 2]).toBeGreaterThan(zUnprotBefore + 0.05);

      // ARAP
      const arapRep = await s.deform.applyArap(m, {
        anchors: [{ vertex: 0 }],
        handles: [{ vertex: 35, target: [1, 1, 0.5] }],
        weights: "cotangent",
        iterations: 5,
      });
      flush();
      expect(arapRep.verticesMoved).toBeGreaterThan(0);

      // Kelvinlet (grab + pinch + symmetry)
      const kelvRep = await s.deform.applyKelvinlet(m, {
        center: [0, 0, 0],
        load: { kind: "grab", force: [0, 0, 0.4] },
        epsilon: 0.4,
        scale: "bi",
        symmetry: { x: true },
      });
      flush();
      expect(kelvRep.verticesMoved).toBeGreaterThan(0);

      // XPBD selfCollision: false vs true on fresh cloth mesh
      const clothA = gridMesh(5, 5, 2);
      const mNoSelf = s.addMesh(clothA.positions, clothA.indices);
      const mWithSelf = s.addMesh(clothA.positions.slice(), clothA.indices);
      flush();

      const xpbdFalse = await s.deform.applyXpbd(mNoSelf, {
        substeps: 2,
        iterations: 4,
        pins: [{ vertex: 0, target: [-1, -1, 0] }],
        handles: [{ vertex: 24, target: [-0.9, -0.9, 0.02], compliance: 0 }],
        selfCollision: false,
      });
      const xpbdTrue = await s.deform.applyXpbd(mWithSelf, {
        substeps: 2,
        iterations: 4,
        pins: [{ vertex: 0, target: [-1, -1, 0] }],
        handles: [{ vertex: 24, target: [-0.9, -0.9, 0.02], compliance: 0 }],
        selfCollision: true,
        collisionThickness: 0.15,
      });
      flush();
      expect(xpbdFalse.verticesMoved).toBeGreaterThan(0);
      expect(xpbdTrue.verticesMoved).toBeGreaterThan(0);
      expect(xpbdFalse.selfCollisionContactsResolved).toBe(0);
    };

    const sync = openSync();
    await runChecks(sync, () => sync.commitFrame());

    const worker = await openWorker();
    await runChecks(worker, () => worker.commitFrame());
  });

  it("#194: collision rules, Scene/WorkerScene drag rigid clamp + cut + softDeform, and restart persistence", async () => {
    // 1) Sync Scene fast-path & rigid/cut/softDeform
    const s = openSync();
    const aData = boxMesh(0, 0, 0, 1, 1, 1);
    const mover = s.addMesh(aData.positions, aData.indices);
    const obstacle = s.addMesh(aData.positions, aData.indices);
    s.setPlacement(obstacle, { rotation: IDENTITY, translation: [4, 0, 0] });
    s.commitFrame();

    expect(s.hasActiveCollisionResponse(mover)).toBe(false);
    expect(s.defaultCollisionRule()).toEqual(DEFAULT_COLLISION_RULE);
    expect(() => s.setCollisionRule(mover, mover, { mode: "rigid" })).toThrow();

    // Fast path drag passes right through when PassThrough
    const passDrag = s.drag(mover);
    passDrag.move({ rotation: IDENTITY, translation: [4, 0, 0] });
    expect(s.placement(mover).translation[0]).toBeCloseTo(4, 4);
    passDrag.cancel();
    expect(s.placement(mover).translation[0]).toBeCloseTo(0, 4);

    // Rigid rule clamps drag & moveWithCollision
    s.setCollisionRule(mover, obstacle, { mode: "rigid", rigidSlide: false });
    expect(s.hasActiveCollisionResponse(mover)).toBe(true);
    expect(s.collisionRule(mover, obstacle).mode).toBe("rigid");

    const rigidDrag = s.drag(mover);
    rigidDrag.move({ rotation: IDENTITY, translation: [3.5, 0, 0] });
    expect(s.placement(mover).translation[0]).toBeLessThanOrEqual(2.05);
    rigidDrag.end();
    expect(s.placement(mover).translation[0]).toBeLessThanOrEqual(2.05);
    s.commitFrame();

    // Direct moveWithCollision + commitCollisionInteraction
    s.setPlacement(mover, { rotation: IDENTITY, translation: [0, 0, 0] });
    const stepRep = s.moveWithCollision(mover, { rotation: IDENTITY, translation: [3.2, 0, 0] }, true);
    expect(stepRep.activePairs.length).toBeGreaterThan(0);
    expect(stepRep.activePairs[0].other).toBe(obstacle);
    expect(stepRep.appliedPlacement.translation[0]).toBeLessThanOrEqual(2.05);
    s.commitCollisionInteraction(mover);

    // Cut rule (onRelease) cuts target only after end()
    s.setPlacement(mover, { rotation: IDENTITY, translation: [0, 0, 0] });
    s.setCollisionRule(mover, obstacle, {
      mode: "cut",
      role: "movingIsCutter",
      cutTrigger: "onRelease",
      cutMethod: "manifold",
    });
    const beforeVerts = s.views(obstacle).positions().length / 3;
    const cutDrag = s.drag(mover);
    cutDrag.move({ rotation: IDENTITY, translation: [3.2, 0, 0] });
    // Before release, previewOnly; obstacle untouched
    expect(s.views(obstacle).positions().length / 3).toBe(beforeVerts);
    cutDrag.end();
    s.commitFrame();
    expect(s.bounds(obstacle)).not.toBeNull();

    // Soft deform elastic vs plastic on thin subdivided sphere vs pusher box
    const sp = sphereMesh([0, 0, 0], 1, 10, 18);
    const soft = s.addMesh(sp.positions, sp.indices);
    const pusher = s.addMesh(aData.positions, aData.indices);
    s.setPlacement(pusher, { rotation: IDENTITY, translation: [3, 0, 0] });
    s.commitFrame();

    s.setCollisionRule(pusher, soft, {
      mode: "softDeform",
      role: "movingIsCutter",
      softSolver: "sdfLaplacian",
      materialMemory: "elastic",
      compliance: 0.2,
    });
    const softPosSnapshot = s.views(soft).positions().slice();
    s.moveWithCollision(pusher, { rotation: IDENTITY, translation: [1.3, 0, 0] });
    expect(s.views(soft).positions().some((v, i) => Math.abs(v - softPosSnapshot[i]) > 1e-4)).toBe(true);
    // Retract and verify elastic recovery
    s.moveWithCollision(pusher, { rotation: IDENTITY, translation: [4.0, 0, 0] }, true);
    s.commitCollisionInteraction(pusher);
    const maxElasticResidual = Math.max(...Array.from(s.views(soft).positions(), (v, i) => Math.abs(v - softPosSnapshot[i])));

    // Plastic retains permanent deformation after retract
    s.setCollisionRule(pusher, soft, {
      mode: "softDeform",
      role: "movingIsCutter",
      softSolver: "sdfLaplacian",
      materialMemory: "plastic",
      compliance: 0.2,
    });
    s.moveWithCollision(pusher, { rotation: IDENTITY, translation: [1.3, 0, 0] });
    s.moveWithCollision(pusher, { rotation: IDENTITY, translation: [4.0, 0, 0] }, true);
    s.commitCollisionInteraction(pusher);
    const maxPlasticResidual = Math.max(...Array.from(s.views(soft).positions(), (v, i) => Math.abs(v - softPosSnapshot[i])));
    expect(maxPlasticResidual).toBeGreaterThan(maxElasticResidual + 1e-3);

    // 2) WorkerScene drag authoritative clamping + restart() rule survival
    const w = await openWorker();
    const wMover = w.addMesh(aData.positions, aData.indices);
    const wWall = w.addMesh(aData.positions, aData.indices);
    w.setPlacement(wWall, { rotation: IDENTITY, translation: [4, 0, 0] });
    w.commitFrame();

    w.setCollisionRule(wMover, wWall, { mode: "rigid", rigidSlide: false });
    expect(w.hasActiveCollisionResponse(wMover)).toBe(true);
    expect(w.collisionRule(wMover, wWall).mode).toBe("rigid");

    const wDrag = w.drag(wMover);
    wDrag.move({ rotation: IDENTITY, translation: [3.6, 0, 0] });
    // Optimistically moved ahead before Worker processes it
    expect(w.placement(wMover).translation[0]).toBeCloseTo(3.6, 4);
    w.commitFrame();

    // Wait for Worker result by sequencing a lightweight query, then commitFrame
    await w.bounds(wMover); // wait until batch drains (or measure)
    await w.inspect.measure(wMover);
    w.commitFrame();
    // Authoritative rigid clamp overrides optimistic local pose EVEN BEFORE drag.end()!
    expect(w.placement(wMover).translation[0]).toBeLessThanOrEqual(2.05);

    // Try pushing deeper when already at boundary (zero delta -> #forcePlacements handles it!)
    wDrag.move({ rotation: IDENTITY, translation: [3.8, 0, 0] });
    expect(w.placement(wMover).translation[0]).toBeCloseTo(3.8, 4);
    w.commitFrame();
    await w.inspect.measure(wMover);
    w.commitFrame();
    expect(w.placement(wMover).translation[0]).toBeLessThanOrEqual(2.05);
    wDrag.end();
    w.commitFrame();
    await w.inspect.measure(wMover);
    w.commitFrame();
    expect(w.placement(wMover).translation[0]).toBeLessThanOrEqual(2.05);

    // Restart worker and verify pair collision rule still survives
    await w.restart();
    expect(w.hasActiveCollisionResponse(wMover)).toBe(true);
    expect(w.collisionRule(wMover, wWall).mode).toBe("rigid");
    const postRestart = await w.moveWithCollision(wMover, { rotation: IDENTITY, translation: [3.5, 0, 0] }, true);
    w.commitFrame();
    expect(postRestart.appliedPlacement.translation[0]).toBeLessThanOrEqual(2.05);
    expect(w.clearCollisionRule(wMover, wWall)).toBe(true);
    expect(w.hasActiveCollisionResponse(wMover)).toBe(false);
  });

  it("#234: fitPrimitive, alignToFittedPrimitive, contours & export, point cloud helpers & reconstruction on Scene & WorkerScene", async () => {
    const runChecks = async (s: Scene | WorkerScene, flush: () => void) => {
      // Sphere fit in world space
      const sp = sphereMesh([0, 0, 0], 2.0, 12, 24);
      const spMesh = s.addMesh(sp.positions, sp.indices);
      s.setPlacement(spMesh, { rotation: IDENTITY, translation: [3, -1, 2] });
      flush();

      const fit = await s.inspect.fitPrimitive(spMesh, { kind: "sphere" });
      expect(fit.primitive.kind).toBe("sphere");
      if (fit.primitive.kind === "sphere") {
        expect(fit.primitive.radius).toBeCloseTo(2.0, 2);
        expect(fit.primitive.center[0]).toBeCloseTo(3, 2);
        expect(fit.primitive.center[1]).toBeCloseTo(-1, 2);
        expect(fit.primitive.center[2]).toBeCloseTo(2, 2);
      }
      expect(fit.rmsError).toBeLessThan(0.05);

      // Cylinder alignment
      const cyl = cylinderMesh(1, 4, 24);
      const cylMesh = s.addMesh(cyl.positions, cyl.indices);
      s.setPlacement(cylMesh, { rotation: rotX(Math.PI / 2), translation: [0, 5, 0] });
      flush();
      const cylFit = await s.inspect.fitPrimitive(cylMesh, { kind: "cylinder" });
      expect(cylFit.primitive.kind).toBe("cylinder");
      const alignedPlacement = await s.inspect.alignToFittedPrimitive(cylMesh, cylFit.primitive, [0, 1, 0], [0, 0, 0]);
      flush();
      expect(alignedPlacement.translation[1]).toBeCloseTo(0, 1);

      // Contours + stack + SVG/JSON/CSV export
      const box = boxMesh(0, 0, 0, 1, 1, 1);
      const boxHandle = s.addMesh(box.positions, box.indices);
      flush();
      const slice = await s.inspect.sliceContours(boxHandle, [0, 0, 0], [0, 0, 1]);
      expect(slice.loops.length).toBe(1);
      expect(slice.totalArea).toBeCloseTo(4, 2);
      const stack = await s.inspect.sliceContourStack(boxHandle, [0, 0, -0.5], [0, 0, 1], 0.5, 3);
      expect(stack.length).toBe(3);

      const svg = sliceContoursExport(stack, "svg");
      const json = sliceContoursExport(slice, "json");
      const csv = sliceContoursExport(slice, "csv");
      expect(svg).toContain("<svg");
      expect(JSON.parse(json)).toBeDefined();
      expect(csv.split("\n").length).toBeGreaterThan(2);

      // Point cloud pack/unpack/parse/filters/normal estimation + surface reconstruction
      const cloudPts: number[] = [];
      const cloudNormals: number[] = [];
      for (let i = 0; i < 6; i++) {
        const phi = (Math.PI * (i + 0.5)) / 6;
        for (let j = 0; j < 12; j++) {
          const theta = (2 * Math.PI * j) / 12;
          const x = Math.sin(phi) * Math.cos(theta);
          const y = Math.sin(phi) * Math.sin(theta);
          const z = Math.cos(phi);
          cloudPts.push(x, y, z);
          cloudNormals.push(x, y, z);
        }
      }
      const packed = packPointCloud(Float32Array.from(cloudPts), Float32Array.from(cloudNormals));
      const unpacked = unpackPointCloud(packed);
      expect(unpacked.count).toBe(72);

      const xyz = parsePointCloudText("0 0 0\n1 0 0\n0 1 0\n0 0 1\n1 1 1\n", "xyz");
      expect(unpackPointCloud(xyz).count).toBe(5);
      const plyText = "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n";
      const parsedPly = parsePointCloudBytes(new TextEncoder().encode(plyText));
      expect(unpackPointCloud(parsedPly).count).toBe(4);

      const sor = pointCloudSor(packed, 6, 2.0);
      expect(sor.report.keptCount).toBeGreaterThan(0);
      const ror = pointCloudRor(packed, 2.0, 2);
      expect(ror.report.keptCount).toBeGreaterThan(0);
      const vox = pointCloudVoxelDownsample(packed, 0.3);
      expect(unpackPointCloud(vox).count).toBeLessThanOrEqual(72);
      const est = pointCloudEstimateNormals(packPointCloud(Float32Array.from(cloudPts)), 8, [0, 0, 5]);
      expect(unpackPointCloud(est).normals).not.toBeNull();

      const recon = await s.inspect.reconstructPointCloud(packed, {
        method: "implicitMlsMarchingCubes",
        voxelSize: 0.35,
      });
      flush();
      expect(s.bounds(recon)).not.toBeNull();
    };

    const sync = openSync();
    await runChecks(sync, () => sync.commitFrame());

    const worker = await openWorker();
    await runChecks(worker, () => worker.commitFrame());
  });

  it("#236: hollowMesh, punchDrainHole (with/without plug), and placeMesh (keepSeparate/embedUnion) on Scene & WorkerScene", async () => {
    const runChecks = async (s: Scene | WorkerScene, flush: () => void) => {
      // Hollow box
      const b = boxMesh(0, 0, 0, 1.5, 1.5, 1.5);
      const boxA = s.addMesh(b.positions, b.indices);
      flush();
      const hollowRep = await s.boolean.hollowMesh(boxA, { wallThickness: 0.25, backend: "bvhSdf", voxelSize: 0.25 });
      flush();
      expect(hollowRep.cavitiesCreated).toBe(1);
      expect(hollowRep.cavityTriangles).toBeGreaterThan(12);

      // Drain hole without plug, then with plug
      const b2 = boxMesh(0, 0, 0, 1.5, 1.5, 1.5);
      const target = s.addMesh(b2.positions, b2.indices);
      flush();
      const drainNoPlug = await s.boolean.punchDrainHole(target, {
        surfacePoint: [0, 0, 1.5],
        surfaceNormal: [0, 0, 1],
        diameter: 0.5,
        depth: 1.0,
        generatePlug: false,
      });
      flush();
      expect(drainNoPlug.holesPunched).toBe(1);
      expect(drainNoPlug.plugs.length).toBe(0);

      const drainWithPlug = await s.boolean.punchDrainHole(target, {
        surfacePoint: [0, 0, -1.5],
        surfaceNormal: [0, 0, -1],
        diameter: 0.5,
        depth: 1.0,
        generatePlug: true,
        plugClearance: 0.02,
      });
      flush();
      expect(drainWithPlug.holesPunched).toBe(1);
      expect(drainWithPlug.plugs.length).toBe(1);
      expect(s.bounds(drainWithPlug.plugs[0])).not.toBeNull();

      // Place mesh keepSeparate + embedUnion
      const baseData = boxMesh(0, 0, 0, 2, 2, 1);
      const stampData = boxMesh(0, 0, 0, 0.3, 0.3, 0.5);
      const base = s.addMesh(baseData.positions, baseData.indices);
      const stamp = s.addMesh(stampData.positions, stampData.indices);
      flush();

      const sepRep = await s.boolean.placeMesh(base, stamp, {
        hitPoint: [0, 0, 1],
        hitNormal: [0, 0, 1],
        upAxis: "posZ",
        embedDepth: 0.1,
        mode: "keepSeparate",
      });
      flush();
      expect(sepRep.added).not.toBeNull();
      expect(s.bounds(sepRep.added as MeshHandle)).not.toBeNull();

      const unionRep = await s.boolean.placeMesh(base, stamp, {
        hitPoint: [0.6, 0, 1],
        hitNormal: [0, 0, 1],
        upAxis: "posZ",
        embedDepth: 0.2,
        mode: "embedUnion",
        booleanMethod: "manifold",
      });
      flush();
      expect(unionRep.added).toBeNull();
      expect(unionRep.keptTriangles).toBeGreaterThan(12);
    };

    const sync = openSync();
    await runChecks(sync, () => sync.commitFrame());

    const worker = await openWorker();
    await runChecks(worker, () => worker.commitFrame());
  });
});
