// Real-WASM integration tests for Issue #195 workbench operations:
// Boolean CSG (+ keepOriginals), Destructive Plane Cut (+/-/cap), Offset/Shell,
// Quadric Simplify + Loop Subdivide, Rigid ICP + 1 undo step, Distance/Thickness/Curvature fields
// (sync Scene and WorkerScene bridge), Transform Scale, Duplicate & Mirror winding, and Camera framing.
import { readFileSync } from "node:fs";
import {
  type EditOp,
  type MeshHandle,
  type PlacementInput,
  type RegisterOptions,
  type RegisterResult,
  type RepairOp,
  Scene,
  type SceneLike,
} from "fast-mesh-wasm";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import * as THREE from "three";
import { afterEach, describe, expect, it } from "vitest";
import { AppState, type CameraPreset, type Item } from "../src/scene/app-state";
import { applyMeshScale, eulerDegToMat3, mat3ToEulerDeg } from "../src/scene/transform-math";
import {
  applySurfaceField,
  computeGaussianCurvature,
  duplicateMesh,
  mirrorMesh,
  runBooleanCsg,
  runIcpAlign,
  runOffset,
  runPlaneCut,
  runShell,
  runSimplify,
  runSubdivide,
} from "../src/scene/workbench-ops";
import {
  applyCameraPose,
  computeSceneBounds,
  orthoToPerspectiveDistance,
  perspectiveToOrthoHalfExtents,
  presetPose,
} from "../src/viewport/camera";

function makeBoxGeometry(
  hx: number,
  hy: number,
  hz: number,
  cx = 0,
  cy = 0,
  cz = 0,
): { positions: Float64Array; indices: Uint32Array } {
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
    0, 2, 1, 0, 3, 2, // -Z
    4, 5, 6, 4, 6, 7, // +Z
    0, 1, 5, 0, 5, 4, // -Y
    2, 3, 7, 2, 7, 6, // +Y
    0, 4, 7, 0, 7, 3, // -X
    1, 2, 6, 1, 6, 5, // +X
  ]);
  return { positions, indices };
}

function addBoxItem(
  store: SceneStore,
  app: AppState,
  name: string,
  opts: {
    hx?: number;
    hy?: number;
    hz?: number;
    cx?: number;
    cy?: number;
    cz?: number;
    tx?: number;
    ty?: number;
    tz?: number;
    withAttributes?: boolean;
  } = {},
): Item {
  const hx = opts.hx ?? 5;
  const hy = opts.hy ?? 5;
  const hz = opts.hz ?? 5;
  const { positions, indices } = makeBoxGeometry(hx, hy, hz, opts.cx ?? 0, opts.cy ?? 0, opts.cz ?? 0);
  const vertCount = positions.length / 3;
  const colors = opts.withAttributes
    ? Uint8Array.from(Array.from({ length: vertCount * 4 }, (_, idx) => (idx % 4 === 3 ? 255 : (idx * 31) % 256)))
    : undefined;
  const uvs = opts.withAttributes
    ? Float32Array.from(Array.from({ length: vertCount * 2 }, (_, idx) => (idx % 2 === 0 ? 0.25 : 0.75)))
    : undefined;
  const item = app.addMesh(store, {
    name,
    positions,
    indices,
    ...(colors ? { colors } : {}),
    ...(uvs ? { uvs } : {}),
    center: [opts.cx ?? 0, opts.cy ?? 0, opts.cz ?? 0],
    radius: Math.hypot(hx, hy, hz),
    volumeId: null,
    preset: null,
  });
  if (opts.tx !== undefined || opts.ty !== undefined || opts.tz !== undefined) {
    store.setPlacement(item.mesh, {
      rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      translation: [opts.tx ?? 0, opts.ty ?? 0, opts.tz ?? 0],
    });
  }
  return item;
}

function liveTriangles(scene: SceneLike, mesh: MeshHandle): number {
  const idx = scene.views(mesh).indices();
  let count = 0;
  for (let k = 0; k + 2 < idx.length; k += 3) {
    if (idx[k] !== 0 || idx[k + 1] !== 0 || idx[k + 2] !== 0) {
      count++;
    }
  }
  return count;
}

function signedVolume(positions: ArrayLike<number>, indices: ArrayLike<number>): number {
  let vol6 = 0;
  for (let k = 0; k + 2 < indices.length; k += 3) {
    const a = indices[k];
    const b = indices[k + 1];
    const c = indices[k + 2];
    if (a === 0 && b === 0 && c === 0) continue;
    const ax = positions[3 * a], ay = positions[3 * a + 1], az = positions[3 * a + 2];
    const bx = positions[3 * b], by = positions[3 * b + 1], bz = positions[3 * b + 2];
    const cx = positions[3 * c], cy = positions[3 * c + 1], cz = positions[3 * c + 2];
    vol6 += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
  }
  return vol6 / 6;
}

function createWorkerHarness(): {
  syncScene: Scene;
  workerStore: SceneStore;
  app: AppState;
  dispose: () => void;
} {
  const syncScene = new Scene();
  const realStore = createSceneStore(syncScene);
  const unsupported = (name: string) => () => {
    throw new Error(`${name} requires synchronous Scene`);
  };
  const workerScene = {
    status: () => ({ busy: false, pending: 0 }),
    views: (mesh: MeshHandle) => syncScene.views(mesh),
    placement: (mesh: MeshHandle) => syncScene.placement(mesh),
    matrix4: (mesh: MeshHandle) => syncScene.matrix4(mesh),
    bounds: (mesh: MeshHandle) => syncScene.bounds(mesh),
    origin: (mesh: MeshHandle) => syncScene.origin(mesh),
    onChanges: (cb: Parameters<Scene["onChanges"]>[0]) => syncScene.onChanges(cb),
    addMesh: (pos: Float64Array | Float32Array, idx: Uint32Array, opts?: Parameters<Scene["addMesh"]>[2]) =>
      syncScene.addMesh(pos, idx, opts),
    removeMesh: (mesh: MeshHandle) => syncScene.removeMesh(mesh),
    repair: (mesh: MeshHandle, op: RepairOp) => syncScene.repair(mesh, op),
    edit: (mesh: MeshHandle, ops: EditOp[]) => syncScene.edit(mesh, ops),
    setPlacement: (mesh: MeshHandle, p: PlacementInput) => syncScene.setPlacement(mesh, p),
    boolean: Object.assign(unsupported("boolean"), {
      cut: unsupported("boolean.cut"),
    }),
    cutByPlane: unsupported("cutByPlane"),
    align: {
      register: (moving: MeshHandle, fixed: MeshHandle, opts?: RegisterOptions): Promise<RegisterResult> =>
        Promise.resolve(syncScene.align.register(moving, fixed, opts)),
    },
    inspect: {
      setColoring: unsupported("inspect.setColoring"),
      removeField: unsupported("inspect.removeField"),
    },
  } as unknown as SceneLike;

  const coloringRangeMap = new Map<MeshHandle, { distanceTo?: MeshHandle; maxDistance?: number }>();

  const workerStore = {
    scene: workerScene,
    getSnapshot: () => realStore.getSnapshot(),
    subscribe: (cb: Parameters<SceneStore["subscribe"]>[0]) => realStore.subscribe(cb),
    select: (m: MeshHandle | null) => realStore.select(m),
    addMesh: (pos: Float64Array, idx: Uint32Array, opts?: Parameters<SceneStore["addMesh"]>[2]) =>
      realStore.addMesh(pos, idx, opts),
    removeMesh: (mesh: MeshHandle) => realStore.removeMesh(mesh),
    repair: (mesh: MeshHandle, op: RepairOp) => realStore.repair(mesh, op),
    edit: (mesh: MeshHandle, ops: EditOp[]) => realStore.edit(mesh, ops),
    setPlacement: (mesh: MeshHandle, p: PlacementInput) => realStore.setPlacement(mesh, p),
    setDisplay: (mesh: MeshHandle, d: { coloring?: { distanceTo?: MeshHandle; maxDistance?: number } | null }) => {
      if (d.coloring === null) {
        coloringRangeMap.delete(mesh);
      } else if (d.coloring?.distanceTo !== undefined) {
        coloringRangeMap.set(mesh, d.coloring);
      }
      realStore.setDisplay(mesh, d as Parameters<SceneStore["setDisplay"]>[1]);
    },
    fetchColoringValues: async (mesh: MeshHandle) => {
      const cfg = coloringRangeMap.get(mesh);
      if (!cfg?.distanceTo) return null;
      const field = syncScene.inspect.addDistanceField({
        name: "worker:distance",
        on: mesh,
        to: cfg.distanceTo,
        maxDistance: cfg.maxDistance ?? 10,
      });
      const vals = syncScene.inspect.fieldValues(field);
      syncScene.inspect.removeField(field);
      return vals;
    },
    undo: () => realStore.undo(),
    redo: () => realStore.redo(),
    dispose: () => realStore.dispose(),
  } as unknown as SceneStore;

  const app = new AppState();
  return {
    syncScene,
    workerStore,
    app,
    dispose: () => {
      realStore.dispose();
      syncScene.free();
    },
  };
}

const cleanups: (() => void)[] = [];

afterEach(() => {
  while (cleanups.length > 0) {
    cleanups.pop()?.();
  }
});

function createSyncSetup(): { scene: Scene; store: SceneStore; app: AppState } {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return { scene, store, app };
}

describe("Issue #195 Real-WASM Workbench Integration", () => {
  it("runs Boolean CSG with keepOriginals true and false cleanly tracking resulting meshes", async () => {
    const { scene, store, app } = createSyncSetup();
    const boxA = addBoxItem(store, app, "BoxA", { hx: 5, hy: 5, hz: 5, tx: 0 });
    const boxB = addBoxItem(store, app, "BoxB", { hx: 5, hy: 5, hz: 5, tx: 4 });

    const unionRes = await runBooleanCsg({
      store,
      app,
      target: boxA,
      source: boxB,
      op: "union",
      keepOriginals: true,
    });
    expect(unionRes.triangles).toBeGreaterThan(12);
    expect(app.getSnapshot().items.length).toBe(3);
    expect(app.getSnapshot().items.some((i) => i.mesh === boxA.mesh)).toBe(true);
    expect(app.getSnapshot().items.some((i) => i.mesh === boxB.mesh)).toBe(true);
    expect(app.getSnapshot().items.some((i) => i.mesh === unionRes.mesh)).toBe(true);
    expect(liveTriangles(scene, unionRes.mesh)).toBe(unionRes.triangles);

    const diffRes = await runBooleanCsg({
      store,
      app,
      target: boxA,
      source: boxB,
      op: "subtract",
      keepOriginals: false,
    });
    expect(diffRes.triangles).toBeGreaterThan(0);
    expect(app.getSnapshot().items.some((i) => i.mesh === boxB.mesh)).toBe(false);
    expect(app.getSnapshot().items.some((i) => i.mesh === diffRes.mesh)).toBe(true);
  });

  it("runs Boolean CSG and destructive Plane Cut through WorkerScene fallback", async () => {
    const { workerStore, app, dispose } = createWorkerHarness();
    cleanups.push(dispose);
    const a = addBoxItem(workerStore, app, "WorkerA", { hx: 5, hy: 5, hz: 5, tx: 0 });
    const b = addBoxItem(workerStore, app, "WorkerB", { hx: 5, hy: 5, hz: 5, tx: 4 });

    const res = await runBooleanCsg({
      store: workerStore,
      app,
      target: a,
      source: b,
      op: "intersect",
      keepOriginals: true,
    });
    expect(app.getSnapshot().items.length).toBe(3);
    expect(res.triangles).toBeGreaterThan(0);

    const cutReport = await runPlaneCut({
      store: workerStore,
      mesh: a.mesh,
      axis: "x",
      offset: 0,
      cap: true,
      keepSide: "+",
    });
    expect(cutReport.keptTriangles).toBeGreaterThan(0);
    const pos = workerStore.scene.views(a.mesh).positions();
    const idx = workerStore.scene.views(a.mesh).indices();
    for (let k = 0; k + 2 < idx.length; k += 3) {
      if (idx[k] === 0 && idx[k + 1] === 0 && idx[k + 2] === 0) continue;
      expect(pos[3 * idx[k]]).toBeGreaterThanOrEqual(-1e-4);
      expect(pos[3 * idx[k + 1]]).toBeGreaterThanOrEqual(-1e-4);
      expect(pos[3 * idx[k + 2]]).toBeGreaterThanOrEqual(-1e-4);
    }
  });

  it("cuts mesh destructively on + and - sides with capping", async () => {
    const { scene, store, app } = createSyncSetup();
    const boxPlus = addBoxItem(store, app, "CutPlus", { hx: 6, hy: 4, hz: 4 });
    const rPlus = await runPlaneCut({ store, mesh: boxPlus.mesh, axis: "x", offset: 0, cap: true, keepSide: "+" });
    expect(rPlus.keptTriangles).toBeGreaterThan(0);
    expect(rPlus.capTriangles).toBeGreaterThan(0);
    const posP = scene.views(boxPlus.mesh).positions();
    const idxP = scene.views(boxPlus.mesh).indices();
    for (let k = 0; k + 2 < idxP.length; k += 3) {
      if (idxP[k] === 0 && idxP[k + 1] === 0 && idxP[k + 2] === 0) continue;
      expect(posP[3 * idxP[k]]).toBeGreaterThanOrEqual(-1e-4);
    }

    const boxMinus = addBoxItem(store, app, "CutMinus", { hx: 6, hy: 4, hz: 4 });
    const rMinus = await runPlaneCut({ store, mesh: boxMinus.mesh, axis: "y", offset: 0, cap: true, keepSide: "-" });
    expect(rMinus.keptTriangles).toBeGreaterThan(0);
    const posM = scene.views(boxMinus.mesh).positions();
    const idxM = scene.views(boxMinus.mesh).indices();
    for (let k = 0; k + 2 < idxM.length; k += 3) {
      if (idxM[k] === 0 && idxM[k + 1] === 0 && idxM[k + 2] === 0) continue;
      expect(posM[3 * idxM[k] + 1]).toBeLessThanOrEqual(1e-4);
    }
  });

  it("runs Offset (auto/normal/bvhSdf/fhEdt), Shell, Loop Subdivide, and Quadric Simplify with undo", async () => {
    const { scene, store, app } = createSyncSetup();
    const item = addBoxItem(store, app, "ResampleBox", { hx: 5, hy: 5, hz: 5 });

    const offAuto = await runOffset({ store, mesh: item.mesh, distance: 1, algorithm: "auto" });
    expect(offAuto.moved).toBeGreaterThan(0);
    await store.undo();

    const offBvh = await runOffset({ store, mesh: item.mesh, distance: 0.5, algorithm: "bvhSdf", voxelSize: 0.5 });
    expect(offBvh.trianglesAdded).toBeGreaterThan(0);
    await store.undo();

    const offEdt = await runOffset({ store, mesh: item.mesh, distance: 0.5, algorithm: "fhEdt", voxelSize: 0.5 });
    expect(offEdt.trianglesAdded).toBeGreaterThan(0);
    await store.undo();

    const shellReport = await runShell({ store, mesh: item.mesh, thickness: 1.5 });
    expect(shellReport.verticesAdded).toBeGreaterThan(0);
    expect(liveTriangles(scene, item.mesh)).toBeGreaterThan(12);
    await store.undo();
    expect(liveTriangles(scene, item.mesh)).toBe(12);

    await runSubdivide({ store, mesh: item.mesh, levels: 2 });
    expect(liveTriangles(scene, item.mesh)).toBe(192);

    await runSimplify({ store, mesh: item.mesh, targetTriangles: 64, preserveBoundary: true });
    expect(liveTriangles(scene, item.mesh)).toBeLessThanOrEqual(80);

    await store.undo();
    expect(liveTriangles(scene, item.mesh)).toBe(192);
  });

  it("runs Rigid ICP alignment reducing RMSE and reverting in one undo step", async () => {
    const { scene, store, app } = createSyncSetup();
    const fixed = addBoxItem(store, app, "FixedBox", { hx: 8, hy: 5, hz: 3 });
    await runSubdivide({ store, mesh: fixed.mesh, levels: 2 });

    const moving = addBoxItem(store, app, "MovingBox", {
      hx: 8,
      hy: 5,
      hz: 3,
      tx: 1.2,
      ty: -0.6,
      tz: 0.4,
    });
    await runSubdivide({ store, mesh: moving.mesh, levels: 2 });

    const res = await runIcpAlign({
      store,
      moving: moving.mesh,
      fixed: fixed.mesh,
      maxIterations: 40,
      overlap: 1,
    });
    expect(res.rmse).toBeLessThan(0.25);
    expect(res.iterations).toBeGreaterThan(0);
    const alignedTx = scene.placement(moving.mesh).translation;
    expect(Math.hypot(alignedTx[0], alignedTx[1], alignedTx[2])).toBeLessThan(0.3);

    await store.undo();
    const revertedTx = scene.placement(moving.mesh).translation;
    expect(revertedTx[0]).toBeCloseTo(1.2, 5);
    expect(revertedTx[1]).toBeCloseTo(-0.6, 5);
    expect(revertedTx[2]).toBeCloseTo(0.4, 5);
  });

  it("computes Distance, Thickness, Mean Curvature, and Gaussian Curvature on sync Scene and WorkerScene", async () => {
    const { scene, store, app } = createSyncSetup();
    const a = addBoxItem(store, app, "FieldA", { hx: 4, hy: 4, hz: 4 });
    await runSubdivide({ store, mesh: a.mesh, levels: 1 });
    const b = addBoxItem(store, app, "FieldB", { hx: 4, hy: 4, hz: 4, tx: 6 });

    for (const mode of ["distance", "thickness", "meanCurvature", "gaussianCurvature"] as const) {
      const res = await applySurfaceField({
        store,
        app,
        state: { mode, mesh: a.mesh, target: b.mesh, map: "viridis", min: 0, max: 10, maxDistance: 20 },
        autoRange: true,
      });
      expect(res.values).not.toBeNull();
      expect(res.values!.length).toBeGreaterThan(0);
    }

    const gauss = computeGaussianCurvature(scene.views(a.mesh).positions(), scene.views(a.mesh).indices());
    expect(Array.from(gauss).some((v) => Number.isFinite(v))).toBe(true);

    // WorkerScene path
    const w = createWorkerHarness();
    cleanups.push(w.dispose);
    const wa = addBoxItem(w.workerStore, w.app, "WA", { hx: 4, hy: 4, hz: 4 });
    const wb = addBoxItem(w.workerStore, w.app, "WB", { hx: 4, hy: 4, hz: 4, tx: 6 });
    const wThick = await applySurfaceField({
      store: w.workerStore,
      app: w.app,
      state: { mode: "thickness", mesh: wa.mesh, target: wb.mesh, map: "grayscale", min: 0, max: 10, maxDistance: 20 },
      autoRange: true,
    });
    expect(wThick.usesOverlay).toBe(true);
    expect(wThick.summary!.max).toBeGreaterThan(0);

    const wCurv = await applySurfaceField({
      store: w.workerStore,
      app: w.app,
      state: { mode: "meanCurvature", mesh: wa.mesh, target: null, map: "coolwarm", min: -1, max: 1, maxDistance: 10 },
      autoRange: true,
    });
    expect(wCurv.usesOverlay).toBe(true);
    expect(wCurv.values!.length).toBeGreaterThan(0);
  });

  it("scales mesh local vertices, duplicates attributes/placement, mirrors with outward winding, and frames 6 camera presets", async () => {
    const { scene, store, app } = createSyncSetup();
    const geom = makeBoxGeometry(3, 4, 5, 2, 0, 0);
    const uvs = Float32Array.from(Array.from({ length: 8 * 2 }, (_, idx) => (idx % 2 === 0 ? 0.25 : 0.75)));
    const orig = app.addMesh(store, {
      name: "PartA",
      positions: geom.positions,
      indices: geom.indices,
      uvs,
      center: [0, 0, 0],
      radius: Math.hypot(5, 4, 5),
      volumeId: null,
      preset: null,
    });
    store.setPlacement(orig.mesh, {
      rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      translation: [10, -2, 3],
    });

    const rot = eulerDegToMat3([15, -30, 45]);
    const backDeg = mat3ToEulerDeg(rot);
    expect(backDeg[0]).toBeCloseTo(15, 3);
    expect(backDeg[1]).toBeCloseTo(-30, 3);
    expect(backDeg[2]).toBeCloseTo(45, 3);

    await applyMeshScale(store, orig.mesh, [2, 1, 1]);
    const scaledBounds = scene.bounds(orig.mesh)!;
    expect(scaledBounds.max[0] - scaledBounds.min[0]).toBeCloseTo(12, 4);
    await store.undo();

    const refMesh = addBoxItem(store, app, "RefBox", { hx: 2, hy: 2, hz: 2, tx: 18 });
    store.setDisplay(orig.mesh, { coloring: { distanceTo: refMesh.mesh, maxDistance: 20, map: "viridis", range: [0, 20] } });
    await new Promise((r) => setTimeout(r, 30));

    const copy = duplicateMesh(store, app, orig);
    await new Promise((r) => setTimeout(r, 30));
    expect(copy.name).toBe("PartA copy");
    expect(scene.placement(copy.mesh).translation).toEqual([10, -2, 3]);
    expect(Array.from(scene.views(copy.mesh).uvs()!)).toEqual(Array.from(scene.views(orig.mesh).uvs()!));
    expect(scene.views(orig.mesh).colors()).not.toBeNull();
    expect(Array.from(scene.views(copy.mesh).colors()!)).toEqual(Array.from(scene.views(orig.mesh).colors()!));

    await mirrorMesh(store, copy.mesh, "x");
    const mirBounds = scene.bounds(copy.mesh)!;
    expect(mirBounds.min[0]).toBeCloseTo(-5, 4);
    expect(mirBounds.max[0]).toBeCloseTo(1, 4);
    expect(signedVolume(scene.views(copy.mesh).positions(), scene.views(copy.mesh).indices())).toBeCloseTo(480, 3);

    const bounds = computeSceneBounds(app.getSnapshot().items, scene, app);
    const persp = new THREE.PerspectiveCamera(45, 16 / 9, 0.1, 1000);
    const ortho = new THREE.OrthographicCamera(-10, 10, 10, -10, -100, 100);
    for (const preset of ["front", "back", "left", "right", "top", "bottom"] as CameraPreset[]) {
      const pose = presetPose(preset, bounds);
      const dx = pose.position[0] - pose.target[0];
      const dy = pose.position[1] - pose.target[1];
      const dz = pose.position[2] - pose.target[2];
      const dot = (dx * pose.up[0] + dy * pose.up[1] + dz * pose.up[2]) / Math.hypot(dx, dy, dz);
      expect(Math.abs(dot)).toBeLessThan(1e-6);
      applyCameraPose(persp, pose, 16 / 9, "perspective");
      applyCameraPose(ortho, pose, 16 / 9, "orthographic");
      expect(ortho.top).toBeCloseTo(pose.halfHeight, 5);
    }

    const { halfHeight } = perspectiveToOrthoHalfExtents(45, 20, 1.5);
    expect(orthoToPerspectiveDistance(halfHeight, -halfHeight, 1, 45)).toBeCloseTo(20, 8);
  });

  it("places base .camera-bar before @media (pointer: coarse) so coarse touch hides it without wrapping desktop .toolbar", () => {
    const stylesCss = readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
    const coarseMatch = /@media\s*\(\s*pointer\s*:\s*coarse\s*\)\s*\{/.exec(stylesCss);
    expect(coarseMatch).not.toBeNull();
    const coarseStart = coarseMatch!.index;
    let depth = 0;
    let coarseEnd = stylesCss.length;
    for (let i = stylesCss.indexOf("{", coarseStart); i < stylesCss.length; i++) {
      const ch = stylesCss[i];
      if (ch === "{") depth++;
      else if (ch === "}") {
        depth--;
        if (depth === 0) {
          coarseEnd = i + 1;
          break;
        }
      }
    }

    const beforeCoarse = stylesCss.slice(0, coarseStart);
    const coarseBlock = stylesCss.slice(coarseStart, coarseEnd);
    const afterCoarse = stylesCss.slice(coarseEnd);

    const baseCameraIdx = beforeCoarse.indexOf(".camera-bar");
    expect(baseCameraIdx).toBeGreaterThanOrEqual(0);
    expect(beforeCoarse.slice(baseCameraIdx)).toMatch(/\.camera-bar\s*\{[^}]*display\s*:\s*flex/);
    expect(coarseBlock).toMatch(/\.camera-bar\s*\{[^}]*display\s*:\s*none/);
    expect(afterCoarse).not.toMatch(/\.camera-bar[^{]*\{[^}]*display\s*:/);
    expect(afterCoarse).not.toMatch(/\.camera-bar\b/);
    expect(beforeCoarse).not.toMatch(/\.toolbar\s*\{[^}]*flex-wrap\s*:\s*wrap/);
  });
});
