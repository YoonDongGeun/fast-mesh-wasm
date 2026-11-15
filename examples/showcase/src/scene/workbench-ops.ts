// Unified workbench operation bridge (#195):
// Boolean CSG, Destructive Plane Cut, Offset/Shell, Resampling, Rigid ICP Align,
// Duplicate/Mirror, and Surface Field Analysis (Contact Distance, Thickness, Mean/Gaussian Curvature),
// working seamlessly with both synchronous `Scene` and `WorkerScene`.
import type {
  BooleanOp,
  CutReport,
  EditOp,
  FieldHandle,
  MeshHandle,
  OffsetAlgorithm,
  OffsetReport,
  Placement,
  RegisterResult,
  Scene,
  SceneLike,
  SectionPlane,
  ShellReport,
  SimplifyReport,
  SubdivideReport,
} from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { bounds, type Vec3 } from "../load/mesh-data";
import type { AppState, Item, SurfaceColormap, SurfaceFieldState } from "./app-state";
import { selectMesh } from "./selection";
import { liveVertexIds } from "./transform-math";

export type MirrorAxis = "x" | "y" | "z";
export type BooleanOperation = BooleanOp;
export type CutAxis = "x" | "y" | "z" | "custom";
export type CutKeepSide = "+" | "-";
export type OffsetModeAlgorithm = "auto" | OffsetAlgorithm;

export interface LiveMeshSnapshot {
  positions: Float64Array;
  indices: Uint32Array;
  colors?: Uint8Array;
  uvs?: Float32Array;
  origin: Vec3;
  placement: Placement;
  center: Vec3;
  radius: number;
  /** Total vertex slot count in the source mesh and mapping from compacted index -> original slot. */
  sourceVertexSlots: number;
  compactedToSource: Uint32Array;
}

/** True when `scene` runs inside a Worker host (`WorkerScene`), false for synchronous `Scene`. */
export function isWorkerScene(scene: SceneLike): scene is WorkerScene {
  return "status" in scene;
}

function asSyncScene(scene: SceneLike): Scene {
  return scene as unknown as Scene;
}

/**
 * Extracts live, compacted mesh buffers (skipping dead `[0,0,0]` triangle slots)
 * alongside optional vertex colors, UVs, origin, and placement.
 */
export function extractLiveMeshData(scene: SceneLike, mesh: MeshHandle): LiveMeshSnapshot {
  const v = scene.views(mesh);
  const srcPos = v.positions();
  const srcIdx = v.indices();
  const srcCol = v.colors?.();
  const srcUv = v.uvs?.();
  const originFn = (scene as { origin?: (m: MeshHandle) => Vec3 }).origin;
  const origin: Vec3 = typeof originFn === "function" ? originFn.call(scene, mesh) : [0, 0, 0];
  const placement = scene.placement(mesh);

  const vertCount = Math.floor(srcPos.length / 3);
  const used = new Uint8Array(vertCount);
  let liveTriCount = 0;

  for (let k = 0; k + 2 < srcIdx.length; k += 3) {
    const a = srcIdx[k];
    const b = srcIdx[k + 1];
    const c = srcIdx[k + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    liveTriCount++;
    used[a] = 1;
    used[b] = 1;
    used[c] = 1;
  }

  const remap = new Int32Array(vertCount).fill(-1);
  let liveVertCount = 0;
  for (let i = 0; i < vertCount; i++) {
    if (used[i] === 1) {
      remap[i] = liveVertCount++;
    }
  }

  // Fallback for vertex-only / all-dead edge case: keep slot 0..vertCount-1 intact
  if (liveTriCount === 0 && vertCount > 0) {
    const positions = new Float64Array(vertCount * 3);
    for (let i = 0; i < vertCount * 3; i++) {
      positions[i] = srcPos[i];
    }
    const { center, radius } = bounds(positions);
    const directMap = new Uint32Array(vertCount);
    for (let i = 0; i < vertCount; i++) directMap[i] = i;
    return {
      positions,
      indices: new Uint32Array(0),
      ...(srcCol ? { colors: new Uint8Array(srcCol.subarray(0, vertCount * 4)) } : {}),
      ...(srcUv ? { uvs: new Float32Array(srcUv.subarray(0, vertCount * 2)) } : {}),
      origin: [origin[0], origin[1], origin[2]],
      placement: {
        rotation: [...placement.rotation],
        translation: [placement.translation[0], placement.translation[1], placement.translation[2]],
        ...(placement.scale ? { scale: [...placement.scale] as [number, number, number] } : {}),
      },
      center,
      radius,
      sourceVertexSlots: vertCount,
      compactedToSource: directMap,
    };
  }

  const positions = new Float64Array(liveVertCount * 3);
  const compactedToSource = new Uint32Array(liveVertCount);
  const colors = srcCol ? new Uint8Array(liveVertCount * 4) : undefined;
  const uvs = srcUv ? new Float32Array(liveVertCount * 2) : undefined;
  for (let oldV = 0; oldV < vertCount; oldV++) {
    const nv = remap[oldV];
    if (nv === -1) continue;
    compactedToSource[nv] = oldV;
    positions[3 * nv] = srcPos[3 * oldV];
    positions[3 * nv + 1] = srcPos[3 * oldV + 1];
    positions[3 * nv + 2] = srcPos[3 * oldV + 2];
    if (colors && srcCol) {
      colors[4 * nv] = srcCol[4 * oldV];
      colors[4 * nv + 1] = srcCol[4 * oldV + 1];
      colors[4 * nv + 2] = srcCol[4 * oldV + 2];
      colors[4 * nv + 3] = srcCol[4 * oldV + 3];
    }
    if (uvs && srcUv) {
      uvs[2 * nv] = srcUv[2 * oldV];
      uvs[2 * nv + 1] = srcUv[2 * oldV + 1];
    }
  }

  const indices = new Uint32Array(liveTriCount * 3);
  let outT = 0;
  for (let k = 0; k + 2 < srcIdx.length; k += 3) {
    const a = srcIdx[k];
    const b = srcIdx[k + 1];
    const c = srcIdx[k + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    indices[outT++] = remap[a];
    indices[outT++] = remap[b];
    indices[outT++] = remap[c];
  }

  const { center, radius } = bounds(positions);
  return {
    positions,
    indices,
    ...(colors ? { colors } : {}),
    ...(uvs ? { uvs } : {}),
    origin: [origin[0], origin[1], origin[2]],
    placement: {
      rotation: [...placement.rotation],
      translation: [placement.translation[0], placement.translation[1], placement.translation[2]],
      ...(placement.scale ? { scale: [...placement.scale] as [number, number, number] } : {}),
    },
    center,
    radius,
    sourceVertexSlots: vertCount,
    compactedToSource,
  };
}

function applyPlacementToStore(store: SceneStore, mesh: MeshHandle, placement: Placement): void {
  store.setPlacement(mesh, {
    rotation: [...placement.rotation],
    translation: [placement.translation[0], placement.translation[1], placement.translation[2]],
    ...(placement.scale ? { scale: [placement.scale[0], placement.scale[1], placement.scale[2]] } : {}),
  });
}

/** Clones an existing mesh's live geometry, attributes, and placement as `"<name> copy"`. */
export function duplicateMesh(store: SceneStore, app: AppState, item: Item): Item {
  const snap = extractLiveMeshData(store.scene, item.mesh);
  const copy = app.addMesh(store, {
    name: `${item.name} copy`,
    positions: snap.positions,
    indices: snap.indices,
    ...(snap.colors ? { colors: snap.colors } : {}),
    ...(snap.uvs ? { uvs: snap.uvs } : {}),
    center: [0, 0, 0],
    radius: snap.radius,
    volumeId: item.volumeId,
    preset: item.preset,
  });
  applyPlacementToStore(store, copy.mesh, snap.placement);
  const srcColoring = store.getSnapshot().meshes.get(item.mesh)?.display.coloring ?? null;
  if (srcColoring !== null) {
    store.setDisplay(copy.mesh, { coloring: { ...srcColoring } });
  }
  selectMesh(app, (m) => store.select(m), store.getSnapshot().selection, copy.mesh, false);
  return copy;
}

/**
 * Mirrors a mesh across the chosen local X/Y/Z axis and reverses every live triangle
 * winding `(a, b, c) -> (a, c, b)` so outward normals are preserved, in a single undo step.
 */
export async function mirrorMesh(store: SceneStore, mesh: MeshHandle, axis: MirrorAxis): Promise<void> {
  const axisIdx = axis === "x" ? 0 : axis === "y" ? 1 : 2;
  const views = store.scene.views(mesh);
  const srcPos = views.positions();
  const srcIdx = views.indices();
  const matrix = store.scene.matrix4(mesh);

  const vertCount = Math.floor(srcPos.length / 3);
  const vertices = liveVertexIds(vertCount, srcIdx);
  if (vertices.length === 0) {
    return;
  }

  // Transform mirrored local coordinates into world space because Scene.edit takes world positions.
  const e0 = matrix[0];
  const e1 = matrix[1];
  const e2 = matrix[2];
  const e4 = matrix[4];
  const e5 = matrix[5];
  const e6 = matrix[6];
  const e8 = matrix[8];
  const e9 = matrix[9];
  const e10 = matrix[10];
  const e12 = matrix[12];
  const e13 = matrix[13];
  const e14 = matrix[14];

  const worldPos = new Float64Array(vertices.length * 3);
  for (let i = 0; i < vertices.length; i++) {
    const slot = vertices[i];
    let lx = srcPos[3 * slot];
    let ly = srcPos[3 * slot + 1];
    let lz = srcPos[3 * slot + 2];
    if (axisIdx === 0) lx = lx === 0 ? 0 : -lx;
    else if (axisIdx === 1) ly = ly === 0 ? 0 : -ly;
    else lz = lz === 0 ? 0 : -lz;

    worldPos[3 * i] = e0 * lx + e4 * ly + e8 * lz + e12;
    worldPos[3 * i + 1] = e1 * lx + e5 * ly + e9 * lz + e13;
    worldPos[3 * i + 2] = e2 * lx + e6 * ly + e10 * lz + e14;
  }

  const liveTris: number[] = [];
  const flippedTris: number[] = [];
  for (let k = 0; k + 2 < srcIdx.length; k += 3) {
    const a = srcIdx[k];
    const b = srcIdx[k + 1];
    const c = srcIdx[k + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    liveTris.push(k / 3);
    flippedTris.push(a, c, b);
  }

  const ops: EditOp[] = [{ op: "setPositions", vertices, positions: worldPos }];
  if (liveTris.length > 0) {
    ops.push({ op: "removeTriangles", triangles: liveTris });
    ops.push({ op: "addTriangles", triangles: flippedTris });
  }
  await store.edit(mesh, ops);
}

/**
 * Replaces an existing mesh's geometry in `store` via one `store.edit` batch using world
 * coordinates (used when WorkerScene falls back to the synchronous `Scene` bridge).
 */
async function replaceMeshGeometryLocal(
  store: SceneStore,
  mesh: MeshHandle,
  localPositions: Float64Array,
  indices: Uint32Array,
): Promise<void> {
  const views = store.scene.views(mesh);
  const oldPos = views.positions();
  const oldIdx = views.indices();
  const matrix = store.scene.matrix4(mesh);

  const oldTris: number[] = [];
  for (let k = 0; k + 2 < oldIdx.length; k += 3) {
    if (oldIdx[k] !== 0 || oldIdx[k + 1] !== 0 || oldIdx[k + 2] !== 0) {
      oldTris.push(k / 3);
    }
  }
  const oldVerts = liveVertexIds(Math.floor(oldPos.length / 3), oldIdx);

  const worldPositions = new Float64Array(localPositions.length);
  for (let i = 0; i < localPositions.length; i += 3) {
    const lx = localPositions[i];
    const ly = localPositions[i + 1];
    const lz = localPositions[i + 2];
    worldPositions[i] = matrix[0] * lx + matrix[4] * ly + matrix[8] * lz + matrix[12];
    worldPositions[i + 1] = matrix[1] * lx + matrix[5] * ly + matrix[9] * lz + matrix[13];
    worldPositions[i + 2] = matrix[2] * lx + matrix[6] * ly + matrix[10] * lz + matrix[14];
  }

  const ops: EditOp[] = [];
  if (oldTris.length > 0) {
    ops.push({ op: "removeTriangles", triangles: oldTris });
  }
  if (oldVerts.length > 0) {
    ops.push({ op: "removeVertices", vertices: oldVerts });
  }
  const addVertOpIdx = ops.length;
  ops.push({ op: "addVertices", positions: worldPositions });
  if (indices.length > 0) {
    const triRefs = Array.from(indices, (v) => ({ ref: [addVertOpIdx, v] as const }));
    ops.push({ op: "addTriangles", triangles: triRefs });
  }
  await store.edit(mesh, ops);
}

function opLabel(op: BooleanOperation): string {
  switch (op) {
    case "union":
      return "union";
    case "intersect":
      return "intersection";
    case "subtract":
      return "difference";
  }
}

export interface BooleanRunResult {
  mesh: MeshHandle;
  name: string;
  triangles: number;
}

/** Runs Boolean CSG (`union` | `intersect` | `subtract`) on both sync `Scene` and `WorkerScene`. */
export async function runBooleanCsg(params: {
  store: SceneStore;
  app: AppState;
  target: Item;
  source: Item;
  op: BooleanOperation;
  keepOriginals: boolean;
}): Promise<BooleanRunResult> {
  const { store, app, target, source, op, keepOriginals } = params;
  const label = opLabel(op);

  if (!isWorkerScene(store.scene)) {
    const syncScene = asSyncScene(store.scene);
    if (keepOriginals) {
      const snap = extractLiveMeshData(syncScene, target.mesh);
      const copy = app.addMesh(store, {
        name: `${target.name} ${label} ${source.name}`,
        positions: snap.positions,
        indices: snap.indices,
        ...(snap.colors ? { colors: snap.colors } : {}),
        ...(snap.uvs ? { uvs: snap.uvs } : {}),
        center: [0, 0, 0],
        radius: snap.radius,
        volumeId: null,
        preset: null,
      });
      applyPlacementToStore(store, copy.mesh, snap.placement);
      const resultMesh = syncScene.boolean(copy.mesh, source.mesh, op);
      const liveCount = extractLiveMeshData(syncScene, resultMesh).indices.length / 3;
      selectMesh(app, (m) => store.select(m), store.getSnapshot().selection, resultMesh, false);
      return { mesh: resultMesh, name: copy.name, triangles: liveCount };
    }

    const resultMesh = syncScene.boolean(target.mesh, source.mesh, op);
    if (source.mesh !== target.mesh) {
      app.removeMesh(store, source);
    }
    const liveCount = extractLiveMeshData(syncScene, resultMesh).indices.length / 3;
    selectMesh(app, (m) => store.select(m), store.getSnapshot().selection, resultMesh, false);
    return { mesh: resultMesh, name: target.name, triangles: liveCount };
  }

  // WorkerScene does not yet expose boolean RPCs: execute on an isolated sync Scene bridge cleanly.
  const targetSnap = extractLiveMeshData(store.scene, target.mesh);
  const sourceSnap = extractLiveMeshData(store.scene, source.mesh);
  const { Scene } = await import("fast-mesh-wasm");
  const tmp = new Scene();
  try {
    const tmpTarget = tmp.addMesh(targetSnap.positions, targetSnap.indices, { origin: [0, 0, 0] });
    tmp.setPlacement(tmpTarget, {
      rotation: [...targetSnap.placement.rotation],
      translation: [...targetSnap.placement.translation],
    });
    const tmpSource =
      source.mesh === target.mesh
        ? tmpTarget
        : tmp.addMesh(sourceSnap.positions, sourceSnap.indices, { origin: [0, 0, 0] });
    if (tmpSource !== tmpTarget) {
      tmp.setPlacement(tmpSource, {
        rotation: [...sourceSnap.placement.rotation],
        translation: [...sourceSnap.placement.translation],
      });
    }
    const outHandle = tmp.boolean(tmpTarget, tmpSource, op);
    const outSnap = extractLiveMeshData(tmp, outHandle);

    if (keepOriginals) {
      const added = app.addMesh(store, {
        name: `${target.name} ${label} ${source.name}`,
        positions: outSnap.positions,
        indices: outSnap.indices,
        center: [0, 0, 0],
        radius: outSnap.radius,
        volumeId: null,
        preset: null,
      });
      applyPlacementToStore(store, added.mesh, outSnap.placement);
      selectMesh(app, (m) => store.select(m), store.getSnapshot().selection, added.mesh, false);
      return { mesh: added.mesh, name: added.name, triangles: outSnap.indices.length / 3 };
    }

    await replaceMeshGeometryLocal(store, target.mesh, outSnap.positions, outSnap.indices);
    applyPlacementToStore(store, target.mesh, outSnap.placement);
    if (source.mesh !== target.mesh) {
      app.removeMesh(store, source);
    }
    selectMesh(app, (m) => store.select(m), store.getSnapshot().selection, target.mesh, false);
    return { mesh: target.mesh, name: target.name, triangles: outSnap.indices.length / 3 };
  } finally {
    tmp.free();
  }
}

/** Builds a world-space SectionPlane from axis, custom normal, signed offset, and keep-side (`+` | `-`). */
export function buildCutPlane(
  axis: CutAxis,
  offset: number,
  keepSide: CutKeepSide,
  customNormal?: readonly [number, number, number],
): SectionPlane {
  let nx = 1;
  let ny = 0;
  let nz = 0;
  if (axis === "x") {
    nx = 1;
    ny = 0;
    nz = 0;
  } else if (axis === "y") {
    nx = 0;
    ny = 1;
    nz = 0;
  } else if (axis === "z") {
    nx = 0;
    ny = 0;
    nz = 1;
  } else if (customNormal) {
    const len = Math.hypot(customNormal[0], customNormal[1], customNormal[2]);
    if (len > 1e-12) {
      nx = customNormal[0] / len;
      ny = customNormal[1] / len;
      nz = customNormal[2] / len;
    }
  }
  const cleanZero = (n: number): number => (n === 0 ? 0 : n);
  const origin: [number, number, number] = [
    cleanZero(nx * offset),
    cleanZero(ny * offset),
    cleanZero(nz * offset),
  ];
  const sign = keepSide === "-" ? -1 : 1;
  return {
    origin,
    normal: [cleanZero(nx * sign), cleanZero(ny * sign), cleanZero(nz * sign)],
  };
}

/** Runs destructive plane cut (`boolean.cut` / `cutByPlane`), discarding the non-kept half cleanly. */
export async function runPlaneCut(params: {
  store: SceneStore;
  mesh: MeshHandle;
  axis: CutAxis;
  customNormal?: readonly [number, number, number];
  offset: number;
  cap: boolean;
  keepSide: CutKeepSide;
}): Promise<CutReport> {
  const { store, mesh, axis, customNormal, offset, cap, keepSide } = params;
  const plane = buildCutPlane(axis, offset, keepSide, customNormal);

  if (!isWorkerScene(store.scene)) {
    const syncScene = asSyncScene(store.scene);
    const report = syncScene.boolean.cut(mesh, plane, { cap });
    syncScene.removeMesh(report.removed);
    return report;
  }

  // WorkerScene does not yet expose cutByPlane RPC: run in temporary sync Scene and write back.
  const snap = extractLiveMeshData(store.scene, mesh);
  const { Scene } = await import("fast-mesh-wasm");
  const tmp = new Scene();
  try {
    const tmpMesh = tmp.addMesh(snap.positions, snap.indices, { origin: [0, 0, 0] });
    tmp.setPlacement(tmpMesh, {
      rotation: [...snap.placement.rotation],
      translation: [...snap.placement.translation],
    });
    const report = tmp.boolean.cut(tmpMesh, plane, { cap });
    const outSnap = extractLiveMeshData(tmp, report.kept);
    await replaceMeshGeometryLocal(store, mesh, outSnap.positions, outSnap.indices);
    return { ...report, kept: mesh };
  } finally {
    tmp.free();
  }
}

/** Offsets a mesh using `"auto"` (resolves to `"normal"`), `"normal"`, `"bvhSdf"`, or `"fhEdt"`. */
export async function runOffset(params: {
  store: SceneStore;
  mesh: MeshHandle;
  distance: number;
  algorithm: OffsetModeAlgorithm;
  voxelSize?: number;
}): Promise<OffsetReport> {
  const { store, mesh, distance, algorithm, voxelSize } = params;
  const resolved: OffsetAlgorithm = algorithm === "auto" ? "normal" : algorithm;
  return store.repair(mesh, {
    op: "offset",
    distance,
    algorithm: resolved,
    ...(resolved !== "normal" && voxelSize !== undefined && voxelSize > 0 ? { voxelSize } : {}),
  });
}

/** Creates a solid walled shell with positive wall `thickness`. */
export async function runShell(params: {
  store: SceneStore;
  mesh: MeshHandle;
  thickness: number;
}): Promise<ShellReport> {
  return params.store.repair(params.mesh, {
    op: "shell",
    thickness: params.thickness,
  });
}

/** Quadric mesh decimation (`simplify`). */
export async function runSimplify(params: {
  store: SceneStore;
  mesh: MeshHandle;
  targetTriangles: number;
  preserveBoundary: boolean;
}): Promise<SimplifyReport> {
  return params.store.repair(params.mesh, {
    op: "simplify",
    targetTriangles: Math.max(1, Math.round(params.targetTriangles)),
    preserveBoundary: params.preserveBoundary,
  });
}

/** Loop subdivision (`levels` clamped to `1..3`). */
export async function runSubdivide(params: {
  store: SceneStore;
  mesh: MeshHandle;
  levels: number;
}): Promise<SubdivideReport> {
  const levels = Math.min(3, Math.max(1, Math.round(params.levels)));
  return params.store.repair(params.mesh, {
    op: "subdivide",
    method: "loop",
    levels,
  });
}

/**
 * Rigid ICP registration (`scene.align.register`) applied through `store.setPlacement`
 * so alignment commits as a single undoable placement step.
 */
export async function runIcpAlign(params: {
  store: SceneStore;
  moving: MeshHandle;
  fixed: MeshHandle;
  maxIterations: number;
  overlap: number;
  maxDistance?: number;
}): Promise<RegisterResult> {
  const { store, moving, fixed, maxIterations, overlap, maxDistance } = params;
  if (!store.scene.align) {
    throw new Error("Rigid registration is unavailable");
  }
  const result = await store.scene.align.register(moving, fixed, {
    maxIterations,
    overlap,
    ...(maxDistance !== undefined && maxDistance > 0 ? { maxDistance } : {}),
    apply: false,
  });
  store.setPlacement(moving, {
    rotation: [...result.placement.rotation],
    translation: [result.placement.translation[0], result.placement.translation[1], result.placement.translation[2]],
  });
  return { ...result, applied: true, notApplied: null };
}

/**
 * Computes discrete Gaussian curvature `K = (2*pi - sum(angles)) / A_mixed` (angle defect / area)
 * per vertex slot (`NaN` for vertices not referenced by any live triangle).
 */
export function computeGaussianCurvature(positions: ArrayLike<number>, indices: ArrayLike<number>): Float32Array {
  const nVerts = Math.floor(positions.length / 3);
  const angleSum = new Float64Array(nVerts);
  const areaSum = new Float64Array(nVerts);
  const used = new Uint8Array(nVerts);
  const boundaryEdge = new Map<string, number>();

  const edgeKey = (u: number, v: number) => (u < v ? `${u}:${v}` : `${v}:${u}`);

  for (let k = 0; k + 2 < indices.length; k += 3) {
    const i0 = indices[k];
    const i1 = indices[k + 1];
    const i2 = indices[k + 2];
    if (i0 === 0 && i1 === 0 && i2 === 0) continue;
    if (i0 >= nVerts || i1 >= nVerts || i2 >= nVerts) continue;

    used[i0] = 1;
    used[i1] = 1;
    used[i2] = 1;

    const edges: [number, number][] = [
      [i0, i1],
      [i1, i2],
      [i2, i0],
    ];
    for (const [u, v] of edges) {
      const ek = edgeKey(u, v);
      boundaryEdge.set(ek, (boundaryEdge.get(ek) ?? 0) + 1);
    }

    const ax = positions[3 * i0];
    const ay = positions[3 * i0 + 1];
    const az = positions[3 * i0 + 2];
    const bx = positions[3 * i1];
    const by = positions[3 * i1 + 1];
    const bz = positions[3 * i1 + 2];
    const cx = positions[3 * i2];
    const cy = positions[3 * i2 + 1];
    const cz = positions[3 * i2 + 2];

    const e01x = bx - ax;
    const e01y = by - ay;
    const e01z = bz - az;
    const e02x = cx - ax;
    const e02y = cy - ay;
    const e02z = cz - az;
    const e12x = cx - bx;
    const e12y = cy - by;
    const e12z = cz - bz;

    const l01 = Math.hypot(e01x, e01y, e01z);
    const l02 = Math.hypot(e02x, e02y, e02z);
    const l12 = Math.hypot(e12x, e12y, e12z);
    if (l01 <= 1e-12 || l02 <= 1e-12 || l12 <= 1e-12) continue;

    const nx = e01y * e02z - e01z * e02y;
    const ny = e01z * e02x - e01x * e02z;
    const nz = e01x * e02y - e01y * e02x;
    const triArea = 0.5 * Math.hypot(nx, ny, nz);
    const thirdArea = triArea / 3;

    const clampCos = (v: number) => Math.max(-1, Math.min(1, v));
    const a0 = Math.acos(clampCos((e01x * e02x + e01y * e02y + e01z * e02z) / (l01 * l02)));
    const a1 = Math.acos(clampCos((-e01x * e12x - e01y * e12y - e01z * e12z) / (l01 * l12)));
    const a2 = Math.max(0, Math.PI - a0 - a1);

    angleSum[i0] += a0;
    angleSum[i1] += a1;
    angleSum[i2] += a2;
    areaSum[i0] += thirdArea;
    areaSum[i1] += thirdArea;
    areaSum[i2] += thirdArea;
  }

  const isBoundary = new Uint8Array(nVerts);
  for (const [ek, count] of boundaryEdge) {
    if (count === 1) {
      const sep = ek.indexOf(":");
      const u = Number(ek.slice(0, sep));
      const v = Number(ek.slice(sep + 1));
      isBoundary[u] = 1;
      isBoundary[v] = 1;
    }
  }

  const out = new Float32Array(nVerts);
  out.fill(Number.NaN);
  for (let v = 0; v < nVerts; v++) {
    if (used[v] === 0 || areaSum[v] <= 1e-16) continue;
    const targetAngle = isBoundary[v] === 1 ? Math.PI : 2 * Math.PI;
    out[v] = (targetAngle - angleSum[v]) / areaSum[v];
  }
  return out;
}

const TURBO_STOPS: readonly [number, [number, number, number]][] = [
  [0.0, [0.19, 0.07, 0.23]],
  [0.25, [0.16, 0.48, 0.93]],
  [0.5, [0.18, 0.84, 0.45]],
  [0.75, [0.98, 0.72, 0.14]],
  [1.0, [0.48, 0.02, 0.01]],
];

const VIRIDIS_STOPS: readonly [number, [number, number, number]][] = [
  [0.0, [0.27, 0.0, 0.33]],
  [0.25, [0.23, 0.32, 0.55]],
  [0.5, [0.13, 0.57, 0.55]],
  [0.75, [0.37, 0.79, 0.38]],
  [1.0, [0.99, 0.91, 0.14]],
];

const COOLWARM_STOPS: readonly [number, [number, number, number]][] = [
  [0.0, [0.23, 0.3, 0.75]],
  [0.5, [0.87, 0.87, 0.87]],
  [1.0, [0.71, 0.02, 0.15]],
];

/** Evaluates a built-in colormap at scalar `value` over `[lo, hi]` into linear-friendly RGB floats `0..1`. */
export function evaluateSurfaceColormap(
  map: SurfaceColormap,
  value: number,
  lo: number,
  hi: number,
): [number, number, number] {
  if (!Number.isFinite(value)) {
    return [0.5, 0.5, 0.5];
  }
  const span = hi > lo ? hi - lo : 1;
  const t = Math.max(0, Math.min(1, (value - lo) / span));
  if (map === "grayscale") {
    return [t, t, t];
  }
  const stops = map === "viridis" ? VIRIDIS_STOPS : map === "coolwarm" ? COOLWARM_STOPS : TURBO_STOPS;
  for (let i = 1; i < stops.length; i++) {
    const [t1, c1] = stops[i];
    if (t <= t1 || i === stops.length - 1) {
      const [t0, c0] = stops[i - 1];
      const u = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
      return [
        c0[0] + (c1[0] - c0[0]) * u,
        c0[1] + (c1[1] - c0[1]) * u,
        c0[2] + (c1[2] - c0[2]) * u,
      ];
    }
  }
  return [0.5, 0.5, 0.5];
}

/** Computes finite min, max, and mean summary statistics over scalar vertex values. */
export function summarizeFieldValues(values: Float32Array): { min: number; max: number; mean: number; count: number } | null {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    count++;
  }
  if (count === 0) return null;
  return { min, max, mean: sum / count, count };
}

export interface AppliedSurfaceField {
  field: FieldHandle | null;
  mesh: MeshHandle | null;
  values: Float32Array | null;
  summary: { min: number; max: number; mean: number; count: number } | null;
  usesOverlay: boolean;
}

/** Releases a previously created native field or store distance coloring + overlay values. */
export function clearAppliedSurfaceField(
  store: SceneStore,
  app: AppState,
  prev: { mesh: MeshHandle | null; field: FieldHandle | null },
): void {
  if (prev.field !== null) {
    try {
      store.scene.inspect?.removeField(prev.field);
    } catch {
      // Field or mesh may already have been removed.
    }
  }
  if (prev.mesh !== null) {
    try {
      store.setDisplay(prev.mesh, { coloring: null });
      if (!isWorkerScene(store.scene)) {
        asSyncScene(store.scene).inspect.setColoring(prev.mesh, null);
      }
    } catch {
      // Mesh may have been removed.
    }
  }
  app.setSurfaceFieldValues(null);
}

/**
 * Computes and applies Surface Field Analysis (`distance` | `thickness` | `meanCurvature` | `gaussianCurvature`)
 * identically across synchronous `Scene` and `WorkerScene`.
 */
export async function applySurfaceField(params: {
  store: SceneStore;
  app: AppState;
  state: SurfaceFieldState;
  autoRange?: boolean;
}): Promise<AppliedSurfaceField> {
  const { store, app, state, autoRange = false } = params;
  if (state.mode === "none" || state.mesh === null) {
    app.setSurfaceFieldValues(null);
    return { field: null, mesh: state.mesh, values: null, summary: null, usesOverlay: false };
  }

  const mesh = state.mesh;
  const maxDist = Math.max(state.maxDistance, 1e-4);
  let lo = state.min;
  let hi = state.max;
  if (!(hi > lo)) {
    hi = lo + 1;
  }

  if (state.mode === "gaussianCurvature") {
    store.setDisplay(mesh, { coloring: null });
    const v = store.scene.views(mesh);
    const values = computeGaussianCurvature(v.positions(), v.indices());
    const summary = summarizeFieldValues(values);
    if (autoRange && summary) {
      lo = summary.min;
      hi = summary.max > summary.min ? summary.max : summary.min + 1;
      app.setSurfaceField({ min: lo, max: hi });
    }
    app.setSurfaceFieldValues({ mesh, values, map: state.map, min: lo, max: hi });
    return { field: null, mesh, values, summary, usesOverlay: true };
  }

  if (state.mode === "distance") {
    if (state.target === null || state.target === mesh) {
      throw new Error("Contact distance mode requires another target mesh");
    }
    app.setSurfaceFieldValues(null);
    if (isWorkerScene(store.scene)) {
      store.setDisplay(mesh, {
        coloring: {
          distanceTo: state.target,
          maxDistance: maxDist,
          map: state.map,
          range: [lo, hi],
        },
      });
      const values = await store.fetchColoringValues(mesh);
      const summary = values ? summarizeFieldValues(values) : null;
      if (autoRange && summary) {
        lo = summary.min;
        hi = summary.max > summary.min ? summary.max : summary.min + 1;
        app.setSurfaceField({ min: lo, max: hi });
        store.setDisplay(mesh, {
          coloring: {
            distanceTo: state.target,
            maxDistance: maxDist,
            map: state.map,
            range: [lo, hi],
          },
        });
      }
      return { field: null, mesh, values, summary, usesOverlay: false };
    }

    const syncScene = asSyncScene(store.scene);
    const field = syncScene.inspect.addDistanceField({
      name: `inspect:distance:${Date.now()}`,
      on: mesh,
      to: state.target,
      maxDistance: maxDist,
    });
    const values = syncScene.inspect.fieldValues(field);
    const summary = summarizeFieldValues(values);
    if (autoRange && summary) {
      lo = summary.min;
      hi = summary.max > summary.min ? summary.max : summary.min + 1;
      app.setSurfaceField({ min: lo, max: hi });
    }
    syncScene.inspect.setColoring(mesh, { field, map: state.map, range: [lo, hi] });
    return { field, mesh, values, summary, usesOverlay: false };
  }

  // "thickness" or "meanCurvature"
  if (!isWorkerScene(store.scene)) {
    app.setSurfaceFieldValues(null);
    const syncScene = asSyncScene(store.scene);
    const name = `inspect:${state.mode}:${Date.now()}`;
    const field =
      state.mode === "thickness"
        ? syncScene.inspect.addThicknessField({ name, on: mesh, maxDistance: maxDist })
        : syncScene.inspect.addCurvatureField({ name, on: mesh });
    const values = syncScene.inspect.fieldValues(field);
    const summary = summarizeFieldValues(values);
    if (autoRange && summary) {
      lo = summary.min;
      hi = summary.max > summary.min ? summary.max : summary.min + 1;
      app.setSurfaceField({ min: lo, max: hi });
    }
    syncScene.inspect.setColoring(mesh, { field, map: state.map, range: [lo, hi] });
    return { field, mesh, values, summary, usesOverlay: false };
  }

  // WorkerScene fallback: evaluate engine Thickness / Curvature via temporary sync Scene and drive client overlay.
  store.setDisplay(mesh, { coloring: null });
  const snap = extractLiveMeshData(store.scene, mesh);
  const { Scene } = await import("fast-mesh-wasm");
  const tmp = new Scene();
  try {
    const tmpMesh = tmp.addMesh(snap.positions, snap.indices, { origin: [0, 0, 0] });
    tmp.setPlacement(tmpMesh, {
      rotation: [...snap.placement.rotation],
      translation: [...snap.placement.translation],
    });
    const field =
      state.mode === "thickness"
        ? tmp.inspect.addThicknessField({ name: "inspect:thickness", on: tmpMesh, maxDistance: maxDist })
        : tmp.inspect.addCurvatureField({ name: "inspect:curvature", on: tmpMesh });
    const rawValues = tmp.inspect.fieldValues(field);
    const values = new Float32Array(snap.sourceVertexSlots);
    values.fill(Number.NaN);
    for (let nv = 0; nv < rawValues.length && nv < snap.compactedToSource.length; nv++) {
      values[snap.compactedToSource[nv]] = rawValues[nv];
    }
    const summary = summarizeFieldValues(values);
    if (autoRange && summary) {
      lo = summary.min;
      hi = summary.max > summary.min ? summary.max : summary.min + 1;
      app.setSurfaceField({ min: lo, max: hi });
    }
    app.setSurfaceFieldValues({ mesh, values, map: state.map, min: lo, max: hi });
    return { field: null, mesh, values, summary, usesOverlay: true };
  } finally {
    tmp.free();
  }
}
