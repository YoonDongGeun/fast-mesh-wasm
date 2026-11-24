// The print-prep ops (#236) on a picked surface point, each through the
// scene's boolean domain and then one committed frame: SDF hollowing, a
// chamfered drain hole (adopting any plug it makes as a new item) and Place
// Mesh with a built-in or scene stamp. The Align to normal choice is kept
// per app, shared by the panel and the viewport tool.
import type {
  BooleanMethod,
  DrainHoleOptions,
  DrainHoleReport,
  HollowBackend,
  HollowOptions,
  HollowReport,
  MeshHandle,
  PlaceMeshOptions,
  PlaceMeshReport,
  PlaceUpAxis,
  Scene,
  SceneLike,
  Vec3,
} from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import type { AppState } from "../../../app/state/app-state";
import type { Wave6PrintPrepState } from "../../../app/state/wave6-state";
import { createBuiltinStampMeshGeometry } from "./stamp-geometry";

const ALLOWED_UP_AXES: readonly PlaceUpAxis[] = ["negZ", "posZ", "posY", "negY"];

const alignToNormalByApp = new WeakMap<AppState, boolean>();

export function getPlaceAlignToNormal(app: AppState): boolean {
  return alignToNormalByApp.get(app) ?? true;
}

export function setPlaceAlignToNormal(app: AppState, align: boolean): void {
  alignToNormalByApp.set(app, align);
}

function isAllowedUpAxis(axis: string): axis is PlaceUpAxis {
  return (ALLOWED_UP_AXES as readonly string[]).includes(axis);
}

function triangleCountOf(scene: Pick<SceneLike, "views">, mesh: MeshHandle): number {
  return Math.trunc(scene.views(mesh).indices().length / 3);
}

export async function runHollowMeshOp(args: {
  store: SceneStore;
  mesh: MeshHandle;
  options: HollowOptions;
}): Promise<{ beforeTriangles: number; afterTriangles: number; report: HollowReport }> {
  const { store, mesh, options } = args;
  const booleanDomain = store.scene.boolean;
  if (!booleanDomain?.hollowMesh) {
    throw new Error("hollowMesh is not available on this scene.");
  }
  const beforeTriangles = triangleCountOf(store.scene, mesh);
  const report = await booleanDomain.hollowMesh(mesh, options);
  await store.commitFrame();
  const afterTriangles = triangleCountOf(store.scene, mesh);
  return { beforeTriangles, afterTriangles, report };
}

/** The Hollow Mesh panel's fields. */
export interface HollowFields {
  wallMm: number;
  backend: HollowBackend;
  /** The voxel size as typed; empty (or not a positive number) means automatic. */
  voxelSizeText: string;
  maxVoxels: number;
  /** Cap the inner shell (at most 12k cavity triangles). */
  capInnerShell: boolean;
}

/** The hollow request from the panel's fields: a wall of at least 0.1, the voxel size when one is given. */
export function hollowOptionsFrom(fields: HollowFields): HollowOptions {
  const parsedVoxel = fields.voxelSizeText.trim() === "" ? undefined : Number(fields.voxelSizeText);
  return {
    wallThickness: Math.max(0.1, fields.wallMm),
    backend: fields.backend,
    ...(parsedVoxel !== undefined && Number.isFinite(parsedVoxel) && parsedVoxel > 0
      ? { voxelSize: parsedVoxel }
      : {}),
    maxVoxels: fields.maxVoxels,
    ...(fields.capInnerShell ? { maxCavityTriangles: 12000 } : {}),
  };
}

/** One line for a hollow run: the triangle counts and the cavity it made. */
export function describeHollowResult(res: Awaited<ReturnType<typeof runHollowMeshOp>>): string {
  return `${res.beforeTriangles} -> ${res.afterTriangles} tris · ${res.report.cavitiesCreated} cavity (${res.report.cavityTriangles} cavity tris, ${res.report.cavityVolume.toFixed(1)} mm3)`;
}

export interface ExecuteDrainHoleArgs {
  store: SceneStore;
  app: AppState;
  target: MeshHandle;
  targetName?: string;
  hitPoint: Readonly<Vec3>;
  hitNormal: Readonly<Vec3>;
  drainParams: Wave6PrintPrepState["drainParams"];
  through?: boolean;
  booleanMethod?: BooleanMethod;
  plugClearance?: number;
}

export async function executeDrainHoleAtHit(args: ExecuteDrainHoleArgs): Promise<{
  beforeTriangles: number;
  afterTriangles: number;
  plugs: MeshHandle[];
  report: DrainHoleReport;
}> {
  const { store, app, target, hitPoint, hitNormal, drainParams } = args;
  const booleanDomain = store.scene.boolean;
  if (!booleanDomain?.punchDrainHole) {
    throw new Error("punchDrainHole is not available on this scene.");
  }
  const beforeTriangles = triangleCountOf(store.scene, target);
  const diameter = Math.max(0.1, drainParams.diameter);
  const outerChamfer = Math.max(0, Math.min(drainParams.outerChamfer, diameter * 0.45));
  const depth = args.through ? undefined : Math.max(drainParams.depth, outerChamfer + 0.05);
  const plugClearance = args.plugClearance ?? Math.min(0.15, diameter * 0.2);

  const opts: DrainHoleOptions = {
    surfacePoint: [hitPoint[0], hitPoint[1], hitPoint[2]],
    surfaceNormal: [hitNormal[0], hitNormal[1], hitNormal[2]],
    diameter,
    ...(depth !== undefined ? { depth } : {}),
    outerChamfer,
    taperAngleDeg: drainParams.taperDeg,
    generatePlug: drainParams.generatePlug,
    ...(drainParams.generatePlug ? { plugClearance } : {}),
    ...(args.booleanMethod !== undefined ? { booleanMethod: args.booleanMethod } : {}),
  };

  const report = await booleanDomain.punchDrainHole(target, opts);
  const baseName = args.targetName ?? "mesh";
  for (let i = 0; i < report.plugs.length; i++) {
    const plugName = report.plugs.length > 1 ? `${baseName} plug ${i + 1}` : `${baseName} plug`;
    app.adoptSceneMesh(store, report.plugs[i], plugName);
  }
  await store.commitFrame();
  const afterTriangles = triangleCountOf(store.scene, target);
  return { beforeTriangles, afterTriangles, plugs: report.plugs, report };
}

/** One line for a drain hole: the triangle counts, and the plugs adopted. */
export function describeDrainResult(res: Awaited<ReturnType<typeof executeDrainHoleAtHit>>): string {
  const plugText = res.plugs.length > 0 ? ` · ${res.plugs.length} plug adopted` : "";
  return `${res.beforeTriangles} -> ${res.afterTriangles} tris (${res.report.holesPunched} hole${plugText})`;
}

export interface ExecutePlaceStampArgs {
  store: SceneStore;
  app: AppState;
  target: MeshHandle;
  targetName?: string;
  hitPoint: Readonly<Vec3>;
  hitNormal: Readonly<Vec3>;
  placeParams: Wave6PrintPrepState["placeParams"];
  stampMesh?: MeshHandle | null;
  alignToNormal?: boolean;
  booleanMethod?: BooleanMethod;
}

export async function executePlaceStampAtHit(args: ExecutePlaceStampArgs): Promise<{
  beforeTriangles: number;
  afterTriangles: number;
  added: MeshHandle | null;
  report: PlaceMeshReport;
}> {
  const { store, app, target, hitPoint, hitNormal, placeParams } = args;
  const booleanDomain = store.scene.boolean;
  if (!booleanDomain?.placeMesh) {
    throw new Error("placeMesh is not available on this scene.");
  }
  if (!isAllowedUpAxis(placeParams.upAxis)) {
    throw new Error(`Unsupported placeMesh upAxis: ${String(placeParams.upAxis)}`);
  }

  let resolvedStamp: MeshHandle;
  let tempStamp: MeshHandle | null = null;
  let tempOwnedByStore = false;

  if (args.stampMesh !== undefined && args.stampMesh !== null) {
    resolvedStamp = args.stampMesh;
  } else if (placeParams.stampSource === "sceneMesh") {
    const items = app.getSnapshot().items;
    const stampItem =
      placeParams.stampMeshId === null
        ? null
        : (items.find((item) => item.id === placeParams.stampMeshId && item.mesh !== target) ?? null);
    if (stampItem === null) {
      throw new Error("Select a stamp source mesh other than the target.");
    }
    resolvedStamp = stampItem.mesh;
  } else {
    const geom = createBuiltinStampMeshGeometry(placeParams.stampSource);
    const rawScene = store.scene as Partial<Scene | WorkerScene>;
    if (typeof rawScene.addMesh === "function") {
      tempStamp = await rawScene.addMesh(geom.positions, geom.indices);
    } else {
      tempStamp = await store.addMesh(geom.positions, geom.indices);
      tempOwnedByStore = true;
    }
    resolvedStamp = tempStamp;
  }

  const beforeTriangles = triangleCountOf(store.scene, target);
  const effectiveNormal: Vec3 =
    args.alignToNormal === false ? [0, 0, 1] : [hitNormal[0], hitNormal[1], hitNormal[2]];

  const opts: PlaceMeshOptions = {
    hitPoint: [hitPoint[0], hitPoint[1], hitPoint[2]],
    hitNormal: effectiveNormal,
    upAxis: placeParams.upAxis,
    twistAngleRad: (placeParams.twistDeg * Math.PI) / 180,
    uniformScale: Math.max(0.01, placeParams.uniformScale),
    embedDepth: placeParams.embedDepth,
    mode: placeParams.mode,
    conformToSurface: placeParams.conformToSurface,
    ...(args.booleanMethod !== undefined ? { booleanMethod: args.booleanMethod } : {}),
  };

  try {
    const report = await booleanDomain.placeMesh(target, resolvedStamp, opts);
    if (report.added !== null) {
      app.adoptSceneMesh(store, report.added, `${args.targetName ?? "target"} stamp`);
    }
    await store.commitFrame();
    const afterTriangles = triangleCountOf(store.scene, target);
    return { beforeTriangles, afterTriangles, added: report.added, report };
  } finally {
    if (tempStamp !== null) {
      const rawScene = store.scene as Partial<Scene | WorkerScene>;
      if (!tempOwnedByStore && typeof rawScene.removeMesh === "function") {
        await rawScene.removeMesh(tempStamp);
      } else {
        await store.removeMesh(tempStamp);
      }
    }
  }
}
