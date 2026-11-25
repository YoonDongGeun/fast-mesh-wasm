import type { Indices, Vec3 } from "../api.js";
import type {
  Analysis,
  AnalyzeOptions,
  AutoRepairReport,
  AutoRepairRequest,
  BridgeReport,
  BridgeStats,
  DegenerateReport,
  FillContinuityStats,
  FillHolesContinuityOptions,
  FillReport,
  JoinOptions,
  JoinReport,
  ManifoldOptions,
  ManifoldReport,
  MeshHandle,
  OrientReport,
  PartialBridgeOptions,
  RemoveReport,
  RepairOp,
  RepairReportOf,
  SelfIntersectOptions,
  SelfIntersectReport,
  SliverOptions,
  SliverReport,
  WeldReport,
} from "../scene.js";
import type {
  AdjacencyKind,
  FaceEditReport,
  FaceSelectionNormalCone,
  ScreenSelectionShape,
  ScreenSelectOptions,
  SplitFacesReport,
} from "../wave6.js";

/** Return type parameterized over synchronous (`Scene`) vs worker (`WorkerScene`) hosts. */
export type RepairReturn<Async extends boolean, T> = Async extends true ? Promise<T> : T;

/** Options for {@link RepairDomain.autoRepair} (`AutoRepairRequest` without `kind`). */
export type AutoRepairOptions = Omit<AutoRepairRequest, "kind">;

/** Options for {@link RepairDomain.removeDegenerate}. */
export type DegenerateOptions = Omit<Extract<RepairOp, { op: "removeDegenerate" }>, "op">;

/** Options for {@link RepairDomain.removeSmallComponents}. */
export type SmallComponentOptions = Omit<Extract<RepairOp, { op: "removeSmallComponents" }>, "op">;

/** Options for {@link RepairDomain.fillHoles}. */
export type FillHolesOptions = Omit<Extract<RepairOp, { op: "fillHoles" }>, "op">;

/** Options for {@link RepairDomain.bridge}. */
export type BridgeLoopsOptions = Omit<Extract<RepairOp, { op: "bridgeLoops" }>, "op">;

/** Options for {@link RepairDomain.clean}. */
export interface CleanOptions {
  areaEpsilon?: number;
  removeUnusedVertices?: boolean;
  minTriangles?: number;
  minArea?: number;
  keepLargest?: boolean;
  keepCavities?: boolean;
}

/** Combined report returned by {@link RepairDomain.clean}. */
export interface CleanReport {
  degenerate: DegenerateReport;
  components: RemoveReport;
}

/** Low-level host backend used by {@link createRepairDomain}. */
export interface RepairBackend<Async extends boolean = false> {
  repair(mesh: MeshHandle, request: AutoRepairRequest): RepairReturn<Async, AutoRepairReport>;
  repair<const O extends RepairOp>(mesh: MeshHandle, op: O): RepairReturn<Async, RepairReportOf<O>>;
  repair<const O extends readonly RepairOp[]>(
    mesh: MeshHandle,
    ops: O,
  ): RepairReturn<Async, { -readonly [K in keyof O]: RepairReportOf<O[K]> }>;
  analyze(mesh: MeshHandle, options?: AnalyzeOptions): RepairReturn<Async, Analysis>;
  join(target: MeshHandle, source: MeshHandle, options: JoinOptions): RepairReturn<Async, JoinReport>;
  fillHoleContinuity(
    mesh: MeshHandle,
    loopIndex: number,
    options?: FillHolesContinuityOptions,
  ): RepairReturn<Async, FillContinuityStats>;
  fillAllHolesContinuity(
    mesh: MeshHandle,
    options?: FillHolesContinuityOptions,
  ): RepairReturn<Async, FillContinuityStats>;
  bridgePartialArcs(
    mesh: MeshHandle,
    loopA: Indices,
    loopB: Indices,
    options?: PartialBridgeOptions,
  ): RepairReturn<Async, BridgeStats>;
  selectFacesScreen(
    mesh: MeshHandle,
    shape: ScreenSelectionShape,
    options: ScreenSelectOptions,
  ): RepairReturn<Async, Uint32Array>;
  selectFacesInSphere(
    mesh: MeshHandle,
    center: Readonly<Vec3>,
    radius: number,
    normalCone?: FaceSelectionNormalCone,
  ): RepairReturn<Async, Uint32Array>;
  selectByCreaseAngle(mesh: MeshHandle, seedTri: number, maxDihedralRad: number): RepairReturn<Async, Uint32Array>;
  selectConnectedShell(mesh: MeshHandle, seedTri: number): RepairReturn<Async, Uint32Array>;
  selectSmallFaceIslands(
    mesh: MeshHandle,
    maxFaceCount: number,
    maxArea?: number,
  ): RepairReturn<Async, Uint32Array>;
  growFaceSelection(
    mesh: MeshHandle,
    selectedTris: Indices,
    rings?: number,
    adjacency?: AdjacencyKind,
  ): RepairReturn<Async, Uint32Array>;
  shrinkFaceSelection(
    mesh: MeshHandle,
    selectedTris: Indices,
    rings?: number,
    adjacency?: AdjacencyKind,
  ): RepairReturn<Async, Uint32Array>;
  fillSelectionHoles(
    mesh: MeshHandle,
    selectedTris: Indices,
    maxHoleTris: number,
  ): RepairReturn<Async, Uint32Array>;
  smoothSelectionBoundary(
    mesh: MeshHandle,
    selectedTris: Indices,
    iterations?: number,
  ): RepairReturn<Async, Uint32Array>;
  faceSelectionToVertexMask(mesh: MeshHandle, selectedTris: Indices): RepairReturn<Async, Float32Array>;
  vertexMaskToFaceSelection(
    mesh: MeshHandle,
    mask: Float32Array | ArrayLike<number>,
    threshold?: number,
    requireAllVertices?: boolean,
  ): RepairReturn<Async, Uint32Array>;
  deleteSelectedFaces(mesh: MeshHandle, selectedTris: Indices): RepairReturn<Async, FaceEditReport>;
  extractSelectedFaces(
    mesh: MeshHandle,
    selectedTris: Indices,
    options?: { normals?: boolean },
  ): RepairReturn<Async, MeshHandle>;
  splitSelectedFaces(
    mesh: MeshHandle,
    selectedTris: Indices,
    options?: { normals?: boolean },
  ): RepairReturn<Async, SplitFacesReport>;
}

/**
 * Callable repair domain namespace (`scene.repair(...)` and `scene.repair.<method>(...)`).
 *
 * Preserves legacy `repair(mesh, op | ops | request)` call signatures while
 * organizing repair operations cleanly under `scene.repair.*`.
 */
export interface RepairDomain<Async extends boolean = false> {
  (mesh: MeshHandle, request: AutoRepairRequest): RepairReturn<Async, AutoRepairReport>;
  <const O extends RepairOp>(mesh: MeshHandle, op: O): RepairReturn<Async, RepairReportOf<O>>;
  <const O extends readonly RepairOp[]>(
    mesh: MeshHandle,
    ops: O,
  ): RepairReturn<Async, { -readonly [K in keyof O]: RepairReportOf<O[K]> }>;

  analyze(mesh: MeshHandle, options?: AnalyzeOptions): RepairReturn<Async, Analysis>;
  autoRepair(mesh: MeshHandle, options?: AutoRepairOptions): RepairReturn<Async, AutoRepairReport>;
  clean(mesh: MeshHandle, options?: CleanOptions): RepairReturn<Async, CleanReport>;
  weld(mesh: MeshHandle, tolerance: number): RepairReturn<Async, WeldReport>;
  removeDegenerate(mesh: MeshHandle, options?: DegenerateOptions): RepairReturn<Async, DegenerateReport>;
  removeSmallComponents(mesh: MeshHandle, options?: SmallComponentOptions): RepairReturn<Async, RemoveReport>;
  orient(mesh: MeshHandle): RepairReturn<Async, OrientReport>;
  fillHoles(mesh: MeshHandle, options?: FillHolesOptions): RepairReturn<Async, FillReport>;
  fillHoleContinuity(
    mesh: MeshHandle,
    loopIndex: number,
    options?: FillHolesContinuityOptions,
  ): RepairReturn<Async, FillContinuityStats>;
  fillAllHolesContinuity(
    mesh: MeshHandle,
    options?: FillHolesContinuityOptions,
  ): RepairReturn<Async, FillContinuityStats>;
  bridge(mesh: MeshHandle, options: BridgeLoopsOptions): RepairReturn<Async, BridgeReport>;
  bridgeLoops(mesh: MeshHandle, options: BridgeLoopsOptions): RepairReturn<Async, BridgeReport>;
  bridgePartialArcs(
    mesh: MeshHandle,
    loopA: Indices,
    loopB: Indices,
    options?: PartialBridgeOptions,
  ): RepairReturn<Async, BridgeStats>;
  join(target: MeshHandle, source: MeshHandle, options: JoinOptions): RepairReturn<Async, JoinReport>;
  makeManifold(mesh: MeshHandle, options?: ManifoldOptions): RepairReturn<Async, ManifoldReport>;
  resolveSelfIntersections(mesh: MeshHandle, options?: SelfIntersectOptions): RepairReturn<Async, SelfIntersectReport>;
  cleanupSlivers(mesh: MeshHandle, options?: SliverOptions): RepairReturn<Async, SliverReport>;
  selectFacesScreen(
    mesh: MeshHandle,
    shape: ScreenSelectionShape,
    options: ScreenSelectOptions,
  ): RepairReturn<Async, Uint32Array>;
  selectFacesInSphere(
    mesh: MeshHandle,
    center: Readonly<Vec3>,
    radius: number,
    normalCone?: FaceSelectionNormalCone,
  ): RepairReturn<Async, Uint32Array>;
  selectByCreaseAngle(mesh: MeshHandle, seedTri: number, maxDihedralRad: number): RepairReturn<Async, Uint32Array>;
  selectConnectedShell(mesh: MeshHandle, seedTri: number): RepairReturn<Async, Uint32Array>;
  selectSmallFaceIslands(
    mesh: MeshHandle,
    maxFaceCount: number,
    maxArea?: number,
  ): RepairReturn<Async, Uint32Array>;
  growFaceSelection(
    mesh: MeshHandle,
    selectedTris: Indices,
    rings?: number,
    adjacency?: AdjacencyKind,
  ): RepairReturn<Async, Uint32Array>;
  shrinkFaceSelection(
    mesh: MeshHandle,
    selectedTris: Indices,
    rings?: number,
    adjacency?: AdjacencyKind,
  ): RepairReturn<Async, Uint32Array>;
  fillSelectionHoles(
    mesh: MeshHandle,
    selectedTris: Indices,
    maxHoleTris: number,
  ): RepairReturn<Async, Uint32Array>;
  smoothSelectionBoundary(
    mesh: MeshHandle,
    selectedTris: Indices,
    iterations?: number,
  ): RepairReturn<Async, Uint32Array>;
  faceSelectionToVertexMask(mesh: MeshHandle, selectedTris: Indices): RepairReturn<Async, Float32Array>;
  vertexMaskToFaceSelection(
    mesh: MeshHandle,
    mask: Float32Array | ArrayLike<number>,
    threshold?: number,
    requireAllVertices?: boolean,
  ): RepairReturn<Async, Uint32Array>;
  deleteSelectedFaces(mesh: MeshHandle, selectedTris: Indices): RepairReturn<Async, FaceEditReport>;
  extractSelectedFaces(
    mesh: MeshHandle,
    selectedTris: Indices,
    options?: { normals?: boolean },
  ): RepairReturn<Async, MeshHandle>;
  splitSelectedFaces(
    mesh: MeshHandle,
    selectedTris: Indices,
    options?: { normals?: boolean },
  ): RepairReturn<Async, SplitFacesReport>;
}

/** Synchronous repair domain exposed by `Scene.repair`. */
export type SyncRepairDomain = RepairDomain<false>;

/** Asynchronous repair domain exposed by `WorkerScene.repair`. */
export type WorkerRepairDomain = RepairDomain<true>;

function isPromiseLike<T>(value: T | Promise<T>): value is Promise<T> {
  return (
    typeof value === "object" &&
    value !== null &&
    "then" in value &&
    typeof (value as { then?: unknown }).then === "function"
  );
}

/** Builds a callable {@link RepairDomain} backed by `backend`. */
export function createRepairDomain<Async extends boolean>(backend: RepairBackend<Async>): RepairDomain<Async> {
  const callable = ((mesh: MeshHandle, op: RepairOp | readonly RepairOp[] | AutoRepairRequest) =>
    backend.repair(mesh, op as RepairOp)) as RepairBackend<Async>["repair"];

  const bridgeLoops = (mesh: MeshHandle, options: BridgeLoopsOptions): RepairReturn<Async, BridgeReport> =>
    backend.repair(mesh, { op: "bridgeLoops", ...options });

  return Object.assign(callable, {
    analyze: (mesh: MeshHandle, options?: AnalyzeOptions) => backend.analyze(mesh, options),
    autoRepair: (mesh: MeshHandle, options: AutoRepairOptions = {}) =>
      backend.repair(mesh, { kind: "auto", ...options }),
    clean: (mesh: MeshHandle, options: CleanOptions = {}): RepairReturn<Async, CleanReport> => {
      const degOp = {
        op: "removeDegenerate",
        areaEpsilon: options.areaEpsilon,
        removeUnusedVertices: options.removeUnusedVertices,
      } as const;
      if (options.minTriangles === undefined && options.minArea === undefined) {
        const res = backend.repair(mesh, degOp);
        const mapDegenerate = (degenerate: DegenerateReport): CleanReport => ({
          degenerate,
          components: {
            removedComponents: 0,
            removedTriangles: 0,
            removedVertices: 0,
          },
        });
        return (isPromiseLike(res) ? res.then(mapDegenerate) : mapDegenerate(res)) as RepairReturn<Async, CleanReport>;
      }
      const res = backend.repair(mesh, [
        degOp,
        {
          op: "removeSmallComponents",
          minTriangles: options.minTriangles,
          minArea: options.minArea,
          keepLargest: options.keepLargest,
          keepCavities: options.keepCavities,
        },
      ] as const);
      const mapReport = ([degenerate, components]: [DegenerateReport, RemoveReport]): CleanReport => ({
        degenerate,
        components,
      });
      return (isPromiseLike(res) ? res.then(mapReport) : mapReport(res)) as RepairReturn<Async, CleanReport>;
    },
    weld: (mesh: MeshHandle, tolerance: number) => backend.repair(mesh, { op: "weld", tolerance }),
    removeDegenerate: (mesh: MeshHandle, options: DegenerateOptions = {}) =>
      backend.repair(mesh, { op: "removeDegenerate", ...options }),
    removeSmallComponents: (mesh: MeshHandle, options: SmallComponentOptions = {}) =>
      backend.repair(mesh, { op: "removeSmallComponents", ...options }),
    orient: (mesh: MeshHandle) => backend.repair(mesh, { op: "orient" }),
    fillHoles: (mesh: MeshHandle, options: FillHolesOptions = {}) =>
      backend.repair(mesh, { op: "fillHoles", ...options }),
    fillHoleContinuity: (mesh: MeshHandle, loopIndex: number, options?: FillHolesContinuityOptions) =>
      backend.fillHoleContinuity(mesh, loopIndex, options),
    fillAllHolesContinuity: (mesh: MeshHandle, options?: FillHolesContinuityOptions) =>
      backend.fillAllHolesContinuity(mesh, options),
    bridge: bridgeLoops,
    bridgeLoops,
    bridgePartialArcs: (mesh: MeshHandle, loopA: Indices, loopB: Indices, options?: PartialBridgeOptions) =>
      backend.bridgePartialArcs(mesh, loopA, loopB, options),
    join: (target: MeshHandle, source: MeshHandle, options: JoinOptions) => backend.join(target, source, options),
    makeManifold: (mesh: MeshHandle, options: ManifoldOptions = {}) =>
      backend.repair(mesh, { op: "makeManifold", ...options }),
    resolveSelfIntersections: (mesh: MeshHandle, options: SelfIntersectOptions = {}) =>
      backend.repair(mesh, { op: "resolveSelfIntersections", ...options }),
    cleanupSlivers: (mesh: MeshHandle, options: SliverOptions = {}) =>
      backend.repair(mesh, { op: "cleanupSlivers", ...options }),
    selectFacesScreen: (mesh: MeshHandle, shape: ScreenSelectionShape, options: ScreenSelectOptions) =>
      backend.selectFacesScreen(mesh, shape, options),
    selectFacesInSphere: (
      mesh: MeshHandle,
      center: Readonly<Vec3>,
      radius: number,
      normalCone?: FaceSelectionNormalCone,
    ) => backend.selectFacesInSphere(mesh, center, radius, normalCone),
    selectByCreaseAngle: (mesh: MeshHandle, seedTri: number, maxDihedralRad: number) =>
      backend.selectByCreaseAngle(mesh, seedTri, maxDihedralRad),
    selectConnectedShell: (mesh: MeshHandle, seedTri: number) => backend.selectConnectedShell(mesh, seedTri),
    selectSmallFaceIslands: (mesh: MeshHandle, maxFaceCount: number, maxArea?: number) =>
      backend.selectSmallFaceIslands(mesh, maxFaceCount, maxArea),
    growFaceSelection: (mesh: MeshHandle, selectedTris: Indices, rings?: number, adjacency?: AdjacencyKind) =>
      backend.growFaceSelection(mesh, selectedTris, rings, adjacency),
    shrinkFaceSelection: (mesh: MeshHandle, selectedTris: Indices, rings?: number, adjacency?: AdjacencyKind) =>
      backend.shrinkFaceSelection(mesh, selectedTris, rings, adjacency),
    fillSelectionHoles: (mesh: MeshHandle, selectedTris: Indices, maxHoleTris: number) =>
      backend.fillSelectionHoles(mesh, selectedTris, maxHoleTris),
    smoothSelectionBoundary: (mesh: MeshHandle, selectedTris: Indices, iterations?: number) =>
      backend.smoothSelectionBoundary(mesh, selectedTris, iterations),
    faceSelectionToVertexMask: (mesh: MeshHandle, selectedTris: Indices) =>
      backend.faceSelectionToVertexMask(mesh, selectedTris),
    vertexMaskToFaceSelection: (
      mesh: MeshHandle,
      mask: Float32Array | ArrayLike<number>,
      threshold?: number,
      requireAllVertices?: boolean,
    ) => backend.vertexMaskToFaceSelection(mesh, mask, threshold, requireAllVertices),
    deleteSelectedFaces: (mesh: MeshHandle, selectedTris: Indices) =>
      backend.deleteSelectedFaces(mesh, selectedTris),
    extractSelectedFaces: (mesh: MeshHandle, selectedTris: Indices, options?: { normals?: boolean }) =>
      backend.extractSelectedFaces(mesh, selectedTris, options),
    splitSelectedFaces: (mesh: MeshHandle, selectedTris: Indices, options?: { normals?: boolean }) =>
      backend.splitSelectedFaces(mesh, selectedTris, options),
  }) as RepairDomain<Async>;
}
