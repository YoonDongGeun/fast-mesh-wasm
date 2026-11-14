import type {
  Analysis,
  AnalyzeOptions,
  AutoRepairReport,
  AutoRepairRequest,
  BridgeReport,
  DegenerateReport,
  FillReport,
  JoinOptions,
  JoinReport,
  ManifoldOptions,
  ManifoldReport,
  MeshHandle,
  OrientReport,
  RemoveReport,
  RepairOp,
  RepairReportOf,
  SelfIntersectOptions,
  SelfIntersectReport,
  SliverOptions,
  SliverReport,
  WeldReport,
} from "../scene.js";

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
  bridge(mesh: MeshHandle, options: BridgeLoopsOptions): RepairReturn<Async, BridgeReport>;
  bridgeLoops(mesh: MeshHandle, options: BridgeLoopsOptions): RepairReturn<Async, BridgeReport>;
  join(target: MeshHandle, source: MeshHandle, options: JoinOptions): RepairReturn<Async, JoinReport>;
  makeManifold(mesh: MeshHandle, options?: ManifoldOptions): RepairReturn<Async, ManifoldReport>;
  resolveSelfIntersections(mesh: MeshHandle, options?: SelfIntersectOptions): RepairReturn<Async, SelfIntersectReport>;
  cleanupSlivers(mesh: MeshHandle, options?: SliverOptions): RepairReturn<Async, SliverReport>;
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
    bridge: bridgeLoops,
    bridgeLoops,
    join: (target: MeshHandle, source: MeshHandle, options: JoinOptions) => backend.join(target, source, options),
    makeManifold: (mesh: MeshHandle, options: ManifoldOptions = {}) =>
      backend.repair(mesh, { op: "makeManifold", ...options }),
    resolveSelfIntersections: (mesh: MeshHandle, options: SelfIntersectOptions = {}) =>
      backend.repair(mesh, { op: "resolveSelfIntersections", ...options }),
    cleanupSlivers: (mesh: MeshHandle, options: SliverOptions = {}) =>
      backend.repair(mesh, { op: "cleanupSlivers", ...options }),
  }) as RepairDomain<Async>;
}
