import { FastMeshError } from "../core.js";
import type { Vec3 } from "../api.js";
import type {
  BooleanOp,
  BooleanOptions,
  CsgNode,
  CutOptions,
  CutReport,
  MeshHandle,
  OffsetReport,
  RepairOp,
  RepairReportOf,
  SectionPlane,
  ShellReport,
  SurfacePathPoint,
  TrimCurveOptions,
  TrimReport,
} from "../scene.js";
import type {
  DrainHoleOptions,
  DrainHoleReport,
  HollowOptions,
  HollowReport,
  PlaceMeshOptions,
  PlaceMeshReport,
} from "../wave6.js";

/** Return type parameterized over synchronous (`Scene`) vs worker (`WorkerScene`) hosts. */
export type BooleanReturn<Async extends boolean, T> = Async extends true ? Promise<T> : T;

/** Options for {@link BooleanDomain.offset}. */
export type OffsetOptions = Omit<Extract<RepairOp, { op: "offset" }>, "op">;

/** Options for {@link BooleanDomain.shell}. */
export type ShellOptions = Omit<Extract<RepairOp, { op: "shell" }>, "op">;

/** Low-level host backend used by {@link createBooleanDomain}. */
export interface BooleanBackend<Async extends boolean = false> {
  readonly isAsync: Async;
  repair<const O extends RepairOp>(mesh: MeshHandle, op: O): BooleanReturn<Async, RepairReportOf<O>>;
  boolean?: (
    target: MeshHandle,
    source: MeshHandle,
    op: BooleanOp,
    options?: Omit<BooleanOptions, "op">,
  ) => BooleanReturn<Async, MeshHandle>;
  booleanBatch?: (
    meshesOrTree: readonly MeshHandle[] | CsgNode,
    treeOrOptions?: CsgNode | BooleanOptions,
    maybeOptions?: BooleanOptions,
  ) => BooleanReturn<Async, MeshHandle>;
  cutByPlane?: (mesh: MeshHandle, plane: SectionPlane, options?: CutOptions) => BooleanReturn<Async, CutReport>;
  trimByCurve?: (
    mesh: MeshHandle,
    optionsOrCurve: TrimCurveOptions | readonly (Readonly<Vec3> | SurfacePathPoint)[],
    maybeOptions?: Omit<TrimCurveOptions, "points">,
  ) => BooleanReturn<Async, TrimReport>;
  hollowMesh(mesh: MeshHandle, options: HollowOptions): BooleanReturn<Async, HollowReport>;
  punchDrainHole(mesh: MeshHandle, options: DrainHoleOptions): BooleanReturn<Async, DrainHoleReport>;
  placeMesh(target: MeshHandle, stamp: MeshHandle, options: PlaceMeshOptions): BooleanReturn<Async, PlaceMeshReport>;
}

/**
 * Callable boolean, cutting, offsetting, and shell solidification domain
 * (`scene.boolean(...)` and `scene.boolean.<method>(...)`).
 */
export interface BooleanDomain<Async extends boolean = false> {
  (
    target: MeshHandle,
    source: MeshHandle,
    op: BooleanOp,
    options?: Omit<BooleanOptions, "op">,
  ): BooleanReturn<Async, MeshHandle>;

  union(target: MeshHandle, source: MeshHandle, options?: Omit<BooleanOptions, "op">): BooleanReturn<Async, MeshHandle>;
  difference(
    target: MeshHandle,
    source: MeshHandle,
    options?: Omit<BooleanOptions, "op">,
  ): BooleanReturn<Async, MeshHandle>;
  subtract(
    target: MeshHandle,
    source: MeshHandle,
    options?: Omit<BooleanOptions, "op">,
  ): BooleanReturn<Async, MeshHandle>;
  intersection(
    target: MeshHandle,
    source: MeshHandle,
    options?: Omit<BooleanOptions, "op">,
  ): BooleanReturn<Async, MeshHandle>;
  intersect(
    target: MeshHandle,
    source: MeshHandle,
    options?: Omit<BooleanOptions, "op">,
  ): BooleanReturn<Async, MeshHandle>;
  batch(meshes: readonly MeshHandle[], tree: CsgNode, options?: BooleanOptions): BooleanReturn<Async, MeshHandle>;
  batch(tree: CsgNode, options?: BooleanOptions): BooleanReturn<Async, MeshHandle>;
  booleanBatch(
    meshes: readonly MeshHandle[],
    tree: CsgNode,
    options?: BooleanOptions,
  ): BooleanReturn<Async, MeshHandle>;
  booleanBatch(tree: CsgNode, options?: BooleanOptions): BooleanReturn<Async, MeshHandle>;
  cut(mesh: MeshHandle, plane: SectionPlane, options?: CutOptions): BooleanReturn<Async, CutReport>;
  cutByPlane(mesh: MeshHandle, plane: SectionPlane, options?: CutOptions): BooleanReturn<Async, CutReport>;
  /**
   * Imprints a world-space polyline onto `mesh` and, with `closed: true`,
   * deletes one side of the loop from the component it lies on: see
   * `Scene.trimByCurve`.
   */
  trimByCurve(mesh: MeshHandle, options: TrimCurveOptions): BooleanReturn<Async, TrimReport>;
  trimByCurve(
    mesh: MeshHandle,
    curve: readonly (Readonly<Vec3> | SurfacePathPoint)[],
    options?: Omit<TrimCurveOptions, "points">,
  ): BooleanReturn<Async, TrimReport>;
  offset(mesh: MeshHandle, options: OffsetOptions): BooleanReturn<Async, OffsetReport>;
  shell(mesh: MeshHandle, thickness: number | ShellOptions): BooleanReturn<Async, ShellReport>;
  hollowMesh(mesh: MeshHandle, options: HollowOptions): BooleanReturn<Async, HollowReport>;
  punchDrainHole(mesh: MeshHandle, options: DrainHoleOptions): BooleanReturn<Async, DrainHoleReport>;
  placeMesh(target: MeshHandle, stamp: MeshHandle, options: PlaceMeshOptions): BooleanReturn<Async, PlaceMeshReport>;
}

/** Synchronous boolean domain exposed by `Scene.boolean`. */
export type SyncBooleanDomain = BooleanDomain<false>;

/** Asynchronous boolean domain exposed by `WorkerScene.boolean`. */
export type WorkerBooleanDomain = BooleanDomain<true>;

function unsupportedError(isAsync: boolean, name: string): never | Promise<never> {
  const err = new FastMeshError(
    "INVALID_ARGUMENT",
    `${name} is not supported by this boolean backend`,
  );
  if (isAsync) {
    return Promise.reject(err);
  }
  throw err;
}

/** Builds a callable {@link BooleanDomain} backed by `backend`. */
export function createBooleanDomain<Async extends boolean>(backend: BooleanBackend<Async>): BooleanDomain<Async> {
  const runBoolean = (
    target: MeshHandle,
    source: MeshHandle,
    op: BooleanOp,
    options?: Omit<BooleanOptions, "op">,
  ): BooleanReturn<Async, MeshHandle> => {
    if (backend.boolean === undefined) {
      return unsupportedError(backend.isAsync, `boolean("${op}")`) as unknown as BooleanReturn<Async, MeshHandle>;
    }
    return backend.boolean(target, source, op, options);
  };

  const batch = (
    meshesOrTree: readonly MeshHandle[] | CsgNode,
    treeOrOptions?: CsgNode | BooleanOptions,
    maybeOptions?: BooleanOptions,
  ): BooleanReturn<Async, MeshHandle> => {
    if (backend.booleanBatch === undefined) {
      return unsupportedError(backend.isAsync, "boolean.batch") as unknown as BooleanReturn<Async, MeshHandle>;
    }
    return backend.booleanBatch(meshesOrTree, treeOrOptions, maybeOptions);
  };

  const cut = (mesh: MeshHandle, plane: SectionPlane, options?: CutOptions): BooleanReturn<Async, CutReport> => {
    if (backend.cutByPlane === undefined) {
      return unsupportedError(backend.isAsync, "boolean.cut") as unknown as BooleanReturn<Async, CutReport>;
    }
    return backend.cutByPlane(mesh, plane, options);
  };

  const trimByCurve = (
    mesh: MeshHandle,
    optionsOrCurve: TrimCurveOptions | readonly (Readonly<Vec3> | SurfacePathPoint)[],
    maybeOptions?: Omit<TrimCurveOptions, "points">,
  ): BooleanReturn<Async, TrimReport> => {
    if (backend.trimByCurve === undefined) {
      return unsupportedError(backend.isAsync, "boolean.trimByCurve") as unknown as BooleanReturn<Async, TrimReport>;
    }
    return backend.trimByCurve(mesh, optionsOrCurve, maybeOptions);
  };

  return Object.assign(runBoolean, {
    union: (target: MeshHandle, source: MeshHandle, options?: Omit<BooleanOptions, "op">) =>
      runBoolean(target, source, "union", options),
    difference: (target: MeshHandle, source: MeshHandle, options?: Omit<BooleanOptions, "op">) =>
      runBoolean(target, source, "subtract", options),
    subtract: (target: MeshHandle, source: MeshHandle, options?: Omit<BooleanOptions, "op">) =>
      runBoolean(target, source, "subtract", options),
    intersection: (target: MeshHandle, source: MeshHandle, options?: Omit<BooleanOptions, "op">) =>
      runBoolean(target, source, "intersect", options),
    intersect: (target: MeshHandle, source: MeshHandle, options?: Omit<BooleanOptions, "op">) =>
      runBoolean(target, source, "intersect", options),
    batch,
    booleanBatch: batch,
    cut,
    cutByPlane: cut,
    trimByCurve,
    offset: (mesh: MeshHandle, options: OffsetOptions) => backend.repair(mesh, { op: "offset", ...options }),
    shell: (mesh: MeshHandle, thickness: number | ShellOptions) =>
      backend.repair(mesh, {
        op: "shell",
        thickness: typeof thickness === "number" ? thickness : thickness.thickness,
      }),
    hollowMesh: (mesh: MeshHandle, options: HollowOptions) => backend.hollowMesh(mesh, options),
    punchDrainHole: (mesh: MeshHandle, options: DrainHoleOptions) => backend.punchDrainHole(mesh, options),
    placeMesh: (target: MeshHandle, stamp: MeshHandle, options: PlaceMeshOptions) =>
      backend.placeMesh(target, stamp, options),
  }) as BooleanDomain<Async>;
}
