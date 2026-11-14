import type {
  MeshHandle,
  OffsetAlgorithm,
  OffsetReport,
  RemeshReport,
  RepairOp,
  RepairReportOf,
  SimplifyReport,
  SubdivideReport,
  SurfaceRemeshReport,
} from "../scene.js";

/** Return type parameterized over synchronous (`Scene`) vs worker (`WorkerScene`) hosts. */
export type RemeshReturn<Async extends boolean, T> = Async extends true ? Promise<T> : T;

/** Options for {@link RemeshDomain.region} / {@link RemeshDomain.remesh}. */
export type RegionRemeshOptions = Omit<Extract<RepairOp, { op: "remesh" }>, "op">;

/** Options for {@link RemeshDomain.surface}. */
export type SurfaceRemeshOptions = Omit<Extract<RepairOp, { op: "remeshSurface" }>, "op">;

/** Options for {@link RemeshDomain.isotropic}. */
export type IsotropicRemeshOptions = Omit<SurfaceRemeshOptions, "mode">;

/** Options for {@link RemeshDomain.adaptive}. */
export type AdaptiveRemeshOptions = Omit<SurfaceRemeshOptions, "mode">;

/** Options for {@link RemeshDomain.simplify}. */
export type SimplifyOptions = Omit<Extract<RepairOp, { op: "simplify" }>, "op">;

/** Options for {@link RemeshDomain.subdivide}. */
export type SubdivideOptions = Omit<Extract<RepairOp, { op: "subdivide" }>, "op">;

/** Options for {@link RemeshDomain.voxel}. */
export interface VoxelRemeshOptions {
  /** Volumetric reconstruction backend (`"fhEdt"` by default). */
  algorithm?: Exclude<OffsetAlgorithm, "normal">;
  /** Grid voxel size in local mesh units (automatic when omitted). */
  voxelSize?: number;
  /** Maximum 3D grid voxel count (`>= 64`, defaults to `16_777_216`). */
  maxVoxels?: number;
}

/** Low-level host backend used by {@link createRemeshDomain}. */
export interface RemeshBackend<Async extends boolean = false> {
  repair<const O extends RepairOp>(mesh: MeshHandle, op: O): RemeshReturn<Async, RepairReportOf<O>>;
}

/** Remesh, simplification, subdivision, and volumetric reconstruction domain. */
export interface RemeshDomain<Async extends boolean = false> {
  remesh(mesh: MeshHandle, options: RegionRemeshOptions): RemeshReturn<Async, RemeshReport>;
  region(mesh: MeshHandle, options: RegionRemeshOptions): RemeshReturn<Async, RemeshReport>;
  surface(mesh: MeshHandle, options: SurfaceRemeshOptions): RemeshReturn<Async, SurfaceRemeshReport>;
  isotropic(
    mesh: MeshHandle,
    options: IsotropicRemeshOptions,
  ): RemeshReturn<Async, SurfaceRemeshReport>;
  adaptive(mesh: MeshHandle, options: AdaptiveRemeshOptions): RemeshReturn<Async, SurfaceRemeshReport>;
  simplify(mesh: MeshHandle, options: SimplifyOptions): RemeshReturn<Async, SimplifyReport>;
  subdivide(mesh: MeshHandle, options?: SubdivideOptions): RemeshReturn<Async, SubdivideReport>;
  voxel(mesh: MeshHandle, options?: VoxelRemeshOptions): RemeshReturn<Async, OffsetReport>;
}

/** Synchronous remesh domain exposed by `Scene.remesh`. */
export type SyncRemeshDomain = RemeshDomain<false>;

/** Asynchronous remesh domain exposed by `WorkerScene.remesh`. */
export type WorkerRemeshDomain = RemeshDomain<true>;

/** Builds a {@link RemeshDomain} backed by `backend`. */
export function createRemeshDomain<Async extends boolean>(backend: RemeshBackend<Async>): RemeshDomain<Async> {
  const region = (mesh: MeshHandle, options: RegionRemeshOptions) =>
    backend.repair(mesh, { op: "remesh", ...options });

  return {
    remesh: region,
    region,
    surface: (mesh: MeshHandle, options: SurfaceRemeshOptions) =>
      backend.repair(mesh, { op: "remeshSurface", ...options }),
    isotropic: (mesh: MeshHandle, options: IsotropicRemeshOptions) =>
      backend.repair(mesh, { op: "remeshSurface", ...options, mode: "isotropic" }),
    adaptive: (mesh: MeshHandle, options: AdaptiveRemeshOptions) =>
      backend.repair(mesh, { op: "remeshSurface", ...options, mode: "adaptive" }),
    simplify: (mesh: MeshHandle, options: SimplifyOptions) =>
      backend.repair(mesh, { op: "simplify", ...options }),
    subdivide: (mesh: MeshHandle, options: SubdivideOptions = {}) =>
      backend.repair(mesh, { op: "subdivide", ...options }),
    voxel: (mesh: MeshHandle, options: VoxelRemeshOptions = {}) =>
      backend.repair(mesh, {
        op: "offset",
        distance: 0,
        algorithm: options.algorithm ?? "fhEdt",
        voxelSize: options.voxelSize,
        maxVoxels: options.maxVoxels,
      }),
  };
}
