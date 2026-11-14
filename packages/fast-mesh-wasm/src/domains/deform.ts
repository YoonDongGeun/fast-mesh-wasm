import type {
  ClayReport,
  CreaseReport,
  FlattenReport,
  GrabReport,
  InflateReport,
  MeshHandle,
  RelaxReport,
  SculptOp,
  SculptReportOf,
  SmoothReport,
  TwistReport,
  VertexProtection,
} from "../scene.js";
import type { ClayOp, CreaseOp, FlattenOp, GrabOp, InflateOp, RelaxOp, SmoothOp, TwistOp } from "./sculpt.js";

/** Return type parameterized over synchronous (`Scene`) vs worker (`WorkerScene`) hosts. */
export type DeformReturn<Async extends boolean, T> = Async extends true ? Promise<T> : T;

/** Deformation operations supporting per-vertex protection weights and geodesic falloff bands. */
export type ProtectedSculptOp = Extract<SculptOp, { protect?: VertexProtection }>;

/** Low-level host backend used by {@link createDeformDomain}. */
export interface DeformBackend<Async extends boolean = false> {
  sculpt<const O extends SculptOp>(mesh: MeshHandle, op: O): DeformReturn<Async, SculptReportOf<O>>;
}

/** Surface deformation and vertex-protection domain. */
export interface DeformDomain<Async extends boolean = false> {
  inflate(mesh: MeshHandle, op: InflateOp): DeformReturn<Async, InflateReport>;
  smooth(mesh: MeshHandle, op: SmoothOp): DeformReturn<Async, SmoothReport>;
  relax(mesh: MeshHandle, op: RelaxOp): DeformReturn<Async, RelaxReport>;
  relaxed(mesh: MeshHandle, op: RelaxOp): DeformReturn<Async, RelaxReport>;
  grab(mesh: MeshHandle, op: GrabOp): DeformReturn<Async, GrabReport>;
  flatten(mesh: MeshHandle, op: FlattenOp): DeformReturn<Async, FlattenReport>;
  crease(mesh: MeshHandle, op: CreaseOp): DeformReturn<Async, CreaseReport>;
  clay(mesh: MeshHandle, op: ClayOp): DeformReturn<Async, ClayReport>;
  twist(mesh: MeshHandle, op: TwistOp): DeformReturn<Async, TwistReport>;

  protect<T extends ProtectedSculptOp>(op: T, protection: VertexProtection): T & { protect: VertexProtection };
  protect<const O extends ProtectedSculptOp>(
    mesh: MeshHandle,
    op: O,
    protection: VertexProtection,
  ): DeformReturn<Async, SculptReportOf<O>>;
}

/** Synchronous deformation domain exposed by `Scene.deform`. */
export type SyncDeformDomain = DeformDomain<false>;

/** Asynchronous deformation domain exposed by `WorkerScene.deform`. */
export type WorkerDeformDomain = DeformDomain<true>;

/** Builds a {@link DeformDomain} backed by `backend`. */
export function createDeformDomain<Async extends boolean>(backend: DeformBackend<Async>): DeformDomain<Async> {
  const relax = (mesh: MeshHandle, op: RelaxOp) => backend.sculpt(mesh, { op: "relax", ...op });

  const protectFn = (<const O extends ProtectedSculptOp>(
    meshOrOp: MeshHandle | O,
    opOrProtection: O | VertexProtection,
    maybeProtection?: VertexProtection,
  ): O | DeformReturn<Async, SculptReportOf<O>> => {
    if (maybeProtection !== undefined) {
      const mesh = meshOrOp as MeshHandle;
      const op = { ...(opOrProtection as O), protect: maybeProtection };
      return backend.sculpt(mesh, op);
    }
    return { ...(meshOrOp as O), protect: opOrProtection as VertexProtection };
  }) as DeformDomain<Async>["protect"];

  return {
    inflate: (mesh: MeshHandle, op: InflateOp) => backend.sculpt(mesh, { op: "inflate", ...op }),
    smooth: (mesh: MeshHandle, op: SmoothOp) => backend.sculpt(mesh, { op: "smooth", ...op }),
    relax,
    relaxed: relax,
    grab: (mesh: MeshHandle, op: GrabOp) => backend.sculpt(mesh, { op: "grab", ...op }),
    flatten: (mesh: MeshHandle, op: FlattenOp) => backend.sculpt(mesh, { op: "flatten", ...op }),
    crease: (mesh: MeshHandle, op: CreaseOp) => backend.sculpt(mesh, { op: "crease", ...op }),
    clay: (mesh: MeshHandle, op: ClayOp) => backend.sculpt(mesh, { op: "clay", ...op }),
    twist: (mesh: MeshHandle, op: TwistOp) => backend.sculpt(mesh, { op: "twist", ...op }),
    protect: protectFn,
  };
}
