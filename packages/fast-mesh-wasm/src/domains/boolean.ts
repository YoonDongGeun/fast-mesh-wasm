import { FastMeshError } from "../core.js";
import type {
  BooleanOp,
  CutOptions,
  CutReport,
  MeshHandle,
  OffsetReport,
  RepairOp,
  RepairReportOf,
  SectionPlane,
  ShellReport,
} from "../scene.js";

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
  boolean?: (target: MeshHandle, source: MeshHandle, op: BooleanOp) => BooleanReturn<Async, MeshHandle>;
  cutByPlane?: (mesh: MeshHandle, plane: SectionPlane, options?: CutOptions) => BooleanReturn<Async, CutReport>;
}

/**
 * Callable boolean, cutting, offsetting, and shell solidification domain
 * (`scene.boolean(...)` and `scene.boolean.<method>(...)`).
 */
export interface BooleanDomain<Async extends boolean = false> {
  (target: MeshHandle, source: MeshHandle, op: BooleanOp): BooleanReturn<Async, MeshHandle>;

  union(target: MeshHandle, source: MeshHandle): BooleanReturn<Async, MeshHandle>;
  difference(target: MeshHandle, source: MeshHandle): BooleanReturn<Async, MeshHandle>;
  subtract(target: MeshHandle, source: MeshHandle): BooleanReturn<Async, MeshHandle>;
  intersection(target: MeshHandle, source: MeshHandle): BooleanReturn<Async, MeshHandle>;
  intersect(target: MeshHandle, source: MeshHandle): BooleanReturn<Async, MeshHandle>;
  cut(mesh: MeshHandle, plane: SectionPlane, options?: CutOptions): BooleanReturn<Async, CutReport>;
  cutByPlane(mesh: MeshHandle, plane: SectionPlane, options?: CutOptions): BooleanReturn<Async, CutReport>;
  offset(mesh: MeshHandle, options: OffsetOptions): BooleanReturn<Async, OffsetReport>;
  shell(mesh: MeshHandle, thickness: number | ShellOptions): BooleanReturn<Async, ShellReport>;
}

/** Synchronous boolean domain exposed by `Scene.boolean`. */
export type SyncBooleanDomain = BooleanDomain<false>;

/** Asynchronous boolean domain exposed by `WorkerScene.boolean`. */
export type WorkerBooleanDomain = BooleanDomain<true>;

function unsupportedError(isAsync: boolean, name: string): never | Promise<never> {
  const err = new FastMeshError(
    "INVALID_ARGUMENT",
    `${name} requires synchronous Scene (worker boolean/cut RPCs are not enabled)`,
  );
  if (isAsync) {
    return Promise.reject(err);
  }
  throw err;
}

/** Builds a callable {@link BooleanDomain} backed by `backend`. */
export function createBooleanDomain<Async extends boolean>(backend: BooleanBackend<Async>): BooleanDomain<Async> {
  const runBoolean = (target: MeshHandle, source: MeshHandle, op: BooleanOp): BooleanReturn<Async, MeshHandle> => {
    if (backend.boolean === undefined) {
      return unsupportedError(backend.isAsync, `boolean("${op}")`) as unknown as BooleanReturn<Async, MeshHandle>;
    }
    return backend.boolean(target, source, op);
  };

  const cut = (mesh: MeshHandle, plane: SectionPlane, options?: CutOptions): BooleanReturn<Async, CutReport> => {
    if (backend.cutByPlane === undefined) {
      return unsupportedError(backend.isAsync, "boolean.cut") as unknown as BooleanReturn<Async, CutReport>;
    }
    return backend.cutByPlane(mesh, plane, options);
  };

  return Object.assign(runBoolean, {
    union: (target: MeshHandle, source: MeshHandle) => runBoolean(target, source, "union"),
    difference: (target: MeshHandle, source: MeshHandle) => runBoolean(target, source, "subtract"),
    subtract: (target: MeshHandle, source: MeshHandle) => runBoolean(target, source, "subtract"),
    intersection: (target: MeshHandle, source: MeshHandle) => runBoolean(target, source, "intersect"),
    intersect: (target: MeshHandle, source: MeshHandle) => runBoolean(target, source, "intersect"),
    cut,
    cutByPlane: cut,
    offset: (mesh: MeshHandle, options: OffsetOptions) => backend.repair(mesh, { op: "offset", ...options }),
    shell: (mesh: MeshHandle, thickness: number | ShellOptions) =>
      backend.repair(mesh, {
        op: "shell",
        thickness: typeof thickness === "number" ? thickness : thickness.thickness,
      }),
  }) as BooleanDomain<Async>;
}
