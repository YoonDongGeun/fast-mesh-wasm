import type { Vec3 } from "../api.js";
import { FastMeshError } from "../core.js";
import type {
  CaptureRestReport,
  CaptureStrokeReport,
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
  StretchReport,
  SurfaceFilter,
  TwistReport,
} from "../scene.js";

/** Return type parameterized over synchronous (`Scene`) vs worker (`WorkerScene`) hosts. */
export type SculptReturn<Async extends boolean, T> = Async extends true ? Promise<T> : T;

export type InflateOp = Omit<Extract<SculptOp, { op: "inflate" }>, "op">;
export type SmoothOp = Omit<Extract<SculptOp, { op: "smooth" }>, "op">;
export type RelaxOp = Omit<Extract<SculptOp, { op: "relax" }>, "op">;
export type GrabOp = Omit<Extract<SculptOp, { op: "grab" }>, "op">;
export type FlattenOp = Omit<Extract<SculptOp, { op: "flatten" }>, "op">;
export type CreaseOp = Omit<Extract<SculptOp, { op: "crease" }>, "op">;
export type ClayOp = Omit<Extract<SculptOp, { op: "clay" }>, "op">;
export type TwistOp = Omit<Extract<SculptOp, { op: "twist" }>, "op">;
export type CaptureStrokeOptions = Omit<Extract<SculptOp, { op: "captureStrokeState" }>, "op">;
export type CaptureRestOptions = Omit<Extract<SculptOp, { op: "captureRestSizes" }>, "op">;
export type SplitStretchedOptions = Omit<Extract<SculptOp, { op: "splitStretched" }>, "op">;

/** Options for {@link SculptDomain.beginStroke}. */
export interface BeginStrokeOptions {
  /** Keep an already active stroke snapshot on the mesh (`false` by default). */
  onlyIfMissing?: boolean;
  /** Also capture per-vertex rest edge sizes for dynamic `splitStretched`. */
  captureRestSizes?: boolean;
  /** Wrap stroke dabs inside a single history step when supported (`true` by default). */
  groupStep?: boolean;
}

/** Sculpt operations that support composable per-vertex surface filters. */
export type FilteredSculptOp = Extract<SculptOp, { filter?: SurfaceFilter }>;

/** Low-level host backend used by {@link createSculptDomain}. */
export interface SculptBackend<Async extends boolean = false> {
  sculpt<const O extends SculptOp>(mesh: MeshHandle, op: O): SculptReturn<Async, SculptReportOf<O>>;
  sculpt<const O extends readonly SculptOp[]>(
    mesh: MeshHandle,
    ops: O,
  ): SculptReturn<Async, { -readonly [K in keyof O]: SculptReportOf<O[K]> }>;
  verticesInSphere(mesh: MeshHandle, center: Readonly<Vec3>, radius: number): SculptReturn<Async, Uint32Array>;
  beginStep?: () => void;
  endStep?: () => void;
  cancelStep?: () => void;
}

/**
 * Callable sculpt domain namespace (`scene.sculpt(...)` and `scene.sculpt.<method>(...)`).
 *
 * Preserves legacy `sculpt(mesh, op | ops)` call signatures while exposing
 * stroke lifecycle, surface filter helpers, brush shorthands, and dynamic
 * topology subdivision.
 */
export interface SculptDomain<Async extends boolean = false> {
  <const O extends SculptOp>(mesh: MeshHandle, op: O): SculptReturn<Async, SculptReportOf<O>>;
  <const O extends readonly SculptOp[]>(
    mesh: MeshHandle,
    ops: O,
  ): SculptReturn<Async, { -readonly [K in keyof O]: SculptReportOf<O[K]> }>;

  beginStroke(mesh: MeshHandle, options?: BeginStrokeOptions): SculptReturn<Async, CaptureStrokeReport>;
  dab<const O extends SculptOp>(mesh: MeshHandle, op: O): SculptReturn<Async, SculptReportOf<O>>;
  dab<const O extends readonly SculptOp[]>(
    mesh: MeshHandle,
    ops: O,
  ): SculptReturn<Async, { -readonly [K in keyof O]: SculptReportOf<O[K]> }>;
  endStroke(): void;
  cancelStroke(): void;

  filter<T extends FilteredSculptOp>(op: T, surfaceFilter: SurfaceFilter): T & { filter: SurfaceFilter };
  filter<const O extends FilteredSculptOp>(
    mesh: MeshHandle,
    op: O,
    surfaceFilter: SurfaceFilter,
  ): SculptReturn<Async, SculptReportOf<O>>;

  inflate(mesh: MeshHandle, op: InflateOp): SculptReturn<Async, InflateReport>;
  smooth(mesh: MeshHandle, op: SmoothOp): SculptReturn<Async, SmoothReport>;
  relax(mesh: MeshHandle, op: RelaxOp): SculptReturn<Async, RelaxReport>;
  grab(mesh: MeshHandle, op: GrabOp): SculptReturn<Async, GrabReport>;
  flatten(mesh: MeshHandle, op: FlattenOp): SculptReturn<Async, FlattenReport>;
  crease(mesh: MeshHandle, op: CreaseOp): SculptReturn<Async, CreaseReport>;
  clay(mesh: MeshHandle, op: ClayOp): SculptReturn<Async, ClayReport>;
  twist(mesh: MeshHandle, op: TwistOp): SculptReturn<Async, TwistReport>;
  captureStrokeState(mesh: MeshHandle, options?: CaptureStrokeOptions): SculptReturn<Async, CaptureStrokeReport>;
  captureRestSizes(mesh: MeshHandle, options?: CaptureRestOptions): SculptReturn<Async, CaptureRestReport>;
  splitStretched(mesh: MeshHandle, op: SplitStretchedOptions): SculptReturn<Async, StretchReport>;
  verticesInSphere(mesh: MeshHandle, center: Readonly<Vec3>, radius: number): SculptReturn<Async, Uint32Array>;
}

/** Synchronous sculpt domain exposed by `Scene.sculpt`. */
export type SyncSculptDomain = SculptDomain<false>;

/** Asynchronous sculpt domain exposed by `WorkerScene.sculpt`. */
export type WorkerSculptDomain = SculptDomain<true>;

function isPromiseLike<T>(value: T | Promise<T>): value is Promise<T> {
  return (
    typeof value === "object" &&
    value !== null &&
    "then" in value &&
    typeof (value as { then?: unknown }).then === "function"
  );
}

/** Builds a callable {@link SculptDomain} backed by `backend`. */
export function createSculptDomain<Async extends boolean>(backend: SculptBackend<Async>): SculptDomain<Async> {
  const callable = ((mesh: MeshHandle, op: SculptOp | readonly SculptOp[]) =>
    backend.sculpt(mesh, op as SculptOp)) as SculptBackend<Async>["sculpt"];

  let strokeDepth = 0;
  let activeStroke = false;

  const filterFn = (<const O extends FilteredSculptOp>(
    meshOrOp: MeshHandle | O,
    opOrFilter: O | SurfaceFilter,
    maybeFilter?: SurfaceFilter,
  ): O | SculptReturn<Async, SculptReportOf<O>> => {
    if (maybeFilter !== undefined) {
      const mesh = meshOrOp as MeshHandle;
      const op = { ...(opOrFilter as O), filter: maybeFilter };
      return backend.sculpt(mesh, op);
    }
    return { ...(meshOrOp as O), filter: opOrFilter as SurfaceFilter };
  }) as SculptDomain<Async>["filter"];

  return Object.assign(callable, {
    beginStroke: (mesh: MeshHandle, options: BeginStrokeOptions = {}): SculptReturn<Async, CaptureStrokeReport> => {
      const grouped = (options.groupStep ?? true) && backend.beginStep !== undefined;
      if (grouped) {
        backend.beginStep?.();
        strokeDepth++;
      }
      activeStroke = true;

      const rollback = () => {
        if (grouped && strokeDepth > 0) {
          strokeDepth--;
          try {
            backend.cancelStep?.();
          } catch {
            // Step may already be closed externally.
          }
        }
        activeStroke = strokeDepth > 0;
      };

      try {
        const res = backend.sculpt(mesh, {
          op: "captureStrokeState",
          onlyIfMissing: options.onlyIfMissing,
        });
        if (!options.captureRestSizes) {
          if (isPromiseLike(res)) {
            return res.catch((err) => {
              rollback();
              throw err;
            }) as SculptReturn<Async, CaptureStrokeReport>;
          }
          return res;
        }
        if (isPromiseLike(res)) {
          return res
            .then((report) =>
              (backend.sculpt(mesh, {
                op: "captureRestSizes",
                onlyIfMissing: options.onlyIfMissing,
              }) as Promise<CaptureRestReport>).then(() => report),
            )
            .catch((err) => {
              rollback();
              throw err;
            }) as SculptReturn<Async, CaptureStrokeReport>;
        }
        backend.sculpt(mesh, { op: "captureRestSizes", onlyIfMissing: options.onlyIfMissing });
        return res;
      } catch (err) {
        rollback();
        throw err;
      }
    },
    dab: callable,
    endStroke: (): void => {
      if (strokeDepth > 0 && backend.endStep !== undefined) {
        strokeDepth--;
        backend.endStep();
      }
      activeStroke = strokeDepth > 0;
    },
    cancelStroke: (): void => {
      if (strokeDepth > 0 && backend.cancelStep !== undefined) {
        strokeDepth = 0;
        activeStroke = false;
        backend.cancelStep();
        return;
      }
      activeStroke = false;
      throw new FastMeshError(
        "STEP_NOT_OPEN",
        backend.cancelStep === undefined
          ? "cancelStroke requires synchronous Scene step history"
          : "no active sculpt stroke",
      );
    },
    filter: filterFn,
    inflate: (mesh: MeshHandle, op: InflateOp) => backend.sculpt(mesh, { op: "inflate", ...op }),
    smooth: (mesh: MeshHandle, op: SmoothOp) => backend.sculpt(mesh, { op: "smooth", ...op }),
    relax: (mesh: MeshHandle, op: RelaxOp) => backend.sculpt(mesh, { op: "relax", ...op }),
    grab: (mesh: MeshHandle, op: GrabOp) => backend.sculpt(mesh, { op: "grab", ...op }),
    flatten: (mesh: MeshHandle, op: FlattenOp) => backend.sculpt(mesh, { op: "flatten", ...op }),
    crease: (mesh: MeshHandle, op: CreaseOp) => backend.sculpt(mesh, { op: "crease", ...op }),
    clay: (mesh: MeshHandle, op: ClayOp) => backend.sculpt(mesh, { op: "clay", ...op }),
    twist: (mesh: MeshHandle, op: TwistOp) => backend.sculpt(mesh, { op: "twist", ...op }),
    captureStrokeState: (mesh: MeshHandle, options: CaptureStrokeOptions = {}) =>
      backend.sculpt(mesh, { op: "captureStrokeState", ...options }),
    captureRestSizes: (mesh: MeshHandle, options: CaptureRestOptions = {}) =>
      backend.sculpt(mesh, { op: "captureRestSizes", ...options }),
    splitStretched: (mesh: MeshHandle, op: SplitStretchedOptions) =>
      backend.sculpt(mesh, { op: "splitStretched", ...op }),
    verticesInSphere: (mesh: MeshHandle, center: Readonly<Vec3>, radius: number) =>
      backend.verticesInSphere(mesh, center, radius),
  }) as SculptDomain<Async>;
}
