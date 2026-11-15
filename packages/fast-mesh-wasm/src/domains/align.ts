import type {
  LandmarkAlignment,
  LandmarkPair,
  MeshHandle,
  RegisterOptions,
  RegisterResult,
  RegistrationOptions,
  RegistrationReport,
  SymmetryPlaneReport,
} from "../scene.js";

/** Return type parameterized over synchronous (`Scene`) vs worker (`WorkerScene`) hosts. */
export type AlignReturn<Async extends boolean, T> = Async extends true ? Promise<T> : T;

/** Low-level host backend used by {@link createAlignDomain}. */
export interface AlignBackend<Async extends boolean = false> {
  register(moving: MeshHandle, fixed: MeshHandle, options?: RegisterOptions): AlignReturn<Async, RegisterResult>;
  alignByLandmarks(
    moving: MeshHandle,
    fixed: MeshHandle,
    pairs: readonly LandmarkPair[],
    allowScaling?: boolean,
  ): AlignReturn<Async, LandmarkAlignment>;
  registerMultistage(
    moving: MeshHandle,
    fixed: MeshHandle,
    options?: RegistrationOptions,
  ): AlignReturn<Async, RegistrationReport>;
  detectSymmetryPlane(mesh: MeshHandle): AlignReturn<Async, SymmetryPlaneReport>;
}

/** Rigid registration, multi-stage alignment, landmark SVD, and symmetry-plane domain (`scene.align.*`). */
export interface AlignDomain<Async extends boolean = false> {
  register(moving: MeshHandle, fixed: MeshHandle, options?: RegisterOptions): AlignReturn<Async, RegisterResult>;
  alignByLandmarks(
    moving: MeshHandle,
    fixed: MeshHandle,
    pairs: readonly LandmarkPair[],
    allowScaling?: boolean,
  ): AlignReturn<Async, LandmarkAlignment>;
  registerMultistage(
    moving: MeshHandle,
    fixed: MeshHandle,
    options?: RegistrationOptions,
  ): AlignReturn<Async, RegistrationReport>;
  detectSymmetryPlane(mesh: MeshHandle): AlignReturn<Async, SymmetryPlaneReport>;
  detectSymmetry(mesh: MeshHandle): AlignReturn<Async, SymmetryPlaneReport>;
}

/** Synchronous alignment domain exposed by `Scene.align`. */
export type SyncAlignDomain = AlignDomain<false>;

/** Asynchronous alignment domain exposed by `WorkerScene.align`. */
export type WorkerAlignDomain = AlignDomain<true>;

/** Builds an {@link AlignDomain} backed by `backend`. */
export function createAlignDomain<Async extends boolean>(backend: AlignBackend<Async>): AlignDomain<Async> {
  const detectSymmetryPlane = (mesh: MeshHandle) => backend.detectSymmetryPlane(mesh);
  return {
    register: (moving: MeshHandle, fixed: MeshHandle, options?: RegisterOptions) =>
      backend.register(moving, fixed, options),
    alignByLandmarks: (
      moving: MeshHandle,
      fixed: MeshHandle,
      pairs: readonly LandmarkPair[],
      allowScaling?: boolean,
    ) => backend.alignByLandmarks(moving, fixed, pairs, allowScaling),
    registerMultistage: (moving: MeshHandle, fixed: MeshHandle, options?: RegistrationOptions) =>
      backend.registerMultistage(moving, fixed, options),
    detectSymmetryPlane,
    detectSymmetry: detectSymmetryPlane,
  };
}
