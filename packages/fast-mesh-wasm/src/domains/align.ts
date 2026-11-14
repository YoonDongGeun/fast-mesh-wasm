import type { MeshHandle, RegisterOptions, RegisterResult } from "../scene.js";

/** Return type parameterized over synchronous (`Scene`) vs worker (`WorkerScene`) hosts. */
export type AlignReturn<Async extends boolean, T> = Async extends true ? Promise<T> : T;

/** Low-level host backend used by {@link createAlignDomain}. */
export interface AlignBackend<Async extends boolean = false> {
  register(moving: MeshHandle, fixed: MeshHandle, options?: RegisterOptions): AlignReturn<Async, RegisterResult>;
}

/** Rigid registration and alignment domain (`scene.align.*`). */
export interface AlignDomain<Async extends boolean = false> {
  register(moving: MeshHandle, fixed: MeshHandle, options?: RegisterOptions): AlignReturn<Async, RegisterResult>;
}

/** Synchronous alignment domain exposed by `Scene.align`. */
export type SyncAlignDomain = AlignDomain<false>;

/** Asynchronous alignment domain exposed by `WorkerScene.align`. */
export type WorkerAlignDomain = AlignDomain<true>;

/** Builds an {@link AlignDomain} backed by `backend`. */
export function createAlignDomain<Async extends boolean>(backend: AlignBackend<Async>): AlignDomain<Async> {
  return {
    register: (moving: MeshHandle, fixed: MeshHandle, options?: RegisterOptions) =>
      backend.register(moving, fixed, options),
  };
}
