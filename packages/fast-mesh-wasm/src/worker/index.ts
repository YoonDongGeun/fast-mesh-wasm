// `fast-mesh-wasm/worker` (browsers): a scene whose engine runs in a Worker,
// with main-thread mirrors for rendering. The Worker script is the separate
// `worker-entry.js`; this entry never loads the WebAssembly module itself.
export {
  createWorkerScene,
  type KeyStatus,
  type WorkerLike,
  WorkerScene,
  type WorkerSceneOptions,
} from "./client.js";
