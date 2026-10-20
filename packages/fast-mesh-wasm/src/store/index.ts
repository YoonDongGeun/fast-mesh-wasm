// `fast-mesh-wasm/store`: an observable store over a `Scene` or `WorkerScene`
// for UI code (React's `useSyncExternalStore`, or plain subscriptions).
// Free of React, three.js and the WebAssembly glue: it only uses the scene
// you give it.
export type { StoreAddMeshOptions } from "./host.js";
export {
  type DisplayState,
  type DistanceColoring,
  EMPTY_SNAPSHOT,
  type MeshState,
  meshState,
  REMOVED_MESH_STATE,
  REMOVED_SECTION_STATE,
  type SceneSnapshot,
  type SectionParts,
  type SectionState,
  sectionState,
  type Status,
} from "./snapshot.js";
export { createSceneStore, SceneStore, type SectionPatch } from "./store.js";
