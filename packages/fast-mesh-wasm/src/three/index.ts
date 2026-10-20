// `fast-mesh-wasm/three`: the three.js adapter over the renderer-neutral Scene.
// `three` is an optional peer dependency; only this entry imports it.
// TypeScript users also need `@types/three` installed (three ships no types);
// it is not declared as a peer; the README says so.
export {
  type AttachTransformControlsOptions,
  attachTransformControls,
  type TransformControlsLike,
} from "./controls.js";
export { type BindMeshOptions, type BoundMesh, bindMesh, type MeshBinding } from "./mesh.js";
export { type BindSectionOptions, type BoundSection, bindSection } from "./section.js";
export type { SceneLike } from "../scene-like.js";
