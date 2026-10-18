// `fast-mesh-wasm/three`: the three.js adapter over the renderer-neutral Scene.
// `three` is an optional peer dependency; only this entry imports it.
// TypeScript users also need `@types/three` installed (three ships no types);
// it is not declared as a peer; the README says so.
export { type BindMeshOptions, type BoundMesh, bindMesh } from "./mesh.js";
