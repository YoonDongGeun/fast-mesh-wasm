---
"fast-mesh-wasm": minor
---

Add `fast-mesh-wasm/three`, a three.js adapter for the `Scene` API. `bindMesh(scene, mesh, options?)` returns a `THREE.Mesh` whose geometry follows the scene through `commitFrame()` (dirty ranges only, colour attribute while colouring is on, `setOpacity`, `dispose`); `attachTransformControls(controls, scene, bound)` turns a TransformControls drag into one scene drag and one undo step; `bindSection(scene, section, bound, options?)` draws a section's contours and fill and clips the bound meshes at its plane (set `renderer.localClippingEnabled = true`). `three` (`>=0.160`) is an optional peer dependency that only this entry imports, and TypeScript users also need `@types/three`. A caller's material is used as given and must not be shared between bound meshes.

Core additions: `DragSession.active` tells whether the drag is still open, `Scene.sectionPlane(section)` returns a section's current plane, and `Scene.removeMesh` now ends an active drag on that mesh (as `undo()` does) instead of leaving it open.

There is also a raw WebGPU example in `examples/webgpu` that uses no 3D library.
