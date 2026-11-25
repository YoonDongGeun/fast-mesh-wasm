---
"fast-mesh-wasm": minor
---

Add primitive mesh generators (#259): `generatePrimitive(shape)` builds a box (per-axis size and segments), cylinder, cone or frustum, UV sphere, icosphere, torus, arrow, n-gon prism or plane/grid from its parameters, every option with a default, and returns `{ positions, indices }`. Every shape is centred on the origin, welded, closed (except the plane) and wound outward; curved shapes converge to their analytic volume as the resolution grows. It is a standalone function, not a scene operation (nothing is stored, no history), usable on the main thread after `init()` or in any Worker; `WorkerScene.generatePrimitive(shape)` runs it in the scene's Worker and transfers the buffers back. The Rust generators are `fast_mesh_ops::primitives`.
