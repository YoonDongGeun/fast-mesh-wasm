---
"fast-mesh-wasm": minor
---

Add wraps (#263): `Scene.wrap(meshes, options)` builds one closed, manifold, outward-wound envelope without self-intersections around one mesh or several, however broken, as a new mesh filled as one undo step. Gaps and holes narrower than `gap` are closed, by a morphological closing on a voxel grid (`voxelSize` or `detail`), and `project` moves the result back onto the input where that keeps it clean. `Scene.wrapPlan(meshes, options)` reports the grid's size and memory first, and a wrap over `maxVoxels` is refused with those numbers. Both are on `WorkerScene` too, where the page stays live while a wrap runs.
