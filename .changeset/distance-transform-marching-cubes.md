---
"fast-mesh-wasm": minor
---

Add selectable 3D distance-transform and Marching Cubes surface offsetting algorithms (`"fhEdt"` Felzenszwalb-Huttenlocher 3D Euclidean Distance Transform + Manifold Marching Cubes, and `"bvhSdf"` exact BVH Signed/Unsigned Distance Field + Manifold Marching Cubes) alongside `"normal"` displacement in `Scene.repair(mesh, { op: "offset", ... })`, `WorkerScene`, and `SceneStore`.
