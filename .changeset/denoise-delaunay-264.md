---
"fast-mesh-wasm": minor
---

Add feature-preserving denoising and Delaunay edge flips (#264), on `Scene` and `WorkerScene`, each one undo step on a whole mesh or on a face region (`region`, triangle ids). `denoise(mesh, { iterations, strength, featureAngle })` smooths scan noise by bilateral normal filtering and a vertex update: flat faces become flat again and edges sharper than the feature angle stay sharp. `delaunayFlips(mesh, { dihedralLimit })` flips edges towards a Delaunay triangulation without moving a vertex; no new triangle turns further than the limit from the triangles it replaces, edges sharper than it never flip, and the smallest angle never drops. Both reports carry the surface's readouts before and after (smallest corner angle, mean normal deviation). The Rust ops are `fast_mesh_ops::denoise` and `fast_mesh_ops::delaunay_flips`, parallel on the threaded build with results bit-identical to the single-threaded one.
