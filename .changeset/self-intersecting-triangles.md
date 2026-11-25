---
"fast-mesh-wasm": minor
---

Add `inspect.selfIntersectingTriangles(mesh)` on `Scene` and `WorkerScene`: the live triangle slots of a mesh that intersect or touch another of its own triangles, ascending, found with the mesh's BVH (exact, and parallel on the threaded build). `analyze({ selfIntersection: true })` only says whether there is any; this lists them, so they can be selected, shown or repaired.
