---
"fast-mesh-wasm": minor
---

JS core follow-ups ([#40](https://github.com/YoonDongGeun/fast-mesh-wasm/issues/40)). `Scene.mode(mesh)` reports when a mesh's derived data follows its edits, and `Scene.bounds(mesh)` the box around the vertices its live triangles use (dead slots keep stale positions, so they are excluded; `null` without live triangles). `Scene.drag(mesh, { coalesce: true })` remembers pointer-rate moves and applies the latest one once at the next `commitFrame()` (or `end()`), so eager fields recompute once per frame.

A Lazy or Manual mesh added with `addMesh` now builds its BVH on first use instead of up front (1M triangles: ~1.3 s to ~20 ms; the first query, update or flush builds it). Dense frames report `"all"` past half the slots dirty instead of sorting the dirty sets, in the render records and the normals. Results are unchanged.
