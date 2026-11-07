---
"fast-mesh-wasm": patch
---

A single `inflate`, `relax` or `grab` repair now costs by the vertices under the brush instead of the whole mesh, with the same result, except that a vertex no triangle uses is no longer in reach of it. Their options are unchanged: `strength` is still in mesh units (the showcase's radius-relative strength, default 0.1, is the showcase's own and does not reach this API). In a list they still scan the mesh. `bindMesh` grows a large mesh's bounds over a small edit instead of walking the mesh on every frame, and makes them exact again when the edits pause, so `geometry.boundingBox` can be larger than the mesh for 250 ms after such an edit.
