---
"fast-mesh-wasm": minor
---

Add the `Scene` API: many meshes in one WebAssembly scene with placements, a BVH per mesh, raycast / closest-point / mesh-distance / contact queries, signed distance fields with colour maps, planar sections, edit op lists, mesh analysis and hole repair, and undo/redo with drags as one step. Render buffers (positions, normals, RGBA8 colours, indices) are read through zero-copy typed-array views, and `commitFrame()` reports the dirty ranges to upload. Errors are `FastMeshError`s with a `code`. The wasm binary grows from 19.5 KB to 733 KB (292 KB gzip).
