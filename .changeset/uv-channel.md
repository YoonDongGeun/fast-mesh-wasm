---
"fast-mesh-wasm": patch
---

Add a UV channel end to end: `addMesh(positions, indices, { uvs })`, zero-copy `views(mesh).uvs()`, a `setUvs` edit op, `TEXCOORD_0` through glTF import and export (external buffers and Draco included), and `bindMesh`'s `uv` attribute and `map` option. Every operation keeps UVs aligned or drops them undoably (`docs/uv-channel.md`); weld keeps seams apart. Meshes without UVs behave as before.
