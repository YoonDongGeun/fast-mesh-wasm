---
"fast-mesh-wasm": patch
---

`bindMesh` no longer throws from its timer when a large mesh, or its scene, goes within 250 ms of a small edit and before the next `commitFrame`. Its documentation now states when the bounds can lag.
