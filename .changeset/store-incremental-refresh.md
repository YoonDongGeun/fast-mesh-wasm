---
"fast-mesh-wasm": patch
---

`SceneStore` refreshes its snapshot incrementally: an action or status change re-derives only the meshes and sections it touches, and scene events in one task make one snapshot. With 1000 meshes an action costs under 25 µs instead of 0.3 to 1.4 ms, and adding 500 meshes onto 1000 takes 31 ms instead of 319 ms.
