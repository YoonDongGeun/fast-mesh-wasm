---
"fast-mesh-wasm": patch
---

`SceneStore` refreshes its snapshot incrementally: an action or status change re-derives only the meshes and sections it touches, and scene events in one task make one snapshot. With 1000 meshes, `select` drops from about 310 µs to under 1 µs, a placement from about 340 µs to 14 µs, and a Worker status delivery from about 520 µs to 55 µs. Adding 500 meshes onto 1000 takes 30 ms instead of 203 ms. Actions still notify subscribers synchronously; updates driven by the scene (its `onChanges` and `onStatus` events) now notify one microtask later, and `getSnapshot()` is up to date before then.
