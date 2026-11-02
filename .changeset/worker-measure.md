---
"fast-mesh-wasm": patch
---

`WorkerScene` gains `measure(mesh)`, the worker-side twin of `Scene.measure`: scalar area and watertight-volume measurements of a mesh, resolving with the same `MeasureResult` (volume `null` with `volumeReason` on open or non-manifold meshes).
