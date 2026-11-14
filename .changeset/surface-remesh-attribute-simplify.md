---
"fast-mesh-wasm": minor
---

Add Botsch-Kobbelt isotropic and curvature-adaptive surface remeshing (`Scene.remesh.isotropic`, `Scene.remesh.adaptive`, `Scene.remesh.surface`, and `{ op: "remeshSurface", ... }`) plus attribute-aware, error-bounded, and crease-locked QEM decimation (`targetRatio`, `maxDeviation`, `preserveUvs`, `lockCreases`, `creaseAngleDeg`, `preventNormalInversion`, and `maxDeviationAchieved`) across `Scene` and `WorkerScene`.
