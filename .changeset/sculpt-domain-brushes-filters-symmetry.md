---
"fast-mesh-wasm": minor
---

Expose the sculpt brush domain across `Scene`, `WorkerScene`, and `SceneStore`:

- Add `sculpt(mesh, op)` and `SculptOp` / `SculptReport` / `SculptReportOf` (100% backwards-compatible with `repair(mesh, op)` and `RepairOp`).
- Add `InflateMethod` (`"vertexNormal" | "averageNormal" | "strokeNormal"`), `SmoothMethod` (`"taubin" | "laplacian" | "hcLaplacian" | "cotangent"`), and `GrabMethod` (`"sphere" | "geodesic" | "kelvinlet"` with `poisson`, `fromStroke`, and fold-safe `adaptiveReach`), plus `captureStrokeState`.
- Add `flatten` (`"both" | "peaks" | "valleys"`), `crease` (`pinch`), `clay` (`"round" | "strips"`), and `twist` (`"vortex" | "kelvinlet"`).
- Add composable `SurfaceFilter` (`"none" | "peaksOnly" | "valleysOnly" | "normalAngle"`), local-axis `Symmetry` (`{ x, y, z }`), geodesic soft-band `VertexProtection.band`, and `verticesInSphere(mesh, center, radius)`.
