---
"fast-mesh-wasm": minor
---

Faster, accurate and cancellable heat-method geodesics (#274). `inspect.geodesicDistances` with `method: "heatMethod"` takes about a second or two on a million triangles instead of ten to twenty-five, and its distances are now right on large meshes: the diffusion step used to stop too early, so past a few hundred thousand triangles far distances came out up to half too short. The operators are kept for later queries of the same mesh, until it changes or another mesh is queried, or `inspect.releaseGeodesics()`. `geodesicDistances` and `shortestSurfacePath` take a `signal` (an `AbortSignal`): on a `WorkerScene` aborting it cancels a running query at its next step.
