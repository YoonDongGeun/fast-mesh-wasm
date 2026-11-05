---
"fast-mesh-wasm": patch
---

`scene.register` no longer reports `converged: true` or applies a pose the data do not determine. A sphere's or a cylinder's spin and a plane's slide, even when facets or noise let the iteration settle with a tiny `rmse`, now report `status: "underconstrained"` and name the motions in the new `undetermined` field. Applying now also requires `rmse` within the new `maxRmse` option (default 1% of the moving mesh's extent, so a wrong local minimum does not move the mesh) and at least half of the expected points in range; the new `notApplied` field says why a result was not applied. `rmse` is accurate on long thin triangles, and bad options no longer end a running drag. This check adds a bounded cost to each call: about 2.4 s on one thread or 0.3 s with threads at 1M triangles.
