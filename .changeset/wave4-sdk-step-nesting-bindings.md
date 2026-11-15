---
"fast-mesh-wasm": minor
---

Add nested step/stroke undo coalescing on `WorkerScene` and `SceneStore` (`beginStep`, `endStep`, `cancelStep`, ordered worker step RPCs, and React `useHistory` controls), plus Wasm/SDK/Worker bindings for manifold/CSG boolean options (`boolean`, `booleanBatch`, `cutByPlane`), printability/quality inspection (`overhangs`, `triangleQuality`, `meshDeviation`), geodesic distances and surface paths (`geodesicDistances`, `shortestSurfacePath`), conformal curve trimming (`trimByCurve`), and multi-stage registration and symmetry detection (`alignByLandmarks`, `registerMultistage`, `detectSymmetryPlane`).
