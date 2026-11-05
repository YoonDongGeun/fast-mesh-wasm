---
"fast-mesh-wasm": patch
---

Mesh registration. `scene.register(moving, fixed, { maxIterations, tolerance, overlap, maxDistance, maxSamples, apply, maxRmse })` rigidly aligns one mesh to another by trimmed point-to-plane ICP, starting from their current placements, and returns `{ placement, rmse, iterations, converged, status, applied, notApplied, maxRmse, samples, inliers, conditioning, undetermined }`. A good result becomes the moving mesh's placement as one undo step (`apply: false` only returns it): it must converge, with `rmse` within `maxRmse`. Poses the data do not determine, such as a plane's slide or a sphere's or a cylinder's spin, report `status: "underconstrained"`, name the motions in `undetermined` and are never applied. Only rotation and translation are solved: the moving scale is kept and the fixed scale followed. Also on `WorkerScene` (async).
