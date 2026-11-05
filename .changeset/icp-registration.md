---
"fast-mesh-wasm": patch
---

Mesh registration. `scene.register(moving, fixed, { maxIterations, tolerance, overlap, maxDistance, maxSamples, apply })` rigidly aligns one mesh to another by trimmed point-to-plane ICP, starting from their current placements, and returns `{ placement, rmse, iterations, converged, status, applied, samples, inliers, conditioning }`. A converged result becomes the moving mesh's placement as one undo step (`apply: false` only returns it); a result that did not converge is never applied. Surfaces that do not determine the alignment, such as a plane sliding along itself, report `status: "underconstrained"` and `converged: false`. Only rotation and translation are solved: the moving scale is kept and the fixed scale followed. Also on `WorkerScene` (async).
