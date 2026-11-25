---
"fast-mesh-wasm": patch
---

Accurate heat-method geodesics on any triangles (#279). `inspect.geodesicDistances` with `method: "heatMethod"` was far off where triangles are stretched, obtuse or large: a sphere stretched 20 times gave a tenth of the distances or less, and a cylinder whose caps are fans was off by up to 16 % on average. Where the mesh has such triangles the surface is now cut into better ones first, inside the solver and without moving it. Against exact distances, the mean error is below 3 % and no distance is off by more than 4 % of the largest one on every mesh tested, stretched spheres, capped cylinders, fans and sheared grids among them. Evenly triangulated meshes give the distances they gave and take no longer; on others the first query of a mesh takes longer. On meshes of less than about two thousand vertices the time step is smaller than before (`timeStepScale` still scales it).
