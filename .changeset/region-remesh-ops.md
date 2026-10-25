---
"fast-mesh-wasm": minor
---

Region remeshing repair op. `scene.repair(mesh, { op: "remesh", center, radius, targetLength, maxRounds })` retriangulates the edges under a spherical region toward a target edge length (unset means the mesh-wide mean) with midpoint splits, midpoint collapses, and angle-improving flips. Boundary edges never collapse; each call is one undo entry; invalid regions report `INVALID_ARGUMENT`.
