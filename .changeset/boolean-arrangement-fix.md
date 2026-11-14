---
"fast-mesh-wasm": patch
---

Replace the boolean mesh splitter (`boolean_plan` / `apply_boolean`, `Scene.boolean`, `WorkerScene.boolean`, and CLI `boolean`) with a canonical simplex-keyed arrangement using exact `orient3d`/`orient2d` predicates, constrained 2D sub-triangulation, seam-bounded patch classification via generalized winding numbers, and strict 2-manifold edge/vertex fan de-pinching (#165). Generic curved overlaps (spheres, tori, shells), varied face triangulations, and coplanar, edge-touching, or vertex-touching solids now produce watertight, outward-wound 2-manifold outputs free of boundary loops, orientation conflicts, and pinch vertices.
