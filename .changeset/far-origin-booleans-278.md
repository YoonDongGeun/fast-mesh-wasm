---
"fast-mesh-wasm": patch
---

Full precision far from the world origin (#278). `booleanBatch` built its result around the world origin in `f32`, so one at 148,000 units was off by 0.008; its origin is now the centre of the bounds of the meshes used. `placeMesh` (a kept-separate stamp, and any mode on a target scaled unevenly) and `punchDrainHole` had the same loss and now work around the hit point, whose `f64` coordinates they use. As with every mesh, the results' `views(mesh).positions()` are relative to `origin(mesh)`, which is no longer zero: `matrix4(mesh)` takes them to the world.
