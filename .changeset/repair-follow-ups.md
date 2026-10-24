---
"fast-mesh-wasm": minor
---

`fillHoles` gains two options: `maxDihedral` rejects minimum-weight triangulations that fold past it (in radians), and `maxRefineVertices` caps the vertices refinement adds per patch. Each filled loop now reports `refineCapped` when refinement stopped early on its budget or rounds. The Rust option structs (`FillOptions`, `AnalyzeOptions`, `DegenerateOptions`, `ComponentOptions`) gain builders, `analyze` uses an early-exit self-intersection check with identical results, and repair out-of-memory now arrives as one `SceneError` variant.
