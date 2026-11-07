---
"fast-mesh-wasm": patch
---

Brush repair ops (`inflate`, `smooth`, `relax`, `grab`) take an optional `protect: { vertices, weights? }`. After the op runs, each listed vertex's move is scaled by its weight: `0` (the default) leaves it byte-identical, `1` keeps the whole move. It stays one undo step, and the op's report describes its move before the scaling. One op alone costs by its brush region, and when its brush holds only vertices at `0` it reports 0 and leaves no step. Rust: `fast_mesh_ops::with_protection` wraps any operation the same way, and `Scene::dab_protected` is the region-sized dab.
