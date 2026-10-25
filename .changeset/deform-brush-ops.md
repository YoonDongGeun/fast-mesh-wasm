---
"fast-mesh-wasm": minor
---

Brush deformation repair ops. `scene.repair(mesh, { op: "inflate", center, radius, falloff, strength })` displaces vertices along their smooth normals (negative strength deflates), and `{ op: "smooth", center, radius, falloff, strength, cycles }` applies masked Taubin smoothing. Each call is one undo entry; invalid brushes report `INVALID_ARGUMENT`.
