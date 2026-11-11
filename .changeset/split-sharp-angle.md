---
"fast-mesh-wasm": minor
---

`splitStretched` takes `sharpAngle`, the angle in degrees past which two faces count as separated by a crease (45 by default, as before). Set it below a model's gentlest crease to keep the flat faces beside that crease flat when edges split. At the default, a flat face beside a 30 degree crease bulges by 0.20 on a 6.0 edge.
