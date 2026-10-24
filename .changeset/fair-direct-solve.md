---
"fast-mesh-wasm": patch
---

Hole fairing is much faster: the per-coordinate conjugate-gradient solve is now one sparse direct solve of the normal equations (nested-dissection order, elimination-tree symbolic factorisation, numeric LDLᵀ), factored once per patch for all three coordinates. On the ten-hole 1M-triangle fill the faired run fell from 10.0 s to 1.5 s on a shared i5-13600K (6.8x; unfaired unchanged), and a single 19,049-vertex hole from 14.6 s to 0.41 s. Results are unchanged within the fairing tolerance: every hole still converges with cotangent weights, and the sphere accuracy figures are identical.
