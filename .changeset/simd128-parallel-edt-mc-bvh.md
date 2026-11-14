---
"fast-mesh-wasm": patch
---

Enable Wasm `simd128` (`+simd128`, `--enable-simd`), add reusable thread-local scratch pools (`ScratchPool`, `ScratchU32`, `ScratchF32`, etc.) in `fast-mesh-store`, implement 4-wide SIMD128 BVH/query ray and closest-point kernels in `fast-mesh-bvh` and `fast-mesh-query`, and optimize Felzenszwalb-Huttenlocher EDT and Marching Cubes hot paths.
