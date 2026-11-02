---
"fast-mesh-wasm": patch
---

`fast-mesh-wasm/io` gains Draco encoding: `encodeDraco(positions, indices)` writes standalone `.drc` bytes and `exportGLTF` takes `{ draco: true }` (async) for `KHR_draco_mesh_compression`, both through draco3d's lazily loaded encoder with quantization-bits, compression-level, and face-order-preservation options.
