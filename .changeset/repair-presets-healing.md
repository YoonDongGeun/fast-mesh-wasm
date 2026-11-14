---
"fast-mesh-wasm": minor
---

Add non-manifold topology healing (`makeManifold`), self-intersection resolution (`resolveSelfIntersections` with `none`, `deleteAndFill`, `cutAndStitch`, and `voxelHeal` modes), sliver cleanup (`cleanupSlivers`), and domain `autoRepair` presets (`general`, `scan3d`, `print3d`, `cadImport`) across Rust ops, Wasm bindings, TypeScript SDK/Worker, and Showcase UI.
