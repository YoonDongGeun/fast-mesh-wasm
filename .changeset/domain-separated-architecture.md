---
"fast-mesh-wasm": minor
---

Organize public mesh processing APIs across Rust (`fast-mesh-ops`, `fast-mesh-scene`, `fast-mesh-cli`) and TypeScript (`Scene`, `WorkerScene`) into seven cohesive domain namespaces (`repair`, `sculpt`, `remesh`, `boolean`, `deform`, `inspect`, `align`) with full backwards compatibility for existing callable signatures and top-level helpers.
