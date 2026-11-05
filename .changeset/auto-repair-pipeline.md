---
"fast-mesh-wasm": patch
---

Add `scene.repair(mesh, { kind: "auto" })`: weld, remove degenerate, orient, remove small components and fill holes as one transaction (one undo step), with a report per stage. On `Scene`, `WorkerScene` and the store.
