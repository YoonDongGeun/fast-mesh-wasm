---
"fast-mesh-wasm": minor
---

The undo history's memory budget is an option, and says when an action is permanent (#281). `new Scene({ historyBudget })`, `createWorkerScene({ historyBudget })` and `createSceneStore(scene, { historyBudget })` take it in bytes; it stays 256 MiB by default (`DEFAULT_HISTORY_BUDGET`). `setHistoryBudget(bytes)` changes it later: a smaller one is met at once, the oldest steps first, and a Worker scene keeps it over a restart. `historyBudget` and `historyBytes` read the limit and what is held. A step larger than the whole budget cannot be undone, such as the removal of a mesh of several million triangles or a join of two: `onPermanent(listener)` is called with it (what was lost, the meshes, those gone for good and the bytes), also for an undone add too large to be redone, and `removalBytes(mesh)` tells beforehand what removing a mesh would take.
