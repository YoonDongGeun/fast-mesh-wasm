---
"fast-mesh-wasm": minor
---

Undoable adds and removals of meshes (#269). `addMesh(positions, indices, { undoable: true })` and `removeMesh(mesh, { undoable: true })` on `Scene`, `WorkerScene` and the store are undo steps, or part of the open one: undo removes an added mesh and brings a removed one back with the same handle, data, placement, normals, colouring and the fields on it, without copying anything whatever its size (removing and restoring a 1M-triangle mesh takes under a microsecond in the engine). `scene.has(mesh)`, `isRestorable(mesh)` and `MeshChange.restorable` tell where a mesh is; a kept mesh counts against the history's memory budget. The store's snapshot follows undo and redo. Without `undoable` nothing changes. `convexHull`, `convexHullOfPoints`, `cutByPlane` and `booleanBatch` now record the mesh they add, so undo removes it instead of emptying it or leaving it behind.
