---
"fast-mesh-wasm": minor
---

Undoable adds in the remaining ops that make a mesh (#271): `wrap`, `extractSelectedFaces`, `splitSelectedFaces`, `reconstructPointCloud`, the plug of `punchDrainHole` and the kept-separate stamp of `placeMesh` record the mesh they add, so undo removes it and redo brings it back with the same handle, on `Scene` and `WorkerScene`. `join` removes its source as part of its undo step: undo restores the target and brings the source back. Before, these meshes stayed after undo, and a joined source was gone for good. The collision rules of a mesh removed undoably now come back with it, once both meshes of the pair are back. A drain hole, split or extraction that fails no longer cancels the caller's open step.
