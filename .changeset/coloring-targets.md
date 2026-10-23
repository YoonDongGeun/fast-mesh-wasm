---
"fast-mesh-wasm": minor
---

Add `SceneStore.coloringTargets(mesh)`: the targets of the store-owned field behind a mesh's colouring, in the field's order (ascending scene id), which is the order its nearest-target indices (`fieldNearest`, `fetchFieldNearest`) refer to; frozen, or `null` without a colouring. Use it to name the partner behind a contact group member's vertex: the order differs from `distanceTo` and from the group's member list.

The rule `coloring(mesh)` returns on `Scene` and `WorkerScene` is now frozen (deeply, the field handle aside), so callers cannot change the scene's copy. A fragment-shaded binding turns its material's `vertexColors` off while patched and restores it on dispose. `<FastMesh edgeSoftness>` draws NaN, an infinite or a negative width as 0 with a console warning instead of throwing inside the canvas.
