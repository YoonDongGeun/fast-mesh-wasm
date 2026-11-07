---
"fast-mesh-wasm": minor
---

Add the `captureRestSizes` and `splitStretched` repair ops: triangles a deformation stretches split on their own once they pass a limit, and the rest of the mesh does not change. Existing vertices keep their ids and positions, nothing collapses or flips, and each area is measured against its own size at capture, so a mesh with fine and coarse areas keeps both. New vertices follow the surface's curvature. The cost follows the stretched edges, so large meshes are fine. Rest sizes are dropped when an edit outdates them, so ask again with `onlyIfMissing`. `scene.origin(mesh)` is new: `views().positions()` are relative to it, and `setPositions` takes positions with it added.
