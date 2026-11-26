---
"fast-mesh-wasm": minor
---

One world shape gives one answer, whatever mix of mesh origin, pose and scale puts it into the scene, near the world origin and far from it (#289). Every world query and op was compared across such mixes; what disagreed is fixed.

Wrong near the world origin too:

- Sections of unevenly scaled meshes, on a plane that is not along the mesh's own axes, were cut along another plane.
- `placeMesh` into a turned target (`"appendShell"`, `"embedUnion"`, `"engraveSubtract"`) twisted the stamp otherwise than on its own, and than the transform it reports.
- Soft collision responses to an unevenly scaled indenter pushed to the wrong points.
- `trimByCurve` on an unevenly scaled mesh ran between its points along other lines than in the world.
- `extrudeBase` took the largest boundary loop of the mesh's own coordinates, which under an uneven scale may be another than the world's.
- `select` with a world brush took an ellipsoid on an unevenly scaled mesh.
- The mesh exporters turned normals the wrong way round under a rotated placement.
- `attachTransformControls` dropped a placement's scale when a drag started; it keeps it now.

Wrong far from the world origin:

- `measure`: the volume lost all its digits (2.7 times too large for a ball of radius 1 at 1e5).
- `closestPoint`, `meshDistance`, `intersects`, `findContacts`, `meshDeviation`, `overhangs`, `triangleQuality`, all fields and rigid collision responses, on unevenly scaled meshes.
- `selectFacesScreen`: its camera was rounded to `f32`. It is taken in `f64` now, so give it as numbers or a `Float64Array`.
- Section contours and fills were rounded to `f32`. They have `points64` and `positions64` next to `points` and `positions`, and `bindSection` draws them around a point near them.

Results near the world origin that were right stay as they were. The options that are in a mesh's own units, before its placement's scale, now say so (`repair`, `hollowMesh`, the freeform deformers, a boolean's tolerance and voxel pitch).
