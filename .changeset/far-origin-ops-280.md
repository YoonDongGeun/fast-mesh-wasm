---
"fast-mesh-wasm": minor
---

Full precision far from the world origin in the remaining ops (#280). `sliceContours`, `sliceContourStack`, `fitPrimitive` and `shortestSurfacePath`, and on a mesh scaled unevenly `trimByCurve`, `selectFacesInSphere`, `selectByCreaseAngle` and `selectSmallFaceIslands`, worked on `f32` world positions, which are off by 0.004 at 1e5; they now keep the mesh's `f64` origin, so that they do the same there as near the world origin. Results near that stay as they were.

- A contour loop has its points in `f64` too, as `points3d64`; `points3d` stays a `Float32Array`. `sliceContoursExport` writes the former. The point a plane is given by may be anywhere on it, however far from the mesh.
- A `PackedPointCloud` can have an `origin`, the world point its `f32` positions are relative to: `packPointCloud` takes one, and `parsePointCloudText` and `parsePointCloudBytes` read the positions in `f64` and find one with `{ center: true }`. The filters keep it, `unpackPointCloud` returns it, and `reconstructPointCloud`, `convexHullOfPoints` and the `cameraHint` of `pointCloudEstimateNormals` count it in. Clouds without one are `f32` world positions, as before.
- `placeMesh` takes `stampAnchor`, the point of the stamp that goes to the hit point. It defaults to the zero of the stamp's own coordinates, as before; a stamp more than 4096 times its size away from that, which used to be placed that far off the hit point, now stands on it with the middle of its underside.
