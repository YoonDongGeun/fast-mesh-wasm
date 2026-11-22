---
"fast-mesh-wasm": minor
---

Add exact 3D convex hulls (#261): `Scene.convexHull(meshes)` for one mesh or several, and `Scene.convexHullOfPoints(points)` for a raw point cloud (flat positions or a `PackedPointCloud`), on `Scene` and `WorkerScene`. The hull is a quickhull on exact orientation predicates, so it is always convex, closed and outward-wound with every input point inside or on it; duplicate, coplanar and collinear points are fine, and a point set without volume is an `INVALID_ARGUMENT` error that says whether it is coincident, collinear or coplanar. The hull is added as a new mesh, filled as one undo step.
