---
"fast-mesh-wasm": minor
---

Add lattice infill (#265): `Scene.latticeFill(mesh, options)` fills the inside of a closed mesh, or the cavity of a hollowed one, with gyroid, Schwarz P or diamond walls of uniform thickness, as one undo step. Without a shell the mesh becomes an open lattice; with `shellThickness`, or in a cavity, its outer surface is kept as it is and shell and walls are one solid. The result is closed, manifold and free of self-intersections, with `cellSize`, `wallThickness` and `shellThickness` in world units. `Scene.latticePlan(mesh, options)` reports the voxel grid, the triangles, the memory and the share of space the walls fill first, and a fill over `maxVoxels` is refused with those numbers. `Scene.latticePreview(mesh, options)` builds the walls on a small grid without changing the mesh. All three are on `WorkerScene` too, where the page stays live meanwhile.
