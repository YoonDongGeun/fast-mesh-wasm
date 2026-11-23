---
"fast-mesh-wasm": minor
---

Add extrusions (#262), on `Scene` and `WorkerScene`, each one undo step. `extrudeFaces(mesh, triangles, { distance, direction })` moves a face selection along each region's average normal, each vertex's average normal or a world direction, and builds side walls along the selection's boundary: holes, several regions and regions that meet at a vertex are handled, and a closed, consistently wound mesh stays so. `extrudeBase(mesh, { height | offset, direction, loops })` runs an open mesh's boundary loops to a flat plane and caps them, turning a surface into a solid that faces outward; outlines nested on the plane become holes. Both report the self-intersections of the result (`selfIntersections`) instead of failing.
