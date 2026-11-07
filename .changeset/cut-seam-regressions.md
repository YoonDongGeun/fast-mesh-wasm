---
"fast-mesh-wasm": patch
---

Fix `cutByPlane` on meshes with UV seams: a default capped cut no longer fails with `a cut rim cannot be capped` once a rim passes 200 edges. Long rims, with or without seams, are now ear-clipped in the cut plane, and a rim of tens of thousands of edges caps in seconds. Those caps have at most 20,000 new vertices, so they can be coarser than their walls. A real hole lying exactly in the plane now stays open on meshes with UVs, as it does without them. A sparse `TEXCOORD_0` without a `bufferView` beside a Draco primitive now reports `uvsSkipped`.
