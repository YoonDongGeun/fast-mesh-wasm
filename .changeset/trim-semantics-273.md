---
"fast-mesh-wasm": minor
---

`trimByCurve` cuts only the component the curve lies on, and its report tells a cut from an imprint (#273). Inside and outside are decided on that component alone (the triangles joined through shared edges; on a mesh with UVs an edge also joins across a seam), so a loop on one part of a mesh no longer deletes or keeps the others, and the side a closed component counts as inside is the one with less world area. A curve with points on several components is now an `INVALID_ARGUMENT` error instead of a partial imprint; where components touch, a point lies on each of them. Curves now cross UV seams: both sides split alike, and no seam opens. `TrimReport` gains `imprintTriangles` and `cutTriangles`, which make up `trianglesRemoved`, and `removedArea` in world units squared: the last two are `0` when the loop separates nothing. A trim of a 1M-triangle mesh is six to ten times faster.
