---
"fast-mesh-wasm": minor
---

Add `contactHeatmap(options)` to `fast-mesh-wasm/store`: a preset distance colouring (everything but `distanceTo`) for two meshes in contact, with penetration from deep red (`depth`) to red, a white contact band across `[-tolerance, +tolerance]`, and the gap from green to blue (`gap`). `exactDepth: true` computes the whole field (`maxDistance: Infinity`) so any penetration depth shows; the default bounds it at `max(depth, gap)`. `gap: 0` colours penetration only (the range is `[-depth, tolerance]`). It throws `INVALID_ARGUMENT` unless `depth > 0`, `gap >= 0` and `0 <= tolerance < min(depth, gap)` (`< depth` when `gap` is 0). A tolerance or colour change reuses the store's field.

Add `SceneStore.fetchColoringValues(mesh)`: a copy of the values of the store-owned field behind a mesh's colouring, or `null` when it has none.
