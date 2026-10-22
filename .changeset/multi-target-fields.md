---
"fast-mesh-wasm": minor
---

Distance fields to several meshes: `DistanceFieldDesc.to` takes a list, each vertex holding the smallest signed distance over the targets (bit-identical to the minimum of single-target fields), with `fieldNearest` / `fetchFieldNearest` telling which target each value came from and `fieldTargets` listing them. Moving one mesh recomputes only the fields on it or measured to it, and removing a target removes it from the field. `fieldStatus` sums the sign status up over the targets, and `fieldTargetStatus` gives each one. In the store, `DistanceColoring.distanceTo` takes a list too (a set: order does not matter), and `setContactGroup(id, { meshes, pairs?, coloring })` colours a group of meshes by contact with their partners.
