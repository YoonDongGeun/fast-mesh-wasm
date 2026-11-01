# Measure → ruler / query

Scalar measurements (`measure`) and world-space queries (`meshDistance`,
`closestPoint`, `raycast`). All reports are in world coordinates — the
placement (including scale) is applied, so scaled meshes measure scaled.

## Ruler: area and volume of one mesh

```ts
import { Scene } from "fast-mesh-wasm";

const scene = new Scene();
const mesh = scene.addMesh(positions, indices);

const m = scene.measure(mesh);
console.log(m.area); // total triangle area, always defined
if (m.closed) {
  console.log(m.volume); // enclosed volume (number, not null)
} else {
  // m.volume is null; m.volumeReason is "open" | "non-manifold".
  // m.boundaryLoops counts the boundary loops/chains (as analyze reports
  // them), m.nonManifoldEdges counts 3+-triangle edges, and
  // m.orientationConflicts counts interior edges used in the same
  // direction by both triangles.
  console.log(m.volumeReason, m.boundaryLoops, m.nonManifoldEdges);
}
```

Notes:

- The volume is the absolute signed sum, so inconsistent winding only
  shows up in `orientationConflicts` while partly cancelling the number.
  For a strict reading, require `orientationConflicts === 0`.
- Thickness (#116 follow-up: `addThicknessField`) and curvature
  (`addCurvatureField`) build on the same mesh state; measure first to
  confirm the mesh is closed before trusting derived fields.

## Gap between two meshes

```ts
// Closest pair in world coordinates, or null when farther than maxDistance.
const gap = scene.meshDistance(a, b, { maxDistance: 10 });
// gap = { distance, a, b, triangleA, triangleB } — 0 when they touch.
```

## Picking and probing

```ts
// Nearest surface point to a world point (one mesh, or every mesh).
const hit = scene.closestPoint(point, { mesh, maxDistance: 5 });
// hit = { mesh, triangle, point, distance } | null

// Nearest ray hit over every mesh.
const ray = scene.raycast(origin, direction, {
  minDistance: 0,
  maxDistance: Infinity,
  cull: "none", // | "back" | "front" (triangle winding)
});
// ray = { mesh, triangle, point, distance, barycentric } | null
```

`direction` need not be unit length; `distance` is in world units.

## When the mesh is stale

`measure` works on any update mode. `raycast` / `closestPoint` /
`meshDistance` bring `"lazy"` meshes up to date first and throw
`STALE_BVH` on a stale `"manual"` mesh — call `update()` (or the
mode's refresh) before querying those.
