# Measure → ruler / query

Scalar measurements (`measure`) and world-space queries (`meshDistance`,
`closestPoint`, `raycast`). All reports are in world coordinates — the
placement (including scale) is applied, so scaled meshes measure scaled.

## Ruler: area and volume of one mesh

```ts
import { init, Scene } from "fast-mesh-wasm";

await init();
const scene = new Scene();

// A unit box: 8 vertices, 12 triangles, outward winding.
const boxPositions = new Float64Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]);
const boxIndices = new Uint32Array([
  0, 3, 2, 0, 2, 1, // bottom (z = 0)
  4, 5, 6, 4, 6, 7, // top (z = 1)
  0, 1, 5, 0, 5, 4, // front (y = 0)
  2, 3, 7, 2, 7, 6, // back (y = 1)
  0, 4, 7, 0, 7, 3, // left (x = 0)
  1, 2, 6, 1, 6, 5, // right (x = 1)
]);
const mesh = scene.addMesh(boxPositions, boxIndices);
// A second box 2 units away, and a point/ray above the first.
const b = scene.addMesh(
  boxPositions.map((v, i) => (i % 3 === 0 ? v + 3 : v)),
  boxIndices,
);
const point: [number, number, number] = [0.5, 0.5, 5];
const origin: [number, number, number] = [0.5, 0.5, 5];
const direction: [number, number, number] = [0, 0, -1];

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
if (!m.closed || m.volume === null) throw new Error("expected a watertight mesh");
if (Math.abs(m.area - 6) > 1e-6 || Math.abs(m.volume - 1) > 1e-6) {
  throw new Error("expected the unit box back");
}
```

Notes:

- The volume is the absolute signed sum, so inconsistent winding only
  shows up in `orientationConflicts` while partly cancelling the number.
  For a strict reading, require `orientationConflicts === 0`.
- Thickness (`addThicknessField`) and curvature (`addCurvatureField`)
  build on the same mesh state; measure first to confirm the mesh is
  closed before trusting derived fields.

## Gap between two meshes

```ts
// Closest pair in world coordinates, or null when farther than maxDistance.
const gap = scene.meshDistance(mesh, b, { maxDistance: 10 });
// gap = { distance, a, b, triangleA, triangleB } — 0 when they touch.
if (gap === null || Math.abs(gap.distance - 2) > 1e-6) throw new Error("expected a gap of 2");
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
if (hit === null || Math.abs(hit.distance - 4) > 1e-6) throw new Error("expected the box top 4 below");
if (ray === null || Math.abs(ray.distance - 4) > 1e-6) throw new Error("expected the ray to hit the box top");
scene.free();
```

`direction` need not be unit length; `distance` is in world units.

## When the mesh is stale

`measure` works on any update mode. `raycast` / `closestPoint` /
`meshDistance` bring `"lazy"` meshes up to date first and throw
`STALE_BVH` on a stale `"manual"` mesh — call `update()` (or the
mode's refresh) before querying those.
