# Auto-repair

The standard cleanup pipeline: diagnose with `analyze`, fix with one
`repair` call, verify with `measure`.

## Setup

A unit box with every defect below: an open bottom, a flipped triangle,
a duplicated vertex, a degenerate triangle, and a one-triangle floater.

```ts
import { init, Scene } from "fast-mesh-wasm";

await init();
const scene = new Scene();

const boxPositions = new Float64Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]);
const boxIndices = new Uint32Array([
  0, 3, 2, 0, 2, 1, // bottom (z = 0)
  4, 5, 6, 4, 6, 7, // top (z = 1)
  0, 1, 5, 0, 5, 4, // front (y = 0)
  2, 3, 7, 2, 7, 6, // back (y = 1)
  0, 4, 7, 0, 7, 3, // left (x = 0)
  1, 2, 6, 1, 6, 5, // right (x = 1)
]);

const positions = new Float64Array([...boxPositions, 0, 0, 0, 5, 5, 5, 6, 5, 5, 5, 6, 5]);
const indices = new Uint32Array([
  ...boxIndices.slice(6, 12), // top, flipped below by swapping two corners
  8, 1, 5, 8, 5, 4, // front on the duplicated vertex 8 (a copy of 0)
  ...boxIndices.slice(18), // back, left, right
  0, 4, 4, // degenerate: a repeated vertex
  9, 10, 11, // a one-triangle floater far away
]);
indices[1] = 6; // (4, 5, 6) -> (4, 6, 5): flipped winding
indices[2] = 5;
const mesh = scene.addMesh(positions, indices);
```

## Inspect first

```ts
const report = scene.analyze(mesh, {
  weldTolerance: 0, // exact duplicates; pass a distance to merge near-vertices
  selfIntersection: false, // set true to also check triangle self-intersection
  maxListed: 1000, // most entries kept in each counted list
});

report.loops; // boundary loops/chains: { id, vertices, perimeter, area, closed, clean }
report.coincidentGroups; // groups of 2+ coincident vertices
report.degenerate; // degenerate triangle count (+ degenerateList ids)
report.duplicate; // repeated-vertex-set triangle count
report.nonManifoldEdges; // 3+-triangle edges (+ nonManifoldEdgeList)
report.components; // edge-connected components: { triangles, area, closed }
report.orientationConflicts; // interior edges used in the same direction twice
report.selfIntersects; // boolean | null (null when not checked)

// The demo mesh: the open bottom plus the floater's rim, one duplicate
// pair, one degenerate triangle, two components, flipped winding.
if (report.loops.length !== 2) throw new Error("expected two loops");
if (report.coincidentGroups !== 1) throw new Error("expected one duplicate pair");
if (report.degenerate !== 1) throw new Error("expected one degenerate triangle");
if (report.components.length !== 2) throw new Error("expected two components");
if (report.orientationConflicts < 1) throw new Error("expected flipped winding");
if (report.selfIntersects !== null) throw new Error("expected no self-intersection check");
```

## Repair in one call

`{ kind: "auto" }` runs weld → remove degenerate → orient → remove small
components → fill holes as one transaction: one undo step, and when a
stage fails nothing changes and nothing is recorded. The library owns the
order and the defaults; pass only what you want to change.

```ts
const auto = scene.repair(mesh, {
  kind: "auto",
  weldTolerance: 1e-6, // default 1e-5, in mesh units
  minTriangles: 8, // default 500; the largest component always stays
  maxEdges: 64, // default 64; Infinity fills any size
  fair: true, // default true: fills follow the surrounding curvature
});

// One report per stage, keyed by op name, iterating in the order they ran.
if (auto.weld.mergedVertices !== 1) throw new Error("expected the duplicate merged");
if (auto.removeDegenerate.removedTriangles !== 1) throw new Error("expected the degenerate triangle gone");
if (auto.orient.flippedTriangles < 1) throw new Error("expected the flipped triangle fixed");
if (auto.removeSmallComponents.removedTriangles !== 1) throw new Error("expected the floater gone");
if (auto.fillHoles.filled.length !== 1) throw new Error("expected the bottom filled");

// One undo step takes all of it back.
scene.undo();
if (scene.analyze(mesh).components.length !== 2) throw new Error("expected the damaged mesh back");
```

## Choose your own ops

For another order or other options, pass a list: also one undo step, and
when any op fails nothing changes and nothing is recorded. Reports come
back in order.

```ts
const [weld, deg, small, orient, fill] = scene.repair(mesh, [
  { op: "weld", tolerance: 1e-6 },
  { op: "removeDegenerate" },
  { op: "removeSmallComponents", minTriangles: 8, keepLargest: true },
  { op: "orient" },
  { op: "fillHoles", maxEdges: 64 },
]);
// weld.mergedVertices / removedTriangles, fill.filled / .skipped / .failed, ...
if (weld.mergedVertices !== 1) throw new Error("expected the duplicate merged");
if (deg.removedTriangles !== 1) throw new Error("expected the degenerate triangle gone");
if (small.removedTriangles !== 1) throw new Error("expected the floater gone");
if (orient.flippedTriangles < 1) throw new Error("expected the flipped triangle fixed");
if (fill.filled.length !== 1) throw new Error("expected the bottom filled");
```

Op-by-op guide:

| Op | When to use |
| --- | --- |
| `weld` | Stitch coincident vertices first — later ops assume a clean topology. |
| `removeDegenerate` | Drop zero-area triangles (`areaEpsilon`, `removeUnusedVertices`). |
| `removeSmallComponents` | Delete floaters (`minTriangles` / `minArea`, `keepLargest`). |
| `orient` | Make winding consistent (`flippedTriangles`, `patches`). |
| `fillHoles` | Close boundary loops (`maxEdges`, `maxPerimeter`, `maxArea`, or explicit `loops` from `analyze(mesh).loops` — at least one bound or `loops` is required; `fair`, `allowUnclean`). |
| `bridgeLoops` | Connect two specific loops instead of capping each. |

`fillHoles` skips loops beyond its bounds (or unclean ones without
`allowUnclean`) and reports them in `skipped` with the reason
(`"bounds"` | `"unclean"`); loops that fail report in `failed`
(`"projectionFolds"` | `"noValidTriangulation"` | `"solver"`).

## Verify

```ts
const m = scene.measure(mesh);
// m.closed === true, m.volume !== null → watertight; proceed to fields/export.
if (!m.closed || m.volume === null) throw new Error("expected a watertight mesh");
if (Math.abs(m.volume - 1) > 1e-6 || Math.abs(m.area - 6) > 1e-6) {
  throw new Error("expected the unit box back");
}
scene.free();
```

Loop ids (`report.loops[k].id`) are content fingerprints valid while the
vertex sequence is unchanged — pass them back to `fillHoles({ loops })`
or `join({ loopA, loopB })` to target specific holes. Any topology edit
invalidates them (`STALE_LOOPS`); re-run `analyze` after each repair
round that changes topology.
