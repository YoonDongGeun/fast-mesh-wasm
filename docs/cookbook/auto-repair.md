# Auto-repair

The standard cleanup pipeline: diagnose with `analyze`, fix with one
`repair` transaction, verify with `measure`.

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
```

## Repair in one transaction

Pass a list: one undo step, and when any op fails nothing changes and
nothing is recorded. Reports come back in order.

```ts
const [weld, deg, small, orient, fill] = scene.repair(mesh, [
  { op: "weld", tolerance: 1e-6 },
  { op: "removeDegenerate" },
  { op: "removeSmallComponents", minTriangles: 8, keepLargest: true },
  { op: "orient" },
  { op: "fillHoles" },
]);
// weld.mergedVertices / removedTriangles, fill.filled / .skipped / .failed, ...
```

Op-by-op guide:

| Op | When to use |
| --- | --- |
| `weld` | Stitch coincident vertices first — later ops assume a clean topology. |
| `removeDegenerate` | Drop zero-area triangles (`areaEpsilon`, `removeUnusedVertices`). |
| `removeSmallComponents` | Delete floaters (`minTriangles` / `minArea`, `keepLargest`). |
| `orient` | Make winding consistent (`flippedTriangles`, `patches`). |
| `fillHoles` | Close boundary loops (`maxEdges`, `maxPerimeter`, `maxArea`, or explicit `loops` from `analyze(mesh).loops`; `fair`, `allowUnclean`). |
| `bridgeLoops` | Connect two specific loops instead of capping each. |

`fillHoles` skips loops beyond its bounds (or unclean ones without
`allowUnclean`) and reports them in `skipped` with the reason
(`"bounds"` | `"unclean"`); loops that fail report in `failed`
(`"projectionFolds"` | `"noValidTriangulation"` | `"solver"`).

## Verify

```ts
const m = scene.measure(mesh);
// m.closed === true, m.volume !== null → watertight; proceed to fields/export.
```

Loop ids (`report.loops[k].id`) are content fingerprints valid while the
vertex sequence is unchanged — pass them back to `fillHoles({ loops })`
or `join({ loopA, loopB })` to target specific holes. Any topology edit
invalidates them (`STALE_LOOPS`); re-run `analyze` after each repair
round that changes topology.
