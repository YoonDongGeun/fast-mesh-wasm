# Selection + mask

Vertex selections over one mesh: paint a set, refine it, then isolate it
(or feed it to an edit). A `VertexSelection` is a snapshot — the mesh,
its ascending vertex ids, and the mesh's content stamp. Any topology
edit (edit, repair, cut, join, boolean, compact, undo/redo) bumps the
stamp and invalidates earlier selections (`STALE_SELECTION`); placement
changes do not. Reselect after each edit, or derive the next selection
from the previous one.

## Paint a selection

```ts
// Sphere brush in local render coords (views(mesh).positions()),
// or world space with space: "world".
const brush = scene.select(mesh, {
  kind: "brush",
  center: [0, 0, 0],
  radius: 1.5,
});

// 2D polygon in mesh-local coords projected onto a plane.
const lasso = scene.select(mesh, {
  kind: "lasso",
  polygon: [x0, y0, x1, y1, x2, y2],
  axis: "xy", // | "xz" | "yz"
});

// The connected component (over shared triangle edges) holding a vertex.
const component = scene.select(mesh, { kind: "component", seed: vertexId });
```

Only vertices a live triangle uses are selectable; dead or never-used
slots are never selected.

## Refine it

```ts
const grown = scene.selectGrow(sel, 2); // +2 rings of edge neighbours
const shrunk = scene.selectShrink(sel, 1); // peel the boundary 1 ring
const inverted = scene.selectInvert(sel); // every used vertex not in sel

scene.selectionAlive(sel); // false once a topology edit invalidated it
```

`grow` / `shrink` / `invert` re-stamp, so their results stay valid
until the next topology edit.

## Mask: keep only the selection

```ts
const report = scene.isolate(mesh, sel);
// report.removedTriangles — triangles fully outside the selection, removed.
// report.selection — the same vertex set with the post-edit stamp.
```

## Typical mask workflow

```ts
const sel = scene.select(mesh, { kind: "brush", center, radius });
const grown = scene.selectGrow(sel, 1);
scene.isolate(mesh, grown); // mesh now holds just the masked region
```

Because `isolate` is itself a topology edit, the selection it returns
(`report.selection`) is the one to keep using — the input is stale
after the call.
