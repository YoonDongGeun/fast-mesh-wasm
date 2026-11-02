# Selection + mask

Vertex selections over one mesh: paint a set, refine it, then isolate it
(or feed it to an edit). A `VertexSelection` is a snapshot — the mesh,
its ascending vertex ids, and the mesh's content stamp. Any topology
edit (edit, repair, cut, join, boolean, compact, undo/redo,
clearHistory) bumps the stamp and invalidates earlier selections
(`STALE_SELECTION`); placement changes do not. Reselect after each
edit, or derive the next selection from the previous one.

## Setup

A unit box. Local render coords are centered: the vertices sit at
(±0.5, ±0.5, ±0.5), the bounding-box center subtracted.

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
const mesh = scene.addMesh(boxPositions, boxIndices);

const center: [number, number, number] = [0.5, 0.5, 0.5];
const radius = 1.5; // seven of the eight vertices
const polygon = [-1, -1, 1, -1, 1, 1, -1, 1]; // the whole box in xy
const vertexId = 0;
```

## Paint a selection

```ts
// Sphere brush in local render coords (views(mesh).positions()),
// or world space with space: "world".
const brush = scene.select(mesh, {
  kind: "brush",
  center,
  radius,
});

// 2D polygon in mesh-local coords projected onto a plane.
const lasso = scene.select(mesh, {
  kind: "lasso",
  polygon,
  axis: "xy", // | "xz" | "yz"
});

// The connected component (over shared triangle edges) holding a vertex.
const component = scene.select(mesh, { kind: "component", seed: vertexId });

if (brush.vertices.length !== 7) throw new Error("expected seven brushed vertices");
if (lasso.vertices.length !== 8) throw new Error("expected the lasso to catch all eight");
if (component.vertices.length !== 8) throw new Error("expected one component of eight");
```

Only vertices a live triangle uses are selectable; dead or never-used
slots are never selected.

## Refine it

```ts
const grown = scene.selectGrow(brush, 2); // +2 rings of edge neighbours
const shrunk = scene.selectShrink(brush, 1); // peel the boundary 1 ring
const inverted = scene.selectInvert(brush); // every used vertex outside the brush

scene.selectionAlive(brush); // false once a topology edit invalidated it
if (!scene.selectionAlive(brush)) throw new Error("expected a live selection");
if (grown.vertices.length !== 8) throw new Error("expected the growth to cover the box");
if (inverted.vertices.length !== 1) throw new Error("expected the one unbrushed vertex");
```

`grow` / `shrink` / `invert` re-stamp, so their results stay valid
until the next topology edit.

## Mask: keep only the selection

```ts
const tight = scene.select(mesh, { kind: "brush", center, radius: 0.5 });
const report = scene.isolate(mesh, tight);
// report.removedTriangles — triangles fully outside the selection, removed.
// report.selection — the same vertex set with the post-edit stamp.
if (report.removedTriangles !== 7) throw new Error("expected seven triangles outside the corner");
if (report.selection.vertices.length !== 1) throw new Error("expected the corner vertex kept");
if (scene.selectionAlive(tight)) throw new Error("expected the input selection to go stale");
```

## Typical mask workflow

```ts
// The selection isolate returns carries the post-edit stamp, so it
// stays valid where the input went stale.
const regrown = scene.selectGrow(report.selection, 1);
if (regrown.vertices.length !== 6) throw new Error("expected the corner plus its neighbours");
scene.isolate(mesh, regrown); // everything outside is already gone: a no-op
scene.free();
```

Because `isolate` is itself a topology edit, the selection it returns
(`report.selection`) is the one to keep using — the input is stale
after the call (when triangles were removed; an isolate with nothing
outside leaves the input live).
