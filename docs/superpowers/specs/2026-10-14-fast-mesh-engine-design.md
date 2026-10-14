# fast-mesh engine design

- **Status:** Draft for review
- **Date:** 2026-10-14
- **Scope:** Turns `fast-mesh-wasm` from two utility functions into an editable mesh engine: storage with undo/redo, BVH, spatial queries and collisions, derived scalar fields and colors, cross-sections, repair operations, and JS integrations (renderer-agnostic core, Worker, three.js, React).

## 1. Goals

1. Build and maintain a BVH for large meshes simply and memory-efficiently.
2. Let multiple meshes interact: raycasts, closest points, distances, intersections, scene-wide contact detection.
3. Support mesh edits tens of times per second, including topology changes, with transaction-level undo/redo.
4. Propagate every edit to exactly what depends on it (BVH, fields, colors, sections) and nothing else.
5. Keep each subsystem independently understandable, testable and replaceable by contributors.
6. Stay renderer-agnostic and framework-agnostic; integrations are optional entry points.

### Scale targets

- 10M+ triangles per mesh (≈5M vertices), several such meshes per scene.
- wasm32 (4 GB address space) in the first releases. Memory is a first-class budget, not an afterthought.
- Interactive edits and queries on the main thread; long operations (full builds, heavy repairs) in a Worker.

### Non-goals (for the first phases)

- Physics response (impulses, solving). We export contact data; physics engines consume it.
- Rendering. The library outputs buffers and dirty ranges; renderers draw.
- Exact penetration depth for arbitrary non-convex mesh pairs.
- Self-intersection resolution.

## 2. Decisions summary

| Topic | Decision |
| --- | --- |
| Audience | General-purpose open source library, renderer- and framework-agnostic |
| Architecture | Layered Rust crates, composed by a `Scene` that acts as the mediator |
| Edits | Vertex moves and topology changes |
| Undo granularity | Transaction: any number of algorithm steps commit as one `Patch` |
| Undo mechanism | Journal (copy-on-first-write per slot), automatic snapshot fallback for large changes |
| Precision | `f32` storage + per-mesh `f64` origin; `f32`/`f64` input accepted |
| Indices | `u32` internally; `u16`/`u32` input accepted |
| Memory model | wasm32; code uses `usize` consistently so a wasm64 build can be added later |
| Execution | Synchronous core + optional Worker wrapper; multithreaded build later |
| Fields | Source fields, built-in fields, user-defined fields; incremental recompute |
| Signed distance | Modes `pseudonormal`, `winding`, `cap`, default `auto` |
| Queries | Raycast, closest point, mesh distance, intersection, region, scene contacts |
| Sections | Plane cross-sections with contours, fill and measurements, per-mesh selection |
| Repair (phase 1) | `analyze`, `weld`, `removeDegenerate`, `removeSmallComponents`, `fillHoles` (min-area) |
| Async sync model | Declarative state, ordered queue, coalescing, cooperative cancellation, revisions |
| Integrations | `fast-mesh-wasm/worker`, `fast-mesh-wasm/three`, `fast-mesh-wasm/react` |

## 3. Architecture

### 3.1 Crates

```
crates/
  fast-mesh-store    Slot storage, attribute channels, adjacency, transactions, Patch, History
  fast-mesh-bvh      BVH build, refit, partial rebuild, quality tracking, winding data
  fast-mesh-query    Raycast, closest point, distance, intersection, region, plane section
  fast-mesh-fields   Field kernels, built-in fields, colormaps, field dependency DAG
  fast-mesh-ops      Mesh algorithms (repair, hole filling; later remeshing, smoothing)
  fast-mesh-scene    Scene: meshes, transforms, broadphase, sections, update propagation
  fast-mesh-wasm     wasm-bindgen bindings for the Scene API and standalone low-level APIs
  fast-mesh          Existing crate; becomes a thin facade re-exporting the crates above
```

Dependencies (each crate depends only on the ones listed, all lower in the stack):

| Crate | Depends on |
| --- | --- |
| `fast-mesh-store` | — |
| `fast-mesh-bvh` | store |
| `fast-mesh-query` | store, bvh |
| `fast-mesh-fields` | store, bvh, query |
| `fast-mesh-ops` | store, bvh, query |
| `fast-mesh-scene` | all of the above |
| `fast-mesh-wasm` | scene (and lower crates for standalone APIs) |

A crate never depends on a crate above it. Each crate has its own unit and property tests and can be used from plain Rust without WebAssembly.

### 3.2 One wasm module

All crates compile into a single `.wasm`. Subsystems share mesh memory, so splitting them into separate modules would force copies. Wasm size is measured in CI on every PR.

### 3.3 Scene as mediator

Users declare relationships ("lower's `gap` field is the distance to upper"). The Scene keeps a dependency graph and, on every commit or flush, updates only what the change affects:

```
upper positions ──▶ upper BVH ──▶ lower.gap ──▶ lower colors
lower positions ──▶ lower BVH ─┘
section.plane   ──▶ section contours (for selected meshes)
```

Users never wire subsystems together themselves. They control timing through update policies (section 7.4) and receive results through events (section 7.5).

## 4. Storage (`fast-mesh-store`)

### 4.1 Layout

| Data | Format | 10M triangles |
| --- | --- | --- |
| Positions | `[f32; 3]` contiguous + mesh origin `[f64; 3]` | ~60 MB |
| Triangles | `[u32; 3]` contiguous | ~120 MB |
| Liveness | bitsets for vertices and triangles | ~2 MB |
| Adjacency (lazy) | corner table: opposite corner per triangle corner + one corner per vertex | ~140 MB |
| Attribute channel | per vertex, e.g. `f32` or `[u8; 4]` | ~20 MB each |

- Positions are stored relative to the mesh origin. `f64` input is converted by subtracting the origin.
- **Adjacency is built lazily**, the first time an algorithm or field needs neighbors, and cached until the next topology change. (Phase 1 rebuilds it after topology changes; incremental maintenance is added with the topology operations that need it inside a transaction.) A user who only moves vertices never pays for it.
- **Any triangle set is accepted**, including non-manifold input. Adjacency marks non-manifold edges and vertices; topology algorithms treat marked elements as boundaries or refuse to touch them.

### 4.2 Slots

- Vertices and triangles live in slots. Deleting frees the slot and clears its liveness bit.
- A freed slot is not reused while any `Patch` in history references it. It returns to the free list when those patches are evicted.
- `compact()` removes free slots and renumbers everything. Because ids change, it clears history. It is explicit, never automatic.

### 4.3 Attribute channels

- **Source channels** hold user-provided values (e.g. a painted mask). They are journaled and undoable.
- **Derived channels** hold computed values (normals, distances, colors). They are not journaled; after undo/redo they are recomputed for the affected vertices. This keeps history small.
- Each source channel has a **topology value policy** for newly created vertices: `interpolate` (weighted by creation parents), `default`, or `nearest` (the creation parent with the largest weight; a spatial nearest-vertex lookup arrives with the Scene).

### 4.4 Transactions and journal

```rust
let mut tx = mesh.begin_edit();
// algorithms write only through tx: move_vertices, add_vertices(with optional parents+weights),
// add_triangles, remove_triangles, write_channel ...
let patch = tx.commit()?;   // or tx.rollback()
```

- On the first write to a slot within a transaction, the old value is copied into the journal; a per-transaction bitset ensures each slot is copied once.
- `commit()` produces a `Patch`: for each channel the touched slot ids, old values and new values; created and removed slots; and a `ChangeSet` (dirty vertices, dirty triangles, `topology_changed`).
- **Snapshot fallback:** for each column the patch stores either per-slot entries or full copies, whichever is smaller in bytes. Large rewrites (e.g. remeshing) therefore never cost more than two copies of the column.
- `rollback()` restores from the journal, so a failing algorithm never leaves a half-edited mesh.
- **Apply and revert are the same operation** with the new or old values. Both emit a `ChangeSet` and flow through the same update path as normal edits.
- `add_vertices` accepts optional creation parents and weights (e.g. an edge split creates a vertex with parents `(a, b)` and weights `(0.5, 0.5)`). Source channels use them for interpolation; when absent, `nearest` is used.

### 4.5 History

- A stack of patches with a byte budget. Exceeding the budget evicts the oldest patches (and releases their slots).
- A new commit after undo discards the redo branch.
- Per-mesh history in the store. Scene-wide history (one undo step can span several meshes) is built on top of it in the Scene phase.
- Applications with their own undo system can ignore `History` and store `Patch` objects themselves; `patch.apply()` / `patch.revert()` are public.

## 5. BVH (`fast-mesh-bvh`)

### 5.1 Layout (10M triangles, max 8 triangles per leaf)

| Data | Format | Size |
| --- | --- | --- |
| Nodes | 32 bytes: AABB (24) + child index or triangle range (8) | ~80 MB |
| Parent links | `u32` per node | ~10 MB |
| Triangle references | `u32` per triangle | ~40 MB |
| Triangle → leaf | `u32` per triangle | ~40 MB |

- The BVH references triangles through its own reference array; triangles are never reordered in storage, so slot ids stay stable for undo.
- Leaf size is configurable (4 trades ~80 MB more for faster queries).
- The BVH is built in mesh-local space. Mesh transforms never trigger rebuilds.

### 5.2 Build

- Binned SAH (16–32 bins), default `"quality"`.
- `"fast"` mode uses a Morton-code (LBVH) build for faster builds and slightly slower queries.
- Built as independent subtrees so the multithreaded build can parallelize without restructuring.
- Standalone API: `Bvh.build(positions, indices, options)` for static meshes outside a Scene.

### 5.3 Incremental updates

The Scene chooses the strategy from the `ChangeSet`:

| Change | Strategy |
| --- | --- |
| Vertex moves only | Refit: dirty triangles → their leaves → ancestors up to the root, each node once |
| Local topology change | Partial rebuild of the smallest subtree covering the dirty leaves |
| Large change (default ≥30% of triangles) or degraded quality | Full rebuild |

- Each subtree stores its SAH cost at build time. When refits push the current cost above a factor (default 1.5×), that subtree is rebuilt.
- Thresholds are options; defaults are set from benchmarks.
- Manual control: `bvh.refit(triangles)`, `bvh.rebuildRegion(triangles)`, `bvh.rebuild()`.

### 5.4 Winding-number data (lazy)

Fast winding-number evaluation needs per-node moment data (~100 MB for 10M triangles). It is built only when a field uses `signed: "winding"` (or `auto` resolves to it) and refit along the same dirty paths as the BVH.

## 6. Queries (`fast-mesh-query`)

| Query | API (JS) | Result |
| --- | --- | --- |
| Raycast | `mesh.raycast(origin, dir, { mode: "closest" \| "any" \| "all", maxDistance })` | triangle, `t`, point, barycentrics, normal |
| Closest point | `mesh.closestPoint(point, { maxDistance })` | point, triangle, distance |
| Mesh distance | `a.distanceTo(b, { maxDistance })` | distance, closest points on both |
| Intersection | `a.intersects(b)` (early exit), `a.intersect(b)` | boolean; triangle pairs; intersection polylines |
| Region | `mesh.query({ sphere \| box }, "vertices" \| "triangles")` | slot ids |
| Plane section | used by `Section` (section 9) | contours, fill triangles |

- **Batch variants** (`raycastMany`, `closestPointMany`) take and return typed arrays to avoid per-call boundary overhead. Built-in fields use them internally.
- Mesh–mesh queries traverse both BVHs simultaneously, pruning pairs of distant or disjoint boxes.
- Triangle–triangle tests use robust orientation predicates (Shewchuk-style) and handle coplanar triangles explicitly.
- Queries against a transformed mesh convert the other operand into local space.
- **Limitation:** distance and closest-point results are exact under rigid transforms with uniform scale. Non-uniform scale distorts distances; the API warns when it is used with distance queries. Raycasts and intersections are exact under any affine transform.

## 7. Scene (`fast-mesh-scene`)

### 7.1 Meshes and transforms

- `scene.addMesh(...)`, `scene.allocMesh(...)` (section 11.2), `mesh.setTransform(matrix4 | { position, rotation, scale })`.
- `setTransform` only stores a matrix: no vertex, BVH or render-buffer work. Its cost is the dependent field updates it triggers.

### 7.2 Broadphase and contacts

- A top-level BVH over mesh world AABBs, rebuilt on each flush (one element per mesh, negligible cost).
- `scene.findContacts({ margin, pairs, filter: { group, mask } })` returns, per candidate pair: `intersecting`, `distance` (when separated within margin), triangle pairs, contact points and normals. Results are returned as typed arrays per field.

### 7.3 Dependency graph

Nodes are mesh geometry, mesh transform, BVHs, fields, colors and sections. Edges are declared implicitly by creating fields and sections. The Scene topologically sorts updates and rejects cycles when a field is registered.

### 7.4 Update policies

```ts
mesh.setUpdatePolicy("eager");   // default: update on commit
mesh.setUpdatePolicy("lazy");    // update before the next query or on scene.flush()
mesh.setUpdatePolicy("manual");  // update only on mesh.update()
scene.flush();
```

Typical use: `lazy` while dragging a brush or rotating a mesh, `flush()` on release.

### 7.5 Events

```ts
scene.on("updated", ({ revision, meshes, sections }) => { ... });
// meshes[i]: { mesh, dirty: { positions, normals, colors, indices: Range[] }, topologyChanged }
scene.on("capacityChanged", ({ mesh }) => { ... }); // render buffers must be recreated
```

## 8. Fields and colors (`fast-mesh-fields`)

### 8.1 Kinds

| Kind | Examples | Computed by | Undo |
| --- | --- | --- | --- |
| Source | painted mask, external values | user writes | journaled |
| Built-in | normals, distance, curvature, thickness | library | recomputed |
| Custom | application-specific | registered kernel | recomputed |

Scalar fields are one `f32` per vertex; vector fields are three.

### 8.2 Built-in fields (phase 1)

| Field | Recompute set when vertices V change |
| --- | --- |
| `normals` | V ∪ 1-ring(V) |
| `distanceTo(other, { maxDistance, signed })` | see 8.3 |
| `curvature("mean" \| "gaussian")` | V ∪ 1-ring(V) (builds adjacency) |
| `thickness({ maxDistance })` | V ∪ vertices whose previous ray hit a dirty triangle ∪ vertices whose ray crosses the dirty region's bounds |

### 8.3 Distance to another mesh

When region R of `other` changes (old ∪ new bounds of dirty triangles, in this mesh's local space):

1. Candidates: this mesh's vertices within `maxDistance` of R, found through this mesh's BVH.
2. Recompute only candidates whose distance to R is ≤ their current value.

When `other`'s transform changes, candidates are vertices within `maxDistance` of `other`'s old or new bounds. Without `maxDistance`, step 1 becomes a linear scan with a cheap point-to-box test. Documentation strongly recommends setting `maxDistance`.

### 8.4 Signed distance modes

| Mode | Method | Use |
| --- | --- | --- |
| `pseudonormal` | angle-weighted pseudonormal at the closest feature | closed meshes, fastest |
| `winding` | fast generalized winding number (threshold 0.5) | open meshes, self-intersections |
| `cap` | inside/outside of the mesh virtually capped with min-area hole triangulations | open meshes where explicit caps are wanted |
| `auto` (default) | `pseudonormal` if closed, else `winding` | — |

- Caps are virtual: they never modify the mesh or history. `mesh.holes()` and `mesh.capHoles({ virtual: true })` expose them for visualization.
- Min-area triangulation (O(n³) in boundary vertices) falls back to a fast approximate triangulation above a configurable boundary size.
- Edits near a boundary recompute only that hole's cap.
- Negative signed distance to another mesh is the per-vertex penetration depth, which is how "how deeply do these meshes overlap" is visualized.

### 8.5 Custom fields

```ts
mesh.fields.register("risk", {
  dependsOn: ["positions", "field:gap", "field:curvature"],
  neighborhood: 0,
  compute(dirty, inputs, out) { /* dirty: Uint32Array, inputs/out: typed array views */ },
});
```

- Kernels run once per batch, never once per vertex across the boundary.
- The same contract is available in Rust as a `FieldKernel` trait; built-ins implement it.

### 8.6 Colors

```ts
mesh.colors.set({ field: "gap", map: "turbo", range: [0, 2], outOfRange: "clamp", nanColor });
```

- Output is an RGBA8 derived channel. Only vertices whose field value changed are recolored.
- Built-in maps: `viridis`, `turbo`, `coolwarm`, `grayscale`; custom gradient stops are allowed.
- Default range is fixed. `range: "auto"` recolors everything when min/max change.
- Applications may also write colors directly by using a source color channel.

## 9. Sections

```ts
const section = scene.addSection({ plane: { origin, normal } | matrix4, meshes: [a, b], fill: true });
section.setPlane(plane);
section.setMeshes([b]);
section.setVisibility(a, { contour: true, fill: false });
section.result(a); // { contours: polylines (closed or open), fill: triangles, length, area }
section.gap(a, b); // minimum distance between the two meshes' contours in the plane
```

- The plane is transformed into each mesh's local space; BVH traversal visits only nodes straddling it; segments are chained into contours; `fill` triangulates contours with holes using the same triangulator as hole filling.
- Only selected meshes are computed. Several sections can exist at once.
- Recompute triggers: plane change (that section only), selected mesh edit **only if the dirty region intersects the plane**, selected mesh transform change.
- Moving or rotating the plane is not a mesh transform: it touches no field, BVH or contact data.
- Clipping (hiding one side) is the renderer's job; the three.js adapter applies per-material clipping planes to the selected meshes only.

## 10. Operations and repair (`fast-mesh-ops`)

### 10.1 Operation contract

- Every operation takes a transaction and writes only through it. Undo, BVH updates, field updates and value interpolation therefore work for every operation automatically.
- Operations provide creation parents and weights for new vertices when they can.
- Operations validate preconditions (e.g. manifold neighborhood) and fail with a typed error before writing, or roll back on failure.

### 10.2 Phase 1

| Operation | Description |
| --- | --- |
| `analyze()` | Report: holes, duplicates, degenerate and duplicate triangles, non-manifold elements, components, self-intersection presence. Never modifies. |
| `weld(tolerance)` | Merge coincident vertices |
| `removeDegenerate()` | Remove zero-area and duplicate triangles |
| `removeSmallComponents(minTriangles \| minArea)` | Remove floating pieces |
| `fillHoles({ maxEdges })` | Min-area triangulation of boundary loops |

### 10.3 Later phases

`orient()`, `splitNonManifold()`, `fillHoles` refinement and fairing, remeshing, smoothing, decimation, self-intersection resolution.

## 11. JavaScript API (`fast-mesh-wasm`)

### 11.1 Entry points

| Import | Contents |
| --- | --- |
| `fast-mesh-wasm` | Synchronous core: `Scene`, `Mesh`, fields, queries, sections, ops, standalone `Bvh`, existing utilities |
| `fast-mesh-wasm/worker` | Same API, asynchronous, running in a Worker |
| `fast-mesh-wasm/three` | three.js adapter (`three` as optional peer dependency) |
| `fast-mesh-wasm/react` | React hooks (`react` as optional peer dependency) |

The core is usable with any renderer through typed-array views and `updated` events.

### 11.2 Zero-copy input and output

```ts
const mesh = scene.allocMesh({ vertices, triangles });
mesh.views.positions().set(data);
mesh.views.indices().set(indices);
mesh.finishLoad();

scene.addMesh(positions, indices); // convenience, copies once
```

- Render buffers are slot-aligned: vertex `i` is element `i`. Removed triangles are written as degenerate `(0, 0, 0)` triangles, which GPUs skip, so dirty slots map directly to dirty buffer ranges and no compaction is needed for rendering.
- Growing capacity emits `capacityChanged`; only then must GPU buffers be recreated.
- **View lifetime rule:** wasm memory growth invalidates views. Views must be re-acquired on each use (cheap). Development builds warn when a detached view is used.

### 11.3 Lifetime and errors

- `Scene` owns meshes, fields, sections and history. `scene.free()` or `using scene = new Scene()` releases everything; a `FinalizationRegistry` is a safety net only.
- Errors are `FastMeshError` with a `code`: `INVALID_INDEX`, `INVALID_ARGUMENT`, `CYCLIC_FIELD`, `NON_MANIFOLD`, `OUT_OF_MEMORY`, `INSTANCE_POISONED`, ...
- Large allocations use fallible reservation so memory exhaustion is reported as `OUT_OF_MEMORY` instead of aborting.
- A Rust panic poisons the instance; every later call throws `INSTANCE_POISONED` rather than returning wrong results.

## 12. Concurrency and UI synchronization

Applies to the Worker wrapper and therefore to React and other async UIs.

1. **Declarative state.** Display and configuration calls describe the desired state (`mesh.display.set({ colors: ... })`), are idempotent, and are safe under React StrictMode double invocation.
2. **Ordered queue.** All state-changing calls to a Worker scene run in arrival order.
3. **Coalescing.** Pending, not yet started requests targeting the same state are collapsed to the last one. Rapid on/off/on/off resolves to the final state without intermediate work.
4. **Cooperative cancellation.** Long computations run in chunks and check between chunks whether they have been superseded.
5. **Revisions.** Every state change gets a monotonically increasing revision; results and events carry it; the main thread ignores results older than what it has already applied.
6. **Display is separate from computation.** Hiding a field keeps its values. Showing it again is immediate if nothing changed, and incremental otherwise.

Worker data paths:

- With `SharedArrayBuffer` (cross-origin isolated pages): render views are shared memory; no copies.
- Without it: on each `updated` event the Worker transfers only dirty ranges.
- **Known limitation:** in Worker mode queries are asynchronous and arrive at least one frame later. For latency-critical picking, use the synchronous core on the main thread or a lightweight picking mesh there.

## 13. Integrations

### 13.1 three.js (`fast-mesh-wasm/three`)

- `bindGeometry(mesh)` creates and keeps a `BufferGeometry` in sync (position, normal, color, index) using dirty ranges and `addUpdateRange`.
- Keeps the object matrix in sync with `mesh.transform`.
- `bindSection(section)` applies clipping planes to the selected meshes' materials and renders contours and fill.

### 13.2 React (`fast-mesh-wasm/react`)

- `useScene()`, `useMeshDisplay(mesh, desiredState)` → `{ status: "idle" | "computing" | "ready" | "error", revision, error }`, `useSection(...)`.
- Subscriptions use `useSyncExternalStore` so components never observe torn state.
- Hooks rely on the section 12 model; UIs can fire changes as fast as users click.
- Works with React Three Fiber through the three.js adapter.

## 14. Testing

| Level | What |
| --- | --- |
| Store | Property tests: random edits then full undo equals the original bit for bit; full redo equals the final state; history stays within budget; rollback restores exactly |
| BVH | Queries equal brute force on random meshes; still equal after random edits with refit or partial rebuild; invariants (parent contains children, each live triangle in exactly one leaf) |
| Queries | Brute-force comparison under random transforms; symmetry (`a.distanceTo(b) == b.distanceTo(a)`); transform invariance; degenerate, coplanar and edge-touching cases |
| Fields | Incremental result equals full recompute after random edits (tolerance for distances); cycle rejection; topology value policies; colormap edge cases |
| Sections | Contours match brute-force plane slicing; incremental recompute only when the plane is hit |
| Ops | Each repair on fixture meshes with known defects; `analyze` before/after; undo restores the input |
| JS (Node) | API behavior, error codes, view invalidation, freeing |
| Browser | Headless browser: Worker mode, `SharedArrayBuffer` mode, three.js adapter |
| Sync model | Property test: random rapid sequences of state changes with random delays converge to the last requested state; stale revisions never render |
| Install | Existing npm/pnpm/bun × Linux/Windows/macOS smoke tests |
| Benchmarks | Build, refit, queries, field updates, sections at 1M and 10M triangles; memory per mesh; wasm size. Tracked over time, not gating |

## 15. Phases

1. Store, transactions, History
2. BVH (build, refit, partial rebuild, quality tracking)
3. Queries (raycast, closest point, distance, intersection, region, plane slicing)
4. Scene (transforms, broadphase, contacts, dependency graph, policies, events)
5. Fields and colors (built-ins, signed modes, custom fields)
6. Sections
7. Repair operations (phase 1 set)
8. JS API and three.js adapter
9. Worker wrapper, sync model, React hooks
10. Later: multithreaded build, `f64` storage, wasm64, phase-2 repairs, approximate penetration depth for closed meshes

Each phase gets its own implementation plan and is released behind changesets. The existing `computeVertexNormals` and `computeBounds` stay as standalone utilities.

## 16. To verify during implementation

- Whether the multithreaded build (`wasm-bindgen-rayon` style) still requires a nightly toolchain.
- Current browser support and performance of wasm memory64.
- Default thresholds (snapshot 50%, rebuild 30%, SAH degradation 1.5×, min-area triangulation size limit) from benchmarks.
- Actual memory per 10M-triangle mesh versus the estimates in this document.
