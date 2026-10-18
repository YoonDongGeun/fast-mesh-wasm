# fast-mesh Phase 8: JavaScript API, three.js adapter and WebGPU example

- **Status:** draft for review.
- **Parent spec:** `2026-10-14-fast-mesh-engine-design.md`, §11 (JavaScript API), §13.1 (three.js) and §15 (phase 8). Where this document differs from the parent, this document wins.
- **Builds on** every Rust phase so far:
  - store and BVH
  - queries
  - Scene
  - fields and colours (5a)
  - sections (6)
  - repair (7)
- **Labels:** what the owner said is marked *(owner)*. Choices I made without asking are marked *(my call)*.

## 1. Intent

The owner pictures a browser app *(owner)* with these features:
- 2–5 meshes loaded at once.
- Each mesh can be rotated and moved.
- Live distance colouring that can be switched on and off.
- Per-mesh transparency.
- Keyboard shortcuts for actions.
- Undo and redo.

Renderers:
- The library must work with any renderer *(owner)*.
- three.js gets an official adapter *(owner)*.
- A plain WebGPU example shows the renderer-neutral path *(owner)*.
- Other renderers (Babylon, vtk.js and so on) are left to examples or later work *(owner)*.

Display settings are not undoable *(owner)*. This covers colouring on or off and transparency. Undo covers mesh edits and placements, as in the Scene today.

Phase 8 has two parts, built in this order:
- **8a: renderer-neutral JS core.** A wasm-bindgen layer plus a TypeScript API over the whole engine, with zero-copy render buffers and dirty ranges.
- **8b: integrations.**
  - `fast-mesh-wasm/three`, the three.js adapter.
  - `examples/webgpu`, a raw WebGPU app that reproduces the owner's scene.

Each part gets its own plan.

## 2. Goals, non-goals, anti-goals

### Goals
1. **Scene API in TypeScript, synchronous on the main thread.** It covers:
   - adding and removing meshes from typed arrays;
   - placements;
   - update modes;
   - undo, redo and history steps, with a single undo step per drag;
   - world queries (raycast for picking, closest point, distance);
   - distance fields and colourings;
   - sections;
   - repair operations;
   - change subscriptions.
2. **Render buffers.** Per mesh: positions, normals, colours and indices, slot-aligned and zero-copy over wasm memory. Each change set reports which ranges of each buffer changed, plus a capacity change when buffers must be recreated.
3. **Incremental vertex normals.** New in Rust. Smooth, area-weighted vertex normals, recomputed only where geometry changed *(my call: renderers need normals for shading, and recomputing them in JS each frame defeats the point)*.
4. **Errors and lifetime.**
   - Errors are thrown as `FastMeshError` with a stable `code`.
   - `scene.free()` and `using` release memory. A `FinalizationRegistry` is only a safety net.
   - A Rust panic poisons the instance: every later call throws `INSTANCE_POISONED`.
5. **three.js adapter.** It covers:
   - a `BufferGeometry` kept in sync through dirty ranges;
   - the object matrix kept in sync with the placement;
   - per-mesh opacity;
   - `TransformControls` drags that become placement changes, one undo step per drag;
   - section clipping planes and contour lines.
6. **WebGPU example.** The owner's scene with no 3D library:
   - 2–5 meshes;
   - drag to rotate and move;
   - distance colouring toggled by a key;
   - opacity;
   - undo and redo on Ctrl+Z and Ctrl+Y.

### Non-goals (this phase)
- Workers, `SharedArrayBuffer` and async APIs: Phase 9.
- React hooks: Phase 9.
- Babylon, vtk.js and other renderer adapters.
- A full-featured editor UI. The examples are small demos.
- The parent spec's `allocMesh` and `finishLoad` streaming load. `addMesh` copies once *(my call; streaming can come later)*.

### Anti-goals
- **No copying on hot paths.** Re-uploading a whole mesh, or copying whole buffers into JS on every change, would defeat the incremental engine.
- **No silent detached views.** Wasm memory growth detaches old typed-array views. The API must make the safe pattern the easy one, and a stale view must never quietly read zeros.
- **No leaking engine ids that look like plain numbers but go stale.**

## 3. Constraints
- **Rust side:** the existing `fast-mesh-wasm` crate, which is `cdylib` + `rlib`. It depends on the engine crates through workspace dependencies.
  - Built for `wasm32-unknown-unknown` with `wasm-bindgen` 0.2.129 (`scripts/build-wasm.mjs`).
  - wasm-opt, with features limited to what rustc emits.
- **TypeScript side:** the existing `packages/fast-mesh-wasm` package.
  - Built with tsdown; ESM only.
  - Two entries, `node` and `web`. Both keep their current `init()` contract.
  - New export `./three`. `three` is an optional peer dependency with range `>=0.160` *(my call)*.
- **Existing utilities stay:** `computeVertexNormals` and `computeBounds` remain as they are (parent spec §15).
- **Node and lint:** Node ≥ 22. CI keeps `publint --strict` and `attw --profile esm-only` passing.
- **Release:** a changeset is required, since this changes the npm package.
- **Rust rules as before:**
  - no `unsafe` in engine crates;
  - typed errors;
  - fallible allocation;
  - MSRV 1.88.
- **`unsafe` in `fast-mesh-wasm`:** only where wasm-bindgen view creation needs it, for example `js_sys::Float32Array::view`. Each such block is commented with its lifetime argument.

## 4. Rust additions

### 4.1 Vertex normals in the Scene (`fast-mesh-scene`)
- **Turning normals on or off:** `scene.set_normals(id, on: bool) -> Result<()>`. Normals are off by default in Rust. The JS layer turns them on for every mesh it adds (§5.2).
- **Formula:** a vertex normal is the normalised sum of the area-weighted normals of its incident proper triangles, in mesh-local `f32`. Vertices with no triangles get `[0, 0, 0]`. The result matches `fast_mesh::compute_vertex_normals` within 1e-6 on the same mesh.
- **Incremental update.** After a change set, recompute the normals of:
  - every vertex of every triangle that was added, removed or changed;
  - every vertex of every triangle incident to a moved vertex;
  - all vertices when the change set is `None` or after compaction.
- **Placement:** a placement change does not touch normals, because they stay local.
- **Timing** follows the mesh's update mode, as colours do. Eager meshes update right after the change. Lazy and Manual meshes update on `update(id)`, `flush()` or the existing update calls.
- **Readers:**
  - `scene.normals(id) -> Result<Option<&[[f32; 3]]>>`. The slice length is `vertex_slots`.
  - `MeshChanges` gains `normals_dirty: Option<Vec<VertexId>>`, with the same semantics as `colors_dirty`.

### 4.2 Render indices (`fast-mesh-wasm` or `fast-mesh-scene`, builder's choice)
- A slot-aligned `u32` index buffer with length `3 · triangle_slots`.
- A live triangle writes its three vertex ids. A removed triangle slot writes `(0, 0, 0)`, which GPUs skip (parent spec §11.2).
- It is kept up to date from change sets, so only changed triangle slots are rewritten.

### 4.3 The wasm binding layer (`fast-mesh-wasm`)
- **Classes:**
  - `#[wasm_bindgen] pub struct WasmScene` wraps `Scene`.
  - Each method converts arguments, calls the Scene and maps errors (§4.4).
  - Ids cross the boundary as two `u32` values (index and generation) and are wrapped by TS objects (§5.1).
- **Views:**
  - Methods such as `positions_ptr(id) -> u32` and `positions_len(id) -> u32` (the same pair for normals, colours and indices) let the TS layer build `Float32Array`, `Uint8Array` and `Uint32Array` views over `wasm.memory.buffer` without copying.
  - A view is valid until the next call that may allocate (§5.3).
- **Change subscription:**
  - The binding layer owns one Scene subscriber (`subscribe()`).
  - `take_changes()` returns a compact encoded record per mesh: flags, then sorted dirty-range lists for positions, normals, colours and indices, plus capacity and removed flags.
  - It travels as one `Uint32Array` copied into JS, to avoid one JS object per range *(my call)*.
  - **Ranges:** consecutive dirty slots merge into `(start, count)` pairs. Gaps of up to 8 slots also merge *(my call: fewer, slightly larger GPU uploads)*.
- **Panics:**
  - Build with `panic = "abort"` behaviour as it is today. A panic traps as `WebAssembly.RuntimeError`.
  - The TS layer catches the trap, sets a module-wide poisoned flag and throws `FastMeshError("INSTANCE_POISONED")`. Every later call throws the same error.
  - A `console_error_panic_hook`-style message is not required. The builder may add one if the size cost is small.

### 4.4 Error codes
- **The class:** `FastMeshError extends Error` with `code: string` and `message`.
- **Codes:**

| Rust error | Code |
| --- | --- |
| `UnknownMesh` | `UNKNOWN_MESH` |
| `UnknownField` | `UNKNOWN_FIELD` |
| `UnknownSection` | `UNKNOWN_SECTION` |
| `InvalidInput` | `INVALID_ARGUMENT` |
| `StaleBvh` | `STALE_BVH` |
| `StepNotOpen` | `STEP_NOT_OPEN` |
| `Corrupted` | `CORRUPTED` |
| any `OutOfMemory` | `OUT_OF_MEMORY` |
| `Ops(StaleLoops)` | `STALE_LOOPS` |
| other store, query, field or ops errors | `INVALID_ARGUMENT`, or a more specific code where one exists |

  - A panic is reported as `INSTANCE_POISONED`.
  - The input-shape checks done in JS (lengths, index ranges) use `INVALID_INDEX` or `INVALID_ARGUMENT`.

## 5. TypeScript API (8a)

### 5.1 Objects
- **`Scene`:**
  - Created with `new Scene(options?)` after `await init()`.
  - Owns everything inside it.
  - `free()` releases it. `[Symbol.dispose]` lets `using` release it automatically.
  - Using a scene after `free()` throws `INVALID_ARGUMENT` ("scene is freed").
- **Handles** (`MeshHandle`, `FieldHandle`, `SectionHandle`):
  - Opaque objects that hold the id privately and are bound to their scene.
  - Methods live on the scene and take handles, e.g. `scene.setPlacement(mesh, p)`.
  - A handle from another scene gives `INVALID_ARGUMENT`. A removed one gives `UNKNOWN_MESH` and so on.
  - Handles compare by identity: the same id always maps to the same handle object, through a per-scene map *(my call)*.

### 5.2 Meshes and placements
- **Adding:** `scene.addMesh(positions, indices, options?) -> MeshHandle`.
  - Accepted inputs: Float32Array or array-like positions; Uint32Array, Uint16Array or array-like indices.
  - Options:
    - `origin?: Vec3`
    - `mode?: "eager" | "lazy" | "manual"`, default `"eager"` *(my call: interactive apps want Eager)*
    - `normals?: boolean`, default `true`
- **Removing:** `scene.removeMesh(mesh)`.
- **Placement:** `scene.setPlacement(mesh, { rotation, translation })` and `scene.placement(mesh)`.
  - `rotation` is a row-major 3×3 matrix or a unit quaternion `[x, y, z, w]`.
  - The quaternion is converted in f64.
  - A matrix that is not a proper rotation gives `INVALID_ARGUMENT`.
- **Matrix helper:** `scene.matrix4(mesh) -> Float64Array(16)`.
  - It is the column-major world matrix from local render coordinates: `placement · translate(origin)`. It is ready for renderers.
  - Callers convert to f32 themselves. Precision is the renderer's concern.
- **Other mesh calls:** `scene.setMode(mesh, mode)`, `scene.update(mesh)`, `scene.flush()`.
- **Mesh facts:** `scene.vertexSlots(mesh)`, `scene.triangleSlots(mesh)`, `scene.vertexCount(mesh)`, `scene.triangleCount(mesh)`.

### 5.3 Render views and the view lifetime rule
- **Getting views:** `scene.views(mesh)` returns `{ positions(): Float32Array, normals(): Float32Array | null, colors(): Uint8Array | null, indices(): Uint32Array }`.
- **Each getter builds a fresh view** over the current wasm memory, which is cheap. Callers must not keep a view across any scene call that may allocate. The docs and the type comments state this rule, and it is the safe pattern the adapters follow.
- **Detached-view guard:** in development builds (`process.env.NODE_ENV !== "production"`, or an `init({ debug: true })` flag, builder's choice), views are wrapped or checked so that touching a detached buffer warns once.
- **Layout:**
  - Positions and normals hold 3 floats per vertex slot. Colours hold 4 bytes (RGBA8) per vertex slot. Indices hold 3 `u32` per triangle slot, and removed triangles are `(0,0,0)`.
  - Dead vertex slots keep stale data. Nothing references them.

### 5.4 Changes
- **Reading changes:** `scene.takeChanges() -> MeshChange[]`. A `MeshChange` contains:
  - `mesh: MeshHandle`
  - `added`, `removed`, `placementChanged`, `capacityChanged`: booleans
  - `positions`, `normals`, `colors`, `indices`: `Range[] | "all"`, where `Range = [start, count]` in elements (vertices or triangles)
  - `fieldsChanged: FieldHandle[]` and `sectionsChanged: SectionHandle[]`
- **`capacityChanged`** is true when `vertexSlots` or `triangleSlots` changed since the last call. Renderers then recreate their GPU buffers and upload everything.
- **Single consumer:** there is one change consumer per JS scene *(my call: adapters share it through a small fan-out helper, `scene.onChanges(listener)`, which drains on `scene.commitFrame()`)*.
  - `commitFrame()` calls `takeChanges()` once and delivers the result to every listener.
  - Apps call it once per animation frame, and the adapters require it.

### 5.5 History
- **Basic calls:** `scene.undo()`, `scene.redo()`, `canUndo()`, `canRedo()`, `clearHistory()`.
- **Steps:** `scene.beginStep()`, `scene.endStep()`, `scene.cancelStep()`.
- **Drags:** `scene.drag(mesh)` returns a `DragSession` with `move(placement)`, `end()` and `cancel()`. It wraps a history step, so a whole drag is one undo step. `cancel()` restores the start placement.
- **Display settings** (colouring, opacity in adapters) never enter history *(owner)*.

### 5.6 Queries
- `scene.raycast(origin, direction, options?)` returns `{ mesh, triangle, point, distance, barycentric } | null`. It uses the world raycast. Option `meshes?: MeshHandle[]` restricts the search.
- `scene.closestPoint(point, options?)`.
- `scene.meshDistance(a, b)` and `scene.intersects(a, b)`.
- `scene.findContacts(options?)`.
- Results are plain objects with `number[]` vectors.

### 5.7 Fields and colours
- **Fields:**
  - `scene.addDistanceField({ name, on, to, maxDistance, sign? }) -> FieldHandle`.
  - `scene.removeField(field)`.
  - `scene.fieldValues(field) -> Float32Array`. This is a view and follows the lifetime rule.
  - `scene.fieldStatus(field)`.
- **Colouring:** `scene.setColoring(mesh, rule | null)`.
  - `rule` is `{ field, map: "turbo" | "viridis" | "coolwarm" | stops, range: [lo, hi], below?, above?, center?, base?, nan? }`, mirroring `ColorRule`.
  - Passing `null` turns colouring off, and `colors()` then returns `null`. This is the owner's on/off toggle.
  - It is a display setting, not history.

### 5.8 Sections
- **Calls:**
  - `scene.addSection({ plane: { origin, normal }, meshes, fill? }) -> SectionHandle`
  - `setSectionPlane`, `setSectionMeshes`, `setSectionFill`, `removeSection`, `updateSections`
- **Reading a section:** `scene.section(section, mesh)` returns `{ status, contours: { points: Float32Array /* xyz world */, closed: boolean }[], length, area, fill?: { positions: Float32Array /* world xyz */, indices: Uint32Array } }`. These are copies, because sections are small.
- **Gap:** `scene.sectionGap(section, a, b)`.

### 5.9 Repair
- **Read-only report:** `scene.analyze(mesh, options?) -> Analysis`, as plain data. `LoopId.hash` is a `bigint`.
- **Single operations:** `scene.repair(mesh, op)`, where `op` is one of `{ op: "weld", tolerance } | { op: "removeDegenerate", ... } | { op: "removeSmallComponents", ... } | { op: "orient" } | { op: "fillHoles", ... }`. It returns that op's report.
- **Pipeline:** `scene.repair(mesh, [op, ...])` runs several ops in one edit, so the whole pipeline is one undo step, and returns their reports in order.
- **Errors:** they map through §4.4.

## 6. three.js adapter (8b, `fast-mesh-wasm/three`)
- **Binding a mesh:** `bindMesh(scene, mesh, options?) -> BoundMesh`. `BoundMesh` exposes:
  - `object: THREE.Mesh`, whose `BufferGeometry` has position, normal, color (when colouring is on) and index attributes;
  - `setOpacity(alpha)`, a display setting that switches the material's transparency and depthWrite;
  - `dispose()`.
- **Keeping geometry in sync:** on `scene.commitFrame()`, the adapter applies the dirty ranges with `attribute.addUpdateRange(start, count)` and `needsUpdate`.
  - When `capacityChanged` is set, it recreates the attributes.
  - When the placement changed, it updates `object.matrix` from `scene.matrix4(mesh)` and sets `matrixAutoUpdate = false`.
- **Colours:**
  - With colouring on, the colour attribute is a normalised `Uint8BufferAttribute` and `material.vertexColors = true`.
  - With colouring off, the attribute is removed and the material uses `options.color`.
- **Drags:** `attachTransformControls(controls, scene, bound)` maps TransformControls' `dragging-changed` and `objectChange` events to `scene.drag(mesh)`, `move` and `end`. One drag gives one undo step. On undo or redo, the object follows the placement through the change stream.
- **Sections:** `bindSection(scene, section, options?) -> BoundSection`.
  - It adds a `THREE.Plane` to `material.clippingPlanes` of the selected meshes' materials, so the renderer needs `localClippingEnabled = true`.
  - It draws contours as `LineSegments` and fill as a mesh, updated when `sectionsChanged` lists the section.
- **Dependencies:** no runtime dependency beyond the `three` peer.
- **Tests:** they run in Node with `three` and no WebGL context, so they cover geometry, attributes, update ranges, matrices and the event mapping.

## 7. WebGPU example (8b, `examples/webgpu`)
- **Setup:** a Vite app (not published) using the same `fast-mesh-wasm` workspace package, with no 3D library.
- **Scene:** 3 to 5 generated meshes, for example a sphere, a torus and a wavy plate, or files dropped onto the page.
- **Rendering:**
  - One render pipeline: a lit shader with per-vertex colour and per-object opacity, plus alpha blending for transparent meshes, which are sorted back to front by object centre.
  - Per mesh, GPU buffers are created from the views, updated with `queue.writeBuffer` over the dirty ranges, and recreated on `capacityChanged`.
- **Interaction:**
  - Click to pick a mesh with `scene.raycast`.
  - Drag to move it; Shift+drag to rotate it (arcball). Each drag is one undo step.
  - Keys:
    - Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z): undo / redo
    - C: toggle distance colouring between the picked mesh and its nearest neighbour
    - 1–9: set the picked mesh's opacity
    - S: toggle a section plane through the scene centre
- **A README** explains the renderer-neutral pattern in about 20 lines:
  1. read the views;
  2. upload the dirty ranges;
  3. recreate buffers on a capacity change;
  4. apply `matrix4`.
- **Browser support:** if WebGPU is unavailable, the page shows a message. There is no WebGL fallback *(my call)*.

## 8. Testing
- **Rust:**
  - Normals: incremental equals from-scratch after random edits, undo, redo and compaction, as a property test.
  - Render indices: removed slots are zero.
  - Range encoding: the round trip is exact, and merging respects the gap rule.
- **TypeScript (vitest, Node):**
  - Lifecycle: init guard, free, `using`, and the finalizer safety net, where testable.
  - Errors: each code is reachable.
  - Poisoning: a test-only export makes Rust panic; afterwards every call throws `INSTANCE_POISONED`.
  - Views:
    - positions, normals, colours and indices match the engine;
    - after forcing memory growth (adding a big mesh), old views are detached and new getters work;
    - the debug guard warns.
  - Changes: ranges cover exactly the edited slots (with the gap merge), `capacityChanged` fires on growth, and `"all"` is reported after compaction.
  - History: a drag gives one undo step; undo and redo restore placements; display settings are not undone.
  - Queries, fields, sections and repair: small end-to-end checks against known values (raycast hit, gap, filled hole).
- **three.js adapter (Node):**
  - attribute contents and update ranges after edits;
  - capacity recreation;
  - matrix sync;
  - opacity flags;
  - TransformControls event mapping with a mocked controls object, giving one undo step;
  - section clipping planes and line geometry.
- **WebGPU example:** it builds in CI (`vite build`). A short manual checklist lives in its README.
- **Size:** record the wasm size (raw and gzip) before and after in `docs/benchmarks.md`. There is no hard limit *(my call)*, but a regression of more than 20% between releases is called out.

## 9. Benchmarks
- JS-side cost per frame for a 1M-triangle mesh under a 64-pose drag with colouring on: `commitFrame()` plus range application in the three.js adapter, measured in Node without GPU upload.
- Initial `addMesh` of 1M triangles from typed arrays, measured end to end in JS.
- Record the results in `docs/benchmarks.md`.

## 10. Left to the builder
- The exact wasm-bindgen method shapes and the change-record encoding, provided §4.3 holds: one typed-array copy per `takeChanges`, and views without copies.
- Whether render indices live in the scene or the binding crate.
- The debug-guard mechanism.
- The arcball details in the WebGPU example, and its mesh generators.
- How each part splits into plan tasks. 8a and 8b get separate plans.

## 11. Decisions after the builder check (they override earlier sections)

1. **Geometry edits from JS, executed in Rust** *(owner: "do it in Rust")*. `scene.edit(mesh, ops)` takes an op list and runs it as one Rust transaction through `Scene::edit`, so a single call is one undo step. If any op fails, nothing changes (rollback). Normals, colours, sections and the BVH follow automatically.
   - Ops:
     - `{ op: "setPositions", vertices: Uint32Array, positions: Float32Array | Float64Array }`. Positions are in world space and are converted to local with the mesh's placement and origin.
     - `{ op: "addVertices", positions }`. Returns new vertex ids in the result.
     - `{ op: "addTriangles", triangles: Uint32Array }`. Returns new triangle ids.
     - `{ op: "removeTriangles", triangles }`
     - `{ op: "removeVertices", vertices }`
   - Ops may refer to vertices added earlier in the same list by index into that op's result, using `{ ref: [opIndex, i] }` *(my call)*.
   - The ops travel to wasm as one encoded buffer, not one call per op.
   - `scene.compact(mesh)` is exposed. It renumbers slots, so its change record reports `"all"` and `capacityChanged`.
   - Undo, redo and drags work as in §5.5.
2. **The three.js adapter mirrors buffers** *(my call)*. `BufferAttribute` arrays are JS-owned copies. On `commitFrame()` only the dirty ranges are copied from fresh views into them. This avoids detached arrays between `commitFrame()` and `renderer.render()` (three.js uploads lazily). The cost is roughly double JS-side memory per mesh, which is acceptable for the owner's 2–5 meshes. The WebGPU example uploads straight from views with `queue.writeBuffer`, so it makes no extra copy. The §9 benchmark measures the mirror copy plus range bookkeeping.
3. **`addMesh` coordinates** *(my call)*.
   - Positions are given in world space, as `Float32Array`, `Float64Array` or an array-like.
   - Without `origin`, the mesh origin is the bounding-box centre, computed in f64, so local `f32` coordinates stay small and precise.
   - With `origin`, that point is used.
   - Positions are stored as `p − origin` in `f32`.
   - The initial placement is the identity.
   - `views().positions()` therefore holds local coordinates, and `matrix4 = placement · translate(origin)` puts them back in the world.
   - The store gains a constructor that takes an explicit origin (with f64 input), next to the existing auto-centring `from_buffers_f64`.
4. **`DragSession.cancel()`** restores the start placement. It does not cancel the user's outer step. `undo()` during a drag ends the drag first *(my call)*.
