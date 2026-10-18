# fast-mesh JS core (Phase 8a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A renderer-neutral TypeScript API over the whole engine, synchronous on the main thread. It covers:
- scene, meshes and placements;
- history and drags;
- render views and dirty ranges;
- queries, fields/colours and sections;
- edits and repair.

**Architecture:**
- **Rust additions:**
  - an explicit-origin mesh constructor;
  - incremental vertex normals in `fast-mesh-scene`;
  - a render-state module in `fast-mesh-wasm` that keeps a slot-aligned index buffer and encodes per-frame change records into one `Vec<u32>`;
  - a `WasmScene` wasm-bindgen class.
- **TypeScript (`packages/fast-mesh-wasm/src`):**
  - wraps `WasmScene` in `Scene` with opaque handles and `FastMeshError`;
  - builds typed-array views over wasm memory, which the init output provides;
  - decodes change records;
  - offers `commitFrame()` / `onChanges` fan-out.

**Tech Stack:**
- Rust 1.99 (MSRV 1.88) with wasm-bindgen `=0.2.129`.
- TypeScript, tsdown and vitest (Node 22+).
- criterion for Rust benches; a vitest-based JS bench script.

**Spec:** `docs/superpowers/specs/2026-10-18-fast-mesh-js-api-design.md`. §11 overrides earlier sections. 8b (three.js adapter, WebGPU example) gets its own plan.

## Global Constraints

### Rust
- **Engine crates:** no `unsafe`, typed errors, fallible allocation (`try_reserve`), MSRV 1.88.
- **`fast-mesh-wasm`:** no `unsafe` (views are built in TS from pointers). Dependencies are only workspace crates plus `wasm-bindgen`; no `js-sys` unless a task proves it necessary.

### Errors
- Rust → JS errors are `JsError` with the message `"<CODE>: <text>"`. TS parses the code into `FastMeshError { code, message }`.
- Codes: `UNKNOWN_MESH`, `UNKNOWN_FIELD`, `UNKNOWN_SECTION`, `INVALID_ARGUMENT`, `INVALID_INDEX`, `STALE_BVH`, `STEP_NOT_OPEN`, `CORRUPTED`, `OUT_OF_MEMORY`, `STALE_LOOPS`, `INSTANCE_POISONED`.
- A panic traps (`WebAssembly.RuntimeError`). TS then sets a module-wide poisoned flag, and every later call throws `INSTANCE_POISONED`.

### Ids and handles
- Ids cross the boundary as `(index: u32, generation: u32)` pairs. Field and section ids are a single `u32`.
- TS handles are opaque objects bound to one scene, deduplicated per id (the same id always gives the same object).

### Render buffer layout (slot-aligned)
| Buffer | Type | Per | Notes |
| --- | --- | --- | --- |
| Positions | `Float32Array` | vertex slot, 3 values | local coordinates |
| Normals | `Float32Array` | vertex slot, 3 values | |
| Colours | `Uint8Array` | vertex slot, RGBA8 | |
| Indices | `Uint32Array` | triangle slot, 3 values | removed triangle → `0,0,0` |

### Coordinates
- `addMesh` takes world positions.
- The origin defaults to the bounding-box centre, computed in f64. Stored positions are `p − origin` as `f32`.
- `matrix4 = placement · translate(origin)`, column-major `Float64Array(16)`.

### Changes
- **Ranges:** `[start, count]` in elements, ascending. Dirty slots whose gap is ≤ 8 slots merge into one range.
- **`"all"`** after add, compaction, or `None` change sets.
- **`capacityChanged`** when slot counts change between frames.

### Defaults
- Mode `"eager"`, normals on, colouring off.
- Display settings are never in history.

### Shared conventions
- **Commits:** conventional messages ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or the implementing model). Each message goes in a NEW temp file, committed with `git commit -F`. Write files UTF-8 with LF.
- **Every task ends green on:**
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - from Task 3 on, also `pnpm build && pnpm --filter fast-mesh-wasm test && pnpm check:pkg`
- **Changeset:** the npm package changes, so add a changeset in the last task (minor bump).

## Review Focus
1. **Views across memory growth.** Take a view, add a large mesh so memory grows, then take a fresh view. The fresh view has correct contents. The old one is detached, and the debug guard warns rather than silently reading zeros. Test: Task 4 `views_survive_memory_growth`.
2. **Generational ids.** A handle to a removed mesh used after its slot is reused by a new mesh throws `UNKNOWN_MESH` and never acts on the new mesh. Test: Task 4 `stale_handle_after_slot_reuse`.
3. **Undo during a drag.** `undo()` in the middle of a `DragSession` ends the drag first. Undoing again restores the pre-drag placement exactly. Test: Task 4 `undo_mid_drag`.
4. **Edits referencing new vertices.** In one `scene.edit`, add vertices, then add triangles that use `{ref:[0,i]}`. If a later op fails, nothing changes (no new slots alive) and no undo step is recorded. Test: Task 6 `edit_ops_atomic_with_refs`.
5. **A panic in Rust.** A test-only export panics. Afterwards every Scene method and every utility function throws `INSTANCE_POISONED`. Test: Task 3 `panic_poisons_instance`.

---

### Task 1: Explicit-origin meshes and incremental vertex normals (Rust)

**Files:**
- Modify: `crates/fast-mesh-store/src/mesh.rs`. Add `Mesh::from_buffers_f64_with_origin(positions: &[f64], indices: &[u32], origin: [f64; 3]) -> Result<Mesh>`, which stores positions as `(p − origin) as f32`, plus a test.
- Create: `crates/fast-mesh-scene/src/normals.rs`.
- Modify:
  - `crates/fast-mesh-scene/src/scene.rs`: `Entry` gains `normals: Option<NormalState>`. Feed it from `record_changes`, `report_everything` and compaction, following the colouring pattern.
  - `crates/fast-mesh-scene/src/report.rs`: `MeshChanges.normals_dirty: Option<Vec<VertexId>>`.
  - `crates/fast-mesh-scene/src/lib.rs`: re-exports.
- Test: `crates/fast-mesh-scene/tests/normals_properties.rs`.

**Interfaces (produces):**
```rust
impl Scene {
    pub fn set_normals(&mut self, id: MeshId, on: bool) -> Result<()>;
    pub fn normals(&self, id: MeshId) -> Result<Option<&[[f32; 3]]>>; // len == vertex_slots when on
}
// MeshChanges { ..., normals_dirty: Option<Vec<VertexId>> }  // same semantics as colors_dirty
```
- **Formula:** normalised sum of the area-weighted normals of incident proper triangles, i.e. un-normalised cross products summed. It must match `fast_mesh::compute_vertex_normals` within 1e-6.
- **Isolated vertices:** `[0,0,0]`.
- **Dirty set:**
  - all vertices of added, removed and changed triangles;
  - all vertices of triangles incident to moved vertices;
  - everything on `None`.
- **Timing:** follows the mesh's update mode, like colours.

- [ ] **Step 1: Write failing tests.**
  - Store: `explicit_origin_stores_offsets`. Positions around 1e6 with origin `[1e6,1e6,1e6]` give local positions within 1e-3 of the offsets.
  - Scene unit tests:
    - `normals_match_compute_vertex_normals_on_add`
    - `normals_follow_vertex_move`: only the 1-ring-of-ring dirty set is reported in `normals_dirty`, and the values equal from-scratch.
    - `normals_off_by_default_and_none`
    - `normals_after_compact_all`
    - `lazy_normals_update_on_flush`
  - Property test `normals_match_from_scratch`: random edits (moves, triangle add/remove, undo/redo, compaction, mode switches) on a sphere with origin ~1000. After `flush()`, the normals equal `compute_vertex_normals` of the live mesh within 1e-5.
- [ ] **Step 2: Run** `cargo test -p fast-mesh-store -p fast-mesh-scene`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cargo test --workspace`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(store): add explicit-origin f64 constructor` and `feat(scene): incremental vertex normals`.

### Task 2: Render state and change-record encoding (Rust, native-testable)

**Files:**
- Create: `crates/fast-mesh-wasm/src/render.rs` (pure Rust, no wasm-bindgen).
- Modify:
  - `crates/fast-mesh-wasm/Cargo.toml`: deps `fast-mesh-scene`, `fast-mesh-store`, `fast-mesh-query`, `fast-mesh-fields`, `fast-mesh-ops` as workspace deps. Add `fast-mesh-scene` to root `[workspace.dependencies]` if it is missing.
  - `crates/fast-mesh-wasm/src/lib.rs`: `mod render;`

**Interfaces (produces):**
```rust
pub(crate) struct RenderState { /* per mesh: index buffer Vec<u32> (3·triangle_slots), last vertex_slots, last triangle_slots */ }
impl RenderState {
    pub(crate) fn new(mesh: &Mesh) -> Result<RenderState, StoreError>;          // full index buffer
    pub(crate) fn apply(&mut self, mesh: &Mesh, changes: &MeshChanges) -> Result<Dirty, StoreError>; // updates indices, returns dirty slot sets
    pub(crate) fn indices(&self) -> &[u32];
}
pub(crate) struct Dirty { pub positions: Option<Vec<u32>>, pub normals: Option<Vec<u32>>, pub colors: Option<Vec<u32>>, pub indices: Option<Vec<u32>>, pub capacity_changed: bool } // None = "all"
pub(crate) fn merge_ranges(sorted_slots: &[u32], gap: u32) -> Vec<(u32, u32)>;  // gap = 8
pub(crate) fn encode(records: &[Record], out: &mut Vec<u32>);                    // format below
```

**Dirty sets:**
- **positions:** moved, added and removed vertices from `changes`. Placement-only changes produce no position ranges.
- **indices:** added, removed and changed triangles.
- **normals:** `normals_dirty`.
- **colors:** `colors_dirty`.

**Encoding.** The whole message is `[record_count, …records]`. Each record is:
- `[mesh_index, mesh_generation, flags]`. Flag bits:
  - 1 added
  - 2 removed
  - 4 placement_changed
  - 8 capacity_changed
  - 16 positions all
  - 32 normals all
  - 64 colors all
  - 128 indices all
- then 4 range lists (positions, normals, colours, indices), each `[n, start0, count0, …]` with `n = 0` when "all" or none;
- then `[fields_n, field ids…]` and `[sections_n, section ids…]`.

The format is documented in `render.rs` and mirrored in the TS decoder (Task 4).

- [ ] **Step 1: Write failing tests:**
  - `merge_ranges_gap_rule`:
    - `[1,2,3,12,13]` with gap 8 gives `[(1,13)]`.
    - `[1,2,3,20]` gives `[(1,3),(20,1)]`.
  - `indices_zero_for_removed_triangles`
  - `indices_follow_triangles_changed`
  - `capacity_changed_on_growth_only`
  - `encode_round_trip`: a decoder in the test parses back what was encoded.
  - `placement_only_change_has_no_position_ranges`
- [ ] **Step 2: Run** `cargo test -p fast-mesh-wasm`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cargo test --workspace`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(wasm): render index buffer and change-record encoding`.

### Task 3: `WasmScene` bindings: meshes, placements, views, changes, history, errors, poisoning

**Files:**
- Create: `crates/fast-mesh-wasm/src/scene.rs` (wasm-bindgen class), `crates/fast-mesh-wasm/src/error.rs`
- Modify:
  - `crates/fast-mesh-wasm/src/lib.rs`
  - `crates/fast-mesh-wasm/Cargo.toml`: feature `test-panic` = exports `__test_panic()`
  - `scripts/build-wasm.mjs`: build with `--features test-panic` only when `FAST_MESH_TEST_BUILD=1`, into the same output. Tests use it, and release builds never include it.
  - `packages/fast-mesh-wasm/package.json`: `test` script builds the test variant first, or CI sets the env; builder's choice, documented.

**Interfaces (produces), via wasm-bindgen exports:**
- `WasmScene::new()`, `free` (wasm-bindgen generates it).
- **Meshes:**
  - `add_mesh(positions_f64: &[f64], indices: &[u32], origin: Option<Vec<f64>>, mode: u8, normals: bool) -> Vec<u32> /*[index, gen]*/`
  - `remove_mesh(i, g)`
  - `set_placement(i, g, rot9: &[f64], t3: &[f64])`, `placement(i, g) -> Vec<f64> /*12*/`, `matrix4(i, g) -> Vec<f64> /*16*/`
  - `set_mode`, `update`, `flush`
  - `vertex_slots`, `triangle_slots`, `vertex_count`, `triangle_count`
- **View pointers:** `positions_ptr/len`, `normals_ptr/len` (0 when off), `colors_ptr/len` (0 when off), `indices_ptr/len`. Lengths are in elements of the TS array type.
- **Changes:** `take_changes() -> Vec<u32>` (Task 2 encoding).
- **History:** `undo() -> bool`, `redo() -> bool`, `can_undo`, `can_redo`, `clear_history`, `begin_step`, `end_step`, `cancel_step`.
- **Errors:** every fallible method returns `Result<_, JsError>` with the `"<CODE>: text"` message from `error.rs` (spec §4.4 table).

**TS (minimal, for testing this task):**
- `packages/fast-mesh-wasm/src/core.ts`:
  - `FastMeshError`
  - `call(fn)`, which maps a `JsError` to `FastMeshError` and a `RuntimeError` to poisoning
  - the poisoned flag, which existing `computeVertexNormals` / `computeBounds` also respect
- `markInitialized` stores the `InitOutput.memory` from `initSync` / `wasmInit` for the views (Task 4).

- [ ] **Step 1: Write failing tests:**
  - Rust (`wasm-bindgen` classes are plain structs in native tests): `add_place_matrix4`, `views_ptr_len_match_slots`, `error_codes_map` (one per code reachable from these methods), `take_changes_after_edit_via_scene` (use internal Scene access).
  - vitest `packages/fast-mesh-wasm/test/core.test.ts`:
    - `panic_poisons_instance`
    - `error_has_code` (removeMesh twice gives `UNKNOWN_MESH`)
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full gate, including `pnpm build`, the package tests and `pnpm check:pkg`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(wasm): WasmScene bindings with views, changes, history and error codes`.

### Task 4: TypeScript `Scene` core: handles, meshes, placements, views, changes, frames, drags

**Files:**
- Create: `packages/fast-mesh-wasm/src/scene.ts`, `packages/fast-mesh-wasm/src/changes.ts` (decoder), `packages/fast-mesh-wasm/src/views.ts` (view builders and debug guard).
- Modify: `packages/fast-mesh-wasm/src/web.ts` and `node.ts` (export the new API), `api.ts` (shared types).
- Test: `packages/fast-mesh-wasm/test/scene.test.ts`

**Interfaces (produces):**
```ts
export class FastMeshError extends Error { readonly code: FastMeshErrorCode }
export class Scene {
  constructor(options?: { debug?: boolean });
  free(): void; [Symbol.dispose](): void;
  addMesh(positions: Positions64, indices: Indices, options?: { origin?: Vec3; mode?: UpdateMode; normals?: boolean }): MeshHandle;
  removeMesh(mesh: MeshHandle): void;
  setPlacement(mesh: MeshHandle, p: { rotation: Mat3 | Quat; translation: Vec3 }): void;
  placement(mesh: MeshHandle): { rotation: Mat3; translation: Vec3 };
  matrix4(mesh: MeshHandle): Float64Array;
  setMode(mesh, mode): void; update(mesh): void; flush(): void;
  vertexSlots(mesh): number; triangleSlots(mesh): number; vertexCount(mesh): number; triangleCount(mesh): number;
  views(mesh: MeshHandle): MeshViews; // { positions(), normals(), colors(), indices() } — fresh views per call
  takeChanges(): MeshChange[];
  onChanges(listener: (changes: MeshChange[]) => void): () => void; // returns unsubscribe
  commitFrame(): MeshChange[]; // takeChanges once, deliver to listeners, return
  undo(): boolean; redo(): boolean; canUndo(): boolean; canRedo(): boolean; clearHistory(): void;
  beginStep(): void; endStep(): void; cancelStep(): void;
  drag(mesh: MeshHandle): DragSession; // move(p), end(), cancel()
}
export type Positions64 = Float32Array | Float64Array | ArrayLike<number>;
export type Quat = readonly [x: number, y: number, z: number, w: number];
export type Mat3 = readonly number[]; // 9, row-major
export interface MeshChange { mesh: MeshHandle; added: boolean; removed: boolean; placementChanged: boolean; capacityChanged: boolean;
  positions: Range[] | "all"; normals: Range[] | "all"; colors: Range[] | "all"; indices: Range[] | "all";
  fieldsChanged: FieldHandle[]; sectionsChanged: SectionHandle[] }
export type Range = readonly [start: number, count: number];
```
- **`Quat` → matrix:** convert in f64 and normalise first. A zero quaternion gives `INVALID_ARGUMENT`.
- **`DragSession`:**
  - `move` calls `setPlacement` inside a step opened by `drag()`.
  - `end()` calls `endStep()`.
  - `cancel()` restores the start placement, then ends the step, so the step records nothing.
  - `undo()` while a drag is active calls `end()` first, then undoes. That needs a scene-level active-drag flag.
- **Debug guard:** when `debug` is on, view getters record the `ArrayBuffer`. Later getters check `byteLength === 0` on previously returned arrays they still track (a WeakRef set) and `console.warn` once. Builder's choice of mechanism, as long as `views_survive_memory_growth` shows the warning.

- [ ] **Step 1: Write failing tests** in `scene.test.ts`:
  - `add_mesh_world_positions_and_origin`: `views().positions()` holds `p − origin`, and `matrix4` maps it back within 1e-6 (tested with a mesh near 1e5).
  - `quaternion_and_matrix_placement`
  - `views_match_engine`: indices equal the input, normals equal `computeVertexNormals`, colours are `null` with colouring off.
  - `views_survive_memory_growth` (Review Focus 1)
  - `stale_handle_after_slot_reuse` (Review Focus 2)
  - `changes_ranges_after_placement_and_drag`: a placement change gives `placementChanged` with empty position ranges.
  - `commit_frame_fans_out_once`: two listeners get the same array, and a second `commitFrame` returns empty.
  - `drag_is_one_undo_step`
  - `undo_mid_drag` (Review Focus 3)
  - `free_then_use_throws`, `using_disposes`
- [ ] **Step 2: Run** `pnpm build && pnpm --filter fast-mesh-wasm test`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** `feat(js): Scene core with handles, views, changes, frames and drags`.

### Task 5: Queries, fields/colours and sections (bindings + TS)

**Files:**
- Modify: `crates/fast-mesh-wasm/src/scene.rs`, `packages/fast-mesh-wasm/src/scene.ts`
- Test: `packages/fast-mesh-wasm/test/features.test.ts`, plus Rust unit tests in `scene.rs`

**Interfaces (produces, TS):**
```ts
raycast(origin: Vec3, direction: Vec3, options?: { meshes?: MeshHandle[]; maxDistance?: number }): { mesh: MeshHandle; triangle: number; point: Vec3; distance: number; barycentric: Vec3 } | null;
closestPoint(point: Vec3, options?: { meshes?: MeshHandle[]; maxDistance?: number }): { mesh: MeshHandle; triangle: number; point: Vec3; distance: number } | null;
meshDistance(a: MeshHandle, b: MeshHandle): { distance: number; a: Vec3; b: Vec3 };
intersects(a: MeshHandle, b: MeshHandle): boolean;
findContacts(options?: { margin?: number }): { a: MeshHandle; b: MeshHandle; distance: number }[];
addDistanceField(d: { name: string; on: MeshHandle; to: MeshHandle; maxDistance: number; sign?: "auto" | "pseudonormal" | "winding" }): FieldHandle;
removeField(f: FieldHandle): void; fieldValues(f: FieldHandle): Float32Array; fieldStatus(f: FieldHandle): FieldStatus;
setColoring(mesh: MeshHandle, rule: ColorRule | null): void;
addSection(d: { plane: { origin: Vec3; normal: Vec3 }; meshes: MeshHandle[]; fill?: boolean }): SectionHandle;
setSectionPlane(s, plane): void; setSectionMeshes(s, meshes): void; setSectionFill(s, fill): void; removeSection(s): void; updateSections(): void;
section(s: SectionHandle, mesh: MeshHandle): { status: "current" | "stale"; contours: { points: Float32Array; closed: boolean }[]; length: number; area: number; fill?: { positions: Float32Array; indices: Uint32Array } };
sectionGap(s: SectionHandle, a: MeshHandle, b: MeshHandle): { distance: number; a: Vec3; b: Vec3 } | null;
```
- **Rust options:** check the exact option fields of the Rust APIs (`fast-mesh-scene` `query.rs`, `contacts.rs`, `fields.rs`, `sections.rs`) and mirror them. Where a TS option above has no Rust counterpart, drop it and say so in the report.
- **`ColorRule`:** mirrors the Rust `ColorRule` (`map: "turbo" | "viridis" | "coolwarm" | { stops: [t, [r,g,b,a]][] }`, `range`, `below`, `above`, `center`, `base`, `nan`), with field names in camelCase.
- **Sections:** results are copies (world xyz in f32).

- [ ] **Step 1: Write failing tests:**
  - `raycast_picks_nearest_mesh` (two spheres)
  - `distance_coloring_toggle`:
    - colours are non-null and non-uniform after `setColoring`;
    - a `takeChanges` reports colour ranges;
    - `setColoring(null)` makes `colors()` return null and leaves history unchanged (`canUndo` as before).
  - `drag_updates_colors_incrementally`: dragging B changes A's colour ranges, which cover fewer than all slots for a small move.
  - `section_contours_and_gap`: two spheres cut by `z=0` give a gap of ≈ 1 within 1e-3.
  - `error_codes_unknown_field_section`
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** `feat(js): queries, distance fields, colouring and sections`.

### Task 6: Edits and repair (bindings + TS)

**Files:**
- Modify: `crates/fast-mesh-wasm/src/scene.rs`, `packages/fast-mesh-wasm/src/scene.ts`
- Create: `crates/fast-mesh-wasm/src/edit.rs` (op-list decoder and runner)
- Test: `packages/fast-mesh-wasm/test/edit.test.ts`, plus Rust unit tests

**Interfaces (produces, TS):**
```ts
type EditOp =
  | { op: "setPositions"; vertices: Uint32Array | number[]; positions: Positions64 }      // world
  | { op: "addVertices"; positions: Positions64 }                                       // world
  | { op: "addTriangles"; triangles: (number | { ref: [opIndex: number, i: number] })[] | Uint32Array }
  | { op: "removeTriangles"; triangles: Uint32Array | number[] }
  | { op: "removeVertices"; vertices: Uint32Array | number[] | { ref: [number, number] }[] };
edit(mesh: MeshHandle, ops: EditOp[]): { added: Uint32Array[] };  // per op: new ids (empty for non-adding ops)
compact(mesh: MeshHandle): void;
analyze(mesh: MeshHandle, options?: { weldTolerance?: number; selfIntersection?: boolean; maxListed?: number }): Analysis;
repair(mesh: MeshHandle, op: RepairOp | RepairOp[]): RepairReport | RepairReport[];
type RepairOp =
  | { op: "weld"; tolerance: number }
  | { op: "removeDegenerate"; areaEpsilon?: number; removeUnusedVertices?: boolean }
  | { op: "removeSmallComponents"; minTriangles?: number; minArea?: number; keepLargest?: boolean }
  | { op: "orient" }
  | { op: "fillHoles"; maxEdges?: number; maxPerimeter?: number; maxArea?: number; loops?: LoopId[]; allowUnclean?: boolean; fair?: boolean; dpLimit?: number };
type LoopId = { minVertex: number; len: number; hash: bigint };
```
- **Encoding:** the op list goes to wasm as one encoded buffer. The builder picks the format, e.g. a `Uint32Array` header plus an `f64` side array. One wasm call per `edit`.
- **Atomicity:** the list runs inside one `Scene::edit`, so it is atomic and one undo step.
- **World → local:** world positions are converted to local inside Rust with the placement inverse and origin.
- **Refs:** they refer to ids produced by earlier ops in the same list. An out-of-range ref gives `INVALID_INDEX` before anything is written.
- **`repair` with an array:** runs every op in one `Scene::edit`, so the whole array is one undo step.
- **Reports:** plain objects mirroring the Rust report structs in camelCase. `LoopId.hash` is a `bigint`.

- [ ] **Step 1: Write failing tests:**
  - `edit_ops_atomic_with_refs` (Review Focus 4)
  - `edit_is_one_undo_step`
  - `set_positions_world_space` on a placed and rotated mesh: positions land where expected in local views, and normals update.
  - `compact_reports_all_and_capacity`
  - `repair_pipeline_one_undo_step`: a hemisphere with holes and flips, repaired with `[orient, fillHoles({maxPerimeter})]`, gives 0 conflicts in `analyze`, and the rim is kept.
  - `stale_loops_error_code`
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** `feat(js): edit op lists, compaction and repair`.

### Task 7: Benchmarks, size, docs, changeset, CI

**Files:**
- Create:
  - `packages/fast-mesh-wasm/bench/scene.bench.ts`: a vitest `bench` or a plain Node script, builder's choice, run with `pnpm --filter fast-mesh-wasm bench`.
  - `.changeset/<name>.md`: minor bump of `fast-mesh-wasm`, describing the Scene API.
- Modify:
  - `docs/benchmarks.md`: new "JavaScript core" section with wasm size (raw and gzip, before and after this phase).
  - `packages/fast-mesh-wasm/README.md`, plus the root `README.md` (API overview with a short example of the owner's scene: add meshes, drag, colouring toggle, undo).
  - `.github/workflows/ci.yml`: run the package tests against the test-panic build if that needs a separate step.
  - The spec: add "§12 Implementation notes", covering rulings and measured numbers.

**Benches** (Node, real numbers on the i5-13600K):
- **Large-scale (general library):**
  - `addMesh` of a 1M-triangle mesh from `Float64Array`
  - a 64-pose drag of one 1M mesh against another with colouring on: per-pose `setPlacement` + `commitFrame()` + reading every dirty range from the views into a JS mirror (simulating an adapter upload)
  - the same with 10M if practical
- **Owner-realistic scene** (in addition, never instead): 6 meshes of 30k triangles with distance colouring between every pair (30 fields), dragging one. Report per-frame ms.

- [ ] **Step 1: Write** the benches.
- [ ] **Step 2: Run** them and record the numbers. Never invent numbers.
- [ ] **Step 3: Write** the docs, the changeset and the CI change.
- [ ] **Step 4: Verify** the full gate. Expected: green.
- [ ] **Step 5: Commit** `bench(js): scene benchmarks`, `docs(js): document the Scene API and record benchmarks`, `chore: changeset for the Scene API`.
