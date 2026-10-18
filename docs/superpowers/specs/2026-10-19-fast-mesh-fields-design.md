# fast-mesh fields and colors, part a (Phase 5a) design

- **Status:** Approved for planning (decisions after the builder check in §11 override earlier sections)
- **Date:** 2026-10-19
- **Parent specs:**
  - `2026-10-14-fast-mesh-engine-design.md` §5.4 and §8;
  - `2026-10-18-fast-mesh-scene-design.md`, including §16 "Open questions for Phase 5".

  Where this document disagrees with them, it wins for Phase 5a.
- **Milestone:** Phase 5: Fields & colors

## 1. Intent

The owner shows how close two scanned meshes are, and how deeply they overlap, as a smooth per-vertex colour. One value per vertex is the **signed distance to the other mesh**:

- positive is the gap;
- zero means touching;
- negative is the penetration depth.

The value is mapped to colours through a colormap and a range the user picks. Two ranges in the owner's words:

- "from contact": only values ≤ 0 are coloured;
- "from some distance": for example, from 0.5 inward, through contact, into penetration.

Outside the range the vertex keeps a base colour.

- **One or both meshes.** Colouring can be on for one mesh only or for both. Each coloured mesh is measured against the other.
- **Open or closed scans.** Scans may be closed or open, and an open scan may or may not be capped on purpose. Sign modes cover both cases (§5).
- **Speed.** It must stay interactive while one mesh is dragged. Only vertices near the contact are recomputed.

Phase 5 is split in two:
- **5a (this document):**
  - a small field framework;
  - the signed distance field with the `pseudonormal`, `winding` and `auto` sign modes;
  - colormaps and colour buffers;
  - Scene integration, including the three §16 questions from Phase 4.
- **5b (later):**
  - the virtual `cap` sign mode, built together with Phase 7 hole filling, which needs the same min-area triangulation;
  - the built-in normals, curvature and thickness fields;
  - user-defined field kernels;
  - an optional sampled SDF grid as a faster approximation for rigid drags. The owner asked about distance fields; *my call:* exact BVH distances first, the grid as an opt-in accelerator later.

## 2. Goals, non-goals, anti-goals

**Goals:**
1. A per-vertex signed distance field from mesh A to mesh B, under the meshes' current placements. It has a `max_distance` beyond which values are "far", and the sign modes `pseudonormal`, `winding` and `auto`.
2. Incremental recomputation after B moves, after B is edited, and after A is edited, as described in §6.
3. Colormaps with a range and out-of-range behaviour, writing an RGBA8 colour buffer per coloured mesh.
4. Scene integration:
   - change reports with several consumers;
   - public local-space accessors;
   - render-dirty reporting for colours and derived channels.

**Non-goals (5b or later):** `cap` mode, other built-in fields, custom kernels, SDF grids, JS bindings (Phase 8), and colour transfer to three.js (Phase 8).

**Anti-goals:**
- Recomputing every vertex of a 1M mesh on each drag frame when the contact area is small.
- A sign that flips randomly near open boundaries without warning. `auto` must choose `winding` for open meshes.
- Making field data part of undo. It is derived, and it is recomputed after undo and redo like everything else.

## 3. Constraints

- **New crate `fast-mesh-fields`:** pure kernels, depending on `fast-mesh-store`, `fast-mesh-bvh` and `fast-mesh-query`.
  - Field registration and update scheduling live in `fast-mesh-scene`, which then depends on `fast-mesh-fields`. This matches the parent's dependency table.
  - No new third-party dependencies.
- **Conventions:**
  - mesh-local f32 inside the kernels;
  - f64 for transforms;
  - failure-atomic operations;
  - no allocation inside query traversals;
  - deterministic results.
- **Docs and checks:** `///` docs, clippy `-D warnings`, the wasm32 build, property tests against brute force, and criterion benches.

## 4. Kernels (`fast-mesh-fields`)

```rust
pub enum SignMode { Pseudonormal, Winding, Auto }

/// Per-mesh data for signs, built from a mesh (and its BVH for winding numbers).
pub struct SignData { /* pseudonormals and/or winding moments */ }
impl SignData {
    pub fn build(mesh: &Mesh, bvh: &Bvh, mode: SignMode) -> Result<SignData>;   // Auto resolves here
    pub fn mode(&self) -> SignMode;                                                // the resolved mode
    pub fn is_current(&self, mesh: &Mesh, bvh: &Bvh) -> bool;
}

/// Signed distances from `points` (in B's local space) to mesh B, written to `out`.
/// Points farther than `max_distance` get `f32::INFINITY` (see §5.3).
pub fn signed_distances(mesh_b: &Mesh, bvh_b: &Bvh, sign: &SignData, points: &[[f32; 3]], max_distance: f32, out: &mut [f32]) -> Result<()>;
```

- **Unsigned part:** Phase 3a's `closest_point`, with `max_distance` and its `Feature` (vertex, edge or face) result.
- **`Pseudonormal`** (Bærentzen–Aanæs) uses the angle-weighted pseudonormal at the closest feature:
  - a face uses its normal;
  - an edge uses the sum of its adjacent face normals;
  - a vertex uses the angle-weighted sum of its incident face normals.

  The sign is `sign(dot(p − closest, n))`, so the sign is exact for closed manifold meshes. The data is per-vertex and per-edge normals, built from `Mesh::adjacency` and `incidence` in O(n).
- **`Winding`** is the fast generalized winding number of Barill et al. 2018.
  - Each BVH node stores order-1 dipole moments: area-weighted normal sum and area-weighted centroid.
  - A point is inside when the winding number is ≥ 0.5.
  - Far nodes use the dipole approximation, controlled by the accuracy parameter `beta = 2.0` (*my call*). Near nodes are summed exactly over their triangles' solid angles.
  - Moments are indexed by BVH node. This needs a dense node index: *my call*, add `NodeId::index(self) -> usize` and `Bvh::node_slots(&self) -> usize` to `fast-mesh-bvh`.
  - The moments are rebuilt in O(n) whenever B's revision or epoch changes. *My call:* refitting them along dirty paths, as in parent §5.4, is a 5b optimisation.
- **`Auto`** picks `Pseudonormal` when B is closed and manifold (no boundary or non-manifold edges in `Adjacency`), and `Winding` otherwise.

## 5. Field values

### 5.1 Storage
- The distance field of mesh A is stored as a store derived channel on A, named `field:<name>`, width 1. A derived channel resizes with A's vertex slots, survives `compact` remapping, and is not undoable. It is read through `Mesh::channel`.
- Dead vertex slots hold NaN.

### 5.2 Exactness
Values are the f32 result of the f64 relative-transform conversion followed by Phase 3a's f32 closest point. They are deterministic. Ties in the closest feature follow Phase 3a.

### 5.3 Far values
- A vertex farther than `max_distance` from B's surface holds `+∞`, meaning far, with no sign.
- *My call:* a vertex deeper inside B than `max_distance` is also reported as `+∞`. Finding it would need a full-mesh scan. This is documented, and `max_distance` should exceed the deepest expected penetration.
- An infinite `max_distance` computes every vertex.

## 6. Scene integration

### 6.1 Registering a distance field
```rust
pub struct DistanceField { pub name: String, pub on: MeshId, pub to: MeshId, pub max_distance: f64, pub sign: SignMode }
pub struct FieldId(..);
impl Scene {
    pub fn add_distance_field(&mut self, desc: DistanceField) -> Result<FieldId>;
    pub fn remove_field(&mut self, id: FieldId) -> Result<()>;
    pub fn update_fields(&mut self) -> Result<()>;          // also run by flush()
    pub fn field_values(&mut self, id: FieldId) -> Result<&[f32]>;   // updates first if stale
}
```

**When fields update:**
- Fields are recomputed by `update_fields`, by `flush`, or before `field_values` and colour reads.
- *My call:* field updates follow the `UpdateMode` of mesh `on`. Eager updates after every change that affects the field, Lazy waits, and Manual waits for `update_fields`.

**What triggers a recompute, and which vertices of A:**
- `max_distance` is finite.
  - **A placement change** of either mesh, or **an edit of B**: the vertices of A's triangles from `triangles_near(A, B, max_distance)`, plus the vertices that held finite values before (so they can return to far).
  - **An edit of A:** the added and moved vertices of A, plus the near candidates above if A's triangles changed.
- `max_distance` is infinite: every vertex of A.
- **Lifecycle:** removing either mesh removes the field. Compacting a mesh keeps the field, which is remapped through the derived channel.

### 6.2 Colours
```rust
pub enum Colormap { Turbo, Viridis, Coolwarm, Grayscale, Custom(Vec<(f32, [u8; 4])>) }
pub struct Coloring { pub field: FieldId, pub map: Colormap, pub range: (f32, f32), pub out_of_range: OutOfRange, pub base: [u8; 4], pub nan: [u8; 4] }
pub enum OutOfRange { Clamp, Base }
impl Scene {
    pub fn set_coloring(&mut self, mesh: MeshId, coloring: Option<Coloring>) -> Result<()>;
    pub fn colors(&mut self, mesh: MeshId) -> Result<Option<&[[u8; 4]]>>;   // one RGBA8 per vertex slot
}
```

- **Value → colour.** A value `v` in `range` is mapped linearly onto the colormap. Outside `range`, `Clamp` uses the end colours and `Base` uses `base`. Far (`+∞`) uses `base`. NaN uses `nan`.
- **The owner's two ranges:**
  - "from contact": `range = (-d, 0)`, out of range `Base`;
  - "from distance": `range = (-d, g)`.
- **Storage.** Colours live in a per-mesh `Vec<[u8; 4]>` inside the Scene, sized to the vertex slots.
  - *My call:* not a store channel, because store channels are f32 and renderers want RGBA8 directly.
  - Only vertices whose field value changed are recoloured.
- *My call:* the colormap tables are built-in arrays of 256 entries, matching the published colormaps.

### 6.3 Answers to the Phase 4 §16 questions
1. **Several consumers of change reports.**
   - `subscribe() -> SubscriberId` and `unsubscribe(id)`, plus `take_changes_for(id) -> Vec<MeshChanges>`.
   - Each subscriber has its own pending reports, merged the same way.
   - `take_changes()` keeps working as the default subscriber.
   - Field updates use an internal hook, not a subscriber.
2. **Local-space accessors.**
   - `bvh(id) -> Result<&Bvh>` returns `StaleBvh` when the BVH is not current.
   - `relative_transform(a, b) -> Result<Rigid>`.
   - `world_to_local(id, p)` and `local_to_world(id, p)`.
3. **Render-dirty signal.** `MeshChanges` gains these fields, all reported to every subscriber:
   - `colors_dirty: Option<Vec<VertexId>>`, where `None` means all;
   - `fields_changed: Vec<FieldId>`;
   - `channels_written: Vec<String>`, for writes through `derived_channel_mut` and for channels added or removed.

## 7. Errors
- New `SceneError` variants:
  - `UnknownField`;
  - `InvalidInput` for a bad range, a negative or NaN `max_distance`, `on == to`, or a duplicate field name on the same mesh.
- `fast-mesh-fields` has its own error enum, wrapped into `SceneError::Fields`.

## 8. Testing

**Kernels against brute force:**
- Signed distance compared with a brute-force closest point plus an exact sign. The sign comes from the exact generalized winding number, a sum of all triangles' solid angles in f64, run on small meshes:
  - closed meshes: a cube, a subdivided sphere and a torus;
  - open meshes: a cube with a missing face, and a terrain sheet.
- **Pseudonormal mode** on closed meshes must equal the exact-winding sign everywhere except within 1e-6 of the surface.
- **Fast winding** must equal the exact winding number within 1e-3 for points beyond 1% of the bounding-box diagonal from the surface.

**Scene, model-based property tests:** random placements, drags, edits, undo/redo, `max_distance` and sign modes on two small meshes. After `update_fields`:
- every vertex value equals a from-scratch computation;
- colours equal the colormap applied to the values;
- `colors_dirty` covers every vertex whose colour changed.

**Incremental behaviour:** a counter shows that a small drag recomputes only near vertices.

## 9. Benchmarks
- Two 1M terrains touching, with the 64-pose drag from Phase 3b and the field on one side and on both sides. Measure `set_placement` + `update_fields` + colouring per pose.
  - *My call:* the target is a few ms per pose when the contact covers a few thousand vertices. This is not a merge gate.
- `SignData::build` time at 1M (pseudonormal and winding).

## 10. Left to the builder
- Module layout.
- The candidate-set data structures.
- Exact colormap table values (from the published sources).
- The beta parameter tuning, within the accuracy test.

## 11. Decisions after the builder check (all *my call*; they override earlier sections)

1. **Sign for open scans.**
   - The winding number never reaches 0.5 behind an open sheet, so `winding` cannot show penetration on open scans.
   - `Auto` therefore resolves to `Pseudonormal` (the side of the surface the point lies on, from the oriented normals) for closed and open meshes alike. For open scans, "inside" means behind the surface, which is what scan users expect. Boundary edges use their single face normal.
   - `Winding` stays available on request, for closed but messy meshes (self-intersections).
   - `Auto` falls back to `Winding` only when the mesh is closed but its orientation is inconsistent, a case where pseudonormals give wrong signs everywhere.
   - `SignData::status()` returns `SignStatus { mode, closed, consistent_orientation, non_manifold_vertices, degenerate_triangles }`, and `Scene::field_status(id)` exposes it, so an application can warn.
2. **Modes and staleness.**
   - A field update brings the BVHs it needs up to date with the query rules from Phase 4 §15.4: Lazy BVHs are updated, and a stale Manual BVH makes the field stale.
   - The field follows mesh `on`'s mode: Eager fields update after each relevant change.
   - A failed or blocked Eager field update is swallowed (the field is marked stale), like a failed Eager BVH update.
   - `update_fields()` returns the first error, for example `StaleBvh(id)`.
   - Readers are `&self` and never update: `field_values(&self, id) -> Result<&[f32]>`, `colors(&self, mesh) -> Result<Option<&[[u8; 4]]>>`, `field_status(&self, id) -> Result<FieldStatus { current: bool, sign: SignStatus }>`.
   - The pattern for renderers is to call `update_fields()` or `flush()`, then read any number of fields and colours together.
3. **Colouring.**
   - `Coloring { field, map, range: (f32, f32), below: OutOfRange, above: OutOfRange, center: Option<f32>, base, nan }`.
   - `center` pins one value, usually 0 (contact), to the middle of the colormap. The mapping is piecewise linear on each side.
   - The owner's ranges:
     - "from contact": `range = (-d, 0)`, `below = Clamp`, `above = Base`;
     - "from distance": `range = (-d, g)`, `below = Clamp`, `above = Base`, `center = Some(0.0)`.
4. **Reserved names.** The `field:` channel prefix is reserved. `add_channel`, `remove_channel` and `derived_channel_mut` reject it with `InvalidInput`.
5. **`relative_transform(a, b)`** maps B's local space into A's local space, the same convention as Phase 3b `Rigid` and `triangles_near`. Field code uses its inverse for A's points.
6. **Candidate rule, corrected.**
   - Which vertices carry values:
     - Values are defined for alive vertices used by at least one alive triangle.
     - Unreferenced alive vertices and dead slots hold NaN.
   - What is recomputed after a change:
     - All vertices, when the report is `None` (after `compact`, `add_mesh` or a `Corrupted` step), when `max_distance` is infinite, or when the field is first registered.
     - Otherwise:
       - the vertices of A's triangles from `triangles_near(A, B, pad(max_distance))`, with `pad(x) = x·(1 + 1e-6) + 1e-6`, so that the f64 pair test never drops a vertex the f32 kernel would keep;
       - plus the vertices that held finite values before;
       - plus A's added, moved and removed vertices;
       - plus the vertices whose corner count crosses zero in either direction.
     - Removed vertices are set to NaN.
   - These rules rely on undo reporting revived slots as added, and that reliance is documented.
7. **`SignData::build(&mut Mesh, &Bvh, mode)`** builds adjacency and incidence when needed.
   - Pseudonormals skip zero-area triangles when they accumulate normals.
   - When `closest_point` reports a zero-area triangle, the sign is taken from the pseudonormal of the nearest non-degenerate feature among that vertex's or edge's incident triangles. The builder may refine this, and it is tested on meshes with degenerate faces.
   - Mapping from `Feature`: `Feature::Edge(k)` joins corners k and (k+1)%3, while `Adjacency` indexes the edge opposite each corner, so edge normals are stored per corner and converted carefully.
8. **BVH generation.** `fast-mesh-bvh` gains `Bvh::generation() -> u64`, incremented by every structural change: build, update, refit, `rebuild_region` and rebuild. It sits alongside `NodeId::index()` and `Bvh::node_slots()`. `SignData::is_current` checks the mesh uid, epoch and revision, plus the BVH generation.
9. **Tests.** Surface tolerances are relative to the bounding-box diagonal (1e-5·diag), not absolute.
10. **Compaction.** It reports `None`, so the field, the colours and the "finite before" set are recomputed in full. No remap is needed.
