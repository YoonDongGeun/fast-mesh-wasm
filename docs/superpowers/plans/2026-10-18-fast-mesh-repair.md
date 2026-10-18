# fast-mesh Repair (Phase 7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mesh repair operations in a new crate `fast-mesh-ops`:
- `analyze`
- `orient`
- `fill_holes`, with refine and fairing
- `remove_small_components`
- `weld`
- `remove_degenerate`

All of them write through transactions. The Scene can run them through `scene.edit`.

**Architecture:**
- `fast-mesh-ops` depends on `fast-mesh-store` and `fast-mesh-query`. Each operation:
  1. builds its own edge, component and loop maps from `tx.mesh()`;
  2. plans and validates;
  3. writes only after that.
- `fast-mesh-query` gains `self_intersecting_pairs`.
- `fast-mesh-scene` depends on `fast-mesh-ops`. `Scene::edit` becomes generic over the closure's error (`E: Into<SceneError>`), and `SceneError` gains `Ops(OpsError)`. Existing callers that return `fast_mesh_store::Result` keep compiling.

**Tech Stack:** Rust 1.99 (MSRV 1.88, edition 2024), proptest 1.12, criterion 0.8. No new third-party dependencies.

**Spec:** `docs/superpowers/specs/2026-10-18-fast-mesh-repair-design.md`. §11 overrides earlier sections.

## Global Constraints
- **Code:** no `unsafe`; no new third-party crates; growth via `try_reserve`, failing with `OutOfMemory` (`OpsError::Store(StoreError::OutOfMemory)`).
- **Atomic per call:** compute the whole plan from `tx.mesh()`, validate it, then write. Per-element failures (an unfillable loop, a non-orientable patch) are reported and leave that element untouched. No partial patch is ever written.
- **Determinism:** no hash-iteration-order dependence and no randomness. Sort keys explicitly.
- **Geometry:** in `f64`, from the stored `f32` local positions, origin excluded.
- **Edges and degenerates:** an edge key is `(min, max)`. Boundary = 1 triangle, interior = 2, non-manifold = 3 or more. Triangles with a repeated vertex id are skipped by every operation except `analyze` and `remove_degenerate`.
- **Components:** triangles connect through shared edges of any kind.
- **`LoopId`:** `{ min_vertex: VertexId, len: u32, hash: u64 }`. The hash covers the vertex sequence rotated to start at `min_vertex`, in loop direction. Use a fixed FNV-1a 64 over the `u32` ids, not `std`'s `RandomState`.
- **Loops at pinch vertices:**
  - A pinch vertex is one whose fan splits into several pieces. It is reported in `pinch_vertices`. Loops through it are `clean` when every fan piece is edge-manifold.
  - `clean: false` only when the vertex touches a non-manifold edge.
  - `fill_holes` skips unclean loops unless `allow_unclean`.
- **`fill_holes` selection:**
  - A loop is filled iff it satisfies every bound that is set: `max_edges`, `max_perimeter`, `max_area`.
  - An explicit `loops: Some(ids)` fills exactly those ids. A loop that no longer matches makes the whole call fail with `OpsError::StaleLoops`, before anything is written.
  - No bound and no `loops` → `OpsError::InvalidInput`.
  - Defaults: `dp_limit = 200`, `fair = true`, `allow_unclean = false`.
- **`remove_small_components`:**
  - A component is removed when it is below any bound that is set (`min_triangles`, `min_area`).
  - `keep_largest` defaults to true. "Largest" means the most area; ties go to the most triangles, then the smallest triangle id.
  - No bound → `InvalidInput`.
- **`remove_degenerate`:**
  - Removes triangles with a repeated id, triangles with area ≤ `area_epsilon · L²` (L is the longest edge, default `area_epsilon = 1e-12`), and duplicate vertex sets (the smallest id is kept).
  - `remove_unused_vertices` defaults to true.
- **`weld`:** a uniform grid plus union-find. Each group keeps its smallest id at its original position. Triangles that collapse are removed, and so are unused vertices. `tolerance = 0` merges exact duplicates.
- **`orient`:**
  - Patches grow across interior edges only, and each patch takes its own majority by area. Ties go to the patch containing the smallest triangle id.
  - A non-orientable patch is reported and left unchanged.
  - Only closed components with no non-manifold edges are then flipped outward (signed volume > 0).
  - A flip swaps the triangle's last two vertices.
- **Creation parents:** an edge split uses `[(a,0.5),(b,0.5)]`; a centroid split uses thirds. Fairing only moves vertices.
- **Commits:** conventional messages ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or the implementing model). Each message goes in a NEW temp file and is committed with `git commit -F`. Files are written UTF-8 with LF via Edit/Write.
- **Every task ends green:** `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.

## Review Focus
1. **Two holes sharing one corner vertex (the owner's bug).** They give two clean loops, both filled by default, and the result is closed. Tests: Task 1 `bowtie_holes_are_two_clean_loops`, Task 5 `bowtie_holes_both_filled`.
2. **The intended border of an open mesh.** The hemisphere rim must survive `fill_holes` whenever `max_perimeter` is below the rim length. Test: Task 5 `hemisphere_rim_kept_by_perimeter_bound`.
3. **Pipeline inside one transaction with stale loop ids.** Ids taken before a `weld` that changes a loop give `StaleLoops`, and nothing is written. Test: Task 5 `stale_loop_id_after_weld_in_same_tx_is_rejected`.
4. **A hole whose patch would duplicate an existing edge.** A mesh edge already joins two non-adjacent loop vertices (for example a thin bridge). The fill must not create a non-manifold edge: such a diagonal is forbidden in the DP. Test: Task 5 `fill_avoids_existing_edges`.
5. **Fairing on a long, thin, or nearly flat hole.** No NaN positions, the solver iteration cap is respected, and `converged` is reported honestly. Test: Task 6 `fairing_thin_hole_no_nan`.

---

### Task 1: Crate scaffold, maps, components and boundary loops

**Files:**
- Create:
  - `crates/fast-mesh-ops/Cargo.toml` (workspace package fields; deps `fast-mesh-store`, `fast-mesh-query` workspace; dev-deps proptest and criterion as in the query crate)
  - `crates/fast-mesh-ops/src/lib.rs`, `crates/fast-mesh-ops/src/error.rs`, `crates/fast-mesh-ops/src/topology.rs`, `crates/fast-mesh-ops/src/test_util.rs`
- Modify:
  - the root `Cargo.toml` `[workspace.dependencies]`: add `fast-mesh-ops`
  - `CONTRIBUTING.md` crate list

**Interfaces (produces):**
```rust
pub enum OpsError { Store(StoreError), Query(QueryError), InvalidInput(&'static str), StaleLoops }
impl From<StoreError> for OpsError; impl From<QueryError> for OpsError; Display + Error
pub type Result<T> = std::result::Result<T, OpsError>;

pub enum EdgeKind { Boundary, Interior, NonManifold }
pub(crate) struct EdgeMap { /* sorted by key */ }
impl EdgeMap {
    pub(crate) fn build(mesh: &Mesh) -> Result<EdgeMap>;          // skips degenerate triangles
    pub(crate) fn kind(&self, a: VertexId, b: VertexId) -> Option<EdgeKind>;
    pub(crate) fn triangles(&self, a: VertexId, b: VertexId) -> &[TriangleId];
    pub(crate) fn edges(&self) -> impl Iterator<Item = ((VertexId, VertexId), &[TriangleId])>;
}
pub struct Component { pub triangles: Vec<TriangleId>, pub area: f64, pub closed: bool }
pub(crate) fn components(mesh: &Mesh, edges: &EdgeMap) -> Result<Vec<Component>>; // sorted by smallest triangle id

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoopId { pub min_vertex: VertexId, pub len: u32, pub hash: u64 }
pub struct BoundaryLoop { pub id: LoopId, pub vertices: Vec<VertexId>, pub perimeter: f64, pub area: f64, pub clean: bool }
pub fn boundary_loops(mesh: &Mesh) -> Result<Vec<BoundaryLoop>>;   // public: callers pick ids from here or analyze
pub(crate) fn pinch_vertices(mesh: &Mesh, edges: &EdgeMap) -> Result<Vec<VertexId>>;
```
- Loop direction follows the boundary half-edges, which is the direction of the single triangle's edge `a→b`.
- Loops are sorted by `(min_vertex, then the vertex after min_vertex)`.
- Test fixtures in `test_util.rs`, all `pub(crate)`:
  - `uv_sphere(stacks, slices, r) -> Mesh`
  - `hemisphere(stacks, slices, r) -> Mesh` (open; its rim is one loop)
  - `grid(n) -> Mesh` (flat, z = 0)
  - `bowtie_holes() -> Mesh`: a grid with two adjacent cells' triangles removed so that the two square holes share exactly one corner vertex
  - `fin() -> Mesh`: three quads on one edge
  - `mobius(n) -> Mesh`
  - `punch_holes(mesh, centers, radius) -> Mesh`: removes the triangles whose centroid lies within `radius` of each centre
  - `flip(mesh, tris) -> Mesh`
  - `crossed_boxes() -> Mesh`: two boxes in one mesh that interpenetrate

- [ ] **Step 1: Write failing tests** in `topology.rs`:
  - `edge_kinds_on_fixtures`: a closed sphere has 0 boundary and 0 non-manifold edges; `grid(4)` has 16 boundary edges; `fin()` has 1 non-manifold edge.
  - `components_count_and_closed`: sphere plus a separate small sphere in one mesh gives 2 components, both closed. The hemisphere gives 1, not closed.
  - `hemisphere_has_one_loop`: its perimeter ≈ `2πr` within 1% (at 64 slices).
  - `bowtie_holes_are_two_clean_loops`: the outer border of the grid plus 2 hole loops of 4 edges each, all `clean`. `pinch_vertices` has exactly the shared corner.
  - `fin_loops_unclean`: the loops through the fin's non-manifold edge endpoints are `clean == false`.
  - `loop_id_is_rotation_invariant_and_stable`: the same loop rebuilt after re-adding the mesh (same ids) gives an equal `LoopId`. Two different loops give different hashes.
  - `degenerate_triangles_ignored_by_maps`: appending `[a,a,b]` triangles changes no edge kind and no loop.
- [ ] **Step 2: Run** `cargo test -p fast-mesh-ops`. Expected: FAIL (does not compile).
- [ ] **Step 3: Implement.**
  - **Fan walk at vertex `b`** for an incoming boundary half-edge `a→b` of triangle `t`:
    1. Start at `t`. Take its edge `(b, c)` where `c` is `t`'s third vertex.
    2. While `(b, c)` is interior, cross to the other triangle on `(b, c)` and continue with that triangle's other edge at `b`.
    3. Stop when `(b, c)` is a boundary edge. The next half-edge is `b→c`, provided that boundary triangle uses it as `b→c`.
    4. If you hit a non-manifold edge, or the walk exceeds the number of triangles at `b`, fall back to pairing the remaining outgoing boundary half-edges at `b` in edge-key order, and mark the loop unclean.
  - Use `try_reserve` for every growth.
- [ ] **Step 4: Run** `cargo test -p fast-mesh-ops`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(ops): add fast-mesh-ops with edge maps, components and boundary loops`.

### Task 2: `self_intersecting_pairs` (query) and `analyze`

**Files:**
- Modify: `crates/fast-mesh-query/src/pairs.rs` (or a new `selfpairs.rs`) and `crates/fast-mesh-query/src/lib.rs`
- Create: `crates/fast-mesh-ops/src/analyze.rs`

**Interfaces:**
- Consumes: Task 1 maps and loops; the query crate's `tri_tri_intersect` (crate-private), BVH pair traversal and `check`.
- Produces:
```rust
// fast-mesh-query
pub fn self_intersecting_pairs(mesh: &Mesh, bvh: &Bvh, out: &mut Vec<(TriangleId, TriangleId)>) -> Result<()>;
// pairs (a, b) with a < b that intersect (exact test) and share no vertex id; sorted; skips degenerate triangles

// fast-mesh-ops
pub struct AnalyzeOptions<'a> { pub weld_tolerance: f32, pub self_intersection: Option<&'a Bvh>, pub max_listed: usize } // Default: 0.0, None, 1000
pub struct Analysis {
    pub loops: Vec<BoundaryLoop>,
    pub coincident_groups: usize, pub coincident: Vec<Vec<VertexId>>,       // groups of ≥2
    pub degenerate: usize, pub degenerate_list: Vec<TriangleId>,
    pub duplicate: usize, pub duplicate_list: Vec<TriangleId>,             // all but the smallest id of each set
    pub non_manifold_edges: usize, pub non_manifold_edge_list: Vec<(VertexId, VertexId)>,
    pub non_manifold_vertices: Vec<VertexId>, pub pinch_vertices: Vec<VertexId>,
    pub components: Vec<Component>,
    pub orientation_conflicts: usize,                                       // interior edges used in the same direction by both triangles
    pub self_intersects: Option<bool>,
}
pub fn analyze(mesh: &Mesh, options: &AnalyzeOptions) -> Result<Analysis>;
```
- Counts are always exact. The lists are truncated to `max_listed`.

- [ ] **Step 1: Write failing tests:**
  - query `self_pairs_crossed_boxes_nonempty_and_sphere_empty`.
  - query `self_pairs_skip_vertex_sharing_neighbours`: a closed sphere gives 0 pairs.
  - ops `analyze_counts_on_damaged_fixture`. Build one fixture with known damage: a sphere with 3 punched holes, 5 flipped triangles, 4 duplicated vertices (same position, new ids, one triangle rewired to each), 2 degenerate triangles, 1 duplicate triangle, and a separate floating 2-triangle piece. Every count must equal the known value.
  - ops `analyze_bowtie_reports_pinch_and_two_clean_holes`.
  - ops `analyze_does_not_modify`: `mesh.revision()` is unchanged.
  - ops `analyze_self_intersection_flag`: crossed boxes give `Some(true)`, a sphere gives `Some(false)`, and without a BVH the flag is `None`.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.** Coincident groups use the weld grid. You may write it here as a `pub(crate)` helper in `weld.rs` that Task 3 reuses, or create `spatial.rs`. Builder's choice; record it.
- [ ] **Step 4: Run** `cargo test --workspace`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(query): add self_intersecting_pairs` and `feat(ops): add analyze`.

### Task 3: `weld`, `remove_degenerate`, `remove_small_components`

**Files:**
- Create: `crates/fast-mesh-ops/src/cleanup.rs` (plus `spatial.rs` if Task 2 did not create it)

**Interfaces (produces):**
```rust
pub struct WeldReport { pub merged_vertices: usize, pub removed_triangles: usize }
pub fn weld(tx: &mut Transaction, tolerance: f32) -> Result<WeldReport>;
pub struct DegenerateOptions { pub area_epsilon: f64, pub remove_unused_vertices: bool } // Default 1e-12, true
pub struct DegenerateReport { pub removed_triangles: usize, pub removed_vertices: usize }
pub fn remove_degenerate(tx: &mut Transaction, options: &DegenerateOptions) -> Result<DegenerateReport>;
pub struct ComponentOptions { pub min_triangles: Option<usize>, pub min_area: Option<f64>, pub keep_largest: bool } // Default None, None, true
pub struct RemoveReport { pub removed_components: usize, pub removed_triangles: usize, pub removed_vertices: usize }
pub fn remove_small_components(tx: &mut Transaction, options: &ComponentOptions) -> Result<RemoveReport>;
```
- [ ] **Step 1: Write failing tests:**
  - `weld_merges_duplicates_and_drops_collapsed`: on the damaged fixture, `weld(0.0)` merges 4 vertices, and `analyze` afterwards shows 0 coincident groups.
  - `weld_tolerance_merges_near`: two vertices 1e-5 apart merge with tolerance 1e-4 and stay apart with 1e-6.
  - `degenerate_and_duplicates_removed`.
  - `small_components_removed_largest_kept`: the floating piece is removed with `min_triangles: Some(10)`. With `keep_largest`, a mesh made only of small pieces keeps its largest one.
  - `no_bound_is_invalid_input`.
  - `idempotent`: a second run of each operation changes nothing (an empty patch).
  - `undo_restores` via `Mesh` history (`begin_edit`, commit, undo): positions and triangles are equal to before.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.** Plan first, then write: `set_triangles` for rewiring, `remove_triangles`, then `remove_vertices` for the unused ones.
- [ ] **Step 4: Run** `cargo test --workspace`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(ops): add weld, remove_degenerate and remove_small_components`.

### Task 4: `orient`

**Files:**
- Create: `crates/fast-mesh-ops/src/orient.rs`

**Interfaces (produces):**
```rust
pub struct OrientReport { pub flipped_triangles: usize, pub patches: usize, pub non_orientable_patches: usize, pub flipped_outward_components: usize }
pub fn orient(tx: &mut Transaction) -> Result<OrientReport>;
```
- [ ] **Step 1: Write failing tests:**
  - `hemisphere_random_flips_fixed`: flip 1%, and separately a 30-triangle patch. Afterwards `analyze().orientation_conflicts == 0`, and every normal matches the majority (dot with the outward radial direction > 0).
  - `inward_sphere_turned_outward`: signed volume > 0 afterwards.
  - `mobius_reported_unchanged`: `non_orientable_patches == 1`, and the triangles are unchanged.
  - `fin_patches_independent`: each quad of the fin keeps its own majority, and no error is raised.
  - `already_consistent_is_empty_patch`.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.** Breadth-first search per patch with a parity bit per triangle. A conflict on revisit marks the patch non-orientable. Majority by area within the patch. Then outward flipping for closed components with no non-manifold edges. One `set_triangles` call for all flips.
- [ ] **Step 4: Run** `cargo test --workspace`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(ops): add orient`.

### Task 5: `fill_holes`: selection, triangulation, atomic write (no refine or fair yet)

**Files:**
- Create: `crates/fast-mesh-ops/src/fill.rs`, `crates/fast-mesh-ops/src/fill_dp.rs`

**Interfaces (produces):**
```rust
pub struct FillOptions { pub max_edges: Option<usize>, pub max_perimeter: Option<f64>, pub max_area: Option<f64>,
                         pub loops: Option<Vec<LoopId>>, pub allow_unclean: bool, pub fair: bool, pub dp_limit: usize }
// Default: None, None, None, None, false, true, 200
pub enum SkipReason { Bounds, Unclean }
pub enum FailReason { ProjectionFolds, NoValidTriangulation, Solver }
pub struct FilledLoop { pub id: LoopId, pub triangles_added: usize, pub vertices_added: usize, pub converged: bool }
pub struct FillReport { pub filled: Vec<FilledLoop>, pub skipped: Vec<(LoopId, SkipReason)>, pub failed: Vec<(LoopId, FailReason)> }
pub fn fill_holes(tx: &mut Transaction, options: &FillOptions) -> Result<FillReport>;
pub(crate) fn triangulate_loop(mesh: &Mesh, loop_vertices: &[VertexId], forbidden: &EdgeMap, dp_limit: usize) -> std::result::Result<Vec<[VertexId; 3]>, FailReason>;
```
- In this task, `fair` and refinement are accepted but not yet applied. `vertices_added = 0` and `converged = true`. Task 6 adds them.
- **DP (Liepa):**
  - `W(i,k) = min over m` of `W(i,m) + W(m,k) + weight(i,m,k)`.
  - `weight` compares lexicographically: (worst dihedral to the adjacent triangles, already-placed patch triangles or the mesh triangle on the loop edge; then area).
  - A diagonal `(i,k)` that already exists as a mesh edge is forbidden (Review Focus 4).
- **Fallback above `dp_limit`:** project onto the best-fit plane (PCA in f64), then call `fast_mesh_query::triangulate`. If it returns `false`, or the projected polygon self-intersects (check with a segment sweep, or `triangulate`'s area check), the loop fails with `ProjectionFolds`.
- **Winding:** patch triangles take the winding opposite to the loop's half-edge direction, so they agree with the neighbouring triangles.
- [ ] **Step 1: Write failing tests:**
  - `sphere_holes_filled_closed_and_consistent`: `punch_holes(uv_sphere(64,128,1), 3 centres, 0.2)` then `fill_holes(max_edges: Some(500), fair: false)`. Afterwards there are 0 loops, `orientation_conflicts == 0`, and the components are closed.
  - `bowtie_holes_both_filled`.
  - `hemisphere_rim_kept_by_perimeter_bound`: a hemisphere with 2 small punched holes and `max_perimeter: Some(1.0)` fills 2 loops and leaves the rim (≈ 2π).
  - `stale_loop_id_after_weld_in_same_tx_is_rejected`:
    1. Take the ids from `boundary_loops`.
    2. In one transaction, run `weld` with a tolerance that merges a loop vertex into a neighbour, so the loop changes.
    3. Then `fill_holes(loops: Some(ids))` returns `Err(StaleLoops)`.
    4. Nothing from the fill is written: the patch equals the weld-only patch.
  - `fill_avoids_existing_edges`: no non-manifold edge after filling a loop whose two non-adjacent vertices are already joined by a mesh edge.
  - `large_loop_uses_projection_fallback`: a 400-edge planar-ish hole is filled when `dp_limit = 200`.
  - `folded_projection_fails_and_writes_nothing`: a helical or strongly non-planar loop above `dp_limit` is reported in `failed` with `ProjectionFolds`, and its loop is still present.
  - `no_bound_invalid_input`.
  - `unclean_loop_skipped_unless_allowed`.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.** Plan every selected loop first. Then write all successful patches with `add_triangles`, in loop-id order.
- [ ] **Step 4: Run** `cargo test --workspace`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(ops): add fill_holes with Liepa triangulation`.

### Task 6: Refinement and fairing

**Files:**
- Create: `crates/fast-mesh-ops/src/refine.rs`, `crates/fast-mesh-ops/src/fair.rs` (sparse matrix plus conjugate gradient)
- Modify: `crates/fast-mesh-ops/src/fill.rs`

**Interfaces:**
- Consumes: Task 5 `triangulate_loop`, `FillOptions.fair`, `FilledLoop`.
- Produces:
  - `pub(crate) fn refine(mesh: &Mesh, loop_vertices: &[VertexId], patch: Vec<[VertexId;3]>) -> Result<RefinedPatch>`. `RefinedPatch` holds the new vertex positions, each with its creation parents (`Vec<(VertexId, f32)>` referencing loop vertices or earlier new vertices by a local index scheme you define), plus its triangles.
  - `pub(crate) fn fair(mesh: &Mesh, patch: &mut RefinedPatch, outer_ring: &[VertexId]) -> FairResult { converged: bool, iterations: usize }`.
- **Density rule (spec §5.3 step 2):**
  - Each loop vertex gets target σ = the mean length of its existing mesh edges. A new vertex gets the mean σ of its parents.
  - Split a triangle at its centroid when the centroid's σ times √2 is less than the distance from the centroid to each of its corners, as in Liepa.
  - Then relax interior patch edges by flipping, until no flip improves the circumcircle criterion. Cap the number of passes.
  - Repeat until no split happens.
- **Fairing:**
  - Solve the bi-Laplacian `L(L x) = 0` for the new vertices' x, y and z separately. Loop vertices and the one-ring of mesh vertices outside the loop are fixed.
  - Cotangent weights, clamped at ≥ 1e-6 of their mean. If CG diverges or produces a non-finite value, retry with uniform weights, and record which was used in `FilledLoop`.
  - CG uses a Jacobi preconditioner, tolerance 1e-10 relative, and an iteration cap of `20·n + 100`.
  - When not converged, keep the best iterate and set `converged: false`. If that iterate is non-finite, the loop fails with `FailReason::Solver` and nothing is written for it.
- **Writing:** new vertices go in with `add_vertex_with_parents`, then the triangles.
- [ ] **Step 1: Write failing tests:**
  - `faired_patch_follows_sphere`: on `uv_sphere(64,128,1)`, punch holes of angular radius 10°, 20° and 30°. After `fill_holes(fair: true, max_edges: Some(1000))`, every new vertex has `| |p| − 1 | < 0.02`.
  - `unfaired_patch_is_flatter`: the same holes with `fair: false` give a larger maximum radius error than with `fair: true`.
  - `density_matches_surroundings`: the mean edge length in the patch is within ×0.5 to ×1.5 of the mean edge length around the loop.
  - `fairing_thin_hole_no_nan`: a 2×60 sliver hole gives finite positions and an honest `converged` flag.
  - `parents_valid`: every new vertex's parent weights sum to 1 within 1e-6, and its parents are alive or new.
  - `undo_restores_after_fill`.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cargo test --workspace`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(ops): refine and fair hole patches`.

### Task 7: Scene integration and the pipeline property test

**Files:**
- Modify:
  - `crates/fast-mesh-scene/Cargo.toml`: dependency `fast-mesh-ops`.
  - `crates/fast-mesh-scene/src/error.rs`: `SceneError::Ops(OpsError)`, `From<OpsError>`, and Display text `"repair operation failed: {e}"`.
  - `crates/fast-mesh-scene/src/scene.rs`: `edit` becomes generic.
  - `crates/fast-mesh-scene/src/lib.rs`: re-export `fast_mesh_ops` as `pub use fast_mesh_ops as ops;`.
- Create: `crates/fast-mesh-ops/tests/pipeline_properties.rs`, `crates/fast-mesh-scene/tests/repair.rs`

**Interfaces:**
- Produces:
  - `pub fn edit<R, E: Into<SceneError>>(&mut self, id: MeshId, f: impl FnOnce(&mut Transaction) -> std::result::Result<R, E>) -> Result<R>`
  - `StoreError` already converts into `SceneError`, so existing call sites compile unchanged. Verify by building the workspace without editing them.
- [ ] **Step 1: Write failing tests:**
  - scene `repair_via_edit_is_undoable`: a hemisphere with holes and flips in an Eager scene. Run `scene.edit(id, |tx| { ops::orient(tx)?; ops::fill_holes(tx, &opts) })`. The BVH is current, `undo()` restores the exact positions and triangles, and `redo()` reapplies.
  - scene `repair_updates_fields_and_sections`: a distance field and a section on the repaired mesh are current and equal from-scratch after the edit.
  - scene `ops_error_maps_to_scene_error`: `fill_holes` without bounds via `edit` gives `Err(SceneError::Ops(OpsError::InvalidInput(_)))`, and the scene is unchanged.
  - ops property test `pipeline_never_panics_and_orients`:
    - Random meshes (sphere or grid) with random damage: holes, flips, duplicated vertices, floating pieces, degenerates.
    - Run `weld(0) → remove_degenerate → remove_small_components(min_triangles 10) → orient → fill_holes(max_edges 300)` in one transaction.
    - It never panics or errors (except `OutOfMemory`). `orientation_conflicts == 0` on orientable components. A second `analyze` agrees with the reports.
    - Use a deterministic seed with `PROPTEST_RNG_SEED` honoured.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cargo test --workspace`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(scene): run repair operations through edit`.

### Task 8: Benchmarks and docs

**Files:**
- Create: `crates/fast-mesh-ops/benches/ops.rs`
- Modify:
  - `crates/fast-mesh-ops/Cargo.toml` (`[[bench]]`)
  - `docs/benchmarks.md` (new "Repair" section)
  - the spec (add "§12 Implementation notes": rulings, deviations, measured numbers)
  - the crate docs

**Interfaces:** consumes everything; produces nothing.

- [ ] **Step 1: Add benches:**
  - `analyze_1M`, with and without the self-intersection check;
  - `orient_1M_1pct_flipped`;
  - `fill_10_holes_1M` (50–500 edges, fair on and off);
  - `weld_1M_10pct_dupes`;
  - `remove_small_components_1M`.
- [ ] **Step 2: Run** `cargo bench -p fast-mesh-ops --bench ops` and record real numbers on the i5-13600K. Never invent a number. If a bench is impractical, say so.
- [ ] **Step 3: Write** `benchmarks.md` and spec §12 in the format of the existing sections.
- [ ] **Step 4: Verify** fmt, clippy and `cargo test --workspace`. Expected: green.
- [ ] **Step 5: Commit** `bench(ops): add repair benchmarks` and `docs(repair): record benchmarks and implementation notes`.
