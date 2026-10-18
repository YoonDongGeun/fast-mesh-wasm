# fast-mesh Sections (Phase 6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plane cross-sections. A standalone slicing query produces contours per mesh, with measurements, fill and an in-plane gap. Scene sections keep those results current under plane moves, placements and edits.

**Architecture:**
- Slicing lives in `fast-mesh-query`:
  - `plane.rs`: the plane type and its frame.
  - `slice.rs`: sides, crossings, BVH walk and topological chaining.
  - `triangulate.rs`: earcut with holes.
  - `section.rs`: 2D measurements, nesting, fill and gap.
- The Scene adds `sections.rs`. Each section keeps one entry per selected mesh. Each selected mesh carries a section hook that is fed by the same plumbing as field hooks (`Entry::record_changes`, `report_placement`, `report_everything`). Recompute timing follows the mesh's update mode.

**Tech Stack:** Rust 1.99 (MSRV 1.88, edition 2024). Workspace crates `fast-mesh-store`, `fast-mesh-bvh`, `fast-mesh-query` and `fast-mesh-scene`. Tests use proptest 1.12; benches use criterion 0.8.

**Spec:** `docs/superpowers/specs/2026-10-18-fast-mesh-sections-design.md`. §11 overrides earlier sections.

## Global Constraints
- No `unsafe`. MSRV 1.88, edition 2024. No new crate dependencies.
- No heap allocation inside BVH traversal: use a fixed stack, as `region.rs` does.
- Outputs grow with `try_reserve`. Failure maps to `QueryError::OutOfMemory` or `SceneError::Query(..)` / `Store(..)`.
- Side rule: `s(p) = n·p − n·origin` in `f64`. A vertex is `+` iff `s ≥ 0`.
- A crossing point is computed from the edge endpoints in canonical order (smaller vertex id first): `t = s(a)/(s(a)−s(b))`, `p = a + t·(b−a)`. Points are identified by their edge key `(min, max)`.
- Default frame: `u` is the world axis least parallel to `n`, made orthogonal to `n` and normalised; `v = n × u`. With `with_axes`, the axes must be orthonormal within `1e-6`, and `n = u × v`.
- Fill counts as failed (`fill_ok: false`) when ear clipping cannot finish, or when the fill area differs from the summed contour area by more than `1e-6` relative.
- Errors:
  - unknown section → `SceneError::UnknownSection` (new);
  - stale or unknown mesh → `UnknownMesh`;
  - live mesh the section does not select → `InvalidInput`;
  - duplicate mesh in a selection, or an invalid plane → `InvalidInput`.
- Before an entry is first computed, `section` returns an empty `SectionResult` and `section_status` returns `Stale`.
- Sections are not part of history. The `field:` prefix stays the only reserved channel prefix.
- Commits:
  - conventional messages ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or the implementing model);
  - each message goes in a NEW temp file under the session scratchpad, committed with `git commit -F`;
  - files are written UTF-8 with LF line endings, using the Edit and Write tools.
- Each task ends green on all three: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.

## Review Focus
1. **Plane through many vertices at once.** An axis-aligned plane through a grid row of a terrain, or through a cube face, must give closed (cube) or end-to-end (terrain) chains with no breaks. Zero-length segments are allowed. Tests: Task 2 `cube_face_flush_plane_is_one_closed_loop` and `terrain_grid_row_plane_is_open_chains`.
2. **Mesh with large origin or local coordinates (~1000), rotated.** World contour points must match the brute-force slice to within `1e-9 · magnitude`. Test: Task 5 `world_points_match_local_slice_with_origin`.
3. **Degenerate triangles** (repeated vertex, zero area) crossing the plane. No panic, and no extra open chains on an otherwise closed mesh. Test: Task 2 `degenerate_triangles_do_not_break_chains`.
4. **Plane missing every mesh, or a mesh with zero triangles.** Empty result, status `Current`, gap `None`. Tests: Task 5 `plane_missing_mesh_gives_empty_current_result`, and Task 4 `gap_none_when_either_empty`.
5. **Removing a mesh, or a section, while subscribers are undrained.** No panic. The removal is reported once per affected mesh, and later reads return `UnknownSection` / `UnknownMesh`. Test: Task 5 `remove_section_and_mesh_report_once`.

---

### Task 1: Plane and segment extraction (`fast-mesh-query`)

**Files:**
- Create: `crates/fast-mesh-query/src/plane.rs`, `crates/fast-mesh-query/src/slice.rs` (segments only for now)
- Modify: `crates/fast-mesh-query/src/lib.rs` (add `mod plane; mod slice; pub use plane::Plane;` and extend the module doc list)
- Modify: `crates/fast-mesh-query/src/test_util.rs` (add fixtures)

**Interfaces:**
- Produces:
  - `pub struct Plane` (fields private) with:
    - `Plane::new(origin: [f64;3], normal: [f64;3]) -> Result<Plane>`: normalises; `InvalidInput` if the normal is non-finite or zero, or the origin is non-finite.
    - `Plane::with_axes(origin, u: [f64;3], v: [f64;3]) -> Result<Plane>`
    - `origin() -> [f64;3]`, `normal() -> [f64;3]`, `u() -> [f64;3]`, `v() -> [f64;3]`
    - `signed_distance(&self, p: [f64;3]) -> f64`
    - `to_uv(&self, p: [f64;3]) -> [f64;2]`, with `(p−origin)·u` and `(p−origin)·v`
    - `from_uv(&self, q: [f64;2]) -> [f64;3]`
  - `#[derive(Debug, Clone, Copy, PartialEq)]`.
  - In `slice.rs`:
    - `pub(crate) struct Segment { pub(crate) tri: TriangleId, pub(crate) from: (VertexId, VertexId), pub(crate) to: (VertexId, VertexId) }`. `from` and `to` are edge keys, already oriented per Task 2's rule. Orientation is computed here.
    - `pub(crate) fn segments(mesh: &Mesh, bvh: &Bvh, plane: &Plane, out: &mut Vec<Segment>) -> Result<()>`
    - `pub(crate) fn segments_brute(mesh: &Mesh, plane: &Plane, out: &mut Vec<Segment>)`, the oracle, behind `#[cfg(test)]`
    - `pub(crate) fn crossing(mesh: &Mesh, plane: &Plane, key: (VertexId, VertexId)) -> [f64;3]`
  - Test fixtures in `test_util.rs`:
    - `uv_sphere(stacks: u32, slices: u32, r: f32) -> Mesh`: closed, outward;
    - `box_mesh(lo: [f32;3], hi: [f32;3]) -> Mesh`: 12 triangles, outward;
    - `torus(major: f32, minor: f32, n: u32, m: u32) -> Mesh`;
    - `fin() -> Mesh`: three quads sharing one edge, non-manifold;
    - `with_degenerates(mesh: Mesh) -> Mesh`: appends zero-area triangles that reuse existing vertices.

- [ ] **Step 1: Write failing tests** in `plane.rs` and `slice.rs` `#[cfg(test)]`:
  - `plane_new_normalises_and_rejects_zero`:
    - `Plane::new([0.;3],[0.,0.,2.])` gives `normal() == [0.,0.,1.]`.
    - A zero normal and a NaN normal both give `Err(InvalidInput(_))`.
  - `default_frame_is_orthonormal_right_handed`:
    - For 1000 random normals, `|u|=|v|=1`, `u·n=v·n=u·v=0` within `1e-12`, and `u×v == n` within `1e-12`.
  - `with_axes_rejects_non_orthonormal`:
    - `u=[1,0,0], v=[1,1,0]` gives `Err`.
  - `uv_round_trip`:
    - `from_uv(to_uv(p))` equals the projection of `p` onto the plane, within `1e-12·|p|`.
  - `segments_match_brute_force`:
    - Meshes: `wavy(40)`, `uv_sphere(32,64,1.)`, `torus(2.,0.5,48,24)`.
    - Planes: 200 random planes, plus 50 planes whose origin is snapped to a random vertex, and 50 containing a random edge.
    - The sorted multiset of `(tri, from, to)` from `segments` equals `segments_brute`.
  - `segment_orientation_rule`:
    - For every segment on `uv_sphere` cut by `z = 0.3`, the triangle's centroid projected onto `(u, v)` lies to the left of `from→to` when the centroid is on the `+` side. Equivalently, the cross product of `(to−from)` and `(centroid−from)` in uv has the sign of `s(centroid)`, for every centroid with `|s| > 1e-9`.
  - `crossing_is_symmetric_in_triangle_order`:
    - For every crossed edge, `crossing(key)` is bit-identical whichever of its two triangles produced it. This holds by construction; assert it through the segment endpoints.
- [ ] **Step 2: Run** `cargo test -p fast-mesh-query plane slice`. Expected: compile failure / FAIL.
- [ ] **Step 3: Implement.**
  - **Box straddle test.** Take the min and max of `s` over the 8 box corners. Inflate by `8·f32::EPSILON·(box magnitude + |n·origin|)`. Visit the node when `min ≤ pad` and `max ≥ −pad`.
  - **Fixed-stack walk.** Copy the stack depth and overflow pattern from `region.rs`.
  - **Per triangle.** Compute the vertex sides (`s ≥ 0` is `+`). Skip triangles with all sides equal. The lone vertex `L` is the one whose side differs from the other two. The two crossed edges are those touching `L`.
  - **Orientation.** For triangle `(a,b,c)` with `L = a`, the edges are `(a,b)` and `(c,a)`. If `L` is `+`, emit `from = key(c,a)`, `to = key(a,b)`. If `L` is `−`, emit the reverse. Verify this against `segment_orientation_rule`; fix the order if the test says otherwise. The test is authoritative.
  - **Staleness.** `segments` calls `crate::check` first.
- [ ] **Step 4: Run** `cargo test -p fast-mesh-query`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(query): add Plane and plane-triangle segment extraction`.

### Task 2: Chaining into contours: the public `slice`

**Files:**
- Modify: `crates/fast-mesh-query/src/slice.rs`
- Modify: `crates/fast-mesh-query/src/lib.rs` (`pub use slice::{slice, Slice, Contour};`)

**Interfaces:**
- Consumes: Task 1 `segments`, `crossing`, `Plane`.
- Produces:

```rust
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Slice { pub contours: Vec<Contour> }
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Contour {
    pub points: Vec<[f64; 3]>,
    pub edges: Vec<(VertexId, VertexId)>,
    pub triangles: Vec<TriangleId>, // len = points.len() - 1, or points.len() when closed
    pub closed: bool,
    pub oriented: bool,
    pub manifold: bool,
}
pub fn slice(mesh: &Mesh, bvh: &Bvh, plane: &Plane, out: &mut Slice) -> Result<()>;
```

  The plane and the output points are both in the mesh's stored local coordinates: the `f32` positions widened to `f64`, origin offset excluded. Callers that work in the world add `mesh.origin()` and the placement themselves, as Task 5 does.

- [ ] **Step 1: Write failing tests:**
  - `sphere_cut_is_one_closed_ccw_loop`:
    - `uv_sphere(64,128,1.)` cut by `z = 0.3` gives 1 contour that is `closed && oriented && manifold`.
    - Its shoelace area in `(u, v)` is positive and within 1% of `π(1−0.09)`.
  - `torus_horizontal_cut_is_outer_and_hole`:
    - The cut at `z = 0` gives 2 closed contours, one with positive and one with negative area.
    - The magnitudes are ≈ `π·2.5²` and `π·1.5²`, within 1%.
  - `cube_face_flush_plane_is_one_closed_loop`:
    - `box_mesh([-1;3],[1;3])` cut by `z = 1.0` exactly, and by `z = -1.0`: each case gives 0 or 1 contours, and every contour is `closed && manifold`.
    - Also `x = 0`, through the edge midpoints: 1 closed contour.
  - `terrain_grid_row_plane_is_open_chains`:
    - `wavy(20)` cut by `y = 7.0`, which passes exactly through a vertex row.
    - Every contour is open, and its first and last edges are boundary edges (each touched by one triangle).
    - Every segment from `segments` is used exactly once.
  - `nonmanifold_fin_splits_chains`:
    - `fin()` cut across the shared edge gives at least 3 open contours, each with `manifold == false`.
  - `degenerate_triangles_do_not_break_chains`:
    - `with_degenerates(uv_sphere(16,32,1.))` cut by 50 random planes through the centre gives exactly 1 closed contour each.
  - `flipped_patch_reports_unoriented`:
    - On a sphere with one triangle's winding reversed, the contour through that triangle gives `oriented == false` and is still closed.
  - `deterministic_order`:
    - Two calls give equal `Slice` values.
    - Each closed contour starts at its smallest edge key.
    - Contours are sorted by their first edge key.
  - `slice_rejects_stale_bvh`:
    - Edit the mesh without updating the BVH; the result is `Err(QueryError::StaleBvh)`.
- [ ] **Step 2: Run** `cargo test -p fast-mesh-query slice`. Expected: FAIL.
- [ ] **Step 3: Implement `slice`.**
  - **Endpoint map.** Sort the segment endpoints by edge key into a temporary vector, or use a `HashMap<(u32,u32), SmallList>`; builder's choice. Use `try_reserve` for every growth.
  - **Edge-key degree.** Directed chaining joins a `to` to a `from` with the same key.
    - Degree 2 with one `from` and one `to`: oriented join.
    - Degree 2 with two `from`s or two `to`s: join undirected, reverse the piece, and mark the contour `oriented = false`.
    - Degree 1 (boundary edge): the chain ends.
    - Degree above 2: split there, and give every piece touching that key `manifold = false`.
  - **Walking.** Start open chains from their degree-1 or degree-above-2 ends, in ascending key order. Then walk the remaining closed loops, starting at each loop's smallest key.
  - **Points.** `points[i] = crossing(edges[i])`.
  - **Reuse.** Reuse `out.contours`' allocations where convenient; correctness first.
- [ ] **Step 4: Run** `cargo test -p fast-mesh-query`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(query): chain plane segments into contours (slice)`.

### Task 3: Polygon triangulation with holes (`fast-mesh-query::triangulate`)

**Files:**
- Create: `crates/fast-mesh-query/src/triangulate.rs`
- Modify: `crates/fast-mesh-query/src/lib.rs` (`pub mod triangulate;`)

**Interfaces:**
- Produces:
  - `pub fn triangulate(outer: &[[f64;2]], holes: &[&[[f64;2]]], out: &mut Vec<[u32;3]>) -> Result<bool>`
  - Indices number `outer` first, then each hole in order.
  - Triangles are counter-clockwise.
  - Input orientation is normalised internally: outer to CCW, holes to CW.
  - Consecutive equal points are skipped. Their indices are never emitted, but they stay counted in the numbering.
  - Returns `Ok(false)` when ear clipping could not finish. The partial triangles are kept.
  - Never panics.

- [ ] **Step 1: Write failing tests:**
  - `square_gives_two_triangles`: total area is 1.
  - `concave_l_shape_area_exact`: an L of area 3 gives a triangle-area sum of 3 within `1e-12`.
  - `square_with_square_hole`: area `16 − 4 = 12` within `1e-12`, and returns `true`.
  - `two_holes`: area is correct, and no triangle centroid lies inside a hole.
  - `duplicate_consecutive_points_ignored`: area is unchanged, and the duplicate indices never appear.
  - `collinear_points_ok`.
  - `cw_input_is_normalised`: a CW outer gives the same area, and every output triangle has positive area.
  - `proptest_never_panics`: random point lists of up to 64 points, including self-intersecting and collinear ones, plus random holes. The call returns without panicking, and every index is below the total.
  - `star_polygon_100_vertices_area`: matches the shoelace area within `1e-9` relative.
- [ ] **Step 2: Run** `cargo test -p fast-mesh-query triangulate`. Expected: FAIL.
- [ ] **Step 3: Implement** the earcut approach:
  - a doubly linked list over a `Vec`;
  - hole bridging: each hole, sorted by leftmost x, connects to the outer vertex found by the ray-cast visibility test;
  - ear test by point-in-triangle against reflex vertices only;
  - on failure, one fallback pass that cures local self-intersections, then give up and return `false`.
  - No z-order hashing is required at this size. Add it only if Task 7's fill bench at 1M exceeds 5 ms.
- [ ] **Step 4: Run** `cargo test -p fast-mesh-query`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(query): add earcut polygon triangulation with holes`.

### Task 4: Measurements, nesting, fill and gap (`fast-mesh-query::section`)

**Files:**
- Create: `crates/fast-mesh-query/src/section.rs`
- Modify: `crates/fast-mesh-query/src/lib.rs` (re-export below)

**Interfaces:**
- Consumes: `Slice`, `Contour`, `Plane::to_uv` / `from_uv`, `triangulate`.
- Produces (all on uv polylines, pure 2D):
  - `pub fn contour_length(points: &[[f64;2]], closed: bool) -> f64`
  - `pub fn signed_area(points: &[[f64;2]]) -> f64`: shoelace; positive means CCW.
  - Fill:

```rust
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Fill { pub points: Vec<[f64;2]>, pub triangles: Vec<[u32;3]> }
pub fn fill(contours: &[(&[[f64;2]], bool /*oriented*/)], out: &mut Fill) -> Result<bool>
```

    - Input: closed contours only.
    - Nesting follows spec §5.3. Oriented contours: CCW are outers, and each CW contour goes to the smallest containing outer. If any contour is unoriented, use even-odd depth by containment for all.
    - Returns the `fill_ok` value. It is false if any `triangulate` returns false, or if `|fill area − Σ signed contour area| > 1e-6·max(1, Σ|area|)`. In even-odd mode, compare against the depth-signed sum.
  - Gap:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gap2 { pub distance: f64, pub a: [f64;2], pub b: [f64;2] }
pub fn contour_gap(a: &[(&[[f64;2]], bool /*closed*/)], b: &[(&[[f64;2]], bool)]) -> Result<Option<Gap2>>
```

    - Returns `None` when either side has no segments.
    - Distance is 0 with the crossing point when segments intersect.
    - Uses a uniform grid sized from the segment count, or a sweep; not all-pairs.

- [ ] **Step 1: Write failing tests:**
  - `length_and_area_of_unit_square`: closed length 4, area 1; reversed order gives area −1.
  - `fill_ring_area`: outer circle r=2 (CCW, 256 points) and inner r=1 (CW) give a fill area of ≈ `3π` within 0.1%, and `true`.
  - `fill_nested_island`: square 10 CCW, square 6 CW, square 2 CCW. The fill area is `100 − 36 + 4`.
  - `fill_unoriented_even_odd`: the same three squares, all flagged unoriented with arbitrary windings, give the same area.
  - `fill_self_intersecting_reports_false_or_ok_area`: a bow-tie input never panics. Either it returns `false`, or the fill area matches the summed area.
  - `gap_concentric_circles`: r=1 and r=3 give a distance of ≈2 within `1e-3` (with 512 points each), and the points lie on the circles.
  - `gap_crossing_is_zero`.
  - `gap_matches_brute_force`: proptest with 2–5 random polylines per side, up to 50 segments each. The result equals the all-pairs minimum within `1e-12`.
  - `gap_none_when_either_empty`.
- [ ] **Step 2: Run** `cargo test -p fast-mesh-query section`. Expected: FAIL.
- [ ] **Step 3: Implement.** Point-in-polygon uses the even-odd ray rule on uv. The containment test for nesting checks one vertex of the inner contour.
- [ ] **Step 4: Run** `cargo test -p fast-mesh-query`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(query): add contour measurements, fill and in-plane gap`.

### Task 5: Scene sections: API, results, reports (full recompute on any relevant change)

**Files:**
- Create: `crates/fast-mesh-scene/src/sections.rs`
- Modify:
  - `crates/fast-mesh-scene/src/scene.rs`: `Entry` gets `section_hooks: Vec<SectionHook>`. `report_everything`, `report_placement` and `record_changes` feed it, the same way as `hooks`. `Scene` gets `sections: Vec<Section>` and `next_section: u32`. `remove_mesh`, `compact`, `update(id)` and `flush` integrate sections.
  - `crates/fast-mesh-scene/src/report.rs`: `SectionId` (opaque `u32`, never reused, like `FieldId`), `MeshChanges.sections_changed: Vec<SectionId>`, and the `Report` field with its `clean` / `is_clean` / merge helpers.
  - `crates/fast-mesh-scene/src/error.rs`: `SceneError::UnknownSection` with Display text `"unknown or removed section id"`.
  - `crates/fast-mesh-scene/src/history.rs`: section hooks must see undo and redo. Check that this already happens through `record_changes`; if not, do it in `mark_corrupted` / `run_step` the way field hooks are handled.
  - `crates/fast-mesh-scene/src/lib.rs`: re-exports `pub use sections::{SectionDesc, SectionResult, SectionContour, SectionStatus, Gap};`, `pub use report::SectionId;`, `pub use fast_mesh_query::Plane;`

**Interfaces:**
- Consumes: Tasks 2–4 (`slice`, `Plane`, `contour_length`, `signed_area`, `fill`, `contour_gap`). Scene: `Entry`, `UpdateMode`, `Placement::apply` / `inverse_apply`, `Mesh::origin`.
- Produces (the public API from spec §6.1, as adjusted by §11):

```rust
pub struct SectionDesc { pub plane: Plane, pub meshes: Vec<MeshId>, pub fill: bool }
pub enum SectionStatus { Current, Stale }
pub struct SectionContour {
    pub points: Vec<[f64;3]>,   // world
    pub uv: Vec<[f64;2]>,
    pub edges: Vec<(VertexId, VertexId)>,
    pub triangles: Vec<TriangleId>,
    pub closed: bool, pub oriented: bool, pub manifold: bool,
    pub length: f64,
    pub area: f64,              // signed, 0 when open
}
pub struct SectionResult {
    pub contours: Vec<SectionContour>,
    pub length: f64,
    pub area: f64,              // sum over closed && oriented contours
    pub fill: Option<Fill>,     // Some iff the section has fill on and the entry was computed; points are uv, world = plane.from_uv
    pub fill_ok: bool,          // true when fill is None
}
pub struct Gap { pub distance: f64, pub a: [f64;3], pub b: [f64;3] }

impl Scene {
    pub fn add_section(&mut self, desc: SectionDesc) -> Result<SectionId>;
    pub fn remove_section(&mut self, id: SectionId) -> Result<()>;
    pub fn set_section_plane(&mut self, id: SectionId, plane: Plane) -> Result<()>;
    pub fn set_section_meshes(&mut self, id: SectionId, meshes: Vec<MeshId>) -> Result<()>;
    pub fn set_section_fill(&mut self, id: SectionId, fill: bool) -> Result<()>;
    pub fn update_sections(&mut self) -> Result<()>;
    pub fn section(&self, id: SectionId, mesh: MeshId) -> Result<&SectionResult>;
    pub fn section_status(&self, id: SectionId, mesh: MeshId) -> Result<SectionStatus>;
    pub fn section_gap(&self, id: SectionId, a: MeshId, b: MeshId) -> Result<Option<Gap>>;
}
```

- Internal:
  - `pub(crate) struct SectionHook { section: SectionId, changes: Option<ChangeSet>, placement_changed: bool }`
  - Each `Section` has `entries: Vec<SectionEntry { mesh, result, stale: bool, computed_once: bool, crossed_vertices: Vec<VertexId> /*sorted*/, crossed_triangles: Vec<TriangleId> /*sorted*/ }>`. The crossed sets are filled here but used in Task 6.
- **Staleness in this task.** Every relevant event stales the entry:
  - any non-empty change set;
  - a placement change;
  - a plane or fill change;
  - selection added;
  - `None` changes.
- **Recompute.**
  - Map the world plane to local: `origin_l = inverse_apply(plane.origin) − mesh.origin()`, `n_l = Rᵀ n`.
  - Build the local `Plane` with axes `Rᵀu` and `Rᵀv`, so that local uv equals world uv.
  - Call `slice`. Map points back with `apply(p + mesh.origin())`.
  - uv comes from the world plane.
  - Then compute measurements, and fill when it is on.
  - Report `sections_changed` on the mesh when the new result differs from the old one, or the entry was never computed.
- **Timing (spec §6.3).**
  - After any staling event on a mesh in Eager mode, recompute right away and swallow errors.
  - Lazy and Manual meshes recompute on `update_sections`, `update(id)` or `flush`.
  - A stale BVH under Manual leaves the entry Stale. `update_sections` returns the first `StaleBvh(id)`.
  - Under Lazy, `update_sections` brings the BVH up to date first, through `Entry::update_bvh`.

- [ ] **Step 1: Write failing unit tests** in `sections.rs` `#[cfg(test)]`, using the scene test helpers already in `scene.rs` and `fields.rs`:
  - `add_section_eager_computes_immediately`: a sphere mesh in Eager mode. After `add_section`, the status is `Current`, there is 1 closed contour, and `length` ≈ `2π·0.954` (for the cut at z=0.3).
  - `lazy_waits_for_update_sections`: the status is `Stale` and the result is empty until `update_sections()`, then `Current`.
  - `plane_move_restales_and_recomputes_eager`.
  - `world_points_match_local_slice_with_origin`:
    - The mesh has origin `[1000.,-1000.,500.]`, a random rotation and a translation.
    - Every world point equals `placement.apply(local_point)`, where `local_point` comes from a direct `slice` on the mesh with the mapped plane, within `1e-9·1000`.
    - uv equals `world_plane.to_uv(point)` within `1e-9·1000`.
  - `plane_missing_mesh_gives_empty_current_result`.
  - `edit_undo_redo_keep_result_equal_to_from_scratch`: after each operation, `section()` equals a freshly computed one from a new section with the same desc.
  - `placement_change_recomputes`.
  - `fill_on_gives_area_matching_contours`: on a closed sphere, `fill_ok` is true, and the fill area equals `area` within `1e-6` relative.
  - `gap_between_two_spheres`: spheres at x=0 and x=3, both radius 1, cut by `z=0` give a gap ≈ 1 within `1e-3`. The points are world points on the line between the centres.
  - Error paths:
    - unknown or removed section gives `UnknownSection`;
    - a removed mesh in `meshes` gives `UnknownMesh`;
    - a duplicate mesh gives `InvalidInput`;
    - `section(id, unselected_live_mesh)` gives `InvalidInput`.
  - `remove_section_and_mesh_report_once`:
    - With two subscribers, `remove_section` reports the id once in `sections_changed` of each selected mesh, for each subscriber.
    - `remove_mesh` drops the mesh from the section without panicking.
    - Later reads give `UnknownSection` / `UnknownMesh`.
  - `compact_recomputes_and_reports`.
  - `manual_stale_bvh_returns_error_and_stays_stale`.
  - `sections_changed_exact`: a plane move reports exactly that section on exactly its meshes, each once.
- [ ] **Step 2: Run** `cargo test -p fast-mesh-scene sections`. Expected: FAIL.
- [ ] **Step 3: Implement** as specified above. Keep `sections.rs` parallel to `fields.rs` in structure (an `impl Scene` block plus private helpers).
- [ ] **Step 4: Run** `cargo test --workspace`. Expected: PASS, with existing tests unchanged apart from the new `MeshChanges` field in literal constructions.
- [ ] **Step 5: Commit** `feat(scene): add plane sections with contours, fill and gap`.

### Task 6: The "can this change reach the plane" rule, and the property test

**Files:**
- Modify: `crates/fast-mesh-scene/src/sections.rs`
- Create: `crates/fast-mesh-scene/tests/sections_properties.rs`

**Interfaces:**
- Consumes: Task 5 `SectionEntry.crossed_vertices` / `crossed_triangles` (filled on every recompute from the contours' `edges` and `triangles`), `SectionHook.changes`.
- Produces: no new public API. A `#[cfg(test)] pub(crate)` recompute counter per scene, to make far edits observable.

**Rule (spec §6.3 together with §11.5).** For a non-`None` change set with no placement change and no plane change, the entry stales iff any of these holds:
1. any vertex in `vertices_moved` or `vertices_removed` is in `crossed_vertices`;
2. any triangle in `triangles_removed` or `triangles_changed` is in `crossed_triangles`;
3. any vertex in `vertices_moved` or `vertices_added` that is not in `crossed_vertices` has an alive incident triangle (current topology; use `Mesh::ensure_incidence` / `incidence().triangles(v)`) whose three vertices are not all on one side;
4. any triangle in `triangles_added` or `triangles_changed` is alive and not all on one side.

Otherwise the entry stays Current, nothing is recomputed, and nothing is reported. `None` changes, placement changes and compaction always stale the entry.

- [ ] **Step 1: Write failing tests:**
  - Unit test `far_edit_does_not_recompute`: on a 64×128 sphere cut at z=0.3, move a vertex at z=−0.9 by 0.01. The recompute counter is unchanged, and `sections_changed` is empty.
  - Unit test `edit_touching_plane_recomputes`: move a crossed vertex. The counter goes up by 1, and the result equals from-scratch.
  - Unit test `vertex_moved_across_plane_recomputes`: move a vertex at z=0.25 (not crossed) to z=0.35. This is criterion 3.
  - Unit test `edge_flip_near_plane_recomputes`: rewrite one crossed triangle's vertex ids via `triangles_changed`. This is criterion 2.
  - Property test `sections_match_from_scratch`, in `tests/sections_properties.rs`. Model it on `fields_properties.rs`.
    - Two meshes (a sphere and a wavy terrain) with origins around 1000.
    - Random sequences of: plane moves, placement changes, vertex moves (near and far from the plane), triangle add and remove, undo and redo, mode switches among Eager, Lazy and Manual, `compact`, selection changes, and fill toggles.
    - After `update_sections()` (ignoring `StaleBvh` under Manual, then calling `update(id)` and retrying), every entry is `Current`, and `section()` equals a from-scratch slice of the same mesh with the same plane and placement.
    - Compare topology fields exactly. Compare points within `1e-9·magnitude`.
    - Use a fixed `ProptestConfig` seed source: `PROPTEST_RNG_SEED` honoured, with a default deterministic config in CI. This follows follow-up #33's advice.
- [ ] **Step 2: Run** `cargo test -p fast-mesh-scene --test sections_properties` and the unit tests. Expected: the far-edit test FAILS, because Task 5 recomputes on every change.
- [ ] **Step 3: Implement the rule** in the section hook processing. Sides are computed with the same side rule as `slice`, using the entry's local plane.
- [ ] **Step 4: Run** `cargo test --workspace`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(scene): recompute sections only when a change reaches the plane`.

### Task 7: Benchmarks and docs

**Files:**
- Modify: `crates/fast-mesh-query/benches/query.rs`: slice benches.
- Modify: `crates/fast-mesh-scene/benches/scene.rs`: drag, far edit, fill and gap.
- Modify: `docs/benchmarks.md`: new "Sections" section.
- Modify: the spec, adding "§12 Implementation notes" with rulings and measured numbers.
- Modify: `crates/fast-mesh-query/src/lib.rs` and `crates/fast-mesh-scene/src/lib.rs` module docs. `CONTRIBUTING.md` changes only if the crate layout description lists modules.

**Interfaces:** consumes everything above; produces nothing new.

- [ ] **Step 1: Add benches:**
  - `slice/terrain_1M`, `slice/sphere_1M`, `slice/terrain_10M`, each at 3 plane positions;
  - `section_drag_1M`: 64 poses sweeping the plane across a 1M terrain, re-slicing per pose (Eager);
  - `section_far_edit_1M`: moving 100 vertices far from the plane;
  - `section_fill_1M_sphere`;
  - `section_gap_two_1M`.
  - Reuse the existing mesh generators in the bench files; add a `uv_sphere` generator to the bench if none exists.
- [ ] **Step 2: Run** `cargo bench -p fast-mesh-query --bench query -- slice` and `cargo bench -p fast-mesh-scene --bench scene -- section`. 10M may use `--sample-size 10`. Record the numbers. Target: about 1 ms for a 1M slice. If it is far above that, report it; do not tune silently.
- [ ] **Step 3: Write docs.**
  - `benchmarks.md` gets a table with hardware noted (i5-13600K), following the format of the existing sections.
  - Spec §12 records deviations, rulings and the measured numbers.
- [ ] **Step 4: Verify** with fmt, clippy and `cargo test --workspace`. Expected: all green.
- [ ] **Step 5: Commit** `bench(sections): add slice, drag, fill and gap benchmarks` and `docs(sections): record benchmarks and implementation notes`.
