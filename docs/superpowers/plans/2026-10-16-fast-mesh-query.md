# fast-mesh queries, part a (Phase 3a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Single-mesh raycast, closest-point and region queries over the Phase 2 BVH, in a new crate `fast-mesh-query`.

**Architecture:** `fast-mesh-bvh` gains a read-only node API (`NodeId`, `NodeView`, stackless `traverse` and `traverse_ordered`). `fast-mesh-query` builds every query on that API, with per-triangle tests in `tri.rs`. Each query checks the BVH is current, never allocates except into the caller's output vector (fallibly), and is tested against a brute-force scan that uses the same per-triangle test.

**Tech Stack:** Rust 2024 (MSRV 1.88, toolchain 1.99.0), proptest 1.12, criterion 0.8 (no default features).

**Spec:** `docs/superpowers/specs/2026-10-16-fast-mesh-query-design.md`

## Global Constraints

- All query inputs and outputs are mesh-local `f32` coordinates. No `f64` origin handling in this crate.
- `fast-mesh-query` depends only on `fast-mesh-store` and `fast-mesh-bvh`. No new third-party runtime dependencies.
- No query allocates except growing the caller's `out` vector, through fallible reservations. On `OutOfMemory`, `out` is truncated back to its length before the call.
- Traversals use no stack and cannot fail (parent links, like `Arena::traverse`).
- Ties are broken by the smallest triangle id, so results are deterministic.
- Every public item has `///` docs. `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --all --check` stay clean, and `cargo doc` has no warnings.
- Files are UTF-8 with LF line endings. Edit them with the editor tools, not Python rewrites, so non-ASCII characters (→, ², γ) survive.
- Commit messages follow Conventional Commits and end with `Co-Authored-By: <the model that wrote the commit> <noreply@anthropic.com>`.

## Review Focus

1. A ray starting exactly on a triangle with `min_t = 0` must hit it at `t = 0` (closed range). Test: `ray_starting_on_the_surface_hits_at_zero` (Task 3).
2. An axis-aligned ray whose origin lies exactly on a box face (`0 × ∞` in the slab test) must not lose the hit or produce NaN. Test: `axis_aligned_ray_on_a_box_face_still_hits` (Task 3).
3. A non-unit `dir` scales `t`: a direction twice as long gives half the `t` and the same point. Test: `t_is_in_units_of_dir` (Task 3).
4. An empty mesh, or one whose triangles were all removed, returns `None` or 0 results and never panics. Tests: `empty_mesh_returns_nothing` in Tasks 3, 4 and 5.
5. A mesh made only of zero-area triangles still has a closest point (on an edge or vertex), and is never hit by rays. Test: `degenerate_only_mesh` (Task 4).

---

### Task 1: BVH node API

**Files:**
- Modify: `crates/fast-mesh-bvh/src/arena.rs` (add `traverse_ordered` next to `traverse`)
- Create: `crates/fast-mesh-bvh/src/node.rs` (public `NodeId`, `NodeKind`, `NodeView`, and the `impl Bvh` methods below)
- Modify: `crates/fast-mesh-bvh/src/lib.rs` (`mod node; pub use node::{NodeId, NodeKind, NodeView};`, plus a crate-doc bullet)
- Modify: `crates/fast-mesh-bvh/src/tree.rs:201-219` (re-implement `query_aabb` on `traverse`, add the doc line and `debug_assert!`)

**Interfaces:**
- Produces:
  - `pub struct NodeId(u32)`: `Clone, Copy, PartialEq, Eq, Hash, Debug`. Its field is private.
  - `pub enum NodeKind<'a> { Inner { left: NodeId, right: NodeId }, Leaf { triangles: &'a [TriangleId] } }`
  - `pub struct NodeView<'a> { pub bounds: Aabb, pub kind: NodeKind<'a> }`
  - `Bvh::root(&self) -> Option<NodeId>`
  - `Bvh::node(&self, id: NodeId) -> NodeView<'_>`
  - `Bvh::parent(&self, id: NodeId) -> Option<NodeId>`
  - `Bvh::traverse(&self, visit: impl FnMut(NodeId, &NodeView) -> bool)`
  - `Bvh::traverse_ordered(&self, first_is_left: impl Fn(&Aabb, &Aabb) -> bool, visit: impl FnMut(NodeId, &NodeView) -> bool)`
  - `Bvh::mesh_uid(&self) -> u64`: the uid of the mesh the BVH was built for. `fast-mesh-query` needs it to tell `MeshMismatch` from `StaleBvh`.

- [ ] **Step 1: Write the failing tests in `crates/fast-mesh-bvh/src/node.rs` (`#[cfg(test)] mod tests`)**

Use a grid mesh like `update.rs`'s `grid(n)` (copy it), built with `max_leaf_size: 2`.

- `traverse_visits_every_node_once`:
  - Collect the ids from `traverse(|_, _| true)`.
  - Assert the count equals `node_count()` and there are no duplicates.
  - Assert the first id is `root()`.
- `leaves_hold_every_alive_triangle_once`: concatenate the `Leaf { triangles }` slices; sorted, they equal `0..triangle_count`.
- `parent_agrees_with_children`:
  - For every inner node, `parent(left) == Some(id)` and `parent(right) == Some(id)`.
  - `parent(root) == None`.
- `ordered_traversal_follows_first_is_left`:
  - With `first_is_left = |_, _| false`, the visit order equals the pre-order of the tree with every node's children swapped.
  - Compute the expected order with a recursive helper over `node()`.
  - It visits every node once.
- `ordered_traversal_prunes`: returning `false` for the root visits exactly one node.
- `empty_tree_has_no_root`: an empty mesh gives `root() == None`, and both traversals call `visit` zero times.

- [ ] **Step 2: Run them and confirm they fail to compile**

Run: `cargo test -p fast-mesh-bvh --lib node`
Expected: compile errors (`NodeId` not found).

- [ ] **Step 3: Implement `Arena::traverse_ordered` in `arena.rs`, and the public API in `node.rs`**

`traverse_ordered` has the same stackless walk as `traverse`, with one difference: the child order is computed per node. It must stay pure, because it is recomputed on the way back up.

```rust
// at an inner node `cur`, arriving from its parent (first arrival):
let left_first = first_is_left(&nodes[a].bounds, &nodes[b].bounds);
let (first, second) = if left_first { (a, b) } else { (b, a) };
// visit(cur) decides descent; descend to `first`.
// arriving back from `first`: go to `second`.
// arriving back from `second`: go to parent (stop if cur == root).
// "back from first" is detected by recomputing (first, second) and comparing `from`.
```

- `node.rs` maps `Node` to `NodeView`. A leaf's `triangles` is `&self.arena.refs[node.leaf_range()]`.
- `root()` is `None` when `self.root == NONE`.
- `parent()` maps `NONE` to `None`.
- `mesh_uid()` returns `self.mesh_uid`.

- [ ] **Step 4: Re-implement `query_aabb` on `traverse`**

Add this doc line: "The BVH must be current for `mesh` ([`Bvh::is_current`]); a stale tree may return wrong results or panic." Add `debug_assert!(self.is_current(mesh), "query_aabb on a stale BVH")`.

- [ ] **Step 5: Run the whole crate's tests**

Run: `cargo test -p fast-mesh-bvh`
Expected: all pass. The existing `query_aabb` tests and property test are unchanged.

Watch for one failure: if an existing test calls `query_aabb` on a stale tree, the `debug_assert!` fires. Fix such a test by updating the BVH first, not by removing the assert.

- [ ] **Step 6: Lint and commit**

Run: `cargo clippy -p fast-mesh-bvh --all-targets -- -D warnings`
Expected: no warnings.

```bash
git add crates/fast-mesh-bvh
git commit -m "feat(bvh): add a read-only node API with stackless ordered traversal"
```

---

### Task 2: `fast-mesh-query` crate, errors and the ray–triangle test

**Files:**
- Modify: `Cargo.toml` (workspace dependency `fast-mesh-bvh = { path = "crates/fast-mesh-bvh", version = "0.1.0" }`)
- Create: `crates/fast-mesh-query/Cargo.toml`: copy `crates/fast-mesh-bvh/Cargo.toml`. Set the description to "Raycast, closest-point and region queries for editable triangle meshes (part of fast-mesh-wasm)", keywords to `["raycast", "mesh", "geometry", "bvh", "3d"]`, and categories to `["graphics", "algorithms"]`. Dependencies: `fast-mesh-store.workspace = true` and `fast-mesh-bvh.workspace = true`. Keep the same dev-dependencies, and `[[bench]] name = "query"`.
- Create: `crates/fast-mesh-query/src/lib.rs`, `src/error.rs`, `src/tri.rs`
- Create: `crates/fast-mesh-query/benches/query.rs` with an empty `criterion_main!` stub, so the crate builds. Task 6 fills it.

**Interfaces:**
- Consumes: `Bvh::is_current`, `Bvh::mesh_uid` (Task 1), `Mesh::uid`.
- Produces:
  - `pub enum QueryError { MeshMismatch, StaleBvh, InvalidInput(&'static str), OutOfMemory }` with `Display` and `std::error::Error`. `pub type Result<T> = std::result::Result<T, QueryError>;`
  - `pub(crate) fn check(mesh: &Mesh, bvh: &Bvh) -> Result<()>`: returns `MeshMismatch` if `bvh.mesh_uid() != mesh.uid()`, else `StaleBvh` if `!bvh.is_current(mesh)`.
  - `pub(crate) struct Appender<'a, T> { out: &'a mut Vec<T>, start: usize }`:
    - `new(out) -> Self`
    - `push(&mut self, v: T) -> Result<()>`: a fallible `try_reserve(1)`. On failure it truncates `out` to `start` and returns `OutOfMemory`.
    - `appended(&mut self) -> &mut [T]`: the range `start..`.
    - `count(&self) -> usize`.
  - `#[cfg(test)] pub(crate) fn fail_allocations_after(n: Option<usize>)`: a thread-local hook checked by `Appender::push`. It mirrors `fast-mesh-store`'s `fail_allocations_after`.
  - In `tri.rs`:
    - `pub enum Cull { None, Back, Front }`. `Default` is `None`. It is re-exported from `lib.rs`.
    - `pub(crate) struct RayPre`: per-ray constants. It holds `origin`, `dir`, `inv_dir: [f32; 3]`, `kx, ky, kz: usize` and `sx, sy, sz: f32`.
    - `RayPre::new(origin: [f32; 3], dir: [f32; 3]) -> RayPre`.
    - `pub(crate) struct TriHit { pub t: f32, pub barycentric: [f32; 3] }`.
    - `pub(crate) fn ray_triangle(pre: &RayPre, p: [[f32; 3]; 3], cull: Cull, min_t: f32, max_t: f32) -> Option<TriHit>`.

- [ ] **Step 1: Write the failing tests in `tri.rs` and `lib.rs`**

`tri.rs` uses the triangle `T = [[0,0,0],[1,0,0],[0,1,0]]`, counter-clockwise seen from +z, so its normal is +z.

- `hits_the_interior`:
  - The ray from `[0.25, 0.25, 1]` with dir `[0, 0, -1]` hits at `t == 1.0`.
  - The barycentric weights sum to 1 within `1e-6`, and equal `[0.5, 0.25, 0.25]` within `1e-6`.
- `misses_outside_and_behind`:
  - The ray from `[2, 2, 1]` down misses.
  - The ray from `[0.25, 0.25, -1]` with dir `[0, 0, -1]` misses, because `t < 0`.
- `culling`:
  - The downward ray above hits the front face, so `Cull::Back` keeps the hit and `Cull::Front` drops it.
  - The upward ray from `[0.25, 0.25, -1]` with dir `[0, 0, 1]` is the reverse: `Cull::Front` keeps it and `Cull::Back` drops it.
- `range_is_closed`:
  - For the downward ray with `t = 1`, `min_t = 1.0` hits and `max_t = 1.0` hits.
  - `max_t = 0.999` misses.
- `parallel_and_degenerate_miss`:
  - The ray `[0.25, 0.25, 0]` with dir `[1, 0, 0]` lies in the plane and misses.
  - The degenerate triangle `[[0,0,0],[1,1,0],[2,2,0]]` is never hit by the downward ray at `[1, 1, 1]`.
- `shared_edge_is_watertight`:
  - Use two triangles sharing the edge `(1,0,0)–(0,1,0)`: `T`, and `[[1,0,0],[1,1,0],[0,1,0]]`.
  - For 1,000 rays aimed at points `s·(1,0,0) + (1−s)·(0,1,0)` for `s` in `(0,1)`, at least one of the two triangles is hit.
  - Also check the shared vertex `(1,0,0)`, where at least one of the triangles touching it must be hit.

`lib.rs` tests:

- `check_reports_mismatch_then_staleness`:
  - A BVH built for mesh A, checked against mesh B, gives `MeshMismatch`.
  - After an edit to A without `update`, `check` gives `StaleBvh`.
  - After `update`, it gives `Ok`.
- `appender_restores_length_on_oom`: with `out = vec![7]`, `fail_allocations_after(Some(1))` makes the second push fail, and afterwards `out == vec![7]`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `cargo test -p fast-mesh-query`
Expected: compile errors (the items don't exist yet).

- [ ] **Step 3: Implement**

`ray_triangle` is the watertight test (Woop, Benthin and Wald 2013). The algorithm is fixed, so here it is:

```text
kz = argmax |dir[i]|; kx = (kz+1)%3; ky = (kx+1)%3; if dir[kz] < 0 { swap(kx, ky) }
sx = dir[kx]/dir[kz]; sy = dir[ky]/dir[kz]; sz = 1/dir[kz]           // in RayPre::new
A,B,C = p[i] - origin
Ax = A[kx] - sx*A[kz]; Ay = A[ky] - sy*A[kz]   (same for B, C)
U = Cx*By - Cy*Bx; V = Ax*Cy - Ay*Cx; W = Bx*Ay - By*Ax
if U == 0 || V == 0 || W == 0 { recompute U, V, W in f64 from the f32 Ax..Cy and cast back }
inside = (U >= 0 && V >= 0 && W >= 0) || (U <= 0 && V <= 0 && W <= 0)   // narrow to one sign per Cull
det = U + V + W; if det == 0 { miss }
T = U*(sz*A[kz]) + V*(sz*B[kz]) + W*(sz*C[kz])
t = T / det; hit iff min_t <= t <= max_t; barycentric = [U, V, W] / det   // corner 0 ↔ U
```

The `culling` test pins which sign of `det` is a front face. `RayPre::new` also stores `inv_dir = 1/dir` (zeros become ±∞), which Task 3 uses.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p fast-mesh-query`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

Run: `cargo clippy -p fast-mesh-query --all-targets -- -D warnings`

```bash
git add Cargo.toml Cargo.lock crates/fast-mesh-query
git commit -m "feat(query): add fast-mesh-query with errors and a watertight ray-triangle test"
```

---

### Task 3: Raycast

**Files:**
- Create: `crates/fast-mesh-query/src/ray.rs`
- Modify: `crates/fast-mesh-query/src/lib.rs` (`mod ray; pub use ray::*;`, crate docs)

**Interfaces:**
- Consumes: `check`, `Appender`, `RayPre`, `ray_triangle`, `Cull` (Task 2); `Bvh::traverse_ordered`, `NodeView`, `NodeKind` (Task 1).
- Produces:
  - `pub struct Ray { pub origin: [f32; 3], pub dir: [f32; 3] }`
  - `pub struct RayOptions { pub min_t: f32, pub max_t: f32, pub cull: Cull }`. `Default` is `0.0`, `f32::INFINITY` and `Cull::None`.
  - `pub struct RayHit { pub triangle: TriangleId, pub t: f32, pub point: [f32; 3], pub barycentric: [f32; 3], pub normal: [f32; 3] }`
  - The functions:
    - `raycast(mesh: &Mesh, bvh: &Bvh, ray: &Ray, opts: &RayOptions) -> Result<Option<RayHit>>`
    - `raycast_any(mesh: &Mesh, bvh: &Bvh, ray: &Ray, opts: &RayOptions) -> Result<Option<RayHit>>`
    - `raycast_all(mesh: &Mesh, bvh: &Bvh, ray: &Ray, opts: &RayOptions, out: &mut Vec<RayHit>) -> Result<usize>`
    - `raycast_many(mesh: &Mesh, bvh: &Bvh, origins: &[[f32; 3]], dirs: &[[f32; 3]], opts: &RayOptions, out: &mut [Option<RayHit>]) -> Result<()>`
  - All four derive `Debug, Clone, Copy, PartialEq` where possible.
  - `pub(crate) fn ray_box(pre: &RayPre, b: &Aabb, max_t: f32) -> Option<f32>`: the entry `t`, clamped to `>= 0`, or `None`.

- [ ] **Step 1: Write the failing tests in `ray.rs`**

Fixtures: a wavy grid like `update.rs`'s `oom_tests::wavy(n)` (copy it), and a `brute_closest` / `brute_all` oracle. The oracles loop over the alive triangles, call `ray_triangle`, and pick results with the same rules as the queries.

- `closest_matches_brute_force`:
  - Use 500 rays from a fixed LCG seed: origins above the grid, directions downward with jitter.
  - The `(triangle, t)` of `raycast` equals the oracle's exactly.
  - Run it with `Quality::Sah` and `Quality::Fast`.
- `all_matches_brute_force_sorted`:
  - `raycast_all` gives the oracle's hit set, sorted by `(t, triangle)`.
  - The return value equals the number appended.
  - Pre-existing elements of `out` are kept.
- `any_hits_iff_closest_hits`: for the same rays, `raycast_any(..).is_some() == raycast(..).is_some()`, and an `any` hit is a real hit (`ray_triangle` agrees).
- `shared_edge_tie_takes_smallest_id`:
  - Aim a ray exactly at a shared edge of two grid triangles.
  - `raycast` returns the smaller id. `raycast_all` returns both, the smaller first.
- `hit_fields`:
  - For a downward ray at `[0.25, 0.25, 1]` onto `T` (Task 2's triangle as a one-triangle mesh): `point == [0.25, 0.25, 0]` and `normal == [0, 0, 1]`, both within `1e-6`.
  - The barycentric weights are as in Task 2.
- `t_is_in_units_of_dir`: dir `[0, 0, -2]` gives `t == 0.5` and the same point.
- `ray_starting_on_the_surface_hits_at_zero`: an origin on `T` with dir `[0, 0, -1]` hits at `t == 0.0`.
- `axis_aligned_ray_on_a_box_face_still_hits`:
  - Use a vertical triangle in the plane `x = 1`, and a ray from `[0, 0.25, 0.25]` with dir `[1, 0, 0]`.
  - Its origin lies on the y and z box faces' planes, and some `dir` components are zero.
  - It hits at `t == 1.0`.
- `empty_mesh_returns_nothing`:
  - On an empty mesh, `raycast`, `raycast_any` and `raycast_all` return `Ok(None)`, `Ok(None)` and `Ok(0)`.
  - The same holds after removing every triangle of a grid and updating.
- `invalid_input`: each gives `InvalidInput`:
  - zero `dir`;
  - NaN in `origin`;
  - `min_t = 2.0, max_t = 1.0`;
  - `min_t = NaN`;
  - `raycast_many` with 2 origins and 1 dir.
- `stale_and_mismatch_are_rejected`: every function returns `StaleBvh` after an un-updated edit.
- `raycast_many_matches_raycast`: elementwise equal to calling `raycast` once per ray.
- `raycast_all_oom_restores_out`: with `fail_allocations_after(Some(0))`, it returns `OutOfMemory` and `out` keeps its old length.

- [ ] **Step 2: Run them and confirm they fail**

Run: `cargo test -p fast-mesh-query ray`
Expected: compile errors.

- [ ] **Step 3: Implement**

`ray_box` is the slab test using `pre.inv_dir`. When an `inv_dir` component is infinite and the box face equals the origin coordinate, the product is NaN. Compute each slab with `min`/`max`, which ignore NaN, so that such a slab does not cull.

Scale the exit `t` by `1 + 2γ₃`, where `γ₃ = 3ε / (1 − 3ε)` and `ε = f32::EPSILON / 2`. A box is hit when `t_enter <= t_exit` and `t_enter <= max_t` and `t_exit >= min_t`.

Traversal order: `first_is_left(l, r)` is `entry(l) <= entry(r)`, with a missed box counting as `+∞`. The visit prunes:
- `raycast`: when `entry > best_t`. Ties are still visited, so the smaller id can win.
- `raycast_any`: everything once a hit exists.
- `raycast_all`: only by `max_t`.

For the hit fields:
- `point = origin + t·dir`;
- `normal = normalize(cross(p1 − p0, p2 − p0))`, computed only for the returned hit;
- `raycast_all` sorts `Appender::appended()` by `(t, triangle)` using `f32::total_cmp`.

Validate the input before `check`. `raycast_many` validates the slice lengths, then loops.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p fast-mesh-query`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
git add crates/fast-mesh-query
git commit -m "feat(query): add closest, any, all and batch raycasts"
```

---

### Task 4: Closest point

**Files:**
- Modify: `crates/fast-mesh-query/src/tri.rs` (add the point–triangle test)
- Create: `crates/fast-mesh-query/src/closest.rs`
- Modify: `crates/fast-mesh-query/src/lib.rs`

**Interfaces:**
- Consumes: `check` (Task 2); `traverse_ordered` (Task 1).
- Produces:
  - `pub enum Feature { Vertex(u8), Edge(u8), Face }`. `Edge(k)` joins corners `k` and `(k + 1) % 3`.
  - `pub(crate) struct TriClosest { pub point: [f32; 3], pub dist2: f32, pub barycentric: [f32; 3], pub feature: Feature }`
  - `pub(crate) fn closest_on_triangle(q: [f32; 3], p: [[f32; 3]; 3]) -> TriClosest`
  - `pub struct ClosestOptions { pub max_distance: f32 }`. `Default` is `f32::INFINITY`.
  - `pub struct ClosestHit { pub triangle: TriangleId, pub point: [f32; 3], pub distance: f32, pub barycentric: [f32; 3], pub feature: Feature }`
  - `closest_point(mesh: &Mesh, bvh: &Bvh, p: [f32; 3], opts: &ClosestOptions) -> Result<Option<ClosestHit>>`
  - `closest_point_many(mesh: &Mesh, bvh: &Bvh, points: &[[f32; 3]], opts: &ClosestOptions, out: &mut [Option<ClosestHit>]) -> Result<()>`
  - `pub(crate) fn box_dist2(q: [f32; 3], b: &Aabb) -> f32`

- [ ] **Step 1: Write the failing tests**

In `tri.rs`, using `T`:

- `voronoi_regions`: one query point per region, with the expected feature, point and barycentric weights within `1e-6`:
  - `[-1, -1, 0]` → `Vertex(0)`, point `[0,0,0]`;
  - `[2, -1, 0]` → `Vertex(1)`;
  - `[-1, 2, 0]` → `Vertex(2)`;
  - `[0.5, -1, 0]` → `Edge(0)`, point `[0.5, 0, 0]`;
  - `[1, 1, 0]` → `Edge(1)`, point `[0.5, 0.5, 0]`;
  - `[-1, 0.5, 0]` → `Edge(2)`, point `[0, 0.5, 0]`;
  - `[0.25, 0.25, 3]` → `Face`, point `[0.25, 0.25, 0]`, `dist2 == 9`.
- `degenerate_triangle_uses_segments`:
  - For `[[0,0,0],[1,1,0],[2,2,0]]`, the query `[1, 0, 0]` gives point `[0.5, 0.5, 0]`, `dist2 == 0.5`, and an `Edge(_)` feature. The result is finite.
  - For `[[1,1,1]; 3]`, the query `[0, 0, 0]` gives `Vertex(0)` and `dist2 == 3`.

In `closest.rs`, with a `brute` oracle (minimum `dist2` over alive triangles, ties going to the smaller id):

- `matches_brute_force`:
  - 500 points from a fixed LCG seed, around and above the wavy grid.
  - Both qualities.
  - The `triangle` and `distance` (`== dist2.sqrt()`) match the oracle exactly.
- `max_distance_is_inclusive`:
  - A point exactly 2 units above a flat grid gets `Some` with `max_distance: 2.0`, and `None` with `1.999`.
- `degenerate_only_mesh`:
  - A mesh whose only triangles are zero-area gets `Some` from `closest_point`, with a finite distance.
  - `raycast` on it (Task 3) gives `None`.
- `empty_mesh_returns_nothing`: `Ok(None)`.
- `invalid_input`: each gives `InvalidInput`:
  - a NaN point;
  - `max_distance = -1.0`;
  - `max_distance = NaN`;
  - `closest_point_many` with mismatched lengths.
- `many_matches_single`: elementwise equal to calling `closest_point` per point.

- [ ] **Step 2: Run them and confirm they fail**

Run: `cargo test -p fast-mesh-query closest`
Expected: compile errors.

- [ ] **Step 3: Implement**

`closest_on_triangle` follows Ericson's `ClosestPtPointTriangle` (Real-Time Collision Detection §5.1.5). Each early return maps to a feature:
- regions A, B, C → `Vertex(0/1/2)`;
- AB → `Edge(0)`, BC → `Edge(1)`, CA → `Edge(2)`;
- the interior → `Face`.

When the face denominator is `0.0` or not finite, the triangle is degenerate. In that case take the best of the three point–segment results, with an edge clamped to its endpoint reporting the corresponding `Vertex`. The best is the smallest `dist2`, with ties going to the lower feature index.

For the traversal:
- `first_is_left(l, r)` is `box_dist2(q, l) <= box_dist2(q, r)`.
- The visit prunes when `box_dist2 > best_dist2`.
- `best_dist2` starts at `max_distance * max_distance`. A candidate is accepted when `dist2 < best`, or when `dist2 == best` and it has the smaller id. Before any hit, `dist2 <= max²` qualifies.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p fast-mesh-query`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
git add crates/fast-mesh-query
git commit -m "feat(query): add closest point with feature classification"
```

---

### Task 5: Region queries

**Files:**
- Modify: `crates/fast-mesh-query/src/tri.rs` (add box–triangle and sphere–triangle)
- Create: `crates/fast-mesh-query/src/region.rs`
- Modify: `crates/fast-mesh-query/src/lib.rs`

**Interfaces:**
- Consumes: `check`, `Appender` (Task 2); `closest_on_triangle` (Task 4); `Bvh::traverse` (Task 1).
- Produces:
  - `pub enum Region { Box(Aabb), Sphere { center: [f32; 3], radius: f32 } }`
  - `pub(crate) fn box_triangle(b: &Aabb, p: [[f32; 3]; 3]) -> bool`
  - `pub(crate) fn sphere_triangle(center: [f32; 3], radius: f32, p: [[f32; 3]; 3]) -> bool`
  - `triangles_in(mesh: &Mesh, bvh: &Bvh, region: &Region, out: &mut Vec<TriangleId>) -> Result<usize>`
  - `vertices_in(mesh: &Mesh, bvh: &Bvh, region: &Region, out: &mut Vec<VertexId>) -> Result<usize>`

- [ ] **Step 1: Write the failing tests**

In `tri.rs`:

- `box_triangle_cases`:
  - Overlap: box `[0.2,0.2,-1]..[0.3,0.3,1]` against `T` → `true`.
  - Separated by a box face axis: `[2,2,-1]..[3,3,1]` → `false`.
  - Separated by the triangle normal: `[0,0,0.1]..[1,1,1]` → `false`.
  - Separated only by an edge-cross axis: box `[0.6,0.6,-1]..[1,1,1]` against `T` → `false`. The box's x/y/z ranges each overlap the triangle's, so only the hypotenuse axis separates them.
  - Touching: `[1,0,0]..[2,1,1]` (shares the corner `(1,0,0)`) → `true`.
- `sphere_triangle_cases`:
  - Center `[0.25,0.25,1]`, radius `1.0` (tangent) → `true`.
  - Radius `0.999` → `false`.

In `region.rs`, with oracles that scan every alive triangle (`box_triangle` / `sphere_triangle`) and every alive vertex:

- `triangles_match_brute_force`:
  - 200 random boxes and 200 random spheres over the wavy grid, both qualities.
  - The sorted output equals the oracle's.
  - Also check that a triangle whose bounding box overlaps a box but whose surface does not is excluded.
- `vertices_match_brute_force`:
  - The same regions.
  - The output equals the alive vertices inside the region that are used by an alive triangle.
  - It is ascending and has no duplicates.
- `vertices_in_sorts_only_the_appended_range`: `out = vec![999]` before the call stays first, and only the appended part is sorted.
- `isolated_vertex_is_not_found`: a vertex added without triangles, inside the region, is absent. This is the documented limitation.
- `empty_box_returns_nothing`: `Aabb::EMPTY` gives 0.
- `empty_mesh_returns_nothing`: 0 for both functions.
- `invalid_input`: radius `-1.0`, radius NaN, and a NaN center give `InvalidInput`.
- `oom_restores_out`: both functions, with `fail_allocations_after(Some(0))`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `cargo test -p fast-mesh-query region`
Expected: compile errors.

- [ ] **Step 3: Implement**

`box_triangle` is Akenine-Möller's SAT with 13 axes: three box normals, the triangle normal, and nine edge crosses. Center the triangle on the box center. An axis separates only when the projected intervals are strictly disjoint (`>`), so touching counts as overlap. A zero cross-product axis is skipped.

`sphere_triangle(c, r, p)` is `closest_on_triangle(c, p).dist2 <= r * r`.

The traversal prunes nodes whose bounds do not overlap the region's bounding box. For a sphere that box is `center ± radius`.

`vertices_in`:
- pushes every corner inside the region (box: closed containment; sphere: `dist2 <= r²`) of every candidate triangle;
- then calls `appended().sort_unstable()` and dedups in place within the appended range. Use `slice::partition_dedup` if stable, otherwise a manual compaction followed by truncate.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p fast-mesh-query`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
git add crates/fast-mesh-query
git commit -m "feat(query): add exact box and sphere region queries"
```

---

### Task 6: Property tests, benchmarks and docs

**Files:**
- Create: `crates/fast-mesh-query/tests/properties.rs`
- Modify: `crates/fast-mesh-query/benches/query.rs`
- Modify: `docs/benchmarks.md` (new section "Queries (fast-mesh-query)")
- Modify: `.github/workflows/ci.yml:31` (add `- run: cargo bench -p fast-mesh-query --no-run` after the bvh line)
- Modify: `CONTRIBUTING.md`:
  - project layout: add `crates/fast-mesh-query/   Raycast, closest-point and region queries over a store mesh and its BVH.`
  - commands table: add a row for `cargo bench -p fast-mesh-query`
- Modify: `docs/superpowers/specs/2026-10-16-fast-mesh-query-design.md` (Status: "Implemented (Phase 3a)", plus a "Deviations" note if any)

**Interfaces:**
- Consumes: every public function from Tasks 3–5; `Bvh::build`, `Bvh::update`, `History`, transactions.

- [ ] **Step 1: Write the property tests**

`queries_match_brute_force_after_edits` (proptest, 64 cases):
- Start from a 6×6 wavy grid.
- Run steps of commit (move a vertex, add or remove triangles), undo and redo. Reuse the `Op`/`Step` shapes from `crates/fast-mesh-bvh/tests/properties.rs`, without compaction or replay.
- After each step, run `Bvh::update` and then 16 random rays, points, boxes and spheres.
- Each query must equal its brute-force oracle, with the same tie rules.
- Both build qualities, `max_leaf_size` from 1 to 8.
- The oracles live in this file. Unit-test oracles are private to their modules, so duplicate the few lines.

`rays_never_escape_a_closed_mesh` (proptest, 256 cases):
- Build a cube surface of side 2, each face a 4×4 grid of quads, with vertices welded through a map on integer coordinates. This makes a closed, watertight mesh.
- A ray from a random interior point (each coordinate in `(-0.9, 0.9)`) in a random non-zero direction must hit (`raycast` returns `Some`).
- Include directions aimed exactly at vertices and edge midpoints of the cube.

- [ ] **Step 2: Run them**

Run: `cargo test -p fast-mesh-query --test properties`
Expected: pass.

Then run: `PROPTEST_CASES=2000 cargo test --release -p fast-mesh-query --test properties`
Expected: pass. If `with_cases` overrides the environment variable (issue #8), temporarily raise the case count in the source, run, and revert. Report the count actually run.

- [ ] **Step 3: Write the benchmarks in `benches/query.rs`**

- Copy `terrain(n)` from `crates/fast-mesh-bvh/benches/bvh.rs`.
- Sizes: 1M (`n = 708`), and 10M (`n = 2237`) when `FAST_MESH_BENCH_LARGE` is set.
- Use a deterministic LCG to make 100,000 rays: origins at `z = 20` over the terrain, directions `(jx, jy, −1)` with jitter in `±0.2`.
- Benches, each over the whole batch with criterion throughput set to `Elements(100_000)` so results read per ray or per point:
  - `raycast`;
  - `raycast_any`;
  - `closest_point` for 100,000 points at random heights `z` in `[-5, 5]`.
- Benches for one sphere of radius 5 at the center:
  - `triangles_in`;
  - `vertices_in`.
- `sample_size(10)`.

- [ ] **Step 4: Run the benches and record them**

Run: `cargo bench -p fast-mesh-query`
Then run: `FAST_MESH_BENCH_LARGE=1 cargo bench -p fast-mesh-query`

Add to `docs/benchmarks.md` a section with:
- the machine line;
- a table with columns 1M and 10M, and rows: raycast per ray, raycast_any per ray, closest point per point, triangles_in and vertices_in (radius 5, with the result counts).

- [ ] **Step 5: Full verification**

Run all of these:
- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `cargo build --release --target wasm32-unknown-unknown -p fast-mesh-wasm`
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p fast-mesh-bvh -p fast-mesh-query`
- `cargo bench -p fast-mesh-query --no-run`

Expected: all succeed.

- [ ] **Step 6: Update the spec status and commit**

```bash
git add crates/fast-mesh-query docs .github/workflows/ci.yml CONTRIBUTING.md
git commit -m "test(query): add property tests, benchmarks and docs"
```
