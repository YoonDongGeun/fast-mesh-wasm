# fast-mesh queries, part b (Phase 3b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mesh-vs-mesh queries under a relative rigid transform: `intersects`, `mesh_distance`, two-sided `triangles_near`, and `intersecting_pairs`. Also a faster ordered traversal for the Phase 3a queries (issue #17).

**Architecture:**
- `fast-mesh-bvh` gains a key-ordered traversal. It uses a fixed-size on-stack array and a stackless fallback, plus a subtree traversal.
- `fast-mesh-query` gains:
  - `Rigid` transforms;
  - exact triangle–triangle intersection through the `robust` crate's `orient3d`/`orient2d`;
  - f64 triangle–triangle distance;
  - a simultaneous two-BVH traversal;
  - the four public queries on top of that traversal.

**Tech Stack:** Rust 2024 (MSRV 1.88, toolchain 1.99.0), `robust` 1.x (new, runtime), proptest 1.12, and criterion 0.8 with no default features.

**Spec:** `docs/superpowers/specs/2026-10-16-fast-mesh-query-pairs-design.md`. Phase 3a's spec, `2026-10-16-fast-mesh-query-design.md`, still binds unchanged behaviour.

## Global Constraints

**Coordinates and errors**
- All query inputs and outputs are mesh-local `f32`. B's results are expressed in A's space. Transformed B vertices are computed in f64 when used and are never stored.
- Every pair query checks `MeshMismatch`/`StaleBvh` for both meshes, after input validation. This matches the 3a order: validate, then `check`.

**Memory and traversal**
- No allocation except caller `out` vectors, through `Appender`, which truncates back on `OutOfMemory`.
- Traversals never allocate on the heap and never fail. They use a fixed array on the call stack. When it is full, they finish the overflowing work with the stackless walk.

**Exactness and ties**
- "Exact" covers only the intersect/touch decision, which uses `robust::orient3d`/`orient2d` on the f64 coordinates. Positive distances are f64 approximations. A distance is compared to its bound as `dist <= f64::from(bound)`.
- Ties go to the smallest id:
  - for `mesh_distance` with distance > 0, the smallest `(triangle_a, triangle_b)`;
  - with distance 0, the search stops at the first intersecting pair found.

**Rigid**
- The rotation must be orthonormal with determinant +1, within `1e-6`. Anything else, or any non-finite value, is `InvalidInput`.

**Dependencies**
- New runtime dependency: `robust` (georust, MIT OR Apache-2.0), latest 1.x, added to `fast-mesh-query` only.

**Code and commits**
- `///` docs on every public item.
- clippy `-D warnings`, fmt, and `cargo doc` with no warnings.
- The wasm32 build must pass.
- UTF-8 with LF. Edit with the editor tools.
- Commits use Conventional Commits and end with `Co-Authored-By: <model> <noreply@anthropic.com>`.

## Review Focus

1. **Exact contact poses:** a B vertex lying exactly on an A face (exactly representable coordinates) must give `intersects == true` and `mesh_distance == 0`. Test: `vertex_exactly_on_face_touches` (Task 5).
2. **Coplanar triangles:** triangles that overlap or only touch at an edge must be classified exactly. Test: `coplanar_cases` (Task 2).
3. **Rotations from f32 sources:** a matrix converted from f32, or produced by a renderer, with about 1e-7 error must be accepted, while 1e-3 error is rejected. Test: `rigid_validation` (Task 2).
4. **Empty meshes:** an empty mesh, or one with all triangles removed, on either side must give `false`, `None` or nothing, without panicking. Test: `empty_side_returns_nothing` (Task 5).
5. **Same mesh twice:** passing the same mesh and BVH as both A and B behaves as two coincident copies, so every triangle intersects itself and `intersects == true`. This is documented. Self-intersection excluding neighbours is a non-goal. Test: `same_mesh_twice_is_two_copies` (Task 5).

---

### Task 1: Key-ordered traversal and the Phase 3a port (issue #17)

**Files:**
- Modify: `crates/fast-mesh-bvh/src/arena.rs` (add the subtree-rooted stackless ordered walk used as the fallback)
- Modify: `crates/fast-mesh-bvh/src/node.rs` (new public API and tests)
- Modify: `crates/fast-mesh-query/src/ray.rs:~140-160` and `crates/fast-mesh-query/src/closest.rs:~70-100` (port to the new API)

**Interfaces:**
- Produces, in `fast-mesh-bvh`:
  - `Bvh::traverse_by_key<K: Copy + PartialOrd>(&self, key: impl FnMut(&Aabb) -> Option<K>, visit: impl FnMut(NodeId, &NodeView, K) -> bool)`
    - Visits nodes depth-first. Of two children, the one with the smaller key is visited first.
    - `key` is evaluated once per node, when its parent is expanded. The root's key is computed once as well.
    - A child whose key is `None` is never visited.
    - `visit` receives the node's key and returns whether to descend. Leaves ignore the return value.
    - Equal keys: the left child first.
  - `Bvh::traverse_from(&self, root: NodeId, visit: impl FnMut(NodeId, &NodeView) -> bool)`: pre-order over the subtree at `root`, stackless.
  - `pub const TRAVERSE_STACK: usize = 64;`, the fixed stack depth, public so tests can reason about overflow.
- `traverse_ordered` stays public and unchanged, for compatibility.

- [ ] **Step 1: Write the failing tests in `node.rs`**

- `by_key_matches_recursive_reference`:
  - Grid mesh with `max_leaf_size: 2`, and key `|b| Some(b.min[0] + 2.0 * b.min[1])`.
  - The visit order (ids, and the keys passed in) equals a recursive reference that sorts each node's two children by key, left first on ties.
  - Every node is visited once.
- `by_key_none_culls_subtree`: a key returning `None` for one inner, non-root child. None of that child's descendants are visited, and all other nodes are.
- `key_evaluated_once_per_node`: count `key` calls with nothing pruned. The count equals `node_count()`.
- `deep_tree_overflow_falls_back`:
  - A strip mesh of 4,000 triangles in a line, `max_leaf_size: 1`, `Quality::Fast`.
  - Assert `depth() > TRAVERSE_STACK`.
  - The visited set equals that of `traverse`.
  - The order is still nearest-first within each fallback subtree. Assert at least that a `visit` returning `false` at the first node prunes everything.
- `traverse_from_visits_subtree_only`: for an inner node, the visited ids equal its subtree from the recursive reference.

- [ ] **Step 2: Run and confirm compile failure.** `cargo test -p fast-mesh-bvh --lib node`

- [ ] **Step 3: Implement**

The main loop:

```text
stack: [(u32 node, K key); TRAVERSE_STACK], len
push (root, key(root))   // skip if None
while let Some((n, k)) = pop():
    if !visit(n, view(n), k) or n is leaf: continue
    kl = key(left); kr = key(right)
    order = (left first unless kr < kl)
    push second (if Some), then first (if Some)
```

When a push finds the stack full, finish that child's subtree at once with the stackless ordered walk rooted at it:
- the order closure compares children's `key(..)`, accepting the extra evaluations only in this overflow path;
- `visit` gets the node's key, which is recomputed there.

Then continue the main loop. Document that the order inside an overflowing subtree is still nearest-first, but `key` may be re-evaluated there.

- [ ] **Step 4: Port the 3a queries to `traverse_by_key`**

- `raycast*`: `key = ray_box(...)` (entry `t`). `visit` prunes with the passed `t`, without recomputing.
- `closest_point*`: `key = Some(box_dist2)`, or `None` beyond `max²`. `visit` prunes with the passed value.
- `raycast_all` keeps pruning by `max_t` only.

Results must be identical. Every existing `fast-mesh-query` unit and property test must pass unchanged.

- [ ] **Step 5: Measure.**

Run `cargo bench -p fast-mesh-query -- 1M/raycast` and `-- 1M/closest` on the parent commit and on this one. Use a temporary worktree for the parent. Record both numbers in the task report. `docs/benchmarks.md` is updated in Task 6.

- [ ] **Step 6: Verify and commit**

Run `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --all --check`.

```bash
git commit -m "perf(bvh,query): key-ordered traversal with a fixed stack (#17)"
```

---

### Task 2: `Rigid` and exact triangle–triangle intersection

**Files:**
- Modify: `crates/fast-mesh-query/Cargo.toml` (`robust = "<latest 1.x>"`), `Cargo.lock`
- Create: `crates/fast-mesh-query/src/rigid.rs`, `crates/fast-mesh-query/src/tritri.rs`
- Modify: `crates/fast-mesh-query/src/lib.rs` (`mod rigid; mod tritri; pub use rigid::Rigid;`, crate doc bullet)

**Interfaces:**
- Produces:
  - `pub struct Rigid { pub rotation: [[f64; 3]; 3], pub translation: [f64; 3] }` (`Debug, Clone, Copy, PartialEq`). It maps B's local coordinates to A's: `p_a = rotation * p_b + translation`, with `rotation` row-major (`rotation[i][j]` is row i, column j).
  - `Rigid::IDENTITY`.
  - `Rigid::inverse(&self) -> Rigid`: transpose, and `-Rᵀt`.
  - `Rigid::apply(&self, p: [f32; 3]) -> [f64; 3]`.
  - `pub(crate) fn validate(&self) -> Result<()>`:
    - `InvalidInput` if any value is non-finite;
    - `InvalidInput` if `|RᵀR − I|` (max over entries) exceeds `1e-6`, or `|det − 1| > 1e-6`.
  - `pub(crate) fn transform_box(&self, b: &Aabb) -> Aabb`:
    - center `c' = R c + t`, half-extent `h'ᵢ = Σⱼ |Rᵢⱼ| hⱼ`, computed in f64;
    - rounded outward to f32, i.e. `min` via `next_down` after the cast if the cast rounded up, and `max` mirrored;
    - an empty box stays empty.
  - `pub(crate) type P3 = [f64; 3];`
  - `pub(crate) fn tri_tri_intersect(a: [P3; 3], b: [P3; 3]) -> bool`: closed, so touching counts. Exact through `robust`.

- [ ] **Step 1: Write the failing tests**

`rigid.rs`:
- `rigid_validation`:
  - `IDENTITY` is OK.
  - A rotation of 30° about z, computed in f64, is OK.
  - The same matrix with each entry rounded through `f32` is OK.
  - A matrix with one entry off by `1e-3` gives `InvalidInput`.
  - A reflection (det −1) gives `InvalidInput`.
  - A NaN translation gives `InvalidInput`.
- `inverse_round_trips`: for the 30° rotation plus a translation, `inverse().apply(...)` of an applied point is within `1e-12` of the original.
- `transform_box_is_conservative`:
  - Use 1,000 seeded random rotations and boxes.
  - All 8 transformed corners of the original box lie inside `transform_box`.
  - Under `IDENTITY` the box is unchanged.

`tritri.rs`, with `A = [[0,0,0],[1,0,0],[0,1,0]]`:
- `crossing_and_separated`:
  - `B = [[0.2,0.2,-1],[0.2,0.2,1],[0.8,0.1,0]]` pierces A: `true`.
  - B shifted by `+5` in x: `false`.
- `touching_cases`, all `true`:
  - B's vertex exactly on A's interior, `[0.25,0.25,0]`, with the rest of B above;
  - B's edge lying on A's edge (collinear overlap) in a different plane;
  - B's vertex exactly on A's vertex.
- `one_ulp_apart_is_separate`: B's lowest vertex at `z = f64::from(f32::MIN_POSITIVE)`, with the others above, gives `false`. This also holds with z set to the next f64 above 0.
- `coplanar_cases`, with B in `z = 0`:
  - overlapping → `true`;
  - sharing only an edge → `true`;
  - sharing only a vertex → `true`;
  - disjoint, with a gap `1e-9` → `false`;
  - B containing A → `true`.
- `degenerate_cases`:
  - a segment B piercing A → `true`;
  - a point B on A → `true`;
  - a point B off A → `false`;
  - both degenerate and crossing collinear segments → `true`.
- `symmetric`: for 2,000 seeded random pairs, including near-touching ones built by snapping a vertex to the other's plane, `tri_tri_intersect(a, b) == tri_tri_intersect(b, a)`.

- [ ] **Step 2: Run and confirm compile failure.** `cargo test -p fast-mesh-query --lib rigid tritri`

- [ ] **Step 3: Implement**

Use the Guigue–Devillers test, taking every sign from `robust::orient3d` (via `robust::Coord3D { x, y, z }`).

Coplanar pairs are those where all of B's vertex signs relative to A are 0, or the reverse. For them:
- project to 2D by dropping the axis of the largest component of the plane normal;
- if the triangle is degenerate, use the dominant axis of the other triangle's normal, or of a non-zero edge cross;
- test, with `robust::orient2d`:
  - edge–edge intersection, closed;
  - a vertex of either triangle inside the other, closed.

Degenerate triangles, where a triangle's own `orient3d` plane test is undefined because all its points are collinear:
- reduce them to segment–triangle tests, using sign tests of the segment endpoints against the triangle plane plus `orient3d` edge-side tests;
- reduce a point to point–triangle containment.

Every branch decides by predicate signs only. No epsilon is used.

- [ ] **Step 4: Run tests and lint, then commit**

```bash
git commit -m "feat(query): add rigid transforms and exact triangle-triangle intersection"
```

---

### Task 3: f64 triangle–triangle distance

**Files:**
- Modify: `crates/fast-mesh-query/src/tri.rs`: make `closest_on_triangle` generic over a private `Real` trait implemented for `f32` and `f64`, keeping the f32 call sites and every existing test unchanged.
- Modify: `crates/fast-mesh-query/src/tritri.rs`

**Interfaces:**
- Consumes: `tri_tri_intersect`, `P3` (Task 2).
- Produces:
  - `pub(crate) fn closest_on_triangle<T: Real>(q: [T; 3], p: [[T; 3]; 3]) -> TriClosest<T>`. `TriClosest` becomes generic. The f32 users keep working through inference.
  - `pub(crate) struct TriTriDist { pub dist2: f64, pub pa: P3, pub pb: P3 }`
  - `pub(crate) fn tri_tri_distance(a: [P3; 3], b: [P3; 3]) -> TriTriDist`:
    - if `tri_tri_intersect`, then `dist2 = 0`, and `pa`/`pb` are a common point (any vertex or edge crossing found, or a midpoint — document which);
    - otherwise the minimum over the 6 vertex–triangle results and the 9 segment–segment results;
    - ties go to the first in a fixed order (the vertices of a against b, then of b against a, then edge pairs in `i*3 + j` order).

- [ ] **Step 1: Write the failing tests**
- `distance_matches_dense_sampling`:
  - 500 seeded random separated pairs.
  - `sqrt(dist2)` is at most the minimum over dense barycentric samples (a 40×40 grid on each triangle, all pairs) plus `1e-9`.
  - It is at least the true minimum minus `1e-9`, where the true minimum comes from an independent brute-force formula. Use a separate segment–segment and point–triangle implementation in the test, with a direct quadratic minimization written fresh.
- `parallel_edges`: two triangles whose closest features are parallel edges 0.5 apart give `dist2 == 0.25` within `1e-12`.
- `intersecting_is_zero`: the crossing pair from Task 2 gives `dist2 == 0`, and `pa == pb` lies on both triangles within `1e-12`.
- `points_are_on_triangles`: for the random pairs, `|pa - pb|² == dist2` within `1e-9`, and each point lies on its triangle within `1e-9` (barycentric check).
- In `tri.rs`: all existing tests are unchanged and pass, plus `f64_matches_f32_on_simple_cases`, which runs the seven Voronoi cases in f64 with exact expectations.

- [ ] **Step 2: Fail. Step 3: Implement. Step 4: Run `cargo test -p fast-mesh-query` and lint. Step 5: Commit.**

The `Real` trait has the minimal arithmetic and constants `closest_on_triangle` uses, including `SLIVER` as a value of `T`. Use segment–segment closest points from Ericson §5.1.9, in f64, with the parallel case handled.

```bash
git commit -m "feat(query): add f64 triangle-triangle distance"
```

---

### Task 4: Simultaneous two-BVH traversal

**Files:**
- Create: `crates/fast-mesh-query/src/pair.rs`

**Interfaces:**
- Consumes: `Bvh::root`, `node`, `traverse_from` (Task 1), `Rigid::transform_box` (Task 2).
- Produces:
  - `pub(crate) const PAIR_STACK: usize = 128;`
  - `pub(crate) enum Step { Descend, Skip, Stop }`
  - `pub(crate) fn traverse_pairs(a: &Bvh, b: &Bvh, xf: &Rigid, key: impl FnMut(&Aabb, &Aabb) -> Option<f64>, visit: impl FnMut(&NodeView, &NodeView, f64) -> Step)`
    - `key` receives A's box and B's box already moved into A's space (`transform_box`), and returns `None` to cull a pair.
    - Pairs are visited depth-first, smaller key first.
    - `visit` is called when a pair is popped, with the pair's key:
      - for a leaf–leaf pair it does the triangle work and returns `Skip` or `Stop`;
      - for any other pair, `Descend` expands it, `Skip` drops it, and `Stop` ends the traversal.
    - Expansion splits whichever of the two nodes is inner and has the larger box surface area (B's box measured in A space). If only one is inner, that one is split.
    - `key` is evaluated once per created pair. The root pair's key is evaluated once as well.

- [ ] **Step 1: Write the failing tests** (two small `wavy` meshes; `build` from `test_util`)

- `all_leaf_pairs_with_overlapping_boxes`:
  - Use `key = |ba, bb| ba.overlaps(bb).then_some(0.0)` and a visit that records leaf pairs.
  - The set of `(tri_a, tri_b)` from those leaves, filtered by an exact triangle-box overlap after transform, equals the brute-force set of all pairs whose transformed boxes overlap.
  - Use several `Rigid`s: identity, the 30° rotation, and a translation.
- `stop_ends_traversal`: returning `Stop` at the first leaf pair means `visit` is never called again.
- `nearest_first_order`:
  - With `key = box–box distance`, the sequence of keys at popped pairs is non-decreasing within each parent's expansion.
  - The first leaf pair visited has the minimum key among all leaf pairs.
- `overflow_fallback_is_complete`:
  - Two strip meshes of 4,000 triangles, `max_leaf_size: 1`, `Quality::Fast`, so the depths exceed `PAIR_STACK / 2`.
  - With an always-overlapping key, the recorded leaf pairs equal the full cross product of the leaves whose boxes overlap.
  - Check it against a nested `traverse` reference.

- [ ] **Step 2: Fail. Step 3: Implement.**

Use a fixed array of `(u32, u32, f64)` of size `PAIR_STACK`.

On overflow, finish the pair `(na, nb)` with:

```text
a.traverse_from(na, |ida, va| { box test vs nb's transformed box via key(...).is_some();
    if leaf: b.traverse_from(nb, |idb, vb| { key check; if leaf: visit(va, vb, k) ...; true }) ; true })
```

That is, a nested stackless walk that calls `visit` on leaf–leaf pairs only and honours `Stop` with an early-exit flag. Ordering inside the fallback is not guaranteed. Document this. Queries stay correct because they re-check bounds in `visit`.

- [ ] **Step 4: Run tests and lint. Step 5: Commit.**

```bash
git commit -m "feat(query): add simultaneous traversal of two BVHs"
```

---

### Task 5: Mesh-vs-mesh queries

**Files:**
- Create: `crates/fast-mesh-query/src/pairs.rs`
- Modify: `crates/fast-mesh-query/src/lib.rs` (`pub use pairs::*;`, crate doc bullet)

**Interfaces:**
- Consumes: `check`, `Appender` (3a); `Rigid` (Task 2); `tri_tri_intersect`, `tri_tri_distance` (Tasks 2–3); `traverse_pairs`, `Step` (Task 4).
- Produces, every function taking `(mesh_a: &Mesh, bvh_a: &Bvh, mesh_b: &Mesh, bvh_b: &Bvh, xf: &Rigid, ...)`:
  - `pub fn intersects(..) -> Result<bool>`
  - `pub struct MeshDistance { pub distance: f32, pub point_a: [f32; 3], pub point_b: [f32; 3], pub triangle_a: TriangleId, pub triangle_b: TriangleId }`. `point_b` is in A's space; document this.
  - `pub fn mesh_distance(.., max_distance: f32) -> Result<Option<MeshDistance>>`
  - `pub fn triangles_near(.., margin: f32, out_a: &mut Vec<TriangleId>, out_b: &mut Vec<TriangleId>) -> Result<()>`
    - Both vectors are appended to, ascending and unique within the appended ranges.
    - On error, both are truncated back.
  - `pub fn intersecting_pairs(.., out: &mut Vec<(TriangleId, TriangleId)>) -> Result<usize>`, sorted by `(a, b)`.

- [ ] **Step 1: Write the failing tests**

Oracles loop over all alive pairs, using `xf.apply` on B's vertices and the same `tri_tri_*`:
- `intersects_matches_brute_force`
- `intersecting_pairs_matches_brute_force`
- `triangles_near_matches_brute_force`, at margins 0, 0.05 and 0.5, both sides.
- `mesh_distance_matches_brute_force`. When separated, `(triangle_a, triangle_b)` and the distance must equal the oracle's (minimum `dist2`, ties to the smallest pair). When intersecting, the distance is 0 and the returned pair intersects.

Each oracle test uses two wavy meshes, 6×6 and 5×5, with B flipped (rotated 180° about x) and placed:
- 0.3 above, so separated;
- 0.0 above, so touching;
- −0.3, so overlapping.

Run under several rotations: identity, 30°, 90° about z (exact), and a random one.

Plus these tests:
- `vertex_exactly_on_face_touches`: B is a single triangle with a vertex at `[0.25, 0.25, 0]` on a flat A, the rest of B above, exact coordinates. `intersects` is true, `mesh_distance` is `Some(0.0)`, and `triangles_near(margin = 0)` reports the A triangle and the B triangle.
- `cross_query_consistency`: for separated poses with `d = mesh_distance(...)`, `triangles_near(margin = d * (1 + 1e-6))` contains `triangle_a` in `out_a` and `triangle_b` in `out_b`.
- `empty_side_returns_nothing`: with A or B empty, or all triangles removed and updated, the queries give `false`, `None`, empty outputs and `Ok(0)`.
- `same_mesh_twice_is_two_copies`: the same mesh and BVH as A and B with `IDENTITY` gives `intersects == true`. `intersecting_pairs` contains every `(t, t)`. The doc states this behaviour.
- `invalid_input`:
  - each `Rigid` failure from Task 2;
  - negative or NaN `margin`;
  - negative or NaN `max_distance`;
  - all of these give `InvalidInput`, before `check`.
- `stale_and_mismatch_on_either_side`: `StaleBvh` or `MeshMismatch` when either side's BVH is stale or belongs to another mesh.
- `oom_restores_outputs`: `fail_allocations_after(Some(0))` for `intersecting_pairs`, and `Some(1)` for `triangles_near` with `out_b` failing. Both vectors are restored.

- [ ] **Step 2: Fail. Step 3: Implement.**

The keys:
- `intersects`, `intersecting_pairs`: the key is `Some(0.0)` if the boxes overlap. `intersects` returns `Stop` on the first intersecting pair.
- `mesh_distance`: the key is the box–box distance, `None` if above the current best or `max_distance`. `visit` prunes when `key > best`. At leaves it computes `tri_tri_distance`, accepting the smaller `dist2`, or an equal one with a smaller `(a, b)`. It returns `Stop` once `dist2 == 0`.
- `triangles_near`: the key is `Some(d)` if the box distance is at most `margin`. At leaves, a pair is near when it intersects, or when `tri_tri_distance` gives `sqrt(dist2) <= margin` (compared in f64). Near pairs push `a` to `out_a` and `b` to `out_b`. At the end, both appended ranges are sorted and deduplicated (`Appender::appended` / `truncate_appended`).

Document the temporary duplicate peak.

- [ ] **Step 4: Run tests and lint. Step 5: Commit.**

```bash
git commit -m "feat(query): add mesh-vs-mesh intersection, distance and proximity queries"
```

---

### Task 6: Property tests, benchmarks and docs

**Files:**
- Modify: `crates/fast-mesh-query/tests/properties.rs` (new property test; reuse its edit machinery)
- Modify: `crates/fast-mesh-query/benches/query.rs`
- Modify: `docs/benchmarks.md`
- Modify: `docs/superpowers/specs/2026-10-16-fast-mesh-query-pairs-design.md` (Status "Implemented (Phase 3b)", plus "Implementation notes" for any deviation)
- Modify: `docs/superpowers/specs/2026-10-16-fast-mesh-query-design.md` (note that #17 replaced the no-stack rule)
- Modify: `CONTRIBUTING.md`: change the crate's layout line to "Raycast, closest-point, region and mesh-vs-mesh queries".

- [ ] **Step 1: Write the property test** `pair_queries_match_brute_force_after_edits`:
  - 64 cases by default, using the existing `cases()` helper so `PROPTEST_CASES` works.
  - Two meshes, each edited by random steps, with `Bvh::update` after each step.
  - A random `Rigid`: a random unit quaternion, converted to a matrix in f64, plus a random translation that keeps the meshes near each other.
  - All four queries checked against the Task 5 oracle rules, including `cross_query_consistency`.
  - Both qualities, `max_leaf_size` 1 to 8.
  - The oracles live in this file, as for 3a.
- [ ] **Step 2: Run** `cargo test -p fast-mesh-query --test properties`, then `PROPTEST_CASES=2000 cargo test --release -p fast-mesh-query --test properties`. Report the counts actually run.
- [ ] **Step 3: Write the benches.**
  - Two 1M terrains (`terrain(708)`), B flipped 180° about x and translated so the surfaces touch in places.
  - Each iteration changes the rotation by a small angle about z, cycling through 64 precomputed `Rigid`s.
  - The benches:
    - `intersects`;
    - `mesh_distance` (no `max_distance`);
    - `triangles_near`, with the margin equal to 1% of the terrain height range;
    - `intersecting_pairs`, with B pushed 0.5 into A.
  - Print the result sizes once.
  - Add 10M with `FAST_MESH_BENCH_LARGE`.
- [ ] **Step 4: Run the benches and record** them in a new `docs/benchmarks.md` section, "Mesh-vs-mesh queries". Also update the Phase 3a table with the Task 1 numbers, marked as after #17, and keep the before numbers in one line.
  - If any target in the spec's §1 is missed by more than 2×, say so in the doc and the report. Those targets are a few ms for `intersects` and `mesh_distance` at 1M vs 1M, and contact-proportional `triangles_near`.
- [ ] **Step 5: Full verification.** Run all of these:
  - `cargo fmt --all --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo build --release --target wasm32-unknown-unknown -p fast-mesh-wasm`
  - `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p fast-mesh-bvh -p fast-mesh-query`
  - `cargo bench -p fast-mesh-query --no-run`
- [ ] **Step 6: Commit.**

```bash
git commit -m "test(query): add pair-query property tests, benchmarks and docs"
```
