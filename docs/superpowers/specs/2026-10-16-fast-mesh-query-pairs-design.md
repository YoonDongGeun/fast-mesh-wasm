# fast-mesh queries, part b (Phase 3b) design

- **Status:** Implemented (Phase 3b)
- **Date:** 2026-10-16
- **Parent specs:** `2026-10-14-fast-mesh-engine-design.md` §6, and `2026-10-16-fast-mesh-query-design.md` (Phase 3a, implemented). Where this document disagrees with them, it wins for Phase 3b.
- **Milestone:** Phase 3: Queries. Also resolves issue #17.

## 1. Intent

The owner's use case is to look at the gap and contact between two surfaces that touch, for example an upper and a lower dental arch. One mesh is moved or rotated interactively, and the degree of contact is shown as a smooth per-vertex color (a scalar field rendered with interpolation).

Most of the time the two surfaces touch somewhat or are slightly apart. Deep interpenetration is uncommon.

The per-vertex colour itself is Phase 5's job: distance fields and signed distance (SDF). Phase 3b supplies what makes that fast and exact while one mesh moves:

- where the two meshes are near each other;
- where they touch or intersect;
- their minimum distance.

All of these must work without rebuilding either BVH when the relative pose changes.

**Success** means the following, all while dragging:
- one mesh is rotated or moved;
- the query that answers "do they touch?" or "what is the minimum gap?" runs in a few milliseconds at 1M vs 1M triangles;
- the query that answers "which triangles of A are within `margin` of B" costs roughly in proportion to the contact area, not to the mesh size.

The owner's words were "the faster the better". The numbers above are my targets, not theirs.

## 2. Goals, non-goals, anti-goals

**Goals:**
1. Mesh-vs-mesh queries under a relative rigid transform:
   - `intersects`;
   - `mesh_distance` (minimum gap and the closest-point pair);
   - `triangles_near` (the triangles of A within `margin` of B);
   - `intersecting_pairs`.
2. Robust triangle–triangle intersection: touching and overlapping are never flipped by rounding, and coplanar triangles are handled.
3. Issue #17: replace the stackless ordered traversal with a faster one that still never allocates on the heap and never fails, and apply it to the Phase 3a queries.

**Non-goals (later phases or issues):**
- Intersection polylines, the outline of the penetration. The owner colours vertices and does not draw outlines. This becomes an issue.
- Signed distance, distance fields and colours: Phase 5.
- Scene broadphase over many meshes, and the transform API on `Mesh`: Phase 4.
- Self-intersection of a single mesh: the parent spec lists it as a non-goal.
- Non-uniform or uniform scale in the relative transform (see §4).
- JS bindings: Phase 8.

**Anti-goals** (failure even if it technically works):
- Rebuilding or refitting a BVH because the relative pose changed.
- A contact query whose answer flips between "touching" and "not touching" for the same input. Results must be deterministic, and exact for the given f64 coordinates.
- A per-frame cost proportional to mesh size when the contact area is small.

## 3. Constraints

These carry over from Phase 3a:
- every query checks `MeshMismatch` and `StaleBvh` for both meshes;
- input validation comes first;
- no allocation except into caller-provided `out` vectors, through fallible reservation with rollback;
- deterministic tie-breaking by the smallest id;
- property tests against brute-force oracles;
- criterion benches at 1M (and 10M with `FAST_MESH_BENCH_LARGE`);
- every public item has `///` docs;
- clippy `-D warnings`;
- the wasm32 build.

**The traversal rule changes** (issue #17). It was "traversals use no stack". It is now **"traversals never allocate on the heap and never fail"**.
- A traversal may use a fixed-size array on the call stack.
- When that array is full, the traversal must continue correctly by falling back to the existing stackless walk for the overflowing subtree.

**New dependency (my call):** the `robust` crate (georust, MIT OR Apache-2.0, no dependencies), for Shewchuk's adaptive `orient3d`/`orient2d`, pinned to the latest 1.x release. The alternative is vendoring the predicates, about 1,000 lines. That is more code to own, with no benefit. It goes into `fast-mesh-query` only.

## 4. Relative transform

```rust
/// Maps mesh B's local coordinates into mesh A's local coordinates: p_a = rotation * p_b + translation.
pub struct Rigid { pub rotation: [[f64; 3]; 3], pub translation: [f64; 3] }
impl Rigid {
    pub const IDENTITY: Rigid;
    pub fn inverse(&self) -> Rigid;
}
```

- **Rotation and translation only.** The owner moves and rotates a mesh and does not scale it. Without scale, distances are preserved and stay exact in meaning.
- **Validation:** the rotation must be orthonormal with determinant +1, within `1e-6` (my call). Otherwise the query returns `InvalidInput`. Non-finite values are `InvalidInput` too.
- **Converting the meshes' f64 origins** into the relative transform is the caller's job, and Phase 4's Scene does it.
- **Vertices:** B's vertices are transformed into A's space in f64 when a triangle pair is tested, never stored.
- **Exactness:** the exact predicates are exact for those transformed f64 coordinates. That is the meaning of "exact" here, and it is documented.
- **Boxes:** a B node box is moved into A space as a conservative AABB. The center is transformed and the half-extents are multiplied by `|rotation|` element-wise. It is computed in f64 and rounded outward.

## 5. Queries

All four take `(mesh_a, bvh_a, mesh_b, bvh_b, &Rigid, ...)` and work in A's local space.

```rust
pub fn intersects(..) -> Result<bool>;
pub struct MeshDistance { pub distance: f32, pub point_a: [f32; 3], pub point_b: [f32; 3], pub triangle_a: TriangleId, pub triangle_b: TriangleId }
pub fn mesh_distance(.., max_distance: f32) -> Result<Option<MeshDistance>>;
pub fn triangles_near(.., margin: f32, out_a: &mut Vec<TriangleId>, out_b: &mut Vec<TriangleId>) -> Result<()>;
pub fn intersecting_pairs(.., out: &mut Vec<(TriangleId, TriangleId)>) -> Result<usize>;
```

**`intersects`**
- True if any alive triangle of A intersects any alive triangle of B. The test is closed, so touching counts.
- It stops at the first pair found.

**`mesh_distance`**
- Returns the minimum distance between the surfaces, with a closest-point pair. `point_b` is in A's space, and its doc says so.
- The distance is 0 when they intersect. The points are then some common point.
- It returns `None` beyond `max_distance` (inclusive), or when either mesh is empty.
- For a positive distance, ties go to the smallest `(triangle_a, triangle_b)`.
- *My call:* once an intersecting pair is found, the distance is 0 and the search stops at that pair. Touching is the common case, and finding the smallest intersecting pair would cost as much as `intersecting_pairs`. The oracle then checks only that the distance is 0 and that the returned pair really intersects.

**`triangles_near`**
- Fills both sides in one call:
  - `out_a` gets each triangle of A within `margin` of B;
  - `out_b` gets each triangle of B within `margin` of A.
- Each triangle appears once, in ascending order.
- A `margin` of 0 means touching or intersecting.
- *My call:* a single two-sided call, so each pair is tested once in one coordinate space and the two sides agree pair for pair. Phase 5 colours both meshes. A second call with the meshes swapped and `Rigid::inverse()` could disagree at the rounding level.
- Candidates are appended, then sorted and deduplicated inside the appended ranges, like `vertices_in`, so no extra memory is used. The temporary peak is documented.
- This is the query Phase 5 uses to recompute only the vertices near the contact.

**`intersecting_pairs`**
- Appends every intersecting pair, sorted by `(a, b)`.

**Invalid input:** `max_distance` or `margin` that is negative or NaN, and an invalid `Rigid`, give `InvalidInput`.

## 6. Algorithms

**Simultaneous traversal**
- It starts from a pair of nodes. At each step it descends into the node with the larger box surface area, or the inner one if the other is a leaf.
- It prunes a pair when the boxes are disjoint after inflating by the current bound. The bound is `margin` for `triangles_near`, the best distance so far for `mesh_distance`, and 0 for the intersection queries.
- The pair stack is a fixed array. On overflow, a pair is finished by nested traversals: walk A's subtree stackless, and for each A leaf walk B's subtree stackless. This is correct, just slower.
- `mesh_distance` visits the nearer child pair first, ordered by box–box distance.

**Triangle–triangle intersection**
- Use the Guigue–Devillers or Möller test with `orient3d` signs from `robust`, so the classification is exact.
- When all six signs are zero (coplanar triangles), project onto the dominant axis of A's normal and use `orient2d` for an edge–edge and containment test.
- Degenerate triangles (segments or points) are handled through the same exact predicates. This means segment–triangle and point–triangle tests.
- Every decision is exact for the f64 inputs.

**Triangle–triangle distance**
- If the pair intersects (exact test), the distance is 0.
- Otherwise, take the minimum over:
  - the 6 vertex-to-triangle distances, using Phase 3a's `closest_on_triangle`, lifted to f64 or with f64 inputs;
  - the 9 edge–edge segment distances.
- This is computed in f64, and the result is returned as f32.

**Precision promises (my call)**
- "Exact" covers only the decision whether two triangles intersect or touch.
- Positive distances are f64 approximations. `margin` and `max_distance` are compared as `f64 distance <= f64::from(bound)`.
- Across queries, results agree within rounding. If `mesh_distance` returns `d > 0`, then `triangles_near` with `margin = d * (1 + 1e-6)` includes its `triangle_a` and `triangle_b`.
- Property tests assert exactly that, not bit equality across queries.

**Issue #17**
- Change `Bvh::traverse_ordered` (or add a sibling API; the builder's choice) so that:
  - the per-node work of the ordered queries is computed once per child, not about 5 times;
  - it keeps "no heap allocation, never fails" through the fixed array with stackless fallback.
- Port `raycast*` and `closest_point*` to it.
- Results must stay identical. All existing tests and property tests must pass unchanged.
- Re-measure the Phase 3a benches. The final review's stack-based prototype was about 2× faster.

## 7. Testing

**Brute-force oracles**
- They loop over all pairs of alive triangles with the same per-pair tests.
- `intersects`, `intersecting_pairs`, `triangles_near` and `mesh_distance` must match exactly, including ties.

**Property tests**
- Random edits on both meshes.
- Random rigid poses, with random rotations, including exactly axis-aligned ones.
- Poses that make the meshes touch exactly: B placed so that a vertex of B lies exactly on a face of A, in coordinates that are exactly representable.

**Exactness unit tests**
- Coplanar overlapping and coplanar touching triangles.
- Edge-on-edge contact.
- Vertex-on-face contact.
- Triangles separated by 1 ulp.
- Degenerate triangles.

**Stack overflow fallback**
- A test with a deliberately deep tree, e.g. a long strip with `max_leaf_size` 1, or a test-only small stack size.
- It must give the same results through the fallback.

**#17 regression:** all Phase 3a tests unchanged, plus the before/after bench numbers.

## 8. Benchmarks

- Two copies of the 1M terrain facing each other with a small gap, so they touch in places. B is rotated by a small angle that changes on every iteration.
- Measured:
  - `intersects`;
  - `mesh_distance`;
  - `triangles_near` with a margin about 1% of the terrain height;
  - `intersecting_pairs` with an overlap.
- 10M as well when `FAST_MESH_BENCH_LARGE` is set.
- Record the contact sizes (the number of pairs or triangles) next to the times.
- Before/after numbers for the Phase 3a queries (#17).

**Speed targets (my call):** the numbers in §1 are targets to measure and record, not merge gates. If a target is missed by more than 2×, the last task records it and a follow-up issue is filed. The phase is not blocked.

## 9. Left to the builder

- The pair-stack size and descend heuristic details.
- The exact form of the #17 traversal API.
- Module layout inside `fast-mesh-query`.
- The f64/f32 boundaries inside the distance code.
- Bench geometry details.

## 10. Implementation notes

Where the implementation differs from the text above:

- **Distance 0 (§5).** `mesh_distance` reports 0 for pairs that intersect or touch, and also for a positive gap that rounds to 0 in `f64`. The search stops at the first such pair. The oracle accepts a returned pair that intersects (decided exactly) or has a computed distance of 0.
- **Margin 0 (§5).** `triangles_near` with a `margin` of 0 finds the pairs that touch, intersect, or are closer than `f64` rounding resolves, the same rule as distance 0 above. A pair with a positive gap whose computed distance rounds to 0 counts as near, so it can be reported although `intersects`, which is exact, says `false` for it.
- **Pruning (§6).** Pairs are pruned when the box gap exceeds the bound plus a relative slack of 2^-30 times the coordinate magnitude, not at the bare bound, because triangle distances carry `f64` rounding that the box gap does not. The slack only adds work. `mesh_distance` prunes in both the traversal's `key` and its `visit`, so the stackless overflow fallback prunes too.
- **Property test oracle.** `tests/properties.rs` runs each pair of alive triangles as two one-triangle meshes through the public API. It compares `mesh_distance` by its `f32` distance and points, since the public API does not report the `f64` squared distance. So the smallest-pair tie rule on `f64` is covered by the unit tests, not the property test.
- **Benchmarks (§8).** B is a second `terrain(n)` mesh, flipped and lifted per pose to touch A at its closest place on a sample grid. Besides the four planned benchmarks there are "apart 0.3" variants of `intersects` and `mesh_distance`, the case the use case calls most common. The sample grid can miss the true closest place, so not every "touching" pose meets: at 1M, 22 of the 64 (about a third) are near misses with gaps of 8e-6 to 0.0027, and at 10M, 3 are. `mesh_distance` on touching poses is therefore also measured separately on the intersecting poses and on the near misses.
- **`mesh_distance` pre-pass (§5, §6).** `mesh_distance` first runs the `intersects` traversal. If it finds an intersecting pair, that pair is returned with distance 0 and its common point; otherwise the distance search runs. This was added after the first benchmark run showed 23 ms for touching meshes at 1M vs 1M: the two queries visited leaf pairs in the same order, but the distance search ran the full triangle distance on every pair and had no bound to prune by.
- **Per-pair lower bound (§6).** Near misses leave many leaf box pairs touching (key 0), so box pruning drops few of them. `mesh_distance`'s distance search and `triangles_near` therefore give each triangle pair a cheap lower bound first: the larger of the gap between the two triangles' `f64` boxes and the gap along each triangle's computed normal (how far the other triangle's corners lie beyond the slab the triangle spans along that normal, which bounds the distance for any direction). The normal is used only when `|n|² >= 1e-4 |u|² |v|²` for its edge vectors `u`, `v`, since a sliver's computed normal can point anywhere. A pair is skipped when its bound is beyond the current bound by the box-pruning slack, so results are unchanged. `triangles_near` also decides each remaining pair with an early-exit form of the triangle distance that stops at the first candidate within the margin and gives exactly the same answer.
- **Public `Rigid::apply` (§4).** `Rigid::apply` and `Rigid::validate` are public, so callers (Phase 5) can move B's points with the exact arithmetic the queries use, and map result points back with `apply` on `Rigid::inverse`.
- **Near-miss search (§6, issue #22).** The distance search and `triangles_near` build each triangle of a leaf pair in `f64` once, with its box and its slab along its normal, instead of once per triangle pair, and keep it while the traversal stays on that leaf. Each triangle's box is first tested against the union box of the other leaf's triangles; that gap is never above the per-pair box gap (exactly, in `f64`), so this skips only pairs the lower bound would skip. `mesh_distance` orders pairs by box gap plus `1e-12` times the distance between box centres, so that among the many touching leaf boxes of a near miss (gap 0) the most overlapping come first, and it still prunes on the box gap alone. Results are unchanged. Folding the intersection pre-pass into the distance search was measured and rejected: it slowed intersecting poses from about 0.4 ms to 1.5 to 2.2 ms at 1M. Instead, the intersection traversal (behind `intersects`, `intersecting_pairs` and the pre-pass) got the same per-leaf caching and an exact box prefilter. A pair is skipped only when the triangles' `f64` boxes are strictly disjoint, so touching pairs are kept.
- **Speed targets (§1, §8).** `mesh_distance` is bimodal on touching poses: about 0.25 ms at 1M vs 1M when the meshes intersect, and about 1.8 ms on the near misses (17 ms before the per-pair lower bound, 4.4 ms before issue #22). Apart 0.3 takes about 0.85 ms (1.3 ms before #22). `intersects` (about 0.3 ms), `triangles_near` (about 1.4 ms) and `intersecting_pairs` pushed 0.5 (about 4.4 ms, from 10.4 ms) meet their targets at 1M. Near misses, the common case, now meet the "few ms" target at 1M but are about 1.8× #22's goal of 1 ms. At 10M they took about 9.6 ms over 3 poses before the box prefilter (15.7 ms before #22); see `docs/benchmarks.md`.
