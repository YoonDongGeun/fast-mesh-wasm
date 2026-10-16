# fast-mesh queries, part a (Phase 3a) design

- **Status:** Implemented (Phase 3a)
- **Date:** 2026-10-16
- **Parent spec:** `2026-10-14-fast-mesh-engine-design.md`, section 6. This document settles what section 6 leaves open for single-mesh queries. Where they disagree, this document wins for Phase 3a.
- **Milestone:** Phase 3: Queries

## 1. Scope

Phase 3 is split in two parts, each with its own spec, plan and pull request:

- **3a (this document):** queries against one mesh: raycast, closest point, region.
- **3b (later):** mesh-vs-mesh distance and intersection (triangle pairs, intersection polylines), with robust orientation predicates.

The plane section listed under Phase 3 in the parent spec moves to Phase 6 (Sections), which is where it is used.

In scope for 3a:

1. A read-only node API in `fast-mesh-bvh`, so that other crates can traverse the tree (section 3).
2. A new crate `fast-mesh-query` with:
   - raycast: closest hit, any hit, all hits, and a batch variant;
   - closest point, with the feature (vertex, edge or face) it lies on, and a batch variant;
   - region queries: triangles and vertices inside a box or a sphere, with exact tests.
3. Brute-force oracles, property tests and criterion benchmarks, as in Phase 2.

Out of scope: 3b; transforms (Phase 4: a transformed mesh is queried by converting the operand into mesh-local space); JS bindings (Phase 8); multithreaded batches (later; the batch functions are where that would go).

## 2. Coordinates and precision

- Every query takes and returns **mesh-local `f32` coordinates**, the space the mesh stores positions in (relative to `Mesh::origin`) and the BVH is built in.
- World-space conversion, including the `f64` origin, belongs to the transform from world space to mesh-local space that Phase 4's Scene applies in `f64`. A Rust caller using `fast-mesh-query` directly subtracts `Mesh::origin` itself.

## 3. BVH node API (`fast-mesh-bvh`)

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeId(u32);

pub enum NodeKind<'a> {
    Inner { left: NodeId, right: NodeId },
    Leaf { triangles: &'a [TriangleId] },
}

pub struct NodeView<'a> {
    pub bounds: Aabb,
    pub kind: NodeKind<'a>,
}

impl Bvh {
    /// The root node, or `None` for an empty tree.
    pub fn root(&self) -> Option<NodeId>;
    pub fn node(&self, id: NodeId) -> NodeView<'_>;
    pub fn parent(&self, id: NodeId) -> Option<NodeId>;
    /// Depth-first pre-order, left child first. `visit` returns whether to
    /// descend into the node's children (ignored for leaves).
    pub fn traverse(&self, visit: impl FnMut(NodeId, &NodeView) -> bool);
    /// Depth-first, visiting first the child for which `first_is_left` says so.
    pub fn traverse_ordered(
        &self,
        first_is_left: impl Fn(&Aabb, &Aabb) -> bool,
        visit: impl FnMut(NodeId, &NodeView) -> bool,
    );
}
```

- The 32-byte `Node` layout stays private. Only `NodeView` is public, so the layout can change without breaking callers.
- Both traversals walk parent links, like the existing private `Arena::traverse`. They use no stack, never allocate and cannot fail, however deep the tree is.
- `traverse_ordered` recomputes `first_is_left` for a node when it returns from a child, to know whether the other child is still to be visited. `first_is_left` must therefore be a pure function of the two boxes.
- The second child is offered to `visit` like the first. A query whose best result improved inside the first child prunes the second there, with no extra mechanism.
- `query_aabb` is re-implemented on `traverse`. Its signature stays. It documents that the BVH must be current for the mesh, and it checks this with `debug_assert!`. The checked, exact box query is `fast_mesh_query::triangles_in`.

## 4. Crate `fast-mesh-query`

It depends on `fast-mesh-store` and `fast-mesh-bvh` only.

```
src/lib.rs      public API re-exports, QueryError, the checks shared by every query
src/ray.rs      raycast, raycast_any, raycast_all, raycast_many
src/closest.rs  closest_point, closest_point_many
src/region.rs   triangles_in, vertices_in
src/tri.rs      per-triangle tests: ray–triangle, point–triangle, box–triangle, sphere–triangle
```

Every query takes `(&Mesh, &Bvh, ...)` and first runs two O(1) checks:

- the BVH was built for another mesh: `QueryError::MeshMismatch`;
- the BVH does not match the mesh's current revision or epoch (`!bvh.is_current(mesh)`): `QueryError::StaleBvh`. A stale tree could return wrong results, or reference triangle slots that no longer exist.

Single-result queries return `Option<...>`. Multi-result queries append to a caller-provided `&mut Vec`. No query allocates on its own.

## 5. Raycast

```rust
pub struct Ray {
    pub origin: [f32; 3],
    pub dir: [f32; 3],
}

pub enum Cull { None, Back, Front }

pub struct RayOptions {
    pub min_t: f32,   // default 0
    pub max_t: f32,   // default f32::INFINITY (the parent spec's maxDistance)
    pub cull: Cull,   // default Cull::None; front faces are counter-clockwise
}

pub struct RayHit {
    pub triangle: TriangleId,
    pub t: f32,                 // point = origin + t * dir
    pub point: [f32; 3],
    pub barycentric: [f32; 3],  // weights of the triangle's corners 0, 1, 2
    pub normal: [f32; 3],       // unit geometric normal, following the winding
}

pub fn raycast(mesh: &Mesh, bvh: &Bvh, ray: &Ray, opts: &RayOptions) -> Result<Option<RayHit>>;
pub fn raycast_any(mesh: &Mesh, bvh: &Bvh, ray: &Ray, opts: &RayOptions) -> Result<Option<RayHit>>;
pub fn raycast_all(mesh: &Mesh, bvh: &Bvh, ray: &Ray, opts: &RayOptions, out: &mut Vec<RayHit>) -> Result<usize>;
pub fn raycast_many(
    mesh: &Mesh,
    bvh: &Bvh,
    origins: &[[f32; 3]],
    dirs: &[[f32; 3]],
    opts: &RayOptions,
    out: &mut [Option<RayHit>],
) -> Result<()>;
```

Semantics:

- **Direction:** `dir` need not be normalized. `t` is measured in units of `dir`, so it is a distance when `dir` is a unit vector.
- **Invalid input:** `InvalidInput` for:
  - a zero or non-finite `dir`;
  - a non-finite `origin`;
  - `min_t > max_t`, or `min_t` or `max_t` that is NaN;
  - in `raycast_many`, slices of different lengths.
- **Hit range:** a hit counts when `min_t <= t <= max_t`.
- **`raycast`:** the hit with the smallest `t`. Ties go to the smallest triangle id, so results are deterministic.
- **`raycast_any`:** some hit in range. It stops at the first one found, for occlusion tests.
- **`raycast_all`:** every hit, sorted by `(t, triangle)` and appended to `out`. It returns the number appended.
- **Shared edges:** a ray that crosses an edge shared by two triangles hits both. `raycast_all` reports both. This is documented.
- **Degenerate triangles:** a triangle with zero area is never hit.
- **Normal:** the normal is computed for the returned hit only, not for every candidate.

Algorithms:

- **Ray–triangle:** the watertight test of Woop, Benthin and Wald (2013). It runs in `f32`, and recomputes an edge function in `f64` only when it evaluates to exactly zero, as the paper does. A ray crossing a shared edge or vertex therefore never slips between triangles. Phase 5's thickness and inside tests rely on this.
- **Ray–box:** the slab test with a precomputed reciprocal direction (zero components become ±infinity). The exit distance is scaled by `1 + 2γ₃` (Ize 2013), so that `f32` rounding never culls a box the ray touches.
- **Traversal:** `traverse_ordered`, entering first the child whose box the ray enters first.
  - A node is pruned when its entry distance exceeds the best `t` found so far, or `max_t`.
  - `raycast_any` prunes everything once it has a hit.
  - `raycast_all` prunes only by `max_t`, and sorts the appended range at the end.
- **Batch:** `raycast_many` is a loop over `raycast` today.

## 6. Closest point

```rust
pub struct ClosestOptions {
    pub max_distance: f32,   // default f32::INFINITY
}

pub enum Feature {
    Vertex(u8),  // corner 0, 1 or 2
    Edge(u8),    // edge k joins corners k and (k + 1) % 3
    Face,
}

pub struct ClosestHit {
    pub triangle: TriangleId,
    pub point: [f32; 3],
    pub distance: f32,
    pub barycentric: [f32; 3],
    pub feature: Feature,
}

pub fn closest_point(mesh: &Mesh, bvh: &Bvh, p: [f32; 3], opts: &ClosestOptions) -> Result<Option<ClosestHit>>;
pub fn closest_point_many(
    mesh: &Mesh,
    bvh: &Bvh,
    points: &[[f32; 3]],
    opts: &ClosestOptions,
    out: &mut [Option<ClosestHit>],
) -> Result<()>;
```

- **Point–triangle:** the Voronoi-region method from Ericson, *Real-Time Collision Detection* §5.1.5. It reports the feature the point lies on, which Phase 5 needs for angle-weighted pseudonormals.
- **Degenerate triangles:** a triangle with zero area is treated as its three edge segments, so there is no division by zero. The feature is the edge or vertex found.
- **Result:** the hit with the smallest distance, with ties going to the smallest triangle id. It is `None` when nothing lies within `max_distance` (inclusive) or the mesh is empty.
- **Traversal:** `traverse_ordered`, entering first the child whose box is nearer to the point (squared distance). A node is pruned when its squared box distance exceeds the best squared distance so far. That starts at `max_distance²`.
- **Invalid input:** `InvalidInput` for a non-finite point, a negative or NaN `max_distance`, or slices of different lengths in `closest_point_many`.

## 7. Region

```rust
pub enum Region {
    Box(Aabb),
    Sphere { center: [f32; 3], radius: f32 },
}

pub fn triangles_in(mesh: &Mesh, bvh: &Bvh, region: &Region, out: &mut Vec<TriangleId>) -> Result<usize>;
pub fn vertices_in(mesh: &Mesh, bvh: &Bvh, region: &Region, out: &mut Vec<VertexId>) -> Result<usize>;
```

**`triangles_in`** appends every alive triangle that intersects the region (closed: touching counts), in no particular order. The test is exact, not by triangle bounding box:

- box: the separating-axis test of Akenine-Möller (13 axes);
- sphere: the closest point on the triangle (section 6) is within `radius`.

**`vertices_in`** appends every vertex inside the region (closed), in ascending id order, without duplicates:

- It visits the triangles whose boxes overlap the region's bounding box and tests their corners.
- It then sorts and deduplicates only the range it appended to `out`. No other memory is used.
- Limitation, documented: a vertex used by no alive triangle is not found, because the BVH holds triangles. Surface selections, such as brushes, are unaffected.

**Invalid input:** `InvalidInput` for a non-finite sphere center or a negative or NaN radius. An empty box (`min > max` on some axis) gives no results.

## 8. Errors

```rust
pub enum QueryError {
    MeshMismatch,
    StaleBvh,
    InvalidInput(&'static str),
    OutOfMemory,
}
```

- It implements `Display` and `std::error::Error`.
- Multi-result queries grow `out` with fallible reservations. If memory runs out they return `OutOfMemory` and truncate `out` back to its length before the call, so a failed query appends nothing. This follows the failure-atomicity rule of Phases 1 and 2.

## 9. Testing

**Per-triangle unit tests:**

- Ray:
  - through a corner, along a shared edge, and parallel to the plane;
  - zero-area triangles;
  - both culling modes;
  - `t` exactly at `min_t` and `max_t`.
- Point: each of the seven Voronoi regions (three vertices, three edges, the face), plus degenerate triangles.
- Box: separation on each kind of axis, and touching cases.
- Sphere: tangent cases.

**Brute-force oracles:** every query is compared with a linear scan over the alive triangles that uses the same per-triangle test. The arithmetic is identical, so `(t, triangle)` and `(distance, triangle)` must match exactly.

**Property tests**, following `crates/fast-mesh-bvh/tests/properties.rs`:

- random edits, undo and redo on a grid mesh, with `Bvh::update` after each step;
- after each step, random rays, points and regions are compared with the oracles;
- both build qualities.

**Watertightness property:** rays from random points inside a closed mesh (a subdivided cube) in random directions always hit.

**Errors:** `MeshMismatch`, `StaleBvh` (an edit without `Bvh::update`), each `InvalidInput` case, and `OutOfMemory` restoring `out`'s length.

**Node API:**

- `traverse` and `traverse_ordered` visit every node once when nothing is pruned.
- `traverse_ordered` honours `first_is_left`.
- `parent` agrees with the children.

## 10. Benchmarks

Criterion, `cargo bench -p fast-mesh-query`, on Phase 2's synthetic terrain meshes: 1M triangles by default, and 10M as well when `FAST_MESH_BENCH_LARGE=1`.

- `raycast` and `raycast_any`: 100,000 random rays, reported per ray.
- `closest_point`: random points near the surface.
- `triangles_in` and `vertices_in`: a sphere about the size of a sculpting brush.

Results are recorded in a new section of `docs/benchmarks.md`. CI adds `cargo bench -p fast-mesh-query --no-run`. CONTRIBUTING lists the crate in the project layout and the bench command.

## Implementation notes

Accepted deviations from the design above:

- `ray_box` clamps its entry distance to `min_t`, and a negative `min_t` is allowed.
- A NaN `max_t` gives `InvalidInput`, like a NaN `min_t`.
- A hit whose `t` or barycentric weights are not finite is rejected.
- "Zero-area" means `det == 0` after the f64 fallback of the ray-triangle test.
- Closest point uses a sliver band: when sin²θ ≤ 1e-5 or the sum is not finite, the three edge segments are also evaluated and the nearest result is kept.
- An infinite sphere radius gives `InvalidInput`.
- A NaN box gives no results.
