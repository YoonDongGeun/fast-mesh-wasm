# fast-mesh BVH (Phase 2) design

- **Status:** Implemented (Phase 2)
- **Date:** 2026-10-15
- **Parent spec:** `2026-10-14-fast-mesh-engine-design.md`, section 5. This document only settles what section 5 leaves open; where they disagree, this document wins for Phase 2.
- **Milestone:** Phase 2: BVH. Closes issue #6.

## 1. Scope

In scope:

1. A vertex → triangles incidence index in `fast-mesh-store` (issue #6).
2. A new crate `fast-mesh-bvh`:
   - build: binned SAH (`Quality::Sah`, default) and Morton/LBVH (`Quality::Fast`);
   - incremental update driven by the store's `ChangeSet`: refit, treelet rebuild, full rebuild;
   - SAH degradation tracking;
   - one query, `query_aabb`, used for correctness tests and by later phases.
3. Criterion benchmarks for build and update at 1M and 10M triangles. Their results set the default thresholds.

Out of scope (later phases): winding-number moment data (spec §5.4, moves to Phase 5 Fields); ray, closest-point and mesh-vs-mesh queries (Phase 3); transforms and broadphase (Phase 4); JS bindings (Phase 8); multithreaded builds (later; the code is structured for it).

Follow-up (done, issue #12): the incidence index follows topology edits instead of being rebuilt after each one (§2).

## 2. Store addition: incidence index

```rust
impl Mesh {
    /// Builds the vertex → triangles index if it is not cached.
    pub fn ensure_incidence(&mut self) -> Result<()>;
    /// Cached index; `None` until built or after a topology change.
    pub fn incidence(&self) -> Option<&Incidence>;
}

impl Incidence {
    /// Every alive triangle using `v`, each once, in ascending id order.
    /// Covers all fans of non-manifold vertices.
    pub fn triangles(&self, v: VertexId) -> &[TriangleId];
}
```

- Compressed rows: `offsets: Vec<u32>` (vertex slots + 1) and `triangles: Vec<u32>`. A triangle that repeats a vertex (`[a, a, b]`) is listed once for `a`.
- Built lazily, with fallible allocation.
- *Updated in issue #12:* once built, the index follows commits, rollbacks, undo and redo. The rows of the vertices an edit touches are rewritten into an overlay (`HashMap<VertexId, Vec<TriangleId>>`) that shadows the compressed rows. The old vertices of rewritten or removed triangles come from the triangle column's journal (commit) or from the mesh before writing (apply, revert). Inside a transaction that changed topology, `incidence()` returns `None` until the commit updates the index or the rollback restores the mesh.
- The index is dropped, and rebuilt compactly by the next `ensure_incidence`, when the overlay holds more than max(4,096, 1/8 of the compressed entries), when memory runs out while updating it, and on `compact`.
- Memory: about 4 bytes per triangle corner plus 4 per vertex slot, roughly 140 MB at 10M triangles. It exists only while something uses it.

## 3. Data layout

| Data | Format | 10M triangles, leaf ≤ 8 |
| --- | --- | --- |
| `nodes` | `Node { min: [f32; 3], max: [f32; 3], a: u32, b: u32 }`, 32 bytes | ≈ 80 MB |
| `parents` | `u32` per node (`NONE` for the root) | ≈ 10 MB |
| `refs` | triangle ids referenced by leaf ranges | 40 MB |
| `tri_leaf` | `u32` per triangle slot: owning leaf, or `NONE` (dead or not in the tree) | 40 MB |
| `treelets` | per treelet: root node, triangle count, SAH cost at build | negligible |

- Inner node: `a` = left child, `b` = right child. Leaf: `a` = first index into `refs`, `b` = `LEAF_BIT | count`.
- The tree contains every alive triangle exactly once, degenerate ones included. Dead slots are not referenced.
- Boxes are in mesh-local space (positions relative to the mesh origin), stored as `f32`. Transforms never touch the BVH.
- An empty mesh gives an empty tree (no root).

### 3.1 Garbage and compaction

Rebuilding part of the tree appends new nodes and refs and leaves the old ones unreachable ("garbage"). The BVH tracks how many entries are garbage. When garbage exceeds 50% of the live nodes or refs, `update` finishes with **storage compaction**: a depth-first copy of the reachable nodes and refs into fresh vectors. This is O(n) and does not re-split anything. The allocation is fallible, and on failure the old storage stays valid.

## 4. Build

- `Bvh::build(&Mesh, &BuildOptions) -> Result<Bvh, BvhError>`.
- **`Quality::Sah`:** top-down binned SAH with 16 bins by default (`sah_bins`, 4–64), over triangle centroids, using an explicit work stack (no recursion). If every centroid on the best axis is identical, split the range in half by index. A range of at most `max_leaf_size` triangles becomes a leaf.
- **`Quality::Fast`:** 30-bit Morton codes of centroids within the mesh bounds, sorted with a radix sort, then split top-down at the highest differing bit (falling back to the midpoint when all codes are equal).
- `max_leaf_size`: 1–16, default 8.
- Builds are deterministic: the same mesh and options give an identical tree.
- **Treelets:** after building, the nodes nearest the root whose subtree holds at most `treelet_size` triangles (default 4,096, chosen from benchmarks: see `docs/benchmarks.md`) become treelet roots. The nodes above them form the small top tree. A mesh smaller than `treelet_size` is a single treelet. Treelets are the unit of partial rebuild and of degradation tracking.
- **Treelet re-split (implementation deviation):** insertions make treelets grow, so after a treelet is rebuilt and holds more than 2 × `treelet_size` triangles it is split by the same rule, applied to its subtree only (the topmost subtrees with at most `treelet_size` triangles, or single leaves). The extra treelet capacity is reserved before the tree is modified; if the reservation turns out too small for an unusually unbalanced subtree, the split is skipped (the tree stays correct).
- Subtrees are built independently, so a later multithreaded build can split work at treelet roots without changing the layout.

## 5. Incremental update

```rust
bvh.update(&mut mesh, &changes) -> Result<UpdateReport, BvhError>   // &mut: may build the mesh's incidence cache
```

`changes` is the `ChangeSet` of every commit, undo or redo since the BVH last matched the mesh. Several sets can be merged by the caller (the Scene in Phase 4). The BVH records the mesh uid and the mesh revision it matches. `update` rejects another mesh (`MeshMismatch`) and then records the new revision. `bvh.is_current(&mesh)` compares revisions (and epochs, below); `bvh.revision()` returns the recorded one.

Implementation additions:

- **Already current:** when the recorded revision equals the mesh's, `update` returns `Unchanged` without looking at `changes`, so applying a change set twice is harmless. Independently, an added triangle that is already in the tree is treated as removed from its current treelet first, so it can never be inserted twice.
- **Compaction (epoch):** `Mesh::epoch()` changes whenever `Mesh::compact` renumbers slots. The BVH records the epoch at each full build. When it differs, `update` rebuilds everything (`Strategy::Full`) whatever the `ChangeSet` says, and manual `refit` and `rebuild_region` also do a full rebuild instead of their partial work.

Algorithm:

1. **Dirty set.** Collect the triangles to refit: those touching `vertices_moved`, found through `Mesh::incidence()` (built on demand), plus the rewritten triangles (`triangles_changed`).
2. **Full-rebuild check.** If added + removed + changed + refit triangles exceed `full_rebuild_fraction` × the alive triangle count (default 0.30), rebuild everything and stop. *As implemented:* the dirty count is `max(removed, added) + moved`, where `removed` and `added` both include the changed triangles and `moved` excludes triangles that are rebuilt anyway, so a remove-and-re-add of the same triangles counts once.
3. **Removals.** For each removed or changed triangle, take it out of its leaf (swap-remove inside the leaf's ref range) and mark the leaf's treelet for rebuild.
4. **Insertions.** For each added or changed triangle, walk the top tree choosing the child whose box grows least, until reaching a treelet root, and mark that treelet for rebuild with the triangle queued.
5. **Treelet rebuilds.** For each marked treelet, gather its triangles (current leaves plus queued) and build a new subtree with the build options. The subtree's nodes and refs are appended, then linked into the parent. The old entries become garbage.
6. **Refit.** Recompute boxes for the leaves of dirty triangles, and walk to the root, updating each ancestor once (sort and deduplicate per level). Also refit the ancestors of rebuilt treelets. *As implemented:* the leaves of moved triangles are deduplicated, then each leaf is refit once with one ancestor walk that stops early at the first box that does not change, instead of a per-level sort and deduplicate.
7. **Degradation.** For every treelet touched by a refit, recompute its SAH cost. If it exceeds `degradation_factor` × its build-time cost (default 1.5), rebuild it as in step 5 and refit its ancestors.
8. **Compaction** if the garbage threshold (§3.1) is passed.

`UpdateReport { strategy: Strategy::{Unchanged, Refit, Partial, Full}, refit_nodes, rebuilt_treelets, compacted }`.

Manual control, as in the parent spec §5.3: `bvh.refit(&mesh, &triangles)`, `bvh.rebuild_region(&mesh, &triangles)` (rebuilds the treelets containing them), `bvh.rebuild(&mesh)`.

Failure atomicity: every fallible allocation happens before the tree is modified, or the update falls back to building a complete new tree and swapping it in. On `OutOfMemory` the BVH is left as it was before the call (still matching the old revision).

Defaults (`full_rebuild_fraction`, `degradation_factor`, `treelet_size`, `max_leaf_size`) are confirmed or adjusted from benchmark results before the phase closes, and the chosen values are recorded in `docs/benchmarks.md`.

Manual `refit` and `rebuild_region` do not change the revision the BVH records; `update`, `build` and `rebuild` do. Full builds reserve an estimated node count and grow on demand; treelet rebuilds reserve their worst case (with at least 1/8 headroom) before mutating.

## 6. Query

```rust
bvh.query_aabb(&mesh, min: [f32; 3], max: [f32; 3], out: &mut Vec<TriangleId>)
```

Appends every alive triangle whose bounding box overlaps the query box (closed intervals), in unspecified order. It is exact against a brute-force scan of triangle boxes, which makes it the correctness oracle for tests. Precise triangle–box tests belong to Phase 3.

## 7. Errors and API surface

`BvhError { MeshMismatch, InvalidOptions(&'static str), OutOfMemory, Store(StoreError) }` with `Display`. Options are validated in `build` and `update`. *Deviation:* the `Store(StoreError)` variant was added for store errors other than `OutOfMemory` (which maps to `BvhError::OutOfMemory`), such as a failure while building the incidence index.

Public types: `Bvh`, `BuildOptions`, `Quality`, `UpdatePolicy`, `UpdateReport`, `Strategy`, `BvhError`, plus read-only inspection (`node_count()`, `byte_size()`, `root_bounds()`, `depth()`, `treelet_count()`, `revision()`) for benchmarks and debugging.

## 8. Testing

- **Unit tests:**
  - build on an empty mesh, a single triangle, identical centroids (forces the fallback split), degenerate triangles, and a mesh with dead slots;
  - each update strategy is chosen as specified;
  - mesh mismatch is rejected.
- **Invariants after every operation:**
  - each parent box contains its children's boxes;
  - each leaf box contains its triangles' boxes;
  - every alive triangle appears in exactly one leaf, and `tri_leaf` agrees;
  - no dead triangle is referenced;
  - parent links are consistent;
  - the garbage count matches the unreachable entries.
- **Property tests** (as in phase 1):
  - random edits through store transactions, with undo and redo, mesh compaction, replayed change sets, and `rebuild_region` on a stale tree before the update;
  - after each step, `update` with the step's `ChangeSet`, check the invariants, and compare `query_aabb` against brute force for random boxes;
  - a fresh `build` must give the same query results;
  - both qualities and several leaf sizes.
- **Incidence:** after random edits, `Incidence` matches a brute-force scan, including non-manifold vertices and repeated-vertex triangles.
- **Benchmarks** (criterion, `cargo bench -p fast-mesh-bvh`):
  - build (both qualities);
  - refit after moving 500 / 50,000 vertices;
  - treelet rebuild after a local topology edit;
  - memory (`byte_size`);
  - on synthetic meshes of 1M triangles, and of 10M when `FAST_MESH_BENCH_LARGE=1`.
  - CI runs `cargo bench --no-run` so the benches keep compiling. Results are not gated.
