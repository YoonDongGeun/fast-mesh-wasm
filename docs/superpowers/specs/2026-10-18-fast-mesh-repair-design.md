# fast-mesh Phase 7: Repair

- **Status:** draft for review
- **Parent spec:** `2026-10-14-fast-mesh-engine-design.md`, §10 (Operations and repair) and §15 (phase 7). Where this document differs, it wins. In particular, `orient()` and hole-fill fairing move up from "later phases" into this phase.
- **Builds on:**
  - Phase 1 store: transactions, patches, creation parents.
  - Phase 3 queries.
  - Phase 6 `fast_mesh_query::triangulate`.
- **Label convention:** *(owner)* marks what the owner said. *(my call)* marks choices made without asking.

## 1. Intent

The owner has met three concrete problems in real meshes:

1. **Flipped triangles.** On an otherwise clean surface, such as a hemisphere, a triangle or a small patch is wound the other way. Its normal points backwards, so anything that uses normals goes wrong there *(owner)*.
2. **Holes.** These include two holes that share one vertex, where a tool filled only one of them *(owner)*.
3. **Floating pieces.** Small parts that are not connected to the main surface *(owner)*.

Filled holes must continue the surrounding surface naturally, not as flat patches *(owner)*. Which openings count as holes is decided by size, so that the intended border of an open mesh, such as a hemisphere's rim, is kept *(owner)*.

This is a personal, general-purpose library for any mesh: scans, CAD exports, open or closed.

## 2. Goals, non-goals, anti-goals

### Goals

1. **`analyze`** reports problems and never modifies the mesh *(my call, accepted)*. It covers:
   - boundary loops (holes and borders), with edge count, perimeter and an area estimate;
   - coincident vertices within a tolerance;
   - degenerate (zero-area or repeated-vertex) and duplicate triangles;
   - non-manifold edges and vertices;
   - connected components, with triangle count and area;
   - orientation conflicts;
   - whether the mesh intersects itself.
2. **`orient`** makes the winding consistent across each component *(owner)*.
   - The majority by area wins.
   - A closed component faces outward *(my call, accepted)*.
3. **`fill_holes`** fills the boundary loops selected by size, using a three-step pipeline: triangulate, refine to the surrounding density, then fair so the patch follows the curvature around it *(owner)*. Two holes sharing a vertex are two loops, and both are filled *(owner)*.
4. **`remove_small_components`** removes pieces below a triangle count or an area *(owner)*.
5. **`weld`** merges coincident vertices within a tolerance. **`remove_degenerate`** removes zero-area and duplicate triangles *(my call, accepted)*.
6. **Every write goes through a `Transaction`.** Undo, BVH, field and section updates therefore work automatically. In a Scene this is `scene.edit(id, |tx| ops::fill_holes(tx, &opts))` *(my call, accepted)*.

### Non-goals (this phase)
- Splitting non-manifold elements (`split_non_manifold`).
- Remeshing, smoothing and decimation.
- Resolving self-intersections. They are only detected.
- Hole-filling choices beyond the size filter, such as picking holes interactively. Callers can still pass explicit loop ids from `analyze`.

### Anti-goals
- **No silent damage.** An operation that cannot do its job on some element (a hole it cannot fill, a component that cannot be oriented) reports that element and leaves it untouched. It never writes a half-done patch.
- **No surprises at shared vertices.** Boundary loops that touch at a vertex must be separate loops. A loop must never cross itself at a shared vertex.
- **No filling of borders.** Filling an open mesh's intended border is a failure, so `fill_holes` without a size bound is rejected.

## 3. Constraints
- Same rules as earlier phases:
  - no `unsafe`;
  - MSRV 1.88, edition 2024;
  - typed errors;
  - fallible allocation (`try_reserve` → `OutOfMemory`);
  - single-threaded.
- No new crate dependencies. Linear solves are written in-tree: conjugate gradient on a sparse matrix, in `f64`.
- **A new crate, `fast-mesh-ops`,** depends on `fast-mesh-store` and `fast-mesh-query`. It does not depend on the Scene, because operations take a `Transaction` *(my call)*.
- **Operations read through `tx.mesh()`.** The store's cached adjacency is not guaranteed inside a transaction, so each operation builds the edge and corner maps it needs from the triangles at the start, and fails before writing if the build fails *(my call)*.
- **New vertices carry creation parents** where a sensible choice exists, so source channels interpolate (§6.4).

## 4. Shared building blocks

### 4.1 Edge map
- An undirected edge key is `(min, max)`.
- For each edge, the map records the triangles using it and the direction each one uses.
- Edge kinds:
  - **Boundary:** one triangle.
  - **Interior:** two triangles.
  - **Non-manifold:** three or more.
- Degenerate triangles (with a repeated vertex id) are skipped by every operation except `remove_degenerate` and `analyze`.

### 4.2 Boundary loops
- A boundary half-edge is the directed edge `a→b` of the single triangle using a boundary edge.
- **Walking at a vertex.** At vertex `b`, the next half-edge is found by rotating around `b`'s triangle fan, starting from the triangle of the incoming half-edge, until the next boundary half-edge leaving `b` is reached. Following the fan this way, not just picking any outgoing boundary edge, is what makes holes that share a vertex come out as separate loops. *(Implemented differently: rotating around `b`'s own fan reproduces the figure-eight bug at a pinch. See §12 item 1 for the rule that was built.)*
- **When the walk is ambiguous:** a vertex with non-manifold edges, or fans that cannot be ordered. The remaining half-edges at that vertex are paired in a deterministic order (by edge key), and each loop involved is marked `clean: false`. *(Refined in §12 items 1–2 and 13.)* `fill_holes` skips loops that are not clean unless `allow_unclean` is set *(my call)*.
- **Each loop records:**
  - its vertices in order;
  - `edges`;
  - `perimeter`;
  - `area`, the vector area magnitude `|Σ p_i × p_{i+1}| / 2`;
  - `clean`.
- **Ids.** A loop id is its index in a deterministic order: by the smallest vertex id in the loop, then by direction. Ids are valid only for the mesh revision that produced them, and `fill_holes` checks this.

### 4.3 Components
- Triangles are connected when they share an edge, of any kind *(my call)*.
- Each component has a triangle count, an area, a closed flag (no boundary edges and no non-manifold edges) and an orientable flag.

## 5. Operations

Each operation is a function in `fast_mesh_ops` that returns a report struct. The writing operations take `&mut Transaction`. Analysis takes `&Mesh`, plus an optional `&Bvh` for the self-intersection check.

### 5.1 `analyze(mesh, options) -> Analysis`
- Options:
  - `weld_tolerance: f32`, used only for counting coincident vertices (default `0.0`, meaning exact duplicates);
  - `self_intersection: Option<&Bvh>`.
- **Reports** (counts plus id lists; the lists that have a count beside them are capped at `options.max_listed`, default 1000, and the counts stay exact; §12 item 8):
  - boundary loops (§4.2);
  - coincident vertex groups;
  - degenerate triangles;
  - duplicate triangles (the same vertex set in any order);
  - non-manifold edges and vertices;
  - components (§4.3);
  - orientation conflicts: the number of interior edges whose two triangles use the edge in the same direction;
  - `self_intersects: Option<bool>`.
- **Self-intersection check.** Look for triangle pairs that intersect and share no vertex, using the BVH and the exact tri-tri test from Phase 3b. Pairs that share a vertex are not tested, and the docs say so *(my call)*. "Degenerate" means different things here and in `analyze`/`remove_degenerate` (§12 item 8).

### 5.2 `orient(tx, options) -> OrientReport`
- **Build patches.** Grow patches across interior edges, flipping the neighbour when it uses the shared edge in the same direction. Non-manifold edges are barriers.
- **Non-orientable patches.** A patch that would need a triangle both flipped and not flipped (a Möbius-like strip) is reported as non-orientable. That patch is left unchanged; the rest of its component is still oriented (§12 item 3).
- **Per component:** among the orientable patches joined through non-manifold edges, choose the orientation with the larger total area. Ties go to the patch containing the smallest triangle id.
- **Closed components:** then flip the whole component if its signed volume is negative.
- **Writing:** flipping a triangle swaps two of its vertices with `set_triangles`.
- **Report:** triangles flipped, components changed, components left unchanged (non-orientable).

### 5.3 `fill_holes(tx, options) -> FillReport`
- **Selection** *(owner: by size)*. `FillOptions` has:
  - `max_edges: Option<usize>`;
  - `max_perimeter: Option<f64>`;
  - `max_area: Option<f64>`;
  - `loops: Option<Vec<LoopId>>`;
  - `allow_unclean: bool` (default false);
  - `fair: bool` (default true).

  A loop is filled when it meets every bound that is set. With no bound and no explicit `loops`, the call fails with `InvalidInput`. When both `loops` and bounds are given, both apply, as an intersection (§12 item 4).
- **Per loop:**
  1. **Triangulate** the loop with minimum-weight dynamic programming over the loop's vertices. The weight is the triangle area plus a penalty on the worst dihedral angle to neighbouring triangles (Liepa 2003). It costs O(n³) time, so it is used up to `dp_limit` edges (default 200 *(my call)*). Longer loops are projected onto their best-fit plane and triangulated with `fast_mesh_query::triangulate`. If the projection folds the loop (the projected polygon self-intersects), the loop is reported as `failed` and left open.
  2. **Refine** to the surrounding density: give each boundary vertex a target edge length equal to the mean length of its existing edges. Split patch triangles by Liepa's rule: split a triangle when √2 × the distance from its centroid to a corner exceeds the target length at the centroid (§12 item 5), then relax by flipping edges to improve triangle shape. Patch-internal only.
  3. **Fair** (when `fair`): move only the new interior vertices so the patch minimises bending.
     - Solve the bi-Laplacian `Δ²x = 0` with conjugate gradient, holding fixed the loop vertices and the ring of existing vertices just outside it, so the patch continues the surrounding curvature.
     - Use cotangent weights, clamped to stay positive; the builder may fall back to uniform weights if cotangents prove unstable, recording the choice (done; §12 item 6).
     - Iteration limit and tolerance are left to the builder. Non-convergence keeps the best iterate and sets `converged: false` in the report, after a retry with uniform weights (§12 item 6).
- **Writing:** the patch is written only once all three steps have succeeded for that loop, so a failed loop writes nothing.
- **Report:** filled loops (triangles and vertices added), skipped loops (and why: bounds, unclean), failed loops (and why).
- **Output orientation:** patch triangles take the winding implied by the loop's direction, so they agree with their neighbours.

### 5.4 `remove_small_components(tx, options) -> RemoveReport`
- Options: `min_triangles: Option<usize>`, `min_area: Option<f64>`, and `keep_largest: bool` (default true, so the largest component always survives).
- A component is removed when it is below any bound that is set. With no bound set, the call fails with `InvalidInput`.
- Its triangles and the vertices left unused are removed.

### 5.5 `weld(tx, tolerance) -> WeldReport`
- Group vertices within `tolerance` using a uniform-grid spatial hash and union-find. Each group keeps its smallest vertex id, at its original position *(my call)*. Cell size and non-finite positions: §12 item 14.
- Triangles are rewritten to the kept ids. Triangles that collapse (a repeated id) are removed. Unused vertices are removed.
- `tolerance = 0` merges exact duplicates only.

### 5.6 `remove_degenerate(tx, options) -> DegenerateReport`
- Removes:
  - triangles with a repeated vertex id;
  - triangles whose area is ≤ `area_epsilon · L²`, where `L` is the longest edge (default `area_epsilon = 1e-12`);
  - duplicate triangles (the same vertex set). The smallest id is kept.
- Leaves unused vertices alone unless `remove_unused_vertices` is set (default true).

## 6. Cross-cutting rules

### 6.1 Atomic per call
- **Validate, then write.** Each operation computes its whole plan from `tx.mesh()` and validates it, then writes.
- **Failures during writing.** A failure while writing (only `OutOfMemory`, or store errors that would indicate a bug) is returned as an error. The caller's transaction is then poisoned, the same contract as other transaction errors. `scene.edit` rolls it back.

### 6.2 Determinism
- The same mesh and options give the same result: no hash-order dependence and no randomness.

### 6.3 Scale
- Tolerances (`weld`, degenerate area) are absolute or relative as documented.
- Geometry runs in `f64`, computed from the stored `f32` positions in local coordinates, origin excluded.

### 6.4 Creation parents
- **Refinement:** splitting an edge gives parents `[(a, 0.5), (b, 0.5)]`. Splitting a triangle at its centroid gives the three vertices with weights 1/3.
- **Fairing** moves vertices without changing their parents.
- **Weld** creates no vertices.

## 7. Errors
- New type `OpsError`, with variants `Store(StoreError)`, `Query(QueryError)`, `InvalidInput(&'static str)` and `StaleLoops` (loop ids from another revision).
- Per-element failures (unfillable hole, non-orientable component) are reported, not errors.

## 8. Testing
- **analyze:**
  - Fixtures with a known count of each problem: holes, a bowtie pair of holes sharing a vertex, duplicates, degenerates, a non-manifold fin, components, flipped patches, and self-intersecting crossed boxes.
  - Analysis never changes the mesh revision.
- **Loops:**
  - Two square holes sharing a corner vertex give two clean loops of 4 edges each.
  - A hemisphere gives one loop, its rim.
- **orient:**
  - A hemisphere with random flipped triangles or patches comes back fully consistent, matching the majority.
  - A closed sphere wound inward comes back outward.
  - A Möbius strip is reported and left unchanged.
  - The result has zero orientation conflicts on orientable input.
- **fill_holes:**
  - A sphere with holes punched in it: after filling, the mesh is closed and its orientation is consistent.
  - Filled vertices lie within a tolerance of the true sphere: radius error below 2% of r for holes up to 30° across, with fairing on. The test pins the fairing quality.
  - Without fairing, the patch is flatter: its radius error is measurably larger.
  - The bowtie pair is filled as two loops.
  - The hemisphere rim is skipped when `max_perimeter` is below the rim length.
  - A loop that cannot be filled writes nothing.
  - Undo restores the mesh exactly.
- **remove_small_components / weld / remove_degenerate:** counts as expected, and idempotent (a second run changes nothing).
- **Property tests:**
  - Random meshes with random damage (punched holes, flipped triangles, duplicated vertices, floating pieces). Running `weld → remove_degenerate → remove_small_components → orient → fill_holes` never panics and leaves zero orientation conflicts on orientable components.
  - Every operation is undoable through the Scene, and the reports agree with a second `analyze`.

## 9. Benchmarks
- `analyze` on 1M triangles, with and without the self-intersection check.
- `orient` on a 1M mesh with 1% flipped triangles.
- `fill_holes` of 10 holes of 50–500 edges on a 1M mesh.
- `weld` on 1M triangles with 10% duplicated vertices.
- Record the results in `docs/benchmarks.md`. Done; numbers in §12 item 12.

## 10. Left to the builder
- Data structures: the edge map, union-find and spatial hash.
- The DP weight's exact dihedral penalty, provided it follows Liepa's minimum-area-then-dihedral spirit.
- CG details: preconditioner, iteration cap, tolerance, and the cotangent clamp.
- The refinement and relaxation loop, provided the density rule of §5.3 step 2 holds.
- Splitting into plan tasks.

## 11. Decisions after the builder check (all *my call*; they override earlier sections)

1. **Public API changes are allowed.**
   - `Scene::edit` becomes generic over its error. Its closure returns `Result<R, E>` with `E: From<StoreError>`, and `edit` returns `Result<R, E>`. *(Built differently: §12 item 9.)* Scene-side failures are carried through a new `From<SceneError>` bound, or the builder may use an equivalent design that keeps existing callers compiling with at most a type annotation. Record the choice.
   - `OpsError` implements `From<StoreError>`; what was built is in §12 item 9. So `scene.edit(id, |tx| ops::fill_holes(tx, &opts))` compiles.
   - `fast-mesh-query` gains a public `self_intersecting_pairs(mesh, bvh, out)`. It returns triangle pairs that intersect and share no vertex, using the exact tri-tri test. `analyze` uses it.
2. **Loop ids are content fingerprints, not revision-bound.**
   - A `LoopId` holds the loop's smallest vertex id, its length, and a hash of its vertex sequence (rotated to start at the smallest vertex).
   - `fill_holes` recomputes the loops from `tx.mesh()` and matches the requested ids by content. Any id that does not match returns `OpsError::StaleLoops` before anything is written.
   - This works inside one transaction after earlier operations have run, so a renumbered or removed loop can never fill the wrong opening.
3. **`orient` and non-manifold edges.**
   - Each patch (grown across interior edges only) takes its own majority by area, independently. Patches meeting only at non-manifold edges are not aligned with each other.
   - "Closed, so face outward" applies only to closed components that have no non-manifold edges, as §4.3 defines. Non-manifold components get per-patch majority only.
   - Orientation conflicts are counted on interior edges only, so "zero conflicts on orientable input" is well defined.
4. **Pinch vertices.**
   - `analyze` counts a vertex whose triangle fan splits into several pieces as a non-manifold vertex, in a separate `pinch_vertices` list. This includes the bowtie corner where two holes touch. `non_manifold_vertices` covers vertices on non-manifold edges.
   - Boundary loops through a pinch vertex are still `clean` when every fan piece at that vertex is edge-manifold, so it can be ordered.
   - A loop is `clean: false` only when the vertex has non-manifold edges. *(Also unclean: open chains, oppositely wound pinches and loops across an orientation seam; §12 items 1–2 and 13.)*
   - The owner's bowtie case is therefore reported as a pinch, and still filled by default.

## 12. Implementation notes

What the implementation settled or found. Items 1 to 6, 8, 9, 13 and 14 correct or qualify wording in earlier sections (§4.2, §5.1 to §5.3, §5.5, §9, §11.1 and §11.4), which now point here; the rest add detail or record deviations. Measured numbers are in item 12 and in `docs/benchmarks.md`.

1. **The fan walk (§4.2).** The literal rule, "rotate around `b`'s fan to the next boundary half-edge", reproduces the figure-eight bug at a pinch vertex: it can return to the incoming loop's own piece and join two holes into one loop. The implemented rule takes the next boundary edge from another fan piece across the gap. With three or more wedges it chooses the smallest counter-clockwise turn about a local normal, and ties go by edge key order.
2. **Open chains and unclean pinches (§4.2).** An open chain of boundary edges around a non-manifold edge is walked whole from its head and comes out as a loop with `closed = false` and `clean = false`; `fill_holes` never fills it, even with `allow_unclean`. A pinch whose wedges are wound oppositely also yields unclean pieces.
3. **`orient` and non-orientable patches (§5.2).** A non-orientable patch leaves only that patch unchanged, not its whole component. The rest of the component is oriented as usual.
4. **Selection in `fill_holes` (§5.3).** Explicit `loops` and the set bounds (`max_edges`, `max_perimeter`, `max_area`) both apply, as an intersection: a listed loop that exceeds a bound is skipped. Ids that match no loop still return `StaleLoops` before anything is written.
5. **Refinement rule (§5.3).** It follows Liepa: split a patch triangle when √2 × the distance from its centroid to a corner is greater than σ at the centroid, where σ is the interpolated target length. This replaces the "longer than √2 × the target" wording.
6. **Fairing.** Cotangent weights are tried first. The solve is retried with uniform weights when the cotangent solve gives a non-finite result, and also when conjugate gradient does not converge. `FilledLoop.weights` (`Option<FairWeights>`) reports which weights were used, and is `None` when fairing did not run. The kept solution is the converged cotangent one, else the converged uniform one; if neither converged, the finite one with the smaller residual, with `converged: false` (the loop is still filled). Only when neither solve is finite is the loop reported as failed and left open.
7. **Triangulation of a loop.** The DP rejects any triangulation that contains a zero-area triangle. The projection path (loops longer than `dp_limit`) runs earcut, which can drop exactly collinear points of the loop; the implementation restores them, so every loop vertex is used. This is a safety net for a fold that projection does not detect.
8. **`max_listed` and "degenerate".** `max_listed` caps only the lists that have a count beside them; the counts are exact, and the loop and component lists are not capped. "Degenerate" has two meanings. In `analyze` and `remove_degenerate` it is a repeated vertex id or an area ≤ ε·L². In `self_intersecting_pairs` a triangle counts as degenerate only when it is exactly collinear.
9. **`Scene::edit`.** It is `edit<R, E: Into<SceneError>>(id, f: FnOnce(&mut Transaction) -> Result<R, E>) -> scene::Result<R>`, not `Result<R, E>` as in §11.1. `SceneError` gained an `Ops(OpsError)` variant with a `From<OpsError>` impl, so `scene.edit(id, |tx| ops::fill_holes(tx, &opts))` compiles and returns `SceneError::Ops` on failure. A closure whose body never fails cannot infer `E` (E0283) and needs a type annotation, for example `Ok::<(), StoreError>(())`.
10. **Poisoned transactions (§6.1).** A write failure inside a store call, after an earlier write in the same operation, poisons the transaction (`commit` returns `TransactionPoisoned`), so `scene.edit` rolls the whole edit back. The operations validate before writing, so this happens only on out-of-memory.
11. **`fast-mesh-bvh`** is a workspace dependency of `fast-mesh-ops`, for the self-intersection option of `analyze`.
12. **Measured numbers** (i5-13600K, 2026-10-18; criterion's headline estimate, the middle of its interval, not a median; tables and conditions in `docs/benchmarks.md`).
    - `analyze`, 1M terrain: 478 ms; with the self-intersection check (BVH prebuilt): 893 ms.
    - `orient`, 1M terrain with 1% flipped: 291 ms (10,026 triangles flipped).
    - `weld`, 1M triangles with 10% of the vertices duplicated: 86 ms (50,269 merged).
    - `remove_small_components`, 1M terrain plus 1,000 two-triangle specks: 117 ms.
    - `fill_holes`, ten holes of 61 to 482 edges in a 1M terrain (83,813 triangles and 40,548 vertices added): 360 ms with `fair` off and 5.16 s with `fair` on. Fairing is about 93% of the time (`fair` defaults to true); no budget was set for it and nothing was tuned.
    - Sphere fairing accuracy (the `unfaired_patch_is_flatter` fixture; largest `| |p| − 1 |` over the new vertices of each hole, on a unit UV sphere of 64 rings and 128 segments with holes of 10°, 20° and 30°): 0.0005, 0.0018 and 0.0082 with fairing, against 0.0103, 0.0507 and 0.1182 without.
13. **Loops across an orientation seam (§4.2, item 2).** A flipped triangle on a hole's rim reverses its boundary edge, and the walk then follows the hole backwards; filling that loop would write an inverted patch. A loop is therefore also `clean = false` when it walks a boundary edge against its triangle's direction, or when it is closed and passes a vertex whose outgoing and incoming boundary edges do not balance. `fill_holes` skips such loops unless `allow_unclean` is set; running `orient` first makes them clean. The Möbius boundary, walked across its seam, is unclean too.
14. **Weld grid (§5.5).** Cells are always `2 × tolerance` wide and do not depend on the mesh extent, so one far outlier cannot widen them and make the grouping quadratic. A coordinate at least 2^60 cells from zero (every coordinate at tolerance 0) is keyed by its exact value instead, so cell coordinates never saturate; at that size its `f32` spacing is at least 2^36 cells, so it can only pair with an equal coordinate. Vertices with a non-finite coordinate are left out of the grid and never merged, in `weld` and in `analyze`'s coincident-vertex count.
