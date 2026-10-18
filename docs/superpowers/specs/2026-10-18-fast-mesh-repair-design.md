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
- **Walking at a vertex.** At vertex `b`, the next half-edge is found by rotating around `b`'s triangle fan, starting from the triangle of the incoming half-edge, until the next boundary half-edge leaving `b` is reached. Following the fan this way, not just picking any outgoing boundary edge, is what makes holes that share a vertex come out as separate loops.
- **When the walk is ambiguous:** a vertex with non-manifold edges, or fans that cannot be ordered. The remaining half-edges at that vertex are paired in a deterministic order (by edge key), and each loop involved is marked `clean: false`. `fill_holes` skips loops that are not clean unless `allow_unclean` is set *(my call)*.
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
- **Reports** (counts plus id lists, capped at `options.max_listed`, default 1000):
  - boundary loops (§4.2);
  - coincident vertex groups;
  - degenerate triangles;
  - duplicate triangles (the same vertex set in any order);
  - non-manifold edges and vertices;
  - components (§4.3);
  - orientation conflicts: the number of interior edges whose two triangles use the edge in the same direction;
  - `self_intersects: Option<bool>`.
- **Self-intersection check.** Look for triangle pairs that intersect and share no vertex, using the BVH and the exact tri-tri test from Phase 3b. Pairs that share a vertex are not tested, and the docs say so *(my call)*.

### 5.2 `orient(tx, options) -> OrientReport`
- **Build patches.** Grow patches across interior edges, flipping the neighbour when it uses the shared edge in the same direction. Non-manifold edges are barriers.
- **Non-orientable patches.** A patch that would need a triangle both flipped and not flipped (a Möbius-like strip) is reported as non-orientable. That component is left unchanged.
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

  A loop is filled when it meets every bound that is set. With no bound and no explicit `loops`, the call fails with `InvalidInput`.
- **Per loop:**
  1. **Triangulate** the loop with minimum-weight dynamic programming over the loop's vertices. The weight is the triangle area plus a penalty on the worst dihedral angle to neighbouring triangles (Liepa 2003). It costs O(n³) time, so it is used up to `dp_limit` edges (default 200 *(my call)*). Longer loops are projected onto their best-fit plane and triangulated with `fast_mesh_query::triangulate`. If the projection folds the loop (the projected polygon self-intersects), the loop is reported as `failed` and left open.
  2. **Refine** to the surrounding density: give each boundary vertex a target edge length equal to the mean length of its existing edges. Split patch triangles whose edges are longer than √2 × the target (Liepa's density rule), then relax by flipping edges to improve triangle shape. Patch-internal only.
  3. **Fair** (when `fair`): move only the new interior vertices so the patch minimises bending.
     - Solve the bi-Laplacian `Δ²x = 0` with conjugate gradient, holding fixed the loop vertices and the ring of existing vertices just outside it, so the patch continues the surrounding curvature.
     - Use cotangent weights, clamped to stay positive; the builder may fall back to uniform weights if cotangents prove unstable, recording the choice.
     - Iteration limit and tolerance are left to the builder. Non-convergence keeps the best iterate and sets `converged: false` in the report.
- **Writing:** the patch is written only once all three steps have succeeded for that loop, so a failed loop writes nothing.
- **Report:** filled loops (triangles and vertices added), skipped loops (and why: bounds, unclean), failed loops (and why).
- **Output orientation:** patch triangles take the winding implied by the loop's direction, so they agree with their neighbours.

### 5.4 `remove_small_components(tx, options) -> RemoveReport`
- Options: `min_triangles: Option<usize>`, `min_area: Option<f64>`, and `keep_largest: bool` (default true, so the largest component always survives).
- A component is removed when it is below any bound that is set. With no bound set, the call fails with `InvalidInput`.
- Its triangles and the vertices left unused are removed.

### 5.5 `weld(tx, tolerance) -> WeldReport`
- Group vertices within `tolerance` using a uniform-grid spatial hash and union-find. Each group keeps its smallest vertex id, at its original position *(my call)*.
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
- Record the results in `docs/benchmarks.md`.

## 10. Left to the builder
- Data structures: the edge map, union-find and spatial hash.
- The DP weight's exact dihedral penalty, provided it follows Liepa's minimum-area-then-dihedral spirit.
- CG details: preconditioner, iteration cap, tolerance, and the cotangent clamp.
- The refinement and relaxation loop, provided the density rule of §5.3 step 2 holds.
- Splitting into plan tasks.
