# Mesh Boolean Arrangement Fix (#165) — Design

**Date:** 2026-11-11  
**Issue:** [#165](https://github.com/YoonDongGeun/fast-mesh-wasm/issues/165)  
**Status:** Approved

---

## 1. Intent and Why

`fast_mesh_ops::boolean_plan` and `apply_boolean` currently fail on generic curved meshes (refusing with `"boolean intersection too complex"` or silently returning open meshes with boundary loops) and produce wrong results on 223 of 576 axis-offset unit-cube triangulation sweeps (163 with `orientation_conflicts > 0` and wrong volume, 60 open).

Mesh booleans are a foundational engine primitive: users invoke them directly via CLI, Rust, `Scene`, and `WorkerScene`, and upcoming features (#194 `cut` collision response and #195 Showcase modeling workbench) rely on exact booleans producing closed, consistently oriented results at interactive to 1M-triangle scales.

### Root Causes Identified in Current `boolean.rs` / `polylines.rs`

1. **Non-canonical intersection endpoints (`1–2 f32 ULP` gaps):** `intersecting_segments` clips each `(tri_a, tri_b)` pair independently in `f64`, tests endpoints with zero-tolerance `in_tri_2d` on rounded `f64` coordinates (dropping valid near-edge segments), and casts endpoints to `f32`. `boolean.rs` deduplicates new vertices only by `f32::to_bits()`. Adjacent pairs sharing an edge or face compute slightly different rounded coordinates, splitting a single geometric piercing point into disconnected interior vertices `~1e-7` apart.
2. **Brittle custom convex piece splitter & post-hoc `stitch`/`weld`:** Disconnected `1e-7` gaps cause `fan_split` / `split_crossings` to drop sliver fan triangles (`"boolean intersection too complex"`) or emit triangles bridging across the intersection curve that fold onto neighbors during final `weld`, creating open boundary loops and orientation conflicts.
3. **Axis-aligned 3-ray `parity()`:** Shooting `+X/+Y/+Z` rays misclassifies triangles whose centroids align with face diagonals, coplanar faces, or silhouette edges, and costs $O(T \log T)$ raycasts across every triangle of both meshes.
4. **Coplanar `Intersect` bug:** Opposing face-to-face coplanar patches (`dot(n_A, n_B) < 0`) were kept in `BooleanOp::Intersect`, leaving zero-volume open sheets on touching solids.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- Pass all 576 unit-cube triangulation/offset combinations (union, subtract, intersect) with `closed == true`, `orientation_conflicts == 0`, and analytic volume match.
- Succeed on generic-offset overlaps of curved manifold meshes (spheres, tori, arbitrary closed solids) at 2k, 50k, and 1M triangles.
- Guarantee **no silent bad output**: every `Ok(BooleanPlan)` / `Ok(BooleanReport)` produces a closed mesh with zero boundary edges and zero orientation conflicts, or returns an explicit `Err(OpsError::InvalidInput(...))`.
- Keep 1M-triangle overlapping sphere booleans fast (patch-level winding classification rather than per-triangle raycasting) and measure a 1M-pair benchmark.
- Preserve existing public signatures (`boolean_plan`, `apply_boolean`, `BooleanOp`, `BooleanPlan`, `BooleanReport`, `Scene::boolean`, JS/Worker bindings, CLI).

### Non-Goals
- Volumetric / voxel / Marching-Cubes booleans (tracked separately in #191 and #194 as a selectable alternative algorithm).
- Supporting open, non-manifold, or self-intersecting input meshes (`check_input` continues to reject them with the exact existing error messages).
- Propagating interpolated UVs from mesh B onto mesh A (existing contract: A's `UV_CHANNEL` is dropped when present and restored on undo).

### Anti-Goals
- Never silently return a mesh with `boundary_loops > 0` or `orientation_conflicts > 0`.
- Never introduce arbitrary-precision `BigRational` allocations per triangle across 1M-triangle meshes in WASM.
- Never break determinism across runs or between single-threaded and `parallel` builds.

---

## 3. Constraints

- **Crates touched:** `crates/fast-mesh-ops` (`src/boolean.rs`, plus internal helper module `src/boolean/` if split for readability), `crates/fast-mesh-cli` (verify orientation-conflict validation matches), `fuzz/fuzz_targets/boolean.rs` (assert `measure().closed` + zero orientation conflicts on `Ok`).
- **Dependencies:** `crates/fast-mesh-ops/Cargo.toml` adds `robust = "1.2.0"` (already in `Cargo.lock` via `fast-mesh-query`). `fast-mesh-fields` is left untouched for #165: `boolean_plan` takes `&Mesh` (immutable), whereas `SignData::build` mutates `Mesh` caches; a small read-only `&Mesh` + `&Bvh` exact/dipole solid-angle winding evaluator lives inside `fast-mesh-ops` using `fast_mesh_ops::reserve` so OOM fault-injection coverage stays intact *(builder call)*.
- **Fallible allocation & OOM discipline:** Respect `fast_mesh_ops::reserve` / `push` helpers so existing OOM fault-injection tests continue to pass.
- **Formatting & linting:** `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` clean.

---

## 4. Approaches Considered

### Approach A: Topological Simplex Indexing + 2D Constrained Triangulation + Patch Winding Classification *(Chosen)*
Identify every intersection vertex by its canonical input simplex pair using exact `robust::orient3d` / `orient2d` signs (inspired by Cherchi et al. 2020), triangulate cut triangles in 2D with constrained edge recovery, and classify connected surface patches once per patch via Generalized Winding Number (`SignData`).
- **Pros:** Eliminates 1-ULP gaps and `stitch` heuristics at the source; fast at 1M triangles ($O(\text{intersecting tris})$ work + $O(\text{patches})$ winding queries); zero rational-arithmetic bloat.
- **Cons:** Replaces the internal split/classify/stitch implementation of `boolean.rs`.

### Approach B: Full Arbitrary-Precision Rational Coordinates (`BigRational` Arrangement)
Compute all plane intersections and 2D splits in exact rationals and round to `f32` at the end.
- **Rejected because:** Heavy heap churn and slowdowns at 1M triangles in WASM, and final `f32` rounding still requires topological collapse cleanup when distinct rational points round to the same `f32` slot.

### Approach C: Epsilon-Cluster Tuning on Current `fan_split` + `stitch`
Cluster `fresh` vertices within `1e-5` and widen `fan_split` tolerances.
- **Rejected because:** Fundamentally non-scalable across mesh densities (at 1M triangles, real mesh edges approach epsilon tolerances while 1-ULP mismatches remain).

---

## 5. Detailed Design

### 5.1 Coordinate Frame & Input Validation
1. `check_input(a, bvh_a)` and `check_input(b, bvh_b)` preserve all existing checks and error strings, plus one additional orientation check *(builder call)*:
   - **Reject inward-wound solids:** Compute net signed volume over alive triangles (and verify no outer shell encloses negative volume, i.e., net `signed_volume <= 0.0` when `triangle_count() > 0`, or any patch evaluates to winding `< -0.5`). Reject inverted closed solids (such as random inward-wound `FuzzBox`es) with `OpsError::InvalidInput("boolean input is wound inward")`, while still allowing valid multi-component hollow shells (e.g., `shell()` outputs with an outward outer wall and inward inner cavity whose net signed volume is positive and winding is in `{0, 1}`).
2. `tool = map_tool(b, xf)` maps B's vertices into A's local `f32` space and builds `tool_bvh`, as today.
   - *Builder call:* If `xf` is non-identity and `f32` rounding of `tool` introduces a degenerate triangle or self-intersection that `b` did not have in its own frame, validate `tool` via `check_input(&tool, &tool_bvh)` (returning `OpsError::InvalidInput` if `f32` collapse corrupted `tool`); every subsequent predicate operates on the exact `f64` widening (`f64::from`) of `a.positions()` and `tool.positions()`.

### 5.2 Canonical Intersection Graph (`VertexKey`)
Find all candidate intersecting triangle pairs `(ta, tb)` via `fast_mesh_query::intersecting_pairs(a, bvh_a, &tool, &tool_bvh, &Rigid::IDENTITY, &mut pairs)`.

First unify any coincident original vertices between A and B: if live vertex `va` of A and live vertex `vb` of `tool` have bit-identical `[f32; 3]` positions, map `OrigB(vb)` directly to `OrigA(va)`.

Every vertex in the arrangement is uniquely keyed by `VertexKey`:
- `OrigA(VertexId)` — original live vertex of A.
- `OrigB(VertexId)` — original live vertex of `tool` (not coincident with any live vertex of A).
- `VertOnEdgeA { vert_b: VertexId, edge_a: (VertexId, VertexId) }` / `VertOnEdgeB { vert_a: VertexId, edge_b: (VertexId, VertexId) }` — when an original vertex of one mesh lies exactly (`orient3d == 0` and exact 2D edge test) on the interior of an undirected edge of the other mesh; resolves to the existing vertex's position (`OrigB(vert_b)` / `OrigA(vert_a)`) while recording the split on `edge_a` / `edge_b`!
- `EdgeEdge { edge_a: (VertexId, VertexId), edge_b: (VertexId, VertexId) }` — canonical undirected edges `u < v` of A and B that intersect at a point strictly interior to both edges. Position computed once in `f64` via closest point between the two 3D segments, cast to `f32`.
- `EdgeTriA { edge_a: (VertexId, VertexId), tri_b: TriangleId }` — canonical undirected edge `u < v` of A piercing the strict interior of `tri_b`. Position computed once in `f64` by intersecting `(pos(u), pos(v))` with `tri_b`'s plane (using canonical sorted corner order of `tri_b` so `plane` is deterministic), clamped inside `tri_b`'s bounding box, cast to `f32`.
- `TriEdgeB { tri_a: TriangleId, edge_b: (VertexId, VertexId) }` — canonical undirected edge `u < v` of B piercing the strict interior of `tri_a`. Position computed once symmetrically.
- `SegSeg { side_a: bool, tri: TriangleId, k1: u32, k2: u32 }` — fallback key when two distinct constraint segments inside one triangle cross strictly in 2D projection (e.g., due to `f32` snap of endpoints).

### 5.3 Exact Simplex Classification per Intersecting Pair `(ta, tb)`
Using `robust::orient3d` and `robust::orient2d` on exact `f64`-widened coordinates (with the exact fallback from `fast-mesh-query` if outside `2^-200..2^200`, or `orient3d` directly since `f32` normal values are in `2^-149..2^128` — well inside `2^-200..2^200`!):
- Note: **Every finite `f32` value (including `f32` subnormals down to `2^-149`) is strictly within `robust`'s exact `[2^-200, 2^200]` safe range!** Thus `robust::orient3d` and `robust::orient2d` are **100% mathematically exact** on all `f64`-widened `f32` mesh coordinates without overflow or underflow!
- **Coplanar pair (`orient3d == 0` for all 3 corners of `ta` against `tb`):**
  - Project onto `ta`'s dominant normal axis (`normal_axis`).
  - Record vertices of `ta` inside/on `tb` and vertices of `tb` inside/on `ta`, plus `EdgeEdge` intersections for crossing edges of `ta` and `tb`.
  - Clip each edge of `tb` to `ta` (producing constraint segments on `ta`) and each edge of `ta` to `tb` (producing constraint segments on `tb`).
- **Transverse / touching pair (not all `orient3d == 0`):**
  - Collect the intersection points on the closed pair `(ta, tb)` by testing all 3 edges of `ta` against `tb` and all 3 edges of `tb` against `ta` via exact `orient3d` signs:
    - Edge `(u, v)` (`u < v`) vs triangle `T = (w0, w1, w2)`:
      - `su == 0 && sv == 0` (edge lies in `plane(T)`): clip 2D segment `(u, v)` against `T` in `T`'s projection plane using exact `orient2d`; yields 0, 1, or 2 canonical `VertexKey`s on `(ta, tb)`.
      - `su == 0` (or `sv == 0`): test the zero-height endpoint in `T` using exact `orient2d` on `T`'s 3 edges (`== 0` at two edges → corner coincidence; `== 0` at one edge → `VertOnEdge`; all 3 strictly inside → vertex-in-triangle).
      - `su * sv < 0`: orient `(p, q)` so `orient3d(w0, w1, w2, p) > 0 > orient3d(w0, w1, w2, q)`. Evaluate `o0 = sign(orient3d(p, q, w0, w1))`, `o1 = sign(orient3d(p, q, w1, w2))`, `o2 = sign(orient3d(p, q, w2, w0))`. If `o0 >= 0 && o1 >= 0 && o2 >= 0` (note `(w0, w1, w2)` has positive normal, so `p -> q` goes from `+` to `-` side, making right-hand `orient3d` signs consistent — verify sign convention with a unit test!), classify by zero count: 2 zeros → corner `Orig`, 1 zero → `EdgeEdge`, 0 zeros → `EdgeTriA`/`TriEdgeB`.
  - Deduplicate the collected `VertexKey`s for pair `(ta, tb)`. If 2 unique keys `(k0, k1)` remain, record constraint segment `(k0, k1)` on both `ta` and `tb` (if 3+ colinear keys occur from coplanar-edge-in-plane cases, sort along the line and chain consecutive pairs).

### 5.4 Coordinate Realization & `f32` Collapse Unification
Before triangulating:
- Every `VertexKey` has a realized `[f32; 3]` position (`OrigA`/`OrigB` keep their exact mesh position; intersection keys take their single canonically computed `[f32; 3]`).
- **Quantization collapse guard:** Two topologically distinct intersection keys on adjacent/same triangles can occasionally round to the **exact same `[f32; 3]`** (or within `0` distance). Unify any `VertexKey`s that have identical `[f32; 3]` bits into a single canonical vertex ID *before* per-triangle triangulation!
  - Because unification is global across both Side A and Side B, if two points collapse to one `f32` point, **both** A and B see the exact same collapsed vertex ID and zero-length constraint segments `(v, v)` drop out cleanly without creating slits or mismatched rims!

### 5.5 Per-Triangle Constrained 2D Triangulation (CDT)
For every triangle `t` (on A or `tool`) that has split edges, interior vertices, or interior constraint segments:
1. **Boundary ring construction:** For each of `t`'s 3 undirected edges `e = (min(u,v), max(u,v))`, sort all split vertices registered on `e` by their exact 1D projection `(pos_f64 - pos_f64(min)) · (pos_f64(max) - pos_f64(min))` (tie-broken by vertex ID). Reverse when `t` traverses `e` from `max` to `min`. This guarantees **100% identical edge subdivision** between the two triangles sharing `e` on the same mesh, with zero T-junctions.
2. **Planar projection:** Project `t` and all its vertices onto the 2D coordinate plane dropping `t`'s largest normal component, flipping 2D `(x, y)` order when needed so `t` is strictly counter-clockwise (`orient2d > 0`).
3. **Segment intersection & point-on-segment resolution in 2D:**
   - If any vertex on `t` lies on the interior of a constraint segment (within exact `orient2d == 0`, or within `f32`-rounding snap if endpoints quantized onto a segment), split that segment at the vertex.
   - If two constraint segments cross strictly (`orient2d` straddle), insert their 2D intersection point (`Fresh` vertex) and split both segments.
4. **Constrained triangulation:**
   - Insert all boundary-split and interior vertices into a 2D triangulation of `t` (starting from `t`'s 3 corners, splitting edges/triangles via `orient2d`), then enforce every constraint segment `(u, v)` via standard constrained edge recovery (finding the cavity of triangles crossed by `u..v` and re-triangulating the left and right cavity polygons via ear-clipping / Delaunay flips).
   - Every emitted sub-triangle `[v0, v1, v2]` has matching 3D winding with `t` and distinct vertex IDs.
   - **Important:** Do NOT drop tiny valid sub-triangles along a split edge or constraint curve! Dropping a sliver sub-triangle creates a hole (boundary loop) between neighbors! Zero-area collinear triangles (`orient2d == 0` in 2D) are resolved by flipping the shared diagonal with the adjacent triangle (or collapsing onto the shared edge on both sides), never by silently deleting a triangle and leaving an unmatched half-edge.

### 5.6 Coplanar Detection & Patch-Based Winding Classification
1. **Mark intersection seam edges:** Every undirected edge `(u, v)` between two arrangement vertices that came from an intersection constraint segment (or where an original vertex/edge touched the other surface) is marked as a **seam barrier**.
2. **Coplanar sub-triangle detection:** A sub-triangle `s` of A (or B) is coplanar with the other mesh iff all 3 of its vertices lie on the other mesh's surface (`Orig` shared / `VertOnEdge` / `EdgeEdge` / `EdgeTri` / or `closest_point` distance `<= tol` with all 3 corners satisfying `orient3d == 0` on a single overlapping triangle of the other mesh, accelerated by BVH leaf query at `s`'s centroid).
   - Actually, even simpler and exact: test whether `s`'s 3 vertices all have `orient3d(q0, q1, q2, p_k) == 0` and 2D interior overlap (centroid inside `(q0, q1, q2)`) for some triangle `q` of the other mesh overlapping `s`'s AABB in the other mesh's BVH!
   - When `s` is coplanar with triangle `q` of the other mesh:
     - On Side B (`is_a == false`): **always drop `s`** (Side A owns kept coplanar surface patches so duplicate faces are never emitted).
     - On Side A (`is_a == true`): compare `dot(normal(s), normal(q))`:
       - **Same normal (`> 0`, flush overlap):** keep for `Union` and `Intersect`; drop for `Subtract`.
       - **Opposite normal (`< 0`, face-to-face contact):** drop for `Union` and `Intersect`; keep for `Subtract`.
3. **Connected Patch Flood-Fill (Non-coplanar sub-triangles):**
   - Build dual adjacency across sub-triangles of Side A (and separately Side B), connecting two sub-triangles iff they share an undirected edge `(u, v)` that is **NOT a seam barrier** and neither sub-triangle is coplanar.
   - Untouched original triangles of A (and B) are simply 1-triangle entries in the same flood-fill over mesh adjacency (`EdgeMap`), so an entire 500k-triangle untouched hemisphere becomes **one single patch** in $O(T)$ integer array hops!
4. **Generalized Winding Number Evaluation (1 query per patch):**
   - Build a read-only BVH dipole / exact solid-angle winding structure (`WindingTree`, Barill et al. 2018, allocated through `fast_mesh_ops::reserve`) over `(tool.mesh, tool.bvh)` and over `(a, bvh_a)` when non-coplanar patches exist *(builder call: avoids mutating `&Mesh` caches or bypassing OOM fault injection)*.
   - For each non-coplanar patch, pick the member triangle with the **largest 3D area squared** and evaluate `w = winding_tree.winding_number(centroid_f64)`. If `w < -0.5` (an inverted component slipped past net-volume check), return `Err(OpsError::InvalidInput("boolean input is wound inward"))`.
   - `inside = w > 0.5`.
   - Apply standard boolean survival rule:
     - `Union`: `!inside` (both A and B)
     - `Subtract`: `!inside` on A, `inside` on B (with B's triangle winding reversed `swap(1, 2)`)
     - `Intersect`: `inside` (both A and B)

### 5.7 Materialization, Manifold Fan Decomposition & Verification Gate
1. **Cancel degenerate opposing pairs:** Remove any exact opposite duplicate triangle pairs `(a, b, c)` and `(a, c, b)` that collapsed due to `f32` quantization.
2. **De-pinch into strict 2-manifold fans *(aligns with `measure.rs` and `fast-mesh-cli` contract)*:**
   - Note that `measure()` defines `closed = boundary_loops == 0 && non_manifold_edges == 0`, and `fast-mesh-cli` (`invalid_boolean_result`) rejects any result with `!closed || orientation_conflicts > 0`.
   - When two shells kiss along an edge (`count(u -> v) == count(v -> u) == k` with `k >= 2`, e.g., edge-touching cubes in `Union`) or at a single vertex (multi-fan pinch vertex):
     - Pair incoming and outgoing half-edges around each `k >= 2` edge in cyclic dihedral angle order around the edge axis so sheets separate without crossing, forming strictly 2-manifold edge-connected triangle fans around every vertex.
     - Duplicate the shared vertex into distinct `PRef::Fresh` vertices per disconnected vertex fan so every edge in the output has **at most 2 incident triangles** (`count(u -> v) == 1 && count(v -> u) == 1`), giving `non_manifold_edges == 0` and `pinch_vertices.is_empty()`.
3. **Plan-level topological verification gate (inside `boolean_plan` before returning `Ok`):**
   - Across all surviving triangles (untouched kept A + rewritten A + kept B), verify that **every** undirected edge `{u, v}` has `count(u -> v) == 1 && count(v -> u) == 1` (`boundary_loops == 0`, `non_manifold_edges == 0`, `orientation_conflicts == 0`).
   - If any boundary edge, non-manifold edge, or orientation conflict remains, return `Err(OpsError::InvalidInput("boolean arrangement failed manifold check"))` rather than silently returning a broken mesh.
4. **Populate `BooleanPlan` & `apply_boolean(tx, plan)`:**
   - Map surviving arrangement vertices to `PRef::Live(va)` (for unsplit `OrigA(va)` fans) and `PRef::Fresh(i)` (for `OrigB`, intersection vertices, and split pinch fans), populating `BooleanPlan { drop_a, rewrite, fresh, tool }`.
   - `apply_boolean(tx, plan)` applies `drop_a`, `fresh`, `rewrite`, and `tool` to `tx` in one atomic edit, plus orphan-vertex cleanup (`corner_count(v) == 0`) on A's dropped vertices. Because Side A and Side B already share identical `PRef` vertex IDs along every seam edge, **no heuristic `stitch` or fuzzy `weld(tol)` is needed** (also preventing accidental welding of fine pre-existing features far from the cut).

---

## 6. Verification & Acceptance Plan

1. **576-run Cube Triangulation & Offset Sweep:**
   - Port the exact 576-combination sweep from Issue #165 comment (all 64 diagonal masks / varied face triangulations × offsets × all 3 ops) as an automated Rust test asserting `closed == true`, `orientation_conflicts == 0`, `non_manifold_edges == 0`, and `|volume - expected| < 1e-5`.
2. **Curved Mesh Matrix (Issue #165 Table):**
   - Automated tests for UV spheres `(1, 32, 32)` + `(0.31, 0.17, 0.23)`, heterogeneous spheres `(1, 32, 32)` vs `(0.9, 39, 43)`, torus vs sphere, `(0.31, 0, 0)` offset, and identical-sphere union (`t = (0, 0, 0)`), asserting closedness, zero orientation conflicts, and analytic lens volume where applicable.
   - Medium-scale test (`~50k` triangles sphere/torus pair) in `cargo test --release` / integration suite.
3. **Coplanar & Degenerate Touch Regressions:**
   - Stacked cubes (`Union`, `Subtract`, AND `Intersect` — verifying `Intersect` of face-touching cubes yields 0 triangles, not an open 2-triangle sheet!), nested spheres, disjoint meshes, vertex-only and edge-only touches.
4. **Fuzz Target Tightening (`fuzz/fuzz_targets/boolean.rs`):**
   - Whenever `boolean_plan` + `apply_boolean` succeeds, assert `boundary_loops` is empty and `orientation_conflicts == 0`.
5. **1M-Triangle Benchmark (`crates/fast-mesh-ops/benches/ops.rs` & `docs/benchmarks.md`):**
   - Measure union/subtract/intersect on two overlapping ~1M-triangle spheres at generic offset `(0.31, 0.17, 0.23)` and record timing.

---

## 7. Left to the Builder

- Exact module organization (keeping `crates/fast-mesh-ops/src/boolean.rs` as entry point with submodules `crates/fast-mesh-ops/src/boolean/*.rs` if >800 lines).
- Internal CDT data structure (half-edge or indexed triangle-neighbor table per local triangle — local triangles typically have 4–20 vertices, so a compact flat `Vec` CDT is cache-resident and allocation-light).
- Exact tie-breaking details for collinear points on a single constraint segment.
