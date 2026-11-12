# Mesh Boolean Arrangement Fix (#165) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the brittle segment-splitting and post-hoc stitching boolean pipeline in `fast-mesh-ops` with a canonical simplex-indexed mesh arrangement, constrained 2D per-triangle triangulation, patch-level winding classification, and 2-manifold fan decomposition so curved and polyhedral booleans succeed without open loops or orientation conflicts up to 1M triangles.

**Architecture:** Split `crates/fast-mesh-ops/src/boolean.rs` into focused internal submodules under `crates/fast-mesh-ops/src/boolean/` (`winding.rs`, `arrangement.rs`, `cdt.rs`, `classify.rs`) behind the unchanged public `boolean_plan` / `apply_boolean` API. Every intersection point is keyed by its canonical input simplex pair using exact `robust::orient3d` / `orient2d` signs so both meshes share identical vertex IDs along every seam without fuzzy post-hoc welding.

**Tech Stack:** Rust (`fast-mesh-ops`, `fast-mesh-query`, `fast-mesh-bvh`, `fast-mesh-store`, `robust = "1.2.0"`), Criterion benchmarks, Cargo/libFuzzer target, TypeScript Vitest suite.

**Spec:** `docs/superpowers/specs/2026-11-11-boolean-arrangement-fix-design.md`

## Global Constraints

- Preserve public signatures and types verbatim: `BooleanOp`, `BooleanPlan`, `BooleanReport`, `boolean_plan`, `apply_boolean` in `crates/fast-mesh-ops/src/boolean.rs`, plus `Scene::boolean` and CLI `fast-mesh boolean`.
- Preserve all existing `check_input` error strings verbatim (`"boolean needs closed meshes"`, `"boolean needs manifold meshes"`, `"boolean input has degenerate triangles"`, `"boolean input has duplicate triangles"`, `"boolean input is inconsistently wound"`, `"boolean input self-intersects"`, `"cannot combine a mesh with itself"`), adding `"boolean input is wound inward"` for closed inputs with net signed volume `<= 0.0`.
- Every growable temporary in `fast-mesh-ops` must use `crate::reserve` / `crate::push` (or `BTreeMap`/`BTreeSet` only after `allocation_allowed()?` guards, or sorted `Vec`s) so `fail_allocations_after` OOM fault-injection tests pass deterministically.
- Results must be deterministic across runs and bit-identical with and without `--features parallel` (no `HashMap`/`HashSet` iteration order dependence).
- Every `Ok` result from `boolean_plan` + `apply_boolean` must satisfy `measure(&mesh).closed == true` (`boundary_loops == 0 && non_manifold_edges == 0`) and `orientation_conflicts == 0`, or return `Err(OpsError::InvalidInput(...))`.
- `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` must pass with zero warnings.

## Review Focus

- **Inward-wound closed inputs** (e.g., `FuzzBox` with `lo > hi`): must return `Err(OpsError::InvalidInput("boolean input is wound inward"))`, while valid hollow shells (`shell()` output: outward outer wall + inward inner cavity, net volume `> 0`) succeed — pinned in Task 1 & Task 4 tests.
- **Identical coincident meshes** (two separate `Mesh` instances with bit-identical spheres/cubes at offset `(0, 0, 0)`): `Union` and `Intersect` must return a closed manifold mesh matching single-mesh volume (not 75 open boundary loops as reported in #165), and `Subtract` must return `0` triangles — pinned in Task 5 tests.
- **Face-to-face opposing coplanar contact in `Intersect`** (stacked cubes sharing `z = 1`): must return `0` triangles rather than an open 2-triangle sheet — pinned in Task 4 tests.
- **Edge-only and vertex-only kissing shells in `Union`** (two cubes touching along one edge `x = 1, y = 1` or corner `(1, 1, 1)`): must decompose into separate 2-manifold fans so `non_manifold_edges == 0`, `pinch_vertices.is_empty()`, and `measure().closed == true` — pinned in Task 4 tests.
- **Non-axis-aligned rigid transform `xf` in `boolean_plan`**: rotated + translated `b` mapped via `map_tool` must pass `check_input` and produce closed, conflict-free results matching world-space volume — pinned in Task 4 & Task 5 tests.

---

### Task 1: Read-Only BVH Winding Number & Signed Volume (`boolean/winding.rs`)

**Files:**
- Modify: `crates/fast-mesh-ops/Cargo.toml` (add `robust = "1.2.0"` under `[dependencies]`)
- Create: `crates/fast-mesh-ops/src/boolean/winding.rs`
- Modify: `crates/fast-mesh-ops/src/boolean.rs` (declare `mod winding;` for internal module wiring)

**Interfaces:**
- Consumes: `fast_mesh_store::{Mesh, TriangleId}`, `fast_mesh_bvh::{Bvh, NodeId, NodeKind}`, `crate::{OpsError, Result, push, reserve}`.
- Produces (in `crate::boolean::winding`):
  - `pub(super) fn signed_volume6(mesh: &Mesh) -> f64` — `6.0 * signed_volume` over alive triangles in `f64` (`> 0.0` for outward closed solids).
  - `pub(super) struct WindingTree`
  - `impl WindingTree`:
    - `pub(super) fn build(mesh: &Mesh, bvh: &Bvh) -> Result<Self>` — precomputes area-weighted dipole/quadrupole moments per BVH node (Barill et al. 2018, reading `&Mesh` only, reserving all `Vec`s via `crate::reserve`).
    - `pub(super) fn winding_number(&self, mesh: &Mesh, bvh: &Bvh, p: [f64; 3]) -> f64` — hierarchical solid-angle evaluation (exact Van Oosterom–Strackee solid angle at leaves and near nodes where `dist^2 <= beta^2 * radius^2` with `beta = 2.3`, dipole/Taylor approximation on far nodes).

- [ ] **Step 1: Write the failing unit tests in `crates/fast-mesh-ops/src/boolean/winding.rs`**

Add `#[cfg(test)] mod tests` testing:
- `unit_cube_winding_and_grazing_rays`: `box_mesh([0,0,0], [1,1,1])` gives `signed_volume6 ≈ 6.0`; `winding_number` at `[0.5, 0.5, 0.5]`, `[0.25, 0.25, 0.125]` (aligned with face diagonals!), and `[0.999, 0.5, 0.5]` is in `(0.99, 1.01)`; at `[-0.001, 0.5, 0.5]` and `[2.0, 0.5, 0.5]` (grazing `y=0.5` diagonal line outside) is in `(-0.01, 0.01)`.
- `inward_cube_has_negative_volume_and_winding`: flipping all triangles of `cube()` gives `signed_volume6 ≈ -6.0` and interior `winding_number ≈ -1.0`.
- `hollow_shell_winding`: `shell()` on a closed `uv_sphere(12, 16, 1.0)` with thickness `0.3` gives `signed_volume6 > 0.0`, `winding_number([0.0, 0.0, 0.0])` in `(-0.02, 0.02)` (inner void), and `winding_number([0.85, 0.0, 0.0])` in `(0.98, 1.02)` (inside wall).
- `winding_build_honours_oom_hook`: arming `crate::fail_allocations_after(Some(0))` causes `WindingTree::build` to return `Err(OpsError::Store(StoreError::OutOfMemory))` (disarm in cleanup).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p fast-mesh-ops boolean::winding`
Expected: FAIL to compile (`winding` module / functions not implemented yet).

- [ ] **Step 3: Add `robust = "1.2.0"` to `crates/fast-mesh-ops/Cargo.toml` and implement `crates/fast-mesh-ops/src/boolean/winding.rs`**

Implement `signed_volume6` and `WindingTree` (using exact Van Oosterom–Strackee solid angle `atan2(det(a, b, c), |a||b||c| + (a·b)|c| + (b·c)|a| + (c·a)|b|) / (2π)` on leaf triangles and dipole/second-order moments per BVH subtree so 1M-triangle queries take microseconds while points arbitrarily close to triangles stay exact).

- [ ] **Step 4: Run tests and clippy to verify they pass**

Run: `cargo test -p fast-mesh-ops boolean::winding && cargo clippy -p fast-mesh-ops --all-targets -- -D warnings`
Expected: PASS, 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/fast-mesh-ops/Cargo.toml Cargo.lock crates/fast-mesh-ops/src/boolean.rs crates/fast-mesh-ops/src/boolean/winding.rs
git commit -m "feat(ops): add read-only BVH winding evaluator for booleans"
```

---

### Task 2: Canonical Simplex Intersection Graph (`boolean/arrangement.rs`)

**Files:**
- Create: `crates/fast-mesh-ops/src/boolean/arrangement.rs`
- Modify: `crates/fast-mesh-ops/src/boolean.rs` (declare `mod arrangement;`)

**Interfaces:**
- Consumes: `fast_mesh_query::{Rigid, intersecting_pairs}`, `robust::{Coord, Coord3D, orient2d, orient3d}`, `fast_mesh_store::{Mesh, TriangleId, VertexId}`, `fast_mesh_bvh::Bvh`, `crate::{OpsError, Result, allocation_allowed, push, reserve}`.
- Produces (in `crate::boolean::arrangement`):
  - `pub(super) type ArrVertexId = u32;`
  - `#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub(super) enum ArrOrigin { LiveA(VertexId), Fresh }`
  - `#[derive(Debug, Clone, Default)] pub(super) struct TriConstraints { pub edge_splits: [Vec<ArrVertexId>; 3], pub inner_verts: Vec<ArrVertexId>, pub segments: Vec<(ArrVertexId, ArrVertexId)> }`
    - `edge_splits[i]` holds strictly interior split vertices along corner `i -> (i + 1) % 3` in exact directed order along that triangle's half-edge (reversed from the canonical `(min, max)` order when `tri[i] > tri[(i + 1) % 3]`).
  - `pub(super) struct Arrangement`:
    - `pub positions: Vec<[f32; 3]>` (indexed by `ArrVertexId`)
    - `pub origins: Vec<ArrOrigin>` (indexed by `ArrVertexId`)
    - `pub tri_a: Vec<(TriangleId, TriConstraints)>` (sorted by `TriangleId`, includes every triangle of `a` with any split edge, inner vertex, or segment)
    - `pub tri_b: Vec<(TriangleId, TriConstraints)>` (sorted by `TriangleId`, same for `tool`)
    - `pub seam_edges: Vec<(ArrVertexId, ArrVertexId)>` (sorted, deduplicated `(min, max)` pairs of every intersection/touch segment across A and B)
  - `pub(super) fn build_arrangement(a: &Mesh, bvh_a: &Bvh, tool: &Mesh, bvh_tool: &Bvh) -> Result<Arrangement>`
    - Uses exact `orient3d` / `orient2d` on `f64`-widened `[f32; 3]` coordinates per §5.2–§5.4 of the spec:
      1. Allocates `ArrVertexId`s for touched live vertices of A (`ArrOrigin::LiveA(va)`) and `tool` (`ArrOrigin::Fresh`, unified directly to `ArrVertexId` of `va` when `a.position(va) == tool.position(vb)` bit-for-bit).
      2. Classifies every pair from `intersecting_pairs` into canonical `VertexKey`s (`OrigA`, `OrigB`, `VertOnEdgeA`, `VertOnEdgeB`, `EdgeEdge`, `EdgeTriA`, `TriEdgeB`), computing each key's `[f32; 3]` position once from canonical sorted vertex/edge order.
      3. Unifies any `ArrVertexId`s that quantize to identical `[f32; 3]` bit patterns (`[u32; 3]`, preferring `LiveA` when one of the unified vertices is a live vertex of A!) and remaps all `edge_splits`, `inner_verts`, `segments`, and `seam_edges`, dropping zero-length `(u, u)` segments and deduplicating.

- [ ] **Step 1: Write the failing unit tests in `crates/fast-mesh-ops/src/boolean/arrangement.rs`**

Add `#[cfg(test)] mod tests` testing:
- `cube_offset_shares_exact_seam_vertices_without_gaps`: `box_mesh([0,0,0],[1,1,1])` vs `box_mesh([0.5, 0.25, 0.125], [1.5, 1.25, 1.125])` -> `seam_edges` forms a single closed loop (every vertex in `seam_edges` has degree 2!) and every `seam_edges` entry appears in both `tri_a` (or A's split boundary edges) and `tri_b`!
- `curved_spheres_form_closed_degree_two_seam_loop`: `uv_sphere(32, 32, 1.0)` vs same sphere shifted by `(0.31, 0.17, 0.23)` -> `seam_edges` is non-empty and every vertex incident to `seam_edges` has even degree (`== 2`), proving zero 1-ULP gaps!
- `coplanar_stacked_cubes_unify_shared_corners_and_edges`: `box_mesh([0,0,0],[1,1,1])` vs `box_mesh([0,0,1],[1,1,2])` -> the 4 shared corners reuse `ArrOrigin::LiveA` and shared edges carry no duplicate coincident vertices.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p fast-mesh-ops boolean::arrangement`
Expected: FAIL to compile (`arrangement` module not implemented yet).

- [ ] **Step 3: Implement `crates/fast-mesh-ops/src/boolean/arrangement.rs`**

Carefully verify the right-hand `orient3d` sign convention in a small helper (`orient3d_exact(a, b, c, d)` matching `fast_mesh_query::tritri`: `robust::orient3d(a, b, c, d)` is `det[a-d, b-d, c-d] = -det[b-a, c-a, d-a]`), and ensure `edge_splits` propagate to BOTH triangles sharing an undirected edge on A (and on B) even if the neighbor triangle was not itself in `intersecting_pairs` (e.g., when an endpoint touches an edge from one side).

- [ ] **Step 4: Run tests and clippy to verify they pass**

Run: `cargo test -p fast-mesh-ops boolean::arrangement && cargo clippy -p fast-mesh-ops --all-targets -- -D warnings`
Expected: PASS, 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/fast-mesh-ops/src/boolean.rs crates/fast-mesh-ops/src/boolean/arrangement.rs
git commit -m "feat(ops): add canonical simplex intersection arrangement"
```

---

### Task 3: Constrained 2D Sub-Triangulation (`boolean/cdt.rs`)

**Files:**
- Create: `crates/fast-mesh-ops/src/boolean/cdt.rs`
- Modify: `crates/fast-mesh-ops/src/boolean.rs` (declare `mod cdt;`)

**Interfaces:**
- Consumes: `crate::boolean::arrangement::{ArrOrigin, ArrVertexId, TriConstraints}`, `robust::{Coord, orient2d}`, `crate::{OpsError, Result, push, reserve}`.
- Produces (in `crate::boolean::cdt`):
  - `pub(super) fn triangulate_constrained(corners: [ArrVertexId; 3], constraints: &TriConstraints, positions: &mut Vec<[f32; 3]>, origins: &mut Vec<ArrOrigin>, seam_edges: &mut Vec<(ArrVertexId, ArrVertexId)>) -> Result<Vec<[ArrVertexId; 3]>>`
    - Projects onto the triangle's dominant normal axis in `f64` with positive CCW 2D orientation (`orient2d(c0, c1, c2) > 0`).
    - Resolves any collinear point-on-segment splits and any strict 2D segment-segment crossings (allocating a new `ArrOrigin::Fresh` vertex in `positions`/`origins` and splitting both crossing segments in `seam_edges` if two constraint segments cross strictly in 2D).
    - Inserts all boundary-edge split vertices and interior vertices into a 2D triangulation and enforces every constraint segment via cavity re-triangulation (`orient2d`-exact ear clipping on the left/right cavity polygons), returning sub-triangles with the parent triangle's 3D winding and zero dropped boundary/constraint edges.

- [ ] **Step 1: Write the failing unit tests in `crates/fast-mesh-ops/src/boolean/cdt.rs`**

Add `#[cfg(test)] mod tests` testing:
- `single_interior_point_splits_into_three_ccw_triangles`: triangle `(0,0,0), (1,0,0), (0,1,0)` with interior point `(0.25, 0.25, 0)` -> 3 triangles, total area `0.5`, matching winding.
- `polyline_chain_across_triangle_preserves_all_constraints_and_boundary`: 4-segment chain from edge 0 through 3 interior points to edge 1 -> every constraint segment appears as an undirected edge in the output triangles, and outer boundary half-edges match the split ring 1-for-1.
- `crossing_segments_insert_intersection_and_recover_both`: two crossing constraint diagonals inside a triangle -> allocates 1 new `Fresh` vertex, splits both constraints in `seam_edges`, and emits valid manifold sub-triangles with total area equal to parent area.
- `near_collinear_edge_and_inner_points_leave_no_holes`: interior point `1e-6` from a split boundary edge -> zero unmatched interior half-edges.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p fast-mesh-ops boolean::cdt`
Expected: FAIL to compile (`cdt` module not implemented yet).

- [ ] **Step 3: Implement `crates/fast-mesh-ops/src/boolean/cdt.rs`**

Implement point-in-triangle / point-on-edge incremental insertion + cavity edge recovery using `robust::orient2d`. Ensure collinear zero-area triangles created during point insertion are resolved by splitting the containing triangle edge rather than dropping triangles, so the planar Euler half-edge invariant holds strictly on every cut triangle.

- [ ] **Step 4: Run tests and clippy to verify they pass**

Run: `cargo test -p fast-mesh-ops boolean::cdt && cargo clippy -p fast-mesh-ops --all-targets -- -D warnings`
Expected: PASS, 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/fast-mesh-ops/src/boolean.rs crates/fast-mesh-ops/src/boolean/cdt.rs
git commit -m "feat(ops): add per-triangle constrained 2D triangulation"
```

---

### Task 4: Patch Classification, 2-Manifold De-Pinching & Boolean Wiring (`boolean/classify.rs` & `boolean.rs`)

**Files:**
- Create: `crates/fast-mesh-ops/src/boolean/classify.rs`
- Modify: `crates/fast-mesh-ops/src/boolean.rs`

**Interfaces:**
- Consumes: Tasks 1–3 (`signed_volume6`, `WindingTree`, `Arrangement`, `ArrOrigin`, `ArrVertexId`, `build_arrangement`, `triangulate_constrained`), `fast_mesh_bvh::{Aabb, Bvh}`, `fast_mesh_store::{Mesh, Transaction, TriangleId, UV_CHANNEL, VertexId}`.
- Produces:
  - Updated `boolean_plan` and `apply_boolean` in `crates/fast-mesh-ops/src/boolean.rs` implementing Spec §5.1, §5.6, and §5.7:
    1. `check_input` rejects closed inputs with `signed_volume6(mesh) <= 0.0` (when `mesh.triangle_count() > 0`) as `OpsError::InvalidInput("boolean input is wound inward")`. When `xf != Rigid::IDENTITY`, `map_tool` output is also checked for degeneracy/self-intersection.
    2. Disjoint fast path (`arrangement.tri_a.is_empty() && arrangement.tri_b.is_empty() && seam_edges.is_empty()`): classifies whole components via `WindingTree` without building sub-triangulations.
    3. Coplanar sub-triangle classification via BVH AABB overlap + exact `orient3d == 0` & 2D centroid-in-triangle check: drops all coplanar sub-triangles on B; on A keeps `flush` (`dot > 0`) for `Union`/`Intersect` and `opposing` (`dot < 0`) for `Subtract`.
    4. Connected non-coplanar patch flood-fill across non-seam edges (spanning both untouched mesh triangles and CDT sub-triangles), evaluating `WindingTree::winding_number` once at the largest-area triangle of each patch (`w < -0.5` -> `Err(OpsError::InvalidInput("boolean input is wound inward"))`, `inside = w > 0.5`).
    5. Cancels exact opposite duplicate triangles `(a, b, c)` / `(a, c, b)`, decomposes any `k >= 2` balanced pinch edges (`count(u->v) == count(v->u) >= 2`) and multi-fan pinch vertices into separate 2-manifold fans via cyclic dihedral pairing around edges, verifies `count(u->v) == 1 && count(v->u) == 1` on every surviving undirected edge (else `Err(OpsError::InvalidInput("boolean arrangement failed manifold check"))`), and populates `BooleanPlan`.
    6. `apply_boolean` applies `drop_a`, `fresh`, `rewrite`, `tool`, and removes orphaned vertices of A (`corner_count(v) == 0`) with no fuzzy `weld(tol)` or `stitch`.

- [ ] **Step 1: Write the failing unit tests in `crates/fast-mesh-ops/src/boolean.rs`**

Add tests alongside the existing `boolean.rs` tests:
- `rejects_inward_wound_cube`: inverted `cube()` as A or B returns `Err(OpsError::InvalidInput("boolean input is wound inward"))`.
- `stacked_cubes_intersect_is_empty_and_subtract_keeps_a`: `cube()` vs `box_mesh([0,0,1],[1,1,2])` -> `Intersect` yields `0` triangles (not an open coplanar sheet!), `Subtract` yields `closed == true` and `volume6 ≈ 6.0`.
- `edge_and_vertex_touching_cubes_are_strictly_manifold`: `cube()` vs `box_mesh([1,1,0],[2,2,1])` (edge touch) and `box_mesh([1,1,1],[2,2,2])` (vertex touch) for `Union`, `Subtract`, `Intersect` -> `crate::measure(&a).unwrap()` has `closed == true` (when `triangle_count() > 0`), `non_manifold_edges == 0`, `orientation_conflicts == 0`, and `analyze` reports `pinch_vertices.is_empty()`.
- `hollow_shell_boolean_succeeds`: `shell()` sphere intersected/subtracted with an offset cube -> `measure(&a).unwrap().closed == true` and `orientation_conflicts == 0`.
- `rotated_tool_boolean_matches_analytic_volume`: cube rotated by 37° around `(1, 2, 3)` and translated to `(0.4, 0.3, 0.2)` -> `Union`, `Subtract`, `Intersect` all closed, 0 orientation conflicts, and `V(A ∪ B) + V(A ∩ B) ≈ V(A) + V(B)` within `1e-4`.
- `boolean_oom_sweep_is_atomic`: `fail_allocations_after(Some(step))` loop over overlapping spheres leaves `a` untouched on every `OutOfMemory` until `Ok`.

- [ ] **Step 2: Run tests to verify the new tests fail**

Run: `cargo test -p fast-mesh-ops boolean::tests`
Expected: FAIL on `rejects_inward_wound_cube`, `stacked_cubes_intersect_is_empty_and_subtract_keeps_a`, and `edge_and_vertex_touching_cubes_are_strictly_manifold`.

- [ ] **Step 3: Implement `crates/fast-mesh-ops/src/boolean/classify.rs` and wire `crates/fast-mesh-ops/src/boolean.rs`**

Replace the old `split_side` / `classify_side` / `stitch` / `parity` code in `boolean.rs` with the new pipeline while keeping `BooleanOp`, `BooleanPlan`, `BooleanReport`, `boolean_plan`, and `apply_boolean` intact.

- [ ] **Step 4: Run `fast-mesh-ops`, `fast-mesh-scene`, and `fast-mesh-cli` tests**

Run: `cargo test -p fast-mesh-ops -p fast-mesh-scene -p fast-mesh-cli && cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS, 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/fast-mesh-ops/src/boolean.rs crates/fast-mesh-ops/src/boolean/classify.rs
git commit -m "fix(ops): replace boolean splitter with canonical manifold arrangement (#165)"
```

---

### Task 5: Issue #165 Sweeps (576 Cubes + Curved Meshes), Fuzz Gate & 1M Benchmark

**Files:**
- Create: `crates/fast-mesh-ops/tests/boolean_sweeps.rs`
- Modify: `fuzz/fuzz_targets/boolean.rs`
- Modify: `crates/fast-mesh-ops/benches/ops.rs`
- Modify: `docs/benchmarks.md`
- Create: `.changeset/boolean-arrangement-fix.md`

**Interfaces:**
- Consumes: `fast_mesh_ops::{BooleanOp, apply_boolean, boolean_plan, measure}`, `fast_mesh_bvh::{BuildOptions, Bvh}`, `fast_mesh_query::Rigid`.
- Produces: Regression suite covering every case in Issue #165, fuzz invariant assertions, and 1M-triangle boolean benchmark.

- [ ] **Step 1: Write the Issue #165 regression suite in `crates/fast-mesh-ops/tests/boolean_sweeps.rs`**

Include:
1. `cube_diagonal_and_offset_sweep_576`: Two unit cubes with varied 6-bit face diagonal masks (e.g., 12 mask pairs × 4 offsets `[(0.5, 0.25, 0.125), (0.37, 0.41, 0.53), (0.5, 0.0, 0.0), (0.25, 0.25, 0.25)]` × 3 ops = 144, plus full 64×3×3 = 576 mask sweep at `(0.5, 0.25, 0.125)` and varied offsets), asserting `m.closed`, `m.non_manifold_edges == 0`, `m.orientation_conflicts == 0`, and `(m.volume.unwrap() - expected).abs() < 1e-5`.
2. `issue_165_curved_mesh_matrix`: Exact generator from Issue #165 (`sphere(1.0, 32, 32)` + `(0.31, 0.17, 0.23)`, `sphere(1.0, 32, 32)` vs `sphere(0.9, 39, 43)` + `(0.31, 0.17, 0.23)`, `torus(1.0, 0.4, 48, 24)` vs `sphere(0.9, 39, 43)` + `(0.31, 0.17, 0.23)`, `sphere(1.0, 32, 32)` + `(0.31, 0.0, 0.0)`, and identical `sphere(1.0, 32, 32)` at `(0, 0, 0)`) across `Union`, `Subtract`, `Intersect` — asserting closedness, `orientation_conflicts == 0`, and analytic lens volume within discretization tolerance (`< 1.5%`).
3. `curved_50k_overlap`: `sphere(1.0, 158, 158)` (49,612 tris) vs `sphere(0.9, 165, 169)` + `(0.31, 0.17, 0.23)` (55,432 tris) for `Union`, `Subtract`, `Intersect`.

- [ ] **Step 2: Tighten `fuzz/fuzz_targets/boolean.rs` invariants**

When `apply_boolean(&mut tx, &plan)` returns `Ok(_)`, commit `tx` and assert:
```rust
let m = fast_mesh_ops::measure(&a).expect("measure");
if a.triangle_count() > 0 {
    assert!(m.closed, "open or non-manifold boolean output: {m:?}");
    assert_eq!(m.orientation_conflicts, 0, "orientation conflicts: {m:?}");
}
```

- [ ] **Step 3: Add 1M-triangle overlapping sphere boolean benchmark to `crates/fast-mesh-ops/benches/ops.rs` and run verification**

Add `boolean_1m_spheres` (`sphere(1.0, 708, 708)` ~1M tris vs shifted `(0.31, 0.17, 0.23)`) to `benches/ops.rs`, run `cargo test --release -p fast-mesh-ops --test boolean_sweeps` and `cargo bench -p fast-mesh-ops -- boolean`, record the measured 1M timing in `docs/benchmarks.md`, and add `.changeset/boolean-arrangement-fix.md` (`patch` for `fast-mesh-wasm`).

- [ ] **Step 4: Run full workspace tests, WASM build, JS tests, and lint gate**

Run:
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `pnpm install --frozen-lockfile --prefer-offline && pnpm build && pnpm --filter fast-mesh-wasm test`
Expected: All Rust tests (including 576-cube sweep and curved matrix), clippy, WASM build, and JS tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/fast-mesh-ops/tests/boolean_sweeps.rs fuzz/fuzz_targets/boolean.rs crates/fast-mesh-ops/benches/ops.rs docs/benchmarks.md .changeset/boolean-arrangement-fix.md
git commit -m "test(ops): add #165 cube/curved sweeps, fuzz invariants and 1M boolean bench"
```
