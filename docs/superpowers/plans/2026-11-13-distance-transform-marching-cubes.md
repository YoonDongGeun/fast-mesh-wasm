# Selectable Distance Transforms & Marching Cubes Offset/Remesh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement reusable 3D grid distance transforms (Felzenszwalb–Huttenlocher 3D EDT and exact BVH-SDF), a watertight 2-manifold Marching Cubes iso-surface extractor, and selectable surface/volumetric offset & voxel-remesh algorithms (`normal`, `fhEdt`, `bvhSdf`) across Rust, WASM/JS, `WorkerScene`, and `examples/showcase`.

**Architecture:** `fast-mesh-fields` gains `GridSpec`, `ScalarGrid`, separable 3D Felzenszwalb–Huttenlocher parabolic-envelope EDT (`edt_3d_sq`, `edt_3d_mask`, `fh_signed_grid`, `fh_unsigned_grid`), and narrow-band BVH distance grids (`bvh_signed_grid`, `bvh_unsigned_grid`). `fast-mesh-ops` depends on `fast-mesh-fields`, adds a face-consistent 256-case Manifold Marching Cubes extractor (`marching_cubes`), and extends `offset` with `OffsetAlgorithm` (`Normal`, `FhEdt`, `BvhSdf`) supporting signed solid offsets/voxel-remeshing on closed meshes and unsigned thickened shells on open sheets. `fast-mesh-wasm`, `packages/fast-mesh-wasm`, and `examples/showcase` expose the selector, optional voxel size, and extended report end-to-end with full Undo/Redo and `WorkerScene` parity.

**Tech Stack:** Rust (`fast-mesh-fields`, `fast-mesh-ops`, `fast-mesh-wasm`, `robust` 1.2.0), TypeScript / WebAssembly (`packages/fast-mesh-wasm`), React 19 + Three.js + Vitest (`examples/showcase`).

**Spec:** `docs/superpowers/specs/2026-11-13-distance-transform-marching-cubes-design.md`

## Global Constraints

- Every growable allocation in `fast-mesh-fields` and `fast-mesh-ops` must use fallible reservation (`try_reserve` / `try_reserve_exact` in `fast-mesh-fields` mapping to `FieldsError::OutOfMemory`, and `crate::{reserve, push, filled, allocation_allowed}` in `fast-mesh-ops` mapping to `OpsError::Store(StoreError::OutOfMemory)`).
- Never use `HashMap` or `HashSet` in non-test code; never use `slice::sort` / `slice::sort_by` / `slice::sort_by_key` (use `sort_unstable` / `sort_unstable_by` / `sort_unstable_by_key` so sorting never allocates untracked heap memory).
- Every kernel with a `--features parallel` path (`edt_3d_sq`, `fh_signed_grid`, `fh_unsigned_grid`, `bvh_signed_grid`, `bvh_unsigned_grid`, `marching_cubes`) must produce bit-for-bit identical output with `parallel` on or off and across thread counts (`1, 2, 8`).
- Grid allocations must enforce `max_voxels` (`DEFAULT_MAX_VOXELS = 16_777_216`) via checked multiplication before reserving memory.
- Operations on `Transaction` (`offset`) must compute and validate their full plan before the first `tx` write so any error or OOM leaves the mesh untouched; when `FhEdt` or `BvhSdf` replaces topology on a mesh with UVs, `tx.drop_channel(UV_CHANNEL)` must be called and `OffsetReport::uvs_dropped` set to `true`.
- Zero compiler or Clippy warnings under `cargo clippy --workspace --all-targets -- -D warnings` and clean `cargo fmt --all --check`.

## Review Focus

- All-infinite or empty 1D slices in `edt_3d_sq` / `edt_3d_mask` must skip non-finite sites before parabola intersection arithmetic so `Inf - Inf = NaN` never occurs and empty grids return `f32::INFINITY` everywhere (pinned in Task 1: `edt_all_infinite_and_single_site_exact`).
- Z-column parity rays in `fh_signed_grid` grazing shared triangle edges or vertices of a grid-aligned cube must toggle parity cleanly once via exact `robust::orient2d` + Simulation-of-Simplicity half-open tie-breaking, never leaking interior columns outside (pinned in Task 1: `fh_signed_grid_grazing_rays_on_grid_aligned_cube`).
- Deep interior voxels beyond `narrow_band` in `bvh_signed_grid` receive `+f32::INFINITY` from `signed_distances` and must propagate negative interior sign `-narrow_band` rather than `+narrow_band`, preventing phantom inner iso-surfaces (pinned in Task 1: `bvh_signed_grid_narrow_band_has_no_inner_sign_flip`).
- Grid values equal to `iso_value` and adjacent `0101`/`1010` saddle faces in `marching_cubes` must never emit open boundary edges, zero-area triangles, non-manifold edges/vertices, or orientation conflicts (pinned in Task 2: `marching_cubes_all_adjacent_face_pairs_and_exact_iso_ties_are_manifold`).
- Closed meshes split along UV seams must still be treated as closed solids by `FhEdt`/`BvhSdf` (allowing `distance < 0` inward shrink and undoably dropping `UV_CHANNEL`), while inward-wound closed meshes (`signed_volume <= 0`) and open sheets with `distance <= 0` fail before mutating `tx` (pinned in Task 3: `volumetric_offset_uv_seamed_sphere_shrinks_and_undoes_uvs` and `volumetric_offset_rejects_inward_solid_and_nonpositive_open_sheet`).

---

### Task 1: 3D Regular Grid, Felzenszwalb–Huttenlocher 3D EDT & BVH Distance Grids (`fast-mesh-fields`)

**Files:**
- Modify: `crates/fast-mesh-fields/Cargo.toml`
- Modify: `Cargo.lock`
- Create: `crates/fast-mesh-fields/src/grid.rs`
- Create: `crates/fast-mesh-fields/src/edt.rs`
- Modify: `crates/fast-mesh-fields/src/lib.rs`

**Interfaces:**
- Consumes: `fast_mesh_store::{Mesh, par}`, `fast_mesh_bvh::Bvh`, `fast_mesh_query::{ClosestOptions, closest_point_many}`, `crate::{FieldsError, Result, SignData, signed_distances}`, `robust::orient2d`.
- Produces (re-exported from `fast_mesh_fields`):
  - `pub const DEFAULT_MAX_VOXELS: usize = 16_777_216;`
  - `pub const DEFAULT_AUTO_GRID_RES: usize = 64;`
  - `#[derive(Debug, Clone, Copy, PartialEq)] pub struct GridSpec { pub origin: [f32; 3], pub dims: [usize; 3], pub voxel_size: f32 }`
    - `GridSpec::validate(&self, max_voxels: usize) -> Result<usize>`
    - `GridSpec::around_bounds(min: [f32; 3], max: [f32; 3], voxel_size: f32, margin_distance: f32, pad_voxels: usize, max_voxels: usize) -> Result<Self>`
    - `GridSpec::index(&self, ix: usize, iy: usize, iz: usize) -> usize` (`(iz * ny + iy) * nx + ix`)
    - `GridSpec::point(&self, ix: usize, iy: usize, iz: usize) -> [f32; 3]`
  - `#[derive(Debug, Clone, PartialEq)] pub struct ScalarGrid`
    - `ScalarGrid::new(spec: GridSpec, values: Vec<f32>, max_voxels: usize) -> Result<Self>`
    - `ScalarGrid::spec(&self) -> GridSpec`
    - `ScalarGrid::values(&self) -> &[f32]`
    - `ScalarGrid::into_values(self) -> Vec<f32>`
    - `ScalarGrid::sample(&self, p: [f32; 3]) -> Option<f32>`
    - `ScalarGrid::sample_with_gradient(&self, p: [f32; 3]) -> Option<(f32, [f32; 3])>`
  - `pub fn edt_3d_sq(dims: [usize; 3], cost: &[f32], out_sq: &mut [f32]) -> Result<()>`
  - `pub fn edt_3d_mask(dims: [usize; 3], mask: &[bool], voxel_size: f32, out: &mut [f32]) -> Result<()>`
  - `pub fn fh_unsigned_grid(mesh: &Mesh, spec: GridSpec, max_voxels: usize) -> Result<ScalarGrid>`
  - `pub fn fh_signed_grid(mesh: &Mesh, spec: GridSpec, max_voxels: usize) -> Result<ScalarGrid>`
  - `pub fn bvh_unsigned_grid(mesh: &Mesh, bvh: &Bvh, spec: GridSpec, narrow_band: f32, max_voxels: usize) -> Result<ScalarGrid>`
  - `pub fn bvh_signed_grid(mesh: &Mesh, bvh: &Bvh, sign: &SignData, spec: GridSpec, narrow_band: f32, max_voxels: usize) -> Result<ScalarGrid>`

- [ ] **Step 1: Add `robust = "1.2.0"` to `crates/fast-mesh-fields/Cargo.toml` and `Cargo.lock`, and write failing unit tests in `grid.rs` and `edt.rs`**

Include tests:
- `grid_spec_validation_and_trilinear_gradient`: rejects `dims < 2`, `voxel_size <= 0`, overflow / `> max_voxels`, non-finite bounds; verifies `sample` and `sample_with_gradient` reproduce linear field `f(x, y, z) = 2x - 3y + 0.5z + 1` to `1e-5`.
- `edt_all_infinite_and_single_site_exact`: all-`INFINITY` input yields all-`INFINITY` with zero `NaN`s; random sparse masks and weighted `cost` grids on `9 × 11 × 7` match brute-force $\min_q ((x - q_x)^2 + (y - q_y)^2 + (z - q_z)^2 + \text{cost}[q])$ to `1e-4`.
- `fh_signed_grid_grazing_rays_on_grid_aligned_cube`: `cube()` `[-1, 1]³` on a grid whose `(ix, iy)` coordinates land directly on `x = -1, 0, 1` and `y = -1, 0, 1` (grazing cube faces, edges, diagonals, and corners) classifies `[0, 0, 0]` strictly negative (`<= -0.5` within `voxel_size`) and exterior corners strictly positive, with zero leaked columns.
- `fh_unsigned_grid_on_open_terrain_sheet`: open `terrain(8)` sheet gives `0.0` at surface-overlapped voxels and grows monotonically with distance above and below the sheet.
- `bvh_signed_grid_narrow_band_has_no_inner_sign_flip`: `cube()` `[-1, 1]³` with `voxel_size = 0.1` and `narrow_band = 0.35` matches `narrow_band = f32::INFINITY` in sign at every single grid node (`(a < 0.0) == (b < 0.0)`), with `[0, 0, 0]` equal to `-0.35` (not `+0.35` or `+INFINITY`).
- `edt_and_grids_parallel_bit_identical`: runs `edt_3d_sq`, `fh_signed_grid`, `fh_unsigned_grid`, `bvh_signed_grid`, and `bvh_unsigned_grid` on a `40 × 40 × 40` grid across forced-sequential and `par::with_threads(1/2/8)` asserting exact `to_bits()` equality.

- [ ] **Step 2: Run `cargo test -p fast-mesh-fields` to confirm compilation/test failure before implementation**

Run: `cargo test --locked --offline -p fast-mesh-fields`
Expected: FAIL (missing modules/functions).

- [ ] **Step 3: Implement `grid.rs`, `edt.rs`, and re-exports in `crates/fast-mesh-fields/src/lib.rs`**

Key implementation details:
- In `edt.rs`, 1D lower envelope skips sites where `!f[q].is_finite() || f[q] >= 1e30`; if no site in the line is finite, fills the line with `1e30` (`f32::INFINITY` at output).
- In `fh_signed_grid`, Z-column parity uses exact `robust::orient2d` on a lexicographically oriented triangle in XY with strict Simulation-of-Simplicity perturbation $\varepsilon_x = \epsilon, \varepsilon_y = \epsilon^2$ (half-open rule: an edge $(A, B)$ with $(A_y, A_x) < (B_y, B_x)$ includes the ray iff `(A_y <= r_y && r_y < B_y)` and `orient2d(A, B, r) > 0`, ensuring every vertical/horizontal/diagonal shared edge and vertex is counted at most once across adjacent triangles) and interpolates $z$ from the triangle plane in `f64`.
- In `bvh_signed_grid`, after `signed_distances(..., narrow_band, &mut values)`, resolve `f32::INFINITY` nodes via 6-neighbor BFS queue seeded from all finite nodes (carrying the sign `+narrow_band` or `-narrow_band` of the seeding finite node; if zero nodes are finite, fill `+narrow_band`). Because `narrow_band >= 2.0 * voxel_size`, finite negative and finite positive bands separate inside from outside across every 6-connected path, and BFS sign propagation is deterministic regardless of thread count.

- [ ] **Step 4: Run `cargo test` and `cargo clippy` on `fast-mesh-fields` (default and `--features parallel`)**

Run: `cargo test --locked --offline -p fast-mesh-fields` and `cargo test --locked --offline -p fast-mesh-fields --features parallel` and `cargo clippy --locked --offline -p fast-mesh-fields --all-targets --features parallel -- -D warnings`
Expected: PASS with 0 warnings.

---

### Task 2: Face-Consistent 256-Case Manifold Marching Cubes (`crates/fast-mesh-ops/src/marching_cubes.rs`)

**Files:**
- Create: `crates/fast-mesh-ops/src/marching_cubes.rs`
- Modify: `crates/fast-mesh-ops/src/lib.rs` (declare `pub mod marching_cubes;` and re-export `IsoMesh`, `marching_cubes_raw`)

**Interfaces:**
- Consumes: `crate::{OpsError, Result, filled, push, reserve, allocation_allowed}`, `fast_mesh_store::par`.
- Produces (`crates/fast-mesh-ops/src/marching_cubes.rs`):
  - `#[derive(Debug, Clone, Default, PartialEq)] pub struct IsoMesh { pub positions: Vec<[f32; 3]>, pub triangles: Vec<[u32; 3]> }`
  - `pub fn marching_cubes_raw(origin: [f32; 3], dims: [usize; 3], voxel_size: f32, values: &[f32], iso_value: f32) -> Result<IsoMesh>`

- [ ] **Step 1: Write failing unit tests in `crates/fast-mesh-ops/src/marching_cubes.rs`**

Include tests:
- `marching_cubes_all_adjacent_face_pairs_and_exact_iso_ties_are_manifold`:
  1. Iterates all 256 cube masks (`0..=255`) embedded in a `3 × 3 × 3` zero-padded (`+1.0` outside, `-1.0` inside, `iso = 0.0`) grid and verifies `analyze` reports `loops.is_empty()`, `non_manifold_edges == 0`, `non_manifold_vertices == 0`, `orientation_conflicts == 0`, `degenerate_triangles == 0`, and positive `measure` volume on non-empty outputs.
  2. Generates 64 deterministic pseudo-random `6 × 6 × 6` grids (outer border `+1.0`, interior random in `{-1.0, 0.0, +1.0}` including exact `0.0 == iso_value` ties!) and asserts every extracted `IsoMesh` is closed, 2-manifold, conflict-free, and outward-wound (`signed volume > 0` when non-empty).
- `marching_cubes_sphere_volume_and_area_converge`: samples analytic sphere SDF $\phi(p) = \|p\| - 1.5$ on `[-2.2, 2.2]³` at `voxel_size = 0.1` (`iso = 0.0`) and asserts closed 2-manifold output with enclosed volume within 2% of $\frac{4}{3}\pi (1.5)^3$ and surface area within 2% of $4\pi (1.5)^2$.
- `marching_cubes_rejects_bad_inputs_and_honours_oom_hook`: rejects `dims < 2`, `values.len() != nx * ny * nz`, non-finite `origin` / `voxel_size <= 0` / `iso_value = NaN`, `NaN` grid value, and passes `fail_allocations_after` sweep from `0..`.
- `marching_cubes_parallel_bit_identical`: runs `marching_cubes_raw` on a `48 × 48 × 48` sphere/gyroid scalar field and verifies `assert_parallel_identical`.

- [ ] **Step 2: Run `cargo test -p fast-mesh-ops marching_cubes` to confirm failure**

Run: `cargo test --locked --offline -p fast-mesh-ops marching_cubes`
Expected: FAIL.

- [ ] **Step 3: Implement `marching_cubes_raw` in `crates/fast-mesh-ops/src/marching_cubes.rs`**

Key implementation details:
- Cube corners `c in 0..8` indexed `dx | (dy << 1) | (dz << 2)`; `inside(v) = v < iso_value` (`NaN` rejected up front before extraction).
- 12 cube edges `0..12` mapped to canonical `(dx0, dy0, dz0, axis)`.
- Build `const CASES: [CubeCase; 256]` via `const fn`:
  - For each of the 6 faces, list its 4 corners in CCW order viewed from outside the cube, identify active boundary edges (`0 -> 1` entry vs `1 -> 0` exit along the CCW walk), and on `0101`/`1010` saddle faces use the face's canonical 2D `(u, v)` coordinate bits (ignoring the face normal's sign!) to pair the two entry/exit pairs identically on `side = 0` and `side = 1` (reversing segment direction on `side = 1` vs `side = 0` because outward face normals are opposite).
  - Chain the directed face segments `from_edge -> to_edge` around the 12 cube edges into disjoint simple cycles `C` (`len >= 3`), and emit fan triangles `(C[0], C[i], C[i + 1])` for `1 <= i < C.len() - 1`.
- Deduplicate active lattice edges across the whole grid by deterministic sorted key `(iz, iy, ix, axis)` (or dense/sparse two-pass active edge scan) with `t = ((f64::from(iso_value) - f64::from(va)) / (f64::from(vb) - f64::from(va))).clamp(1e-4, 1.0 - 1e-4)`, so memory scales only with active surface area $O(\text{active edges})$, NOT $3 \cdot N_x N_y N_z$!

- [ ] **Step 4: Run `cargo test` and `cargo clippy` on `fast-mesh-ops` (default and `--features parallel`)**

Run: `cargo test --locked --offline -p fast-mesh-ops marching_cubes --features parallel` and `cargo clippy --locked --offline -p fast-mesh-ops --all-targets --features parallel -- -D warnings`
Expected: PASS with 0 warnings.

---

### Task 3: Wire `ScalarGrid` + Marching Cubes + Selectable `OffsetAlgorithm` in `fast-mesh-ops`

**Files:**
- Modify: `crates/fast-mesh-ops/Cargo.toml`
- Modify: `Cargo.lock`
- Modify: `fuzz/Cargo.lock`
- Modify: `crates/fast-mesh-ops/src/error.rs`
- Modify: `crates/fast-mesh-ops/src/marching_cubes.rs`
- Modify: `crates/fast-mesh-ops/src/offset.rs`
- Modify: `crates/fast-mesh-ops/src/lib.rs`
- Modify: `crates/fast-mesh-ops/src/uv_table.rs`
- Modify: `docs/uv-channel.md`

**Interfaces:**
- Consumes: Task 1's `fast_mesh_fields::{DEFAULT_AUTO_GRID_RES, DEFAULT_MAX_VOXELS, FieldsError, GridSpec, ScalarGrid, SignData, SignMode, bvh_signed_grid, bvh_unsigned_grid, fh_signed_grid, fh_unsigned_grid}` and Task 2's `marching_cubes_raw`.
- Produces (re-exported from `fast_mesh_ops`):
  - `impl From<fast_mesh_fields::FieldsError> for OpsError`
  - `pub fn marching_cubes(grid: &ScalarGrid, iso_value: f32) -> Result<IsoMesh>`
  - `#[derive(Debug, Clone, Copy, Default, PartialEq)] pub enum OffsetAlgorithm { #[default] Normal, FhEdt { voxel_size: Option<f32> }, BvhSdf { voxel_size: Option<f32> } }`
  - `pub struct OffsetOptions { pub distance: f32, pub algorithm: OffsetAlgorithm, pub max_voxels: usize }` + `OffsetOptionsBuilder::{distance, algorithm, max_voxels, build}`
  - `pub struct OffsetReport { pub moved: usize, pub max_displacement: f32, pub degenerate_removed: usize, pub vertices_removed: usize, pub vertices_added: usize, pub triangles_added: usize, pub triangles_removed: usize, pub voxel_size: f32, pub uvs_dropped: bool }`
  - `pub fn offset(tx: &mut Transaction, options: &OffsetOptions) -> Result<OffsetReport>`
  - `pub fn offset_with_bvh(tx: &mut Transaction, bvh: Option<&Bvh>, options: &OffsetOptions) -> Result<OffsetReport>`

- [ ] **Step 1: Add `fast-mesh-fields` dependency to `fast-mesh-ops` (`Cargo.toml`, `Cargo.lock`, `fuzz/Cargo.lock`) and write failing tests in `offset.rs` and `uv_table.rs`**

Include tests:
- `volumetric_offset_closed_cube_and_l_shape_are_manifold_and_self_intersection_free`:
  - Concave L-shaped closed solid offset inward and outward with large distance (`0.35`) where `Normal` self-intersects (`analyze` with BVH reports self-intersections > 0), whereas both `FhEdt` and `BvhSdf` produce closed, 2-manifold, conflict-free, **zero-self-intersection** meshes (`self_intersections == 0`) with expected grown/shrunk volumes!
  - `distance == 0.0` with `FhEdt` and `BvhSdf` voxel-remeshes a closed box into a closed 2-manifold of matching volume within voxel tolerance, while `Normal` at `0.0` remains a no-op.
- `volumetric_offset_open_sheet_capsules_and_rejects_nonpositive`:
  - Open `grid(4)` sheet with `distance = 0.25` under both `FhEdt` and `BvhSdf` produces a closed, manifold capsule (`loops.is_empty()`, positive volume $\approx 4 \times 4 \times 0.5$ plus rounded rims).
  - Open `grid(4)` sheet with `distance = 0.0` or `distance = -0.2` returns `Err(OpsError::InvalidInput(_))` and leaves `tx` empty.
- `volumetric_offset_uv_seamed_sphere_shrinks_and_undoes_uvs`:
  - Textured `with_planar(box_of_faces())` (closed geometrically, split at UV seams) shrinks inward cleanly with `distance = -0.15` under `BvhSdf` and `FhEdt`, drops `UV_CHANNEL` (`report.uvs_dropped == true`, `mesh.uvs().is_none()`), and `patch.revert(&mut mesh)` restores the original vertices, triangles, and exact UVs!
- `volumetric_offset_rejects_inward_solid_and_bad_voxel_size_and_honours_oom`:
  - Flipped inward-wound cube returns `Err(OpsError::InvalidInput("volumetric offset input is wound inward"))` with empty `tx`.
  - `voxel_size = Some(0.0)` / `Some(-0.1)` / `Some(f32::NAN)` or `max_voxels = 8` returns `Err` with empty `tx`.
  - `fail_allocations_after` sweep on `FhEdt` and `BvhSdf` leaves `tx` untouched on every OOM point.

- [ ] **Step 2: Run `cargo test -p fast-mesh-ops offset` to verify failure**

Run: `cargo test --locked --offline -p fast-mesh-ops offset`
Expected: FAIL.

- [ ] **Step 3: Implement `From<FieldsError>`, `marching_cubes(&ScalarGrid, ...)`, `OffsetAlgorithm`, `offset`, `offset_with_bvh`, and update `uv_table.rs` + `docs/uv-channel.md`**

Key implementation details:
- Automatic `voxel_size`: compute live triangle vertex AABB `[min, max]`, `diag = ||max - min||`; default `h = (diag / 64.0).max(1e-5)`, and if `around_bounds` exceeds `max_voxels` when `voxel_size` was `None`, scale `h` up minimally so `around_bounds` fits `max_voxels` (when explicit `Some(h)` exceeds `max_voxels`, return `OpsError::InvalidInput("grid exceeds maximum voxel count")`).
- Read-only topology check copy: build a compact `Mesh` of live proper triangles; if `tx.mesh().uvs().is_some()` (or duplicate positions exist), run `weld(&mut work_tx, 0.0)` on the working copy before `Bvh::build` + `SignData::build(&mut work, &work_bvh, SignMode::Auto)` so UV seams on closed surfaces classify as closed solids (`bvh` argument may be reused directly when `tx.mesh().uvs().is_none()` and `bvh.is_current(tx.mesh())`).
- Before mutating `tx`, collect live `TriangleId`s to remove, live `VertexId`s to remove, and reserve `mapped_tris`; then `tx.remove_triangles`, `tx.remove_vertices`, `tx.add_vertices(&iso.positions)`, `tx.add_triangles(&mapped_tris)`, and `if has_uvs { tx.drop_channel(UV_CHANNEL)?; }`.

- [ ] **Step 4: Run full workspace Rust tests & Clippy**

Run: `cargo test --locked --offline --workspace` and `cargo clippy --locked --offline --workspace --all-targets -- -D warnings`
Expected: PASS with 0 warnings.

---

### Task 4: WASM Bridge, TypeScript API (`Scene` / `WorkerScene` / `SceneStore`) & Package Tests

**Files:**
- Modify: `crates/fast-mesh-wasm/src/edit.rs`
- Modify: `packages/fast-mesh-wasm/src/scene.ts`
- Modify: `packages/fast-mesh-wasm/src/index.ts`
- Modify: `packages/fast-mesh-wasm/test/edit.test.ts`
- Create: `.changeset/distance-transform-marching-cubes.md`

**Interfaces:**
- Consumes: `fast_mesh_ops::{OffsetAlgorithm, OffsetOptions, OffsetReport, offset}` in `crates/fast-mesh-wasm/src/edit.rs`.
- Produces (exported from `fast-mesh-wasm`):
  - `export type OffsetAlgorithm = "normal" | "fhEdt" | "bvhSdf";`
  - `RepairOp` `"offset"` branch:
    `{ op: "offset"; distance: number; algorithm?: OffsetAlgorithm; voxelSize?: number; maxVoxels?: number }`
  - `export interface OffsetReport`:
    `{ moved: number; maxDisplacement: number; degenerateRemoved: number; verticesRemoved: number; verticesAdded: number; trianglesAdded: number; trianglesRemoved: number; voxelSize: number; uvsDropped: boolean }`

- [ ] **Step 1: Add failing tests in `packages/fast-mesh-wasm/test/edit.test.ts` and Rust unit tests in `crates/fast-mesh-wasm/src/edit.rs`**

Include tests:
- Decode/encode tests in `crates/fast-mesh-wasm/src/edit.rs` for code `11` (`offset`) covering default `NaN` (`Normal`), `1.0` (`FhEdt`), `2.0` (`BvhSdf`), invalid algorithm code (`3.0`), non-positive `voxelSize`, and `maxVoxels` bounds.
- Vitest cases in `packages/fast-mesh-wasm/test/edit.test.ts`:
  - Existing `"normal"` default behavior updated with the extended `OffsetReport` fields (`verticesAdded: 0, trianglesAdded: 0, trianglesRemoved: 0, voxelSize: 0, uvsDropped: false`).
  - Closed cube / concave mesh offset with `algorithm: "fhEdt"` and `algorithm: "bvhSdf"` (both explicit `voxelSize` and auto `voxelSize`), verifying closed manifold analysis (`loops.length === 0`, `nonManifoldEdges === 0`, `orientationConflicts === 0`), `distance: 0` voxel remeshing, and Undo/Redo restoring exact initial vertex/triangle counts.
  - Open quad sheet with `algorithm: "fhEdt"` and `distance: 0.2` producing a closed capsule (`loops.length === 0`), and `distance: -0.2` throwing `INVALID_ARGUMENT` without adding an undo step.
  - Mesh with UVs (`addMesh(..., { uvs })`) offset via `algorithm: "bvhSdf"` reporting `uvsDropped: true`, `views(m).uvs() === null`, and `undo()` restoring `views(m).uvs()` exactly.
  - `WorkerScene` parity test running `await workerScene.repair(m, { op: "offset", distance: 0.15, algorithm: "fhEdt", voxelSize: 0.1 })` and undo/redo.

- [ ] **Step 2: Implement wire encoding/decoding in `crates/fast-mesh-wasm/src/edit.rs` and `packages/fast-mesh-wasm/src/scene.ts` (`index.ts`), plus `.changeset/distance-transform-marching-cubes.md`**

Wire format for op code `11`:
- Inputs (`4 × f64`): `distance, algorithm (NaN or 0 -> Normal, 1 -> FhEdt, 2 -> BvhSdf), voxel_size (NaN -> None, else finite > 0), max_voxels (NaN -> DEFAULT_MAX_VOXELS, else integer >= 8)`.
- Report (`9 × f64`): `moved, max_displacement, degenerate_removed, vertices_removed, vertices_added, triangles_added, triangles_removed, voxel_size, uvs_dropped (0 or 1)`.
- Validate `algorithm`, `voxelSize`, and `maxVoxels` on the JS side (`codeIn` / positive finite checks throwing `INVALID_ARGUMENT`) as well as in Rust `decode_repair`.

- [ ] **Step 3: Build WASM + TS package and run Rust + Vitest suites**

Run: `cargo test --locked --offline -p fast-mesh-wasm`, `pnpm.cmd build`, `pnpm.cmd --filter fast-mesh-wasm test`, followed by `pnpm.cmd build` (restoring production `dist/` without `__test_panic`).
Expected: PASS (all Rust and Vitest tests green).

---

### Task 5: Showcase Offset, Shell & Voxel Remesh Controls (`examples/showcase`)

**Files:**
- Modify: `examples/showcase/src/panels/repair-summary.ts`
- Modify: `examples/showcase/src/panels/RepairPanel.tsx`
- Create or Modify: `examples/showcase/src/panels/__tests__/repair-offset.test.ts` (or existing showcase repair test file)

**Interfaces:**
- Consumes: `OffsetAlgorithm`, `OffsetReport`, `RepairOp`, `RepairReport` from `fast-mesh-wasm`; `LengthInput`, `formatLength`, `LengthUnit` from `./LengthInput` & `./units`.
- Produces:
  - Updated `describeReport("offset", report, unit)` in `repair-summary.ts`:
    - When `r.trianglesAdded > 0 || r.trianglesRemoved > 0` (volumetric `fhEdt` / `bvhSdf`): ``Offset: rebuilt ${plural(r.trianglesAdded, "triangle")}, ${plural(r.verticesAdded, "vertex", "vertices")} (voxel ${formatLength(r.voxelSize, unit)})`` (or `"Offset: cleared mesh"` if eroded to 0 triangles).
    - Otherwise (`normal`): existing ``Offset: moved ${plural(r.moved, "vertex", "vertices")} (max ${formatLength(r.maxDisplacement, unit)})``.
  - **Offset & Shell** section in `RepairPanel.tsx` with `data-offset-algorithm`, `data-offset-distance`, `data-offset-voxel-size`, `data-offset-run`, `data-shell-thickness`, and `data-shell-run` attributes for deterministic UI testing:
    - Algorithm `<select>`: `"normal"` (`Normal displacement`), `"fhEdt"` (`FH 3D EDT + Marching Cubes`), `"bvhSdf"` (`BVH SDF + Marching Cubes`).
    - Signed offset distance `<input type="number">` (unit-converted via `fromDisplayLength` / `lengthInputValue` so negative, `0`, and positive offsets work in `mm` / `in`) + optional Voxel size `<LengthInput allowEmpty placeholder="auto">` when `algorithm !== "normal"`.
    - Primary button calling `store.repair(mesh, { op: "offset", distance, algorithm, ...(voxelSize !== undefined ? { voxelSize } : {}) })` (disabled when `algorithm === "normal" && distance === 0`, enabled at `distance === 0` with label `"Voxel remesh"` when `algorithm !== "normal"`).
    - Shell wall thickness `<LengthInput>` + `"Shell"` button calling `store.repair(mesh, { op: "shell", thickness })`.

- [ ] **Step 1: Write failing Vitest tests in `examples/showcase` for `describeReport("offset", ...)` and the new Offset/Shell panel helpers**

Run: `pnpm.cmd --filter fast-mesh-wasm-example-showcase test`
Expected: FAIL before implementation.

- [ ] **Step 2: Implement `repair-summary.ts` and `RepairPanel.tsx` updates**

Keep styling consistent with existing `RepairPanel.tsx` `<details>` / tool rows (`className="row"`, `LengthInput`, `busy` / `own` / `report` state handling).

- [ ] **Step 3: Run Showcase tests and production build**

Run: `pnpm.cmd --filter fast-mesh-wasm-example-showcase test` and `pnpm.cmd --filter fast-mesh-wasm-example-showcase build`
Expected: PASS (`tsc --noEmit` + `vite build` + `vitest run` all exit 0).
