# Selectable Distance Transforms & Marching Cubes Offset/Remesh Design (Issue #191)

- **Issue:** [#191](https://github.com/YoonDongGeun/fast-mesh-wasm/issues/191)
- **Date:** 2026-11-13
- **Branch:** `feat/191-dt-marching-cubes`

## 1. Intent & Motivation

Downstream applications built on `fast-mesh-wasm` need both fast topology-preserving surface offsets and self-intersection-free volumetric offsets/remeshes, plus reusable 3D distance-transform grids for upcoming collision carving and soft-body contact (#194). Rather than hard-coding one offset heuristic, the engine exposes all three complementary strategies behind a runtime selector (`normal`, `fhEdt`, `bvhSdf`) alongside standalone `ScalarGrid`, Felzenszwalb–Huttenlocher 3D EDT, BVH-SDF grid, and Manifold Marching Cubes APIs, and demonstrates them interactively in `examples/showcase`.

## 2. Goals, Non-Goals & Anti-Goals

### Goals
1. **Reusable 3D Grid & Distance-Transform Kernels (`fast-mesh-fields`):**
   - `GridSpec` and `ScalarGrid` with `DEFAULT_MAX_VOXELS = 16_777_216` ($256^3$) safety cap, symmetric padded bounds `GridSpec::around_bounds`, trilinear interpolation `sample`, and central-difference `sample_with_gradient`.
   - Exact **Felzenszwalb–Huttenlocher (2012) 3D Euclidean Distance Transform** (`edt_3d_sq`, `edt_3d_mask`), plus mesh-seeded `fh_signed_grid` (closed solids via exact `robust::orient2d` Z-column ray parity + dual inside/outside EDT with half-voxel zero-crossing alignment) and `fh_unsigned_grid` (any mesh via exact Akenine-Möller SAT triangle-voxel overlap + binary EDT).
   - **Exact BVH Distance Grids** (`bvh_signed_grid` using narrow-band `signed_distances` + sign propagation to far components, and `bvh_unsigned_grid` using narrow-band `closest_point_many`).
   - Bit-for-bit identical output with and without `--features parallel` for any thread count.
2. **Manifold Marching Cubes (`fast-mesh-ops`):**
   - Self-contained 256-case **Manifold Marching Cubes** (`marching_cubes`) with canonical lattice-edge vertex deduplication (`(ix, iy, iz, axis)`), clamped interior edge interpolation `t.clamp(1e-4, 1.0 - 1e-4)`, and consistent 2D face-saddle disambiguation so every iso-surface strictly inside a padded grid is a closed, crack-free, outward-oriented 2-manifold (`boundary_edges == 0`, `non_manifold_edges == 0`, `non_manifold_vertices == 0`, `orientation_conflicts == 0`).
3. **Unified Selectable `offset` Operation (`fast-mesh-ops`, `fast-mesh-wasm`, JS/TS, Worker):**
   - `OffsetAlgorithm`: `Normal` (default, existing `p + d * n`), `FhEdt { voxel_size: Option<f32> }`, `BvhSdf { voxel_size: Option<f32> }`.
   - **Closed vs Open Mesh Policy (User-Selected Option 1):**
     - **Closed solid** (after merging exact `0.0`-coincident UV-seam vertices on a temporary read-only copy when UVs exist): signed offset where `distance > 0` expands outward, `distance == 0` voxel-remeshes the surface (for `FhEdt`/`BvhSdf`; no-op for `Normal`), and `distance < 0` shrinks inward (returning an empty mesh if eroded completely; rejecting inward-wound closed solids with `signed_volume <= 0`).
     - **Open sheet/surface:** `FhEdt` and `BvhSdf` with `distance > 0` generate a closed watertight thickened shell at unsigned distance `distance` around the sheet; `distance <= 0` returns `Err(OpsError::InvalidInput("open mesh volumetric offset requires positive distance"))`.
   - Single atomic `Transaction` replacement preserving Undo/Redo and dropping `UV_CHANNEL` undoably (`uvs_dropped = true`) when `FhEdt`/`BvhSdf` replaces mesh topology.
4. **Showcase Workbench Controls (`examples/showcase`):**
   - Expose **Offset, Shell & Voxel Remesh** in `RepairPanel.tsx` with algorithm picker (`normal` / `fhEdt` / `bvhSdf`), unit-aware distance and optional voxel pitch inputs, `distance = 0` voxel remesh support, and updated `repair-summary.ts` report formatting.

### Non-Goals
- Adaptive octree / Dual Contouring sharp-feature hermite grids (regular grid + Manifold Marching Cubes covers #191 and #194 cleanly).
- Texture UV re-projection onto Marching Cubes topology (`UV_CHANNEL` is explicitly dropped and restored on undo per `docs/uv-channel.md`).

### Anti-Goals
- Silent self-intersections, open cracks, or inverted normals on `FhEdt`/`BvhSdf` outputs.
- Unbounded grid allocations capable of panicking or exhausting 32-bit WASM linear memory (`GridSpec` enforces `max_voxels` and fallible `try_reserve` everywhere).
- Phantom inner shells on `bvh_signed_grid` from unsigned `+INFINITY` far cutoffs inside thick solids.

## 3. Architectural Approaches Considered

1. **Selected — Two-Layer Engine Split (`fast-mesh-fields` Grid/EDT/SDF + `fast-mesh-ops` Manifold Marching Cubes & `offset`):**
   - Keeps scalar distance grids and FH-EDT in `fast-mesh-fields` (zero dependency on `fast-mesh-ops`) and mesh reconstruction / transaction editing in `fast-mesh-ops` (which depends on `fast-mesh-fields`). Enables #194 `softDeform` to sample distance/gradients without mesh extraction while #194 volumetric `cut` reuses both layers.
2. **Single-Crate Private Module Inside `fast-mesh-ops`:**
   - Avoids touching `fast-mesh-fields`, but prevents standalone distance-grid queries and #194 scene-level collision fields from reusing FH-EDT without pulling in repair transactions.
3. **Rust Distance Array + JS Marching Tetrahedra (`isosurface.ts`):**
   - Copies multi-megabyte voxel buffers across WASM↔JS, inflates triangle counts by 2–3×, and cannot run inside Rust `Scene::edit` transactions or `fast-mesh-cli`.

## 4. Detailed Technical Specification

### 4.1 `crates/fast-mesh-fields`: `grid.rs` & `edt.rs`

- **Constants:**
  - `pub const DEFAULT_MAX_VOXELS: usize = 16_777_216;`
  - `pub const DEFAULT_AUTO_GRID_RES: usize = 64;`
- **`GridSpec`:**
  - `pub origin: [f32; 3]`, `pub dims: [usize; 3]`, `pub voxel_size: f32`.
  - Flat index `(iz * ny + iy) * nx + ix`, node position `[origin[0] + ix as f32 * h, origin[1] + iy as f32 * h, origin[2] + iz as f32 * h]`.
  - `pub fn validate(&self, max_voxels: usize) -> Result<usize>`: requires finite `origin`, finite positive `voxel_size`, `dims.iter().all(|&d| d >= 2)`, checked product `<= max_voxels`, and finite upper corner `origin[k] + (dims[k] - 1) as f32 * voxel_size`.
  - `pub fn around_bounds(min: [f32; 3], max: [f32; 3], voxel_size: f32, margin_distance: f32, pad_voxels: usize, max_voxels: usize) -> Result<Self>`: builds a grid centered on `(min + max) * 0.5` with `margin = margin_distance.max(0.0) + (pad_voxels.max(2) as f32) * voxel_size` on all 6 sides and `dims[k] >= 4`.
- **`ScalarGrid`:**
  - Wraps `spec: GridSpec` and `values: Vec<f32>`.
  - `pub fn sample(&self, p: [f32; 3]) -> Option<f32>`: trilinear interpolation when `p` lies inside `[origin, upper]`, `None` outside or non-finite.
  - `pub fn sample_with_gradient(&self, p: [f32; 3]) -> Option<(f32, [f32; 3])>`: trilinear value + analytic/central-difference gradient in local units.
- **Felzenszwalb–Huttenlocher 3D EDT (`edt.rs`):**
  - `pub fn edt_3d_sq(dims: [usize; 3], cost: &[f32], out_sq: &mut [f32]) -> Result<()>`:
    - Separable X → Y → Z 1D lower parabolic envelope in `f64`.
    - Sites with `!c.is_finite() || c >= EDT_INF_F32` (`EDT_INF = 1e30_f64`) are skipped before computing parabola intersections `((f[q] + q²) - (f[p] + p²)) / (2(q - p))`, preventing `Inf - Inf = NaN` and handling all-infinite lines cleanly (`out_sq` receives `f32::INFINITY` when no finite site exists in the whole grid).
    - Negative finite costs are rejected with `FieldsError::InvalidInput("edt cost must be non-negative")`.
  - `pub fn edt_3d_mask(dims: [usize; 3], mask: &[bool], voxel_size: f32, out: &mut [f32]) -> Result<()>`:
    - Seeds `0.0` where `mask[i]` is `true` and `f32::INFINITY` where `false`, runs `edt_3d_sq`, and writes `voxel_size * sq.sqrt()` (`f32::INFINITY` if `mask` is all `false`).
  - `pub fn fh_unsigned_grid(mesh: &Mesh, spec: GridSpec, max_voxels: usize) -> Result<ScalarGrid>`:
    - Marks every grid node whose `[p - 0.5 * h, p + 0.5 * h]` voxel overlaps any live non-degenerate triangle (exact `f64` Akenine-Möller SAT triangle-box test), then runs `edt_3d_mask`.
  - `pub fn fh_signed_grid(mesh: &Mesh, spec: GridSpec, max_voxels: usize) -> Result<ScalarGrid>`:
    - Classifies every `(ix, iy, iz)` grid node as inside/outside via Z-column ray parity over `(x_0 + ix * h, y_0 + iy * h)` using exact `robust::orient2d` with half-open top-left / lexicographic edge tie-breaking (Simulation of Simplicity) so rays grazing shared edges or vertices toggle parity cleanly once.
    - Runs two `edt_3d_sq` passes (`inside -> nearest outside` and `outside -> nearest inside`) and writes:
      - outside: `+h * (d_in.sqrt() - 0.5)`
      - inside: `-h * (d_out.sqrt() - 0.5)`
      (when the mesh occupies 0 grid nodes, falls back to `fh_unsigned_grid` so sub-voxel solids are not lost if `d > 0`, or all `+INFINITY` if empty).
- **BVH Distance Grids (`grid.rs`):**
  - `pub fn bvh_unsigned_grid(mesh: &Mesh, bvh: &Bvh, spec: GridSpec, narrow_band: f32, max_voxels: usize) -> Result<ScalarGrid>`:
    - Evaluates `closest_point_many` with `max_distance = narrow_band` (if finite; must be `> 0`), clamping `None` hits to `narrow_band`.
  - `pub fn bvh_signed_grid(mesh: &Mesh, bvh: &Bvh, sign: &SignData, spec: GridSpec, narrow_band: f32, max_voxels: usize) -> Result<ScalarGrid>`:
    - Evaluates `signed_distances` with `max_distance = narrow_band` (when `narrow_band` is finite, requires `narrow_band >= 2.0 * spec.voxel_size`), then resolves `f32::INFINITY` far voxels by flood-filling / propagating the sign of adjacent finite nodes (`+narrow_band` outside, `-narrow_band` inside; if no finite nodes exist, `+narrow_band`), guaranteeing zero false sign flips deep inside thick solids.

### 4.2 `crates/fast-mesh-ops`: `marching_cubes.rs` & `offset.rs`

- **`marching_cubes.rs`:**
  - `pub struct IsoMesh { pub positions: Vec<[f32; 3]>, pub triangles: Vec<[u32; 3]> }`
  - `pub fn marching_cubes(grid: &ScalarGrid, iso_value: f32) -> Result<IsoMesh>` (and slice-based `marching_cubes_raw(origin: [f32; 3], dims: [usize; 3], voxel_size: f32, values: &[f32], iso_value: f32) -> Result<IsoMesh>`).
  - Convention: `inside(v) = v < iso_value` (so for standard SDFs where inside `< 0` and outside `> 0`, `iso_value = d` encloses `{ x : phi(x) <= d }` with outward normals pointing toward `+grad(phi)`).
  - Canonical 256-entry compile-time table built from 2D face-consistent directed segments:
    - Each face `(axis, side)` orders its 4 corners in `(u, v)` coordinates of the two remaining axes so `side = 0` and `side = 1` make identical diagonal choices on `0101` / `1010` saddles (connecting active edges so inside corners `(0,0)` & `(1,1)` stay separated).
    - Active face segments chain into vertex-disjoint directed cycles on the 12 cube edges and fan-triangulate `(c[0], c[i], c[i + 1])`.
  - Lattice edge deduplication: canonical undirected grid edge `(ix, iy, iz, axis)` (`axis in 0..3`) mapped deterministically to a single `u32` vertex index, with `t = ((iso - va) / (vb - va)).clamp(1e-4, 1.0 - 1e-4)` computed in `f64` and narrowed to `f32`.
- **`offset.rs`:**
  - `OffsetAlgorithm`: `Normal`, `FhEdt { voxel_size: Option<f32> }`, `BvhSdf { voxel_size: Option<f32> }`.
  - `OffsetOptions`: `{ pub distance: f32, pub algorithm: OffsetAlgorithm, pub max_voxels: usize }` (`Default` -> `distance: 0.0`, `algorithm: OffsetAlgorithm::Normal`, `max_voxels: DEFAULT_MAX_VOXELS`).
  - `OffsetReport`:
    - `moved: usize`, `max_displacement: f32`, `degenerate_removed: usize`, `vertices_removed: usize`, `vertices_added: usize`, `triangles_added: usize`, `triangles_removed: usize`, `voxel_size: f32`, `uvs_dropped: bool`.
  - Automatic `voxel_size` when `None`: `bbox_diag / 64.0` (clamped upward if needed so `GridSpec::around_bounds` fits within `max_voxels`, and rejecting degenerate zero-extent meshes).
  - Closed vs open detection on `FhEdt` / `BvhSdf`:
    - Builds a compact read-only working mesh of live proper triangles; if `tx.mesh().uvs().is_some()`, welds exact `0.0`-coincident vertices on that working copy so UV seams on closed surfaces do not masquerade as open holes.
    - Builds `Bvh` + `SignData::build(..., SignMode::Auto)`.
    - If `status.closed && status.consistent_orientation`:
      - If `status.signed_volume <= 0.0`: returns `Err(OpsError::InvalidInput("volumetric offset input is wound inward"))`.
      - Builds `fh_signed_grid` or `bvh_signed_grid` (with `narrow_band = distance.abs() + 3.0 * h`) over `GridSpec::around_bounds(min, max, h, distance.max(0.0), 3, max_voxels)` and extracts `marching_cubes(&grid, distance)`.
    - Otherwise (open mesh or non-manifold boundary):
      - If `distance <= 0.0`: returns `Err(OpsError::InvalidInput("open mesh volumetric offset requires positive distance"))`.
      - Builds `fh_unsigned_grid` or `bvh_unsigned_grid` (with `narrow_band = distance + 3.0 * h`) over `GridSpec::around_bounds(min, max, h, distance, 3, max_voxels)` and extracts `marching_cubes(&grid, distance)`.
  - Transaction commit:
    - Reserves all removal/addition buffers before the first `tx` mutation so any error or OOM leaves `tx` untouched.
    - Removes all existing live triangles and live vertices, adds `iso.positions` and `iso.triangles`, and if `tx.mesh().uvs().is_some()` calls `tx.drop_channel(UV_CHANNEL)?` and sets `report.uvs_dropped = true`.

### 4.3 WASM, TypeScript (`packages/fast-mesh-wasm`) & Showcase (`examples/showcase`)

- **`crates/fast-mesh-wasm/src/edit.rs` & `packages/fast-mesh-wasm/src/scene.ts`:**
  - `OffsetAlgorithm` TS union: `"normal" | "fhEdt" | "bvhSdf"`.
  - `RepairOp` `"offset"` variant:
    `{ op: "offset"; distance: number; algorithm?: OffsetAlgorithm; voxelSize?: number; maxVoxels?: number }`.
  - Flat op buffer for code `11`: `[distance, algorithm_code (0 normal, 1 fhEdt, 2 bvhSdf, NaN -> 0), voxel_size (NaN -> auto), max_voxels (NaN -> default)]`.
  - Flat report buffer for `offset`: `[moved, max_displacement, degenerate_removed, vertices_removed, vertices_added, triangles_added, triangles_removed, voxel_size, uvs_dropped (0 or 1)]`.
  - `OffsetReport` TS interface gains `verticesAdded: number`, `trianglesAdded: number`, `trianglesRemoved: number`, `voxelSize: number`, `uvsDropped: boolean`.
- **`examples/showcase`:**
  - `RepairPanel.tsx`: adds an **Offset & Shell** block with:
    - Algorithm `<select>` (`normal`, `fhEdt`, `bvhSdf`)
    - Distance `<LengthInput>` (signed; `0` enabled when `fhEdt`/`bvhSdf` is chosen for Voxel Remesh)
    - Optional Voxel size `<LengthInput>` (visible when `fhEdt` or `bvhSdf` is active; empty = automatic `1/64` diagonal)
    - **Offset** button (label switches to **Voxel remesh** when `distance === 0` and algorithm is volumetric)
    - Wall thickness `<LengthInput>` + **Shell** button.
  - `repair-summary.ts`: formats both `normal` (`Offset: moved N vertices ...`) and volumetric `fhEdt`/`bvhSdf` (`Offset: rebuilt M triangles, V vertices (pitch ...)`).

## 5. Engine Invariants & Verification

- Fallible reservations (`try_reserve` / `reserve` / `push` / `filled` / `allocation_allowed`) on all growable allocations; zero `HashMap`/`HashSet` iteration order dependence; `sort_unstable*` only.
- Bit-identical outputs between sequential and `--features parallel` across `edt_3d_sq`, `fh_signed_grid`, `fh_unsigned_grid`, `bvh_signed_grid`, `bvh_unsigned_grid`, and `marching_cubes`.
- `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `pnpm --filter fast-mesh-wasm test`, `pnpm --filter fast-mesh-wasm-example-showcase test`, and `pnpm build` all green.
