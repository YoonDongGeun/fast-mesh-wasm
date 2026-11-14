# Sculpt Domain: Multi-Algorithm Brushes, Surface Filters, Mirror Symmetry & 1M Frame-Time Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modularize brush sculpting into a dedicated `sculpt` domain across Rust, Wasm/TypeScript, and `examples/showcase`, delivering selectable `inflate` / `smooth` / `grab` algorithms (#177), composable `SurfaceFilter`s + `Flatten`/`Crease`/`Clay`/`Twist` brushes + local `X`/`Y`/`Z` mirror `Symmetry` (#192), and `O(|brush|)` stroke/mask/grab paths keeping 1M-triangle sculpting under `33 ms` frame `p95` and `50 ms` max main-thread task (#197).

**Architecture:** Extract brush sculpting from `crates/fast-mesh-ops/src/deform.rs` into `crates/fast-mesh-ops/src/sculpt/` (keeping full `fast_mesh_ops::deform` re-exports) built around shared `SurfaceFilter` and max-envelope `Symmetry` modifiers plus two width-3 `bound_to_topology` stroke channels (`STROKE_POS_CHANNEL` + `STROKE_NORMAL_CHANNEL`). Pair whole-mesh ops with bit-for-bit `*_region` kernels driven by `fast_mesh_scene::Scene::dab_protected_band`, an `O(1)` epoch-stamped lazy `StrokeSnapshot` on `Entry` (interpolated on `split_stretched`), `Scene::vertices_in_sphere`, and `SculptOp` / `scene.sculpt` TypeScript bindings (alongside backwards-compatible `RepairOp` / `scene.repair`).

**Tech Stack:** Rust (`fast-mesh-ops`, `fast-mesh-scene`, `fast-mesh-wasm`), `wasm-bindgen`, TypeScript (`packages/fast-mesh-wasm`, Web Workers, Vitest), React + Three.js (`examples/showcase`), Headless Chrome CDP (`scripts/sculpt-frames.mjs`).

**Spec:** `docs/superpowers/specs/2026-11-13-sculpt-algorithms-filters-symmetry-design.md`

## Global Constraints

- **Fallible Allocations & Rollback Safety:** Every growable temporary in `fast-mesh-ops` and `fast-mesh-scene` uses `crate::reserve`, `crate::push`, `crate::filled`, or `Vec::try_reserve` -> `StoreError::OutOfMemory` with zero partial writes on error.
- **Deterministic Non-Test Engine Code:** Zero `HashMap`/`HashSet` and zero allocating `slice::sort` / `sort_by` (`sort_unstable` / `sort_unstable_by` only) in non-test code across `fast-mesh-ops` and `fast-mesh-scene`.
- **Store Channel Width Invariant:** Do NOT modify `fast-mesh-store`; `ChannelDesc::width` remains in `1..=4` (`STROKE_POS_CHANNEL = "stroke_rest_pos"` width `3`, `STROKE_NORMAL_CHANNEL = "stroke_rest_normal"` width `3`).
- **Bit-for-Bit Region & Parallel Parity:** `Scene::dab` / `Scene::dab_protected_band` must match whole-mesh `sculpt::*` bit-for-bit on triangle-used vertices across `UpdateMode::{Eager, Lazy, Manual}`, and `--features parallel` builds must match sequential builds bit-for-bit.
- **Scene Field-Unit Discipline:** Every new `&mut self` public method on `fast_mesh_scene::Scene` must call `self.drop_field_unit()` first and be exercised in `crates/fast-mesh-scene/src/fields.rs` (`fields::tests::mutating_calls_abandon_open_unit`).
- **UV Channel Preservation:** All sculpt brushes only call `tx.set_positions` and preserve `UV_CHANNEL`; every new brush is covered in `crates/fast-mesh-ops/src/uv_table.rs` and `docs/uv-channel.md`.
- **Disk-Safe Cargo Invocations on Windows:** Always pass `$env:CARGO_PROFILE_DEV_DEBUG="0"; $env:CARGO_PROFILE_TEST_DEBUG="0"; $env:CARGO_INCREMENTAL="0"` before running `cargo` commands.

## Review Focus

1. **Centerline Mirror Symmetry Overlap:** Two or more reflected brushes overlapping on `x = 0` / `y = 0` / `z = 0` must cancel cross-plane displacements to `0.0` AND preserve the exact smooth single-brush falloff profile along tangent/normal directions via max-envelope scaling $\gamma(v) = \max_m a_m(v) / \sum_m a_m(v)$ without creating a flat $100\%$ plateau or $C^0$ kink (tested in Task 1).
2. **`StrokeNormal` & `from_stroke` Without Prior Capture:** Calling `inflate` (`InflateMethod::StrokeNormal`) or `grab` (`from_stroke: true`) on a mesh where `capture_stroke_state` was never called must fall back cleanly to current mesh positions and smooth normals (never panicking, erroring, or reading `NaN`), while calling after `capture_stroke_state` + `split_stretched` uses interpolated parent rest state (tested in Tasks 1, 2, 3).
3. **`Grab` Shortened/Reversed Drag Snap-Back (`from_stroke: true`):** When a `from_stroke` + `adaptive_reach` grab widens on Move 1 and shortens back toward `[0, 0, 0]` on Move 2, vertices moved on Move 1 whose rest positions now lie outside $R_{\text{eff}}$ must snap back bit-for-bit to $P_{\text{rest}}[v]$ in both whole-mesh `grab` and `Scene::dab` (tested in Tasks 2 & 3).
4. **`GrabMethod::Geodesic` Air-Gap Isolation vs UV Seam Continuity:** Two parallel disconnected sheets separated by `0.05 * radius` must move ONLY the seed-connected sheet while a single sheet split along a UV seam (duplicate vertex IDs sharing identical `[u32; 3]` position bits) moves seamlessly across the cut without cracking (tested in Task 2).
5. **1M Mask Liveness & Soft-Band Under Live Splits (#197):** After `splitStretched` splits edges during a stroke on a 1M-triangle mesh, masked dabs must hold masked vertices + inherited split midpoints byte-identical without scanning 3M triangle indices on the main thread, while topology removals (`weld` / `simplify`) still mark a broken mask stale (tested in Tasks 3 & 5).

---

### Task 1: `fast-mesh-ops::sculpt` Core Domain, Stroke Channels, Filters, Symmetry & `Inflate` / `Smooth` (#177 + #192)

**Files:**
- Create: `crates/fast-mesh-ops/src/sculpt/mod.rs`
- Create: `crates/fast-mesh-ops/src/sculpt/stroke.rs`
- Create: `crates/fast-mesh-ops/src/sculpt/filter.rs`
- Create: `crates/fast-mesh-ops/src/sculpt/inflate.rs`
- Create: `crates/fast-mesh-ops/src/sculpt/smooth.rs`
- Modify: `crates/fast-mesh-ops/src/deform.rs`
- Modify: `crates/fast-mesh-ops/src/lib.rs`

**Interfaces:**
- Consumes: `fast_mesh::compute_vertex_normals`, `fast_mesh_store::{ChannelDesc, Incidence, Mesh, StoreError, Transaction, VertexId}`, `crate::topology::VertexFans`, `crate::{OpsError, Result, filled, push, reserve}`.
- Produces (`fast_mesh_ops::sculpt`, re-exported from `fast_mesh_ops::deform` and `fast_mesh_ops`):
  - `pub const STROKE_POS_CHANNEL: &str = "stroke_rest_pos";`
  - `pub const STROKE_NORMAL_CHANNEL: &str = "stroke_rest_normal";`
  - `pub struct StrokeSlice<'a> { pub positions: &'a [[f32; 3]], pub normals: &'a [[f32; 3]] }`
  - `pub fn capture_stroke_state(mesh: &mut Mesh) -> Result<()>`
  - `pub fn ensure_stroke_state(mesh: &mut Mesh) -> Result<()>`
  - `pub enum SurfaceFilter { None, PeaksOnly { threshold: f32, contrast: f32 }, ValleysOnly { threshold: f32, contrast: f32 }, NormalAngle { normal: Option<[f32; 3]>, max_angle_deg: f32 } }`
  - `pub struct Symmetry { pub x: bool, pub y: bool, pub z: bool }` (`Symmetry::NONE`, `Symmetry::active_masks(&self) -> ([u8; 8], usize)`)
  - `pub enum InflateMethod { VertexNormal, AverageNormal, StrokeNormal }` (default `VertexNormal`)
  - `pub struct InflateOptions { pub brush: Brush, pub method: InflateMethod, pub filter: SurfaceFilter, pub symmetry: Symmetry }` + `InflateOptionsBuilder` + `InflateReport`
  - `pub fn inflate(tx: &mut Transaction, options: &InflateOptions) -> Result<InflateReport>`
  - `pub fn inflate_region(tx: &mut Transaction, options: &InflateOptions, region: &[VertexId]) -> Result<InflateReport>`
  - `pub fn inflate_region_with_stroke(tx: &mut Transaction, options: &InflateOptions, region: &[VertexId], stroke: Option<StrokeSlice<'_>>) -> Result<InflateReport>`
  - `pub enum SmoothMethod { Taubin, Laplacian, HcLaplacian, Cotangent }` (default `Taubin`)
  - `pub struct SmoothOptions { pub brush: Brush, pub cycles: u32, pub method: SmoothMethod, pub filter: SurfaceFilter, pub symmetry: Symmetry }` + `SmoothOptionsBuilder` + `SmoothReport`
  - `pub fn smooth(tx: &mut Transaction, options: &SmoothOptions) -> Result<SmoothReport>`
  - `pub fn smooth_region(tx: &mut Transaction, options: &SmoothOptions, region: &[VertexId]) -> Result<SmoothReport>`

- [ ] **Step 1: Write failing tests for stroke channels, `SurfaceFilter`, max-envelope `Symmetry`, `InflateMethod`, and `SmoothMethod`**

Add unit tests in `crates/fast-mesh-ops/src/sculpt/` covering:
1. `capture_stroke_state` creates two width-3 `bound_to_topology` channels (`STROKE_POS_CHANNEL` & `STROKE_NORMAL_CHANNEL`), interpolates parent values across `split_stretched`, and `InflateMethod::StrokeNormal` moves vertices along frozen normals across 5 repeated dabs (linearly equal to `5 * single_dab` rather than curving along updated normals), while falling back to current smooth normals if `capture_stroke_state` was never called.
2. `InflateMethod::AverageNormal` translates all affected vertices along the single falloff-weighted unit average normal direction rather than diverging radially on a curved ridge.
3. `SmoothMethod` independent expected values:
   - `HcLaplacian` (`HC_ALPHA = 0.1`, `HC_BETA = 0.5`) shrinks a closed sphere's radius strictly less than `Laplacian` over 6 cycles while reducing high-frequency vertex noise.
   - `Cotangent` (edge cotangent weights clamped to `[1e-4, 1e3]`) keeps an irregularly tessellated flat interior vertex stationary in `(x, y)` within `1e-5` (zero tangential drift), whereas uniform `Laplacian` drifts toward the denser side.
4. `SurfaceFilter`:
   - `PeaksOnly { threshold: 0.05, contrast: 0.8 }` with `Smooth` lowers a raised bump while leaving an inverted pit untouched (`0.0` displacement); `ValleysOnly` fills the pit while leaving the peak untouched.
   - `NormalAngle { normal: Some([0.0, 0.0, 1.0]), max_angle_deg: 35.0 }` affects top-face vertices of a cube while leaving perpendicular side-face vertices (`90°`) untouched (`0.0` displacement).
5. `Symmetry`:
   - Disjoint mirror spheres (`x: true`) move `+x` and `-x` hemispheres identically (`p'[-x] = [-p'[+x].x, p'[+x].y, p'[+x].z]`).
   - Centerline overlap (`brush.center = [0.05 * R, 0.0, 0.0]`, `Symmetry { x: true, y: false, z: false }`): vertices on `x = 0` keep `dx == 0.0` and match the smooth single-brush normal displacement profile `strength * falloff(t)` (verified against `1.0` plateau to rule out `min(1, 1/sum)` kinks).
6. Region parity (`inflate_region == inflate`, `smooth_region == smooth` bit-for-bit across all methods, filters, and `x/y/z` symmetry combinations) and `fail_allocations_after` OOM rollback sweeps.

- [ ] **Step 2: Run cargo test to verify RED state**

Run: `$env:CARGO_PROFILE_DEV_DEBUG="0"; $env:CARGO_PROFILE_TEST_DEBUG="0"; $env:CARGO_INCREMENTAL="0"; cargo test -p fast-mesh-ops sculpt`
Expected: FAIL to compile / missing module `sculpt` and new types.

- [ ] **Step 3: Implement `sculpt/{mod,stroke,filter,inflate,smooth}.rs` and wire `deform.rs` & `lib.rs`**

- `stroke.rs`: Implement `STROKE_POS_CHANNEL` & `STROKE_NORMAL_CHANNEL` via `ChannelDesc::source(..., 3).bound_to_topology()`, `capture_stroke_state` (computes `compute_vertex_normals` in `f32`, writes positions + normals via `add_channel_with_values` after `remove_channel` if present), `ensure_stroke_state`, and `StrokeSlice<'a>` helper reading either explicit slice or mesh channels with fallback to current mesh positions/normals when absent or `NaN`.
- `filter.rs`: Implement `SurfaceFilter` validation (`threshold`, `contrast` in `[0.0, 1.0]`, `max_angle_deg` in `(0.0, 180.0]`, `normal` finite & non-zero when `Some`) and evaluation over CSR/fan 1-rings (`PeaksOnly`/`ValleysOnly` scale-invariant prominence $h_v = ((p_v - \bar{p}_v) \cdot \hat{n}_v) / \ell_v$ with smoothstep width $\max(10^{-5}, 0.25 (1 - \text{contrast}))$; `NormalAngle` inner `0.75 * max_angle` full weight + smoothstep taper to `0` at `max_angle`), plus `Symmetry` (`active_masks` returning `bits` in `0..8` and max-envelope combiner $\gamma(v) = \max_m a_m(v) / \sum_m a_m(v)$).
- `inflate.rs` & `smooth.rs`: Implement whole-mesh and `*_region` kernels with bit-for-bit parity (note: for `Symmetry` when `region` is supplied, `region` already covers the union of reflected spheres!). Keep existing `InflateOptions { brush: .., ..Default::default() }` and `SmoothOptions { brush: .., cycles: .., method: .., ..Default::default() }` struct-literal call sites compiling by updating any exhaustive struct literals inside `crates/`.
- Re-export `sculpt` items from `crates/fast-mesh-ops/src/deform.rs` (keeping existing `grab` in `deform.rs` until Task 2 moves it) and export `pub mod sculpt;` in `crates/fast-mesh-ops/src/lib.rs`.

- [ ] **Step 4: Run `fast-mesh-ops` tests (sequential and `--features parallel`) to verify GREEN**

Run: `$env:CARGO_PROFILE_DEV_DEBUG="0"; $env:CARGO_PROFILE_TEST_DEBUG="0"; $env:CARGO_INCREMENTAL="0"; cargo test -p fast-mesh-ops --features parallel`
Expected: PASS with zero warnings.

- [ ] **Step 5: Commit Task 1**

```powershell
git add crates/fast-mesh-ops/src/lib.rs crates/fast-mesh-ops/src/deform.rs crates/fast-mesh-ops/src/sculpt
git commit -m "feat(ops): add sculpt domain module, stroke channels, filters, symmetry and #177 inflate/smooth methods"
```

---

### Task 2: `fast-mesh-ops::sculpt` Multi-Algorithm `Grab` (#177), Extra Brushes (`Flatten`, `Crease`, `Clay`, `Twist` #192), Soft-Band `protect` & UV Table

**Files:**
- Create: `crates/fast-mesh-ops/src/sculpt/grab.rs`
- Create: `crates/fast-mesh-ops/src/sculpt/plane.rs`
- Create: `crates/fast-mesh-ops/src/sculpt/crease.rs`
- Create: `crates/fast-mesh-ops/src/sculpt/twist.rs`
- Modify: `crates/fast-mesh-ops/src/sculpt/mod.rs`
- Modify: `crates/fast-mesh-ops/src/deform.rs`
- Modify: `crates/fast-mesh-ops/src/protect.rs`
- Modify: `crates/fast-mesh-ops/src/uv_table.rs`
- Modify: `crates/fast-mesh-ops/benches/ops.rs`
- Modify: `docs/uv-channel.md`

**Interfaces:**
- Consumes: Task 1 `sculpt::{Brush, Falloff, StrokeSlice, SurfaceFilter, Symmetry}`.
- Produces (`fast_mesh_ops::sculpt`, re-exported from `fast_mesh_ops::deform` and `fast_mesh_ops`):
  - `pub const GRAB_MAX_STRETCH: f32 = 0.8;`
  - `pub fn grab_widest_reach(brush: &Brush, direction: [f32; 3], method: GrabMethod) -> f32` (uses slope `1.0` for `Linear`, `1.5` for `Smooth`, `1.75` for `Kelvinlet`: `(|brush.strength| * ||direction|| * slope / GRAB_MAX_STRETCH).max(brush.radius)`)
  - `pub enum GrabMethod { Sphere, Geodesic, Kelvinlet }` (default `Sphere`)
  - `pub struct GrabOptions { pub brush: Brush, pub direction: [f32; 3], pub method: GrabMethod, pub poisson: f32, pub from_stroke: bool, pub adaptive_reach: bool, pub filter: SurfaceFilter, pub symmetry: Symmetry }` (`Default`: `poisson = 0.4`, `from_stroke = false`, `adaptive_reach = false`) + `GrabOptionsBuilder` + `GrabReport`
  - `pub fn grab(tx: &mut Transaction, options: &GrabOptions) -> Result<GrabReport>`
  - `pub fn grab_region(tx: &mut Transaction, options: &GrabOptions, region: &[VertexId]) -> Result<GrabReport>`
  - `pub fn grab_region_with_stroke(tx: &mut Transaction, options: &GrabOptions, region: &[VertexId], stroke: Option<StrokeSlice<'_>>, zero_protected: &[VertexId]) -> Result<GrabReport>`
  - `pub enum FlattenMode { Both, Peaks, Valleys }` (default `Both`)
  - `pub struct FlattenOptions { pub brush: Brush, pub mode: FlattenMode, pub plane_origin: Option<[f32; 3]>, pub plane_normal: Option<[f32; 3]>, pub filter: SurfaceFilter, pub symmetry: Symmetry }` + `FlattenOptionsBuilder` + `FlattenReport` + `flatten` + `flatten_region`
  - `pub struct CreaseOptions { pub brush: Brush, pub pinch: f32, pub filter: SurfaceFilter, pub symmetry: Symmetry }` (`pinch` in `[0.0, 1.0]`, default `0.5`) + `CreaseOptionsBuilder` + `CreaseReport` + `crease` + `crease_region`
  - `pub enum ClayProfile { Round, Strips }` (default `Round`)
  - `pub struct ClayOptions { pub brush: Brush, pub profile: ClayProfile, pub tangent: Option<[f32; 3]>, pub filter: SurfaceFilter, pub symmetry: Symmetry }` + `ClayOptionsBuilder` + `ClayReport` + `clay` + `clay_region`
  - `pub enum TwistMethod { Vortex, Kelvinlet }` (default `Vortex`)
  - `pub struct TwistOptions { pub brush: Brush, pub axis: Option<[f32; 3]>, pub method: TwistMethod, pub filter: SurfaceFilter, pub symmetry: Symmetry }` + `TwistOptionsBuilder` + `TwistReport` + `twist` + `twist_region`
  - In `protect.rs`: `pub fn with_protection_band<R, E: From<OpsError>>(tx: &mut Transaction, region: Option<&[VertexId]>, vertices: &[VertexId], weights: &[f32], band: f32, f: impl FnOnce(&mut Transaction) -> std::result::Result<R, E>) -> std::result::Result<R, E>`

- [ ] **Step 1: Write failing tests for `GrabMethod` (`Sphere`/`Geodesic`/`Kelvinlet`), `adaptive_reach` + snap-back, `Flatten`/`Crease`/`Clay`/`Twist`, `with_protection_band`, and `uv_table`**

Add unit tests in `crates/fast-mesh-ops/src/sculpt/` and `protect.rs` / `uv_table.rs` covering:
1. `GrabMethod::Geodesic`:
   - Two parallel separate sheets `z = 0.0` and `z = 0.05` (`radius = 1.0`, `center = [0.0, 0.0, 0.0]`): `Sphere` moves both sheets, while `Geodesic` moves ONLY the `z = 0.0` sheet (`z = 0.05` stays byte-identical!).
   - Single sheet split down `x = 0` by a UV seam (duplicate vertex IDs with identical `[f32; 3]` coordinates): `Geodesic` propagates zero-cost across coincident seam positions so both halves stay watertight with `0.0` gap.
2. `GrabMethod::Kelvinlet`:
   - Calibrated center displacement at `r = 0` equals `brush.strength * direction` within `1e-6`, decays smoothly to `0.0` at `r >= R_eff`, and `poisson = 0.45` vs `poisson = 0.0` produces distinct coupled lateral displacement off-axis (`y != 0` when pulling along `+x`). Invalid `poisson < 0.0` or `> 0.5` returns `OpsError::InvalidInput`.
3. `adaptive_reach` & `from_stroke`:
   - Tangential drag along a flat/curved surface (`L = 1.2`, `R = 0.2`) widens $R_{\text{eff}}$ via `ahead = max((along / r) * min(1, 2 * (1 - r / R_wide)))` and produces zero flipped triangle normals, whereas normal pull straight off a flat plane (`+z`, where all surface offsets have `along == 0`) keeps $R_{\text{eff}} == R$ (narrow horn!).
   - Snap-back: after `capture_stroke_state`, Move 1 (`direction = [1.0, 0.0, 0.0]`, `from_stroke: true, adaptive_reach: true`) moves wide vertices; Move 2 (`direction = [0.05, 0.0, 0.0]`, `from_stroke: true, adaptive_reach: true`) snaps outer vertices cleanly back to `rest_pos` bit-for-bit.
4. Extra brushes (#192):
   - `Flatten`: `Both` brings peaks down and valleys up toward the brush plane; `Peaks` (scrape) only moves `signed_dist > 0`; `Valleys` (fill) only moves `signed_dist < 0`.
   - `Crease`: carves normal groove (`strength < 0`) while pulling vertices radially toward brush center in the tangent plane (`pinch > 0`); `strength == 0.0, pinch > 0.0` acts as pure Pinch.
   - `Clay`: `Round` vs `Strips` (`tangent: Some([1.0, 0.0, 0.0])`) lifts vertices to the offset brush plane (`h = 0.25 * R * sign(strength)`), with `Strips` forming a square `2R x 2R` footprint reaching corners at $(\pm 0.8 R, \pm 0.8 R)$ (`> R` Euclidean!).
   - `Twist`: `Vortex` rotates vertices in the plane perpendicular to `axis` by `strength * f` radians preserving axial height & radius; `TwistMethod::Kelvinlet` applies calibrated twisting Kelvinlet; under `Symmetry { x: true, .. }`, axial reflection multiplies `strength` by `det(M) = -1` so mirrored vortices counter-rotate symmetrically!
5. `with_protection_band`:
   - Masked vertices (`weight == 0.0`) remain byte-identical (`0.0` move), vertices within edge-Dijkstra distance `< band` scale smoothly by `s * s * (3 - 2 * s)` (`s = dist / band`), and vertices beyond `band` move 100%, matching between `region = Some(...)` and `region = None`.
6. Region parity (`*_region == *` bit-for-bit for all 5 brushes), OOM sweeps (`fail_allocations_after`), and `uv_table` entries preserving UVs.

- [ ] **Step 2: Run cargo test to verify RED state**

Run: `$env:CARGO_PROFILE_DEV_DEBUG="0"; $env:CARGO_PROFILE_TEST_DEBUG="0"; $env:CARGO_INCREMENTAL="0"; cargo test -p fast-mesh-ops`
Expected: FAIL to compile missing `grab.rs`, `plane.rs`, `crease.rs`, `twist.rs`, and `with_protection_band`.

- [ ] **Step 3: Implement `grab.rs`, `plane.rs`, `crease.rs`, `twist.rs`, thin `deform.rs`, `protect.rs` soft-band, and `uv_table.rs` / `docs/uv-channel.md`**

- `grab.rs`:
  - Follow Spec §5.3.C verbatim: `FALLOFF_SLOPE` (`1.0` `Linear`, `1.5` `Smooth`, `1.75` `Kelvinlet`), smooth `ahead` gating over live candidate rest positions (skipping sorted `zero_protected` vertices so masked regions match `GrabRegion`!), `Geodesic` seed triangle rule (strictly among live non-degenerate triangles with at least one corner strictly inside `Sphere(c, R_eff)`, ties broken by ascending `TriangleId`, zero-cost seam linking by sorted `([u32; 3] bits, VertexId)` slice — NO `HashMap`!), calibrated regularized `Kelvinlet` ($\epsilon = 0.35 R_{\text{eff}}$, $(1 - t^2)^2$ window), and `from_stroke` snap-back skipping `same_bits(&curr, &target)` writes.
- `plane.rs`, `crease.rs`, `twist.rs`: Implement whole-mesh and `*_region` functions sharing Task 1's `SurfaceFilter` and max-envelope `Symmetry` combiner (with axial `det(M)` sign flip on `Twist`).
- `deform.rs`: Replace migrated contents with clean `pub use crate::sculpt::*;` re-exports so every existing `fast_mesh_ops::deform::*` import compiles unchanged.
- `protect.rs`: Implement `with_protection_band` using deterministic `BinaryHeap` Dijkstra over `mesh.incidence()` (or local triangle scan when `incidence()` is `None`), restricted to vertices within `band` along mesh edges of `weight == 0.0` seeds near `region`, with zero `HashMap`.
- Update `crates/fast-mesh-ops/src/uv_table.rs`, `crates/fast-mesh-ops/benches/ops.rs`, and `docs/uv-channel.md`.

- [ ] **Step 4: Run full `fast-mesh-ops` test suite to verify GREEN**

Run: `$env:CARGO_PROFILE_DEV_DEBUG="0"; $env:CARGO_PROFILE_TEST_DEBUG="0"; $env:CARGO_INCREMENTAL="0"; cargo test -p fast-mesh-ops --features parallel`
Expected: PASS with zero warnings.

- [ ] **Step 5: Commit Task 2**

```powershell
git add crates/fast-mesh-ops crates/fast-mesh/ docs/uv-channel.md
git commit -m "feat(ops): add #177 grab methods, #192 flatten/crease/clay/twist brushes and localized soft-band protection"
```

---

### Task 3: `fast-mesh-scene` BVH Dabs, Lazy `O(1)` Stroke Snapshot, `vertices_in_sphere`, Soft-Band & 100k/1M Benchmarks (#177 + #192 + #197)

**Files:**
- Modify: `crates/fast-mesh-scene/src/sculpt.rs`
- Modify: `crates/fast-mesh-scene/src/scene.rs`
- Modify: `crates/fast-mesh-scene/src/fields.rs`
- Modify: `crates/fast-mesh-scene/benches/sculpt.rs`
- Create: `docs/sculpt-benchmarks.md`

**Interfaces:**
- Consumes: Task 1 & 2 `fast_mesh_ops::sculpt::*` and `fast_mesh_ops::protect::with_protection_band`.
- Produces (`fast_mesh_scene`):
  - `pub enum Dab { Inflate(InflateOptions), Smooth(SmoothOptions), Grab(GrabOptions), Flatten(FlattenOptions), Crease(CreaseOptions), Clay(ClayOptions), Twist(TwistOptions) }`
  - `Scene::capture_stroke_state(&mut self, id: MeshId, only_if_missing: bool) -> Result<()>` (`O(1)` epoch bump on `Entry`; invalidates cleanly on non-`split_stretched` topology edits, `undo`, `redo`, `cancel_step`, `compact`)
  - `Scene::dab_protected_band(&mut self, id: MeshId, dab: &Dab, vertices: &[VertexId], weights: &[f32], band: f32) -> Result<DabReport>` (`dab_protected` delegates with `band = 0.0`)
  - `Scene::vertices_in_sphere(&mut self, id: MeshId, center: [f32; 3], radius: f32) -> Result<Vec<VertexId>>`

- [ ] **Step 1: Write failing parity, lazy stroke-snapshot, `vertices_in_sphere`, and `fields` abandon tests**

Add tests in `crates/fast-mesh-scene/src/sculpt.rs` and `crates/fast-mesh-scene/src/fields.rs`:
1. `Scene::dab` bit-for-bit parity against `Scene::edit` whole-mesh `sculpt::*` across `UpdateMode::{Eager, Lazy, Manual}` for every new `Dab` variant (`Inflate` × 3 methods, `Smooth` × 4 methods, `Grab` × 3 methods + `adaptive_reach`, `Flatten` × 3 modes, `Crease`, `Clay` × `Round`/`Strips`, `Twist` × 2 methods), `SurfaceFilter` × 4, and `Symmetry` (`x`, `xy`, `xyz`).
2. Multi-step `capture_stroke_state` -> `dab(Grab { from_stroke: true, adaptive_reach: true })` -> `split_stretched` -> `dab(Grab { shortened })` -> `undo` / `redo` parity against a standalone `Mesh` running whole-mesh `capture_stroke_state` + `grab` + `split_stretched` + `grab`!
3. `Scene::dab_protected_band` with `band > 0.0` matches whole-mesh `with_protection_band` bit-for-bit and skips creating an undo step when `moved == 0`.
4. `Scene::vertices_in_sphere` returns ascending live triangle-used `VertexId`s inside `(center, radius)`, rejects non-finite/negative radius with `OpsError::InvalidInput`, respects `Manual` stale BVH (`SceneError::StaleBvh`), and add `capture_stroke_state`, `dab_protected_band`, and `vertices_in_sphere` to `fields::tests::mutating_calls_abandon_open_unit`.

- [ ] **Step 2: Run `fast-mesh-scene` tests to verify RED state**

Run: `$env:CARGO_PROFILE_DEV_DEBUG="0"; $env:CARGO_PROFILE_TEST_DEBUG="0"; $env:CARGO_INCREMENTAL="0"; cargo test -p fast-mesh-scene`
Expected: FAIL to compile missing `Dab` variants and `Scene` methods.

- [ ] **Step 3: Implement `Entry` lazy `StrokeSnapshot`, `Dab` BVH region gathering, `dab_protected_band`, `vertices_in_sphere`, and 100k/1M benchmark runner**

- On `Entry` (`scene.rs`): maintain a lazy `StrokeSnapshot`:
  - `epoch: u32` (`0` = no captured stroke; `capture_stroke_state(id, only_if_missing)` increments `next_epoch` in `O(1)` unless `only_if_missing && epoch != 0`, clearing `touched`).
  - `slot_epoch: Vec<u32>`, `rest_pos: Vec<[f32; 3]>`, `rest_normal: Vec<[f32; 3]>`, `touched: Vec<VertexId>`.
  - Before a dab executes on `region` when `epoch != 0`, lazily populate `rest_pos[v]` and `rest_normal[v]` (computed locally from `entry.mesh.incidence()` over `v`'s triangle fan — exact same `compute_vertex_normals` area-weighted cross-product sum normalized to unit length!) for every `v` in `region` AND its 1-ring neighbors whose `slot_epoch[v] != epoch`! That keeps pointer-down `O(1)` and per-dab capture `O(|brush|)` while guaranteeing exact parity with whole-mesh `capture_stroke_state` AND guaranteeing both parent endpoints `(a, b)` of any stretched edge already have captured rest state!
  - In `Scene::split_stretched`: when `entry.stroke.epoch != 0`, interpolate `rest_pos[new] = 0.5 * (rest_pos[a] + rest_pos[b])` and `rest_normal[new] = normalize(0.5 * (rest_normal[a] + rest_normal[b]))` for every `[new, a, b]` in `StretchReport::vertices` in `O(splits)`, and push `new` into `touched` if either `a` or `b` was touched.
  - On any other topology change (`patch.changes().topology_changed()` outside `split_stretched`), `undo`, `redo`, `cancel_step`, or `compact`, reset `stroke.epoch = 0` and clear `touched`.
- In `sculpt.rs`:
  - Compute query radius per brush (`brush.radius * 3f32.sqrt()` for `ClayProfile::Strips`, `grab_widest_reach` for `Grab { adaptive_reach: true }`, `brush.radius` otherwise), union BVH `vertices_in` spheres across all `symmetry.active_masks()` reflected centers (extending radius by `+ band` when `band > 0.0` to collect local protection seeds, while passing the brush's own candidate `region` to `*_region`), and union `stroke.touched` when `Grab { from_stroke: true }`.
  - Pass `StrokeSlice` into `inflate_region_with_stroke` / `grab_region_with_stroke` when `stroke.epoch != 0` (and fall back to whole-mesh `capture_stroke_state` if `Manual` stale BVH forces the non-region path).
- Update `crates/fast-mesh-scene/benches/sculpt.rs` (or add a release benchmark test/example) to measure single-dab median latency at **100k** and **1M** triangles across every `InflateMethod`, `SmoothMethod`, `GrabMethod`, extra brush, filter, and symmetry mode, and write the measured numbers into `docs/sculpt-benchmarks.md`.

- [ ] **Step 4: Run `fast-mesh-scene` tests & 100k/1M benchmark measurement**

Run: `$env:CARGO_PROFILE_DEV_DEBUG="0"; $env:CARGO_PROFILE_TEST_DEBUG="0"; $env:CARGO_INCREMENTAL="0"; cargo test -p fast-mesh-scene --features parallel`
Expected: PASS with zero warnings, and `docs/sculpt-benchmarks.md` populated with real 100k & 1M timings.

- [ ] **Step 5: Commit Task 3**

```powershell
git add crates/fast-mesh-scene docs/sculpt-benchmarks.md
git commit -m "feat(scene): add lazy O(1) stroke snapshot, multi-brush BVH dabs, vertices_in_sphere and 100k/1M benchmarks"
```

---

### Task 4: `fast-mesh-wasm` FFI & `packages/fast-mesh-wasm` TypeScript Sculpt Domain API (`Scene`, `WorkerScene`, `SceneStore`)

**Files:**
- Modify: `crates/fast-mesh-wasm/src/edit.rs`
- Modify: `crates/fast-mesh-wasm/src/scene.rs`
- Modify: `packages/fast-mesh-wasm/src/scene.ts`
- Modify: `packages/fast-mesh-wasm/src/web.ts`
- Modify: `packages/fast-mesh-wasm/src/node.ts`
- Modify: `packages/fast-mesh-wasm/src/worker/protocol.ts`
- Modify: `packages/fast-mesh-wasm/src/worker/engine.ts`
- Modify: `packages/fast-mesh-wasm/src/worker/client.ts`
- Modify: `packages/fast-mesh-wasm/src/store/host.ts`
- Modify: `packages/fast-mesh-wasm/src/store/store.ts`
- Modify: `packages/fast-mesh-wasm/src/store/index.ts`
- Modify: `packages/fast-mesh-wasm/test/scene.test.ts` (and/or add `packages/fast-mesh-wasm/test/sculpt.test.ts`)
- Create: `.changeset/sculpt-domain-brushes-filters-symmetry.md`

**Interfaces:**
- Consumes: Task 3 `fast_mesh_scene::{Dab, Scene}` and `fast_mesh_ops::sculpt::*`.
- Produces (`packages/fast-mesh-wasm`):
  - Types exported from `fast-mesh-wasm` (`web.ts`, `node.ts`, `scene.ts`) and `fast-mesh-wasm/store` where applicable:
    - `InflateMethod = "vertexNormal" | "averageNormal" | "strokeNormal"`
    - `SmoothMethod = "taubin" | "laplacian" | "hcLaplacian" | "cotangent"`
    - `GrabMethod = "sphere" | "geodesic" | "kelvinlet"`
    - `FlattenMode = "both" | "peaks" | "valleys"`
    - `ClayProfile = "round" | "strips"`
    - `TwistMethod = "vortex" | "kelvinlet"`
    - `SurfaceFilter = { kind: "none" } | { kind: "peaksOnly"; threshold?: number; contrast?: number } | { kind: "valleysOnly"; threshold?: number; contrast?: number } | { kind: "normalAngle"; normal?: Vec3; maxAngleDeg?: number }`
    - `Symmetry = { x?: boolean; y?: boolean; z?: boolean }`
    - `VertexProtection = { vertices: Indices; weights: ArrayLike<number>; band?: number }`
    - `FlattenReport`, `CreaseReport`, `ClayReport`, `TwistReport` (`{ moved: number; maxDisplacement: number }`)
    - `SculptOp` union (`inflate | smooth | relax | grab | flatten | crease | clay | twist | captureStrokeState | splitStretched | captureRestSizes | remesh`, all included in `RepairOp` for 100% backwards compatibility) + `SculptReport`
  - Methods:
    - `Scene.sculpt(mesh, op)` (`alias` delegating to `repair` with `SculptOp` narrowing), `WorkerScene.sculpt(mesh, op)`, `SceneStore.sculpt(mesh, op)`
    - `Scene.verticesInSphere(mesh: MeshHandle, center: Vec3, radius: number): Uint32Array`
    - `WorkerScene.verticesInSphere(mesh: MeshHandle, center: Vec3, radius: number): Promise<Uint32Array>`
    - `SceneStore.verticesInSphere(mesh: MeshHandle, center: Vec3, radius: number): Promise<Uint32Array>`

- [ ] **Step 1: Write failing Vitest tests in `packages/fast-mesh-wasm/test/sculpt.test.ts` (and Rust `edit.rs` codec tests)**

Test on BOTH sync `Scene` and `WorkerScene` (plus `SceneStore`):
1. `inflate` (`vertexNormal`, `averageNormal`, `strokeNormal` with `captureStrokeState`), `smooth` (`taubin`, `laplacian`, `hcLaplacian`, `cotangent`), and `grab` (`sphere`, `geodesic` on two separated sheets, `kelvinlet` with `poisson`, `fromStroke` + `adaptiveReach`).
2. `SurfaceFilter` (`peaksOnly`, `valleysOnly`, `normalAngle`), extra brushes (`flatten`, `crease`, `clay`, `twist`), and `symmetry: { x: true, y: true, z: false }` executed via both `scene.sculpt(...)` and `scene.repair(...)`.
3. `protect` with `band > 0` holding masked vertices at 0 displacement and feathering neighbors smoothly, plus `verticesInSphere` returning identical `Uint32Array` on `Scene`, `WorkerScene`, and `SceneStore`.
4. Invalid option inputs (`poisson: 0.9`, `maxAngleDeg: 0`, `captureStrokeState` mixed in a multi-op list) throw `FastMeshError` (`INVALID_ARGUMENT`) with zero mesh mutation.

- [ ] **Step 2: Run Rust `fast-mesh-wasm` tests to verify RED**

Run: `$env:CARGO_PROFILE_DEV_DEBUG="0"; $env:CARGO_PROFILE_TEST_DEBUG="0"; $env:CARGO_INCREMENTAL="0"; cargo test -p fast-mesh-wasm`
Expected: FAIL before `edit.rs` / `scene.rs` updates.

- [ ] **Step 3: Implement `edit.rs`, `WasmScene`, TS `scene.ts`, Worker protocol/client/engine, `SceneStore`, and `.changeset`**

- Exact `edit.rs` wire layout (all `f64`, matching Spec §5.5):
  - `5` `inflate`: `cx, cy, cz, radius, falloff, strength, method (NaN/0 vertexNormal, 1 averageNormal, 2 strokeNormal)`
  - `6` `smooth`: `cx, cy, cz, radius, falloff, strength, cycles, method (NaN/0 taubin, 1 laplacian, 2 hcLaplacian, 3 cotangent)` (`10` `relax` keeps its 7-arg wire layout for legacy callers, mapping to `SmoothMethod::Laplacian`)
  - `9` `grab`: `cx, cy, cz, radius, falloff, strength, dx, dy, dz, method (NaN/0 sphere, 1 geodesic, 2 kelvinlet), poisson (NaN -> 0.4), from_stroke (0/1), adaptive_reach (0/1)`
  - `18` `captureStrokeState`: `only_if_missing (0/1)` (must be sole op in list, runs `self.scene.capture_stroke_state(id, only_if_missing)`, returns empty `Vec`)
  - `19` `flatten`: `cx, cy, cz, radius, falloff, strength, mode (0 both, 1 peaks, 2 valleys), has_origin (0/1), ox, oy, oz, has_normal (0/1), nx, ny, nz` -> report `[moved, max_displacement]`
  - `20` `crease`: `cx, cy, cz, radius, falloff, strength, pinch (NaN -> 0.5)` -> report `[moved, max_displacement]`
  - `21` `clay`: `cx, cy, cz, radius, falloff, strength, profile (0 round, 1 strips), has_tangent (0/1), tx, ty, tz` -> report `[moved, max_displacement]`
  - `22` `twist`: `cx, cy, cz, radius, falloff, strength, method (0 vortex, 1 kelvinlet), has_axis (0/1), ax, ay, az` -> report `[moved, max_displacement]`
  - `32` `protect`: `n, band (0.0 when unset), id × n, weight × n`, wrapping any brush op (`5, 6, 9, 10, 19, 20, 21, 22`)
  - `33` `brushModifiers`: `sym_mask (0..7 bitmask x=1,y=2,z=4), filter_kind (0 none, 1 peaksOnly, 2 valleysOnly, 3 normalAngle), p0, p1, has_vec (0/1), vx, vy, vz`, wrapping any brush op (may compose in either order with `32` `protect`!).
- Expose `WasmScene::vertices_in_sphere(&mut self, index: u32, generation: u32, cx: f64, cy: f64, cz: f64, radius: f64) -> Result<Vec<u32>, Error>` (`cx, cy, cz, radius` in local mesh coordinates, matching `inflate`/`grab`/`splitStretched`), wire `verticesInSphere` query in `worker/protocol.ts`, `worker/engine.ts`, `worker/client.ts`, `store/host.ts`, and `store/store.ts`, and wire `sculpt` domain methods + exports in `web.ts`, `node.ts`, and `store/index.ts`.
- Create `.changeset/sculpt-domain-brushes-filters-symmetry.md` (`"fast-mesh-wasm": minor`).

- [ ] **Step 4: Build Wasm + TS package and run Vitest + Rust tests to verify GREEN**

Run: `$env:CARGO_PROFILE_DEV_DEBUG="0"; $env:CARGO_PROFILE_TEST_DEBUG="0"; $env:CARGO_INCREMENTAL="0"; cargo test -p fast-mesh-wasm; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; pnpm build; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; pnpm --filter fast-mesh-wasm test`
Expected: PASS across all Rust and Vitest tests (Sync + Worker + Store).

- [ ] **Step 5: Commit Task 4**

```powershell
git add crates/fast-mesh-wasm packages/fast-mesh-wasm .changeset/sculpt-domain-brushes-filters-symmetry.md
git commit -m "feat(wasm): expose sculpt domain ops, brush modifiers, soft-band protect and verticesInSphere"
```

---

### Task 5: `examples/showcase` Sculpt Toolbox, Settings Persistence, Engine-Driven Grab/Mask & #197 1M Frame-Time Verification

**Files:**
- Modify: `examples/showcase/src/panels/settings.ts`
- Modify: `examples/showcase/src/scene/app-state.ts`
- Modify: `examples/showcase/src/viewport/sculpt.ts`
- Modify: `examples/showcase/src/viewport/mask-protect.ts`
- Modify: `examples/showcase/src/viewport/Viewport.tsx`
- Modify: `examples/showcase/src/viewport/Toolbox.tsx`
- Modify: `examples/showcase/src/panels/repair-summary.ts`
- Modify: `examples/showcase/test/sculpt-stroke.test.ts` (and related Showcase tests)

**Interfaces:**
- Consumes: Task 4 `fast-mesh-wasm` (`SculptOp`, `InflateMethod`, `SmoothMethod`, `GrabMethod`, `TwistMethod`, `SurfaceFilter`, `Symmetry`, `verticesInSphere`, `protect.band`).
- Produces (`examples/showcase`):
  - `SculptTool = "inflate" | "deflate" | "smooth" | "grab" | "flatten" | "scrape" | "fill" | "crease" | "pinch" | "clay" | "clayStrips" | "twist"`
  - Persisted settings in `ShowcaseSettings` (`settings.ts`):
    - `sculptMethods: { inflate: InflateMethod; smooth: SmoothMethod; grab: GrabMethod; twist: TwistMethod }` (defaults: `{ inflate: "vertexNormal", smooth: "laplacian", grab: "sphere", twist: "vortex" }`)
    - `sculptGrabPoisson: number` (default `0.4`)
    - `sculptFilter: { kind: "none" | "peaksOnly" | "valleysOnly" | "normalAngle"; threshold: number; contrast: number; maxAngleDeg: number }` (default `{ kind: "none", threshold: 0.05, contrast: 0.5, maxAngleDeg: 45 }`)
    - `sculptSymmetry: { x: boolean; y: boolean; z: boolean }` (default `{ x: false, y: false, z: false }`)
  - UI controls in `SculptToolbox` (`Toolbox.tsx`):
    - Tool buttons (`data-action="sculpt-<tool>"`)
    - Per-brush algorithm `<select data-action="sculpt-method">` + one-line description subtitle when active tool is `inflate`/`deflate`, `smooth`, `grab`, or `twist` (plus Poisson slider when `grab === "kelvinlet"`)
    - Filter `<select data-action="sculpt-filter">` + threshold/angle slider when active
    - Symmetry toggle buttons `data-action="sculpt-sym-x"`, `data-action="sculpt-sym-y"`, `data-action="sculpt-sym-z"`
  - #197 1M-triangle fast paths in `Viewport.tsx` & `mask-protect.ts`:
    - `Grab` stroke calls `store.repair(mesh, { op: "captureStrokeState", onlyIfMissing: false })` at press and `store.repair(mesh, protectedDab(mesh, { op: "grab", ..., fromStroke: true, adaptiveReach: true }))` on `s.move(...)` (estimating gizmo ring radius cleanly without `O(|mesh|)` JS scans).
    - Mask painting uses `await store.verticesInSphere(mesh, hitPoint, radius)` (and symmetric reflected spheres when `sculptSymmetry` is active!) instead of `paintMaskIds` scanning 500k vertices on the main thread.
    - `protectedDab` uses engine-localized soft-band `{ vertices: nearMasked, weights: zeros, band: op.radius * MASK_SOFT_BAND }` (filtering only `m.vertices` — NOT scanning 3M triangle indices or building `maskEdgeField` on the main thread!), and `offTriangles` updates `maskCache` incrementally in `O(dirty_ranges)` when `c.indices !== "all"` and dirty triangles contain no zeroed `(0, 0, 0)` removal slots!
    - Guard `ensureDensity` (`meanEdgeLength`) so it is never called above `REMESH_MAX_TRIANGLES` (`remeshTooBig`).

- [ ] **Step 1: Write failing Showcase Vitest tests for `SculptToolbox` pickers, settings persistence, `dabOperation` mapping, and incremental mask/grab behavior**

Add/update tests in `examples/showcase/test/` verifying:
1. `settings.ts` round-trips `sculptMethods`, `sculptGrabPoisson`, `sculptFilter`, and `sculptSymmetry` in `localStorage` and rejects invalid/corrupt values back to defaults.
2. `dabOperation` maps every `SculptTool`, selected algorithm, `SurfaceFilter`, and `Symmetry` to the exact `RepairOp` / `SculptOp` payload (including `clayStrips` stroke tangent vector, `pinch` -> `crease` with `strength: 0, pinch: ...`, `scrape` -> `flatten` `peaks`, `fill` -> `flatten` `valleys`).
3. `Toolbox.tsx` renders `data-action="sculpt-method"` with one-line descriptions per algorithm, `data-action="sculpt-filter"`, and `data-action="sculpt-sym-x|y|z"` toggles.
4. Engine-driven `Grab` + masked dabs + live splits preserve masked vertices & undo in a single Ctrl+Z step on Sync scene.

- [ ] **Step 2: Run Showcase Vitest to verify RED**

Run: `pnpm --filter fast-mesh-wasm-example-showcase test`
Expected: FAIL on new settings / toolbox / `dabOperation` assertions.

- [ ] **Step 3: Implement `settings.ts`, `app-state.ts`, `sculpt.ts`, `mask-protect.ts`, `Viewport.tsx`, `Toolbox.tsx`, and `repair-summary.ts`**

Implement all UI controls, settings persistence, engine `grab` stroke wiring (keeping `captureStrokeState` ordered before the first `grab` move inside `s`, and opening `beginStep` on Sync scene so the whole grab + live splits undo in one `Ctrl+Z`!), `store.verticesInSphere` mask painting, and localized `band` protection + incremental `maskCache` liveness tracking.

- [ ] **Step 4: Run Showcase Vitest, production build, `sculpt-frames.mjs --verify`, and #197 1M frame-time benchmark**

Run:
1. `pnpm --filter fast-mesh-wasm-example-showcase test`
2. `pnpm --filter fast-mesh-wasm-example-showcase build`
3. `node scripts/sculpt-frames.mjs --verify`
4. `node scripts/sculpt-frames.mjs --tris 100000,1000000 --tools inflate,grab` (plus verify 1M masked live-split stroke frame `p95 < 33 ms` and longest main-thread task `< 50 ms` on both Sync and Worker!).
Expected: All Vitest tests PASS, `--verify` exits `0` (including fold check), and 1M `grab` + masked live-split strokes keep `p95 < 33 ms` and max task `< 50 ms` on both Sync and Worker scenes. Update `docs/sculpt-benchmarks.md` with the 100k & 1M Showcase frame-time table as well.

- [ ] **Step 5: Commit Task 5**

```powershell
git add examples/showcase docs/sculpt-benchmarks.md
git commit -m "feat(showcase): add #177/#192 sculpt brush controls and fix #197 1M grab/mask frame stalls"
```
