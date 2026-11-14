# Sculpt Domain Architecture: Brush Algorithms, Surface Filters, Extra Brushes, Mirror Symmetry & 1M Frame-Time Optimization — Design Specification

- **Date:** 2026-11-13
- **Issues:** [#177](https://github.com/YoonDongGeun/fast-mesh-wasm/issues/177), [#192](https://github.com/YoonDongGeun/fast-mesh-wasm/issues/192), [#197](https://github.com/YoonDongGeun/fast-mesh-wasm/issues/197)
- **Branch:** `feat/177-192-sculpt-brushes`
- **Architectural References:** [MeshInspector / MeshLib](https://github.com/MeshInspector/MeshLib) (domain modularization & localized `O(|region|)` operators), [elalish/manifold](https://github.com/elalish/manifold) (deterministic float geometry invariants & parallel-ready design), de Goes & James 2017 (*Regularized Kelvinlets*).

---

## 1. Intent, Domain Separation & Why

Downstream builders of web-based 3D sculpting, dental/medical CAD, and mesh processing tools — as well as users of `examples/showcase` — need three things together:

1. **Runtime-Selectable, Physically/Geometrically Distinct Sculpt Kernels (#177 & #192):**
   - **Inflate:** Per-vertex smooth normal (`VertexNormal`), falloff-weighted average normal (`AverageNormal`, lifts detail without folding sharp ridges), and stroke-start frozen normal (`StrokeNormal`, prevents normal-feedback runaway across repeated dabs).
   - **Smooth:** Volume-preserving `Taubin` ($\lambda/\mu$), fast uniform `Laplacian`, Vollmer–Mencl–Müller `HcLaplacian` (resists shrinkage on thin walls), and `Cotangent` (avoids tangential vertex drift on unevenly tessellated meshes).
   - **Grab:** Euclidean `Sphere`, surface-connected `Geodesic` (never jumps across air gaps to opposing dental arches, lips, or fingers), and regularized linear-elastic `Kelvinlet` (couples axial pull and Poisson lateral bulge/necking).
   - **Composable `SurfaceFilter`:** `None`, `PeaksOnly`, `ValleysOnly`, `NormalAngle` (planar/crease-respecting cone).
   - **Extra Brushes:** Plane-based `Flatten` (`Both` / `Peaks` [scrape] / `Valleys` [fill]), `Crease` (+ `Pinch`), `Clay` (`Round` / `Strips`), and `Twist` (`Vortex` / `Kelvinlet`).
   - **Local-Axis Mirror `Symmetry`:** `X` / `Y` / `Z` (`1..8` passes) inside one `O(|brush|)` BVH dab and one undo step.

2. **Clean Domain Modularization (MeshInspector / MeshLib Pattern):**
   - Previously, mesh repair (`weld`, `fill_holes`, `orient`), brush sculpting (`inflate`, `smooth`, `grab`), global remeshing (`simplify`, `subdivide`, `offset`, `shell`), and boolean operations lived flat in `crates/fast-mesh-ops/src/` and piled onto a single `Scene.repair` surface.
   - Starting in this branch, brush sculpting is organized into a dedicated **`sculpt` domain module** (`crates/fast-mesh-ops/src/sculpt/`, re-exported from `fast_mesh_ops::sculpt` and kept backwards-compatible at `fast_mesh_ops::deform`), paired with clean `SculptOp` / `scene.sculpt(mesh, op)` TypeScript domain types alongside backwards-compatible `RepairOp` / `scene.repair(...)` support. Subsequent steps continue domain separation across `repair/`, `remesh/`, `boolean/` (with `elalish/manifold`-style generalized winding numbers + directional symbolic perturbation + SDF/EDT fallback), and `deform/` (#193).

3. **Zero Main-Thread Long Tasks at 1M Triangles (#197):**
   - Two Showcase paths previously caused 70–173 ms main-thread stalls at 1M triangles:
     1. Tangential `Grab` scanned all 500k vertices on the JS main thread (`new GrabRegion`) and serialized tens of thousands of `setPositions` coordinates per pointer move.
     2. `Mask` painting (`paintMaskIds`) and masked dabs after `splitStretched` (`maskLive` + `maskEdgeField`) scanned all 500k vertices / 3M triangle indices on the JS main thread to rebuild whole-mesh CSR graphs whenever a live split occurred.
   - Moving interactive `Grab` into the engine (`fromStroke: true, adaptiveReach: true` with lazy `O(|brush|)` stroke-state capture) and replacing JS whole-mesh mask scans with **BVH-driven `O(|brush|)` mask picking and localized soft-band evaluation** eliminates all `O(|mesh|)` main-thread work at 1M triangles.

---

## 2. Goals, Non-Goals & Anti-Goals

### Goals
1. Deliver all **#177** brush algorithms (`InflateMethod` × 3, `SmoothMethod` × 4, `GrabMethod` × 3) and **#192** features (`SurfaceFilter` × 4, `Flatten`/`Scrape`/`Fill`, `Crease`/`Pinch`, `Clay`/`ClayStrips`, `Twist` × 2, local `X`/`Y`/`Z` `Symmetry`) in `fast-mesh-ops`, `fast-mesh-scene`, `fast-mesh-wasm`, `packages/fast-mesh-wasm` (`Scene`, `WorkerScene`, `SceneStore`), and `examples/showcase`.
2. Modularize `crates/fast-mesh-ops/src/sculpt/` (`mod.rs`, `stroke.rs`, `filter.rs`, `inflate.rs`, `smooth.rs`, `grab.rs`, `plane.rs`, `crease.rs`, `twist.rs`) with full `fast_mesh_ops::deform` backwards re-exports, and expose `scene.sculpt` / `SculptOp` in TypeScript while keeping `scene.repair` / `RepairOp` 100% compatible.
3. Resolve **#197**: at 1M triangles, tangential `Grab` along the surface (without folding) and masked dabs/painting with live splits keep frame `p95 < 33 ms` and longest main-thread task `< 50 ms` on both Sync and Worker scenes (`scripts/sculpt-frames.mjs` & `--verify`).
4. Measure and record 100k-triangle and 1M-triangle `Scene::dab` timings per algorithm in `docs/sculpt-benchmarks.md`.

### Non-Goals
- Arbitrary world-gizmo mirror planes (mirror symmetry uses local mesh axes `x = 0`, `y = 0`, `z = 0`).
- Manifold boolean rewrite (#194 / dedicated boolean domain step) and 1M STL load-time decimation (#198) in this PR — tracked as their own focused branches/PRs immediately following this branch.

### Anti-Goals
- Any `O(|mesh|)` scan on the JS main thread or inside `Scene::dab` on Eager/Lazy BVH meshes (including at stroke start!).
- Modifying `fast-mesh-store`'s `ChannelDesc::width` `1..=4` public contract.
- Any divergence between `Scene::dab` (`*_region`) and whole-mesh `sculpt::*` for live triangle-used vertices.

---

## 3. Constraints & Engine Invariants

1. **Fallible Allocations, Zero `HashMap`, Zero Allocating Sorts:** All heap growth in `fast-mesh-ops` and `fast-mesh-scene` uses `reserve`, `push`, `filled`, or `try_reserve` -> `OutOfMemory` with zero partial writes on failure; no `HashMap`/`HashSet` and no `slice::sort*` (`sort_unstable*` only) in non-test engine code.
2. **Bit-for-Bit Region & Parallel Parity:** `Scene::dab` matches whole-mesh `sculpt::*` bit-for-bit across `Eager`, `Lazy`, and `Manual` modes, and `--features parallel` builds match sequential builds bit-for-bit.
3. **`Scene` Field-Unit Discipline:** Every new `&mut self` method on `fast_mesh_scene::Scene` calls `self.drop_field_unit()` first and is listed in `fields::tests::mutating_calls_abandon_open_unit`.
4. **UV Channel Preservation:** All sculpt operations only update positions via `tx.set_positions` and preserve `UV_CHANNEL`; covered in `crates/fast-mesh-ops/src/uv_table.rs` and `docs/uv-channel.md`.

---

## 4. Approaches Considered

### Approach 1 (Selected): `sculpt/` Domain Pipeline + Dual Width-3 Topology-Bound Channels + Localized BVH Masking
- **Summary:** Organize all brush kernels under `crates/fast-mesh-ops/src/sculpt/` sharing orthogonal `SurfaceFilter` and `Symmetry` modifiers. Store stroke-start rest positions and smooth normals in two width-3 `bound_to_topology` source channels (`STROKE_POS_CHANNEL` + `STROKE_NORMAL_CHANNEL`), supporting both eager whole-mesh capture (`capture_stroke_state`) for standalone callers and **`O(|brush|)` lazy region capture (`begin_stroke` + `capture_stroke_region`)** in `Scene::dab` so 1M-triangle stroke starts cost `O(1)`! Drive Showcase `Grab` through engine `grab` (`fromStroke: true, adaptiveReach: true`) and replace JS whole-mesh mask scans with BVH region queries (`verticesInSphere`) and localized edge-band propagation.
- **Pros:** Keeps `fast-mesh-store` (`width` `1..=4`) untouched; automatic parent interpolation during `split_stretched`; solves #177, #192, and #197 together with zero duplicated math.
- **Cons:** Slightly richer `StrokeGrabCache` in `fast-mesh-scene` to support directional `ahead` gating and lazy rest capture.

### Approach 2: Per-Brush Inline Options + JS-Side `GrabRegion` / Whole-Mesh `maskEdgeField`
- **Rejected Why:** Duplicates filter/symmetry logic across 7 brushes and leaves `O(|mesh|)` main-thread scans at 1M triangles (#197).

---

## 5. Detailed Design (Incorporating Builder Check Resolutions)

### 5.1 Two Width-3 Topology-Bound Stroke Channels & Lazy Region Capture *(Builder Call #1)*
- **Why two width-3 channels:** `fast-mesh-store` enforces `ChannelDesc::width` in `1..=4` (`crates/fast-mesh-store/src/mesh.rs:369`). Splitting stroke state into two width-3 channels keeps `fast-mesh-store` 100% unchanged:
  - `pub const STROKE_POS_CHANNEL: &str = "stroke_rest_pos";` (`ChannelDesc::source(..., 3).bound_to_topology()`, default `[f32::NAN; 3]` meaning "not yet captured in this stroke").
  - `pub const STROKE_NORMAL_CHANNEL: &str = "stroke_rest_normal";` (`ChannelDesc::source(..., 3).bound_to_topology()`, default `[f32::NAN; 3]`).
- **Two Capture Modes:**
  1. **Whole-mesh `capture_stroke_state(mesh: &mut Mesh) -> Result<()>`** (and `ensure_stroke_state`): fills both channels (`STROKE_POS_CHANNEL` + `STROKE_NORMAL_CHANNEL`) for all live vertices up front for standalone `fast-mesh-ops` callers.
  2. **`O(1)` Lazy `StrokeSnapshot` on `fast_mesh_scene::Entry`:**
     - Calling `Scene::capture_stroke_state(id, only_if_missing)` bumps an `O(1)` stroke epoch on `Entry` (`~0 ms` on `pointerdown` even at 1M triangles) rather than copying 500k vertices up front.
     - Inside `Scene::dab`, before mutating any vertex in `region`, `Entry` lazily records initial pre-stroke positions and smooth normals for newly touched `region` vertices (and their 1-ring neighbors) stamped with the current stroke epoch.
     - On `Scene::split_stretched`, `StretchReport::vertices: Vec<[VertexId; 3]>` (`[new, a, b]`) interpolates lazy rest positions and rest normals for split midpoints in `O(splits)` (`0.5 * (rest[a] + rest[b])`, re-normalized for normals), matching `bound_to_topology` channel interpolation while invalidating cleanly on any other topology edit, undo, or redo.

### 5.2 Composable `SurfaceFilter` & Max-Envelope `Symmetry` *(Builder Call #2)*
- **`SurfaceFilter`:**
  - `None`: $\phi(v) = 1.0$.
  - `PeaksOnly { threshold, contrast }` / `ValleysOnly { threshold, contrast }`: scale-invariant prominence $h_v = \frac{(p_v - \bar{p}_v) \cdot \hat{n}_v}{\ell_v}$ over 1-ring unique neighbors $N(v)$ with mean edge length $\ell_v$; `s = h_v` (`PeaksOnly`) or `-h_v` (`ValleysOnly`); smoothstep ramp of width $0.25 \cdot (1.0 - \text{contrast})$ above `threshold` (`threshold`, `contrast` in `[0.0, 1.0]`).
  - `NormalAngle { normal: Option<[f32; 3]>, max_angle_deg: f32 }`: cone half-angle in `(0.0, 180.0]`; `None` uses falloff-weighted brush normal; $\phi(v) = 1.0$ across inner 75% of cone, smoothstep taper to `0.0` at `max_angle_deg`.
- **Mirror `Symmetry` with Max-Envelope Centerline Normalization *(Builder Call #2)*:**
  - Active reflection masks `bits` in `0..8` (`M = diag(sx, sy, sz)`, $\det(M) = s_x s_y s_z$).
  - Polar vectors (`center`, `direction`, plane `origin`/`normal`, `tangent`) reflect as $M v$; axial twist reflects `axis' = M * axis`, `strength' = det(M) * strength`.
  - Each active pass $m$ evaluates against the same pre-dab (or rest) positions, yielding displacement $\Delta p_m(v)$ and scalar envelope weight $a_m(v) = \text{falloff}(t_m(v)) \cdot \phi_m(v) \in [0, 1]$.
  - **Max-Envelope Combination:** Let $S(v) = \sum_m a_m(v)$ and $M_{\max}(v) = \max_m a_m(v)$. When $S(v) > 0$:
    $$\gamma(v) = \frac{M_{\max}(v)}{S(v)}, \qquad \Delta p(v) = \gamma(v) \sum_m \Delta p_m(v)$$
    - Disjoint mirror spheres: only one $a_m > 0 \implies \gamma(v) = 1.0$.
    - Centerline overlap ($K$ passes with equal $a_0(v)$): $\gamma(v) = 1/K$, so cross-plane components cancel to `0` AND tangent/normal components preserve the **exact smooth single-brush falloff $a_0(v)$** without any $C^0$ kink or flat punch plateau!

### 5.3 Brush Algorithms (#177) & Extra Brushes (#192)
1. **`Inflate` (`InflateMethod`):**
   - `VertexNormal` (default), `AverageNormal` (falloff-weighted unit average normal), `StrokeNormal` (frozen normal from stroke snapshot / `STROKE_NORMAL_CHANNEL`).
2. **`Smooth` (`SmoothMethod`):**
   - `Taubin` (`0.33` / `-0.34`, default), `Laplacian` (`1.0`), `HcLaplacian` (`HC_ALPHA = 0.1`, `HC_BETA = 0.5` Vollmer–Mencl–Müller pushback toward pre-dab positions), `Cotangent` (clamped `[1e-4, 1e3]` edge cotangent weights precomputed in CSR region buffers).
3. **`Grab` (`GrabMethod` + Fold-Safe `adaptive_reach` + Seed Rule) *(Builder Calls #3 & #4)*:**
   - `GrabMethod`: `Sphere` (default), `Geodesic`, `Kelvinlet` (with `poisson: f32` in `[0.0, 0.5]`, default `0.4`, $\epsilon = 0.35 R_{\text{eff}}$, calibrated so $u(0) = \text{strength} \cdot \text{direction}$ and windowed by $(1 - t^2)^2$).
   - **Fold-Safe `adaptive_reach` with `FALLOFF_SLOPE` & Directional `ahead` Gating *(Builder Call #3)*:**
     - When `adaptive_reach == true`: let $\vec{g} = \text{strength} \cdot \text{direction}$, $L = \|\vec{g}\|$, and kernel max-derivative factor `slope`:
       - `Falloff::Smooth` (`Sphere` / `Geodesic`): `slope = 1.5`
       - `Falloff::Linear` (`Sphere` / `Geodesic`): `slope = 1.0`
       - `GrabMethod::Kelvinlet`: `slope = 1.75` (exceeds peak derivative of calibrated $(1 - t^2)^2$ Kelvinlet across $\nu \in [0, 0.5]$, preventing elastic Jacobian inversion!)
     - Unbounded fold-free radius: $R_{\text{wide}} = L \cdot \text{slope} / \text{GRAB\_MAX\_STRETCH}$ (`GRAB_MAX_STRETCH = 0.8`). When $R_{\text{wide}} \le R$, $R_{\text{eff}} = R$.
     - Smooth directional `ahead` gating (matching `sculpt.ts:351-371` so pulling straight off a surface pulls a narrow horn of radius $R$ while tangential drags widen smoothly without folding): when $R_{\text{wide}} > R$, let $\hat{u} = \text{sign}(\text{strength}) \cdot \text{direction} / \|\text{direction}\|$. Over live candidate vertices with rest offset $\vec{d} = P_{\text{rest}}[v] - c$, $r = \|\vec{d}\| < R_{\text{wide}}$, and $\text{along} = \vec{d} \cdot \hat{u} > 0$ (excluding zero-weight fully protected vertices when `protect` is active):
       $$\text{ahead} = \max_v \left( \frac{\text{along}}{r} \cdot \min\!\left(1,\; 2\left(1 - \frac{r}{R_{\text{wide}}}\right)\right) \right), \qquad R_{\text{eff}} = \max(R,\; R_{\text{wide}} \cdot \text{ahead})$$
     - Snap-back & BVH completeness invariant: when `from_stroke == true`, any vertex moved in an earlier dab of the current stroke whose rest position now falls outside $R_{\text{eff}}$ snaps back to $P_{\text{rest}}[v]$. Because every unmoved vertex still has $P_{\text{curr}}[v] = P_{\text{rest}}[v]$ while every moved vertex is tracked in `StrokeSnapshot::touched`, the union $\text{BVH}(c, \max(R, R_{\text{wide}})) \cup \text{StrokeSnapshot::touched}$ contains 100% of $V_{\text{rest}}$ and all snap-back vertices, so `Scene::dab` matches whole-mesh `grab` bit-for-bit!
   - **`GrabMethod::Geodesic` Seed Triangle Rule *(Builder Call #4)*:**
     - Both whole-mesh `grab` and `grab_region` select seed triangle $T^*$ strictly among live non-degenerate triangles that have **at least one corner strictly inside `Sphere(c, R_eff)`** ($\|P[u] - c\| < R_{\text{eff}}$), minimizing $d_0 = \|\text{closest\_point}(T, c) - c\|$ (ties broken by ascending `TriangleId`), and seed corners with $d(u) = d_0 + \|P[u] - q\|$. Zero-cost-links coincident base positions (`[u32; 3]` bits) so UV seams stay closed. Guarantees 100% bit-for-bit parity between `Scene::dab` and whole-mesh `grab`!
4. **`Flatten` (`FlattenMode`):** `Both` (flatten), `Peaks` (scrape), `Valleys` (fill) onto explicit or auto-fitted brush plane.
5. **`Crease` (`CreaseOptions`):** Normal groove/ridge (`strength * f^2`) + tangent-plane radial pinch (`pinch * f`, pure `Pinch` when `strength == 0`).
6. **`Clay` (`ClayProfile`):** `Round` and `Strips` (oriented $2R \times 2R$ tangent-frame box footprint, BVH query radius $\sqrt{3} R$) lifting/carving to offset brush plane.
7. **`Twist` (`TwistMethod`):** `Vortex` (Rodrigues falloff rotation) and `Kelvinlet` (calibrated regularized twisting Kelvinlet windowed by $(1 - t^2)^2$).

### 5.4 Resolving #197: O(|brush|) BVH Mask Painting, Localized Soft-Band & Fast 1M Grab
1. **BVH Region Query Exposed on `Scene` & `WorkerScene`:**
   - Add `Scene::vertices_in_sphere(&mut self, id: MeshId, center: [f32; 3], radius: f32) -> Result<Vec<VertexId>>` (`fast-mesh-scene`, calling `self.drop_field_unit()` + `entry.update_bvh()` + `fast_mesh_query::vertices_in`), exposed on `WasmScene`, sync `Scene`, and `WorkerScene` (`scene.verticesInSphere(mesh, center, radius): Uint32Array | Promise<Uint32Array>`).
   - Showcase `MaskLayer` (`Viewport.tsx`) uses `scene.verticesInSphere` instead of JS `paintMaskIds` (`O(500k)` scan), dropping 1M mask-paint tasks from **100–150 ms to < 2 ms**!
2. **Engine-Localized Soft-Band Mask Protection (Zero `O(|mesh|)` Rebuild After Live Splits):**
   - Extend `VertexProtection` / opcode `32` with `band: f32` (`0.0` default = exact listed weights as today).
   - When `band > 0.0` and weights are `0.0` (masked vertex set), `Scene::dab_protected` queries `vertices_in(Sphere(center, q_radius + band))` via BVH, intersects with the masked set in `O(|brush|)`, runs a localized Dijkstra edge-distance pass out to `band` over only that neighborhood via `entry.mesh.incidence()`, and applies `smoothstep(dist / band)` weights.
   - Showcase eliminates JS `maskEdgeField` (`3M` triangle index scan + `24 MB` CSR allocation) on sculpted dabs and updates mask liveness incrementally via `StretchReport.vertices` (`[new, a, b]`) in `O(splits)`.
3. **Fast 1M Tangential Grab:**
   - Driving `Grab` (`fromStroke: true, adaptiveReach: true`) through `Scene::dab` eliminates JS `GrabRegion` (`O(500k)` scan) and `setPositions` array marshaling.
   - `Entry`'s lazy `O(1)` stroke epoch avoids whole-mesh copy on pointer-down, and `grab_region` skips `tx.set_positions` for unchanged (`same_bits`) vertices so `WasmScene::take_changes` and `Bvh::update` remain strictly proportional to actually moved vertices (`< 4–8 ms` per drag frame at 1M triangles, well under #197's `p95 < 33 ms` and `< 50 ms` max-task budget).

### 5.5 Wasm FFI (`edit.rs`) & TypeScript Domain API (`packages/fast-mesh-wasm`)
- **Rust `edit.rs` Opcodes:**
  - `5` `inflate`: `cx, cy, cz, radius, falloff, strength, method (0 vertex, 1 average, 2 stroke)`
  - `6` `smooth`: `cx, cy, cz, radius, falloff, strength, cycles, method (0 taubin, 1 laplacian, 2 hcLaplacian, 3 cotangent)`
  - `9` `grab`: `cx, cy, cz, radius, falloff, strength, dx, dy, dz, method (0 sphere, 1 geodesic, 2 kelvinlet), poisson, from_stroke, adaptive_reach`
  - `18` `captureStrokeState`: `only_if_missing (0 or 1)` (standalone non-edit op; resets/captures stroke state)
  - `19` `flatten`: `cx, cy, cz, radius, falloff, strength, mode (0 both, 1 peaks, 2 valleys), has_origin, ox, oy, oz, has_normal, nx, ny, nz`
  - `20` `crease`: `cx, cy, cz, radius, falloff, strength, pinch`
  - `21` `clay`: `cx, cy, cz, radius, falloff, strength, profile (0 round, 1 strips), has_tangent, tx, ty, tz`
  - `22` `twist`: `cx, cy, cz, radius, falloff, strength, method (0 vortex, 1 kelvinlet), has_axis, ax, ay, az`
  - `32` `protect`: `n, band (0 for exact weights), id × n, weight × n`, wrapping any brush op
  - `33` `brushModifiers`: `sym_mask (0..7), filter_kind (0 none, 1 peaks, 2 valleys, 3 normalAngle), p0, p1, has_vec, vx, vy, vz`, wrapping any brush op
- **TypeScript Domain Exports (`packages/fast-mesh-wasm`):**
  - Export dedicated `SculptOp` / `SculptReport` union types + `scene.sculpt(...)` / `workerScene.sculpt(...)` / `store.sculpt(...)` domain aliases alongside full `RepairOp` / `scene.repair(...)` support, so callers can adopt domain-separated APIs without breaking any existing `repair` call site.
  - Export `scene.verticesInSphere(mesh, center, radius)` on both `Scene` and `WorkerScene`.
  - Add `.changeset/sculpt-domain-brushes-filters-symmetry.md` (`minor`).

### 5.6 Showcase `SculptToolbox` (`Toolbox.tsx`), `settings.ts` & Verification
- **Tools (`SculptTool`):** `inflate`, `deflate`, `smooth` (defaults to `laplacian` in Showcase interactive sculpting to preserve current interactive feel, while engine `SmoothOptions::default()` stays `Taubin` — with all 4 methods `laplacian`, `taubin`, `hcLaplacian`, `cotangent` selectable in the picker!), `grab`, `flatten`, `scrape`, `fill`, `crease`, `pinch`, `clay`, `clayStrips`, `twist`.
- **Per-Brush Method Picker (`data-action="sculpt-method"`):** Persisted in `ShowcaseSettings.sculptMethods`, showing one-line descriptions per algorithm.
- **Filter (`data-action="sculpt-filter"`) & Symmetry (`data-action="sculpt-sym-x|y|z"`):** Persisted in `ShowcaseSettings.sculptFilter` and `ShowcaseSettings.sculptSymmetry`.
- **Verification Gates:**
  - Independent-expected-value unit tests per algorithm, filter, extra brush, and symmetry combination.
  - 100k & 1M dab benchmark table in `docs/sculpt-benchmarks.md`.
  - Headless Chrome `node scripts/sculpt-frames.mjs --verify` + 1M tangential grab & masked live-split frame-time check (#197 acceptance: frame `p95 < 33 ms`, longest main-thread task `< 50 ms`).

---

## 6. What Is Left to the Builder

- Exact file boundaries inside `crates/fast-mesh-ops/src/sculpt/` and Showcase `Toolbox.tsx` responsive toolbar layout.
- Whether `Scene`'s lazy stroke snapshot uses `Entry`-side epoch-stamped `Vec`s mirrored over `StretchReport::vertices` plus `STROKE_*_CHANNEL` fallback, as long as `Scene::dab` and whole-mesh `sculpt::*` stay bit-for-bit identical and pointer-down stays `O(1)`.
