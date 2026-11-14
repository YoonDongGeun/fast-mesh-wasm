# Sculpt Brush, Filter, Symmetry & Stroke Snapshot Benchmarks (100k & 1M Triangles)

This document records the single-dab release latencies of `fast_mesh_scene::Scene::dab`, `Scene::dab_protected_band`, `Scene::capture_stroke_state`, and `Scene::vertices_in_sphere` across **100k** and **1M** triangle meshes (#177, #192, #197).

## Methodology

- **Harness:** `crates/fast-mesh-scene/benches/sculpt.rs` compiled with Cargo's optimized `bench` (`opt-level = 3`) profile (`$env:CARGO_PROFILE_BENCH_DEBUG="0"; $env:CARGO_INCREMENTAL="0"`).
- **Meshes:** Origin-centered wavy terrain grids with unit vertex spacing:
  - **100k:** `terrain(224)` — `50,625` vertices, `100,352` triangles (`193` vertices inside `r = 8`, `1,789` inside `r = 24`).
  - **1M:** `terrain(708)` — `502,681` vertices, `1,002,528` triangles (`193` vertices inside `r = 8`, `1,789` inside `r = 24`).
- **End-to-End Stroke Loop:** Every `Scene::dab` / `Scene::dab_protected_band` measurement runs on an `UpdateMode::Eager` mesh with `scene.set_normals(id, true)` enabled inside an open history step (`scene.begin_step()`) and drains `scene.take_changes()` after each dab — measuring the full real-time viewport frame cost (BVH sphere gather + lazy 1-ring stroke capture when active + brush kernel + undo journal patch + incremental Eager BVH refit + incremental Eager vertex-normal recompute + change-report drain).

### Reproducing

```powershell
# Fast release median & p95 Markdown table (100k + 1M):
$env:CARGO_PROFILE_BENCH_DEBUG="0"; $env:CARGO_INCREMENTAL="0"; $env:FAST_MESH_BENCH_MD="1"; cargo bench -p fast-mesh-scene --bench sculpt

# Full Criterion statistical suite:
$env:CARGO_PROFILE_BENCH_DEBUG="0"; $env:CARGO_INCREMENTAL="0"; Remove-Item Env:FAST_MESH_BENCH_MD -ErrorAction SilentlyContinue; cargo bench -p fast-mesh-scene --bench sculpt
```

---

## Summary Highlights

1. **`O(1)` Pointer-Down Stroke Snapshot (`Scene::capture_stroke_state`):**
   - Whole-mesh `sculpt::capture_stroke_state` takes **`1,710.5 µs` (`1.71 ms`)** at 100k and **`14,795.9 µs` (`14.80 ms`)** at 1M triangles.
   - `Scene::capture_stroke_state` bumps `Entry::stroke.epoch` in `O(1)` (**`< 0.1 µs` p95** at both 100k and 1M) and lazily captures rest positions and smooth normals only for vertices in each dab's brush region + 1-ring while remaining bit-for-bit identical (`0` ULP) to whole-mesh `capture_stroke_state` — including across mid-stroke `Scene::split_stretched`.
2. **Sub-millisecond `O(|brush|)` Scaling from 100k to 1M (10× Mesh Growth):**
   - Every single-sphere `r = 8` (`~193` vertices) brush, method, surface filter, and soft-protection band dab runs in **`85–206 µs`** on **both** 100k and 1M triangles (~1% of a 16.67 ms 60 fps frame), outperforming whole-mesh `scan` by **18×–88×** at 1M triangles:
     - `inflate/vertex_r8`: **`126.5 µs`** (`Scene::dab`) vs **`11,178.9 µs`** (`scan/inflate_r8`) — **88.4× speedup**.
     - `smooth/laplacian_r8` (2 cycles): **`139.8 µs`** (`Scene::dab`) vs **`8,471.1 µs`** (`scan/smooth_r8`) — **60.6× speedup**.
     - `grab/sphere_r8`: **`122.2 µs`** (`Scene::dab`) vs **`2,171.6 µs`** (`scan/grab_r8`) — **17.8× speedup**.
3. **Multi-Plane Mirror Symmetry (`Symmetry`):**
   - Off-axis `r = 8` dabs scale with active mirror octants rather than mesh size (`99.8 µs` `none`, `209.4 µs` `x`, `440.3 µs` `xy`, `541.3 µs` `xyz` on 1M triangles — all under `0.55 ms`).

---

## Measured Single-Dab Release Latencies (100k vs. 1M)

### 1. Stroke State Capture (Pointer-Down)

| Benchmark | 100k Median (µs) | 100k p95 (µs) | 1M Median (µs) | 1M p95 (µs) | Notes |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `stroke/scene_capture_o1` | 0.00 | 0.10 | 0.00 | 0.10 | `Scene::capture_stroke_state(id, false)` (`O(1)` epoch bump) |
| `stroke/whole_mesh_capture` | 1710.50 | 2135.20 | 14795.90 | 15932.40 | Standalone `sculpt::capture_stroke_state(&mut mesh)` (`O(V + F)`) |

### 2. Core Brushes & Methods (`Inflate`, `Smooth`, `Grab`)

| Benchmark | 100k Median (µs) | 100k p95 (µs) | 1M Median (µs) | 1M p95 (µs) | Notes |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `inflate/vertex_r8` | 116.10 | 198.70 | 126.50 | 200.60 | `InflateMethod::VertexNormal`, `r = 8` (`193` verts) |
| `inflate/vertex_r24` | 1278.70 | 1836.90 | 922.70 | 1093.40 | `InflateMethod::VertexNormal`, `r = 24` (`1,789` verts) |
| `inflate/average_r8` | 186.00 | 235.30 | 126.00 | 187.60 | `InflateMethod::AverageNormal` (Draw), `r = 8` |
| `inflate/stroke_r8` | 152.40 | 181.00 | 127.10 | 159.80 | `InflateMethod::StrokeNormal` (Layer), lazy stroke slice |
| `scan/inflate_r8` | 1654.80 | 2202.70 | 11178.90 | 11441.60 | Whole-mesh `Scene::edit` baseline (`r = 8`) |
| `smooth/taubin_r8` | 153.80 | 168.50 | 140.50 | 195.50 | `SmoothMethod::Taubin` (`λ = 0.33, μ = -0.34`), `2` cycles |
| `smooth/laplacian_r8` | 148.50 | 193.00 | 139.80 | 179.10 | `SmoothMethod::Laplacian`, `2` cycles, `r = 8` |
| `smooth/laplacian_r24` | 1277.40 | 1455.30 | 1171.50 | 1347.70 | `SmoothMethod::Laplacian`, `2` cycles, `r = 24` |
| `smooth/hc_laplacian_r8` | 130.30 | 169.60 | 138.00 | 168.90 | `SmoothMethod::HcLaplacian` (`α = 0.1, β = 0.5`), `2` cycles |
| `smooth/cotangent_r8` | 138.50 | 150.10 | 161.40 | 188.10 | `SmoothMethod::Cotangent`, `2` cycles |
| `scan/smooth_r8` | 841.70 | 993.00 | 8471.10 | 8762.20 | Whole-mesh `Scene::edit` baseline (`r = 8`) |
| `grab/sphere_r8` | 109.10 | 124.30 | 122.20 | 150.80 | `GrabMethod::Sphere`, `r = 8` |
| `grab/sphere_r24` | 860.10 | 934.90 | 864.20 | 951.10 | `GrabMethod::Sphere`, `r = 24` |
| `grab/geodesic_r8` | 145.00 | 202.10 | 155.00 | 220.10 | `GrabMethod::Geodesic` (local Dijkstra), `r = 8` |
| `grab/kelvinlet_r8` | 118.60 | 169.70 | 118.90 | 152.20 | `GrabMethod::Kelvinlet` (`ν = 0.25`), `r = 8` |
| `grab/stroke_adaptive_r8` | 100.70 | 162.90 | 120.20 | 164.60 | `from_stroke: true`, `adaptive_reach: true`, `r = 8` |
| `scan/grab_r8` | 274.70 | 385.80 | 2171.60 | 2447.80 | Whole-mesh `Scene::edit` baseline (`r = 8`) |

### 3. Extra Brushes (`Flatten`, `Crease`, `Pinch`, `Clay`, `Twist`)

| Benchmark | 100k Median (µs) | 100k p95 (µs) | 1M Median (µs) | 1M p95 (µs) | Notes |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `flatten/both_r8` | 111.80 | 122.20 | 123.90 | 156.70 | `FlattenMode::Both` |
| `flatten/peaks_r8` | 79.50 | 95.80 | 93.50 | 105.30 | `FlattenMode::Peaks` (Scrape) |
| `flatten/valleys_r8` | 100.20 | 110.10 | 115.70 | 164.20 | `FlattenMode::Valleys` (Fill) |
| `crease/groove_r8` | 135.20 | 161.10 | 155.50 | 198.10 | `Crease` (`strength != 0`, `pinch = 0.5`) |
| `crease/pinch_only_r8` | 137.30 | 146.30 | 153.40 | 192.40 | `Pinch` (`strength = 0`, `pinch = 0.6`) |
| `clay/round_r8` | 131.10 | 175.90 | 121.10 | 164.30 | `ClayProfile::Round` |
| `clay/strips_r8` | 205.40 | 315.10 | 201.10 | 230.50 | `ClayProfile::Strips` (oriented cube, `q_radius = r * √3`) |
| `twist/vortex_r8` | 146.40 | 160.00 | 123.20 | 139.50 | `TwistMethod::Vortex` |
| `twist/kelvinlet_r8` | 108.80 | 124.70 | 117.90 | 136.00 | `TwistMethod::Kelvinlet` |

### 4. Surface Filters, Mirror Symmetry, Soft Protection Band & Sphere Query

| Benchmark | 100k Median (µs) | 100k p95 (µs) | 1M Median (µs) | 1M p95 (µs) | Notes |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `filter/none_r8` | 109.50 | 131.00 | 117.90 | 161.90 | `SurfaceFilter::None` |
| `filter/peaks_only_r8` | 93.40 | 135.60 | 103.30 | 117.10 | `SurfaceFilter::PeaksOnly { threshold: 0.01, contrast: 0.5 }` |
| `filter/valleys_only_r8` | 85.10 | 98.50 | 93.70 | 103.90 | `SurfaceFilter::ValleysOnly { threshold: 0.01, contrast: 0.5 }` |
| `filter/normal_angle_r8` | 114.20 | 124.30 | 130.10 | 185.90 | `SurfaceFilter::NormalAngle { normal: None, max_angle_deg: 45.0 }` |
| `symmetry/none_r8` | 99.20 | 111.00 | 99.80 | 120.80 | `1` sphere at `[28, 28, z]` |
| `symmetry/x_r8` | 212.60 | 241.90 | 209.40 | 230.00 | `2` reflected spheres (`±X`) |
| `symmetry/xy_r8` | 438.40 | 535.00 | 440.30 | 544.00 | `4` reflected spheres (`±X, ±Y`) |
| `symmetry/xyz_r8` | 562.40 | 657.30 | 541.30 | 643.10 | `8` reflected spheres (`±X, ±Y, ±Z`) |
| `protect/dab_protected_band_r8` | 147.40 | 210.00 | 156.50 | 237.30 | `Scene::dab_protected_band` (`band = 3.0`, half-region held at `0.0`) |
| `query/vertices_in_sphere_r8` | 20.60 | 21.60 | 20.70 | 22.70 | `Scene::vertices_in_sphere` (`r = 8`, `193` verts) |
| `query/vertices_in_sphere_r24` | 192.50 | 224.80 | 189.80 | 207.30 | `Scene::vertices_in_sphere` (`r = 24`, `1,789` verts) |

---

## Showcase Browser Frame Times (100k & 1M Triangles)

Measured against the production Showcase build (`examples/showcase/dist`) on 2026-11-14 in headless Google Chrome `157.0.8103.78` over CDP on an `Intel Core i5-13600K` (Windows):

### 1. Standard Stroke Frame Times (`scripts/sculpt-frames.mjs`)

Command: `node scripts/sculpt-frames.mjs --tris 100000,1000000 --tools inflate,grab` (`sphereStl`, brush `radius = 0.1`, stroke & live remeshes off, `240` pointer events spaced by `8 ms` along a `60 px`-radius circle around the framed sphere center).

| Mode | Triangles | Tool | Frame Median (ms) | Frame p95 (ms) | Frame Max (ms) | Longest Task (ms) | Dabs / Stroke Elapsed |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| `sync` (`?local`) | 99,852 | `inflate` | 6.1 | 6.1 | 6.2 | 0.0 | 628 dabs in 1929 ms |
| `sync` (`?local`) | 99,852 | `grab` | 6.1 | 6.1 | 12.2 | 0.0 | 87 dabs in 1933 ms |
| `worker` | 99,852 | `inflate` | 6.1 | 6.1 | 6.2 | 0.0 | 587 dabs in 1922 ms |
| `worker` | 99,852 | `grab` | 6.1 | 6.1 | 6.2 | 0.0 | 108 dabs in 1925 ms |
| `sync` (`?local`) | 999,996 | `inflate` | 6.1 | 6.1 | 12.1 | 0.0 | 639 dabs in 1921 ms |
| `sync` (`?local`) | 999,996 | `grab` | 6.1 | 97.0 | 127.2 | 125.0 | 160 dabs in 12162 ms |
| `worker` | 999,996 | `inflate` | 6.1 | 6.1 | 6.2 | 0.0 | 621 dabs in 1929 ms |
| `worker` | 999,996 | `grab` | 6.1 | 30.3 | 42.4 | 0.0 | 94 dabs in 2185 ms |

Notes on `999,996`-triangle `grab` in `sculpt-frames.mjs`:
- A `60 px`-radius circle (`120 px` diameter) across the framed radius-`2` sphere corresponds to a `~1.2`-unit tangential displacement (`12×` the `0.1` brush radius). Fold-safe `adaptive_reach` (`GRAB_MAX_STRETCH = 0.8`) widens the influence radius past `~1.4` units and moves `~122k+` vertices (`~25%` of the 500k-vertex mesh) per dab.
- On `worker`, heavy Wasm work runs off the main thread (`p95 = 30.3 ms < 33 ms`, `longTask = 0.0 ms < 50 ms`). On `sync` (`?local`), running those `~122k`-vertex edits + BVH updates on the main thread reaches `p95 = 97.0 ms` (`125.0 ms` max task).

### 2. 1M Mask Paint & Masked Live-Split Strokes (`999,996` Triangles)

Measured in Chrome CDP (`PerformanceObserver` `longtask` + `requestAnimationFrame` gaps) on the `999,996`-triangle sphere after masking a surface patch (`radius = 0.25`) and stroking across the mask border with `split stretched triangles` enabled (`liveRemesh = true`, `LIVE_HUGE_MAX_SPLITS = 2,000` above `100k` triangles):

| Mode | Action | Frame p50 (ms) | Frame p95 (ms) | Frame Max (ms) | Longest Task (ms) | Notes |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `sync` (`?local`) | `maskPaint` (80 events) | 6.1 | 6.1 | 30.3 | 0.0 | `Scene.verticesInSphere` BVH pick + `<FastMesh onClick>` disabled while painting (`overlay builds = 54`, max `9.9 ms`) |
| `worker` | `maskPaint` (80 events) | 6.1 | 6.1 | 30.3 | 0.0 | `WorkerScene.verticesInSphere` (`overlay builds = 50`, max `12.4 ms`) |
| `sync` (`?local`) | `mask+live+inflate` (`r = 0.25`) | 6.1 | 6.1 | 60.6 | 67.0 | `144` dabs (`128` masked drops), `2` live passes (`+0` verts); one-off main-thread incidence index build on first sculpt dab |
| `worker` | `mask+live+inflate` (`r = 0.25`) | 6.1 | 6.1 | 48.5 | 56.0 | `123` dabs (`111` masked drops), `2` live passes (`+0` verts) |
| `sync` (`?local`) | `mask+live+grab` (`55 px`, `r = 0.15`) | 6.1 | 12.2 | 1733.7 | 1707.0 | `62` dabs (`52` masked drops), `3` live passes (`+186` verts); single-threaded Wasm BVH rebuild after split triggers max spike |
| `worker` | `mask+live+grab` (`55 px`, `r = 0.15`) | 6.1 | 6.1 | 84.9 | 89.0 | `72` dabs (`56` masked drops), `2` live passes (`+179` verts); main-thread `three/mesh.ts:computeBounds` on `indices` growth |
| `worker` | `mask+live+grab` (`120 px`, `r = 0.25`) | 6.1 | 6.1 | 103.1 | 107.0 | `66` dabs (`20` masked drops), `3` live passes (`+2,025` verts); capped per pass instead of spiraling unboundedly |

Key Showcase #197 fixes:
- **Mask Painting (`<FastMesh onClick>` + `verticesInSphere`):** Suppressing Three.js's per-triangle pointer click handler while `maskPaint` is armed eliminated the `105–106 ms` main-thread raycast stall on both Sync and Worker (`maskPaint` dropped to `p95 = 6.1 ms`, `longTask = 0.0 ms`).
- **Shared `MaskLiveTracker` & O(|mask|) Protection:** Sharing one `MaskLiveTracker` across `MaskLayer` and `SculptLayer` preserves `verticesInSphere` live trust across stroke handoff and `adoptSplits` without `O(triangles)` `buildMaskLiveCache` rescans on Dab 1; `protectMaskedDab` emits localized `protect.band` ops in `O(|masked|)`.
- **Bounded 1M Live Splits:** Updating the live cooldown timer on `splitStretched` completion and capping passes above `100k` triangles at `LIVE_HUGE_MAX_SPLITS = 2,000` prevents Sync 1M grab strokes from spiraling into `+922k`-vertex cascades. Remaining spikes during 1M topology-changing live splits occur in Worker `three/mesh.ts:computeBounds()` (`72–107 ms`) or Sync main-thread Wasm BVH rebuilds (`~1.5–1.9 s` when splits/large grabs cross BVH rebuild thresholds).
