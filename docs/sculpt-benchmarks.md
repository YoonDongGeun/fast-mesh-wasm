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
