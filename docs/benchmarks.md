# Benchmarks

Reference results for `cargo bench -p fast-mesh-bvh`. Times are criterion medians and depend on the machine, so compare runs only on the same hardware.

- **Machine:** Intel Core i5-13600K, 32 GB RAM, Windows 11, single thread
- **Toolchain:** Rust 1.99.0, release profile (LTO, `codegen-units = 1`)
- **Date:** 2026-10-15
- **Meshes:** synthetic height fields (`terrain(n)` in `benches/bvh.rs`): 1,002,528 and 10,008,338 triangles
- **Options:** defaults (`max_leaf_size` 8, `sah_bins` 16, `treelet_size` 4,096)

## BVH (fast-mesh-bvh)

| | 1M triangles | 10M triangles |
| --- | --- | --- |
| Nodes / depth / treelets | 338,389 / 19 / 270 | 3,146,083 / 23 / 4,096 |
| Memory (`byte_size`) | 21.6 MB | 215.3 MB |
| Build, `Quality::Sah` | 439 ms | 5.23 s |
| Build, `Quality::Fast` | 115 ms | 1.70 s |
| Move 484 vertices, commit + update (refit) | 0.16 ms | 0.14 ms |
| Move 50,176 vertices, commit + update (refit) | 15.7 ms | 17.3 ms |
| Remove + re-add 200 triangles, commit + update (treelet rebuild) | 6.3 ms | 2.8 ms |
| First topology edit after loading (one-time array growth) | 6.5 ms | 55.8 ms |

Update times are steady state: one mesh and one BVH are edited repeatedly, as in an interactive session. Small edits cost about the same at 1M and 10M because they touch only the affected leaves and treelets. The first edit after loading a mesh is slower once: the store's columns and the BVH's arrays grow from their exact initial size, which copies them one time.

## Queries (fast-mesh-query)

Results for `cargo bench -p fast-mesh-query`, measured on the same `terrain(n)` meshes with a BVH built with default options. Ray and point benchmarks run a batch of 100,000 queries on one thread, and the table divides the criterion median by the batch size.

- **Machine:** Intel Core i5-13600K, 32 GB RAM, Windows 11, single thread
- **Toolchain:** Rust 1.99.0, release profile (LTO, `codegen-units = 1`)
- **Date:** 2026-10-16
- **Rays:** origins at `z = 20` over the terrain, directions `(jx, jy, -1)` with jitter in ±0.2 (most rays hit; those near the border may miss)
- **Points:** random positions over the terrain at heights in `[-5, 5]`, no `max_distance`
- **Regions:** one sphere of radius 5 at the center of the terrain

| | 1M triangles | 10M triangles |
| --- | --- | --- |
| `raycast`, per ray | 1.07 µs | 1.70 µs |
| `raycast_any`, per ray | 1.00 µs | 1.60 µs |
| `closest_point`, per point | 2.52 µs | 3.44 µs |
| `triangles_in`, sphere radius 5 | 6.8 µs (114 triangles) | 7.7 µs (157 triangles) |
| `vertices_in`, sphere radius 5 | 2.9 µs (44 vertices) | 6.8 µs (63 vertices) |

These are the numbers after issue #17 (key-ordered traversal with a fixed stack). Before #17, on 2026-10-16: `raycast` 1.99 µs and 2.67 µs, `raycast_any` 1.82 µs and 2.45 µs, `closest_point` 2.80 µs and 3.62 µs, `triangles_in` 6.6 µs and 6.9 µs, `vertices_in` 3.0 µs and 4.1 µs (1M and 10M). Ray queries are about 1.9× faster at 1M and 1.5× at 10M; `closest_point` is about 10% faster; region queries are unchanged within noise.

Ray and point queries cost little more at 10M than at 1M because the tree is only a few levels deeper. Region queries cost about the same at both sizes because they visit only the nodes near the sphere. The 1M column is the fastest of three runs on 2026-10-17 (the others measured `raycast` at 1.28 µs and 1.63 µs, `closest_point` at 3.26 µs and 2.74 µs, `triangles_in` at 10.9 µs and 9.1 µs); the 10M column is from the last run, which also measured the 1M slowly. Runs on this machine differ by up to 50% beyond criterion's own intervals, so treat differences below that spread as noise, and compare only numbers from the same run.

## Mesh-vs-mesh queries (fast-mesh-query)

Results for `cargo bench -p fast-mesh-query`, same machine and toolchain as above, run on 2026-10-17. Each size has two `terrain(n)` meshes with default BVHs. B is flipped 180° about x and placed over A, then turned by `0.1° * i` about z around the middle of the terrain, for `i` in 0..64. Every iteration uses the next of these 64 precomputed `Rigid`s, so each time is a median over the cycle (the pose changes the contact a lot, hence the wide criterion intervals). Each pose is lifted so that the surfaces just touch at their closest place, measured on a grid of samples of A, so the true surfaces cross by a small amount ("touching" below). "Pushed" lowers B 0.5 into A, and "apart" lifts it 0.3 away. The `triangles_near` margin is 1% of the terrain height range (0.080).

Result sizes at poses 0, 31 and 63 (0°, 3.1°, 6.3°):

| | 1M triangles | 10M triangles |
| --- | --- | --- |
| `triangles_near` (triangles of A + of B) | 4,607 + 4,607 / 188 + 188 / 182 + 182 | 48,414 + 48,414 / 359 + 359 / 353 + 353 |
| `intersecting_pairs`, pushed 0.5 | 18,554 / 3,188 / 928 pairs | 194,919 / 7,766 / 6,344 pairs |
| `mesh_distance`, touching | 0 at every pose | 0 at every pose |
| `mesh_distance`, apart 0.3 | 0.299 / 0.287 / 0.279 | 0.299 / 0.267 / 0.298 |

Times (criterion median over the 64-pose cycle):

| | 1M triangles | 10M triangles |
| --- | --- | --- |
| `intersects`, touching | 0.72 ms | 1.20 ms |
| `intersects`, apart 0.3 | 30 µs | 0.045 µs (rejected almost at once) |
| `mesh_distance`, touching | 23 ms | 49 ms |
| `mesh_distance`, apart 0.3 | 5.5 ms | 62 ms |
| `triangles_near`, margin 0.080 | 11.5 ms | 91 ms |
| `intersecting_pairs`, pushed 0.5 | 14.8 ms | 199 ms |

The cost of `intersects` grows little with size, and `triangles_near` and `intersecting_pairs` follow the contact size: the 10M poses at 0° have 10× the contact of the 1M ones. 10M results have criterion intervals of 2× to 6× around the median, because the contact varies so much across the cycle; read them as orders of magnitude.

**Against the spec's targets (§1).**
- `intersects` takes 0.7 ms at 1M vs 1M: within the "few ms" target.
- `mesh_distance` takes 5.5 ms when the meshes are 0.3 apart, under 2× a "few ms" (about 3 ms) target, but 23 ms when they touch, which misses it by more than 2×. A follow-up issue should look at the touching case; the cause was not investigated here.
- `triangles_near` costs in proportion to contact: 11.5 ms at 1M for a few hundred to 4,600 triangles per side, and 91 ms at 10M for 350 to 48,000.

## How the defaults were chosen

`treelet_size` sets the cost of a topology edit, because every edit rebuilds the treelets it touches. Measured on 1M triangles (first-edit timing, before the steady-state benchmark existed, so absolute numbers are higher than the table above):

| `treelet_size` | Remove + re-add 200 triangles | Move 484 vertices |
| --- | --- | --- |
| 65,536 | 29.6 ms | 2.4 ms |
| 16,384 | 17.6 ms | 2.1 ms |
| 4,096 | 8.9 ms | 2.1 ms |

The default is **4,096**: topology edits cost about a third of the 65,536 setting, while moves and builds are unaffected. Even at 10M triangles the top tree stays small (4,096 treelets).

`full_rebuild_fraction` (0.30) and `degradation_factor` (1.5) are kept. Moving about 10% of the vertices of a 1M mesh (50,176 of 502,681) refits in about 16 ms, against 439 ms for a full SAH build, so falling back to a full rebuild earlier than 30% dirty would be slower.

**Moves right after a topology edit.** Refits find the triangles of moved vertices through the mesh's incidence index. Since issue #12 the index follows topology edits (only the touched rows are rewritten) instead of being rebuilt by the next move. Measured on 1M triangles in one run on 2026-10-16, before and after that change:

| | Before (rebuild per edit) | After (index follows edits) |
| --- | --- | --- |
| Remove + re-add 200 triangles, then move 484 vertices | 22.3 ms | 6.2 ms |
| Remove + re-add 200 triangles alone (unchanged) | 4.8 ms | 4.6 ms |
| Move 484 vertices alone (unchanged) | 0.07 ms | 0.07 ms |

These were measured on a later day than the main table above, so compare them only with each other. The index is still rebuilt in full after `Mesh::compact`, or after edits that leave more than 1/8 of it rewritten.

## Reproducing

```sh
cargo bench -p fast-mesh-bvh                          # 1M triangles
FAST_MESH_BENCH_LARGE=1 cargo bench -p fast-mesh-bvh  # adds 10M (needs ~2 GB RAM, several minutes)
cargo bench -p fast-mesh-query                        # the same for the query benchmarks (single-mesh and mesh-vs-mesh)
FAST_MESH_BENCH_LARGE=1 cargo bench -p fast-mesh-query
```
