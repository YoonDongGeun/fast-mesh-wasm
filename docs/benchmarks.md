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
- **Date:** 2026-10-16 for the numbers before #17; 2026-10-17 for the numbers after #17 and for the mesh-vs-mesh section
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

Ray and point queries cost little more at 10M than at 1M because the tree is only a few levels deeper. Region queries cost about the same at both sizes because they visit only the nodes near the sphere. The 1M and 10M columns come from different runs. The 1M "after" column is the fastest of four runs on 2026-10-17 (the others measured `raycast` at 1.28, 1.63 and 1.31 µs, `closest_point` at 3.26, 2.74 and 3.30 µs, `triangles_in` at 10.9, 9.1 and 10.7 µs), while the "before" numbers are a single run, so the 1M speedups above are optimistic; the median of the four runs gives about 1.5× for `raycast` and about 0.9× (no gain) for `closest_point`. The 10M column is from the third run (the fourth measured `raycast` at 1.77 µs and `closest_point` at 4.38 µs at 10M). Runs on this machine differ by up to 50% beyond criterion's own intervals, so treat differences below that spread as noise, and compare only numbers from the same run.

## Mesh-vs-mesh queries (fast-mesh-query)

Results for `cargo bench -p fast-mesh-query`, same machine and toolchain as above, run on 2026-10-17. Each size has two `terrain(n)` meshes with default BVHs. B is flipped 180° about x and placed over A, then turned by `0.1° * i` about z around the middle of the terrain, for `i` in 0..64. Every iteration uses the next of these 64 precomputed `Rigid`s, so each time is a median over the cycle (the pose changes the contact a lot, hence the wide criterion intervals). Each pose is lifted so that the surfaces just touch at their closest place, measured on a grid of samples of A ("touching" below). The grid can miss the true closest place, so not every touching pose meets: at 1M, 42 of the 64 intersect and 22 (about a third) are near misses with true gaps of 8e-6 to 0.0027; at 10M, 61 intersect and 3 are near misses (gaps 0.0008 to 0.0019). Near misses are the use case's most common case, so `mesh_distance` on touching poses is also measured on each kind separately. "Pushed" lowers B 0.5 into A, and "apart" lifts it 0.3 away. The `triangles_near` margin is 1% of the terrain height range (0.080).

Result sizes at poses 0, 31 and 63 (0°, 3.1°, 6.3°):

| | 1M triangles | 10M triangles |
| --- | --- | --- |
| `triangles_near` (triangles of A + of B) | 4,607 + 4,607 / 188 + 188 / 182 + 182 | 48,414 + 48,414 / 359 + 359 / 353 + 353 |
| `intersecting_pairs`, pushed 0.5 | 18,554 / 3,188 / 928 pairs | 194,919 / 7,766 / 6,344 pairs |
| `mesh_distance`, touching | 0 at all three (over all 64: 0 at 42 poses, 8e-6 to 0.0027 at 22) | 0 at all three (over all 64: 0 at 61 poses, 0.0008 to 0.0019 at 3) |
| `mesh_distance`, apart 0.3 | 0.299 / 0.287 / 0.279 | 0.299 / 0.267 / 0.298 |

Times (criterion median over each cycle: the 64 poses, or for the split rows the poses of that kind). Machine speed drifted by up to 2× during these runs, as `intersects`, whose code did not change, shows; so each before/after pair below is from back-to-back runs, and only the numbers in one column pair compare with each other.

| | 1M, before | 1M, after | 10M, before | 10M, after |
| --- | --- | --- | --- | --- |
| `intersects`, touching (unchanged code) | 0.49 ms | 0.50 ms | 1.49 ms | 1.51 ms |
| `intersects`, apart 0.3 (unchanged) | 22 µs | 22 µs | 0.048 µs (rejected at the root) | 0.045 µs |
| `mesh_distance`, touching, all 64 poses | 6.1 ms | 1.7 ms | 3.5 ms | 2.3 ms |
| — intersecting poses only | 0.42 ms | 0.39 ms | 0.86 ms | 1.32 ms |
| — near-miss poses only | 17.0 ms | 4.2 ms | 38 ms (3 poses) | 22 ms (3 poses) |
| `mesh_distance`, apart 0.3 | 4.8 ms | 1.2 ms | 34 ms | 12.5 ms |
| `triangles_near`, margin 0.080 | 7.9 ms | 2.3 ms, then 2.2 ms | 43 ms | 23.5 ms |
| `intersecting_pairs`, pushed 0.5 (unchanged) | 11.6 ms | 10.7 ms | 74 ms | 154 ms (intervals 29 to 265 ms: noise) |

"Before" is the code before the per-pair lower bound (commit `0b2d684`); "after" adds it, and for `triangles_near` also the early-exit pair test (the second 1M figure, from interleaved runs: 2.41 ms without it and 2.16 ms with it). The 1M pair comes from two runs on 2026-10-17 a few minutes apart, in which the unchanged `intersects` measured the same. The 10M pair comes from two later runs in which the machine was slower (1M `intersects` at 0.84 and 0.45 ms in them; the 1M results of that pair: near misses 27.0 → 7.3 ms, apart 7.6 → 2.1 ms, `triangles_near` 12.0 → 3.4 ms, intersecting poses 0.70 → 0.70 ms). Earlier, before this change, a run on 2026-10-17 had given 9.9 ms touching (23 ms before the intersection pre-pass) and 7.5 ms apart at 1M.

**Touching is bimodal.** `mesh_distance` takes about 0.4 ms at 1M on the poses that intersect: it runs the exact `intersects` traversal first and stops at the first intersecting pair. On the near misses it falls through to the full distance search, which took about 17 ms before the lower bound and about 4 ms after, roughly three times the apart 0.3 cost (1.2 ms), not "about the apart cost": with a gap of microns, many leaf box pairs touch (key 0), and every triangle pair in them used to get the full 15-candidate triangle distance plus the exact intersection test. Now each triangle pair first gets a cheap lower bound (the larger of the gap between the two triangles' boxes and the gap along each triangle's normal), and is skipped when that bound is beyond the best distance so far, by the same slack rule as box pruning; the results are unchanged.

The cost of `intersects` grows little with size, and `triangles_near` and `intersecting_pairs` follow the contact size: the 10M poses at 0° have 10× the contact of the 1M ones. 10M results have criterion intervals of 2× to 6× around the median, because the contact varies so much across the cycle; read them as orders of magnitude.

### Near-miss search (issue #22)

Same benchmarks, run on 2026-10-17, "before" from a worktree of `main` (commit `b7fb26b`) and "after" from this change, back to back:

| | 1M, before | 1M, after | 10M, before | 10M, after |
| --- | --- | --- | --- | --- |
| `intersects`, touching (unchanged code) | 0.46 ms | 0.46 ms | 0.96 ms, 1.06 ms | 0.98 ms |
| `mesh_distance`, touching, all 64 poses | 1.96 ms | 0.98 ms | 1.46 ms, 1.49 ms | 1.26 ms |
| — intersecting poses only | 0.41 ms | 0.39 ms | 0.87 ms, 0.87 ms | 0.95 ms |
| — near-miss poses only | 4.41 ms | 2.09 ms | 15.7 ms, 15.8 ms (3 poses) | 9.6 ms (3 poses) |
| `mesh_distance`, apart 0.3 | 1.31 ms | 0.85 ms | 10.2 ms, 10.4 ms | 6.8 ms |
| `triangles_near`, margin 0.080 | 2.23 ms | 1.55 ms | 15.0 ms, 15.9 ms | 10.6 ms |

The 1M pair is the first of two rounds, in which the unchanged `intersects` measured the same on both sides; in the second round the machine was slower for the "before" run (`intersects` 0.50 ms, near misses 7.2 ms, intersecting poses 0.68 ms) and the "after" numbers repeated within 3% (near misses 2.12 ms, apart 0.85 ms, `triangles_near` 1.44 ms). The 10M "before" column has two runs, one before and one after the "after" run. Intersecting poses run the same code before and after (the exact intersection pre-pass), so their 10M difference is noise.

What changed, with results unchanged:
- **Per-leaf-pair caching.** The per-pair lower bound used to rebuild both triangles in `f64`, their boxes and their normals for every triangle pair. Now each triangle of a leaf pair is built once, with its box and its slab along its normal, and kept while the traversal stays on the same leaf. The bounds are bit-identical.
- **Leaf prefilter.** Each triangle's box is first tested against the union box of the other leaf's triangles, and skipped against the whole leaf when that gap is beyond the bound. Its gap is never larger than the per-pair box gap, so it skips only pairs the lower bound would skip.
- **Tie order.** `mesh_distance` orders pairs by box gap plus `1e-12` times the distance between the box centres, and still prunes on the box gap alone. On a near miss most leaf box pairs touch (gap 0), so the order among them used to be left-child-first. Now the most overlapping pairs come first and the bound falls sooner. Without this, the near misses took 2.9 to 4.6 ms at 1M instead of 2.1 ms.
- **The pre-pass stays.** The alternative was to drop the exact intersection pre-pass and let the distance search stop at its first pair at distance 0 (the triangle distance already runs the exact test). In two runs, near misses took 1.5 and 2.3 ms that way, against 2.1 ms with the pre-pass. But intersecting poses went from 0.4 ms to 1.5 and 2.2 ms. So the pre-pass is kept.

On the 22 near-miss poses at 1M (single cold runs), the pre-pass costs 0.17 to 0.37 ms on most poses, about 1 to 1.5 ms on four, and 5 ms on one. The whole query visits about 350 to 1,300 leaf pairs on most poses (pre-pass included), about 2,000 on three, and 7,400 on the slowest.

**Box prefilter in the intersection traversal.** A follow-up commit applies the same per-leaf caching to the traversal behind `intersects`, `intersecting_pairs` and the pre-pass. It also skips a pair when the two triangles' boxes are disjoint: first a B triangle against the union box of A's leaf, then pair by pair. The skip is exact. The boxes are the min and max of the same `f64` corners the exact test gets, and the test is strict, so pairs whose boxes only touch are kept. Results are unchanged. 1M, two rounds back to back on 2026-10-18, before (commit `abe83bc`) and after; the second round repeated the first within 2%:

| 1M | before | after |
| --- | --- | --- |
| `intersects`, touching | 0.436 ms | 0.291 ms |
| `mesh_distance`, touching, all 64 poses | 0.94 ms | 0.80 ms |
| — intersecting poses only | 0.379 ms | 0.251 ms |
| — near-miss poses only | 2.01 ms | 1.84 ms |
| `mesh_distance`, apart 0.3 (does not use this path) | 0.82 ms | 0.83 ms |
| `triangles_near`, margin 0.080 (does not use this path) | 1.37 ms | 1.38 ms |
| `intersecting_pairs`, pushed 0.5 | 10.4 ms | 4.4 ms |

**Against the spec's targets (§1).**
- `intersects` takes about 0.3 ms at 1M vs 1M: within the "few ms" target.
- `mesh_distance` takes about 0.25 ms at 1M when the meshes intersect, about 0.85 ms when they are 0.3 apart, and about 1.8 to 2.1 ms on the near misses (9.6 ms at 10M, measured before the box prefilter). All meet the "few ms" target at 1M. Near misses miss the issue's goal of about 1 ms by about 1.8×, and 10M near misses miss the "few ms" target.
- `triangles_near` costs in proportion to contact: about 1.4 to 1.5 ms at 1M for a few hundred to 4,600 triangles per side, and about 11 ms at 10M for 350 to 48,000.
- `intersecting_pairs` with B pushed 0.5 into A takes about 4.4 ms at 1M for 900 to 18,500 pairs.

## Scene (fast-mesh-scene)

Results for `cargo bench -p fast-mesh-scene`. The first group compares the scene's world queries with calling Phase 3 directly on the same meshes and BVHs; the second times `find_contacts`.

- **Machine:** Intel Core i5-13600K, 32 GB RAM, Windows 11, single thread
- **Toolchain:** Rust 1.99.0, release profile (LTO, `codegen-units = 1`)
- **Date:** 2026-10-18
- **Meshes:** `terrain(708)` (1,002,528 triangles) for the overhead group; `terrain(32)` (2,048 triangles) for `find_contacts`. Default options, Eager mode.
- **Rays:** as in the query benchmarks (100,000 rays per iteration from `z = 20`, directions `(jx, jy, -1)` with jitter in ±0.2); the table divides the criterion median by the batch size.
- **Poses:** a flipped copy of the terrain over the terrain, turned by 0.1° more at every step (64 poses) and lifted to just touch, as in the mesh-vs-mesh queries. The direct calls take the pose as a `Rigid`; the scene takes it as B's placement (A sits at the identity), which gives the same relative transform.

### Overhead against direct calls (two 1M meshes)

| | Direct (Phase 3) | Scene | Overhead |
| --- | --- | --- | --- |
| `raycast`, per ray (scene of 1 mesh at the identity) | 1.46 µs | 1.76 µs | +0.30 µs (+20%) |
| `raycast`, per ray (scene of 2 meshes, B flipped above A) | | 1.90 µs | |
| `mesh_distance`, one fixed pose | 213 µs | 227 µs | +14 µs (+7%) |
| `mesh_distance`, all 64 poses per iteration (per query) | 83.98 ms (1.312 ms) | | |
| `set_placement` + `mesh_distance`, all 64 poses per iteration (per query) | | 84.52 ms (1.321 ms) | +0.6% |
| `set_placement` alone (per call) | | 0.15 µs | |

The two 64-pose benchmarks run all 64 poses in each iteration, so every sample costs the same and the pair is a clean comparison. The raycast rows are from the first run (direct 143 to 153 ms per batch of 100,000, scene of 1 mesh 169 to 182 ms); the pair rows were re-run later on the same day, when the machine was faster: an earlier run of the fixed-pose pair measured 246 µs direct and 249 µs scene, so the one-pose overhead lies between +1% and +7%.

- **`set_placement` meets the target, and `mesh_distance` meets it over the poses.** The spec's target (§13) is below 5% overhead, and it is not a merge gate. A pair query costs the scene one relative transform per call and a placement change touches no BVH. The fixed-pose rows differ by 1% to 7% between runs and the 64-pose pair by 0.6%.
- **`raycast` does not meet it on a 1 µs query.** The scene adds about 0.3 µs per ray. Its work per ray is fixed (normalising the direction, mapping the ray into each mesh's local space in `f64`, the padded world-box test, a small ordering vector, and mapping the hit back), so it shows on a query this short; it was not profiled further. A second mesh adds about 0.14 µs.

### `find_contacts`

Meshes of 2,048 triangles in a row, each moved 30 units along x, so every mesh overlaps its neighbours and nothing else (side 32). Margin 0.5, no filter.

| Meshes | Pairs tested | Contacts | Triangles listed (A + B) | `find_contacts` |
| --- | --- | --- | --- | --- |
| 2 | 1 | 1 | 29 + 29 | 51.9 µs |
| 8 | 28 | 7 | 203 + 203 | 517 µs |
| 32 | 496 | 31 | 899 + 899 | 2.27 ms |

The time follows the number of contacts (about 52 to 74 µs each) rather than the number of pairs tested by the broadphase, which grows much faster: 496 pairs at 32 meshes cost no more per contact than 1 pair at 2.

A re-run on the same machine on the same day (when the 1M rows below were added) measured 47.3 µs, 332 µs and 1.50 ms for 2, 8 and 32 meshes, with the same contacts and triangle lists. `find_contacts` did not change between the runs, so the gap is how much the machine varies between runs. Compare rows only within one run.

### `find_contacts` on two touching 1M meshes

This is the owner's per-frame case: the two `terrain(708)` meshes of the overhead group, B flipped above A and lifted to just touch (the 64 poses above), with margin 0.05 and no filter. Measured on 2026-10-18, on a later run than the overhead table.

| | Contacts | Triangles listed (A + B) | Time |
| --- | --- | --- | --- |
| `find_contacts`, the fixed pose (31) | 1 | 63 + 63 | 625 µs |
| `set_placement` + `find_contacts`, all 64 poses per iteration (per pose) | 64 (1 per pose) | 12,201 + 12,201 (about 191 + 191 per pose) | 105.9 ms (1.65 ms) |

Over the poses, a frame's `find_contacts` costs about 1.25 times a `mesh_distance` (1.32 ms per pose in the overhead table). It does more than `mesh_distance`: it finds every triangle pair within the margin, not only the closest one.

## Fields (fast-mesh-fields, fast-mesh-scene)

Results for `cargo bench -p fast-mesh-fields` and the `scene fields` group of `cargo bench -p fast-mesh-scene`, measured on 2026-10-19.

- **Machine:** Intel Core i5-13600K, 32 GB RAM, Windows 11, single thread
- **Toolchain:** Rust 1.99.0, release profile (LTO, `codegen-units = 1`)
- **Meshes:** `terrain(708)`, 1,002,528 triangles, default BVH

### Sign data and signed distances

`signed_distances` runs a batch of 100,000 points over the terrain, at heights within ±2 of it; the table divides the criterion median by the batch size. `max_distance` 1 prunes the closest-point search, and infinity computes every point exactly.

| | Pseudonormal | Winding |
| --- | --- | --- |
| `SignData::build`, 1M triangles | 147 ms | 38.0 ms |
| `signed_distances`, per point, `max_distance` 1 | 1.12 µs | 35.9 µs |
| `signed_distances`, per point, `max_distance` ∞ | 2.20 µs | 75.9 µs |

Pseudonormal builds slower (it accumulates angle-weighted normals per vertex and edge) but costs only the closest-point query per point. Winding builds faster (it stores node moments) but each point sums dipole terms over many nodes and exact solid angles over the near leaves: on a terrain, which is flat, the order-1 expansion needs `beta = 8` and therefore many near leaves, which is the likely reason it is about 30 times slower per point (not profiled). That is why winding is opt-in, or the `Auto` fallback for closed meshes with inconsistent orientation, and the default is pseudonormal.

**Every edit of the target rebuilds its sign data.** `SignData::is_current` checks the target mesh's revision, so any edit of mesh B (the `to` mesh of a field) rebuilds its sign data over the whole mesh on the next field update: O(n), about 150 ms at 1M triangles in pseudonormal mode (the table above). A topology edit costs more, because it also drops B's cached adjacency (the incidence index is kept up to date), and the build has to rebuild it. The `sign_build_pseudonormal_cold_adjacency` benchmark builds on a fresh mesh whose incidence index is cached but whose adjacency is not. Two runs on 2026-10-18, on a day when the machine was slower than for the table above:

| 1M, pseudonormal | Run 1 | Run 2 |
| --- | --- | --- |
| `SignData::build`, adjacency cached | 223 ms | 232 ms |
| `SignData::build`, adjacency not cached | 340 ms | 339 ms |

So a topology edit of B costs about 1.5 times the build alone, an estimated ~220 ms at the table's speed (scaled, not measured). Placement changes and edits of A do not rebuild the sign data. An incremental refit is planned for 5b.

### The 64-pose drag of two touching 1M terrains

The two `terrain(708)` meshes of the overhead group, B flipped above A and lifted to just touch (the 64 poses above), both meshes Lazy. A distance field from A to B with `max_distance` 0.08 (the `triangles_near` margin above), coloured from contact (`range = (-0.08, 0)`, `below = Clamp`, `above = Base`). Each pose runs `set_placement`, `update_fields`, `colors` for each coloured mesh and `take_changes`, as a renderer would. "Both sides" adds a second field from B to A and colours B too.

| | One side (field on A) | Both sides (fields on A and B) |
| --- | --- | --- |
| Candidates recomputed per pose, average (largest) | 219 (2,061) | 438 (4,122) |
| Vertices recoloured per pose, average (largest) | 116 (1,579) | 232 (3,158) |
| Time per iteration of 64 poses | 114.7 ms | 231.8 ms |
| Time per pose | 1.79 ms | 3.62 ms |

The candidate counts come from a counter that is not in the code (the one in the unit tests, made visible for a single run), so the benchmark itself prints only the recoloured counts. The largest counts are at pose 0, where the contact is widest (4,607 triangles of each mesh within 0.080); at the later poses it is a few hundred vertices. The time per pose is within the target of the design (a few ms per pose); the work per pose is not set by the number of vertices recomputed (a few hundred take well under 1 ms at 1 to 2 µs each), so the pair test that finds the candidates is likely the larger part, as `triangles_near` at this margin takes about 1.4 ms per pose (1.37 to 1.38 ms above, measured on another run). That split was not profiled. The two-sided case costs twice the one-sided one, since each side repeats the whole update.

## Sections (fast-mesh-query, fast-mesh-scene)

Results for the `query slice` groups of `cargo bench -p fast-mesh-query --bench query -- slice` and the `scene sections` group of `cargo bench -p fast-mesh-scene --bench scene -- sections`, measured on 2026-10-18.

- **Machine:** Intel Core i5-13600K, 32 GB RAM, Windows 11, single thread
- **Toolchain:** Rust 1.99.0, release profile (LTO, `codegen-units = 1`)
- **Meshes:** `terrain(708)` (1,002,528 triangles) and `terrain(2237)` (10,008,338, with `FAST_MESH_BENCH_LARGE=1` and 10 samples); a UV sphere of radius 100 with 500 rings of 1,000 segments (998,000 triangles, outward wound, closed). Default BVH options, Eager mode in the scene.
- **Planes:** on the terrain, `x = const` at 20%, 50% and 80% of the side (plus 0.37, between grid lines); on the sphere, tilted by `(0.1, 0.05, 1)` at `z` = -50.5, 0.3 and 60.7.

### `slice`

Time per call, reusing one `Slice`. Target: about 1 ms for a 1M slice.

| Mesh | Plane 1 | Plane 2 | Plane 3 | Cut |
| --- | --- | --- | --- | --- |
| `terrain_1M` | 128 µs | 131 µs | 114 µs | 1 open contour, 1,417 points |
| `sphere_1M` | 240 µs | 268 µs | 226 µs | 1 closed contour, 2,070 to 2,072 points |
| `terrain_10M` | 586 µs | 615 µs | 798 µs | 1 open contour, 4,475 points |

All are under the target and nothing was tuned. The cost grows with the cut, not the mesh: ten times the triangles gives about 5 times the time on the terrain, and the cut has about 3 times the points. `slice` allocates its per-call temporaries; the drag below does not show whether that matters.

### In the scene

Eager meshes, one section, a world plane. Criterion medians.

| Benchmark | Time | Notes |
| --- | --- | --- |
| `section_drag_1M` | 16.8 ms per 64 poses (262 µs per pose) | `set_section_plane` and a read per pose; terrain, no fill. Includes the mapping to world and `(u, v)` and the length. |
| `section_far_edit_1M` | 21.1 µs | Move 100 vertices about 354 from a plane at the middle; the entry stays Current. |
| `section_far_edit_1M` baseline | 14.2 µs | The same edit, no section. The section costs about 7 µs. |
| `section_fill_1M_sphere`, fill on | 0.560 ms (first run: 4.62 ms) | `set_section_plane` plus the recompute; fill of one 2,072-point loop gives 2,070 triangles, `fill_ok` true. |
| `section_fill_1M_sphere`, fill off | 0.342 ms (first run: 0.341 ms) | The same without the fill: the fill costs about 0.22 ms (first run: about 4.3 ms). |
| `section_gap_two_1M`, `section_gap` alone | 0.153 ms (first run: 2.55 ms) | Two spheres 230 apart (30 between them), one loop each of 2,072 points. |
| `section_gap_two_1M`, plane change then `section_gap` | 0.900 ms (first run: 3.27 ms) | Both entries recomputed (about 0.7 ms), then the gap. |

The first run's fill and gap costs were found and fixed after review; the other rows are from the first run. Before and after, run back to back on 2026-10-18 (`cargo bench -p fast-mesh-scene --bench scene -- "section_fill|section_gap"`, before at commit `79909e7`):

| Benchmark | Before | After |
| --- | --- | --- |
| `section_fill_1M_sphere`, fill on | 4.47 ms | 0.560 ms |
| `section_fill_1M_sphere`, fill off | 0.348 ms | 0.342 ms |
| `section_gap_two_1M`, `section_gap` alone | 2.39 ms | 0.153 ms |
| `section_gap_two_1M`, plane change then `section_gap` | 3.23 ms | 0.900 ms |

- **Fill.** The ear test of `triangulate` walked the whole ring for every candidate ear, so a loop of `n` points cost `O(n²)`: about 0.23 ms at 500 points, 0.92 ms at 1,000, 4.0 ms at 2,072 and 16 ms at 4,144, four times per doubling (measured during the final review with a separate harness, not by the benches above). Above 80 points the ring is now indexed by z-order (Morton) code, and the ear test looks only at the nodes whose codes lie between those of the candidate triangle's box corners, as Mapbox's earcut does. It tests the same nodes, so the triangles are unchanged. The fill now costs about 0.22 ms on the 2,072-point loop, including the nesting and the area check.
- **Gap.** `contour_gap` took the segments of the first side in contour order, starting from a pair of first segments, so its cost depended on where that contour started: about 2.5 ms when it started on the far side, 0.13 ms when it started at the closest point (measured during the final review with a separate harness). The segments are now taken nearest the other side's bounding box first, and the search stops at the first whose box gap reaches the best distance found. The result is the same pair.

## Repair (fast-mesh-ops)

Results for `cargo bench -p fast-mesh-ops --bench ops`, measured on 2026-10-18.

- **Machine:** Intel Core i5-13600K, 32 GB RAM, Windows 11, single thread
- **Toolchain:** Rust 1.99.0, release profile (LTO, `codegen-units = 1`)
- **Meshes:** `terrain(708)` (1,002,528 triangles, 502,681 vertices), with the damage described per row. 10 samples per benchmark.
- **Timing:** `analyze` reads the mesh. The writing operations run a transaction (`begin_edit`, the operation, `commit`) on a mesh rebuilt from buffers for every iteration; the rebuild is outside the timing and the commit is inside it. The times are the middle value of criterion's interval (its headline estimate of the mean time per iteration), not medians.

| Benchmark | Time | Input and result |
| --- | --- | --- |
| `analyze`, no self-intersection check | 478 ms | One component, one boundary loop (the rim). |
| `analyze`, with self-intersection check | 893 ms | The same, with a prebuilt BVH; `self_intersects` is `Some(false)`. The check costs about 415 ms. |
| `orient_1M_1pct_flipped` | 291 ms | Every 100th triangle flipped (10,026); one patch, all 10,026 flipped back. |
| `fill_10_holes_1M`, `fair` off | 360 ms | Ten round holes punched in the terrain (loops of 61, 78, 128, 165, 232, 313, 388, 436, 454 and 482 edges); all ten filled, 83,813 triangles and 40,548 vertices added. The 2,832-edge rim is skipped by `max_edges = 1000`. |
| `fill_10_holes_1M`, `fair` on | 5.16 s | The same fill with fairing: about 4.8 s more than without. |
| `weld_1M_10pct_dupes` | 86 ms | Every 10th vertex duplicated (50,269 duplicates, 552,950 vertices); each corner on such a vertex takes the duplicate with probability 1/2. Tolerance 0 merges all 50,269; no triangle collapses. |
| `remove_small_components_1M` | 117 ms | The terrain plus 1,000 separate two-triangle quads, `min_triangles = 10`: 1,000 components, 2,000 triangles and 4,000 vertices removed. |

No target was set for these (the plan asks only that they be recorded), and nothing was tuned.

- **Fairing dominates a fill.** Fairing is about 93% of the 5.16 s, for 40,548 new vertices. Refinement to the surrounding density makes the patches large (on this terrain the edges are about 1 long, so a hole of 482 edges gets thousands of triangles), and each patch needs a conjugate-gradient solve. `fair` defaults to true, so a default fill of this input costs about 5 s. A fill that needs only a closed surface can set `fair: false` and costs 360 ms for the same ten holes, with a flatter patch (see the accuracy below). The fairing cost grows with the square of the patch size (conjugate gradient on the bi-Laplacian), so one large hole can take seconds on its own.
- **Accuracy of fairing.** On a unit UV sphere (64 rings, 128 segments) with holes of 10, 20 and 30 degrees, the largest `| |p| - 1 |` over the new vertices is 0.0005, 0.0018 and 0.0082 with fairing, and 0.0103, 0.0507 and 0.1182 without (the `unfaired_patch_is_flatter` test, printed once).
- **Weights.** All ten holes of the fair-on fill used cotangent weights and converged; none needed the uniform retry (printed once by the bench).

## JavaScript core (fast-mesh-wasm)

Results for `pnpm build && pnpm --filter fast-mesh-wasm bench -- --10m`, a plain Node script (`packages/fast-mesh-wasm/bench/scene.bench.mjs`, `performance.now()`, medians over the iterations shown). It drives the `Scene` class of the built package, so everything crosses the wasm boundary as a user's code would.

- **Machine:** Intel Core i5-13600K, 32 GB RAM, Windows 11, Node 24.13.0, release wasm build (`wasm-opt -O3`), single thread
- **Date:** 2026-10-18. Two full runs; the tables show run 1, and run 2's numbers are given where they differ. Another branch's parallel-processing work shared the machine and may have been using the CPU in either run, so treat the numbers as upper bounds.
- **Meshes:** height fields like the Rust benches' (`terrain(n)`, here with a different height function): 1,002,528 triangles (`n = 708`) and 9,999,392 (`n = 2236`), given as `Float64Array` world positions and `Uint32Array` indices.
- **A frame** is what an adapter does on the JS side without a GPU: `drag.move(placement)`, `scene.commitFrame()`, and copying every dirty range of every changed buffer from fresh views into a JS mirror (the three.js adapter's job, spec section 11 item 2). The mirror is allocated once, from the first frame.

### Adding a mesh

| | 1M triangles | 10M triangles |
| --- | --- | --- |
| `addMesh`, eager (BVH, normals, incidence index, render buffers) | 1.09 s median of 7 runs, p95 1.9 s (run 2: 1.18 s) | 12.7 s median of 3 runs (run 2: 13.1 s) |
| Two meshes added to a scene | 2.2 s (run 2: 2.4 s) | 26.8 s (run 2: 24.7 s) |

`addMesh` includes the BVH build, which takes 0.44 s at 1M and 5.2 s at 10M on its own (BVH table above); the rest is the store, normals, incidence index and render buffers, and the copy of the input into wasm memory. This was not broken down further.

### The 64-pose drag of two touching 1M and 10M meshes, colouring on

B is flipped over A, lifted until the surfaces touch at pose 0 (measured with `meshDistance`), and turned by 0.1° more at each of 64 poses, as in the Rust drag benchmark above. A distance field on A to B colours A (`range = [-0.08, 0]`, `below = "clamp"`, `above = "base"`). Meshes are in the default eager mode. "Contact" has `maxDistance` 0.08, as the Rust benchmark does. "Wide" has `maxDistance` 8, which puts nearly every vertex of the pair in range and is a stress case, not an expected one. The 64 poses are repeated 3 times after a warm-up cycle (once at 10M).

| Per frame | 1M, contact | 1M, wide | 10M, contact |
| --- | --- | --- | --- |
| Total, median (p95) | 1.76 ms (4.0 ms) | 448 ms (753 ms) | 38.6 ms (72.5 ms) |
| `drag.move` (the engine's eager work happens here) | 1.75 ms | 448 ms | 38.4 ms |
| `commitFrame()` | 0.01 ms | 0.06 ms | 0.07 ms |
| Copy of the dirty ranges into the mirror | 0.01 ms | 0.02 ms | 0.05 ms |
| Colour slots dirty, median | 151 | 6,933 | 3,525 |
| Setup: `addDistanceField` + `setColoring` | 0.32 s | 0.21 s | 2.3 s |
| First `commitFrame()` plus full mirror copy | 4.4 ms | 3.7 ms | 44.9 ms |

Run 2 measured the contact frame at 1.95 ms (1M) and 30.5 ms (10M) and the wide frame at 425 ms. The 10M frame time spreads widely because the contact varies a lot over the 64 poses (minimum 1.2 ms, maximum 108 ms in run 1).

- **The binding is not the cost.** `commitFrame()` plus the range copies take 0.02 to 0.12 ms per frame, because only the recoloured vertices are dirty (under 1 KB of colours per frame at contact, about 30 KB wide). The time is the engine's own, in `drag.move`, and the 1M contact frame (1.76 ms) is close to the Rust benchmark's 1.79 ms per pose for a one-sided field, though the terrains are not the same.
- **The wide case is the field recompute.** With nearly every vertex in range, each pose recomputes distances for much of A; this was not profiled further. An app that wants a whole-surface colour map should expect this cost, and one that wants contact colouring should keep `maxDistance` small.

### The owner's scene: 6 meshes of 30,258 triangles, 30 distance fields

Six `terrain(123)` tiles in a 3 by 2 grid, a distance field from every mesh to every other (30 fields, all kept up to date), each mesh coloured by the field to its next neighbour. One mesh is dragged along a small circle with a slight turn, 400 frames after 20 warm-up frames. Adding the 30 fields and colouring takes 0.13 s.

| Per frame | Contact (tile gap 0.5, `maxDistance` 1) | Wide (tile gap 2, `maxDistance` 10) |
| --- | --- | --- |
| Total, median (p95) | 1.02 ms (2.2 ms) | 105 ms (134 ms) |
| `drag.move` | 1.00 ms | 105 ms |
| `commitFrame()` + copy | 0.02 ms | 0.07 ms |

Run 2: 1.01 ms (2.18 ms) and 112 ms (181 ms). In the contact case the whole frame uses about 6% of a 60 fps frame's 16.7 ms at the median. The wide case recomputes nearly every vertex of the moved tile and its neighbours on each move, and is not a frame-rate workload.

### Size of the wasm binary

`dist/wasm/fast_mesh_wasm_bg.wasm` after `wasm-opt -O3`, gzip measured with `gzip -9`:

| | Raw | gzip |
| --- | --- | --- |
| Before this phase (commit `eff7e80`: `computeVertexNormals` and `computeBounds` only) | 19,561 B (19.1 KiB) | 9,003 B |
| After this phase (the whole engine behind `WasmScene`) | 733,183 B (716 KiB) | 292,300 B (285 KiB) |

The binary grew 37 times because it now carries the BVH, queries, fields, sections and repair, which were not in the package before. The spec's size rule (section 8: call out a release that grows by more than 20%) applies from the next release on. The test build (`pnpm --filter fast-mesh-wasm test`) adds the `__test_panic` export and is not what is published.

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
cargo bench -p fast-mesh-scene                        # scene overhead and find_contacts on two 1M meshes, find_contacts over 2-32 small meshes, and the field drag
cargo bench -p fast-mesh-fields                       # SignData::build and signed_distances on 1M triangles
cargo bench -p fast-mesh-query --bench query -- slice   # slice; add FAST_MESH_BENCH_LARGE=1 for the 10M terrain
cargo bench -p fast-mesh-scene --bench scene -- sections   # section drag, far edit, fill and gap
pnpm build && pnpm --filter fast-mesh-wasm bench -- --10m   # the JavaScript core: addMesh, the 64-pose drag (1M and 10M) and the 6-mesh scene; add --quick for a short run (release build needed; 10M needs a few GB of RAM)
cargo bench -p fast-mesh-ops                          # analyze, orient, fill_holes (fair on and off), weld and remove_small_components on 1M triangles
```
