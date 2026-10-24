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

### The three.js adapter: the same drag with `bindMesh` on both meshes

`bindMesh` is attached to both 1M meshes (A, coloured, and the dragged B) and the benchmark calls `drag.move(pose)` then `scene.commitFrame()`; the adapters' listeners copy the dirty ranges into the `BufferAttribute` mirrors and set the update ranges inside `commitFrame()`. It runs in Node with `three` and no GL context, so the GPU upload is not measured. Scene, poses and `maxDistance` are those of the table above (`pnpm --filter fast-mesh-wasm bench -- --adapter` runs just this case). 3 cycles of 64 poses after a warm-up cycle, run after the adapter started converting colours from sRGB to linear per element; another test job may have shared the CPU, which mostly affects the engine's `move` time.

| Per frame | 1M, contact | 1M, wide |
| --- | --- | --- |
| Total, median (p95) | 2.89 ms (6.25 ms) | 385 ms (948 ms) |
| `drag.move` | 2.82 ms | 385 ms |
| `commitFrame()` with both adapters applying | 0.06 ms (p95 0.15) | 0.20 ms (p95 0.42) |
| `bindMesh` x2 + first `commitFrame()` (builds both geometries) | 141 ms | 146 ms |

This table comes from a later run on the same machine than the header table of this section (the adapter case was added after it). Before the colour conversion, the same case measured 1.85 ms and 417 ms in total, `commitFrame()` 0.03 and 0.10 ms, and setup 91 and 64 ms; the conversion roughly doubles the adapter's share (0.06 and 0.20 ms) and the setup. The adapter still adds little next to the plain mirror copy of the core table (0.02 to 0.12 ms): the cost of a frame is the engine's work in `drag.move`. The adapter's bounding-box walk runs only when positions or indices change, so it does not appear in a placement-only drag. A GPU upload of the same ranges (the renderer uploading each update range: `bufferSubData` in WebGL, `writeBuffer` in WebGPU) comes on top and was not measured.

### Size of the wasm binary

`dist/wasm/fast_mesh_wasm_bg.wasm` after `wasm-opt -O3`, gzip measured with `gzip -9`:

| | Raw | gzip |
| --- | --- | --- |
| Before this phase (commit `eff7e80`: `computeVertexNormals` and `computeBounds` only) | 19,561 B (19.1 KiB) | 9,003 B |
| After this phase (the whole engine behind `WasmScene`) | 733,183 B (716 KiB) | 292,300 B (285 KiB) |

The binary grew 37 times because it now carries the BVH, queries, fields, sections and repair, which were not in the package before. The spec's size rule (section 8: call out a release that grows by more than 20%) applies from the next release on. The test build (`pnpm --filter fast-mesh-wasm test`) adds the `__test_panic` export and is not what is published.

## Worker scene (fast-mesh-wasm/worker)

Results for `pnpm build && pnpm --filter fast-mesh-wasm bench -- --worker --10m`. The `WorkerScene` from the built package (`dist/worker.js`) drives the engine in a Node `worker_threads` Worker (`bench/worker-harness.mjs`), which serves it as the browser Worker script does, with the default 4 ms yield budget. Messages are structured clones with transferred `ArrayBuffer`s, as in a browser, but this is Node's Worker and event loop, not a browser's.

- **Machine:** Intel Core i5-13600K (14 cores, 20 threads), 32 GB RAM, Windows 11, Node 24.13.0
- **Wasm:** release build (`wasm-opt -O3`) **without** the `parallel` feature: the Worker runs the engine on one thread, as the published npm package does.
- **Date:** 2026-10-19. Two full runs, back to back, with no other `cargo`, `rustc` or bench process running (CPU load 3% before them). The tables show run 1; run 2 is given in parentheses where it differs by more than about 10% or where only one sample was taken.
- **Meshes:** the `terrain(n)` height fields of the JavaScript core section: 1,002,528 triangles (`n = 708`) and 9,999,392 (`n = 2236`).
- **What is timed.** On the main thread: the synchronous calls; the scene's message handler applying a result to the mirrors ("handler", timed around the scene's listener); the adapter-style copy of the delivered dirty ranges into a JS mirror (as in the core section); and the longest event-loop block, from a `setImmediate` probe loop, which also covers the structured-clone of each message. In the Worker: each chunk of work the engine reports (a request, a field build or a repair operation, timed from the end of the previous one), and the longest block of the Worker's event loop, from the same kind of probe. Requests are sent with `commitFrame()` and the bench waits for their result before the next step (lock step), so a round trip is one batch's whole cost.

### Adding a mesh: main-thread blocks

`scene.addMesh(positions, indices)`, then `commitFrame()` sends it, the result comes back with every buffer whole, and the next `commitFrame()` delivers the `added` record, whose whole buffers the bench copies into its mirror (what an adapter does once for a new mesh). 1M: median of 7 runs; 10M: 1 run per run.

| | 1M triangles | 10M triangles |
| --- | --- | --- |
| **Longest main-thread block** (probe), whole add | **9.30 ms** (max 9.83; run 2: max 12.8) | **63.5 ms** (run 2: 63.2) |
| `addMesh()` call (copies the inputs, scans for the box centre) | 9.25 ms | 63.4 ms (run 2: 62.9) |
| `commitFrame()` that posts it (buffers transferred) | 0.03 ms | 0.06 ms (run 2: 0.30) |
| Result handler (whole buffers transferred into the mirrors) | 0.08 ms (max 0.32) | 0.11 ms (run 2: 0.07) |
| The app's full copy of the new mesh's buffers (next `commitFrame()`) | 4.0 ms | 52.8 ms (run 2: 34.3) |
| Worker: the `addMesh` request (one Rust call) | 1.13 s | 12.7 s (run 2: 12.2) |
| Round trip, `addMesh` to result applied | 1.14 s | 12.7 s (run 2: 12.2) |

- **The spec's target (§6: under 16 ms per message) is met at 1M and missed at 10M, but not by a message.** No message handler takes more than 0.32 ms at either size. The longest block is the synchronous `addMesh()` call itself. A separate measurement of its parts with the same operations in plain Node (median of 5) gives, at 1M: `Float64Array.from(positions)` 1.6 ms, `Uint32Array.from(indices)` 2.3 ms, the box-centre scan 4.7 ms; at 10M: 16.8 ms, 16.6 ms and 33.7 ms. Passing `origin` skips the scan (about half of it); the copies are needed because the buffers are transferred.
- **The Worker is busy for the whole add**: 1.1 s at 1M and 12 s at 10M in one Rust call, which cannot be interrupted. A request sent meanwhile (a toggle, a drag) waits behind it; the page stays responsive. (Since issue #48 a mesh of 65,536 triangles or more is built in the background, and such a request no longer waits: see "Chunked long calls".)
- The app's own full copy of a new mesh (4 ms at 1M, 34 to 53 ms at 10M) happens in its `commitFrame()` listener, outside the library; an adapter that uploads a new 10M mesh in one frame pays this whatever the scene runs on.

### Colouring toggle: round trip

Two touching meshes (B flipped over A, as in the 64-pose drag) with a distance field on A to B colouring A. Each toggle calls `setColoring(A, null)` or `setColoring(A, rule)` and `commitFrame()`, then waits for the result. The field is kept, so turning colouring on is a recolour, not a field build (§9.5). 10 toggles each way at 1M, 3 at 10M.

| Round trip, median (p95) | 1M, contact | 1M, wide | 10M, contact |
| --- | --- | --- | --- |
| Colouring off | 0.06 ms (0.26) (run 2: 0.13) | 0.06 ms (0.17) | 0.21 ms (0.34) (run 2: 0.15) |
| Colouring on | 1.00 ms (1.36) | 1.11 ms (1.62) | 9.95 ms (10.9) |
| Worker: the `coloring` request, on | 0.67 ms | 0.80 ms | 6.58 ms |
| Main: result handler / copy of the colour ranges, on | 0.01 / 0.12 ms | 0.01 / 0.12 ms | 0.03 / 1.28 ms |

"Contact" and "wide" are the core section's `maxDistance` 0.08 and 8. Turning colouring on recolours every vertex of A and sends the whole colour buffer (2 MB at 1M, 20 MB at 10M; the handler takes 0.01 to 0.03 ms for it); the time is the Worker's recolour plus the message. Turning it off sends no colour bytes.

### The 64-pose drag of two touching meshes, colouring on

The drag of the core section through the Worker scene: per frame `drag.move(pose)` (optimistic: the placement mirror moves at once), `commitFrame()` (sends the move, delivers the previous result), the copy of the delivered dirty ranges, then waiting for the move's result. 1M contact: 3 cycles of 64 poses after a warm-up cycle; 1M wide and 10M contact: 1 cycle.

| Per frame | 1M, contact | 1M, wide | 10M, contact |
| --- | --- | --- | --- |
| **Main-thread cost** (move + `commitFrame()` + copy + result handler), median (p95) | **0.03 ms** (0.05) | **0.08 ms** (0.11) | **0.12 ms** (0.21) |
| of which the result handler (applying the transferred ranges), median (p95) | 0.01 ms (0.03) | 0.04 ms (0.06) | 0.06 ms (0.10) |
| Dirty bytes per frame, median | 0.6 KB | 27.7 KB | 14.1 KB |
| Worker: the `drag` move request, median | 1.76 ms | 402 ms (run 2: 391) | 30.9 ms |
| Round trip, move to result applied, median (p95) | 1.84 ms (4.06) | 402 ms (723) | 31.4 ms (74.1) |
| Longest Worker block, max | 4.39 ms | 741 ms | 77.1 ms (run 2: 77.8) |

- **The main thread pays almost nothing for a drag**: about 0.1 ms per frame even at 10M, against 1.76 ms (1M) and 38.6 ms (10M) per frame on the main thread with the synchronous `Scene` (core section). The engine's work moved to the Worker unchanged (1.76 ms against 1.75 ms per move at 1M).
- **Colours lag the drag by the round trip.** The mesh follows the pointer at once; its colours catch up one round trip later: about 2 ms at 1M (within the frame), about 31 ms at 10M (two frames at 60 fps), and 0.4 s in the wide stress case. Moves sent meanwhile coalesce in the Worker, so by design a slow move does not build a backlog (the lock-step bench does not exercise this).

### The longest single requests

The Worker blocks for the whole of one request (ruling 10.1-6: no Rust-side chunking in 9a; since issue #48 the field work runs in steps, see "Chunked long calls"). The two cases the spec's chunking list names, on 1M meshes: a field recompute after one edit, and a section update over several meshes. A third mesh C (another 1M tile beside the pair) carries no field. Each edit moves one vertex in the contact area (`edit` with `setPositions`), 10 edits per row; the section rows change the plane 16 times.

| | Longest Worker block, median (p95) |
| --- | --- |
| One-vertex edit of C (no field depends on it) | 0.13 ms (9.06, the first edit) |
| One-vertex edit of A, 1 field (on A, to B) | 0.06 ms (7.17, the first edit) |
| One-vertex edit of A, 2 contact fields (A to B, **B to A**) | **93.6 ms** (105) |
| One-vertex edit of A, 4 fields (also both directions with `maxDistance` 8) | **348 ms** (375) |
| Adding the B-to-A contact field (one field build) | 218 ms (run 2: 203) |
| Adding a wide field (one field build), each of two | 106 and 111 ms |
| Section over 3 meshes of 1M, plane change, fill off | 0.69 ms (1.22) |
| Section over 3 meshes of 1M, plane change, fill on | 0.67 ms (0.78) |
| `addSection` over the 3 meshes, fill off / on | 1.93 / 0.74 ms |

- **One edit can block the Worker for about 0.1 to 0.35 s.** An edit of A is cheap for the field on A (only the moved vertex is recomputed), but every field measured *to* A (here B to A) recomputes after it, and that costs about 94 ms at 1M even for one moved vertex. The size is consistent with rebuilding A's sign data, which every edit of a field's target does ("Sign data and signed distances" above: `SignData::build` takes 147 ms at 1M, pseudonormal); this was not profiled. With both wide fields as well, one edit takes 348 ms. These are single Rust calls (`run_eager_fields`), so a toggle or drag sent meanwhile waits up to that long; the page itself is not blocked. If this matters, the follow-up is Rust-side work: per-field steps for the recompute, or incremental sign data after small edits.
- **Section updates are not a concern at this size**: under 1.3 ms for three 1M meshes, with or without fill.
- For comparison, the other long single requests measured above: the 1M wide drag move (median 0.4 s, up to 0.74 s), adding a 1M mesh (1.1 s) and a 10M mesh (12 s).

### Many small requests in one frame

1000 `setPlacement` calls in one frame, spread over 10 meshes of 5,000 triangles, with no fields. Each is its own undo step, so none is coalesced (ruling 10.1-1). 20 rounds after a warm-up round.

| | Median (p95) |
| --- | --- |
| Main: queueing the 1000 calls | 0.47 ms |
| Round trip, `commitFrame()` to result applied | 3.88 ms (6.65) (run 2: p95 5.12) |
| Main: result handler | 0.07 ms |
| Worker: the 1000 requests, summed | 2.30 ms |
| Longest Worker block | 3.16 ms (max 5.02) (run 2: max 3.88) |
| Synchronous `Scene`, the same 1000 calls + `commitFrame()` | 2.00 ms (8.44) |

The batch runs back to back inside the 4 ms yield budget, so the Worker yields at most once for it; the Worker side costs about the same as the synchronous scene (2.3 ms against 2.0 ms), and the main thread spends about 0.5 ms.

## Store (fast-mesh-wasm/store)

Results for `pnpm build && pnpm --filter fast-mesh-wasm bench:store` (`packages/fast-mesh-wasm/bench/store.bench.mjs`), before and after the incremental snapshot refresh (issue #50).

- **Scene:** a `SceneStore` over the built package holds 1000 small meshes (closed tetrahedra, added through the store). 100 of them are coloured by distance to a neighbour, and there are 10 sections of 100 meshes each. The scene work under each action is tiny, so the numbers are mostly the store's own cost.
- **Local host:** a synchronous `Scene`, used for the first two tables. It has no pending statuses, so its placements and edits change no status keys.
- **Worker host:** the engine in a `worker_threads` Worker (`bench/worker-harness.mjs`), used for the third table. A placement there changes status keys, which the store maps to the states that read them (the dependency index).
- **Machine:** Intel Core i5-13600K, 32 GB RAM, Windows 11, Node 24.13.0, release wasm build.
- **Date:** 2026-10-23. One run each. Other agents were building and benchmarking on the machine, so treat the numbers as upper bounds, the p95 columns most of all.
- **Before:** every action and event re-derived every mesh and section, at 5 to 8 status lookups per mesh and 9 or more per section.
- **After:** only the states that an action touches, or that read a status key that changed, are re-derived. Scene events in one task make one snapshot.

| Local host, per action or event, median (p95), µs | Before | After |
| --- | --- | --- |
| `select(mesh)` (store only) | 313 (442) | 0.60 (0.80) |
| `setDisplay(opacity)` (a change every call) + `store.commitFrame()` | 1033 (1485) | 78.4 (157) |
| `setPlacement` of a coloured mesh in a section | 342 (549) | 13.9 (25.6) |
| `store.commitFrame()`, nothing changed | 675 (875) | 19.8 (33.3) |
| A scene event (`scene.commitFrame()`) + `getSnapshot()` | 354 (617) | 19.3 (26.4) |

| Local host, burst of adds through the store | Before | After |
| --- | --- | --- |
| 1000 adds into an empty store (median of 10 runs) | 116 ms | 26.9 ms |
| 500 more adds onto the 1000 meshes | 203 ms (406 µs per add) | 29.5 ms (59.1 µs per add) |

| Worker host, `setPlacement` of a coloured mesh in a section, 300 round trips, median (p95), µs | Before | After |
| --- | --- | --- |
| The action (its `placement:` key goes `"computing"`) | 514 (830) | 140 (205) |
| A status delivery: the store's work when results arrive (timed across the scene's `onStatus` listeners, up to a snapshot read) | 517 (818) | 55.1 (115) |

**What is left.**
- **The scene call itself.** A local `commitFrame()` costs about 19 µs with nothing to do. The Worker action also includes the client's own work for the call.
- **A copy of the snapshot's map.** Every action that changes a mesh's state still copies the snapshot's mesh map, and likewise the section map for a section. This keeps the old snapshot immutable.
  - The copy is a native `Map` copy, O(meshes), with no status lookups: about 45 µs at 1500 entries, so about 300 µs at 10,000 meshes.
  - `setDisplay` + `commitFrame()` copies the map twice: once for `desired` and once for the applied display.
  - An N-add burst is therefore still O(N²), with a small constant.

## Wide-range drag (drag-speed part A)

A drag where most of the coloured mesh is within the field's `maxDistance`: two touching 1M terrains (B flipped over A) with a distance field on A to B, `maxDistance` 8, colour range 0 to 6. Part A made the per-pose recompute faster (exact, single-threaded) and made the Worker step it and drop a stale pose. Part of #48; the profile is in `docs/superpowers/notes/2026-10-21-wide-drag-profile.md`.

- **Machine:** Intel Core i5-13600K, 32 GB RAM, Windows 11, Rust 1.99.0 release profile, Node 24.13.0, wasm release build without the `parallel` feature.
- **Date:** 2026-10-21 (baseline and the near-search change) and 2026-10-22 (the re-run at the end of part A and the Worker tables).

### Native, single-threaded

`cargo bench -p fast-mesh-scene --bench scene -- "wide drag"`: the terrain and touching poses of the 64-pose field drag above, 4 of its poses (0, 16, 32, 48, so every pose recolours most of A: 221,381 vertices per pose on average for the field on A). Each pose runs `set_placement`, `update_fields`, `colors` and `take_changes`. Criterion median of 10 samples, divided by the 4 poses.

| Per pose | Before (`f58c958`) | Near search past a limit (`4fbb41d`) | End of part A (`16cf700`) | Speed-up |
| --- | --- | --- | --- | --- |
| Field on A | 13.89 s | 484 ms | **483 ms** | **28.7×** |
| Fields on A and B | 27.77 s | 960 ms | 997 ms | 27.9× |
| Opening a step unit (field on A: `set_placement` and the first one-point `field_step`) | | | 40.1 ms | |

- **The cause** was the near-triangle search: at this range it listed every pair of triangles within 8 units, hundreds of millions per pose, which was 96.6% of the time. The search now gives up past one near pair per vertex slot of A and the update recomputes every vertex instead, with the same values, colours and change records bit for bit.
- **The spec's target (§6.3, at least 3× per pose) is met** with about 29×.
- The end-of-part-A run is about 4% slower than `4fbb41d` for both fields (criterion: +3.9%, p < 0.05) and the same for the field on A; the step API added in between does not run on this path, so this is within run-to-run variation on this machine.
- Opening a step unit costs about 40 ms per pose: the field's preparation and its near search up to the limit, which is one call and cannot be split. The contact drag stays on the near search: 1.74 to 1.48 ms per pose (see the profile note).
- Warm-starting the closest-point query from each vertex's last closest triangle was tried and dropped: within noise (453 against 457 ms per pose), and its exactness was not proven.

### Worker scene

`pnpm build && pnpm --filter fast-mesh-wasm bench -- --worker --10m` (or `-- --worker --wide [--10m]` for these cases only), in the harness of the Worker scene section. The engine recomputes the due fields at the end of each batch in steps of 4,096 vertices, yielding every 4 ms, and at each step boundary drops the recompute when a newer move of either mesh is queued, unless 200 ms have passed since the last completed recompute (the starvation guard), when it finishes it instead; a queued drag end lifts the guard. A pose's colours are applied all at once. Measured after `eb8b625` with the drag-end rule (starvation guard, `setImmediate` yield in the engine), Node 24.13.0, release build. The continuous drag ends the drag right after its last move (the pointer is released). Two terrains:

- **9a terrain:** the pair of the 64-pose Worker drag (the JavaScript `terrain(n)`). Here few vertices change colour per pose (27.7 KB of dirty bytes per frame at 1M).
- **Native terrain:** the terrain and touching poses of the native bench above, built in JavaScript (f64 heights), where every pose recolours most of A.

Two ways to drag:

- **Lock step:** one move to poses 16, 32, 48 and 0 in turn, then wait until its colours are applied (8 moves at 1M, 4 at 10M).
- **Continuous at 60 Hz:** a move every 16.7 ms whatever the Worker is doing, turning B through the 64 poses and back, for 10 s (1M) or 30 s (10M), then the pointer stops.

| | 1M, 9a terrain | 1M, native terrain | 10M, 9a terrain | 10M, native terrain |
| --- | --- | --- | --- | --- |
| **Lock step: move to colours applied**, median (p95) | **70.9 ms** (75.6) | **635 ms** (650) | **631 ms** (664) | **6.68 s** (6.95) |
| Continuous: moves sent / poses recoloured / recomputes dropped | 600 / 41 / 102 | 600 / 12 / 27 | 1800 / 43 / 1 | 1800 / 4 / 1 |
| **Continuous: update interval during the drag**, median (max) | **252 ms** (304) | **851 ms** (916) | **714 ms** (805) | **7.61 s** (7.61) |
| **Continuous: last move, then release, to its colours applied** | **133 ms** | **635 ms** | **1.06 s** | **9.89 s** |
| Continuous: last move to its colours, pointer paused without ending (previous run) | 119 ms | 818 ms | 1.03 s | 12.4 s |
| Continuous: most batches waiting for a result | 7 | 44 | 48 | 477 |
| Continuous: longest Worker block | 102 ms | 129 ms | 677 ms | 1.00 s |
| Main thread per frame, median (longest block) | 0.04 ms (0.51) | 0.04 ms (6.65) | 0.03 ms (0.98) | 0.03 ms (12.5) |

The lock-step Worker blocks of the previous run (the same code without the drag-end rule): longest block per pose 58.4, 62.9, 510 and 644 ms, nearly all of it opening the step unit (the first step).

- **The latest pose wins, with a starvation guard.** During a continuous drag each recompute takes longer than a frame, so a newer move is queued at every step boundary and the recompute is dropped, until 200 ms have passed since the last completed one. The recompute under way then finishes and is shown, and the newest queued move (the older ones are superseded at once) starts the next. So the colours update about every 200 ms plus one recompute (252 ms and 851 ms at 1M), each update the exact colours of one pose. At 10M a recompute alone outlasts the guard, so almost none is dropped (one, by the drag's end) and the colours update once per recompute. When the drag ends, its end drops a recompute kept for an older pose, so the last pose's colours arrive within about one recompute plus the step or unit opening under way: 133 ms and 635 ms at 1M, 1.06 s and 9.89 s at 10M (lock-step recomputes 71 ms, 635 ms, 631 ms and 6.68 s; at 10M on the native terrain that is about 1.5 recomputes, and the extra time is not profiled). A pointer that pauses without ending the drag may wait up to two recomputes, as the kept one finishes first (previous run: 818 ms at 1M and 12.4 s at 10M on the native terrain). The batches waiting for a result are the moves sent during one recompute (up to 477 during a 7 s 10M recompute at 60 Hz); they are answered at once when it ends, and the main thread pays about 0.05 ms per frame.
- **Before the guard** (Task 5, `01dd3d9`), no pose finished during a continuous drag (1 of 600 recoloured at 1M) and the last arrived one recompute after the pointer stopped (81 ms and 683 ms at 1M, 0.96 s and 6.64 s at 10M).
- **Before part A** the same 1M 9a-terrain drag took 402 ms per move in the Worker (one Rust call, up to 741 ms); it now takes 72 ms (the 64-pose lock-step drag of the Worker scene section, re-run: median 71.9 ms, p95 80.0, longest Worker block 75.8 ms). The one-vertex edit of A with 4 fields (the longest single requests) went from 348 to 249 ms.
- **The longest Worker block is opening a unit**: about 60 ms at 1M and 0.5 to 0.65 s at 10M (up to 1.0 s seen during the continuous 10M drag). A request sent meanwhile waits that long; part B's threads shorten it. (Since issue #48 the open is stepped too: about 4 ms, every step under about 15 ms at 1M; see "Chunked long calls".)
- **The engine's yield.** The engine yields with `setImmediate` where there is one (Node) and with a `MessageChannel` tick in browsers. In a Node `worker_threads` Worker a chain of `MessageChannel` ticks after chunks of 1 ms or more lets no `parentPort` message in (measured: none of about 60 messages sent at 60 Hz over 1 s of 1 ms or 4 ms chunks, while `setImmediate` let all of them in), so no recompute could be dropped there. The Task 5 table was measured with a `setImmediate` stand-in in the bench harness; the engine now does this itself and the harness no longer patches it. A browser Worker (headless Chrome 155, both examples' Vite builds) admits a message sent a third of the way into a 227 ms recompute: the recompute is dropped and its result posted 83 ms after it started. The Worker scene tables above this section were measured before the change; the Task 5 re-run with a `setImmediate` tick gave the same numbers within about 10% (1M contact drag 1.90 ms, 10M contact drag 30.9 ms, many small requests 3.97 ms).

## Threaded Worker (drag-speed part B)

The wide-range drag of the section above, in a Worker scene on a cross-origin isolated page, single-threaded (`threads: 1`) against threaded (the default pool, 8 threads on this machine), in the same headless browser. The threaded engine is the same Rust code built with wasm atomics and the `parallel` feature, its pool on Web Workers (`wasm-bindgen-rayon`); its results are bit-identical (`scripts/browser-check.mjs`, check `compare`).

- **Machine:** Intel Core i5-13600K (20 hardware threads, `hardwareConcurrency` 20), 32 GB RAM, Windows 11.
- **Browser:** headless Chrome 155.0.8059.85 over the DevTools protocol, against `vite preview` of `examples/webgpu` (its `check.html` page, with the COOP/COEP middleware).
- **Build:** release `pnpm build` at `9c53ce0` (single-threaded `.wasm` 772 KB, threaded 976 KB; nightly-2026-10-15 for the threaded one).
- **Date:** 2026-10-22.
- **Command:** `node scripts/browser-check.mjs --build --bench --n 708 --kind 9a` (and `--kind native`; at 10M `--n 2236 --lock-step 4 --seconds 30`). Each run measures `threads: 1`, then the default.
- **Same terrains, poses and drags as the Node tables above:** the 9a and native terrains, lock step (one move to poses 16, 32, 48 and 0 in turn, waiting for its colours; 8 moves at 1M, 4 at 10M), and a continuous 60 Hz drag through the 64 poses and back (10 s at 1M, 30 s at 10M) that ends right after its last move. The engine (step size, yield, starvation guard of 200 ms) is unchanged.
- **Worker block** is the longest time the Worker's event loop did not run (a `MessageChannel` probe in the Worker, as the Node harness's), measured per lock-step pose.

| | 1M, 9a terrain | 1M, native terrain | 10M, 9a terrain | 10M, native terrain |
| --- | --- | --- | --- | --- |
| **Longest Worker block per pose**, lock step, median: 1 thread → 8 | 47.3 → **23.5 ms** (2.0×) | 50.9 → **21.0 ms** (2.4×) | 671 → **380 ms** (1.8×) | 825 → **397 ms** (2.1×) |
| Lock step: move to colours applied, median | 60.5 → 37.5 ms (1.6×) | 623 → 140 ms (4.5×) | 907 → 637 ms (1.4×) | 10.26 → 2.97 s (3.5×) |
| **Continuous: update interval during the drag**, median | 252 → **222 ms** | 792 → **428 ms** (1.9×) | 915 → **637 ms** (1.4×) | 10.4 s (one interval) → **2.96 s** (3.5×) |
| **Continuous: last move, then release, to its colours applied** | 69.0 → **34.7 ms** | 611 → **298 ms** | 912 → **615 ms** | 10.05 → **2.71 s** |
| Continuous: moves sent / poses recoloured / units dropped | 600 / 40 / 152 → 600 / 45 / 362 | 600 / 12 / 41 → 600 / 24 / 108 | 1800 / 34 / 1 → 1800 / 49 / 1 | 1800 / 3 / 1 → 1800 / 11 / 1 |
| Continuous: longest Worker block | 62.6 → 31.8 ms | 97.1 → 60.3 ms | 727 → 428 ms | 859 → 411 ms |
| Continuous: most batches waiting for a result | 6 → 3 | 40 → 20 | 61 → 41 | 630 → 193 |
| Main thread per frame, median (longest) | 0.03 (0.11) → 0.03 ms (0.12) | 0.03 (8.06) → 0.04 ms (11.6) | 0.04 (0.42) → 0.03 ms (0.17) | 0.03 (15.2) → 0.03 ms (17.1) |
| Set-up (two meshes, the field, first colours) | 3.08 → 0.99 s | 3.50 → 1.05 s | 49.1 → 39.6 s | 56.4 → 39.6 s |
| WebAssembly memory after set-up | 215 → 258 MB | 215 → 259 MB | 2,122 → 2,390 MB | 2,124 → 2,408 MB |

- **The longest Worker block halves.** It is opening a step unit (the field's preparation and its near search, one call), and 8 threads make it 1.8 to 2.4 times shorter: about 21 to 24 ms at 1M and about 0.4 s at 10M. A request sent during a drag waits at most that long.
- **A recompute is 1.4 to 4.5 times faster.** The gain is largest where most of the mesh is recomputed (native terrain: 4.5× at 1M, 3.5× at 10M). On the 9a terrain few vertices change per pose, the unit open is most of a recompute, and the gain is that of the open (1.6× and 1.4×).
- **The update interval** during a sustained drag is about the 200 ms starvation guard plus one recompute, so at 1M on the 9a terrain it barely moves (252 to 222 ms: the guard dominates), while on the native terrain it nearly halves (792 to 428 ms). At 10M a recompute outlasts the guard, and the colours update once per recompute: 637 ms and 2.96 s. Single-threaded, the 10M native drag recoloured 3 poses in 30 s (one interval between two of them inside the drag).
- **After release** the last pose's colours arrive within about one recompute in every case, threaded and single-threaded.
- **Browser against Node, single-threaded.** At 1M the browser's single-threaded numbers match the Node `worker_threads` table above (lock step 60.5 against 70.9 ms and 623 against 635 ms; interval 252 and 792 against 252 and 851 ms). At 10M the browser is slower: 907 ms against 631 ms and 10.3 s against 6.68 s per lock-step pose. That gap is not profiled; the threaded numbers are compared only with the single-threaded ones of the same browser run.
- **Memory.** The threaded engine's shared memory starts at 18 pages (1.1 MiB, about 18 MB once the pool's thread stacks are allocated) and may grow to 65,536 pages (4 GiB). It held about 12 to 20% more after set-up than the single-threaded engine here (the pool's per-thread state and allocation); 10M fits.
- **Indicative, other browsers** (Task 3's check, the 1M 9a lock step only, longest Worker block per pose, 1 thread → 8): Firefox 157 63.4 → 22.4 ms, Playwright WebKit 27.2 on Windows 49.5 → 21.8 ms. These were not re-run here.

## Chunked long calls (issue #48)

The Worker used to block for the whole of one engine call. Now the field work, the sign data and the near search run in budgeted steps: a field build, the recompute after an edit, and the wide drag's unit open. A large `addMesh` is built in the background. The Worker yields between steps every 4 ms, as before. Results are bit-identical, which the twin tests check.

- **Machine:** Intel Core i5-13600K, 32 GB RAM, Windows 11, Node 24.13.0. Release builds of `main` (`ae40f1d`) and of the branch. The Worker is single-threaded, in the `worker_threads` harness of the Worker scene section.
- **Date:** 2026-10-24. Each pair of "main" and "now" columns was run back to back in the same session, unless a row says otherwise. Other agents shared the machine during the earlier 10M runs (see the note there).
- **Commands:**
  - `pnpm build && pnpm --filter fast-mesh-wasm bench -- --worker --queued`, with `--10m --no-1m` for the 10M case.
  - `-- --worker --contact` for the contact drag.
  - `-- --worker --wide` for the wide drag.
  - For "main", the same bench script runs against main's `dist/`.

### A toggle queued behind a long request

The bench sends a frame with the long request, then at once a frame with a colouring toggle of a small, unrelated pair (60 × 60 terrains). The Worker receives the toggle while it is working on the long request.

- The long requests run on two 1M terrains (B flipped over A): a contact field build (B to A, `maxDistance` 0.08) and a wide one (A to B, `maxDistance` 8).
- The edit moves one vertex of A, with fields A to B and B to A, so A's sign data are rebuilt.
- The add is a third 1M terrain.
- Each row is the median of 5 runs at 1M, and of 2 at 10M.

| 1M triangles | Longest Worker block: main → now | Long request's round trip: main → now | Toggle's round trip: main → now |
| --- | --- | --- | --- |
| Contact field build | 118 → **5.6 ms** | 118 → 118 ms | 118 → 118 ms |
| Wide field build | 115 → **5.9 ms** | 115 → 115 ms | 115 → 115 ms |
| One-vertex edit, fields both ways | 96.5 → **5.3 ms** | 96.5 → 97.6 ms | 96.5 → 97.6 ms |
| `addMesh` | 1,177 → **20.5 ms** | 1,178 → 6.6 ms; mesh ready 1,178 → 1,140 ms | 1,178 → **8.2 ms** |

| 10M triangles | Longest Worker block: main → now | Long request's round trip: main → now | Toggle's round trip: main → now |
| --- | --- | --- | --- |
| Contact field build | 3,356 → **18.2 ms** | 3,356 → 2,361 ms | 3,356 → 2,361 ms |
| Wide field build | 2,374 → **17.7 ms** | 2,375 → 1,144 ms | 2,375 → 1,144 ms |
| One-vertex edit, fields both ways | 1,357 → **14.2 ms** | 1,358 → 1,102 ms | 1,358 → 1,102 ms |
| `addMesh` | 16,088 → **136 ms** | 16,088 → 69 ms; mesh ready 16,088 → 12,245 ms | 16,088 → **71 ms** |

- **The Worker is never blocked for long now.** At 1M the longest block during a field build or an edit is one step, about 5 to 6 ms, against 0.1 s. At 10M it is 14 to 18 ms, against 1.4 to 3.4 s. Batches sent meanwhile are received at once, and superseded ones are coalesced. A field build for a display that is already superseded is dropped at the next step boundary, as 9a dropped it before it started.
- **A toggle still waits for a field build or an edit.** Requests run in order, and other requests wait for a field recompute (§6.1 of the drag-speed design). So its round trip is the long request's, about the same as before at 1M. The total work is unchanged within noise at 1M. The 10M rows are faster now, but main's 10M run was earlier in the session, while other agents were busy on the machine, so read them as an upper bound for main.
- **A toggle no longer waits for an `addMesh`.**
  - A mesh of 65,536 triangles or more is built in the background: steps of the BVH build run between other requests and while the Worker is idle.
  - The toggle is answered in 8 ms at 1M (main: 1.18 s) and 71 ms at 10M (main: 16 s). At 10M, 69 ms of that is the add's own batch: the copy of the 10M inputs into wasm and the mesh's store, which come before the build.
  - The mesh lands about as soon as before (1.14 s at 1M).
  - The longest block is the landing: the insert, the normals, the render buffers and packing the whole new mesh into its result. That is 20 ms at 1M and 136 ms at 10M, O(n) copies that stay in one piece. (Since issue #64 these run in steps before the landing: see "Landing a background add" below.)

### No regression on the drags

| 1M, median | Main | Now |
| --- | --- | --- |
| Contact drag (64 poses, `maxDistance` 0.08), move to colours applied | 1.97 ms (p95 4.19) | 2.02 ms (p95 4.36) |
| Contact drag, longest Worker block | 5.55 ms | 5.75 ms |
| Colouring toggle on / off, round trip | 1.05 / 0.12 ms | 1.08 / 0.14 ms |
| Wide drag, 9a terrain, lock step: move to colours | 78.8 ms | 77.4 ms |
| Wide drag, 9a terrain: longest Worker block per pose | 58.0 ms | **11.8 ms** |
| Wide drag, native terrain, lock step: move to colours | 655 ms | 637 ms |
| Wide drag, native terrain: longest Worker block per pose | 62.5 ms | **12.7 ms** |
| Continuous 60 Hz wide drag, 9a terrain: update interval / release to colours | 256 / 138 ms | 269 / 91 ms |
| Continuous 60 Hz wide drag, native terrain: update interval / release to colours | 801 / 653 ms | 824 / 654 ms |
| Continuous wide drag: longest Worker block (9a / native) | 71.6 / 105 ms | 23.6 / 17.9 ms |

- **The contact path is unchanged within noise.** It stays on the near search, now stepped: 2.02 against 1.97 ms per pose.
- **The wide drag's unit open is no longer one call.** The first step of a recompute takes about 4 ms (main: 58 to 62 ms, all of it the open), and every later step is under about 15 ms. A recompute takes the same time as before.

### Landing a background add, and near-search parts (issue #64)

Landing a background add used to be one call: the insert, the vertex normals, and packing the whole mesh into its frame. Now:

- the normals are computed in the build's steps, after the BVH and still outside the scene (`MeshBuild::set_normals`);
- the last stage of the BVH build (subtree counts, treelet roots and their costs), which was one pass over the tree, is stepped too;
- the engine copies the positions, normals and indices out of wasm memory in steps of their own (6.4 MB each by default) before the mesh lands, and the landing frame sends these copies.

The landing itself is then an O(1) insert. Results are bit-identical: twin tests compare `NormalBuild` with one call, `MeshBuild` with normals with `add_mesh` + `set_normals`, and the landing record with a synchronous add's, with and without an edit in the landing batch. A threaded engine that finishes an add for a waiting request still lands it in one call and packs it into the frame, as before.

`pnpm --filter fast-mesh-wasm bench -- --worker --queued --add-only` (with `--10m --no-1m` for 10M): main (`96b48ec`) against this branch, run back to back twice on 2026-10-24. Other agents were using the machine, so the absolute numbers are higher than in the table above. In a quieter run that morning, main's 1M block was 20.2 ms and this branch's 10.7 ms.

| `addMesh`, longest Worker block | Main, run 1 / run 2 | Now, run 1 / run 2 |
| --- | --- | --- |
| 1M | 25.2 / 21.6 ms | **14.0 / 16.3 ms** |
| 10M | 184 / 207 ms | **103 / 108 ms** |

- **At 1M the landing is no longer the longest block.** What remains are the BVH build's own steps (100,000 units: about 8 to 10 ms in wasm, the tree build's last step up to about twice that) and the add's own batch (about 8 to 10 ms, below).
- **At 10M the longest block is now the add's own batch**, about 100 ms (its round trip equals the block), and no longer the landing (150 to 207 ms).
- **The pieces, timed in wasm at 10M** through the internal `meshAdds`:

  | Piece | Before | Now |
  | --- | --- | --- |
  | Landing (`finish`) | 57 ms (the normals) | 0.04 ms |
  | Copy of the buffers (240 MB) | 35 ms in one piece | 39 steps of 1.0 to 1.5 ms |
  | Normals | inside the landing | 57 steps of 0.3 to 4 ms |
  | Treelet stage of the BVH build | 33 ms in one step | steps of 0.3 to 9 ms |

  At 1M the landing goes from 5.7 to 0.04 ms, and the copy from 3.3 ms in one piece to 4 steps of under 1 ms.
- **Memory.** The copies are allocated at the first copy step and held until the landing frame is posted: 24 MB at 1M and 240 MB at 10M. That is the same size the frame allocated before, held for longer.

**The add's own batch is not stepped.** It is `begin_add_mesh`: the wasm-bindgen copy of the inputs into wasm memory, `Mesh::from_buffers_f64` (finite check, bounding box, conversion to `f32`, index check) and `MeshBuild::new`.

| | 1M | 10M |
| --- | --- | --- |
| `begin_add_mesh` in wasm, without / with `origin` | 9.0 / 7.3 ms | 102 / 71 ms |
| Native: copy of the inputs (`Vec::clone`) | 4 ms | 42 ms |
| Native: `Mesh::from_buffers_f64` | 7.5 to 9 ms | 77 to 93 ms |
| Native: `MeshBuild::new` | 0.8 ms | 7 to 8 ms |

`MeshBuild::new` itself is small. The cost is the store built from the buffers, and the copy into wasm memory in front of it. Stepping that would need a chunked upload into wasm memory and a resumable store construction. That is a new protocol and new store API, not a cheap change, so it is left for a follow-up.

**Near-search parts.** A stepped near search split a node pair until the product of its sides' triangle counts was at most 2^18. It estimated each count from the node's depth, which is exact only on a balanced tree. It now reads the BVH's subtree counts (`Bvh::subtree_triangles`), which the BVH keeps up to date at 4 bytes per node. So a part holds at most 2^18 triangle pairs on any tree. On a split-chain BVH (depth 21 for 640 triangles) the depth estimate left a part of 376,320 pairs. Results are unchanged, and the drags did not regress:

| 1M, median, main → now (back to back, same session) | |
| --- | --- |
| Wide drag, 9a terrain, lock step: move to colours | 99.7 → 92.5 ms |
| Wide drag, 9a terrain: longest Worker block per pose | 14.0 → 13.1 ms |
| Wide drag, native terrain, lock step: move to colours | 829 → 810 ms |
| Wide drag, native terrain: longest Worker block per pose | 16.3 → 13.4 ms |
| Continuous 60 Hz wide drag (9a / native): longest Worker block | 26.8 / 23.8 → 27.1 / 19.9 ms |
| Contact drag: move to colours applied | 3.20 → 3.28 ms (p95 6.77 → 6.82) |
| Contact drag: longest Worker block | 7.08 → 7.28 ms |

The 10M near-search part of the native table below (up to 39 ms) was not measured again.

### Native steps

Each step was timed with `Points(4096)`, native, single-threaded, on the scene crate's terrain pair (B flipped 7.9 above A). The scratch profile is not committed.

| | 1M: total, longest step | 10M: total, longest step |
| --- | --- | --- |
| Contact field build (adjacency, incidence, sign data, near search, values) | 252 ms, 4.6 ms (one call: 230 ms) | 2.6 s, 20 to 24 ms |
| One-vertex edit, fields both ways (sign data rebuilt) | 152 ms, 4.5 ms (one call: 135 ms) | 1.7 s, 26 ms |
| Wide field build | 585 ms, 5.8 ms (one call: 590 ms) | 7.0 s, 23 ms |
| Wide recompute after a move | 470 ms, 6 to 7 ms (one call: 470 ms) | 5.7 s, up to 39 ms (one near-search part) |
| BVH build (`BvhBuild`, 100,000 units a step) | 408 ms, 4.5 ms (`Bvh::build`: 412 ms) | |
| BVH build with the `parallel` feature (20 threads) | 315 ms, 10 ms (`Bvh::build`: 80 ms) | |

- **Stepping adds 0 to 12% natively.** The sign data passes cost the same in steps, apart from the per-step overhead.
- **A stepped BVH build with threads is about four times slower than the one-call parallel build.** It is still faster than single-threaded. A step builds small subtrees in parallel and large ones in parts, but splits the large nodes in between one at a time. The Worker uses it only for background adds, where the total matters less than not blocking.
- **The longest single pieces at 10M:**
  - a near-search part where the surfaces intersect over a wide area (each part is bounded to 2^18 triangle pairs);
  - the transitions between sign data passes;
  - the finish of a full field, and its apply, which now runs in a step of its own.

## Parallel (the `parallel` feature)

Every benchmark that the `parallel` feature touches, with the feature off and on. Results are bit-identical either way (the tests check it); only the time changes.

- **Machine:** Intel Core i5-13600K (14 cores: 6 performance cores with two threads each, 8 efficiency cores; 20 hardware threads), 32 GB RAM, Windows 11
- **Threads:** rayon's default global pool, 20 threads (`RAYON_NUM_THREADS` unset). "Off" is the build without the feature and runs on one thread.
- **Toolchain:** Rust 1.99.0, release profile (LTO, `codegen-units = 1`)
- **Date:** 2026-10-18 and 2026-10-19
- **Method:** the bench binaries for both builds were built first, then each crate's benchmarks ran off and then on, back to back, in two full rounds. Before each run the process list was checked for other `cargo`, `rustc` or bench processes. Round 2 was quiet throughout and is the one reported. Round 1 had one other `cargo` process at one point and agrees with round 2 within about 10% on every row except the two noted below.
- **Inputs:** those of the sections above (`terrain(708)` with 1,002,528 triangles, `terrain(2237)` with 10,008,338 for the BVH, the 998,000-triangle sphere), plus the many-mesh scenes described below. The times are criterion's middle estimate.

| Benchmark | Off | On | Speed-up |
| --- | --- | --- | --- |
| BVH build 1M, `Quality::Sah` | 408 ms | 65.6 ms | 6.2× |
| BVH build 1M, `Quality::Fast` | 113 ms | 52.2 ms | 2.2× |
| BVH build 10M, `Quality::Sah` | 4.55 s | 731 ms | 6.2× |
| BVH build 10M, `Quality::Fast` | 1.51 s | 650 ms | 2.3× |
| `SignData::build`, pseudonormal | 132 ms | 35.6 ms | 3.7× |
| `SignData::build`, pseudonormal, adjacency not cached | 194 ms | 100 ms | 1.9× |
| `SignData::build`, winding | 35.5 ms | 21.2 ms | 1.7× |
| `signed_distances`, 100,000 points, pseudonormal, `max_distance` 1 | 79.0 ms | 5.20 ms | 15× |
| `signed_distances`, 100,000 points, pseudonormal, `max_distance` ∞ | 137 ms | 8.93 ms | 15× |
| `signed_distances`, 100,000 points, winding, `max_distance` 1 | 2.39 s | 195 ms | 12× |
| `signed_distances`, 100,000 points, winding, `max_distance` ∞ | 4.54 s | 382 ms | 12× |
| `raycast_many`, 100,000 rays | 76.6 ms | 5.26 ms | 15× |
| `closest_point_many`, 100,000 points | 192 ms | 12.1 ms | 16× |
| `triangles_near`, 1M vs 1M, margin 0.080 (64-pose cycle) | 1.38 ms | 0.400 ms | 3.5× |
| `intersecting_pairs`, 1M vs 1M, pushed 0.5 (64-pose cycle) | 4.46 ms | 0.677 ms | 6.6× |
| `analyze`, no self-intersection check | 489 ms | 405 ms | 1.2× |
| `analyze`, with self-intersection check | 908 ms | 419 ms | 2.2× |
| `fill_10_holes_1M`, `fair` off | 353 ms | 350 ms | 1.0× |
| `fill_10_holes_1M`, `fair` on | 4.27 s | 1.77 s | 2.4× |
| `weld_1M_10pct_dupes` | 97.3 ms | 66.9 ms | 1.5× |
| `section_drag_1M`, 64 planes | 15.8 ms | 15.1 ms | 1.05× |
| `section_drag_3x333k`, 64 planes over three meshes | 23.0 ms | 11.3 ms | 2.0× |
| `section_fill_1M_sphere`, fill off | 339 µs | 328 µs | 1.0× |
| `section_fill_1M_sphere`, fill on | 537 µs | 527 µs | 1.0× |
| `section_gap_two_1M`, `section_gap` alone | 151 µs | 149 µs | 1.0× |
| `section_gap_two_1M`, plane change (both meshes) + `section_gap` | 840 µs | 513 µs | 1.6× |
| Field drag 1M, one side, 64 poses | 97.5 ms | 30.7 ms | 3.2× |
| Field drag 1M, both sides, 64 poses | 194 ms | 37.7 ms | 5.2× |
| `many_mesh_drag_30k`, 64 poses | 352 ms | 57.1 ms | 6.2× |
| `many_mesh_drag_30k_touching`, 64 poses (measured 2026-10-19, after the others) | 98.8 ms | 31.3 ms | 3.2× |
| `many_mesh_drag_30k_apart`, 64 poses | 319 µs | 318 µs | 1.0× |
| `many_mesh_drag_1M`, 64 poses | 39.2 s | 3.60 s | 11× |

**The many-mesh scenes.** Six `terrain(n)` meshes in a row along x, each overlapping the next by an eighth of its side, so that neighbouring sheets cross along curves. There is a distance field for every ordered pair (30 fields, `max_distance` 0.5, sign `Auto`), and each mesh is coloured by its field to the next. All meshes are Lazy. The middle mesh (the third) is dragged through 64 poses, rising and falling by up to 0.25 and sliding 0.1 along x. Each pose makes ten fields due (five on the moved mesh, five measured to it) and runs `set_placement`, `update_fields`, `colors` of every mesh and `take_changes`. `many_mesh_drag_30k` uses `terrain(122)` (29,768 triangles per mesh, the owner's scale) and recolours about 390 vertices per pose. `many_mesh_drag_30k_touching` overlaps neighbours by one cell (step 121), the owner's common pose of surfaces just touching or just apart: only a strip along each shared edge is within reach, and about 200 vertices are recoloured per pose. `many_mesh_drag_30k_apart` puts the same meshes a whole side apart, so no field has anything within reach and no vertex is recoloured. `many_mesh_drag_1M` uses `terrain(708)` and recolours about 46,600 vertices per pose. It runs only with `FAST_MESH_BENCH_LARGE=1`, because a run without the feature takes about 10 minutes. The 30k benchmarks take 50 samples, the others 10.

**Owner-scale check (spec anti-goal: parallel at most 1.05 times sequential).** Met with the thresholds as they were, so none was changed:
- `many_mesh_drag_30k`: parallel takes 0.16 times the sequential time (57.1 ms against 352 ms; round 1: 57.8 against 387 ms).
- `many_mesh_drag_30k_touching`: parallel takes 0.32 times the sequential time (31.3 ms against 98.8 ms; second round 28.7 against 99.1 ms). Measured later than the rest of the table, after rebasing onto Phase 8b, in two back-to-back rounds with no other `cargo` or `rustc` process running (CPU load 14% and 7% before them). An earlier pair of rounds gave 31.1 against 106 ms, with another checkout's build running, and 30.5 against 102 ms, quiet. The field threshold counts A's whole triangle count for the near search as soon as the root boxes are within reach, so here it takes the parallel path although only a strip is near; it still wins, because the ten due fields run side by side and each one's remaining work (the known candidates, the sign and distance passes) is not small. In the same rounds the 30k row measured 61.4 and 68.4 ms against 376 and 354 ms, and the apart row 321 and 337 µs against 328 and 334 µs (1.01 times at most).
- `many_mesh_drag_30k_apart`: parallel takes 1.00 times the sequential time (318 µs against 319 µs; round 1: 318 against 612 µs). Nothing is in reach, so every field's work estimate is small, and the fields run one after another on the calling thread as without the feature. An earlier separate run of this benchmark with 10 samples measured 345 and 359 µs off against 349 and 380 µs on: at about 5 µs per pose, the spread between runs is larger than the 5% margin, so the 50-sample round 2 is the one to read.

**Where the gain is small.**
- **One mesh, one section** (`section_drag_1M`, 1.05×): one slice per plane, and slicing itself is sequential. Several meshes or sections in one update gain (2.0× over three meshes).
- **`analyze` without the self-intersection check** (1.2×): its passes run side by side, but they all wait for the edge map's sort, which stays sequential.
- **Fill without fairing** (1.0×): nothing in it is parallel. With fairing on, the three coordinate solves run side by side, with parallel matrix products from 1,024 rows.
- **The sphere fill and the gap alone** (1.0×): single small computations under every threshold.


**BVH memory.** With the feature, the BVH bench prints a smaller `byte_size` for the same nodes and depth: 20.2 MB against 21.6 MB at 1M, and 193.5 MB against 215.3 MB at 10M. The difference is reserved capacity, because `byte_size` sums the arrays' `capacity()`. The build without the feature reserves the expected node count up front, while the parallel build reserves the exact count.

**Round-to-round differences.** Round 1 differs from round 2 by more than 10% on two rows, both feature-off: `many_mesh_drag_1M` (51.3 s against 39.2 s) and `many_mesh_drag_30k_apart` (612 µs against 319 µs). The feature-on results of both rounds agree within 2% on both rows.

**Thresholds.** Each parallel site runs sequentially below a named threshold, measured on this machine by the task that added it (the measurements are in each constant's documentation):

| Site | Constant | Value |
| --- | --- | --- |
| BVH full build: split into parts built in parallel | `PAR_MIN_SUBTREE` | 8,192 triangles |
| BVH full build: parallel bound and SAH bin reductions of a part | `PAR_MIN_PRIMS` | 65,536 triangles |
| `signed_distances` | `PAR_MIN_POINTS` | 512 points |
| `SignData::build`, per pass | `PAR_MIN_SIGN_ELEMS` | 8,192 elements |
| `raycast_many`, `closest_point_many` | `PAR_MIN_QUERIES` | 256 queries |
| `triangles_near`, `intersecting_pairs` | `PAR_MIN_PAIR_TRIS` | 2,048 triangles (A and B together) |
| `self_intersecting_pairs` | `PAR_MIN_SELF_TRIS` | 1,024 triangles |
| `analyze` passes side by side | `PAR_MIN_ANALYZE_TRIS` | 2,048 triangles |
| Fairing: the three solves side by side | `PAR_MIN_FAIR_SOLVE` | 128 new vertices |
| Fairing: matrix products | `PAR_MIN_FAIR_ROWS` | 1,024 rows |
| Weld: keys and neighbour search | `PAR_MIN_WELD_VERTS` | 16,384 vertex slots |
| Scene: fields due in one update | `PAR_MIN_FIELD_WORK` | 12,288 (work estimate, summed) |
| Scene: section entries due in one update | `PAR_MIN_SECTION_WORK` | 8,192 triangles (summed) |

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
pnpm build && pnpm --filter fast-mesh-wasm bench -- --worker --10m   # the Worker scene in a worker_threads Worker (1M, then 10M with --10m; add --quick for a short run, --wide for just the wide-range drag)
pnpm build && pnpm --filter fast-mesh-wasm bench -- --worker --queued   # issue #48: a toggle queued behind a field build, an edit and an addMesh (1M; --10m adds 10M, --no-1m skips 1M, --add-only runs just the addMesh); --contact for just the 1M contact toggle and drag
pnpm build && pnpm --filter fast-mesh-wasm bench:store   # the store over 1000 meshes: per-action cost and add bursts on a Scene, placement status changes on a WorkerScene (add --quick for a short run)
pnpm build && node scripts/browser-check.mjs --build --bench --kind 9a   # the threaded Worker against threads: 1 in headless Chrome (1M; --kind native, --n 2236 --lock-step 4 --seconds 30 for 10M, --threads N for another pool size)
cargo bench -p fast-mesh-scene --bench scene -- "wide drag"   # the native wide-range drag (1M; FAST_MESH_BENCH_LARGE=1 adds 10M, and the 6 x 1M many-mesh drag) and opening a step unit
cargo bench -p fast-mesh-ops                          # analyze, orient, fill_holes (fair on and off), weld and remove_small_components on 1M triangles
cargo bench -p fast-mesh-scene --bench scene -- "many meshes"   # the many-mesh drags (6 x 30k crossing, touching and apart)
FAST_MESH_BENCH_LARGE=1 cargo bench -p fast-mesh-scene --bench scene -- "many meshes"   # adds 6 x 1M (about 10 minutes without the feature)
cargo bench -p fast-mesh-scene --features parallel          # any of the above with the parallel feature (same for the other crates)
```
