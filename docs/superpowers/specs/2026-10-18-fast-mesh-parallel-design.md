# fast-mesh: multithreading (stage 1, native Rust)

- **Status:** draft for review
- **Parent spec:** `2026-10-14-fast-mesh-engine-design.md` §15 item 10 lists "multithreaded build" under later work. This document brings it forward.
- **Runs in parallel with** Phase 8a (`feat/fast-mesh-js`). This stage touches only engine crates behind a cargo feature. It does not touch `fast-mesh-wasm`, the npm package, build scripts or CI wasm steps, so the two branches do not conflict.
- **Labels:** *(owner)* marks what the owner said. *(my call)* marks choices made without asking.

## 1. Intent

- In real browser and React apps, single-threaded updates during interaction can look sluggish *(owner)*. The usual answer is to compute in parallel *(owner)*.
- fast-mesh is a general-purpose library. Performance work targets large meshes and complex scenes, not only the owner's own scale *(owner)*.

There are two stages:
1. **Stage 1 (this document).** Parallelise the engine's hot paths with rayon, behind a cargo feature `parallel`. Measure natively.
2. **Stage 2 (separate spec, after 8a merges).** A wasm threads build:
   - wasm32 atomics through nightly and build-std, plus `wasm-bindgen-rayon`;
   - the threaded build is used only on cross-origin-isolated pages, falling back to the single-threaded build;
   - both artifacts ship in the npm package.

## 2. Goals, non-goals, anti-goals

### Goals
1. Add a `parallel` cargo feature, off by default, to each engine crate that has hot loops. Enabling it on `fast-mesh-scene` (and `fast-mesh-ops`) turns it on for the crates they depend on.
2. Parallelise, each with a size threshold below which code runs sequentially *(my call: small work must not pay thread overhead)*:
   - **BVH build.** Recursive SAH splits use `rayon::join` above a subtree size threshold. Centroid and bound reductions are parallel above a threshold.
   - **Distance fields.** Per-vertex `signed_distances` runs over candidate vertices in parallel chunks.
   - **Field updates across fields.** In the Scene, the compute phase of all fields due in one update (`run_eager_fields`, `update_fields`) runs in parallel across fields. The write phase (values, colours, reports) stays sequential and in field order.
   - **Normals.** Per-vertex recomputation over dirty vertices runs in parallel chunks.
   - **Sections.** Slicing several meshes, or several sections, in one update runs in parallel. The write phase stays sequential.
   - **Repair.**
     - `self_intersecting_pairs` uses parallel subtree-pair traversal.
     - `analyze` runs its independent passes concurrently.
     - Fairing runs its three coordinate solves concurrently, with a parallel sparse matrix-vector product above a threshold.
     - The weld grid builds its cells in parallel.
   - **Queries in batches.** `closest_point_many` and `raycast_many` run per query in parallel.
3. **Determinism.** Every result is identical, bit for bit, with `parallel` on or off and for any thread count. Reductions use fixed chunk boundaries and a fixed combine order, never order-dependent float sums *(my call: reproducibility and the existing property tests depend on it)*.
4. **Proof.**
   - Every existing test suite passes with `--features parallel`.
   - New tests compare parallel against sequential outputs bit for bit on large inputs.

### Non-goals
- The wasm threads build and the JS worker/pool plumbing (stage 2).
- Changing any public API. The feature is invisible except for speed.
- Making `Scene` itself `Sync` or callable from several threads. Parallelism stays inside a call.

### Anti-goals
- **No slower small scenes.** With `parallel` on, the owner-scale scene (6 meshes × 30k triangles, all pairs coloured, dragging one) must not get slower than sequential. The thresholds guarantee this.
- **No nondeterminism, and no data races.** Data races are impossible without `unsafe`. Order-dependent results are forbidden by goal 3.

## 3. Constraints
- **Dependency:** `rayon` is the one new dependency, optional and enabled only by `parallel`. The version is pinned in `[workspace.dependencies]`.
- **Code rules:** no `unsafe`. MSRV 1.88. Fallible allocation still applies; allocations in parallel paths use `try_reserve` before work is split.
- **wasm builds:** the `wasm32-unknown-unknown` build without the feature must stay byte-for-byte unaffected in behaviour. CI keeps building it.
- **CI:** add a job, or steps, running `cargo test --workspace --features parallel` (on a crate list where the feature exists) and `cargo clippy` with the feature.
- **Thresholds:**
  - Constants are named per site and documented.
  - Initial values come from measurement on the i5-13600K, as the point where parallel wins.
  - An environment override for benches only is the builder's choice.

## 4. Testing
- **Full suite with the feature:** the whole existing suite under `--features parallel`, including the property tests.
- **Bit-identity tests**, new, each with inputs above its threshold:
  - BVH node arrays;
  - field values and colours after a drag sequence;
  - normals;
  - section results;
  - analyze reports;
  - fill results with fairing on.
- **Thread counts:** run under `RAYON_NUM_THREADS=1` and the default count in CI, so the single-thread pool path is also covered.

## 5. Benchmarks
- **Existing benches** run with and without `parallel` at 1M and 10M where they exist today:
  - BVH build;
  - field drag (one-sided and two-sided);
  - `analyze`;
  - fill with fairing;
  - the section drag.
- **New scene benches:**
  - **Many-mesh drag:** 6 meshes of 1M triangles with distance colouring between every pair, dragging one. This is where field-level parallelism shows.
  - **Owner-scale drag:** 6 meshes of 30k triangles, all pairs coloured, dragging one. It checks the no-regression anti-goal.
- **Recording:** results go into `docs/benchmarks.md` as a "Parallel" section, with core count and thread count noted.

## 6. Left to the builder
- Exact threshold values (measured), and chunk sizes.
- How the Scene splits the compute and write phases for fields and sections, provided writes stay sequential and in order.
- Whether to add `parallel` to `fast-mesh-store`. Probably not needed.
- Splitting into plan tasks.

## 7. Decisions after the builder check (all *my call*; they override earlier sections)

1. **Normals are out of stage 1.** The incremental Scene normals channel is being added by Phase 8a on another branch. Parallelising it waits for stage 2, after 8a merges, so this branch stays conflict-free. `fast_mesh::compute_vertex_normals` (the `fast-mesh` crate) is not touched either.
2. **`SignData::build` is in scope.** It rebuilds over the whole target mesh on every edit of that mesh (~150 ms at 1M), so it is the biggest per-edit cost.
   - Its per-vertex and per-corner loops (pseudonormals, edge normals, winding moments) run in parallel chunks.
   - It still runs inside the Scene's sequential prepare phase, but is internally parallel.
   - The `sign_build_*` benches join the §5 list.
3. **Allocation in parallel paths.**
   - Workers may allocate, but only fallibly (`try_reserve`). When several workers fail, the error from the lowest work-item index wins, so the error is deterministic.
   - The parallel SAH build allocates per subtree and splices the results deterministically. It does not pre-reserve the 2n−1 worst case.
   - The existing out-of-memory fault-injection hooks are thread-local. While a hook is armed (test builds only), every parallel site runs sequentially, so the existing OOM tests keep covering the code they cover today.
   - A dedicated test covers the lowest-index error rule in parallel.
4. **Scene benches.**
   - Six meshes, with a distance field for every ordered pair: 30 fields.
   - Each mesh is coloured by its field to the next mesh by index.
   - Dragging one mesh therefore makes 10 fields due per pose (5 on the moved mesh, 5 measured to it).
   - "Not slower" at owner scale is a recorded measurement in `docs/benchmarks.md`, not a CI gate. Parallel must be within 5% of sequential (criterion noise) or faster, and thresholds are tuned until it is.
5. **Determinism baseline.**
   - Results must be bit-identical between `parallel` on and off, for any thread count.
   - They need not match today's main bit for bit. The sequential build may adopt the same fixed chunked reductions (for example, chunked CG dot products and gather-based transposed products), and any golden values in tests are refreshed.
   - Property tests that compare against from-scratch computation stay as they are.
   - §3's "unaffected" means unaffected in API and behaviour class, not in last-bit float results.

## 8. Implementation notes

What the build decided along the way, where it differs from the sections above, and what it measured. The full benchmark tables are in `docs/benchmarks.md`, section "Parallel".

### Rulings
1. **Shared helpers live in `fast_mesh_store::par`.** `fast-mesh-store` gets a `parallel` feature after all (§6 expected it would not need one). It is the common base of every engine crate, so the deterministic helpers (`map_chunks`, `map_range`, `join`, `force_sequential`, `parallel_entries`) need no new crate. The cost is a `doc(hidden)` public module in the store's API.
2. **`force_sequential` is thread-local.** The thread that arms a fault-injection hook is the one that calls the parallel helper, so a thread-local flag covers every site that thread enters, and tests running side by side cannot turn each other's flag off. It is never used in production.
3. **The helpers take an explicit `oom: impl Fn() -> E`** instead of requiring `E: From<StoreError>`. This keeps the public API unchanged: no new `From` impls on `QueryError` and the other error types. Call sites pass one extra argument.
4. **The scene's field threshold uses a work estimate,** not the exact candidate count. The near search now runs inside the parallel compute phase, so the candidates are not known when the decision is made. The estimate counts every vertex slot of A when all are recomputed; otherwise it counts the known candidates, plus A's triangle count for the near search when B's root box is within the margin of A's.
5. **`signed_distances_bit_identical` is smaller than planned:** about 30k points in pseudonormal mode and about 8k in winding mode (still more than 60 chunks), and it returns after the reference run when the feature is off. The planned 300k points cost CI time without strengthening the proof.
6. **`triangles_near` is parallel too** (a subtree-pair split, as for self pairs). It was not in §2's list, but the one-sided field drag is bound by it.
7. **The weld neighbour search must keep memory O(n).** The first version stored every close pair, so it could run out of memory with the feature where the build without it succeeds. The ruling was a chunk-local spanning-forest reduction, or dropping the parallel search.
8. **The weld uses a lock-free union-find instead,** superseding the spanning forest of ruling 7, which is not strictly O(n) when a cluster spans many chunks. One `AtomicU32` parent per vertex slot, no `unsafe`. A union links the larger root under the smaller by compare-exchange and retries if that root moved. A find halves paths, only ever re-pointing a vertex to its grandparent. Every link points from a root to a smaller id, so each group's root is its smallest member however the unions interleave, and the roots match the sequential union-find. The re-reviewer checked the argument, and a stress probe and the bit-identity tests back it.

### Deviations and limits
- **Normals** are out of stage 1, as §7.1 says.
- **The BVH build forks at most 24 joins deep** and builds the rest of a chain as one sequential part, which bounds the stack on degenerate SAH chains. Splitting long chains loses their join parallelism.
- **The feature-off build does not split into parts:** below the threshold, without the feature, or when forced sequential, it builds in one pass into the arena with the same chunked reductions. So where parts are cut cannot change the tree.
- **The feature-off `Quality::Fast` build was measured about 5% slower than before stage 1** (the chunked reductions). In the "Parallel" runs it took 113 ms, against 115 ms in the BVH table above it, which was measured on another day.
- **Still sequential:** the Fast build's per-triangle boxes and Morton sort; `analyze`'s edge-map sort (which limits `analyze` to 1.2× without the self-intersection check); the weld's sort; fairing's dot products and vector updates, which are chunked in a fixed order so the sequential build gives the same bits. A parallel sort helper would be follow-up work.
- **No golden values in the scene and fields bit-identity tests.** CI runs on Linux and the test meshes use `sin` and `atan2`, so the reference is the forced-sequential path, which is the code the feature-off build runs. The BVH test has golden values.
- **Thread-local test counters undercount** once their site runs on workers (for example `LEAF_VISITS` in the pair queries and `PAIRS` in the weld). Their documentation says so, and the tests that read them stay below the thresholds or force the sequential path.
- **Not measured: a barely-touching owner-scale scene.** The owner-scale check covers meshes that cross and meshes far apart. It does not cover meshes that only just touch, where the pair split still runs for little work. That is follow-up work.
- **BVH `byte_size` is smaller with the feature** for the same tree: 20.2 against 21.6 MB at 1M. `byte_size` sums capacity. The build without the feature reserves the expected node count up front; the parallel build reserves the exact count.
- **`many_mesh_drag_1M` runs only with `FAST_MESH_BENCH_LARGE=1`.** Without the feature it takes about 10 minutes per run.
- **Nested parallelism:** inside the parallel field compute, `triangles_near` (above its threshold) forks again. Rayon handles the nesting, and the owner-scale check below covers it.

### Thresholds
Named constants, one per site, each documented with the measurement behind it (i5-13600K):

| Site | Constant | Value |
| --- | --- | --- |
| BVH build, parts | `PAR_MIN_SUBTREE` | 8,192 triangles |
| BVH build, reductions of a part | `PAR_MIN_PRIMS` | 65,536 triangles |
| `signed_distances` | `PAR_MIN_POINTS` | 512 points |
| `SignData::build`, per pass | `PAR_MIN_SIGN_ELEMS` | 8,192 elements |
| `raycast_many`, `closest_point_many` | `PAR_MIN_QUERIES` | 256 queries |
| `triangles_near`, `intersecting_pairs` | `PAR_MIN_PAIR_TRIS` | 2,048 triangles |
| `self_intersecting_pairs` | `PAR_MIN_SELF_TRIS` | 1,024 triangles |
| `analyze` | `PAR_MIN_ANALYZE_TRIS` | 2,048 triangles |
| Fairing solves | `PAR_MIN_FAIR_SOLVE` | 128 new vertices |
| Fairing products | `PAR_MIN_FAIR_ROWS` | 1,024 rows |
| Weld | `PAR_MIN_WELD_VERTS` | 16,384 vertex slots |
| Scene fields | `PAR_MIN_FIELD_WORK` | 12,288 (work estimate) |
| Scene sections | `PAR_MIN_SECTION_WORK` | 8,192 triangles |

The final measurement (below) needed no change to any of them.

### Measured (20 threads; feature off → on)
- **BVH build:** SAH 408 → 65.6 ms at 1M (6.2×) and 4.55 s → 731 ms at 10M (6.2×); Fast 113 → 52.2 ms (2.2×) and 1.51 s → 650 ms (2.3×).
- **Sign data and distances:** `SignData::build` pseudonormal 132 → 35.6 ms (3.7×); `signed_distances` 15× (pseudonormal) and 12× (winding).
- **Batch queries:** `raycast_many` 15×, `closest_point_many` 16×. Pair queries 1M vs 1M: `triangles_near` 1.38 → 0.40 ms (3.5×), `intersecting_pairs` 4.46 → 0.68 ms (6.6×).
- **Repair:** `analyze` 1.2×, with the self-intersection check 2.2×; fill with fairing 4.27 → 1.77 s (2.4×); weld 1.5×.
- **Sections:** one mesh 1.05×, three meshes 2.0×, the two-sphere plane change with the gap 1.6×.
- **Field drags (1M):** one side 97.5 → 30.7 ms (3.2×), both sides 194 → 37.7 ms (5.2×).
- **Many meshes (§7.4):** `many_mesh_drag_1M` 39.2 → 3.60 s (11×).
- **Owner-scale anti-goal:** `many_mesh_drag_30k` 352 → 57.1 ms, so parallel takes 0.16 times the sequential time. The added `many_mesh_drag_30k_apart` (the same meshes out of reach, so the fields run sequentially even with the feature) measured 319 → 318 µs, 1.00 times. Both are within the 1.05 limit.
