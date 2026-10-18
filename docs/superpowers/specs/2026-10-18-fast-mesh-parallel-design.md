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
