# fast-mesh Parallel (stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelise the engine's hot paths with rayon behind an off-by-default `parallel` cargo feature. Results must be bit-identical with the feature on or off and for any thread count.

**Architecture:**
- **Shared helpers:** a small helper module `fast_mesh_store::par` (feature `parallel` in fast-mesh-store) provides deterministic chunked map, `join` and lowest-index error combining. Without the feature the helpers run sequentially, so call sites are identical in both builds.
- **Feature propagation:** each crate's `parallel` feature enables its dependencies' `parallel`.
- **Call sites:** hot loops switch to the helpers. Below a named per-site threshold they run sequentially.
- **Scene:** field and section updates split into a parallel compute phase and a sequential, ordered write phase.

**Tech Stack:** Rust 1.99 (MSRV 1.88), rayon (optional), criterion, proptest.

**Spec:** `docs/superpowers/specs/2026-10-18-fast-mesh-parallel-design.md`. §7 overrides earlier sections.

**Branch / worktree:** `feat/fast-mesh-parallel` at `C:/Users/YoonCarrot/orca/projects/fast-mesh-wasm-parallel`. Never touch the other checkout (`.../fast-mesh-wasm`, Phase 8a). Never touch `crates/fast-mesh-wasm`, `crates/fast-mesh`, `packages/`, `scripts/`, or the wasm steps of CI.

## Global Constraints
- **Code rules:**
  - No `unsafe`. MSRV 1.88.
  - `rayon` is the only new dependency. It is optional, pinned in `[workspace.dependencies]`, and enabled only via `parallel`.
- **Allocation:**
  - Fallible allocation everywhere (`try_reserve`). Workers may allocate, but only fallibly.
  - Parallel errors resolve to the error of the lowest work-item index.
- **Determinism:**
  - Bit-identical outputs with `parallel` on and off, for any `RAYON_NUM_THREADS`.
  - Reductions use fixed chunk boundaries and a fixed combine order.
  - Sequential code may adopt the same chunked reductions. Golden values may be refreshed. Matching today's main bit for bit is not required.
- **Test-build fallback:** while any out-of-memory fault-injection hook is armed (test builds), every parallel site runs sequentially.
- **Thresholds:** named `const`s per site (e.g. `PAR_MIN_VERTICES`), documented, with values from measurement on the i5-13600K.
- **Owner-scale guarantee:** the 6×30k scene with parallel on must be ≤ 1.05 × sequential, recorded in `docs/benchmarks.md`.
- **Public API:** unchanged.
- **Commits:**
  - Conventional messages ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or the implementing model).
  - Each message goes in a NEW temp file, committed with `git commit -F`.
  - Files are UTF-8 with LF.
- **Every task ends green on:**
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo clippy --workspace --all-targets --features <crates>/parallel -- -D warnings`
  - `cargo test --workspace`
  - `cargo test --workspace --features <crates>/parallel` (the Task 1 alias or command)
  - the same parallel test run with `RAYON_NUM_THREADS=1`

## Review Focus
1. **The same results on 1 thread, 2 threads and N threads.** A 2-thread pool shards differently from an N-thread pool, but the outputs must still be identical. Each bit-identity test runs under `ThreadPoolBuilder::new().num_threads(k).build()?.install(..)` for k ∈ {1, 2, 8}. Tests: every task's `*_bit_identical` test.
2. **OOM under parallel.**
   - When a hook is armed, parallel sites run sequentially, so the existing OOM sweeps still pass with `--features parallel`.
   - A forced failure in work item 7 of 20 returns that error, not item 12's.
   - Test: Task 1 `lowest_index_error_wins`.
3. **Tiny inputs.** A 12-triangle cube and a 1-vertex field never enter rayon (stay below threshold) and give identical results. Tests: Tasks 1–4 `small_inputs_stay_sequential`, which use a test counter of parallel entries.
4. **Scene ordering.** With several fields due in one update, reports (`fields_changed`, `colors_dirty`) list items in the same order as the sequential build, and colours are identical. Test: Task 2 `parallel_field_updates_report_in_order`.
5. **The wasm build without the feature still compiles.** CI's existing `cargo build --target wasm32-unknown-unknown -p fast-mesh-scene` stays green, and rayon must not be pulled in by default. Test: Task 1, a `cargo tree -e features -p fast-mesh-scene` check that rayon is absent without the feature (a script step in CI).

---

### Task 1: Feature plumbing, `fast_mesh_store::par`, CI, and the parallel BVH build

**Files:**
- **Modify:**
  - root `Cargo.toml`: `rayon = "1.10"` in `[workspace.dependencies]`, or the current 1.x version that supports MSRV 1.88. Verify.
  - each engine crate's `Cargo.toml` (store, bvh, query, fields, scene, ops): `[features] parallel = ["fast-mesh-store/parallel", ...deps' parallel, "dep:rayon"]` where the crate uses rayon itself.
- **Create:** `crates/fast-mesh-store/src/par.rs`, made public with `#[doc(hidden)] pub mod par;`.
- **Modify:** `crates/fast-mesh-bvh/src/build.rs` (parallel subtree build and reductions).
- **Modify:** `.github/workflows/ci.yml`. In the Rust job, add steps for:
  - clippy and tests with `--features fast-mesh-scene/parallel,fast-mesh-ops/parallel`;
  - the same tests with `RAYON_NUM_THREADS=1`;
  - the rayon-absent `cargo tree` check.
- **Modify:** `CONTRIBUTING.md` (how to run with the feature).

**Interfaces (produces):**
```rust
// fast_mesh_store::par — identical signatures with and without the feature
pub fn enabled() -> bool;                                   // cfg!(feature = "parallel")
pub fn map_chunks<T: Sync, R: Send, E: Send>(
    items: &[T], chunk: usize, threshold: usize,
    f: impl Fn(usize /*chunk index*/, &[T]) -> Result<R, E> + Sync,
) -> Result<Vec<R>, E>;                                     // results in chunk order; lowest chunk error wins; sequential if items.len() < threshold or !enabled() or force_sequential()
pub fn map_range<R: Send, E: Send>(len: usize, chunk: usize, threshold: usize,
    f: impl Fn(std::ops::Range<usize>) -> Result<R, E> + Sync) -> Result<Vec<R>, E>;
pub fn join<A: Send, B: Send>(parallel: bool, a: impl FnOnce() -> A + Send, b: impl FnOnce() -> B + Send) -> (A, B);
pub fn set_force_sequential(on: bool);                      // process-wide AtomicBool; test helpers in each crate set it while a fault hook is armed
pub fn force_sequential() -> bool;
#[cfg(any(test, feature = "test-counters"))] pub fn parallel_entries() -> usize; // counter for small_inputs_stay_sequential
```
- **Wiring the fallback:** each crate's existing `fail_allocations_after(n)` test hook calls `par::set_force_sequential(n.is_some())`.
- **Parallel BVH build:**
  - Subtrees above `PAR_MIN_SUBTREE` primitives build with `join` into separate node vectors, spliced afterwards in the same order the sequential build emits nodes. Alternatively, the sequential build adopts the same split-then-splice layout so both match.
  - Centroid and bound reductions use `map_range` above `PAR_MIN_PRIMS`.

- [ ] **Step 1: Write failing tests.**
  - store par:
    - `map_chunks_order_and_results`;
    - `lowest_index_error_wins` (chunks 7 and 12 fail, result is chunk 7's error, for k ∈ {1, 2, 8} pools);
    - `below_threshold_is_sequential` (counter unchanged).
  - bvh: `build_bit_identical`. A 200k-triangle wavy mesh, SAH and Fast, compared across k ∈ {1, 2, 8} pools and a feature-off reference. Compare node arrays (`node_slots`, bounds bits, children, leaf ranges) through the public `NodeView` traversal.
  - bvh: `small_inputs_stay_sequential` (12-triangle cube).
- [ ] **Step 2: Run** `cargo test -p fast-mesh-store -p fast-mesh-bvh --features fast-mesh-bvh/parallel`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full gate, including the `RAYON_NUM_THREADS=1` run and a feature-off workspace test. Expected: PASS.
- [ ] **Step 5: Bench.** `cargo bench -p fast-mesh-bvh` with and without the feature (1M; 10M if practical). Note the numbers in the report.
- [ ] **Step 6: Commit** `feat(parallel): feature plumbing and deterministic helpers` and `perf(bvh): parallel SAH build behind the parallel feature`.

### Task 2: Fields: `signed_distances`, `SignData::build`, and parallel field updates in the Scene

**Files:**
- **Modify:** `crates/fast-mesh-fields/src/distance.rs`, `crates/fast-mesh-fields/src/sign.rs` (and `winding.rs` if the moments are built there).
- **Modify:** `crates/fast-mesh-scene/src/fields.rs`. Split `update_field` into `compute` (pure, `&self` inputs, returns a plan of new values) and `apply` (writes, reports). `run_eager_fields` / `update_fields` compute every due field via `map_chunks` over the due list (threshold: total candidate count ≥ `PAR_MIN_FIELD_WORK`), then apply in field order.

**Interfaces:** no public change. Internal `FieldPlan` or similar, chosen by the builder.

- **Borrowing:**
  - `prepare` (BVH updates, `SignData` rebuild, which needs `&mut Mesh`) stays sequential, before compute.
  - Compute borrows meshes immutably.
  - `SignData::build` parallelises internally (per-vertex pseudonormals, per-edge normals, per-node winding moments) via `map_range` over `PAR_MIN_SIGN_ELEMS`.

- [ ] **Step 1: Write failing tests:**
  - fields: `signed_distances_bit_identical` (1M-triangle-scale points: 300k sample points against a 200k mesh, k ∈ {1, 2, 8} and feature-off reference).
  - fields: `sign_build_bit_identical` (Pseudonormal and Winding).
  - scene: `parallel_field_updates_report_in_order`. Six meshes with 30 directed fields; drag one. Reports and values are identical to the feature-off run, in the same order.
  - fields and scene: `small_inputs_stay_sequential`.
- [ ] **Step 2: Run** the tests with the feature. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full gate. Expected: PASS, including existing `fields_properties` with the feature.
- [ ] **Step 5: Bench.** The field drag benches and `sign_build_*`, with and without the feature. Note them.
- [ ] **Step 6: Commit** `perf(fields): parallel distances and sign data` and `perf(scene): compute due fields in parallel, apply in order`.

### Task 3: Sections, batch queries and self-intersection pairs

**Files:**
- **Modify:** `crates/fast-mesh-scene/src/sections.rs`. Split compute and apply: entries due in one `run_sections` / `update_sections` compute their slices via `map_chunks` above `PAR_MIN_SECTION_WORK` (estimated by triangle count of the meshes involved), then apply in order.
- **Modify:** `crates/fast-mesh-query/src/closest.rs` and `ray.rs` (`closest_point_many`, `raycast_many`: per-query parallel above `PAR_MIN_QUERIES`).
- **Modify:** `crates/fast-mesh-query/src/selfpairs.rs` (or wherever `self_intersecting_pairs` lives). Parallel over top-level subtree pairs above `PAR_MIN_SELF_TRIS`, then concatenate in a deterministic order and sort as today.

- [ ] **Step 1: Write failing tests:**
  - `sections_bit_identical`: 4 sections × 3 meshes, plane drag; compared across thread counts and a feature-off reference.
  - `batch_queries_bit_identical`
  - `self_pairs_bit_identical` (crossed dense meshes)
  - `small_inputs_stay_sequential`
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Bench.** Section drag, the query batch benches, and `analyze` with self-intersection, with and without the feature.
- [ ] **Step 6: Commit** `perf(scene): parallel section updates` and `perf(query): parallel batch queries and self-intersection pairs`.

### Task 4: Repair: analyze passes, weld grid, fairing

**Files:**
- **Modify:** `crates/fast-mesh-ops/src/analyze.rs`. Run the independent passes concurrently with `join`, above `PAR_MIN_ANALYZE_TRIS`:
  - edge map + loops + components;
  - coincident groups;
  - degenerate and duplicate scans;
  - self-intersection.
- **Modify:** `crates/fast-mesh-ops/src/weld.rs` (cell assignment and sort keys built in parallel chunks, then the same deterministic sort).
- **Modify:** `crates/fast-mesh-ops/src/fair.rs`.
  - Solve the x, y and z systems concurrently (`join` of three).
  - Use a parallel SpMV and transpose product above `PAR_MIN_FAIR_ROWS`, as a gather-based transpose so there are no scatter races.
  - Chunked dot products with a fixed chunk size in both builds.
- **Refresh:** the sphere accuracy tests' values if the last bits move. The accuracy bounds stay.

- [ ] **Step 1: Write failing tests:**
  - `analyze_bit_identical`
  - `weld_bit_identical`
  - `fill_faired_bit_identical` (fair on, 3 holes, k ∈ {1, 2, 8} and a feature-off reference)
  - `small_inputs_stay_sequential`
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full gate. Expected: PASS, including the ops pipeline property test with the feature.
- [ ] **Step 5: Bench.** `crates/fast-mesh-ops/benches/ops.rs` with and without the feature.
- [ ] **Step 6: Commit** `perf(ops): parallel analyze, weld and fairing`.

### Task 5: Scene benches, threshold tuning, docs

**Files:**
- **Modify:** `crates/fast-mesh-scene/benches/scene.rs`. Add:
  - `many_mesh_drag_1M`: 6 × 1M meshes, 30 directed fields, each mesh coloured by its field to the next by index, dragging one with 64 poses.
  - `many_mesh_drag_30k`: the same with 6 × 30k.
- **Modify:** threshold constants (tuned).
- **Modify:** `docs/benchmarks.md`. Add a "Parallel" section with core and thread count, plus feature on/off for every bench touched in Tasks 1–4 and the two new ones.
- **Modify:** the spec. Add "§8 Implementation notes" with rulings, deviations and measured numbers.

- [ ] **Step 1: Add** the two benches.
- [ ] **Step 2: Run** every affected bench with and without the feature. Tune the thresholds until `many_mesh_drag_30k` (parallel) ≤ 1.05 × sequential. Record real numbers only.
- [ ] **Step 3: Write** the docs.
- [ ] **Step 4: Run** the full gate. Expected: green.
- [ ] **Step 5: Commit** `bench(scene): many-mesh drag benches`, `perf(parallel): tune thresholds`, and `docs(parallel): record benchmarks and implementation notes`.
