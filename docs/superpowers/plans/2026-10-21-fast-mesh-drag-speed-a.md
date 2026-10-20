# Wide-range drag colouring, part A (exact speed-ups and interruptible recompute) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the per-pose recompute of distance fields during a wide-range drag at least 3× faster, single-threaded, with no change to any value bit. Make it run in steps that the Worker can abandon when a newer pose arrives. Colours always show the complete result for exactly one pose.

**Architecture:**
- **Profile first.** Measure where the 1M wide drag spends its time.
- **Warm start.** Each field caches, per vertex, the target triangle that was closest at the last update. The next query is seeded with that triangle's distance at the new pose, which is an exact upper bound, plus any other exact win the profile points to.
- **Steppable updates in `fast-mesh-scene`.** A scene can defer its Eager field runs and drive them in steps: prepare all due fields, compute their candidate values in chunks under a budget, then apply all of them at once. A step update can be abandoned and leaves no trace.
- **Worker abandonment.** The Worker engine uses this to drop a stale recompute at a step boundary, following spec §6.1.

**Tech Stack:** Rust 2024, stable, MSRV 1.88; criterion; wasm-bindgen `=0.2.129`; TypeScript Worker engine (`packages/fast-mesh-wasm/src/worker/engine.ts`); vitest.

**Spec:** `docs/superpowers/specs/2026-10-20-fast-mesh-drag-speed-design.md`. §6 overrides earlier sections. Part B (threads) gets its own plan after this one.

## Global Constraints

- **Exactness.** Warm and cold give bit-identical values, colours and change records. Stepped and one-shot `update_fields` give bit-identical values, colours and change records. That covers both sign modes, finite and infinite `maxDistance`, and edits.
- **One pose at a time.** All fields due at a pose are stepped as one unit and applied together. Abandoning drops the whole unit, leaving values, colours, change records and hooks exactly as they were.
- **Abandon rule (§6.1).** At a step boundary the Worker abandons the unit when a queued request is any of these:
  - a newer drag move or `setPlacement` of the coloured mesh or the target of any field in the unit;
  - an edit, repair or compact of either mesh;
  - `undo` or `redo`;
  - removal of either mesh or of the field.

  Every other request waits until the unit completes. 9a's arrival order and "one result per batch, in order" stay.
- **Target (§6.3).** The native, single-threaded, 1M wide drag (`maxDistance` 8, the 9a "wide" setup) is at least 3× faster per pose than today. If it misses, record the profile and the result, and continue.
- **Scope (§6.4).** Only the per-pose recompute of due fields. Field builds, sign-data builds, edits and `addMesh` stay in #48. The PR says "part of #48".
- **API stability.** `update_fields`, `closest_point` and every existing public signature are unchanged. New API is additive.
- **No `parallel`-feature dependence.** Everything works with the feature off and on, with bit-identical results either way.
- **Gate.** Every task ends green on:
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`, and the same with `--features parallel` on `fast-mesh-scene`
  - `cargo test --workspace`, and `cargo test -p fast-mesh-scene --features parallel`
  - `pnpm build`
  - `pnpm --filter fast-mesh-wasm test`
  - `pnpm check:pkg`
  - both example builds (`fast-mesh-wasm-example-webgpu`, `fast-mesh-wasm-example-react`)
  - a final `pnpm build`

## Review Focus

1. **Equal-distance ties.** A vertex equidistant from two target triangles must get the same value, sign included, warm or cold. Owned by Task 2: `warm_ties_match_cold`.
2. **A cached triangle that no longer exists or has moved.** A target edit removes or moves the cached triangle. The warm path must never read a dead triangle, and must still match cold. Owned by Task 2: `warm_after_target_edit`.
3. **Abandon, then a different change.** Abandon a unit at pose P1, then apply an edit, then update. The result equals a cold one-shot at the final state, and nothing from P1 leaks. Owned by Task 3: `abandon_then_edit_matches_one_shot`.
4. **Other mutating calls while a unit is open.** Any scene mutation while a step unit is open, for example `set_placement` of an unrelated mesh, abandons the unit first instead of computing against changed inputs. Owned by Task 3: `mutation_abandons_open_unit`.
5. **Drag end during a pending recompute.** In the Worker, `drag.end()` arrives while a unit is mid-step. The final pose is computed to completion and statuses settle `ready`. Owned by Task 4: `drag_end_completes_final_pose`.

---

### Task 1: Wide-drag benchmark and profile

**Files:**
- Modify: `crates/fast-mesh-scene/benches/scene.rs`. Add `bench_wide_drag`, registered in `benches()`.
- Create: `docs/superpowers/notes/2026-10-21-wide-drag-profile.md`.

**Interfaces:**
- **Produces:** a criterion group `scene wide drag 1M` with `field on A` and `fields on A and B`. It uses the same 64 poses as `bench_fields`, with `maxDistance` 8.0, so it matches the 9a JS "wide" case. 10M runs only under `FAST_MESH_BENCH_10M=1`; reuse the env var if one exists, otherwise add it.
- **Produces:** a profile note. It lists the share of time spent in each of these, per pose, at 1M:
  - candidate search (`triangles_near`);
  - closest-point queries;
  - sign tests (pseudonormal and winding);
  - apply/recolour;
  - other.
  It names the top exact optimisations to make in Task 2, and gives the baseline median per pose for both variants. Tasks 2 and 5 compare against that baseline.

- [ ] **Step 1: Add `bench_wide_drag`** next to `bench_fields`, reusing `field_scene` and `poses`.
- [ ] **Step 2: Run it.** Run `cargo bench -p fast-mesh-scene --bench scene -- "wide drag"` and record the medians.
- [ ] **Step 3: Profile one variant at 1M with a sampling profiler.**
  - Use what is available on Windows, such as `cargo flamegraph` or Superluminal.
  - If none is available, use temporary `Instant` counters around the phases in a scratch branch of the code. Do not commit them.
  - Write the note: percentages, baseline numbers, and a ranked list of exact optimisations with expected gains.
- [ ] **Step 4: Commit** `bench(scene): wide-range drag benchmark and profile`.

### Task 2: Warm start and the profile's other exact optimisations

**Files:**
- Modify: `crates/fast-mesh-query/src/closest.rs`, adding a seeded closest-point variant (crate-public or `#[doc(hidden)] pub`).
- Modify: `crates/fast-mesh-fields/src/distance.rs`, adding a seeded `signed_distances` variant that takes and updates hints.
- Modify: `crates/fast-mesh-scene/src/fields.rs`, adding a per-field hint cache that is maintained through edits and used in `compute_field`.
- Test: in-module tests in each crate, plus a scene property test in `fields.rs`.

**Interfaces:**
- **Produces, `fast-mesh-query`:**
  ```rust
  pub fn closest_point_seeded(mesh: &Mesh, bvh: &Bvh, p: [f32; 3], opts: &ClosestOptions, seed: Option<TriangleId>) -> Result<Option<ClosestHit>>
  ```
  Same result as `closest_point`, bit for bit, including the chosen feature on ties. `seed` is a live triangle whose distance at `p` bounds the search. A dead or out-of-range seed is ignored.
- **Produces, `fast-mesh-fields`:**
  ```rust
  pub fn signed_distances_seeded(mesh_b, bvh_b, sign, points_in_b, max_distance, hints: &mut [u32], out) -> Result<()>
  ```
  `hints[i] == u32::MAX` means no hint. On return, each hint is the closest triangle, or `u32::MAX` when far. Its values equal `signed_distances` bit for bit.
- **Produces, `fast-mesh-scene`:** each field keeps `hints: Vec<u32>` sized to A's vertex slots. The hints are cleared for a vertex whose slot is reused, and all of them are cleared when B's topology changes (triangles added or removed). That is safe and simple; a remap is optional.

- [ ] **Step 1: Write the failing tests.**
  - `closest_seeded_matches_cold` (query): random meshes, points and seeds, including dead and far seeds. Assert `closest_point_seeded == closest_point` field by field: distance bits, point bits, triangle and feature.
  - `warm_ties_match_cold` (Review Focus 1): a point exactly equidistant from two triangles (two coplanar triangles sharing an edge, and a symmetric "V"). Seed each triangle in turn and assert the result equals cold.
  - `seeded_signed_matches` (fields): over random points in both sign modes, `signed_distances_seeded` gives the same `out` bits as `signed_distances`. The hints after a run name a triangle whose distance equals `|out|`.
  - `warm_drag_matches_cold` (scene property test): random drags of A and B and random edits of both, with finite and infinite `maxDistance`, in both sign modes. After each update, the values, colours and `take_changes()` bits equal those of a twin scene that has the hint cache disabled. Use a `#[cfg(test)]` switch.
  - `warm_after_target_edit` (Review Focus 2): delete and move triangles of B that are cached hints, update, and assert the result matches the twin.
- [ ] **Step 2: Run** `cargo test -p fast-mesh-query -p fast-mesh-fields -p fast-mesh-scene`. Expected: FAIL (the functions do not exist).
- [ ] **Step 3: Implement.**
  - **Seeded query.** Compute the seed's exact point-triangle distance `d0` with the same routine as the cold path. Run the existing traversal with its initial best set to `min(d0, max_distance)`. Keep the seed as the current best only if the cold path's tie-breaking would pick it. Otherwise reproduce the cold tie-break, for example by keeping the bound but not the candidate, so the traversal re-finds the winner. State the tie rule in a comment.
  - **Scene.** Use the hints in `compute_field` for every candidate, and write them back in `apply_field`. Hints are field state: abandoned and failed updates must not change them.
  - **Profile wins.** Apply any other exact optimisation that Task 1's note ranks above warm start, with a test for each that compares against the unoptimised path.
- [ ] **Step 4: Run** the tests, then `cargo bench -p fast-mesh-scene --bench scene -- "wide drag"`. Record the new medians against Task 1's baseline in the profile note.
- [ ] **Step 5: Commit** `perf(fields): warm-started distance fields` (plus a separate commit per other optimisation).

### Task 3: Deferred Eager fields and step updates in `fast-mesh-scene`

**Files:**
- Modify: `crates/fast-mesh-scene/src/fields.rs` and `crates/fast-mesh-scene/src/scene.rs` (the `run_eager_fields` call sites at `scene.rs:446` and `:565`).
- Test: in-module tests in `fields.rs`.

**Interfaces:**
- **Produces:**
  ```rust
  pub enum FieldBudget { Points(usize) }            // candidate vertices per step; extend later
  pub enum StepState { Idle, Pending, Done }       // Done: the last step applied the unit
  impl Scene {
      pub fn set_defer_fields(&mut self, on: bool);  // Eager field runs are left due instead of run
      pub fn field_step(&mut self, budget: FieldBudget) -> Result<StepState>;
      pub fn abandon_field_step(&mut self);          // drops an open unit; no-op when Idle
      pub fn field_step_pending(&self) -> bool;      // a unit is open
      pub fn field_step_meshes(&self) -> Vec<MeshId>; // the on/to meshes of the open unit, for the abandon rule
  }
  ```
- **Behaviour:**
  - With `set_defer_fields(true)`, the two `run_eager_fields` sites only leave fields due. `update_fields()` still works and runs to completion.
  - The first `field_step` call opens a unit. It collects every due field that is not Lazy or Manual, as `run_fields(Eager)` does, runs `prepare_field` for each, and computes candidate lists.
  - Each `field_step` computes the values of up to `budget` candidates, across the unit's fields in field order, into the unit's buffers.
  - When every value is computed, `field_step` applies all fields as `update_due` does and returns `Done`. Values, colours and change records are bit-identical to one-shot `update_fields`.
  - Any `&mut self` scene call other than `field_step`, `abandon_field_step` and `set_defer_fields` first abandons an open unit. Read-only calls do not.
  - An error during a step abandons the unit and is returned. The fields stay stale and due, as with `run_eager_fields`.

- [ ] **Step 1: Write the failing tests.**
  - `stepped_matches_one_shot`: random drags and edits on two scenes, one stepped with `Points(1)`, `Points(37)` and `Points(usize::MAX)`, the other with `update_fields`. Assert bit-identical values, colours and `take_changes()`, both with and without the `parallel` feature.
  - `abandon_leaves_no_trace`: open a unit, step partway, abandon. Assert values, colours, hints and `take_changes()` are unchanged, and that `update_fields()` afterwards equals the twin.
  - `abandon_then_edit_matches_one_shot` (Review Focus 3).
  - `mutation_abandons_open_unit` (Review Focus 4): with a unit open, `set_placement` of an unrelated mesh abandons it (`field_step_pending() == false`), and the next steps compute from the new state.
  - `defer_leaves_fields_due`: with deferral on, `set_placement` leaves `field_status(id).current == false` and does no work. Count it with the existing test counter, `take()`.
- [ ] **Step 2: Run** `cargo test -p fast-mesh-scene`. Expected: FAIL.
- [ ] **Step 3: Implement.** Split `compute_field` so that candidate selection and per-candidate value computation are separate, and the value loop can resume from a cursor. Keep `update_due` as the one-shot path over the same pieces.
- [ ] **Step 4: Run** `cargo test -p fast-mesh-scene` and `cargo test -p fast-mesh-scene --features parallel`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(scene): deferred Eager fields and step updates`.

### Task 4: Wasm bindings and Worker abandonment

**Files:**
- Modify: `crates/fast-mesh-wasm/src/scene.rs`: bindings `setDeferFields`, `fieldStep(points)` (returns 0 Idle, 1 Pending, 2 Done), `abandonFieldStep`, `fieldStepMeshes()` (returns `Uint32Array` of mesh ids).
- Modify: `packages/fast-mesh-wasm/src/worker/engine.ts`: turn deferral on at start-up and step the unit; `test/harness/node-worker.mjs` if needed.
- Test: `packages/fast-mesh-wasm/test/worker-drag-speed.test.ts`.

**Interfaces:**
- **Consumes:** Task 3's API.
- **Behaviour in `engine.ts`:**
  - After running the request or requests of a batch that leave fields due, the engine steps with `fieldStep(STEP_POINTS)`. `STEP_POINTS` is a constant, default 4096, and is overridable through `EngineOptions.stepPoints` for tests.
  - It calls `#maybeYield()` between steps, using 9a's 4 ms budget.
  - **At each step boundary**, it peeks at the queue. If any queued request matches the abandon rule (Global Constraints) for `fieldStepMeshes()`, it calls `abandonFieldStep()` and continues draining. The next batch that leaves fields due opens a new unit. If nothing matches, it keeps stepping, and other requests wait.
  - **The result** of the batch that completes the unit carries the colour ranges and `field`/`display` statuses, using 9a's existing packing. A batch whose unit was abandoned carries no colour ranges for it. Its `display:<id>` keys stay `computing` until the unit that supersedes it completes.
  - **Scene semantics stay the same:** `update_fields` (fetch, queries) still forces completion when a query needs current fields. If a unit is open, a query first runs the unit to completion.

- [ ] **Step 1: Write the failing tests** in `worker-drag-speed.test.ts`. Use the Node harness, two 1M-free small terrains with a wide `maxDistance`, and `stepPoints: 16` so units span many steps.
  - `stale_pose_is_abandoned`: send 10 drag moves, each delivered after the previous one's first step (use the engine's `onChunk` hook). Assert:
    - exactly one unit completes;
    - its colours equal a fresh `Scene` at the final pose (`debugDump` colours);
    - no result carries colours from an earlier pose.
  - `unrelated_requests_wait`: during a unit, queue a raycast and a colouring change on a third mesh. Assert they are answered after the unit completes, in arrival order, and that the unit was not abandoned.
  - `edit_abandons_unit`: an edit of the coloured mesh queued mid-unit abandons it. The final colours equal a fresh scene after the edit.
  - `drag_end_completes_final_pose` (Review Focus 5): `end()` mid-unit. The final pose completes, and `status("display:<id>")` settles `ready`.
  - `statuses_stay_computing_while_abandoned`: between an abandon and the completing unit, `display:<id>` reads `computing`.
- [ ] **Step 2: Run** `pnpm --filter fast-mesh-wasm test`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** the bindings and the engine changes.
- [ ] **Step 4: Run** the full gate. Expected: PASS. The existing worker-client, worker-lifecycle and store tests stay green.
- [ ] **Step 5: Commit** `feat(worker): step and abandon stale field recomputes`.

### Task 5: Benchmarks, docs and notes

**Files:**
- Modify: `packages/fast-mesh-wasm/bench/scene.bench.mjs` (Worker wide drag: median per pose, and time from last move to colours applied, at 1M and 10M).
- Modify: `docs/benchmarks.md` (a new "Wide-range drag" section: the native before/after table from Tasks 1 and 2, and the Worker latency table).
- Modify: the spec, adding §8 "Part A implementation notes": rulings, numbers, and whether the 3× target was met.
- Modify: README (one paragraph under the Worker section: wide-range colouring is stepped and stale poses are dropped, and what latency to expect).
- Create: `.changeset/<name>.md` (patch bump for `fast-mesh-wasm`: faster wide-range distance colouring and dropping stale drag poses in the Worker scene).

- [ ] **Step 1: Run** the native bench (`cargo bench -p fast-mesh-scene --bench scene -- "wide drag"`) and the Worker bench (`pnpm --filter fast-mesh-wasm bench -- --worker`, plus `--10m`) on a release build. Record real numbers only.
- [ ] **Step 2: Write** the docs, the spec notes and the changeset.
- [ ] **Step 3: Run** the full gate. Expected: green.
- [ ] **Step 4: Commit** `bench(worker): wide-range drag latency` and `docs(perf): part A results`.
