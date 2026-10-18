# fast-mesh Scene (Phase 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a new crate `fast-mesh-scene` that owns meshes and their BVHs. It provides rigid world placements, per-mesh update modes, world-space queries, broadphase contacts, change reports, and one undo history whose steps can span several meshes.

**Architecture:** `fast-mesh-store` gains two public helpers: `Patch::release` and `ChangeSet::merge`. The Scene keeps a generational slot map of `Entry { mesh, bvh, placement, mode, pending: Option<ChangeSet>, report }`, a step-based history of `Edit(MeshId, Patch)` / `Placement(MeshId, old, new)` entries, and world-space wrappers over Phase 3a/3b.

**Tech Stack:** Rust 2024 (MSRV 1.88, toolchain 1.99.0), proptest 1.12, criterion 0.8 with no default features.

**Spec:** `docs/superpowers/specs/2026-10-18-fast-mesh-scene-design.md`. Its §15 overrides earlier sections where they differ.

## Global Constraints

- `fast-mesh-scene` depends only on `fast-mesh-store`, `fast-mesh-bvh` and `fast-mesh-query`. No new third-party runtime dependencies.
- Mesh data and history operations are all-or-nothing. If an Eager BVH update fails, the BVH is left pending (§15.1).
- World inputs and outputs are f64. f64 bounds become Phase 3 f32 options rounded outward, and results are re-filtered in f64 (§15.9).
- Placements are validated with the `Rigid` rule (orthonormal, det +1 within `1e-6`, finite). Relative rotations are re-orthonormalised in f64 before Phase 3b calls (§15.2).
- Stale `MeshId`s give `SceneError::UnknownMesh`. A stale Manual BVH gives `SceneError::StaleBvh(id)` and is never skipped silently.
- `out` vectors are appended to and truncated back on error. Ties are deterministic (lower `MeshId` index first).
- Every public item has `///` docs. clippy `-D warnings`, fmt, rustdoc with no warnings, and the wasm32 build must all pass.
- Files are UTF-8 with LF. Edit with the editor tools. Commits follow Conventional Commits and end with `Co-Authored-By: <model> <noreply@anthropic.com>`.

## Review Focus

1. **Ctrl+Z mid-drag.** `undo` while a step is open closes the step and undoes the whole drag. Test `undo_closes_open_step` (Task 4).
2. **Undo across meshes with a failure.** An injected OOM in the second mesh's revert leaves both meshes and the history unchanged. Test `step_undo_is_atomic` (Task 4).
3. **Large-coordinate placements.** Meshes with origins around 1e6 and translations around 1e6 still give exact relative transforms for integer data. Test `relative_rigid_is_exact_for_integer_placements` (Task 2).
4. **Removed or compacted mesh in history.** Undo after `remove_mesh` or `compact(id)` skips that mesh's dropped entries and still undoes the others. Test `history_survives_remove_and_compact` (Task 4).
5. **Lazy mesh queried while brushing.** A query brings a Lazy BVH up to date, and a stale Manual one makes the query fail. Test `queries_update_lazy_and_reject_stale_manual` (Task 5).

---

### Task 1: Store helpers, `Patch::release` and `ChangeSet::merge`

**Files:**
- Modify `crates/fast-mesh-store/src/patch.rs`: add `PatchSide`, `Patch::release`, `ChangeSet::merge`.
- Modify `crates/fast-mesh-store/src/history.rs`: replace the private `release` with `Patch::release`. Behaviour is unchanged.
- Modify `crates/fast-mesh-store/src/lib.rs`: re-export `PatchSide`.

**Interfaces (produces):**
- `pub enum PatchSide { Undoable, Redoable }`
- `impl Patch { pub fn release(&self, mesh: &mut Mesh, side: PatchSide) }`
  - Same uid and epoch check as today's private `release`.
  - `Undoable` releases `vertices_removed`/`triangles_removed`.
  - `Redoable` releases `vertices_added`/`triangles_added`.
- `impl ChangeSet { pub fn merge(&mut self, later: &ChangeSet) }`: applies the net-effect rules of spec §15.7. All lists stay sorted and disjoint.

**Steps:**
- [ ] Write the failing tests.
  - `release_matches_history_eviction`: a slot freed by a patch is reused only after `release(Undoable)`.
  - `merge_matches_replay`: proptest. Random commit sequences on a mesh. Merge each step's `ChangeSet` and compare with a brute-force classification of each slot (alive before or after, positions or vertex ids changed). Assert the lists are sorted, disjoint and correctly classified.
  - Existing history tests must still pass.
- [ ] Run `cargo test -p fast-mesh-store` and confirm the new tests fail.
- [ ] Implement.
- [ ] Run `cargo test -p fast-mesh-store`, clippy and fmt.
- [ ] Commit: `feat(store): add Patch::release and ChangeSet::merge`.

---

### Task 2: Crate skeleton, registry, placements and transforms

**Files:**
- Modify `Cargo.toml`: add the workspace dependency `fast-mesh-query = { path = "crates/fast-mesh-query", version = "0.1.0" }`.
- Create `crates/fast-mesh-scene/Cargo.toml`, modelled on `fast-mesh-query`'s. Dependencies: store, bvh, query. Add the dev-dependencies and `[[bench]] name = "scene"`.
- Create `src/lib.rs`, `src/error.rs`, `src/id.rs` (generational slot map), `src/placement.rs`, `src/scene.rs`.
- Create `benches/scene.rs` as a stub with one empty criterion group.

**Interfaces (produces):**
- `SceneError` from spec §11. It implements `Display`, `Error`, and `From` for `StoreError`/`BvhError`/`QueryError`. `pub type Result<T>`.
- `MeshId` (`Copy, Eq, Hash, Debug, Ord` by `(index, generation)`).
- `Placement { rotation, translation }`, with `Placement::IDENTITY`, `validate()`, `apply(p: [f64;3]) -> [f64;3]`, `inverse_apply`.
- `pub(crate) fn relative_rigid(a: &Placement, origin_a: [f64;3], b: &Placement, origin_b: [f64;3]) -> Rigid`. Use the formula in spec §5, then Gram–Schmidt re-orthonormalisation of R (§15.2).
- `pub(crate) fn world_to_local(p: &Placement, origin: [f64;3], q: [f64;3]) -> [f64;3]` and `local_to_world(...)`.
- `UpdateMode { Eager, Lazy, Manual }` (default `Eager`).
- `MeshOptions { build, bvh_policy, mode, placement }`, with a `Default`.
- `SceneOptions { history_budget: usize }`. Default: 256 MiB.
- On `Scene`: `new`, `add_mesh`, `remove_mesh` (history interplay comes in Task 4; for now it just removes), `mesh`, `ids`, `placement`, `set_placement` (history recording comes in Task 4), `is_current`.

**Steps:**
- [ ] Write the failing tests.
  - `stale_ids_are_rejected`
  - `ids_iterate_in_index_order`
  - `placement_validation` (the `Rigid` cases)
  - `relative_rigid_is_exact_for_integer_placements`: origins about 1e6, 90° rotations, integer translations. Transforming B's vertices with the relative `Rigid` must equal the hand-computed world→A-local values exactly.
  - `relative_rotation_of_near_orthonormal_inputs_validates`: two placements each 0.9e-6 off. The relative `Rigid` passes `Rigid::validate`.
  - `world_local_round_trip`
- [ ] Fail. Implement. Pass, with clippy and fmt.
- [ ] Commit: `feat(scene): add fast-mesh-scene with registry and rigid placements`.

---

### Task 3: Editing, update modes, change reports, maintenance

**Files:** `src/scene.rs`, plus a new `src/report.rs` if it helps.

**Interfaces (produces):**
- `Scene::edit<R>(id, f: impl FnOnce(&mut Transaction) -> fast_mesh_store::Result<R>) -> Result<R>`. It commits, and after the commit:
  - pushes the patch to the history (Task 4 hooks in here; for now keep it in a placeholder that Task 4 replaces);
  - merges the `ChangeSet` into the mesh's pending changes and report;
  - updates the BVH when the mode is Eager (§15.1 on failure).
- `set_mode`, `update(id)`, `flush()`, `is_current(id)`.
- `take_changes() -> Vec<MeshChanges>` (§15.6).
- `add_channel(id, desc)`, `remove_channel(id, name)`, `derived_channel_mut(id, name) -> Result<&mut [f32]>`.
- `compact(id) -> Result<Remap>`. It rebuilds the BVH and reports `changes: None`. Its history interplay comes in Task 4.

**Steps:**
- [ ] Write the failing tests.
  - `eager_bvh_stays_current`
  - `lazy_waits_for_flush`
  - `manual_waits_for_update`
  - `pending_changes_merge_across_commits`: Lazy across 5 commits, then `flush`. The BVH validates and matches a fresh build in queries.
  - `eager_update_failure_leaves_bvh_pending`: inject an OOM into `Bvh::update` (use `fast_mesh_bvh`'s test hook through a `#[cfg(test)]` feature if needed, or provoke it with a test-only flag in the Scene). The edit succeeds and the next `update` succeeds.
  - `take_changes_reports_and_resets`
  - `added_then_removed_is_not_reported`
  - `compact_reports_everything`
- [ ] Fail. Implement. Pass.
- [ ] Commit: `feat(scene): add editing, update modes and change reports`.

---

### Task 4: Steps and history

**Files:** a new `src/history.rs`, and `src/scene.rs`.

**Interfaces (produces):**
- `begin_step`, `end_step`, `cancel_step`, `undo`, `redo`, `can_undo`, `can_redo`, `clear_history` (§7, §15.3).
- Budget eviction with `Patch::release(.., Undoable)`. When the redo stack is dropped, release with `Redoable`.
- `remove_mesh` and `compact` drop that mesh's entries (§15.5, §15.10). `remove_mesh` releases its entries' slots.
- `set_placement` records `Placement(id, old, new)`.

**Steps:**
- [ ] Write the failing tests.
  - `step_groups_edits_across_meshes`
  - `implicit_steps_per_operation`
  - `undo_closes_open_step`
  - `cancel_step_reverts_open_entries`
  - `new_step_clears_redo`
  - `step_undo_is_atomic`: inject an OOM into the second mesh's apply or revert. Both meshes and the stacks are unchanged.
  - `budget_evicts_oldest_steps_and_releases_slots`
  - `history_survives_remove_and_compact`
  - `undo_updates_bvhs_per_mode`
- [ ] Fail. Implement. Pass.
- [ ] Commit: `feat(scene): add step-based undo across meshes`.

---

### Task 5: World queries, broadphase and contacts

**Files:** new `src/query.rs` and `src/contacts.rs`.

**Interfaces (produces):** spec §8 and §9, with §15.4 and §15.9:
- `raycast`
- `closest_point`, `closest_point_on`
- `intersects`, `mesh_distance`, `triangles_near`
- `find_contacts(margin, &ContactFilter, &mut Vec<Contact>) -> Result<usize>`
- `pub(crate) fn world_box(entry) -> Aabb64`: conservative f64 box from the BVH root box placed by the transform.
- `SceneRayHit`, `SceneClosestHit`, `SceneMeshDistance` (world points), `Contact`, `ContactFilter`.

**Steps:**
- [ ] Write the failing tests.
  - Each query must equal the Phase 3 query called directly, with a hand-computed local transform, on 2–3 placed meshes (integer and rotated placements).
  - `raycast_picks_nearest_mesh_and_breaks_ties_by_id`
  - `queries_update_lazy_and_reject_stale_manual`
  - `find_contacts_matches_all_pairs`, against brute-force Phase 3b calls per pair.
  - `bounds_round_outward`: a hit exactly at `max_t` (f64) is kept, and a hit just beyond it is dropped.
- [ ] Fail. Implement. Pass.
- [ ] Commit: `feat(scene): add world-space queries and contacts`.

---

### Task 6: Model-based property tests, benchmarks, docs

**Files:**
- `crates/fast-mesh-scene/tests/properties.rs`
- `benches/scene.rs`
- `docs/benchmarks.md` (new "Scene" section)
- `.github/workflows/ci.yml`: add `cargo bench -p fast-mesh-scene --no-run`
- `CONTRIBUTING.md`: layout line and bench row
- Spec status set to Implemented, with notes

**Steps:**
- [ ] Property test `scene_matches_model`:
  - 64 cases by default; honour `PROPTEST_CASES`.
  - 2–3 meshes; random edits, placements, open and close step, cancel, undo, redo, mode changes, flush, remove, compact.
  - After each operation, check mesh data and placements against the model. Eager BVHs must be current and valid. After `flush`, every non-Manual BVH must be current.
  - A random world raycast and `mesh_distance` must equal direct Phase 3 calls.
- [ ] Run it, then run it with `PROPTEST_CASES=2000` in release.
- [ ] Benches (spec §13):
  - Scene overhead against direct calls for a raycast, `mesh_distance`, and `set_placement`+`mesh_distance` (two 1M terrains);
  - `find_contacts` over 2, 8 and 32 small meshes.
  - Record the results in `docs/benchmarks.md`.
- [ ] Full verification:
  - fmt
  - clippy `-D warnings`
  - `cargo test --workspace`
  - wasm32 build
  - rustdoc `-D warnings`
  - `cargo bench -p fast-mesh-scene --no-run`
- [ ] Commit: `test(scene): add model-based property tests, benchmarks and docs`.
