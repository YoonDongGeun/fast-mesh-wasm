# fast-mesh fields and colors, part a (Phase 5a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Signed per-vertex distance fields between scene meshes, recomputed only near the contact, colored through colormaps into RGBA8 buffers. Also closes the three open Phase 4 Scene questions.

**Architecture:**
- The new crate `fast-mesh-fields` holds pure kernels: `SignData` (pseudonormals, fast winding moments), `signed_distances`, and colormaps with `color_of`.
- `fast-mesh-scene` gains:
  - change-report subscribers;
  - local-space accessors;
  - render-dirty reporting;
  - a distance-field registry with per-mode scheduling, the corrected candidate rule, and color buffers.
- `fast-mesh-bvh` gains a dense node index and a generation counter.

**Tech Stack:** Rust 2024 (MSRV 1.88, toolchain 1.99.0), proptest 1.12, criterion 0.8 with no default features.

**Spec:** `docs/superpowers/specs/2026-10-19-fast-mesh-fields-design.md`. Its §11 overrides earlier sections.

## Global Constraints

- `fast-mesh-fields` depends only on store, bvh and query. `fast-mesh-scene` adds `fast-mesh-fields`. No new third-party dependencies.
- Field values live in a derived store channel `field:<name>` (width 1). Far values are `+∞`. Unreferenced alive vertices and dead slots hold NaN.
- `Auto` resolves to `Pseudonormal`. It falls back to `Winding` only for a closed mesh whose orientation is inconsistent (§11.1).
- Readers take `&self` and never update. Updates happen only through `update_fields`/`flush`, or an Eager trigger (§11.2).
- `pad(x) = x·(1 + 1e-6) + 1e-6` is used for `triangles_near` candidates (§11.6).
- Fields and colors are not undoable. The `field:` prefix is reserved.
- Kernels never allocate inside traversal. Operations are all-or-nothing. Results are deterministic.
- `///` docs everywhere. clippy `-D warnings`, fmt, rustdoc and the wasm32 build must be clean. Files are UTF-8 with LF, edited with the editor tools.
- Commits use Conventional Commits and end with `Co-Authored-By: <model> <noreply@anthropic.com>`.

## Review Focus

1. **Penetrating an open sheet from behind** gives a negative value (Auto resolves to Pseudonormal). Test: `open_sheet_penetration_is_negative` (Task 2).
2. **A mesh with flipped faces** that is closed resolves Auto to Winding, and the status reports the inconsistency. Test: `auto_falls_back_on_inconsistent_orientation` (Task 3).
3. **Dragging back out of contact** restores far values (`+∞`) and base colors for vertices that left the range. Test: `leaving_contact_restores_far` (Task 6).
4. **Undo of an edit that removed vertices** turns those slots back to NaN, and revived slots are recomputed. Test: `undo_keeps_field_consistent` (Task 6).
5. **A stale Manual BVH** blocks a field update: `update_fields` returns `StaleBvh`, and `field_status.current` is false. Test: `manual_bvh_blocks_field_update` (Task 6).

---

### Task 1: BVH node index and generation

**Files:**
- `crates/fast-mesh-bvh/src/node.rs`
- `src/tree.rs`
- `src/update.rs`

**Produces:**
- `NodeId::index(self) -> usize`: dense, below `node_slots()`.
- `Bvh::node_slots(&self) -> usize`.
- `Bvh::generation(&self) -> u64`: incremented by `build`, `update` (whenever a node changes), `refit`, `rebuild_region`, `rebuild` and storage compaction.

**Steps:**
- [ ] Write the failing tests:
  - `node_indices_are_dense_and_unique`;
  - `generation_changes_on_every_structural_change` (all the paths above);
  - `generation_unchanged_when_update_is_unchanged`.
- [ ] Implement.
- [ ] Run `cargo test -p fast-mesh-bvh`, clippy and fmt.
- [ ] Commit: `feat(bvh): expose dense node indices and a generation counter`.

### Task 2: `fast-mesh-fields` crate, pseudonormal signs, signed distances

**Files:**
- `Cargo.toml` (workspace dependency `fast-mesh-fields`)
- `crates/fast-mesh-fields/{Cargo.toml, src/lib.rs, src/error.rs, src/sign.rs, src/distance.rs}`
- `benches/fields.rs` (stub)

**Produces:**
- `FieldsError { InvalidInput(&'static str), OutOfMemory, Store(StoreError), Query(QueryError) }` and `Result`.
- `SignMode { Pseudonormal, Winding, Auto }`.
- `SignStatus { mode, closed, consistent_orientation, non_manifold_vertices: usize, degenerate_triangles: usize }`.
- `SignData::build(&mut Mesh, &Bvh, SignMode) -> Result<SignData>`. In this task it supports Pseudonormal, and Auto when it resolves to Pseudonormal. Winding arrives in Task 3, and `Winding`/fallback returns `InvalidInput("winding not yet supported")` until then.
- `SignData::status()` and `SignData::is_current(&Mesh, &Bvh)`, which checks uid, epoch, revision and BVH generation.
- `signed_distances(mesh_b, bvh_b, &SignData, points_in_b: &[[f32;3]], max_distance: f32, out: &mut [f32]) -> Result<()>`. Far points get `+∞`. NaN inputs give `InvalidInput`.

**Steps:**
- [ ] Write the failing tests. Brute force uses an exact f64 generalized winding number over all triangles, on small meshes.
  - `closed_meshes_match_exact_sign`:
    - a cube, an icosphere of about 320 triangles, and a torus;
    - random points;
    - Pseudonormal sign equals the exact-winding sign outside 1e-5·diagonal of the surface;
    - the magnitude equals brute-force closest distance.
  - `open_sheet_penetration_is_negative`: a terrain sheet with normals up. Points below it read negative and points above read positive, with the right magnitude.
  - `degenerate_faces_do_not_break_signs`: a closed mesh with a few zero-area triangles inserted.
  - `far_points_are_infinite`.
  - `status_reports_topology`: open, closed, non-manifold vertices and degenerate triangles.
  - `is_current_tracks_revision_and_bvh_generation`.
- [ ] Implement.
  - Angle-weighted vertex pseudonormals, built from incidence.
  - Edge normals stored per corner, built from adjacency. Note that `Feature::Edge(k)` joins corners k and (k+1)%3.
  - Zero-area triangles are skipped in the accumulation.
  - For a degenerate reported triangle, use the nearest non-degenerate feature (§11.7).
  - Closed means no boundary or non-manifold edges.
  - Orientation is consistent when every shared edge is traversed in opposite directions.
- [ ] Pass the tests, plus clippy, fmt and the wasm32 build of the crate.
- [ ] Commit: `feat(fields): add fast-mesh-fields with pseudonormal signed distances`.

### Task 3: Fast winding numbers and Auto fallback

**Files:** `crates/fast-mesh-fields/src/winding.rs`, `src/sign.rs`

**Produces:**
- Winding support in `SignData` (per-node order-1 dipole moments, indexed by `NodeId::index()`).
- `pub fn winding_number(&self, mesh, bvh, p: [f32;3]) -> f64`, with `beta = 2.0`.
- Auto resolves to Winding for a closed mesh with inconsistent orientation.

**Steps:**
- [ ] Write the failing tests:
  - `fast_winding_matches_exact`: within 1e-3 for points beyond 1% of the diagonal from the surface, on the closed meshes and on a cube with a missing face.
  - `winding_sign_on_closed_meshes`: points inside give w ≈ 1, outside give ≈ 0.
  - `auto_falls_back_on_inconsistent_orientation`: a cube with one flipped face. Auto resolves to Winding, the status shows `consistent_orientation == false`, and the signs stay correct.
  - `moments_rebuild_when_bvh_changes`.
- [ ] Implement. Moments are the area-weighted normal sum and the area-weighted centroid per node. A far node uses the dipole term when `|p − c| > beta·r`; a near leaf uses exact solid angles (Van Oosterom–Strackee).
- [ ] Pass the tests.
- [ ] Commit: `feat(fields): add fast winding numbers and the Auto fallback`.

### Task 4: Colormaps

**Files:** `crates/fast-mesh-fields/src/color.rs`

**Produces:**
- `Colormap { Turbo, Viridis, Coolwarm, Grayscale, Custom(Vec<(f32, [u8;4])>) }`, with 256-entry built-in tables.
- `OutOfRange { Clamp, Base }`.
- `ColorRule { map, range, below, above, center: Option<f32>, base, nan }`.
- `pub fn color_of(rule: &ColorRule, v: f32) -> [u8;4]`.
- `ColorRule::validate() -> Result<()>`. It rejects `range.0 >= range.1`, non-finite ends, a `center` outside the range, and custom stops that are unsorted or outside [0, 1].

**Steps:**
- [ ] Write the failing tests:
  - `range_ends_and_center_map_to_table_positions`;
  - `below_and_above_are_independent` (the two ranges in spec §11.3);
  - `infinite_is_base`, `nan_is_nan_color`;
  - `custom_stops_interpolate`;
  - `invalid_rules_rejected`.
- [ ] Implement and pass.
- [ ] Commit: `feat(fields): add colormaps and color rules`.

### Task 5: Scene subscribers, accessors, render-dirty reports

**Files:** `crates/fast-mesh-scene/src/{scene.rs, report.rs, query.rs, lib.rs}`

**Produces:**
- `SubscriberId`, `subscribe()`, `unsubscribe(id) -> Result<()>`, `take_changes_for(id) -> Result<Vec<MeshChanges>>`. `take_changes()` is the default subscriber. Every subscriber gets the same merged reports.
- `bvh(id) -> Result<&Bvh>`, which returns `StaleBvh` unless current.
- `relative_transform(a, b) -> Result<Rigid>`, mapping B-local into A-local.
- `world_to_local(id, [f64;3]) -> Result<[f64;3]>` and `local_to_world`.
- `MeshChanges` gains `colors_dirty: Option<Vec<VertexId>>` (`None` means all; empty when unchanged), `fields_changed: Vec<FieldId>` (the `FieldId` type is defined here as an opaque `Copy` id for Task 6), and `channels_written: Vec<String>`.
- `derived_channel_mut`, `add_channel` and `remove_channel` report into `channels_written`, and they reject the `field:` prefix.

**Steps:**
- [ ] Write the failing tests:
  - `two_subscribers_each_get_all_changes`;
  - `unsubscribed_id_rejected`;
  - `bvh_accessor_rejects_stale`;
  - `relative_transform_matches_pair_queries`;
  - `channel_writes_are_reported`;
  - `field_prefix_is_reserved`.

  Existing tests must keep passing.
- [ ] Implement and pass. Also run the existing property tests.
- [ ] Commit: `feat(scene): add change subscribers, local accessors and render-dirty reports`.

### Task 6: Distance fields and colors in the Scene

**Files:** `crates/fast-mesh-scene/src/fields.rs` (new), `scene.rs`, `history.rs` hooks if needed, `Cargo.toml`

**Produces:**
- `DistanceField { name, on, to, max_distance: f64, sign: SignMode }`.
- `add_distance_field(desc) -> Result<FieldId>`, `remove_field(id)`, `update_fields() -> Result<()>` (also called by `flush`).
- `field_values(&self, id) -> Result<&[f32]>` and `field_status(&self, id) -> Result<FieldStatus { current, sign: SignStatus }>`.
- `set_coloring(mesh, Option<Coloring>) -> Result<()>`, where `Coloring` is `ColorRule` plus `field`.
- `colors(&self, mesh) -> Result<Option<&[[u8;4]]>>`.

Behaviour (spec §6.1, §11.2, §11.6, §11.10):
- Fields follow mesh `on`'s mode.
- BVHs are prepared with the query rules; a stale Manual BVH leaves the field stale.
- The candidate rule uses `pad`.
- Removed vertices get NaN.
- A `None` report triggers a full recompute.
- Colors are recomputed only for changed values, and the changes are reported in `colors_dirty`.
- `fields_changed` is reported.
- Removing either mesh removes its fields and colorings.

**Steps:**
- [ ] Write the failing tests:
  - `field_matches_from_scratch` after random placements and edits;
  - `small_drag_recomputes_only_near_vertices` (a test counter);
  - `leaving_contact_restores_far`;
  - `undo_keeps_field_consistent`;
  - `manual_bvh_blocks_field_update`;
  - `eager_vs_lazy_field_scheduling`;
  - `colors_follow_values_and_report_dirty`;
  - `one_sided_and_two_sided_coloring`;
  - `compact_recomputes_everything`;
  - `removing_a_mesh_removes_its_fields`;
  - `invalid_field_descs_rejected`: `on == to`, a duplicate name, `max_distance` negative or NaN.
- [ ] Implement and pass.
- [ ] Commit: `feat(scene): add distance fields and colors`.

### Task 7: Property tests, benchmarks, docs

**Files:**
- `crates/fast-mesh-scene/tests/properties.rs` (extend the model, or add `fields_properties.rs`)
- `crates/fast-mesh-fields/benches/fields.rs`
- `crates/fast-mesh-scene/benches/scene.rs`
- `docs/benchmarks.md`
- the spec status
- CI, adding `cargo bench -p fast-mesh-fields --no-run` and the wasm32 build of `fast-mesh-fields`
- CONTRIBUTING (layout line and bench row)

**Steps:**
- [ ] Property test `fields_match_from_scratch`:
  - 64 cases by default, honouring `PROPTEST_CASES`;
  - two small meshes (closed and open variants) with random placements, drags, edits, undo/redo, modes and `max_distance`;
  - one-sided and two-sided coloring.

  After each `update_fields`, check all three:
  - every value equals a from-scratch `signed_distances`;
  - colors equal `color_of`;
  - `colors_dirty` covers every changed color.
- [ ] Run it, then run it with `PROPTEST_CASES=2000` in release.
- [ ] Benchmarks:
  - `SignData::build` at 1M, for pseudonormal and winding;
  - `signed_distances` per point;
  - the 64-pose drag of two touching 1M terrains with the field one-sided and two-sided (`set_placement` + `update_fields` + colors), reporting the candidate counts.

  Record them in `docs/benchmarks.md`.
- [ ] Full verification:
  - fmt;
  - clippy;
  - `cargo test --workspace`;
  - the wasm32 build of the fields and scene crates;
  - rustdoc;
  - `cargo bench --no-run` for both crates.
- [ ] Commit: `test(fields): add property tests, benchmarks and docs`.
