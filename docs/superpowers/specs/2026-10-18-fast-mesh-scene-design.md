# fast-mesh Scene (Phase 4) design

- **Status:** Implemented (Phase 4). The owner skipped the questions; §15 records the builder-check decisions, and "Implementation notes" at the end records where the code differs from the wording above.
- **Date:** 2026-10-18
- **Parent spec:** `2026-10-14-fast-mesh-engine-design.md` §3.3, §4.5 and §7. Where this document disagrees with it, this document wins for Phase 4.
- **Milestone:** Phase 4: Scene

## 1. Intent

The Scene is the mediator the parent spec describes. It owns meshes and their BVHs, places each mesh in the world, keeps every BVH up to date according to that mesh's update policy, answers queries in world coordinates, and keeps one undo history across all meshes.

The owner's use case is the one from Phase 3b. One dental-arch-like mesh is moved or rotated against another; the per-vertex contact colour itself comes in Phase 5. Undo works on steps, where a step is a group of edits and moves, possibly across several meshes, undone as one ("several things treated as one bundle and undone as a unit", the owner's words).

The owner asked to skip the questions, so the decisions below marked *my call* are mine.

## 2. Goals, non-goals, anti-goals

**Goals:**
1. Scene registry: add and remove meshes, with stable ids.
2. World placement as a rigid transform per mesh, in f64. The mesh's f64 origin is handled exactly.
3. BVH ownership and update policies: eager, lazy and manual, plus `flush`.
4. Queries in world coordinates, built on Phase 3a/3b:
   - raycast over all meshes;
   - closest point, against one mesh or all meshes;
   - mesh-vs-mesh `intersects`, `mesh_distance` and `triangles_near`.
5. Broadphase over mesh world boxes, and `find_contacts` across all mesh pairs.
6. Scene-wide history: steps that group edits and transform changes across meshes, with undo and redo of whole steps under a memory budget.
7. Change reporting: what changed since the caller last asked, per mesh, so renderers and the later Phase 5 fields can react.

**Non-goals (later phases):**
- *My call:* the general dependency graph over fields, colours and sections waits for Phase 5/6, when those nodes exist. Phase 4 has exactly one dependency, mesh geometry to its BVH, so a graph engine now would be speculative.
- *My call:* scale in transforms, which Phase 3b does not support. Scale is added in a later phase together with its distance caveats.
- Contact normals and contact points beyond the pair queries' outputs. *My call:* add them when a consumer needs them.
- Sections (Phase 6). JS API and events as callbacks (Phase 8). Worker and sync (Phase 9).

**Anti-goals:**
- A transform change that touches any vertex, BVH or buffer. It must only store a matrix.
- A partially applied undo or redo step. A failure leaves every mesh as it was.
- A query that silently uses a stale BVH.

## 3. Constraints

- **New crate `fast-mesh-scene`.** It depends on `fast-mesh-store`, `fast-mesh-bvh` and `fast-mesh-query`. No new third-party dependencies.
- **Failure atomicity** as in Phases 1–3: an operation that fails, for example on `OutOfMemory`, leaves the Scene unchanged.
- **Phase 3 rules for query outputs:** `out` vectors are appended to and rolled back on error, and ties are deterministic.
- **Docs and checks:** `///` docs on every public item, clippy `-D warnings`, fmt, and the wasm32 build.
- **Tests:** property tests against a model, criterion benches.

## 4. Ids and registry

```rust
pub struct MeshId { index: u32, generation: u32 }   // Copy, Eq, Hash, Debug; stale ids are rejected
pub struct MeshOptions { pub build: BuildOptions, pub bvh_policy: fast_mesh_bvh::UpdatePolicy, pub mode: UpdateMode, pub placement: Placement }
impl Scene {
    pub fn new(options: SceneOptions /* history budget */) -> Scene;
    pub fn add_mesh(&mut self, mesh: Mesh, options: MeshOptions) -> Result<MeshId>;   // builds the BVH
    pub fn remove_mesh(&mut self, id: MeshId) -> Result<Mesh>;
    pub fn mesh(&self, id: MeshId) -> Result<&Mesh>;
    pub fn ids(&self) -> impl Iterator<Item = MeshId>;
}
```

- **Generational ids.** A removed id never matches again; using one gives `UnknownMesh`.
- **Removing a mesh** drops its entries from the history. Steps left empty are dropped. *My call.* Undo of other meshes keeps working.

## 5. Placement (world transforms)

```rust
/// world = rotation * p + translation, where p is the mesh's world-space point
/// (origin + local position), all in f64.
pub struct Placement { pub rotation: [[f64; 3]; 3], pub translation: [f64; 3] }
impl Scene {
    pub fn placement(&self, id: MeshId) -> Result<Placement>;
    pub fn set_placement(&mut self, id: MeshId, p: Placement) -> Result<()>;   // recorded in the current step
}
```

- **Validation:** same rule as `Rigid` (orthonormal, det +1 within 1e-6, finite), giving `InvalidInput`.
- **Cost:** `set_placement` only stores the placement and records the change. Nothing else is touched.
- **Relative transforms:** for mesh-vs-mesh queries, the Scene computes the relative `Rigid` from B's local space to A's local space in f64, origins included.

  ```text
  p_world = R_B (o_B + p_B) + t_B
  p_A     = R_Aᵀ (p_world − t_A) − o_A
  => R = R_Aᵀ R_B,   t = R_Aᵀ (R_B o_B + t_B − t_A) − o_A
  ```

  This is the "convert the operand into local space" rule from parent §6.
- **World to local** for a point or ray uses the same f64 formula, then rounds to f32. World results are reported in f64.

## 6. BVHs and update policies

```rust
pub enum UpdateMode { Eager, Lazy, Manual }   // see §15.8
impl Scene {
    pub fn set_mode(&mut self, id: MeshId, mode: UpdateMode) -> Result<()>;
    pub fn update(&mut self, id: MeshId) -> Result<()>;   // bring one BVH up to date now
    pub fn flush(&mut self) -> Result<()>;               // bring every Lazy BVH up to date
    pub fn is_current(&self, id: MeshId) -> Result<bool>;
}
```

- **Eager** (the default) updates the BVH right after every commit, undo and redo of that mesh.
- **Lazy** accumulates the changes. The BVH is updated before the next query that needs it, or on `flush()`/`update(id)`. This is the policy to use while dragging or brushing.
- **Manual** updates only on `update(id)`. A query that needs a stale Manual BVH returns `SceneError::StaleBvh(id)` and never updates it implicitly.
- **Accumulated changes** are kept as one merged `ChangeSet` per mesh, which `Bvh::update` takes. Merging (*my call*: in `fast-mesh-store`, as `ChangeSet::merge(&mut self, later: &ChangeSet)`) keeps the net effect per slot:
  - added then removed: neither;
  - removed then added again (slot reuse): counts as changed when the slot was alive before, otherwise as added;
  - moved and changed are unions, minus the slots that end up removed.

  Undo and redo produce change sets too, and they merge the same way.

## 7. Editing and history

```rust
impl Scene {
    /// Runs `f` in a transaction on mesh `id`, commits it, and records the patch.
    pub fn edit<R>(&mut self, id: MeshId, f: impl FnOnce(&mut Transaction) -> fast_mesh_store::Result<R>) -> Result<R>;
    pub fn begin_step(&mut self);          // nested calls count; the outermost end closes the step
    pub fn end_step(&mut self) -> Result<()>;
    pub fn undo(&mut self) -> Result<bool>; // false: nothing to undo
    pub fn redo(&mut self) -> Result<bool>;
    pub fn can_undo(&self) -> bool; pub fn can_redo(&self) -> bool;
    pub fn clear_history(&mut self) -> Result<()>;
}
```

**Steps**
- A step is the ordered list of entries recorded between `begin_step` and the matching `end_step`. Entries are `Edit(mesh, Patch)` and `Placement(mesh, old, new)`.
- Outside an explicit step, each `edit` or `set_placement` is its own step.
- An empty step is not recorded.
- A new step clears the redo stack.
- `undo` reverts a step's entries in reverse order; `redo` reapplies them in order.
- **Atomicity.** If an entry fails partway, for example on OOM, the entries already reverted are reapplied (or vice versa), so the step is all-or-nothing. If that recovery also fails, the Scene reports `SceneError::Corrupted`. *My call:* this is the same rare failure the store already accepts for patches.

**Budget**
- *My call:* a step's size is the sum of its patches' `byte_size()` plus a constant per placement entry.
- When the budget is exceeded, the oldest steps are evicted, as in `History`.

**Releasing reserved slots**
- Evicting or discarding a patch must release the slots it reserves, which `History` does with crate-private code.
- *My call:* add one public store method so the Scene can do the same:

  ```rust
  impl Patch {
      /// Releases the slots this patch reserves. Call it once the patch is dropped from an
      /// undo history (`Undoable`) or from a redo history (`Redoable`).
      pub fn release(&self, mesh: &mut Mesh, side: PatchSide);
  }
  pub enum PatchSide { Undoable, Redoable }
  ```

  `History` then uses it internally, with unchanged behaviour.

- **A mesh's own `History` is not used inside the Scene.** The Scene keeps one history across all meshes. A user who edits a mesh outside the Scene (after `remove_mesh`) can use `History` as before.

## 8. Queries in world coordinates

Each query brings the BVHs it needs up to date: Lazy BVHs are updated, and stale Manual ones give `StaleBvh`. Inputs and outputs are world-space f64.

```rust
pub struct SceneRayHit { pub mesh: MeshId, pub hit: fast_mesh_query::RayHit /* local */, pub point: [f64; 3], pub t: f64 }
pub struct SceneRayOptions { pub min_t: f64, pub max_t: f64, pub cull: fast_mesh_query::Cull }   // world units; defaults 0.0, INFINITY, Cull::None
pub struct SceneClosestOptions { pub max_distance: f64 }                                           // world units; default INFINITY
pub fn raycast(&mut self, origin: [f64; 3], dir: [f64; 3], opts: &SceneRayOptions) -> Result<Option<SceneRayHit>>;          // all meshes, nearest
pub fn closest_point(&mut self, p: [f64; 3], opts: &SceneClosestOptions) -> Result<Option<SceneClosestHit>>;               // all meshes
pub fn closest_point_on(&mut self, id: MeshId, p: [f64; 3], opts: &SceneClosestOptions) -> Result<Option<SceneClosestHit>>;
pub fn intersects(&mut self, a: MeshId, b: MeshId) -> Result<bool>;
pub fn mesh_distance(&mut self, a: MeshId, b: MeshId, max_distance: f64) -> Result<Option<SceneMeshDistance>>;      // points in world
pub fn triangles_near(&mut self, a: MeshId, b: MeshId, margin: f64, out_a: &mut Vec<TriangleId>, out_b: &mut Vec<TriangleId>) -> Result<()>;
```

- **Mesh-vs-mesh queries** compute the relative `Rigid` (§5) and call Phase 3b. A rigid transform preserves distances, so distances and margins carry over unchanged.
- **Cross-mesh raycast** works mesh by mesh:
  - the ray goes into each mesh's local space (exact in f64, then f32);
  - meshes are culled first by the broadphase (§9);
  - the nearest world `t` wins, with ties going to the lower `MeshId` index.
  - *My call:* `dir` is normalised in world space so that `t` is a world distance.
- **Cross-mesh closest point** works the same way: meshes are visited by broadphase box distance and pruned by the best distance so far.
- *My call:* queries take `&mut self` because Lazy updates mutate. A read-only `query_current` variant can come later if needed.

## 9. Broadphase and contacts

- **The top level** is a small list of each mesh's world AABB: its BVH root box placed by the rigid transform, conservative in f64. It is rebuilt whenever a placement or BVH changes; it has one entry per mesh.
- *My call:* a sorted sweep-and-prune or brute-force N² over meshes is enough for "several meshes". Use a BVH over meshes only if benchmarks show it is needed.

```rust
pub struct Contact { pub a: MeshId, pub b: MeshId, pub intersecting: bool, pub distance: f64, pub triangles_a: Vec<TriangleId>, pub triangles_b: Vec<TriangleId> }
pub fn find_contacts(&mut self, margin: f64, filter: &ContactFilter, out: &mut Vec<Contact>) -> Result<usize>;
pub struct ContactFilter { pub pairs: Option<Vec<(MeshId, MeshId)>> /* None = all */, /* groups/masks: my call, later */ }
```

- **Candidates** are mesh pairs whose world boxes are within `margin`.
- **Per candidate:**
  - `intersecting` comes from `intersects`;
  - `distance` comes from `mesh_distance(max = margin)` and is 0 when intersecting;
  - `triangles_a`/`_b` come from `triangles_near(margin)`.
- Pairs beyond `margin` are not reported.
- The output is sorted by `(a, b)` index.
- *My call:* group/mask filters from the parent spec are deferred until there is a consumer. An explicit pair list covers the owner's two-mesh case.

## 10. Change reporting (events)

```rust
pub struct MeshChanges { pub id: MeshId, pub changes: Option<ChangeSet> /* None: everything */, pub placement_changed: bool, pub added: bool, pub removed: bool }
pub fn take_changes(&mut self) -> Vec<MeshChanges>;   // everything since the last call, then reset
```

- Every commit, undo, redo, placement change and removal adds to a per-mesh pending report. `take_changes` drains them, in id order.
- This is the Rust form of the parent spec's `updated` event. Phase 8 turns it into JS events with render dirty ranges.
- *My call:* render buffers and capacity events are out of scope, because the core has no render buffers yet.

## 11. Errors

```rust
pub enum SceneError { UnknownMesh, StaleBvh(MeshId), InvalidInput(&'static str), StepNotOpen, Corrupted, Store(StoreError), Bvh(BvhError), Query(QueryError) }
```

- *My call:* `OutOfMemory` from a lower crate is forwarded wrapped, as `Store(OutOfMemory)` and so on, not flattened.

## 12. Testing

**Model-based property tests**
- Random sequences over 2–3 meshes: edits, placements, steps (open and close), undo, redo, policy changes, `flush`, `remove_mesh`.
- A plain model tracks the expected geometry and placements per mesh, and the step stack.
- After each operation:
  - every mesh equals the model;
  - every Eager BVH is current and valid;
  - Lazy BVHs become current after `flush`;
  - world queries equal Phase 3 queries called directly with hand-computed local transforms.

**Unit tests**
- Relative-transform exactness: integer placements give exact relative `Rigid`s.
- Step atomicity under injected OOM.
- Release of reserved slots on eviction: a freed slot is reused only after its patch leaves the history.
- `ChangeSet::merge` against a brute-force replay.
- Stale ids.

**`find_contacts`** against all-pairs Phase 3b calls.

## 13. Benchmarks

- **Scene overhead** against calling Phase 3 directly, for one raycast, one `mesh_distance` and one `set_placement` plus `mesh_distance` between two 1M meshes. Target: below 5% overhead. *My call*, and not a merge gate.
- **`find_contacts`** over 2, 8 and 32 meshes.

## 14. Left to the builder

- Internal storage layout.
- The broadphase algorithm, as long as it is exact and conservative.
- How `edit` threads the transaction borrow.
- Module layout.

## 15. Decisions after the builder check (all *my call*)

1. **Eager BVH update failure.** Edits, undo and redo are all-or-nothing for mesh data and history. If the Eager BVH update that follows fails (e.g. `OutOfMemory`), the operation still succeeds; the BVH keeps its merged pending changes and is treated like Lazy until it is updated by the next query, `update(id)` or `flush()` (which then report the error if it fails again). Rationale: making BVH updates part of the atomic unit would need a full copy of every touched BVH. Tests assert "every Eager BVH is current" except right after an injected allocation failure.
2. **Relative rotations.** Placements are stored exactly as given (validated with the 1e-6 rule). Before a mesh-vs-mesh query the relative rotation `R_Aᵀ R_B` is re-orthonormalised in f64 (Gram–Schmidt on its rows), so Phase 3b's validation always accepts it. The change is within ~1e-6 and is documented.
3. **Open steps.** `undo`, `redo` and `clear_history` close an open step first (as if `end_step` was called), so Ctrl+Z mid-drag undoes the drag. `cancel_step()` reverts the open step's entries and discards it (Esc). `remove_mesh` is allowed while a step is open and drops that mesh's entries. An `edit` that fails leaves the open step unchanged. `SceneError::StepNotOpen` is returned by `end_step`/`cancel_step` with no open step.
4. **What queries need.** A query over all meshes needs every mesh current: Lazy BVHs are updated, and any stale Manual BVH makes the query fail with `StaleBvh(id)` (never skipped silently). Queries naming meshes (`closest_point_on`, pair queries, `find_contacts` with a pair filter) only update the named meshes. World boxes come from BVH root boxes after that update.
5. **Mesh maintenance inside the Scene.** Wrappers: `add_channel(id, desc)`, `remove_channel(id, name)`, `derived_channel_mut(id, name)` (not undoable, as in the store), and `compact(id)`. `compact` releases nothing (its slots no longer exist), drops that mesh's Edit history entries, which go stale with the epoch (steps left empty are dropped), keeps its Placement entries (they stay valid), rebuilds its BVH and reports the mesh as fully changed.
6. **Change reports.** `MeshChanges { id, changes: Option<ChangeSet>, placement_changed, added, removed }`. `changes == None` means "treat everything as changed" (after `compact`, `add_mesh`, or an undo/redo that failed with `Corrupted`; see §16), and `take_changes` is infallible. Merging into the pending report allocates infallibly, so running out of memory there aborts rather than falling back to `None`. A mesh added and removed between two calls is not reported. `placement_changed` is true if the placement changed at any point since the last call, even if undone.
7. **`ChangeSet::merge` rules.** Net effect per slot: added then removed → neither; removed then re-added → moved (vertex) or changed (triangle) when the slot was alive before, else added; moved/changed are unions minus slots that end up added or removed; channel lists are unioned per channel name, minus vertices that end up added or removed. A re-added (removed, then added) vertex joins every channel list named in either input, and in a merged set a moved vertex may have changed values in any channel, so consumers treat it as dirty for all channels. Tests check the store invariants (sorted, disjoint, moved ⊂ alive-before) against a brute-force replay; the BVH itself would accept a plain union.
8. **Names.** The Scene's policy enum is `UpdateMode { Eager, Lazy, Manual }` (`fast_mesh_bvh::UpdatePolicy` keeps its meaning). `MeshOptions` gains `bvh_policy: fast_mesh_bvh::UpdatePolicy`.
9. **f64 options to f32.** World f64 bounds are converted to Phase 3's f32 options rounding outward (`min_t` down, `max_t`/`max_distance`/`margin` up), and results are filtered again in f64 against the exact bound, so no result outside the bound is returned and none inside is lost.
10. **`remove_mesh`** releases the slots its dropped entries reserved (`Patch::release`) before returning the mesh.

## 16. Implementation notes

Where the implementation differs from, or makes precise, the wording above:

- **World options are `f64`.** `raycast` takes `SceneRayOptions` and `closest_point` / `closest_point_on` take `SceneClosestOptions` (§8), not Phase 3's `f32` `RayOptions` and `ClosestOptions`. `min_t`, `max_t` and `max_distance` are world distances. Per §15.9 they are rounded outward to `f32` for the Phase 3 call, and hits are filtered again against the exact `f64` bound.
- **Triangle lists may overshoot a margin that is not an `f32` value.** `triangles_near` and the lists of `find_contacts` round `margin` up to `f32` and cannot be filtered again per triangle, so they may hold triangles up to one `f32` step (relative `6e-8`) beyond it. `distance` and `intersecting` are exact against the `f64` margin.
- **World boxes are recomputed per query.** §9 says the top level is "rebuilt whenever a placement or BVH changes". The implementation derives each mesh's world box from its BVH root box and placement inside every query that needs it, so nothing has to be kept in step. There is one box per mesh, so the cost is a few operations per mesh.
- **`compact` keeps placement entries.** Only the mesh's edit entries leave the undo and redo stacks and an open step (§15.5); its placement entries stay valid and stay.
- **After `Corrupted`, touched meshes are fully changed.** Each mesh with an edit entry in the failed step is reported with `changes: None` and its BVH is rebuilt from scratch on its next update (right away for Eager), once per mesh however many entries it has in the step. Each mesh with a placement entry in it is reported as `placement_changed`, since the failed recovery may leave its placement at either value. The history may no longer match the meshes, so the caller should consider `clear_history`.
- **`changes: None` only after `add_mesh`, `compact` or `Corrupted`** (§15.6). `ChangeSet::merge` and `ChangeSet::clone`, which fold changes into the pending report and the BVH's pending changes, allocate infallibly: if memory runs out there, the process aborts. There is no fallback to `None` on out-of-memory.
- **Ctrl+Z mid-drag always undoes the drag** (§15.3). `undo` with a non-empty open step undoes that step directly and only then applies the budget: the oldest undo steps are evicted first, and if the undone step alone exceeds the budget it is dropped from the redo stack again. The undo itself always happens, and `can_undo` agrees with `undo`. `redo` and `clear_history` close the open step as `end_step` would: `redo` has nothing to redo while a step has entries (recording dropped the redo stack), and `clear_history` drops everything anyway.
- **A failed `undo`/`redo` changes nothing.** On an error other than `Corrupted`, the meshes, both stacks and the open step (entries and nesting depth) are as before; the step is closed only when the undo or redo succeeds.
- **Placement runs coalesce.** Within an open step, a placement change of the mesh whose placement entry is the step's last entry merges into that entry, which keeps its first `old` and takes the new value; any other entry in between (an edit, or another mesh's placement) starts a new entry. A drag at 60 fps records one 192-byte entry instead of hundreds, and undo restores the placement from before the drag. A run that ends where it started stays as one entry: its undo changes nothing but is reported as `placement_changed`. Outside an open step each placement change stays a step of its own. Setting the current placement again records and reports nothing.
- **Out of memory is wrapped where it happened.** It surfaces as `Store(OutOfMemory)` when the store, the history or the scene's own bookkeeping ran out, and as `Query(OutOfMemory)` when a Phase 3 query did (for example a result vector in `triangles_near`). See §11.
- **The property test checks reports and eviction.** The history budget is drawn from {unlimited, 100–800 bytes}. Each model mesh keeps a shadow `Mesh` that gets the same edits, and the model applies, reverts and releases the shadow's patches by its own copy of the history rules, so it knows each entry's exact size and which steps are evicted; the shadow's free slots must equal the scene mesh's after every action. After every action `take_changes` must report exactly the added and removed ids since the last call, every changed slot among the moved or changed ones, `None` only after `compact` (the test injects no faults), and `placement_changed` exactly when a placement change ran.
- **BVH validity is checked through query equality.** The property test cannot reach a mesh's BVH, so "every Eager BVH is valid" means that after every action `closest_point_on` on each Eager mesh, and `mesh_distance` between Eager meshes, equal Phase 3 on a fresh build; a full set of world queries runs on a share of the actions and at the end. The unit tests call `Bvh::validate` directly.
- **Placement entries count 192 bytes** against the history budget: the old and the new value, 96 bytes each (§7).

### Open questions for Phase 5

- **`take_changes` has a single consumer.** It drains the reports destructively, so a second reader (for example a renderer beside an undo-history panel) would miss what the first took. Phase 5 should either hook into the Scene internally, or reports should become per subscriber (each with its own cursor or pending set).
- **No public accessors for local-space bulk queries.** There is no `bvh(id)` or `relative_transform(a, b)`, so a caller that wants to run many Phase 3 queries in a mesh's local space must go through the world-space wrappers one call at a time.
- **Some changes are not reported.** Writes through `derived_channel_mut`, and `add_channel` / `remove_channel`, are neither undoable nor reported (as in the store). A renderer needs to know when to re-upload such data, so Phase 5 needs a render-dirty signal for them.
