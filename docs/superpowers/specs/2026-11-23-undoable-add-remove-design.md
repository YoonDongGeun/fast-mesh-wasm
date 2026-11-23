# Undoable add and remove mesh (#269)

- **Status:** Implemented.
- **Date:** 2026-11-23
- **Parent spec:** `2026-10-18-fast-mesh-scene-design.md` (§4 registry, §7 history, §10 reports).
  This document amends §4 ("removing a mesh drops its entries from the history") for the new
  undoable calls only; the existing calls keep their behaviour.

## 1. Intent

Undo removes a mesh that a step created and brings back a mesh that a step removed, with the same
id, data and display state. Tools stop adding an empty mesh and filling it as one edit, and stop
emptying the meshes they replace.

## 2. API

Opt-in, so every existing caller keeps its behaviour (file loads stay permanent, `remove_mesh` still
returns the mesh and drops its history, and the empty-then-fill pattern works as it did):

| Layer | Add | Remove |
| --- | --- | --- |
| Rust `Scene` | `MeshOptions { undoable: true, .. }` on `add_mesh` / `add_mesh_built` | `remove_mesh_undoable(id)` |
| wasm | `add_mesh_undoable` | `remove_mesh_undoable` |
| TS `Scene`, `WorkerScene`, `SceneStore` | `addMesh(p, i, { undoable: true })` | `removeMesh(mesh, { undoable: true })` |

An undoable add or removal is recorded like an edit: in the open step, or as a step of its own. It
drops the redo stack. `is_restorable(id)`, `restorable_ids()` and `restorable_id(index, generation)`
tell which meshes the history can still bring back; the TS scenes have `has(mesh)` and
`isRestorable(mesh)`, the store `isRestorable(mesh)`.

The scene ops that create a mesh as part of their own step record that add too, so their undo
removes it: `convex_hull`, `convex_hull_points`, `cut_by_plane` (the removed side) and
`boolean_batch`. The other ops that add meshes (extract and split faces, drain plugs, placed stamps,
point-cloud reconstruction) and `join` (which removes its source) keep their permanent adds and
removal for now; migrating them is a follow-up.

## 3. Keeping a removed mesh

- **Parked, not copied.** A removed mesh's whole `Entry` (store `Mesh` with every channel and UVs,
  BVH, placement, mode, normals, colouring, pending BVH changes) stays in its slot of the slot
  map, which is flagged as parked: lookups and iteration miss it. No buffer is moved or copied,
  and no BVH or normals are rebuilt on the way back.
- **Ids stay stable.** A parked slot keeps its generation and is not on the free list, so no new
  mesh can take it. Restoring flips the slot back to live: the `MeshId` is the one it had. Only
  when the history drops the entry for good is the slot freed (generation bumped, as for
  `remove_mesh`), and the id never matches again.
- **What goes with it.** The fields on the mesh (distance fields measured from it, thickness and
  curvature fields) and its colouring by them are parked with it (`ParkedLinks`) and come back,
  recomputed in full on their next update (targets removed meanwhile are dropped; a field left
  without targets goes, with its colouring). Its elastic rest state is kept too.
- **What does not.** Links owned by other meshes are cut exactly as `remove_mesh` cuts them, and are
  not restored: fields on other meshes measured to it alone are removed (with colourings by them),
  fields measured to it and others lose it as a target, it leaves the sections that cut it, and
  its collision pair rules go. These are other meshes' or the app's state, which the SDK store
  and the apps already prune on removal; restoring them would fight their own bookkeeping.

## 4. History entries

`Change::Add { id, bytes }` and `Change::Remove { id, bytes }`. Undo of an add and redo of a
removal park the mesh; redo of an add and undo of a removal restore it. Within a step entries run in
order, so "add, then fill" undoes as "empty, then park" and redoes as "restore, then fill".

Parking and restoring only flip the slot's flag while the step runs, which cannot fail, so the
step stays all or nothing. Once the whole step has run, `Scene::settle_presence` brings the
registries in line: it cuts the links of the newly parked meshes and puts back what the restored
ones kept, and reports both. A step that fails flips the flags back and never touches a field, a
section, a collision rule or a report. Eviction that an undo of an open step causes runs after the
settling (`History::trim`), since it may free a mesh that was just parked.

When the history drops an entry (eviction, a new step dropping redo, `cancel_step`,
`clear_history`), the side it is on says what it holds:

| Entry | Dropped from undo | Dropped from redo |
| --- | --- | --- |
| Add | mesh is live: nothing (it stays, for good) | mesh is parked: freed |
| Remove | mesh is parked: freed | mesh is live: nothing |

`remove_mesh` (permanent) on a live mesh drops its entries as before, Add and Remove included.
`compact` keeps Add and Remove entries (they stay valid), as it keeps placement entries.

## 5. Budget

A parked mesh counts its bytes: the store mesh (positions, triangles, liveness, channels, caches),
the BVH, normals, colours and parked field state, measured when it is parked (about 65 MB for 1M
triangles). Only the entry that would bring it back counts it: its removal on the undo stack, its
add on the redo stack. A live mesh counts nothing, so creating large meshes does not push older
steps out of the history.

So an undo or a redo can exceed the budget, by parking meshes. It always happens; then:

- after an **undo**, the furthest redo steps go first (so that a run of undos can go all the way
  back), then the oldest undo steps; a step that alone exceeds the budget is dropped at once (its
  add is undone, and cannot be redone);
- after a **redo** of a removal, the oldest undo steps go first, then the furthest redo steps.

A removal larger than the whole budget is dropped as soon as its step closes, which makes it
permanent, as any step of that size. Eviction frees the parked mesh at once.

## 6. Change reports

- **Parking** reports the mesh as `removed` to every subscriber, unless the subscriber was never
  told it was added (then it reports nothing, as for add-then-remove).
- **Restoring** reports it as `added` (everything new), except to a subscriber that has not yet
  taken the removal: that pending removal is withdrawn and the mesh is reported fully changed
  instead (`changes: None`, every colour and normal dirty, `placement_changed`), so to that reader
  the mesh never left.
- A subscriber added or removed while a mesh is parked keeps every report list in step: parked
  entries have one report per subscriber too.
- wasm records gain flag `1024` (restorable) on removals the history can revert, as of the moment
  the changes are taken; the TS `MeshChange` carries it as `restorable: true`. A consumer keeps what
  it needs to show the mesh again (handles, names, materials) for those removals only.

## 7. SDK

- **`Scene`** keeps the handles of restorable meshes, so a restored mesh comes back with the same
  `MeshHandle` object. Handles of meshes the history dropped are pruned after the next undo, redo,
  cancel or clear.
- **Worker engine.** An undoable add runs inline, never in the background, so it lands in the step
  it was asked in. After a request that can move meshes (an undoable removal, undo, redo, a
  cancelled step) the engine compares every mesh with the scene. A parked mesh keeps its client id
  and the fields on it; requests on it fail in the scene with `UNKNOWN_MESH`. A result says where
  the moved meshes now are (`presence`: live, parked or gone) and what that left of fields
  (`fieldTargets`) and sections (`sectionMeshes`): which links were cut depends on the order of the
  steps, and a mesh may leave and come back within one batch, so the client cannot infer it.
- **Worker client.** The state of a parked mesh is kept; calls on it give `UNKNOWN_MESH`, and its
  mirrors are refilled from the restore record. A `removeMesh` newer than a result decides over what
  the result says. A mesh added and undone before any record of it was packed still gets a removal
  record, since its handle was given out at the add. `restart()` clears the history, so it rebuilds
  the live meshes from the mirrors and drops the parked ones.
- **`SceneStore`** follows the local scene after each undo, redo and cancel, and the Worker by the
  results' presence and the change records. A mesh that left is out of the snapshot and the
  selection, and its state is kept while it is restorable. A mesh that is back has the display it
  had. What the scene cuts and does not restore, the store asks for again, since it is declarative:
  each mesh keeps the colouring the app last asked for (`asked`), each section the meshes it last
  asked for, and whatever of them is in the scene is what is requested. Requests that raced with an
  undo in flight and failed are sent again. Contact groups are not rejoined: their owner sets them.

## 8. Showcase

`AppState.follow(store)` keeps the list in step with the store's snapshot: a mesh that left leaves
the list, kept aside with its place, groups, collision pair sides and whether it was selected while
the store says it is restorable; a mesh that is back is listed again as the same item (name, colour,
material, locks). The material is disposed when it leaves; three.js sets it up afresh on reuse.
Create, duplicate, convex hull, the two-mesh and N-way booleans use undoable adds and removals in one
step each, and the mesh list's and a group's Remove are undoable. Opening a project clears the
history first, so that nothing from before comes back. Tools still using the empty-then-fill pattern
keep working unchanged.

## 9. Cost

Parking flips a flag and restoring flips it back, whatever the mesh's size: on a 1M-triangle mesh a
removal with its undo takes about 0.5 µs in the engine, where emptying the mesh with one edit and
undoing it took about 590 ms.
