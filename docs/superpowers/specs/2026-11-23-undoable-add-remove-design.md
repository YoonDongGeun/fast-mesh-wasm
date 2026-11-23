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
returns the mesh and drops its history):

| Layer | Add | Remove |
| --- | --- | --- |
| Rust `Scene` | `MeshOptions { undoable: true, .. }` on `add_mesh` / `add_mesh_built` | `remove_mesh_undoable(id)` |
| wasm | `add_mesh*(…, undoable)`, `begin_add_mesh_upload(…, undoable)` | `remove_mesh_undoable(index, generation)` |
| TS `Scene` / `WorkerScene` | `addMesh(p, i, { undoable: true })` | `removeMesh(mesh, { undoable: true })` |
| `SceneStore` | `addMesh(p, i, { undoable: true })` | `removeMesh(mesh, { undoable: true })` |

An undoable add or removal is recorded like an edit: in the open step, or as a step of its own. It
drops the redo stack. `Scene::is_restorable(id)` and `restorable_ids()` tell which removed meshes the
history can still bring back.

The scene ops that create a mesh as part of their own step record that add too, so their undo
removes it: `convex_hull`, `convex_hull_points`, `cut_by_plane` (the removed side) and
`boolean_batch`. Ops other agents are building on (extract and split faces, drain plugs, placed
stamps, reconstruction) keep their permanent adds for now; migrating them is a follow-up.

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

An Add or Remove entry counts the bytes of the mesh it may hold: the store mesh (positions,
triangles, liveness, channels, caches), the BVH, normals, colours and parked field state, measured
when it is recorded. The count is the same on both stacks, like a patch's. Eviction of the oldest
steps frees parked meshes' data. A single add or removal larger than the whole budget is dropped as
soon as its step closes (the add stays; the removal becomes permanent), as for any step.

## 6. Change reports

- **Parking** reports the mesh as `removed` to every subscriber, unless the subscriber was never
  told it was added (then it reports nothing, as for add-then-remove).
- **Restoring** reports it as `added` (everything new), except to a subscriber that has not yet
  taken the removal: that pending removal is withdrawn and the mesh is reported fully changed
  instead (`changes: None`, every colour and normal dirty, `placement_changed`), so to that reader
  the mesh never left.
- wasm records gain flag `1024` (restorable) on removals the history can revert; the TS
  `MeshChange` carries it as `restorable: true`. A consumer keeps what it needs to show the mesh
  again (handles, names, materials) for those removals only.

## 7. SDK

- **`Scene`** keeps the handles of restorable meshes, so a restored mesh comes back with the same
  `MeshHandle` object. Handles of meshes the history dropped are pruned when it reports them gone.
- **Worker.** `addMesh` and `removeMesh` requests carry `undoable`. An undoable add runs inline (not
  in the background), so it lands in the step it was asked in. The engine keeps a parked mesh's
  client-id mapping and its fields; the client keeps the mesh state of a removal the engine can
  revert, gives `UNKNOWN_MESH` for calls on it meanwhile, and refills its mirrors from the restore
  record. `restart()` clears the history, so parked meshes are dropped for good there.
- **`SceneStore`** follows the removal and restore records: a removed mesh leaves the snapshot (and
  the selection); a restored one comes back with the display state it had. Other meshes' state is
  pruned on removal exactly as `removeMesh` does.

## 8. Showcase

`AppState` follows the records too: a removed item is parked with its name, colour and material
(the material's GPU programs are disposed and rebuilt on reuse); a restored mesh is listed again as
the same item. Create, convex hull, the N-way boolean and plane cut use undoable adds and removals,
and the mesh list's remove button is undoable. Tools still using the empty-then-fill pattern keep
working unchanged.
