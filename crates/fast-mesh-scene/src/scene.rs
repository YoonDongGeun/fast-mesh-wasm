//! The scene: a registry of meshes with BVHs and placements.

use fast_mesh_bvh::{BuildOptions, Bvh, UpdatePolicy};
use fast_mesh_store::{
    ChangeSet, ChannelDesc, Mesh, Patch, PatchSide, Remap, StoreError, Transaction,
};

use crate::id::SlotMap;
use crate::report::Report;
use crate::{MeshChanges, MeshId, Placement, Result, SceneError};

/// When a mesh's BVH is brought up to date after the mesh changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpdateMode {
    /// Right after every commit, undo and redo of that mesh.
    #[default]
    Eager,
    /// Before the next query that needs it, or on `flush` / `update`.
    Lazy,
    /// Only on an explicit `update`; queries that need it fail instead.
    Manual,
}

/// Options for adding a mesh to a [`Scene`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MeshOptions {
    /// How the mesh's BVH is built.
    pub build: BuildOptions,
    /// When the BVH falls back to heavier update strategies.
    pub bvh_policy: UpdatePolicy,
    /// When the BVH is brought up to date.
    pub mode: UpdateMode,
    /// Where the mesh starts out in the world.
    pub placement: Placement,
}

/// Options for a [`Scene`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneOptions {
    /// Most bytes the shared undo history may hold.
    pub history_budget: usize,
}

impl Default for SceneOptions {
    /// A history budget of 256 MiB.
    fn default() -> Self {
        SceneOptions {
            history_budget: 256 << 20,
        }
    }
}

/// One mesh of the scene with everything attached to it.
pub(crate) struct Entry {
    pub(crate) mesh: Mesh,
    pub(crate) bvh: Bvh,
    pub(crate) placement: Placement,
    pub(crate) mode: UpdateMode,
    /// Changes not yet applied to the BVH, merged into one set. `None` when
    /// none are waiting.
    pub(crate) pending: Option<ChangeSet>,
    pub(crate) report: Report,
}

impl Entry {
    /// Folds a commit's (or undo's or redo's) changes into the BVH's pending
    /// changes and into the report.
    fn record_changes(&mut self, changes: &ChangeSet) {
        match &mut self.pending {
            Some(p) => p.merge(changes),
            None => self.pending = Some(changes.clone()),
        }
        self.report.merge(changes);
    }

    /// Brings the BVH up to date with the pending changes. On error the
    /// pending changes are kept, so a later call can retry.
    fn update_bvh(&mut self) -> Result<()> {
        if self.bvh.is_current(&self.mesh) {
            self.pending = None;
            return Ok(());
        }
        #[cfg(test)]
        fault::before_bvh_update()?;
        let none = ChangeSet::default();
        let changes = self.pending.as_ref().unwrap_or(&none);
        self.bvh.update(&mut self.mesh, changes)?;
        self.pending = None;
        Ok(())
    }

    /// After a change: brings an Eager BVH up to date. A failure is not an
    /// error of the change (§15.1): the pending changes stay and the BVH is
    /// treated like Lazy until an update succeeds.
    fn after_change(&mut self) {
        if self.mode == UpdateMode::Eager {
            let _ = self.update_bvh();
        }
    }
}

/// Several meshes, each with a BVH and a placement in the world.
pub struct Scene {
    entries: SlotMap<Entry>,
    /// Reports of meshes removed since the last `take_changes`.
    removed: Vec<MeshChanges>,
    /// Placeholder for the shared undo history: the committed patches in
    /// order. Task 4 replaces it; patches enter only through `record_patch`.
    history: Vec<(MeshId, Patch)>,
    // Used by the shared history of a later task.
    #[allow(dead_code)]
    history_budget: usize,
}

impl Scene {
    /// An empty scene.
    pub fn new(options: SceneOptions) -> Scene {
        Scene {
            entries: SlotMap::new(),
            removed: Vec::new(),
            history: Vec::new(),
            history_budget: options.history_budget,
        }
    }

    /// Hands a committed patch to the history. This is the single hand-off
    /// point the shared history of Task 4 hooks into.
    fn record_patch(history: &mut Vec<(MeshId, Patch)>, id: MeshId, patch: Patch) {
        history.push((id, patch));
    }

    fn entry_mut(&mut self, id: MeshId) -> Result<&mut Entry> {
        self.entries.get_mut(id).ok_or(SceneError::UnknownMesh)
    }

    /// Adds a mesh, building its BVH, and returns its id.
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] for an invalid placement, or a BVH or
    /// memory error from the build. The scene is unchanged on error.
    pub fn add_mesh(&mut self, mesh: Mesh, options: MeshOptions) -> Result<MeshId> {
        options.placement.validate()?;
        let mut bvh = Bvh::build(&mesh, &options.build)?;
        bvh.set_policy(options.bvh_policy)?;
        self.entries.insert(Entry {
            mesh,
            bvh,
            placement: options.placement,
            mode: options.mode,
            pending: None,
            report: Report::added(),
        })
    }

    /// Removes a mesh and returns it. The id never matches again. Its
    /// recorded patches leave the history, releasing the slots they reserve.
    /// The removal shows up in the next [`Scene::take_changes`], unless the
    /// mesh was added since the last call.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id.
    pub fn remove_mesh(&mut self, id: MeshId) -> Result<Mesh> {
        self.entry(id)?;
        // Reserve first so the report cannot be lost after the removal.
        self.removed
            .try_reserve(1)
            .map_err(|_| SceneError::Store(StoreError::OutOfMemory))?;
        let mut entry = self.entries.remove(id).ok_or(SceneError::UnknownMesh)?;
        // Placeholder history: Task 4 replaces this with the real eviction.
        self.history.retain(|(owner, patch)| {
            if *owner == id {
                patch.release(&mut entry.mesh, PatchSide::Undoable);
                false
            } else {
                true
            }
        });
        if !entry.report.added {
            self.removed.push(entry.report.take(id, true));
        }
        Ok(entry.mesh)
    }

    /// Runs `f` in a transaction on mesh `id`, commits it, and records the
    /// patch in the history. The changes are added to the mesh's report and
    /// to its BVH's pending changes; an Eager BVH is then brought up to date.
    ///
    /// An empty transaction records and reports nothing.
    ///
    /// If the Eager BVH update fails (for example out of memory), the edit
    /// still succeeds: the BVH keeps its pending changes and is treated like
    /// Lazy until [`Scene::update`], [`Scene::flush`] or a query updates it.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id, or [`SceneError::Store`]
    /// when `f` fails or the commit fails. The transaction is then rolled back
    /// and the scene is unchanged.
    pub fn edit<R>(
        &mut self,
        id: MeshId,
        f: impl FnOnce(&mut Transaction) -> fast_mesh_store::Result<R>,
    ) -> Result<R> {
        let entry = self.entries.get_mut(id).ok_or(SceneError::UnknownMesh)?;
        let mut tx = entry.mesh.begin_edit();
        // Dropping the transaction on error rolls it back.
        let r = f(&mut tx)?;
        let patch = tx.commit()?;
        if patch.is_empty() {
            return Ok(r);
        }
        entry.record_changes(patch.changes());
        Self::record_patch(&mut self.history, id, patch);
        entry.after_change();
        Ok(r)
    }

    /// Sets when the mesh's BVH is brought up to date. Switching to
    /// [`UpdateMode::Eager`] updates a stale BVH right away; if that update
    /// fails, the mode is still set and the BVH stays pending, as after a
    /// failed Eager update in [`Scene::edit`].
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id.
    pub fn set_mode(&mut self, id: MeshId, mode: UpdateMode) -> Result<()> {
        let entry = self.entry_mut(id)?;
        entry.mode = mode;
        entry.after_change();
        Ok(())
    }

    /// Brings the mesh's BVH up to date now, whatever its mode.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`], or a BVH or memory error from the update.
    /// On error the BVH keeps its pending changes.
    pub fn update(&mut self, id: MeshId) -> Result<()> {
        self.entry_mut(id)?.update_bvh()
    }

    /// Brings every Lazy BVH up to date, and every Eager one left stale by a
    /// failed update. Manual BVHs are left alone.
    ///
    /// # Errors
    /// The first BVH or memory error. Every mesh is still attempted; those
    /// that failed keep their pending changes.
    pub fn flush(&mut self) -> Result<()> {
        let mut first = Ok(());
        for (_, entry) in self.entries.iter_mut() {
            if entry.mode != UpdateMode::Manual {
                let r = entry.update_bvh();
                if first.is_ok() {
                    first = r;
                }
            }
        }
        first
    }

    /// Everything that happened to the meshes since the last call, one entry
    /// per affected mesh in id order, then resets every report.
    ///
    /// A mesh added and removed between two calls is not reported, and
    /// neither is a mesh whose changes cancel out (for example vertices added
    /// and removed again).
    pub fn take_changes(&mut self) -> Vec<MeshChanges> {
        let mut out = std::mem::take(&mut self.removed);
        for (id, entry) in self.entries.iter_mut() {
            if entry.report.is_clean() {
                entry.report = Report::clean();
            } else {
                out.push(entry.report.take(id, false));
            }
        }
        out.sort_by_key(|c| c.id);
        out
    }

    /// Adds a channel to the mesh, filled with its default value. Not
    /// undoable and not reported, as in the store.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`], or the store's error (for example a
    /// duplicate name). The scene is unchanged on error.
    pub fn add_channel(&mut self, id: MeshId, desc: ChannelDesc) -> Result<()> {
        Ok(self.entry_mut(id)?.mesh.add_channel(desc)?)
    }

    /// Removes a channel from the mesh and returns its description. Not
    /// undoable and not reported, as in the store.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`], or the store's error for an unknown name.
    pub fn remove_channel(&mut self, id: MeshId, name: &str) -> Result<ChannelDesc> {
        Ok(self.entry_mut(id)?.mesh.remove_channel(name)?)
    }

    /// Writable values of a derived channel. Writes are neither undoable nor
    /// reported, as in the store.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`], or the store's error for an unknown or
    /// non-derived channel.
    pub fn derived_channel_mut(&mut self, id: MeshId, name: &str) -> Result<&mut [f32]> {
        Ok(self.entry_mut(id)?.mesh.derived_channel_mut(name)?)
    }

    /// Removes the mesh's dead slots and renumbers the rest (see
    /// [`Mesh::compact`]). The mesh's recorded patches no longer apply and
    /// leave the history. Its BVH is rebuilt whatever its mode, and the next
    /// [`Scene::take_changes`] reports it as fully changed (`changes: None`).
    ///
    /// If the rebuild fails, compaction still succeeds and the BVH stays
    /// pending, as after a failed Eager update in [`Scene::edit`].
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`], or a store error from the compaction; the
    /// scene is then unchanged.
    pub fn compact(&mut self, id: MeshId) -> Result<Remap> {
        let entry = self.entries.get_mut(id).ok_or(SceneError::UnknownMesh)?;
        let remap = entry.mesh.compact()?;
        // The slots of dropped patches no longer exist: nothing to release.
        // Placeholder history: Task 4 replaces this.
        self.history.retain(|(owner, _)| *owner != id);
        // Every slot id changed; the BVH update rebuilds from scratch because
        // the mesh's epoch changed, whatever the pending changes say.
        entry.pending = Some(ChangeSet::default());
        entry.report.changes = None;
        let _ = entry.update_bvh();
        Ok(remap)
    }

    fn entry(&self, id: MeshId) -> Result<&Entry> {
        self.entries.get(id).ok_or(SceneError::UnknownMesh)
    }

    /// The mesh behind `id`.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id.
    pub fn mesh(&self, id: MeshId) -> Result<&Mesh> {
        Ok(&self.entry(id)?.mesh)
    }

    /// The ids of all meshes, in index order.
    pub fn ids(&self) -> impl Iterator<Item = MeshId> {
        self.entries.iter().map(|(id, _)| id)
    }

    /// Where the mesh sits in the world.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id.
    pub fn placement(&self, id: MeshId) -> Result<Placement> {
        Ok(self.entry(id)?.placement)
    }

    /// Moves the mesh. Only stores the placement; nothing else is touched.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`], or [`SceneError::InvalidInput`] when the
    /// placement is not a proper rotation. The scene is unchanged on error.
    pub fn set_placement(&mut self, id: MeshId, placement: Placement) -> Result<()> {
        placement.validate()?;
        let entry = self.entries.get_mut(id).ok_or(SceneError::UnknownMesh)?;
        entry.placement = placement;
        entry.report.placement_changed = true;
        Ok(())
    }

    /// Whether the mesh's BVH is up to date with the mesh.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id.
    pub fn is_current(&self, id: MeshId) -> Result<bool> {
        let e = self.entry(id)?;
        Ok(e.bvh.is_current(&e.mesh))
    }
}

/// Test-only fault injection for the Scene's calls into lower crates, whose
/// own fault hooks are crate-private.
#[cfg(test)]
pub(crate) mod fault {
    use std::cell::Cell;

    use fast_mesh_bvh::BvhError;

    thread_local! {
        /// Calls to `Bvh::update` left before the one that fails.
        static BVH_UPDATE: Cell<Option<u32>> = const { Cell::new(None) };
    }

    /// Makes the `n`th coming `Bvh::update` call (1 = the next) fail with
    /// `OutOfMemory` without reaching the BVH. Fires once.
    pub(crate) fn fail_bvh_update_on(n: u32) {
        assert!(n > 0);
        BVH_UPDATE.with(|c| c.set(Some(n - 1)));
    }

    /// Checked just before every `Bvh::update` call.
    pub(crate) fn before_bvh_update() -> Result<(), BvhError> {
        BVH_UPDATE.with(|c| match c.get() {
            Some(0) => {
                c.set(None);
                Err(BvhError::OutOfMemory)
            }
            Some(n) => {
                c.set(Some(n - 1));
                Ok(())
            }
            None => Ok(()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri() -> Mesh {
        Mesh::from_buffers(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[0, 1, 2]).unwrap()
    }

    #[test]
    fn stale_ids_are_rejected() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(tri(), MeshOptions::default()).unwrap();
        assert!(scene.mesh(a).is_ok());
        assert!(scene.remove_mesh(a).is_ok());
        assert_eq!(scene.mesh(a).err(), Some(SceneError::UnknownMesh));
        assert_eq!(scene.placement(a).err(), Some(SceneError::UnknownMesh));
        assert_eq!(
            scene.set_placement(a, Placement::IDENTITY),
            Err(SceneError::UnknownMesh)
        );
        assert_eq!(scene.is_current(a).err(), Some(SceneError::UnknownMesh));
        assert_eq!(scene.remove_mesh(a).err(), Some(SceneError::UnknownMesh));
        // The slot is reused, but the old id still does not match.
        let b = scene.add_mesh(tri(), MeshOptions::default()).unwrap();
        assert_ne!(a, b);
        assert_eq!(scene.mesh(a).err(), Some(SceneError::UnknownMesh));
        assert!(scene.mesh(b).is_ok());
    }

    #[test]
    fn ids_iterate_in_index_order() {
        let mut scene = Scene::new(SceneOptions::default());
        let ids: Vec<_> = (0..4)
            .map(|_| scene.add_mesh(tri(), MeshOptions::default()).unwrap())
            .collect();
        scene.remove_mesh(ids[1]).unwrap();
        assert_eq!(scene.ids().collect::<Vec<_>>(), [ids[0], ids[2], ids[3]]);
        let again = scene.add_mesh(tri(), MeshOptions::default()).unwrap();
        // Reuses index 1, so it iterates between ids[0] and ids[2].
        assert_eq!(
            scene.ids().collect::<Vec<_>>(),
            [ids[0], again, ids[2], ids[3]]
        );
    }

    #[test]
    fn placement_round_trip_and_validation() {
        let mut scene = Scene::new(SceneOptions::default());
        let bad = Placement {
            translation: [f64::NAN, 0.0, 0.0],
            ..Placement::IDENTITY
        };
        let opts = MeshOptions {
            placement: bad,
            ..MeshOptions::default()
        };
        assert!(matches!(
            scene.add_mesh(tri(), opts),
            Err(SceneError::InvalidInput(_))
        ));
        assert_eq!(scene.ids().count(), 0);
        let id = scene.add_mesh(tri(), MeshOptions::default()).unwrap();
        let moved = Placement {
            translation: [1.0, 2.0, 3.0],
            ..Placement::IDENTITY
        };
        scene.set_placement(id, moved).unwrap();
        assert_eq!(scene.placement(id), Ok(moved));
        assert!(scene.set_placement(id, bad).is_err());
        assert_eq!(scene.placement(id), Ok(moved));
        assert_eq!(scene.is_current(id), Ok(true));
    }

    /// A flat `n` x `n` grid of quads, two triangles each.
    fn grid(n: u32) -> Mesh {
        let mut pos = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                pos.extend_from_slice(&[x as f32, y as f32, 0.0]);
            }
        }
        let mut idx = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                let b = a + 1;
                let c = a + n + 1;
                idx.extend_from_slice(&[a, b, c, b, c + 1, c]);
            }
        }
        Mesh::from_buffers(&pos, &idx).unwrap()
    }

    fn with_mode(mode: UpdateMode) -> MeshOptions {
        MeshOptions {
            mode,
            ..MeshOptions::default()
        }
    }

    fn lift(scene: &mut Scene, id: MeshId, v: u32, z: f32) {
        scene
            .edit(id, |tx| {
                let p = tx.mesh().position(v).unwrap();
                tx.set_positions(&[v], &[[p[0], p[1], z]])
            })
            .unwrap();
    }

    fn assert_valid(scene: &Scene, id: MeshId) {
        let e = scene.entries.get(id).unwrap();
        e.bvh.validate(&e.mesh).unwrap();
    }

    #[test]
    fn eager_bvh_stays_current() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(grid(4), MeshOptions::default()).unwrap();
        let r = scene
            .edit(id, |tx| {
                tx.set_positions(&[6], &[[1.0, 1.0, 2.0]]).map(|()| 7)
            })
            .unwrap();
        assert_eq!(r, 7);
        assert_eq!(scene.is_current(id), Ok(true));
        assert_valid(&scene, id);
        assert!(scene.entries.get(id).unwrap().pending.is_none());
        assert_eq!(scene.history.len(), 1);
    }

    #[test]
    fn failed_edit_changes_nothing() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        scene.take_changes();
        let before = scene.mesh(id).unwrap().revision();
        let err = scene.edit(id, |tx| {
            tx.set_positions(&[0], &[[5.0, 5.0, 5.0]])?;
            tx.remove_vertices(&[1])
        });
        assert!(matches!(err, Err(SceneError::Store(_))));
        assert_eq!(scene.mesh(id).unwrap().revision(), before);
        assert_eq!(scene.mesh(id).unwrap().position(0), Some([0.0, 0.0, 0.0]));
        assert!(scene.history.is_empty());
        assert!(scene.take_changes().is_empty());
        assert_eq!(scene.is_current(id), Ok(true));
        scene.remove_mesh(id).unwrap();
        assert_eq!(
            scene.edit(id, |_| Ok(())).err(),
            Some(SceneError::UnknownMesh)
        );
    }

    #[test]
    fn lazy_waits_for_flush() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene
            .add_mesh(grid(4), with_mode(UpdateMode::Lazy))
            .unwrap();
        lift(&mut scene, id, 6, 1.0);
        assert_eq!(scene.is_current(id), Ok(false));
        scene.flush().unwrap();
        assert_eq!(scene.is_current(id), Ok(true));
        assert_valid(&scene, id);
    }

    #[test]
    fn manual_waits_for_update() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene
            .add_mesh(grid(4), with_mode(UpdateMode::Manual))
            .unwrap();
        lift(&mut scene, id, 6, 1.0);
        assert_eq!(scene.is_current(id), Ok(false));
        scene.flush().unwrap();
        assert_eq!(scene.is_current(id), Ok(false));
        scene.update(id).unwrap();
        assert_eq!(scene.is_current(id), Ok(true));
        assert_valid(&scene, id);
        // Switching to Eager brings a stale BVH up to date.
        lift(&mut scene, id, 7, 1.0);
        assert_eq!(scene.is_current(id), Ok(false));
        scene.set_mode(id, UpdateMode::Eager).unwrap();
        assert_eq!(scene.is_current(id), Ok(true));
        scene.remove_mesh(id).unwrap();
        assert_eq!(
            scene.set_mode(id, UpdateMode::Lazy),
            Err(SceneError::UnknownMesh)
        );
        assert_eq!(scene.update(id), Err(SceneError::UnknownMesh));
    }

    #[test]
    fn pending_changes_merge_across_commits() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene
            .add_mesh(grid(6), with_mode(UpdateMode::Lazy))
            .unwrap();
        lift(&mut scene, id, 8, 1.5);
        scene
            .edit(id, |tx| tx.remove_triangles(&[0, 1, 5]))
            .unwrap();
        let added = scene
            .edit(id, |tx| {
                let v = tx.add_vertices(&[[3.5, 3.5, 2.0]])?;
                tx.add_triangles(&[[v[0], 10, 11], [v[0], 11, 18]])
            })
            .unwrap();
        lift(&mut scene, id, 8, -0.5);
        scene
            .edit(id, |tx| {
                tx.remove_triangles(&[added[1]])?;
                tx.set_triangles(&[7], &[[0, 1, 2]])
            })
            .unwrap();
        assert_eq!(scene.history.len(), 5);
        assert_eq!(scene.is_current(id), Ok(false));
        scene.flush().unwrap();
        assert_eq!(scene.is_current(id), Ok(true));
        assert_valid(&scene, id);
        let e = scene.entries.get(id).unwrap();
        let fresh = Bvh::build(&e.mesh, &BuildOptions::default()).unwrap();
        for (lo, hi) in [
            ([-1.0, -1.0, -1.0], [7.0, 7.0, 3.0]),
            ([0.5, 0.5, -1.0], [2.5, 2.5, 3.0]),
            ([2.0, 2.0, 0.5], [4.0, 4.0, 2.5]),
            ([3.0, 0.0, -1.0], [3.2, 6.0, 0.0]),
        ] {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            e.bvh.query_aabb(&e.mesh, lo, hi, &mut a);
            fresh.query_aabb(&e.mesh, lo, hi, &mut b);
            a.sort_unstable();
            b.sort_unstable();
            assert!(!b.is_empty());
            assert_eq!(a, b);
        }
    }

    #[test]
    fn eager_update_failure_leaves_bvh_pending() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(grid(4), MeshOptions::default()).unwrap();
        fault::fail_bvh_update_on(1);
        lift(&mut scene, id, 6, 1.0);
        assert_eq!(scene.mesh(id).unwrap().position(6), Some([1.0, 1.0, 1.0]));
        assert_eq!(scene.history.len(), 1);
        assert_eq!(scene.is_current(id), Ok(false));
        assert!(scene.entries.get(id).unwrap().pending.is_some());
        // A second failing edit merges into the pending changes.
        fault::fail_bvh_update_on(1);
        lift(&mut scene, id, 12, 1.0);
        assert_eq!(scene.is_current(id), Ok(false));
        scene.update(id).unwrap();
        assert_eq!(scene.is_current(id), Ok(true));
        assert!(scene.entries.get(id).unwrap().pending.is_none());
        assert_valid(&scene, id);
        // A failing explicit update reports the error and keeps the changes.
        fault::fail_bvh_update_on(1);
        lift(&mut scene, id, 7, 1.0);
        fault::fail_bvh_update_on(1);
        assert_eq!(
            scene.update(id),
            Err(SceneError::Bvh(fast_mesh_bvh::BvhError::OutOfMemory))
        );
        assert_eq!(scene.is_current(id), Ok(false));
        // Treated as Lazy until updated: flush picks it up.
        scene.flush().unwrap();
        assert_eq!(scene.is_current(id), Ok(true));
        assert_valid(&scene, id);
    }

    #[test]
    fn take_changes_reports_and_resets() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let b = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let first = scene.take_changes();
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].id, a);
        assert_eq!(first[1].id, b);
        assert!(first.iter().all(|c| c.added && c.changes.is_none()));
        assert!(scene.take_changes().is_empty());

        lift(&mut scene, b, 4, 1.0);
        lift(&mut scene, b, 1, 1.0);
        let moved = Placement {
            translation: [1.0, 0.0, 0.0],
            ..Placement::IDENTITY
        };
        scene.set_placement(a, moved).unwrap();
        scene.set_placement(a, Placement::IDENTITY).unwrap();
        let ch = scene.take_changes();
        assert_eq!(ch.len(), 2);
        assert_eq!(ch[0].id, a);
        assert!(ch[0].placement_changed && !ch[0].added && !ch[0].removed);
        assert_eq!(ch[0].changes.as_ref().map(ChangeSet::is_empty), Some(true));
        assert_eq!(ch[1].id, b);
        assert!(!ch[1].placement_changed);
        assert_eq!(ch[1].changes.as_ref().unwrap().vertices_moved, [1, 4]);
        assert!(scene.take_changes().is_empty());

        // An empty edit is neither recorded nor reported.
        let recorded = scene.history.len();
        scene.edit(a, |_| Ok(())).unwrap();
        assert_eq!(scene.history.len(), recorded);
        assert!(scene.take_changes().is_empty());

        // Removal is reported, with the changes made before it.
        lift(&mut scene, a, 0, 1.0);
        scene.remove_mesh(a).unwrap();
        let ch = scene.take_changes();
        assert_eq!(ch.len(), 1);
        assert_eq!(ch[0].id, a);
        assert!(ch[0].removed && !ch[0].added);
        assert_eq!(ch[0].changes.as_ref().unwrap().vertices_moved, [0]);
        assert!(scene.take_changes().is_empty());
    }

    #[test]
    fn added_then_removed_is_not_reported() {
        let mut scene = Scene::new(SceneOptions::default());
        let keep = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        scene.take_changes();
        let gone = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        lift(&mut scene, gone, 0, 1.0);
        scene.remove_mesh(gone).unwrap();
        assert!(scene.take_changes().is_empty());

        // Vertices added and removed again between two calls are not reported.
        let v = scene
            .edit(keep, |tx| tx.add_vertices(&[[9.0, 9.0, 9.0]]))
            .unwrap();
        scene.edit(keep, |tx| tx.remove_vertices(&v)).unwrap();
        let ch = scene.take_changes();
        assert!(ch.is_empty(), "{ch:?}");
    }

    #[test]
    fn compact_reports_everything() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(grid(3), MeshOptions::default()).unwrap();
        scene.take_changes();
        scene.edit(id, |tx| tx.remove_triangles(&[0])).unwrap();
        scene.edit(id, |tx| tx.remove_vertices(&[0])).unwrap();
        let remap = scene.compact(id).unwrap();
        assert_eq!(remap.vertex(0), None);
        assert_eq!(remap.vertex(1), Some(0));
        assert_eq!(remap.triangle(0), None);
        assert_eq!(remap.triangle(1), Some(0));
        assert_eq!(scene.mesh(id).unwrap().vertex_slots(), 15);
        assert_eq!(scene.is_current(id), Ok(true));
        assert_valid(&scene, id);
        let ch = scene.take_changes();
        assert_eq!(ch.len(), 1);
        assert!(ch[0].changes.is_none() && !ch[0].added);
        assert!(scene.take_changes().is_empty());
        // Even a Manual mesh gets its BVH rebuilt.
        let m = scene
            .add_mesh(grid(2), with_mode(UpdateMode::Manual))
            .unwrap();
        scene.edit(m, |tx| tx.remove_triangles(&[3])).unwrap();
        scene.compact(m).unwrap();
        assert_eq!(scene.is_current(m), Ok(true));
        assert_valid(&scene, m);
    }

    #[test]
    fn channel_wrappers() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(grid(1), MeshOptions::default()).unwrap();
        scene
            .add_channel(id, ChannelDesc::derived("curv", 1))
            .unwrap();
        scene.derived_channel_mut(id, "curv").unwrap()[2] = 4.0;
        assert_eq!(scene.mesh(id).unwrap().channel("curv").unwrap()[2], 4.0);
        assert!(matches!(
            scene.add_channel(id, ChannelDesc::derived("curv", 1)),
            Err(SceneError::Store(_))
        ));
        assert_eq!(scene.remove_channel(id, "curv").unwrap().name, "curv");
        assert!(scene.derived_channel_mut(id, "curv").is_err());
        assert_eq!(scene.is_current(id), Ok(true));
    }
}
