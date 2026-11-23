//! Undoable adds and removals: parking a mesh with everything that is its
//! own, and restoring it under the same id.
//!
//! A parked mesh's [`Entry`] stays in its slot of the slot map, moved out of
//! sight rather than copied: its store mesh, BVH, placement, mode, normals,
//! colouring and the fields on it come back as they were. Its slot keeps its
//! generation and is not reused until the history drops the add or removal
//! that is the last way back (see `history::release_in`).

use fast_mesh_store::StoreError;

use crate::contacts::ElasticRestState;
use crate::fields::ParkedFields;
use crate::history::{Change, History, StepHost};
use crate::id::SlotMap;
use crate::report::Report;
use crate::scene::Entry;
use crate::{MeshId, Result, Scene, SceneError};

/// What a parked mesh keeps from the scene's registries beside its entry.
#[derive(Default)]
pub(crate) struct ParkedLinks {
    /// The fields on it (see [`Scene::park_fields_of`]).
    pub(crate) fields: ParkedFields,
    /// Its elastic rest state, if it had one.
    pub(crate) rest: Option<ElasticRestState>,
}

impl Scene {
    /// Removes a mesh as an undo step: recorded in the open step, or as a
    /// step of its own (dropping the redo stack). Undo brings it back under
    /// the same id; until then the id gives [`SceneError::UnknownMesh`], like
    /// that of a mesh removed with [`Scene::remove_mesh`].
    ///
    /// The mesh is parked, not copied: its store mesh (every channel and its
    /// UVs), BVH, placement, mode, normals, colouring, the fields on it and
    /// its elastic rest state are kept as they are and come back with it.
    /// The fields on it are recomputed in full on their next update, since
    /// their targets may have changed meanwhile; a target removed for good
    /// by then leaves the field, and a field left without targets goes,
    /// with its colouring.
    ///
    /// Links that belong to other meshes are cut as by
    /// [`Scene::remove_mesh`] and are not restored: the fields on other
    /// meshes measured to it alone are removed (with colourings by them),
    /// those measured to it and others lose it as a target (recomputed in
    /// full, at once when Eager), it leaves the sections that cut it
    /// (reported in its final [`MeshChanges::sections_changed`](crate::MeshChanges::sections_changed)),
    /// and its collision pair rules go.
    ///
    /// The removal shows up in every subscriber's next
    /// [`Scene::take_changes_for`] (unless the mesh was added since that
    /// subscriber's last call). When it is undone before a subscriber took
    /// the removal, that subscriber gets no removal and the mesh as fully
    /// changed instead; otherwise the mesh is reported as added again.
    ///
    /// The kept mesh counts against the history budget (see
    /// [`SceneOptions::history_budget`](crate::SceneOptions::history_budget))
    /// until the history drops the step, which frees it for good: eviction,
    /// [`Scene::clear_history`], or a new step after it was redone. A
    /// removal larger than the whole budget is dropped as soon as its step
    /// closes, which makes it permanent. [`Scene::is_restorable`] tells
    /// whether the history can still bring a removed mesh back.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id, or
    /// [`SceneError::Store`] when out of memory. The scene is unchanged on
    /// error.
    pub fn remove_mesh_undoable(&mut self, id: MeshId) -> Result<()> {
        self.drop_field_unit();
        self.entry(id)?;
        // Make room first, so recording cannot fail after the removal.
        self.history.reserve()?;
        let kept_on = self.park_mesh(id)?;
        let bytes = self.entries.get_parked(id).map_or(0, Entry::byte_size);
        self.history
            .record(&mut self.entries, Change::Remove { id, bytes });
        // The fields that lost a target are due in full.
        self.run_eager_fields(Some(&kept_on));
        Ok(())
    }

    /// Whether `id` is a mesh removed by an undoable step that undo or redo
    /// can still bring back (see [`Scene::remove_mesh_undoable`] and
    /// [`MeshOptions::undoable`](crate::MeshOptions::undoable)). False for a
    /// live mesh, and for one removed for good.
    pub fn is_restorable(&self, id: MeshId) -> bool {
        self.entries.get_parked(id).is_some()
    }

    /// The meshes [`Scene::is_restorable`] is true for, in index order.
    pub fn restorable_ids(&self) -> impl Iterator<Item = MeshId> + '_ {
        self.entries.parked_ids()
    }

    /// Records the undoable add of live mesh `id`; `History::reserve` must
    /// have succeeded since the last record.
    pub(crate) fn record_add(&mut self, id: MeshId) {
        let bytes = self.entries.get(id).map_or(0, Entry::byte_size);
        self.history
            .record(&mut self.entries, Change::Add { id, bytes });
    }

    /// Runs `f` on the history with the rest of the scene, so that undo and
    /// redo can park and restore meshes. The history is moved out meanwhile
    /// (an empty placeholder allocates nothing) and put back after.
    pub(crate) fn with_history<T>(&mut self, f: impl FnOnce(&mut History, &mut Scene) -> T) -> T {
        let mut history = std::mem::replace(&mut self.history, History::new(0));
        let r = f(&mut history, self);
        self.history = history;
        r
    }

    /// Parks live mesh `id`: cuts the links other meshes own (as
    /// [`Scene::remove_mesh`] does), keeps what is its own, reports the
    /// removal and moves the entry out of sight. Returns the live meshes
    /// whose fields lost it as a target, which are due in full.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] when `id` is not live, or
    /// [`SceneError::Store`] when out of memory; nothing changed then.
    pub(crate) fn park_mesh(&mut self, id: MeshId) -> Result<Vec<MeshId>> {
        if self.entries.get(id).is_none() {
            return Err(SceneError::UnknownMesh);
        }
        // Reserve first so no report can be lost after the removal.
        for s in &mut self.subscribers {
            s.removed
                .try_reserve(1)
                .map_err(|_| SceneError::Store(StoreError::OutOfMemory))?;
        }
        let (fields, kept_on) = self.park_fields_of(id)?;
        // Nothing fails from here on.
        self.drop_sections_of(id);
        self.collision
            .pair_rules
            .retain(|&(a, b), _| a != id && b != id);
        let rest = self.collision.rest_positions.remove(&id);
        let Some(entry) = self.entries.get_mut(id) else {
            return Err(SceneError::Corrupted);
        };
        entry.stroke.invalidate_on_mesh(&mut entry.mesh);
        entry.parked = ParkedLinks { fields, rest };
        for (s, r) in self.subscribers.iter_mut().zip(&mut entry.reports) {
            if r.added {
                // Never seen: nothing to report, now or when it is back.
                *r = Report::clean();
            } else {
                s.removed.push(r.take(id, true));
            }
        }
        self.entries.park(id);
        Ok(kept_on)
    }

    /// Restores parked mesh `id` under the same id, with what it kept, and
    /// reports it: as added, or, to a subscriber that has not taken its
    /// removal yet, as fully changed (the removal is withdrawn). Its BVH and
    /// normals are as they were; its fields are put back (see
    /// [`Scene::unpark_fields`]).
    ///
    /// # Errors
    /// [`SceneError::Corrupted`] when `id` is not parked, or
    /// [`SceneError::Store`] when out of memory; nothing changed then.
    pub(crate) fn unpark_mesh(&mut self, id: MeshId) -> Result<()> {
        let entry = self
            .entries
            .get_parked_mut(id)
            .ok_or(SceneError::Corrupted)?;
        let links = std::mem::take(&mut entry.parked);
        if let Err(e) = self.reserve_unpark_fields(&links.fields) {
            if let Some(entry) = self.entries.get_parked_mut(id) {
                entry.parked = links;
            }
            return Err(e);
        }
        // Nothing fails from here on.
        self.entries.unpark(id);
        let ParkedLinks { fields, rest } = links;
        if let Some(rest) = rest {
            self.collision.rest_positions.insert(id, rest);
        }
        let Some(entry) = self.entries.get_mut(id) else {
            return Err(SceneError::Corrupted);
        };
        entry.content += 1;
        for (s, r) in self.subscribers.iter_mut().zip(&mut entry.reports) {
            *r = match s.removed.iter().position(|c| c.id == id) {
                Some(k) => Report::restored(s.removed.remove(k)),
                None => Report::added(),
            };
        }
        self.unpark_fields(id, fields);
        Ok(())
    }
}

impl StepHost for Scene {
    fn entries(&mut self) -> &mut SlotMap<Entry> {
        &mut self.entries
    }

    fn park(&mut self, id: MeshId) -> Result<Vec<MeshId>> {
        self.park_mesh(id).map_err(|e| {
            // The history only names meshes it can park.
            if e == SceneError::UnknownMesh {
                SceneError::Corrupted
            } else {
                e
            }
        })
    }

    fn unpark(&mut self, id: MeshId) -> Result<()> {
        self.unpark_mesh(id)
    }
}
