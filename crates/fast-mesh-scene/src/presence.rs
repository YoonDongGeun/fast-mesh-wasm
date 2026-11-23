//! Undoable adds and removals: parking a mesh with everything that is its
//! own, and restoring it under the same id.
//!
//! A parked mesh's [`Entry`] stays in its slot of the slot map, out of sight:
//! nothing is moved or copied. Its store mesh, BVH, placement, mode, normals,
//! colouring and the fields on it come back as they were. Its slot keeps its
//! generation and is not reused until the history drops the add or removal
//! that is the last way back (see `history::release_in`).
//!
//! Undo, redo and cancel flip the slots between live and parked while a step
//! runs, which cannot fail. Once the whole step has run,
//! [`Scene::settle_presence`] brings the registries in line (fields, sections,
//! collision rules, reports); a step that fails flips the slots back and never
//! gets there.

use crate::contacts::ElasticRestState;
use crate::fields::ParkedFields;
use crate::history::{Change, Done};
use crate::report::Report;
use crate::scene::Entry;
use crate::{MeshId, Result, Scene};

/// What a parked mesh keeps from the scene's registries beside its entry.
pub(crate) struct ParkedLinks {
    /// The fields on it (see [`Scene::park_fields_of`]).
    fields: ParkedFields,
    /// Its elastic rest state, if it had one.
    rest: Option<ElasticRestState>,
}

impl ParkedLinks {
    /// Approximate heap memory held.
    pub(crate) fn byte_size(&self) -> usize {
        self.fields.byte_size()
            + self
                .rest
                .as_ref()
                .map_or(0, |r| r.rest_positions.capacity() * 12)
    }
}

impl Scene {
    /// Removes a mesh as an undo step: recorded in the open step, or as a
    /// step of its own (dropping the redo stack). Undo brings it back under
    /// the same id; until then the id gives
    /// [`SceneError::UnknownMesh`](crate::SceneError::UnknownMesh), like that
    /// of a mesh removed with [`Scene::remove_mesh`].
    ///
    /// The mesh is parked, not copied: its store mesh (every channel and its
    /// UVs), BVH, placement, mode, normals, colouring, the fields on it and
    /// its elastic rest state stay as they are and come back with it. A
    /// distance field on it is recomputed in full on its next update, since
    /// its targets may have changed meanwhile; a target removed for good by
    /// then leaves the field, and a field left without targets goes, with
    /// its colouring. Until the mesh is back, the ids of the fields on it
    /// give [`SceneError::UnknownField`](crate::SceneError::UnknownField).
    ///
    /// Links that belong to other meshes or to the scene are cut as by
    /// [`Scene::remove_mesh`] and are not restored: the fields on other
    /// meshes measured to it alone are removed (with colourings by them),
    /// those measured to it and others lose it as a target (recomputed in
    /// full, at once when Eager), it leaves the sections that cut it, and its
    /// collision pair rules go.
    ///
    /// The removal shows up in every subscriber's next
    /// [`Scene::take_changes_for`], unless the mesh was added since that
    /// subscriber's last call. When it is undone, the mesh is reported as
    /// added again; to a subscriber that has not taken the removal yet, the
    /// removal is withdrawn and the mesh is reported as fully changed
    /// instead.
    ///
    /// The kept mesh counts against the history budget (see
    /// [`SceneOptions::history_budget`](crate::SceneOptions::history_budget))
    /// while it is out, until the history drops the step and frees it for
    /// good: by eviction or [`Scene::clear_history`]. A removal larger than the whole budget
    /// is dropped as soon as its step closes, which makes it permanent.
    /// [`Scene::is_restorable`] tells whether the history can still bring a
    /// removed mesh back.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`](crate::SceneError::UnknownMesh) for a
    /// stale id, or [`SceneError::Store`](crate::SceneError::Store) when
    /// there is no memory to record the removal. The scene is unchanged on
    /// error.
    pub fn remove_mesh_undoable(&mut self, id: MeshId) -> Result<()> {
        self.drop_field_unit();
        self.entry(id)?;
        // Make room first, so recording cannot fail after the removal.
        self.history.reserve()?;
        self.entries.park(id);
        let kept_on = self.settle_presence(&[Done::Presence(id)]);
        let bytes = self.entries.get_parked(id).map_or(0, Entry::byte_size);
        self.history
            .record(&mut self.entries, Change::Remove { id, bytes });
        // The fields that lost a target are due in full.
        self.run_eager_fields(Some(&kept_on));
        Ok(())
    }

    /// Whether `id` is a mesh removed by an undoable step (or whose undoable
    /// add was undone) that undo or redo can still bring back: see
    /// [`Scene::remove_mesh_undoable`] and
    /// [`MeshOptions::undoable`](crate::MeshOptions::undoable). False for a
    /// live mesh, and for one removed for good.
    pub fn is_restorable(&self, id: MeshId) -> bool {
        self.entries.get_parked(id).is_some()
    }

    /// The meshes [`Scene::is_restorable`] is true for, in index order.
    pub fn restorable_ids(&self) -> impl Iterator<Item = MeshId> + '_ {
        self.entries.parked_ids()
    }

    /// The id of the restorable mesh (see [`Scene::is_restorable`]) with this
    /// [`MeshId::index`] and [`MeshId::generation`], or `None` when there is
    /// none. The counterpart of [`Scene::mesh_id`], for ids passed back
    /// across an FFI boundary.
    pub fn restorable_id(&self, index: u32, generation: u32) -> Option<MeshId> {
        self.entries.parked_id(index, generation)
    }

    /// Records the undoable add of live mesh `id`. `History::reserve` must
    /// have succeeded since the last record. It holds nothing until it is
    /// undone: the mesh is measured then.
    pub(crate) fn record_add(&mut self, id: MeshId) {
        self.history
            .record(&mut self.entries, Change::Add { id, bytes: 0 });
    }

    /// Brings the scene's registries in line with the meshes a step parked
    /// or restored (the [`Done::Presence`] entries of `done`; a mesh flipped
    /// back and forth within the step is left alone). Returns the live
    /// meshes whose fields lost a target, which are due in full.
    ///
    /// Every field on a newly parked mesh is taken out before any field is
    /// dropped for losing a target: meshes parked in one step keep their
    /// fields between each other, and get them back together.
    ///
    /// Its small bookkeeping allocations are infallible, like the merging of
    /// change reports.
    pub(crate) fn settle_presence(&mut self, done: &[Done]) -> Vec<MeshId> {
        let mut parked = Vec::new();
        let mut restored = Vec::new();
        for d in done {
            let Done::Presence(id) = d else { continue };
            let is_parked = self.entries.get_parked(*id).is_some();
            let Some(entry) = self.entries.get_any_mut(*id) else {
                continue;
            };
            if is_parked == entry.parked.is_some() {
                continue;
            }
            let list = if is_parked {
                &mut parked
            } else {
                &mut restored
            };
            if !list.contains(id) {
                list.push(*id);
            }
        }
        for &id in &parked {
            let links = ParkedLinks {
                fields: self.park_fields_of(id),
                rest: self.collision.rest_positions.remove(&id),
            };
            if let Some(entry) = self.entries.get_parked_mut(id) {
                entry.parked = Some(links);
            }
        }
        let mut kept_on = Vec::new();
        for &id in &parked {
            kept_on.extend(self.drop_fields_of(id));
            self.drop_sections_of(id);
            self.collision.drop_mesh(id);
            if let Some(entry) = self.entries.get_parked_mut(id) {
                entry.stroke.invalidate_on_mesh(&mut entry.mesh);
                for (s, r) in self.subscribers.iter_mut().zip(&mut entry.reports) {
                    if r.added {
                        // Never told of it: nothing to report.
                        *r = Report::clean();
                    } else {
                        s.removed.push(r.take(id, true));
                    }
                }
            }
        }
        for &id in &restored {
            let Some(entry) = self.entries.get_mut(id) else {
                continue;
            };
            let links = entry.parked.take();
            entry.content += 1;
            for (s, r) in self.subscribers.iter_mut().zip(&mut entry.reports) {
                *r = match s.removed.iter().position(|c| c.id == id) {
                    Some(k) => Report::restored(s.removed.remove(k)),
                    None => Report::added(),
                };
            }
            if let Some(ParkedLinks { fields, rest }) = links {
                if let Some(rest) = rest {
                    self.collision.rest_positions.insert(id, rest);
                }
                self.unpark_fields(id, fields);
            }
        }
        kept_on.retain(|&id| self.entries.get(id).is_some());
        kept_on.sort_unstable();
        kept_on.dedup();
        kept_on
    }
}
