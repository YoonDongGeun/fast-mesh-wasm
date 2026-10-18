//! The scene: a registry of meshes with BVHs and placements.

use fast_mesh_bvh::{BuildOptions, Bvh, UpdatePolicy};
use fast_mesh_store::{ChangeSet, ChannelDesc, Mesh, Remap, StoreError, Transaction, VertexId};

use crate::fields::{ColorState, Field, FieldHook};
use crate::history::{Change, Done, History};
use crate::id::SlotMap;
use crate::report::Report;
use crate::sections::{Section, SectionHook};
use crate::{FieldId, MeshChanges, MeshId, Placement, Result, SceneError, SectionId, SubscriberId};

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
    /// The BVH must be rebuilt from scratch: the mesh's content is not
    /// described by `pending` (after a [`SceneError::Corrupted`] undo).
    pub(crate) rebuild: bool,
    /// One pending report per subscriber, in the order of
    /// `Scene::subscribers`.
    pub(crate) reports: Vec<Report>,
    /// One pending change accumulator per field this mesh is part of (as
    /// mesh `on` or `to`): the internal hook field updates read.
    pub(crate) hooks: Vec<FieldHook>,
    /// One pending change accumulator per section selecting this mesh.
    pub(crate) section_hooks: Vec<SectionHook>,
    /// The mesh's colouring and its colours, if it has one.
    pub(crate) coloring: Option<ColorState>,
}

impl Entry {
    /// Marks the mesh as changed in unknown ways, after a failed recovery:
    /// it is reported as fully changed, and the next BVH update rebuilds
    /// from scratch even if the revision happens to match. The caller then
    /// calls [`Entry::after_change`], which rebuilds an Eager BVH right away.
    pub(crate) fn flag_unknown(&mut self) {
        self.pending = Some(ChangeSet::default());
        self.rebuild = true;
        self.report_everything();
    }

    /// Reports the mesh as fully changed to every subscriber, colours
    /// included (vertex ids may have been renumbered), and to every field
    /// and section hook.
    pub(crate) fn report_everything(&mut self) {
        for r in &mut self.reports {
            r.changes = None;
            r.colors_dirty = None;
        }
        for h in &mut self.hooks {
            h.changes = None;
        }
        for h in &mut self.section_hooks {
            h.changes = None;
        }
    }

    /// Reports a placement change to every subscriber, field hook and
    /// section hook.
    pub(crate) fn report_placement(&mut self) {
        for r in &mut self.reports {
            r.placement_changed = true;
        }
        for h in &mut self.hooks {
            h.placement_changed = true;
        }
        for h in &mut self.section_hooks {
            h.placement_changed = true;
        }
    }

    /// Reports recoloured vertices to every subscriber (`None`: all).
    pub(crate) fn report_colors(&mut self, ids: Option<&[VertexId]>) {
        for r in &mut self.reports {
            r.colors_changed(ids);
        }
    }

    /// Reports a change of field `id`'s values to every subscriber.
    pub(crate) fn report_field(&mut self, id: FieldId) {
        for r in &mut self.reports {
            r.field_changed(id);
        }
    }

    /// Reports a change of section `id`'s result for this mesh to every
    /// subscriber.
    pub(crate) fn report_section(&mut self, id: SectionId) {
        for r in &mut self.reports {
            r.section_changed(id);
        }
    }

    /// Reports a write to channel `name` to every subscriber.
    fn report_channel(&mut self, name: &str) {
        for r in &mut self.reports {
            r.channel_written(name);
        }
    }

    /// Folds a commit's (or undo's or redo's) changes into the BVH's pending
    /// changes, into every subscriber's report and into every field and
    /// section hook.
    pub(crate) fn record_changes(&mut self, changes: &ChangeSet) {
        match &mut self.pending {
            Some(p) => p.merge(changes),
            None => self.pending = Some(changes.clone()),
        }
        for r in &mut self.reports {
            r.merge(changes);
        }
        for h in &mut self.hooks {
            if let Some(c) = &mut h.changes {
                c.merge(changes);
            }
        }
        for h in &mut self.section_hooks {
            if let Some(c) = &mut h.changes {
                c.merge(changes);
            }
        }
        // Slots only grow through commits, undo, redo and cancel, all of
        // which pass here.
        self.grow_colors();
    }

    /// Whether the BVH is up to date with the mesh: not flagged for a
    /// rebuild, and current by the BVH's own check.
    pub(crate) fn is_current(&self) -> bool {
        !self.rebuild && self.bvh.is_current(&self.mesh)
    }

    /// Brings the BVH up to date with the pending changes. On error the
    /// pending changes are kept, so a later call can retry.
    pub(crate) fn update_bvh(&mut self) -> Result<()> {
        if self.rebuild {
            #[cfg(test)]
            fault::before_bvh_update()?;
            #[cfg(test)]
            fault::count_rebuild();
            let mut bvh = Bvh::build(&self.mesh, self.bvh.options())?;
            bvh.set_policy(*self.bvh.policy())?;
            self.bvh = bvh;
            self.rebuild = false;
            self.pending = None;
            return Ok(());
        }
        if self.bvh.is_current(&self.mesh) {
            self.pending = None;
            return Ok(());
        }
        #[cfg(test)]
        fault::before_bvh_update()?;
        // Invariant: whenever the BVH is stale, `pending` is `Some` (every
        // commit, undo, redo and cancel goes through `record_changes`, and
        // `compact`, which changes the epoch, sets it). The BVH rebuilds on an
        // epoch change whatever the changes say, so the empty fallback is
        // only a guard.
        debug_assert!(self.pending.is_some(), "stale BVH without pending changes");
        let none = ChangeSet::default();
        let changes = self.pending.as_ref().unwrap_or(&none);
        self.bvh.update(&mut self.mesh, changes)?;
        self.pending = None;
        Ok(())
    }

    /// After a change: brings an Eager BVH up to date. A failure is not an
    /// error of the change (§15.1): the pending changes stay and the BVH is
    /// treated like Lazy until an update succeeds.
    pub(crate) fn after_change(&mut self) {
        if self.mode == UpdateMode::Eager {
            let _ = self.update_bvh();
        }
    }
}

/// A consumer of change reports.
struct Subscriber {
    id: SubscriberId,
    /// Reports of meshes removed since its last `take_changes_for`.
    removed: Vec<MeshChanges>,
}

/// Several meshes, each with a BVH and a placement in the world.
pub struct Scene {
    pub(crate) entries: SlotMap<Entry>,
    /// The subscribers in ascending id order; the default one comes first
    /// and is never removed. Each entry's `reports` follow this order.
    subscribers: Vec<Subscriber>,
    /// The id the next `subscribe` hands out.
    next_subscriber: u64,
    /// The shared undo history across all meshes.
    history: History,
    /// The registered fields, in registration order.
    pub(crate) fields: Vec<Field>,
    /// The id the next `add_distance_field` hands out.
    pub(crate) next_field: u32,
    /// The registered sections, in registration order.
    pub(crate) sections: Vec<Section>,
    /// The id the next `add_section` hands out.
    pub(crate) next_section: u32,
    /// Test-only count of section entry recomputes.
    #[cfg(test)]
    pub(crate) section_recomputes: usize,
}

impl Scene {
    /// An empty scene.
    pub fn new(options: SceneOptions) -> Scene {
        Scene {
            entries: SlotMap::new(),
            subscribers: vec![Subscriber {
                id: SubscriberId::DEFAULT,
                removed: Vec::new(),
            }],
            next_subscriber: SubscriberId::DEFAULT.0 + 1,
            history: History::new(options.history_budget),
            fields: Vec::new(),
            next_field: 0,
            sections: Vec::new(),
            next_section: 0,
            #[cfg(test)]
            section_recomputes: 0,
        }
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
            rebuild: false,
            reports: vec![Report::added(); self.subscribers.len()],
            hooks: Vec::new(),
            section_hooks: Vec::new(),
            coloring: None,
        })
    }

    /// Removes a mesh and returns it. The id never matches again. Its
    /// history entries (undo, redo and in an open step) are dropped, steps
    /// left empty with them, and its patches release the slots they reserve.
    /// The removal shows up in every subscriber's next
    /// [`Scene::take_changes_for`], unless the mesh was added since that
    /// subscriber's last call.
    ///
    /// The fields on the mesh and those measured to it are removed with it
    /// (see [`Scene::remove_field`]), and so are colourings by them. The
    /// mesh leaves every section selecting it, which is reported in its
    /// final [`MeshChanges::sections_changed`].
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id.
    pub fn remove_mesh(&mut self, id: MeshId) -> Result<Mesh> {
        self.entry(id)?;
        // Reserve first so no report can be lost after the removal.
        for s in &mut self.subscribers {
            s.removed
                .try_reserve(1)
                .map_err(|_| SceneError::Store(StoreError::OutOfMemory))?;
        }
        self.drop_fields_of(id);
        self.drop_sections_of(id);
        let mut entry = self.entries.remove(id).ok_or(SceneError::UnknownMesh)?;
        self.history.forget(id, Some(&mut entry.mesh));
        for (s, r) in self.subscribers.iter_mut().zip(&mut entry.reports) {
            if !r.added {
                s.removed.push(r.take(id, true));
            }
        }
        Ok(entry.mesh)
    }

    /// Runs `f` in a transaction on mesh `id`, commits it, and records the
    /// patch in the history: in the open step, or as a step of its own when
    /// none is open. Recording drops the redo stack. The changes are added to
    /// the mesh's report and to its BVH's pending changes; an Eager BVH is
    /// then brought up to date, and so are the Eager fields and section
    /// entries involving the mesh (a failed update leaves them stale and
    /// never fails the edit).
    ///
    /// An empty transaction records and reports nothing.
    ///
    /// If the Eager BVH update fails (for example out of memory), the edit
    /// still succeeds: the BVH keeps its pending changes and is treated like
    /// Lazy until [`Scene::update`], [`Scene::flush`] or a query updates it.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id, or [`SceneError::Store`]
    /// when `f` fails, the commit fails or there is no memory to record the
    /// patch. The transaction is then rolled back and the scene (including
    /// an open step) is unchanged.
    pub fn edit<R>(
        &mut self,
        id: MeshId,
        f: impl FnOnce(&mut Transaction) -> fast_mesh_store::Result<R>,
    ) -> Result<R> {
        let entry = self.entries.get_mut(id).ok_or(SceneError::UnknownMesh)?;
        // Make room first, so recording cannot fail after the commit.
        self.history.reserve()?;
        let mut tx = entry.mesh.begin_edit();
        // Dropping the transaction on error rolls it back.
        let r = f(&mut tx)?;
        let patch = tx.commit()?;
        if patch.is_empty() {
            return Ok(r);
        }
        entry.record_changes(patch.changes());
        self.history
            .record(&mut self.entries, Change::Edit(id, patch));
        if let Some(entry) = self.entries.get_mut(id) {
            entry.after_change();
        }
        self.run_eager(Some(&[id]));
        Ok(r)
    }

    /// Opens a history step, or one more nesting level of the open step.
    /// Edits and placement changes until the matching [`Scene::end_step`]
    /// are undone and redone together, across meshes.
    pub fn begin_step(&mut self) {
        self.history.begin();
    }

    /// Ends one nesting level of the open step; the outermost call closes
    /// it. A step with no entries is not recorded. Closing may evict the
    /// oldest steps to meet the history budget.
    ///
    /// # Errors
    /// [`SceneError::StepNotOpen`] when no step is open.
    pub fn end_step(&mut self) -> Result<()> {
        self.history.end(&mut self.entries)
    }

    /// Reverts the open step's entries, in reverse order, and discards the
    /// step, however deeply nested (for example on Esc mid-drag). The
    /// reverts are reported and update BVHs like an undo.
    ///
    /// # Errors
    /// [`SceneError::StepNotOpen`] when no step is open. A store error (for
    /// example out of memory) leaves the meshes and the open step unchanged;
    /// [`SceneError::Corrupted`] means that even restoring them failed, with
    /// the consequences described for [`Scene::undo`].
    pub fn cancel_step(&mut self) -> Result<()> {
        let done = self.history.cancel(&mut self.entries);
        let done = self.after_corrupted(done)?;
        self.absorb(done);
        Ok(())
    }

    /// Passes a history result through; after [`SceneError::Corrupted`]
    /// (meshes flagged as fully changed) runs every Eager field and section
    /// entry first.
    fn after_corrupted<T>(&mut self, r: Result<T>) -> Result<T> {
        if r.as_ref().err() == Some(&SceneError::Corrupted) {
            self.run_eager(None);
        }
        r
    }

    /// Updates the Eager fields and section entries involving `touched`
    /// (all with `None`), swallowing failures (§6.3).
    fn run_eager(&mut self, touched: Option<&[MeshId]>) {
        self.run_eager_fields(touched);
        self.run_eager_sections(touched);
    }

    /// Closes an open step, then undoes the latest step: its entries are
    /// reverted in reverse order, all or nothing. Returns `false` when there
    /// is nothing to undo. The changes are reported and reach the BVHs as
    /// for [`Scene::edit`]; a placement entry gets its old value back and is
    /// reported as `placement_changed`.
    ///
    /// A non-empty open step is the latest step, and it is undone even when
    /// it is larger than the whole history budget (Ctrl+Z mid-drag undoes
    /// the drag). The budget is applied afterwards, which may evict the
    /// oldest undo steps and, for a step larger than the budget, drop it
    /// from the redo stack again.
    ///
    /// # Errors
    /// A store error (for example out of memory): the meshes, both stacks
    /// and an open step are then unchanged (the step stays open).
    /// [`SceneError::Corrupted`] if restoring them after such an error
    /// failed too: the step stays where it was, the meshes it edits may be
    /// partly changed, each of them is reported as fully changed
    /// (`changes: None`) and its BVH is rebuilt from scratch on its next
    /// update (right away for Eager), and each mesh with a placement entry
    /// in it is reported as `placement_changed`. The history may no longer
    /// match the meshes, so the caller should consider
    /// [`Scene::clear_history`].
    pub fn undo(&mut self) -> Result<bool> {
        let done = self.history.undo(&mut self.entries);
        let done = self.after_corrupted(done)?;
        Ok(done.map(|d| self.absorb(d)).is_some())
    }

    /// Closes an open step, then redoes the latest undone step: its entries
    /// are re-applied in order, all or nothing. Returns `false` when there is
    /// nothing to redo. Reporting and errors as for [`Scene::undo`].
    ///
    /// # Errors
    /// As for [`Scene::undo`].
    pub fn redo(&mut self) -> Result<bool> {
        let done = self.history.redo(&mut self.entries);
        let done = self.after_corrupted(done)?;
        Ok(done.map(|d| self.absorb(d)).is_some())
    }

    /// True if [`Scene::undo`] would undo something, counting a non-empty
    /// open step.
    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    /// True if [`Scene::redo`] would redo something.
    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// Closes an open step, then drops every undo and redo step, releasing
    /// the slots their patches reserve. The meshes are not changed.
    ///
    /// # Errors
    /// None at present; the `Result` leaves room for future checks.
    pub fn clear_history(&mut self) -> Result<()> {
        self.history.clear(&mut self.entries);
        Ok(())
    }

    /// Feeds the outcome of an undo, redo or cancel into the reports and
    /// pending changes, then runs the Eager updates (BVHs, then fields and
    /// sections), as after an edit.
    fn absorb(&mut self, done: Vec<Done>) {
        for (id, changes) in &done {
            if let Some(e) = self.entries.get_mut(*id) {
                match changes {
                    Some(c) => e.record_changes(c),
                    None => e.report_placement(),
                }
            }
        }
        // Placement entries leave the BVHs alone.
        for (id, _) in done.iter().filter(|(_, c)| c.is_some()) {
            if let Some(e) = self.entries.get_mut(*id) {
                e.after_change();
            }
        }
        let touched: Vec<MeshId> = done.iter().map(|(id, _)| *id).collect();
        self.run_eager(Some(&touched));
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
        self.run_eager(Some(&[id]));
        Ok(())
    }

    /// Brings the mesh's BVH up to date now, whatever its mode. Eager fields
    /// involving the mesh that were left stale (for example by this BVH
    /// being a stale Manual one) then catch up, and so does every stale
    /// section entry of the mesh, whatever its mode.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`], or a BVH or memory error from the update.
    /// On error the BVH keeps its pending changes. Then the first section
    /// error (for example out of memory); such entries stay stale.
    pub fn update(&mut self, id: MeshId) -> Result<()> {
        self.entry_mut(id)?.update_bvh()?;
        self.run_eager_fields(Some(&[id]));
        self.run_sections(|_| true, Some(&[id]))
    }

    /// Brings every Lazy BVH up to date, and every Eager one left stale by a
    /// failed update; then does the same for the fields, which follow the
    /// mode of the mesh they are on (see [`Scene::update_fields`]), and for
    /// the section entries, which follow the mode of their mesh (see
    /// [`Scene::update_sections`]). Manual BVHs, and fields and section
    /// entries on Manual meshes, are left alone.
    ///
    /// # Errors
    /// The first BVH, memory or field error, for example
    /// [`SceneError::StaleBvh`] for a field measured to a stale Manual mesh:
    /// such a field makes every `flush` return `StaleBvh` until that BVH is
    /// brought up to date with [`Scene::update`]. Every mesh and field is
    /// still attempted; those that failed keep their pending changes.
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
        let r = self.run_fields(|mode| mode != UpdateMode::Manual, None);
        if first.is_ok() {
            first = r;
        }
        let r = self.run_sections(|mode| mode != UpdateMode::Manual, None);
        if first.is_ok() {
            first = r;
        }
        first
    }

    /// Everything that happened to the meshes since the last call, one entry
    /// per affected mesh in id order, then resets every report of the
    /// default subscriber. Other subscribers keep theirs (see
    /// [`Scene::subscribe`]).
    ///
    /// A mesh added and removed between two calls is not reported, and
    /// neither is a mesh whose changes cancel out (for example vertices added
    /// and removed again).
    pub fn take_changes(&mut self) -> Vec<MeshChanges> {
        // The default subscriber is always first.
        self.take_at(0)
    }

    /// Adds a consumer of change reports and returns its id. It starts with
    /// a clean report for every mesh, so it sees only what happens from now
    /// on, and drains its reports with [`Scene::take_changes_for`]. Every
    /// subscriber is told about the same changes, merged the same way.
    /// A subscriber that never drains keeps accumulating reports (merged
    /// change sets and removals) until it is drained or unsubscribed.
    pub fn subscribe(&mut self) -> SubscriberId {
        let id = SubscriberId(self.next_subscriber);
        self.next_subscriber += 1;
        for (_, entry) in self.entries.iter_mut() {
            entry.reports.push(Report::clean());
        }
        self.subscribers.push(Subscriber {
            id,
            removed: Vec::new(),
        });
        id
    }

    /// Removes a subscriber and drops its pending reports. Its id is
    /// rejected from then on.
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] for an unknown or already removed id, or
    /// for the default subscriber (which backs [`Scene::take_changes`]).
    pub fn unsubscribe(&mut self, id: SubscriberId) -> Result<()> {
        if id == SubscriberId::DEFAULT {
            return Err(SceneError::InvalidInput(
                "the default subscriber cannot be removed",
            ));
        }
        let i = self.subscriber_index(id)?;
        self.subscribers.remove(i);
        for (_, entry) in self.entries.iter_mut() {
            entry.reports.remove(i);
        }
        Ok(())
    }

    /// [`Scene::take_changes`] for subscriber `id`: everything that happened
    /// to the meshes since its last call (or since it subscribed), one entry
    /// per affected mesh in id order, then resets its reports.
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] for an unknown or removed subscriber.
    pub fn take_changes_for(&mut self, id: SubscriberId) -> Result<Vec<MeshChanges>> {
        let i = self.subscriber_index(id)?;
        Ok(self.take_at(i))
    }

    fn subscriber_index(&self, id: SubscriberId) -> Result<usize> {
        self.subscribers
            .binary_search_by_key(&id, |s| s.id)
            .map_err(|_| SceneError::InvalidInput("unknown or removed subscriber"))
    }

    /// Drains the reports of the subscriber at index `i`.
    fn take_at(&mut self, i: usize) -> Vec<MeshChanges> {
        let mut out = std::mem::take(&mut self.subscribers[i].removed);
        for (id, entry) in self.entries.iter_mut() {
            let r = &mut entry.reports[i];
            if r.is_clean() {
                *r = Report::clean();
            } else {
                out.push(r.take(id, false));
            }
        }
        out.sort_by_key(|c| c.id);
        out
    }

    /// Adds a channel to the mesh, filled with its default value. Not
    /// undoable, as in the store; reported in
    /// [`MeshChanges::channels_written`]. Names starting with `field:` are
    /// reserved for the scene's fields.
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] for a reserved name,
    /// [`SceneError::UnknownMesh`], or the store's error (for example a
    /// duplicate name). The scene is unchanged on error.
    pub fn add_channel(&mut self, id: MeshId, desc: ChannelDesc) -> Result<()> {
        check_name(&desc.name)?;
        let entry = self.entry_mut(id)?;
        let name = desc.name.clone();
        entry.mesh.add_channel(desc)?;
        entry.report_channel(&name);
        Ok(())
    }

    /// Removes a channel from the mesh and returns its description. Not
    /// undoable, as in the store; reported in
    /// [`MeshChanges::channels_written`]. Recorded patches skip the removed
    /// channel from then on, so undo and redo will not restore the values of
    /// a removed source channel.
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] for a reserved name (starting with
    /// `field:`), [`SceneError::UnknownMesh`], or the store's error for an
    /// unknown name.
    pub fn remove_channel(&mut self, id: MeshId, name: &str) -> Result<ChannelDesc> {
        check_name(name)?;
        let entry = self.entry_mut(id)?;
        let desc = entry.mesh.remove_channel(name)?;
        entry.report_channel(name);
        Ok(desc)
    }

    /// Writable values of a derived channel. Writes are not undoable, as in
    /// the store. The call itself is reported in
    /// [`MeshChanges::channels_written`], since the slice may be written
    /// after it returns.
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] for a reserved name (starting with
    /// `field:`), [`SceneError::UnknownMesh`], or the store's error for an
    /// unknown or non-derived channel.
    pub fn derived_channel_mut(&mut self, id: MeshId, name: &str) -> Result<&mut [f32]> {
        check_name(name)?;
        let entry = self.entry_mut(id)?;
        // Look the channel up first, so a failed call reports nothing.
        entry.mesh.derived_channel_mut(name)?;
        entry.report_channel(name);
        Ok(entry.mesh.derived_channel_mut(name)?)
    }

    /// Removes the mesh's dead slots and renumbers the rest (see
    /// [`Mesh::compact`]). The mesh's recorded patches no longer apply, so
    /// its edit entries leave the undo and redo stacks and an open step;
    /// steps left empty are dropped. Its placement entries stay valid and
    /// are kept. Nothing is released: the reserved slots no longer exist.
    /// Its BVH is rebuilt whatever its mode, and the next
    /// [`Scene::take_changes`] reports it as fully changed (`changes: None`,
    /// `colors_dirty: None`). Its field channels and colours are remapped
    /// with it; the fields involving it are recomputed in full on their
    /// next update, and so are its section entries.
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
        self.history.forget(id, None);
        // Every slot id changed; the BVH update rebuilds from scratch because
        // the mesh's epoch changed, whatever the pending changes say.
        entry.pending = Some(ChangeSet::default());
        entry.report_everything();
        let _ = entry.update_bvh();
        // The field channels were remapped with the mesh; so are the colours.
        entry.recolor_all();
        self.run_eager(Some(&[id]));
        Ok(remap)
    }

    pub(crate) fn entry(&self, id: MeshId) -> Result<&Entry> {
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

    /// Moves the mesh. Stores the placement, reports it as
    /// `placement_changed` and records the old and new value in the history
    /// (in the open step, or as a step of its own), which drops the redo
    /// stack. Setting the current placement again records and reports
    /// nothing. No BVH is touched; Eager fields and section entries
    /// involving the mesh are updated (a failure leaves them stale).
    ///
    /// Within an open step, consecutive placement changes of the same mesh
    /// merge into one entry that keeps the first old value and the last new
    /// one, so a drag of any length records one entry and undo still
    /// restores the placement from before the drag. Any other entry in
    /// between (an edit, or another mesh's placement) starts a new entry.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`], [`SceneError::InvalidInput`] when the
    /// placement is not a proper rotation, or [`SceneError::Store`] when
    /// there is no memory to record it. The scene is unchanged on error.
    pub fn set_placement(&mut self, id: MeshId, placement: Placement) -> Result<()> {
        placement.validate()?;
        let entry = self.entries.get_mut(id).ok_or(SceneError::UnknownMesh)?;
        let old = entry.placement;
        if old == placement {
            return Ok(());
        }
        self.history.reserve()?;
        entry.placement = placement;
        entry.report_placement();
        self.history
            .record_placement(&mut self.entries, id, old, placement);
        self.run_eager(Some(&[id]));
        Ok(())
    }

    /// Whether the mesh's BVH is up to date with the mesh.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id.
    pub fn is_current(&self, id: MeshId) -> Result<bool> {
        Ok(self.entry(id)?.is_current())
    }

    /// Makes the BVHs a query needs current (§15.4): those of `only`, or of
    /// every mesh when `None`. Every non-Manual BVH among them is updated
    /// (Lazy ones, and Eager ones left stale by a failed update); then a
    /// stale Manual BVH, including one flagged for a rebuild, fails the
    /// query with [`SceneError::StaleBvh`] for the lowest such id. Ids in
    /// `only` must be known.
    ///
    /// # Errors
    /// The first BVH or memory error from an update, then
    /// [`SceneError::StaleBvh`].
    pub(crate) fn prepare(&mut self, only: Option<&[MeshId]>) -> Result<()> {
        let wanted = |id: MeshId| only.is_none_or(|ids| ids.contains(&id));
        for (id, entry) in self.entries.iter_mut() {
            if wanted(id) && entry.mode != UpdateMode::Manual {
                entry.update_bvh()?;
            }
        }
        match self
            .entries
            .iter()
            .find(|(id, e)| wanted(*id) && e.mode == UpdateMode::Manual && !e.is_current())
        {
            Some((id, _)) => Err(SceneError::StaleBvh(id)),
            None => Ok(()),
        }
    }
}

/// The channel-name prefix reserved for the scene's fields.
pub(crate) const FIELD_PREFIX: &str = "field:";

/// Rejects a channel name in the reserved `field:` namespace.
fn check_name(name: &str) -> Result<()> {
    if name.starts_with(FIELD_PREFIX) {
        return Err(SceneError::InvalidInput(
            "channel names starting with `field:` are reserved",
        ));
    }
    Ok(())
}

/// Test-only fault injection for the Scene's calls into lower crates, whose
/// own fault hooks are crate-private.
#[cfg(test)]
pub(crate) mod fault {
    use std::cell::Cell;

    use fast_mesh_bvh::BvhError;
    use fast_mesh_store::StoreError;

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

    thread_local! {
        /// Patch apply/revert calls counted since the first pending fault.
        static PATCH_CALLS: Cell<u32> = const { Cell::new(0) };
        /// Bit `k` set: call `k` (counted in `PATCH_CALLS`) fails.
        static PATCH_FAILS: Cell<u64> = const { Cell::new(0) };
    }

    /// Makes the `n`th coming patch apply or revert of a history entry
    /// (1 = the next) fail with the store's `OutOfMemory` without reaching
    /// the store. Fires once; several calls arm several failures, each
    /// counted from now.
    pub(crate) fn fail_patch_on(n: u32) {
        let at = PATCH_CALLS.with(Cell::get) + n;
        assert!(n > 0 && at < 64);
        PATCH_FAILS.with(|f| f.set(f.get() | 1 << at));
    }

    /// Checked just before every patch apply or revert of a history entry.
    pub(crate) fn before_patch() -> Result<(), StoreError> {
        let fails = PATCH_FAILS.with(Cell::get);
        if fails == 0 {
            return Ok(());
        }
        let call = PATCH_CALLS.with(|c| {
            c.set(c.get() + 1);
            c.get()
        });
        if fails & 1 << call == 0 {
            return Ok(());
        }
        let left = fails & !(1 << call);
        PATCH_FAILS.with(|f| f.set(left));
        if left == 0 {
            PATCH_CALLS.with(|c| c.set(0));
        }
        Err(StoreError::OutOfMemory)
    }

    thread_local! {
        /// Rebuilds of a BVH flagged for one (after `Corrupted`).
        static REBUILDS: Cell<u32> = const { Cell::new(0) };
    }

    /// Counts one rebuild of a flagged BVH.
    pub(crate) fn count_rebuild() {
        REBUILDS.with(|c| c.set(c.get() + 1));
    }

    /// Rebuilds of flagged BVHs on this thread since the last call.
    pub(crate) fn take_rebuilds() -> u32 {
        REBUILDS.with(|c| c.replace(0))
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
        assert_eq!(scene.history.undo_len(), 1);
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
        assert_eq!(scene.history.undo_len(), 0);
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
        assert_eq!(scene.history.undo_len(), 5);
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
        assert_eq!(scene.history.undo_len(), 1);
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
        let recorded = scene.history.undo_len();
        scene.edit(a, |_| Ok(())).unwrap();
        assert_eq!(scene.history.undo_len(), recorded);
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

    fn z(scene: &Scene, id: MeshId, v: u32) -> f32 {
        scene.mesh(id).unwrap().position(v).unwrap()[2]
    }

    fn moved(x: f64) -> Placement {
        Placement {
            translation: [x, 0.0, 0.0],
            ..Placement::IDENTITY
        }
    }

    #[test]
    fn step_groups_edits_across_meshes() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let b = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        assert!(!scene.can_undo() && !scene.can_redo());
        scene.begin_step();
        lift(&mut scene, a, 0, 1.0);
        lift(&mut scene, b, 1, 2.0);
        scene.set_placement(a, moved(3.0)).unwrap();
        lift(&mut scene, a, 0, 4.0);
        assert!(scene.can_undo());
        scene.end_step().unwrap();
        assert_eq!(scene.history.undo_len(), 1);
        scene.take_changes();

        assert_eq!(scene.undo(), Ok(true));
        assert_eq!((z(&scene, a, 0), z(&scene, b, 1)), (0.0, 0.0));
        assert_eq!(scene.placement(a), Ok(Placement::IDENTITY));
        assert!(!scene.can_undo() && scene.can_redo());
        let ch = scene.take_changes();
        assert_eq!(ch.len(), 2);
        assert!(ch[0].placement_changed);
        assert_eq!(ch[0].changes.as_ref().unwrap().vertices_moved, [0]);
        assert_eq!(ch[1].changes.as_ref().unwrap().vertices_moved, [1]);
        assert_eq!(scene.undo(), Ok(false));

        assert_eq!(scene.redo(), Ok(true));
        assert_eq!((z(&scene, a, 0), z(&scene, b, 1)), (4.0, 2.0));
        assert_eq!(scene.placement(a), Ok(moved(3.0)));
        assert!(scene.can_undo() && !scene.can_redo());
        assert_eq!(scene.redo(), Ok(false));
        assert_valid(&scene, a);
        assert_valid(&scene, b);
    }

    #[test]
    fn implicit_steps_per_operation() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let b = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        lift(&mut scene, a, 0, 1.0);
        lift(&mut scene, b, 0, 1.0);
        scene.set_placement(a, moved(1.0)).unwrap();
        // Empty edits, unchanged placements and empty steps record nothing.
        scene.edit(a, |_| Ok(())).unwrap();
        scene.set_placement(a, moved(1.0)).unwrap();
        scene.begin_step();
        scene.end_step().unwrap();
        assert_eq!(scene.history.undo_len(), 3);
        scene.take_changes();

        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(scene.placement(a), Ok(Placement::IDENTITY));
        assert_eq!((z(&scene, a, 0), z(&scene, b, 0)), (1.0, 1.0));
        let ch = scene.take_changes();
        assert_eq!(ch.len(), 1);
        assert!(ch[0].placement_changed);
        assert_eq!(ch[0].changes.as_ref().map(ChangeSet::is_empty), Some(true));
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!((z(&scene, a, 0), z(&scene, b, 0)), (1.0, 0.0));
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(z(&scene, a, 0), 0.0);
        assert_eq!(scene.undo(), Ok(false));

        // Nested steps: only the outermost end closes the step.
        scene.begin_step();
        scene.begin_step();
        lift(&mut scene, a, 1, 1.0);
        scene.end_step().unwrap();
        lift(&mut scene, b, 1, 1.0);
        scene.end_step().unwrap();
        assert_eq!(scene.end_step(), Err(SceneError::StepNotOpen));
        assert_eq!(scene.history.undo_len(), 1);
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!((z(&scene, a, 1), z(&scene, b, 1)), (0.0, 0.0));
    }

    #[test]
    fn undo_closes_open_step() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        lift(&mut scene, a, 4, 9.0);
        scene.begin_step();
        lift(&mut scene, a, 0, 1.0);
        lift(&mut scene, a, 0, 2.0);
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!((z(&scene, a, 0), z(&scene, a, 4)), (0.0, 9.0));
        assert_eq!(scene.end_step(), Err(SceneError::StepNotOpen));
        assert_eq!(scene.redo(), Ok(true));
        assert_eq!(z(&scene, a, 0), 2.0);
        // redo and clear_history close an open step too.
        scene.begin_step();
        assert_eq!(scene.redo(), Ok(false));
        assert_eq!(scene.end_step(), Err(SceneError::StepNotOpen));
        scene.begin_step();
        lift(&mut scene, a, 1, 1.0);
        scene.clear_history().unwrap();
        assert_eq!(scene.end_step(), Err(SceneError::StepNotOpen));
        assert!(!scene.can_undo() && !scene.can_redo());
        assert_eq!(z(&scene, a, 1), 1.0);
    }

    #[test]
    fn cancel_step_reverts_open_entries() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        lift(&mut scene, a, 4, 9.0);
        assert_eq!(scene.cancel_step(), Err(SceneError::StepNotOpen));
        scene.take_changes();
        scene.begin_step();
        scene.begin_step();
        lift(&mut scene, a, 0, 1.0);
        scene.set_placement(a, moved(2.0)).unwrap();
        let v = scene
            .edit(a, |tx| tx.add_vertices(&[[5.0, 5.0, 5.0]]))
            .unwrap();
        scene.cancel_step().unwrap();
        // Cancel discards the whole step, however deeply nested.
        assert_eq!(scene.end_step(), Err(SceneError::StepNotOpen));
        assert_eq!(scene.cancel_step(), Err(SceneError::StepNotOpen));
        assert_eq!((z(&scene, a, 0), z(&scene, a, 4)), (0.0, 9.0));
        assert_eq!(scene.placement(a), Ok(Placement::IDENTITY));
        assert!(!scene.mesh(a).unwrap().is_vertex_alive(v[0]));
        assert_eq!(scene.history.undo_len(), 1);
        assert!(!scene.can_redo());
        assert_valid(&scene, a);
        let ch = scene.take_changes();
        assert_eq!(ch.len(), 1);
        assert!(ch[0].placement_changed);
        // The cancelled vertex's slot was released, so it is reused.
        let again = scene
            .edit(a, |tx| tx.add_vertices(&[[6.0, 6.0, 6.0]]))
            .unwrap();
        assert_eq!(again, v);
    }

    #[test]
    fn new_step_clears_redo() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let b = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let v = scene
            .edit(a, |tx| tx.add_vertices(&[[5.0, 5.0, 5.0]]))
            .unwrap();
        lift(&mut scene, a, 1, 1.0);
        scene.undo().unwrap();
        scene.undo().unwrap();
        assert!(scene.can_redo());
        // An empty step keeps the redo stack.
        scene.begin_step();
        scene.end_step().unwrap();
        assert!(scene.can_redo());
        // A recorded entry, even on another mesh, drops it.
        lift(&mut scene, b, 0, 1.0);
        assert!(!scene.can_redo());
        assert_eq!(scene.redo(), Ok(false));
        assert_eq!(z(&scene, a, 1), 0.0);
        // The dropped redo released the slot its add had created.
        let again = scene
            .edit(a, |tx| tx.add_vertices(&[[6.0, 6.0, 6.0]]))
            .unwrap();
        assert_eq!(again, v);
        // A placement change drops the redo stack too.
        scene.undo().unwrap();
        assert!(scene.can_redo());
        scene.set_placement(b, moved(1.0)).unwrap();
        assert!(!scene.can_redo());
    }

    #[test]
    fn step_undo_is_atomic() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let b = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        scene.begin_step();
        lift(&mut scene, a, 0, 1.0);
        scene.set_placement(b, moved(1.0)).unwrap();
        lift(&mut scene, b, 1, 2.0);
        scene.end_step().unwrap();
        scene.take_changes();
        let revs = |s: &Scene| (s.mesh(a).unwrap().revision(), s.mesh(b).unwrap().revision());
        let before = revs(&scene);
        let oom = Err(SceneError::Store(StoreError::OutOfMemory));

        // Undo reverts b first, then a: the second mesh's revert fails, so
        // b's revert is re-applied.
        fault::fail_patch_on(2);
        assert_eq!(scene.undo(), oom);
        assert_eq!(revs(&scene), before);
        assert_eq!((z(&scene, a, 0), z(&scene, b, 1)), (1.0, 2.0));
        assert_eq!(scene.placement(b), Ok(moved(1.0)));
        assert!(scene.can_undo() && !scene.can_redo());
        assert_eq!(scene.history.undo_len(), 1);
        assert!(scene.take_changes().is_empty());
        assert_eq!(
            (scene.is_current(a), scene.is_current(b)),
            (Ok(true), Ok(true))
        );
        assert_valid(&scene, a);
        assert_valid(&scene, b);

        // A failure on the very first entry changes nothing either.
        fault::fail_patch_on(1);
        assert_eq!(scene.undo(), oom);
        assert_eq!(revs(&scene), before);

        assert_eq!(scene.undo(), Ok(true));
        assert_eq!((z(&scene, a, 0), z(&scene, b, 1)), (0.0, 0.0));
        let undone = revs(&scene);
        scene.take_changes();

        // Redo applies a, then b: b's apply fails, so a is reverted again.
        fault::fail_patch_on(2);
        assert_eq!(scene.redo(), oom);
        assert_eq!(revs(&scene), undone);
        assert_eq!((z(&scene, a, 0), z(&scene, b, 1)), (0.0, 0.0));
        assert_eq!(scene.placement(b), Ok(Placement::IDENTITY));
        assert!(!scene.can_undo() && scene.can_redo());
        assert!(scene.take_changes().is_empty());
        assert_eq!(scene.redo(), Ok(true));
        assert_eq!((z(&scene, a, 0), z(&scene, b, 1)), (1.0, 2.0));

        // Cancelling an open step is all-or-nothing as well.
        scene.begin_step();
        lift(&mut scene, a, 2, 1.0);
        lift(&mut scene, b, 2, 1.0);
        fault::fail_patch_on(2);
        assert_eq!(scene.cancel_step(), oom.map(|_| ()));
        assert_eq!((z(&scene, a, 2), z(&scene, b, 2)), (1.0, 1.0));
        scene.cancel_step().unwrap();
        assert_eq!((z(&scene, a, 2), z(&scene, b, 2)), (0.0, 0.0));
    }

    #[test]
    fn failed_recovery_reports_corrupted() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(grid(2), with_mode(UpdateMode::Lazy))
            .unwrap();
        let b = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let c = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        scene.begin_step();
        lift(&mut scene, a, 0, 1.0);
        scene.set_placement(c, moved(1.0)).unwrap();
        lift(&mut scene, b, 1, 2.0);
        scene.end_step().unwrap();
        scene.flush().unwrap();
        scene.take_changes();
        let rev_a = scene.mesh(a).unwrap().revision();
        // Undo order: b, c, a. Call 2 (a's revert) fails, then call 3 (b's
        // recovery apply) too, so b stays reverted and a unchanged.
        fault::fail_patch_on(2);
        fault::fail_patch_on(3);
        assert_eq!(scene.undo(), Err(SceneError::Corrupted));
        assert_eq!(scene.mesh(a).unwrap().revision(), rev_a);
        assert_eq!((z(&scene, a, 0), z(&scene, b, 1)), (1.0, 0.0));

        // Every mesh with an edit in the step is reported fully changed; the
        // placement-only mesh is reported as moved, since its placement may
        // be left at either value (here recovery put it back).
        let ch = scene.take_changes();
        assert_eq!(ch.iter().map(|c| c.id).collect::<Vec<_>>(), [a, b, c]);
        assert!(ch[..2].iter().all(|c| c.changes.is_none()));
        assert!(ch[2].placement_changed);
        assert_eq!(ch[2].changes.as_ref().map(ChangeSet::is_empty), Some(true));
        assert_eq!(scene.placement(c), Ok(moved(1.0)));
        // b (Eager) was rebuilt right away. a (Lazy) waits for a rebuild,
        // even though its revision still matches its BVH.
        assert_eq!(scene.is_current(b), Ok(true));
        assert_valid(&scene, b);
        assert_eq!(scene.is_current(a), Ok(false));
        scene.flush().unwrap();
        assert_eq!(scene.is_current(a), Ok(true));
        assert_valid(&scene, a);
        assert!(!scene.entries.get(a).unwrap().rebuild);

        // The caller recovers with clear_history; editing goes on normally.
        scene.clear_history().unwrap();
        assert!(!scene.can_undo() && !scene.can_redo());
        lift(&mut scene, b, 2, 1.0);
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(z(&scene, b, 2), 0.0);
        assert_valid(&scene, b);
    }

    /// A failed recovery rebuilds each touched Eager BVH once, however many
    /// entries its mesh has in the step, and reports a placement the
    /// recovery never got back to.
    #[test]
    fn corrupted_step_rebuilds_once_and_reports_placements() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let b = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let c = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        scene.begin_step();
        for k in 0..3 {
            lift(&mut scene, a, k, 1.0);
            lift(&mut scene, b, k, 1.0);
        }
        scene.set_placement(c, moved(1.0)).unwrap();
        scene.end_step().unwrap();
        scene.take_changes();
        fault::take_rebuilds();
        // Undo order: c's placement, then b (call 1), then a (call 2, fails).
        // Recovery re-applies b (call 3, fails), so c stays reverted.
        fault::fail_patch_on(2);
        fault::fail_patch_on(3);
        assert_eq!(scene.undo(), Err(SceneError::Corrupted));
        assert_eq!(scene.placement(c), Ok(Placement::IDENTITY));
        // Six edit entries on two Eager meshes: two rebuilds, not six.
        assert_eq!(fault::take_rebuilds(), 2);
        for id in [a, b] {
            assert_eq!(scene.is_current(id), Ok(true));
            assert_valid(&scene, id);
        }
        let ch = scene.take_changes();
        assert_eq!(ch.iter().map(|c| c.id).collect::<Vec<_>>(), [a, b, c]);
        assert!(ch[..2].iter().all(|c| c.changes.is_none()));
        assert!(ch[2].placement_changed);
        assert_eq!(ch[2].changes.as_ref().map(ChangeSet::is_empty), Some(true));
    }

    /// Ctrl+Z on an open step larger than the whole budget undoes it; only
    /// then is it dropped from the redo stack to meet the budget.
    #[test]
    fn undo_of_over_budget_open_step_still_undoes() {
        let mut scene = Scene::new(SceneOptions {
            history_budget: 1000,
        });
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let b = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        // The reported repro: 20 placements of one mesh in an open step.
        scene.begin_step();
        for k in 1..=20 {
            scene.set_placement(a, moved(f64::from(k))).unwrap();
        }
        assert!(scene.can_undo());
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(scene.placement(a), Ok(Placement::IDENTITY));
        assert_eq!(scene.end_step(), Err(SceneError::StepNotOpen));

        // Entries on alternating meshes: 20 x 192 bytes, far over budget.
        lift(&mut scene, a, 4, 3.0);
        scene.begin_step();
        for k in 1..=10 {
            scene.set_placement(a, moved(f64::from(k))).unwrap();
            scene.set_placement(b, moved(f64::from(-k))).unwrap();
        }
        assert!(scene.history.bytes() > 1000);
        assert!(scene.can_undo());
        scene.take_changes();
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(scene.placement(a), Ok(Placement::IDENTITY));
        assert_eq!(scene.placement(b), Ok(Placement::IDENTITY));
        assert_eq!(scene.end_step(), Err(SceneError::StepNotOpen));
        let ch = scene.take_changes();
        assert_eq!(ch.len(), 2);
        assert!(ch.iter().all(|c| c.placement_changed));
        // Recording evicted the older edit step; meeting the budget after the
        // undo dropped the undone step from the redo stack.
        assert!(!scene.can_undo() && !scene.can_redo());
        assert_eq!(scene.history.bytes(), 0);
        assert_eq!(z(&scene, a, 4), 3.0);

        // An open step within budget stays redoable after the undo.
        scene.begin_step();
        scene.set_placement(b, moved(2.0)).unwrap();
        assert_eq!(scene.undo(), Ok(true));
        assert!(scene.can_redo());
        assert_eq!(scene.redo(), Ok(true));
        assert_eq!(scene.placement(b), Ok(moved(2.0)));
    }

    /// Setting the current placement again is neither recorded nor
    /// reported.
    #[test]
    fn unchanged_placement_is_not_reported() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        scene.take_changes();
        scene.set_placement(a, Placement::IDENTITY).unwrap();
        assert!(scene.take_changes().is_empty());
        assert!(!scene.can_undo());
        scene.set_placement(a, moved(1.0)).unwrap();
        scene.take_changes();
        scene.set_placement(a, moved(1.0)).unwrap();
        assert!(scene.take_changes().is_empty());
        assert_eq!(scene.history.undo_len(), 1);
    }

    /// A drag inside an open step records one placement entry per run.
    #[test]
    fn placement_runs_coalesce_in_open_step() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let b = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let bytes = crate::history::PLACEMENT_BYTES;

        // 300 frames of a drag: one entry, one entry's bytes.
        scene.begin_step();
        for k in 1..=300 {
            scene.set_placement(a, moved(f64::from(k))).unwrap();
        }
        assert_eq!(scene.history.bytes(), bytes);
        scene.end_step().unwrap();
        assert_eq!(step_bytes(&scene), [bytes]);
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(scene.placement(a), Ok(Placement::IDENTITY));
        assert_eq!(scene.redo(), Ok(true));
        assert_eq!(scene.placement(a), Ok(moved(300.0)));

        // An edit or another mesh's placement between them breaks the run.
        scene.begin_step();
        scene.set_placement(a, moved(1.0)).unwrap();
        scene.set_placement(a, moved(2.0)).unwrap();
        lift(&mut scene, a, 0, 1.0);
        scene.set_placement(a, moved(3.0)).unwrap();
        scene.set_placement(b, moved(4.0)).unwrap();
        scene.set_placement(a, moved(5.0)).unwrap();
        scene.set_placement(a, moved(6.0)).unwrap();
        scene.end_step().unwrap();
        let last = *step_bytes(&scene).last().unwrap();
        let edit = last - 4 * bytes;
        assert!(edit > 0 && edit != bytes, "{last}");
        // Undo goes back through every run to the placement before the step.
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(scene.placement(a), Ok(moved(300.0)));
        assert_eq!(scene.placement(b), Ok(Placement::IDENTITY));
        assert_eq!(z(&scene, a, 0), 0.0);
        assert_eq!(scene.redo(), Ok(true));
        assert_eq!(scene.placement(a), Ok(moved(6.0)));
        assert_eq!(scene.placement(b), Ok(moved(4.0)));

        // Without an open step every placement is a step of its own.
        let before = scene.history.undo_len();
        scene.set_placement(a, moved(7.0)).unwrap();
        scene.set_placement(a, moved(8.0)).unwrap();
        assert_eq!(scene.history.undo_len(), before + 2);

        // Cancelling a coalesced drag puts the placement back.
        scene.begin_step();
        for k in 0..10 {
            scene.set_placement(b, moved(f64::from(k) + 0.5)).unwrap();
        }
        scene.cancel_step().unwrap();
        assert_eq!(scene.placement(b), Ok(moved(4.0)));

        // A drag back to where it started still leaves one entry, whose
        // undo is reported as a placement change.
        scene.begin_step();
        scene.set_placement(b, moved(9.0)).unwrap();
        scene.set_placement(b, moved(4.0)).unwrap();
        scene.end_step().unwrap();
        scene.take_changes();
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(scene.placement(b), Ok(moved(4.0)));
        let ch = scene.take_changes();
        assert_eq!(ch.len(), 1);
        assert!(ch[0].placement_changed);
    }

    /// A failed undo or redo leaves the open step open and unchanged.
    #[test]
    fn failed_undo_or_redo_keeps_open_step() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let oom = Err(SceneError::Store(StoreError::OutOfMemory));

        // A non-empty open step: its own undo fails.
        lift(&mut scene, a, 0, 1.0);
        scene.begin_step();
        lift(&mut scene, a, 1, 1.0);
        fault::fail_patch_on(1);
        assert_eq!(scene.undo(), oom);
        assert_eq!((z(&scene, a, 0), z(&scene, a, 1)), (1.0, 1.0));
        lift(&mut scene, a, 2, 1.0);
        assert_eq!(scene.end_step(), Ok(()));
        assert_eq!(scene.history.undo_len(), 2);
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!((z(&scene, a, 1), z(&scene, a, 2)), (0.0, 0.0));

        // An empty open step: the undo of the latest closed step fails.
        scene.begin_step();
        fault::fail_patch_on(1);
        assert_eq!(scene.undo(), oom);
        assert_eq!(scene.end_step(), Ok(()));
        assert_eq!(z(&scene, a, 0), 1.0);

        // An empty open step: the redo fails.
        scene.begin_step();
        fault::fail_patch_on(1);
        assert_eq!(scene.redo(), oom);
        assert_eq!(scene.end_step(), Ok(()));
        assert_eq!((z(&scene, a, 1), z(&scene, a, 2)), (0.0, 0.0));
        assert_eq!(scene.redo(), Ok(true));
        assert_eq!((z(&scene, a, 1), z(&scene, a, 2)), (1.0, 1.0));
    }

    /// Bytes of each undo step, oldest first.
    fn step_bytes(scene: &Scene) -> Vec<usize> {
        scene.history.undo_step_bytes().collect()
    }

    #[test]
    fn budget_evicts_oldest_steps_and_releases_slots() {
        let remove_three = |scene: &mut Scene, id: MeshId| {
            for t in 0..3 {
                scene.edit(id, |tx| tx.remove_triangles(&[t])).unwrap();
            }
        };
        let mut probe = Scene::new(SceneOptions::default());
        let p = probe.add_mesh(grid(3), MeshOptions::default()).unwrap();
        remove_three(&mut probe, p);
        let sizes = step_bytes(&probe);
        assert_eq!(sizes.len(), 3);
        assert!(sizes.iter().all(|&s| s > 0));

        // Room for the newest two steps only.
        let mut scene = Scene::new(SceneOptions {
            history_budget: sizes[1] + sizes[2],
        });
        let id = scene.add_mesh(grid(3), MeshOptions::default()).unwrap();
        remove_three(&mut scene, id);
        assert_eq!(step_bytes(&scene), sizes[1..]);
        assert_eq!(scene.history.bytes(), sizes[1] + sizes[2]);
        // The evicted step released the slot it had freed; the others not.
        assert_eq!(scene.mesh(id).unwrap().free_triangle_slots(), [0]);
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(scene.undo(), Ok(false));
        let m = scene.mesh(id).unwrap();
        assert!(!m.is_triangle_alive(0) && m.is_triangle_alive(1) && m.is_triangle_alive(2));
        // The released slot is reused by the next add.
        let t = scene.edit(id, |tx| tx.add_triangles(&[[0, 1, 4]])).unwrap();
        assert_eq!(t, [0]);

        // With no budget, every step is evicted as soon as it closes.
        let mut zero = Scene::new(SceneOptions { history_budget: 0 });
        let id = zero.add_mesh(grid(3), MeshOptions::default()).unwrap();
        zero.begin_step();
        zero.edit(id, |tx| tx.remove_triangles(&[4])).unwrap();
        assert!(zero.can_undo());
        zero.end_step().unwrap();
        assert!(!zero.can_undo());
        assert_eq!(zero.history.bytes(), 0);
        assert_eq!(zero.mesh(id).unwrap().free_triangle_slots(), [4]);

        // clear_history releases what the stacks reserved.
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(grid(3), MeshOptions::default()).unwrap();
        scene.edit(id, |tx| tx.remove_triangles(&[2])).unwrap();
        let v = scene
            .edit(id, |tx| tx.add_vertices(&[[9.0, 9.0, 9.0]]))
            .unwrap();
        scene.undo().unwrap();
        scene.clear_history().unwrap();
        assert!(!scene.can_undo() && !scene.can_redo());
        assert_eq!(scene.history.bytes(), 0);
        let m = scene.mesh(id).unwrap();
        assert_eq!(m.free_triangle_slots(), [2]);
        assert_eq!(m.free_vertex_slots(), v);

        // A placement entry counts a constant.
        scene.set_placement(id, moved(1.0)).unwrap();
        assert_eq!(scene.history.bytes(), crate::history::PLACEMENT_BYTES);
    }

    #[test]
    fn history_survives_remove_and_compact() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let b = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let c = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        scene.begin_step();
        lift(&mut scene, a, 0, 1.0);
        scene.edit(b, |tx| tx.remove_triangles(&[0])).unwrap();
        lift(&mut scene, c, 0, 1.0);
        scene.end_step().unwrap();
        lift(&mut scene, b, 1, 1.0);
        let vb = scene
            .edit(b, |tx| tx.add_vertices(&[[7.0, 7.0, 7.0]]))
            .unwrap();
        scene.undo().unwrap();
        assert_eq!(scene.history.undo_len(), 2);

        // b leaves: its entries go, the step left empty with them, and the
        // slots they reserved (undo and redo side) are released.
        let mb = scene.remove_mesh(b).unwrap();
        assert_eq!(mb.free_triangle_slots(), [0]);
        assert_eq!(mb.free_vertex_slots(), vb);
        assert_eq!(scene.history.undo_len(), 1);
        assert!(!scene.can_redo());

        // c compacts: its edit entries go (nothing to release), and the
        // step left empty with them; its placement entry stays.
        lift(&mut scene, c, 1, 1.0);
        scene.set_placement(c, moved(1.0)).unwrap();
        assert_eq!(scene.history.undo_len(), 3);
        scene.compact(c).unwrap();
        assert_eq!(scene.history.undo_len(), 2);
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(scene.placement(c), Ok(Placement::IDENTITY));
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(z(&scene, a, 0), 0.0);
        assert_eq!((z(&scene, c, 0), z(&scene, c, 1)), (1.0, 1.0));
        assert_eq!(scene.undo(), Ok(false));
        assert_eq!(scene.redo(), Ok(true));
        assert_eq!(z(&scene, a, 0), 1.0);
        assert_eq!(scene.redo(), Ok(true));
        assert_eq!(scene.placement(c), Ok(moved(1.0)));

        // Removing a mesh while a step is open drops its open entries.
        scene.begin_step();
        lift(&mut scene, a, 1, 1.0);
        lift(&mut scene, c, 2, 1.0);
        scene.remove_mesh(c).unwrap();
        scene.end_step().unwrap();
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(z(&scene, a, 1), 0.0);
        assert_eq!(scene.history.undo_len(), 1);
        assert_valid(&scene, a);
    }

    /// Undo can bring the mesh back to the revision a stale BVH was built
    /// for: the BVH is current while `pending` is still `Some`. A later edit
    /// and update must still leave a valid BVH that answers like a fresh one.
    #[test]
    fn manual_bvh_current_by_revision_with_pending_changes() {
        let mut scene = Scene::new(SceneOptions::default());
        let m = scene
            .add_mesh(grid(4), with_mode(UpdateMode::Manual))
            .unwrap();
        lift(&mut scene, m, 6, 1.0);
        assert_eq!(scene.is_current(m), Ok(false));
        scene.undo().unwrap();
        assert_eq!(scene.is_current(m), Ok(true));
        assert!(scene.entries.get(m).unwrap().pending.is_some());

        lift(&mut scene, m, 7, 2.0);
        assert_eq!(scene.is_current(m), Ok(false));
        scene.update(m).unwrap();
        assert_eq!(scene.is_current(m), Ok(true));
        assert_valid(&scene, m);

        let points = [[1.0, 1.0, 3.0], [2.5, 1.5, -1.0], [0.0, 4.0, 0.5]];
        let e = scene.entries.get(m).unwrap();
        let fresh = Bvh::build(&e.mesh, &BuildOptions::default()).unwrap();
        let wants = points.map(|p| {
            fast_mesh_query::closest_point(
                &e.mesh,
                &fresh,
                p,
                &fast_mesh_query::ClosestOptions::default(),
            )
            .unwrap()
            .map(|h| (h.triangle, h.distance))
        });
        for (p, want) in points.into_iter().zip(wants) {
            let got = scene
                .closest_point_on(m, p.map(f64::from), &crate::SceneClosestOptions::default())
                .unwrap()
                .map(|h| (h.hit.triangle, h.hit.distance));
            assert_eq!(got, want);
        }
    }

    #[test]
    fn undo_updates_bvhs_per_mode() {
        let mut scene = Scene::new(SceneOptions::default());
        let e = scene.add_mesh(grid(4), MeshOptions::default()).unwrap();
        let l = scene
            .add_mesh(grid(4), with_mode(UpdateMode::Lazy))
            .unwrap();
        let m = scene
            .add_mesh(grid(4), with_mode(UpdateMode::Manual))
            .unwrap();
        scene.begin_step();
        for id in [e, l, m] {
            lift(&mut scene, id, 6, 1.0);
        }
        scene.end_step().unwrap();
        scene.flush().unwrap();
        scene.update(m).unwrap();
        scene.take_changes();

        let check = |scene: &mut Scene| {
            assert_eq!(scene.is_current(e), Ok(true));
            assert_eq!(scene.is_current(l), Ok(false));
            assert_eq!(scene.is_current(m), Ok(false));
            let ch = scene.take_changes();
            assert_eq!(ch.len(), 3);
            assert!(
                ch.iter()
                    .all(|c| c.changes.as_ref().unwrap().vertices_moved == [6])
            );
            scene.flush().unwrap();
            assert_eq!(scene.is_current(l), Ok(true));
            assert_eq!(scene.is_current(m), Ok(false));
            scene.update(m).unwrap();
            for id in [e, l, m] {
                assert_valid(scene, id);
            }
        };
        scene.undo().unwrap();
        check(&mut scene);
        scene.redo().unwrap();
        check(&mut scene);

        // A failing Eager update does not fail the undo (§15.1).
        fault::fail_bvh_update_on(1);
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(scene.is_current(e), Ok(false));
        assert!(scene.entries.get(e).unwrap().pending.is_some());
        scene.flush().unwrap();
        assert_eq!(scene.is_current(e), Ok(true));
        assert_valid(&scene, e);
    }

    #[test]
    fn two_subscribers_each_get_all_changes() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let b = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        // A new subscriber does not see the adds made before it.
        let s = scene.subscribe();
        assert_eq!(scene.take_changes_for(s), Ok(Vec::new()));
        assert_eq!(scene.take_changes().len(), 2);

        lift(&mut scene, a, 4, 1.0);
        scene.set_placement(b, moved(1.0)).unwrap();
        let mine = scene.take_changes();
        assert_eq!(mine.len(), 2);
        // Draining the default subscriber leaves the other one's reports.
        lift(&mut scene, a, 1, 1.0);
        let theirs = scene.take_changes_for(s).unwrap();
        assert_eq!(theirs.len(), 2);
        assert_eq!(theirs[0].id, a);
        assert_eq!(theirs[0].changes.as_ref().unwrap().vertices_moved, [1, 4]);
        assert_eq!(theirs[0].colors_dirty, Some(Vec::new()));
        assert!(theirs[0].fields_changed.is_empty());
        assert!(theirs[1].placement_changed && theirs[1].id == b);
        let mine = scene.take_changes();
        assert_eq!(mine.len(), 1);
        assert_eq!(mine[0].changes.as_ref().unwrap().vertices_moved, [1]);

        // Removals, compaction and new meshes reach every subscriber.
        let s2 = scene.subscribe();
        assert_ne!(s, s2);
        let c = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        scene.compact(b).unwrap();
        scene.remove_mesh(a).unwrap();
        for got in [
            scene.take_changes(),
            scene.take_changes_for(s).unwrap(),
            scene.take_changes_for(s2).unwrap(),
        ] {
            assert_eq!(got.len(), 3, "{got:?}");
            assert!(got[0].id == a && got[0].removed);
            assert!(got[1].id == b && got[1].changes.is_none() && !got[1].added);
            assert!(got[2].id == c && got[2].added);
        }
        assert!(scene.take_changes_for(s2).unwrap().is_empty());
    }

    /// Whether a removal is reported depends on each subscriber's own report:
    /// one that saw the add gets `removed`, one that did not gets nothing.
    #[test]
    fn add_then_remove_is_per_subscriber() {
        let mut scene = Scene::new(SceneOptions::default());
        let s = scene.subscribe();
        let id = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let added = scene.take_changes_for(s).unwrap();
        assert!(added[0].added && added[0].colors_dirty.is_none());
        lift(&mut scene, id, 0, 1.0);
        scene.remove_mesh(id).unwrap();
        let got = scene.take_changes_for(s).unwrap();
        assert_eq!(got.len(), 1);
        assert!(got[0].id == id && got[0].removed && !got[0].added);
        assert_eq!(got[0].changes.as_ref().unwrap().vertices_moved, [0]);
        assert!(scene.take_changes().is_empty());
    }

    #[test]
    fn unsubscribed_id_rejected() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        scene.take_changes();
        let s = scene.subscribe();
        let keep = scene.subscribe();
        scene.unsubscribe(s).unwrap();
        assert!(matches!(
            scene.take_changes_for(s),
            Err(SceneError::InvalidInput(_))
        ));
        assert!(matches!(
            scene.unsubscribe(s),
            Err(SceneError::InvalidInput(_))
        ));
        // The others keep working, and a removal still cannot fail.
        lift(&mut scene, a, 0, 1.0);
        scene.remove_mesh(a).unwrap();
        assert_eq!(scene.take_changes_for(keep).unwrap().len(), 1);
        assert_eq!(scene.take_changes().len(), 1);
        // Ids are never reused.
        assert!(scene.subscribe() != s);
    }

    #[test]
    fn channel_writes_are_reported() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(grid(1), MeshOptions::default()).unwrap();
        let s = scene.subscribe();
        scene.take_changes();
        scene
            .add_channel(id, ChannelDesc::derived("curv", 1))
            .unwrap();
        scene
            .add_channel(id, ChannelDesc::derived("other", 1))
            .unwrap();
        scene.derived_channel_mut(id, "curv").unwrap()[0] = 1.0;
        scene.derived_channel_mut(id, "curv").unwrap()[1] = 1.0;
        let ch = scene.take_changes();
        assert_eq!(ch.len(), 1);
        assert_eq!(ch[0].channels_written, ["curv", "other"]);
        assert_eq!(ch[0].changes.as_ref().map(ChangeSet::is_empty), Some(true));
        assert!(!ch[0].placement_changed && !ch[0].added);
        scene.remove_channel(id, "other").unwrap();
        // Failed calls report nothing.
        assert!(scene.derived_channel_mut(id, "nope").is_err());
        assert!(scene.remove_channel(id, "nope").is_err());
        assert_eq!(scene.take_changes()[0].channels_written, ["other"]);
        assert!(scene.take_changes().is_empty());
        assert_eq!(
            scene.take_changes_for(s).unwrap()[0].channels_written,
            ["curv", "other"]
        );
    }

    #[test]
    fn field_prefix_is_reserved() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(grid(1), MeshOptions::default()).unwrap();
        scene.take_changes();
        assert!(matches!(
            scene.add_channel(id, ChannelDesc::derived("field:d", 1)),
            Err(SceneError::InvalidInput(_))
        ));
        assert!(scene.mesh(id).unwrap().channel("field:d").is_none());
        assert!(matches!(
            scene.remove_channel(id, "field:d"),
            Err(SceneError::InvalidInput(_))
        ));
        assert!(matches!(
            scene.derived_channel_mut(id, "field:d"),
            Err(SceneError::InvalidInput(_))
        ));
        assert!(scene.take_changes().is_empty());
        // Only the exact prefix is reserved.
        scene
            .add_channel(id, ChannelDesc::derived("fields", 1))
            .unwrap();
    }
}
