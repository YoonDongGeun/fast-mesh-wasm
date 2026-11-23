//! The shared undo history: steps of edits, placement changes and undoable
//! adds and removals across meshes, limited by a memory budget.

use std::collections::VecDeque;
use std::mem::size_of;

use fast_mesh_store::{ChangeSet, Patch, PatchSide, StoreError};

use crate::id::SlotMap;
use crate::scene::Entry;
use crate::{MeshId, Placement, Result, SceneError};

/// Bytes a placement entry counts against the budget: its old and new value.
pub(crate) const PLACEMENT_BYTES: usize = 2 * size_of::<Placement>();

/// One recorded change of a step.
///
/// The patch is stored inline, not boxed: boxing it would allocate after the
/// commit, and that allocation cannot be made fallible on stable Rust, so a
/// failure would abort instead of leaving the scene unchanged. The space a
/// placement entry wastes is bounded by the patch header size.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Change {
    /// A committed transaction on a mesh.
    Edit(MeshId, Patch),
    /// A placement change of a mesh.
    Placement {
        id: MeshId,
        old: Placement,
        new: Placement,
    },
    /// An undoable add: undo parks the mesh, redo restores it. `bytes` is
    /// what the mesh held when it was added (see `Entry::byte_size`).
    Add { id: MeshId, bytes: usize },
    /// An undoable removal: the mesh is parked; undo restores it, redo parks
    /// it again. `bytes` is what the parked mesh holds.
    Remove { id: MeshId, bytes: usize },
}

impl Change {
    fn mesh(&self) -> MeshId {
        match self {
            Change::Edit(id, _)
            | Change::Placement { id, .. }
            | Change::Add { id, .. }
            | Change::Remove { id, .. } => *id,
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Change::Edit(_, p) => p.byte_size(),
            Change::Placement { .. } => PLACEMENT_BYTES,
            Change::Add { bytes, .. } | Change::Remove { bytes, .. } => *bytes,
        }
    }

    /// An add or a removal, which a recovery cannot leave half done.
    fn is_presence(&self) -> bool {
        matches!(self, Change::Add { .. } | Change::Remove { .. })
    }
}

/// Releases what `change` holds once the history drops it from `side`: the
/// slots its patch reserves (on its mesh, live or parked), or the parked mesh
/// it was the last way back to. Call once, when the change is discarded.
///
/// An add dropped from the redo stack, or a removal dropped from the undo
/// stack, frees its parked mesh for good (its id never matches again). An add
/// dropped from the undo stack, or a removal from the redo stack, leaves its
/// live mesh alone: it just stays.
///
/// Every change in the history belongs to a live or parked mesh: `remove_mesh`
/// drops a live mesh's changes before the mesh goes, and a parked mesh is
/// freed only here. A change whose mesh is already gone (its parked mesh was
/// freed by another change of the same batch) releases nothing.
fn release_in(change: &Change, entries: &mut SlotMap<Entry>, side: PatchSide) {
    match change {
        Change::Edit(id, p) => {
            if let Some(e) = entries.get_any_mut(*id) {
                p.release(&mut e.mesh, side);
            }
        }
        Change::Placement { .. } => {}
        Change::Add { id, .. } => {
            if side == PatchSide::Redoable {
                drop(entries.release_parked(*id));
            }
        }
        Change::Remove { id, .. } => {
            if side == PatchSide::Undoable {
                drop(entries.release_parked(*id));
            }
        }
    }
}

/// What the history needs from the scene to undo or redo a step: its meshes,
/// and parking and restoring one (undoable adds and removals).
pub(crate) trait StepHost {
    /// The meshes, live and parked.
    fn entries(&mut self) -> &mut SlotMap<Entry>;
    /// Parks live mesh `id` (the undo of an add, the redo of a removal) and
    /// returns the live meshes whose fields lost it as a target. All or
    /// nothing: on error nothing changed.
    fn park(&mut self, id: MeshId) -> Result<Vec<MeshId>>;
    /// Restores parked mesh `id` under the same id (the redo of an add, the
    /// undo of a removal). All or nothing.
    fn unpark(&mut self, id: MeshId) -> Result<()>;
}

/// The changes of one step, in recording order.
#[derive(Default)]
pub(crate) struct Step {
    changes: Vec<Change>,
    bytes: usize,
}

/// What one change did when a step was undone or redone.
pub(crate) enum Done {
    /// An edit was applied or reverted: the mesh and its changes.
    Edit(MeshId, ChangeSet),
    /// A placement entry was applied or reverted.
    Placement(MeshId),
    /// A mesh was parked: the live meshes whose fields lost it as a target.
    Parked(Vec<MeshId>),
    /// A parked mesh was restored.
    Restored(MeshId),
}

/// Undo and redo stacks of steps, plus the step being recorded.
///
/// `bytes` counts every step held (undo, redo and the open one). Eviction
/// drops the oldest undo steps; it runs whenever a change is recorded, at
/// which point the redo stack is already empty. The open step is never
/// evicted while open; once closed, it is evicted like any other step, so a
/// step larger than the whole budget is dropped as soon as it closes. The
/// one exception is `undo` on a non-empty open step: the step is undone
/// first and then meets the budget on the redo stack (see `undo`).
pub(crate) struct History {
    undo: VecDeque<Step>,
    redo: Vec<Step>,
    open: Step,
    /// Nesting depth of `begin_step`; 0 when no step is open.
    depth: u32,
    budget: usize,
    bytes: usize,
}

impl History {
    pub(crate) fn new(budget: usize) -> History {
        History {
            undo: VecDeque::new(),
            redo: Vec::new(),
            open: Step::default(),
            depth: 0,
            budget,
            bytes: 0,
        }
    }

    pub(crate) fn begin(&mut self) {
        self.depth = self.depth.saturating_add(1);
    }

    /// Ends one nesting level; the outermost closes the step.
    pub(crate) fn end(&mut self, entries: &mut SlotMap<Entry>) -> Result<()> {
        if self.depth == 0 {
            return Err(SceneError::StepNotOpen);
        }
        self.depth -= 1;
        if self.depth == 0 {
            self.close(entries);
        }
        Ok(())
    }

    /// Closes the open step, whatever its depth. An empty step is dropped.
    pub(crate) fn close(&mut self, entries: &mut SlotMap<Entry>) {
        self.depth = 0;
        if self.open.changes.is_empty() {
            return;
        }
        // `reserve` made room for this push when the first change was recorded.
        self.undo.push_back(std::mem::take(&mut self.open));
        self.evict(entries);
    }

    /// Makes room for one more change, so that `record` (and the `close`
    /// after it) cannot fail. Call before the operation that produces the
    /// change; on error nothing has changed.
    pub(crate) fn reserve(&mut self) -> Result<()> {
        #[cfg(test)]
        crate::scene::fault::before_reserve()?;
        let oom = |_| SceneError::Store(StoreError::OutOfMemory);
        self.open.changes.try_reserve(1).map_err(oom)?;
        self.undo.try_reserve(1).map_err(oom)
    }

    /// Records a change in the open step, or as a step of its own when no
    /// step is open. Drops the redo stack. `reserve` must have succeeded
    /// since the last `record`.
    pub(crate) fn record(&mut self, entries: &mut SlotMap<Entry>, change: Change) {
        self.drop_redo(entries);
        let bytes = change.bytes();
        self.open.changes.push(change);
        self.open.bytes += bytes;
        self.bytes += bytes;
        if self.depth == 0 {
            self.close(entries);
        } else {
            self.evict(entries);
        }
    }

    /// Records a placement change like `record`, except that it merges into
    /// the open step's last entry when that is a placement entry of the same
    /// mesh: the merged entry keeps its `old` and takes the new value, so a
    /// drag records one entry however many frames it lasts. Any other entry
    /// in between (an edit, or another mesh's placement) breaks the run. A
    /// merge adds no bytes. `reserve` must have succeeded since the last
    /// `record`.
    pub(crate) fn record_placement(
        &mut self,
        entries: &mut SlotMap<Entry>,
        id: MeshId,
        old: Placement,
        new: Placement,
    ) {
        if self.depth > 0
            && let Some(Change::Placement {
                id: last,
                new: last_new,
                ..
            }) = self.open.changes.last_mut()
            && *last == id
        {
            // The run's previous entry recorded `old` as its new value.
            debug_assert!(*last_new == old, "placement run out of step");
            *last_new = new;
            // Recording the previous entry dropped the redo stack already.
            debug_assert!(self.redo.is_empty(), "redo steps beside an open step");
            return;
        }
        self.record(entries, Change::Placement { id, old, new });
    }

    fn drop_redo(&mut self, entries: &mut SlotMap<Entry>) {
        for step in self.redo.drain(..) {
            self.bytes -= step.bytes;
            for c in &step.changes {
                release_in(c, entries, PatchSide::Redoable);
            }
        }
    }

    fn evict(&mut self, entries: &mut SlotMap<Entry>) {
        while self.bytes > self.budget {
            let Some(step) = self.undo.pop_front() else {
                break;
            };
            self.bytes -= step.bytes;
            for c in &step.changes {
                release_in(c, entries, PatchSide::Undoable);
            }
        }
    }

    /// Closes the open step, then drops every step, releasing their slots.
    pub(crate) fn clear(&mut self, entries: &mut SlotMap<Entry>) {
        self.close(entries);
        for step in self.undo.drain(..) {
            for c in &step.changes {
                release_in(c, entries, PatchSide::Undoable);
            }
        }
        self.drop_redo(entries);
        self.bytes = 0;
    }

    /// Drops the oldest redo steps until the budget is met. Only needed after
    /// an open step was undone straight onto the redo stack, since an open
    /// step is not evicted while open and may exceed the budget.
    fn evict_redo(&mut self, entries: &mut SlotMap<Entry>) {
        while self.bytes > self.budget && !self.redo.is_empty() {
            let step = self.redo.remove(0);
            self.bytes -= step.bytes;
            for c in &step.changes {
                release_in(c, entries, PatchSide::Redoable);
            }
        }
    }

    /// True if an interactive step is currently open.
    pub(crate) fn is_open(&self) -> bool {
        self.depth > 0
    }

    /// True if `undo` would undo something (including the open step).
    pub(crate) fn can_undo(&self) -> bool {
        !self.undo.is_empty() || !self.open.changes.is_empty()
    }

    /// True if `redo` would redo something.
    pub(crate) fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Closes the open step and undoes the latest step. `None` when there is
    /// nothing to undo. On error the meshes, both stacks and the open step
    /// are unchanged, except that [`SceneError::Corrupted`] drops the failed
    /// step (its patches' known states were released in `run_step`).
    ///
    /// A non-empty open step is undone directly, before any eviction, so
    /// that it is undone even when it exceeds the budget (§15.3: Ctrl+Z
    /// mid-drag undoes the drag). It then goes to the redo stack, and the
    /// budget is met by evicting the oldest undo steps, then, if that is not
    /// enough, by dropping it from the redo stack again.
    pub(crate) fn undo<H: StepHost>(&mut self, host: &mut H) -> Result<Option<Vec<Done>>> {
        let has_open = !self.open.changes.is_empty();
        if !has_open && self.undo.is_empty() {
            self.depth = 0;
            return Ok(None);
        }
        self.redo
            .try_reserve(1)
            .map_err(|_| SceneError::Store(StoreError::OutOfMemory))?;
        if has_open {
            // Recording an entry dropped the redo stack, and nothing can
            // fill it while the step stays open.
            debug_assert!(self.redo.is_empty(), "redo steps beside an open step");
            let done = match run_step(&self.open, host, false) {
                Ok(done) => done,
                Err(e) => {
                    if e == SceneError::Corrupted {
                        self.drop_open();
                    }
                    return Err(e);
                }
            };
            self.depth = 0;
            self.redo.push(std::mem::take(&mut self.open));
            self.evict(host.entries());
            self.evict_redo(host.entries());
            return Ok(Some(done));
        }
        let step = self.undo.pop_back().ok_or(SceneError::Corrupted)?;
        match run_step(&step, host, false) {
            Ok(done) => {
                self.depth = 0;
                self.redo.push(step);
                Ok(Some(done))
            }
            Err(e) => {
                if e == SceneError::Corrupted {
                    self.bytes -= step.bytes;
                } else {
                    self.undo.push_back(step);
                }
                Err(e)
            }
        }
    }

    /// Closes the open step and redoes the latest undone step. `None` when
    /// there is nothing to redo. Errors as for `undo`.
    pub(crate) fn redo<H: StepHost>(&mut self, host: &mut H) -> Result<Option<Vec<Done>>> {
        if self.redo.is_empty() {
            // Closing may evict the closed step, as `end_step` would; there
            // is nothing to redo either way.
            self.close(host.entries());
            return Ok(None);
        }
        // A non-empty redo stack means the open step is empty (recording
        // drops the redo stack), so there is nothing to close but the depth.
        debug_assert!(
            self.open.changes.is_empty(),
            "redo steps beside an open step"
        );
        self.undo
            .try_reserve(1)
            .map_err(|_| SceneError::Store(StoreError::OutOfMemory))?;
        let step = self.redo.pop().ok_or(SceneError::Corrupted)?;
        match run_step(&step, host, true) {
            Ok(done) => {
                self.depth = 0;
                self.undo.push_back(step);
                Ok(Some(done))
            }
            Err(e) => {
                if e == SceneError::Corrupted {
                    self.bytes -= step.bytes;
                } else {
                    self.redo.push(step);
                }
                Err(e)
            }
        }
    }

    /// Reverts the open step's changes and discards the step, releasing the
    /// slots its patches created. On error the step stays open, unchanged,
    /// except that [`SceneError::Corrupted`] discards it.
    pub(crate) fn cancel<H: StepHost>(&mut self, host: &mut H) -> Result<Vec<Done>> {
        if self.depth == 0 {
            return Err(SceneError::StepNotOpen);
        }
        let done = match run_step(&self.open, host, false) {
            Ok(done) => done,
            Err(e) => {
                if e == SceneError::Corrupted {
                    self.drop_open();
                }
                return Err(e);
            }
        };
        let step = std::mem::take(&mut self.open);
        self.depth = 0;
        self.bytes -= step.bytes;
        for c in &step.changes {
            release_in(c, host.entries(), PatchSide::Redoable);
        }
        Ok(done)
    }

    /// Discards the open step after [`SceneError::Corrupted`]: its patches'
    /// known states were released in `run_step`, so only the bytes go.
    fn drop_open(&mut self) {
        let step = std::mem::take(&mut self.open);
        self.depth = 0;
        self.bytes -= step.bytes;
    }

    /// Drops changes of mesh `id`; steps left empty are dropped too (the open
    /// step stays open).
    ///
    /// - With `mesh` (permanent removal of a live mesh): every change of the
    ///   mesh goes, and the slots its patches reserve on `mesh` are released.
    ///   Its add and removal entries hold nothing then: a live mesh's add can
    ///   only be on the undo side, and its removal only on the redo side.
    /// - Without (compaction): only its edit changes go, releasing nothing,
    ///   since their slots no longer exist; placement, add and removal
    ///   changes stay valid and are kept.
    pub(crate) fn forget(&mut self, id: MeshId, mut mesh: Option<&mut fast_mesh_store::Mesh>) {
        let mut drop_from = |step: &mut Step, side: PatchSide| {
            step.changes.retain(|c| {
                if c.mesh() != id {
                    return true;
                }
                match mesh.as_deref_mut() {
                    Some(m) => {
                        if let Change::Edit(_, p) = c {
                            p.release(m, side);
                        }
                        false
                    }
                    // Compaction: only patches go stale with the epoch.
                    None => !matches!(c, Change::Edit(..)),
                }
            });
            step.bytes = step.changes.iter().map(Change::bytes).sum();
        };
        for step in &mut self.undo {
            drop_from(step, PatchSide::Undoable);
        }
        for step in &mut self.redo {
            drop_from(step, PatchSide::Redoable);
        }
        drop_from(&mut self.open, PatchSide::Undoable);
        self.undo.retain(|s| !s.changes.is_empty());
        self.redo.retain(|s| !s.changes.is_empty());
        self.bytes = self.undo.iter().map(|s| s.bytes).sum::<usize>()
            + self.redo.iter().map(|s| s.bytes).sum::<usize>()
            + self.open.bytes;
    }

    /// Number of closed steps that can be undone.
    #[cfg(test)]
    pub(crate) fn undo_len(&self) -> usize {
        self.undo.len()
    }

    /// Bytes of each undo step, oldest first.
    #[cfg(test)]
    pub(crate) fn undo_step_bytes(&self) -> impl Iterator<Item = usize> + '_ {
        self.undo.iter().map(|s| s.bytes)
    }

    /// Bytes held by every step.
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Undoes (`forward == false`, changes in reverse order) or redoes
/// (`forward == true`, in order) one step, all or nothing.
///
/// If a change fails, the changes already run are run the other way, in the
/// opposite order, and the error is returned with every mesh as before. If
/// that recovery fails too, the result is [`SceneError::Corrupted`]: every
/// mesh with an edit in the step is marked fully changed (see
/// `Entry::flag_unknown`), since what it holds is no longer known, and every
/// mesh with a placement entry in the step is reported as
/// `placement_changed`, since its placement may be left at either value.
/// The caller drops the failed step, so the patches whose end state this run
/// determines are released for it here; a patch a failed change touched is
/// left reserved, since its state is unknown.
fn run_step<H: StepHost>(step: &Step, host: &mut H, forward: bool) -> Result<Vec<Done>> {
    let mut done: Vec<Done> = Vec::new();
    done.try_reserve_exact(step.changes.len())
        .map_err(|_| SceneError::Store(StoreError::OutOfMemory))?;
    let order = |i: usize| {
        if forward {
            i
        } else {
            step.changes.len() - 1 - i
        }
    };
    for i in 0..step.changes.len() {
        match run_change(&step.changes[order(i)], host, forward) {
            Ok(d) => done.push(d),
            Err(e) => {
                for j in (0..i).rev() {
                    if run_change(&step.changes[order(j)], host, !forward).is_err() {
                        mark_corrupted(step, host.entries());
                        release_partial(step, host.entries(), forward, i, j);
                        return Err(SceneError::Corrupted);
                    }
                }
                if e == SceneError::Corrupted {
                    // The step names a mesh that is gone (defensive: removal
                    // drops its changes), but recovery restored everything,
                    // so every patch is back at a known state.
                    release_recovered(step, host.entries(), forward);
                }
                return Err(e);
            }
        }
    }
    Ok(done)
}

/// Releases the patches of a step dropped after [`SceneError::Corrupted`]
/// for the state each is known to be in. Change `i` is the one whose run
/// failed and `j` the one whose recovery failed; both are left reserved,
/// unless they are adds or removals, which fail all or nothing: `i` is then
/// still at the far state and `j` at the run's end state. Changes run and
/// recovered, or never run, are at the step's far state (after it for undo,
/// before it for redo); changes run and never recovered are at the run's end
/// state.
fn release_partial(step: &Step, entries: &mut SlotMap<Entry>, forward: bool, i: usize, j: usize) {
    let order = |k: usize| {
        if forward {
            k
        } else {
            step.changes.len() - 1 - k
        }
    };
    // At the far state: undoable when the step ran backwards, redoable
    // when it ran forwards.
    let far = if forward {
        PatchSide::Redoable
    } else {
        PatchSide::Undoable
    };
    // At the run's end state: the other side.
    let end = if forward {
        PatchSide::Undoable
    } else {
        PatchSide::Redoable
    };
    for k in (j + 1..i).chain(i + 1..step.changes.len()) {
        release_in(&step.changes[order(k)], entries, far);
    }
    for k in 0..j {
        release_in(&step.changes[order(k)], entries, end);
    }
    if step.changes[order(i)].is_presence() {
        release_in(&step.changes[order(i)], entries, far);
    }
    if step.changes[order(j)].is_presence() {
        release_in(&step.changes[order(j)], entries, end);
    }
}

/// Releases every patch of a step dropped after [`SceneError::Corrupted`]
/// whose recovery still restored everything: all are at the step's far
/// state.
fn release_recovered(step: &Step, entries: &mut SlotMap<Entry>, forward: bool) {
    let far = if forward {
        PatchSide::Redoable
    } else {
        PatchSide::Undoable
    };
    for c in &step.changes {
        release_in(c, entries, far);
    }
}

/// After a failed recovery: flags every mesh the step touched, then brings
/// the Eager BVHs up to date.
///
/// Flagging first and updating after means each mesh is rebuilt once, however
/// many entries it has in the step: the first `after_change` rebuilds and
/// clears the flag, the later ones find the BVH current. This needs no
/// allocation, which matters on a path reached through out-of-memory errors.
fn mark_corrupted(step: &Step, entries: &mut SlotMap<Entry>) {
    for c in &step.changes {
        let Some(e) = entries.get_mut(c.mesh()) else {
            continue;
        };
        match c {
            Change::Edit(..) => e.flag_unknown(),
            Change::Placement { .. } => e.report_placement(),
            // Parking and restoring are all or nothing, and report
            // themselves.
            Change::Add { .. } | Change::Remove { .. } => {}
        }
    }
    for c in &step.changes {
        if let Change::Edit(id, _) = c
            && let Some(e) = entries.get_mut(*id)
        {
            e.after_change();
        }
    }
}

/// Applies (`forward`) or reverts one change. On error the scene is
/// unchanged.
fn run_change<H: StepHost>(change: &Change, host: &mut H, forward: bool) -> Result<Done> {
    match change {
        Change::Edit(id, patch) => {
            let entry = host.entries().get_mut(*id).ok_or(SceneError::Corrupted)?;
            #[cfg(test)]
            crate::scene::fault::before_patch()?;
            let changes = if forward {
                patch.apply(&mut entry.mesh)?
            } else {
                patch.revert(&mut entry.mesh)?
            };
            Ok(Done::Edit(*id, changes))
        }
        Change::Placement { id, old, new } => {
            let entry = host.entries().get_mut(*id).ok_or(SceneError::Corrupted)?;
            entry.placement = if forward { *new } else { *old };
            Ok(Done::Placement(*id))
        }
        Change::Add { id, .. } | Change::Remove { id, .. } => {
            #[cfg(test)]
            crate::scene::fault::before_patch()?;
            // Redoing an add, or undoing a removal, restores the mesh.
            if matches!(change, Change::Add { .. }) == forward {
                host.unpark(*id)?;
                Ok(Done::Restored(*id))
            } else {
                let touched = host.park(*id)?;
                Ok(Done::Parked(touched))
            }
        }
    }
}
