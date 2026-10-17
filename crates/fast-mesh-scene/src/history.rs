//! The shared undo history: steps of edits and placement changes across
//! meshes, limited by a memory budget.

use std::collections::VecDeque;
use std::mem::size_of;

use fast_mesh_store::{ChangeSet, Mesh, Patch, PatchSide, StoreError};

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
}

impl Change {
    fn mesh(&self) -> MeshId {
        match self {
            Change::Edit(id, _) | Change::Placement { id, .. } => *id,
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Change::Edit(_, p) => p.byte_size(),
            Change::Placement { .. } => PLACEMENT_BYTES,
        }
    }

    /// Releases the slots the change reserves on `mesh` (its own mesh). Call
    /// once, when the change is discarded.
    fn release(&self, mesh: &mut Mesh, side: PatchSide) {
        if let Change::Edit(_, p) = self {
            p.release(mesh, side);
        }
    }
}

/// Releases the slots `change` reserves, looking its mesh up in `entries`.
/// Every change in the history belongs to a live mesh: `remove_mesh` drops a
/// mesh's changes before the mesh goes.
fn release_in(change: &Change, entries: &mut SlotMap<Entry>, side: PatchSide) {
    if let Some(e) = entries.get_mut(change.mesh()) {
        change.release(&mut e.mesh, side);
    }
}

/// The changes of one step, in recording order.
#[derive(Default)]
pub(crate) struct Step {
    changes: Vec<Change>,
    bytes: usize,
}

/// What one change did when a step was undone or redone: the mesh, and its
/// changes (`None` for a placement change).
pub(crate) type Done = (MeshId, Option<ChangeSet>);

/// Undo and redo stacks of steps, plus the step being recorded.
///
/// `bytes` counts every step held (undo, redo and the open one). Eviction
/// drops the oldest undo steps only; it runs whenever a change is recorded,
/// at which point the redo stack is already empty. The open step is never
/// evicted while open; once closed, it is evicted like any other step, so a
/// step larger than the whole budget is dropped as soon as it closes.
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

    /// True if `undo` would undo something (including the open step).
    pub(crate) fn can_undo(&self) -> bool {
        !self.undo.is_empty() || !self.open.changes.is_empty()
    }

    /// True if `redo` would redo something.
    pub(crate) fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Closes the open step and undoes the latest step. `None` when there is
    /// nothing to undo. On error the meshes and both stacks are unchanged
    /// (except for [`SceneError::Corrupted`]).
    pub(crate) fn undo(&mut self, entries: &mut SlotMap<Entry>) -> Result<Option<Vec<Done>>> {
        self.close(entries);
        if self.undo.is_empty() {
            return Ok(None);
        }
        self.redo
            .try_reserve(1)
            .map_err(|_| SceneError::Store(StoreError::OutOfMemory))?;
        let step = self.undo.pop_back().ok_or(SceneError::Corrupted)?;
        match run_step(&step, entries, false) {
            Ok(done) => {
                self.redo.push(step);
                Ok(Some(done))
            }
            Err(e) => {
                self.undo.push_back(step);
                Err(e)
            }
        }
    }

    /// Closes the open step and redoes the latest undone step. `None` when
    /// there is nothing to redo. Errors as for `undo`.
    pub(crate) fn redo(&mut self, entries: &mut SlotMap<Entry>) -> Result<Option<Vec<Done>>> {
        self.close(entries);
        if self.redo.is_empty() {
            return Ok(None);
        }
        self.undo
            .try_reserve(1)
            .map_err(|_| SceneError::Store(StoreError::OutOfMemory))?;
        let step = self.redo.pop().ok_or(SceneError::Corrupted)?;
        match run_step(&step, entries, true) {
            Ok(done) => {
                self.undo.push_back(step);
                Ok(Some(done))
            }
            Err(e) => {
                self.redo.push(step);
                Err(e)
            }
        }
    }

    /// Reverts the open step's changes and discards the step, releasing the
    /// slots its patches created. On error the step stays open, unchanged.
    pub(crate) fn cancel(&mut self, entries: &mut SlotMap<Entry>) -> Result<Vec<Done>> {
        if self.depth == 0 {
            return Err(SceneError::StepNotOpen);
        }
        let done = run_step(&self.open, entries, false)?;
        let step = std::mem::take(&mut self.open);
        self.depth = 0;
        self.bytes -= step.bytes;
        for c in &step.changes {
            release_in(c, entries, PatchSide::Redoable);
        }
        Ok(done)
    }

    /// Drops every change of mesh `id`; steps left empty are dropped too
    /// (the open step stays open). With `mesh`, the slots the dropped patches
    /// reserve on it are released; without, nothing is (after a compaction
    /// those slots no longer exist).
    pub(crate) fn forget(&mut self, id: MeshId, mut mesh: Option<&mut Mesh>) {
        let mut drop_from = |step: &mut Step, side: PatchSide| {
            step.changes.retain(|c| {
                if c.mesh() != id {
                    return true;
                }
                if let Some(m) = mesh.as_deref_mut() {
                    c.release(m, side);
                }
                false
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
/// that recovery fails too, the result is [`SceneError::Corrupted`].
fn run_step(step: &Step, entries: &mut SlotMap<Entry>, forward: bool) -> Result<Vec<Done>> {
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
        match run_change(&step.changes[order(i)], entries, forward) {
            Ok(changes) => done.push((step.changes[order(i)].mesh(), changes)),
            Err(e) => {
                for j in (0..i).rev() {
                    if run_change(&step.changes[order(j)], entries, !forward).is_err() {
                        return Err(SceneError::Corrupted);
                    }
                }
                return Err(e);
            }
        }
    }
    Ok(done)
}

/// Applies (`forward`) or reverts one change. Returns the mesh's changes, or
/// `None` for a placement change. On error the mesh is unchanged.
fn run_change(
    change: &Change,
    entries: &mut SlotMap<Entry>,
    forward: bool,
) -> Result<Option<ChangeSet>> {
    let entry = entries
        .get_mut(change.mesh())
        .ok_or(SceneError::Corrupted)?;
    match change {
        Change::Edit(_, patch) => {
            #[cfg(test)]
            crate::scene::fault::before_patch()?;
            let changes = if forward {
                patch.apply(&mut entry.mesh)?
            } else {
                patch.revert(&mut entry.mesh)?
            };
            Ok(Some(changes))
        }
        Change::Placement { old, new, .. } => {
            entry.placement = if forward { *new } else { *old };
            Ok(None)
        }
    }
}
