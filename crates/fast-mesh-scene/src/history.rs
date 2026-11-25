//! The shared undo history: steps of edits, placement changes and undoable
//! adds and removals across meshes, limited by a memory budget.

use std::collections::VecDeque;
use std::mem::size_of;

use fast_mesh_store::{ChangeSet, Mesh, Patch, PatchSide, StoreError};

use crate::id::SlotMap;
use crate::scene::Entry;
use crate::{MeshId, Placement, Result, SceneError};

/// Bytes a placement entry counts against the budget: its old and new value.
pub(crate) const PLACEMENT_BYTES: usize = 2 * size_of::<Placement>();

/// What can no longer be done with a [`PermanentStep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistorySide {
    /// The step was just made (or redone), and cannot be undone.
    Undo,
    /// The step was just undone, and cannot be redone.
    Redo,
}

/// A step the history let go of as it was made, undone or redone, since it is
/// larger than the budget (see
/// [`Scene::take_permanent`](crate::Scene::take_permanent)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermanentStep {
    /// What can no longer be done with it.
    pub lost: HistorySide,
    /// The meshes it edited, moved, added or removed, each once, in recording
    /// order. Those in `gone` no longer match.
    pub meshes: Vec<MeshId>,
    /// Of those, the meshes that are out of the scene with nothing left to
    /// bring them back.
    pub gone: Vec<MeshId>,
    /// The bytes it counted, against
    /// [`Scene::history_budget`](crate::Scene::history_budget).
    pub bytes: u64,
}

/// The most [`PermanentStep`]s kept for `take_permanent`: the latest.
const MAX_PERMANENT: usize = 256;

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
    /// what the mesh held when it was last parked by its undo (see
    /// `Entry::byte_size`).
    Add { id: MeshId, bytes: usize },
    /// An undoable removal: the mesh is parked, undo restores it and redo
    /// parks it again. `bytes` is what the mesh held when it was last parked.
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

    /// The bytes the change holds on `side`. Only the entry a parked mesh is
    /// restored by holds it: its removal on the undo side, its add on the
    /// redo side. On the other side the mesh is live, or held by another
    /// entry.
    fn bytes(&self, side: PatchSide) -> u64 {
        let bytes = match self {
            Change::Edit(_, p) => p.byte_size(),
            Change::Placement { .. } => PLACEMENT_BYTES,
            Change::Add { bytes, .. } if side == PatchSide::Redoable => *bytes,
            Change::Remove { bytes, .. } if side == PatchSide::Undoable => *bytes,
            Change::Add { .. } | Change::Remove { .. } => 0,
        };
        bytes as u64
    }
}

/// Releases what `change` holds once the history drops it from `side`: the
/// slots its patch reserves (on its mesh, live or parked), or the parked mesh
/// it was the last way back to. Call once, when the change is discarded.
///
/// An add dropped from the redo side, or a removal dropped from the undo
/// side, frees its parked mesh: the id never matches again. An add dropped
/// from the undo side, or a removal from the redo side, leaves its live mesh
/// alone: it just stays.
///
/// Every change in the history belongs to a live or a parked mesh, or to one
/// another change of the same batch just freed: `remove_mesh` drops a live
/// mesh's changes before the mesh goes, and a parked mesh is freed only here.
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

/// The changes of one step, in recording order.
#[derive(Default)]
pub(crate) struct Step {
    changes: Vec<Change>,
    /// What the changes hold on the stack the step is on.
    bytes: u64,
}

impl Step {
    /// Counts what the step holds now that it is on `side`'s stack: the
    /// meshes it just parked are measured.
    fn recount(&mut self, entries: &SlotMap<Entry>, side: PatchSide) {
        for c in &mut self.changes {
            if let Change::Add { id, bytes } | Change::Remove { id, bytes } = c
                && let Some(e) = entries.get_parked(*id)
            {
                *bytes = e.byte_size();
            }
        }
        self.bytes = self.changes.iter().map(|c| c.bytes(side)).sum();
    }

    /// The report that the step cannot be undone, or redone, any more.
    fn permanent(&self, lost: HistorySide) -> PermanentStep {
        let mut meshes = Vec::new();
        let mut gone = Vec::new();
        for c in &self.changes {
            let id = c.mesh();
            if !meshes.contains(&id) {
                meshes.push(id);
            }
            let out = match c {
                Change::Remove { .. } => lost == HistorySide::Undo,
                Change::Add { .. } => lost == HistorySide::Redo,
                Change::Edit(..) | Change::Placement { .. } => false,
            };
            if out && !gone.contains(&id) {
                gone.push(id);
            }
        }
        PermanentStep {
            lost,
            meshes,
            gone,
            bytes: self.bytes,
        }
    }
}

/// What one change did when a step was undone or redone.
pub(crate) enum Done {
    /// An edit was applied or reverted: the mesh and its changes.
    Edit(MeshId, ChangeSet),
    /// A placement entry was applied or reverted.
    Placement(MeshId),
    /// The mesh was parked or restored: its slot says which. The scene has
    /// yet to bring its registries in line (`Scene::settle_presence`).
    Presence(MeshId),
}

/// Undo and redo stacks of steps, plus the step being recorded.
///
/// `bytes` counts every step held (undo, redo and the open one). Eviction
/// drops the oldest undo steps; it runs whenever a change is recorded, at
/// which point the redo stack is already empty. The open step is never
/// evicted while open; once closed, it is evicted like any other step, so a
/// step larger than the whole budget is dropped as soon as it closes. An
/// undo or a redo always happens, and then meets the budget, which it
/// exceeds only when it undid a non-empty open step or parked meshes (see
/// `trim_undone` and `trim_redone`). A step dropped as it closes, is undone
/// or is redone is noted in `permanent`.
pub(crate) struct History {
    undo: VecDeque<Step>,
    redo: Vec<Step>,
    open: Step,
    /// Nesting depth of `begin_step`; 0 when no step is open.
    depth: u32,
    budget: usize,
    bytes: u64,
    /// Steps dropped as they were made, undone or redone, not taken yet.
    permanent: VecDeque<PermanentStep>,
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
            permanent: VecDeque::new(),
        }
    }

    /// Most bytes the steps may hold.
    pub(crate) fn budget(&self) -> usize {
        self.budget
    }

    /// Changes the budget and meets it: the oldest undo steps go first, then
    /// the furthest redo steps. An open step is kept, and meets it once
    /// closed.
    pub(crate) fn set_budget(&mut self, entries: &mut SlotMap<Entry>, budget: usize) {
        self.budget = budget;
        self.evict(entries);
        self.evict_redo(entries, 0);
    }

    /// Bytes held by every step.
    pub(crate) fn held(&self) -> u64 {
        self.bytes
    }

    /// Whether the steps hold more than the budget.
    fn over(&self) -> bool {
        self.bytes > self.budget as u64
    }

    /// Takes the steps noted as permanent since the last call.
    pub(crate) fn take_permanent(&mut self) -> Vec<PermanentStep> {
        std::mem::take(&mut self.permanent).into()
    }

    /// Whether `take_permanent` would return any.
    pub(crate) fn has_permanent(&self) -> bool {
        !self.permanent.is_empty()
    }

    /// Notes the step just made, undone or redone as permanent: `dropped`,
    /// the last of those dropped to meet the budget, is it once `left` (what
    /// is left of the stack it was on) is empty.
    fn note_permanent(&mut self, dropped: Option<Step>, left: usize, lost: HistorySide) {
        let Some(step) = dropped.filter(|_| left == 0) else {
            return;
        };
        if self.permanent.len() == MAX_PERMANENT {
            self.permanent.pop_front();
        }
        self.permanent.push_back(step.permanent(lost));
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
        let dropped = self.evict(entries);
        self.note_permanent(dropped, self.undo.len(), HistorySide::Undo);
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
        let bytes = change.bytes(PatchSide::Undoable);
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

    /// Drops the oldest undo steps until the budget is met, or none are left.
    /// Returns the last one dropped.
    fn evict(&mut self, entries: &mut SlotMap<Entry>) -> Option<Step> {
        let mut last = None;
        while self.over() {
            let Some(step) = self.evict_one(entries) else {
                break;
            };
            last = Some(step);
        }
        last
    }

    /// Drops the oldest undo step and returns it, its changes released.
    fn evict_one(&mut self, entries: &mut SlotMap<Entry>) -> Option<Step> {
        let step = self.undo.pop_front()?;
        self.bytes -= step.bytes;
        for c in &step.changes {
            release_in(c, entries, PatchSide::Undoable);
        }
        Some(step)
    }

    /// Closes the open step, then drops every step, releasing their slots.
    pub(crate) fn clear(&mut self, entries: &mut SlotMap<Entry>) {
        self.depth = 0;
        let open = std::mem::take(&mut self.open);
        for step in self.undo.drain(..).chain([open]) {
            for c in &step.changes {
                release_in(c, entries, PatchSide::Undoable);
            }
        }
        self.drop_redo(entries);
        self.bytes = 0;
    }

    /// Drops the furthest redo steps until the budget is met, keeping the
    /// last `keep`. Returns the last one dropped. Only needed after an undo,
    /// a redo (see `trim_undone` and `trim_redone`) or a smaller budget.
    fn evict_redo(&mut self, entries: &mut SlotMap<Entry>, keep: usize) -> Option<Step> {
        let mut bytes = self.bytes;
        let mut n = 0;
        while bytes > self.budget as u64 && self.redo.len() - n > keep {
            bytes -= self.redo[n].bytes;
            n += 1;
        }
        self.bytes = bytes;
        let mut last = None;
        for step in self.redo.drain(..n) {
            for c in &step.changes {
                release_in(c, entries, PatchSide::Redoable);
            }
            last = Some(step);
        }
        last
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
    /// mid-drag undoes the drag). It then goes to the redo stack.
    ///
    /// The caller meets the budget with `trim_undone` once it has settled the
    /// parked and restored meshes, some of which that may free.
    pub(crate) fn undo(&mut self, entries: &mut SlotMap<Entry>) -> Result<Option<Vec<Done>>> {
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
            let done = match run_step(&self.open, entries, false) {
                Ok(done) => done,
                Err(e) => {
                    if e == SceneError::Corrupted {
                        self.drop_open();
                    }
                    return Err(e);
                }
            };
            self.depth = 0;
            let mut step = std::mem::take(&mut self.open);
            self.bytes -= step.bytes;
            step.recount(entries, PatchSide::Redoable);
            self.bytes += step.bytes;
            self.redo.push(step);
            return Ok(Some(done));
        }
        let mut step = self.undo.pop_back().ok_or(SceneError::Corrupted)?;
        match run_step(&step, entries, false) {
            Ok(done) => {
                self.depth = 0;
                self.bytes -= step.bytes;
                step.recount(entries, PatchSide::Redoable);
                self.bytes += step.bytes;
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
    /// there is nothing to redo. Errors as for `undo`. The caller meets the
    /// budget with `trim_redone`, as for `undo`.
    pub(crate) fn redo(&mut self, entries: &mut SlotMap<Entry>) -> Result<Option<Vec<Done>>> {
        if self.redo.is_empty() {
            // Closing may evict the closed step, as `end_step` would; there
            // is nothing to redo either way.
            self.close(entries);
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
        let mut step = self.redo.pop().ok_or(SceneError::Corrupted)?;
        match run_step(&step, entries, true) {
            Ok(done) => {
                self.depth = 0;
                self.bytes -= step.bytes;
                step.recount(entries, PatchSide::Undoable);
                self.bytes += step.bytes;
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

    /// Meets the budget after `undo`. What was undone before goes first (it
    /// is the furthest to redo), then the oldest undo steps, then, if that
    /// is not enough, the step itself. A step larger than the whole budget
    /// that parked meshes goes at once, sparing the undo steps.
    pub(crate) fn trim_undone(&mut self, entries: &mut SlotMap<Entry>) {
        self.evict_redo(entries, 1);
        let adds = |s: &Step| s.changes.iter().any(|c| matches!(c, Change::Add { .. }));
        let mut dropped = None;
        if self
            .redo
            .last()
            .is_some_and(|s| s.bytes > self.budget as u64 && adds(s))
        {
            dropped = self.evict_redo(entries, 0);
        }
        self.evict(entries);
        let dropped = self.evict_redo(entries, 0).or(dropped);
        self.note_permanent(dropped, self.redo.len(), HistorySide::Redo);
    }

    /// Meets the budget after `redo`, which exceeds it only when it parked
    /// meshes: the oldest undo steps go first, then the furthest redo steps,
    /// then, if that is not enough, the step itself (its removals are then
    /// permanent).
    pub(crate) fn trim_redone(&mut self, entries: &mut SlotMap<Entry>) {
        while self.over() && self.undo.len() > 1 {
            self.evict_one(entries);
        }
        self.evict_redo(entries, 0);
        let dropped = self.evict(entries);
        self.note_permanent(dropped, self.undo.len(), HistorySide::Undo);
    }

    /// Reverts the open step's changes and takes the step out, for the caller
    /// to `discard` once it has settled the parked and restored meshes. On
    /// error the step stays open, unchanged, except that
    /// [`SceneError::Corrupted`] discards it.
    pub(crate) fn cancel(&mut self, entries: &mut SlotMap<Entry>) -> Result<(Vec<Done>, Step)> {
        if self.depth == 0 {
            return Err(SceneError::StepNotOpen);
        }
        let done = match run_step(&self.open, entries, false) {
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
        Ok((done, step))
    }

    /// Releases what a step reverted by `cancel` holds: the slots its patches
    /// created, and the meshes it added.
    pub(crate) fn discard(step: &Step, entries: &mut SlotMap<Entry>) {
        for c in &step.changes {
            release_in(c, entries, PatchSide::Redoable);
        }
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
    /// - With `mesh` (removal for good): every change of the mesh goes, and
    ///   the slots its patches reserve on `mesh` are released. Its add and
    ///   removal entries hold nothing: the mesh is live, so its add is on the
    ///   undo side and a removal on the redo side.
    /// - Without (compaction): only its edit changes go, releasing nothing,
    ///   since their slots no longer exist; placement, add and removal
    ///   changes stay valid and are kept.
    pub(crate) fn forget(&mut self, id: MeshId, mut mesh: Option<&mut Mesh>) {
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
            step.bytes = step.changes.iter().map(|c| c.bytes(side)).sum();
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
        self.bytes = self.undo.iter().map(|s| s.bytes).sum::<u64>()
            + self.redo.iter().map(|s| s.bytes).sum::<u64>()
            + self.open.bytes;
    }

    /// Number of closed steps that can be undone.
    #[cfg(test)]
    pub(crate) fn undo_len(&self) -> usize {
        self.undo.len()
    }

    /// Number of steps that can be redone.
    #[cfg(test)]
    pub(crate) fn redo_len(&self) -> usize {
        self.redo.len()
    }

    /// Bytes of each undo step, oldest first.
    #[cfg(test)]
    pub(crate) fn undo_step_bytes(&self) -> impl Iterator<Item = usize> + '_ {
        self.undo.iter().map(|s| s.bytes as usize)
    }

    /// Bytes held by every step.
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.bytes as usize
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
///
/// Adds and removals only flip their mesh between live and parked here, which
/// cannot fail, so they always recover. The scene's registries follow once
/// the whole step has run (`Scene::settle_presence`), and a step that fails
/// never touches them.
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
            Ok(d) => done.push(d),
            Err(e) => {
                for j in (0..i).rev() {
                    if run_change(&step.changes[order(j)], entries, !forward).is_err() {
                        // The adds and removals still to recover: no mesh may
                        // stay where the scene's registries do not expect it.
                        for k in (0..j).rev() {
                            run_presence(&step.changes[order(k)], entries, !forward);
                        }
                        mark_corrupted(step, entries);
                        release_partial(step, entries, forward, i, j);
                        return Err(SceneError::Corrupted);
                    }
                }
                if e == SceneError::Corrupted {
                    // The step names a mesh that is gone (defensive: removal
                    // drops its changes), but recovery restored everything,
                    // so every patch is back at a known state.
                    release_recovered(step, entries, forward);
                }
                return Err(e);
            }
        }
    }
    Ok(done)
}

/// Releases the patches of a step dropped after [`SceneError::Corrupted`]
/// for the state each is known to be in. Change `i` is the one whose run
/// failed and `j` the one whose recovery failed; both are left reserved.
/// Changes run and recovered, or never run, are at the step's far state
/// (after it for undo, before it for redo), and so is every add and removal;
/// edits run and never recovered are at the run's end state.
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
        let c = &step.changes[order(k)];
        let presence = matches!(c, Change::Add { .. } | Change::Remove { .. });
        release_in(c, entries, if presence { far } else { end });
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

/// Applies (`forward`) or reverts one change. On error the mesh is unchanged.
fn run_change(change: &Change, entries: &mut SlotMap<Entry>, forward: bool) -> Result<Done> {
    match change {
        Change::Edit(id, patch) => {
            let entry = entries.get_mut(*id).ok_or(SceneError::Corrupted)?;
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
            let entry = entries.get_mut(*id).ok_or(SceneError::Corrupted)?;
            entry.placement = if forward { *new } else { *old };
            Ok(Done::Placement(*id))
        }
        Change::Add { id, .. } | Change::Remove { id, .. } => {
            if run_presence(change, entries, forward) {
                Ok(Done::Presence(*id))
            } else {
                // Defensive: the mesh is not where the change expects it.
                Err(SceneError::Corrupted)
            }
        }
    }
}

/// Parks or restores the mesh of an add or a removal: redoing an add, or
/// undoing a removal, restores it. False when the mesh is not where the
/// change expects it, and for other changes, which are left alone.
fn run_presence(change: &Change, entries: &mut SlotMap<Entry>, forward: bool) -> bool {
    match change {
        Change::Add { id, .. } if forward => entries.unpark(*id),
        Change::Add { id, .. } => entries.park(*id),
        Change::Remove { id, .. } if forward => entries.park(*id),
        Change::Remove { id, .. } => entries.unpark(*id),
        Change::Edit(..) | Change::Placement { .. } => false,
    }
}
