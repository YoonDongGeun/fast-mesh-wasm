use std::collections::VecDeque;

use crate::compact::Remap;
use crate::{ChangeSet, Mesh, Patch, PatchSide, Result, StoreError};

/// Undo/redo stacks of [`Patch`]es for one mesh, limited by a memory budget.
///
/// Every commit on the mesh must be pushed here (or the history cleared): undo and
/// redo only work when the mesh revision matches the top patch. A history belongs to
/// the mesh of its first pushed patch; every method that takes a mesh rejects any
/// other mesh with [`StoreError::PatchMismatch`].
///
/// The history also decides when removed slots may be reused. A slot freed by a
/// patch stays reserved while that patch can still be undone, and a slot created by
/// an undone patch stays reserved while it can be redone. Evicting or discarding the
/// patch releases the slot. Patches kept outside a `History` release their slots
/// through [`Mesh::release_patch_slots`], at most once and only once no history
/// still holds the patch; slots never released that way are reclaimed by
/// [`Mesh::compact`]. Patches made stale by a direct [`Mesh::compact`] release
/// nothing: their slots no longer exist.
///
/// The budget counts undo and redo patches together, but eviction only drops undo
/// patches (oldest first). While a large redo stack exists, [`History::bytes`] may
/// therefore exceed [`History::budget`]; the redo stack is dropped by the next push.
pub struct History {
    undo: VecDeque<Patch>,
    redo: Vec<Patch>,
    budget: usize,
    bytes: usize,
    /// Uid of the mesh this history belongs to, set by the first push.
    mesh_uid: Option<u64>,
}

impl History {
    /// A history that keeps at most `budget_bytes` of patches (undo and redo combined).
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            undo: VecDeque::new(),
            redo: Vec::new(),
            budget: budget_bytes,
            bytes: 0,
            mesh_uid: None,
        }
    }

    fn check_mesh(&self, mesh: &Mesh) -> Result<()> {
        match self.mesh_uid {
            Some(uid) if uid != mesh.uid => Err(StoreError::PatchMismatch),
            _ => Ok(()),
        }
    }

    /// Records a just-committed patch. Empty patches are ignored and keep the redo
    /// stack. Otherwise the redo stack is discarded and the oldest patches are evicted
    /// until the budget is met (possibly including this one, if it alone is too large).
    pub fn push(&mut self, mesh: &mut Mesh, patch: Patch) -> Result<()> {
        self.check_mesh(mesh)?;
        if patch.mesh_uid != mesh.uid || patch.revision != mesh.revision {
            return Err(StoreError::PatchMismatch);
        }
        self.mesh_uid = Some(mesh.uid);
        if patch.is_empty() {
            return Ok(());
        }
        for p in self.redo.drain(..) {
            self.bytes -= p.byte_size();
            p.release(mesh, PatchSide::Redoable);
        }
        self.bytes += patch.byte_size();
        self.undo.push_back(patch);
        self.evict(mesh);
        Ok(())
    }

    /// Undoes the latest patch. Returns `None` when there is nothing to undo.
    pub fn undo(&mut self, mesh: &mut Mesh) -> Result<Option<ChangeSet>> {
        self.check_mesh(mesh)?;
        let Some(patch) = self.undo.pop_back() else {
            return Ok(None);
        };
        match patch.revert(mesh) {
            Ok(changes) => {
                self.redo.push(patch);
                Ok(Some(changes))
            }
            Err(e) => {
                self.undo.push_back(patch);
                Err(e)
            }
        }
    }

    /// Redoes the latest undone patch. Returns `None` when there is nothing to redo.
    pub fn redo(&mut self, mesh: &mut Mesh) -> Result<Option<ChangeSet>> {
        self.check_mesh(mesh)?;
        let Some(patch) = self.redo.pop() else {
            return Ok(None);
        };
        match patch.apply(mesh) {
            Ok(changes) => {
                self.undo.push_back(patch);
                Ok(Some(changes))
            }
            Err(e) => {
                self.redo.push(patch);
                Err(e)
            }
        }
    }

    /// Drops all patches and releases the slots they reserved.
    pub fn clear(&mut self, mesh: &mut Mesh) -> Result<()> {
        self.check_mesh(mesh)?;
        for p in self.undo.drain(..) {
            p.release(mesh, PatchSide::Undoable);
        }
        for p in self.redo.drain(..) {
            p.release(mesh, PatchSide::Redoable);
        }
        self.bytes = 0;
        Ok(())
    }

    /// Clears the history and compacts the mesh (see [`Mesh::compact`]).
    pub fn compact(&mut self, mesh: &mut Mesh) -> Result<Remap> {
        self.check_mesh(mesh)?;
        let remap = mesh.compact()?;
        // Old patches are stale now, so their reserved slots are simply dropped.
        self.undo.clear();
        self.redo.clear();
        self.bytes = 0;
        Ok(remap)
    }

    /// Changes the budget, evicting old undo patches if needed.
    pub fn set_budget(&mut self, mesh: &mut Mesh, budget_bytes: usize) -> Result<()> {
        self.check_mesh(mesh)?;
        self.budget = budget_bytes;
        self.evict(mesh);
        Ok(())
    }

    fn evict(&mut self, mesh: &mut Mesh) {
        while self.bytes > self.budget {
            let Some(p) = self.undo.pop_front() else {
                break;
            };
            self.bytes -= p.byte_size();
            p.release(mesh, PatchSide::Undoable);
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    /// Memory held by all patches (undo and redo), in bytes. Can exceed the budget
    /// while a redo stack exists; see the type docs.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn budget(&self) -> usize {
        self.budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> Mesh {
        Mesh::from_buffers(
            &[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap()
    }

    fn move_vertex(m: &mut Mesh, h: &mut History, v: u32, x: f32) {
        let mut tx = m.begin_edit();
        tx.set_positions(&[v], &[[x, 0., 0.]]).unwrap();
        let p = tx.commit().unwrap();
        h.push(m, p).unwrap();
    }

    #[test]
    fn undo_and_redo_walk_the_stack() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        move_vertex(&mut m, &mut h, 0, 5.0);
        move_vertex(&mut m, &mut h, 0, 6.0);
        assert_eq!(h.undo(&mut m).unwrap().unwrap().vertices_moved, vec![0]);
        assert_eq!(m.position(0).unwrap()[0], 5.0);
        h.undo(&mut m).unwrap();
        assert_eq!(m.position(0).unwrap()[0], 0.0);
        assert!(h.undo(&mut m).unwrap().is_none());
        h.redo(&mut m).unwrap();
        h.redo(&mut m).unwrap();
        assert_eq!(m.position(0).unwrap()[0], 6.0);
        assert!(h.redo(&mut m).unwrap().is_none());
    }

    #[test]
    fn a_new_commit_discards_redo() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        move_vertex(&mut m, &mut h, 0, 5.0);
        h.undo(&mut m).unwrap();
        move_vertex(&mut m, &mut h, 1, 7.0);
        assert!(!h.can_redo());
        assert_eq!(h.undo_len(), 1);
    }

    #[test]
    fn empty_commits_keep_redo() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        move_vertex(&mut m, &mut h, 0, 5.0);
        h.undo(&mut m).unwrap();
        let empty = m.begin_edit().commit().unwrap();
        h.push(&mut m, empty).unwrap();
        assert!(h.can_redo());
    }

    #[test]
    fn budget_evicts_the_oldest_patches() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        move_vertex(&mut m, &mut h, 0, 1.0);
        let one = h.bytes();
        h.set_budget(&mut m, one * 2).unwrap();
        move_vertex(&mut m, &mut h, 0, 2.0);
        move_vertex(&mut m, &mut h, 0, 3.0);
        assert_eq!(h.undo_len(), 2);
        assert!(h.bytes() <= h.budget());
        h.undo(&mut m).unwrap();
        h.undo(&mut m).unwrap();
        assert_eq!(
            m.position(0).unwrap()[0],
            1.0,
            "the first edit can no longer be undone"
        );
    }

    #[test]
    fn removed_slots_are_reserved_until_their_patch_is_evicted() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[1]).unwrap();
        tx.remove_vertices(&[3]).unwrap();
        let p = tx.commit().unwrap();
        h.push(&mut m, p).unwrap();
        assert!(m.free_vertex_slots().is_empty(), "reserved while undoable");

        let mut tx = m.begin_edit();
        assert_eq!(
            tx.add_vertices(&[[0.; 3]]).unwrap(),
            vec![4],
            "slot 3 is not reused"
        );
        tx.rollback();

        h.clear(&mut m).unwrap();
        assert_eq!(m.free_vertex_slots(), &[3]);
        assert_eq!(m.free_triangle_slots(), &[1]);
    }

    #[test]
    fn created_slots_of_discarded_redo_are_released() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        let mut tx = m.begin_edit();
        tx.add_vertices(&[[0.; 3]]).unwrap();
        let p = tx.commit().unwrap();
        h.push(&mut m, p).unwrap();
        h.undo(&mut m).unwrap();
        assert!(m.free_vertex_slots().is_empty(), "reserved for redo");
        move_vertex(&mut m, &mut h, 0, 1.0);
        assert_eq!(m.free_vertex_slots(), &[4]);
    }

    #[test]
    fn history_compact_empties_history_and_applies_remap() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[0]).unwrap();
        tx.remove_vertices(&[1]).unwrap();
        let p = tx.commit().unwrap();
        h.push(&mut m, p).unwrap();

        let remap = h.compact(&mut m).unwrap();
        assert!(!h.can_undo());
        assert_eq!(h.bytes(), 0);
        assert_eq!(remap.vertex(1), None);
        assert_eq!(remap.vertex(2), Some(1));
        assert_eq!(remap.vertex(3), Some(2));
        assert_eq!(remap.triangle(0), None);
        assert_eq!(remap.triangle(1), Some(0));
        assert_eq!(m.vertex_count(), 3);
        assert_eq!(m.position(1), Some([1., 1., 0.]));
        assert_eq!(m.triangle(0), Some([0, 1, 2]));
    }

    /// Four vertices, no triangles.
    fn points() -> Mesh {
        Mesh::from_buffers(&[0.; 12], &[]).unwrap()
    }

    fn remove_vertex(m: &mut Mesh, h: &mut History, v: u32) {
        let mut tx = m.begin_edit();
        tx.remove_vertices(&[v]).unwrap();
        let p = tx.commit().unwrap();
        h.push(m, p).unwrap();
    }

    #[test]
    fn direct_compact_then_clear_does_not_corrupt() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[0]).unwrap();
        tx.remove_vertices(&[1]).unwrap();
        let p = tx.commit().unwrap();
        h.push(&mut m, p).unwrap();
        m.compact().unwrap();
        // The patch is stale: its reserved slot 1 no longer exists (vertex 1 is now
        // the old vertex 2, alive).
        h.clear(&mut m).unwrap();
        assert!(m.free_vertex_slots().is_empty());
        assert!(m.free_triangle_slots().is_empty());
        let mut tx = m.begin_edit();
        assert_eq!(tx.add_vertices(&[[0.; 3]]).unwrap(), vec![3]);
    }

    #[test]
    fn direct_compact_then_push_or_set_budget_does_not_corrupt() {
        let mut m = points();
        let mut h = History::new(1 << 20);
        remove_vertex(&mut m, &mut h, 0);
        h.clear(&mut m).unwrap();
        assert_eq!(m.free_vertex_slots(), &[0]);
        // Re-create slot 0, then undo: the redo patch reserves slot 0.
        let mut tx = m.begin_edit();
        assert_eq!(tx.add_vertices(&[[1.; 3]]).unwrap(), vec![0]);
        let p = tx.commit().unwrap();
        h.push(&mut m, p).unwrap();
        remove_vertex(&mut m, &mut h, 3);
        h.undo(&mut m).unwrap();
        h.undo(&mut m).unwrap();
        // Now: vertices 1, 2, 3 alive; redo holds "remove 3" and "add 0".
        m.compact().unwrap();
        assert_eq!(m.vertex_count(), 3);
        move_vertex(&mut m, &mut h, 0, 5.0); // discards the stale redo patches
        assert!(m.free_vertex_slots().is_empty(), "slot 0 is alive again");
        h.set_budget(&mut m, 0).unwrap();
        assert!(m.free_vertex_slots().is_empty());
        let mut tx = m.begin_edit();
        assert_eq!(tx.add_vertices(&[[0.; 3]]).unwrap(), vec![3]);
    }

    #[test]
    fn undo_of_a_patch_made_stale_by_direct_compact_fails_cleanly() {
        let mut m = points();
        let mut h = History::new(1 << 20);
        remove_vertex(&mut m, &mut h, 1);
        m.compact().unwrap();
        let revision = m.revision();
        assert_eq!(h.undo(&mut m).err(), Some(StoreError::StalePatch));
        assert_eq!(m.revision(), revision);
        assert_eq!(m.vertex_count(), 3);
        h.clear(&mut m).unwrap();
        assert!(m.free_vertex_slots().is_empty());
    }

    #[test]
    fn history_used_with_a_second_mesh_is_rejected() {
        let mut a = points();
        let mut b = points();
        let mut h = History::new(1 << 20);
        remove_vertex(&mut a, &mut h, 1);
        assert_eq!(h.clear(&mut b).err(), Some(StoreError::PatchMismatch));
        assert_eq!(
            h.set_budget(&mut b, 0).err(),
            Some(StoreError::PatchMismatch)
        );
        assert_eq!(h.undo(&mut b).err(), Some(StoreError::PatchMismatch));
        assert_eq!(h.redo(&mut b).err(), Some(StoreError::PatchMismatch));
        assert_eq!(h.compact(&mut b).err(), Some(StoreError::PatchMismatch));
        let mut tx = b.begin_edit();
        tx.remove_vertices(&[2]).unwrap();
        let pb = tx.commit().unwrap();
        assert_eq!(h.push(&mut b, pb).err(), Some(StoreError::PatchMismatch));
        assert!(b.free_vertex_slots().is_empty());
        assert_eq!(b.vertex_slots(), 4, "b was not compacted");
        assert_eq!(h.undo_len(), 1, "history is unchanged");
        h.clear(&mut a).unwrap();
        assert_eq!(a.free_vertex_slots(), &[1]);
    }

    #[test]
    fn rejects_patches_that_are_not_the_latest_commit() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[1.; 3]]).unwrap();
        let first = tx.commit().unwrap();
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[2.; 3]]).unwrap();
        let _second = tx.commit().unwrap();
        assert_eq!(h.push(&mut m, first).err(), Some(StoreError::PatchMismatch));
    }
}
