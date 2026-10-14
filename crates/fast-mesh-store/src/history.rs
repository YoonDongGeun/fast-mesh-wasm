use std::collections::VecDeque;

use crate::compact::Remap;
use crate::{ChangeSet, Mesh, Patch, Result, StoreError};

/// Undo/redo stacks of [`Patch`]es for one mesh, limited by a memory budget.
///
/// Every commit on the mesh must be pushed here (or the history cleared): undo and
/// redo only work when the mesh revision matches the top patch.
///
/// The history also decides when removed slots may be reused. A slot freed by a
/// patch stays reserved while that patch can still be undone, and a slot created by
/// an undone patch stays reserved while it can be redone. Evicting or discarding the
/// patch releases the slot. Patches kept outside a `History` never release their
/// slots; those are reclaimed by [`Mesh::compact`].
pub struct History {
    undo: VecDeque<Patch>,
    redo: Vec<Patch>,
    budget: usize,
    bytes: usize,
}

impl History {
    /// A history that keeps at most `budget_bytes` of patches (undo and redo combined).
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            undo: VecDeque::new(),
            redo: Vec::new(),
            budget: budget_bytes,
            bytes: 0,
        }
    }

    /// Records a just-committed patch. Empty patches are ignored and keep the redo
    /// stack. Otherwise the redo stack is discarded and the oldest patches are evicted
    /// until the budget is met (possibly including this one, if it alone is too large).
    pub fn push(&mut self, mesh: &mut Mesh, patch: Patch) -> Result<()> {
        if patch.mesh_uid != mesh.uid || patch.revision != mesh.revision {
            return Err(StoreError::PatchMismatch);
        }
        if patch.is_empty() {
            return Ok(());
        }
        for p in self.redo.drain(..) {
            self.bytes -= p.byte_size();
            mesh.release_vertices(&p.changes.vertices_added);
            mesh.release_triangles(&p.changes.triangles_added);
        }
        self.bytes += patch.byte_size();
        self.undo.push_back(patch);
        self.evict(mesh);
        Ok(())
    }

    /// Undoes the latest patch. Returns `None` when there is nothing to undo.
    pub fn undo(&mut self, mesh: &mut Mesh) -> Result<Option<ChangeSet>> {
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
    pub fn clear(&mut self, mesh: &mut Mesh) {
        for p in self.undo.drain(..) {
            mesh.release_vertices(&p.changes.vertices_removed);
            mesh.release_triangles(&p.changes.triangles_removed);
        }
        for p in self.redo.drain(..) {
            mesh.release_vertices(&p.changes.vertices_added);
            mesh.release_triangles(&p.changes.triangles_added);
        }
        self.bytes = 0;
    }

    /// Clears the history and compacts the mesh (see [`Mesh::compact`]).
    pub fn compact(&mut self, mesh: &mut Mesh) -> Result<Remap> {
        let remap = mesh.compact()?;
        // Old patches are stale now, so their reserved slots are simply dropped.
        self.undo.clear();
        self.redo.clear();
        self.bytes = 0;
        Ok(remap)
    }

    /// Changes the budget, evicting old patches if needed.
    pub fn set_budget(&mut self, mesh: &mut Mesh, budget_bytes: usize) {
        self.budget = budget_bytes;
        self.evict(mesh);
    }

    fn evict(&mut self, mesh: &mut Mesh) {
        while self.bytes > self.budget {
            let Some(p) = self.undo.pop_front() else {
                break;
            };
            self.bytes -= p.byte_size();
            mesh.release_vertices(&p.changes.vertices_removed);
            mesh.release_triangles(&p.changes.triangles_removed);
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

    /// Memory held by all patches, in bytes.
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
        h.set_budget(&mut m, one * 2);
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

        h.clear(&mut m);
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
