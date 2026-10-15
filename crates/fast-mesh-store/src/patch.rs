use crate::channel::Channel;
use crate::column::ColumnPatch;
use crate::{Mesh, Result, StoreError, TriangleId, VertexId};

/// What an edit, undo or redo changed. All lists are sorted and free of duplicates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChangeSet {
    pub vertices_added: Vec<VertexId>,
    pub vertices_removed: Vec<VertexId>,
    /// Existing vertices whose position changed.
    pub vertices_moved: Vec<VertexId>,
    pub triangles_added: Vec<TriangleId>,
    pub triangles_removed: Vec<TriangleId>,
    /// Existing triangles whose vertex ids changed.
    pub triangles_changed: Vec<TriangleId>,
    /// Source channels written for existing vertices: (channel name, vertex ids).
    pub channels: Vec<(String, Vec<VertexId>)>,
}

impl ChangeSet {
    /// True when triangles were added, removed or rewritten.
    pub fn topology_changed(&self) -> bool {
        !(self.triangles_added.is_empty()
            && self.triangles_removed.is_empty()
            && self.triangles_changed.is_empty())
    }

    pub fn is_empty(&self) -> bool {
        !self.topology_changed()
            && self.vertices_added.is_empty()
            && self.vertices_removed.is_empty()
            && self.vertices_moved.is_empty()
            && self.channels.iter().all(|(_, v)| v.is_empty())
    }

    fn inverted(mut self) -> Self {
        std::mem::swap(&mut self.vertices_added, &mut self.vertices_removed);
        std::mem::swap(&mut self.triangles_added, &mut self.triangles_removed);
        self
    }
}

/// The recorded effect of one committed transaction. Re-apply it with
/// [`Patch::apply`] (redo) or undo it with [`Patch::revert`].
///
/// A patch only applies to the mesh that produced it, at the matching revision:
/// `revert` needs the mesh exactly as the commit left it, `apply` needs it exactly as
/// it was before the commit.
#[derive(Debug, Clone)]
pub struct Patch {
    pub(crate) mesh_uid: u64,
    pub(crate) epoch: u32,
    pub(crate) base_revision: u64,
    pub(crate) revision: u64,
    pub(crate) positions: ColumnPatch<[f32; 3]>,
    pub(crate) triangles: ColumnPatch<[u32; 3]>,
    pub(crate) vertex_live: ColumnPatch<u64>,
    pub(crate) triangle_live: ColumnPatch<u64>,
    pub(crate) vertex_refs: ColumnPatch<u32>,
    /// Per source channel, keyed by channel id (not name).
    pub(crate) channel_data: Vec<(u64, ColumnPatch<f32>)>,
    pub(crate) changes: ChangeSet,
}

impl Patch {
    /// The changes this patch makes when applied.
    pub fn changes(&self) -> &ChangeSet {
        &self.changes
    }

    /// True when the transaction changed nothing.
    pub fn is_empty(&self) -> bool {
        self.base_revision == self.revision
    }

    /// Mesh revision before the commit.
    pub fn base_revision(&self) -> u64 {
        self.base_revision
    }

    /// Mesh revision after the commit.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Approximate heap memory held by this patch.
    pub fn byte_size(&self) -> usize {
        let c = &self.changes;
        let ids = c.vertices_added.len()
            + c.vertices_removed.len()
            + c.vertices_moved.len()
            + c.triangles_added.len()
            + c.triangles_removed.len()
            + c.triangles_changed.len()
            + c.channels.iter().map(|(_, v)| v.len()).sum::<usize>();
        self.positions.byte_size()
            + self.triangles.byte_size()
            + self.vertex_live.byte_size()
            + self.triangle_live.byte_size()
            + self.vertex_refs.byte_size()
            + self
                .channel_data
                .iter()
                .map(|(_, p)| 8 + p.byte_size())
                .sum::<usize>()
            + ids * 4
    }

    /// Re-applies the patch (redo). The mesh must be at [`Patch::base_revision`].
    /// If memory runs out, the mesh is left unchanged.
    pub fn apply(&self, mesh: &mut Mesh) -> Result<ChangeSet> {
        self.check(mesh, self.base_revision)?;
        // Phase A (the only fallible step): reserve capacity on every column. This
        // never changes a length, so a failure leaves the mesh exactly as it was.
        let vertex_slots = mesh.positions.len().max(self.positions.required_len());
        mesh.positions
            .try_reserve_to(self.positions.required_len())?;
        mesh.triangles
            .try_reserve_to(self.triangles.required_len())?;
        mesh.vertex_live
            .try_reserve_to(self.vertex_live.required_len())?;
        mesh.triangle_live
            .try_reserve_to(self.triangle_live.required_len())?;
        mesh.vertex_refs
            .try_reserve_to(self.vertex_refs.required_len())?;
        for c in &mut mesh.channels {
            c.data.try_reserve_to(self.channel_len(c, vertex_slots))?;
        }

        // Phase B: grow within the reserved capacity. Only adds dead slots.
        mesh.positions
            .resize_reserved(self.positions.required_len(), [0.0; 3]);
        mesh.triangles
            .resize_reserved(self.triangles.required_len(), [0; 3]);
        mesh.vertex_live
            .resize_reserved(self.vertex_live.required_len(), 0);
        mesh.triangle_live
            .resize_reserved(self.triangle_live.required_len(), 0);
        mesh.vertex_refs
            .resize_reserved(self.vertex_refs.required_len(), 0);
        for c in &mut mesh.channels {
            let len = self.channel_len(c, vertex_slots);
            c.data.resize_reserved(len, c.desc.default);
        }

        // Phase C: write. From here on nothing can fail (the incidence index is
        // dropped instead if it cannot follow).
        let c = &self.changes;
        let ids = [
            &c.triangles_added[..],
            &c.triangles_removed[..],
            &c.triangles_changed[..],
        ];
        mesh.with_incidence(|inc, m| inc.remove_alive(m, &ids));
        self.positions.apply(&mut mesh.positions);
        self.triangles.apply(&mut mesh.triangles);
        self.vertex_live.apply(&mut mesh.vertex_live);
        self.triangle_live.apply(&mut mesh.triangle_live);
        self.vertex_refs.apply(&mut mesh.vertex_refs);
        for (id, p) in &self.channel_data {
            if let Some(i) = mesh.channel_index_by_id(*id) {
                p.apply(&mut mesh.channels[i].data);
            }
        }
        mesh.live_vertices = mesh.live_vertices + c.vertices_added.len() - c.vertices_removed.len();
        mesh.live_triangles =
            mesh.live_triangles + c.triangles_added.len() - c.triangles_removed.len();
        mesh.revision = self.revision;
        if c.topology_changed() {
            mesh.adjacency = None;
            mesh.with_incidence(|inc, m| inc.insert_alive(m, &ids));
        }
        Ok(self.changes.clone())
    }

    /// Undoes the patch. The mesh must be at [`Patch::revision`].
    pub fn revert(&self, mesh: &mut Mesh) -> Result<ChangeSet> {
        self.check(mesh, self.revision)?;
        let c = &self.changes;
        let ids = [
            &c.triangles_added[..],
            &c.triangles_removed[..],
            &c.triangles_changed[..],
        ];
        mesh.with_incidence(|inc, m| inc.remove_alive(m, &ids));
        self.positions.revert(&mut mesh.positions);
        self.triangles.revert(&mut mesh.triangles);
        self.vertex_live.revert(&mut mesh.vertex_live);
        self.triangle_live.revert(&mut mesh.triangle_live);
        self.vertex_refs.revert(&mut mesh.vertex_refs);
        for (id, p) in &self.channel_data {
            if let Some(i) = mesh.channel_index_by_id(*id) {
                p.revert(&mut mesh.channels[i].data);
            }
        }
        mesh.live_vertices = mesh.live_vertices + c.vertices_removed.len() - c.vertices_added.len();
        mesh.live_triangles =
            mesh.live_triangles + c.triangles_removed.len() - c.triangles_added.len();
        mesh.revision = self.base_revision;
        if c.topology_changed() {
            mesh.adjacency = None;
            mesh.with_incidence(|inc, m| inc.insert_alive(m, &ids));
        }
        Ok(self.changes.clone().inverted())
    }

    /// Length channel `c` needs after [`Patch::apply`]: every vertex slot, and
    /// everything this patch writes to it.
    fn channel_len(&self, c: &Channel, vertex_slots: usize) -> usize {
        let own = self
            .channel_data
            .iter()
            .find(|(id, _)| *id == c.id)
            .map_or(0, |(_, p)| p.required_len());
        own.max(vertex_slots * c.desc.width)
    }

    fn check(&self, mesh: &Mesh, expected_revision: u64) -> Result<()> {
        if self.mesh_uid != mesh.uid {
            return Err(StoreError::PatchMismatch);
        }
        if self.epoch != mesh.epoch {
            return Err(StoreError::StalePatch);
        }
        if mesh.revision != expected_revision {
            return Err(StoreError::PatchMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::column::fail_allocations_after;
    use crate::{ChannelDesc, Mesh, StoreError};

    fn quad() -> Mesh {
        Mesh::from_buffers(
            &[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap()
    }

    /// Lengths of every column that must stay aligned with the slot counts.
    fn lengths(m: &Mesh) -> Vec<usize> {
        let mut l = vec![
            m.positions.len(),
            m.vertex_refs.len(),
            m.vertex_live.len(),
            m.triangles.len(),
            m.triangle_live.len(),
        ];
        l.extend(m.channels.iter().map(|c| c.data.len()));
        l
    }

    #[test]
    fn apply_that_runs_out_of_memory_leaves_every_column_unchanged() {
        let mut m = quad();
        m.add_channel(ChannelDesc::source("c", 2)).unwrap();
        m.add_channel(ChannelDesc::derived("d", 1)).unwrap();
        let mut tx = m.begin_edit();
        // 4 + 70 vertices: the liveness bitset grows past one word.
        let vs = tx.add_vertices(&[[2.; 3]; 70]).unwrap();
        tx.add_triangles(&[[0, vs[0], vs[1]]]).unwrap();
        tx.write_channel("c", &[0], &[5., 6.]).unwrap();
        let patch = tx.commit().unwrap();
        patch.revert(&mut m).unwrap();
        // Recreate the exact pre-commit layout (revert keeps appended slots allocated).
        m.positions.truncate(4);
        m.vertex_refs.truncate(4);
        m.vertex_live.truncate(1);
        m.triangles.truncate(2);
        m.triangle_live.truncate(1);
        for c in &mut m.channels {
            let w = c.desc.width;
            c.data.truncate(4 * w);
        }
        let before = lengths(&m);
        let revision = m.revision();
        for k in 0.. {
            fail_allocations_after(Some(k));
            let result = patch.apply(&mut m);
            fail_allocations_after(None);
            match result {
                Err(e) => {
                    assert_eq!(e, StoreError::OutOfMemory);
                    assert_eq!(lengths(&m), before, "failed after {k} allocations");
                    assert_eq!(m.revision(), revision);
                    assert_eq!((m.vertex_count(), m.triangle_count()), (4, 2));
                }
                Ok(_) => {
                    assert!(k > 0, "apply needs to allocate");
                    assert_eq!((m.vertex_count(), m.triangle_count()), (74, 3));
                    assert_eq!(lengths(&m), vec![74, 74, 2, 3, 1, 148, 74]);
                    assert_eq!(&m.channel("c").unwrap()[..2], &[5., 6.]);
                    assert_eq!(m.triangle(2), Some([0, vs[0], vs[1]]));
                    break;
                }
            }
        }
    }
}
