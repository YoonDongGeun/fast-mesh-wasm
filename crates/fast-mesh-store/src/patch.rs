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
    pub(crate) channel_data: Vec<(String, ColumnPatch<f32>)>,
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
                .map(|(n, p)| n.len() + p.byte_size())
                .sum::<usize>()
            + ids * 4
    }

    /// Re-applies the patch (redo). The mesh must be at [`Patch::base_revision`].
    /// If memory runs out, the mesh is left unchanged.
    pub fn apply(&self, mesh: &mut Mesh) -> Result<ChangeSet> {
        self.check(mesh, self.base_revision)?;
        // Growing is the only fallible step. It only adds dead slots, so a failure here
        // leaves the mesh logically unchanged.
        mesh.positions
            .grow_to(self.positions.required_len(), [0.0; 3])?;
        mesh.triangles.grow_to(self.triangles.required_len(), [0; 3])?;
        mesh.vertex_live.grow_to(self.vertex_live.required_len(), 0)?;
        mesh.triangle_live.grow_to(self.triangle_live.required_len(), 0)?;
        mesh.vertex_refs.grow_to(self.vertex_refs.required_len(), 0)?;
        for (name, p) in &self.channel_data {
            if let Some(i) = mesh.channel_index(name) {
                mesh.channels[i].data.grow_to(p.required_len(), 0.0)?;
            }
        }
        mesh.grow_channels()?;

        // From here on nothing can fail.
        self.positions.apply(&mut mesh.positions);
        self.triangles.apply(&mut mesh.triangles);
        self.vertex_live.apply(&mut mesh.vertex_live);
        self.triangle_live.apply(&mut mesh.triangle_live);
        self.vertex_refs.apply(&mut mesh.vertex_refs);
        for (name, p) in &self.channel_data {
            if let Some(i) = mesh.channel_index(name) {
                p.apply(&mut mesh.channels[i].data);
            }
        }
        let c = &self.changes;
        mesh.live_vertices = mesh.live_vertices + c.vertices_added.len() - c.vertices_removed.len();
        mesh.live_triangles =
            mesh.live_triangles + c.triangles_added.len() - c.triangles_removed.len();
        mesh.revision = self.revision;
        if c.topology_changed() {
            mesh.on_topology_changed();
        }
        Ok(self.changes.clone())
    }

    /// Undoes the patch. The mesh must be at [`Patch::revision`].
    pub fn revert(&self, mesh: &mut Mesh) -> Result<ChangeSet> {
        self.check(mesh, self.revision)?;
        self.positions.revert(&mut mesh.positions);
        self.triangles.revert(&mut mesh.triangles);
        self.vertex_live.revert(&mut mesh.vertex_live);
        self.triangle_live.revert(&mut mesh.triangle_live);
        self.vertex_refs.revert(&mut mesh.vertex_refs);
        for (name, p) in &self.channel_data {
            if let Some(i) = mesh.channel_index(name) {
                p.revert(&mut mesh.channels[i].data);
            }
        }
        let c = &self.changes;
        mesh.live_vertices = mesh.live_vertices + c.vertices_removed.len() - c.vertices_added.len();
        mesh.live_triangles =
            mesh.live_triangles + c.triangles_removed.len() - c.triangles_added.len();
        mesh.revision = self.base_revision;
        if c.topology_changed() {
            mesh.on_topology_changed();
        }
        Ok(self.changes.clone().inverted())
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
