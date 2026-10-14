use crate::column::{Column, try_vec};
use crate::mesh::full_bits;
use crate::{Mesh, Result, StoreError, TriangleId, VertexId};

/// Old slot id → new id mapping produced by [`Mesh::compact`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Remap {
    /// Indexed by old vertex slot; [`Remap::REMOVED`] for dead slots.
    pub vertices: Vec<VertexId>,
    /// Indexed by old triangle slot; [`Remap::REMOVED`] for dead slots.
    pub triangles: Vec<TriangleId>,
}

impl Remap {
    pub const REMOVED: u32 = u32::MAX;

    pub fn vertex(&self, old: VertexId) -> Option<VertexId> {
        self.vertices
            .get(old as usize)
            .copied()
            .filter(|&v| v != Self::REMOVED)
    }

    pub fn triangle(&self, old: TriangleId) -> Option<TriangleId> {
        self.triangles
            .get(old as usize)
            .copied()
            .filter(|&t| t != Self::REMOVED)
    }
}

/// Appends with a fallible reservation, so allocation failure is an error, not an abort.
fn try_push<T>(v: &mut Vec<T>, x: T) -> Result<()> {
    v.try_reserve(1).map_err(|_| StoreError::OutOfMemory)?;
    v.push(x);
    Ok(())
}

impl Mesh {
    /// Removes dead slots and renumbers alive vertices and triangles in their current
    /// order. All existing patches become stale (`StalePatch`); use
    /// [`crate::History::compact`] to clear a history at the same time.
    ///
    /// Failure-atomic: if memory runs out, the mesh is left unchanged.
    pub fn compact(&mut self) -> Result<Remap> {
        // Phase 1 (fallible): build every new column as a local. `self` is not written.
        let mut vmap = try_vec(self.vertex_slots(), Remap::REMOVED)?;
        let mut alive_v: Vec<usize> = Vec::new();
        for v in 0..self.vertex_slots() as u32 {
            if self.is_vertex_alive(v) {
                vmap[v as usize] = alive_v.len() as u32;
                try_push(&mut alive_v, v as usize)?;
            }
        }
        let mut tmap = try_vec(self.triangle_slots(), Remap::REMOVED)?;
        let mut tris: Vec<[u32; 3]> = Vec::new();
        for t in 0..self.triangle_slots() as u32 {
            if let Some(tri) = self.triangle(t) {
                tmap[t as usize] = tris.len() as u32;
                try_push(&mut tris, tri.map(|v| vmap[v as usize]))?;
            }
        }

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut refs: Vec<u32> = Vec::new();
        for &v in &alive_v {
            try_push(&mut positions, self.positions.get(v))?;
            try_push(&mut refs, self.vertex_refs.get(v))?;
        }
        let mut channel_data: Vec<Vec<f32>> = Vec::with_capacity(self.channels.len());
        for c in &self.channels {
            let w = c.desc.width;
            let data = c.data.as_slice();
            let mut compacted: Vec<f32> = Vec::new();
            compacted
                .try_reserve_exact(alive_v.len() * w)
                .map_err(|_| StoreError::OutOfMemory)?;
            for &v in &alive_v {
                for &x in &data[v * w..(v + 1) * w] {
                    try_push(&mut compacted, x)?;
                }
            }
            channel_data.push(compacted);
        }
        let vertex_live = full_bits(alive_v.len())?;
        let triangle_live = full_bits(tris.len())?;

        // Phase 2 (infallible): install the new columns and bump the revision.
        for (c, data) in self.channels.iter_mut().zip(channel_data) {
            c.data = Column::from_vec(data);
        }
        self.vertex_live = Column::from_vec(vertex_live);
        self.triangle_live = Column::from_vec(triangle_live);
        self.positions = Column::from_vec(positions);
        self.vertex_refs = Column::from_vec(refs);
        self.triangles = Column::from_vec(tris);
        self.live_vertices = alive_v.len();
        self.live_triangles = self.triangles.len();
        self.vertex_free.clear();
        self.triangle_free.clear();
        self.epoch += 1;
        self.next_revision += 1;
        self.revision = self.next_revision;
        self.on_topology_changed();
        Ok(Remap {
            vertices: vmap,
            triangles: tmap,
        })
    }
}
