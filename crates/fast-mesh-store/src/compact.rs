use crate::column::{Column, try_vec};
use crate::mesh::full_bits;
use crate::{Mesh, Result, TriangleId, VertexId};

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

impl Mesh {
    /// Removes dead slots and renumbers alive vertices and triangles in their current
    /// order. All existing patches become stale (`StalePatch`); use
    /// [`crate::History::compact`] to clear a history at the same time.
    pub fn compact(&mut self) -> Result<Remap> {
        let mut vmap = try_vec(self.vertex_slots(), Remap::REMOVED)?;
        let mut alive_v = Vec::with_capacity(self.vertex_count());
        for v in 0..self.vertex_slots() as u32 {
            if self.is_vertex_alive(v) {
                vmap[v as usize] = alive_v.len() as u32;
                alive_v.push(v as usize);
            }
        }
        let mut tmap = try_vec(self.triangle_slots(), Remap::REMOVED)?;
        let mut tris = Vec::with_capacity(self.triangle_count());
        for t in 0..self.triangle_slots() as u32 {
            if let Some(tri) = self.triangle(t) {
                tmap[t as usize] = tris.len() as u32;
                tris.push(tri.map(|v| vmap[v as usize]));
            }
        }

        let positions = alive_v.iter().map(|&v| self.positions.get(v)).collect();
        let refs = alive_v.iter().map(|&v| self.vertex_refs.get(v)).collect();
        for c in &mut self.channels {
            let w = c.desc.width;
            let data = c.data.as_slice();
            let compacted = alive_v
                .iter()
                .flat_map(|&v| data[v * w..(v + 1) * w].iter().copied())
                .collect();
            c.data = Column::from_vec(compacted);
        }
        self.vertex_live = Column::from_vec(full_bits(alive_v.len())?);
        self.triangle_live = Column::from_vec(full_bits(tris.len())?);
        self.positions = Column::from_vec(positions);
        self.vertex_refs = Column::from_vec(refs);
        self.triangles = Column::from_vec(tris);
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
