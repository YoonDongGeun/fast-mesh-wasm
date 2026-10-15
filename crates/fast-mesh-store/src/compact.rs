use crate::column::{Column, try_reserve, try_reserve_exact, try_vec};
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

/// Appends with a fallible reservation, so allocation failure is an error, not an
/// abort. Callers reserve the final size up front; this then never reallocates.
fn try_push<T>(v: &mut Vec<T>, x: T) -> Result<()> {
    try_reserve(v, 1)?;
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
        try_reserve_exact(&mut alive_v, self.live_vertices)?;
        for v in 0..self.vertex_slots() as u32 {
            if self.is_vertex_alive(v) {
                vmap[v as usize] = alive_v.len() as u32;
                try_push(&mut alive_v, v as usize)?;
            }
        }
        let mut tmap = try_vec(self.triangle_slots(), Remap::REMOVED)?;
        let mut tris: Vec<[u32; 3]> = Vec::new();
        try_reserve_exact(&mut tris, self.live_triangles)?;
        for t in 0..self.triangle_slots() as u32 {
            if let Some(tri) = self.triangle(t) {
                tmap[t as usize] = tris.len() as u32;
                try_push(&mut tris, tri.map(|v| vmap[v as usize]))?;
            }
        }

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut refs: Vec<u32> = Vec::new();
        try_reserve_exact(&mut positions, alive_v.len())?;
        try_reserve_exact(&mut refs, alive_v.len())?;
        for &v in &alive_v {
            try_push(&mut positions, self.positions.get(v))?;
            try_push(&mut refs, self.vertex_refs.get(v))?;
        }
        let mut channel_data: Vec<Vec<f32>> = Vec::new();
        try_reserve_exact(&mut channel_data, self.channels.len())?;
        for c in &self.channels {
            let w = c.desc.width;
            let data = c.data.as_slice();
            let mut compacted: Vec<f32> = Vec::new();
            try_reserve_exact(&mut compacted, alive_v.len() * w)?;
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
        self.on_slots_renumbered();
        Ok(Remap {
            vertices: vmap,
            triangles: tmap,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{ChannelDesc, History, Mesh, StoreError};

    #[test]
    fn compact_renumbers_alive_elements_in_order() {
        let mut m = Mesh::from_buffers(
            &[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap();
        m.add_channel(ChannelDesc::source("mask", 1)).unwrap();
        let mut tx = m.begin_edit();
        tx.write_channel("mask", &[0, 1, 2, 3], &[0., 1., 2., 3.])
            .unwrap();
        tx.remove_triangles(&[0]).unwrap();
        tx.remove_vertices(&[1]).unwrap();
        let patch = tx.commit().unwrap();

        let epoch = m.epoch();
        let remap = m.compact().unwrap();
        assert_ne!(m.epoch(), epoch, "compact must change the epoch");
        assert_eq!(remap.vertices, vec![0, u32::MAX, 1, 2]);
        assert_eq!(remap.triangle(1), Some(0));
        assert_eq!(remap.triangle(0), None);
        assert_eq!((m.vertex_slots(), m.triangle_slots()), (3, 1));
        assert_eq!(m.triangle(0), Some([0, 1, 2]));
        assert_eq!(m.position(1), Some([1., 1., 0.]));
        assert_eq!(m.channel("mask").unwrap(), &[0., 2., 3.]);
        assert_eq!(m.corner_count(0), 1);
        assert_eq!(patch.revert(&mut m).err(), Some(StoreError::StalePatch));
    }

    #[test]
    fn compacted_columns_have_no_slack() {
        let positions = vec![0.0f32; 3 * 100];
        let indices: Vec<u32> = (0..98).flat_map(|i| [i, i + 1, i + 2]).collect();
        let mut m = Mesh::from_buffers(&positions, &indices).unwrap();
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[97]).unwrap();
        tx.remove_vertices(&[99]).unwrap();
        tx.commit().unwrap();
        m.compact().unwrap();
        assert_eq!((m.vertex_slots(), m.triangle_slots()), (99, 97));
        assert_eq!(m.positions.capacity(), 99);
        assert_eq!(m.vertex_refs.capacity(), 99);
        assert_eq!(m.triangles.capacity(), 97);
    }

    #[test]
    fn history_compact_clears_the_stacks() {
        let mut m = Mesh::from_buffers(&[0., 0., 0.], &[]).unwrap();
        let mut h = History::new(1 << 20);
        let mut tx = m.begin_edit();
        tx.remove_vertices(&[0]).unwrap();
        let p = tx.commit().unwrap();
        h.push(&mut m, p).unwrap();
        h.compact(&mut m).unwrap();
        assert!(!h.can_undo());
        assert_eq!(m.vertex_slots(), 0);
        assert!(m.free_vertex_slots().is_empty());
    }
}
