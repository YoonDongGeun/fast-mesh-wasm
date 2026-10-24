use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::adjacency::Adjacency;
use crate::channel::{Channel, ChannelDesc, ChannelKind};
use crate::column::{Column, try_reserve, try_vec};
use crate::incidence::Incidence;
use crate::{Patch, PatchSide, PatchState, Result, StoreError, TriangleId, VertexId};

static NEXT_MESH_UID: AtomicU64 = AtomicU64::new(1);

/// An editable triangle mesh stored in slots.
///
/// Vertices and triangles are addressed by slot ids that stay stable across edits,
/// undo and redo. Removed slots are dead: `positions()` and `triangles()` still contain
/// them (removed triangles read `[0, 0, 0]`, which renderers draw as nothing), and
/// `is_vertex_alive` / `is_triangle_alive` tell them apart. Only [`Mesh::compact`]
/// renumbers slots.
pub struct Mesh {
    pub(crate) uid: u64,
    pub(crate) epoch: u32,
    pub(crate) revision: u64,
    pub(crate) next_revision: u64,
    pub(crate) origin: [f64; 3],
    pub(crate) positions: Column<[f32; 3]>,
    pub(crate) triangles: Column<[u32; 3]>,
    pub(crate) vertex_live: Column<u64>,
    pub(crate) triangle_live: Column<u64>,
    /// Number of triangle corners referencing each vertex.
    pub(crate) vertex_refs: Column<u32>,
    pub(crate) channels: Vec<Channel>,
    /// Id for the next added channel.
    pub(crate) next_channel_id: u64,
    pub(crate) vertex_free: Vec<VertexId>,
    pub(crate) triangle_free: Vec<TriangleId>,
    pub(crate) live_vertices: usize,
    pub(crate) live_triangles: usize,
    /// Cached corner table; dropped on every topology change.
    pub(crate) adjacency: Option<Adjacency>,
    /// Cached vertex → triangles index; follows edits, dropped by `compact`.
    pub(crate) incidence: Option<Incidence>,
    /// True inside a transaction that changed topology: `incidence` is stale
    /// until the commit updates it or the rollback restores the mesh.
    pub(crate) incidence_pending: bool,
}

pub(crate) fn bit(bits: &Column<u64>, i: usize) -> bool {
    let word = i / 64;
    word < bits.len() && (bits.get(word) >> (i % 64)) & 1 == 1
}

pub(crate) fn set_bit(bits: &mut Column<u64>, i: usize, on: bool) -> Result<()> {
    let word = i / 64;
    bits.grow_to(word + 1, 0)?;
    let old = bits.get(word);
    let mask = 1u64 << (i % 64);
    let new = if on { old | mask } else { old & !mask };
    if new != old {
        bits.set(word, new);
    }
    Ok(())
}

pub(crate) fn full_bits(len: usize) -> Result<Vec<u64>> {
    let mut words = try_vec(len.div_ceil(64), u64::MAX)?;
    if let Some(last) = words.last_mut()
        && !len.is_multiple_of(64)
    {
        *last = (1u64 << (len % 64)) - 1;
    }
    Ok(words)
}

/// Shows ids and counts, not the (potentially huge) buffers.
impl fmt::Debug for Mesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mesh")
            .field("uid", &self.uid)
            .field("revision", &self.revision)
            .field("origin", &self.origin)
            .field("vertices", &self.live_vertices)
            .field("vertex_slots", &self.vertex_slots())
            .field("triangles", &self.live_triangles)
            .field("triangle_slots", &self.triangle_slots())
            .field(
                "channels",
                &self
                    .channels
                    .iter()
                    .map(|c| c.desc.name.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

impl Mesh {
    /// An empty mesh with its origin at zero.
    pub fn new() -> Self {
        // Empty liveness bitsets need no allocation, so this cannot fail.
        Self::assemble(
            [0.0; 3],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    /// Builds a mesh from flat `[x, y, z, ...]` positions and `[a, b, c, ...]` indices.
    pub fn from_buffers(positions: &[f32], indices: &[u32]) -> Result<Self> {
        if !positions.len().is_multiple_of(3) {
            return Err(StoreError::InvalidArgument(
                "positions length must be a multiple of 3",
            ));
        }
        let mut points = try_vec(positions.len() / 3, [0.0f32; 3])?;
        for (p, c) in points.iter_mut().zip(positions.as_chunks::<3>().0) {
            *p = *c;
        }
        Self::build([0.0; 3], points, indices)
    }

    /// Builds a mesh from double-precision positions. The origin is set to the center
    /// of the bounding box and positions are stored as `f32` offsets from it, which
    /// keeps precision for coordinates far from zero. Every coordinate, the
    /// computed origin and every offset (as an `f32`) must be finite.
    pub fn from_buffers_f64(positions: &[f64], indices: &[u32]) -> Result<Self> {
        if !positions.len().is_multiple_of(3) {
            return Err(StoreError::InvalidArgument(
                "positions length must be a multiple of 3",
            ));
        }
        let not_finite = StoreError::InvalidArgument("positions must be finite");
        if !positions.iter().all(|x| x.is_finite()) {
            return Err(not_finite);
        }
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for c in positions.as_chunks::<3>().0 {
            for k in 0..3 {
                min[k] = min[k].min(c[k]);
                max[k] = max[k].max(c[k]);
            }
        }
        let origin = if positions.is_empty() {
            [0.0; 3]
        } else {
            [0, 1, 2].map(|k| (min[k] + max[k]) * 0.5)
        };
        Self::with_origin_checked(positions, indices, origin)
    }

    /// Builds a mesh from double-precision positions relative to a given origin:
    /// positions are stored as `(p - origin)` in `f32`. Every coordinate, the
    /// origin and every offset (as an `f32`) must be finite.
    pub fn from_buffers_f64_with_origin(
        positions: &[f64],
        indices: &[u32],
        origin: [f64; 3],
    ) -> Result<Self> {
        if !positions.len().is_multiple_of(3) {
            return Err(StoreError::InvalidArgument(
                "positions length must be a multiple of 3",
            ));
        }
        if !positions.iter().all(|x| x.is_finite()) {
            return Err(StoreError::InvalidArgument("positions must be finite"));
        }
        Self::with_origin_checked(positions, indices, origin)
    }

    /// Shared tail of the `f64` constructors; `positions` is already validated.
    /// Offsets from the origin too large for an `f32` are rejected like
    /// non-finite input.
    fn with_origin_checked(positions: &[f64], indices: &[u32], origin: [f64; 3]) -> Result<Self> {
        let not_finite = StoreError::InvalidArgument("positions must be finite");
        if !origin.iter().all(|x| x.is_finite()) {
            return Err(not_finite);
        }
        let mut points = try_vec(positions.len() / 3, [0.0f32; 3])?;
        for (p, c) in points.iter_mut().zip(positions.as_chunks::<3>().0) {
            *p = [0, 1, 2].map(|k| (c[k] - origin[k]) as f32);
            if !p.iter().all(|x| x.is_finite()) {
                return Err(not_finite);
            }
        }
        Self::build(origin, points, indices)
    }

    fn build(origin: [f64; 3], points: Vec<[f32; 3]>, indices: &[u32]) -> Result<Self> {
        if !indices.len().is_multiple_of(3) {
            return Err(StoreError::InvalidArgument(
                "indices length must be a multiple of 3",
            ));
        }
        let vertex_count = points.len();
        let mut refs = try_vec(vertex_count, 0u32)?;
        let mut tris = try_vec(indices.len() / 3, [0u32; 3])?;
        for (t, c) in tris.iter_mut().zip(indices.as_chunks::<3>().0) {
            for &v in c {
                if v as usize >= vertex_count {
                    return Err(StoreError::InvalidVertex(v));
                }
                refs[v as usize] += 1;
            }
            *t = *c;
        }
        Self::from_parts(origin, points, tris, refs)
    }

    /// A mesh whose vertices and triangles are all alive.
    fn from_parts(
        origin: [f64; 3],
        points: Vec<[f32; 3]>,
        tris: Vec<[u32; 3]>,
        refs: Vec<u32>,
    ) -> Result<Self> {
        let vertex_live = full_bits(points.len())?;
        let triangle_live = full_bits(tris.len())?;
        Ok(Self::assemble(
            origin,
            points,
            tris,
            refs,
            vertex_live,
            triangle_live,
        ))
    }

    fn assemble(
        origin: [f64; 3],
        points: Vec<[f32; 3]>,
        tris: Vec<[u32; 3]>,
        refs: Vec<u32>,
        vertex_live: Vec<u64>,
        triangle_live: Vec<u64>,
    ) -> Self {
        let (v, t) = (points.len(), tris.len());
        Self {
            uid: NEXT_MESH_UID.fetch_add(1, Ordering::Relaxed),
            epoch: 0,
            revision: 0,
            next_revision: 0,
            origin,
            positions: Column::from_vec(points),
            triangles: Column::from_vec(tris),
            vertex_live: Column::from_vec(vertex_live),
            triangle_live: Column::from_vec(triangle_live),
            vertex_refs: Column::from_vec(refs),
            channels: Vec::new(),
            next_channel_id: 0,
            vertex_free: Vec::new(),
            triangle_free: Vec::new(),
            live_vertices: v,
            live_triangles: t,
            adjacency: None,
            incidence: None,
            incidence_pending: false,
        }
    }

    /// Process-unique id of this mesh. Patches only apply to the mesh that created them.
    pub fn uid(&self) -> u64 {
        self.uid
    }

    /// Changes on every commit, undo and redo. Equal revisions mean equal content.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Changes whenever [`Mesh::compact`] renumbers slots. Data indexed by slot id
    /// (such as an acceleration structure) is stale once the epoch differs.
    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    /// World-space origin that positions are relative to.
    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }

    /// Number of vertex slots, alive or dead. Render buffers use this length.
    pub fn vertex_slots(&self) -> usize {
        self.positions.len()
    }

    /// Number of triangle slots, alive or dead.
    pub fn triangle_slots(&self) -> usize {
        self.triangles.len()
    }

    /// Number of alive vertices.
    pub fn vertex_count(&self) -> usize {
        self.live_vertices
    }

    /// Number of alive triangles.
    pub fn triangle_count(&self) -> usize {
        self.live_triangles
    }

    pub fn is_vertex_alive(&self, v: VertexId) -> bool {
        bit(&self.vertex_live, v as usize)
    }

    pub fn is_triangle_alive(&self, t: TriangleId) -> bool {
        bit(&self.triangle_live, t as usize)
    }

    /// Position relative to [`Mesh::origin`], or `None` for a dead or missing vertex.
    pub fn position(&self, v: VertexId) -> Option<[f32; 3]> {
        self.is_vertex_alive(v)
            .then(|| self.positions.get(v as usize))
    }

    /// Position in world space (origin added in `f64`).
    pub fn world_position(&self, v: VertexId) -> Option<[f64; 3]> {
        self.position(v)
            .map(|p| [0, 1, 2].map(|k| self.origin[k] + p[k] as f64))
    }

    /// Slot-aligned positions, including dead slots.
    pub fn positions(&self) -> &[[f32; 3]] {
        self.positions.as_slice()
    }

    /// Vertex ids of a triangle, or `None` for a dead or missing triangle.
    pub fn triangle(&self, t: TriangleId) -> Option<[VertexId; 3]> {
        self.is_triangle_alive(t)
            .then(|| self.triangles.get(t as usize))
    }

    /// Slot-aligned triangles; dead slots read `[0, 0, 0]`.
    pub fn triangles(&self) -> &[[VertexId; 3]] {
        self.triangles.as_slice()
    }

    /// How many triangle corners use vertex `v` (a vertex repeated in one triangle counts twice).
    pub fn corner_count(&self, v: VertexId) -> u32 {
        if (v as usize) < self.vertex_refs.len() {
            self.vertex_refs.get(v as usize)
        } else {
            0
        }
    }

    /// Dead vertex slots ready for reuse.
    pub fn free_vertex_slots(&self) -> &[VertexId] {
        &self.vertex_free
    }

    /// Dead triangle slots ready for reuse.
    pub fn free_triangle_slots(&self) -> &[TriangleId] {
        &self.triangle_free
    }

    /// Adds a channel filled with its default value. Adding a channel is not undoable.
    pub fn add_channel(&mut self, desc: ChannelDesc) -> Result<()> {
        if !(1..=4).contains(&desc.width) {
            return Err(StoreError::InvalidArgument("channel width must be 1 to 4"));
        }
        if self.channel_index(&desc.name).is_some() {
            return Err(StoreError::ChannelExists(desc.name));
        }
        let data = try_vec(self.vertex_slots() * desc.width, desc.default)?;
        try_reserve(&mut self.channels, 1)?;
        let id = self.next_channel_id;
        self.next_channel_id += 1;
        self.channels.push(Channel {
            id,
            desc,
            data: Column::from_vec(data),
        });
        Ok(())
    }

    /// Removes a channel. Not undoable; patches that recorded it skip it from now on.
    pub fn remove_channel(&mut self, name: &str) -> Result<ChannelDesc> {
        let i = self
            .channel_index(name)
            .ok_or_else(|| StoreError::UnknownChannel(name.into()))?;
        Ok(self.channels.remove(i).desc)
    }

    pub fn channel_desc(&self, name: &str) -> Option<&ChannelDesc> {
        self.channel_index(name).map(|i| &self.channels[i].desc)
    }

    /// Slot-aligned values, `width` per vertex.
    pub fn channel(&self, name: &str) -> Option<&[f32]> {
        self.channel_index(name)
            .map(|i| self.channels[i].data.as_slice())
    }

    /// Writable values of a derived channel. Writes are not recorded for undo.
    pub fn derived_channel_mut(&mut self, name: &str) -> Result<&mut [f32]> {
        let i = self
            .channel_index(name)
            .ok_or_else(|| StoreError::UnknownChannel(name.into()))?;
        if self.channels[i].desc.kind != ChannelKind::Derived {
            return Err(StoreError::NotDerivedChannel(name.into()));
        }
        Ok(self.channels[i].data.as_mut_slice())
    }

    pub(crate) fn channel_index(&self, name: &str) -> Option<usize> {
        self.channels.iter().position(|c| c.desc.name == name)
    }

    /// Index of the channel with this id, or `None` if it was removed.
    pub(crate) fn channel_index_by_id(&self, id: u64) -> Option<usize> {
        self.channels.iter().position(|c| c.id == id)
    }

    /// Puts dead vertex slots on the free list. Ids that are alive or out of range
    /// are skipped: they are not free slots of this mesh.
    pub(crate) fn release_vertices(&mut self, ids: &[VertexId]) {
        for &v in ids {
            if (v as usize) < self.vertex_slots() && !self.is_vertex_alive(v) {
                self.vertex_free.push(v);
            }
        }
    }

    /// Puts dead triangle slots on the free list, skipping alive or out-of-range ids.
    pub(crate) fn release_triangles(&mut self, ids: &[TriangleId]) {
        for &t in ids {
            if (t as usize) < self.triangle_slots() && !self.is_triangle_alive(t) {
                self.triangle_free.push(t);
            }
        }
    }

    /// Returns the slots held by a [`Patch`] kept outside [`crate::History`] —
    /// for example on an application's own undo stack or a scene-wide history —
    /// to the free lists, applying the same rules the history uses internally:
    /// [`PatchState::Applied`] releases the slots the patch removed,
    /// [`PatchState::Undone`] the slots it created.
    ///
    /// Unlike [`Patch::release`], a patch from another mesh or from before a
    /// [`Mesh::compact`] is an error ([`StoreError::PatchMismatch`] and
    /// [`StoreError::StalePatch`], respectively), not a silent skip. Slots that
    /// are alive — reused since the patch was made — are skipped.
    ///
    /// Release each patch at most once, and only once no history still holds it:
    /// releasing twice, or releasing while a history still holds the patch,
    /// lists a still-dead slot twice and corrupts slot reuse.
    pub fn release_patch_slots(&mut self, patch: &Patch, state: PatchState) -> Result<()> {
        if patch.mesh_uid != self.uid {
            return Err(StoreError::PatchMismatch);
        }
        if patch.epoch != self.epoch {
            return Err(StoreError::StalePatch);
        }
        let side = match state {
            PatchState::Applied => PatchSide::Undoable,
            PatchState::Undone => PatchSide::Redoable,
        };
        patch.release(self, side);
        Ok(())
    }

    /// Called by every topology write inside a transaction.
    pub(crate) fn on_topology_write(&mut self) {
        self.adjacency = None;
        self.incidence_pending = true;
    }

    /// Called when slots are renumbered: every topology cache is dropped.
    pub(crate) fn on_slots_renumbered(&mut self) {
        self.adjacency = None;
        self.incidence = None;
        self.incidence_pending = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PatchState;

    fn quad() -> Mesh {
        Mesh::from_buffers(
            &[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap()
    }

    #[test]
    fn builds_from_flat_buffers() {
        let m = quad();
        assert_eq!((m.vertex_count(), m.triangle_count()), (4, 2));
        assert_eq!(m.position(2), Some([1., 1., 0.]));
        assert_eq!(m.triangle(1), Some([0, 2, 3]));
        assert_eq!(m.corner_count(0), 2);
        assert_eq!(m.corner_count(1), 1);
    }

    #[test]
    fn rejects_bad_buffers() {
        assert!(matches!(
            Mesh::from_buffers(&[0., 0.], &[]),
            Err(StoreError::InvalidArgument(_))
        ));
        assert_eq!(
            Mesh::from_buffers(&[0., 0., 0.], &[0, 0, 5]).err(),
            Some(StoreError::InvalidVertex(5))
        );
    }

    #[test]
    fn f64_input_keeps_precision_through_the_origin() {
        let far = 1.0e7;
        let m = Mesh::from_buffers_f64(&[far, far, 0.0, far + 0.001, far, 0.0], &[]).unwrap();
        let o = m.origin();
        assert!((o[0] - (far + 0.0005)).abs() < 1e-6 && o[1] == far && o[2] == 0.0);
        let p = m.world_position(1).unwrap();
        assert!((p[0] - (far + 0.001)).abs() < 1e-6, "got {}", p[0]);
    }

    #[test]
    fn f64_input_must_be_finite() {
        let bad = StoreError::InvalidArgument("positions must be finite");
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                Mesh::from_buffers_f64(&[0., 0., 0., x, 1., 2.], &[]).err(),
                Some(bad.clone())
            );
        }
        // Finite input whose bounding-box center overflows.
        assert_eq!(
            Mesh::from_buffers_f64(&[f64::MAX, 0., 0., f64::MAX, 0., 0.], &[]).err(),
            Some(bad)
        );
    }

    #[test]
    fn explicit_origin_stores_offsets() {
        let o = [1.0e6; 3];
        let offsets = [[0.25, -0.5, 1.0], [-3.0, 2.125, 0.0], [10.0, 0.0, -7.5]];
        let flat: Vec<f64> = offsets
            .iter()
            .flat_map(|d| [0, 1, 2].map(|k| o[k] + d[k]))
            .collect();
        let m = Mesh::from_buffers_f64_with_origin(&flat, &[0, 1, 2], o).unwrap();
        assert_eq!(m.origin(), o);
        for (v, d) in offsets.iter().enumerate() {
            let p = m.position(v as u32).unwrap();
            for k in 0..3 {
                assert!((p[k] as f64 - d[k]).abs() < 1e-3, "{v}: {p:?} vs {d:?}");
            }
        }
        let bad = StoreError::InvalidArgument("positions must be finite");
        assert_eq!(
            Mesh::from_buffers_f64_with_origin(&flat, &[], [f64::NAN, 0., 0.]).err(),
            Some(bad.clone())
        );
        assert_eq!(
            Mesh::from_buffers_f64_with_origin(&[f64::INFINITY, 0., 0.], &[], [0.; 3]).err(),
            Some(bad)
        );
        assert!(matches!(
            Mesh::from_buffers_f64_with_origin(&[0., 0.], &[], [0.; 3]),
            Err(StoreError::InvalidArgument(_))
        ));
    }

    #[test]
    fn offsets_overflowing_f32_are_rejected() {
        let bad = StoreError::InvalidArgument("positions must be finite");
        // Finite in f64, but 1e39 away from the origin does not fit an f32.
        assert_eq!(
            Mesh::from_buffers_f64_with_origin(&[1.0e39, 0., 0.], &[], [0.; 3]).err(),
            Some(bad.clone())
        );
        assert_eq!(
            Mesh::from_buffers_f64_with_origin(&[0., 0., 0.], &[], [0., -1.0e39, 0.]).err(),
            Some(bad.clone())
        );
        // The auto-centred constructor shares the check.
        assert_eq!(
            Mesh::from_buffers_f64(&[-1.0e300, 0., 0., 1.0e300, 0., 0.], &[]).err(),
            Some(bad)
        );
    }

    #[test]
    fn out_of_range_ids_read_as_dead() {
        let m = quad();
        assert!(!m.is_vertex_alive(99));
        assert_eq!(m.position(99), None);
        assert_eq!(m.triangle(99), None);
    }

    #[test]
    fn channels_are_slot_aligned_and_unique() {
        let mut m = quad();
        m.add_channel(ChannelDesc::source("mask", 1).with_default(0.5))
            .unwrap();
        assert_eq!(m.channel("mask"), Some(&[0.5; 4][..]));
        assert_eq!(
            m.add_channel(ChannelDesc::source("mask", 1)),
            Err(StoreError::ChannelExists("mask".into()))
        );
        assert!(matches!(
            m.add_channel(ChannelDesc::source("wide", 5)),
            Err(StoreError::InvalidArgument(_))
        ));
        assert_eq!(
            m.derived_channel_mut("mask").err(),
            Some(StoreError::NotDerivedChannel("mask".into()))
        );
        m.add_channel(ChannelDesc::derived("dist", 1)).unwrap();
        m.derived_channel_mut("dist").unwrap()[3] = 7.0;
        assert_eq!(m.channel("dist").unwrap()[3], 7.0);
        assert_eq!(m.remove_channel("dist").unwrap().name, "dist");
        assert_eq!(m.channel("dist"), None);
    }

    #[test]
    fn debug_output_shows_counts() {
        let mut m = quad();
        m.ensure_adjacency().unwrap();
        let s = format!("{m:?}");
        assert!(
            s.contains("vertices: 4") && s.contains("triangles: 2"),
            "{s}"
        );
        assert!(format!("{:?}", m.adjacency().unwrap()).starts_with("Adjacency"));
    }

    #[test]
    fn every_mesh_gets_a_distinct_uid() {
        assert_ne!(Mesh::new().uid(), Mesh::new().uid());
    }

    #[test]
    fn release_patch_slots_follows_history_rules() {
        let mut m = quad();
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[1]).unwrap();
        tx.remove_vertices(&[3]).unwrap();
        let removal = tx.commit().unwrap();
        assert!(m.free_vertex_slots().is_empty(), "reserved by the patch");
        // The wrong state releases nothing.
        m.release_patch_slots(&removal, PatchState::Undone).unwrap();
        assert!(m.free_vertex_slots().is_empty());
        m.release_patch_slots(&removal, PatchState::Applied)
            .unwrap();
        assert_eq!(m.free_vertex_slots(), &[3]);
        assert_eq!(m.free_triangle_slots(), &[1]);

        let mut tx = m.begin_edit();
        tx.add_vertices(&[[0.; 3]]).unwrap();
        let creation = tx.commit().unwrap();
        creation.revert(&mut m).unwrap();
        assert!(m.free_vertex_slots().is_empty(), "reserved for redo");
        m.release_patch_slots(&creation, PatchState::Applied)
            .unwrap();
        assert!(m.free_vertex_slots().is_empty());
        m.release_patch_slots(&creation, PatchState::Undone)
            .unwrap();
        assert_eq!(m.free_vertex_slots(), &[3]);
    }

    #[test]
    fn release_patch_slots_rejects_a_foreign_mesh() {
        let mut m = quad();
        let mut other = quad();
        let mut tx = other.begin_edit();
        tx.remove_triangles(&[1]).unwrap();
        tx.remove_vertices(&[3]).unwrap();
        let patch = tx.commit().unwrap();
        assert_eq!(
            m.release_patch_slots(&patch, PatchState::Applied).err(),
            Some(StoreError::PatchMismatch)
        );
        assert!(m.free_vertex_slots().is_empty());
        assert!(m.free_triangle_slots().is_empty());
    }

    #[test]
    fn release_patch_slots_rejects_an_earlier_epoch() {
        let mut m = quad();
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[1]).unwrap();
        tx.remove_vertices(&[3]).unwrap();
        let patch = tx.commit().unwrap();
        m.compact().unwrap();
        assert_eq!(
            m.release_patch_slots(&patch, PatchState::Applied).err(),
            Some(StoreError::StalePatch)
        );
        assert!(m.free_vertex_slots().is_empty());
        assert!(m.free_triangle_slots().is_empty());
    }

    #[test]
    fn release_patch_slots_skips_alive_slots() {
        let mut m = quad();
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[1]).unwrap();
        tx.remove_vertices(&[3]).unwrap();
        let patch = tx.commit().unwrap();
        m.release_patch_slots(&patch, PatchState::Applied).unwrap();
        assert_eq!(m.free_vertex_slots(), &[3]);
        // Reuse the slots, then release again: the alive slots are skipped, so
        // a second release after reuse is a safe no-op.
        let mut tx = m.begin_edit();
        assert_eq!(tx.add_vertices(&[[0.; 3]]).unwrap(), vec![3]);
        assert_eq!(tx.add_triangles(&[[0, 1, 2]]).unwrap(), vec![1]);
        tx.commit().unwrap();
        m.release_patch_slots(&patch, PatchState::Applied).unwrap();
        assert!(m.free_vertex_slots().is_empty());
        assert!(m.free_triangle_slots().is_empty());
        let mut tx = m.begin_edit();
        assert_eq!(tx.add_vertices(&[[0.; 3]]).unwrap(), vec![4]);
        assert_eq!(tx.add_triangles(&[[0, 1, 2]]).unwrap(), vec![2]);
        tx.rollback();
    }

    #[test]
    fn release_patch_slots_with_history_holding_the_patch() {
        use crate::History;
        let mut m = quad();
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[1]).unwrap();
        tx.remove_vertices(&[3]).unwrap();
        let patch = tx.commit().unwrap();
        let mut h = History::new(1 << 20);
        h.push(&mut m, patch.clone()).unwrap();
        // An external undo stack drops its copy first; the slots are reused, so
        // when the history later releases the same patch its slots are alive
        // and skipped: no duplicate free-list entries.
        m.release_patch_slots(&patch, PatchState::Applied).unwrap();
        let mut tx = m.begin_edit();
        assert_eq!(tx.add_vertices(&[[0.; 3]]).unwrap(), vec![3]);
        assert_eq!(tx.add_triangles(&[[0, 1, 2]]).unwrap(), vec![1]);
        tx.commit().unwrap();
        h.clear(&mut m).unwrap();
        assert!(m.free_vertex_slots().is_empty());
        assert!(m.free_triangle_slots().is_empty());
        let mut tx = m.begin_edit();
        assert_eq!(tx.add_vertices(&[[0.; 3]]).unwrap(), vec![4]);
        assert_eq!(tx.add_triangles(&[[0, 1, 2]]).unwrap(), vec![2]);
        tx.rollback();
    }
}
