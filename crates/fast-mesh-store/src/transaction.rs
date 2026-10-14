use crate::channel::ChannelKind;
use crate::column::{Prepared, try_reserve_exact};
use crate::mesh::set_bit;
use crate::patch::{ChangeSet, Patch};
use crate::{Mesh, Result, StoreError, TriangleId, VertexId};

/// An open edit on a [`Mesh`]. All writes go through it.
///
/// Every method validates its whole input before writing, so a returned validation
/// error leaves the transaction unchanged and usable. If a write itself fails (out of
/// memory), the transaction is poisoned and `commit` rolls it back.
///
/// Dropping a transaction without committing rolls it back.
pub struct Transaction<'m> {
    mesh: &'m mut Mesh,
    allocated_vertices: Vec<VertexId>,
    removed_vertices: Vec<VertexId>,
    moved_vertices: Vec<VertexId>,
    allocated_triangles: Vec<TriangleId>,
    removed_triangles: Vec<TriangleId>,
    changed_triangles: Vec<TriangleId>,
    /// Slots taken from the free lists, in order, so rollback can return them.
    popped_vertices: Vec<VertexId>,
    popped_triangles: Vec<TriangleId>,
    /// Per channel (same order as `mesh.channels`): vertices written by `write_channel`.
    channel_writes: Vec<Vec<VertexId>>,
    live_at_start: (usize, usize),
    poisoned: bool,
    finished: bool,
}

impl Mesh {
    /// Opens a transaction. The mesh cannot be used otherwise until it ends.
    pub fn begin_edit(&mut self) -> Transaction<'_> {
        self.positions.begin();
        self.triangles.begin();
        self.vertex_live.begin();
        self.triangle_live.begin();
        self.vertex_refs.begin();
        for c in &mut self.channels {
            c.data.begin();
        }
        Transaction {
            channel_writes: vec![Vec::new(); self.channels.len()],
            live_at_start: (self.live_vertices, self.live_triangles),
            mesh: self,
            allocated_vertices: Vec::new(),
            removed_vertices: Vec::new(),
            moved_vertices: Vec::new(),
            allocated_triangles: Vec::new(),
            removed_triangles: Vec::new(),
            changed_triangles: Vec::new(),
            popped_vertices: Vec::new(),
            popped_triangles: Vec::new(),
            poisoned: false,
            finished: false,
        }
    }
}

fn sorted(mut ids: Vec<u32>) -> Vec<u32> {
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn without(ids: Vec<u32>, a: &[u32], b: &[u32]) -> Vec<u32> {
    ids.into_iter()
        .filter(|i| a.binary_search(i).is_err() && b.binary_search(i).is_err())
        .collect()
}

fn check_unique(ids: &[u32], what: &'static str) -> Result<()> {
    let mut s = ids.to_vec();
    s.sort_unstable();
    if s.windows(2).any(|w| w[0] == w[1]) {
        return Err(StoreError::InvalidArgument(what));
    }
    Ok(())
}

/// Patch vectors of every journaled column, built before any journal is consumed.
struct PreparedColumns {
    positions: Prepared<[f32; 3]>,
    triangles: Prepared<[u32; 3]>,
    vertex_live: Prepared<u64>,
    triangle_live: Prepared<u64>,
    vertex_refs: Prepared<u32>,
    /// Per channel, in `mesh.channels` order: `None` for derived channels.
    channels: Vec<Option<Prepared<f32>>>,
}

impl PreparedColumns {
    fn new(m: &Mesh) -> Result<Self> {
        let mut channels = Vec::new();
        try_reserve_exact(&mut channels, m.channels.len())?;
        for c in &m.channels {
            channels.push(match c.desc.kind {
                ChannelKind::Source => Some(c.data.prepare_finish()?),
                ChannelKind::Derived => None,
            });
        }
        Ok(Self {
            positions: m.positions.prepare_finish()?,
            triangles: m.triangles.prepare_finish()?,
            vertex_live: m.vertex_live.prepare_finish()?,
            triangle_live: m.triangle_live.prepare_finish()?,
            vertex_refs: m.vertex_refs.prepare_finish()?,
            channels,
        })
    }
}

impl Transaction<'_> {
    /// Read access to the mesh as edited so far.
    pub fn mesh(&self) -> &Mesh {
        self.mesh
    }

    fn check_vertex(&self, v: VertexId) -> Result<()> {
        if self.mesh.is_vertex_alive(v) {
            Ok(())
        } else {
            Err(StoreError::InvalidVertex(v))
        }
    }

    fn check_triangle(&self, t: TriangleId) -> Result<()> {
        if self.mesh.is_triangle_alive(t) {
            Ok(())
        } else {
            Err(StoreError::InvalidTriangle(t))
        }
    }

    /// Runs the write phase of an operation; any failure poisons the transaction.
    fn write<R>(&mut self, f: impl FnOnce(&mut Self) -> Result<R>) -> Result<R> {
        if self.poisoned {
            return Err(StoreError::TransactionPoisoned);
        }
        let r = f(self);
        if r.is_err() {
            self.poisoned = true;
        }
        r
    }

    fn alloc_vertex(
        &mut self,
        position: [f32; 3],
        parents: &[(VertexId, f32)],
    ) -> Result<VertexId> {
        let m = &mut *self.mesh;
        let id = match m.vertex_free.pop() {
            Some(id) => {
                self.popped_vertices.push(id);
                debug_assert!(!m.is_vertex_alive(id), "free list held alive vertex {id}");
                m.positions.set(id as usize, position);
                id
            }
            None => {
                let id = m.positions.push(position)?;
                m.vertex_refs.push(0)?;
                for c in &mut m.channels {
                    for _ in 0..c.desc.width {
                        c.data.push(c.desc.default)?;
                    }
                }
                u32::try_from(id).map_err(|_| StoreError::OutOfMemory)?
            }
        };
        for c in &mut m.channels {
            let w = c.desc.width;
            for k in 0..w {
                let value = c.initial_value(parents, k);
                c.data.set(id as usize * w + k, value);
            }
        }
        set_bit(&mut m.vertex_live, id as usize, true)?;
        m.live_vertices += 1;
        self.allocated_vertices.push(id);
        Ok(id)
    }

    fn alloc_triangle(&mut self, tri: [VertexId; 3]) -> Result<TriangleId> {
        let m = &mut *self.mesh;
        let id = match m.triangle_free.pop() {
            Some(id) => {
                self.popped_triangles.push(id);
                debug_assert!(
                    !m.is_triangle_alive(id),
                    "free list held alive triangle {id}"
                );
                m.triangles.set(id as usize, tri);
                id
            }
            None => {
                let id = m.triangles.push(tri)?;
                u32::try_from(id).map_err(|_| StoreError::OutOfMemory)?
            }
        };
        for v in tri {
            let r = m.vertex_refs.get(v as usize);
            m.vertex_refs.set(v as usize, r + 1);
        }
        set_bit(&mut m.triangle_live, id as usize, true)?;
        m.live_triangles += 1;
        self.allocated_triangles.push(id);
        Ok(id)
    }

    fn release_corners(&mut self, tri: [VertexId; 3]) {
        let m = &mut *self.mesh;
        for v in tri {
            let r = m.vertex_refs.get(v as usize);
            m.vertex_refs.set(v as usize, r - 1);
        }
    }

    /// Adds vertices without creation parents; source channels get their default value.
    pub fn add_vertices(&mut self, positions: &[[f32; 3]]) -> Result<Vec<VertexId>> {
        self.write(|tx| positions.iter().map(|&p| tx.alloc_vertex(p, &[])).collect())
    }

    /// Adds a vertex created from existing vertices, e.g. an edge split creates a vertex
    /// with parents `[(a, 0.5), (b, 0.5)]`. Source channels derive its values from the
    /// parents according to their [`crate::TopologyPolicy`].
    pub fn add_vertex_with_parents(
        &mut self,
        position: [f32; 3],
        parents: &[(VertexId, f32)],
    ) -> Result<VertexId> {
        for &(v, w) in parents {
            self.check_vertex(v)?;
            if !w.is_finite() || w < 0.0 {
                return Err(StoreError::InvalidArgument(
                    "parent weights must be finite and non-negative",
                ));
            }
        }
        self.write(|tx| tx.alloc_vertex(position, parents))
    }

    /// Sets positions of existing vertices.
    pub fn set_positions(&mut self, ids: &[VertexId], positions: &[[f32; 3]]) -> Result<()> {
        if ids.len() != positions.len() {
            return Err(StoreError::LengthMismatch {
                expected: ids.len(),
                got: positions.len(),
            });
        }
        for &v in ids {
            self.check_vertex(v)?;
        }
        self.write(|tx| {
            for (&v, &p) in ids.iter().zip(positions) {
                tx.mesh.positions.set(v as usize, p);
            }
            tx.moved_vertices.extend_from_slice(ids);
            Ok(())
        })
    }

    /// Removes vertices that no triangle uses.
    pub fn remove_vertices(&mut self, ids: &[VertexId]) -> Result<()> {
        check_unique(ids, "duplicate vertex id")?;
        for &v in ids {
            self.check_vertex(v)?;
            let corners = self.mesh.corner_count(v);
            if corners > 0 {
                return Err(StoreError::VertexInUse { vertex: v, corners });
            }
        }
        self.write(|tx| {
            for &v in ids {
                set_bit(&mut tx.mesh.vertex_live, v as usize, false)?;
                tx.mesh.live_vertices -= 1;
            }
            tx.removed_vertices.extend_from_slice(ids);
            Ok(())
        })
    }

    /// Adds triangles between existing vertices.
    pub fn add_triangles(&mut self, triangles: &[[VertexId; 3]]) -> Result<Vec<TriangleId>> {
        for t in triangles {
            for &v in t {
                self.check_vertex(v)?;
            }
        }
        self.write(|tx| triangles.iter().map(|&t| tx.alloc_triangle(t)).collect())
    }

    /// Removes triangles. Their vertices stay; remove them separately if unused.
    pub fn remove_triangles(&mut self, ids: &[TriangleId]) -> Result<()> {
        check_unique(ids, "duplicate triangle id")?;
        for &t in ids {
            self.check_triangle(t)?;
        }
        self.write(|tx| {
            for &t in ids {
                let tri = tx.mesh.triangles.get(t as usize);
                tx.release_corners(tri);
                tx.mesh.triangles.set(t as usize, [0; 3]);
                set_bit(&mut tx.mesh.triangle_live, t as usize, false)?;
                tx.mesh.live_triangles -= 1;
            }
            tx.removed_triangles.extend_from_slice(ids);
            Ok(())
        })
    }

    /// Replaces the vertex ids of existing triangles (e.g. an edge flip).
    pub fn set_triangles(&mut self, ids: &[TriangleId], triangles: &[[VertexId; 3]]) -> Result<()> {
        if ids.len() != triangles.len() {
            return Err(StoreError::LengthMismatch {
                expected: ids.len(),
                got: triangles.len(),
            });
        }
        check_unique(ids, "duplicate triangle id")?;
        for (&t, tri) in ids.iter().zip(triangles) {
            self.check_triangle(t)?;
            for &v in tri {
                self.check_vertex(v)?;
            }
        }
        self.write(|tx| {
            for (&t, &tri) in ids.iter().zip(triangles) {
                let old = tx.mesh.triangles.get(t as usize);
                tx.release_corners(old);
                for v in tri {
                    let r = tx.mesh.vertex_refs.get(v as usize);
                    tx.mesh.vertex_refs.set(v as usize, r + 1);
                }
                tx.mesh.triangles.set(t as usize, tri);
            }
            tx.changed_triangles.extend_from_slice(ids);
            Ok(())
        })
    }

    /// Writes source channel values: `values` holds `width` values per vertex in `ids`.
    pub fn write_channel(&mut self, name: &str, ids: &[VertexId], values: &[f32]) -> Result<()> {
        let ci = self
            .mesh
            .channel_index(name)
            .ok_or_else(|| StoreError::UnknownChannel(name.into()))?;
        let desc = &self.mesh.channels[ci].desc;
        if desc.kind != ChannelKind::Source {
            return Err(StoreError::NotSourceChannel(name.into()));
        }
        let w = desc.width;
        if values.len() != ids.len() * w {
            return Err(StoreError::LengthMismatch {
                expected: ids.len() * w,
                got: values.len(),
            });
        }
        for &v in ids {
            self.check_vertex(v)?;
        }
        self.write(|tx| {
            let data = &mut tx.mesh.channels[ci].data;
            for (n, &v) in ids.iter().enumerate() {
                for k in 0..w {
                    data.set(v as usize * w + k, values[n * w + k]);
                }
            }
            tx.channel_writes[ci].extend_from_slice(ids);
            Ok(())
        })
    }

    /// Ends the transaction and returns its patch. An empty transaction returns an
    /// empty patch and leaves the revision unchanged. If memory for the patch runs
    /// out, the transaction is rolled back and `OutOfMemory` is returned.
    pub fn commit(mut self) -> Result<Patch> {
        self.finished = true;
        if self.poisoned {
            self.rollback_inner();
            return Err(StoreError::TransactionPoisoned);
        }
        // Phase 1 (fallible, writes nothing): build every patch vector while all
        // journals are intact. If memory runs out, roll back as if never committed.
        let prepared = match PreparedColumns::new(self.mesh) {
            Ok(p) => p,
            Err(e) => {
                self.rollback_inner();
                return Err(e);
            }
        };
        // Phase 2 (infallible): consume the journals.
        let m = &mut *self.mesh;

        // Slots allocated and removed within this transaction were never visible:
        // they go straight back to the free lists.
        let allocated_v = sorted(std::mem::take(&mut self.allocated_vertices));
        let (created_v, transient_v): (Vec<_>, Vec<_>) = allocated_v
            .iter()
            .copied()
            .partition(|&v| m.is_vertex_alive(v));
        let allocated_t = sorted(std::mem::take(&mut self.allocated_triangles));
        let (created_t, transient_t): (Vec<_>, Vec<_>) = allocated_t
            .iter()
            .copied()
            .partition(|&t| m.is_triangle_alive(t));
        m.vertex_free.extend_from_slice(&transient_v);
        m.triangle_free.extend_from_slice(&transient_t);

        let freed_v = without(
            sorted(std::mem::take(&mut self.removed_vertices)),
            &allocated_v,
            &[],
        );
        let freed_t = without(
            sorted(std::mem::take(&mut self.removed_triangles)),
            &allocated_t,
            &[],
        );
        let moved = without(
            sorted(std::mem::take(&mut self.moved_vertices)),
            &allocated_v,
            &freed_v,
        );
        let changed_t = without(
            sorted(std::mem::take(&mut self.changed_triangles)),
            &allocated_t,
            &freed_t,
        );

        let mut channels = Vec::new();
        let mut channel_data = Vec::new();
        for ((ci, c), prepared) in m.channels.iter_mut().enumerate().zip(prepared.channels) {
            let Some(prepared) = prepared else {
                c.data.end_journal(); // derived: not undoable
                continue;
            };
            let patch = c.data.seal(prepared);
            {
                let written = std::mem::take(&mut self.channel_writes[ci]);
                let written = without(sorted(written), &allocated_v, &freed_v);
                if !written.is_empty() {
                    channels.push((c.desc.name.clone(), written));
                }
                if !patch.is_empty() {
                    channel_data.push((c.id, patch));
                }
            }
        }

        let changes = ChangeSet {
            vertices_added: created_v,
            vertices_removed: freed_v,
            vertices_moved: moved,
            triangles_added: created_t,
            triangles_removed: freed_t,
            triangles_changed: changed_t,
            channels,
        };
        let base_revision = m.revision;
        if !changes.is_empty() {
            m.next_revision += 1;
            m.revision = m.next_revision;
        }
        let patch = Patch {
            mesh_uid: m.uid,
            epoch: m.epoch,
            base_revision,
            revision: m.revision,
            positions: m.positions.seal(prepared.positions),
            triangles: m.triangles.seal(prepared.triangles),
            vertex_live: m.vertex_live.seal(prepared.vertex_live),
            triangle_live: m.triangle_live.seal(prepared.triangle_live),
            vertex_refs: m.vertex_refs.seal(prepared.vertex_refs),
            channel_data,
            changes,
        };
        if patch.changes.topology_changed() {
            m.on_topology_changed();
        }
        Ok(patch)
    }

    /// Discards every change made in this transaction.
    pub fn rollback(mut self) {
        self.finished = true;
        self.rollback_inner();
    }

    fn rollback_inner(&mut self) {
        let m = &mut *self.mesh;
        m.positions.rollback();
        m.triangles.rollback();
        m.vertex_live.rollback();
        m.triangle_live.rollback();
        m.vertex_refs.rollback();
        for c in &mut m.channels {
            c.data.rollback();
        }
        while let Some(v) = self.popped_vertices.pop() {
            m.vertex_free.push(v);
        }
        while let Some(t) = self.popped_triangles.pop() {
            m.triangle_free.push(t);
        }
        (m.live_vertices, m.live_triangles) = self.live_at_start;
    }
}

impl Drop for Transaction<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.rollback_inner();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::column::fail_allocations_after;
    use crate::{ChannelDesc, Mesh, StoreError, TopologyPolicy};

    fn quad() -> Mesh {
        Mesh::from_buffers(
            &[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap()
    }

    #[test]
    fn moving_vertices_commits_a_reversible_patch() {
        let mut m = quad();
        let mut tx = m.begin_edit();
        tx.set_positions(&[2], &[[5., 5., 5.]]).unwrap();
        let patch = tx.commit().unwrap();
        assert_eq!(m.position(2), Some([5., 5., 5.]));
        assert_eq!(patch.changes().vertices_moved, vec![2]);
        assert!(!patch.changes().topology_changed());

        let undo = patch.revert(&mut m).unwrap();
        assert_eq!(m.position(2), Some([1., 1., 0.]));
        assert_eq!(undo.vertices_moved, vec![2]);
        patch.apply(&mut m).unwrap();
        assert_eq!(m.position(2), Some([5., 5., 5.]));
    }

    #[test]
    fn adding_and_removing_topology_round_trips() {
        let mut m = quad();
        let mut tx = m.begin_edit();
        let v = tx.add_vertices(&[[2., 0., 0.]]).unwrap()[0];
        let t = tx.add_triangles(&[[1, v, 2]]).unwrap()[0];
        tx.remove_triangles(&[0]).unwrap();
        let patch = tx.commit().unwrap();
        assert_eq!((v, t), (4, 2));
        assert_eq!((m.vertex_count(), m.triangle_count()), (5, 2));
        assert_eq!(
            m.triangles()[0],
            [0, 0, 0],
            "removed triangles read as degenerate"
        );
        assert_eq!(m.corner_count(1), 1);

        let undo = patch.revert(&mut m).unwrap();
        assert_eq!(undo.triangles_added, vec![0]);
        assert_eq!(undo.triangles_removed, vec![2]);
        assert_eq!(undo.vertices_removed, vec![4]);
        assert_eq!((m.vertex_count(), m.triangle_count()), (4, 2));
        assert_eq!(m.triangle(0), Some([0, 1, 2]));
        assert!(!m.is_vertex_alive(4));
        assert_eq!(m.corner_count(1), 1);
    }

    #[test]
    fn validation_errors_leave_the_transaction_usable() {
        let mut m = quad();
        let mut tx = m.begin_edit();
        assert_eq!(
            tx.set_positions(&[9], &[[0.; 3]]),
            Err(StoreError::InvalidVertex(9))
        );
        assert_eq!(
            tx.remove_vertices(&[0]),
            Err(StoreError::VertexInUse {
                vertex: 0,
                corners: 2
            })
        );
        assert!(matches!(
            tx.remove_triangles(&[0, 0]),
            Err(StoreError::InvalidArgument(_))
        ));
        tx.set_positions(&[0], &[[1., 2., 3.]]).unwrap();
        tx.commit().unwrap();
        assert_eq!(m.position(0), Some([1., 2., 3.]));
    }

    #[test]
    fn rollback_and_drop_discard_everything() {
        let mut m = quad();
        let before = m.revision();
        let mut tx = m.begin_edit();
        tx.add_vertices(&[[9.; 3]]).unwrap();
        tx.remove_triangles(&[1]).unwrap();
        tx.rollback();
        assert_eq!(
            (m.vertex_slots(), m.vertex_count(), m.triangle_count()),
            (4, 4, 2)
        );
        assert_eq!(m.triangle(1), Some([0, 2, 3]));
        assert_eq!(m.revision(), before);

        {
            let mut tx = m.begin_edit();
            tx.set_positions(&[0], &[[7.; 3]]).unwrap();
        }
        assert_eq!(m.position(0), Some([0., 0., 0.]));
    }

    #[test]
    fn empty_transactions_keep_the_revision() {
        let mut m = quad();
        let patch = m.begin_edit().commit().unwrap();
        assert!(patch.is_empty());
        assert_eq!(m.revision(), 0);
    }

    #[test]
    fn new_vertices_take_channel_values_from_parents() {
        let mut m = quad();
        m.add_channel(ChannelDesc::source("mask", 1)).unwrap();
        m.add_channel(ChannelDesc::source("id", 1).with_policy(TopologyPolicy::Nearest))
            .unwrap();
        let mut tx = m.begin_edit();
        tx.write_channel("mask", &[0, 1], &[2.0, 4.0]).unwrap();
        tx.write_channel("id", &[0, 1], &[10.0, 20.0]).unwrap();
        let v = tx
            .add_vertex_with_parents([0.5, 0., 0.], &[(0, 0.75), (1, 0.25)])
            .unwrap();
        let patch = tx.commit().unwrap();
        assert_eq!(m.channel("mask").unwrap()[v as usize], 2.5);
        assert_eq!(m.channel("id").unwrap()[v as usize], 10.0);
        assert_eq!(
            patch.changes().channels,
            vec![
                ("mask".to_string(), vec![0, 1]),
                ("id".to_string(), vec![0, 1])
            ]
        );
        patch.revert(&mut m).unwrap();
        assert_eq!(&m.channel("mask").unwrap()[..2], &[0.0, 0.0]);
    }

    #[test]
    fn patches_skip_a_channel_that_was_removed_and_re_added_under_the_same_name() {
        // Width change (used to panic) and equal width (used to corrupt the new channel).
        for new_width in [1, 4] {
            let mut m = quad();
            m.add_channel(ChannelDesc::source("c", 4)).unwrap();
            let mut tx = m.begin_edit();
            tx.write_channel("c", &[0, 1], &[1.0; 8]).unwrap();
            let patch = tx.commit().unwrap();
            m.remove_channel("c").unwrap();
            m.add_channel(ChannelDesc::source("c", new_width).with_default(9.0))
                .unwrap();
            let untouched = vec![9.0; 4 * new_width];
            patch.revert(&mut m).unwrap();
            assert_eq!(m.channel("c").unwrap(), &untouched[..], "width {new_width}");
            patch.apply(&mut m).unwrap();
            assert_eq!(m.channel("c").unwrap(), &untouched[..], "width {new_width}");
        }
    }

    #[test]
    fn commit_that_runs_out_of_memory_rolls_back_everything() {
        type State = (Vec<[f32; 3]>, Vec<[u32; 3]>, Vec<f32>, usize, usize, u64);
        fn state(m: &Mesh) -> State {
            (
                m.positions()[..4].to_vec(),
                m.triangles()[..2].to_vec(),
                m.channel("c").unwrap()[..4].to_vec(),
                m.vertex_count(),
                m.triangle_count(),
                m.revision(),
            )
        }
        // Sparse patches, and snapshot patches (every position rewritten).
        for moved in [vec![1], vec![0, 1, 2, 3]] {
            let mut m = quad();
            m.add_channel(ChannelDesc::source("c", 1)).unwrap();
            m.add_channel(ChannelDesc::derived("d", 1)).unwrap();
            let before = state(&m);
            for k in 0.. {
                // A journal left behind by a failed commit would trip `begin()`.
                let mut tx = m.begin_edit();
                tx.set_positions(&moved, &vec![[7.; 3]; moved.len()])
                    .unwrap();
                let v = tx.add_vertices(&[[1.; 3]]).unwrap()[0];
                tx.add_triangles(&[[0, 1, v]]).unwrap();
                tx.remove_triangles(&[1]).unwrap();
                tx.write_channel("c", &[0], &[3.]).unwrap();
                fail_allocations_after(Some(k));
                let result = tx.commit();
                fail_allocations_after(None);
                match result {
                    Err(e) => {
                        assert_eq!(e, StoreError::OutOfMemory);
                        assert_eq!(state(&m), before, "failed after {k} allocations");
                        assert_eq!(m.vertex_slots(), 4, "appended slots truncated");
                        assert_eq!(m.channel("d").unwrap().len(), 4);
                    }
                    Ok(patch) => {
                        assert!(k > 0, "commit needs to allocate");
                        assert_eq!(m.position(1), Some([7.; 3]));
                        assert_eq!(m.triangle(1), None);
                        assert_eq!(m.channel("c").unwrap()[0], 3.);
                        // The journals were intact: the patch restores everything.
                        patch.revert(&mut m).unwrap();
                        assert_eq!(state(&m), before);
                        break;
                    }
                }
            }
        }
    }

    #[test]
    fn writing_a_derived_channel_in_a_transaction_is_rejected() {
        let mut m = quad();
        m.add_channel(ChannelDesc::derived("dist", 1)).unwrap();
        let mut tx = m.begin_edit();
        assert_eq!(
            tx.write_channel("dist", &[0], &[1.0]),
            Err(StoreError::NotSourceChannel("dist".into()))
        );
    }

    #[test]
    fn slots_created_and_removed_in_one_transaction_are_reused() {
        let mut m = quad();
        let mut tx = m.begin_edit();
        let v = tx.add_vertices(&[[0.; 3]]).unwrap()[0];
        tx.remove_vertices(&[v]).unwrap();
        let patch = tx.commit().unwrap();
        assert!(patch.is_empty());
        assert_eq!(m.free_vertex_slots(), &[v]);
        let mut tx = m.begin_edit();
        assert_eq!(tx.add_vertices(&[[1.; 3]]).unwrap(), vec![v]);
    }

    #[test]
    fn undoing_an_append_past_a_bitset_word_kills_the_slot() {
        // 64 vertices fill exactly one liveness word; vertex 64 starts a new word.
        let positions = vec![0.0f32; 64 * 3];
        let mut m = Mesh::from_buffers(&positions, &[]).unwrap();
        let mut tx = m.begin_edit();
        let v = tx.add_vertices(&[[1.; 3]]).unwrap()[0];
        let t = tx.add_triangles(&[[0, 1, v]]).unwrap()[0];
        let patch = tx.commit().unwrap();
        assert_eq!(v, 64);
        patch.revert(&mut m).unwrap();
        assert!(!m.is_vertex_alive(64));
        assert!(!m.is_triangle_alive(t));
        assert_eq!(m.triangles()[t as usize], [0, 0, 0]);
        assert_eq!(m.vertex_count(), 64);
        patch.apply(&mut m).unwrap();
        assert!(m.is_vertex_alive(64));
        assert_eq!(m.triangle(t), Some([0, 1, 64]));
    }

    #[test]
    fn patches_only_apply_at_the_matching_revision() {
        let mut m = quad();
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[1.; 3]]).unwrap();
        let first = tx.commit().unwrap();
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[2.; 3]]).unwrap();
        let _second = tx.commit().unwrap();
        assert_eq!(first.revert(&mut m).err(), Some(StoreError::PatchMismatch));
        let mut other = quad();
        assert_eq!(
            first.apply(&mut other).err(),
            Some(StoreError::PatchMismatch)
        );
    }
}
