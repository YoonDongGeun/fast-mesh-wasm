//! Vertex → triangles incidence index.
//!
//! Lists, for every vertex slot, all alive triangles that use it. Unlike
//! [`crate::Adjacency::triangles_around`] it covers every fan of a non-manifold
//! vertex, which is what BVH refits need after vertices move.
//!
//! Once built, the index follows topology edits instead of being rebuilt: the
//! rows of the vertices an edit touches are rewritten into an overlay that
//! shadows the compressed rows. When the overlay grows past a fraction of the
//! index, the index is dropped and the next [`Mesh::ensure_incidence`] rebuilds
//! it compactly.

use std::collections::HashMap;

use crate::column::{try_reserve, try_reserve_map, try_vec};
use crate::{Mesh, Result, StoreError, TriangleId, VertexId};

/// Overlay size (entries plus rows) always allowed before the index is rebuilt.
const MIN_OVERLAY: usize = 4096;

/// Every alive triangle using each vertex, stored in compressed rows.
#[derive(Debug)]
pub struct Incidence {
    /// `offsets[v]..offsets[v + 1]` is the range of `triangles` for vertex `v`.
    offsets: Vec<u32>,
    triangles: Vec<TriangleId>,
    /// Rows rewritten since the compressed rows were built. A vertex present
    /// here ignores its compressed row.
    overlay: HashMap<VertexId, Vec<TriangleId>>,
    /// Total entries in `overlay`.
    overlay_len: usize,
}

/// The distinct vertices of a triangle: one repeating a vertex (`[a, a, b]`)
/// is listed once for it.
fn distinct(tri: [VertexId; 3]) -> impl Iterator<Item = VertexId> {
    let [a, b, c] = tri;
    [
        Some(a),
        (b != a).then_some(b),
        (c != a && c != b).then_some(c),
    ]
    .into_iter()
    .flatten()
}

impl Incidence {
    pub(crate) fn build(mesh: &Mesh) -> Result<Self> {
        let mut build = IncidenceBuild::new(mesh);
        build.step(mesh, usize::MAX)?;
        build.take().ok_or(StoreError::OutOfMemory)
    }

    /// Every alive triangle using `v`, each once, in ascending id order. Empty for
    /// unused, dead or out-of-range vertices.
    pub fn triangles(&self, v: VertexId) -> &[TriangleId] {
        match self.overlay.get(&v) {
            Some(row) => row,
            None => compressed_row(&self.offsets, &self.triangles, v),
        }
    }

    /// Lists triangle `t`, whose vertices are `tri`, in the index.
    fn insert(&mut self, t: TriangleId, tri: [VertexId; 3]) -> Result<()> {
        for v in distinct(tri) {
            if self.triangles(v).binary_search(&t).is_ok() {
                continue;
            }
            let row = self.row_mut(v)?;
            if let Err(i) = row.binary_search(&t) {
                try_reserve(row, 1)?;
                row.insert(i, t);
                self.overlay_len += 1;
            }
        }
        Ok(())
    }

    /// The overlay row of `v`, created from its compressed row if needed.
    fn row_mut(&mut self, v: VertexId) -> Result<&mut Vec<TriangleId>> {
        if !self.overlay.contains_key(&v) {
            let base = compressed_row(&self.offsets, &self.triangles, v);
            let mut row = Vec::new();
            try_reserve(&mut row, base.len() + 1)?;
            row.extend_from_slice(base);
            try_reserve_map(&mut self.overlay, 1)?;
            self.overlay_len += row.len();
            self.overlay.insert(v, row);
        }
        Ok(self.overlay.get_mut(&v).expect("inserted above"))
    }

    /// Takes `ids` that are alive now out of the index, before they are rewritten.
    pub(crate) fn remove_alive(&mut self, mesh: &Mesh, ids: &[&[TriangleId]]) -> Result<()> {
        for &t in ids.iter().flat_map(|ids| ids.iter()) {
            if let Some(tri) = mesh.triangle(t) {
                self.remove(t, tri)?;
            }
        }
        Ok(())
    }

    /// Lists `ids` that are alive now, after they were written.
    pub(crate) fn insert_alive(&mut self, mesh: &Mesh, ids: &[&[TriangleId]]) -> Result<()> {
        for &t in ids.iter().flat_map(|ids| ids.iter()) {
            if let Some(tri) = mesh.triangle(t) {
                self.insert(t, tri)?;
            }
        }
        Ok(())
    }

    /// Takes triangle `t`, whose vertices were `tri`, out of the index. Does
    /// nothing for rows that do not list it.
    pub(crate) fn remove(&mut self, t: TriangleId, tri: [VertexId; 3]) -> Result<()> {
        for v in distinct(tri) {
            if self.triangles(v).binary_search(&t).is_err() {
                continue;
            }
            let row = self.row_mut(v)?;
            if let Ok(i) = row.binary_search(&t) {
                row.remove(i);
                self.overlay_len -= 1;
            }
        }
        Ok(())
    }

    /// True once the overlay is large enough that a compact rebuild pays off.
    /// Rows count too, so emptied rows of removed vertices are not free.
    fn overgrown(&self) -> bool {
        self.overlay_len + self.overlay.len() > MIN_OVERLAY.max(self.triangles.len() / 8)
    }
}

/// Where an [`IncidenceBuild`] stands: the pass it is in, with that pass's
/// cursor (a triangle or vertex slot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pass {
    /// Counting the triangles of each vertex, triangle by triangle.
    Count(usize),
    /// Turning the counts into row offsets, vertex by vertex.
    Prefix(usize),
    /// Listing each vertex's triangles, triangle by triangle.
    Fill(usize),
    /// Finished: the index is ready to be taken.
    Done,
}

/// A resumable build of a mesh's [`Incidence`]: the build of
/// [`Mesh::ensure_incidence`] cut into steps of a bounded number of elements
/// (triangle or vertex slots), so that a caller can yield between them.
///
/// Each step runs the same loop bodies as the one-call build, over the next
/// elements in the same order, so the finished index is the same, entry for
/// entry. A build is tied to the mesh state it started from (its uid, epoch
/// and revision); [`Mesh::step_incidence`] starts over when the mesh has
/// changed since, and caches the index when the build is done.
#[derive(Debug)]
pub struct IncidenceBuild {
    uid: u64,
    epoch: u32,
    revision: u64,
    pass: Pass,
    offsets: Vec<u32>,
    triangles: Vec<TriangleId>,
    cursor: Vec<u32>,
}

impl IncidenceBuild {
    /// A build of `mesh`'s incidence index as the mesh is now, with nothing
    /// done.
    pub fn new(mesh: &Mesh) -> IncidenceBuild {
        IncidenceBuild {
            uid: mesh.uid(),
            epoch: mesh.epoch(),
            revision: mesh.revision(),
            pass: Pass::Count(0),
            offsets: Vec::new(),
            triangles: Vec::new(),
            cursor: Vec::new(),
        }
    }

    /// Whether the build started from `mesh` as it is now.
    pub fn is_for(&self, mesh: &Mesh) -> bool {
        self.uid == mesh.uid() && self.epoch == mesh.epoch() && self.revision == mesh.revision()
    }

    /// Whether every pass is done.
    pub fn is_done(&self) -> bool {
        self.pass == Pass::Done
    }

    /// Runs the build over up to `budget` more elements (at least one, unless
    /// it is done) of `mesh`, which must be the mesh state it started from.
    /// Returns whether it is done.
    fn step(&mut self, mesh: &Mesh, budget: usize) -> Result<bool> {
        debug_assert!(
            self.is_for(mesh),
            "an incidence build stepped on another mesh"
        );
        let tris = mesh.triangles();
        let slots = mesh.vertex_slots();
        let mut left = budget.max(1);
        while left > 0 {
            match self.pass {
                Pass::Count(from) => {
                    if from == 0 {
                        self.offsets = try_vec(slots + 1, 0u32)?;
                    }
                    let to = tris.len().min(from.saturating_add(left));
                    for (t, &tri) in tris.iter().enumerate().take(to).skip(from) {
                        if mesh.is_triangle_alive(t as TriangleId) {
                            for v in distinct(tri) {
                                self.offsets[v as usize + 1] += 1;
                            }
                        }
                    }
                    left -= (to - from).min(left);
                    self.pass = if to == tris.len() {
                        Pass::Prefix(0)
                    } else {
                        Pass::Count(to)
                    };
                }
                Pass::Prefix(from) => {
                    let to = slots.min(from.saturating_add(left));
                    for v in from..to {
                        self.offsets[v + 1] += self.offsets[v];
                    }
                    left -= (to - from).min(left);
                    if to < slots {
                        self.pass = Pass::Prefix(to);
                        continue;
                    }
                    self.triangles = try_vec(self.offsets[slots] as usize, 0 as TriangleId)?;
                    self.cursor = try_vec(slots, 0u32)?;
                    self.cursor.copy_from_slice(&self.offsets[..slots]);
                    self.pass = Pass::Fill(0);
                }
                Pass::Fill(from) => {
                    let to = tris.len().min(from.saturating_add(left));
                    // Triangles are visited in ascending id order, so every row comes
                    // out sorted.
                    for (t, &tri) in tris.iter().enumerate().take(to).skip(from) {
                        if mesh.is_triangle_alive(t as TriangleId) {
                            for v in distinct(tri) {
                                let at = &mut self.cursor[v as usize];
                                self.triangles[*at as usize] = t as TriangleId;
                                *at += 1;
                            }
                        }
                    }
                    left -= (to - from).min(left);
                    if to < tris.len() {
                        self.pass = Pass::Fill(to);
                        continue;
                    }
                    self.cursor = Vec::new();
                    self.pass = Pass::Done;
                }
                Pass::Done => return Ok(true),
            }
        }
        Ok(self.pass == Pass::Done)
    }

    /// The finished index, once [`IncidenceBuild::step`] returned `true`.
    fn take(&mut self) -> Option<Incidence> {
        (self.pass == Pass::Done).then(|| Incidence {
            offsets: std::mem::take(&mut self.offsets),
            triangles: std::mem::take(&mut self.triangles),
            overlay: HashMap::new(),
            overlay_len: 0,
        })
    }
}

fn compressed_row<'a>(
    offsets: &[u32],
    triangles: &'a [TriangleId],
    v: VertexId,
) -> &'a [TriangleId] {
    let v = v as usize;
    if v + 1 >= offsets.len() {
        return &[];
    }
    &triangles[offsets[v] as usize..offsets[v + 1] as usize]
}

impl Mesh {
    /// Builds the vertex → triangles index if it is not cached. Once built, it
    /// follows edits, rollbacks, undo and redo. It is dropped by [`Mesh::compact`],
    /// by edits that leave so many rewritten rows that rebuilding is cheaper, and
    /// if memory runs out while following an edit.
    pub fn ensure_incidence(&mut self) -> Result<()> {
        debug_assert!(!self.incidence_pending, "called inside a transaction");
        if self.incidence.is_none() {
            self.incidence = Some(Incidence::build(self)?);
        }
        Ok(())
    }

    /// The cached index, if [`Mesh::ensure_incidence`] built it. `None` inside a
    /// transaction that changed topology, until it commits or rolls back.
    pub fn incidence(&self) -> Option<&Incidence> {
        self.incidence.as_ref().filter(|_| !self.incidence_pending)
    }

    /// [`Mesh::ensure_incidence`] in steps: runs `build` over up to `budget`
    /// more elements and, once it is done, caches its index. Returns whether
    /// the index is cached, at once when it already is. When the mesh changed
    /// since `build` started, `build` starts over first. The index is the
    /// same as from [`Mesh::ensure_incidence`], entry for entry.
    ///
    /// # Errors
    /// As for [`Mesh::ensure_incidence`]; `build` then starts over.
    pub fn step_incidence(&mut self, build: &mut IncidenceBuild, budget: usize) -> Result<bool> {
        debug_assert!(!self.incidence_pending, "called inside a transaction");
        if self.incidence.is_some() {
            return Ok(true);
        }
        if !build.is_for(self) {
            *build = IncidenceBuild::new(self);
        }
        match build.step(self, budget) {
            Ok(done) => {
                if done {
                    self.incidence = build.take();
                }
                Ok(done)
            }
            Err(e) => {
                *build = IncidenceBuild::new(self);
                Err(e)
            }
        }
    }

    /// Runs `f` on the cached index. Drops the index if `f` fails (out of
    /// memory) or the overlay grew large enough that a rebuild pays off.
    pub(crate) fn with_incidence(&mut self, f: impl FnOnce(&mut Incidence, &Mesh) -> Result<()>) {
        if let Some(mut inc) = self.incidence.take()
            && f(&mut inc, self).is_ok()
            && !inc.overgrown()
        {
            self.incidence = Some(inc);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Mesh;
    use crate::column::fail_allocations_after;

    /// Three triangles share vertex 0; a fourth touches it only at vertex 0
    /// (a bowtie), so the vertex is non-manifold.
    fn fan_and_bowtie() -> Mesh {
        Mesh::from_buffers(
            &[
                0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., -1., 0., 0., -1., -1., 0.,
            ],
            &[0, 1, 2, 0, 2, 3, 3, 2, 1, 0, 4, 5],
        )
        .unwrap()
    }

    /// Brute-force rows for every vertex slot.
    fn expected(m: &Mesh) -> Vec<Vec<u32>> {
        let mut rows = vec![Vec::new(); m.vertex_slots()];
        for t in 0..m.triangle_slots() as u32 {
            if let Some(tri) = m.triangle(t) {
                for v in tri {
                    if rows[v as usize].last() != Some(&t) {
                        rows[v as usize].push(t);
                    }
                }
            }
        }
        rows
    }

    fn assert_matches(m: &Mesh) {
        let inc = m.incidence().expect("index kept");
        for (v, row) in expected(m).iter().enumerate() {
            assert_eq!(inc.triangles(v as u32), row.as_slice(), "vertex {v}");
        }
    }

    #[test]
    fn lists_every_triangle_of_every_fan() {
        let mut m = fan_and_bowtie();
        m.ensure_incidence().unwrap();
        let inc = m.incidence().unwrap();
        assert_eq!(inc.triangles(0), &[0, 1, 3]);
        assert_eq!(inc.triangles(2), &[0, 1, 2]);
        assert_eq!(inc.triangles(5), &[3]);
        assert!(inc.triangles(99).is_empty());
    }

    #[test]
    fn repeated_vertices_are_listed_once() {
        let mut m = Mesh::from_buffers(&[0., 0., 0., 1., 0., 0.], &[0, 0, 1]).unwrap();
        m.ensure_incidence().unwrap();
        assert_eq!(m.incidence().unwrap().triangles(0), &[0]);
        assert_eq!(m.incidence().unwrap().triangles(1), &[0]);
    }

    #[test]
    fn topology_edits_undo_and_redo_keep_the_index_current() {
        let mut m = fan_and_bowtie();
        m.ensure_incidence().unwrap();
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[0., 0., 1.]]).unwrap();
        tx.commit().unwrap();
        assert_matches(&m);

        let mut tx = m.begin_edit();
        tx.remove_triangles(&[1]).unwrap();
        assert!(tx.mesh().incidence().is_none(), "hidden while stale");
        let v = tx.add_vertices(&[[2., 2., 0.]]).unwrap()[0];
        tx.add_triangles(&[[v, 2, 3], [0, 0, v]]).unwrap();
        tx.set_triangles(&[2], &[[3, 1, 5]]).unwrap();
        let patch = tx.commit().unwrap();
        assert_matches(&m);
        assert_eq!(m.incidence().unwrap().triangles(0), &[0, 3, 5]);

        patch.revert(&mut m).unwrap();
        assert_matches(&m);
        patch.apply(&mut m).unwrap();
        assert_matches(&m);
    }

    #[test]
    fn rollback_restores_the_index() {
        let mut m = fan_and_bowtie();
        m.ensure_incidence().unwrap();
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[0, 3]).unwrap();
        tx.add_triangles(&[[1, 2, 3]]).unwrap();
        tx.rollback();
        assert_matches(&m);
    }

    /// A strip of `n` triangles over `n + 2` vertices.
    fn strip(n: u32) -> Mesh {
        let positions: Vec<f32> = (0..n + 2)
            .flat_map(|i| [(i / 2) as f32, (i % 2) as f32, 0.])
            .collect();
        let indices: Vec<u32> = (0..n).flat_map(|i| [i, i + 1, i + 2]).collect();
        Mesh::from_buffers(&positions, &indices).unwrap()
    }

    #[test]
    fn small_edits_keep_the_index_and_large_ones_drop_it() {
        let mut m = strip(10_000);
        m.ensure_incidence().unwrap();
        let mut tx = m.begin_edit();
        tx.set_triangles(&[5_000], &[[5_002, 5_001, 5_000]])
            .unwrap();
        tx.commit().unwrap();
        assert_matches(&m);

        // Rewriting every triangle would leave the whole index in the overlay.
        let ids: Vec<u32> = (0..10_000).collect();
        let flipped: Vec<[u32; 3]> = ids.iter().map(|&i| [i + 2, i + 1, i]).collect();
        let mut tx = m.begin_edit();
        tx.set_triangles(&ids, &flipped).unwrap();
        let patch = tx.commit().unwrap();
        assert!(m.incidence().is_none());
        m.ensure_incidence().unwrap();
        assert_matches(&m);
        patch.revert(&mut m).unwrap();
        assert!(m.incidence().is_none());
    }

    #[test]
    fn compaction_drops_the_index() {
        let mut m = fan_and_bowtie();
        m.ensure_incidence().unwrap();
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[1]).unwrap();
        tx.commit().unwrap();
        m.compact().unwrap();
        assert!(m.incidence().is_none());
        m.ensure_incidence().unwrap();
        assert_matches(&m);
    }

    #[test]
    fn running_out_of_memory_drops_the_index_but_not_the_edit() {
        let mut dropped = false;
        for n in 0..64 {
            let mut m = fan_and_bowtie();
            m.ensure_incidence().unwrap();
            let mut tx = m.begin_edit();
            tx.remove_triangles(&[1]).unwrap();
            fail_allocations_after(Some(n));
            let committed = tx.commit().is_ok();
            fail_allocations_after(None);
            match m.incidence() {
                Some(_) => assert_matches(&m),
                None => {
                    assert!(committed, "a rollback keeps the index");
                    dropped = true;
                }
            }
        }
        assert!(dropped, "some budget must drop the index");
    }

    #[test]
    fn stepped_build_matches_one_call() {
        use crate::IncidenceBuild;
        let mixed = crate::adjacency::tests::mixed;
        let mut whole = mixed();
        whole.ensure_incidence().unwrap();
        let reference = whole.incidence().unwrap();
        for budget in [1, 3, 50, usize::MAX] {
            let mut m = mixed();
            let mut build = IncidenceBuild::new(&m);
            while !m.step_incidence(&mut build, budget).unwrap() {
                assert!(m.incidence().is_none(), "cached before the last step");
            }
            assert!(build.is_done());
            let got = m.incidence().unwrap();
            assert_eq!(got.offsets, reference.offsets, "budget {budget}");
            assert_eq!(got.triangles, reference.triangles, "budget {budget}");
            assert_matches(&m);
        }
        // Started over after an edit.
        let mut m = mixed();
        let mut build = IncidenceBuild::new(&m);
        assert!(!m.step_incidence(&mut build, 5).unwrap());
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[7]).unwrap();
        tx.commit().unwrap();
        assert!(!build.is_for(&m));
        while !m.step_incidence(&mut build, 5).unwrap() {}
        assert_matches(&m);
    }
}
