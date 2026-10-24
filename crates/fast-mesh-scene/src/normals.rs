//! Smooth vertex normals, kept up to date incrementally from change sets.

use fast_mesh_store::{ChangeSet, Mesh, StoreError, TriangleId, VertexId};

use crate::scene::Entry;
use crate::{MeshId, Result, Scene, SceneError};

fn oom() -> SceneError {
    SceneError::Store(StoreError::OutOfMemory)
}

/// Grows `v` to `len` items of `fill` (or truncates it), failing instead of
/// aborting when memory runs out.
fn try_resize<T: Copy>(v: &mut Vec<T>, len: usize, fill: T) -> Result<()> {
    v.try_reserve(len.saturating_sub(v.len()))
        .map_err(|_| oom())?;
    v.resize(len, fill);
    Ok(())
}

/// The un-normalised normal of triangle `tri`: twice its area long, so
/// summing them weights by area. Zero for a triangle repeating a vertex.
fn face_normal(p: &[[f32; 3]], tri: [VertexId; 3]) -> [f32; 3] {
    let [a, b, c] = tri.map(|v| p[v as usize]);
    let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ]
}

/// Appends `vs` to `dirty`, failing instead of aborting when memory runs out.
fn push(dirty: &mut Vec<VertexId>, vs: &[VertexId]) -> Result<()> {
    dirty.try_reserve(vs.len()).map_err(|_| oom())?;
    dirty.extend_from_slice(vs);
    Ok(())
}

fn add(sum: &mut [f32; 3], n: [f32; 3]) {
    for k in 0..3 {
        sum[k] += n[k];
    }
}

/// `n` scaled to unit length; zero stays zero.
fn normalized(n: [f32; 3]) -> [f32; 3] {
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 0.0 { n.map(|x| x / len) } else { n }
}

/// A mesh's vertex normals, one per vertex slot, with the changes since
/// they were last computed.
pub(crate) struct NormalState {
    normals: Vec<[f32; 3]>,
    /// The triangles as they were when the normals were last computed, so
    /// that the old vertices of a removed or rewritten triangle are known.
    /// Entries of slots that were dead then are meaningless.
    triangles: Vec<[VertexId; 3]>,
    /// Changes since the last computation; `None` means "everything".
    pending: Option<ChangeSet>,
}

impl NormalState {
    /// The normals of `mesh`, computed in full.
    pub(crate) fn new(mesh: &Mesh) -> Result<NormalState> {
        let mut state = NormalState {
            normals: Vec::new(),
            triangles: Vec::new(),
            pending: None,
        };
        state.compute_all(mesh)?;
        Ok(state)
    }

    pub(crate) fn normals(&self) -> &[[f32; 3]] {
        &self.normals
    }

    /// Folds a commit's (or undo's or redo's) changes into the pending
    /// changes, and grows the normals to cover new vertex slots (zero until
    /// computed). If memory runs out while growing, everything becomes
    /// pending and the normals stay short until the next update succeeds.
    pub(crate) fn record(&mut self, changes: &ChangeSet, vertex_slots: usize) {
        if let Some(p) = &mut self.pending {
            p.merge(changes);
        }
        if vertex_slots > self.normals.len()
            && try_resize(&mut self.normals, vertex_slots, [0.0; 3]).is_err()
        {
            self.pending = None;
        }
    }

    /// Marks every normal as stale (after a compaction or a failed
    /// recovery).
    pub(crate) fn mark_all(&mut self) {
        self.pending = None;
    }

    /// Recomputes every normal from scratch (see [`NormalBuild`], run to
    /// the end). On error the state is unchanged.
    fn compute_all(&mut self, mesh: &Mesh) -> Result<()> {
        let mut build = NormalBuild::new();
        while !build.step(mesh, usize::MAX)? {}
        *self = build.finish().ok_or_else(oom)?;
        Ok(())
    }

    /// Recomputes the normals the pending changes `c` may have changed and
    /// returns their vertex ids, ascending without duplicates: every vertex
    /// of every triangle added, removed or rewritten (old and new
    /// vertices), and every vertex of every triangle using a moved vertex.
    /// Added, removed and moved vertices are included too.
    fn compute_changed(&mut self, mesh: &mut Mesh, c: &ChangeSet) -> Result<Vec<VertexId>> {
        // Every allocation comes first, so a failure leaves the state as it was.
        mesh.ensure_incidence()?;
        let (slots, tri_slots) = (mesh.vertex_slots(), mesh.triangle_slots());
        try_resize(&mut self.normals, slots, [0.0; 3])?;
        try_resize(&mut self.triangles, tri_slots, [0; 3])?;
        let mesh = &*mesh;
        let inc = mesh.incidence().expect("built above");
        let tris = mesh.triangles();
        let mut dirty: Vec<VertexId> = Vec::new();
        for &t in c.triangles_removed.iter().chain(&c.triangles_changed) {
            if let Some(old) = self.triangles.get(t as usize) {
                push(&mut dirty, old)?;
            }
        }
        for &t in c.triangles_added.iter().chain(&c.triangles_changed) {
            if let Some(tri) = mesh.triangle(t) {
                push(&mut dirty, &tri)?;
            }
        }
        push(&mut dirty, &c.vertices_added)?;
        push(&mut dirty, &c.vertices_removed)?;
        push(&mut dirty, &c.vertices_moved)?;
        for &v in &c.vertices_moved {
            for &t in inc.triangles(v) {
                push(&mut dirty, &tris[t as usize])?;
            }
        }
        dirty.sort_unstable();
        dirty.dedup();
        dirty.retain(|&v| (v as usize) < slots);
        // Ascending triangle order per vertex, as in `compute_all`, so the
        // result matches a full recomputation.
        let p = mesh.positions();
        for &v in &dirty {
            let mut sum = [0.0f32; 3];
            for &t in inc.triangles(v) {
                let tri = tris[t as usize];
                let n = face_normal(p, tri);
                // A triangle using `v` twice counts twice, as in `compute_all`
                // (its face normal is zero anyway).
                for &w in &tri {
                    if w == v {
                        add(&mut sum, n);
                    }
                }
            }
            self.normals[v as usize] = normalized(sum);
        }
        for &t in c
            .triangles_added
            .iter()
            .chain(&c.triangles_removed)
            .chain(&c.triangles_changed)
        {
            if let Some(slot) = self.triangles.get_mut(t as usize) {
                *slot = tris[t as usize];
            }
        }
        self.pending = Some(ChangeSet::default());
        Ok(dirty)
    }
}

/// Vertex slots zeroed, or normalised, per unit of a [`NormalBuild::step`]
/// budget.
const VERTICES_PER_UNIT: usize = 16;

/// Triangle slots summed per unit of a [`NormalBuild::step`] budget.
const TRIANGLES_PER_UNIT: usize = 4;

/// Where a [`NormalBuild`] stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalStage {
    /// Zeroing the normals, then copying the triangles, from this slot on.
    Zero(usize),
    /// Summing the face normals, from this triangle slot on.
    Sum(usize),
    /// Normalising, from this vertex slot on.
    Normalize(usize),
    Done,
}

/// [`NormalState::new`] cut into steps, so that a caller can yield between
/// them (a mesh added in the background computes its normals this way,
/// outside the scene). The normals are the same, bit for bit: the face
/// normals are summed in ascending triangle order either way. The build is
/// tied to the mesh it is stepped on, which must not change meanwhile.
#[derive(Debug)]
pub(crate) struct NormalBuild {
    normals: Vec<[f32; 3]>,
    triangles: Vec<[VertexId; 3]>,
    stage: NormalStage,
}

impl NormalBuild {
    pub(crate) fn new() -> NormalBuild {
        NormalBuild {
            normals: Vec::new(),
            triangles: Vec::new(),
            stage: NormalStage::Zero(0),
        }
    }

    pub(crate) fn is_done(&self) -> bool {
        self.stage == NormalStage::Done
    }

    /// The normals so far (final once done).
    pub(crate) fn normals(&self) -> &[[f32; 3]] {
        &self.normals
    }

    /// Advances by about `budget` units (at least one): the normals are
    /// zeroed and the triangles copied (16 vertex slots or 4 triangle slots
    /// a unit), the face normals of the alive triangles summed (4 triangle
    /// slots a unit), then every normal normalised (16 a unit). Returns
    /// whether the normals are done.
    ///
    /// # Errors
    /// Out of memory (both arrays are reserved at full size first); the
    /// build is unusable after it.
    pub(crate) fn step(&mut self, mesh: &Mesh, budget: usize) -> Result<bool> {
        let mut left = budget.max(1);
        let (slots, tri_slots) = (mesh.vertex_slots(), mesh.triangle_slots());
        // `n` units of work at `per` items each, from `from` to at most `end`.
        let take = |left: &mut usize, from: usize, end: usize, per: usize| {
            let to = end.min(from.saturating_add(left.saturating_mul(per)));
            *left = left.saturating_sub((to - from).div_ceil(per));
            to
        };
        loop {
            match self.stage {
                NormalStage::Zero(from) => {
                    if from == 0 {
                        self.normals.try_reserve_exact(slots).map_err(|_| oom())?;
                        self.triangles
                            .try_reserve_exact(tri_slots)
                            .map_err(|_| oom())?;
                    }
                    // Zero the normals first, then copy the triangles: one
                    // cursor over both, the triangles past `slots`.
                    let end = slots + tri_slots;
                    let to = if from < slots {
                        take(&mut left, from, slots, VERTICES_PER_UNIT)
                    } else {
                        take(&mut left, from, end, TRIANGLES_PER_UNIT)
                    };
                    if from < slots {
                        self.normals.resize(to, [0.0; 3]);
                    } else {
                        self.triangles
                            .extend_from_slice(&mesh.triangles()[from - slots..to - slots]);
                    }
                    self.stage = if to == end {
                        NormalStage::Sum(0)
                    } else {
                        NormalStage::Zero(to)
                    };
                }
                NormalStage::Sum(from) => {
                    let to = take(&mut left, from, tri_slots, TRIANGLES_PER_UNIT);
                    let p = mesh.positions();
                    for (t, &tri) in self.triangles.iter().enumerate().take(to).skip(from) {
                        if mesh.is_triangle_alive(t as TriangleId) {
                            let n = face_normal(p, tri);
                            for v in tri {
                                add(&mut self.normals[v as usize], n);
                            }
                        }
                    }
                    self.stage = if to == tri_slots {
                        NormalStage::Normalize(0)
                    } else {
                        NormalStage::Sum(to)
                    };
                }
                NormalStage::Normalize(from) => {
                    let to = take(&mut left, from, slots, VERTICES_PER_UNIT);
                    for n in &mut self.normals[from..to] {
                        *n = normalized(*n);
                    }
                    self.stage = if to == slots {
                        NormalStage::Done
                    } else {
                        NormalStage::Normalize(to)
                    };
                }
                NormalStage::Done => return Ok(true),
            }
            if left == 0 {
                return Ok(self.is_done());
            }
        }
    }

    /// The normals once done, as [`NormalState::new`] computes them, with
    /// nothing pending; `None` before.
    pub(crate) fn finish(self) -> Option<NormalState> {
        self.is_done().then_some(NormalState {
            normals: self.normals,
            triangles: self.triangles,
            pending: Some(ChangeSet::default()),
        })
    }
}

impl Entry {
    /// Reports vertices with new normals to every subscriber (`None`: all).
    pub(crate) fn report_normals(&mut self, ids: Option<&[VertexId]>) {
        for r in &mut self.reports {
            r.normals_changed(ids);
        }
    }

    /// Brings the normals up to date with their pending changes and reports
    /// the recomputed vertices. Does nothing when normals are off. On error
    /// the pending changes are kept, so a later call can retry.
    pub(crate) fn update_normals(&mut self) -> Result<()> {
        let Some(ns) = &mut self.normals else {
            return Ok(());
        };
        match ns.pending.take() {
            None => {
                if let Err(e) = ns.compute_all(&self.mesh) {
                    ns.pending = None;
                    return Err(e);
                }
                self.report_normals(None);
            }
            Some(c) if c.is_empty() && ns.normals.len() == self.mesh.vertex_slots() => {
                ns.pending = Some(c);
            }
            Some(c) => match ns.compute_changed(&mut self.mesh, &c) {
                Ok(dirty) => self.report_normals(Some(&dirty)),
                Err(e) => {
                    ns.pending = Some(c);
                    return Err(e);
                }
            },
        }
        Ok(())
    }
}

impl Scene {
    /// Turns the mesh's vertex normals on or off (off by default). Turning
    /// them on computes them right away, whatever the mesh's
    /// [`UpdateMode`](crate::UpdateMode); either switch is reported as
    /// `normals_dirty: None`. Setting the current state again does nothing.
    ///
    /// While on, the normals follow the mesh like its BVH: an Eager mesh's
    /// right after every edit, undo, redo and cancel, a Lazy one's on
    /// [`Scene::flush`] or [`Scene::update`], a Manual one's on
    /// [`Scene::update`] only. A compaction recomputes them in full right
    /// away. Placement changes leave them alone: they are mesh-local.
    ///
    /// Memory: besides 12 bytes of normal per vertex slot, a snapshot of the
    /// triangles (12 bytes per triangle slot) is kept while normals are on,
    /// so the old vertices of removed and rewritten triangles are known.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`], or [`SceneError::Store`] when there is
    /// no memory for them. The scene is unchanged on error.
    pub fn set_normals(&mut self, id: MeshId, on: bool) -> Result<()> {
        self.drop_field_unit();
        let entry = self.entries.get_mut(id).ok_or(SceneError::UnknownMesh)?;
        if on == entry.normals.is_some() {
            return Ok(());
        }
        entry.normals = if on {
            Some(NormalState::new(&entry.mesh)?)
        } else {
            None
        };
        entry.report_normals(None);
        Ok(())
    }

    /// The mesh's vertex normals in mesh-local space, one per vertex slot,
    /// or `None` when normals are off. A normal is the normalised sum of the
    /// area-weighted normals of the alive triangles using the vertex; a
    /// vertex used by none (or only by degenerate ones), dead slots
    /// included, reads `[0, 0, 0]`.
    ///
    /// The normals of a Lazy or Manual mesh are those of its last update
    /// (new vertex slots read zero until then). The slice may differ in
    /// length from `mesh.vertex_slots()` until the next successful update:
    /// shorter if memory ran out while following an edit, or longer if the
    /// recomputation after a compaction failed.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id.
    pub fn normals(&self, id: MeshId) -> Result<Option<&[[f32; 3]]>> {
        Ok(self.entry(id)?.normals.as_ref().map(NormalState::normals))
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::{Mesh, TriangleId, VertexId};

    use crate::{MeshId, MeshOptions, Scene, SceneError, SceneOptions, UpdateMode};

    /// An `n` x `n` grid of unit squares with a wave in z, so that normals
    /// differ from vertex to vertex.
    fn wavy(n: u32) -> Mesh {
        let mut pos = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                let (fx, fy) = (x as f32, y as f32);
                pos.extend_from_slice(&[fx, fy, 0.3 * (fx * 0.9).sin() * (fy * 0.7).cos()]);
            }
        }
        let mut idx = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                let (b, c) = (a + 1, a + n + 1);
                idx.extend_from_slice(&[a, b, c, b, c + 1, c]);
            }
        }
        Mesh::from_buffers(&pos, &idx).unwrap()
    }

    fn scene_with(mode: UpdateMode) -> (Scene, MeshId) {
        let mut scene = Scene::new(SceneOptions::default());
        let options = MeshOptions {
            mode,
            ..MeshOptions::default()
        };
        let id = scene.add_mesh(wavy(6), options).unwrap();
        (scene, id)
    }

    /// `fast_mesh::compute_vertex_normals` over the alive triangles.
    fn reference(mesh: &Mesh) -> Vec<[f32; 3]> {
        let flat: Vec<f32> = mesh.positions().iter().flatten().copied().collect();
        let idx: Vec<u32> = (0..mesh.triangle_slots() as TriangleId)
            .filter_map(|t| mesh.triangle(t))
            .flatten()
            .collect();
        let n = fast_mesh::compute_vertex_normals(&flat, &idx).unwrap();
        n.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
    }

    fn assert_matches(scene: &Scene, id: MeshId) {
        let mesh = scene.mesh(id).unwrap();
        let got = scene.normals(id).unwrap().expect("normals are on");
        let want = reference(mesh);
        assert_eq!(got.len(), mesh.vertex_slots());
        for (v, (g, w)) in got.iter().zip(&want).enumerate() {
            for k in 0..3 {
                assert!((g[k] - w[k]).abs() <= 1e-6, "vertex {v}: {g:?} vs {w:?}");
            }
        }
    }

    fn normals_dirty(scene: &mut Scene, id: MeshId) -> Option<Vec<VertexId>> {
        scene
            .take_changes()
            .into_iter()
            .find(|c| c.id == id)
            .map_or(Some(Vec::new()), |c| c.normals_dirty)
    }

    fn move_by(scene: &mut Scene, id: MeshId, v: VertexId, d: [f32; 3]) {
        scene
            .edit(id, |tx| {
                let p = tx.mesh().position(v).unwrap();
                tx.set_positions(&[v], &[std::array::from_fn(|k| p[k] + d[k])])
            })
            .unwrap();
    }

    /// Every vertex of every alive triangle using `v`, ascending.
    fn ring_of_ring(mesh: &Mesh, v: VertexId) -> Vec<VertexId> {
        let mut out: Vec<VertexId> = (0..mesh.triangle_slots() as TriangleId)
            .filter_map(|t| mesh.triangle(t))
            .filter(|tri| tri.contains(&v))
            .flatten()
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    #[test]
    fn normals_match_compute_vertex_normals_on_add() {
        let (mut scene, id) = scene_with(UpdateMode::Eager);
        scene.set_normals(id, true).unwrap();
        assert_matches(&scene, id);
        let n = scene.normals(id).unwrap().unwrap();
        assert!(n.iter().all(|n| n[2] > 0.0), "upward wound grid");
    }

    #[test]
    fn normals_follow_vertex_move() {
        let (mut scene, id) = scene_with(UpdateMode::Eager);
        scene.set_normals(id, true).unwrap();
        scene.take_changes();
        let v = 3 * 7 + 3;
        move_by(&mut scene, id, v, [0.1, -0.2, 0.7]);
        let want = ring_of_ring(scene.mesh(id).unwrap(), v);
        assert!(want.len() > 1 && want.len() < 20, "{want:?}");
        assert_eq!(normals_dirty(&mut scene, id), Some(want));
        assert_matches(&scene, id);
        // An undo reports the same set and restores the values.
        scene.undo().unwrap();
        let want = ring_of_ring(scene.mesh(id).unwrap(), v);
        assert_eq!(normals_dirty(&mut scene, id), Some(want));
        assert_matches(&scene, id);
        // Topology edits report the old and new triangles' vertices.
        let mesh = scene.mesh(id).unwrap();
        let old = mesh.triangle(10).unwrap();
        scene.edit(id, |tx| tx.remove_triangles(&[10])).unwrap();
        let mut want = old.to_vec();
        want.sort_unstable();
        assert_eq!(normals_dirty(&mut scene, id), Some(want));
        assert_matches(&scene, id);
        let mesh = scene.mesh(id).unwrap();
        let t = (0..mesh.triangle_slots() as TriangleId)
            .find(|&t| mesh.is_triangle_alive(t))
            .unwrap();
        let [a, b, c] = mesh.triangle(t).unwrap();
        let far = 6 * 7 + 6;
        scene
            .edit(id, |tx| tx.set_triangles(&[t], &[[a, b, far]]))
            .unwrap();
        let mut want = vec![a, b, c, far];
        want.sort_unstable();
        want.dedup();
        assert_eq!(normals_dirty(&mut scene, id), Some(want));
        assert_matches(&scene, id);
        // A placement leaves normals alone.
        let mut p = scene.placement(id).unwrap();
        p.translation = [5.0, 0.0, 0.0];
        scene.set_placement(id, p).unwrap();
        assert_eq!(normals_dirty(&mut scene, id), Some(Vec::new()));
    }

    #[test]
    fn normal_build_in_steps_matches_compute_all() {
        // Dead triangles and a vertex no triangle uses: their slots are
        // skipped and stay zero, as in one call.
        let (mut scene, id) = scene_with(UpdateMode::Eager);
        scene
            .edit(id, |tx| {
                tx.remove_triangles(&[0, 1, 7, 30])?;
                tx.add_vertices(&[[9.0, 9.0, 9.0]])
            })
            .unwrap();
        let mesh = scene.mesh(id).unwrap();
        let want = super::NormalState::new(mesh).unwrap();
        let bits =
            |n: &[[f32; 3]]| -> Vec<u32> { n.iter().flatten().map(|x| x.to_bits()).collect() };
        for budget in [1, 2, 7, 100, usize::MAX] {
            let mut build = super::NormalBuild::new();
            let mut steps = 0;
            while !build.step(mesh, budget).unwrap() {
                steps += 1;
            }
            if budget == 1 {
                assert!(steps > mesh.triangle_slots() / 8, "{steps}");
            }
            assert!(build.step(mesh, budget).unwrap(), "stays done");
            let got = build.finish().unwrap();
            assert_eq!(bits(got.normals()), bits(want.normals()), "budget {budget}");
            assert_eq!(got.triangles, want.triangles);
            assert_eq!(got.pending, Some(fast_mesh_store::ChangeSet::default()));
        }
        // Unfinished, it gives nothing.
        let mut build = super::NormalBuild::new();
        build.step(mesh, 1).unwrap();
        assert!(build.finish().is_none());
    }

    #[test]
    fn normals_off_by_default_and_none() {
        let (mut scene, id) = scene_with(UpdateMode::Eager);
        assert_eq!(scene.normals(id), Ok(None));
        assert_eq!(normals_dirty(&mut scene, id), None, "added");
        move_by(&mut scene, id, 5, [0.0, 0.0, 1.0]);
        assert_eq!(normals_dirty(&mut scene, id), Some(Vec::new()), "off");
        scene.set_normals(id, true).unwrap();
        assert_eq!(normals_dirty(&mut scene, id), None);
        scene.set_normals(id, true).unwrap();
        assert!(scene.take_changes().is_empty(), "no change");
        scene.set_normals(id, false).unwrap();
        assert_eq!(scene.normals(id), Ok(None));
        assert_eq!(normals_dirty(&mut scene, id), None);
        scene.remove_mesh(id).unwrap();
        assert_eq!(scene.set_normals(id, true), Err(SceneError::UnknownMesh));
        assert_eq!(scene.normals(id), Err(SceneError::UnknownMesh));
    }

    #[test]
    fn normals_after_compact_all() {
        for mode in [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual] {
            let (mut scene, id) = scene_with(mode);
            scene.set_normals(id, true).unwrap();
            let mesh = scene.mesh(id).unwrap();
            let doomed: Vec<TriangleId> = (0..mesh.triangle_slots() as TriangleId)
                .filter(|&t| mesh.triangle(t).unwrap().contains(&0))
                .collect();
            scene
                .edit(id, |tx| {
                    tx.remove_triangles(&doomed)?;
                    tx.remove_vertices(&[0])
                })
                .unwrap();
            scene.take_changes();
            scene.compact(id).unwrap();
            assert_eq!(scene.mesh(id).unwrap().vertex_slots(), 48);
            assert_eq!(normals_dirty(&mut scene, id), None, "{mode:?}");
            assert_matches(&scene, id);
        }
    }

    #[test]
    fn lazy_normals_update_on_flush() {
        let (mut scene, id) = scene_with(UpdateMode::Lazy);
        scene.set_normals(id, true).unwrap();
        scene.take_changes();
        let before = scene.normals(id).unwrap().unwrap().to_vec();
        move_by(&mut scene, id, 10, [0.0, 0.0, 0.8]);
        let added = scene
            .edit(id, |tx| {
                let v = tx.add_vertices(&[[0.5, 0.5, 2.0], [9.0, 9.0, 9.0]])?;
                tx.add_triangles(&[[0, 1, v[0]]])?;
                Ok::<_, SceneError>(v)
            })
            .unwrap();
        let slots = scene.mesh(id).unwrap().vertex_slots();
        let now = scene.normals(id).unwrap().unwrap();
        assert_eq!(now.len(), slots, "grown with the slots");
        assert_eq!(&now[..before.len()], &before[..], "stale until flushed");
        assert_eq!(normals_dirty(&mut scene, id), Some(Vec::new()));
        scene.flush().unwrap();
        assert_matches(&scene, id);
        let mesh = scene.mesh(id).unwrap();
        let mut want = ring_of_ring(mesh, 10);
        want.extend([0, 1, added[0], added[1]]);
        want.sort_unstable();
        want.dedup();
        assert_eq!(normals_dirty(&mut scene, id), Some(want));
        assert_eq!(
            scene.normals(id).unwrap().unwrap()[added[1] as usize],
            [0.0; 3]
        );

        // Manual: flush leaves the normals alone, `update` brings them up.
        scene.set_mode(id, UpdateMode::Manual).unwrap();
        move_by(&mut scene, id, 20, [0.0, 0.3, 0.0]);
        scene.flush().unwrap();
        assert_eq!(normals_dirty(&mut scene, id), Some(Vec::new()));
        scene.update(id).unwrap();
        assert_eq!(
            normals_dirty(&mut scene, id),
            Some(ring_of_ring(scene.mesh(id).unwrap(), 20))
        );
        assert_matches(&scene, id);
        // Switching to Eager catches up at once.
        move_by(&mut scene, id, 30, [0.2, 0.0, 0.0]);
        scene.set_mode(id, UpdateMode::Eager).unwrap();
        assert_matches(&scene, id);
    }
}
