//! Corner-table adjacency.
//!
//! Corner `c = 3 * t + i` is the `i`-th corner of triangle `t`. The edge opposite a
//! corner joins the triangle's other two vertices. For every corner the table stores
//! the corner across that edge in the neighboring triangle, or marks the edge as a
//! boundary (no neighbor) or non-manifold (more than one neighbor). Orientation is not
//! assumed to be consistent.

use std::collections::HashSet;

use crate::column::{try_reserve_exact, try_vec};
use crate::{Mesh, Result, StoreError, TriangleId, VertexId};

const NONE: u32 = u32::MAX;

/// Entries an [`AdjacencyBuild`] or [`IncidenceBuild`](crate::IncidenceBuild)
/// step zeroes per element of its budget.
pub(crate) const ZERO_PER_ELEMENT: usize = 16;
const NON_MANIFOLD: u32 = u32::MAX - 1;

/// Kind of the edge opposite a corner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// Shared by exactly two triangles.
    Interior,
    /// Used by one triangle only.
    Boundary,
    /// Shared by three or more triangles.
    NonManifold,
}

/// Corner table of a [`Mesh`]: for every triangle corner, the corner across the
/// opposite edge, plus one corner per vertex and a non-manifold flag per vertex.
///
/// Built by [`Mesh::ensure_adjacency`] and cached until the next topology change.
/// See the module docs for the corner numbering.
#[derive(Debug)]
pub struct Adjacency {
    opposite: Vec<u32>,
    vertex_corner: Vec<u32>,
    non_manifold_vertices: Vec<u64>,
}

impl Adjacency {
    pub(crate) fn build(mesh: &Mesh) -> Result<Self> {
        let mut build = AdjacencyBuild::new(mesh);
        build.step(mesh, usize::MAX)?;
        build.take().ok_or(StoreError::OutOfMemory)
    }

    /// The corner across the edge opposite `corner`, if that edge is interior.
    pub fn opposite(&self, corner: u32) -> Option<u32> {
        self.opposite
            .get(corner as usize)
            .copied()
            .filter(|&o| o < NON_MANIFOLD)
    }

    /// Kind of the edge opposite `corner`. Corners of dead triangles read as boundary.
    pub fn edge_kind(&self, corner: u32) -> EdgeKind {
        match self.opposite.get(corner as usize).copied().unwrap_or(NONE) {
            NONE => EdgeKind::Boundary,
            NON_MANIFOLD => EdgeKind::NonManifold,
            _ => EdgeKind::Interior,
        }
    }

    /// One corner at vertex `v`, or `None` if no triangle uses it.
    pub fn vertex_corner(&self, v: VertexId) -> Option<u32> {
        self.vertex_corner
            .get(v as usize)
            .copied()
            .filter(|&c| c != NONE)
    }

    /// True when the triangles around `v` do not form a single fan: two surfaces meet
    /// only at `v` (a "bowtie"), or an edge at `v` is non-manifold.
    pub fn is_non_manifold_vertex(&self, v: VertexId) -> bool {
        self.non_manifold_vertices
            .get(v as usize / 64)
            .is_some_and(|w| (w >> (v % 64)) & 1 == 1)
    }

    /// Triangles in the fan of `v` reached from [`Adjacency::vertex_corner`] through
    /// interior edges. For a non-manifold vertex this is one fan, not every triangle.
    pub fn triangles_around(&self, mesh: &Mesh, v: VertexId) -> Vec<TriangleId> {
        let Some(start) = self.vertex_corner(v) else {
            return Vec::new();
        };
        let tris = mesh.triangles();
        let mut fan = vec![start / 3];
        let mut visited = HashSet::from([start / 3]);
        let mut stack = vec![start];
        while let Some(c) = stack.pop() {
            let (t, i) = (c / 3, c % 3);
            for e in [(i + 1) % 3, (i + 2) % 3] {
                let Some(o) = self.opposite(3 * t + e) else {
                    continue;
                };
                let s = o / 3;
                if visited.contains(&s) {
                    continue;
                }
                if let Some(j) = tris[s as usize].iter().position(|&x| x == v) {
                    visited.insert(s);
                    fan.push(s);
                    stack.push(3 * s + j as u32);
                }
            }
        }
        fan
    }
}

/// Where an [`AdjacencyBuild`] stands: the pass it is in, with that pass's
/// cursor (a triangle or vertex slot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pass {
    /// Counting the corners of each vertex, triangle by triangle.
    Count(usize),
    /// Turning the counts into row offsets, vertex by vertex.
    Prefix(usize),
    /// Zeroing the corner rows, up to this many entries.
    Zero(usize),
    /// Listing each vertex's corners, triangle by triangle.
    Fill(usize),
    /// Finding each corner's opposite corner, triangle by triangle.
    Opposite(usize),
    /// A corner and the fan count of each vertex, vertex by vertex.
    Fans(usize),
    /// Finished: the table is ready to be taken.
    Done,
}

/// A resumable build of a mesh's [`Adjacency`]: the build of
/// [`Mesh::ensure_adjacency`] cut into steps of a bounded number of elements
/// (triangle or vertex slots), so that a caller can yield between them.
///
/// Each step runs the same loop bodies as the one-call build, over the next
/// elements in the same order, so the finished table is the same, entry for
/// entry. A build is tied to the mesh state it started from (its uid, epoch
/// and revision); [`Mesh::step_adjacency`] starts over when the mesh has
/// changed since, and caches the table when the build is done.
#[derive(Debug)]
pub struct AdjacencyBuild {
    uid: u64,
    epoch: u32,
    revision: u64,
    pass: Pass,
    offsets: Vec<u32>,
    incident: Vec<u32>,
    cursor: Vec<u32>,
    opposite: Vec<u32>,
    vertex_corner: Vec<u32>,
    non_manifold_vertices: Vec<u64>,
    max_valence: u32,
    scratch: FanScratch,
}

impl AdjacencyBuild {
    /// A build of `mesh`'s adjacency as the mesh is now, with nothing done.
    pub fn new(mesh: &Mesh) -> AdjacencyBuild {
        AdjacencyBuild {
            uid: mesh.uid(),
            epoch: mesh.epoch(),
            revision: mesh.revision(),
            pass: Pass::Count(0),
            offsets: Vec::new(),
            incident: Vec::new(),
            cursor: Vec::new(),
            opposite: Vec::new(),
            vertex_corner: Vec::new(),
            non_manifold_vertices: Vec::new(),
            max_valence: 0,
            scratch: FanScratch {
                by_triangle: Vec::new(),
                parent: Vec::new(),
            },
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
    /// it is done) of `mesh`, which must be the mesh state it started from
    /// (see [`AdjacencyBuild::is_for`]). Returns whether it is done.
    fn step(&mut self, mesh: &Mesh, budget: usize) -> Result<bool> {
        debug_assert!(
            self.is_for(mesh),
            "an adjacency build stepped on another mesh"
        );
        let tris = mesh.triangles();
        let alive = |t: usize| mesh.is_triangle_alive(t as TriangleId);
        let vslots = mesh.vertex_slots();
        let mut left = budget.max(1);
        while left > 0 {
            match self.pass {
                Pass::Count(from) => {
                    if from == 0 {
                        // Incident corners per vertex, in compressed rows.
                        self.offsets = try_vec(vslots + 1, 0u32)?;
                    }
                    let to = tris.len().min(from.saturating_add(left));
                    for (t, tri) in tris.iter().enumerate().take(to).skip(from) {
                        if alive(t) {
                            for &v in tri {
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
                    let to = vslots.min(from.saturating_add(left));
                    for v in from..to {
                        self.offsets[v + 1] += self.offsets[v];
                        let valence = self.offsets[v + 1] - self.offsets[v];
                        self.max_valence = self.max_valence.max(valence);
                    }
                    left -= (to - from).min(left);
                    if to < vslots {
                        self.pass = Pass::Prefix(to);
                        continue;
                    }
                    try_reserve_exact(&mut self.incident, self.offsets[vslots] as usize)?;
                    self.pass = Pass::Zero(0);
                }
                Pass::Zero(from) => {
                    // Zeroing is cheap: many entries per element of the budget.
                    let total = self.offsets[vslots] as usize;
                    let to = total.min(from.saturating_add(left.saturating_mul(ZERO_PER_ELEMENT)));
                    self.incident.resize(to, 0);
                    left -= (to - from).div_ceil(ZERO_PER_ELEMENT).min(left);
                    if to < total {
                        self.pass = Pass::Zero(to);
                        continue;
                    }
                    self.cursor = try_vec(vslots + 1, 0u32)?;
                    self.cursor.copy_from_slice(&self.offsets);
                    self.pass = Pass::Fill(0);
                }
                Pass::Fill(from) => {
                    let to = tris.len().min(from.saturating_add(left));
                    for (t, tri) in tris.iter().enumerate().take(to).skip(from) {
                        if alive(t) {
                            for (i, &v) in tri.iter().enumerate() {
                                let at = &mut self.cursor[v as usize];
                                self.incident[*at as usize] = (3 * t + i) as u32;
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
                    // Filled in corner order by the next pass.
                    try_reserve_exact(&mut self.opposite, 3 * tris.len())?;
                    self.pass = Pass::Opposite(0);
                }
                Pass::Opposite(from) => {
                    let to = tris.len().min(from.saturating_add(left));
                    let (offsets, incident) = (&self.offsets, &self.incident);
                    let corners_of = |v: u32| {
                        &incident[offsets[v as usize] as usize..offsets[v as usize + 1] as usize]
                    };
                    for (t, tri) in tris.iter().enumerate().take(to).skip(from) {
                        if !alive(t) {
                            self.opposite.extend([NONE; 3]);
                            continue;
                        }
                        for i in 0..3 {
                            let (u, w) = (tri[(i + 1) % 3], tri[(i + 2) % 3]);
                            if u == w {
                                self.opposite.push(NONE);
                                continue;
                            }
                            // Triangles on edge u-w are incident to both ends: scan the
                            // shorter list, so a high-valence vertex is not scanned once
                            // per edge.
                            let (scan, other_end) = if corners_of(u).len() <= corners_of(w).len() {
                                (u, w)
                            } else {
                                (w, u)
                            };
                            let (mut count, mut found, mut last) = (0, NONE, NONE);
                            for &d in corners_of(scan) {
                                let s = d / 3;
                                if s as usize == t || s == last {
                                    continue;
                                }
                                last = s;
                                let other = tris[s as usize];
                                if !other.contains(&other_end) {
                                    continue;
                                }
                                count += 1;
                                found = (0..3)
                                    .find(|&j| other[j] != u && other[j] != w)
                                    .map_or(NON_MANIFOLD, |j| 3 * s + j as u32);
                            }
                            self.opposite.push(match count {
                                0 => NONE,
                                1 => found,
                                _ => NON_MANIFOLD,
                            });
                        }
                    }
                    left -= (to - from).min(left);
                    if to < tris.len() {
                        self.pass = Pass::Opposite(to);
                        continue;
                    }
                    self.vertex_corner = try_vec(vslots, NONE)?;
                    self.non_manifold_vertices = try_vec(vslots.div_ceil(64), 0u64)?;
                    // Scratch buffers for `fan_count`, sized for the highest valence once.
                    let most = self.max_valence as usize;
                    try_reserve_exact(&mut self.scratch.by_triangle, most)?;
                    try_reserve_exact(&mut self.scratch.parent, most)?;
                    self.pass = Pass::Fans(0);
                }
                Pass::Fans(from) => {
                    let to = vslots.min(from.saturating_add(left));
                    for v in from as u32..to as u32 {
                        let (lo, hi) = (self.offsets[v as usize], self.offsets[v as usize + 1]);
                        let corners = &self.incident[lo as usize..hi as usize];
                        if let Some(&c) = corners.first() {
                            self.vertex_corner[v as usize] = c;
                        }
                        if !corners.is_empty()
                            && fan_count(corners, &self.opposite, &mut self.scratch) != 1
                        {
                            self.non_manifold_vertices[v as usize / 64] |= 1 << (v % 64);
                        }
                    }
                    left -= (to - from).min(left);
                    if to < vslots {
                        self.pass = Pass::Fans(to);
                        continue;
                    }
                    self.offsets = Vec::new();
                    self.incident = Vec::new();
                    self.pass = Pass::Done;
                }
                Pass::Done => return Ok(true),
            }
        }
        Ok(self.pass == Pass::Done)
    }

    /// The finished table, once [`AdjacencyBuild::step`] returned `true`.
    fn take(&mut self) -> Option<Adjacency> {
        (self.pass == Pass::Done).then(|| Adjacency {
            opposite: std::mem::take(&mut self.opposite),
            vertex_corner: std::mem::take(&mut self.vertex_corner),
            non_manifold_vertices: std::mem::take(&mut self.non_manifold_vertices),
        })
    }
}

/// Reusable buffers for [`fan_count`], allocated once per build.
#[derive(Debug)]
struct FanScratch {
    /// `(triangle, local corner index)`, sorted, to find a triangle's corner fast.
    by_triangle: Vec<(u32, u32)>,
    /// Union-find parents, one per corner at the vertex.
    parent: Vec<u32>,
}

/// Number of fans the corners at one vertex form when joined across interior edges.
/// A non-manifold edge at the vertex makes the result 0 (never "one fan").
/// Runs in O(k log k) for k corners; the scratch buffers hold at least k entries.
fn fan_count(corners: &[u32], opposite: &[u32], scratch: &mut FanScratch) -> usize {
    let FanScratch {
        by_triangle,
        parent,
    } = scratch;
    by_triangle.clear();
    by_triangle.extend(corners.iter().enumerate().map(|(a, &c)| (c / 3, a as u32)));
    by_triangle.sort_unstable();
    parent.clear();
    parent.extend(0..corners.len() as u32);
    fn root(parent: &mut [u32], mut a: u32) -> u32 {
        while parent[a as usize] != a {
            parent[a as usize] = parent[parent[a as usize] as usize];
            a = parent[a as usize];
        }
        a
    }
    for (a, &c) in corners.iter().enumerate() {
        let (t, i) = (c / 3, c % 3);
        for e in [(i + 1) % 3, (i + 2) % 3] {
            match opposite[(3 * t + e) as usize] {
                NON_MANIFOLD => return 0,
                NONE => {}
                o => {
                    // First corner of the neighboring triangle at this vertex.
                    let s = o / 3;
                    let k = by_triangle.partition_point(|&(tri, _)| tri < s);
                    if let Some(&(tri, b)) = by_triangle.get(k)
                        && tri == s
                    {
                        let (ra, rb) = (root(parent, a as u32), root(parent, b));
                        parent[ra as usize] = rb;
                    }
                }
            }
        }
    }
    (0..corners.len() as u32)
        .filter(|&a| root(parent, a) == a)
        .count()
}

impl Mesh {
    /// Builds the adjacency table if it is not cached. It is cached until the next
    /// topology change (commit, undo or redo that adds, removes or rewrites triangles).
    pub fn ensure_adjacency(&mut self) -> Result<()> {
        if self.adjacency.is_none() {
            self.adjacency = Some(Adjacency::build(self)?);
        }
        Ok(())
    }

    /// The cached adjacency table, if [`Mesh::ensure_adjacency`] built it and no
    /// topology change has happened since.
    pub fn adjacency(&self) -> Option<&Adjacency> {
        self.adjacency.as_ref()
    }

    /// [`Mesh::ensure_adjacency`] in steps: runs `build` over up to `budget`
    /// more elements and, once it is done, caches its table. Returns whether
    /// the adjacency is cached, at once when it already is. When the mesh
    /// changed since `build` started, `build` starts over first. The table is
    /// the same as from [`Mesh::ensure_adjacency`], entry for entry.
    ///
    /// # Errors
    /// As for [`Mesh::ensure_adjacency`]; `build` then starts over.
    pub fn step_adjacency(&mut self, build: &mut AdjacencyBuild, budget: usize) -> Result<bool> {
        if self.adjacency.is_some() {
            return Ok(true);
        }
        if !build.is_for(self) {
            *build = AdjacencyBuild::new(self);
        }
        match build.step(self, budget) {
            Ok(done) => {
                if done {
                    self.adjacency = build.take();
                }
                Ok(done)
            }
            Err(e) => {
                *build = AdjacencyBuild::new(self);
                Err(e)
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Two triangles sharing edge 0-2: 0-1-2 and 0-2-3.
    fn quad() -> Mesh {
        Mesh::from_buffers(
            &[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap()
    }

    #[test]
    fn finds_the_shared_edge() {
        let mut m = quad();
        m.ensure_adjacency().unwrap();
        let a = m.adjacency().unwrap();
        // Corner 1 (vertex 1 of triangle 0) faces edge 2-0, shared with triangle 1,
        // whose corner facing that edge is vertex 3 (corner 5).
        assert_eq!(a.opposite(1), Some(5));
        assert_eq!(a.opposite(5), Some(1));
        assert_eq!(a.edge_kind(0), EdgeKind::Boundary);
        assert_eq!(a.edge_kind(1), EdgeKind::Interior);
        assert!(!a.is_non_manifold_vertex(0));
        let mut around = a.triangles_around(&m, 0);
        around.sort();
        assert_eq!(around, vec![0, 1]);
    }

    #[test]
    fn three_triangles_on_one_edge_are_non_manifold() {
        let mut m = Mesh::from_buffers(
            &[0., 0., 0., 1., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 1.],
            &[0, 1, 2, 0, 1, 3, 0, 1, 4],
        )
        .unwrap();
        m.ensure_adjacency().unwrap();
        let a = m.adjacency().unwrap();
        assert_eq!(a.edge_kind(2), EdgeKind::NonManifold);
        assert_eq!(a.opposite(2), None);
        assert!(a.is_non_manifold_vertex(0));
        assert!(!a.is_non_manifold_vertex(2));
    }

    #[test]
    fn bowtie_vertex_is_non_manifold() {
        // Two triangles touching only at vertex 0.
        let mut m = Mesh::from_buffers(
            &[
                0., 0., 0., 1., 0., 0., 1., 1., 0., -1., 0., 0., -1., -1., 0.,
            ],
            &[0, 1, 2, 0, 3, 4],
        )
        .unwrap();
        m.ensure_adjacency().unwrap();
        let a = m.adjacency().unwrap();
        assert!(a.is_non_manifold_vertex(0));
        assert_eq!(
            a.triangles_around(&m, 0).len(),
            1,
            "only one fan is reachable"
        );
    }

    #[test]
    fn inconsistent_orientation_still_connects() {
        // Same quad, second triangle wound the other way.
        let mut m = Mesh::from_buffers(
            &[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
            &[0, 1, 2, 0, 3, 2],
        )
        .unwrap();
        m.ensure_adjacency().unwrap();
        assert_eq!(m.adjacency().unwrap().opposite(1), Some(4));
    }

    #[test]
    fn high_valence_fan_is_built_correctly() {
        // Open fan around vertex 0: triangles (0, i, i + 1) for rim vertices 1..=N.
        const N: u32 = 20_000;
        let positions = vec![0.0f32; 3 * (N as usize + 1)];
        let indices: Vec<u32> = (1..N).flat_map(|i| [0, i, i + 1]).collect();
        let mut m = Mesh::from_buffers(&positions, &indices).unwrap();
        m.ensure_adjacency().unwrap();
        let a = m.adjacency().unwrap();
        let tris = N - 1;
        for t in 0..tris {
            // Corner 0 (the center) faces the rim edge: boundary.
            assert_eq!(a.edge_kind(3 * t), EdgeKind::Boundary);
            // Corner 1 (vertex t + 1) faces edge (t + 2, 0), shared with triangle t + 1,
            // whose corner there is vertex t + 3 (local 2).
            let next = (t + 1 < tris).then(|| 3 * (t + 1) + 2);
            assert_eq!(a.opposite(3 * t + 1), next, "triangle {t}");
            // Corner 2 (vertex t + 2) faces edge (0, t + 1), shared with triangle t - 1,
            // whose corner there is vertex t (local 1).
            let prev = (t > 0).then(|| 3 * (t - 1) + 1);
            assert_eq!(a.opposite(3 * t + 2), prev, "triangle {t}");
        }
        assert!(!a.is_non_manifold_vertex(0));
        assert!(!a.is_non_manifold_vertex(N / 2));
        let mut around = a.triangles_around(&m, 0);
        assert_eq!(around.len(), tris as usize);
        around.sort_unstable();
        assert!(around.iter().copied().eq(0..tris));
    }

    #[test]
    fn topology_changes_drop_the_cache_but_moves_keep_it() {
        let mut m = quad();
        m.ensure_adjacency().unwrap();
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[0., 0., 1.]]).unwrap();
        tx.commit().unwrap();
        assert!(m.adjacency().is_some());

        let mut tx = m.begin_edit();
        tx.remove_triangles(&[1]).unwrap();
        let patch = tx.commit().unwrap();
        assert!(m.adjacency().is_none());
        m.ensure_adjacency().unwrap();
        assert_eq!(m.adjacency().unwrap().edge_kind(1), EdgeKind::Boundary);

        patch.revert(&mut m).unwrap();
        assert!(m.adjacency().is_none());
    }

    /// A mixed mesh for the stepped builds: a grid with some triangles
    /// removed, a non-manifold edge, a bowtie vertex, a repeated-vertex
    /// triangle and an unused vertex.
    pub(crate) fn mixed() -> Mesh {
        let n = 12u32;
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                positions.extend([x as f32, y as f32, ((x * 7 + y * 3) % 5) as f32 * 0.1]);
            }
        }
        let mut indices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        let extra = (n + 1) * (n + 1);
        positions.extend([0., 0., 5., 3., 3., 5., -1., -1., 0., 50., 50., 50.]);
        // A third triangle on edge 0-1, a bowtie at vertex `n`, and a triangle
        // repeating a vertex.
        indices.extend([0, 1, extra, n, extra + 1, extra + 2, 5, 5, 6]);
        let mut m = Mesh::from_buffers(&positions, &indices).unwrap();
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[3, 40, 41, 100]).unwrap();
        tx.commit().unwrap();
        m
    }

    fn table(a: &Adjacency) -> (Vec<u32>, Vec<u32>, Vec<u64>) {
        (
            a.opposite.clone(),
            a.vertex_corner.clone(),
            a.non_manifold_vertices.clone(),
        )
    }

    #[test]
    fn stepped_build_matches_one_call() {
        let mut whole = mixed();
        whole.ensure_adjacency().unwrap();
        let reference = table(whole.adjacency().unwrap());
        for budget in [1, 2, 7, 64, 1000, usize::MAX] {
            let mut m = mixed();
            let mut build = AdjacencyBuild::new(&m);
            let mut steps = 0;
            while !m.step_adjacency(&mut build, budget).unwrap() {
                steps += 1;
                assert!(m.adjacency().is_none(), "cached before the last step");
            }
            assert!(build.is_done());
            assert_eq!(table(m.adjacency().unwrap()), reference, "budget {budget}");
            if budget == 1 {
                assert!(
                    steps > 2 * m.triangle_slots(),
                    "{steps} steps of one element"
                );
            }
            // Cached: nothing more to do.
            assert!(m.step_adjacency(&mut build, 1).unwrap());
        }
    }

    #[test]
    fn stepped_build_starts_over_after_an_edit() {
        let mut m = mixed();
        let mut build = AdjacencyBuild::new(&m);
        for _ in 0..50 {
            assert!(!m.step_adjacency(&mut build, 3).unwrap());
        }
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[10, 11]).unwrap();
        tx.commit().unwrap();
        assert!(!build.is_for(&m));
        while !m.step_adjacency(&mut build, 5).unwrap() {}
        let got = table(m.adjacency().unwrap());
        m.adjacency = None;
        m.ensure_adjacency().unwrap();
        assert_eq!(got, table(m.adjacency().unwrap()));
    }

    #[test]
    fn stepped_build_out_of_memory_starts_over() {
        use crate::column::fail_allocations_after;
        let mut m = mixed();
        let mut build = AdjacencyBuild::new(&m);
        assert!(!m.step_adjacency(&mut build, 4).unwrap());
        fail_allocations_after(Some(0));
        let mut failed = false;
        for _ in 0..10_000 {
            match m.step_adjacency(&mut build, 16) {
                Err(e) => {
                    assert_eq!(e, StoreError::OutOfMemory);
                    failed = true;
                    break;
                }
                Ok(done) => assert!(!done),
            }
        }
        fail_allocations_after(None);
        assert!(failed, "some pass allocates");
        assert!(m.adjacency().is_none());
        while !m.step_adjacency(&mut build, 64).unwrap() {}
        let got = table(m.adjacency().unwrap());
        m.adjacency = None;
        m.ensure_adjacency().unwrap();
        assert_eq!(got, table(m.adjacency().unwrap()));
    }
}
