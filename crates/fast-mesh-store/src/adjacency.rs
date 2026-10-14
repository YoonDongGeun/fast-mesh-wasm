//! Corner-table adjacency.
//!
//! Corner `c = 3 * t + i` is the `i`-th corner of triangle `t`. The edge opposite a
//! corner joins the triangle's other two vertices. For every corner the table stores
//! the corner across that edge in the neighboring triangle, or marks the edge as a
//! boundary (no neighbor) or non-manifold (more than one neighbor). Orientation is not
//! assumed to be consistent.

use crate::column::try_vec;
use crate::{Mesh, Result, TriangleId, VertexId};

const NONE: u32 = u32::MAX;
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

pub struct Adjacency {
    opposite: Vec<u32>,
    vertex_corner: Vec<u32>,
    non_manifold_vertices: Vec<u64>,
}

impl Adjacency {
    pub(crate) fn build(mesh: &Mesh) -> Result<Self> {
        let tris = mesh.triangles();
        let alive = |t: usize| mesh.is_triangle_alive(t as TriangleId);
        let vslots = mesh.vertex_slots();

        // Incident corners per vertex, in compressed rows.
        let mut offsets = try_vec(vslots + 1, 0u32)?;
        for (t, tri) in tris.iter().enumerate() {
            if alive(t) {
                for &v in tri {
                    offsets[v as usize + 1] += 1;
                }
            }
        }
        for v in 0..vslots {
            offsets[v + 1] += offsets[v];
        }
        let mut incident = try_vec(offsets[vslots] as usize, 0u32)?;
        let mut cursor = offsets.clone();
        for (t, tri) in tris.iter().enumerate() {
            if alive(t) {
                for (i, &v) in tri.iter().enumerate() {
                    incident[cursor[v as usize] as usize] = (3 * t + i) as u32;
                    cursor[v as usize] += 1;
                }
            }
        }
        let corners_of =
            |v: u32| &incident[offsets[v as usize] as usize..offsets[v as usize + 1] as usize];

        let mut opposite = try_vec(3 * tris.len(), NONE)?;
        for (t, tri) in tris.iter().enumerate() {
            if !alive(t) {
                continue;
            }
            for i in 0..3 {
                let (u, w) = (tri[(i + 1) % 3], tri[(i + 2) % 3]);
                if u == w {
                    continue;
                }
                let (mut count, mut found, mut last) = (0, NONE, NONE);
                for &d in corners_of(u) {
                    let s = d / 3;
                    if s as usize == t || s == last {
                        continue;
                    }
                    last = s;
                    let other = tris[s as usize];
                    if !other.contains(&w) {
                        continue;
                    }
                    count += 1;
                    found = (0..3)
                        .find(|&j| other[j] != u && other[j] != w)
                        .map_or(NON_MANIFOLD, |j| 3 * s + j as u32);
                }
                opposite[3 * t + i] = match count {
                    0 => NONE,
                    1 => found,
                    _ => NON_MANIFOLD,
                };
            }
        }

        let mut vertex_corner = try_vec(vslots, NONE)?;
        let mut non_manifold_vertices = try_vec(vslots.div_ceil(64), 0u64)?;
        for v in 0..vslots as u32 {
            let corners = corners_of(v);
            if let Some(&c) = corners.first() {
                vertex_corner[v as usize] = c;
            }
            if fan_count(corners, &opposite) != 1 && !corners.is_empty() {
                non_manifold_vertices[v as usize / 64] |= 1 << (v % 64);
            }
        }
        Ok(Self {
            opposite,
            vertex_corner,
            non_manifold_vertices,
        })
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
        let mut stack = vec![start];
        while let Some(c) = stack.pop() {
            let (t, i) = (c / 3, c % 3);
            for e in [(i + 1) % 3, (i + 2) % 3] {
                let Some(o) = self.opposite(3 * t + e) else {
                    continue;
                };
                let s = o / 3;
                if fan.contains(&s) {
                    continue;
                }
                if let Some(j) = tris[s as usize].iter().position(|&x| x == v) {
                    fan.push(s);
                    stack.push(3 * s + j as u32);
                }
            }
        }
        fan
    }
}

/// Number of fans the corners at one vertex form when joined across interior edges.
/// A non-manifold edge at the vertex makes the result 0 (never "one fan").
fn fan_count(corners: &[u32], opposite: &[u32]) -> usize {
    let mut parent: Vec<usize> = (0..corners.len()).collect();
    fn root(parent: &mut [usize], mut a: usize) -> usize {
        while parent[a] != a {
            parent[a] = parent[parent[a]];
            a = parent[a];
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
                    if let Some(b) = corners.iter().position(|&x| x / 3 == o / 3) {
                        let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                        parent[ra] = rb;
                    }
                }
            }
        }
    }
    (0..corners.len())
        .filter(|&a| root(&mut parent, a) == a)
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
}

#[cfg(test)]
mod tests {
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
}
