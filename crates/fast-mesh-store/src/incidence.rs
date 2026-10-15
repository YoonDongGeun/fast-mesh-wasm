//! Vertex → triangles incidence index.
//!
//! Lists, for every vertex slot, all alive triangles that use it. Unlike
//! [`crate::Adjacency::triangles_around`] it covers every fan of a non-manifold
//! vertex, which is what BVH refits need after vertices move.

use crate::column::try_vec;
use crate::{Mesh, Result, TriangleId, VertexId};

/// Every alive triangle using each vertex, stored in compressed rows.
#[derive(Debug)]
pub struct Incidence {
    /// `offsets[v]..offsets[v + 1]` is the range of `triangles` for vertex `v`.
    offsets: Vec<u32>,
    triangles: Vec<TriangleId>,
}

impl Incidence {
    pub(crate) fn build(mesh: &Mesh) -> Result<Self> {
        let tris = mesh.triangles();
        let slots = mesh.vertex_slots();
        // A triangle repeating a vertex ([a, a, b]) is listed once for it.
        let distinct = |tri: &[VertexId; 3]| {
            let [a, b, c] = *tri;
            [
                Some(a),
                (b != a).then_some(b),
                (c != a && c != b).then_some(c),
            ]
        };
        let mut offsets = try_vec(slots + 1, 0u32)?;
        for (t, tri) in tris.iter().enumerate() {
            if mesh.is_triangle_alive(t as TriangleId) {
                for v in distinct(tri).into_iter().flatten() {
                    offsets[v as usize + 1] += 1;
                }
            }
        }
        for v in 0..slots {
            offsets[v + 1] += offsets[v];
        }
        let mut triangles = try_vec(offsets[slots] as usize, 0 as TriangleId)?;
        let mut cursor = try_vec(slots, 0u32)?;
        cursor.copy_from_slice(&offsets[..slots]);
        // Triangles are visited in ascending id order, so every row comes out sorted.
        for (t, tri) in tris.iter().enumerate() {
            if mesh.is_triangle_alive(t as TriangleId) {
                for v in distinct(tri).into_iter().flatten() {
                    triangles[cursor[v as usize] as usize] = t as TriangleId;
                    cursor[v as usize] += 1;
                }
            }
        }
        Ok(Self { offsets, triangles })
    }

    /// Every alive triangle using `v`, each once, in ascending id order. Empty for
    /// unused, dead or out-of-range vertices.
    pub fn triangles(&self, v: VertexId) -> &[TriangleId] {
        let v = v as usize;
        if v + 1 >= self.offsets.len() {
            return &[];
        }
        &self.triangles[self.offsets[v] as usize..self.offsets[v + 1] as usize]
    }
}

impl Mesh {
    /// Builds the vertex → triangles index if it is not cached. It stays cached
    /// while only positions or channels change and is dropped on every topology
    /// change, like the adjacency table.
    pub fn ensure_incidence(&mut self) -> Result<()> {
        if self.incidence.is_none() {
            self.incidence = Some(Incidence::build(self)?);
        }
        Ok(())
    }

    /// The cached index, if [`Mesh::ensure_incidence`] built it and no topology
    /// change has happened since.
    pub fn incidence(&self) -> Option<&Incidence> {
        self.incidence.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use crate::Mesh;

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
    fn moves_keep_the_index_and_topology_changes_drop_it() {
        let mut m = fan_and_bowtie();
        m.ensure_incidence().unwrap();
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[0., 0., 1.]]).unwrap();
        tx.commit().unwrap();
        assert!(m.incidence().is_some());

        let mut tx = m.begin_edit();
        tx.remove_triangles(&[1]).unwrap();
        let patch = tx.commit().unwrap();
        assert!(m.incidence().is_none());
        m.ensure_incidence().unwrap();
        assert_eq!(m.incidence().unwrap().triangles(0), &[0, 3]);

        patch.revert(&mut m).unwrap();
        assert!(m.incidence().is_none());
    }
}
