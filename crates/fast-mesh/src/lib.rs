//! Fast triangle mesh utilities.
//!
//! Meshes use flat buffers, the same layout as WebGL / three.js `BufferGeometry`:
//! - `positions`: `[x0, y0, z0, x1, y1, z1, ...]`, length = vertex count × 3
//! - `indices`: `[a0, b0, c0, a1, b1, c1, ...]`, length = triangle count × 3
//!
//! This crate has no WebAssembly dependencies so it can be tested natively and
//! reused from plain Rust. The JS bindings live in `fast-mesh-wasm`.

use std::fmt;

/// Errors returned when input buffers do not describe a valid triangle mesh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshError {
    /// `positions.len()` is not a multiple of 3.
    PositionsNotMultipleOf3 { len: usize },
    /// `indices.len()` is not a multiple of 3.
    IndicesNotMultipleOf3 { len: usize },
    /// An index points past the last vertex.
    IndexOutOfRange { index: u32, vertex_count: usize },
    /// The mesh has no vertices.
    Empty,
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PositionsNotMultipleOf3 { len } => {
                write!(f, "positions length must be a multiple of 3, got {len}")
            }
            Self::IndicesNotMultipleOf3 { len } => {
                write!(f, "indices length must be a multiple of 3, got {len}")
            }
            Self::IndexOutOfRange {
                index,
                vertex_count,
            } => write!(
                f,
                "index {index} is out of range for a mesh with {vertex_count} vertices"
            ),
            Self::Empty => write!(f, "mesh has no vertices"),
        }
    }
}

impl std::error::Error for MeshError {}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

fn vertex_count(positions: &[f32]) -> Result<usize, MeshError> {
    #[allow(clippy::manual_is_multiple_of)] // `is_multiple_of` needs a newer Rust than the MSRV
    let bad = positions.len() % 3 != 0;
    if bad {
        return Err(MeshError::PositionsNotMultipleOf3 {
            len: positions.len(),
        });
    }
    Ok(positions.len() / 3)
}

fn validate_indices(indices: &[u32], vertex_count: usize) -> Result<(), MeshError> {
    #[allow(clippy::manual_is_multiple_of)] // `is_multiple_of` needs a newer Rust than the MSRV
    let bad = indices.len() % 3 != 0;
    if bad {
        return Err(MeshError::IndicesNotMultipleOf3 { len: indices.len() });
    }
    match indices.iter().find(|&&i| i as usize >= vertex_count) {
        Some(&index) => Err(MeshError::IndexOutOfRange {
            index,
            vertex_count,
        }),
        None => Ok(()),
    }
}

/// Computes smooth per-vertex normals.
///
/// Each face normal is weighted by the triangle's area, accumulated on its three
/// vertices, then normalized. Vertices not used by any non-degenerate triangle
/// get a `[0, 0, 0]` normal. Returns a buffer with the same length as `positions`.
pub fn compute_vertex_normals(positions: &[f32], indices: &[u32]) -> Result<Vec<f32>, MeshError> {
    let count = vertex_count(positions)?;
    validate_indices(indices, count)?;

    let mut normals = vec![0.0f32; positions.len()];
    let p = |i: usize| [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];

    for &tri in indices.as_chunks::<3>().0 {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (pa, pb, pc) = (p(a), p(b), p(c));
        let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        // The cross product's length is twice the triangle area, which gives area weighting for free.
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        for v in [a, b, c] {
            normals[v * 3] += n[0];
            normals[v * 3 + 1] += n[1];
            normals[v * 3 + 2] += n[2];
        }
    }

    for n in normals.as_chunks_mut::<3>().0 {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 0.0 {
            n.iter_mut().for_each(|x| *x /= len);
        }
    }
    Ok(normals)
}

/// Computes the axis-aligned bounding box of all vertices.
pub fn compute_bounds(positions: &[f32]) -> Result<Aabb, MeshError> {
    if vertex_count(positions)? == 0 {
        return Err(MeshError::Empty);
    }
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in positions.as_chunks::<3>().0 {
        for axis in 0..3 {
            min[axis] = min[axis].min(v[axis]);
            max[axis] = max[axis].max(v[axis]);
        }
    }
    Ok(Aabb { min, max })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Unit quad on the XY plane, two counter-clockwise triangles facing +Z.
    const QUAD_POSITIONS: [f32; 12] = [0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.];
    const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

    #[test]
    fn flat_quad_normals_point_up() {
        let normals = compute_vertex_normals(&QUAD_POSITIONS, &QUAD_INDICES).unwrap();
        assert_eq!(normals.len(), QUAD_POSITIONS.len());
        for n in normals.as_chunks::<3>().0 {
            assert_eq!(n, &[0., 0., 1.]);
        }
    }

    #[test]
    fn unused_vertex_gets_zero_normal() {
        let mut positions = QUAD_POSITIONS.to_vec();
        positions.extend_from_slice(&[5., 5., 5.]);
        let normals = compute_vertex_normals(&positions, &QUAD_INDICES).unwrap();
        assert_eq!(&normals[12..], &[0., 0., 0.]);
    }

    #[test]
    fn rejects_bad_buffers() {
        assert_eq!(
            compute_vertex_normals(&[0., 0.], &[]),
            Err(MeshError::PositionsNotMultipleOf3 { len: 2 })
        );
        assert_eq!(
            compute_vertex_normals(&QUAD_POSITIONS, &[0, 1]),
            Err(MeshError::IndicesNotMultipleOf3 { len: 2 })
        );
        assert_eq!(
            compute_vertex_normals(&QUAD_POSITIONS, &[0, 1, 9]),
            Err(MeshError::IndexOutOfRange {
                index: 9,
                vertex_count: 4
            })
        );
    }

    #[test]
    fn bounds_of_quad() {
        let b = compute_bounds(&QUAD_POSITIONS).unwrap();
        assert_eq!(b.min, [0., 0., 0.]);
        assert_eq!(b.max, [1., 1., 0.]);
        assert_eq!(compute_bounds(&[]), Err(MeshError::Empty));
    }
}
