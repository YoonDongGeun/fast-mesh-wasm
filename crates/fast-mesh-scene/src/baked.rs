//! Meshes baked through placements into a frame where a query runs.
//!
//! Phase 3 (`fast-mesh-query`) only takes rigid maps between mesh-local
//! spaces, so a query involving a scaled placement runs on copies baked
//! through the placements' affine maps: the baked mesh lives in the frame
//! the query runs in (another mesh's local space, or the world) with the
//! identity placement. Slot ids are preserved 1:1 (dead slots are removed
//! again after the build), so triangle ids map back, and barycentric
//! coordinates survive (affine maps preserve them).

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_store::{Mesh, StoreError, TriangleId, VertexId};

use crate::{Result, SceneError};

fn oom() -> SceneError {
    SceneError::Store(StoreError::OutOfMemory)
}

/// A copy of `mesh` with its stored positions (origin-relative `f32`) moved
/// by `map` into a frame, with its BVH built by `options`: the mesh the query
/// runs on, with the identity placement. Slot ids match `mesh` exactly: dead
/// triangle and vertex slots are removed again after the build.
///
/// `map` must be finite for every stored position (checked); a baked mesh of
/// a mesh without triangles is empty.
///
/// # Errors
/// [`SceneError::InvalidInput`] when a mapped position leaves `f32` range; a
/// store or BVH error from the build.
pub(crate) fn baked_mesh(
    mesh: &Mesh,
    map: impl Fn([f32; 3]) -> [f64; 3],
    options: &BuildOptions,
) -> Result<(Mesh, Bvh)> {
    let mut positions: Vec<f32> = Vec::new();
    positions
        .try_reserve(mesh.vertex_slots().saturating_mul(3))
        .map_err(|_| oom())?;
    for &p in mesh.positions() {
        let q = map(p);
        let f = [q[0] as f32, q[1] as f32, q[2] as f32];
        if !f.iter().all(|c| c.is_finite()) {
            return Err(SceneError::InvalidInput(
                "a scaled mesh leaves f32 range in the query frame",
            ));
        }
        positions.extend_from_slice(&f);
    }
    let mut indices: Vec<u32> = Vec::new();
    indices
        .try_reserve(mesh.triangle_slots().saturating_mul(3))
        .map_err(|_| oom())?;
    for &t in mesh.triangles() {
        indices.extend_from_slice(&t);
    }
    let mut baked = Mesh::from_buffers(&positions, &indices)?;
    // `from_buffers` wakes every slot; put the dead ones back to sleep so
    // the baked mesh matches `mesh` slot for slot.
    let dead_tris: Vec<TriangleId> = (0..mesh.triangle_slots() as TriangleId)
        .filter(|&t| mesh.triangle(t).is_none())
        .collect();
    let dead_verts: Vec<VertexId> = (0..mesh.vertex_slots() as VertexId)
        .filter(|&v| !mesh.is_vertex_alive(v))
        .collect();
    if !dead_tris.is_empty() || !dead_verts.is_empty() {
        let mut tx = baked.begin_edit();
        if !dead_tris.is_empty() {
            tx.remove_triangles(&dead_tris)?;
        }
        if !dead_verts.is_empty() {
            tx.remove_vertices(&dead_verts)?;
        }
        let _ = tx.commit()?;
    }
    let bvh = Bvh::build(&baked, options)?;
    Ok((baked, bvh))
}

#[cfg(test)]
mod tests {
    use fast_mesh_bvh::BuildOptions;
    use fast_mesh_store::Mesh;

    use super::*;

    fn quad() -> Mesh {
        Mesh::from_buffers(
            &[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap()
    }

    #[test]
    fn baked_mesh_preserves_slots() {
        let mut mesh = quad();
        let mut tx = mesh.begin_edit();
        tx.remove_triangles(&[1]).unwrap();
        tx.remove_vertices(&[3]).unwrap();
        let _ = tx.commit().unwrap();
        let (baked, bvh) =
            baked_mesh(&mesh, |p| p.map(f64::from), &BuildOptions::default()).unwrap();
        assert_eq!(baked.vertex_slots(), mesh.vertex_slots());
        assert_eq!(baked.triangle_slots(), mesh.triangle_slots());
        for v in 0..mesh.vertex_slots() as VertexId {
            assert_eq!(baked.is_vertex_alive(v), mesh.is_vertex_alive(v));
        }
        for t in 0..mesh.triangle_slots() as TriangleId {
            assert_eq!(baked.triangle(t).is_some(), mesh.triangle(t).is_some());
        }
        assert_eq!(baked.positions(), mesh.positions());
        assert!(bvh.is_current(&baked));
        // Empty meshes bake to empty meshes.
        let (empty, _) =
            baked_mesh(&Mesh::new(), |p| p.map(f64::from), &BuildOptions::default()).unwrap();
        assert_eq!(empty.triangle_slots(), 0);
    }

    #[test]
    fn baked_mesh_rejects_unrepresentable_maps() {
        let mesh = quad();
        assert!(matches!(
            baked_mesh(
                &mesh,
                |_| [f64::from(f32::MAX) * 2.0, 0.0, 0.0],
                &BuildOptions::default()
            ),
            Err(SceneError::InvalidInput(_))
        ));
    }
}
