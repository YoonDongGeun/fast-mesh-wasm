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

use crate::placement::local_to_world;
use crate::scene::Entry;
use crate::{Placement, Result, SceneError};

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
    let baked = baked_copy(mesh, map)?;
    let bvh = Bvh::build(&baked, options)?;
    Ok((baked, bvh))
}

/// [`baked_mesh`] without a BVH.
///
/// # Errors
/// As for [`baked_mesh`].
pub(crate) fn baked_copy(mesh: &Mesh, map: impl Fn([f32; 3]) -> [f64; 3]) -> Result<Mesh> {
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
    Ok(baked)
}

/// How far from the world origin a mesh may be, in units of its largest local
/// coordinate, for `f32` world positions to be about as good as its own: they
/// are rounded by up to eight steps of those.
const NEAR: f64 = 16.0;

/// A copy of a scene mesh as it is placed, in world axes: its positions are
/// offsets from its own origin, in `f64`, and as precise as the mesh's,
/// however far from the world origin it is. Slot ids match the mesh's.
///
/// That origin is where the mesh's own is in the world, or for a mesh near
/// the world origin (see [`NEAR`]) zero: its positions are the world's then.
///
/// # Errors
/// As for [`baked_mesh`].
pub(crate) fn baked_in_place(entry: &Entry) -> Result<Mesh> {
    let (placement, mesh) = (entry.placement, &entry.mesh);
    let at = local_to_world(&placement, mesh.origin(), [0.0; 3]);
    let alive = (0..mesh.vertex_slots() as VertexId).filter_map(|v| mesh.position(v));
    let largest = alive.flatten().fold(0.0f32, |m, c| m.max(c.abs()));
    let reach = NEAR * placement.max_scale() * f64::from(largest);
    if at.iter().all(|c| c.abs() <= reach) {
        return baked_copy(mesh, |p| {
            local_to_world(&placement, mesh.origin(), p.map(f64::from))
        });
    }
    let turn = Placement {
        translation: [0.0; 3],
        ..placement
    };
    let copy = baked_copy(mesh, |p| turn.apply(p.map(f64::from)))?;
    Ok(copy.with_origin(at)?)
}

#[cfg(test)]
mod tests {
    use fast_mesh_bvh::BuildOptions;
    use fast_mesh_store::Mesh;

    use super::*;
    use crate::{MeshOptions, Scene, SceneOptions};

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
    fn baked_in_place_is_around_the_world_origin_only_near_it() {
        let placed = |translation: [f64; 3], scale: [f64; 3]| {
            let mut scene = Scene::new(SceneOptions::default());
            let placement = Placement {
                translation,
                scale,
                ..Placement::IDENTITY
            };
            let options = MeshOptions {
                placement,
                ..MeshOptions::default()
            };
            let id = scene.add_mesh(quad(), options).unwrap();
            let mesh = baked_in_place(scene.entry(id).unwrap()).unwrap();
            (mesh.origin(), mesh.position(2).unwrap())
        };
        // Positions of up to 2: near is up to 32 away.
        let near = placed([-32.0, 0.0, 31.0], [1.0; 3]);
        assert_eq!(near, ([0.0; 3], [-30.0, 2.0, 31.0]));
        let far = placed([-32.0, 0.0, 33.0], [1.0; 3]);
        assert_eq!(far, ([-32.0, 0.0, 33.0], [2.0, 2.0, 0.0]));
        let scaled = placed([0.0, 0.0, 33.0], [1.0, 2.0, 1.0]);
        assert_eq!(scaled, ([0.0; 3], [2.0, 4.0, 33.0]));
        let far = placed([1.0e7, 0.0, 0.0], [1.0, 2.0, 1.0]);
        assert_eq!(far, ([1.0e7, 0.0, 0.0], [2.0, 4.0, 0.0]));
    }

    #[test]
    fn baked_in_place_goes_by_the_vertices_there_are() {
        let mut mesh = quad();
        let mut tx = mesh.begin_edit();
        tx.add_vertices(&[[900.0, 0.0, 0.0]]).unwrap();
        let _ = tx.commit().unwrap();
        let options = MeshOptions {
            placement: Placement {
                translation: [1000.0, 0.0, 0.0],
                ..Placement::IDENTITY
            },
            ..MeshOptions::default()
        };
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(mesh, options).unwrap();
        let origin = |scene: &Scene| {
            let mesh = baked_in_place(scene.entry(id).unwrap()).unwrap();
            mesh.origin()
        };
        assert_eq!(origin(&scene), [0.0; 3]);
        scene.edit(id, |tx| tx.remove_vertices(&[4])).unwrap();
        assert_eq!(origin(&scene), [1000.0, 0.0, 0.0]);
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
