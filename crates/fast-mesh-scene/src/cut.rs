//! Cutting a mesh by a world plane into two capped meshes.
//!
//! [`Scene::cut_by_plane`] maps the world plane into the mesh's local frame
//! (like a section, but without needing a BVH), splits the mesh with
//! [`ops::cut_by_plane`](fast_mesh_ops::cut_by_plane) and adds the removed
//! side as a new mesh with the same placement and update mode. The original
//! handle keeps the `+` side; the report's `added` is the `-` side.
//!
//! Only positions and triangles cross over (see the ops docs); the new mesh
//! has no normals, colouring, fields or section entries. Undo restores the
//! cut mesh in one step; the added mesh stays, like the removed source of
//! [`Scene::join`].

use crate::ops::{CutOptions as OpsCutOptions, cut_by_plane as ops_cut_by_plane};
use crate::placement::world_to_local;
use crate::{MeshId, MeshOptions, Plane, Result, Scene};

/// What to cut: a world plane and whether to cap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutOptions {
    /// The cutting plane, in world coordinates. The `+` side
    /// (`signed_distance >= 0`) stays on the original mesh.
    pub plane: Plane,
    /// Close both openings with flat, unfaired patches, across UV seams on a
    /// mesh with UVs. Without it both
    /// parts keep the open rim the cut leaves.
    pub cap: bool,
}

/// What [`Scene::cut_by_plane`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CutReport {
    /// The new mesh holding the `-` side.
    pub added: MeshId,
    /// Alive triangles of the kept (`+`) mesh, cap included.
    pub kept_triangles: usize,
    /// Alive triangles of the added (`-`) mesh, cap included.
    pub removed_triangles: usize,
    /// Cap triangles added on both sides together.
    pub cap_triangles: usize,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    a.map(|c| c / n)
}

/// The world plane in the mesh's stored local frame (origin excluded), as
/// for a section: the axes are the world `(u, v)` mapped through the
/// placement's linear inverse, then re-orthonormalised, so that the local
/// plane's zero set is the preimage of the world's, with matching sides.
/// `s >= 0` then means the same side in both frames.
fn local_plane(
    placement: &crate::Placement,
    mesh_origin: [f64; 3],
    plane: &Plane,
) -> Result<Plane> {
    let origin = world_to_local(placement, mesh_origin, plane.origin());
    let u = unit(placement.inverse_direction(plane.u()));
    let v = placement.inverse_direction(plane.v());
    let d = dot(v, u);
    let v = unit(std::array::from_fn(|k| v[k] - d * u[k]));
    Ok(Plane::with_axes(origin, u, v)?)
}

impl Scene {
    /// Cuts the mesh with a world plane: the `+` side
    /// (`signed_distance >= 0`) stays on the mesh and the `-` side is added
    /// as a new mesh, both with the mesh's placement, so the world shape is
    /// kept. With `options.cap` (the usual case) both openings are closed
    /// with flat, unfaired patches; without it both parts keep the open rim.
    /// Cutting a closed mesh with `cap` yields two closed meshes. On a mesh
    /// with UVs, closed means closed by exact position, so seams count as
    /// closed. A rim that runs into a real boundary stays open; on a mesh
    /// without UVs, split vertices are such a boundary, so weld them first.
    ///
    /// The cut runs as one edit of the mesh, so it undoes in one step; the
    /// added mesh is permanent, like the removed source of [`Scene::join`].
    /// Only positions and triangles cross over: the added mesh has no
    /// normals, colouring, fields or section entries, and new vertices take
    /// channel defaults (crossing vertices on the kept side interpolate
    /// their parents). UVs carry over to both sides, interpolated at the
    /// crossing vertices and across the caps.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id;
    /// [`SceneError::InvalidInput`] when the plane does not cut the mesh, or
    /// a rim cannot be capped; [`SceneError::Ops`] or [`SceneError::Store`]
    /// from the cut or the add. A failed cut changes nothing. If adding the
    /// removed side fails (out of memory), the cut of the original stays
    /// recorded and the added mesh is missing.
    pub fn cut_by_plane(&mut self, id: MeshId, options: &CutOptions) -> Result<CutReport> {
        self.drop_field_unit();
        let local = {
            let entry = self.entry(id)?;
            local_plane(&entry.placement, entry.mesh.origin(), &options.plane)?
        };
        let output = self.edit(id, |tx| {
            ops_cut_by_plane(tx, &local, &OpsCutOptions { cap: options.cap })
        })?;
        let (placement, mode) = {
            let entry = self.entry(id)?;
            (entry.placement, entry.mode)
        };
        let removed_triangles = output.report.removed_triangles;
        let cap_triangles = output.report.cap_triangles;
        let added = self.add_mesh(
            output.removed,
            MeshOptions {
                placement,
                mode,
                ..MeshOptions::default()
            },
        )?;
        Ok(CutReport {
            added,
            kept_triangles: output.report.kept_triangles,
            removed_triangles,
            cap_triangles,
        })
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::Mesh;

    use crate::ops::OpsError;
    use crate::{CutOptions, MeshOptions, Placement, Plane, Scene, SceneError, SceneOptions};

    /// A closed unit cube, outward wound.
    fn cube() -> Mesh {
        Mesh::from_buffers(
            &[
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                1.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, //
                1.0, 0.0, 1.0, //
                1.0, 1.0, 1.0, //
                0.0, 1.0, 1.0,
            ],
            &[
                0, 2, 1, 0, 3, 2, //
                4, 5, 6, 4, 6, 7, //
                0, 1, 5, 0, 5, 4, //
                2, 3, 7, 2, 7, 6, //
                1, 2, 6, 1, 6, 5, //
                3, 0, 4, 3, 4, 7, //
            ],
        )
        .unwrap()
    }

    fn plane(origin: [f64; 3], normal: [f64; 3]) -> Plane {
        Plane::new(origin, normal).unwrap()
    }

    #[test]
    fn cut_keeps_placement_and_undoes_in_one_step() {
        let mut scene = Scene::new(SceneOptions::default());
        // A unit cube moved to [10, 0, 0]: the world plane x = 10.5 cuts its
        // middle, whatever the local coordinates say.
        let placement = Placement {
            translation: [10.0, 0.0, 0.0],
            ..Placement::IDENTITY
        };
        let id = scene
            .add_mesh(
                cube(),
                MeshOptions {
                    placement,
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        scene.take_changes();
        let before = scene.mesh(id).unwrap().triangle_count();

        let report = scene
            .cut_by_plane(
                id,
                &CutOptions {
                    plane: plane([10.5, 0.0, 0.0], [1.0, 0.0, 0.0]),
                    cap: true,
                },
            )
            .unwrap();
        // Both parts closed, both placed where the cube was.
        assert!(
            crate::ops::boundary_loops(scene.mesh(id).unwrap())
                .unwrap()
                .is_empty()
        );
        assert!(
            crate::ops::boundary_loops(scene.mesh(report.added).unwrap())
                .unwrap()
                .is_empty()
        );
        assert_eq!(scene.placement(report.added), Ok(placement));
        assert_eq!(scene.mode(report.added), Ok(scene.mode(id).unwrap()));
        assert_ne!(report.added, id);
        // The kept side is the + one: its world box starts at the plane.
        let kept_world: Vec<[f64; 3]> = (0..scene.mesh(id).unwrap().vertex_slots() as u32)
            .filter(|&v| scene.mesh(id).unwrap().is_vertex_alive(v))
            .map(|v| {
                placement.apply([
                    scene.mesh(id).unwrap().origin()[0]
                        + f64::from(scene.mesh(id).unwrap().position(v).unwrap()[0]),
                    scene.mesh(id).unwrap().origin()[1]
                        + f64::from(scene.mesh(id).unwrap().position(v).unwrap()[1]),
                    scene.mesh(id).unwrap().origin()[2]
                        + f64::from(scene.mesh(id).unwrap().position(v).unwrap()[2]),
                ])
            })
            .collect();
        assert!(kept_world.iter().all(|p| p[0] >= 10.5 - 1e-6));
        assert_eq!(
            report.kept_triangles,
            scene.mesh(id).unwrap().triangle_count()
        );
        assert_eq!(
            report.removed_triangles,
            scene.mesh(report.added).unwrap().triangle_count()
        );

        // One undo step restores the cut mesh; the added part stays.
        assert!(scene.can_undo());
        assert!(scene.undo().unwrap());
        assert_eq!(scene.mesh(id).unwrap().triangle_count(), before);
        assert!(scene.mesh(report.added).is_ok());
        assert!(!scene.can_undo());
        assert!(scene.redo().unwrap());
        assert_ne!(scene.mesh(id).unwrap().triangle_count(), before);
    }

    #[test]
    fn scaled_cut_follows_the_scale() {
        // The unit cube scaled by [2, 1, 1] and moved to [0, 3, 0]: the
        // world shape is x in [0, 2]. The world plane x = 1 cuts its middle.
        let mut scene = Scene::new(SceneOptions::default());
        let placement = Placement {
            translation: [0.0, 3.0, 0.0],
            scale: [2.0, 1.0, 1.0],
            ..Placement::IDENTITY
        };
        let id = scene
            .add_mesh(
                cube(),
                MeshOptions {
                    placement,
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        scene.take_changes();
        let report = scene
            .cut_by_plane(
                id,
                &CutOptions {
                    plane: plane([1.0, 3.0, 0.0], [1.0, 0.0, 0.0]),
                    cap: true,
                },
            )
            .unwrap();
        assert!(
            crate::ops::boundary_loops(scene.mesh(id).unwrap())
                .unwrap()
                .is_empty()
        );
        assert!(
            crate::ops::boundary_loops(scene.mesh(report.added).unwrap())
                .unwrap()
                .is_empty()
        );
        assert_eq!(scene.placement(report.added), Ok(placement));
        // The kept world box is the + half: x in [1, 2], y in [3, 4].
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mesh = scene.mesh(id).unwrap();
        for v in 0..mesh.vertex_slots() as u32 {
            let Some(p) = mesh.position(v) else {
                continue;
            };
            let w = placement.apply([
                mesh.origin()[0] + f64::from(p[0]),
                mesh.origin()[1] + f64::from(p[1]),
                mesh.origin()[2] + f64::from(p[2]),
            ]);
            for k in 0..3 {
                lo[k] = lo[k].min(w[k]);
                hi[k] = hi[k].max(w[k]);
            }
        }
        assert!((lo[0] - 1.0).abs() < 1e-4, "{lo:?}");
        assert!((hi[0] - 2.0).abs() < 1e-4, "{hi:?}");
        assert!((lo[1] - 3.0).abs() < 1e-4, "{lo:?}");
        assert!((hi[1] - 4.0).abs() < 1e-4, "{hi:?}");
    }

    #[test]
    fn missing_and_uncapped_cuts() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(cube(), MeshOptions::default()).unwrap();
        scene.take_changes();
        // A plane past the cube misses it: nothing changes.
        assert_eq!(
            scene.cut_by_plane(
                id,
                &CutOptions {
                    plane: plane([5.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                    cap: true,
                },
            ),
            Err(SceneError::Ops(OpsError::InvalidInput(
                "plane does not cut the mesh",
            )))
        );
        assert_eq!(scene.mesh(id).unwrap().triangle_count(), 12);
        assert!(!scene.can_undo());
        // Without cap both parts stay open.
        let report = scene
            .cut_by_plane(
                id,
                &CutOptions {
                    plane: plane([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]),
                    cap: false,
                },
            )
            .unwrap();
        assert_eq!(report.cap_triangles, 0);
        assert!(
            !crate::ops::boundary_loops(scene.mesh(id).unwrap())
                .unwrap()
                .is_empty()
        );
        assert!(
            !crate::ops::boundary_loops(scene.mesh(report.added).unwrap())
                .unwrap()
                .is_empty()
        );
    }
}
