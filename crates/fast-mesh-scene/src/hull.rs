//! Convex hulls of scene meshes and of point sets, added as new meshes.

use fast_mesh_ops::append_mesh;
use fast_mesh_ops::hull::{ConvexHull, convex_hull, push_hull_points};
use fast_mesh_store::Mesh;

use crate::placement::local_to_world;
use crate::{MeshId, MeshOptions, Placement, Result, Scene, SceneError};

/// Outcome of [`Scene::convex_hull`] and [`Scene::convex_hull_points`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvexHullReport {
    /// The new mesh holding the hull.
    pub added: MeshId,
    /// Points the hull was built from (for meshes, the vertices their
    /// triangles use).
    pub input_points: usize,
    /// Vertices of the hull.
    pub vertices: usize,
    /// Triangles of the hull.
    pub triangles: usize,
}

impl Scene {
    /// Adds the convex hull of the meshes `ids` as a new closed,
    /// outward-wound mesh (see [`fast_mesh_ops::hull`]), built from the
    /// vertices their live triangles use. The meshes are not changed.
    ///
    /// When every mesh has the same placement (one mesh, say), the hull is
    /// built in their own space and the new mesh gets that placement, so it
    /// follows them exactly: with one origin too, its vertices have exactly
    /// the meshes' local positions. Otherwise the hull is built in the world
    /// and placed at [`Placement::IDENTITY`].
    ///
    /// The new mesh is added as one undo step (an undoable add, see
    /// [`MeshOptions::undoable`]): undo removes it again and redo brings it
    /// back under the same id.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id;
    /// [`SceneError::InvalidInput`] for an empty list;
    /// [`SceneError::Ops`] with
    /// [`OpsError::InvalidInput`](fast_mesh_ops::OpsError::InvalidInput) when the points
    /// have no volume (all on one point, line or plane), or a hull vertex
    /// does not fit the new mesh's `f32` positions; [`SceneError::Store`]
    /// when out of memory. The scene is unchanged on error.
    pub fn convex_hull(&mut self, ids: &[MeshId]) -> Result<ConvexHullReport> {
        self.drop_field_unit();
        let Some(&first) = ids.first() else {
            return Err(SceneError::InvalidInput(
                "convex hull needs at least one mesh",
            ));
        };
        let (placement, origin) = {
            let e = self.entry(first)?;
            (e.placement, e.mesh.origin())
        };
        let mut local = true;
        let mut same_origin = true;
        for &id in ids {
            let e = self.entry(id)?;
            local &= e.placement == placement;
            same_origin &= e.mesh.origin() == origin;
        }
        let mut points = Vec::new();
        for &id in ids {
            let e = self.entry(id)?;
            let (p, o) = (e.placement, e.mesh.origin());
            if local && same_origin {
                push_hull_points(&e.mesh, |l| l.map(f64::from), &mut points)?;
            } else if local {
                push_hull_points(
                    &e.mesh,
                    |l| std::array::from_fn(|k| o[k] + f64::from(l[k])),
                    &mut points,
                )?;
            } else {
                push_hull_points(
                    &e.mesh,
                    |l| local_to_world(&p, o, l.map(f64::from)),
                    &mut points,
                )?;
            }
        }
        let hull = convex_hull(&points)?;
        let placement = if local {
            placement
        } else {
            Placement::IDENTITY
        };
        // Exact local positions with a shared origin; else around the hull.
        let (mesh_origin, offset) = if local && same_origin {
            (origin, [0.0; 3])
        } else {
            let c = hull.center(&points);
            (c, c)
        };
        self.add_hull(&hull, &points, mesh_origin, offset, placement)
    }

    /// Adds the convex hull of world `points` as a new closed, outward-wound
    /// mesh at [`Placement::IDENTITY`], with its origin at the centre of its
    /// vertices' box. One undoable add, as for [`Scene::convex_hull`].
    ///
    /// # Errors
    /// [`SceneError::Ops`] with
    /// [`OpsError::InvalidInput`](fast_mesh_ops::OpsError::InvalidInput) for fewer than
    /// four points, a coordinate that is not finite or beyond `1e100`, points
    /// without volume, or a hull vertex that does not fit the new mesh's
    /// `f32` positions; [`SceneError::Store`] when out of memory. The scene
    /// is unchanged on error.
    pub fn convex_hull_points(&mut self, points: &[[f64; 3]]) -> Result<ConvexHullReport> {
        self.drop_field_unit();
        let hull = convex_hull(points)?;
        let c = hull.center(points);
        self.add_hull(&hull, points, c, c, Placement::IDENTITY)
    }

    /// Builds a mesh with `origin` holding `hull` (its points minus
    /// `offset`, as local positions) and adds it undoably with `placement`.
    fn add_hull(
        &mut self,
        hull: &ConvexHull,
        points: &[[f64; 3]],
        origin: [f64; 3],
        offset: [f64; 3],
        placement: Placement,
    ) -> Result<ConvexHullReport> {
        let source = hull.to_append_source(points, offset)?;
        let mut mesh = Mesh::from_buffers_f64_with_origin(&[], &[], origin)?;
        let mut tx = mesh.begin_edit();
        append_mesh(&mut tx, &source)?;
        // A patch that only adds slots reserves none: dropping it is free.
        drop(tx.commit()?);
        let added = self.add_mesh(
            mesh,
            MeshOptions {
                placement,
                undoable: true,
                ..MeshOptions::default()
            },
        )?;
        Ok(ConvexHullReport {
            added,
            input_points: points.len(),
            vertices: hull.vertices.len(),
            triangles: hull.triangles.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_ops::OpsError;
    use fast_mesh_store::Mesh;

    use crate::{MeshOptions, Placement, Scene, SceneError, SceneOptions};

    fn cube(at: [f32; 3]) -> Mesh {
        let p: Vec<f32> = (0..8)
            .flat_map(|i| {
                [i & 1, (i >> 1) & 1, (i >> 2) & 1]
                    .iter()
                    .enumerate()
                    .map(|(k, &c)| at[k] + c as f32)
                    .collect::<Vec<_>>()
            })
            .collect();
        let idx = [
            0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4,
            6, 1, 3, 5, 3, 7, 5,
        ];
        Mesh::from_buffers(&p, &idx).unwrap()
    }

    /// Six times the signed world volume of mesh `id`.
    fn volume6(scene: &Scene, id: crate::MeshId) -> f64 {
        let mesh = scene.mesh(id).unwrap();
        let placement = scene.placement(id).unwrap();
        let o = mesh.origin();
        let mut v = 0.0;
        for t in 0..mesh.triangle_slots() as u32 {
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };
            let [a, b, c] = tri.map(|x| {
                let p = mesh.position(x).unwrap().map(f64::from);
                placement.apply([o[0] + p[0], o[1] + p[1], o[2] + p[2]])
            });
            v += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        v
    }

    #[test]
    fn one_mesh_hull_keeps_its_placement_and_undoes_in_one_step() {
        let mut scene = Scene::new(SceneOptions::default());
        let placement = Placement {
            translation: [5.0, -2.0, 1.0],
            scale: [2.0, 1.0, 3.0],
            ..Placement::IDENTITY
        };
        let id = scene
            .add_mesh(
                cube([0.0; 3]),
                MeshOptions {
                    placement,
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        scene.take_changes();
        let r = scene.convex_hull(&[id]).unwrap();
        assert_eq!((r.input_points, r.vertices, r.triangles), (8, 8, 12));
        assert_eq!(scene.placement(r.added).unwrap(), placement);
        let hull = scene.mesh(r.added).unwrap();
        assert_eq!(hull.origin(), scene.mesh(id).unwrap().origin());
        for p in hull.positions() {
            assert!(scene.mesh(id).unwrap().positions().contains(p));
        }
        assert!((volume6(&scene, r.added) - 6.0 * 6.0).abs() < 1e-9);
        let changes = scene.take_changes();
        assert!(changes.iter().any(|c| c.id == r.added));

        // Undo removes the hull mesh (#269); redo brings it back as it was.
        assert!(scene.undo().unwrap());
        assert!(scene.mesh(r.added).is_err());
        assert!(scene.is_restorable(r.added));
        assert!(!scene.can_undo(), "the hull is one step");
        assert!(scene.redo().unwrap());
        assert_eq!(scene.mesh(r.added).unwrap().triangle_count(), 12);
        assert_eq!(scene.placement(r.added).unwrap(), placement);
        assert_eq!(scene.mesh(id).unwrap().triangle_count(), 12);
    }

    #[test]
    fn several_meshes_hull_in_the_world() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(cube([0.0; 3]), MeshOptions::default())
            .unwrap();
        let b = scene
            .add_mesh(
                cube([0.0; 3]),
                MeshOptions {
                    placement: Placement {
                        translation: [3.0, 0.0, 0.0],
                        ..Placement::IDENTITY
                    },
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        let r = scene.convex_hull(&[a, b]).unwrap();
        assert_eq!(r.input_points, 16);
        // A 4 x 1 x 1 box.
        assert_eq!((r.vertices, r.triangles), (8, 12));
        assert_eq!(scene.placement(r.added).unwrap(), Placement::IDENTITY);
        assert!((volume6(&scene, r.added) - 24.0).abs() < 1e-9);
        assert_eq!(scene.mesh(r.added).unwrap().origin(), [2.0, 0.5, 0.5]);
    }

    #[test]
    fn point_hulls_and_errors_leave_the_scene_unchanged() {
        let mut scene = Scene::new(SceneOptions::default());
        let pts = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.1, 0.1, 0.1],
        ];
        let r = scene.convex_hull_points(&pts).unwrap();
        assert_eq!((r.input_points, r.vertices, r.triangles), (5, 4, 4));
        assert!((volume6(&scene, r.added) - 8.0).abs() < 1e-9);
        let before = scene.ids().count();
        scene.take_changes();

        let flat = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        assert_eq!(
            scene.convex_hull_points(&flat),
            Err(SceneError::Ops(OpsError::InvalidInput(
                "convex hull points are coplanar"
            )))
        );
        assert_eq!(
            scene.convex_hull(&[]),
            Err(SceneError::InvalidInput(
                "convex hull needs at least one mesh"
            ))
        );
        let gone = scene
            .add_mesh(cube([0.0; 3]), MeshOptions::default())
            .unwrap();
        scene.remove_mesh(gone).unwrap();
        assert_eq!(scene.convex_hull(&[gone]), Err(SceneError::UnknownMesh));
        assert_eq!(scene.ids().count(), before);
    }
}
