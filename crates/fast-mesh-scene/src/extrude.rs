//! Extruding faces and bases of scene meshes, measured in the world.

use fast_mesh_ops::{self as ops, BaseLevel, BaseLoops, ExtrudeDirection, ExtrudeFacesReport};
use fast_mesh_store::TriangleId;

use crate::{MeshId, Placement, Result, Scene, SceneError};

/// What [`Scene::extrude_faces`] does, in world units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtrudeFacesOptions {
    /// How far the selection moves; negative pushes it in. Finite, non-zero.
    pub distance: f64,
    /// Which way it moves: along world normals, or a world direction.
    pub direction: ExtrudeDirection,
    /// Check the result for self-intersections and report them.
    pub check_self_intersections: bool,
}

/// Where [`Scene::extrude_base`] puts the base plane, which is perpendicular
/// to [`ExtrudeBaseOptions::direction`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BasePlane {
    /// At this world height, measured from the world origin along the
    /// opposite of the direction: for `direction = [0, -1, 0]`, the plane
    /// `y = height`.
    Height(f64),
    /// This far in the world past the mesh's furthest vertex along the
    /// direction (its lowest point, for a direction that runs down).
    Below(f64),
}

/// What [`Scene::extrude_base`] does, in world units.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtrudeBaseOptions {
    /// The world direction the walls run in (any length).
    pub direction: [f64; 3],
    /// Where the base plane is.
    pub plane: BasePlane,
    /// Which boundary loops are extruded.
    pub loops: BaseLoops,
    /// Refine the cap to the density of its walls.
    pub refine: bool,
    /// Check the result for self-intersections and report them.
    pub check_self_intersections: bool,
}

/// What [`Scene::extrude_base`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtrudeBaseReport {
    /// The base plane's world height, as for [`BasePlane::Height`].
    pub height: f64,
    /// The op's report; its [`level`](ops::ExtrudeBaseReport::level) is in
    /// the mesh's local frame.
    pub base: ops::ExtrudeBaseReport,
}

/// The linear part of a placement, row-major: `rotation * scale`.
fn linear(p: &Placement) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| p.rotation[i][j] * p.scale[j]))
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Scene {
    /// Extrudes the selected `triangles` of mesh `id` and walls the
    /// selection's boundary (see [`ops::extrude_faces`]), as one undo step.
    /// The distance, the direction and the normals are measured in the
    /// world, through the mesh's placement. The triangles keep their ids, so
    /// the same selection extrudes the caps again.
    ///
    /// Self-intersections the result would have are reported
    /// ([`ExtrudeFacesReport::self_intersections`]), and it is written all
    /// the same; the check needs no BVH.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id; [`SceneError::Ops`] as
    /// for [`ops::extrude_faces`]; [`SceneError::Store`] when out of memory.
    /// The scene is unchanged on error.
    pub fn extrude_faces(
        &mut self,
        id: MeshId,
        triangles: &[TriangleId],
        options: &ExtrudeFacesOptions,
    ) -> Result<ExtrudeFacesReport> {
        let options = ops::ExtrudeFacesOptions {
            distance: options.distance,
            direction: options.direction,
            to_world: linear(&self.entry(id)?.placement),
            check_self_intersections: options.check_self_intersections,
        };
        self.edit(id, |tx| ops::extrude_faces(tx, triangles, &options))
    }

    /// Extrudes boundary loops of mesh `id` along a world direction to a flat
    /// base perpendicular to it and caps them (see [`ops::extrude_base`]),
    /// as one undo step.
    ///
    /// Self-intersections the result would have are reported
    /// ([`ops::ExtrudeBaseReport::self_intersections`]), and it is written
    /// all the same; the check needs no BVH.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id; [`SceneError::Ops`] as
    /// for [`ops::extrude_base`]; [`SceneError::InvalidInput`] for a
    /// direction that is zero or not finite; [`SceneError::Store`] when out
    /// of memory. The scene is unchanged on error.
    pub fn extrude_base(
        &mut self,
        id: MeshId,
        options: &ExtrudeBaseOptions,
    ) -> Result<ExtrudeBaseReport> {
        let d = options.direction;
        let len = dot(d, d).sqrt();
        if !(len.is_finite() && len > 0.0) {
            return Err(SceneError::InvalidInput(
                "extrude base direction must be finite and non-zero",
            ));
        }
        let d = d.map(|x| x / len);
        let (placement, origin) = {
            let e = self.entry(id)?;
            (e.placement, e.mesh.origin())
        };
        // A stored point `p` sits at the world depth `dot(p, normal) + zero`
        // along `d`; its height is the opposite.
        let m = linear(&placement);
        let normal: [f64; 3] = std::array::from_fn(|j| dot([m[0][j], m[1][j], m[2][j]], d));
        let zero = dot(origin, normal) + dot(placement.translation, d);
        let options = ops::ExtrudeBaseOptions {
            direction: placement.inverse_direction(d),
            normal,
            level: match options.plane {
                BasePlane::Height(h) => BaseLevel::At(-h - zero),
                BasePlane::Below(offset) => BaseLevel::Beyond(offset),
            },
            loops: options.loops.clone(),
            refine: options.refine,
            check_self_intersections: options.check_self_intersections,
        };
        let base = self.edit(id, |tx| ops::extrude_base(tx, &options))?;
        Ok(ExtrudeBaseReport {
            height: -(base.level + zero),
            base,
        })
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_ops::{OpsError, boundary_loops};
    use fast_mesh_store::Mesh;

    use super::*;
    use crate::{MeshOptions, SceneOptions, UpdateMode};

    fn cube() -> Mesh {
        let p: Vec<f32> = (0..8)
            .flat_map(|i| [i & 1, (i >> 1) & 1, (i >> 2) & 1].map(|c| c as f32))
            .collect();
        // Triangles 10 and 11 are the +z face.
        let idx = [
            0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0, 2, 3, 0, 3,
            1, 4, 5, 7, 4, 7, 6,
        ];
        Mesh::from_buffers(&p, &idx).unwrap()
    }

    /// A flat `n` by `n` sheet of unit cells around `(x, y) = (100, 50)` at
    /// `z = 3`, facing `+z`, with its origin at its middle.
    fn sheet(n: u32) -> Mesh {
        let mut p = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                let half = f64::from(n) / 2.0;
                p.extend([100.0 - half + f64::from(x), 50.0 - half + f64::from(y), 3.0]);
            }
        }
        let mut idx = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                idx.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        Mesh::from_buffers_f64(&p, &idx).unwrap()
    }

    /// A rotation by 90 degrees about x (z goes to -y), a scale and a shift.
    fn posed() -> Placement {
        Placement {
            rotation: [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            translation: [5.0, -2.0, 1.0],
            scale: [2.0, 3.0, 0.5],
        }
    }

    /// The signed world volume of mesh `id`, about the world origin.
    fn volume(scene: &Scene, id: MeshId) -> f64 {
        let mesh = scene.mesh(id).unwrap();
        let placement = scene.placement(id).unwrap();
        let mut v = 0.0;
        for t in 0..mesh.triangle_slots() as u32 {
            if let Some(tri) = mesh.triangle(t) {
                let [a, b, c] = tri.map(|x| placement.apply(mesh.world_position(x).unwrap()));
                v += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]);
            }
        }
        v / 6.0
    }

    fn add(scene: &mut Scene, mesh: Mesh, placement: Placement) -> MeshId {
        let options = MeshOptions {
            placement,
            ..MeshOptions::default()
        };
        scene.add_mesh(mesh, options).unwrap()
    }

    fn by(distance: f64, direction: ExtrudeDirection) -> ExtrudeFacesOptions {
        ExtrudeFacesOptions {
            distance,
            direction,
            check_self_intersections: true,
        }
    }

    fn base(direction: [f64; 3], plane: BasePlane) -> ExtrudeBaseOptions {
        ExtrudeBaseOptions {
            direction,
            plane,
            loops: BaseLoops::Largest,
            refine: true,
            check_self_intersections: true,
        }
    }

    #[test]
    fn faces_extrude_by_world_units_in_one_step() {
        // The +z face faces -y in the world and is 2 x 3 there.
        for (direction, distance, gain) in [
            (ExtrudeDirection::RegionNormal, 0.25, 1.5),
            (ExtrudeDirection::VertexNormals, 0.25, 1.5),
            (ExtrudeDirection::Fixed([0.0, -7.0, 0.0]), 0.25, 1.5),
            (ExtrudeDirection::Fixed([0.0, -4.0, 3.0]), 1.0, 4.8),
        ] {
            let mut scene = Scene::new(SceneOptions::default());
            let id = add(&mut scene, cube(), posed());
            let before = volume(&scene, id);
            assert!((before - 3.0).abs() < 1e-6);
            scene.take_changes();
            let r = scene
                .extrude_faces(id, &[10, 11], &by(distance, direction))
                .unwrap();
            assert_eq!((r.regions, r.wall_triangles.len()), (1, 8));
            assert_eq!(r.self_intersections, Some(vec![]));
            assert!(
                (volume(&scene, id) - before - gain).abs() < 1e-5,
                "{direction:?}: {}",
                volume(&scene, id)
            );
            assert!(scene.take_changes().iter().any(|c| c.id == id));

            assert!(scene.undo().unwrap());
            assert!(!scene.can_undo(), "one step");
            assert_eq!(scene.mesh(id).unwrap().triangle_count(), 12);
            assert_eq!(volume(&scene, id), before);
            assert!(scene.redo().unwrap());
            assert!((volume(&scene, id) - before - gain).abs() < 1e-5);
        }
    }

    #[test]
    fn faces_report_crossings_without_a_current_bvh() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = add(&mut scene, cube(), Placement::IDENTITY);
        scene.set_mode(id, UpdateMode::Manual).unwrap();
        let r = scene
            .extrude_faces(id, &[10, 11], &by(-2.0, ExtrudeDirection::RegionNormal))
            .unwrap();
        assert!(!r.self_intersections.unwrap().is_empty());
        assert!(!scene.is_current(id).unwrap());
        let mut options = by(0.5, ExtrudeDirection::RegionNormal);
        options.check_self_intersections = false;
        let r = scene.extrude_faces(id, &[0, 1], &options).unwrap();
        assert_eq!(r.self_intersections, None);
    }

    #[test]
    fn faces_errors_leave_the_scene_unchanged() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = add(&mut scene, cube(), Placement::IDENTITY);
        scene.take_changes();
        let options = by(1.0, ExtrudeDirection::RegionNormal);
        assert_eq!(
            scene.extrude_faces(id, &[], &options),
            Err(SceneError::Ops(OpsError::InvalidInput(
                "extrude needs at least one selected triangle"
            )))
        );
        assert!(!scene.can_undo());
        assert!(scene.take_changes().is_empty());
        scene.remove_mesh(id).unwrap();
        assert_eq!(
            scene.extrude_faces(id, &[0], &options),
            Err(SceneError::UnknownMesh)
        );
    }

    #[test]
    fn bases_reach_a_world_height_in_one_step() {
        // The sheet's world plane: y = -(3 * 0.5) - 2 = -3.5, facing -y, and
        // it spans 8 x 12 there (n = 4 with scales 2 and 3).
        let mut scene = Scene::new(SceneOptions::default());
        let id = add(&mut scene, sheet(4), posed());
        for (options, height, depth) in [
            (base([0.0, 1.0, 0.0], BasePlane::Height(-1.0)), -1.0, 4.5),
            (base([0.0, 3.0, 0.0], BasePlane::Below(1.25)), 2.25, 1.25),
            // A direction of any length, and no refinement.
            (
                ExtrudeBaseOptions {
                    refine: false,
                    ..base([0.0, 5.0, 0.0], BasePlane::Below(2.0))
                },
                1.5,
                2.0,
            ),
        ] {
            let r = scene.extrude_base(id, &options).unwrap();
            assert!((r.height - height).abs() < 1e-5, "{}", r.height);
            assert!(!r.base.flipped);
            assert_eq!(r.base.self_intersections, Some(vec![]));
            let loops = boundary_loops(scene.mesh(id).unwrap()).unwrap();
            assert!(loops.is_empty(), "closed");
            assert!(
                (volume(&scene, id) - 96.0 * depth).abs() < 1e-2,
                "{} vs {}",
                volume(&scene, id),
                96.0 * depth
            );
            assert!(scene.undo().unwrap());
            assert!(!scene.can_undo(), "one step");
            assert_eq!(scene.mesh(id).unwrap().triangle_count(), 32);
        }
        // Toward -y it faces the base: turned around.
        let r = scene
            .extrude_base(id, &base([0.0, -1.0, 0.0], BasePlane::Below(1.0)))
            .unwrap();
        assert!(r.base.flipped);
        assert!((r.height + 4.5).abs() < 1e-5);
        assert!((volume(&scene, id) - 96.0).abs() < 1e-2);
    }

    #[test]
    fn base_errors_leave_the_scene_unchanged() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = add(&mut scene, sheet(4), posed());
        let err = scene.extrude_base(id, &base([0.0; 3], BasePlane::Below(1.0)));
        assert!(matches!(err, Err(SceneError::InvalidInput(_))), "{err:?}");
        // The plane y = -4, behind the sheet at y = -3.5.
        let err = scene.extrude_base(id, &base([0.0, 1.0, 0.0], BasePlane::Height(4.0)));
        assert!(
            matches!(err, Err(SceneError::Ops(OpsError::InvalidInput(m))) if m.contains("beyond")),
            "{err:?}"
        );
        let mut stale = base([0.0, 1.0, 0.0], BasePlane::Below(1.0));
        let mut lp = boundary_loops(scene.mesh(id).unwrap()).unwrap()[0].id;
        lp.hash ^= 1;
        stale.loops = BaseLoops::Ids(vec![lp]);
        assert_eq!(
            scene.extrude_base(id, &stale),
            Err(SceneError::Ops(OpsError::StaleLoops))
        );
        assert!(!scene.can_undo());
        scene.remove_mesh(id).unwrap();
        assert_eq!(
            scene.extrude_base(id, &base([0.0, 1.0, 0.0], BasePlane::Below(1.0))),
            Err(SceneError::UnknownMesh)
        );
    }
}
