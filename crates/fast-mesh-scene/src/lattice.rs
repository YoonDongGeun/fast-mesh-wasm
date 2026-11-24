//! Lattice infill of scene meshes, measured in the world.

use fast_mesh_ops::lattice::{self, LatticeOptions, LatticePlan, LatticePreview, LatticeReport};

use crate::{MeshId, Result, Scene};

impl Scene {
    /// Plans the voxel grid of [`Scene::lattice_fill`] for the same mesh and
    /// options, without running it: its dimensions, voxel size, voxel count,
    /// and estimated triangles and memory, and whether it is within
    /// [`LatticeOptions::max_voxels`]. Costs a few passes over the mesh.
    ///
    /// # Errors
    /// As for [`Scene::lattice_fill`], except that the mesh is not checked
    /// for being closed, and that a grid over the limit comes back with
    /// [`LatticePlan::fits`] false.
    pub fn lattice_plan(&self, id: MeshId, options: &LatticeOptions) -> Result<LatticePlan> {
        let e = self.entry(id)?;
        Ok(lattice::lattice_plan(
            &e.mesh,
            &in_world(self, id, options)?,
        )?)
    }

    /// Fills the inside of the closed mesh `id`, or the cavities it encloses,
    /// with a lattice (see [`fast_mesh_ops::lattice`]), as one undo step.
    /// With a shell, or in a cavity, the mesh's outer surface is kept as it
    /// is; without one, the mesh becomes an open lattice.
    ///
    /// Lengths and volumes are measured in the world, through the mesh's
    /// placement scale, which replaces [`LatticeOptions::scale`]. The cells
    /// are aligned with the mesh's own axes.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`](crate::SceneError::UnknownMesh) for a
    /// stale id; [`SceneError::Ops`](crate::SceneError::Ops) as for
    /// [`lattice::lattice_fill`];
    /// [`SceneError::Store`](crate::SceneError::Store) when out of memory.
    /// The scene is unchanged on error.
    pub fn lattice_fill(&mut self, id: MeshId, options: &LatticeOptions) -> Result<LatticeReport> {
        let options = in_world(self, id, options)?;
        self.edit(id, |tx| lattice::lattice_fill(tx, &options))
    }

    /// The walls [`Scene::lattice_fill`] would put into mesh `id`, on a grid
    /// of at most `max_voxels` voxels, in the mesh's local frame; nothing in
    /// the scene changes (see [`lattice::lattice_preview`]).
    ///
    /// # Errors
    /// As for [`Scene::lattice_fill`], except that nothing is checked against
    /// [`LatticeOptions::max_voxels`] and that a lattice without walls is an
    /// empty preview.
    pub fn lattice_preview(
        &self,
        id: MeshId,
        options: &LatticeOptions,
        max_voxels: usize,
    ) -> Result<LatticePreview> {
        let options = in_world(self, id, options)?;
        let mesh = &self.entry(id)?.mesh;
        Ok(lattice::lattice_preview(mesh, &options, max_voxels)?)
    }
}

/// `options` with the placement's scale, so that lengths are world lengths.
fn in_world(scene: &Scene, id: MeshId, options: &LatticeOptions) -> Result<LatticeOptions> {
    Ok(LatticeOptions {
        scale: scene.entry(id)?.placement.scale,
        ..*options
    })
}

#[cfg(test)]
mod tests {
    use fast_mesh_ops::OpsError;
    use fast_mesh_ops::boolean::HollowOptions;
    use fast_mesh_ops::lattice::{LatticeKind, LatticeOptions, LatticeRegion};
    use fast_mesh_ops::primitives::{self, BoxOptions, IcosphereOptions, Primitive};
    use fast_mesh_store::Mesh;

    use crate::domains::SceneInspectOps;
    use crate::{MeshId, MeshOptions, Placement, Scene, SceneError, SceneOptions};

    fn mesh(primitive: &Primitive) -> Mesh {
        let src = primitives::generate(primitive).unwrap();
        Mesh::from_buffers(src.positions.as_flattened(), src.triangles.as_flattened()).unwrap()
    }

    fn cube(side: f64) -> Mesh {
        mesh(&Primitive::Box(BoxOptions {
            size: [side; 3],
            segments: [2; 3],
        }))
    }

    fn ball() -> Mesh {
        mesh(&Primitive::Icosphere(IcosphereOptions {
            radius: 1.0,
            subdivisions: 3,
        }))
    }

    fn add(scene: &mut Scene, mesh: Mesh, placement: Placement) -> MeshId {
        scene
            .add_mesh(
                mesh,
                MeshOptions {
                    placement,
                    ..MeshOptions::default()
                },
            )
            .unwrap()
    }

    fn turned(scale: [f64; 3]) -> Placement {
        let (s, c) = 0.3f64.sin_cos();
        Placement {
            rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            translation: [5.0, -2.0, 1.0],
            scale,
        }
    }

    fn options() -> LatticeOptions {
        LatticeOptions::builder()
            .cell_size(1.0)
            .wall_thickness(0.2)
            .build()
    }

    /// Closed, manifold and free of self-intersecting triangles; returns its
    /// volume in the world.
    fn checked_volume(scene: &mut Scene, id: MeshId) -> f64 {
        let a = fast_mesh_ops::analyze(scene.mesh(id).unwrap(), &Default::default()).unwrap();
        assert!(a.loops.is_empty());
        assert_eq!(a.non_manifold_edges, 0);
        assert!(a.pinch_vertices.is_empty());
        assert_eq!(a.orientation_conflicts, 0);
        assert_eq!(a.degenerate, 0);
        let mut crossing = Vec::new();
        scene
            .self_intersecting_triangles(id, &mut crossing)
            .unwrap();
        assert!(crossing.is_empty());
        // The mesh's own volume, scaled.
        let [x, y, z] = scene.placement(id).unwrap().scale;
        scene.measure(id).unwrap().volume.unwrap() * x * y * z
    }

    #[test]
    fn lengths_are_world_lengths() {
        let mut scene = Scene::new(SceneOptions::default());
        // Two units along every axis in the world, stored differently.
        let plain = add(&mut scene, cube(2.0), turned([1.0; 3]));
        let scaled = add(&mut scene, cube(0.5), turned([4.0; 3]));
        let want = LatticeKind::Gyroid.solid_fraction(1.0, 0.2) * 8.0;
        let mut counts = Vec::new();
        for id in [plain, scaled] {
            let plan = scene.lattice_plan(id, &options()).unwrap();
            assert!((plan.region_volume - 8.0).abs() < 1e-5);
            let r = scene.lattice_fill(id, &options()).unwrap();
            assert_eq!(r.plan, plan);
            let volume = checked_volume(&mut scene, id);
            assert!((volume / want - 1.0).abs() < 0.04, "{volume} {want}");
            assert!((r.solid_volume / volume - 1.0).abs() < 1e-3);
            counts.push((r.vertices, r.triangles));
        }
        assert_eq!(counts[0], counts[1]);
        // A scale that differs by axis.
        let slab = add(&mut scene, cube(1.0), turned([2.0, 1.5, 3.0]));
        let r = scene.lattice_fill(slab, &options()).unwrap();
        assert!((r.plan.region_volume - 9.0).abs() < 1e-5);
        let volume = checked_volume(&mut scene, slab);
        assert!((r.solid_volume / volume - 1.0).abs() < 1e-3);
    }

    #[test]
    fn a_fill_is_one_undo_step() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = add(&mut scene, ball(), Placement::IDENTITY);
        let before = scene.mesh(id).unwrap().triangle_count();
        scene.take_changes();
        let shelled = LatticeOptions {
            shell_thickness: 0.2,
            ..options()
        };
        let r = scene.lattice_fill(id, &shelled).unwrap();
        assert_eq!(
            scene.mesh(id).unwrap().triangle_count(),
            before + r.triangles
        );
        checked_volume(&mut scene, id);
        assert!(scene.take_changes().iter().any(|c| c.id == id));
        assert!(scene.undo().unwrap());
        assert_eq!(scene.mesh(id).unwrap().triangle_count(), before);
        assert!(!scene.can_undo());
        assert!(scene.redo().unwrap());
        assert_eq!(
            scene.mesh(id).unwrap().triangle_count(),
            before + r.triangles
        );
        assert!(scene.undo().unwrap());

        // Open, and in the cavity of the hollowed mesh: undo brings the
        // surfaces back.
        let open = scene.lattice_fill(id, &options()).unwrap();
        assert_eq!(scene.mesh(id).unwrap().triangle_count(), open.triangles);
        assert!(scene.undo().unwrap());
        assert_eq!(scene.mesh(id).unwrap().triangle_count(), before);
        let hollow = HollowOptions {
            wall_thickness: 0.2,
            voxel_size: Some(0.08),
            ..HollowOptions::default()
        };
        scene.hollow_mesh(id, &hollow).unwrap();
        let hollowed = scene.mesh(id).unwrap().triangle_count();
        let r = scene.lattice_fill(id, &options()).unwrap();
        assert_eq!(r.plan.region, LatticeRegion::Cavity);
        assert_eq!(
            scene.mesh(id).unwrap().triangle_count(),
            before + r.triangles
        );
        checked_volume(&mut scene, id);
        assert!(scene.undo().unwrap());
        assert_eq!(scene.mesh(id).unwrap().triangle_count(), hollowed);
    }

    #[test]
    fn a_preview_changes_nothing() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = add(&mut scene, cube(0.5), turned([4.0; 3]));
        scene.take_changes();
        let p = scene.lattice_preview(id, &options(), 20_000).unwrap();
        assert!(!p.walls.triangles.is_empty() && p.plan.voxels <= 20_000);
        // In the mesh's local frame.
        assert!(
            p.walls
                .positions
                .iter()
                .flatten()
                .all(|c| c.abs() <= 0.2501)
        );
        assert!(scene.take_changes().is_empty() && !scene.can_undo());
    }

    #[test]
    fn errors_leave_the_scene_unchanged() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = add(&mut scene, cube(2.0), Placement::IDENTITY);
        scene.take_changes();
        let tight = LatticeOptions {
            max_voxels: 1000,
            ..options()
        };
        assert!(!scene.lattice_plan(id, &tight).unwrap().fits);
        assert_eq!(
            scene.lattice_fill(id, &tight),
            Err(SceneError::Ops(OpsError::InvalidInput(
                "lattice grid exceeds max_voxels"
            )))
        );
        let gone = add(&mut scene, cube(1.0), Placement::IDENTITY);
        scene.remove_mesh(gone).unwrap();
        let unknown = Err(SceneError::UnknownMesh);
        assert_eq!(scene.lattice_fill(gone, &options()), unknown);
        assert_eq!(
            scene.lattice_plan(gone, &options()).map(|_| ()),
            Err(SceneError::UnknownMesh)
        );
        assert_eq!(
            scene.lattice_preview(gone, &options(), 1000).map(|_| ()),
            Err(SceneError::UnknownMesh)
        );
        assert!(!scene.can_undo());
    }
}
