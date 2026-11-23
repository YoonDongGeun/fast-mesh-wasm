//! Wraps of scene meshes, added as new meshes.

use fast_mesh_ops::append::AppendSource;
use fast_mesh_ops::append_mesh;
use fast_mesh_ops::wrap::{self, WrapOptions, WrapPlan};
use fast_mesh_store::Mesh;

use crate::placement::local_to_world;
use crate::{MeshId, MeshOptions, Placement, Result, Scene, SceneError};

/// Outcome of [`Scene::wrap`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneWrapReport {
    /// The new mesh holding the wrap.
    pub added: MeshId,
    /// The voxel grid used, in the frame of the new mesh.
    pub plan: WrapPlan,
    /// Vertices of the wrap.
    pub vertices: usize,
    /// Triangles of the wrap.
    pub triangles: usize,
    /// Vertices of the wrap moved onto the input.
    pub projected: usize,
}

/// The frame several meshes are wrapped in: `placement` of the result, and
/// the centre the positions are stored around.
struct Frame {
    /// The placement every mesh shares, for a wrap in their own space.
    shared: Option<Placement>,
    center: [f64; 3],
    min: [f32; 3],
    max: [f32; 3],
}

impl Frame {
    /// A position of a mesh with `placement` and `origin`, in this frame but
    /// not around its centre.
    fn map(&self, placement: &Placement, origin: [f64; 3], p: [f32; 3]) -> [f64; 3] {
        let p = p.map(f64::from);
        if self.shared.is_some() {
            std::array::from_fn(|k| origin[k] + p[k])
        } else {
            local_to_world(placement, origin, p)
        }
    }
}

impl Scene {
    /// Plans the voxel grid of [`Scene::wrap`] for the same meshes and
    /// options, without running it: its dimensions, voxel size, voxel count
    /// and estimated memory, and whether it is within
    /// [`WrapOptions::max_voxels`]. Cheap: one pass over the vertices.
    ///
    /// # Errors
    /// As for [`Scene::wrap`], except that a grid over the limit comes back
    /// with [`WrapPlan::fits`] false.
    pub fn wrap_plan(&self, ids: &[MeshId], options: &WrapOptions) -> Result<WrapPlan> {
        let frame = self.wrap_frame(ids)?;
        Ok(wrap::wrap_plan(frame.min, frame.max, options)?)
    }

    /// Adds one closed, manifold, outward-wound envelope without
    /// self-intersections around the meshes `ids` as a new mesh, with gaps
    /// and holes narrower than [`WrapOptions::gap`] closed (see
    /// [`fast_mesh_ops::wrap`]). The meshes may be open, broken or
    /// overlapping, and are not changed.
    ///
    /// When every mesh has the same placement and it is rigid (no scale), the
    /// wrap is built in their own space and the new mesh gets that placement,
    /// so it follows them exactly; otherwise it is built in the world and
    /// placed at [`Placement::IDENTITY`]. Either way the lengths in `options`
    /// are world lengths. The new mesh's origin is the centre of the meshes'
    /// bounds.
    ///
    /// The new mesh is added empty and the wrap lands in it as one edit, so
    /// the wrap is one undo step: undo empties the new mesh and redo fills
    /// it again. The mesh itself stays.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id;
    /// [`SceneError::InvalidInput`] for an empty list;
    /// [`SceneError::Ops`] with
    /// [`OpsError::InvalidInput`](fast_mesh_ops::OpsError::InvalidInput) for
    /// invalid options, meshes without a triangle with area, a grid over
    /// [`WrapOptions::max_voxels`], meshes that enclose no volume at this gap
    /// and voxel size, or bounds that do not fit `f32`;
    /// [`SceneError::Store`] when out of memory. The scene is unchanged on
    /// error.
    pub fn wrap(&mut self, ids: &[MeshId], options: &WrapOptions) -> Result<SceneWrapReport> {
        self.drop_field_unit();
        let frame = self.wrap_frame(ids)?;
        let c = frame.center;
        let mut soup = AppendSource::default();
        for &id in ids {
            let e = self.entry(id)?;
            let (p, o) = (e.placement, e.mesh.origin());
            wrap::push_soup(
                &e.mesh,
                |l| {
                    let q = frame.map(&p, o, l);
                    std::array::from_fn(|k| q[k] - c[k])
                },
                &mut soup,
            )?;
        }
        let (out, report) = wrap::wrap(&soup, options)?;
        drop(soup);
        let empty = Mesh::from_buffers_f64_with_origin(&[], &[], c)?;
        let added = self.add_mesh(
            empty,
            MeshOptions {
                placement: frame.shared.unwrap_or(Placement::IDENTITY),
                ..MeshOptions::default()
            },
        )?;
        if let Err(e) = self.edit(added, |tx| append_mesh(tx, &out)) {
            // Added and removed before any report: the scene is unchanged.
            self.remove_mesh(added)?;
            return Err(e);
        }
        Ok(SceneWrapReport {
            added,
            plan: report.plan,
            vertices: report.vertices,
            triangles: report.triangles,
            projected: report.projected,
        })
    }

    /// The frame the meshes `ids` are wrapped in, and their bounds around its
    /// centre.
    fn wrap_frame(&self, ids: &[MeshId]) -> Result<Frame> {
        let Some(&first) = ids.first() else {
            return Err(SceneError::InvalidInput("wrap needs at least one mesh"));
        };
        let placement = self.entry(first)?.placement;
        let mut shared = placement.scale == [1.0; 3];
        for &id in ids {
            shared &= self.entry(id)?.placement == placement;
        }
        let mut frame = Frame {
            shared: shared.then_some(placement),
            center: [0.0; 3],
            min: [0.0; 3],
            max: [0.0; 3],
        };
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for &id in ids {
            let e = self.entry(id)?;
            let (p, o) = (e.placement, e.mesh.origin());
            wrap::extend_bounds(&e.mesh, |l| frame.map(&p, o, l), &mut lo, &mut hi);
        }
        if lo[0] > hi[0] {
            return Err(fast_mesh_ops::OpsError::InvalidInput(
                "wrap needs at least one triangle with area",
            )
            .into());
        }
        frame.center = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
        frame.min = std::array::from_fn(|k| (lo[k] - frame.center[k]) as f32);
        frame.max = std::array::from_fn(|k| (hi[k] - frame.center[k]) as f32);
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_ops::OpsError;
    use fast_mesh_ops::primitives::{self, BoxOptions, IcosphereOptions, Primitive};
    use fast_mesh_ops::wrap::WrapOptions;
    use fast_mesh_store::Mesh;

    use crate::{MeshId, MeshOptions, Placement, Scene, SceneError, SceneOptions};

    fn mesh(primitive: &Primitive) -> Mesh {
        let src = primitives::generate(primitive).unwrap();
        Mesh::from_buffers(src.positions.as_flattened(), src.triangles.as_flattened()).unwrap()
    }

    fn cube() -> Mesh {
        mesh(&Primitive::Box(BoxOptions {
            size: [1.0; 3],
            segments: [4; 3],
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

    fn at(translation: [f64; 3]) -> Placement {
        Placement {
            translation,
            ..Placement::IDENTITY
        }
    }

    /// The world bounds and signed volume of mesh `id`.
    fn world(scene: &Scene, id: MeshId) -> ([f64; 3], [f64; 3], f64) {
        let mesh = scene.mesh(id).unwrap();
        let placement = scene.placement(id).unwrap();
        let o = mesh.origin();
        let at = |v: u32| {
            let p = mesh.position(v).unwrap().map(f64::from);
            placement.apply([o[0] + p[0], o[1] + p[1], o[2] + p[2]])
        };
        let (mut lo, mut hi, mut v) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3], 0.0);
        for t in 0..mesh.triangle_slots() as u32 {
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };
            let [a, b, c] = tri.map(at);
            for p in [a, b, c] {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            v += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0;
        }
        (lo, hi, v)
    }

    fn options(gap: f32, detail: u32) -> WrapOptions {
        WrapOptions::builder().gap(gap).detail(detail).build()
    }

    /// Closed, manifold and free of self-intersecting triangles.
    fn assert_clean(scene: &mut Scene, id: MeshId) {
        let analysis =
            fast_mesh_ops::analyze(scene.mesh(id).unwrap(), &Default::default()).unwrap();
        assert!(analysis.loops.is_empty());
        assert_eq!(analysis.non_manifold_edges, 0);
        assert!(analysis.pinch_vertices.is_empty());
        assert_eq!(analysis.orientation_conflicts, 0);
        assert_eq!(analysis.degenerate, 0);
        let mut crossing = Vec::new();
        scene
            .self_intersecting_triangles(id, &mut crossing)
            .unwrap();
        assert!(crossing.is_empty());
    }

    #[test]
    fn touching_meshes_wrap_in_the_world_as_one_undo_step() {
        let mut scene = Scene::new(SceneOptions::default());
        // Two unit boxes touching at x = 5.5, the second scaled to 2 along x.
        let a = add(&mut scene, cube(), at([5.0, 0.0, 0.0]));
        let b = add(
            &mut scene,
            cube(),
            Placement {
                translation: [6.5, 0.0, 0.0],
                scale: [2.0, 1.0, 1.0],
                ..Placement::IDENTITY
            },
        );
        scene.take_changes();
        let before = scene.mesh(a).unwrap().triangle_count();
        let opts = options(0.2, 48);
        let plan = scene.wrap_plan(&[a, b], &opts).unwrap();
        assert_eq!(plan.voxel_size, 3.0 / 48.0);
        let r = scene.wrap(&[a, b], &opts).unwrap();
        assert_eq!(r.plan, plan);
        assert_eq!(scene.placement(r.added).unwrap(), Placement::IDENTITY);
        assert_eq!(scene.mesh(r.added).unwrap().origin(), [6.0, 0.0, 0.0]);
        assert_eq!(scene.mesh(r.added).unwrap().triangle_count(), r.triangles);
        assert_clean(&mut scene, r.added);
        let (lo, hi, volume) = world(&scene, r.added);
        assert!((lo[0] - 4.5).abs() < 0.05 && (hi[0] - 7.5).abs() < 0.05);
        assert!((volume - 3.0).abs() < 0.06, "{volume}");
        assert!(scene.take_changes().iter().any(|c| c.id == r.added));

        assert!(scene.undo().unwrap());
        assert_eq!(scene.mesh(r.added).unwrap().triangle_count(), 0);
        assert!(!scene.can_undo(), "the wrap is one step");
        assert!(scene.redo().unwrap());
        assert_eq!(scene.mesh(r.added).unwrap().triangle_count(), r.triangles);
        assert_eq!(scene.mesh(a).unwrap().triangle_count(), before);
    }

    #[test]
    fn a_shared_rigid_placement_is_kept_and_a_scaled_one_is_not() {
        let mut scene = Scene::new(SceneOptions::default());
        let (s, c) = 0.3f64.sin_cos();
        let placement = Placement {
            rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            translation: [5.0, -2.0, 1.0],
            ..Placement::IDENTITY
        };
        let ball = mesh(&Primitive::Icosphere(IcosphereOptions {
            radius: 1.0,
            subdivisions: 4,
        }));
        let id = add(&mut scene, ball, placement);
        let r = scene.wrap(&[id], &options(0.2, 40)).unwrap();
        assert_eq!(scene.placement(r.added).unwrap(), placement);
        assert_clean(&mut scene, r.added);
        let (lo, hi, volume) = world(&scene, r.added);
        assert!((lo[0] - 4.0).abs() < 0.06 && (hi[2] - 2.0).abs() < 0.06);
        assert!((volume / (4.0 / 3.0 * std::f64::consts::PI) - 1.0).abs() < 0.02);

        // With a scale, lengths would not be world lengths in its own space.
        let scaled = add(
            &mut scene,
            cube(),
            Placement {
                scale: [10.0; 3],
                ..at([1.0, 2.0, 3.0])
            },
        );
        let opts = WrapOptions::builder().voxel_size(Some(0.25)).build();
        let r = scene.wrap(&[scaled], &opts).unwrap();
        assert_eq!(scene.placement(r.added).unwrap(), Placement::IDENTITY);
        assert_eq!(r.plan.dims, [40 + 2 * 4 + 1; 3]);
        let (_, _, volume) = world(&scene, r.added);
        assert!((volume / 1000.0 - 1.0).abs() < 0.01, "{volume}");
    }

    #[test]
    fn errors_leave_the_scene_unchanged() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = add(&mut scene, cube(), Placement::IDENTITY);
        scene.take_changes();
        let before = scene.ids().count();
        let opts = options(0.1, 32);
        assert_eq!(
            scene.wrap(&[], &opts),
            Err(SceneError::InvalidInput("wrap needs at least one mesh"))
        );
        let tight = WrapOptions {
            max_voxels: 1000,
            ..opts
        };
        assert!(!scene.wrap_plan(&[a], &tight).unwrap().fits);
        assert_eq!(
            scene.wrap(&[a], &tight),
            Err(SceneError::Ops(OpsError::InvalidInput(
                "wrap grid exceeds max_voxels"
            )))
        );
        let empty = add(&mut scene, Mesh::new(), Placement::IDENTITY);
        assert!(matches!(
            scene.wrap(&[empty], &opts),
            Err(SceneError::Ops(OpsError::InvalidInput(_)))
        ));
        let gone = add(&mut scene, cube(), Placement::IDENTITY);
        scene.remove_mesh(gone).unwrap();
        assert_eq!(scene.wrap(&[a, gone], &opts), Err(SceneError::UnknownMesh));
        assert_eq!(
            scene.wrap_plan(&[gone], &opts),
            Err(SceneError::UnknownMesh)
        );
        assert_eq!(scene.ids().count(), before + 1);
        assert!(!scene.can_undo());
    }
}
