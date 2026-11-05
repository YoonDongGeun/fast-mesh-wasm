//! Boolean operations (union, subtract, intersect) between two meshes.
//!
//! [`Scene::boolean`] combines mesh B into mesh A in the world: B is mapped
//! into A's local frame (rigid pairs through the relative transform,
//! scaled ones through a baked copy, like the pair queries), the
//! [`ops::boolean_plan`](fast_mesh_ops::boolean_plan) runs, and the plan
//! applies as one edit. The result keeps A's placement, so the world shape
//! is the boolean of the world shapes. B is unchanged.
//!
//! Only positions and triangles cross over (see the ops docs); the new mesh
//! has no normals, colouring, fields or section entries. Undo restores the
//! combined mesh in one step. Combining a mesh with itself is a no-op for
//! union and intersect, and clears it for subtract.

use fast_mesh_query::Rigid;
use fast_mesh_store::{StoreError, TriangleId, VertexId};

use crate::baked::baked_mesh;
use crate::placement::relative_affine;
use crate::{MeshId, Result, Scene};

/// Which boolean to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    /// Whatever is in either mesh.
    Union,
    /// Whatever is in A but not in B.
    Subtract,
    /// Whatever is in both meshes.
    Intersect,
}

/// What [`Scene::boolean`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BooleanReport {
    /// Alive triangles of the result.
    pub kept_triangles: usize,
    /// Triangles added (split pieces and the tool's kept pieces), minus the
    /// ones the cleanup merged or removed again.
    pub triangles_added: usize,
    /// A's UVs were dropped; undo restores them. The result mixes split pieces
    /// and B's surface, which have no UVs in A's space.
    pub uvs_dropped: bool,
}

impl Scene {
    /// Combines mesh B into mesh A: union, subtract or intersect as placed
    /// in the world. The result replaces A with A's placement, so the world
    /// shape is the boolean of the world shapes; B is unchanged. Closed
    /// manifold inputs give a closed result; touches leave pinch vertices.
    ///
    /// The combination runs as one edit of A, so it undoes in one step.
    /// Combining a mesh with itself is a no-op for union and intersect
    /// (nothing is recorded), and clears it for subtract. Only positions
    /// and triangles cross over: new vertices take channel defaults, and A's
    /// UV channel is dropped ([`BooleanReport::uvs_dropped`]).
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id;
    /// [`SceneError::InvalidInput`] when an input is not closed, manifold,
    /// consistently wound or free of degenerate, duplicate and
    /// self-intersecting triangles, or when a mapped position leaves `f32`
    /// range; [`SceneError::Ops`], [`SceneError::Query`],
    /// [`SceneError::Store`] or [`SceneError::StaleBvh`] from the plan, the
    /// bakes and the edit. A failed boolean changes nothing. A stale Manual
    /// BVH fails the query with [`SceneError::StaleBvh`], like the pair
    /// queries.
    pub fn boolean(&mut self, a: MeshId, b: MeshId, op: BooleanOp) -> Result<BooleanReport> {
        self.entry(a)?;
        self.entry(b)?;
        if a == b {
            return self.boolean_self(a, op);
        }
        // Bring both BVHs up to date (§15.4): rigid pairs plan against the
        // live meshes, scaled ones against baked copies.
        self.prepare(Some(&[a, b]))?;
        let ops_op = match op {
            BooleanOp::Union => crate::ops::BooleanOp::Union,
            BooleanOp::Subtract => crate::ops::BooleanOp::Subtract,
            BooleanOp::Intersect => crate::ops::BooleanOp::Intersect,
        };
        if self.is_rigid_pair(a, b)? {
            let plan = {
                let (ea, eb) = (self.entry(a)?, self.entry(b)?);
                let xf = crate::placement::relative_rigid(
                    &ea.placement,
                    ea.mesh.origin(),
                    &eb.placement,
                    eb.mesh.origin(),
                );
                crate::ops::boolean_plan(&ea.mesh, &ea.bvh, &eb.mesh, &eb.bvh, &xf, ops_op)?
            };
            let report = self.edit(a, |tx| crate::ops::apply_boolean(tx, &plan))?;
            return Ok(BooleanReport {
                kept_triangles: report.kept_triangles,
                triangles_added: report.triangles_added,
                uvs_dropped: report.uvs_dropped,
            });
        }
        // A scaled placement on either side: the pair query would run on
        // baked copies, and so does the boolean — baked into A's local
        // space, where the result lands.
        let plan = {
            let (ea, eb) = (self.entry(a)?, self.entry(b)?);
            let affine = relative_affine(
                &ea.placement,
                ea.mesh.origin(),
                &eb.placement,
                eb.mesh.origin(),
            );
            let (baked, baked_bvh) = baked_mesh(
                &eb.mesh,
                |l| affine.apply(l.map(f64::from)),
                eb.bvh.options(),
            )?;
            crate::ops::boolean_plan(
                &ea.mesh,
                &ea.bvh,
                &baked,
                &baked_bvh,
                &Rigid::IDENTITY,
                ops_op,
            )?
        };
        let report = self.edit(a, |tx| crate::ops::apply_boolean(tx, &plan))?;
        Ok(BooleanReport {
            kept_triangles: report.kept_triangles,
            triangles_added: report.triangles_added,
            uvs_dropped: report.uvs_dropped,
        })
    }

    /// Combines a mesh with itself: union and intersect change nothing (and
    /// record nothing), subtract clears the mesh in one undo step.
    fn boolean_self(&mut self, a: MeshId, op: BooleanOp) -> Result<BooleanReport> {
        match op {
            BooleanOp::Union | BooleanOp::Intersect => Ok(BooleanReport {
                kept_triangles: self.mesh(a)?.triangle_count(),
                triangles_added: 0,
                uvs_dropped: false,
            }),
            BooleanOp::Subtract => {
                self.edit(a, |tx| {
                    let all: Vec<TriangleId> = (0..tx.mesh().triangle_slots() as TriangleId)
                        .filter(|&t| tx.mesh().triangle(t).is_some())
                        .collect();
                    if all.is_empty() {
                        return Ok::<(), StoreError>(());
                    }
                    tx.remove_triangles(&all)?;
                    let mut orphans: Vec<VertexId> = Vec::new();
                    for v in 0..tx.mesh().vertex_slots() as VertexId {
                        if tx.mesh().is_vertex_alive(v) && tx.mesh().corner_count(v) == 0 {
                            orphans.push(v);
                        }
                    }
                    if !orphans.is_empty() {
                        tx.remove_vertices(&orphans)?;
                    }
                    Ok::<(), StoreError>(())
                })?;
                Ok(BooleanReport {
                    kept_triangles: 0,
                    triangles_added: 0,
                    uvs_dropped: false,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::Mesh;

    use crate::{MeshOptions, Placement, Scene, SceneError, SceneOptions};

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

    /// Six times the signed world-space volume, via the placement.
    fn world_volume6(scene: &Scene, id: crate::MeshId) -> f64 {
        let mesh = scene.mesh(id).unwrap();
        let placement = scene.placement(id).unwrap();
        let origin = mesh.origin();
        let mut v = 0.0;
        for t in 0..mesh.triangle_slots() as u32 {
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };
            let [a, b, c] = tri.map(|x| {
                let p = mesh.position(x).unwrap().map(f64::from);
                placement.apply([origin[0] + p[0], origin[1] + p[1], origin[2] + p[2]])
            });
            v += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        v
    }

    fn placed_cube(scene: &mut Scene, placement: Placement) -> crate::MeshId {
        scene
            .add_mesh(
                cube(),
                MeshOptions {
                    placement,
                    ..MeshOptions::default()
                },
            )
            .unwrap()
    }

    #[test]
    fn boolean_respects_placement_and_undoes_in_one_step() {
        let mut scene = Scene::new(SceneOptions::default());
        // A unit cube at [10, 0, 0] and one at [10.5, 0, 0]: the world
        // overlap is half a cube, whatever the local coordinates say.
        let a = placed_cube(
            &mut scene,
            Placement {
                translation: [10.0, 0.0, 0.0],
                ..Placement::IDENTITY
            },
        );
        let b = placed_cube(
            &mut scene,
            Placement {
                translation: [10.5, 0.0, 0.0],
                ..Placement::IDENTITY
            },
        );
        scene.take_changes();
        let before = scene.mesh(a).unwrap().triangle_count();

        let report = scene.boolean(a, b, crate::BooleanOp::Subtract).unwrap();
        assert!(
            crate::ops::boundary_loops(scene.mesh(a).unwrap())
                .unwrap()
                .is_empty()
        );
        assert_eq!(scene.placement(a).unwrap().translation, [10.0, 0.0, 0.0]);
        // The tool is unchanged.
        assert_eq!(scene.mesh(b).unwrap().triangle_count(), 12);
        // The world volume lost half a cube.
        assert!(
            (world_volume6(&scene, a) - 6.0 * 0.5).abs() < 1e-3,
            "{}",
            world_volume6(&scene, a)
        );
        assert_eq!(
            report.kept_triangles,
            scene.mesh(a).unwrap().triangle_count()
        );

        // One undo step restores the combined mesh; the tool was never in
        // the history.
        assert!(scene.can_undo());
        assert!(scene.undo().unwrap());
        assert_eq!(scene.mesh(a).unwrap().triangle_count(), before);
        assert!(!scene.can_undo());
        assert!(scene.redo().unwrap());
        assert_ne!(scene.mesh(a).unwrap().triangle_count(), before);
    }

    #[test]
    fn scaled_boolean_follows_the_scale() {
        // The unit cube scaled by [2, 1, 1] and moved to [0, 3, 0]: the
        // world shape is x in [0, 2]. Another unit cube at [1, 3, 0]
        // overlaps its middle in the world.
        let mut scene = Scene::new(SceneOptions::default());
        let placement = Placement {
            translation: [0.0, 3.0, 0.0],
            scale: [2.0, 1.0, 1.0],
            ..Placement::IDENTITY
        };
        let a = placed_cube(&mut scene, placement);
        let b = placed_cube(
            &mut scene,
            Placement {
                translation: [1.0, 3.0, 0.0],
                ..Placement::IDENTITY
            },
        );
        scene.take_changes();
        let report = scene.boolean(a, b, crate::BooleanOp::Intersect).unwrap();
        assert!(
            crate::ops::boundary_loops(scene.mesh(a).unwrap())
                .unwrap()
                .is_empty()
        );
        assert_eq!(scene.placement(a), Ok(placement));
        // The intersect is the world box x in [1, 2], y, z in [0, 1]
        // shifted by [0, 3, 0]: volume 1.
        assert!(
            (world_volume6(&scene, a) - 6.0).abs() < 1e-2,
            "{}",
            world_volume6(&scene, a)
        );
        assert_eq!(
            report.kept_triangles,
            scene.mesh(a).unwrap().triangle_count()
        );
    }

    #[test]
    fn boolean_rejects_unknown_meshes_and_clears_self() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(cube(), MeshOptions::default()).unwrap();
        let b = scene.add_mesh(cube(), MeshOptions::default()).unwrap();
        scene.take_changes();
        // Unknown ids first (before any stale-BVH or plan error).
        let missing = scene.remove_mesh(b).unwrap();
        let _ = missing;
        assert_eq!(
            scene.boolean(a, b, crate::BooleanOp::Union),
            Err(SceneError::UnknownMesh)
        );
        assert!(!scene.can_undo());
        // Union with itself changes nothing and records nothing.
        let before = scene.mesh(a).unwrap().triangle_count();
        let report = scene.boolean(a, a, crate::BooleanOp::Union).unwrap();
        assert_eq!(report.kept_triangles, before);
        assert!(!scene.can_undo());
        // Subtract clears it in one undo step.
        scene.boolean(a, a, crate::BooleanOp::Subtract).unwrap();
        assert_eq!(scene.mesh(a).unwrap().triangle_count(), 0);
        assert!(scene.undo().unwrap());
        assert_eq!(scene.mesh(a).unwrap().triangle_count(), before);
    }
}
