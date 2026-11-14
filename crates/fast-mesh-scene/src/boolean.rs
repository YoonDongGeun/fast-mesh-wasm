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

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::Rigid;
use fast_mesh_store::{Mesh, StoreError, TriangleId, UV_CHANNEL, VertexId};

use crate::baked::baked_mesh;
use crate::ops::boolean::{BooleanMethod, BooleanOptions, CsgNode};
use crate::placement::{is_relative_rigid, relative_affine, relative_rigid};
use crate::{MeshId, MeshOptions, Placement, Result, Scene};

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

/// Whether a boolean must read `mesh` with its UV seams closed: it has UVs and
/// an open boundary. Whether that boundary is only seams is for the boolean's
/// own input checks to find, on the welded copy.
fn has_open_uvs(mesh: &Mesh) -> Result<bool> {
    Ok(mesh.uvs().is_some() && !crate::ops::boundary_loops(mesh)?.is_empty())
}

/// `baked` with exact duplicates merged and a new BVH. Slot ids stay, so the
/// same weld of the original gives the very mesh a plan was made for.
fn welded(mut baked: Mesh, options: &BuildOptions) -> Result<(Mesh, Bvh)> {
    let mut tx = baked.begin_edit();
    crate::ops::weld(&mut tx, 0.0)?;
    tx.commit()?;
    let bvh = Bvh::build(&baked, options)?;
    Ok((baked, bvh))
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
    /// UV channel is dropped ([`BooleanReport::uvs_dropped`]). Since UVs do not
    /// survive, a mesh with UVs is read with exactly coincident vertices
    /// merged, so seams do not make a closed surface count as open; every
    /// input check below applies to the merged geometry. B is unchanged, and
    /// undo restores A's split vertices.
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
        // A with its seams closed, when it has them: the plan's vertex ids are
        // then those of the same weld, run first in the edit below.
        let (a_seams, b_seams) = (
            has_open_uvs(&self.entry(a)?.mesh)?,
            has_open_uvs(&self.entry(b)?.mesh)?,
        );
        let a_copy = if a_seams {
            let ea = self.entry(a)?;
            let options = ea.bvh.options();
            let (baked, _) = baked_mesh(&ea.mesh, |l| l.map(f64::from), options)?;
            Some(welded(baked, options)?)
        } else {
            None
        };
        let apply = |tx: &mut fast_mesh_store::Transaction, plan: &crate::ops::BooleanPlan| {
            if a_seams {
                tx.drop_channel(UV_CHANNEL)?;
                crate::ops::weld(tx, 0.0)?;
            }
            let mut report = crate::ops::apply_boolean(tx, plan)?;
            report.uvs_dropped |= a_seams;
            Ok::<_, crate::ops::OpsError>(report)
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
                let b_copy = if b_seams {
                    let options = eb.bvh.options();
                    let (baked, _) = baked_mesh(&eb.mesh, |l| l.map(f64::from), options)?;
                    Some(welded(baked, options)?)
                } else {
                    None
                };
                let (ma, ba) = a_copy.as_ref().map_or((&ea.mesh, &ea.bvh), |(m, b)| (m, b));
                let (mb, bb) = b_copy.as_ref().map_or((&eb.mesh, &eb.bvh), |(m, b)| (m, b));
                crate::ops::boolean_plan(ma, ba, mb, bb, &xf, ops_op)?
            };
            let report = self.edit(a, |tx| apply(tx, &plan))?;
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
            let (baked, baked_bvh) = if b_seams {
                welded(baked, eb.bvh.options())?
            } else {
                (baked, baked_bvh)
            };
            let (ma, ba) = a_copy.as_ref().map_or((&ea.mesh, &ea.bvh), |(m, b)| (m, b));
            crate::ops::boolean_plan(ma, ba, &baked, &baked_bvh, &Rigid::IDENTITY, ops_op)?
        };
        let report = self.edit(a, |tx| apply(tx, &plan))?;
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

    /// Computes a world-space CSG boolean between `a` and `b` using [`BooleanOptions`],
    /// replacing `a` in a single atomic undo step.
    ///
    /// Supports both [`BooleanMethod::Manifold`] (with optional seam sliver snapping and
    /// `max_triangles` simplification) and volumetric fallbacks (`BvhSdf`, `FhEdt`).
    pub fn boolean_with_options(
        &mut self,
        a: MeshId,
        b: MeshId,
        options: &BooleanOptions,
    ) -> Result<BooleanReport> {
        self.entry(a)?;
        self.entry(b)?;
        let scene_op = match options.op {
            crate::ops::BooleanOp::Union => BooleanOp::Union,
            crate::ops::BooleanOp::Subtract => BooleanOp::Subtract,
            crate::ops::BooleanOp::Intersect => BooleanOp::Intersect,
        };
        if a == b {
            if matches!(options.op, crate::ops::BooleanOp::Subtract) {
                return self.boolean_self(a, BooleanOp::Subtract);
            }
            if options.method == BooleanMethod::Manifold && options.max_triangles.is_none() {
                return self.boolean_self(a, scene_op);
            }
        }

        self.prepare(Some(&[a, b]))?;
        let (a_seams, b_seams) = (
            has_open_uvs(&self.entry(a)?.mesh)?,
            has_open_uvs(&self.entry(b)?.mesh)?,
        );

        let (tool_mesh, xf): (Mesh, Rigid) = {
            let (ea, eb) = (self.entry(a)?, self.entry(b)?);
            if is_relative_rigid(&ea.placement, &eb.placement) {
                let xf = relative_rigid(
                    &ea.placement,
                    ea.mesh.origin(),
                    &eb.placement,
                    eb.mesh.origin(),
                );
                let (baked, _) = baked_mesh(&eb.mesh, |l| l.map(f64::from), eb.bvh.options())?;
                let mb = if b_seams {
                    welded(baked, eb.bvh.options())?.0
                } else {
                    baked
                };
                (mb, xf)
            } else {
                let affine = relative_affine(
                    &ea.placement,
                    ea.mesh.origin(),
                    &eb.placement,
                    eb.mesh.origin(),
                );
                let (baked, _) = baked_mesh(
                    &eb.mesh,
                    |l| affine.apply(l.map(f64::from)),
                    eb.bvh.options(),
                )?;
                let baked = if b_seams {
                    welded(baked, eb.bvh.options())?.0
                } else {
                    baked
                };
                (baked, Rigid::IDENTITY)
            }
        };

        let report = self.edit(a, |tx| {
            if a_seams {
                tx.drop_channel(UV_CHANNEL)?;
                crate::ops::weld(tx, 0.0)?;
            }
            let mut r =
                crate::ops::boolean::apply_boolean_with_options(tx, &tool_mesh, xf, options)?;
            r.uvs_dropped |= a_seams;
            Ok::<_, crate::ops::OpsError>(r)
        })?;

        Ok(BooleanReport {
            kept_triangles: report.kept_triangles,
            triangles_added: report.triangles_added,
            uvs_dropped: report.uvs_dropped,
        })
    }

    /// Evaluates an N-ary CSG tree over `meshes` in world coordinates and inserts
    /// the combined mesh into the scene at [`Placement::IDENTITY`], returning its [`MeshId`].
    ///
    /// Source meshes remain untouched in the scene.
    pub fn boolean_batch(
        &mut self,
        meshes: &[MeshId],
        tree: &CsgNode,
        options: &BooleanOptions,
    ) -> Result<MeshId> {
        for &id in meshes {
            self.entry(id)?;
        }
        self.prepare(Some(meshes))?;

        let mut prepared: Vec<(Option<Mesh>, Rigid)> = Vec::new();
        prepared
            .try_reserve_exact(meshes.len())
            .map_err(|_| StoreError::OutOfMemory)?;

        for &id in meshes {
            let entry = self.entry(id)?;
            let has_seams = has_open_uvs(&entry.mesh)?;
            if !entry.placement.is_scaled() && !has_seams {
                let xf = relative_rigid(
                    &Placement::IDENTITY,
                    [0.0; 3],
                    &entry.placement,
                    entry.mesh.origin(),
                );
                prepared.push((None, xf));
            } else {
                let affine = relative_affine(
                    &Placement::IDENTITY,
                    [0.0; 3],
                    &entry.placement,
                    entry.mesh.origin(),
                );
                let (baked, _) = baked_mesh(
                    &entry.mesh,
                    |l| affine.apply(l.map(f64::from)),
                    entry.bvh.options(),
                )?;
                let baked = if has_seams {
                    welded(baked, entry.bvh.options())?.0
                } else {
                    baked
                };
                prepared.push((Some(baked), Rigid::IDENTITY));
            }
        }

        let mut pairs: Vec<(&Mesh, Rigid)> = Vec::new();
        pairs
            .try_reserve_exact(meshes.len())
            .map_err(|_| StoreError::OutOfMemory)?;
        for (i, &id) in meshes.iter().enumerate() {
            let (ref baked_opt, xf) = prepared[i];
            let mesh_ref = match baked_opt {
                Some(m) => m,
                None => &self.entry(id)?.mesh,
            };
            pairs.push((mesh_ref, xf));
        }

        let combined = crate::ops::boolean::boolean_batch(&pairs, tree, options)?;
        drop(pairs);
        drop(prepared);
        self.add_mesh(combined, MeshOptions::default())
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

    #[test]
    fn boolean_with_options_supports_volumetric_and_manifold_modes_with_undo() {
        use crate::ops::boolean::{BooleanMethod, BooleanOp, BooleanOptions};

        let mut scene = Scene::new(SceneOptions::default());
        let mut mesh_a = cube();
        mesh_a.add_uvs(&[0.0; 16]).unwrap();
        let a = scene.add_mesh(mesh_a, MeshOptions::default()).unwrap();
        let b = placed_cube(
            &mut scene,
            Placement {
                translation: [0.5, 0.0, 0.0],
                ..Placement::IDENTITY
            },
        );
        scene.take_changes();

        let before = scene.mesh(a).unwrap().triangle_count();
        let opts = BooleanOptions {
            op: BooleanOp::Union,
            method: BooleanMethod::BvhSdf,
            voxel_pitch: Some(0.1),
            ..BooleanOptions::default()
        };
        let report = scene.boolean_with_options(a, b, &opts).unwrap();
        assert!(report.uvs_dropped);
        assert!(report.kept_triangles > 0);
        assert!(
            crate::ops::boundary_loops(scene.mesh(a).unwrap())
                .unwrap()
                .is_empty()
        );
        assert!((world_volume6(&scene, a) - 9.0).abs() < 0.75);

        assert!(scene.undo().unwrap());
        assert_eq!(scene.mesh(a).unwrap().triangle_count(), before);
        assert!(scene.mesh(a).unwrap().uvs().is_some());
    }

    #[test]
    fn scene_boolean_batch_reduces_multiple_placed_meshes() {
        use crate::ops::boolean::{BooleanMethod, BooleanOp, BooleanOptions, CsgNode};

        let mut scene = Scene::new(SceneOptions::default());
        let id0 = placed_cube(&mut scene, Placement::IDENTITY);
        let id1 = placed_cube(
            &mut scene,
            Placement {
                translation: [0.5, 0.0, 0.0],
                ..Placement::IDENTITY
            },
        );
        let id2 = placed_cube(
            &mut scene,
            Placement {
                translation: [1.0, 0.0, 0.0],
                ..Placement::IDENTITY
            },
        );
        let tree = CsgNode::reduce_all(BooleanOp::Union, 3).unwrap();

        for method in [
            BooleanMethod::Manifold,
            BooleanMethod::BvhSdf,
            BooleanMethod::FhEdt,
        ] {
            let opts = BooleanOptions {
                op: BooleanOp::Union,
                method,
                voxel_pitch: Some(0.1),
                ..BooleanOptions::default()
            };
            let out_id = scene.boolean_batch(&[id0, id1, id2], &tree, &opts).unwrap();
            let mesh = scene.mesh(out_id).unwrap();
            assert!(
                crate::ops::boundary_loops(mesh).unwrap().is_empty(),
                "{method:?}"
            );
            let tol = if method == BooleanMethod::Manifold {
                1e-3
            } else {
                0.75
            };
            assert!(
                (world_volume6(&scene, out_id) - 12.0).abs() < tol,
                "{method:?}: vol6={}",
                world_volume6(&scene, out_id)
            );
            assert_eq!(scene.mesh(id0).unwrap().triangle_count(), 12);
        }
    }
}
