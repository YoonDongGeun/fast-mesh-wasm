//! Rigid registration of one mesh to another: placements in and out of the
//! [`fast_mesh_ops::icp()`] solver.
//!
//! The solver runs in the fixed mesh's frame: its stored positions, with its
//! scale applied when it has one (on a baked copy), so that distances there
//! are world distances.

use fast_mesh_bvh::Bvh;
use fast_mesh_ops::align::{
    LandmarkAlignment, LandmarkPair, RegistrationOptions, RegistrationReport, SymmetryPlaneReport,
    align_landmarks, detect_symmetry_plane, motion_to_rigid, register_multistage, rigid_to_matrix4,
    rigid_to_motion,
};
use fast_mesh_ops::{IcpMotion, IcpOptions, IcpStatus, icp, sample_surface};
use fast_mesh_query::Rigid;
use fast_mesh_store::{Mesh, StoreError};

use crate::baked::baked_mesh;
use crate::placement::orthonormalize;
use crate::{MeshId, Placement, Result, Scene, SceneError};

/// Options for [`Scene::register`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegisterOptions {
    /// The solver's options: iterations, tolerance, overlap and maximum
    /// distance, in world units.
    pub icp: IcpOptions,
    /// The most surface points of the moving mesh to use, evenly spaced over
    /// its triangles' centroids and its vertices
    /// ([`fast_mesh_ops::sample_surface`]); at least 1. The `rmse` is measured
    /// on these. Default 100,000.
    pub max_samples: usize,
    /// Whether a good result becomes the moving mesh's placement, as one undo
    /// step: it must converge and pass the guards in [`NotApplied`]. Default
    /// true.
    pub apply: bool,
    /// The largest `rmse`, in world units, that may be applied; `None` means
    /// [`DEFAULT_MAX_RMSE_FRACTION`] of the moving points' extent. Zero or
    /// greater; `f64::INFINITY` turns the guard off.
    pub max_rmse: Option<f64>,
}

/// The default [`RegisterOptions::max_rmse`], as a fraction of the moving
/// points' extent. A wrong local minimum measured about 3%, and the same mesh
/// with noise of 1% of its size correctly aligned at about 0.4% (`icp_survey`
/// in `fast-mesh-ops`). Scans noisier than this need their own limit.
pub const DEFAULT_MAX_RMSE_FRACTION: f64 = 0.01;

/// Why a result was not applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotApplied {
    /// [`RegisterOptions::apply`] was false.
    NotRequested,
    /// The status is not [`IcpStatus::Converged`].
    NotConverged,
    /// `rmse` is above [`RegisterOptions::max_rmse`]: most likely a wrong local
    /// minimum, which a converged status does not rule out.
    RmseAboveLimit,
    /// Fewer than half of the points `overlap` expects were within
    /// `max_distance`: the fit rests on a small patch.
    TooFewInliers,
}

impl Default for RegisterOptions {
    fn default() -> Self {
        RegisterOptions {
            icp: IcpOptions::default(),
            max_samples: 100_000,
            apply: true,
            max_rmse: None,
        }
    }
}

/// What [`Scene::register`] found.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisterReport {
    /// The moving mesh's aligned placement, with its scale unchanged: the
    /// converged result, or the best estimate when it did not converge.
    pub placement: Placement,
    /// The root mean square distance, in world units, from the used points to
    /// the fixed mesh at `placement`; infinite when no point has a
    /// correspondence.
    pub rmse: f64,
    /// Solver steps taken.
    pub iterations: usize,
    /// Whether the iteration settled and the data determine every motion
    /// locally ([`fast_mesh_ops::icp()`] tests this by restarting from displaced
    /// poses). That is not proof of the right alignment, since ICP is local and
    /// can settle in a wrong minimum: judge that from `rmse`.
    pub converged: bool,
    /// Why iteration stopped.
    pub status: IcpStatus,
    /// Whether `placement` was set on the moving mesh.
    pub applied: bool,
    /// Why it was not; `None` exactly when `applied`.
    pub not_applied: Option<NotApplied>,
    /// The `rmse` limit that applied, in world units.
    pub max_rmse: f64,
    /// Surface points of the moving mesh sampled.
    pub samples: usize,
    /// Sampled points used at `placement`: the rest were trimmed or out of
    /// range.
    pub inliers: usize,
    /// A warning of a weak motion, from geometry alone; see
    /// [`fast_mesh_ops::IcpReport::conditioning`].
    pub conditioning: f64,
    /// The motions the data do not determine, in world coordinates; non-empty
    /// only with [`IcpStatus::Underconstrained`]. The pose along them is
    /// arbitrary.
    pub undetermined: Vec<IcpMotion>,
}

/// `a` after `b`.
fn compose(a: &Rigid, b: &Rigid) -> Rigid {
    let (ra, rb) = (&a.rotation, &b.rotation);
    let t = b.translation;
    Rigid {
        rotation: std::array::from_fn(|i| {
            std::array::from_fn(|j| ra[i][0] * rb[0][j] + ra[i][1] * rb[1][j] + ra[i][2] * rb[2][j])
        }),
        translation: std::array::from_fn(|i| {
            ra[i][0] * t[0] + ra[i][1] * t[1] + ra[i][2] * t[2] + a.translation[i]
        }),
    }
}

/// A motion in the frame `frame`, in world coordinates.
fn motion_to_world(m: &IcpMotion, frame: &Rigid) -> IcpMotion {
    let r = &frame.rotation;
    let turn = |v: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|i| r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2])
    };
    let about = turn(m.about);
    IcpMotion {
        translation: turn(m.translation),
        rotation: turn(m.rotation),
        about: std::array::from_fn(|i| about[i] + frame.translation[i]),
        alternative_rmse: m.alternative_rmse,
    }
}

/// The moving mesh's sampled surface points as `scale * (origin + local)`.
fn scaled_samples(mesh: &Mesh, placement: &Placement, max: usize) -> Result<Vec<[f64; 3]>> {
    let origin = mesh.origin();
    let (s, local) = (placement.scale, sample_surface(mesh, max)?);
    let mut out = Vec::new();
    out.try_reserve_exact(local.len())
        .map_err(|_| SceneError::Store(StoreError::OutOfMemory))?;
    out.extend(
        local
            .iter()
            .map(|p| -> [f64; 3] { std::array::from_fn(|i| s[i] * (origin[i] + p[i])) }),
    );
    Ok(out)
}

/// Rigid frame mapping `scale * p_local` (origin-relative stored positions)
/// to world space.
fn frame_of(placement: &Placement, origin: [f64; 3]) -> Rigid {
    let s = placement.scale;
    let r = orthonormalize(placement.rotation);
    let scaled_origin: [f64; 3] = std::array::from_fn(|i| s[i] * origin[i]);
    Rigid {
        rotation: r,
        translation: std::array::from_fn(|i| {
            r[i][0] * scaled_origin[0]
                + r[i][1] * scaled_origin[1]
                + r[i][2] * scaled_origin[2]
                + placement.translation[i]
        }),
    }
}

/// Converts a world frame mapping `scale * p_local` back to a `Placement`
/// preserving the mesh's `origin` and `scale`.
fn placement_from_scaled_frame(out: &Rigid, scale: [f64; 3], origin: [f64; 3]) -> Placement {
    let r = orthonormalize(out.rotation);
    let scaled_origin: [f64; 3] = std::array::from_fn(|i| scale[i] * origin[i]);
    let translation: [f64; 3] = std::array::from_fn(|i| {
        out.translation[i]
            - (r[i][0] * scaled_origin[0] + r[i][1] * scaled_origin[1] + r[i][2] * scaled_origin[2])
    });
    Placement {
        rotation: r,
        translation,
        scale,
    }
}

/// Bakes a mesh through `placement.scale` (without `origin`) when scaled.
fn maybe_baked(mesh: &Mesh, placement: &Placement, bvh: &Bvh) -> Result<Option<(Mesh, Bvh)>> {
    if placement.is_scaled() {
        let s = placement.scale;
        Ok(Some(baked_mesh(
            mesh,
            |p| std::array::from_fn(|i| s[i] * f64::from(p[i])),
            bvh.options(),
        )?))
    } else {
        Ok(None)
    }
}

fn canonicalize_normal(mut n: [f64; 3]) -> [f64; 3] {
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 0.0 {
        n = n.map(|c| c / len);
    }
    let dom = if n[0].abs() >= n[1].abs() && n[0].abs() >= n[2].abs() {
        0
    } else if n[1].abs() >= n[2].abs() {
        1
    } else {
        2
    };
    if n[dom] < 0.0 {
        n = n.map(|c| -c);
    }
    n
}

impl Scene {
    /// Rigidly aligns the mesh `moving` to the mesh `fixed` by point-to-plane
    /// ICP ([`fast_mesh_ops::icp()`]), using points on `moving`'s surface and
    /// `fixed`'s BVH.
    ///
    /// **The initial guess is the current placements.** ICP is local, so put
    /// the meshes roughly in place first, with [`Scene::set_placement`]; a
    /// good guess is off by a small part of the shape's features.
    ///
    /// **Applying.** With [`RegisterOptions::apply`] (the default) a good result
    /// becomes `moving`'s placement as one undo step, or as part of the open
    /// step. Good means converged, with `rmse` within
    /// [`RegisterOptions::max_rmse`] and enough points in range ([`NotApplied`]
    /// says which failed). Anything else changes nothing, whatever the option,
    /// but its estimate is returned; set it with [`Scene::set_placement`] if it
    /// will do.
    ///
    /// **Undetermined poses.** A sphere's spin, a cylinder's spin and a plane's
    /// slide are not fixed by the data, though facets or noise can make the
    /// iteration settle on some value with a tiny `rmse`. Those report
    /// [`IcpStatus::Underconstrained`], not converged, and list the motions in
    /// `undetermined`. Coarse regular meshes count too: a 16-sided sphere has
    /// exact fits 22.5 degrees apart.
    ///
    /// **Scales.** Only rotation and translation are solved. `moving`'s scale
    /// is applied to its points and kept as it is. `fixed`'s scale, uniform
    /// or not, is followed exactly, on a copy of `fixed` baked through it with
    /// its own BVH, so each such call costs a BVH build. Distances and options
    /// are in world units either way.
    ///
    /// **Partial overlap.** Set [`IcpOptions::overlap`] a little below the
    /// fraction of `moving` that lies over `fixed`, or nothing is rejected.
    ///
    /// Only `fixed`'s BVH is used, and only it is brought up to date.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id;
    /// [`SceneError::InvalidInput`] when the ids are one mesh, `max_samples` is
    /// zero or `max_rmse` is negative or NaN; [`SceneError::StaleBvh`] for a stale Manual `fixed`;
    /// [`SceneError::Ops`] when a mesh has no triangles, an option is invalid,
    /// or a placement takes the points out of `f32` range in `fixed`'s
    /// frame. On error the scene is unchanged.
    pub fn register(
        &mut self,
        moving: MeshId,
        fixed: MeshId,
        options: &RegisterOptions,
    ) -> Result<RegisterReport> {
        self.drop_field_unit();
        self.entry(moving)?;
        self.entry(fixed)?;
        if moving == fixed {
            return Err(SceneError::InvalidInput("cannot register a mesh to itself"));
        }
        if options.max_samples == 0 {
            return Err(SceneError::InvalidInput("max_samples must be at least 1"));
        }
        if options.max_rmse.is_some_and(|v| v.is_nan() || v < 0.0) {
            return Err(SceneError::InvalidInput("max_rmse must be zero or greater"));
        }
        self.prepare(Some(&[fixed]))?;
        let (m, f) = (self.entry(moving)?, self.entry(fixed)?);
        let points = scaled_samples(&m.mesh, &m.placement, options.max_samples)?;
        let moving_scale = m.placement.scale;
        // The moving vertices, once scaled, to the world.
        let world = Rigid {
            rotation: orthonormalize(m.placement.rotation),
            translation: m.placement.translation,
        };
        let frame = frame_of(&f.placement, f.mesh.origin());
        let initial = compose(&frame.inverse(), &world);
        let baked = maybe_baked(&f.mesh, &f.placement, &f.bvh)?;
        let (mesh, bvh) = baked
            .as_ref()
            .map_or((&f.mesh, &f.bvh), |(mesh, bvh)| (mesh, bvh));
        let solved = icp(&points, mesh, bvh, &initial, &options.icp)?;
        let out = compose(&frame, &solved.transform);
        let placement = Placement {
            rotation: out.rotation,
            translation: out.translation,
            scale: moving_scale,
        };
        let converged = solved.converged();
        let max_rmse = options
            .max_rmse
            .unwrap_or(DEFAULT_MAX_RMSE_FRACTION * solved.extent);
        let expected = (options.icp.overlap * solved.samples as f64).ceil();
        let not_applied = if !converged {
            Some(NotApplied::NotConverged)
        } else if !options.apply {
            Some(NotApplied::NotRequested)
        } else if solved.rmse > max_rmse {
            Some(NotApplied::RmseAboveLimit)
        } else if (solved.inliers as f64) * 2.0 < expected {
            Some(NotApplied::TooFewInliers)
        } else {
            None
        };
        if not_applied.is_none() {
            self.set_placement(moving, placement)?;
        }
        Ok(RegisterReport {
            placement,
            rmse: solved.rmse,
            iterations: solved.iterations,
            converged,
            status: solved.status,
            applied: not_applied.is_none(),
            not_applied,
            max_rmse,
            samples: solved.samples,
            inliers: solved.inliers,
            conditioning: solved.conditioning,
            undetermined: solved
                .undetermined
                .iter()
                .map(|m| motion_to_world(m, &frame))
                .collect(),
        })
    }

    /// Aligns `moving` toward `fixed` using `N >= 3` world-space landmark pairs
    /// (`align_landmarks`) and applies the resulting rigid or uniformly scaled
    /// world transform onto `moving`'s placement as a single undoable edit step.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id, [`SceneError::InvalidInput`]
    /// if `moving == fixed` or scale is degenerate, or [`SceneError::Ops`] when
    /// landmark pairs are fewer than 3 or degenerate.
    pub fn align_by_landmarks(
        &mut self,
        moving: MeshId,
        fixed: MeshId,
        pairs: &[LandmarkPair],
        allow_scaling: bool,
    ) -> Result<LandmarkAlignment> {
        self.drop_field_unit();
        let old = self.entry(moving)?.placement;
        self.entry(fixed)?;
        if moving == fixed {
            return Err(SceneError::InvalidInput("cannot align a mesh to itself"));
        }
        let alignment = align_landmarks(pairs, allow_scaling)?;
        if !alignment.scale.is_finite() || alignment.scale <= 1e-12 {
            return Err(SceneError::InvalidInput(
                "landmark alignment produced a degenerate scale",
            ));
        }
        let r_delta = alignment.rotation_matrix();
        let t_delta = alignment.motion.translation;
        let r_old = orthonormalize(old.rotation);
        let c = alignment.scale;
        let rotation = orthonormalize(std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                r_delta[i][0] * r_old[0][j]
                    + r_delta[i][1] * r_old[1][j]
                    + r_delta[i][2] * r_old[2][j]
            })
        }));
        let translation = std::array::from_fn(|i| {
            c * (r_delta[i][0] * old.translation[0]
                + r_delta[i][1] * old.translation[1]
                + r_delta[i][2] * old.translation[2])
                + t_delta[i]
        });
        let next = Placement {
            rotation,
            translation,
            scale: old.scale.map(|s| c * s),
        };
        self.set_placement(moving, next)?;
        Ok(alignment)
    }

    /// Runs multi-stage global + fine registration ([`register_multistage`]) of
    /// `moving` onto `fixed` and updates `moving`'s placement in the scene as a
    /// single undoable edit step when a valid correspondence pose is found.
    ///
    /// Returns [`RegistrationReport`] with world-space delta `motion` and
    /// column-major `transform` (transforming `moving`'s pre-call world points to
    /// its aligned world pose).
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id, [`SceneError::InvalidInput`]
    /// if `moving == fixed`, [`SceneError::StaleBvh`] for an un-updated Manual
    /// `fixed`, or [`SceneError::Ops`] on invalid options or empty meshes.
    pub fn register_multistage(
        &mut self,
        moving: MeshId,
        fixed: MeshId,
        options: &RegistrationOptions,
    ) -> Result<RegistrationReport> {
        self.drop_field_unit();
        self.entry(moving)?;
        self.entry(fixed)?;
        if moving == fixed {
            return Err(SceneError::InvalidInput("cannot register a mesh to itself"));
        }
        self.prepare(Some(&[fixed]))?;
        let (m, f) = (self.entry(moving)?, self.entry(fixed)?);
        let frame_m = frame_of(&m.placement, m.mesh.origin());
        let frame_f = frame_of(&f.placement, f.mesh.origin());
        let moving_scale = m.placement.scale;
        let moving_origin = m.mesh.origin();
        let initial_rigid = compose(&frame_f.inverse(), &frame_m);
        let initial_motion = rigid_to_motion(&initial_rigid);

        let baked_src = maybe_baked(&m.mesh, &m.placement, &m.bvh)?;
        let baked_tgt = maybe_baked(&f.mesh, &f.placement, &f.bvh)?;
        let src_mesh = baked_src.as_ref().map_or(&m.mesh, |(mesh, _)| mesh);
        let (tgt_mesh, tgt_bvh) = baked_tgt
            .as_ref()
            .map_or((&f.mesh, &f.bvh), |(mesh, bvh)| (mesh, bvh));

        let mut report =
            register_multistage(src_mesh, tgt_mesh, tgt_bvh, Some(initial_motion), options)?;
        if report.final_rmse.is_finite() {
            let solved = motion_to_rigid(&report.motion);
            let out = compose(&frame_f, &solved);
            let next = placement_from_scaled_frame(&out, moving_scale, moving_origin);
            self.set_placement(moving, next)?;
            let delta_world = compose(&out, &frame_m.inverse());
            report.motion = rigid_to_motion(&delta_world);
            report.transform = rigid_to_matrix4(&delta_world, 1.0);
        }
        Ok(report)
    }

    /// Detects the dominant reflection symmetry plane of `id` in world coordinates.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id, [`SceneError::StaleBvh`] for an
    /// un-updated Manual mesh, or [`SceneError::Ops`] when the mesh has no triangles.
    pub fn detect_symmetry_plane(&mut self, id: MeshId) -> Result<SymmetryPlaneReport> {
        self.drop_field_unit();
        self.entry(id)?;
        self.prepare(Some(&[id]))?;
        let entry = self.entry(id)?;
        let frame = frame_of(&entry.placement, entry.mesh.origin());
        let baked = maybe_baked(&entry.mesh, &entry.placement, &entry.bvh)?;
        let (mesh, bvh) = baked
            .as_ref()
            .map_or((&entry.mesh, &entry.bvh), |(m, b)| (m, b));
        let local = detect_symmetry_plane(mesh, bvh)?;
        let r = &frame.rotation;
        let world_n = canonicalize_normal(std::array::from_fn(|i| {
            r[i][0] * local.plane_normal[0]
                + r[i][1] * local.plane_normal[1]
                + r[i][2] * local.plane_normal[2]
        }));
        let world_p = std::array::from_fn(|i| {
            r[i][0] * local.plane_point[0]
                + r[i][1] * local.plane_point[1]
                + r[i][2] * local.plane_point[2]
                + frame.translation[i]
        });
        Ok(SymmetryPlaneReport {
            plane_normal: world_n,
            plane_point: world_p,
            symmetry_error_rms: local.symmetry_error_rms,
            symmetry_score: local.symmetry_score,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MeshOptions, SceneOptions};
    use fast_mesh_ops::align::{CoarseAlignMethod, IcpObjective};

    fn asymmetric_wedge_mesh(shift: [f32; 3]) -> Mesh {
        // Asymmetric hexahedral wedge: distinct principal axes (width 3, depth 2, sloped height 1..1.8).
        #[rustfmt::skip]
        let pos: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
            [3.0, 0.0, 1.8],
            [2.4, 2.0, 1.1],
            [0.0, 2.0, 1.4],
        ]
        .into_iter()
        .map(|p| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]])
        .collect();
        #[rustfmt::skip]
        let flat_pos: Vec<f32> = pos.iter().flat_map(|p| *p).collect();
        #[rustfmt::skip]
        let indices: Vec<u32> = vec![
            0, 2, 1, 0, 3, 2, // bottom
            4, 5, 6, 4, 6, 7, // top
            0, 1, 5, 0, 5, 4, // front
            1, 2, 6, 1, 6, 5, // right
            2, 3, 7, 2, 7, 6, // back
            3, 0, 4, 3, 4, 7, // left
        ];
        Mesh::from_buffers(&flat_pos, &indices).expect("valid wedge")
    }

    fn box_buffers(hx: f32, hy: f32, hz: f32, center: [f32; 3]) -> Mesh {
        let [cx, cy, cz] = center;
        let flat_pos: Vec<f32> = vec![
            cx - hx,
            cy - hy,
            cz - hz,
            cx + hx,
            cy - hy,
            cz - hz,
            cx + hx,
            cy + hy,
            cz - hz,
            cx - hx,
            cy + hy,
            cz - hz,
            cx - hx,
            cy - hy,
            cz + hz,
            cx + hx,
            cy - hy,
            cz + hz,
            cx + hx,
            cy + hy,
            cz + hz,
            cx - hx,
            cy + hy,
            cz + hz,
        ];
        let indices: Vec<u32> = vec![
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7,
            6, 3, 0, 4, 3, 4, 7,
        ];
        Mesh::from_buffers(&flat_pos, &indices).expect("valid box")
    }

    fn rot_z(angle: f64) -> [[f64; 3]; 3] {
        let (s, c) = angle.sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    #[test]
    fn scene_align_by_landmarks_is_single_undo_step_and_supports_scaling() {
        let mut scene = Scene::new(SceneOptions::default());
        let fixed_mesh = asymmetric_wedge_mesh([0.0, 0.0, 0.0]);
        let moving_mesh = asymmetric_wedge_mesh([0.0, 0.0, 0.0]);
        let fixed = scene.add_mesh(fixed_mesh, MeshOptions::default()).unwrap();
        let moving = scene.add_mesh(moving_mesh, MeshOptions::default()).unwrap();

        let initial_placement = Placement {
            rotation: rot_z(0.75),
            translation: [4.5, -2.0, 1.25],
            scale: [1.0, 1.0, 1.0],
        };
        scene.set_placement(moving, initial_placement).unwrap();

        // Target world placement we expect moving to align with (scaled 2x + shifted + rotated).
        let target_placement = Placement {
            rotation: rot_z(-0.4),
            translation: [-1.0, 3.0, 0.5],
            scale: [2.0, 2.0, 2.0],
        };
        scene.set_placement(fixed, target_placement).unwrap();

        let sample_pts = [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 2.0, 0.0],
            [0.0, 2.0, 1.4],
        ];
        let pairs: Vec<LandmarkPair> = sample_pts
            .iter()
            .map(|&p| LandmarkPair {
                source: initial_placement.apply(p),
                target: target_placement.apply(p),
            })
            .collect();

        let report = scene
            .align_by_landmarks(moving, fixed, &pairs, true)
            .expect("landmark alignment succeeds");
        assert!(report.rmse < 1e-10, "rmse was {}", report.rmse);
        assert!((report.scale - 2.0).abs() < 1e-10);

        let aligned = scene.placement(moving).unwrap();
        for p in sample_pts {
            let got = aligned.apply(p);
            let want = target_placement.apply(p);
            let err = ((got[0] - want[0]).powi(2)
                + (got[1] - want[1]).powi(2)
                + (got[2] - want[2]).powi(2))
            .sqrt();
            assert!(err < 1e-9, "aligned point error {err}");
        }

        // Verify a single undo step restores initial_placement cleanly.
        assert!(scene.undo().unwrap());
        let after_undo = scene.placement(moving).unwrap();
        assert_eq!(after_undo, initial_placement);
        assert!(scene.redo().unwrap());
        assert_eq!(scene.placement(moving).unwrap(), aligned);
    }

    #[test]
    fn scene_register_multistage_recovers_large_rotation_and_is_undoable() {
        let mut scene = Scene::new(SceneOptions::default());
        let fixed_mesh = asymmetric_wedge_mesh([1.0, -0.5, 0.25]);
        let moving_mesh = asymmetric_wedge_mesh([1.0, -0.5, 0.25]);
        let fixed = scene.add_mesh(fixed_mesh, MeshOptions::default()).unwrap();
        let moving = scene.add_mesh(moving_mesh, MeshOptions::default()).unwrap();

        let initial_placement = Placement {
            rotation: rot_z(std::f64::consts::FRAC_PI_2),
            translation: [12.0, -7.0, 4.0],
            scale: [1.5, 1.5, 1.5],
        };
        let fixed_placement = Placement {
            rotation: rot_z(0.25),
            translation: [-2.0, 1.5, -1.0],
            scale: [1.5, 1.5, 1.5],
        };
        scene.set_placement(moving, initial_placement).unwrap();
        scene.set_placement(fixed, fixed_placement).unwrap();

        let opts = RegistrationOptions {
            coarse_method: CoarseAlignMethod::OrientedBoundingBox,
            objective: IcpObjective::SymmetricNormal,
            max_iterations: 60,
            tolerance: 1e-7,
            overlap: 1.0,
            max_distance: None,
            target_rmse: 1e-7,
            sample_count: 4000,
            anderson_window: 3,
        };
        let report = scene
            .register_multistage(moving, fixed, &opts)
            .expect("multistage succeeds");
        assert!(report.converged, "status was {:?}", report.status);
        assert!(report.final_rmse < 1e-4, "rmse {}", report.final_rmse);

        let aligned = scene.placement(moving).unwrap();
        let probe = [1.5, 0.5, 0.75];
        let got = aligned.apply(probe);
        let want = fixed_placement.apply(probe);
        let err =
            ((got[0] - want[0]).powi(2) + (got[1] - want[1]).powi(2) + (got[2] - want[2]).powi(2))
                .sqrt();
        assert!(err < 1e-3, "aligned probe error {err}");

        assert!(scene.undo().unwrap());
        assert_eq!(scene.placement(moving).unwrap(), initial_placement);
    }

    #[test]
    fn scene_detect_symmetry_plane_transforms_with_placement() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene
            .add_mesh(
                box_buffers(2.0, 1.0, 0.5, [0.0, 0.0, 0.0]),
                MeshOptions::default(),
            )
            .unwrap();
        let placed = Placement {
            rotation: rot_z(std::f64::consts::FRAC_PI_4),
            translation: [5.0, -3.0, 2.0],
            scale: [2.0, 1.0, 1.0],
        };
        scene.set_placement(id, placed).unwrap();
        let sym = scene.detect_symmetry_plane(id).unwrap();
        assert!(sym.symmetry_score > 0.99, "score {}", sym.symmetry_score);
        // Plane must pass near the placed centroid `[5.0, -3.0, 2.0]`.
        let diff = [
            sym.plane_point[0] - 5.0,
            sym.plane_point[1] + 3.0,
            sym.plane_point[2] - 2.0,
        ];
        let plane_dist = (diff[0] * sym.plane_normal[0]
            + diff[1] * sym.plane_normal[1]
            + diff[2] * sym.plane_normal[2])
            .abs();
        assert!(plane_dist < 1e-3, "plane distance to center {plane_dist}");
    }

    #[test]
    fn scene_registration_rejects_self_and_stale_ids() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene
            .add_mesh(
                asymmetric_wedge_mesh([0.0, 0.0, 0.0]),
                MeshOptions::default(),
            )
            .unwrap();
        assert!(matches!(
            scene.register_multistage(id, id, &RegistrationOptions::default()),
            Err(SceneError::InvalidInput(_))
        ));
        assert!(matches!(
            scene.align_by_landmarks(id, id, &[], false),
            Err(SceneError::InvalidInput(_))
        ));
    }
}
