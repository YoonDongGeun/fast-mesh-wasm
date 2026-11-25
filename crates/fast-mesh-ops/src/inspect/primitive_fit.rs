//! Analytical primitive fitting (Plane, Sphere, Cylinder, Cone) and datum alignment.

use std::f64::consts::FRAC_PI_2;

use fast_mesh_query::Rigid;
use fast_mesh_store::{Mesh, TriangleId, VertexId};

use crate::align::jacobi_eigen_3x3;
use crate::error::{OpsError, Result};
use crate::icp::{add, cross, dot, norm, sub, unit};
use crate::{allocation_allowed, filled, push, reserve};

/// Primitive geometry family to fit or automatically classify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveKind {
    /// Infinite plane defined by origin and unit normal.
    Plane,
    /// Sphere defined by center and radius.
    Sphere,
    /// Circular cylinder defined by axis point, unit axis direction, radius, and height.
    Cylinder,
    /// Right circular cone defined by apex, unit axis direction, half-angle (rad), and height.
    Cone,
    /// Automatically evaluate all primitive families and pick the lowest-error valid model.
    Auto,
}

/// Fitted analytical surface primitive in world coordinates.
#[derive(Debug, Clone, PartialEq)]
pub enum FittedPrimitive {
    /// Plane passing through `origin` with unit `normal`.
    Plane {
        /// Point lying on the fitted plane (inlier centroid).
        origin: [f64; 3],
        /// Unit plane normal vector.
        normal: [f64; 3],
    },
    /// Sphere with `center` and positive `radius`.
    Sphere {
        /// Sphere center in world coordinates.
        center: [f64; 3],
        /// Positive sphere radius.
        radius: f64,
    },
    /// Circular cylinder bounded axially by its inlier span.
    Cylinder {
        /// Midpoint on the cylinder axis spanning its inliers.
        axis_origin: [f64; 3],
        /// Unit axis direction vector.
        axis_dir: [f64; 3],
        /// Positive cylinder radius.
        radius: f64,
        /// Axial span height across inliers.
        height: f64,
    },
    /// Single-nappe right circular cone with vertex at `apex`.
    Cone {
        /// Cone apex (vertex) position in world coordinates.
        apex: [f64; 3],
        /// Unit axis direction pointing from `apex` into the cone nappe.
        axis_dir: [f64; 3],
        /// Half-opening angle in radians, strictly in `(0, pi/2)`.
        half_angle_rad: f64,
        /// Axial span height across inliers.
        height: f64,
    },
}

impl FittedPrimitive {
    /// The primitive moved by `by`.
    fn moved(mut self, by: [f64; 3]) -> Self {
        let point = match &mut self {
            Self::Plane { origin, .. } => origin,
            Self::Sphere { center, .. } => center,
            Self::Cylinder { axis_origin, .. } => axis_origin,
            Self::Cone { apex, .. } => apex,
        };
        *point = add(*point, by);
        self
    }
}

/// Configuration options for RANSAC outlier rejection and nonlinear refinement.
#[derive(Debug, Clone, PartialEq)]
pub struct FitPrimitiveOptions {
    /// Distance threshold for inlier classification (`0.0` derives `0.02 * diag`).
    pub inlier_threshold: f64,
    /// Number of deterministic RANSAC sampling trials per candidate model.
    pub ransac_iterations: usize,
    /// Maximum Levenberg-Marquardt nonlinear optimization iterations.
    pub lm_iterations: usize,
    /// Seed for the deterministic PRNG.
    pub seed: u64,
    /// Minimum fraction of points required as inliers in `[0.0, 1.0]`.
    pub min_inlier_ratio: f64,
    /// Relative weight on surface normal agreement when ranking RANSAC hypotheses.
    pub normal_weight: f64,
}

impl Default for FitPrimitiveOptions {
    fn default() -> Self {
        Self {
            inlier_threshold: 0.0,
            ransac_iterations: 256,
            lm_iterations: 50,
            seed: 0x5EED_2340_1234_5678,
            min_inlier_ratio: 0.2,
            normal_weight: 0.1,
        }
    }
}

/// Result summary and per-sample signed residuals from primitive fitting.
#[derive(Debug, Clone, PartialEq)]
pub struct FitPrimitiveReport {
    /// Fitted primitive geometry.
    pub primitive: FittedPrimitive,
    /// Root-mean-square orthogonal distance across inliers (or all samples if none).
    pub rms_error: f64,
    /// Maximum absolute orthogonal distance across inliers (or all samples if none).
    pub max_error: f64,
    /// Number of samples within `inlier_threshold`.
    pub inlier_count: usize,
    /// Fraction of evaluated samples classified as inliers.
    pub inlier_ratio: f64,
    /// Signed orthogonal distance residuals per input point or mesh vertex slot.
    pub residuals: Vec<f32>,
}

/// Fit an analytical primitive to a slice of 3D points (and optional normals).
///
/// The points are `f32`, which far from zero holds them only to its steps there (0.008 at 1e5).
/// Subtract a common point from them in `f64` first and add it to the primitive after, as
/// [`fit_primitive_mesh`] does with [`Mesh::origin`].
pub fn fit_primitive_points(
    points: &[[f32; 3]],
    normals: Option<&[[f32; 3]]>,
    kind: PrimitiveKind,
    options: &FitPrimitiveOptions,
) -> Result<FitPrimitiveReport> {
    validate_options(options)?;
    let min_req = min_points_for_kind(kind);
    if points.len() < min_req {
        return Err(OpsError::InvalidInput("too few points for primitive fit"));
    }
    if let Some(nrm) = normals
        && nrm.len() != points.len()
    {
        return Err(OpsError::InvalidInput(
            "normals slice length must match points",
        ));
    }

    allocation_allowed()?;
    let mut pts_f64 = Vec::new();
    reserve(&mut pts_f64, points.len())?;
    for &p in points {
        if !p.iter().all(|v| v.is_finite()) {
            return Err(OpsError::InvalidInput("point coordinates must be finite"));
        }
        pts_f64.push([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
    }

    let nrm_f64 = match normals {
        Some(slice) => {
            let mut v = Vec::new();
            reserve(&mut v, slice.len())?;
            for &n in slice {
                if !n.iter().all(|c| c.is_finite()) {
                    return Err(OpsError::InvalidInput("normal components must be finite"));
                }
                let nd = [f64::from(n[0]), f64::from(n[1]), f64::from(n[2])];
                v.push(unit(nd).unwrap_or([0.0, 0.0, 0.0]));
            }
            Some(v)
        }
        None => None,
    };

    fit_points_f64(&pts_f64, nrm_f64.as_deref(), kind, options)
}

/// Fit an analytical primitive to vertices of a mesh (all alive vertices or selected triangles).
///
/// Fitted around [`Mesh::origin`], so the primitive, in world coordinates, is as precise far from
/// the world origin as near it.
pub fn fit_primitive_mesh(
    mesh: &Mesh,
    triangles: Option<&[TriangleId]>,
    kind: PrimitiveKind,
    options: &FitPrimitiveOptions,
) -> Result<FitPrimitiveReport> {
    validate_options(options)?;
    let v_slots = mesh.vertex_slots();
    if v_slots == 0 {
        return Err(OpsError::InvalidInput("mesh has no vertices"));
    }

    allocation_allowed()?;
    let mut selected = filled(v_slots, false)?;
    let mut accum_nrm = filled(v_slots, [0.0f64; 3])?;
    let position = |v: VertexId| mesh.position(v).map(|p| p.map(f64::from));

    match triangles {
        Some(tri_ids) => {
            if tri_ids.is_empty() {
                return Err(OpsError::InvalidInput("triangle selection is empty"));
            }
            for &tid in tri_ids {
                let Some([a, b, c]) = mesh.triangle(tid) else {
                    return Err(OpsError::InvalidInput(
                        "selected triangle does not exist in mesh",
                    ));
                };
                selected[a as usize] = true;
                selected[b as usize] = true;
                selected[c as usize] = true;
                let (Some(pa), Some(pb), Some(pc)) = (position(a), position(b), position(c)) else {
                    return Err(OpsError::InvalidInput("triangle references dead vertex"));
                };
                let area_n = cross(sub(pb, pa), sub(pc, pa));
                accum_nrm[a as usize] = add(accum_nrm[a as usize], area_n);
                accum_nrm[b as usize] = add(accum_nrm[b as usize], area_n);
                accum_nrm[c as usize] = add(accum_nrm[c as usize], area_n);
            }
        }
        None => {
            for ti in 0..mesh.triangle_slots() {
                let tid = ti as TriangleId;
                let Some([a, b, c]) = mesh.triangle(tid) else {
                    continue;
                };
                let (Some(pa), Some(pb), Some(pc)) = (position(a), position(b), position(c)) else {
                    continue;
                };
                let area_n = cross(sub(pb, pa), sub(pc, pa));
                accum_nrm[a as usize] = add(accum_nrm[a as usize], area_n);
                accum_nrm[b as usize] = add(accum_nrm[b as usize], area_n);
                accum_nrm[c as usize] = add(accum_nrm[c as usize], area_n);
            }
            for (vi, flag) in selected.iter_mut().enumerate() {
                if mesh.is_vertex_alive(vi as VertexId) {
                    *flag = true;
                }
            }
        }
    }

    let mut indices = Vec::new();
    let mut pts = Vec::new();
    let mut nrms = Vec::new();
    let mut any_nrm = false;
    for (vi, &is_sel) in selected.iter().enumerate() {
        if !is_sel {
            continue;
        }
        let Some(p) = position(vi as VertexId) else {
            continue;
        };
        if !p.iter().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput("mesh contains non-finite vertex"));
        }
        push(&mut indices, vi)?;
        push(&mut pts, p)?;
        let n = unit(accum_nrm[vi]).unwrap_or([0.0, 0.0, 0.0]);
        if norm(n) > 0.5 {
            any_nrm = true;
        }
        push(&mut nrms, n)?;
    }

    let min_req = min_points_for_kind(kind);
    if pts.len() < min_req {
        return Err(OpsError::InvalidInput(
            "too few selected mesh vertices for primitive fit",
        ));
    }

    let report = fit_points_f64(
        &pts,
        if any_nrm { Some(&nrms) } else { None },
        kind,
        options,
    )?;

    let mut full_residuals = filled(v_slots, 0.0f32)?;
    for (local_i, &slot) in indices.iter().enumerate() {
        full_residuals[slot] = report.residuals[local_i];
    }

    Ok(FitPrimitiveReport {
        primitive: report.primitive.moved(mesh.origin()),
        rms_error: report.rms_error,
        max_error: report.max_error,
        inlier_count: report.inlier_count,
        inlier_ratio: report.inlier_ratio,
        residuals: full_residuals,
    })
}

/// Compute a rigid transform aligning a fitted primitive's datum axis/origin to `target_axis`/`target_origin`.
pub fn datum_alignment_rigid(
    primitive: &FittedPrimitive,
    target_axis: [f64; 3],
    target_origin: [f64; 3],
) -> Result<Rigid> {
    let tgt_dir = unit(target_axis).ok_or(OpsError::InvalidInput(
        "target_axis must be non-zero finite",
    ))?;
    if !target_origin.iter().all(|v| v.is_finite()) {
        return Err(OpsError::InvalidInput("target_origin must be finite"));
    }

    let (src_origin, src_dir_opt) = match primitive {
        FittedPrimitive::Plane { origin, normal } => (
            *origin,
            Some(unit(*normal).ok_or(OpsError::InvalidInput("invalid plane normal"))?),
        ),
        FittedPrimitive::Sphere { center, .. } => (*center, None),
        FittedPrimitive::Cylinder {
            axis_origin,
            axis_dir,
            ..
        } => (
            *axis_origin,
            Some(unit(*axis_dir).ok_or(OpsError::InvalidInput("invalid cylinder axis"))?),
        ),
        FittedPrimitive::Cone { apex, axis_dir, .. } => (
            *apex,
            Some(unit(*axis_dir).ok_or(OpsError::InvalidInput("invalid cone axis"))?),
        ),
    };

    if !src_origin.iter().all(|v| v.is_finite()) {
        return Err(OpsError::InvalidInput("primitive origin must be finite"));
    }

    let rot = match src_dir_opt {
        None => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        Some(src_dir) => shortest_arc_matrix(src_dir, tgt_dir),
    };

    let rotated_origin = mat_mul_vec(rot, src_origin);
    let translation = sub(target_origin, rotated_origin);
    let rigid = Rigid {
        rotation: rot,
        translation,
    };
    rigid.validate()?;
    Ok(rigid)
}

// ============================================================================
// Core Pipeline (Normalized RANSAC + LM Refinement + Auto Model Selection)
// ============================================================================

fn validate_options(options: &FitPrimitiveOptions) -> Result<()> {
    if !options.inlier_threshold.is_finite() || options.inlier_threshold < 0.0 {
        return Err(OpsError::InvalidInput(
            "inlier_threshold must be non-negative and finite",
        ));
    }
    if !options.min_inlier_ratio.is_finite() || !(0.0..=1.0).contains(&options.min_inlier_ratio) {
        return Err(OpsError::InvalidInput(
            "min_inlier_ratio must be in [0.0, 1.0]",
        ));
    }
    if !options.normal_weight.is_finite() || options.normal_weight < 0.0 {
        return Err(OpsError::InvalidInput(
            "normal_weight must be non-negative and finite",
        ));
    }
    Ok(())
}

fn min_points_for_kind(kind: PrimitiveKind) -> usize {
    match kind {
        PrimitiveKind::Plane => 3,
        PrimitiveKind::Sphere => 4,
        PrimitiveKind::Cylinder => 5,
        PrimitiveKind::Cone => 6,
        PrimitiveKind::Auto => 3,
    }
}

fn fit_points_f64(
    points: &[[f64; 3]],
    input_normals: Option<&[[f64; 3]]>,
    kind: PrimitiveKind,
    options: &FitPrimitiveOptions,
) -> Result<FitPrimitiveReport> {
    let n = points.len();
    let (mut bmin, mut bmax) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for &p in points {
        for k in 0..3 {
            bmin[k] = bmin[k].min(p[k]);
            bmax[k] = bmax[k].max(p[k]);
        }
    }
    let mu = [
        0.5 * (bmin[0] + bmax[0]),
        0.5 * (bmin[1] + bmax[1]),
        0.5 * (bmin[2] + bmax[2]),
    ];
    let diag = norm(sub(bmax, bmin));
    if !diag.is_finite() || diag <= 1e-12 {
        return Err(OpsError::InvalidInput(
            "input points are coincident (zero bounding diagonal)",
        ));
    }
    let inv_diag = 1.0 / diag;

    let mut q_pts = Vec::new();
    reserve(&mut q_pts, n)?;
    for &p in points {
        q_pts.push([
            (p[0] - mu[0]) * inv_diag,
            (p[1] - mu[1]) * inv_diag,
            (p[2] - mu[2]) * inv_diag,
        ]);
    }

    let tau_norm = if options.inlier_threshold > 0.0 {
        (options.inlier_threshold * inv_diag).max(1e-7)
    } else {
        0.02
    };

    let (work_normals, has_oriented_normals): (Vec<[f64; 3]>, bool) = match input_normals {
        Some(nrm) => (nrm.to_vec(), true),
        None => {
            if n >= 6 {
                (estimate_local_pca_normals(&q_pts, 12.min(n - 1)), false)
            } else {
                (vec![[0.0, 0.0, 0.0]; n], false)
            }
        }
    };

    let ctx = FitContext {
        pts: &q_pts,
        normals: &work_normals,
        has_oriented_normals,
        tau: tau_norm,
        options,
    };

    let best_model = match kind {
        PrimitiveKind::Plane => Some(fit_plane_norm(&ctx)?),
        PrimitiveKind::Sphere => Some(fit_sphere_norm(&ctx)?),
        PrimitiveKind::Cylinder => Some(fit_cylinder_norm(&ctx)?),
        PrimitiveKind::Cone => Some(fit_cone_norm(&ctx)?),
        PrimitiveKind::Auto => select_best_auto(&ctx)?,
    };

    let norm_model = best_model.ok_or(OpsError::InvalidInput(
        "failed to fit valid primitive within constraints",
    ))?;

    let primitive = denormalize_primitive(&norm_model, mu, diag);
    build_report(points, primitive, options, tau_norm * diag)
}

struct FitContext<'a> {
    pts: &'a [[f64; 3]],
    normals: &'a [[f64; 3]],
    has_oriented_normals: bool,
    tau: f64,
    options: &'a FitPrimitiveOptions,
}

fn select_best_auto(ctx: &FitContext<'_>) -> Result<Option<FittedPrimitive>> {
    let n = ctx.pts.len();
    let mut candidates: Vec<(usize, FittedPrimitive)> = Vec::new();
    if n >= 3
        && let Ok(p) = fit_plane_norm(ctx)
    {
        candidates.push((3, p));
    }
    if n >= 4
        && let Ok(s) = fit_sphere_norm(ctx)
        && is_reasonable_sphere(&s)
    {
        candidates.push((4, s));
    }
    if n >= 5
        && let Ok(c) = fit_cylinder_norm(ctx)
        && is_reasonable_cylinder(&c)
    {
        candidates.push((5, c));
    }
    if n >= 6
        && let Ok(k) = fit_cone_norm(ctx)
        && is_reasonable_cone(&k)
    {
        candidates.push((6, k));
    }

    let mut best_score = f64::INFINITY;
    let mut best_meets_ratio = false;
    let mut best_prim = None;

    for (dof, prim) in candidates {
        let stats = eval_primitive_norm(
            ctx.pts,
            ctx.normals,
            ctx.has_oriented_normals,
            &prim,
            ctx.tau,
            ctx.options.normal_weight,
        );
        let meets_ratio = stats.inlier_ratio + 1e-12 >= ctx.options.min_inlier_ratio;
        let score = stats.msac_loss + (dof as f64) * 0.005;

        let better = match (meets_ratio, best_meets_ratio) {
            (true, false) => true,
            (false, true) => false,
            _ => score < best_score,
        };
        if better {
            best_score = score;
            best_meets_ratio = meets_ratio;
            best_prim = Some(prim);
        }
    }

    Ok(best_prim)
}

fn is_reasonable_sphere(p: &FittedPrimitive) -> bool {
    match p {
        FittedPrimitive::Sphere { center, radius } => {
            radius.is_finite() && *radius > 1e-4 && *radius <= 6.0 && norm(*center) <= 8.0
        }
        _ => false,
    }
}

fn is_reasonable_cylinder(p: &FittedPrimitive) -> bool {
    match p {
        FittedPrimitive::Cylinder {
            axis_origin,
            radius,
            height,
            ..
        } => {
            radius.is_finite()
                && height.is_finite()
                && *radius > 1e-4
                && *radius <= 6.0
                && *height > 1e-5
                && norm(*axis_origin) <= 8.0
        }
        _ => false,
    }
}

fn is_reasonable_cone(p: &FittedPrimitive) -> bool {
    match p {
        FittedPrimitive::Cone {
            apex,
            half_angle_rad,
            height,
            ..
        } => {
            half_angle_rad.is_finite()
                && height.is_finite()
                && *half_angle_rad >= 0.035
                && *half_angle_rad <= FRAC_PI_2 - 0.035
                && *height > 1e-5
                && norm(*apex) <= 12.0
        }
        _ => false,
    }
}

// ============================================================================
// Plane Fitting
// ============================================================================

fn fit_plane_norm(ctx: &FitContext<'_>) -> Result<FittedPrimitive> {
    let pts = ctx.pts;
    let n = pts.len();
    let mut rng = Rng::new(ctx.options.seed ^ 0xA11C_E001);

    let mut best_score = f64::INFINITY;
    let mut best_model =
        fit_plane_pca_indices(pts, &(0..n).collect::<Vec<_>>()).unwrap_or(FittedPrimitive::Plane {
            origin: pts[0],
            normal: [0.0, 0.0, 1.0],
        });

    let score = eval_primitive_norm(
        pts,
        ctx.normals,
        ctx.has_oriented_normals,
        &best_model,
        ctx.tau,
        ctx.options.normal_weight,
    )
    .msac_loss;
    best_score = best_score.min(score);

    let iters = ctx.options.ransac_iterations.max(1);
    for _ in 0..iters {
        let Some([i0, i1, i2]) = sample_distinct::<3>(&mut rng, n) else {
            break;
        };
        let Some(normal) = unit(cross(sub(pts[i1], pts[i0]), sub(pts[i2], pts[i0]))) else {
            continue;
        };
        let origin = [
            (pts[i0][0] + pts[i1][0] + pts[i2][0]) / 3.0,
            (pts[i0][1] + pts[i1][1] + pts[i2][1]) / 3.0,
            (pts[i0][2] + pts[i1][2] + pts[i2][2]) / 3.0,
        ];
        let cand = FittedPrimitive::Plane { origin, normal };
        let st = eval_primitive_norm(
            pts,
            ctx.normals,
            ctx.has_oriented_normals,
            &cand,
            ctx.tau,
            ctx.options.normal_weight,
        );
        if st.msac_loss < best_score {
            best_score = st.msac_loss;
            best_model = cand;
        }
    }

    let mut current = best_model;
    for _ in 0..3 {
        let inliers = collect_inliers(pts, &current, ctx.tau);
        if inliers.len() < 3 {
            break;
        }
        if let Some(refined) = fit_plane_pca_indices(pts, &inliers) {
            current = refined;
        }
    }

    if let FittedPrimitive::Plane { origin, mut normal } = current {
        if ctx.has_oriented_normals {
            let inliers = collect_inliers(pts, &current, ctx.tau);
            let mut sum_dot = 0.0;
            for &idx in &inliers {
                sum_dot += dot(ctx.normals[idx], normal);
            }
            if sum_dot < 0.0 {
                normal = [-normal[0], -normal[1], -normal[2]];
            }
        }
        current = FittedPrimitive::Plane { origin, normal };
    }

    Ok(current)
}

fn fit_plane_pca_indices(pts: &[[f64; 3]], indices: &[usize]) -> Option<FittedPrimitive> {
    if indices.len() < 3 {
        return None;
    }
    let m = indices.len() as f64;
    let mut mean = [0.0; 3];
    for &i in indices {
        mean = add(mean, pts[i]);
    }
    mean = [mean[0] / m, mean[1] / m, mean[2] / m];

    let mut cov = [[0.0f64; 3]; 3];
    for &i in indices {
        let d = sub(pts[i], mean);
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += d[r] * d[c];
            }
        }
    }
    let (_vals, vecs) = jacobi_eigen_3x3(cov);
    let normal = unit([vecs[0][2], vecs[1][2], vecs[2][2]])?;
    Some(FittedPrimitive::Plane {
        origin: mean,
        normal,
    })
}

// ============================================================================
// Sphere Fitting
// ============================================================================

fn fit_sphere_norm(ctx: &FitContext<'_>) -> Result<FittedPrimitive> {
    let pts = ctx.pts;
    let n = pts.len();
    let mut rng = Rng::new(ctx.options.seed ^ 0x5F37_59DF);

    let all_idx: Vec<usize> = (0..n).collect();
    let mut best_model = fit_sphere_algebraic(pts, &all_idx).unwrap_or(FittedPrimitive::Sphere {
        center: [0.0, 0.0, 0.0],
        radius: 0.5,
    });
    let mut best_score = eval_primitive_norm(
        pts,
        ctx.normals,
        ctx.has_oriented_normals,
        &best_model,
        ctx.tau,
        ctx.options.normal_weight,
    )
    .msac_loss;

    let iters = ctx.options.ransac_iterations.max(1);
    for _ in 0..iters {
        let Some([i0, i1, i2, i3]) = sample_distinct::<4>(&mut rng, n) else {
            break;
        };
        let Some((center, radius)) = circumsphere_4pts(pts[i0], pts[i1], pts[i2], pts[i3]) else {
            continue;
        };
        if !(1e-4..=20.0).contains(&radius) || norm(center) > 25.0 {
            continue;
        }
        let cand = FittedPrimitive::Sphere { center, radius };
        let st = eval_primitive_norm(
            pts,
            ctx.normals,
            ctx.has_oriented_normals,
            &cand,
            ctx.tau,
            ctx.options.normal_weight,
        );
        if st.msac_loss < best_score {
            best_score = st.msac_loss;
            best_model = cand;
        }
    }

    let mut current = best_model;
    for pass in 0..2 {
        let tol = if pass == 0 { ctx.tau * 1.5 } else { ctx.tau };
        let inliers = collect_inliers(pts, &current, tol);
        let active = if inliers.len() >= 4 {
            &inliers
        } else {
            &all_idx
        };
        if pass == 0
            && let Some(alg) = fit_sphere_algebraic(pts, active)
            && eval_primitive_norm(
                pts,
                ctx.normals,
                ctx.has_oriented_normals,
                &alg,
                ctx.tau,
                ctx.options.normal_weight,
            )
            .msac_loss
                < best_score
        {
            current = alg;
        }
        current = refine_sphere_lm(pts, active, current, ctx.options.lm_iterations);
    }

    Ok(current)
}

fn circumsphere_4pts(
    p0: [f64; 3],
    p1: [f64; 3],
    p2: [f64; 3],
    p3: [f64; 3],
) -> Option<([f64; 3], f64)> {
    let a = [sub(p1, p0), sub(p2, p0), sub(p3, p0)];
    let b = [
        0.5 * dot(a[0], a[0]),
        0.5 * dot(a[1], a[1]),
        0.5 * dot(a[2], a[2]),
    ];
    let delta = solve_3x3(a, b)?;
    let center = add(p0, delta);
    let radius = norm(delta);
    if radius.is_finite() && radius > 1e-8 {
        Some((center, radius))
    } else {
        None
    }
}

fn fit_sphere_algebraic(pts: &[[f64; 3]], indices: &[usize]) -> Option<FittedPrimitive> {
    if indices.len() < 4 {
        return None;
    }
    let mut ata = [[0.0f64; 4]; 4];
    let mut atb = [0.0f64; 4];
    for &idx in indices {
        let p = pts[idx];
        let row = [2.0 * p[0], 2.0 * p[1], 2.0 * p[2], 1.0];
        let rhs = dot(p, p);
        for r in 0..4 {
            atb[r] += row[r] * rhs;
            for c in 0..4 {
                ata[r][c] += row[r] * row[c];
            }
        }
    }
    let x = solve_nxn::<4>(ata, atb)?;
    let center = [x[0], x[1], x[2]];
    let r2 = dot(center, center) + x[3];
    if r2 <= 1e-12 || !r2.is_finite() {
        return None;
    }
    Some(FittedPrimitive::Sphere {
        center,
        radius: r2.sqrt(),
    })
}

fn refine_sphere_lm(
    pts: &[[f64; 3]],
    indices: &[usize],
    initial: FittedPrimitive,
    max_iters: usize,
) -> FittedPrimitive {
    let FittedPrimitive::Sphere {
        mut center,
        mut radius,
    } = initial
    else {
        return initial;
    };
    if indices.len() < 4 || max_iters == 0 {
        return initial;
    }

    let mut lambda = 1e-3;
    let mut cost = sphere_cost(pts, indices, center, radius);

    for _ in 0..max_iters {
        let mut jtj = [[0.0f64; 4]; 4];
        let mut jtr = [0.0f64; 4];
        for &idx in indices {
            let d = sub(pts[idx], center);
            let dist = norm(d);
            if dist <= 1e-12 {
                continue;
            }
            let r = dist - radius;
            let inv = 1.0 / dist;
            let jac = [-d[0] * inv, -d[1] * inv, -d[2] * inv, -1.0];
            for a in 0..4 {
                jtr[a] += jac[a] * r;
                for b in 0..4 {
                    jtj[a][b] += jac[a] * jac[b];
                }
            }
        }
        let mut mat = jtj;
        for (a, row) in mat.iter_mut().enumerate() {
            row[a] += lambda * row[a].max(1e-6);
        }
        let neg_jtr = [-jtr[0], -jtr[1], -jtr[2], -jtr[3]];
        let Some(step) = solve_nxn::<4>(mat, neg_jtr) else {
            lambda *= 10.0;
            continue;
        };
        let step_norm =
            (step[0] * step[0] + step[1] * step[1] + step[2] * step[2] + step[3] * step[3]).sqrt();
        if step_norm < 1e-11 {
            break;
        }
        let cand_c = [
            center[0] + step[0],
            center[1] + step[1],
            center[2] + step[2],
        ];
        let cand_r = radius + step[3];
        if !cand_r.is_finite() || cand_r <= 1e-6 || cand_r > 20.0 {
            lambda *= 10.0;
            continue;
        }
        let cand_cost = sphere_cost(pts, indices, cand_c, cand_r);
        if cand_cost < cost {
            center = cand_c;
            radius = cand_r;
            let rel = (cost - cand_cost) / cost.max(1e-15);
            cost = cand_cost;
            lambda = (lambda * 0.3).max(1e-9);
            if rel < 1e-11 {
                break;
            }
        } else {
            lambda = (lambda * 5.0).min(1e9);
        }
    }

    FittedPrimitive::Sphere { center, radius }
}

fn sphere_cost(pts: &[[f64; 3]], indices: &[usize], center: [f64; 3], radius: f64) -> f64 {
    let mut sum = 0.0;
    for &idx in indices {
        let r = norm(sub(pts[idx], center)) - radius;
        sum += r * r;
    }
    sum
}

// ============================================================================
// Cylinder Fitting
// ============================================================================

fn fit_cylinder_norm(ctx: &FitContext<'_>) -> Result<FittedPrimitive> {
    let pts = ctx.pts;
    let n = pts.len();
    let mut rng = Rng::new(ctx.options.seed ^ 0xC11D_EE03);

    let mut mean = [0.0; 3];
    for &p in pts {
        mean = add(mean, p);
    }
    mean = [
        mean[0] / (n as f64),
        mean[1] / (n as f64),
        mean[2] / (n as f64),
    ];
    let mut cov = [[0.0; 3]; 3];
    for &p in pts {
        let d = sub(p, mean);
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += d[r] * d[c];
            }
        }
    }
    let (_vals, vecs) = jacobi_eigen_3x3(cov);
    let mut fallback_axes = vec![
        [vecs[0][0], vecs[1][0], vecs[2][0]],
        [vecs[0][1], vecs[1][1], vecs[2][1]],
        [vecs[0][2], vecs[1][2], vecs[2][2]],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    for k in 0..12 {
        let y = 1.0 - (2.0 * (k as f64) + 1.0) / 24.0;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let phi = (k as f64) * 2.399963229728653;
        fallback_axes.push([r * phi.cos(), y, r * phi.sin()]);
    }

    let mut best_score = f64::INFINITY;
    let mut best_model = FittedPrimitive::Cylinder {
        axis_origin: mean,
        axis_dir: [0.0, 0.0, 1.0],
        radius: 0.3,
        height: 1.0,
    };

    for &ax in &fallback_axes {
        if let Some([i0, i1, i2]) = sample_distinct::<3>(&mut rng, n)
            && let Some(cand) = cylinder_from_axis_3pts(ax, pts[i0], pts[i1], pts[i2])
        {
            let st = eval_primitive_norm(
                pts,
                ctx.normals,
                ctx.has_oriented_normals,
                &cand,
                ctx.tau,
                ctx.options.normal_weight,
            );
            if st.msac_loss < best_score {
                best_score = st.msac_loss;
                best_model = cand;
            }
        }
    }

    let iters = ctx.options.ransac_iterations.max(1);
    for iter in 0..iters {
        let axis_cand = if iter % 3 != 2 {
            let Some([i0, i1, i2]) = sample_distinct::<3>(&mut rng, n) else {
                break;
            };
            let n0 = ctx.normals[i0];
            let n1 = ctx.normals[i1];
            if let Some(ax) = unit(cross(n0, n1)) {
                cylinder_from_axis_3pts(ax, pts[i0], pts[i1], pts[i2])
            } else {
                None
            }
        } else if let Some([i0, i1, i2]) = sample_distinct::<3>(&mut rng, n) {
            let ax = fallback_axes[iter % fallback_axes.len()];
            cylinder_from_axis_3pts(ax, pts[i0], pts[i1], pts[i2])
        } else {
            None
        };

        if let Some(cand) = axis_cand {
            let st = eval_primitive_norm(
                pts,
                ctx.normals,
                ctx.has_oriented_normals,
                &cand,
                ctx.tau,
                ctx.options.normal_weight,
            );
            if st.msac_loss < best_score {
                best_score = st.msac_loss;
                best_model = cand;
            }
        }
    }

    let all_idx: Vec<usize> = (0..n).collect();
    let mut current = best_model;
    for pass in 0..2 {
        let tol = if pass == 0 { ctx.tau * 1.5 } else { ctx.tau };
        let inliers = collect_inliers(pts, &current, tol);
        let active = if inliers.len() >= 5 {
            &inliers
        } else {
            &all_idx
        };
        current = refine_cylinder_lm(pts, active, current, ctx.options.lm_iterations);
    }

    let final_inliers = collect_inliers(pts, &current, ctx.tau);
    let span_indices = if final_inliers.len() >= 2 {
        &final_inliers
    } else {
        &all_idx
    };
    Ok(finalize_cylinder_span(pts, span_indices, current))
}

fn cylinder_from_axis_3pts(
    axis_dir_raw: [f64; 3],
    p0: [f64; 3],
    p1: [f64; 3],
    p2: [f64; 3],
) -> Option<FittedPrimitive> {
    let a = canonical_dir(unit(axis_dir_raw)?);
    let (u, v) = ortho_basis(a);
    let q0 = [dot(p0, u), dot(p0, v)];
    let q1 = [dot(p1, u), dot(p1, v)];
    let q2 = [dot(p2, u), dot(p2, v)];
    let (c2d, radius) = circumcircle_2d(q0, q1, q2)?;
    if !(1e-4..=20.0).contains(&radius) || c2d[0].hypot(c2d[1]) > 25.0 {
        return None;
    }
    let t_mean = (dot(p0, a) + dot(p1, a) + dot(p2, a)) / 3.0;
    let axis_origin = [
        c2d[0] * u[0] + c2d[1] * v[0] + t_mean * a[0],
        c2d[0] * u[1] + c2d[1] * v[1] + t_mean * a[1],
        c2d[0] * u[2] + c2d[1] * v[2] + t_mean * a[2],
    ];
    Some(FittedPrimitive::Cylinder {
        axis_origin,
        axis_dir: a,
        radius,
        height: 1.0,
    })
}

fn circumcircle_2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<([f64; 2], f64)> {
    let bx = b[0] - a[0];
    let by = b[1] - a[1];
    let cx = c[0] - a[0];
    let cy = c[1] - a[1];
    let d = 2.0 * (bx * cy - by * cx);
    if d.abs() <= 1e-12 {
        return None;
    }
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let ux = (cy * b2 - by * c2) / d;
    let uy = (bx * c2 - cx * b2) / d;
    let r = ux.hypot(uy);
    if r.is_finite() && r > 1e-8 {
        Some(([a[0] + ux, a[1] + uy], r))
    } else {
        None
    }
}

fn refine_cylinder_lm(
    pts: &[[f64; 3]],
    indices: &[usize],
    initial: FittedPrimitive,
    max_iters: usize,
) -> FittedPrimitive {
    let FittedPrimitive::Cylinder {
        mut axis_origin,
        mut axis_dir,
        mut radius,
        height,
    } = initial
    else {
        return initial;
    };
    if indices.len() < 5 || max_iters == 0 {
        return initial;
    }

    axis_dir = unit(axis_dir).unwrap_or([0.0, 0.0, 1.0]);
    let mut lambda = 1e-3;
    let mut cost = cylinder_cost(pts, indices, axis_origin, axis_dir, radius);

    for _ in 0..max_iters {
        let (u, v) = ortho_basis(axis_dir);
        let mut jtj = [[0.0f64; 5]; 5];
        let mut jtr = [0.0f64; 5];

        for &idx in indices {
            let d = sub(pts[idx], axis_origin);
            let z = dot(d, axis_dir);
            let du = dot(d, u);
            let dv = dot(d, v);
            let rho = du.hypot(dv);
            if rho <= 1e-12 {
                continue;
            }
            let r = rho - radius;
            let eu = du / rho;
            let ev = dv / rho;
            let jac = [-eu, -ev, -z * eu, -z * ev, -1.0];
            for a in 0..5 {
                jtr[a] += jac[a] * r;
                for b in 0..5 {
                    jtj[a][b] += jac[a] * jac[b];
                }
            }
        }

        let mut mat = jtj;
        for (a, row) in mat.iter_mut().enumerate() {
            row[a] += lambda * row[a].max(1e-6);
        }
        let neg_jtr = [-jtr[0], -jtr[1], -jtr[2], -jtr[3], -jtr[4]];
        let Some(step) = solve_nxn::<5>(mat, neg_jtr) else {
            lambda *= 10.0;
            continue;
        };
        let step_norm = step.iter().map(|s| s * s).sum::<f64>().sqrt();
        if step_norm < 1e-11 {
            break;
        }
        let cand_o = [
            axis_origin[0] + step[0] * u[0] + step[1] * v[0],
            axis_origin[1] + step[0] * u[1] + step[1] * v[1],
            axis_origin[2] + step[0] * u[2] + step[1] * v[2],
        ];
        let cand_a_raw = [
            axis_dir[0] + step[2] * u[0] + step[3] * v[0],
            axis_dir[1] + step[2] * u[1] + step[3] * v[1],
            axis_dir[2] + step[2] * u[2] + step[3] * v[2],
        ];
        let Some(cand_a) = unit(cand_a_raw) else {
            lambda *= 10.0;
            continue;
        };
        let cand_r = radius + step[4];
        if !cand_r.is_finite() || cand_r <= 1e-6 || cand_r > 20.0 {
            lambda *= 10.0;
            continue;
        }
        let cand_cost = cylinder_cost(pts, indices, cand_o, cand_a, cand_r);
        if cand_cost < cost {
            axis_origin = cand_o;
            axis_dir = cand_a;
            radius = cand_r;
            let rel = (cost - cand_cost) / cost.max(1e-15);
            cost = cand_cost;
            lambda = (lambda * 0.3).max(1e-9);
            if rel < 1e-11 {
                break;
            }
        } else {
            lambda = (lambda * 5.0).min(1e9);
        }
    }

    FittedPrimitive::Cylinder {
        axis_origin,
        axis_dir: canonical_dir(axis_dir),
        radius,
        height,
    }
}

fn cylinder_cost(
    pts: &[[f64; 3]],
    indices: &[usize],
    axis_origin: [f64; 3],
    axis_dir: [f64; 3],
    radius: f64,
) -> f64 {
    let mut sum = 0.0;
    for &idx in indices {
        let d = sub(pts[idx], axis_origin);
        let z = dot(d, axis_dir);
        let radial = sub(d, [axis_dir[0] * z, axis_dir[1] * z, axis_dir[2] * z]);
        let r = norm(radial) - radius;
        sum += r * r;
    }
    sum
}

fn finalize_cylinder_span(
    pts: &[[f64; 3]],
    indices: &[usize],
    model: FittedPrimitive,
) -> FittedPrimitive {
    let FittedPrimitive::Cylinder {
        axis_origin,
        axis_dir,
        radius,
        ..
    } = model
    else {
        return model;
    };
    let (mut tmin, mut tmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for &idx in indices {
        let t = dot(sub(pts[idx], axis_origin), axis_dir);
        tmin = tmin.min(t);
        tmax = tmax.max(t);
    }
    if !tmin.is_finite() || !tmax.is_finite() {
        return FittedPrimitive::Cylinder {
            axis_origin,
            axis_dir,
            radius,
            height: 0.0,
        };
    }
    let mid = 0.5 * (tmin + tmax);
    let centered_origin = add(
        axis_origin,
        [axis_dir[0] * mid, axis_dir[1] * mid, axis_dir[2] * mid],
    );
    FittedPrimitive::Cylinder {
        axis_origin: centered_origin,
        axis_dir,
        radius,
        height: (tmax - tmin).max(1e-12),
    }
}

// ============================================================================
// Cone Fitting
// ============================================================================

fn fit_cone_norm(ctx: &FitContext<'_>) -> Result<FittedPrimitive> {
    let pts = ctx.pts;
    let n = pts.len();
    let mut rng = Rng::new(ctx.options.seed ^ 0xC0AE_2345);

    let mut mean = [0.0; 3];
    for &p in pts {
        mean = add(mean, p);
    }
    mean = [
        mean[0] / (n as f64),
        mean[1] / (n as f64),
        mean[2] / (n as f64),
    ];
    let mut cov = [[0.0; 3]; 3];
    for &p in pts {
        let d = sub(p, mean);
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += d[r] * d[c];
            }
        }
    }
    let (_vals, vecs) = jacobi_eigen_3x3(cov);
    let mut dir_seeds = vec![
        [vecs[0][0], vecs[1][0], vecs[2][0]],
        [-vecs[0][0], -vecs[1][0], -vecs[2][0]],
        [vecs[0][1], vecs[1][1], vecs[2][1]],
        [-vecs[0][1], -vecs[1][1], -vecs[2][1]],
        [vecs[0][2], vecs[1][2], vecs[2][2]],
        [-vecs[0][2], -vecs[1][2], -vecs[2][2]],
    ];
    for k in 0..8 {
        let y = 1.0 - (2.0 * (k as f64) + 1.0) / 16.0;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let phi = (k as f64) * 2.399963229728653;
        let d = [r * phi.cos(), y, r * phi.sin()];
        dir_seeds.push(d);
        dir_seeds.push([-d[0], -d[1], -d[2]]);
    }

    let mut best_score = f64::INFINITY;
    let mut best_model = FittedPrimitive::Cone {
        apex: [mean[0], mean[1], mean[2] - 0.5],
        axis_dir: [0.0, 0.0, 1.0],
        half_angle_rad: 0.35,
        height: 1.0,
    };

    for &d in &dir_seeds {
        if let Some(cand) = cone_from_axis_profile(pts, d) {
            let st = eval_primitive_norm(
                pts,
                ctx.normals,
                ctx.has_oriented_normals,
                &cand,
                ctx.tau,
                ctx.options.normal_weight,
            );
            if st.msac_loss < best_score {
                best_score = st.msac_loss;
                best_model = cand;
            }
        }
    }

    let iters = ctx.options.ransac_iterations.max(1);
    for _ in 0..iters {
        let Some([i0, i1, i2]) = sample_distinct::<3>(&mut rng, n) else {
            break;
        };
        if let Some(cand) = cone_from_3pt_normals(
            pts[i0],
            pts[i1],
            pts[i2],
            ctx.normals[i0],
            ctx.normals[i1],
            ctx.normals[i2],
        ) {
            let st = eval_primitive_norm(
                pts,
                ctx.normals,
                ctx.has_oriented_normals,
                &cand,
                ctx.tau,
                ctx.options.normal_weight,
            );
            if st.msac_loss < best_score {
                best_score = st.msac_loss;
                best_model = cand;
            }
        }
    }

    let all_idx: Vec<usize> = (0..n).collect();
    let mut current = best_model;
    for pass in 0..2 {
        let tol = if pass == 0 { ctx.tau * 1.5 } else { ctx.tau };
        let inliers = collect_inliers(pts, &current, tol);
        let active = if inliers.len() >= 6 {
            &inliers
        } else {
            &all_idx
        };
        current = refine_cone_lm(pts, active, current, ctx.options.lm_iterations);
    }

    let final_inliers = collect_inliers(pts, &current, ctx.tau);
    let span_indices = if final_inliers.len() >= 2 {
        &final_inliers
    } else {
        &all_idx
    };
    Ok(finalize_cone_span(pts, span_indices, current))
}

fn cone_from_3pt_normals(
    p0: [f64; 3],
    p1: [f64; 3],
    p2: [f64; 3],
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
) -> Option<FittedPrimitive> {
    if norm(n0) < 0.5 || norm(n1) < 0.5 || norm(n2) < 0.5 {
        return None;
    }
    let mat = [n0, n1, n2];
    let rhs = [dot(n0, p0), dot(n1, p1), dot(n2, p2)];
    let apex = solve_3x3(mat, rhs)?;
    if norm(apex) > 25.0 {
        return None;
    }
    let s0 = unit(sub(p0, apex))?;
    let s1 = unit(sub(p1, apex))?;
    let s2 = unit(sub(p2, apex))?;
    let mut axis = unit(cross(sub(s1, s0), sub(s2, s0)))?;
    if dot(axis, add(add(s0, s1), s2)) < 0.0 {
        axis = [-axis[0], -axis[1], -axis[2]];
    }
    let cos_phi = ((dot(s0, axis) + dot(s1, axis) + dot(s2, axis)) / 3.0).clamp(0.02, 0.999);
    let half_angle = cos_phi.acos();
    if !(0.02..=(FRAC_PI_2 - 0.02)).contains(&half_angle) {
        return None;
    }
    Some(FittedPrimitive::Cone {
        apex,
        axis_dir: axis,
        half_angle_rad: half_angle,
        height: 1.0,
    })
}

fn cone_from_axis_profile(pts: &[[f64; 3]], axis_raw: [f64; 3]) -> Option<FittedPrimitive> {
    let a = unit(axis_raw)?;
    let n = pts.len() as f64;
    let mut mean = [0.0; 3];
    for &p in pts {
        mean = add(mean, p);
    }
    mean = [mean[0] / n, mean[1] / n, mean[2] / n];

    let mut sz = 0.0;
    let mut sr = 0.0;
    let mut szz = 0.0;
    let mut szr = 0.0;
    for &p in pts {
        let d = sub(p, mean);
        let z = dot(d, a);
        let radial = sub(d, [a[0] * z, a[1] * z, a[2] * z]);
        let r = norm(radial);
        sz += z;
        sr += r;
        szz += z * z;
        szr += z * r;
    }
    let mz = sz / n;
    let mr = sr / n;
    let denom = szz - n * mz * mz;
    if denom.abs() <= 1e-12 {
        return None;
    }
    let slope = (szr - n * mz * mr) / denom;
    if !slope.is_finite() || slope.abs() < 0.03 {
        return None;
    }
    let intercept = mr - slope * mz;
    let z_apex = -intercept / slope;
    let apex = add(mean, [a[0] * z_apex, a[1] * z_apex, a[2] * z_apex]);
    let axis_dir = if slope >= 0.0 {
        a
    } else {
        [-a[0], -a[1], -a[2]]
    };
    let half_angle = slope.abs().atan();
    if !(0.02..=(FRAC_PI_2 - 0.02)).contains(&half_angle) || norm(apex) > 25.0 {
        return None;
    }
    Some(FittedPrimitive::Cone {
        apex,
        axis_dir,
        half_angle_rad: half_angle,
        height: 1.0,
    })
}

fn refine_cone_lm(
    pts: &[[f64; 3]],
    indices: &[usize],
    initial: FittedPrimitive,
    max_iters: usize,
) -> FittedPrimitive {
    let FittedPrimitive::Cone {
        mut apex,
        mut axis_dir,
        mut half_angle_rad,
        height,
    } = initial
    else {
        return initial;
    };
    if indices.len() < 6 || max_iters == 0 {
        return initial;
    }

    axis_dir = unit(axis_dir).unwrap_or([0.0, 0.0, 1.0]);
    let mut lambda = 1e-3;
    let mut cost = cone_cost(pts, indices, apex, axis_dir, half_angle_rad);

    for _ in 0..max_iters {
        let (u, v) = ortho_basis(axis_dir);
        let sin_p = half_angle_rad.sin();
        let cos_p = half_angle_rad.cos();
        let mut jtj = [[0.0f64; 6]; 6];
        let mut jtr = [0.0f64; 6];

        for &idx in indices {
            let d = sub(pts[idx], apex);
            let z = dot(d, axis_dir);
            let du = dot(d, u);
            let dv = dot(d, v);
            let r_perp = du.hypot(dv);
            let s = z * cos_p + r_perp * sin_p;

            let (res, jac) = if s >= 0.0 {
                let res = r_perp * cos_p - z * sin_p;
                if r_perp > 1e-12 {
                    let r_hat = [
                        (du * u[0] + dv * v[0]) / r_perp,
                        (du * u[1] + dv * v[1]) / r_perp,
                        (du * u[2] + dv * v[2]) / r_perp,
                    ];
                    let nc = [
                        cos_p * r_hat[0] - sin_p * axis_dir[0],
                        cos_p * r_hat[1] - sin_p * axis_dir[1],
                        cos_p * r_hat[2] - sin_p * axis_dir[2],
                    ];
                    let eu = du / r_perp;
                    let ev = dv / r_perp;
                    (res, [-nc[0], -nc[1], -nc[2], -s * eu, -s * ev, -s])
                } else {
                    (
                        res,
                        [
                            sin_p * axis_dir[0],
                            sin_p * axis_dir[1],
                            sin_p * axis_dir[2],
                            0.0,
                            0.0,
                            -s,
                        ],
                    )
                }
            } else {
                let dn = norm(d);
                if dn <= 1e-12 {
                    continue;
                }
                let inv = 1.0 / dn;
                (dn, [-d[0] * inv, -d[1] * inv, -d[2] * inv, 0.0, 0.0, 0.0])
            };

            for a in 0..6 {
                jtr[a] += jac[a] * res;
                for b in 0..6 {
                    jtj[a][b] += jac[a] * jac[b];
                }
            }
        }

        let mut mat = jtj;
        for (a, row) in mat.iter_mut().enumerate() {
            row[a] += lambda * row[a].max(1e-6);
        }
        let neg_jtr = [-jtr[0], -jtr[1], -jtr[2], -jtr[3], -jtr[4], -jtr[5]];
        let Some(step) = solve_nxn::<6>(mat, neg_jtr) else {
            lambda *= 10.0;
            continue;
        };
        let step_norm = step.iter().map(|s| s * s).sum::<f64>().sqrt();
        if step_norm < 1e-11 {
            break;
        }
        let cand_apex = [apex[0] + step[0], apex[1] + step[1], apex[2] + step[2]];
        let cand_a_raw = [
            axis_dir[0] + step[3] * u[0] + step[4] * v[0],
            axis_dir[1] + step[3] * u[1] + step[4] * v[1],
            axis_dir[2] + step[3] * u[2] + step[4] * v[2],
        ];
        let Some(cand_a) = unit(cand_a_raw) else {
            lambda *= 10.0;
            continue;
        };
        let cand_phi = half_angle_rad + step[5];
        if !cand_phi.is_finite()
            || !(0.02..=(FRAC_PI_2 - 0.02)).contains(&cand_phi)
            || norm(cand_apex) > 25.0
        {
            lambda *= 10.0;
            continue;
        }
        let cand_cost = cone_cost(pts, indices, cand_apex, cand_a, cand_phi);
        if cand_cost < cost {
            apex = cand_apex;
            axis_dir = cand_a;
            half_angle_rad = cand_phi;
            let rel = (cost - cand_cost) / cost.max(1e-15);
            cost = cand_cost;
            lambda = (lambda * 0.3).max(1e-9);
            if rel < 1e-11 {
                break;
            }
        } else {
            lambda = (lambda * 5.0).min(1e9);
        }
    }

    FittedPrimitive::Cone {
        apex,
        axis_dir,
        half_angle_rad,
        height,
    }
}

fn cone_cost(
    pts: &[[f64; 3]],
    indices: &[usize],
    apex: [f64; 3],
    axis_dir: [f64; 3],
    half_angle_rad: f64,
) -> f64 {
    let sin_p = half_angle_rad.sin();
    let cos_p = half_angle_rad.cos();
    let mut sum = 0.0;
    for &idx in indices {
        let d = sub(pts[idx], apex);
        let z = dot(d, axis_dir);
        let radial = sub(d, [axis_dir[0] * z, axis_dir[1] * z, axis_dir[2] * z]);
        let r_perp = norm(radial);
        let s = z * cos_p + r_perp * sin_p;
        let res = if s >= 0.0 {
            r_perp * cos_p - z * sin_p
        } else {
            norm(d)
        };
        sum += res * res;
    }
    sum
}

fn finalize_cone_span(
    pts: &[[f64; 3]],
    indices: &[usize],
    model: FittedPrimitive,
) -> FittedPrimitive {
    let FittedPrimitive::Cone {
        apex,
        axis_dir,
        half_angle_rad,
        ..
    } = model
    else {
        return model;
    };
    let (mut tmin, mut tmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for &idx in indices {
        let t = dot(sub(pts[idx], apex), axis_dir);
        tmin = tmin.min(t);
        tmax = tmax.max(t);
    }
    let height = if tmin.is_finite() && tmax.is_finite() {
        (tmax - tmin).max(1e-12)
    } else {
        0.0
    };
    FittedPrimitive::Cone {
        apex,
        axis_dir,
        half_angle_rad,
        height,
    }
}

// ============================================================================
// Evaluation, Reporting & Denormalization
// ============================================================================

struct EvalStats {
    msac_loss: f64,
    inlier_ratio: f64,
}

fn signed_residual_point(p: [f64; 3], primitive: &FittedPrimitive) -> (f64, Option<[f64; 3]>) {
    match primitive {
        FittedPrimitive::Plane { origin, normal } => (dot(sub(p, *origin), *normal), Some(*normal)),
        FittedPrimitive::Sphere { center, radius } => {
            let d = sub(p, *center);
            let dn = norm(d);
            let n = if dn > 1e-12 {
                Some([d[0] / dn, d[1] / dn, d[2] / dn])
            } else {
                None
            };
            (dn - *radius, n)
        }
        FittedPrimitive::Cylinder {
            axis_origin,
            axis_dir,
            radius,
            ..
        } => {
            let d = sub(p, *axis_origin);
            let z = dot(d, *axis_dir);
            let radial = sub(d, [axis_dir[0] * z, axis_dir[1] * z, axis_dir[2] * z]);
            let rn = norm(radial);
            let n = if rn > 1e-12 {
                Some([radial[0] / rn, radial[1] / rn, radial[2] / rn])
            } else {
                None
            };
            (rn - *radius, n)
        }
        FittedPrimitive::Cone {
            apex,
            axis_dir,
            half_angle_rad,
            ..
        } => {
            let d = sub(p, *apex);
            let z = dot(d, *axis_dir);
            let radial = sub(d, [axis_dir[0] * z, axis_dir[1] * z, axis_dir[2] * z]);
            let r_perp = norm(radial);
            let sin_p = half_angle_rad.sin();
            let cos_p = half_angle_rad.cos();
            let s = z * cos_p + r_perp * sin_p;
            if s >= 0.0 {
                let res = r_perp * cos_p - z * sin_p;
                let n = if r_perp > 1e-12 {
                    Some([
                        cos_p * (radial[0] / r_perp) - sin_p * axis_dir[0],
                        cos_p * (radial[1] / r_perp) - sin_p * axis_dir[1],
                        cos_p * (radial[2] / r_perp) - sin_p * axis_dir[2],
                    ])
                } else {
                    Some([-axis_dir[0], -axis_dir[1], -axis_dir[2]])
                };
                (res, n)
            } else {
                let dn = norm(d);
                let n = if dn > 1e-12 {
                    Some([d[0] / dn, d[1] / dn, d[2] / dn])
                } else {
                    None
                };
                (dn, n)
            }
        }
    }
}

fn eval_primitive_norm(
    pts: &[[f64; 3]],
    normals: &[[f64; 3]],
    has_oriented_normals: bool,
    primitive: &FittedPrimitive,
    tau: f64,
    normal_weight: f64,
) -> EvalStats {
    let inv_tau2 = 1.0 / (tau * tau).max(1e-18);
    let mut sum_loss = 0.0;
    let mut inliers = 0usize;

    for (i, &p) in pts.iter().enumerate() {
        let (r, surf_n) = signed_residual_point(p, primitive);
        let r_abs = r.abs();
        if r_abs <= tau {
            inliers += 1;
        }
        let mut term = (r * r * inv_tau2).min(1.0);
        if normal_weight > 0.0
            && r_abs <= tau
            && let Some(sn) = surf_n
        {
            let pn = normals[i];
            if norm(pn) > 0.5 {
                let alignment = if has_oriented_normals {
                    dot(pn, sn).clamp(-1.0, 1.0)
                } else {
                    dot(pn, sn).abs().clamp(0.0, 1.0)
                };
                term += normal_weight * (1.0 - alignment.max(0.0));
            }
        }
        sum_loss += term;
    }

    let n = pts.len().max(1) as f64;
    EvalStats {
        msac_loss: sum_loss / n,
        inlier_ratio: (inliers as f64) / n,
    }
}

fn collect_inliers(pts: &[[f64; 3]], primitive: &FittedPrimitive, tau: f64) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, &p) in pts.iter().enumerate() {
        if signed_residual_point(p, primitive).0.abs() <= tau {
            out.push(i);
        }
    }
    out
}

fn denormalize_primitive(prim: &FittedPrimitive, mu: [f64; 3], scale: f64) -> FittedPrimitive {
    let unscale_pt = |q: [f64; 3]| -> [f64; 3] {
        [
            mu[0] + q[0] * scale,
            mu[1] + q[1] * scale,
            mu[2] + q[2] * scale,
        ]
    };
    match prim {
        FittedPrimitive::Plane { origin, normal } => FittedPrimitive::Plane {
            origin: unscale_pt(*origin),
            normal: unit(*normal).unwrap_or([0.0, 0.0, 1.0]),
        },
        FittedPrimitive::Sphere { center, radius } => FittedPrimitive::Sphere {
            center: unscale_pt(*center),
            radius: radius * scale,
        },
        FittedPrimitive::Cylinder {
            axis_origin,
            axis_dir,
            radius,
            height,
        } => FittedPrimitive::Cylinder {
            axis_origin: unscale_pt(*axis_origin),
            axis_dir: unit(*axis_dir).unwrap_or([0.0, 0.0, 1.0]),
            radius: radius * scale,
            height: height * scale,
        },
        FittedPrimitive::Cone {
            apex,
            axis_dir,
            half_angle_rad,
            height,
        } => FittedPrimitive::Cone {
            apex: unscale_pt(*apex),
            axis_dir: unit(*axis_dir).unwrap_or([0.0, 0.0, 1.0]),
            half_angle_rad: *half_angle_rad,
            height: height * scale,
        },
    }
}

fn build_report(
    pts: &[[f64; 3]],
    primitive: FittedPrimitive,
    options: &FitPrimitiveOptions,
    tau_world: f64,
) -> Result<FitPrimitiveReport> {
    let mut residuals = Vec::new();
    reserve(&mut residuals, pts.len())?;

    let mut inlier_count = 0usize;
    let mut sum_sq_inlier = 0.0;
    let mut max_inlier = 0.0f64;
    let mut sum_sq_all = 0.0;
    let mut max_all = 0.0f64;

    for &p in pts {
        let r = signed_residual_point(p, &primitive).0;
        let ar = r.abs();
        sum_sq_all += r * r;
        max_all = max_all.max(ar);
        if ar <= tau_world {
            inlier_count += 1;
            sum_sq_inlier += r * r;
            max_inlier = max_inlier.max(ar);
        }
        residuals.push(r as f32);
    }

    let inlier_ratio = (inlier_count as f64) / (pts.len() as f64);
    if inlier_ratio + 1e-12 < options.min_inlier_ratio {
        return Err(OpsError::InvalidInput(
            "primitive fit inlier ratio fell below min_inlier_ratio",
        ));
    }

    let (rms_error, max_error) = if inlier_count > 0 {
        ((sum_sq_inlier / (inlier_count as f64)).sqrt(), max_inlier)
    } else {
        ((sum_sq_all / (pts.len() as f64)).sqrt(), max_all)
    };

    Ok(FitPrimitiveReport {
        primitive,
        rms_error,
        max_error,
        inlier_count,
        inlier_ratio,
        residuals,
    })
}

// ============================================================================
// Lightweight Flat KD-Tree for Unoriented Local PCA Normals
// ============================================================================

struct KdNode {
    point_idx: usize,
    axis: u8,
    left: usize,
    right: usize,
}

fn estimate_local_pca_normals(pts: &[[f64; 3]], k: usize) -> Vec<[f64; 3]> {
    let n = pts.len();
    if n < 3 || k < 2 {
        return vec![[0.0, 0.0, 0.0]; n];
    }
    let mut order: Vec<usize> = (0..n).collect();
    let mut nodes = Vec::with_capacity(n);
    let root = build_kdtree(pts, &mut order, &mut nodes);
    if root == usize::MAX {
        return vec![[0.0, 0.0, 0.0]; n];
    }

    let mut out = vec![[0.0, 0.0, 0.0]; n];
    let target_k = k.min(n - 1);
    let mut heap: Vec<(f64, usize)> = Vec::with_capacity(target_k + 1);

    for (i, &p) in pts.iter().enumerate() {
        heap.clear();
        query_knn(pts, &nodes, root, p, i, target_k, &mut heap);
        if heap.len() < 2 {
            continue;
        }
        let mut mean = p;
        let cnt = (heap.len() + 1) as f64;
        for &(_, j) in &heap {
            mean = add(mean, pts[j]);
        }
        mean = [mean[0] / cnt, mean[1] / cnt, mean[2] / cnt];
        let mut cov = [[0.0f64; 3]; 3];
        let mut accum = |q: [f64; 3]| {
            let d = sub(q, mean);
            for r in 0..3 {
                for c in 0..3 {
                    cov[r][c] += d[r] * d[c];
                }
            }
        };
        accum(p);
        for &(_, j) in &heap {
            accum(pts[j]);
        }
        let (_vals, vecs) = jacobi_eigen_3x3(cov);
        out[i] = unit([vecs[0][2], vecs[1][2], vecs[2][2]]).unwrap_or([0.0, 0.0, 0.0]);
    }
    out
}

fn build_kdtree(pts: &[[f64; 3]], slice: &mut [usize], nodes: &mut Vec<KdNode>) -> usize {
    if slice.is_empty() {
        return usize::MAX;
    }
    let (mut bmin, mut bmax) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for &idx in slice.iter() {
        let p = pts[idx];
        for c in 0..3 {
            bmin[c] = bmin[c].min(p[c]);
            bmax[c] = bmax[c].max(p[c]);
        }
    }
    let ext = sub(bmax, bmin);
    let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
        0usize
    } else if ext[1] >= ext[2] {
        1usize
    } else {
        2usize
    };
    let mid = slice.len() / 2;
    slice.select_nth_unstable_by(mid, |&a, &b| pts[a][axis].total_cmp(&pts[b][axis]));
    let point_idx = slice[mid];
    let node_id = nodes.len();
    nodes.push(KdNode {
        point_idx,
        axis: axis as u8,
        left: usize::MAX,
        right: usize::MAX,
    });
    let (left_slice, right_with_mid) = slice.split_at_mut(mid);
    let left = build_kdtree(pts, left_slice, nodes);
    let right = build_kdtree(pts, &mut right_with_mid[1..], nodes);
    nodes[node_id].left = left;
    nodes[node_id].right = right;
    node_id
}

fn query_knn(
    pts: &[[f64; 3]],
    nodes: &[KdNode],
    node_id: usize,
    target: [f64; 3],
    skip_idx: usize,
    k: usize,
    best: &mut Vec<(f64, usize)>,
) {
    if node_id == usize::MAX {
        return;
    }
    let node = &nodes[node_id];
    let p = pts[node.point_idx];
    if node.point_idx != skip_idx {
        let d = sub(p, target);
        let d2 = dot(d, d);
        if best.len() < k {
            best.push((d2, node.point_idx));
            if best.len() == k {
                best.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
        } else if d2 < best[k - 1].0 {
            best[k - 1] = (d2, node.point_idx);
            let mut pos = k - 1;
            while pos > 0 && best[pos].0 < best[pos - 1].0 {
                best.swap(pos, pos - 1);
                pos -= 1;
            }
        }
    }

    let axis = usize::from(node.axis);
    let diff = target[axis] - p[axis];
    let (near, far) = if diff <= 0.0 {
        (node.left, node.right)
    } else {
        (node.right, node.left)
    };
    query_knn(pts, nodes, near, target, skip_idx, k, best);
    let worst = if best.len() < k {
        f64::INFINITY
    } else {
        best[k - 1].0
    };
    if diff * diff < worst {
        query_knn(pts, nodes, far, target, skip_idx, k, best);
    }
}

// ============================================================================
// Linear Algebra, Rotation & Deterministic PRNG Helpers
// ============================================================================

fn shortest_arc_matrix(a: [f64; 3], b: [f64; 3]) -> [[f64; 3]; 3] {
    let c = dot(a, b).clamp(-1.0, 1.0);
    if c > 1.0 - 1e-13 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    if c < -1.0 + 1e-12 {
        let u = orthogonal_unit(a);
        return [
            [
                2.0 * u[0] * u[0] - 1.0,
                2.0 * u[0] * u[1],
                2.0 * u[0] * u[2],
            ],
            [
                2.0 * u[1] * u[0],
                2.0 * u[1] * u[1] - 1.0,
                2.0 * u[1] * u[2],
            ],
            [
                2.0 * u[2] * u[0],
                2.0 * u[2] * u[1],
                2.0 * u[2] * u[2] - 1.0,
            ],
        ];
    }
    let v = cross(a, b);
    let k = [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]];
    let inv = 1.0 / (1.0 + c);
    let mut r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for row in 0..3 {
        for col in 0..3 {
            let k2 = k[row][0] * k[0][col] + k[row][1] * k[1][col] + k[row][2] * k[2][col];
            r[row][col] += k[row][col] + inv * k2;
        }
    }
    r
}

fn orthogonal_unit(v: [f64; 3]) -> [f64; 3] {
    let cand = if v[0].abs() <= v[1].abs() && v[0].abs() <= v[2].abs() {
        [1.0, 0.0, 0.0]
    } else if v[1].abs() <= v[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    unit(cross(v, cand)).unwrap_or([1.0, 0.0, 0.0])
}

fn ortho_basis(a: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let u = orthogonal_unit(a);
    let v = unit(cross(a, u)).unwrap_or([0.0, 1.0, 0.0]);
    (u, v)
}

fn canonical_dir(d: [f64; 3]) -> [f64; 3] {
    let ax = d[0].abs();
    let ay = d[1].abs();
    let az = d[2].abs();
    let neg = if ax >= ay && ax >= az {
        d[0] < 0.0
    } else if ay >= az {
        d[1] < 0.0
    } else {
        d[2] < 0.0
    };
    if neg { [-d[0], -d[1], -d[2]] } else { d }
}

fn mat_mul_vec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

fn solve_3x3(m: [[f64; 3]; 3], rhs: [f64; 3]) -> Option<[f64; 3]> {
    solve_nxn::<3>(m, rhs)
}

fn solve_nxn<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let mut piv_row = col;
        let mut piv_val = a[col][col].abs();
        for (r, row) in a.iter().enumerate().skip(col + 1) {
            if row[col].abs() > piv_val {
                piv_val = row[col].abs();
                piv_row = r;
            }
        }
        if piv_val <= 1e-14 || !piv_val.is_finite() {
            return None;
        }
        if piv_row != col {
            a.swap(col, piv_row);
            b.swap(col, piv_row);
        }
        let inv_diag = 1.0 / a[col][col];
        let (head, tail) = a.split_at_mut(col + 1);
        let piv_slice = &head[col][col..];
        let b_col = b[col];
        for (off, row) in tail.iter_mut().enumerate() {
            let factor = row[col] * inv_diag;
            if factor == 0.0 {
                continue;
            }
            for (cell, &piv) in row[col..].iter_mut().zip(piv_slice) {
                *cell -= factor * piv;
            }
            b[col + 1 + off] -= factor * b_col;
        }
    }
    let mut x = [0.0; N];
    for r in (0..N).rev() {
        let mut sum = b[r];
        for c in (r + 1)..N {
            sum -= a[r][c] * x[c];
        }
        let val = sum / a[r][r];
        if !val.is_finite() {
            return None;
        }
        x[r] = val;
    }
    Some(x)
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut z = self.0;
        self.0 = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn gen_usize(&mut self, bound: usize) -> usize {
        ((self.next_u64() as u128 * bound as u128) >> 64) as usize
    }
}

fn sample_distinct<const K: usize>(rng: &mut Rng, n: usize) -> Option<[usize; K]> {
    if n < K {
        return None;
    }
    let mut out = [0usize; K];
    for i in 0..K {
        for _ in 0..32 {
            let idx = rng.gen_usize(n);
            if !out[..i].contains(&idx) {
                out[i] = idx;
                break;
            }
            if i == K - 1 && out[..i].contains(&idx) {
                return None;
            }
        }
    }
    Some(out)
}
