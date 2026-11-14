//! Mesh alignment domain: closed-form landmarks, coarse PCA/OBB pre-alignment,
//! multi-objective fine ICP with Anderson acceleration, and symmetry detection.
//!
//! - [`align_landmarks`] solves the exact closed-form rigid or scaled similarity
//!   transform between N >= 3 3D landmark pairs via Umeyama's SVD method.
//! - [`register_multistage`] combines optional [`CoarseAlignMethod`] pre-alignment
//!   with [`IcpObjective`] fine ICP and Anderson acceleration.
//! - [`detect_symmetry_plane`] finds the dominant reflection symmetry plane of a
//!   mesh and scores its mirror conformity in `[0.0, 1.0]`.
//! - [`icp`] and [`sample_surface`] remain re-exported from [`crate::icp`].

use fast_mesh_bvh::Bvh;
use fast_mesh_query::{
    ClosestHit, ClosestOptions, Rigid, closest_point_many, distance_to_triangle,
};
use fast_mesh_store::{Mesh, TriangleId, par};

use crate::icp::{
    DISTANCE_CHUNK, MAX_STEP_ANGLE, MIN_CONDITIONING, MIN_CORRESPONDENCES, PAR_MIN_DISTANCES, add,
    apply, compose_step, corners_of, cross, dot, jacobi_eigen, norm, orthonormal, rotation_of,
    spread, sub, unit,
};
pub use crate::icp::{IcpMotion, IcpOptions, IcpReport, IcpStatus, icp, sample_surface};
use crate::{OpsError, Result, filled, push, reserve};

const IDENTITY3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const SYMMETRY_SAMPLES: usize = 4096;
const COARSE_SCORE_SAMPLES: usize = 2048;

/// A corresponding 3D landmark pair from a moving source to a fixed target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LandmarkPair {
    /// Point on the moving source geometry.
    pub source: [f64; 3],
    /// Corresponding point on the fixed target geometry.
    pub target: [f64; 3],
}

/// Result of [`align_landmarks`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LandmarkAlignment {
    /// Rigid rotation (axis-angle) and translation `t = mu_q - scale * R * mu_p`.
    pub motion: IcpMotion,
    /// Uniform scale factor (`1.0` when scaling is disabled).
    pub scale: f64,
    /// Root mean square distance between transformed source points and targets.
    pub rmse: f64,
}

impl LandmarkAlignment {
    /// Extracts the rigid part `(rotation, translation)` of the similarity transform.
    #[must_use]
    pub fn rigid(&self) -> Rigid {
        motion_to_rigid(&self.motion)
    }

    /// Extracts the `3x3` proper rotation matrix (`det == +1`).
    #[must_use]
    pub fn rotation_matrix(&self) -> [[f64; 3]; 3] {
        orthonormal(rotation_of(self.motion.rotation))
    }

    /// Returns the column-major `4x4` homogeneous transformation matrix matching
    /// WebGL/three.js conventions (`x' = scale * R * x + t`).
    #[must_use]
    pub fn transform_matrix(&self) -> [f64; 16] {
        rigid_to_matrix4(&self.rigid(), self.scale)
    }

    /// Applies the fitted similarity transform `scale * R * p + t` to `point`.
    #[must_use]
    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        let r = self.rotation_matrix();
        let t = self.motion.translation;
        std::array::from_fn(|i| {
            self.scale * (r[i][0] * point[0] + r[i][1] * point[1] + r[i][2] * point[2]) + t[i]
        })
    }
}

/// Strategy used for global coarse pre-alignment in [`register_multistage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoarseAlignMethod {
    /// Skip global coarse search and start directly from `initial`.
    None,
    /// Align surface area-weighted centroids and test the 4 right-handed
    /// principal-axis sign permutations.
    #[default]
    PcaCentroid,
    /// Align oriented bounding frame centers and test all 24 proper octahedral
    /// axis-and-sign permutations.
    OrientedBoundingBox,
}

/// Objective minimized during the fine ICP stage of [`register_multistage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IcpObjective {
    /// Closed-form SVD Procrustes minimization of squared point-to-point distance.
    PointToPoint,
    /// Linearized point-to-plane squared distance minimization.
    PointToPlane,
    /// Rusinkiewicz (2019) symmetric normal objective:
    /// `sum <(R p_i + t - q_i), (R n_{p,i} + n_{q,i})>^2`.
    #[default]
    SymmetricNormal,
}

/// Options configuring [`register_multistage`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegistrationOptions {
    /// Global coarse alignment strategy run before fine ICP.
    pub coarse_method: CoarseAlignMethod,
    /// Distance/normal objective used by the fine ICP solver.
    pub objective: IcpObjective,
    /// Maximum fine ICP iterations. `0` evaluates the coarse/initial pose only.
    pub max_iterations: usize,
    /// Relative pose-step stopping tolerance (`step / extent`).
    pub tolerance: f64,
    /// Fraction `(0, 1]` of closest correspondences kept each iteration.
    pub overlap: f64,
    /// Optional correspondence cutoff radius; `None` matches all distances.
    pub max_distance: Option<f64>,
    /// Early-stop trimmed RMSE threshold (`<= target_rmse` triggers convergence
    /// when `target_rmse > 0.0`). Default `0.0`.
    pub target_rmse: f64,
    /// Maximum deterministic surface samples drawn from the source mesh.
    pub sample_count: usize,
    /// Anderson acceleration history window `m` (`0` disables acceleration).
    pub anderson_window: usize,
}

impl Default for RegistrationOptions {
    fn default() -> Self {
        RegistrationOptions {
            coarse_method: CoarseAlignMethod::PcaCentroid,
            objective: IcpObjective::SymmetricNormal,
            max_iterations: 50,
            tolerance: 1e-6,
            overlap: 1.0,
            max_distance: None,
            target_rmse: 0.0,
            sample_count: 10_000,
            anderson_window: 3,
        }
    }
}

/// Result of [`register_multistage`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegistrationReport {
    /// Final rigid motion parameter vector (axis-angle + translation).
    pub motion: IcpMotion,
    /// Column-major `4x4` homogeneous transform matrix (WebGL/three.js layout).
    pub transform: [f64; 16],
    /// Final trimmed RMS Euclidean closest-point distance.
    pub final_rmse: f64,
    /// Number of fine ICP iterations taken.
    pub iterations: usize,
    /// Termination status of the fine ICP solver.
    pub status: IcpStatus,
    /// Convenience flag equivalent to `status == IcpStatus::Converged`.
    pub converged: bool,
}

impl RegistrationReport {
    /// Converts `self.motion` into a [`Rigid`] rotation-translation pair.
    #[must_use]
    pub fn rigid(&self) -> Rigid {
        motion_to_rigid(&self.motion)
    }

    /// Applies the aligned rigid transform to `point`.
    #[must_use]
    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        apply(&self.rigid(), point)
    }
}

/// Reflection symmetry plane detection report returned by [`detect_symmetry_plane`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetryPlaneReport {
    /// Unit normal of the detected symmetry plane.
    pub plane_normal: [f64; 3],
    /// A point lying on the symmetry plane (the mesh centroid projected onto the plane).
    pub plane_point: [f64; 3],
    /// RMS closest-point deviation between reflected surface samples and the mesh.
    pub symmetry_error_rms: f64,
    /// Normalized symmetry score in `[0.0, 1.0]` (`1.0` = exact mirror symmetry).
    pub symmetry_score: f64,
}

/// Converts a [`Rigid`] transform into an [`IcpMotion`] (`about = [0, 0, 0]`).
#[must_use]
pub fn rigid_to_motion(rigid: &Rigid) -> IcpMotion {
    IcpMotion {
        translation: rigid.translation,
        rotation: axis_angle_of(rigid.rotation),
        about: [0.0; 3],
        alternative_rmse: 0.0,
    }
}

/// Converts an [`IcpMotion`] (axis-angle rotation about `motion.about`) into a [`Rigid`].
#[must_use]
pub fn motion_to_rigid(motion: &IcpMotion) -> Rigid {
    let r = orthonormal(rotation_of(motion.rotation));
    let translation =
        std::array::from_fn(|i| motion.about[i] + motion.translation[i] - dot(r[i], motion.about));
    Rigid {
        rotation: r,
        translation,
    }
}

/// Converts `(rigid, scale)` into a column-major `4x4` matrix (`[col0, col1, col2, col3]`).
#[must_use]
pub fn rigid_to_matrix4(rigid: &Rigid, scale: f64) -> [f64; 16] {
    let r = &rigid.rotation;
    let t = rigid.translation;
    [
        scale * r[0][0],
        scale * r[1][0],
        scale * r[2][0],
        0.0,
        scale * r[0][1],
        scale * r[1][1],
        scale * r[2][1],
        0.0,
        scale * r[0][2],
        scale * r[1][2],
        scale * r[2][2],
        0.0,
        t[0],
        t[1],
        t[2],
        1.0,
    ]
}

/// Computes the exact axis-angle vector `w = theta * axis` with `theta in [0, pi]`
/// such that `rotation_of(w)` recovers `r`.
pub(crate) fn axis_angle_of(r: [[f64; 3]; 3]) -> [f64; 3] {
    let r = orthonormal(r);
    let v = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
    let sin_theta = 0.5 * norm(v);
    let cos_theta = (0.5 * (r[0][0] + r[1][1] + r[2][2] - 1.0)).clamp(-1.0, 1.0);
    let theta = sin_theta.atan2(cos_theta);
    if theta < 1e-8 {
        let factor = 0.5 * (1.0 + theta * theta / 6.0);
        return v.map(|c| factor * c);
    }
    if theta < std::f64::consts::PI - 1e-4 {
        let scale = theta / (2.0 * sin_theta);
        return v.map(|c| scale * c);
    }
    // Near `theta == pi`, `sin(theta)` vanishes so extract the axis from the
    // symmetric projection `0.5 * (R + R^T) - cos(theta) * I = (1 - cos(theta)) a a^T`.
    let m = [
        [
            r[0][0] - cos_theta,
            0.5 * (r[0][1] + r[1][0]),
            0.5 * (r[0][2] + r[2][0]),
        ],
        [
            0.5 * (r[1][0] + r[0][1]),
            r[1][1] - cos_theta,
            0.5 * (r[1][2] + r[2][1]),
        ],
        [
            0.5 * (r[2][0] + r[0][2]),
            0.5 * (r[2][1] + r[1][2]),
            r[2][2] - cos_theta,
        ],
    ];
    let mut best_col = [1.0, 0.0, 0.0];
    let mut best_len2 = -1.0;
    for &col in &m {
        let l2 = dot(col, col);
        if l2 > best_len2 {
            best_len2 = l2;
            best_col = col;
        }
    }
    let mut axis = unit(best_col).unwrap_or([1.0, 0.0, 0.0]);
    if sin_theta > 1e-13 && dot(axis, v) < 0.0 {
        axis = axis.map(|c| -c);
    }
    axis.map(|c| c * theta)
}

#[inline]
fn mat3_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
    })
}

#[inline]
fn mat3_mul_transpose_left(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| a[0][i] * b[0][j] + a[1][i] * b[1][j] + a[2][i] * b[2][j])
    })
}

#[inline]
fn mat3_mul_transpose_right(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| a[i][0] * b[j][0] + a[i][1] * b[j][1] + a[i][2] * b[j][2])
    })
}

#[inline]
fn mat3_vec(a: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| dot(a[i], v))
}

#[inline]
fn mat3_det(a: [[f64; 3]; 3]) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

#[inline]
fn compose_rigid(a: &Rigid, b: &Rigid) -> Rigid {
    let product = mat3_mul(a.rotation, b.rotation);
    let t = mat3_vec(a.rotation, b.translation);
    Rigid {
        rotation: orthonormal(product),
        translation: add(t, a.translation),
    }
}

/// Symmetric `3x3` eigensolver returning eigenvalues sorted descending and
/// right-handed orthonormal eigenvector columns (`det(V) == +1`) with
/// deterministic dominant-component sign convention.
#[allow(clippy::needless_range_loop)]
pub(crate) fn jacobi_eigen_3x3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = IDENTITY3;
    for _ in 0..48 {
        let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        let diag = a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2];
        if off == 0.0 || (diag > 0.0 && off <= 1e-36 * diag) {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let root = (tau * tau + 1.0).sqrt();
            let t = if tau >= 0.0 {
                1.0 / (tau + root)
            } else {
                -1.0 / (-tau + root)
            };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for row in &mut a {
                let (ap, aq) = (row[p], row[q]);
                row[p] = c * ap - s * aq;
                row[q] = s * ap + c * aq;
            }
            for k in 0..3 {
                let (ap, aq) = (a[p][k], a[q][k]);
                a[p][k] = c * ap - s * aq;
                a[q][k] = s * ap + c * aq;
            }
            for row in &mut v {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    let vals = [a[0][0], a[1][1], a[2][2]];
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| vals[j].total_cmp(&vals[i]).then(i.cmp(&j)));
    let sorted_vals = [vals[order[0]], vals[order[1]], vals[order[2]]];
    let mut cols: [[f64; 3]; 3] =
        std::array::from_fn(|k| [v[0][order[k]], v[1][order[k]], v[2][order[k]]]);
    for col in &mut cols {
        *col = unit(*col).unwrap_or([1.0, 0.0, 0.0]);
        let dom = if col[0].abs() >= col[1].abs() && col[0].abs() >= col[2].abs() {
            0
        } else if col[1].abs() >= col[2].abs() {
            1
        } else {
            2
        };
        if col[dom] < 0.0 {
            *col = col.map(|c| -c);
        }
    }
    // Keep right-handed `V` without altering any eigenspace.
    let c0 = cols[0];
    let proj0 = dot(cols[1], c0);
    let c1 = unit([
        cols[1][0] - proj0 * c0[0],
        cols[1][1] - proj0 * c0[1],
        cols[1][2] - proj0 * c0[2],
    ])
    .unwrap_or_else(|| orthogonal_axis(c0));
    let mut c2 = unit(cross(c0, c1)).unwrap_or_else(|| cross(c0, c1));
    if dot(c2, cols[2]) < 0.0 {
        c2 = cross(c0, c1);
    }
    let sorted_v = std::array::from_fn(|row| [c0[row], c1[row], c2[row]]);
    (sorted_vals, sorted_v)
}

#[inline]
fn orthogonal_axis(u: [f64; 3]) -> [f64; 3] {
    let e = if u[0].abs() <= u[1].abs() && u[0].abs() <= u[2].abs() {
        [1.0, 0.0, 0.0]
    } else if u[1].abs() <= u[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let d = dot(e, u);
    unit([e[0] - d * u[0], e[1] - d * u[1], e[2] - d * u[2]]).unwrap_or([0.0, 1.0, 0.0])
}

/// Pure-Rust `f64` `3x3` singular value decomposition `M = U D V^T`.
///
/// Uses symmetric Jacobi eigendecomposition on scale-normalized `M^T M`,
/// direct left singular vector projection `M v_k` orthogonalized via Modified
/// Gram-Schmidt, and cross-product completion for rank-1/rank-2 inputs.
#[allow(clippy::needless_range_loop)]
pub(crate) fn svd_3x3(m: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3]) {
    let scale = m.iter().flatten().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    if !scale.is_finite() || scale == 0.0 {
        return (IDENTITY3, [0.0; 3], IDENTITY3);
    }
    let mh: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| m[i][j] / scale));
    let ata = mat3_mul_transpose_left(mh, mh);
    let (_evals, v) = jacobi_eigen_3x3(ata);
    let v0 = [v[0][0], v[1][0], v[2][0]];
    let v1 = [v[0][1], v[1][1], v[2][1]];
    let v2 = [v[0][2], v[1][2], v[2][2]];

    let w0 = mat3_vec(mh, v0);
    let n0 = norm(w0);
    if n0 == 0.0 {
        return (IDENTITY3, [0.0; 3], IDENTITY3);
    }
    let u0 = w0.map(|c| c / n0);
    let d0 = scale * n0;

    let w1_raw = mat3_vec(mh, v1);
    let p10 = dot(w1_raw, u0);
    let w1 = [
        w1_raw[0] - p10 * u0[0],
        w1_raw[1] - p10 * u0[1],
        w1_raw[2] - p10 * u0[2],
    ];
    let n1 = norm(w1);
    let (u1, d1) = if n1 > 1e-12 * n0 {
        (w1.map(|c| c / n1), scale * n1)
    } else {
        (orthogonal_axis(u0), 0.0)
    };

    let w2_raw = mat3_vec(mh, v2);
    let p20 = dot(w2_raw, u0);
    let w2_perp0 = [
        w2_raw[0] - p20 * u0[0],
        w2_raw[1] - p20 * u0[1],
        w2_raw[2] - p20 * u0[2],
    ];
    let p21 = dot(w2_perp0, u1);
    let w2 = [
        w2_perp0[0] - p21 * u1[0],
        w2_perp0[1] - p21 * u1[1],
        w2_perp0[2] - p21 * u1[2],
    ];
    let n2 = norm(w2);
    let (u2, d2) = if n2 > 1e-12 * n0 {
        (w2.map(|c| c / n2), scale * n2)
    } else {
        // Rank <= 2: complete an orthonormal frame via `u0 x u1`; `S` corrects orientation.
        (unit(cross(u0, u1)).unwrap_or_else(|| cross(u0, u1)), 0.0)
    };

    let u: [[f64; 3]; 3] = std::array::from_fn(|row| [u0[row], u1[row], u2[row]]);
    (u, [d0, d1, d2], v)
}

/// Core Umeyama alignment on centered cross-covariance without heap allocation.
fn umeyama_core<I>(
    count: usize,
    make_pairs: impl Fn() -> I,
    allow_scaling: bool,
) -> Option<(Rigid, f64)>
where
    I: Iterator<Item = ([f64; 3], [f64; 3])>,
{
    if count == 0 {
        return None;
    }
    let inv_n = 1.0 / count as f64;
    let (mut sum_p, mut sum_q) = ([0.0; 3], [0.0; 3]);
    for (p, q) in make_pairs() {
        sum_p = add(sum_p, p);
        sum_q = add(sum_q, q);
    }
    let mu_p = sum_p.map(|c| c * inv_n);
    let mu_q = sum_q.map(|c| c * inv_n);

    let mut var_p = 0.0;
    let mut sigma = [[0.0; 3]; 3];
    for (p, q) in make_pairs() {
        let dp = sub(p, mu_p);
        let dq = sub(q, mu_q);
        var_p += dot(dp, dp);
        for (row, &dpa) in sigma.iter_mut().zip(&dp) {
            for (cell, &dqb) in row.iter_mut().zip(&dq) {
                *cell += dpa * dqb;
            }
        }
    }
    var_p *= inv_n;
    if !var_p.is_finite() || var_p <= 1e-24 {
        return None;
    }
    for row in &mut sigma {
        for cell in row {
            *cell *= inv_n;
        }
    }

    let (u, d, v) = svd_3x3(sigma);
    let vu_t = mat3_mul_transpose_right(v, u);
    let sign = if mat3_det(vu_t) < 0.0 { -1.0 } else { 1.0 };
    let vs: [[f64; 3]; 3] = std::array::from_fn(|row| [v[row][0], v[row][1], sign * v[row][2]]);
    let rot = orthonormal(mat3_mul_transpose_right(vs, u));
    let c = if allow_scaling {
        ((d[0] + d[1] + sign * d[2]) / var_p).max(0.0)
    } else {
        1.0
    };
    let r_mu_p = mat3_vec(rot, mu_p);
    let t = [
        mu_q[0] - c * r_mu_p[0],
        mu_q[1] - c * r_mu_p[1],
        mu_q[2] - c * r_mu_p[2],
    ];
    Some((
        Rigid {
            rotation: rot,
            translation: t,
        },
        c,
    ))
}

/// Computes the closed-form least-squares rigid (`allow_scaling == false`) or
/// uniform similarity (`allow_scaling == true`) transformation aligning
/// `pairs[i].source` onto `pairs[i].target` via Umeyama's SVD method.
///
/// Guarantees a proper rotation (`det(R) == +1`, never a reflection) and handles
/// planar (`rank-2`) landmark configurations cleanly.
///
/// # Errors
/// [`OpsError::InvalidInput`] if fewer than 3 pairs are provided, any coordinate
/// is non-finite, or source landmark variance is degenerate.
pub fn align_landmarks(pairs: &[LandmarkPair], allow_scaling: bool) -> Result<LandmarkAlignment> {
    if pairs.len() < 3 {
        return Err(OpsError::InvalidInput(
            "landmark alignment requires at least 3 pairs",
        ));
    }
    for pair in pairs {
        if !pair
            .source
            .iter()
            .chain(&pair.target)
            .all(|v| v.is_finite())
        {
            return Err(OpsError::InvalidInput(
                "landmark coordinates must be finite",
            ));
        }
    }
    let Some((rigid, scale)) = umeyama_core(
        pairs.len(),
        || pairs.iter().map(|p| (p.source, p.target)),
        allow_scaling,
    ) else {
        return Err(OpsError::InvalidInput(
            "landmark source points are degenerate (zero variance)",
        ));
    };
    let mut sq_sum = 0.0;
    for pair in pairs {
        let rotated = mat3_vec(rigid.rotation, pair.source);
        let mapped = [
            scale * rotated[0] + rigid.translation[0],
            scale * rotated[1] + rigid.translation[1],
            scale * rotated[2] + rigid.translation[2],
        ];
        let diff = sub(mapped, pair.target);
        sq_sum += dot(diff, diff);
    }
    let rmse = (sq_sum / pairs.len() as f64).sqrt();
    Ok(LandmarkAlignment {
        motion: rigid_to_motion(&rigid),
        scale,
        rmse,
    })
}

/// Symmetric pivoted `6x6` LDLT linear solver for `H x = g`.
///
/// Returns `Some((x, conditioning))` when all 6 Schur pivots pass
/// [`MIN_CONDITIONING`] * `max_diag`, or `None` when the system is rank-deficient
/// or underconstrained so callers can fall back cleanly to spectral projection.
#[allow(clippy::needless_range_loop)]
pub(crate) fn solve_ldlt_6x6(mut a: [[f64; 6]; 6], b: [f64; 6]) -> Option<([f64; 6], f64)> {
    let max_diag = (0..6).fold(0.0_f64, |acc, i| acc.max(a[i][i]));
    if !max_diag.is_finite() || max_diag <= 0.0 {
        return None;
    }
    let tol = MIN_CONDITIONING * max_diag;
    let mut perm = [0usize, 1, 2, 3, 4, 5];
    let mut l = [[0.0_f64; 6]; 6];
    let mut d = [0.0_f64; 6];

    for k in 0..6 {
        let mut piv_idx = k;
        let mut piv_val = a[k][k];
        for i in k + 1..6 {
            if a[i][i] > piv_val {
                piv_val = a[i][i];
                piv_idx = i;
            }
        }
        if !piv_val.is_finite() || piv_val <= tol {
            return None;
        }
        if piv_idx != k {
            perm.swap(k, piv_idx);
            for r in 0..6 {
                a[r].swap(k, piv_idx);
            }
            a.swap(k, piv_idx);
            for c in 0..k {
                let tmp = l[k][c];
                l[k][c] = l[piv_idx][c];
                l[piv_idx][c] = tmp;
            }
        }
        l[k][k] = 1.0;
        d[k] = a[k][k];
        let inv_d = 1.0 / d[k];
        for i in k + 1..6 {
            let factor = a[i][k] * inv_d;
            l[i][k] = factor;
            for j in i..6 {
                a[i][j] -= factor * a[k][j];
                a[j][i] = a[i][j];
            }
        }
    }

    let mut bp = [0.0_f64; 6];
    for i in 0..6 {
        bp[i] = b[perm[i]];
    }
    let mut y = [0.0_f64; 6];
    for i in 0..6 {
        let mut s = bp[i];
        for j in 0..i {
            s -= l[i][j] * y[j];
        }
        y[i] = s;
    }
    let mut xp = [0.0_f64; 6];
    for i in (0..6).rev() {
        let mut s = y[i] / d[i];
        for j in i + 1..6 {
            s -= l[j][i] * xp[j];
        }
        xp[i] = s;
    }
    let mut x = [0.0_f64; 6];
    for i in 0..6 {
        x[perm[i]] = xp[i];
    }
    if !x.iter().all(|v| v.is_finite()) {
        return None;
    }
    let min_d = d.iter().copied().fold(f64::INFINITY, f64::min).max(0.0);
    Some((x, (min_d / max_diag).min(1.0)))
}

/// Solves symmetric `6x6` normal equations `H x = g`, preferring pivoted LDLT and
/// falling back to Jacobi spectral pseudoinverse when rank-deficient.
fn solve_normal_6x6(h: &[[f64; 6]; 6], g: &[f64; 6]) -> ([f64; 6], f64) {
    if let Some((x, _)) = solve_ldlt_6x6(*h, *g) {
        // Compute true spectral conditioning only when needed or report LDLT bound;
        // Jacobi on 6x6 costs ~1 microsecond and matches `icp.rs` conditioning cleanly.
        let (values, _) = jacobi_eigen(*h);
        let largest = values.iter().copied().fold(0.0, f64::max);
        let smallest = values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .max(0.0);
        let cond = if largest > 0.0 {
            smallest / largest
        } else {
            0.0
        };
        if cond >= MIN_CONDITIONING {
            return (x, cond);
        }
    }
    let (values, vectors) = jacobi_eigen(*h);
    let largest = values.iter().copied().fold(0.0, f64::max);
    let smallest = values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .max(0.0);
    let mut x = [0.0; 6];
    if largest <= 0.0 {
        return (x, 0.0);
    }
    for (k, &value) in values.iter().enumerate() {
        if value < MIN_CONDITIONING * largest {
            continue;
        }
        let along = (0..6).map(|i| vectors[i][k] * g[i]).sum::<f64>() / value;
        for (xi, row) in x.iter_mut().zip(&vectors) {
            *xi += row[k] * along;
        }
    }
    (x, smallest / largest)
}

/// Exact `f64` closest point and barycentric coordinates on triangle `[a, b, c]`
/// (Ericson, Real-Time Collision Detection, 5.1.5).
pub(crate) fn closest_on_triangle_f64(p: [f64; 3], tri: [[f64; 3]; 3]) -> ([f64; 3], [f64; 3]) {
    let [a, b, c] = tri;
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        let v = if denom > 0.0 { d1 / denom } else { 0.0 };
        return (
            [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]],
            [1.0 - v, v, 0.0],
        );
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        let w = if denom > 0.0 { d2 / denom } else { 0.0 };
        return (
            [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]],
            [1.0 - w, 0.0, w],
        );
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        let w = if denom > 0.0 { (d4 - d3) / denom } else { 0.0 };
        let bc = sub(c, b);
        return (
            [b[0] + w * bc[0], b[1] + w * bc[1], b[2] + w * bc[2]],
            [0.0, 1.0 - w, w],
        );
    }
    let denom_inv = {
        let s = va + vb + vc;
        if s > 0.0 { 1.0 / s } else { 0.0 }
    };
    let v = vb * denom_inv;
    let w = vc * denom_inv;
    let u = 1.0 - v - w;
    (
        [
            a[0] + ab[0] * v + ac[0] * w,
            a[1] + ab[1] * v + ac[1] * w,
            a[2] + ab[2] * v + ac[2] * w,
        ],
        [u, v, w],
    )
}

struct SampledSurface {
    points: Vec<[f64; 3]>,
    normals: Vec<[f64; 3]>,
}

fn check_mesh_finite_and_nonempty(mesh: &Mesh, label: &'static str) -> Result<()> {
    if mesh.triangle_count() == 0 {
        return Err(OpsError::InvalidInput(label));
    }
    for t in 0..mesh.triangle_slots() {
        let Some([a, b, c]) = mesh.triangle(t as TriangleId) else {
            continue;
        };
        for v in [a, b, c] {
            let p = mesh
                .position(v)
                .ok_or(OpsError::InvalidInput("triangle references invalid vertex"))?;
            if !p.iter().all(|c| c.is_finite()) {
                return Err(OpsError::InvalidInput(
                    "mesh vertex positions must be finite",
                ));
            }
        }
    }
    Ok(())
}

/// Computes area-weighted unit vertex normals (`[0.0; 3]` for zero-area isolated vertices).
fn compute_vertex_normals(mesh: &Mesh) -> Result<Vec<[f64; 3]>> {
    let mut accum = filled(mesh.vertex_slots(), [0.0_f64; 3])?;
    for t in 0..mesh.triangle_slots() {
        let Some([a, b, c]) = mesh.triangle(t as TriangleId) else {
            continue;
        };
        let (Some(pa), Some(pb), Some(pc)) = (mesh.position(a), mesh.position(b), mesh.position(c))
        else {
            continue;
        };
        let pa = pa.map(f64::from);
        let pb = pb.map(f64::from);
        let pc = pc.map(f64::from);
        let raw = cross(sub(pb, pa), sub(pc, pa));
        for v in [a, b, c] {
            let cur = &mut accum[v as usize];
            *cur = add(*cur, raw);
        }
    }
    for n in &mut accum {
        *n = unit(*n).unwrap_or([0.0; 3]);
    }
    Ok(accum)
}

/// Deterministically samples up to `max_samples` points matching `sample_surface`,
/// paired with unit surface normals.
fn sample_surface_with_normals(mesh: &Mesh, max_samples: usize) -> Result<SampledSurface> {
    let mut used = filled(mesh.vertex_slots(), false)?;
    let mut vnorm = filled(mesh.vertex_slots(), [0.0_f64; 3])?;
    for t in 0..mesh.triangle_slots() {
        let Some([a, b, c]) = mesh.triangle(t as TriangleId) else {
            continue;
        };
        used[a as usize] = true;
        used[b as usize] = true;
        used[c as usize] = true;
        if let (Some(pa), Some(pb), Some(pc)) =
            (mesh.position(a), mesh.position(b), mesh.position(c))
        {
            let raw = cross(
                sub(pb.map(f64::from), pa.map(f64::from)),
                sub(pc.map(f64::from), pa.map(f64::from)),
            );
            for v in [a, b, c] {
                vnorm[v as usize] = add(vnorm[v as usize], raw);
            }
        }
    }
    let total = mesh.triangle_count() + used.iter().filter(|&&u| u).count();
    let take = total.min(max_samples);
    let mut points = Vec::new();
    let mut normals = Vec::new();
    reserve(&mut points, take)?;
    reserve(&mut normals, take)?;
    if take == 0 {
        return Ok(SampledSurface { points, normals });
    }
    let mut seen = 0;
    let mut offer =
        |points: &mut Vec<[f64; 3]>, normals: &mut Vec<[f64; 3]>, p: [f64; 3], n: [f64; 3]| {
            let next = (points.len() as u128 * total as u128).div_ceil(take as u128);
            if points.len() < take && seen as u128 == next {
                points.push(p);
                normals.push(n);
            }
            seen += 1;
        };
    for t in 0..mesh.triangle_slots() {
        let Some([a, b, c]) = mesh.triangle(t as TriangleId) else {
            continue;
        };
        if let (Some(pa), Some(pb), Some(pc)) =
            (mesh.position(a), mesh.position(b), mesh.position(c))
        {
            let pa = pa.map(f64::from);
            let pb = pb.map(f64::from);
            let pc = pc.map(f64::from);
            let centroid = std::array::from_fn(|i| (pa[i] + pb[i] + pc[i]) / 3.0);
            let n = unit(cross(sub(pb, pa), sub(pc, pa))).unwrap_or([0.0, 0.0, 1.0]);
            offer(&mut points, &mut normals, centroid, n);
        }
    }
    for (v, &is_used) in used.iter().enumerate() {
        if !is_used {
            continue;
        }
        if let Some(p) = mesh.position(v as u32) {
            let n = unit(vnorm[v]).unwrap_or([0.0, 0.0, 1.0]);
            offer(&mut points, &mut normals, p.map(f64::from), n);
        }
    }
    Ok(SampledSurface { points, normals })
}

struct SurfaceFrame {
    centroid: [f64; 3],
    axes: [[f64; 3]; 3],
    evals: [f64; 3],
    obb_center: [f64; 3],
    bbox_diag: f64,
}

/// Computes surface area-weighted centroid, covariance principal frame, OBB center,
/// and live vertex bounding-box diagonal.
fn compute_surface_frame(mesh: &Mesh) -> SurfaceFrame {
    let mut area_sum = 0.0_f64;
    let mut weighted_pos = [0.0_f64; 3];
    let mut min_xyz = [f64::INFINITY; 3];
    let mut max_xyz = [f64::NEG_INFINITY; 3];
    let mut vert_sum = [0.0_f64; 3];
    let mut vert_count = 0usize;

    for t in 0..mesh.triangle_slots() {
        let Some([a, b, c]) = mesh.triangle(t as TriangleId) else {
            continue;
        };
        let (Some(pa), Some(pb), Some(pc)) = (mesh.position(a), mesh.position(b), mesh.position(c))
        else {
            continue;
        };
        let pa = pa.map(f64::from);
        let pb = pb.map(f64::from);
        let pc = pc.map(f64::from);
        for v in [pa, pb, pc] {
            for k in 0..3 {
                min_xyz[k] = min_xyz[k].min(v[k]);
                max_xyz[k] = max_xyz[k].max(v[k]);
            }
            vert_sum = add(vert_sum, v);
            vert_count += 1;
        }
        let area = 0.5 * norm(cross(sub(pb, pa), sub(pc, pa)));
        if area > 0.0 {
            let mid = std::array::from_fn(|k| (pa[k] + pb[k] + pc[k]) / 3.0);
            area_sum += area;
            weighted_pos = add(weighted_pos, mid.map(|x| area * x));
        }
    }

    let bbox_diag = if min_xyz[0].is_finite() {
        norm(sub(max_xyz, min_xyz))
    } else {
        0.0
    };
    let fallback_mean = if vert_count > 0 {
        vert_sum.map(|x| x / vert_count as f64)
    } else {
        [0.0; 3]
    };
    let centroid = if area_sum > 0.0 {
        weighted_pos.map(|x| x / area_sum)
    } else {
        fallback_mean
    };

    let mut cov = [[0.0_f64; 3]; 3];
    if area_sum > 0.0 {
        for t in 0..mesh.triangle_slots() {
            let Some([a, b, c]) = mesh.triangle(t as TriangleId) else {
                continue;
            };
            let (Some(pa), Some(pb), Some(pc)) =
                (mesh.position(a), mesh.position(b), mesh.position(c))
            else {
                continue;
            };
            let pa = sub(pa.map(f64::from), centroid);
            let pb = sub(pb.map(f64::from), centroid);
            let pc = sub(pc.map(f64::from), centroid);
            let area = 0.5 * norm(cross(sub(pb, pa), sub(pc, pa)));
            if area <= 0.0 {
                continue;
            }
            let pm: [f64; 3] = std::array::from_fn(|k| (pa[k] + pb[k] + pc[k]) / 3.0);
            let w = area / 12.0;
            for i in 0..3 {
                for j in 0..3 {
                    cov[i][j] +=
                        w * (9.0 * pm[i] * pm[j] + pa[i] * pa[j] + pb[i] * pb[j] + pc[i] * pc[j]);
                }
            }
        }
        for row in &mut cov {
            for cell in row {
                *cell /= area_sum;
            }
        }
    } else if vert_count > 0 {
        for t in 0..mesh.triangle_slots() {
            let Some([a, b, c]) = mesh.triangle(t as TriangleId) else {
                continue;
            };
            for vid in [a, b, c] {
                if let Some(p) = mesh.position(vid) {
                    let dp = sub(p.map(f64::from), centroid);
                    for i in 0..3 {
                        for j in 0..3 {
                            cov[i][j] += dp[i] * dp[j] / vert_count as f64;
                        }
                    }
                }
            }
        }
    }

    let (evals, axes) = jacobi_eigen_3x3(cov);

    // Project live vertices into the principal axis basis to locate exact OBB center.
    let mut obb_min = [f64::INFINITY; 3];
    let mut obb_max = [f64::NEG_INFINITY; 3];
    for t in 0..mesh.triangle_slots() {
        let Some([a, b, c]) = mesh.triangle(t as TriangleId) else {
            continue;
        };
        for vid in [a, b, c] {
            if let Some(p) = mesh.position(vid) {
                let p = p.map(f64::from);
                for k in 0..3 {
                    let axis = [axes[0][k], axes[1][k], axes[2][k]];
                    let coord = dot(p, axis);
                    obb_min[k] = obb_min[k].min(coord);
                    obb_max[k] = obb_max[k].max(coord);
                }
            }
        }
    }
    let obb_center = if obb_min[0].is_finite() {
        let local_mid = std::array::from_fn(|k| 0.5 * (obb_min[k] + obb_max[k]));
        mat3_vec(axes, local_mid)
    } else {
        centroid
    };

    SurfaceFrame {
        centroid,
        axes,
        evals,
        obb_center,
        bbox_diag,
    }
}

/// Stratified deterministic subsample up to `max_count` from `src`.
fn subsample_points(src: &[[f64; 3]], max_count: usize) -> Result<Vec<[f64; 3]>> {
    let take = src.len().min(max_count);
    let mut out = Vec::new();
    reserve(&mut out, take)?;
    if take == 0 {
        return Ok(out);
    }
    if take == src.len() {
        out.extend_from_slice(src);
        return Ok(out);
    }
    for i in 0..take {
        let idx = (i * src.len()) / take;
        out.push(src[idx]);
    }
    Ok(out)
}

#[derive(Clone, Copy)]
struct Kept {
    point: u32,
    distance: f64,
}

/// Selects the closest `keep` items deterministically, independent of sort stability.
fn trim_kept(kept: &mut Vec<Kept>, keep: usize) {
    if kept.len() > keep {
        kept.select_nth_unstable_by(keep, |a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then(a.point.cmp(&b.point))
        });
        kept.truncate(keep);
    }
    kept.sort_unstable_by_key(|a| a.point);
}

fn score_candidates(
    points: &[[f64; 3]],
    target: &Mesh,
    target_bvh: &Bvh,
    overlap: f64,
    max_distance: Option<f64>,
    candidates: &[Rigid],
) -> Result<Rigid> {
    let n = points.len();
    let keep = (overlap * n as f64).ceil().clamp(1.0, n as f64) as usize;
    let query = ClosestOptions {
        max_distance: max_distance.map_or(f32::INFINITY, |d| d as f32),
    };
    let mut moved = filled(n, [0.0; 3])?;
    let mut rounded = filled(n, [0.0; 3])?;
    let mut hits = filled(n, None)?;
    let mut distances = filled(n, 0.0)?;
    let mut kept = Vec::new();
    reserve(&mut kept, n)?;

    let evaluate_set = |query_opts: ClosestOptions,
                        moved: &mut Vec<[f64; 3]>,
                        rounded: &mut Vec<[f32; 3]>,
                        hits: &mut Vec<Option<ClosestHit>>,
                        distances: &mut Vec<f64>,
                        kept: &mut Vec<Kept>|
     -> Result<Option<Rigid>> {
        let mut best: Option<(Rigid, usize, f64)> = None;
        for &cand in candidates {
            let mut in_f32_range = true;
            for ((p, m), r) in points.iter().zip(moved.iter_mut()).zip(rounded.iter_mut()) {
                *m = apply(&cand, *p);
                *r = m.map(|c| c as f32);
                if !r.iter().all(|c| c.is_finite()) {
                    in_f32_range = false;
                    break;
                }
            }
            if !in_f32_range {
                continue;
            }
            closest_point_many(target, target_bvh, rounded, &query_opts, hits)?;
            par::for_each_chunk_mut(distances, DISTANCE_CHUNK, PAR_MIN_DISTANCES, |i, slots| {
                let start = i * DISTANCE_CHUNK;
                for (k, slot) in slots.iter_mut().enumerate() {
                    let Some(hit) = hits[start + k] else {
                        continue;
                    };
                    *slot = corners_of(target, hit).map_or(f64::from(hit.distance), |tri| {
                        distance_to_triangle(moved[start + k], tri)
                    });
                }
            });
            kept.clear();
            for (i, hit) in hits.iter().enumerate() {
                if hit.is_some() {
                    kept.push(Kept {
                        point: i as u32,
                        distance: distances[i],
                    });
                }
            }
            if kept.is_empty() {
                continue;
            }
            trim_kept(kept, keep);
            let sq: f64 = kept.iter().map(|k| k.distance * k.distance).sum();
            let rmse = (sq / kept.len() as f64).sqrt();
            let inliers = kept.len();
            let better = match best {
                None => true,
                Some((_, best_inliers, best_rmse)) => {
                    inliers > best_inliers || (inliers == best_inliers && rmse < best_rmse - 1e-12)
                }
            };
            if better {
                best = Some((cand, inliers, rmse));
            }
        }
        Ok(best.map(|(r, _, _)| r))
    };

    if let Some(best) = evaluate_set(
        query,
        &mut moved,
        &mut rounded,
        &mut hits,
        &mut distances,
        &mut kept,
    )? {
        return Ok(best);
    }
    if max_distance.is_some() {
        let unbounded = ClosestOptions {
            max_distance: f32::INFINITY,
        };
        if let Some(best) = evaluate_set(
            unbounded,
            &mut moved,
            &mut rounded,
            &mut hits,
            &mut distances,
            &mut kept,
        )? {
            return Ok(best);
        }
    }
    Ok(candidates[0])
}

/// Runs global PCA or OBB coarse alignment against `target` and returns the best
/// initial [`Rigid`] pose. Always includes `initial` as candidate 0.
fn run_coarse_stage(
    source: &Mesh,
    target: &Mesh,
    target_bvh: &Bvh,
    sample_points: &[[f64; 3]],
    initial: Rigid,
    options: &RegistrationOptions,
) -> Result<Rigid> {
    if options.coarse_method == CoarseAlignMethod::None {
        return Ok(initial);
    }
    let src_frame = compute_surface_frame(source);
    let tgt_frame = compute_surface_frame(target);

    type CoarsePerm = (usize, usize, usize, f64);
    let (center_src, center_tgt, perms): ([f64; 3], [f64; 3], &[CoarsePerm]) =
        match options.coarse_method {
            CoarseAlignMethod::None => return Ok(initial),
            CoarseAlignMethod::PcaCentroid => {
                // 4 proper sign permutations keeping the principal axis order.
                const PCA_PERMS: [(usize, usize, usize, f64); 1] = [(0, 1, 2, 1.0)];
                (src_frame.centroid, tgt_frame.centroid, &PCA_PERMS)
            }
            CoarseAlignMethod::OrientedBoundingBox => {
                // 6 axis permutations x 4 sign combinations = all 24 proper octahedral rotations.
                const ALL_PERMS: [(usize, usize, usize, f64); 6] = [
                    (0, 1, 2, 1.0),
                    (0, 2, 1, -1.0),
                    (1, 0, 2, -1.0),
                    (1, 2, 0, 1.0),
                    (2, 0, 1, 1.0),
                    (2, 1, 0, -1.0),
                ];
                (src_frame.obb_center, tgt_frame.obb_center, &ALL_PERMS)
            }
        };

    let signs = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)];
    let mut candidates = Vec::new();
    reserve(&mut candidates, 1 + perms.len() * signs.len() * 2)?;
    candidates.push(initial);

    let push_rot_group = |cands: &mut Vec<Rigid>, cs: [f64; 3], ct: [f64; 3]| {
        for &(p0, p1, p2, parity) in perms {
            for &(s0, s1) in &signs {
                let s2 = parity * s0 * s1;
                let mut perm_mat = [[0.0; 3]; 3];
                perm_mat[p0][0] = s0;
                perm_mat[p1][1] = s1;
                perm_mat[p2][2] = s2;
                let r = orthonormal(mat3_mul_transpose_right(
                    mat3_mul(tgt_frame.axes, perm_mat),
                    src_frame.axes,
                ));
                let t = sub(ct, mat3_vec(r, cs));
                cands.push(Rigid {
                    rotation: r,
                    translation: t,
                });
            }
        }
    };

    push_rot_group(&mut candidates, center_src, center_tgt);
    if options.coarse_method == CoarseAlignMethod::OrientedBoundingBox
        && norm(sub(src_frame.obb_center, src_frame.centroid)) > 1e-6 * src_frame.bbox_diag.max(1.0)
    {
        // Also test centroid anchors when OBB center diverges from surface centroid.
        push_rot_group(&mut candidates, src_frame.centroid, tgt_frame.centroid);
    }

    let coarse_pts = subsample_points(sample_points, COARSE_SCORE_SAMPLES)?;
    score_candidates(
        &coarse_pts,
        target,
        target_bvh,
        options.overlap,
        options.max_distance,
        &candidates,
    )
}

struct MultiEvaluator<'a> {
    points: &'a [[f64; 3]],
    src_normals: &'a [[f64; 3]],
    target: &'a Mesh,
    target_bvh: &'a Bvh,
    target_vnorms: &'a [[f64; 3]],
    query: ClosestOptions,
    keep: usize,
    objective: IcpObjective,
    moved: Vec<[f64; 3]>,
    moved_normals: Vec<[f64; 3]>,
    rounded: Vec<[f32; 3]>,
    hits: Vec<Option<ClosestHit>>,
    distances: Vec<f64>,
    closest_pts: Vec<[f64; 3]>,
    target_normals: Vec<[f64; 3]>,
    kept: Vec<Kept>,
}

impl<'a> MultiEvaluator<'a> {
    fn new(
        source: &'a SampledSurface,
        target: &'a Mesh,
        target_bvh: &'a Bvh,
        target_vnorms: &'a [[f64; 3]],
        options: &RegistrationOptions,
    ) -> Result<Self> {
        let n = source.points.len();
        let mut kept = Vec::new();
        reserve(&mut kept, n)?;
        Ok(MultiEvaluator {
            points: &source.points,
            src_normals: &source.normals,
            target,
            target_bvh,
            target_vnorms,
            query: ClosestOptions {
                max_distance: options.max_distance.map_or(f32::INFINITY, |d| d as f32),
            },
            keep: (options.overlap * n as f64).ceil().clamp(1.0, n as f64) as usize,
            objective: options.objective,
            moved: filled(n, [0.0; 3])?,
            moved_normals: filled(n, [0.0; 3])?,
            rounded: filled(n, [0.0; 3])?,
            hits: filled(n, None)?,
            distances: filled(n, 0.0)?,
            closest_pts: filled(n, [0.0; 3])?,
            target_normals: filled(n, [0.0; 3])?,
            kept,
        })
    }

    fn evaluate(&mut self, transform: &Rigid) -> Result<Option<f64>> {
        for (((p, np), (m, nm)), r) in self
            .points
            .iter()
            .zip(self.src_normals)
            .zip(self.moved.iter_mut().zip(&mut self.moved_normals))
            .zip(&mut self.rounded)
        {
            *m = apply(transform, *p);
            *r = m.map(|c| c as f32);
            if !r.iter().all(|c| c.is_finite()) {
                return Ok(None);
            }
            *nm = mat3_vec(transform.rotation, *np);
        }
        closest_point_many(
            self.target,
            self.target_bvh,
            &self.rounded,
            &self.query,
            &mut self.hits,
        )?;
        let (target, target_vnorms, moved, hits) =
            (self.target, self.target_vnorms, &self.moved, &self.hits);
        let obj = self.objective;
        // Pack `(distance, closest_pt, target_normal)` per chunk in parallel.
        par::for_each_chunk_mut(
            &mut self.distances,
            DISTANCE_CHUNK,
            PAR_MIN_DISTANCES,
            |i, slots| {
                let start = i * DISTANCE_CHUNK;
                for (k, slot) in slots.iter_mut().enumerate() {
                    let idx = start + k;
                    let Some(hit) = hits[idx] else {
                        continue;
                    };
                    *slot = corners_of(target, hit).map_or(f64::from(hit.distance), |tri| {
                        distance_to_triangle(moved[idx], tri)
                    });
                }
            },
        );
        // Closest point / normal lookup runs over kept indices after trimming when unbounded,
        // or directly per hit; computing exact triangle geometry here preserves parallel determinism.
        par::for_each_chunk_mut(
            &mut self.closest_pts,
            DISTANCE_CHUNK,
            PAR_MIN_DISTANCES,
            |i, slots| {
                let start = i * DISTANCE_CHUNK;
                for (k, slot) in slots.iter_mut().enumerate() {
                    let idx = start + k;
                    let Some(hit) = hits[idx] else {
                        continue;
                    };
                    if let Some(tri) = corners_of(target, hit) {
                        let (q, _) = closest_on_triangle_f64(moved[idx], tri);
                        *slot = q;
                    } else {
                        *slot = hit.point.map(f64::from);
                    }
                }
            },
        );
        if obj != IcpObjective::PointToPoint {
            par::for_each_chunk_mut(
                &mut self.target_normals,
                DISTANCE_CHUNK,
                PAR_MIN_DISTANCES,
                |i, slots| {
                    let start = i * DISTANCE_CHUNK;
                    for (k, slot) in slots.iter_mut().enumerate() {
                        let idx = start + k;
                        let Some(hit) = hits[idx] else {
                            continue;
                        };
                        *slot = target_hit_normal(target, target_vnorms, hit, obj);
                    }
                },
            );
        }
        self.kept.clear();
        for (i, hit) in self.hits.iter().enumerate() {
            if hit.is_some() {
                self.kept.push(Kept {
                    point: i as u32,
                    distance: self.distances[i],
                });
            }
        }
        if self.kept.is_empty() {
            return Ok(Some(f64::INFINITY));
        }
        trim_kept(&mut self.kept, self.keep);
        let sq: f64 = self.kept.iter().map(|k| k.distance * k.distance).sum();
        Ok(Some((sq / self.kept.len() as f64).sqrt()))
    }

    fn compute_step(
        &self,
        current: &Rigid,
        center: [f64; 3],
        extent: f64,
    ) -> Option<(Rigid, f64, f64)> {
        if self.kept.len() < MIN_CORRESPONDENCES {
            return None;
        }
        match self.objective {
            IcpObjective::PointToPoint => self.step_point_to_point(current, center, extent),
            IcpObjective::PointToPlane | IcpObjective::SymmetricNormal => {
                self.step_normal_equations(current, center, extent)
            }
        }
    }

    fn step_point_to_point(
        &self,
        current: &Rigid,
        center: [f64; 3],
        extent: f64,
    ) -> Option<(Rigid, f64, f64)> {
        let count = self.kept.len();
        let (delta, _) = umeyama_core(
            count,
            || {
                self.kept.iter().map(|k| {
                    (
                        self.moved[k.point as usize],
                        self.closest_pts[k.point as usize],
                    )
                })
            },
            false,
        )?;
        let next = compose_rigid(&delta, current);
        let c_next = apply(&delta, center);
        let t_rel = sub(c_next, center);
        let w = axis_angle_of(delta.rotation);
        let step_frac = (norm(t_rel) + norm(w) * extent) / extent;

        // Check kept point covariance rank so degenerate planar/collinear point clouds
        // report conditioning accurately without falsely flagging 3D point clouds.
        let mut cov = [[0.0_f64; 3]; 3];
        let mut mean = [0.0_f64; 3];
        for k in &self.kept {
            mean = add(mean, self.closest_pts[k.point as usize]);
        }
        mean = mean.map(|v| v / count as f64);
        for k in &self.kept {
            let dp = sub(self.closest_pts[k.point as usize], mean);
            for i in 0..3 {
                for j in 0..3 {
                    cov[i][j] += dp[i] * dp[j] / count as f64;
                }
            }
        }
        let (evals, _) = jacobi_eigen_3x3(cov);
        let cond = if evals[0] > 0.0 {
            (evals[1] / evals[0]).clamp(0.0, 1.0)
        } else {
            0.0
        };
        Some((next, step_frac, cond))
    }

    #[allow(clippy::needless_range_loop)]
    fn step_normal_equations(
        &self,
        current: &Rigid,
        center: [f64; 3],
        extent: f64,
    ) -> Option<(Rigid, f64, f64)> {
        let mut h = [[0.0_f64; 6]; 6];
        let mut g = [0.0_f64; 6];
        let mut rows = 0usize;

        for k in &self.kept {
            let idx = k.point as usize;
            let p = self.moved[idx];
            let q = self.closest_pts[idx];
            let nq = self.target_normals[idx];
            if dot(nq, nq) <= 0.0 {
                continue;
            }
            let (jt, jr, rhs) = match self.objective {
                IcpObjective::PointToPlane => {
                    let residual = dot(sub(q, p), nq);
                    let lever = cross(sub(p, center), nq);
                    (nq, lever, residual)
                }
                IcpObjective::SymmetricNormal => {
                    let np = self.moved_normals[idx];
                    // First-order Gauss-Newton linearization of `sum <(R p + t - q), (R n_p + n_q)>^2`
                    // around `R(w) ~ I + [w]x` about `center`:
                    // `(w x p_c) . (n_p + n_q) + (p_c - q_c) . (w x n_p) = w . (p_c x n_q + q_c x n_p)`.
                    let n_sum = add(np, nq);
                    let n_len2 = dot(n_sum, n_sum);
                    if n_len2 <= 1e-12 || dot(np, nq) < -0.5 {
                        continue;
                    }
                    let n_half = n_sum.map(|c| 0.5 * c);
                    let pc = sub(p, center);
                    let qc = sub(q, center);
                    let residual = dot(sub(q, p), n_half);
                    let lever = add(cross(pc, nq), cross(qc, np)).map(|c| 0.5 * c);
                    (n_half, lever, residual)
                }
                IcpObjective::PointToPoint => unreachable!(),
            };
            rows += 1;
            let row = [
                jt[0],
                jt[1],
                jt[2],
                jr[0] / extent,
                jr[1] / extent,
                jr[2] / extent,
            ];
            for i in 0..6 {
                g[i] += row[i] * rhs;
                for j in i..6 {
                    h[i][j] += row[i] * row[j];
                }
            }
        }
        if rows < MIN_CORRESPONDENCES {
            return None;
        }
        for i in 0..6 {
            for j in 0..i {
                h[i][j] = h[j][i];
            }
        }
        let (mut step, cond) = solve_normal_6x6(&h, &g);
        let max_diag = (0..6).fold(0.0_f64, |acc, i| acc.max(h[i][i]));
        let cur_sq: f64 = self.kept.iter().map(|k| k.distance * k.distance).sum();
        let cur_rmse = (cur_sq / self.kept.len().max(1) as f64).sqrt();
        // Smooth Levenberg-Marquardt damping on well-ranked systems far from alignment:
        // suppresses tangential/cylindrical slide (`cond ~ 1e-5`) while preserving `cond`
        // for Underconstrained detection and vanishing quadratically as `cur_rmse -> 0`.
        let norm_err = (cur_rmse / extent.max(1e-12)).clamp(0.0, 1.0);
        if cond >= MIN_CONDITIONING && max_diag > 0.0 && norm_err > 1e-4 {
            let lm = (0.05 * norm_err * norm_err).clamp(0.0, 1e-2) * max_diag;
            let mut h_lm = h;
            for i in 0..6 {
                h_lm[i][i] += lm;
            }
            let (lm_step, _) = solve_normal_6x6(&h_lm, &g);
            step = lm_step;
        }
        let raw_t = norm([step[0], step[1], step[2]]);
        let raw_w = norm([step[3] / extent, step[4] / extent, step[5] / extent]);
        let rotation = [step[3] / extent, step[4] / extent, step[5] / extent];
        let angle = raw_w;
        let t_scale = if raw_t > 0.5 * extent {
            0.5 * extent / raw_t
        } else {
            1.0
        };
        let r_scale = if angle > MAX_STEP_ANGLE {
            MAX_STEP_ANGLE / angle
        } else {
            1.0
        };
        let shrink = t_scale.min(r_scale);
        let translation = [step[0], step[1], step[2]].map(|v| v * shrink);
        let rotation = rotation.map(|v| v * shrink);
        let next = compose_step(current, center, translation, rotation);
        let step_frac = (norm(translation) + norm(rotation) * extent) / extent;
        Some((next, step_frac, cond))
    }
}

/// Evaluates target surface normal at `hit`, using smooth crease-preserving
/// barycentric vertex normals for [`IcpObjective::SymmetricNormal`].
fn target_hit_normal(
    target: &Mesh,
    target_vnorms: &[[f64; 3]],
    hit: ClosestHit,
    obj: IcpObjective,
) -> [f64; 3] {
    let Some([a, b, c]) = target.triangle(hit.triangle) else {
        return [0.0; 3];
    };
    let (Some(pa), Some(pb), Some(pc)) =
        (target.position(a), target.position(b), target.position(c))
    else {
        return [0.0; 3];
    };
    let pa = pa.map(f64::from);
    let pb = pb.map(f64::from);
    let pc = pc.map(f64::from);
    let Some(n_face) = unit(cross(sub(pb, pa), sub(pc, pa))) else {
        return [0.0; 3];
    };
    if obj == IcpObjective::PointToPlane {
        return n_face;
    }
    let pick = |vid: u32| -> [f64; 3] {
        let vn = target_vnorms[vid as usize];
        if dot(vn, vn) > 0.0 && dot(vn, n_face) > 0.85 {
            vn
        } else {
            n_face
        }
    };
    let na = pick(a);
    let nb = pick(b);
    let nc = pick(c);
    let [u, v, w] = hit.barycentric.map(f64::from);
    let interp = [
        u * na[0] + v * nb[0] + w * nc[0],
        u * na[1] + v * nb[1] + w * nc[1],
        u * na[2] + v * nb[2] + w * nc[2],
    ];
    unit(interp).unwrap_or(n_face)
}

/// Solves `(A^T A + ridge I) x = b` in place for small dense symmetric `n x n` (`n <= 16`).
#[allow(clippy::needless_range_loop)]
fn solve_small_sym_system<'b>(a: &mut [f64], b: &'b mut [f64], n: usize) -> Option<&'b [f64]> {
    if n == 0 {
        return Some(b);
    }
    let tr: f64 = (0..n).map(|i| a[i * n + i]).sum();
    let ridge = (1e-10 * tr / n as f64).max(1e-18);
    for i in 0..n {
        a[i * n + i] += ridge;
    }
    // Gaussian elimination with partial pivoting on small `n <= 16` symmetric system.
    for k in 0..n {
        let mut piv = k;
        let mut max_v = a[k * n + k].abs();
        for i in k + 1..n {
            let v = a[i * n + k].abs();
            if v > max_v {
                max_v = v;
                piv = i;
            }
        }
        if max_v <= 1e-24 || !max_v.is_finite() {
            return None;
        }
        if piv != k {
            for j in k..n {
                a.swap(k * n + j, piv * n + j);
            }
            b.swap(k, piv);
        }
        let diag = a[k * n + k];
        for i in k + 1..n {
            let factor = a[i * n + k] / diag;
            a[i * n + k] = 0.0;
            for j in k + 1..n {
                a[i * n + j] -= factor * a[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= a[i * n + j] * b[j];
        }
        b[i] = s / a[i * n + i];
        if !b[i].is_finite() {
            return None;
        }
    }
    Some(b)
}

struct AndersonState {
    window: usize,
    anchor: Rigid,
    center0: [f64; 3],
    extent: f64,
    x_hist: Vec<[f64; 6]>,
    f_hist: Vec<[f64; 6]>,
    r_hist: Vec<[f64; 6]>,
}

impl AndersonState {
    fn new(window: usize, anchor: Rigid, center0: [f64; 3], extent: f64) -> Result<Self> {
        let cap = window.min(16).saturating_add(1);
        let mut x_hist = Vec::new();
        let mut f_hist = Vec::new();
        let mut r_hist = Vec::new();
        if window > 0 {
            reserve(&mut x_hist, cap)?;
            reserve(&mut f_hist, cap)?;
            reserve(&mut r_hist, cap)?;
        }
        Ok(AndersonState {
            window: window.min(16),
            anchor,
            center0,
            extent,
            x_hist,
            f_hist,
            r_hist,
        })
    }

    fn clear(&mut self, anchor: Rigid) {
        self.anchor = anchor;
        self.x_hist.clear();
        self.f_hist.clear();
        self.r_hist.clear();
    }

    fn encode(&self, pose: &Rigid) -> [f64; 6] {
        let c_cur = apply(pose, self.center0);
        let c_ref = apply(&self.anchor, self.center0);
        let dt = sub(c_cur, c_ref);
        let d_rot = orthonormal(mat3_mul_transpose_right(
            pose.rotation,
            self.anchor.rotation,
        ));
        let w = axis_angle_of(d_rot);
        [
            dt[0],
            dt[1],
            dt[2],
            w[0] * self.extent,
            w[1] * self.extent,
            w[2] * self.extent,
        ]
    }

    fn decode(&self, vec: &[f64; 6]) -> Rigid {
        let w = [
            vec[3] / self.extent,
            vec[4] / self.extent,
            vec[5] / self.extent,
        ];
        let d_rot = rotation_of(w);
        let rotation = orthonormal(mat3_mul(d_rot, self.anchor.rotation));
        let c_ref = apply(&self.anchor, self.center0);
        let c_new = [c_ref[0] + vec[0], c_ref[1] + vec[1], c_ref[2] + vec[2]];
        let r_c0 = mat3_vec(rotation, self.center0);
        Rigid {
            rotation,
            translation: sub(c_new, r_c0),
        }
    }

    fn extrapolate(&mut self, cur: &Rigid, plain: &Rigid) -> Result<Option<Rigid>> {
        if self.window == 0 {
            return Ok(None);
        }
        let x_k = self.encode(cur);
        // Re-anchor before approaching branch cut `||w|| ~ pi`.
        let rot_norm = norm([x_k[3], x_k[4], x_k[5]]) / self.extent;
        if rot_norm > 1.0 {
            self.clear(*cur);
        }
        let x_k = self.encode(cur);
        let f_k = self.encode(plain);
        let r_k: [f64; 6] = std::array::from_fn(|i| f_k[i] - x_k[i]);

        if self.x_hist.len() > self.window {
            self.x_hist.remove(0);
            self.f_hist.remove(0);
            self.r_hist.remove(0);
        }
        self.x_hist.push(x_k);
        self.f_hist.push(f_k);
        self.r_hist.push(r_k);

        let m = self.r_hist.len().saturating_sub(1);
        if m == 0 {
            return Ok(None);
        }

        let mut gram = [0.0_f64; 256];
        let mut rhs = [0.0_f64; 16];
        for j in 0..m {
            let dr_j: [f64; 6] = std::array::from_fn(|d| self.r_hist[j + 1][d] - self.r_hist[j][d]);
            rhs[j] = (0..6).map(|d| dr_j[d] * r_k[d]).sum();
            for l in j..m {
                let dr_l: [f64; 6] =
                    std::array::from_fn(|d| self.r_hist[l + 1][d] - self.r_hist[l][d]);
                let v: f64 = (0..6).map(|d| dr_j[d] * dr_l[d]).sum();
                gram[j * m + l] = v;
                gram[l * m + j] = v;
            }
        }
        let Some(gamma) = solve_small_sym_system(&mut gram[..m * m], &mut rhs[..m], m) else {
            return Ok(None);
        };
        let l1: f64 = gamma.iter().map(|v| v.abs()).sum();
        if !l1.is_finite() || l1 > 8.0 {
            return Ok(None);
        }
        let mut x_aa = f_k;
        for (j, &g_j) in gamma.iter().enumerate() {
            for (d, x) in x_aa.iter_mut().enumerate() {
                let df = self.f_hist[j + 1][d] - self.f_hist[j][d];
                *x -= g_j * df;
            }
        }
        let jump_rot = norm([x_aa[3] - x_k[3], x_aa[4] - x_k[4], x_aa[5] - x_k[5]]) / self.extent;
        if !x_aa.iter().all(|v| v.is_finite()) || jump_rot > MAX_STEP_ANGLE * 2.0 {
            return Ok(None);
        }
        Ok(Some(self.decode(&x_aa)))
    }
}

fn validate_registration_inputs(
    source: &Mesh,
    target: &Mesh,
    initial: Option<&IcpMotion>,
    options: &RegistrationOptions,
) -> Result<()> {
    check_mesh_finite_and_nonempty(source, "source mesh has no triangles")?;
    check_mesh_finite_and_nonempty(target, "target mesh has no triangles")?;
    if options.sample_count == 0 {
        return Err(OpsError::InvalidInput("sample_count must be at least 1"));
    }
    if !options.tolerance.is_finite() || options.tolerance < 0.0 {
        return Err(OpsError::InvalidInput(
            "tolerance must be finite and zero or greater",
        ));
    }
    if !(options.overlap > 0.0 && options.overlap <= 1.0) {
        return Err(OpsError::InvalidInput("overlap must be in (0, 1]"));
    }
    if options.max_distance.is_some_and(|d| d.is_nan() || d <= 0.0) {
        return Err(OpsError::InvalidInput("max_distance must be positive"));
    }
    if !options.target_rmse.is_finite() || options.target_rmse < 0.0 {
        return Err(OpsError::InvalidInput(
            "target_rmse must be finite and zero or greater",
        ));
    }
    if initial.is_some_and(|init| {
        !init
            .rotation
            .iter()
            .chain(&init.translation)
            .chain(&init.about)
            .all(|v| v.is_finite())
    }) {
        return Err(OpsError::InvalidInput("initial motion must be finite"));
    }
    Ok(())
}

/// Runs multi-stage global coarse alignment (`CoarseAlignMethod`) followed by
/// multi-objective fine ICP (`IcpObjective`) with optional Anderson acceleration
/// and strict monotonic trimmed-RMSE safeguarding.
///
/// # Errors
/// [`OpsError::InvalidInput`] on empty meshes or invalid options,
/// [`OpsError::Query`] on BVH failures, or [`OpsError::Store`] on OOM.
pub fn register_multistage(
    source: &Mesh,
    target: &Mesh,
    target_bvh: &Bvh,
    initial: Option<IcpMotion>,
    options: &RegistrationOptions,
) -> Result<RegistrationReport> {
    validate_registration_inputs(source, target, initial.as_ref(), options)?;

    let sampled = sample_surface_with_normals(source, options.sample_count)?;
    if sampled.points.is_empty() {
        return Err(OpsError::InvalidInput("source mesh has no surface samples"));
    }
    let init_rigid = initial.as_ref().map_or(Rigid::IDENTITY, motion_to_rigid);
    let start = run_coarse_stage(
        source,
        target,
        target_bvh,
        &sampled.points,
        init_rigid,
        options,
    )?;

    let target_vnorms = if options.objective == IcpObjective::SymmetricNormal {
        compute_vertex_normals(target)?
    } else {
        Vec::new()
    };

    let (center0, extent) = spread(&sampled.points);
    let mut eval = MultiEvaluator::new(&sampled, target, target_bvh, &target_vnorms, options)?;
    let mut aa = AndersonState::new(options.anderson_window, start, center0, extent)?;

    let mut cur = start;
    let Some(mut cur_rmse) = eval.evaluate(&cur)? else {
        return Ok(build_registration_report(
            cur,
            f64::INFINITY,
            0,
            IcpStatus::Diverged,
        ));
    };
    let mut cur_inliers = eval.kept.len();
    let mut best = (cur, cur_rmse, cur_inliers);
    let mut iterations = 0usize;
    let mut settled_status: Option<IcpStatus> = None;

    let status = loop {
        if let Some(st) = settled_status {
            break st;
        }
        if cur_inliers < MIN_CORRESPONDENCES {
            break IcpStatus::TooFewCorrespondences;
        }
        if options.target_rmse > 0.0 && iterations > 0 && cur_rmse <= options.target_rmse {
            break IcpStatus::Converged;
        }
        if iterations == options.max_iterations {
            break IcpStatus::MaxIterations;
        }

        let center = apply(&cur, center0);
        let Some((mut plain, mut plain_step_frac, cond)) = eval.compute_step(&cur, center, extent)
        else {
            break IcpStatus::TooFewCorrespondences;
        };
        iterations += 1;

        // Evaluate full plain step and backtrack if Euclidean trimmed RMSE worsens.
        let mut plain_eval = eval.evaluate(&plain)?;
        if plain_eval.is_none() {
            plain = cur;
            plain_eval = Some(cur_rmse);
            plain_step_frac = 0.0;
        }
        let mut plain_rmse = plain_eval.unwrap_or(f64::INFINITY);
        let mut plain_inliers = eval.kept.len();

        // Guard only against correspondence collapse or severe divergence (`> 4x` cur_rmse),
        // letting tangent-plane / symmetric-normal steps slide along curved surfaces cleanly.
        let overshoot = plain_inliers < MIN_CORRESPONDENCES
            || (plain_inliers < cur_inliers / 2)
            || (plain_inliers == cur_inliers && plain_rmse > cur_rmse * 4.0);
        if overshoot && plain_step_frac > 0.0 {
            let mut damp = 0.5_f64;
            let orig_step_frac = plain_step_frac;
            let d_rot = orthonormal(mat3_mul_transpose_right(plain.rotation, cur.rotation));
            let w_full = axis_angle_of(d_rot);
            let t_full = sub(apply(&plain, center0), center);
            for _ in 0..4 {
                let d = compose_step(
                    &cur,
                    center,
                    t_full.map(|v| v * damp),
                    w_full.map(|v| v * damp),
                );
                if let Some(d_rmse) = eval.evaluate(&d)? {
                    let d_inliers = eval.kept.len();
                    let improved_over_plain = d_inliers > plain_inliers
                        || (d_inliers == plain_inliers && d_rmse < plain_rmse);
                    if improved_over_plain {
                        plain = d;
                        plain_rmse = d_rmse;
                        plain_inliers = d_inliers;
                        plain_step_frac = orig_step_frac * damp;
                    }
                    if d_inliers >= cur_inliers && d_rmse <= cur_rmse * 1.1 {
                        break;
                    }
                }
                damp *= 0.5;
            }
        }

        let mut next = plain;
        let mut next_rmse = plain_rmse;
        let mut next_inliers = plain_inliers;
        let mut used_aa = false;

        if let Some(aa_pose) = aa.extrapolate(&cur, &plain)? {
            if let Some(aa_rmse) = eval.evaluate(&aa_pose)? {
                let aa_inliers = eval.kept.len();
                // Strict monotonic trimmed-RMSE safeguard against plain and current states.
                let beats_plain = aa_inliers >= plain_inliers
                    && (aa_inliers > plain_inliers || aa_rmse <= plain_rmse);
                let non_worse_cur =
                    aa_inliers > cur_inliers || (aa_inliers == cur_inliers && aa_rmse <= cur_rmse);
                if beats_plain && non_worse_cur {
                    next = aa_pose;
                    next_rmse = aa_rmse;
                    next_inliers = aa_inliers;
                    used_aa = true;
                }
            }
            if !used_aa {
                aa.clear(plain);
            }
        }

        // Re-synchronize `eval.kept` only if we took `plain` after evaluating `aa_pose`.
        if !used_aa && options.anderson_window > 0 {
            let _ = eval.evaluate(&next)?;
            next_inliers = eval.kept.len();
        }

        let step_moved = if used_aa {
            let d_rot = orthonormal(mat3_mul_transpose_right(next.rotation, cur.rotation));
            let w = axis_angle_of(d_rot);
            let t = sub(apply(&next, center0), center);
            (norm(t) + norm(w) * extent) / extent
        } else {
            plain_step_frac
        };

        cur = next;
        cur_rmse = next_rmse;
        cur_inliers = next_inliers;
        if cur_inliers > best.2 || (cur_inliers == best.2 && cur_rmse < best.1) {
            best = (cur, cur_rmse, cur_inliers);
        }

        if options.target_rmse > 0.0 && cur_rmse <= options.target_rmse {
            settled_status = Some(IcpStatus::Converged);
        } else if step_moved <= options.tolerance {
            settled_status = Some(if cond >= MIN_CONDITIONING {
                IcpStatus::Converged
            } else {
                IcpStatus::Underconstrained
            });
        }
    };

    let (final_pose, final_rmse) = if settled_status == Some(status)
        || status == IcpStatus::Converged
        || status == IcpStatus::Underconstrained
    {
        (cur, cur_rmse)
    } else {
        (best.0, best.1)
    };
    Ok(build_registration_report(
        final_pose, final_rmse, iterations, status,
    ))
}

fn build_registration_report(
    transform_rigid: Rigid,
    final_rmse: f64,
    iterations: usize,
    status: IcpStatus,
) -> RegistrationReport {
    RegistrationReport {
        motion: rigid_to_motion(&transform_rigid),
        transform: rigid_to_matrix4(&transform_rigid, 1.0),
        final_rmse,
        iterations,
        status,
        converged: status == IcpStatus::Converged,
    }
}

#[inline]
fn reflect_point(p: [f64; 3], n: [f64; 3], d: f64) -> [f64; 3] {
    let signed = 2.0 * (dot(p, n) - d);
    [
        p[0] - signed * n[0],
        p[1] - signed * n[1],
        p[2] - signed * n[2],
    ]
}

fn canonicalize_plane_normal(mut n: [f64; 3]) -> [f64; 3] {
    n = unit(n).unwrap_or([1.0, 0.0, 0.0]);
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

struct SymmetryEvalBuf {
    rf: Vec<[f32; 3]>,
    r64: Vec<[f64; 3]>,
    hits: Vec<Option<ClosestHit>>,
    q_pts: Vec<[f64; 3]>,
}

impl SymmetryEvalBuf {
    fn new(len: usize) -> Result<Self> {
        Ok(SymmetryEvalBuf {
            rf: filled(len, [0.0; 3])?,
            r64: filled(len, [0.0; 3])?,
            hits: filled(len, None)?,
            q_pts: filled(len, [0.0; 3])?,
        })
    }

    fn evaluate_rms(
        &mut self,
        points: &[[f64; 3]],
        mesh: &Mesh,
        bvh: &Bvh,
        normal: [f64; 3],
        offset: f64,
    ) -> Result<f64> {
        let query = ClosestOptions {
            max_distance: f32::INFINITY,
        };
        for ((p, rf), r64) in points
            .iter()
            .zip(self.rf.iter_mut())
            .zip(self.r64.iter_mut())
        {
            let q = reflect_point(*p, normal, offset);
            *r64 = q;
            *rf = q.map(|c| c as f32);
        }
        closest_point_many(mesh, bvh, &self.rf, &query, &mut self.hits)?;
        let (r64_ref, hits_ref) = (&self.r64, &self.hits);
        par::for_each_chunk_mut(
            &mut self.q_pts,
            DISTANCE_CHUNK,
            PAR_MIN_DISTANCES,
            |i, slots| {
                let start = i * DISTANCE_CHUNK;
                for (k, slot) in slots.iter_mut().enumerate() {
                    let idx = start + k;
                    let Some(hit) = hits_ref[idx] else {
                        *slot = r64_ref[idx];
                        continue;
                    };
                    if let Some(tri) = corners_of(mesh, hit) {
                        let (q, _) = closest_on_triangle_f64(r64_ref[idx], tri);
                        *slot = q;
                    } else {
                        *slot = hit.point.map(f64::from);
                    }
                }
            },
        );
        let mut sq_sum = 0.0_f64;
        let mut count = 0usize;
        for ((r, q), hit) in self.r64.iter().zip(&self.q_pts).zip(&self.hits) {
            if hit.is_some() {
                let diff = sub(*r, *q);
                sq_sum += dot(diff, diff);
                count += 1;
            }
        }
        if count == 0 {
            return Ok(f64::INFINITY);
        }
        Ok((sq_sum / count as f64).sqrt())
    }
}

/// Closed-form eigenvalue plane update using the reflection identity:
/// `sum ||r(p_i) - q_i||^2 = const + n^T [ 4 sum (m_i - m_bar)(m_i - m_bar)^T - sum (p_i - q_i)(p_i - q_i)^T ] n`
/// with optimal `d^*(n) = dot(n, m_bar)`.
fn symmetry_eigensolver_update(
    points: &[[f64; 3]],
    closest_pts: &[[f64; 3]],
    hits: &[Option<ClosestHit>],
    cur_normal: [f64; 3],
) -> Option<([f64; 3], f64)> {
    let mut m_sum = [0.0_f64; 3];
    let mut count = 0usize;
    for ((p, q), hit) in points.iter().zip(closest_pts).zip(hits) {
        if hit.is_some() {
            m_sum = add(
                m_sum,
                [
                    0.5 * (p[0] + q[0]),
                    0.5 * (p[1] + q[1]),
                    0.5 * (p[2] + q[2]),
                ],
            );
            count += 1;
        }
    }
    if count < 3 {
        return None;
    }
    let m_bar = m_sum.map(|c| c / count as f64);
    let mut mat = [[0.0_f64; 3]; 3];
    for ((p, q), hit) in points.iter().zip(closest_pts).zip(hits) {
        if hit.is_none() {
            continue;
        }
        let dm = [
            0.5 * (p[0] + q[0]) - m_bar[0],
            0.5 * (p[1] + q[1]) - m_bar[1],
            0.5 * (p[2] + q[2]) - m_bar[2],
        ];
        let dpq = sub(*p, *q);
        for i in 0..3 {
            for j in 0..3 {
                mat[i][j] += 4.0 * dm[i] * dm[j] - dpq[i] * dpq[j];
            }
        }
    }
    let (_vals, vecs) = jacobi_eigen_3x3(mat);
    // `vecs` column 2 corresponds to the smallest eigenvalue of `mat`.
    let mut best_n = [vecs[0][2], vecs[1][2], vecs[2][2]];
    if dot(best_n, cur_normal) < 0.0 {
        best_n = best_n.map(|c| -c);
    }
    // Reject jumps larger than 45 deg so correspondence basin cannot jump away.
    if dot(best_n, cur_normal) < std::f64::consts::FRAC_1_SQRT_2 {
        return None;
    }
    let best_d = dot(best_n, m_bar);
    Some((best_n, best_d))
}

impl SymmetryEvalBuf {
    /// Tangent point-to-plane plane update `(n, d) -> (unit(n + u), d + dd)` with `u perp n`.
    #[allow(clippy::needless_range_loop)]
    fn tangent_update(
        &self,
        points: &[[f64; 3]],
        mesh: &Mesh,
        cur_normal: [f64; 3],
        cur_d: f64,
        bbox_diag: f64,
    ) -> Option<([f64; 3], f64)> {
        let t1 = orthogonal_axis(cur_normal);
        let t2 = unit(cross(cur_normal, t1))?;
        let mut h = [[0.0_f64; 3]; 3];
        let mut g = [0.0_f64; 3];
        let mut rows = 0usize;

        for (((p, r), q), hit) in points
            .iter()
            .zip(&self.r64)
            .zip(&self.q_pts)
            .zip(&self.hits)
        {
            let Some(hit) = hit else {
                continue;
            };
            let Some(tri) = corners_of(mesh, *hit) else {
                continue;
            };
            let Some(nq) = unit(cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]))) else {
                continue;
            };
            let dist_plane = dot(*p, cur_normal) - cur_d;
            let nn = dot(cur_normal, nq);
            let j0 = -2.0 * (dot(*p, t1) * nn + dist_plane * dot(t1, nq));
            let j1 = -2.0 * (dot(*p, t2) * nn + dist_plane * dot(t2, nq));
            let j2 = 2.0 * nn * bbox_diag;
            let row = [j0, j1, j2];
            let rhs = dot(sub(*q, *r), nq);
            rows += 1;
            for i in 0..3 {
                g[i] += row[i] * rhs;
                for j in i..3 {
                    h[i][j] += row[i] * row[j];
                }
            }
        }
        if rows < 6 {
            return None;
        }
        for i in 0..3 {
            for j in 0..i {
                h[i][j] = h[j][i];
            }
        }
        let mut mat_flat = [
            h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
        ];
        let mut g_vec = [g[0], g[1], g[2]];
        let sol = solve_small_sym_system(&mut mat_flat, &mut g_vec, 3)?;
        let step_u = [
            sol[0] * t1[0] + sol[1] * t2[0],
            sol[0] * t1[1] + sol[1] * t2[1],
            sol[0] * t1[2] + sol[1] * t2[2],
        ];
        let u_len = norm(step_u);
        let max_step = 0.25;
        let scale = if u_len > max_step {
            max_step / u_len
        } else {
            1.0
        };
        let new_n = unit(add(cur_normal, step_u.map(|c| c * scale)))?;
        let new_d = cur_d + sol[2] * bbox_diag * scale;
        Some((new_n, new_d))
    }
}

/// Detects the dominant reflection symmetry plane of `mesh` (with BVH `bvh`).
///
/// Samples the surface deterministically, tests principal-axis and diagonal
/// candidate symmetry planes (plus azimuthal candidates across nearly degenerate
/// eigenspaces), refines the top planes via iterative reflective ICP, and returns
/// the best plane along with RMS deviation and normalized score in `[0.0, 1.0]`.
///
/// # Errors
/// [`OpsError::InvalidInput`] if `mesh` has no triangles or non-finite coordinates,
/// [`OpsError::Query`] on BVH failures, or [`OpsError::Store`] on OOM.
pub fn detect_symmetry_plane(mesh: &Mesh, bvh: &Bvh) -> Result<SymmetryPlaneReport> {
    check_mesh_finite_and_nonempty(mesh, "mesh has no triangles")?;
    let sampled = sample_surface_with_normals(mesh, SYMMETRY_SAMPLES)?;
    if sampled.points.is_empty() {
        return Err(OpsError::InvalidInput("mesh has no surface samples"));
    }

    let frame = compute_surface_frame(mesh);
    let diag = frame.bbox_diag;
    if !diag.is_finite() || diag <= 0.0 {
        return Err(OpsError::InvalidInput("mesh has zero bounding-box extent"));
    }

    let e0 = [frame.axes[0][0], frame.axes[1][0], frame.axes[2][0]];
    let e1 = [frame.axes[0][1], frame.axes[1][1], frame.axes[2][1]];
    let e2 = [frame.axes[0][2], frame.axes[1][2], frame.axes[2][2]];

    let mut candidate_normals: Vec<[f64; 3]> = Vec::new();
    let push_unique = |list: &mut Vec<[f64; 3]>, raw: [f64; 3]| -> Result<()> {
        let Some(u) = unit(raw) else {
            return Ok(());
        };
        if !list.iter().any(|existing| dot(*existing, u).abs() > 0.9999) {
            push(list, u)?;
        }
        Ok(())
    };

    for v in [
        e0,
        e1,
        e2,
        add(e0, e1),
        sub(e0, e1),
        add(e0, e2),
        sub(e0, e2),
        add(e1, e2),
        sub(e1, e2),
        add(add(e0, e1), e2),
        sub(add(e0, e1), e2),
        add(sub(e0, e1), e2),
        sub(sub(e0, e1), e2),
    ] {
        push_unique(&mut candidate_normals, v)?;
    }

    let eval_scale = frame.evals[0].max(1e-24);
    for (ei, ej, li, lj) in [
        (e0, e1, frame.evals[0], frame.evals[1]),
        (e1, e2, frame.evals[1], frame.evals[2]),
        (e0, e2, frame.evals[0], frame.evals[2]),
    ] {
        if (li - lj).abs() <= 0.25 * eval_scale {
            for k in 0..16 {
                let theta = std::f64::consts::PI * (k as f64) / 16.0;
                let (s, c) = theta.sin_cos();
                let v = [
                    c * ei[0] + s * ej[0],
                    c * ei[1] + s * ej[1],
                    c * ei[2] + s * ej[2],
                ];
                push_unique(&mut candidate_normals, v)?;
            }
        }
    }

    let coarse_pts = subsample_points(&sampled.points, 1024)?;
    let mut coarse_buf = SymmetryEvalBuf::new(coarse_pts.len())?;

    let mut scored = Vec::new();
    reserve(&mut scored, candidate_normals.len())?;
    for (idx, &n) in candidate_normals.iter().enumerate() {
        let d = dot(frame.centroid, n);
        let rms = coarse_buf.evaluate_rms(&coarse_pts, mesh, bvh, n, d)?;
        scored.push((rms, idx, n));
    }
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut fine_buf = SymmetryEvalBuf::new(sampled.points.len())?;

    let refine_count = scored.len().min(3);
    let mut best_report: Option<SymmetryPlaneReport> = None;
    for &(_, _, start_n) in scored.iter().take(refine_count) {
        let mut cur_n = start_n;
        let mut cur_d = dot(frame.centroid, cur_n);
        let mut cur_rms = fine_buf.evaluate_rms(&sampled.points, mesh, bvh, cur_n, cur_d)?;

        for _ in 0..16 {
            let mut next_choice: Option<([f64; 3], f64, f64)> = None;
            let eig_cand = symmetry_eigensolver_update(
                &sampled.points,
                &fine_buf.q_pts,
                &fine_buf.hits,
                cur_n,
            );
            let tan_cand = fine_buf.tangent_update(&sampled.points, mesh, cur_n, cur_d, diag);
            for (cand_n, cand_d) in [eig_cand, tan_cand].into_iter().flatten() {
                let rms = fine_buf.evaluate_rms(&sampled.points, mesh, bvh, cand_n, cand_d)?;
                let better = rms < cur_rms - 1e-12 * diag
                    && next_choice.is_none_or(|(_, _, best_r)| rms < best_r);
                if better {
                    next_choice = Some((cand_n, cand_d, rms));
                }
            }
            let Some((new_n, new_d, new_rms)) = next_choice else {
                break;
            };
            let change = norm(sub(new_n, cur_n)) + (new_d - cur_d).abs() / diag;
            cur_n = new_n;
            cur_d = new_d;
            cur_rms = new_rms;
            // Recompute correspondences on final chosen iterate before next step.
            let _ = fine_buf.evaluate_rms(&sampled.points, mesh, bvh, cur_n, cur_d)?;
            if change < 1e-8 {
                break;
            }
        }

        let plane_normal = canonicalize_plane_normal(cur_n);
        let signed_d = if dot(plane_normal, cur_n) < 0.0 {
            -cur_d
        } else {
            cur_d
        };
        let proj = signed_d - dot(frame.centroid, plane_normal);
        let plane_point = [
            frame.centroid[0] + proj * plane_normal[0],
            frame.centroid[1] + proj * plane_normal[1],
            frame.centroid[2] + proj * plane_normal[2],
        ];
        let norm_rms = (cur_rms / diag).max(0.0);
        let symmetry_score = (-6.0 * norm_rms).exp().clamp(0.0, 1.0);
        let candidate_report = SymmetryPlaneReport {
            plane_normal,
            plane_point,
            symmetry_error_rms: cur_rms,
            symmetry_score,
        };
        if best_report.is_none_or(|b| candidate_report.symmetry_error_rms < b.symmetry_error_rms) {
            best_report = Some(candidate_report);
        }
    }

    best_report.ok_or(OpsError::InvalidInput("failed to detect symmetry plane"))
}
