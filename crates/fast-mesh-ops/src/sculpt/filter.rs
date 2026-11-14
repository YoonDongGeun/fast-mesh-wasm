//! Composable surface feature filters ([`SurfaceFilter`]) and local-axis mirror
//! [`Symmetry`] with max-envelope centerline combination.

use fast_mesh_store::VertexId;

use super::{Falloff, falloff_value, normalized_dist};
use crate::{OpsError, Result};

/// Per-vertex feature mask composed multiplicatively with a brush's spatial
/// falloff (`a(v) = falloff(t) * filter(v)`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum SurfaceFilter {
    /// No feature filtering (`1.0` everywhere).
    #[default]
    None,
    /// Masks out valleys and flat regions, affecting raised ridges and peaks
    /// whose scale-invariant 1-ring prominence exceeds `threshold`.
    PeaksOnly {
        /// Prominence cutoff in `[0.0, 1.0]`.
        threshold: f32,
        /// Ramp sharpness in `[0.0, 1.0]` (`1.0` = crisp cutoff).
        contrast: f32,
    },
    /// Masks out peaks and flat regions, affecting recessed cavities and pits
    /// whose negative scale-invariant 1-ring prominence exceeds `threshold`.
    ValleysOnly {
        /// Cavity prominence cutoff in `[0.0, 1.0]`.
        threshold: f32,
        /// Ramp sharpness in `[0.0, 1.0]` (`1.0` = crisp cutoff).
        contrast: f32,
    },
    /// Masks vertices whose smooth normal deviates from a reference normal by
    /// more than `max_angle_deg` (`1.0` across the inner 75% of the cone,
    /// smoothstep-tapering to `0.0` at `max_angle_deg`).
    NormalAngle {
        /// Reference normal in local coordinates (reflected per symmetry pass),
        /// or `None` to use the pass's falloff-weighted average brush normal.
        normal: Option<[f32; 3]>,
        /// Cone half-angle in degrees, in `(0.0, 180.0]`.
        max_angle_deg: f32,
    },
}

/// Mirror symmetry across local coordinate planes `x = 0`, `y = 0`, `z = 0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Symmetry {
    /// Mirror across `x = 0`.
    pub x: bool,
    /// Mirror across `y = 0`.
    pub y: bool,
    /// Mirror across `z = 0`.
    pub z: bool,
}

impl Symmetry {
    /// No symmetry (`1` pass, mask `0`).
    pub const NONE: Self = Self {
        x: false,
        y: false,
        z: false,
    };

    /// Active reflection bitmasks (`bit 0 = X`, `bit 1 = Y`, `bit 2 = Z`) in
    /// ascending order, together with the active count (`1`, `2`, `4`, or `8`).
    pub fn active_masks(&self) -> ([u8; 8], usize) {
        let mut masks = [0u8; 8];
        let mut len = 0;
        for bits in 0u8..8 {
            if (bits & 1 != 0 && !self.x)
                || (bits & 2 != 0 && !self.y)
                || (bits & 4 != 0 && !self.z)
            {
                continue;
            }
            masks[len] = bits;
            len += 1;
        }
        (masks, len)
    }
}

/// Reflects a polar 3D vector by symmetry bitmask `bits` (`0..8`).
pub(crate) fn reflect_vec(bits: u8, v: [f32; 3]) -> [f32; 3] {
    [
        if bits & 1 != 0 { -v[0] } else { v[0] },
        if bits & 2 != 0 { -v[1] } else { v[1] },
        if bits & 4 != 0 { -v[2] } else { v[2] },
    ]
}

/// Validates [`SurfaceFilter`] parameters up front before any allocations or
/// writes occur.
pub(crate) fn check_surface_filter(filter: &SurfaceFilter) -> Result<()> {
    match filter {
        SurfaceFilter::None => Ok(()),
        SurfaceFilter::PeaksOnly {
            threshold,
            contrast,
        }
        | SurfaceFilter::ValleysOnly {
            threshold,
            contrast,
        } => {
            if !(0.0..=1.0).contains(threshold) {
                return Err(OpsError::InvalidInput(
                    "surface filter threshold must be finite and in [0, 1]",
                ));
            }
            if !(0.0..=1.0).contains(contrast) {
                return Err(OpsError::InvalidInput(
                    "surface filter contrast must be finite and in [0, 1]",
                ));
            }
            Ok(())
        }
        SurfaceFilter::NormalAngle {
            normal,
            max_angle_deg,
        } => {
            if !(*max_angle_deg > 0.0 && *max_angle_deg <= 180.0) {
                return Err(OpsError::InvalidInput(
                    "surface filter max_angle_deg must be finite and in (0, 180]",
                ));
            }
            if let Some(n) = normal {
                if n.iter().any(|x| !x.is_finite()) {
                    return Err(OpsError::InvalidInput(
                        "surface filter normal must be finite",
                    ));
                }
                let len =
                    (f64::from(n[0]).powi(2) + f64::from(n[1]).powi(2) + f64::from(n[2]).powi(2))
                        .sqrt();
                if !(len > 0.0 && len.is_finite()) {
                    return Err(OpsError::InvalidInput(
                        "surface filter normal must be non-zero",
                    ));
                }
            }
            Ok(())
        }
    }
}

pub(crate) fn filter_needs_normals(filter: &SurfaceFilter) -> bool {
    !matches!(filter, SurfaceFilter::None)
}

pub(crate) fn filter_needs_prominence(filter: &SurfaceFilter) -> bool {
    matches!(
        filter,
        SurfaceFilter::PeaksOnly { .. } | SurfaceFilter::ValleysOnly { .. }
    )
}

/// Scale-invariant 1-ring prominence `((p_v - mean(p_u)) . n_v) / mean_edge_len`
/// over the ascending deduplicated neighbor list `neighbors`.
pub(crate) fn vertex_prominence(
    p_v: [f32; 3],
    n_v: [f32; 3],
    neighbors: &[VertexId],
    positions: &[[f32; 3]],
) -> f32 {
    if neighbors.is_empty() || (n_v[0] == 0.0 && n_v[1] == 0.0 && n_v[2] == 0.0) {
        return 0.0;
    }
    let count = neighbors.len() as f32;
    let mut centroid = [0.0f32; 3];
    let mut edge_sum = 0.0f32;
    for &u in neighbors {
        let q = positions[u as usize];
        centroid[0] += q[0];
        centroid[1] += q[1];
        centroid[2] += q[2];
        let dx = q[0] - p_v[0];
        let dy = q[1] - p_v[1];
        let dz = q[2] - p_v[2];
        edge_sum += (dx * dx + dy * dy + dz * dz).sqrt();
    }
    centroid[0] /= count;
    centroid[1] /= count;
    centroid[2] /= count;
    let mean_len = edge_sum / count;
    if !(mean_len > 0.0 && mean_len.is_finite()) {
        return 0.0;
    }
    let diff = [
        p_v[0] - centroid[0],
        p_v[1] - centroid[1],
        p_v[2] - centroid[2],
    ];
    let h = (diff[0] * n_v[0] + diff[1] * n_v[1] + diff[2] * n_v[2]) / mean_len;
    if h.is_finite() { h } else { 0.0 }
}

/// Falloff-weighted unit average normal over `candidates` inside `Sphere(center, radius)`.
/// `normals` is parallel to `candidates` (which must be in ascending `VertexId` order).
pub(crate) fn falloff_weighted_normal(
    candidates: &[VertexId],
    normals: &[[f32; 3]],
    positions: &[[f32; 3]],
    center: [f32; 3],
    radius: f32,
    falloff: Falloff,
) -> [f32; 3] {
    let mut sum = [0.0f64; 3];
    for (&v, &n) in candidates.iter().zip(normals) {
        let p = &positions[v as usize];
        if let Some(t) = normalized_dist(center, radius, p) {
            let w = f64::from(falloff_value(falloff, t));
            sum[0] += w * f64::from(n[0]);
            sum[1] += w * f64::from(n[1]);
            sum[2] += w * f64::from(n[2]);
        }
    }
    let len = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
    if len > 0.0 && len.is_finite() {
        [
            (sum[0] / len) as f32,
            (sum[1] / len) as f32,
            (sum[2] / len) as f32,
        ]
    } else {
        [0.0; 3]
    }
}

fn smoothstep01(x: f32) -> f32 {
    let u = x.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// Evaluates `filter` in `[0.0, 1.0]` for candidate index `cand_idx` in symmetry
/// pass `bits`.
pub(crate) fn eval_surface_filter(
    filter: &SurfaceFilter,
    bits: u8,
    cand_idx: usize,
    normals: &[[f32; 3]],
    prominences: &[f32],
    auto_brush_normal: Option<[f32; 3]>,
) -> f32 {
    match *filter {
        SurfaceFilter::None => 1.0,
        SurfaceFilter::PeaksOnly {
            threshold,
            contrast,
        } => {
            let s = prominences[cand_idx];
            if s <= threshold {
                return 0.0;
            }
            let width = (0.25 * (1.0 - contrast)).max(1e-5);
            smoothstep01((s - threshold) / width)
        }
        SurfaceFilter::ValleysOnly {
            threshold,
            contrast,
        } => {
            let s = -prominences[cand_idx];
            if s <= threshold {
                return 0.0;
            }
            let width = (0.25 * (1.0 - contrast)).max(1e-5);
            smoothstep01((s - threshold) / width)
        }
        SurfaceFilter::NormalAngle {
            normal,
            max_angle_deg,
        } => {
            let n_v = normals[cand_idx];
            if n_v[0] == 0.0 && n_v[1] == 0.0 && n_v[2] == 0.0 {
                return 0.0;
            }
            let ref_unit = match normal {
                Some(explicit) => {
                    let r = reflect_vec(bits, explicit);
                    let len = (f64::from(r[0]).powi(2)
                        + f64::from(r[1]).powi(2)
                        + f64::from(r[2]).powi(2))
                    .sqrt();
                    if !(len > 0.0 && len.is_finite()) {
                        return 0.0;
                    }
                    [
                        (f64::from(r[0]) / len) as f32,
                        (f64::from(r[1]) / len) as f32,
                        (f64::from(r[2]) / len) as f32,
                    ]
                }
                None => {
                    let Some(avg) = auto_brush_normal else {
                        return 0.0;
                    };
                    if avg[0] == 0.0 && avg[1] == 0.0 && avg[2] == 0.0 {
                        return 0.0;
                    }
                    avg
                }
            };
            let cos_theta = (n_v[0] * ref_unit[0] + n_v[1] * ref_unit[1] + n_v[2] * ref_unit[2])
                .clamp(-1.0, 1.0);
            let angle = cos_theta.acos();
            let max_rad = max_angle_deg.to_radians();
            let inner = 0.75 * max_rad;
            if angle <= inner {
                1.0
            } else if angle >= max_rad {
                0.0
            } else {
                smoothstep01((max_rad - angle) / (max_rad - inner))
            }
        }
    }
}

/// Per-vertex max-envelope combiner across active mirror symmetry passes:
/// `gamma(v) = max_m a_m(v) / sum_m a_m(v)`, `dp(v) = gamma(v) * sum_m dp_m(v)`.
#[derive(Clone, Copy, Default)]
pub(crate) struct MaxEnvelope {
    hits: u8,
    sum_a: f32,
    max_a: f32,
    sum_dp: [f32; 3],
    single_pos: [f32; 3],
}

impl MaxEnvelope {
    pub(crate) fn add(&mut self, a: f32, dp: [f32; 3], single_pos: [f32; 3]) {
        if !(a > 0.0 && a.is_finite()) {
            return;
        }
        if self.hits == 0 {
            self.single_pos = single_pos;
        }
        self.hits += 1;
        self.sum_a += a;
        self.max_a = self.max_a.max(a);
        self.sum_dp[0] += dp[0];
        self.sum_dp[1] += dp[1];
        self.sum_dp[2] += dp[2];
    }

    pub(crate) fn resolve(&self, base_pos: [f32; 3]) -> Option<([f32; 3], f32)> {
        if self.hits == 0 || !(self.sum_a > 0.0 && self.sum_a.is_finite()) {
            return None;
        }
        if self.hits == 1 {
            let length =
                (self.sum_dp[0].powi(2) + self.sum_dp[1].powi(2) + self.sum_dp[2].powi(2)).sqrt();
            return (length.is_finite()
                && length > 0.0
                && self.single_pos.iter().all(|x| x.is_finite()))
            .then_some((self.single_pos, length));
        }
        let gamma = self.max_a / self.sum_a;
        let dp = [
            gamma * self.sum_dp[0],
            gamma * self.sum_dp[1],
            gamma * self.sum_dp[2],
        ];
        let length = (dp[0].powi(2) + dp[1].powi(2) + dp[2].powi(2)).sqrt();
        let displaced = [
            base_pos[0] + dp[0],
            base_pos[1] + dp[1],
            base_pos[2] + dp[2],
        ];
        (length.is_finite() && length > 0.0 && displaced.iter().all(|x| x.is_finite()))
            .then_some((displaced, length))
    }
}
