//! Rotational vortex and twisting-Kelvinlet brush kernels ([`twist`], [`twist_region`]).

use fast_mesh_store::{Mesh, Transaction, TriangleId, VertexId};

use super::filter::{
    MaxEnvelope, SurfaceFilter, Symmetry, check_surface_filter, eval_surface_filter,
    falloff_weighted_normal, filter_needs_prominence, reflect_vec, vertex_prominence,
};
use super::{
    Brush, Falloff, collect_proper_neighbors, falloff_value, has_any_region_member,
    normalized_dist, region_index, vertex_normal_from_triangles, write_moves,
};
use crate::topology::VertexFans;
use crate::{OpsError, Result, filled, push, reserve};

/// Normalized regularization core scale (`epsilon / R`) for [`TwistMethod::Kelvinlet`].
const TWIST_KELVINLET_EPSILON: f64 = 0.35;

/// Rotational kernel for [`twist`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TwistMethod {
    /// Exact Rodrigues rotation in the plane perpendicular to the twist axis by
    /// `strength * falloff(t)` radians, preserving axial height and cylindrical radius.
    #[default]
    Vortex,
    /// Calibrated regularized twisting Kelvinlet (de Goes & James 2017) windowed
    /// by `(1 - t^2)^2` to compact support at the brush rim.
    Kelvinlet,
}

/// Options for [`twist`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TwistOptions {
    /// The brush sphere, falloff, and twist angle scale in radians (`strength`).
    pub brush: Brush,
    /// Explicit twist axis in local coordinates (reflected as an axial vector per
    /// symmetry pass), or `None` to use the pass's falloff-weighted average normal.
    pub axis: Option<[f32; 3]>,
    /// Rotational kernel (`Vortex` by default).
    pub method: TwistMethod,
    /// Optional surface feature filter.
    pub filter: SurfaceFilter,
    /// Mirror symmetry axes (odd reflections flip axial twist handedness by `det(M) = -1`).
    pub symmetry: Symmetry,
}

impl TwistOptions {
    /// A builder for [`TwistOptions`]: default brush, auto normal axis,
    /// [`TwistMethod::Vortex`], no filter, and no symmetry.
    pub fn builder() -> TwistOptionsBuilder {
        TwistOptionsBuilder::default()
    }
}

/// A builder for [`TwistOptions`], see [`TwistOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct TwistOptionsBuilder {
    options: TwistOptions,
}

impl TwistOptionsBuilder {
    /// Sets the entire [`Brush`].
    pub fn brush(mut self, brush: Brush) -> Self {
        self.options.brush = brush;
        self
    }

    /// Center of the brush in local coordinates.
    pub fn center(mut self, center: [f32; 3]) -> Self {
        self.options.brush.center = center;
        self
    }

    /// Radius of influence in local units.
    pub fn radius(mut self, radius: f32) -> Self {
        self.options.brush.radius = radius;
        self
    }

    /// How the effect fades with distance from the center.
    pub fn falloff(mut self, falloff: Falloff) -> Self {
        self.options.brush.falloff = falloff;
        self
    }

    /// Twist angle scale in radians (`> 0` counter-clockwise around `axis`).
    pub fn strength(mut self, strength: f32) -> Self {
        self.options.brush.strength = strength;
        self
    }

    /// Explicit twist axis, or `None` for falloff-weighted brush normal.
    pub fn axis(mut self, axis: Option<[f32; 3]>) -> Self {
        self.options.axis = axis;
        self
    }

    /// Rotational kernel (`Vortex` or `Kelvinlet`).
    pub fn method(mut self, method: TwistMethod) -> Self {
        self.options.method = method;
        self
    }

    /// Surface feature filter.
    pub fn filter(mut self, filter: SurfaceFilter) -> Self {
        self.options.filter = filter;
        self
    }

    /// Mirror symmetry axes.
    pub fn symmetry(mut self, symmetry: Symmetry) -> Self {
        self.options.symmetry = symmetry;
        self
    }

    /// The options as set.
    pub fn build(self) -> TwistOptions {
        self.options
    }
}

/// What [`twist`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TwistReport {
    /// How many vertices moved (displacement above 0).
    pub moved: usize,
    /// The largest displacement applied, in local units; 0 when nothing moved.
    pub max_displacement: f32,
}

fn normalize_axis(v: [f32; 3]) -> Option<[f32; 3]> {
    let len = (f64::from(v[0]).powi(2) + f64::from(v[1]).powi(2) + f64::from(v[2]).powi(2)).sqrt();
    (len > 0.0 && len.is_finite()).then(|| {
        [
            (f64::from(v[0]) / len) as f32,
            (f64::from(v[1]) / len) as f32,
            (f64::from(v[2]) / len) as f32,
        ]
    })
}

fn check_twist(options: &TwistOptions) -> Result<()> {
    let brush = &options.brush;
    if !brush.radius.is_finite() || brush.radius <= 0.0 {
        return Err(OpsError::InvalidInput(
            "twist brush radius must be positive and finite",
        ));
    }
    if brush.center.iter().any(|c| !c.is_finite()) {
        return Err(OpsError::InvalidInput("twist brush center must be finite"));
    }
    if !brush.strength.is_finite() {
        return Err(OpsError::InvalidInput(
            "twist brush strength must be finite",
        ));
    }
    if let Some(a) = options.axis {
        if a.iter().any(|x| !x.is_finite()) {
            return Err(OpsError::InvalidInput("twist axis must be finite"));
        }
        if normalize_axis(a).is_none() {
            return Err(OpsError::InvalidInput("twist axis must be non-zero"));
        }
    }
    check_surface_filter(&options.filter)
}

/// Rotates vertices under `options.brush` around [`TwistOptions::axis`] (or the
/// brush's falloff-weighted average normal) according to [`TwistOptions::method`],
/// [`TwistOptions::filter`], and [`TwistOptions::symmetry`].
///
/// Under mirror symmetry, twist treats the axis as an axial vector (`axis' = M * axis`,
/// `strength' = det(M) * strength`), so mirrored vortices counter-rotate symmetrically.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when brush, axis, or filter parameters are invalid.
/// - [`OpsError::Store`] when memory runs out.
///
/// Nothing is written in any of these cases.
pub fn twist(tx: &mut Transaction, options: &TwistOptions) -> Result<TwistReport> {
    check_twist(options)?;
    if options.brush.strength == 0.0 {
        return Ok(TwistReport::default());
    }
    let mesh = tx.mesh();
    if mesh.vertex_slots() == 0
        || !has_any_region_member(mesh, &options.brush, options.symmetry, true)
    {
        return Ok(TwistReport::default());
    }

    let positions = mesh.positions();
    let (masks, mask_count) = options.symmetry.active_masks();
    let mut candidates: Vec<VertexId> = Vec::new();
    for (v, p) in positions.iter().enumerate() {
        let v = v as VertexId;
        if !mesh.is_vertex_alive(v) || mesh.corner_count(v) == 0 {
            continue;
        }
        let hit = masks[..mask_count].iter().any(|&bits| {
            normalized_dist(
                reflect_vec(bits, options.brush.center),
                options.brush.radius,
                p,
            )
            .is_some()
        });
        if hit {
            push(&mut candidates, v)?;
        }
    }
    if candidates.is_empty() {
        return Ok(TwistReport::default());
    }

    let need_normals = options.axis.is_none() || !matches!(options.filter, SurfaceFilter::None);
    let fans = if need_normals {
        Some(VertexFans::build(mesh)?)
    } else {
        None
    };

    let (ids, moved_to, report) = twist_candidates(mesh, options, &candidates, |v| {
        fans.as_ref().map_or(&[], |f| f.fan(v))
    })?;
    write_moves(tx, &ids, &moved_to)?;
    Ok(report)
}

/// [`twist`] over the candidate vertices in `region`, reading only `region` and
/// the mesh's incidence index ([`Mesh::ensure_incidence`]).
///
/// # Errors
/// As for [`twist`], plus [`OpsError::InvalidInput`] if the mesh lacks a
/// current incidence index.
pub fn twist_region(
    tx: &mut Transaction,
    options: &TwistOptions,
    region: &[VertexId],
) -> Result<TwistReport> {
    check_twist(options)?;
    let (ids, moved_to, report) = {
        let mesh = tx.mesh();
        let index = region_index(mesh)?;
        if options.brush.strength == 0.0 {
            return Ok(TwistReport::default());
        }
        let positions = mesh.positions();
        let (masks, mask_count) = options.symmetry.active_masks();

        let mut candidates: Vec<VertexId> = Vec::new();
        for &v in region {
            if !mesh.is_vertex_alive(v) || mesh.corner_count(v) == 0 {
                continue;
            }
            let p = &positions[v as usize];
            let hit = masks[..mask_count].iter().any(|&bits| {
                normalized_dist(
                    reflect_vec(bits, options.brush.center),
                    options.brush.radius,
                    p,
                )
                .is_some()
            });
            if hit {
                push(&mut candidates, v)?;
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.is_empty() {
            return Ok(TwistReport::default());
        }

        twist_candidates(mesh, options, &candidates, |v| index.triangles(v))?
    };
    write_moves(tx, &ids, &moved_to)?;
    Ok(report)
}

fn twist_candidates<'f>(
    mesh: &Mesh,
    options: &TwistOptions,
    candidates: &[VertexId],
    triangles_of: impl Fn(VertexId) -> &'f [TriangleId],
) -> Result<(Vec<VertexId>, Vec<[f32; 3]>, TwistReport)> {
    let positions = mesh.positions();
    let brush = &options.brush;
    let (masks, mask_count) = options.symmetry.active_masks();

    let need_normals = options.axis.is_none() || !matches!(options.filter, SurfaceFilter::None);
    let mut cand_normals: Vec<[f32; 3]> = Vec::new();
    if need_normals {
        reserve(&mut cand_normals, candidates.len())?;
        for &v in candidates {
            cand_normals.push(vertex_normal_from_triangles(mesh, triangles_of(v)));
        }
    }

    let mut prominences: Vec<f32> = Vec::new();
    if filter_needs_prominence(&options.filter) {
        reserve(&mut prominences, candidates.len())?;
        let mut adjacent: Vec<VertexId> = Vec::new();
        for (&v, &n_v) in candidates.iter().zip(&cand_normals) {
            collect_proper_neighbors(mesh, v, triangles_of(v), &mut adjacent)?;
            prominences.push(vertex_prominence(
                positions[v as usize],
                n_v,
                &adjacent,
                positions,
            ));
        }
    }

    let need_auto_normal = options.axis.is_none()
        || matches!(
            options.filter,
            SurfaceFilter::NormalAngle { normal: None, .. }
        );
    let mut envelopes = filled(candidates.len(), MaxEnvelope::default())?;

    for &bits in &masks[..mask_count] {
        let center_m = reflect_vec(bits, brush.center);
        let auto_normal = need_auto_normal.then(|| {
            falloff_weighted_normal(
                candidates,
                &cand_normals,
                positions,
                center_m,
                brush.radius,
                brush.falloff,
            )
        });
        let Some(axis_m) = (match options.axis {
            Some(explicit) => normalize_axis(reflect_vec(bits, explicit)),
            None => auto_normal.filter(|n| n[0] != 0.0 || n[1] != 0.0 || n[2] != 0.0),
        }) else {
            continue;
        };

        let det_m = if bits.count_ones() % 2 == 1 {
            -1.0f32
        } else {
            1.0f32
        };
        let strength_m = det_m * brush.strength;

        for (cand_idx, &v) in candidates.iter().enumerate() {
            let p = positions[v as usize];
            let Some(t) = normalized_dist(center_m, brush.radius, &p) else {
                continue;
            };
            let phi = eval_surface_filter(
                &options.filter,
                bits,
                cand_idx,
                &cand_normals,
                &prominences,
                auto_normal,
            );
            let d = [p[0] - center_m[0], p[1] - center_m[1], p[2] - center_m[2]];

            match options.method {
                TwistMethod::Vortex => {
                    let f_val = falloff_value(brush.falloff, t);
                    let a_m = f_val * phi;
                    if !(a_m > 0.0 && a_m.is_finite()) {
                        continue;
                    }
                    let theta = f64::from(strength_m) * f64::from(a_m);
                    let ax = axis_m.map(f64::from);
                    let df = d.map(f64::from);
                    let parallel = df[0] * ax[0] + df[1] * ax[1] + df[2] * ax[2];
                    let d_perp = [
                        df[0] - parallel * ax[0],
                        df[1] - parallel * ax[1],
                        df[2] - parallel * ax[2],
                    ];
                    let cr = [
                        ax[1] * d_perp[2] - ax[2] * d_perp[1],
                        ax[2] * d_perp[0] - ax[0] * d_perp[2],
                        ax[0] * d_perp[1] - ax[1] * d_perp[0],
                    ];
                    let (sin_t, cos_t) = theta.sin_cos();
                    let cm1 = cos_t - 1.0;
                    let dp = [
                        (cm1 * d_perp[0] + sin_t * cr[0]) as f32,
                        (cm1 * d_perp[1] + sin_t * cr[1]) as f32,
                        (cm1 * d_perp[2] + sin_t * cr[2]) as f32,
                    ];
                    let single_pos = [p[0] + dp[0], p[1] + dp[1], p[2] + dp[2]];
                    envelopes[cand_idx].add(a_m, dp, single_pos);
                }
                TwistMethod::Kelvinlet => {
                    let win = (1.0 - t * t).powi(2);
                    let a_m = win * phi;
                    if !(a_m > 0.0 && a_m.is_finite()) {
                        continue;
                    }
                    let e = TWIST_KELVINLET_EPSILON;
                    let tf = f64::from(t);
                    let rho2 = tf * tf + e * e;
                    let rho = rho2.sqrt();
                    let rho5 = rho2 * rho2 * rho;
                    let kernel = (e * e * e) * (2.0 * rho2 + 3.0 * e * e) / (5.0 * rho5);
                    let omega = f64::from(strength_m) * f64::from(a_m) * kernel;
                    let ax = axis_m.map(f64::from);
                    let df = d.map(f64::from);
                    let cr = [
                        ax[1] * df[2] - ax[2] * df[1],
                        ax[2] * df[0] - ax[0] * df[2],
                        ax[0] * df[1] - ax[1] * df[0],
                    ];
                    let dp = [
                        (omega * cr[0]) as f32,
                        (omega * cr[1]) as f32,
                        (omega * cr[2]) as f32,
                    ];
                    let single_pos = [p[0] + dp[0], p[1] + dp[1], p[2] + dp[2]];
                    envelopes[cand_idx].add(a_m, dp, single_pos);
                }
            }
        }
    }

    let mut report = TwistReport::default();
    let mut ids: Vec<VertexId> = Vec::new();
    let mut moved_to: Vec<[f32; 3]> = Vec::new();
    for (&v, env) in candidates.iter().zip(&envelopes) {
        let p = positions[v as usize];
        if let Some((displaced, length)) = env.resolve(p) {
            push(&mut ids, v)?;
            push(&mut moved_to, displaced)?;
            report.max_displacement = report.max_displacement.max(length);
        }
    }
    report.moved = ids.len();
    Ok((ids, moved_to, report))
}
