//! Plane-fitting sculpt brushes: [`flatten`] (`Both` / `Peaks` / `Valleys`) and
//! [`clay`] (`Round` / `Strips`).

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

/// Which side(s) of the brush plane [`flatten`] projects toward the plane.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FlattenMode {
    /// Pulls both peaks (`signed_dist > 0`) down and valleys (`signed_dist < 0`)
    /// up toward the brush plane.
    #[default]
    Both,
    /// Scrape / trim: only pushes vertices above the brush plane (`signed_dist > 0`)
    /// down toward the plane.
    Peaks,
    /// Fill: only lifts vertices below the brush plane (`signed_dist < 0`) up
    /// toward the plane.
    Valleys,
}

/// Options for [`flatten`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FlattenOptions {
    /// The brush sphere, falloff, and strength.
    pub brush: Brush,
    /// Whether to flatten both sides, peaks only (scrape), or valleys only (fill).
    pub mode: FlattenMode,
    /// Explicit plane origin in local coordinates (reflected per symmetry pass),
    /// or `None` to fit the falloff-weighted centroid of vertices under the brush.
    pub plane_origin: Option<[f32; 3]>,
    /// Explicit plane normal in local coordinates (reflected per symmetry pass),
    /// or `None` to fit the falloff-weighted unit average normal under the brush.
    pub plane_normal: Option<[f32; 3]>,
    /// Optional surface feature filter.
    pub filter: SurfaceFilter,
    /// Mirror symmetry axes.
    pub symmetry: Symmetry,
}

impl FlattenOptions {
    /// A builder for [`FlattenOptions`]: default brush, [`FlattenMode::Both`],
    /// auto-fitted plane, no filter, and no symmetry.
    pub fn builder() -> FlattenOptionsBuilder {
        FlattenOptionsBuilder::default()
    }
}

/// A builder for [`FlattenOptions`], see [`FlattenOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct FlattenOptionsBuilder {
    options: FlattenOptions,
}

impl FlattenOptionsBuilder {
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

    /// Blend factor toward the plane (`0.0..=1.0` typical; must be finite).
    pub fn strength(mut self, strength: f32) -> Self {
        self.options.brush.strength = strength;
        self
    }

    /// Whether to move both sides, peaks only, or valleys only.
    pub fn mode(mut self, mode: FlattenMode) -> Self {
        self.options.mode = mode;
        self
    }

    /// Explicit plane origin, or `None` for falloff-weighted centroid.
    pub fn plane_origin(mut self, plane_origin: Option<[f32; 3]>) -> Self {
        self.options.plane_origin = plane_origin;
        self
    }

    /// Explicit plane normal, or `None` for falloff-weighted normal.
    pub fn plane_normal(mut self, plane_normal: Option<[f32; 3]>) -> Self {
        self.options.plane_normal = plane_normal;
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
    pub fn build(self) -> FlattenOptions {
        self.options
    }
}

/// What [`flatten`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FlattenReport {
    /// How many vertices moved (displacement above 0).
    pub moved: usize,
    /// The largest displacement applied, in local units; 0 when nothing moved.
    pub max_displacement: f32,
}

/// Footprint profile for [`clay`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ClayProfile {
    /// Circular brush sphere footprint.
    #[default]
    Round,
    /// Oriented `2R x 2R x 2R` tangent-frame square strip footprint aligned to
    /// the stroke tangent.
    Strips,
}

/// Options for [`clay`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ClayOptions {
    /// The brush sphere, falloff, and signed strength (`> 0` builds up to
    /// `+0.25 * R`, `< 0` carves down to `-0.25 * R`).
    pub brush: Brush,
    /// Circular [`ClayProfile::Round`] or square [`ClayProfile::Strips`] footprint.
    pub profile: ClayProfile,
    /// Optional stroke tangent direction for [`ClayProfile::Strips`] (reflected
    /// per symmetry pass and projected onto the brush plane).
    pub tangent: Option<[f32; 3]>,
    /// Optional surface feature filter.
    pub filter: SurfaceFilter,
    /// Mirror symmetry axes.
    pub symmetry: Symmetry,
}

impl ClayOptions {
    /// A builder for [`ClayOptions`]: default brush, [`ClayProfile::Round`],
    /// auto tangent, no filter, and no symmetry.
    pub fn builder() -> ClayOptionsBuilder {
        ClayOptionsBuilder::default()
    }
}

/// A builder for [`ClayOptions`], see [`ClayOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct ClayOptionsBuilder {
    options: ClayOptions,
}

impl ClayOptionsBuilder {
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

    /// Signed strength (`> 0` builds clay, `< 0` carves clay).
    pub fn strength(mut self, strength: f32) -> Self {
        self.options.brush.strength = strength;
        self
    }

    /// Footprint profile (`Round` or `Strips`).
    pub fn profile(mut self, profile: ClayProfile) -> Self {
        self.options.profile = profile;
        self
    }

    /// Optional stroke tangent for [`ClayProfile::Strips`].
    pub fn tangent(mut self, tangent: Option<[f32; 3]>) -> Self {
        self.options.tangent = tangent;
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
    pub fn build(self) -> ClayOptions {
        self.options
    }
}

/// What [`clay`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ClayReport {
    /// How many vertices moved (displacement above 0).
    pub moved: usize,
    /// The largest displacement applied, in local units; 0 when nothing moved.
    pub max_displacement: f32,
}

fn check_nonzero_vec(v: [f32; 3], finite_err: &'static str, zero_err: &'static str) -> Result<()> {
    if v.iter().any(|x| !x.is_finite()) {
        return Err(OpsError::InvalidInput(finite_err));
    }
    let len = (f64::from(v[0]).powi(2) + f64::from(v[1]).powi(2) + f64::from(v[2]).powi(2)).sqrt();
    if !(len > 0.0 && len.is_finite()) {
        return Err(OpsError::InvalidInput(zero_err));
    }
    Ok(())
}

fn normalize_vec(v: [f32; 3]) -> Option<[f32; 3]> {
    let len = (f64::from(v[0]).powi(2) + f64::from(v[1]).powi(2) + f64::from(v[2]).powi(2)).sqrt();
    (len > 0.0 && len.is_finite()).then(|| {
        [
            (f64::from(v[0]) / len) as f32,
            (f64::from(v[1]) / len) as f32,
            (f64::from(v[2]) / len) as f32,
        ]
    })
}

fn check_flatten(options: &FlattenOptions) -> Result<()> {
    let brush = &options.brush;
    if !brush.radius.is_finite() || brush.radius <= 0.0 {
        return Err(OpsError::InvalidInput(
            "flatten brush radius must be positive and finite",
        ));
    }
    if brush.center.iter().any(|c| !c.is_finite()) {
        return Err(OpsError::InvalidInput(
            "flatten brush center must be finite",
        ));
    }
    if !brush.strength.is_finite() {
        return Err(OpsError::InvalidInput(
            "flatten brush strength must be finite",
        ));
    }
    if let Some(o) = options.plane_origin
        && o.iter().any(|x| !x.is_finite())
    {
        return Err(OpsError::InvalidInput(
            "flatten plane_origin must be finite",
        ));
    }
    if let Some(n) = options.plane_normal {
        check_nonzero_vec(
            n,
            "flatten plane_normal must be finite",
            "flatten plane_normal must be non-zero",
        )?;
    }
    check_surface_filter(&options.filter)
}

fn check_clay(options: &ClayOptions) -> Result<()> {
    let brush = &options.brush;
    if !brush.radius.is_finite() || brush.radius <= 0.0 {
        return Err(OpsError::InvalidInput(
            "clay brush radius must be positive and finite",
        ));
    }
    if brush.center.iter().any(|c| !c.is_finite()) {
        return Err(OpsError::InvalidInput("clay brush center must be finite"));
    }
    if !brush.strength.is_finite() {
        return Err(OpsError::InvalidInput("clay brush strength must be finite"));
    }
    if let Some(t) = options.tangent {
        check_nonzero_vec(
            t,
            "clay tangent must be finite",
            "clay tangent must be non-zero",
        )?;
    }
    check_surface_filter(&options.filter)
}

fn falloff_weighted_centroid(
    candidates: &[VertexId],
    positions: &[[f32; 3]],
    center: [f32; 3],
    radius: f32,
    falloff: Falloff,
) -> Option<[f32; 3]> {
    let mut sum = [0.0f64; 3];
    let mut w_sum = 0.0f64;
    for &v in candidates {
        let p = &positions[v as usize];
        if let Some(t) = normalized_dist(center, radius, p) {
            let w = f64::from(falloff_value(falloff, t));
            if w > 0.0 {
                sum[0] += w * f64::from(p[0]);
                sum[1] += w * f64::from(p[1]);
                sum[2] += w * f64::from(p[2]);
                w_sum += w;
            }
        }
    }
    (w_sum > 0.0 && w_sum.is_finite()).then(|| {
        [
            (sum[0] / w_sum) as f32,
            (sum[1] / w_sum) as f32,
            (sum[2] / w_sum) as f32,
        ]
    })
}

/// Projects vertices under `options.brush` toward an explicit or auto-fitted
/// brush plane according to [`FlattenOptions::mode`], [`FlattenOptions::filter`],
/// and [`FlattenOptions::symmetry`].
///
/// # Errors
/// - [`OpsError::InvalidInput`] when brush, plane, or filter parameters are invalid.
/// - [`OpsError::Store`] when memory runs out.
///
/// Nothing is written in any of these cases.
pub fn flatten(tx: &mut Transaction, options: &FlattenOptions) -> Result<FlattenReport> {
    check_flatten(options)?;
    if options.brush.strength == 0.0 {
        return Ok(FlattenReport::default());
    }
    let mesh = tx.mesh();
    if mesh.vertex_slots() == 0
        || !has_any_region_member(mesh, &options.brush, options.symmetry, true)
    {
        return Ok(FlattenReport::default());
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
            normalized_dist(reflect_vec(bits, options.brush.center), options.brush.radius, p)
                .is_some()
        });
        if hit {
            push(&mut candidates, v)?;
        }
    }
    if candidates.is_empty() {
        return Ok(FlattenReport::default());
    }

    let need_normals =
        options.plane_normal.is_none() || !matches!(options.filter, SurfaceFilter::None);
    let fans = if need_normals {
        Some(VertexFans::build(mesh)?)
    } else {
        None
    };

    let (ids, moved_to, report) = flatten_candidates(mesh, options, &candidates, |v| {
        fans.as_ref().map_or(&[], |f| f.fan(v))
    })?;
    write_moves(tx, &ids, &moved_to)?;
    Ok(report)
}

/// [`flatten`] over the candidate vertices in `region`, reading only `region`
/// and the mesh's incidence index ([`Mesh::ensure_incidence`]).
///
/// # Errors
/// As for [`flatten`], plus [`OpsError::InvalidInput`] if the mesh lacks a
/// current incidence index.
pub fn flatten_region(
    tx: &mut Transaction,
    options: &FlattenOptions,
    region: &[VertexId],
) -> Result<FlattenReport> {
    check_flatten(options)?;
    let (ids, moved_to, report) = {
        let mesh = tx.mesh();
        let index = region_index(mesh)?;
        if options.brush.strength == 0.0 {
            return Ok(FlattenReport::default());
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
                normalized_dist(reflect_vec(bits, options.brush.center), options.brush.radius, p)
                    .is_some()
            });
            if hit {
                push(&mut candidates, v)?;
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.is_empty() {
            return Ok(FlattenReport::default());
        }

        flatten_candidates(mesh, options, &candidates, |v| index.triangles(v))?
    };
    write_moves(tx, &ids, &moved_to)?;
    Ok(report)
}

fn flatten_candidates<'f>(
    mesh: &Mesh,
    options: &FlattenOptions,
    candidates: &[VertexId],
    triangles_of: impl Fn(VertexId) -> &'f [TriangleId],
) -> Result<(Vec<VertexId>, Vec<[f32; 3]>, FlattenReport)> {
    let positions = mesh.positions();
    let brush = &options.brush;
    let (masks, mask_count) = options.symmetry.active_masks();

    let need_normals =
        options.plane_normal.is_none() || !matches!(options.filter, SurfaceFilter::None);
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

    let need_auto_normal = options.plane_normal.is_none()
        || matches!(options.filter, SurfaceFilter::NormalAngle { normal: None, .. });
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
        let Some(plane_n) = (match options.plane_normal {
            Some(explicit) => normalize_vec(reflect_vec(bits, explicit)),
            None => auto_normal.filter(|n| n[0] != 0.0 || n[1] != 0.0 || n[2] != 0.0),
        }) else {
            continue;
        };
        let Some(plane_o) = (match options.plane_origin {
            Some(explicit) => Some(reflect_vec(bits, explicit)),
            None => falloff_weighted_centroid(
                candidates,
                positions,
                center_m,
                brush.radius,
                brush.falloff,
            ),
        }) else {
            continue;
        };

        for (cand_idx, &v) in candidates.iter().enumerate() {
            let p = positions[v as usize];
            let Some(t) = normalized_dist(center_m, brush.radius, &p) else {
                continue;
            };
            let signed_dist = (p[0] - plane_o[0]) * plane_n[0]
                + (p[1] - plane_o[1]) * plane_n[1]
                + (p[2] - plane_o[2]) * plane_n[2];
            match options.mode {
                FlattenMode::Both => {
                    if signed_dist == 0.0 {
                        continue;
                    }
                }
                FlattenMode::Peaks => {
                    if !(signed_dist > 0.0) {
                        continue;
                    }
                }
                FlattenMode::Valleys => {
                    if !(signed_dist < 0.0) {
                        continue;
                    }
                }
            }
            let f_val = falloff_value(brush.falloff, t);
            let phi = eval_surface_filter(
                &options.filter,
                bits,
                cand_idx,
                &cand_normals,
                &prominences,
                auto_normal,
            );
            let a_m = f_val * phi;
            if !(a_m > 0.0 && a_m.is_finite()) {
                continue;
            }
            let delta = -brush.strength * a_m * signed_dist;
            let dp = [delta * plane_n[0], delta * plane_n[1], delta * plane_n[2]];
            let single_pos = [p[0] + dp[0], p[1] + dp[1], p[2] + dp[2]];
            envelopes[cand_idx].add(a_m, dp, single_pos);
        }
    }

    let mut report = FlattenReport::default();
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

fn clay_reach(brush: &Brush, profile: ClayProfile) -> f32 {
    match profile {
        ClayProfile::Round => brush.radius,
        ClayProfile::Strips => brush.radius * 3.0_f32.sqrt(),
    }
}

fn orthonormal_strip_axes(normal: [f32; 3], raw_tangent: Option<[f32; 3]>) -> ([f32; 3], [f32; 3]) {
    let n = normal.map(f64::from);
    let project = |v: [f64; 3]| -> Option<[f64; 3]> {
        let d = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
        let p = [v[0] - d * n[0], v[1] - d * n[1], v[2] - d * n[2]];
        let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        (len > 1e-12 && len.is_finite()).then_some([p[0] / len, p[1] / len, p[2] / len])
    };

    let t = raw_tangent
        .and_then(|tv| project(tv.map(f64::from)))
        .or_else(|| {
            let fallback = if n[0].abs() < 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            project(fallback)
        })
        .unwrap_or([1.0, 0.0, 0.0]);
    let b_raw = [
        n[1] * t[2] - n[2] * t[1],
        n[2] * t[0] - n[0] * t[2],
        n[0] * t[1] - n[1] * t[0],
    ];
    let b_len = (b_raw[0] * b_raw[0] + b_raw[1] * b_raw[1] + b_raw[2] * b_raw[2]).sqrt();
    let b = if b_len > 0.0 && b_len.is_finite() {
        [b_raw[0] / b_len, b_raw[1] / b_len, b_raw[2] / b_len]
    } else {
        [0.0, 1.0, 0.0]
    };
    (t.map(|x| x as f32), b.map(|x| x as f32))
}

/// Builds up or carves flat-topped clay dabs/strips toward an offset brush plane
/// (`h = 0.25 * R * sign(strength)`).
///
/// For [`ClayProfile::Strips`], the footprint is an oriented `2R x 2R x 2R` box
/// aligned with [`ClayOptions::tangent`], bounded by a sphere of radius
/// `sqrt(3) * R`.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when brush, tangent, or filter parameters are invalid.
/// - [`OpsError::Store`] when memory runs out.
///
/// Nothing is written in any of these cases.
pub fn clay(tx: &mut Transaction, options: &ClayOptions) -> Result<ClayReport> {
    check_clay(options)?;
    if options.brush.strength == 0.0 {
        return Ok(ClayReport::default());
    }
    let reach = clay_reach(&options.brush, options.profile);
    let reach_brush = Brush {
        radius: reach,
        ..options.brush
    };
    let mesh = tx.mesh();
    if mesh.vertex_slots() == 0
        || !has_any_region_member(mesh, &reach_brush, options.symmetry, true)
    {
        return Ok(ClayReport::default());
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
            normalized_dist(reflect_vec(bits, options.brush.center), reach, p).is_some()
        });
        if hit {
            push(&mut candidates, v)?;
        }
    }
    if candidates.is_empty() {
        return Ok(ClayReport::default());
    }

    let fans = VertexFans::build(mesh)?;
    let (ids, moved_to, report) =
        clay_candidates(mesh, options, &candidates, reach, |v| fans.fan(v))?;
    write_moves(tx, &ids, &moved_to)?;
    Ok(report)
}

/// [`clay`] over the candidate vertices in `region`, reading only `region` and
/// the mesh's incidence index ([`Mesh::ensure_incidence`]).
///
/// For [`ClayProfile::Strips`], `region` should cover `sqrt(3) * brush.radius`
/// around each active symmetry brush center.
///
/// # Errors
/// As for [`clay`], plus [`OpsError::InvalidInput`] if the mesh lacks a current
/// incidence index.
pub fn clay_region(
    tx: &mut Transaction,
    options: &ClayOptions,
    region: &[VertexId],
) -> Result<ClayReport> {
    check_clay(options)?;
    let (ids, moved_to, report) = {
        let mesh = tx.mesh();
        let index = region_index(mesh)?;
        if options.brush.strength == 0.0 {
            return Ok(ClayReport::default());
        }
        let reach = clay_reach(&options.brush, options.profile);
        let positions = mesh.positions();
        let (masks, mask_count) = options.symmetry.active_masks();

        let mut candidates: Vec<VertexId> = Vec::new();
        for &v in region {
            if !mesh.is_vertex_alive(v) || mesh.corner_count(v) == 0 {
                continue;
            }
            let p = &positions[v as usize];
            let hit = masks[..mask_count].iter().any(|&bits| {
                normalized_dist(reflect_vec(bits, options.brush.center), reach, p).is_some()
            });
            if hit {
                push(&mut candidates, v)?;
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.is_empty() {
            return Ok(ClayReport::default());
        }

        clay_candidates(mesh, options, &candidates, reach, |v| index.triangles(v))?
    };
    write_moves(tx, &ids, &moved_to)?;
    Ok(report)
}

fn clay_candidates<'f>(
    mesh: &Mesh,
    options: &ClayOptions,
    candidates: &[VertexId],
    reach: f32,
    triangles_of: impl Fn(VertexId) -> &'f [TriangleId],
) -> Result<(Vec<VertexId>, Vec<[f32; 3]>, ClayReport)> {
    let positions = mesh.positions();
    let brush = &options.brush;
    let (masks, mask_count) = options.symmetry.active_masks();

    let mut cand_normals: Vec<[f32; 3]> = Vec::new();
    reserve(&mut cand_normals, candidates.len())?;
    for &v in candidates {
        cand_normals.push(vertex_normal_from_triangles(mesh, triangles_of(v)));
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

    let sign = if brush.strength < 0.0 { -1.0f32 } else { 1.0f32 };
    let offset_h = 0.25 * brush.radius * sign;
    let strength_abs = brush.strength.abs();
    let mut envelopes = filled(candidates.len(), MaxEnvelope::default())?;

    for &bits in &masks[..mask_count] {
        let center_m = reflect_vec(bits, brush.center);
        let mut plane_n = falloff_weighted_normal(
            candidates,
            &cand_normals,
            positions,
            center_m,
            brush.radius,
            brush.falloff,
        );
        if plane_n == [0.0; 3] && reach > brush.radius {
            plane_n = falloff_weighted_normal(
                candidates,
                &cand_normals,
                positions,
                center_m,
                reach,
                brush.falloff,
            );
        }
        if plane_n == [0.0; 3] {
            continue;
        }
        let Some(o_base) = falloff_weighted_centroid(
            candidates,
            positions,
            center_m,
            brush.radius,
            brush.falloff,
        )
        .or_else(|| {
            (reach > brush.radius).then(|| {
                falloff_weighted_centroid(candidates, positions, center_m, reach, brush.falloff)
            })?
        }) else {
            continue;
        };

        let strip_axes = (options.profile == ClayProfile::Strips).then(|| {
            let raw_t = options.tangent.map(|tv| reflect_vec(bits, tv));
            orthonormal_strip_axes(plane_n, raw_t)
        });

        for (cand_idx, &v) in candidates.iter().enumerate() {
            let p = positions[v as usize];
            let t = match strip_axes {
                None => {
                    let Some(t) = normalized_dist(center_m, brush.radius, &p) else {
                        continue;
                    };
                    t
                }
                Some((t_axis, b_axis)) => {
                    let d = [p[0] - center_m[0], p[1] - center_m[1], p[2] - center_m[2]];
                    let u = (d[0] * t_axis[0] + d[1] * t_axis[1] + d[2] * t_axis[2]).abs();
                    let w = (d[0] * b_axis[0] + d[1] * b_axis[1] + d[2] * b_axis[2]).abs();
                    let h = (d[0] * plane_n[0] + d[1] * plane_n[1] + d[2] * plane_n[2]).abs();
                    let t_box = u.max(w).max(h) / brush.radius;
                    if !(t_box.is_finite() && t_box < 1.0) {
                        continue;
                    }
                    t_box
                }
            };

            let d_plane = (p[0] - o_base[0]) * plane_n[0]
                + (p[1] - o_base[1]) * plane_n[1]
                + (p[2] - o_base[2]) * plane_n[2]
                - offset_h;
            if !(d_plane * sign < 0.0) {
                continue;
            }

            let f_val = falloff_value(brush.falloff, t);
            let phi = eval_surface_filter(
                &options.filter,
                bits,
                cand_idx,
                &cand_normals,
                &prominences,
                Some(plane_n),
            );
            let a_m = f_val * phi;
            if !(a_m > 0.0 && a_m.is_finite()) {
                continue;
            }
            let blend = (strength_abs * a_m).min(1.0);
            let delta = -blend * d_plane;
            let dp = [delta * plane_n[0], delta * plane_n[1], delta * plane_n[2]];
            let single_pos = [p[0] + dp[0], p[1] + dp[1], p[2] + dp[2]];
            envelopes[cand_idx].add(a_m, dp, single_pos);
        }
    }

    let mut report = ClayReport::default();
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
