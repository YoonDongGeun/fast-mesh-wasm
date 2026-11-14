//! Crease and pinch brush kernel ([`crease`], [`crease_region`]).

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

/// Options for [`crease`].
///
/// Combines a sharp normal groove/ridge (`brush.strength * f^2`) with radial
/// tangent-plane pinching toward the brush center (`pinch * f`). Setting
/// `brush.strength = 0.0` with `pinch > 0.0` acts as a pure Pinch brush.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CreaseOptions {
    /// The brush sphere, falloff, and signed normal-groove strength (`< 0` carves
    /// a crease, `> 0` raises a sharp ridge, `0.0` pinches without normal offset).
    pub brush: Brush,
    /// Tangent-plane radial pinch factor in `[0.0, 1.0]` (`0.5` by default).
    pub pinch: f32,
    /// Optional surface feature filter.
    pub filter: SurfaceFilter,
    /// Mirror symmetry axes.
    pub symmetry: Symmetry,
}

impl Default for CreaseOptions {
    fn default() -> Self {
        Self {
            brush: Brush::default(),
            pinch: 0.5,
            filter: SurfaceFilter::None,
            symmetry: Symmetry::NONE,
        }
    }
}

impl CreaseOptions {
    /// A builder for [`CreaseOptions`]: default brush, `pinch = 0.5`, no filter,
    /// and no symmetry.
    pub fn builder() -> CreaseOptionsBuilder {
        CreaseOptionsBuilder::default()
    }
}

/// A builder for [`CreaseOptions`], see [`CreaseOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct CreaseOptionsBuilder {
    options: CreaseOptions,
}

impl CreaseOptionsBuilder {
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

    /// Signed normal displacement scale (`< 0` carves groove, `> 0` raises ridge).
    pub fn strength(mut self, strength: f32) -> Self {
        self.options.brush.strength = strength;
        self
    }

    /// Tangent-plane radial pinch factor in `[0.0, 1.0]`.
    pub fn pinch(mut self, pinch: f32) -> Self {
        self.options.pinch = pinch;
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
    pub fn build(self) -> CreaseOptions {
        self.options
    }
}

/// What [`crease`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CreaseReport {
    /// How many vertices moved (displacement above 0).
    pub moved: usize,
    /// The largest displacement applied, in local units; 0 when nothing moved.
    pub max_displacement: f32,
}

fn check_crease(options: &CreaseOptions) -> Result<()> {
    let brush = &options.brush;
    if !brush.radius.is_finite() || brush.radius <= 0.0 {
        return Err(OpsError::InvalidInput(
            "crease brush radius must be positive and finite",
        ));
    }
    if brush.center.iter().any(|c| !c.is_finite()) {
        return Err(OpsError::InvalidInput("crease brush center must be finite"));
    }
    if !brush.strength.is_finite() {
        return Err(OpsError::InvalidInput(
            "crease brush strength must be finite",
        ));
    }
    if !(0.0..=1.0).contains(&options.pinch) {
        return Err(OpsError::InvalidInput(
            "crease pinch must be finite and in [0.0, 1.0]",
        ));
    }
    check_surface_filter(&options.filter)
}

/// Carves or raises a sharp crease under `options.brush` while pinching vertices
/// radially in the tangent plane toward the brush center.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when brush, pinch, or filter parameters are invalid.
/// - [`OpsError::Store`] when memory runs out.
///
/// Nothing is written in any of these cases.
pub fn crease(tx: &mut Transaction, options: &CreaseOptions) -> Result<CreaseReport> {
    check_crease(options)?;
    if options.brush.strength == 0.0 && options.pinch == 0.0 {
        return Ok(CreaseReport::default());
    }
    let mesh = tx.mesh();
    if mesh.vertex_slots() == 0
        || !has_any_region_member(mesh, &options.brush, options.symmetry, true)
    {
        return Ok(CreaseReport::default());
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
        return Ok(CreaseReport::default());
    }

    let fans = VertexFans::build(mesh)?;
    let (ids, moved_to, report) = crease_candidates(mesh, options, &candidates, |v| fans.fan(v))?;
    write_moves(tx, &ids, &moved_to)?;
    Ok(report)
}

/// [`crease`] over the candidate vertices in `region`, reading only `region`
/// and the mesh's incidence index ([`Mesh::ensure_incidence`]).
///
/// # Errors
/// As for [`crease`], plus [`OpsError::InvalidInput`] if the mesh lacks a
/// current incidence index.
pub fn crease_region(
    tx: &mut Transaction,
    options: &CreaseOptions,
    region: &[VertexId],
) -> Result<CreaseReport> {
    check_crease(options)?;
    let (ids, moved_to, report) = {
        let mesh = tx.mesh();
        let index = region_index(mesh)?;
        if options.brush.strength == 0.0 && options.pinch == 0.0 {
            return Ok(CreaseReport::default());
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
            return Ok(CreaseReport::default());
        }

        crease_candidates(mesh, options, &candidates, |v| index.triangles(v))?
    };
    write_moves(tx, &ids, &moved_to)?;
    Ok(report)
}

fn crease_candidates<'f>(
    mesh: &Mesh,
    options: &CreaseOptions,
    candidates: &[VertexId],
    triangles_of: impl Fn(VertexId) -> &'f [TriangleId],
) -> Result<(Vec<VertexId>, Vec<[f32; 3]>, CreaseReport)> {
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

    let need_auto_normal = matches!(options.filter, SurfaceFilter::NormalAngle { normal: None, .. });
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

        for (cand_idx, (&v, &n_v)) in candidates.iter().zip(&cand_normals).enumerate() {
            let p = positions[v as usize];
            let Some(t) = normalized_dist(center_m, brush.radius, &p) else {
                continue;
            };
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

            let normal_scale = brush.strength * f_val * a_m;
            let offset = [
                center_m[0] - p[0],
                center_m[1] - p[1],
                center_m[2] - p[2],
            ];
            let proj = offset[0] * n_v[0] + offset[1] * n_v[1] + offset[2] * n_v[2];
            let tan = [
                offset[0] - proj * n_v[0],
                offset[1] - proj * n_v[1],
                offset[2] - proj * n_v[2],
            ];
            let pinch_scale = options.pinch * a_m;
            let dp = [
                normal_scale * n_v[0] + pinch_scale * tan[0],
                normal_scale * n_v[1] + pinch_scale * tan[1],
                normal_scale * n_v[2] + pinch_scale * tan[2],
            ];
            let single_pos = [p[0] + dp[0], p[1] + dp[1], p[2] + dp[2]];
            envelopes[cand_idx].add(a_m, dp, single_pos);
        }
    }

    let mut report = CreaseReport::default();
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
