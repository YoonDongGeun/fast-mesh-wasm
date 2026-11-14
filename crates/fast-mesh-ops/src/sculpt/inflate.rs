//! Normal-based inflate/deflate brush kernels (`VertexNormal`, `AverageNormal`,
//! `StrokeNormal`) with [`SurfaceFilter`] and [`Symmetry`].

use fast_mesh::compute_vertex_normals;
use fast_mesh_store::{Mesh, Transaction, TriangleId, VertexId};

use super::filter::{
    MaxEnvelope, SurfaceFilter, Symmetry, check_surface_filter, eval_surface_filter,
    falloff_weighted_normal, filter_needs_prominence, reflect_vec, vertex_prominence,
};
use super::stroke::{StrokeSlice, stroke_normal_at};
use super::{
    Brush, Falloff, collect_proper_neighbors, falloff_value, has_any_region_member,
    normalized_dist, region_index, vertex_normal_from_triangles, write_moves,
};
use crate::topology::VertexFans;
use crate::{OpsError, Result, allocation_allowed, filled, push, reserve};

/// Which normal direction [`inflate`] pushes vertices along.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum InflateMethod {
    /// Each vertex moves along its own current smooth normal.
    #[default]
    VertexNormal,
    /// Every vertex in a brush pass translates along the single falloff-weighted
    /// unit average normal of that pass, lifting ridges without radial divergence.
    AverageNormal,
    /// Each vertex moves along its stroke-start smooth normal (from
    /// [`StrokeSlice`] or [`STROKE_NORMAL_CHANNEL`](super::STROKE_NORMAL_CHANNEL),
    /// falling back to its current smooth normal if uncaptured).
    StrokeNormal,
}

/// Options for [`inflate`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct InflateOptions {
    /// The brush sphere, falloff, and signed strength.
    pub brush: Brush,
    /// Normal evaluation mode.
    pub method: InflateMethod,
    /// Optional surface feature filter.
    pub filter: SurfaceFilter,
    /// Mirror symmetry axes.
    pub symmetry: Symmetry,
}

impl InflateOptions {
    /// A builder for [`InflateOptions`]: default brush, `VertexNormal`, no
    /// filter, and no symmetry until set.
    pub fn builder() -> InflateOptionsBuilder {
        InflateOptionsBuilder::default()
    }
}

/// A builder for [`InflateOptions`], see [`InflateOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct InflateOptionsBuilder {
    options: InflateOptions,
}

impl InflateOptionsBuilder {
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

    /// Displacement scale in local units; negative pulls instead of pushing.
    pub fn strength(mut self, strength: f32) -> Self {
        self.options.brush.strength = strength;
        self
    }

    /// Normal evaluation mode.
    pub fn method(mut self, method: InflateMethod) -> Self {
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
    pub fn build(self) -> InflateOptions {
        self.options
    }
}

/// What [`inflate`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct InflateReport {
    /// How many vertices moved (displacement above 0).
    pub moved: usize,
    /// The largest displacement applied, in local units; 0 when nothing moved.
    pub max_displacement: f32,
}

fn check_inflate(options: &InflateOptions) -> Result<()> {
    let brush = &options.brush;
    if !brush.radius.is_finite() || brush.radius <= 0.0 {
        return Err(OpsError::InvalidInput(
            "inflate brush radius must be positive and finite",
        ));
    }
    if brush.center.iter().any(|c| !c.is_finite()) {
        return Err(OpsError::InvalidInput(
            "inflate brush center must be finite",
        ));
    }
    if !brush.strength.is_finite() {
        return Err(OpsError::InvalidInput(
            "inflate brush strength must be finite",
        ));
    }
    check_surface_filter(&options.filter)
}

/// Pushes the vertices under `options.brush` along their normals according to
/// [`InflateOptions::method`], [`InflateOptions::filter`], and
/// [`InflateOptions::symmetry`].
///
/// # Errors
/// - [`OpsError::InvalidInput`] when the brush or filter parameters are invalid.
/// - [`OpsError::Store`] when memory runs out.
///
/// Nothing is written in any of these cases.
pub fn inflate(tx: &mut Transaction, options: &InflateOptions) -> Result<InflateReport> {
    check_inflate(options)?;
    let mesh = tx.mesh();
    if mesh.vertex_slots() == 0
        || !has_any_region_member(mesh, &options.brush, options.symmetry, true)
    {
        return Ok(InflateReport::default());
    }

    let positions = mesh.positions();
    let (masks, mask_count) = options.symmetry.active_masks();

    // Collect ascending candidate vertices across active symmetry spheres.
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
        return Ok(InflateReport::default());
    }

    // Slot-aligned whole-mesh normals via `compute_vertex_normals`, matching `inflate_region`.
    let mut flat: Vec<f32> = Vec::new();
    reserve(&mut flat, positions.len() * 3)?;
    flat.extend(positions.iter().flatten().copied());
    let mut indices: Vec<u32> = Vec::new();
    reserve(&mut indices, mesh.triangle_slots() * 3)?;
    for t in 0..mesh.triangle_slots() as u32 {
        if let Some(tri) = mesh.triangle(t) {
            indices.extend(tri);
        }
    }
    allocation_allowed()?;
    let all_normals = compute_vertex_normals(&flat, &indices)
        .map_err(|_| OpsError::InvalidInput("mesh buffers are not a valid mesh"))?;

    let mut cand_normals: Vec<[f32; 3]> = Vec::new();
    reserve(&mut cand_normals, candidates.len())?;
    for &v in &candidates {
        let base = 3 * v as usize;
        cand_normals.push([
            all_normals[base],
            all_normals[base + 1],
            all_normals[base + 2],
        ]);
    }

    let fans = if filter_needs_prominence(&options.filter) {
        Some(VertexFans::build(mesh)?)
    } else {
        None
    };

    let (ids, moved_to, report) =
        inflate_candidates(mesh, options, &candidates, &cand_normals, None, |v| {
            fans.as_ref().map_or(&[], |f| f.fan(v))
        })?;
    write_moves(tx, &ids, &moved_to)?;
    Ok(report)
}

/// [`inflate`] over the ascending candidate vertices in `region`, reading only
/// `region` and the mesh's incidence index ([`Mesh::ensure_incidence`]).
///
/// When [`InflateOptions::symmetry`] is active, `region` must cover the union
/// of all active reflected brush spheres.
///
/// # Errors
/// As for [`inflate`], plus [`OpsError::InvalidInput`] if the mesh lacks a
/// current incidence index.
pub fn inflate_region(
    tx: &mut Transaction,
    options: &InflateOptions,
    region: &[VertexId],
) -> Result<InflateReport> {
    inflate_region_with_stroke(tx, options, region, None)
}

/// [`inflate_region`] reading stroke-start normals from `stroke` first (when
/// `Some`) before falling back to [`STROKE_NORMAL_CHANNEL`](super::STROKE_NORMAL_CHANNEL)
/// and current smooth normals.
///
/// # Errors
/// As for [`inflate_region`].
pub fn inflate_region_with_stroke(
    tx: &mut Transaction,
    options: &InflateOptions,
    region: &[VertexId],
    stroke: Option<StrokeSlice<'_>>,
) -> Result<InflateReport> {
    check_inflate(options)?;
    let (ids, moved_to, report) = {
        let mesh = tx.mesh();
        let index = region_index(mesh)?;
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
            return Ok(InflateReport::default());
        }

        let mut cand_normals: Vec<[f32; 3]> = Vec::new();
        reserve(&mut cand_normals, candidates.len())?;
        for &v in &candidates {
            cand_normals.push(vertex_normal_from_triangles(mesh, index.triangles(v)));
        }

        inflate_candidates(mesh, options, &candidates, &cand_normals, stroke, |v| {
            index.triangles(v)
        })?
    };
    write_moves(tx, &ids, &moved_to)?;
    Ok(report)
}

fn inflate_candidates<'f>(
    mesh: &Mesh,
    options: &InflateOptions,
    candidates: &[VertexId],
    cand_normals: &[[f32; 3]],
    stroke: Option<StrokeSlice<'_>>,
    triangles_of: impl Fn(VertexId) -> &'f [TriangleId],
) -> Result<(Vec<VertexId>, Vec<[f32; 3]>, InflateReport)> {
    let positions = mesh.positions();
    let brush = &options.brush;
    let (masks, mask_count) = options.symmetry.active_masks();

    let mut prominences: Vec<f32> = Vec::new();
    if filter_needs_prominence(&options.filter) {
        reserve(&mut prominences, candidates.len())?;
        let mut adjacent: Vec<VertexId> = Vec::new();
        for (&v, &n_v) in candidates.iter().zip(cand_normals) {
            collect_proper_neighbors(mesh, v, triangles_of(v), &mut adjacent)?;
            prominences.push(vertex_prominence(
                positions[v as usize],
                n_v,
                &adjacent,
                positions,
            ));
        }
    }

    let need_avg_normal = options.method == InflateMethod::AverageNormal
        || matches!(
            options.filter,
            SurfaceFilter::NormalAngle { normal: None, .. }
        );

    let mut envelopes = filled(candidates.len(), MaxEnvelope::default())?;

    for &bits in &masks[..mask_count] {
        let center_m = reflect_vec(bits, brush.center);
        let avg_normal = need_avg_normal.then(|| {
            falloff_weighted_normal(
                candidates,
                cand_normals,
                positions,
                center_m,
                brush.radius,
                brush.falloff,
            )
        });

        for (idx, (&v, &n_curr)) in candidates.iter().zip(cand_normals).enumerate() {
            let p = positions[v as usize];
            let Some(t) = normalized_dist(center_m, brush.radius, &p) else {
                continue;
            };
            let f_val = falloff_value(brush.falloff, t);
            let phi = eval_surface_filter(
                &options.filter,
                bits,
                idx,
                cand_normals,
                &prominences,
                avg_normal,
            );
            let a_m = f_val * phi;
            if !(a_m > 0.0 && a_m.is_finite()) {
                continue;
            }
            let dir = match options.method {
                InflateMethod::VertexNormal => n_curr,
                InflateMethod::AverageNormal => avg_normal.unwrap_or([0.0; 3]),
                InflateMethod::StrokeNormal => stroke_normal_at(mesh, stroke, v, n_curr),
            };
            let w = brush.strength * a_m;
            let dp = [w * dir[0], w * dir[1], w * dir[2]];
            let single_pos = [p[0] + dp[0], p[1] + dp[1], p[2] + dp[2]];
            envelopes[idx].add(a_m, dp, single_pos);
        }
    }

    let mut report = InflateReport::default();
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
