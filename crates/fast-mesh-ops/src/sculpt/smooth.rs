//! Surface relaxation brushes (`Taubin`, `Laplacian`, `HcLaplacian`,
//! `Cotangent`) with [`SurfaceFilter`] and [`Symmetry`].

use fast_mesh_store::{Mesh, Transaction, TriangleId, VertexId};

use super::filter::{
    MaxEnvelope, SurfaceFilter, Symmetry, check_surface_filter, eval_surface_filter,
    falloff_weighted_normal, filter_needs_normals, filter_needs_prominence, reflect_vec,
    vertex_prominence,
};
use super::{
    Brush, Falloff, clear_and_reserve, collect_proper_neighbors, falloff_value,
    has_any_region_member, normalized_dist, region_index, vertex_normal_from_triangles,
    write_moves,
};
use crate::topology::{VertexFans, cross, dot, proper_triangle, sub};
use crate::{OpsError, Result, filled, push, reserve};

/// The shrinking Taubin step: each pass moves a vertex `w * λ` of the way
/// from its position toward its neighbors' centroid.
pub const TAUBIN_LAMBDA: f32 = 0.33;

/// The inflating Taubin step: each pass moves a vertex `w * μ` of the way
/// from its position toward its neighbors' centroid, pushing it back out past
/// where the λ pass left it.
pub const TAUBIN_MU: f32 = -0.34;

/// Vollmer–Mencl–Müller HC-Laplacian blend toward pre-dab rest positions (`α`).
pub const HC_ALPHA: f32 = 0.1;

/// Vollmer–Mencl–Müller HC-Laplacian self vs. neighbor pushback balance (`β`).
pub const HC_BETA: f32 = 0.5;

const COT_WEIGHT_MIN: f64 = 1e-4;
const COT_WEIGHT_MAX: f64 = 1e3;
const COT_DEGENERATE_SIN: f64 = 1e-12;

/// Relaxation kernel for [`smooth`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SmoothMethod {
    /// Volume-preserving Taubin λ/μ cycles (`0.33` / `-0.34`, the default).
    #[default]
    Taubin,
    /// Uniform Laplacian umbrella relaxation (`1.0`).
    Laplacian,
    /// Vollmer–Mencl–Müller HC-Laplacian (`α = 0.1`, `β = 0.5`) pushing back
    /// toward pre-dab positions to resist shrinkage on thin walls and spheres.
    HcLaplacian,
    /// Discrete cotangent-weighted Laplacian (edge weights clamped to
    /// `[1e-4, 1e3]`), eliminating tangential drift on uneven triangulations.
    Cotangent,
}

/// Options for [`smooth`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SmoothOptions {
    /// The brush to apply.
    pub brush: Brush,
    /// How many cycles to run; 0 moves nothing.
    pub cycles: u32,
    /// How each cycle relaxes.
    pub method: SmoothMethod,
    /// Optional surface feature filter.
    pub filter: SurfaceFilter,
    /// Mirror symmetry axes.
    pub symmetry: Symmetry,
}

impl SmoothOptions {
    /// A builder for [`SmoothOptions`]: default brush, 0 cycles, `Taubin`, no
    /// filter, and no symmetry until set.
    pub fn builder() -> SmoothOptionsBuilder {
        SmoothOptionsBuilder::default()
    }
}

/// A builder for [`SmoothOptions`], see [`SmoothOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct SmoothOptionsBuilder {
    options: SmoothOptions,
}

impl SmoothOptionsBuilder {
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

    /// Displacement scale; the per-vertex weight is `strength * falloff * filter`.
    pub fn strength(mut self, strength: f32) -> Self {
        self.options.brush.strength = strength;
        self
    }

    /// How many cycles to run.
    pub fn cycles(mut self, cycles: u32) -> Self {
        self.options.cycles = cycles;
        self
    }

    /// How each cycle relaxes (`Taubin` by default).
    pub fn method(mut self, method: SmoothMethod) -> Self {
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
    pub fn build(self) -> SmoothOptions {
        self.options
    }
}

/// What [`smooth`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SmoothReport {
    /// How many vertices moved (displacement above 0).
    pub moved: usize,
    /// The largest displacement applied, in local units; 0 when nothing moved.
    pub max_displacement: f32,
}

fn check_smooth(options: &SmoothOptions) -> Result<()> {
    let brush = &options.brush;
    if !brush.radius.is_finite() || brush.radius <= 0.0 {
        return Err(OpsError::InvalidInput(
            "smooth brush radius must be positive and finite",
        ));
    }
    if brush.center.iter().any(|c| !c.is_finite()) {
        return Err(OpsError::InvalidInput("smooth brush center must be finite"));
    }
    if !brush.strength.is_finite() {
        return Err(OpsError::InvalidInput(
            "smooth brush strength must be finite",
        ));
    }
    check_surface_filter(&options.filter)
}

/// Smooths the vertices under `options.brush` according to
/// [`SmoothOptions::method`], [`SmoothOptions::filter`], and
/// [`SmoothOptions::symmetry`].
///
/// # Errors
/// - [`OpsError::InvalidInput`] when the brush or filter parameters are invalid.
/// - [`OpsError::Store`] when memory runs out.
///
/// Nothing is written in any of these cases.
pub fn smooth(tx: &mut Transaction, options: &SmoothOptions) -> Result<SmoothReport> {
    check_smooth(options)?;
    if options.cycles == 0 {
        return Ok(SmoothReport::default());
    }
    let mesh = tx.mesh();
    if mesh.vertex_slots() == 0
        || !has_any_region_member(mesh, &options.brush, options.symmetry, false)
    {
        return Ok(SmoothReport::default());
    }

    let fans = VertexFans::build(mesh)?;
    let positions = mesh.positions();
    let (masks, mask_count) = options.symmetry.active_masks();

    let mut candidates: Vec<VertexId> = Vec::new();
    for (v, p) in positions.iter().enumerate() {
        let v = v as VertexId;
        if !mesh.is_vertex_alive(v) {
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
        return Ok(SmoothReport::default());
    }

    let (ids, moved_to, report) = smooth_candidates(mesh, options, &candidates, |v| fans.fan(v))?;
    write_moves(tx, &ids, &moved_to)?;
    Ok(report)
}

/// [`smooth`] over the ascending candidate vertices in `region`, reading only
/// `region` and the mesh's incidence index ([`Mesh::ensure_incidence`]).
///
/// When [`SmoothOptions::symmetry`] is active, `region` must cover the union
/// of all active reflected brush spheres.
///
/// # Errors
/// As for [`smooth`], plus [`OpsError::InvalidInput`] if the mesh lacks a
/// current incidence index.
pub fn smooth_region(
    tx: &mut Transaction,
    options: &SmoothOptions,
    region: &[VertexId],
) -> Result<SmoothReport> {
    check_smooth(options)?;
    if options.cycles == 0 {
        return Ok(SmoothReport::default());
    }
    let (ids, moved_to, report) = {
        let mesh = tx.mesh();
        let index = region_index(mesh)?;
        let positions = mesh.positions();
        let (masks, mask_count) = options.symmetry.active_masks();

        let mut candidates: Vec<VertexId> = Vec::new();
        for &v in region {
            if !mesh.is_vertex_alive(v) {
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
            return Ok(SmoothReport::default());
        }

        smooth_candidates(mesh, options, &candidates, |v| index.triangles(v))?
    };
    write_moves(tx, &ids, &moved_to)?;
    Ok(report)
}

/// A neighbor in a pass's CSR 1-ring: either another moving member of this pass
/// or a fixed vertex outside the pass.
#[derive(Clone, Copy)]
enum Source {
    Member(u32),
    Fixed([f32; 3]),
}

/// Half-cotangent `0.5 * cot(angle at pivot)` between `(a - pivot)` and `(b - pivot)`.
fn half_cotangent_at(pivot: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f64 {
    let p0 = pivot.map(f64::from);
    let u = sub(a.map(f64::from), p0);
    let w = sub(b.map(f64::from), p0);
    let c = cross(u, w);
    let sin = dot(c, c).sqrt();
    let prod = (dot(u, u) * dot(w, w)).sqrt();
    if sin > COT_DEGENERATE_SIN * prod {
        0.5 * dot(u, w) / sin
    } else {
        0.0
    }
}

fn compute_cotangent_row(
    mesh: &Mesh,
    v: VertexId,
    triangles: &[TriangleId],
    adjacent: &[VertexId],
    positions: &[[f32; 3]],
    raw_scratch: &mut Vec<f64>,
    out_weights: &mut Vec<f32>,
) -> Result<()> {
    let raw_cap = clear_and_reserve(raw_scratch, adjacent.len())?;
    raw_scratch.resize(adjacent.len(), 0.0);
    debug_assert_eq!(raw_scratch.capacity(), raw_cap);

    let pv = positions[v as usize];
    for &t in triangles {
        let Some([a, b, c]) = proper_triangle(mesh, t) else {
            continue;
        };
        let (u1, u2) = if a == v {
            (b, c)
        } else if b == v {
            (c, a)
        } else if c == v {
            (a, b)
        } else {
            continue;
        };
        let pu1 = positions[u1 as usize];
        let pu2 = positions[u2 as usize];
        if let Ok(i1) = adjacent.binary_search(&u1) {
            raw_scratch[i1] += half_cotangent_at(pu2, pv, pu1);
        }
        if let Ok(i2) = adjacent.binary_search(&u2) {
            raw_scratch[i2] += half_cotangent_at(pu1, pv, pu2);
        }
    }

    reserve(out_weights, adjacent.len())?;
    for &w in raw_scratch.iter() {
        let clamped = if w.is_finite() {
            w.clamp(COT_WEIGHT_MIN, COT_WEIGHT_MAX) as f32
        } else {
            COT_WEIGHT_MIN as f32
        };
        out_weights.push(clamped);
    }
    Ok(())
}

fn smooth_candidates<'f>(
    mesh: &Mesh,
    options: &SmoothOptions,
    candidates: &[VertexId],
    triangles_of: impl Fn(VertexId) -> &'f [TriangleId],
) -> Result<(Vec<VertexId>, Vec<[f32; 3]>, SmoothReport)> {
    let positions = mesh.positions();
    let brush = &options.brush;
    let (masks, mask_count) = options.symmetry.active_masks();

    let mut cand_normals: Vec<[f32; 3]> = Vec::new();
    if filter_needs_normals(&options.filter) {
        reserve(&mut cand_normals, candidates.len())?;
        for &v in candidates {
            cand_normals.push(vertex_normal_from_triangles(mesh, triangles_of(v)));
        }
    }

    let mut prominences: Vec<f32> = Vec::new();
    let mut adjacent: Vec<VertexId> = Vec::new();
    if filter_needs_prominence(&options.filter) {
        reserve(&mut prominences, candidates.len())?;
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

    let need_auto_normal = matches!(
        options.filter,
        SurfaceFilter::NormalAngle { normal: None, .. }
    );
    let mut envelopes = filled(candidates.len(), MaxEnvelope::default())?;

    // Reusable per-pass scratch buffers.
    let mut pass_members: Vec<(usize, VertexId, f32)> = Vec::new();
    let mut starts: Vec<u32> = Vec::new();
    let mut sources: Vec<Source> = Vec::new();
    let mut cot_weights: Vec<f32> = Vec::new();
    let mut raw_cot_scratch: Vec<f64> = Vec::new();
    let mut current: Vec<[f32; 3]> = Vec::new();
    let mut before: Vec<[f32; 3]> = Vec::new();
    let mut hc_lap: Vec<[f32; 3]> = Vec::new();
    let mut hc_b: Vec<[f32; 3]> = Vec::new();

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

        pass_members.clear();
        for (cand_idx, &v) in candidates.iter().enumerate() {
            let p = &positions[v as usize];
            let Some(t) = normalized_dist(center_m, brush.radius, p) else {
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
            if a_m > 0.0 && a_m.is_finite() {
                push(&mut pass_members, (cand_idx, v, a_m))?;
            }
        }
        if pass_members.is_empty() {
            continue;
        }

        let m_len = pass_members.len();
        let starts_cap = clear_and_reserve(&mut starts, m_len + 1)?;
        starts.push(0);
        sources.clear();
        cot_weights.clear();

        for &(_, v, _) in &pass_members {
            let tris = triangles_of(v);
            collect_proper_neighbors(mesh, v, tris, &mut adjacent)?;
            for &u in &adjacent {
                let source = match pass_members.binary_search_by_key(&u, |m| m.1) {
                    Ok(k) => Source::Member(k as u32),
                    Err(_) => Source::Fixed(positions[u as usize]),
                };
                push(&mut sources, source)?;
            }
            if options.method == SmoothMethod::Cotangent && !adjacent.is_empty() {
                compute_cotangent_row(
                    mesh,
                    v,
                    tris,
                    &adjacent,
                    positions,
                    &mut raw_cot_scratch,
                    &mut cot_weights,
                )?;
            }
            starts.push(sources.len() as u32);
        }
        debug_assert_eq!(starts.capacity(), starts_cap);

        let cur_cap = clear_and_reserve(&mut current, m_len)?;
        current.extend(pass_members.iter().map(|&(_, v, _)| positions[v as usize]));
        debug_assert_eq!(current.capacity(), cur_cap);

        let bef_cap = clear_and_reserve(&mut before, m_len)?;

        match options.method {
            SmoothMethod::Taubin | SmoothMethod::Laplacian => {
                let sub_passes: &[f32] = match options.method {
                    SmoothMethod::Taubin => &[TAUBIN_LAMBDA, TAUBIN_MU],
                    SmoothMethod::Laplacian => &[1.0],
                    _ => unreachable!(),
                };
                for _ in 0..options.cycles {
                    for &k in sub_passes {
                        before.clear();
                        before.extend_from_slice(&current);
                        debug_assert_eq!(before.capacity(), bef_cap);
                        for (m, &(_, _, a_m)) in pass_members.iter().enumerate() {
                            let row = &sources[starts[m] as usize..starts[m + 1] as usize];
                            if row.is_empty() {
                                continue;
                            }
                            let mut centroid = [0.0f32; 3];
                            for source in row {
                                let q = match *source {
                                    Source::Member(u) => before[u as usize],
                                    Source::Fixed(q) => q,
                                };
                                for axis in 0..3 {
                                    centroid[axis] += q[axis];
                                }
                            }
                            let inv = row.len() as f32;
                            for c in &mut centroid {
                                *c /= inv;
                            }
                            let w = brush.strength * a_m;
                            let p = before[m];
                            for axis in 0..3 {
                                current[m][axis] = p[axis] + w * k * (centroid[axis] - p[axis]);
                            }
                        }
                    }
                }
            }
            SmoothMethod::HcLaplacian => {
                let lap_cap = clear_and_reserve(&mut hc_lap, m_len)?;
                hc_lap.resize(m_len, [0.0; 3]);
                debug_assert_eq!(hc_lap.capacity(), lap_cap);

                let b_cap = clear_and_reserve(&mut hc_b, m_len)?;
                hc_b.resize(m_len, [0.0; 3]);
                debug_assert_eq!(hc_b.capacity(), b_cap);

                for _ in 0..options.cycles {
                    before.clear();
                    before.extend_from_slice(&current);
                    debug_assert_eq!(before.capacity(), bef_cap);
                    for (m, &(_, v, _)) in pass_members.iter().enumerate() {
                        let row = &sources[starts[m] as usize..starts[m + 1] as usize];
                        let q_m = before[m];
                        if row.is_empty() {
                            hc_lap[m] = q_m;
                            hc_b[m] = [0.0; 3];
                            continue;
                        }
                        let mut centroid = [0.0f32; 3];
                        for source in row {
                            let q = match *source {
                                Source::Member(u) => before[u as usize],
                                Source::Fixed(q) => q,
                            };
                            for axis in 0..3 {
                                centroid[axis] += q[axis];
                            }
                        }
                        let inv = row.len() as f32;
                        for c in &mut centroid {
                            *c /= inv;
                        }
                        hc_lap[m] = centroid;
                        let o_m = positions[v as usize];
                        for axis in 0..3 {
                            hc_b[m][axis] = centroid[axis]
                                - (HC_ALPHA * o_m[axis] + (1.0 - HC_ALPHA) * q_m[axis]);
                        }
                    }
                    for (m, &(_, _, a_m)) in pass_members.iter().enumerate() {
                        let row = &sources[starts[m] as usize..starts[m + 1] as usize];
                        if row.is_empty() {
                            continue;
                        }
                        let mut b_bar = [0.0f32; 3];
                        for source in row {
                            if let Source::Member(u) = *source {
                                let bu = hc_b[u as usize];
                                for axis in 0..3 {
                                    b_bar[axis] += bu[axis];
                                }
                            }
                        }
                        let inv = row.len() as f32;
                        for b in &mut b_bar {
                            *b /= inv;
                        }
                        let w = brush.strength * a_m;
                        let q_m = before[m];
                        let lap_m = hc_lap[m];
                        let bm = hc_b[m];
                        for axis in 0..3 {
                            let hc_target =
                                lap_m[axis] - (HC_BETA * bm[axis] + (1.0 - HC_BETA) * b_bar[axis]);
                            current[m][axis] = q_m[axis] + w * (hc_target - q_m[axis]);
                        }
                    }
                }
            }
            SmoothMethod::Cotangent => {
                for _ in 0..options.cycles {
                    before.clear();
                    before.extend_from_slice(&current);
                    debug_assert_eq!(before.capacity(), bef_cap);
                    for (m, &(_, _, a_m)) in pass_members.iter().enumerate() {
                        let lo = starts[m] as usize;
                        let hi = starts[m + 1] as usize;
                        let row = &sources[lo..hi];
                        if row.is_empty() {
                            continue;
                        }
                        let w_row = &cot_weights[lo..hi];
                        let mut sum = [0.0f32; 3];
                        let mut w_total = 0.0f32;
                        for (source, &ew) in row.iter().zip(w_row) {
                            let q = match *source {
                                Source::Member(u) => before[u as usize],
                                Source::Fixed(q) => q,
                            };
                            sum[0] += ew * q[0];
                            sum[1] += ew * q[1];
                            sum[2] += ew * q[2];
                            w_total += ew;
                        }
                        if !(w_total > 0.0 && w_total.is_finite()) {
                            continue;
                        }
                        let centroid = [sum[0] / w_total, sum[1] / w_total, sum[2] / w_total];
                        let w = brush.strength * a_m;
                        let p = before[m];
                        for axis in 0..3 {
                            current[m][axis] = p[axis] + w * (centroid[axis] - p[axis]);
                        }
                    }
                }
            }
        }

        for (m, &(cand_idx, v, a_m)) in pass_members.iter().enumerate() {
            let p0 = positions[v as usize];
            let p1 = current[m];
            let dp = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            envelopes[cand_idx].add(a_m, dp, p1);
        }
    }

    let mut report = SmoothReport::default();
    let mut ids: Vec<VertexId> = Vec::new();
    let mut moved_to: Vec<[f32; 3]> = Vec::new();
    for (&v, env) in candidates.iter().zip(&envelopes) {
        let p0 = positions[v as usize];
        if let Some((displaced, length)) = env.resolve(p0) {
            push(&mut ids, v)?;
            push(&mut moved_to, displaced)?;
            report.max_displacement = report.max_displacement.max(length);
        }
    }
    report.moved = ids.len();
    Ok((ids, moved_to, report))
}
