//! Multi-algorithm grab brush kernels (`Sphere`, `Geodesic`, `Kelvinlet`),
//! fold-safe `adaptive_reach`, and stroke-anchored snap-back (`from_stroke`).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use fast_mesh_store::{Mesh, StoreError, Transaction, TriangleId, VertexId};

use super::filter::{
    MaxEnvelope, SurfaceFilter, Symmetry, check_surface_filter, eval_surface_filter,
    falloff_weighted_normal, filter_needs_normals, filter_needs_prominence, reflect_vec,
    vertex_prominence,
};
use super::stroke::{StrokeSlice, stroke_position_at};
use super::{
    Brush, Falloff, clear_and_reserve, collect_proper_neighbors, falloff_value, normalized_dist,
    region_index, vertex_normal_from_triangles, write_moves,
};
use crate::topology::{VertexFans, cross, dot, proper_triangle, sub};
use crate::{OpsError, Result, allocation_allowed, filled, push, reserve};

/// Maximum surface stretch slope along a tangential drag before [`GrabOptions::adaptive_reach`]
/// widens the brush radius (`0.8 < 1.0`, guaranteeing no surface fold-over).
pub const GRAB_MAX_STRETCH: f32 = 0.8;

/// Normalized regularization core scale (`epsilon / R_eff`) for [`GrabMethod::Kelvinlet`].
const KELVINLET_EPSILON_RATIO: f64 = 0.35;

/// Spatial kernel for [`grab`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GrabMethod {
    /// Euclidean 3D sphere distance from the brush center.
    #[default]
    Sphere,
    /// Surface-connected Dijkstra geodesic distance from the nearest triangle
    /// inside the brush sphere, bridging coincident UV seams at zero cost while
    /// isolating disconnected sheets across air gaps.
    Geodesic,
    /// Calibrated regularized linear-elastic Kelvinlet (de Goes & James 2017)
    /// windowed smoothly to compact support at `R_eff`.
    Kelvinlet,
}

/// Options for [`grab`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrabOptions {
    /// The brush sphere, falloff, and signed strength.
    pub brush: Brush,
    /// Drag vector in local coordinates.
    pub direction: [f32; 3],
    /// Distance or elasticity kernel (`Sphere` by default).
    pub method: GrabMethod,
    /// Elastic Poisson ratio in `[0.0, 0.5]` for [`GrabMethod::Kelvinlet`] (`0.4` by default).
    pub poisson: f32,
    /// Evaluate offsets from stroke-start rest positions (`STROKE_POS_CHANNEL` /
    /// [`StrokeSlice`]), snapping previously moved vertices back to rest if a
    /// shortened drag leaves them outside the current reach.
    pub from_stroke: bool,
    /// Automatically widen the effective radius on tangential surface drags so
    /// the surface stretches by at most [`GRAB_MAX_STRETCH`] without folding.
    pub adaptive_reach: bool,
    /// Optional surface feature filter.
    pub filter: SurfaceFilter,
    /// Mirror symmetry axes.
    pub symmetry: Symmetry,
}

impl Default for GrabOptions {
    fn default() -> Self {
        Self {
            brush: Brush::default(),
            direction: [0.0; 3],
            method: GrabMethod::Sphere,
            poisson: 0.4,
            from_stroke: false,
            adaptive_reach: false,
            filter: SurfaceFilter::None,
            symmetry: Symmetry::NONE,
        }
    }
}

impl GrabOptions {
    /// A builder for [`GrabOptions`]: default brush, zero direction, `Sphere`,
    /// `poisson = 0.4`, `from_stroke = false`, `adaptive_reach = false`.
    pub fn builder() -> GrabOptionsBuilder {
        GrabOptionsBuilder::default()
    }
}

/// A builder for [`GrabOptions`], see [`GrabOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct GrabOptionsBuilder {
    options: GrabOptions,
}

impl GrabOptionsBuilder {
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

    /// Displacement scale in local units.
    pub fn strength(mut self, strength: f32) -> Self {
        self.options.brush.strength = strength;
        self
    }

    /// The drag vector in local coordinates.
    pub fn direction(mut self, direction: [f32; 3]) -> Self {
        self.options.direction = direction;
        self
    }

    /// Distance or elasticity kernel.
    pub fn method(mut self, method: GrabMethod) -> Self {
        self.options.method = method;
        self
    }

    /// Elastic Poisson ratio in `[0.0, 0.5]` for [`GrabMethod::Kelvinlet`].
    pub fn poisson(mut self, poisson: f32) -> Self {
        self.options.poisson = poisson;
        self
    }

    /// Whether to anchor displacements to stroke-start rest positions.
    pub fn from_stroke(mut self, from_stroke: bool) -> Self {
        self.options.from_stroke = from_stroke;
        self
    }

    /// Whether to widen reach on tangential drags to prevent folding.
    pub fn adaptive_reach(mut self, adaptive_reach: bool) -> Self {
        self.options.adaptive_reach = adaptive_reach;
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
    pub fn build(self) -> GrabOptions {
        self.options
    }
}

/// What [`grab`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GrabReport {
    /// How many vertices moved (displacement above 0).
    pub moved: usize,
    /// The largest displacement applied, in local units; 0 when nothing moved.
    pub max_displacement: f32,
}

fn grab_slope(falloff: Falloff, method: GrabMethod) -> f32 {
    match method {
        GrabMethod::Kelvinlet => 1.75,
        GrabMethod::Sphere | GrabMethod::Geodesic => match falloff {
            Falloff::Linear => 1.0,
            Falloff::Smooth => 1.5,
        },
    }
}

/// Upper bound on the effective brush radius [`GrabOptions::adaptive_reach`] can
/// choose for `(brush, direction, method)`:
/// `(|brush.strength| * ||direction|| * slope / GRAB_MAX_STRETCH).max(brush.radius)`.
pub fn grab_widest_reach(brush: &Brush, direction: [f32; 3], method: GrabMethod) -> f32 {
    let dir_len = (direction[0] * direction[0]
        + direction[1] * direction[1]
        + direction[2] * direction[2])
        .sqrt();
    let length = brush.strength.abs() * dir_len;
    let slope = grab_slope(brush.falloff, method);
    let wide = length * slope / GRAB_MAX_STRETCH;
    if wide.is_finite() {
        wide.max(brush.radius)
    } else {
        brush.radius
    }
}

fn check_grab(options: &GrabOptions) -> Result<()> {
    let brush = &options.brush;
    if !brush.radius.is_finite() || brush.radius <= 0.0 {
        return Err(OpsError::InvalidInput(
            "grab brush radius must be positive and finite",
        ));
    }
    if brush.center.iter().any(|c| !c.is_finite()) {
        return Err(OpsError::InvalidInput("grab brush center must be finite"));
    }
    if !brush.strength.is_finite() {
        return Err(OpsError::InvalidInput("grab brush strength must be finite"));
    }
    if options.direction.iter().any(|c| !c.is_finite()) {
        return Err(OpsError::InvalidInput("grab direction must be finite"));
    }
    if !(0.0..=0.5).contains(&options.poisson) {
        return Err(OpsError::InvalidInput(
            "grab kelvinlet poisson ratio must be finite and in [0.0, 0.5]",
        ));
    }
    check_surface_filter(&options.filter)
}

fn same_bits(a: &[f32; 3], b: &[f32; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn base_pos_of(
    mesh: &Mesh,
    from_stroke: bool,
    stroke: Option<StrokeSlice<'_>>,
    v: VertexId,
) -> [f32; 3] {
    if from_stroke {
        stroke_position_at(mesh, stroke, v)
    } else {
        mesh.positions()[v as usize]
    }
}

/// Drags the vertices under `options.brush` along `options.direction` according
/// to [`GrabOptions::method`], [`GrabOptions::filter`], and
/// [`GrabOptions::symmetry`].
///
/// # Errors
/// - [`OpsError::InvalidInput`] when brush, direction, Poisson ratio, or filter
///   parameters are invalid.
/// - [`OpsError::Store`] when memory runs out.
///
/// Nothing is written in any of these cases.
pub fn grab(tx: &mut Transaction, options: &GrabOptions) -> Result<GrabReport> {
    check_grab(options)?;
    let mesh = tx.mesh();
    if mesh.vertex_slots() == 0 {
        return Ok(GrabReport::default());
    }
    if !options.from_stroke
        && (options.brush.strength == 0.0 || options.direction == [0.0, 0.0, 0.0])
    {
        return Ok(GrabReport::default());
    }

    let positions = mesh.positions();
    let (masks, mask_count) = options.symmetry.active_masks();
    let max_reach = if options.adaptive_reach {
        grab_widest_reach(&options.brush, options.direction, options.method)
    } else {
        options.brush.radius
    };

    // Zero-allocation pre-scan so empty regions never allocate temporaries.
    let any_work = (0..mesh.vertex_slots() as VertexId).any(|v| {
        if !mesh.is_vertex_alive(v) {
            return false;
        }
        let curr = positions[v as usize];
        let base = base_pos_of(mesh, options.from_stroke, None, v);
        if options.from_stroke && !same_bits(&curr, &base) {
            return true;
        }
        masks[..mask_count]
            .iter()
            .any(|&bits| normalized_dist(reflect_vec(bits, options.brush.center), max_reach, &base).is_some())
    });
    if !any_work {
        return Ok(GrabReport::default());
    }

    let mut candidates: Vec<VertexId> = Vec::new();
    for v in 0..mesh.vertex_slots() as VertexId {
        if !mesh.is_vertex_alive(v) {
            continue;
        }
        let curr = positions[v as usize];
        let base = base_pos_of(mesh, options.from_stroke, None, v);
        let hit = (options.from_stroke && !same_bits(&curr, &base))
            || masks[..mask_count].iter().any(|&bits| {
                normalized_dist(reflect_vec(bits, options.brush.center), max_reach, &base).is_some()
            });
        if hit {
            push(&mut candidates, v)?;
        }
    }
    if candidates.is_empty() {
        return Ok(GrabReport::default());
    }

    let need_fans = options.method == GrabMethod::Geodesic || filter_needs_normals(&options.filter);
    let fans = if need_fans {
        Some(VertexFans::build(mesh)?)
    } else {
        None
    };

    let (ids, moved_to, report) = grab_candidates(
        mesh,
        options,
        &candidates,
        None,
        &[],
        |v| fans.as_ref().map_or(&[], |f| f.fan(v)),
    )?;
    write_moves(tx, &ids, &moved_to)?;
    Ok(report)
}

/// [`grab`] over the candidate vertices in `region`.
///
/// `Sphere` and `Kelvinlet` with [`SurfaceFilter::None`] need no incidence
/// index; `Geodesic` and active surface filters read [`Mesh::ensure_incidence`].
///
/// # Errors
/// As for [`grab`], plus [`OpsError::InvalidInput`] if an incidence index is
/// needed and missing.
pub fn grab_region(
    tx: &mut Transaction,
    options: &GrabOptions,
    region: &[VertexId],
) -> Result<GrabReport> {
    grab_region_with_stroke(tx, options, region, None, &[])
}

/// [`grab_region`] with explicit [`StrokeSlice`] rest state and optional
/// `zero_protected` fully masked vertices excluded from `adaptive_reach`
/// directional `ahead` gating.
///
/// # Errors
/// As for [`grab_region`].
pub fn grab_region_with_stroke(
    tx: &mut Transaction,
    options: &GrabOptions,
    region: &[VertexId],
    stroke: Option<StrokeSlice<'_>>,
    zero_protected: &[VertexId],
) -> Result<GrabReport> {
    check_grab(options)?;
    let (ids, moved_to, report) = {
        let mesh = tx.mesh();
        let need_index =
            options.method == GrabMethod::Geodesic || filter_needs_normals(&options.filter);
        let index = if need_index {
            Some(region_index(mesh)?)
        } else {
            None
        };
        if !options.from_stroke
            && (options.brush.strength == 0.0 || options.direction == [0.0, 0.0, 0.0])
        {
            return Ok(GrabReport::default());
        }

        let positions = mesh.positions();
        let (masks, mask_count) = options.symmetry.active_masks();
        let max_reach = if options.adaptive_reach {
            grab_widest_reach(&options.brush, options.direction, options.method)
        } else {
            options.brush.radius
        };

        let mut candidates: Vec<VertexId> = Vec::new();
        for &v in region {
            if !mesh.is_vertex_alive(v) {
                continue;
            }
            let curr = positions[v as usize];
            let base = base_pos_of(mesh, options.from_stroke, stroke, v);
            let hit = (options.from_stroke && !same_bits(&curr, &base))
                || masks[..mask_count].iter().any(|&bits| {
                    normalized_dist(reflect_vec(bits, options.brush.center), max_reach, &base)
                        .is_some()
                });
            if hit {
                push(&mut candidates, v)?;
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.is_empty() {
            return Ok(GrabReport::default());
        }

        grab_candidates(
            mesh,
            options,
            &candidates,
            stroke,
            zero_protected,
            |v| index.map_or(&[], |idx| idx.triangles(v)),
        )?
    };
    write_moves(tx, &ids, &moved_to)?;
    Ok(report)
}

#[derive(Clone, Copy)]
struct GeodesicEntry {
    dist: f64,
    idx: u32,
}

impl PartialEq for GeodesicEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist.to_bits() == other.dist.to_bits() && self.idx == other.idx
    }
}

impl Eq for GeodesicEntry {}

impl PartialOrd for GeodesicEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GeodesicEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.idx.cmp(&self.idx))
    }
}

fn push_heap<T: Ord>(heap: &mut BinaryHeap<T>, item: T) -> Result<()> {
    allocation_allowed()?;
    heap.try_reserve(1)
        .map_err(|_| OpsError::Store(StoreError::OutOfMemory))?;
    heap.push(item);
    Ok(())
}

/// Closest point on triangle `(a, b, c)` to `p` in `f64` (Ericson §5.1.5).
fn closest_point_on_triangle(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]];
    }

    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]];
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let bc = sub(c, b);
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return [b[0] + w * bc[0], b[1] + w * bc[1], b[2] + w * bc[2]];
    }

    let sum = va + vb + vc;
    if sum > 0.0 && sum.is_finite() {
        let denom = 1.0 / sum;
        let v = vb * denom;
        let w = vc * denom;
        [
            a[0] + ab[0] * v + ac[0] * w,
            a[1] + ab[1] * v + ac[1] * w,
            a[2] + ab[2] * v + ac[2] * w,
        ]
    } else {
        a
    }
}

/// Evaluates the calibrated regularized Grab Kelvinlet (`u(0) = g`, windowed by `(1 - t^2)^2`).
/// Returns `(window_weight, dp)`.
fn eval_grab_kelvinlet(
    offset: [f32; 3],
    r_eff: f32,
    t: f32,
    g: [f32; 3],
    poisson: f32,
    phi: f32,
) -> (f32, [f32; 3]) {
    let win = (1.0 - t * t).powi(2);
    let a_m = win * phi;
    if !(a_m > 0.0 && a_m.is_finite()) {
        return (0.0, [0.0; 3]);
    }
    let nu = f64::from(poisson);
    let beta = 1.0 / (4.0 * (1.0 - nu));
    let e = KELVINLET_EPSILON_RATIO;
    let tf = f64::from(t);
    let rho2 = tf * tf + e * e;
    let rho = rho2.sqrt();
    let rho3 = rho2 * rho;
    let scale = e / (1.5 - beta);

    let iso = scale * ((1.0 - beta) / rho + (e * e) / (2.0 * rho3));
    let r_eff_sq = f64::from(r_eff) * f64::from(r_eff);
    let x = offset.map(f64::from);
    let gf = g.map(f64::from);
    let x_dot_g = x[0] * gf[0] + x[1] * gf[1] + x[2] * gf[2];
    let aniso = if r_eff_sq > 0.0 {
        scale * beta * x_dot_g / (rho3 * r_eff_sq)
    } else {
        0.0
    };
    let factor = f64::from(a_m);
    let dp = [
        (factor * (iso * gf[0] + aniso * x[0])) as f32,
        (factor * (iso * gf[1] + aniso * x[1])) as f32,
        (factor * (iso * gf[2] + aniso * x[2])) as f32,
    ];
    (a_m, dp)
}

struct PassRadiusInput<'a> {
    mesh: &'a Mesh,
    brush: &'a Brush,
    center_m: [f32; 3],
    dir_m: [f32; 3],
    method: GrabMethod,
    adaptive_reach: bool,
    candidates: &'a [VertexId],
    base_positions: &'a [[f32; 3]],
    sorted_zero_protected: &'a [VertexId],
}

fn effective_pass_radius(input: &PassRadiusInput<'_>) -> f32 {
    if !input.adaptive_reach {
        return input.brush.radius;
    }
    let dir_m = input.dir_m;
    let dir_len = (dir_m[0] * dir_m[0] + dir_m[1] * dir_m[1] + dir_m[2] * dir_m[2]).sqrt();
    let length = input.brush.strength.abs() * dir_len;
    let slope = grab_slope(input.brush.falloff, input.method);
    let r_wide = length * slope / GRAB_MAX_STRETCH;
    if !(r_wide > input.brush.radius
        && r_wide.is_finite()
        && dir_len > 0.0
        && input.brush.strength != 0.0)
    {
        return input.brush.radius;
    }

    let sign = if input.brush.strength < 0.0 {
        -1.0f32
    } else {
        1.0f32
    };
    let ux = sign * dir_m[0] / dir_len;
    let uy = sign * dir_m[1] / dir_len;
    let uz = sign * dir_m[2] / dir_len;

    let mut ahead = 0.0f32;
    for (&v, p) in input.candidates.iter().zip(input.base_positions) {
        if input.mesh.corner_count(v) == 0 || input.sorted_zero_protected.binary_search(&v).is_ok()
        {
            continue;
        }
        let dx = p[0] - input.center_m[0];
        let dy = p[1] - input.center_m[1];
        let dz = p[2] - input.center_m[2];
        let along = dx * ux + dy * uy + dz * uz;
        if along <= 0.0 {
            continue;
        }
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist > 0.0 && dist < r_wide {
            let term = (along / dist) * (2.0 * (1.0 - dist / r_wide)).min(1.0);
            if term > ahead {
                ahead = term;
            }
        }
    }
    input.brush.radius.max(r_wide * ahead)
}

fn grab_candidates<'f>(
    mesh: &Mesh,
    options: &GrabOptions,
    candidates: &[VertexId],
    stroke: Option<StrokeSlice<'_>>,
    zero_protected: &[VertexId],
    triangles_of: impl Fn(VertexId) -> &'f [TriangleId],
) -> Result<(Vec<VertexId>, Vec<[f32; 3]>, GrabReport)> {
    let curr_positions = mesh.positions();
    let brush = &options.brush;
    let (masks, mask_count) = options.symmetry.active_masks();

    let mut base_positions: Vec<[f32; 3]> = Vec::new();
    reserve(&mut base_positions, candidates.len())?;
    for &v in candidates {
        base_positions.push(base_pos_of(mesh, options.from_stroke, stroke, v));
    }

    let mut owned_zp: Vec<VertexId> = Vec::new();
    let sorted_zp: &[VertexId] = if zero_protected.windows(2).all(|w| w[0] <= w[1]) {
        zero_protected
    } else {
        reserve(&mut owned_zp, zero_protected.len())?;
        owned_zp.extend_from_slice(zero_protected);
        owned_zp.sort_unstable();
        owned_zp.dedup();
        &owned_zp
    };

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
                curr_positions[v as usize],
                n_v,
                &adjacent,
                curr_positions,
            ));
        }
    }

    let need_auto_normal = matches!(options.filter, SurfaceFilter::NormalAngle { normal: None, .. });
    let mut envelopes = filled(candidates.len(), MaxEnvelope::default())?;

    // Reusable Geodesic scratch buffers across symmetry passes.
    let mut pass_in_sphere: Vec<(usize, VertexId, [f32; 3])> = Vec::new();
    let mut tri_list: Vec<TriangleId> = Vec::new();
    let mut seam_order: Vec<([u32; 3], u32)> = Vec::new();
    let mut seam_next: Vec<u32> = Vec::new();
    let mut geo_dist: Vec<f64> = Vec::new();
    let mut heap: BinaryHeap<GeodesicEntry> = BinaryHeap::new();

    for &bits in &masks[..mask_count] {
        let center_m = reflect_vec(bits, brush.center);
        let dir_m = reflect_vec(bits, options.direction);
        let r_eff = effective_pass_radius(&PassRadiusInput {
            mesh,
            brush,
            center_m,
            dir_m,
            method: options.method,
            adaptive_reach: options.adaptive_reach,
            candidates,
            base_positions: &base_positions,
            sorted_zero_protected: sorted_zp,
        });
        let auto_normal = need_auto_normal.then(|| {
            falloff_weighted_normal(
                candidates,
                &cand_normals,
                curr_positions,
                center_m,
                r_eff,
                brush.falloff,
            )
        });

        match options.method {
            GrabMethod::Sphere => {
                for (cand_idx, &p_base) in base_positions.iter().enumerate() {
                    let Some(t) = normalized_dist(center_m, r_eff, &p_base) else {
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
                    let w = brush.strength * a_m;
                    let dp = [w * dir_m[0], w * dir_m[1], w * dir_m[2]];
                    let single_pos = [
                        p_base[0] + dp[0],
                        p_base[1] + dp[1],
                        p_base[2] + dp[2],
                    ];
                    envelopes[cand_idx].add(a_m, dp, single_pos);
                }
            }
            GrabMethod::Kelvinlet => {
                let g = [
                    brush.strength * dir_m[0],
                    brush.strength * dir_m[1],
                    brush.strength * dir_m[2],
                ];
                for (cand_idx, &p_base) in base_positions.iter().enumerate() {
                    let Some(t) = normalized_dist(center_m, r_eff, &p_base) else {
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
                    let offset = [
                        p_base[0] - center_m[0],
                        p_base[1] - center_m[1],
                        p_base[2] - center_m[2],
                    ];
                    let (a_m, dp) =
                        eval_grab_kelvinlet(offset, r_eff, t, g, options.poisson, phi);
                    if a_m > 0.0 {
                        let single_pos = [
                            p_base[0] + dp[0],
                            p_base[1] + dp[1],
                            p_base[2] + dp[2],
                        ];
                        envelopes[cand_idx].add(a_m, dp, single_pos);
                    }
                }
            }
            GrabMethod::Geodesic => {
                pass_in_sphere.clear();
                for (cand_idx, (&v, &p_base)) in
                    candidates.iter().zip(&base_positions).enumerate()
                {
                    if normalized_dist(center_m, r_eff, &p_base).is_some() {
                        push(&mut pass_in_sphere, (cand_idx, v, p_base))?;
                    }
                }
                let m = pass_in_sphere.len();
                if m == 0 {
                    continue;
                }

                // Seed triangle T*: strictly among live non-degenerate triangles with at least
                // one corner strictly inside Sphere(center_m, r_eff), ties broken by TriangleId.
                tri_list.clear();
                for &(_, v, _) in &pass_in_sphere {
                    for &t in triangles_of(v) {
                        if proper_triangle(mesh, t).is_some() {
                            push(&mut tri_list, t)?;
                        }
                    }
                }
                tri_list.sort_unstable();
                tri_list.dedup();

                let c_m64 = center_m.map(f64::from);
                let mut best_seed: Option<(TriangleId, [f64; 3], f64)> = None;
                for &t in &tri_list {
                    let Some([a, b, c_v]) = proper_triangle(mesh, t) else {
                        continue;
                    };
                    let pa = base_pos_of(mesh, options.from_stroke, stroke, a).map(f64::from);
                    let pb = base_pos_of(mesh, options.from_stroke, stroke, b).map(f64::from);
                    let pc = base_pos_of(mesh, options.from_stroke, stroke, c_v).map(f64::from);
                    let cr = cross(sub(pb, pa), sub(pc, pa));
                    let area2_sq = dot(cr, cr);
                    if !(area2_sq > 0.0 && area2_sq.is_finite()) {
                        continue;
                    }
                    let q = closest_point_on_triangle(c_m64, pa, pb, pc);
                    let dq = sub(q, c_m64);
                    let d0_sq = dot(dq, dq);
                    if !d0_sq.is_finite() {
                        continue;
                    }
                    if best_seed.is_none_or(|(_, _, best_sq)| d0_sq < best_sq) {
                        best_seed = Some((t, q, d0_sq));
                    }
                }
                let Some((seed_tri, best_q, best_d0_sq)) = best_seed else {
                    continue;
                };
                let d0 = best_d0_sq.sqrt();

                // Zero-cost UV seam circular ring by sorted ([u32; 3] bits, local_idx).
                let so_cap = clear_and_reserve(&mut seam_order, m)?;
                for (k, &(_, _, p)) in pass_in_sphere.iter().enumerate() {
                    let key = p.map(|x| (x + 0.0).to_bits());
                    seam_order.push((key, k as u32));
                }
                debug_assert_eq!(seam_order.capacity(), so_cap);
                seam_order.sort_unstable();

                let sn_cap = clear_and_reserve(&mut seam_next, m)?;
                seam_next.resize(m, 0);
                debug_assert_eq!(seam_next.capacity(), sn_cap);
                let mut start = 0;
                while start < m {
                    let group = seam_order[start].0;
                    let mut end = start + 1;
                    while end < m && seam_order[end].0 == group {
                        end += 1;
                    }
                    for i in start..end - 1 {
                        seam_next[seam_order[i].1 as usize] = seam_order[i + 1].1;
                    }
                    seam_next[seam_order[end - 1].1 as usize] = seam_order[start].1;
                    start = end;
                }

                let gd_cap = clear_and_reserve(&mut geo_dist, m)?;
                geo_dist.resize(m, f64::INFINITY);
                debug_assert_eq!(geo_dist.capacity(), gd_cap);
                heap.clear();

                let seed_corners = mesh.triangle(seed_tri).unwrap_or([0; 3]);
                for u in seed_corners {
                    if let Ok(k) = pass_in_sphere.binary_search_by_key(&u, |x| x.1) {
                        let pu = pass_in_sphere[k].2.map(f64::from);
                        let du = d0 + dot(sub(pu, best_q), sub(pu, best_q)).sqrt();
                        let t_seed = (du as f32) / r_eff;
                        if t_seed.is_finite() && t_seed < 1.0 && du < geo_dist[k] {
                            geo_dist[k] = du;
                            push_heap(
                                &mut heap,
                                GeodesicEntry {
                                    dist: du,
                                    idx: k as u32,
                                },
                            )?;
                        }
                    }
                }

                while let Some(GeodesicEntry { dist: d_k, idx }) = heap.pop() {
                    let k = idx as usize;
                    if d_k > geo_dist[k] {
                        continue;
                    }
                    // 1. Propagate zero-cost across coincident UV seam vertices.
                    let mut s = seam_next[k] as usize;
                    while s != k {
                        if d_k < geo_dist[s] {
                            geo_dist[s] = d_k;
                            push_heap(
                                &mut heap,
                                GeodesicEntry {
                                    dist: d_k,
                                    idx: s as u32,
                                },
                            )?;
                        }
                        s = seam_next[s] as usize;
                    }
                    // 2. Relax 1-ring proper triangle edges within Sphere(center_m, r_eff).
                    let v_k = pass_in_sphere[k].1;
                    let p_k = pass_in_sphere[k].2.map(f64::from);
                    collect_proper_neighbors(mesh, v_k, triangles_of(v_k), &mut adjacent)?;
                    for &nb in &adjacent {
                        if let Ok(nb_k) = pass_in_sphere.binary_search_by_key(&nb, |x| x.1) {
                            let p_nb = pass_in_sphere[nb_k].2.map(f64::from);
                            let diff = sub(p_nb, p_k);
                            let cand = d_k + dot(diff, diff).sqrt();
                            let t_cand = (cand as f32) / r_eff;
                            if t_cand.is_finite() && t_cand < 1.0 && cand < geo_dist[nb_k] {
                                geo_dist[nb_k] = cand;
                                push_heap(
                                    &mut heap,
                                    GeodesicEntry {
                                        dist: cand,
                                        idx: nb_k as u32,
                                    },
                                )?;
                            }
                        }
                    }
                }

                for (&(cand_idx, _, p_base), &d) in pass_in_sphere.iter().zip(&geo_dist) {
                    if !d.is_finite() {
                        continue;
                    }
                    let t = (d as f32) / r_eff;
                    if !(t.is_finite() && t < 1.0) {
                        continue;
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
                    let w = brush.strength * a_m;
                    let dp = [w * dir_m[0], w * dir_m[1], w * dir_m[2]];
                    let single_pos = [
                        p_base[0] + dp[0],
                        p_base[1] + dp[1],
                        p_base[2] + dp[2],
                    ];
                    envelopes[cand_idx].add(a_m, dp, single_pos);
                }
            }
        }
    }

    let mut report = GrabReport::default();
    let mut ids: Vec<VertexId> = Vec::new();
    let mut moved_to: Vec<[f32; 3]> = Vec::new();
    if options.from_stroke {
        for ((&v, &p_base), env) in candidates.iter().zip(&base_positions).zip(&envelopes) {
            let curr = curr_positions[v as usize];
            let target = match env.resolve(p_base) {
                Some((displaced, _)) => displaced,
                None => p_base,
            };
            if same_bits(&curr, &target) || !target.iter().all(|x| x.is_finite()) {
                continue;
            }
            let dx = target[0] - curr[0];
            let dy = target[1] - curr[1];
            let dz = target[2] - curr[2];
            let step_len = (dx * dx + dy * dy + dz * dz).sqrt();
            if !step_len.is_finite() {
                continue;
            }
            push(&mut ids, v)?;
            push(&mut moved_to, target)?;
            report.max_displacement = report.max_displacement.max(step_len);
        }
    } else {
        for ((&v, &p_base), env) in candidates.iter().zip(&base_positions).zip(&envelopes) {
            if let Some((displaced, length)) = env.resolve(p_base) {
                push(&mut ids, v)?;
                push(&mut moved_to, displaced)?;
                report.max_displacement = report.max_displacement.max(length);
            }
        }
    }
    report.moved = ids.len();
    Ok((ids, moved_to, report))
}
