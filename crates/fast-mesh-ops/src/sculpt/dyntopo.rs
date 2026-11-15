//! Localized dynamic topology (Dyntopo) remeshing for interactive sculpting.
//!
//! Operates strictly within a brush sphere `S(center, radius)` in
//! `O(|ROI|)` working memory without scanning or cloning the whole mesh when an
//! ROI triangle list (or BVH) and vertex incidence index are provided:
//!
//! 1. **Longest-first edge split** (`len > detail_size * max_detail_ratio`) at
//!    edge midpoints, linearly interpolating bound channels (`stroke_rest_pos`,
//!    `stroke_rest_normal`, UVs, etc.) onto new vertices.
//! 2. **Short-edge collapse** (`len < detail_size * min_detail_ratio` when the
//!    selected [`DyntopoMode`] allows collapse), gated on pinned/boundary locks,
//!    the strict 2-manifold link condition, duplicate-face rejection, and fan
//!    normal/area preservation.
//! 3. **Localized valence/Delaunay edge flip** (`equalize_valence`) over
//!    interior manifold quads inside the brush sphere using a strictly
//!    decreasing valence/length potential so flips never cycle or fold faces.
//! 4. **Optional tangential relaxation** (`smooth_tangent`) sliding unlocked
//!    interior vertices gently along their tangent plane inside the brush
//!    sphere with fan-fold validation.

use std::cmp::Ordering;

use fast_mesh_bvh::Bvh;
use fast_mesh_query::{Region, triangles_in};
use fast_mesh_store::{Incidence, Mesh, StoreError, Transaction, TriangleId, VertexId};

use crate::{OpsError, Result, push, reserve};

/// Default short-edge collapse ratio (`0.4 * L`).
pub const DEFAULT_DYNTOPO_MIN_RATIO: f32 = 0.4;

/// Default long-edge split ratio (`1.33 * L`).
pub const DEFAULT_DYNTOPO_MAX_RATIO: f32 = 1.33;

/// Default target edge length used by [`DyntopoOptions::default`].
pub const DEFAULT_DYNTOPO_DETAIL_SIZE: f32 = 0.25;

/// Minimum dihedral normal dot product required to flip an interior edge
/// (`~cos(75 deg)`), preserving sharp geometric creases and corners.
const FLIP_SHARP_DOT: f64 = 0.25;

/// Tangential relaxation strength before radial falloff.
const SMOOTH_TANGENT_STEP: f64 = 0.45;

/// Hard safety limit on subdivision recursion depth per dab to ensure float
/// termination even on extreme coarse-to-fine scale ratios.
const MAX_SPLIT_DEPTH: u8 = 14;

/// Dynamic topology detail mode controlling split/collapse behavior and how
/// [`DyntopoOptions::detail_size`] maps to edge lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DyntopoMode {
    /// Only splits edges longer than `d_max`; never collapses short edges.
    SplitOnly,
    /// Splits edges longer than `d_max` and collapses edges shorter than
    /// `d_min`, treating `detail_size` as absolute mesh-space edge length `L`.
    #[default]
    SplitCollapse,
    /// Absolute mesh-space detail mode (`L = detail_size`); splits and
    /// collapses unless `min_detail_ratio == 0.0`.
    ConstantDetail,
    /// Brush-relative detail mode (`L = radius * detail_size`); splits and
    /// collapses unless `min_detail_ratio == 0.0`.
    BrushRelative,
}

impl DyntopoMode {
    #[inline]
    fn allows_collapse(self) -> bool {
        !matches!(self, DyntopoMode::SplitOnly)
    }

    #[inline]
    fn target_length(self, radius: f32, detail_size: f32) -> f64 {
        match self {
            DyntopoMode::BrushRelative => f64::from(radius) * f64::from(detail_size),
            DyntopoMode::SplitOnly | DyntopoMode::SplitCollapse | DyntopoMode::ConstantDetail => {
                f64::from(detail_size)
            }
        }
    }
}

/// Configuration for [`apply_dyntopo`] and `Scene` dyntopo methods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DyntopoOptions {
    /// Detail sizing and split/collapse policy.
    pub mode: DyntopoMode,
    /// Target edge length `L` in mesh units (or fraction of brush radius for
    /// [`DyntopoMode::BrushRelative`]).
    pub detail_size: f32,
    /// Collapse threshold factor (`d_min = L * min_detail_ratio`). Setting to
    /// `0.0` disables collapses even in collapsing modes.
    pub min_detail_ratio: f32,
    /// Split threshold factor (`d_max = L * max_detail_ratio`).
    pub max_detail_ratio: f32,
    /// Whether to flip interior manifold edges inside the ROI to equalize
    /// vertex valence toward 6 (with Delaunay tie-breaking).
    pub equalize_valence: bool,
    /// Whether to apply a localized tangential smoothing pass inside the ROI.
    pub smooth_tangent: bool,
}

impl Default for DyntopoOptions {
    fn default() -> Self {
        Self {
            mode: DyntopoMode::SplitCollapse,
            detail_size: DEFAULT_DYNTOPO_DETAIL_SIZE,
            min_detail_ratio: DEFAULT_DYNTOPO_MIN_RATIO,
            max_detail_ratio: DEFAULT_DYNTOPO_MAX_RATIO,
            equalize_valence: true,
            smooth_tangent: false,
        }
    }
}

impl DyntopoOptions {
    /// Constructs options for `mode` and `detail_size` with standard defaults.
    pub fn new(mode: DyntopoMode, detail_size: f32) -> Self {
        Self {
            mode,
            detail_size,
            ..Self::default()
        }
    }

    /// Sets `(min_detail_ratio, max_detail_ratio)`.
    pub fn with_ratios(mut self, min_detail_ratio: f32, max_detail_ratio: f32) -> Self {
        self.min_detail_ratio = min_detail_ratio;
        self.max_detail_ratio = max_detail_ratio;
        self
    }

    /// Sets [`Self::equalize_valence`].
    pub fn with_equalize_valence(mut self, equalize_valence: bool) -> Self {
        self.equalize_valence = equalize_valence;
        self
    }

    /// Sets [`Self::smooth_tangent`].
    pub fn with_smooth_tangent(mut self, smooth_tangent: bool) -> Self {
        self.smooth_tangent = smooth_tangent;
        self
    }
}

/// Summary and topological delta returned by [`apply_dyntopo`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DyntopoPatch {
    /// Number of edge splits applied.
    pub splits: usize,
    /// Number of short-edge collapses applied.
    pub collapses: usize,
    /// Number of interior edge flips applied.
    pub flips: usize,
    /// Number of ROI vertices displaced by tangential relaxation.
    pub smoothed: usize,
    /// Newly added vertices whose direct committed parents are `[new_v, a, b]`
    /// in creation order (for incremental stroke-rest interpolation).
    pub vertices: Vec<[VertexId; 3]>,
    /// All committed new vertex IDs created in this pass.
    pub new_vertices: Vec<VertexId>,
    /// Pre-existing vertex IDs removed by collapses.
    pub removed_vertices: Vec<VertexId>,
    /// Pre-existing triangle IDs removed by collapses.
    pub removed_triangles: Vec<TriangleId>,
    /// Newly allocated triangle IDs created beyond updated existing slots.
    pub created_triangles: Vec<TriangleId>,
    /// Pre-existing triangle IDs whose vertex triples changed in place.
    pub modified_triangles: Vec<TriangleId>,
}

impl DyntopoPatch {
    /// Returns `true` when no split, collapse, flip, or smooth move occurred.
    pub fn is_empty(&self) -> bool {
        self.splits == 0 && self.collapses == 0 && self.flips == 0 && self.smoothed == 0
    }
}

/// Alias matching report naming across `fast_mesh_ops::sculpt`.
pub type DyntopoReport = DyntopoPatch;

/// Validates `center`, `radius`, and `options`, returning `(d_min, d_max)`.
pub fn validate_dyntopo_inputs(
    center: [f32; 3],
    radius: f32,
    options: &DyntopoOptions,
) -> Result<(f64, f64)> {
    if center.iter().any(|c| !c.is_finite()) {
        return Err(OpsError::InvalidInput("dyntopo center must be finite"));
    }
    if !radius.is_finite() || radius <= 0.0 {
        return Err(OpsError::InvalidInput(
            "dyntopo radius must be finite and positive",
        ));
    }
    if !options.detail_size.is_finite() || options.detail_size <= 0.0 {
        return Err(OpsError::InvalidInput(
            "dyntopo detail_size must be finite and positive",
        ));
    }
    if !options.min_detail_ratio.is_finite() || options.min_detail_ratio < 0.0 {
        return Err(OpsError::InvalidInput(
            "dyntopo min_detail_ratio must be finite and non-negative",
        ));
    }
    if !options.max_detail_ratio.is_finite()
        || options.max_detail_ratio <= 0.0
        || options.max_detail_ratio <= options.min_detail_ratio
    {
        return Err(OpsError::InvalidInput(
            "dyntopo max_detail_ratio must be finite, positive, and greater than min_detail_ratio",
        ));
    }
    let target = options.mode.target_length(radius, options.detail_size);
    let d_min = target * f64::from(options.min_detail_ratio);
    let d_max = target * f64::from(options.max_detail_ratio);
    if !target.is_finite()
        || target <= 0.0
        || !d_min.is_finite()
        || !d_max.is_finite()
        || d_max <= 0.0
    {
        return Err(OpsError::InvalidInput(
            "dyntopo effective edge length thresholds overflowed or underflowed",
        ));
    }
    Ok((d_min, d_max))
}

/// Applies localized dynamic topology inside sphere `(center, radius)`.
///
/// Scans alive mesh triangles once to seed the brush ROI when no BVH is
/// provided, then executes all remeshing work inside sparse `O(|ROI|)` local
/// structures and commits atomically inside `keeping_bound_channels`.
///
/// # Errors
/// - [`OpsError::InvalidInput`] on non-finite/non-positive parameters.
/// - [`OpsError::Store`] on transaction or out-of-memory failure (atomically
///   with zero writes to `tx` on out-of-memory).
pub fn apply_dyntopo(
    tx: &mut Transaction<'_>,
    center: [f32; 3],
    radius: f32,
    options: &DyntopoOptions,
) -> Result<DyntopoPatch> {
    apply_dyntopo_protected(tx, center, radius, options, &[])
}

/// Like [`apply_dyntopo`], keeping all `pinned` vertices untouched during
/// collapses, edge flips, and tangential smoothing.
///
/// # Errors
/// Returns [`OpsError::InvalidInput`] if any `pinned` vertex is not alive, or
/// any error of [`apply_dyntopo`].
pub fn apply_dyntopo_protected(
    tx: &mut Transaction<'_>,
    center: [f32; 3],
    radius: f32,
    options: &DyntopoOptions,
    pinned: &[VertexId],
) -> Result<DyntopoPatch> {
    validate_dyntopo_inputs(center, radius, options)?;
    validate_pinned(tx.mesh(), pinned)?;
    let roi = collect_roi_triangles_scan(tx.mesh(), center, radius)?;
    apply_dyntopo_core(tx, center, radius, options, &roi, pinned, false)
}

/// Applies localized dynamic topology using a pre-queried set of ROI triangles
/// intersecting sphere `(center, radius)` (e.g. from `fast_mesh_query::triangles_in`).
///
/// When `tx.mesh().incidence()` is already populated, runs in strict
/// `O(|ROI|)` time and memory with zero whole-mesh scans.
///
/// # Errors
/// Returns [`OpsError::InvalidInput`] when inputs or any `roi_triangles` /
/// `pinned` ID is dead/invalid, or [`OpsError::Store`] on allocation failure.
pub fn apply_dyntopo_in_roi(
    tx: &mut Transaction<'_>,
    center: [f32; 3],
    radius: f32,
    options: &DyntopoOptions,
    roi_triangles: &[TriangleId],
    pinned: &[VertexId],
) -> Result<DyntopoPatch> {
    apply_dyntopo_in_roi_impl(tx, center, radius, options, roi_triangles, pinned, false)
}

/// Like [`apply_dyntopo_in_roi`], restricting edge collapses to single-step
/// parent pairs `[new_v, a, b]` so `DyntopoPatch::vertices` maps 1:1 with every
/// created vertex for incremental stroke-snapshot updates.
#[doc(hidden)]
pub fn apply_dyntopo_in_roi_binary_parents(
    tx: &mut Transaction<'_>,
    center: [f32; 3],
    radius: f32,
    options: &DyntopoOptions,
    roi_triangles: &[TriangleId],
    pinned: &[VertexId],
) -> Result<DyntopoPatch> {
    apply_dyntopo_in_roi_impl(tx, center, radius, options, roi_triangles, pinned, true)
}

/// Applies localized dynamic topology using `bvh` to query ROI triangles in
/// `O(log N + K)` before executing sparse local remeshing.
///
/// # Errors
/// Returns BVH query errors mapped via [`OpsError::InvalidInput`] /
/// [`OpsError::Store`], or errors of [`apply_dyntopo_in_roi`].
pub fn apply_dyntopo_with_bvh(
    tx: &mut Transaction<'_>,
    bvh: &Bvh,
    center: [f32; 3],
    radius: f32,
    options: &DyntopoOptions,
    pinned: &[VertexId],
) -> Result<DyntopoPatch> {
    validate_dyntopo_inputs(center, radius, options)?;
    validate_pinned(tx.mesh(), pinned)?;
    let mut roi = Vec::new();
    triangles_in(tx.mesh(), bvh, &Region::Sphere { center, radius }, &mut roi)
        .map_err(query_err_to_ops)?;
    apply_dyntopo_core(tx, center, radius, options, &roi, pinned, false)
}

fn apply_dyntopo_in_roi_impl(
    tx: &mut Transaction<'_>,
    center: [f32; 3],
    radius: f32,
    options: &DyntopoOptions,
    roi_triangles: &[TriangleId],
    pinned: &[VertexId],
    binary_parents_only: bool,
) -> Result<DyntopoPatch> {
    validate_dyntopo_inputs(center, radius, options)?;
    validate_pinned(tx.mesh(), pinned)?;
    for &t in roi_triangles {
        if !tx.mesh().is_triangle_alive(t) {
            return Err(OpsError::InvalidInput(
                "dyntopo roi contains dead or unknown triangle",
            ));
        }
    }
    apply_dyntopo_core(
        tx,
        center,
        radius,
        options,
        roi_triangles,
        pinned,
        binary_parents_only,
    )
}

fn query_err_to_ops(err: fast_mesh_query::QueryError) -> OpsError {
    match err {
        fast_mesh_query::QueryError::OutOfMemory => OpsError::Store(StoreError::OutOfMemory),
        fast_mesh_query::QueryError::InvalidInput(msg) => OpsError::InvalidInput(msg),
        fast_mesh_query::QueryError::MeshMismatch | fast_mesh_query::QueryError::StaleBvh => {
            OpsError::InvalidInput("dyntopo BVH does not match mesh")
        }
    }
}

fn validate_pinned(mesh: &Mesh, pinned: &[VertexId]) -> Result<()> {
    for &v in pinned {
        if !mesh.is_vertex_alive(v) {
            return Err(OpsError::InvalidInput("dyntopo pinned vertex is not alive"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Geometry & Vector Helpers (f64 precision)
// ---------------------------------------------------------------------------

#[inline]
fn f64v(p: [f32; 3]) -> [f64; 3] {
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dist2(a: [f32; 3], b: [f32; 3]) -> f64 {
    let d = sub3(f64v(a), f64v(b));
    dot3(d, d)
}

#[inline]
fn midpoint(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        ((f64::from(a[0]) + f64::from(b[0])) * 0.5) as f32,
        ((f64::from(a[1]) + f64::from(b[1])) * 0.5) as f32,
        ((f64::from(a[2]) + f64::from(b[2])) * 0.5) as f32,
    ]
}

#[inline]
fn point_in_sphere(p: [f32; 3], center: [f64; 3], r2: f64) -> bool {
    let d = sub3(f64v(p), center);
    dot3(d, d) <= r2
}

#[inline]
fn segment_intersects_sphere(a: [f32; 3], b: [f32; 3], center: [f64; 3], r2: f64) -> bool {
    let pa = f64v(a);
    let pb = f64v(b);
    let ab = sub3(pb, pa);
    let ac = sub3(center, pa);
    let ab2 = dot3(ab, ab);
    let t = if ab2 > 0.0 {
        (dot3(ac, ab) / ab2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let closest = [pa[0] + t * ab[0], pa[1] + t * ab[1], pa[2] + t * ab[2]];
    let diff = sub3(closest, center);
    dot3(diff, diff) <= r2
}

/// Exact distance-squared from `c` to triangle `[a, b, c_tri]` (Ericson Voronoi).
fn sphere_triangle_intersects(center: [f64; 3], r2: f64, tri: [[f32; 3]; 3]) -> bool {
    let a = f64v(tri[0]);
    let b = f64v(tri[1]);
    let c = f64v(tri[2]);
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    let ap = sub3(center, a);

    let d1 = dot3(ab, ap);
    let d2 = dot3(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return dot3(ap, ap) <= r2;
    }

    let bp = sub3(center, b);
    let d3 = dot3(ab, bp);
    let d4 = dot3(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return dot3(bp, bp) <= r2;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let q = [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]];
        let d = sub3(center, q);
        return dot3(d, d) <= r2;
    }

    let cp = sub3(center, c);
    let d5 = dot3(ab, cp);
    let d6 = dot3(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return dot3(cp, cp) <= r2;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let q = [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]];
        let d = sub3(center, q);
        return dot3(d, d) <= r2;
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = sub3(c, b);
        let q = [b[0] + w * bc[0], b[1] + w * bc[1], b[2] + w * bc[2]];
        let d = sub3(center, q);
        return dot3(d, d) <= r2;
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let q = [
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    ];
    let d = sub3(center, q);
    dot3(d, d) <= r2
}

#[inline]
fn grow_push<T>(v: &mut Vec<T>, min_extra: usize, item: T) -> Result<()> {
    if v.len() == v.capacity() {
        let extra = v.len().max(min_extra);
        reserve(v, extra)?;
    }
    v.push(item);
    Ok(())
}

fn collect_roi_triangles_scan(
    mesh: &Mesh,
    center: [f32; 3],
    radius: f32,
) -> Result<Vec<TriangleId>> {
    let r2 = f64::from(radius) * f64::from(radius);
    let c = f64v(center);
    let pos = mesh.positions();
    let slots = mesh.triangle_slots();
    let mut out = Vec::new();
    for t in 0..slots {
        let tid = t as TriangleId;
        if let Some([a, b, c_id]) = mesh.triangle(tid) {
            let tri_pos = [pos[a as usize], pos[b as usize], pos[c_id as usize]];
            if sphere_triangle_intersects(c, r2, tri_pos) {
                grow_push(&mut out, 64, tid)?;
            }
        }
    }
    Ok(out)
}

#[inline]
fn tri_area2_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> ([f64; 3], f64) {
    let pa = f64v(a);
    let pb = f64v(b);
    let pc = f64v(c);
    let n = cross3(sub3(pb, pa), sub3(pc, pa));
    (n, dot3(n, n))
}

#[inline]
fn runs_directed(tri: [u32; 3], a: u32, b: u32) -> bool {
    (tri[0] == a && tri[1] == b) || (tri[1] == a && tri[2] == b) || (tri[2] == a && tri[0] == b)
}

#[inline]
fn apex_of(tri: [u32; 3], a: u32, b: u32) -> Option<u32> {
    if (tri[0] == a && tri[1] == b) || (tri[0] == b && tri[1] == a) {
        Some(tri[2])
    } else if (tri[1] == a && tri[2] == b) || (tri[1] == b && tri[2] == a) {
        Some(tri[0])
    } else if (tri[2] == a && tri[0] == b) || (tri[2] == b && tri[0] == a) {
        Some(tri[1])
    } else {
        None
    }
}

#[inline]
fn min_sin2(p: [[f32; 3]; 3]) -> f64 {
    let a = f64v(p[0]);
    let b = f64v(p[1]);
    let c = f64v(p[2]);
    let ab = sub3(b, a);
    let bc = sub3(c, b);
    let ca = sub3(a, c);
    let l_ab2 = dot3(ab, ab);
    let l_bc2 = dot3(bc, bc);
    let l_ca2 = dot3(ca, ca);
    let n = cross3(ab, sub3(c, a));
    let area2 = dot3(n, n);
    if area2 <= 0.0 || l_ab2 <= 0.0 || l_bc2 <= 0.0 || l_ca2 <= 0.0 {
        return 0.0;
    }
    let s_a = area2 / (l_ab2 * l_ca2);
    let s_b = area2 / (l_ab2 * l_bc2);
    let s_c = area2 / (l_bc2 * l_ca2);
    s_a.min(s_b).min(s_c)
}

// ---------------------------------------------------------------------------
// Fallible Sparse Containers (never depend on bucket iteration order)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct IdIndex {
    buckets: Vec<(u32, u32)>,
    mask: usize,
    len: usize,
}

impl IdIndex {
    fn init(initial_cap: usize) -> Result<Self> {
        let cap = initial_cap.next_power_of_two().max(64);
        let buckets = crate::filled(cap, (u32::MAX, 0))?;
        Ok(Self {
            buckets,
            mask: cap - 1,
            len: 0,
        })
    }

    #[inline]
    fn hash(k: u32) -> usize {
        (k.wrapping_mul(0x9E37_79B9) ^ (k >> 16)) as usize
    }

    #[inline]
    fn get(&self, key: u32) -> Option<u32> {
        let mut idx = Self::hash(key) & self.mask;
        loop {
            let (k, v) = self.buckets[idx];
            if k == key {
                return Some(v);
            }
            if k == u32::MAX {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    fn insert(&mut self, key: u32, val: u32) -> Result<()> {
        debug_assert_ne!(key, u32::MAX);
        if (self.len + 1) * 10 >= self.buckets.len() * 7 {
            self.grow()?;
        }
        let mut idx = Self::hash(key) & self.mask;
        loop {
            let (k, v) = &mut self.buckets[idx];
            if *k == u32::MAX {
                *k = key;
                *v = val;
                self.len += 1;
                return Ok(());
            }
            if *k == key {
                *v = val;
                return Ok(());
            }
            idx = (idx + 1) & self.mask;
        }
    }

    fn grow(&mut self) -> Result<()> {
        let new_cap = (self.buckets.len() * 2).max(64);
        let mut new_buckets = crate::filled(new_cap, (u32::MAX, 0))?;
        let new_mask = new_cap - 1;
        for &(k, v) in &self.buckets {
            if k == u32::MAX {
                continue;
            }
            let mut idx = Self::hash(k) & new_mask;
            while new_buckets[idx].0 != u32::MAX {
                idx = (idx + 1) & new_mask;
            }
            new_buckets[idx] = (k, v);
        }
        self.buckets = new_buckets;
        self.mask = new_mask;
        Ok(())
    }
}

struct MaxHeap<T> {
    data: Vec<T>,
}

impl<T: Ord> MaxHeap<T> {
    fn with_reserve(cap: usize) -> Result<Self> {
        let mut data = Vec::new();
        reserve(&mut data, cap)?;
        Ok(Self { data })
    }

    fn push(&mut self, item: T) -> Result<()> {
        grow_push(&mut self.data, 64, item)?;
        let mut i = self.data.len() - 1;
        while i > 0 {
            let p = (i - 1) >> 1;
            if self.data[i] > self.data[p] {
                self.data.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
        Ok(())
    }

    fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let top = self.data.swap_remove(0);
        let n = self.data.len();
        let mut i = 0;
        loop {
            let left = (i << 1) + 1;
            if left >= n {
                break;
            }
            let right = left + 1;
            let best = if right < n && self.data[right] > self.data[left] {
                right
            } else {
                left
            };
            if self.data[best] > self.data[i] {
                self.data.swap(i, best);
                i = best;
            } else {
                break;
            }
        }
        Some(top)
    }
}

// ---------------------------------------------------------------------------
// Fallback Vertex->Triangles CSR (only when Mesh::incidence() is absent)
// ---------------------------------------------------------------------------

struct FallbackRows {
    offsets: Vec<u32>,
    tris: Vec<TriangleId>,
}

impl FallbackRows {
    fn build(mesh: &Mesh) -> Result<Self> {
        let v_slots = mesh.vertex_slots();
        let t_slots = mesh.triangle_slots();
        let mut deg = crate::filled(v_slots, 0u32)?;
        let mut total = 0usize;
        for t in 0..t_slots {
            if let Some(corners) = mesh.triangle(t as TriangleId) {
                for v in corners {
                    deg[v as usize] += 1;
                    total += 1;
                }
            }
        }
        let mut offsets = crate::filled(v_slots + 1, 0u32)?;
        let mut cur = 0u32;
        for i in 0..v_slots {
            offsets[i] = cur;
            cur += deg[i];
        }
        offsets[v_slots] = cur;
        let mut write = crate::filled(v_slots, 0u32)?;
        write.copy_from_slice(&offsets[..v_slots]);
        let mut tris = crate::filled(total, 0u32)?;
        for t in 0..t_slots {
            let tid = t as TriangleId;
            if let Some(corners) = mesh.triangle(tid) {
                for v in corners {
                    let pos = write[v as usize] as usize;
                    tris[pos] = tid;
                    write[v as usize] = (pos + 1) as u32;
                }
            }
        }
        Ok(Self { offsets, tris })
    }

    #[inline]
    fn get(&self, v: VertexId) -> &[TriangleId] {
        let vi = v as usize;
        let s = self.offsets[vi] as usize;
        let e = self.offsets[vi + 1] as usize;
        &self.tris[s..e]
    }
}

enum TriLookup<'a> {
    Cached(&'a Incidence),
    Built(FallbackRows),
}

impl<'a> TriLookup<'a> {
    fn triangles(&self, v: VertexId) -> &[TriangleId] {
        match self {
            TriLookup::Cached(inc) => inc.triangles(v),
            TriLookup::Built(rows) => rows.get(v),
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse Local Workspace
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct LocalVert {
    orig_id: Option<VertexId>,
    pos: [f32; 3],
    parents: Option<[u32; 2]>,
    depth: u8,
    pinned: bool,
    fresh_from_split: bool,
    fresh_from_collapse: bool,
    moved: bool,
    alive: bool,
    row_loaded: bool,
    row: Vec<u32>,
    committed_id: Option<VertexId>,
}

#[derive(Clone)]
struct LocalTri {
    orig_id: Option<TriangleId>,
    orig_corners: [u32; 3],
    v: [u32; 3],
    alive: bool,
    in_initial_roi: bool,
}

/// Binary max-heap key ordered by descending length, breaking ties deterministically.
#[derive(Clone, Copy, Eq, PartialEq)]
struct LongEdge {
    len2_bits: u64,
    a: u32,
    b: u32,
}

impl Ord for LongEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.len2_bits
            .cmp(&other.len2_bits)
            .then_with(|| other.a.cmp(&self.a))
            .then_with(|| other.b.cmp(&self.b))
    }
}

impl PartialOrd for LongEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Binary max-heap key ordered by ascending length (shortest popped first).
#[derive(Clone, Copy, Eq, PartialEq)]
struct ShortEdge {
    len2_bits: u64,
    a: u32,
    b: u32,
}

impl Ord for ShortEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .len2_bits
            .cmp(&self.len2_bits)
            .then_with(|| other.a.cmp(&self.a))
            .then_with(|| other.b.cmp(&self.b))
    }
}

impl PartialOrd for ShortEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct PinSet<'a> {
    sorted_ref: Option<&'a [VertexId]>,
    owned: Vec<VertexId>,
}

impl<'a> PinSet<'a> {
    fn from_slice(pinned: &'a [VertexId]) -> Result<Self> {
        if pinned.is_empty() {
            return Ok(Self {
                sorted_ref: Some(pinned),
                owned: Vec::new(),
            });
        }
        if pinned.windows(2).all(|w| w[0] < w[1]) {
            return Ok(Self {
                sorted_ref: Some(pinned),
                owned: Vec::new(),
            });
        }
        let mut owned = Vec::new();
        reserve(&mut owned, pinned.len())?;
        owned.extend_from_slice(pinned);
        owned.sort_unstable();
        owned.dedup();
        Ok(Self {
            sorted_ref: None,
            owned,
        })
    }

    #[inline]
    fn contains(&self, v: VertexId) -> bool {
        let slice = self.sorted_ref.unwrap_or(&self.owned);
        slice.binary_search(&v).is_ok()
    }
}

enum EdgeOwners {
    NotTwo,
    Two(u32, u32),
}

struct Work<'a> {
    mesh: &'a Mesh,
    lookup: TriLookup<'a>,
    pins: PinSet<'a>,
    center: [f64; 3],
    r2: f64,
    d_min2: f64,
    d_max2: f64,
    verts: Vec<LocalVert>,
    tris: Vec<LocalTri>,
    vert_map: IdIndex,
    tri_map: IdIndex,
    splits: usize,
    collapses: usize,
    flips: usize,
    smoothed: usize,
}

impl<'a> Work<'a> {
    fn new(
        mesh: &'a Mesh,
        roi_triangles: &[TriangleId],
        pinned: &'a [VertexId],
        center: [f32; 3],
        radius: f32,
        d_min: f64,
        d_max: f64,
    ) -> Result<Self> {
        let lookup = match mesh.incidence() {
            Some(inc) => TriLookup::Cached(inc),
            None => TriLookup::Built(FallbackRows::build(mesh)?),
        };
        let pins = PinSet::from_slice(pinned)?;

        let mut sorted_roi = Vec::new();
        reserve(&mut sorted_roi, roi_triangles.len())?;
        sorted_roi.extend_from_slice(roi_triangles);
        sorted_roi.sort_unstable();
        sorted_roi.dedup();

        let est_tris = sorted_roi.len() * 3 + 32;
        let est_verts = sorted_roi.len() * 2 + 32;
        let mut verts = Vec::new();
        reserve(&mut verts, est_verts)?;
        let mut tris = Vec::new();
        reserve(&mut tris, est_tris)?;
        let vert_map = IdIndex::init(est_verts * 2)?;
        let tri_map = IdIndex::init(est_tris * 2)?;

        let mut work = Self {
            mesh,
            lookup,
            pins,
            center: f64v(center),
            r2: f64::from(radius) * f64::from(radius),
            d_min2: d_min * d_min,
            d_max2: d_max * d_max,
            verts,
            tris,
            vert_map,
            tri_map,
            splits: 0,
            collapses: 0,
            flips: 0,
            smoothed: 0,
        };

        for tid in sorted_roi {
            work.ensure_tri_loaded(tid, true)?;
        }
        Ok(work)
    }

    fn ensure_vert_loaded(&mut self, vid: VertexId) -> Result<u32> {
        if let Some(lv) = self.vert_map.get(vid) {
            return Ok(lv);
        }
        let lv = self.verts.len() as u32;
        let mut row = Vec::new();
        reserve(&mut row, 8)?;
        grow_push(
            &mut self.verts,
            64,
            LocalVert {
                orig_id: Some(vid),
                pos: self.mesh.positions()[vid as usize],
                parents: None,
                depth: 0,
                pinned: self.pins.contains(vid),
                fresh_from_split: false,
                fresh_from_collapse: false,
                moved: false,
                alive: true,
                row_loaded: false,
                row,
                committed_id: None,
            },
        )?;
        self.vert_map.insert(vid, lv)?;
        Ok(lv)
    }

    #[inline]
    fn push_vert_row(&mut self, lv: u32, lt: u32) -> Result<()> {
        grow_push(&mut self.verts[lv as usize].row, 8, lt)
    }

    #[inline]
    fn remove_vert_row(&mut self, lv: u32, lt: u32) {
        let row = &mut self.verts[lv as usize].row;
        if let Some(pos) = row.iter().position(|&x| x == lt) {
            row.swap_remove(pos);
        }
    }

    fn ensure_tri_loaded(&mut self, tid: TriangleId, mark_roi: bool) -> Result<u32> {
        if let Some(lt) = self.tri_map.get(tid) {
            if mark_roi {
                self.tris[lt as usize].in_initial_roi = true;
            }
            return Ok(lt);
        }
        let [v0, v1, v2] = self
            .mesh
            .triangle(tid)
            .ok_or(OpsError::Store(StoreError::InvalidTriangle(tid)))?;
        let l0 = self.ensure_vert_loaded(v0)?;
        let l1 = self.ensure_vert_loaded(v1)?;
        let l2 = self.ensure_vert_loaded(v2)?;

        let lt = self.tris.len() as u32;
        grow_push(
            &mut self.tris,
            64,
            LocalTri {
                orig_id: Some(tid),
                orig_corners: [l0, l1, l2],
                v: [l0, l1, l2],
                alive: true,
                in_initial_roi: mark_roi,
            },
        )?;
        self.tri_map.insert(tid, lt)?;
        self.push_vert_row(l0, lt)?;
        self.push_vert_row(l1, lt)?;
        self.push_vert_row(l2, lt)?;
        Ok(lt)
    }

    fn ensure_row_loaded(&mut self, lv: u32) -> Result<()> {
        if self.verts[lv as usize].row_loaded {
            return Ok(());
        }
        let Some(orig_v) = self.verts[lv as usize].orig_id else {
            self.verts[lv as usize].row_loaded = true;
            return Ok(());
        };
        let mut inc_buf = Vec::new();
        let inc = self.lookup.triangles(orig_v);
        reserve(&mut inc_buf, inc.len())?;
        inc_buf.extend_from_slice(inc);
        // Sort by TriangleId so triangle/vertex insertion order into Work never
        // depends on incremental Incidence row order.
        inc_buf.sort_unstable();
        for tid in inc_buf {
            self.ensure_tri_loaded(tid, false)?;
        }
        self.verts[lv as usize].row_loaded = true;
        Ok(())
    }

    fn new_local_vert(
        &mut self,
        pos: [f32; 3],
        p0: u32,
        p1: u32,
        depth: u8,
        from_split: bool,
        from_collapse: bool,
    ) -> Result<u32> {
        let lv = self.verts.len() as u32;
        let mut row = Vec::new();
        reserve(&mut row, 8)?;
        grow_push(
            &mut self.verts,
            64,
            LocalVert {
                orig_id: None,
                pos,
                parents: Some([p0, p1]),
                depth,
                pinned: false,
                fresh_from_split: from_split,
                fresh_from_collapse: from_collapse,
                moved: false,
                alive: true,
                row_loaded: true,
                row,
                committed_id: None,
            },
        )?;
        Ok(lv)
    }

    fn new_local_tri(&mut self, v: [u32; 3], in_roi: bool) -> Result<u32> {
        let lt = self.tris.len() as u32;
        grow_push(
            &mut self.tris,
            64,
            LocalTri {
                orig_id: None,
                orig_corners: v,
                v,
                alive: true,
                in_initial_roi: in_roi,
            },
        )?;
        self.push_vert_row(v[0], lt)?;
        self.push_vert_row(v[1], lt)?;
        self.push_vert_row(v[2], lt)?;
        Ok(lt)
    }

    fn collect_edge_owners(&self, a: u32, b: u32, out: &mut Vec<u32>) -> Result<()> {
        out.clear();
        for &lt in &self.verts[a as usize].row {
            let tri = &self.tris[lt as usize];
            if tri.alive && tri.v.contains(&b) {
                if out.len() == out.capacity() {
                    reserve(out, 4)?;
                }
                out.push(lt);
            }
        }
        Ok(())
    }

    fn edge_owners_manifold(&self, a: u32, b: u32) -> EdgeOwners {
        let mut first = None;
        let mut second = None;
        for &lt in &self.verts[a as usize].row {
            let tri = &self.tris[lt as usize];
            if tri.alive && tri.v.contains(&b) {
                if first.is_none() {
                    first = Some(lt);
                } else if second.is_none() {
                    second = Some(lt);
                } else {
                    return EdgeOwners::NotTwo;
                }
            }
        }
        match (first, second) {
            (Some(t0), Some(t1)) => EdgeOwners::Two(t0, t1),
            _ => EdgeOwners::NotTwo,
        }
    }

    #[inline]
    fn has_edge(&self, a: u32, b: u32) -> bool {
        for &lt in &self.verts[a as usize].row {
            let tri = &self.tris[lt as usize];
            if tri.alive && tri.v.contains(&b) {
                return true;
            }
        }
        false
    }

    /// Checks whether vertex `v` is an interior 2-manifold disk vertex: every
    /// edge incident to `v` has 1 outgoing and 1 incoming half-edge in `v`'s
    /// live fan, and all incident triangles form one single cycle around `v`.
    fn is_interior_manifold_disk(&self, v: u32, fan_deg_out: &mut usize) -> bool {
        let mut live_tris = 0usize;
        for &lt in &self.verts[v as usize].row {
            if self.tris[lt as usize].alive {
                live_tris += 1;
            }
        }
        if live_tris < 3 {
            return false;
        }
        // Stack-free walk around the vertex fan using cyclic outgoing->incoming edges.
        // Pick first live triangle's outgoing neighbor.
        let mut start_t = u32::MAX;
        for &lt in &self.verts[v as usize].row {
            if self.tris[lt as usize].alive {
                start_t = lt;
                break;
            }
        }
        if start_t == u32::MAX {
            return false;
        }
        let tri0 = self.tris[start_t as usize].v;
        let Some(_) = next_around_corner(tri0, v) else {
            return false;
        };
        let mut cur_t = start_t;
        let mut visited = 1usize;

        while visited <= live_tris {
            // In triangle `cur_t`, step across outgoing edge `(v, next)` to
            // the neighbor triangle where it forms the incoming edge `next -> v`.
            let cur_v = self.tris[cur_t as usize].v;
            let Some(next) = next_around_corner(cur_v, v) else {
                return false;
            };
            let mut match_t = None;
            let mut matches = 0usize;
            for &lt in &self.verts[v as usize].row {
                let tri = &self.tris[lt as usize];
                if !tri.alive || lt == cur_t {
                    continue;
                }
                if runs_directed(tri.v, next, v) {
                    match_t = Some(lt);
                    matches += 1;
                } else if runs_directed(tri.v, v, next) {
                    // Orientation conflict on edge (v, next).
                    return false;
                }
            }
            if matches != 1 {
                return false;
            }
            let nt = match_t.unwrap();
            if nt == start_t {
                *fan_deg_out = live_tris;
                return visited == live_tris;
            }
            cur_t = nt;
            visited += 1;
        }
        false
    }

    // -----------------------------------------------------------------------
    // Pass 1: Longest-First Edge Split
    // -----------------------------------------------------------------------

    fn run_splits(&mut self) -> Result<()> {
        let mut cand_edges = Vec::new();
        reserve(&mut cand_edges, self.tris.len() * 3 + 16)?;
        let initial_tris = self.tris.len();

        for lt in 0..initial_tris {
            let tri = self.tris[lt].v;
            let pa = self.verts[tri[0] as usize].pos;
            let pb = self.verts[tri[1] as usize].pos;
            let pc = self.verts[tri[2] as usize].pos;

            let e01 = segment_intersects_sphere(pa, pb, self.center, self.r2);
            let e12 = segment_intersects_sphere(pb, pc, self.center, self.r2);
            let e20 = segment_intersects_sphere(pc, pa, self.center, self.r2);

            let pairs = [
                (tri[0], tri[1], e01),
                (tri[1], tri[2], e12),
                (tri[2], tri[0], e20),
            ];
            for &(u, v, hits) in &pairs {
                if hits {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    cand_edges.push((a, b));
                }
            }

            // Coarse-triangle fallback: if a giant triangle intersects the brush
            // sphere in its interior without touching any edge, seed its longest
            // edge so subdivision marches into the sphere interior.
            if !e01 && !e12 && !e20 && self.tris[lt].in_initial_roi {
                let d01 = dist2(pa, pb);
                let d12 = dist2(pb, pc);
                let d20 = dist2(pc, pa);
                let (u, v) = if d01 >= d12 && d01 >= d20 {
                    (tri[0], tri[1])
                } else if d12 >= d20 {
                    (tri[1], tri[2])
                } else {
                    (tri[2], tri[0])
                };
                if dist2(self.verts[u as usize].pos, self.verts[v as usize].pos) > self.d_max2 {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    cand_edges.push((a, b));
                }
            }
        }

        cand_edges.sort_unstable();
        cand_edges.dedup();

        let mut heap = MaxHeap::with_reserve(cand_edges.len() * 2 + 16)?;
        for (a, b) in cand_edges {
            let d2 = dist2(self.verts[a as usize].pos, self.verts[b as usize].pos);
            if d2 > self.d_max2 {
                heap.push(LongEdge {
                    len2_bits: d2.to_bits(),
                    a,
                    b,
                })?;
            }
        }

        let mut owners = Vec::new();
        reserve(&mut owners, 8)?;

        while let Some(edge) = heap.pop() {
            let (a, b) = (edge.a, edge.b);
            if !self.verts[a as usize].alive || !self.verts[b as usize].alive {
                continue;
            }
            let next_depth = self.verts[a as usize]
                .depth
                .max(self.verts[b as usize].depth)
                + 1;
            if next_depth > MAX_SPLIT_DEPTH {
                continue;
            }
            let pa = self.verts[a as usize].pos;
            let pb = self.verts[b as usize].pos;
            let d2 = dist2(pa, pb);
            if d2 <= self.d_max2 {
                continue;
            }
            let mid = midpoint(pa, pb);
            let d_am2 = dist2(pa, mid);
            let d_mb2 = dist2(mid, pb);
            if d_am2 <= 0.0 || d_mb2 <= 0.0 || d_am2 >= d2 || d_mb2 >= d2 {
                continue;
            }

            self.ensure_row_loaded(a)?;
            self.ensure_row_loaded(b)?;
            self.collect_edge_owners(a, b, &mut owners)?;
            if owners.is_empty() {
                continue;
            }

            // Split edge `(a, b)` at `mid`.
            let m = self.new_local_vert(mid, a, b, next_depth, true, false)?;
            self.splits += 1;

            // Deterministic ordering of owner triangles before splitting.
            owners.sort_unstable();
            for &lt in &owners {
                let old_v = self.tris[lt as usize].v;
                let Some(c) = apex_of(old_v, a, b) else {
                    continue;
                };
                self.ensure_row_loaded(c)?;
                let (t_keep, t_add) = if runs_directed(old_v, a, b) {
                    ([a, m, c], [m, b, c])
                } else {
                    ([b, m, c], [m, a, c])
                };

                // Update `lt` in place: `lt` loses the corner moved into `t_add`.
                let lost = if runs_directed(old_v, a, b) { b } else { a };
                self.tris[lt as usize].v = t_keep;
                self.remove_vert_row(lost, lt);
                self.push_vert_row(m, lt)?;

                let child_in_roi = sphere_triangle_intersects(
                    self.center,
                    self.r2,
                    [
                        self.verts[t_add[0] as usize].pos,
                        self.verts[t_add[1] as usize].pos,
                        self.verts[t_add[2] as usize].pos,
                    ],
                );
                let new_lt = self.new_local_tri(t_add, child_in_roi)?;

                // Queue any resulting edges > d_max that intersect the sphere
                // (or lie inside an intersecting coarse child triangle).
                let pc = self.verts[c as usize].pos;
                let keep_in_roi = sphere_triangle_intersects(
                    self.center,
                    self.r2,
                    [
                        self.verts[t_keep[0] as usize].pos,
                        self.verts[t_keep[1] as usize].pos,
                        self.verts[t_keep[2] as usize].pos,
                    ],
                );
                let check_edges = [
                    (a, m, pa, mid, keep_in_roi || child_in_roi),
                    (m, b, mid, pb, keep_in_roi || child_in_roi),
                    (m, c, mid, pc, keep_in_roi || child_in_roi),
                ];
                let _ = new_lt;
                for (u, v, pu, pv, tri_hits) in check_edges {
                    let ed2 = dist2(pu, pv);
                    if ed2 > self.d_max2
                        && (segment_intersects_sphere(pu, pv, self.center, self.r2) || tri_hits)
                    {
                        let (eu, ev) = if u < v { (u, v) } else { (v, u) };
                        heap.push(LongEdge {
                            len2_bits: ed2.to_bits(),
                            a: eu,
                            b: ev,
                        })?;
                    }
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Pass 2: Short-Edge Collapse
    // -----------------------------------------------------------------------

    fn run_collapses(&mut self, allow_chains: bool) -> Result<()> {
        if self.d_min2 <= 0.0 {
            return Ok(());
        }
        let mut cand = Vec::new();
        reserve(&mut cand, self.tris.len() * 3 + 16)?;
        for lt in 0..self.tris.len() {
            if !self.tris[lt].alive {
                continue;
            }
            let tri = self.tris[lt].v;
            for (u, v) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                if self.edge_collapsible_endpoints(u, v, allow_chains) {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    cand.push((a, b));
                }
            }
        }
        cand.sort_unstable();
        cand.dedup();

        let mut heap = MaxHeap::with_reserve(cand.len() + 32)?;
        for (a, b) in cand {
            let d2 = dist2(self.verts[a as usize].pos, self.verts[b as usize].pos);
            if d2 < self.d_min2 {
                heap.push(ShortEdge {
                    len2_bits: d2.to_bits(),
                    a,
                    b,
                })?;
            }
        }

        let mut n_a = Vec::new();
        let mut n_b = Vec::new();
        let mut common = Vec::new();
        let mut fan_tris = Vec::new();
        let mut post_tri_keys = Vec::new();
        reserve(&mut n_a, 16)?;
        reserve(&mut n_b, 16)?;
        reserve(&mut common, 16)?;
        reserve(&mut fan_tris, 32)?;
        reserve(&mut post_tri_keys, 32)?;

        while let Some(edge) = heap.pop() {
            let (a, b) = (edge.a, edge.b);
            if !self.edge_collapsible_endpoints(a, b, allow_chains) {
                continue;
            }
            let pa = self.verts[a as usize].pos;
            let pb = self.verts[b as usize].pos;
            if dist2(pa, pb) >= self.d_min2 {
                continue;
            }

            self.ensure_row_loaded(a)?;
            self.ensure_row_loaded(b)?;

            let EdgeOwners::Two(t0, t1) = self.edge_owners_manifold(a, b) else {
                continue;
            };
            let tv0 = self.tris[t0 as usize].v;
            let tv1 = self.tris[t1 as usize].v;
            if runs_directed(tv0, a, b) == runs_directed(tv1, a, b) {
                continue;
            }
            let (Some(c), Some(d)) = (apex_of(tv0, a, b), apex_of(tv1, a, b)) else {
                continue;
            };
            if c == d {
                continue;
            }

            // Lock boundary and non-manifold vertices (each endpoint must be a
            // strictly interior 2-manifold disk).
            let mut deg_a = 0usize;
            let mut deg_b = 0usize;
            if !self.is_interior_manifold_disk(a, &mut deg_a)
                || !self.is_interior_manifold_disk(b, &mut deg_b)
            {
                continue;
            }
            if deg_a < 3 || deg_b < 3 {
                continue;
            }

            // Also ensure c and d do not drop below manifold minimum degree 3 if interior.
            self.ensure_row_loaded(c)?;
            self.ensure_row_loaded(d)?;

            // Link condition: `N(a) ∩ N(b) == {c, d}`.
            self.collect_neighbors(a, &mut n_a)?;
            self.collect_neighbors(b, &mut n_b)?;
            common.clear();
            let (mut i, mut j) = (0usize, 0usize);
            while i < n_a.len() && j < n_b.len() {
                match n_a[i].cmp(&n_b[j]) {
                    Ordering::Less => i += 1,
                    Ordering::Greater => j += 1,
                    Ordering::Equal => {
                        if common.len() == common.capacity() {
                            reserve(&mut common, 8)?;
                        }
                        common.push(n_a[i]);
                        i += 1;
                        j += 1;
                    }
                }
            }
            let expected = if c < d { [c, d] } else { [d, c] };
            if common.as_slice() != expected {
                continue;
            }

            // Gather surviving union of incident triangles `(a.row ∪ b.row) \ {t0, t1}`.
            fan_tris.clear();
            for &lt in &self.verts[a as usize].row {
                if lt != t0 && lt != t1 && self.tris[lt as usize].alive {
                    if fan_tris.len() == fan_tris.capacity() {
                        reserve(&mut fan_tris, 16)?;
                    }
                    fan_tris.push(lt);
                }
            }
            for &lt in &self.verts[b as usize].row {
                if lt != t0 && lt != t1 && self.tris[lt as usize].alive {
                    if fan_tris.len() == fan_tris.capacity() {
                        reserve(&mut fan_tris, 16)?;
                    }
                    fan_tris.push(lt);
                }
            }
            fan_tris.sort_unstable();
            fan_tris.dedup();
            if fan_tris.len() < 3 {
                continue;
            }

            // Reject duplicate faces (e.g. tetrahedron collapse) & verify geometry.
            let mid = midpoint(pa, pb);
            if !point_in_sphere(mid, self.center, self.r2) {
                continue;
            }
            post_tri_keys.clear();
            let mut valid_geom = true;
            for &lt in &fan_tris {
                let tv = self.tris[lt as usize].v;
                let (n_before, area2_before) = tri_area2_normal(
                    self.verts[tv[0] as usize].pos,
                    self.verts[tv[1] as usize].pos,
                    self.verts[tv[2] as usize].pos,
                );
                let replaced = tv.map(|v| if v == a || v == b { u32::MAX } else { v });
                // Every corner must remain distinct after substituting {a,b} -> m.
                let q = replaced.map(|v| {
                    if v == u32::MAX {
                        mid
                    } else {
                        self.verts[v as usize].pos
                    }
                });
                let (n_after, area2_after) = tri_area2_normal(q[0], q[1], q[2]);
                if area2_after <= 1e-24 || area2_after < 1e-4 * area2_before {
                    valid_geom = false;
                    break;
                }
                let denom = (area2_before * area2_after).sqrt();
                if denom <= 0.0 || dot3(n_before, n_after) <= 0.05 * denom {
                    valid_geom = false;
                    break;
                }
                let mut key = replaced;
                key.sort_unstable();
                if key[0] == key[1] || key[1] == key[2] {
                    valid_geom = false;
                    break;
                }
                if post_tri_keys.len() == post_tri_keys.capacity() {
                    reserve(&mut post_tri_keys, 16)?;
                }
                post_tri_keys.push(key);
            }
            if !valid_geom {
                continue;
            }
            post_tri_keys.sort_unstable();
            if post_tri_keys.windows(2).any(|w| w[0] == w[1]) {
                continue;
            }

            // Apply collapse `(a, b) -> m`.
            let m = self.new_local_vert(mid, a, b, 0, false, true)?;
            self.collapses += 1;
            self.verts[a as usize].alive = false;
            self.verts[b as usize].alive = false;
            self.tris[t0 as usize].alive = false;
            self.tris[t1 as usize].alive = false;

            for &lt in &fan_tris {
                let tv = &mut self.tris[lt as usize].v;
                for slot in tv.iter_mut() {
                    if *slot == a || *slot == b {
                        *slot = m;
                    }
                }
                self.push_vert_row(m, lt)?;
            }

            if allow_chains {
                self.collect_neighbors(m, &mut n_a)?;
                for &w in &n_a {
                    if self.edge_collapsible_endpoints(m, w, true) {
                        let d2 = dist2(mid, self.verts[w as usize].pos);
                        if d2 < self.d_min2 {
                            let (eu, ev) = if m < w { (m, w) } else { (w, m) };
                            heap.push(ShortEdge {
                                len2_bits: d2.to_bits(),
                                a: eu,
                                b: ev,
                            })?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn edge_collapsible_endpoints(&self, a: u32, b: u32, allow_chains: bool) -> bool {
        if a == b {
            return false;
        }
        let va = &self.verts[a as usize];
        let vb = &self.verts[b as usize];
        if !va.alive || !vb.alive || va.pinned || vb.pinned {
            return false;
        }
        if va.fresh_from_split || vb.fresh_from_split {
            return false;
        }
        if !allow_chains && (va.fresh_from_collapse || vb.fresh_from_collapse) {
            return false;
        }
        point_in_sphere(va.pos, self.center, self.r2)
            && point_in_sphere(vb.pos, self.center, self.r2)
    }

    fn collect_neighbors(&self, v: u32, out: &mut Vec<u32>) -> Result<()> {
        out.clear();
        for &lt in &self.verts[v as usize].row {
            let tri = &self.tris[lt as usize];
            if !tri.alive {
                continue;
            }
            for &c in &tri.v {
                if c != v {
                    if out.len() == out.capacity() {
                        reserve(out, 16)?;
                    }
                    out.push(c);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        Ok(())
    }

    fn vertex_valence_and_target(&mut self, v: u32, nbrs: &mut Vec<u32>) -> Result<(i32, i32)> {
        self.ensure_row_loaded(v)?;
        let mut live_tris = 0i32;
        nbrs.clear();
        for &lt in &self.verts[v as usize].row {
            let tri = &self.tris[lt as usize];
            if !tri.alive {
                continue;
            }
            live_tris += 1;
            for &c in &tri.v {
                if c != v {
                    if nbrs.len() == nbrs.capacity() {
                        reserve(nbrs, nbrs.len().max(16))?;
                    }
                    nbrs.push(c);
                }
            }
        }
        nbrs.sort_unstable();
        nbrs.dedup();
        let val = nbrs.len() as i32;
        // Closed disk around `v` has #tris == #nbrs; boundary vertex has #tris < #nbrs.
        let target = if live_tris == val { 6 } else { 4 };
        Ok((val, target))
    }

    // -----------------------------------------------------------------------
    // Pass 3: Localized Valence Equalization & Delaunay Edge Flips
    // -----------------------------------------------------------------------

    fn run_flips(&mut self) -> Result<()> {
        let mut cand = Vec::new();
        reserve(&mut cand, self.tris.len() * 3 + 16)?;
        let mut nbrs = Vec::new();
        reserve(&mut nbrs, 32)?;

        for _pass in 0..2 {
            cand.clear();
            for lt in 0..self.tris.len() {
                if !self.tris[lt].alive {
                    continue;
                }
                let tri = self.tris[lt].v;
                for (u, v) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                    let vu = &self.verts[u as usize];
                    let vv = &self.verts[v as usize];
                    if vu.alive
                        && vv.alive
                        && !vu.pinned
                        && !vv.pinned
                        && point_in_sphere(vu.pos, self.center, self.r2)
                        && point_in_sphere(vv.pos, self.center, self.r2)
                    {
                        let (a, b) = if u < v { (u, v) } else { (v, u) };
                        cand.push((a, b));
                    }
                }
            }
            cand.sort_unstable();
            cand.dedup();

            let mut pass_flips = 0usize;
            for &(a, b) in &cand {
                if !self.verts[a as usize].alive || !self.verts[b as usize].alive {
                    continue;
                }
                self.ensure_row_loaded(a)?;
                self.ensure_row_loaded(b)?;

                let EdgeOwners::Two(t0, t1) = self.edge_owners_manifold(a, b) else {
                    continue;
                };
                let tv0 = self.tris[t0 as usize].v;
                let tv1 = self.tris[t1 as usize].v;
                if runs_directed(tv0, a, b) == runs_directed(tv1, a, b) {
                    continue;
                }
                let (Some(c), Some(d)) = (apex_of(tv0, a, b), apex_of(tv1, a, b)) else {
                    continue;
                };
                if c == d || c == a || c == b || d == a || d == b {
                    continue;
                }
                self.ensure_row_loaded(c)?;
                self.ensure_row_loaded(d)?;
                if self.has_edge(c, d) {
                    continue;
                }

                let pa = self.verts[a as usize].pos;
                let pb = self.verts[b as usize].pos;

                // Canonical orientation: `t0` runs `a -> b -> c`, `t1` runs `b -> a -> d`.
                let (t_abc, t_bad, p_tri0, p_tri1) = if runs_directed(tv0, a, b) {
                    let pc = self.verts[c as usize].pos;
                    let pd = self.verts[d as usize].pos;
                    (t0, t1, [pa, pb, pc], [pb, pa, pd])
                } else {
                    let pc = self.verts[c as usize].pos;
                    let pd = self.verts[d as usize].pos;
                    (t1, t0, [pa, pb, pd], [pb, pa, pc])
                };
                let (c_ord, d_ord) = if runs_directed(tv0, a, b) {
                    (c, d)
                } else {
                    (d, c)
                };
                let (pc_ord, pd_ord) = (
                    self.verts[c_ord as usize].pos,
                    self.verts[d_ord as usize].pos,
                );
                // Do not let valence flipping re-introduce edges above `d_max`.
                let new_d2 = dist2(pc_ord, pd_ord);
                if new_d2 > self.d_max2 && new_d2 > dist2(pa, pb) {
                    continue;
                }

                let (n0, a0_2) = tri_area2_normal(p_tri0[0], p_tri0[1], p_tri0[2]);
                let (n1, a1_2) = tri_area2_normal(p_tri1[0], p_tri1[1], p_tri1[2]);
                let avg_area2 = (a0_2 + a1_2) * 0.5;
                if a0_2 <= 1e-24 || a1_2 <= 1e-24 {
                    continue;
                }
                let dihedral_old = dot3(n0, n1) / (a0_2 * a1_2).sqrt();
                if dihedral_old < FLIP_SHARP_DOT {
                    continue;
                }

                // Candidate flipped triangles: `t_abc -> [a, d_ord, c_ord]`,
                // `t_bad -> [b, c_ord, d_ord]`.
                let q0 = [pa, pd_ord, pc_ord];
                let q1 = [pb, pc_ord, pd_ord];
                let (m0, b0_2) = tri_area2_normal(q0[0], q0[1], q0[2]);
                let (m1, b1_2) = tri_area2_normal(q1[0], q1[1], q1[2]);
                if b0_2 <= 1e-4 * avg_area2 || b1_2 <= 1e-4 * avg_area2 {
                    continue;
                }
                let n_avg = [n0[0] + n1[0], n0[1] + n1[1], n0[2] + n1[2]];
                let n_avg_len = dot3(n_avg, n_avg).sqrt();
                if n_avg_len <= 0.0 {
                    continue;
                }
                if dot3(m0, n_avg) <= 0.1 * b0_2.sqrt() * n_avg_len
                    || dot3(m1, n_avg) <= 0.1 * b1_2.sqrt() * n_avg_len
                {
                    continue;
                }
                if dot3(m0, m1) < FLIP_SHARP_DOT * (b0_2 * b1_2).sqrt() {
                    continue;
                }

                let (va, ta) = self.vertex_valence_and_target(a, &mut nbrs)?;
                let (vb, tb) = self.vertex_valence_and_target(b, &mut nbrs)?;
                let (vc, tc) = self.vertex_valence_and_target(c, &mut nbrs)?;
                let (vd, td) = self.vertex_valence_and_target(d, &mut nbrs)?;

                if (ta == 6 && va <= 3) || (tb == 6 && vb <= 3) || va <= 2 || vb <= 2 {
                    continue;
                }
                let dev_before =
                    (va - ta).abs() + (vb - tb).abs() + (vc - tc).abs() + (vd - td).abs();
                let dev_after = ((va - 1) - ta).abs()
                    + ((vb - 1) - tb).abs()
                    + ((vc + 1) - tc).abs()
                    + ((vd + 1) - td).abs();

                let strictly_better_valence = dev_after < dev_before;
                let equal_valence_delaunay = dev_after == dev_before && {
                    let min_before = min_sin2(p_tri0).min(min_sin2(p_tri1));
                    let min_after = min_sin2(q0).min(min_sin2(q1));
                    dist2(pc_ord, pd_ord) < dist2(pa, pb) * 0.99 && min_after > min_before * 1.05
                };
                if !strictly_better_valence && !equal_valence_delaunay {
                    continue;
                }

                // Commit flip in local workspace:
                // `t_abc: [a, b, c_ord] -> [a, d_ord, c_ord]` (loses `b`, gains `d_ord`)
                // `t_bad: [b, a, d_ord] -> [b, c_ord, d_ord]` (loses `a`, gains `c_ord`).
                self.tris[t_abc as usize].v = [a, d_ord, c_ord];
                self.tris[t_bad as usize].v = [b, c_ord, d_ord];
                self.remove_vert_row(b, t_abc);
                self.push_vert_row(d_ord, t_abc)?;
                self.remove_vert_row(a, t_bad);
                self.push_vert_row(c_ord, t_bad)?;
                self.flips += 1;
                pass_flips += 1;
            }
            if pass_flips == 0 {
                break;
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Pass 4: Localized Tangential Relaxation
    // -----------------------------------------------------------------------

    fn run_smooth_tangent(&mut self) -> Result<()> {
        let mut active = Vec::new();
        reserve(&mut active, self.verts.len())?;
        for lv in 0..self.verts.len() as u32 {
            let v = &self.verts[lv as usize];
            if v.alive && !v.pinned && point_in_sphere(v.pos, self.center, self.r2) {
                active.push(lv);
            }
        }
        // Snapshot pre-smooth positions for order-independent Jacobi updates.
        let mut targets = Vec::new();
        reserve(&mut targets, active.len())?;
        let mut nbrs = Vec::new();
        reserve(&mut nbrs, 16)?;
        let radius = self.r2.sqrt();

        for &lv in &active {
            self.ensure_row_loaded(lv)?;
            let mut deg = 0usize;
            if !self.is_interior_manifold_disk(lv, &mut deg) {
                continue;
            }
            let p_cur = self.verts[lv as usize].pos;
            let pc = f64v(p_cur);

            // Compute area-weighted smooth normal over incident fan.
            let mut n_sum = [0.0f64; 3];
            let mut fan_min_area2 = f64::INFINITY;
            for &lt in &self.verts[lv as usize].row {
                let tri = &self.tris[lt as usize];
                if !tri.alive {
                    continue;
                }
                let (n, a2) = tri_area2_normal(
                    self.verts[tri.v[0] as usize].pos,
                    self.verts[tri.v[1] as usize].pos,
                    self.verts[tri.v[2] as usize].pos,
                );
                n_sum[0] += n[0];
                n_sum[1] += n[1];
                n_sum[2] += n[2];
                fan_min_area2 = fan_min_area2.min(a2);
            }
            let n_len = dot3(n_sum, n_sum).sqrt();
            if n_len <= 1e-18 || fan_min_area2 <= 1e-24 {
                continue;
            }
            let n_unit = [n_sum[0] / n_len, n_sum[1] / n_len, n_sum[2] / n_len];

            self.collect_neighbors(lv, &mut nbrs)?;
            if nbrs.is_empty() {
                continue;
            }
            let mut centroid = [0.0f64; 3];
            for &w in &nbrs {
                let pw = f64v(self.verts[w as usize].pos);
                centroid[0] += pw[0];
                centroid[1] += pw[1];
                centroid[2] += pw[2];
            }
            let inv = 1.0 / (nbrs.len() as f64);
            let avg = [centroid[0] * inv, centroid[1] * inv, centroid[2] * inv];
            let disp = sub3(avg, pc);
            let proj = dot3(disp, n_unit);
            let tang = [
                disp[0] - proj * n_unit[0],
                disp[1] - proj * n_unit[1],
                disp[2] - proj * n_unit[2],
            ];
            let d = dot3(sub3(pc, self.center), sub3(pc, self.center)).sqrt();
            let u = (d / radius).clamp(0.0, 1.0);
            let falloff = (1.0 - u * u).powi(2);
            let weight = SMOOTH_TANGENT_STEP * falloff;
            if weight <= 0.0 {
                continue;
            }
            let candidate = [
                (pc[0] + weight * tang[0]) as f32,
                (pc[1] + weight * tang[1]) as f32,
                (pc[2] + weight * tang[2]) as f32,
            ];
            if !point_in_sphere(candidate, self.center, self.r2)
                || candidate.map(f32::to_bits) == p_cur.map(f32::to_bits)
            {
                continue;
            }

            // Validate that moving `lv -> candidate` keeps all fan triangles non-inverted.
            let mut ok = true;
            for &lt in &self.verts[lv as usize].row {
                let tri = &self.tris[lt as usize];
                if !tri.alive {
                    continue;
                }
                let old_pts = tri.v.map(|c| self.verts[c as usize].pos);
                let new_pts = tri.v.map(|c| {
                    if c == lv {
                        candidate
                    } else {
                        self.verts[c as usize].pos
                    }
                });
                let (nb, a2b) = tri_area2_normal(old_pts[0], old_pts[1], old_pts[2]);
                let (na, a2a) = tri_area2_normal(new_pts[0], new_pts[1], new_pts[2]);
                let denom = (a2b * a2a).sqrt();
                if a2a <= 1e-4 * a2b || denom <= 0.0 || dot3(nb, na) <= 0.1 * denom {
                    ok = false;
                    break;
                }
            }
            if ok {
                targets.push((lv, candidate));
            }
        }

        for (lv, candidate) in targets {
            let old_bits = self.verts[lv as usize].pos.map(f32::to_bits);
            if candidate.map(f32::to_bits) != old_bits {
                self.verts[lv as usize].pos = candidate;
                if self.verts[lv as usize].orig_id.is_some() {
                    self.verts[lv as usize].moved = true;
                }
                self.smoothed += 1;
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Atomic Reservation + Commit
    // -----------------------------------------------------------------------

    fn into_commit_plan(self) -> CommitPlan {
        CommitPlan {
            verts: self.verts,
            tris: self.tris,
            splits: self.splits,
            collapses: self.collapses,
            flips: self.flips,
            smoothed: self.smoothed,
        }
    }
}

struct CommitPlan {
    verts: Vec<LocalVert>,
    tris: Vec<LocalTri>,
    splits: usize,
    collapses: usize,
    flips: usize,
    smoothed: usize,
}

impl CommitPlan {
    fn resolve_parents_reserved(
        &self,
        lv: u32,
        queue: &mut Vec<(u32, f32)>,
        out: &mut Vec<(VertexId, f32)>,
    ) {
        queue.clear();
        out.clear();
        let Some([p0, p1]) = self.verts[lv as usize].parents else {
            return;
        };
        debug_assert!(queue.capacity() >= 2);
        queue.push((p0, 0.5));
        queue.push((p1, 0.5));

        while let Some((u, w)) = queue.pop() {
            let vu = &self.verts[u as usize];
            if let Some(orig) = vu.orig_id {
                debug_assert!(out.len() < out.capacity());
                out.push((orig, w));
            } else if vu.alive {
                let cid = vu
                    .committed_id
                    .expect("earlier live vertex already committed");
                debug_assert!(out.len() < out.capacity());
                out.push((cid, w));
            } else if let Some([gp0, gp1]) = vu.parents {
                debug_assert!(queue.len() + 2 <= queue.capacity());
                queue.push((gp0, w * 0.5));
                queue.push((gp1, w * 0.5));
            }
        }

        out.sort_unstable_by_key(|&(id, _)| id);
        let mut write = 0usize;
        for read in 0..out.len() {
            if write > 0 && out[write - 1].0 == out[read].0 {
                out[write - 1].1 += out[read].1;
            } else {
                out[write] = out[read];
                write += 1;
            }
        }
        out.truncate(write);
    }

    fn commit(mut self, tx: &mut Transaction<'_>) -> Result<DyntopoPatch> {
        if self.splits == 0 && self.collapses == 0 && self.flips == 0 && self.smoothed == 0 {
            return Ok(DyntopoPatch::default());
        }

        // Gather all deterministic commit specs & pre-reserve EVERY Vec BEFORE
        // the first write to `tx` so any OOM rolls back cleanly with zero writes.
        let mut add_vert_locals = Vec::new();
        let mut moved_base_locals = Vec::new();
        let mut dead_vert_ids = Vec::new();
        for lv in 0..self.verts.len() as u32 {
            let v = &self.verts[lv as usize];
            match v.orig_id {
                None if v.alive => push(&mut add_vert_locals, lv)?,
                Some(vid) if !v.alive => push(&mut dead_vert_ids, vid)?,
                Some(_) if v.moved => push(&mut moved_base_locals, lv)?,
                _ => {}
            }
        }

        let mut moved_ids = Vec::new();
        let mut moved_pos = Vec::new();
        reserve(&mut moved_ids, moved_base_locals.len())?;
        reserve(&mut moved_pos, moved_base_locals.len())?;
        for &lv in &moved_base_locals {
            let v = &self.verts[lv as usize];
            moved_ids.push(v.orig_id.unwrap());
            moved_pos.push(v.pos);
        }

        let mut mod_tri_locals = Vec::new();
        let mut new_tri_locals = Vec::new();
        let mut dead_tri_ids = Vec::new();
        for lt in 0..self.tris.len() as u32 {
            let t = &self.tris[lt as usize];
            match t.orig_id {
                None if t.alive => push(&mut new_tri_locals, lt)?,
                Some(tid) if !t.alive => push(&mut dead_tri_ids, tid)?,
                Some(_) if t.v != t.orig_corners => push(&mut mod_tri_locals, lt)?,
                _ => {}
            }
        }

        let mut mod_tri_ids = Vec::new();
        let mut mod_tri_corners = Vec::new();
        let mut new_tri_corners = Vec::new();
        reserve(&mut mod_tri_ids, mod_tri_locals.len())?;
        reserve(&mut mod_tri_corners, mod_tri_locals.len())?;
        reserve(&mut new_tri_corners, new_tri_locals.len())?;

        // Pre-reserve output patch buffers before writing.
        let mut patch = DyntopoPatch {
            splits: self.splits,
            collapses: self.collapses,
            flips: self.flips,
            smoothed: self.smoothed,
            ..DyntopoPatch::default()
        };
        reserve(&mut patch.vertices, add_vert_locals.len())?;
        reserve(&mut patch.new_vertices, add_vert_locals.len())?;
        reserve(&mut patch.removed_vertices, dead_vert_ids.len())?;
        reserve(&mut patch.removed_triangles, dead_tri_ids.len())?;
        reserve(&mut patch.created_triangles, new_tri_locals.len())?;
        reserve(&mut patch.modified_triangles, mod_tri_locals.len())?;

        let mut parent_queue = Vec::new();
        let mut parent_buf = Vec::new();
        reserve(&mut parent_queue, 16)?;
        reserve(&mut parent_buf, 16)?;

        for &lv in &add_vert_locals {
            let mut stack_count = 0usize;
            parent_queue.clear();
            if let Some([p0, p1]) = self.verts[lv as usize].parents {
                if parent_queue.capacity() < 8 {
                    let need = 8 - parent_queue.capacity();
                    reserve(&mut parent_queue, need)?;
                }
                parent_queue.push((p0, 0.5));
                parent_queue.push((p1, 0.5));
            }
            while let Some((u, w)) = parent_queue.pop() {
                let vu = &self.verts[u as usize];
                if vu.orig_id.is_some() || vu.alive {
                    stack_count += 1;
                } else if let Some([gp0, gp1]) = vu.parents {
                    if parent_queue.len() + 2 > parent_queue.capacity() {
                        let extra = parent_queue.len().max(8);
                        reserve(&mut parent_queue, extra)?;
                    }
                    parent_queue.push((gp0, w * 0.5));
                    parent_queue.push((gp1, w * 0.5));
                }
            }
            if stack_count > parent_buf.capacity() {
                let extra = stack_count - parent_buf.capacity();
                reserve(&mut parent_buf, extra)?;
            }
        }

        // Commit writes atomically while keeping topology-bound stroke channels.
        let mut inner_err: Option<OpsError> = None;
        let created_tris = tx.keeping_bound_channels(|tx| -> Option<Vec<TriangleId>> {
            for &lv in &add_vert_locals {
                self.resolve_parents_reserved(lv, &mut parent_queue, &mut parent_buf);
                let pos = self.verts[lv as usize].pos;
                let new_vid = match tx.add_vertex_with_parents(pos, &parent_buf) {
                    Ok(id) => id,
                    Err(e) => {
                        inner_err = Some(OpsError::Store(e));
                        return None;
                    }
                };
                self.verts[lv as usize].committed_id = Some(new_vid);
                patch.new_vertices.push(new_vid);
                if let Some([p0, p1]) = self.verts[lv as usize].parents {
                    let c0 = self.verts[p0 as usize]
                        .orig_id
                        .or(self.verts[p0 as usize].committed_id);
                    let c1 = self.verts[p1 as usize]
                        .orig_id
                        .or(self.verts[p1 as usize].committed_id);
                    if let (Some(id0), Some(id1)) = (c0, c1) {
                        patch.vertices.push([new_vid, id0, id1]);
                    }
                }
            }

            if !moved_ids.is_empty()
                && let Err(e) = tx.set_positions(&moved_ids, &moved_pos)
            {
                inner_err = Some(OpsError::Store(e));
                return None;
            }

            let resolve_vert = |local: u32| -> VertexId {
                let v = &self.verts[local as usize];
                v.orig_id
                    .or(v.committed_id)
                    .expect("live corner has committed vertex id")
            };

            for &lt in &mod_tri_locals {
                let t = &self.tris[lt as usize];
                let tid = t.orig_id.unwrap();
                mod_tri_ids.push(tid);
                mod_tri_corners.push(t.v.map(resolve_vert));
                patch.modified_triangles.push(tid);
            }
            if !mod_tri_ids.is_empty()
                && let Err(e) = tx.set_triangles(&mod_tri_ids, &mod_tri_corners)
            {
                inner_err = Some(OpsError::Store(e));
                return None;
            }

            for &lt in &new_tri_locals {
                new_tri_corners.push(self.tris[lt as usize].v.map(resolve_vert));
            }
            let added_tris = if !new_tri_corners.is_empty() {
                match tx.add_triangles(&new_tri_corners) {
                    Ok(ids) => ids,
                    Err(e) => {
                        inner_err = Some(OpsError::Store(e));
                        return None;
                    }
                }
            } else {
                Vec::new()
            };

            if !dead_tri_ids.is_empty()
                && let Err(e) = tx.remove_triangles(&dead_tri_ids)
            {
                inner_err = Some(OpsError::Store(e));
                return None;
            }
            if !dead_vert_ids.is_empty()
                && let Err(e) = tx.remove_vertices(&dead_vert_ids)
            {
                inner_err = Some(OpsError::Store(e));
                return None;
            }
            Some(added_tris)
        });

        let created_tris = match created_tris {
            Some(ids) => ids,
            None => return Err(inner_err.expect("error set when inner fails")),
        };

        patch.created_triangles.extend(created_tris);
        patch.removed_triangles.extend(dead_tri_ids);
        patch.removed_vertices.extend(dead_vert_ids);
        Ok(patch)
    }
}

#[inline]
fn next_around_corner(tri: [u32; 3], v: u32) -> Option<u32> {
    if tri[0] == v {
        Some(tri[1])
    } else if tri[1] == v {
        Some(tri[2])
    } else if tri[2] == v {
        Some(tri[0])
    } else {
        None
    }
}

fn apply_dyntopo_core(
    tx: &mut Transaction<'_>,
    center: [f32; 3],
    radius: f32,
    options: &DyntopoOptions,
    roi_triangles: &[TriangleId],
    pinned: &[VertexId],
    binary_parents_only: bool,
) -> Result<DyntopoPatch> {
    if roi_triangles.is_empty() {
        return Ok(DyntopoPatch::default());
    }
    let (d_min, d_max) = validate_dyntopo_inputs(center, radius, options)?;
    let allow_collapse_chains = !binary_parents_only;
    let mut work = Work::new(
        tx.mesh(),
        roi_triangles,
        pinned,
        center,
        radius,
        d_min,
        d_max,
    )?;

    work.run_splits()?;
    if options.mode.allows_collapse() && options.min_detail_ratio > 0.0 {
        work.run_collapses(allow_collapse_chains)?;
    }
    if options.equalize_valence {
        work.run_flips()?;
    }
    if options.smooth_tangent {
        work.run_smooth_tangent()?;
    }
    let plan = work.into_commit_plan();
    plan.commit(tx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{grid, parts, sweep, uv_sphere};

    #[test]
    fn oom_sweep_leaves_transaction_untouched_on_every_failure() {
        // Test split + collapse + flip + smooth OOM atomicity on a mixed grid.
        let mut mesh = grid(6);
        // Scale to 0..6 and shift to center (3, 3, 0), compressing one row so both
        // splits and collapses occur in the same dab.
        let pos: Vec<[f32; 3]> = mesh
            .positions()
            .iter()
            .map(|p| {
                let x = if (p[0] - 2.0).abs() < 0.01 { 1.1 } else { p[0] };
                [x, p[1], p[2]]
            })
            .collect();
        let ids: Vec<VertexId> = (0..pos.len() as VertexId).collect();
        let mut tx = mesh.begin_edit();
        tx.set_positions(&ids, &pos).unwrap();
        tx.commit().unwrap();

        let opts = DyntopoOptions {
            mode: DyntopoMode::SplitCollapse,
            detail_size: 1.0,
            min_detail_ratio: 0.4,
            max_detail_ratio: 1.33,
            equalize_valence: true,
            smooth_tangent: true,
        };
        let points = sweep(mesh, |tx| {
            apply_dyntopo(tx, [3.0, 3.0, 0.0], 2.2, &opts).map(|_| ())
        });
        assert!(points > 0);
    }

    #[test]
    fn small_input_does_not_spawn_rayon_and_is_deterministic() {
        use fast_mesh_store::par;
        let before = par::parallel_entries();
        let mut a = uv_sphere(8, 12, 2.0);
        let mut b = uv_sphere(8, 12, 2.0);
        let opts = DyntopoOptions::new(DyntopoMode::SplitCollapse, 0.25)
            .with_smooth_tangent(true)
            .with_equalize_valence(true);

        let r1 = {
            let mut tx = a.begin_edit();
            let r = apply_dyntopo(&mut tx, [0.0, 0.0, 2.0], 1.2, &opts).unwrap();
            tx.commit().unwrap();
            r
        };
        let r2 = {
            let mut tx = b.begin_edit();
            let r = apply_dyntopo(&mut tx, [0.0, 0.0, 2.0], 1.2, &opts).unwrap();
            tx.commit().unwrap();
            r
        };
        assert_eq!(par::parallel_entries(), before);
        assert_eq!(r1, r2);
        assert_eq!(parts(&a), parts(&b));
        assert!(r1.splits > 0 || r1.collapses > 0);
    }
}
