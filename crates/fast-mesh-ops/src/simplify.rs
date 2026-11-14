//! Mesh simplification by quadric-error edge collapse.
//!
//! [`simplify`] shrinks a mesh toward [`SimplifyOptions::target_triangles`]
//! by repeatedly collapsing the edge whose collapse adds the least quadric
//! error (Garland-Heckbert): each vertex holds the sum of the squared
//! distances to its incident face planes, an edge collapses to the point
//! minimizing the summed quadrics, and the vertex quadrics merge. Boundary
//! vertices stay put when [`SimplifyOptions::preserve_boundary`] is set, so
//! open rims keep their shape. Collapses that would break the link condition
//! or fold the fan are skipped. The whole plan is computed on working copies
//! first and committed through one [`Transaction`](fast_mesh_store::Transaction),
//! which keeps the call atomic under sweep: either every collapse lands or
//! nothing does.
//!
//! UVs ([`Mesh::uvs`](fast_mesh_store::Mesh::uvs)) follow: a survivor takes the
//! UV at its new position's parameter along the collapsed edge. The quadrics
//! ignore UVs, so textures can stretch. With `preserve_boundary`, seams (which
//! are boundaries) stay exactly as they are.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use fast_mesh_store::{Transaction, TriangleId, UV_CHANNEL, VertexId};

use crate::{OpsError, Result, filled, push, reserve};

/// The parameter along `a`-`b` of the point nearest to `p`, in `[0, 1]`; the
/// middle of a zero-length edge.
fn edge_param(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f32 {
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    if len2 > 0.0 {
        let along = (p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1] + (p[2] - a[2]) * d[2];
        (along / len2).clamp(0.0, 1.0) as f32
    } else {
        0.5
    }
}

/// Options for [`simplify`].
///
/// For attribute-aware, error-bounded, ratio-based, or crease-locked
/// decimation, use [`DecimateOptions`] with [`simplify_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimplifyOptions {
    /// Stop once this many triangles are alive. Must be at least 1; a target
    /// at or above the live count leaves the mesh alone.
    pub target_triangles: usize,
    /// Keep boundary vertices in place: no edge touching a boundary vertex
    /// collapses. Default `true`, so open rims keep their shape.
    pub preserve_boundary: bool,
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        SimplifyOptions {
            target_triangles: 0,
            preserve_boundary: true,
        }
    }
}

impl SimplifyOptions {
    /// A builder for [`SimplifyOptions`].
    pub fn builder() -> SimplifyOptionsBuilder {
        SimplifyOptionsBuilder::default()
    }
}

/// Builder for [`SimplifyOptions`].
#[derive(Debug, Clone, Default)]
pub struct SimplifyOptionsBuilder {
    options: SimplifyOptions,
}

impl SimplifyOptionsBuilder {
    /// Stop once this many triangles are alive. Required: the default 0 is
    /// rejected by [`simplify`].
    pub fn target_triangles(mut self, target: usize) -> Self {
        self.options.target_triangles = target;
        self
    }

    /// Keep boundary vertices in place.
    pub fn preserve_boundary(mut self, preserve: bool) -> Self {
        self.options.preserve_boundary = preserve;
        self
    }

    /// The options as set.
    pub fn build(self) -> SimplifyOptions {
        self.options
    }
}

/// Extended options for [`simplify_with`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecimateOptions {
    /// Stop once this many triangles are alive (used when `target_ratio` is `None`).
    /// Must be at least 1 unless `target_ratio` or `max_deviation` is set.
    pub target_triangles: usize,
    /// Target fraction of input live triangles in `(0.0, 1.0]`. When `Some(r)`,
    /// resolves the triangle budget to `(live * r).round().max(1.0) as usize`.
    pub target_ratio: Option<f32>,
    /// Maximum allowed geometric root-quadric deviation. Collapses whose
    /// geometric quadric error exceeds `delta^2` are stopped/skipped cleanly.
    pub max_deviation: Option<f64>,
    /// Keep boundary vertices in place: no edge touching a boundary vertex
    /// collapses. Default `true`, so open rims keep their shape.
    pub preserve_boundary: bool,
    /// Use Hoppe (1999) appearance-aware UV quadrics, protect UV chart borders,
    /// and block collapses that fold 2D UV triangles. Default `false`.
    pub preserve_uvs: bool,
    /// Penalize perpendicular displacement off sharp dihedral edges exceeding
    /// [`DecimateOptions::crease_angle_deg`] and lock crease corners. Default `false`.
    pub lock_creases: bool,
    /// Dihedral angle threshold in degrees (`[0.0, 180.0]`) for crease detection
    /// when [`DecimateOptions::lock_creases`] is `true`. Default `30.0`.
    pub crease_angle_deg: f32,
    /// Reject collapses that invert any 1-ring incident triangle normal by 90°
    /// or more. Default `true`.
    pub prevent_normal_inversion: bool,
}

/// Type alias for [`DecimateOptions`].
pub type SimplifyExtendedOptions = DecimateOptions;

impl Default for DecimateOptions {
    fn default() -> Self {
        DecimateOptions {
            target_triangles: 0,
            target_ratio: None,
            max_deviation: None,
            preserve_boundary: true,
            preserve_uvs: false,
            lock_creases: false,
            crease_angle_deg: 30.0,
            prevent_normal_inversion: true,
        }
    }
}

impl From<SimplifyOptions> for DecimateOptions {
    fn from(o: SimplifyOptions) -> Self {
        DecimateOptions {
            target_triangles: o.target_triangles,
            preserve_boundary: o.preserve_boundary,
            ..DecimateOptions::default()
        }
    }
}

impl DecimateOptions {
    /// A builder for [`DecimateOptions`].
    pub fn builder() -> DecimateOptionsBuilder {
        DecimateOptionsBuilder::default()
    }
}

/// Builder for [`DecimateOptions`].
#[derive(Debug, Clone, Default)]
pub struct DecimateOptionsBuilder {
    options: DecimateOptions,
}

impl DecimateOptionsBuilder {
    /// Stop once this many triangles are alive (when `target_ratio` is `None`).
    pub fn target_triangles(mut self, target: usize) -> Self {
        self.options.target_triangles = target;
        self
    }

    /// Fraction of live triangles to keep, in `(0.0, 1.0]`.
    pub fn target_ratio(mut self, ratio: f32) -> Self {
        self.options.target_ratio = Some(ratio);
        self
    }

    /// Maximum allowed geometric root-quadric error in mesh units.
    pub fn max_deviation(mut self, max_deviation: f64) -> Self {
        self.options.max_deviation = Some(max_deviation);
        self
    }

    /// Keep boundary vertices in place.
    pub fn preserve_boundary(mut self, preserve: bool) -> Self {
        self.options.preserve_boundary = preserve;
        self
    }

    /// Enable Hoppe (1999) appearance-aware UV quadrics and anti-tear checks.
    pub fn preserve_uvs(mut self, preserve: bool) -> Self {
        self.options.preserve_uvs = preserve;
        self
    }

    /// Lock sharp crease edges/corners and restrict ridge sliding along creases.
    pub fn lock_creases(mut self, lock: bool) -> Self {
        self.options.lock_creases = lock;
        self
    }

    /// Dihedral crease angle threshold in degrees (`[0.0, 180.0]`).
    pub fn crease_angle_deg(mut self, deg: f32) -> Self {
        self.options.crease_angle_deg = deg;
        self
    }

    /// Reject 1-ring triangle normal inversions (`> 90°`) on collapse.
    pub fn prevent_normal_inversion(mut self, prevent: bool) -> Self {
        self.options.prevent_normal_inversion = prevent;
        self
    }

    /// The options as set.
    pub fn build(self) -> DecimateOptions {
        self.options
    }
}

/// Outcome of [`simplify`] and [`simplify_with`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SimplifyReport {
    /// Edge collapses performed.
    pub collapsed: usize,
    /// Triangles removed (one per boundary collapse, two per interior one).
    pub triangles_removed: usize,
    /// Vertices removed (one per collapse).
    pub vertices_removed: usize,
    /// Maximum root-quadric geometric error observed across accepted collapses.
    pub max_deviation_achieved: f64,
}

/// A working triangle: `id` is its slot in the store.
struct WorkTri {
    id: TriangleId,
    corners: [u32; 3],
    alive: bool,
}

/// A collapse candidate, ordered cheapest-first in the max-heap.
struct HeapEntry {
    cost: f64,
    a: u32,
    b: u32,
    ver_a: u32,
    ver_b: u32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
            && self.a == other.a
            && self.b == other.b
            && self.ver_a == other.ver_a
            && self.ver_b == other.ver_b
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` pops the greatest: rank cheaper edges greater, with
        // the vertex ids and versions breaking ties deterministically.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.a.cmp(&self.a))
            .then_with(|| other.b.cmp(&self.b))
            .then_with(|| other.ver_a.cmp(&self.ver_a))
            .then_with(|| other.ver_b.cmp(&self.ver_b))
    }
}

/// A symmetric 4x4 quadric as its upper triangle:
/// `q00 q01 q02 q03 q11 q12 q13 q22 q23 q33`.
type Quadric = [f64; 10];

fn f64v(p: &[f32; 3]) -> [f64; 3] {
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Weighted quadric of any unit plane `n . p + d = 0` (`d = -n . p0`).
fn plane_quadric(n: [f64; 3], p0: [f64; 3], weight: f64) -> Quadric {
    let d = -dot3(n, p0);
    if !d.is_finite() {
        return [0.0; 10];
    }
    outer_quadric([n[0], n[1], n[2], d], weight)
}

/// Weighted symmetric outer product `weight * v * v^T` as upper triangle.
fn outer_quadric(v: [f64; 4], weight: f64) -> Quadric {
    let mut q = [0.0; 10];
    let mut k = 0;
    for i in 0..4 {
        for j in i..4 {
            let val = weight * v[i] * v[j];
            if !val.is_finite() {
                return [0.0; 10];
            }
            q[k] = val;
            k += 1;
        }
    }
    q
}

/// The quadric of one face plane: squared distance to it.
fn face_quadric(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> Quadric {
    let n = cross3(sub3(p1, p0), sub3(p2, p0));
    let len = dot3(n, n).sqrt();
    if !len.is_finite() || len == 0.0 {
        return [0.0; 10];
    }
    let n = [n[0] / len, n[1] / len, n[2] / len];
    if !n.iter().all(|c| c.is_finite()) {
        return [0.0; 10];
    }
    plane_quadric(n, p0, 1.0)
}

fn add_quadric(dst: &mut Quadric, src: &Quadric) {
    for k in 0..10 {
        dst[k] += src[k];
    }
}

/// Additive Hoppe (1999) appearance-attribute accumulator for scalar `(u, v)`
/// channels, Schur-complemented back to a 3D spatial quadric over `p`.
#[derive(Clone, Copy, Default)]
struct AttrQuadric {
    w: f64,
    gu: [f64; 3],
    du: f64,
    gv: [f64; 3],
    dv: f64,
    q_lin: Quadric,
}

impl AttrQuadric {
    fn from_face(p: [[f64; 3]; 3], uv: [[f64; 2]; 3], weight: f64) -> Self {
        if !weight.is_finite() || weight <= 0.0 {
            return Self::default();
        }
        let e1 = sub3(p[1], p[0]);
        let e2 = sub3(p[2], p[0]);
        let n = cross3(e1, e2);
        let n2 = dot3(n, n);
        if !n2.is_finite() || n2 <= 1e-24 {
            return Self::default();
        }
        let c1 = cross3(e2, n);
        let c2 = cross3(n, e1);
        let g1 = [c1[0] / n2, c1[1] / n2, c1[2] / n2];
        let g2 = [c2[0] / n2, c2[1] / n2, c2[2] / n2];

        let du1 = uv[1][0] - uv[0][0];
        let du2 = uv[2][0] - uv[0][0];
        let dv1 = uv[1][1] - uv[0][1];
        let dv2 = uv[2][1] - uv[0][1];
        let gu = [
            du1 * g1[0] + du2 * g2[0],
            du1 * g1[1] + du2 * g2[1],
            du1 * g1[2] + du2 * g2[2],
        ];
        let gv = [
            dv1 * g1[0] + dv2 * g2[0],
            dv1 * g1[1] + dv2 * g2[1],
            dv1 * g1[2] + dv2 * g2[2],
        ];
        let du = uv[0][0] - dot3(gu, p[0]);
        let dv = uv[0][1] - dot3(gv, p[0]);
        if !gu.iter().all(|c| c.is_finite())
            || !gv.iter().all(|c| c.is_finite())
            || !du.is_finite()
            || !dv.is_finite()
        {
            return Self::default();
        }
        let qu = outer_quadric([gu[0], gu[1], gu[2], du], weight);
        let qv = outer_quadric([gv[0], gv[1], gv[2], dv], weight);
        let mut q_lin = qu;
        add_quadric(&mut q_lin, &qv);
        Self {
            w: weight,
            gu: [gu[0] * weight, gu[1] * weight, gu[2] * weight],
            du: du * weight,
            gv: [gv[0] * weight, gv[1] * weight, gv[2] * weight],
            dv: dv * weight,
            q_lin,
        }
    }

    fn merge(&self, other: &Self) -> Self {
        let mut q_lin = self.q_lin;
        add_quadric(&mut q_lin, &other.q_lin);
        Self {
            w: self.w + other.w,
            gu: [
                self.gu[0] + other.gu[0],
                self.gu[1] + other.gu[1],
                self.gu[2] + other.gu[2],
            ],
            du: self.du + other.du,
            gv: [
                self.gv[0] + other.gv[0],
                self.gv[1] + other.gv[1],
                self.gv[2] + other.gv[2],
            ],
            dv: self.dv + other.dv,
            q_lin,
        }
    }

    /// Schur complement reducing the 5D `(x, y, z, u, v)` quadric into 3D.
    fn schur_3d(&self) -> Quadric {
        if !self.w.is_finite() || self.w <= 1e-24 {
            return [0.0; 10];
        }
        let inv_w = 1.0 / self.w;
        let sub_u = outer_quadric([self.gu[0], self.gu[1], self.gu[2], self.du], inv_w);
        let sub_v = outer_quadric([self.gv[0], self.gv[1], self.gv[2], self.dv], inv_w);
        let mut out = self.q_lin;
        for k in 0..10 {
            out[k] -= sub_u[k] + sub_v[k];
        }
        out
    }
}

fn cross2d(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f64 {
    let abx = f64::from(b[0]) - f64::from(a[0]);
    let aby = f64::from(b[1]) - f64::from(a[1]);
    let acx = f64::from(c[0]) - f64::from(a[0]);
    let acy = f64::from(c[1]) - f64::from(a[1]);
    abx * acy - aby * acx
}

/// The quadric error of `p`: `p^T A p + 2 b^T p + q33`.
fn quad_error(q: &Quadric, p: [f64; 3]) -> f64 {
    let [x, y, z] = p;
    x * x * q[0]
        + 2.0 * x * y * q[1]
        + 2.0 * x * z * q[2]
        + 2.0 * x * q[3]
        + y * y * q[4]
        + 2.0 * y * z * q[5]
        + 2.0 * y * q[6]
        + z * z * q[7]
        + 2.0 * z * q[8]
        + q[9]
}

/// The point minimizing `q` and its error: the linear solve, or the midpoint
/// when the 3x3 block is singular or the solve is not finite.
fn optimal_position(q: &Quadric, a: [f64; 3], b: [f64; 3]) -> ([f64; 3], f64) {
    let (a00, a01, a02, b0) = (q[0], q[1], q[2], q[3]);
    let (a11, a12, b1) = (q[4], q[5], q[6]);
    let (a22, b2) = (q[7], q[8]);
    let det = a00 * (a11 * a22 - a12 * a12) - a01 * (a01 * a22 - a12 * a02)
        + a02 * (a01 * a12 - a11 * a02);
    if det.is_finite() && det.abs() > 1e-12 {
        // Adjugate over the determinant (Cramer on `A x = -b`).
        let inv = [
            (a11 * a22 - a12 * a12) / det,
            (a02 * a12 - a01 * a22) / det,
            (a01 * a12 - a11 * a02) / det,
            (a02 * a12 - a01 * a22) / det,
            (a00 * a22 - a02 * a02) / det,
            (a02 * a01 - a00 * a12) / det,
            (a01 * a12 - a11 * a02) / det,
            (a02 * a01 - a00 * a12) / det,
            (a00 * a11 - a01 * a01) / det,
        ];
        // Symmetric: inv01 == inv10 and so on; index the upper triangle.
        let x = -(inv[0] * b0 + inv[1] * b1 + inv[2] * b2);
        let y = -(inv[1] * b0 + inv[4] * b1 + inv[5] * b2);
        let z = -(inv[2] * b0 + inv[5] * b1 + inv[8] * b2);
        let p = [x, y, z];
        if p.iter().all(|c| c.is_finite()) {
            let cost = quad_error(q, p);
            if cost.is_finite() && cost >= -1e-9 * (1.0 + q[9].abs()) {
                return (p, cost.max(0.0));
            }
        }
    }
    let mid = [
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
        (a[2] + b[2]) / 2.0,
    ];
    let cost = quad_error(q, mid);
    if cost.is_finite() {
        (mid, cost.max(0.0))
    } else {
        // Degenerate quadrics (huge coordinates): fall back to the squared
        // edge length so the ordering stays finite and shortest-first.
        let d = sub3(a, b);
        (mid, dot3(d, d))
    }
}

/// The undirected key of the edge `a`-`b`.
fn key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// Whether `corners` carries both `a` and `b`.
fn carries(corners: &[u32; 3], a: u32, b: u32) -> bool {
    corners.contains(&a) && corners.contains(&b)
}

/// The corner of a triangle carrying `(a, b)` that is neither endpoint.
fn apex_of(corners: &[u32; 3], a: u32, b: u32) -> u32 {
    if corners[0] != a && corners[0] != b {
        corners[0]
    } else if corners[1] != a && corners[1] != b {
        corners[1]
    } else {
        corners[2]
    }
}

/// Reusable scratch buffers for edge collapses inside [`simplify_with`].
#[derive(Default)]
struct CollapseScratch {
    owners: Vec<usize>,
    na: Vec<u32>,
    nb: Vec<u32>,
    apexes: Vec<u32>,
    common: Vec<u32>,
    fan: Vec<usize>,
    v2t_drop: Vec<u32>,
    ring: Vec<u32>,
    tri_keys: Vec<[u32; 3]>,
}

/// Fills `owners` with the sorted, unique alive working triangles carrying `(a, b)`.
fn fill_owners(
    a: u32,
    b: u32,
    v2t: &[Vec<u32>],
    tris: &[WorkTri],
    owners: &mut Vec<usize>,
) -> Result<()> {
    owners.clear();
    reserve(owners, 4)?;
    for &i in &v2t[a as usize] {
        let t = &tris[i as usize];
        if t.alive && carries(&t.corners, a, b) {
            push(owners, i as usize)?;
        }
    }
    owners.sort_unstable();
    owners.dedup();
    Ok(())
}

/// Fills `out` with the sorted, unique alive vertices sharing an alive triangle with `v`.
fn fill_neighbors(v: u32, v2t: &[Vec<u32>], tris: &[WorkTri], out: &mut Vec<u32>) -> Result<()> {
    out.clear();
    reserve(out, 16)?;
    for &i in &v2t[v as usize] {
        let t = &tris[i as usize];
        if !t.alive || !t.corners.contains(&v) {
            continue;
        }
        for &w in &t.corners {
            if w != v {
                push(out, w)?;
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(())
}

fn tri_normal(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> [f64; 3] {
    cross3(sub3(p1, p0), sub3(p2, p0))
}

fn sorted_triple(mut c: [u32; 3]) -> [u32; 3] {
    if c[0] > c[1] {
        c.swap(0, 1);
    }
    if c[1] > c[2] {
        c.swap(1, 2);
    }
    if c[0] > c[1] {
        c.swap(0, 1);
    }
    c
}

const CREASE_PENALTY_WEIGHT: f64 = 100.0;

fn validate_decimate(options: &DecimateOptions) -> Result<()> {
    if let Some(r) = options.target_ratio
        && (!r.is_finite() || r <= 0.0 || r > 1.0)
    {
        return Err(OpsError::InvalidInput(
            "simplify target_ratio must be finite and in (0, 1]",
        ));
    }
    if let Some(d) = options.max_deviation
        && (!d.is_finite() || d < 0.0)
    {
        return Err(OpsError::InvalidInput(
            "simplify max_deviation must be finite and non-negative",
        ));
    }
    if !options.crease_angle_deg.is_finite() || !(0.0..=180.0).contains(&options.crease_angle_deg) {
        return Err(OpsError::InvalidInput(
            "simplify crease_angle_deg must be finite and in [0, 180]",
        ));
    }
    if options.target_triangles == 0
        && options.target_ratio.is_none()
        && options.max_deviation.is_none()
    {
        return Err(OpsError::InvalidInput(
            "simplify target_triangles must be at least 1",
        ));
    }
    Ok(())
}

/// Evaluates collapse candidate `(a, b)`, respecting boundary, crease, and UV seam locks.
/// Returns `(target_pos_f64, heap_cost, geom_cost)` or `None` when forbidden by constraints.
#[allow(clippy::too_many_arguments)]
fn candidate_solve(
    a: u32,
    b: u32,
    options: &DecimateOptions,
    has_uvs: bool,
    positions: &[[f32; 3]],
    is_boundary: &[bool],
    vert_crease: &[u8],
    crease_adj: &[Vec<u32>],
    quadrics: &[Quadric],
    aux_quadrics: Option<&[Quadric]>,
    attr_quadrics: Option<&[AttrQuadric]>,
) -> Option<([f64; 3], f64, f64)> {
    let ba = is_boundary[a as usize];
    let bb = is_boundary[b as usize];
    if options.preserve_boundary && (ba || bb) {
        return None;
    }

    let mut forced: Option<[f64; 3]> = None;
    let pa = f64v(&positions[a as usize]);
    let pb = f64v(&positions[b as usize]);

    if options.lock_creases {
        let ca = vert_crease[a as usize];
        let cb = vert_crease[b as usize];
        let is_ce = crease_adj[a as usize].contains(&b);
        if ca == 2 && cb == 2 {
            return None;
        }
        if ca >= 1 && cb >= 1 && !is_ce {
            return None;
        }
        if ca == 2 || (ca >= 1 && cb == 0) {
            forced = Some(pa);
        } else if cb == 2 || (cb >= 1 && ca == 0) {
            forced = Some(pb);
        }
    }

    if forced.is_none() && options.preserve_uvs && has_uvs && ba != bb {
        forced = Some(if ba { pa } else { pb });
    }

    let mut q_geom = quadrics[a as usize];
    add_quadric(&mut q_geom, &quadrics[b as usize]);
    let mut q_total = q_geom;
    if let Some(aux) = aux_quadrics {
        add_quadric(&mut q_total, &aux[a as usize]);
        add_quadric(&mut q_total, &aux[b as usize]);
    }
    if let Some(attr) = attr_quadrics {
        let merged = attr[a as usize].merge(&attr[b as usize]).schur_3d();
        add_quadric(&mut q_total, &merged);
    }

    let (pnew, heap_cost) = if let Some(fp) = forced {
        let cost = quad_error(&q_total, fp);
        let c = if cost.is_finite() {
            cost.max(0.0)
        } else {
            let d = sub3(pa, pb);
            dot3(d, d)
        };
        (fp, c)
    } else {
        optimal_position(&q_total, pa, pb)
    };
    let raw_geom = quad_error(&q_geom, pnew);
    let geom_cost = if raw_geom.is_finite() {
        raw_geom.max(0.0)
    } else {
        heap_cost
    };
    Some((pnew, heap_cost, geom_cost))
}

/// Shrinks the mesh toward `options.target_triangles` by quadric-error edge
/// collapses (see the module docs). Boundary vertices stay put when
/// `preserve_boundary` is set. Stops early when no valid collapse is left,
/// which can leave more triangles than the target on meshes with locked or
/// non-manifold edges.
///
/// A target at or above the live count, and an empty mesh, succeed without
/// writing. Otherwise at least one collapse runs and the patch is non-empty.
///
/// # Errors
/// [`OpsError::InvalidInput`] for a zero target; [`OpsError::Store`] when
/// memory runs out.
pub fn simplify(tx: &mut Transaction, options: &SimplifyOptions) -> Result<SimplifyReport> {
    simplify_with(tx, &DecimateOptions::from(*options))
}

/// Shrinks the mesh according to [`DecimateOptions`] with optional ratio target,
/// geometric error bound, Hoppe UV preservation, crease locking, and fold checks.
///
/// # Errors
/// [`OpsError::InvalidInput`] on invalid options; [`OpsError::Store`] on OOM.
pub fn simplify_with(tx: &mut Transaction, options: &DecimateOptions) -> Result<SimplifyReport> {
    validate_decimate(options)?;
    let mesh = tx.mesh();
    let base = mesh.positions();
    let mut positions: Vec<[f32; 3]> = Vec::new();
    reserve(&mut positions, base.len())?;
    positions.extend_from_slice(base);
    let slots = positions.len();
    let mut was_alive = filled(slots, false)?;
    for (v, alive) in was_alive.iter_mut().enumerate() {
        *alive = mesh.is_vertex_alive(v as VertexId);
    }

    let mut uvs: Option<Vec<f32>> = None;
    if let Some(from) = mesh.uvs() {
        let mut copy = Vec::new();
        reserve(&mut copy, from.len())?;
        copy.extend_from_slice(from);
        uvs = Some(copy);
    }
    let mut tris: Vec<WorkTri> = Vec::new();
    reserve(&mut tris, mesh.triangle_slots())?;
    for t in 0..mesh.triangle_slots() as u32 {
        let Some(corners) = mesh.triangle(t) else {
            continue;
        };
        if corners[0] == corners[1] || corners[1] == corners[2] || corners[2] == corners[0] {
            continue;
        }
        push(
            &mut tris,
            WorkTri {
                id: t,
                corners,
                alive: true,
            },
        )?;
    }
    let mut live = tris.len();
    let target = if let Some(r) = options.target_ratio {
        ((live as f64) * f64::from(r)).round().max(1.0) as usize
    } else if options.target_triangles > 0 {
        options.target_triangles
    } else {
        1
    };
    if live <= target {
        return Ok(SimplifyReport::default());
    }

    let mut quadrics: Vec<Quadric> = Vec::new();
    reserve(&mut quadrics, slots)?;
    quadrics.resize(slots, [0.0; 10]);
    for tri in &tris {
        let c = tri.corners;
        let q = face_quadric(
            f64v(&positions[c[0] as usize]),
            f64v(&positions[c[1] as usize]),
            f64v(&positions[c[2] as usize]),
        );
        for &v in &c {
            add_quadric(&mut quadrics[v as usize], &q);
        }
    }

    // Collect undirected edges paired with owning working triangle indices.
    let mut edge_tris: Vec<((u32, u32), u32)> = Vec::new();
    reserve(&mut edge_tris, 3 * tris.len())?;
    for (ti, tri) in tris.iter().enumerate() {
        let c = tri.corners;
        for k in 0..3 {
            push(&mut edge_tris, (key(c[k], c[(k + 1) % 3]), ti as u32))?;
        }
    }
    edge_tris.sort_unstable();

    let mut pairs: Vec<(u32, u32)> = Vec::new();
    reserve(&mut pairs, edge_tris.len() / 2 + 1)?;
    let mut is_boundary = filled(slots, false)?;
    let mut aux_quadrics: Option<Vec<Quadric>> = if options.lock_creases {
        let mut aq = Vec::new();
        reserve(&mut aq, slots)?;
        aq.resize(slots, [0.0; 10]);
        Some(aq)
    } else {
        None
    };
    let mut crease_adj: Vec<Vec<u32>> = Vec::new();
    if options.lock_creases {
        reserve(&mut crease_adj, slots)?;
        crease_adj.resize_with(slots, Vec::new);
    }
    let cos_crease = f64::from(options.crease_angle_deg).to_radians().cos();

    {
        let mut start = 0;
        while start < edge_tris.len() {
            let e = edge_tris[start].0;
            let mut end = start + 1;
            while end < edge_tris.len() && edge_tris[end].0 == e {
                end += 1;
            }
            push(&mut pairs, e)?;
            let count = end - start;
            if count == 1 {
                is_boundary[e.0 as usize] = true;
                is_boundary[e.1 as usize] = true;
            } else if options.lock_creases {
                let mut is_sharp = count > 2;
                if count == 2 {
                    let t0 = tris[edge_tris[start].1 as usize].corners;
                    let t1 = tris[edge_tris[start + 1].1 as usize].corners;
                    let n0 = tri_normal(
                        f64v(&positions[t0[0] as usize]),
                        f64v(&positions[t0[1] as usize]),
                        f64v(&positions[t0[2] as usize]),
                    );
                    let n1 = tri_normal(
                        f64v(&positions[t1[0] as usize]),
                        f64v(&positions[t1[1] as usize]),
                        f64v(&positions[t1[2] as usize]),
                    );
                    let l0 = dot3(n0, n0).sqrt();
                    let l1 = dot3(n1, n1).sqrt();
                    if l0 > 0.0 && l1 > 0.0 {
                        let c = (dot3(n0, n1) / (l0 * l1)).clamp(-1.0, 1.0);
                        is_sharp = c < cos_crease - 1e-12;
                    }
                }
                if is_sharp {
                    push(&mut crease_adj[e.0 as usize], e.1)?;
                    push(&mut crease_adj[e.1 as usize], e.0)?;
                    let p0 = f64v(&positions[e.0 as usize]);
                    let p1 = f64v(&positions[e.1 as usize]);
                    let edir = sub3(p1, p0);
                    let elen = dot3(edir, edir).sqrt();
                    if elen > 1e-12
                        && let Some(aq) = &mut aux_quadrics
                    {
                        let d_unit = [edir[0] / elen, edir[1] / elen, edir[2] / elen];
                        for &(_, ti) in &edge_tris[start..end] {
                            let tc = tris[ti as usize].corners;
                            let fn_raw = tri_normal(
                                f64v(&positions[tc[0] as usize]),
                                f64v(&positions[tc[1] as usize]),
                                f64v(&positions[tc[2] as usize]),
                            );
                            let fn_len = dot3(fn_raw, fn_raw).sqrt();
                            if fn_len > 1e-12 {
                                let nf =
                                    [fn_raw[0] / fn_len, fn_raw[1] / fn_len, fn_raw[2] / fn_len];
                                let perp_raw = cross3(d_unit, nf);
                                let pl = dot3(perp_raw, perp_raw).sqrt();
                                if pl > 1e-12 {
                                    let pn = [perp_raw[0] / pl, perp_raw[1] / pl, perp_raw[2] / pl];
                                    let pq = plane_quadric(pn, p0, CREASE_PENALTY_WEIGHT);
                                    add_quadric(&mut aq[e.0 as usize], &pq);
                                    add_quadric(&mut aq[e.1 as usize], &pq);
                                }
                            }
                        }
                    }
                }
            }
            start = end;
        }
    }

    let mut vert_crease = filled(slots, 0u8)?;
    if options.lock_creases {
        for v in 0..slots {
            crease_adj[v].sort_unstable();
            crease_adj[v].dedup();
            let deg = crease_adj[v].len();
            if deg == 1 || deg >= 3 {
                vert_crease[v] = 2;
            } else if deg == 2 {
                let pv = f64v(&positions[v]);
                let p0 = f64v(&positions[crease_adj[v][0] as usize]);
                let p1 = f64v(&positions[crease_adj[v][1] as usize]);
                let d0 = sub3(p0, pv);
                let d1 = sub3(p1, pv);
                let l0 = dot3(d0, d0).sqrt();
                let l1 = dot3(d1, d1).sqrt();
                let sharp_turn = if l0 > 1e-12 && l1 > 1e-12 {
                    dot3(d0, d1) / (l0 * l1) > -0.5
                } else {
                    true
                };
                vert_crease[v] = if sharp_turn { 2 } else { 1 };
            }
        }
    }

    let mut attr_quadrics: Option<Vec<AttrQuadric>> = None;
    if options.preserve_uvs
        && let Some(uv) = &uvs
    {
        let mut sum_pos = 0.0;
        let mut sum_uv = 0.0;
        for &(u, v) in &pairs {
            let d3 = sub3(f64v(&positions[u as usize]), f64v(&positions[v as usize]));
            sum_pos += dot3(d3, d3).sqrt();
            let du = f64::from(uv[2 * u as usize]) - f64::from(uv[2 * v as usize]);
            let dv = f64::from(uv[2 * u as usize + 1]) - f64::from(uv[2 * v as usize + 1]);
            sum_uv += (du * du + dv * dv).sqrt();
        }
        let uv_scale = if sum_uv > 1e-12 {
            (sum_pos / sum_uv).clamp(1e-4, 1e4)
        } else {
            1.0
        };
        let uv_weight = 0.25 * uv_scale * uv_scale;
        let mut aq = Vec::new();
        reserve(&mut aq, slots)?;
        aq.resize(slots, AttrQuadric::default());
        for tri in &tris {
            let c = tri.corners;
            let pts = c.map(|v| f64v(&positions[v as usize]));
            let uvs_f = c.map(|v| {
                let i = 2 * v as usize;
                [f64::from(uv[i]), f64::from(uv[i + 1])]
            });
            let fq = AttrQuadric::from_face(pts, uvs_f, uv_weight);
            for &v in &c {
                aq[v as usize] = aq[v as usize].merge(&fq);
            }
        }
        attr_quadrics = Some(aq);
    }

    let mut v2t: Vec<Vec<u32>> = Vec::new();
    reserve(&mut v2t, slots)?;
    v2t.resize_with(slots, Vec::new);
    for (i, tri) in tris.iter().enumerate() {
        for &v in &tri.corners {
            push(&mut v2t[v as usize], i as u32)?;
        }
    }

    let mut vlive = was_alive.clone();
    let mut moved = filled(slots, false)?;
    let mut v_ver = filled(slots, 0u32)?;
    let mut heap = BinaryHeap::new();
    heap.try_reserve(pairs.len())
        .map_err(|_| OpsError::Store(fast_mesh_store::StoreError::OutOfMemory))?;
    for &(a, b) in &pairs {
        let Some((_, cost, _)) = candidate_solve(
            a,
            b,
            options,
            uvs.is_some(),
            &positions,
            &is_boundary,
            &vert_crease,
            &crease_adj,
            &quadrics,
            aux_quadrics.as_deref(),
            attr_quadrics.as_deref(),
        ) else {
            continue;
        };
        heap.push(HeapEntry {
            cost,
            a,
            b,
            ver_a: 0,
            ver_b: 0,
        });
    }

    let max_cost2 = options.max_deviation.map(|d| d * d);
    let pure_geom = !options.preserve_uvs && !options.lock_creases;
    let mut max_dev_achieved = 0.0f64;
    let mut collapsed = 0usize;
    let mut sc = CollapseScratch::default();

    while live > target {
        let Some(top) = heap.pop() else {
            break;
        };
        if pure_geom
            && let Some(limit2) = max_cost2
            && top.cost > limit2
        {
            break;
        }
        let (a, b) = (top.a, top.b);
        if !vlive[a as usize]
            || !vlive[b as usize]
            || v_ver[a as usize] != top.ver_a
            || v_ver[b as usize] != top.ver_b
        {
            continue;
        }
        fill_owners(a, b, &v2t, &tris, &mut sc.owners)?;
        if sc.owners.is_empty() || sc.owners.len() > 2 {
            continue;
        }
        // Link condition: the common neighbours must be exactly the apexes
        // of the owners, or the merge would duplicate triangles.
        fill_neighbors(a, &v2t, &tris, &mut sc.na)?;
        fill_neighbors(b, &v2t, &tris, &mut sc.nb)?;
        sc.apexes.clear();
        reserve(&mut sc.apexes, sc.owners.len())?;
        for &o in &sc.owners {
            push(&mut sc.apexes, apex_of(&tris[o].corners, a, b))?;
        }
        sc.apexes.sort_unstable();
        sc.apexes.dedup();
        if sc.apexes.len() != sc.owners.len() {
            continue;
        }
        sc.common.clear();
        reserve(&mut sc.common, sc.na.len().min(sc.nb.len()))?;
        for &w in &sc.na {
            if w != a && w != b && sc.nb.binary_search(&w).is_ok() {
                push(&mut sc.common, w)?;
            }
        }
        if sc.common != sc.apexes {
            continue;
        }

        let Some((pnew, _heap_cost, geom_cost)) = candidate_solve(
            a,
            b,
            options,
            uvs.is_some(),
            &positions,
            &is_boundary,
            &vert_crease,
            &crease_adj,
            &quadrics,
            aux_quadrics.as_deref(),
            attr_quadrics.as_deref(),
        ) else {
            continue;
        };
        if let Some(limit2) = max_cost2
            && geom_cost > limit2
        {
            continue;
        }

        // Keep the smaller id so repeated runs agree on the survivor.
        let (keep, drop) = if a < b { (a, b) } else { (b, a) };
        if !pnew.iter().all(|c| c.is_finite()) {
            continue;
        }
        let pnew32 = [pnew[0] as f32, pnew[1] as f32, pnew[2] as f32];
        if !pnew32.iter().all(|c| c.is_finite()) {
            continue;
        }
        let pa = f64v(&positions[a as usize]);
        let pb = f64v(&positions[b as usize]);

        sc.fan.clear();
        reserve(&mut sc.fan, sc.na.len() + sc.nb.len())?;
        for &v in &[a, b] {
            for &i in &v2t[v as usize] {
                let idx = i as usize;
                if !tris[idx].alive || sc.owners.contains(&idx) {
                    continue;
                }
                push(&mut sc.fan, idx)?;
            }
        }
        sc.fan.sort_unstable();
        sc.fan.dedup();
        let keep_pos = [
            f64::from(pnew32[0]),
            f64::from(pnew32[1]),
            f64::from(pnew32[2]),
        ];

        let candidate_uv = uvs.as_ref().map(|uv| {
            let f = edge_param(keep_pos, pa, pb);
            let (ia, ib) = (2 * a as usize, 2 * b as usize);
            [
                uv[ia] + f * (uv[ib] - uv[ia]),
                uv[ia + 1] + f * (uv[ib + 1] - uv[ia + 1]),
            ]
        });

        sc.tri_keys.clear();
        reserve(&mut sc.tri_keys, sc.fan.len())?;
        let mut folds = false;
        for &i in &sc.fan {
            let old = tris[i].corners;
            let mapped = old.map(|v| if v == drop { keep } else { v });
            push(&mut sc.tri_keys, sorted_triple(mapped))?;
            let pts = [
                f64v(&positions[old[0] as usize]),
                f64v(&positions[old[1] as usize]),
                f64v(&positions[old[2] as usize]),
            ];
            let before = tri_normal(pts[0], pts[1], pts[2]);
            if dot3(before, before) == 0.0 {
                folds = true;
                break;
            }
            let new = old.map(|v| {
                if v == keep || v == drop {
                    keep_pos
                } else {
                    f64v(&positions[v as usize])
                }
            });
            let after = tri_normal(new[0], new[1], new[2]);
            if dot3(after, after) == 0.0
                || (options.prevent_normal_inversion && dot3(after, before) <= 0.0)
            {
                folds = true;
                break;
            }
            if options.preserve_uvs
                && let (Some(uv), Some(new_uv)) = (&uvs, candidate_uv)
            {
                let uv_v = |v: u32| -> [f32; 2] { [uv[2 * v as usize], uv[2 * v as usize + 1]] };
                let buv = cross2d(uv_v(old[0]), uv_v(old[1]), uv_v(old[2]));
                if buv.abs() > 1e-16 {
                    let auv_pts = old.map(|v| {
                        if v == keep || v == drop {
                            new_uv
                        } else {
                            uv_v(v)
                        }
                    });
                    let auv = cross2d(auv_pts[0], auv_pts[1], auv_pts[2]);
                    if auv * buv <= 0.0 || auv.abs() <= 1e-16 {
                        folds = true;
                        break;
                    }
                }
            }
        }
        if !folds && sc.tri_keys.len() > 1 {
            sc.tri_keys.sort_unstable();
            if sc.tri_keys.windows(2).any(|w| w[0] == w[1]) {
                folds = true;
            }
        }
        if folds {
            continue;
        }

        if let (Some(uv), Some(new_uv)) = (&mut uvs, candidate_uv) {
            let ik = 2 * keep as usize;
            uv[ik] = new_uv[0];
            uv[ik + 1] = new_uv[1];
        }
        let mut qsum = quadrics[keep as usize];
        add_quadric(&mut qsum, &quadrics[drop as usize]);
        quadrics[keep as usize] = qsum;
        if let Some(aux) = &mut aux_quadrics {
            let extra = aux[drop as usize];
            add_quadric(&mut aux[keep as usize], &extra);
        }
        if let Some(attr) = &mut attr_quadrics {
            attr[keep as usize] = attr[keep as usize].merge(&attr[drop as usize]);
        }
        if options.lock_creases {
            vert_crease[keep as usize] = vert_crease[a as usize].max(vert_crease[b as usize]);
            let drop_neighbors = core::mem::take(&mut crease_adj[drop as usize]);
            crease_adj[keep as usize].retain(|&w| w != keep && w != drop);
            for w in drop_neighbors {
                crease_adj[w as usize].retain(|&u| u != drop);
                if w != keep && vlive[w as usize] {
                    push(&mut crease_adj[keep as usize], w)?;
                    push(&mut crease_adj[w as usize], keep)?;
                    crease_adj[w as usize].sort_unstable();
                    crease_adj[w as usize].dedup();
                }
            }
            crease_adj[keep as usize].sort_unstable();
            crease_adj[keep as usize].dedup();
        }
        is_boundary[keep as usize] |= is_boundary[drop as usize];
        positions[keep as usize] = pnew32;
        moved[keep as usize] = true;
        let dev = geom_cost.max(0.0).sqrt();
        if dev > max_dev_achieved {
            max_dev_achieved = dev;
        }

        for &o in &sc.owners {
            tris[o].alive = false;
        }
        live -= sc.owners.len();
        // Prune dead triangles from `keep`'s adjacency, then transfer only the
        // surviving triangles from `drop` into `keep` without duplicates.
        v2t[keep as usize].retain(|&i| tris[i as usize].alive);
        sc.v2t_drop.clear();
        reserve(&mut sc.v2t_drop, v2t[drop as usize].len())?;
        sc.v2t_drop.extend_from_slice(&v2t[drop as usize]);
        v2t[drop as usize].clear();
        reserve(&mut v2t[keep as usize], sc.v2t_drop.len())?;
        for &idx in &sc.v2t_drop {
            let t = &mut tris[idx as usize];
            if !t.alive {
                continue;
            }
            t.corners = t.corners.map(|v| if v == drop { keep } else { v });
            push(&mut v2t[keep as usize], idx)?;
        }
        v2t[keep as usize].sort_unstable();
        v2t[keep as usize].dedup();
        vlive[drop as usize] = false;
        v_ver[keep as usize] = v_ver[keep as usize].wrapping_add(1);
        collapsed += 1;

        // Re-arm the survivor's edges with current quadrics and vertex versions.
        sc.ring.clear();
        reserve(&mut sc.ring, 16)?;
        for &i in &v2t[keep as usize] {
            let t = &tris[i as usize];
            if !t.alive || !t.corners.contains(&keep) {
                continue;
            }
            for &w in &t.corners {
                if w != keep && vlive[w as usize] {
                    push(&mut sc.ring, w)?;
                }
            }
        }
        sc.ring.sort_unstable();
        sc.ring.dedup();
        if heap.capacity() - heap.len() < sc.ring.len() {
            heap.try_reserve(sc.ring.len())
                .map_err(|_| OpsError::Store(fast_mesh_store::StoreError::OutOfMemory))?;
        }
        for &w in &sc.ring {
            let (lo, hi) = key(keep, w);
            let Some((_, cost, _)) = candidate_solve(
                lo,
                hi,
                options,
                uvs.is_some(),
                &positions,
                &is_boundary,
                &vert_crease,
                &crease_adj,
                &quadrics,
                aux_quadrics.as_deref(),
                attr_quadrics.as_deref(),
            ) else {
                continue;
            };
            heap.push(HeapEntry {
                cost,
                a: lo,
                b: hi,
                ver_a: v_ver[lo as usize],
                ver_b: v_ver[hi as usize],
            });
        }
    }

    #[cfg(test)]
    for list in &v2t {
        assert!(
            list.windows(2).all(|w| w[0] < w[1]),
            "duplicate or unsorted entry in v2t: {list:?}"
        );
    }

    if collapsed == 0 {
        return Ok(SimplifyReport::default());
    }

    // Reserve everything before the first write; past this point only
    // infallible pushes into reserved capacity remain.
    let mut set_p: Vec<VertexId> = Vec::new();
    let mut set_p_to: Vec<[f32; 3]> = Vec::new();
    let mut set_t: Vec<TriangleId> = Vec::new();
    let mut set_t_to: Vec<[VertexId; 3]> = Vec::new();
    let mut del_t: Vec<TriangleId> = Vec::new();
    let mut del_v: Vec<VertexId> = Vec::new();
    let mut set_uv: Vec<VertexId> = Vec::new();
    let mut set_uv_to: Vec<f32> = Vec::new();
    if let Some(uv) = &uvs {
        for v in 0..slots as u32 {
            if vlive[v as usize] && moved[v as usize] {
                push(&mut set_uv, v)?;
                let i = 2 * v as usize;
                reserve(&mut set_uv_to, 2)?;
                set_uv_to.extend_from_slice(&uv[i..i + 2]);
            }
        }
    }
    let mut n_moved = 0;
    let mut n_rewired = 0;
    let mut n_dead_tri = 0;
    let mut n_dead_vert = 0;
    for v in 0..slots as u32 {
        if was_alive[v as usize] && !vlive[v as usize] {
            n_dead_vert += 1;
        } else if vlive[v as usize]
            && moved[v as usize]
            && positions[v as usize] != base[v as usize]
        {
            n_moved += 1;
        }
    }
    for tri in &tris {
        if !tri.alive {
            n_dead_tri += 1;
        } else {
            let current = mesh
                .triangle(tri.id)
                .expect("simplify: working id must exist");
            if current != tri.corners {
                n_rewired += 1;
            }
        }
    }
    reserve(&mut set_p, n_moved)?;
    reserve(&mut set_p_to, n_moved)?;
    reserve(&mut set_t, n_rewired)?;
    reserve(&mut set_t_to, n_rewired)?;
    reserve(&mut del_t, n_dead_tri)?;
    reserve(&mut del_v, n_dead_vert)?;
    for v in 0..slots as u32 {
        if was_alive[v as usize] && !vlive[v as usize] {
            del_v.push(v);
        } else if vlive[v as usize]
            && moved[v as usize]
            && positions[v as usize] != base[v as usize]
        {
            set_p.push(v);
            set_p_to.push(positions[v as usize]);
        }
    }
    for tri in &tris {
        if !tri.alive {
            del_t.push(tri.id);
        } else {
            let current = mesh
                .triangle(tri.id)
                .expect("simplify: working id must exist");
            if current != tri.corners {
                set_t.push(tri.id);
                set_t_to.push(tri.corners);
            }
        }
    }
    tx.set_positions(&set_p, &set_p_to)?;
    if uvs.is_some() {
        tx.write_channel(UV_CHANNEL, &set_uv, &set_uv_to)?;
    }
    tx.set_triangles(&set_t, &set_t_to)?;
    tx.remove_triangles(&del_t)?;
    tx.remove_vertices(&del_v)?;

    let live_now = tris.iter().filter(|t| t.alive).count();
    let started = live_now + del_t.len();
    Ok(SimplifyReport {
        collapsed,
        triangles_removed: started - live_now,
        vertices_removed: del_v.len(),
        max_deviation_achieved: max_dev_achieved,
    })
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::{Mesh, Transaction, VertexId};

    use super::*;
    use crate::test_util::{box_mesh, edit, grid, mesh_of, parts, sweep, uv_sphere};

    fn live_count(mesh: &Mesh) -> usize {
        (0..mesh.triangle_slots() as u32)
            .filter(|&t| mesh.triangle(t).is_some())
            .count()
    }

    fn assert_no_degenerate(mesh: &Mesh) {
        for t in 0..mesh.triangle_slots() as u32 {
            if let Some(c) = mesh.triangle(t) {
                assert!(
                    c[0] != c[1] && c[1] != c[2] && c[2] != c[0],
                    "degenerate: {c:?}"
                );
                for &v in &c {
                    assert!(mesh.is_vertex_alive(v as VertexId));
                }
            }
        }
    }

    #[test]
    fn decimate_cube_to_target_ratio() {
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        assert_eq!(live_count(&mesh), 12);
        let options = SimplifyOptions::builder().target_triangles(6).build();
        let (report, empty) = edit(&mut mesh, |tx| simplify(tx, &options));
        assert!(!empty);
        assert!(report.collapsed > 0, "expected collapses, got {report:?}");
        let now = live_count(&mesh);
        assert!(now <= 6, "expected at most 6 triangles, got {now}");
        assert!(now > 0, "must keep some triangles");
        assert_eq!(report.triangles_removed, 12 - now);
        assert_eq!(report.vertices_removed, report.collapsed);
        assert_no_degenerate(&mesh);
    }

    #[test]
    fn decimate_sphere_to_target_ratio() {
        let mut mesh = uv_sphere(8, 16, 1.0);
        let before = live_count(&mesh);
        assert!(before > 100, "fixture too small: {before}");
        let target = before / 2;
        let options = SimplifyOptions::builder().target_triangles(target).build();
        let (report, empty) = edit(&mut mesh, |tx| simplify(tx, &options));
        assert!(!empty);
        assert!(report.collapsed > 0, "expected collapses, got {report:?}");
        let now = live_count(&mesh);
        assert!(
            now <= target,
            "expected at most {target} triangles, got {now}"
        );
        assert!(now > 0, "must keep some triangles");
        assert_no_degenerate(&mesh);
    }

    #[test]
    fn boundary_preserved_when_enabled() {
        // A 4x4 grid: 25 vertices, 16 on the rim. Locking the rim keeps
        // every rim vertex alive at its position.
        let mut mesh = grid(4);
        let rim: Vec<[f32; 3]> = mesh.positions().to_vec();
        let options = SimplifyOptions::builder()
            .target_triangles(16)
            .preserve_boundary(true)
            .build();
        let (report, empty) = edit(&mut mesh, |tx| simplify(tx, &options));
        assert!(!empty);
        assert!(report.collapsed > 0, "expected collapses, got {report:?}");
        assert!(live_count(&mesh) <= 16);
        let n = 4 + 1;
        for y in 0..=4 {
            for x in 0..=4 {
                if x == 0 || y == 0 || x == 4 || y == 4 {
                    let v = y * n + x;
                    assert!(
                        mesh.is_vertex_alive(v as VertexId),
                        "rim vertex {v} removed"
                    );
                    assert_eq!(mesh.positions()[v as usize], rim[v as usize]);
                }
            }
        }
        assert_no_degenerate(&mesh);
    }

    #[test]
    fn boundary_option_changes_locked_quad() {
        // Two triangles, every vertex on the rim: locked, nothing collapses;
        // unlocked, the diagonal collapses.
        let quad = || {
            mesh_of(
                &[0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                &[[0, 1, 2], [0, 2, 3]],
            )
        };
        let mut locked = quad();
        let before = parts(&locked);
        let (report, empty) = edit(&mut locked, |tx| {
            simplify(
                tx,
                &SimplifyOptions::builder()
                    .target_triangles(1)
                    .preserve_boundary(true)
                    .build(),
            )
        });
        assert_eq!((report.collapsed, empty), (0, true));
        assert_eq!(parts(&locked), before);

        let mut open = quad();
        let (report, empty) = edit(&mut open, |tx| {
            simplify(
                tx,
                &SimplifyOptions::builder()
                    .target_triangles(1)
                    .preserve_boundary(false)
                    .build(),
            )
        });
        assert!(!empty);
        assert!(report.collapsed > 0, "expected collapses, got {report:?}");
        assert!(live_count(&open) < 2);
    }

    #[test]
    fn invalid_target_and_noop_target() {
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        {
            let mut tx = mesh.begin_edit();
            assert_eq!(
                simplify(&mut tx, &SimplifyOptions::builder().build()),
                Err(OpsError::InvalidInput(
                    "simplify target_triangles must be at least 1"
                ))
            );
        }
        // Target at/above the count succeeds without writing.
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        let before = parts(&mesh);
        let (report, empty) = edit(&mut mesh, |tx| {
            simplify(tx, &SimplifyOptions::builder().target_triangles(12).build())
        });
        assert_eq!((report.collapsed, empty), (0, true));
        assert_eq!(parts(&mesh), before);
        let (report, empty) = edit(&mut mesh, |tx| {
            simplify(
                tx,
                &SimplifyOptions::builder().target_triangles(10_000).build(),
            )
        });
        assert_eq!((report.collapsed, empty), (0, true));
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn simplify_sweep_keeps_atomicity() {
        let n = sweep(grid(4), |tx: &mut Transaction| {
            simplify(tx, &SimplifyOptions::builder().target_triangles(16).build())?;
            Ok(())
        });
        assert!(n > 0);
    }

    #[test]
    fn simplify_large_mesh_without_v2t_bloat() {
        let mut mesh = uv_sphere(160, 320, 1.0);
        let before = live_count(&mesh);
        assert!(before >= 100_000, "fixture too small: {before}");

        let started = std::time::Instant::now();
        let options = SimplifyOptions::builder().target_triangles(10_000).build();
        let (report, empty) = edit(&mut mesh, |tx| simplify(tx, &options));
        let elapsed = started.elapsed();

        assert!(!empty);
        assert!(report.collapsed > 0, "expected collapses, got {report:?}");
        let now = live_count(&mesh);
        assert!(now <= 10_000, "expected at most 10_000 tris, got {now}");
        assert!(now > 0);
        assert!(
            elapsed.as_secs_f64() < 10.0,
            "100k->10k simplify took too long: {elapsed:?}"
        );
        assert_no_degenerate(&mesh);

        let analysis = crate::analyze(&mesh, &crate::AnalyzeOptions::default()).unwrap();
        assert_eq!(analysis.non_manifold_edges, 0);
        assert!(analysis.non_manifold_vertices.is_empty());
        assert_eq!(analysis.duplicate, 0);
        assert_eq!(analysis.degenerate, 0);
        assert_eq!(analysis.orientation_conflicts, 0);
        assert!(analysis.loops.is_empty());
        assert!(analysis.pinch_vertices.is_empty());
    }

    #[test]
    fn decimate_target_ratio_and_max_deviation_bound() {
        let mut sphere = uv_sphere(16, 32, 1.0);
        let initial = live_count(&sphere);
        let opts = DecimateOptions::builder()
            .target_ratio(0.5)
            .max_deviation(0.05)
            .build();
        let (report, empty) = edit(&mut sphere, |tx| simplify_with(tx, &opts));
        assert!(!empty);
        assert!(report.collapsed > 0);
        assert!(report.max_deviation_achieved <= 0.05 + 1e-12);
        assert!(live_count(&sphere) < initial);

        let mut tighter = uv_sphere(16, 32, 1.0);
        let strict = DecimateOptions::builder()
            .target_triangles(16)
            .max_deviation(0.0005)
            .build();
        let (strict_report, _) = edit(&mut tighter, |tx| simplify_with(tx, &strict));
        assert!(strict_report.max_deviation_achieved <= 0.0005 + 1e-12);
        assert!(live_count(&tighter) > live_count(&sphere));
    }

    #[test]
    fn decimate_validation_rejects_bad_inputs() {
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        let bad = [
            DecimateOptions::builder().target_ratio(0.0).build(),
            DecimateOptions::builder().target_ratio(1.5).build(),
            DecimateOptions::builder().target_ratio(f32::NAN).build(),
            DecimateOptions::builder().max_deviation(-0.1).build(),
            DecimateOptions::builder().crease_angle_deg(181.0).build(),
        ];
        for o in &bad {
            let mut tx = mesh.begin_edit();
            assert!(matches!(
                simplify_with(&mut tx, o),
                Err(OpsError::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn decimate_locks_sharp_box_corners_and_creases() {
        let mut cube = box_mesh([0.0; 3], [1.0; 3]);
        let opts = DecimateOptions::builder()
            .target_triangles(4)
            .lock_creases(true)
            .crease_angle_deg(30.0)
            .build();
        let (report, empty) = edit(&mut cube, |tx| simplify_with(tx, &opts));
        assert!(empty);
        assert_eq!(report.collapsed, 0);
        assert_eq!(live_count(&cube), 12);
    }

    #[test]
    fn decimate_preserve_uvs_avoids_uv_folding() {
        let mut m = grid(10);
        let uvs: Vec<f32> = m
            .positions()
            .iter()
            .flat_map(|p| [p[0] / 10.0, p[1] / 10.0])
            .collect();
        m.add_uvs(&uvs).unwrap();
        let opts = DecimateOptions::builder()
            .target_ratio(0.3)
            .preserve_uvs(true)
            .build();
        let (report, empty) = edit(&mut m, |tx| simplify_with(tx, &opts));
        assert!(!empty);
        assert!(report.collapsed > 0);
        let out_uvs = m.uvs().unwrap();
        for t in 0..m.triangle_slots() as u32 {
            if let Some([a, b, c]) = m.triangle(t) {
                let area = cross2d(
                    [out_uvs[2 * a as usize], out_uvs[2 * a as usize + 1]],
                    [out_uvs[2 * b as usize], out_uvs[2 * b as usize + 1]],
                    [out_uvs[2 * c as usize], out_uvs[2 * c as usize + 1]],
                );
                assert!(area > 0.0, "folded UV triangle {a},{b},{c}: area={area}");
            }
        }
    }
}
