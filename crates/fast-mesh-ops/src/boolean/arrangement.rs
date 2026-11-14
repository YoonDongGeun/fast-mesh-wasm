//! Canonical simplex-keyed intersection arrangement for mesh booleans.

use fast_mesh_bvh::Bvh;
use fast_mesh_query::{Rigid, intersecting_pairs};
use fast_mesh_store::{Mesh, StoreError, TriangleId, VertexId};
use robust::{Coord, Coord3D, orient2d, orient3d};

use crate::{OpsError, Result, allocation_allowed, push, reserve};

pub(super) type ArrVertexId = u32;

type D3 = [f64; 3];
type D2 = [f64; 2];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ArrOrigin {
    LiveA(VertexId),
    Fresh,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct TriConstraints {
    pub edge_splits: [Vec<ArrVertexId>; 3],
    pub inner_verts: Vec<ArrVertexId>,
    pub segments: Vec<(ArrVertexId, ArrVertexId)>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct Arrangement {
    pub positions: Vec<[f32; 3]>,
    pub origins: Vec<ArrOrigin>,
    pub tri_a: Vec<(TriangleId, TriConstraints)>,
    pub tri_b: Vec<(TriangleId, TriConstraints)>,
    pub seam_edges: Vec<(ArrVertexId, ArrVertexId)>,
    corner_a: Vec<(VertexId, ArrVertexId)>,
    corner_b: Vec<(VertexId, ArrVertexId)>,
}

impl Arrangement {
    pub(super) fn corners_a(&self, a: &Mesh, t: TriangleId) -> [ArrVertexId; 3] {
        let tri = a.triangle(t).expect("corners_a called on live triangle");
        tri.map(|v| {
            let idx = self
                .corner_a
                .binary_search_by_key(&v, |&(vid, _)| vid)
                .expect("cut triangle corner in A must have an ArrVertexId");
            self.corner_a[idx].1
        })
    }

    pub(super) fn corners_b(&self, tool: &Mesh, t: TriangleId) -> [ArrVertexId; 3] {
        let tri = tool.triangle(t).expect("corners_b called on live triangle");
        tri.map(|v| {
            let idx = self
                .corner_b
                .binary_search_by_key(&v, |&(vid, _)| vid)
                .expect("cut triangle corner in tool must have an ArrVertexId");
            self.corner_b[idx].1
        })
    }
}

fn oom() -> OpsError {
    OpsError::Store(StoreError::OutOfMemory)
}

fn sign(x: f64) -> i8 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

fn c3(p: D3) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

fn c2(p: D2) -> Coord<f64> {
    Coord { x: p[0], y: p[1] }
}

/// Exact right-hand orientation sign of `det[b - a, c - a, d - a]`:
/// `+1` when `d` lies on the side of `abc` that `(b - a) × (c - a)` points to.
///
/// Every finite `f32` value (including `f32` subnormals down to `2^-149`) lies
/// strictly within Shewchuk's exact `[2^-200, 2^200]` exponent window.
pub(super) fn orient3d_exact(a: D3, b: D3, c: D3, d: D3) -> i8 {
    debug_assert!([a, b, c, d].iter().flatten().all(|x| x.is_finite()));
    sign(-orient3d(c3(a), c3(b), c3(c), c3(d)))
}

fn orient2d_exact(a: D2, b: D2, c: D2) -> i8 {
    debug_assert!([a, b, c].iter().flatten().all(|x| x.is_finite()));
    sign(orient2d(c2(a), c2(b), c2(c)))
}

fn sub3(a: D3, b: D3) -> D3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: D3, b: D3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn canon_edge(u: VertexId, v: VertexId) -> (VertexId, VertexId) {
    if u < v { (u, v) } else { (v, u) }
}

fn pos_f64(mesh: &Mesh, v: VertexId) -> D3 {
    mesh.position(v).expect("live vertex").map(f64::from)
}

fn normalize_zero(p: [f32; 3]) -> [f32; 3] {
    p.map(|c| if c == 0.0 { 0.0 } else { c })
}

fn pos_bits(p: [f32; 3]) -> [u32; 3] {
    normalize_zero(p).map(f32::to_bits)
}

/// 2D coordinate projection dropping a triangle's dominant normal axis,
/// oriented so the source triangle is strictly counter-clockwise (`orient2d > 0`).
#[derive(Clone, Copy)]
struct Plane2D {
    axis: usize,
    flip: bool,
}

impl Plane2D {
    fn from_tri(t: &[D3; 3]) -> Option<Self> {
        let drop_axis = |p: D3, k: usize| [p[(k + 1) % 3], p[(k + 2) % 3]];
        let mut best: Option<(usize, f64, bool)> = None;
        for k in 0..3 {
            let [a, b, c] = t.map(|p| drop_axis(p, k));
            let s = orient2d_exact(a, b, c);
            if s == 0 {
                continue;
            }
            let area = orient2d(c2(a), c2(b), c2(c)).abs();
            if best.is_none_or(|(_, best_area, _)| area > best_area) {
                best = Some((k, area, s < 0));
            }
        }
        best.map(|(axis, _, flip)| Plane2D { axis, flip })
    }

    fn project(self, p: D3) -> D2 {
        let u = p[(self.axis + 1) % 3];
        let v = p[(self.axis + 2) % 3];
        if self.flip { [v, u] } else { [u, v] }
    }
}

/// Canonical simplex key for an arrangement vertex (§5.2).
///
/// `VertOnEdgeA` and `VertOnEdgeB` resolve directly to `OrigB(vert_b)` and
/// `OrigA(vert_a)` respectively while registering the split on `edge_a` / `edge_b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum VertexKey {
    OrigA(VertexId),
    OrigB(VertexId),
    EdgeEdge {
        edge_a: (VertexId, VertexId),
        edge_b: (VertexId, VertexId),
    },
    EdgeTriA {
        edge_a: (VertexId, VertexId),
        tri_b: TriangleId,
    },
    TriEdgeB {
        tri_a: TriangleId,
        edge_b: (VertexId, VertexId),
    },
}

/// Where an intersection point sits on a triangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TriLoc {
    Corner(usize),
    Edge(usize),
    Inner,
}

#[derive(Debug, Clone, Copy)]
struct PairHit {
    loc_a: TriLoc,
    loc_b: TriLoc,
    key: VertexKey,
}

/// Exact classification of a 2D point `p` against non-degenerate triangle `t2`.
/// `orient_sign` is `orient2d_exact(t2[0], t2[1], t2[2])` (`+1` or `-1`).
fn classify_in_tri_2d(p: D2, t2: &[D2; 3], orient_sign: i8) -> Option<TriLoc> {
    let s = [0, 1, 2].map(|k| orient_sign * orient2d_exact(t2[k], t2[(k + 1) % 3], p));
    if s[0] < 0 || s[1] < 0 || s[2] < 0 {
        return None;
    }
    for k in 0..3 {
        if s[k] == 0 && s[(k + 2) % 3] == 0 {
            return Some(TriLoc::Corner(k));
        }
    }
    if let Some(k) = s.iter().position(|&v| v == 0) {
        return Some(TriLoc::Edge(k));
    }
    Some(TriLoc::Inner)
}

/// Exact classification of a 3D segment `(u, v)` that strictly straddles `plane(w)`
/// (`su * sv < 0`, where `su = orient3d_exact(w[0], w[1], w[2], u)`).
fn classify_straddling_edge_3d(u: D3, su: i8, v: D3, w: &[D3; 3]) -> Option<TriLoc> {
    let (p, q) = if su > 0 { (u, v) } else { (v, u) };
    let o = [0, 1, 2].map(|k| orient3d_exact(q, p, w[k], w[(k + 1) % 3]));
    if o[0] < 0 || o[1] < 0 || o[2] < 0 {
        return None;
    }
    for k in 0..3 {
        if o[k] == 0 && o[(k + 2) % 3] == 0 {
            return Some(TriLoc::Corner(k));
        }
    }
    if let Some(k) = o.iter().position(|&v| v == 0) {
        return Some(TriLoc::Edge(k));
    }
    Some(TriLoc::Inner)
}

/// Closest points of two 3D segments `p1..q1` and `p2..q2` (Ericson 5.1.9).
fn seg_seg_closest(p1: D3, q1: D3, p2: D3, q2: D3) -> (D3, D3) {
    let d1 = sub3(q1, p1);
    let d2 = sub3(q2, p2);
    let r = sub3(p1, p2);
    let a = dot3(d1, d1);
    let e = dot3(d2, d2);
    let f = dot3(d2, r);
    let at = |p: D3, d: D3, t: f64| -> D3 { std::array::from_fn(|k| p[k] + t * d[k]) };
    let unit = |num: f64, den: f64| {
        let t = num / den;
        if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    let (s, t) = if a <= 0.0 && e <= 0.0 {
        (0.0, 0.0)
    } else if a <= 0.0 {
        (0.0, unit(f, e))
    } else {
        let c = dot3(d1, r);
        if e <= 0.0 {
            (unit(-c, a), 0.0)
        } else {
            let b = dot3(d1, d2);
            let denom = a * e - b * b;
            let s0 = if denom > 0.0 {
                unit(b * f - c * e, denom)
            } else {
                0.0
            };
            let t0 = (b * s0 + f) / e;
            if t0 < 0.0 {
                (unit(-c, a), 0.0)
            } else if t0 > 1.0 {
                (unit(b - c, a), 1.0)
            } else {
                (s0, t0)
            }
        }
    };
    (at(p1, d1, s), at(p2, d2, t))
}

fn clamp_to_overlap_box(p: D3, lo1: D3, hi1: D3, lo2: D3, hi2: D3) -> D3 {
    std::array::from_fn(|k| {
        let lo = lo1[k].max(lo2[k]);
        let hi = hi1[k].min(hi2[k]);
        if lo <= hi {
            p[k].clamp(lo, hi)
        } else {
            p[k].clamp(lo1[k], hi1[k])
        }
    })
}

fn compute_edge_edge_f64(
    a: &Mesh,
    edge_a: (VertexId, VertexId),
    tool: &Mesh,
    edge_b: (VertexId, VertexId),
) -> D3 {
    let (pa0, pa1) = (pos_f64(a, edge_a.0), pos_f64(a, edge_a.1));
    let (pb0, pb1) = (pos_f64(tool, edge_b.0), pos_f64(tool, edge_b.1));
    let proj = |p: D3, k: usize| c2([p[(k + 1) % 3], p[(k + 2) % 3]]);
    let mut best = (0usize, -1.0_f64, 0.0_f64, 0.0_f64);
    for k in 0..3 {
        let (a0, a1) = (proj(pa0, k), proj(pa1, k));
        let db0 = orient2d(a0, a1, proj(pb0, k));
        let db1 = orient2d(a0, a1, proj(pb1, k));
        let span = (db0 - db1).abs();
        if span > best.1 {
            best = (k, span, db0, db1);
        }
    }
    let (k, _, db0, db1) = best;
    let (b0, b1) = (proj(pb0, k), proj(pb1, k));
    let da0 = orient2d(b0, b1, proj(pa0, k));
    let da1 = orient2d(b0, b1, proj(pa1, k));
    let denom_a = da0 - da1;
    let denom_b = db0 - db1;
    let (qa, qb) =
        if da0 != da1 && db0 != db1 && (da0 / denom_a).is_finite() && (db0 / denom_b).is_finite() {
            let s = (da0 / denom_a).clamp(0.0, 1.0);
            let t = (db0 / denom_b).clamp(0.0, 1.0);
            (
                std::array::from_fn(|i| pa0[i] + s * (pa1[i] - pa0[i])),
                std::array::from_fn(|i| pb0[i] + t * (pb1[i] - pb0[i])),
            )
        } else {
            seg_seg_closest(pa0, pa1, pb0, pb1)
        };
    let mid = std::array::from_fn(|k| 0.5 * (qa[k] + qb[k]));
    let lo1 = std::array::from_fn(|k| pa0[k].min(pa1[k]));
    let hi1 = std::array::from_fn(|k| pa0[k].max(pa1[k]));
    let lo2 = std::array::from_fn(|k| pb0[k].min(pb1[k]));
    let hi2 = std::array::from_fn(|k| pb0[k].max(pb1[k]));
    clamp_to_overlap_box(mid, lo1, hi1, lo2, hi2)
}

fn compute_edge_tri_f64(
    edge_mesh: &Mesh,
    edge: (VertexId, VertexId),
    tri_mesh: &Mesh,
    tri_id: TriangleId,
) -> D3 {
    let (p0, p1) = (pos_f64(edge_mesh, edge.0), pos_f64(edge_mesh, edge.1));
    let mut tv = tri_mesh.triangle(tri_id).expect("live triangle");
    tv.sort_unstable();
    let [w0, w1, w2] = tv.map(|v| pos_f64(tri_mesh, v));
    // Adaptive Shewchuk determinant heights guarantee zero cancellation and exact
    // opposite signs at `p0` and `p1`.
    let d0 = orient3d(c3(w0), c3(w1), c3(w2), c3(p0));
    let d1 = orient3d(c3(w0), c3(w1), c3(w2), c3(p1));
    let denom = d0 - d1;
    let t = if denom != 0.0 && (d0 / denom).is_finite() {
        (d0 / denom).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let pt = std::array::from_fn(|k| p0[k] + t * (p1[k] - p0[k]));
    let tri_lo = std::array::from_fn(|k| w0[k].min(w1[k]).min(w2[k]));
    let tri_hi = std::array::from_fn(|k| w0[k].max(w1[k]).max(w2[k]));
    let seg_lo = std::array::from_fn(|k| p0[k].min(p1[k]));
    let seg_hi = std::array::from_fn(|k| p0[k].max(p1[k]));
    clamp_to_overlap_box(pt, tri_lo, tri_hi, seg_lo, seg_hi)
}

fn key_pos_f64(a: &Mesh, tool: &Mesh, key: VertexKey) -> D3 {
    match key {
        VertexKey::OrigA(va) => pos_f64(a, va),
        VertexKey::OrigB(vb) => pos_f64(tool, vb),
        VertexKey::EdgeEdge { edge_a, edge_b } => compute_edge_edge_f64(a, edge_a, tool, edge_b),
        VertexKey::EdgeTriA { edge_a, tri_b } => compute_edge_tri_f64(a, edge_a, tool, tri_b),
        VertexKey::TriEdgeB { tri_a, edge_b } => compute_edge_tri_f64(tool, edge_b, a, tri_a),
    }
}

fn key_pos_and_origin(a: &Mesh, tool: &Mesh, key: VertexKey) -> Result<([f32; 3], ArrOrigin)> {
    match key {
        VertexKey::OrigA(va) => {
            let p = a.position(va).expect("live vertex of A");
            Ok((p, ArrOrigin::LiveA(va)))
        }
        VertexKey::OrigB(vb) => {
            let p = tool.position(vb).expect("live vertex of tool");
            Ok((normalize_zero(p), ArrOrigin::Fresh))
        }
        VertexKey::EdgeEdge { .. } | VertexKey::EdgeTriA { .. } | VertexKey::TriEdgeB { .. } => {
            let d = key_pos_f64(a, tool, key);
            let f = normalize_zero([d[0] as f32, d[1] as f32, d[2] as f32]);
            if !f.iter().all(|c| c.is_finite()) {
                return Err(OpsError::InvalidInput("boolean point leaves f32 range"));
            }
            Ok((f, ArrOrigin::Fresh))
        }
    }
}

#[derive(Default)]
struct RawBuffers {
    keys: Vec<VertexKey>,
    edge_splits_a: Vec<((VertexId, VertexId), VertexKey)>,
    edge_splits_b: Vec<((VertexId, VertexId), VertexKey)>,
    inner_a: Vec<(TriangleId, VertexKey)>,
    inner_b: Vec<(TriangleId, VertexKey)>,
    segs_a: Vec<(TriangleId, VertexKey, VertexKey)>,
    segs_b: Vec<(TriangleId, VertexKey, VertexKey)>,
}

impl RawBuffers {
    fn record_hit(
        &mut self,
        ta_id: TriangleId,
        ta: &[VertexId; 3],
        loc_a: TriLoc,
        tb_id: TriangleId,
        tb: &[VertexId; 3],
        loc_b: TriLoc,
    ) -> Result<PairHit> {
        let key = match (loc_a, loc_b) {
            (TriLoc::Corner(i), TriLoc::Corner(j)) => {
                // Push OrigB(tb[j]) as well so corners_b and bit-pattern unification
                // map tb[j] directly onto LiveA(ta[i]).
                push(&mut self.keys, VertexKey::OrigB(tb[j]))?;
                VertexKey::OrigA(ta[i])
            }
            (TriLoc::Corner(i), TriLoc::Edge(j)) => {
                let k = VertexKey::OrigA(ta[i]);
                push(
                    &mut self.edge_splits_b,
                    (canon_edge(tb[j], tb[(j + 1) % 3]), k),
                )?;
                k
            }
            (TriLoc::Corner(i), TriLoc::Inner) => {
                let k = VertexKey::OrigA(ta[i]);
                push(&mut self.inner_b, (tb_id, k))?;
                k
            }
            (TriLoc::Edge(i), TriLoc::Corner(j)) => {
                let k = VertexKey::OrigB(tb[j]);
                push(
                    &mut self.edge_splits_a,
                    (canon_edge(ta[i], ta[(i + 1) % 3]), k),
                )?;
                k
            }
            (TriLoc::Inner, TriLoc::Corner(j)) => {
                let k = VertexKey::OrigB(tb[j]);
                push(&mut self.inner_a, (ta_id, k))?;
                k
            }
            (TriLoc::Edge(i), TriLoc::Edge(j)) => {
                let ea = canon_edge(ta[i], ta[(i + 1) % 3]);
                let eb = canon_edge(tb[j], tb[(j + 1) % 3]);
                let k = VertexKey::EdgeEdge {
                    edge_a: ea,
                    edge_b: eb,
                };
                push(&mut self.edge_splits_a, (ea, k))?;
                push(&mut self.edge_splits_b, (eb, k))?;
                k
            }
            (TriLoc::Edge(i), TriLoc::Inner) => {
                let ea = canon_edge(ta[i], ta[(i + 1) % 3]);
                let k = VertexKey::EdgeTriA {
                    edge_a: ea,
                    tri_b: tb_id,
                };
                push(&mut self.edge_splits_a, (ea, k))?;
                push(&mut self.inner_b, (tb_id, k))?;
                k
            }
            (TriLoc::Inner, TriLoc::Edge(j)) => {
                let eb = canon_edge(tb[j], tb[(j + 1) % 3]);
                let k = VertexKey::TriEdgeB {
                    tri_a: ta_id,
                    edge_b: eb,
                };
                push(&mut self.inner_a, (ta_id, k))?;
                push(&mut self.edge_splits_b, (eb, k))?;
                k
            }
            (TriLoc::Inner, TriLoc::Inner) => unreachable!("simplices meet through a boundary"),
        };
        push(&mut self.keys, key)?;
        Ok(PairHit { loc_a, loc_b, key })
    }

    fn record_seg(
        &mut self,
        ta_id: TriangleId,
        tb_id: TriangleId,
        k0: VertexKey,
        k1: VertexKey,
    ) -> Result<()> {
        if k0 != k1 {
            let (lo, hi) = if k0 < k1 { (k0, k1) } else { (k1, k0) };
            push(&mut self.segs_a, (ta_id, lo, hi))?;
            push(&mut self.segs_b, (tb_id, lo, hi))?;
        }
        Ok(())
    }
}

/// Fixed-capacity stack buffer for a single triangle-triangle pair's hits.
struct HitList {
    buf: [PairHit; 24],
    len: usize,
}

impl HitList {
    fn new() -> Self {
        let dummy = PairHit {
            loc_a: TriLoc::Inner,
            loc_b: TriLoc::Inner,
            key: VertexKey::OrigA(0),
        };
        HitList {
            buf: [dummy; 24],
            len: 0,
        }
    }

    fn push(&mut self, hit: PairHit) {
        if !self.buf[..self.len].iter().any(|h| h.key == hit.key) {
            self.buf[self.len] = hit;
            self.len += 1;
        }
    }

    fn as_slice(&self) -> &[PairHit] {
        &self.buf[..self.len]
    }
}

fn classify_pair(
    a: &Mesh,
    ta_id: TriangleId,
    tool: &Mesh,
    tb_id: TriangleId,
    raw: &mut RawBuffers,
) -> Result<()> {
    let (Some(ta), Some(tb)) = (a.triangle(ta_id), tool.triangle(tb_id)) else {
        return Ok(());
    };
    if ta[0] == ta[1] || ta[1] == ta[2] || ta[0] == ta[2] {
        return Err(OpsError::InvalidInput(
            "boolean input has degenerate triangles",
        ));
    }
    if tb[0] == tb[1] || tb[1] == tb[2] || tb[0] == tb[2] {
        return Err(OpsError::InvalidInput(
            "boolean input has degenerate triangles",
        ));
    }
    let pa = ta.map(|v| pos_f64(a, v));
    let pb = tb.map(|v| pos_f64(tool, v));
    let Some(plane_a) = Plane2D::from_tri(&pa) else {
        return Err(OpsError::InvalidInput(
            "boolean input has degenerate triangles",
        ));
    };
    let Some(plane_b) = Plane2D::from_tri(&pb) else {
        return Err(OpsError::InvalidInput(
            "boolean input has degenerate triangles",
        ));
    };

    let da = pa.map(|p| orient3d_exact(pb[0], pb[1], pb[2], p));
    let db = pb.map(|p| orient3d_exact(pa[0], pa[1], pa[2], p));

    let mut hits = HitList::new();

    if da == [0; 3] && db == [0; 3] {
        // Coplanar pair (§5.3): project onto `ta`'s dominant plane.
        let ta2 = pa.map(|p| plane_a.project(p));
        let tb2 = pb.map(|p| plane_a.project(p));
        let sign_b = orient2d_exact(tb2[0], tb2[1], tb2[2]);
        debug_assert_ne!(sign_b, 0);

        for (i, &p2) in ta2.iter().enumerate() {
            if let Some(loc_b) = classify_in_tri_2d(p2, &tb2, sign_b) {
                let h = raw.record_hit(ta_id, &ta, TriLoc::Corner(i), tb_id, &tb, loc_b)?;
                hits.push(h);
            }
        }
        for (j, &q2) in tb2.iter().enumerate() {
            if let Some(loc_a) = classify_in_tri_2d(q2, &ta2, 1) {
                let h = raw.record_hit(ta_id, &ta, loc_a, tb_id, &tb, TriLoc::Corner(j))?;
                hits.push(h);
            }
        }
        for i in 0..3 {
            let (a0, a1) = (ta2[i], ta2[(i + 1) % 3]);
            for j in 0..3 {
                let (b0, b1) = (tb2[j], tb2[(j + 1) % 3]);
                let d1 = orient2d_exact(a0, a1, b0);
                let d2 = orient2d_exact(a0, a1, b1);
                let d3 = orient2d_exact(b0, b1, a0);
                let d4 = orient2d_exact(b0, b1, a1);
                if d1 * d2 < 0 && d3 * d4 < 0 {
                    let h =
                        raw.record_hit(ta_id, &ta, TriLoc::Edge(i), tb_id, &tb, TriLoc::Edge(j))?;
                    hits.push(h);
                }
            }
        }

        // Clip each edge `j` of `tb` against `ta`.
        for j in 0..3 {
            let mut on_eb = [(0.0f64, VertexKey::OrigA(0)); 8];
            let mut n = 0usize;
            let dir = sub3(pb[(j + 1) % 3], pb[j]);
            for h in hits.as_slice() {
                let on_edge = matches!(h.loc_b, TriLoc::Edge(e) if e == j)
                    || matches!(h.loc_b, TriLoc::Corner(c) if c == j || c == (j + 1) % 3);
                if on_edge {
                    let t = dot3(sub3(key_pos_f64(a, tool, h.key), pb[j]), dir);
                    on_eb[n] = (t, h.key);
                    n += 1;
                }
            }
            on_eb[..n].sort_unstable_by(|x, y| x.0.total_cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
            for w in on_eb[..n].windows(2) {
                raw.record_seg(ta_id, tb_id, w[0].1, w[1].1)?;
            }
        }

        // Clip each edge `i` of `ta` against `tb`.
        for i in 0..3 {
            let mut on_ea = [(0.0f64, VertexKey::OrigA(0)); 8];
            let mut n = 0usize;
            let dir = sub3(pa[(i + 1) % 3], pa[i]);
            for h in hits.as_slice() {
                let on_edge = matches!(h.loc_a, TriLoc::Edge(e) if e == i)
                    || matches!(h.loc_a, TriLoc::Corner(c) if c == i || c == (i + 1) % 3);
                if on_edge {
                    let t = dot3(sub3(key_pos_f64(a, tool, h.key), pa[i]), dir);
                    on_ea[n] = (t, h.key);
                    n += 1;
                }
            }
            on_ea[..n].sort_unstable_by(|x, y| x.0.total_cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
            for w in on_ea[..n].windows(2) {
                raw.record_seg(ta_id, tb_id, w[0].1, w[1].1)?;
            }
        }
        return Ok(());
    }

    // Non-coplanar transverse / touching pair (§5.3).
    let ta2 = pa.map(|p| plane_a.project(p));
    let tb2 = pb.map(|p| plane_b.project(p));

    for i in 0..3 {
        if da[i] == 0
            && let Some(loc_b) = classify_in_tri_2d(plane_b.project(pa[i]), &tb2, 1)
        {
            let h = raw.record_hit(ta_id, &ta, TriLoc::Corner(i), tb_id, &tb, loc_b)?;
            hits.push(h);
        }
    }
    for j in 0..3 {
        if db[j] == 0
            && let Some(loc_a) = classify_in_tri_2d(plane_a.project(pb[j]), &ta2, 1)
        {
            let h = raw.record_hit(ta_id, &ta, loc_a, tb_id, &tb, TriLoc::Corner(j))?;
            hits.push(h);
        }
    }
    for i in 0..3 {
        let next = (i + 1) % 3;
        if da[i] * da[next] < 0
            && let Some(loc_b) = classify_straddling_edge_3d(pa[i], da[i], pa[next], &pb)
        {
            let h = raw.record_hit(ta_id, &ta, TriLoc::Edge(i), tb_id, &tb, loc_b)?;
            hits.push(h);
        }
    }
    for j in 0..3 {
        let next = (j + 1) % 3;
        if db[j] * db[next] < 0
            && let Some(loc_a) = classify_straddling_edge_3d(pb[j], db[j], pb[next], &pa)
        {
            let h = raw.record_hit(ta_id, &ta, loc_a, tb_id, &tb, TriLoc::Edge(j))?;
            hits.push(h);
        }
    }

    let slice = hits.as_slice();
    if slice.len() == 2 {
        raw.record_seg(ta_id, tb_id, slice[0].key, slice[1].key)?;
    } else if slice.len() > 2 {
        let p0 = key_pos_f64(a, tool, slice[0].key);
        let p_last = key_pos_f64(a, tool, slice[slice.len() - 1].key);
        let dir = sub3(p_last, p0);
        let mut line_pts = [(0.0f64, VertexKey::OrigA(0)); 24];
        for (idx, h) in slice.iter().enumerate() {
            line_pts[idx] = (dot3(sub3(key_pos_f64(a, tool, h.key), p0), dir), h.key);
        }
        let sub_slice = &mut line_pts[..slice.len()];
        sub_slice.sort_unstable_by(|x, y| x.0.total_cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
        for w in sub_slice.windows(2) {
            raw.record_seg(ta_id, tb_id, w[0].1, w[1].1)?;
        }
    }
    Ok(())
}

/// Finds every alive triangle of `mesh` that has an edge split, inner vertex,
/// or segment in the raw buffers, in ascending `TriangleId` order.
fn collect_candidate_triangles(
    mesh: &Mesh,
    edge_splits_raw: &[((VertexId, VertexId), VertexKey)],
    inner_raw: &[(TriangleId, VertexKey)],
    segs_raw: &[(TriangleId, VertexKey, VertexKey)],
) -> Result<Vec<TriangleId>> {
    let mut out = Vec::new();
    let has_edge_splits = !edge_splits_raw.is_empty();
    let (min_v, max_v) = if has_edge_splits {
        let lo = edge_splits_raw[0].0.0;
        let hi = edge_splits_raw
            .iter()
            .fold(0, |acc, &((_, v), _)| acc.max(v));
        (lo, hi)
    } else {
        (u32::MAX, 0)
    };

    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        let has_seg = segs_raw
            .binary_search_by_key(&t, |&(tid, _, _)| tid)
            .is_ok();
        let has_inner = !has_seg && inner_raw.binary_search_by_key(&t, |&(tid, _)| tid).is_ok();
        let has_split = !has_seg
            && !has_inner
            && has_edge_splits
            && [0, 1, 2].into_iter().any(|i| {
                let e = canon_edge(tri[i], tri[(i + 1) % 3]);
                e.0 >= min_v
                    && e.0 <= max_v
                    && edge_splits_raw
                        .binary_search_by_key(&e, |&(ek, _)| ek)
                        .is_ok()
            });
        if has_seg || has_inner || has_split {
            push(&mut out, t)?;
        }
    }
    Ok(out)
}

type SplitTable = Vec<((VertexId, VertexId), Vec<ArrVertexId>)>;

fn finalize_edge_splits(
    edge_splits_raw: &[((VertexId, VertexId), VertexKey)],
    raw_keys: &[VertexKey],
    key_to_arr: &[ArrVertexId],
    corner_map: &[(VertexId, ArrVertexId)],
    positions: &[[f32; 3]],
) -> Result<SplitTable> {
    let mut out: SplitTable = Vec::new();
    let mut i = 0usize;
    while i < edge_splits_raw.len() {
        let edge = edge_splits_raw[i].0;
        let mut j = i + 1;
        while j < edge_splits_raw.len() && edge_splits_raw[j].0 == edge {
            j += 1;
        }
        let id_u = corner_map[corner_map
            .binary_search_by_key(&edge.0, |&(v, _)| v)
            .expect("split edge endpoint u allocated")]
        .1;
        let id_v = corner_map[corner_map
            .binary_search_by_key(&edge.1, |&(v, _)| v)
            .expect("split edge endpoint v allocated")]
        .1;
        let pu = positions[id_u as usize].map(f64::from);
        let pv = positions[id_v as usize].map(f64::from);
        let dir = sub3(pv, pu);

        let mut splits: Vec<ArrVertexId> = Vec::new();
        for &(_, k) in &edge_splits_raw[i..j] {
            let r = raw_keys.binary_search(&k).expect("split key registered");
            let s = key_to_arr[r];
            if s != id_u && s != id_v {
                push(&mut splits, s)?;
            }
        }
        splits.sort_unstable_by(|&a_id, &b_id| {
            let pa = positions[a_id as usize].map(f64::from);
            let pb = positions[b_id as usize].map(f64::from);
            let ta = dot3(sub3(pa, pu), dir);
            let tb = dot3(sub3(pb, pu), dir);
            ta.total_cmp(&tb).then_with(|| a_id.cmp(&b_id))
        });
        splits.dedup();
        if !splits.is_empty() {
            push(&mut out, (edge, splits))?;
        }
        i = j;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn assemble_side_constraints(
    mesh: &Mesh,
    cand_tris: &[TriangleId],
    corner_map: &[(VertexId, ArrVertexId)],
    split_table: &SplitTable,
    inner_raw: &[(TriangleId, VertexKey)],
    segs_raw: &[(TriangleId, VertexKey, VertexKey)],
    raw_keys: &[VertexKey],
    key_to_arr: &[ArrVertexId],
    seam_edges: &mut Vec<(ArrVertexId, ArrVertexId)>,
) -> Result<Vec<(TriangleId, TriConstraints)>> {
    let mut out = Vec::new();
    for &t in cand_tris {
        let tri = mesh.triangle(t).expect("live candidate triangle");
        let corners = tri.map(|v| {
            corner_map[corner_map
                .binary_search_by_key(&v, |&(vid, _)| vid)
                .expect("candidate corner allocated")]
            .1
        });
        let mut tc = TriConstraints::default();

        // 1. Edge splits in exact directed half-edge order.
        for i in 0..3 {
            let (u, v) = (tri[i], tri[(i + 1) % 3]);
            let e = canon_edge(u, v);
            if let Ok(idx) = split_table.binary_search_by_key(&e, |&(ek, _)| ek) {
                let canon_splits = &split_table[idx].1;
                reserve(&mut tc.edge_splits[i], canon_splits.len())?;
                if u < v {
                    tc.edge_splits[i].extend_from_slice(canon_splits);
                } else {
                    tc.edge_splits[i].extend(canon_splits.iter().copied().rev());
                }
            }
        }

        // 2. Strictly interior vertices (filtering any vertex that unified onto the boundary).
        let in_lo = inner_raw.partition_point(|&(tid, _)| tid < t);
        let in_hi = in_lo + inner_raw[in_lo..].partition_point(|&(tid, _)| tid == t);
        for &(_, k) in &inner_raw[in_lo..in_hi] {
            let v = key_to_arr[raw_keys.binary_search(&k).expect("inner key registered")];
            let on_boundary =
                corners.contains(&v) || tc.edge_splits.iter().any(|es| es.contains(&v));
            if !on_boundary {
                push(&mut tc.inner_verts, v)?;
            }
        }
        tc.inner_verts.sort_unstable();
        tc.inner_verts.dedup();

        // 3. Constraint segments (decomposing boundary segments along edge_splits chains).
        let pos_on_edge =
            |edge_idx: usize, x: ArrVertexId, es: &[Vec<ArrVertexId>; 3]| -> Option<usize> {
                if x == corners[edge_idx] {
                    Some(0)
                } else if let Some(p) = es[edge_idx].iter().position(|&s| s == x) {
                    Some(1 + p)
                } else if x == corners[(edge_idx + 1) % 3] {
                    Some(1 + es[edge_idx].len())
                } else {
                    None
                }
            };
        let chain_at = |edge_idx: usize, idx: usize, es: &[Vec<ArrVertexId>; 3]| -> ArrVertexId {
            if idx == 0 {
                corners[edge_idx]
            } else if idx == 1 + es[edge_idx].len() {
                corners[(edge_idx + 1) % 3]
            } else {
                es[edge_idx][idx - 1]
            }
        };

        let seg_lo = segs_raw.partition_point(|&(tid, _, _)| tid < t);
        let seg_hi = seg_lo + segs_raw[seg_lo..].partition_point(|&(tid, _, _)| tid == t);
        for &(_, k0, k1) in &segs_raw[seg_lo..seg_hi] {
            let u = key_to_arr[raw_keys.binary_search(&k0).expect("seg k0 registered")];
            let v = key_to_arr[raw_keys.binary_search(&k1).expect("seg k1 registered")];
            if u == v {
                continue;
            }
            if let Some((edge_idx, pu, pv)) = (0..3).find_map(|i| {
                Some((
                    i,
                    pos_on_edge(i, u, &tc.edge_splits)?,
                    pos_on_edge(i, v, &tc.edge_splits)?,
                ))
            }) {
                let (lo, hi) = (pu.min(pv), pu.max(pv));
                for m in lo..hi {
                    let x = chain_at(edge_idx, m, &tc.edge_splits);
                    let y = chain_at(edge_idx, m + 1, &tc.edge_splits);
                    if x != y {
                        let seg = (x.min(y), x.max(y));
                        push(&mut tc.segments, seg)?;
                        push(seam_edges, seg)?;
                    }
                }
            } else {
                let seg = (u.min(v), u.max(v));
                push(&mut tc.segments, seg)?;
                push(seam_edges, seg)?;
            }
        }
        tc.segments.sort_unstable();
        tc.segments.dedup();

        if tc.edge_splits.iter().any(|es| !es.is_empty())
            || !tc.inner_verts.is_empty()
            || !tc.segments.is_empty()
        {
            push(&mut out, (t, tc))?;
        }
    }
    Ok(out)
}

pub(super) fn build_arrangement(
    a: &Mesh,
    bvh_a: &Bvh,
    tool: &Mesh,
    bvh_tool: &Bvh,
) -> Result<Arrangement> {
    allocation_allowed()?;
    let mut pairs: Vec<(TriangleId, TriangleId)> = Vec::new();
    intersecting_pairs(a, bvh_a, tool, bvh_tool, &Rigid::IDENTITY, &mut pairs)?;
    if pairs.is_empty() {
        return Ok(Arrangement::default());
    }

    let mut raw = RawBuffers::default();
    for &(ta, tb) in &pairs {
        classify_pair(a, ta, tool, tb, &mut raw)?;
    }

    raw.edge_splits_a.sort_unstable();
    raw.edge_splits_a.dedup();
    raw.edge_splits_b.sort_unstable();
    raw.edge_splits_b.dedup();
    raw.inner_a.sort_unstable();
    raw.inner_a.dedup();
    raw.inner_b.sort_unstable();
    raw.inner_b.dedup();
    raw.segs_a.sort_unstable();
    raw.segs_a.dedup();
    raw.segs_b.sort_unstable();
    raw.segs_b.dedup();

    let cand_tris_a =
        collect_candidate_triangles(a, &raw.edge_splits_a, &raw.inner_a, &raw.segs_a)?;
    let cand_tris_b =
        collect_candidate_triangles(tool, &raw.edge_splits_b, &raw.inner_b, &raw.segs_b)?;

    // Ensure all 3 corners of every candidate cut triangle have allocated keys.
    for &t in &cand_tris_a {
        for v in a.triangle(t).expect("live candidate in A") {
            push(&mut raw.keys, VertexKey::OrigA(v))?;
        }
    }
    for &t in &cand_tris_b {
        for v in tool.triangle(t).expect("live candidate in tool") {
            push(&mut raw.keys, VertexKey::OrigB(v))?;
        }
    }
    raw.keys.sort_unstable();
    raw.keys.dedup();

    // Realize canonical [f32; 3] positions once per VertexKey (§5.2, §5.4).
    let num_raw = raw.keys.len();
    let mut raw_pos: Vec<[f32; 3]> = Vec::new();
    let mut raw_origin: Vec<ArrOrigin> = Vec::new();
    reserve(&mut raw_pos, num_raw)?;
    reserve(&mut raw_origin, num_raw)?;
    for &k in &raw.keys {
        let (p, orig) = key_pos_and_origin(a, tool, k)?;
        raw_pos.push(p);
        raw_origin.push(orig);
    }

    // Unify all keys that quantize to identical [f32; 3] bit patterns (flushing
    // sub-f64 near-zero trigonometric residuals <= 1e-15 * scale onto 0.0 so exact
    // zero axes share bit patterns with ~6e-17 sin/cos artifacts), preferring LiveA.
    let max_coord = raw_pos
        .iter()
        .flatten()
        .fold(0.0_f64, |acc, &c| acc.max(f64::from(c).abs()));
    let flush_zero_tol = (max_coord * 1e-15).max(1e-24) as f32;
    let mut by_bits: Vec<([u32; 3], usize)> = Vec::new();
    reserve(&mut by_bits, num_raw)?;
    for (idx, &p) in raw_pos.iter().enumerate() {
        let flushed = p.map(|c| if c.abs() <= flush_zero_tol { 0.0 } else { c });
        by_bits.push((pos_bits(flushed), idx));
    }
    by_bits.sort_unstable();

    // Group identical bit patterns -> (first_raw_idx, chosen_pos, chosen_origin, group_range).
    let mut groups: Vec<(usize, [f32; 3], ArrOrigin, usize, usize)> = Vec::new();
    let mut g_start = 0usize;
    while g_start < by_bits.len() {
        let bits = by_bits[g_start].0;
        let mut g_end = g_start + 1;
        while g_end < by_bits.len() && by_bits[g_end].0 == bits {
            g_end += 1;
        }
        let first_raw_idx = by_bits[g_start].1;
        let mut live_va: Option<VertexId> = None;
        for &(_, r_idx) in &by_bits[g_start..g_end] {
            if let ArrOrigin::LiveA(va) = raw_origin[r_idx] {
                live_va = Some(live_va.map_or(va, |cur| cur.min(va)));
            }
        }
        let (chosen_pos, chosen_origin) = match live_va {
            Some(va) => (a.position(va).expect("live A vertex"), ArrOrigin::LiveA(va)),
            None => (raw_pos[first_raw_idx], ArrOrigin::Fresh),
        };
        push(
            &mut groups,
            (first_raw_idx, chosen_pos, chosen_origin, g_start, g_end),
        )?;
        g_start = g_end;
    }
    // Order unified vertices deterministically by their earliest canonical VertexKey index.
    groups.sort_unstable_by_key(|&(first_idx, _, _, _, _)| first_idx);

    let num_arr = groups.len();
    let _ = ArrVertexId::try_from(num_arr).map_err(|_| oom())?;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut origins: Vec<ArrOrigin> = Vec::new();
    let mut key_to_arr: Vec<ArrVertexId> = Vec::new();
    reserve(&mut positions, num_arr)?;
    reserve(&mut origins, num_arr)?;
    reserve(&mut key_to_arr, num_raw)?;
    key_to_arr.resize(num_raw, 0);

    for (arr_idx, &(_, pos, orig, gs, ge)) in groups.iter().enumerate() {
        let arr_id = arr_idx as ArrVertexId;
        positions.push(pos);
        origins.push(orig);
        for &(_, r_idx) in &by_bits[gs..ge] {
            key_to_arr[r_idx] = arr_id;
        }
    }

    // Build sorted corner lookup tables for corners_a / corners_b.
    let mut corner_a: Vec<(VertexId, ArrVertexId)> = Vec::new();
    let mut corner_b: Vec<(VertexId, ArrVertexId)> = Vec::new();
    for (r_idx, &k) in raw.keys.iter().enumerate() {
        match k {
            VertexKey::OrigA(va) => push(&mut corner_a, (va, key_to_arr[r_idx]))?,
            VertexKey::OrigB(vb) => push(&mut corner_b, (vb, key_to_arr[r_idx]))?,
            _ => {}
        }
    }

    let split_table_a = finalize_edge_splits(
        &raw.edge_splits_a,
        &raw.keys,
        &key_to_arr,
        &corner_a,
        &positions,
    )?;
    let split_table_b = finalize_edge_splits(
        &raw.edge_splits_b,
        &raw.keys,
        &key_to_arr,
        &corner_b,
        &positions,
    )?;

    let mut seam_edges: Vec<(ArrVertexId, ArrVertexId)> = Vec::new();
    let tri_a = assemble_side_constraints(
        a,
        &cand_tris_a,
        &corner_a,
        &split_table_a,
        &raw.inner_a,
        &raw.segs_a,
        &raw.keys,
        &key_to_arr,
        &mut seam_edges,
    )?;
    let tri_b = assemble_side_constraints(
        tool,
        &cand_tris_b,
        &corner_b,
        &split_table_b,
        &raw.inner_b,
        &raw.segs_b,
        &raw.keys,
        &key_to_arr,
        &mut seam_edges,
    )?;
    seam_edges.sort_unstable();
    seam_edges.dedup();

    Ok(Arrangement {
        positions,
        origins,
        tri_a,
        tri_b,
        seam_edges,
        corner_a,
        corner_b,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use fast_mesh_bvh::{BuildOptions, Bvh};
    use fast_mesh_query::Rigid;
    use fast_mesh_store::{Mesh, StoreError, TriangleId, VertexId};

    use super::*;
    use crate::test_util::{assert_identical_across_threads, box_mesh, mesh_of, uv_sphere};
    use crate::{OpsError, fail_allocations_after};

    fn bvh(mesh: &Mesh) -> Bvh {
        Bvh::build(mesh, &BuildOptions::default()).unwrap()
    }

    fn edge_key(u: ArrVertexId, v: ArrVertexId) -> (ArrVertexId, ArrVertexId) {
        (u.min(v), u.max(v))
    }

    /// Collects every undirected segment represented on one side (both explicit
    /// `tc.segments` and elementary split boundary sub-edges).
    fn side_represented_edges(
        arr: &Arrangement,
        mesh: &Mesh,
        is_a: bool,
        tris: &[(TriangleId, TriConstraints)],
    ) -> BTreeSet<(ArrVertexId, ArrVertexId)> {
        let mut out = BTreeSet::new();
        for (t, tc) in tris {
            let corners = if is_a {
                arr.corners_a(mesh, *t)
            } else {
                arr.corners_b(mesh, *t)
            };
            for &(u, v) in &tc.segments {
                assert!(u < v, "segment ({u}, {v}) must be canonical u < v");
                out.insert((u, v));
            }
            for i in 0..3 {
                let mut chain = Vec::with_capacity(2 + tc.edge_splits[i].len());
                chain.push(corners[i]);
                chain.extend_from_slice(&tc.edge_splits[i]);
                chain.push(corners[(i + 1) % 3]);
                for w in chain.windows(2) {
                    assert_ne!(w[0], w[1], "zero-length sub-edge on boundary chain");
                    out.insert(edge_key(w[0], w[1]));
                }
            }
        }
        out
    }

    /// Asserts that `edge_splits` match in exact reverse order across every
    /// shared undirected mesh edge where either adjacent triangle is cut.
    fn assert_consistent_edge_splits(mesh: &Mesh, tris: &[(TriangleId, TriConstraints)]) {
        type EdgeUse = (bool, Vec<ArrVertexId>);
        let by_tri: BTreeMap<TriangleId, &TriConstraints> =
            tris.iter().map(|(t, tc)| (*t, tc)).collect();
        let mut by_edge: BTreeMap<(VertexId, VertexId), Vec<EdgeUse>> = BTreeMap::new();
        for t in 0..mesh.triangle_slots() as TriangleId {
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };
            let splits = by_tri.get(&t);
            for i in 0..3 {
                let (u, v) = (tri[i], tri[(i + 1) % 3]);
                let key = (u.min(v), u.max(v));
                let forward = u < v;
                let s = splits.map_or_else(Vec::new, |tc| tc.edge_splits[i].clone());
                by_edge.entry(key).or_default().push((forward, s));
            }
        }
        for (ekey, uses) in by_edge {
            if uses.iter().all(|(_, s)| s.is_empty()) {
                continue;
            }
            assert_eq!(uses.len(), 2, "closed mesh edge {ekey:?} has 2 triangles");
            let canon = |&(fwd, ref s): &(bool, Vec<ArrVertexId>)| -> Vec<ArrVertexId> {
                if fwd {
                    s.clone()
                } else {
                    s.iter().copied().rev().collect()
                }
            };
            assert_eq!(
                canon(&uses[0]),
                canon(&uses[1]),
                "mismatched edge splits across shared mesh edge {ekey:?}"
            );
        }
    }

    /// Asserts `seam_edges` forms a single closed loop with every incident vertex of degree 2.
    fn assert_single_closed_seam_loop(seam_edges: &[(ArrVertexId, ArrVertexId)]) {
        assert!(!seam_edges.is_empty(), "seam_edges must not be empty");
        for w in seam_edges.windows(2) {
            assert!(
                w[0] < w[1],
                "seam_edges must be strictly sorted and deduplicated"
            );
        }
        let mut adj: BTreeMap<ArrVertexId, Vec<ArrVertexId>> = BTreeMap::new();
        for &(u, v) in seam_edges {
            assert!(u < v, "seam edge ({u}, {v}) must have u < v");
            adj.entry(u).or_default().push(v);
            adj.entry(v).or_default().push(u);
        }
        for (&v, nbrs) in &adj {
            assert_eq!(
                nbrs.len(),
                2,
                "seam vertex {v} must have degree 2, got {}",
                nbrs.len()
            );
        }
        // Verify connectivity: DFS from the first vertex visits every vertex in `adj`.
        let start = *adj.keys().next().unwrap();
        let mut visited = BTreeSet::new();
        let mut stack = vec![start];
        while let Some(u) = stack.pop() {
            if visited.insert(u) {
                for &n in &adj[&u] {
                    stack.push(n);
                }
            }
        }
        assert_eq!(
            visited.len(),
            adj.len(),
            "seam_edges must form a single connected loop"
        );
    }

    #[test]
    fn orient3d_exact_follows_right_hand_rule() {
        let o = [0.0, 0.0, 0.0];
        let x = [1.0, 0.0, 0.0];
        let y = [0.0, 1.0, 0.0];
        assert_eq!(orient3d_exact(o, x, y, [0.0, 0.0, 1.0]), 1);
        assert_eq!(orient3d_exact(o, x, y, [0.0, 0.0, -1.0]), -1);
        assert_eq!(orient3d_exact(o, x, y, [0.25, 0.75, 0.0]), 0);
        let tiny = f64::from(f32::from_bits(1));
        assert_eq!(orient3d_exact(o, x, y, [0.25, 0.25, tiny]), 1);
        assert_eq!(orient3d_exact(o, x, y, [0.25, 0.25, -tiny]), -1);
    }

    #[test]
    fn cube_offset_shares_exact_seam_vertices_without_gaps() {
        let a = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let tool = box_mesh([0.5, 0.25, 0.125], [1.5, 1.25, 1.125]);
        let (bvh_a, bvh_tool) = (bvh(&a), bvh(&tool));

        let arr = build_arrangement(&a, &bvh_a, &tool, &bvh_tool).unwrap();
        assert_single_closed_seam_loop(&arr.seam_edges);

        let edges_a = side_represented_edges(&arr, &a, true, &arr.tri_a);
        let edges_b = side_represented_edges(&arr, &tool, false, &arr.tri_b);
        for &seam in &arr.seam_edges {
            assert!(
                edges_a.contains(&seam),
                "seam edge {seam:?} missing from tri_a representation"
            );
            assert!(
                edges_b.contains(&seam),
                "seam edge {seam:?} missing from tri_b representation"
            );
        }
        assert_consistent_edge_splits(&a, &arr.tri_a);
        assert_consistent_edge_splits(&tool, &arr.tri_b);
    }

    #[test]
    fn curved_spheres_form_closed_degree_two_seam_loop() {
        let a = uv_sphere(32, 32, 1.0);
        let b = uv_sphere(32, 32, 1.0);
        let xf = Rigid {
            translation: [0.31, 0.17, 0.23],
            ..Rigid::IDENTITY
        };
        let tool = super::super::map_tool(&b, &xf).unwrap();
        let (bvh_a, bvh_tool) = (bvh(&a), bvh(&tool));

        let arr = build_arrangement(&a, &bvh_a, &tool, &bvh_tool).unwrap();
        assert_single_closed_seam_loop(&arr.seam_edges);
        assert_consistent_edge_splits(&a, &arr.tri_a);
        assert_consistent_edge_splits(&tool, &arr.tri_b);

        // Also verify axis-aligned offset (meridian EdgeEdge crossings) and heterogeneous spheres.
        for (b_mesh, shift) in [
            (uv_sphere(32, 32, 1.0), [0.31, 0.0, 0.0]),
            (uv_sphere(39, 43, 0.9), [0.31, 0.17, 0.23]),
        ] {
            let xf_case = Rigid {
                translation: shift,
                ..Rigid::IDENTITY
            };
            let tool_case = super::super::map_tool(&b_mesh, &xf_case).unwrap();
            let bvh_case = bvh(&tool_case);
            let arr_case = build_arrangement(&a, &bvh_a, &tool_case, &bvh_case).unwrap();
            assert_single_closed_seam_loop(&arr_case.seam_edges);
            assert_consistent_edge_splits(&a, &arr_case.tri_a);
            assert_consistent_edge_splits(&tool_case, &arr_case.tri_b);
        }
    }

    #[test]
    fn coplanar_stacked_cubes_unify_shared_corners_and_edges() {
        let a = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let tool = box_mesh([0.0, 0.0, 1.0], [1.0, 1.0, 2.0]);
        let (bvh_a, bvh_tool) = (bvh(&a), bvh(&tool));

        let arr = build_arrangement(&a, &bvh_a, &tool, &bvh_tool).unwrap();
        assert!(!arr.tri_a.is_empty());
        assert!(!arr.tri_b.is_empty());

        // No two arrangement vertices may share identical [f32; 3] coordinates.
        let mut seen_bits = BTreeSet::new();
        for p in &arr.positions {
            let bits = p.map(|c| if c == 0.0 { 0u32 } else { c.to_bits() });
            assert!(seen_bits.insert(bits), "duplicate vertex position {p:?}");
        }

        // All 4 shared z = 1 corners must reuse ArrOrigin::LiveA(va).
        let shared_coords = [
            [0.0f32, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        for coord in shared_coords {
            let expected_va = (0..a.vertex_slots() as VertexId)
                .find(|&v| a.position(v) == Some(coord))
                .expect("corner in A");
            let hits: Vec<usize> = arr
                .positions
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| (p == coord).then_some(i))
                .collect();
            assert_eq!(hits.len(), 1, "shared corner {coord:?} must appear once");
            assert_eq!(arr.origins[hits[0]], ArrOrigin::LiveA(expected_va));
        }

        // Shared edges carry no duplicate coincident split vertices.
        for (_, tc) in arr.tri_a.iter().chain(&arr.tri_b) {
            for splits in &tc.edge_splits {
                assert!(
                    splits.is_empty(),
                    "expected no interior splits on aligned stacked cubes, got {splits:?}"
                );
            }
        }
    }

    #[test]
    fn vertex_touching_shared_edge_propagates_split_to_both_triangles() {
        let a = box_mesh([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let tet_positions = [
            2.0f32, 1.0, 2.0, // 0: touches midpoint of A's edge (2,0,2)-(2,2,2)
            3.0, 0.5, 3.0, // 1
            3.0, 1.5, 3.0, // 2
            2.5, 1.0, 3.5, // 3
        ];
        let tet_tris = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
        let tool = mesh_of(&tet_positions, &tet_tris);
        let (bvh_a, bvh_tool) = (bvh(&a), bvh(&tool));

        let arr = build_arrangement(&a, &bvh_a, &tool, &bvh_tool).unwrap();
        assert_consistent_edge_splits(&a, &arr.tri_a);

        let v_lo = (0..a.vertex_slots() as VertexId)
            .find(|&v| a.position(v) == Some([2.0, 0.0, 2.0]))
            .unwrap();
        let v_hi = (0..a.vertex_slots() as VertexId)
            .find(|&v| a.position(v) == Some([2.0, 2.0, 2.0]))
            .unwrap();
        let ekey = (v_lo.min(v_hi), v_lo.max(v_hi));
        let mut incident_in_tri_a = 0usize;
        for (t, tc) in &arr.tri_a {
            let tri = a.triangle(*t).unwrap();
            for i in 0..3 {
                let k = (tri[i].min(tri[(i + 1) % 3]), tri[i].max(tri[(i + 1) % 3]));
                if k == ekey {
                    assert_eq!(tc.edge_splits[i].len(), 1);
                    let split_id = tc.edge_splits[i][0];
                    assert_eq!(arr.positions[split_id as usize], [2.0, 1.0, 2.0]);
                    incident_in_tri_a += 1;
                }
            }
        }
        assert_eq!(
            incident_in_tri_a, 2,
            "both incident triangles on A must receive the edge split"
        );
    }

    #[test]
    fn quantization_collapse_prefers_live_a_and_drops_zero_segments() {
        let a = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let sub_q = f32::from_bits(0.25f32.to_bits() - 1); // 0.25 - 2^-26, exact in f32
        let tet_positions = [
            sub_q, 0.5, 1.75, // 0: x + z = 2 - 2^-26
            sub_q, 1.5, 1.75, // 1: x + z = 2 - 2^-26
            1.75, 1.0, sub_q, // 2: x + z = 2 - 2^-26
            2.0, 1.0, 2.0, // 3: apex outside A so winding is outward
        ];
        let tet_tris = [[0, 1, 2], [0, 2, 3], [1, 3, 2], [0, 3, 1]];
        let tool = mesh_of(&tet_positions, &tet_tris);
        let (bvh_a, bvh_tool) = (bvh(&a), bvh(&tool));

        let arr = build_arrangement(&a, &bvh_a, &tool, &bvh_tool).unwrap();
        let va = (0..a.vertex_slots() as VertexId)
            .find(|&v| a.position(v) == Some([1.0, 1.0, 1.0]))
            .unwrap();
        let hits: Vec<usize> = arr
            .positions
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| (p == [1.0, 1.0, 1.0]).then_some(i))
            .collect();
        assert_eq!(
            hits.len(),
            1,
            "[1, 1, 1] must be unified to a single ArrVertexId"
        );
        let id_va = hits[0] as ArrVertexId;
        assert_eq!(arr.origins[hits[0]], ArrOrigin::LiveA(va));

        for &(u, v) in &arr.seam_edges {
            assert_ne!(u, v, "zero-length collapsed segment must be dropped");
        }
        for (t, tc) in &arr.tri_a {
            let corners = arr.corners_a(&a, *t);
            for i in 0..3 {
                assert!(
                    !tc.edge_splits[i].contains(&corners[i])
                        && !tc.edge_splits[i].contains(&corners[(i + 1) % 3]),
                    "collapsed split must not remain on its own edge endpoint"
                );
            }
            assert!(
                !tc.inner_verts.contains(&id_va) || !corners.contains(&id_va),
                "corner must not be duplicated in inner_verts"
            );
            for &(u, v) in &tc.segments {
                assert_ne!(u, v);
            }
        }
    }

    #[test]
    fn arrangement_is_oom_safe_and_deterministic() {
        let a = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let tool = box_mesh([0.5, 0.25, 0.125], [1.5, 1.25, 1.125]);
        let (bvh_a, bvh_tool) = (bvh(&a), bvh(&tool));

        let reference = assert_identical_across_threads("build_arrangement", || {
            build_arrangement(&a, &bvh_a, &tool, &bvh_tool).unwrap()
        });
        assert!(!reference.seam_edges.is_empty());

        let mut n = 0usize;
        loop {
            fail_allocations_after(Some(n));
            let r = build_arrangement(&a, &bvh_a, &tool, &bvh_tool);
            fail_allocations_after(None);
            match r {
                Ok(got) => {
                    assert!(n > 0, "arrangement on intersecting cubes must allocate");
                    assert_eq!(got, reference);
                    break;
                }
                Err(e) => {
                    assert_eq!(e, OpsError::Store(StoreError::OutOfMemory));
                }
            }
            n += 1;
        }
    }

    #[test]
    fn grazing_edge_edge_crossing_avoids_cancellation() {
        let h = 2.0_f32.powi(-28);
        let a_positions = [
            0.0f32, 0.0, 0.0, // 0
            1.0, 0.0, 0.0, // 1: edge (0, 1) along x-axis
            0.5, 1.0, 0.0, // 2
            0.5, 0.25, 1.0, // 3
        ];
        let a_tris = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
        let a = mesh_of(&a_positions, &a_tris);

        let tool_positions = [
            0.0f32, -h, 0.0, // 0
            1.0, h, 0.0, // 1: crosses A's edge (0, 1) at [0.5, 0.0, 0.0]
            0.5, 0.5, 0.5, // 2
            0.5, 0.5, -0.5, // 3
        ];
        let tool_tris = [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]];
        let tool = mesh_of(&tool_positions, &tool_tris);

        // Direct check on compute_edge_edge_f64 at grazing angle theta ~ 5.4e-9 rad:
        let exact_mid = compute_edge_edge_f64(&a, (0, 1), &tool, (0, 1));
        assert!(
            (exact_mid[0] - 0.5).abs() < 1e-12
                && exact_mid[1].abs() <= f64::from(h)
                && exact_mid[2] == 0.0,
            "expected grazing EdgeEdge crossing at [0.5, 0.0, 0.0], got {exact_mid:?}"
        );

        // Asymmetric grazing crossing at x = 0.25 (s = 0.25):
        let asym_positions = [
            0.0f32,
            -h,
            0.0,
            1.0,
            3.0 * h,
            0.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            -0.5,
        ];
        let asym_tool = mesh_of(&asym_positions, &tool_tris);
        let exact_quarter = compute_edge_edge_f64(&a, (0, 1), &asym_tool, (0, 1));
        assert!(
            (exact_quarter[0] - 0.25).abs() < 1e-12
                && exact_quarter[1].abs() <= f64::from(h)
                && exact_quarter[2] == 0.0,
            "expected asymmetric grazing EdgeEdge crossing at [0.25, 0.0, 0.0], got {exact_quarter:?}"
        );

        // End-to-end arrangement check: must realize [0.5, 0.0, 0.0] as Fresh (never
        // collapsing onto OrigA(0) at [0.0, 0.0, 0.0]) and split both edges cleanly.
        let (bvh_a, bvh_tool) = (bvh(&a), bvh(&tool));
        let arr = build_arrangement(&a, &bvh_a, &tool, &bvh_tool).unwrap();

        let crossing_hits: Vec<usize> = arr
            .positions
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| (p == [0.5, 0.0, 0.0]).then_some(i))
            .collect();
        assert_eq!(
            crossing_hits.len(),
            1,
            "grazing EdgeEdge crossing must materialize at [0.5, 0.0, 0.0], positions: {:?}",
            arr.positions
        );
        let cross_id = crossing_hits[0] as ArrVertexId;
        assert_eq!(arr.origins[crossing_hits[0]], ArrOrigin::Fresh);

        // Both incident triangles of edge (0, 1) in A and in tool must carry `cross_id`.
        let count_split = |mesh: &Mesh, tris: &[(TriangleId, TriConstraints)]| -> usize {
            let mut count = 0usize;
            for (t, tc) in tris {
                let tri = mesh.triangle(*t).unwrap();
                for i in 0..3 {
                    let e = canon_edge(tri[i], tri[(i + 1) % 3]);
                    if e == (0, 1) && tc.edge_splits[i].contains(&cross_id) {
                        count += 1;
                    }
                }
            }
            count
        };
        assert_eq!(count_split(&a, &arr.tri_a), 2);
        assert_eq!(count_split(&tool, &arr.tri_b), 2);
        assert_consistent_edge_splits(&a, &arr.tri_a);
        assert_consistent_edge_splits(&tool, &arr.tri_b);
        assert_single_closed_seam_loop(&arr.seam_edges);
    }
}
