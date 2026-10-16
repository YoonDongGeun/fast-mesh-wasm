//! Exact triangle-triangle intersection.
//!
//! Every decision is the sign of an exact predicate (`robust::orient3d` and
//! `robust::orient2d`) on the given `f64` coordinates, with no epsilon. So
//! triangles that touch at a single point always count as intersecting, and a
//! gap of one `f64` ulp always counts as a gap.
//!
//! The general case is the Guigue-Devillers test. Coplanar triangles are
//! projected onto a coordinate plane and tested edge against edge, plus one
//! vertex of each inside the other. Degenerate triangles (all three corners
//! collinear) are the union of their edges and are tested as segments.

use robust::{Coord, Coord3D, orient2d, orient3d};

use crate::tri::closest_on_triangle;

/// A point in `f64`.
pub(crate) type P3 = [f64; 3];

type P2 = [f64; 2];

/// The sign of `det[b - a, c - a, d - a]`: positive when `d` is on the side of
/// the plane `abc` that `(b - a) × (c - a)` points to.
fn side(a: P3, b: P3, c: P3, d: P3) -> i8 {
    if ![a, b, c, d].iter().flatten().all(|&x| safe(x)) {
        return exact::side(a, b, c, d);
    }
    let c3 = |p: P3| Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    };
    // `orient3d(a, b, c, d)` is `det[a - d, b - d, c - d]`, which is the
    // negation of the determinant above.
    sign(-orient3d(c3(a), c3(b), c3(c), c3(d)))
}

/// The sign of `det[b - a, c - a]`, the 2D orientation of `a`, `b`, `c`.
fn side2(a: P2, b: P2, c: P2) -> i8 {
    if ![a, b, c].iter().flatten().all(|&x| safe(x)) {
        return exact::side2(a, b, c);
    }
    sign(orient2d_approx(a, b, c))
}

/// `robust::orient2d`: `det[b - a, c - a]` approximately, with the exact sign
/// when every coordinate is [`safe`].
fn orient2d_approx(a: P2, b: P2, c: P2) -> f64 {
    let c2 = |p: P2| Coord { x: p[0], y: p[1] };
    orient2d(c2(a), c2(b), c2(c))
}

/// Whether `robust` is exact for a coordinate: zero, or between `2^-200` and
/// `2^200` in magnitude.
///
/// `robust` implements Shewchuk's predicates, which are exact only when no
/// intermediate value underflows or overflows. With every coordinate in this
/// range, every value they compute stays in the normal `f64` range. Outside
/// it (a coordinate such as `2^-1074`, for instance), the predicates fall back
/// to [`exact`].
fn safe(x: f64) -> bool {
    const MIN: f64 = f64::from_bits((1023 - 200) << 52);
    const MAX: f64 = f64::from_bits((1023 + 200) << 52);
    x == 0.0 || (MIN..=MAX).contains(&x.abs())
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

/// `p` with coordinate `axis` dropped.
fn drop(p: P3, axis: usize) -> P2 {
    [p[(axis + 1) % 3], p[(axis + 2) % 3]]
}

/// The axis of the largest component of `t`'s normal, or `None` when the
/// three points are collinear (the normal is exactly zero). Dropping this axis
/// projects `t`'s plane one-to-one onto a coordinate plane.
fn normal_axis(t: &[P3; 3]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for k in 0..3 {
        let [a, b, c] = t.map(|p| drop(p, k));
        // The normal's component `k` is zero exactly when this sign is.
        if side2(a, b, c) == 0 {
            continue;
        }
        // Its size, approximately (only used to pick the largest).
        let size = orient2d_approx(a, b, c).abs();
        if best.is_none_or(|(_, s)| size > s) {
            best = Some((k, size));
        }
    }
    best.map(|(k, _)| k)
}

/// True when the closed triangles `a` and `b` share at least one point.
/// Touching at a vertex or along an edge counts. Exact for the given
/// coordinates.
pub(crate) fn tri_tri_intersect(a: [P3; 3], b: [P3; 3]) -> bool {
    // Where a's corners lie relative to b's plane. All zero when b is
    // degenerate.
    let da = a.map(|p| side(b[0], b[1], b[2], p));
    if same_strict_side(da) {
        return false;
    }
    let db = b.map(|p| side(a[0], a[1], a[2], p));
    if same_strict_side(db) {
        return false;
    }
    if da == [0; 3] || db == [0; 3] {
        // Coplanar, or at least one triangle is degenerate.
        let (na, nb) = (normal_axis(&a), normal_axis(&b));
        return match (na, nb) {
            (Some(axis), Some(_)) => coplanar(a, b, axis),
            _ => {
                (0..3).any(|i| seg_tri(a[i], a[(i + 1) % 3], &b, nb))
                    || (0..3).any(|i| seg_tri(b[i], b[(i + 1) % 3], &a, na))
            }
        };
    }
    guigue_devillers(a, da, b, db)
}

/// The distance between two triangles, with a closest pair of points.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TriTriDist {
    /// Squared distance, exactly `0` when the triangles intersect.
    pub dist2: f64,
    /// The point on `a`.
    pub pa: P3,
    /// The point on `b`.
    pub pb: P3,
}

/// The distance between the closed triangles `a` and `b`.
///
/// When [`tri_tri_intersect`] says they meet, `dist2` is exactly `0` and
/// `pa == pb` is a common point, up to rounding: the midpoint of the
/// closest pair found among the candidates below, which also include the points
/// where an edge of one crosses the plane of the other.
///
/// Otherwise it is the minimum over the 6 vertex-triangle distances and the 9
/// edge-edge distances. Ties go to the first in this order: the vertices of `a`
/// against `b`, then the vertices of `b` against `a`, then edge pairs
/// `(i, j)` in `i * 3 + j` order. This is exact up to `f64` rounding of the
/// closest points.
pub(crate) fn tri_tri_distance(a: [P3; 3], b: [P3; 3]) -> TriTriDist {
    let mut best = TriTriDist {
        dist2: f64::INFINITY,
        pa: a[0],
        pb: b[0],
    };
    let mut offer = |pa: P3, pb: P3| {
        let d = sub3(pa, pb);
        let dist2 = dot3(d, d);
        // `NaN` never wins, and a tie keeps the earlier candidate.
        if dist2 < best.dist2 {
            best = TriTriDist { dist2, pa, pb };
        }
    };
    for v in a {
        offer(v, closest_on_triangle(v, b).point);
    }
    for v in b {
        offer(closest_on_triangle(v, a).point, v);
    }
    for i in 0..3 {
        for j in 0..3 {
            let (pa, pb) = seg_seg_closest(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3]);
            offer(pa, pb);
        }
    }
    if !tri_tri_intersect(a, b) {
        return best;
    }
    // An edge may pierce the other triangle's interior without any vertex or
    // edge pair being near. Offer where each edge crosses the other's plane.
    for (edges, other, flip) in [(a, b, false), (b, a, true)] {
        let n = cross3(sub3(other[1], other[0]), sub3(other[2], other[0]));
        let height = |p: P3| dot3(n, sub3(p, other[0]));
        for i in 0..3 {
            let (p, q) = (edges[i], edges[(i + 1) % 3]);
            let (hp, hq) = (height(p), height(q));
            if hp * hq > 0.0 || hp == hq {
                continue;
            }
            let t = hp / (hp - hq);
            let x: P3 = std::array::from_fn(|k| p[k] + t * (q[k] - p[k]));
            let y = closest_on_triangle(x, other).point;
            if flip {
                offer(y, x);
            } else {
                offer(x, y);
            }
        }
    }
    let mid: P3 = std::array::from_fn(|k| 0.5 * (best.pa[k] + best.pb[k]));
    TriTriDist {
        dist2: 0.0,
        pa: mid,
        pb: mid,
    }
}

/// Whether `tri_tri_distance(a, b).dist2.sqrt() <= bound`, for `bound >= 0`,
/// decided with the same candidates and arithmetic but without finding the
/// minimum: it stops at the first candidate within `bound`, and runs the
/// intersection test only when none is.
///
/// The answers agree exactly. `sqrt` is correctly rounded, hence monotone, so
/// the minimum is within `bound` exactly when some candidate is. A `NaN`
/// candidate is never the minimum nor within `bound`. An intersecting pair has
/// `dist2 == 0`, within any `bound >= 0`. And when no candidate is a number,
/// `dist2` stays infinite, within only an infinite `bound`, checked first.
pub(crate) fn tri_tri_within(a: [P3; 3], b: [P3; 3], bound: f64) -> bool {
    if bound == f64::INFINITY {
        return true;
    }
    let near = |pa: P3, pb: P3| {
        let d = sub3(pa, pb);
        dot3(d, d).sqrt() <= bound
    };
    a.iter().any(|&v| near(v, closest_on_triangle(v, b).point))
        || b.iter().any(|&v| near(closest_on_triangle(v, a).point, v))
        || (0..3).any(|i| {
            (0..3).any(|j| {
                let (pa, pb) = seg_seg_closest(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3]);
                near(pa, pb)
            })
        })
        || tri_tri_intersect(a, b)
}

fn sub3(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Closest points of the segments `p1..q1` and `p2..q2` (Ericson, Real-Time
/// Collision Detection, 5.1.9). Parallel segments, where the closest pair is
/// not unique, get the pair at `s = 0` on the first segment, reclamped if the
/// foot on the second falls outside it; either way the distance is right.
fn seg_seg_closest(p1: P3, q1: P3, p2: P3, q2: P3) -> (P3, P3) {
    let d1 = sub3(q1, p1);
    let d2 = sub3(q2, p2);
    let r = sub3(p1, p2);
    let a = dot3(d1, d1);
    let e = dot3(d2, d2);
    let f = dot3(d2, r);
    let at = |p: P3, d: P3, t: f64| -> P3 { std::array::from_fn(|k| p[k] + t * d[k]) };
    // A division that is `0` instead of non-finite, clamped to `[0, 1]`.
    let unit = |num: f64, den: f64| {
        let t = num / den;
        if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    let (s, t);
    if a <= 0.0 && e <= 0.0 {
        (s, t) = (0.0, 0.0);
    } else if a <= 0.0 {
        (s, t) = (0.0, unit(f, e));
    } else {
        let c = dot3(d1, r);
        if e <= 0.0 {
            (s, t) = (unit(-c, a), 0.0);
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
                (s, t) = (unit(-c, a), 0.0);
            } else if t0 > 1.0 {
                (s, t) = (unit(b - c, a), 1.0);
            } else {
                (s, t) = (s0, t0);
            }
        }
    }
    (at(p1, d1, s), at(p2, d2, t))
}

fn same_strict_side(d: [i8; 3]) -> bool {
    (d[0] > 0 && d[1] > 0 && d[2] > 0) || (d[0] < 0 && d[1] < 0 && d[2] < 0)
}

/// The corner alone on its side of the other triangle's plane, given the
/// corners' signs `d` (not all zero, not all strictly on one side), and
/// whether that corner counts as being on the positive side. Its two
/// neighbours lie on the other closed side.
fn lone_corner(d: [i8; 3]) -> (usize, bool) {
    for i in 0..3 {
        let (j, k) = (d[(i + 1) % 3], d[(i + 2) % 3]);
        if d[i] > 0 && j <= 0 && k <= 0 {
            return (i, true);
        }
        if d[i] < 0 && j >= 0 && k >= 0 {
            return (i, false);
        }
    }
    // Every remaining case has one zero and the other two strictly on the
    // same side; the zero corner then counts as being on the opposite side.
    let i = (0..3).find(|&i| d[i] == 0).unwrap_or(0);
    (i, d[(i + 1) % 3] < 0)
}

/// Guigue and Devillers (2003), for two non-degenerate triangles that are not
/// coplanar and that both straddle or touch the other's plane. `da` and `db`
/// are the corner signs from [`tri_tri_intersect`].
fn guigue_devillers(a: [P3; 3], da: [i8; 3], b: [P3; 3], db: [i8; 3]) -> bool {
    // Rotate a so its lone corner comes first; if that corner is on the
    // negative side of b, swap b's last two corners (flipping b's normal) so
    // that it is on the positive side.
    let (i, positive) = lone_corner(da);
    let [p1, mut q1, mut r1] = [a[i], a[(i + 1) % 3], a[(i + 2) % 3]];
    let (mut b, mut db) = (b, db);
    if !positive {
        b.swap(1, 2);
        db.swap(1, 2);
    }
    // The same for b against a.
    let (j, positive) = lone_corner(db);
    let [p2, q2, r2] = [b[j], b[(j + 1) % 3], b[(j + 2) % 3]];
    if !positive {
        std::mem::swap(&mut q1, &mut r1);
    }
    // The segments where each triangle crosses the line of the two planes
    // overlap exactly when these two orientations allow it.
    side(q1, p2, p1, q2) <= 0 && side(p1, p2, r1, r2) <= 0
}

/// Two coplanar non-degenerate triangles, projected by dropping `axis`.
fn coplanar(a: [P3; 3], b: [P3; 3], axis: usize) -> bool {
    let a = a.map(|p| drop(p, axis));
    let b = b.map(|p| drop(p, axis));
    // If no edges meet, the triangles are either disjoint or one holds the
    // other entirely, so one corner of each settles it.
    (0..3).any(|i| (0..3).any(|j| seg_seg_2d(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3])))
        || in_tri_2d(a[0], &b)
        || in_tri_2d(b[0], &a)
}

/// True when the closed segment `pq` (possibly a single point) meets the closed
/// triangle `t`. `axis` is `normal_axis(t)`.
fn seg_tri(p: P3, q: P3, t: &[P3; 3], axis: Option<usize>) -> bool {
    let Some(axis) = axis else {
        // A degenerate triangle is the union of its edges.
        return (0..3).any(|i| seg_seg_3d(p, q, t[i], t[(i + 1) % 3]));
    };
    let sp = side(t[0], t[1], t[2], p);
    let sq = side(t[0], t[1], t[2], q);
    if sp * sq > 0 {
        return false;
    }
    if sp == 0 && sq == 0 {
        let t2 = t.map(|v| drop(v, axis));
        let (p2, q2) = (drop(p, axis), drop(q, axis));
        return in_tri_2d(p2, &t2) || (0..3).any(|i| seg_seg_2d(p2, q2, t2[i], t2[(i + 1) % 3]));
    }
    // pq meets the plane at one point. It lies in t exactly when the line
    // through p and q passes every edge of t on the same side (or through it).
    let s = [0, 1, 2].map(|i| side(p, q, t[i], t[(i + 1) % 3]));
    !(s.contains(&1) && s.contains(&-1))
}

/// True when the closed segments `pq` and `rs` in 3D (either possibly a point)
/// share a point.
fn seg_seg_3d(p: P3, q: P3, r: P3, s: P3) -> bool {
    if side(p, q, r, s) != 0 {
        return false;
    }
    // Coplanar: find a plane through three of the points that are not
    // collinear and project along its normal.
    let axis = [[p, q, r], [p, q, s], [r, s, p], [r, s, q]]
        .iter()
        .find_map(normal_axis);
    match axis {
        Some(k) => seg_seg_2d(drop(p, k), drop(q, k), drop(r, k), drop(s, k)),
        // All four points on one line: the segments meet when an endpoint of
        // one lies within the other.
        None => between(p, q, r) || between(p, q, s) || between(r, s, p) || between(r, s, q),
    }
}

/// For `x` collinear with `a` and `b`: whether it lies on the closed segment.
fn between<const N: usize>(a: [f64; N], b: [f64; N], x: [f64; N]) -> bool {
    (0..N).all(|k| a[k].min(b[k]) <= x[k] && x[k] <= a[k].max(b[k]))
}

/// True when the closed 2D segments `pq` and `rs` (either possibly a point)
/// share a point.
fn seg_seg_2d(p: P2, q: P2, r: P2, s: P2) -> bool {
    let (d1, d2) = (side2(p, q, r), side2(p, q, s));
    let (d3, d4) = (side2(r, s, p), side2(r, s, q));
    (d1 * d2 < 0 && d3 * d4 < 0)
        || (d1 == 0 && between(p, q, r))
        || (d2 == 0 && between(p, q, s))
        || (d3 == 0 && between(r, s, p))
        || (d4 == 0 && between(r, s, q))
}

/// True when `x` lies in the closed 2D triangle `t`, which must not be
/// degenerate.
fn in_tri_2d(x: P2, t: &[P2; 3]) -> bool {
    let s = [0, 1, 2].map(|i| side2(t[i], t[(i + 1) % 3], x));
    !(s.contains(&1) && s.contains(&-1))
}

/// Exact orientation signs for any finite coordinates, slow. Used only when a
/// coordinate is outside the range where `robust` is exact.
mod exact {
    use super::{P2, P3};

    /// Bits in the accumulator. A product of three `f64`s is an integer of at
    /// most 159 bits times `2^e` with `-3222 <= e <= 2913`; 24 such terms
    /// with a sign bit fit in `6135 + 159 + 5 + 1` bits.
    const LIMBS: usize = 100;
    /// Bit 0 of the accumulator has weight `2^-BASE`.
    const BASE: i32 = 3 * 1074;

    /// The integer significand and exponent of `x`: `x = m * 2^e`.
    fn decode(x: f64) -> (u64, i32) {
        let bits = x.to_bits();
        let exp = ((bits >> 52) & 0x7ff) as i32;
        let frac = bits & ((1 << 52) - 1);
        if exp == 0 {
            (frac, -1074)
        } else {
            (frac | (1 << 52), exp - 1075)
        }
    }

    /// A two's complement integer of `64 * LIMBS` bits.
    struct Acc([u64; LIMBS]);

    impl Acc {
        /// Adds `negate ? -x : x` for `x = f[0] * f[1] * f[2]`.
        fn add_product(&mut self, f: [f64; 3], negate: bool) {
            let d = f.map(decode);
            let negate = negate ^ (f.iter().filter(|x| x.is_sign_negative()).count() % 2 == 1);
            // The significand product, up to 159 bits, in three limbs.
            let p = u128::from(d[0].0) * u128::from(d[1].0);
            let lo = (p as u64 as u128) * u128::from(d[2].0);
            let hi = (p >> 64) * u128::from(d[2].0);
            let l0 = lo as u64;
            let mid = (lo >> 64) + (hi as u64 as u128);
            let l1 = mid as u64;
            let l2 = ((mid >> 64) + (hi >> 64)) as u64;
            if l0 | l1 | l2 == 0 {
                return;
            }
            let shift = (d[0].1 + d[1].1 + d[2].1 + BASE) as usize;
            let (at, bit) = (shift / 64, shift % 64);
            let mut term = [0u64; 4];
            for (i, l) in [l0, l1, l2].into_iter().enumerate() {
                let wide = u128::from(l) << bit;
                term[i] |= wide as u64;
                term[i + 1] |= (wide >> 64) as u64;
            }
            let mut carry = false;
            for i in at..LIMBS {
                let t = term.get(i - at).copied().unwrap_or(0);
                if t == 0 && !carry && i >= at + 4 {
                    break;
                }
                let (r, c) = if negate {
                    let (r1, c1) = self.0[i].overflowing_sub(t);
                    let (r2, c2) = r1.overflowing_sub(u64::from(carry));
                    (r2, c1 || c2)
                } else {
                    let (r1, c1) = self.0[i].overflowing_add(t);
                    let (r2, c2) = r1.overflowing_add(u64::from(carry));
                    (r2, c1 || c2)
                };
                self.0[i] = r;
                carry = c;
            }
        }

        fn sign(&self) -> i8 {
            if self.0[LIMBS - 1] >> 63 == 1 {
                -1
            } else if self.0.iter().any(|&l| l != 0) {
                1
            } else {
                0
            }
        }
    }

    /// The sign of the determinant of `n` rows `(1, p_i)`, where each `p_i`
    /// has `n - 1` coordinates. That is `det[p_1 - p_0, ..., p_{n-1} - p_0]`.
    fn det_with_ones<const N: usize>(rows: [[f64; N]; N]) -> i8 {
        let mut acc = Acc([0; LIMBS]);
        // Every permutation of the columns, by Heap's algorithm.
        let mut perm: [usize; N] = std::array::from_fn(|i| i);
        let mut odd = false;
        let mut c = [0usize; N];
        let mut term = |perm: &[usize; N], odd: bool| {
            let mut f = [1.0; 3];
            let mut n = 0;
            for (i, &j) in perm.iter().enumerate() {
                if j != 0 {
                    f[n] = rows[i][j];
                    n += 1;
                }
            }
            acc.add_product(f, odd);
        };
        term(&perm, odd);
        let mut i = 1;
        while i < N {
            if c[i] < i {
                if i % 2 == 0 {
                    perm.swap(0, i);
                } else {
                    perm.swap(c[i], i);
                }
                odd = !odd;
                term(&perm, odd);
                c[i] += 1;
                i = 1;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        acc.sign()
    }

    /// The sign of `det[b - a, c - a, d - a]`, exactly.
    pub(super) fn side(a: P3, b: P3, c: P3, d: P3) -> i8 {
        det_with_ones([a, b, c, d].map(|p| [1.0, p[0], p[1], p[2]]))
    }

    /// The sign of `det[b - a, c - a]`, exactly.
    pub(super) fn side2(a: P2, b: P2, c: P2) -> i8 {
        det_with_ones([a, b, c].map(|p| [1.0, p[0], p[1]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::Lcg;

    const A: [P3; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

    fn both(a: [P3; 3], b: [P3; 3]) -> bool {
        let ab = tri_tri_intersect(a, b);
        assert_eq!(ab, tri_tri_intersect(b, a), "asymmetric for {a:?} {b:?}");
        ab
    }

    #[test]
    fn orient_sign_convention() {
        let o = [0.0; 3];
        assert!(side(o, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) > 0);
        assert!(side(o, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]) < 0);
        assert_eq!(
            side(o, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 7.0, 0.0]),
            0
        );
    }

    #[test]
    fn crossing_and_separated() {
        let b = [[0.2, 0.2, -1.0], [0.2, 0.2, 1.0], [0.8, 0.1, 0.0]];
        assert!(both(A, b));
        let shifted = b.map(|p| [p[0] + 5.0, p[1], p[2]]);
        assert!(!both(A, shifted));
    }

    #[test]
    fn touching_cases() {
        // A vertex of B on A's interior, the rest of B above.
        assert!(both(
            A,
            [[0.25, 0.25, 0.0], [0.25, 0.25, 1.0], [0.5, 0.25, 1.0]]
        ));
        // An edge of B overlapping an edge of A, B in the plane y = 0.
        assert!(both(A, [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [0.5, 0.0, 1.0]]));
        // A vertex of B on a vertex of A.
        assert!(both(A, [[1.0, 0.0, 0.0], [2.0, 0.0, 1.0], [2.0, 1.0, 1.0]]));
        // A vertex of B on an edge of A.
        assert!(both(
            A,
            [[0.5, 0.0, 0.0], [0.5, -1.0, 1.0], [0.5, 1.0, 1.0]]
        ));
    }

    #[test]
    fn one_ulp_apart_is_separate() {
        for z in [f64::from(f32::MIN_POSITIVE), f64::from_bits(1)] {
            let b = [[0.25, 0.25, z], [0.25, 0.25, 1.0], [0.5, 0.25, 1.0]];
            assert!(!both(A, b), "z = {z:e}");
            let below = b.map(|p| [p[0], p[1], -p[2]]);
            assert!(!both(A, below), "z = -{z:e}");
        }
    }

    #[test]
    fn coplanar_cases() {
        // Overlapping.
        assert!(both(A, [[0.2, 0.2, 0.0], [1.5, 0.2, 0.0], [0.2, 1.5, 0.0]]));
        // Sharing only an edge.
        assert!(both(A, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]));
        // Sharing only a vertex.
        assert!(both(
            A,
            [[1.0, 0.0, 0.0], [2.0, -1.0, 0.0], [2.0, 1.0, 0.0]]
        ));
        // Disjoint, a gap of 1e-9 across A's long edge.
        let g = 1e-9;
        assert!(!both(
            A,
            [[1.0 + g, g, 0.0], [g, 1.0 + g, 0.0], [1.0, 1.0, 0.0]]
        ));
        // B containing A, and the reverse.
        assert!(both(
            A,
            [[-1.0, -1.0, 0.0], [3.0, -1.0, 0.0], [-1.0, 3.0, 0.0]]
        ));
        // Crossing like a star of David: no vertex inside the other.
        assert!(both(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.5, 0.0]],
            [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, -0.5, 0.0]]
        ));
        // Coplanar in a tilted plane.
        let tilt = |p: P3| [p[0], p[1], p[0] + 2.0 * p[1]];
        assert!(both(
            A.map(tilt),
            [[0.2, 0.2, 0.0], [1.5, 0.2, 0.0], [0.2, 1.5, 0.0]].map(tilt)
        ));
        assert!(!both(
            A.map(tilt),
            [[1.0, 0.5, 0.0], [2.0, 0.5, 0.0], [1.0, 1.5, 0.0]].map(tilt)
        ));
    }

    #[test]
    fn degenerate_cases() {
        // A segment piercing A.
        assert!(both(
            A,
            [[0.2, 0.2, -1.0], [0.2, 0.2, 1.0], [0.2, 0.2, 1.0]]
        ));
        // A segment beside A.
        assert!(!both(
            A,
            [[2.0, 0.2, -1.0], [2.0, 0.2, 1.0], [2.0, 0.2, 0.0]]
        ));
        // A segment lying in A's plane, crossing it.
        assert!(both(
            A,
            [[-1.0, 0.25, 0.0], [2.0, 0.25, 0.0], [0.5, 0.25, 0.0]]
        ));
        // A point on A, and off A.
        assert!(both(A, [[0.25, 0.25, 0.0]; 3]));
        assert!(both(A, [[1.0, 0.0, 0.0]; 3]));
        assert!(!both(A, [[0.25, 0.25, 1e-9]; 3]));
        assert!(!both(A, [[0.75, 0.75, 0.0]; 3]));
        // Both degenerate: crossing segments.
        let s = [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.5, 0.0]];
        assert!(both(
            s,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.75, 0.25, 0.0]]
        ));
        // Skew segments miss.
        assert!(!both(
            s,
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.75, 0.25, 1.0]]
        ));
        // Collinear segments: overlapping, touching at an end, and apart.
        assert!(both(s, [[0.5, 0.5, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]]));
        assert!(both(s, [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]]));
        assert!(!both(
            s,
            [[1.5, 1.5, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]]
        ));
        // Points.
        assert!(both([[1.0, 2.0, 3.0]; 3], [[1.0, 2.0, 3.0]; 3]));
        assert!(!both([[1.0, 2.0, 3.0]; 3], [[1.0, 2.0, 3.5]; 3]));
        assert!(both(s, [[0.25, 0.25, 0.0]; 3]));
        assert!(!both(s, [[0.25, 0.25, 1.0]; 3]));
    }

    /// Reference: two closed triangles meet exactly when an edge of one meets
    /// the other.
    fn by_edges(a: [P3; 3], b: [P3; 3]) -> bool {
        let (na, nb) = (normal_axis(&a), normal_axis(&b));
        (0..3).any(|i| seg_tri(a[i], a[(i + 1) % 3], &b, nb))
            || (0..3).any(|i| seg_tri(b[i], b[(i + 1) % 3], &a, na))
    }

    #[test]
    fn small_integer_grid_matches_edge_reference() {
        // Coordinates in {0, 1, 2} hit every coplanar, collinear and touching
        // configuration many times over.
        let mut rng = Lcg(7);
        let mut hits = 0;
        for _ in 0..20_000 {
            let mut pick = || std::array::from_fn(|_| (rng.next() * 3.0).floor() as f64);
            let a: [P3; 3] = std::array::from_fn(|_| pick());
            let b: [P3; 3] = std::array::from_fn(|_| pick());
            let got = both(a, b);
            assert_eq!(got, by_edges(a, b), "{a:?} {b:?}");
            hits += usize::from(got);
        }
        assert!(hits > 1000 && hits < 19_000, "{hits}");
    }

    #[test]
    fn exact_fallback_matches_robust() {
        assert!(safe(0.0) && safe(-0.0) && safe(2f64.powi(-200)) && safe(-2f64.powi(200)));
        assert!(!safe(2f64.powi(-201)) && !safe(2f64.powi(201)) && !safe(f64::from_bits(1)));
        let robust3 = |a: P3, b: P3, c: P3, d: P3| {
            let c3 = |p: P3| Coord3D {
                x: p[0],
                y: p[1],
                z: p[2],
            };
            sign(-orient3d(c3(a), c3(b), c3(c), c3(d)))
        };
        let mut rng = Lcg(99);
        let mut zeros = 0;
        for i in 0..5000 {
            // Half small integers (many exact zeros), half random reals.
            let mut pick = || -> P3 {
                std::array::from_fn(|_| {
                    let r = f64::from(rng.next());
                    if i % 2 == 0 {
                        (r * 3.0).floor() - 1.0
                    } else {
                        r * 2.0 - 1.0
                    }
                })
            };
            let [a, b, c, d] = [pick(), pick(), pick(), pick()];
            let s = robust3(a, b, c, d);
            assert_eq!(exact::side(a, b, c, d), s, "{a:?} {b:?} {c:?} {d:?}");
            let p2 = |p: P3| [p[0], p[1]];
            assert_eq!(
                exact::side2(p2(a), p2(b), p2(c)),
                sign(orient2d_approx(p2(a), p2(b), p2(c)))
            );
            zeros += usize::from(s == 0);
        }
        assert!(zeros > 100, "{zeros}");
    }

    #[test]
    fn extreme_scales_stay_exact() {
        // Scaling by a power of two is exact and cannot change the answer, but
        // takes the coordinates far outside where `robust` alone is exact:
        // down to 2^-1074 and up to 2^1001.
        let mut rng = Lcg(5);
        for _ in 0..3000 {
            let mut pick = || std::array::from_fn(|_| (rng.next() * 3.0).floor() as f64);
            let a: [P3; 3] = std::array::from_fn(|_| pick());
            let b: [P3; 3] = std::array::from_fn(|_| pick());
            let want = tri_tri_intersect(a, b);
            // Built from bits: `powi(-1074)` may compute 1 / 2^1074 = 1 / inf = 0
            // on some targets.
            let pow2 = |e: i64| f64::from_bits(((1023 + e) as u64) << 52);
            for scale in [f64::from_bits(1), pow2(-600), pow2(1000)] {
                let s = |t: [P3; 3]| t.map(|p| p.map(|v| v * scale));
                assert_eq!(both(s(a), s(b)), want, "{a:?} {b:?} x {scale:e}");
            }
        }
        // Mixed scales: a tiny offset from a unit triangle.
        let tiny = f64::from_bits(1);
        let b = [[0.25, 0.25, tiny], [0.25, 0.25, 1.0], [0.5, 0.25, 1.0]];
        assert!(!both(A, b));
        let b = [[0.25, 0.25, -tiny], [0.25, 0.25, 1.0], [0.5, 0.25, 1.0]];
        assert!(both(A, b));
    }

    fn vsub(a: P3, b: P3) -> P3 {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn vdot(a: P3, b: P3) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn at(a: P3, d: P3, s: f64) -> P3 {
        [a[0] + s * d[0], a[1] + s * d[1], a[2] + s * d[2]]
    }

    /// Squared distance from `x` to the segment `a..b`: the minimum of a
    /// quadratic in one variable over `[0, 1]`.
    fn oracle_pt_seg(x: P3, a: P3, b: P3) -> f64 {
        let d = vsub(b, a);
        let s = if vdot(d, d) > 0.0 {
            (vdot(vsub(x, a), d) / vdot(d, d)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let r = vsub(x, at(a, d, s));
        vdot(r, r)
    }

    /// Squared distance from `x` to the triangle `t`: the minimum of a
    /// quadratic in `(u, v)` over `u, v >= 0, u + v <= 1`. Solves the normal
    /// equations, and if the solution is outside, the minimum is on an edge.
    fn oracle_pt_tri(x: P3, t: [P3; 3]) -> f64 {
        let (e1, e2) = (vsub(t[1], t[0]), vsub(t[2], t[0]));
        let w = vsub(x, t[0]);
        let (a, b, c) = (vdot(e1, e1), vdot(e1, e2), vdot(e2, e2));
        let det = a * c - b * b;
        let edges = (0..3)
            .map(|i| oracle_pt_seg(x, t[i], t[(i + 1) % 3]))
            .fold(f64::INFINITY, f64::min);
        if det <= 1e-12 {
            return edges;
        }
        let (r1, r2) = (vdot(w, e1), vdot(w, e2));
        let u = (c * r1 - b * r2) / det;
        let v = (a * r2 - b * r1) / det;
        if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
            let p = at(at(t[0], e1, u), e2, v);
            let r = vsub(x, p);
            return vdot(r, r).min(edges);
        }
        edges
    }

    /// Squared distance between the segments `p..q` and `r..s`: minimizes
    /// `|p + a d1 - r - b d2|^2` over `[0, 1]^2` by solving for the stationary
    /// point, and otherwise on the four sides of the square.
    fn oracle_seg_seg(p: P3, q: P3, r: P3, s: P3) -> f64 {
        let (d1, d2) = (vsub(q, p), vsub(s, r));
        let w = vsub(p, r);
        let (a, b, c) = (vdot(d1, d1), vdot(d1, d2), vdot(d2, d2));
        let (e, f) = (vdot(d1, w), vdot(d2, w));
        let sides = oracle_pt_seg(p, r, s)
            .min(oracle_pt_seg(q, r, s))
            .min(oracle_pt_seg(r, p, q))
            .min(oracle_pt_seg(s, p, q));
        let det = a * c - b * b;
        if det <= 1e-12 {
            return sides;
        }
        let u = (b * f - c * e) / det;
        let v = (a * f - b * e) / det;
        if (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v) {
            let g = vsub(at(p, d1, u), at(r, d2, v));
            return vdot(g, g).min(sides);
        }
        sides
    }

    /// The true squared distance between two disjoint triangles.
    fn oracle_dist2(a: [P3; 3], b: [P3; 3]) -> f64 {
        let mut best = f64::INFINITY;
        for i in 0..3 {
            best = best.min(oracle_pt_tri(a[i], b)).min(oracle_pt_tri(b[i], a));
            for j in 0..3 {
                best = best.min(oracle_seg_seg(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3]));
            }
        }
        best
    }

    /// Random triangle pairs, with B pushed to `x` in `[1.5, 3]`: always apart.
    fn separated_pairs(n: usize, seed: u64) -> Vec<([P3; 3], [P3; 3])> {
        let mut rng = Lcg(seed);
        (0..n)
            .map(|_| {
                let mut pt = |dx: f64| -> P3 {
                    [
                        f64::from(rng.next()) + dx,
                        f64::from(rng.next()),
                        f64::from(rng.next()),
                    ]
                };
                let a: [P3; 3] = std::array::from_fn(|_| pt(0.0));
                let b: [P3; 3] = std::array::from_fn(|_| pt(1.5));
                (a, b)
            })
            .collect()
    }

    /// Points of a triangle on a barycentric grid of `n + 1` points per side.
    fn grid(t: [P3; 3], n: usize) -> Vec<P3> {
        let mut out = Vec::new();
        for i in 0..=n {
            for j in 0..=(n - i) {
                let (u, v) = (i as f64 / n as f64, j as f64 / n as f64);
                out.push(std::array::from_fn(|k| {
                    (1.0 - u - v) * t[0][k] + u * t[1][k] + v * t[2][k]
                }));
            }
        }
        out
    }

    #[test]
    fn distance_matches_dense_sampling() {
        for (a, b) in separated_pairs(500, 0xd157) {
            let got = tri_tri_distance(a, b).dist2.sqrt();
            let (ga, gb) = (grid(a, 19), grid(b, 19));
            let mut dense = f64::INFINITY;
            for &p in &ga {
                for &q in &gb {
                    let d = vsub(p, q);
                    dense = dense.min(vdot(d, d));
                }
            }
            let truth = oracle_dist2(a, b).sqrt();
            assert!(got <= dense.sqrt() + 1e-9, "{a:?} {b:?}: {got} > {dense}");
            assert!(got >= truth - 1e-9, "{a:?} {b:?}: {got} < {truth}");
            // Two-sided: every candidate is a real pair, so only this bound
            // catches a non-minimal answer.
            assert!(got <= truth + 1e-9, "{a:?} {b:?}: {got} > {truth}");
        }
    }

    #[test]
    fn piercing_pairs_have_zero_distance() {
        let mut rng = Lcg(0x91e2c);
        let mut r = || f64::from(rng.next());
        for _ in 0..300 {
            let a: [P3; 3] = std::array::from_fn(|_| std::array::from_fn(|_| r()));
            let (e1, e2) = (vsub(a[1], a[0]), vsub(a[2], a[0]));
            let n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            // A point well inside A, and B with an edge through it.
            let (u, v) = (0.1 + 0.3 * r(), 0.1 + 0.3 * r());
            let x = at(at(a[0], e1, u), e2, v);
            let (s1, s2) = (0.2 + r(), 0.2 + r());
            let b = [at(x, n, -s1), at(x, n, s2), std::array::from_fn(|_| r())];
            let d = tri_tri_distance(a, b);
            assert_eq!(d.dist2, 0.0, "{a:?} {b:?}");
            assert_eq!(d.pa, d.pb);
            assert!(on_triangle(d.pa, a, 1e-9), "{a:?} {b:?}: {:?}", d.pa);
            assert!(on_triangle(d.pa, b, 1e-9), "{a:?} {b:?}: {:?}", d.pa);
        }
    }

    #[test]
    fn seg_seg_closest_cases() {
        let z = [0.0; 3];
        let x = |v: f64, y: f64| [v, y, 0.0];
        // Parallel and overlapping (any closest pair; this one is fixed).
        let (pa, pb) = seg_seg_closest(z, x(2.0, 0.0), x(1.0, 1.0), x(3.0, 1.0));
        assert_eq!((pa, pb), (x(1.0, 0.0), x(1.0, 1.0)));
        // Parallel and disjoint: end to end.
        let (pa, pb) = seg_seg_closest(z, x(1.0, 0.0), x(3.0, 1.0), x(4.0, 1.0));
        assert_eq!((pa, pb), (x(1.0, 0.0), x(3.0, 1.0)));
        // Antiparallel and disjoint.
        let (pa, pb) = seg_seg_closest(z, x(1.0, 0.0), x(4.0, 1.0), x(3.0, 1.0));
        assert_eq!((pa, pb), (x(1.0, 0.0), x(3.0, 1.0)));
        // A point against a segment, in both roles.
        let (pa, pb) = seg_seg_closest(x(1.0, 1.0), x(1.0, 1.0), z, x(2.0, 0.0));
        assert_eq!((pa, pb), (x(1.0, 1.0), x(1.0, 0.0)));
        let (pa, pb) = seg_seg_closest(z, x(2.0, 0.0), x(1.0, 1.0), x(1.0, 1.0));
        assert_eq!((pa, pb), (x(1.0, 0.0), x(1.0, 1.0)));
        // Two points.
        let (p, q) = ([1.0, 2.0, 3.0], [4.0, 6.0, 3.0]);
        assert_eq!(seg_seg_closest(p, p, q, q), (p, q));
        // Skew segments crossing in projection.
        let (pa, pb) = seg_seg_closest(z, x(2.0, 0.0), [1.0, -1.0, 1.0], [1.0, 1.0, 1.0]);
        assert_eq!((pa, pb), (x(1.0, 0.0), [1.0, 0.0, 1.0]));
    }

    #[test]
    fn parallel_edges() {
        // A's hypotenuse (1,0,0)-(0,1,0) and B's edge, parallel and 0.5 away
        // in the plane's normal direction; no other feature is closer.
        let b = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [1.0, 1.0, 1.5]];
        let d = tri_tri_distance(A, b);
        assert!((d.dist2 - 0.25).abs() < 1e-12, "{}", d.dist2);
        assert!((oracle_dist2(A, b) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn intersecting_is_zero() {
        let b = [[0.2, 0.2, -1.0], [0.2, 0.2, 1.0], [0.8, 0.1, 0.0]];
        let d = tri_tri_distance(A, b);
        assert_eq!(d.dist2, 0.0);
        assert_eq!(d.pa, d.pb);
        assert!(on_triangle(d.pa, A, 1e-12), "{:?}", d.pa);
        assert!(on_triangle(d.pa, b, 1e-12), "{:?}", d.pa);
        // Touching at one vertex.
        let c = [[1.0, 0.0, 0.0], [2.0, 0.0, 1.0], [2.0, 1.0, 1.0]];
        let d = tri_tri_distance(A, c);
        assert_eq!(d.dist2, 0.0);
        assert!(on_triangle(d.pa, A, 1e-12) && on_triangle(d.pa, c, 1e-12));
        assert_eq!(d.pa, d.pb);
    }

    /// Whether `x` is within `tol` of the plane of `t` and inside it, by
    /// barycentric coordinates.
    fn on_triangle(x: P3, t: [P3; 3], tol: f64) -> bool {
        let (e1, e2) = (vsub(t[1], t[0]), vsub(t[2], t[0]));
        let w = vsub(x, t[0]);
        let (a, b, c) = (vdot(e1, e1), vdot(e1, e2), vdot(e2, e2));
        let det = a * c - b * b;
        let (r1, r2) = (vdot(w, e1), vdot(w, e2));
        let u = (c * r1 - b * r2) / det;
        let v = (a * r2 - b * r1) / det;
        let p = at(at(t[0], e1, u), e2, v);
        let off = vsub(x, p);
        vdot(off, off).sqrt() <= tol && u >= -tol && v >= -tol && u + v <= 1.0 + tol
    }

    #[test]
    fn points_are_on_triangles() {
        for (a, b) in separated_pairs(500, 0x9017) {
            let d = tri_tri_distance(a, b);
            let g = vsub(d.pa, d.pb);
            assert!((vdot(g, g) - d.dist2).abs() < 1e-9);
            assert!(on_triangle(d.pa, a, 1e-9), "{a:?}: {:?}", d.pa);
            assert!(on_triangle(d.pb, b, 1e-9), "{b:?}: {:?}", d.pb);
        }
    }

    #[test]
    fn within_matches_distance() {
        // Random pairs, half of them overlapping (many intersect), half apart,
        // against bounds at, just around and far from each pair's distance.
        let mut rng = Lcg(0x3171);
        let (mut yes, mut hits) = (0, 0);
        for i in 0..20_000 {
            let mut r = || f64::from(rng.next());
            let dx = if i % 2 == 0 { 0.0 } else { 1.0 + r() };
            let mut a: [P3; 3] = std::array::from_fn(|_| [r(), r(), r()]);
            let mut b: [P3; 3] = std::array::from_fn(|_| [r() + dx, r(), r()]);
            if i % 4 == 3 {
                // Small integers: touching, coplanar and degenerate pairs.
                let mut g = || std::array::from_fn(|_| (r() * 3.0).floor());
                (a, b) = (std::array::from_fn(|_| g()), std::array::from_fn(|_| g()));
            }
            let d = tri_tri_distance(a, b).dist2.sqrt();
            hits += usize::from(d == 0.0);
            for bound in [
                0.0,
                d,
                d.next_up(),
                d.next_down().max(0.0),
                0.5 * d,
                2.0 * d,
                0.25,
                f64::INFINITY,
            ] {
                let got = tri_tri_within(a, b, bound);
                assert_eq!(got, d <= bound, "{a:?} {b:?} {bound}");
                yes += usize::from(got);
            }
        }
        assert!(hits > 1000 && yes > 50_000, "{hits} {yes}");
    }

    #[test]
    fn symmetric() {
        let mut rng = Lcg(2026);
        let mut hits = 0;
        for i in 0..2000 {
            let mut pick = || std::array::from_fn(|_| f64::from(rng.next()) * 2.0 - 1.0);
            let a: [P3; 3] = std::array::from_fn(|_| pick());
            let mut b: [P3; 3] = std::array::from_fn(|_| pick());
            if i % 2 == 0 {
                // Snap a vertex of B onto A's plane, as nearly as f64 allows.
                let e1: P3 = std::array::from_fn(|k| a[1][k] - a[0][k]);
                let e2: P3 = std::array::from_fn(|k| a[2][k] - a[0][k]);
                let n = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                let nn: f64 = n.iter().map(|v| v * v).sum();
                let d: f64 = (0..3).map(|k| (b[0][k] - a[0][k]) * n[k]).sum::<f64>() / nn;
                b[0] = std::array::from_fn(|k| b[0][k] - d * n[k]);
            }
            let got = both(a, b);
            assert_eq!(got, by_edges(a, b), "{a:?} {b:?}");
            hits += usize::from(got);
        }
        assert!(hits > 100, "{hits}");
    }
}
