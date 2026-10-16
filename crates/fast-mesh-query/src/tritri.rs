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
#[allow(dead_code)] // Used by the mesh-vs-mesh queries, added later in Phase 3b.
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
            for scale in [2f64.powi(-1074), 2f64.powi(-600), 2f64.powi(1000)] {
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
