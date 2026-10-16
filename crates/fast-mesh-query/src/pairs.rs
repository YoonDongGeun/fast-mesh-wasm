//! Mesh-vs-mesh queries: intersection, distance and proximity between two
//! meshes placed relative to each other by a [`Rigid`] transform.
//!
//! Every query walks both BVHs at once ([`traverse_pairs_with`]) and prunes
//! through the traversal's `key` as well as its `visit`, so that the
//! stackless overflow fallback, which shows `visit` only leaf pairs, prunes
//! too.

use std::cell::Cell;

use fast_mesh_bvh::{Aabb, Bvh, NodeKind, NodeView};
use fast_mesh_store::{Mesh, TriangleId};

use crate::pair::{PAIR_STACK, Step, traverse_pairs_with};
use crate::ray::points;
use crate::tritri::{P3, TriTriDist, tri_tri_distance, tri_tri_intersect};
use crate::{Appender, QueryError, Result, Rigid, check};

/// The closest pair of points between two meshes, found by [`mesh_distance`].
///
/// Both points are in mesh A's local space: `point_b` is the point of B's
/// triangle `triangle_b` as placed by the query's [`Rigid`] transform, not a
/// point in B's own coordinates. Map it back with [`Rigid::inverse`] if needed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshDistance {
    /// The distance between the two surfaces. 0 when they touch, intersect,
    /// or are closer than `f64` rounding resolves: a positive gap whose
    /// computed distance rounds to 0 is reported as 0. See [`mesh_distance`].
    pub distance: f32,
    /// The closest point on A's triangle `triangle_a`, in A's local space.
    pub point_a: [f32; 3],
    /// The closest point on B's triangle `triangle_b`, in A's local space
    /// (not B's).
    pub point_b: [f32; 3],
    /// The triangle of A that `point_a` lies on.
    pub triangle_a: TriangleId,
    /// The triangle of B that `point_b` lies on.
    pub triangle_b: TriangleId,
}

/// Whether any alive triangle of A intersects or touches any alive triangle
/// of B, with B placed in A's space by `xf`.
///
/// The test is closed and exact for B's vertices as transformed by `xf` in
/// `f64`: touching at a single point counts, a gap of one `f64` ulp does not.
/// The search stops at the first intersecting pair found.
///
/// Passing the same mesh and BVH as both A and B means two coincident copies
/// of it, not one mesh tested against itself: with [`Rigid::IDENTITY`] every
/// triangle intersects its own copy, so the result is `true` for any mesh with
/// a triangle.
///
/// # Errors
/// [`QueryError::InvalidInput`] when `xf` is not a finite proper rotation plus
/// translation (see [`Rigid`]); then [`QueryError::MeshMismatch`] or
/// [`QueryError::StaleBvh`] when `bvh_a` or `bvh_b` does not match its current
/// mesh.
pub fn intersects(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
) -> Result<bool> {
    validate(mesh_a, bvh_a, mesh_b, bvh_b, xf, None)?;
    Ok(intersects_with::<PAIR_STACK>(
        mesh_a, bvh_a, mesh_b, bvh_b, xf,
    ))
}

/// The minimum distance between the surfaces of A and B, with B placed in A's
/// space by `xf`, and a closest pair of points. `None` when the distance is
/// greater than `max_distance`, or when either mesh has no alive triangle.
///
/// `max_distance` is inclusive and may be infinite; it is compared as
/// `f64 distance <= f64::from(max_distance)`. Positive distances are computed
/// in `f64` from B's vertices as transformed by `xf`, and are exact up to
/// rounding. Among pairs of triangles at the same minimum distance (equal
/// squared distance in `f64`), the smallest `(triangle_a, triangle_b)` wins.
///
/// A distance of 0 means the meshes touch, intersect, or are closer than `f64`
/// rounding resolves: a positive gap whose computed distance rounds to 0 is
/// reported as 0. The search stops at the first pair of triangles with
/// distance 0 that it finds. That pair intersects (decided exactly) or has a
/// computed distance of 0; it is not necessarily the smallest such pair, and
/// its two points coincide up to rounding. Use [`intersecting_pairs`] for
/// every intersecting pair.
///
/// Both points are in A's space; see [`MeshDistance`]. Passing the same mesh
/// twice means two coincident copies, as for [`intersects`].
///
/// # Errors
/// [`QueryError::InvalidInput`] for an invalid `xf` (see [`Rigid`]) or a
/// `max_distance` that is negative or NaN; then [`QueryError::MeshMismatch`] or
/// [`QueryError::StaleBvh`] as for [`intersects`].
pub fn mesh_distance(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    max_distance: f32,
) -> Result<Option<MeshDistance>> {
    validate(
        mesh_a,
        bvh_a,
        mesh_b,
        bvh_b,
        xf,
        Some((max_distance, "max_distance must not be negative or NaN")),
    )?;
    Ok(mesh_distance_with::<PAIR_STACK>(
        mesh_a,
        bvh_a,
        mesh_b,
        bvh_b,
        xf,
        max_distance,
    ))
}

/// Appends to `out_a` every alive triangle of A within `margin` of B, and to
/// `out_b` every alive triangle of B within `margin` of A, with B placed in
/// A's space by `xf`. Elements already in the vectors are kept and stay in
/// front; each appended part is in ascending order without duplicates.
///
/// A pair of triangles is near when it intersects or touches (decided
/// exactly), or when its `f64` distance is at most `margin`, compared as
/// `f64 distance <= f64::from(margin)`. So a `margin` of 0 finds the triangles
/// that touch the other mesh. Both sides come from the same pair tests, so they
/// agree pair for pair; a second call with the meshes swapped and
/// [`Rigid::inverse`] could differ at the rounding level.
///
/// Before deduplication, each vector temporarily holds a triangle once per
/// near pair it belongs to (a triangle near `k` triangles of the other mesh is
/// held `k` times), and keeps that capacity afterwards.
///
/// Passing the same mesh twice means two coincident copies, as for
/// [`intersects`].
///
/// # Errors
/// [`QueryError::InvalidInput`] for an invalid `xf` (see [`Rigid`]) or a
/// `margin` that is negative or NaN; then [`QueryError::MeshMismatch`] or
/// [`QueryError::StaleBvh`] as for [`intersects`]; and
/// [`QueryError::OutOfMemory`] when either vector cannot grow. On any error
/// both vectors keep their original lengths.
#[allow(clippy::too_many_arguments)] // The shared five, a bound and two outputs.
pub fn triangles_near(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    margin: f32,
    out_a: &mut Vec<TriangleId>,
    out_b: &mut Vec<TriangleId>,
) -> Result<()> {
    validate(
        mesh_a,
        bvh_a,
        mesh_b,
        bvh_b,
        xf,
        Some((margin, "margin must not be negative or NaN")),
    )?;
    triangles_near_with::<PAIR_STACK>(mesh_a, bvh_a, mesh_b, bvh_b, xf, margin, out_a, out_b)
}

/// Appends to `out` every pair `(a, b)` of an alive triangle `a` of A and an
/// alive triangle `b` of B that intersect or touch, with B placed in A's space
/// by `xf`, and returns how many were appended. The appended pairs are sorted
/// by `(a, b)`; elements already in `out` are kept and stay in front.
///
/// The test is exact, as for [`intersects`]. Passing the same mesh twice means
/// two coincident copies: with [`Rigid::IDENTITY`] every `(t, t)` is a pair,
/// along with every pair of neighbouring triangles.
///
/// # Errors
/// As for [`intersects`], and [`QueryError::OutOfMemory`] when `out` cannot
/// grow. On any error `out` keeps its original length.
pub fn intersecting_pairs(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    out: &mut Vec<(TriangleId, TriangleId)>,
) -> Result<usize> {
    validate(mesh_a, bvh_a, mesh_b, bvh_b, xf, None)?;
    intersecting_pairs_with::<PAIR_STACK>(mesh_a, bvh_a, mesh_b, bvh_b, xf, out)
}

/// Checks the inputs in the documented order: the transform, then the bound
/// (with its error message) if any, then each mesh against its BVH.
fn validate(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    bound: Option<(f32, &'static str)>,
) -> Result<()> {
    xf.validate()?;
    if let Some((v, msg)) = bound
        && (v.is_nan() || v < 0.0)
    {
        return Err(QueryError::InvalidInput(msg));
    }
    check(mesh_a, bvh_a)?;
    check(mesh_b, bvh_b)
}

/// The relative slack on pruning bounds; see [`beyond`].
const PRUNE_SLACK: f64 = 1.0 / (1u64 << 30) as f64;

/// The largest absolute coordinate of A's root box and of B's root box moved
/// into A's space: the magnitude that the rounding error of the triangle
/// distances scales with. 0 when either tree is empty (nothing is visited
/// then). It is infinite only if a moved box overflows `f32`, and then
/// [`beyond`] prunes nothing.
fn magnitude(bvh_a: &Bvh, bvh_b: &Bvh, xf: &Rigid) -> f64 {
    let (Some(ra), Some(rb)) = (bvh_a.root(), bvh_b.root()) else {
        return 0.0;
    };
    let boxes = [
        bvh_a.node(ra).bounds,
        xf.transform_box(&bvh_b.node(rb).bounds),
    ];
    boxes
        .iter()
        .flat_map(|b| b.min.iter().chain(&b.max))
        .map(|v| f64::from(v.abs()))
        .fold(0.0, f64::max)
}

/// Whether a pair of nodes whose boxes are `box_d` apart ([`box_distance`])
/// can be dropped under `bound`, for meshes of coordinate magnitude
/// `magnitude` ([`magnitude`]).
///
/// The box gap is a true lower bound on the distance between the boxes'
/// contents up to `f64` rounding of its own, and with an exact transform
/// (identity, 90° turns, translations that keep B's vertices on the `f32`
/// grid) it is essentially exact. The triangle distances, however, carry an
/// absolute error of a few `f64` ulps of the coordinates. So a pair whose
/// computed distance is within `bound` can sit in boxes slightly more than
/// `bound` apart. Pruning only beyond `bound + 2^-30 * (magnitude + bound)`
/// covers that error with a wide margin, slivers included. The slack only
/// adds work; which pairs are accepted is still decided by the computed
/// distance against `bound` itself.
fn beyond(box_d: f64, bound: f64, magnitude: f64) -> bool {
    box_d > bound + PRUNE_SLACK * (magnitude + bound)
}

/// The Euclidean distance between boxes `p` and `q` in `f64`, 0 when they
/// overlap or touch, and infinite when either is empty. A lower bound on the
/// distance between any point of `p` and any point of `q`, up to `f64`
/// rounding; queries prune on it through [`beyond`].
fn box_distance(p: &Aabb, q: &Aabb) -> f64 {
    if p.is_empty() || q.is_empty() {
        return f64::INFINITY;
    }
    (0..3)
        .map(|k| {
            let gap = (f64::from(p.min[k]) - f64::from(q.max[k]))
                .max(f64::from(q.min[k]) - f64::from(p.max[k]))
                .max(0.0);
            gap * gap
        })
        .sum::<f64>()
        .sqrt()
}

/// The pair key for the intersection queries: 0 when the boxes overlap (closed),
/// otherwise cull.
fn overlap_key(p: &Aabb, q: &Aabb) -> Option<f64> {
    p.overlaps(q).then_some(0.0)
}

/// The triangles of a leaf, or `None` for an inner node.
fn leaf<'a>(view: &NodeView<'a>) -> Option<&'a [TriangleId]> {
    match view.kind {
        NodeKind::Leaf { triangles } => Some(triangles),
        NodeKind::Inner { .. } => None,
    }
}

/// A triangle of A in `f64`.
fn tri_a(mesh: &Mesh, t: TriangleId) -> [P3; 3] {
    points(mesh, t).map(|p| p.map(f64::from))
}

/// A triangle of B moved into A's space with [`Rigid::apply`], the transform
/// [`Rigid::transform_box`] is conservative for.
fn tri_b(mesh: &Mesh, xf: &Rigid, t: TriangleId) -> [P3; 3] {
    points(mesh, t).map(|p| xf.apply(p))
}

#[cfg(test)]
thread_local! {
    /// Test-only: how many leaf pairs [`for_each_pair`] has tested on this thread.
    static LEAF_VISITS: Cell<usize> = const { Cell::new(0) };
}

/// Calls `pair` for every triangle pair of every leaf pair that the traversal
/// reaches with `key`, B's triangle already in A's space, until `pair` returns
/// `false`. `prune(k)` drops a popped pair with key `k`.
#[allow(clippy::too_many_arguments)] // The shared five and three callbacks.
fn for_each_pair<const N: usize>(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    key: impl FnMut(&Aabb, &Aabb) -> Option<f64>,
    prune: impl Fn(f64) -> bool,
    mut pair: impl FnMut(TriangleId, [P3; 3], TriangleId, [P3; 3]) -> bool,
) {
    traverse_pairs_with::<N>(bvh_a, bvh_b, xf, key, |va, vb, k| {
        if prune(k) {
            return Step::Skip;
        }
        let (Some(ta), Some(tb)) = (leaf(va), leaf(vb)) else {
            return Step::Descend;
        };
        #[cfg(test)]
        LEAF_VISITS.with(|c| c.set(c.get() + 1));
        for &b in tb {
            let pb = tri_b(mesh_b, xf, b);
            for &a in ta {
                if !pair(a, tri_a(mesh_a, a), b, pb) {
                    return Step::Stop;
                }
            }
        }
        Step::Skip
    });
}

fn intersects_with<const N: usize>(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
) -> bool {
    let mut found = false;
    for_each_pair::<N>(
        mesh_a,
        bvh_a,
        mesh_b,
        bvh_b,
        xf,
        overlap_key,
        |_| false,
        |_, pa, _, pb| {
            found = tri_tri_intersect(pa, pb);
            !found
        },
    );
    found
}

fn mesh_distance_with<const N: usize>(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    max_distance: f32,
) -> Option<MeshDistance> {
    // The distance of the best pair so far, or `max_distance` before one is
    // found. Pairs whose boxes are farther (with slack) are pruned, in `key`
    // and again in `visit`, as the bound shrinks after a pair is pushed.
    let bound = Cell::new(f64::from(max_distance));
    let m = magnitude(bvh_a, bvh_b, xf);
    let mut best: Option<(TriangleId, TriangleId, TriTriDist)> = None;
    for_each_pair::<N>(
        mesh_a,
        bvh_a,
        mesh_b,
        bvh_b,
        xf,
        |p, q| {
            let d = box_distance(p, q);
            (!beyond(d, bound.get(), m)).then_some(d)
        },
        |k| beyond(k, bound.get(), m),
        |a, pa, b, pb| {
            let d = tri_tri_distance(pa, pb);
            let dist = d.dist2.sqrt();
            if dist > bound.get() {
                return true;
            }
            let better = best.is_none_or(|(ba, bb, bd)| {
                d.dist2 < bd.dist2 || (d.dist2 == bd.dist2 && (a, b) < (ba, bb))
            });
            if better {
                best = Some((a, b, d));
                bound.set(dist);
            }
            // Nothing beats 0; stop at the first touching pair.
            d.dist2 != 0.0
        },
    );
    best.map(|(a, b, d)| MeshDistance {
        distance: d.dist2.sqrt() as f32,
        point_a: d.pa.map(|v| v as f32),
        point_b: d.pb.map(|v| v as f32),
        triangle_a: a,
        triangle_b: b,
    })
}

#[allow(clippy::too_many_arguments)] // As for `triangles_near`.
fn triangles_near_with<const N: usize>(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    margin: f32,
    out_a: &mut Vec<TriangleId>,
    out_b: &mut Vec<TriangleId>,
) -> Result<()> {
    let margin = f64::from(margin);
    let m = magnitude(bvh_a, bvh_b, xf);
    let mut ap_a = Appender::new(out_a);
    let mut ap_b = Appender::new(out_b);
    let mut failed = None;
    for_each_pair::<N>(
        mesh_a,
        bvh_a,
        mesh_b,
        bvh_b,
        xf,
        |p, q| {
            let d = box_distance(p, q);
            (!beyond(d, margin, m)).then_some(d)
        },
        // `key` already culled every pair beyond the fixed margin.
        |_| false,
        |a, pa, b, pb| {
            // `tri_tri_distance` gives exactly 0 for pairs that intersect.
            if tri_tri_distance(pa, pb).dist2.sqrt() <= margin
                && let Err(e) = ap_a.push(a).and_then(|()| ap_b.push(b))
            {
                failed = Some(e);
                return false;
            }
            true
        },
    );
    if let Some(e) = failed {
        // A failed push restored its own vector; restore the other one too.
        ap_a.truncate_appended(0);
        ap_b.truncate_appended(0);
        return Err(e);
    }
    ap_a.sort_dedup_appended();
    ap_b.sort_dedup_appended();
    Ok(())
}

fn intersecting_pairs_with<const N: usize>(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    out: &mut Vec<(TriangleId, TriangleId)>,
) -> Result<usize> {
    let mut ap = Appender::new(out);
    let mut failed = None;
    for_each_pair::<N>(
        mesh_a,
        bvh_a,
        mesh_b,
        bvh_b,
        xf,
        overlap_key,
        |_| false,
        |a, pa, b, pb| {
            if tri_tri_intersect(pa, pb)
                && let Err(e) = ap.push((a, b))
            {
                failed = Some(e);
                return false;
            }
            true
        },
    );
    if let Some(e) = failed {
        return Err(e);
    }
    // Each triangle is in exactly one leaf, so every pair is found once.
    ap.appended().sort_unstable();
    Ok(ap.count())
}

#[cfg(test)]
mod tests {
    use fast_mesh_bvh::Quality;

    use super::*;
    use crate::fail_allocations_after;
    use crate::test_util::{Lcg, build, random_rotation, wavy};

    type M3 = [[f64; 3]; 3];

    /// Rotation by 180° about x: B is turned upside down to face A.
    const FLIP_X: M3 = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    /// Rotation by 90° about z, with exact entries.
    const ROT_Z90: M3 = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];

    fn rot_z(deg: f64) -> M3 {
        let (s, c) = deg.to_radians().sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    fn mul(a: &M3, b: &M3) -> M3 {
        std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
    }

    /// Identity, 30° and an exact 90° about z, and a random rotation.
    fn rotations() -> [M3; 4] {
        [
            Rigid::IDENTITY.rotation,
            rot_z(30.0),
            ROT_Z90,
            random_rotation(&mut Lcg(7)),
        ]
    }

    fn alive(mesh: &Mesh) -> impl Iterator<Item = TriangleId> + '_ {
        (0..mesh.triangle_slots() as TriangleId).filter(|&t| mesh.is_triangle_alive(t))
    }

    /// B turned by `rot` after the flip, with its lowest vertex (once turned;
    /// on ties the one nearest B's middle) right over A's vertex `(3, 2, 2)`,
    /// one of A's highest, near the middle, and `gap` above it. So B is
    /// `gap` above A when `gap > 0`, and its lowest vertex touches or dips
    /// below that peak of A otherwise.
    fn pose(ma: &Mesh, mb: &Mesh, rot: &M3, gap: f64) -> Rigid {
        let rotation = mul(rot, &FLIP_X);
        let mut xf = Rigid {
            rotation,
            translation: [0.0; 3],
        };
        let off_middle = |p: &[f32; 3]| (p[0] - 2.5).abs() + (p[1] - 2.5).abs();
        let low = *mb
            .positions()
            .iter()
            .min_by(|p, q| {
                let (zp, zq) = (xf.apply(**p)[2], xf.apply(**q)[2]);
                zp.total_cmp(&zq)
                    .then(off_middle(p).total_cmp(&off_middle(q)))
            })
            .unwrap();
        let top_a = ma
            .positions()
            .iter()
            .map(|p| f64::from(p[2]))
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(top_a, 2.0);
        let moved = xf.apply(low);
        xf.translation = [3.0 - moved[0], 2.0 - moved[1], top_a + gap - moved[2]];
        xf
    }

    /// Two wavy meshes with their BVHs, for both build qualities.
    fn scenes() -> Vec<(Mesh, Bvh, Mesh, Bvh)> {
        [Quality::Sah, Quality::Fast]
            .into_iter()
            .map(|q| {
                let (ma, mb) = (wavy(6), wavy(5));
                let (ba, bb) = (build(&ma, q), build(&mb, q));
                (ma, ba, mb, bb)
            })
            .collect()
    }

    /// A scene posed by [`pose`].
    struct Posed {
        ma: Mesh,
        ba: Bvh,
        mb: Mesh,
        bb: Bvh,
        xf: Rigid,
        gap: f64,
        /// Whether the rotation is about z, so that B is level over A.
        level: bool,
    }

    /// Every scene under every rotation, separated (0.3), touching (0) and
    /// overlapping (-0.3).
    fn posed() -> Vec<Posed> {
        let mut out = Vec::new();
        for (i, rot) in rotations().iter().enumerate() {
            for gap in [0.3, 0.0, -0.3] {
                for (ma, ba, mb, bb) in scenes() {
                    let xf = pose(&ma, &mb, rot, gap);
                    assert_eq!(xf.validate(), Ok(()));
                    let level = i < 3;
                    out.push(Posed {
                        ma,
                        ba,
                        mb,
                        bb,
                        xf,
                        gap,
                        level,
                    });
                }
            }
        }
        out
    }

    /// Every pair of alive triangles: whether they intersect, and their distance.
    fn brute(ma: &Mesh, mb: &Mesh, xf: &Rigid) -> Vec<(TriangleId, TriangleId, bool, TriTriDist)> {
        let mut out = Vec::new();
        for a in alive(ma) {
            for b in alive(mb) {
                let (pa, pb) = (tri_a(ma, a), tri_b(mb, xf, b));
                out.push((a, b, tri_tri_intersect(pa, pb), tri_tri_distance(pa, pb)));
            }
        }
        out
    }

    fn brute_near(
        all: &[(TriangleId, TriangleId, bool, TriTriDist)],
        margin: f32,
    ) -> (Vec<TriangleId>, Vec<TriangleId>) {
        let near = all
            .iter()
            .filter(|(_, _, hit, d)| *hit || d.dist2.sqrt() <= f64::from(margin));
        let mut a: Vec<_> = near.clone().map(|e| e.0).collect();
        let mut b: Vec<_> = near.map(|e| e.1).collect();
        a.sort_unstable();
        a.dedup();
        b.sort_unstable();
        b.dedup();
        (a, b)
    }

    /// The minimum `dist2`, ties to the smallest pair.
    fn brute_min(
        all: &[(TriangleId, TriangleId, bool, TriTriDist)],
    ) -> Option<(TriangleId, TriangleId, TriTriDist)> {
        let mut best: Option<(TriangleId, TriangleId, TriTriDist)> = None;
        for &(a, b, _, d) in all {
            if best.is_none_or(|(_, _, bd)| d.dist2 < bd.dist2) {
                best = Some((a, b, d));
            }
        }
        best
    }

    fn near(
        ma: &Mesh,
        ba: &Bvh,
        mb: &Mesh,
        bb: &Bvh,
        xf: &Rigid,
        margin: f32,
    ) -> (Vec<TriangleId>, Vec<TriangleId>) {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        triangles_near(ma, ba, mb, bb, xf, margin, &mut a, &mut b).unwrap();
        (a, b)
    }

    #[test]
    fn intersects_matches_brute_force() {
        let mut hits = 0;
        for Posed {
            ma,
            ba,
            mb,
            bb,
            xf,
            gap,
            level,
        } in posed()
        {
            let want = brute(&ma, &mb, &xf).iter().any(|e| e.2);
            assert_eq!(intersects(&ma, &ba, &mb, &bb, &xf), Ok(want), "gap {gap}");
            assert_eq!(intersects_with::<0>(&ma, &ba, &mb, &bb, &xf), want);
            assert_eq!(intersects_with::<2>(&ma, &ba, &mb, &bb, &xf), want);
            if gap > 0.0 {
                assert!(!want);
            }
            if gap <= 0.0 && level {
                assert!(want, "gap {gap}");
            }
            hits += usize::from(want);
        }
        assert!(hits >= 12, "{hits}");
    }

    #[test]
    fn intersecting_pairs_matches_brute_force() {
        let mut total = 0;
        for Posed {
            ma,
            ba,
            mb,
            bb,
            xf,
            gap,
            ..
        } in posed()
        {
            let want: Vec<_> = brute(&ma, &mb, &xf)
                .iter()
                .filter(|e| e.2)
                .map(|e| (e.0, e.1))
                .collect();
            // Every overlapping pose meets, the tilted random one included,
            // so the contact paths run under a general rotation too.
            if gap < 0.0 {
                assert!(!want.is_empty(), "gap {gap}");
            }
            let mut out = vec![(999, 999)];
            let n = intersecting_pairs(&ma, &ba, &mb, &bb, &xf, &mut out).unwrap();
            assert_eq!(n, want.len());
            assert_eq!(out[0], (999, 999));
            assert_eq!(out[1..], want[..]);
            for got in [
                intersecting_pairs_with::<0>(&ma, &ba, &mb, &bb, &xf, &mut out),
                intersecting_pairs_with::<2>(&ma, &ba, &mb, &bb, &xf, &mut out),
            ] {
                assert_eq!(got, Ok(want.len()));
            }
            assert_eq!(out[1 + want.len()..1 + 2 * want.len()], want[..]);
            assert_eq!(out[1 + 2 * want.len()..], want[..]);
            total += want.len();
        }
        assert!(total > 100);
    }

    #[test]
    fn triangles_near_matches_brute_force() {
        let mut found = [0; 3];
        for Posed {
            ma, ba, mb, bb, xf, ..
        } in posed()
        {
            let all = brute(&ma, &mb, &xf);
            for (i, margin) in [0.0, 0.05, 0.5].into_iter().enumerate() {
                let want = brute_near(&all, margin);
                let (mut a, mut b) = (vec![999], vec![998]);
                triangles_near(&ma, &ba, &mb, &bb, &xf, margin, &mut a, &mut b).unwrap();
                assert_eq!((a[0], b[0]), (999, 998));
                assert_eq!((&a[1..], &b[1..]), (&want.0[..], &want.1[..]));
                let (mut a, mut b) = (Vec::new(), Vec::new());
                triangles_near_with::<0>(&ma, &ba, &mb, &bb, &xf, margin, &mut a, &mut b).unwrap();
                assert_eq!((a, b), want);
                let (mut a, mut b) = (Vec::new(), Vec::new());
                triangles_near_with::<2>(&ma, &ba, &mb, &bb, &xf, margin, &mut a, &mut b).unwrap();
                assert_eq!((a, b), want);
                found[i] += want.0.len() + want.1.len();
            }
        }
        assert!(found.iter().all(|&n| n > 50), "{found:?}");
        assert!(found[0] <= found[1] && found[1] < found[2], "{found:?}");
    }

    #[test]
    fn mesh_distance_matches_brute_force() {
        let mut separated = 0;
        for Posed {
            ma,
            ba,
            mb,
            bb,
            xf,
            gap,
            level,
        } in posed()
        {
            let all = brute(&ma, &mb, &xf);
            let (wa, wb, wd) = brute_min(&all).unwrap();
            let got = mesh_distance(&ma, &ba, &mb, &bb, &xf, f32::INFINITY)
                .unwrap()
                .unwrap();
            for other in [
                mesh_distance_with::<0>(&ma, &ba, &mb, &bb, &xf, f32::INFINITY),
                mesh_distance_with::<2>(&ma, &ba, &mb, &bb, &xf, f32::INFINITY),
            ] {
                let other = other.unwrap();
                if wd.dist2 > 0.0 {
                    assert_eq!(other, got);
                } else {
                    assert_eq!(other.distance, 0.0);
                }
            }
            if wd.dist2 == 0.0 {
                assert!(gap <= 0.0);
                assert_eq!(got.distance, 0.0);
                let (pa, pb) = (tri_a(&ma, got.triangle_a), tri_b(&mb, &xf, got.triangle_b));
                // The pair intersects, or its positive distance rounds to 0
                // (seen here: a vertex of B one f64 ulp below a peak of A,
                // where both closest points round to the peak).
                assert!(
                    tri_tri_intersect(pa, pb) || tri_tri_distance(pa, pb).dist2 == 0.0,
                    "gap {gap}, {pa:?} {pb:?}"
                );
                continue;
            }
            separated += 1;
            let d = wd.dist2.sqrt();
            let want = MeshDistance {
                distance: d as f32,
                point_a: wd.pa.map(|v| v as f32),
                point_b: wd.pb.map(|v| v as f32),
                triangle_a: wa,
                triangle_b: wb,
            };
            assert_eq!(got, want, "gap {gap}");
            // B's lowest vertex is `gap` above A's highest; level poses put
            // one right over the other.
            if gap > 0.0 {
                assert!(d >= gap - 1e-9, "{d} vs gap {gap}");
            }
            if level {
                assert!((d - gap).abs() < 1e-6, "{d} vs gap {gap}");
            }
            // `max_distance` is inclusive.
            let above = (d as f32).next_up();
            let below = (d as f32).next_down();
            assert!(f64::from(below) < d && d <= f64::from(above));
            assert_eq!(
                mesh_distance(&ma, &ba, &mb, &bb, &xf, above),
                Ok(Some(want))
            );
            assert_eq!(mesh_distance(&ma, &ba, &mb, &bb, &xf, below), Ok(None));
        }
        assert!(separated >= 8);
    }

    /// A flat triangle A and a single-triangle B that touches it with one vertex
    /// at `[0.25, 0.25, 0]`, under the given transform.
    fn touching(b_points: [[f32; 3]; 3]) -> (Mesh, Bvh, Mesh, Bvh) {
        let a = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let ma = Mesh::from_buffers(&a, &[0, 1, 2]).unwrap();
        let mb = Mesh::from_buffers(b_points.as_flattened(), &[0, 1, 2]).unwrap();
        let (ba, bb) = (build(&ma, Quality::Sah), build(&mb, Quality::Sah));
        (ma, ba, mb, bb)
    }

    #[test]
    fn vertex_exactly_on_face_touches() {
        let cases = [
            (
                [[0.25, 0.25, 0.0], [1.0, 0.25, 1.0], [0.25, 1.0, 1.0]],
                Rigid::IDENTITY,
            ),
            (
                [[0.0, 0.0, 0.0], [0.5, 0.0, 1.0], [0.0, 0.5, 1.0]],
                Rigid {
                    rotation: ROT_Z90,
                    translation: [0.25, 0.25, 0.0],
                },
            ),
        ];
        for (b, xf) in cases {
            let (ma, ba, mb, bb) = touching(b);
            assert_eq!(intersects(&ma, &ba, &mb, &bb, &xf), Ok(true));
            let d = mesh_distance(&ma, &ba, &mb, &bb, &xf, 1.0)
                .unwrap()
                .unwrap();
            assert_eq!(d.distance, 0.0);
            assert_eq!((d.triangle_a, d.triangle_b), (0, 0));
            assert_eq!(d.point_a, [0.25, 0.25, 0.0]);
            assert_eq!(near(&ma, &ba, &mb, &bb, &xf, 0.0), (vec![0], vec![0]));
            let mut pairs = Vec::new();
            assert_eq!(
                intersecting_pairs(&ma, &ba, &mb, &bb, &xf, &mut pairs),
                Ok(1)
            );

            // Lifted by 2^-20, they no longer touch.
            let lift = 2f64.powi(-20);
            let mut up = xf;
            up.translation[2] += lift;
            assert_eq!(intersects(&ma, &ba, &mb, &bb, &up), Ok(false));
            assert_eq!(near(&ma, &ba, &mb, &bb, &up, 0.0), (vec![], vec![]));
            let d = mesh_distance(&ma, &ba, &mb, &bb, &up, 1.0)
                .unwrap()
                .unwrap();
            assert_eq!(d.distance, lift as f32);
            assert_eq!(d.point_b, [0.25, 0.25, lift as f32]);

            // Lifted by 0.5, the distance is exactly 0.5 and `max_distance` is
            // inclusive.
            let mut up = xf;
            up.translation[2] += 0.5;
            let d = mesh_distance(&ma, &ba, &mb, &bb, &up, 0.5)
                .unwrap()
                .unwrap();
            assert_eq!(d.distance, 0.5);
            assert_eq!(
                mesh_distance(&ma, &ba, &mb, &bb, &up, 0.5f32.next_down()),
                Ok(None)
            );
            assert_eq!(near(&ma, &ba, &mb, &bb, &up, 0.5), (vec![0], vec![0]));
            assert_eq!(
                near(&ma, &ba, &mb, &bb, &up, 0.5f32.next_down()),
                (vec![], vec![])
            );
        }
    }

    /// Exact transforms keep the box gap exact while the triangle distances
    /// carry `f64` rounding at the coordinates' magnitude. A's vertex `p` is
    /// A's extreme point along x, and B's triangles lie in the plane `gap`
    /// beyond it, around `p`'s projection. So the true distance equals the box
    /// gap, and the computed one (a projection onto B's face) can fall just
    /// below it. B's two triangles sit in separate leaves, so the second leaf
    /// pair is pruned against the bound set by the first.
    #[test]
    fn pruning_keeps_pairs_at_rounding_level() {
        use fast_mesh_bvh::BuildOptions;
        let opts = BuildOptions {
            max_leaf_size: 1,
            ..Default::default()
        };
        let mut rng = Lcg(2024);
        let ulp = 2f32.powi(-14); // The f32 spacing in [512, 1024).
        for _ in 0..500 {
            let mut f = |lo: f32, span: f32| lo + rng.next() * span;
            let p = [800.0, f(600.0, 300.0), f(600.0, 300.0)];
            let gap = ulp * (1 + (f(0.0, 4.0) as i32)) as f32;
            let x = p[0] + gap;
            let a = [
                p,
                [f(700.0, 100.0), f(600.0, 300.0), f(600.0, 300.0)],
                [f(700.0, 100.0), f(600.0, 300.0), f(600.0, 300.0)],
            ];
            // Each triangle surrounds (p.y, p.z): corners below-left,
            // below-right and above.
            let mut b = Vec::new();
            for _ in 0..2 {
                b.push([x, p[1] - f(1.0, 99.0), p[2] - f(1.0, 99.0)]);
                b.push([x, p[1] + f(1.0, 99.0), p[2] - f(1.0, 99.0)]);
                b.push([x, p[1], p[2] + f(1.0, 99.0)]);
            }
            let ma = Mesh::from_buffers(a.as_flattened(), &[0, 1, 2]).unwrap();
            let mb = Mesh::from_buffers(b.as_flattened(), &[0, 1, 2, 3, 4, 5]).unwrap();
            let ba = Bvh::build(&ma, &opts).unwrap();
            let bb = Bvh::build(&mb, &opts).unwrap();
            let xf = Rigid::IDENTITY;

            let all = brute(&ma, &mb, &xf);
            let (wa, wb, wd) = brute_min(&all).unwrap();
            let d = wd.dist2.sqrt();
            let want = MeshDistance {
                distance: d as f32,
                point_a: wd.pa.map(|v| v as f32),
                point_b: wd.pb.map(|v| v as f32),
                triangle_a: wa,
                triangle_b: wb,
            };
            let got = mesh_distance(&ma, &ba, &mb, &bb, &xf, f32::INFINITY).unwrap();
            assert_eq!(got, Some(want), "gap {gap}, oracle {d}");

            // The smallest f32 bound that admits the computed distance.
            let mut bound = d as f32;
            if f64::from(bound) < d {
                bound = bound.next_up();
            }
            assert!(
                mesh_distance(&ma, &ba, &mb, &bb, &xf, bound)
                    .unwrap()
                    .is_some()
            );
            let (na, nb) = near(&ma, &ba, &mb, &bb, &xf, bound);
            assert!(
                na.contains(&wa) && nb.contains(&wb),
                "gap {gap}, oracle {d}"
            );
        }
    }

    /// Pairs whose box gap is above the bound by less than the slack are still
    /// tested, through each query's `key` and `mesh_distance`'s `visit` check.
    /// A's far triangle sets the magnitude to 1e6 (slack about 9.3e-4); B's two
    /// triangles face A's vertex at the origin from 0.5 and 0.5001, in
    /// separate leaves.
    #[test]
    fn pruning_uses_the_slack() {
        use fast_mesh_bvh::BuildOptions;
        let opts = BuildOptions {
            max_leaf_size: 1,
            ..Default::default()
        };
        let a = [
            [0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [-1e6, 0.0, 0.0],
            [-1e6, 1.0, 0.0],
            [-1e6, 0.0, 1.0],
        ];
        let face = |x: f32| [[x, -1.0, -1.0], [x, 1.0, -1.0], [x, 0.0, 1.0]];
        let b = [face(0.5), face(0.5001)].concat();
        let ma = Mesh::from_buffers(a.as_flattened(), &[0, 1, 2, 3, 4, 5]).unwrap();
        let mb = Mesh::from_buffers(b.as_flattened(), &[0, 1, 2, 3, 4, 5]).unwrap();
        let (ba, bb) = (
            Bvh::build(&ma, &opts).unwrap(),
            Bvh::build(&mb, &opts).unwrap(),
        );
        let xf = Rigid::IDENTITY;
        let m = magnitude(&ba, &bb, &xf);
        assert!(PRUNE_SLACK * m > 5e-4);
        let visits = |f: &mut dyn FnMut()| {
            LEAF_VISITS.with(|c| c.set(0));
            f();
            LEAF_VISITS.with(|c| c.get())
        };

        // `mesh_distance`'s `visit` check: the second face is popped after
        // the bound fell to 0.5, with its key 0.5001 inside the slack.
        let n = visits(&mut || {
            let d = mesh_distance(&ma, &ba, &mb, &bb, &xf, f32::INFINITY).unwrap();
            assert_eq!(d.unwrap().distance, 0.5);
        });
        assert_eq!(n, 2);

        // Both `key`s: a bound just under 0.5 culls nothing near the origin.
        let n = visits(&mut || {
            let d = mesh_distance(&ma, &ba, &mb, &bb, &xf, 0.4999).unwrap();
            assert_eq!(d, None);
        });
        assert_eq!(n, 2);
        let n = visits(&mut || {
            assert_eq!(near(&ma, &ba, &mb, &bb, &xf, 0.4999), (vec![], vec![]));
        });
        assert_eq!(n, 2);
    }

    #[test]
    fn prune_slack_covers_rounding() {
        // A box gap above the bound by less than the slack is kept; one above
        // it by more is pruned; an infinite bound prunes nothing.
        let (m, bound) = (1000.0, 1e-4);
        let slack = PRUNE_SLACK * (m + bound);
        assert!(slack > 1e-7 && slack < 1e-6);
        assert!(!beyond(bound + slack / 2.0, bound, m));
        assert!(beyond(bound + 2.0 * slack, bound, m));
        assert!(!beyond(1e300, f64::INFINITY, m));
        assert!(beyond(1e-9, 0.0, 0.0));
    }

    #[test]
    fn cross_query_consistency() {
        let mut checked = 0;
        for Posed {
            ma,
            ba,
            mb,
            bb,
            xf,
            gap,
            ..
        } in posed()
        {
            if gap <= 0.0 {
                continue;
            }
            let d = mesh_distance(&ma, &ba, &mb, &bb, &xf, f32::INFINITY)
                .unwrap()
                .unwrap();
            assert!(d.distance > 0.0);
            let (a, b) = near(&ma, &ba, &mb, &bb, &xf, d.distance * (1.0 + 1e-6));
            assert!(a.contains(&d.triangle_a), "{d:?}");
            assert!(b.contains(&d.triangle_b), "{d:?}");
            checked += 1;
        }
        assert!(checked >= 8);
    }

    /// Runs all four queries, expecting an error from each, and checks that
    /// the outputs are untouched.
    fn errors(
        ma: &Mesh,
        ba: &Bvh,
        mb: &Mesh,
        bb: &Bvh,
        xf: &Rigid,
        margin: f32,
        max_distance: f32,
    ) -> [QueryError; 4] {
        let (mut a, mut b, mut p) = (vec![7], vec![8], vec![(9, 9)]);
        let out = [
            intersects(ma, ba, mb, bb, xf).unwrap_err(),
            mesh_distance(ma, ba, mb, bb, xf, max_distance).unwrap_err(),
            triangles_near(ma, ba, mb, bb, xf, margin, &mut a, &mut b).unwrap_err(),
            intersecting_pairs(ma, ba, mb, bb, xf, &mut p).unwrap_err(),
        ];
        assert_eq!((a, b, p), (vec![7], vec![8], vec![(9, 9)]));
        out
    }

    #[test]
    fn empty_side_returns_nothing() {
        let check_empty = |ma: &Mesh, ba: &Bvh, mb: &Mesh, bb: &Bvh| {
            let xf = Rigid::IDENTITY;
            assert_eq!(intersects(ma, ba, mb, bb, &xf), Ok(false));
            assert_eq!(mesh_distance(ma, ba, mb, bb, &xf, f32::INFINITY), Ok(None));
            assert_eq!(near(ma, ba, mb, bb, &xf, 1e9), (vec![], vec![]));
            let mut p = Vec::new();
            assert_eq!(intersecting_pairs(ma, ba, mb, bb, &xf, &mut p), Ok(0));
            assert!(p.is_empty());
        };
        let full = wavy(4);
        let bfull = build(&full, Quality::Sah);
        let empty = Mesh::new();
        let bempty = build(&empty, Quality::Sah);
        let mut removed = wavy(4);
        let mut bremoved = build(&removed, Quality::Sah);
        let all: Vec<TriangleId> = (0..removed.triangle_slots() as TriangleId).collect();
        let mut tx = removed.begin_edit();
        tx.remove_triangles(&all).unwrap();
        let p = tx.commit().unwrap();
        bremoved.update(&mut removed, p.changes()).unwrap();

        for (m, b) in [(&empty, &bempty), (&removed, &bremoved)] {
            check_empty(m, b, &full, &bfull);
            check_empty(&full, &bfull, m, b);
            check_empty(m, b, m, b);
        }
    }

    #[test]
    fn same_mesh_twice_is_two_copies() {
        let m = wavy(4);
        let b = build(&m, Quality::Sah);
        let xf = Rigid::IDENTITY;
        assert_eq!(intersects(&m, &b, &m, &b, &xf), Ok(true));
        let mut pairs = Vec::new();
        intersecting_pairs(&m, &b, &m, &b, &xf, &mut pairs).unwrap();
        for t in alive(&m) {
            assert!(pairs.binary_search(&(t, t)).is_ok(), "({t}, {t}) missing");
        }
        assert_eq!(
            mesh_distance(&m, &b, &m, &b, &xf, 0.0)
                .unwrap()
                .unwrap()
                .distance,
            0.0
        );
    }

    #[test]
    fn invalid_input() {
        let m = wavy(3);
        let b = build(&m, Quality::Sah);
        let other = wavy(3);
        let bother = build(&other, Quality::Sah);
        let ok = Rigid::IDENTITY;
        let mut off = ok;
        off.rotation[0][0] += 1e-3;
        let mut reflection = ok;
        reflection.rotation[2][2] = -1.0;
        let mut nan = ok;
        nan.translation[1] = f64::NAN;
        let mut inf = ok;
        inf.rotation[2][1] = f64::INFINITY;
        let is_invalid = |e: QueryError| matches!(e, QueryError::InvalidInput(_));
        // The BVH of `m` is passed with `other`: validation comes before `check`.
        for xf in [off, reflection, nan, inf] {
            for e in errors(&m, &b, &other, &b, &xf, 0.0, 0.0) {
                assert!(is_invalid(e), "{e:?}");
            }
        }
        for bad in [-1.0, f32::NAN, -f32::MIN_POSITIVE] {
            let mut x = (vec![7], vec![8]);
            let e = triangles_near(&m, &b, &other, &b, &ok, bad, &mut x.0, &mut x.1);
            assert!(is_invalid(e.unwrap_err()));
            assert_eq!(x, (vec![7], vec![8]));
            let e = mesh_distance(&m, &b, &other, &b, &ok, bad);
            assert!(is_invalid(e.unwrap_err()));
        }
        // Infinite bounds are fine.
        let mut x = (Vec::new(), Vec::new());
        let r = triangles_near(
            &m,
            &b,
            &other,
            &bother,
            &ok,
            f32::INFINITY,
            &mut x.0,
            &mut x.1,
        );
        assert_eq!(r, Ok(()));
        assert!(!x.0.is_empty());
    }

    #[test]
    fn stale_and_mismatch_on_either_side() {
        let (ma, ba) = (wavy(3), build(&wavy(3), Quality::Sah));
        let (mb, bb) = (wavy(2), build(&wavy(2), Quality::Sah));
        let xf = Rigid::IDENTITY;
        let mm = [QueryError::MeshMismatch; 4];
        // `ba` was built for a different mesh with the same content.
        assert_eq!(errors(&ma, &ba, &mb, &bb, &xf, 0.1, 1.0), mm);
        let ba = build(&ma, Quality::Sah);
        assert_eq!(errors(&ma, &ba, &mb, &bb, &xf, 0.1, 1.0), mm);
        let bb = build(&mb, Quality::Sah);
        assert_eq!(intersects(&ma, &ba, &mb, &bb, &xf), Ok(true));
        assert_eq!(errors(&ma, &bb, &mb, &bb, &xf, 0.1, 1.0), mm);
        assert_eq!(errors(&ma, &ba, &mb, &ba, &xf, 0.1, 1.0), mm);

        let stale = [QueryError::StaleBvh; 4];
        let edit = |m: &mut Mesh| {
            let mut tx = m.begin_edit();
            tx.set_positions(&[0], &[[0.0, 0.0, 0.5]]).unwrap();
            tx.commit().unwrap();
        };
        let (mut ma, mut mb) = (ma, mb);
        edit(&mut ma);
        assert_eq!(errors(&ma, &ba, &mb, &bb, &xf, 0.1, 1.0), stale);
        let ba = build(&ma, Quality::Sah);
        edit(&mut mb);
        assert_eq!(errors(&ma, &ba, &mb, &bb, &xf, 0.1, 1.0), stale);
    }

    #[test]
    fn oom_restores_outputs() {
        let (ma, ba, mb, bb) = scenes().remove(0);
        let xf = pose(&ma, &mb, &Rigid::IDENTITY.rotation, -0.3);

        let mut p = vec![(7, 7)];
        fail_allocations_after(Some(0));
        let r = intersecting_pairs(&ma, &ba, &mb, &bb, &xf, &mut p);
        fail_allocations_after(None);
        assert_eq!(r, Err(QueryError::OutOfMemory));
        assert_eq!(p, vec![(7, 7)]);

        // The first push (to `out_a`) succeeds, the second (to `out_b`) fails.
        let (mut a, mut b) = (vec![7], vec![8]);
        fail_allocations_after(Some(1));
        let r = triangles_near(&ma, &ba, &mb, &bb, &xf, 0.0, &mut a, &mut b);
        fail_allocations_after(None);
        assert_eq!(r, Err(QueryError::OutOfMemory));
        assert_eq!((a, b), (vec![7], vec![8]));
    }
}
