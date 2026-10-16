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
use crate::tritri::{P3, TriTriDist, tri_tri_distance, tri_tri_intersect, tri_tri_within};
use crate::{Appender, QueryError, Result, Rigid, check};

/// The closest pair of points between two meshes, found by [`mesh_distance`].
///
/// Both points are in mesh A's local space: `point_b` is the point of B's
/// triangle `triangle_b` as placed by the query's [`Rigid`] transform, not a
/// point in B's own coordinates. Map it back into B's space with
/// [`Rigid::apply`] on [`Rigid::inverse`]; map B's own points into A's space
/// with [`Rigid::apply`], the arithmetic the queries use.
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
/// that touch, intersect, or are closer than `f64` rounding resolves, as for a
/// distance of 0 in [`mesh_distance`]: a pair with a positive gap whose
/// computed distance rounds to 0 counts as near, so a `margin` of 0 can find
/// pairs for which [`intersects`] (exact) would be `false`.
///
/// Both sides come from the same pair tests, so they agree pair for pair; a
/// second call with the meshes swapped and [`Rigid::inverse`] could differ at
/// the rounding level.
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

/// The conditioning guard on [`plane_gap`]: a triangle's plane is used only
/// when `|n|² >= PLANE_GUARD * |u|² * |v|²`, for its edge vectors `u`, `v`
/// from its first corner and `n = u × v`. That is `sin θ >= 0.01` for the
/// angle `θ` between `u` and `v`.
///
/// Each component of the computed `n` is off by at most a few `f64` ulps of
/// `|u| |v|`, so its direction is off by about `ε / sin θ`: at most a few
/// hundred ulps behind this guard, while for a sliver (`sin θ` near 0) the
/// computed normal can point anywhere. [`plane_gap`] stays a lower bound for
/// any direction (it measures the triangle's own extent along `n` too), but a
/// wrong direction makes the bound useless, and a zero or underflowing `n` would
/// divide by zero. So slivers fall back to the box gap alone.
const PLANE_GUARD: f64 = 1e-4;

/// A lower bound on the distance between triangles `a` and `b` (both in A's
/// space), up to `f64` rounding of a few ulps of the coordinates: the larger
/// of the gap between their boxes ([`Tri::box_gap`]) and the gap along each
/// one's normal ([`Tri::plane_gap`]). 0 when nothing separates them.
///
/// Queries skip a pair whose bound is [`beyond`] their current bound, the rule
/// they prune boxes by, so a skipped pair could not have been accepted: the
/// slack covers this bound's rounding and the triangle distance's alike.
/// Intersecting triangles get 0 up to rounding (both gaps are 0 in exact
/// arithmetic), well under the slack. The queries test it with
/// [`skip_pair`].
#[cfg(test)]
fn pair_lower_bound(a: &[P3; 3], b: &[P3; 3]) -> f64 {
    let (a, b) = (Tri::new(*a), Tri::new(*b));
    a.box_gap(&b.lo, &b.hi)
        .max(a.plane_gap(&b.p))
        .max(b.plane_gap(&a.p))
}

/// Whether a pair of triangles can be skipped under `bound`, for meshes of
/// magnitude `m`: whether their `pair_lower_bound` (defined for the tests: the
/// larger of [`Tri::box_gap`] and each one's [`Tri::plane_gap`]) is
/// [`beyond`] `bound`.
///
/// The three gaps are tested one at a time, cheapest first, and the rest are
/// not computed once one is beyond. That decides exactly as testing their
/// maximum would: the maximum exceeds `bound` plus the slack exactly when one
/// of them does (`f64::max` ignores a NaN, as `beyond` rejects one).
fn skip_pair(a: &Tri, b: &Tri, bound: f64, m: f64) -> bool {
    beyond(a.box_gap(&b.lo, &b.hi), bound, m)
        || beyond(a.plane_gap(&b.p), bound, m)
        || beyond(b.plane_gap(&a.p), bound, m)
}

/// A triangle in `f64` (in A's space) with everything [`skip_pair`] needs of
/// it alone: its box and its slab along its computed normal. The queries compute it once per triangle and leaf pair,
/// not once per triangle pair. The arithmetic is the same as computing it
/// afresh for every pair, so the bounds are bit-identical either way.
#[derive(Debug, Clone, Copy)]
struct Tri {
    /// The corners.
    p: [P3; 3],
    /// The low corner of the box: per axis the smallest of the corners.
    lo: P3,
    /// The high corner of the box: per axis the largest of the corners.
    hi: P3,
    /// The slab along the computed normal, or `None` when the triangle fails
    /// [`PLANE_GUARD`].
    slab: Option<Slab>,
}

/// A triangle's computed normal `n = u × v` (for its edge vectors `u`, `v`
/// from its first corner), `|n|`, and the heights `lo..=hi` that its corners
/// span along `n`, measured from its first corner: 0, `n·u` and `n·v`, all
/// about 0.
#[derive(Debug, Clone, Copy)]
struct Slab {
    /// The computed normal.
    n: P3,
    /// `|n|`.
    len: f64,
    /// The lowest height of the corners along `n`.
    lo: f64,
    /// The highest height of the corners along `n`.
    hi: f64,
}

/// `p - q`.
fn sub(p: P3, q: P3) -> P3 {
    [p[0] - q[0], p[1] - q[1], p[2] - q[2]]
}

/// `p · q`.
fn dot(p: P3, q: P3) -> f64 {
    p[0] * q[0] + p[1] * q[1] + p[2] * q[2]
}

impl Tri {
    /// A placeholder for unused slots of a [`Chunk`].
    const UNUSED: Tri = Tri {
        p: [[0.0; 3]; 3],
        lo: [0.0; 3],
        hi: [0.0; 3],
        slab: None,
    };

    /// The triangle with corners `p`.
    fn new(p: [P3; 3]) -> Tri {
        let lo = std::array::from_fn(|k| p[0][k].min(p[1][k]).min(p[2][k]));
        let hi = std::array::from_fn(|k| p[0][k].max(p[1][k]).max(p[2][k]));
        let (u, v) = (sub(p[1], p[0]), sub(p[2], p[0]));
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let nn = dot(n, n);
        // Written so that NaN fails too.
        let slab = (nn > 0.0 && nn >= PLANE_GUARD * dot(u, u) * dot(v, v)).then(|| {
            let (hu, hv) = (dot(n, u), dot(n, v));
            Slab {
                n,
                len: nn.sqrt(),
                lo: hu.min(hv).min(0.0),
                hi: hu.max(hv).max(0.0),
            }
        });
        Tri { p, lo, hi, slab }
    }

    /// The distance between this triangle's box and the box `lo..=hi`, as
    /// [`box_distance`] does it but on `f64` corners.
    ///
    /// Every step is monotone under rounding (subtraction, `max`, squaring a
    /// non-negative value, the sum, `sqrt`), so a box `lo..=hi` that holds
    /// another one never gets a larger result than it: a gap to a union of
    /// boxes is at most the gap to each of them, in `f64` as in exact
    /// arithmetic.
    fn box_gap(&self, lo: &P3, hi: &P3) -> f64 {
        (0..3)
            .map(|k| {
                let gap = (self.lo[k] - hi[k]).max(lo[k] - self.hi[k]).max(0.0);
                gap * gap
            })
            .sum::<f64>()
            .sqrt()
    }

    /// The gap between this triangle `t` and the triangle with corners `o`
    /// along `t`'s computed normal `n`: how far `o`'s corners all lie beyond
    /// the slab that `t`'s corners span along `n`, over `|n|`. 0 when they
    /// overlap along `n`, or when `t` fails [`PLANE_GUARD`].
    ///
    /// For any direction `n` (exactly as computed, rounding and all), every
    /// point of `t` projects into `t`'s slab and every point of `o` into `o`'s,
    /// so the gap between the slabs is at most the distance between the
    /// triangles. With an exact normal `t`'s slab is just its plane (heights 0)
    /// and this is the distance from the plane to `o`'s nearest corner, when
    /// all of `o` lies strictly on one side. The heights carry `f64` rounding of
    /// a few ulps of `|n|` times the coordinates, so the result is off by a few
    /// ulps of the coordinates.
    fn plane_gap(&self, o: &[P3; 3]) -> f64 {
        let Some(s) = &self.slab else {
            return 0.0;
        };
        let h = o.map(|p| dot(s.n, sub(p, self.p[0])));
        let o_lo = h[0].min(h[1]).min(h[2]);
        let o_hi = h[0].max(h[1]).max(h[2]);
        let gap = (o_lo - s.hi).max(s.lo - o_hi);
        if gap > 0.0 { gap / s.len } else { 0.0 }
    }
}

/// The most triangles of one leaf that a [`Chunk`] holds. BVH leaves hold at
/// most 16 triangles; a larger leaf would be taken in chunks of this size.
const CHUNK: usize = 16;

/// Up to [`CHUNK`] consecutive triangles of one leaf as [`Tri`]s, with the
/// union of their boxes.
///
/// The traversal often visits several leaf pairs in a row that share a leaf
/// (it splits one side down to its leaves against a fixed leaf of the other),
/// so a chunk is only refilled when the triangles change.
struct Chunk {
    /// The address and length of the triangle ids it was filled from, which
    /// identify them within one query (the BVHs are borrowed, so they cannot
    /// move). `None` before the first fill.
    from: Option<(*const TriangleId, usize)>,
    /// The triangles, in the order of the ids; slots past `len` are unused.
    tris: [Tri; CHUNK],
    /// How many slots are in use.
    len: usize,
    /// The low corner of the union of the triangles' boxes.
    lo: P3,
    /// The high corner of the union of the triangles' boxes.
    hi: P3,
}

impl Chunk {
    /// An empty chunk.
    fn new() -> Chunk {
        Chunk {
            from: None,
            tris: [Tri::UNUSED; CHUNK],
            len: 0,
            lo: [f64::INFINITY; 3],
            hi: [f64::NEG_INFINITY; 3],
        }
    }

    /// Fills the chunk with the triangles `ids` (at most [`CHUNK`]), each
    /// placed in A's space by `place`, unless it already holds them.
    fn fill(&mut self, ids: &[TriangleId], place: impl Fn(TriangleId) -> [P3; 3]) {
        let from = Some((ids.as_ptr(), ids.len()));
        if self.from == from {
            return;
        }
        self.from = from;
        self.len = ids.len();
        self.lo = [f64::INFINITY; 3];
        self.hi = [f64::NEG_INFINITY; 3];
        for (slot, &t) in self.tris.iter_mut().zip(ids) {
            *slot = Tri::new(place(t));
            for k in 0..3 {
                self.lo[k] = self.lo[k].min(slot.lo[k]);
                self.hi[k] = self.hi[k].max(slot.hi[k]);
            }
        }
    }

    /// The triangles in use.
    fn tris(&self) -> &[Tri] {
        &self.tris[..self.len]
    }
}

/// The pair key for the intersection queries: 0 when the boxes overlap (closed),
/// otherwise cull.
fn overlap_key(p: &Aabb, q: &Aabb) -> Option<f64> {
    p.overlaps(q).then_some(0.0)
}

/// The weight of the distance between box centres in [`distance_key`]'s
/// order.
const TIE_WEIGHT: f64 = 1e-12;

/// `mesh_distance`'s pair key for A's box `p` and B's moved box `q`:
/// `(order, gap)`, or `None` to cull the pair when `gap` is [`beyond`]
/// `bound` (for meshes of magnitude `m`).
///
/// `gap` is [`box_distance`], the only value pruning reads, here and through
/// the key in [`for_each_near_pair`]. `order` is
/// `gap + TIE_WEIGHT * |centre(p) - centre(q)|`, and the traversal compares
/// keys as tuples, so it orders by `order` alone unless two are equal. Since
/// no pruning test reads `order`, the extra term cannot change what is
/// pruned, and needs no slack of its own; it changes only the visiting order,
/// which `mesh_distance`'s answer does not depend on (see
/// `mesh_distance_with`).
///
/// The term breaks the ties among the many leaf box pairs of a near miss
/// that touch (gap 0): the pair whose centres are closest overlaps the most
/// and tends to hold the closest triangles, so the bound falls sooner. At
/// `1e-12` per unit of centre distance it reorders only gaps that differ by
/// less than about `1e-12` times the mesh size.
fn distance_key(p: &Aabb, q: &Aabb, bound: f64, m: f64) -> Option<(f64, f64)> {
    let gap = box_distance(p, q);
    if beyond(gap, bound, m) {
        return None;
    }
    let (cp, cq) = (p.center(), q.center());
    let centres = (0..3)
        .map(|k| {
            let d = f64::from(cp[k]) - f64::from(cq[k]);
            d * d
        })
        .sum::<f64>()
        .sqrt();
    Some((gap + TIE_WEIGHT * centres, gap))
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
    /// Test-only: how many leaf pairs [`for_each_pair`] and
    /// [`for_each_near_pair`] have tested on this thread.
    static LEAF_VISITS: Cell<usize> = const { Cell::new(0) };
    /// Test-only: how many triangle pairs [`for_each_near_pair`] has skipped
    /// by a lower bound on this thread.
    static PAIR_SKIPS: Cell<usize> = const { Cell::new(0) };
}

/// Test-only: counts `n` triangle pairs skipped by a lower bound.
fn count_skips(n: usize) {
    #[cfg(test)]
    PAIR_SKIPS.with(|c| c.set(c.get() + n));
    #[cfg(not(test))]
    let _ = n;
}

/// Calls `pair` for every triangle pair of every leaf pair that the traversal
/// reaches with `key`, B's triangle already in A's space, until `pair` returns
/// `false`. For the intersection queries, which have no distance to bound.
fn for_each_pair<const N: usize>(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    key: impl FnMut(&Aabb, &Aabb) -> Option<f64>,
    mut pair: impl FnMut(TriangleId, [P3; 3], TriangleId, [P3; 3]) -> bool,
) {
    traverse_pairs_with::<N, f64>(bvh_a, bvh_b, xf, key, |va, vb, _| {
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

/// The distance queries' traversal: calls `pair(a, pa, b, pb)` for every
/// triangle pair of every leaf pair that the traversal reaches with `key`
/// that [`skip_pair`] does not skip under `bound()` (for meshes of magnitude
/// `m`), B's triangle already in A's space, until `pair` returns `false`.
/// `bound()` is read afresh at every test, so it may shrink as `pair` runs.
///
/// A popped node pair is dropped when `gap(k)`, the box gap ([`box_distance`])
/// carried in its key `k`, is [`beyond`] `bound()`.
///
/// Leaf pairs are taken a [`Chunk`] of each side at a time. Before any
/// triangle pair is tested, each triangle's box is tested against the union
/// box of the other side's chunk, and a triangle beyond it is skipped against
/// that whole chunk. That gap is at most the box gap to each triangle of the
/// chunk ([`Tri::box_gap`]: exactly, not just up to rounding), which is the
/// first gap [`skip_pair`] tests, and `beyond` only grows with the gap. So
/// every pair this skips, `skip_pair` would have skipped too, under the same
/// `bound()`: the prefilter changes how much work is done, never which pairs
/// reach `pair`.
#[allow(clippy::too_many_arguments)] // The shared five, three callbacks and `m`.
fn for_each_near_pair<const N: usize, K: Copy + PartialOrd>(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    key: impl FnMut(&Aabb, &Aabb) -> Option<K>,
    gap: impl Fn(K) -> f64,
    bound: impl Fn() -> f64,
    m: f64,
    mut pair: impl FnMut(TriangleId, [P3; 3], TriangleId, [P3; 3]) -> bool,
) {
    let (mut ca, mut cb) = (Chunk::new(), Chunk::new());
    traverse_pairs_with::<N, K>(bvh_a, bvh_b, xf, key, |va, vb, k| {
        if beyond(gap(k), bound(), m) {
            return Step::Skip;
        }
        let (Some(ta), Some(tb)) = (leaf(va), leaf(vb)) else {
            return Step::Descend;
        };
        #[cfg(test)]
        LEAF_VISITS.with(|c| c.set(c.get() + 1));
        for ids_a in ta.chunks(CHUNK) {
            ca.fill(ids_a, |a| tri_a(mesh_a, a));
            for ids_b in tb.chunks(CHUNK) {
                cb.fill(ids_b, |b| tri_b(mesh_b, xf, b));
                if !chunk_pairs(ids_a, &ca, ids_b, &cb, &bound, m, &mut pair) {
                    return Step::Stop;
                }
            }
        }
        Step::Skip
    });
}

/// The triangle pairs of two filled chunks for [`for_each_near_pair`], B's
/// triangles in the outer loop. Returns `false` when `pair` did.
fn chunk_pairs(
    ids_a: &[TriangleId],
    ca: &Chunk,
    ids_b: &[TriangleId],
    cb: &Chunk,
    bound: &impl Fn() -> f64,
    m: f64,
    pair: &mut impl FnMut(TriangleId, [P3; 3], TriangleId, [P3; 3]) -> bool,
) -> bool {
    // Each A triangle's box gap to B's union box, once per chunk pair; the
    // bound is checked against it afresh for each B triangle.
    let mut gap_a = [0.0; CHUNK];
    for (g, t) in gap_a.iter_mut().zip(ca.tris()) {
        *g = t.box_gap(&cb.lo, &cb.hi);
    }
    for (&b, tb) in ids_b.iter().zip(cb.tris()) {
        if beyond(tb.box_gap(&ca.lo, &ca.hi), bound(), m) {
            count_skips(ids_a.len());
            continue;
        }
        for ((&a, ta), &g) in ids_a.iter().zip(ca.tris()).zip(&gap_a) {
            let bound = bound();
            if beyond(g, bound, m) || skip_pair(ta, tb, bound, m) {
                count_skips(1);
                continue;
            }
            if !pair(a, ta.p, b, tb.p) {
                return false;
            }
        }
    }
    true
}

/// The first intersecting pair `(a, b)` that the traversal finds, if any.
fn first_intersection<const N: usize>(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
) -> Option<(TriangleId, TriangleId)> {
    let mut found = None;
    for_each_pair::<N>(
        mesh_a,
        bvh_a,
        mesh_b,
        bvh_b,
        xf,
        overlap_key,
        |a, pa, b, pb| {
            if tri_tri_intersect(pa, pb) {
                found = Some((a, b));
            }
            found.is_none()
        },
    );
    found
}

fn intersects_with<const N: usize>(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
) -> bool {
    first_intersection::<N>(mesh_a, bvh_a, mesh_b, bvh_b, xf).is_some()
}

fn mesh_distance_with<const N: usize>(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    max_distance: f32,
) -> Option<MeshDistance> {
    // Touching or intersecting meshes have distance 0, and the cheap exact
    // intersection test finds a pair far faster than the distance search, which
    // starts with no bound to prune by. Any intersecting pair will do.
    //
    // Measured against folding the test into the distance search (issue #22):
    // `tri_tri_distance` already runs the exact test and the search stops at
    // the first pair at distance 0, so the fold would only drop this pass. That
    // saves the pass's cost on near misses, but took intersecting poses from
    // 0.4 ms to 1.5 to 2.2 ms at 1M, since the search must first walk the
    // pairs its order puts before an intersecting one.
    if let Some((a, b)) = first_intersection::<N>(mesh_a, bvh_a, mesh_b, bvh_b, xf) {
        let d = tri_tri_distance(tri_a(mesh_a, a), tri_b(mesh_b, xf, b));
        return Some(MeshDistance {
            distance: 0.0,
            point_a: d.pa.map(|v| v as f32),
            point_b: d.pb.map(|v| v as f32),
            triangle_a: a,
            triangle_b: b,
        });
    }
    // The distance of the best pair so far, or `max_distance` before one is
    // found. Pairs whose boxes are farther (with slack) are pruned, in `key`
    // and again in `visit`, as the bound shrinks after a pair is pushed.
    //
    // The answer does not depend on the order in which pairs are visited
    // (except which pair is reported for a distance of 0, which is
    // unspecified): the bound is always the computed distance of some pair or
    // `max_distance`, so it never falls below the minimum, and a pair is only
    // ever skipped under the `beyond` slack rule, so no pair within the bound
    // is lost. The minimum `dist2`, ties to the smallest `(a, b)`, is then
    // found whatever the order. The order only decides how soon the bound
    // falls, hence how much is pruned; that is what `distance_key` tunes.
    let bound = Cell::new(f64::from(max_distance));
    let m = magnitude(bvh_a, bvh_b, xf);
    let mut best: Option<(TriangleId, TriangleId, TriTriDist)> = None;
    for_each_near_pair::<N, (f64, f64)>(
        mesh_a,
        bvh_a,
        mesh_b,
        bvh_b,
        xf,
        |p, q| distance_key(p, q, bound.get(), m),
        |(_, gap)| gap,
        || bound.get(),
        m,
        // Near misses leave many leaf pairs at key 0 (boxes touch), so most
        // of their triangle pairs are dropped by the lower bounds in
        // `for_each_near_pair`, not by the boxes.
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
    for_each_near_pair::<N, f64>(
        mesh_a,
        bvh_a,
        mesh_b,
        bvh_b,
        xf,
        |p, q| {
            let d = box_distance(p, q);
            (!beyond(d, margin, m)).then_some(d)
        },
        |gap| gap,
        || margin,
        m,
        |a, pa, b, pb| {
            // Exactly `tri_tri_distance(pa, pb).dist2.sqrt() <= margin`
            // (0 for pairs that intersect), but stops at the first candidate
            // within the margin.
            if tri_tri_within(pa, pb, margin)
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

    /// A near miss (B `1e-4` above A, so many leaf boxes touch): the per-pair
    /// lower bound skips most triangle pairs, and the results still match the
    /// brute force.
    #[test]
    fn near_miss_skips_pairs_by_lower_bound() {
        let skips = |f: &mut dyn FnMut()| {
            PAIR_SKIPS.with(|c| c.set(0));
            f();
            PAIR_SKIPS.with(|c| c.get())
        };
        let gap = 1e-4;
        let mut near_skips = [0; 3];
        for rot in rotations() {
            for (ma, ba, mb, bb) in scenes() {
                let xf = pose(&ma, &mb, &rot, gap);
                let all = brute(&ma, &mb, &xf);
                assert!(!all.iter().any(|e| e.2));
                let (wa, wb, wd) = brute_min(&all).unwrap();
                let want = MeshDistance {
                    distance: wd.dist2.sqrt() as f32,
                    point_a: wd.pa.map(|v| v as f32),
                    point_b: wd.pb.map(|v| v as f32),
                    triangle_a: wa,
                    triangle_b: wb,
                };
                let n = skips(&mut || {
                    let got = mesh_distance(&ma, &ba, &mb, &bb, &xf, f32::INFINITY);
                    assert_eq!(got, Ok(Some(want)));
                    let got = mesh_distance_with::<0>(&ma, &ba, &mb, &bb, &xf, f32::INFINITY);
                    assert_eq!(got, Some(want));
                });
                assert!(n > 0, "{rot:?}");
                for (i, margin) in [0.0, 1e-4, 0.05].into_iter().enumerate() {
                    near_skips[i] += skips(&mut || {
                        let got = near(&ma, &ba, &mb, &bb, &xf, margin);
                        assert_eq!(got, brute_near(&all, margin), "margin {margin}");
                    });
                }
            }
        }
        // Under a margin, boxes cull more pairs (their bound is fixed from the
        // start), so a pose can leave nothing for the lower bound to skip.
        assert!(near_skips.iter().all(|&n| n > 0), "{near_skips:?}");
    }

    /// A smooth bumpy `n` x `n` sheet, `z = 0.8 sin(0.7 x) cos(0.5 y)`, so that
    /// a flipped copy over it meets it at a single place, as the bench terrain
    /// does.
    fn bumps(n: u32) -> Mesh {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                let (fx, fy) = (x as f32, y as f32);
                positions.extend([fx, fy, 0.8 * (0.7 * fx).sin() * (0.5 * fy).cos()]);
            }
        }
        let mut indices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        Mesh::from_buffers(&positions, &indices).unwrap()
    }

    /// The brute-force answer of `mesh_distance` with an infinite bound, for
    /// meshes that do not meet.
    fn oracle(all: &[(TriangleId, TriangleId, bool, TriTriDist)]) -> MeshDistance {
        assert!(!all.iter().any(|e| e.2));
        let (a, b, d) = brute_min(all).unwrap();
        MeshDistance {
            distance: d.dist2.sqrt() as f32,
            point_a: d.pa.map(|v| v as f32),
            point_b: d.pb.map(|v| v as f32),
            triangle_a: a,
            triangle_b: b,
        }
    }

    /// Near misses, the case the leaf-pair caches, the leaf prefilters and the
    /// tie order speed up: B a tiny gap over A, level or tilted, on the wavy
    /// meshes and on smooth bumpy sheets, with leaves of 1 to 16 triangles.
    /// `mesh_distance` gives exactly the oracle's pair, points and distance,
    /// through the stack and the overflow fallback, and `max_distance` just
    /// above and below the distance keeps or drops it.
    #[test]
    fn near_misses_match_brute_force() {
        use fast_mesh_bvh::BuildOptions;
        let check = |ma: &Mesh, mb: &Mesh, xf: &Rigid| {
            let all = brute(ma, mb, xf);
            let want = oracle(&all);
            let d = brute_min(&all).unwrap().2.dist2.sqrt();
            assert!(d > 0.0 && d < 0.01, "{d}");
            for (leaf, quality) in [1, 4, 16]
                .into_iter()
                .flat_map(|l| [(l, Quality::Sah), (l, Quality::Fast)])
            {
                let opts = BuildOptions {
                    quality,
                    max_leaf_size: leaf,
                    ..Default::default()
                };
                let (ba, bb) = (
                    Bvh::build(ma, &opts).unwrap(),
                    Bvh::build(mb, &opts).unwrap(),
                );
                let got = mesh_distance(ma, &ba, mb, &bb, xf, f32::INFINITY);
                assert_eq!(got, Ok(Some(want)), "leaf {leaf}, {xf:?}");
                for got in [
                    mesh_distance_with::<0>(ma, &ba, mb, &bb, xf, f32::INFINITY),
                    mesh_distance_with::<2>(ma, &ba, mb, &bb, xf, f32::INFINITY),
                    mesh_distance_with::<5>(ma, &ba, mb, &bb, xf, f32::INFINITY),
                ] {
                    assert_eq!(got, Some(want), "leaf {leaf}");
                }
                let mut above = want.distance;
                if f64::from(above) < d {
                    above = above.next_up();
                }
                let got = mesh_distance(ma, &ba, mb, &bb, xf, above);
                assert_eq!(got, Ok(Some(want)));
                let below = above.next_down();
                assert_eq!(mesh_distance(ma, &ba, mb, &bb, xf, below), Ok(None));
                // `triangles_near` shares the lower bound and the prefilters.
                for margin in [0.0, above, 2.0 * above, 0.05] {
                    let got = near(ma, &ba, mb, &bb, xf, margin);
                    assert_eq!(got, brute_near(&all, margin), "margin {margin}");
                }
            }
        };

        let mut cases = 0;
        let mut rng = Lcg(0x22);
        let tilt = |rng: &mut Lcg| -> M3 {
            // A random turn by up to about 3° about a random axis.
            let r = random_rotation(rng);
            let s = 0.05;
            std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    let id = if i == j { 1.0 } else { 0.0 };
                    id + s * (r[i][j] - r[j][i])
                })
            })
        };
        // Wavy meshes: B's lowest vertex `gap` over a peak of A.
        let (ma, mb) = (wavy(6), wavy(5));
        for gap in [1e-6, 1e-5, 3e-4, 3e-3] {
            for rot in [Rigid::IDENTITY.rotation, rot_z(30.0), ROT_Z90] {
                check(&ma, &mb, &pose(&ma, &mb, &rot, gap));
                cases += 1;
            }
        }
        // Bumpy sheets: B flipped over A, turned a little, and lowered until
        // the gap is near `gap`.
        let (ma, mb) = (bumps(8), bumps(7));
        for gap in [1e-5, 1e-4, 2e-3] {
            for deg in [0.0, 2.5, 7.0] {
                let skew = tilt(&mut rng);
                // Orthonormalized, so that `Rigid::validate` accepts it.
                let rot = orthonormal(&mul(&rot_z(deg), &skew));
                let mut xf = Rigid {
                    rotation: mul(&rot, &FLIP_X),
                    translation: [0.3, 7.4, 2.0],
                };
                for _ in 0..4 {
                    let d = brute_min(&brute(&ma, &mb, &xf)).unwrap().2.dist2.sqrt();
                    xf.translation[2] -= d - gap;
                }
                assert_eq!(xf.validate(), Ok(()));
                check(&ma, &mb, &xf);
                cases += 1;
            }
        }
        assert_eq!(cases, 21);
    }

    /// `m` with its rows made orthonormal (Gram–Schmidt), and the third row
    /// their cross product, so that it is a proper rotation.
    fn orthonormal(m: &M3) -> M3 {
        let dot = |p: &[f64; 3], q: &[f64; 3]| (0..3).map(|k| p[k] * q[k]).sum::<f64>();
        let unit = |p: [f64; 3]| {
            let n = dot(&p, &p).sqrt();
            p.map(|v| v / n)
        };
        let r0 = unit(m[0]);
        let d = dot(&m[1], &r0);
        let r1 = unit(std::array::from_fn(|k| m[1][k] - d * r0[k]));
        let r2 = [
            r0[1] * r1[2] - r0[2] * r1[1],
            r0[2] * r1[0] - r0[0] * r1[2],
            r0[0] * r1[1] - r0[1] * r1[0],
        ];
        [r0, r1, r2]
    }

    #[test]
    fn pair_lower_bound_is_below_the_distance() {
        // Random pairs, slivers and coplanar ones included: the bound never
        // exceeds the computed distance by more than rounding.
        let mut rng = Lcg(0x10b0);
        let mut useful = 0;
        for i in 0..20_000 {
            let mut r = || f64::from(rng.next());
            let a: [P3; 3] = std::array::from_fn(|_| [r(), r(), r()]);
            let mut b: [P3; 3] = std::array::from_fn(|_| [r() + 0.5, r(), r() + 0.3 * r()]);
            match i % 4 {
                // A sliver: the third corner nearly on the first edge.
                1 => b[2] = std::array::from_fn(|k| 0.5 * (b[0][k] + b[1][k]) + 1e-9 * r()),
                // Coplanar with A: A moved within its own plane.
                2 => {
                    let (s, t) = (0.5 + r(), r() - 0.5);
                    b = a.map(|p| {
                        std::array::from_fn(|k| {
                            p[k] + s * (a[1][k] - a[0][k]) + t * (a[2][k] - a[0][k])
                        })
                    });
                }
                _ => {}
            }
            let lb = pair_lower_bound(&a, &b);
            let d = tri_tri_distance(a, b).dist2.sqrt();
            assert!(lb <= d + 1e-12, "{a:?} {b:?}: {lb} > {d}");
            useful += usize::from(lb > 0.5 * d && d > 0.0);
        }
        assert!(useful > 1000, "{useful}");
    }

    /// The leaf prefilter's premise: a triangle's box gap to the union box of
    /// a chunk is never above its box gap to any triangle of the chunk, in
    /// `f64` exactly, so the prefilter skips only pairs `skip_pair` skips.
    #[test]
    fn union_box_gap_is_below_each_pair_gap() {
        let mut rng = Lcg(0x5e1f);
        let mut r = || f64::from(rng.next());
        let tri = |r: &mut dyn FnMut() -> f64, at: P3, size: f64| -> [P3; 3] {
            std::array::from_fn(|_| std::array::from_fn(|k| at[k] + size * r()))
        };
        let mut skipped = 0;
        for _ in 0..2000 {
            let at: P3 = [10.0 * r(), 10.0 * r(), 10.0 * r()];
            let ids: Vec<TriangleId> = (0..1 + (r() * 16.0) as TriangleId).collect();
            let chunk_tris: Vec<[P3; 3]> = ids.iter().map(|_| tri(&mut r, at, 3.0)).collect();
            let mut chunk = Chunk::new();
            chunk.fill(&ids, |t| chunk_tris[t as usize]);
            assert_eq!(chunk.tris().len(), ids.len());
            let near: P3 = [10.0 * r(), 10.0 * r(), 10.0 * r()];
            let other = Tri::new(tri(&mut r, near, 3.0));
            let union_gap = other.box_gap(&chunk.lo, &chunk.hi);
            let bound = 0.5 * r();
            for t in chunk.tris() {
                assert!(union_gap <= other.box_gap(&t.lo, &t.hi));
                assert!(union_gap <= t.box_gap(&other.lo, &other.hi));
                if beyond(union_gap, bound, 10.0) {
                    assert!(skip_pair(t, &other, bound, 10.0));
                    assert!(skip_pair(&other, t, bound, 10.0));
                    skipped += 1;
                }
            }
        }
        assert!(skipped > 1000, "{skipped}");
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
        // Neighbours (two triangles sharing an edge) touch along that edge,
        // so both orders are pairs.
        let tris: Vec<_> = alive(&m).collect();
        let shares_edge = |t: TriangleId, u: TriangleId| {
            let (vt, vu) = (m.triangle(t).unwrap(), m.triangle(u).unwrap());
            vt.iter().filter(|v| vu.contains(v)).count() == 2
        };
        let (t, u) = tris
            .iter()
            .flat_map(|&t| tris.iter().map(move |&u| (t, u)))
            .find(|&(t, u)| shares_edge(t, u))
            .unwrap();
        assert!(pairs.binary_search(&(t, u)).is_ok(), "({t}, {u}) missing");
        assert!(pairs.binary_search(&(u, t)).is_ok(), "({u}, {t}) missing");
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
