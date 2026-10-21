//! Closest-point queries against a mesh, accelerated by its BVH.

use std::cell::Cell;

use fast_mesh_bvh::{Aabb, Bvh, NodeKind};
use fast_mesh_store::{Mesh, TriangleId, par};

use crate::ray::{PAR_MIN_QUERIES, QUERY_CHUNK, points};
use crate::tri::{TriClosest, closest_on_triangle};
use crate::{Feature, QueryError, Result, check};

/// Options for [`closest_point`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosestOptions {
    /// Largest distance that counts as found (inclusive). Default
    /// `f32::INFINITY`.
    pub max_distance: f32,
}

impl Default for ClosestOptions {
    fn default() -> Self {
        ClosestOptions {
            max_distance: f32::INFINITY,
        }
    }
}

/// The point of a mesh nearest to a query point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosestHit {
    /// The triangle the point lies on.
    pub triangle: TriangleId,
    /// The nearest point, in mesh-local coordinates.
    pub point: [f32; 3],
    /// Distance from the query to `point`.
    pub distance: f32,
    /// Weights of the triangle's corners 0, 1 and 2 at `point`.
    pub barycentric: [f32; 3],
    /// The vertex, edge or face of the triangle that `point` lies on.
    pub feature: Feature,
}

/// Squared distance from `q` to the box `b`; zero when `q` is inside.
pub(crate) fn box_dist2(q: [f32; 3], b: &Aabb) -> f32 {
    let mut d2 = 0.0;
    for (i, &c) in q.iter().enumerate() {
        let d = (b.min[i] - c).max(c - b.max[i]).max(0.0);
        d2 += d * d;
    }
    d2
}

fn validate_point(p: [f32; 3]) -> Result<()> {
    if !p.iter().all(|c| c.is_finite()) {
        return Err(QueryError::InvalidInput("query point must be finite"));
    }
    Ok(())
}

fn validate(p: [f32; 3], opts: &ClosestOptions) -> Result<()> {
    validate_point(p)?;
    validate_opts(opts)
}

fn validate_opts(opts: &ClosestOptions) -> Result<()> {
    if opts.max_distance.is_nan() || opts.max_distance < 0.0 {
        return Err(QueryError::InvalidInput(
            "max_distance must be zero or greater, and not NaN",
        ));
    }
    Ok(())
}

fn closest(mesh: &Mesh, bvh: &Bvh, q: [f32; 3], opts: &ClosestOptions) -> Option<ClosestHit> {
    let max2 = opts.max_distance * opts.max_distance;
    let best: Cell<Option<(TriangleId, TriClosest<f32>)>> = Cell::new(None);
    let best_dist2 = || best.get().map_or(max2, |(_, c)| c.dist2);
    bvh.traverse_by_key(
        |b| {
            let d2 = box_dist2(q, b);
            (d2 <= max2).then_some(d2)
        },
        |_, node, d2| {
            // Boxes at exactly the best distance are still visited, so that a
            // tie can go to the smaller id.
            if d2 > best_dist2() {
                return false;
            }
            if let NodeKind::Leaf { triangles } = node.kind {
                for &id in triangles {
                    let c = closest_on_triangle(q, points(mesh, id));
                    let better = match best.get() {
                        None => c.dist2 <= max2,
                        Some((bid, b)) => c.dist2 < b.dist2 || (c.dist2 == b.dist2 && id < bid),
                    };
                    if better {
                        best.set(Some((id, c)));
                    }
                }
            }
            true
        },
    );
    best.get().map(|(triangle, c)| ClosestHit {
        triangle,
        point: c.point,
        distance: c.dist2.sqrt(),
        barycentric: c.barycentric,
        feature: c.feature,
    })
}

/// How far above the squared distance `d2` [`closest_seeded`] still visits
/// boxes: a generous bound on how much the computed distance of a triangle
/// can undercut the computed distance to a box that holds it, for
/// coordinates of magnitude `mag`.
///
/// Rounding can put a triangle's computed closest point slightly outside the
/// triangle's box: by a few `f32` ulps of `mag` at an edge (`delta` allows
/// 16), and in practice as little on a face. A point `delta` outside the box
/// is closer than the box by at most `2·sqrt(3·d2)·delta + 3·delta²`, and
/// the two sums of squares round by a few ulps of `d2` more (the `2^-12`
/// term). Exact arithmetic would need no window at all.
fn stray_window(d2: f32, mag: f32) -> f32 {
    let delta = 16.0 * f32::EPSILON * mag;
    d2 * (1.0 / 4096.0) + 4.0 * d2.sqrt() * delta + 4.0 * delta * delta
}

/// The squared distance from `q` to the box of the triangle `p`.
fn triangle_box_dist2(q: [f32; 3], p: [[f32; 3]; 3]) -> f32 {
    let lo = std::array::from_fn(|k| p[0][k].min(p[1][k]).min(p[2][k]));
    let hi = std::array::from_fn(|k| p[0][k].max(p[1][k]).max(p[2][k]));
    box_dist2(q, &Aabb::new(lo, hi))
}

/// [`closest`], when [`closest_seeded`] cannot vouch for its own result.
fn cold_fallback(mesh: &Mesh, bvh: &Bvh, q: [f32; 3], opts: &ClosestOptions) -> Option<ClosestHit> {
    #[cfg(test)]
    FALLBACKS.with(|c| c.set(c.get() + 1));
    closest(mesh, bvh, q, opts)
}

#[cfg(test)]
thread_local! {
    /// Test-only count of the seeded queries that fell back to [`closest`].
    static FALLBACKS: Cell<usize> = const { Cell::new(0) };
}

/// [`closest`], bounded from the start by the distance to `seed` (an alive
/// triangle within `max_distance`): the same result, bit for bit.
///
/// **The tie rule.** Both runs keep the smallest `(distance², id)` among the
/// triangles they evaluate: a smaller computed squared distance wins, and an
/// equal one goes to the smaller id. Neither takes the seed as a candidate
/// unless the traversal reaches it, so the seed never wins a tie it would
/// not win cold.
///
/// **Why the result is the same.** The traversal order depends only on the
/// boxes, so both runs meet the nodes in the same order and differ only in
/// what they prune. The cold run prunes a node whose box distance exceeds
/// its best so far; the seeded run one whose box distance exceeds
/// `b + stray_window(b)` for `b = min(d(seed), best)`. If no triangle's
/// computed distance undercut its boxes' distances, every box holding a
/// triangle at least as close as the answer would be visited by both, and
/// both would return the smallest `(distance², id)` over the mesh. Rounding
/// can make a triangle undercut its boxes, by at most [`stray_window`]:
///
/// - the cold winner is never pruned here: its boxes are within the window
///   of its distance, which is at most `b` (the seed is reached, as its own
///   box is checked below, so `b` never drops under the answer);
/// - nothing closer than the cold winner can be found here unless the cold
///   run pruned it, which only a triangle that undercuts its own box can
///   cause; so the winner's own box is checked, and the query falls back to
///   [`closest`] when it is closer than the winner's distance.
fn closest_seeded(
    mesh: &Mesh,
    bvh: &Bvh,
    q: [f32; 3],
    opts: &ClosestOptions,
    seed: TriangleId,
) -> Option<ClosestHit> {
    let max2 = opts.max_distance * opts.max_distance;
    let sp = points(mesh, seed);
    let d0 = closest_on_triangle(q, sp).dist2;
    if d0.is_nan() || d0 > max2 {
        // No bound to gain.
        return closest(mesh, bvh, q, opts);
    }
    if triangle_box_dist2(q, sp) > d0 {
        return cold_fallback(mesh, bvh, q, opts);
    }
    let mag = bvh.root_bounds().map_or(0.0, |b| {
        b.min
            .iter()
            .chain(&b.max)
            .fold(0.0f32, |m, v| m.max(v.abs()))
    });
    let mag = q.iter().fold(mag, |m, v| m.max(v.abs()));
    let reach = |b: f32| b + stray_window(b, mag);
    let best: Cell<Option<(TriangleId, TriClosest<f32>)>> = Cell::new(None);
    // `reach(min(d0, best))`.
    let limit = Cell::new(reach(d0));
    bvh.traverse_by_key(
        |b| {
            let d2 = box_dist2(q, b);
            (d2 <= max2).then_some(d2)
        },
        |_, node, d2| {
            if d2 > limit.get() {
                return false;
            }
            if let NodeKind::Leaf { triangles } = node.kind {
                for &id in triangles {
                    let c = closest_on_triangle(q, points(mesh, id));
                    // The cold rule, as in `closest`.
                    let better = match best.get() {
                        None => c.dist2 <= max2,
                        Some((bid, b)) => c.dist2 < b.dist2 || (c.dist2 == b.dist2 && id < bid),
                    };
                    if better {
                        best.set(Some((id, c)));
                        if c.dist2 < d0 {
                            limit.set(reach(c.dist2));
                        }
                    }
                }
            }
            true
        },
    );
    match best.get() {
        Some((triangle, c)) if triangle_box_dist2(q, points(mesh, triangle)) <= c.dist2 => {
            Some(ClosestHit {
                triangle,
                point: c.point,
                distance: c.dist2.sqrt(),
                barycentric: c.barycentric,
                feature: c.feature,
            })
        }
        _ => cold_fallback(mesh, bvh, q, opts),
    }
}

/// [`closest_point`] warm-started from `seed`, typically the triangle the
/// previous query from about the same point returned. The result is the same
/// as [`closest_point`]'s, bit for bit, including the triangle chosen among
/// equally distant ones; the seed only bounds the search, so a good one
/// makes it faster. A seed that is `None`, out of range or a dead triangle
/// is ignored.
///
/// # Errors
/// As for [`closest_point`].
pub fn closest_point_seeded(
    mesh: &Mesh,
    bvh: &Bvh,
    p: [f32; 3],
    opts: &ClosestOptions,
    seed: Option<TriangleId>,
) -> Result<Option<ClosestHit>> {
    validate(p, opts)?;
    check(mesh, bvh)?;
    let seed = seed.filter(|&t| (t as usize) < mesh.triangle_slots() && mesh.is_triangle_alive(t));
    Ok(match seed {
        Some(seed) => closest_seeded(mesh, bvh, p, opts, seed),
        None => closest(mesh, bvh, p, opts),
    })
}

/// The point of the mesh nearest to `p` within `opts.max_distance`, or `None`
/// when the mesh has no triangle that close. Equal distances go to the smallest
/// triangle id. Triangles with zero area take part, as their edges.
///
/// # Errors
/// [`QueryError::InvalidInput`] for a non-finite `p`, or a `max_distance` that
/// is negative or NaN; then [`QueryError::MeshMismatch`] or
/// [`QueryError::StaleBvh`] when `bvh` does not match the current `mesh`.
pub fn closest_point(
    mesh: &Mesh,
    bvh: &Bvh,
    p: [f32; 3],
    opts: &ClosestOptions,
) -> Result<Option<ClosestHit>> {
    validate(p, opts)?;
    check(mesh, bvh)?;
    Ok(closest(mesh, bvh, p, opts))
}

/// [`closest_point`] for many points: `out[i]` gets the result for `points[i]`.
///
/// # Errors
/// [`QueryError::InvalidInput`] when the slices differ in length or any point
/// is invalid, and otherwise as for [`closest_point`]. Everything is validated
/// before anything is written, so on error `out` is unchanged.
pub fn closest_point_many(
    mesh: &Mesh,
    bvh: &Bvh,
    points: &[[f32; 3]],
    opts: &ClosestOptions,
    out: &mut [Option<ClosestHit>],
) -> Result<()> {
    if points.len() != out.len() {
        return Err(QueryError::InvalidInput(
            "points and out must have the same length",
        ));
    }
    validate_opts(opts)?;
    for &p in points {
        validate_point(p)?;
    }
    check(mesh, bvh)?;
    par::for_each_chunk_mut(out, QUERY_CHUNK, PAR_MIN_QUERIES, |i, slots| {
        let points = &points[i * QUERY_CHUNK..];
        for (&p, slot) in points.iter().zip(slots) {
            *slot = closest(mesh, bvh, p, opts);
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{Lcg, build, wavy};
    use crate::{Cull, Ray, RayOptions, raycast};
    use fast_mesh_bvh::{Bvh, Quality};

    const OPEN: ClosestOptions = ClosestOptions {
        max_distance: f32::INFINITY,
    };

    /// The alive triangle with the smallest squared distance, ties to the smaller id.
    fn brute(mesh: &Mesh, q: [f32; 3], max: f32) -> Option<(TriangleId, f32)> {
        let mut best: Option<(TriangleId, f32)> = None;
        for id in 0..mesh.triangle_slots() as TriangleId {
            if !mesh.is_triangle_alive(id) {
                continue;
            }
            let d2 = closest_on_triangle(q, points(mesh, id)).dist2;
            if d2 <= max * max && best.is_none_or(|(_, b)| d2 < b) {
                best = Some((id, d2));
            }
        }
        best
    }

    /// 500 points around and above a `wavy(n)` grid.
    fn queries(n: u32) -> Vec<[f32; 3]> {
        let mut rng = Lcg(0xc105e57);
        let span = n as f32 + 6.0;
        (0..500)
            .map(|_| {
                [
                    rng.next() * span - 3.0,
                    rng.next() * span - 3.0,
                    rng.next() * 6.0 - 1.0,
                ]
            })
            .collect()
    }

    #[test]
    fn matches_brute_force() {
        let mesh = wavy(16);
        for quality in [Quality::Sah, Quality::Fast] {
            let bvh = build(&mesh, quality);
            for q in queries(16) {
                let got = closest_point(&mesh, &bvh, q, &OPEN).unwrap().unwrap();
                let (id, d2) = brute(&mesh, q, f32::INFINITY).unwrap();
                assert_eq!((got.triangle, got.distance), (id, d2.sqrt()), "{q:?}");
                let c = closest_on_triangle(q, points(&mesh, id));
                assert_eq!(
                    (got.point, got.barycentric, got.feature),
                    (c.point, c.barycentric, c.feature)
                );
            }
        }
    }

    #[test]
    fn max_distance_filters_like_brute_force() {
        let mesh = wavy(16);
        let bvh = build(&mesh, Quality::Sah);
        for max_distance in [0.5, 1.5] {
            let opts = ClosestOptions { max_distance };
            let mut found = 0;
            for q in queries(16) {
                let got = closest_point(&mesh, &bvh, q, &opts)
                    .unwrap()
                    .map(|h| (h.triangle, h.distance));
                let want = brute(&mesh, q, max_distance).map(|(id, d2)| (id, d2.sqrt()));
                assert_eq!(got, want, "{q:?}");
                found += usize::from(got.is_some());
            }
            assert!(found > 0);
        }
    }

    #[test]
    fn max_distance_is_inclusive() {
        let flat =
            Mesh::from_buffers(&[0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0, 0.0], &[0, 1, 2]).unwrap();
        let bvh = build(&flat, Quality::Sah);
        let q = [1.0, 1.0, 2.0];
        let at = |max_distance| ClosestOptions { max_distance };
        let h = closest_point(&flat, &bvh, q, &at(2.0)).unwrap().unwrap();
        assert_eq!(h.distance, 2.0);
        assert_eq!(closest_point(&flat, &bvh, q, &at(1.999)), Ok(None));
    }

    #[test]
    fn equal_distance_takes_smallest_id() {
        // The point (0.5, 0.5, 0) lies on the diagonal shared by both triangles.
        let p = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let mesh = Mesh::from_buffers(&p, &[0, 2, 3, 0, 1, 2]).unwrap();
        for quality in [Quality::Sah, Quality::Fast] {
            let bvh = build(&mesh, quality);
            let h = closest_point(&mesh, &bvh, [0.5, 0.5, 1.0], &OPEN)
                .unwrap()
                .unwrap();
            assert_eq!(h.triangle, 0);
        }
    }

    #[test]
    fn degenerate_only_mesh() {
        let p = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0];
        let mesh = Mesh::from_buffers(&p, &[0, 1, 2]).unwrap();
        let bvh = build(&mesh, Quality::Sah);
        let h = closest_point(&mesh, &bvh, [1.0, 0.0, 0.0], &OPEN)
            .unwrap()
            .unwrap();
        assert_eq!(h.triangle, 0);
        assert!(h.distance.is_finite());
        assert_eq!(h.distance, 0.5f32.sqrt());
        let ray = Ray {
            origin: [1.0, 1.0, 1.0],
            dir: [0.0, 0.0, -1.0],
        };
        let opts = RayOptions {
            cull: Cull::None,
            ..Default::default()
        };
        assert_eq!(raycast(&mesh, &bvh, &ray, &opts), Ok(None));
    }

    #[test]
    fn empty_mesh_returns_nothing() {
        let mesh = Mesh::new();
        let bvh = build(&mesh, Quality::Sah);
        assert_eq!(closest_point(&mesh, &bvh, [0.0; 3], &OPEN), Ok(None));
        let mut out = [None];
        closest_point_many(&mesh, &bvh, &[[0.0; 3]], &OPEN, &mut out).unwrap();
        assert_eq!(out, [None]);
    }

    #[test]
    fn invalid_input() {
        let mesh = wavy(2);
        let bvh = build(&mesh, Quality::Sah);
        let invalid = |r: Result<Option<ClosestHit>>| matches!(r, Err(QueryError::InvalidInput(_)));
        let nan = [f32::NAN, 0.0, 0.0];
        assert!(invalid(closest_point(&mesh, &bvh, nan, &OPEN)));
        for max_distance in [-1.0, f32::NAN] {
            let opts = ClosestOptions { max_distance };
            assert!(invalid(closest_point(&mesh, &bvh, [0.0; 3], &opts)));
            let mut out = [None];
            assert!(matches!(
                closest_point_many(&mesh, &bvh, &[[0.0; 3]], &opts, &mut out),
                Err(QueryError::InvalidInput(_))
            ));
        }
        let mut out = [None, None];
        assert!(matches!(
            closest_point_many(&mesh, &bvh, &[[0.0; 3]], &OPEN, &mut out),
            Err(QueryError::InvalidInput(_))
        ));
    }

    #[test]
    fn many_validates_before_writing() {
        let mesh = wavy(2);
        let bvh = build(&mesh, Quality::Sah);
        let mut out = [None, None];
        let r = closest_point_many(&mesh, &bvh, &[[0.0; 3], [f32::NAN; 3]], &OPEN, &mut out);
        assert!(matches!(r, Err(QueryError::InvalidInput(_))));
        assert_eq!(out, [None, None]);
    }

    #[test]
    fn stale_and_mismatch_are_rejected() {
        let mut mesh = wavy(2);
        let bvh = build(&mesh, Quality::Sah);
        let other = wavy(2);
        assert_eq!(
            closest_point(&other, &bvh, [0.0; 3], &OPEN),
            Err(QueryError::MeshMismatch)
        );
        let mut tx = mesh.begin_edit();
        tx.set_positions(&[0], &[[0.0, 0.0, 1.0]]).unwrap();
        tx.commit().unwrap();
        assert_eq!(
            closest_point(&mesh, &bvh, [0.0; 3], &OPEN),
            Err(QueryError::StaleBvh)
        );
        let mut out = [None];
        assert_eq!(
            closest_point_many(&mesh, &bvh, &[[0.0; 3]], &OPEN, &mut out),
            Err(QueryError::StaleBvh)
        );
    }

    #[test]
    fn many_matches_single() {
        let mesh = wavy(16);
        let bvh: Bvh = build(&mesh, Quality::Sah);
        let qs = queries(16);
        let mut out = vec![None; qs.len()];
        closest_point_many(&mesh, &bvh, &qs, &OPEN, &mut out).unwrap();
        for (q, got) in qs.iter().zip(&out) {
            assert_eq!(*got, closest_point(&mesh, &bvh, *q, &OPEN).unwrap());
        }
    }

    #[test]
    fn box_distance() {
        use fast_mesh_bvh::Aabb;
        let b = Aabb::new([0.0; 3], [1.0; 3]);
        assert_eq!(box_dist2([0.5; 3], &b), 0.0);
        assert_eq!(box_dist2([2.0, 0.5, 0.5], &b), 1.0);
        assert_eq!(box_dist2([-1.0, 2.0, 0.5], &b), 2.0);
    }

    /// Field by field, with every float compared by its bits.
    fn same_hit(a: Option<ClosestHit>, b: Option<ClosestHit>) -> bool {
        let key = |h: Option<ClosestHit>| {
            h.map(|h| {
                (
                    h.triangle,
                    h.point.map(f32::to_bits),
                    h.distance.to_bits(),
                    h.barycentric.map(f32::to_bits),
                    h.feature,
                )
            })
        };
        key(a) == key(b)
    }

    /// Asserts that every seed in `seeds` gives the cold result at `q`.
    fn assert_seeded(mesh: &Mesh, bvh: &Bvh, q: [f32; 3], opts: &ClosestOptions, seeds: &[u32]) {
        let cold = closest_point(mesh, bvh, q, opts).unwrap();
        let none = closest_point_seeded(mesh, bvh, q, opts, None).unwrap();
        assert!(same_hit(none, cold), "{q:?}, no seed: {none:?} != {cold:?}");
        for &s in seeds {
            let warm = closest_point_seeded(mesh, bvh, q, opts, Some(s)).unwrap();
            assert!(
                same_hit(warm, cold),
                "{q:?}, seed {s}: {warm:?} != {cold:?}"
            );
        }
    }

    #[test]
    fn closest_seeded_matches_cold() {
        let mut rng = Lcg(0x5eed);
        let mut grid = wavy(16);
        // Dead triangles, to seed with.
        let dead = [3, 40, 41, 200];
        let mut tx = grid.begin_edit();
        tx.remove_triangles(&dead).unwrap();
        tx.commit().unwrap();
        // The same grid turned and moved off the integer lattice, so that
        // its coordinates and distances round.
        let rot = crate::test_util::random_rotation(&mut rng);
        let turned: Vec<f32> = grid
            .positions()
            .iter()
            .flat_map(|p| {
                let r: [f64; 3] = std::array::from_fn(|i| {
                    (0..3).map(|j| rot[i][j] * f64::from(p[j])).sum::<f64>() + 0.3137
                });
                r.map(|c| c as f32)
            })
            .collect();
        let tris: Vec<u32> = grid.triangles().iter().flatten().copied().collect();
        let turned = Mesh::from_buffers(&turned, &tris).unwrap();
        FALLBACKS.with(|c| c.set(0));
        let mut warm = 0;
        for (mesh, dead) in [(&grid, &dead[..]), (&turned, &[][..])] {
            let slots = mesh.triangle_slots() as u32;
            for quality in [Quality::Sah, Quality::Fast] {
                let bvh = build(mesh, quality);
                for max_distance in [f32::INFINITY, 0.75, 2.5] {
                    let opts = ClosestOptions { max_distance };
                    for q in queries(16) {
                        // The cold winner, near and far live triangles, dead
                        // and out-of-range ids.
                        let mut seeds: Vec<u32> = dead.to_vec();
                        seeds.extend([slots, slots + 7, u32::MAX]);
                        if let Some(h) = closest_point(mesh, &bvh, q, &OPEN).unwrap() {
                            seeds.push(h.triangle);
                            seeds.push(h.triangle ^ 1);
                        }
                        for _ in 0..4 {
                            let t = (rng.next() * slots as f32) as u32;
                            seeds.push(t.min(slots - 1));
                        }
                        assert_seeded(mesh, &bvh, q, &opts, &seeds);
                        warm += seeds.len();
                    }
                }
            }
        }
        // Nearly every live seed within reach takes the warm path.
        let fell_back = FALLBACKS.with(Cell::get);
        assert!(
            fell_back * 20 < warm,
            "{fell_back} of {warm} seeded queries fell back"
        );
    }

    #[test]
    fn warm_ties_match_cold() {
        // Two coplanar triangles sharing the diagonal of the unit square, and
        // a symmetric "V" folded along x = 0.5: points on the shared edge, or
        // above it, are equally far from both.
        let square = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let flat = Mesh::from_buffers(&square, &[0, 2, 3, 0, 1, 2]).unwrap();
        let v = [
            0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0,
            1.0,
        ];
        let vee = Mesh::from_buffers(&v, &[0, 2, 3, 0, 3, 1, 2, 4, 5, 2, 5, 3]).unwrap();
        // A fan of eight triangles around one vertex: a point above it is
        // equally far from all of them.
        let mut fan_pos = vec![0.0, 0.0, 0.0];
        for k in 0..8 {
            let a = k as f32 * std::f32::consts::FRAC_PI_4;
            fan_pos.extend([a.cos(), a.sin(), 0.0]);
        }
        let fan_idx: Vec<u32> = (0..8).flat_map(|k| [0, 1 + k, 1 + (k + 1) % 8]).collect();
        let fan = Mesh::from_buffers(&fan_pos, &fan_idx).unwrap();
        let mut points = Vec::new();
        for k in 0..=16 {
            let t = k as f32 / 16.0;
            for h in [0.0, 0.25, 1.0, 3.0] {
                points.push([t, t, h]);
                points.push([0.5, t, h]);
                points.push([0.5, t, -h]);
            }
        }
        points.extend([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, -2.5]]);
        for mesh in [&flat, &vee, &fan] {
            let ids: Vec<u32> = (0..mesh.triangle_slots() as u32).collect();
            for quality in [Quality::Sah, Quality::Fast] {
                let bvh = build(mesh, quality);
                for max_distance in [f32::INFINITY, 1.0, 3.0] {
                    let opts = ClosestOptions { max_distance };
                    for &q in &points {
                        assert_seeded(mesh, &bvh, q, &opts, &ids);
                    }
                }
            }
        }
    }

    #[test]
    fn seeded_rejects_invalid_input() {
        let mesh = wavy(2);
        let bvh = build(&mesh, Quality::Sah);
        let nan = [f32::NAN, 0.0, 0.0];
        assert!(matches!(
            closest_point_seeded(&mesh, &bvh, nan, &OPEN, Some(0)),
            Err(QueryError::InvalidInput(_))
        ));
        let other = wavy(2);
        assert_eq!(
            closest_point_seeded(&other, &bvh, [0.0; 3], &OPEN, Some(0)),
            Err(QueryError::MeshMismatch)
        );
    }

    #[test]
    fn batch_queries_bit_identical() {
        use crate::raycast_many;
        use crate::test_util::assert_parallel_identical;
        let mesh = wavy(150);
        let bvh = build(&mesh, Quality::Sah);
        let mut rng = Lcg(0xba7c4);
        let count = 20_000;
        let span = 156.0;
        let mut around = |z: f32| [rng.next() * span - 3.0, rng.next() * span - 3.0, z];
        let points: Vec<[f32; 3]> = (0..count).map(|k| around((k % 7) as f32 - 1.0)).collect();
        let origins: Vec<[f32; 3]> = (0..count).map(|_| around(5.0)).collect();
        let dirs: Vec<[f32; 3]> = (0..count)
            .map(|k| [(k % 5) as f32 * 0.2 - 0.4, (k % 3) as f32 * 0.3 - 0.3, -1.0])
            .collect();
        // Debug prints every float in full, so equal strings mean equal bits.
        let closest = assert_parallel_identical("closest_point_many", || {
            let mut out = vec![None; count];
            closest_point_many(&mesh, &bvh, &points, &OPEN, &mut out).unwrap();
            format!("{out:?}")
        });
        let one: Vec<_> = points
            .iter()
            .map(|&p| closest_point(&mesh, &bvh, p, &OPEN).unwrap())
            .collect();
        assert_eq!(closest, format!("{one:?}"));
        let opts = RayOptions::default();
        let hits = assert_parallel_identical("raycast_many", || {
            let mut out = vec![None; count];
            raycast_many(&mesh, &bvh, &origins, &dirs, &opts, &mut out).unwrap();
            format!("{out:?}")
        });
        let one: Vec<_> = origins
            .iter()
            .zip(&dirs)
            .map(|(&origin, &dir)| raycast(&mesh, &bvh, &Ray { origin, dir }, &opts).unwrap())
            .collect();
        assert_eq!(hits, format!("{one:?}"));
        assert!(one.iter().filter(|h| h.is_some()).count() > count / 2);
    }

    #[test]
    fn small_inputs_stay_sequential() {
        use crate::{Rigid, intersecting_pairs, raycast_many, self_intersecting_pairs};
        use crate::{test_util::crossed_ripples, triangles_near};
        use fast_mesh_store::par;
        let before = par::parallel_entries();
        let mesh = wavy(20);
        let bvh = build(&mesh, Quality::Sah);
        let points = &queries(20)[..200];
        let mut out = vec![None; points.len()];
        closest_point_many(&mesh, &bvh, points, &OPEN, &mut out).unwrap();
        let dirs = vec![[0.0, 0.0, -1.0]; points.len()];
        let mut hits = vec![None; points.len()];
        raycast_many(
            &mesh,
            &bvh,
            points,
            &dirs,
            &RayOptions::default(),
            &mut hits,
        )
        .unwrap();
        let (mut near_a, mut near_b, mut pairs) = (Vec::new(), Vec::new(), Vec::new());
        let id = Rigid::IDENTITY;
        triangles_near(&mesh, &bvh, &mesh, &bvh, &id, 0.5, &mut near_a, &mut near_b).unwrap();
        intersecting_pairs(&mesh, &bvh, &mesh, &bvh, &id, &mut pairs).unwrap();
        assert!(!near_a.is_empty() && !pairs.is_empty());
        let crossed = crossed_ripples(10);
        let crossed_bvh = build(&crossed, Quality::Sah);
        pairs.clear();
        self_intersecting_pairs(&crossed, &crossed_bvh, &mut pairs).unwrap();
        assert!(!pairs.is_empty());
        assert_eq!(par::parallel_entries(), before);
    }
}
