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
