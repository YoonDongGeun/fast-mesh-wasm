//! Ray casts against a mesh, accelerated by its BVH.

use fast_mesh_bvh::{Aabb, Bvh, NodeKind};
use fast_mesh_store::{Mesh, TriangleId};

use crate::tri::{RayPre, TriHit, ray_triangle};
use crate::{Appender, Cull, QueryError, Result, check};

/// A ray `origin + t * dir`, in mesh-local coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    /// Where the ray starts.
    pub origin: [f32; 3],
    /// The direction. It need not be normalized: `t` is measured in units of
    /// `dir`, so it is a distance only when `dir` has unit length.
    pub dir: [f32; 3],
}

/// Options shared by the raycasts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayOptions {
    /// Smallest `t` that counts as a hit (inclusive). Default `0.0`.
    pub min_t: f32,
    /// Largest `t` that counts as a hit (inclusive). Default `f32::INFINITY`.
    pub max_t: f32,
    /// Which faces to ignore. Default [`Cull::None`].
    pub cull: Cull,
}

impl Default for RayOptions {
    fn default() -> Self {
        RayOptions {
            min_t: 0.0,
            max_t: f32::INFINITY,
            cull: Cull::None,
        }
    }
}

/// Where a ray hit a triangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// The triangle that was hit.
    pub triangle: TriangleId,
    /// The ray parameter of the hit, in units of the ray's `dir`.
    pub t: f32,
    /// The hit point, `origin + t * dir`.
    pub point: [f32; 3],
    /// Weights of the triangle's corners 0, 1 and 2 at the hit.
    pub barycentric: [f32; 3],
    /// The unit geometric normal, facing the side the counter-clockwise
    /// winding is seen from.
    pub normal: [f32; 3],
}

/// `γ₃ = 3ε / (1 − 3ε)` with `ε = f32::EPSILON / 2` (Pharr et al., Ize 2013).
const GAMMA3: f32 = 3.0 * (f32::EPSILON / 2.0) / (1.0 - 3.0 * (f32::EPSILON / 2.0));
/// Factor that moves a slab distance away from zero.
const GROW: f32 = 1.0 + 2.0 * GAMMA3;
/// Factor that moves a slab distance towards zero.
const SHRINK: f32 = 1.0 - 2.0 * GAMMA3;

/// Slab test of the ray against `b`. Returns the entry `t`, clamped to
/// `>= min_t`, when the ray's `[min_t, max_t]` range meets the box.
///
/// The interval is widened by `2γ₃` relative at both ends, so `f32` rounding
/// never culls a box the ray touches. A slab whose `dir` component is zero
/// and whose face plane contains the origin gives `0 * ∞ = NaN`; `max`/`min`
/// ignore NaN, so such a slab does not cull.
pub(crate) fn ray_box(pre: &RayPre, b: &Aabb, min_t: f32, max_t: f32) -> Option<f32> {
    let mut enter = f32::NEG_INFINITY;
    let mut exit = f32::INFINITY;
    for i in 0..3 {
        let inv = pre.inv_dir[i];
        let (near, far) = if inv < 0.0 {
            (b.max[i], b.min[i])
        } else {
            (b.min[i], b.max[i])
        };
        enter = enter.max((near - pre.origin[i]) * inv);
        exit = exit.min((far - pre.origin[i]) * inv);
    }
    let enter = enter * if enter < 0.0 { GROW } else { SHRINK };
    let exit = exit * if exit > 0.0 { GROW } else { SHRINK };
    (enter <= exit && enter <= max_t && exit >= min_t).then_some(enter.max(min_t))
}

fn validate(origin: [f32; 3], dir: [f32; 3]) -> Result<()> {
    if !origin.iter().all(|c| c.is_finite()) {
        return Err(QueryError::InvalidInput("ray origin must be finite"));
    }
    if !dir.iter().all(|c| c.is_finite()) {
        return Err(QueryError::InvalidInput("ray direction must be finite"));
    }
    if dir == [0.0; 3] {
        return Err(QueryError::InvalidInput("ray direction must not be zero"));
    }
    Ok(())
}

fn validate_opts(opts: &RayOptions) -> Result<()> {
    if opts.min_t.is_nan() || opts.max_t.is_nan() || opts.min_t > opts.max_t {
        return Err(QueryError::InvalidInput(
            "min_t and max_t must not be NaN, and min_t must not exceed max_t",
        ));
    }
    Ok(())
}

pub(crate) fn points(mesh: &Mesh, id: TriangleId) -> [[f32; 3]; 3] {
    let positions = mesh.positions();
    mesh.triangles()[id as usize].map(|v| positions[v as usize])
}

fn make_hit(pre: &RayPre, p: [[f32; 3]; 3], triangle: TriangleId, h: TriHit) -> RayHit {
    let point = std::array::from_fn(|i| pre.origin[i] + h.t * pre.dir[i]);
    // In f64 so that tiny triangles do not underflow to a zero normal.
    let e1: [f64; 3] = std::array::from_fn(|i| f64::from(p[1][i]) - f64::from(p[0][i]));
    let e2: [f64; 3] = std::array::from_fn(|i| f64::from(p[2][i]) - f64::from(p[0][i]));
    let n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    RayHit {
        triangle,
        t: h.t,
        point,
        barycentric: h.barycentric,
        normal: n.map(|c| (c / len) as f32),
    }
}

/// Visits, nearest box first, every node whose box the ray meets within
/// `[opts.min_t, bound()]`, calling `leaf` for each triangle of a visited leaf.
/// `bound` is read again before each node, so it can shrink as hits are found.
fn walk(
    bvh: &Bvh,
    pre: &RayPre,
    opts: &RayOptions,
    bound: impl Fn() -> Option<f32>,
    mut leaf: impl FnMut(TriangleId),
) {
    let entry = |b: &Aabb| ray_box(pre, b, opts.min_t, opts.max_t).unwrap_or(f32::INFINITY);
    bvh.traverse_ordered(
        |l, r| entry(l) <= entry(r),
        |_, node| {
            let Some(max_t) = bound() else {
                return false;
            };
            if ray_box(pre, &node.bounds, opts.min_t, max_t).is_none() {
                return false;
            }
            if let NodeKind::Leaf { triangles } = node.kind {
                for &id in triangles {
                    leaf(id);
                }
            }
            true
        },
    );
}

fn closest(
    mesh: &Mesh,
    bvh: &Bvh,
    origin: [f32; 3],
    dir: [f32; 3],
    opts: &RayOptions,
) -> Option<RayHit> {
    let pre = RayPre::new(origin, dir);
    let best: std::cell::Cell<Option<(TriangleId, TriHit)>> = std::cell::Cell::new(None);
    let best_t = || best.get().map_or(opts.max_t, |(_, h)| h.t);
    walk(
        bvh,
        &pre,
        opts,
        // Boxes entered exactly at `best_t` are still visited, so that a tie
        // can go to the smaller id.
        || Some(best_t()),
        |id| {
            let Some(h) = ray_triangle(&pre, points(mesh, id), opts.cull, opts.min_t, best_t())
            else {
                return;
            };
            let better = match best.get() {
                None => true,
                Some((bid, bh)) => h.t < bh.t || (h.t == bh.t && id < bid),
            };
            if better {
                best.set(Some((id, h)));
            }
        },
    );
    best.get()
        .map(|(id, h)| make_hit(&pre, points(mesh, id), id, h))
}

/// The nearest hit with `opts.min_t <= t <= opts.max_t`, or `None`. Equal `t`
/// goes to the smallest triangle id, so the result is deterministic.
///
/// A ray crossing an edge or vertex shared by several triangles hits all of
/// them (the test is watertight); triangles with zero area are never hit.
///
/// # Errors
/// [`QueryError::InvalidInput`] for a non-finite origin, a zero or non-finite
/// direction, or `min_t` or `max_t` that is NaN, or `min_t` greater than `max_t`; then
/// [`QueryError::MeshMismatch`] or [`QueryError::StaleBvh`] when `bvh` does not
/// match the current `mesh`.
pub fn raycast(mesh: &Mesh, bvh: &Bvh, ray: &Ray, opts: &RayOptions) -> Result<Option<RayHit>> {
    validate(ray.origin, ray.dir)?;
    validate_opts(opts)?;
    check(mesh, bvh)?;
    Ok(closest(mesh, bvh, ray.origin, ray.dir, opts))
}

/// Some hit with `opts.min_t <= t <= opts.max_t`, or `None`. It stops at the
/// first hit found, so it suits occlusion tests; which hit it returns is
/// unspecified.
///
/// # Errors
/// As for [`raycast`].
pub fn raycast_any(mesh: &Mesh, bvh: &Bvh, ray: &Ray, opts: &RayOptions) -> Result<Option<RayHit>> {
    validate(ray.origin, ray.dir)?;
    validate_opts(opts)?;
    check(mesh, bvh)?;
    let pre = RayPre::new(ray.origin, ray.dir);
    let found: std::cell::Cell<Option<(TriangleId, TriHit)>> = std::cell::Cell::new(None);
    walk(
        bvh,
        &pre,
        opts,
        || found.get().is_none().then_some(opts.max_t),
        |id| {
            if found.get().is_none()
                && let Some(h) =
                    ray_triangle(&pre, points(mesh, id), opts.cull, opts.min_t, opts.max_t)
            {
                found.set(Some((id, h)));
            }
        },
    );
    Ok(found
        .get()
        .map(|(id, h)| make_hit(&pre, points(mesh, id), id, h)))
}

/// Appends every hit with `opts.min_t <= t <= opts.max_t` to `out`, sorted by
/// `(t, triangle)`, and returns how many were appended. Elements already in
/// `out` are kept. A ray crossing an edge shared by two triangles reports both.
///
/// # Errors
/// As for [`raycast`], and [`QueryError::OutOfMemory`] when `out` cannot grow;
/// on any error `out` keeps its original length.
pub fn raycast_all(
    mesh: &Mesh,
    bvh: &Bvh,
    ray: &Ray,
    opts: &RayOptions,
    out: &mut Vec<RayHit>,
) -> Result<usize> {
    validate(ray.origin, ray.dir)?;
    validate_opts(opts)?;
    check(mesh, bvh)?;
    let pre = RayPre::new(ray.origin, ray.dir);
    let mut ap = Appender::new(out);
    let failed: std::cell::Cell<Option<QueryError>> = std::cell::Cell::new(None);
    walk(
        bvh,
        &pre,
        opts,
        || failed.get().is_none().then_some(opts.max_t),
        |id| {
            if failed.get().is_some() {
                return;
            }
            let p = points(mesh, id);
            if let Some(h) = ray_triangle(&pre, p, opts.cull, opts.min_t, opts.max_t)
                && let Err(e) = ap.push(make_hit(&pre, p, id, h))
            {
                failed.set(Some(e));
            }
        },
    );
    if let Some(e) = failed.get() {
        return Err(e);
    }
    ap.appended()
        .sort_unstable_by(|a, b| a.t.total_cmp(&b.t).then(a.triangle.cmp(&b.triangle)));
    Ok(ap.count())
}

/// [`raycast`] for many rays: `out[i]` gets the nearest hit of the ray
/// `origins[i] + t * dirs[i]`.
///
/// # Errors
/// [`QueryError::InvalidInput`] when the three slices differ in length or any
/// ray is invalid, and otherwise as for [`raycast`]. Every ray is validated
/// before anything is written, so on error `out` is unchanged.
pub fn raycast_many(
    mesh: &Mesh,
    bvh: &Bvh,
    origins: &[[f32; 3]],
    dirs: &[[f32; 3]],
    opts: &RayOptions,
    out: &mut [Option<RayHit>],
) -> Result<()> {
    if origins.len() != dirs.len() || origins.len() != out.len() {
        return Err(QueryError::InvalidInput(
            "origins, dirs and out must have the same length",
        ));
    }
    for (&o, &d) in origins.iter().zip(dirs) {
        validate(o, d)?;
    }
    validate_opts(opts)?;
    check(mesh, bvh)?;
    for ((&o, &d), slot) in origins.iter().zip(dirs).zip(out) {
        *slot = closest(mesh, bvh, o, d, opts);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{Lcg, build, wavy};
    use crate::tri::{RayPre, ray_triangle};
    use crate::{QueryError, fail_allocations_after};
    use fast_mesh_bvh::{BuildOptions, Bvh, Quality};
    use fast_mesh_store::{Mesh, TriangleId};

    const T: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    const DOWN: [f32; 3] = [0.0, 0.0, -1.0];

    fn one_triangle(p: &[f32; 9]) -> (Mesh, Bvh) {
        let mesh = Mesh::from_buffers(p, &[0, 1, 2]).unwrap();
        let bvh = build(&mesh, Quality::Sah);
        (mesh, bvh)
    }

    /// Every hit in range, by testing every alive triangle.
    fn brute_all(mesh: &Mesh, ray: &Ray, opts: &RayOptions) -> Vec<(TriangleId, f32)> {
        let pre = RayPre::new(ray.origin, ray.dir);
        let mut hits = Vec::new();
        for id in 0..mesh.triangle_slots() as TriangleId {
            if !mesh.is_triangle_alive(id) {
                continue;
            }
            if let Some(h) = ray_triangle(&pre, points(mesh, id), opts.cull, opts.min_t, opts.max_t)
            {
                hits.push((id, h.t));
            }
        }
        hits.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        hits
    }

    fn brute_closest(mesh: &Mesh, ray: &Ray, opts: &RayOptions) -> Option<(TriangleId, f32)> {
        brute_all(mesh, ray, opts).first().copied()
    }

    /// 500 rays from above a `wavy(n)` grid, pointing down with some jitter.
    fn rays(n: u32) -> Vec<Ray> {
        let mut rng = Lcg(0x5eed);
        let span = n as f32 + 2.0;
        (0..500)
            .map(|_| Ray {
                origin: [rng.next() * span - 1.0, rng.next() * span - 1.0, 5.0],
                dir: [
                    rng.next() * 0.6 - 0.3,
                    rng.next() * 0.6 - 0.3,
                    -1.0 - rng.next(),
                ],
            })
            .collect()
    }

    /// 500 nearly horizontal rays through a `wavy(n)` grid; most cross it
    /// several times.
    fn shallow_rays(n: u32) -> Vec<Ray> {
        let mut rng = Lcg(0xab1e);
        (0..500)
            .map(|_| {
                let a = rng.next() * std::f32::consts::TAU;
                Ray {
                    origin: [
                        rng.next() * n as f32,
                        rng.next() * n as f32,
                        0.5 + rng.next(),
                    ],
                    dir: [a.cos(), a.sin(), 0.2 * rng.next() - 0.1],
                }
            })
            .collect()
    }

    fn opts() -> RayOptions {
        RayOptions::default()
    }

    #[test]
    fn options_default() {
        let o = RayOptions::default();
        assert_eq!(o.min_t, 0.0);
        assert_eq!(o.max_t, f32::INFINITY);
        assert_eq!(o.cull, Cull::None);
    }

    #[test]
    fn closest_matches_brute_force() {
        let mesh = wavy(16);
        for quality in [Quality::Sah, Quality::Fast] {
            let bvh = build(&mesh, quality);
            let mut hits = 0;
            for ray in rays(16).into_iter().chain(shallow_rays(16)) {
                let got = raycast(&mesh, &bvh, &ray, &opts())
                    .unwrap()
                    .map(|h| (h.triangle, h.t));
                assert_eq!(got, brute_closest(&mesh, &ray, &opts()), "{ray:?}");
                hits += usize::from(got.is_some());
            }
            assert!(hits > 300, "only {hits} rays hit");
        }
    }

    #[test]
    fn all_matches_brute_force_sorted() {
        let mesh = wavy(16);
        let bvh = build(&mesh, Quality::Sah);
        let sentinel = RayHit {
            triangle: 999,
            t: -1.0,
            point: [0.0; 3],
            barycentric: [0.0; 3],
            normal: [0.0; 3],
        };
        let mut multi = 0;
        for ray in rays(16).into_iter().chain(shallow_rays(16)) {
            let mut out = vec![sentinel];
            let n = raycast_all(&mesh, &bvh, &ray, &opts(), &mut out).unwrap();
            assert_eq!(n, out.len() - 1);
            assert_eq!(out[0], sentinel);
            let got: Vec<_> = out[1..].iter().map(|h| (h.triangle, h.t)).collect();
            assert_eq!(got, brute_all(&mesh, &ray, &opts()), "{ray:?}");
            multi += usize::from(n > 1);
        }
        assert!(multi > 10, "only {multi} rays hit more than once");
    }

    #[test]
    fn any_hits_iff_closest_hits() {
        let mesh = wavy(16);
        let bvh = build(&mesh, Quality::Sah);
        for ray in rays(16).into_iter().chain(shallow_rays(16)) {
            let any = raycast_any(&mesh, &bvh, &ray, &opts()).unwrap();
            let closest = raycast(&mesh, &bvh, &ray, &opts()).unwrap();
            assert_eq!(any.is_some(), closest.is_some());
            if let Some(h) = any {
                let pre = RayPre::new(ray.origin, ray.dir);
                let real = ray_triangle(
                    &pre,
                    points(&mesh, h.triangle),
                    Cull::None,
                    0.0,
                    f32::INFINITY,
                );
                assert_eq!(real.map(|r| r.t), Some(h.t));
            }
        }
    }

    #[test]
    fn shared_edge_tie_takes_smallest_id() {
        // Triangles 0 and 1 share the diagonal from (0,0,0) to (1,1,2).
        let mesh = wavy(4);
        let bvh = build(&mesh, Quality::Sah);
        let ray = Ray {
            origin: [0.5, 0.5, 10.0],
            dir: DOWN,
        };
        let h = raycast(&mesh, &bvh, &ray, &opts()).unwrap().unwrap();
        assert_eq!((h.triangle, h.t), (0, 9.0));
        let mut out = Vec::new();
        assert_eq!(raycast_all(&mesh, &bvh, &ray, &opts(), &mut out), Ok(2));
        assert_eq!(
            out.iter().map(|h| (h.triangle, h.t)).collect::<Vec<_>>(),
            [(0, 9.0), (1, 9.0)]
        );
    }

    #[test]
    fn shared_edge_tie_with_reversed_ids_takes_smallest_id() {
        // Same as above, but the triangle listed first is the one with the larger
        // id in each leaf order the tree might pick.
        let p = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let mesh = Mesh::from_buffers(&p, &[0, 2, 3, 0, 1, 2]).unwrap();
        for quality in [Quality::Sah, Quality::Fast] {
            let opts1 = BuildOptions {
                quality,
                max_leaf_size: 1,
                ..Default::default()
            };
            let bvh = Bvh::build(&mesh, &opts1).unwrap();
            let ray = Ray {
                origin: [0.5, 0.5, 1.0],
                dir: DOWN,
            };
            let h = raycast(&mesh, &bvh, &ray, &opts()).unwrap().unwrap();
            assert_eq!((h.triangle, h.t), (0, 1.0));
        }
    }

    #[test]
    fn hit_fields() {
        let (mesh, bvh) = one_triangle(&T);
        let ray = Ray {
            origin: [0.25, 0.25, 1.0],
            dir: DOWN,
        };
        let h = raycast(&mesh, &bvh, &ray, &opts()).unwrap().unwrap();
        assert_eq!(h.triangle, 0);
        assert_eq!(h.t, 1.0);
        for (a, b) in h.point.iter().zip([0.25, 0.25, 0.0]) {
            assert!((a - b).abs() < 1e-6);
        }
        for (a, b) in h.normal.iter().zip([0.0, 0.0, 1.0]) {
            assert!((a - b).abs() < 1e-6);
        }
        for (a, b) in h.barycentric.iter().zip([0.5, 0.25, 0.25]) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn t_is_in_units_of_dir() {
        let (mesh, bvh) = one_triangle(&T);
        let ray = Ray {
            origin: [0.25, 0.25, 1.0],
            dir: [0.0, 0.0, -2.0],
        };
        let h = raycast(&mesh, &bvh, &ray, &opts()).unwrap().unwrap();
        assert_eq!(h.t, 0.5);
        for (a, b) in h.point.iter().zip([0.25, 0.25, 0.0]) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn ray_starting_on_the_surface_hits_at_zero() {
        let (mesh, bvh) = one_triangle(&T);
        let ray = Ray {
            origin: [0.25, 0.25, 0.0],
            dir: DOWN,
        };
        let h = raycast(&mesh, &bvh, &ray, &opts()).unwrap().unwrap();
        assert_eq!(h.t, 0.0);
    }

    #[test]
    fn axis_aligned_ray_on_a_box_face_still_hits() {
        // The triangle's box is x = 1, y in [0.25, 1], z in [0.25, 1]: the
        // origin lies on the planes of its y and z min faces, where dir is zero.
        let p = [1.0, 0.25, 0.25, 1.0, 1.0, 0.25, 1.0, 0.25, 1.0];
        let (mesh, bvh) = one_triangle(&p);
        let ray = Ray {
            origin: [0.0, 0.25, 0.25],
            dir: [1.0, 0.0, 0.0],
        };
        let h = raycast(&mesh, &bvh, &ray, &opts()).unwrap().unwrap();
        assert_eq!(h.t, 1.0);
        assert!(raycast_any(&mesh, &bvh, &ray, &opts()).unwrap().is_some());
        // The same with negative zeros in dir.
        let ray = Ray {
            origin: [0.0, 0.25, 0.25],
            dir: [1.0, -0.0, -0.0],
        };
        assert_eq!(
            raycast(&mesh, &bvh, &ray, &opts()).unwrap().map(|h| h.t),
            Some(1.0)
        );
    }

    #[test]
    fn negative_min_t_finds_hits_behind_the_origin() {
        let mesh = wavy(8);
        let bvh = build(&mesh, Quality::Sah);
        let ray = Ray {
            origin: [3.3, 4.6, -5.0],
            dir: DOWN,
        };
        let o = RayOptions {
            min_t: f32::NEG_INFINITY,
            ..opts()
        };
        let got = raycast(&mesh, &bvh, &ray, &o)
            .unwrap()
            .map(|h| (h.triangle, h.t));
        assert!(got.is_some());
        assert_eq!(got, brute_closest(&mesh, &ray, &o));
        assert_eq!(raycast(&mesh, &bvh, &ray, &opts()), Ok(None));
    }

    #[test]
    fn max_t_and_cull_are_honored() {
        let mesh = wavy(16);
        let bvh = build(&mesh, Quality::Sah);
        for o in [
            RayOptions {
                min_t: 3.5,
                max_t: 4.5,
                cull: Cull::None,
            },
            RayOptions {
                cull: Cull::Back,
                ..opts()
            },
            RayOptions {
                cull: Cull::Front,
                ..opts()
            },
        ] {
            for ray in rays(16).into_iter().chain(shallow_rays(16)) {
                let got = raycast(&mesh, &bvh, &ray, &o)
                    .unwrap()
                    .map(|h| (h.triangle, h.t));
                assert_eq!(got, brute_closest(&mesh, &ray, &o));
                let mut out = Vec::new();
                raycast_all(&mesh, &bvh, &ray, &o, &mut out).unwrap();
                let all: Vec<_> = out.iter().map(|h| (h.triangle, h.t)).collect();
                assert_eq!(all, brute_all(&mesh, &ray, &o));
            }
        }
    }

    #[test]
    fn empty_mesh_returns_nothing() {
        let ray = Ray {
            origin: [0.5, 0.5, 5.0],
            dir: DOWN,
        };
        let check_empty = |mesh: &Mesh, bvh: &Bvh| {
            assert_eq!(raycast(mesh, bvh, &ray, &opts()), Ok(None));
            assert_eq!(raycast_any(mesh, bvh, &ray, &opts()), Ok(None));
            let mut out = Vec::new();
            assert_eq!(raycast_all(mesh, bvh, &ray, &opts(), &mut out), Ok(0));
            assert!(out.is_empty());
        };
        let mesh = Mesh::new();
        let bvh = build(&mesh, Quality::Sah);
        check_empty(&mesh, &bvh);

        let mut mesh = wavy(4);
        let mut bvh = build(&mesh, Quality::Sah);
        let all: Vec<TriangleId> = (0..mesh.triangle_slots() as TriangleId).collect();
        let mut tx = mesh.begin_edit();
        tx.remove_triangles(&all).unwrap();
        let p = tx.commit().unwrap();
        bvh.update(&mut mesh, p.changes()).unwrap();
        check_empty(&mesh, &bvh);
    }

    #[test]
    fn invalid_input() {
        let (mesh, bvh) = one_triangle(&T);
        let good = Ray {
            origin: [0.25, 0.25, 1.0],
            dir: DOWN,
        };
        let bad_rays = [
            Ray {
                dir: [0.0; 3],
                ..good
            },
            Ray {
                dir: [0.0, f32::INFINITY, -1.0],
                ..good
            },
            Ray {
                origin: [f32::NAN, 0.0, 0.0],
                ..good
            },
        ];
        let bad_opts = [
            RayOptions {
                min_t: 2.0,
                max_t: 1.0,
                ..opts()
            },
            RayOptions {
                min_t: f32::NAN,
                ..opts()
            },
        ];
        let cases = bad_rays
            .iter()
            .map(|r| (*r, opts()))
            .chain(bad_opts.iter().map(|o| (good, *o)));
        for (ray, o) in cases {
            let is_invalid =
                |r: Result<Option<RayHit>>| matches!(r, Err(QueryError::InvalidInput(_)));
            assert!(is_invalid(raycast(&mesh, &bvh, &ray, &o)));
            assert!(is_invalid(raycast_any(&mesh, &bvh, &ray, &o)));
            let mut out = Vec::new();
            assert!(matches!(
                raycast_all(&mesh, &bvh, &ray, &o, &mut out),
                Err(QueryError::InvalidInput(_))
            ));
            let mut many = [None];
            assert!(matches!(
                raycast_many(&mesh, &bvh, &[ray.origin], &[ray.dir], &o, &mut many),
                Err(QueryError::InvalidInput(_))
            ));
        }
        let mut out = [None, None];
        assert!(matches!(
            raycast_many(
                &mesh,
                &bvh,
                &[good.origin; 2],
                &[good.dir],
                &opts(),
                &mut out
            ),
            Err(QueryError::InvalidInput(_))
        ));
        let mut out = [None];
        assert!(matches!(
            raycast_many(
                &mesh,
                &bvh,
                &[good.origin; 2],
                &[good.dir; 2],
                &opts(),
                &mut out
            ),
            Err(QueryError::InvalidInput(_))
        ));
    }

    #[test]
    fn nan_max_t_is_invalid() {
        let (mesh, bvh) = one_triangle(&T);
        let ray = Ray {
            origin: [0.25, 0.25, 1.0],
            dir: DOWN,
        };
        let o = RayOptions {
            max_t: f32::NAN,
            ..opts()
        };
        assert!(matches!(
            raycast(&mesh, &bvh, &ray, &o),
            Err(QueryError::InvalidInput(_))
        ));
        assert!(matches!(
            raycast_any(&mesh, &bvh, &ray, &o),
            Err(QueryError::InvalidInput(_))
        ));
        let mut out = Vec::new();
        assert!(matches!(
            raycast_all(&mesh, &bvh, &ray, &o, &mut out),
            Err(QueryError::InvalidInput(_))
        ));
        let mut many = [None];
        assert!(matches!(
            raycast_many(&mesh, &bvh, &[ray.origin], &[ray.dir], &o, &mut many),
            Err(QueryError::InvalidInput(_))
        ));
    }

    #[test]
    fn invalid_input_wins_over_stale() {
        let (mut mesh, bvh) = one_triangle(&T);
        let mut tx = mesh.begin_edit();
        tx.set_positions(&[0], &[[0.0, 0.0, 1.0]]).unwrap();
        tx.commit().unwrap();
        let ray = Ray {
            origin: [0.25, 0.25, 1.0],
            dir: [0.0; 3],
        };
        assert!(matches!(
            raycast(&mesh, &bvh, &ray, &opts()),
            Err(QueryError::InvalidInput(_))
        ));
    }

    #[test]
    fn stale_and_mismatch_are_rejected() {
        let (mut mesh, bvh) = one_triangle(&T);
        let (other, _) = one_triangle(&T);
        let ray = Ray {
            origin: [0.25, 0.25, 1.0],
            dir: DOWN,
        };
        let mut out = Vec::new();
        let mut many = [None];
        assert_eq!(
            raycast(&other, &bvh, &ray, &opts()),
            Err(QueryError::MeshMismatch)
        );

        let mut tx = mesh.begin_edit();
        tx.set_positions(&[0], &[[0.0, 0.0, 1.0]]).unwrap();
        tx.commit().unwrap();
        let stale = Err(QueryError::StaleBvh);
        assert_eq!(raycast(&mesh, &bvh, &ray, &opts()), stale);
        assert_eq!(raycast_any(&mesh, &bvh, &ray, &opts()), stale);
        assert_eq!(
            raycast_all(&mesh, &bvh, &ray, &opts(), &mut out),
            Err(QueryError::StaleBvh)
        );
        assert_eq!(
            raycast_many(&mesh, &bvh, &[ray.origin], &[ray.dir], &opts(), &mut many),
            Err(QueryError::StaleBvh)
        );
    }

    #[test]
    fn raycast_many_matches_raycast() {
        let mesh = wavy(16);
        let bvh = build(&mesh, Quality::Sah);
        let rs = rays(16);
        let origins: Vec<_> = rs.iter().map(|r| r.origin).collect();
        let dirs: Vec<_> = rs.iter().map(|r| r.dir).collect();
        let mut out = vec![None; rs.len()];
        raycast_many(&mesh, &bvh, &origins, &dirs, &opts(), &mut out).unwrap();
        for (ray, got) in rs.iter().zip(&out) {
            assert_eq!(*got, raycast(&mesh, &bvh, ray, &opts()).unwrap());
        }
    }

    #[test]
    fn raycast_all_oom_restores_out() {
        let (mesh, bvh) = one_triangle(&T);
        let ray = Ray {
            origin: [0.25, 0.25, 1.0],
            dir: DOWN,
        };
        let sentinel = RayHit {
            triangle: 7,
            t: 0.0,
            point: [0.0; 3],
            barycentric: [0.0; 3],
            normal: [0.0; 3],
        };
        let mut out = vec![sentinel];
        fail_allocations_after(Some(0));
        let r = raycast_all(&mesh, &bvh, &ray, &opts(), &mut out);
        fail_allocations_after(None);
        assert_eq!(r, Err(QueryError::OutOfMemory));
        assert_eq!(out, vec![sentinel]);
    }

    #[test]
    fn ray_box_is_conservative_and_clamps_entry() {
        use fast_mesh_bvh::Aabb;
        let b = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let pre = RayPre::new([0.5, 0.5, 3.0], DOWN);
        let e = ray_box(&pre, &b, 0.0, f32::INFINITY).unwrap();
        assert!(e <= 2.0 && e > 1.99);
        assert_eq!(ray_box(&pre, &b, 0.0, 1.5), None);
        assert_eq!(ray_box(&pre, &b, 3.5, f32::INFINITY), None);
        // Inside the box: the entry is clamped to min_t.
        let pre = RayPre::new([0.5, 0.5, 0.5], DOWN);
        assert_eq!(ray_box(&pre, &b, 0.0, f32::INFINITY), Some(0.0));
        // Pointing away from it.
        let pre = RayPre::new([0.5, 0.5, 3.0], [0.0, 0.0, 1.0]);
        assert_eq!(ray_box(&pre, &b, 0.0, f32::INFINITY), None);
        // Grazing a face of a flat box.
        let flat = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 0.0]);
        let pre = RayPre::new([-1.0, 0.5, 0.0], [1.0, 0.0, 0.0]);
        assert!(ray_box(&pre, &flat, 0.0, f32::INFINITY).is_some());
    }
}
