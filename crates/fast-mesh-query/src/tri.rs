//! Watertight ray-triangle intersection (Woop, Benthin and Wald 2013).

/// Which triangle faces a ray ignores. A triangle's front is the side its
/// counter-clockwise winding is seen from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Cull {
    /// Hit both faces.
    #[default]
    None,
    /// Ignore back faces.
    Back,
    /// Ignore front faces.
    Front,
}

/// Per-ray constants, computed once and reused for every triangle.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RayPre {
    pub origin: [f32; 3],
    pub dir: [f32; 3],
    /// `1 / dir`, with zero components giving infinities.
    pub inv_dir: [f32; 3],
    kx: usize,
    ky: usize,
    kz: usize,
    sx: f32,
    sy: f32,
    sz: f32,
}

impl RayPre {
    pub(crate) fn new(origin: [f32; 3], dir: [f32; 3]) -> RayPre {
        let a = [dir[0].abs(), dir[1].abs(), dir[2].abs()];
        let kz = if a[0] >= a[1] && a[0] >= a[2] {
            0
        } else if a[1] >= a[2] {
            1
        } else {
            2
        };
        let mut kx = (kz + 1) % 3;
        let mut ky = (kx + 1) % 3;
        if dir[kz] < 0.0 {
            std::mem::swap(&mut kx, &mut ky);
        }
        RayPre {
            origin,
            dir,
            inv_dir: [1.0 / dir[0], 1.0 / dir[1], 1.0 / dir[2]],
            kx,
            ky,
            kz,
            sx: dir[kx] / dir[kz],
            sy: dir[ky] / dir[kz],
            sz: 1.0 / dir[kz],
        }
    }
}

/// A ray-triangle hit.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TriHit {
    /// Distance along the ray, in units of `dir`.
    pub t: f32,
    /// Weights of the triangle's three corners at the hit.
    pub barycentric: [f32; 3],
}

/// Watertight test of a ray against the triangle `p`, hitting when
/// `min_t <= t <= max_t`. Rays through a shared edge or vertex hit at least one
/// of the triangles meeting there.
///
/// A zero-area triangle is never hit. "Zero area" means that `det`, the sum of
/// the three edge functions, is exactly zero as computed (after the `f64`
/// fallback). A triangle that is collinear in real numbers but whose `det`
/// rounds to a nonzero value may still be hit, at a point that is valid for
/// the rounded geometry. A hit whose `t` or weights are not finite is
/// rejected.
pub(crate) fn ray_triangle(
    pre: &RayPre,
    p: [[f32; 3]; 3],
    cull: Cull,
    min_t: f32,
    max_t: f32,
) -> Option<TriHit> {
    let (kx, ky, kz) = (pre.kx, pre.ky, pre.kz);
    let a = sub(p[0], pre.origin);
    let b = sub(p[1], pre.origin);
    let c = sub(p[2], pre.origin);
    let ax = a[kx] - pre.sx * a[kz];
    let ay = a[ky] - pre.sy * a[kz];
    let bx = b[kx] - pre.sx * b[kz];
    let by = b[ky] - pre.sy * b[kz];
    let cx = c[kx] - pre.sx * c[kz];
    let cy = c[ky] - pre.sy * c[kz];

    let mut u = cx * by - cy * bx;
    let mut v = ax * cy - ay * cx;
    let mut w = bx * ay - by * ax;
    if u == 0.0 || v == 0.0 || w == 0.0 {
        let (ax, ay, bx, by, cx, cy) = (
            f64::from(ax),
            f64::from(ay),
            f64::from(bx),
            f64::from(by),
            f64::from(cx),
            f64::from(cy),
        );
        u = (cx * by - cy * bx) as f32;
        v = (ax * cy - ay * cx) as f32;
        w = (bx * ay - by * ax) as f32;
    }

    let inside = match cull {
        Cull::None => (u >= 0.0 && v >= 0.0 && w >= 0.0) || (u <= 0.0 && v <= 0.0 && w <= 0.0),
        Cull::Back => u >= 0.0 && v >= 0.0 && w >= 0.0,
        Cull::Front => u <= 0.0 && v <= 0.0 && w <= 0.0,
    };
    if !inside {
        return None;
    }
    let det = u + v + w;
    if det == 0.0 {
        return None;
    }
    let t_scaled = u * (pre.sz * a[kz]) + v * (pre.sz * b[kz]) + w * (pre.sz * c[kz]);
    let t = t_scaled / det;
    // `t` overflows to infinity when `t_scaled` does, which an infinite `max_t`
    // would otherwise accept.
    if !(t.is_finite() && min_t <= t && t <= max_t) {
        return None;
    }
    // Dividing rather than multiplying by `1 / det` keeps the weights finite
    // when `det` is subnormal (its reciprocal overflows).
    let barycentric = [u / det, v / det, w / det];
    if !barycentric.iter().all(|b| b.is_finite()) {
        return None;
    }
    Some(TriHit { t, barycentric })
}

/// The part of a triangle that is closest to a query point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feature {
    /// The corner with this index (0, 1 or 2).
    Vertex(u8),
    /// The edge joining corners `k` and `(k + 1) % 3`.
    Edge(u8),
    /// The interior of the triangle.
    Face,
}

/// The point of a triangle closest to a query point.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TriClosest {
    pub point: [f32; 3],
    /// Squared distance from the query to `point`.
    pub dist2: f32,
    /// Weights of the triangle's three corners at `point`.
    pub barycentric: [f32; 3],
    pub feature: Feature,
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = sub(a, b);
    dot(d, d)
}

/// The point `a + t * (b - a)`.
fn lerp_point(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    std::array::from_fn(|k| a[k] + t * (b[k] - a[k]))
}

/// Closest point of the segment `p[i]..p[j]` to `q`, as an edge or an end vertex.
fn on_segment(q: [f32; 3], p: [[f32; 3]; 3], i: usize, j: usize, edge: u8) -> TriClosest {
    let ab = sub(p[j], p[i]);
    let len2 = dot(ab, ab);
    let t = if len2 > 0.0 {
        dot(sub(q, p[i]), ab) / len2
    } else {
        0.0
    };
    let (t, feature) = if t.is_nan() || t <= 0.0 {
        (0.0, Feature::Vertex(i as u8))
    } else if t >= 1.0 {
        (1.0, Feature::Vertex(j as u8))
    } else {
        (t, Feature::Edge(edge))
    };
    let mut barycentric = [0.0; 3];
    barycentric[i] = 1.0 - t;
    barycentric[j] = t;
    let point = lerp_point(p[i], p[j], t);
    TriClosest {
        point,
        dist2: dist2(q, point),
        barycentric,
        feature,
    }
}

/// The point of the triangle `p` closest to `q` (Ericson, Real-Time Collision
/// Detection, 5.1.5), with the feature it lies on. A triangle whose area
/// underflows to zero is treated as its three edges.
pub(crate) fn closest_on_triangle(q: [f32; 3], p: [[f32; 3]; 3]) -> TriClosest {
    let [a, b, c] = p;
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(q, a);
    let bp = sub(q, b);
    let cp = sub(q, c);
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    let vc = d1 * d4 - d3 * d2;
    let vb = d5 * d2 - d1 * d6;
    let va = d3 * d6 - d5 * d4;
    let sum = va + vb + vc;

    if sum == 0.0 || !sum.is_finite() {
        let mut best = on_segment(q, p, 0, 1, 0);
        for (i, j, edge) in [(1, 2, 1), (2, 0, 2)] {
            let c = on_segment(q, p, i, j, edge);
            if c.dist2 < best.dist2 {
                best = c;
            }
        }
        return best;
    }

    let vertex = |k: u8| {
        let mut barycentric = [0.0; 3];
        barycentric[k as usize] = 1.0;
        let point = p[k as usize];
        TriClosest {
            point,
            dist2: dist2(q, point),
            barycentric,
            feature: Feature::Vertex(k),
        }
    };
    let edge = |i: usize, j: usize, t: f32, k: u8| {
        let mut barycentric = [0.0; 3];
        barycentric[i] = 1.0 - t;
        barycentric[j] = t;
        let point = lerp_point(p[i], p[j], t);
        TriClosest {
            point,
            dist2: dist2(q, point),
            barycentric,
            feature: Feature::Edge(k),
        }
    };

    if d1 <= 0.0 && d2 <= 0.0 {
        return vertex(0);
    }
    if d3 >= 0.0 && d4 <= d3 {
        return vertex(1);
    }
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return edge(0, 1, d1 / (d1 - d3), 0);
    }
    if d6 >= 0.0 && d5 <= d6 {
        return vertex(2);
    }
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return edge(2, 0, 1.0 - d2 / (d2 - d6), 2);
    }
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return edge(1, 2, (d4 - d3) / ((d4 - d3) + (d5 - d6)), 1);
    }
    let v = vb / sum;
    let w = vc / sum;
    let point = std::array::from_fn(|k| a[k] + ab[k] * v + ac[k] * w);
    TriClosest {
        point,
        dist2: dist2(q, point),
        barycentric: [1.0 - v - w, v, w],
        feature: Feature::Face,
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    const DOWN: [f32; 3] = [0.0, 0.0, -1.0];
    const UP: [f32; 3] = [0.0, 0.0, 1.0];

    fn cast(o: [f32; 3], d: [f32; 3], p: [[f32; 3]; 3], cull: Cull) -> Option<TriHit> {
        ray_triangle(&RayPre::new(o, d), p, cull, 0.0, f32::INFINITY)
    }

    #[test]
    fn hits_the_interior() {
        let h = cast([0.25, 0.25, 1.0], DOWN, T, Cull::None).unwrap();
        assert_eq!(h.t, 1.0);
        let sum: f32 = h.barycentric.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        for (a, b) in h.barycentric.iter().zip([0.5, 0.25, 0.25]) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn misses_outside_and_behind() {
        assert!(cast([2.0, 2.0, 1.0], DOWN, T, Cull::None).is_none());
        assert!(cast([0.25, 0.25, -1.0], DOWN, T, Cull::None).is_none());
    }

    #[test]
    fn culling() {
        let o = [0.25, 0.25, 1.0];
        assert!(cast(o, DOWN, T, Cull::None).is_some());
        assert!(cast(o, DOWN, T, Cull::Back).is_some());
        assert!(cast(o, DOWN, T, Cull::Front).is_none());
        let u = [0.25, 0.25, -1.0];
        assert!(cast(u, UP, T, Cull::None).is_some());
        assert!(cast(u, UP, T, Cull::Front).is_some());
        assert!(cast(u, UP, T, Cull::Back).is_none());
    }

    #[test]
    fn range_is_closed() {
        let pre = RayPre::new([0.25, 0.25, 1.0], DOWN);
        assert!(ray_triangle(&pre, T, Cull::None, 1.0, f32::INFINITY).is_some());
        assert!(ray_triangle(&pre, T, Cull::None, 0.0, 1.0).is_some());
        assert!(ray_triangle(&pre, T, Cull::None, 0.0, 0.999).is_none());
    }

    #[test]
    fn parallel_and_degenerate_miss() {
        assert!(cast([0.25, 0.25, 0.0], [1.0, 0.0, 0.0], T, Cull::None).is_none());
        let d = [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]];
        assert!(cast([1.0, 1.0, 1.0], DOWN, d, Cull::None).is_none());
    }

    #[test]
    fn subnormal_det_gives_finite_weights() {
        // Edge functions of about 1e-44 are subnormal in f32, so `1 / det`
        // overflows. The hit must still have finite weights.
        let s = 1e-22;
        let p = [[0.0, 0.0, 0.0], [s, 0.0, 0.0], [0.0, s, 0.0]];
        let o = [s / 4.0, s / 4.0, 1.0];
        let h = cast(o, DOWN, p, Cull::None).unwrap();
        assert_eq!(h.t, 1.0);
        assert!(h.barycentric.iter().all(|b| b.is_finite()));
        let sum: f32 = h.barycentric.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn overflowing_t_is_not_a_hit() {
        // The real t is 3e38, but t_scaled = u*z + v*z + w*z overflows to
        // infinity, which `max_t = INFINITY` would otherwise accept.
        let z = -3e38;
        let p = [[-10.0, -10.0, z], [10.0, -10.0, z], [0.0, 10.0, z]];
        let o = [0.0, 0.0, 0.0];
        let pre = RayPre::new(o, DOWN);
        let a = sub(p[0], o);
        assert!((pre.sz * a[2]).is_finite());
        assert_eq!(cast(o, DOWN, p, Cull::None).map(|h| h.t), None);
    }

    #[test]
    fn shared_edge_is_watertight() {
        let t2 = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        for i in 1..=1000 {
            let s = i as f32 / 1001.0;
            let o = [s, 1.0 - s, 1.0];
            assert!(
                cast(o, DOWN, T, Cull::None).is_some() || cast(o, DOWN, t2, Cull::None).is_some(),
                "leak at s = {s}"
            );
        }
        // The shared vertex (1,0,0) belongs to both and to a third triangle.
        let t3 = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
        let o = [1.0, 0.0, 1.0];
        assert!(
            [T, t2, t3]
                .iter()
                .any(|p| cast(o, DOWN, *p, Cull::None).is_some())
        );
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6)
    }

    #[test]
    fn voronoi_regions() {
        let cases = [
            (
                [-1.0, -1.0, 0.0],
                Feature::Vertex(0),
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ),
            (
                [2.0, -1.0, 0.0],
                Feature::Vertex(1),
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ),
            (
                [-1.0, 2.0, 0.0],
                Feature::Vertex(2),
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ),
            (
                [0.5, -1.0, 0.0],
                Feature::Edge(0),
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
            ),
            (
                [1.0, 1.0, 0.0],
                Feature::Edge(1),
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5],
            ),
            (
                [-1.0, 0.5, 0.0],
                Feature::Edge(2),
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 0.5],
            ),
            (
                [0.25, 0.25, 3.0],
                Feature::Face,
                [0.25, 0.25, 0.0],
                [0.5, 0.25, 0.25],
            ),
        ];
        for (q, feature, point, bary) in cases {
            let c = closest_on_triangle(q, T);
            assert_eq!(c.feature, feature, "{q:?}");
            assert!(close(c.point, point), "{q:?}: {:?}", c.point);
            assert!(close(c.barycentric, bary), "{q:?}: {:?}", c.barycentric);
        }
        assert_eq!(closest_on_triangle([0.25, 0.25, 3.0], T).dist2, 9.0);
    }

    #[test]
    fn degenerate_triangle_uses_segments() {
        let line = [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]];
        let c = closest_on_triangle([1.0, 0.0, 0.0], line);
        assert!(close(c.point, [0.5, 0.5, 0.0]));
        assert_eq!(c.dist2, 0.5);
        assert!(matches!(c.feature, Feature::Edge(_)));
        assert!(c.barycentric.iter().all(|b| b.is_finite()));

        let dot = [[1.0, 1.0, 1.0]; 3];
        let c = closest_on_triangle([0.0, 0.0, 0.0], dot);
        assert_eq!(c.feature, Feature::Vertex(0));
        assert_eq!(c.dist2, 3.0);
    }
}
