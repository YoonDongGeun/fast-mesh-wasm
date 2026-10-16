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
// Used by the queries added in a later task.
#[allow(dead_code)]
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

#[allow(dead_code)]
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
// Used by the queries added in a later task.
#[allow(dead_code)]
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
// Used by the queries added in a later task.
#[allow(dead_code)]
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
    if !(min_t <= t && t <= max_t) {
        return None;
    }
    let inv = 1.0 / det;
    Some(TriHit {
        t,
        barycentric: [u * inv, v * inv, w * inv],
    })
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
}
