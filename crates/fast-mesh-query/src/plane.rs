//! Cutting planes with a 2D frame.

use crate::{QueryError, Result};

/// A plane in mesh-local coordinates, with an orthonormal 2D frame `(u, v)`
/// in it. The frame is right-handed: `normal = u × v`.
///
/// The signed distance of a point `p` is `n·p − n·origin`, in `f64`; points
/// with a signed distance of zero or more are on the plane's `+` side. So a
/// point exactly in the plane counts as `+`, and flipping the normal is not
/// always the same cut: a mesh face lying in the plane is cut with one sign
/// of the normal and not the other (see [`slice`](crate::slice())).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    origin: [f64; 3],
    normal: [f64; 3],
    u: [f64; 3],
    v: [f64; 3],
}

/// How far from orthonormal the axes given to [`Plane::with_axes`] may be.
const AXES_TOLERANCE: f64 = 1e-6;

impl Plane {
    /// The plane through `origin` with normal `normal`, which is normalised.
    ///
    /// The frame is the default one: `u` is the world axis least parallel to
    /// the normal (the first such axis on ties), made orthogonal to it and
    /// normalised, and `v = normal × u`. It changes only when the normal
    /// changes which axis that is.
    ///
    /// # Errors
    /// [`QueryError::InvalidInput`] when `origin` is not finite, or `normal`
    /// is zero or not finite.
    pub fn new(origin: [f64; 3], normal: [f64; 3]) -> Result<Plane> {
        check_finite(origin, "plane origin must be finite")?;
        check_finite(normal, "plane normal must be finite")?;
        // Scale by the largest component first, so that huge normals do not
        // overflow and tiny ones do not underflow.
        let big = normal.iter().fold(0.0f64, |m, c| m.max(c.abs()));
        if big == 0.0 {
            return Err(QueryError::InvalidInput("plane normal must not be zero"));
        }
        let n = normalize(normal.map(|c| c / big));
        let k = (0..3)
            .min_by(|&i, &j| n[i].abs().total_cmp(&n[j].abs()))
            .unwrap_or(0);
        let mut axis = [0.0; 3];
        axis[k] = 1.0;
        let u = normalize(std::array::from_fn(|i| axis[i] - n[k] * n[i]));
        let v = cross(n, u);
        Ok(Plane {
            origin,
            normal: n,
            u,
            v,
        })
    }

    /// The plane through `origin` spanned by the frame `(u, v)`, with normal
    /// `u × v` (normalised). The axes are kept as given.
    ///
    /// # Errors
    /// [`QueryError::InvalidInput`] when an argument is not finite, or the axes
    /// are not orthonormal within `1e-6` (unit length and orthogonal).
    pub fn with_axes(origin: [f64; 3], u: [f64; 3], v: [f64; 3]) -> Result<Plane> {
        check_finite(origin, "plane origin must be finite")?;
        check_finite(u, "plane axes must be finite")?;
        check_finite(v, "plane axes must be finite")?;
        let unit = |a: [f64; 3]| (dot(a, a).sqrt() - 1.0).abs() <= AXES_TOLERANCE;
        if !(unit(u) && unit(v) && dot(u, v).abs() <= AXES_TOLERANCE) {
            return Err(QueryError::InvalidInput("plane axes must be orthonormal"));
        }
        Ok(Plane {
            origin,
            normal: normalize(cross(u, v)),
            u,
            v,
        })
    }

    /// The point the plane was given through, origin of the 2D frame.
    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }

    /// The unit normal, pointing to the `+` side.
    pub fn normal(&self) -> [f64; 3] {
        self.normal
    }

    /// The first axis of the 2D frame.
    pub fn u(&self) -> [f64; 3] {
        self.u
    }

    /// The second axis of the 2D frame.
    pub fn v(&self) -> [f64; 3] {
        self.v
    }

    /// The signed distance `n·p − n·origin` of `p` from the plane: positive on
    /// the side the normal points to.
    pub fn signed_distance(&self, p: [f64; 3]) -> f64 {
        dot(self.normal, p) - dot(self.normal, self.origin)
    }

    /// The 2D coordinates `((p − origin)·u, (p − origin)·v)` of `p`, which is
    /// projected onto the plane along the normal.
    pub fn to_uv(&self, p: [f64; 3]) -> [f64; 2] {
        let d = sub(p, self.origin);
        [dot(d, self.u), dot(d, self.v)]
    }

    /// The point of the plane with 2D coordinates `q`: `origin + q₀·u + q₁·v`.
    pub fn from_uv(&self, q: [f64; 2]) -> [f64; 3] {
        std::array::from_fn(|k| self.origin[k] + q[0] * self.u[k] + q[1] * self.v[k])
    }
}

fn check_finite(a: [f64; 3], what: &'static str) -> Result<()> {
    if a.iter().all(|c| c.is_finite()) {
        Ok(())
    } else {
        Err(QueryError::InvalidInput(what))
    }
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let len = dot(a, a).sqrt();
    a.map(|c| c / len)
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{Lcg, random_normal};

    fn random_vec(rng: &mut Lcg, scale: f64) -> [f64; 3] {
        std::array::from_fn(|_| (f64::from(rng.next()) * 2.0 - 1.0) * scale)
    }

    fn norm(a: [f64; 3]) -> f64 {
        dot(a, a).sqrt()
    }

    #[test]
    fn plane_new_normalises_and_rejects_zero() {
        let p = Plane::new([0.0; 3], [0.0, 0.0, 2.0]).unwrap();
        assert_eq!(p.normal(), [0.0, 0.0, 1.0]);
        for bad in [[0.0; 3], [f64::NAN, 0.0, 1.0], [f64::INFINITY, 0.0, 0.0]] {
            assert!(matches!(
                Plane::new([0.0; 3], bad),
                Err(QueryError::InvalidInput(_))
            ));
        }
        assert!(matches!(
            Plane::new([f64::NAN, 0.0, 0.0], [0.0, 0.0, 1.0]),
            Err(QueryError::InvalidInput(_))
        ));
        // A huge but finite normal does not overflow to an invalid one.
        let p = Plane::new([0.0; 3], [1e300, 1e300, 0.0]).unwrap();
        assert!((norm(p.normal()) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn default_frame_is_orthonormal_right_handed() {
        let mut rng = Lcg(0x91a7);
        let mut normals: Vec<[f64; 3]> = (0..1000).map(|_| random_normal(&mut rng)).collect();
        // The axes themselves and the ties between them.
        normals.extend([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]);
        for n in normals {
            let p = Plane::new([0.0; 3], n).unwrap();
            let (n, u, v) = (p.normal(), p.u(), p.v());
            assert!((norm(u) - 1.0).abs() < 1e-12, "{n:?}");
            assert!((norm(v) - 1.0).abs() < 1e-12, "{n:?}");
            assert!(dot(u, n).abs() < 1e-12, "{n:?}");
            assert!(dot(v, n).abs() < 1e-12, "{n:?}");
            assert!(dot(u, v).abs() < 1e-12, "{n:?}");
            let w = cross(u, v);
            assert!((0..3).all(|k| (w[k] - n[k]).abs() < 1e-12), "{n:?}");
        }
    }

    #[test]
    fn with_axes_rejects_non_orthonormal() {
        let o = [1.0, 2.0, 3.0];
        assert!(matches!(
            Plane::with_axes(o, [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]),
            Err(QueryError::InvalidInput(_))
        ));
        assert!(matches!(
            Plane::with_axes(o, [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            Err(QueryError::InvalidInput(_))
        ));
        assert!(matches!(
            Plane::with_axes(o, [f64::NAN, 0.0, 0.0], [0.0, 1.0, 0.0]),
            Err(QueryError::InvalidInput(_))
        ));
        let p = Plane::with_axes(o, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        assert_eq!(p.normal(), [1.0, 0.0, 0.0]);
        assert_eq!(
            (p.origin(), p.u(), p.v()),
            (o, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        );
        // Within 1e-6 of orthonormal is accepted.
        assert!(Plane::with_axes(o, [1.0 + 5e-7, 0.0, 0.0], [5e-7, 1.0, 0.0]).is_ok());
    }

    #[test]
    fn uv_round_trip() {
        let mut rng = Lcg(0x3c0f);
        for _ in 0..1000 {
            let o = random_vec(&mut rng, 1.0);
            let plane = Plane::new(o, random_normal(&mut rng)).unwrap();
            let p = random_vec(&mut rng, 100.0);
            let s = plane.signed_distance(p);
            let n = plane.normal();
            let projected: [f64; 3] = std::array::from_fn(|k| p[k] - s * n[k]);
            let q = plane.from_uv(plane.to_uv(p));
            let tol = 1e-12 * norm(p);
            assert!((0..3).all(|k| (q[k] - projected[k]).abs() <= tol), "{p:?}");
        }
    }

    #[test]
    fn signed_distance_uses_the_side_rule() {
        let plane = Plane::new([0.0, 0.0, 1.0], [0.0, 0.0, 3.0]).unwrap();
        assert_eq!(plane.signed_distance([5.0, -2.0, 1.0]), 0.0);
        assert_eq!(plane.signed_distance([0.0, 0.0, 3.5]), 2.5);
        assert_eq!(plane.signed_distance([0.0, 0.0, 0.0]), -1.0);
    }
}
