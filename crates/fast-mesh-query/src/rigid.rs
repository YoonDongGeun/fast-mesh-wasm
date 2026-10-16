//! Rigid transforms (rotation plus translation) between two meshes' local
//! coordinates.

use fast_mesh_bvh::Aabb;

use crate::{QueryError, Result};

/// How far the rotation may be from orthonormal with determinant +1.
const TOLERANCE: f64 = 1e-6;

/// Maps mesh B's local coordinates into mesh A's local coordinates:
/// `p_a = rotation * p_b + translation`.
///
/// `rotation` is row-major (`rotation[i][j]` is row `i`, column `j`) and must
/// be a proper rotation: orthonormal with determinant +1, within `1e-6` per
/// entry. Scale and reflection are not supported, so distances measured in A's
/// space are distances between the meshes. Queries given anything else, or
/// non-finite values, fail with [`QueryError::InvalidInput`].
///
/// B's vertices are transformed in `f64` when they are tested and never stored.
/// "Exact" intersection results are exact for those transformed `f64`
/// coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rigid {
    /// The rotation, row-major.
    pub rotation: [[f64; 3]; 3],
    /// The translation, applied after the rotation.
    pub translation: [f64; 3],
}

impl Rigid {
    /// The transform that leaves every point where it is.
    pub const IDENTITY: Rigid = Rigid {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
    };

    /// The transform that maps A's coordinates back into B's: the transposed
    /// rotation and the translation `-rotationᵀ * translation`.
    pub fn inverse(&self) -> Rigid {
        let r = &self.rotation;
        let rotation: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| r[j][i]));
        let t = self.translation;
        let translation = std::array::from_fn(|i| {
            -(rotation[i][0] * t[0] + rotation[i][1] * t[1] + rotation[i][2] * t[2])
        });
        Rigid {
            rotation,
            translation,
        }
    }

    /// Checks that every value is finite and that the rotation is a proper
    /// rotation (orthonormal with determinant +1) within `1e-6` per entry: the
    /// check every mesh-vs-mesh query runs first.
    ///
    /// # Errors
    /// [`QueryError::InvalidInput`] for a non-finite value, a rotation that is
    /// not orthonormal, or a reflection.
    pub fn validate(&self) -> Result<()> {
        let r = &self.rotation;
        if !r
            .iter()
            .flatten()
            .chain(&self.translation)
            .all(|v| v.is_finite())
        {
            return Err(QueryError::InvalidInput("transform has a non-finite value"));
        }
        for i in 0..3 {
            for j in 0..3 {
                let dot = r[0][i] * r[0][j] + r[1][i] * r[1][j] + r[2][i] * r[2][j];
                let want = if i == j { 1.0 } else { 0.0 };
                if (dot - want).abs() > TOLERANCE {
                    return Err(QueryError::InvalidInput(
                        "transform rotation is not orthonormal",
                    ));
                }
            }
        }
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        if (det - 1.0).abs() > TOLERANCE {
            return Err(QueryError::InvalidInput(
                "transform rotation is a reflection",
            ));
        }
        Ok(())
    }

    /// Maps a point of B into A's space: `rotation * p + translation`, in
    /// `f64`.
    ///
    /// This is the exact arithmetic (same expression, same order) the queries
    /// use to move B's vertices, so points transformed with it agree bit for
    /// bit with what the queries tested. To map a result point such as
    /// [`MeshDistance::point_b`](crate::MeshDistance::point_b) back into B's
    /// space, use `apply` on [`Rigid::inverse`].
    ///
    /// ```
    /// use fast_mesh_query::Rigid;
    ///
    /// // A quarter turn about z, then a step along x.
    /// let xf = Rigid {
    ///     rotation: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    ///     translation: [1.0, 0.0, 0.0],
    /// };
    /// assert_eq!(xf.validate(), Ok(()));
    /// let in_a = xf.apply([1.0, 0.0, 0.0]);
    /// assert_eq!(in_a, [1.0, 1.0, 0.0]);
    /// let back = xf.inverse().apply(in_a.map(|v| v as f32));
    /// assert_eq!(back, [1.0, 0.0, 0.0]);
    /// ```
    pub fn apply(&self, p: [f32; 3]) -> [f64; 3] {
        self.apply_f64(p.map(f64::from))
    }

    /// [`Rigid::apply`] for an `f64` point. `transform_box` relies on this
    /// exact expression and evaluation order.
    pub(crate) fn apply_f64(&self, p: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        let t = self.translation;
        std::array::from_fn(|i| r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2] + t[i])
    }

    /// A box in A's space holding every point of `b` (a box in B's space) as
    /// mapped by [`Rigid::apply`], rounded outward to `f32`. An empty box stays
    /// empty.
    ///
    /// Each output bound is `apply` evaluated at the corner of `b` that
    /// minimizes (or maximizes) that coordinate: the corner picks `min` or `max`
    /// on each axis by the sign of the rotation entry. Because rounding is
    /// monotone, `apply` of any point in `b` lands inside these bounds as
    /// computed, not just in real arithmetic. The result is the exact image box
    /// `center ± |rotation| * half_extent` up to rounding.
    pub(crate) fn transform_box(&self, b: &Aabb) -> Aabb {
        if b.is_empty() {
            return Aabb::EMPTY;
        }
        let lo = b.min.map(f64::from);
        let hi = b.max.map(f64::from);
        let r = &self.rotation;
        let t = self.translation;
        let mut out = Aabb::EMPTY;
        for i in 0..3 {
            let low = |j: usize| if r[i][j] >= 0.0 { lo[j] } else { hi[j] };
            let high = |j: usize| if r[i][j] >= 0.0 { hi[j] } else { lo[j] };
            let min = r[i][0] * low(0) + r[i][1] * low(1) + r[i][2] * low(2) + t[i];
            let max = r[i][0] * high(0) + r[i][1] * high(1) + r[i][2] * high(2) + t[i];
            out.min[i] = round_down(min);
            out.max[i] = round_up(max);
        }
        out
    }
}

/// The largest `f32` at or below `x`.
fn round_down(x: f64) -> f32 {
    let f = x as f32;
    if f64::from(f) > x { f.next_down() } else { f }
}

/// The smallest `f32` at or above `x`.
fn round_up(x: f64) -> f32 {
    let f = x as f32;
    if f64::from(f) < x { f.next_up() } else { f }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{Lcg, random_rotation};

    fn rot_z(deg: f64) -> [[f64; 3]; 3] {
        let (s, c) = deg.to_radians().sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    fn rigid(rotation: [[f64; 3]; 3], translation: [f64; 3]) -> Rigid {
        Rigid {
            rotation,
            translation,
        }
    }

    fn is_invalid(r: Rigid) -> bool {
        matches!(r.validate(), Err(QueryError::InvalidInput(_)))
    }

    #[test]
    fn rigid_validation() {
        assert_eq!(Rigid::IDENTITY.validate(), Ok(()));
        let r30 = rot_z(30.0);
        assert_eq!(rigid(r30, [1.0, 2.0, 3.0]).validate(), Ok(()));
        let r30_f32 = r30.map(|row| row.map(|v| f64::from(v as f32)));
        assert_ne!(r30_f32, r30);
        assert_eq!(rigid(r30_f32, [0.0; 3]).validate(), Ok(()));

        let mut off = r30;
        off[0][0] += 1e-3;
        assert!(is_invalid(rigid(off, [0.0; 3])));
        let reflection = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];
        assert!(is_invalid(rigid(reflection, [0.0; 3])));
        assert!(is_invalid(rigid(r30, [0.0, f64::NAN, 0.0])));
        let mut inf = r30;
        inf[2][1] = f64::INFINITY;
        assert!(is_invalid(rigid(inf, [0.0; 3])));
    }

    #[test]
    fn inverse_round_trips() {
        let r = rigid(rot_z(30.0), [1.5, -2.0, 0.25]);
        let inv = r.inverse();
        assert_eq!(inv.validate(), Ok(()));
        for p in [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-7.5, 0.125, 4.0]] {
            let back = inv.apply_f64(r.apply(p));
            for k in 0..3 {
                assert!(
                    (back[k] - f64::from(p[k])).abs() <= 1e-12,
                    "{back:?} vs {p:?}"
                );
            }
        }
        // `inverse` of a point stored as f32 round-trips through `apply` too.
        let p = [0.5f32, -1.0, 2.0];
        let q = inv.apply(p);
        let q32 = q.map(|v| v as f32);
        let back = r.apply(q32);
        for k in 0..3 {
            assert!((back[k] - f64::from(p[k])).abs() <= 1e-6);
        }
    }

    /// Asserts that `apply` puts every corner of `b`, and each of `extra`, inside `t`.
    fn assert_inside(r: &Rigid, b: &Aabb, t: &Aabb, extra: &[[f32; 3]]) {
        let corners = (0..8).map(|c: u32| -> [f32; 3] {
            std::array::from_fn(|k| if c >> k & 1 == 0 { b.min[k] } else { b.max[k] })
        });
        for p in corners.chain(extra.iter().copied()) {
            let q = r.apply(p);
            let inside = q
                .iter()
                .zip(t.min.iter().zip(&t.max))
                .all(|(&v, (&lo, &hi))| f64::from(lo) <= v && v <= f64::from(hi));
            assert!(inside, "{p:?} -> {q:?} outside {t:?}");
        }
    }

    #[test]
    fn transform_box_is_conservative() {
        let mut rng = Lcg(42);
        for _ in 0..1000 {
            let r = rigid(
                random_rotation(&mut rng),
                std::array::from_fn(|_| f64::from(rng.next()) * 200.0 - 100.0),
            );
            assert_eq!(r.validate(), Ok(()));
            let lo: [f32; 3] = std::array::from_fn(|_| rng.next() * 20.0 - 10.0);
            let size: [f32; 3] = std::array::from_fn(|_| rng.next() * 5.0);
            let b = Aabb::new(lo, std::array::from_fn(|k| lo[k] + size[k]));
            let inner: Vec<[f32; 3]> = (0..4)
                .map(|_| std::array::from_fn(|k| b.min[k] + rng.next() * size[k]))
                .filter(|p: &[f32; 3]| (0..3).all(|k| p[k] <= b.max[k]))
                .collect();
            assert_inside(&r, &b, &r.transform_box(&b), &inner);
            assert_eq!(Rigid::IDENTITY.transform_box(&b), b);
        }
        assert!(Rigid::IDENTITY.transform_box(&Aabb::EMPTY).is_empty());
        let r = rigid(rot_z(30.0), [1.0, 2.0, 3.0]);
        assert!(r.transform_box(&Aabb::EMPTY).is_empty());
    }

    #[test]
    fn transform_box_rounds_outward() {
        // A 90 degree turn about z built from f64 sin/cos: entries like 6e-17
        // instead of 0 move the corners off the f32 grid, so the box must widen.
        let r = rigid(rot_z(90.0), [0.1, 0.0, 0.0]);
        let b = Aabb::new([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);
        let t = r.transform_box(&b);
        assert_inside(&r, &b, &t, &[]);
        // Still tight: within an f32 ulp or so of the exact box.
        assert!((t.min[0] - (0.1 - 2.0)).abs() < 1e-5);
        assert!((t.max[1] - 2.0).abs() < 1e-5);
    }
}
