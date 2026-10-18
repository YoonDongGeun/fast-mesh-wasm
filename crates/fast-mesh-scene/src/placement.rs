//! Placements: where a mesh sits in the world.

use fast_mesh_query::{QueryError, Rigid};

use crate::{Result, SceneError};

/// A rigid transform from a mesh's points to the world:
/// `world = rotation * p + translation`, where `p` is the mesh's own point
/// (its origin plus a local position), all in `f64`.
///
/// `rotation` is row-major and must be a proper rotation (orthonormal,
/// determinant +1, within `1e-6` per entry, the same rule as
/// [`Rigid::validate`]). Placements are stored exactly as given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    /// The rotation, row-major.
    pub rotation: [[f64; 3]; 3],
    /// The translation, applied after the rotation.
    pub translation: [f64; 3],
}

impl Placement {
    /// The placement that leaves the mesh where its coordinates say.
    pub const IDENTITY: Placement = Placement {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
    };

    /// Checks that every value is finite and the rotation is a proper rotation.
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] otherwise.
    pub fn validate(&self) -> Result<()> {
        let rigid = Rigid {
            rotation: self.rotation,
            translation: self.translation,
        };
        rigid.validate().map_err(|e| match e {
            QueryError::InvalidInput(m) => SceneError::InvalidInput(m),
            other => SceneError::Query(other),
        })
    }

    /// Maps a point of the mesh to the world: `rotation * p + translation`.
    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let (r, t) = (&self.rotation, &self.translation);
        // Mirrors `Rigid::apply_f64` and must stay bit-identical to it.
        std::array::from_fn(|i| r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2] + t[i])
    }

    /// Maps a world point back to the mesh: `rotationᵀ * (q - translation)`.
    pub fn inverse_apply(&self, q: [f64; 3]) -> [f64; 3] {
        let t = &self.translation;
        let d = [q[0] - t[0], q[1] - t[1], q[2] - t[2]];
        let r = &self.rotation;
        std::array::from_fn(|i| r[0][i] * d[0] + r[1][i] * d[1] + r[2][i] * d[2])
    }
}

impl Default for Placement {
    fn default() -> Self {
        Placement::IDENTITY
    }
}

/// A mesh-local point to the world: `R (o + p) + t`.
pub(crate) fn local_to_world(p: &Placement, origin: [f64; 3], local: [f64; 3]) -> [f64; 3] {
    p.apply([
        origin[0] + local[0],
        origin[1] + local[1],
        origin[2] + local[2],
    ])
}

/// A world point to mesh-local coordinates: `Rᵀ (q - t) - o`.
pub(crate) fn world_to_local(p: &Placement, origin: [f64; 3], q: [f64; 3]) -> [f64; 3] {
    let w = p.inverse_apply(q);
    [w[0] - origin[0], w[1] - origin[1], w[2] - origin[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
}

/// Gram-Schmidt on the rows, so the rotation is orthonormal to f64 rounding.
fn orthonormalize(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let r0 = unit(m[0]);
    let d = dot(m[1], r0);
    let r1 = unit([
        m[1][0] - d * r0[0],
        m[1][1] - d * r0[1],
        m[1][2] - d * r0[2],
    ]);
    // Third row from the cross product: keeps the determinant at +1.
    let r2 = [
        r0[1] * r1[2] - r0[2] * r1[1],
        r0[2] * r1[0] - r0[0] * r1[2],
        r0[0] * r1[1] - r0[1] * r1[0],
    ];
    [r0, r1, r2]
}

/// The transform from mesh B's local space to mesh A's local space, origins
/// included: `R = R_Aᵀ R_B`, `t = R_Aᵀ (R_B o_B + t_B - t_A) - o_A`, in `f64`.
/// `R` is re-orthonormalised (within about 1e-6 of the exact product) so that
/// `Rigid::validate` always accepts it.
pub(crate) fn relative_rigid(
    a: &Placement,
    origin_a: [f64; 3],
    b: &Placement,
    origin_b: [f64; 3],
) -> Rigid {
    let (ra, rb) = (&a.rotation, &b.rotation);
    let rotation: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| ra[0][i] * rb[0][j] + ra[1][i] * rb[1][j] + ra[2][i] * rb[2][j])
    });
    // B's origin in the world, relative to A's translation, then into A's frame.
    let wb = b.apply(origin_b);
    let w = [
        wb[0] - a.translation[0],
        wb[1] - a.translation[1],
        wb[2] - a.translation[2],
    ];
    let translation =
        std::array::from_fn(|i| ra[0][i] * w[0] + ra[1][i] * w[1] + ra[2][i] * w[2] - origin_a[i]);
    Rigid {
        rotation: orthonormalize(rotation),
        translation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RZ90: [[f64; 3]; 3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    const RX90: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];

    fn p(rotation: [[f64; 3]; 3], translation: [f64; 3]) -> Placement {
        Placement {
            rotation,
            translation,
        }
    }

    #[test]
    fn placement_validation() {
        assert!(Placement::IDENTITY.validate().is_ok());
        assert!(p(RZ90, [1.0, 2.0, 3.0]).validate().is_ok());
        let mut scaled = Placement::IDENTITY;
        scaled.rotation[0][0] = 2.0;
        assert!(matches!(
            scaled.validate(),
            Err(SceneError::InvalidInput(_))
        ));
        let mut mirror = Placement::IDENTITY;
        mirror.rotation[2][2] = -1.0;
        assert!(matches!(
            mirror.validate(),
            Err(SceneError::InvalidInput(_))
        ));
        let mut nan = Placement::IDENTITY;
        nan.translation[1] = f64::NAN;
        assert!(matches!(nan.validate(), Err(SceneError::InvalidInput(_))));
        let mut inf = Placement::IDENTITY;
        inf.rotation[0][1] = f64::INFINITY;
        assert!(matches!(inf.validate(), Err(SceneError::InvalidInput(_))));
    }

    #[test]
    fn relative_rigid_is_exact_for_integer_placements() {
        let a = p(RZ90, [10.0, -4.0, 7.0]);
        let b = p(RX90, [-3.0, 8.0, 2.0]);
        let oa = [1_000_000.0, 2_000_000.0, -1_500_000.0];
        let ob = [-1_000_000.0, 1_000_001.0, 2_500_000.0];
        let rel = relative_rigid(&a, oa, &b, ob);
        for local in [[1.0f32, 2.0, 3.0], [-5.0, 0.0, 4.0], [0.5, 0.25, -8.0]] {
            let got = rel.apply(local);
            let l = local.map(f64::from);
            let world = b.apply([ob[0] + l[0], ob[1] + l[1], ob[2] + l[2]]);
            // Hand-computed Rz90ᵀ (world - t_A) - o_A with Rz90ᵀ (x, y, z) = (y, -x, z).
            let d = [world[0] - 10.0, world[1] + 4.0, world[2] - 7.0];
            let want = [d[1] - oa[0], -d[0] - oa[1], d[2] - oa[2]];
            assert_eq!(got, want);
        }
    }

    #[test]
    fn relative_rotation_of_near_orthonormal_inputs_validates() {
        // Shears: each input is within tolerance, but the product's columns
        // are 2e off orthogonal, so only re-orthonormalising makes it valid.
        let e = 0.9e-6;
        let mut a = p(RZ90, [0.0; 3]);
        a.rotation[0][0] = e;
        let mut b = p(RZ90, [0.0; 3]);
        b.rotation[0][0] = e;
        assert!(a.validate().is_ok() && b.validate().is_ok());
        let (ra, rb) = (&a.rotation, &b.rotation);
        let raw: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| ra[0][i] * rb[0][j] + ra[1][i] * rb[1][j] + ra[2][i] * rb[2][j])
        });
        let raw_rigid = Rigid {
            rotation: raw,
            translation: [0.0; 3],
        };
        assert!(raw_rigid.validate().is_err(), "raw product must be invalid");
        let rel = relative_rigid(&a, [0.0; 3], &b, [0.0; 3]);
        assert_eq!(rel.validate(), Ok(()));
        for (row, raw_row) in rel.rotation.iter().zip(&raw) {
            for (x, y) in row.iter().zip(raw_row) {
                assert!((x - y).abs() < 2e-6, "{x} vs {y}");
            }
        }
    }

    #[test]
    fn world_local_round_trip() {
        let pl = p(RX90, [3.0, -2.0, 5.0]);
        let origin = [100.0, -50.0, 25.0];
        let local = [0.25, -1.5, 3.0];
        let world = local_to_world(&pl, origin, local);
        let back = world_to_local(&pl, origin, world);
        for (b, l) in back.iter().zip(local) {
            assert!((b - l).abs() < 1e-12);
        }
    }
}
