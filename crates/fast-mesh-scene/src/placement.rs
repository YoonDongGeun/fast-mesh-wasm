//! Placements: where a mesh sits in the world.
//!
//! A placement maps a mesh's points to the world as
//! `world = rotation * (scale * p) + translation`, where `p` is the mesh's
//! own point (its origin plus a local position), all in `f64`. The scale is
//! applied first, per axis, then the rotation, then the translation.
//!
//! The default scale `[1, 1, 1]` is the rigid placement the scene used
//! before scales existed; every snapshot and default keeps working.

use fast_mesh_query::{QueryError, Rigid};

use crate::{Result, SceneError};

/// A transform from a mesh's points to the world:
/// `world = rotation * (scale * p) + translation`, where `p` is the mesh's
/// own point (its origin plus a local position), all in `f64`.
///
/// `rotation` is row-major and must be a proper rotation (orthonormal,
/// determinant +1, within `1e-6` per entry, the same rule as
/// [`Rigid::validate`]). `scale` is per axis, applied before the rotation,
/// and must be finite and positive: zero would collapse the mesh (no
/// inverse) and a negative one would mirror it (flipping winding, culling
/// and signs). Placements are stored exactly as given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    /// The rotation, row-major, applied after the scale.
    pub rotation: [[f64; 3]; 3],
    /// The translation, applied after the rotation.
    pub translation: [f64; 3],
    /// The per-axis scale, applied first. `[1, 1, 1]` leaves the mesh rigid.
    pub scale: [f64; 3],
}

impl Placement {
    /// The placement that leaves the mesh where its coordinates say.
    pub const IDENTITY: Placement = Placement {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
        scale: [1.0; 3],
    };

    /// Checks that the rotation is a proper rotation, and that the
    /// translation and the scale are finite with every scale component
    /// positive.
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
        })?;
        if !self.scale.iter().all(|s| s.is_finite()) {
            return Err(SceneError::InvalidInput("placement scale must be finite"));
        }
        if !self.scale.iter().all(|&s| s > 0.0) {
            return Err(SceneError::InvalidInput("placement scale must be positive"));
        }
        Ok(())
    }

    /// Whether the placement scales: any component differs from 1.
    pub fn is_scaled(&self) -> bool {
        self.scale != [1.0; 3]
    }

    /// The uniform scale when every component is equal, `None` otherwise.
    pub fn uniform_scale(&self) -> Option<f64> {
        (self.scale[0] == self.scale[1] && self.scale[1] == self.scale[2]).then_some(self.scale[0])
    }

    /// The largest scale component.
    pub(crate) fn max_scale(&self) -> f64 {
        self.scale[0].max(self.scale[1]).max(self.scale[2])
    }

    /// Maps a point of the mesh to the world: `rotation * (scale * p) +
    /// translation`.
    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let (r, t, s) = (&self.rotation, &self.translation, &self.scale);
        // Mirrors `Rigid::apply_f64` with the scale folded into the point,
        // so an unscaled placement stays bit-identical to it (`1.0 * p ==
        // p` exactly).
        std::array::from_fn(|i| {
            r[i][0] * (s[0] * p[0]) + r[i][1] * (s[1] * p[1]) + r[i][2] * (s[2] * p[2]) + t[i]
        })
    }

    /// Maps a world point back to the mesh: `(scale⁻¹ * rotationᵀ) * (q -
    /// translation)`.
    pub fn inverse_apply(&self, q: [f64; 3]) -> [f64; 3] {
        let t = &self.translation;
        let d = [q[0] - t[0], q[1] - t[1], q[2] - t[2]];
        let r = &self.rotation;
        let s = &self.scale;
        std::array::from_fn(|i| (r[0][i] * d[0] + r[1][i] * d[1] + r[2][i] * d[2]) / s[i])
    }

    /// Maps a world direction to the mesh's local frame: `scale⁻¹ *
    /// rotationᵀ * d`. A ray mapped this way keeps its parameter: with
    /// `o_l` the mapped origin, `apply(o_l + t * d_l)` is the world point at
    /// `t`, since the placement is affine.
    pub(crate) fn inverse_direction(&self, d: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        let s = &self.scale;
        std::array::from_fn(|i| (r[0][i] * d[0] + r[1][i] * d[1] + r[2][i] * d[2]) / s[i])
    }

    /// Maps a mesh-local direction to the world: `rotation * (scale * d)`.
    /// Renderers use this alongside [`Placement::normal_to_world`] to pose
    /// mesh-local vectors.
    pub fn apply_direction(&self, d: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        let s = &self.scale;
        std::array::from_fn(|i| {
            r[i][0] * (s[0] * d[0]) + r[i][1] * (s[1] * d[1]) + r[i][2] * (s[2] * d[2])
        })
    }

    /// Maps a mesh-local unit normal to a world unit normal: `normalize(R *
    /// (n / S))`, the inverse-transpose for a rotation with a diagonal
    /// scale. Sides are preserved (the scale is positive), so culling and
    /// signs follow. Renderers use this to pose mesh-local normals.
    pub fn normal_to_world(&self, n: [f64; 3]) -> [f64; 3] {
        unit(std::array::from_fn(|i| {
            let r = &self.rotation;
            let s = &self.scale;
            r[i][0] * (n[0] / s[0]) + r[i][1] * (n[1] / s[1]) + r[i][2] * (n[2] / s[2])
        }))
    }

    /// Maps a world unit normal's direction to the mesh's local frame:
    /// `normalize(S * Rᵀ * n)`. The zero set of the local plane through
    /// `inverse_apply` of a world point with this normal is the preimage of
    /// the world plane's zero set, with matching sides.
    pub fn normal_to_local(&self, n: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        let s = &self.scale;
        unit(std::array::from_fn(|i| {
            s[i] * (r[0][i] * n[0] + r[1][i] * n[1] + r[2][i] * n[2])
        }))
    }
}

impl Default for Placement {
    fn default() -> Self {
        Placement::IDENTITY
    }
}

/// A mesh-local point to the world: `R (S (o + p)) + t`.
pub(crate) fn local_to_world(p: &Placement, origin: [f64; 3], local: [f64; 3]) -> [f64; 3] {
    p.apply([
        origin[0] + local[0],
        origin[1] + local[1],
        origin[2] + local[2],
    ])
}

/// A world point to mesh-local coordinates: `S⁻¹ Rᵀ (q - t) - o`.
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

/// An affine map between two meshes' local spaces: `matrix * p +
/// translation`, origins included. [`relative_affine`] builds the one from
/// mesh B's local space to mesh A's. With unscaled placements it is the same
/// map as [`relative_rigid`], bit for bit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Affine {
    pub(crate) matrix: [[f64; 3]; 3],
    pub(crate) translation: [f64; 3],
}

impl Affine {
    /// Maps a point of B's local space into A's.
    pub(crate) fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let (m, t) = (&self.matrix, &self.translation);
        std::array::from_fn(|i| m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + t[i])
    }
}

/// The transform from mesh B's local space to mesh A's local space, origins
/// included, with scales: `M_A⁻¹ M_B`, where each `M` is `R (S (o + ·)) +
/// t`, in `f64`.
pub(crate) fn relative_affine(
    a: &Placement,
    origin_a: [f64; 3],
    b: &Placement,
    origin_b: [f64; 3],
) -> Affine {
    // B's origin in the world, relative to A's translation, then into A's
    // scaled frame: `S_A⁻¹ R_Aᵀ (M_B(o_B) - t_A) - o_A`.
    let wb = b.apply(origin_b);
    let w = [
        wb[0] - a.translation[0],
        wb[1] - a.translation[1],
        wb[2] - a.translation[2],
    ];
    let ra = &a.rotation;
    let unscaled: [f64; 3] =
        std::array::from_fn(|i| ra[0][i] * w[0] + ra[1][i] * w[1] + ra[2][i] * w[2]);
    let translation = std::array::from_fn(|i| unscaled[i] / a.scale[i] - origin_a[i]);
    // The linear part: `S_A⁻¹ R_Aᵀ R_B S_B`.
    let (rb, sa, sb) = (&b.rotation, &a.scale, &b.scale);
    let matrix: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (ra[0][i] * rb[0][j] + ra[1][i] * rb[1][j] + ra[2][i] * rb[2][j]) * sb[j] / sa[i]
        })
    });
    Affine {
        matrix,
        translation,
    }
}

/// Whether `relative_affine(a, oa, b, ob)` is a rigid map (both placements
/// unscaled): then [`relative_rigid`] is that map, re-orthonormalised.
pub(crate) fn is_relative_rigid(a: &Placement, b: &Placement) -> bool {
    !a.is_scaled() && !b.is_scaled()
}

/// The transform from mesh B's local space to mesh A's local space, origins
/// included: `R = R_Aᵀ R_B`, `t = R_Aᵀ (R_B o_B + t_B - t_A) - o_A`, in `f64`.
/// Only for unscaled placements (see [`is_relative_rigid`]); with scales the
/// map is affine (see [`relative_affine`]).
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
            scale: [1.0; 3],
        }
    }

    fn with_scale(translation: [f64; 3], scale: [f64; 3]) -> Placement {
        Placement {
            translation,
            scale,
            ..Placement::IDENTITY
        }
    }

    #[test]
    fn placement_validation() {
        assert!(Placement::IDENTITY.validate().is_ok());
        assert!(p(RZ90, [1.0, 2.0, 3.0]).validate().is_ok());
        assert!(with_scale([0.0; 3], [2.0, 0.5, 3.0]).validate().is_ok());
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
        // Scales must be finite and positive.
        for scale in [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -2.0, 1.0],
            [f64::NAN, 1.0, 1.0],
            [1.0, f64::INFINITY, 1.0],
        ] {
            assert!(
                matches!(
                    with_scale([0.0; 3], scale).validate(),
                    Err(SceneError::InvalidInput(_))
                ),
                "scale {scale:?}"
            );
        }
        assert!(Placement::IDENTITY.uniform_scale().is_some());
        assert_eq!(with_scale([0.0; 3], [2.0; 3]).uniform_scale(), Some(2.0));
        assert_eq!(with_scale([0.0; 3], [2.0, 1.0, 1.0]).uniform_scale(), None);
        assert!(!Placement::IDENTITY.is_scaled());
        assert!(with_scale([0.0; 3], [1.0, 1.0, 2.0]).is_scaled());
    }

    #[test]
    fn scaled_apply_round_trips() {
        let pl = Placement {
            rotation: RZ90,
            translation: [10.0, -4.0, 7.0],
            scale: [2.0, 0.5, 3.0],
        };
        assert!(pl.validate().is_ok());
        // world = R (S p) + t, checked by hand: S p = (2, -0.5, 9) for
        // p = (1, -1, 3); Rz90 (x, y, z) = (-y, x, z); plus t.
        assert_eq!(pl.apply([1.0, -1.0, 3.0]), [10.5, -2.0, 16.0]);
        for p in [[0.0; 3], [1.0, 2.0, -3.0], [-100.5, 0.25, 999.0]] {
            let back = pl.inverse_apply(pl.apply(p));
            for (b, x) in back.iter().zip(p) {
                assert!((b - x).abs() < 1e-12, "{back:?} vs {p:?}");
            }
        }
        // Directions map without the translation, keeping the ray
        // parameter: `apply(o_l + t d_l)` is the world point `apply(o) +
        // t d`.
        let o = [1.0, 2.0, 3.0];
        let d = unit([1.0, -2.0, 0.5]);
        let base = pl.apply(o);
        let ol = pl.inverse_apply(base);
        let dl = pl.inverse_direction(d);
        for t in [0.0, 0.5, 10.0] {
            let want: [f64; 3] = std::array::from_fn(|k| base[k] + t * d[k]);
            let got: [f64; 3] = pl.apply(std::array::from_fn(|k| ol[k] + t * dl[k]));
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() < 1e-9, "{got:?} vs {want:?}");
            }
        }
        // Unscaled placements behave exactly as before.
        let rigid = p(RZ90, [10.0, -4.0, 7.0]);
        assert_eq!(
            rigid.apply([1.0, 2.0, 3.0]),
            [-2.0 + 10.0, 1.0 - 4.0, 3.0 + 7.0]
        );
    }

    #[test]
    fn normals_follow_inverse_transpose() {
        // Non-uniform scale [2, 1, 1]: a local 45-degree normal in xy maps
        // to a world normal tilted towards x, still unit, on the same side.
        let pl = with_scale([5.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
        let n = unit([1.0, 1.0, 0.0]);
        let w = pl.normal_to_world(n);
        assert!((dot(w, w) - 1.0).abs() < 1e-15);
        // Inverse-transpose: R (n / S) is (0.5, 1, 0) normalised.
        let want = unit([0.5, 1.0, 0.0]);
        for (g, x) in w.iter().zip(want) {
            assert!((g - x).abs() < 1e-15, "{w:?} vs {want:?}");
        }
        // Round trip through the local frame.
        let back = pl.normal_to_local(w);
        for (b, x) in back.iter().zip(n) {
            assert!((b - x).abs() < 1e-15, "{back:?} vs {n:?}");
        }
        // The local plane with this normal is the preimage of the world
        // plane: sides match.
        let origin_w = pl.apply([0.0; 3]);
        let local_origin = pl.inverse_apply(origin_w);
        let nl = pl.normal_to_local([0.0, 0.0, 1.0]);
        for q in [[1.0, 0.0, 5.0], [0.0, -3.0, -2.0], [7.0, 7.0, 0.5]] {
            let l = pl.inverse_apply(q);
            let side_w = dot(q, [0.0, 0.0, 1.0]) - dot(origin_w, [0.0, 0.0, 1.0]);
            let side_l = dot(
                [
                    l[0] - local_origin[0],
                    l[1] - local_origin[1],
                    l[2] - local_origin[2],
                ],
                nl,
            );
            assert!(side_w * side_l > 0.0 || (side_w == 0.0 && side_l == 0.0));
        }
    }

    #[test]
    fn relative_affine_matches_apply() {
        // B's local points mapped by the affine land where B's placement
        // puts them relative to A: applying A's placement gives the world.
        let a = Placement {
            rotation: RZ90,
            translation: [10.0, -4.0, 7.0],
            scale: [2.0, 0.5, 1.0],
        };
        let b = Placement {
            rotation: RX90,
            translation: [-3.0, 8.0, 2.0],
            scale: [0.5, 3.0, 2.0],
        };
        let (oa, ob) = ([1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]);
        let rel = relative_affine(&a, oa, &b, ob);
        for local in [[0.0; 3], [1.0, -1.0, 0.5], [-100.0, 200.0, -300.0]] {
            // The map takes origin-exclusive local points, like
            // `relative_rigid`: `rel(local)` is where B's stored point
            // `local` sits in A's stored space.
            let in_a = rel.apply(local);
            let world_direct = b.apply([ob[0] + local[0], ob[1] + local[1], ob[2] + local[2]]);
            let world_via_a = a.apply([oa[0] + in_a[0], oa[1] + in_a[1], oa[2] + in_a[2]]);
            for (g, w) in world_via_a.iter().zip(world_direct) {
                assert!((g - w).abs() < 1e-9, "{world_via_a:?} vs {world_direct:?}");
            }
        }
        // Unscaled: the affine is the rigid map before orthonormalisation.
        let c = p(RZ90, [1.0, 0.0, 0.0]);
        let d = p(RX90, [0.0, 1.0, 0.0]);
        let aff = relative_affine(&c, [0.0; 3], &d, [0.0; 3]);
        let rig = relative_rigid(&c, [0.0; 3], &d, [0.0; 3]);
        for p in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 2.0, 3.0]] {
            let g = aff.apply(p.map(f64::from));
            let w = rig.apply(p.map(|x| x as f32));
            for (g, w) in g.iter().zip(w) {
                assert!((g - w).abs() < 1e-6, "{g:?} vs {w:?}");
            }
        }
        assert!(is_relative_rigid(&c, &d));
        assert!(!is_relative_rigid(&a, &b));
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
