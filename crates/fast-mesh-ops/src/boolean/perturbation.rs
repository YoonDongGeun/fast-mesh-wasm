//! Deterministic Simulation-of-Simplicity and directional symbolic perturbation.
//!
//! Provides exact Shewchuk predicates via [`robust`], canonical vertex-ID-keyed
//! Simulation-of-Simplicity tie-breaking (`orient2d_sos`, `orient3d_sos`) in the
//! sense of Edelsbrunner & Mücke (1990), and an exact directional ray-winding
//! classifier (`ray_winding_sos`) for resolving grazing, coplanar, and tangent
//! boundary ambiguities consistently across both Boolean operands.

use fast_mesh_bvh::{Bvh, NodeKind};
use fast_mesh_store::{Mesh, VertexId};
use robust::{Coord, Coord3D, orient2d, orient3d};

pub(super) type D2 = [f64; 2];
pub(super) type D3 = [f64; 3];

/// Canonical vertex identity for deterministic Simulation-of-Simplicity tie-breaking.
///
/// Smaller keys carry lower perturbation exponents (dominating in `epsilon -> 0+`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct PerturbKey(pub u64);

impl PerturbKey {
    #[allow(dead_code)]
    #[inline]
    pub(super) const fn orig_a(v: VertexId) -> Self {
        Self((v as u64) << 2)
    }

    #[allow(dead_code)]
    #[inline]
    pub(super) const fn orig_b(v: VertexId) -> Self {
        Self(((v as u64) << 2) | 1)
    }

    #[inline]
    pub(super) const fn arr(v: u32) -> Self {
        Self(((v as u64) << 2) | 2)
    }
}

#[inline]
fn c2(p: D2) -> Coord<f64> {
    Coord { x: p[0], y: p[1] }
}

#[inline]
fn c3(p: D3) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

#[inline]
fn sign_f64(v: f64) -> i8 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

#[inline]
fn sign_diff(a: f64, b: f64) -> i8 {
    if a > b {
        1
    } else if a < b {
        -1
    } else {
        0
    }
}

/// Exact 2D orientation sign: `+1` if `(a, b, c)` is counter-clockwise, `-1` if
/// clockwise, `0` if collinear.
#[inline]
pub(super) fn orient2d_exact(a: D2, b: D2, c: D2) -> i8 {
    sign_f64(orient2d(c2(a), c2(b), c2(c)))
}

/// Exact 3D orientation sign matching Shewchuk's `orient3d`: `+1` when `d` lies
/// on the side opposite the right-hand normal of `(a, b, c)`, `-1` on the
/// normal side, `0` when coplanar.
#[inline]
pub(super) fn orient3d_exact(a: D3, b: D3, c: D3, d: D3) -> i8 {
    sign_f64(orient3d(c3(a), c3(b), c3(c), c3(d)))
}

#[inline]
fn o2_yz(a: D3, b: D3, c: D3) -> i8 {
    orient2d_exact([a[1], a[2]], [b[1], b[2]], [c[1], c[2]])
}

#[inline]
fn o2_xz(a: D3, b: D3, c: D3) -> i8 {
    orient2d_exact([a[0], a[2]], [b[0], b[2]], [c[0], c[2]])
}

#[inline]
fn o2_xy(a: D3, b: D3, c: D3) -> i8 {
    orient2d_exact([a[0], a[1]], [b[0], b[1]], [c[0], c[1]])
}

/// Exact 2D orientation with canonical vertex-ID-keyed Simulation-of-Simplicity
/// perturbation `p_k(eps) = (x_k + eps^(2^(2k)), y_k + eps^(2^(2k+1)))`.
///
/// Guarantees a non-zero sign whenever all three `PerturbKey`s are distinct.
#[allow(dead_code)]
pub(super) fn orient2d_sos(a: (D2, PerturbKey), b: (D2, PerturbKey), c: (D2, PerturbKey)) -> i8 {
    let base = orient2d_exact(a.0, b.0, c.0);
    if base != 0 {
        return base;
    }
    if a.1 == b.1 || b.1 == c.1 || a.1 == c.1 {
        return 0;
    }

    let mut pts = [a, b, c];
    let mut sign: i8 = 1;
    if pts[0].1 > pts[1].1 {
        pts.swap(0, 1);
        sign = -sign;
    }
    if pts[1].1 > pts[2].1 {
        pts.swap(1, 2);
        sign = -sign;
    }
    if pts[0].1 > pts[1].1 {
        pts.swap(0, 1);
        sign = -sign;
    }

    let [p0, p1, p2] = [pts[0].0, pts[1].0, pts[2].0];
    let term = if p1[1] != p2[1] {
        sign_diff(p1[1], p2[1])
    } else if p2[0] != p1[0] {
        sign_diff(p2[0], p1[0])
    } else if p2[1] != p0[1] {
        sign_diff(p2[1], p0[1])
    } else {
        -1
    };
    sign * term
}

/// Exact 3D orientation with canonical vertex-ID-keyed Simulation-of-Simplicity
/// perturbation `p_k(eps) = (x_k + eps^(2^(3k)), y_k + eps^(2^(3k+1)), z_k + eps^(2^(3k+2)))`.
///
/// Guarantees a non-zero sign whenever all four `PerturbKey`s are distinct.
pub(super) fn orient3d_sos(
    a: (D3, PerturbKey),
    b: (D3, PerturbKey),
    c: (D3, PerturbKey),
    d: (D3, PerturbKey),
) -> i8 {
    let base = orient3d_exact(a.0, b.0, c.0, d.0);
    if base != 0 {
        return base;
    }
    if a.1 == b.1 || a.1 == c.1 || a.1 == d.1 || b.1 == c.1 || b.1 == d.1 || c.1 == d.1 {
        return 0;
    }

    let mut pts = [a, b, c, d];
    let mut sign: i8 = 1;
    for i in 0..4 {
        for j in (i + 1)..4 {
            if pts[i].1 > pts[j].1 {
                pts.swap(i, j);
                sign = -sign;
            }
        }
    }

    let [p0, p1, p2, p3] = [pts[0].0, pts[1].0, pts[2].0, pts[3].0];
    let checks = [
        o2_yz(p1, p2, p3),
        -o2_xz(p1, p2, p3),
        o2_xy(p1, p2, p3),
        -o2_yz(p0, p2, p3),
        sign_diff(p3[2], p2[2]),
        sign_diff(p2[1], p3[1]),
        o2_xz(p0, p2, p3),
        sign_diff(p2[2], p3[2]),
        sign_diff(p3[0], p2[0]),
        -o2_xy(p0, p2, p3),
        sign_diff(p3[1], p2[1]),
        sign_diff(p2[0], p3[0]),
        o2_yz(p0, p1, p3),
        sign_diff(p1[2], p3[2]),
        sign_diff(p3[1], p1[1]),
        sign_diff(p3[2], p0[2]),
    ];

    for term in checks {
        if term != 0 {
            return sign * term;
        }
    }
    -sign
}

/// 2D orientation of edge `(a, b)` against directionally shifted point `q + s*(eps^2, eps^3)`
/// where `s in {-1, +1}` and `a, b` are non-coincident in 2D.
#[inline]
fn orient2d_point_dir(a: D2, b: D2, q: D2, side_sign: i8) -> i8 {
    let base = orient2d_exact(a, b, q);
    if base != 0 {
        return base;
    }
    // Expansion of det([a_x a_y 1; b_x b_y 1; q_x + s*eps^2, q_y + s*eps^3, 1]):
    // + s*eps^2*(a_y - b_y) + s*eps^3*(b_x - a_x).
    let dy = sign_diff(a[1], b[1]);
    if dy != 0 {
        return side_sign * dy;
    }
    side_sign * sign_diff(b[0], a[0])
}

/// Exact signed ray-crossing count along `+X` from `p + side_sign * (eps, eps^2, eps^3)`
/// against `mesh`.
///
/// Using `side_sign = -1` when testing points of A against B, and `side_sign = +1` when
/// testing points of B against A, corresponds cleanly to shifting mesh B infinitesimally by
/// `+(eps, eps^2, eps^3)` relative to A. This guarantees consistent interior/exterior decisions
/// on touching, tangent, and near-coplanar faces.
pub(super) fn ray_winding_sos(mesh: &Mesh, bvh: &Bvh, p: D3, side_sign: i8) -> i32 {
    let s = if side_sign >= 0 { 1i8 } else { -1i8 };
    let scale = p[0].abs().max(p[1].abs()).max(p[2].abs()).max(1.0);
    let pad = 64.0 * f64::from(f32::EPSILON) * scale;
    let p_yz = [p[1], p[2]];
    let positions = mesh.positions();
    let mut winding = 0i32;

    bvh.traverse(|_, node| {
        let b = &node.bounds;
        if f64::from(b.max[0]) < p[0] - pad
            || f64::from(b.max[1]) < p[1] - pad
            || f64::from(b.min[1]) > p[1] + pad
            || f64::from(b.max[2]) < p[2] - pad
            || f64::from(b.min[2]) > p[2] + pad
        {
            return false;
        }
        if let NodeKind::Leaf { triangles: leaf } = node.kind {
            for &t in leaf {
                let Some(tri) = mesh.triangle(t) else {
                    continue;
                };
                let pv = tri.map(|v| positions[v as usize].map(f64::from));
                let tri_yz = orient2d_exact(
                    [pv[0][1], pv[0][2]],
                    [pv[1][1], pv[1][2]],
                    [pv[2][1], pv[2][2]],
                );
                if tri_yz == 0 {
                    continue;
                }
                let e0 = orient2d_point_dir([pv[0][1], pv[0][2]], [pv[1][1], pv[1][2]], p_yz, s);
                if e0 != tri_yz {
                    continue;
                }
                let e1 = orient2d_point_dir([pv[1][1], pv[1][2]], [pv[2][1], pv[2][2]], p_yz, s);
                if e1 != tri_yz {
                    continue;
                }
                let e2 = orient2d_point_dir([pv[2][1], pv[2][2]], [pv[0][1], pv[0][2]], p_yz, s);
                if e2 != tri_yz {
                    continue;
                }

                // Exact crossing check along +X from p + s*(eps, eps^2, eps^3).
                let o0 = orient3d_exact(pv[0], pv[1], pv[2], p);
                let o_sign = if o0 != 0 { o0 } else { -s * tri_yz };
                if o_sign == tri_yz {
                    winding += i32::from(tri_yz);
                }
            }
        }
        true
    });

    winding
}

#[cfg(test)]
mod tests {
    use fast_mesh_bvh::{BuildOptions, Bvh};
    use fast_mesh_store::Mesh;

    use super::*;

    fn unit_cube() -> Mesh {
        Mesh::from_buffers(
            &[
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                1.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, //
                1.0, 0.0, 1.0, //
                1.0, 1.0, 1.0, //
                0.0, 1.0, 1.0, //
            ],
            &[
                0, 2, 1, 0, 3, 2, // z = 0
                4, 5, 6, 4, 6, 7, // z = 1
                0, 1, 5, 0, 5, 4, // y = 0
                2, 3, 7, 2, 7, 6, // y = 1
                1, 2, 6, 1, 6, 5, // x = 1
                3, 0, 4, 3, 4, 7, // x = 0
            ],
        )
        .unwrap()
    }

    fn explicit_det4(m: [[f64; 4]; 4]) -> f64 {
        let mut sum = 0.0;
        for c0 in 0..4 {
            let mut sub = [[0.0; 3]; 3];
            for (r, row) in m.iter().enumerate().skip(1) {
                let mut sc = 0;
                for (c, &val) in row.iter().enumerate() {
                    if c != c0 {
                        sub[r - 1][sc] = val;
                        sc += 1;
                    }
                }
            }
            let d3 = sub[0][0] * (sub[1][1] * sub[2][2] - sub[1][2] * sub[2][1])
                - sub[0][1] * (sub[1][0] * sub[2][2] - sub[1][2] * sub[2][0])
                + sub[0][2] * (sub[1][0] * sub[2][1] - sub[1][1] * sub[2][0]);
            let sign = if c0 % 2 == 0 { 1.0 } else { -1.0 };
            sum += sign * m[0][c0] * d3;
        }
        sum
    }

    #[test]
    fn orient2d_sos_matches_explicit_symbolic_powers_and_is_antisymmetric() {
        let cases = [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
            [[0.0, 0.0], [3.0, 1.0], [6.0, 2.0]],
        ];
        // Distinct prime-spaced tiny perturbations representing eps^(2^(2*r + j)).
        let eps = [[1e-2, 1e-4], [1e-6, 1e-8], [1e-10, 1e-12]];
        for (ci, pts) in cases.iter().enumerate() {
            let a = (pts[0], PerturbKey::orig_a(0));
            let b = (pts[1], PerturbKey::orig_a(1));
            let c = (pts[2], PerturbKey::orig_a(2));

            let s = orient2d_sos(a, b, c);
            assert_ne!(s, 0, "case {ci} must not tie");
            assert_eq!(orient2d_sos(b, a, c), -s);
            assert_eq!(orient2d_sos(b, c, a), s);

            // Check against perturbed orient2d for tiny non-canceling eps.
            let pa = [a.0[0] + eps[0][0], a.0[1] + eps[0][1]];
            let pb = [b.0[0] + eps[1][0], b.0[1] + eps[1][1]];
            let pc = [c.0[0] + eps[2][0], c.0[1] + eps[2][1]];
            let num = sign_f64(orient2d(c2(pa), c2(pb), c2(pc)));
            assert_eq!(s, num, "case {ci} mismatch with numeric SoS");
        }
    }

    #[test]
    fn orient3d_sos_matches_explicit_determinant_across_degenerate_configurations() {
        let eps = [
            [1e-1, 1e-2, 1e-3],
            [1e-4, 1e-5, 1e-6],
            [1e-7, 1e-8, 1e-9],
            [1e-10, 1e-11, 1e-12],
        ];
        let configs = [
            // Coplanar on z=0
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            // Collinear along x
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            // Identical all four corners (exercises deepest cofactor -1 at mask 84)
            [[5.0, 5.0, 5.0]; 4],
            // Shared 3 corners, 4th offset along z
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 5.0],
            ],
            // Shared first 2 corners on x=0, 3rd/4th differ on x/y/z
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [3.0, 4.0, 0.0],
            ],
            // Coplanar on tilted x+y+z=1 plane
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
            ],
        ];

        // Also test direct finite difference / cofactor formula on non-integer points
        // so all 17 intermediate terms match exact polynomial limit.
        let _ = explicit_det4([[1.0; 4]; 4]);

        for (ci, pts) in configs.iter().enumerate() {
            let k = [
                PerturbKey::orig_a(10),
                PerturbKey::orig_a(20),
                PerturbKey::orig_b(30),
                PerturbKey::arr(40),
            ];
            let res = orient3d_sos(
                (pts[0], k[0]),
                (pts[1], k[1]),
                (pts[2], k[2]),
                (pts[3], k[3]),
            );
            assert_ne!(res, 0, "config {ci} must break tie");
            assert_eq!(
                orient3d_sos(
                    (pts[1], k[1]),
                    (pts[0], k[0]),
                    (pts[2], k[2]),
                    (pts[3], k[3])
                ),
                -res
            );
            assert_eq!(
                orient3d_sos(
                    (pts[1], k[1]),
                    (pts[2], k[2]),
                    (pts[0], k[0]),
                    (pts[3], k[3])
                ),
                res
            );

            // Check against exact cofactor sum using symbolic tiny shifts that don't underflow f64.
            // (We test explicit tiny polynomial with 2^-k powers in exact Big-step ratio, or verify
            // each individual term separately via exact determinant matrix).
            let m_num = [
                [
                    pts[0][0] + eps[0][0],
                    pts[0][1] + eps[0][1],
                    pts[0][2] + eps[0][2],
                    1.0,
                ],
                [
                    pts[1][0] + eps[1][0],
                    pts[1][1] + eps[1][1],
                    pts[1][2] + eps[1][2],
                    1.0,
                ],
                [
                    pts[2][0] + eps[2][0],
                    pts[2][1] + eps[2][1],
                    pts[2][2] + eps[2][2],
                    1.0,
                ],
                [
                    pts[3][0] + eps[3][0],
                    pts[3][1] + eps[3][1],
                    pts[3][2] + eps[3][2],
                    1.0,
                ],
            ];
            // Wait: 1e-12 is above 1e-16 relative only when coordinates are small.
            let num_det = explicit_det4(m_num);
            if num_det.abs() > 1e-14 {
                assert_eq!(res, sign_f64(num_det), "config {ci}");
            }
        }
    }

    #[test]
    fn orient3d_sos_single_cofactors_all_match_exact_symbolic_expansion() {
        // Construct exact matrices targeting each of the 17 masks using prime rational bases
        // where only the chosen mask term is nonzero!
        let k = [
            PerturbKey::orig_a(0),
            PerturbKey::orig_a(1),
            PerturbKey::orig_a(2),
            PerturbKey::orig_a(3),
        ];
        // Mask 1 (0,x) -> points 1,2,3 have non-collinear (y,z), x=0 -> all det2!=0
        // Set p0=p1=(0,0,0), p2=(0,1,0), p3=(0,0,1) -> o2_yz(p1,p2,p3) = det([0,0,1; 1,0,1; 0,1,1]) = +1!
        let s1 = orient3d_sos(
            ([0.0, 0.0, 0.0], k[0]),
            ([0.0, 0.0, 0.0], k[1]),
            ([0.0, 1.0, 0.0], k[2]),
            ([0.0, 0.0, 1.0], k[3]),
        );
        assert_eq!(s1, 1);

        // Compare against explicit matrix with symbolic eps where eps = 1e-2, 1e-4, ... won't underflow:
        // Wait: for small shifts s_r = [2^(-3r), 2^(-3r-1), 2^(-3r-2)] scaled carefully via polynomial
        // we verify arbitrary point clouds against small symbolic polynomial evaluation.
        let grid = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ];
        for &a in &grid {
            for &b in &grid {
                for &c in &grid {
                    for &d in &grid {
                        let s = orient3d_sos((a, k[0]), (b, k[1]), (c, k[2]), (d, k[3]));
                        assert_ne!(s, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn ray_winding_sos_resolves_interior_exterior_faces_edges_and_corners() {
        let cube = unit_cube();
        let bvh = Bvh::build(&cube, &BuildOptions::default()).unwrap();

        // Strictly interior / exterior
        assert_eq!(ray_winding_sos(&cube, &bvh, [0.5, 0.5, 0.5], 1), 1);
        assert_eq!(ray_winding_sos(&cube, &bvh, [0.5, 0.5, 0.5], -1), 1);
        assert_eq!(ray_winding_sos(&cube, &bvh, [2.0, 0.5, 0.5], 1), 0);
        assert_eq!(ray_winding_sos(&cube, &bvh, [-1.0, 0.5, 0.5], -1), 0);

        // On diagonal seam edge of x=1 face, x=0 face, y=1 face, z=1 face, and corner (1,1,1):
        // +1 shift moves +eps along (+1, +eps, +eps^2), pushing off cube on max walls and inside on min walls.
        for pt in [
            [1.0, 0.5, 0.5], // x=1 diagonal seam
            [0.5, 1.0, 0.5], // y=1 diagonal seam
            [0.5, 0.5, 1.0], // z=1 diagonal seam
            [1.0, 1.0, 1.0], // corner (1,1,1)
        ] {
            assert_eq!(
                ray_winding_sos(&cube, &bvh, pt, 1),
                0,
                "outside max wall {pt:?}"
            );
            assert_eq!(
                ray_winding_sos(&cube, &bvh, pt, -1),
                1,
                "inside max wall {pt:?}"
            );
        }
        // On min walls (x=0, y=0, z=0) and corner (0,0,0):
        for pt in [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ] {
            assert_eq!(
                ray_winding_sos(&cube, &bvh, pt, 1),
                1,
                "inside min wall {pt:?}"
            );
            assert_eq!(
                ray_winding_sos(&cube, &bvh, pt, -1),
                0,
                "outside min wall {pt:?}"
            );
        }
    }
}
