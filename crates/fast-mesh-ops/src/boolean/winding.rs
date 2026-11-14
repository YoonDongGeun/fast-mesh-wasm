//! Read-only generalized winding number and signed volume evaluation.
//!
//! [`WindingTree`] precomputes Barill et al. (2018) multipole moments (dipole,
//! quadrupole, and second-order spatial moments) over an existing [`Bvh`]
//! without mutating or copying the mesh. Queries evaluate exact
//! Van Oosterom–Strackee solid angles (with exact [`robust::orient3d`]
//! determinant signs) at leaves and near inner nodes (`dist^2 <= beta^2 * r^2`
//! with `beta = 2.3`), and the second-order Taylor expansion on far subtrees.

use std::f64::consts::PI;

use fast_mesh_bvh::{Aabb, Bvh, NodeId, NodeKind};
use fast_mesh_store::{Mesh, TriangleId};
use robust::{Coord3D, orient3d};

use crate::{Result, filled, reserve};

/// Far-field acceptance threshold from Barill et al. 2018 (`dist > beta * r`).
const BETA: f64 = 2.3;
const BETA2: f64 = BETA * BETA;

/// Shewchuk exactness bounds (`[2^-200, 2^200]`, plus `0.0`) for [`orient3d`].
const MIN_SAFE: f64 = f64::from_bits(0x3370_0000_0000_0000);
const MAX_SAFE: f64 = f64::from_bits(0x4c70_0000_0000_0000);

type D3 = [f64; 3];

/// Symmetric `3x3` tensor packed as `[xx, yy, zz, xy, xz, yz]`.
type Sym3 = [f64; 6];

fn add(a: D3, b: D3) -> D3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: D3, b: D3) -> D3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: D3, s: f64) -> D3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: D3, b: D3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: D3, b: D3) -> D3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: D3) -> f64 {
    dot(a, a).sqrt()
}

fn widen(p: [f32; 3]) -> D3 {
    p.map(f64::from)
}

fn safe_coord(x: f64) -> bool {
    let a = x.abs();
    a == 0.0 || (MIN_SAFE..=MAX_SAFE).contains(&a)
}

/// Symmetrized outer product `0.5 * (u ⊗ v + v ⊗ u)` in `[xx, yy, zz, xy, xz, yz]` order.
fn sym_outer(u: D3, v: D3) -> Sym3 {
    [
        u[0] * v[0],
        u[1] * v[1],
        u[2] * v[2],
        0.5 * (u[0] * v[1] + u[1] * v[0]),
        0.5 * (u[0] * v[2] + u[2] * v[0]),
        0.5 * (u[1] * v[2] + u[2] * v[1]),
    ]
}

fn sym_trace(s: &Sym3) -> f64 {
    s[0] + s[1] + s[2]
}

fn sym_mul_vec(s: &Sym3, r: D3) -> D3 {
    [
        s[0] * r[0] + s[3] * r[1] + s[4] * r[2],
        s[3] * r[0] + s[1] * r[1] + s[5] * r[2],
        s[4] * r[0] + s[5] * r[1] + s[2] * r[2],
    ]
}

fn box_center(bounds: &Aabb) -> D3 {
    if bounds.is_empty() {
        return [0.0; 3];
    }
    std::array::from_fn(|k| 0.5 * (f64::from(bounds.min[k]) + f64::from(bounds.max[k])))
}

fn farthest_corner(c: D3, bounds: &Aabb) -> f64 {
    let d: D3 = std::array::from_fn(|k| {
        (f64::from(bounds.min[k]) - c[k])
            .abs()
            .max((f64::from(bounds.max[k]) - c[k]).abs())
    });
    norm(d)
}

/// Returns `6.0 * signed_volume` over all alive triangles of `mesh` in `f64`
/// (`> 0.0` for outward-wound closed solids).
///
/// Evaluated relative to the first alive proper triangle's first vertex so
/// meshes translated far from the origin preserve precision.
pub(super) fn signed_volume6(mesh: &Mesh) -> f64 {
    let positions = mesh.positions();
    let mut origin: Option<D3> = None;
    let mut sum = 0.0;
    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some([a, b, c]) = mesh.triangle(t) else {
            continue;
        };
        if a == b || b == c || a == c {
            continue;
        }
        let pa0 = widen(positions[a as usize]);
        let o = *origin.get_or_insert(pa0);
        let pa = sub(pa0, o);
        let pb = sub(widen(positions[b as usize]), o);
        let pc = sub(widen(positions[c as usize]), o);
        sum += dot(pa, cross(pb, pc));
    }
    sum
}

/// Multipole moments of one BVH subtree about its area-weighted `centroid`.
#[derive(Debug, Clone, Copy, Default)]
struct NodeMoment {
    /// 0th moment (dipole): `∫ n_i dA`.
    normal: D3,
    /// Total triangle area of the subtree.
    area: f64,
    /// Expansion center `c` (area-weighted centroid, or box center if zero area).
    centroid: D3,
    /// Bounding ball radius around `centroid` enclosing all triangles in the subtree.
    radius: f64,
    /// 1st spatial moment (quadrupole): `M[i][j] = ∫ n_i (y_j - c_j) dA`.
    m1: [[f64; 3]; 3],
    /// 2nd spatial moment: `Q[i][(j,k)] = ∫ n_i (y_j - c_j)(y_k - c_k) dA`.
    m2: [Sym3; 3],
}

impl NodeMoment {
    /// Second-order Taylor approximation of the solid angle (in steradians) of
    /// this subtree seen from `p`, given `r = centroid - p` and `dist2 = |r|^2 > 0`.
    fn far_solid_angle(&self, r: D3, dist2: f64) -> f64 {
        let inv_r2 = 1.0 / dist2;
        let inv_r = inv_r2.sqrt();
        let s3 = inv_r * inv_r2;
        let s5 = s3 * inv_r2;
        let s7 = s5 * inv_r2;

        // Dipole: (N · r) / |r|^3
        let w0 = dot(self.normal, r) * s3;

        // Quadrupole: tr(M) / |r|^3 - 3 (r^T M r) / |r|^5
        let tr_m1 = self.m1[0][0] + self.m1[1][1] + self.m1[2][2];
        let mut r_m1_r = 0.0;
        for i in 0..3 {
            r_m1_r += r[i] * dot(self.m1[i], r);
        }
        let w1 = tr_m1 * s3 - 3.0 * r_m1_r * s5;

        // Second-order moment: 1/2 Σ_{i,j,k} Q_{ijk} ∂²(r_i/|r|^3)/(∂r_j ∂r_k)
        let mut sum_iik = 0.0;
        let mut sum_ijj = 0.0;
        let mut sum_ijk = 0.0;
        for i in 0..3 {
            let qr = sym_mul_vec(&self.m2[i], r);
            sum_iik += qr[i];
            sum_ijj += sym_trace(&self.m2[i]) * r[i];
            sum_ijk += r[i] * dot(r, qr);
        }
        let w2 = -1.5 * s5 * (2.0 * sum_iik + sum_ijj) + 7.5 * s7 * sum_ijk;

        w0 + w1 + w2
    }
}

fn leaf_moment(mesh: &Mesh, triangles: &[TriangleId], bounds: &Aabb) -> NodeMoment {
    let positions = mesh.positions();
    let mut normal = [0.0; 3];
    let mut area = 0.0;
    let mut weighted = [0.0; 3];

    for &t in triangles {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        let [v0, v1, v2] = tri.map(|v| widen(positions[v as usize]));
        let n = scale(cross(sub(v1, v0), sub(v2, v0)), 0.5);
        let w = norm(n);
        normal = add(normal, n);
        area += w;
        weighted = add(weighted, scale(add(v0, add(v1, v2)), w / 3.0));
    }

    let centroid = if area > 0.0 {
        scale(weighted, 1.0 / area)
    } else {
        box_center(bounds)
    };

    let mut radius = 0.0f64;
    let mut m1 = [[0.0; 3]; 3];
    let mut m2 = [[0.0; 6]; 3];

    for &t in triangles {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        let [v0, v1, v2] = tri.map(|v| widen(positions[v as usize]));
        for v in [v0, v1, v2] {
            radius = radius.max(norm(sub(v, centroid)));
        }
        let n = scale(cross(sub(v1, v0), sub(v2, v0)), 0.5);
        let d0 = sub(v0, centroid);
        let d1 = sub(v1, centroid);
        let d2 = sub(v2, centroid);
        let s = add(d0, add(d1, d2));
        let d_bar = scale(s, 1.0 / 3.0);

        let d0_sq = sym_outer(d0, d0);
        let d1_sq = sym_outer(d1, d1);
        let d2_sq = sym_outer(d2, d2);
        let s_sq = sym_outer(s, s);
        let cov: Sym3 = std::array::from_fn(|p| (d0_sq[p] + d1_sq[p] + d2_sq[p] + s_sq[p]) / 12.0);

        for i in 0..3 {
            for j in 0..3 {
                m1[i][j] += n[i] * d_bar[j];
            }
            for p in 0..6 {
                m2[i][p] += n[i] * cov[p];
            }
        }
    }

    NodeMoment {
        normal,
        area,
        centroid,
        radius,
        m1,
        m2,
    }
}

fn inner_moment(l: &NodeMoment, r: &NodeMoment, bounds: &Aabb) -> NodeMoment {
    let area = l.area + r.area;
    let centroid = if area > 0.0 {
        scale(
            add(scale(l.centroid, l.area), scale(r.centroid, r.area)),
            1.0 / area,
        )
    } else {
        box_center(bounds)
    };

    let from_children = (norm(sub(l.centroid, centroid)) + l.radius)
        .max(norm(sub(r.centroid, centroid)) + r.radius);
    let radius = if bounds.is_empty() {
        from_children
    } else {
        from_children.min(farthest_corner(centroid, bounds))
    };

    let mut normal = [0.0; 3];
    let mut m1 = [[0.0; 3]; 3];
    let mut m2 = [[0.0; 6]; 3];

    for child in [l, r] {
        let delta = sub(child.centroid, centroid);
        let delta_sq = sym_outer(delta, delta);
        for i in 0..3 {
            normal[i] += child.normal[i];
            for j in 0..3 {
                m1[i][j] += child.m1[i][j] + child.normal[i] * delta[j];
            }
            let m1_delta = sym_outer(child.m1[i], delta);
            for p in 0..6 {
                m2[i][p] += child.m2[i][p] + 2.0 * m1_delta[p] + child.normal[i] * delta_sq[p];
            }
        }
    }

    NodeMoment {
        normal,
        area,
        centroid,
        radius,
        m1,
        m2,
    }
}

/// Exact signed solid angle (in steradians) of triangle `[v0, v1, v2]` seen
/// from `p` (Van Oosterom & Strackee 1983), positive when facing away from `p`.
fn solid_angle([v0, v1, v2]: [D3; 3], p: D3) -> f64 {
    let a = sub(v0, p);
    let b = sub(v1, p);
    let c = sub(v2, p);
    let (la, lb, lc) = (norm(a), norm(b), norm(c));
    if la == 0.0 || lb == 0.0 || lc == 0.0 {
        return 0.0;
    }
    let num = if [v0, v1, v2, p].iter().flatten().all(|&x| safe_coord(x)) {
        let c3 = |q: D3| Coord3D {
            x: q[0],
            y: q[1],
            z: q[2],
        };
        orient3d(c3(v0), c3(v1), c3(v2), c3(p))
    } else {
        dot(a, cross(b, c))
    };
    if num == 0.0 {
        return 0.0;
    }
    let den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
    2.0 * num.atan2(den)
}

/// Read-only hierarchical generalized winding number accelerator over a [`Bvh`].
#[derive(Debug, Clone)]
pub(super) struct WindingTree {
    nodes: Vec<NodeMoment>,
}

impl WindingTree {
    /// Precomputes area-weighted dipole, quadrupole, and second-order moments
    /// per BVH node bottom-up.
    pub(super) fn build(mesh: &Mesh, bvh: &Bvh) -> Result<Self> {
        let mut nodes = filled(bvh.node_slots(), NodeMoment::default())?;
        let mut order: Vec<NodeId> = Vec::new();
        reserve(&mut order, bvh.node_count())?;
        bvh.traverse(|id, _| {
            order.push(id);
            true
        });
        for &id in order.iter().rev() {
            let view = bvh.node(id);
            nodes[id.index()] = match view.kind {
                NodeKind::Leaf { triangles } => leaf_moment(mesh, triangles, &view.bounds),
                NodeKind::Inner { left, right } => {
                    inner_moment(&nodes[left.index()], &nodes[right.index()], &view.bounds)
                }
            };
        }
        Ok(Self { nodes })
    }

    /// Evaluates the generalized winding number of `mesh` at `p` (`~1.0` inside
    /// an outward closed solid, `~0.0` outside). Allocates nothing.
    pub(super) fn winding_number(&self, mesh: &Mesh, bvh: &Bvh, p: [f64; 3]) -> f64 {
        let positions = mesh.positions();
        let mut total = 0.0;
        bvh.traverse(|id, view| match view.kind {
            NodeKind::Leaf { triangles } => {
                for &t in triangles {
                    if let Some(tri) = mesh.triangle(t) {
                        let corners = tri.map(|v| widen(positions[v as usize]));
                        total += solid_angle(corners, p);
                    }
                }
                false
            }
            NodeKind::Inner { .. } => {
                let m = &self.nodes[id.index()];
                let r = sub(m.centroid, p);
                let dist2 = dot(r, r);
                if dist2 > BETA2 * m.radius * m.radius {
                    total += m.far_solid_angle(r, dist2);
                    false
                } else {
                    true
                }
            }
        });
        total / (4.0 * PI)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use fast_mesh_bvh::{BuildOptions, Bvh};
    use fast_mesh_store::{Mesh, StoreError, TriangleId};

    use super::{WindingTree, signed_volume6};
    use crate::test_util::{box_mesh, edit, flip, hemisphere, uv_sphere};
    use crate::{OpsError, ShellOptions, fail_allocations_after, shell};

    fn cube() -> Mesh {
        box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    }

    fn build_bvh(mesh: &Mesh) -> Bvh {
        Bvh::build(mesh, &BuildOptions::default()).unwrap()
    }

    /// Brute-force Van Oosterom–Strackee winding sum over all alive triangles,
    /// as an independent reference for the hierarchical Taylor evaluation.
    fn exact_winding_ref(mesh: &Mesh, p: [f64; 3]) -> f64 {
        let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let norm = |a: [f64; 3]| dot(a, a).sqrt();

        let mut total = 0.0;
        for t in 0..mesh.triangle_slots() as TriangleId {
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };
            let [a, b, c] = tri.map(|v| sub(mesh.positions()[v as usize].map(f64::from), p));
            let (la, lb, lc) = (norm(a), norm(b), norm(c));
            if la == 0.0 || lb == 0.0 || lc == 0.0 {
                continue;
            }
            let num = dot(a, cross(b, c));
            if num == 0.0 {
                continue;
            }
            let den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
            total += num.atan2(den) / (2.0 * PI);
        }
        total
    }

    #[test]
    fn unit_cube_winding_and_grazing_rays() {
        let mesh = cube();
        assert!(
            (signed_volume6(&mesh) - 6.0).abs() < 1e-12,
            "signed_volume6 = {}",
            signed_volume6(&mesh)
        );

        let bvh = build_bvh(&mesh);
        let tree = WindingTree::build(&mesh, &bvh).unwrap();

        for inside in [[0.5, 0.5, 0.5], [0.25, 0.25, 0.125], [0.999, 0.5, 0.5]] {
            let w = tree.winding_number(&mesh, &bvh, inside);
            assert!(
                w > 0.99 && w < 1.01,
                "expected inside winding ~1.0 at {inside:?}, got {w}"
            );
        }

        for outside in [[-0.001, 0.5, 0.5], [2.0, 0.5, 0.5]] {
            let w = tree.winding_number(&mesh, &bvh, outside);
            assert!(
                w > -0.01 && w < 0.01,
                "expected outside winding ~0.0 at {outside:?}, got {w}"
            );
        }
    }

    #[test]
    fn inward_cube_has_negative_volume_and_winding() {
        let clean = cube();
        let all: Vec<TriangleId> = (0..clean.triangle_count() as TriangleId).collect();
        let mesh = flip(&clean, &all);

        assert!(
            (signed_volume6(&mesh) - (-6.0)).abs() < 1e-12,
            "signed_volume6 = {}",
            signed_volume6(&mesh)
        );

        let bvh = build_bvh(&mesh);
        let tree = WindingTree::build(&mesh, &bvh).unwrap();
        let w = tree.winding_number(&mesh, &bvh, [0.5, 0.5, 0.5]);
        assert!(
            (w - (-1.0)).abs() < 0.01,
            "expected inward winding ~-1.0, got {w}"
        );
    }

    #[test]
    fn hollow_shell_winding() {
        let mut mesh = uv_sphere(12, 16, 1.0);
        let _ = edit(&mut mesh, |tx| shell(tx, &ShellOptions { thickness: 0.3 }));

        let vol6 = signed_volume6(&mesh);
        assert!(vol6 > 0.0, "expected positive signed_volume6, got {vol6}");

        let bvh = build_bvh(&mesh);
        let tree = WindingTree::build(&mesh, &bvh).unwrap();

        let w_void = tree.winding_number(&mesh, &bvh, [0.0, 0.0, 0.0]);
        assert!(
            w_void > -0.02 && w_void < 0.02,
            "expected inner void winding ~0.0, got {w_void}"
        );

        let w_wall = tree.winding_number(&mesh, &bvh, [0.85, 0.0, 0.0]);
        assert!(
            w_wall > 0.98 && w_wall < 1.02,
            "expected inside-wall winding ~1.0, got {w_wall}"
        );
    }

    #[test]
    fn winding_build_honours_oom_hook() {
        let mesh = uv_sphere(8, 12, 1.0);
        let bvh = build_bvh(&mesh);

        fail_allocations_after(Some(0));
        let res0 = WindingTree::build(&mesh, &bvh);
        fail_allocations_after(None);
        assert_eq!(res0.err(), Some(OpsError::Store(StoreError::OutOfMemory)));

        fail_allocations_after(Some(1));
        let res1 = WindingTree::build(&mesh, &bvh);
        fail_allocations_after(None);
        assert_eq!(res1.err(), Some(OpsError::Store(StoreError::OutOfMemory)));
    }

    #[test]
    fn far_field_taylor_matches_exact_to_fifth_order() {
        // An open curved hemisphere has non-zero 0th, 1st and 2nd moments simultaneously.
        let mesh = hemisphere(12, 24, 1.0);
        let bvh = Bvh::build(
            &mesh,
            &BuildOptions {
                max_leaf_size: 2,
                ..BuildOptions::default()
            },
        )
        .unwrap();
        let tree = WindingTree::build(&mesh, &bvh).unwrap();

        let dir = [0.53, -0.71, 0.46];
        let err5 = {
            let p = dir.map(|c| 5.0 * c);
            (tree.winding_number(&mesh, &bvh, p) - exact_winding_ref(&mesh, p)).abs()
        };
        let err10 = {
            let p = dir.map(|c| 10.0 * c);
            (tree.winding_number(&mesh, &bvh, p) - exact_winding_ref(&mesh, p)).abs()
        };
        // At d=5 (~5x radius > beta=2.3), the root itself uses the Taylor expansion.
        // With 0th + 1st + 2nd spatial moments (dipole + quadrupole + second-order),
        // the remainder is O(d^-5), so doubling d shrinks the error by ~32x.
        assert!(err5 < 1e-4, "err at d=5 too large: {err5:.3e}");
        assert!(
            err10 < err5 / 16.0,
            "expected O(d^-5) decay: err5={err5:.3e}, err10={err10:.3e}"
        );
    }

    #[test]
    fn sub_ulp_proximity_to_slanted_face_keeps_exact_sign() {
        // Tetrahedron (0,0,0), (1,0,0), (0,1,0), (0,0,1): slanted face x + y + z = 1.
        let mesh = crate::test_util::mesh_of(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]],
        );
        let bvh = build_bvh(&mesh);
        let tree = WindingTree::build(&mesh, &bvh).unwrap();

        // Points 1e-13 inside and outside the slanted face x + y + z = 1.
        let inside = [1.0 / 3.0 - 1e-13, 1.0 / 3.0, 1.0 / 3.0];
        let outside = [1.0 / 3.0 + 1e-13, 1.0 / 3.0, 1.0 / 3.0];
        let w_in = tree.winding_number(&mesh, &bvh, inside);
        let w_out = tree.winding_number(&mesh, &bvh, outside);
        assert!((w_in - 1.0).abs() < 1e-9, "sub-ulp inside: {w_in}");
        assert!(w_out.abs() < 1e-9, "sub-ulp outside: {w_out}");
    }
}
