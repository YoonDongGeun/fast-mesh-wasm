//! Fixtures and brute-force references shared by the field tests.

use std::collections::HashMap;
use std::f64::consts::PI;

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_store::Mesh;

/// The cube `[-1, 1]³`, 12 triangles facing out.
pub(crate) fn cube_buffers() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let positions = (0..8)
        .map(|i| [i & 1, (i >> 1) & 1, (i >> 2) & 1].map(|b| b as f32 * 2.0 - 1.0))
        .collect();
    let triangles = vec![
        [0, 2, 3],
        [0, 3, 1],
        [4, 5, 7],
        [4, 7, 6],
        [0, 1, 5],
        [0, 5, 4],
        [2, 6, 7],
        [2, 7, 3],
        [0, 4, 6],
        [0, 6, 2],
        [1, 3, 7],
        [1, 7, 5],
    ];
    (positions, triangles)
}

/// A mesh from position and triangle lists.
pub(crate) fn mesh(positions: &[[f32; 3]], triangles: &[[u32; 3]]) -> Mesh {
    Mesh::from_buffers(positions.as_flattened(), triangles.as_flattened()).unwrap()
}

/// The cube `[-1, 1]³`.
pub(crate) fn cube() -> Mesh {
    let (p, t) = cube_buffers();
    mesh(&p, &t)
}

/// The unit icosahedron subdivided twice: 320 triangles facing out.
pub(crate) fn icosphere() -> Mesh {
    let g = (1.0 + 5f64.sqrt()) / 2.0;
    let mut pos: Vec<[f64; 3]> = vec![
        [-1.0, g, 0.0],
        [1.0, g, 0.0],
        [-1.0, -g, 0.0],
        [1.0, -g, 0.0],
        [0.0, -1.0, g],
        [0.0, 1.0, g],
        [0.0, -1.0, -g],
        [0.0, 1.0, -g],
        [g, 0.0, -1.0],
        [g, 0.0, 1.0],
        [-g, 0.0, -1.0],
        [-g, 0.0, 1.0],
    ];
    let mut tris: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..2 {
        let mut mid: HashMap<(u32, u32), u32> = HashMap::new();
        let mut midpoint = |a: u32, b: u32, pos: &mut Vec<[f64; 3]>| {
            *mid.entry((a.min(b), a.max(b))).or_insert_with(|| {
                let (pa, pb) = (pos[a as usize], pos[b as usize]);
                pos.push(std::array::from_fn(|k| (pa[k] + pb[k]) / 2.0));
                (pos.len() - 1) as u32
            })
        };
        let mut next = Vec::new();
        for [a, b, c] in tris {
            let ab = midpoint(a, b, &mut pos);
            let bc = midpoint(b, c, &mut pos);
            let ca = midpoint(c, a, &mut pos);
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        tris = next;
    }
    let positions: Vec<[f32; 3]> = pos
        .iter()
        .map(|p| {
            let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            p.map(|c| (c / n) as f32)
        })
        .collect();
    mesh(&positions, &tris)
}

/// A torus about z with radii 1 and 0.4, 24 × 12 quads split in two, facing out.
pub(crate) fn torus() -> Mesh {
    let (nu, nv) = (24u32, 12u32);
    let mut positions = Vec::new();
    for i in 0..nu {
        for j in 0..nv {
            let u = 2.0 * PI * f64::from(i) / f64::from(nu);
            let v = 2.0 * PI * f64::from(j) / f64::from(nv);
            let r = 1.0 + 0.4 * v.cos();
            positions.push([r * u.cos(), r * u.sin(), 0.4 * v.sin()].map(|c| c as f32));
        }
    }
    let id = |i: u32, j: u32| (i % nu) * nv + (j % nv);
    let mut tris = Vec::new();
    for i in 0..nu {
        for j in 0..nv {
            let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
            tris.extend([[a, b, c], [a, c, d]]);
        }
    }
    mesh(&positions, &tris)
}

/// A regular tetrahedron, facing out. Its dihedral angles are about 70.5°,
/// so its edge and vertex normals differ from each face normal by more than
/// 90°: a sign from the wrong feature normal shows up here.
pub(crate) fn tetrahedron() -> Mesh {
    let (p, t) = tetrahedron_buffers();
    mesh(&p, &t)
}

/// The positions and triangles of [`tetrahedron`].
pub(crate) fn tetrahedron_buffers() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let p = vec![
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ];
    (p, vec![[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
}

/// A five-pointed star extruded along z, facing out: sharp convex edges at
/// the tips and sharp concave edges between them.
pub(crate) fn star_prism() -> Mesh {
    let n = 5u32;
    let rim = 2 * n;
    let mut positions = Vec::new();
    for z in [-0.5f64, 0.5] {
        for i in 0..rim {
            let a = 2.0 * PI * f64::from(i) / f64::from(rim);
            let r = if i % 2 == 0 { 1.0 } else { 0.3 };
            positions.push([r * a.cos(), r * a.sin(), z].map(|c| c as f32));
        }
    }
    positions.push([0.0, 0.0, -0.5]);
    positions.push([0.0, 0.0, 0.5]);
    let (bc, tc) = (2 * rim, 2 * rim + 1);
    let mut tris = Vec::new();
    for i in 0..rim {
        let j = (i + 1) % rim;
        let (b0, b1, t0, t1) = (i, j, rim + i, rim + j);
        tris.extend([[tc, t0, t1], [bc, b1, b0], [b0, b1, t1], [b0, t1, t0]]);
    }
    mesh(&positions, &tris)
}

/// Height of the terrain sheet at `(x, y)` in vertex coordinates.
pub(crate) fn terrain_height(x: f32, y: f32) -> f32 {
    0.6 * (0.5 * x).sin() * (0.4 * y).cos()
}

/// An `n × n` cell terrain sheet over `[0, n]²`, two triangles per cell, normals up.
pub(crate) fn terrain(n: u32) -> Mesh {
    let mut positions = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            let (fx, fy) = (x as f32, y as f32);
            positions.push([fx, fy, terrain_height(fx, fy)]);
        }
    }
    let mut tris = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            tris.extend([[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]);
        }
    }
    mesh(&positions, &tris)
}

/// The height of the piecewise-linear terrain surface over `(x, y)`, in `f64`.
pub(crate) fn terrain_surface(mesh: &Mesh, n: u32, x: f64, y: f64) -> f64 {
    let (cx, cy) = (x.floor() as u32, y.floor() as u32);
    let (fx, fy) = (x - f64::from(cx), y - f64::from(cy));
    let z = |i: u32, j: u32| f64::from(mesh.positions()[((cy + j) * (n + 1) + cx + i) as usize][2]);
    if fx >= fy {
        // Triangle (0,0), (1,0), (1,1).
        z(0, 0) + (z(1, 0) - z(0, 0)) * fx + (z(1, 1) - z(1, 0)) * fy
    } else {
        // Triangle (0,0), (1,1), (0,1).
        z(0, 0) + (z(1, 1) - z(0, 1)) * fx + (z(0, 1) - z(0, 0)) * fy
    }
}

/// A BVH with small leaves, so that the tests exercise deep trees.
pub(crate) fn build_bvh(mesh: &Mesh) -> Bvh {
    let opts = BuildOptions {
        max_leaf_size: 2,
        ..Default::default()
    };
    Bvh::build(mesh, &opts).unwrap()
}

/// A deterministic linear congruential generator.
pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    /// Uniform in `[0, 1)`.
    pub(crate) fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `[lo, hi)` on every axis.
    pub(crate) fn point(&mut self, lo: [f32; 3], hi: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|k| lo[k] + (hi[k] - lo[k]) * self.next())
    }
}

/// A point on a random alive triangle, at random barycentric coordinates.
pub(crate) fn surface_point(mesh: &Mesh, rng: &mut Lcg) -> [f32; 3] {
    let t = loop {
        let t = (rng.next() * mesh.triangle_slots() as f32) as u32;
        if mesh.is_triangle_alive(t) {
            break t;
        }
    };
    let [a, b, c] = mesh
        .triangle(t)
        .unwrap()
        .map(|v| mesh.positions()[v as usize]);
    let (mut u, mut v) = (rng.next(), rng.next());
    if u + v > 1.0 {
        (u, v) = (1.0 - u, 1.0 - v);
    }
    std::array::from_fn(|k| a[k] + (b[k] - a[k]) * u + (c[k] - a[k]) * v)
}

/// Random points: half uniform in the mesh's box grown by a quarter of the
/// diagonal, half near the surface (offset by up to `near` in a random direction).
pub(crate) fn sample_points(mesh: &Mesh, rng: &mut Lcg, count: usize, near: f32) -> Vec<[f32; 3]> {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in mesh.positions() {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let grow = diagonal(mesh) as f32 / 4.0;
    let (lo, hi) = (lo.map(|c| c - grow), hi.map(|c| c + grow));
    (0..count)
        .map(|i| {
            if i % 2 == 0 {
                rng.point(lo, hi)
            } else {
                let s = surface_point(mesh, rng);
                let d = rng.point([-near; 3], [near; 3]);
                std::array::from_fn(|k| s[k] + d[k])
            }
        })
        .collect()
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn len(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn corners(mesh: &Mesh, t: usize) -> [[f64; 3]; 3] {
    mesh.triangles()[t].map(|v| mesh.positions()[v as usize].map(f64::from))
}

fn alive_triangles(mesh: &Mesh) -> impl Iterator<Item = usize> + '_ {
    (0..mesh.triangle_slots()).filter(|&t| mesh.is_triangle_alive(t as u32))
}

/// The exact generalized winding number of `mesh` at `p`: the sum of the
/// triangles' solid angles (Van Oosterom–Strackee) over 4π, in `f64`.
pub(crate) fn exact_winding(mesh: &Mesh, p: [f32; 3]) -> f64 {
    let p = p.map(f64::from);
    let mut total = 0.0;
    for t in alive_triangles(mesh) {
        let [a, b, c] = corners(mesh, t).map(|x| sub(x, p));
        let (la, lb, lc) = (len(a), len(b), len(c));
        let num = dot(a, cross(b, c));
        let den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
        total += 2.0 * num.atan2(den);
    }
    total / (4.0 * PI)
}

/// Distance from `p` to the segment `a`–`b`.
fn segment_distance(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    let t = if l2 > 0.0 {
        (dot(sub(p, a), ab) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    len(sub(
        p,
        [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t],
    ))
}

/// Distance from `p` to a triangle: to its plane when the projection falls
/// inside, otherwise to the nearest edge.
fn triangle_distance(p: [f64; 3], [a, b, c]: [[f64; 3]; 3]) -> f64 {
    let n = cross(sub(b, a), sub(c, a));
    let nn = dot(n, n);
    if nn > 0.0 {
        let inside = [(a, b), (b, c), (c, a)]
            .iter()
            .all(|&(u, v)| dot(cross(sub(v, u), sub(p, u)), n) >= 0.0);
        if inside {
            return dot(sub(p, a), n).abs() / nn.sqrt();
        }
    }
    segment_distance(p, a, b)
        .min(segment_distance(p, b, c))
        .min(segment_distance(p, c, a))
}

/// The distance from `p` to the nearest alive triangle of `mesh`, by brute force.
pub(crate) fn brute_distance(mesh: &Mesh, p: [f32; 3]) -> f64 {
    let p = p.map(f64::from);
    alive_triangles(mesh)
        .map(|t| triangle_distance(p, corners(mesh, t)))
        .fold(f64::INFINITY, f64::min)
}

/// Checks signed distances: magnitudes against brute force and signs against
/// the exact winding number, outside `1e-5 · diagonal` of the surface.
/// Returns how many signs were checked.
pub(crate) fn check_against_exact(mesh: &Mesh, points: &[[f32; 3]], values: &[f32]) -> usize {
    let tol = 1e-5 * diagonal(mesh);
    let mut checked = 0;
    for (&p, &value) in points.iter().zip(values) {
        let brute = brute_distance(mesh, p);
        assert!(
            (f64::from(value.abs()) - brute).abs() <= tol,
            "magnitude at {p:?}: {value} vs brute force {brute}"
        );
        if brute > tol {
            let inside = exact_winding(mesh, p) >= 0.5;
            assert_eq!(value < 0.0, inside, "sign at {p:?}: value {value}");
            checked += 1;
        }
    }
    checked
}

/// The length of the bounding-box diagonal of the mesh's vertices.
pub(crate) fn diagonal(mesh: &Mesh) -> f64 {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in mesh.positions() {
        for k in 0..3 {
            lo[k] = lo[k].min(f64::from(p[k]));
            hi[k] = hi[k].max(f64::from(p[k]));
        }
    }
    len(sub(hi, lo))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The closed fixtures are oriented outward: the winding number is 1 at an
    /// inside point and 0 at an outside one.
    #[test]
    fn closed_fixtures_face_out() {
        let cases = [
            (cube(), [0.1, 0.2, 0.3], [3.0, 0.0, 0.0]),
            (icosphere(), [0.1, -0.2, 0.3], [0.0, 2.0, 0.0]),
            (torus(), [1.0, 0.0, 0.1], [0.0, 0.0, 0.0]),
            (tetrahedron(), [0.1, 0.1, 0.0], [2.0, 2.0, 2.0]),
            (star_prism(), [0.5, 0.0, 0.2], [0.5, 0.5, 0.0]),
        ];
        for (m, inside, outside) in cases {
            assert!((exact_winding(&m, inside) - 1.0).abs() < 1e-9);
            assert!(exact_winding(&m, outside).abs() < 1e-9);
        }
        assert_eq!(icosphere().triangle_count(), 320);
    }

    #[test]
    fn terrain_surface_interpolates_vertices() {
        let n = 8;
        let m = terrain(n);
        for (x, y) in [(2.0, 3.0), (5.0, 1.0)] {
            let z = terrain_surface(&m, n, x, y);
            assert!((z - f64::from(terrain_height(x as f32, y as f32))).abs() < 1e-6);
        }
        // Directly on the surface the brute-force distance is zero.
        let (x, y) = (2.3, 4.6);
        let z = terrain_surface(&m, n, x, y);
        assert!(brute_distance(&m, [x as f32, y as f32, z as f32]) < 1e-6);
    }
}
