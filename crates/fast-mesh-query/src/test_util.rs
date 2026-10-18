//! Fixtures shared by the query tests.

use fast_mesh_bvh::{BuildOptions, Bvh, Quality};
use fast_mesh_store::Mesh;

/// A `(n + 1) x (n + 1)` grid of vertices with `z = (x + y) % 3`, two triangles per cell.
pub(crate) fn wavy(n: u32) -> Mesh {
    let mut positions = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            positions.extend([x as f32, y as f32, ((x + y) % 3) as f32]);
        }
    }
    let mut indices = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// A BVH with small leaves, so that the tests exercise deep trees.
pub(crate) fn build(mesh: &Mesh, quality: Quality) -> Bvh {
    let opts = BuildOptions {
        quality,
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
}

/// A random direction for a plane normal: uniform in the cube `[-1, 1)^3`,
/// rejecting lengths up to 0.1.
pub(crate) fn random_normal(rng: &mut Lcg) -> [f64; 3] {
    loop {
        let n: [f64; 3] = std::array::from_fn(|_| f64::from(rng.next()) * 2.0 - 1.0);
        if n.iter().map(|c| c * c).sum::<f64>() > 0.01 {
            return n;
        }
    }
}

/// A random rotation matrix (row-major) from a random unit quaternion.
pub(crate) fn random_rotation(rng: &mut Lcg) -> [[f64; 3]; 3] {
    loop {
        let q: [f64; 4] = std::array::from_fn(|_| f64::from(rng.next()) * 2.0 - 1.0);
        let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        if n < 0.1 {
            continue;
        }
        let [w, x, y, z] = q.map(|v| v / n);
        return [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ];
    }
}

/// A closed UV sphere around the origin with outward winding: one vertex at
/// each pole and `stacks - 1` rings of `slices` vertices.
pub(crate) fn uv_sphere(stacks: u32, slices: u32, r: f32) -> Mesh {
    use std::f64::consts::PI;
    let mut positions = vec![0.0, 0.0, r];
    for i in 1..stacks {
        let theta = PI * f64::from(i) / f64::from(stacks);
        for j in 0..slices {
            let phi = 2.0 * PI * f64::from(j) / f64::from(slices);
            let (rs, rc) = (f64::from(r) * theta.sin(), f64::from(r) * theta.cos());
            positions.extend([(rs * phi.cos()) as f32, (rs * phi.sin()) as f32, rc as f32]);
        }
    }
    positions.extend([0.0, 0.0, -r]);
    let south = 1 + (stacks - 1) * slices;
    let ring = |i: u32, j: u32| 1 + (i - 1) * slices + j % slices;
    let mut indices = Vec::new();
    for j in 0..slices {
        indices.extend([0, ring(1, j), ring(1, j + 1)]);
        for i in 1..stacks - 1 {
            let (a, b, c, d) = (
                ring(i, j),
                ring(i + 1, j),
                ring(i + 1, j + 1),
                ring(i, j + 1),
            );
            indices.extend([a, b, c, a, c, d]);
        }
        indices.extend([ring(stacks - 1, j), south, ring(stacks - 1, j + 1)]);
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// The box `[lo, hi]` as 12 triangles with outward winding. Corner `i` takes
/// `hi` on axis `k` when bit `k` of `i` is set.
pub(crate) fn box_mesh(lo: [f32; 3], hi: [f32; 3]) -> Mesh {
    let mut positions = Vec::new();
    for i in 0..8 {
        positions.extend((0..3).map(|k| if i >> k & 1 == 1 { hi[k] } else { lo[k] }));
    }
    let indices = [
        0, 4, 6, 0, 6, 2, // -x
        1, 3, 7, 1, 7, 5, // +x
        0, 1, 5, 0, 5, 4, // -y
        2, 6, 7, 2, 7, 3, // +y
        0, 2, 3, 0, 3, 1, // -z
        4, 5, 7, 4, 7, 6, // +z
    ];
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// A closed torus around the z axis with outward winding: `n` segments around
/// the major circle and `m` around the tube.
pub(crate) fn torus(major: f32, minor: f32, n: u32, m: u32) -> Mesh {
    use std::f64::consts::PI;
    let mut positions = Vec::new();
    for i in 0..n {
        let phi = 2.0 * PI * f64::from(i) / f64::from(n);
        for j in 0..m {
            let theta = 2.0 * PI * f64::from(j) / f64::from(m);
            let w = f64::from(major) + f64::from(minor) * theta.cos();
            positions.extend([
                (w * phi.cos()) as f32,
                (w * phi.sin()) as f32,
                (f64::from(minor) * theta.sin()) as f32,
            ]);
        }
    }
    let at = |i: u32, j: u32| (i % n) * m + j % m;
    let mut indices = Vec::new();
    for i in 0..n {
        for j in 0..m {
            let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            indices.extend([a, b, c, a, c, d]);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// Three unit quads that share the edge from vertex 0 at the origin to vertex
/// 1 at `(0, 0, 1)`, fanned 120 degrees apart: that edge is non-manifold.
pub(crate) fn fin() -> Mesh {
    let mut positions = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let mut indices = Vec::new();
    for k in 0..3u32 {
        let angle = 2.0 * std::f64::consts::PI * f64::from(k) / 3.0;
        let (x, y) = (angle.cos() as f32, angle.sin() as f32);
        positions.extend([x, y, 0.0, x, y, 1.0]);
        let (p, q) = (2 + 2 * k, 3 + 2 * k);
        indices.extend([0, p, q, 0, q, 1]);
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// `mesh` (alive triangles only, origin dropped) with zero-area triangles
/// appended: for every fifth triangle `[a, b, c]`, the triangles `[a, a, b]`,
/// `[b, c, c]` and `[c, c, c]`, which reuse its vertices.
pub(crate) fn with_degenerates(mesh: Mesh) -> Mesh {
    let positions: Vec<f32> = mesh.positions().iter().flatten().copied().collect();
    let alive: Vec<[u32; 3]> = (0..mesh.triangle_slots() as u32)
        .filter(|&t| mesh.is_triangle_alive(t))
        .map(|t| mesh.triangles()[t as usize])
        .collect();
    let mut indices: Vec<u32> = alive.iter().flatten().copied().collect();
    for &[a, b, c] in alive.iter().step_by(5) {
        indices.extend([a, a, b, b, c, c, c, c, c]);
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// Six times the signed volume enclosed by the mesh: positive for a closed mesh
/// with outward winding.
pub(crate) fn signed_volume6(mesh: &Mesh) -> f64 {
    let p = mesh.positions();
    (0..mesh.triangle_slots() as u32)
        .filter(|&t| mesh.is_triangle_alive(t))
        .map(|t| {
            let [a, b, c] = mesh.triangles()[t as usize].map(|v| p[v as usize].map(f64::from));
            a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
        })
        .sum()
}

/// An `n` x `n` sheet of unit cells over `[0, n]^2`, rippled along x
/// (`along_y` false) or along y, with amplitude 1.
pub(crate) fn ripple(n: u32, along_y: bool) -> Mesh {
    let (positions, indices) = ripple_buffers(n, along_y, 0);
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// The two [`ripple`] sheets of size `n` in one mesh, crossing each other
/// along many curves.
pub(crate) fn crossed_ripples(n: u32) -> Mesh {
    let (mut positions, mut indices) = ripple_buffers(n, false, 0);
    let (p, i) = ripple_buffers(n, true, (positions.len() / 3) as u32);
    positions.extend(p);
    indices.extend(i);
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// The buffers of a [`ripple`] sheet, its vertex ids starting at `base`.
fn ripple_buffers(n: u32, along_y: bool, base: u32) -> (Vec<f32>, Vec<u32>) {
    let mut positions = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            let t = if along_y { y } else { x } as f32;
            positions.extend([x as f32, y as f32, (t * 0.7).sin()]);
        }
    }
    let mut indices = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let a = base + y * (n + 1) + x;
            indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
        }
    }
    (positions, indices)
}

/// Runs `run` with parallel work forced off (the code a feature-off build
/// runs), then on pools of 1, 2 and 8 threads and on the global pool,
/// asserting every run matches the first; with the `parallel` feature, also
/// that the global run took a parallel path. Returns the first run.
pub(crate) fn assert_parallel_identical<R: PartialEq + std::fmt::Debug + Send>(
    what: &str,
    run: impl Fn() -> R + Sync,
) -> R {
    assert_runs_identical(what, true, run)
}

/// [`assert_parallel_identical`], asserting that the global run took a
/// parallel path only when `splits` (a search whose root boxes are rejected
/// has nothing to split).
pub(crate) fn assert_runs_identical<R: PartialEq + std::fmt::Debug + Send>(
    what: &str,
    splits: bool,
    run: impl Fn() -> R + Sync,
) -> R {
    use fast_mesh_store::par;
    par::set_force_sequential(true);
    let reference = run();
    par::set_force_sequential(false);
    if !par::enabled() {
        // Every run below would be this one again.
        return reference;
    }
    for k in [1, 2, 8] {
        let got = par::with_threads(k, &run);
        assert!(got == reference, "{what}: {k} threads");
    }
    let before = par::parallel_entries();
    assert!(run() == reference, "{what}: global pool");
    if splits {
        assert!(par::parallel_entries() > before, "{what}: ran sequentially");
    }
    reference
}
