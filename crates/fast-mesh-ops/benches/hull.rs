//! Convex hull benchmarks at 1M input points.
//!
//! `cargo bench -p fast-mesh-ops --bench hull` (add `--features parallel` for
//! the threaded classification). Each benchmark prints what its input holds
//! and what the hull came out as once.
//!
//! - `cube_1m`: uniform in a cube; few points end on the hull.
//! - `ball_1m`: uniform in a ball; a few thousand end on the hull.
//! - `sphere_1m`: on a sphere; every point is a hull vertex, quickhull's
//!   hardest case.
//! - `mesh_1m`: the vertices of a 1M-vertex UV-sphere mesh, through
//!   `convex_hull_mesh` (gathering the points and building the new mesh
//!   included).

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fast_mesh_ops::{convex_hull, convex_hull_mesh};
use fast_mesh_store::Mesh;

const N: usize = 1_000_000;

/// A deterministic linear congruential generator.
struct Lcg(u64);

impl Lcg {
    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in the ball of radius 1, by rejection.
    fn in_ball(&mut self) -> [f64; 3] {
        loop {
            let p = [0, 1, 2].map(|_| 2.0 * self.unit() - 1.0);
            let r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
            if r2 < 1.0 && r2 > 1e-6 {
                return p;
            }
        }
    }
}

fn cube(n: usize) -> Vec<[f64; 3]> {
    let mut r = Lcg(1);
    (0..n).map(|_| [r.unit(), r.unit(), r.unit()]).collect()
}

fn ball(n: usize) -> Vec<[f64; 3]> {
    let mut r = Lcg(2);
    (0..n).map(|_| r.in_ball()).collect()
}

fn sphere(n: usize) -> Vec<[f64; 3]> {
    let mut r = Lcg(3);
    (0..n)
        .map(|_| {
            let p = r.in_ball();
            let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            p.map(|c| c / l)
        })
        .collect()
}

/// A closed UV sphere with `stacks * slices` vertices and poles.
fn uv_sphere(stacks: u32, slices: u32) -> Mesh {
    let mut positions = vec![0.0f32, 0.0, 1.0];
    for i in 1..stacks {
        let theta = std::f64::consts::PI * f64::from(i) / f64::from(stacks);
        for j in 0..slices {
            let phi = std::f64::consts::TAU * f64::from(j) / f64::from(slices);
            let (s, c) = theta.sin_cos();
            positions.extend([(s * phi.cos()) as f32, (s * phi.sin()) as f32, c as f32]);
        }
    }
    positions.extend([0.0, 0.0, -1.0]);
    let south = (positions.len() / 3 - 1) as u32;
    let ring = |i: u32, j: u32| 1 + (i - 1) * slices + j % slices;
    let mut indices = Vec::new();
    for j in 0..slices {
        indices.extend([0, ring(1, j), ring(1, j + 1)]);
    }
    for i in 1..stacks - 1 {
        for j in 0..slices {
            let (a, b, c, d) = (
                ring(i, j),
                ring(i, j + 1),
                ring(i + 1, j),
                ring(i + 1, j + 1),
            );
            indices.extend([a, c, b, b, c, d]);
        }
    }
    for j in 0..slices {
        indices.extend([ring(stacks - 1, j), south, ring(stacks - 1, j + 1)]);
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn bench_hull(c: &mut Criterion) {
    let mut group = c.benchmark_group("convex_hull");
    group.sample_size(10);
    group.throughput(Throughput::Elements(N as u64));
    for (name, points) in [
        ("cube_1m", cube(N)),
        ("ball_1m", ball(N)),
        ("sphere_1m", sphere(N)),
    ] {
        let hull = convex_hull(&points).unwrap();
        eprintln!(
            "convex_hull/{name}: {} points -> {} vertices, {} triangles",
            points.len(),
            hull.vertices.len(),
            hull.triangles.len()
        );
        group.bench_function(name, |b| {
            b.iter(|| black_box(convex_hull(&points).unwrap()))
        });
    }
    let mesh = uv_sphere(1000, 1000);
    let hull = convex_hull_mesh(&mesh).unwrap();
    eprintln!(
        "convex_hull/mesh_1m: {} vertices, {} triangles -> {} vertices, {} triangles",
        mesh.vertex_count(),
        mesh.triangle_count(),
        hull.vertex_count(),
        hull.triangle_count()
    );
    group.bench_function("mesh_1m", |b| {
        b.iter(|| black_box(convex_hull_mesh(&mesh).unwrap()))
    });
    group.finish();
}

criterion_group!(benches, bench_hull);
criterion_main!(benches);
