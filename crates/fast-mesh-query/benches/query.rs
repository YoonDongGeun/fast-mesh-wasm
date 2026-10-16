//! Query benchmarks on synthetic height fields.
//!
//! `cargo bench -p fast-mesh-query` runs 1M triangles; set `FAST_MESH_BENCH_LARGE=1`
//! to add 10M. Batch benchmarks run 100,000 queries per iteration with criterion
//! throughput set to the batch size, so results read per ray or per point. Each
//! size prints the result counts of the region queries once.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::{
    ClosestOptions, Ray, RayOptions, Region, closest_point, raycast, raycast_any, triangles_in,
    vertices_in,
};
use fast_mesh_store::Mesh;

const BATCH: usize = 100_000;

/// Height field over an n×n grid (2n² triangles), so boxes are not all coplanar.
fn terrain(n: u32) -> Mesh {
    let mut positions = Vec::with_capacity(((n + 1) * (n + 1) * 3) as usize);
    for y in 0..=n {
        for x in 0..=n {
            let (fx, fy) = (x as f32, y as f32);
            positions.extend([fx, fy, (fx * 0.05).sin() * (fy * 0.07).cos() * 4.0]);
        }
    }
    let mut indices = Vec::with_capacity((n * n * 6) as usize);
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// A deterministic linear congruential generator.
struct Lcg(u64);

impl Lcg {
    /// Uniform in `[0, 1)`.
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn bench_size(c: &mut Criterion, label: &str, n: u32) {
    let mesh = terrain(n);
    let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
    let side = n as f32;

    // Rays start at z = 20 over the terrain and point down with some jitter.
    let mut rng = Lcg(0x5eed);
    let rays: Vec<Ray> = (0..BATCH)
        .map(|_| Ray {
            origin: [rng.next() * side, rng.next() * side, 20.0],
            dir: [rng.next() * 0.4 - 0.2, rng.next() * 0.4 - 0.2, -1.0],
        })
        .collect();
    // Points at random heights around the surface.
    let points: Vec<[f32; 3]> = (0..BATCH)
        .map(|_| {
            [
                rng.next() * side,
                rng.next() * side,
                rng.next() * 10.0 - 5.0,
            ]
        })
        .collect();
    let sphere = Region::Sphere {
        center: [side / 2.0, side / 2.0, 0.0],
        radius: 5.0,
    };

    let (mut tris, mut verts) = (Vec::new(), Vec::new());
    triangles_in(&mesh, &bvh, &sphere, &mut tris).unwrap();
    vertices_in(&mesh, &bvh, &sphere, &mut verts).unwrap();
    eprintln!(
        "{label}: {} triangles; sphere of radius 5 holds {} triangles and {} vertices",
        mesh.triangle_count(),
        tris.len(),
        verts.len()
    );

    let mut group = c.benchmark_group(format!("query {label}"));
    group.sample_size(10);
    group.throughput(Throughput::Elements(BATCH as u64));
    let opts = RayOptions::default();
    group.bench_function("raycast per ray", |b| {
        b.iter(|| {
            for ray in &rays {
                black_box(raycast(&mesh, &bvh, ray, &opts).unwrap());
            }
        })
    });
    group.bench_function("raycast_any per ray", |b| {
        b.iter(|| {
            for ray in &rays {
                black_box(raycast_any(&mesh, &bvh, ray, &opts).unwrap());
            }
        })
    });
    let closest_opts = ClosestOptions::default();
    group.bench_function("closest_point per point", |b| {
        b.iter(|| {
            for &p in &points {
                black_box(closest_point(&mesh, &bvh, p, &closest_opts).unwrap());
            }
        })
    });
    group.throughput(Throughput::Elements(1));
    group.bench_function("triangles_in sphere r=5", |b| {
        b.iter(|| {
            tris.clear();
            black_box(triangles_in(&mesh, &bvh, &sphere, &mut tris).unwrap());
        })
    });
    group.bench_function("vertices_in sphere r=5", |b| {
        b.iter(|| {
            verts.clear();
            black_box(vertices_in(&mesh, &bvh, &sphere, &mut verts).unwrap());
        })
    });
    group.finish();
}

fn benches(c: &mut Criterion) {
    bench_size(c, "1M", 708);
    if std::env::var_os("FAST_MESH_BENCH_LARGE").is_some() {
        bench_size(c, "10M", 2237);
    }
}

criterion_group!(group, benches);
criterion_main!(group);
