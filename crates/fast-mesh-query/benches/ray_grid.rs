//! A grid of parallel rays at a terrain: by tiles ([`RayGrid`]) against one
//! [`raycast`] per ray, both on this thread. Criterion throughput is the ray
//! count, so the results read per ray.
//!
//! `cargo bench -p fast-mesh-query --bench ray_grid` casts 1024 x 1024 rays at
//! 20k, 1M (and with `FAST_MESH_BENCH_LARGE=1`, 10M) triangles: many rays per
//! triangle, about one, and many triangles per ray.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::{RAY_TILE, Ray, RayGrid, RayOptions, raycast};
use fast_mesh_store::Mesh;

const SIDE: usize = 1024;

/// Height field over an n×n grid (2n² triangles) of side 1.
fn terrain(n: u32) -> Mesh {
    let mut positions = Vec::with_capacity(((n + 1) * (n + 1) * 3) as usize);
    for y in 0..=n {
        for x in 0..=n {
            let (fx, fy) = (x as f32 / n as f32, y as f32 / n as f32);
            positions.extend([fx, fy, (fx * 30.0).sin() * (fy * 40.0).cos() * 0.05]);
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

fn bench(c: &mut Criterion) {
    let mut sizes = vec![("20k", 100), ("1m", 708)];
    if std::env::var("FAST_MESH_BENCH_LARGE").is_ok_and(|v| v == "1") {
        sizes.push(("10m", 2237));
    }
    for (label, n) in sizes {
        let mesh = terrain(n);
        let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
        let pitch = 1.0 / SIDE as f64;
        for (name, dir) in [("straight", [0.0, 0.0, -1.0]), ("tilted", [0.3, 0.2, -1.0])] {
            // Start above the terrain, so that a tilted ray lands on it too.
            let origin: [f64; 3] = [0.5 * pitch - dir[0], 0.5 * pitch - dir[1], 1.0];
            let (su, sv) = ([pitch, 0.0, 0.0], [0.0, pitch, 0.0]);
            let mut group = c.benchmark_group(format!("ray_grid {label} {name}"));
            group.sample_size(10);
            group.throughput(Throughput::Elements((SIDE * SIDE) as u64));
            group.bench_function("tiles", |b| {
                b.iter(|| {
                    let grid = RayGrid::new(&mesh, &bvh, origin, dir, su, sv).unwrap();
                    let mut hits = 0;
                    for row in (0..SIDE).step_by(RAY_TILE) {
                        for col in (0..SIDE).step_by(RAY_TILE) {
                            let mut t = [f32::INFINITY; RAY_TILE * RAY_TILE];
                            let (c, r) = (col as u32, row as u32);
                            let hit = grid.nearest_in_tile(c, r, RAY_TILE, RAY_TILE, 0.0, &mut t);
                            hits += black_box(hit).count_ones();
                            black_box(t);
                        }
                    }
                    hits
                })
            });
            group.bench_function("per ray", |b| {
                let opts = RayOptions::default();
                b.iter(|| {
                    let mut hits = 0;
                    for row in 0..SIDE {
                        for col in 0..SIDE {
                            let o: [f64; 3] = std::array::from_fn(|k| {
                                origin[k] + col as f64 * su[k] + row as f64 * sv[k]
                            });
                            let ray = Ray {
                                origin: o.map(|c| c as f32),
                                dir: dir.map(|c| c as f32),
                            };
                            let hit = raycast(&mesh, &bvh, &ray, &opts).unwrap();
                            hits += u32::from(black_box(hit).is_some());
                        }
                    }
                    hits
                })
            });
            group.finish();
        }
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
