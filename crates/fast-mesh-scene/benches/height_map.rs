//! Height map benchmarks: sampling 1024 x 1024 and 4096 x 4096 images of one
//! terrain mesh, straight down and tilted, and adding reliefs of them.
//!
//! `cargo bench -p fast-mesh-scene --bench height_map` takes terrains of 20k
//! and 1M triangles; set `FAST_MESH_BENCH_LARGE=1` to add 10M, and 4096 x 4096
//! reliefs (33.5M triangles). Add `--features parallel` for the threads.
//! Criterion throughput is the pixel count, so the results read per pixel.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fast_mesh_ops::heightfield::{HeightfieldOptions, heightfield};
use fast_mesh_scene::{HeightMapOptions, MeshOptions, Placement, Scene, SceneOptions};
use fast_mesh_store::Mesh;

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

fn large() -> bool {
    std::env::var("FAST_MESH_BENCH_LARGE").is_ok_and(|v| v == "1")
}

fn bench_sampling(c: &mut Criterion) {
    let mut sizes = vec![("20k", 100), ("1m", 708)];
    if large() {
        sizes.push(("10m", 2237));
    }
    for (label, n) in sizes {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(terrain(n), MeshOptions::default()).unwrap();
        let mut group = c.benchmark_group(format!("height_map {label}"));
        group.sample_size(10);
        for (name, direction) in [("down", [0.0, 0.0, -1.0]), ("tilted", [0.3, 0.2, -1.0])] {
            for resolution in [1024, 4096] {
                let options = HeightMapOptions {
                    direction,
                    resolution,
                    ..HeightMapOptions::default()
                };
                let frame = scene.height_map_frame(&[id], &options).unwrap();
                let pixels = u64::from(frame.width) * u64::from(frame.height);
                group.throughput(Throughput::Elements(pixels));
                group.bench_function(format!("{name} {resolution}"), |b| {
                    b.iter(|| {
                        let heights = scene.height_map(&[id], &frame, 0..frame.height);
                        black_box(heights.unwrap())
                    })
                });
            }
        }
        group.throughput(Throughput::Elements(1));
        group.bench_function("frame", |b| {
            let options = HeightMapOptions::default();
            b.iter(|| scene.height_map_frame(&[id], black_box(&options)).unwrap())
        });
        group.finish();
    }
}

fn bench_relief(c: &mut Criterion) {
    let mut group = c.benchmark_group("heightfield");
    group.sample_size(10);
    for side in [1024u32, 4096] {
        let heights: Vec<f32> = (0..side * side)
            .map(|i| {
                let (x, y) = ((i % side) as f32, (i / side) as f32);
                (x * 0.02).sin() * (y * 0.03).cos() + 1.0
            })
            .collect();
        let options = HeightfieldOptions {
            pixel_size: [0.1; 2],
            base: 1.0,
        };
        group.throughput(Throughput::Elements(u64::from(side) * u64::from(side)));
        group.bench_function(format!("generate {side}"), |b| {
            b.iter(|| heightfield(black_box(&heights), side, side, &options).unwrap())
        });
        if side > 1024 && !large() {
            continue;
        }
        // Into the scene: the mesh, its BVH and the undo record, in a fresh
        // scene each time.
        group.bench_function(format!("add {side}"), |b| {
            b.iter(|| {
                let mut scene = Scene::new(SceneOptions::default());
                let placement = Placement::IDENTITY;
                let r = scene.add_heightfield(&heights, side, side, &options, placement);
                black_box(r.unwrap());
                scene
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sampling, bench_relief);
criterion_main!(benches);
