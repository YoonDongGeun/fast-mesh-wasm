//! Field benchmarks on a synthetic height field. The scene's field benchmarks
//! (the 64-pose drag) are in `fast-mesh-scene`.
//!
//! `cargo bench -p fast-mesh-fields` measures `SignData::build` (pseudonormal
//! and winding) on 1M triangles, the pseudonormal build again with the mesh's
//! adjacency not cached (as after a topology edit, which drops it), and `signed_distances` in both modes for a
//! batch of 100,000 points near the surface, with criterion throughput set to
//! the batch size so results read per point.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_fields::{SignData, SignMode, signed_distances};
use fast_mesh_store::Mesh;

const BATCH: usize = 100_000;

/// Height field over an n×n grid (2n² triangles).
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

/// Deterministic points over the terrain, within 2 units above or below it.
fn points(n: u32) -> Vec<[f32; 3]> {
    let mut state = 12345u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..BATCH)
        .map(|_| {
            let (x, y) = (next() * n as f32, next() * n as f32);
            let z = (x * 0.05).sin() * (y * 0.07).cos() * 4.0 + (next() - 0.5) * 4.0;
            [x, y, z]
        })
        .collect()
}

fn fields(c: &mut Criterion) {
    let n = 708; // about 1M triangles
    let mut mesh = terrain(n);
    let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
    let tris = mesh.triangle_count();

    let mut group = c.benchmark_group(format!("fields/{tris}"));
    group.sample_size(10);
    group.bench_function("sign_build_pseudonormal", |b| {
        b.iter(|| black_box(SignData::build(&mut mesh, &bvh, SignMode::Pseudonormal).unwrap()))
    });

    // A topology edit drops the cached adjacency (the incidence index is kept
    // up to date), so the next build pays for it. A fresh mesh has a new uid,
    // so each iteration builds its own mesh and BVH outside the timing and
    // warms only the incidence index.
    group.bench_function("sign_build_pseudonormal_cold_adjacency", |b| {
        b.iter_batched(
            || {
                let mut fresh = terrain(n);
                let fresh_bvh = Bvh::build(&fresh, &BuildOptions::default()).unwrap();
                fresh.ensure_incidence().unwrap();
                (fresh, fresh_bvh)
            },
            |(mut fresh, fresh_bvh)| {
                let sign = SignData::build(&mut fresh, &fresh_bvh, SignMode::Pseudonormal).unwrap();
                black_box(sign);
                (fresh, fresh_bvh)
            },
            BatchSize::PerIteration,
        )
    });

    group.bench_function("sign_build_winding", |b| {
        b.iter(|| black_box(SignData::build(&mut mesh, &bvh, SignMode::Winding).unwrap()))
    });

    let pts = points(n);
    let mut out = vec![0.0f32; BATCH];
    group.throughput(Throughput::Elements(BATCH as u64));
    for (name, mode) in [
        ("pseudonormal", SignMode::Pseudonormal),
        ("winding", SignMode::Winding),
    ] {
        let sign = SignData::build(&mut mesh, &bvh, mode).unwrap();
        for max_distance in [1.0f32, f32::INFINITY] {
            group.bench_function(format!("signed_distances/{name}/max_{max_distance}"), |b| {
                b.iter(|| {
                    signed_distances(&mesh, &bvh, &sign, &pts, max_distance, &mut out).unwrap();
                    black_box(&out);
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, fields);
criterion_main!(benches);
