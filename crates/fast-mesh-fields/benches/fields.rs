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
use fast_mesh_fields::{
    GridSpec, SignData, SignMode, edt_3d_mask, edt_3d_sq, fh_signed_grid, fh_unsigned_grid,
    signed_distances,
};
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

fn fields_for_size(c: &mut Criterion, n: u32) {
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
                (fresh, fresh_bvh, black_box(sign))
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

fn fields(c: &mut Criterion) {
    for n in [224u32, 708u32] {
        fields_for_size(c, n);
    }
}

/// Closed outward-wound UV sphere (`2 * (stacks - 1) * slices` triangles).
fn sphere_mesh(center: [f32; 3], r: f32, stacks: u32, slices: u32) -> Mesh {
    use std::f64::consts::PI;
    let mut positions = Vec::with_capacity(((1 + (stacks - 1) * slices + 1) * 3) as usize);
    positions.extend([center[0], center[1], center[2] + r]);
    for i in 1..stacks {
        let theta = PI * f64::from(i) / f64::from(stacks);
        let (rs, rc) = (f64::from(r) * theta.sin(), f64::from(r) * theta.cos());
        for j in 0..slices {
            let phi = 2.0 * PI * f64::from(j) / f64::from(slices);
            positions.extend([
                center[0] + (rs * phi.cos()) as f32,
                center[1] + (rs * phi.sin()) as f32,
                center[2] + rc as f32,
            ]);
        }
    }
    let south = (positions.len() / 3) as u32;
    positions.extend([center[0], center[1], center[2] - r]);

    let ring = |i: u32, j: u32| 1 + (i - 1) * slices + j % slices;
    let mut indices = Vec::with_capacity((2 * (stacks - 1) * slices * 3) as usize);
    for j in 0..slices {
        indices.extend([0, ring(1, j), ring(1, j + 1)]);
        for i in 1..stacks - 1 {
            let (a, b, c_idx, d) = (
                ring(i, j),
                ring(i + 1, j),
                ring(i + 1, j + 1),
                ring(i, j + 1),
            );
            indices.extend([a, b, c_idx, a, c_idx, d]);
        }
        indices.extend([ring(stacks - 1, j), south, ring(stacks - 1, j + 1)]);
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn bench_edt(c: &mut Criterion) {
    let mut sq_group = c.benchmark_group("edt_3d_sq");
    sq_group.sample_size(10);
    for dim in [128usize, 256usize] {
        let dims = [dim, dim, dim];
        let total = dim * dim * dim;
        let center = (dim as f32) * 0.5;
        let radius_sq = ((dim as f32) * 0.35).powi(2);
        let mut cost = vec![f32::INFINITY; total];
        for iz in 0..dim {
            let dz = iz as f32 - center;
            for iy in 0..dim {
                let dy = iy as f32 - center;
                let row = (iz * dim + iy) * dim;
                for ix in 0..dim {
                    let dx = ix as f32 - center;
                    if dx * dx + dy * dy + dz * dz <= radius_sq {
                        cost[row + ix] = 0.0;
                    }
                }
            }
        }
        let mut out_sq = vec![0.0f32; total];
        sq_group.throughput(Throughput::Elements(total as u64));
        sq_group.bench_function(format!("{dim}_cubed"), |b| {
            b.iter(|| {
                edt_3d_sq(dims, &cost, &mut out_sq).unwrap();
                black_box(&out_sq);
            })
        });
    }
    sq_group.finish();

    let dim = 128usize;
    let dims = [dim, dim, dim];
    let total = dim * dim * dim;
    let center = (dim as f32) * 0.5;
    let radius_sq = ((dim as f32) * 0.35).powi(2);
    let mut inside = vec![false; total];
    for iz in 0..dim {
        let dz = iz as f32 - center;
        for iy in 0..dim {
            let dy = iy as f32 - center;
            let row = (iz * dim + iy) * dim;
            for ix in 0..dim {
                let dx = ix as f32 - center;
                inside[row + ix] = dx * dx + dy * dy + dz * dz <= radius_sq;
            }
        }
    }
    let mut out_dist = vec![0.0f32; total];
    let mut mask_group = c.benchmark_group("edt_3d_mask");
    mask_group.sample_size(10);
    mask_group.throughput(Throughput::Elements(total as u64));
    mask_group.bench_function("128_cubed", |b| {
        b.iter(|| {
            edt_3d_mask(dims, &inside, 1.0, &mut out_dist).unwrap();
            black_box(&out_dist);
        })
    });
    mask_group.finish();

    let spec = GridSpec {
        origin: [0.0, 0.0, 0.0],
        dims,
        voxel_size: 1.0,
    };
    let max_voxels = total;
    let sphere_100k = sphere_mesh([64.0, 64.0, 64.0], 44.0, 224, 225);
    let sphere_1m = sphere_mesh([64.0, 64.0, 64.0], 44.0, 708, 709);

    let mut fh_group = c.benchmark_group("fh_grid");
    fh_group.sample_size(10);
    fh_group.throughput(Throughput::Elements(total as u64));
    fh_group.bench_function("fh_unsigned_grid/100k_128_cubed", |b| {
        b.iter(|| black_box(fh_unsigned_grid(&sphere_100k, spec, max_voxels).unwrap()))
    });
    fh_group.bench_function("fh_signed_grid/100k_128_cubed", |b| {
        b.iter(|| black_box(fh_signed_grid(&sphere_100k, spec, max_voxels).unwrap()))
    });
    fh_group.bench_function("fh_signed_grid/1m_128_cubed", |b| {
        b.iter(|| black_box(fh_signed_grid(&sphere_1m, spec, max_voxels).unwrap()))
    });
    fh_group.finish();
}

criterion_group!(benches, fields, bench_edt);
criterion_main!(benches);
