//! Wrap benchmarks at about 1M input triangles.
//!
//! `cargo bench -p fast-mesh-ops --bench wrap` (add `--features parallel` for
//! the threaded distance transforms, closest-point batches and marching
//! cubes). Each benchmark prints its grid and what the wrap came out as once.
//!
//! The input is a bumpy, scan-like closed sphere of 1.31M triangles (an
//! icosphere of 8 subdivisions displaced along its normals) with 24 holes
//! about 0.1 across, and a box of 12 triangles that touches it, wrapped
//! together with a 0.15 gap:
//!
//! - `scan_1m_detail_128`: the default detail, 128 voxels along the longest
//!   side.
//! - `scan_1m_detail_256`: twice the detail, eight times the voxels.
//! - `scan_1m_detail_128_projected`: the default detail, with the result's
//!   vertices projected onto the input (and its self-intersection checks).

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use fast_mesh_ops::append::AppendSource;
use fast_mesh_ops::primitives::{BoxOptions, IcosphereOptions, cuboid, icosphere};
use fast_mesh_ops::wrap::{WrapOptions, wrap};

/// The bumpy sphere with holes, and a box touching it.
fn scan() -> AppendSource {
    let mut s = icosphere(&IcosphereOptions {
        radius: 1.0,
        subdivisions: 8,
    })
    .unwrap();
    for p in &mut s.positions {
        let q = p.map(f64::from);
        let bump = 1.0 + 0.04 * (7.0 * q[0]).sin() * (7.0 * q[1]).sin() * (7.0 * q[2]).sin();
        *p = q.map(|c| (c * bump) as f32);
    }
    // 24 holes on a spiral.
    let centers: Vec<[f64; 3]> = (0..24)
        .map(|i| {
            let z = 1.0 - (2.0 * f64::from(i) + 1.0) / 24.0;
            let r = (1.0 - z * z).sqrt();
            let a = 2.399_963 * f64::from(i);
            [r * a.cos(), r * a.sin(), z]
        })
        .collect();
    let p = |v: u32| s.positions[v as usize].map(f64::from);
    s.triangles.retain(|t| {
        let c: [f64; 3] = std::array::from_fn(|k| t.iter().map(|&v| p(v)[k]).sum::<f64>() / 3.0);
        centers
            .iter()
            .all(|o| (0..3).map(|k| (c[k] - o[k]).powi(2)).sum::<f64>() > 0.05 * 0.05)
    });
    let cube = cuboid(&BoxOptions {
        size: [0.6; 3],
        segments: [1; 3],
    })
    .unwrap();
    let base = s.positions.len() as u32;
    s.positions
        .extend(cube.positions.iter().map(|q| [q[0] + 1.3, q[1], q[2]]));
    s.triangles
        .extend(cube.triangles.iter().map(|t| t.map(|v| v + base)));
    s
}

fn bench_wrap(c: &mut Criterion) {
    let input = scan();
    let mut group = c.benchmark_group("wrap");
    group.sample_size(10);
    for (name, detail, project) in [
        ("scan_1m_detail_128", 128, false),
        ("scan_1m_detail_256", 256, false),
        ("scan_1m_detail_128_projected", 128, true),
    ] {
        let options = WrapOptions::builder()
            .gap(0.15)
            .detail(detail)
            .project(project)
            .build();
        let (_, report) = wrap(&input, &options).unwrap();
        let plan = report.plan;
        eprintln!(
            "wrap/{name}: {} triangles -> grid {:?} ({} voxels, {} MB), {} vertices, {} triangles, {} projected",
            input.triangles.len(),
            plan.dims,
            plan.voxels,
            plan.bytes >> 20,
            report.vertices,
            report.triangles,
            report.projected
        );
        group.bench_function(name, |b| {
            b.iter(|| black_box(wrap(&input, &options).unwrap()))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_wrap);
criterion_main!(benches);
