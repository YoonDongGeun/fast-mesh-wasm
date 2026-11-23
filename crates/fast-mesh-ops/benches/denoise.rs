//! Denoising and Delaunay flip benchmarks at about 1M triangles.
//!
//! `cargo bench -p fast-mesh-ops --bench denoise` (add `--features parallel`
//! for the threaded passes). Each benchmark runs the op in a transaction it
//! then drops, so every iteration starts from the same mesh, and prints what
//! the op did once.
//!
//! - `denoise_1m`: a cube of 289 cells a side with noise of 15% of a cell,
//!   default options (5 rounds).
//! - `delaunay_1m`: a 707 x 707 grid with jittered vertices on a gentle
//!   swell, every cell split along the same diagonal, default options.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fast_mesh_ops::primitives::{BoxOptions, PlaneOptions, Primitive, generate};
use fast_mesh_ops::{DelaunayOptions, DenoiseOptions, delaunay_flips, denoise};
use fast_mesh_store::Mesh;

/// A deterministic generator, uniform in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
    }
}

/// `primitive` with every vertex passed through `f`.
fn mesh_of(primitive: &Primitive, mut f: impl FnMut([f32; 3]) -> [f32; 3]) -> Mesh {
    let src = generate(primitive).unwrap();
    let positions: Vec<f32> = src.positions.iter().flat_map(|&p| f(p)).collect();
    let indices: Vec<u32> = src.triangles.iter().flatten().copied().collect();
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn noisy_cube() -> Mesh {
    let mut r = Lcg(1);
    let cube = Primitive::Box(BoxOptions {
        size: [289.0; 3],
        segments: [289; 3],
    });
    mesh_of(&cube, |p| p.map(|x| x + 0.15 * r.next()))
}

fn jittered_grid() -> Mesh {
    let mut r = Lcg(2);
    let plane = Primitive::Plane(PlaneOptions {
        size: [707.0; 2],
        segments: [707; 2],
        ..Default::default()
    });
    mesh_of(&plane, |p| {
        let q = p.map(|x| x + 0.2 * r.next());
        // The plane's two axes carry the grid; lift along the third.
        let lift = 2.0 * (q[0] * 0.05).sin() * (q[2] * 0.04).cos();
        [q[0], lift, q[2]]
    })
}

fn bench_denoise(c: &mut Criterion) {
    let mut group = c.benchmark_group("denoise");
    group.sample_size(10);

    let mut cube = noisy_cube();
    let n = cube.triangle_count();
    group.throughput(Throughput::Elements(n as u64));
    let options = DenoiseOptions::default();
    let r = denoise(&mut cube.begin_edit(), &options, None).unwrap();
    eprintln!(
        "denoise/denoise_1m: {n} triangles, moved {}, mean normal deviation {:.2} -> {:.2} deg",
        r.moved, r.before.mean_normal_deviation_deg, r.after.mean_normal_deviation_deg
    );
    group.bench_function("denoise_1m", |b| {
        b.iter(|| black_box(denoise(&mut cube.begin_edit(), &options, None).unwrap()))
    });

    let mut grid = jittered_grid();
    let n = grid.triangle_count();
    group.throughput(Throughput::Elements(n as u64));
    let options = DelaunayOptions::default();
    let r = delaunay_flips(&mut grid.begin_edit(), &options, None).unwrap();
    eprintln!(
        "denoise/delaunay_1m: {n} triangles, {} flips in {} passes, min angle {:.2} -> {:.2} deg",
        r.flips, r.passes, r.before.min_angle_deg, r.after.min_angle_deg
    );
    group.bench_function("delaunay_1m", |b| {
        b.iter(|| black_box(delaunay_flips(&mut grid.begin_edit(), &options, None).unwrap()))
    });
    group.finish();
}

criterion_group!(benches, bench_denoise);
criterion_main!(benches);
