//! Repair benchmarks on synthetic height fields.
//!
//! `cargo bench -p fast-mesh-ops` runs 1M triangles (`terrain(708)`). Each
//! benchmark prints what its input holds and what the operation did once.
//! Operations that write (`orient`, `fill_holes`, `weld`,
//! `remove_small_components`) are timed on a mesh rebuilt from buffers for every
//! iteration, outside the timing, and include the transaction's commit. `analyze`
//! reads only; its self-intersection variant takes a prebuilt BVH.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_ops::{
    AnalyzeOptions, ComponentOptions, FillOptions, analyze, boundary_loops, fill_holes, orient,
    remove_small_components, weld,
};
use fast_mesh_store::Mesh;

const N: u32 = 708;

type Buffers = (Vec<f32>, Vec<u32>);

fn height(x: f32, y: f32) -> f32 {
    (x * 0.05).sin() * (y * 0.07).cos() * 4.0
}

/// Height field over an n×n grid (2n² triangles), as buffers. Vertex `(x, y)`
/// is `y·(n+1) + x`.
fn terrain_buffers(n: u32) -> Buffers {
    let mut positions = Vec::with_capacity(((n + 1) * (n + 1) * 3) as usize);
    for y in 0..=n {
        for x in 0..=n {
            let (fx, fy) = (x as f32, y as f32);
            positions.extend([fx, fy, height(fx, fy)]);
        }
    }
    let mut indices = Vec::with_capacity((n * n * 6) as usize);
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
        }
    }
    (positions, indices)
}

fn build((positions, indices): &Buffers) -> Mesh {
    Mesh::from_buffers(positions, indices).unwrap()
}

/// A deterministic linear congruential generator.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// The terrain with every 100th triangle (1%) wound the other way.
fn flipped_terrain(n: u32) -> (Buffers, usize) {
    let (positions, mut indices) = terrain_buffers(n);
    let mut flipped = 0;
    for t in indices.as_chunks_mut::<3>().0.iter_mut().step_by(100) {
        t.swap(1, 2);
        flipped += 1;
    }
    ((positions, indices), flipped)
}

/// The terrain with ten round holes, each punched by dropping the triangles
/// whose centroid lies within its radius, so each leaves one boundary loop.
fn holed_terrain(n: u32, radii: &[f32; 10]) -> Buffers {
    let (positions, indices) = terrain_buffers(n);
    let side = n as f32;
    let centres: Vec<[f32; 2]> = (0..10)
        .map(|i| {
            [
                side * (0.1 + 0.2 * (i % 5) as f32) + 3.0 * i as f32,
                side * (0.25 + 0.5 * (i / 5) as f32) + 7.0 * i as f32,
            ]
        })
        .collect();
    let mut kept = Vec::with_capacity(indices.len());
    for t in indices.as_chunks::<3>().0 {
        let c = |k: usize| {
            let p = |i: usize| positions[t[i] as usize * 3 + k];
            (p(0) + p(1) + p(2)) / 3.0
        };
        let (cx, cy) = (c(0), c(1));
        let in_hole = centres.iter().zip(radii).any(|(h, &r)| {
            let (dx, dy) = (cx - h[0], cy - h[1]);
            dx * dx + dy * dy < r * r
        });
        if !in_hole {
            kept.extend_from_slice(t);
        }
    }
    (positions, kept)
}

/// The terrain in which every 10th vertex has a duplicate at the same
/// position, and each triangle corner that uses such a vertex takes the
/// duplicate with probability 1/2.
fn duplicated_terrain(n: u32) -> (Buffers, usize) {
    let (mut positions, mut indices) = terrain_buffers(n);
    let base = positions.len() as u32 / 3;
    let mut dupe_of = vec![u32::MAX; base as usize];
    let mut dupes = 0;
    for v in (0..base).step_by(10) {
        dupe_of[v as usize] = base + dupes;
        let i = v as usize * 3;
        let p = [positions[i], positions[i + 1], positions[i + 2]];
        positions.extend(p);
        dupes += 1;
    }
    let mut rng = Lcg(0x5eed);
    for i in indices.iter_mut() {
        let d = dupe_of[*i as usize];
        if d != u32::MAX && rng.next_u32() & 1 == 1 {
            *i = d;
        }
    }
    ((positions, indices), dupes as usize)
}

/// The terrain plus `count` separate two-triangle quads beside it.
fn terrain_with_specks(n: u32, count: u32) -> Buffers {
    let (mut positions, mut indices) = terrain_buffers(n);
    let mut next = positions.len() as u32 / 3;
    for i in 0..count {
        let x = n as f32 + 10.0 + (i % 100) as f32 * 4.0;
        let y = (i / 100) as f32 * 4.0;
        positions.extend([
            x,
            y,
            0.0,
            x + 1.0,
            y,
            0.0,
            x + 1.0,
            y + 1.0,
            0.0,
            x,
            y + 1.0,
            0.0,
        ]);
        indices.extend([next, next + 1, next + 2, next, next + 2, next + 3]);
        next += 4;
    }
    (positions, indices)
}

fn bench_analyze(c: &mut Criterion) {
    let mesh = build(&terrain_buffers(N));
    let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
    let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
    eprintln!(
        "analyze_1M: {} triangles, {} loops, {} components",
        mesh.triangle_count(),
        a.loops.len(),
        a.components.len()
    );
    let with = AnalyzeOptions {
        self_intersection: Some(&bvh),
        ..AnalyzeOptions::default()
    };
    let a = analyze(&mesh, &with).unwrap();
    eprintln!("analyze_1M: self_intersects = {:?}", a.self_intersects);

    let mut group = c.benchmark_group("ops analyze_1M");
    group.sample_size(10);
    group.bench_function("analyze", |b| {
        b.iter(|| black_box(analyze(&mesh, &AnalyzeOptions::default()).unwrap()))
    });
    group.bench_function("analyze with self-intersection", |b| {
        b.iter(|| black_box(analyze(&mesh, &with).unwrap()))
    });
    group.finish();
}

fn bench_orient(c: &mut Criterion) {
    let (buffers, flipped) = flipped_terrain(N);
    let mut mesh = build(&buffers);
    let mut tx = mesh.begin_edit();
    let r = orient(&mut tx).unwrap();
    tx.rollback();
    eprintln!(
        "orient_1M_1pct_flipped: {} triangles, {flipped} flipped; report: {r:?}",
        mesh.triangle_count()
    );
    let mut group = c.benchmark_group("ops orient");
    group.sample_size(10);
    group.bench_function("orient_1M_1pct_flipped", |b| {
        b.iter_batched(
            || build(&buffers),
            |mut mesh| {
                let mut tx = mesh.begin_edit();
                black_box(orient(&mut tx).unwrap());
                black_box(tx.commit().unwrap());
                mesh
            },
            BatchSize::PerIteration,
        )
    });
    group.finish();
}

fn bench_fill(c: &mut Criterion) {
    // Radii in grid cells; the loops come out at roughly 50 to 500 edges.
    let radii = [8.0, 62.0, 16.0, 55.0, 30.0, 50.0, 40.0, 60.0, 10.0, 22.0];
    let buffers = holed_terrain(N, &radii);
    let mesh = build(&buffers);
    let loops = boundary_loops(&mesh).unwrap();
    let mut sizes: Vec<usize> = loops.iter().map(|l| l.vertices.len()).collect();
    sizes.sort_unstable();
    eprintln!(
        "fill_10_holes_1M: {} triangles, {} loops of {sizes:?} edges",
        mesh.triangle_count(),
        loops.len()
    );
    let mut group = c.benchmark_group("ops fill_holes");
    group.sample_size(10);
    for (name, fair) in [("fair off", false), ("fair on", true)] {
        let options = FillOptions {
            max_edges: Some(1000),
            fair,
            ..FillOptions::default()
        };
        let mut m = build(&buffers);
        let mut tx = m.begin_edit();
        let r = fill_holes(&mut tx, &options).unwrap();
        tx.rollback();
        let tris: usize = r.filled.iter().map(|f| f.triangles_added).sum();
        let verts: usize = r.filled.iter().map(|f| f.vertices_added).sum();
        eprintln!(
            "fill_10_holes_1M {name}: filled {}, skipped {}, failed {}, {tris} triangles and {verts} vertices added",
            r.filled.len(),
            r.skipped.len(),
            r.failed.len()
        );
        group.bench_function(format!("fill_10_holes_1M {name}"), |b| {
            b.iter_batched(
                || build(&buffers),
                |mut mesh| {
                    let mut tx = mesh.begin_edit();
                    black_box(fill_holes(&mut tx, &options).unwrap());
                    black_box(tx.commit().unwrap());
                    mesh
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

fn bench_weld(c: &mut Criterion) {
    let (buffers, dupes) = duplicated_terrain(N);
    let mut mesh = build(&buffers);
    let mut tx = mesh.begin_edit();
    let r = weld(&mut tx, 0.0).unwrap();
    tx.rollback();
    eprintln!(
        "weld_1M_10pct_dupes: {} triangles, {} vertices ({dupes} duplicates added); report: {r:?}",
        mesh.triangle_count(),
        mesh.vertex_count()
    );
    let mut group = c.benchmark_group("ops weld");
    group.sample_size(10);
    group.bench_function("weld_1M_10pct_dupes", |b| {
        b.iter_batched(
            || build(&buffers),
            |mut mesh| {
                let mut tx = mesh.begin_edit();
                black_box(weld(&mut tx, 0.0).unwrap());
                black_box(tx.commit().unwrap());
                mesh
            },
            BatchSize::PerIteration,
        )
    });
    group.finish();
}

fn bench_components(c: &mut Criterion) {
    let buffers = terrain_with_specks(N, 1000);
    let mut mesh = build(&buffers);
    let options = ComponentOptions {
        min_triangles: Some(10),
        ..ComponentOptions::default()
    };
    let mut tx = mesh.begin_edit();
    let r = remove_small_components(&mut tx, &options).unwrap();
    tx.rollback();
    eprintln!(
        "remove_small_components_1M: {} triangles; report: {r:?}",
        mesh.triangle_count()
    );
    let mut group = c.benchmark_group("ops remove_small_components");
    group.sample_size(10);
    group.bench_function("remove_small_components_1M", |b| {
        b.iter_batched(
            || build(&buffers),
            |mut mesh| {
                let mut tx = mesh.begin_edit();
                black_box(remove_small_components(&mut tx, &options).unwrap());
                black_box(tx.commit().unwrap());
                mesh
            },
            BatchSize::PerIteration,
        )
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_analyze,
    bench_orient,
    bench_fill,
    bench_weld,
    bench_components
);
criterion_main!(benches);
