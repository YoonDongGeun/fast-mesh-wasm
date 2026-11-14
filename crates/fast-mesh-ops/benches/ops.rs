//! Repair benchmarks on synthetic height fields.
//!
//! `cargo bench -p fast-mesh-ops` runs 1M triangles (`terrain(708)`). Each
//! benchmark prints what its input holds and what the operation did once.
//! Operations that write (`orient`, `fill_holes`, `weld`,
//! `remove_small_components`) are timed on a mesh rebuilt from buffers for every
//! iteration, outside the timing, and include the transaction's commit. `analyze`
//! reads only; its self-intersection variant takes a prebuilt BVH. `icp` reads
//! only too, with a prebuilt BVH and points displaced from the same terrain.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_ops::deform::{Brush, Falloff, GrabOptions, grab};
use fast_mesh_ops::remesh::{Region, RemeshOptions, remesh};
use fast_mesh_ops::stretch::{StretchOptions, capture_rest_sizes, split_stretched};
use fast_mesh_ops::{
    AnalyzeOptions, BooleanOp, ComponentOptions, FillOptions, IcpOptions, analyze, apply_boolean,
    boolean_plan, boundary_loops, fill_holes, icp, orient, remove_small_components, sample_surface,
    weld,
};
use fast_mesh_query::Rigid;
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
        if fair {
            let fairing: Vec<_> = r.filled.iter().map(|f| (f.weights, f.converged)).collect();
            eprintln!("fill_10_holes_1M fair on: (weights, converged) per hole: {fairing:?}");
        }
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

fn bench_icp(c: &mut Criterion) {
    let mesh = build(&terrain_buffers(N));
    let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
    // The vertices displaced by the inverse of a known small motion: 0.1 degrees
    // about z (1.2 at the far corner) and a third of a cell along each axis.
    let (s, c0) = 0.002f64.sin_cos();
    let truth = Rigid {
        rotation: [[c0, -s, 0.0], [s, c0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.3, -0.2, 0.1],
    };
    let inverse = truth.inverse();
    let mut group = c.benchmark_group("icp");
    group.sample_size(10);
    for (name, samples, overlap) in [
        ("100k_samples", 100_000, 1.0),
        ("100k_samples_overlap_0.8", 100_000, 0.8),
        ("all_samples", usize::MAX, 1.0),
    ] {
        let r = &inverse.rotation;
        let t = inverse.translation;
        let points: Vec<[f64; 3]> = sample_surface(&mesh, samples)
            .unwrap()
            .iter()
            .map(|p| {
                std::array::from_fn(|i| r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2] + t[i])
            })
            .collect();
        let options = IcpOptions {
            overlap,
            ..IcpOptions::default()
        };
        let r = icp(&points, &mesh, &bvh, &Rigid::IDENTITY, &options).unwrap();
        println!(
            "icp {name}: {} points, {} used, {} iterations, {:?}, rmse {:.2e}, conditioning {:.2e}",
            r.samples, r.inliers, r.iterations, r.status, r.rmse, r.conditioning
        );
        group.bench_function(name, |b| {
            b.iter(|| black_box(icp(&points, &mesh, &bvh, &Rigid::IDENTITY, &options).unwrap()))
        });
    }
    group.finish();
}

/// One stroke over the terrain's middle: 48 grab dabs, nothing split yet.
fn stroked(n: u32) -> (Mesh, Region) {
    let mut mesh = build(&terrain_buffers(n));
    capture_rest_sizes(&mut mesh).unwrap();
    let c = n as f32 / 2.0;
    for k in 0..48 {
        let brush = Brush {
            center: [c + k as f32 / 2.0, c, 0.0],
            radius: 12.0,
            falloff: Falloff::Smooth,
            strength: 2.0,
        };
        let options = GrabOptions {
            brush,
            direction: [0.0, 0.0, 1.0],
            ..Default::default()
        };
        let mut tx = mesh.begin_edit();
        grab(&mut tx, &options).unwrap();
        tx.commit().unwrap();
    }
    let region = Region {
        center: [c, c, 0.0],
        radius: 64.0,
    };
    (mesh, region)
}

/// `split_stretched` against the remesh it replaces, on one stroke. The plan
/// and its writes are timed, then rolled back so every sample starts alike.
fn bench_stretch(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops stretch");
    group.sample_size(10);
    for (name, n) in [("100k", 224), ("1M", N)] {
        let buffers = terrain_buffers(n);
        group.bench_function(format!("capture_rest_sizes_{name}"), |b| {
            b.iter_batched(
                || build(&buffers),
                |mut mesh| {
                    capture_rest_sizes(&mut mesh).unwrap();
                    mesh
                },
                BatchSize::PerIteration,
            )
        });
        let (mut mesh, region) = stroked(n);
        let options = StretchOptions {
            region,
            ..StretchOptions::default()
        };
        let mut tx = mesh.begin_edit();
        let r = split_stretched(&mut tx, &options).unwrap();
        tx.rollback();
        eprintln!(
            "split_stretched_{name}: {} splits, {} remaining",
            r.splits, r.remaining
        );
        group.bench_function(format!("split_stretched_{name}"), |b| {
            b.iter(|| {
                let mut tx = mesh.begin_edit();
                black_box(split_stretched(&mut tx, &options).unwrap());
                tx.rollback();
            })
        });
        let whole = StretchOptions {
            region: Region {
                center: [0.0; 3],
                radius: 1e6,
            },
            limit: 1e6,
            ..StretchOptions::default()
        };
        group.bench_function(format!("split_stretched_nothing_to_do_whole_{name}"), |b| {
            b.iter(|| {
                let mut tx = mesh.begin_edit();
                black_box(split_stretched(&mut tx, &whole).unwrap());
                tx.rollback();
            })
        });
        let live = RemeshOptions {
            region,
            target_length: Some(1.1),
            max_rounds: 2,
            split_only: true,
        };
        let mut tx = mesh.begin_edit();
        let r = remesh(&mut tx, &live).unwrap();
        tx.rollback();
        eprintln!("remesh_split_only_{name}: {r:?}");
        group.bench_function(format!("remesh_split_only_{name}"), |b| {
            b.iter(|| {
                let mut tx = mesh.begin_edit();
                black_box(remesh(&mut tx, &live).unwrap());
                tx.rollback();
            })
        });
    }
    group.finish();
}

/// Closed outward-wound UV sphere of radius `r` (`2 * (stacks - 1) * slices` triangles).
fn sphere(r: f32, stacks: u32, slices: u32) -> Mesh {
    use std::f64::consts::PI;
    let mut positions = Vec::with_capacity(((1 + (stacks - 1) * slices + 1) * 3) as usize);
    positions.extend([0.0, 0.0, r]);
    for i in 1..stacks {
        let theta = PI * f64::from(i) / f64::from(stacks);
        let (rs, rc) = (f64::from(r) * theta.sin(), f64::from(r) * theta.cos());
        for j in 0..slices {
            let phi = 2.0 * PI * f64::from(j) / f64::from(slices);
            positions.extend([(rs * phi.cos()) as f32, (rs * phi.sin()) as f32, rc as f32]);
        }
    }
    let south = (positions.len() / 3) as u32;
    positions.extend([0.0, 0.0, -r]);

    let ring = |i: u32, j: u32| 1 + (i - 1) * slices + j % slices;
    let mut indices = Vec::with_capacity((2 * (stacks - 1) * slices * 3) as usize);
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

fn bench_boolean(c: &mut Criterion) {
    let mut a = sphere(1.0, N, N);
    let b = sphere(1.0, N, N);
    let bvh_a = Bvh::build(&a, &BuildOptions::default()).unwrap();
    let bvh_b = Bvh::build(&b, &BuildOptions::default()).unwrap();
    let xf = Rigid {
        translation: [0.31, 0.17, 0.23],
        ..Rigid::IDENTITY
    };
    let mut group = c.benchmark_group("ops boolean");
    group.sample_size(10);
    for (name, op) in [
        ("boolean_1m_spheres_union", BooleanOp::Union),
        ("boolean_1m_spheres_subtract", BooleanOp::Subtract),
        ("boolean_1m_spheres_intersect", BooleanOp::Intersect),
    ] {
        let plan = boolean_plan(&a, &bvh_a, &b, &bvh_b, &xf, op).unwrap();
        let mut tx = a.begin_edit();
        let r = apply_boolean(&mut tx, &plan).unwrap();
        tx.rollback();
        eprintln!(
            "{name}: {} + {} triangles; report: {r:?}",
            a.triangle_count(),
            b.triangle_count()
        );
        group.bench_function(name, |bencher| {
            bencher.iter(|| {
                let plan = boolean_plan(&a, &bvh_a, &b, &bvh_b, &xf, op).unwrap();
                let mut tx = a.begin_edit();
                black_box(apply_boolean(&mut tx, &plan).unwrap());
                tx.rollback();
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_analyze,
    bench_orient,
    bench_fill,
    bench_weld,
    bench_components,
    bench_icp,
    bench_stretch,
    bench_boolean
);
criterion_main!(benches);
