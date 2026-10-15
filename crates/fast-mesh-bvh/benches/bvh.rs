//! Build and update benchmarks on synthetic meshes.
//!
//! `cargo bench -p fast-mesh-bvh` runs 1M triangles; set `FAST_MESH_BENCH_LARGE=1`
//! to add 10M. Each size prints the BVH's node count, depth and memory once.
//!
//! Update benchmarks measure the steady state of an interactive session: one mesh
//! and one BVH are edited over and over. The first edit after loading a mesh is
//! slower, because the store's and the BVH's arrays grow once; that one-time
//! cost is printed separately.

use std::hint::black_box;
use std::time::Instant;

use criterion::{Criterion, criterion_group, criterion_main};
use fast_mesh_bvh::{BuildOptions, Bvh, Quality};
use fast_mesh_store::Mesh;

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

/// Ids of the vertices in a square patch of `side`×`side` grid points.
fn patch(n: u32, side: u32) -> Vec<u32> {
    let start = (n / 2 - side / 2) * (n + 1) + n / 2 - side / 2;
    (0..side)
        .flat_map(|dy| (0..side).map(move |dx| start + dy * (n + 1) + dx))
        .collect()
}

/// Moves `ids` up or down by 0.5 and updates the BVH.
fn move_patch(m: &mut Mesh, bvh: &mut Bvh, ids: &[u32], up: bool) {
    let dz = if up { 0.5 } else { -0.5 };
    let moved: Vec<[f32; 3]> = ids
        .iter()
        .map(|&v| {
            let p = m.position(v).unwrap();
            [p[0], p[1], p[2] + dz]
        })
        .collect();
    let mut tx = m.begin_edit();
    tx.set_positions(ids, &moved).unwrap();
    let patch = tx.commit().unwrap();
    black_box(bvh.update(m, patch.changes()).unwrap());
}

/// Removes 200 alive triangles starting near `first` and adds them back.
fn replace_triangles(m: &mut Mesh, bvh: &mut Bvh, first: u32) {
    let doomed: Vec<u32> = (first..m.triangle_slots() as u32)
        .filter(|&t| m.is_triangle_alive(t))
        .take(200)
        .collect();
    let replacements: Vec<[u32; 3]> = doomed.iter().map(|&t| m.triangle(t).unwrap()).collect();
    let mut tx = m.begin_edit();
    tx.remove_triangles(&doomed).unwrap();
    tx.add_triangles(&replacements).unwrap();
    let patch = tx.commit().unwrap();
    black_box(bvh.update(m, patch.changes()).unwrap());
}

fn bench_size(c: &mut Criterion, label: &str, n: u32) {
    let mesh = terrain(n);
    let sah = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
    eprintln!(
        "{label}: {} triangles, {} nodes, depth {}, {} treelets, {:.1} MB",
        mesh.triangle_count(),
        sah.node_count(),
        sah.depth(),
        sah.treelet_count(),
        sah.byte_size() as f64 / 1e6
    );
    {
        let mut m = terrain(n);
        let mut bvh = Bvh::build(&m, &BuildOptions::default()).unwrap();
        let start = Instant::now();
        replace_triangles(&mut m, &mut bvh, (n / 2) * n * 2);
        eprintln!(
            "{label}: first topology edit after loading took {:?} (one-time growth)",
            start.elapsed()
        );
    }

    let mut group = c.benchmark_group(label);
    group.sample_size(10);
    for quality in [Quality::Sah, Quality::Fast] {
        let opts = BuildOptions {
            quality,
            ..Default::default()
        };
        group.bench_function(format!("build {quality:?}"), |b| {
            b.iter(|| black_box(Bvh::build(&mesh, &opts).unwrap()))
        });
    }

    let mut m = terrain(n);
    m.ensure_incidence().unwrap();
    let mut bvh = Bvh::build(&m, &BuildOptions::default()).unwrap();
    for side in [22u32, 224] {
        let ids = patch(n, side);
        let mut up = true;
        group.bench_function(format!("move {} vertices + update", ids.len()), |b| {
            b.iter(|| {
                move_patch(&mut m, &mut bvh, &ids, up);
                up = !up;
            })
        });
    }
    let mut first = (n / 2) * n * 2;
    group.bench_function("remove + add 200 triangles + update", |b| {
        b.iter(|| {
            replace_triangles(&mut m, &mut bvh, first);
            first = (first + 997) % (n * n * 2);
        })
    });
    // Mixed workload: each move follows a topology edit, so the incidence index
    // must follow the edit rather than be rebuilt.
    let ids = patch(n, 22);
    let mut up = true;
    group.bench_function("remove + add 200 triangles, then move 484 vertices", |b| {
        b.iter(|| {
            replace_triangles(&mut m, &mut bvh, first);
            first = (first + 997) % (n * n * 2);
            move_patch(&mut m, &mut bvh, &ids, up);
            up = !up;
        })
    });
    group.finish();
    assert!(
        bvh.validate(&m).is_ok(),
        "the benchmarked BVH must stay valid"
    );
}

fn benches(c: &mut Criterion) {
    bench_size(c, "1M", 708);
    if std::env::var_os("FAST_MESH_BENCH_LARGE").is_some() {
        bench_size(c, "10M", 2237);
    }
}

criterion_group!(bvh, benches);
criterion_main!(bvh);
