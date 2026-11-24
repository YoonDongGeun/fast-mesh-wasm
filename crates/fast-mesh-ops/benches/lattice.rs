//! Lattice infill benchmarks at about 1M input triangles.
//!
//! `cargo bench -p fast-mesh-ops --bench lattice` (add `--features parallel`
//! for the threaded distances, field and marching cubes). Each benchmark
//! prints its grid and what the lattice came out as once.
//!
//! The input is a bumpy, scan-like closed sphere of 1.31M triangles (an
//! icosphere of 8 subdivisions displaced along its normals), 2 across:
//!
//! - `gyroid_shell_2m`: a gyroid 5 cells across in a shell, with walls a tenth
//!   of a cell thick, on a grid of about 2M voxels.
//! - `gyroid_shell_16m`: half the cells and walls, eight times the voxels.
//! - `diamond_open_4m`: an open diamond lattice, about 4M voxels.
//! - `gyroid_cavity_4m`: the cavity of the sphere hollowed first, about 4M
//!   voxels.
//! - `preview_1m`: the preview of `gyroid_shell_16m` on 1M voxels.
//! - `plan`: what `gyroid_shell_16m` plans.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use fast_mesh_ops::append::AppendSource;
use fast_mesh_ops::boolean::{HollowOptions, hollow_mesh};
use fast_mesh_ops::lattice::{
    LatticeKind, LatticeOptions, lattice_fill, lattice_plan, lattice_preview,
};
use fast_mesh_ops::primitives::{IcosphereOptions, icosphere};
use fast_mesh_store::Mesh;

fn bumpy() -> AppendSource {
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
    s
}

fn mesh_of(s: &AppendSource) -> Mesh {
    Mesh::from_buffers(s.positions.as_flattened(), s.triangles.as_flattened()).unwrap()
}

fn fill(mesh: &mut Mesh, options: &LatticeOptions) -> (usize, usize, usize) {
    let mut tx = mesh.begin_edit();
    let r = lattice_fill(&mut tx, options).unwrap();
    tx.commit().unwrap();
    (r.vertices, r.triangles, r.fragments)
}

fn bench_lattice(c: &mut Criterion) {
    let input = bumpy();
    let mut hollow = mesh_of(&input);
    {
        let mut tx = hollow.begin_edit();
        hollow_mesh(
            &mut tx,
            &HollowOptions {
                wall_thickness: 0.06,
                ..HollowOptions::default()
            },
        )
        .unwrap();
        tx.commit().unwrap();
    }
    let hollow = {
        let t = (0..hollow.triangle_slots() as u32).filter_map(|t| hollow.triangle(t));
        AppendSource {
            positions: hollow.positions().to_vec(),
            triangles: t.collect(),
        }
    };
    let max_voxels = 20_000_000;
    let gyroid = |voxels: usize, shell: f32| {
        // 25 voxels to a cell and 2.5 to a wall.
        let voxel = 2.08 / voxels as f32;
        let cell = 25.0 * voxel;
        LatticeOptions::builder()
            .kind(LatticeKind::Gyroid)
            .cell_size(cell)
            .wall_thickness(0.1 * cell)
            .shell_thickness(shell)
            .voxel_size(Some(voxel))
            .max_voxels(max_voxels)
            .build()
    };
    let fine = gyroid(250, 0.03);
    let cases = [
        ("gyroid_shell_2m", &input, gyroid(125, 0.04)),
        ("gyroid_shell_16m", &input, fine),
        (
            "diamond_open_4m",
            &input,
            LatticeOptions {
                kind: LatticeKind::Diamond,
                ..gyroid(156, 0.0)
            },
        ),
        ("gyroid_cavity_4m", &hollow, gyroid(156, 0.0)),
    ];
    let mut group = c.benchmark_group("lattice");
    group.sample_size(10);
    for (name, part, options) in cases {
        let mut mesh = mesh_of(part);
        let plan = lattice_plan(&mesh, &options).unwrap();
        let (vertices, triangles, fragments) = fill(&mut mesh, &options);
        eprintln!(
            "lattice/{name}: {} triangles -> {:?} grid {:?} ({} voxels, about {} MB), {vertices} vertices, {triangles} triangles (estimated {}), {fragments} pieces left out",
            part.triangles.len(),
            plan.region,
            plan.dims,
            plan.voxels,
            plan.bytes >> 20,
            plan.triangles,
        );
        group.bench_function(name, |b| {
            b.iter_batched(
                || mesh_of(part),
                |mut mesh| black_box(fill(&mut mesh, &options)),
                BatchSize::PerIteration,
            )
        });
    }
    let mesh = mesh_of(&input);
    let preview = lattice_preview(&mesh, &fine, 1_000_000).unwrap();
    eprintln!(
        "lattice/preview_1m: grid {:?}, walls {}, cropped {}, {} triangles",
        preview.plan.dims,
        preview.plan.wall_thickness,
        preview.plan.cropped,
        preview.walls.triangles.len()
    );
    group.bench_function("preview_1m", |b| {
        b.iter(|| black_box(lattice_preview(&mesh, &fine, 1_000_000).unwrap()))
    });
    group.bench_function("plan", |b| {
        b.iter(|| black_box(lattice_plan(&mesh, &fine).unwrap()))
    });
    group.finish();
}

criterion_group!(benches, bench_lattice);
criterion_main!(benches);
