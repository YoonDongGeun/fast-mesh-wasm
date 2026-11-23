//! Extrusion benchmarks on meshes of about a million triangles.
//!
//! `cargo bench -p fast-mesh-ops --bench extrude` (add `--features parallel`
//! for the threaded self-intersection search). Each benchmark prints once what
//! it selects and what came out. Only the operation is timed: each iteration
//! is rolled back outside the clock.
//!
//! - `faces/patch`: a patch of about a thousand triangles of a sphere; what is
//!   left is the scan of the mesh.
//! - `faces/half`: half the sphere along its vertex normals.
//! - `faces/scattered`: every fifth triangle, so about 210,000 regions.
//! - `base/terrain`: a wavy sheet with a 2,828-edge rim, with and without cap
//!   refinement.
//! - `base/dome`: an open hemisphere with a 4,096-edge rim.
//!
//! The `no_check` variants skip the self-intersection check.

use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use fast_mesh_ops::extrude::{
    BaseLevel, ExtrudeBaseOptions, ExtrudeDirection, ExtrudeFacesOptions, extrude_base,
    extrude_faces,
};
use fast_mesh_ops::primitives::{Primitive, UvSphereOptions, generate};
use fast_mesh_store::{Mesh, TriangleId};

fn sphere() -> Mesh {
    let s = generate(&Primitive::UvSphere(UvSphereOptions {
        radius: 1.0,
        segments: 1024,
        rings: 512,
        ..Default::default()
    }))
    .unwrap();
    let flat: Vec<f32> = s.positions.iter().flatten().copied().collect();
    let index: Vec<u32> = s.triangles.iter().flatten().copied().collect();
    Mesh::from_buffers(&flat, &index).unwrap()
}

/// The sphere's triangles whose first corner passes `pick`.
fn picked(mesh: &Mesh, pick: impl Fn([f32; 3]) -> bool) -> Vec<TriangleId> {
    (0..mesh.triangle_slots() as TriangleId)
        .filter(|&t| pick(mesh.position(mesh.triangle(t).unwrap()[0]).unwrap()))
        .collect()
}

/// A wavy open sheet of `n` by `n` cells.
fn terrain(n: u32) -> Mesh {
    let mut flat = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            let (fx, fy) = (f64::from(x), f64::from(y));
            let z = 4.0 * (fx * 0.03).sin() * (fy * 0.05).cos() + 0.2 * (fx * 0.9).sin();
            flat.extend([x as f32, y as f32, z as f32]);
        }
    }
    let mut index = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            index.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
        }
    }
    Mesh::from_buffers(&flat, &index).unwrap()
}

/// The open upper half of a sphere: `rings` rings of `sides` vertices below
/// the pole.
fn dome(rings: u32, sides: u32) -> Mesh {
    let mut flat = vec![0.0f32, 0.0, 1.0];
    for i in 1..=rings {
        let a = std::f64::consts::FRAC_PI_2 * f64::from(i) / f64::from(rings);
        for j in 0..sides {
            let b = std::f64::consts::TAU * f64::from(j) / f64::from(sides);
            flat.extend([a.sin() * b.cos(), a.sin() * b.sin(), a.cos()].map(|c| c as f32));
        }
    }
    let at = |i: u32, j: u32| 1 + (i - 1) * sides + j % sides;
    let mut index = Vec::new();
    for j in 0..sides {
        index.extend([0, at(1, j), at(1, j + 1)]);
        for i in 1..rings {
            let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            index.extend([a, b, c, a, c, d]);
        }
    }
    Mesh::from_buffers(&flat, &index).unwrap()
}

fn bench_faces(c: &mut Criterion) {
    let mut group = c.benchmark_group("extrude_faces");
    group.sample_size(10);
    let mut mesh = sphere();
    let cases: [(&str, Vec<TriangleId>, ExtrudeDirection, f64); 3] = [
        (
            "patch",
            picked(&mesh, |p| p[0] > 0.998),
            ExtrudeDirection::RegionNormal,
            0.1,
        ),
        (
            "half",
            picked(&mesh, |p| p[0] > 0.0),
            ExtrudeDirection::VertexNormals,
            0.1,
        ),
        (
            "scattered",
            (0..mesh.triangle_slots() as TriangleId)
                .filter(|t| t % 5 == 0)
                .collect(),
            ExtrudeDirection::RegionNormal,
            0.001,
        ),
    ];
    for (name, sel, direction, distance) in cases {
        for check in [true, false] {
            let options = ExtrudeFacesOptions {
                distance,
                direction,
                check_self_intersections: check,
                ..Default::default()
            };
            let mut tx = mesh.begin_edit();
            let r = extrude_faces(&mut tx, &sel, &options).unwrap();
            tx.rollback();
            let name = if check {
                format!("{name}_1m")
            } else {
                format!("{name}_1m_no_check")
            };
            eprintln!(
                "extrude_faces/{name}: {} of {} triangles, {} regions -> {} walls, {:?} crossing",
                sel.len(),
                mesh.triangle_count(),
                r.regions,
                r.wall_triangles.len(),
                r.self_intersections.map(|s| s.len())
            );
            group.bench_function(name, |b| {
                b.iter_custom(|iters| {
                    let mut spent = Duration::ZERO;
                    for _ in 0..iters {
                        let mut tx = mesh.begin_edit();
                        let start = Instant::now();
                        extrude_faces(&mut tx, &sel, &options).unwrap();
                        spent += start.elapsed();
                        tx.rollback();
                    }
                    spent
                })
            });
        }
    }
    group.finish();
}

fn bench_base(c: &mut Criterion) {
    let mut group = c.benchmark_group("extrude_base");
    group.sample_size(10);
    let down = ExtrudeBaseOptions::along([0.0, 0.0, -1.0], BaseLevel::Beyond(0.5));
    for (name, mut mesh) in [("terrain_1m", terrain(707)), ("dome_1m", dome(128, 4096))] {
        for (variant, refine, check) in [
            ("", true, true),
            ("_no_refine", false, true),
            ("_no_check", true, false),
        ] {
            let options = ExtrudeBaseOptions {
                refine,
                check_self_intersections: check,
                ..down.clone()
            };
            let mut tx = mesh.begin_edit();
            let r = extrude_base(&mut tx, &options).unwrap();
            tx.rollback();
            let name = format!("{name}{variant}");
            eprintln!(
                "extrude_base/{name}: {} triangles, rim {} -> {} cap triangles, {:?} crossing",
                mesh.triangle_count(),
                r.base_vertices,
                r.cap_triangles,
                r.self_intersections.map(|s| s.len())
            );
            group.bench_function(name, |b| {
                b.iter_custom(|iters| {
                    let mut spent = Duration::ZERO;
                    for _ in 0..iters {
                        let mut tx = mesh.begin_edit();
                        let start = Instant::now();
                        extrude_base(&mut tx, &options).unwrap();
                        spent += start.elapsed();
                        tx.rollback();
                    }
                    spent
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_faces, bench_base);
criterion_main!(benches);
