//! Primitive generator benchmarks at about a million triangles each.
//!
//! `cargo bench -p fast-mesh-ops --bench primitives` times every shape from
//! its parameters to its finished buffers (positions and triangles), and
//! prints its size once.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fast_mesh_ops::primitives::{
    ArrowOptions, Axis, BoxOptions, ConeOptions, CylinderOptions, IcosphereOptions, PlaneOptions,
    Primitive, TorusOptions, UvSphereOptions, generate,
};

/// Each shape at about 1M triangles.
fn shapes() -> Vec<(&'static str, Primitive)> {
    vec![
        (
            "box_1m",
            Primitive::Box(BoxOptions {
                segments: [289, 289, 289],
                ..Default::default()
            }),
        ),
        (
            "cylinder_1m",
            Primitive::Cylinder(CylinderOptions {
                segments: 1024,
                rings: 512,
                cap_rings: 1,
                axis: Axis::Y,
                ..Default::default()
            }),
        ),
        (
            "cone_1m",
            Primitive::Cone(ConeOptions {
                segments: 1024,
                rings: 511,
                cap_rings: 1,
                ..Default::default()
            }),
        ),
        (
            "uv_sphere_1m",
            Primitive::UvSphere(UvSphereOptions {
                segments: 1024,
                rings: 512,
                ..Default::default()
            }),
        ),
        (
            "icosphere_1.3m",
            Primitive::Icosphere(IcosphereOptions {
                subdivisions: 8,
                ..Default::default()
            }),
        ),
        (
            "torus_1m",
            Primitive::Torus(TorusOptions {
                segments: 1024,
                tube_segments: 512,
                ..Default::default()
            }),
        ),
        (
            "arrow_1m",
            Primitive::Arrow(ArrowOptions {
                segments: 1 << 17,
                ..Default::default()
            }),
        ),
        (
            "plane_1m",
            Primitive::Plane(PlaneOptions {
                segments: [708, 708],
                ..Default::default()
            }),
        ),
    ]
}

fn bench_primitives(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitives");
    for (name, shape) in shapes() {
        let out = generate(&shape).unwrap();
        println!(
            "{name}: {} vertices, {} triangles",
            out.positions.len(),
            out.triangles.len()
        );
        group.throughput(Throughput::Elements(out.triangles.len() as u64));
        group.bench_function(name, |b| b.iter(|| generate(black_box(&shape)).unwrap()));
    }
    group.finish();
}

criterion_group!(benches, bench_primitives);
criterion_main!(benches);
