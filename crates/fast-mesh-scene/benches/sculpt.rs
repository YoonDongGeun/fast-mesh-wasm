//! Sculpt dab benchmarks: what one brush dab costs as the mesh grows while the
//! brush stays the same size, and as the brush grows on the same mesh.
//!
//! `cargo bench -p fast-mesh-scene --bench sculpt` runs terrains of 100k
//! (`terrain(224)`) and 1M (`terrain(708)`) triangles, spacing 1;
//! `FAST_MESH_BENCH_LARGE=1` adds 10M. Each iteration is one dab on an Eager
//! mesh with vertex normals on, inside an open history step, the way a stroke
//! runs, and drains the change report as a renderer would. Dabs walk between a
//! few centers and flip their sign, so the surface stays near where it began;
//! the step is cancelled every few hundred dabs to keep it from growing.
//!
//! `dab/<tool>` is [`Scene::dab`] at radius 8 (about 200 vertices), `dab-r24`
//! at radius 24 (about 1,800). `scan/<tool>` is [`Scene::edit`] with the
//! whole-mesh deform function, at radius 8: what a dab cost before [`Scene::dab`].

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use fast_mesh_scene::ops::deform::{
    Falloff, GrabOptions, InflateOptions, SmoothMethod, SmoothOptions, grab, inflate, smooth,
};
use fast_mesh_scene::{Dab, MeshId, MeshOptions, Scene, SceneOptions};
use fast_mesh_store::Mesh;

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

fn sculpt_scene(n: u32) -> (Scene, MeshId, [f32; 3]) {
    let mut scene = Scene::new(SceneOptions::default());
    let id = scene.add_mesh(terrain(n), MeshOptions::default()).unwrap();
    scene.set_normals(id, true).unwrap();
    let mid = n as f32 / 2.0;
    let z = (mid * 0.05).sin() * (mid * 0.07).cos() * 4.0;
    (scene, id, [mid, mid, z])
}

#[derive(Clone, Copy)]
enum Tool {
    Inflate,
    Smooth,
    Grab,
}

/// Dab number `k` of `tool` around `center`.
fn dab_of(tool: Tool, center: [f32; 3], radius: f32, k: u32) -> Dab {
    let sign = if k.is_multiple_of(2) { 1.0 } else { -1.0 };
    let at = [center[0] + (k % 8) as f32, center[1], center[2]];
    match tool {
        Tool::Inflate => Dab::Inflate(
            InflateOptions::builder()
                .center(at)
                .radius(radius)
                .falloff(Falloff::Smooth)
                .strength(0.05 * sign)
                .build(),
        ),
        Tool::Smooth => Dab::Smooth(
            SmoothOptions::builder()
                .center(at)
                .radius(radius)
                .falloff(Falloff::Smooth)
                .strength(0.5)
                .cycles(2)
                .method(SmoothMethod::Laplacian)
                .build(),
        ),
        Tool::Grab => Dab::Grab(
            GrabOptions::builder()
                .center(at)
                .radius(radius)
                .falloff(Falloff::Smooth)
                .strength(1.0)
                .direction([0.0, 0.0, 0.1 * sign])
                .build(),
        ),
    }
}

fn apply(scene: &mut Scene, id: MeshId, dab: &Dab, scan: bool) {
    if !scan {
        black_box(scene.dab(id, dab).unwrap());
        return;
    }
    match dab {
        Dab::Inflate(o) => {
            black_box(scene.edit(id, |tx| inflate(tx, o)).unwrap());
        }
        Dab::Smooth(o) => {
            black_box(scene.edit(id, |tx| smooth(tx, o)).unwrap());
        }
        Dab::Grab(o) => {
            black_box(scene.edit(id, |tx| grab(tx, o)).unwrap());
        }
    }
}

fn bench_size(c: &mut Criterion, label: &str, n: u32) {
    let (mut scene, id, center) = sculpt_scene(n);
    for radius in [8.0f32, 24.0] {
        let region = scene
            .mesh(id)
            .unwrap()
            .positions()
            .iter()
            .filter(|p| (0..3).map(|k| (p[k] - center[k]).powi(2)).sum::<f32>() < radius * radius)
            .count();
        println!("{label}: {region} vertices under a brush of radius {radius}");
    }
    // The vertex-to-triangles index: the first dab builds it, once.
    scene
        .dab(id, &dab_of(Tool::Grab, center, 8.0, 0))
        .and_then(|_| scene.undo())
        .unwrap();
    let mut group = c.benchmark_group(format!("sculpt/{label}"));
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(4));
    for (name, tool) in [
        ("inflate", Tool::Inflate),
        ("smooth", Tool::Smooth),
        ("grab", Tool::Grab),
    ] {
        for (kind, radius, scan) in [
            ("dab", 8.0, false),
            ("dab-r24", 24.0, false),
            ("scan", 8.0, true),
        ] {
            group.bench_function(format!("{kind}/{name}"), |b| {
                let mut k = 0u32;
                scene.begin_step();
                b.iter(|| {
                    apply(&mut scene, id, &dab_of(tool, center, radius, k), scan);
                    k += 1;
                    black_box(scene.take_changes());
                    if k.is_multiple_of(256) {
                        scene.cancel_step().unwrap();
                        scene.begin_step();
                    }
                });
                scene.cancel_step().unwrap();
                scene.take_changes();
            });
        }
    }
    group.finish();
}

fn benches(c: &mut Criterion) {
    bench_size(c, "100k", 224);
    bench_size(c, "1M", 708);
    if std::env::var_os("FAST_MESH_BENCH_LARGE").is_some() {
        bench_size(c, "10M", 2237);
    }
}

criterion_group!(sculpt, benches);
criterion_main!(sculpt);
