//! Heat-method geodesics at about a million triangles.
//!
//! `cargo bench -p fast-mesh-scene --bench geodesic` times the parts of a
//! first query on its own (assembling the operators, building the multigrid
//! levels, the diffusion, the gradient pass and the Poisson solve), then
//! whole queries through a scene: the first on a mesh, a second on the same
//! mesh (which reuses the operators), and one from 16 sources; the graph
//! method for comparison. With `--features parallel`, on rayon's pool.
//!
//! The meshes are an icosphere (1.3M triangles), the same with jittered
//! vertices and everything in a scrambled order, as a scan might come, and a
//! UV sphere (1.0M triangles) queried from its equator, whose needles at the
//! poles are the hard case for the diffusion. Two more have to be cut into
//! other triangles first: the icosphere drawn out twenty times, queried from
//! its tip, and a cylinder whose caps are fans (1.0M triangles). Their sizes,
//! the memory the operators take and the error against the great circles (or
//! the meridians) are printed once.

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use fast_mesh_fields::{GeodesicPhase, GeodesicQuery};
use fast_mesh_ops::primitives::{IcosphereOptions, Primitive, UvSphereOptions, generate};
use fast_mesh_scene::{GeodesicMethod, GeodesicOptions, MeshId, MeshOptions, Scene, SceneOptions};
use fast_mesh_store::Mesh;

const HEAT: GeodesicOptions = GeodesicOptions {
    method: GeodesicMethod::HeatMethod,
    max_distance: None,
    time_step_scale: 1.0,
};

fn primitive(shape: &Primitive) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let s = generate(shape).unwrap();
    (s.positions, s.triangles)
}

fn icosphere() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    primitive(&Primitive::Icosphere(IcosphereOptions {
        radius: 1.0,
        subdivisions: 8,
    }))
}

/// The icosphere with jittered vertices, vertices and triangles scrambled.
fn scan() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let (p, mut t) = icosphere();
    let mut state = 99u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };
    let n = p.len();
    let mut to: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        to.swap(i, next() as usize % (i + 1));
    }
    let mut out = vec![[0.0f32; 3]; n];
    for (v, &k) in to.iter().enumerate() {
        out[k as usize] = p[v].map(|c| c + (next() % 1000) as f32 * 1.4e-6);
    }
    for tri in &mut t {
        *tri = tri.map(|v| to[v as usize]);
    }
    for i in (1..t.len()).rev() {
        t.swap(i, next() as usize % (i + 1));
    }
    (out, t)
}

fn uv_sphere() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    primitive(&Primitive::UvSphere(UvSphereOptions {
        radius: 1.0,
        segments: 1024,
        rings: 512,
        ..Default::default()
    }))
}

/// The icosphere, 20 times as long along Y.
fn stretched() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let (mut p, t) = icosphere();
    for v in &mut p {
        v[1] *= 20.0;
    }
    (p, t)
}

/// A cylinder along Z whose caps are fans around one vertex each.
fn fan_cylinder() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let (segments, rings) = (1024u32, 488u32);
    let mut p = Vec::new();
    let mut t = Vec::new();
    for k in 0..=rings {
        for s in 0..segments {
            let a = s as f32 / segments as f32 * std::f32::consts::TAU;
            p.push([a.cos(), a.sin(), 3.0 * k as f32 / rings as f32]);
        }
    }
    let (low, high) = (p.len() as u32, p.len() as u32 + 1);
    p.extend([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]);
    for s in 0..segments {
        let n = (s + 1) % segments;
        for k in 0..rings {
            let (a, b) = (k * segments + s, k * segments + n);
            t.push([a, b, b + segments]);
            t.push([a, b + segments, a + segments]);
        }
        t.push([low, n, s]);
        t.push([high, rings * segments + s, rings * segments + n]);
    }
    (p, t)
}

/// Mean relative error against the meridians of the stretched sphere from
/// its tip, beyond a tenth of the way to the other one.
fn meridian_error(m: &Mesh, distances: &[f32]) -> f64 {
    // The length of a meridian up to so many steps from the tip.
    let steps = 100_000;
    let angle = |k: f64| k * std::f64::consts::PI / f64::from(steps);
    let mut length = vec![0.0];
    for k in 0..steps {
        let a = angle(f64::from(k) + 0.5);
        let step = angle(1.0) * a.cos().hypot(20.0 * a.sin());
        length.push(length[k as usize] + step);
    }
    let far = length[steps as usize];
    let (mut sum, mut count) = (0.0, 0);
    for (p, &d) in m.positions().iter().zip(distances) {
        let [x, y, z] = p.map(f64::from);
        let at = x.hypot(z).atan2(y / 20.0) / angle(1.0);
        let k = (at as usize).min(steps as usize - 1);
        let exact = length[k] + (at - k as f64) * (length[k + 1] - length[k]);
        if exact > 0.1 * far {
            sum += (f64::from(d) - exact).abs() / exact;
            count += 1;
        }
    }
    sum / f64::from(count)
}

/// Mean relative error against the great circles, from a quarter radian on.
fn sphere_error(m: &Mesh, source: u32, distances: &[f32]) -> f64 {
    let at = |v: usize| m.positions()[v].map(f64::from);
    let a = at(source as usize);
    let (mut sum, mut count) = (0.0, 0);
    for (v, &d) in distances.iter().enumerate() {
        let b = at(v);
        let cross = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let sin = cross.iter().map(|c| c * c).sum::<f64>().sqrt();
        let exact = sin.atan2(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
        if exact > 0.25 {
            sum += (f64::from(d) - exact).abs() / exact;
            count += 1;
        }
    }
    sum / f64::from(count)
}

/// Runs a whole first query; how long it spent in `phase`.
fn time_in(mesh: &Mesh, source: u32, phase: GeodesicPhase) -> Duration {
    let mut q = GeodesicQuery::new(mesh, &[source], &HEAT).unwrap();
    let mut spent = Duration::ZERO;
    loop {
        let now = q.phase();
        let start = Instant::now();
        let done = q.step(mesh, 1).unwrap();
        if now == phase {
            spent += start.elapsed();
        }
        if done {
            return spent;
        }
    }
}

fn query(scene: &Scene, id: MeshId, sources: &[u32], options: &GeodesicOptions) -> Vec<f32> {
    let mut out = Vec::new();
    scene
        .geodesic_distances(id, sources, options, &mut out)
        .unwrap();
    out
}

fn geodesics(c: &mut Criterion) {
    for (name, (p, t)) in [
        ("icosphere_1.3m", icosphere()),
        ("scan_1.3m", scan()),
        ("uv_sphere_1m", uv_sphere()),
        ("stretched_1.3m", stretched()),
        ("fan_cylinder_1m", fan_cylinder()),
    ] {
        let mesh = Mesh::from_buffers(p.as_flattened(), t.as_flattened()).unwrap();
        let n = mesh.vertex_slots() as u32;
        // (On the UV sphere: on the equator.)
        let mut source = n / 2;
        if name.starts_with("stretched") {
            let y = |v: &u32| p[*v as usize][1];
            source = (0..n).max_by(|a, b| y(a).total_cmp(&y(b))).unwrap();
        }

        let mut q = GeodesicQuery::new(&mesh, &[source], &HEAT).unwrap();
        while !q.step(&mesh, usize::MAX).unwrap() {}
        let d = q.distances().unwrap();
        let error = match name {
            "stretched_1.3m" => format!("{:.2} %", 100.0 * meridian_error(&mesh, d)),
            "fan_cylinder_1m" => "not known".into(),
            _ => format!("{:.2} %", 100.0 * sphere_error(&mesh, source, d)),
        };
        println!(
            "{name}: {} triangles, {n} vertices; operators {} MB; mean error {error}",
            mesh.triangle_count(),
            q.solver().map_or(0, |s| s.byte_size()) / 1_000_000,
        );
        drop(q);

        let mut group = c.benchmark_group(format!("geodesic/{name}"));
        group.sample_size(10);
        for (part, phase) in [
            ("assemble", GeodesicPhase::Assemble),
            ("factorize", GeodesicPhase::Factorize),
            ("diffuse", GeodesicPhase::Diffuse),
            ("gradient", GeodesicPhase::Gradient),
            ("poisson", GeodesicPhase::Poisson),
        ] {
            group.bench_function(part, |b| {
                b.iter_custom(|iterations| {
                    (0..iterations).map(|_| time_in(&mesh, source, phase)).sum()
                })
            });
        }

        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(mesh, MeshOptions::default()).unwrap();
        group.bench_function("first_query", |b| {
            b.iter(|| {
                scene.release_geodesics();
                black_box(query(&scene, id, &[source], &HEAT))
            })
        });
        query(&scene, id, &[source], &HEAT);
        group.bench_function("second_query", |b| {
            b.iter(|| black_box(query(&scene, id, &[source / 2], &HEAT)))
        });
        let many: Vec<u32> = (0..16).map(|k| n / 17 * k).collect();
        group.bench_function("second_query_16_sources", |b| {
            b.iter(|| black_box(query(&scene, id, &many, &HEAT)))
        });
        let graph = GeodesicOptions {
            method: GeodesicMethod::Dijkstra,
            ..HEAT
        };
        group.bench_function("graph_query", |b| {
            b.iter(|| black_box(query(&scene, id, &[source], &graph)))
        });
        group.finish();
    }
}

criterion_group!(benches, geodesics);
criterion_main!(benches);
