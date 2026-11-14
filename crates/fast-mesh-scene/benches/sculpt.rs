//! Sculpt dab benchmarks across 100k (`terrain(224)`) and 1M (`terrain(708)`)
//! triangles: every brush (`Inflate` × 3, `Smooth` × 4, `Grab` × 3 + adaptive
//! stroke, `Flatten` × 3, `Crease` × 2, `Clay` × 2, `Twist` × 2), `SurfaceFilter`
//! × 4, `Symmetry` (`none`, `x`, `xy`, `xyz`), `dab_protected_band`,
//! `vertices_in_sphere`, lazy `O(1)` `capture_stroke_state`, and whole-mesh
//! `scan` baselines.
//!
//! - Standard Criterion suite:
//!   `cargo bench -p fast-mesh-scene --bench sculpt`
//! - Fast release median/p95 table (prints Markdown summary for `docs/sculpt-benchmarks.md`):
//!   `$env:FAST_MESH_BENCH_MD="1"; cargo bench -p fast-mesh-scene --bench sculpt`

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::Criterion;
use fast_mesh_scene::ops::sculpt::{
    ClayOptions, ClayProfile, CreaseOptions, Falloff, FlattenMode, FlattenOptions, GrabMethod,
    GrabOptions, InflateMethod, InflateOptions, SmoothMethod, SmoothOptions, SurfaceFilter,
    Symmetry, TwistMethod, TwistOptions, capture_stroke_state as mesh_capture_stroke_state, grab,
    inflate, smooth,
};
use fast_mesh_scene::{Dab, MeshId, MeshOptions, Scene, SceneOptions};
use fast_mesh_store::{Mesh, VertexId};

fn terrain_height(fx: f32, fy: f32) -> f32 {
    (fx * 0.12).cos() * (fy * 0.12).cos() * 1.8 + (fx * 0.65).sin() * (fy * 0.65).cos() * 0.45
}

/// Origin-centered wavy terrain with `2 * n * n` triangles and `(n + 1)^2` vertices.
fn terrain(n: u32) -> Mesh {
    let half = n as f32 * 0.5;
    let mut positions = Vec::with_capacity(((n + 1) * (n + 1) * 3) as usize);
    for y in 0..=n {
        for x in 0..=n {
            let fx = x as f32 - half;
            let fy = y as f32 - half;
            positions.extend([fx, fy, terrain_height(fx, fy)]);
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

fn sculpt_scene(n: u32) -> (Scene, MeshId, [f32; 3], [f32; 3]) {
    let mut scene = Scene::new(SceneOptions::default());
    let id = scene.add_mesh(terrain(n), MeshOptions::default()).unwrap();
    scene.set_normals(id, true).unwrap();
    let center = [0.0, 0.0, terrain_height(0.0, 0.0)];
    let off_axis = [28.0, 28.0, terrain_height(28.0, 28.0)];
    // Warm up cached vertex-to-triangles incidence index before timing dabs.
    scene.ensure_incidence(id).unwrap();
    (scene, id, center, off_axis)
}

#[derive(Clone, Copy)]
enum Case {
    Inflate(InflateMethod, f32, bool),
    Smooth(SmoothMethod, f32, bool),
    Grab(GrabMethod, f32, bool),
    GrabFromStrokeAdaptive,
    Flatten(FlattenMode),
    CreaseGroove,
    CreasePinchOnly,
    Clay(ClayProfile),
    Twist(TwistMethod),
    Filter(SurfaceFilter),
    Sym(Symmetry),
}

impl Case {
    fn uses_off_axis(self) -> bool {
        matches!(self, Case::Sym(_))
    }

    fn needs_stroke(self) -> bool {
        matches!(
            self,
            Case::Inflate(InflateMethod::StrokeNormal, _, _) | Case::GrabFromStrokeAdaptive
        )
    }

    fn is_scan(self) -> bool {
        matches!(
            self,
            Case::Inflate(_, _, true) | Case::Smooth(_, _, true) | Case::Grab(_, _, true)
        )
    }

    fn make_dab(self, base: [f32; 3], k: u32) -> Dab {
        let sign = if k.is_multiple_of(2) { 1.0 } else { -1.0 };
        let at = [base[0] + (k % 8) as f32, base[1], base[2]];
        match self {
            Case::Inflate(method, radius, _) => Dab::Inflate(
                InflateOptions::builder()
                    .center(at)
                    .radius(radius)
                    .falloff(Falloff::Smooth)
                    .strength(0.05 * sign)
                    .method(method)
                    .build(),
            ),
            Case::Smooth(method, radius, _) => Dab::Smooth(
                SmoothOptions::builder()
                    .center(at)
                    .radius(radius)
                    .falloff(Falloff::Smooth)
                    .strength(0.5)
                    .cycles(2)
                    .method(method)
                    .build(),
            ),
            Case::Grab(method, radius, _) => Dab::Grab(
                GrabOptions::builder()
                    .center(at)
                    .radius(radius)
                    .falloff(Falloff::Smooth)
                    .strength(1.0)
                    .direction([0.05 * sign, 0.0, 0.1 * sign])
                    .method(method)
                    .poisson(0.25)
                    .build(),
            ),
            Case::GrabFromStrokeAdaptive => Dab::Grab(
                GrabOptions::builder()
                    .center(base)
                    .radius(8.0)
                    .falloff(Falloff::Smooth)
                    .strength(1.0)
                    .direction([0.35 * sign, 0.15, 0.25 * sign])
                    .method(GrabMethod::Sphere)
                    .from_stroke(true)
                    .adaptive_reach(true)
                    .build(),
            ),
            Case::Flatten(mode) => Dab::Flatten(
                FlattenOptions::builder()
                    .center(at)
                    .radius(8.0)
                    .falloff(Falloff::Smooth)
                    .strength(0.5)
                    .mode(mode)
                    .build(),
            ),
            Case::CreaseGroove => Dab::Crease(
                CreaseOptions::builder()
                    .center(at)
                    .radius(8.0)
                    .falloff(Falloff::Smooth)
                    .strength(0.06 * sign)
                    .pinch(0.5)
                    .build(),
            ),
            Case::CreasePinchOnly => Dab::Crease(
                CreaseOptions::builder()
                    .center(at)
                    .radius(8.0)
                    .falloff(Falloff::Smooth)
                    .strength(0.0)
                    .pinch(0.6)
                    .build(),
            ),
            Case::Clay(profile) => Dab::Clay(
                ClayOptions::builder()
                    .center(at)
                    .radius(8.0)
                    .falloff(Falloff::Smooth)
                    .strength(0.35 * sign)
                    .profile(profile)
                    .tangent(Some([1.0, 0.2, 0.0]))
                    .build(),
            ),
            Case::Twist(method) => Dab::Twist(
                TwistOptions::builder()
                    .center(at)
                    .radius(8.0)
                    .falloff(Falloff::Smooth)
                    .strength(0.3 * sign)
                    .method(method)
                    .build(),
            ),
            Case::Filter(filter) => Dab::Inflate(
                InflateOptions::builder()
                    .center(at)
                    .radius(8.0)
                    .falloff(Falloff::Smooth)
                    .strength(0.05 * sign)
                    .filter(filter)
                    .build(),
            ),
            Case::Sym(sym) => Dab::Inflate(
                InflateOptions::builder()
                    .center(at)
                    .radius(8.0)
                    .falloff(Falloff::Smooth)
                    .strength(0.05 * sign)
                    .symmetry(sym)
                    .build(),
            ),
        }
    }
}

fn all_cases() -> Vec<(&'static str, Case)> {
    vec![
        (
            "inflate/vertex_r8",
            Case::Inflate(InflateMethod::VertexNormal, 8.0, false),
        ),
        (
            "inflate/vertex_r24",
            Case::Inflate(InflateMethod::VertexNormal, 24.0, false),
        ),
        (
            "inflate/average_r8",
            Case::Inflate(InflateMethod::AverageNormal, 8.0, false),
        ),
        (
            "inflate/stroke_r8",
            Case::Inflate(InflateMethod::StrokeNormal, 8.0, false),
        ),
        (
            "scan/inflate_r8",
            Case::Inflate(InflateMethod::VertexNormal, 8.0, true),
        ),
        (
            "smooth/taubin_r8",
            Case::Smooth(SmoothMethod::Taubin, 8.0, false),
        ),
        (
            "smooth/laplacian_r8",
            Case::Smooth(SmoothMethod::Laplacian, 8.0, false),
        ),
        (
            "smooth/laplacian_r24",
            Case::Smooth(SmoothMethod::Laplacian, 24.0, false),
        ),
        (
            "smooth/hc_laplacian_r8",
            Case::Smooth(SmoothMethod::HcLaplacian, 8.0, false),
        ),
        (
            "smooth/cotangent_r8",
            Case::Smooth(SmoothMethod::Cotangent, 8.0, false),
        ),
        (
            "scan/smooth_r8",
            Case::Smooth(SmoothMethod::Laplacian, 8.0, true),
        ),
        ("grab/sphere_r8", Case::Grab(GrabMethod::Sphere, 8.0, false)),
        (
            "grab/sphere_r24",
            Case::Grab(GrabMethod::Sphere, 24.0, false),
        ),
        (
            "grab/geodesic_r8",
            Case::Grab(GrabMethod::Geodesic, 8.0, false),
        ),
        (
            "grab/kelvinlet_r8",
            Case::Grab(GrabMethod::Kelvinlet, 8.0, false),
        ),
        ("grab/stroke_adaptive_r8", Case::GrabFromStrokeAdaptive),
        ("scan/grab_r8", Case::Grab(GrabMethod::Sphere, 8.0, true)),
        ("flatten/both_r8", Case::Flatten(FlattenMode::Both)),
        ("flatten/peaks_r8", Case::Flatten(FlattenMode::Peaks)),
        ("flatten/valleys_r8", Case::Flatten(FlattenMode::Valleys)),
        ("crease/groove_r8", Case::CreaseGroove),
        ("crease/pinch_only_r8", Case::CreasePinchOnly),
        ("clay/round_r8", Case::Clay(ClayProfile::Round)),
        ("clay/strips_r8", Case::Clay(ClayProfile::Strips)),
        ("twist/vortex_r8", Case::Twist(TwistMethod::Vortex)),
        ("twist/kelvinlet_r8", Case::Twist(TwistMethod::Kelvinlet)),
        ("filter/none_r8", Case::Filter(SurfaceFilter::None)),
        (
            "filter/peaks_only_r8",
            Case::Filter(SurfaceFilter::PeaksOnly {
                threshold: 0.01,
                contrast: 0.5,
            }),
        ),
        (
            "filter/valleys_only_r8",
            Case::Filter(SurfaceFilter::ValleysOnly {
                threshold: 0.01,
                contrast: 0.5,
            }),
        ),
        (
            "filter/normal_angle_r8",
            Case::Filter(SurfaceFilter::NormalAngle {
                normal: None,
                max_angle_deg: 45.0,
            }),
        ),
        (
            "symmetry/none_r8",
            Case::Sym(Symmetry {
                x: false,
                y: false,
                z: false,
            }),
        ),
        (
            "symmetry/x_r8",
            Case::Sym(Symmetry {
                x: true,
                y: false,
                z: false,
            }),
        ),
        (
            "symmetry/xy_r8",
            Case::Sym(Symmetry {
                x: true,
                y: true,
                z: false,
            }),
        ),
        (
            "symmetry/xyz_r8",
            Case::Sym(Symmetry {
                x: true,
                y: true,
                z: true,
            }),
        ),
    ]
}

fn run_one(scene: &mut Scene, id: MeshId, dab: &Dab, scan: bool) {
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
        _ => unreachable!(),
    }
}

fn median_and_p95_us(mut samples_ns: Vec<u128>) -> (f64, f64) {
    samples_ns.sort_unstable();
    let mid = samples_ns[samples_ns.len() / 2] as f64 / 1_000.0;
    let p95_idx = ((samples_ns.len() * 95) / 100).min(samples_ns.len() - 1);
    let p95 = samples_ns[p95_idx] as f64 / 1_000.0;
    (mid, p95)
}

fn run_markdown_table() {
    println!("# Sculpt Benchmark Report (Release)");
    for (label, n) in [("100k", 224u32), ("1M", 708u32)] {
        let (mut scene, id, center, off_axis) = sculpt_scene(n);
        let r8 = scene.vertices_in_sphere(id, center, 8.0).unwrap().len();
        let r24 = scene.vertices_in_sphere(id, center, 24.0).unwrap().len();
        let tris = scene.mesh(id).unwrap().triangle_count();
        let verts = scene.mesh(id).unwrap().vertex_count();
        println!(
            "\n## {label} Terrain ({verts} vertices, {tris} triangles; r=8 -> {r8} verts, r=24 -> {r24} verts)\n"
        );
        println!("| Case | Median (µs) | p95 (µs) |");
        println!("| :--- | ---: | ---: |");

        // 1. Lazy O(1) Scene::capture_stroke_state vs whole-mesh sculpt::capture_stroke_state
        {
            let mut samples = Vec::with_capacity(51);
            for _ in 0..10 {
                scene.capture_stroke_state(id, false).unwrap();
            }
            for _ in 0..51 {
                let t0 = Instant::now();
                scene.capture_stroke_state(id, false).unwrap();
                black_box(&scene);
                samples.push(t0.elapsed().as_nanos());
            }
            let (med, p95) = median_and_p95_us(samples);
            println!("| `stroke/scene_capture_o1` | {med:.2} | {p95:.2} |");
        }
        {
            let mut standalone = terrain(n);
            mesh_capture_stroke_state(&mut standalone).unwrap();
            let iters = if n >= 700 { 9 } else { 21 };
            let mut samples = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                mesh_capture_stroke_state(&mut standalone).unwrap();
                black_box(&standalone);
                samples.push(t0.elapsed().as_nanos());
            }
            let (med, p95) = median_and_p95_us(samples);
            println!("| `stroke/whole_mesh_capture` | {med:.2} | {p95:.2} |");
        }

        // 2. All brush, filter, symmetry, and whole-mesh scan cases
        for (name, case) in all_cases() {
            let base = if case.uses_off_axis() {
                off_axis
            } else {
                center
            };
            let scan = case.is_scan();
            let iters: u32 = if scan && n >= 700 { 11 } else { 41 };
            let warmup: u32 = if scan && n >= 700 { 2 } else { 6 };

            scene.begin_step();
            if case.needs_stroke() {
                scene.capture_stroke_state(id, false).unwrap();
            }
            for k in 0..warmup {
                let dab = case.make_dab(base, k);
                run_one(&mut scene, id, &dab, scan);
                black_box(scene.take_changes());
            }
            scene.cancel_step().unwrap();
            scene.take_changes();

            let mut samples = Vec::with_capacity(iters as usize);
            scene.begin_step();
            if case.needs_stroke() {
                scene.capture_stroke_state(id, false).unwrap();
            }
            for k in 0..iters {
                let dab = case.make_dab(base, k);
                let t0 = Instant::now();
                run_one(&mut scene, id, &dab, scan);
                black_box(scene.take_changes());
                samples.push(t0.elapsed().as_nanos());
            }
            scene.cancel_step().unwrap();
            scene.take_changes();

            let (med, p95) = median_and_p95_us(samples);
            println!("| `{name}` | {med:.2} | {p95:.2} |");
        }

        // 3. Soft protection band (`dab_protected_band`, band = 3.0)
        {
            let sphere_verts = scene.vertices_in_sphere(id, center, 10.0).unwrap();
            let positions = scene.mesh(id).unwrap().positions();
            let held: Vec<VertexId> = sphere_verts
                .into_iter()
                .filter(|&v| positions[v as usize][0] < 0.0)
                .collect();
            let zeros = vec![0.0f32; held.len()];
            scene.begin_step();
            for k in 0..6u32 {
                let dab =
                    Case::Inflate(InflateMethod::VertexNormal, 8.0, false).make_dab(center, k);
                black_box(
                    scene
                        .dab_protected_band(id, &dab, &held, &zeros, 3.0)
                        .unwrap(),
                );
                black_box(scene.take_changes());
            }
            scene.cancel_step().unwrap();
            scene.take_changes();

            let mut samples = Vec::with_capacity(41);
            scene.begin_step();
            for k in 0..41u32 {
                let dab =
                    Case::Inflate(InflateMethod::VertexNormal, 8.0, false).make_dab(center, k);
                let t0 = Instant::now();
                black_box(
                    scene
                        .dab_protected_band(id, &dab, &held, &zeros, 3.0)
                        .unwrap(),
                );
                black_box(scene.take_changes());
                samples.push(t0.elapsed().as_nanos());
            }
            scene.cancel_step().unwrap();
            scene.take_changes();
            let (med, p95) = median_and_p95_us(samples);
            println!("| `protect/dab_protected_band_r8` | {med:.2} | {p95:.2} |");
        }

        // 4. BVH `vertices_in_sphere` (`r = 8` and `r = 24`)
        for (qname, radius) in [
            ("query/vertices_in_sphere_r8", 8.0f32),
            ("query/vertices_in_sphere_r24", 24.0f32),
        ] {
            for k in 0..6u32 {
                let at = [center[0] + (k % 8) as f32, center[1], center[2]];
                black_box(scene.vertices_in_sphere(id, at, radius).unwrap());
            }
            let mut samples = Vec::with_capacity(51);
            for k in 0..51u32 {
                let at = [center[0] + (k % 8) as f32, center[1], center[2]];
                let t0 = Instant::now();
                black_box(scene.vertices_in_sphere(id, at, radius).unwrap());
                samples.push(t0.elapsed().as_nanos());
            }
            let (med, p95) = median_and_p95_us(samples);
            println!("| `{qname}` | {med:.2} | {p95:.2} |");
        }
    }
}

fn bench_size(c: &mut Criterion, label: &str, n: u32) {
    let (mut scene, id, center, off_axis) = sculpt_scene(n);
    let mut group = c.benchmark_group(format!("sculpt/{label}"));
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(2));

    for (name, case) in all_cases() {
        let base = if case.uses_off_axis() {
            off_axis
        } else {
            center
        };
        let scan = case.is_scan();
        group.bench_function(name, |b| {
            let mut k = 0u32;
            scene.begin_step();
            if case.needs_stroke() {
                scene.capture_stroke_state(id, false).unwrap();
            }
            b.iter(|| {
                let dab = case.make_dab(base, k);
                run_one(&mut scene, id, &dab, scan);
                k += 1;
                black_box(scene.take_changes());
                if k.is_multiple_of(256) {
                    scene.cancel_step().unwrap();
                    scene.begin_step();
                    if case.needs_stroke() {
                        scene.capture_stroke_state(id, false).unwrap();
                    }
                }
            });
            scene.cancel_step().unwrap();
            scene.take_changes();
        });
    }

    group.bench_function("query/vertices_in_sphere_r8", |b| {
        let mut k = 0u32;
        b.iter(|| {
            let at = [center[0] + (k % 8) as f32, center[1], center[2]];
            k += 1;
            black_box(scene.vertices_in_sphere(id, at, 8.0).unwrap());
        });
    });

    group.finish();
}

fn main() {
    if std::env::var_os("FAST_MESH_BENCH_MD").is_some() {
        run_markdown_table();
        return;
    }
    let mut c = Criterion::default().configure_from_args();
    bench_size(&mut c, "100k", 224);
    bench_size(&mut c, "1M", 708);
    if std::env::var_os("FAST_MESH_BENCH_LARGE").is_some() {
        bench_size(&mut c, "10M", 2237);
    }
    c.final_summary();
}
