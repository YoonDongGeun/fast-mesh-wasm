//! Scene benchmarks.
//!
//! `cargo bench -p fast-mesh-scene` runs the groups below.
//!
//! **Overhead against calling Phase 3 directly**, on `terrain(708)` (1M
//! triangles). A raycast batch of 100,000 rays (criterion throughput is the
//! batch size, so results read per ray) runs on a scene with one mesh at the
//! identity placement and directly on the same mesh and BVH. The mesh-vs-mesh
//! benchmarks put a flipped copy of the terrain over the terrain, as
//! `benches/query.rs` of `fast-mesh-query` does, and turn it by a little more
//! about z at every iteration, cycling through 64 poses that just touch:
//! `mesh_distance` at one fixed pose, and `set_placement` plus `mesh_distance`
//! through the 64 poses. The direct calls get the pose as a `Rigid`. Another
//! benchmark casts the same rays on a scene of both meshes. Two last ones run
//! `find_contacts` (margin 0.05) on the two touching meshes, the owner's
//! per-frame case: at the fixed pose, and with `set_placement` through the 64
//! poses.
//!
//! **Fields**, on the same two touching terrains: a distance field from A to B
//! (`max_distance` 0.08), coloured from contact, with both meshes Lazy so that
//! the steps show apart. Each iteration runs the 64 poses, and each pose
//! calls `set_placement`, `update_fields`, reads `colors` and drains
//! `take_changes`, as a renderer would. The field is on A only, or also on B
//! (a second field from B to A, both meshes coloured). Setup prints the
//! number of recoloured vertices per pose.
//!
//! **Wide drag**: the same drag with `max_distance` 8 and the colour range 0
//! to 6, the 9a JS "wide" case, where nearly every vertex is in range and
//! every pose recomputes almost all of them. At 1M, and at 10M
//! (`terrain(2237)`) with `FAST_MESH_BENCH_10M=1`.
//!
//! **Sections**, on `terrain(708)` and on 1M-triangle UV spheres: a plane
//! dragged across a terrain through 64 poses (and across three terrains of
//! 333k triangles side by side, one section over all three), 100 vertices
//! moved far from the
//! plane (and the same edit on a scene without a section, to compare), the
//! fill of a sphere's cross-section, and the gap between the cross-sections
//! of two spheres.
//!
//! **`find_contacts`** over 2, 8 and 32 small meshes (`terrain(32)`, 2,048
//! triangles each) in a row, each overlapping its neighbour a little.
//!
//! **Many meshes**: six terrains in a row, each overlapping the next by an
//! eighth of its side, with a distance field for every ordered pair (30
//! fields, `max_distance` 0.5) and each mesh coloured by its field to the
//! next. The middle mesh is dragged through 64 poses (ten fields due per
//! pose), as in the field drag: `many_mesh_drag_30k` (`terrain(122)`, the
//! owner's scale), `many_mesh_drag_30k_touching` (the same meshes
//! overlapping by one cell, so neighbours barely touch and only a strip of
//! each is in reach), `many_mesh_drag_30k_apart` (the same meshes a whole
//! side apart, so nothing is in reach) and, with `FAST_MESH_BENCH_LARGE=1`,
//! `many_mesh_drag_1M` (`terrain(708)`; about 40 s per iteration without
//! the `parallel` feature).

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::{Plane, Ray, RayOptions, Rigid, mesh_distance, raycast};
use fast_mesh_scene::{
    ColorRule, Coloring, Colormap, Contact, ContactFilter, DistanceField, MeshId, MeshOptions,
    OutOfRange, Placement, Scene, SceneOptions, SceneRayOptions, SectionDesc, SignMode, UpdateMode,
};
use fast_mesh_store::Mesh;

const BATCH: usize = 100_000;

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

/// A sphere of radius `r` about the origin, outward wound: `rings` bands of
/// `segs` quads between two poles (2·segs·(rings − 1) triangles).
fn uv_sphere(rings: u32, segs: u32, r: f32) -> Mesh {
    let mut positions = vec![0.0, 0.0, r];
    for i in 1..rings {
        let phi = std::f32::consts::PI * i as f32 / rings as f32;
        for j in 0..segs {
            let th = std::f32::consts::TAU * j as f32 / segs as f32;
            positions.extend([
                r * phi.sin() * th.cos(),
                r * phi.sin() * th.sin(),
                r * phi.cos(),
            ]);
        }
    }
    positions.extend([0.0, 0.0, -r]);
    let south = 1 + (rings - 1) * segs;
    let ring = |i: u32, j: u32| 1 + (i - 1) * segs + j % segs;
    let mut indices = Vec::new();
    for j in 0..segs {
        indices.extend([0, ring(1, j), ring(1, j + 1)]);
        indices.extend([south, ring(rings - 1, j + 1), ring(rings - 1, j)]);
    }
    for i in 1..rings - 1 {
        for j in 0..segs {
            let (a, b, c, d) = (
                ring(i, j),
                ring(i, j + 1),
                ring(i + 1, j),
                ring(i + 1, j + 1),
            );
            indices.extend([a, c, b, b, c, d]);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// A deterministic linear congruential generator.
struct Lcg(u64);

impl Lcg {
    /// Uniform in `[0, 1)`.
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// The transform that places B, a copy of the terrain, over A: flipped 180°
/// about x, moved so its sheet faces A's with `lift` between their heights, and
/// then turned by `deg` degrees about z around the middle of the terrain.
fn pose(side: f64, lift: f64, deg: f64) -> Rigid {
    // The flip: (x, y, z) -> (x, -y, -z), then + (0, side, lift).
    let flip = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    let t = [0.0, side, lift];
    let (s, c) = deg.to_radians().sin_cos();
    let rz = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
    let mid = [side / 2.0, side / 2.0, 0.0];
    let rotation = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..3).map(|k| rz[i][k] * flip[k][j]).sum())
    });
    let translation =
        std::array::from_fn(|i| (0..3).map(|k| rz[i][k] * (t[k] - mid[k])).sum::<f64>() + mid[i]);
    Rigid {
        rotation,
        translation,
    }
}

/// The terrain height at `(x, y)`, as `terrain` builds it.
fn height(x: f64, y: f64) -> f64 {
    (x * 0.05).sin() * (y * 0.07).cos() * 4.0
}

/// The `lift` at which B, turned by `deg`, just touches A at its closest place,
/// found on a grid of sample points of A (at most about 250,000 of them).
fn contact_lift(side: f64, deg: f64) -> f64 {
    let (s, c) = deg.to_radians().sin_cos();
    let mid = side / 2.0;
    let step = (side / 500.0).ceil().max(1.0);
    let mut lift = f64::NEG_INFINITY;
    let mut y = 0.0;
    while y <= side {
        let mut x = 0.0;
        while x <= side {
            let (dx, dy) = (x - mid, y - mid);
            let (qx, qy) = (c * dx + s * dy + mid, -s * dx + c * dy + mid);
            let (u, v) = (qx, side - qy);
            if (0.0..=side).contains(&u) && (0.0..=side).contains(&v) {
                lift = lift.max(height(x, y) + height(u, v));
            }
            x += step;
        }
        y += step;
    }
    lift
}

/// 64 poses, turning B by 0.1° more each time (0° to 6.3°), each lifted to
/// touch A at its closest place.
fn poses(side: f64) -> Vec<Rigid> {
    (0..64)
        .map(|i| {
            let deg = f64::from(i) * 0.1;
            pose(side, contact_lift(side, deg), deg)
        })
        .collect()
}

fn placement(xf: &Rigid) -> Placement {
    Placement {
        rotation: xf.rotation,
        translation: xf.translation,
    }
}

/// A scene with a history budget small enough that the placement entries the
/// benchmark records do not pile up.
fn scene() -> Scene {
    Scene::new(SceneOptions {
        history_budget: 1 << 20,
    })
}

fn bench_overhead(c: &mut Criterion, label: &str, n: u32) {
    let side = f64::from(n);
    let mesh_a = terrain(n);
    let opts = BuildOptions::default();
    let bvh_a = Bvh::build(&mesh_a, &opts).unwrap();
    let mesh_b = terrain(n);
    let bvh_b = Bvh::build(&mesh_b, &opts).unwrap();
    let touching = poses(side);

    // One mesh at the identity placement, and two meshes: A and the flipped B.
    let mut one = scene();
    one.add_mesh(terrain(n), MeshOptions::default()).unwrap();
    let mut two = scene();
    let a = two.add_mesh(terrain(n), MeshOptions::default()).unwrap();
    let b = two
        .add_mesh(
            terrain(n),
            MeshOptions {
                placement: placement(&touching[0]),
                ..Default::default()
            },
        )
        .unwrap();

    // Rays start at z = 20 over the terrain and point down with some jitter.
    let mut rng = Lcg(0x5eed);
    let rays: Vec<Ray> = (0..BATCH)
        .map(|_| Ray {
            origin: [rng.next() * side as f32, rng.next() * side as f32, 20.0],
            dir: [rng.next() * 0.4 - 0.2, rng.next() * 0.4 - 0.2, -1.0],
        })
        .collect();
    let world_rays: Vec<([f64; 3], [f64; 3])> = rays
        .iter()
        .map(|r| (r.origin.map(f64::from), r.dir.map(f64::from)))
        .collect();

    let mut group = c.benchmark_group(format!("scene {label}"));
    group.sample_size(10);
    group.throughput(Throughput::Elements(BATCH as u64));
    let ray_opts = RayOptions::default();
    group.bench_function("raycast direct, per ray", |bn| {
        bn.iter(|| {
            for ray in &rays {
                black_box(raycast(&mesh_a, &bvh_a, ray, &ray_opts).unwrap());
            }
        })
    });
    let scene_opts = SceneRayOptions::default();
    group.bench_function("raycast scene of 1 mesh, per ray", |bn| {
        bn.iter(|| {
            for (o, d) in &world_rays {
                black_box(one.raycast(*o, *d, &scene_opts).unwrap());
            }
        })
    });
    group.bench_function("raycast scene of 2 meshes, per ray", |bn| {
        bn.iter(|| {
            for (o, d) in &world_rays {
                black_box(two.raycast(*o, *d, &scene_opts).unwrap());
            }
        })
    });

    group.throughput(Throughput::Elements(1));
    // One fixed pose.
    let fixed = 31;
    two.set_placement(b, placement(&touching[fixed])).unwrap();
    group.bench_function("mesh_distance direct, one pose", |bn| {
        bn.iter(|| {
            black_box(
                mesh_distance(
                    &mesh_a,
                    &bvh_a,
                    &mesh_b,
                    &bvh_b,
                    &touching[fixed],
                    f32::INFINITY,
                )
                .unwrap(),
            );
        })
    });
    group.bench_function("mesh_distance scene, one pose", |bn| {
        bn.iter(|| {
            black_box(two.mesh_distance(a, b, f64::INFINITY).unwrap());
        })
    });
    // All 64 poses in every iteration, so each sample costs the same; the
    // times below are per iteration (64 queries).
    group.throughput(Throughput::Elements(64));
    group.bench_function("mesh_distance direct, all 64 poses", |bn| {
        bn.iter(|| {
            for xf in &touching {
                black_box(
                    mesh_distance(&mesh_a, &bvh_a, &mesh_b, &bvh_b, xf, f32::INFINITY).unwrap(),
                );
            }
        })
    });
    let placements: Vec<Placement> = touching.iter().map(placement).collect();
    group.bench_function("set_placement + mesh_distance scene, all 64 poses", |bn| {
        bn.iter(|| {
            for p in &placements {
                two.set_placement(b, *p).unwrap();
                black_box(two.mesh_distance(a, b, f64::INFINITY).unwrap());
            }
        })
    });
    group.bench_function("set_placement alone, all 64 poses", |bn| {
        bn.iter(|| {
            for p in &placements {
                two.set_placement(b, *p).unwrap();
            }
        })
    });

    // The owner's per-frame case: contacts between two touching 1M meshes.
    let filter = ContactFilter::default();
    let mut out: Vec<Contact> = Vec::new();
    let (mut found, mut ta, mut tb) = (0, 0, 0);
    for p in &placements {
        two.set_placement(b, *p).unwrap();
        out.clear();
        found += two
            .find_contacts(CONTACT_MARGIN, &filter, &mut out)
            .unwrap();
        for c in &out {
            (ta, tb) = (ta + c.triangles_a.len(), tb + c.triangles_b.len());
        }
    }
    eprintln!(
        "find_contacts on two touching {label} meshes, margin {CONTACT_MARGIN}, over 64 poses: \
         {found} contacts, {ta} + {tb} triangles listed"
    );
    two.set_placement(b, placements[fixed]).unwrap();
    out.clear();
    two.find_contacts(CONTACT_MARGIN, &filter, &mut out)
        .unwrap();
    eprintln!(
        "  at the fixed pose: {} contacts, {} + {} triangles listed",
        out.len(),
        out.iter().map(|c| c.triangles_a.len()).sum::<usize>(),
        out.iter().map(|c| c.triangles_b.len()).sum::<usize>()
    );
    group.throughput(Throughput::Elements(1));
    group.bench_function("find_contacts scene of 2 touching meshes, one pose", |bn| {
        bn.iter(|| {
            out.clear();
            black_box(
                two.find_contacts(CONTACT_MARGIN, &filter, &mut out)
                    .unwrap(),
            );
        })
    });
    group.throughput(Throughput::Elements(64));
    group.bench_function("set_placement + find_contacts scene, all 64 poses", |bn| {
        bn.iter(|| {
            for p in &placements {
                two.set_placement(b, *p).unwrap();
                out.clear();
                black_box(
                    two.find_contacts(CONTACT_MARGIN, &filter, &mut out)
                        .unwrap(),
                );
            }
        })
    });
    group.finish();
}

/// The `max_distance` of the field benchmarks.
const FIELD_DISTANCE: f64 = 0.08;

/// The colour range of the field benchmarks: from contact.
const FIELD_RANGE: (f32, f32) = (-0.08, 0.0);

/// The `max_distance` of the wide drag benchmarks, the 9a JS "wide" case.
const WIDE_DISTANCE: f64 = 8.0;

/// The colour range of the wide drag benchmarks, as the 9a JS "wide" case.
const WIDE_RANGE: (f32, f32) = (0.0, 6.0);

/// Two Lazy terrains with B at the first pose, a field on A to B, and on B
/// to A when `both` is true, each with `max_distance` and its mesh coloured
/// over `range`.
fn field_scene(
    n: u32,
    first: &Placement,
    both: bool,
    max_distance: f64,
    range: (f32, f32),
) -> (Scene, MeshId) {
    let lazy = |placement| MeshOptions {
        mode: UpdateMode::Lazy,
        placement,
        ..Default::default()
    };
    let mut scene = scene();
    let a = scene
        .add_mesh(terrain(n), lazy(Placement::IDENTITY))
        .unwrap();
    let b = scene.add_mesh(terrain(n), lazy(*first)).unwrap();
    let rule = ColorRule {
        map: Colormap::Turbo,
        range,
        below: OutOfRange::Clamp,
        above: OutOfRange::Base,
        center: None,
        base: [200, 200, 200, 255],
        nan: [255, 0, 255, 255],
    };
    let sides = if both {
        vec![(a, b), (b, a)]
    } else {
        vec![(a, b)]
    };
    for (on, to) in sides {
        let field = scene
            .add_distance_field(DistanceField {
                name: "d".to_owned(),
                on,
                to,
                max_distance,
                sign: SignMode::Auto,
            })
            .unwrap();
        let rule = rule.clone();
        scene
            .set_coloring(on, Some(Coloring { field, rule }))
            .unwrap();
    }
    scene.update_fields().unwrap();
    (scene, b)
}

fn bench_fields(c: &mut Criterion, label: &str, n: u32) {
    let group = format!("scene fields {label}");
    bench_drag(c, &group, label, n, FIELD_DISTANCE, FIELD_RANGE, |name| {
        format!("{name}: set_placement + update_fields + colors, all 64 poses")
    });
}

/// The field drag of [`bench_fields`] with the wide `max_distance` 8 of the
/// 9a JS "wide" case, where nearly every vertex is in range and every pose
/// recomputes almost all of them.
fn bench_wide_drag(c: &mut Criterion, label: &str, n: u32) {
    let group = format!("scene wide drag {label}");
    bench_drag(
        c,
        &group,
        label,
        n,
        WIDE_DISTANCE,
        WIDE_RANGE,
        str::to_owned,
    );
}

/// Drags B through the 64 poses of [`poses`] over A, with the field on A
/// only and on both meshes (see [`field_scene`]), in the criterion group
/// `group`; `bench_name` names each benchmark from its variant.
fn bench_drag(
    c: &mut Criterion,
    group: &str,
    label: &str,
    n: u32,
    max_distance: f64,
    range: (f32, f32),
    bench_name: impl Fn(&str) -> String,
) {
    let touching = poses(f64::from(n));
    let placements: Vec<Placement> = touching.iter().map(placement).collect();
    let mut group = c.benchmark_group(group);
    group.sample_size(10);
    group.throughput(Throughput::Elements(64));
    for (name, both) in [("field on A", false), ("fields on A and B", true)] {
        let (mut scene, b) = field_scene(n, &placements[0], both, max_distance, range);
        let ids: Vec<MeshId> = scene.ids().collect();
        // Setup: what the 64 poses recolour.
        let (mut dirty, mut worst) = (0usize, 0usize);
        for p in &placements {
            scene.set_placement(b, *p).unwrap();
            scene.update_fields().unwrap();
            let d: usize = scene
                .take_changes()
                .iter()
                .map(|c| c.colors_dirty.as_ref().map_or(0, Vec::len))
                .sum();
            (dirty, worst) = (dirty + d, worst.max(d));
        }
        eprintln!(
            "{name}, {label}, max_distance {max_distance}, over 64 poses: {} vertices \
             recoloured per pose on average (at most {worst})",
            dirty / 64
        );
        group.bench_function(bench_name(name), |bn| {
            bn.iter(|| {
                for p in &placements {
                    scene.set_placement(b, *p).unwrap();
                    scene.update_fields().unwrap();
                    for &m in &ids {
                        black_box(scene.colors(m).unwrap());
                    }
                    black_box(scene.take_changes());
                }
            })
        });
    }
    group.finish();
}

/// The margin of the two-mesh `find_contacts` benchmarks.
const CONTACT_MARGIN: f64 = 0.05;

fn bench_contacts(c: &mut Criterion) {
    // Small meshes in a row, each moved 30 units along x: with a side of 32,
    // every mesh overlaps its neighbours a little and nothing else.
    let (n, step) = (32u32, 30.0);
    let mut group = c.benchmark_group("scene find_contacts");
    group.sample_size(20);
    group.throughput(Throughput::Elements(1));
    for count in [2usize, 8, 32] {
        let mut scene = scene();
        for k in 0..count {
            let mut p = Placement::IDENTITY;
            p.translation[0] = k as f64 * step;
            scene
                .add_mesh(
                    terrain(n),
                    MeshOptions {
                        placement: p,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        let mut out: Vec<Contact> = Vec::new();
        let found = scene
            .find_contacts(0.5, &ContactFilter::default(), &mut out)
            .unwrap();
        let (ta, tb) = out.iter().fold((0, 0), |(a, b), c| {
            (a + c.triangles_a.len(), b + c.triangles_b.len())
        });
        eprintln!(
            "find_contacts over {count} meshes of {} triangles: {found} contacts, \
             {ta} + {tb} triangles listed",
            scene
                .mesh(scene.ids().next().unwrap())
                .unwrap()
                .triangle_count()
        );
        group.bench_function(format!("{count} meshes, margin 0.5"), |bn| {
            bn.iter(|| {
                out.clear();
                black_box(
                    scene
                        .find_contacts(0.5, &ContactFilter::default(), &mut out)
                        .unwrap(),
                );
            })
        });
    }
    group.finish();
}

/// A plane across a sphere at height `z`, tilted so it misses the pole axis.
fn z_plane(z: f64) -> Plane {
    Plane::new([0.0, 0.0, z], [0.1, 0.05, 1.0]).unwrap()
}

/// The heights the sphere benchmarks cycle through.
const SPHERE_ZS: [f64; 3] = [-50.5, 0.3, 60.7];

/// An Eager scene (the default mode) with one mesh.
fn eager_scene(mesh: Mesh) -> (Scene, MeshId) {
    let mut scene = scene();
    let id = scene.add_mesh(mesh, MeshOptions::default()).unwrap();
    (scene, id)
}

fn bench_sections(c: &mut Criterion) {
    let n = 708;
    let side = f64::from(n);
    let mut group = c.benchmark_group("scene sections 1M");
    group.sample_size(10);

    // A plane dragged across the terrain through 64 poses, re-sliced at each
    // (Eager: `set_section_plane` computes).
    let (mut scene, id) = eager_scene(terrain(n));
    let planes: Vec<Plane> = (0..64)
        .map(|i| {
            let x = (i as f64 + 0.5) / 64.0 * side + 0.37;
            Plane::new([x, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap()
        })
        .collect();
    let section = scene
        .add_section(SectionDesc {
            plane: planes[0],
            meshes: vec![id],
            fill: false,
        })
        .unwrap();
    group.throughput(Throughput::Elements(64));
    group.bench_function(
        "section_drag_1M: 64 planes, set_section_plane + read",
        |b| {
            b.iter(|| {
                for p in &planes {
                    scene.set_section_plane(section, *p).unwrap();
                    black_box(scene.section(section, id).unwrap().length);
                }
                black_box(scene.take_changes());
            })
        },
    );

    // The same drag over three terrains of a third of the size side by side
    // in y, one section over all three: three entries due per plane, cut in
    // parallel with the `parallel` feature.
    let third = 408;
    let mut scene3 = self::scene();
    let ids: Vec<MeshId> = (0..3)
        .map(|k| {
            let opts = MeshOptions {
                placement: Placement {
                    translation: [0.0, f64::from(k * (third + 1)), 0.0],
                    ..Placement::IDENTITY
                },
                ..Default::default()
            };
            scene3.add_mesh(terrain(third), opts).unwrap()
        })
        .collect();
    let planes3: Vec<Plane> = (0..64)
        .map(|i| {
            let x = (i as f64 + 0.5) / 64.0 * f64::from(third) + 0.37;
            Plane::new([x, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap()
        })
        .collect();
    let section3 = scene3
        .add_section(SectionDesc {
            plane: planes3[0],
            meshes: ids.clone(),
            fill: false,
        })
        .unwrap();
    group.bench_function(
        "section_drag_3x333k: 64 planes, set_section_plane (3 meshes) + read",
        |b| {
            b.iter(|| {
                for p in &planes3 {
                    scene3.set_section_plane(section3, *p).unwrap();
                    for &m in &ids {
                        black_box(scene3.section(section3, m).unwrap().length);
                    }
                }
                black_box(scene3.take_changes());
            })
        },
    );
    drop(scene3);

    // Moving 100 vertices near the x = 0 edge of the terrain, about 354 from
    // a plane at x = side / 2, up and down by 0.5 in z. The baseline is the
    // same edit on a scene with no section: the difference is what the
    // section costs.
    let moved: Vec<u32> = (0..100).map(|k| (k / 10) * (n + 1) + k % 10).collect();
    let base: Vec<[f32; 3]> = moved
        .iter()
        .map(|&v| {
            let (x, y) = ((v % (n + 1)) as f32, (v / (n + 1)) as f32);
            [x, y, (x * 0.05).sin() * (y * 0.07).cos() * 4.0]
        })
        .collect();
    group.throughput(Throughput::Elements(1));
    for with_section in [false, true] {
        let (mut scene, id) = eager_scene(terrain(n));
        let section = with_section.then(|| {
            scene
                .add_section(SectionDesc {
                    plane: Plane::new([side / 2.0 + 0.37, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap(),
                    meshes: vec![id],
                    fill: false,
                })
                .unwrap()
        });
        let mut up = false;
        let name = if with_section {
            "section_far_edit_1M: move 100 vertices, section at the middle"
        } else {
            "section_far_edit_1M baseline: move 100 vertices, no section"
        };
        group.bench_function(name, |b| {
            b.iter(|| {
                up = !up;
                let to: Vec<[f32; 3]> = base
                    .iter()
                    .map(|p| [p[0], p[1], p[2] + if up { 0.5 } else { 0.0 }])
                    .collect();
                scene.edit(id, |tx| tx.set_positions(&moved, &to)).unwrap();
                black_box(scene.take_changes());
            })
        });
        if let Some(section) = section {
            // The far edits must have left the entry Current, or the
            // benchmark measures recomputes.
            eprintln!(
                "section_far_edit_1M: status after the edits {:?}",
                scene.section_status(section, id).unwrap()
            );
        }
    }

    // Fill: a 998,000-triangle sphere cut at three heights in turn; the
    // `fill: false` run shows what the contours alone cost.
    for fill in [false, true] {
        let (mut scene, id) = eager_scene(uv_sphere(500, 1000, 100.0));
        let section = scene
            .add_section(SectionDesc {
                plane: z_plane(SPHERE_ZS[0]),
                meshes: vec![id],
                fill,
            })
            .unwrap();
        let r = scene.section(section, id).unwrap();
        eprintln!(
            "sphere_1M fill={fill}: {} contours, {} points, fill triangles {:?}, fill_ok {}, \
             area {:.1}",
            r.contours.len(),
            r.contours.iter().map(|c| c.points.len()).sum::<usize>(),
            r.fill.as_ref().map(|f| f.triangles.len()),
            r.fill_ok,
            r.area
        );
        let mut i = 0;
        let name = if fill {
            "section_fill_1M_sphere: set_section_plane, fill on"
        } else {
            "section_fill_1M_sphere baseline: set_section_plane, fill off"
        };
        group.bench_function(name, |b| {
            b.iter(|| {
                i = (i + 1) % 3;
                scene
                    .set_section_plane(section, z_plane(SPHERE_ZS[i]))
                    .unwrap();
                black_box(scene.section(section, id).unwrap().area);
            })
        });
    }

    // Gap: two spheres side by side (centres 230 apart, radius 100, so 30
    // between them), one section over both. The gap on its own, and a plane
    // change that recomputes both entries followed by the gap.
    let mut gap_scene = self::scene();
    let a = gap_scene
        .add_mesh(uv_sphere(500, 1000, 100.0), MeshOptions::default())
        .unwrap();
    let b = gap_scene
        .add_mesh(
            uv_sphere(500, 1000, 100.0),
            MeshOptions {
                placement: Placement {
                    translation: [230.0, 0.0, 0.0],
                    ..Placement::IDENTITY
                },
                ..Default::default()
            },
        )
        .unwrap();
    let section = gap_scene
        .add_section(SectionDesc {
            plane: z_plane(SPHERE_ZS[1]),
            meshes: vec![a, b],
            fill: false,
        })
        .unwrap();
    eprintln!(
        "section_gap_two_1M: {:?}",
        gap_scene.section_gap(section, a, b).unwrap()
    );
    group.bench_function("section_gap_two_1M: section_gap alone", |bn| {
        bn.iter(|| black_box(gap_scene.section_gap(section, a, b).unwrap()))
    });
    let mut i = 0;
    group.bench_function(
        "section_gap_two_1M: set_section_plane (both meshes) + section_gap",
        |bn| {
            bn.iter(|| {
                i = (i + 1) % 3;
                gap_scene
                    .set_section_plane(section, z_plane(SPHERE_ZS[i]))
                    .unwrap();
                black_box(gap_scene.section_gap(section, a, b).unwrap());
            })
        },
    );
    group.finish();
}

/// The `max_distance` of the many-mesh benchmarks.
const MANY_DISTANCE: f64 = 0.5;

/// The mesh the many-mesh benchmarks drag.
const MANY_MOVED: usize = 2;

/// Six Lazy `terrain(n)` meshes in a row along x, each moved `step` from the
/// last, with a distance field for every ordered pair (30 fields) and each
/// mesh coloured by its field to the next by index (the last by its field to
/// the first). With `step` below `n` every mesh overlaps its neighbours,
/// whose sheets cross it along curves; with `step` above `n` they are apart.
/// Returns the scene, the dragged mesh and its 64 placements: it rises and
/// falls by up to 0.25 and slides 0.1 along x over the poses.
fn many_mesh_scene(n: u32, step: f64) -> (Scene, MeshId, Vec<Placement>) {
    let at = |x: f64, z: f64| Placement {
        translation: [x, 0.0, z],
        ..Placement::IDENTITY
    };
    let mut scene = scene();
    let ids: Vec<MeshId> = (0..6)
        .map(|k| {
            let opts = MeshOptions {
                mode: UpdateMode::Lazy,
                placement: at(k as f64 * step, 0.0),
                ..Default::default()
            };
            scene.add_mesh(terrain(n), opts).unwrap()
        })
        .collect();
    let rule = ColorRule {
        map: Colormap::Turbo,
        range: (-0.5, 0.5),
        below: OutOfRange::Clamp,
        above: OutOfRange::Base,
        center: None,
        base: [200, 200, 200, 255],
        nan: [255, 0, 255, 255],
    };
    for (i, &on) in ids.iter().enumerate() {
        for (j, &to) in ids.iter().enumerate() {
            if i == j {
                continue;
            }
            let field = scene
                .add_distance_field(DistanceField {
                    name: format!("d{i}{j}"),
                    on,
                    to,
                    max_distance: MANY_DISTANCE,
                    sign: SignMode::Auto,
                })
                .unwrap();
            if j == (i + 1) % ids.len() {
                let rule = rule.clone();
                scene
                    .set_coloring(on, Some(Coloring { field, rule }))
                    .unwrap();
            }
        }
    }
    scene.update_fields().unwrap();
    let x0 = MANY_MOVED as f64 * step;
    let placements = (0..64)
        .map(|i| {
            let t = f64::from(i) / 64.0;
            at(x0 + 0.1 * t, 0.25 * (std::f64::consts::TAU * t).sin())
        })
        .collect();
    (scene, ids[MANY_MOVED], placements)
}

/// The many-mesh drag: `set_placement` of the middle mesh, `update_fields`
/// (ten fields due: five on the moved mesh, five measured to it), `colors`
/// of every mesh and `take_changes`, through the 64 poses.
fn bench_many_mesh(c: &mut Criterion, name: &str, n: u32, step: f64) {
    let (mut scene, moved, placements) = many_mesh_scene(n, step);
    let ids: Vec<MeshId> = scene.ids().collect();
    let (mut dirty, mut worst) = (0usize, 0usize);
    for p in &placements {
        scene.set_placement(moved, *p).unwrap();
        scene.update_fields().unwrap();
        let d: usize = scene
            .take_changes()
            .iter()
            .map(|c| c.colors_dirty.as_ref().map_or(0, Vec::len))
            .sum();
        (dirty, worst) = (dirty + d, worst.max(d));
    }
    eprintln!(
        "{name}: 6 x {} triangles, step {step}, 30 fields, max_distance {MANY_DISTANCE}, over 64 \
         poses: {} vertices recoloured per pose on average (at most {worst})",
        scene.mesh(moved).unwrap().triangle_count(),
        dirty / 64
    );
    let mut group = c.benchmark_group("scene many meshes");
    // More samples at the owner's scale, where on and off are compared
    // within 5%.
    group.sample_size(if n <= 122 { 50 } else { 10 });
    group.throughput(Throughput::Elements(64));
    group.bench_function(
        format!("{name}: set_placement + update_fields + colors, all 64 poses"),
        |bn| {
            bn.iter(|| {
                for p in &placements {
                    scene.set_placement(moved, *p).unwrap();
                    scene.update_fields().unwrap();
                    for &m in &ids {
                        black_box(scene.colors(m).unwrap());
                    }
                    black_box(scene.take_changes());
                }
            })
        },
    );
    group.finish();
}

fn benches(c: &mut Criterion) {
    bench_sections(c);
    bench_overhead(c, "1M", 708);
    bench_fields(c, "1M", 708);
    bench_wide_drag(c, "1M", 708);
    if std::env::var_os("FAST_MESH_BENCH_10M").is_some() {
        // 2237: 10,008,338 triangles.
        bench_wide_drag(c, "10M", 2237);
    }
    bench_contacts(c);
    // 122: 29,768 triangles per mesh; 708: 1,002,528. Neighbours overlap by
    // an eighth of the side, by one cell, or are a whole side apart.
    bench_many_mesh(c, "many_mesh_drag_30k", 122, 122.0 * 7.0 / 8.0);
    bench_many_mesh(c, "many_mesh_drag_30k_touching", 122, 121.0);
    bench_many_mesh(c, "many_mesh_drag_30k_apart", 122, 244.0);
    if std::env::var_os("FAST_MESH_BENCH_LARGE").is_some() {
        bench_many_mesh(c, "many_mesh_drag_1M", 708, 708.0 * 7.0 / 8.0);
    }
}

criterion_group!(group, benches);
criterion_main!(group);
