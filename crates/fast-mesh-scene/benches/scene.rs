//! Scene benchmarks.
//!
//! `cargo bench -p fast-mesh-scene` runs two groups.
//!
//! **Overhead against calling Phase 3 directly**, on `terrain(708)` (1M
//! triangles). A raycast batch of 100,000 rays (criterion throughput is the
//! batch size, so results read per ray) runs on a scene with one mesh at the
//! identity placement and directly on the same mesh and BVH. The mesh-vs-mesh
//! benchmarks put a flipped copy of the terrain over the terrain, as
//! `benches/query.rs` of `fast-mesh-query` does, and turn it by a little more
//! about z at every iteration, cycling through 64 poses that just touch:
//! `mesh_distance` at one fixed pose, and `set_placement` plus `mesh_distance`
//! through the 64 poses. The direct calls get the pose as a `Rigid`. A last
//! benchmark casts the same rays on a scene of both meshes.
//!
//! **`find_contacts`** over 2, 8 and 32 small meshes (`terrain(32)`, 2,048
//! triangles each) in a row, each overlapping its neighbour a little.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::{Ray, RayOptions, Rigid, mesh_distance, raycast};
use fast_mesh_scene::{
    Contact, ContactFilter, MeshOptions, Placement, Scene, SceneOptions, SceneRayOptions,
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
    // 64 poses.
    let mut i = 0;
    group.bench_function("mesh_distance direct, 64 poses", |bn| {
        bn.iter(|| {
            i = (i + 1) % 64;
            black_box(
                mesh_distance(
                    &mesh_a,
                    &bvh_a,
                    &mesh_b,
                    &bvh_b,
                    &touching[i],
                    f32::INFINITY,
                )
                .unwrap(),
            );
        })
    });
    let placements: Vec<Placement> = touching.iter().map(placement).collect();
    group.bench_function("set_placement + mesh_distance scene, 64 poses", |bn| {
        bn.iter(|| {
            i = (i + 1) % 64;
            two.set_placement(b, placements[i]).unwrap();
            black_box(two.mesh_distance(a, b, f64::INFINITY).unwrap());
        })
    });
    group.bench_function("set_placement alone", |bn| {
        bn.iter(|| {
            i = (i + 1) % 64;
            two.set_placement(b, placements[i]).unwrap();
        })
    });
    group.finish();
}

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

fn benches(c: &mut Criterion) {
    bench_overhead(c, "1M", 708);
    bench_contacts(c);
}

criterion_group!(group, benches);
criterion_main!(group);
