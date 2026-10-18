//! Query benchmarks on synthetic height fields.
//!
//! `cargo bench -p fast-mesh-query` runs 1M triangles; set `FAST_MESH_BENCH_LARGE=1`
//! to add 10M. Batch benchmarks run 100,000 queries per iteration with criterion
//! throughput set to the batch size, so results read per ray or per point:
//! one call per query, and the same batches through `raycast_many` and
//! `closest_point_many`. Each
//! size prints the result counts of the region queries once. The mesh-vs-mesh
//! benchmarks place a flipped copy of the terrain over the terrain and turn it
//! by a little more about z at every iteration, cycling through 64 poses; they
//! print their result sizes at three poses. `mesh_distance` on the touching
//! poses is also measured separately on those that intersect and on the near
//! misses. The `slice` groups cut a terrain (at 10M too when large) and a UV
//! sphere with a plane at three positions each, and print the contour and
//! segment counts once.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::{
    ClosestOptions, Plane, Ray, RayOptions, Region, Rigid, Slice, closest_point,
    closest_point_many, intersecting_pairs, intersects, mesh_distance, raycast, raycast_any,
    raycast_many, slice, triangles_in, triangles_near, vertices_in,
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

fn bench_size(c: &mut Criterion, label: &str, n: u32) {
    let mesh = terrain(n);
    let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
    let side = n as f32;

    // Rays start at z = 20 over the terrain and point down with some jitter.
    let mut rng = Lcg(0x5eed);
    let rays: Vec<Ray> = (0..BATCH)
        .map(|_| Ray {
            origin: [rng.next() * side, rng.next() * side, 20.0],
            dir: [rng.next() * 0.4 - 0.2, rng.next() * 0.4 - 0.2, -1.0],
        })
        .collect();
    // Points at random heights around the surface.
    let points: Vec<[f32; 3]> = (0..BATCH)
        .map(|_| {
            [
                rng.next() * side,
                rng.next() * side,
                rng.next() * 10.0 - 5.0,
            ]
        })
        .collect();
    let sphere = Region::Sphere {
        center: [side / 2.0, side / 2.0, 0.0],
        radius: 5.0,
    };

    let (mut tris, mut verts) = (Vec::new(), Vec::new());
    triangles_in(&mesh, &bvh, &sphere, &mut tris).unwrap();
    vertices_in(&mesh, &bvh, &sphere, &mut verts).unwrap();
    eprintln!(
        "{label}: {} triangles; sphere of radius 5 holds {} triangles and {} vertices",
        mesh.triangle_count(),
        tris.len(),
        verts.len()
    );

    let mut group = c.benchmark_group(format!("query {label}"));
    group.sample_size(10);
    group.throughput(Throughput::Elements(BATCH as u64));
    let opts = RayOptions::default();
    group.bench_function("raycast per ray", |b| {
        b.iter(|| {
            for ray in &rays {
                black_box(raycast(&mesh, &bvh, ray, &opts).unwrap());
            }
        })
    });
    group.bench_function("raycast_any per ray", |b| {
        b.iter(|| {
            for ray in &rays {
                black_box(raycast_any(&mesh, &bvh, ray, &opts).unwrap());
            }
        })
    });
    let closest_opts = ClosestOptions::default();
    group.bench_function("closest_point per point", |b| {
        b.iter(|| {
            for &p in &points {
                black_box(closest_point(&mesh, &bvh, p, &closest_opts).unwrap());
            }
        })
    });
    // The same batches through the batch calls, parallel with the
    // `parallel` feature.
    let origins: Vec<[f32; 3]> = rays.iter().map(|r| r.origin).collect();
    let dirs: Vec<[f32; 3]> = rays.iter().map(|r| r.dir).collect();
    let mut hits = vec![None; BATCH];
    group.bench_function("raycast_many per ray", |b| {
        b.iter(|| {
            raycast_many(&mesh, &bvh, &origins, &dirs, &opts, &mut hits).unwrap();
            black_box(&hits);
        })
    });
    let mut nearest = vec![None; BATCH];
    group.bench_function("closest_point_many per point", |b| {
        b.iter(|| {
            closest_point_many(&mesh, &bvh, &points, &closest_opts, &mut nearest).unwrap();
            black_box(&nearest);
        })
    });
    group.throughput(Throughput::Elements(1));
    group.bench_function("triangles_in sphere r=5", |b| {
        b.iter(|| {
            tris.clear();
            black_box(triangles_in(&mesh, &bvh, &sphere, &mut tris).unwrap());
        })
    });
    group.bench_function("vertices_in sphere r=5", |b| {
        b.iter(|| {
            verts.clear();
            black_box(vertices_in(&mesh, &bvh, &sphere, &mut verts).unwrap());
        })
    });
    group.finish();
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
/// found on a grid of sample points of A (at most about 250,000 of them). B's
/// vertex `(u, v)` is placed over `(x, y)` where `(u, side - v)` is `(x, y)`
/// turned back by `deg` about the middle; the gap there is
/// `lift - height(x, y) - height(u, v)`.
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
            // Turned back by `deg`.
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
/// touch A at its closest place and then by `push` further into A.
fn poses(side: f64, push: f64) -> Vec<Rigid> {
    (0..64)
        .map(|i| {
            let deg = f64::from(i) * 0.1;
            pose(side, contact_lift(side, deg) - push, deg)
        })
        .collect()
}

fn bench_pairs(c: &mut Criterion, label: &str, n: u32) {
    let (ma, mb) = (terrain(n), terrain(n));
    let opts = BuildOptions::default();
    let (ba, bb) = (
        Bvh::build(&ma, &opts).unwrap(),
        Bvh::build(&mb, &opts).unwrap(),
    );
    let side = f64::from(n);

    let z = ma.positions().iter().map(|p| f64::from(p[2]));
    let range = z.clone().fold(f64::NEG_INFINITY, f64::max) - z.fold(f64::INFINITY, f64::min);
    let margin = (range * 0.01) as f32;

    // Touching at the closest place (the surface between the sample points
    // can dip a little past it), and pushed 0.5 into A.
    let touching = poses(side, 0.0);
    let pushed = poses(side, 0.5);
    // The same poses lifted 0.3 clear of A: the "slightly apart" case.
    let apart = poses(side, -0.3);
    let unit = |s: &str| format!("query {label} {s}");

    // The touching poses split by whether the meshes really intersect: the
    // sample grid misses the closest place by a little, so some poses are
    // near misses, whose `mesh_distance` falls through to the full search.
    let (hit, miss): (Vec<Rigid>, Vec<Rigid>) = touching
        .iter()
        .partition(|xf| intersects(&ma, &ba, &mb, &bb, xf).unwrap());
    let gaps: Vec<f32> = miss
        .iter()
        .map(|xf| {
            mesh_distance(&ma, &ba, &mb, &bb, xf, f32::INFINITY)
                .unwrap()
                .unwrap()
                .distance
        })
        .collect();
    eprintln!(
        "{label}: touching poses: {} intersect, {} are near misses with gaps {:e} to {:e}",
        hit.len(),
        miss.len(),
        gaps.iter().copied().fold(f32::INFINITY, f32::min),
        gaps.iter().copied().fold(0.0, f32::max),
    );

    // The result sizes, once, at three of the poses.
    for i in [0usize, 31, 63] {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        triangles_near(&ma, &ba, &mb, &bb, &touching[i], margin, &mut a, &mut b).unwrap();
        let d = mesh_distance(&ma, &ba, &mb, &bb, &touching[i], f32::INFINITY).unwrap();
        let mut pairs = Vec::new();
        intersecting_pairs(&ma, &ba, &mb, &bb, &pushed[i], &mut pairs).unwrap();
        let gap = mesh_distance(&ma, &ba, &mb, &bb, &apart[i], f32::INFINITY).unwrap();
        eprintln!(
            "{label} pose {i} ({:.1} deg): touching: intersects {:?}, distance {:?}, near margin \
             {margin:.3} holds {} + {} triangles; pushed 0.5 gives {} pairs; apart 0.3: \
             intersects {:?}, distance {:?}",
            i as f64 * 0.1,
            intersects(&ma, &ba, &mb, &bb, &touching[i]).unwrap(),
            d.map(|d| d.distance),
            a.len(),
            b.len(),
            pairs.len(),
            intersects(&ma, &ba, &mb, &bb, &apart[i]).unwrap(),
            gap.map(|d| d.distance),
        );
    }

    let mut group = c.benchmark_group(unit("pairs"));
    group.sample_size(10);
    group.throughput(Throughput::Elements(1));
    let mut i = 0;
    group.bench_function("intersects", |b| {
        b.iter(|| {
            i = (i + 1) % 64;
            black_box(intersects(&ma, &ba, &mb, &bb, &touching[i]).unwrap());
        })
    });
    group.bench_function("intersects apart 0.3", |b| {
        b.iter(|| {
            i = (i + 1) % 64;
            black_box(intersects(&ma, &ba, &mb, &bb, &apart[i]).unwrap());
        })
    });
    group.bench_function("mesh_distance", |b| {
        b.iter(|| {
            i = (i + 1) % 64;
            black_box(mesh_distance(&ma, &ba, &mb, &bb, &touching[i], f32::INFINITY).unwrap());
        })
    });
    // Each split cycles through its own poses.
    for (name, set) in [
        ("mesh_distance touching, intersecting poses", &hit),
        ("mesh_distance touching, near-miss poses", &miss),
    ] {
        if set.is_empty() {
            continue;
        }
        let mut j = 0;
        group.bench_function(name, |b| {
            b.iter(|| {
                j = (j + 1) % set.len();
                black_box(mesh_distance(&ma, &ba, &mb, &bb, &set[j], f32::INFINITY).unwrap());
            })
        });
    }
    group.bench_function("mesh_distance apart 0.3", |b| {
        b.iter(|| {
            i = (i + 1) % 64;
            black_box(mesh_distance(&ma, &ba, &mb, &bb, &apart[i], f32::INFINITY).unwrap());
        })
    });
    let (mut out_a, mut out_b) = (Vec::new(), Vec::new());
    group.bench_function("triangles_near margin 1% of height", |b| {
        b.iter(|| {
            i = (i + 1) % 64;
            out_a.clear();
            out_b.clear();
            triangles_near(
                &ma,
                &ba,
                &mb,
                &bb,
                &touching[i],
                margin,
                &mut out_a,
                &mut out_b,
            )
            .unwrap();
            black_box((out_a.len(), out_b.len()));
        })
    });
    let mut pairs = Vec::new();
    group.bench_function("intersecting_pairs pushed 0.5", |b| {
        b.iter(|| {
            i = (i + 1) % 64;
            pairs.clear();
            black_box(intersecting_pairs(&ma, &ba, &mb, &bb, &pushed[i], &mut pairs).unwrap());
        })
    });
    group.finish();
}

/// Cuts `mesh` with each plane, after printing what the cut holds.
fn bench_slice(c: &mut Criterion, name: &str, mesh: &Mesh, planes: &[(String, Plane)]) {
    let bvh = Bvh::build(mesh, &BuildOptions::default()).unwrap();
    let mut group = c.benchmark_group(format!("query slice/{name}"));
    group.sample_size(if name.ends_with("10M") { 10 } else { 100 });
    group.throughput(Throughput::Elements(1));
    let mut out = Slice::default();
    for (label, plane) in planes {
        slice(mesh, &bvh, plane, &mut out).unwrap();
        let points: usize = out.contours.iter().map(|c| c.points.len()).sum();
        eprintln!(
            "slice/{name} {label}: {} contours, {points} points, {} closed",
            out.contours.len(),
            out.contours.iter().filter(|c| c.closed).count()
        );
        group.bench_function(label, |b| {
            b.iter(|| {
                slice(mesh, &bvh, plane, &mut out).unwrap();
                black_box(out.contours.len());
            })
        });
    }
    group.finish();
}

/// Three planes across a terrain of side `n`, normal along x, between grid
/// lines.
fn terrain_planes(n: u32) -> Vec<(String, Plane)> {
    [0.2, 0.5, 0.8]
        .map(|f| {
            let x = (f * f64::from(n)).floor() + 0.37;
            (
                format!("x = {x}"),
                Plane::new([x, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap(),
            )
        })
        .into()
}

fn bench_slices(c: &mut Criterion) {
    bench_slice(c, "terrain_1M", &terrain(708), &terrain_planes(708));
    // 998,000 triangles, radius 100: planes through the cap, the middle and
    // below it, tilted so that they miss the pole axis.
    let sphere = uv_sphere(500, 1000, 100.0);
    let planes: Vec<(String, Plane)> = [-50.5, 0.3, 60.7]
        .map(|z| {
            (
                format!("z = {z}"),
                Plane::new([0.0, 0.0, z], [0.1, 0.05, 1.0]).unwrap(),
            )
        })
        .into();
    bench_slice(c, "sphere_1M", &sphere, &planes);
    if std::env::var_os("FAST_MESH_BENCH_LARGE").is_some() {
        bench_slice(c, "terrain_10M", &terrain(2237), &terrain_planes(2237));
    }
}

fn benches(c: &mut Criterion) {
    bench_slices(c);
    bench_size(c, "1M", 708);
    bench_pairs(c, "1M", 708);
    if std::env::var_os("FAST_MESH_BENCH_LARGE").is_some() {
        bench_size(c, "10M", 2237);
        bench_pairs(c, "10M", 2237);
    }
}

criterion_group!(group, benches);
criterion_main!(group);
