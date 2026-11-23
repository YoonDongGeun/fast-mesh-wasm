use fast_mesh_store::{Mesh, TriangleId};

use super::*;
use crate::AnalyzeOptions;
use crate::denoise::points;
use crate::test_util::{assert_parallel_identical, edit, mesh_of, parts, sweep, uv_sphere};

/// A deterministic generator, uniform in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
    }
}

fn copy(mesh: &Mesh) -> Mesh {
    let (positions, triangles) = parts(mesh);
    mesh_of(&positions, &triangles)
}

fn normal(mesh: &Mesh, c: [u32; 3]) -> [f64; 3] {
    unit_normal(points(mesh.positions(), c)).unwrap()
}

/// The live triangles of `mesh`.
fn triangles(mesh: &Mesh) -> Vec<[u32; 3]> {
    (0..mesh.triangle_slots() as TriangleId)
        .filter_map(|t| mesh.triangle(t))
        .collect()
}

/// Whether `mesh` has the undirected edge `a`-`b`.
fn has_edge(mesh: &Mesh, a: u32, b: u32) -> bool {
    triangles(mesh)
        .iter()
        .any(|c| c.contains(&a) && c.contains(&b))
}

/// The largest sum of the two angles facing an edge, in degrees, over the
/// edges of two triangles.
fn worst_facing(mesh: &Mesh) -> f64 {
    let tris = triangles(mesh);
    let p = mesh.positions();
    let mut worst: f64 = 0.0;
    for (i, x) in tris.iter().enumerate() {
        for y in &tris[i + 1..] {
            let shared: Vec<u32> = x.iter().copied().filter(|v| y.contains(v)).collect();
            if shared.len() != 2 {
                continue;
            }
            let apex = |c: &[u32; 3]| *c.iter().find(|v| !shared.contains(v)).unwrap();
            let at = |o: u32| {
                let q = points(p, [o, shared[0], shared[1]]);
                angle(sub3(q[1], q[0]), sub3(q[2], q[0]))
            };
            worst = worst.max((at(apex(x)) + at(apex(y))).to_degrees());
        }
    }
    worst
}

fn area(mesh: &Mesh) -> f64 {
    triangles(mesh)
        .iter()
        .map(|&c| {
            let q = points(mesh.positions(), c);
            let n = crate::remesh::cross3(sub3(q[1], q[0]), sub3(q[2], q[0]));
            0.5 * dot3(n, n).sqrt()
        })
        .sum()
}

/// A flat strip of `n` parallelogram cells, sheared by 0.8, each split
/// along its long diagonal: the worst way.
fn strip(n: u32) -> Mesh {
    let mut positions = Vec::new();
    for i in 0..=n {
        positions.extend([i as f32, 0.0, 0.0]);
    }
    for i in 0..=n {
        positions.extend([i as f32 + 0.8, 1.0, 0.0]);
    }
    let (b, t) = (|i: u32| i, |i: u32| n + 1 + i);
    let mut tris = Vec::new();
    for i in 0..n {
        tris.push([b(i), b(i + 1), t(i + 1)]);
        tris.push([b(i), t(i + 1), t(i)]);
    }
    mesh_of(&positions, &tris)
}

#[test]
fn flips_raise_the_minimum_angle_on_a_strip() {
    let mut mesh = strip(12);
    let before = copy(&mesh);
    let (r, empty) = edit(&mut mesh, |tx| {
        delaunay_flips(tx, &DelaunayOptions::default(), None)
    });
    assert!(!empty);
    assert!(r.converged && r.flips >= 12, "{r:?}");
    // From 22.3 degrees (the long diagonals) to 27.4 (the short ones).
    assert!((r.before.min_angle_deg - 22.29).abs() < 0.01, "{r:?}");
    assert!(r.after.min_angle_deg > 27.3, "{r:?}");
    assert_eq!(r.after, crate::surface_stats(&mesh, None).unwrap());
    assert!(worst_facing(&before) > 250.0);
    assert!(
        worst_facing(&mesh) <= 180.0 + 1e-6,
        "{}",
        worst_facing(&mesh)
    );
    // The vertices and the surface are as they were.
    assert_eq!(mesh.positions(), before.positions());
    assert_eq!(mesh.triangle_count(), before.triangle_count());
    assert!((area(&mesh) - area(&before)).abs() < 1e-9);
    let a = crate::analyze(&mesh, &AnalyzeOptions::default()).unwrap();
    assert_eq!(
        (a.non_manifold_edges, a.orientation_conflicts, a.loops.len()),
        (0, 0, 1)
    );
    // A Delaunay strip has nothing left to flip.
    let (r, empty) = edit(&mut mesh, |tx| {
        delaunay_flips(tx, &DelaunayOptions::default(), None)
    });
    assert!(empty && r.flips == 0 && r.converged);
}

/// Two sheared strips meeting at a right-angled crease along the row
/// `m`: `b` and `m` in the plane `z = 0`, `m` and `t` in the plane `y = 1`.
fn roof(n: u32) -> (Mesh, Vec<(u32, u32)>) {
    let mut positions = Vec::new();
    for i in 0..=n {
        positions.extend([i as f32, 0.0, 0.0]);
    }
    for i in 0..=n {
        positions.extend([i as f32 + 0.8, 1.0, 0.0]);
    }
    for i in 0..=n {
        positions.extend([i as f32 + 1.6, 1.0, 1.0]);
    }
    let b = |i: u32| i;
    let m = |i: u32| n + 1 + i;
    let t = |i: u32| 2 * (n + 1) + i;
    let mut tris = Vec::new();
    let mut crease = Vec::new();
    for i in 0..n {
        tris.push([b(i), b(i + 1), m(i + 1)]);
        tris.push([b(i), m(i + 1), m(i)]);
        tris.push([m(i), m(i + 1), t(i + 1)]);
        tris.push([m(i), t(i + 1), t(i)]);
        crease.push((m(i), m(i + 1)));
    }
    (mesh_of(&positions, &tris), crease)
}

#[test]
fn creases_sharper_than_the_limit_never_flip() {
    let (mut mesh, crease) = roof(10);
    let options = DelaunayOptions {
        dihedral_limit_deg: 60.0,
    };
    let (r, _) = edit(&mut mesh, |tx| delaunay_flips(tx, &options, None));
    assert!(r.flips >= 20 && r.after.min_angle_deg > 27.3, "{r:?}");
    for (a, b) in crease {
        assert!(has_edge(&mesh, a, b), "crease edge {a}-{b} flipped");
    }
    // Every triangle still lies in one of the two planes.
    for c in triangles(&mesh) {
        let n = normal(&mesh, c);
        let up = (n[2] - 1.0).abs() < 1e-9;
        let side = (n[1] + 1.0).abs() < 1e-9;
        assert!(up || side, "{c:?} has normal {n:?}");
    }
}

/// A `n x n` grid over `[0, n]^2` with jittered vertices, lifted onto
/// `height`, two triangles per cell.
fn lifted_grid(n: u32, seed: u64, height: impl Fn(f32, f32) -> f32) -> Mesh {
    let mut r = Lcg(seed);
    let mut positions = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            let inner = x > 0 && x < n && y > 0 && y < n;
            let j = if inner { 0.2 } else { 0.0 };
            let (px, py) = (x as f32 + j * r.next(), y as f32 + j * r.next());
            positions.extend([px, py, height(px, py)]);
        }
    }
    let mut tris = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            tris.push([a, a + 1, a + n + 2]);
            tris.push([a, a + n + 2, a + n + 1]);
        }
    }
    mesh_of(&positions, &tris)
}

/// Whether the point `(x, y)` is in the `xy` shadow of triangle `c`.
fn covers(mesh: &Mesh, c: [u32; 3], x: f64, y: f64) -> bool {
    let q = points(mesh.positions(), c);
    let side = |i: usize| {
        let (a, b) = (q[i], q[(i + 1) % 3]);
        (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
    };
    let s = [side(0), side(1), side(2)];
    s.iter().all(|&v| v >= -1e-12) || s.iter().all(|&v| v <= 1e-12)
}

/// The largest angle between a triangle of `mesh` that `original` does not
/// have and any of `original`'s triangles under it, in degrees, sampled over
/// each such triangle's `xy` shadow.
fn worst_drift(original: &Mesh, mesh: &Mesh) -> f64 {
    const N: i32 = 4;
    let old = triangles(original);
    let mut worst: f64 = 0.0;
    for c in triangles(mesh) {
        if old.iter().any(|&o| same_triangle(o, c)) {
            continue;
        }
        let q = points(mesh.positions(), c);
        let n = normal(mesh, c);
        for i in 0..=N {
            for j in 0..=N - i {
                // Pulled a little towards the centroid, off the edges.
                let w = [i, j, N - i - j].map(|x| 0.98 * f64::from(x) / f64::from(N) + 0.02 / 3.0);
                let at = |k: usize| (0..3).map(|m| w[m] * q[m][k]).sum::<f64>();
                for &o in old.iter().filter(|&&o| covers(original, o, at(0), at(1))) {
                    worst = worst.max(angle(n, normal(original, o)).to_degrees());
                }
            }
        }
    }
    worst
}

/// The edges of `original` whose triangles are more than `limit` degrees
/// apart that `mesh` no longer has.
fn lost_sharp_edges(original: &Mesh, mesh: &Mesh, limit: f64) -> usize {
    let old = triangles(original);
    let mut lost = 0;
    for (i, x) in old.iter().enumerate() {
        for y in &old[i + 1..] {
            let shared: Vec<u32> = x.iter().copied().filter(|v| y.contains(v)).collect();
            if shared.len() == 2
                && angle(normal(original, *x), normal(original, *y)).to_degrees() > limit
                && !has_edge(mesh, shared[0], shared[1])
            {
                lost += 1;
            }
        }
    }
    lost
}

#[test]
fn flips_never_change_the_surface_beyond_the_dihedral_limit() {
    let mut flips = 0;
    let mut check = |original: &Mesh, limit: f32, what: &str| {
        let mut mesh = copy(original);
        let options = DelaunayOptions {
            dihedral_limit_deg: limit,
        };
        let (r, _) = edit(&mut mesh, |tx| delaunay_flips(tx, &options, None));
        assert!(r.converged, "{what}: {r:?}");
        assert!(
            r.after.min_angle_deg >= r.before.min_angle_deg,
            "{what}: {r:?}"
        );
        assert_eq!(mesh.positions(), original.positions());
        let drift = worst_drift(original, &mesh);
        assert!(drift <= f64::from(limit) + 1e-6, "{what}: {drift} degrees");
        assert_eq!(
            lost_sharp_edges(original, &mesh, f64::from(limit)),
            0,
            "{what}"
        );
        flips += r.flips;
        r.flips
    };
    // A ridge along x = 6 (two slopes of 0.4, 44 degrees apart) and a
    // gentle swell.
    let ridge = lifted_grid(12, 3, |x, y| 0.4 * (x - 6.0).abs() + 0.2 * (0.5 * y).sin());
    for limit in [5.0, 10.0, 20.0] {
        assert!(check(&ridge, limit, "ridge") > 0);
    }
    // Rough surfaces, stretched so that many flips follow each other: each
    // flip on its own within the limit is not enough here.
    for seed in 1..=6u64 {
        for amp in [0.02f32, 0.05, 0.1, 0.2] {
            let mut r = Lcg(seed * 977);
            let (mut positions, tris) = parts(&lifted_grid(10, seed, |_, _| 0.0));
            for p in positions.chunks_mut(3) {
                p[2] = amp * r.next();
                p[0] *= 2.5;
            }
            let rough = mesh_of(&positions, &tris);
            for limit in [5.0, 10.0, 20.0, 30.0] {
                check(
                    &rough,
                    limit,
                    &format!("seed {seed}, noise {amp}, limit {limit}"),
                );
            }
        }
    }
    assert!(flips > 1000, "{flips}");
}

#[test]
fn closed_surfaces_stay_closed_and_manifold() {
    let sphere = uv_sphere(12, 24, 1.0);
    let mut r = Lcg(5);
    let (mut positions, tris) = parts(&sphere);
    for p in &mut positions {
        *p += 0.02 * r.next();
    }
    let mut mesh = mesh_of(&positions, &tris);
    let options = DelaunayOptions {
        dihedral_limit_deg: 30.0,
    };
    let (rep, _) = edit(&mut mesh, |tx| delaunay_flips(tx, &options, None));
    assert!(rep.flips > 0 && rep.converged);
    let a = crate::analyze(&mesh, &AnalyzeOptions::default()).unwrap();
    assert_eq!(a.non_manifold_edges, 0);
    assert!(a.non_manifold_vertices.is_empty());
    assert_eq!(a.orientation_conflicts, 0);
    assert!(a.loops.is_empty());
    assert_eq!(a.duplicate, 0);
}

#[test]
fn only_edges_inside_the_region_flip() {
    let mut mesh = strip(8);
    let before = triangles(&mesh);
    // The first three cells.
    let region: Vec<TriangleId> = (0..6).collect();
    let (r, _) = edit(&mut mesh, |tx| {
        delaunay_flips(tx, &DelaunayOptions::default(), Some(&region))
    });
    assert!(r.flips >= 3, "{r:?}");
    assert_eq!(r.before.triangles, 6);
    for t in 6..mesh.triangle_slots() as TriangleId {
        assert_eq!(mesh.triangle(t), Some(before[t as usize]), "triangle {t}");
    }
    let (r, empty) = edit(&mut mesh, |tx| {
        delaunay_flips(tx, &DelaunayOptions::default(), Some(&[]))
    });
    assert!(empty && r.flips == 0);
}

#[test]
fn rejects_bad_options() {
    let mut mesh = strip(2);
    for limit in [-1.0f32, 91.0, f32::NAN] {
        let options = DelaunayOptions {
            dihedral_limit_deg: limit,
        };
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            delaunay_flips(&mut tx, &options, None),
            Err(OpsError::InvalidInput(_))
        ));
    }
    let dead = [mesh.triangle_slots() as TriangleId];
    let mut tx = mesh.begin_edit();
    assert!(matches!(
        delaunay_flips(&mut tx, &DelaunayOptions::default(), Some(&dead)),
        Err(OpsError::InvalidInput(_))
    ));
}

#[test]
fn keeps_positions_and_uvs() {
    let mut mesh = strip(4);
    let uvs: Vec<f32> = (0..2 * mesh.vertex_slots())
        .map(|i| i as f32 * 0.1)
        .collect();
    mesh.add_uvs(&uvs).unwrap();
    let positions = mesh.positions().to_vec();
    let (r, _) = edit(&mut mesh, |tx| {
        delaunay_flips(tx, &DelaunayOptions::default(), None)
    });
    assert!(r.flips > 0);
    assert_eq!(mesh.uvs().unwrap(), &uvs[..]);
    assert_eq!(mesh.positions(), &positions[..]);
}

#[test]
fn identical_with_and_without_parallel() {
    // 2 * 100^2 = 20000 triangles, above the parallel threshold.
    let mesh = lifted_grid(100, 9, |x, y| 0.1 * (0.3 * x).sin() * (0.2 * y).cos());
    assert!(mesh.triangle_count() > PAR_MIN);
    assert_parallel_identical("delaunay flips", || {
        let mut m = copy(&mesh);
        let (r, _) = edit(&mut m, |tx| {
            delaunay_flips(tx, &DelaunayOptions::default(), None)
        });
        (parts(&m), format!("{r:?}"))
    });
}

#[test]
fn sweep_keeps_atomicity() {
    let n = sweep(strip(3), |tx| {
        delaunay_flips(tx, &DelaunayOptions::default(), None)?;
        Ok(())
    });
    assert!(n > 0);
}
