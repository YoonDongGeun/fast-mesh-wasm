use fast_mesh_store::{Mesh, TriangleId, VertexId};

use super::*;
use crate::primitives::{BoxOptions, Primitive, generate};
use crate::test_util::{assert_parallel_identical, edit, grid, mesh_of, parts, sweep};

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

/// The cube `[-1, 1]^3` with `n` cells along each edge.
fn cube(n: u32) -> Mesh {
    let src = generate(&Primitive::Box(BoxOptions {
        size: [2.0; 3],
        segments: [n; 3],
    }))
    .unwrap();
    let flat: Vec<f32> = src.positions.iter().flatten().copied().collect();
    mesh_of(&flat, &src.triangles)
}

/// `mesh` with every vertex moved by up to `amp` along each axis.
fn noisy(mesh: &Mesh, amp: f32, seed: u64) -> Mesh {
    let mut r = Lcg(seed);
    let (mut positions, triangles) = parts(mesh);
    for p in &mut positions {
        *p += amp * r.next();
    }
    mesh_of(&positions, &triangles)
}

/// A copy of `mesh`'s positions and triangles.
fn copy(mesh: &Mesh) -> Mesh {
    let (positions, triangles) = parts(mesh);
    mesh_of(&positions, &triangles)
}

fn normal(mesh: &Mesh, t: TriangleId) -> [f64; 3] {
    let c = mesh.triangle(t).unwrap();
    unit_normal(points(mesh.positions(), c)).unwrap()
}

fn angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    angle(a, b).to_degrees()
}

/// The mean and largest angle between each triangle's normal and its
/// normal on `clean`, and the mean dihedral angle at `clean`'s crease edges.
fn cube_errors(mesh: &Mesh, clean: &Mesh) -> (f64, f64, f64) {
    let mut sum = 0.0;
    let mut max: f64 = 0.0;
    let n = mesh.triangle_slots();
    for t in 0..n as TriangleId {
        let e = angle_deg(normal(mesh, t), normal(clean, t));
        sum += e;
        max = max.max(e);
    }
    let corners = corners_of(clean).unwrap();
    let fans = VertexFans::build(clean).unwrap();
    let tw = twins(&corners, &fans).unwrap();
    let mut crease = 0.0;
    let mut count = 0;
    for (h, &other) in tw.iter().enumerate() {
        let u = other as usize;
        if u > h && angle_deg(normal(clean, (h / 3) as u32), normal(clean, (u / 3) as u32)) > 45.0 {
            crease += angle_deg(normal(mesh, (h / 3) as u32), normal(mesh, (u / 3) as u32));
            count += 1;
        }
    }
    (sum / n as f64, max, crease / f64::from(count))
}

#[test]
fn noisy_cube_regains_flat_faces_and_sharp_edges() {
    let clean = cube(12);
    // Up to 15% of a cell along each axis.
    let mut mesh = noisy(&clean, 0.15 * 2.0 / 12.0, 7);
    let (mean0, max0, crease0) = cube_errors(&mesh, &clean);
    let options = DenoiseOptions {
        iterations: 10,
        strength: 1.0,
        ..DenoiseOptions::default()
    };
    let (r, empty) = edit(&mut mesh, |tx| denoise(tx, &options, None));
    assert!(!empty);
    let (mean, max, crease) = cube_errors(&mesh, &clean);
    // Flat faces: from 9 degrees off their face's normal on average (and
    // up to 30) to within a degree and a half of it, a third of one on average.
    assert!(mean0 > 8.0 && max0 > 25.0, "noise {mean0} {max0}");
    assert!(mean < 0.35, "mean {mean0} -> {mean}");
    assert!(max < 1.5, "max {max0} -> {max}");
    // Sharp edges: the creases stay at right angles.
    assert!((crease - 90.0).abs() < 0.25, "crease {crease0} -> {crease}");
    assert_eq!(r.iterations, 10);
    assert_eq!(r.reverted, 0);
    assert!(r.moved > 0 && r.max_displacement > 0.0);
    assert_eq!(r.before.triangles, mesh.triangle_count());
    // The readout: 144 of the 2592 edges are creases at 90 degrees, so a
    // clean cube reads 5 degrees.
    let (dev0, dev) = (
        r.before.mean_normal_deviation_deg,
        r.after.mean_normal_deviation_deg,
    );
    assert!(dev0 > 10.0 && dev > 5.0 && dev < 5.5, "{dev0} -> {dev}");
    assert_eq!(r.after, surface_stats(&mesh, None).unwrap());
}

#[test]
fn a_low_feature_angle_keeps_more_and_full_strength_smooths_more() {
    let clean = cube(8);
    let noisy0 = noisy(&clean, 0.2 * 2.0 / 8.0, 3);
    let run = |options: DenoiseOptions| {
        let mut mesh = copy(&noisy0);
        edit(&mut mesh, |tx| denoise(tx, &options, None));
        cube_errors(&mesh, &clean).0
    };
    let weak = run(DenoiseOptions {
        iterations: 2,
        strength: 0.2,
        ..DenoiseOptions::default()
    });
    let strong = run(DenoiseOptions {
        iterations: 2,
        strength: 1.0,
        ..DenoiseOptions::default()
    });
    assert!(strong < weak, "{strong} vs {weak}");
    // A feature angle below the noise keeps it.
    let rigid = run(DenoiseOptions {
        iterations: 2,
        feature_angle_deg: 0.5,
        ..DenoiseOptions::default()
    });
    assert!(rigid > weak, "{rigid} vs {weak}");
}

#[test]
fn only_vertices_inside_the_region_move() {
    let clean = cube(6);
    let mut mesh = noisy(&clean, 0.05, 11);
    let before = mesh.positions().to_vec();
    // The faces of the +z side.
    let region: Vec<TriangleId> = (0..mesh.triangle_slots() as TriangleId)
        .filter(|&t| normal(&clean, t)[2] > 0.9)
        .collect();
    let (r, _) = edit(&mut mesh, |tx| {
        denoise(tx, &DenoiseOptions::default(), Some(&region))
    });
    assert!(r.moved > 0);
    assert_eq!(r.before.triangles, region.len());
    let inside: Vec<bool> = (0..mesh.vertex_slots() as VertexId)
        .map(|v| {
            (0..mesh.triangle_slots() as TriangleId)
                .filter(|t| mesh.triangle(*t).unwrap().contains(&v))
                .all(|t| region.contains(&t))
        })
        .collect();
    for (v, p) in mesh.positions().iter().enumerate() {
        if !inside[v] {
            assert_eq!(*p, before[v], "vertex {v} outside the region moved");
        }
    }
    // An empty region changes nothing.
    let (r, empty) = edit(&mut mesh, |tx| {
        denoise(tx, &DenoiseOptions::default(), Some(&[]))
    });
    assert!(empty);
    assert_eq!((r.moved, r.before.triangles), (0, 0));
}

#[test]
fn nothing_to_do_changes_nothing() {
    let mut mesh = noisy(&cube(3), 0.05, 5);
    for options in [
        DenoiseOptions {
            iterations: 0,
            ..DenoiseOptions::default()
        },
        DenoiseOptions {
            strength: 0.0,
            ..DenoiseOptions::default()
        },
    ] {
        let (r, empty) = edit(&mut mesh, |tx| denoise(tx, &options, None));
        assert!(empty);
        assert_eq!(r.moved, 0);
        assert_eq!(r.before, r.after);
    }
    // A flat grid is already denoised.
    let mut flat = grid(4);
    let (r, empty) = edit(&mut flat, |tx| {
        denoise(tx, &DenoiseOptions::default(), None)
    });
    assert!(empty && r.moved == 0);
}

#[test]
fn rejects_bad_options() {
    let mut mesh = cube(2);
    let bad = [
        DenoiseOptions {
            iterations: MAX_DENOISE_ITERATIONS + 1,
            ..DenoiseOptions::default()
        },
        DenoiseOptions {
            strength: 1.5,
            ..DenoiseOptions::default()
        },
        DenoiseOptions {
            strength: f32::NAN,
            ..DenoiseOptions::default()
        },
        DenoiseOptions {
            feature_angle_deg: 0.0,
            ..DenoiseOptions::default()
        },
        DenoiseOptions {
            feature_angle_deg: 181.0,
            ..DenoiseOptions::default()
        },
    ];
    for options in bad {
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            denoise(&mut tx, &options, None),
            Err(OpsError::InvalidInput(_))
        ));
    }
    let dead = [mesh_slots(&mesh)];
    let mut tx = mesh.begin_edit();
    assert!(matches!(
        denoise(&mut tx, &DenoiseOptions::default(), Some(&dead)),
        Err(OpsError::InvalidInput(_))
    ));
}

fn mesh_slots(mesh: &Mesh) -> TriangleId {
    mesh.triangle_slots() as TriangleId
}

#[test]
fn keeps_uvs_and_triangles() {
    let mut mesh = noisy(&cube(4), 0.05, 9);
    let uvs: Vec<f32> = (0..2 * mesh.vertex_slots())
        .map(|i| i as f32 * 0.01)
        .collect();
    mesh.add_uvs(&uvs).unwrap();
    let (_, triangles) = parts(&mesh);
    edit(&mut mesh, |tx| {
        denoise(tx, &DenoiseOptions::default(), None)
    });
    assert_eq!(mesh.uvs().unwrap(), &uvs[..]);
    assert_eq!(parts(&mesh).1, triangles);
}

#[test]
fn identical_with_and_without_parallel() {
    // Above the parallel threshold: 6 * 2 * 40^2 = 19200 triangles.
    let mesh = noisy(&cube(40), 0.01, 13);
    assert!(mesh.triangle_count() > PAR_MIN);
    let options = DenoiseOptions {
        iterations: 2,
        ..DenoiseOptions::default()
    };
    assert_parallel_identical("denoise", || {
        let mut m = copy(&mesh);
        let (r, _) = edit(&mut m, |tx| denoise(tx, &options, None));
        (parts(&m), format!("{r:?}"))
    });
}

#[test]
fn sweep_keeps_atomicity() {
    let n = sweep(noisy(&cube(2), 0.05, 1), |tx| {
        denoise(tx, &DenoiseOptions::default(), None)?;
        Ok(())
    });
    assert!(n > 0);
}

#[test]
fn noise_that_turns_triangles_over_does_not_block_the_smoothing() {
    let clean = cube(16);
    let options = DenoiseOptions {
        iterations: 20,
        strength: 1.0,
        feature_angle_deg: 60.0,
    };
    let run = |cells: f32| {
        let mut mesh = noisy(&clean, cells * 2.0 / 16.0, 21);
        let turned = |m: &Mesh| {
            (0..m.triangle_slots() as TriangleId)
                .filter(|&t| dot3(normal(m, t), normal(&clean, t)) <= 0.0)
                .count()
        };
        let (turned0, (mean0, ..)) = (turned(&mesh), cube_errors(&mesh, &clean));
        let (r, _) = edit(&mut mesh, |tx| denoise(tx, &options, None));
        let (mean, ..) = cube_errors(&mesh, &clean);
        (turned0, turned(&mesh), mean0, mean, r.reverted)
    };
    // Noise of 45% of a cell turns some triangles over. They are no reason
    // to put their vertices back, so the surface around them smooths too.
    let (turned0, turned, mean0, mean, reverted) = run(0.45);
    assert!(turned0 > 20 && turned <= turned0, "{turned0} -> {turned}");
    assert!(mean0 > 25.0 && mean < 0.15 * mean0, "{mean0} -> {mean}");
    assert_eq!(reverted, 0);
    // At 65% the moves themselves would turn a few over; those are put back.
    let (turned0, turned, mean0, mean, reverted) = run(0.65);
    assert!(turned < turned0 && mean < 0.4 * mean0, "{mean0} -> {mean}");
    assert!(reverted > 0 && reverted < 50, "{reverted}");
}

#[test]
fn small_meshes_stay_sequential() {
    let before = par::parallel_entries();
    let mut mesh = noisy(&cube(4), 0.05, 2);
    edit(&mut mesh, |tx| {
        denoise(tx, &DenoiseOptions::default(), None)
    });
    edit(&mut mesh, |tx| {
        crate::delaunay_flips(tx, &crate::DelaunayOptions::default(), None)
    });
    assert_eq!(par::parallel_entries(), before);
}
