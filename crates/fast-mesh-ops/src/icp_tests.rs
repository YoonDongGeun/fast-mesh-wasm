//! Tests of [`super::icp`]. Every expected transform is chosen here, before
//! the solver runs: the points are the fixed surface moved by its inverse, or
//! a resampling of the same analytic surface.

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::Rigid;
use fast_mesh_store::Mesh;

use super::*;
use crate::test_util::{assert_parallel_identical, grid, mesh_of, parts, uv_sphere};

/// A rotation of `angle` radians about the unit `axis`.
fn turn(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let k = unit(axis).unwrap();
    let (s, c) = angle.sin_cos();
    let v = 1.0 - c;
    [
        [
            c + k[0] * k[0] * v,
            k[0] * k[1] * v - k[2] * s,
            k[0] * k[2] * v + k[1] * s,
        ],
        [
            k[1] * k[0] * v + k[2] * s,
            c + k[1] * k[1] * v,
            k[1] * k[2] * v - k[0] * s,
        ],
        [
            k[2] * k[0] * v - k[1] * s,
            k[2] * k[1] * v + k[0] * s,
            c + k[2] * k[2] * v,
        ],
    ]
}

/// The largest entry-wise difference between two rotations and the distance
/// between two translations.
fn error(a: &Rigid, b: &Rigid) -> (f64, f64) {
    let rotation = (0..3)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| (a.rotation[i][j] - b.rotation[i][j]).abs())
        .fold(0.0, f64::max);
    (rotation, norm(sub(a.translation, b.translation)))
}

/// The stored positions of `mesh` mapped by the inverse of `truth`, so that
/// `truth` is the transform that puts them back on `mesh`.
fn moved_points(mesh: &Mesh, truth: &Rigid) -> Vec<[f64; 3]> {
    let inverse = truth.inverse();
    sample_vertices(mesh, usize::MAX)
        .unwrap()
        .iter()
        .map(|p| inverse.apply(*p))
        .collect()
}

fn bvh(mesh: &Mesh) -> Bvh {
    Bvh::build(mesh, &BuildOptions::default()).unwrap()
}

/// A closed lumpy shape without symmetries: a UV sphere's directions pushed
/// out by an analytic radius and squashed per axis. The same surface at any
/// resolution.
fn blob(stacks: u32, slices: u32) -> Mesh {
    let sphere = uv_sphere(stacks, slices, 1.0);
    let (positions, triangles) = parts(&sphere);
    let shaped: Vec<f32> = positions
        .chunks_exact(3)
        .flat_map(|d| {
            let (x, y, z) = (f64::from(d[0]), f64::from(d[1]), f64::from(d[2]));
            let r = 1.0 + 0.25 * (3.0 * x + 1.0).sin() * (2.0 * y).cos() + 0.15 * (4.0 * z).sin();
            [(1.3 * r * x) as f32, (0.9 * r * y) as f32, (0.6 * r * z) as f32]
        })
        .collect();
    mesh_of(&shaped, &triangles)
}

/// A non-periodic height at `(x, y)`.
fn relief(x: f64, y: f64) -> f64 {
    0.5 * (0.9 * x + 0.3 * y).sin() + 0.4 * (0.7 * y - 0.05 * x * x).cos() + 0.03 * x
}

/// A height field over `[x0, x0 + n h] x [y0, y0 + n h]`.
fn terrain(n: u32, h: f64, x0: f64, y0: f64) -> Mesh {
    let mut positions = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let (x, y) = (x0 + f64::from(i) * h, y0 + f64::from(j) * h);
            positions.extend([x as f32, y as f32, relief(x, y) as f32]);
        }
    }
    let mut triangles = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            triangles.extend([[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]);
        }
    }
    mesh_of(&positions, &triangles)
}

fn small_motion() -> Rigid {
    Rigid {
        rotation: turn([1.0, 2.0, 0.5], 0.12),
        translation: [0.08, -0.05, 0.06],
    }
}

#[test]
fn recovers_a_known_transform_of_an_asymmetric_mesh() {
    let fixed = blob(32, 64);
    let tree = bvh(&fixed);
    let truth = small_motion();
    let points = moved_points(&fixed, &truth);
    let r = icp(
        &points,
        &fixed,
        &tree,
        &Rigid::IDENTITY,
        &IcpOptions::default(),
    )
    .unwrap();
    assert_eq!(r.status, IcpStatus::Converged, "{r:?}");
    assert!(r.converged());
    let (rotation, translation) = error(&r.transform, &truth);
    // The points are the mesh's own vertices: exact but for f32 positions.
    assert!(rotation < 1e-5, "rotation error {rotation}");
    assert!(translation < 1e-5, "translation error {translation}");
    assert!(r.rmse < 1e-5, "{r:?}");
    assert_eq!(r.samples, points.len());
    assert_eq!(r.inliers, points.len());
    assert!(r.iterations >= 2 && r.iterations <= 50, "{r:?}");
    assert!(r.conditioning >= MIN_CONDITIONING, "{r:?}");
}

#[test]
fn recovers_a_transform_between_two_tessellations_of_one_surface() {
    // Different vertices on the same analytic surface: the error is bounded
    // by the chord error of the coarser tessellation, not by rounding.
    let fixed = blob(96, 192);
    let tree = bvh(&fixed);
    let truth = small_motion();
    let moving = blob(41, 73);
    let points = moved_points(&moving, &truth);
    let r = icp(
        &points,
        &fixed,
        &tree,
        &Rigid::IDENTITY,
        &IcpOptions::default(),
    )
    .unwrap();
    assert_eq!(r.status, IcpStatus::Converged, "{r:?}");
    let (rotation, translation) = error(&r.transform, &truth);
    assert!(rotation < 5e-3, "rotation error {rotation}");
    assert!(translation < 5e-3, "translation error {translation}");
    assert!(r.rmse < 5e-3, "{r:?}");
}

#[test]
fn recovers_a_sphere_center() {
    // A sphere's rotation is fixed only by its facets; its center is fixed by
    // the surface. Same tessellation, displaced by much less than a facet.
    let fixed = uv_sphere(24, 48, 2.0);
    let tree = bvh(&fixed);
    let truth = Rigid {
        rotation: turn([0.3, -0.5, 1.0], 0.01),
        translation: [0.09, -0.04, 0.07],
    };
    let points = moved_points(&fixed, &truth);
    let r = icp(
        &points,
        &fixed,
        &tree,
        &Rigid::IDENTITY,
        &IcpOptions::default(),
    )
    .unwrap();
    assert_eq!(r.status, IcpStatus::Converged, "{r:?}");
    // The moving sphere's center, in the points' frame, is `truth`'s inverse
    // applied to the origin: it must land back on the origin.
    let center = apply(&r.transform, truth.inverse().translation);
    assert!(norm(center) < 1e-4, "center error {center:?}");
    let (rotation, _) = error(&r.transform, &truth);
    assert!(rotation < 1e-3, "rotation error {rotation}");
    assert!(r.rmse < 1e-4, "{r:?}");
}

#[test]
fn a_sphere_finer_tessellation_constrains_its_spin_less() {
    // Only the facets fix a sphere's rotation, so the conditioning falls with
    // the square of the facet size; the center comes back regardless.
    let run = |stacks, slices| {
        let fixed = uv_sphere(stacks, slices, 1.0);
        let tree = bvh(&fixed);
        let center = [0.01, -0.02, 0.015];
        let points: Vec<[f64; 3]> = sample_vertices(&fixed, usize::MAX)
            .unwrap()
            .iter()
            .map(|p| add(p.map(f64::from), center))
            .collect();
        let r = icp(
            &points,
            &fixed,
            &tree,
            &Rigid::IDENTITY,
            &IcpOptions::default(),
        )
        .unwrap();
        assert!(norm(apply(&r.transform, center)) < 1e-4, "{r:?}");
        assert!(r.rmse < 1e-4, "{r:?}");
        r.conditioning
    };
    let (coarse, fine) = (run(32, 64), run(64, 128));
    assert!(coarse > 2.0 * fine, "{coarse} {fine}");
    assert!(fine > 0.0 && coarse < 0.1, "{coarse} {fine}");
}

#[test]
fn a_tube_sliding_along_its_axis_is_underconstrained() {
    // An open tube: every normal is perpendicular to z, so z has no constraint
    // at all, on any tessellation. Displaced radially too: that part is fixed.
    let (rings, around) = (40u32, 64u32);
    let mut positions = Vec::new();
    for i in 0..=rings {
        for j in 0..around {
            let a = f64::from(j) / f64::from(around) * std::f64::consts::TAU;
            positions.extend([a.cos() as f32, a.sin() as f32, (f64::from(i) * 0.1) as f32]);
        }
    }
    let mut triangles = Vec::new();
    for i in 0..rings {
        for j in 0..around {
            let (a, b) = (i * around + j, i * around + (j + 1) % around);
            let (c, d) = (a + around, b + around);
            triangles.extend([[a, b, d], [a, d, c]]);
        }
    }
    let fixed = mesh_of(&positions, &triangles);
    let tree = bvh(&fixed);
    let offset = [0.05, -0.03, 0.2];
    let points: Vec<[f64; 3]> = sample_vertices(&fixed, usize::MAX)
        .unwrap()
        .iter()
        .map(|p| add(p.map(f64::from), offset))
        .collect();
    let r = icp(
        &points,
        &fixed,
        &tree,
        &Rigid::IDENTITY,
        &IcpOptions {
            overlap: 0.9,
            ..IcpOptions::default()
        },
    )
    .unwrap();
    assert_eq!(r.status, IcpStatus::Underconstrained, "{r:?}");
    assert!(!r.converged());
    let t = r.transform.translation;
    assert!((t[0] + 0.05).abs() < 1e-3 && (t[1] - 0.03).abs() < 1e-3, "{r:?}");
    assert!(t[2].abs() < 1e-3, "{r:?}");
}

#[test]
fn partial_overlap_with_outliers_recovers_when_trimmed() {
    // The moving terrain covers 70% of the fixed one in x, and 6% of its
    // points are noise far off the surface.
    let fixed = terrain(100, 0.1, 0.0, 0.0);
    let tree = bvh(&fixed);
    let truth = Rigid {
        rotation: turn([0.2, 1.0, -0.4], 0.03),
        translation: [0.02, -0.03, 0.05],
    };
    let moving = terrain(100, 0.1, 3.0, 0.0);
    let mut points = moved_points(&moving, &truth);
    let total = points.len();
    let inverse = truth.inverse();
    for (i, p) in points.iter_mut().enumerate().filter(|(i, _)| i % 17 == 0) {
        let lift = 1.0 + (i % 5) as f64 * 0.4;
        let q = apply(&truth, *p);
        *p = apply(&inverse, [q[0], q[1], q[2] + lift]);
    }
    let run = |overlap| {
        icp(
            &points,
            &fixed,
            &tree,
            &Rigid::IDENTITY,
            &IcpOptions {
                overlap,
                ..IcpOptions::default()
            },
        )
        .unwrap()
    };
    let trimmed = run(0.6);
    assert_eq!(trimmed.status, IcpStatus::Converged, "{trimmed:?}");
    let (rotation, translation) = error(&trimmed.transform, &truth);
    assert!(rotation < 1e-4, "rotation error {rotation}");
    assert!(translation < 1e-3, "translation error {translation}");
    assert!(trimmed.rmse < 1e-4, "{trimmed:?}");
    assert_eq!(trimmed.samples, total);
    assert_eq!(trimmed.inliers, (0.6 * total as f64).ceil() as usize);
    // Every point counted: the outliers and the overhang drag it off.
    let plain = run(1.0);
    let (plain_rotation, plain_translation) = error(&plain.transform, &truth);
    assert!(
        plain_rotation > 10.0 * rotation.max(1e-4)
            || plain_translation > 10.0 * translation.max(1e-3),
        "{plain:?}"
    );
    assert!(plain.rmse > 0.05, "{plain:?}");
}

#[test]
fn max_distance_drops_far_points() {
    let fixed = blob(24, 48);
    let tree = bvh(&fixed);
    let truth = small_motion();
    let mut points = moved_points(&fixed, &truth);
    let good = points.len();
    points.extend((0..20).map(|i| [50.0 + f64::from(i), 40.0, 40.0]));
    let r = icp(
        &points,
        &fixed,
        &tree,
        &Rigid::IDENTITY,
        &IcpOptions {
            max_distance: 5.0,
            ..IcpOptions::default()
        },
    )
    .unwrap();
    assert_eq!(r.status, IcpStatus::Converged, "{r:?}");
    assert_eq!(r.inliers, good);
    assert!(error(&r.transform, &truth).0 < 1e-5, "{r:?}");
}

#[test]
fn plane_sliding_in_plane_is_not_converged_and_stays_finite() {
    // Every normal is +z, so z and the two tilts are constrained and x, y and
    // the spin are not. The plane comes back to z = 0; what it cannot know it
    // leaves alone.
    let fixed = grid(40);
    let tree = bvh(&fixed);
    let offset = [3.5, 2.25, 0.5];
    let points: Vec<[f64; 3]> = sample_vertices(&grid(20), usize::MAX)
        .unwrap()
        .iter()
        .map(|p| add(p.map(f64::from), offset))
        .collect();
    let r = icp(
        &points,
        &fixed,
        &tree,
        &Rigid::IDENTITY,
        &IcpOptions::default(),
    )
    .unwrap();
    assert_eq!(r.status, IcpStatus::Underconstrained, "{r:?}");
    assert!(!r.converged());
    assert!(r.conditioning < MIN_CONDITIONING, "{r:?}");
    assert!(r.rmse.is_finite() && r.rmse < 1e-5, "{r:?}");
    let values = r.transform.rotation.iter().flatten().chain(&r.transform.translation);
    assert!(values.into_iter().all(|v| v.is_finite()), "{r:?}");
    // z is corrected, x and y are not slid.
    assert!((r.transform.translation[2] + 0.5).abs() < 1e-5, "{r:?}");
    assert!(r.transform.translation[0].abs() < 1e-5, "{r:?}");
    assert!(r.transform.translation[1].abs() < 1e-5, "{r:?}");
    // No spin either.
    assert!(error(&r.transform, &Rigid::IDENTITY).0 < 1e-5, "{r:?}");
}

#[test]
fn tilted_fine_plane_is_still_underconstrained() {
    // f32 rounding of a tilted plane's vertices jitters its face normals; that
    // noise must not count as a constraint.
    let n = 300;
    let tilt = turn([1.0, 1.0, 0.3], 0.7);
    let plane = |p: &[f32; 3]| -> [f64; 3] {
        let q = [f64::from(p[0]) * 0.01, f64::from(p[1]) * 0.01, 0.0];
        [dot(tilt[0], q), dot(tilt[1], q), dot(tilt[2], q)]
    };
    let flat = grid(n);
    let (positions, triangles) = parts(&flat);
    let tilted: Vec<f32> = positions
        .chunks_exact(3)
        .flat_map(|c| plane(&[c[0], c[1], c[2]]).map(|v| v as f32))
        .collect();
    let fixed = mesh_of(&tilted, &triangles);
    let tree = bvh(&fixed);
    let points: Vec<[f64; 3]> = sample_vertices(&grid(n / 3), usize::MAX)
        .unwrap()
        .iter()
        .map(|p| add(plane(p), [0.2 * tilt[0][0], 0.2 * tilt[1][0], 0.2 * tilt[2][0]]))
        .collect();
    let r = icp(
        &points,
        &fixed,
        &tree,
        &Rigid::IDENTITY,
        &IcpOptions::default(),
    )
    .unwrap();
    assert_eq!(r.status, IcpStatus::Underconstrained, "{r:?}");
    assert!(!r.converged());
}

#[test]
fn running_out_of_iterations_is_reported() {
    let fixed = blob(24, 48);
    let tree = bvh(&fixed);
    let points = moved_points(&fixed, &small_motion());
    let r = icp(
        &points,
        &fixed,
        &tree,
        &Rigid::IDENTITY,
        &IcpOptions {
            max_iterations: 1,
            ..IcpOptions::default()
        },
    )
    .unwrap();
    assert_eq!(r.status, IcpStatus::MaxIterations, "{r:?}");
    assert!(!r.converged());
    assert_eq!(r.iterations, 1);
    assert!(r.rmse.is_finite() && r.rmse > 1e-5, "{r:?}");
}

#[test]
fn zero_iterations_measure_the_initial_pose() {
    // The unit plane at height 2, seen from z = 0: every distance is 2.
    let fixed = grid(4);
    let tree = bvh(&fixed);
    let points: Vec<[f64; 3]> = sample_vertices(&fixed, usize::MAX)
        .unwrap()
        .iter()
        .map(|p| add(p.map(f64::from), [0.0, 0.0, 2.0]))
        .collect();
    let r = icp(
        &points,
        &fixed,
        &tree,
        &Rigid::IDENTITY,
        &IcpOptions {
            max_iterations: 0,
            ..IcpOptions::default()
        },
    )
    .unwrap();
    assert_eq!(r.status, IcpStatus::MaxIterations);
    assert_eq!(r.iterations, 0);
    assert_eq!(r.transform, Rigid::IDENTITY);
    assert!((r.rmse - 2.0).abs() < 1e-6, "{r:?}");
    assert_eq!(r.conditioning, 0.0);
}

#[test]
fn too_few_correspondences_are_reported_with_finite_numbers() {
    let fixed = grid(4);
    let tree = bvh(&fixed);
    let options = IcpOptions {
        max_distance: 0.5,
        ..IcpOptions::default()
    };
    // None in range: no measure at all.
    let far = [[100.0, 100.0, 100.0]; 10];
    let r = icp(&far, &fixed, &tree, &Rigid::IDENTITY, &options).unwrap();
    assert_eq!(r.status, IcpStatus::TooFewCorrespondences, "{r:?}");
    assert_eq!(r.inliers, 0);
    assert_eq!(r.rmse, f64::INFINITY);
    assert_eq!(r.transform, Rigid::IDENTITY);
    // Three in range: three cannot fix six degrees of freedom.
    let three = [[1.0, 1.0, 0.25], [2.0, 1.0, 0.25], [1.0, 2.0, 0.25]];
    let r = icp(&three, &fixed, &tree, &Rigid::IDENTITY, &options).unwrap();
    assert_eq!(r.status, IcpStatus::TooFewCorrespondences, "{r:?}");
    assert_eq!(r.inliers, 3);
    assert!((r.rmse - 0.25).abs() < 1e-6, "{r:?}");
    assert!(!r.converged());
}

#[test]
fn degenerate_triangles_do_not_produce_nan() {
    // All of the fixed mesh's triangles have zero area: no plane to use.
    let fixed = mesh_of(
        &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        &[[0, 1, 2]],
    );
    let tree = bvh(&fixed);
    let points = [[0.5, 1.0, 0.0]; 8];
    let r = icp(
        &points,
        &fixed,
        &tree,
        &Rigid::IDENTITY,
        &IcpOptions::default(),
    )
    .unwrap();
    assert_eq!(r.status, IcpStatus::TooFewCorrespondences, "{r:?}");
    assert!(r.rmse.is_finite(), "{r:?}");
}

#[test]
fn invalid_input_is_rejected() {
    let fixed = grid(2);
    let tree = bvh(&fixed);
    let points = [[0.0; 3]; 8];
    let run = |p: &[[f64; 3]], o: IcpOptions| icp(p, &fixed, &tree, &Rigid::IDENTITY, &o);
    let base = IcpOptions::default();
    let invalid = |r: Result<IcpReport>| matches!(r, Err(OpsError::InvalidInput(_)));
    assert!(invalid(run(&[], base)));
    assert!(invalid(run(&[[f64::NAN, 0.0, 0.0]], base)));
    for overlap in [0.0, -0.5, 1.5, f64::NAN] {
        assert!(invalid(run(&points, IcpOptions { overlap, ..base })));
    }
    for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(invalid(run(&points, IcpOptions { tolerance, ..base })));
    }
    for max_distance in [0.0, -1.0, f32::NAN] {
        assert!(invalid(run(&points, IcpOptions { max_distance, ..base })));
    }
    let empty = Mesh::new();
    let empty_tree = bvh(&empty);
    assert!(invalid(icp(
        &points,
        &empty,
        &empty_tree,
        &Rigid::IDENTITY,
        &base
    )));
    // A bad initial transform, and one that leaves f32 range.
    let bad = Rigid {
        rotation: [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
    };
    assert!(matches!(
        icp(&points, &fixed, &tree, &bad, &base),
        Err(OpsError::Query(_))
    ));
    let far = Rigid {
        translation: [1e300, 0.0, 0.0],
        ..Rigid::IDENTITY
    };
    assert!(matches!(
        icp(&points, &fixed, &tree, &far, &base),
        Err(OpsError::Query(_))
    ));
}

#[test]
fn sampling_is_even_and_skips_unused_vertices() {
    // Four vertices used, one floating and one dead slot in the middle.
    let mut mesh = mesh_of(
        &[
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 9.0, 9.0, 9.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 8.0, 8.0,
            8.0,
        ],
        &[[0, 1, 3], [0, 3, 4]],
    );
    let mut tx = mesh.begin_edit();
    tx.remove_vertices(&[5]).unwrap();
    let _ = tx.commit().unwrap();
    assert_eq!(
        sample_vertices(&mesh, usize::MAX).unwrap(),
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    );
    assert_eq!(
        sample_vertices(&mesh, 2).unwrap(),
        vec![[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    );
    assert!(sample_vertices(&mesh, 0).unwrap().is_empty());
    let big = grid(50);
    assert_eq!(sample_vertices(&big, 100).unwrap().len(), 100);
}

#[test]
fn results_are_identical_with_and_without_parallelism() {
    // Above the query batch's parallel threshold (256 points).
    let fixed = terrain(120, 0.1, 0.0, 0.0);
    let tree = bvh(&fixed);
    let truth = Rigid {
        rotation: turn([0.2, 1.0, -0.4], 0.03),
        translation: [0.02, -0.03, 0.05],
    };
    let points = moved_points(&terrain(120, 0.1, 2.0, 0.0), &truth);
    assert!(points.len() > 256);
    let options = IcpOptions {
        overlap: 0.7,
        ..IcpOptions::default()
    };
    let run = || {
        let r = icp(&points, &fixed, &tree, &Rigid::IDENTITY, &options).unwrap();
        // Compare the bits: `PartialEq` would equate 0.0 and -0.0.
        let values = r
            .transform
            .rotation
            .iter()
            .flatten()
            .chain(&r.transform.translation)
            .chain([&r.rmse, &r.conditioning])
            .map(|v| v.to_bits())
            .collect::<Vec<u64>>();
        (values, r.iterations, r.status, r.inliers)
    };
    assert_parallel_identical("icp", run);
}
