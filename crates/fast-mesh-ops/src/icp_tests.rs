//! Tests of [`super::icp`]. Every expected transform is chosen here, before
//! the solver runs: the points are the fixed surface moved by its inverse, or
//! a resampling of the same analytic surface.

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::Rigid;
use fast_mesh_store::Mesh;

use super::*;
use crate::test_util::{assert_parallel_identical, box_mesh, grid, mesh_of, parts, uv_sphere};

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
    sample_surface(mesh, usize::MAX)
        .unwrap()
        .iter()
        .map(|p| apply(&inverse, *p))
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
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|d| {
            let (x, y, z) = (f64::from(d[0]), f64::from(d[1]), f64::from(d[2]));
            let r = 1.0 + 0.25 * (3.0 * x + 1.0).sin() * (2.0 * y).cos() + 0.15 * (4.0 * z).sin();
            [
                (1.3 * r * x) as f32,
                (0.9 * r * y) as f32,
                (0.6 * r * z) as f32,
            ]
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
    // Its rotation is not determined: any lattice step of the facets fits as
    // exactly as the truth. Its center is.
    assert_eq!(r.status, IcpStatus::Underconstrained, "{r:?}");
    assert!(!r.undetermined.is_empty(), "{r:?}");
    for m in &r.undetermined {
        assert!(norm(m.translation) < 0.2, "{m:?}");
    }
    // The moving sphere's center, in the points' frame, is `truth`'s inverse
    // applied to the origin: it must land back on the origin.
    let center = apply(&r.transform, truth.inverse().translation);
    assert!(norm(center) < 1e-4, "center error {center:?}");
    assert!(r.rmse < 1e-3, "{r:?}");
}

#[test]
fn a_sphere_finer_tessellation_constrains_its_spin_less() {
    // Only the facets fix a sphere's rotation, so the conditioning falls with
    // the square of the facet size; the center comes back regardless.
    let run = |stacks, slices| {
        let fixed = uv_sphere(stacks, slices, 1.0);
        let tree = bvh(&fixed);
        let center = [0.01, -0.02, 0.015];
        let points: Vec<[f64; 3]> = sample_surface(&fixed, usize::MAX)
            .unwrap()
            .iter()
            .map(|p| add(*p, center))
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
    let points: Vec<[f64; 3]> = sample_surface(&fixed, usize::MAX)
        .unwrap()
        .iter()
        .map(|p| add(*p, offset))
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
    assert!(
        (t[0] + 0.05).abs() < 1e-3 && (t[1] - 0.03).abs() < 1e-3,
        "{r:?}"
    );
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
    let points: Vec<[f64; 3]> = sample_surface(&grid(20), usize::MAX)
        .unwrap()
        .iter()
        .map(|p| add(*p, offset))
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
    let values = r
        .transform
        .rotation
        .iter()
        .flatten()
        .chain(&r.transform.translation);
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
    let plane = |p: [f64; 3]| -> [f64; 3] {
        let q = [p[0] * 0.01, p[1] * 0.01, 0.0];
        [dot(tilt[0], q), dot(tilt[1], q), dot(tilt[2], q)]
    };
    let flat = grid(n);
    let (positions, triangles) = parts(&flat);
    let tilted: Vec<f32> = positions
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|c| plane([f64::from(c[0]), f64::from(c[1]), 0.0]).map(|v| v as f32))
        .collect();
    let fixed = mesh_of(&tilted, &triangles);
    let tree = bvh(&fixed);
    let points: Vec<[f64; 3]> = sample_surface(&grid(n / 3), usize::MAX)
        .unwrap()
        .iter()
        .map(|p| {
            add(
                plane(*p),
                [0.2 * tilt[0][0], 0.2 * tilt[1][0], 0.2 * tilt[2][0]],
            )
        })
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
    let points: Vec<[f64; 3]> = sample_surface(&fixed, usize::MAX)
        .unwrap()
        .iter()
        .map(|p| add(*p, [0.0, 0.0, 2.0]))
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
    let fixed = mesh_of(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0], &[[0, 1, 2]]);
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
        assert!(invalid(run(
            &points,
            IcpOptions {
                max_distance,
                ..base
            }
        )));
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
    // Two triangles on four vertices, one floating vertex and one dead slot.
    let mut mesh = mesh_of(
        &[
            0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 9.0, 9.0, 9.0, 3.0, 3.0, 0.0, 0.0, 3.0, 0.0, 8.0, 8.0,
            8.0,
        ],
        &[[0, 1, 3], [0, 3, 4]],
    );
    let mut tx = mesh.begin_edit();
    tx.remove_vertices(&[5]).unwrap();
    let _ = tx.commit().unwrap();
    let all = vec![
        [2.0, 1.0, 0.0], // the centroids
        [1.0, 2.0, 0.0],
        [0.0, 0.0, 0.0], // the used vertices
        [3.0, 0.0, 0.0],
        [3.0, 3.0, 0.0],
        [0.0, 3.0, 0.0],
    ];
    assert_eq!(sample_surface(&mesh, usize::MAX).unwrap(), all);
    assert_eq!(
        sample_surface(&mesh, 3).unwrap(),
        vec![all[0], all[2], all[4]]
    );
    assert!(sample_surface(&mesh, 0).unwrap().is_empty());
    let big = grid(50);
    assert_eq!(sample_surface(&big, 100).unwrap().len(), 100);
}

#[test]
fn a_box_is_fully_constrained_from_its_own_samples() {
    // Eight corners and twelve faces: the corners alone tie to too few planes.
    let fixed = box_mesh([0.0; 3], [3.0, 2.0, 1.0]);
    let tree = bvh(&fixed);
    let truth = Rigid {
        rotation: turn([1.0, 0.5, -0.2], 0.02),
        translation: [0.05, -0.03, 0.02],
    };
    let points = moved_points(&fixed, &truth);
    assert_eq!(points.len(), 20);
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
    assert!(rotation < 1e-5 && translation < 1e-5, "{r:?}");
    assert!(r.conditioning > 1e-3, "{r:?}");
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

/// A deterministic value in `[-1, 1)` per call.
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f64 / (1u64 << 23) as f64 - 1.0
    }
}

/// A square of `n x n` cells spanning `[0, size]^2`, with z noise of up to
/// `noise` drawn from `seed`.
fn noisy_plane(n: u32, size: f64, noise: f64, seed: u64) -> Mesh {
    let mut rng = Noise(seed);
    let mut positions = Vec::new();
    let h = size / f64::from(n);
    for j in 0..=n {
        for i in 0..=n {
            positions.extend([
                (f64::from(i) * h) as f32,
                (f64::from(j) * h) as f32,
                (noise * rng.next()) as f32,
            ]);
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

/// A closed cylinder of radius 1 along z over `[0, 2]`, with flat capped ends.
fn capped_cylinder(rings: u32, around: u32) -> Mesh {
    let mut positions = Vec::new();
    for i in 0..=rings {
        let z = 2.0 * f64::from(i) / f64::from(rings);
        for j in 0..around {
            let a = f64::from(j) / f64::from(around) * std::f64::consts::TAU;
            positions.extend([a.cos() as f32, a.sin() as f32, z as f32]);
        }
    }
    let (bottom, top) = ((rings + 1) * around, (rings + 1) * around + 1);
    positions.extend([0.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
    let mut triangles = Vec::new();
    for j in 0..around {
        let next = (j + 1) % around;
        triangles.push([bottom, next, j]);
        triangles.push([top, rings * around + j, rings * around + next]);
        for i in 0..rings {
            let (a, b) = (i * around + j, i * around + next);
            triangles.extend([[a, b, b + around], [a, b + around, a + around]]);
        }
    }
    mesh_of(&positions, &triangles)
}

fn run_default(points: &[[f64; 3]], fixed: &Mesh, options: &IcpOptions) -> IcpReport {
    icp(points, fixed, &bvh(fixed), &Rigid::IDENTITY, options).unwrap()
}

fn displaced(mesh: &Mesh, motion: &Rigid) -> Vec<[f64; 3]> {
    sample_surface(mesh, usize::MAX)
        .unwrap()
        .iter()
        .map(|p| apply(motion, *p))
        .collect()
}

#[test]
fn a_noisy_plane_never_reports_convergence() {
    // Two different noise draws on one plane: nothing fixes x, y or the spin.
    let fixed = noisy_plane(100, 10.0, 1e-3, 1);
    let moving = noisy_plane(40, 4.0, 1e-3, 2);
    let motion = Rigid {
        translation: [3.0, 3.0, 0.01],
        ..Rigid::IDENTITY
    };
    let points = displaced(&moving, &motion);
    for tolerance in [1e-6, 1e-3] {
        let options = IcpOptions {
            tolerance,
            ..IcpOptions::default()
        };
        let r = run_default(&points, &fixed, &options);
        assert_eq!(r.status, IcpStatus::Underconstrained, "{tolerance} {r:?}");
        assert!(!r.converged());
        assert!(r.undetermined.len() >= 2, "{r:?}");
    }
}

#[test]
fn a_capped_cylinder_does_not_report_its_spin_as_determined() {
    let fixed = capped_cylinder(40, 128);
    let moving = capped_cylinder(23, 97);
    let (s, c) = 0.05f64.sin_cos();
    let motion = Rigid {
        rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.01, -0.01, 0.02],
    };
    let points = displaced(&moving, &motion);
    for tolerance in [1e-6, 1e-5, 1e-3] {
        let options = IcpOptions {
            tolerance,
            ..IcpOptions::default()
        };
        let r = run_default(&points, &fixed, &options);
        assert_eq!(r.status, IcpStatus::Underconstrained, "{tolerance} {r:?}");
        // What is undetermined is the spin about z, and only that.
        assert_eq!(r.undetermined.len(), 1, "{r:?}");
        let m = &r.undetermined[0];
        let spin = m.rotation[2].abs();
        assert!(
            spin > 10.0 * m.rotation[0].abs().max(m.rotation[1].abs()),
            "{m:?}"
        );
        // The center and the axis are recovered regardless.
        assert!(
            norm(apply(&r.transform, sub([0.0; 3], motion.translation))) < 0.05,
            "{r:?}"
        );
    }
}

#[test]
fn a_same_tessellation_sphere_does_not_snap_to_a_converged_facet() {
    // Spin by 0.06 rad: the nearest exact fit of a 64-sliced sphere is a lattice
    // step of 2 pi / 64 away from the truth, and fits as well as the truth.
    let fixed = uv_sphere(32, 64, 1.0);
    let (s, c) = 0.06f64.sin_cos();
    let motion = Rigid {
        rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
    };
    let points = displaced(&fixed, &motion);
    let r = run_default(&points, &fixed, &IcpOptions::default());
    assert!(!r.converged(), "{r:?}");
    assert_eq!(r.status, IcpStatus::Underconstrained, "{r:?}");
    assert!(!r.undetermined.is_empty(), "{r:?}");
}

#[test]
fn a_different_tessellation_sphere_is_underconstrained_at_any_tolerance() {
    let fixed = uv_sphere(64, 128, 1.0);
    let moving = uv_sphere(29, 53, 1.0);
    let motion = Rigid {
        translation: [0.02, -0.01, 0.015],
        ..Rigid::IDENTITY
    };
    let points = displaced(&moving, &motion);
    for tolerance in [1e-6, 1e-3] {
        let options = IcpOptions {
            tolerance,
            ..IcpOptions::default()
        };
        let r = run_default(&points, &fixed, &options);
        assert_eq!(r.status, IcpStatus::Underconstrained, "{tolerance} {r:?}");
        assert_eq!(r.undetermined.len(), 3, "{r:?}");
        // The center still comes back.
        assert!(
            norm(apply(&r.transform, motion.translation)) < 2e-3,
            "{r:?}"
        );
    }
}

#[test]
fn well_determined_noisy_scans_keep_converging() {
    // Two tessellations of one asymmetric surface, one with noise of 0.1% of
    // its size, and a partial overlap of the terrain with noise.
    let fixed = blob(64, 128);
    let truth = small_motion();
    let mut rng = Noise(7);
    let points: Vec<[f64; 3]> = moved_points(&blob(41, 73), &truth)
        .iter()
        .map(|p| {
            add(
                *p,
                [1e-3 * rng.next(), 1e-3 * rng.next(), 1e-3 * rng.next()],
            )
        })
        .collect();
    let options = IcpOptions {
        tolerance: 1e-4,
        ..IcpOptions::default()
    };
    let r = run_default(&points, &fixed, &options);
    assert_eq!(r.status, IcpStatus::Converged, "{r:?}");
    assert!(r.undetermined.is_empty(), "{r:?}");
    let (rotation, translation) = error(&r.transform, &truth);
    assert!(rotation < 5e-3 && translation < 5e-3, "{r:?}");

    let fixed = terrain(100, 0.1, 0.0, 0.0);
    let truth = Rigid {
        rotation: turn([0.2, 1.0, -0.4], 0.03),
        translation: [0.02, -0.03, 0.05],
    };
    let points: Vec<[f64; 3]> = moved_points(&terrain(100, 0.1, 3.0, 0.0), &truth)
        .iter()
        .map(|p| add(*p, [0.0, 0.0, 2e-3 * rng.next()]))
        .collect();
    let options = IcpOptions {
        overlap: 0.6,
        tolerance: 1e-4,
        ..IcpOptions::default()
    };
    let r = run_default(&points, &fixed, &options);
    assert_eq!(r.status, IcpStatus::Converged, "{r:?}");
    assert!(r.undetermined.is_empty(), "{r:?}");
    let (rotation, translation) = error(&r.transform, &truth);
    assert!(rotation < 5e-3 && translation < 5e-3, "{r:?}");
}

#[test]
fn a_long_thin_box_lands_with_an_accurate_rmse() {
    let fixed = box_mesh([0.0; 3], [100.0, 0.7, 1.0]);
    let motion = Rigid {
        translation: [0.05, 0.02, -0.03],
        ..Rigid::IDENTITY
    };
    let points = displaced(&fixed, &motion);
    let r = run_default(&points, &fixed, &IcpOptions::default());
    assert_eq!(r.status, IcpStatus::Converged, "{r:?}");
    let truth = motion.inverse();
    assert!(error(&r.transform, &truth).1 < 1e-5, "{r:?}");
    // On the surface to within the f32 positions, whose step at 100 is 8e-6.
    assert!(r.rmse < 1e-5, "{r:?}");
}

#[test]
fn a_far_guess_settles_in_a_wrong_minimum_that_only_its_rmse_shows() {
    // Nearly half a turn off, on an asymmetric mesh: every motion is locally
    // determined, and the fit is wrong. This is the rmse guard's job.
    let fixed = blob(32, 64);
    let (s, c) = 2.8f64.sin_cos();
    let motion = Rigid {
        rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
    };
    let points = displaced(&fixed, &motion);
    let options = IcpOptions {
        tolerance: 1e-3,
        ..IcpOptions::default()
    };
    let r = run_default(&points, &fixed, &options);
    assert_eq!(r.status, IcpStatus::Converged, "{r:?}");
    assert!(r.rmse > 0.02 * r.extent, "{r:?}");
    assert!(error(&r.transform, &motion.inverse()).0 > 0.1, "{r:?}");
}

#[test]
fn gentle_relief_and_a_small_key_stay_determined() {
    // Relief 100 times flatter than usual: conditioning 2e-6, still determined.
    let fixed = flattened(&terrain(100, 0.1, 0.0, 0.0), 0.01);
    let motion = Rigid {
        translation: [0.1, 0.05, 0.01],
        ..Rigid::IDENTITY
    };
    let r = run_default(&displaced(&fixed, &motion), &fixed, &IcpOptions::default());
    assert_eq!(r.status, IcpStatus::Converged, "{r:?}");
    assert!(error(&r.transform, &motion.inverse()).1 < 1e-4, "{r:?}");
    // A bump on a cylinder fixes the spin that the bare cylinder leaves open.
    let fixed = keyed_cylinder(80, 256);
    let (s, c) = 0.03f64.sin_cos();
    let truth = Rigid {
        rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.01, 0.0, 0.0],
    };
    let points = moved_points(&keyed_cylinder(40, 128), &truth);
    let r = run_default(&points, &fixed, &IcpOptions::default());
    assert_eq!(r.status, IcpStatus::Converged, "{r:?}");
    assert!(error(&r.transform, &truth).0 < 5e-3, "{r:?}");
}

/// The mesh with its z coordinates multiplied.
fn flattened(mesh: &Mesh, factor: f32) -> Mesh {
    let (mut positions, triangles) = parts(mesh);
    for c in positions.as_chunks_mut::<3>().0 {
        c[2] *= factor;
    }
    mesh_of(&positions, &triangles)
}

/// A capped cylinder with a bump on one side, which alone fixes the spin.
fn keyed_cylinder(rings: u32, around: u32) -> Mesh {
    let (mut positions, triangles) = parts(&capped_cylinder(rings, around));
    for c in positions.as_chunks_mut::<3>().0 {
        let radius = (c[0] * c[0] + c[1] * c[1]).sqrt();
        if radius > 0.5 && c[0] > 0.9 && c[2] > 0.8 && c[2] < 1.2 {
            c[0] *= 1.1;
            c[1] *= 1.1;
        }
    }
    mesh_of(&positions, &triangles)
}

/// Prints how each case behaves, for the numbers in `docs/benchmarks.md`.
/// `cargo test -p fast-mesh-ops --lib icp_survey -- --ignored --nocapture`
#[test]
#[ignore = "prints a table"]
fn icp_survey() {
    let spin = |a: f64| {
        let (s, c) = a.sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    };
    let mut rng = Noise(11);
    let small = small_motion();
    let t = |translation| Rigid {
        translation,
        ..Rigid::IDENTITY
    };
    let noisy = |points: Vec<[f64; 3]>, a: f64, rng: &mut Noise| -> Vec<[f64; 3]> {
        points
            .iter()
            .map(|p| add(*p, [a * rng.next(), a * rng.next(), a * rng.next()]))
            .collect()
    };
    let far_terrain = terrain(200, 0.5, 0.0, 0.0);
    let box_points = displaced(
        &box_mesh([0.0; 3], [3.0, 2.0, 1.0]),
        &t([0.05, -0.03, 0.02]),
    );
    #[rustfmt::skip]
    let cases: Vec<(&str, Mesh, Vec<[f64; 3]>, f64)> = vec![
        ("blob same", blob(32, 64), moved_points(&blob(32, 64), &small), 1.0),
        ("blob resampled", blob(64, 128), moved_points(&blob(41, 73), &small), 1.0),
        ("blob noise 1e-3", blob(64, 128), noisy(moved_points(&blob(41, 73), &small), 1e-3, &mut rng), 1.0),
        ("blob noise 1e-2", blob(64, 128), noisy(moved_points(&blob(41, 73), &small), 1e-2, &mut rng), 1.0),
        ("blob overlap .8", blob(64, 128), moved_points(&blob(41, 73), &small), 0.8),
        ("box", box_mesh([0.0; 3], [3.0, 2.0, 1.0]), box_points, 1.0),
        ("sliver box", box_mesh([0.0; 3], [100.0, 0.7, 1.0]), displaced(&box_mesh([0.0; 3], [100.0, 0.7, 1.0]), &t([0.05, 0.02, -0.03])), 1.0),
        ("terrain partial", terrain(100, 0.1, 0.0, 0.0), moved_points(&terrain(100, 0.1, 3.0, 0.0), &small), 0.6),
        ("terrain 0.5 cells", far_terrain, displaced(&terrain(200, 0.5, 0.0, 0.0), &t([0.2, -0.1, 0.1])), 1.0),
        ("cube-ish 8x16 sphere", uv_sphere(8, 16, 1.0), displaced(&uv_sphere(8, 16, 1.0), &t([0.02, 0.01, 0.0])), 1.0),
        ("sphere 32x64 spin", uv_sphere(32, 64, 1.0), displaced(&uv_sphere(32, 64, 1.0), &Rigid { rotation: spin(0.06), translation: [0.0; 3] }), 1.0),
        ("sphere 128x256 spin", uv_sphere(128, 256, 1.0), displaced(&uv_sphere(128, 256, 1.0), &Rigid { rotation: spin(0.06), translation: [0.0; 3] }), 1.0),
        ("sphere resampled", uv_sphere(64, 128, 1.0), displaced(&uv_sphere(29, 53, 1.0), &t([0.02, -0.01, 0.015])), 1.0),
        ("cylinder resampled", capped_cylinder(40, 128), displaced(&capped_cylinder(23, 97), &t([0.01, -0.01, 0.02])), 1.0),
        ("plane exact", grid(40), displaced(&grid(20), &t([3.5, 2.25, 0.5])), 1.0),
        ("plane noise 1e-3", noisy_plane(100, 10.0, 1e-3, 1), displaced(&noisy_plane(40, 4.0, 1e-3, 2), &t([3.0, 3.0, 0.01])), 1.0),
        ("relief x0.01", flattened(&terrain(100, 0.1, 0.0, 0.0), 0.01), displaced(&flattened(&terrain(100, 0.1, 0.0, 0.0), 0.01), &t([0.1, 0.05, 0.01])), 1.0),
        ("relief x0.01 resampled", flattened(&terrain(150, 0.1, 0.0, 0.0), 0.01), moved_points(&flattened(&terrain(53, 0.17, 2.0, 2.0), 0.01), &t([0.1, 0.05, 0.01])), 1.0),
        ("relief x0.001 noisy", flattened(&terrain(150, 0.1, 0.0, 0.0), 0.001), noisy(moved_points(&flattened(&terrain(53, 0.17, 2.0, 2.0), 0.001), &t([0.1, 0.05, 0.01])), 3e-3, &mut rng), 1.0),
        ("keyed cylinder", keyed_cylinder(40, 128), displaced(&keyed_cylinder(40, 128), &Rigid { rotation: spin(0.03), translation: [0.01, 0.0, 0.0] }), 1.0),
        ("keyed cyl resampled", keyed_cylinder(80, 256), moved_points(&keyed_cylinder(40, 128), &Rigid { rotation: spin(0.03), translation: [0.01, 0.0, 0.0] }), 1.0),
        ("far guess blob", blob(32, 64), displaced(&blob(32, 64), &Rigid { rotation: spin(2.8), translation: [0.0; 3] }), 1.0),
        ("far guess blob 1.2", blob(32, 64), displaced(&blob(32, 64), &Rigid { rotation: spin(1.2), translation: [0.2, 0.0, 0.0] }), 1.0),
        ("far terrain partial", terrain(100, 0.1, 0.0, 0.0), displaced(&terrain(100, 0.1, 3.0, 0.0), &t([-1.5, 1.2, 0.3])), 0.6),
        ("plane same noise", noisy_plane(100, 10.0, 1e-3, 1), displaced(&noisy_plane(100, 10.0, 1e-3, 1), &t([0.2, 0.1, 0.01])), 1.0),
    ];
    for (name, fixed, points, overlap) in &cases {
        for tolerance in [1e-6, 1e-3] {
            let options = IcpOptions {
                overlap: *overlap,
                tolerance,
                ..IcpOptions::default()
            };
            let r = run_default(points, fixed, &options);
            println!(
                "{name:22} tol {tolerance:.0e} {:?} it {} rmse {:.2e} cond {:.1e} undetermined {} inliers {} of {}",
                r.status,
                r.iterations,
                r.rmse,
                r.conditioning,
                r.undetermined.len(),
                r.inliers,
                r.samples
            );
        }
    }
}
