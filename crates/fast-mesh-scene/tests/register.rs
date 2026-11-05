//! `Scene::register`: placements in and out of the solver, scales, undo and
//! modes. The solver itself is tested in `fast-mesh-ops`. Every expected
//! placement is chosen before the call: the moving mesh starts displaced from
//! a placement known to put it exactly on the fixed mesh.

use fast_mesh_scene::ops::{IcpOptions, IcpStatus};
use fast_mesh_scene::{
    MeshId, MeshOptions, NotApplied, Placement, RegisterOptions, Scene, SceneError, SceneOptions,
    UpdateMode,
};
use fast_mesh_store::{Mesh, par};

/// A rotation of `angle` radians about the z axis, then of `tilt` about x.
fn rotation(angle: f64, tilt: f64) -> [[f64; 3]; 3] {
    let (s, c) = angle.sin_cos();
    let (st, ct) = tilt.sin_cos();
    [[c, -s, 0.0], [ct * s, ct * c, -st], [st * s, st * c, ct]]
}

/// A lumpy closed surface without symmetries, its stored coordinates divided
/// per axis by `scale` (so that placing it with `scale` gives one shape) and
/// moved by `shift`.
fn blob_buffers(
    stacks: u32,
    slices: u32,
    scale: [f64; 3],
    shift: [f64; 3],
) -> (Vec<f64>, Vec<u32>) {
    let point = |x: f64, y: f64, z: f64| {
        let r = 1.0 + 0.25 * (3.0 * x + 1.0).sin() * (2.0 * y).cos() + 0.15 * (4.0 * z).sin();
        [
            1.3 * r * x / scale[0] + shift[0],
            0.9 * r * y / scale[1] + shift[1],
            0.6 * r * z / scale[2] + shift[2],
        ]
    };
    let mut positions = Vec::new();
    positions.extend(point(0.0, 0.0, 1.0));
    for i in 1..stacks {
        let phi = std::f64::consts::PI * f64::from(i) / f64::from(stacks);
        for j in 0..slices {
            let a = std::f64::consts::TAU * f64::from(j) / f64::from(slices);
            positions.extend(point(phi.sin() * a.cos(), phi.sin() * a.sin(), phi.cos()));
        }
    }
    positions.extend(point(0.0, 0.0, -1.0));
    let south = 1 + (stacks - 1) * slices;
    let ring = |i: u32, j: u32| 1 + (i - 1) * slices + j % slices;
    let mut indices = Vec::new();
    for j in 0..slices {
        indices.extend([0, ring(1, j), ring(1, j + 1)]);
        indices.extend([south, ring(stacks - 1, j + 1), ring(stacks - 1, j)]);
        for i in 1..stacks - 1 {
            let (a, b, c, d) = (
                ring(i, j),
                ring(i + 1, j),
                ring(i + 1, j + 1),
                ring(i, j + 1),
            );
            indices.extend([a, b, c, a, c, d]);
        }
    }
    (positions, indices)
}

fn mesh(buffers: &(Vec<f64>, Vec<u32>)) -> Mesh {
    let f32s: Vec<f32> = buffers.0.iter().map(|&v| v as f32).collect();
    Mesh::from_buffers(&f32s, &buffers.1).unwrap()
}

/// A flat square of `n x n` cells of side 1, at z = 0.
fn plane(n: u32) -> Mesh {
    let mut positions = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            positions.extend([x as f32, y as f32, 0.0]);
        }
    }
    let mut indices = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn add(scene: &mut Scene, m: Mesh) -> MeshId {
    scene.add_mesh(m, MeshOptions::default()).unwrap()
}

fn close(a: &Placement, b: &Placement, tol: f64) -> bool {
    let rotation = a
        .rotation
        .iter()
        .flatten()
        .zip(b.rotation.iter().flatten())
        .all(|(x, y)| (x - y).abs() <= tol);
    let translation = a
        .translation
        .iter()
        .zip(&b.translation)
        .all(|(x, y)| (x - y).abs() <= tol);
    rotation && translation && a.scale == b.scale
}

/// A small displacement of `p`: the initial guess.
fn nudged(p: &Placement) -> Placement {
    let d = rotation(0.06, -0.04);
    let r = &p.rotation;
    Placement {
        rotation: std::array::from_fn(|i| {
            std::array::from_fn(|j| d[i][0] * r[0][j] + d[i][1] * r[1][j] + d[i][2] * r[2][j])
        }),
        translation: [
            p.translation[0] + 0.05,
            p.translation[1] - 0.04,
            p.translation[2] + 0.03,
        ],
        scale: p.scale,
    }
}

#[test]
fn register_recovers_the_placement_and_undoes_as_one_step() {
    let mut scene = Scene::new(SceneOptions::default());
    let buffers = blob_buffers(32, 64, [1.0; 3], [0.0; 3]);
    let fixed = add(&mut scene, mesh(&buffers));
    let moving = add(&mut scene, mesh(&buffers));
    let target = Placement {
        rotation: rotation(0.7, 0.4),
        translation: [10.0, -20.0, 5.0],
        scale: [1.0; 3],
    };
    scene.set_placement(fixed, target).unwrap();
    let start = nudged(&target);
    scene.set_placement(moving, start).unwrap();
    let _ = scene.take_changes();

    let r = scene
        .register(moving, fixed, &RegisterOptions::default())
        .unwrap();
    assert_eq!(r.status, IcpStatus::Converged, "{r:?}");
    assert!(r.converged && r.applied, "{r:?}");
    assert!(close(&r.placement, &target, 1e-5), "{r:?}");
    assert!(r.rmse < 1e-5, "{r:?}");
    assert_eq!(r.inliers, r.samples);
    assert_eq!(scene.placement(moving).unwrap(), r.placement);
    let changes = scene.take_changes();
    assert!(
        changes
            .iter()
            .any(|c| c.id == moving && c.placement_changed)
    );
    // One undo step: it restores the start, and the next undo is the
    // `set_placement` before it.
    assert!(scene.undo().unwrap());
    assert_eq!(scene.placement(moving).unwrap(), start);
    assert_eq!(scene.placement(fixed).unwrap(), target);
    assert!(scene.undo().unwrap());
    assert_eq!(scene.placement(moving).unwrap(), Placement::IDENTITY);
    assert!(scene.redo().unwrap());
    assert!(scene.redo().unwrap());
    assert_eq!(scene.placement(moving).unwrap(), r.placement);
}

#[test]
fn apply_false_returns_the_placement_without_touching_the_scene() {
    let mut scene = Scene::new(SceneOptions::default());
    let buffers = blob_buffers(24, 48, [1.0; 3], [0.0; 3]);
    let fixed = add(&mut scene, mesh(&buffers));
    let moving = add(&mut scene, mesh(&buffers));
    let start = nudged(&Placement::IDENTITY);
    scene.set_placement(moving, start).unwrap();
    let _ = scene.take_changes();
    let options = RegisterOptions {
        apply: false,
        ..RegisterOptions::default()
    };
    let r = scene.register(moving, fixed, &options).unwrap();
    assert!(r.converged && !r.applied, "{r:?}");
    assert!(close(&r.placement, &Placement::IDENTITY, 1e-5), "{r:?}");
    assert_eq!(scene.placement(moving).unwrap(), start);
    assert!(scene.take_changes().is_empty());
    // Nothing recorded: the next undo is the `set_placement`.
    assert!(scene.undo().unwrap());
    assert_eq!(scene.placement(moving).unwrap(), Placement::IDENTITY);
}

#[test]
fn a_result_that_did_not_converge_is_not_applied() {
    // A plane on a plane cannot tell where it is along the plane.
    let mut scene = Scene::new(SceneOptions::default());
    let fixed = add(&mut scene, plane(30));
    let moving = add(&mut scene, plane(10));
    let start = Placement {
        translation: [4.0, 3.0, 0.5],
        ..Placement::IDENTITY
    };
    scene.set_placement(moving, start).unwrap();
    let r = scene
        .register(moving, fixed, &RegisterOptions::default())
        .unwrap();
    assert_eq!(r.status, IcpStatus::Underconstrained, "{r:?}");
    assert!(!r.converged && !r.applied, "{r:?}");
    assert_eq!(scene.placement(moving).unwrap(), start);
    // The returned estimate is finite, with z fixed and x, y left alone.
    assert!(r.placement.translation[2].abs() < 1e-5, "{r:?}");
    assert!((r.placement.translation[0] - 4.0).abs() < 1e-5, "{r:?}");
    assert!((r.placement.translation[1] - 3.0).abs() < 1e-5, "{r:?}");
    assert!(r.rmse.is_finite() && r.rmse < 1e-5, "{r:?}");
    assert!(r.conditioning.is_finite());
}

#[test]
fn moving_scale_is_kept_and_not_solved_for() {
    // The stored moving mesh is the shape divided by a per-axis scale.
    let scale = [2.0, 0.5, 4.0];
    let mut scene = Scene::new(SceneOptions::default());
    let fixed = add(&mut scene, mesh(&blob_buffers(32, 64, [1.0; 3], [0.0; 3])));
    let moving = add(&mut scene, mesh(&blob_buffers(32, 64, scale, [0.0; 3])));
    let target = Placement {
        scale,
        ..Placement::IDENTITY
    };
    scene.set_placement(moving, nudged(&target)).unwrap();
    let r = scene
        .register(moving, fixed, &RegisterOptions::default())
        .unwrap();
    assert!(r.converged, "{r:?}");
    assert_eq!(r.placement.scale, scale);
    assert!(close(&r.placement, &target, 1e-5), "{r:?}");
    assert!(r.rmse < 1e-5, "{r:?}");
}

#[test]
fn fixed_non_uniform_scale_is_followed_in_world_units() {
    let scale = [2.0, 0.5, 4.0];
    let mut scene = Scene::new(SceneOptions::default());
    let fixed = add(&mut scene, mesh(&blob_buffers(32, 64, scale, [0.0; 3])));
    let moving = add(&mut scene, mesh(&blob_buffers(32, 64, [1.0; 3], [0.0; 3])));
    let on_fixed = Placement {
        rotation: rotation(0.3, 0.2),
        translation: [1.0, 2.0, 3.0],
        scale,
    };
    scene.set_placement(fixed, on_fixed).unwrap();
    let target = Placement {
        scale: [1.0; 3],
        ..on_fixed
    };
    scene.set_placement(moving, nudged(&target)).unwrap();
    let r = scene
        .register(moving, fixed, &RegisterOptions::default())
        .unwrap();
    assert!(r.converged, "{r:?}");
    assert!(close(&r.placement, &target, 1e-5), "{r:?}");
    assert!(r.rmse < 1e-5, "{r:?}");
    // The rmse is in world units: 0.75 above a plane scaled by 3 (in the
    // fixed mesh's stored units it would read 0.25).
    let mut scene = Scene::new(SceneOptions::default());
    let fixed = add(&mut scene, plane(10));
    let moving = add(&mut scene, plane(4));
    scene
        .set_placement(
            fixed,
            Placement {
                scale: [3.0, 3.0, 3.0],
                ..Placement::IDENTITY
            },
        )
        .unwrap();
    scene
        .set_placement(
            moving,
            Placement {
                translation: [5.0, 5.0, 0.75],
                ..Placement::IDENTITY
            },
        )
        .unwrap();
    let measure = RegisterOptions {
        icp: IcpOptions {
            max_iterations: 0,
            ..IcpOptions::default()
        },
        ..RegisterOptions::default()
    };
    let r = scene.register(moving, fixed, &measure).unwrap();
    assert!((r.rmse - 0.75).abs() < 1e-6, "{r:?}");
    assert!(!r.converged && !r.applied, "{r:?}");
}

#[test]
fn both_meshes_scaled_differently() {
    // One shape, stored divided by a different scale per mesh.
    let (scale_f, scale_m) = ([2.0, 0.5, 4.0], [0.25, 3.0, 1.5]);
    let mut scene = Scene::new(SceneOptions::default());
    let fixed = add(&mut scene, mesh(&blob_buffers(32, 64, scale_f, [0.0; 3])));
    let moving = add(&mut scene, mesh(&blob_buffers(32, 64, scale_m, [0.0; 3])));
    let on_fixed = Placement {
        rotation: rotation(-0.5, 0.3),
        translation: [-3.0, 1.0, 2.0],
        scale: scale_f,
    };
    scene.set_placement(fixed, on_fixed).unwrap();
    let target = Placement {
        scale: scale_m,
        ..on_fixed
    };
    let start = nudged(&target);
    scene.set_placement(moving, start).unwrap();
    let r = scene
        .register(moving, fixed, &RegisterOptions::default())
        .unwrap();
    assert!(r.converged && r.applied, "{r:?}");
    assert_eq!(r.placement.scale, scale_m);
    assert_eq!(scene.placement(fixed).unwrap(), on_fixed);
    assert!(close(&r.placement, &target, 1e-5), "{r:?}");
    assert!(r.rmse < 1e-5, "{r:?}");
    assert!(scene.undo().unwrap());
    assert_eq!(scene.placement(moving).unwrap(), start);
}

#[test]
fn an_undetermined_pose_is_not_converged_or_applied_and_names_the_motion() {
    // A capped cylinder along the world x axis, resampled: its spin is free.
    let cylinder = |rings: u32, around: u32| {
        let mut positions = Vec::new();
        for i in 0..=rings {
            let x = 2.0 * f64::from(i) / f64::from(rings);
            for j in 0..around {
                let a = std::f64::consts::TAU * f64::from(j) / f64::from(around);
                positions.extend([x as f32, a.cos() as f32, a.sin() as f32]);
            }
        }
        let (start, end) = ((rings + 1) * around, (rings + 1) * around + 1);
        positions.extend([0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let mut indices = Vec::new();
        for j in 0..around {
            let next = (j + 1) % around;
            indices.extend([
                start,
                next,
                j,
                end,
                rings * around + j,
                rings * around + next,
            ]);
            for i in 0..rings {
                let (a, b) = (i * around + j, i * around + next);
                indices.extend([a, b, b + around, a, b + around, a + around]);
            }
        }
        Mesh::from_buffers(&positions, &indices).unwrap()
    };
    let mut scene = Scene::new(SceneOptions::default());
    let fixed = add(&mut scene, cylinder(40, 128));
    let moving = add(&mut scene, cylinder(23, 97));
    let start = Placement {
        translation: [0.02, 0.01, -0.01],
        ..Placement::IDENTITY
    };
    scene.set_placement(moving, start).unwrap();
    let options = RegisterOptions {
        icp: IcpOptions {
            tolerance: 1e-3,
            ..IcpOptions::default()
        },
        ..RegisterOptions::default()
    };
    let r = scene.register(moving, fixed, &options).unwrap();
    assert_eq!(r.status, IcpStatus::Underconstrained, "{r:?}");
    assert!(!r.converged && !r.applied, "{r:?}");
    assert_eq!(r.not_applied, Some(NotApplied::NotConverged));
    assert_eq!(scene.placement(moving).unwrap(), start);
    assert_eq!(r.undetermined.len(), 1, "{r:?}");
    // In world coordinates: a spin about x.
    let w = r.undetermined[0].rotation;
    assert!(w[0].abs() > 10.0 * w[1].abs().max(w[2].abs()), "{r:?}");
}

#[test]
fn a_wrong_minimum_is_not_applied_unless_the_limit_allows_it() {
    // Nearly half a turn off: it settles, locally determined, on a wrong fit.
    let mut scene = Scene::new(SceneOptions::default());
    let buffers = blob_buffers(32, 64, [1.0; 3], [0.0; 3]);
    let fixed = add(&mut scene, mesh(&buffers));
    let moving = add(&mut scene, mesh(&buffers));
    let start = Placement {
        rotation: rotation(2.8, 0.0),
        ..Placement::IDENTITY
    };
    scene.set_placement(moving, start).unwrap();
    let mut options = RegisterOptions {
        icp: IcpOptions {
            tolerance: 1e-3,
            ..IcpOptions::default()
        },
        ..RegisterOptions::default()
    };
    let r = scene.register(moving, fixed, &options).unwrap();
    assert!(r.converged, "{r:?}");
    assert!(!r.applied, "{r:?}");
    assert_eq!(r.not_applied, Some(NotApplied::RmseAboveLimit));
    assert!(r.rmse > r.max_rmse, "{r:?}");
    assert_eq!(scene.placement(moving).unwrap(), start);
    options.max_rmse = Some(f64::INFINITY);
    let r = scene.register(moving, fixed, &options).unwrap();
    assert!(r.applied && r.not_applied.is_none(), "{r:?}");
    options.max_rmse = Some(-1.0);
    assert!(matches!(
        scene.register(moving, fixed, &options),
        Err(SceneError::InvalidInput(_))
    ));
}

#[test]
fn a_fit_resting_on_a_small_patch_is_not_applied() {
    // The cutoff admits only the points near where the patch happens to touch.
    let mut scene = Scene::new(SceneOptions::default());
    let buffers = blob_buffers(32, 64, [1.0; 3], [0.0; 3]);
    let fixed = add(&mut scene, mesh(&buffers));
    let moving = add(&mut scene, mesh(&buffers));
    scene
        .set_placement(
            moving,
            Placement {
                translation: [2.2, 0.0, 0.0],
                ..Placement::IDENTITY
            },
        )
        .unwrap();
    let options = RegisterOptions {
        icp: IcpOptions {
            max_distance: 0.25,
            tolerance: 1e-3,
            ..IcpOptions::default()
        },
        max_rmse: Some(f64::INFINITY),
        ..RegisterOptions::default()
    };
    let r = scene.register(moving, fixed, &options).unwrap();
    assert_eq!(r.not_applied, Some(NotApplied::TooFewInliers), "{r:?}");
    assert!(r.inliers * 2 < r.samples, "{r:?}");
}

#[test]
fn origins_far_from_zero_work() {
    // Stored around 1000 and placed back near zero: the solver's frames must
    // carry the origins through f64.
    let mut scene = Scene::new(SceneOptions::default());
    let far = [1000.0, -2000.0, 500.0];
    let (positions, indices) = blob_buffers(24, 48, [1.0; 3], far);
    let fixed_mesh = Mesh::from_buffers_f64(&positions, &indices).unwrap();
    assert_ne!(fixed_mesh.origin(), [0.0; 3]);
    let moving_mesh = Mesh::from_buffers_f64(&positions, &indices).unwrap();
    let fixed = add(&mut scene, fixed_mesh);
    let moving = add(&mut scene, moving_mesh);
    let target = Placement {
        rotation: rotation(0.2, 0.1),
        translation: [-1000.0, 2000.0, -500.0],
        scale: [1.0; 3],
    };
    scene.set_placement(fixed, target).unwrap();
    scene.set_placement(moving, nudged(&target)).unwrap();
    let r = scene
        .register(moving, fixed, &RegisterOptions::default())
        .unwrap();
    assert!(r.converged, "{r:?}");
    assert!(close(&r.placement, &target, 1e-4), "{r:?}");
    assert!(r.rmse < 1e-4, "{r:?}");
}

#[test]
fn max_samples_caps_the_points_used() {
    let mut scene = Scene::new(SceneOptions::default());
    let buffers = blob_buffers(48, 96, [1.0; 3], [0.0; 3]);
    let fixed = add(&mut scene, mesh(&buffers));
    let moving = add(&mut scene, mesh(&buffers));
    scene
        .set_placement(moving, nudged(&Placement::IDENTITY))
        .unwrap();
    let options = RegisterOptions {
        max_samples: 500,
        ..RegisterOptions::default()
    };
    let r = scene.register(moving, fixed, &options).unwrap();
    assert_eq!(r.samples, 500);
    assert_eq!(r.inliers, 500);
    assert!(r.converged, "{r:?}");
    assert!(close(&r.placement, &Placement::IDENTITY, 1e-5), "{r:?}");
}

#[test]
fn stale_ids_invalid_input_and_manual_bvhs_are_errors() {
    let mut scene = Scene::new(SceneOptions::default());
    let buffers = blob_buffers(8, 16, [1.0; 3], [0.0; 3]);
    let fixed = add(&mut scene, mesh(&buffers));
    let moving = add(&mut scene, mesh(&buffers));
    let options = RegisterOptions::default();
    assert!(matches!(
        scene.register(moving, moving, &options),
        Err(SceneError::InvalidInput(_))
    ));
    let none = RegisterOptions {
        max_samples: 0,
        ..options
    };
    assert!(matches!(
        scene.register(moving, fixed, &none),
        Err(SceneError::InvalidInput(_))
    ));
    let gone = add(&mut scene, mesh(&buffers));
    scene.remove_mesh(gone).unwrap();
    assert_eq!(
        scene.register(gone, fixed, &options).unwrap_err(),
        SceneError::UnknownMesh
    );
    assert_eq!(
        scene.register(moving, gone, &options).unwrap_err(),
        SceneError::UnknownMesh
    );
    // A stale Manual fixed BVH is the caller's to update.
    scene.set_mode(fixed, UpdateMode::Manual).unwrap();
    scene
        .edit(fixed, |tx| -> Result<(), SceneError> {
            tx.set_positions(&[0], &[[0.0, 0.0, 1.1]])?;
            Ok(())
        })
        .unwrap();
    assert_eq!(
        scene.register(moving, fixed, &options).unwrap_err(),
        SceneError::StaleBvh(fixed)
    );
    scene.update(fixed).unwrap();
    assert!(scene.register(moving, fixed, &options).is_ok());
}

#[test]
fn results_do_not_depend_on_the_parallel_feature() {
    // Enough samples for the query batch to go parallel with the feature on.
    let buffers = blob_buffers(32, 64, [1.0; 3], [0.0; 3]);
    let run = || {
        let mut scene = Scene::new(SceneOptions::default());
        let fixed = add(&mut scene, mesh(&buffers));
        let moving = add(&mut scene, mesh(&buffers));
        scene
            .set_placement(moving, nudged(&Placement::IDENTITY))
            .unwrap();
        scene
            .register(moving, fixed, &RegisterOptions::default())
            .unwrap()
    };
    par::set_force_sequential(true);
    let sequential = run();
    par::set_force_sequential(false);
    assert_eq!(run(), sequential);
}
