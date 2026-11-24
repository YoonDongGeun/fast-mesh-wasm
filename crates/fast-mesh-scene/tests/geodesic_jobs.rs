//! Geodesic queries of scene meshes in steps, and the operators kept
//! between them.

use fast_mesh_ops::primitives::{IcosphereOptions, Primitive, generate};
use fast_mesh_scene::{
    GeodesicMethod, GeodesicOptions, GeodesicPhase, MeshId, MeshOptions, Placement, Scene,
    SceneError, SceneOptions,
};
use fast_mesh_store::{Mesh, StoreError};

fn sphere() -> Mesh {
    let s = generate(&Primitive::Icosphere(IcosphereOptions {
        radius: 1.0,
        subdivisions: 4,
    }))
    .unwrap();
    Mesh::from_buffers(s.positions.as_flattened(), s.triangles.as_flattened()).unwrap()
}

fn scaled_sphere(scale: [f32; 3]) -> Mesh {
    let m = sphere();
    let p: Vec<[f32; 3]> = m
        .positions()
        .iter()
        .map(|v| std::array::from_fn(|k| v[k] * scale[k]))
        .collect();
    Mesh::from_buffers(p.as_flattened(), m.triangles().as_flattened()).unwrap()
}

fn scene_with(mesh: Mesh) -> (Scene, MeshId) {
    let mut scene = Scene::new(SceneOptions::default());
    let id = scene.add_mesh(mesh, MeshOptions::default()).unwrap();
    (scene, id)
}

const HEAT: GeodesicOptions = GeodesicOptions {
    method: GeodesicMethod::HeatMethod,
    max_distance: None,
    time_step_scale: 1.0,
};

fn distances(scene: &Scene, id: MeshId, sources: &[u32], options: &GeodesicOptions) -> Vec<f32> {
    let mut out = Vec::new();
    scene
        .geodesic_distances(id, sources, options, &mut out)
        .unwrap();
    out
}

/// Whether the mesh's operators are kept: a query starts with them.
fn kept(scene: &Scene, id: MeshId) -> bool {
    let job = scene.begin_geodesic_distances(id, &[0], &HEAT).unwrap();
    assert_eq!(job.mesh(), id);
    match job.phase() {
        GeodesicPhase::Assemble => false,
        GeodesicPhase::Diffuse => true,
        other => panic!("starts with {other:?}"),
    }
}

fn nudge(scene: &mut Scene, id: MeshId, vertex: u32) {
    let p = scene.mesh(id).unwrap().positions()[vertex as usize].map(|c| c * 1.05);
    scene
        .edit(id, |tx| {
            tx.set_positions(&[vertex], &[p])?;
            Ok::<_, StoreError>(())
        })
        .unwrap();
}

#[test]
fn steps_change_nothing_and_operators_are_kept() {
    let (scene, id) = scene_with(sphere());
    assert!(!kept(&scene, id));
    let first = distances(&scene, id, &[3], &HEAT);
    assert!(kept(&scene, id));
    // With the operators of the first query, in steps, and from anew.
    for _ in 0..2 {
        let mut job = scene.begin_geodesic_distances(id, &[3], &HEAT).unwrap();
        let mut steps = 0;
        while !scene.step_geodesic_distances(&mut job, 1).unwrap() {
            steps += 1;
        }
        assert!(steps > 10, "{steps} steps");
        assert_eq!(job.phase(), GeodesicPhase::Done);
        assert_eq!(scene.finish_geodesic_distances(job).unwrap(), first);
        scene.release_geodesics();
        assert!(!kept(&scene, id));
    }
    // Finishing runs what is left.
    let mut job = scene.begin_geodesic_distances(id, &[3], &HEAT).unwrap();
    let _ = scene.step_geodesic_distances(&mut job, 1).unwrap();
    assert_eq!(scene.finish_geodesic_distances(job).unwrap(), first);

    // The graph method keeps no operators, and drops none.
    let graph = GeodesicOptions {
        method: GeodesicMethod::Dijkstra,
        ..HEAT
    };
    assert_ne!(distances(&scene, id, &[3], &graph), first);
    assert!(kept(&scene, id));
    scene.release_geodesics();
    distances(&scene, id, &[3], &graph);
    assert!(!kept(&scene, id));
}

#[test]
fn a_cancelled_query_leaves_its_operators() {
    let (scene, id) = scene_with(sphere());
    let mut job = scene.begin_geodesic_distances(id, &[0], &HEAT).unwrap();
    while matches!(
        job.phase(),
        GeodesicPhase::Assemble | GeodesicPhase::Factorize
    ) {
        assert!(!kept(&scene, id));
        assert!(!scene.step_geodesic_distances(&mut job, 1).unwrap());
    }
    drop(job);
    assert!(kept(&scene, id));
}

#[test]
fn edits_drop_the_operators() {
    let (mut scene, id) = scene_with(sphere());
    let other = scene.add_mesh(sphere(), MeshOptions::default()).unwrap();
    let before = distances(&scene, id, &[0], &HEAT);
    let mut job = scene.begin_geodesic_distances(id, &[0], &HEAT).unwrap();
    assert!(!scene.step_geodesic_distances(&mut job, 1).unwrap());

    nudge(&mut scene, other, 9);
    assert!(kept(&scene, id));
    nudge(&mut scene, id, 40);
    assert!(!kept(&scene, id));
    // The query under way does not go on with the changed mesh.
    assert!(matches!(
        scene.step_geodesic_distances(&mut job, 1),
        Err(SceneError::InvalidInput(_))
    ));
    let after = distances(&scene, id, &[0], &HEAT);
    assert_ne!(after, before);
    assert!(kept(&scene, id));

    assert!(scene.undo().unwrap());
    assert!(!kept(&scene, id));
    assert_eq!(distances(&scene, id, &[0], &HEAT), before);
    // A mesh that leaves the scene takes them with it, and returns without.
    scene.remove_mesh_undoable(id).unwrap();
    assert!(matches!(
        scene.begin_geodesic_distances(id, &[0], &HEAT),
        Err(SceneError::UnknownMesh)
    ));
    assert!(scene.undo().unwrap());
    assert!(!kept(&scene, id));
    assert_eq!(distances(&scene, id, &[0], &HEAT), before);
}

#[test]
fn one_mesh_keeps_operators_at_a_time() {
    let (mut scene, a) = scene_with(sphere());
    let b = scene
        .add_mesh(scaled_sphere([1.0, 1.5, 1.0]), MeshOptions::default())
        .unwrap();
    distances(&scene, a, &[0], &HEAT);
    assert!(kept(&scene, a) && !kept(&scene, b));
    distances(&scene, b, &[0], &HEAT);
    assert!(!kept(&scene, a) && kept(&scene, b));
    scene.remove_mesh(b).unwrap();
    distances(&scene, a, &[0], &HEAT);
    assert!(kept(&scene, a));
}

#[test]
fn distances_are_in_world_lengths() {
    let (mut scene, id) = scene_with(sphere());
    let local = distances(&scene, id, &[0], &HEAT);
    let far = local.iter().copied().fold(0.0f32, f32::max);
    let turn = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];

    // Moved, turned and scaled alike: the same operators.
    scene
        .set_placement(
            id,
            Placement {
                rotation: turn,
                translation: [5.0, 6.0, -7.0],
                scale: [2.0; 3],
            },
        )
        .unwrap();
    assert!(kept(&scene, id));
    let world = distances(&scene, id, &[0], &HEAT);
    for (w, l) in world.iter().zip(&local) {
        assert_eq!(*w, (f64::from(*l) * 2.0) as f32);
    }
    let near = GeodesicOptions {
        max_distance: Some(far),
        ..HEAT
    };
    let cut = distances(&scene, id, &[0], &near);
    assert!(cut.iter().any(|d| d.is_infinite()));
    for (c, w) in cut.iter().zip(&world) {
        assert!(*c == *w || (c.is_infinite() && *w > far * 0.999));
    }

    // Scaled per axis: the distances of the scaled shape.
    let axes = [1.0, 3.0, 0.5];
    let place = |translation: [f64; 3], scale: [f64; 3]| Placement {
        rotation: turn,
        translation,
        scale,
    };
    scene.set_placement(id, place([0.0; 3], axes)).unwrap();
    assert!(!kept(&scene, id));
    let got = distances(&scene, id, &[0], &HEAT);
    let (shape, as_shape) = scene_with(scaled_sphere(axes.map(|c| c as f32)));
    assert_eq!(got, distances(&shape, as_shape, &[0], &HEAT));
    // Moving it keeps the scaled operators, scaling it otherwise does not.
    scene
        .set_placement(id, place([9.0, 0.0, 1.0], axes))
        .unwrap();
    assert!(kept(&scene, id));
    assert_eq!(distances(&scene, id, &[0], &HEAT), got);
    scene
        .set_placement(id, place([9.0, 0.0, 1.0], [1.0, 3.0, 0.6]))
        .unwrap();
    assert!(!kept(&scene, id));
    let _ = distances(&scene, id, &[0], &HEAT);
    nudge(&mut scene, id, 12);
    assert!(!kept(&scene, id));
    // And back to none.
    scene.set_placement(id, Placement::IDENTITY).unwrap();
    assert!(!kept(&scene, id));
}

#[test]
fn bad_arguments_are_refused_at_the_start() {
    let (scene, id) = scene_with(sphere());
    for (sources, options) in [
        (vec![], HEAT),
        (vec![u32::MAX], HEAT),
        (
            vec![0],
            GeodesicOptions {
                time_step_scale: 0.0,
                ..HEAT
            },
        ),
        (
            vec![0],
            GeodesicOptions {
                max_distance: Some(-1.0),
                ..HEAT
            },
        ),
    ] {
        assert!(
            scene
                .begin_geodesic_distances(id, &sources, &options)
                .is_err()
        );
        let mut out = vec![1.5];
        assert!(
            scene
                .geodesic_distances(id, &sources, &options, &mut out)
                .is_err()
        );
        assert_eq!(out, [1.5]);
    }
}
