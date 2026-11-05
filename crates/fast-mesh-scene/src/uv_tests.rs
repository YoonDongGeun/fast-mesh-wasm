//! The scene-level rows of `docs/uv-channel.md`: join, cut, boolean, compact,
//! undo and redo, and the change reports.

use fast_mesh_ops::boundary_loops;
use fast_mesh_ops::simplify::{SimplifyOptions, simplify};
use fast_mesh_ops::subdivide::{SubdivideMethod, SubdivideOptions, subdivide};
use fast_mesh_store::Mesh;

use crate::{BooleanOp, CutOptions, JoinOptions, MeshId, MeshOptions, Plane, Scene, SceneOptions};

fn box_mesh(at: f32) -> Mesh {
    let mut positions = Vec::new();
    for z in [0.0, 1.0] {
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            positions.extend([x + at, y, z]);
        }
    }
    Mesh::from_buffers(
        &positions,
        &[
            0, 2, 1, 0, 3, 2, //
            4, 5, 6, 4, 6, 7, //
            0, 1, 5, 0, 5, 4, //
            2, 3, 7, 2, 7, 6, //
            1, 2, 6, 1, 6, 5, //
            3, 0, 4, 3, 4, 7, //
        ],
    )
    .unwrap()
}

/// `u = x / 4, v = y / 4`.
fn planar(mut mesh: Mesh) -> Mesh {
    let uvs: Vec<f32> = mesh
        .positions()
        .iter()
        .flat_map(|p| [p[0] / 4.0, p[1] / 4.0])
        .collect();
    mesh.add_uvs(&uvs).unwrap();
    mesh
}

fn add(scene: &mut Scene, mesh: Mesh) -> MeshId {
    scene.add_mesh(mesh, MeshOptions::default()).unwrap()
}

fn assert_planar(mesh: &Mesh) {
    let uvs = mesh.uvs().unwrap();
    assert_eq!(uvs.len(), 2 * mesh.vertex_slots());
    let o = mesh.origin();
    for v in 0..mesh.vertex_slots() as u32 {
        if let Some(p) = mesh.position(v) {
            let want = [
                (o[0] + f64::from(p[0])) as f32 / 4.0,
                (o[1] + f64::from(p[1])) as f32 / 4.0,
            ];
            let got = &uvs[2 * v as usize..2 * v as usize + 2];
            assert!(
                (want[0] - got[0]).abs() < 1e-5 && (want[1] - got[1]).abs() < 1e-5,
                "vertex {v}: {got:?}, want {want:?}"
            );
        }
    }
}

fn written(scene: &mut Scene, id: MeshId) -> Vec<String> {
    scene
        .take_changes()
        .into_iter()
        .find(|c| c.id == id)
        .map_or(Vec::new(), |c| c.channels_written)
}

#[test]
fn boolean_drops_uvs_reports_it_and_undo_redo_follow() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = add(&mut scene, planar(box_mesh(0.0)));
    let b = add(&mut scene, box_mesh(0.5));
    let before = scene.mesh(a).unwrap().uvs().unwrap().to_vec();
    scene.take_changes();

    let r = scene.boolean(a, b, BooleanOp::Union).unwrap();
    assert!(r.uvs_dropped);
    assert!(scene.mesh(a).unwrap().uvs().is_none());
    assert_eq!(written(&mut scene, a), ["uv"]);

    assert!(scene.undo().unwrap());
    let back = scene.mesh(a).unwrap().uvs().unwrap();
    assert_eq!(&back[..before.len()], &before[..]);
    assert_eq!(back.len(), 2 * scene.mesh(a).unwrap().vertex_slots());
    assert_eq!(written(&mut scene, a), ["uv"]);

    assert!(scene.redo().unwrap());
    assert!(scene.mesh(a).unwrap().uvs().is_none());
    assert_eq!(written(&mut scene, a), ["uv"]);
}

#[test]
fn a_boolean_with_itself_keeps_uvs() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = add(&mut scene, planar(box_mesh(0.0)));
    let r = scene.boolean(a, a, BooleanOp::Union).unwrap();
    assert!(!r.uvs_dropped);
    assert_planar(scene.mesh(a).unwrap());
}

#[test]
fn evicting_a_boolean_from_the_history_forgets_the_uvs() {
    let mut scene = Scene::new(SceneOptions {
        history_budget: 0,
        ..SceneOptions::default()
    });
    let a = add(&mut scene, planar(box_mesh(0.0)));
    let b = add(&mut scene, box_mesh(0.5));
    scene.boolean(a, b, BooleanOp::Union).unwrap();
    assert!(!scene.can_undo());
    assert!(scene.mesh(a).unwrap().uvs().is_none());
    // The dropped values are freed: new UVs start a fresh channel.
    let mesh = scene.mesh(a).unwrap();
    assert!(!format!("{mesh:?}").contains("uv"));
}

#[test]
fn cut_gives_both_sides_aligned_uvs_and_undoes() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = add(&mut scene, planar(box_mesh(0.0)));
    let before = scene.mesh(a).unwrap().uvs().unwrap().to_vec();
    let r = scene
        .cut_by_plane(
            a,
            &CutOptions {
                plane: Plane::new([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap(),
                cap: true,
            },
        )
        .unwrap();
    assert_planar(scene.mesh(a).unwrap());
    assert_planar(scene.mesh(r.added).unwrap());
    assert!(scene.undo().unwrap());
    assert_eq!(
        &scene.mesh(a).unwrap().uvs().unwrap()[..before.len()],
        &before[..]
    );
}

/// Two open quads facing each other, for a join.
fn quads(a_uvs: bool, b_uvs: bool) -> (Scene, MeshId, MeshId, JoinOptions) {
    let mut a = Mesh::from_buffers(
        &[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
        &[0, 2, 1, 0, 3, 2],
    )
    .unwrap();
    let mut b = Mesh::from_buffers(
        &[0., 0., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1.],
        &[0, 1, 2, 0, 2, 3],
    )
    .unwrap();
    let options = JoinOptions {
        loop_a: boundary_loops(&a).unwrap()[0].id,
        loop_b: boundary_loops(&b).unwrap()[0].id,
        max_edges: None,
    };
    if a_uvs {
        a.add_uvs(&[0.1; 8]).unwrap();
    }
    if b_uvs {
        b.add_uvs(&[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            .unwrap();
    }
    let mut scene = Scene::new(SceneOptions::default());
    let a = add(&mut scene, a);
    let b = add(&mut scene, b);
    (scene, a, b, options)
}

#[test]
fn join_with_uvs_on_both_sides_carries_them_over() {
    let (mut scene, a, b, options) = quads(true, true);
    let r = scene.join(a, b, &options).unwrap();
    assert!(!r.uvs_dropped);
    let uvs = scene.mesh(a).unwrap().uvs().unwrap();
    assert_eq!(&uvs[..8], &[0.1; 8]);
    assert_eq!(&uvs[8..16], &[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
}

#[test]
fn join_with_uvs_on_one_side_drops_them_and_undo_restores_the_targets() {
    let (mut scene, a, b, options) = quads(true, false);
    let r = scene.join(a, b, &options).unwrap();
    assert!(r.uvs_dropped);
    assert!(scene.mesh(a).unwrap().uvs().is_none());
    assert!(scene.undo().unwrap());
    assert_eq!(&scene.mesh(a).unwrap().uvs().unwrap()[..8], &[0.1; 8]);

    let (mut scene, a, b, options) = quads(false, true);
    let r = scene.join(a, b, &options).unwrap();
    assert!(r.uvs_dropped);
    assert!(scene.mesh(a).unwrap().uvs().is_none());

    let (mut scene, a, b, options) = quads(false, false);
    assert!(!scene.join(a, b, &options).unwrap().uvs_dropped);
}

#[test]
fn compact_remaps_uvs() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = add(&mut scene, planar(box_mesh(0.0)));
    let plane = Plane::new([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap();
    scene
        .cut_by_plane(a, &CutOptions { plane, cap: false })
        .unwrap();
    let slots = scene.mesh(a).unwrap().vertex_slots();
    scene.compact(a).unwrap();
    let mesh = scene.mesh(a).unwrap();
    assert!(mesh.vertex_slots() < slots);
    assert_planar(mesh);
}

#[test]
fn undo_and_redo_of_an_interpolating_op_restore_uvs_exactly() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = add(&mut scene, planar(box_mesh(0.0)));
    let refine = SubdivideOptions {
        method: SubdivideMethod::Loop,
        levels: 2,
    };
    scene.edit(a, |tx| subdivide(tx, &refine)).unwrap();
    let before = scene.mesh(a).unwrap().uvs().unwrap().to_vec();
    let options = SimplifyOptions {
        target_triangles: 40,
        preserve_boundary: true,
    };
    scene.edit(a, |tx| simplify(tx, &options)).unwrap();
    let after = scene.mesh(a).unwrap().uvs().unwrap().to_vec();
    assert_ne!(after, before);
    assert!(scene.undo().unwrap());
    assert_eq!(scene.mesh(a).unwrap().uvs().unwrap(), &before[..]);
    assert!(scene.redo().unwrap());
    assert_eq!(scene.mesh(a).unwrap().uvs().unwrap(), &after[..]);
}
