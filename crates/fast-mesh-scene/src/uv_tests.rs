//! The scene-level rows of `docs/uv-channel.md`: join, cut, boolean, compact,
//! undo and redo, and the change reports.

use fast_mesh_ops::simplify::{SimplifyOptions, simplify};
use fast_mesh_ops::subdivide::{SubdivideMethod, SubdivideOptions, subdivide};
use fast_mesh_ops::{OpsError, boundary_loops};
use fast_mesh_store::Mesh;

use crate::{
    BooleanOp, CutOptions, JoinOptions, MeshId, MeshOptions, Placement, Plane, Scene, SceneError,
    SceneOptions,
};

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

/// A closed box whose `x = at` face is its own UV island: its four corners are
/// split from the rest, so the mesh is open until those are welded.
fn seamed_box(at: f32, uvs: bool) -> Mesh {
    let whole = box_mesh(at);
    let mut positions: Vec<f32> = whole.positions().iter().flatten().copied().collect();
    let mut triangles: Vec<[u32; 3]> = whole.triangles().to_vec();
    // Corners 0, 3, 4 and 7 lie on the face made of the last two triangles.
    let extra: Vec<u32> = [0u32, 3, 4, 7]
        .iter()
        .map(|&v| {
            let k = (positions.len() / 3) as u32;
            let p = whole.positions()[v as usize];
            positions.extend(p);
            k
        })
        .collect();
    let face = triangles.len() - 2;
    for t in &mut triangles[face..] {
        *t = t.map(|v| match v {
            0 => extra[0],
            3 => extra[1],
            4 => extra[2],
            7 => extra[3],
            v => v,
        });
    }
    let indices: Vec<u32> = triangles.iter().flatten().copied().collect();
    let mut mesh = Mesh::from_buffers(&positions, &indices).unwrap();
    if uvs {
        let mut values = vec![0.25; 2 * mesh.vertex_slots()];
        for u in &mut values[16..] {
            *u = 0.75;
        }
        mesh.add_uvs(&values).unwrap();
    }
    mesh
}

/// Six times the signed volume of `mesh`, in its local frame.
fn volume6(mesh: &Mesh) -> f64 {
    let mut total = 0.0;
    for t in 0..mesh.triangle_slots() as u32 {
        if let Some(tri) = mesh.triangle(t) {
            let [a, b, c] = tri.map(|v| mesh.position(v).unwrap().map(f64::from));
            total += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
    }
    total
}

#[test]
fn a_closed_mesh_with_uv_seams_can_be_combined_and_undoes_exactly() {
    for op in [BooleanOp::Union, BooleanOp::Subtract, BooleanOp::Intersect] {
        let mut scene = Scene::new(SceneOptions::default());
        let a = add(&mut scene, seamed_box(0.0, true));
        let b = add(&mut scene, seamed_box(0.5, true));
        let before = scene.mesh(a).unwrap().uvs().unwrap().to_vec();
        let (vertices, triangles) = {
            let m = scene.mesh(a).unwrap();
            (m.vertex_count(), m.triangle_count())
        };
        let r = scene.boolean(a, b, op).unwrap();
        assert!(r.uvs_dropped);
        let mesh = scene.mesh(a).unwrap();
        assert!(mesh.uvs().is_none());
        assert!(boundary_loops(mesh).unwrap().is_empty(), "{op:?} is closed");

        // The same boolean on whole boxes gives the same solid.
        let mut plain = Scene::new(SceneOptions::default());
        let pa = add(&mut plain, box_mesh(0.0));
        let pb = add(&mut plain, box_mesh(0.5));
        plain.boolean(pa, pb, op).unwrap();
        let (got, want) = (volume6(mesh), volume6(plain.mesh(pa).unwrap()));
        assert!((got - want).abs() < 1e-5, "{op:?}: {got} vs {want}");
        // The tool keeps its seam and UVs.
        assert_eq!(scene.mesh(b).unwrap().vertex_count(), 12);
        assert!(scene.mesh(b).unwrap().uvs().is_some());

        assert!(scene.undo().unwrap());
        let mesh = scene.mesh(a).unwrap();
        assert_eq!(mesh.vertex_count(), vertices);
        assert_eq!(mesh.triangle_count(), triangles);
        assert_eq!(&mesh.uvs().unwrap()[..before.len()], &before[..]);
        assert!(scene.redo().unwrap());
        assert!(scene.mesh(a).unwrap().uvs().is_none());
    }
}

#[test]
fn a_seamed_mesh_combines_under_a_scaled_placement_too() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = add(&mut scene, seamed_box(0.0, true));
    let b = add(&mut scene, seamed_box(0.0, true));
    let placement = Placement {
        translation: [0.5, 0.0, 0.0],
        scale: [1.0, 1.0, 1.0],
        ..Placement::IDENTITY
    };
    scene.set_placement(b, placement).unwrap();
    let scaled = Placement {
        scale: [2.0, 1.0, 1.0],
        ..placement
    };
    scene.set_placement(b, scaled).unwrap();
    let r = scene.boolean(a, b, BooleanOp::Subtract).unwrap();
    assert!(r.uvs_dropped);
    let mesh = scene.mesh(a).unwrap();
    assert!(boundary_loops(mesh).unwrap().is_empty());
    assert!(
        (volume6(mesh) - 6.0 * 0.5).abs() < 1e-5,
        "{}",
        volume6(mesh)
    );
}

#[test]
fn booleans_still_reject_what_is_really_open() {
    // UVs on a mesh with a real hole: welding does not close it.
    let mut scene = Scene::new(SceneOptions::default());
    let mut open = box_mesh(0.0);
    open.add_uvs(&[0.0; 16]).unwrap();
    let mut tx = open.begin_edit();
    tx.remove_triangles(&[0]).unwrap();
    tx.commit().unwrap();
    let a = add(&mut scene, open);
    let b = add(&mut scene, box_mesh(0.5));
    let uvs = scene.mesh(a).unwrap().uvs().unwrap().to_vec();
    assert_eq!(
        scene.boolean(a, b, BooleanOp::Union).unwrap_err(),
        SceneError::Ops(OpsError::InvalidInput("boolean needs closed meshes"))
    );
    assert_eq!(scene.mesh(a).unwrap().uvs().unwrap(), &uvs[..], "untouched");
    // A seam without UVs is not known to be one: rejected as before.
    let mut scene = Scene::new(SceneOptions::default());
    let a = add(&mut scene, seamed_box(0.0, false));
    let b = add(&mut scene, box_mesh(0.5));
    assert_eq!(
        scene.boolean(a, b, BooleanOp::Union).unwrap_err(),
        SceneError::Ops(OpsError::InvalidInput("boolean needs closed meshes"))
    );
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
    let mut scene = Scene::new(SceneOptions { history_budget: 0 });
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

/// Vertices, triangles, positions and UVs of `id`, for comparing states.
fn state(scene: &Scene, id: MeshId) -> (usize, usize, Vec<[f32; 3]>, Vec<f32>) {
    let m = scene.mesh(id).unwrap();
    let alive = (0..m.vertex_slots() as u32).filter(|&v| m.is_vertex_alive(v));
    let positions = alive.clone().map(|v| m.position(v).unwrap()).collect();
    let uvs = m.uvs().unwrap();
    let uvs = alive.flat_map(|v| [uvs[2 * v as usize], uvs[2 * v as usize + 1]]);
    (
        m.vertex_count(),
        m.triangle_count(),
        positions,
        uvs.collect(),
    )
}

#[test]
fn a_cut_across_a_seam_closes_both_sides_reports_the_cap_and_undoes() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = add(&mut scene, seamed_box(0.0, true));
    let whole = volume6(scene.mesh(a).unwrap());
    let before = state(&scene, a);
    // Crosses the seamed face, whose corners are split.
    let plane = Plane::new([0.0, 0.5, 0.0], [0.0, 1.0, 0.0]).unwrap();
    let r = scene
        .cut_by_plane(a, &CutOptions { plane, cap: true })
        .unwrap();
    let (kept, removed) = (scene.mesh(a).unwrap(), scene.mesh(r.added).unwrap());
    // Each of the two rims closes against a seam, with at least two triangles.
    assert!(r.cap_triangles >= 4, "{r:?}");
    assert_eq!(r.kept_triangles, kept.triangle_count());
    assert_eq!(r.removed_triangles, removed.triangle_count());
    // Half the solid on each side, wound outward: a side left open or a cap
    // wound wrong would not add up.
    let (vk, vr) = (volume6(kept), volume6(removed));
    assert!((vk - whole / 2.0).abs() < 1e-4, "{vk} of {whole}");
    assert!((vr - whole / 2.0).abs() < 1e-4, "{vr} of {whole}");
    let after = state(&scene, a);

    // The undo brings back the seam exactly, and the redo the cap.
    assert!(scene.undo().unwrap());
    assert_eq!(state(&scene, a), before);
    assert!(scene.redo().unwrap());
    assert_eq!(state(&scene, a), after);

    // What comes out is closed by position, which a boolean checks for itself.
    let tool = add(&mut scene, seamed_box(0.25, true));
    let b = scene.boolean(a, tool, BooleanOp::Union).unwrap();
    assert!(b.uvs_dropped);
    assert!(boundary_loops(scene.mesh(a).unwrap()).unwrap().is_empty());
}

#[test]
fn a_failed_cut_across_a_seam_changes_nothing() {
    // The seamed face wound against the rest: closed by position, but the rim
    // runs against itself. The seamed face is the last two triangles.
    let seamed = seamed_box(0.0, true);
    let positions: Vec<f32> = seamed.positions().iter().flatten().copied().collect();
    let mut triangles = seamed.triangles().to_vec();
    let face = triangles.len() - 2;
    for t in &mut triangles[face..] {
        t.swap(1, 2);
    }
    let indices: Vec<u32> = triangles.iter().flatten().copied().collect();
    let mut mesh = Mesh::from_buffers(&positions, &indices).unwrap();
    mesh.add_uvs(seamed.uvs().unwrap()).unwrap();
    let mut scene = Scene::new(SceneOptions::default());
    let a = add(&mut scene, mesh);
    let before = state(&scene, a);
    let plane = Plane::new([0.0, 0.5, 0.0], [0.0, 1.0, 0.0]).unwrap();
    let e = scene
        .cut_by_plane(a, &CutOptions { plane, cap: true })
        .unwrap_err();
    assert_eq!(
        e,
        SceneError::Ops(OpsError::InvalidInput("a cut rim cannot be capped"))
    );
    assert_eq!(state(&scene, a), before);
    assert!(!scene.undo().unwrap(), "nothing was recorded");
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
