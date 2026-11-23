//! Undoable adds and removals: ids and data come back identical, mixed steps
//! undo together, the budget frees parked meshes, the registries and reports
//! follow, and a model-based property test over random sequences.

use std::collections::{BTreeMap, BTreeSet};

use fast_mesh_fields::{Colormap, OutOfRange};
use fast_mesh_store::{ChannelDesc, Mesh, StoreError, TriangleId, VertexId};
use proptest::prelude::*;
use proptest::test_runner::RngSeed;

use crate::normals::NormalState;
use crate::scene::fault;
use crate::{
    CollisionMode, CollisionRule, ColorRule, Coloring, CsgNode, CutOptions, DistanceField, FieldId,
    MeshBuild, MeshId, MeshOptions, Placement, Plane, Scene, SceneClosestOptions, SceneError,
    SceneOptions, SectionDesc, SignMode, SubscriberId, ThicknessField, UpdateMode,
};

fn grid(n: u32) -> Mesh {
    let mut pos = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            pos.extend_from_slice(&[x as f32, y as f32, 0.0]);
        }
    }
    let mut idx = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            let (b, c) = (a + 1, a + n + 1);
            idx.extend_from_slice(&[a, b, c, b, c + 1, c]);
        }
    }
    let mut mesh = Mesh::from_buffers(&pos, &idx).unwrap();
    let uvs: Vec<f32> = mesh
        .positions()
        .iter()
        .flat_map(|p| [p[0] * 0.25, p[1] * 0.5])
        .collect();
    mesh.add_uvs(&uvs).unwrap();
    mesh
}

/// A closed cube, outward wound.
fn cube(at: [f32; 3]) -> Mesh {
    let mut pos = Vec::new();
    for z in [0.0, 1.0] {
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            pos.extend_from_slice(&[x + at[0], y + at[1], z + at[2]]);
        }
    }
    Mesh::from_buffers(
        &pos,
        &[
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6,
            5, 3, 0, 4, 3, 4, 7,
        ],
    )
    .unwrap()
}

fn moved(x: f64) -> Placement {
    Placement {
        translation: [x, 0.0, 0.0],
        ..Placement::IDENTITY
    }
}

fn undoable() -> MeshOptions {
    MeshOptions {
        undoable: true,
        ..MeshOptions::default()
    }
}

fn lift(scene: &mut Scene, id: MeshId, v: u32, z: f32) {
    scene
        .edit(id, |tx| {
            let p = tx.mesh().position(v).unwrap();
            tx.set_positions(&[v], &[[p[0], p[1], z]])
        })
        .unwrap();
}

/// Everything a mesh holds that edits and placements change, slot by slot.
#[derive(Debug, Clone, PartialEq)]
struct Snap {
    vertices: BTreeMap<VertexId, ([u32; 3], Option<[u32; 2]>)>,
    triangles: BTreeMap<TriangleId, [VertexId; 3]>,
    placement: Placement,
}

fn vertex(mesh: &Mesh, v: VertexId) -> Option<([u32; 3], Option<[u32; 2]>)> {
    let p = mesh.position(v)?;
    let uv = mesh.uvs().map(|u| {
        let i = 2 * v as usize;
        [u[i].to_bits(), u[i + 1].to_bits()]
    });
    Some((p.map(f32::to_bits), uv))
}

fn snap(scene: &Scene, id: MeshId) -> Snap {
    let mesh = scene.mesh(id).unwrap();
    Snap {
        vertices: (0..mesh.vertex_slots() as u32)
            .filter_map(|v| Some((v, vertex(mesh, v)?)))
            .collect(),
        triangles: (0..mesh.triangle_slots() as u32)
            .filter_map(|t| Some((t, mesh.triangle(t)?)))
            .collect(),
        placement: scene.placement(id).unwrap(),
    }
}

fn restorable(scene: &Scene) -> Vec<MeshId> {
    scene.restorable_ids().collect()
}

fn assert_valid(scene: &Scene, id: MeshId) {
    let e = scene.entries.get(id).unwrap();
    e.bvh.validate(&e.mesh).unwrap();
}

#[test]
fn add_undo_redo_keeps_id_and_data() {
    for mode in [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual] {
        for (defer_bvh, built) in [(false, false), (true, false), (false, true)] {
            let mut scene = Scene::new(SceneOptions::default());
            let other = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
            assert!(!scene.can_undo(), "adds are permanent by default");
            let options = MeshOptions {
                placement: moved(2.0),
                mode,
                defer_bvh,
                ..undoable()
            };
            let id = if built {
                let mut build = MeshBuild::new(grid(3), options).unwrap();
                build.set_normals(true);
                scene.add_mesh_built(build).unwrap()
            } else {
                scene.add_mesh(grid(3), options).unwrap()
            };
            scene.update(id).unwrap();
            assert!(scene.can_undo());
            let before = snap(&scene, id);
            let uid = scene.mesh(id).unwrap().uid();
            let normals = scene.normals(id).unwrap().map(<[_]>::to_vec);
            assert_eq!(normals.is_some(), built);

            assert_eq!(scene.undo(), Ok(true));
            assert_eq!(scene.mesh(id).err(), Some(SceneError::UnknownMesh));
            assert_eq!(scene.mesh_id(id.index(), id.generation()), None);
            assert_eq!(scene.ids().collect::<Vec<_>>(), [other]);
            assert_eq!(restorable(&scene), [id]);
            assert!(scene.is_restorable(id) && !scene.is_restorable(other));
            let parts = (id.index(), id.generation());
            assert_eq!(scene.restorable_id(parts.0, parts.1), Some(id));
            assert_eq!(scene.restorable_id(parts.0, parts.1 + 1), None);
            let parts = (other.index(), other.generation());
            assert_eq!(scene.restorable_id(parts.0, parts.1), None);
            assert!(!scene.can_undo() && scene.can_redo());
            // A parked mesh is out of reach of every call.
            assert_eq!(
                scene.set_placement(id, Placement::IDENTITY),
                Err(SceneError::UnknownMesh)
            );
            assert_eq!(scene.remove_mesh_undoable(id), Err(SceneError::UnknownMesh));
            assert_eq!(scene.remove_mesh(id).err(), Some(SceneError::UnknownMesh));

            fault::take_rebuilds();
            assert_eq!(scene.redo(), Ok(true));
            assert_eq!(snap(&scene, id), before);
            assert_eq!(scene.mesh(id).unwrap().uid(), uid, "the same mesh, moved");
            assert_eq!(scene.mode(id), Ok(mode));
            assert_eq!(
                scene.is_current(id),
                Ok(true),
                "the BVH came back as it was"
            );
            assert_eq!(fault::take_rebuilds(), 0);
            assert_eq!(
                scene.normals(id).unwrap().map(<[_]>::to_vec),
                normals,
                "normals came back as they were"
            );
            assert_valid(&scene, id);
            assert!(restorable(&scene).is_empty());
            assert_eq!(scene.undo(), Ok(true));
            assert_eq!(scene.redo(), Ok(true));
            assert_eq!(snap(&scene, id), before);
        }
    }
}

#[test]
fn remove_undo_redo_keeps_id_data_and_display_state() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = scene.add_mesh(grid(4), MeshOptions::default()).unwrap();
    let b = scene
        .add_mesh(cube([1.0, 1.0, 0.5]), MeshOptions::default())
        .unwrap();
    scene.set_normals(a, true).unwrap();
    scene
        .add_channel(a, ChannelDesc::derived("mine", 1))
        .unwrap();
    scene.derived_channel_mut(a, "mine").unwrap()[3] = 7.0;
    lift(&mut scene, a, 6, 0.25);
    scene.set_placement(a, moved(0.5)).unwrap();
    let near = scene
        .add_distance_field(DistanceField::new("near", a, b, 10.0, SignMode::Auto))
        .unwrap();
    let wall = scene
        .add_thickness_field(ThicknessField {
            name: "wall".into(),
            on: a,
            max_distance: 5.0,
        })
        .unwrap();
    let coloring = Coloring {
        field: near,
        rule: ColorRule {
            map: Colormap::Turbo,
            range: (-1.0, 1.0),
            center: None,
            below: OutOfRange::Clamp,
            above: OutOfRange::Clamp,
            base: [1, 2, 3, 255],
            nan: [9, 9, 9, 255],
        },
    };
    scene.set_coloring(a, Some(coloring.clone())).unwrap();
    scene.flush().unwrap();
    scene.take_changes();
    let before = snap(&scene, a);
    let values = scene.field_values(near).unwrap().to_vec();
    let colors = scene.colors(a).unwrap().unwrap().to_vec();
    let normals = scene.normals(a).unwrap().unwrap().to_vec();
    let uid = scene.mesh(a).unwrap().uid();

    for _ in 0..2 {
        scene.remove_mesh_undoable(a).unwrap();
        assert_eq!(scene.mesh(a).err(), Some(SceneError::UnknownMesh));
        assert_eq!(
            scene.field_values(near).err(),
            Some(SceneError::UnknownField)
        );
        assert_eq!(
            scene.field_values(wall).err(),
            Some(SceneError::UnknownField)
        );
        assert!(scene.entries.get(b).unwrap().hooks.is_empty());
        // The scene goes on without it.
        scene.flush().unwrap();
        assert_eq!(scene.ids().collect::<Vec<_>>(), [b]);

        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(snap(&scene, a), before);
        assert_eq!(scene.mesh(a).unwrap().uid(), uid);
        assert_eq!(scene.coloring(a).unwrap(), Some(&coloring));
        assert_eq!(scene.normals(a).unwrap().unwrap(), normals);
        assert_eq!(scene.mesh(a).unwrap().channel("mine").unwrap()[3], 7.0);
        scene.flush().unwrap();
        assert!(scene.field_status(near).unwrap().current);
        assert!(scene.field_status(wall).unwrap().current);
        assert_eq!(scene.field_values(near).unwrap(), values);
        assert_eq!(scene.colors(a).unwrap().unwrap(), colors);
        assert_valid(&scene, a);

        assert_eq!(scene.redo(), Ok(true));
        assert_eq!(scene.mesh(a).err(), Some(SceneError::UnknownMesh));
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(snap(&scene, a), before);
        scene.flush().unwrap();
        assert_eq!(scene.field_values(near).unwrap(), values);
        // Edits from before the removal still undo.
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(scene.placement(a), Ok(Placement::IDENTITY));
        assert_eq!(scene.redo(), Ok(true));
        assert_eq!(snap(&scene, a), before);
        assert_eq!(scene.redo(), Ok(true), "the removal is redone last");
        assert_eq!(scene.undo(), Ok(true));
    }
}

#[test]
fn mixed_steps_undo_together() {
    for mode in [UpdateMode::Eager, UpdateMode::Lazy] {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let before = snap(&scene, a);
        scene.take_changes();

        // The pattern tools use: add empty, fill, move, and remove an input,
        // all in one step.
        scene.begin_step();
        let n = scene
            .add_mesh(Mesh::new(), MeshOptions { mode, ..undoable() })
            .unwrap();
        scene
            .edit(n, |tx| {
                let v = tx.add_vertices(&[[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])?;
                tx.add_triangles(&[[v[0], v[1], v[2]]])
            })
            .unwrap();
        scene.set_placement(n, moved(3.0)).unwrap();
        lift(&mut scene, a, 4, 1.0);
        scene.remove_mesh_undoable(a).unwrap();
        scene.end_step().unwrap();
        let after = snap(&scene, n);
        assert_eq!(scene.ids().collect::<Vec<_>>(), [n]);
        assert_eq!(restorable(&scene), [a]);
        assert_eq!(scene.history.undo_len(), 1);

        for _ in 0..3 {
            assert_eq!(scene.undo(), Ok(true));
            assert_eq!(scene.ids().collect::<Vec<_>>(), [a]);
            assert_eq!(restorable(&scene), [n]);
            assert_eq!(snap(&scene, a), before);
            assert_valid(&scene, a);
            assert!(!scene.can_undo());
            assert_eq!(scene.redo(), Ok(true));
            assert_eq!(scene.ids().collect::<Vec<_>>(), [n]);
            assert_eq!(snap(&scene, n), after);
            scene.flush().unwrap();
            assert_valid(&scene, n);
            let hit = scene
                .closest_point_on(n, [3.2, 0.2, 5.0], &SceneClosestOptions::default())
                .unwrap()
                .unwrap();
            assert!((hit.distance - 5.0).abs() < 1e-6);
            assert!(!scene.can_redo());
        }
    }
}

#[test]
fn cancel_takes_adds_back_and_restores_removals() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
    let before = snap(&scene, a);
    scene.take_changes();
    scene.begin_step();
    let n = scene.add_mesh(grid(1), undoable()).unwrap();
    lift(&mut scene, n, 0, 2.0);
    lift(&mut scene, a, 0, 2.0);
    scene.remove_mesh_undoable(a).unwrap();
    scene.cancel_step().unwrap();
    assert_eq!(scene.ids().collect::<Vec<_>>(), [a]);
    assert_eq!(snap(&scene, a), before);
    assert!(restorable(&scene).is_empty(), "the cancelled add is freed");
    assert!(!scene.can_undo() && !scene.can_redo());
    assert_eq!(scene.history.bytes(), 0);
    // `a` was never taken as removed: no removal, at most a full change.
    let ch = scene.take_changes();
    assert!(ch.iter().all(|c| c.id == a && !c.removed && !c.added));
    // The freed slot is reused under a new id.
    let m = scene.add_mesh(grid(1), MeshOptions::default()).unwrap();
    assert_eq!(m.index(), n.index());
    assert_ne!(m, n);
    assert_eq!(scene.mesh(n).err(), Some(SceneError::UnknownMesh));
}

#[test]
fn a_mesh_added_and_removed_in_one_step() {
    let mut scene = Scene::new(SceneOptions::default());
    scene.begin_step();
    let n = scene.add_mesh(grid(1), undoable()).unwrap();
    scene.remove_mesh_undoable(n).unwrap();
    scene.end_step().unwrap();
    assert!(scene.is_restorable(n));
    for _ in 0..2 {
        assert_eq!(scene.undo(), Ok(true));
        assert!(scene.mesh(n).is_err() && scene.is_restorable(n));
        assert_eq!(scene.redo(), Ok(true));
        assert!(scene.mesh(n).is_err() && scene.is_restorable(n));
    }
    assert!(scene.take_changes().is_empty());
    scene.clear_history().unwrap();
    assert!(!scene.is_restorable(n));
}

#[test]
fn dropping_the_last_way_back_frees_the_mesh() {
    // A new step drops an undone add.
    let mut scene = Scene::new(SceneOptions::default());
    let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
    let n = scene.add_mesh(grid(3), undoable()).unwrap();
    scene.undo().unwrap();
    assert!(scene.is_restorable(n));
    lift(&mut scene, a, 0, 1.0);
    assert!(!scene.is_restorable(n));
    assert!(!scene.can_redo());

    // `clear_history` makes a removal permanent and an add final.
    let n = scene.add_mesh(grid(3), undoable()).unwrap();
    scene.remove_mesh_undoable(a).unwrap();
    scene.clear_history().unwrap();
    assert_eq!(scene.history.bytes(), 0);
    assert!(!scene.is_restorable(a));
    assert!(scene.mesh(n).is_ok());
    assert!(!scene.can_undo() && !scene.can_redo());

    // Redoing and recording over an undone removal keeps the live mesh.
    scene.remove_mesh_undoable(n).unwrap();
    scene.undo().unwrap();
    lift(&mut scene, n, 0, 1.0);
    assert!(scene.mesh(n).is_ok() && restorable(&scene).is_empty());

    // Patches of a parked mesh are released before it goes: undo of the
    // removal after an eviction of older steps still works.
    scene.remove_mesh_undoable(n).unwrap();
    scene.undo().unwrap();
    assert_eq!(scene.mesh(n).unwrap().position(0).unwrap()[2], 1.0);
    scene.undo().unwrap();
    assert_eq!(scene.mesh(n).unwrap().position(0).unwrap()[2], 0.0);
}

#[test]
fn budget_counts_and_frees_parked_meshes() {
    let bytes_of = |n: u32| {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(grid(n), MeshOptions::default()).unwrap();
        scene.remove_mesh_undoable(id).unwrap();
        scene.history.bytes()
    };
    let one = bytes_of(20);
    let raw = 21 * 21 * (12 + 8) + 800 * 12;
    assert!(one > raw, "{one} bytes for a mesh of at least {raw}");
    assert!(bytes_of(40) > 3 * one, "it grows with the mesh");

    // Room for two removals of that size, not three.
    let mut scene = Scene::new(SceneOptions {
        history_budget: 2 * one + one / 2,
    });
    let ids: Vec<MeshId> = (0..4)
        .map(|_| scene.add_mesh(grid(20), MeshOptions::default()).unwrap())
        .collect();
    for (k, &id) in ids.iter().enumerate() {
        scene.remove_mesh_undoable(id).unwrap();
        assert_eq!(scene.history.undo_len(), (k + 1).min(2));
        assert_eq!(restorable(&scene).len(), (k + 1).min(2));
        assert!(scene.history.bytes() <= 2 * one + one / 2);
    }
    assert_eq!(restorable(&scene), ids[2..]);
    // The evicted ids are gone for good, their slots reused.
    let fresh = scene.add_mesh(grid(1), MeshOptions::default()).unwrap();
    assert!(ids[..2].iter().any(|i| i.index() == fresh.index()));
    assert!(!ids.contains(&fresh));
    assert_eq!(scene.undo(), Ok(true));
    assert_eq!(scene.undo(), Ok(true));
    assert_eq!(scene.undo(), Ok(false));
    assert!(scene.mesh(ids[2]).is_ok() && scene.mesh(ids[3]).is_ok());

    // An undoable add holds nothing until it is undone. The furthest redo
    // goes first, so that every add can be undone.
    let mut scene = Scene::new(SceneOptions {
        history_budget: 2 * one + one / 2,
    });
    let ids: Vec<MeshId> = (0..4)
        .map(|_| scene.add_mesh(grid(20), undoable()).unwrap())
        .collect();
    assert_eq!(scene.history.undo_len(), 4);
    assert_eq!(scene.history.bytes(), 0);
    while scene.undo().unwrap() {}
    assert_eq!(scene.ids().count(), 0);
    assert_eq!(restorable(&scene), ids[..2]);
    assert_eq!(scene.history.redo_len(), 2);
    while scene.redo().unwrap() {}
    assert_eq!(scene.ids().collect::<Vec<_>>(), ids[..2]);
    assert_eq!(scene.history.bytes(), 0);

    // Undone, redone and removed: the mesh counts once, on whichever entry
    // holds it.
    scene.remove_mesh_undoable(ids[0]).unwrap();
    let held = scene.history.bytes();
    assert!(held > one / 2 && held < 2 * one);
    scene.undo().unwrap();
    assert_eq!(scene.history.bytes(), 0);
    scene.redo().unwrap();
    assert_eq!(scene.history.bytes(), held);

    // Larger than the whole budget: a removal is dropped as its step closes
    // (it is permanent), an add undoes and cannot be redone, and neither
    // costs the history before it.
    let mut scene = Scene::new(SceneOptions {
        history_budget: 4096,
    });
    let a = scene.add_mesh(grid(20), MeshOptions::default()).unwrap();
    lift(&mut scene, a, 0, 1.0);
    let n = scene.add_mesh(grid(20), undoable()).unwrap();
    assert_eq!(scene.history.undo_len(), 2);
    assert_eq!(scene.undo(), Ok(true));
    assert!(scene.mesh(n).is_err() && !scene.is_restorable(n));
    assert!(scene.can_undo() && !scene.can_redo());
    assert_eq!(scene.undo(), Ok(true));
    assert_eq!(scene.mesh(a).unwrap().position(0).unwrap()[2], 0.0);
    scene.remove_mesh_undoable(a).unwrap();
    assert!(scene.mesh(a).is_err() && !scene.is_restorable(a));
    assert_eq!(scene.history.bytes(), 0);
    let mut scene = Scene::new(SceneOptions { history_budget: 64 });

    // An open step is not evicted while open; its undo happens whatever its
    // size, then the budget frees what it parked.
    let a = scene.add_mesh(grid(20), MeshOptions::default()).unwrap();
    scene.take_changes();
    scene.begin_step();
    let n = scene.add_mesh(grid(20), undoable()).unwrap();
    scene.remove_mesh_undoable(a).unwrap();
    assert!(scene.can_undo() && scene.is_restorable(a));
    assert_eq!(scene.undo(), Ok(true));
    assert!(scene.mesh(a).is_ok() && scene.mesh(n).is_err());
    assert!(!scene.is_restorable(n) && !scene.can_redo());
    scene.take_changes();
    assert_eq!(scene.ids().collect::<Vec<_>>(), [a]);
}

#[test]
fn a_mesh_that_grew_meets_the_budget_when_its_removal_is_redone() {
    let run = |budget: usize, grow: bool| {
        let mut scene = Scene::new(SceneOptions {
            history_budget: budget,
        });
        let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
        let m = scene.add_mesh(grid(20), MeshOptions::default()).unwrap();
        lift(&mut scene, a, 0, 1.0);
        scene.remove_mesh_undoable(m).unwrap();
        let bytes = scene.history.bytes();
        scene.undo().unwrap();
        if grow {
            // Not history: the mesh holds more when it is parked again.
            scene.set_normals(m, true).unwrap();
        }
        scene.take_changes();
        assert_eq!(scene.redo(), Ok(true));
        assert!(scene.mesh(m).is_err());
        (scene, m, bytes)
    };
    let (scene, m, bytes) = run(usize::MAX, false);
    assert!(scene.is_restorable(m) && scene.history.bytes() == bytes);
    let (scene, m, _) = run(usize::MAX, true);
    let grown = scene.history.bytes();
    assert!(scene.is_restorable(m) && grown > bytes);
    // Room for the removal alone: the older edit goes.
    let (mut scene, m, _) = run(grown - 1, true);
    assert!(scene.is_restorable(m) && scene.history.undo_len() == 1);
    assert!(scene.history.bytes() < grown);
    scene.undo().unwrap();
    assert!(scene.mesh(m).is_ok());
    // No room for it: it is redone, and permanent.
    let (mut scene, m, _) = run(bytes, true);
    assert!(!scene.is_restorable(m) && scene.history.bytes() == 0);
    assert!(scene.take_changes().iter().any(|c| c.id == m && c.removed));
}

#[test]
fn evicted_patches_of_a_parked_mesh_release_their_slots() {
    let run = |budget: usize| {
        let mut scene = Scene::new(SceneOptions {
            history_budget: budget,
        });
        let a = scene.add_mesh(grid(1), MeshOptions::default()).unwrap();
        let m = scene.add_mesh(grid(8), MeshOptions::default()).unwrap();
        let cut: Vec<TriangleId> = (0..64).collect();
        scene.edit(m, |tx| tx.remove_triangles(&cut)).unwrap();
        assert!(scene.mesh(m).unwrap().free_triangle_slots().is_empty());
        scene.remove_mesh_undoable(m).unwrap();
        (scene, a, m)
    };
    let (scene, ..) = run(usize::MAX);
    let (mut scene, a, m) = run(scene.history.bytes());
    assert_eq!(scene.history.undo_len(), 2);
    // Evicts the edit of the parked mesh, not its removal.
    scene.set_placement(a, moved(1.0)).unwrap();
    assert_eq!(scene.history.undo_len(), 2);
    assert!(scene.is_restorable(m));
    scene.undo().unwrap();
    scene.undo().unwrap();
    assert_eq!(scene.mesh(m).unwrap().triangle_count(), 64);
    assert_eq!(scene.mesh(m).unwrap().free_triangle_slots().len(), 64);
    // The freed slots are reused.
    scene
        .edit(m, |tx| {
            tx.add_triangles(&[[0, 1, 9]]).map(|t| assert!(t[0] < 64))
        })
        .unwrap();
    assert_valid(&scene, m);
}

#[test]
fn permanent_removal_drops_its_entries_only() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
    scene.begin_step();
    let n = scene.add_mesh(grid(1), undoable()).unwrap();
    lift(&mut scene, n, 0, 1.0);
    lift(&mut scene, a, 0, 1.0);
    scene.end_step().unwrap();
    scene.remove_mesh_undoable(n).unwrap();
    scene.undo().unwrap();
    // `n` is live again, with an add in undo and a removal in redo.
    let mesh = scene.remove_mesh(n).unwrap();
    assert_eq!(mesh.position(0).unwrap()[2], 1.0);
    assert!(restorable(&scene).is_empty());
    assert!(!scene.can_redo(), "the removal's step was left empty");
    assert_eq!(scene.undo(), Ok(true));
    assert_eq!(scene.mesh(a).unwrap().position(0).unwrap()[2], 0.0);
    assert!(scene.mesh(n).is_err());
    assert_eq!(scene.redo(), Ok(true));
    assert!(scene.mesh(n).is_err());

    // A target removed for good while its field's mesh is parked.
    let b = scene
        .add_mesh(cube([0.0; 3]), MeshOptions::default())
        .unwrap();
    let c = scene
        .add_mesh(cube([3.0, 0.0, 0.0]), MeshOptions::default())
        .unwrap();
    let both = scene
        .add_distance_field(DistanceField {
            to: vec![b, c],
            ..DistanceField::new("both", a, b, 100.0, SignMode::Auto)
        })
        .unwrap();
    let only = scene
        .add_distance_field(DistanceField::new("only", a, b, 100.0, SignMode::Auto))
        .unwrap();
    scene.remove_mesh_undoable(a).unwrap();
    scene.remove_mesh(b).unwrap();
    assert_eq!(scene.undo(), Ok(true));
    scene.flush().unwrap();
    assert_eq!(scene.field_targets(both).unwrap(), [c]);
    assert!(scene.field_status(both).unwrap().current);
    assert!(scene.field_nearest(both).unwrap().iter().all(|&n| n == 0));
    assert_eq!(
        scene.field_values(only).err(),
        Some(SceneError::UnknownField)
    );
    assert!(scene.mesh(a).unwrap().channel("field:only").is_none());
    // The name is free again.
    scene
        .add_distance_field(DistanceField::new("only", a, c, 100.0, SignMode::Auto))
        .unwrap();
}

#[test]
fn compact_keeps_add_and_removal_entries() {
    let mut scene = Scene::new(SceneOptions::default());
    let n = scene.add_mesh(grid(2), undoable()).unwrap();
    scene.edit(n, |tx| tx.remove_triangles(&[0, 1])).unwrap();
    scene.compact(n).unwrap();
    let before = snap(&scene, n);
    assert_eq!(
        scene.undo(),
        Ok(true),
        "the add; the edit went with compact"
    );
    assert!(scene.mesh(n).is_err());
    assert_eq!(scene.redo(), Ok(true));
    assert_eq!(snap(&scene, n), before);
    scene.remove_mesh_undoable(n).unwrap();
    assert_eq!(scene.undo(), Ok(true));
    scene.compact(n).unwrap();
    assert_eq!(scene.redo(), Ok(true));
    assert!(scene.mesh(n).is_err());
}

#[test]
fn reports_follow_for_every_subscriber() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
    let late = scene.subscribe();
    scene.take_changes();
    let get = |v: Vec<crate::MeshChanges>| {
        assert!(v.len() <= 1, "one record per mesh");
        v.into_iter().next()
    };

    // Removed, taken by one subscriber only, then back.
    scene.remove_mesh_undoable(a).unwrap();
    let c = get(scene.take_changes()).unwrap();
    assert!(c.id == a && c.removed && !c.added);
    scene.undo().unwrap();
    let c = get(scene.take_changes()).unwrap();
    assert!(c.id == a && c.added && !c.removed && c.changes.is_none());
    assert!(c.colors_dirty.is_none() && c.normals_dirty.is_none());
    // The other one never saw it leave: fully changed instead.
    let c = get(scene.take_changes_for(late).unwrap()).unwrap();
    assert!(c.id == a && !c.added && !c.removed && c.changes.is_none());
    assert!(c.placement_changed);

    // Added, then undone and redone between two calls.
    let n = scene.add_mesh(grid(1), undoable()).unwrap();
    scene.undo().unwrap();
    assert!(scene.take_changes().is_empty(), "added and gone: nothing");
    scene.redo().unwrap();
    let c = get(scene.take_changes()).unwrap();
    assert!(c.id == n && c.added);
    let c = get(scene.take_changes_for(late).unwrap()).unwrap();
    assert!(c.id == n && c.added, "never saw any of it but the add");

    // Subscribing and unsubscribing while a mesh is parked.
    scene.undo().unwrap();
    scene.take_changes();
    let third = scene.subscribe();
    scene.unsubscribe(late).unwrap();
    scene.redo().unwrap();
    let c = get(scene.take_changes()).unwrap();
    assert!(c.id == n && c.added);
    let c = get(scene.take_changes_for(third).unwrap()).unwrap();
    assert!(c.id == n && c.added);
    lift(&mut scene, n, 0, 1.0);
    assert_eq!(scene.take_changes_for(third).unwrap().len(), 1);

    // A failed undo reports nothing it should not.
    scene.begin_step();
    lift(&mut scene, a, 0, 1.0);
    scene.remove_mesh_undoable(n).unwrap();
    scene.end_step().unwrap();
    scene.take_changes();
    fault::fail_patch_on(1);
    assert_eq!(
        scene.undo(),
        Err(SceneError::Store(StoreError::OutOfMemory))
    );
    assert!(scene.take_changes().is_empty());
    assert!(scene.is_restorable(n));
    assert_eq!(scene.undo(), Ok(true));
    let ch = scene.take_changes();
    assert_eq!(
        ch.iter().map(|c| (c.id, c.added)).collect::<Vec<_>>(),
        [(a, false), (n, true)]
    );
}

#[test]
fn failed_steps_leave_every_link_alone() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = scene.add_mesh(grid(3), MeshOptions::default()).unwrap();
    let b = scene
        .add_mesh(cube([1.0, 1.0, 1.0]), MeshOptions::default())
        .unwrap();
    scene.begin_step();
    lift(&mut scene, a, 0, 0.5);
    let n = scene.add_mesh(cube([0.5, 0.5, 0.5]), undoable()).unwrap();
    scene.end_step().unwrap();
    // Links owned by others: undoing the add would cut them.
    let to_n = scene
        .add_distance_field(DistanceField::new("n", a, n, 10.0, SignMode::Auto))
        .unwrap();
    let on_n = scene
        .add_distance_field(DistanceField::new("on", n, b, 10.0, SignMode::Auto))
        .unwrap();
    let section = scene
        .add_section(SectionDesc {
            plane: Plane::new([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]).unwrap(),
            meshes: vec![a, n],
            fill: false,
        })
        .unwrap();
    let rule = CollisionRule {
        mode: CollisionMode::Rigid,
        ..CollisionRule::default()
    };
    scene.set_collision_rule(a, n, rule).unwrap();
    scene.take_changes();

    // The add is parked first, then the edit's revert fails.
    fault::fail_patch_on(1);
    assert_eq!(
        scene.undo(),
        Err(SceneError::Store(StoreError::OutOfMemory))
    );
    assert!(scene.mesh(n).is_ok());
    assert!(scene.field_status(to_n).unwrap().current);
    assert!(scene.field_status(on_n).unwrap().current);
    assert!(scene.section(section, n).is_ok());
    assert_eq!(scene.collision_rule(a, n), Ok(rule));
    assert!(scene.take_changes().is_empty());

    // It works the next time, and cuts what others own.
    assert_eq!(scene.undo(), Ok(true));
    assert_eq!(
        scene.field_status(to_n).err(),
        Some(SceneError::UnknownField)
    );
    assert_eq!(
        scene.field_status(on_n).err(),
        Some(SceneError::UnknownField)
    );
    assert!(scene.section(section, n).is_err());
    assert!(scene.section(section, a).is_ok());
    assert_eq!(scene.redo(), Ok(true));
    scene.flush().unwrap();
    assert!(
        scene.field_status(on_n).unwrap().current,
        "its own came back"
    );
    assert_eq!(
        scene.field_status(to_n).err(),
        Some(SceneError::UnknownField)
    );
    assert!(scene.section(section, n).is_err());
    // Its collision pair rules are its own too (#271).
    assert_eq!(scene.collision_rule(a, n), Ok(rule));
    assert!(scene.mesh(a).unwrap().channel("field:n").is_none());

    // A failed reservation changes nothing either.
    fault::fail_reserve_on(1);
    assert!(scene.remove_mesh_undoable(b).is_err());
    assert!(scene.mesh(b).is_ok());
    fault::fail_reserve_on(1);
    assert!(scene.add_mesh(grid(1), undoable()).is_err());
    assert_eq!(scene.ids().count(), 3);
}

#[test]
fn collision_rules_come_back_once_both_meshes_have() {
    let rule = |contact_margin| CollisionRule {
        mode: CollisionMode::Rigid,
        contact_margin,
        ..CollisionRule::default()
    };
    // The first two leave in either order, in one step or one by one, and
    // come back in the opposite one.
    for (first, second) in [(0, 1), (1, 0)] {
        for together in [false, true] {
            let mut scene = Scene::new(SceneOptions::default());
            let ids: Vec<_> = (0..3)
                .map(|_| scene.add_mesh(grid(2), MeshOptions::default()).unwrap())
                .collect();
            let (a, b, c) = (ids[0], ids[1], ids[2]);
            let rules = [((a, b), 0.25), ((b, c), 0.5), ((c, a), 0.75)];
            for ((x, y), margin) in rules {
                scene.set_collision_rule(x, y, rule(margin)).unwrap();
            }
            let check = |scene: &Scene| {
                for ((x, y), margin) in rules {
                    if scene.mesh(x).is_ok() && scene.mesh(y).is_ok() {
                        assert_eq!(scene.collision_rule(x, y), Ok(rule(margin)));
                    }
                }
                let live = rules
                    .iter()
                    .filter(|((x, y), _)| scene.mesh(*x).is_ok() && scene.mesh(*y).is_ok());
                assert_eq!(scene.collision.pair_rules.len(), live.count());
            };
            if together {
                scene.begin_step();
            }
            for k in [first, second] {
                scene.remove_mesh_undoable(ids[k]).unwrap();
                check(&scene);
                assert!(scene.collision_rule(ids[k], c).is_err());
            }
            if together {
                scene.end_step().unwrap();
            }
            assert!(scene.collision.pair_rules.is_empty());
            assert!(!scene.has_active_collision_response(c));
            for _ in 0..2 {
                while scene.can_undo() {
                    scene.undo().unwrap();
                    check(&scene);
                }
                assert_eq!(scene.collision.pair_rules.len(), 3);
                while scene.can_redo() {
                    scene.redo().unwrap();
                    check(&scene);
                }
                assert!(scene.collision.pair_rules.is_empty());
            }
        }
    }

    // Out by the undo of an add, and back by its redo.
    let mut scene = Scene::new(SceneOptions::default());
    let a = scene.add_mesh(grid(2), undoable()).unwrap();
    let b = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
    scene.set_collision_rule(a, b, rule(0.25)).unwrap();
    scene.remove_mesh_undoable(b).unwrap();
    assert!(scene.undo().unwrap());
    assert_eq!(scene.collision_rule(a, b), Ok(rule(0.25)));
    assert!(scene.undo().unwrap());
    assert!(scene.mesh(a).is_err() && scene.mesh(b).is_ok());
    assert!(scene.collision.pair_rules.is_empty());
    assert!(scene.redo().unwrap());
    assert_eq!(scene.collision_rule(a, b), Ok(rule(0.25)));

    // Its rules with a mesh that is gone for good go.
    scene.clear_history().unwrap();
    scene.remove_mesh_undoable(a).unwrap();
    scene.remove_mesh(b).unwrap();
    assert!(scene.undo().unwrap());
    assert!(scene.mesh(a).is_ok());
    assert!(scene.collision.pair_rules.is_empty());
}

#[test]
fn a_rule_between_meshes_that_return_one_by_one_waits_for_both() {
    let rule = CollisionRule {
        mode: CollisionMode::Rigid,
        ..CollisionRule::default()
    };
    let mut scene = Scene::new(SceneOptions::default());
    let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
    let b = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
    scene.set_collision_rule(a, b, rule).unwrap();
    scene.remove_mesh_undoable(a).unwrap();
    scene.remove_mesh_undoable(b).unwrap();
    // The rule left with `a`, which the history holds behind `b`: `b` is back
    // first, then `a`. Flip them by hand for the opposite order.
    scene.entries.unpark(a);
    scene.settle_presence(&[crate::history::Done::Presence(a)]);
    assert!(scene.mesh(a).is_ok() && scene.mesh(b).is_err());
    assert!(scene.collision.pair_rules.is_empty());
    scene.entries.unpark(b);
    scene.settle_presence(&[crate::history::Done::Presence(b)]);
    assert_eq!(scene.collision_rule(a, b), Ok(rule));
}

#[test]
fn corrupted_step_keeps_meshes_where_they_were() {
    // Undo order: lift b (1), restore n, lift a (2); (2) fails, n is parked
    // again, then b's recovery (3) fails too.
    let mut scene = Scene::new(SceneOptions::default());
    let a = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
    let b = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
    let n = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
    scene.begin_step();
    lift(&mut scene, a, 0, 1.0);
    scene.remove_mesh_undoable(n).unwrap();
    lift(&mut scene, b, 0, 1.0);
    scene.end_step().unwrap();
    fault::fail_patch_on(2);
    fault::fail_patch_on(3);
    assert_eq!(scene.undo(), Err(SceneError::Corrupted));
    // The step is gone, so the removal cannot be undone: freed.
    assert!(!scene.can_undo() && !scene.is_restorable(n));
    assert!(scene.mesh(n).is_err());
    assert_valid(&scene, a);
    assert_valid(&scene, b);

    // The other order: the restore comes first and is never recovered by the
    // failing loop itself.
    let n = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
    let f = scene
        .add_distance_field(DistanceField::new("f", a, n, 10.0, SignMode::Auto))
        .unwrap();
    let mut add = || {
        scene.begin_step();
        lift(&mut scene, a, 1, 1.0);
        lift(&mut scene, b, 1, 1.0);
        let m = scene.add_mesh(grid(1), undoable()).unwrap();
        scene.end_step().unwrap();
        m
    };
    let m = add();
    let to_m = scene
        .add_distance_field(DistanceField::new("m", a, m, 10.0, SignMode::Auto))
        .unwrap();
    fault::fail_patch_on(2);
    fault::fail_patch_on(3);
    assert_eq!(scene.undo(), Err(SceneError::Corrupted));
    assert!(
        scene.mesh(m).is_ok(),
        "the add is put back: it stays for good"
    );
    assert!(scene.field_status(to_m).is_ok() && scene.field_status(f).is_ok());
    scene.flush().unwrap();
    assert!(scene.field_status(to_m).unwrap().current);
    assert!(!scene.can_undo());
}

#[test]
fn meshes_parked_in_one_step_keep_their_fields_between_each_other() {
    let mut scene = Scene::new(SceneOptions::default());
    let other = scene.add_mesh(grid(3), MeshOptions::default()).unwrap();
    scene.begin_step();
    let p = scene.add_mesh(grid(2), undoable()).unwrap();
    let t = scene.add_mesh(cube([0.5, 0.5, 0.5]), undoable()).unwrap();
    scene.end_step().unwrap();
    let pt = scene
        .add_distance_field(DistanceField::new("pt", p, t, 10.0, SignMode::Auto))
        .unwrap();
    let po = scene
        .add_distance_field(DistanceField {
            to: vec![t, other],
            ..DistanceField::new("po", p, t, 10.0, SignMode::Auto)
        })
        .unwrap();
    let op = scene
        .add_distance_field(DistanceField {
            to: vec![p, t],
            ..DistanceField::new("op", other, p, 10.0, SignMode::Auto)
        })
        .unwrap();
    let values = scene.field_values(pt).unwrap().to_vec();
    scene.undo().unwrap();
    for f in [pt, po, op] {
        assert_eq!(scene.field_status(f).err(), Some(SceneError::UnknownField));
    }
    assert!(scene.mesh(other).unwrap().channel("field:op").is_none());
    scene.redo().unwrap();
    // The other mesh's field, measured to them, does not come back.
    assert_eq!(scene.field_status(op).err(), Some(SceneError::UnknownField));
    for _ in 0..3 {
        scene.undo().unwrap();
        assert!(scene.entries.get(other).unwrap().hooks.is_empty());
        scene.redo().unwrap();
        scene.flush().unwrap();
        assert_eq!(scene.field_targets(pt).unwrap(), [t]);
        assert_eq!(scene.field_targets(po).unwrap(), [t, other]);
        assert!(scene.field_status(pt).unwrap().current);
        assert!(scene.field_status(po).unwrap().current);
        assert_eq!(scene.field_values(pt).unwrap(), values);
        // One hook per field, never two.
        assert_eq!(scene.entries.get(t).unwrap().hooks.len(), 2);
        assert_eq!(scene.entries.get(other).unwrap().hooks.len(), 1);
        assert_eq!(scene.entries.get(p).unwrap().hooks.len(), 2);
    }
}

#[test]
fn eager_fields_follow_a_removal_and_its_undo_at_once() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = scene.add_mesh(grid(3), MeshOptions::default()).unwrap();
    let near = scene
        .add_mesh(cube([1.0, 1.0, 0.5]), MeshOptions::default())
        .unwrap();
    let far = scene
        .add_mesh(cube([1.0, 1.0, 5.0]), MeshOptions::default())
        .unwrap();
    let f = scene
        .add_distance_field(DistanceField {
            to: vec![near, far],
            ..DistanceField::new("f", a, near, 100.0, SignMode::Auto)
        })
        .unwrap();
    scene.remove_mesh_undoable(near).unwrap();
    // Recomputed at once to the remaining target.
    assert!(scene.field_status(f).unwrap().current);
    assert_eq!(scene.field_targets(f).unwrap(), [far]);
    assert!(scene.field_values(f).unwrap().iter().all(|&d| d >= 5.0));
    // Links of others are not restored.
    scene.undo().unwrap();
    assert_eq!(scene.field_targets(f).unwrap(), [far]);
    assert!(scene.field_status(f).unwrap().current);
}

#[test]
fn ops_that_create_meshes_undo_as_one_step() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = scene
        .add_mesh(cube([0.0; 3]), MeshOptions::default())
        .unwrap();
    let b = scene
        .add_mesh(cube([0.5, 0.5, 0.5]), MeshOptions::default())
        .unwrap();
    let before = snap(&scene, a);
    let tree = CsgNode::reduce_all(fast_mesh_ops::BooleanOp::Union, 2).unwrap();
    let opts = crate::BooleanOptions::default();

    let r = scene.boolean_batch(&[a, b], &tree, &opts).unwrap();
    let union = snap(&scene, r);
    scene.undo().unwrap();
    assert!(scene.mesh(r).is_err() && !scene.can_undo());
    scene.redo().unwrap();
    assert_eq!(snap(&scene, r), union);
    scene.undo().unwrap();

    // In a caller's step: the result replaces its inputs.
    scene.begin_step();
    let r = scene.boolean_batch(&[a, b], &tree, &opts).unwrap();
    scene.remove_mesh_undoable(a).unwrap();
    scene.remove_mesh_undoable(b).unwrap();
    scene.end_step().unwrap();
    assert_eq!(scene.ids().collect::<Vec<_>>(), [r]);
    scene.undo().unwrap();
    let mut ids: Vec<MeshId> = scene.ids().collect();
    ids.sort_unstable();
    assert_eq!(ids, [a, b]);
    assert_eq!(snap(&scene, a), before);

    // A cut inside an open step stays part of it.
    scene.begin_step();
    let options = CutOptions {
        plane: Plane::new([0.5, 0.5, 0.5], [1.0, 0.0, 0.0]).unwrap(),
        cap: true,
    };
    let cut = scene.cut_by_plane(a, &options).unwrap();
    scene.remove_mesh_undoable(cut.added).unwrap();
    scene.end_step().unwrap();
    assert_eq!(scene.history.undo_len(), 1);
    scene.undo().unwrap();
    assert_eq!(snap(&scene, a), before);
    assert!(scene.mesh(cut.added).is_err());
    // A failed cut records nothing and leaves no step open.
    let miss = CutOptions {
        plane: Plane::new([50.0, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap(),
        cap: true,
    };
    scene.clear_history().unwrap();
    assert!(scene.cut_by_plane(a, &miss).is_err());
    assert_eq!(scene.end_step(), Err(SceneError::StepNotOpen));
    assert!(!scene.can_undo());
}

// ---- Property test ----

/// The case count: `PROPTEST_CASES` when set, else `default`.
fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn config() -> ProptestConfig {
    let mut c = ProptestConfig::with_cases(cases(192));
    c.rng_seed = match std::env::var("PROPTEST_RNG_SEED").as_deref() {
        Ok("random") => RngSeed::Random,
        Ok(s) => s
            .parse()
            .map(RngSeed::Fixed)
            .unwrap_or(RngSeed::Fixed(0x0269_ADD5)),
        Err(_) => RngSeed::Fixed(0x0269_ADD5),
    };
    c.failure_persistence = None;
    c
}

#[derive(Debug, Clone)]
enum Op {
    Add { undoable: bool, n: u32, mode: u8 },
    Remove { undoable: bool, k: usize },
    Move { k: usize, v: usize, z: i8 },
    Grow { k: usize, at: i8 },
    Cut { k: usize, t: usize },
    Place { k: usize, x: i8 },
    Compact { k: usize },
    Field { k: usize, to: usize },
    Begin,
    End,
    Cancel,
    Undo,
    Redo,
    Clear,
    Take { second: bool },
    Flush,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        4 => (any::<bool>(), 1u32..4, 0u8..3).prop_map(|(undoable, n, mode)| Op::Add {
            undoable,
            n,
            mode
        }),
        4 => (prop::bool::weighted(0.8), 0usize..8)
            .prop_map(|(undoable, k)| Op::Remove { undoable, k }),
        5 => (0usize..8, 0usize..32, any::<i8>()).prop_map(|(k, v, z)| Op::Move { k, v, z }),
        3 => (0usize..8, any::<i8>()).prop_map(|(k, at)| Op::Grow { k, at }),
        3 => (0usize..8, 0usize..32).prop_map(|(k, t)| Op::Cut { k, t }),
        3 => (0usize..8, any::<i8>()).prop_map(|(k, x)| Op::Place { k, x }),
        1 => (0usize..8).prop_map(|k| Op::Compact { k }),
        2 => (0usize..8, 0usize..8).prop_map(|(k, to)| Op::Field { k, to }),
        3 => Just(Op::Begin),
        3 => Just(Op::End),
        1 => Just(Op::Cancel),
        9 => Just(Op::Undo),
        7 => Just(Op::Redo),
        1 => Just(Op::Clear),
        4 => any::<bool>().prop_map(|second| Op::Take { second }),
        3 => Just(Op::Flush),
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Edit,
    Place,
    Presence,
}

type State = BTreeMap<MeshId, Snap>;

#[derive(Debug, Clone)]
struct ModelStep {
    before: State,
    after: State,
    entries: Vec<(MeshId, Kind)>,
}

/// The history rules on whole-scene snapshots.
#[derive(Debug, Default)]
struct Model {
    undo: Vec<ModelStep>,
    redo: Vec<ModelStep>,
    /// The state before the open step and its entries.
    open: Option<(State, Vec<(MeshId, Kind)>)>,
    depth: u32,
}

impl Model {
    fn close(&mut self, now: &State) {
        self.depth = 0;
        if let Some((before, entries)) = self.open.take()
            && !entries.is_empty()
        {
            self.undo.push(ModelStep {
                before,
                after: now.clone(),
                entries,
            });
        }
    }

    /// Records an entry that took the scene from `before` to `now`.
    fn record(&mut self, before: &State, now: &State, id: MeshId, kind: Kind) {
        self.redo.clear();
        let (_, entries) = self
            .open
            .get_or_insert_with(|| (before.clone(), Vec::new()));
        // Placement runs of one mesh coalesce in an open step.
        if !(self.depth > 0 && kind == Kind::Place && entries.last() == Some(&(id, kind))) {
            entries.push((id, kind));
        }
        if self.depth == 0 {
            self.close(now);
        }
    }

    /// Runs `f` on every state kept.
    fn each_state(&mut self, mut f: impl FnMut(&mut State)) {
        for s in self.undo.iter_mut().chain(&mut self.redo) {
            f(&mut s.before);
            f(&mut s.after);
        }
        if let Some((before, _)) = &mut self.open {
            f(before);
        }
    }

    /// Drops the entries of `id` that `drop` picks; steps left empty go.
    fn forget(&mut self, id: MeshId, drop: impl Fn(Kind) -> bool) {
        let keep = |e: &(MeshId, Kind)| e.0 != id || !drop(e.1);
        for s in self.undo.iter_mut().chain(&mut self.redo) {
            s.entries.retain(keep);
        }
        if let Some((_, entries)) = &mut self.open {
            entries.retain(keep);
        }
        self.undo.retain(|s| !s.entries.is_empty());
        self.redo.retain(|s| !s.entries.is_empty());
    }

    /// The meshes the history can bring back: named by an add or a removal it
    /// still holds, and not live.
    fn restorable(&self, now: &State) -> BTreeSet<MeshId> {
        let steps = self.undo.iter().chain(&self.redo).map(|s| &s.entries);
        steps
            .chain(self.open.iter().map(|(_, e)| e))
            .flatten()
            .filter(|(id, k)| *k == Kind::Presence && !now.contains_key(id))
            .map(|(id, _)| *id)
            .collect()
    }
}

fn state(scene: &Scene) -> State {
    scene.ids().map(|id| (id, snap(scene, id))).collect()
}

/// What a consumer of change reports knows, kept up from the reports alone.
#[derive(Default)]
struct Mirror {
    meshes: State,
}

impl Mirror {
    fn take(&mut self, scene: &mut Scene, subscriber: Option<SubscriberId>) {
        let changes = match subscriber {
            None => scene.take_changes(),
            Some(s) => scene.take_changes_for(s).unwrap(),
        };
        let mut seen = BTreeSet::new();
        for c in changes {
            assert!(seen.insert(c.id), "one record per mesh");
            if c.removed {
                assert!(!c.added);
                assert!(
                    self.meshes.remove(&c.id).is_some(),
                    "removed a mesh it knew"
                );
                continue;
            }
            if c.added {
                let fresh = snap(scene, c.id);
                assert!(
                    self.meshes.insert(c.id, fresh).is_none(),
                    "added a mesh it did not know"
                );
                continue;
            }
            let known = self.meshes.get_mut(&c.id).expect("changed a mesh it knew");
            let Some(set) = &c.changes else {
                *known = snap(scene, c.id);
                continue;
            };
            let mesh = scene.mesh(c.id).unwrap();
            for v in set.vertices_added.iter().chain(&set.vertices_moved) {
                known.vertices.insert(*v, vertex(mesh, *v).unwrap());
            }
            for (_, ids) in &set.channels {
                for v in ids {
                    known.vertices.insert(*v, vertex(mesh, *v).unwrap());
                }
            }
            for v in &set.vertices_removed {
                known.vertices.remove(v);
            }
            for t in set.triangles_added.iter().chain(&set.triangles_changed) {
                known.triangles.insert(*t, mesh.triangle(*t).unwrap());
            }
            for t in &set.triangles_removed {
                known.triangles.remove(t);
            }
            if !set.channels_toggled.is_empty() {
                for (v, val) in &mut known.vertices {
                    *val = vertex(mesh, *v).unwrap();
                }
            }
            if c.placement_changed {
                known.placement = scene.placement(c.id).unwrap();
            }
        }
        assert_eq!(self.meshes, state(scene), "the reports tell everything");
    }
}

/// A compact copy of a live mesh with the old vertex id of each new one.
fn compact_copy(mesh: &Mesh) -> (Mesh, Vec<VertexId>) {
    let mut ids = Vec::new();
    let mut new_of = BTreeMap::new();
    let mut pos = Vec::new();
    for v in 0..mesh.vertex_slots() as u32 {
        if let Some(p) = mesh.position(v) {
            new_of.insert(v, ids.len() as u32);
            ids.push(v);
            pos.extend_from_slice(&p);
        }
    }
    let idx: Vec<u32> = (0..mesh.triangle_slots() as u32)
        .filter_map(|t| mesh.triangle(t))
        .flat_map(|t| t.map(|v| new_of[&v]))
        .collect();
    (Mesh::from_buffers(&pos, &idx).unwrap(), ids)
}

/// Checks a field that is up to date against a field computed from scratch
/// on compact copies (absolute values: rounding may flip a sign at zero).
fn check_field(scene: &Scene, id: FieldId, on: MeshId) {
    let mut fresh = Scene::new(SceneOptions::default());
    let mut copy = |scene: &Scene, m: MeshId| {
        let (mesh, ids) = compact_copy(scene.mesh(m).unwrap());
        let options = MeshOptions {
            placement: scene.placement(m).unwrap(),
            ..MeshOptions::default()
        };
        (fresh.add_mesh(mesh, options).unwrap(), ids)
    };
    let (a, ids) = copy(scene, on);
    let to: Vec<MeshId> = scene
        .field_targets(id)
        .unwrap()
        .iter()
        .map(|&t| copy(scene, t).0)
        .collect();
    let f = fresh
        .add_distance_field(DistanceField {
            name: "f".into(),
            on: a,
            to,
            max_distance: 1e6,
            sign: SignMode::Pseudonormal,
        })
        .unwrap();
    let want = fresh.field_values(f).unwrap();
    let got = scene.field_values(id).unwrap();
    for (k, &v) in ids.iter().enumerate() {
        let (w, g) = (want[k].abs(), got[v as usize].abs());
        assert!(
            (w.is_nan() && g.is_nan()) || w == g || (w - g).abs() < 1e-3,
            "field at {v}: {g}, want {w}"
        );
    }
}

struct Run {
    scene: Scene,
    second: SubscriberId,
    model: Model,
    now: State,
    mirrors: [Mirror; 2],
    ids: BTreeSet<MeshId>,
    fields: Vec<(FieldId, MeshId)>,
    budget: usize,
}

impl Run {
    fn pick(&self, k: usize) -> Option<MeshId> {
        let n = self.now.len();
        (n > 0).then(|| *self.now.keys().nth(k % n).unwrap())
    }

    /// Notes what an operation that recorded (`recorded`) or did not did.
    fn after(&mut self, recorded: Option<(MeshId, Kind)>) {
        let before = std::mem::replace(&mut self.now, state(&self.scene));
        if let Some((id, kind)) = recorded {
            self.model.record(&before, &self.now, id, kind);
        }
    }

    fn apply(&mut self, op: &Op) {
        match *op {
            Op::Add { undoable, n, mode } => {
                if self.now.len() >= 6 {
                    return;
                }
                let mode = [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual][mode as usize];
                // Manual meshes need explicit updates to be queried.
                let mode = if mode == UpdateMode::Manual {
                    UpdateMode::Eager
                } else {
                    mode
                };
                let options = MeshOptions {
                    mode,
                    undoable,
                    defer_bvh: n == 2,
                    ..MeshOptions::default()
                };
                let id = self.scene.add_mesh(grid(n), options).unwrap();
                assert!(self.ids.insert(id), "ids are never reused");
                if n != 3 {
                    self.scene.set_normals(id, true).unwrap();
                }
                self.after(undoable.then_some((id, Kind::Presence)));
                if !undoable {
                    let s = self.now[&id].clone();
                    self.model.each_state(|st| {
                        st.insert(id, s.clone());
                    });
                }
            }
            Op::Remove { undoable, k } => {
                let Some(id) = self.pick(k) else { return };
                if undoable {
                    self.scene.remove_mesh_undoable(id).unwrap();
                    self.after(Some((id, Kind::Presence)));
                } else {
                    self.scene.remove_mesh(id).unwrap();
                    self.after(None);
                    self.model.forget(id, |_| true);
                    self.model.each_state(|st| {
                        st.remove(&id);
                    });
                }
            }
            Op::Move { k, v, z } => {
                let Some(id) = self.pick(k) else { return };
                let alive: Vec<_> = self.now[&id].vertices.keys().copied().collect();
                if alive.is_empty() {
                    return;
                }
                let v = alive[v % alive.len()];
                let p = self.scene.mesh(id).unwrap().position(v).unwrap();
                let z = f32::from(z) / 32.0;
                if p[2] == z {
                    return;
                }
                lift(&mut self.scene, id, v, z);
                self.after(Some((id, Kind::Edit)));
            }
            Op::Grow { k, at } => {
                let Some(id) = self.pick(k) else { return };
                let x = f32::from(at) / 16.0;
                self.scene
                    .edit(id, |tx| {
                        let v =
                            tx.add_vertices(&[[x, 0.0, 1.0], [x + 1.0, 0.0, 1.0], [x, 1.0, 1.0]])?;
                        tx.add_triangles(&[[v[0], v[1], v[2]]])
                    })
                    .unwrap();
                self.after(Some((id, Kind::Edit)));
            }
            Op::Cut { k, t } => {
                let Some(id) = self.pick(k) else { return };
                let alive: Vec<_> = self.now[&id].triangles.keys().copied().collect();
                if alive.is_empty() {
                    return;
                }
                let t = alive[t % alive.len()];
                self.scene.edit(id, |tx| tx.remove_triangles(&[t])).unwrap();
                self.after(Some((id, Kind::Edit)));
            }
            Op::Place { k, x } => {
                let Some(id) = self.pick(k) else { return };
                let p = moved(f64::from(x) / 8.0);
                if self.now[&id].placement == p {
                    return;
                }
                self.scene.set_placement(id, p).unwrap();
                self.after(Some((id, Kind::Place)));
            }
            Op::Compact { k } => {
                let Some(id) = self.pick(k) else { return };
                self.scene.compact(id).unwrap();
                self.after(None);
                self.model.forget(id, |kind| kind == Kind::Edit);
                let geometry = self.now[&id].clone();
                self.model.each_state(|st| {
                    if let Some(s) = st.get_mut(&id) {
                        s.vertices.clone_from(&geometry.vertices);
                        s.triangles.clone_from(&geometry.triangles);
                    }
                });
            }
            Op::Field { k, to } => {
                let (Some(on), Some(to)) = (self.pick(k), self.pick(to)) else {
                    return;
                };
                if on == to || self.fields.len() >= 6 {
                    return;
                }
                let name = format!("f{}", self.fields.len());
                let mut desc = DistanceField::new(name, on, to, 1e6, SignMode::Pseudonormal);
                // A second target now and then.
                if let Some(&third) = self.now.keys().find(|&&m| m != on && m != to)
                    && k % 2 == 0
                {
                    desc.to.push(third);
                }
                let f = self.scene.add_distance_field(desc).unwrap();
                self.fields.push((f, on));
            }
            Op::Begin => {
                self.scene.begin_step();
                self.model.depth += 1;
                if self.model.open.is_none() {
                    self.model.open = Some((self.now.clone(), Vec::new()));
                }
            }
            Op::End => {
                if self.model.depth == 0 {
                    assert_eq!(self.scene.end_step(), Err(SceneError::StepNotOpen));
                    return;
                }
                self.scene.end_step().unwrap();
                self.model.depth -= 1;
                if self.model.depth == 0 {
                    self.model.close(&self.now);
                }
            }
            Op::Cancel => {
                if self.model.depth == 0 {
                    assert_eq!(self.scene.cancel_step(), Err(SceneError::StepNotOpen));
                    return;
                }
                self.scene.cancel_step().unwrap();
                self.now = state(&self.scene);
                self.model.depth = 0;
                let (before, _) = self.model.open.take().unwrap();
                assert_eq!(self.now, before);
            }
            Op::Undo => {
                let open = self.model.open.as_ref().is_some_and(|o| !o.1.is_empty());
                let expect = open || !self.model.undo.is_empty();
                if open {
                    // It is undone before it can be evicted.
                    self.model.close(&self.now);
                } else {
                    self.model.open = None;
                    self.model.depth = 0;
                }
                assert_eq!(self.scene.undo(), Ok(expect));
                self.now = state(&self.scene);
                if expect {
                    let s = self.model.undo.pop().unwrap();
                    assert_eq!(self.now, s.before);
                    self.model.redo.push(s);
                }
            }
            Op::Redo => {
                let expect = !self.model.redo.is_empty();
                assert_eq!(self.scene.redo(), Ok(expect));
                self.model.close(&self.now);
                self.now = state(&self.scene);
                if expect {
                    let s = self.model.redo.pop().unwrap();
                    assert_eq!(self.now, s.after);
                    self.model.undo.push(s);
                }
            }
            Op::Clear => {
                self.scene.clear_history().unwrap();
                self.model = Model::default();
            }
            Op::Take { second } => {
                let s = second.then_some(self.second);
                self.mirrors[usize::from(second)].take(&mut self.scene, s);
            }
            Op::Flush => {
                self.scene.flush().unwrap();
                let scene = &self.scene;
                for (id, e) in scene.entries.iter() {
                    assert!(e.is_current(), "{id:?}");
                    e.bvh.validate(&e.mesh).unwrap();
                    if let Some(normals) = scene.normals(id).unwrap() {
                        let fresh = NormalState::new(&e.mesh).unwrap();
                        for v in self.now[&id].vertices.keys() {
                            assert_eq!(normals[*v as usize], fresh.normals()[*v as usize]);
                        }
                    }
                }
                for &(f, on) in &self.fields {
                    if scene.field_status(f).is_ok_and(|s| s.current)
                        && scene
                            .field_targets(f)
                            .unwrap()
                            .iter()
                            .all(|t| self.now.contains_key(t))
                    {
                        check_field(scene, f, on);
                    }
                }
            }
        }
    }

    fn check(&mut self) {
        // Evicted: the oldest steps go first, on both stacks.
        let (undo, redo) = (self.scene.history.undo_len(), self.scene.history.redo_len());
        assert!(undo <= self.model.undo.len() && redo <= self.model.redo.len());
        if self.budget == usize::MAX {
            assert_eq!((undo, redo), (self.model.undo.len(), self.model.redo.len()));
        }
        self.model.undo.drain(..self.model.undo.len() - undo);
        self.model.redo.drain(..self.model.redo.len() - redo);
        if self.model.depth == 0 {
            assert!(self.scene.history.bytes() <= self.budget);
        }
        assert_eq!(self.now, state(&self.scene));
        let open = self.model.open.as_ref().is_some_and(|o| !o.1.is_empty());
        assert_eq!(self.scene.can_undo(), open || undo > 0);
        assert_eq!(self.scene.can_redo(), redo > 0);
        let parked: BTreeSet<MeshId> = self.scene.restorable_ids().collect();
        assert_eq!(parked, self.model.restorable(&self.now));
        for id in &parked {
            assert!(self.scene.entries.get_parked(*id).unwrap().parked.is_some());
        }
        for (id, e) in self.scene.entries.iter() {
            assert!(e.parked.is_none());
            assert_eq!(e.reports.len(), 2);
            let mut hooks: Vec<FieldId> = e.hooks.iter().map(|h| h.field).collect();
            let n = hooks.len();
            hooks.sort_unstable();
            hooks.dedup();
            assert_eq!(hooks.len(), n, "one hook per field on {id:?}");
            // Every hook belongs to a registered field.
            assert!(hooks.iter().all(|f| self.scene.field_status(*f).is_ok()));
        }
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn random_sequences_match_the_model(
        ops in prop::collection::vec(op(), 1..70),
        budget in prop_oneof![
            3 => Just(usize::MAX),
            1 => 200usize..6000,
            3 => 6000usize..200_000,
        ],
    ) {
        let mut scene = Scene::new(SceneOptions { history_budget: budget });
        let second = scene.subscribe();
        let mut run = Run {
            scene,
            second,
            model: Model::default(),
            now: State::new(),
            mirrors: Default::default(),
            ids: BTreeSet::new(),
            fields: Vec::new(),
            budget,
        };
        for op in &ops {
            run.apply(op);
            run.check();
        }
        run.apply(&Op::Flush);
        run.apply(&Op::Take { second: false });
        run.apply(&Op::Take { second: true });
        // All the way back and forth again.
        while run.scene.can_undo() {
            run.apply(&Op::Undo);
            run.check();
        }
        run.apply(&Op::Take { second: false });
        while run.scene.can_redo() {
            run.apply(&Op::Redo);
            run.check();
        }
        run.apply(&Op::Flush);
        run.apply(&Op::Take { second: false });
        run.apply(&Op::Take { second: true });
        run.scene.clear_history().unwrap();
        prop_assert!(run.scene.restorable_ids().next().is_none());
        prop_assert_eq!(run.scene.history.bytes(), 0);
    }
}
