//! The history budget: changing it, what it evicts, and the report of steps
//! too large for it (#281). Entries of the application in the history (#282).

use fast_mesh_ops::boundary_loops;
use fast_mesh_store::Mesh;

use crate::EntryEvent::{Dropped, Redone, Undone};
use crate::history::ENTRY_BYTES;
use crate::scene::fault;
use crate::{
    DistanceField, HistoryEntry, HistorySide, JoinOptions, MeshId, MeshOptions, PermanentStep,
    Placement, Scene, SceneError, SceneOptions, SignMode, UpdateMode,
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
    Mesh::from_buffers(&pos, &idx).unwrap()
}

fn scene(budget: usize) -> Scene {
    Scene::new(SceneOptions {
        history_budget: budget,
    })
}

fn undoable() -> MeshOptions {
    MeshOptions {
        undoable: true,
        ..MeshOptions::default()
    }
}

fn moved(x: f64) -> Placement {
    Placement {
        translation: [x, 0.0, 0.0],
        ..Placement::IDENTITY
    }
}

/// Raises vertices `0..n` to `z`: an edit whose patch grows with `n`.
fn lift(scene: &mut Scene, id: MeshId, n: u32, z: f32) {
    let ids: Vec<u32> = (0..n).collect();
    scene
        .edit(id, |tx| {
            let to: Vec<[f32; 3]> = ids
                .iter()
                .map(|&v| {
                    let p = tx.mesh().position(v).unwrap();
                    [p[0], p[1], z]
                })
                .collect();
            tx.set_positions(&ids, &to)
        })
        .unwrap();
}

fn z(scene: &Scene, id: MeshId) -> f32 {
    scene.mesh(id).unwrap().position(0).unwrap()[2]
}

/// A scene with one mesh and `steps` lifts of it, to 1, 2, 3, ...
fn lifted(budget: usize, steps: u32) -> (Scene, MeshId) {
    let mut scene = scene(budget);
    let id = scene.add_mesh(grid(8), MeshOptions::default()).unwrap();
    for k in 1..=steps {
        lift(&mut scene, id, 81, k as f32);
    }
    (scene, id)
}

#[test]
fn the_default_budget_is_256_mib() {
    assert_eq!(
        Scene::new(SceneOptions::default()).history_budget(),
        256 << 20
    );
    assert_eq!(scene(1000).history_budget(), 1000);
}

#[test]
fn a_smaller_budget_evicts_the_oldest_undo_steps() {
    let (scene, _) = lifted(usize::MAX, 6);
    let step = scene.history_bytes() as usize / 6;
    assert!(step > 0 && scene.history.undo_len() == 6);

    for keep in [5, 2, 0] {
        let (mut scene, id) = lifted(usize::MAX, 6);
        scene
            .set_history_budget(keep * step + step / 2 * usize::from(keep > 0))
            .unwrap();
        assert_eq!(scene.history.undo_len(), keep);
        assert!(scene.history_bytes() <= scene.history_budget() as u64);
        assert_eq!(z(&scene, id), 6.0, "the mesh is not changed");
        // The newest are the ones left.
        for k in (6 - keep..6).rev() {
            assert_eq!(scene.undo(), Ok(true));
            assert_eq!(z(&scene, id), k as f32);
        }
        assert_eq!(scene.undo(), Ok(false));
        for k in 6 - keep + 1..=6 {
            assert_eq!(scene.redo(), Ok(true));
            assert_eq!(z(&scene, id), k as f32);
        }
        assert_eq!(scene.redo(), Ok(false));
    }

    // Changing it has the effect it has from the start.
    let (mut late, id) = lifted(usize::MAX, 3);
    late.set_history_budget(2 * step).unwrap();
    lift(&mut late, id, 81, 4.0);
    let (early, _) = lifted(2 * step, 4);
    assert_eq!(late.history.undo_len(), early.history.undo_len());
    assert_eq!(late.history_bytes(), early.history_bytes());
    // A larger one keeps more from then on, and brings nothing back.
    late.set_history_budget(usize::MAX).unwrap();
    assert_eq!(late.history.undo_len(), 2);
    lift(&mut late, id, 81, 5.0);
    assert_eq!(late.history.undo_len(), 3);
    late.clear_history().unwrap();
    assert_eq!(late.history_bytes(), 0);
    assert_eq!(late.history_budget(), usize::MAX);
}

#[test]
fn a_smaller_budget_then_evicts_the_furthest_redo_steps() {
    let (mut scene, id) = lifted(usize::MAX, 6);
    let step = scene.history_bytes() as usize / 6;
    for _ in 0..4 {
        scene.undo().unwrap();
    }
    // Two to undo, four to redo: room for three.
    scene.set_history_budget(3 * step + step / 2).unwrap();
    assert_eq!(scene.history.undo_len(), 0);
    assert_eq!(scene.history.redo_len(), 3);
    assert_eq!(z(&scene, id), 2.0);
    for k in 3..=5 {
        assert_eq!(scene.redo(), Ok(true));
        assert_eq!(z(&scene, id), k as f32);
    }
    assert_eq!(scene.redo(), Ok(false));
    scene.set_history_budget(0).unwrap();
    assert!(!scene.can_undo() && scene.history_bytes() == 0);
}

#[test]
fn a_smaller_budget_keeps_the_open_step_until_it_closes() {
    let (mut scene, id) = lifted(usize::MAX, 2);
    scene.begin_step();
    lift(&mut scene, id, 81, 3.0);
    scene.set_history_budget(1).unwrap();
    assert_eq!(scene.history.undo_len(), 0);
    assert!(scene.can_undo(), "the open step");
    assert!(scene.take_permanent().is_empty());
    scene.cancel_step().unwrap();
    assert_eq!(z(&scene, id), 2.0);
    assert!(!scene.can_undo());

    scene.begin_step();
    lift(&mut scene, id, 81, 4.0);
    assert_eq!(scene.undo(), Ok(true));
    assert_eq!(z(&scene, id), 2.0);
    assert!(!scene.can_redo());

    scene.begin_step();
    lift(&mut scene, id, 81, 5.0);
    scene.end_step().unwrap();
    assert!(!scene.can_undo());
    assert_eq!(z(&scene, id), 5.0);
}

#[test]
fn a_smaller_budget_frees_the_meshes_of_dropped_steps() {
    let mut scene = scene(usize::MAX);
    let ids: Vec<MeshId> = (0..3)
        .map(|_| scene.add_mesh(grid(16), MeshOptions::default()).unwrap())
        .collect();
    for &id in &ids {
        scene.remove_mesh_undoable(id).unwrap();
    }
    let one = scene.history_bytes() as usize / 3;
    scene.set_history_budget(2 * one + one / 2).unwrap();
    assert_eq!(scene.restorable_ids().collect::<Vec<_>>(), ids[1..]);
    assert_eq!(scene.undo(), Ok(true));
    assert_eq!(scene.undo(), Ok(true));
    assert_eq!(scene.undo(), Ok(false));
    assert!(scene.mesh(ids[0]).is_err());
    assert_eq!(scene.mesh(ids[1]).unwrap().triangle_count(), 512);
    // Undone removals hold nothing; an undone add does, until it goes.
    let n = scene.add_mesh(grid(16), undoable()).unwrap();
    scene.undo().unwrap();
    assert!(scene.is_restorable(n));
    scene.set_history_budget(0).unwrap();
    assert!(!scene.is_restorable(n) && !scene.can_redo());
    assert!(scene.take_permanent().is_empty(), "the caller asked for it");
}

#[test]
fn removal_bytes_is_what_the_removal_then_holds() {
    let mut scene = scene(usize::MAX);
    let a = scene.add_mesh(grid(20), MeshOptions::default()).unwrap();
    let b = scene
        .add_mesh(
            grid(6),
            MeshOptions {
                placement: moved(1.0),
                ..MeshOptions::default()
            },
        )
        .unwrap();
    scene.set_normals(a, true).unwrap();
    scene
        .add_distance_field(DistanceField::new(
            "d",
            a,
            b,
            f64::INFINITY,
            SignMode::Pseudonormal,
        ))
        .unwrap();
    for id in [a, b] {
        let want = scene.removal_bytes(id).unwrap();
        let before = scene.history_bytes();
        scene.remove_mesh_undoable(id).unwrap();
        let held = scene.history_bytes() - before;
        assert!(held.abs_diff(want) < 512, "{held} bytes, {want} expected");
        assert_eq!(scene.removal_bytes(id), Err(SceneError::UnknownMesh));
    }
    // It tells what is permanent before it happens.
    scene.undo().unwrap();
    scene.undo().unwrap();
    scene.clear_history().unwrap();
    let bytes = scene.removal_bytes(a).unwrap() as usize;
    for (budget, kept) in [(2 * bytes, true), (bytes / 2, false)] {
        scene.set_history_budget(budget).unwrap();
        scene.remove_mesh_undoable(a).unwrap();
        assert_eq!(scene.is_restorable(a), kept);
        assert_eq!(scene.take_permanent().len(), usize::from(!kept));
        assert_eq!(scene.undo(), Ok(kept));
    }
    assert_eq!(scene.mesh(a).err(), Some(SceneError::UnknownMesh));
}

#[test]
fn steps_too_large_to_keep_are_reported() {
    let mut scene = scene(4096);
    assert!(!scene.has_permanent());
    let a = scene.add_mesh(grid(20), MeshOptions::default()).unwrap();
    let b = scene.add_mesh(grid(20), MeshOptions::default()).unwrap();

    // Older steps that make room are not reported.
    for k in 0..200 {
        scene.set_placement(a, moved(f64::from(k) + 1.0)).unwrap();
    }
    assert!(scene.history.undo_len() < 200);
    assert!(scene.take_permanent().is_empty());

    // An edit of two meshes in one step.
    scene.begin_step();
    scene.set_placement(b, moved(1.0)).unwrap();
    lift(&mut scene, a, 441, 1.0);
    scene.set_placement(b, moved(2.0)).unwrap();
    assert!(scene.take_permanent().is_empty(), "still open");
    scene.end_step().unwrap();
    assert!(scene.has_permanent());
    let [step] = scene.take_permanent().try_into().unwrap();
    let PermanentStep {
        lost,
        meshes,
        gone,
        bytes,
    } = step;
    assert_eq!(
        (lost, meshes, gone),
        (HistorySide::Undo, vec![b, a], vec![])
    );
    assert!(bytes > 4096);
    assert!(!scene.has_permanent() && scene.take_permanent().is_empty());

    // A removal.
    scene.remove_mesh_undoable(b).unwrap();
    assert!(!scene.is_restorable(b));
    let [step] = scene.take_permanent().try_into().unwrap();
    assert_eq!((step.lost, step.gone), (HistorySide::Undo, vec![b]));

    // An add that cannot be redone once undone; the older step stays.
    scene.set_placement(a, moved(0.5)).unwrap();
    let n = scene.add_mesh(grid(20), undoable()).unwrap();
    assert!(scene.take_permanent().is_empty());
    scene.undo().unwrap();
    let [step] = scene.take_permanent().try_into().unwrap();
    assert_eq!(step.lost, HistorySide::Redo);
    assert_eq!((step.meshes, step.gone), (vec![n], vec![n]));
    assert!(scene.can_undo() && !scene.can_redo());

    // Undone in an open step.
    scene.begin_step();
    lift(&mut scene, a, 441, 3.0);
    scene.undo().unwrap();
    let [step] = scene.take_permanent().try_into().unwrap();
    assert_eq!(step.lost, HistorySide::Redo);
    assert_eq!((step.meshes, step.gone), (vec![a], vec![]));

    // A cancelled step, and clearing, are the caller's doing.
    scene.begin_step();
    lift(&mut scene, a, 441, 4.0);
    scene.cancel_step().unwrap();
    scene.begin_step();
    lift(&mut scene, a, 441, 5.0);
    scene.clear_history().unwrap();
    assert!(scene.take_permanent().is_empty());
}

#[test]
fn a_join_larger_than_the_budget_is_reported() {
    // Two quads facing away from each other.
    let quad = |z: f32, idx: [u32; 6]| {
        let pos = [0., 0., z, 1., 0., z, 1., 1., z, 0., 1., z];
        Mesh::from_buffers(&pos, &idx).unwrap()
    };
    let run = |budget: usize| {
        let mut scene = scene(budget);
        let (qa, qb) = (quad(0.0, [0, 2, 1, 0, 3, 2]), quad(1.0, [0, 1, 2, 0, 2, 3]));
        let options = JoinOptions {
            loop_a: boundary_loops(&qa).unwrap()[0].id,
            loop_b: boundary_loops(&qb).unwrap()[0].id,
            max_edges: None,
        };
        let a = scene.add_mesh(qa, MeshOptions::default()).unwrap();
        let b = scene.add_mesh(qb, MeshOptions::default()).unwrap();
        scene.join(a, b, &options).unwrap();
        (scene, a, b)
    };
    let (mut kept, _, b) = run(usize::MAX);
    assert!(kept.is_restorable(b) && kept.take_permanent().is_empty());
    let (mut lost, a, b) = run(64);
    assert!(!lost.is_restorable(b) && !lost.can_undo());
    let [step] = lost.take_permanent().try_into().unwrap();
    assert_eq!(step.lost, HistorySide::Undo);
    assert_eq!((step.meshes, step.gone), (vec![a, b], vec![b]));
}

#[test]
fn a_redone_removal_that_no_longer_fits_is_reported() {
    let mut scene = scene(usize::MAX);
    let m = scene.add_mesh(grid(20), MeshOptions::default()).unwrap();
    scene.remove_mesh_undoable(m).unwrap();
    let bytes = scene.history_bytes() as usize;
    scene.undo().unwrap();
    scene.set_history_budget(bytes / 2).unwrap();
    assert_eq!(scene.redo(), Ok(true));
    assert!(!scene.is_restorable(m));
    let [step] = scene.take_permanent().try_into().unwrap();
    assert_eq!((step.lost, step.gone), (HistorySide::Undo, vec![m]));
}

#[test]
fn only_the_latest_reports_are_kept() {
    let mut scene = scene(0);
    let ids: Vec<MeshId> = (0..300)
        .map(|_| {
            let options = MeshOptions {
                mode: UpdateMode::Manual,
                defer_bvh: true,
                ..MeshOptions::default()
            };
            scene.add_mesh(grid(1), options).unwrap()
        })
        .collect();
    for &id in &ids {
        scene.set_placement(id, moved(1.0)).unwrap();
    }
    let steps = scene.take_permanent();
    assert_eq!(steps.len(), 256);
    let got: Vec<MeshId> = steps.iter().map(|s| s.meshes[0]).collect();
    assert_eq!(got, ids[44..]);
}

/// Records an entry without bytes or a key.
fn entry(scene: &mut Scene, id: u64) {
    scene
        .record_entry(HistoryEntry {
            id,
            ..HistoryEntry::default()
        })
        .unwrap();
}

/// Records an entry that merges under `key`.
fn merged(scene: &mut Scene, id: u64, key: u64) {
    scene
        .record_entry(HistoryEntry {
            id,
            merge: Some(key),
            ..HistoryEntry::default()
        })
        .unwrap();
}

fn one_mesh(budget: usize) -> (Scene, MeshId) {
    let mut scene = scene(budget);
    let id = scene.add_mesh(grid(2), MeshOptions::default()).unwrap();
    (scene, id)
}

#[test]
fn entries_interleave_with_mesh_changes() {
    let (mut scene, id) = one_mesh(usize::MAX);
    assert!(scene.take_entry_events().is_empty());
    entry(&mut scene, 1);
    lift(&mut scene, id, 1, 1.0);
    entry(&mut scene, 2);
    scene.set_placement(id, moved(1.0)).unwrap();
    scene.remove_mesh_undoable(id).unwrap();
    entry(&mut scene, 3);
    assert_eq!(scene.history.undo_len(), 6);
    assert!(scene.take_entry_events().is_empty());

    let there = |s: &Scene| s.mesh(id).is_ok();
    let x = |s: &Scene| s.placement(id).unwrap().translation[0];
    let undo = |s: &mut Scene| {
        assert_eq!(s.undo(), Ok(true));
        s.take_entry_events()
    };
    for _ in 0..2 {
        assert_eq!(undo(&mut scene), [Undone(3)]);
        assert!(!there(&scene));
        assert_eq!(undo(&mut scene), []);
        assert!(there(&scene) && x(&scene) == 1.0);
        assert_eq!(undo(&mut scene), []);
        assert_eq!(x(&scene), 0.0);
        assert_eq!(undo(&mut scene), [Undone(2)]);
        assert_eq!(z(&scene, id), 1.0);
        assert_eq!(undo(&mut scene), []);
        assert_eq!(z(&scene, id), 0.0);
        assert_eq!(undo(&mut scene), [Undone(1)]);
        assert!(!scene.can_undo());
        assert_eq!(scene.undo(), Ok(false));
        assert_eq!(scene.next_entry_event(), None);

        let mut seen = Vec::new();
        while scene.redo().unwrap() {
            seen.extend(scene.take_entry_events());
        }
        assert_eq!(seen, [Redone(1), Redone(2), Redone(3)]);
        assert!(!there(&scene));
    }
    // The mesh reports nothing for entries.
    scene.take_changes();
    entry(&mut scene, 4);
    scene.undo().unwrap();
    assert!(scene.take_changes().is_empty());
}

#[test]
fn entries_join_the_open_step_and_cancel_with_it() {
    let (mut scene, id) = one_mesh(usize::MAX);
    scene.begin_step();
    entry(&mut scene, 1);
    lift(&mut scene, id, 1, 1.0);
    scene.begin_step();
    entry(&mut scene, 2);
    scene.end_step().unwrap();
    scene.end_step().unwrap();
    assert_eq!(scene.history.undo_len(), 1);
    scene.undo().unwrap();
    assert_eq!(scene.take_entry_events(), [Undone(2), Undone(1)]);
    assert_eq!(z(&scene, id), 0.0);
    scene.redo().unwrap();
    assert_eq!(scene.take_entry_events(), [Redone(1), Redone(2)]);
    assert_eq!(z(&scene, id), 1.0);

    // Cancelled: undone, and gone.
    scene.begin_step();
    entry(&mut scene, 3);
    lift(&mut scene, id, 1, 2.0);
    entry(&mut scene, 4);
    scene.cancel_step().unwrap();
    let events = [Undone(4), Undone(3), Dropped(3), Dropped(4)];
    for e in events {
        assert_eq!(scene.next_entry_event(), Some(e));
    }
    assert_eq!(scene.next_entry_event(), None);
    assert_eq!(z(&scene, id), 1.0);
    assert!(!scene.can_redo());

    // An open step with only an entry in it is undone like any other.
    scene.begin_step();
    entry(&mut scene, 5);
    assert!(scene.can_undo());
    assert_eq!(scene.undo(), Ok(true));
    assert_eq!(scene.take_entry_events(), [Undone(5)]);
    assert_eq!(scene.end_step(), Err(SceneError::StepNotOpen));
    assert_eq!(scene.redo(), Ok(true));
    assert_eq!(scene.take_entry_events(), [Redone(5)]);
}

#[test]
fn entries_of_one_key_merge_in_an_open_step() {
    let (mut scene, id) = one_mesh(usize::MAX);
    scene.begin_step();
    merged(&mut scene, 1, 7);
    assert_eq!(scene.next_entry_event(), None);
    merged(&mut scene, 2, 7);
    assert!(scene.take_entry_events().is_empty());
    // The one in between is not needed any more.
    merged(&mut scene, 3, 7);
    assert_eq!(scene.take_entry_events(), [Dropped(2)]);
    merged(&mut scene, 4, 8);
    entry(&mut scene, 5);
    entry(&mut scene, 6);
    merged(&mut scene, 7, 7);
    lift(&mut scene, id, 1, 1.0);
    merged(&mut scene, 8, 7);
    scene.end_step().unwrap();
    assert_eq!(scene.next_entry_event(), None);
    // Seven entries are held (1 and 3 as one change), and an edit.
    assert_eq!(scene.history_bytes(), 7 * ENTRY_BYTES + edit_bytes());
    for _ in 0..2 {
        scene.undo().unwrap();
        let events = [
            Undone(8),
            Undone(7),
            Undone(6),
            Undone(5),
            Undone(4),
            Undone(1),
        ];
        assert_eq!(scene.take_entry_events(), events);
        scene.redo().unwrap();
        let events = [
            Redone(3),
            Redone(4),
            Redone(5),
            Redone(6),
            Redone(7),
            Redone(8),
        ];
        assert_eq!(scene.take_entry_events(), events);
    }
    // Steps of their own do not merge, nor do steps with each other.
    merged(&mut scene, 9, 7);
    merged(&mut scene, 10, 7);
    scene.begin_step();
    merged(&mut scene, 11, 7);
    scene.end_step().unwrap();
    assert_eq!(scene.history.undo_len(), 4);
    for k in [11, 10, 9] {
        scene.undo().unwrap();
        assert_eq!(scene.take_entry_events(), [Undone(k)]);
    }
    // Dropped together, each once.
    scene.clear_history().unwrap();
    let mut dropped: Vec<u64> = scene
        .take_entry_events()
        .into_iter()
        .map(|e| match e {
            Dropped(id) => id,
            e => panic!("{e:?}"),
        })
        .collect();
    dropped.sort_unstable();
    assert_eq!(dropped, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    assert_eq!(scene.history_bytes(), 0);

    // Cancelled while merged.
    scene.begin_step();
    merged(&mut scene, 12, 7);
    merged(&mut scene, 13, 7);
    scene.cancel_step().unwrap();
    let events = [Undone(12), Dropped(12), Dropped(13)];
    assert_eq!(scene.take_entry_events(), events);
}

/// The bytes of one `lift` of one vertex of a 2 x 2 grid.
fn edit_bytes() -> u64 {
    let (mut scene, id) = one_mesh(usize::MAX);
    lift(&mut scene, id, 1, 1.0);
    scene.history_bytes()
}

#[test]
fn a_placement_run_is_broken_by_an_entry() {
    let (mut scene, id) = one_mesh(usize::MAX);
    scene.begin_step();
    scene.set_placement(id, moved(1.0)).unwrap();
    scene.set_placement(id, moved(2.0)).unwrap();
    entry(&mut scene, 1);
    scene.set_placement(id, moved(3.0)).unwrap();
    scene.end_step().unwrap();
    let placements = 2 * crate::history::PLACEMENT_BYTES as u64;
    assert_eq!(scene.history_bytes(), placements + ENTRY_BYTES);
}

#[test]
fn entries_count_against_the_budget_and_are_dropped_by_it() {
    let sized = |id: u64, bytes: u64| HistoryEntry {
        id,
        bytes,
        merge: None,
    };
    const E: u64 = ENTRY_BYTES;
    let budget = 10 * E + edit_bytes();
    let (mut scene, id) = one_mesh(budget as usize);
    scene.record_entry(sized(1, 2 * E)).unwrap();
    assert_eq!(scene.history_bytes(), 3 * E);
    scene.record_entry(sized(2, 3 * E)).unwrap();
    lift(&mut scene, id, 1, 1.0);
    assert_eq!(scene.next_entry_event(), None);
    // The oldest goes to make room.
    scene.record_entry(sized(3, 4 * E)).unwrap();
    assert_eq!(scene.take_entry_events(), [Dropped(1)]);
    assert_eq!(scene.history_bytes(), 9 * E + edit_bytes());
    // Undone, it counts the same: it is kept for the redo.
    let held = scene.history_bytes();
    scene.undo().unwrap();
    assert_eq!(scene.take_entry_events(), [Undone(3)]);
    assert_eq!(scene.history_bytes(), held);
    // A new step drops the redo branch.
    scene.record_entry(sized(4, 0)).unwrap();
    assert_eq!(scene.take_entry_events(), [Dropped(3)]);
    assert!(!scene.can_redo());
    lift(&mut scene, id, 1, 2.0);
    scene.undo().unwrap();
    scene.undo().unwrap();
    assert_eq!(scene.take_entry_events(), [Undone(4)]);
    lift(&mut scene, id, 1, 3.0);
    assert_eq!(scene.take_entry_events(), [Dropped(4)]);

    // Larger than the whole budget: dropped at once, and reported.
    scene.record_entry(sized(5, budget)).unwrap();
    assert_eq!(scene.take_entry_events(), [Dropped(2), Dropped(5)]);
    assert!(!scene.can_undo());
    let [step] = scene.take_permanent().try_into().unwrap();
    assert_eq!((step.lost, step.meshes.len()), (HistorySide::Undo, 0));
    assert_eq!(step.bytes, budget + E);

    // Undone in an open step: told, then dropped.
    scene.begin_step();
    scene.record_entry(sized(6, 2 * budget)).unwrap();
    scene.undo().unwrap();
    assert_eq!(scene.take_entry_events(), [Undone(6), Dropped(6)]);

    // A smaller budget drops them too.
    for k in 7..10 {
        scene.record_entry(sized(k, E)).unwrap();
    }
    scene.undo().unwrap();
    scene.take_entry_events();
    scene.set_history_budget(4 * E as usize).unwrap();
    assert_eq!(scene.take_entry_events(), [Dropped(7)]);
    scene.set_history_budget(0).unwrap();
    assert_eq!(scene.take_entry_events(), [Dropped(8), Dropped(9)]);
    assert_eq!(scene.history_bytes(), 0);

    // Absurd sizes do not overflow, in one open step either.
    scene.set_history_budget(usize::MAX).unwrap();
    scene.begin_step();
    for k in 10..200 {
        scene.record_entry(sized(k, u64::MAX)).unwrap();
    }
    scene.cancel_step().unwrap();
    assert_eq!(scene.history_bytes(), 0);
    assert_eq!(scene.take_entry_events().len(), 2 * 190);
}

#[test]
fn entries_stay_when_meshes_are_removed_for_good_or_compacted() {
    let (mut scene, id) = one_mesh(usize::MAX);
    scene.begin_step();
    lift(&mut scene, id, 1, 1.0);
    entry(&mut scene, 1);
    scene.end_step().unwrap();
    lift(&mut scene, id, 1, 2.0);
    scene.compact(id).unwrap();
    assert_eq!(scene.history.undo_len(), 1);
    scene.remove_mesh(id).unwrap();
    assert_eq!(scene.history.undo_len(), 1);
    assert_eq!(scene.history_bytes(), ENTRY_BYTES);
    assert_eq!(scene.undo(), Ok(true));
    assert_eq!(scene.take_entry_events(), [Undone(1)]);
    assert_eq!(scene.redo(), Ok(true));
    assert_eq!(scene.take_entry_events(), [Redone(1)]);
}

#[test]
fn a_step_that_fails_has_no_events() {
    let (mut scene, id) = one_mesh(usize::MAX);
    fault::fail_reserve_on(1);
    let entry_0 = HistoryEntry::default();
    assert!(scene.record_entry(entry_0).is_err());
    assert!(!scene.can_undo() && scene.next_entry_event().is_none());

    scene.begin_step();
    lift(&mut scene, id, 1, 1.0);
    entry(&mut scene, 1);
    lift(&mut scene, id, 1, 2.0);
    entry(&mut scene, 2);
    // The first edit's revert fails, and everything recovers.
    fault::fail_patch_on(2);
    assert!(matches!(scene.cancel_step(), Err(SceneError::Store(_))));
    assert_eq!(scene.next_entry_event(), None);
    scene.end_step().unwrap();
    fault::fail_patch_on(2);
    assert!(matches!(scene.undo(), Err(SceneError::Store(_))));
    assert_eq!(scene.next_entry_event(), None);
    assert_eq!(z(&scene, id), 2.0);
    scene.undo().unwrap();
    assert_eq!(scene.take_entry_events(), [Undone(2), Undone(1)]);
    fault::fail_patch_on(2);
    assert!(matches!(scene.redo(), Err(SceneError::Store(_))));
    assert_eq!(scene.next_entry_event(), None);
    scene.redo().unwrap();
    assert_eq!(scene.take_entry_events(), [Redone(1), Redone(2)]);

    // The recovery fails too: the step is dropped, its entries with it.
    fault::fail_patch_on(2);
    fault::fail_patch_on(3);
    assert_eq!(scene.undo(), Err(SceneError::Corrupted));
    let mut events = scene.take_entry_events();
    events.sort_by_key(|e| match e {
        Undone(id) | Redone(id) | Dropped(id) => *id,
    });
    assert_eq!(events, [Dropped(1), Dropped(2)]);
    assert!(!scene.can_undo());
}
