//! Property test for multi-target distance fields: after any sequence of
//! placements, drags, edits, undo, redo, compactions and mode changes of any
//! of four meshes, a field measured to several meshes must hold, bit for
//! bit, the minimum of the single-target fields to each of them (ties to the
//! lowest target index), and its nearest-target indices must name the target
//! that minimum came from.
//!
//! Mesh A is an open sheet; B, C and D are cubes around it (closed or with a
//! face missing), at coordinates around 1000. A has a field to B, C and D,
//! another to the same meshes in another order (D, B, C), and one to each
//! alone; B has a field to C and A (in that order) and one to each alone.
//! Every field shares one sign mode and one `max_distance` (finite or
//! infinite) per case. The walk also removes target meshes (the reference
//! is then the single fields of the targets left), and either updates in
//! one shot or defers the Eager fields and steps them with a random budget.

use fast_mesh_scene::{
    DistanceField, FieldBudget, FieldId, MeshId, MeshOptions, Placement, Scene, SceneError,
    SceneOptions, SignMode, StepState, UpdateMode,
};
use fast_mesh_store::{Mesh, VertexId};
use proptest::prelude::*;

/// The case count: `PROPTEST_CASES` when set, else `default`.
fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

const ORIGIN: [f64; 3] = [1000.0, -1000.5, 999.7];

/// A grid sheet, 8 x 8 squares of 0.1, in the plane `z = 0` of `ORIGIN`.
fn sheet() -> Mesh {
    let n = 8u32;
    let mut pos = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            pos.extend_from_slice(&[
                ORIGIN[0] + f64::from(x) * 0.1,
                ORIGIN[1] + f64::from(y) * 0.1,
                ORIGIN[2],
            ]);
        }
    }
    let mut idx = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            let c = a + n + 1;
            idx.extend_from_slice(&[a, a + 1, c, a + 1, c + 1, c]);
        }
    }
    Mesh::from_buffers_f64(&pos, &idx).unwrap()
}

/// A cube of side `size` with its minimum corner at `ORIGIN + at`,
/// outward oriented; `open` leaves out one face.
fn cube(at: [f64; 3], size: f64, open: bool) -> Mesh {
    let mut pos = Vec::new();
    for i in 0..8u32 {
        let b = |bit: u32| if i & bit != 0 { size } else { 0.0 };
        pos.extend_from_slice(&[
            ORIGIN[0] + at[0] + b(1),
            ORIGIN[1] + at[1] + b(2),
            ORIGIN[2] + at[2] + b(4),
        ]);
    }
    let mut idx = vec![
        0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6,
        1, 3, 5, 3, 7, 5,
    ];
    if open {
        idx.truncate(idx.len() - 6);
    }
    Mesh::from_buffers_f64(&pos, &idx).unwrap()
}

#[derive(Debug, Clone)]
enum Op {
    /// A placement of mesh `.0` with a small turn about z and a small
    /// translation.
    Place(u8, i8, [i8; 3]),
    /// Moves a vertex of mesh `.0` by a small amount.
    Move(u8, u32, [i8; 3]),
    RemoveTriangle(u8, u32),
    /// Splits a triangle at its centroid.
    Split(u8, u32),
    /// Three placements in one step.
    Drag(u8, i8),
    Undo,
    Redo,
    Compact(u8),
    SetMode(u8, u8),
    /// Removes mesh B, C or D (`1 + .0 % 3`).
    RemoveTarget(u8),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        4 => (0u8..4, any::<i8>(), any::<[i8; 3]>()).prop_map(|(m, d, t)| Op::Place(m, d, t)),
        3 => (0u8..4, any::<u32>(), any::<[i8; 3]>()).prop_map(|(m, v, d)| Op::Move(m, v, d)),
        1 => (0u8..4, any::<u32>()).prop_map(|(m, t)| Op::RemoveTriangle(m, t)),
        1 => (0u8..4, any::<u32>()).prop_map(|(m, t)| Op::Split(m, t)),
        2 => (0u8..4, any::<i8>()).prop_map(|(m, d)| Op::Drag(m, d)),
        2 => Just(Op::Undo),
        1 => Just(Op::Redo),
        1 => (0u8..4).prop_map(Op::Compact),
        1 => (0u8..4, 0u8..3).prop_map(|(m, k)| Op::SetMode(m, k)),
        1 => (0u8..3).prop_map(Op::RemoveTarget),
    ]
}

fn update_mode(k: u8) -> UpdateMode {
    [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual][usize::from(k) % 3]
}

/// `i8` as a fraction in `[-1, 1]`.
fn unit(x: i8) -> f64 {
    f64::from(x) / 127.0
}

/// A rotation by `deg` degrees about the vertical axis through `ORIGIN`,
/// then a translation by `t`.
fn turned(deg: f64, t: [f64; 3]) -> Placement {
    let (s, c) = deg.to_radians().sin_cos();
    let r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
    let translation = std::array::from_fn(|i| {
        ORIGIN[i] - (0..3).map(|j| r[i][j] * ORIGIN[j]).sum::<f64>() + t[i]
    });
    Placement {
        rotation: r,
        translation,
    }
}

fn alive_vertices(scene: &Scene, m: MeshId) -> Vec<VertexId> {
    let mesh = scene.mesh(m).unwrap();
    (0..mesh.vertex_slots() as VertexId)
        .filter(|&v| mesh.is_vertex_alive(v))
        .collect()
}

fn alive_triangles(scene: &Scene, m: MeshId) -> Vec<u32> {
    let mesh = scene.mesh(m).unwrap();
    (0..mesh.triangle_slots() as u32)
        .filter(|&t| mesh.is_triangle_alive(t))
        .collect()
}

fn pick<T: Copy>(items: &[T], k: u32) -> Option<T> {
    (!items.is_empty()).then(|| items[k as usize % items.len()])
}

/// Applies one operation; those that do not apply (to a removed mesh, say)
/// are skipped, and no mesh drops below five triangles.
fn apply(scene: &mut Scene, meshes: &[MeshId; 4], op: &Op) {
    let which = |m: &u8| meshes[usize::from(*m) % 4];
    let gone = |m: MeshId| scene.mesh(m).is_err();
    match op {
        Op::Place(m, ..)
        | Op::Move(m, ..)
        | Op::RemoveTriangle(m, _)
        | Op::Split(m, _)
        | Op::Drag(m, _)
        | Op::Compact(m)
        | Op::SetMode(m, _)
            if gone(which(m)) => {}
        Op::RemoveTarget(k) => {
            let m = meshes[1 + usize::from(*k) % 3];
            if !gone(m) {
                scene.remove_mesh(m).unwrap();
            }
        }
        Op::Place(m, deg, t) => {
            let t = t.map(|x| unit(x) * 0.15);
            scene
                .set_placement(which(m), turned(unit(*deg) * 25.0, t))
                .unwrap();
        }
        Op::Move(m, v, d) => {
            let m = which(m);
            if let Some(v) = pick(&alive_vertices(scene, m), *v) {
                let d = d.map(|x| (unit(x) * 0.1) as f32);
                scene
                    .edit(m, |tx| {
                        let p = tx.mesh().position(v).unwrap();
                        tx.set_positions(&[v], &[[p[0] + d[0], p[1] + d[1], p[2] + d[2]]])
                    })
                    .unwrap();
            }
        }
        Op::RemoveTriangle(m, t) => {
            let m = which(m);
            let tris = alive_triangles(scene, m);
            if tris.len() > 4
                && let Some(t) = pick(&tris, *t)
            {
                scene.edit(m, |tx| tx.remove_triangles(&[t])).unwrap();
            }
        }
        Op::Split(m, t) => {
            let m = which(m);
            if let Some(t) = pick(&alive_triangles(scene, m), *t) {
                scene
                    .edit(m, |tx| {
                        let [i, j, k] = tx.mesh().triangle(t).unwrap();
                        let ps = [i, j, k].map(|v| tx.mesh().position(v).unwrap());
                        let c = std::array::from_fn(|d| (ps[0][d] + ps[1][d] + ps[2][d]) / 3.0);
                        let v = tx.add_vertices(&[c])?[0];
                        tx.remove_triangles(&[t])?;
                        tx.add_triangles(&[[i, j, v], [j, k, v], [k, i, v]])
                    })
                    .unwrap();
            }
        }
        Op::Drag(m, d) => {
            let m = which(m);
            scene.begin_step();
            for k in 1..=3 {
                let d = 0.01 * f64::from(k) * (1.0 + unit(*d));
                scene
                    .set_placement(m, turned(d * 50.0, [d, -d, d]))
                    .unwrap();
            }
            scene.end_step().unwrap();
        }
        Op::Undo => {
            scene.undo().unwrap();
        }
        Op::Redo => {
            scene.redo().unwrap();
        }
        Op::Compact(m) => {
            scene.compact(which(m)).unwrap();
        }
        Op::SetMode(m, k) => {
            scene.set_mode(which(m), update_mode(*k)).unwrap();
        }
    }
}

fn update_all(scene: &mut Scene) {
    for _ in 0..8 {
        match scene.update_fields() {
            Ok(()) => return,
            Err(SceneError::StaleBvh(id)) => scene.update(id).unwrap(),
            Err(e) => panic!("update_fields failed: {e}"),
        }
    }
    panic!("update_fields kept failing");
}

/// Steps the deferred Eager fields with `budget` until none is due, then
/// brings the others along, as `update_all`.
fn step_all(scene: &mut Scene, budget: usize) {
    for _ in 0..100_000 {
        match scene.field_step(FieldBudget::Points(budget)) {
            Ok(StepState::Idle) => return update_all(scene),
            Ok(_) => {}
            Err(SceneError::StaleBvh(id)) => scene.update(id).unwrap(),
            Err(e) => panic!("field_step failed: {e}"),
        }
    }
    panic!("field_step never went idle");
}

/// Checks the multi-target field `multi` against the single-target fields
/// `singles` (each with its target), taken in the field's current target
/// order. A field whose `on` mesh, or every target, was removed is gone,
/// and so are its singles: nothing to check. Returns whether a value is
/// finite.
fn check(
    scene: &Scene,
    multi: FieldId,
    singles: &[(MeshId, FieldId)],
    at: &str,
) -> Result<bool, TestCaseError> {
    let Ok(targets) = scene.field_targets(multi) else {
        return Ok(false);
    };
    let got = scene.field_values(multi).unwrap();
    let nearest = scene.field_nearest(multi).unwrap();
    let parts: Vec<&[f32]> = targets
        .iter()
        .map(|t| {
            let (_, f) = singles.iter().find(|(m, _)| m == t).unwrap();
            scene.field_values(*f).unwrap()
        })
        .collect();
    for (v, &g) in got.iter().enumerate() {
        let (mut want, mut from) = (parts[0][v], u32::MAX);
        if !want.is_nan() {
            (want, from) = (f32::INFINITY, u32::MAX);
            for (k, p) in parts.iter().enumerate() {
                if p[v] < want {
                    (want, from) = (p[v], k as u32);
                }
            }
        }
        prop_assert_eq!(g.to_bits(), want.to_bits(), "{} vertex {}", at, v);
        let n = nearest.get(v).copied().unwrap_or(u32::MAX);
        prop_assert_eq!(n, from, "{} vertex {} nearest", at, v);
    }
    Ok(got.iter().any(|v| v.is_finite()))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases(48)))]

    #[test]
    fn multi_target_is_min_over_single_targets(
        open in prop::array::uniform3(any::<bool>()),
        modes in prop::array::uniform4(0u8..3),
        winding in any::<bool>(),
        infinite in any::<bool>(),
        stepped in any::<bool>(),
        budget in 1usize..200,
        batches in prop::collection::vec(prop::collection::vec(op(), 1..4), 1..12),
    ) {
        let mut scene = Scene::new(SceneOptions::default());
        let options = |k: u8| MeshOptions {
            mode: update_mode(k),
            ..MeshOptions::default()
        };
        let meshes = [
            scene.add_mesh(sheet(), options(modes[0])).unwrap(),
            scene.add_mesh(cube([0.1, 0.1, -0.2], 0.3, open[0]), options(modes[1])).unwrap(),
            scene.add_mesh(cube([0.3, 0.2, -0.1], 0.3, open[1]), options(modes[2])).unwrap(),
            scene.add_mesh(cube([0.55, 0.5, -0.05], 0.2, open[2]), options(modes[3])).unwrap(),
        ];
        for &m in &meshes {
            scene.update(m).unwrap();
        }
        let [a, b, c, d] = meshes;
        let sign = if winding { SignMode::Winding } else { SignMode::Pseudonormal };
        let md = if infinite { f64::INFINITY } else { 0.12 };
        let mut add = |name: &str, on: MeshId, to: Vec<MeshId>| {
            scene
                .add_distance_field(DistanceField {
                    name: name.to_owned(),
                    on,
                    to,
                    max_distance: md,
                    sign,
                })
                .unwrap()
        };
        let on_a = vec![(b, add("b", a, vec![b])), (c, add("c", a, vec![c])), (d, add("d", a, vec![d]))];
        let groups = [
            (add("m", a, vec![b, c, d]), on_a.clone()),
            // The same targets in another order: the same values, the indices
            // permuted.
            (add("p", a, vec![d, b, c]), on_a),
            (add("m", b, vec![c, a]), vec![(c, add("c", b, vec![c])), (a, add("a", b, vec![a]))]),
        ];
        update_all(&mut scene);
        let mut finite = false;
        for (multi, singles) in &groups {
            finite |= check(&scene, *multi, singles, "initial")?;
        }
        prop_assert!(finite);
        scene.set_defer_fields(stepped);
        for (i, batch) in batches.iter().enumerate() {
            for o in batch {
                apply(&mut scene, &meshes, o);
            }
            if stepped {
                step_all(&mut scene, budget);
            } else {
                update_all(&mut scene);
            }
            for (multi, singles) in &groups {
                check(&scene, *multi, singles, &format!("batch {i}"))?;
            }
        }
    }
}
