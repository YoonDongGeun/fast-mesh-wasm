//! Property test for vertex normals: after any sequence of vertex moves,
//! triangle additions, removals and rewrites, vertex removals, undo, redo,
//! cancelled drags, placement changes, mode switches, normal toggles and
//! compactions, the normals must equal `fast_mesh::compute_vertex_normals`
//! of the live mesh (alive triangles only) within `1e-5` once the meshes are
//! brought up to date (`flush`, then `update` for Manual meshes). Normals
//! that changed must be reported in `normals_dirty`.
//!
//! The mesh is a closed UV sphere built from `f64` coordinates around 1000
//! with an explicit origin, so that local coordinates are small offsets.
//!
//! Seeds: `PROPTEST_RNG_SEED` is honoured; without it the seed is fixed, so
//! that CI runs are deterministic.

use fast_mesh_scene::{MeshId, MeshOptions, Placement, Scene, SceneOptions, UpdateMode};
use fast_mesh_store::{Mesh, TriangleId, VertexId};
use proptest::prelude::*;
use proptest::test_runner::RngSeed;

/// The seed used when `PROPTEST_RNG_SEED` is not set.
const DEFAULT_SEED: u64 = 0x4e0a_3a15;

/// `PROPTEST_CASES` (else `cases`) and `PROPTEST_RNG_SEED` (else a fixed
/// seed). `with_cases` alone would ignore the environment variables.
fn config(cases: u32) -> ProptestConfig {
    let base = ProptestConfig::default();
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(cases);
    let rng_seed = match base.rng_seed {
        RngSeed::Random => RngSeed::Fixed(DEFAULT_SEED),
        fixed => fixed,
    };
    ProptestConfig {
        cases,
        rng_seed,
        ..base
    }
}

/// Normal tolerance.
const TOL: f32 = 1e-5;

const ORIGIN: [f64; 3] = [1000.0, -1000.0, 1000.0];
const SPHERE_CENTRE: [f64; 3] = [1000.2, -999.8, 1000.1];

/// A closed UV sphere of radius 0.5, outward wound, stored relative to
/// `ORIGIN`.
fn sphere() -> Mesh {
    let (stacks, slices, r) = (8u32, 12u32, 0.5);
    let c = SPHERE_CENTRE;
    let mut pos = vec![c[0], c[1], c[2] + r];
    for i in 1..stacks {
        let theta = std::f64::consts::PI * f64::from(i) / f64::from(stacks);
        for j in 0..slices {
            let phi = std::f64::consts::TAU * f64::from(j) / f64::from(slices);
            let (rs, rc) = (r * theta.sin(), r * theta.cos());
            pos.extend([c[0] + rs * phi.cos(), c[1] + rs * phi.sin(), c[2] + rc]);
        }
    }
    pos.extend([c[0], c[1], c[2] - r]);
    let south = 1 + (stacks - 1) * slices;
    let ring = |i: u32, j: u32| 1 + (i - 1) * slices + j % slices;
    let mut idx = Vec::new();
    for j in 0..slices {
        idx.extend([0, ring(1, j), ring(1, j + 1)]);
        for i in 1..stacks - 1 {
            let (a, b, c, d) = (
                ring(i, j),
                ring(i + 1, j),
                ring(i + 1, j + 1),
                ring(i, j + 1),
            );
            idx.extend([a, b, c, a, c, d]);
        }
        idx.extend([ring(stacks - 1, j), south, ring(stacks - 1, j + 1)]);
    }
    Mesh::from_buffers_f64_with_origin(&pos, &idx, ORIGIN).unwrap()
}

#[derive(Debug, Clone)]
enum Op {
    Move(u32, [i8; 3]),
    RemoveTriangle(u32),
    Split(u32),
    Rewrite(u32, u32),
    AddTriangle(u32, u32, [i8; 3]),
    RemoveIsolated(u32),
    CancelledDrag(u32),
    Place(i8),
    Undo,
    Redo,
    SetMode(u8),
    Toggle,
    Compact,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        4 => (any::<u32>(), any::<[i8; 3]>()).prop_map(|(v, d)| Op::Move(v, d)),
        2 => any::<u32>().prop_map(Op::RemoveTriangle),
        2 => any::<u32>().prop_map(Op::Split),
        1 => (any::<u32>(), any::<u32>()).prop_map(|(t, v)| Op::Rewrite(t, v)),
        1 => (any::<u32>(), any::<u32>(), any::<[i8; 3]>())
            .prop_map(|(p, q, x)| Op::AddTriangle(p, q, x)),
        1 => any::<u32>().prop_map(Op::RemoveIsolated),
        1 => any::<u32>().prop_map(Op::CancelledDrag),
        1 => any::<i8>().prop_map(Op::Place),
        2 => Just(Op::Undo),
        1 => Just(Op::Redo),
        1 => (0u8..3).prop_map(Op::SetMode),
        1 => Just(Op::Toggle),
        1 => Just(Op::Compact),
    ]
}

fn update_mode(k: u8) -> UpdateMode {
    [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual][usize::from(k % 3)]
}

/// `i8` as a fraction in `[-1, 1]`.
fn unit(x: i8) -> f32 {
    f32::from(x) / 127.0
}

fn alive_vertices(mesh: &Mesh) -> Vec<VertexId> {
    (0..mesh.vertex_slots() as VertexId)
        .filter(|&v| mesh.is_vertex_alive(v))
        .collect()
}

fn alive_triangles(mesh: &Mesh) -> Vec<TriangleId> {
    (0..mesh.triangle_slots() as TriangleId)
        .filter(|&t| mesh.is_triangle_alive(t))
        .collect()
}

/// Picks one of `items` by `k`, if there are any.
fn pick<T: Copy>(items: &[T], k: u32) -> Option<T> {
    (!items.is_empty()).then(|| items[k as usize % items.len()])
}

fn move_vertex(scene: &mut Scene, m: MeshId, v: VertexId, d: [f32; 3]) {
    scene
        .edit(m, |tx| {
            let p = tx.mesh().position(v).unwrap();
            tx.set_positions(&[v], &[std::array::from_fn(|k| p[k] + d[k])])
        })
        .unwrap();
}

/// Applies one operation. Operations that do not apply (nothing to undo, no
/// such vertex) are skipped; the mesh never drops below five triangles.
fn apply(scene: &mut Scene, m: MeshId, on: &mut bool, op: &Op) {
    let mesh = scene.mesh(m).unwrap();
    let verts = alive_vertices(mesh);
    let tris = alive_triangles(mesh);
    match op {
        Op::Move(v, d) => {
            if let Some(v) = pick(&verts, *v) {
                move_vertex(scene, m, v, d.map(|x| unit(x) * 0.05));
            }
        }
        Op::RemoveTriangle(t) => {
            if tris.len() > 5
                && let Some(t) = pick(&tris, *t)
            {
                scene.edit(m, |tx| tx.remove_triangles(&[t])).unwrap();
            }
        }
        Op::Split(t) => {
            if let Some(t) = pick(&tris, *t) {
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
        Op::Rewrite(t, v) => {
            if let (Some(t), Some(v)) = (pick(&tris, *t), pick(&verts, *v)) {
                let [a, b, _] = mesh.triangle(t).unwrap();
                scene
                    .edit(m, |tx| tx.set_triangles(&[t], &[[a, b, v]]))
                    .unwrap();
            }
        }
        Op::AddTriangle(p, q, x) => {
            let (Some(p), Some(q)) = (pick(&verts, *p), pick(&verts, *q)) else {
                return;
            };
            if p == q {
                return;
            }
            scene
                .edit(m, |tx| {
                    let (pp, pq) = (
                        tx.mesh().position(p).unwrap(),
                        tx.mesh().position(q).unwrap(),
                    );
                    let x = std::array::from_fn(|k| (pp[k] + pq[k]) * 0.5 + unit(x[k]) * 0.1);
                    let v = tx.add_vertices(&[x])?;
                    tx.add_triangles(&[[v[0], p, q]])
                })
                .unwrap();
        }
        Op::RemoveIsolated(v) => {
            let isolated: Vec<VertexId> = verts
                .iter()
                .copied()
                .filter(|&v| mesh.corner_count(v) == 0)
                .collect();
            if let Some(v) = pick(&isolated, *v) {
                scene.edit(m, |tx| tx.remove_vertices(&[v])).unwrap();
            }
        }
        Op::CancelledDrag(v) => {
            scene.begin_step();
            if let Some(v) = pick(&verts, *v) {
                move_vertex(scene, m, v, [0.0, 0.0, 0.1]);
            }
            scene.cancel_step().unwrap();
        }
        Op::Place(x) => {
            let p = Placement {
                translation: [f64::from(unit(*x)), 0.5, -0.25],
                ..Placement::default()
            };
            scene.set_placement(m, p).unwrap();
        }
        Op::Undo => {
            scene.undo().unwrap();
        }
        Op::Redo => {
            scene.redo().unwrap();
        }
        Op::SetMode(k) => scene.set_mode(m, update_mode(*k)).unwrap(),
        Op::Toggle => {
            *on = !*on;
            scene.set_normals(m, *on).unwrap();
        }
        Op::Compact => {
            scene.compact(m).unwrap();
        }
    }
}

/// `fast_mesh::compute_vertex_normals` over the alive triangles.
fn reference(mesh: &Mesh) -> Vec<[f32; 3]> {
    let flat: Vec<f32> = mesh.positions().iter().flatten().copied().collect();
    let idx: Vec<u32> = alive_triangles(mesh)
        .into_iter()
        .flat_map(|t| mesh.triangle(t).unwrap())
        .collect();
    let n = fast_mesh::compute_vertex_normals(&flat, &idx).unwrap();
    n.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}

proptest! {
    #![proptest_config(config(64))]

    /// See the module docs.
    #[test]
    fn normals_match_from_scratch(
        mode in 0u8..3,
        batches in prop::collection::vec(prop::collection::vec(op(), 1..6), 1..12),
    ) {
        let mut scene = Scene::new(SceneOptions::default());
        let options = MeshOptions { mode: update_mode(mode), ..MeshOptions::default() };
        let m = scene.add_mesh(sphere(), options).unwrap();
        prop_assert_eq!(scene.mesh(m).unwrap().origin(), ORIGIN);
        scene.set_normals(m, true).unwrap();
        let mut on = true;
        scene.take_changes();
        let mut before = scene.normals(m).unwrap().unwrap().to_vec();

        for (b, batch) in batches.iter().enumerate() {
            for op in batch {
                apply(&mut scene, m, &mut on, op);
            }
            scene.flush().unwrap();
            scene.update(m).unwrap();
            let at = format!("batch {b}: {batch:?}");
            let dirty = scene
                .take_changes()
                .into_iter()
                .find(|c| c.id == m)
                .map_or(Some(Vec::new()), |c| c.normals_dirty);
            let Some(now) = scene.normals(m).unwrap() else {
                prop_assert!(!on, "{}", at);
                before.clear();
                continue;
            };
            let mesh = scene.mesh(m).unwrap();
            prop_assert_eq!(now.len(), mesh.vertex_slots(), "{}", at);
            for (v, (g, w)) in now.iter().zip(reference(mesh)).enumerate() {
                prop_assert!(
                    (0..3).all(|k| (g[k] - w[k]).abs() <= TOL),
                    "{}: vertex {} has {:?}, want {:?}", at, v, g, w
                );
            }
            if let Some(dirty) = &dirty {
                prop_assert!(now.len() >= before.len(), "{}: slots shrank unreported", at);
                for (v, n) in now.iter().enumerate() {
                    // A new slot is reported unless it is still zero (a
                    // vertex added and removed again since the last check).
                    let changed = before.get(v).map_or(*n != [0.0; 3], |o| o != n);
                    prop_assert!(
                        !changed || dirty.binary_search(&(v as VertexId)).is_ok(),
                        "{}: vertex {} changed normal but is not in normals_dirty", at, v
                    );
                }
            }
            before = now.to_vec();
        }
    }
}
