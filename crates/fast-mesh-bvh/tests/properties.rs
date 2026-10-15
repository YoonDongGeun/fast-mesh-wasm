//! Property tests: after any sequence of edits, undos and redos, an updated BVH
//! must satisfy every invariant and answer box queries exactly like brute force
//! and like a freshly built tree.

use fast_mesh_bvh::{Aabb, BuildOptions, Bvh, Quality, UpdatePolicy};
use fast_mesh_store::{History, Mesh, Transaction};
use proptest::prelude::*;

#[derive(Debug, Clone)]
enum Op {
    Move(u32, [i8; 3]),
    AddVertex(u32, u32, u8),
    AddTriangle([u32; 3]),
    RemoveTriangle(u32),
    SetTriangle(u32, [u32; 3]),
}

#[derive(Debug, Clone)]
enum Step {
    Commit(Vec<Op>),
    Undo,
    Redo,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        4 => (any::<u32>(), any::<[i8; 3]>()).prop_map(|(v, d)| Op::Move(v, d)),
        1 => (any::<u32>(), any::<u32>(), any::<u8>()).prop_map(|(a, b, w)| Op::AddVertex(a, b, w)),
        2 => any::<[u32; 3]>().prop_map(Op::AddTriangle),
        2 => any::<u32>().prop_map(Op::RemoveTriangle),
        1 => (any::<u32>(), any::<[u32; 3]>()).prop_map(|(t, v)| Op::SetTriangle(t, v)),
    ]
}

fn step() -> impl Strategy<Value = Step> {
    prop_oneof![
        5 => prop::collection::vec(op(), 1..8).prop_map(Step::Commit),
        1 => Just(Step::Undo),
        1 => Just(Step::Redo),
    ]
}

fn config() -> impl Strategy<Value = (BuildOptions, UpdatePolicy)> {
    (
        prop_oneof![Just(Quality::Sah), Just(Quality::Fast)],
        1u32..=8,
        8u32..=64,
        0.05f32..=1.0,
        1.0f32..=2.0,
    )
        .prop_map(|(quality, leaf, treelet, fraction, factor)| {
            (
                BuildOptions {
                    quality,
                    max_leaf_size: leaf,
                    treelet_size: treelet.max(leaf),
                    ..Default::default()
                },
                UpdatePolicy {
                    full_rebuild_fraction: fraction,
                    degradation_factor: factor,
                },
            )
        })
}

/// A wavy n×n grid, so boxes are not all coplanar.
fn grid(n: u32) -> Mesh {
    let mut positions = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            positions.extend([x as f32, y as f32, ((x * 7 + y * 3) % 5) as f32 * 0.3]);
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

fn alive(m: &Mesh, vertices: bool) -> Vec<u32> {
    let n = if vertices {
        m.vertex_slots()
    } else {
        m.triangle_slots()
    } as u32;
    (0..n)
        .filter(|&i| {
            if vertices {
                m.is_vertex_alive(i)
            } else {
                m.is_triangle_alive(i)
            }
        })
        .collect()
}

fn pick(ids: &[u32], seed: u32) -> Option<u32> {
    (!ids.is_empty()).then(|| ids[seed as usize % ids.len()])
}

fn run(tx: &mut Transaction<'_>, op: &Op) {
    let vs = alive(tx.mesh(), true);
    let ts = alive(tx.mesh(), false);
    let v = |s: u32| pick(&vs, s);
    match *op {
        Op::Move(s, d) => {
            if let Some(v) = v(s) {
                let p = tx.mesh().position(v).unwrap();
                let q = [0, 1, 2].map(|k| p[k] + d[k] as f32 * 0.05);
                tx.set_positions(&[v], &[q]).unwrap();
            }
        }
        Op::AddVertex(a, b, w) => {
            if let (Some(a), Some(b)) = (v(a), v(b)) {
                let (pa, pb) = (
                    tx.mesh().position(a).unwrap(),
                    tx.mesh().position(b).unwrap(),
                );
                let t = w as f32 / 255.0;
                let p = [0, 1, 2].map(|k| pa[k] * (1.0 - t) + pb[k] * t);
                tx.add_vertex_with_parents(p, &[(a, 1.0 - t), (b, t)])
                    .unwrap();
            }
        }
        Op::AddTriangle(s) => {
            if let (Some(a), Some(b), Some(c)) = (v(s[0]), v(s[1]), v(s[2])) {
                tx.add_triangles(&[[a, b, c]]).unwrap();
            }
        }
        Op::RemoveTriangle(s) => {
            if let Some(t) = pick(&ts, s) {
                tx.remove_triangles(&[t]).unwrap();
            }
        }
        Op::SetTriangle(t, s) => {
            if let (Some(t), Some(a), Some(b), Some(c)) = (pick(&ts, t), v(s[0]), v(s[1]), v(s[2]))
            {
                tx.set_triangles(&[t], &[[a, b, c]]).unwrap();
            }
        }
    }
}

fn query(bvh: &Bvh, mesh: &Mesh, q: &Aabb) -> Vec<u32> {
    let mut out = Vec::new();
    bvh.query_aabb(mesh, q.min, q.max, &mut out);
    out.sort();
    out
}

fn brute(mesh: &Mesh, q: &Aabb) -> Vec<u32> {
    alive(mesh, false)
        .into_iter()
        .filter(|&t| Aabb::of_triangle(mesh.positions(), mesh.triangle(t).unwrap()).overlaps(q))
        .collect()
}

/// Boxes spread over the grid, from tiny to covering everything.
fn probe_boxes(n: u32) -> Vec<Aabb> {
    let n = n as f32;
    let mut boxes = vec![Aabb::new([-1.0; 3], [n + 1.0, n + 1.0, 5.0])];
    for i in 0..6 {
        let c = i as f32 * n / 6.0;
        let r = 0.3 + i as f32 * 0.7;
        boxes.push(Aabb::new([c - r, c - r, -1.0], [c + r, c + r, 2.0]));
        boxes.push(Aabb::new([c, n - c, 0.0], [c + 0.1, n - c + 0.1, 0.4]));
    }
    boxes
}

fn check(bvh: &Bvh, mesh: &Mesh, options: &BuildOptions, n: u32) -> Result<(), TestCaseError> {
    bvh.validate(mesh).map_err(TestCaseError::fail)?;
    prop_assert!(bvh.is_current(mesh));
    let fresh = Bvh::build(mesh, options).unwrap();
    for q in probe_boxes(n) {
        let got = query(bvh, mesh, &q);
        prop_assert_eq!(&got, &brute(mesh, &q));
        prop_assert_eq!(&got, &query(&fresh, mesh, &q));
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn updated_trees_match_brute_force(
        (options, policy) in config(),
        n in 3u32..10,
        steps in prop::collection::vec(step(), 1..25),
    ) {
        let mut m = grid(n);
        let mut bvh = Bvh::build(&m, &options).unwrap();
        bvh.set_policy(policy).unwrap();
        let mut h = History::new(usize::MAX);
        for step in &steps {
            let changes = match step {
                Step::Commit(ops) => {
                    let mut tx = m.begin_edit();
                    for op in ops {
                        run(&mut tx, op);
                    }
                    let patch = tx.commit().unwrap();
                    let changes = patch.changes().clone();
                    h.push(&mut m, patch).unwrap();
                    Some(changes)
                }
                Step::Undo => h.undo(&mut m).unwrap(),
                Step::Redo => h.redo(&mut m).unwrap(),
            };
            if let Some(changes) = changes {
                bvh.update(&mut m, &changes).unwrap();
            }
            check(&bvh, &m, &options, n)?;
        }
    }
}
