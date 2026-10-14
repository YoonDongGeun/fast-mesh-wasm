//! Property tests: random edit sequences must keep the store consistent, and
//! undo/redo/rollback must restore earlier states exactly.

use fast_mesh_store::{ChannelDesc, History, Mesh, TopologyPolicy, Transaction};
use proptest::prelude::*;

#[derive(Debug, Clone)]
enum Op {
    Move(u32, [i8; 3]),
    AddVertex(u32, u32, u8),
    AddTriangle([u32; 3]),
    RemoveTriangle(u32),
    RemoveVertex(u32),
    SetTriangle(u32, [u32; 3]),
    Paint(u32, i8),
}

#[derive(Debug, Clone)]
enum Step {
    Commit(Vec<Op>),
    Rollback(Vec<Op>),
    Undo,
    Redo,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        3 => (any::<u32>(), any::<[i8; 3]>()).prop_map(|(v, d)| Op::Move(v, d)),
        2 => (any::<u32>(), any::<u32>(), any::<u8>()).prop_map(|(a, b, w)| Op::AddVertex(a, b, w)),
        2 => any::<[u32; 3]>().prop_map(Op::AddTriangle),
        2 => any::<u32>().prop_map(Op::RemoveTriangle),
        1 => any::<u32>().prop_map(Op::RemoveVertex),
        1 => (any::<u32>(), any::<[u32; 3]>()).prop_map(|(t, v)| Op::SetTriangle(t, v)),
        2 => (any::<u32>(), any::<i8>()).prop_map(|(v, x)| Op::Paint(v, x)),
    ]
}

fn ops() -> impl Strategy<Value = Vec<Op>> {
    prop::collection::vec(op(), 1..6)
}

fn step() -> impl Strategy<Value = Step> {
    prop_oneof![
        4 => ops().prop_map(Step::Commit),
        1 => ops().prop_map(Step::Rollback),
        2 => Just(Step::Undo),
        2 => Just(Step::Redo),
    ]
}

/// An n×n grid of quads with two source channels and one derived channel.
/// `grid(8)` has 81 vertices, so edits cross a 64-bit liveness word boundary.
fn grid(n: u32) -> Mesh {
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
            let (b, c, d) = (a + 1, a + n + 2, a + n + 1);
            indices.extend([a, b, c, a, c, d]);
        }
    }
    let mut m = Mesh::from_buffers(&positions, &indices).unwrap();
    m.add_channel(ChannelDesc::source("mask", 1)).unwrap();
    m.add_channel(ChannelDesc::source("label", 2).with_policy(TopologyPolicy::Nearest))
        .unwrap();
    m.add_channel(ChannelDesc::derived("dist", 1)).unwrap();
    m
}

fn alive_vertices(m: &Mesh) -> Vec<u32> {
    (0..m.vertex_slots() as u32)
        .filter(|&v| m.is_vertex_alive(v))
        .collect()
}

fn alive_triangles(m: &Mesh) -> Vec<u32> {
    (0..m.triangle_slots() as u32)
        .filter(|&t| m.is_triangle_alive(t))
        .collect()
}

fn pick(ids: &[u32], seed: u32) -> Option<u32> {
    (!ids.is_empty()).then(|| ids[seed as usize % ids.len()])
}

/// Applies one random op. Validation errors are expected and ignored; they must not
/// change anything.
fn run(tx: &mut Transaction<'_>, op: &Op) {
    let vs = alive_vertices(tx.mesh());
    let ts = alive_triangles(tx.mesh());
    let v = |s: u32| pick(&vs, s);
    match *op {
        Op::Move(s, d) => {
            if let Some(v) = v(s) {
                let p = tx.mesh().position(v).unwrap();
                let q = [0, 1, 2].map(|k| p[k] + d[k] as f32 * 0.25);
                tx.set_positions(&[v], &[q]).unwrap();
            }
        }
        Op::AddVertex(a, b, w) => {
            if let (Some(a), Some(b)) = (v(a), v(b)) {
                let w = w as f32 / 255.0;
                tx.add_vertex_with_parents([0.5; 3], &[(a, w), (b, 1.0 - w)])
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
        Op::RemoveVertex(s) => {
            if let Some(v) = v(s) {
                let _ = tx.remove_vertices(&[v]); // fails while in use
            }
        }
        Op::SetTriangle(t, s) => {
            if let (Some(t), Some(a), Some(b), Some(c)) = (pick(&ts, t), v(s[0]), v(s[1]), v(s[2]))
            {
                tx.set_triangles(&[t], &[[a, b, c]]).unwrap();
            }
        }
        Op::Paint(s, x) => {
            if let Some(v) = v(s) {
                tx.write_channel("mask", &[v], &[x as f32]).unwrap();
                tx.write_channel("label", &[v], &[x as f32, -(x as f32)])
                    .unwrap();
            }
        }
    }
}

/// Everything observable about alive elements, compared bit for bit.
#[derive(Debug, PartialEq)]
struct State {
    vertices: Vec<(u32, [u32; 3], Vec<u32>)>,
    triangles: Vec<(u32, [u32; 3])>,
    counts: (usize, usize),
}

fn state(m: &Mesh) -> State {
    let channel = |name: &str, v: u32, w: usize| -> Vec<u32> {
        m.channel(name).unwrap()[v as usize * w..(v as usize + 1) * w]
            .iter()
            .map(|x| x.to_bits())
            .collect()
    };
    State {
        vertices: alive_vertices(m)
            .into_iter()
            .map(|v| {
                let mut values = channel("mask", v, 1);
                values.extend(channel("label", v, 2));
                (v, m.position(v).unwrap().map(f32::to_bits), values)
            })
            .collect(),
        triangles: alive_triangles(m)
            .into_iter()
            .map(|t| (t, m.triangle(t).unwrap()))
            .collect(),
        counts: (m.vertex_count(), m.triangle_count()),
    }
}

fn check_invariants(m: &Mesh) {
    let vs = alive_vertices(m);
    let ts = alive_triangles(m);
    assert_eq!(m.vertex_count(), vs.len());
    assert_eq!(m.triangle_count(), ts.len());
    let mut corners = vec![0u32; m.vertex_slots()];
    for &t in &ts {
        for v in m.triangle(t).unwrap() {
            assert!(m.is_vertex_alive(v), "triangle {t} uses dead vertex {v}");
            corners[v as usize] += 1;
        }
    }
    for v in 0..m.vertex_slots() as u32 {
        let expected = if m.is_vertex_alive(v) {
            corners[v as usize]
        } else {
            0
        };
        assert_eq!(m.corner_count(v), expected, "corner count of vertex {v}");
    }
    for t in 0..m.triangle_slots() as u32 {
        if !m.is_triangle_alive(t) {
            assert_eq!(m.triangles()[t as usize], [0, 0, 0]);
        }
    }
    let mut free_v = m.free_vertex_slots().to_vec();
    free_v.sort_unstable();
    assert!(
        free_v.windows(2).all(|w| w[0] != w[1]),
        "duplicate free vertex slot"
    );
    assert!(
        free_v.iter().all(|&v| !m.is_vertex_alive(v)),
        "alive vertex on free list"
    );
    let mut free_t = m.free_triangle_slots().to_vec();
    free_t.sort_unstable();
    assert!(
        free_t.windows(2).all(|w| w[0] != w[1]),
        "duplicate free triangle slot"
    );
    assert!(
        free_t.iter().all(|&t| !m.is_triangle_alive(t)),
        "alive triangle on free list"
    );
    for name in ["mask", "dist"] {
        assert_eq!(m.channel(name).unwrap().len(), m.vertex_slots());
    }
    assert_eq!(m.channel("label").unwrap().len(), m.vertex_slots() * 2);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn undo_all_and_redo_all_restore_exact_states(commits in prop::collection::vec(ops(), 1..12)) {
        let mut m = grid(8);
        let mut h = History::new(usize::MAX);
        let initial = state(&m);
        for ops in &commits {
            let mut tx = m.begin_edit();
            for op in ops {
                run(&mut tx, op);
            }
            let patch = tx.commit().unwrap();
            h.push(&mut m, patch).unwrap();
            check_invariants(&m);
        }
        let last = state(&m);
        while h.undo(&mut m).unwrap().is_some() {
            check_invariants(&m);
        }
        prop_assert_eq!(state(&m), initial);
        while h.redo(&mut m).unwrap().is_some() {
            check_invariants(&m);
        }
        prop_assert_eq!(state(&m), last);
    }

    #[test]
    fn interleaved_steps_match_a_model(steps in prop::collection::vec(step(), 1..30), budget in 200usize..4000) {
        let mut m = grid(8);
        let mut h = History::new(budget);
        // States before each patch on the undo stack, and after each patch on the redo stack.
        let mut undo_states: Vec<State> = Vec::new();
        let mut redo_states: Vec<State> = Vec::new();
        for step in &steps {
            match step {
                Step::Commit(ops) => {
                    let before = state(&m);
                    let mut tx = m.begin_edit();
                    for op in ops {
                        run(&mut tx, op);
                    }
                    let patch = tx.commit().unwrap();
                    let empty = patch.is_empty();
                    h.push(&mut m, patch).unwrap();
                    if !empty {
                        undo_states.push(before);
                        redo_states.clear();
                    }
                    let evicted = undo_states.len() - h.undo_len();
                    undo_states.drain(..evicted);
                    prop_assert!(h.bytes() <= h.budget());
                }
                Step::Rollback(ops) => {
                    let before = state(&m);
                    let revision = m.revision();
                    let mut tx = m.begin_edit();
                    for op in ops {
                        run(&mut tx, op);
                    }
                    tx.rollback();
                    prop_assert_eq!(state(&m), before);
                    prop_assert_eq!(m.revision(), revision);
                }
                Step::Undo => {
                    let current = state(&m);
                    if h.undo(&mut m).unwrap().is_some() {
                        prop_assert_eq!(&state(&m), undo_states.last().unwrap());
                        undo_states.pop();
                        redo_states.push(current);
                    } else {
                        prop_assert!(undo_states.is_empty());
                    }
                }
                Step::Redo => {
                    let current = state(&m);
                    if h.redo(&mut m).unwrap().is_some() {
                        prop_assert_eq!(&state(&m), redo_states.last().unwrap());
                        redo_states.pop();
                        undo_states.push(current);
                    } else {
                        prop_assert!(redo_states.is_empty());
                    }
                }
            }
            check_invariants(&m);
        }
    }

    #[test]
    fn compact_preserves_alive_geometry(commits in prop::collection::vec(ops(), 1..8)) {
        let mut m = grid(8);
        let mut h = History::new(usize::MAX);
        for ops in &commits {
            let mut tx = m.begin_edit();
            for op in ops {
                run(&mut tx, op);
            }
            let patch = tx.commit().unwrap();
            h.push(&mut m, patch).unwrap();
        }
        let before = state(&m);
        let remap = h.compact(&mut m).unwrap();
        check_invariants(&m);
        prop_assert!(m.free_vertex_slots().is_empty());
        prop_assert_eq!(m.vertex_slots(), m.vertex_count());
        prop_assert_eq!(m.triangle_slots(), m.triangle_count());
        for (v, pos, values) in &before.vertices {
            let nv = remap.vertex(*v).unwrap();
            prop_assert_eq!(m.position(nv).unwrap().map(f32::to_bits), *pos);
            prop_assert_eq!(m.channel("mask").unwrap()[nv as usize].to_bits(), values[0]);
        }
        for (t, tri) in &before.triangles {
            let nt = remap.triangle(*t).unwrap();
            prop_assert_eq!(m.triangle(nt).unwrap(), tri.map(|v| remap.vertex(v).unwrap()));
        }
    }
}
