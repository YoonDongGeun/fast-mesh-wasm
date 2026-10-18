//! Property test for the repair pipeline: on a damaged sphere or grid,
//! `weld(0) → remove_degenerate → remove_small_components(10) → orient →
//! fill_holes(max_edges 300)` in one transaction never panics and never
//! fails (short of running out of memory), every report agrees with an
//! `analyze` of the mesh before and after its step, and the result has no
//! orientation conflicts when every patch was orientable.
//!
//! The damage: holes (triangles left out), flipped triangles, duplicated
//! vertices (a corner moved to an exact copy of its vertex), floating pieces
//! of 1 to 14 triangles, and degenerate triangles (a repeated vertex id, a
//! repeated triangle, a corner on an exact copy of another corner).
//!
//! Seeds: `PROPTEST_RNG_SEED` is honoured; without it the seed is fixed, so
//! that CI runs are deterministic.

use fast_mesh_ops::{
    Analysis, AnalyzeOptions, ComponentOptions, DegenerateOptions, FillOptions, LoopId, OpsError,
    analyze, fill_holes, orient, remove_degenerate, remove_small_components, weld,
};
use fast_mesh_store::{Mesh, StoreError, Transaction, TriangleId};
use proptest::prelude::*;
use proptest::test_runner::RngSeed;

/// The seed used when `PROPTEST_RNG_SEED` is not set.
const DEFAULT_SEED: u64 = 0x0e9a_1207;

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

/// Components below this many triangles are removed.
const MIN_TRIANGLES: usize = 10;
/// Loops up to this many edges are filled.
const MAX_EDGES: usize = 300;

type Soup = (Vec<[f32; 3]>, Vec<[u32; 3]>);

/// A closed UV sphere of radius 1, outward wound.
fn sphere(stacks: u32, slices: u32) -> Soup {
    let mut pos = vec![[0.0f32, 0.0, 1.0]];
    for i in 1..stacks {
        let theta = std::f64::consts::PI * f64::from(i) / f64::from(stacks);
        for j in 0..slices {
            let phi = std::f64::consts::TAU * f64::from(j) / f64::from(slices);
            let (s, c) = theta.sin_cos();
            pos.push([(s * phi.cos()) as f32, (s * phi.sin()) as f32, c as f32]);
        }
    }
    pos.push([0.0, 0.0, -1.0]);
    let south = 1 + (stacks - 1) * slices;
    let ring = |i: u32, j: u32| 1 + (i - 1) * slices + j % slices;
    let mut tris = Vec::new();
    for j in 0..slices {
        tris.push([0, ring(1, j), ring(1, j + 1)]);
        for i in 1..stacks - 1 {
            let (a, b, c, d) = (
                ring(i, j),
                ring(i + 1, j),
                ring(i + 1, j + 1),
                ring(i, j + 1),
            );
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
        tris.push([ring(stacks - 1, j), south, ring(stacks - 1, j + 1)]);
    }
    (pos, tris)
}

/// An open `n x n` grid of unit squares with a wave in z.
fn grid(n: u32) -> Soup {
    let mut pos = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            let (fx, fy) = (f64::from(x), f64::from(y));
            let z = 0.3 * (fx * 0.9).sin() * (fy * 0.7).cos();
            pos.push([fx as f32, fy as f32, z as f32]);
        }
    }
    let mut tris = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            let c = a + n + 1;
            tris.push([a, a + 1, c]);
            tris.push([a + 1, c + 1, c]);
        }
    }
    (pos, tris)
}

#[derive(Debug, Clone)]
struct Damage {
    holes: Vec<u32>,
    flips: Vec<u32>,
    /// `(triangle, corner)`: the corner moves to a new exact copy of its vertex.
    splits: Vec<(u32, u8)>,
    /// Triangle counts of floating pieces.
    pieces: Vec<u8>,
    /// `(triangle, kind)`: a degenerate triangle built from that triangle.
    degenerates: Vec<(u32, u8)>,
}

fn damage() -> impl Strategy<Value = Damage> {
    (
        prop::collection::vec(any::<u32>(), 0..5),
        prop::collection::vec(any::<u32>(), 0..8),
        prop::collection::vec((any::<u32>(), 0u8..3), 0..5),
        prop::collection::vec(1u8..15, 0..4),
        prop::collection::vec((any::<u32>(), 0u8..3), 0..5),
    )
        .prop_map(|(holes, flips, splits, pieces, degenerates)| Damage {
            holes,
            flips,
            splits,
            pieces,
            degenerates,
        })
}

/// `soup` with `d` applied.
fn damaged((mut pos, base): Soup, d: &Damage) -> Mesh {
    let n = base.len() as u32;
    let mut tris = base.clone();
    for &f in &d.flips {
        tris[(f % n) as usize].swap(1, 2);
    }
    for &(t, c) in &d.splits {
        let corner = &mut tris[(t % n) as usize][usize::from(c)];
        pos.push(pos[*corner as usize]);
        *corner = pos.len() as u32 - 1;
    }
    let mut out: Vec<[u32; 3]> = tris
        .iter()
        .enumerate()
        .filter(|(k, _)| !d.holes.iter().any(|&h| (h % n) as usize == *k))
        .map(|(_, t)| *t)
        .collect();
    for &(t, kind) in &d.degenerates {
        let [a, b, c] = base[(t % n) as usize];
        match kind {
            0 => out.push([a, a, b]),
            1 => out.push([a, c, b]),
            _ => {
                pos.push(pos[a as usize]);
                out.push([a, b, pos.len() as u32 - 1]);
            }
        }
    }
    // Small strips far from everything.
    for (k, &count) in d.pieces.iter().enumerate() {
        let x0 = 10.0 + 3.0 * k as f32;
        let first = pos.len() as u32;
        for i in 0..=u32::from(count) / 2 + 1 {
            let x = x0 + 0.1 * i as f32;
            pos.push([x, 0.0, 5.0]);
            pos.push([x, 0.1, 5.0]);
        }
        for i in 0..u32::from(count) {
            let (q, odd) = (first + 2 * (i / 2), i % 2 == 1);
            out.push(if odd {
                [q + 1, q + 2, q + 3]
            } else {
                [q, q + 2, q + 1]
            });
        }
    }
    let flat: Vec<f32> = pos.iter().flatten().copied().collect();
    let idx: Vec<u32> = out.iter().flatten().copied().collect();
    Mesh::from_buffers(&flat, &idx).unwrap()
}

fn full(tx: &Transaction) -> Analysis {
    let options = AnalyzeOptions {
        max_listed: usize::MAX,
        ..AnalyzeOptions::default()
    };
    analyze(tx.mesh(), &options).unwrap()
}

/// The component `remove_small_components` keeps although small: most
/// area, then most triangles, then the smallest triangle id.
fn largest(a: &Analysis) -> Option<usize> {
    (0..a.components.len()).max_by(|&i, &j| {
        let (p, q) = (&a.components[i], &a.components[j]);
        p.area
            .total_cmp(&q.area)
            .then(p.triangles.len().cmp(&q.triangles.len()))
            .then(q.triangles[0].cmp(&p.triangles[0]))
    })
}

/// The value of a repair step; returns from the test case on its error,
/// passing on `OutOfMemory` (the one allowed failure) and failing otherwise.
macro_rules! step {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(OpsError::Store(StoreError::OutOfMemory)) => return Ok(()),
            Err(e) => {
                return Err(TestCaseError::fail(format!(
                    "{} failed: {e}",
                    stringify!($e)
                )))
            }
        }
    };
}

/// Runs the pipeline, checking each step's report against `analyze`.
fn pipeline(tx: &mut Transaction) -> Result<(), TestCaseError> {
    // weld(0)
    let a = full(tx);
    let merged: usize = a.coincident.iter().map(|g| g.len() - 1).sum();
    let r = step!(weld(tx, 0.0));
    prop_assert_eq!(r.merged_vertices, merged);
    let a = full(tx);
    prop_assert_eq!(a.coincident_groups, 0);

    // remove_degenerate
    let mut doomed: Vec<TriangleId> = a
        .degenerate_list
        .iter()
        .chain(&a.duplicate_list)
        .copied()
        .collect();
    doomed.sort_unstable();
    doomed.dedup();
    let r = step!(remove_degenerate(tx, &DegenerateOptions::default()));
    prop_assert_eq!(r.removed_triangles, doomed.len());
    let a = full(tx);
    prop_assert_eq!((a.degenerate, a.duplicate), (0, 0));
    prop_assert_eq!(a.coincident_groups, 0);

    // remove_small_components
    let keep = largest(&a);
    let small: Vec<usize> = (0..a.components.len())
        .filter(|&i| a.components[i].triangles.len() < MIN_TRIANGLES && Some(i) != keep)
        .collect();
    let options = ComponentOptions {
        min_triangles: Some(MIN_TRIANGLES),
        ..ComponentOptions::default()
    };
    let r = step!(remove_small_components(tx, &options));
    prop_assert_eq!(r.removed_components, small.len());
    let removed: usize = small.iter().map(|&i| a.components[i].triangles.len()).sum();
    prop_assert_eq!(r.removed_triangles, removed);
    let a = full(tx);
    let below = |a: &Analysis| {
        a.components
            .iter()
            .filter(|c| c.triangles.len() < MIN_TRIANGLES)
            .count()
    };
    prop_assert!(below(&a) <= 1, "{:?}", a.components);

    // orient
    let r = step!(orient(tx));
    let a = full(tx);
    let orientable = r.non_orientable_patches == 0;
    if orientable {
        prop_assert_eq!(a.orientation_conflicts, 0);
    }

    // fill_holes
    let options = FillOptions {
        max_edges: Some(MAX_EDGES),
        ..FillOptions::default()
    };
    let r = step!(fill_holes(tx, &options));
    let mut reported: Vec<LoopId> = r
        .filled
        .iter()
        .map(|f| f.id)
        .chain(r.skipped.iter().map(|s| s.0))
        .chain(r.failed.iter().map(|f| f.0))
        .collect();
    reported.sort_unstable_by_key(|id| (id.min_vertex, id.len, id.hash));
    let mut before: Vec<LoopId> = a.loops.iter().map(|l| l.id).collect();
    before.sort_unstable_by_key(|id| (id.min_vertex, id.len, id.hash));
    prop_assert_eq!(reported, before, "every loop is reported once");

    let after = full(tx);
    prop_assert_eq!(after.coincident_groups, 0);
    prop_assert_eq!((after.degenerate, after.duplicate), (0, 0));
    prop_assert_eq!(after.non_manifold_edges, a.non_manifold_edges);
    prop_assert!(below(&after) <= 1, "{:?}", after.components);
    if orientable {
        prop_assert_eq!(after.orientation_conflicts, 0);
    }
    for l in &after.loops {
        prop_assert!(
            r.filled.iter().all(|f| f.id != l.id),
            "filled loop {:?} is still open",
            l.id
        );
        if l.vertices.len() <= MAX_EDGES {
            let left =
                r.skipped.iter().any(|s| s.0 == l.id) || r.failed.iter().any(|f| f.0 == l.id);
            prop_assert!(left, "loop {:?} was neither filled nor reported", l.id);
        }
    }
    Ok(())
}

fn base() -> impl Strategy<Value = Soup> {
    prop_oneof![
        (3u32..8, 4u32..12).prop_map(|(st, sl)| sphere(st, sl)),
        (2u32..9).prop_map(grid),
    ]
}

proptest! {
    #![proptest_config(config(64))]

    #[test]
    fn pipeline_never_panics_and_orients(soup in base(), d in damage()) {
        let mut mesh = damaged(soup, &d);
        let mut tx = mesh.begin_edit();
        pipeline(&mut tx)?;
        tx.commit().unwrap();
    }
}
