//! The `parallel` feature must not change a single bit of a built tree, for
//! any thread count. Every test passes with and without the feature; the
//! golden hashes tie the two builds together.

use fast_mesh_bvh::{BuildOptions, Bvh, NodeKind, Quality};
use fast_mesh_store::{Mesh, par};

/// Thread counts the parallel build is compared under.
const POOLS: [usize; 3] = [1, 2, 8];

/// `|t - round(t)|`: a triangle wave in [0, 0.5] built from exact float
/// operations only, so the mesh is the same on every platform (unlike `sin`).
fn wave(t: f32) -> f32 {
    (t - t.round()).abs()
}

/// Wavy n×n grid (2n² triangles).
fn wavy(n: u32) -> Mesh {
    let mut positions = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            let (fx, fy) = (x as f32, y as f32);
            positions.extend([fx, fy, 8.0 * wave(fx * 0.031) * wave(fy * 0.047)]);
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

/// `node_slots` and every node reachable from the root, in traversal order:
/// index, bound bits, and children or (after `u32::MAX`) triangles.
type Tree = (usize, Vec<(usize, [u32; 6], Vec<u32>)>);

fn nodes(bvh: &Bvh) -> Tree {
    let mut out = Vec::new();
    bvh.traverse(|id, view| {
        let b = view.bounds;
        let bits = [b.min[0], b.min[1], b.min[2], b.max[0], b.max[1], b.max[2]].map(f32::to_bits);
        let kind = match view.kind {
            NodeKind::Inner { left, right } => vec![left.index() as u32, right.index() as u32],
            NodeKind::Leaf { triangles } => {
                let mut v = vec![u32::MAX];
                v.extend_from_slice(triangles);
                v
            }
        };
        out.push((id.index(), bits, kind));
        true
    });
    (bvh.node_slots(), out)
}

/// FNV-1a over the node records.
fn hash(tree: &Tree) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |x: u64| {
        for byte in x.to_le_bytes() {
            h = (h ^ byte as u64).wrapping_mul(0x0100_0000_01b3);
        }
    };
    eat(tree.0 as u64);
    for (i, bits, kind) in &tree.1 {
        eat(*i as u64);
        bits.iter().for_each(|&b| eat(b as u64));
        kind.iter().for_each(|&k| eat(k as u64));
    }
    h
}

#[test]
fn build_bit_identical() {
    let mesh = wavy(317);
    assert!(mesh.triangle_count() >= 200_000);
    // Golden hashes of the sequential trees: the same with and without the
    // feature, so a feature-off run checks the feature-on result too.
    for (quality, golden) in [
        (Quality::Sah, 10369004575951647529u64),
        (Quality::Fast, 5049794685599701823),
    ] {
        let opts = BuildOptions {
            quality,
            ..Default::default()
        };
        par::set_force_sequential(true);
        let reference = nodes(&Bvh::build(&mesh, &opts).unwrap());
        par::set_force_sequential(false);
        assert_eq!(hash(&reference), golden, "{quality:?} golden hash");
        for k in POOLS {
            let tree = par::with_threads(k, || nodes(&Bvh::build(&mesh, &opts).unwrap()));
            assert!(tree == reference, "{quality:?} differs with {k} threads");
        }
        // And on the global pool, from this thread.
        let before = par::parallel_entries();
        let tree = nodes(&Bvh::build(&mesh, &opts).unwrap());
        assert!(tree == reference, "{quality:?} differs on the global pool");
        if par::enabled() {
            assert!(
                par::parallel_entries() > before,
                "{quality:?} ran sequentially"
            );
        }
    }
}

#[test]
fn small_inputs_stay_sequential() {
    let positions = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
    ];
    let indices = [
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, //
        1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7,
    ];
    let cube = Mesh::from_buffers(&positions, &indices).unwrap();
    assert_eq!(cube.triangle_count(), 12);
    let before = par::parallel_entries();
    for quality in [Quality::Sah, Quality::Fast] {
        let opts = BuildOptions {
            quality,
            ..Default::default()
        };
        Bvh::build(&cube, &opts).unwrap();
    }
    assert_eq!(par::parallel_entries(), before);
}
