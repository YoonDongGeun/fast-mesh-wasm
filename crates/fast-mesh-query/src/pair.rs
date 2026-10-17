//! Simultaneous depth-first traversal of two BVHs, the core of the
//! mesh-vs-mesh queries.
//!
//! The traversal never allocates on the heap and never fails: pending pairs
//! live in a fixed array on the call stack, and a pair that does not fit is
//! finished at once by a nested parent-link walk of the two subtrees.

use fast_mesh_bvh::{Aabb, Bvh, NodeId, NodeKind, NodeView};

use crate::Rigid;

/// How many pending node pairs [`traverse_pairs`] keeps in its fixed stack
/// (40 bytes each). Pending pairs grow by at most one per level of the two
/// trees combined, so balanced builds never come close.
pub(crate) const PAIR_STACK: usize = 128;

/// What [`traverse_pairs`] does with a pair once `visit` has seen it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Step {
    /// Expand the pair into its two child pairs.
    Descend,
    /// Drop the pair and everything below it.
    Skip,
    /// End the whole traversal.
    Stop,
}

/// A pair waiting on the stack: a node of A, a node of B, B's box already
/// moved into A's space, and the pair's key.
#[derive(Clone, Copy)]
struct Pending<K> {
    a: NodeId,
    b: NodeId,
    b_box: Aabb,
    key: K,
}

/// Visits pairs of nodes of `a` and `b` depth-first, smaller key first, with B
/// placed in A's space by `xf`.
///
/// `key` gets A's box and B's box moved into A's space
/// ([`Rigid::transform_box`]), and returns `None` to cull the pair with
/// everything below it. It runs once per created pair, the root pair included.
/// Keys are compared with [`PartialOrd`], so a key can carry more than the
/// value it orders by (a tuple orders by its first field, then the next).
///
/// `visit` gets each pair when it is popped, with its key. For a leaf–leaf
/// pair it does the triangle work and returns [`Step::Skip`] or [`Step::Stop`]
/// (`Descend` acts as `Skip` there). For any other pair, `Descend` expands it,
/// `Skip` drops it and `Stop` ends the traversal. Expanding splits the inner
/// node with the larger box surface area (B's measured in A's space, A on
/// ties), or the only inner one. Of the two child pairs, the one with the
/// smaller key is visited first (the left child on equal or incomparable keys).
///
/// Nothing is visited when either tree is empty. Up to [`PAIR_STACK`] pairs
/// wait on a fixed stack; see [`traverse_pairs_with`] for what happens beyond.
///
/// Test-only: the queries call `traverse_pairs_with::<PAIR_STACK>` directly,
/// so that their tests can force the fallback with a smaller stack.
#[cfg(test)]
pub(crate) fn traverse_pairs<K: Copy + PartialOrd>(
    a: &Bvh,
    b: &Bvh,
    xf: &Rigid,
    key: impl FnMut(&Aabb, &Aabb) -> Option<K>,
    visit: impl FnMut(&NodeView, &NodeView, K) -> Step,
) {
    traverse_pairs_with::<PAIR_STACK, K>(a, b, xf, key, visit);
}

/// [`traverse_pairs`] with an `N`-entry stack, so tests can force the
/// overflow fallback.
///
/// A child pair that does not fit on the full stack is finished at once, before
/// the pairs below it on the stack, by [`finish_stackless`]. In there `visit`
/// sees only leaf–leaf pairs, in no particular key order, and `Stop` still ends
/// everything. Queries stay correct regardless, since they check their bounds
/// again in `visit`.
pub(crate) fn traverse_pairs_with<const N: usize, K: Copy + PartialOrd>(
    a: &Bvh,
    b: &Bvh,
    xf: &Rigid,
    mut key: impl FnMut(&Aabb, &Aabb) -> Option<K>,
    mut visit: impl FnMut(&NodeView, &NodeView, K) -> Step,
) {
    let (Some(ra), Some(rb)) = (a.root(), b.root()) else {
        return;
    };
    let b_box = xf.transform_box(&b.node(rb).bounds);
    let Some(k) = key(&a.node(ra).bounds, &b_box) else {
        return;
    };
    let root = Pending {
        a: ra,
        b: rb,
        b_box,
        key: k,
    };
    if N == 0 {
        finish_stackless(a, b, xf, root, &mut key, &mut visit);
        return;
    }
    let mut stack = [root; N];
    let mut len = 1;
    while len > 0 {
        len -= 1;
        let p = stack[len];
        let va = a.node(p.a);
        let vb = b.node(p.b);
        match visit(&va, &vb, p.key) {
            Step::Descend => {}
            Step::Skip => continue,
            Step::Stop => return,
        }
        let (split_a, left, right) = match (va.kind, vb.kind) {
            (NodeKind::Leaf { .. }, NodeKind::Leaf { .. }) => continue,
            (NodeKind::Inner { left, right }, NodeKind::Leaf { .. }) => (true, left, right),
            (NodeKind::Leaf { .. }, NodeKind::Inner { left, right }) => (false, left, right),
            (
                NodeKind::Inner { left, right },
                NodeKind::Inner {
                    left: bl,
                    right: br,
                },
            ) => {
                if va.bounds.surface_area() >= p.b_box.surface_area() {
                    (true, left, right)
                } else {
                    (false, bl, br)
                }
            }
        };
        // Splitting A keeps B's moved box; splitting B moves each child's box
        // once, for the one pair it makes.
        let [l, r] = [left, right].map(|c| {
            let (a_box, child) = if split_a {
                (a.node(c).bounds, Pending { a: c, ..p })
            } else {
                let b_box = xf.transform_box(&b.node(c).bounds);
                (va.bounds, Pending { b: c, b_box, ..p })
            };
            key(&a_box, &child.b_box).map(|key| Pending { key, ..child })
        });
        let (first, second) = match (l, r) {
            (Some(lp), Some(rp)) if rp.key < lp.key => (r, l),
            _ => (l, r),
        };
        // The second pair goes below the first. It always fits, as an entry
        // was just popped.
        if let Some(s) = second {
            stack[len] = s;
            len += 1;
        }
        if let Some(f) = first {
            if len < N {
                stack[len] = f;
                len += 1;
            } else if !finish_stackless(a, b, xf, f, &mut key, &mut visit) {
                return;
            }
        }
    }
}

/// Visits every leaf–leaf pair below the pending pair `p` that `key` does not
/// cull, without a stack: a parent-link walk of A's subtree, and for each of
/// its leaves one of B's subtree. Inner pairs are only tested with `key`, not
/// shown to `visit`. `key` runs once per pair it tests (never for `p` itself,
/// whose key is known). Returns `false` when `visit` returned [`Step::Stop`].
fn finish_stackless<K: Copy>(
    a: &Bvh,
    b: &Bvh,
    xf: &Rigid,
    p: Pending<K>,
    key: &mut impl FnMut(&Aabb, &Aabb) -> Option<K>,
    visit: &mut impl FnMut(&NodeView, &NodeView, K) -> Step,
) -> bool {
    let mut stopped = false;
    a.traverse_from(p.a, |ida, va| {
        if stopped {
            return false;
        }
        // The key of (`ida`, `p.b`).
        let ka = if ida == p.a {
            p.key
        } else {
            match key(&va.bounds, &p.b_box) {
                Some(k) => k,
                None => return false,
            }
        };
        if let NodeKind::Leaf { .. } = va.kind {
            b.traverse_from(p.b, |idb, vb| {
                if stopped {
                    return false;
                }
                let kb = if idb == p.b {
                    ka
                } else {
                    match key(&va.bounds, &xf.transform_box(&vb.bounds)) {
                        Some(k) => k,
                        None => return false,
                    }
                };
                if let NodeKind::Leaf { .. } = vb.kind
                    && visit(va, vb, kb) == Step::Stop
                {
                    stopped = true;
                }
                true
            });
        }
        true
    });
    !stopped
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use fast_mesh_bvh::{NodeId, NodeKind, Quality};
    use fast_mesh_store::{Mesh, TriangleId};

    use super::*;
    use crate::test_util::{build, wavy};

    fn rot_z(deg: f64) -> [[f64; 3]; 3] {
        let (s, c) = deg.to_radians().sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    /// Identity, the 30° rotation about z and a plain translation.
    fn rigids() -> [Rigid; 3] {
        [
            Rigid::IDENTITY,
            Rigid {
                rotation: rot_z(30.0),
                translation: [0.5, -0.25, 0.3],
            },
            Rigid {
                rotation: Rigid::IDENTITY.rotation,
                translation: [1.5, -0.75, 0.5],
            },
        ]
    }

    /// Two small wavy meshes and their BVHs, for both build qualities.
    fn scenes() -> Vec<(Mesh, Bvh, Mesh, Bvh)> {
        [Quality::Sah, Quality::Fast]
            .into_iter()
            .map(|q| {
                let (ma, mb) = (wavy(6), wavy(5));
                let (ba, bb) = (build(&ma, q), build(&mb, q));
                (ma, ba, mb, bb)
            })
            .collect()
    }

    /// Euclidean distance between two boxes, 0 when they overlap.
    fn box_dist(p: &Aabb, q: &Aabb) -> f64 {
        (0..3)
            .map(|k| {
                let gap = f64::from(p.min[k] - q.max[k])
                    .max(f64::from(q.min[k] - p.max[k]))
                    .max(0.0);
                gap * gap
            })
            .sum::<f64>()
            .sqrt()
    }

    fn overlap_key(p: &Aabb, q: &Aabb) -> Option<f64> {
        p.overlaps(q).then_some(0.0)
    }

    /// A leaf, identified by the address of its triangle slice.
    fn leaf_id(view: &NodeView) -> Option<usize> {
        match view.kind {
            NodeKind::Leaf { triangles } => Some(triangles.as_ptr() as usize),
            NodeKind::Inner { .. } => None,
        }
    }

    /// Every leaf of `bvh` with its bounds and triangles.
    fn leaves(bvh: &Bvh) -> Vec<(usize, Aabb, Vec<TriangleId>)> {
        let mut out = Vec::new();
        bvh.traverse(|_, v| {
            if let NodeKind::Leaf { triangles } = v.kind {
                out.push((leaf_id(v).unwrap(), v.bounds, triangles.to_vec()));
            }
            true
        });
        out
    }

    /// The leaf pairs whose boxes satisfy `keep`, by nested loops over all leaves.
    fn reference_leaf_pairs(
        a: &Bvh,
        b: &Bvh,
        xf: &Rigid,
        keep: impl Fn(&Aabb, &Aabb) -> bool,
    ) -> Vec<(usize, usize)> {
        let lb = leaves(b);
        let mut out = Vec::new();
        for (ia, ba, _) in leaves(a) {
            for (ib, bb, _) in &lb {
                if keep(&ba, &xf.transform_box(bb)) {
                    out.push((ia, *ib));
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// The leaf pairs visited by `traverse_pairs_with::<N>` under `key`, sorted,
    /// after checking each is visited once with the key of its boxes.
    fn leaf_pairs_with<const N: usize>(
        a: &Bvh,
        b: &Bvh,
        xf: &Rigid,
        key: impl Fn(&Aabb, &Aabb) -> Option<f64>,
    ) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        traverse_pairs_with::<N, f64>(a, b, xf, &key, |va, vb, k| {
            if let (Some(ia), Some(ib)) = (leaf_id(va), leaf_id(vb)) {
                assert_eq!(Some(k), key(&va.bounds, &xf.transform_box(&vb.bounds)));
                out.push((ia, ib));
                Step::Skip
            } else {
                Step::Descend
            }
        });
        let n = out.len();
        out.sort_unstable();
        out.dedup();
        assert_eq!(out.len(), n, "a leaf pair was visited twice");
        out
    }

    /// Triangle pairs whose tight boxes overlap once B's box is transformed.
    fn triangle_pairs(
        ma: &Mesh,
        mb: &Mesh,
        xf: &Rigid,
        candidates: impl IntoIterator<Item = (TriangleId, TriangleId)>,
    ) -> HashSet<(TriangleId, TriangleId)> {
        let tri_box =
            |m: &Mesh, t: TriangleId| Aabb::of_triangle(m.positions(), m.triangles()[t as usize]);
        candidates
            .into_iter()
            .filter(|&(ta, tb)| tri_box(ma, ta).overlaps(&xf.transform_box(&tri_box(mb, tb))))
            .collect()
    }

    #[test]
    fn all_leaf_pairs_with_overlapping_boxes() {
        for (ma, ba, mb, bb) in scenes() {
            for xf in rigids() {
                let pairs = leaf_pairs_with::<PAIR_STACK>(&ba, &bb, &xf, overlap_key);
                assert_eq!(
                    pairs,
                    reference_leaf_pairs(&ba, &bb, &xf, |p, q| p.overlaps(q))
                );
                assert!(!pairs.is_empty());

                let tris: HashMap<usize, Vec<TriangleId>> = leaves(&ba)
                    .into_iter()
                    .chain(leaves(&bb))
                    .map(|(id, _, t)| (id, t))
                    .collect();
                let from_leaves = pairs.iter().flat_map(|(ia, ib)| {
                    let tb = &tris[ib];
                    tris[ia]
                        .iter()
                        .flat_map(move |&x| tb.iter().map(move |&y| (x, y)))
                });
                let all = (0..ma.triangle_count() as TriangleId)
                    .flat_map(|x| (0..mb.triangle_count() as TriangleId).map(move |y| (x, y)));
                let want = triangle_pairs(&ma, &mb, &xf, all);
                assert!(!want.is_empty());
                assert_eq!(triangle_pairs(&ma, &mb, &xf, from_leaves), want);
            }
        }
    }

    #[test]
    fn stop_ends_traversal() {
        for (_, ba, _, bb) in scenes() {
            for xf in rigids() {
                let (mut calls, mut stopped_at) = (0, None);
                traverse_pairs(&ba, &bb, &xf, overlap_key, |va, vb, _| {
                    calls += 1;
                    if leaf_id(va).is_some() && leaf_id(vb).is_some() {
                        stopped_at.get_or_insert(calls);
                        Step::Stop
                    } else {
                        Step::Descend
                    }
                });
                assert_eq!(stopped_at, Some(calls), "visit ran after Stop");
            }
        }
    }

    /// A node, identified by its children or its triangle slice.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum Id {
        Inner(NodeId, NodeId),
        Leaf(usize),
    }

    fn id_of(view: &NodeView) -> Id {
        match view.kind {
            NodeKind::Inner { left, right } => Id::Inner(left, right),
            NodeKind::Leaf { .. } => Id::Leaf(leaf_id(view).unwrap()),
        }
    }

    /// The ids of an inner node's two children.
    fn children(bvh: &Bvh, id: Id) -> Option<[Id; 2]> {
        match id {
            Id::Inner(l, r) => Some([l, r].map(|c| id_of(&bvh.node(c)))),
            Id::Leaf(_) => None,
        }
    }

    #[test]
    fn nearest_first_order() {
        let mut checked = 0;
        for (_, ba, _, bb) in scenes() {
            for mut xf in rigids() {
                xf.translation[2] += 4.0; // B lies wholly above A.
                let key = |p: &Aabb, q: &Aabb| Some(box_dist(p, q));

                // Every created pair is popped, as the key never culls.
                let mut popped = Vec::new();
                let mut boxes = HashMap::new();
                traverse_pairs(&ba, &bb, &xf, key, |va, vb, k| {
                    let (ia, ib) = (id_of(va), id_of(vb));
                    popped.push((ia, ib, k));
                    boxes.insert((ia, ib), (va.bounds, xf.transform_box(&vb.bounds)));
                    Step::Descend
                });
                let at: HashMap<(Id, Id), usize> = popped
                    .iter()
                    .enumerate()
                    .map(|(i, &(x, y, _))| ((x, y), i))
                    .collect();
                assert_eq!(at.len(), popped.len(), "a pair was popped twice");

                for (i, &(x, y, _)) in popped.iter().enumerate() {
                    let split_a = children(&ba, x).map(|cs| cs.map(|c| at.get(&(c, y)).copied()));
                    let split_b = children(&bb, y).map(|cs| cs.map(|c| at.get(&(x, c)).copied()));
                    let created = match (split_a, split_b) {
                        (None, None) => continue,
                        (Some(ca), None) => ca,
                        (None, Some(cb)) => cb,
                        (Some(ca), Some(cb)) => {
                            let (box_a, box_b) = boxes[&(x, y)];
                            if box_a.surface_area() >= box_b.surface_area() {
                                assert_eq!(cb, [None, None]);
                                ca
                            } else {
                                assert_eq!(ca, [None, None]);
                                cb
                            }
                        }
                    };
                    let [Some(c0), Some(c1)] = created else {
                        panic!("pair {i} was not expanded into two child pairs");
                    };
                    let (first, second) = (c0.min(c1), c0.max(c1));
                    assert_eq!(
                        first,
                        i + 1,
                        "the first child is popped right after its parent"
                    );
                    assert!(popped[first].2 <= popped[second].2, "children out of order");
                    checked += 1;
                }

                // With pruning, as `mesh_distance` will do, the search finds the
                // nearest pair of leaf boxes.
                let mut best = f64::INFINITY;
                traverse_pairs(&ba, &bb, &xf, key, |va, vb, k| {
                    if k > best {
                        Step::Skip
                    } else if leaf_id(va).is_some() && leaf_id(vb).is_some() {
                        best = k;
                        Step::Skip
                    } else {
                        Step::Descend
                    }
                });
                let lb = leaves(&bb);
                let want = leaves(&ba)
                    .iter()
                    .flat_map(|(_, p, _)| {
                        lb.iter().map(|(_, q, _)| box_dist(p, &xf.transform_box(q)))
                    })
                    .fold(f64::INFINITY, f64::min);
                assert!(want > 0.0);
                assert_eq!(best, want);
            }
        }
        assert!(checked > 100);
    }

    #[test]
    fn overflow_fallback_is_complete() {
        let always = |p: &Aabb, q: &Aabb| Some(box_dist(p, q));
        for (_, ba, _, bb) in scenes() {
            for xf in rigids() {
                let want = reference_leaf_pairs(&ba, &bb, &xf, |p, q| p.overlaps(q));
                assert_eq!(leaf_pairs_with::<0>(&ba, &bb, &xf, overlap_key), want);
                assert_eq!(leaf_pairs_with::<1>(&ba, &bb, &xf, overlap_key), want);
                assert_eq!(leaf_pairs_with::<2>(&ba, &bb, &xf, overlap_key), want);
                assert_eq!(leaf_pairs_with::<3>(&ba, &bb, &xf, overlap_key), want);
                assert_eq!(leaf_pairs_with::<4>(&ba, &bb, &xf, overlap_key), want);

                let all = reference_leaf_pairs(&ba, &bb, &xf, |_, _| true);
                assert_eq!(all.len(), leaves(&ba).len() * leaves(&bb).len());
                assert_eq!(leaf_pairs_with::<0>(&ba, &bb, &xf, always), all);
                assert_eq!(leaf_pairs_with::<2>(&ba, &bb, &xf, always), all);
                assert_eq!(leaf_pairs_with::<4>(&ba, &bb, &xf, always), all);
            }
        }
    }

    /// Visits until the `stop_at`-th leaf pair returns `Stop`, and returns the
    /// number of leaf pairs visited.
    fn leaf_visits_until_stop<const N: usize>(
        a: &Bvh,
        b: &Bvh,
        xf: &Rigid,
        stop_at: usize,
    ) -> usize {
        let mut leaf_visits = 0;
        traverse_pairs_with::<N, f64>(a, b, xf, overlap_key, |va, vb, _| {
            if leaf_id(va).is_some() && leaf_id(vb).is_some() {
                leaf_visits += 1;
                if leaf_visits >= stop_at {
                    return Step::Stop;
                }
                return Step::Skip;
            }
            Step::Descend
        });
        leaf_visits
    }

    #[test]
    fn stop_ends_traversal_inside_the_fallback() {
        for (_, ba, _, bb) in scenes() {
            for xf in rigids() {
                for stop_at in [1, 7, 60] {
                    assert_eq!(leaf_visits_until_stop::<0>(&ba, &bb, &xf, stop_at), stop_at);
                    assert_eq!(leaf_visits_until_stop::<1>(&ba, &bb, &xf, stop_at), stop_at);
                    assert_eq!(leaf_visits_until_stop::<3>(&ba, &bb, &xf, stop_at), stop_at);
                }
            }
        }
    }

    #[test]
    fn key_runs_once_per_created_pair() {
        let (_, ba, _, bb) = scenes().remove(0);
        let xf = rigids()[1];
        let count = |n: &mut usize| {
            *n += 1;
            Some(0.0_f64)
        };

        // On the stack, every created pair is popped once, as nothing is culled.
        let (mut keys, mut visits) = (0, 0);
        traverse_pairs(
            &ba,
            &bb,
            &xf,
            |_, _| count(&mut keys),
            |_, _, _| {
                visits += 1;
                Step::Descend
            },
        );
        assert_eq!(keys, visits);

        // Without a stack: the root pair, then each other A node against B's
        // root, then each A leaf against each other B node.
        let mut keys = 0;
        traverse_pairs_with::<0, f64>(
            &ba,
            &bb,
            &xf,
            |_, _| count(&mut keys),
            |_, _, _| Step::Descend,
        );
        let leaves_a = leaves(&ba).len();
        assert_eq!(
            keys,
            1 + (ba.node_count() - 1) + leaves_a * (bb.node_count() - 1)
        );
    }

    #[test]
    fn empty_tree_visits_nothing() {
        let empty = Mesh::new();
        let bempty = build(&empty, Quality::Sah);
        let (_, ba, _, _) = scenes().remove(0);
        for (a, b) in [(&ba, &bempty), (&bempty, &ba), (&bempty, &bempty)] {
            traverse_pairs(
                a,
                b,
                &Rigid::IDENTITY,
                |_, _| panic!("key on an empty tree"),
                |_, _, _: f64| panic!("visit on an empty tree"),
            );
        }
    }
}
