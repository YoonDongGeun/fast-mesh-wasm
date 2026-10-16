//! Read-only view of the tree, for queries built on top of the BVH.

use fast_mesh_store::TriangleId;

use crate::arena::{NONE, Node};
use crate::{Aabb, Bvh};

/// How many pending nodes [`Bvh::traverse_by_key`] keeps in its fixed stack.
/// Deeper pending work falls back to a stackless walk.
pub const TRAVERSE_STACK: usize = 64;

/// Identifies a node of one [`Bvh`]. Only valid until that BVH is next
/// modified (updated, refit or rebuilt).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u32);

/// What a node holds.
#[derive(Debug, Clone, Copy)]
pub enum NodeKind<'a> {
    /// An inner node with two children.
    Inner {
        /// The left child.
        left: NodeId,
        /// The right child.
        right: NodeId,
    },
    /// A leaf holding triangles.
    Leaf {
        /// The triangles stored in this leaf.
        triangles: &'a [TriangleId],
    },
}

/// A node's bounds and contents.
#[derive(Debug, Clone, Copy)]
pub struct NodeView<'a> {
    /// Box enclosing everything below the node.
    pub bounds: Aabb,
    /// Children or triangles.
    pub kind: NodeKind<'a>,
}

impl Bvh {
    fn view(&self, node: &Node) -> NodeView<'_> {
        let kind = if node.is_leaf() {
            NodeKind::Leaf {
                triangles: &self.arena.refs[node.leaf_range()],
            }
        } else {
            NodeKind::Inner {
                left: NodeId(node.a),
                right: NodeId(node.b),
            }
        };
        NodeView {
            bounds: node.bounds,
            kind,
        }
    }

    /// The root node, or `None` for an empty tree.
    pub fn root(&self) -> Option<NodeId> {
        (self.root != NONE).then_some(NodeId(self.root))
    }

    /// The bounds and contents of `id`. `id` must come from this BVH and not
    /// predate its last modification; otherwise this may panic or return
    /// unrelated data.
    pub fn node(&self, id: NodeId) -> NodeView<'_> {
        self.view(&self.arena.nodes[id.0 as usize])
    }

    /// The parent of `id`, or `None` for the root.
    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        let p = self.arena.parents[id.0 as usize];
        (p != NONE).then_some(NodeId(p))
    }

    /// Visits the tree in depth-first pre-order (left child first), without a
    /// stack. `visit` returns whether to descend into the node's children
    /// (ignored for leaves).
    pub fn traverse(&self, mut visit: impl FnMut(NodeId, &NodeView) -> bool) {
        self.arena
            .traverse(self.root, |i, n| visit(NodeId(i), &self.view(n)));
    }

    /// Like [`Bvh::traverse`], but at each inner node `first_is_left` gets the
    /// left and right child bounds and says whether the left child is visited
    /// first. It must be pure: it may be called more than once per node, and an
    /// impure predicate (one that gives different answers for the same bounds)
    /// can make the traversal skip subtrees or loop forever. It is still
    /// memory-safe.
    pub fn traverse_ordered(
        &self,
        first_is_left: impl Fn(&Aabb, &Aabb) -> bool,
        mut visit: impl FnMut(NodeId, &NodeView) -> bool,
    ) {
        self.arena
            .traverse_ordered(self.root, first_is_left, |i, n| {
                visit(NodeId(i), &self.view(n))
            });
    }

    /// Visits the tree depth-first, nearest first: of two children, the one
    /// with the smaller key is visited first (the left one on equal keys, or
    /// when the keys do not compare).
    ///
    /// `key` maps a node's bounds to its key; a child whose key is `None` is
    /// skipped together with its subtree, and so is the root. `visit` gets the
    /// node, its view and its key, and returns whether to descend into its
    /// children (ignored for leaves).
    ///
    /// `key` is evaluated once per node, when its parent is expanded (the root
    /// first). The traversal never allocates on the heap and never fails: it
    /// keeps up to [`TRAVERSE_STACK`] pending nodes in a fixed array, and a
    /// subtree that does not fit is finished at once by a parent-link walk, in
    /// the same order, where `key` may be evaluated more than once per node. So
    /// `key` should be pure; an impure one there can skip subtrees or loop
    /// forever, though it stays memory-safe.
    pub fn traverse_by_key<K: Copy + PartialOrd>(
        &self,
        key: impl FnMut(&Aabb) -> Option<K>,
        visit: impl FnMut(NodeId, &NodeView, K) -> bool,
    ) {
        self.traverse_by_key_with::<TRAVERSE_STACK, K>(key, visit);
    }

    /// [`Bvh::traverse_by_key`] with an `N`-entry stack, so tests can force
    /// the overflow fallback.
    pub(crate) fn traverse_by_key_with<const N: usize, K: Copy + PartialOrd>(
        &self,
        key: impl FnMut(&Aabb) -> Option<K>,
        mut visit: impl FnMut(NodeId, &NodeView, K) -> bool,
    ) {
        self.arena
            .traverse_by_key::<N, K>(self.root, key, |i, n, k| visit(NodeId(i), &self.view(n), k));
    }

    /// Visits the subtree rooted at `root` in depth-first pre-order (left child
    /// first), without a stack. `visit` returns whether to descend into the
    /// node's children (ignored for leaves). `root` must come from this BVH and
    /// not predate its last modification; otherwise this may panic or visit
    /// unrelated nodes.
    pub fn traverse_from(&self, root: NodeId, mut visit: impl FnMut(NodeId, &NodeView) -> bool) {
        self.arena
            .traverse(root.0, |i, n| visit(NodeId(i), &self.view(n)));
    }

    /// The uid of the mesh this BVH was built for (see `Mesh::uid`).
    pub fn mesh_uid(&self) -> u64 {
        self.mesh_uid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BuildOptions;
    use fast_mesh_store::Mesh;

    /// n×n grid of quads in the XY plane.
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
                indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        Mesh::from_buffers(&positions, &indices).unwrap()
    }

    fn build(mesh: &Mesh) -> Bvh {
        let opts = BuildOptions {
            max_leaf_size: 2,
            ..Default::default()
        };
        Bvh::build(mesh, &opts).unwrap()
    }

    #[test]
    fn traverse_visits_every_node_once() {
        let bvh = build(&grid(6));
        let mut ids = Vec::new();
        bvh.traverse(|id, _| {
            ids.push(id);
            true
        });
        assert_eq!(ids.len(), bvh.node_count());
        assert_eq!(ids.first().copied(), bvh.root());
        let mut unique = ids.clone();
        unique.sort_by_key(|id| id.0);
        unique.dedup();
        assert_eq!(unique.len(), ids.len());
    }

    #[test]
    fn leaves_hold_every_alive_triangle_once() {
        let mesh = grid(6);
        let bvh = build(&mesh);
        let mut all = Vec::new();
        bvh.traverse(|_, view| {
            if let NodeKind::Leaf { triangles } = view.kind {
                all.extend_from_slice(triangles);
            }
            true
        });
        all.sort();
        let expected: Vec<u32> = (0..mesh.triangle_count() as u32).collect();
        assert_eq!(all, expected);
    }

    #[test]
    fn parent_agrees_with_children() {
        let bvh = build(&grid(6));
        let root = bvh.root().unwrap();
        assert_eq!(bvh.parent(root), None);
        let mut inner = 0;
        bvh.traverse(|id, view| {
            if let NodeKind::Inner { left, right } = view.kind {
                inner += 1;
                assert_eq!(bvh.parent(left), Some(id));
                assert_eq!(bvh.parent(right), Some(id));
            }
            true
        });
        assert!(inner > 0);
    }

    fn pre_order_swapped(bvh: &Bvh, id: NodeId, out: &mut Vec<NodeId>) {
        out.push(id);
        if let NodeKind::Inner { left, right } = bvh.node(id).kind {
            pre_order_swapped(bvh, right, out);
            pre_order_swapped(bvh, left, out);
        }
    }

    #[test]
    fn ordered_traversal_follows_first_is_left() {
        let bvh = build(&grid(6));
        let mut expected = Vec::new();
        pre_order_swapped(&bvh, bvh.root().unwrap(), &mut expected);
        let mut got = Vec::new();
        bvh.traverse_ordered(
            |_, _| false,
            |id, _| {
                got.push(id);
                true
            },
        );
        assert_eq!(got, expected);
        assert_eq!(got.len(), bvh.node_count());
    }

    /// Pre-order following `first_is_left` on the children's bounds.
    fn pre_order_by(
        bvh: &Bvh,
        id: NodeId,
        first_is_left: &impl Fn(&Aabb, &Aabb) -> bool,
        out: &mut Vec<NodeId>,
    ) {
        out.push(id);
        if let NodeKind::Inner { left, right } = bvh.node(id).kind {
            let (l, r) = (bvh.node(left).bounds, bvh.node(right).bounds);
            let (first, second) = if first_is_left(&l, &r) {
                (left, right)
            } else {
                (right, left)
            };
            pre_order_by(bvh, first, first_is_left, out);
            pre_order_by(bvh, second, first_is_left, out);
        }
    }

    #[test]
    fn ordered_traversal_with_mixed_choices() {
        let bvh = build(&grid(6));
        let pred = |l: &Aabb, r: &Aabb| l.min[0] < r.min[0];
        // The predicate must pick left at some nodes and right at others.
        let (mut lefts, mut rights) = (0, 0);
        bvh.traverse(|_, view| {
            if let NodeKind::Inner { left, right } = view.kind {
                if pred(&bvh.node(left).bounds, &bvh.node(right).bounds) {
                    lefts += 1;
                } else {
                    rights += 1;
                }
            }
            true
        });
        assert!(lefts > 0 && rights > 0, "lefts {lefts}, rights {rights}");

        let mut expected = Vec::new();
        pre_order_by(&bvh, bvh.root().unwrap(), &pred, &mut expected);
        let mut got = Vec::new();
        bvh.traverse_ordered(pred, |id, _| {
            got.push(id);
            true
        });
        assert_eq!(got, expected);
        assert_eq!(got.len(), bvh.node_count());
    }

    #[test]
    fn pruning_an_inner_node_skips_only_its_subtree() {
        let bvh = build(&grid(6));
        let root = bvh.root().unwrap();
        let NodeKind::Inner { left, right } = bvh.node(root).kind else {
            panic!("expected an inner root");
        };
        let NodeKind::Inner { .. } = bvh.node(left).kind else {
            panic!("expected an inner left child");
        };
        let mut subtree = Vec::new();
        pre_order_by(&bvh, left, &|_, _| true, &mut subtree);
        assert!(subtree.len() > 1);

        let run = |ordered: bool| {
            let mut got = Vec::new();
            let visit = |id: NodeId, _: &NodeView| {
                got.push(id);
                id != left
            };
            if ordered {
                bvh.traverse_ordered(|_, _| true, visit);
            } else {
                bvh.traverse(visit);
            }
            got
        };
        for got in [run(false), run(true)] {
            assert_eq!(got.len(), bvh.node_count() - (subtree.len() - 1));
            assert!(got.contains(&left) && got.contains(&right));
            for id in &subtree[1..] {
                assert!(!got.contains(id));
            }
            let mut unique = got.clone();
            unique.sort_by_key(|id| id.0);
            unique.dedup();
            assert_eq!(unique.len(), got.len());
        }
    }

    #[test]
    fn single_leaf_tree() {
        let single = Mesh::from_buffers(&[0., 0., 0., 1., 0., 0., 0., 1., 0.], &[0, 1, 2]).unwrap();
        let bvh = build(&single);
        let root = bvh.root().unwrap();
        assert!(matches!(
            bvh.node(root).kind,
            NodeKind::Leaf { triangles } if triangles == [0]
        ));
        assert_eq!(bvh.parent(root), None);
        let mut calls = 0;
        bvh.traverse(|_, _| {
            calls += 1;
            true
        });
        bvh.traverse_ordered(
            |_, _| true,
            |_, _| {
                calls += 1;
                true
            },
        );
        assert_eq!(calls, 2);
    }

    #[test]
    fn ordered_traversal_prunes() {
        let bvh = build(&grid(6));
        let mut visited = 0;
        bvh.traverse_ordered(
            |_, _| true,
            |_, _| {
                visited += 1;
                false
            },
        );
        assert_eq!(visited, 1);
    }

    /// Reference for [`Bvh::traverse_by_key`]: recursive pre-order that visits
    /// the child with the smaller key first (left on ties), skips children
    /// whose key is `None`, and descends below `id` only if `descend(id)`.
    fn by_key_ref<K: Copy + PartialOrd>(
        bvh: &Bvh,
        id: NodeId,
        k: K,
        key: &impl Fn(&Aabb) -> Option<K>,
        descend: &impl Fn(NodeId) -> bool,
        out: &mut Vec<(NodeId, K)>,
    ) {
        out.push((id, k));
        if !descend(id) {
            return;
        }
        if let NodeKind::Inner { left, right } = bvh.node(id).kind {
            let kl = key(&bvh.node(left).bounds);
            let kr = key(&bvh.node(right).bounds);
            let order = if matches!((kl, kr), (Some(l), Some(r)) if r < l) {
                [(right, kr), (left, kl)]
            } else {
                [(left, kl), (right, kr)]
            };
            for (c, kc) in order {
                if let Some(kc) = kc {
                    by_key_ref(bvh, c, kc, key, descend, out);
                }
            }
        }
    }

    fn reference<K: Copy + PartialOrd>(
        bvh: &Bvh,
        key: &impl Fn(&Aabb) -> Option<K>,
        descend: &impl Fn(NodeId) -> bool,
    ) -> Vec<(NodeId, K)> {
        let mut out = Vec::new();
        let root = bvh.root().unwrap();
        if let Some(k) = key(&bvh.node(root).bounds) {
            by_key_ref(bvh, root, k, key, descend, &mut out);
        }
        out
    }

    /// Runs the keyed traversal with an `N`-entry stack. Returns the visits and
    /// how many times `key` was called.
    fn run_by_key<const N: usize, K: Copy + PartialOrd + std::fmt::Debug>(
        bvh: &Bvh,
        key: &impl Fn(&Aabb) -> Option<K>,
        descend: &impl Fn(NodeId) -> bool,
    ) -> (Vec<(NodeId, K)>, usize) {
        let mut got = Vec::new();
        let mut calls = 0;
        bvh.traverse_by_key_with::<N, K>(
            |b| {
                calls += 1;
                key(b)
            },
            |id, view, k| {
                assert_eq!(key(&view.bounds), Some(k));
                got.push((id, k));
                descend(id)
            },
        );
        (got, calls)
    }

    /// A distance-like key from a point inside the grid, so that some nodes
    /// visit their left child first and others their right one.
    fn slanted(b: &Aabb) -> Option<f32> {
        Some((b.min[0] - 3.1).abs() + 2.0 * (b.max[1] - 2.6).abs())
    }

    #[test]
    fn by_key_matches_recursive_reference() {
        let bvh = build(&grid(6));
        let all = |_: NodeId| true;
        let expected = reference(&bvh, &slanted, &all);
        assert_eq!(expected.len(), bvh.node_count());
        // The key must pick the left child at some nodes and the right one at
        // others, or the test proves nothing.
        let (mut lefts, mut rights) = (0, 0);
        bvh.traverse(|_, view| {
            if let NodeKind::Inner { left, right } = view.kind {
                if slanted(&bvh.node(right).bounds) < slanted(&bvh.node(left).bounds) {
                    rights += 1;
                } else {
                    lefts += 1;
                }
            }
            true
        });
        assert!(lefts > 0 && rights > 0, "lefts {lefts}, rights {rights}");

        let mut got = Vec::new();
        bvh.traverse_by_key(slanted, |id, _, k| {
            got.push((id, k));
            true
        });
        assert_eq!(got, expected);
        let mut unique: Vec<u32> = got.iter().map(|(id, _)| id.0).collect();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), bvh.node_count());
    }

    /// An inner node other than the root whose bounds no other node shares.
    fn unique_inner_non_root(bvh: &Bvh) -> NodeId {
        let mut all = Vec::new();
        bvh.traverse(|id, view| {
            all.push((id, view.bounds, matches!(view.kind, NodeKind::Inner { .. })));
            true
        });
        all.iter()
            .find(|&&(id, b, inner)| {
                inner
                    && Some(id) != bvh.root()
                    && all.iter().filter(|&&(_, o, _)| o == b).count() == 1
            })
            .expect("no inner node with unique bounds")
            .0
    }

    fn subtree(bvh: &Bvh, id: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        pre_order_by(bvh, id, &|_, _| true, &mut out);
        out
    }

    #[test]
    fn by_key_none_culls_subtree() {
        let bvh = build(&grid(6));
        let culled = unique_inner_non_root(&bvh);
        let hidden = subtree(&bvh, culled);
        assert!(hidden.len() > 1);
        let cull_bounds = bvh.node(culled).bounds;
        let key = |b: &Aabb| if *b == cull_bounds { None } else { slanted(b) };
        let all = |_: NodeId| true;
        let expected = reference(&bvh, &key, &all);
        assert_eq!(expected.len(), bvh.node_count() - hidden.len());

        let mut got = Vec::new();
        bvh.traverse_by_key(key, |id, _, k| {
            got.push((id, k));
            true
        });
        assert_eq!(got, expected);
        for id in &hidden {
            assert!(!got.iter().any(|(g, _)| g == id));
        }
    }

    #[test]
    fn key_evaluated_once_per_node() {
        let bvh = build(&grid(6));
        let mut calls = 0;
        let mut visits = 0;
        bvh.traverse_by_key(
            |b| {
                calls += 1;
                slanted(b)
            },
            |_, _, _| {
                visits += 1;
                true
            },
        );
        assert_eq!(visits, bvh.node_count());
        assert_eq!(calls, bvh.node_count());
    }

    #[test]
    fn small_stack_overflow_falls_back() {
        // SAH and Morton trees stay far shallower than `TRAVERSE_STACK`, so the
        // fallback is forced with a tiny stack instead.
        let bvh = build(&grid(8));
        assert!(bvh.depth() > 4);
        let mut plain = Vec::new();
        bvh.traverse(|id, _| {
            plain.push(id.0);
            true
        });
        plain.sort();

        let culled = unique_inner_non_root(&bvh);
        let cull_bounds = bvh.node(culled).bounds;
        let culling = |b: &Aabb| if *b == cull_bounds { None } else { slanted(b) };

        let all = |_: NodeId| true;
        let prune_one = |id: NodeId| id != culled;
        let prune_root = |_: NodeId| false;
        type Key<'a> = &'a dyn Fn(&Aabb) -> Option<f32>;
        type Descend<'a> = &'a dyn Fn(NodeId) -> bool;
        let cases: [(Key, Descend); 4] = [
            (&slanted, &all),
            (&slanted, &prune_one),
            (&culling, &all),
            (&slanted, &prune_root),
        ];
        for (key, descend) in cases {
            let expected = reference(&bvh, &key, &descend);
            let runs = [
                run_by_key::<0, f32>(&bvh, &key, &descend),
                run_by_key::<1, f32>(&bvh, &key, &descend),
                run_by_key::<2, f32>(&bvh, &key, &descend),
                run_by_key::<3, f32>(&bvh, &key, &descend),
                run_by_key::<4, f32>(&bvh, &key, &descend),
                run_by_key::<TRAVERSE_STACK, f32>(&bvh, &key, &descend),
            ];
            for (got, _) in &runs {
                // Nearest first everywhere, the fallback subtrees included.
                assert_eq!(*got, expected);
            }
            // The full stack never overflows here, so no key is evaluated
            // twice.
            assert!(runs[5].1 <= bvh.node_count());
        }

        let (got, calls) = run_by_key::<2, f32>(&bvh, &slanted, &all);
        assert!(calls > bvh.node_count(), "the fallback never ran");
        let mut ids: Vec<u32> = got.iter().map(|(id, _)| id.0).collect();
        ids.sort();
        assert_eq!(ids, plain);

        let (got, _) = run_by_key::<2, f32>(&bvh, &slanted, &prune_root);
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn traverse_from_visits_subtree_only() {
        let bvh = build(&grid(6));
        let root = bvh.root().unwrap();
        let NodeKind::Inner { left, .. } = bvh.node(root).kind else {
            panic!("expected an inner root");
        };
        for id in [left, root] {
            let expected = subtree(&bvh, id);
            let mut got = Vec::new();
            bvh.traverse_from(id, |i, _| {
                got.push(i);
                true
            });
            assert_eq!(got, expected);
        }
        let mut got = Vec::new();
        bvh.traverse_from(left, |i, _| {
            got.push(i);
            false
        });
        assert_eq!(got, [left]);
    }

    #[test]
    fn by_key_on_empty_and_single_leaf_trees() {
        let empty = build(&Mesh::new());
        let (mut keys, mut visits) = (0, 0);
        empty.traverse_by_key(
            |_| {
                keys += 1;
                Some(0.0)
            },
            |_, _, _| {
                visits += 1;
                true
            },
        );
        assert_eq!((keys, visits), (0, 0));

        let single = Mesh::from_buffers(&[0., 0., 0., 1., 0., 0., 0., 1., 0.], &[0, 1, 2]).unwrap();
        let bvh = build(&single);
        let mut visits = 0;
        let root_key = slanted(&bvh.root_bounds().unwrap());
        bvh.traverse_by_key(slanted, |_, _, k| {
            assert_eq!(Some(k), root_key);
            visits += 1;
            true
        });
        assert_eq!(visits, 1);
        bvh.traverse_by_key(|_| None::<f32>, |_, _, _| panic!("culled root visited"));
    }

    #[test]
    fn empty_tree_has_no_root() {
        let mesh = Mesh::new();
        let bvh = build(&mesh);
        assert_eq!(bvh.root(), None);
        let mut calls = 0;
        bvh.traverse(|_, _| {
            calls += 1;
            true
        });
        bvh.traverse_ordered(
            |_, _| true,
            |_, _| {
                calls += 1;
                true
            },
        );
        assert_eq!(calls, 0);
    }
}
