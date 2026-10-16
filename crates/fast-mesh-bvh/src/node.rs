//! Read-only view of the tree, for queries built on top of the BVH.

use fast_mesh_store::TriangleId;

use crate::arena::{NONE, Node};
use crate::{Aabb, Bvh};

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
    /// first. It must be pure: it may be called more than once per node.
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
