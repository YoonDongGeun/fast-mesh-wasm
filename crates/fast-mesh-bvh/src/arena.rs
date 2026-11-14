//! Node storage shared by building, updating and querying.
//!
//! No traversal allocates on the heap and none can fail, however deep or
//! unbalanced the tree is. Most walk parent links instead of using a stack; the
//! keyed traversal uses a fixed array on the call stack and falls back to a
//! parent-link walk for any subtree that does not fit.

use crate::Aabb;

trait KeyEval<K> {
    fn eval_one(&mut self, a: &Aabb) -> Option<K>;
    fn eval_pair(&mut self, a: &Aabb, b: &Aabb) -> (Option<K>, Option<K>);
}

struct SingleKey<F>(F);

impl<K, F: FnMut(&Aabb) -> Option<K>> KeyEval<K> for SingleKey<F> {
    fn eval_one(&mut self, a: &Aabb) -> Option<K> {
        (self.0)(a)
    }

    fn eval_pair(&mut self, a: &Aabb, b: &Aabb) -> (Option<K>, Option<K>) {
        ((self.0)(a), (self.0)(b))
    }
}

struct PairKey<F>(F);

impl<K, F: FnMut(&Aabb, &Aabb) -> (Option<K>, Option<K>)> KeyEval<K> for PairKey<F> {
    fn eval_one(&mut self, a: &Aabb) -> Option<K> {
        (self.0)(a, &Aabb::EMPTY).0
    }

    fn eval_pair(&mut self, a: &Aabb, b: &Aabb) -> (Option<K>, Option<K>) {
        (self.0)(a, b)
    }
}

pub(crate) const NONE: u32 = u32::MAX;
pub(crate) const LEAF_BIT: u32 = 1 << 31;

/// 32-byte node. Inner: `a` = left child, `b` = right child.
/// Leaf: `a` = first index into `refs`, `b` = `LEAF_BIT | count`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Node {
    pub(crate) bounds: Aabb,
    pub(crate) a: u32,
    pub(crate) b: u32,
}

const _: () = assert!(std::mem::size_of::<Node>() == 32);

impl Node {
    pub(crate) fn leaf(bounds: Aabb, first: u32, count: u32) -> Self {
        Self {
            bounds,
            a: first,
            b: LEAF_BIT | count,
        }
    }

    pub(crate) fn is_leaf(&self) -> bool {
        self.b & LEAF_BIT != 0
    }

    pub(crate) fn leaf_range(&self) -> std::ops::Range<usize> {
        let first = self.a as usize;
        first..first + (self.b & !LEAF_BIT) as usize
    }
}

/// Nodes, parent links, leaf triangle references and the triangle → leaf map.
pub(crate) struct Arena {
    pub(crate) nodes: Vec<Node>,
    pub(crate) parents: Vec<u32>,
    pub(crate) refs: Vec<u32>,
    /// Indexed by triangle slot: the leaf holding it, or `NONE`.
    pub(crate) tri_leaf: Vec<u32>,
}

impl Arena {
    /// Visits the subtree rooted at `root` in depth-first pre-order. `visit`
    /// returns whether to descend into the node's children (ignored for leaves).
    pub(crate) fn traverse(&self, root: u32, mut visit: impl FnMut(u32, &Node) -> bool) {
        if root == NONE {
            return;
        }
        let stop = self.parents[root as usize];
        let mut cur = root;
        let mut from = stop;
        loop {
            let node = &self.nodes[cur as usize];
            let next;
            if from == self.parents[cur as usize] {
                // First arrival from above.
                if visit(cur, node) && !node.is_leaf() {
                    next = node.a;
                } else {
                    next = self.parents[cur as usize];
                }
            } else if from == node.a {
                next = node.b;
            } else {
                next = self.parents[cur as usize];
            }
            if cur == root && next == stop {
                return;
            }
            from = cur;
            cur = next;
        }
    }

    /// Like [`Arena::traverse`], but at each inner node `first_is_left` decides
    /// from the two child boxes (left, right) which child is visited first.
    /// `first_is_left` must be pure: it is evaluated again on the way back up to
    /// tell which child the walk is returning from.
    pub(crate) fn traverse_ordered(
        &self,
        root: u32,
        first_is_left: impl Fn(&Aabb, &Aabb) -> bool,
        mut visit: impl FnMut(u32, &Node) -> bool,
    ) {
        if root == NONE {
            return;
        }
        let stop = self.parents[root as usize];
        let mut cur = root;
        let mut from = stop;
        loop {
            let node = &self.nodes[cur as usize];
            let parent = self.parents[cur as usize];
            let next;
            if from == parent {
                // First arrival from above.
                if visit(cur, node) && !node.is_leaf() {
                    next = self.first_child(node, &first_is_left).0;
                } else {
                    next = parent;
                }
            } else {
                let (first, second) = self.first_child(node, &first_is_left);
                next = if from == first { second } else { parent };
            }
            if cur == root && next == stop {
                return;
            }
            from = cur;
            cur = next;
        }
    }

    /// Visits the subtree at `root` depth-first, the child with the smaller key
    /// first (left on ties), with a stack of `N` entries on the call stack.
    /// Children whose key is `None` are skipped; `visit` gets each node's key
    /// and returns whether to descend (ignored for leaves). `key` runs once per
    /// node, except inside a subtree that does not fit on the stack: that one is
    /// finished at once by [`Arena::traverse_by_pair_stackless`], in the same
    /// order, where `key` may run again for the same node.
    pub(crate) fn traverse_by_key<const N: usize, K: Copy + PartialOrd>(
        &self,
        root: u32,
        key: impl FnMut(&Aabb) -> Option<K>,
        visit: impl FnMut(u32, &Node, K) -> bool,
    ) {
        self.traverse_by_pair_impl::<N, K, _>(root, SingleKey(key), visit);
    }

    /// Pair-keyed variant of [`Arena::traverse_by_key`] that evaluates both
    /// children of an inner node in a single call, enabling 4-wide / SIMD
    /// culling while preserving identical child ordering and stackless overflow.
    pub(crate) fn traverse_by_pair<const N: usize, K: Copy + PartialOrd>(
        &self,
        root: u32,
        pair_key: impl FnMut(&Aabb, &Aabb) -> (Option<K>, Option<K>),
        visit: impl FnMut(u32, &Node, K) -> bool,
    ) {
        self.traverse_by_pair_impl::<N, K, _>(root, PairKey(pair_key), visit);
    }

    fn traverse_by_pair_impl<const N: usize, K: Copy + PartialOrd, E: KeyEval<K>>(
        &self,
        root: u32,
        mut eval: E,
        mut visit: impl FnMut(u32, &Node, K) -> bool,
    ) {
        if root == NONE {
            return;
        }
        let Some(root_key) = eval.eval_one(&self.nodes[root as usize].bounds) else {
            return;
        };
        if N == 0 {
            self.traverse_by_pair_stackless(root, root_key, &mut eval, visit);
            return;
        }
        let mut stack = [(root, root_key); N];
        let mut len = 1;
        while len > 0 {
            len -= 1;
            let (cur, k) = stack[len];
            let node = &self.nodes[cur as usize];
            if !visit(cur, node, k) || node.is_leaf() {
                continue;
            }
            let [(first, kf), (second, ks)] = self.children_by_pair(node, &mut eval);
            // The second child goes below the first. It always fits, as an
            // entry was just popped.
            if let Some(ks) = ks {
                stack[len] = (second, ks);
                len += 1;
            }
            if let Some(kf) = kf {
                if len < N {
                    stack[len] = (first, kf);
                    len += 1;
                } else {
                    // Finished right away, before the second child is popped:
                    // the order the stack would have given.
                    self.traverse_by_pair_stackless(first, kf, &mut eval, &mut visit);
                }
            }
        }
    }

    /// [`Arena::traverse_by_pair_impl`] without a stack, walking parent links:
    /// the overflow fallback. `root_key` is `root`'s key. Each inner node's
    /// child keys are evaluated again on the way back up, to tell which child
    /// the walk returns from, so keys must be pure.
    fn traverse_by_pair_stackless<K: Copy + PartialOrd, E: KeyEval<K>>(
        &self,
        root: u32,
        root_key: K,
        eval: &mut E,
        mut visit: impl FnMut(u32, &Node, K) -> bool,
    ) {
        let stop = self.parents[root as usize];
        let mut cur = root;
        let mut from = stop;
        // The key of `cur` whenever the walk arrives at it from above.
        let mut k = root_key;
        loop {
            let node = &self.nodes[cur as usize];
            let parent = self.parents[cur as usize];
            let mut next = parent;
            if from == parent {
                // First arrival from above.
                if visit(cur, node, k) && !node.is_leaf() {
                    let [(first, kf), (second, ks)] = self.children_by_pair(node, eval);
                    if let Some(kf) = kf {
                        (next, k) = (first, kf);
                    } else if let Some(ks) = ks {
                        (next, k) = (second, ks);
                    }
                }
            } else {
                let [(first, _), (second, ks)] = self.children_by_pair(node, eval);
                if from == first
                    && let Some(ks) = ks
                {
                    (next, k) = (second, ks);
                }
            }
            if cur == root && next == stop {
                return;
            }
            from = cur;
            cur = next;
        }
    }

    fn children_by_pair<K: Copy + PartialOrd, E: KeyEval<K>>(
        &self,
        node: &Node,
        eval: &mut E,
    ) -> [(u32, Option<K>); 2] {
        let (kl, kr) = eval.eval_pair(
            &self.nodes[node.a as usize].bounds,
            &self.nodes[node.b as usize].bounds,
        );
        if matches!((kl, kr), (Some(l), Some(r)) if r < l) {
            [(node.b, kr), (node.a, kl)]
        } else {
            [(node.a, kl), (node.b, kr)]
        }
    }

    /// Visits every leaf whose node bounds overlap `query`, batching two-child
    /// and four-grandchild tests through [`crate::Aabb4`] in pre-order DFS
    /// without heap allocations.
    pub(crate) fn traverse_aabb_leaves(
        &self,
        root: u32,
        query: &Aabb,
        mut visit_leaf: impl FnMut(&[u32]),
    ) {
        if root == NONE || !self.nodes[root as usize].bounds.overlaps(query) {
            return;
        }
        let mut stack = [0u32; crate::TRAVERSE_STACK];
        stack[0] = root;
        let mut len = 1usize;
        while len > 0 {
            len -= 1;
            let cur = stack[len];
            let node = &self.nodes[cur as usize];
            if node.is_leaf() {
                visit_leaf(&self.refs[node.leaf_range()]);
                continue;
            }
            let a = node.a;
            let b = node.b;
            let na = &self.nodes[a as usize];
            let nb = &self.nodes[b as usize];
            let mask = crate::Aabb4::from_two(&na.bounds, &nb.bounds).overlaps_aabb(query);
            let hit_a = (mask & 1) != 0;
            let hit_b = (mask & 2) != 0;

            if hit_a && hit_b && !na.is_leaf() && !nb.is_leaf() && len + 4 <= crate::TRAVERSE_STACK
            {
                let gmask = crate::Aabb4::from_aabbs([
                    &self.nodes[na.a as usize].bounds,
                    &self.nodes[na.b as usize].bounds,
                    &self.nodes[nb.a as usize].bounds,
                    &self.nodes[nb.b as usize].bounds,
                ])
                .overlaps_aabb(query);
                for (lane, child) in [(3, nb.b), (2, nb.a), (1, na.b), (0, na.a)] {
                    if (gmask & (1 << lane)) != 0 {
                        stack[len] = child;
                        len += 1;
                    }
                }
                continue;
            }

            if hit_b {
                stack[len] = b;
                len += 1;
            }
            if hit_a {
                if len < crate::TRAVERSE_STACK {
                    stack[len] = a;
                    len += 1;
                } else {
                    self.traverse(a, |_, n| {
                        if !n.bounds.overlaps(query) {
                            return false;
                        }
                        if n.is_leaf() {
                            visit_leaf(&self.refs[n.leaf_range()]);
                        }
                        true
                    });
                }
            }
        }
    }

    /// `(first, second)` children of an inner node under `first_is_left`.
    fn first_child(
        &self,
        node: &Node,
        first_is_left: &impl Fn(&Aabb, &Aabb) -> bool,
    ) -> (u32, u32) {
        let (a, b) = (node.a, node.b);
        if first_is_left(
            &self.nodes[a as usize].bounds,
            &self.nodes[b as usize].bounds,
        ) {
            (a, b)
        } else {
            (b, a)
        }
    }

    /// Recomputes a leaf's box from its triangles.
    pub(crate) fn refit_leaf(&mut self, leaf: u32, positions: &[[f32; 3]], triangles: &[[u32; 3]]) {
        let range = self.nodes[leaf as usize].leaf_range();
        let mut b = Aabb::EMPTY;
        for &t in &self.refs[range] {
            b.grow(&Aabb::of_triangle(positions, triangles[t as usize]));
        }
        self.nodes[leaf as usize].bounds = b;
    }

    /// Recomputes the boxes of `node`'s ancestors, stopping early once a box
    /// does not change. Returns how many nodes were rewritten.
    pub(crate) fn refit_ancestors(&mut self, node: u32) -> usize {
        let mut written = 0;
        let mut cur = self.parents[node as usize];
        while cur != NONE {
            let n = self.nodes[cur as usize];
            let b = self.nodes[n.a as usize]
                .bounds
                .union(&self.nodes[n.b as usize].bounds);
            if b == n.bounds {
                break;
            }
            self.nodes[cur as usize].bounds = b;
            written += 1;
            cur = self.parents[cur as usize];
        }
        written
    }

    /// Number of nodes and of triangle references in a subtree.
    pub(crate) fn subtree_size(&self, root: u32) -> (usize, usize) {
        let (mut nodes, mut refs) = (0, 0);
        self.traverse(root, |_, n| {
            nodes += 1;
            if n.is_leaf() {
                refs += n.leaf_range().len();
            }
            true
        });
        (nodes, refs)
    }

    /// Surface-area-heuristic cost of a subtree, relative to its root's area:
    /// inner nodes cost their area, leaves their area times their triangle count.
    pub(crate) fn sah_cost(&self, root: u32) -> f32 {
        let root_area = self.nodes[root as usize].bounds.surface_area();
        let mut cost = 0.0f64;
        self.traverse(root, |_, n| {
            let area = n.bounds.surface_area() as f64;
            cost += if n.is_leaf() {
                area * n.leaf_range().len() as f64
            } else {
                area
            };
            true
        });
        if root_area > 0.0 {
            (cost / root_area as f64) as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// root(0) -> [1, 2], 1 -> [3, 4]; leaves 2, 3, 4 hold triangles 0, 1, 2.
    fn small() -> Arena {
        let leaf = |t: u32, x: f32| Node::leaf(Aabb::new([x, 0.0, 0.0], [x + 1.0, 1.0, 0.0]), t, 1);
        let inner = |a, b| Node {
            bounds: Aabb::EMPTY,
            a,
            b,
        };
        Arena {
            nodes: vec![
                inner(1, 2),
                inner(3, 4),
                leaf(0, 0.0),
                leaf(1, 1.0),
                leaf(2, 2.0),
            ],
            parents: vec![NONE, 0, 0, 1, 1],
            refs: vec![0, 1, 2],
            tri_leaf: vec![2, 3, 4],
        }
    }

    #[test]
    fn traverses_in_pre_order_and_can_prune() {
        let a = small();
        let mut seen = Vec::new();
        a.traverse(0, |i, _| {
            seen.push(i);
            true
        });
        assert_eq!(seen, vec![0, 1, 3, 4, 2]);

        let mut pruned = Vec::new();
        a.traverse(0, |i, _| {
            pruned.push(i);
            i != 1
        });
        assert_eq!(pruned, vec![0, 1, 2]);

        let mut sub = Vec::new();
        a.traverse(1, |i, _| {
            sub.push(i);
            true
        });
        assert_eq!(
            sub,
            vec![1, 3, 4],
            "a subtree traversal stays inside the subtree"
        );
    }

    #[test]
    fn refits_ancestors_and_counts_subtrees() {
        let mut a = small();
        a.refit_ancestors(3);
        assert_eq!(
            a.nodes[1].bounds,
            Aabb::new([1.0, 0.0, 0.0], [3.0, 1.0, 0.0])
        );
        assert_eq!(
            a.nodes[0].bounds,
            Aabb::new([0.0, 0.0, 0.0], [3.0, 1.0, 0.0])
        );
        assert_eq!(a.subtree_size(0), (5, 3));
        assert_eq!(a.subtree_size(1), (3, 2));
    }
}
