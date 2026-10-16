use std::collections::HashMap;

use fast_mesh_store::{Mesh, TriangleId};

use crate::arena::{Arena, NONE, Node};
use crate::build::{Prims, build_subtree, expected_nodes};
use crate::error::{Result, try_reserve, try_reserve_exact, try_reserve_map, try_vec};
use crate::{Aabb, BuildOptions, NodeKind, UpdatePolicy};

/// A subtree that is rebuilt as one unit.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Treelet {
    pub(crate) root: u32,
    /// SAH cost when the treelet was last built.
    pub(crate) build_cost: f32,
}

/// Bounding volume hierarchy over the alive triangles of one [`Mesh`], in
/// mesh-local coordinates. Keep it current with [`Bvh::update`].
pub struct Bvh {
    pub(crate) arena: Arena,
    pub(crate) root: u32,
    pub(crate) treelets: Vec<Treelet>,
    /// Treelet root node → index into `treelets`.
    pub(crate) treelet_of_root: HashMap<u32, usize>,
    /// Nodes and refs no longer reachable from the root.
    pub(crate) garbage_nodes: usize,
    pub(crate) garbage_refs: usize,
    pub(crate) mesh_uid: u64,
    /// `Mesh::epoch` at the last full build; slot ids are stale once it differs.
    pub(crate) epoch: u32,
    pub(crate) revision: u64,
    pub(crate) options: BuildOptions,
    pub(crate) policy: UpdatePolicy,
}

impl std::fmt::Debug for Bvh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bvh")
            .field("nodes", &self.node_count())
            .field("treelets", &self.treelets.len())
            .field("depth", &self.depth())
            .field("bytes", &self.byte_size())
            .finish()
    }
}

/// Collects every alive triangle of `mesh` into `prims`.
pub(crate) fn alive_prims(mesh: &Mesh) -> Result<Prims> {
    let mut prims = Prims::with_capacity(mesh.triangle_count())?;
    let (positions, triangles) = (mesh.positions(), mesh.triangles());
    for (t, &tri) in triangles.iter().enumerate() {
        if mesh.is_triangle_alive(t as TriangleId) {
            prims.push(t as u32, Aabb::of_triangle(positions, tri));
        }
    }
    Ok(prims)
}

impl Bvh {
    /// Builds a BVH over the alive triangles of `mesh`.
    pub fn build(mesh: &Mesh, options: &BuildOptions) -> Result<Bvh> {
        options.validate()?;
        let mut bvh = Self::assemble(mesh, options, &UpdatePolicy::default())?;
        bvh.revision = mesh.revision();
        Ok(bvh)
    }

    /// Replaces the update policy (see [`Bvh::update`]).
    pub fn set_policy(&mut self, policy: UpdatePolicy) -> Result<()> {
        policy.validate()?;
        self.policy = policy;
        Ok(())
    }

    pub fn options(&self) -> &BuildOptions {
        &self.options
    }

    pub fn policy(&self) -> &UpdatePolicy {
        &self.policy
    }

    /// The mesh revision this BVH matches (see [`Mesh::revision`]). Manual
    /// [`Bvh::refit`] and [`Bvh::rebuild_region`] calls do not change it.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// True when the BVH matches the mesh's current revision (and the mesh has
    /// not been compacted since).
    pub fn is_current(&self, mesh: &Mesh) -> bool {
        self.mesh_uid == mesh.uid()
            && self.epoch == mesh.epoch()
            && self.revision == mesh.revision()
    }

    /// Builds a complete new tree. Nothing outside the returned value is touched,
    /// so callers can swap it in only on success.
    pub(crate) fn assemble(
        mesh: &Mesh,
        options: &BuildOptions,
        policy: &UpdatePolicy,
    ) -> Result<Bvh> {
        let mut prims = alive_prims(mesh)?;
        let n = prims.len();
        let mut arena = Arena {
            nodes: Vec::new(),
            parents: Vec::new(),
            refs: Vec::new(),
            tri_leaf: try_vec(mesh.triangle_slots(), NONE)?,
        };
        let expected = expected_nodes(n, options.max_leaf_size);
        try_reserve_exact(&mut arena.nodes, expected)?;
        try_reserve_exact(&mut arena.parents, expected)?;
        try_reserve_exact(&mut arena.refs, n)?;
        let root = if n == 0 {
            NONE
        } else {
            build_subtree(&mut arena, &mut prims, options, NONE)?
        };
        let mut bvh = Bvh {
            arena,
            root,
            treelets: Vec::new(),
            treelet_of_root: HashMap::new(),
            garbage_nodes: 0,
            garbage_refs: 0,
            mesh_uid: mesh.uid(),
            epoch: mesh.epoch(),
            revision: mesh.revision(),
            options: *options,
            policy: *policy,
        };
        bvh.assign_treelets()?;
        Ok(bvh)
    }

    /// Marks the topmost subtrees holding at most `treelet_size` triangles (or
    /// single leaves) as treelets. Requires a tree without garbage.
    fn assign_treelets(&mut self) -> Result<()> {
        self.treelets.clear();
        self.treelet_of_root.clear();
        if self.root == NONE {
            return Ok(());
        }
        // Children always have larger indices than their parents, so a reverse
        // sweep sees children first.
        let nodes = &self.arena.nodes;
        let mut counts = try_vec(nodes.len(), 0u32)?;
        for i in (0..nodes.len()).rev() {
            let n = &nodes[i];
            counts[i] = if n.is_leaf() {
                n.leaf_range().len() as u32
            } else {
                counts[n.a as usize] + counts[n.b as usize]
            };
        }
        let limit = self.options.treelet_size;
        let is_treelet_root = |i: u32, n: &Node| n.is_leaf() || counts[i as usize] <= limit;
        let mut found = 0;
        self.arena.traverse(self.root, |i, n| {
            let root = is_treelet_root(i, n);
            found += usize::from(root);
            !root
        });
        let mut roots = Vec::new();
        try_reserve(&mut roots, found)?;
        try_reserve_map(&mut self.treelet_of_root, found)?;
        try_reserve(&mut self.treelets, found)?;
        self.arena.traverse(self.root, |i, n| {
            let root = is_treelet_root(i, n);
            if root {
                roots.push(i);
            }
            !root
        });
        for root in roots {
            self.treelet_of_root.insert(root, self.treelets.len());
            self.treelets.push(Treelet {
                root,
                build_cost: self.arena.sah_cost(root),
            });
        }
        Ok(())
    }

    /// Treelet containing `node` (a node inside or at the root of a treelet), or
    /// `None` when the walk up reaches the root without meeting a treelet root
    /// (which only happens for nodes outside every treelet).
    pub(crate) fn treelet_of(&self, mut node: u32) -> Option<usize> {
        while node != NONE {
            if let Some(&t) = self.treelet_of_root.get(&node) {
                return Some(t);
            }
            node = self.arena.parents[node as usize];
        }
        None
    }

    /// Appends every alive triangle whose bounding box overlaps the box
    /// `[min, max]` (touching counts), in no particular order.
    ///
    /// The BVH must be current for `mesh` ([`Bvh::is_current`]); a stale tree
    /// may return wrong results or panic.
    pub fn query_aabb(&self, mesh: &Mesh, min: [f32; 3], max: [f32; 3], out: &mut Vec<TriangleId>) {
        debug_assert!(self.is_current(mesh), "query_aabb on a stale BVH");
        let query = Aabb::new(min, max);
        let (positions, triangles) = (mesh.positions(), mesh.triangles());
        self.traverse(|_, node| {
            if !node.bounds.overlaps(&query) {
                return false;
            }
            if let NodeKind::Leaf { triangles: leaf } = node.kind {
                for &t in leaf {
                    if Aabb::of_triangle(positions, triangles[t as usize]).overlaps(&query) {
                        out.push(t);
                    }
                }
            }
            true
        });
    }

    /// Reachable nodes.
    pub fn node_count(&self) -> usize {
        self.arena.nodes.len() - self.garbage_nodes
    }

    pub fn treelet_count(&self) -> usize {
        self.treelets.len()
    }

    /// Bounds of all triangles, or `None` for an empty tree.
    pub fn root_bounds(&self) -> Option<Aabb> {
        (self.root != NONE).then(|| self.arena.nodes[self.root as usize].bounds)
    }

    /// Longest root-to-leaf path, counting nodes (0 for an empty tree).
    pub fn depth(&self) -> usize {
        let mut deepest = 0;
        self.arena.traverse(self.root, |i, n| {
            if n.is_leaf() {
                let mut d = 1;
                let mut cur = i;
                while self.arena.parents[cur as usize] != NONE {
                    cur = self.arena.parents[cur as usize];
                    d += 1;
                }
                deepest = deepest.max(d);
            }
            true
        });
        deepest
    }

    /// Heap memory held by the BVH, garbage included.
    pub fn byte_size(&self) -> usize {
        let a = &self.arena;
        a.nodes.capacity() * size_of::<Node>()
            + (a.parents.capacity() + a.refs.capacity() + a.tri_leaf.capacity()) * 4
            + self.treelets.capacity() * size_of::<Treelet>()
            + self.treelet_of_root.capacity() * 16
    }

    /// Copies the reachable nodes and refs into fresh storage in depth-first
    /// order, dropping garbage. On allocation failure nothing changes.
    pub(crate) fn compact_storage(&mut self) -> Result<()> {
        let live_nodes = self.node_count();
        let live_refs = self.arena.refs.len() - self.garbage_refs;
        let mut new_index = try_vec(self.arena.nodes.len(), NONE)?;
        let mut order = Vec::new();
        try_reserve(&mut order, live_nodes)?;
        let mut nodes = Vec::new();
        try_reserve(&mut nodes, live_nodes)?;
        let mut parents = Vec::new();
        try_reserve(&mut parents, live_nodes)?;
        let mut refs = Vec::new();
        try_reserve(&mut refs, live_refs)?;
        let mut tri_leaf = try_vec(self.arena.tri_leaf.len(), NONE)?;
        let mut roots = HashMap::new();
        try_reserve_map(&mut roots, self.treelets.len())?;

        // Phase 2: nothing below allocates.
        self.arena.traverse(self.root, |i, _| {
            new_index[i as usize] = order.len() as u32;
            order.push(i);
            true
        });
        for &old in &order {
            let n = self.arena.nodes[old as usize];
            let me = nodes.len() as u32;
            let parent = self.arena.parents[old as usize];
            parents.push(if parent == NONE {
                NONE
            } else {
                new_index[parent as usize]
            });
            if n.is_leaf() {
                let first = refs.len() as u32;
                for &t in &self.arena.refs[n.leaf_range()] {
                    refs.push(t);
                    tri_leaf[t as usize] = me;
                }
                nodes.push(Node::leaf(n.bounds, first, n.leaf_range().len() as u32));
            } else {
                nodes.push(Node {
                    bounds: n.bounds,
                    a: new_index[n.a as usize],
                    b: new_index[n.b as usize],
                });
            }
        }
        for (i, t) in self.treelets.iter_mut().enumerate() {
            t.root = new_index[t.root as usize];
            roots.insert(t.root, i);
        }
        if self.root != NONE {
            self.root = new_index[self.root as usize];
        }
        self.arena = Arena {
            nodes,
            parents,
            refs,
            tri_leaf,
        };
        self.treelet_of_root = roots;
        self.garbage_nodes = 0;
        self.garbage_refs = 0;
        Ok(())
    }

    /// Whether `node` is reachable from the root, by walking parent links and
    /// checking that each parent really links to the child.
    fn reachable(&self, mut node: u32) -> bool {
        let a = &self.arena;
        for _ in 0..a.nodes.len() {
            if node == self.root {
                return true;
            }
            let parent = a.parents[node as usize];
            if parent == NONE {
                return false;
            }
            let p = &a.nodes[parent as usize];
            if p.is_leaf() || (p.a != node && p.b != node) {
                return false;
            }
            node = parent;
        }
        false
    }

    /// Checks every structural invariant against `mesh`. Meant for tests and
    /// debugging; returns a description of the first violation.
    pub fn validate(&self, mesh: &Mesh) -> std::result::Result<(), String> {
        let a = &self.arena;
        let (positions, triangles) = (mesh.positions(), mesh.triangles());
        if self.root != NONE && a.parents[self.root as usize] != NONE {
            return Err("root has a parent".into());
        }
        let mut error: Option<String> = None;
        let (mut reachable, mut refs_seen) = (0usize, 0usize);
        a.traverse(self.root, |i, n| {
            if error.is_some() {
                return false;
            }
            reachable += 1;
            if n.is_leaf() {
                for &t in &a.refs[n.leaf_range()] {
                    refs_seen += 1;
                    if !mesh.is_triangle_alive(t) {
                        error = Some(format!("leaf {i} references dead triangle {t}"));
                    } else if a.tri_leaf.get(t as usize) != Some(&i) {
                        error = Some(format!(
                            "tri_leaf of triangle {t} does not point at leaf {i}"
                        ));
                    } else if !n
                        .bounds
                        .contains(&Aabb::of_triangle(positions, triangles[t as usize]))
                    {
                        error = Some(format!("leaf {i} box does not contain triangle {t}"));
                    }
                }
            } else {
                for c in [n.a, n.b] {
                    if a.parents[c as usize] != i {
                        error = Some(format!("child {c} of node {i} has the wrong parent"));
                    } else if !n.bounds.contains(&a.nodes[c as usize].bounds) {
                        error = Some(format!("node {i} box does not contain child {c}"));
                    }
                }
            }
            true
        });
        if let Some(e) = error {
            return Err(e);
        }
        if refs_seen != mesh.triangle_count() {
            return Err(format!(
                "tree holds {refs_seen} triangles, mesh has {}",
                mesh.triangle_count()
            ));
        }
        if reachable + self.garbage_nodes != a.nodes.len() {
            return Err(format!(
                "{reachable} reachable + {} garbage nodes != {} stored",
                self.garbage_nodes,
                a.nodes.len()
            ));
        }
        if refs_seen + self.garbage_refs != a.refs.len() {
            return Err("reachable + garbage refs != stored refs".into());
        }
        // Every triangle → leaf entry must point at a reachable leaf holding it.
        for (t, &leaf) in a.tri_leaf.iter().enumerate() {
            if leaf == NONE {
                continue;
            }
            let holds = a
                .nodes
                .get(leaf as usize)
                .is_some_and(|n| n.is_leaf() && a.refs[n.leaf_range()].contains(&(t as u32)));
            if !holds || !self.reachable(leaf) {
                return Err(format!(
                    "tri_leaf of triangle {t} points at node {leaf}, which is not a reachable leaf holding it"
                ));
            }
        }
        for (i, t) in self.treelets.iter().enumerate() {
            if self.treelet_of_root.get(&t.root) != Some(&i) {
                return Err(format!("treelet {i} is not indexed by its root"));
            }
        }
        if self.root != NONE {
            // Every leaf must sit inside exactly one treelet.
            let mut leaves_in_treelets = 0;
            for t in &self.treelets {
                a.traverse(t.root, |_, n| {
                    if n.is_leaf() {
                        leaves_in_treelets += 1;
                    }
                    true
                });
            }
            let mut leaves = 0;
            a.traverse(self.root, |_, n| {
                if n.is_leaf() {
                    leaves += 1;
                }
                true
            });
            if leaves != leaves_in_treelets {
                return Err(format!(
                    "{leaves} leaves but {leaves_in_treelets} inside treelets"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
impl Bvh {
    /// Test-only: number of triangles in each treelet.
    pub(crate) fn treelet_triangle_counts(&self) -> Vec<usize> {
        self.treelets
            .iter()
            .map(|t| self.arena.subtree_size(t.root).1)
            .collect()
    }

    /// Test-only: everything that defines the tree, to compare before and after
    /// a failed operation. `tri_leaf` entries are compared as (triangle, leaf)
    /// pairs so growing it with `NONE` does not count as a change.
    pub(crate) fn fingerprint(&self) -> String {
        let a = &self.arena;
        let leaves: Vec<(usize, u32)> = a
            .tri_leaf
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l != NONE)
            .map(|(t, &l)| (t, l))
            .collect();
        let mut treelets: Vec<(u32, usize)> =
            self.treelet_of_root.iter().map(|(&r, &i)| (r, i)).collect();
        treelets.sort();
        format!(
            "{:?}|{:?}|{:?}|{:?}|{}|{:?}|{}|{}|{}|{}",
            a.nodes
                .iter()
                .map(|n| (n.bounds, n.a, n.b))
                .collect::<Vec<_>>(),
            a.parents,
            a.refs,
            leaves,
            self.root,
            treelets,
            self.garbage_nodes,
            self.garbage_refs,
            self.epoch,
            self.revision,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BvhError, Quality};

    /// n×n grid of quads in the XY plane.
    pub(crate) fn grid(n: u32) -> Mesh {
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

    fn brute(mesh: &Mesh, q: &Aabb) -> Vec<u32> {
        (0..mesh.triangle_slots() as u32)
            .filter(|&t| {
                mesh.triangle(t)
                    .is_some_and(|tri| Aabb::of_triangle(mesh.positions(), tri).overlaps(q))
            })
            .collect()
    }

    #[test]
    fn builds_and_queries_both_qualities() {
        let mesh = grid(20);
        for quality in [Quality::Sah, Quality::Fast] {
            let opts = BuildOptions {
                quality,
                treelet_size: 64,
                ..Default::default()
            };
            let bvh = Bvh::build(&mesh, &opts).unwrap();
            bvh.validate(&mesh).unwrap();
            assert!(bvh.treelet_count() > 1);
            assert_eq!(
                bvh.root_bounds(),
                Some(Aabb::new([0.0; 3], [20.0, 20.0, 0.0]))
            );
            let q = Aabb::new([3.5, 3.5, -1.0], [7.0, 5.0, 1.0]);
            let mut got = Vec::new();
            bvh.query_aabb(&mesh, q.min, q.max, &mut got);
            got.sort();
            assert_eq!(got, brute(&mesh, &q), "{quality:?}");
        }
    }

    #[test]
    fn empty_mesh_and_single_triangle() {
        let empty = Mesh::new();
        let bvh = Bvh::build(&empty, &BuildOptions::default()).unwrap();
        bvh.validate(&empty).unwrap();
        assert_eq!(
            (bvh.node_count(), bvh.depth(), bvh.root_bounds()),
            (0, 0, None)
        );
        let mut out = Vec::new();
        bvh.query_aabb(&empty, [-1.0; 3], [1.0; 3], &mut out);
        assert!(out.is_empty());

        let one = Mesh::from_buffers(&[0., 0., 0., 1., 0., 0., 0., 1., 0.], &[0, 1, 2]).unwrap();
        let bvh = Bvh::build(&one, &BuildOptions::default()).unwrap();
        bvh.validate(&one).unwrap();
        assert_eq!(
            (bvh.node_count(), bvh.depth(), bvh.treelet_count()),
            (1, 1, 1)
        );
    }

    #[test]
    fn dead_slots_are_skipped() {
        let mut mesh = grid(4);
        let mut tx = mesh.begin_edit();
        tx.remove_triangles(&[0, 5, 9]).unwrap();
        tx.commit().unwrap();
        let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
        bvh.validate(&mesh).unwrap();
        let mut out = Vec::new();
        bvh.query_aabb(&mesh, [-1.0; 3], [10.0; 3], &mut out);
        assert_eq!(out.len(), mesh.triangle_count());
        assert!(!out.contains(&5));
    }

    #[test]
    fn degenerate_and_non_finite_triangles_do_not_break_the_build() {
        let mesh = Mesh::from_buffers(
            &[
                0.,
                0.,
                0.,
                1.,
                0.,
                0.,
                f32::NAN,
                1.,
                0.,
                1e30,
                1e30,
                1e30,
                2.,
                2.,
                2.,
            ],
            &[0, 0, 0, 0, 1, 2, 1, 3, 4, 4, 4, 4, 0, 1, 4],
        )
        .unwrap();
        for quality in [Quality::Sah, Quality::Fast] {
            let opts = BuildOptions {
                quality,
                max_leaf_size: 1,
                ..Default::default()
            };
            let bvh = Bvh::build(&mesh, &opts).unwrap();
            bvh.validate(&mesh).unwrap();
            let mut out = Vec::new();
            let q = Aabb::new([0.5, -1.0, -1.0], [1.5, 1.0, 1.0]);
            bvh.query_aabb(&mesh, q.min, q.max, &mut out);
            out.sort();
            assert_eq!(out, brute(&mesh, &q));
        }
    }

    #[test]
    fn invalid_options_are_rejected() {
        let opts = BuildOptions {
            max_leaf_size: 17,
            ..Default::default()
        };
        assert!(matches!(
            Bvh::build(&grid(1), &opts),
            Err(BvhError::InvalidOptions(_))
        ));
    }

    #[test]
    fn compaction_preserves_the_tree() {
        let mesh = grid(10);
        let mut bvh = Bvh::build(
            &mesh,
            &BuildOptions {
                treelet_size: 16,
                ..Default::default()
            },
        )
        .unwrap();
        let before = bvh.node_count();
        bvh.compact_storage().unwrap();
        bvh.validate(&mesh).unwrap();
        assert_eq!(bvh.node_count(), before);
    }
}
