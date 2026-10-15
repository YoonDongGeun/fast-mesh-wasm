//! Incremental updates driven by the store's `ChangeSet`.
//!
//! Every fallible allocation happens in a preparation phase before the tree is
//! touched; the mutation phase cannot fail. Two follow-up steps — rebuilding
//! degraded treelets and compacting storage — are optimizations: if memory runs
//! out there they are skipped and the tree stays correct.

use fast_mesh_store::{ChangeSet, Mesh, TriangleId};

use crate::Aabb;
use crate::arena::{NONE, Node};
use crate::build::{Prims, build_subtree, max_nodes, reserve_headroom};
use crate::error::{BvhError, Result, try_reserve, try_reserve_map, try_vec};
use crate::tree::{Bvh, Treelet};

/// How `update` brought the tree up to date.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Nothing relevant changed.
    Unchanged,
    /// Only boxes were recomputed.
    Refit,
    /// Some treelets were rebuilt (and boxes refit).
    Partial,
    /// The whole tree was rebuilt.
    Full,
}

/// What `update` did, for diagnostics and benchmarks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpdateReport {
    pub strategy: Strategy,
    /// Inner node boxes rewritten while propagating refits.
    pub refit_nodes: usize,
    /// Treelets rebuilt (topology changes plus degradation).
    pub rebuilt_treelets: usize,
    /// Whether garbage was compacted away.
    pub compacted: bool,
}

impl UpdateReport {
    fn new(strategy: Strategy) -> Self {
        Self {
            strategy,
            refit_nodes: 0,
            rebuilt_treelets: 0,
            compacted: false,
        }
    }
}

/// Sorted union of two sorted, deduplicated id lists.
fn merge(a: &[u32], b: &[u32]) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    try_reserve(&mut out, a.len() + b.len())?;
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

impl Bvh {
    /// Brings the BVH up to date with `mesh`, given the [`ChangeSet`]s of every
    /// commit, undo and redo since the BVH last matched (merge them when there are
    /// several). Builds the mesh's incidence index when vertices moved.
    ///
    /// Chooses the cheapest correct strategy: refit for moves, treelet rebuilds for
    /// local topology changes, a full rebuild when more than
    /// `UpdatePolicy::full_rebuild_fraction` of the triangles are dirty. On error the
    /// BVH is unchanged.
    ///
    /// When the BVH already matches the mesh's revision, `changes` is ignored and
    /// the result is [`Strategy::Unchanged`], so applying a change set twice is
    /// harmless. When the mesh was compacted since the last full build
    /// ([`Mesh::epoch`] changed), every slot id is stale and the tree is rebuilt
    /// from scratch ([`Strategy::Full`]) whatever `changes` says.
    pub fn update(&mut self, mesh: &mut Mesh, changes: &ChangeSet) -> Result<UpdateReport> {
        if mesh.uid() != self.mesh_uid {
            return Err(BvhError::MeshMismatch);
        }
        if mesh.epoch() != self.epoch {
            *self = Bvh::assemble(mesh, &self.options, &self.policy)?;
            return Ok(UpdateReport::new(Strategy::Full));
        }
        if mesh.revision() == self.revision {
            return Ok(UpdateReport::new(Strategy::Unchanged));
        }
        if !changes.vertices_moved.is_empty() {
            mesh.ensure_incidence()?;
        }
        let mesh = &*mesh;
        let removed = merge(&changes.triangles_removed, &changes.triangles_changed)?;
        let added = merge(&changes.triangles_added, &changes.triangles_changed)?;
        let mut moved = Vec::new();
        if let Some(inc) = mesh
            .incidence()
            .filter(|_| !changes.vertices_moved.is_empty())
        {
            let total: usize = changes
                .vertices_moved
                .iter()
                .map(|&v| inc.triangles(v).len())
                .sum();
            try_reserve(&mut moved, total)?;
            for &v in &changes.vertices_moved {
                moved.extend_from_slice(inc.triangles(v));
            }
            moved.sort_unstable();
            moved.dedup();
            // Triangles that are rebuilt anyway need no refit.
            moved.retain(|t| removed.binary_search(t).is_err() && added.binary_search(t).is_err());
        }

        let dirty = removed.len().max(added.len()) + moved.len();
        if removed.is_empty() && added.is_empty() && moved.is_empty() {
            self.revision = mesh.revision();
            return Ok(UpdateReport::new(Strategy::Unchanged));
        }
        let alive = mesh.triangle_count().max(1);
        if self.root == NONE || dirty as f32 > self.policy.full_rebuild_fraction * alive as f32 {
            *self = Bvh::assemble(mesh, &self.options, &self.policy)?;
            return Ok(UpdateReport::new(Strategy::Full));
        }

        let mut report = UpdateReport::new(if removed.is_empty() && added.is_empty() {
            Strategy::Refit
        } else {
            Strategy::Partial
        });
        // Leaves of moved triangles, gathered after the rebuilds (which move
        // triangles to new leaves); reserved now so the refit cannot fail.
        let mut leaves = Vec::new();
        try_reserve(&mut leaves, moved.len())?;
        let mut touched = self.rebuild_treelets(mesh, &removed, &added)?;
        report.rebuilt_treelets = touched.rebuilt;
        report.refit_nodes += touched.refit_nodes;

        // Refit each leaf of a moved triangle once and propagate upwards.
        let (positions, triangles) = (mesh.positions(), mesh.triangles());
        for &t in &moved {
            match self.arena.tri_leaf.get(t as usize) {
                Some(&leaf) if leaf != NONE => leaves.push(leaf),
                _ => {}
            }
        }
        leaves.sort_unstable();
        leaves.dedup();
        for &leaf in &leaves {
            self.arena.refit_leaf(leaf, positions, triangles);
            report.refit_nodes += self.arena.refit_ancestors(leaf);
            if let Some(i) = self.treelet_of(leaf) {
                touched.mark(i);
            }
        }

        // Best effort: rebuild treelets whose quality degraded. Treelets split
        // off during this update (indices past `touched`) were just built.
        for i in 0..touched.refit.len() {
            if !touched.refit[i] || touched.rebuilt_now[i] {
                continue;
            }
            let t = self.treelets[i];
            if self.arena.sah_cost(t.root) > self.policy.degradation_factor * t.build_cost
                && self.rebuild_one(mesh, i).is_ok()
            {
                report.rebuilt_treelets += 1;
                report.strategy = Strategy::Partial;
            }
        }

        // Best effort: drop garbage once it exceeds half of the live tree.
        let live_refs = self.arena.refs.len() - self.garbage_refs;
        if (2 * self.garbage_nodes > self.node_count() || 2 * self.garbage_refs > live_refs)
            && self.compact_storage().is_ok()
        {
            report.compacted = true;
        }
        self.revision = mesh.revision();
        Ok(report)
    }

    /// Recomputes the boxes of the leaves holding `triangles` and their ancestors.
    ///
    /// If the mesh was compacted since the last full build ([`Mesh::epoch`]
    /// changed), slot ids are stale, so this rebuilds the whole tree instead (like
    /// [`Bvh::rebuild`], which also records the mesh revision). On error the BVH
    /// is unchanged.
    pub fn refit(&mut self, mesh: &Mesh, triangles: &[TriangleId]) -> Result<()> {
        if mesh.uid() != self.mesh_uid {
            return Err(BvhError::MeshMismatch);
        }
        if mesh.epoch() != self.epoch {
            return self.rebuild(mesh);
        }
        for &t in triangles {
            let leaf = self.arena.tri_leaf.get(t as usize).copied().unwrap_or(NONE);
            if leaf != NONE {
                self.arena
                    .refit_leaf(leaf, mesh.positions(), mesh.triangles());
                self.arena.refit_ancestors(leaf);
            }
        }
        Ok(())
    }

    /// Rebuilds the treelets containing `triangles` from their current contents
    /// (triangles that died since are dropped).
    ///
    /// If the mesh was compacted since the last full build ([`Mesh::epoch`]
    /// changed), slot ids are stale, so this rebuilds the whole tree instead (like
    /// [`Bvh::rebuild`], which also records the mesh revision). Memory for every
    /// affected treelet is reserved before the first one is rebuilt, so on error
    /// the BVH is unchanged.
    pub fn rebuild_region(&mut self, mesh: &Mesh, triangles: &[TriangleId]) -> Result<()> {
        if mesh.uid() != self.mesh_uid {
            return Err(BvhError::MeshMismatch);
        }
        if mesh.epoch() != self.epoch {
            return self.rebuild(mesh);
        }
        let mut which = Vec::new();
        try_reserve(&mut which, triangles.len())?;
        for &t in triangles {
            if let Some(i) = self.treelet_of_triangle(t) {
                which.push(i);
            }
        }
        which.sort_unstable();
        which.dedup();
        let mut budget = Budget::default();
        for &i in &which {
            budget.add(
                self.arena.subtree_size(self.treelets[i].root).1,
                self.options.treelet_size,
            );
        }
        let mut scratch = self.reserve_rebuilds(&budget)?;
        // Mutation phase: nothing below can fail.
        for &i in &which {
            self.replace_treelet(mesh, i, &[], &[], &[], &mut scratch);
        }
        Ok(())
    }

    /// Rebuilds the whole tree from `mesh`. On error the BVH is unchanged.
    pub fn rebuild(&mut self, mesh: &Mesh) -> Result<()> {
        if mesh.uid() != self.mesh_uid {
            return Err(BvhError::MeshMismatch);
        }
        *self = Bvh::assemble(mesh, &self.options, &self.policy)?;
        Ok(())
    }

    /// Removes `removed` from the tree and inserts `added`, rebuilding each
    /// affected treelet. All allocation happens before the first write.
    fn rebuild_treelets(&mut self, mesh: &Mesh, removed: &[u32], added: &[u32]) -> Result<Touched> {
        let mut touched = Touched::new(self.treelets.len())?;
        if removed.is_empty() && added.is_empty() {
            return Ok(touched);
        }
        // tri_leaf must cover new slots; growing it with NONE changes nothing else.
        if self.arena.tri_leaf.len() < mesh.triangle_slots() {
            let extra = mesh.triangle_slots() - self.arena.tri_leaf.len();
            reserve_headroom(&mut self.arena.tri_leaf, extra)?;
            self.arena.tri_leaf.resize(mesh.triangle_slots(), NONE);
        }
        let positions = mesh.positions();
        let triangles = mesh.triangles();
        // Which treelet each removal and insertion belongs to. An added triangle
        // that is already in the tree (a change set applied twice, or merged sets
        // re-listing an addition) is also treated as removed from where it is, so
        // it can never be gathered twice.
        let mut queued: Vec<Vec<u32>> = Vec::new();
        try_reserve(&mut queued, self.treelets.len())?;
        queued.resize_with(self.treelets.len(), Vec::new);
        for &t in removed.iter().chain(added) {
            if let Some(i) = self.treelet_of_triangle(t) {
                touched.rebuild[i] = true;
            }
        }
        for &t in added {
            if !mesh.is_triangle_alive(t) {
                continue;
            }
            let i = self.choose_treelet(&Aabb::of_triangle(positions, triangles[t as usize]));
            touched.rebuild[i] = true;
            try_reserve(&mut queued[i], 1)?;
            queued[i].push(t);
        }
        // Reserve for the largest rebuild and for all appended nodes and refs.
        let mut budget = Budget::default();
        for (i, q) in queued.iter().enumerate() {
            if touched.rebuild[i] {
                let (_, refs) = self.arena.subtree_size(self.treelets[i].root);
                budget.add(refs + q.len(), self.options.treelet_size);
            }
        }
        let mut scratch = self.reserve_rebuilds(&budget)?;

        // Mutation phase: nothing below can fail. Clear removed and re-added
        // triangles first so a rebuilt treelet never overwrites the leaf of a
        // triangle inserted elsewhere.
        for &t in removed.iter().chain(added) {
            if let Some(leaf) = self.arena.tri_leaf.get_mut(t as usize) {
                *leaf = NONE;
            }
        }
        for (i, q) in queued.iter().enumerate() {
            if touched.rebuild[i] {
                touched.refit_nodes +=
                    self.replace_treelet(mesh, i, removed, added, q, &mut scratch);
                touched.rebuilt += 1;
                touched.rebuilt_now[i] = true;
            }
        }
        Ok(touched)
    }

    /// Rebuilds one treelet with its own allocation (used for degradation). On
    /// error the tree is unchanged.
    fn rebuild_one(&mut self, mesh: &Mesh, i: usize) -> Result<()> {
        let mut budget = Budget::default();
        budget.add(
            self.arena.subtree_size(self.treelets[i].root).1,
            self.options.treelet_size,
        );
        let mut scratch = self.reserve_rebuilds(&budget)?;
        self.replace_treelet(mesh, i, &[], &[], &[], &mut scratch);
        Ok(())
    }

    /// Treelet holding triangle `t`, if the tree holds it.
    fn treelet_of_triangle(&self, t: TriangleId) -> Option<usize> {
        match self.arena.tri_leaf.get(t as usize) {
            Some(&leaf) if leaf != NONE => self.treelet_of(leaf),
            _ => None,
        }
    }

    /// Reserves everything a batch of treelet rebuilds described by `budget`
    /// needs, so that [`Bvh::replace_treelet`] cannot fail. Only capacity
    /// changes, so on error the tree is unchanged.
    fn reserve_rebuilds(&mut self, budget: &Budget) -> Result<Scratch> {
        let prims = Prims::with_capacity(budget.max_n)?;
        let counts = if budget.extra_treelets > 0 {
            try_vec(max_nodes(budget.max_n), 0u32)?
        } else {
            Vec::new()
        };
        reserve_headroom(&mut self.arena.nodes, budget.nodes)?;
        reserve_headroom(&mut self.arena.parents, budget.nodes)?;
        reserve_headroom(&mut self.arena.refs, budget.refs)?;
        if budget.extra_treelets > 0 {
            try_reserve(&mut self.treelets, budget.extra_treelets)?;
        }
        // Every rebuild inserts its new root; splits insert their extra roots.
        try_reserve_map(
            &mut self.treelet_of_root,
            budget.rebuilds + budget.extra_treelets,
        )?;
        Ok(Scratch { prims, counts })
    }

    /// Builds a new subtree for treelet `i` from its triangles that are still
    /// alive and in neither `removed` nor `added`, plus `insert`, links it in
    /// place of the old one and refits the ancestors. A result holding more than
    /// twice `treelet_size` triangles is split into several treelets. Capacity
    /// must already be reserved with [`Bvh::reserve_rebuilds`]. Dropped
    /// triangles whose `tri_leaf` still points into the old treelet get `NONE`.
    /// Returns how many ancestor boxes changed.
    fn replace_treelet(
        &mut self,
        mesh: &Mesh,
        i: usize,
        removed: &[u32],
        added: &[u32],
        insert: &[u32],
        scratch: &mut Scratch,
    ) -> usize {
        let old_root = self.treelets[i].root;
        let (positions, triangles) = (mesh.positions(), mesh.triangles());
        let prims = &mut scratch.prims;
        prims.clear();
        let (mut old_nodes, mut old_refs) = (0usize, 0usize);
        // Taken out (without allocating) so the traversal can clear entries.
        let mut tri_leaf = std::mem::take(&mut self.arena.tri_leaf);
        let arena = &self.arena;
        arena.traverse(old_root, |node, n| {
            old_nodes += 1;
            if n.is_leaf() {
                old_refs += n.leaf_range().len();
                for &t in &arena.refs[n.leaf_range()] {
                    if mesh.is_triangle_alive(t)
                        && removed.binary_search(&t).is_err()
                        && added.binary_search(&t).is_err()
                    {
                        prims.push(t, Aabb::of_triangle(positions, triangles[t as usize]));
                    } else if tri_leaf[t as usize] == node {
                        tri_leaf[t as usize] = NONE;
                    }
                }
            }
            true
        });
        self.arena.tri_leaf = tri_leaf;
        for &t in insert {
            prims.push(t, Aabb::of_triangle(positions, triangles[t as usize]));
        }
        let parent = self.arena.parents[old_root as usize];
        let new_root = build_subtree(&mut self.arena, prims, &self.options, parent)
            .expect("capacity for the rebuild was reserved before mutating");
        if parent == NONE {
            self.root = new_root;
        } else {
            let p = &mut self.arena.nodes[parent as usize];
            if p.a == old_root {
                p.a = new_root;
            } else {
                p.b = new_root;
            }
        }
        self.garbage_nodes += old_nodes;
        self.garbage_refs += old_refs;
        self.treelet_of_root.remove(&old_root);
        self.treelet_of_root.insert(new_root, i);
        self.treelets[i] = Treelet {
            root: new_root,
            build_cost: self.arena.sah_cost(new_root),
        };
        if prims.len() > 2 * self.options.treelet_size as usize {
            self.split_treelet(i, &mut scratch.counts);
        }
        self.arena.refit_ancestors(new_root)
    }

    /// Splits treelet `i`, which must be the subtree built last (its nodes are
    /// the tail of the arena, children after parents), by the rule
    /// `assign_treelets` uses: the topmost subtrees with at most `treelet_size`
    /// triangles, or single leaves. Best effort and allocation-free: skipped when
    /// `counts` or the reserved treelet capacity is too small.
    fn split_treelet(&mut self, i: usize, counts: &mut [u32]) {
        let root = self.treelets[i].root;
        let first = root as usize;
        let nodes = &self.arena.nodes;
        if nodes.len() - first > counts.len() {
            return;
        }
        for k in (first..nodes.len()).rev() {
            let n = &nodes[k];
            counts[k - first] = if n.is_leaf() {
                n.leaf_range().len() as u32
            } else {
                counts[n.a as usize - first] + counts[n.b as usize - first]
            };
        }
        let limit = self.options.treelet_size;
        let counts = &*counts;
        let is_root = |k: u32, n: &Node| n.is_leaf() || counts[k as usize - first] <= limit;
        let mut found = 0usize;
        self.arena.traverse(root, |k, n| {
            let r = is_root(k, n);
            found += usize::from(r);
            !r
        });
        let extra = found.saturating_sub(1);
        // The map needs room for all `found` inserts: removing the old root
        // may leave a tombstone that does not free its room.
        if extra == 0
            || self.treelets.capacity() - self.treelets.len() < extra
            || self.treelet_of_root.capacity() - self.treelet_of_root.len() < found
        {
            return;
        }
        let arena = &self.arena;
        let treelets = &mut self.treelets;
        let map = &mut self.treelet_of_root;
        map.remove(&root);
        let mut slot = Some(i);
        arena.traverse(root, |k, n| {
            let r = is_root(k, n);
            if r {
                let t = Treelet {
                    root: k,
                    build_cost: arena.sah_cost(k),
                };
                if let Some(i) = slot.take() {
                    treelets[i] = t;
                    map.insert(k, i);
                } else {
                    map.insert(k, treelets.len());
                    treelets.push(t);
                }
            }
            !r
        });
    }

    /// Descends the top tree, at each node taking the child whose box grows least,
    /// until reaching a treelet root.
    fn choose_treelet(&self, b: &Aabb) -> usize {
        let mut node = self.root;
        loop {
            if let Some(&i) = self.treelet_of_root.get(&node) {
                return i;
            }
            let n = &self.arena.nodes[node as usize];
            let growth = |c: u32| {
                let cb = &self.arena.nodes[c as usize].bounds;
                cb.union(b).surface_area() - cb.surface_area()
            };
            node = if growth(n.b) < growth(n.a) { n.b } else { n.a };
        }
    }
}

/// Per-treelet bookkeeping for one update.
struct Touched {
    rebuild: Vec<bool>,
    rebuilt_now: Vec<bool>,
    refit: Vec<bool>,
    rebuilt: usize,
    refit_nodes: usize,
}

impl Touched {
    fn new(n: usize) -> Result<Self> {
        Ok(Self {
            rebuild: try_vec(n, false)?,
            rebuilt_now: try_vec(n, false)?,
            refit: try_vec(n, false)?,
            rebuilt: 0,
            refit_nodes: 0,
        })
    }

    /// Records a refit in `treelet`. Treelets split off during this update are
    /// not tracked: they were just built.
    fn mark(&mut self, treelet: usize) {
        if let Some(r) = self.refit.get_mut(treelet) {
            *r = true;
        }
    }
}

/// Sizes of a batch of treelet rebuilds, accumulated before reserving.
#[derive(Default)]
struct Budget {
    /// Largest rebuild, in triangles.
    max_n: usize,
    nodes: usize,
    refs: usize,
    rebuilds: usize,
    /// Upper bound on the treelets added by splitting oversized results.
    extra_treelets: usize,
}

impl Budget {
    /// Adds a rebuild of at most `n` triangles.
    fn add(&mut self, n: usize, treelet_size: u32) {
        self.max_n = self.max_n.max(n);
        self.nodes += max_nodes(n);
        self.refs += n;
        self.rebuilds += 1;
        let size = treelet_size as usize;
        if n > 2 * size {
            self.extra_treelets += n.div_ceil(size) * 2 + 1;
        }
    }
}

/// Scratch memory reserved for a batch of treelet rebuilds.
struct Scratch {
    prims: Prims,
    /// Per-node triangle counts of a freshly built subtree, for splitting it.
    counts: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::{History, Mesh};

    use crate::{Aabb, BuildOptions, Bvh, BvhError, Strategy, UpdatePolicy};

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

    fn opts() -> BuildOptions {
        BuildOptions {
            max_leaf_size: 4,
            treelet_size: 32,
            ..Default::default()
        }
    }

    fn hits(bvh: &Bvh, mesh: &Mesh, min: [f32; 3], max: [f32; 3]) -> Vec<u32> {
        let mut out = Vec::new();
        bvh.query_aabb(mesh, min, max, &mut out);
        out.sort();
        out
    }

    #[test]
    fn moving_a_vertex_refits() {
        let mut m = grid(16);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[0.0, 0.0, 5.0]]).unwrap();
        let p = tx.commit().unwrap();
        let report = bvh.update(&mut m, p.changes()).unwrap();
        assert_eq!(report.strategy, Strategy::Refit);
        assert!(report.refit_nodes > 0);
        bvh.validate(&m).unwrap();
        assert!(bvh.is_current(&m));
        assert_eq!(
            hits(&bvh, &m, [-1.0, -1.0, 4.0], [0.5, 0.5, 6.0]),
            vec![0, 1]
        );
    }

    #[test]
    fn local_topology_edits_rebuild_only_their_treelets() {
        let mut m = grid(16);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let treelets = bvh.treelet_count();
        assert!(treelets > 4);
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[0, 1]).unwrap();
        let v = tx.add_vertices(&[[0.5, 0.5, 2.0]]).unwrap()[0];
        tx.add_triangles(&[[0, 1, v]]).unwrap();
        let p = tx.commit().unwrap();
        let report = bvh.update(&mut m, p.changes()).unwrap();
        assert_eq!(report.strategy, Strategy::Partial);
        assert!(report.rebuilt_treelets >= 1 && report.rebuilt_treelets < treelets);
        bvh.validate(&m).unwrap();
        assert_eq!(
            hits(&bvh, &m, [0.0, 0.0, 1.5], [1.0, 1.0, 2.5]),
            p.changes().triangles_added
        );
    }

    #[test]
    fn large_edits_rebuild_everything() {
        let mut m = grid(8);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let all: Vec<u32> = (0..m.vertex_slots() as u32).collect();
        let lifted: Vec<[f32; 3]> = all
            .iter()
            .map(|&v| {
                let p = m.position(v).unwrap();
                [p[0], p[1], 1.0]
            })
            .collect();
        let mut tx = m.begin_edit();
        tx.set_positions(&all, &lifted).unwrap();
        let p = tx.commit().unwrap();
        let report = bvh.update(&mut m, p.changes()).unwrap();
        assert_eq!(report.strategy, Strategy::Full);
        bvh.validate(&m).unwrap();
    }

    #[test]
    fn empty_changes_only_record_the_revision() {
        let mut m = grid(2);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut tx = m.begin_edit();
        tx.add_vertices(&[[9.0; 3]]).unwrap();
        let p = tx.commit().unwrap();
        assert!(!bvh.is_current(&m));
        let report = bvh.update(&mut m, p.changes()).unwrap();
        assert_eq!(report.strategy, Strategy::Unchanged);
        assert!(bvh.is_current(&m));
    }

    #[test]
    fn undo_and_redo_change_sets_keep_the_tree_correct() {
        let mut m = grid(12);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut h = History::new(1 << 20);
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[10, 11, 12]).unwrap();
        tx.set_positions(&[40], &[[3.0, 3.0, 3.0]]).unwrap();
        let p = tx.commit().unwrap();
        bvh.update(&mut m, p.changes()).unwrap();
        h.push(&mut m, p).unwrap();
        bvh.validate(&m).unwrap();

        let undo = h.undo(&mut m).unwrap().unwrap();
        bvh.update(&mut m, &undo).unwrap();
        bvh.validate(&m).unwrap();
        let redo = h.redo(&mut m).unwrap().unwrap();
        bvh.update(&mut m, &redo).unwrap();
        bvh.validate(&m).unwrap();
    }

    #[test]
    fn emptying_and_refilling_the_mesh() {
        let mut m = grid(2);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let all: Vec<u32> = (0..8).collect();
        let mut tx = m.begin_edit();
        tx.remove_triangles(&all).unwrap();
        let p = tx.commit().unwrap();
        bvh.update(&mut m, p.changes()).unwrap();
        bvh.validate(&m).unwrap();
        assert_eq!(bvh.root_bounds(), None);
        let undo = p.revert(&mut m).unwrap();
        bvh.update(&mut m, &undo).unwrap();
        bvh.validate(&m).unwrap();
        assert_eq!(
            bvh.root_bounds(),
            Some(Aabb::new([0.0; 3], [2.0, 2.0, 0.0]))
        );
    }

    #[test]
    fn another_mesh_is_rejected() {
        let m = grid(2);
        let mut other = grid(2);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut tx = other.begin_edit();
        tx.set_positions(&[0], &[[1.0; 3]]).unwrap();
        let p = tx.commit().unwrap();
        assert_eq!(
            bvh.update(&mut other, p.changes()),
            Err(BvhError::MeshMismatch)
        );
        assert_eq!(bvh.refit(&other, &[0]), Err(BvhError::MeshMismatch));
        assert_eq!(bvh.rebuild(&other), Err(BvhError::MeshMismatch));
        bvh.validate(&m).unwrap();
    }

    #[test]
    fn degradation_triggers_a_treelet_rebuild() {
        let mut m = grid(16);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        bvh.set_policy(UpdatePolicy {
            degradation_factor: 1.0,
            full_rebuild_fraction: 1.0,
        })
        .unwrap();
        // Scatter a few vertices far away: refits blow up the boxes they touch.
        let ids = [0u32, 50, 100, 150];
        let far: Vec<[f32; 3]> = ids
            .iter()
            .map(|&v| [v as f32, 0.0, 100.0 + v as f32])
            .collect();
        let mut tx = m.begin_edit();
        tx.set_positions(&ids, &far).unwrap();
        let p = tx.commit().unwrap();
        let report = bvh.update(&mut m, p.changes()).unwrap();
        assert!(report.rebuilt_treelets > 0, "{report:?}");
        bvh.validate(&m).unwrap();
    }

    #[test]
    fn manual_refit_and_region_rebuild() {
        let mut m = grid(8);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[0.0, 0.0, 2.0]]).unwrap();
        tx.commit().unwrap();
        bvh.refit(&m, &[0, 1]).unwrap();
        bvh.validate(&m).unwrap();
        bvh.rebuild_region(&m, &[0]).unwrap();
        bvh.validate(&m).unwrap();
        bvh.rebuild(&m).unwrap();
        bvh.validate(&m).unwrap();
        assert!(bvh.is_current(&m));
    }

    #[test]
    fn garbage_is_compacted_away() {
        let mut m = grid(16);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut compacted = false;
        for round in 0..40u32 {
            let t = (round * 37) % m.triangle_slots() as u32;
            let Some(tri) = m.triangle(t) else { continue };
            let mut tx = m.begin_edit();
            tx.remove_triangles(&[t]).unwrap();
            tx.add_triangles(&[tri]).unwrap();
            let p = tx.commit().unwrap();
            compacted |= bvh.update(&mut m, p.changes()).unwrap().compacted;
            bvh.validate(&m).unwrap();
        }
        assert!(compacted);
    }

    #[test]
    fn replaying_a_change_set_does_not_duplicate_triangles() {
        let mut m = grid(16);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut tx = m.begin_edit();
        let v = tx.add_vertices(&[[0.5, 0.5, 2.0]]).unwrap()[0];
        tx.add_triangles(&[[0, 1, v]]).unwrap();
        let p = tx.commit().unwrap();
        assert_eq!(
            bvh.update(&mut m, p.changes()).unwrap().strategy,
            Strategy::Partial
        );
        assert_eq!(
            bvh.update(&mut m, p.changes()).unwrap().strategy,
            Strategy::Unchanged
        );
        bvh.validate(&m).unwrap();
        assert_eq!(bvh.revision(), m.revision());
    }

    #[test]
    fn re_listed_additions_are_inserted_once() {
        let mut m = grid(16);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut tx = m.begin_edit();
        let v = tx.add_vertices(&[[0.5, 0.5, 2.0]]).unwrap()[0];
        tx.add_triangles(&[[0, 1, v]]).unwrap();
        let first = tx.commit().unwrap().changes().clone();
        bvh.update(&mut m, &first).unwrap();
        let mut tx = m.begin_edit();
        tx.add_triangles(&[[20, 21, v]]).unwrap();
        let second = tx.commit().unwrap().changes().clone();
        // A merged set that re-lists the earlier addition.
        let mut merged = second.clone();
        merged
            .triangles_added
            .extend_from_slice(&first.triangles_added);
        merged.triangles_added.sort_unstable();
        merged.triangles_added.dedup();
        let report = bvh.update(&mut m, &merged).unwrap();
        assert_eq!(report.strategy, Strategy::Partial);
        bvh.validate(&m).unwrap();
        let all = hits(&bvh, &m, [-1.0; 3], [20.0, 20.0, 5.0]);
        assert_eq!(all.len(), m.triangle_count());
    }

    #[test]
    fn compaction_of_the_mesh_forces_a_full_rebuild() {
        let mut m = grid(8);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[0, 1, 2, 30]).unwrap();
        let p = tx.commit().unwrap();
        bvh.update(&mut m, p.changes()).unwrap();
        m.compact().unwrap();
        assert!(!bvh.is_current(&m));
        let report = bvh
            .update(&mut m, &fast_mesh_store::ChangeSet::default())
            .unwrap();
        assert_eq!(report.strategy, Strategy::Full);
        assert!(bvh.is_current(&m));
        bvh.validate(&m).unwrap();
        assert_eq!(
            hits(&bvh, &m, [-1.0; 3], [9.0, 9.0, 1.0]).len(),
            m.triangle_count()
        );

        // Manual refit and region rebuilds also notice the renumbering.
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[5, 6]).unwrap();
        tx.commit().unwrap();
        let mut a = Bvh::build(&m, &opts()).unwrap();
        let mut b = Bvh::build(&m, &opts()).unwrap();
        m.compact().unwrap();
        a.refit(&m, &[0]).unwrap();
        a.validate(&m).unwrap();
        assert!(a.is_current(&m));
        b.rebuild_region(&m, &[0]).unwrap();
        b.validate(&m).unwrap();
        assert!(b.is_current(&m));
    }

    #[test]
    fn region_rebuild_on_a_stale_tree_then_update() {
        let mut m = grid(8);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[5]).unwrap();
        let p = tx.commit().unwrap();
        bvh.rebuild_region(&m, &[4, 5]).unwrap();
        bvh.validate(&m).unwrap();
        bvh.update(&mut m, p.changes()).unwrap();
        bvh.validate(&m).unwrap();
        assert!(bvh.is_current(&m));
    }

    #[test]
    fn treelets_stay_bounded_as_the_mesh_grows() {
        let mut m = grid(8);
        let options = BuildOptions {
            max_leaf_size: 4,
            treelet_size: 64,
            ..Default::default()
        };
        let mut bvh = Bvh::build(&m, &options).unwrap();
        for round in 0..14u32 {
            // Add about 20% more triangles, spread over the mesh.
            let alive: Vec<u32> = (0..m.triangle_slots() as u32)
                .filter(|&t| m.is_triangle_alive(t))
                .collect();
            let extra: Vec<[u32; 3]> = alive
                .iter()
                .step_by(5)
                .map(|&t| m.triangle(t).unwrap())
                .collect();
            let mut tx = m.begin_edit();
            tx.add_triangles(&extra).unwrap();
            let p = tx.commit().unwrap();
            let report = bvh.update(&mut m, p.changes()).unwrap();
            assert_eq!(report.strategy, Strategy::Partial, "round {round}");
            bvh.validate(&m).unwrap();
            let largest = bvh.treelet_triangle_counts().into_iter().max().unwrap();
            assert!(largest <= 128, "round {round}: treelet of {largest}");
        }
        // A one-triangle edit now rebuilds a bounded amount.
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[7]).unwrap();
        let p = tx.commit().unwrap();
        assert_eq!(bvh.update(&mut m, p.changes()).unwrap().rebuilt_treelets, 1);
        bvh.validate(&m).unwrap();
    }

    /// Splitting removes the old root and inserts one entry per new treelet.
    /// The removal can leave a tombstone that does not give its room back, so
    /// the map needs spare room for every inserted entry, not just the net gain.
    #[test]
    fn treelet_split_needs_room_for_every_new_root() {
        let m = grid(8);
        // One treelet over the whole mesh: its nodes are the whole arena.
        let options = BuildOptions {
            max_leaf_size: 4,
            treelet_size: 1024,
            ..Default::default()
        };
        let base = Bvh::build(&m, &options).unwrap();
        assert_eq!(base.treelet_count(), 1);
        let mut counts = vec![0u32; base.arena.nodes.len()];
        // Number of treelets a split at size 32 produces.
        let found = {
            let mut probe = Bvh::build(&m, &options).unwrap();
            probe.options.treelet_size = 32;
            probe.treelets.reserve(64);
            probe.treelet_of_root.reserve(64);
            probe.split_treelet(0, &mut counts);
            probe.validate(&m).unwrap();
            probe.treelet_count()
        };
        assert!(found > 1);
        for spare in [found - 1, found] {
            let mut bvh = Bvh::build(&m, &options).unwrap();
            bvh.options.treelet_size = 32;
            bvh.treelets.reserve(64);
            // Fill a fresh map with placeholder keys until exactly `spare`
            // entries fit without growing.
            let mut map = std::collections::HashMap::with_capacity(found + 16);
            map.insert(bvh.root, 0usize);
            let mut filler = u32::MAX;
            while map.capacity() - map.len() > spare {
                map.insert(filler, usize::MAX);
                filler -= 1;
            }
            assert_eq!(map.capacity() - map.len(), spare);
            let capacity = map.capacity();
            bvh.treelet_of_root = map;
            bvh.split_treelet(0, &mut counts);
            assert_eq!(bvh.treelet_of_root.capacity(), capacity, "spare {spare}");
            let expected = if spare == found { found } else { 1 };
            assert_eq!(bvh.treelet_count(), expected, "spare {spare}");
            bvh.treelet_of_root.retain(|_, &mut i| i != usize::MAX);
            bvh.validate(&m).unwrap();
        }
    }
}

#[cfg(test)]
mod oom_tests {
    use fast_mesh_store::Mesh;

    use crate::error::fail_allocations_after;
    use crate::{BuildOptions, Bvh, BvhError};

    fn wavy(n: u32) -> Mesh {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                positions.extend([x as f32, y as f32, ((x + y) % 3) as f32]);
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

    /// Fails the k-th allocation of an update for every k until the update
    /// succeeds; each failure must leave the tree exactly as it was.
    fn assert_atomic(edit: impl Fn(&mut Mesh) -> fast_mesh_store::ChangeSet) {
        let opts = BuildOptions {
            max_leaf_size: 2,
            treelet_size: 8,
            ..Default::default()
        };
        for k in 0.. {
            let mut m = wavy(8);
            let mut bvh = Bvh::build(&m, &opts).unwrap();
            m.ensure_incidence().unwrap();
            let changes = edit(&mut m);
            let before = bvh.fingerprint();
            fail_allocations_after(Some(k));
            let result = bvh.update(&mut m, &changes);
            fail_allocations_after(None);
            match result {
                Err(BvhError::OutOfMemory) => {
                    assert_eq!(bvh.fingerprint(), before, "allocation {k}")
                }
                Err(e) => panic!("unexpected error {e}"),
                Ok(_) => {
                    bvh.validate(&m).unwrap();
                    assert!(k > 0, "the update must allocate at least once");
                    return;
                }
            }
        }
    }

    #[test]
    fn failed_partial_updates_leave_the_tree_unchanged() {
        assert_atomic(|m| {
            let mut tx = m.begin_edit();
            tx.remove_triangles(&[3, 40]).unwrap();
            tx.add_triangles(&[[0, 10, 20]]).unwrap();
            tx.set_positions(&[30], &[[3.0, 3.0, 4.0]]).unwrap();
            tx.commit().unwrap().changes().clone()
        });
    }

    #[test]
    fn failed_region_rebuilds_leave_the_tree_unchanged() {
        let opts = BuildOptions {
            max_leaf_size: 2,
            treelet_size: 8,
            ..Default::default()
        };
        // Triangles far apart, so the region spans several treelets.
        let region = [0u32, 40, 80, 120];
        for k in 0.. {
            let m = wavy(8);
            let mut bvh = Bvh::build(&m, &opts).unwrap();
            let mut which: Vec<_> = region
                .iter()
                .map(|&t| bvh.treelet_of_triangle(t).unwrap())
                .collect();
            which.dedup();
            assert!(which.len() > 1, "the region must span several treelets");
            let before = bvh.fingerprint();
            fail_allocations_after(Some(k));
            let result = bvh.rebuild_region(&m, &region);
            fail_allocations_after(None);
            match result {
                Err(BvhError::OutOfMemory) => {
                    assert_eq!(bvh.fingerprint(), before, "allocation {k}")
                }
                Err(e) => panic!("unexpected error {e}"),
                Ok(()) => {
                    assert!(k > 1, "the rebuild must allocate several times");
                    assert_ne!(bvh.fingerprint(), before);
                    bvh.validate(&m).unwrap();
                    return;
                }
            }
        }
    }

    #[test]
    fn failed_full_rebuilds_leave_the_tree_unchanged() {
        assert_atomic(|m| {
            let all: Vec<u32> = (0..m.triangle_slots() as u32).step_by(2).collect();
            let mut tx = m.begin_edit();
            tx.remove_triangles(&all).unwrap();
            tx.commit().unwrap().changes().clone()
        });
    }
}
