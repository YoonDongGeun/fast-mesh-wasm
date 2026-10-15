//! Top-down subtree builders: binned SAH and Morton-code (LBVH).
//!
//! Both work on an index permutation (`Prims::order`) so the primitive arrays are
//! never moved. Full builds start from an estimated node count and grow the
//! arena on demand, so they can fail with `OutOfMemory`. When the caller has
//! reserved the worst case up front (`max_nodes(n)` nodes and parents and `n`
//! refs, as treelet rebuilds do), building never allocates and cannot fail.

use crate::arena::{Arena, LEAF_BIT, NONE, Node};
use crate::error::{Result, try_reserve, try_reserve_exact};
use crate::{Aabb, BuildOptions, Quality};

/// Triangles to build a subtree from, plus scratch space.
pub(crate) struct Prims {
    pub(crate) ids: Vec<u32>,
    pub(crate) boxes: Vec<Aabb>,
    pub(crate) centers: Vec<[f32; 3]>,
    keys: Vec<u32>,
    order: Vec<u32>,
    scratch: Vec<u32>,
}

/// Most nodes a subtree over `n` triangles can need.
pub(crate) fn max_nodes(n: usize) -> usize {
    if n == 0 { 1 } else { 2 * n - 1 }
}

/// Typical node count for `n` triangles with leaves of up to `leaf` triangles,
/// used as the initial reservation of full builds (they grow if needed).
pub(crate) fn expected_nodes(n: usize, leaf: u32) -> usize {
    (3 * n.div_ceil(leaf as usize) + 1).min(max_nodes(n))
}

/// Ensures room for `needed` more elements, reserving at least an eighth of the
/// current length extra so repeated small rebuilds do not copy the whole array
/// each time, while keeping slack far below the 2x of doubling.
pub(crate) fn reserve_headroom<T>(v: &mut Vec<T>, needed: usize) -> Result<()> {
    if v.capacity() - v.len() < needed {
        try_reserve_exact(v, needed.max(v.len() / 8))?;
    }
    Ok(())
}

impl Prims {
    /// Empty set with room for `n` triangles.
    pub(crate) fn with_capacity(n: usize) -> Result<Self> {
        let mut p = Self {
            ids: Vec::new(),
            boxes: Vec::new(),
            centers: Vec::new(),
            keys: Vec::new(),
            order: Vec::new(),
            scratch: Vec::new(),
        };
        try_reserve(&mut p.ids, n)?;
        try_reserve(&mut p.boxes, n)?;
        try_reserve(&mut p.centers, n)?;
        try_reserve(&mut p.keys, n)?;
        try_reserve(&mut p.order, n)?;
        try_reserve(&mut p.scratch, n)?;
        Ok(p)
    }

    pub(crate) fn clear(&mut self) {
        self.ids.clear();
        self.boxes.clear();
        self.centers.clear();
    }

    /// Adds a triangle. The caller stays within the reserved capacity.
    pub(crate) fn push(&mut self, id: u32, bounds: Aabb) {
        debug_assert!(
            self.ids.len() < self.ids.capacity(),
            "Prims capacity exceeded"
        );
        self.ids.push(id);
        self.boxes.push(bounds);
        self.centers.push(bounds.center());
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    fn range_bounds(&self, start: usize, end: usize) -> (Aabb, Aabb) {
        let mut bounds = Aabb::EMPTY;
        let mut centers = Aabb::EMPTY;
        for &i in &self.order[start..end] {
            bounds.grow(&self.boxes[i as usize]);
            centers.grow_point(self.centers[i as usize]);
        }
        (bounds, centers)
    }
}

/// Builds a subtree over all of `prims`, appending to `arena`, and returns its root.
/// `tri_leaf` must cover every triangle id in `prims`. Nodes are allocated as
/// needed; when the caller has already reserved `max_nodes(n)` nodes and parents and
/// `n` refs (as treelet rebuilds do), this cannot fail.
///
/// Not atomic: on `Err` the arena may already hold some of the new nodes, refs
/// and `tri_leaf` entries. Callers either build into a fresh arena they discard
/// on error (full builds) or reserve the worst case first so no error can occur
/// (treelet rebuilds).
pub(crate) fn build_subtree(
    arena: &mut Arena,
    prims: &mut Prims,
    opts: &BuildOptions,
    parent: u32,
) -> Result<u32> {
    let n = prims.len();
    if arena.refs.capacity() - arena.refs.len() < n {
        try_reserve(&mut arena.refs, n)?;
    }
    prims.order.clear();
    prims.order.extend(0..n as u32);
    if opts.quality == Quality::Fast && n > 1 {
        morton_sort(prims);
    }

    let root = push_node(arena, parent)?;
    // Continue with the smaller child and push the larger: the stack never holds
    // more than log2(n) entries, so a fixed array is enough.
    let mut stack = [(0u32, 0usize, 0usize); 64];
    let mut depth = 0;
    let mut task = (root, 0, n);
    loop {
        let (node, start, end) = task;
        let (bounds, centers) = prims.range_bounds(start, end);
        if end - start <= opts.max_leaf_size as usize {
            let first = arena.refs.len() as u32;
            for &i in &prims.order[start..end] {
                let id = prims.ids[i as usize];
                arena.refs.push(id);
                arena.tri_leaf[id as usize] = node;
            }
            arena.nodes[node as usize] = Node::leaf(bounds, first, (end - start) as u32);
            if depth == 0 {
                return Ok(root);
            }
            depth -= 1;
            task = stack[depth];
            continue;
        }
        let mid = match opts.quality {
            Quality::Sah => sah_split(prims, start, end, &centers, opts.sah_bins as usize),
            Quality::Fast => morton_split(prims, start, end),
        };
        let left = push_node(arena, node)?;
        let right = push_node(arena, node)?;
        arena.nodes[node as usize] = Node {
            bounds,
            a: left,
            b: right,
        };
        let (small, large) = if mid - start <= end - mid {
            ((left, start, mid), (right, mid, end))
        } else {
            ((right, mid, end), (left, start, mid))
        };
        stack[depth] = large;
        depth += 1;
        task = small;
    }
}

fn push_node(arena: &mut Arena, parent: u32) -> Result<u32> {
    if arena.nodes.len() == arena.nodes.capacity() {
        let more = arena.nodes.len().max(16);
        try_reserve(&mut arena.nodes, more)?;
    }
    if arena.parents.len() == arena.parents.capacity() {
        let more = arena.parents.len().max(16);
        try_reserve(&mut arena.parents, more)?;
    }
    let index = arena.nodes.len() as u32;
    arena.nodes.push(Node {
        bounds: Aabb::EMPTY,
        a: NONE,
        b: LEAF_BIT,
    });
    arena.parents.push(parent);
    Ok(index)
}

/// Chooses the cheapest binned SAH split and partitions `order[start..end]`.
/// Falls back to the middle when no split separates the centers.
fn sah_split(prims: &mut Prims, start: usize, end: usize, centers: &Aabb, bins: usize) -> usize {
    let mid = start + (end - start) / 2;
    let mut best: Option<(f32, usize, usize)> = None; // (cost, axis, last left bin)
    for axis in 0..3 {
        let lo = centers.min[axis];
        let extent = centers.max[axis] - lo;
        if !extent.is_finite() || extent <= 0.0 {
            continue;
        }
        let bin_of =
            |c: f32| (((c - lo) / extent) * bins as f32).clamp(0.0, (bins - 1) as f32) as usize;
        let mut counts = [0u32; 64];
        let mut boxes = [Aabb::EMPTY; 64];
        for &i in &prims.order[start..end] {
            let b = bin_of(prims.centers[i as usize][axis]);
            counts[b] += 1;
            boxes[b].grow(&prims.boxes[i as usize]);
        }
        // Sweep from the right to get the right-hand side areas and counts.
        let mut right_area = [0.0f32; 64];
        let mut right_count = [0u32; 64];
        let mut acc = Aabb::EMPTY;
        let mut cnt = 0;
        for b in (1..bins).rev() {
            acc.grow(&boxes[b]);
            cnt += counts[b];
            right_area[b] = acc.surface_area();
            right_count[b] = cnt;
        }
        let mut left = Aabb::EMPTY;
        let mut left_count = 0;
        for b in 0..bins - 1 {
            left.grow(&boxes[b]);
            left_count += counts[b];
            if left_count == 0 || right_count[b + 1] == 0 {
                continue;
            }
            let cost = left.surface_area() * left_count as f32
                + right_area[b + 1] * right_count[b + 1] as f32;
            if best.is_none_or(|(c, _, _)| cost < c) {
                best = Some((cost, axis, b));
            }
        }
    }
    let Some((_, axis, split_bin)) = best else {
        return mid;
    };
    let lo = centers.min[axis];
    let extent = centers.max[axis] - lo;
    let in_left = |c: f32| {
        (((c - lo) / extent) * bins as f32).clamp(0.0, (bins - 1) as f32) as usize <= split_bin
    };
    let (mut i, mut j) = (start, end);
    while i < j {
        if in_left(prims.centers[prims.order[i] as usize][axis]) {
            i += 1;
        } else {
            j -= 1;
            prims.order.swap(i, j);
        }
    }
    if i == start || i == end { mid } else { i }
}

/// Splits a Morton-sorted range at the highest bit where its codes differ.
fn morton_split(prims: &Prims, start: usize, end: usize) -> usize {
    let key = |i: usize| prims.keys[prims.order[i] as usize];
    let (first, last) = (key(start), key(end - 1));
    if first == last {
        return start + (end - start) / 2;
    }
    let bit = 31 - (first ^ last).leading_zeros();
    let mask = 1u32 << bit;
    // Codes are sorted, so those with the bit set form a suffix.
    let (mut lo, mut hi) = (start, end - 1);
    while lo < hi {
        let m = lo + (hi - lo) / 2;
        if key(m) & mask != 0 {
            hi = m
        } else {
            lo = m + 1
        }
    }
    lo
}

/// 30-bit Morton codes of the centers within their bounds, then a stable LSD
/// radix sort of `order` by code (ties keep id order, so builds are deterministic).
fn morton_sort(prims: &mut Prims) {
    let mut bounds = Aabb::EMPTY;
    for &c in &prims.centers {
        bounds.grow_point(c);
    }
    prims.keys.clear();
    for &c in &prims.centers {
        let q = [0, 1, 2].map(|k| {
            let extent = bounds.max[k] - bounds.min[k];
            let t = if extent > 0.0 && extent.is_finite() {
                (c[k] - bounds.min[k]) / extent
            } else {
                0.0
            };
            (t.clamp(0.0, 1.0) * 1023.0) as u32
        });
        prims
            .keys
            .push((spread(q[0]) << 2) | (spread(q[1]) << 1) | spread(q[2]));
    }
    let n = prims.order.len();
    prims.scratch.clear();
    prims.scratch.resize(n, 0);
    for shift in [0, 8, 16, 24] {
        let mut counts = [0usize; 257];
        for &i in &prims.order {
            counts[((prims.keys[i as usize] >> shift) & 0xff) as usize + 1] += 1;
        }
        for b in 0..256 {
            counts[b + 1] += counts[b];
        }
        for &i in &prims.order {
            let b = ((prims.keys[i as usize] >> shift) & 0xff) as usize;
            prims.scratch[counts[b]] = i;
            counts[b] += 1;
        }
        std::mem::swap(&mut prims.order, &mut prims.scratch);
    }
}

/// Spreads the low 10 bits of `x` so that there are two zero bits between each.
fn spread(mut x: u32) -> u32 {
    x &= 0x3ff;
    x = (x | (x << 16)) & 0x030000ff;
    x = (x | (x << 8)) & 0x0300f00f;
    x = (x | (x << 4)) & 0x030c30c3;
    x = (x | (x << 2)) & 0x09249249;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arena_for(n: usize, tris: usize) -> Arena {
        let mut a = Arena {
            nodes: Vec::new(),
            parents: Vec::new(),
            refs: Vec::new(),
            tri_leaf: vec![NONE; tris],
        };
        a.nodes.reserve(max_nodes(n));
        a.parents.reserve(max_nodes(n));
        a.refs.reserve(n);
        a
    }

    /// `n` unit triangles spaced along x.
    fn row(n: u32) -> Prims {
        let mut p = Prims::with_capacity(n as usize).unwrap();
        for i in 0..n {
            let x = i as f32 * 2.0;
            p.push(i, Aabb::new([x, 0.0, 0.0], [x + 1.0, 1.0, 0.0]));
        }
        p
    }

    fn leaves(a: &Arena, root: u32) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        a.traverse(root, |_, n| {
            if n.is_leaf() {
                out.push(a.refs[n.leaf_range()].to_vec());
            }
            true
        });
        out
    }

    #[test]
    fn spread_interleaves_bits() {
        assert_eq!(spread(0b11), 0b1001);
        assert_eq!(spread(0x3ff), 0x09249249);
    }

    #[test]
    fn both_builders_cover_every_triangle_once_within_leaf_size() {
        for quality in [Quality::Sah, Quality::Fast] {
            let opts = BuildOptions {
                quality,
                max_leaf_size: 4,
                ..Default::default()
            };
            let mut prims = row(37);
            let mut arena = arena_for(37, 37);
            let root = build_subtree(&mut arena, &mut prims, &opts, NONE).unwrap();
            let mut all: Vec<u32> = leaves(&arena, root).into_iter().flatten().collect();
            all.sort();
            assert_eq!(all, (0..37).collect::<Vec<_>>(), "{quality:?}");
            assert!(leaves(&arena, root).iter().all(|l| l.len() <= 4));
            for t in 0..37 {
                let leaf = arena.tri_leaf[t as usize];
                assert!(arena.refs[arena.nodes[leaf as usize].leaf_range()].contains(&t));
            }
            assert_eq!(
                arena.nodes[root as usize].bounds,
                Aabb::new([0.0, 0.0, 0.0], [73.0, 1.0, 0.0])
            );
        }
    }

    #[test]
    fn sah_separates_spatial_clusters() {
        let mut p = Prims::with_capacity(8).unwrap();
        for i in 0..8u32 {
            let x = if i % 2 == 0 { 0.0 } else { 100.0 } + i as f32 * 0.01;
            p.push(i, Aabb::new([x, 0.0, 0.0], [x + 0.5, 0.5, 0.0]));
        }
        let mut arena = arena_for(8, 8);
        let opts = BuildOptions {
            max_leaf_size: 4,
            ..Default::default()
        };
        let root = build_subtree(&mut arena, &mut p, &opts, NONE).unwrap();
        let mut groups = leaves(&arena, root);
        groups.iter_mut().for_each(|g| g.sort());
        groups.sort();
        assert_eq!(groups, vec![vec![0, 2, 4, 6], vec![1, 3, 5, 7]]);
    }

    #[test]
    fn identical_centers_fall_back_to_halving() {
        for quality in [Quality::Sah, Quality::Fast] {
            let mut p = Prims::with_capacity(9).unwrap();
            for i in 0..9 {
                p.push(i, Aabb::new([0.0; 3], [1.0; 3]));
            }
            let mut arena = arena_for(9, 9);
            let opts = BuildOptions {
                quality,
                max_leaf_size: 2,
                ..Default::default()
            };
            let root = build_subtree(&mut arena, &mut p, &opts, NONE).unwrap();
            assert_eq!(leaves(&arena, root).iter().map(Vec::len).sum::<usize>(), 9);
            assert!(leaves(&arena, root).iter().all(|l| l.len() <= 2));
        }
    }

    #[test]
    fn empty_input_builds_one_empty_leaf() {
        let mut p = Prims::with_capacity(0).unwrap();
        let mut arena = arena_for(0, 0);
        let root = build_subtree(&mut arena, &mut p, &BuildOptions::default(), NONE).unwrap();
        assert!(arena.nodes[root as usize].is_leaf());
        assert!(arena.nodes[root as usize].bounds.is_empty());
        assert_eq!(arena.nodes[root as usize].leaf_range().len(), 0);
    }

    #[test]
    fn builds_are_deterministic() {
        let shape = |arena: &Arena| arena.nodes.iter().map(|n| (n.a, n.b)).collect::<Vec<_>>();
        for quality in [Quality::Sah, Quality::Fast] {
            let opts = BuildOptions {
                quality,
                max_leaf_size: 3,
                ..Default::default()
            };
            let (mut a1, mut a2) = (arena_for(50, 50), arena_for(50, 50));
            build_subtree(&mut a1, &mut row(50), &opts, NONE).unwrap();
            build_subtree(&mut a2, &mut row(50), &opts, NONE).unwrap();
            assert_eq!(shape(&a1), shape(&a2));
            assert_eq!(a1.refs, a2.refs);
        }
    }
}
