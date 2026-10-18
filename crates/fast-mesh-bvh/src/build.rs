//! Top-down subtree builders: binned SAH and Morton-code (LBVH).
//!
//! Both work on an index permutation (`Prims::order`) so the primitive arrays are
//! never moved. Full builds start from an estimated node count and grow the
//! arena on demand, so they can fail with `OutOfMemory`. When the caller has
//! reserved the worst case up front (`max_nodes(n)` nodes and parents and `n`
//! refs, as treelet rebuilds do), building never allocates and cannot fail.
//!
//! In full builds, nodes over at least [`PAR_MIN_SUBTREE`] triangles reduce
//! their bounds and SAH bins over fixed chunks. With the `parallel` feature,
//! the top of the tree is also split into parts that are built on the pool and
//! then spliced into one arena; without it, the same tree is built in one pass.
//! The node layout and every reduction are the same either way, so the result
//! does not depend on the feature or the thread count. (Compared with builds
//! before chunking, a bound that is a zero of either sign may come out with the
//! other sign, because `f32::min`/`max` may return either zero.)

use fast_mesh_store::par;

use crate::arena::{Arena, LEAF_BIT, NONE, Node};
use crate::error::{BvhError, Result, allocation_allowed, try_reserve, try_reserve_exact};
use crate::{Aabb, BuildOptions, Quality};

/// Full builds split subtrees of at least this many triangles into separately
/// built parts, whose two halves run in parallel with the `parallel` feature.
/// Smaller subtrees are built in one sequential pass. On the i5-13600K, 1M
/// triangle SAH builds take the same time (within noise) from 2K to 8K and get
/// slower above: 70 ms at 16K, 95 ms at 64K.
pub(crate) const PAR_MIN_SUBTREE: usize = 8 * 1024;

/// The bound and SAH bin reductions of a split part run on the pool from this
/// many triangles.
const PAR_MIN_PRIMS: usize = 64 * 1024;

/// Triangles per chunk of those reductions. Fixed, so the combined result does
/// not depend on the thread count.
const PAR_CHUNK: usize = 16 * 1024;

/// Triangles to build a subtree from, plus scratch space.
pub(crate) struct Prims {
    pub(crate) ids: Vec<u32>,
    pub(crate) boxes: Vec<Aabb>,
    pub(crate) centers: Vec<[f32; 3]>,
    keys: Vec<u32>,
    order: Vec<u32>,
    scratch: Vec<u32>,
}

/// The read-only primitive arrays the builders index through `order`.
#[derive(Clone, Copy)]
struct Src<'a> {
    ids: &'a [u32],
    boxes: &'a [Aabb],
    centers: &'a [[f32; 3]],
    keys: &'a [u32],
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

    /// Resets `order` to the identity and, for Morton builds, sorts it.
    fn prepare(&mut self, opts: &BuildOptions) {
        let n = self.len();
        self.order.clear();
        self.order.extend(0..n as u32);
        if opts.quality == Quality::Fast && n > 1 {
            morton_sort(self);
        }
    }

    /// The primitive arrays and, separately, the mutable order.
    fn split(&mut self) -> (Src<'_>, &mut [u32]) {
        let src = Src {
            ids: &self.ids,
            boxes: &self.boxes,
            centers: &self.centers,
            keys: &self.keys,
        };
        (src, &mut self.order)
    }
}

/// Box of the triangles in `order` and box of their centers.
fn range_bounds(src: Src, order: &[u32]) -> (Aabb, Aabb) {
    let mut bounds = Aabb::EMPTY;
    let mut centers = Aabb::EMPTY;
    for &i in order {
        bounds.grow(&src.boxes[i as usize]);
        centers.grow_point(src.centers[i as usize]);
    }
    (bounds, centers)
}

/// [`range_bounds`] over fixed chunks, combined in chunk order.
fn range_bounds_chunked(src: Src, order: &[u32]) -> Result<(Aabb, Aabb)> {
    allocation_allowed()?;
    let oom = || BvhError::OutOfMemory;
    let parts = par::map_chunks(order, PAR_CHUNK, PAR_MIN_PRIMS, oom, |_, c| {
        Ok(range_bounds(src, c))
    })?;
    let mut bounds = Aabb::EMPTY;
    let mut centers = Aabb::EMPTY;
    for (b, c) in &parts {
        bounds.grow(b);
        centers.grow(c);
    }
    Ok((bounds, centers))
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
    prims.prepare(opts);
    let (src, order) = prims.split();
    let root = push_node(&mut arena.nodes, &mut arena.parents, parent)?;
    let mut out = Out {
        nodes: &mut arena.nodes,
        parents: &mut arena.parents,
        refs: &mut arena.refs,
        tri_leaf: Some(&mut arena.tri_leaf),
    };
    build_seq(&mut out, src, order, opts, root, usize::MAX)?;
    Ok(root)
}

/// Builds a whole tree over `prims` (at least one triangle) into the empty
/// `arena` and returns its root. Allocates as it goes; on `Err` the arena is
/// garbage.
pub(crate) fn build_full(arena: &mut Arena, prims: &mut Prims, opts: &BuildOptions) -> Result<u32> {
    build_full_with(arena, prims, opts, PAR_MIN_SUBTREE)
}

/// [`build_full`] with chunked reductions and parts from `min_part` triangles.
/// Both ways give the same tree: parts only when work can go to the pool.
fn build_full_with(
    arena: &mut Arena,
    prims: &mut Prims,
    opts: &BuildOptions,
    min_part: usize,
) -> Result<u32> {
    if prims.len() < min_part || !par::enabled() || par::force_sequential() {
        build_full_seq(arena, prims, opts, min_part)
    } else {
        build_full_parts(arena, prims, opts, min_part)
    }
}

/// [`build_full_with`] in one sequential pass straight into the arena.
fn build_full_seq(
    arena: &mut Arena,
    prims: &mut Prims,
    opts: &BuildOptions,
    min_part: usize,
) -> Result<u32> {
    let n = prims.len();
    let expected = expected_nodes(n, opts.max_leaf_size);
    try_reserve_exact(&mut arena.nodes, expected)?;
    try_reserve_exact(&mut arena.parents, expected)?;
    try_reserve_exact(&mut arena.refs, n)?;
    prims.prepare(opts);
    let (src, order) = prims.split();
    let root = push_node(&mut arena.nodes, &mut arena.parents, NONE)?;
    let mut out = Out {
        nodes: &mut arena.nodes,
        parents: &mut arena.parents,
        refs: &mut arena.refs,
        tri_leaf: Some(&mut arena.tri_leaf),
    };
    build_seq(&mut out, src, order, opts, root, min_part)?;
    Ok(root)
}

/// [`build_full_with`] from parts built on the pool, then spliced.
fn build_full_parts(
    arena: &mut Arena,
    prims: &mut Prims,
    opts: &BuildOptions,
    min_part: usize,
) -> Result<u32> {
    let n = prims.len();
    prims.prepare(opts);
    let (src, order) = prims.split();
    let parts = build_parts(src, order, opts, min_part, 0)?;
    // Each part adds its nodes but its root, which its parent placed; each
    // split adds its two children. Plus the root.
    let (mut nodes, mut splits) = (1, 0);
    for p in &parts {
        match p {
            Part::Seq { nodes: n, .. } => nodes += n.len() - 1,
            Part::Split { .. } => (nodes, splits) = (nodes + 2, splits + 1),
        }
    }
    try_reserve_exact(&mut arena.nodes, nodes)?;
    try_reserve_exact(&mut arena.parents, nodes)?;
    try_reserve_exact(&mut arena.refs, n)?;
    // Each split takes one pending slot and adds two.
    let mut pending = Vec::new();
    try_reserve_exact(&mut pending, splits + 1)?;
    let root = push_node(&mut arena.nodes, &mut arena.parents, NONE)?;
    pending.push((root, NONE));
    splice(arena, parts, &mut pending);
    debug_assert!(pending.is_empty() && arena.nodes.len() == nodes);
    Ok(root)
}

/// A piece of a full build, in a list in the order the pieces are spliced.
enum Part {
    /// A subtree built in one sequential pass, numbered from its root at 0;
    /// leaf ref ranges index `refs`.
    Seq {
        nodes: Vec<Node>,
        parents: Vec<u32>,
        refs: Vec<u32>,
    },
    /// An inner node. The list continues with its smaller child's parts, then
    /// its larger child's (the left one first when equal), the order in which
    /// the sequential builder emits their nodes.
    Split { bounds: Aabb, left_first: bool },
}

/// Most nested [`par::join`]s in a full build. Balanced trees need about
/// log2(n / `PAR_MIN_SUBTREE`) (under 16 even at 100M triangles); degenerate
/// split chains stop forking here and build the rest as one part, so the call
/// stack stays bounded however deep the tree is.
const MAX_FORK_DEPTH: usize = 24;

/// The parts of the subtree over `order`, given `forks` enclosing joins.
///
/// A subtree below `min_part` triangles (or at the fork limit) is one
/// [`Part::Seq`]. Otherwise it is split: a smaller half below `min_part` is
/// built right away and the larger half continues in the loop; when both
/// halves are larger, they are built with [`par::join`]. Recursion therefore
/// only happens through joins, at most [`MAX_FORK_DEPTH`] deep. Every node of
/// at least `min_part` triangles uses the chunked reductions wherever it is
/// built, so where the parts are cut does not change the tree.
///
/// When both halves fail, the smaller (first spliced) half's error is returned.
fn build_parts(
    src: Src,
    mut order: &mut [u32],
    opts: &BuildOptions,
    min_part: usize,
    forks: usize,
) -> Result<Vec<Part>> {
    let mut list = Vec::new();
    loop {
        let cur = std::mem::take(&mut order);
        let m = cur.len();
        if m < min_part || m <= opts.max_leaf_size as usize || forks >= MAX_FORK_DEPTH {
            push_part(&mut list, seq_part(src, cur, opts, min_part)?)?;
            return Ok(list);
        }
        let (bounds, centers) = range_bounds_chunked(src, cur)?;
        let mid = match opts.quality {
            Quality::Sah => sah_split(src, cur, &centers, opts.sah_bins as usize, true)?,
            Quality::Fast => morton_split(src, cur),
        };
        let left_first = mid <= m - mid;
        let (lo, hi) = cur.split_at_mut(mid);
        let (small, large) = if left_first { (lo, hi) } else { (hi, lo) };
        push_part(&mut list, Part::Split { bounds, left_first })?;
        if small.len() < min_part {
            push_part(&mut list, seq_part(src, small, opts, min_part)?)?;
            order = large;
            continue;
        }
        let (first, second) = par::join(
            true,
            || build_parts(src, small, opts, min_part, forks + 1),
            || build_parts(src, large, opts, min_part, forks + 1),
        );
        let (first, second) = both(first, second)?;
        try_reserve(&mut list, first.len() + second.len())?;
        list.extend(first);
        list.extend(second);
        return Ok(list);
    }
}

/// Both results, or the first one's error when both failed.
fn both<A, B>(a: Result<A>, b: Result<B>) -> Result<(A, B)> {
    Ok((a?, b?))
}

fn push_part(list: &mut Vec<Part>, part: Part) -> Result<()> {
    try_reserve(list, 1)?;
    list.push(part);
    Ok(())
}

/// The subtree over `order` built in one sequential pass into its own arrays.
fn seq_part(src: Src, order: &mut [u32], opts: &BuildOptions, min_part: usize) -> Result<Part> {
    let m = order.len();
    let expected = expected_nodes(m, opts.max_leaf_size);
    let (mut nodes, mut parents, mut refs) = (Vec::new(), Vec::new(), Vec::new());
    try_reserve_exact(&mut nodes, expected)?;
    try_reserve_exact(&mut parents, expected)?;
    try_reserve_exact(&mut refs, m)?;
    let root = push_node(&mut nodes, &mut parents, NONE)?;
    let mut out = Out {
        nodes: &mut nodes,
        parents: &mut parents,
        refs: &mut refs,
        tri_leaf: None,
    };
    build_seq(&mut out, src, order, opts, root, min_part)?;
    Ok(Part::Seq {
        nodes,
        parents,
        refs,
    })
}

/// Writes `parts` into `arena`. `pending` holds the (slot, parent) of the
/// subtree whose parts come next on top, starting with the root; it and the
/// arena have the capacity reserved. Iterative, so the call stack does not
/// grow with the tree's depth.
fn splice(arena: &mut Arena, parts: Vec<Part>, pending: &mut Vec<(u32, u32)>) {
    for part in parts {
        let (root, parent) = pending.pop().expect("a slot for every part");
        arena.parents[root as usize] = parent;
        match part {
            Part::Split { bounds, left_first } => {
                let left = arena.nodes.len() as u32;
                let right = left + 1;
                for _ in 0..2 {
                    arena.nodes.push(Node {
                        bounds: Aabb::EMPTY,
                        a: NONE,
                        b: LEAF_BIT,
                    });
                    arena.parents.push(root);
                }
                arena.nodes[root as usize] = Node {
                    bounds,
                    a: left,
                    b: right,
                };
                let (first, second) = if left_first {
                    (left, right)
                } else {
                    (right, left)
                };
                pending.push((second, root));
                pending.push((first, root));
            }
            Part::Seq {
                nodes,
                parents,
                refs,
            } => splice_seq(arena, &nodes, &parents, &refs, root),
        }
    }
}

/// Copies a sequential part into `arena` with its root at slot `root`.
fn splice_seq(arena: &mut Arena, nodes: &[Node], parents: &[u32], refs: &[u32], root: u32) {
    // Local node 0 goes to `root`, the others after the arena's end.
    let base = arena.nodes.len() as u32;
    let at = |k: u32| if k == 0 { root } else { base + k - 1 };
    let first_ref = arena.refs.len() as u32;
    for (k, &node) in nodes.iter().enumerate() {
        let here = at(k as u32);
        let node = if node.is_leaf() {
            for &t in &refs[node.leaf_range()] {
                arena.tri_leaf[t as usize] = here;
            }
            Node {
                a: node.a + first_ref,
                ..node
            }
        } else {
            Node {
                a: at(node.a),
                b: at(node.b),
                ..node
            }
        };
        if k == 0 {
            arena.nodes[root as usize] = node;
        } else {
            arena.nodes.push(node);
            arena.parents.push(at(parents[k]));
        }
    }
    arena.refs.extend_from_slice(refs);
}

/// Where [`build_seq`] writes: an arena's arrays, or a part's (no `tri_leaf`).
struct Out<'a> {
    nodes: &'a mut Vec<Node>,
    parents: &'a mut Vec<u32>,
    refs: &'a mut Vec<u32>,
    tri_leaf: Option<&'a mut Vec<u32>>,
}

/// Builds the subtree over `order` below the already allocated node `root`.
/// Nodes over at least `min_chunked` triangles use the chunked reductions,
/// which allocate. With `min_chunked = usize::MAX` this cannot fail when
/// `max_nodes(order.len()) - 1` more nodes and parents and `order.len()` refs
/// are reserved.
fn build_seq(
    out: &mut Out,
    src: Src,
    order: &mut [u32],
    opts: &BuildOptions,
    root: u32,
    min_chunked: usize,
) -> Result<()> {
    // Continue with the smaller child and push the larger: the stack never holds
    // more than log2(n) entries, so a fixed array is enough.
    let mut stack = [(0u32, 0usize, 0usize); 64];
    let mut depth = 0;
    let mut task = (root, 0, order.len());
    loop {
        let (node, start, end) = task;
        let chunked = end - start >= min_chunked;
        let (bounds, centers) = if chunked {
            range_bounds_chunked(src, &order[start..end])?
        } else {
            range_bounds(src, &order[start..end])
        };
        if end - start <= opts.max_leaf_size as usize {
            let first = out.refs.len() as u32;
            for &i in &order[start..end] {
                let id = src.ids[i as usize];
                out.refs.push(id);
                if let Some(tri_leaf) = &mut out.tri_leaf {
                    tri_leaf[id as usize] = node;
                }
            }
            out.nodes[node as usize] = Node::leaf(bounds, first, (end - start) as u32);
            if depth == 0 {
                return Ok(());
            }
            depth -= 1;
            task = stack[depth];
            continue;
        }
        let range = &mut order[start..end];
        let mid = start
            + match opts.quality {
                Quality::Sah => sah_split(src, range, &centers, opts.sah_bins as usize, chunked)?,
                Quality::Fast => morton_split(src, range),
            };
        let left = push_node(out.nodes, out.parents, node)?;
        let right = push_node(out.nodes, out.parents, node)?;
        out.nodes[node as usize] = Node {
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

fn push_node(nodes: &mut Vec<Node>, parents: &mut Vec<u32>, parent: u32) -> Result<u32> {
    if nodes.len() == nodes.capacity() {
        let more = nodes.len().max(16);
        try_reserve(nodes, more)?;
    }
    if parents.len() == parents.capacity() {
        let more = parents.len().max(16);
        try_reserve(parents, more)?;
    }
    let index = nodes.len() as u32;
    nodes.push(Node {
        bounds: Aabb::EMPTY,
        a: NONE,
        b: LEAF_BIT,
    });
    parents.push(parent);
    Ok(index)
}

/// Most SAH bins per axis (`BuildOptions::sah_bins` is at most 64).
const MAX_BINS: usize = 64;

/// Per-axis triangle counts and boxes of the SAH bins.
#[derive(Clone, Copy)]
struct Bins {
    counts: [[u32; MAX_BINS]; 3],
    boxes: [[Aabb; MAX_BINS]; 3],
}

impl Bins {
    const EMPTY: Bins = Bins {
        counts: [[0; MAX_BINS]; 3],
        boxes: [[Aabb::EMPTY; MAX_BINS]; 3],
    };

    fn add(&mut self, other: &Bins) {
        for axis in 0..3 {
            for b in 0..MAX_BINS {
                self.counts[axis][b] += other.counts[axis][b];
                self.boxes[axis][b].grow(&other.boxes[axis][b]);
            }
        }
    }
}

/// Bin of center coordinate `c` on an axis starting at `lo` with `extent`.
fn bin_of(c: f32, lo: f32, extent: f32, bins: usize) -> usize {
    (((c - lo) / extent) * bins as f32).clamp(0.0, (bins - 1) as f32) as usize
}

/// Bins the triangles in `order` on every axis where `usable`.
fn fill_bins(src: Src, order: &[u32], centers: &Aabb, bins: usize, usable: [bool; 3]) -> Bins {
    let mut out = Bins::EMPTY;
    for &i in order {
        let (c, bounds) = (src.centers[i as usize], &src.boxes[i as usize]);
        for axis in 0..3 {
            if usable[axis] {
                let lo = centers.min[axis];
                let b = bin_of(c[axis], lo, centers.max[axis] - lo, bins);
                out.counts[axis][b] += 1;
                out.boxes[axis][b].grow(bounds);
            }
        }
    }
    out
}

/// Chooses the cheapest binned SAH split and partitions `order`; returns the
/// split position within it. Falls back to the middle when no split separates
/// the centers. `chunked` bins in fixed chunks (on the pool when large), which
/// allocates; otherwise this never allocates and cannot fail.
fn sah_split(
    src: Src,
    order: &mut [u32],
    centers: &Aabb,
    bins: usize,
    chunked: bool,
) -> Result<usize> {
    let mid = order.len() / 2;
    let usable = [0, 1, 2].map(|axis| {
        let extent = centers.max[axis] - centers.min[axis];
        extent.is_finite() && extent > 0.0
    });
    let all = if chunked {
        allocation_allowed()?;
        let oom = || BvhError::OutOfMemory;
        let parts = par::map_chunks(order, PAR_CHUNK, PAR_MIN_PRIMS, oom, |_, c| {
            Ok(fill_bins(src, c, centers, bins, usable))
        })?;
        let mut all = Bins::EMPTY;
        parts.iter().for_each(|p| all.add(p));
        all
    } else {
        fill_bins(src, order, centers, bins, usable)
    };
    let mut best: Option<(f32, usize, usize)> = None; // (cost, axis, last left bin)
    for axis in (0..3).filter(|&axis| usable[axis]) {
        let (counts, boxes) = (&all.counts[axis], &all.boxes[axis]);
        // Sweep from the right to get the right-hand side areas and counts.
        let mut right_area = [0.0f32; MAX_BINS];
        let mut right_count = [0u32; MAX_BINS];
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
        return Ok(mid);
    };
    let lo = centers.min[axis];
    let extent = centers.max[axis] - lo;
    let in_left = |i: u32| bin_of(src.centers[i as usize][axis], lo, extent, bins) <= split_bin;
    let (mut i, mut j) = (0, order.len());
    while i < j {
        if in_left(order[i]) {
            i += 1;
        } else {
            j -= 1;
            order.swap(i, j);
        }
    }
    Ok(if i == 0 || i == order.len() { mid } else { i })
}

/// Splits a Morton-sorted `order` at the highest bit where its codes differ;
/// returns the split position within it.
fn morton_split(src: Src, order: &[u32]) -> usize {
    let key = |i: usize| src.keys[order[i] as usize];
    let (first, last) = (key(0), key(order.len() - 1));
    if first == last {
        return order.len() / 2;
    }
    let bit = 31 - (first ^ last).leading_zeros();
    let mask = 1u32 << bit;
    // Codes are sorted, so those with the bit set form a suffix.
    let (mut lo, mut hi) = (0, order.len() - 1);
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

    /// `n` triangles scattered by a fixed integer hash, with varied sizes.
    fn scatter(n: u32) -> Prims {
        let mut p = Prims::with_capacity(n as usize).unwrap();
        for i in 0..n {
            let h = |k: u32| (i.wrapping_mul(2654435761).wrapping_add(k * 40503) >> 8) % 1000;
            let c = [h(1), h(2), h(3)].map(|v| v as f32 * 0.1);
            let s = (h(4) % 7) as f32 * 0.05 + 0.01;
            p.push(i, Aabb::new(c, [c[0] + s, c[1] + s, c[2] + s]));
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

    fn empty(tris: usize) -> Arena {
        Arena {
            nodes: Vec::new(),
            parents: Vec::new(),
            refs: Vec::new(),
            tri_leaf: vec![NONE; tris],
        }
    }

    /// Asserts two arenas hold the same tree in the same slots.
    fn assert_same(a: &Arena, b: &Arena, tag: &str) {
        let flat = |a: &Arena| {
            a.nodes
                .iter()
                .map(|n| {
                    (
                        n.bounds.min.map(f32::to_bits),
                        n.bounds.max.map(f32::to_bits),
                        n.a,
                        n.b,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert!(flat(a) == flat(b), "{tag}: nodes differ");
        assert_eq!(a.parents, b.parents, "{tag}");
        assert_eq!(a.refs, b.refs, "{tag}");
        assert_eq!(a.tri_leaf, b.tri_leaf, "{tag}");
    }

    type Full = fn(&mut Arena, &mut Prims, &BuildOptions, usize) -> Result<u32>;

    /// The one-pass full build, the spliced parts and (while every node fits
    /// one reduction chunk) the treelet builder give the same layout: same
    /// nodes in the same slots, refs, parents and leaf map.
    #[test]
    fn spliced_parts_match_the_sequential_layout() {
        for quality in [Quality::Sah, Quality::Fast] {
            for (n, leaf, min_part) in [(37, 4, 9), (500, 3, 20), (3000, 8, 64), (3000, 1, 2)] {
                let opts = BuildOptions {
                    quality,
                    max_leaf_size: leaf,
                    ..Default::default()
                };
                for make in [row, scatter] {
                    let tag = format!("{quality:?} n={n} leaf={leaf} part={min_part}");
                    let mut classic = arena_for(n as usize, n as usize);
                    let r = build_subtree(&mut classic, &mut make(n), &opts, NONE).unwrap();
                    for full in [build_full_seq as Full, build_full_parts] {
                        let mut arena = empty(n as usize);
                        assert_eq!(full(&mut arena, &mut make(n), &opts, min_part).unwrap(), r);
                        assert_same(&classic, &arena, &tag);
                    }
                }
            }
        }
    }

    /// Several reduction chunks per node: the one-pass and spliced builds agree.
    #[test]
    fn chunked_builds_agree() {
        let n = 3 * PAR_CHUNK as u32 + 123;
        for quality in [Quality::Sah, Quality::Fast] {
            let opts = BuildOptions {
                quality,
                ..Default::default()
            };
            let (mut seq, mut parts) = (empty(n as usize), empty(n as usize));
            build_full_seq(&mut seq, &mut scatter(n), &opts, 1024).unwrap();
            build_full_parts(&mut parts, &mut scatter(n), &opts, 1024).unwrap();
            assert_same(&seq, &parts, &format!("{quality:?}"));
        }
    }

    /// Groups of `group` identical tiny triangles at x = -q^g, so SAH peels off
    /// a few groups per level: a split chain hundreds of levels deep.
    fn chain(groups: u32, group: u32) -> Prims {
        let mut p = Prims::with_capacity((groups * group) as usize).unwrap();
        let q = 1.0f32 - 2.0 / 16.0;
        for g in 0..groups {
            let x = -q.powi(g as i32);
            let s = x.abs() * 0.01;
            for k in 0..group {
                p.push(g * group + k, Aabb::new([x - s, -s, -s], [x + s, s, s]));
            }
        }
        p
    }

    fn depth(a: &Arena) -> usize {
        // Parents come before their children.
        let mut depth = vec![0usize; a.nodes.len()];
        for i in 1..a.nodes.len() {
            depth[i] = depth[a.parents[i] as usize] + 1;
        }
        depth.into_iter().max().unwrap_or(0)
    }

    /// Degenerate split chains, with both halves of every split at least
    /// `min_part`, build on a small stack both ways and agree.
    #[test]
    fn deep_chains_build_on_a_small_stack() {
        let (groups, min_part) = (600, 32);
        let n = (groups * 2 * min_part) as usize;
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || {
                let opts = BuildOptions::default();
                let mut seq = empty(n);
                build_full_seq(&mut seq, &mut chain(groups, 2 * min_part), &opts, 32).unwrap();
                let mut parts = empty(n);
                build_full_parts(&mut parts, &mut chain(groups, 2 * min_part), &opts, 32).unwrap();
                assert_same(&seq, &parts, "chain");
                assert!(depth(&seq) > 200, "depth {}", depth(&seq));
            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// Out of memory at any point of a full build fails cleanly, then the
    /// build succeeds with the same tree once memory is back.
    #[test]
    fn full_builds_fail_cleanly_out_of_memory() {
        use crate::error::fail_allocations_after;
        let n = 3000;
        for full in [build_full_seq as Full, build_full_parts] {
            let mut reference = empty(n as usize);
            full(
                &mut reference,
                &mut scatter(n),
                &BuildOptions::default(),
                64,
            )
            .unwrap();
            let mut failures = 0;
            for k in 0.. {
                let mut arena = empty(n as usize);
                let mut prims = scatter(n);
                fail_allocations_after(Some(k));
                let result = full(&mut arena, &mut prims, &BuildOptions::default(), 64);
                fail_allocations_after(None);
                match result {
                    Ok(_) => {
                        assert_same(&reference, &arena, "after failures");
                        break;
                    }
                    Err(e) => {
                        assert_eq!(e, BvhError::OutOfMemory);
                        failures += 1;
                    }
                }
            }
            assert!(failures > 10, "only {failures} allocations");
        }
    }

    #[test]
    fn the_first_error_wins() {
        let (a, b) = (BvhError::MeshMismatch, BvhError::OutOfMemory);
        assert_eq!(both::<(), ()>(Err(a.clone()), Err(b.clone())), Err(a));
        assert_eq!(both::<(), ()>(Ok(()), Err(b.clone())), Err(b));
        assert_eq!(both(Ok(1), Ok(2)), Ok((1, 2)));
    }
}
