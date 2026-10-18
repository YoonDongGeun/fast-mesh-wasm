//! Grouping vertices that lie within a tolerance of each other.

use std::sync::atomic::{AtomicU32, Ordering};

use fast_mesh_store::{Mesh, StoreError, VertexId, par};

use crate::topology::{UnionFind, point};
use crate::{OpsError, Result, filled, reserve};

/// [`weld_roots`] keys the vertices and searches their neighbours in
/// parallel chunks of [`WELD_CHUNK`] from this many vertex slots, with the
/// `parallel` feature.
///
/// Measured on the i5-13600K through `weld` on terrains with every 10th
/// vertex duplicated: at 9k vertices parallel is even at tolerance 0 and 1.1
/// times faster at 0.01, at 18k 1.3 and 2 times faster, and at 73k 1.45
/// and 3.9 times. The sort stays sequential.
pub(crate) const PAR_MIN_WELD_VERTS: usize = 16 * 1024;

/// Vertex slots (or keyed vertices) per chunk of a parallel weld.
const WELD_CHUNK: usize = 8 * 1024;

/// A vertex's cell: per coordinate `(exact, key)`.
type Cell = [(bool, i64); 3];

/// For every vertex slot, the smallest id of its weld group: the alive
/// vertices linked through chains of pairs at most `tolerance` apart (in
/// local units, `f64` distance between the stored `f32` positions). A vertex
/// with no partner, every dead slot and every vertex with a non-finite
/// coordinate (never merged) map to themselves. Tolerance 0 groups exact
/// duplicates only (`-0.0` equals `0.0`).
///
/// Vertices are binned on a uniform grid of cells `2 * tolerance` wide, so
/// that a pair within the tolerance lies in the same or adjacent cells with
/// a wide margin for the rounding of the cell coordinates; only those pairs
/// are measured. The grid does not depend on the mesh extent, so one far
/// outlier does not widen the cells. A coordinate at least `2^60` cells from
/// zero (every coordinate at tolerance 0) is keyed by its exact value
/// instead, which keeps cell coordinates from saturating: its `f32` spacing
/// there is at least `2^36` cells, so it can only pair with an equal coordinate.
///
/// # Errors
/// [`OpsError::InvalidInput`] when `tolerance` is negative, NaN or infinite.
pub(crate) fn weld_roots(mesh: &Mesh, tolerance: f32) -> Result<Vec<VertexId>> {
    if !(tolerance >= 0.0 && tolerance.is_finite()) {
        return Err(OpsError::InvalidInput(
            "weld tolerance must be finite and not negative",
        ));
    }
    let slots = mesh.vertex_slots();
    let tol = f64::from(tolerance);
    let cell = 2.0 * tol;
    let exact_from = cell * (1u64 << 60) as f64;
    // Per coordinate `(exact, key)`: the exact value's bits (adding 0.0
    // turns -0.0 into 0.0), or the cell coordinate, below `2^60` in size.
    let cell_of = |v: VertexId| -> Cell {
        mesh.positions()[v as usize].map(|x| {
            if f64::from(x).abs() >= exact_from {
                (true, i64::from((x + 0.0).to_bits()))
            } else {
                (false, (f64::from(x) / cell).floor() as i64)
            }
        })
    };
    let parallel = par::enabled() && !par::force_sequential() && slots >= PAR_MIN_WELD_VERTS;
    let keep = |v: VertexId| {
        mesh.is_vertex_alive(v) && mesh.positions()[v as usize].iter().all(|x| x.is_finite())
    };
    // The alive vertices with finite coordinates, keyed, in slot order.
    let mut cells: Vec<(Cell, VertexId)> = Vec::new();
    if parallel {
        let oom = || OpsError::Store(StoreError::OutOfMemory);
        let keyed = par::map_range(slots, WELD_CHUNK, 0, oom, |r| {
            let mut part: Vec<(Cell, VertexId)> = Vec::new();
            reserve(&mut part, r.len())?;
            for v in r.start as VertexId..r.end as VertexId {
                if keep(v) {
                    part.push((cell_of(v), v));
                }
            }
            Ok(part)
        })?;
        reserve(&mut cells, keyed.iter().map(Vec::len).sum())?;
        for part in &keyed {
            cells.extend_from_slice(part);
        }
    } else {
        reserve(&mut cells, mesh.vertex_count())?;
        for v in 0..slots as VertexId {
            if keep(v) {
                cells.push((cell_of(v), v));
            }
        }
    }
    // The keys are distinct (each holds its vertex), so any sort gives
    // this order.
    cells.sort_unstable();
    if parallel {
        // Chunks of the sorted vertices search their neighbours on the
        // pool and join each close pair at once, in a shared union-find
        // that always links the larger root under the smaller: its groups,
        // and so the roots (each group's smallest id), do not depend on the
        // order of the unions. It takes no memory beyond its parents.
        let uf = AtomicUnionFind::new(slots)?;
        let oom = || OpsError::Store(StoreError::OutOfMemory);
        par::map_range(cells.len(), WELD_CHUNK, 0, oom, |r| {
            close_pairs(mesh, &cells, tol, r, |v, u| {
                uf.union(v, u);
                Ok(())
            })
        })?;
        let mut roots = filled(slots, 0)?;
        for (v, r) in roots.iter_mut().enumerate() {
            *r = uf.find(v as u32);
        }
        return Ok(roots);
    }
    let mut uf = UnionFind::new();
    uf.reset(slots)?;
    close_pairs(mesh, &cells, tol, 0..cells.len(), |v, u| {
        uf.union(v, u);
        Ok(())
    })?;
    let mut roots = filled(slots, 0)?;
    for (v, r) in roots.iter_mut().enumerate() {
        *r = uf.find(v as u32);
    }
    Ok(roots)
}

/// A union-find that threads share, for the parallel neighbour search of
/// [`weld_roots`]. Every link points from a root to a smaller id, so the
/// links form a forest and each group's root is its smallest member, however
/// the unions interleave.
struct AtomicUnionFind {
    parent: Vec<AtomicU32>,
}

impl AtomicUnionFind {
    /// `n` singletons.
    fn new(n: usize) -> Result<Self> {
        let mut parent = Vec::new();
        reserve(&mut parent, n)?;
        parent.extend((0..n as u32).map(AtomicU32::new));
        Ok(AtomicUnionFind { parent })
    }

    /// The root of `x`'s tree, halving the path on the way: each step may
    /// point a vertex at its grandparent, a smaller id in the same tree.
    fn find(&self, mut x: u32) -> u32 {
        loop {
            let p = self.parent[x as usize].load(Ordering::Acquire);
            if p == x {
                return x;
            }
            let g = self.parent[p as usize].load(Ordering::Acquire);
            if g != p {
                let _ = self.parent[x as usize].compare_exchange(
                    p,
                    g,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
            }
            x = g;
        }
    }

    /// Joins the groups of `a` and `b`: links the larger root under the
    /// smaller, retrying if another thread linked it first.
    fn union(&self, a: u32, b: u32) {
        loop {
            let (ra, rb) = (self.find(a), self.find(b));
            if ra == rb {
                return;
            }
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            if self.parent[hi as usize]
                .compare_exchange(hi, lo, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }
}

/// Calls `found(v, u)` for each pair of the vertices `cells[range]` (`v`)
/// and a vertex `u > v` in the same or an adjacent cell of the sorted
/// `cells`, at most `tol` apart; stops at the first error.
fn close_pairs(
    mesh: &Mesh,
    cells: &[(Cell, VertexId)],
    tol: f64,
    range: std::ops::Range<usize>,
    mut found: impl FnMut(VertexId, VertexId) -> Result<()>,
) -> Result<()> {
    // Exact coordinates are matched as they are; cell coordinates also in
    // the adjacent cells.
    let reach = |(exact, _): (bool, i64)| if exact { 0 } else { 1 };
    let shift = |(exact, k): (bool, i64), d: i64| (exact, k + d);
    for &(c, v) in &cells[range] {
        let p = point(mesh, v);
        for dx in -reach(c[0])..=reach(c[0]) {
            for dy in -reach(c[1])..=reach(c[1]) {
                for dz in -reach(c[2])..=reach(c[2]) {
                    let n = [shift(c[0], dx), shift(c[1], dy), shift(c[2], dz)];
                    let lo = cells.partition_point(|&(k, _)| k < n);
                    for &(_, u) in cells[lo..].iter().take_while(|&&(k, _)| k == n) {
                        if u <= v {
                            continue;
                        }
                        #[cfg(test)]
                        PAIRS.with(|c| c.set(c.get() + 1));
                        let q = point(mesh, u);
                        let d2 = (0..3).map(|k| (q[k] - p[k]) * (q[k] - p[k])).sum::<f64>();
                        if d2 <= tol * tol {
                            found(v, u)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    /// Test-only count of the vertex pairs whose distance was measured, on
    /// the calling thread only: tests that read it run sequentially.
    static PAIRS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::par;

    use super::*;
    use crate::test_util::{assert_parallel_identical, grid, parts};

    /// `weld_roots` and the number of pairs it measured, run sequentially
    /// ([`PAIRS`] counts only the calling thread).
    fn counted(mesh: &Mesh, tolerance: f32) -> (Vec<VertexId>, usize) {
        PAIRS.with(|c| c.set(0));
        par::set_force_sequential(true);
        let roots = weld_roots(mesh, tolerance).unwrap();
        par::set_force_sequential(false);
        (roots, PAIRS.with(|c| c.get()))
    }

    #[test]
    fn dense_cluster_bit_identical() {
        // 2,000 vertices within 0.002 of one point off the grid, all within
        // the tolerance of each other: about 2 million close pairs, which
        // the parallel search joins as it finds them instead of storing.
        let n = 150;
        let g = (n + 1) * (n + 1);
        let cluster: Vec<[f32; 3]> = (0..2000)
            .map(|i| {
                let d = |k: u32| 0.001 * ((i * k) % 17) as f32 / 17.0;
                [75.3 + d(3), 75.3 + d(5), 0.5 + d(7)]
            })
            .collect();
        let mesh = grid_and(n, &cluster);
        assert!(mesh.vertex_count() >= PAR_MIN_WELD_VERTS);
        let roots =
            assert_parallel_identical("weld_roots cluster", || weld_roots(&mesh, 0.01).unwrap());
        let (grid_roots, cluster_roots) = roots.split_at(g as usize);
        assert!(
            grid_roots
                .iter()
                .enumerate()
                .all(|(v, &r)| r == v as VertexId)
        );
        assert!(cluster_roots.iter().all(|&r| r == g));
    }

    #[test]
    fn weld_bit_identical() {
        // A grid of spacing 1 with every 7th vertex copied, nudged by up to
        // 0.01, and a few outliers; every 5th copy is then removed, leaving a
        // dead slot (never merged) where it was.
        let n = 300;
        let base = grid(n);
        let extra: Vec<[f32; 3]> = base
            .positions()
            .iter()
            .step_by(7)
            .enumerate()
            .map(|(i, p)| {
                [
                    p[0] + 0.001 * (i % 11) as f32,
                    p[1],
                    p[2] - 0.0005 * (i % 3) as f32,
                ]
            })
            .chain([[1e30, 0.0, 0.0], [f32::NAN, 0.0, 0.0], [1e30, 0.0, 0.0]])
            .collect();
        let mut mesh = grid_and(n, &extra);
        let g = (n + 1) * (n + 1);
        let dead: Vec<VertexId> = (g..g + extra.len() as u32).step_by(5).collect();
        let mut tx = mesh.begin_edit();
        tx.remove_vertices(&dead).unwrap();
        tx.commit().unwrap();
        assert!(mesh.vertex_count() >= PAR_MIN_WELD_VERTS);
        for tolerance in [0.0, 0.004, 0.02, 0.6] {
            let roots =
                assert_parallel_identical("weld_roots", || weld_roots(&mesh, tolerance).unwrap());
            let merged = roots
                .iter()
                .enumerate()
                .filter(|&(v, &r)| r != v as VertexId)
                .count();
            assert!(merged > 0, "{tolerance}");
            // Dead slots map to themselves and no alive vertex joins them.
            for (v, &r) in roots.iter().enumerate() {
                assert_eq!(
                    dead.binary_search(&r).is_ok(),
                    dead.binary_search(&(v as VertexId)).is_ok()
                );
            }
        }
    }

    /// [`grid`]`(n)` with extra vertices at `extra`, after the grid's.
    fn grid_and(n: u32, extra: &[[f32; 3]]) -> Mesh {
        let (mut positions, triangles) = parts(&grid(n));
        positions.extend(extra.iter().flatten());
        let indices: Vec<u32> = triangles.into_iter().flatten().collect();
        Mesh::from_buffers(&positions, &indices).unwrap()
    }

    #[test]
    fn far_outlier_keeps_the_grid_linear() {
        // One vertex at 1e30 once widened every cell past the whole grid, so
        // that all 19881 grid vertices were measured against each other.
        let n = 140;
        let g = (n + 1) * (n + 1);
        let mesh = grid_and(n, &[[1e30, 0.0, 0.0], [3.004, 5.0, 0.0], [1e30, 0.0, 0.0]]);
        for tolerance in [0.01, 1e-30] {
            let (roots, pairs) = counted(&mesh, tolerance);
            assert!(pairs < 4 * g as usize, "{pairs} pairs measured");
            let merged: Vec<(usize, VertexId)> = roots
                .iter()
                .enumerate()
                .filter(|&(v, &r)| r != v as VertexId)
                .map(|(v, &r)| (v, r))
                .collect();
            let near = if tolerance == 0.01 {
                vec![(g as usize + 1, 5 * (n + 1) + 3)]
            } else {
                vec![]
            };
            // The two outliers are exact copies of each other.
            let expected = [near, vec![(g as usize + 2, g)]].concat();
            assert_eq!(merged, expected);
        }
    }

    #[test]
    fn non_finite_vertices_are_never_merged() {
        let inf = f32::INFINITY;
        let n = 100;
        let g = (n + 1) * (n + 1);
        let mesh = grid_and(
            n,
            &[
                [inf, 0.0, 0.0],
                [inf, 0.0, 0.0],
                [f32::NAN, 1.0, 1.0],
                [f32::NAN, 1.0, 1.0],
                [7.0, 2.0, 0.0],
            ],
        );
        for tolerance in [0.0, 0.25] {
            let (roots, pairs) = counted(&mesh, tolerance);
            assert!(pairs < 4 * g as usize, "{pairs} pairs measured");
            let merged: Vec<(usize, VertexId)> = roots
                .iter()
                .enumerate()
                .filter(|&(v, &r)| r != v as VertexId)
                .map(|(v, &r)| (v, r))
                .collect();
            assert_eq!(merged, [(g as usize + 4, 2 * (n + 1) + 7)]);
        }
    }

    #[test]
    fn tiny_tolerance_keeps_the_grid_usable() {
        // 40401 distinct vertices; with saturating cells every one of them
        // would land in the same cell and be compared with all the others.
        let mesh = grid(200);
        let (roots, pairs) = counted(&mesh, 1e-30);
        assert!(roots.iter().enumerate().all(|(v, &r)| r == v as VertexId));
        assert_eq!(pairs, 0);
    }

    #[test]
    fn tolerance_below_resolution_still_merges_exact_copies() {
        let mesh = Mesh::from_buffers(&[1e6, 0.0, 0.0, 1e6, 0.0, 0.0, 1e6 + 0.0625, 0.0, 0.0], &[])
            .unwrap();
        assert_eq!(weld_roots(&mesh, 1e-30).unwrap(), [0, 0, 2]);
    }
}
