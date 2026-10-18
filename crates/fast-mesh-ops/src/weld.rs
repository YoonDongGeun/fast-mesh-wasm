//! Grouping vertices that lie within a tolerance of each other.

use fast_mesh_store::{Mesh, VertexId};

use crate::topology::{UnionFind, point};
use crate::{OpsError, Result, filled, push, reserve};

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
/// there is over `2^36` cells, so it can only pair with an equal coordinate.
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
    let cell_of = |v: VertexId| -> [(bool, i64); 3] {
        mesh.positions()[v as usize].map(|x| {
            if f64::from(x).abs() >= exact_from {
                (true, i64::from((x + 0.0).to_bits()))
            } else {
                (false, (f64::from(x) / cell).floor() as i64)
            }
        })
    };
    let mut cells: Vec<([(bool, i64); 3], VertexId)> = Vec::new();
    reserve(&mut cells, mesh.vertex_count())?;
    for v in 0..slots as VertexId {
        if mesh.is_vertex_alive(v) && mesh.positions()[v as usize].iter().all(|x| x.is_finite()) {
            push(&mut cells, (cell_of(v), v))?;
        }
    }
    cells.sort_unstable();
    let mut uf = UnionFind::new();
    uf.reset(slots)?;
    // Exact coordinates are matched as they are; cell coordinates also in
    // the adjacent cells.
    let reach = |(exact, _): (bool, i64)| if exact { 0 } else { 1 };
    let shift = |(exact, k): (bool, i64), d: i64| (exact, k + d);
    for &(c, v) in &cells {
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
                            uf.union(v, u);
                        }
                    }
                }
            }
        }
    }
    let mut roots = filled(slots, 0)?;
    for (v, r) in roots.iter_mut().enumerate() {
        *r = uf.find(v as u32);
    }
    Ok(roots)
}

#[cfg(test)]
thread_local! {
    /// Test-only count of the vertex pairs whose distance was measured.
    static PAIRS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{grid, parts};

    /// `weld_roots` and the number of pairs it measured.
    fn counted(mesh: &Mesh, tolerance: f32) -> (Vec<VertexId>, usize) {
        PAIRS.with(|c| c.set(0));
        let roots = weld_roots(mesh, tolerance).unwrap();
        (roots, PAIRS.with(|c| c.get()))
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
