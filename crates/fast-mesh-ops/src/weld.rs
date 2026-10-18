//! Grouping vertices that lie within a tolerance of each other.

use fast_mesh_store::{Mesh, VertexId};

use crate::topology::{UnionFind, point};
use crate::{OpsError, Result, filled, push, reserve};

/// For every vertex slot, the smallest id of its weld group: the alive
/// vertices linked through chains of pairs at most `tolerance` apart (in
/// local units, `f64` distance between the stored `f32` positions). A vertex
/// with no partner, and every dead slot, maps to itself. Tolerance 0 groups
/// exact duplicates only (`-0.0` equals `0.0`).
///
/// Vertices are binned on a uniform grid of cells `2 * tolerance` wide, so
/// that a pair within the tolerance lies in the same or adjacent cells with
/// a wide margin for the rounding of the cell coordinates; only those pairs
/// are measured. Cells are never narrower than the `f32` resolution of the
/// mesh extent: a tolerance below it cannot separate stored positions, and
/// cell coordinates would saturate and pile every vertex into one cell. Wider
/// cells only add candidates; the distance test decides.
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
    let extent = (0..slots as VertexId)
        .filter(|&v| mesh.is_vertex_alive(v))
        .flat_map(|v| mesh.positions()[v as usize])
        .fold(0.0f64, |m, x| m.max(f64::from(x).abs()));
    let cell = (2.0 * tol)
        .max(extent * f64::from(f32::EPSILON))
        .max(f64::MIN_POSITIVE);
    // Exact duplicates share a cell keyed by their coordinates' bits; adding
    // 0.0 turns -0.0 into 0.0.
    let cell_of = |v: VertexId| -> [i64; 3] {
        let p = mesh.positions()[v as usize];
        if tolerance == 0.0 {
            p.map(|x| i64::from((x + 0.0).to_bits()))
        } else {
            p.map(|x| (f64::from(x) / cell).floor() as i64)
        }
    };
    let mut cells: Vec<([i64; 3], VertexId)> = Vec::new();
    reserve(&mut cells, mesh.vertex_count())?;
    for v in 0..slots as VertexId {
        if mesh.is_vertex_alive(v) {
            push(&mut cells, (cell_of(v), v))?;
        }
    }
    cells.sort_unstable();
    let mut uf = UnionFind::new();
    uf.reset(slots)?;
    let reach: i64 = if tolerance == 0.0 { 0 } else { 1 };
    for &(c, v) in &cells {
        let p = point(mesh, v);
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                for dz in -reach..=reach {
                    let n = [
                        c[0].saturating_add(dx),
                        c[1].saturating_add(dy),
                        c[2].saturating_add(dz),
                    ];
                    let lo = cells.partition_point(|&(k, _)| k < n);
                    for &(_, u) in cells[lo..].iter().take_while(|&&(k, _)| k == n) {
                        if u <= v {
                            continue;
                        }
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
mod tests {
    use super::*;
    use crate::test_util::grid;

    #[test]
    fn tiny_tolerance_keeps_the_grid_usable() {
        // 40401 distinct vertices; with saturating cells every one of them
        // would land in the same cell and be compared with all the others.
        let mesh = grid(200);
        let roots = weld_roots(&mesh, 1e-30).unwrap();
        assert!(roots.iter().enumerate().all(|(v, &r)| r == v as VertexId));
    }

    #[test]
    fn tolerance_below_resolution_still_merges_exact_copies() {
        let mesh = Mesh::from_buffers(&[1e6, 0.0, 0.0, 1e6, 0.0, 0.0, 1e6 + 0.0625, 0.0, 0.0], &[])
            .unwrap();
        assert_eq!(weld_roots(&mesh, 1e-30).unwrap(), [0, 0, 2]);
    }
}
