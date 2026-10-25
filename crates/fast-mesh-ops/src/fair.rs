//! Fairing a refined hole patch.
//!
//! [`fair`] moves the new vertices of a [`RefinedPatch`] so that the patch
//! minimises bending and continues the curvature around it: it solves the
//! discrete bi-Laplacian `L(L x) = 0` for the new vertices, with the loop
//! vertices and the ring of mesh vertices just outside the loop fixed, by a
//! sparse direct solve of the normal equations.

use fast_mesh_store::{Mesh, VertexId};

use crate::fill::{FailReason, FairWeights};
use crate::refine::RefinedPatch;
use crate::sparse;
use crate::topology::{VertexFans, cross, dot, point, sub};
use crate::{Result, filled, push, reserve};

/// A solve counts as converged once its residual is this small relative to
/// the right-hand side.
const TOLERANCE: f64 = 1e-10;

/// Cotangent weights are clamped to at least this times their mean.
const CLAMP: f64 = 1e-6;

/// A corner whose triangle's doubled area is at most this times the product
/// of its two edges adds no cotangent weight.
const DEGENERATE: f64 = 1e-12;

/// How fairing went.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FairResult {
    /// Whether every coordinate's solve reached the tolerance.
    pub(crate) converged: bool,
    /// Solver iterations, the most of the three coordinates (0: the direct
    /// solve takes none).
    pub(crate) iterations: usize,
    /// The weights of the solution kept.
    pub(crate) weights: FairWeights,
}

/// Moves the new vertices of `patch` (a refinement of a loop of `mesh`;
/// `fans` are the mesh's vertex fans) to minimise bending.
///
/// - The energy is `Σ_k (L x)_k² / A_k` over the patch vertices `k` (loop
///   and new), where `L` is the Laplacian of the patch plus the mesh
///   triangles around the loop and `A_k` is a third of the area around `k`.
///   Its minimiser over the new vertices solves the bi-Laplacian `L(L x) = 0`
///   with the loop vertices and their mesh neighbours outside the loop fixed;
///   the normal equations are symmetric positive definite.
/// - `L` has cotangent weights `(cot α + cot β) / 2`, clamped to at least
///   `1e-6` of their mean. When that solve does not converge or is not
///   finite, it is retried with uniform weights (and unit areas).
/// - The normal equations are factored once, as a sparse LDLᵀ, and solved
///   for the three coordinates; a coordinate counts as converged at `1e-10`
///   relative to its right-hand side.
/// - The solution kept is the first converged one; when neither converges,
///   the finite one with the smaller relative residual, `converged: false`.
///   When neither is finite, the loop fails with [`FailReason::Solver`] and
///   `patch` is left as it was.
///
/// # Errors
/// [`OpsError::Store`](crate::OpsError::Store) when memory runs out.
pub(crate) fn fair(
    mesh: &Mesh,
    fans: &VertexFans,
    patch: &mut RefinedPatch,
) -> Result<std::result::Result<FairResult, FailReason>> {
    let system = System::build(mesh, fans, patch)?;
    let cotangent = system.solve(patch, FairWeights::Cotangent)?;
    let Some(chosen) = choose(cotangent, || system.solve(patch, FairWeights::Uniform))? else {
        return Ok(Err(FailReason::Solver));
    };
    let first = patch.old.len();
    patch.points[first..].copy_from_slice(&chosen.x);
    Ok(Ok(FairResult {
        converged: chosen.converged,
        iterations: chosen.iterations,
        weights: chosen.weights,
    }))
}

/// The solution to keep: the cotangent one when it converged, else the
/// uniform one (solved only then) when that converged, else the finite one
/// with the smaller residual; `None` when neither is finite.
fn choose(
    cotangent: Option<Attempt>,
    uniform: impl FnOnce() -> Result<Option<Attempt>>,
) -> Result<Option<Attempt>> {
    if cotangent.as_ref().is_some_and(|c| c.converged) {
        return Ok(cotangent);
    }
    Ok(match (cotangent, uniform()?) {
        (_, Some(u)) if u.converged => Some(u),
        (Some(c), Some(u)) => Some(if c.residual <= u.residual { c } else { u }),
        (c, u) => c.or(u),
    })
}

/// One finite solve.
struct Attempt {
    /// The new vertices' positions.
    x: Vec<[f64; 3]>,
    converged: bool,
    iterations: usize,
    /// The largest relative residual of the three coordinates.
    residual: f64,
    weights: FairWeights,
}

/// The geometry of the fairing problem in local indices: the patch's (old,
/// then new), then the ring of mesh vertices around the loop.
struct System {
    /// Local indices below this are patch vertices, whose Laplacian rows
    /// enter the energy.
    patch_len: usize,
    /// Local indices from this up to `patch_len` are free (new vertices).
    first_free: usize,
    /// Every local position.
    points: Vec<[f64; 3]>,
    /// Every edge with a patch endpoint, sorted, with its cotangent weight.
    edges: Vec<((u32, u32), f64)>,
    /// A third of the area around each patch vertex.
    mass: Vec<f64>,
    /// The nested-dissection elimination order of the free vertices: entry
    /// `p` is the free index (local minus `first_free`) eliminated at
    /// position `p`.
    order: Vec<u32>,
}

impl System {
    fn build(mesh: &Mesh, fans: &VertexFans, patch: &RefinedPatch) -> Result<System> {
        let old = &patch.old;
        let patch_len = patch.points.len();
        // The mesh triangles around the loop and the ring of their vertices
        // outside it.
        let mut around = Vec::new();
        let mut ring: Vec<VertexId> = Vec::new();
        for &v in old {
            let fan = fans.fan(v);
            reserve(&mut around, fan.len())?;
            around.extend_from_slice(fan);
            reserve(&mut ring, 2 * fan.len())?;
            for &t in fan {
                ring.extend(
                    mesh.triangles()[t as usize]
                        .iter()
                        .filter(|u| old.binary_search(u).is_err()),
                );
            }
        }
        around.sort_unstable();
        around.dedup();
        ring.sort_unstable();
        ring.dedup();
        let local = |v: VertexId| match old.binary_search(&v) {
            Ok(i) => i as u32,
            Err(_) => {
                let i = ring
                    .binary_search(&v)
                    .expect("vertex outside the loop is in the ring");
                (patch_len + i) as u32
            }
        };
        let mut points = Vec::new();
        reserve(&mut points, patch_len + ring.len())?;
        points.extend_from_slice(&patch.points);
        points.extend(ring.iter().map(|&v| point(mesh, v)));

        let mut mass = filled(patch_len, 0.0)?;
        let mut corners: Vec<((u32, u32), f64)> = Vec::new();
        reserve(&mut corners, 3 * (patch.triangles.len() + around.len()))?;
        let mesh_tris = around
            .iter()
            .map(|&t| mesh.triangles()[t as usize].map(local));
        for t in patch.triangles.iter().copied().chain(mesh_tris) {
            let p = t.map(|v| points[v as usize]);
            let doubled = cross(sub(p[1], p[0]), sub(p[2], p[0]));
            let area = 0.5 * dot(doubled, doubled).sqrt();
            for r in 0..3 {
                let (a, b, c) = (t[r], t[(r + 1) % 3], t[(r + 2) % 3]);
                if (a as usize) < patch_len {
                    mass[a as usize] += area / 3.0;
                }
                if b as usize >= patch_len && c as usize >= patch_len {
                    continue;
                }
                let (u, w) = (sub(p[(r + 1) % 3], p[r]), sub(p[(r + 2) % 3], p[r]));
                let sin = dot(cross(u, w), cross(u, w)).sqrt();
                let cot = if sin > DEGENERATE * (dot(u, u) * dot(w, w)).sqrt() {
                    dot(u, w) / sin
                } else {
                    0.0
                };
                corners.push(((b.min(c), b.max(c)), 0.5 * cot));
            }
        }
        // A total order, so the sums below do not depend on the sort.
        corners.sort_unstable_by(|x, y| x.0.cmp(&y.0).then(x.1.total_cmp(&y.1)));
        let mut edges: Vec<((u32, u32), f64)> = Vec::new();
        for (k, w) in corners {
            match edges.last_mut() {
                Some(last) if last.0 == k => last.1 += w,
                _ => push(&mut edges, (k, w))?,
            }
        }
        // The graph of BᵀB for the nested-dissection order, as free
        // indices, each edge once: B's row k holds the free neighbours of
        // patch vertex k (and k itself when free), so every pair of them
        // shares the row.
        let first = old.len();
        let mut free_nbrs: Vec<Vec<u32>> = Vec::new();
        reserve(&mut free_nbrs, patch_len)?;
        free_nbrs.resize(patch_len, Vec::new());
        for &((a, b), _) in &edges {
            for (k, j) in [(a, b), (b, a)] {
                if (k as usize) < patch_len && (first..patch_len).contains(&(j as usize)) {
                    push(&mut free_nbrs[k as usize], (j as usize - first) as u32)?;
                }
            }
        }
        let mut nd_pairs: Vec<u64> = Vec::new();
        for (k, nbrs) in free_nbrs.iter().enumerate() {
            for (i, &u) in nbrs.iter().enumerate() {
                for &v in &nbrs[i + 1..] {
                    push(&mut nd_pairs, (u.min(v) as u64) << 32 | u.max(v) as u64)?;
                }
            }
            if k >= first {
                let this = (k - first) as u32;
                for &u in nbrs {
                    if u != this {
                        push(
                            &mut nd_pairs,
                            (u.min(this) as u64) << 32 | u.max(this) as u64,
                        )?;
                    }
                }
            }
        }
        nd_pairs.sort_unstable();
        nd_pairs.dedup();
        let mut nd_edges: Vec<(u32, u32)> = Vec::new();
        reserve(&mut nd_edges, nd_pairs.len())?;
        nd_edges.extend(nd_pairs.iter().map(|&p| ((p >> 32) as u32, p as u32)));
        let order =
            sparse::nested_dissection(patch_len - first, &patch.points[first..], &nd_edges)?;
        Ok(System {
            patch_len,
            first_free: first,
            points,
            edges,
            mass,
            order,
        })
    }

    /// Solves with `weights`; `None` when the weights or the solution are
    /// not finite.
    fn solve(&self, patch: &RefinedPatch, weights: FairWeights) -> Result<Option<Attempt>> {
        let (np, first) = (self.patch_len, self.first_free);
        let free = np - first;
        let cotangent = weights == FairWeights::Cotangent;
        let floor = if cotangent {
            let mean = self.edges.iter().map(|e| e.1).sum::<f64>() / self.edges.len() as f64;
            if !(mean.is_finite() && mean > 0.0) {
                return Ok(None);
            }
            CLAMP * mean
        } else {
            0.0
        };
        let weight = |w: f64| if cotangent { w.max(floor) } else { 1.0 };

        // The Laplacian row of each patch vertex `k`, scaled by `1 / √A_k`,
        // split into the free columns (a sparse matrix `B`, rows by `k`) and
        // the fixed part `c` per coordinate: the energy is `|B x + c|²`.
        let mut count = filled(np + 1, 0u32)?;
        let mut diagonal = filled(np, 0.0f64)?;
        for &((a, b), w) in &self.edges {
            for (k, j) in [(a, b), (b, a)] {
                if (k as usize) < np {
                    diagonal[k as usize] += weight(w);
                    if (first..np).contains(&(j as usize)) {
                        count[k as usize + 1] += 1;
                    }
                }
            }
        }
        for k in first..np {
            count[k + 1] += 1;
        }
        for k in 0..np {
            count[k + 1] += count[k];
        }
        let entries = count[np] as usize;
        let mut cols = filled(entries, 0u32)?;
        let mut vals = filled(entries, 0.0f64)?;
        let mut fixed = filled(np, [0.0f64; 3])?;
        let mut next = filled(np, 0u32)?;
        next.copy_from_slice(&count[..np]);
        let scale = |k: usize| {
            if cotangent {
                1.0 / self.mass[k].sqrt()
            } else {
                1.0
            }
        };
        for &((a, b), w) in &self.edges {
            for (k, j) in [(a, b), (b, a)] {
                let (k, j) = (k as usize, j as usize);
                if k >= np {
                    continue;
                }
                let v = weight(w) * scale(k);
                if (first..np).contains(&j) {
                    let at = next[k] as usize;
                    cols[at] = (j - first) as u32;
                    vals[at] = v;
                    next[k] += 1;
                } else {
                    let p = self.points[j];
                    for d in 0..3 {
                        fixed[k][d] += v * p[d];
                    }
                }
            }
        }
        for k in 0..np {
            let v = -diagonal[k] * scale(k);
            if k >= first {
                let at = next[k] as usize;
                cols[at] = (k - first) as u32;
                vals[at] = v;
            } else {
                let p = self.points[k];
                for d in 0..3 {
                    fixed[k][d] += v * p[d];
                }
            }
        }
        if vals
            .iter()
            .chain(fixed.iter().flatten())
            .any(|x| !x.is_finite())
        {
            return Ok(None);
        }
        // The right-hand sides: Bᵀ(-c) per coordinate.
        let mut rhs = [
            filled(free, 0.0f64)?,
            filled(free, 0.0f64)?,
            filled(free, 0.0f64)?,
        ];
        for k in 0..np {
            let f = fixed[k];
            for t in count[k] as usize..count[k + 1] as usize {
                let (c, v) = (cols[t] as usize, vals[t]);
                rhs[0][c] -= v * f[0];
                rhs[1][c] -= v * f[1];
                rhs[2][c] -= v * f[2];
            }
        }
        // Factored once, solved for the three coordinates.
        let Some(factor) = sparse::factor_ordered(&count, &cols, &vals, free, &self.order)? else {
            return Ok(None);
        };
        let mut xs = [
            filled(free, 0.0f64)?,
            filled(free, 0.0f64)?,
            filled(free, 0.0f64)?,
        ];
        for d in 0..3 {
            factor.solve(&rhs[d], &mut xs[d])?;
        }
        // Each coordinate judged by its true residual: relative to the
        // right-hand side, or to the start iterate's residual when that is
        // zero (a zero right-hand side has the solution 0).
        let mut bx = filled(np, 0.0f64)?;
        let mut ax = filled(free, 0.0f64)?;
        let mut x0 = filled(free, 0.0f64)?;
        let mut x = filled(free, [0.0f64; 3])?;
        let (mut converged, mut residual) = (true, 0.0f64);
        for d in 0..3 {
            let res = true_residual(&count, &cols, &vals, &xs[d], &rhs[d], &mut bx, &mut ax);
            let rhs_norm = norm(&rhs[d]);
            let reference = if rhs_norm > 0.0 {
                rhs_norm
            } else {
                for (i, xi) in x0.iter_mut().enumerate() {
                    *xi = patch.points[first + i][d];
                }
                true_residual(&count, &cols, &vals, &x0, &rhs[d], &mut bx, &mut ax)
            };
            if !(xs[d].iter().all(|x| x.is_finite()) && res.is_finite()) {
                return Ok(None);
            }
            converged &= res <= TOLERANCE * reference;
            residual = residual.max(if reference > 0.0 {
                res / reference
            } else {
                0.0
            });
            for (xi, v) in x.iter_mut().zip(&xs[d]) {
                xi[d] = *v;
            }
        }
        Ok(Some(Attempt {
            x,
            converged,
            iterations: 0,
            residual,
            weights,
        }))
    }
}

/// `|rhs - Bᵀ B x|`, with `B` in compressed rows; `bx` (one entry per row
/// of `B`) and `ax` (one per column) are scratch.
fn true_residual(
    starts: &[u32],
    cols: &[u32],
    vals: &[f64],
    x: &[f64],
    rhs: &[f64],
    bx: &mut [f64],
    ax: &mut [f64],
) -> f64 {
    for (k, o) in bx.iter_mut().enumerate() {
        let mut s = 0.0;
        for t in starts[k] as usize..starts[k + 1] as usize {
            s += vals[t] * x[cols[t] as usize];
        }
        *o = s;
    }
    ax.fill(0.0);
    for (k, s) in bx.iter().enumerate() {
        for t in starts[k] as usize..starts[k + 1] as usize {
            ax[cols[t] as usize] += vals[t] * s;
        }
    }
    let mut s = 0.0;
    for (r, a) in rhs.iter().zip(ax.iter()) {
        s += (r - a) * (r - a);
    }
    s.sqrt()
}

/// The Euclidean norm.
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary_loops;
    use crate::fill_dp::triangulate_loop;
    use crate::refine::refine;
    use crate::test_util::*;
    use crate::topology::EdgeMap;

    /// The refined patch of the shortest loop of `mesh` (the hole, not a
    /// grid's border), with the mesh's fans.
    fn refined(mesh: &Mesh) -> (VertexFans, RefinedPatch) {
        let loops = boundary_loops(mesh).unwrap();
        let l = loops.iter().min_by_key(|l| l.vertices.len()).unwrap();
        let edges = EdgeMap::build(mesh).unwrap();
        let fans = VertexFans::build(mesh).unwrap();
        let tris = triangulate_loop(mesh, &l.vertices, &edges, 200, None)
            .unwrap()
            .unwrap();
        let patch = refine(mesh, &fans, &edges, &l.vertices, &tris, None).unwrap();
        (fans, patch)
    }

    /// [`grid_without`]`(8)` missing its middle 4 x 4 cells, tilted onto the
    /// plane `z = 0.3 x + 0.2 y`.
    fn tilted_hole() -> Mesh {
        let cells: Vec<(u32, u32)> = (2..6).flat_map(|y| (2..6).map(move |x| (x, y))).collect();
        let (mut positions, triangles) = parts(&grid_without(8, &cells));
        for p in positions.chunks_mut(3) {
            p[2] = 0.3 * p[0] + 0.2 * p[1];
        }
        mesh_of(&positions, &triangles)
    }

    #[test]
    fn tilted_plane_is_reproduced() {
        // Fairing keeps a planar hole in its plane with either weights.
        let mesh = tilted_hole();
        for weights in [FairWeights::Cotangent, FairWeights::Uniform] {
            let (fans, mut patch) = refined(&mesh);
            assert!(patch.new_vertices() > 0);
            let system = System::build(&mesh, &fans, &patch).unwrap();
            let a = system.solve(&patch, weights).unwrap().unwrap();
            assert!(a.converged && a.residual <= TOLERANCE);
            let first = patch.old.len();
            patch.points[first..].copy_from_slice(&a.x);
            for p in &patch.points[first..] {
                assert!((p[2] - 0.3 * p[0] - 0.2 * p[1]).abs() < 1e-6, "{p:?}");
                assert!((2.0..=6.0).contains(&p[0]) && (2.0..=6.0).contains(&p[1]));
            }
        }
        let (fans, mut patch) = refined(&mesh);
        let r = fair(&mesh, &fans, &mut patch).unwrap().unwrap();
        assert_eq!((r.converged, r.weights), (true, FairWeights::Cotangent));
        assert_eq!(r.iterations, 0);
    }

    fn attempt(converged: bool, residual: f64, weights: FairWeights) -> Option<Attempt> {
        Some(Attempt {
            x: Vec::new(),
            converged,
            iterations: 1,
            residual,
            weights,
        })
    }

    fn chosen(c: Option<Attempt>, u: Option<Attempt>) -> Option<(FairWeights, bool)> {
        choose(c, || Ok(u))
            .unwrap()
            .map(|a| (a.weights, a.converged))
    }

    #[test]
    fn direct_solve_takes_no_iterations() {
        // The direct solve factors once and solves: no iterations, converged.
        let mesh = tilted_hole();
        let (fans, mut patch) = refined(&mesh);
        assert!(patch.new_vertices() > 0);
        let r = fair(&mesh, &fans, &mut patch).unwrap().unwrap();
        assert_eq!((r.converged, r.iterations), (true, 0));
    }

    #[test]
    fn falls_back_to_uniform_weights() {
        use FairWeights::{Cotangent as C, Uniform as U};
        // A converged cotangent solve never runs the uniform one.
        let kept = choose(attempt(true, 0.0, C), || panic!("solved uniform")).unwrap();
        assert_eq!(kept.map(|a| a.weights), Some(C));
        assert_eq!(
            chosen(attempt(false, 1e-3, C), attempt(true, 0.0, U)),
            Some((U, true))
        );
        assert_eq!(chosen(None, attempt(true, 0.0, U)), Some((U, true)));
        // Neither converged: the smaller residual, flagged.
        assert_eq!(
            chosen(attempt(false, 1e-6, C), attempt(false, 1e-3, U)),
            Some((C, false))
        );
        assert_eq!(
            chosen(attempt(false, 1e-3, C), attempt(false, 1e-6, U)),
            Some((U, false))
        );
        assert_eq!(chosen(None, attempt(false, 1e-3, U)), Some((U, false)));
        assert_eq!(chosen(attempt(false, 1e-3, C), None), Some((C, false)));
        assert_eq!(chosen(None, None), None);
    }

    #[test]
    fn non_finite_ring_fails_the_solver() {
        // A NaN on the ring around the loop makes both systems non-finite.
        let (mut positions, triangles) = parts(&tilted_hole());
        positions[3 * 10 + 2] = f32::NAN; // vertex (1, 1), diagonal to corner (2, 2)
        let mesh = mesh_of(&positions, &triangles);
        let (fans, mut patch) = refined(&mesh);
        assert!(patch.new_vertices() > 0);
        let before = patch.points.clone();
        assert_eq!(
            fair(&mesh, &fans, &mut patch).unwrap(),
            Err(FailReason::Solver)
        );
        assert_eq!(patch.points, before);
    }
}
