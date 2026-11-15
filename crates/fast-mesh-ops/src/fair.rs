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
use crate::topology::{EdgeMap, VertexFans, cross, dot, point, sub};
use crate::{Result, filled, push, reserve};

/// Geometric continuity mode used when fairing a filled hole patch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum FillContinuity {
    /// Minimal-area / minimal-dihedral DP triangulation (Liepa stage 1), with
    /// optional interior refinement and no harmonic or biharmonic fairing.
    Flat,
    /// `G1` tangent continuity via cotangent Laplace-Beltrami harmonic membrane
    /// fairing (`L x_interior = 0` with pinned boundary positions).
    TangentG1,
    /// `G2` curvature continuity via thin-plate biharmonic fairing (`L^2 x_interior = 0`)
    /// incorporating boundary position and 1-ring boundary vertex constraints.
    #[default]
    CurvatureG2,
}

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
    if patch.points.len() == patch.old.len() {
        return Ok(Ok(FairResult {
            converged: true,
            iterations: 0,
            weights: FairWeights::Cotangent,
        }));
    }
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

/// Solves the `G1` harmonic membrane problem `L x_interior = 0` with the
/// boundary vertices of `patch` pinned at their current positions.
///
/// Uses cotangent Laplace weights (clamped to `CLAMP * mean`) factored with
/// nested-dissection sparse LDLᵀ, falling back to Jacobi-preconditioned PCG
/// and then uniform weights when needed.
///
/// # Errors
/// [`OpsError::Store`](crate::OpsError::Store) when memory runs out.
pub(crate) fn fair_harmonic_g1(
    mesh: &Mesh,
    fans: &VertexFans,
    patch: &mut RefinedPatch,
) -> Result<std::result::Result<FairResult, FailReason>> {
    if patch.points.len() == patch.old.len() {
        return Ok(Ok(FairResult {
            converged: true,
            iterations: 0,
            weights: FairWeights::Cotangent,
        }));
    }
    let system = System::build(mesh, fans, patch)?;
    let cotangent = system.solve_harmonic(patch, FairWeights::Cotangent)?;
    let Some(chosen) = choose(cotangent, || {
        system.solve_harmonic(patch, FairWeights::Uniform)
    })?
    else {
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

/// Applies the requested [`FillContinuity`] fairing to `patch`, running
/// `smooth_iterations.max(1)` outer solve + interior Delaunay relaxation
/// passes when `continuity` is not [`FillContinuity::Flat`].
///
/// # Errors
/// [`OpsError::Store`](crate::OpsError::Store) when memory runs out.
pub(crate) fn fair_with_continuity(
    mesh: &Mesh,
    fans: &VertexFans,
    edges: &EdgeMap,
    patch: &mut RefinedPatch,
    continuity: FillContinuity,
    smooth_iterations: usize,
) -> Result<std::result::Result<FairResult, FailReason>> {
    if continuity == FillContinuity::Flat || patch.points.len() == patch.old.len() {
        return Ok(Ok(FairResult {
            converged: true,
            iterations: 0,
            weights: FairWeights::Cotangent,
        }));
    }
    let passes = smooth_iterations.max(1);
    let mut last = FairResult {
        converged: true,
        iterations: 0,
        weights: FairWeights::Cotangent,
    };
    for pass in 0..passes {
        let res = match continuity {
            FillContinuity::Flat => unreachable!(),
            FillContinuity::TangentG1 => fair_harmonic_g1(mesh, fans, patch)?,
            FillContinuity::CurvatureG2 => fair(mesh, fans, patch)?,
        };
        let Ok(r) = res else {
            return Ok(res);
        };
        last = r;
        if pass + 1 < passes {
            let flipped = relax_interior_delaunay(patch, edges)?;
            // G1 membrane with fixed weights on fixed triangles is idempotent unless
            // edge flips altered the interior triangulation; G2 also updates curved
            // cotangent weights and 1-ring areas on one extra pass.
            if !flipped && continuity == FillContinuity::TangentG1 {
                break;
            }
        }
    }
    Ok(Ok(last))
}

/// Flips interior patch edges that violate the intrinsic 3D Delaunay criterion
/// (`cot alpha + cot beta < 0`) when doing so preserves non-degenerate triangles
/// without folding over or duplicating an existing edge.
///
/// Boundary edges and edges belonging to the existing mesh (`edges`) are never
/// flipped. Returns `true` if at least one flip occurred.
///
/// # Errors
/// [`OpsError::Store`](crate::OpsError::Store) when memory runs out.
pub(crate) fn relax_interior_delaunay(patch: &mut RefinedPatch, edges: &EdgeMap) -> Result<bool> {
    let nt = patch.triangles.len();
    if nt < 2 {
        return Ok(false);
    }
    let old_len = patch.old.len();
    let max_sweeps = 4usize;
    let mut any_flipped = false;

    for _ in 0..max_sweeps {
        let mut records: Vec<(u32, u32, bool, u32, u32)> = Vec::new();
        reserve(&mut records, 3 * nt)?;
        for (ti, &t) in patch.triangles.iter().enumerate() {
            for e in 0..3 {
                let (u, v, opp) = (t[e], t[(e + 1) % 3], t[(e + 2) % 3]);
                let (lo, hi, fwd) = if u < v { (u, v, true) } else { (v, u, false) };
                records.push((lo, hi, fwd, ti as u32, opp));
            }
        }
        records.sort_unstable();

        let mut existing_edges: Vec<(u32, u32)> = Vec::new();
        reserve(&mut existing_edges, records.len())?;
        for r in &records {
            if existing_edges.last().copied() != Some((r.0, r.1)) {
                existing_edges.push((r.0, r.1));
            }
        }

        let mut touched = filled(nt, false)?;
        let mut added_diags: Vec<(u32, u32)> = Vec::new();
        let mut sweep_flipped = false;

        let mut idx = 0usize;
        while idx < records.len() {
            let (lo, hi) = (records[idx].0, records[idx].1);
            let mut end = idx + 1;
            while end < records.len() && records[end].0 == lo && records[end].1 == hi {
                end += 1;
            }
            if end - idx == 2 {
                let r0 = records[idx];
                let r1 = records[idx + 1];
                if r0.2 != r1.2 {
                    let (fwd_rec, rev_rec) = if r0.2 { (r0, r1) } else { (r1, r0) };
                    let (t1, a) = (fwd_rec.3 as usize, fwd_rec.4);
                    let (t2, b) = (rev_rec.3 as usize, rev_rec.4);
                    if !touched[t1] && !touched[t2] && a != b {
                        let is_mesh_edge = (lo as usize) < old_len
                            && (hi as usize) < old_len
                            && edges
                                .kind(patch.old[lo as usize], patch.old[hi as usize])
                                .is_some();
                        if !is_mesh_edge {
                            let diag = (a.min(b), a.max(b));
                            let dup_mesh = (diag.0 as usize) < old_len
                                && (diag.1 as usize) < old_len
                                && edges
                                    .kind(patch.old[diag.0 as usize], patch.old[diag.1 as usize])
                                    .is_some();
                            let dup_patch = existing_edges.binary_search(&diag).is_ok()
                                || added_diags.contains(&diag);
                            if !dup_mesh
                                && !dup_patch
                                && should_flip_quad(&patch.points, lo, hi, a, b)
                            {
                                // t1 = (lo, hi, a), t2 = (hi, lo, b) -> [b, hi, a], [a, lo, b].
                                patch.triangles[t1] = [b, hi, a];
                                patch.triangles[t2] = [a, lo, b];
                                touched[t1] = true;
                                touched[t2] = true;
                                push(&mut added_diags, diag)?;
                                sweep_flipped = true;
                                any_flipped = true;
                            }
                        }
                    }
                }
            }
            idx = end;
        }
        if !sweep_flipped {
            break;
        }
    }
    Ok(any_flipped)
}

/// Checks the intrinsic 3D Delaunay condition plus non-degenerate, non-folding
/// orientation before flipping `(lo, hi)` between `(lo, hi, a)` and `(hi, lo, b)`.
fn should_flip_quad(points: &[[f64; 3]], lo: u32, hi: u32, a: u32, b: u32) -> bool {
    let (plo, phi, pa, pb) = (
        points[lo as usize],
        points[hi as usize],
        points[a as usize],
        points[b as usize],
    );
    let Some(cot_a) = corner_cot(pa, plo, phi) else {
        return false;
    };
    let Some(cot_b) = corner_cot(pb, phi, plo) else {
        return false;
    };
    if cot_a + cot_b >= -1e-6 {
        return false;
    }
    let Some(n_old1) = unit_tri_normal(plo, phi, pa) else {
        return false;
    };
    let Some(n_old2) = unit_tri_normal(phi, plo, pb) else {
        return false;
    };
    if dot(n_old1, n_old2) < 0.0 {
        return false;
    }
    let Some(n_new1) = unit_tri_normal(pb, phi, pa) else {
        return false;
    };
    let Some(n_new2) = unit_tri_normal(pa, plo, pb) else {
        return false;
    };
    let avg = [
        n_old1[0] + n_old2[0],
        n_old1[1] + n_old2[1],
        n_old1[2] + n_old2[2],
    ];
    let avg_len = dot(avg, avg).sqrt();
    if avg_len <= 1e-12 {
        return false;
    }
    let avg_n = [avg[0] / avg_len, avg[1] / avg_len, avg[2] / avg_len];
    if dot(n_new1, avg_n) <= 0.25 || dot(n_new2, avg_n) <= 0.25 || dot(n_new1, n_new2) <= 0.25 {
        return false;
    }
    // Check the new diagonal's cotangent sum improves strictly over negative.
    let (Some(new_cot_lo), Some(new_cot_hi)) = (corner_cot(plo, pb, pa), corner_cot(phi, pa, pb))
    else {
        return false;
    };
    new_cot_lo + new_cot_hi > cot_a + cot_b + 1e-6
}

fn corner_cot(p: [f64; 3], q: [f64; 3], r: [f64; 3]) -> Option<f64> {
    let (u, w) = (sub(q, p), sub(r, p));
    let c = cross(u, w);
    let sin = dot(c, c).sqrt();
    let denom = DEGENERATE * (dot(u, u) * dot(w, w)).sqrt();
    (sin > denom && denom.is_finite()).then(|| dot(u, w) / sin)
}

fn unit_tri_normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Option<[f64; 3]> {
    let n = cross(sub(b, a), sub(c, a));
    let len = dot(n, n).sqrt();
    (len.is_finite() && len > 1e-15).then(|| [n[0] / len, n[1] / len, n[2] / len])
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

    /// Solves the `G1` harmonic system `L_II x_int = -L_IB x_bnd` with `weights`;
    /// returns `None` when weights or solved positions are non-finite.
    fn solve_harmonic(
        &self,
        patch: &RefinedPatch,
        weights: FairWeights,
    ) -> Result<Option<Attempt>> {
        let (np, first) = (self.patch_len, self.first_free);
        let free = np - first;
        if free == 0 {
            return Ok(Some(Attempt {
                x: Vec::new(),
                converged: true,
                iterations: 0,
                residual: 0.0,
                weights,
            }));
        }
        let cotangent = weights == FairWeights::Cotangent;
        let mut interior_count = 0usize;
        let mut raw_sum = 0.0f64;
        for &((a, b), w) in &self.edges {
            let (a, b) = (a as usize, b as usize);
            if a < np && b < np && (a >= first || b >= first) {
                interior_count += 1;
                raw_sum += w;
            }
        }
        if interior_count == 0 {
            return Ok(None);
        }
        let floor = if cotangent {
            let mean = raw_sum / interior_count as f64;
            if !(mean.is_finite() && mean > 0.0) {
                return Ok(None);
            }
            CLAMP * mean
        } else {
            0.0
        };

        let mut entries: Vec<(u32, u32, f64)> = Vec::new();
        reserve(&mut entries, 3 * interior_count)?;
        let mut rhs = [
            filled(free, 0.0f64)?,
            filled(free, 0.0f64)?,
            filled(free, 0.0f64)?,
        ];
        for &((a, b), raw_w) in &self.edges {
            let (a, b) = (a as usize, b as usize);
            if a >= np || b >= np || (a < first && b < first) {
                continue;
            }
            let w = if cotangent { raw_w.max(floor) } else { 1.0 };
            if !w.is_finite() {
                return Ok(None);
            }
            match (a >= first, b >= first) {
                (true, true) => {
                    let (ui, uj) = ((a - first) as u32, (b - first) as u32);
                    entries.push((ui, ui, w));
                    entries.push((uj, uj, w));
                    entries.push((ui, uj, -w));
                }
                (true, false) => {
                    let ui = (a - first) as u32;
                    entries.push((ui, ui, w));
                    let p = self.points[b];
                    for d in 0..3 {
                        rhs[d][ui as usize] += w * p[d];
                    }
                }
                (false, true) => {
                    let uj = (b - first) as u32;
                    entries.push((uj, uj, w));
                    let p = self.points[a];
                    for d in 0..3 {
                        rhs[d][uj as usize] += w * p[d];
                    }
                }
                (false, false) => {}
            }
        }
        if rhs.iter().flatten().any(|v| !v.is_finite()) {
            return Ok(None);
        }

        let (row_starts, cols, vals) = build_symmetric_csr(free, &entries)?;
        let mut xs = [
            filled(free, 0.0f64)?,
            filled(free, 0.0f64)?,
            filled(free, 0.0f64)?,
        ];
        let mut ax = filled(free, 0.0f64)?;
        let mut x0 = filled(free, 0.0f64)?;

        let mut converged = false;
        let mut residual = f64::INFINITY;
        let mut iterations = 0usize;

        if let Some(factor) = sparse::factor_spd_ordered(free, &entries, &self.order)? {
            let mut ok = true;
            let mut rel_max = 0.0f64;
            for d in 0..3 {
                factor.solve(&rhs[d], &mut xs[d])?;
                let (c, rel, finite) = eval_csr_residual(
                    &row_starts,
                    &cols,
                    &vals,
                    &xs[d],
                    &rhs[d],
                    &patch.points[first..np],
                    d,
                    &mut x0,
                    &mut ax,
                );
                if !finite {
                    ok = false;
                    break;
                }
                ok &= c;
                rel_max = rel_max.max(rel);
            }
            if ok {
                converged = true;
                residual = rel_max;
            }
        }

        if !converged {
            let max_iters = (free * 4).max(128);
            let mut pcg_conv = true;
            let mut pcg_res = 0.0f64;
            let csr = sparse::SymCsr {
                n: free,
                row_starts: &row_starts,
                cols: &cols,
                vals: &vals,
            };
            for d in 0..3 {
                for (dst, pt) in xs[d][..free].iter_mut().zip(&patch.points[first..np]) {
                    *dst = pt[d];
                }
                let r = sparse::solve_pcg(&csr, &rhs[d], &mut xs[d], TOLERANCE, max_iters)?;
                iterations = iterations.max(r.iterations);
                let (c, rel, finite) = eval_csr_residual(
                    &row_starts,
                    &cols,
                    &vals,
                    &xs[d],
                    &rhs[d],
                    &patch.points[first..np],
                    d,
                    &mut x0,
                    &mut ax,
                );
                if !finite {
                    return Ok(None);
                }
                pcg_conv &= c && r.converged;
                pcg_res = pcg_res.max(rel);
            }
            converged = pcg_conv;
            residual = pcg_res;
        }

        let mut x = filled(free, [0.0f64; 3])?;
        for (xi, ((&vx, &vy), &vz)) in x
            .iter_mut()
            .zip(xs[0].iter().zip(xs[1].iter()).zip(xs[2].iter()))
        {
            *xi = [vx, vy, vz];
        }
        Ok(Some(Attempt {
            x,
            converged,
            iterations,
            residual,
            weights,
        }))
    }
}

/// Evaluates `|rhs - A x|` for CSR `A` against the tolerance reference norm.
#[allow(clippy::too_many_arguments)]
fn eval_csr_residual(
    row_starts: &[u32],
    cols: &[u32],
    vals: &[f64],
    x: &[f64],
    rhs: &[f64],
    init_points: &[[f64; 3]],
    axis: usize,
    x0: &mut [f64],
    ax: &mut [f64],
) -> (bool, f64, bool) {
    if x.iter().any(|v| !v.is_finite()) {
        return (false, f64::INFINITY, false);
    }
    let res = csr_residual(row_starts, cols, vals, x, rhs, ax);
    if !res.is_finite() {
        return (false, f64::INFINITY, false);
    }
    let rhs_norm = norm(rhs);
    let reference = if rhs_norm > 0.0 {
        rhs_norm
    } else {
        for (i, xi) in x0.iter_mut().enumerate() {
            *xi = init_points[i][axis];
        }
        csr_residual(row_starts, cols, vals, x0, rhs, ax)
    };
    let conv = res <= TOLERANCE * reference;
    let rel = if reference > 0.0 {
        res / reference
    } else {
        0.0
    };
    (conv, rel, true)
}

/// Converts symmetric undirected/diagonal triplets into sorted merged CSR.
fn build_symmetric_csr(
    n: usize,
    entries: &[(u32, u32, f64)],
) -> Result<(Vec<u32>, Vec<u32>, Vec<f64>)> {
    let mut triplets: Vec<(u32, u32, f64)> = Vec::new();
    reserve(&mut triplets, entries.len() * 2)?;
    for &(u, v, w) in entries {
        triplets.push((u, v, w));
        if u != v {
            triplets.push((v, u, w));
        }
    }
    triplets.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.total_cmp(&b.2)));

    let mut row_starts = filled(n + 1, 0u32)?;
    let mut cols: Vec<u32> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    reserve(&mut cols, triplets.len())?;
    reserve(&mut vals, triplets.len())?;

    let mut i = 0usize;
    while i < triplets.len() {
        let r = triplets[i].0;
        let c = triplets[i].1;
        let mut sum = triplets[i].2;
        i += 1;
        while i < triplets.len() && triplets[i].0 == r && triplets[i].1 == c {
            sum += triplets[i].2;
            i += 1;
        }
        cols.push(c);
        vals.push(sum);
        row_starts[r as usize + 1] += 1;
    }
    for r in 0..n {
        row_starts[r + 1] += row_starts[r];
    }
    Ok((row_starts, cols, vals))
}

fn csr_residual(
    row_starts: &[u32],
    cols: &[u32],
    vals: &[f64],
    x: &[f64],
    rhs: &[f64],
    ax: &mut [f64],
) -> f64 {
    let mut s = 0.0;
    for i in 0..rhs.len() {
        let mut sum = 0.0;
        for t in row_starts[i] as usize..row_starts[i + 1] as usize {
            sum += vals[t] * x[cols[t] as usize];
        }
        ax[i] = sum;
        let diff = rhs[i] - sum;
        s += diff * diff;
    }
    s.sqrt()
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

    #[test]
    fn harmonic_g1_reproduces_tilted_plane_and_saddle() {
        let mesh = tilted_hole();
        let (fans, mut patch) = refined(&mesh);
        assert!(patch.new_vertices() > 0);
        let r = fair_harmonic_g1(&mesh, &fans, &mut patch).unwrap().unwrap();
        assert!(r.converged && r.iterations == 0);
        let first = patch.old.len();
        for p in &patch.points[first..] {
            assert!((p[2] - 0.3 * p[0] - 0.2 * p[1]).abs() < 1e-6, "{p:?}");
            assert!((2.0..=6.0).contains(&p[0]) && (2.0..=6.0).contains(&p[1]));
        }

        // Saddle hole z = 0.08 * ((x - 4)^2 - (y - 4)^2) obeys the maximum principle.
        let (mut pos, tris) = parts(&tilted_hole());
        for p in pos.chunks_mut(3) {
            let (dx, dy) = (p[0] - 4.0, p[1] - 4.0);
            p[2] = 0.08 * (dx * dx - dy * dy);
        }
        let saddle = mesh_of(&pos, &tris);
        let (sfans, mut spatch) = refined(&saddle);
        let sr = fair_harmonic_g1(&saddle, &sfans, &mut spatch)
            .unwrap()
            .unwrap();
        assert!(sr.converged);
        let old_len = spatch.old.len();
        let min_z = spatch.points[..old_len]
            .iter()
            .map(|p| p[2])
            .fold(f64::INFINITY, f64::min);
        let max_z = spatch.points[..old_len]
            .iter()
            .map(|p| p[2])
            .fold(f64::NEG_INFINITY, f64::max);
        for p in &spatch.points[old_len..] {
            assert!(
                p[2] >= min_z - 1e-6 && p[2] <= max_z + 1e-6,
                "z={} outside [{min_z}, {max_z}]",
                p[2]
            );
        }
    }

    #[test]
    fn continuity_modes_fit_sphere_and_cylinder() {
        let cells: Vec<(u32, u32)> = (2..6).flat_map(|y| (2..6).map(move |x| (x, y))).collect();
        let (mut spos, stris) = parts(&grid_without(8, &cells));
        let radius = 10.0f32;
        for p in spos.chunks_mut(3) {
            let (x, y) = (p[0] - 4.0, p[1] - 4.0);
            p[2] = (radius * radius - x * x - y * y).sqrt();
        }
        let sphere_cap = mesh_of(&spos, &stris);
        let loops = boundary_loops(&sphere_cap).unwrap();
        let hole = loops.iter().min_by_key(|l| l.vertices.len()).unwrap();
        let edges = EdgeMap::build(&sphere_cap).unwrap();
        let fans = VertexFans::build(&sphere_cap).unwrap();
        let base_tris = triangulate_loop(&sphere_cap, &hole.vertices, &edges, 200, None)
            .unwrap()
            .unwrap();

        let radial_err = |patch: &RefinedPatch| -> f64 {
            patch.points[patch.old.len()..]
                .iter()
                .map(|p| {
                    let r = ((p[0] - 4.0).powi(2) + (p[1] - 4.0).powi(2) + p[2] * p[2]).sqrt();
                    (r - radius as f64).abs()
                })
                .fold(0.0, f64::max)
        };

        let mut p_flat =
            refine(&sphere_cap, &fans, &edges, &hole.vertices, &base_tris, None).unwrap();
        let before = p_flat.points.clone();
        fair_with_continuity(
            &sphere_cap,
            &fans,
            &edges,
            &mut p_flat,
            FillContinuity::Flat,
            2,
        )
        .unwrap()
        .unwrap();
        assert_eq!(p_flat.points, before);

        let mut p_g1 =
            refine(&sphere_cap, &fans, &edges, &hole.vertices, &base_tris, None).unwrap();
        fair_with_continuity(
            &sphere_cap,
            &fans,
            &edges,
            &mut p_g1,
            FillContinuity::TangentG1,
            1,
        )
        .unwrap()
        .unwrap();

        let mut p_g2 =
            refine(&sphere_cap, &fans, &edges, &hole.vertices, &base_tris, None).unwrap();
        fair_with_continuity(
            &sphere_cap,
            &fans,
            &edges,
            &mut p_g2,
            FillContinuity::CurvatureG2,
            2,
        )
        .unwrap()
        .unwrap();

        assert!(
            radial_err(&p_g2) < radial_err(&p_g1) * 0.25 && radial_err(&p_g2) < 0.05,
            "sphere G2 err={} vs G1 err={}",
            radial_err(&p_g2),
            radial_err(&p_g1)
        );

        // Cylinder patch: y unchanged, x^2 + z^2 = R^2 centered at x = 4.
        let (mut cpos, ctris) = parts(&grid_without(8, &cells));
        for p in cpos.chunks_mut(3) {
            let x = p[0] - 4.0;
            p[2] = (radius * radius - x * x).sqrt();
        }
        let cyl = mesh_of(&cpos, &ctris);
        let cloops = boundary_loops(&cyl).unwrap();
        let chole = cloops.iter().min_by_key(|l| l.vertices.len()).unwrap();
        let cedges = EdgeMap::build(&cyl).unwrap();
        let cfans = VertexFans::build(&cyl).unwrap();
        let ctris_base = triangulate_loop(&cyl, &chole.vertices, &cedges, 200, None)
            .unwrap()
            .unwrap();
        let mut cyl_g2 = refine(&cyl, &cfans, &cedges, &chole.vertices, &ctris_base, None).unwrap();
        fair_with_continuity(
            &cyl,
            &cfans,
            &cedges,
            &mut cyl_g2,
            FillContinuity::CurvatureG2,
            2,
        )
        .unwrap()
        .unwrap();
        let cyl_err = cyl_g2.points[cyl_g2.old.len()..]
            .iter()
            .map(|p| (((p[0] - 4.0).powi(2) + p[2] * p[2]).sqrt() - radius as f64).abs())
            .fold(0.0, f64::max);
        assert!(
            cyl_err < 0.05,
            "cylinder G2 radial error too high: {cyl_err}"
        );
    }

    #[test]
    fn relax_interior_delaunay_flips_bad_diagonal() {
        let mesh = tilted_hole();
        let edges = EdgeMap::build(&mesh).unwrap();
        // Flat elongated diamond: diagonal (0, 2) is much longer than (1, 3).
        let mut patch = RefinedPatch {
            old: vec![0, 10, 20],
            points: vec![
                [0.0, 0.0, 0.0],
                [5.0, -0.5, 0.0],
                [10.0, 0.0, 0.0],
                [5.0, 0.5, 0.0],
            ],
            parents: vec![[(0, 0.5), (1, 0.5), (2, 0.0)]],
            triangles: vec![[0, 2, 3], [2, 0, 1]],
            capped: false,
        };
        assert!(relax_interior_delaunay(&mut patch, &edges).unwrap());
        let has_short_diag = patch
            .triangles
            .iter()
            .any(|t| t.contains(&1) && t.contains(&3));
        assert!(
            has_short_diag,
            "expected flip to (1, 3): {:?}",
            patch.triangles
        );
    }
}
