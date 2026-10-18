//! Fairing a refined hole patch.
//!
//! [`fair`] moves the new vertices of a [`RefinedPatch`] so that the patch
//! minimises bending and continues the curvature around it: it solves the
//! discrete bi-Laplacian `L(L x) = 0` for the new vertices, with the loop
//! vertices and the ring of mesh vertices just outside the loop fixed, by
//! conjugate gradient on each coordinate.

use fast_mesh_store::{Mesh, VertexId};

use crate::fill::{FailReason, FairWeights};
use crate::refine::RefinedPatch;
use crate::topology::{VertexFans, cross, dot, point, sub};
use crate::{Result, filled, push, reserve};

/// Conjugate gradient stops once the residual is this small relative to the
/// right-hand side.
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
    /// Conjugate gradient iterations, the most of the three coordinates.
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
/// - Each coordinate is solved by conjugate gradient with a Jacobi
///   preconditioner from the refined positions, to `1e-10` relative to the
///   right-hand side, at most `20 n + 100` iterations for `n` new vertices.
///   A solve that stops short keeps its best iterate.
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
        Ok(System {
            patch_len,
            first_free: old.len(),
            points,
            edges,
            mass,
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
        let b = Sparse {
            starts: &count,
            cols: &cols,
            vals: &vals,
        };
        let mut jacobi = filled(free, 0.0f64)?;
        for (c, v) in cols.iter().zip(&vals) {
            jacobi[*c as usize] += v * v;
        }
        if jacobi.iter().any(|d| !(d.is_finite() && *d > 0.0)) {
            return Ok(None);
        }

        let mut cg = Cg::new(free, np)?;
        let mut x = filled(free, [0.0f64; 3])?;
        let mut rhs = filled(free, 0.0f64)?;
        let mut tmp = filled(np, 0.0f64)?;
        let cap = 20 * free + 100;
        let (mut converged, mut iterations, mut residual) = (true, 0, 0.0f64);
        for d in 0..3 {
            for (t, f) in tmp.iter_mut().zip(&fixed) {
                *t = -f[d];
            }
            b.transpose_times(&tmp, &mut rhs);
            for (i, xi) in cg.x.iter_mut().enumerate() {
                *xi = patch.points[first + i][d];
            }
            let r = cg.solve(&b, &jacobi, &rhs, cap);
            if !r.finite {
                return Ok(None);
            }
            converged &= r.converged;
            iterations = iterations.max(r.iterations);
            residual = residual.max(r.residual);
            for (xi, v) in x.iter_mut().zip(&cg.x) {
                xi[d] = *v;
            }
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

/// A sparse matrix in compressed rows.
struct Sparse<'a> {
    /// Row `k` is `cols[starts[k]..starts[k + 1]]` with `vals` alongside.
    starts: &'a [u32],
    cols: &'a [u32],
    vals: &'a [f64],
}

impl Sparse<'_> {
    fn rows(&self) -> usize {
        self.starts.len() - 1
    }

    fn row(&self, k: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let range = self.starts[k] as usize..self.starts[k + 1] as usize;
        self.cols[range.clone()]
            .iter()
            .zip(&self.vals[range])
            .map(|(&c, &v)| (c as usize, v))
    }

    /// `out = B v`.
    fn times(&self, v: &[f64], out: &mut [f64]) {
        for (k, o) in out.iter_mut().enumerate().take(self.rows()) {
            *o = self.row(k).map(|(c, x)| x * v[c]).sum();
        }
    }

    /// `out = Bᵀ v`.
    fn transpose_times(&self, v: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for (k, &vk) in v.iter().enumerate().take(self.rows()) {
            for (c, x) in self.row(k) {
                out[c] += x * vk;
            }
        }
    }
}

/// The outcome of one conjugate gradient solve.
struct CgResult {
    converged: bool,
    iterations: usize,
    /// The best residual relative to the right-hand side.
    residual: f64,
    /// Whether the best iterate is finite.
    finite: bool,
}

/// Conjugate gradient on `Bᵀ B x = rhs`, with its vectors.
struct Cg {
    x: Vec<f64>,
    best: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
    /// `B p`, one entry per row of `B`.
    bp: Vec<f64>,
}

fn dot_n(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Cg {
    fn new(n: usize, rows: usize) -> Result<Cg> {
        Ok(Cg {
            x: filled(n, 0.0)?,
            best: filled(n, 0.0)?,
            r: filled(n, 0.0)?,
            z: filled(n, 0.0)?,
            p: filled(n, 0.0)?,
            ap: filled(n, 0.0)?,
            bp: filled(rows, 0.0)?,
        })
    }

    /// `ap = Bᵀ B p`.
    fn apply(&mut self, b: &Sparse) {
        b.times(&self.p, &mut self.bp);
        b.transpose_times(&self.bp, &mut self.ap);
    }

    /// `z = r / diag`.
    fn precondition(&mut self, diag: &[f64]) {
        for ((z, r), d) in self.z.iter_mut().zip(&self.r).zip(diag) {
            *z = r / d;
        }
    }

    /// Solves from the current `x` with the Jacobi preconditioner `diag`
    /// (the diagonal of `Bᵀ B`), at most `cap` iterations; leaves the best
    /// iterate in `x`.
    fn solve(&mut self, b: &Sparse, diag: &[f64], rhs: &[f64], cap: usize) -> CgResult {
        self.p.copy_from_slice(&self.x);
        self.apply(b);
        for ((r, b), ap) in self.r.iter_mut().zip(rhs).zip(&self.ap) {
            *r = b - ap;
        }
        let start = dot_n(&self.r, &self.r).sqrt();
        let norm = dot_n(rhs, rhs).sqrt();
        // A zero right-hand side has the solution 0: measure from the start.
        let reference = if norm > 0.0 { norm } else { start };
        let mut result = CgResult {
            converged: start <= TOLERANCE * reference,
            iterations: 0,
            residual: if reference > 0.0 {
                start / reference
            } else {
                0.0
            },
            finite: self.x.iter().all(|x| x.is_finite()) && start.is_finite(),
        };
        if result.converged || !result.finite {
            return result;
        }
        self.best.copy_from_slice(&self.x);
        let mut best = start;
        self.precondition(diag);
        self.p.copy_from_slice(&self.z);
        let mut rz = dot_n(&self.r, &self.z);
        for iteration in 1..=cap {
            self.apply(b);
            let pap = dot_n(&self.p, &self.ap);
            if !(pap.is_finite() && pap > 0.0) {
                break;
            }
            let alpha = rz / pap;
            for i in 0..self.x.len() {
                self.x[i] += alpha * self.p[i];
                self.r[i] -= alpha * self.ap[i];
            }
            let res = dot_n(&self.r, &self.r).sqrt();
            result.iterations = iteration;
            if !res.is_finite() {
                break;
            }
            if res < best {
                best = res;
                self.best.copy_from_slice(&self.x);
            }
            if res <= TOLERANCE * reference {
                result.converged = true;
                break;
            }
            self.precondition(diag);
            let next = dot_n(&self.r, &self.z);
            let beta = next / rz;
            rz = next;
            for i in 0..self.p.len() {
                self.p[i] = self.z[i] + beta * self.p[i];
            }
        }
        self.x.copy_from_slice(&self.best);
        // The recurrence drifts from the true residual: judge the iterate
        // kept by `rhs - Bᵀ B x` itself.
        self.p.copy_from_slice(&self.x);
        self.apply(b);
        for ((r, b), ap) in self.r.iter_mut().zip(rhs).zip(&self.ap) {
            *r = b - ap;
        }
        let res = dot_n(&self.r, &self.r).sqrt();
        result.converged = res <= TOLERANCE * reference;
        result.residual = res / reference;
        result.finite = self.x.iter().all(|x| x.is_finite()) && res.is_finite();
        result
    }
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
        let tris = triangulate_loop(mesh, &l.vertices, &edges, 200)
            .unwrap()
            .unwrap();
        let patch = refine(mesh, &fans, &edges, &l.vertices, &tris).unwrap();
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
        assert!(r.iterations > 0 && r.iterations <= 20 * patch.new_vertices() + 100);
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
