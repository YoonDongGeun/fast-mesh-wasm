//! Rigid registration: point-to-plane ICP against a mesh and its BVH.
//!
//! [`icp()`] moves a cloud of points onto a fixed mesh. Each iteration finds
//! every point's closest point on the mesh with the BVH, keeps the closest
//! `overlap` fraction of them (trimmed ICP, so a partial overlap and outliers
//! do not pull the result), and solves one linearised least-squares step for
//! the rotation and translation that reduce the distances to those
//! triangles' planes.
//!
//! The iteration is local: it starts at the `initial` transform and finds the
//! nearest minimum, so the start has to be roughly right. "Converged" means
//! the iteration stopped moving with every degree of freedom constrained, not
//! that the alignment is the right one: read [`IcpReport::rmse`] for that.
//!
//! Directions the surface does not constrain (a plane sliding in its own
//! plane, a cylinder along its axis) are left out of each step instead of
//! being solved from rounding noise, and are reported through
//! [`IcpStatus::Underconstrained`].

use fast_mesh_bvh::Bvh;
use fast_mesh_query::{ClosestHit, ClosestOptions, QueryError, Rigid, closest_point_many};
use fast_mesh_store::{Mesh, VertexId};

use crate::{OpsError, Result, filled, reserve};

/// Fewest correspondences that can determine a rigid transform.
const MIN_CORRESPONDENCES: usize = 6;

/// A degree of freedom counts as constrained when its eigenvalue of the
/// normalised normal equations is at least this fraction of the largest. The
/// eigenvalue is a squared singular value, so this is a ratio of `1e-3` in
/// how strongly the distances respond to that motion. It sits above the noise
/// of `f32` positions, whose rounding jitters the face normals of a finely
/// tessellated, tilted plane.
pub(crate) const MIN_CONDITIONING: f64 = 1e-6;

/// The largest rotation of one step, in radians. Larger steps are shortened:
/// the linearisation means nothing that far out.
const MAX_STEP_ANGLE: f64 = 0.5;

/// Options for [`icp()`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IcpOptions {
    /// The most steps to take. Zero evaluates the initial transform only.
    /// Default 50.
    pub max_iterations: usize,
    /// Stops when one step moves every point by at most this fraction of the
    /// points' extent (their largest distance from their centroid). Zero or
    /// greater. Default `1e-6`.
    pub tolerance: f64,
    /// The fraction of the points expected to lie over the fixed mesh, in
    /// `(0, 1]`: each iteration uses that many of them, closest first. Set it
    /// a little below the true overlap for partial scans. Default 1 (every
    /// point counts, so nothing is rejected).
    pub overlap: f64,
    /// Points farther than this from the fixed mesh are not used. Positive.
    /// Default `f32::INFINITY`.
    pub max_distance: f32,
}

impl Default for IcpOptions {
    fn default() -> Self {
        IcpOptions {
            max_iterations: 50,
            tolerance: 1e-6,
            overlap: 1.0,
            max_distance: f32::INFINITY,
        }
    }
}

/// Why [`icp()`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcpStatus {
    /// A step fell below the tolerance with all six degrees of freedom
    /// constrained.
    Converged,
    /// `max_iterations` steps ran out first.
    MaxIterations,
    /// A step fell below the tolerance, but the used points leave a degree of
    /// freedom unconstrained, so the transform along it is arbitrary: it
    /// stayed where the initial transform put it. Distances are still
    /// minimised along the other directions.
    Underconstrained,
    /// Fewer than six usable correspondences (see [`IcpOptions::max_distance`]).
    TooFewCorrespondences,
    /// A step took a point out of the range the query can represent.
    Diverged,
}

/// What [`icp()`] found. Every number is finite, except that `rmse` is
/// infinite when no point has a correspondence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IcpReport {
    /// The transform from the points to the fixed mesh's local coordinates.
    /// When the status is [`IcpStatus::Converged`] or
    /// [`IcpStatus::Underconstrained`], the pose iteration stopped at;
    /// otherwise the best pose seen (most points used, then lowest `rmse`).
    pub transform: Rigid,
    /// The root mean square distance from the used points to the fixed mesh
    /// at `transform`.
    pub rmse: f64,
    /// Steps taken.
    pub iterations: usize,
    /// Why iteration stopped.
    pub status: IcpStatus,
    /// Points given.
    pub samples: usize,
    /// Points used at `transform`.
    pub inliers: usize,
    /// The smallest eigenvalue over the largest of the normalised normal
    /// equations at the last step, in `[0, 1]`: how well the worst-constrained
    /// direction is determined. Below `1e-6` it counts as unconstrained. Zero
    /// when no step ran.
    pub conditioning: f64,
}

impl IcpReport {
    /// Whether the status is [`IcpStatus::Converged`].
    pub fn converged(&self) -> bool {
        self.status == IcpStatus::Converged
    }
}

/// Up to `max_samples` evenly spaced vertices of `mesh` that its alive
/// triangles use, as their stored local positions, in vertex order. Vertices
/// no triangle uses are not on the surface, so they are left out.
///
/// # Errors
/// [`OpsError::Store`] when out of memory.
pub fn sample_vertices(mesh: &Mesh, max_samples: usize) -> Result<Vec<[f32; 3]>> {
    let mut used = filled(mesh.vertex_slots(), false)?;
    for t in 0..mesh.triangle_slots() {
        if let Some(c) = mesh.triangle(t as u32) {
            for v in c {
                used[v as usize] = true;
            }
        }
    }
    let count = used.iter().filter(|&&u| u).count();
    let take = count.min(max_samples);
    let mut out = Vec::new();
    reserve(&mut out, take)?;
    if take == 0 {
        return Ok(out);
    }
    let used = used.iter().enumerate().filter(|(_, u)| **u);
    for (seen, (v, _)) in used.enumerate() {
        // Taken when it starts a new stride of `count / take`. `u128`: the
        // product of two slot counts.
        let next = (out.len() as u128 * count as u128).div_ceil(take as u128);
        if out.len() < take && seen as u128 == next {
            if let Some(p) = mesh.position(v as VertexId) {
                out.push(p);
            }
        }
    }
    Ok(out)
}

fn validate(points: &[[f64; 3]], fixed: &Mesh, options: &IcpOptions) -> Result<()> {
    let invalid = |m| Err(OpsError::InvalidInput(m));
    if points.is_empty() {
        return invalid("icp needs at least one point");
    }
    if u32::try_from(points.len()).is_err() {
        return invalid("icp takes fewer than 2^32 points");
    }
    if fixed.triangle_count() == 0 {
        return invalid("icp needs a fixed mesh with triangles");
    }
    if !points.iter().flatten().all(|c| c.is_finite()) {
        return invalid("icp points must be finite");
    }
    if !(options.tolerance >= 0.0 && options.tolerance.is_finite()) {
        return invalid("tolerance must be finite and zero or greater");
    }
    if !(options.overlap > 0.0 && options.overlap <= 1.0) {
        return invalid("overlap must be in (0, 1]");
    }
    if options.max_distance.is_nan() || options.max_distance <= 0.0 {
        return invalid("max_distance must be positive");
    }
    Ok(())
}

/// A correspondence kept this iteration.
#[derive(Clone, Copy)]
struct Kept {
    /// The point's index.
    point: u32,
    /// The closest distance.
    distance: f32,
}

/// The inputs and working buffers of one call.
struct Evaluator<'a> {
    points: &'a [[f64; 3]],
    fixed: &'a Mesh,
    bvh: &'a Bvh,
    query: ClosestOptions,
    keep: usize,
    moved: Vec<[f64; 3]>,
    rounded: Vec<[f32; 3]>,
    hits: Vec<Option<ClosestHit>>,
    kept: Vec<Kept>,
}

impl Evaluator<'_> {
    /// Finds the correspondences at `transform` and trims them. Returns the
    /// rmse over the kept ones (infinite with none), or `None` when a moved
    /// point leaves `f32` range.
    fn evaluate(&mut self, transform: &Rigid) -> Result<Option<f64>> {
        for ((p, m), r) in self
            .points
            .iter()
            .zip(&mut self.moved)
            .zip(&mut self.rounded)
        {
            *m = apply(transform, *p);
            *r = m.map(|c| c as f32);
            if !r.iter().all(|c| c.is_finite()) {
                return Ok(None);
            }
        }
        closest_point_many(
            self.fixed,
            self.bvh,
            &self.rounded,
            &self.query,
            &mut self.hits,
        )?;
        self.kept.clear();
        for (i, hit) in self.hits.iter().enumerate() {
            if let Some(h) = hit {
                self.kept.push(Kept {
                    point: i as u32,
                    distance: h.distance,
                });
            }
        }
        if self.kept.len() > self.keep {
            // A total order, so the kept set is unique: it does not depend on
            // the selection's internals.
            let order = |a: &Kept, b: &Kept| {
                a.distance
                    .total_cmp(&b.distance)
                    .then(a.point.cmp(&b.point))
            };
            self.kept.select_nth_unstable_by(self.keep - 1, order);
            self.kept.truncate(self.keep);
        }
        if self.kept.is_empty() {
            return Ok(Some(f64::INFINITY));
        }
        let sum: f64 = self
            .kept
            .iter()
            .map(|k| f64::from(k.distance) * f64::from(k.distance))
            .sum();
        Ok(Some((sum / self.kept.len() as f64).sqrt()))
    }

    /// The normal equations of the kept correspondences, about `center` with
    /// rotations scaled by `extent`: `(hessian, gradient, rows)`, in the
    /// unknowns `(translation, rotation * extent)`. Corners are read in `f64`,
    /// so the residuals carry no `f32` query rounding. A correspondence on a
    /// zero-area triangle has no plane and is skipped.
    fn assemble(&self, center: [f64; 3], extent: f64) -> ([[f64; 6]; 6], [f64; 6], usize) {
        let mut h = [[0.0; 6]; 6];
        let mut g = [0.0; 6];
        let mut rows = 0;
        for k in &self.kept {
            let Some((a, n)) = self.hits[k.point as usize].and_then(|hit| self.plane(hit)) else {
                continue;
            };
            let p = self.moved[k.point as usize];
            let residual = dot(n, sub(p, a));
            let lever = cross(sub(p, center), n);
            let row = [
                n[0],
                n[1],
                n[2],
                lever[0] / extent,
                lever[1] / extent,
                lever[2] / extent,
            ];
            for (i, ri) in row.iter().enumerate() {
                g[i] -= ri * residual;
                for (j, rj) in row.iter().enumerate().skip(i) {
                    h[i][j] += ri * rj;
                }
            }
            rows += 1;
        }
        for i in 0..6 {
            for j in 0..i {
                h[i][j] = h[j][i];
            }
        }
        (h, g, rows)
    }

    /// A corner of the hit's triangle and its unit normal, or `None` for a
    /// triangle with no area.
    fn plane(&self, hit: ClosestHit) -> Option<([f64; 3], [f64; 3])> {
        let [a, b, c] = self.fixed.triangle(hit.triangle)?;
        let a = self.fixed.position(a)?.map(f64::from);
        let b = self.fixed.position(b)?.map(f64::from);
        let c = self.fixed.position(c)?.map(f64::from);
        Some((a, unit(cross(sub(b, a), sub(c, a)))?))
    }
}

/// Aligns `points` to the mesh `fixed` by rigid, point-to-plane ICP, starting
/// from `initial`. See the [module docs](self).
///
/// The report's transform maps the points to `fixed`'s local coordinates (the
/// stored `f32` positions, without the origin). The correspondence search
/// rounds moved points to `f32`; the plane distances the solver minimises are
/// computed in `f64`. Results do not depend on the `parallel` feature or the
/// thread count.
///
/// # Errors
/// [`OpsError::InvalidInput`] for no points, non-finite points, a fixed mesh
/// without triangles, or an invalid option; [`OpsError::Query`] for an invalid
/// `initial`, one that takes a point out of `f32` range, or a `bvh` that does
/// not match `fixed`; [`OpsError::Store`] when out of memory.
pub fn icp(
    points: &[[f64; 3]],
    fixed: &Mesh,
    bvh: &Bvh,
    initial: &Rigid,
    options: &IcpOptions,
) -> Result<IcpReport> {
    validate(points, fixed, options)?;
    initial.validate()?;
    let n = points.len();
    let (center0, extent) = spread(points);
    let keep = (options.overlap * n as f64).ceil().clamp(1.0, n as f64) as usize;
    let mut eval = Evaluator {
        points,
        fixed,
        bvh,
        query: ClosestOptions {
            max_distance: options.max_distance,
        },
        keep,
        moved: filled(n, [0.0; 3])?,
        rounded: filled(n, [0.0; 3])?,
        hits: filled(n, None)?,
        kept: Vec::new(),
    };
    reserve(&mut eval.kept, n)?;

    let mut transform = *initial;
    // The first evaluation always sets it, or returns.
    let mut best = (*initial, f64::INFINITY, 0);
    let mut last = best;
    let mut evaluated = false;
    let mut iterations = 0;
    let mut conditioning = 0.0;
    // Set once a step fell below the tolerance: the next pass only evaluates.
    let mut settled = None;
    let status = loop {
        let Some(rmse) = eval.evaluate(&transform)? else {
            if !evaluated {
                return Err(QueryError::InvalidInput("query point must be finite").into());
            }
            break IcpStatus::Diverged;
        };
        let inliers = eval.kept.len();
        last = (transform, rmse, inliers);
        if !evaluated || inliers > best.2 || (inliers == best.2 && rmse < best.1) {
            best = last;
        }
        evaluated = true;
        if let Some(status) = settled {
            break status;
        }
        if inliers < MIN_CORRESPONDENCES {
            break IcpStatus::TooFewCorrespondences;
        }
        if iterations == options.max_iterations {
            break IcpStatus::MaxIterations;
        }
        let center = apply(&transform, center0);
        let (h, g, rows) = eval.assemble(center, extent);
        if rows < MIN_CORRESPONDENCES {
            break IcpStatus::TooFewCorrespondences;
        }
        let (step, c) = solve(&h, &g);
        conditioning = c;
        iterations += 1;
        let rotation = [step[3] / extent, step[4] / extent, step[5] / extent];
        let angle = norm(rotation);
        let shrink = if angle > MAX_STEP_ANGLE {
            MAX_STEP_ANGLE / angle
        } else {
            1.0
        };
        let translation = [step[0], step[1], step[2]].map(|v| v * shrink);
        let rotation = rotation.map(|v| v * shrink);
        transform = compose_step(&transform, center, translation, rotation);
        // An upper bound on how far any point moved, over the extent.
        if (norm(translation) + norm(rotation) * extent) / extent <= options.tolerance {
            settled = Some(if conditioning >= MIN_CONDITIONING {
                IcpStatus::Converged
            } else {
                IcpStatus::Underconstrained
            });
        }
    };
    let (transform, rmse, inliers) = if settled == Some(status) {
        last
    } else {
        best
    };
    Ok(IcpReport {
        transform,
        rmse,
        iterations,
        status,
        samples: n,
        inliers,
        conditioning,
    })
}

/// The centroid of `points` and their largest distance from it (1 when they
/// all coincide, so that dividing by it is safe).
fn spread(points: &[[f64; 3]]) -> ([f64; 3], f64) {
    let mut sum = [0.0; 3];
    for p in points {
        sum = add(sum, *p);
    }
    let center = sum.map(|c| c / points.len() as f64);
    let far = points
        .iter()
        .map(|p| dot(sub(*p, center), sub(*p, center)))
        .fold(0.0, f64::max)
        .sqrt();
    (center, if far > 0.0 { far } else { 1.0 })
}

/// Solves `h x = g` over the constrained eigendirections only, and returns
/// `x` with the conditioning (smallest eigenvalue over the largest).
fn solve(h: &[[f64; 6]; 6], g: &[f64; 6]) -> ([f64; 6], f64) {
    let (values, vectors) = jacobi_eigen(*h);
    let largest = values.iter().copied().fold(0.0, f64::max);
    let smallest = values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .max(0.0);
    let mut x = [0.0; 6];
    if largest <= 0.0 {
        return (x, 0.0);
    }
    for (k, &value) in values.iter().enumerate() {
        if value < MIN_CONDITIONING * largest {
            continue;
        }
        let along = (0..6).map(|i| vectors[i][k] * g[i]).sum::<f64>() / value;
        for (xi, row) in x.iter_mut().zip(&vectors) {
            *xi += row[k] * along;
        }
    }
    (x, smallest / largest)
}

/// The eigenvalues of a symmetric matrix and its eigenvectors (columns), by
/// cyclic Jacobi rotations.
fn jacobi_eigen(mut a: [[f64; 6]; 6]) -> ([f64; 6], [[f64; 6]; 6]) {
    let mut v = [[0.0; 6]; 6];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    // Jacobi converges quadratically: 6 x 6 settles within a dozen sweeps.
    for _ in 0..32 {
        let off: f64 = (0..6)
            .flat_map(|i| (i + 1..6).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        let diag: f64 = (0..6).map(|i| a[i][i] * a[i][i]).sum();
        if off <= 1e-36 * diag {
            break;
        }
        for p in 0..6 {
            for q in p + 1..6 {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in &mut a {
                    let (ap, aq) = (row[p], row[q]);
                    row[p] = c * ap - s * aq;
                    row[q] = s * ap + c * aq;
                }
                for k in 0..6 {
                    let (ap, aq) = (a[p][k], a[q][k]);
                    a[p][k] = c * ap - s * aq;
                    a[q][k] = s * ap + c * aq;
                }
                for row in &mut v {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    (std::array::from_fn(|i| a[i][i]), v)
}

/// `transform` followed by the rotation vector `rotation` about `center` and
/// the step `translation`, re-orthonormalised.
fn compose_step(
    transform: &Rigid,
    center: [f64; 3],
    translation: [f64; 3],
    rotation: [f64; 3],
) -> Rigid {
    let d = rotation_of(rotation);
    let r = &transform.rotation;
    let product: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| d[i][0] * r[0][j] + d[i][1] * r[1][j] + d[i][2] * r[2][j])
    });
    let t = sub(transform.translation, center);
    let turned = [dot(d[0], t), dot(d[1], t), dot(d[2], t)];
    Rigid {
        rotation: orthonormal(product),
        translation: add(add(center, translation), turned),
    }
}

/// Rodrigues' formula for a rotation vector.
fn rotation_of(w: [f64; 3]) -> [[f64; 3]; 3] {
    let angle = norm(w);
    // sin(a)/a and (1 - cos(a))/a², with their limits near zero.
    let (s, c) = if angle < 1e-8 {
        (1.0, 0.5)
    } else {
        (angle.sin() / angle, (1.0 - angle.cos()) / (angle * angle))
    };
    let [x, y, z] = w;
    [
        [
            1.0 - c * (y * y + z * z),
            c * x * y - s * z,
            c * x * z + s * y,
        ],
        [
            c * x * y + s * z,
            1.0 - c * (x * x + z * z),
            c * y * z - s * x,
        ],
        [
            c * x * z - s * y,
            c * y * z + s * x,
            1.0 - c * (x * x + y * y),
        ],
    ]
}

/// Gram-Schmidt on the rows; the third row is the cross product, which keeps
/// the determinant at +1. The input is within rounding of a rotation, so the
/// fallbacks are never reached in practice.
fn orthonormal(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let r0 = unit(m[0]).unwrap_or([1.0, 0.0, 0.0]);
    let d = dot(m[1], r0);
    let r1 = unit([
        m[1][0] - d * r0[0],
        m[1][1] - d * r0[1],
        m[1][2] - d * r0[2],
    ])
    .unwrap_or([0.0, 1.0, 0.0]);
    [r0, r1, cross(r0, r1)]
}

fn apply(x: &Rigid, p: [f64; 3]) -> [f64; 3] {
    let r = &x.rotation;
    let t = x.translation;
    std::array::from_fn(|i| r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2] + t[i])
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// `a` over its length, or `None` for the zero vector.
fn unit(a: [f64; 3]) -> Option<[f64; 3]> {
    let n = norm(a);
    (n > 0.0).then(|| a.map(|c| c / n))
}

#[cfg(test)]
#[path = "icp_tests.rs"]
mod tests;
