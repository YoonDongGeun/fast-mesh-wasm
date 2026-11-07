//! Triangulating one boundary loop.
//!
//! [`triangulate_loop`] is Liepa's minimum-weight dynamic programming (2003)
//! up to a loop length limit, and above it a projection onto the loop's
//! best-fit plane handed to [`fast_mesh_query::triangulate`]. Both give
//! patch triangles wound against the loop's direction, so that each one
//! agrees with the mesh triangle on the loop edge it shares.

use fast_mesh_query::triangulate::triangulate;
use fast_mesh_store::{Mesh, StoreError, VertexId};

use crate::fill::FailReason;
use crate::topology::{EdgeMap, cross, dot, point, sub};
use crate::{OpsError, Result, filled, push, reserve};

const NONE: u32 = u32::MAX;

/// Dihedral costs closer than this count as equal, so that the area decides.
const COST_EPSILON: f64 = 1e-9;

/// A triangle counts as degenerate when its doubled area is at most this
/// times its longest edge squared.
const DEGENERATE: f64 = 1e-12;

/// The weight of a partial triangulation, compared lexicographically: the
/// worst dihedral cost, then the total area.
#[derive(Clone, Copy)]
struct Weight {
    /// `1 - cos` of the worst angle between neighbouring normals: 0 when
    /// coplanar, 2 when folded flat back. Monotone in the angle, so it orders
    /// triangulations like the angle does.
    dihedral: f64,
    area: f64,
}

impl Weight {
    const NONE: Weight = Weight {
        dihedral: f64::INFINITY,
        area: f64::INFINITY,
    };

    fn is_none(self) -> bool {
        self.dihedral == f64::INFINITY
    }

    fn better_than(self, other: Weight) -> bool {
        if other.is_none() {
            return !self.is_none();
        }
        if (self.dihedral - other.dihedral).abs() > COST_EPSILON {
            self.dihedral < other.dihedral
        } else {
            self.area < other.area
        }
    }
}

/// The patch triangles closing the loop `loop_vertices` (in
/// [`BoundaryLoop::vertices`](crate::BoundaryLoop::vertices) order), each
/// wound against the loop's direction: the loop edge `a -> b` appears in its
/// patch triangle as `b -> a`.
///
/// - Up to `dp_limit` edges: the minimum-weight triangulation over the loop's
///   vertices (Liepa 2003), `W(i, k) = min over m of W(i, m) + W(m, k) +
///   weight(i, m, k)`. A triangle's weight is the worst dihedral angle to
///   its neighbours (the patch triangles already placed on its two inner
///   sides or, on a loop edge, the mesh triangle there), then its area; sums
///   take the worst angle and add the areas. A degenerate triangle costs the
///   largest angle. `O(n³)` time, `O(n²)` memory.
/// - Above it: the loop projected onto its best-fit plane (principal
///   component analysis in `f64`) and ear-clipped. The loop fails with
///   [`FailReason::ProjectionFolds`] when the projection is not a simple
///   polygon wound the loop's way, when a mesh triangle on a loop edge lands
///   on the inner side of that edge, or when the ear clipping does not finish
///   or its triangles do not tile the polygon exactly.
///
/// Either way no diagonal may be an edge `forbidden` already has (the edge
/// would turn non-manifold), no triangle may repeat a vertex id or the
/// vertex set of a triangle `forbidden` has; [`FailReason::NoValidTriangulation`]
/// otherwise. `forbidden` is the edge map of `mesh`.
///
/// `max_dihedral` caps the worst dihedral angle of the search's
/// triangulation, in radians; a best triangulation past it fails the same
/// way. It applies to the search only, not the projection.
///
/// # Errors
/// [`OpsError::Store`](crate::OpsError::Store) when memory runs out.
pub(crate) fn triangulate_loop(
    mesh: &Mesh,
    loop_vertices: &[VertexId],
    forbidden: &EdgeMap,
    dp_limit: usize,
    max_dihedral: Option<f64>,
) -> Result<std::result::Result<Vec<[VertexId; 3]>, FailReason>> {
    let n = loop_vertices.len();
    if n < 3 {
        return Ok(Err(FailReason::NoValidTriangulation));
    }
    if n <= dp_limit {
        liepa(mesh, loop_vertices, forbidden, max_dihedral)
    } else {
        projected(mesh, loop_vertices, forbidden)
    }
}

/// The unit normal of triangle `[a, b, c]` (positions), or zero when it is
/// degenerate.
fn unit_normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let n = cross(sub(b, a), sub(c, a));
    let longest = [sub(b, a), sub(c, b), sub(a, c)]
        .iter()
        .map(|e| dot(*e, *e))
        .fold(0.0, f64::max);
    let len2 = dot(n, n);
    if len2.is_nan() || len2 <= (DEGENERATE * longest).powi(2) {
        return [0.0; 3];
    }
    n.map(|x| x / len2.sqrt())
}

/// `1 - cos` of the angle between unit normals; 2 when either is zero.
fn dihedral(n: [f64; 3], m: [f64; 3]) -> f64 {
    if n == [0.0; 3] || m == [0.0; 3] {
        return 2.0;
    }
    1.0 - dot(n, m).clamp(-1.0, 1.0)
}

/// The unit normal of the mesh triangle on each loop edge `v[i] -> v[i + 1]`
/// (as stored), zero when there is none or it is degenerate.
fn edge_normals(mesh: &Mesh, v: &[VertexId], edges: &EdgeMap) -> Result<Vec<[f64; 3]>> {
    let n = v.len();
    let mut out = Vec::new();
    reserve(&mut out, n)?;
    for i in 0..n {
        let normal = match edges.triangles(v[i], v[(i + 1) % n]).first() {
            Some(&t) => {
                let [a, b, c] = mesh.triangles()[t as usize].map(|x| point(mesh, x));
                unit_normal(a, b, c)
            }
            None => [0.0; 3],
        };
        out.push(normal);
    }
    Ok(out)
}

/// Whether `[a, b, c]` has the vertex set of a triangle of the mesh.
fn existing_triangle(mesh: &Mesh, edges: &EdgeMap, [a, b, c]: [VertexId; 3]) -> bool {
    edges
        .triangles(a, b)
        .iter()
        .any(|&t| mesh.triangles()[t as usize].contains(&c))
}

/// The minimum-weight triangulation of the loop `v`.
fn liepa(
    mesh: &Mesh,
    v: &[VertexId],
    edges: &EdgeMap,
    max_dihedral: Option<f64>,
) -> Result<std::result::Result<Vec<[VertexId; 3]>, FailReason>> {
    // `Weight::dihedral` is `1 - cos`, at most 2 (folded flat back); past π
    // every triangulation passes.
    let limit = max_dihedral.map(|m| 1.0 - m.min(std::f64::consts::PI).cos());
    let n = v.len();
    let mut p = Vec::new();
    reserve(&mut p, n)?;
    p.extend(v.iter().map(|&x| point(mesh, x)));
    let loop_normal = edge_normals(mesh, v, edges)?;
    // Sub-polygon `v[i..=k]`, closed by the chord `i`-`k`, is entry `i * n + k`:
    // its weight, the apex `m` of the triangle on the chord, and that
    // triangle's normal (for a loop edge, the mesh triangle's).
    // A table too large to index cannot be allocated either.
    let cells = n
        .checked_mul(n)
        .ok_or(OpsError::Store(StoreError::OutOfMemory))?;
    let mut weight = filled(cells, Weight::NONE)?;
    let mut apex = filled(cells, NONE)?;
    let mut normal = filled(cells, [0.0; 3])?;
    for i in 0..n - 1 {
        weight[i * n + i + 1] = Weight {
            dihedral: 0.0,
            area: 0.0,
        };
        normal[i * n + i + 1] = loop_normal[i];
    }
    for gap in 2..n {
        for i in 0..n - gap {
            let k = i + gap;
            let closing = i == 0 && k == n - 1;
            // A chord that is already a mesh edge would become non-manifold.
            if !closing && (v[i] == v[k] || edges.kind(v[i], v[k]).is_some()) {
                continue;
            }
            let mut best = Weight::NONE;
            for m in i + 1..k {
                let (left, right) = (weight[i * n + m], weight[m * n + k]);
                if left.is_none() || right.is_none() || v[m] == v[i] || v[m] == v[k] {
                    continue;
                }
                if n == 3 && existing_triangle(mesh, edges, [v[0], v[1], v[2]]) {
                    continue;
                }
                // Wound against the loop: `k -> m -> i`.
                let (a, b, c) = (p[k], p[m], p[i]);
                let t = unit_normal(a, b, c);
                let mut worst = left
                    .dihedral
                    .max(right.dihedral)
                    .max(dihedral(t, normal[i * n + m]))
                    .max(dihedral(t, normal[m * n + k]));
                if closing {
                    worst = worst.max(dihedral(t, loop_normal[n - 1]));
                }
                let doubled = cross(sub(b, a), sub(c, a));
                let area = 0.5 * dot(doubled, doubled).sqrt();
                let w = Weight {
                    dihedral: worst,
                    area: left.area + right.area + area,
                };
                // Partial weights only grow as triangles join, so a partial
                // past the limit can never come back under it.
                if limit.is_some_and(|l| w.dihedral > l) {
                    continue;
                }
                if w.better_than(best) {
                    best = w;
                    apex[i * n + k] = m as u32;
                    normal[i * n + k] = t;
                }
            }
            weight[i * n + k] = best;
        }
    }
    if weight[n - 1].is_none() {
        return Ok(Err(FailReason::NoValidTriangulation));
    }
    let mut out = Vec::new();
    reserve(&mut out, n - 2)?;
    // Unfold the chords from the closing one; at most `n - 2` are pending.
    let mut stack = Vec::new();
    reserve(&mut stack, n)?;
    stack.push((0, n - 1));
    while let Some((i, k)) = stack.pop() {
        if k - i < 2 {
            continue;
        }
        // Chosen only when every alternative was forbidden.
        if normal[i * n + k] == [0.0; 3] {
            return Ok(Err(FailReason::NoValidTriangulation));
        }
        let m = apex[i * n + k] as usize;
        out.push([v[k], v[m], v[i]]);
        stack.push((m, k));
        stack.push((i, m));
    }
    Ok(Ok(out))
}

/// The eigenvector of the symmetric matrix `a` with the smallest eigenvalue,
/// by cyclic Jacobi rotations.
fn smallest_eigenvector(mut a: [[f64; 3]; 3]) -> [f64; 3] {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        let diag = a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2];
        if off.is_nan() || off <= 1e-30 * diag {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            // The rotation in the `p`-`q` plane that zeroes `a[p][q]`.
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for row in a.iter_mut().chain(v.iter_mut()) {
                let (x, y) = (row[p], row[q]);
                row[p] = c * x - s * y;
                row[q] = s * x + c * y;
            }
            let (x, y) = (a[p], a[q]);
            a[p] = std::array::from_fn(|k| c * x[k] - s * y[k]);
            a[q] = std::array::from_fn(|k| s * x[k] + c * y[k]);
        }
    }
    let i = (0..3)
        .min_by(|&i, &j| a[i][i].total_cmp(&a[j][j]))
        .expect("three eigenvalues");
    [v[0][i], v[1][i], v[2][i]]
}

/// `tris` (counter-clockwise, numbering the `n` loop points) with the loop
/// points that no triangle uses put back. Where a triangle `x, y, c` has an
/// edge `x -> y` over a run of unused points `d1 .. dk` (the loop between `x`
/// and `y`, forward), those points lie on that edge, and the triangle becomes
/// the fan `x d1 c`, `d1 d2 c`, ..., `dk y c`, keeping its winding.
fn restore_dropped(n: usize, tris: &[[u32; 3]]) -> Result<Vec<[u32; 3]>> {
    let mut used = filled(n, false)?;
    for t in tris {
        for &i in t {
            used[i as usize] = true;
        }
    }
    let mut work = Vec::new();
    reserve(&mut work, tris.len())?;
    work.extend_from_slice(tris);
    let mut out = Vec::new();
    reserve(&mut out, n.saturating_sub(2).max(tris.len()))?;
    let next = |i: u32| ((i as usize + 1) % n) as u32;
    while let Some(t) = work.pop() {
        let run = (0..3).find(|&e| {
            let (x, y) = (t[e], t[(e + 1) % 3]);
            let mut d = next(x);
            if d == y {
                return false;
            }
            while d != y {
                if used[d as usize] {
                    return false;
                }
                d = next(d);
            }
            true
        });
        let Some(e) = run else {
            push(&mut out, t)?;
            continue;
        };
        let (x, y, c) = (t[e], t[(e + 1) % 3], t[(e + 2) % 3]);
        let (mut prev, mut d) = (x, next(x));
        while d != y {
            push(&mut work, [prev, d, c])?;
            used[d as usize] = true;
            prev = d;
            d = next(d);
        }
        push(&mut work, [prev, y, c])?;
    }
    Ok(out)
}

/// Twice the signed area of the 2D triangle `a, b, c`.
fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// The loop `v` triangulated in projection onto its best-fit plane.
fn projected(
    mesh: &Mesh,
    v: &[VertexId],
    edges: &EdgeMap,
) -> Result<std::result::Result<Vec<[VertexId; 3]>, FailReason>> {
    let n = v.len();
    let mut p = Vec::new();
    reserve(&mut p, n)?;
    p.extend(v.iter().map(|&x| point(mesh, x)));
    let mut centre = [0.0; 3];
    for q in &p {
        centre = [centre[0] + q[0], centre[1] + q[1], centre[2] + q[2]];
    }
    let centre = centre.map(|c| c / n as f64);
    let mut cov = [[0.0; 3]; 3];
    // The vector area, `sum p_i x p_(i+1)` about the centre: the loop runs
    // counter-clockwise around it.
    let mut newell = [0.0; 3];
    for i in 0..n {
        let d = sub(p[i], centre);
        for (r, row) in cov.iter_mut().enumerate() {
            for (c, x) in row.iter_mut().enumerate() {
                *x += d[r] * d[c];
            }
        }
        let e = cross(d, sub(p[(i + 1) % n], centre));
        newell = [newell[0] + e[0], newell[1] + e[1], newell[2] + e[2]];
    }
    let mut normal = smallest_eigenvector(cov);
    if dot(normal, newell) < 0.0 {
        normal = normal.map(|x| -x);
    }
    // A frame `(u, w)` with `u x w = normal`, so the loop projects
    // counter-clockwise when it is not folded.
    let helper = if normal[0].abs() < 0.6 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = cross(helper, normal);
    let u = u.map(|x| x / dot(u, u).sqrt());
    let w = cross(normal, u);
    let flat = |q: [f64; 3]| {
        let d = sub(q, centre);
        [dot(d, u), dot(d, w)]
    };
    let mut flat_loop = Vec::new();
    reserve(&mut flat_loop, n)?;
    flat_loop.extend(p.iter().map(|&q| flat(q)));
    let area = flat_area(&flat_loop);
    if area.is_nan() || area <= 0.0 {
        return Ok(Err(FailReason::ProjectionFolds));
    }
    // The mesh triangle on loop edge `a -> b` lies on the edge's left seen
    // from outside, so on its right here: outside the polygon. On the
    // inner side, the surface folds over the projection.
    for i in 0..n {
        let (a, b) = (v[i], v[(i + 1) % n]);
        if let Some(&t) = edges.triangles(a, b).first() {
            let x = mesh.triangles()[t as usize]
                .into_iter()
                .find(|&x| x != a && x != b)
                .expect("proper triangle");
            let side = orient2d(flat_loop[i], flat_loop[(i + 1) % n], flat(point(mesh, x)));
            if side.is_nan() || side >= 0.0 {
                return Ok(Err(FailReason::ProjectionFolds));
            }
        }
    }
    clip_flat(mesh, v, edges, &flat_loop, false)
}

/// Twice the signed area of a flat polygon.
fn flat_area(flat_loop: &[[f64; 2]]) -> f64 {
    let origin = flat_loop[0];
    (1..flat_loop.len() - 1)
        .map(|i| orient2d(origin, flat_loop[i], flat_loop[i + 1]))
        .sum()
}

/// The loop `v`, which lies in the plane with normal `normal`, triangulated in
/// projection onto that plane by ear clipping. Near-linear in the loop's size,
/// where the search of [`triangulate_loop`] is cubic.
///
/// Unlike the best-fit projection, this does not look at the mesh triangles on
/// the loop's edges: it is for a loop that really is flat, where the walls
/// leaving it on the inner side of an edge (the part of a sphere above the cut)
/// do not fold over the patch. It fails with [`FailReason::ProjectionFolds`]
/// when the projection is not a simple polygon, and with
/// [`FailReason::NoValidTriangulation`] when a diagonal repeats a mesh edge and
/// cannot be flipped away.
///
/// # Errors
/// [`OpsError::Store`](crate::OpsError::Store) when memory runs out.
pub(crate) fn triangulate_flat_loop(
    mesh: &Mesh,
    v: &[VertexId],
    edges: &EdgeMap,
    normal: [f64; 3],
) -> Result<std::result::Result<Vec<[VertexId; 3]>, FailReason>> {
    let n = v.len();
    if n < 3 {
        return Ok(Err(FailReason::NoValidTriangulation));
    }
    let helper = if normal[0].abs() < 0.6 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = cross(helper, normal);
    let u = u.map(|x| x / dot(u, u).sqrt());
    let w = cross(normal, u);
    let mut flat_loop = Vec::new();
    reserve(&mut flat_loop, n)?;
    flat_loop.extend(v.iter().map(|&x| {
        let q = point(mesh, x);
        [dot(q, u), dot(q, w)]
    }));
    // The loop runs either way around the plane's normal, by which side the
    // walls are on: counter-clockwise is what the clipping below needs.
    let area = flat_area(&flat_loop);
    if !area.is_finite() || area == 0.0 {
        return Ok(Err(FailReason::ProjectionFolds));
    }
    if area < 0.0 {
        for q in &mut flat_loop {
            q[1] = -q[1];
        }
    }
    clip_flat(mesh, v, edges, &flat_loop, true)
}

/// The most diagonals [`flip_wall_edges`] flips: a plane close to a few vertices
/// needs a few.
const MAX_CHORD_FLIPS: usize = 64;

/// Flips diagonals of the triangulation `tris` that repeat an edge `edges` has,
/// where the two triangles on one form a convex quad. A wall edge can join two
/// rim vertices when the plane passes within rounding of the wall vertex: the
/// crossings next to it then sit almost on it. What cannot be flipped stays, for
/// the caller's check to refuse.
fn flip_wall_edges(v: &[VertexId], edges: &EdgeMap, flat_loop: &[[f64; 2]], tris: &mut [[u32; 3]]) {
    let n = v.len();
    let taken = |x: u32, y: u32| {
        let d = (x as usize + n - y as usize) % n;
        d != 1 && d != n - 1 && edges.kind(v[x as usize], v[y as usize]).is_some()
    };
    let area = |a: u32, b: u32, c: u32| {
        orient2d(
            flat_loop[a as usize],
            flat_loop[b as usize],
            flat_loop[c as usize],
        )
    };
    // A triangle rotated so that it starts with the directed edge `x -> y`.
    let rotated = |t: [u32; 3], x: u32, y: u32| {
        (0..3)
            .map(|k| [t[k], t[(k + 1) % 3], t[(k + 2) % 3]])
            .find(|r| r[0] == x && r[1] == y)
    };
    for _ in 0..MAX_CHORD_FLIPS {
        let bad = tris.iter().enumerate().find_map(|(i, t)| {
            let [a, b, c] = *t;
            [(a, b), (b, c), (c, a)]
                .into_iter()
                .find(|&(x, y)| taken(x, y))
                .map(|e| (i, e))
        });
        let Some((i, (p, q))) = bad else {
            return;
        };
        let Some((j, far)) = tris
            .iter()
            .enumerate()
            .find_map(|(j, t)| rotated(*t, q, p).map(|r| (j, r[2])))
        else {
            return;
        };
        let Some([_, _, near]) = rotated(tris[i], p, q) else {
            return;
        };
        // The quad `p, far, q, near`: the other diagonal is `far`-`near`.
        if taken(far, near) || area(p, far, near) <= 0.0 || area(far, q, near) <= 0.0 {
            return;
        }
        tris[i] = [p, far, near];
        tris[j] = [far, q, near];
    }
}

/// Ear-clips the counter-clockwise flat polygon `flat_loop` of the loop `v`,
/// wound against the loop. With `flip`, diagonals that repeat a mesh edge are
/// flipped away where they can be, rather than failing the loop.
fn clip_flat(
    mesh: &Mesh,
    v: &[VertexId],
    edges: &EdgeMap,
    flat_loop: &[[f64; 2]],
    flip: bool,
) -> Result<std::result::Result<Vec<[VertexId; 3]>, FailReason>> {
    let n = v.len();
    let polygon_area = flat_area(flat_loop);
    let mut tris = Vec::new();
    if !triangulate(flat_loop, &[], &mut tris)? {
        return Ok(Err(FailReason::ProjectionFolds));
    }
    // A simple polygon is tiled exactly by counter-clockwise triangles.
    let tiles = |tris: &[[u32; 3]]| {
        let mut tiled = 0.0;
        for t in tris {
            let [a, b, c] = t.map(|i| flat_loop[i as usize]);
            let area = orient2d(a, b, c);
            if area.is_nan() || area <= 0.0 {
                return false;
            }
            tiled += area;
        }
        (tiled - polygon_area).abs() <= 1e-9 * polygon_area
    };
    if !tiles(&tris) {
        return Ok(Err(FailReason::ProjectionFolds));
    }
    // Collinear points left out of every triangle go back in. The result
    // must still tile the polygon, with `n - 2` triangles; coincident points
    // give triangles without area and fail here.
    let mut tris = restore_dropped(n, &tris)?;
    if tris.len() != n - 2 || !tiles(&tris) {
        return Ok(Err(FailReason::NoValidTriangulation));
    }
    if flip {
        flip_wall_edges(v, edges, flat_loop, &mut tris);
    }
    let mut out = Vec::new();
    reserve(&mut out, n - 2)?;
    for t in &tris {
        let [a, b, c] = t.map(|i| v[i as usize]);
        let adjacent = |x: u32, y: u32| {
            let d = (x as usize + n - y as usize) % n;
            d == 1 || d == n - 1
        };
        let [i, j, k] = *t;
        let chords = [(i, j), (j, k), (k, i)];
        let bad_chord = chords
            .iter()
            .any(|&(x, y)| !adjacent(x, y) && edges.kind(v[x as usize], v[y as usize]).is_some());
        if a == b || b == c || a == c || bad_chord || existing_triangle(mesh, edges, [a, b, c]) {
            return Ok(Err(FailReason::NoValidTriangulation));
        }
        // Counter-clockwise like the loop: reverse to wind against it.
        out.push([c, b, a]);
    }
    Ok(Ok(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropped_collinear_points_are_put_back() {
        // A square 0, 2, 4, 6 with midpoints 1, 3, 5, 7, clipped as if the
        // midpoints had been dropped.
        let p = [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 2.0],
            [0.0, 2.0],
            [0.0, 1.0],
        ];
        let tris = restore_dropped(8, &[[0, 2, 4], [0, 4, 6]]).unwrap();
        assert_eq!(tris.len(), 6);
        let mut doubled = 0.0;
        for t in &tris {
            let area = orient2d(p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
            assert!(area > 0.0, "{t:?}");
            doubled += area;
        }
        assert_eq!(doubled, 8.0);
        assert!((0..8).all(|i| tris.iter().any(|t| t.contains(&i))));
    }

    #[test]
    fn jacobi_finds_the_smallest_axis() {
        let a = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 0.1]];
        let e = smallest_eigenvector(a);
        // A e = lambda e for some lambda, with |e| = 1.
        let ae: [f64; 3] = std::array::from_fn(|r| dot(a[r], e));
        let lambda = dot(ae, e);
        assert!((dot(e, e) - 1.0).abs() < 1e-12);
        for r in 0..3 {
            assert!((ae[r] - lambda * e[r]).abs() < 1e-9);
        }
        assert!(lambda < 0.1);
    }
}
