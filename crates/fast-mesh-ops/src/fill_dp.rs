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
pub(crate) fn unit_normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
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

/// `tris` (counter-clockwise) with dropped collinear ring vertices put back.
/// Each ring is `(start, len, forward)`, where `forward` indicates whether
/// stepping `+1` modulo `len` keeps the filled region on the left of edges.
fn restore_dropped_multi(
    total_verts: usize,
    rings: &[(usize, usize, bool)],
    tris: &[[u32; 3]],
) -> Result<Vec<[u32; 3]>> {
    let mut used = filled(total_verts, false)?;
    for t in tris {
        for &i in t {
            used[i as usize] = true;
        }
    }
    let mut ring_of = filled(total_verts, 0u32)?;
    let mut next_on_ring = filled(total_verts, 0u32)?;
    for (r_idx, &(start, len, forward)) in rings.iter().enumerate() {
        for local in 0..len {
            let v = start + local;
            ring_of[v] = r_idx as u32;
            let next_local = if forward {
                (local + 1) % len
            } else {
                (local + len - 1) % len
            };
            next_on_ring[v] = (start + next_local) as u32;
        }
    }

    let mut work = Vec::new();
    reserve(&mut work, tris.len())?;
    work.extend_from_slice(tris);
    let mut out = Vec::new();
    reserve(&mut out, total_verts.saturating_sub(2).max(tris.len()))?;
    while let Some(t) = work.pop() {
        let run = (0..3).find(|&e| {
            let (x, y) = (t[e], t[(e + 1) % 3]);
            if ring_of[x as usize] != ring_of[y as usize] {
                return false;
            }
            let mut d = next_on_ring[x as usize];
            if d == y {
                return false;
            }
            let max_steps = rings[ring_of[x as usize] as usize].1;
            let mut steps = 0usize;
            while d != y {
                if used[d as usize] || steps >= max_steps {
                    return false;
                }
                d = next_on_ring[d as usize];
                steps += 1;
            }
            true
        });
        let Some(e) = run else {
            push(&mut out, t)?;
            continue;
        };
        let (x, y, c) = (t[e], t[(e + 1) % 3], t[(e + 2) % 3]);
        let mut prev = x;
        let mut d = next_on_ring[x as usize];
        while d != y {
            push(&mut work, [prev, d, c])?;
            used[d as usize] = true;
            prev = d;
            d = next_on_ring[d as usize];
        }
        push(&mut work, [prev, y, c])?;
    }
    Ok(out)
}

/// Splits any unpaired non-boundary halfedge `(x, y)` when ring vertices lie
/// strictly collinearly along `flat[x] -> flat[y]` (which happens when earcut
/// bridges a hole to an outer vertex across collinear boundary/bridge edges).
fn split_collinear_halfedges(
    flat: &[[f64; 2]],
    rings: &[(usize, usize, bool)],
    tris: &mut Vec<[u32; 3]>,
) -> Result<()> {
    let total_verts = flat.len();
    let mut next_on_ring = filled(total_verts, 0u32)?;
    for &(start, len, forward) in rings {
        for local in 0..len {
            let v = start + local;
            let next_local = if forward {
                (local + 1) % len
            } else {
                (local + len - 1) % len
            };
            next_on_ring[v] = (start + next_local) as u32;
        }
    }

    let mut on_seg: Vec<(f64, u32)> = Vec::new();
    let max_passes = total_verts + 4;
    for _ in 0..max_passes {
        let mut dir_edges: Vec<(u32, u32)> = Vec::new();
        reserve(&mut dir_edges, tris.len() * 3)?;
        for t in tris.iter() {
            dir_edges.push((t[0], t[1]));
            dir_edges.push((t[1], t[2]));
            dir_edges.push((t[2], t[0]));
        }
        dir_edges.sort_unstable();

        let mut split_target: Option<(usize, usize, Vec<u32>)> = None;
        for (ti, t) in tris.iter().enumerate() {
            for e in 0..3 {
                let (x, y) = (t[e], t[(e + 1) % 3]);
                if next_on_ring[x as usize] == y {
                    continue;
                }
                if dir_edges.binary_search(&(y, x)).is_ok() {
                    continue;
                }
                let (px, py) = (flat[x as usize], flat[y as usize]);
                let dx = py[0] - px[0];
                let dy = py[1] - px[1];
                let len2 = dx * dx + dy * dy;
                if !(len2.is_finite() && len2 > 0.0) {
                    continue;
                }
                on_seg.clear();
                for (vi, &pv) in flat.iter().enumerate() {
                    let v = vi as u32;
                    if v == x || v == y {
                        continue;
                    }
                    let cross = orient2d(px, py, pv);
                    if cross.abs() > 1e-10 * len2 {
                        continue;
                    }
                    let proj = (pv[0] - px[0]) * dx + (pv[1] - px[1]) * dy;
                    let param = proj / len2;
                    if param > 1e-10 && param < 1.0 - 1e-10 {
                        push(&mut on_seg, (param, v))?;
                    }
                }
                if !on_seg.is_empty() {
                    on_seg.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                    let mut mids: Vec<u32> = Vec::new();
                    reserve(&mut mids, on_seg.len())?;
                    for &(_, v) in &on_seg {
                        mids.push(v);
                    }
                    split_target = Some((ti, e, mids));
                    break;
                }
            }
            if split_target.is_some() {
                break;
            }
        }
        let Some((ti, e, mids)) = split_target else {
            break;
        };
        let t = tris[ti];
        let (x, y, c) = (t[e], t[(e + 1) % 3], t[(e + 2) % 3]);
        tris[ti] = [x, mids[0], c];
        reserve(tris, mids.len())?;
        let mut prev = mids[0];
        for &m in &mids[1..] {
            tris.push([prev, m, c]);
            prev = m;
        }
        tris.push([prev, y, c]);
    }
    Ok(())
}

/// Restores collinear points dropped along a single counter-clockwise loop.
fn restore_dropped(n: usize, tris: &[[u32; 3]]) -> Result<Vec<[u32; 3]>> {
    restore_dropped_multi(n, &[(0, n, true)], tris)
}

/// Best-fit projection plane `(centre, normal, u, w)` of a closed loop.
#[derive(Clone, Copy)]
pub(crate) struct PlaneFrame {
    pub(crate) centre: [f64; 3],
    pub(crate) normal: [f64; 3],
    pub(crate) u: [f64; 3],
    pub(crate) w: [f64; 3],
}

impl PlaneFrame {
    /// Projects `q` into the 2D right-handed plane frame `(u, w)`.
    pub(crate) fn project(&self, q: [f64; 3]) -> [f64; 2] {
        let d = sub(q, self.centre);
        [dot(d, self.u), dot(d, self.w)]
    }
}

/// Computes the PCA best-fit plane of `v`, oriented so that `v` winds
/// counter-clockwise around `normal` (matching its Newell vector).
///
/// # Errors
/// [`OpsError::Store`](crate::OpsError::Store) when memory runs out.
pub(crate) fn best_fit_frame(mesh: &Mesh, v: &[VertexId]) -> Result<PlaneFrame> {
    let n = v.len();
    let mut p = Vec::new();
    reserve(&mut p, n)?;
    p.extend(v.iter().map(|&x| point(mesh, x)));
    let mut centre = [0.0; 3];
    for q in &p {
        centre = [centre[0] + q[0], centre[1] + q[1], centre[2] + q[2]];
    }
    let centre = centre.map(|c| c / n.max(1) as f64);
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
    let helper = if normal[0].abs() < 0.6 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let raw_u = cross(helper, normal);
    let u_len = dot(raw_u, raw_u).sqrt();
    let u = if u_len > 0.0 && u_len.is_finite() {
        raw_u.map(|x| x / u_len)
    } else {
        [1.0, 0.0, 0.0]
    };
    let w = cross(normal, u);
    Ok(PlaneFrame {
        centre,
        normal,
        u,
        w,
    })
}

/// Twice the signed area of the 2D triangle `a, b, c`.
pub(crate) fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// The loop `v` triangulated in projection onto its best-fit plane.
fn projected(
    mesh: &Mesh,
    v: &[VertexId],
    edges: &EdgeMap,
) -> Result<std::result::Result<Vec<[VertexId; 3]>, FailReason>> {
    let n = v.len();
    let frame = best_fit_frame(mesh, v)?;
    let mut flat_loop = Vec::new();
    reserve(&mut flat_loop, n)?;
    flat_loop.extend(v.iter().map(|&x| frame.project(point(mesh, x))));
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
            let side = orient2d(
                flat_loop[i],
                flat_loop[(i + 1) % n],
                frame.project(point(mesh, x)),
            );
            if side.is_nan() || side >= 0.0 {
                return Ok(Err(FailReason::ProjectionFolds));
            }
        }
    }
    clip_flat(mesh, v, edges, &flat_loop, false)
}

/// Twice the signed area of a flat polygon.
pub(crate) fn flat_area(flat_loop: &[[f64; 2]]) -> f64 {
    let origin = flat_loop[0];
    (1..flat_loop.len() - 1)
        .map(|i| orient2d(origin, flat_loop[i], flat_loop[i + 1]))
        .sum()
}

/// `true` when `v` behaves like the outer perimeter of an approximately planar
/// open sheet (such as a grid border or an island rim): adjacent mesh triangles
/// are nearly coplanar with `v`'s plane and lie strictly inside `v`'s 2D
/// projection, so filling `v` directly would fold triangles back over the mesh.
///
/// # Errors
/// [`OpsError::Store`](crate::OpsError::Store) when memory runs out.
pub(crate) fn is_coplanar_outer_rim(mesh: &Mesh, v: &[VertexId], edges: &EdgeMap) -> Result<bool> {
    let n = v.len();
    if n < 3 {
        return Ok(false);
    }
    let frame = best_fit_frame(mesh, v)?;
    let mut flat_loop = Vec::new();
    reserve(&mut flat_loop, n)?;
    flat_loop.extend(v.iter().map(|&x| frame.project(point(mesh, x))));
    let area = flat_area(&flat_loop);
    if !area.is_finite() || area <= 0.0 {
        return Ok(false);
    }
    let mut checked = 0usize;
    let mut coplanar_inside = 0usize;
    for i in 0..n {
        let (a, b) = (v[i], v[(i + 1) % n]);
        let Some(&t_idx) = edges.triangles(a, b).first() else {
            continue;
        };
        let tri = mesh.triangles()[t_idx as usize];
        let Some(x) = tri.into_iter().find(|&u| u != a && u != b) else {
            continue;
        };
        let tri_n = unit_normal(
            point(mesh, tri[0]),
            point(mesh, tri[1]),
            point(mesh, tri[2]),
        );
        if tri_n == [0.0; 3] {
            continue;
        }
        checked += 1;
        let side = orient2d(
            flat_loop[i],
            flat_loop[(i + 1) % n],
            frame.project(point(mesh, x)),
        );
        if side.is_finite() && side > 0.0 && dot(tri_n, frame.normal) > 0.25 {
            coplanar_inside += 1;
        }
    }
    Ok(checked > 0 && coplanar_inside * 4 >= checked * 3)
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

/// Triangulates an `outer` boundary loop containing zero or more nested inner
/// `islands` via projected multi-ring ear clipping, collinear restoration, and
/// constrained 2D Delaunay edge relaxation.
///
/// Output triangles are wound consistently against `outer` (and against inner
/// islands whose loop directions oppose `outer`).
///
/// # Errors
/// [`OpsError::Store`](crate::OpsError::Store) when memory runs out.
pub(crate) fn triangulate_with_islands(
    mesh: &Mesh,
    outer: &[VertexId],
    islands: &[&[VertexId]],
    edges: &EdgeMap,
) -> Result<std::result::Result<Vec<[VertexId; 3]>, FailReason>> {
    if islands.is_empty() {
        return triangulate_loop(mesh, outer, edges, 200, None);
    }
    if outer.len() < 3 || islands.iter().any(|isl| isl.len() < 3) {
        return Ok(Err(FailReason::NoValidTriangulation));
    }

    let frame = best_fit_frame(mesh, outer)?;
    let total_verts = islands.iter().fold(outer.len(), |s, isl| s + isl.len());
    let mut all_verts = Vec::new();
    reserve(&mut all_verts, total_verts)?;
    let mut flat_all = Vec::new();
    reserve(&mut flat_all, total_verts)?;

    all_verts.extend_from_slice(outer);
    flat_all.extend(outer.iter().map(|&v| frame.project(point(mesh, v))));
    let outer_area = flat_area(&flat_all[..outer.len()]);
    if !outer_area.is_finite() || outer_area <= 0.0 {
        return Ok(Err(FailReason::ProjectionFolds));
    }

    // Check adjacent mesh faces outside `outer` do not fold over it.
    for i in 0..outer.len() {
        let (a, b) = (outer[i], outer[(i + 1) % outer.len()]);
        if let Some(&t) = edges.triangles(a, b).first() {
            let Some(x) = mesh.triangles()[t as usize]
                .into_iter()
                .find(|&u| u != a && u != b)
            else {
                continue;
            };
            let side = orient2d(
                flat_all[i],
                flat_all[(i + 1) % outer.len()],
                frame.project(point(mesh, x)),
            );
            if side.is_nan() || side >= 0.0 {
                return Ok(Err(FailReason::ProjectionFolds));
            }
        }
    }

    let mut rings: Vec<(usize, usize, bool)> = Vec::new();
    reserve(&mut rings, 1 + islands.len())?;
    rings.push((0, outer.len(), true));

    let mut target_area = outer_area;
    let mut boundary_edges: Vec<(u32, u32)> = Vec::new();
    reserve(&mut boundary_edges, total_verts)?;
    for i in 0..outer.len() {
        let u = i as u32;
        let v = ((i + 1) % outer.len()) as u32;
        boundary_edges.push((u.min(v), u.max(v)));
    }

    for &isl in islands {
        let start = flat_all.len();
        all_verts.extend_from_slice(isl);
        flat_all.extend(isl.iter().map(|&v| frame.project(point(mesh, v))));
        let isl_area = flat_area(&flat_all[start..]);
        if !isl_area.is_finite() || isl_area == 0.0 {
            return Ok(Err(FailReason::ProjectionFolds));
        }
        target_area -= isl_area.abs();
        rings.push((start, isl.len(), isl_area < 0.0));
        for i in 0..isl.len() {
            let u = (start + i) as u32;
            let v = (start + (i + 1) % isl.len()) as u32;
            boundary_edges.push((u.min(v), u.max(v)));
        }
    }
    if !target_area.is_finite() || target_area <= 0.0 {
        return Ok(Err(FailReason::ProjectionFolds));
    }
    boundary_edges.sort_unstable();
    boundary_edges.dedup();

    let hole_slices: Vec<&[[f64; 2]]> = rings[1..]
        .iter()
        .map(|&(start, len, _)| &flat_all[start..start + len])
        .collect();
    let mut tris = Vec::new();
    if !triangulate(&flat_all[..outer.len()], &hole_slices, &mut tris)? {
        return Ok(Err(FailReason::ProjectionFolds));
    }

    let tiles = |cand: &[[u32; 3]]| {
        let mut sum = 0.0;
        for t in cand {
            let [a, b, c] = t.map(|idx| flat_all[idx as usize]);
            let a2 = orient2d(a, b, c);
            if a2.is_nan() || a2 <= 0.0 {
                return false;
            }
            sum += a2;
        }
        (sum - target_area).abs() <= 1e-7 * target_area
    };
    if !tiles(&tris) {
        return Ok(Err(FailReason::ProjectionFolds));
    }

    split_collinear_halfedges(&flat_all, &rings, &mut tris)?;
    let mut tris = restore_dropped_multi(total_verts, &rings, &tris)?;
    let expected_tris = total_verts + 2 * islands.len() - 2;
    if tris.len() != expected_tris || !tiles(&tris) {
        return Ok(Err(FailReason::NoValidTriangulation));
    }

    constrain_delaunay_2d(&flat_all, &all_verts, &boundary_edges, edges, &mut tris)?;

    let mut out = Vec::new();
    reserve(&mut out, tris.len())?;
    for t in &tris {
        let [a, b, c] = t.map(|idx| all_verts[idx as usize]);
        let [i, j, k] = *t;
        let bad_chord = [(i, j), (j, k), (k, i)].iter().any(|&(x, y)| {
            let e = (x.min(y), x.max(y));
            boundary_edges.binary_search(&e).is_err()
                && edges
                    .kind(all_verts[x as usize], all_verts[y as usize])
                    .is_some()
        });
        if a == b || b == c || a == c || bad_chord || existing_triangle(mesh, edges, [a, b, c]) {
            return Ok(Err(FailReason::NoValidTriangulation));
        }
        out.push([c, b, a]);
    }
    Ok(Ok(out))
}

/// Relaxes interior 2D triangles via constrained Delaunay edge flips while
/// preserving all constrained boundary edges and avoiding existing mesh chords.
fn constrain_delaunay_2d(
    flat: &[[f64; 2]],
    all_verts: &[VertexId],
    boundary_edges: &[(u32, u32)],
    mesh_edges: &EdgeMap,
    tris: &mut [[u32; 3]],
) -> Result<()> {
    let nt = tris.len();
    if nt < 2 {
        return Ok(());
    }
    for _ in 0..16 {
        let mut records: Vec<(u32, u32, bool, u32, u32)> = Vec::new();
        reserve(&mut records, 3 * nt)?;
        for (ti, &t) in tris.iter().enumerate() {
            for e in 0..3 {
                let (u, v, opp) = (t[e], t[(e + 1) % 3], t[(e + 2) % 3]);
                let (lo, hi, fwd) = if u < v { (u, v, true) } else { (v, u, false) };
                records.push((lo, hi, fwd, ti as u32, opp));
            }
        }
        records.sort_unstable();

        let mut existing: Vec<(u32, u32)> = Vec::new();
        reserve(&mut existing, records.len())?;
        for r in &records {
            if existing.last().copied() != Some((r.0, r.1)) {
                existing.push((r.0, r.1));
            }
        }

        let mut touched = filled(nt, false)?;
        let mut added_diags: Vec<(u32, u32)> = Vec::new();
        let mut flipped = false;

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
                if r0.2 != r1.2 && boundary_edges.binary_search(&(lo, hi)).is_err() {
                    let (fwd_rec, rev_rec) = if r0.2 { (r0, r1) } else { (r1, r0) };
                    let (t1, a) = (fwd_rec.3 as usize, fwd_rec.4);
                    let (t2, b) = (rev_rec.3 as usize, rev_rec.4);
                    if !touched[t1] && !touched[t2] && a != b {
                        let diag = (a.min(b), a.max(b));
                        let dup_mesh = mesh_edges
                            .kind(all_verts[diag.0 as usize], all_verts[diag.1 as usize])
                            .is_some()
                            && boundary_edges.binary_search(&diag).is_err();
                        let dup = dup_mesh
                            || existing.binary_search(&diag).is_ok()
                            || added_diags.contains(&diag);
                        if !dup && should_flip_2d(flat, lo, hi, a, b, all_verts, mesh_edges) {
                            tris[t1] = [b, hi, a];
                            tris[t2] = [a, lo, b];
                            touched[t1] = true;
                            touched[t2] = true;
                            push(&mut added_diags, diag)?;
                            flipped = true;
                        }
                    }
                }
            }
            idx = end;
        }
        if !flipped {
            break;
        }
    }
    Ok(())
}

fn should_flip_2d(
    flat: &[[f64; 2]],
    lo: u32,
    hi: u32,
    a: u32,
    b: u32,
    all_verts: &[VertexId],
    mesh_edges: &EdgeMap,
) -> bool {
    let (plo, phi, pa, pb) = (
        flat[lo as usize],
        flat[hi as usize],
        flat[a as usize],
        flat[b as usize],
    );
    // Flipped CCW triangles: [pb, phi, pa] and [pa, plo, pb].
    let area1 = orient2d(pb, phi, pa);
    let area2 = orient2d(pa, plo, pb);
    if !(area1.is_finite() && area2.is_finite() && area1 > 1e-12 && area2 > 1e-12) {
        return false;
    }
    let old1 = orient2d(pa, plo, phi);
    let old2 = orient2d(pb, phi, plo);
    if old1 <= 1e-12 || old2 <= 1e-12 {
        return false;
    }
    let is_bad_chord = mesh_edges
        .kind(all_verts[lo as usize], all_verts[hi as usize])
        .is_some();
    let cot2 = |p: [f64; 2], q: [f64; 2], r: [f64; 2], area: f64| {
        ((q[0] - p[0]) * (r[0] - p[0]) + (q[1] - p[1]) * (r[1] - p[1])) / area
    };
    let old_cot = cot2(pa, plo, phi, old1) + cot2(pb, phi, plo, old2);
    if is_bad_chord {
        return true;
    }
    let new_cot = cot2(plo, pb, pa, area2) + cot2(phi, pa, pb, area1);
    old_cot < -1e-6 && new_cot > old_cot + 1e-6
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

    #[test]
    fn triangulate_with_islands_and_outer_rim_detection() {
        use crate::boundary_loops;
        use crate::test_util::*;

        // 8x8 grid with middle 4x4 carved out and a 2x2 island kept at the center (3..5, 3..5).
        let mut cells = Vec::new();
        for y in 2..6u32 {
            for x in 2..6u32 {
                if !((3..5).contains(&x) && (3..5).contains(&y)) {
                    cells.push((x, y));
                }
            }
        }
        // Build combined mesh keeping the center island (3..5, 3..5) separated by empty ring.
        // Wait: (3,3)..(5,5) shares corners only if diagonal cells exist; removing 1..7 keeping 3..5 leaves clean moat.
        let mut moat = Vec::new();
        for y in 1..7u32 {
            for x in 1..7u32 {
                if !((3..5).contains(&x) && (3..5).contains(&y)) {
                    moat.push((x, y));
                }
            }
        }
        drop(cells);
        let mesh = grid_without(8, &moat);
        let loops = boundary_loops(&mesh).unwrap();
        assert_eq!(loops.len(), 3);
        let edges = EdgeMap::build(&mesh).unwrap();

        // Outer sheet border: longest perimeter (32 edges), adjacent triangles coplanar inside.
        let outer_border = loops.iter().max_by_key(|l| l.vertices.len()).unwrap();
        assert!(is_coplanar_outer_rim(&mesh, &outer_border.vertices, &edges).unwrap());

        // Inner moat hole (24 edges, from x/y in 1..7): adjacent triangles lie outside.
        let hole = loops
            .iter()
            .find(|l| l.vertices.len() == 24)
            .expect("24-vertex hole");
        assert!(!is_coplanar_outer_rim(&mesh, &hole.vertices, &edges).unwrap());

        // Center 2x2 island border (8 edges): coplanar outer rim of the island itself.
        let island = loops
            .iter()
            .min_by_key(|l| l.vertices.len())
            .expect("8-vertex island");
        assert_eq!(island.vertices.len(), 8);
        assert!(is_coplanar_outer_rim(&mesh, &island.vertices, &edges).unwrap());

        let tris = triangulate_with_islands(&mesh, &hole.vertices, &[&island.vertices], &edges)
            .unwrap()
            .unwrap();
        assert_eq!(tris.len(), 24 + 8);
        // Combined mesh + patch triangles must be closed around the island/hole (only outer_border remains!).
        let (_, mut base_tris) = parts(&mesh);
        let (pos, _) = parts(&mesh);
        for t in tris {
            base_tris.push(t);
        }
        let patched = mesh_of(&pos, &base_tris);
        let after = boundary_loops(&patched).unwrap();
        assert_eq!(
            after.len(),
            1,
            "expected only outer sheet border left: {after:?}"
        );
        assert_eq!(after[0].vertices.len(), outer_border.vertices.len());
    }
}
