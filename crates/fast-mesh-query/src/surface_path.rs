//! Intrinsic surface shortest path extraction from a vertex distance field.
//!
//! [`shortest_surface_path`] traces steepest descent along a scalar vertex
//! distance field (such as the geodesic distance field produced by Dijkstra or
//! Crane's heat method in `fast-mesh-fields`) across triangle interiors and
//! shared edges from `target_vertex` back to the closest reachable source, then
//! reverses the sequence so it starts on a source vertex and ends at
//! `target_vertex`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use fast_mesh_store::Mesh;

use crate::{Appender, QueryError, Result, allocation_allowed};

/// Relative snap threshold to treat a barycentric coordinate near `0` or `1` as
/// hitting a vertex instead of remaining inside an edge.
const EDGE_SNAP_EPS: f64 = 1e-5;

/// Sample along a surface path returned by [`shortest_surface_path`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfacePathPoint {
    /// 3D coordinates of this point on [`Self::triangle`].
    pub position: [f32; 3],
    /// Live triangle slot containing this point.
    pub triangle: u32,
    /// Non-negative barycentric coordinates `[w0, w1, w2]` summing to `1.0`
    /// with respect to `mesh.triangle(self.triangle)`.
    pub barycentric: [f32; 3],
}

#[inline]
fn try_reserve_exact<T>(v: &mut Vec<T>, additional: usize) -> Result<()> {
    if additional > 0 {
        allocation_allowed()?;
        v.try_reserve_exact(additional)
            .map_err(|_| QueryError::OutOfMemory)?;
    }
    Ok(())
}

#[inline]
fn try_filled<T: Copy>(len: usize, val: T) -> Result<Vec<T>> {
    let mut v = Vec::new();
    try_reserve_exact(&mut v, len)?;
    v.resize(len, val);
    Ok(v)
}

#[inline]
fn try_push<T>(v: &mut Vec<T>, item: T) -> Result<()> {
    if v.len() == v.capacity() {
        allocation_allowed()?;
        v.try_reserve(1).map_err(|_| QueryError::OutOfMemory)?;
    }
    v.push(item);
    Ok(())
}

#[inline]
fn widen(p: [f32; 3]) -> [f64; 3] {
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Geometry of a non-degenerate triangle oriented as `(v0, v1, v2)` along with
/// the steepest-descent vector `g = -grad(phi)` in the triangle plane.
struct TriDescent {
    v: [u32; 3],
    p: [[f64; 3]; 3],
    area2: f64,
    n_hat: [f64; 3],
    grad_vecs: [[f64; 3]; 3],
    descent: [f64; 3],
    descent_norm: f64,
}

fn tri_descent(mesh: &Mesh, distances: &[f32], tri_id: u32) -> Option<TriDescent> {
    let tri = mesh.triangle(tri_id)?;
    if tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0] {
        return None;
    }
    let phi = [
        f64::from(distances[tri[0] as usize]),
        f64::from(distances[tri[1] as usize]),
        f64::from(distances[tri[2] as usize]),
    ];
    if !phi.iter().all(|v| v.is_finite()) {
        return None;
    }
    let pos = mesh.positions();
    let p = tri.map(|idx| widen(pos[idx as usize]));
    let e0 = sub(p[2], p[1]);
    let e1 = sub(p[0], p[2]);
    let e2 = sub(p[1], p[0]);
    let n = cross(e2, sub(p[2], p[0]));
    let area2 = norm(n);
    let max_e_sq = dot(e0, e0).max(dot(e1, e1)).max(dot(e2, e2));
    if !area2.is_finite() || area2 <= 1e-24 * max_e_sq || area2 == 0.0 {
        return None;
    }

    let n_hat = scale(n, 1.0 / area2);
    let inv_area2 = 1.0 / area2;
    let grad_vecs = [
        scale(cross(n_hat, e0), inv_area2),
        scale(cross(n_hat, e1), inv_area2),
        scale(cross(n_hat, e2), inv_area2),
    ];
    let grad = add(
        add(scale(grad_vecs[0], phi[0]), scale(grad_vecs[1], phi[1])),
        scale(grad_vecs[2], phi[2]),
    );
    let descent = scale(grad, -1.0);
    let descent_norm = norm(descent);
    Some(TriDescent {
        v: tri,
        p,
        area2,
        n_hat,
        grad_vecs,
        descent,
        descent_norm,
    })
}

fn make_vertex_point(
    mesh: &Mesh,
    v: u32,
    v_tri_off: &[u32],
    v_tris: &[u32],
) -> Result<SurfacePathPoint> {
    let start = v_tri_off[v as usize] as usize;
    let end = v_tri_off[v as usize + 1] as usize;
    let incident = &v_tris[start..end];
    if incident.is_empty() {
        return Err(QueryError::InvalidInput(
            "vertex has no incident live triangle",
        ));
    }
    let pos = mesh.positions();
    let p_v = pos[v as usize];
    // Prefer a non-degenerate triangle when selecting the carrier triangle.
    let mut chosen = None;
    for &t in incident {
        if let Some(corners) = mesh.triangle(t) {
            if chosen.is_none() {
                chosen = Some((t, corners));
            }
            let p = corners.map(|idx| widen(pos[idx as usize]));
            let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
            if dot(n, n) > 0.0 {
                chosen = Some((t, corners));
                break;
            }
        }
    }
    let (tri_id, corners) = chosen.ok_or(QueryError::InvalidInput(
        "vertex has no incident live triangle",
    ))?;
    let mut bary = [0.0f32; 3];
    for k in 0..3 {
        if corners[k] == v {
            bary[k] = 1.0;
            break;
        }
    }
    Ok(SurfacePathPoint {
        position: p_v,
        triangle: tri_id,
        barycentric: bary,
    })
}

fn make_edge_point(mesh: &Mesh, tri_id: u32, a: u32, b: u32, wa: f64) -> Option<SurfacePathPoint> {
    let corners = mesh.triangle(tri_id)?;
    let wa = wa.clamp(0.0, 1.0);
    let wb = 1.0 - wa;
    let mut bary64 = [0.0f64; 3];
    for k in 0..3 {
        if corners[k] == a {
            bary64[k] = wa;
        } else if corners[k] == b {
            bary64[k] = wb;
        }
    }
    let pos = mesh.positions();
    let p0 = widen(pos[corners[0] as usize]);
    let p1 = widen(pos[corners[1] as usize]);
    let p2 = widen(pos[corners[2] as usize]);
    let p = add(
        add(scale(p0, bary64[0]), scale(p1, bary64[1])),
        scale(p2, bary64[2]),
    );
    let b0 = bary64[0] as f32;
    let b1 = bary64[1] as f32;
    let b2 = bary64[2] as f32;
    Some(SurfacePathPoint {
        position: [p[0] as f32, p[1] as f32, p[2] as f32],
        triangle: tri_id,
        barycentric: [b0, b1, b2],
    })
}

/// Finds the unique live triangle different from `from_tri` containing edge `(a, b)`,
/// or `None` if `(a, b)` is a boundary or non-manifold edge.
fn opposite_triangle(
    a: u32,
    b: u32,
    from_tri: u32,
    v_tri_off: &[u32],
    v_tris: &[u32],
) -> Option<u32> {
    let sa = &v_tris[v_tri_off[a as usize] as usize..v_tri_off[a as usize + 1] as usize];
    let sb = &v_tris[v_tri_off[b as usize] as usize..v_tri_off[b as usize + 1] as usize];
    let mut found: Option<u32> = None;
    for &ta in sa {
        if ta == from_tri {
            continue;
        }
        if sb.contains(&ta) {
            if found.is_some() {
                return None; // Non-manifold edge
            }
            found = Some(ta);
        }
    }
    found
}

/// Traces an intrinsic shortest surface path from `sources` to `target_vertex`
/// along the vertex distance field `distances`, writing path points in order
/// from the reached source to `target_vertex` into `out`.
///
/// Tracing proceeds downhill from `target_vertex` via steepest-descent across
/// triangle interiors and edges (with graph-based escape across flat regions or
/// local minima), using only fallible allocations compatible with OOM fault
/// injection.
///
/// # Errors
/// - [`QueryError::InvalidInput`] if `distances.len() != mesh.vertex_slots()`,
///   `sources` is empty, `target_vertex` or any source is out of range / dead /
///   lacks incident live triangles, `distances[target_vertex]` is non-finite or
///   negative, any live triangle vertex position is non-finite, or no source in
///   `sources` is reachable in `target_vertex`'s connected component.
/// - [`QueryError::OutOfMemory`] if any internal or output allocation fails.
///
/// On error `out` is left unchanged.
pub fn shortest_surface_path(
    mesh: &Mesh,
    distances: &[f32],
    sources: &[u32],
    target_vertex: u32,
    out: &mut Vec<SurfacePathPoint>,
) -> Result<()> {
    let slots = mesh.vertex_slots();
    if distances.len() != slots {
        return Err(QueryError::InvalidInput(
            "distances length must equal mesh.vertex_slots()",
        ));
    }
    if sources.is_empty() {
        return Err(QueryError::InvalidInput("sources must not be empty"));
    }
    if target_vertex as usize >= slots || !mesh.is_vertex_alive(target_vertex) {
        return Err(QueryError::InvalidInput(
            "target_vertex is out of range or dead",
        ));
    }
    for &s in sources {
        if s as usize >= slots || !mesh.is_vertex_alive(s) {
            return Err(QueryError::InvalidInput(
                "source vertex is out of range or dead",
            ));
        }
    }
    let target_dist = f64::from(distances[target_vertex as usize]);
    if !target_dist.is_finite() || target_dist < 0.0 {
        return Err(QueryError::InvalidInput(
            "target_vertex distance must be finite and non-negative",
        ));
    }

    let pos = mesh.positions();
    let mut deg = try_filled(slots + 1, 0u32)?;
    for t in 0..mesh.triangle_slots() as u32 {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        for &v in &tri {
            if !pos[v as usize].iter().all(|c| c.is_finite()) {
                return Err(QueryError::InvalidInput(
                    "mesh vertex position must be finite",
                ));
            }
            deg[v as usize + 1] += 1;
        }
    }
    for i in 0..slots {
        deg[i + 1] += deg[i];
    }
    let mut v_tris = try_filled(deg[slots] as usize, 0u32)?;
    let mut fill = try_filled(slots, 0u32)?;
    for t in 0..mesh.triangle_slots() as u32 {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        for &v in &tri {
            let p = (deg[v as usize] + fill[v as usize]) as usize;
            v_tris[p] = t;
            fill[v as usize] += 1;
        }
    }
    drop(fill);

    let mut is_source = try_filled(slots, false)?;
    for &s in sources {
        is_source[s as usize] = true;
    }

    let start_pt = make_vertex_point(mesh, target_vertex, &deg, &v_tris)?;
    if is_source[target_vertex as usize] {
        let orig_len = out.len();
        let mut ap = Appender::new(out);
        ap.push(start_pt)?;
        let _ = ap;
        out.drain(..orig_len);
        return Ok(());
    }

    trace_path(
        mesh,
        distances,
        &is_source,
        &deg,
        &v_tris,
        target_vertex,
        start_pt,
        out,
    )
}

#[derive(Clone, Copy)]
enum TraceState {
    AtVertex(u32),
    OnEdge {
        from_tri: u32,
        a: u32,
        b: u32,
        wa: f64,
    },
}

enum StepChoice {
    Vertex {
        next: u32,
        next_phi: f64,
        slope: f64,
    },
    Edge {
        tri: u32,
        a: u32,
        b: u32,
        wa: f64,
        next_phi: f64,
        slope: f64,
    },
}

impl StepChoice {
    fn slope(&self) -> f64 {
        match *self {
            Self::Vertex { slope, .. } | Self::Edge { slope, .. } => slope,
        }
    }
}

fn best_vertex_step(
    mesh: &Mesh,
    distances: &[f32],
    v_tri_off: &[u32],
    v_tris: &[u32],
    visited: &[bool],
    v: u32,
    cur_phi: f64,
) -> Option<StepChoice> {
    let pos = mesh.positions();
    let p_v = widen(pos[v as usize]);
    let eps_phi = 1e-12 * cur_phi.abs().max(1.0);
    let mut best: Option<StepChoice> = None;

    let start = v_tri_off[v as usize] as usize;
    let end = v_tri_off[v as usize + 1] as usize;
    for &t in &v_tris[start..end] {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        // Candidate edge moves along incident triangle edges.
        for &u in &tri {
            if u == v || visited[u as usize] {
                continue;
            }
            let phi_u = f64::from(distances[u as usize]);
            if !phi_u.is_finite() || phi_u >= cur_phi - eps_phi {
                continue;
            }
            let edge_len = norm(sub(widen(pos[u as usize]), p_v));
            if edge_len <= 0.0 {
                continue;
            }
            let slope = (cur_phi - phi_u) / edge_len;
            if best.as_ref().is_none_or(|b| slope > b.slope()) {
                best = Some(StepChoice::Vertex {
                    next: u,
                    next_phi: phi_u,
                    slope,
                });
            }
        }

        // Candidate interior gradient descent through triangle `t`.
        let Some(td) = tri_descent(mesh, distances, t) else {
            continue;
        };
        if td.descent_norm <= 0.0 || !td.descent_norm.is_finite() {
            continue;
        }
        let Some(k) = (0..3).find(|&i| td.v[i] == v) else {
            continue;
        };
        let b = td.v[(k + 1) % 3];
        let c = td.v[(k + 2) % 3];
        let eb = sub(td.p[(k + 1) % 3], td.p[k]);
        let ec = sub(td.p[(k + 2) % 3], td.p[k]);

        let alpha = dot(td.descent, cross(ec, td.n_hat)) / td.area2;
        let beta = dot(td.descent, cross(td.n_hat, eb)) / td.area2;
        let denom = alpha + beta;
        if denom <= 0.0 || !denom.is_finite() {
            continue;
        }
        if alpha <= EDGE_SNAP_EPS * denom || beta <= EDGE_SNAP_EPS * denom {
            continue;
        }
        let wb = (alpha / denom).clamp(0.0, 1.0);
        let wc = 1.0 - wb;
        let phi_b = f64::from(distances[b as usize]);
        let phi_c = f64::from(distances[c as usize]);
        let next_phi = wb * phi_b + wc * phi_c;
        if next_phi >= cur_phi - eps_phi {
            continue;
        }
        let slope = td.descent_norm;
        if best.as_ref().is_none_or(|b_best| slope > b_best.slope()) {
            best = Some(StepChoice::Edge {
                tri: t,
                a: b,
                b: c,
                wa: wb,
                next_phi,
                slope,
            });
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn step_from_edge(
    mesh: &Mesh,
    distances: &[f32],
    is_source: &[bool],
    v_tri_off: &[u32],
    v_tris: &[u32],
    visited: &[bool],
    from_tri: u32,
    a: u32,
    b: u32,
    wa: f64,
    cur_phi: f64,
    allow_interior: bool,
) -> Option<StepChoice> {
    let eps_phi = 1e-12 * cur_phi.abs().max(1.0);
    if allow_interior
        && let Some(tri_next) = opposite_triangle(a, b, from_tri, v_tri_off, v_tris)
        && let Some(td) = tri_descent(mesh, distances, tri_next)
        && td.descent_norm > 0.0
        && let (Some(ia), Some(ib)) = (
            (0..3).find(|&k| td.v[k] == a),
            (0..3).find(|&k| td.v[k] == b),
        )
    {
        let ic = 3 - ia - ib;
        let c = td.v[ic];
        let wb = 1.0 - wa;
        let dwa = dot(td.descent, td.grad_vecs[ia]);
        let dwb = dot(td.descent, td.grad_vecs[ib]);
        let dwc = dot(td.descent, td.grad_vecs[ic]);
        let dw_scale = dwa.abs() + dwb.abs() + dwc.abs();
        if dwc > 1e-9 * dw_scale {
            let phi_c = f64::from(distances[c as usize]);
            if is_source[c as usize] && phi_c < cur_phi {
                return Some(StepChoice::Vertex {
                    next: c,
                    next_phi: phi_c,
                    slope: td.descent_norm,
                });
            }
            let t_a = if dwa < -1e-15 * dw_scale {
                -wa / dwa
            } else {
                f64::INFINITY
            };
            let t_b = if dwb < -1e-15 * dw_scale {
                -wb / dwb
            } else {
                f64::INFINITY
            };
            let t_exit = t_a.min(t_b);
            if t_exit.is_finite() && t_exit > 0.0 {
                let exit_wa = (wa + t_exit * dwa).clamp(0.0, 1.0);
                let exit_wb = (wb + t_exit * dwb).clamp(0.0, 1.0);
                let exit_wc = (t_exit * dwc).clamp(0.0, 1.0);
                let sum = exit_wa + exit_wb + exit_wc;
                if sum > 0.0 {
                    let na = exit_wa / sum;
                    let nb = exit_wb / sum;
                    let nc = exit_wc / sum;
                    let next_phi = na * f64::from(distances[a as usize])
                        + nb * f64::from(distances[b as usize])
                        + nc * f64::from(distances[c as usize]);
                    if next_phi < cur_phi - eps_phi {
                        if t_a <= t_b {
                            // Exited opposite vertex `a` onto edge `(b, c)`.
                            if nc <= EDGE_SNAP_EPS && !visited[b as usize] {
                                return Some(StepChoice::Vertex {
                                    next: b,
                                    next_phi: f64::from(distances[b as usize]),
                                    slope: td.descent_norm,
                                });
                            }
                            if nb <= EDGE_SNAP_EPS && !visited[c as usize] {
                                return Some(StepChoice::Vertex {
                                    next: c,
                                    next_phi: f64::from(distances[c as usize]),
                                    slope: td.descent_norm,
                                });
                            }
                            return Some(StepChoice::Edge {
                                tri: tri_next,
                                a: b,
                                b: c,
                                wa: nb / (nb + nc),
                                next_phi,
                                slope: td.descent_norm,
                            });
                        } else {
                            // Exited opposite vertex `b` onto edge `(a, c)`.
                            if nc <= EDGE_SNAP_EPS && !visited[a as usize] {
                                return Some(StepChoice::Vertex {
                                    next: a,
                                    next_phi: f64::from(distances[a as usize]),
                                    slope: td.descent_norm,
                                });
                            }
                            if na <= EDGE_SNAP_EPS && !visited[c as usize] {
                                return Some(StepChoice::Vertex {
                                    next: c,
                                    next_phi: f64::from(distances[c as usize]),
                                    slope: td.descent_norm,
                                });
                            }
                            return Some(StepChoice::Edge {
                                tri: tri_next,
                                a,
                                b: c,
                                wa: na / (na + nc),
                                next_phi,
                                slope: td.descent_norm,
                            });
                        }
                    }
                }
            }
        }
    }

    // Boundary, ridge/valley, degenerate neighbor, or edge-crossing cap:
    // collapse along edge `(a, b)` toward the better endpoint.
    let phi_a = f64::from(distances[a as usize]);
    let phi_b = f64::from(distances[b as usize]);
    let cand_a = (!visited[a as usize] && phi_a.is_finite()).then_some((a, phi_a));
    let cand_b = (!visited[b as usize] && phi_b.is_finite()).then_some((b, phi_b));
    match (cand_a, cand_b) {
        (Some((va, pa)), Some((vb, pb))) => {
            if pa <= pb {
                Some(StepChoice::Vertex {
                    next: va,
                    next_phi: pa,
                    slope: 0.0,
                })
            } else {
                Some(StepChoice::Vertex {
                    next: vb,
                    next_phi: pb,
                    slope: 0.0,
                })
            }
        }
        (Some((va, pa)), None) => Some(StepChoice::Vertex {
            next: va,
            next_phi: pa,
            slope: 0.0,
        }),
        (None, Some((vb, pb))) => Some(StepChoice::Vertex {
            next: vb,
            next_phi: pb,
            slope: 0.0,
        }),
        (None, None) => None,
    }
}

#[derive(Clone, Copy, PartialEq)]
struct EscapeNode {
    cost: f64,
    vertex: u32,
}

impl Eq for EscapeNode {}

impl Ord for EscapeNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.vertex.cmp(&self.vertex))
    }
}

impl PartialOrd for EscapeNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn push_escape(heap: &mut BinaryHeap<EscapeNode>, node: EscapeNode) -> Result<()> {
    if heap.len() == heap.capacity() {
        allocation_allowed()?;
        heap.try_reserve(1).map_err(|_| QueryError::OutOfMemory)?;
    }
    heap.push(node);
    Ok(())
}

/// Finds a short edge-graph path from `start` to either any source vertex or an
/// unvisited vertex with strictly lower `distances` value. Returns the vertex
/// sequence AFTER `start`, ending at the discovered goal vertex.
#[allow(clippy::too_many_arguments)]
fn escape_via_dijkstra(
    mesh: &Mesh,
    distances: &[f32],
    is_source: &[bool],
    v_tri_off: &[u32],
    v_tris: &[u32],
    visited: &[bool],
    start: u32,
    cur_phi: f64,
) -> Result<Vec<u32>> {
    let slots = mesh.vertex_slots();
    let pos = mesh.positions();
    let eps_phi = 1e-9 * cur_phi.abs().max(1.0);

    // Phase 1 seeks a strict downhill vertex or a source over finite-distance vertices;
    // Phase 2 searches any reachable source across all alive triangles if Phase 1 fails.
    for require_finite in [true, false] {
        let mut dist = try_filled(slots, f64::INFINITY)?;
        let mut parent = try_filled(slots, u32::MAX)?;
        let mut heap = BinaryHeap::new();

        dist[start as usize] = 0.0;
        push_escape(
            &mut heap,
            EscapeNode {
                cost: 0.0,
                vertex: start,
            },
        )?;

        let mut goal: Option<u32> = None;
        while let Some(EscapeNode { cost, vertex: u }) = heap.pop() {
            if cost > dist[u as usize] {
                continue;
            }
            if u != start {
                let phi_u = f64::from(distances[u as usize]);
                let ok_downhill = require_finite
                    && !visited[u as usize]
                    && phi_u.is_finite()
                    && phi_u < cur_phi - eps_phi;
                if is_source[u as usize] || ok_downhill {
                    goal = Some(u);
                    break;
                }
            }
            let p_u = widen(pos[u as usize]);
            let t_start = v_tri_off[u as usize] as usize;
            let t_end = v_tri_off[u as usize + 1] as usize;
            for &t in &v_tris[t_start..t_end] {
                let Some(tri) = mesh.triangle(t) else {
                    continue;
                };
                for &nbr in &tri {
                    if nbr == u {
                        continue;
                    }
                    if require_finite && !distances[nbr as usize].is_finite() {
                        continue;
                    }
                    let w = norm(sub(widen(pos[nbr as usize]), p_u));
                    let next_cost = cost + w;
                    if next_cost < dist[nbr as usize] {
                        dist[nbr as usize] = next_cost;
                        parent[nbr as usize] = u;
                        push_escape(
                            &mut heap,
                            EscapeNode {
                                cost: next_cost,
                                vertex: nbr,
                            },
                        )?;
                    }
                }
            }
        }

        if let Some(g) = goal {
            let mut chain: Vec<u32> = Vec::new();
            let mut cur = g;
            while cur != start {
                try_push(&mut chain, cur)?;
                cur = parent[cur as usize];
            }
            chain.reverse();
            return Ok(chain);
        }
    }

    Err(QueryError::InvalidInput(
        "no source vertex is reachable from target_vertex",
    ))
}

#[allow(clippy::too_many_arguments)]
fn trace_path(
    mesh: &Mesh,
    distances: &[f32],
    is_source: &[bool],
    v_tri_off: &[u32],
    v_tris: &[u32],
    target_vertex: u32,
    start_pt: SurfacePathPoint,
    out: &mut Vec<SurfacePathPoint>,
) -> Result<()> {
    let slots = mesh.vertex_slots();
    let mut visited = try_filled(slots, false)?;
    visited[target_vertex as usize] = true;

    let orig_len = out.len();
    let res: Result<()> = (|| {
        let mut ap = Appender::new(out);
        ap.push(start_pt)?;

        let mut state = TraceState::AtVertex(target_vertex);
        let mut cur_phi = f64::from(distances[target_vertex as usize]);
        let mut consecutive_edges = 0usize;
        let max_edge_steps = mesh.triangle_count().saturating_mul(2).max(256);
        let max_total_steps = slots
            .saturating_mul(8)
            .saturating_add(mesh.triangle_count().saturating_mul(2))
            .max(64);

        for _ in 0..max_total_steps {
            match state {
                TraceState::AtVertex(v) => {
                    if is_source[v as usize] {
                        break;
                    }
                    consecutive_edges = 0;
                    if let Some(choice) =
                        best_vertex_step(mesh, distances, v_tri_off, v_tris, &visited, v, cur_phi)
                    {
                        match choice {
                            StepChoice::Vertex { next, next_phi, .. } => {
                                visited[next as usize] = true;
                                let pt = make_vertex_point(mesh, next, v_tri_off, v_tris)?;
                                ap.push(pt)?;
                                state = TraceState::AtVertex(next);
                                cur_phi = next_phi;
                            }
                            StepChoice::Edge {
                                tri,
                                a,
                                b,
                                wa,
                                next_phi,
                                ..
                            } => {
                                if let Some(pt) = make_edge_point(mesh, tri, a, b, wa) {
                                    ap.push(pt)?;
                                }
                                consecutive_edges = 1;
                                state = TraceState::OnEdge {
                                    from_tri: tri,
                                    a,
                                    b,
                                    wa,
                                };
                                cur_phi = next_phi;
                            }
                        }
                    } else {
                        let chain = escape_via_dijkstra(
                            mesh, distances, is_source, v_tri_off, v_tris, &visited, v, cur_phi,
                        )?;
                        let mut last_v = v;
                        for w in chain {
                            visited[w as usize] = true;
                            let pt = make_vertex_point(mesh, w, v_tri_off, v_tris)?;
                            ap.push(pt)?;
                            last_v = w;
                        }
                        state = TraceState::AtVertex(last_v);
                        cur_phi = f64::from(distances[last_v as usize]);
                        if !cur_phi.is_finite() {
                            cur_phi = 0.0;
                        }
                    }
                }
                TraceState::OnEdge { from_tri, a, b, wa } => {
                    let allow_interior = consecutive_edges < max_edge_steps;
                    if let Some(choice) = step_from_edge(
                        mesh,
                        distances,
                        is_source,
                        v_tri_off,
                        v_tris,
                        &visited,
                        from_tri,
                        a,
                        b,
                        wa,
                        cur_phi,
                        allow_interior,
                    ) {
                        match choice {
                            StepChoice::Vertex { next, next_phi, .. } => {
                                visited[next as usize] = true;
                                let pt = make_vertex_point(mesh, next, v_tri_off, v_tris)?;
                                ap.push(pt)?;
                                consecutive_edges = 0;
                                state = TraceState::AtVertex(next);
                                cur_phi = next_phi;
                            }
                            StepChoice::Edge {
                                tri,
                                a,
                                b,
                                wa,
                                next_phi,
                                ..
                            } => {
                                if let Some(pt) = make_edge_point(mesh, tri, a, b, wa) {
                                    ap.push(pt)?;
                                }
                                consecutive_edges += 1;
                                state = TraceState::OnEdge {
                                    from_tri: tri,
                                    a,
                                    b,
                                    wa,
                                };
                                cur_phi = next_phi;
                            }
                        }
                    } else {
                        // Both endpoints already visited (e.g., flat edge saddle): jump out
                        // from the closer endpoint.
                        let anchor = if wa >= 0.5 { a } else { b };
                        let anchor_pt = make_vertex_point(mesh, anchor, v_tri_off, v_tris)?;
                        ap.push(anchor_pt)?;
                        if is_source[anchor as usize] {
                            state = TraceState::AtVertex(anchor);
                            break;
                        }
                        let chain = escape_via_dijkstra(
                            mesh, distances, is_source, v_tri_off, v_tris, &visited, anchor,
                            cur_phi,
                        )?;
                        let mut last_v = anchor;
                        for w in chain {
                            visited[w as usize] = true;
                            let pt = make_vertex_point(mesh, w, v_tri_off, v_tris)?;
                            ap.push(pt)?;
                            last_v = w;
                        }
                        consecutive_edges = 0;
                        state = TraceState::AtVertex(last_v);
                        cur_phi = f64::from(distances[last_v as usize]);
                        if !cur_phi.is_finite() {
                            cur_phi = 0.0;
                        }
                    }
                }
            }
        }

        if let TraceState::AtVertex(v) = state
            && !is_source[v as usize]
        {
            return Err(QueryError::InvalidInput(
                "failed to reach a source vertex within step bound",
            ));
        }
        if matches!(state, TraceState::OnEdge { .. }) {
            return Err(QueryError::InvalidInput(
                "failed to reach a source vertex within step bound",
            ));
        }

        ap.appended().reverse();
        Ok(())
    })();

    match res {
        Ok(()) => {
            out.drain(..orig_len);
            Ok(())
        }
        Err(e) => {
            out.truncate(orig_len);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fail_allocations_after;

    fn flat_grid(nx: u32, ny: u32) -> Mesh {
        let mut pos = Vec::new();
        for y in 0..=ny {
            for x in 0..=nx {
                pos.extend_from_slice(&[x as f32, y as f32, 0.0]);
            }
        }
        let mut idx = Vec::new();
        let stride = nx + 1;
        for y in 0..ny {
            for x in 0..nx {
                let a = y * stride + x;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                idx.extend_from_slice(&[a, b, d, a, d, c]);
            }
        }
        Mesh::from_buffers(&pos, &idx).unwrap()
    }

    fn polyline_length(pts: &[SurfacePathPoint]) -> f64 {
        pts.windows(2)
            .map(|w| norm(sub(widen(w[1].position), widen(w[0].position))))
            .sum()
    }

    #[test]
    fn non_diagonal_planar_grid_crosses_triangle_interiors() {
        let m = flat_grid(4, 4);
        let source = 0u32; // (0, 0)
        let target = 3 * 5 + 4; // (4, 3), exact distance = 5.0
        let dist: Vec<f32> = m
            .positions()
            .iter()
            .map(|p| (f64::from(p[0]).hypot(f64::from(p[1]))) as f32)
            .collect();

        let mut path = Vec::new();
        shortest_surface_path(&m, &dist, &[source], target, &mut path).unwrap();
        assert!(path.len() >= 3, "expected intermediate edge crossings");
        assert!((norm(sub(widen(path[0].position), [0.0, 0.0, 0.0]))) < 1e-5);
        assert!((norm(sub(widen(path.last().unwrap().position), [4.0, 3.0, 0.0]))) < 1e-5);

        let has_interior_edge = path.iter().any(|pt| {
            pt.barycentric
                .iter()
                .filter(|&&w| w > 1e-3 && w < 0.999)
                .count()
                == 2
        });
        assert!(
            has_interior_edge,
            "expected straight cut across triangle edges: {path:?}"
        );

        let len = polyline_length(&path);
        assert!(
            (len - 5.0).abs() < 0.08,
            "expected near-Euclidean length 5.0, got {len}"
        );
    }

    #[test]
    fn oom_fault_injection_leaves_output_unchanged() {
        let m = flat_grid(3, 3);
        let dist: Vec<f32> = m
            .positions()
            .iter()
            .map(|p| (f64::from(p[0]).hypot(f64::from(p[1]))) as f32)
            .collect();
        let sentinel = vec![SurfacePathPoint {
            position: [-9.0, -9.0, -9.0],
            triangle: 0,
            barycentric: [1.0, 0.0, 0.0],
        }];

        for fail_at in 0..40 {
            let mut out = sentinel.clone();
            fail_allocations_after(Some(fail_at));
            let res = shortest_surface_path(&m, &dist, &[0], 3 * 4 + 3, &mut out);
            fail_allocations_after(None);
            match res {
                Ok(()) => break,
                Err(QueryError::OutOfMemory) => assert_eq!(out, sentinel),
                Err(other) => panic!("unexpected error at alloc {fail_at}: {other:?}"),
            }
        }
    }
}
