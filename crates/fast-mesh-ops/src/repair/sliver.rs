//! Sliver triangle cleanup: needle collapses and cap edge flips.

use fast_mesh_store::{Transaction, TriangleId, UV_CHANNEL, VertexId};

use super::manifold::{is_proper, runs_forward_tri, sorted_triple, unit_normal_raw};
use super::norm64;
use crate::analyze::AREA_EPSILON;
use crate::topology::{cross, dot, sub};
use crate::{OpsError, Result, allocation_allowed, filled, push, reserve};

const NORMAL_DOT_MIN: f64 = 0.2;
const MIN_ANGLE_IMPROVE_DEG: f64 = 1e-4;

/// Options for [`cleanup_slivers()`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliverOptions {
    /// Maximum allowed circumradius-to-2-inradius aspect ratio (equilateral =
    /// `1.0`). Triangles above this are targeted as slivers. Default `20.0`.
    pub max_aspect_ratio: f32,
    /// Minimum interior angle in degrees before a triangle is treated as a
    /// sliver. Default `2.0`.
    pub min_angle_deg: f32,
    /// Maximum interior angle in degrees before a triangle is treated as a
    /// cap sliver. Default `176.0`.
    pub max_angle_deg: f32,
    /// Maximum cleanup passes over the mesh. Default `3`.
    pub max_iterations: u32,
}

impl Default for SliverOptions {
    fn default() -> Self {
        Self {
            max_aspect_ratio: 20.0,
            min_angle_deg: 2.0,
            max_angle_deg: 176.0,
            max_iterations: 3,
        }
    }
}

impl SliverOptions {
    /// A builder for [`SliverOptions`].
    pub fn builder() -> SliverOptionsBuilder {
        SliverOptionsBuilder::default()
    }
}

/// Builder for [`SliverOptions`].
#[derive(Debug, Clone, Default)]
pub struct SliverOptionsBuilder {
    options: SliverOptions,
}

impl SliverOptionsBuilder {
    /// Maximum allowed aspect ratio.
    pub fn max_aspect_ratio(mut self, max_aspect_ratio: f32) -> Self {
        self.options.max_aspect_ratio = max_aspect_ratio;
        self
    }

    /// Minimum interior angle threshold in degrees.
    pub fn min_angle_deg(mut self, min_angle_deg: f32) -> Self {
        self.options.min_angle_deg = min_angle_deg;
        self
    }

    /// Maximum interior angle threshold in degrees.
    pub fn max_angle_deg(mut self, max_angle_deg: f32) -> Self {
        self.options.max_angle_deg = max_angle_deg;
        self
    }

    /// Maximum cleanup iterations.
    pub fn max_iterations(mut self, max_iterations: u32) -> Self {
        self.options.max_iterations = max_iterations;
        self
    }

    /// The options as set.
    pub fn build(self) -> SliverOptions {
        self.options
    }
}

/// What [`cleanup_slivers()`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SliverReport {
    /// Needle triangles eliminated by shortest-edge collapse.
    pub collapsed_needles: usize,
    /// Cap triangles eliminated by flipping the opposite interior manifold edge.
    pub flipped_caps: usize,
    /// Zero-area / repeated-vertex degenerate triangles removed directly.
    pub removed_degenerate: usize,
    /// Vertices removed as unused after collapses and degenerate cleanup.
    pub removed_vertices: usize,
}

fn check_options(o: &SliverOptions) -> Result<()> {
    if !o.max_aspect_ratio.is_finite() || o.max_aspect_ratio < 1.0 {
        return Err(OpsError::InvalidInput(
            "max_aspect_ratio must be finite and at least 1.0",
        ));
    }
    if !o.min_angle_deg.is_finite() || !(0.0..=60.0).contains(&o.min_angle_deg) {
        return Err(OpsError::InvalidInput(
            "min_angle_deg must be finite and in [0, 60]",
        ));
    }
    if !o.max_angle_deg.is_finite() || !(60.0..=180.0).contains(&o.max_angle_deg) {
        return Err(OpsError::InvalidInput(
            "max_angle_deg must be finite and in [60, 180]",
        ));
    }
    if o.min_angle_deg > o.max_angle_deg {
        return Err(OpsError::InvalidInput(
            "min_angle_deg must not exceed max_angle_deg",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct TriMetrics {
    area: f64,
    aspect: f64,
    min_angle_deg: f64,
    max_angle_deg: f64,
    shortest_edge: (VertexId, VertexId),
    longest_edge: (VertexId, VertexId),
    l_min: f64,
    l_mid: f64,
    l_max: f64,
}

#[inline]
fn f64v(p: [f32; 3]) -> [f64; 3] {
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

fn tri_metrics(pos: &[[f32; 3]], tri: [VertexId; 3]) -> Option<TriMetrics> {
    if !is_proper(tri) {
        return None;
    }
    let p = [
        f64v(pos[tri[0] as usize]),
        f64v(pos[tri[1] as usize]),
        f64v(pos[tri[2] as usize]),
    ];
    if p.iter().flatten().any(|x| !x.is_finite()) {
        return None;
    }
    let e01 = sub(p[1], p[0]);
    let e12 = sub(p[2], p[1]);
    let e20 = sub(p[0], p[2]);
    let l01 = norm64(e01);
    let l12 = norm64(e12);
    let l20 = norm64(e20);
    let area2 = norm64(cross(e01, sub(p[2], p[0])));
    let area = 0.5 * area2;
    if area <= AREA_EPSILON || l01 <= 1e-12 || l12 <= 1e-12 || l20 <= 1e-12 {
        return None;
    }

    let ang0 = area2.atan2(dot(e01, sub(p[2], p[0]))).to_degrees();
    let ang1 = area2.atan2(dot(sub(p[0], p[1]), e12)).to_degrees();
    let ang2 = area2
        .atan2(dot(sub(p[0], p[2]), sub(p[1], p[2])))
        .to_degrees();
    let min_angle_deg = ang0.min(ang1).min(ang2);
    let max_angle_deg = ang0.max(ang1).max(ang2);
    let perim = l01 + l12 + l20;
    let aspect = (l01 * l12 * l20 * perim) / (16.0 * area * area);

    let mut edges = [
        (l01, tri[0], tri[1]),
        (l12, tri[1], tri[2]),
        (l20, tri[2], tri[0]),
    ];
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));

    Some(TriMetrics {
        area,
        aspect,
        min_angle_deg,
        max_angle_deg,
        shortest_edge: (edges[0].1, edges[0].2),
        longest_edge: (edges[2].1, edges[2].2),
        l_min: edges[0].0,
        l_mid: edges[1].0,
        l_max: edges[2].0,
    })
}

fn edge_triangles(
    vert_tris: &[Vec<u32>],
    tris: &[[VertexId; 3]],
    t_alive: &[bool],
    u: VertexId,
    v: VertexId,
) -> [Option<u32>; 3] {
    let mut out = [None; 3];
    let mut n = 0usize;
    for &t in &vert_tris[u as usize] {
        if !t_alive[t as usize] {
            continue;
        }
        let tri = tris[t as usize];
        if (tri[0] == v || tri[1] == v || tri[2] == v)
            && (tri[0] == u || tri[1] == u || tri[2] == u)
        {
            if n < 3 {
                out[n] = Some(t);
            }
            n += 1;
            if n >= 3 {
                return out;
            }
        }
    }
    out
}

fn edge_degree(
    vert_tris: &[Vec<u32>],
    tris: &[[VertexId; 3]],
    t_alive: &[bool],
    u: VertexId,
    v: VertexId,
) -> usize {
    let mut n = 0usize;
    for &t in &vert_tris[u as usize] {
        if !t_alive[t as usize] {
            continue;
        }
        let tri = tris[t as usize];
        if tri[0] == v || tri[1] == v || tri[2] == v {
            n += 1;
        }
    }
    n
}

fn is_boundary_or_nm_vertex(
    vert_tris: &[Vec<u32>],
    tris: &[[VertexId; 3]],
    t_alive: &[bool],
    v: VertexId,
) -> (bool, bool) {
    let mut boundary = false;
    let mut non_manifold = false;
    for &t in &vert_tris[v as usize] {
        if !t_alive[t as usize] {
            continue;
        }
        let tri = tris[t as usize];
        for w in tri {
            if w == v {
                continue;
            }
            let d = edge_degree(vert_tris, tris, t_alive, v, w);
            if d == 1 {
                boundary = true;
            } else if d > 2 {
                non_manifold = true;
            }
        }
    }
    (boundary, non_manifold)
}

fn vertex_neighbors(
    vert_tris: &[Vec<u32>],
    tris: &[[VertexId; 3]],
    t_alive: &[bool],
    v: VertexId,
    exclude: VertexId,
    out: &mut Vec<VertexId>,
) -> Result<()> {
    out.clear();
    for &t in &vert_tris[v as usize] {
        if !t_alive[t as usize] {
            continue;
        }
        for w in tris[t as usize] {
            if w != v && w != exclude {
                push(out, w)?;
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(())
}

fn opposite_vertex(tri: [VertexId; 3], u: VertexId, v: VertexId) -> Option<VertexId> {
    if tri[0] != u && tri[0] != v {
        Some(tri[0])
    } else if tri[1] != u && tri[1] != v {
        Some(tri[1])
    } else if tri[2] != u && tri[2] != v {
        Some(tri[2])
    } else {
        None
    }
}

fn remove_from_vert(vert_tris: &mut [Vec<u32>], v: VertexId, tri_id: u32) {
    let list = &mut vert_tris[v as usize];
    if let Some(pos) = list.iter().position(|&x| x == tri_id) {
        list.swap_remove(pos);
    }
}

fn try_flip_cap(
    pos: &[[f32; 3]],
    tris: &mut [[VertexId; 3]],
    t_alive: &[bool],
    vert_tris: &mut [Vec<u32>],
    t1: u32,
    u: VertexId,
    v: VertexId,
) -> Result<bool> {
    let inc = edge_triangles(vert_tris, tris, t_alive, u, v);
    let (Some(a), Some(b), None) = (inc[0], inc[1], inc[2]) else {
        return Ok(false);
    };
    let t2 = if a == t1 { b } else { a };

    let tri1 = tris[t1 as usize];
    let tri2 = tris[t2 as usize];
    let Some(apex1) = opposite_vertex(tri1, u, v) else {
        return Ok(false);
    };
    let Some(apex2) = opposite_vertex(tri2, u, v) else {
        return Ok(false);
    };
    if apex1 == apex2 {
        return Ok(false);
    }
    if edge_degree(vert_tris, tris, t_alive, apex1, apex2) > 0 {
        return Ok(false);
    }

    // Orient directed edge `(eu, ev)` in `tri1`'s cyclic order: `tri1 = [apex1, eu, ev]`.
    let (eu, ev) = if runs_forward_tri(tri1, u, v) {
        (u, v)
    } else {
        (v, u)
    };
    if !runs_forward_tri(tri2, ev, eu) {
        return Ok(false);
    }

    let Some(n1_old) = unit_normal_raw(pos, tri1) else {
        return Ok(false);
    };
    let Some(n2_old) = unit_normal_raw(pos, tri2) else {
        return Ok(false);
    };
    if dot(n1_old, n2_old) <= NORMAL_DOT_MIN {
        return Ok(false);
    }

    let new_tri1 = [apex1, eu, apex2];
    let new_tri2 = [apex1, apex2, ev];
    let Some(n1_new) = unit_normal_raw(pos, new_tri1) else {
        return Ok(false);
    };
    let Some(n2_new) = unit_normal_raw(pos, new_tri2) else {
        return Ok(false);
    };
    if dot(n1_old, n1_new) <= NORMAL_DOT_MIN || dot(n2_old, n2_new) <= NORMAL_DOT_MIN {
        return Ok(false);
    }

    let Some(m1_old) = tri_metrics(pos, tri1) else {
        return Ok(false);
    };
    let Some(m2_old) = tri_metrics(pos, tri2) else {
        return Ok(false);
    };
    let Some(m1_new) = tri_metrics(pos, new_tri1) else {
        return Ok(false);
    };
    let Some(m2_new) = tri_metrics(pos, new_tri2) else {
        return Ok(false);
    };
    if m1_new.min_angle_deg.min(m2_new.min_angle_deg)
        <= m1_old.min_angle_deg.min(m2_old.min_angle_deg) + MIN_ANGLE_IMPROVE_DEG
    {
        return Ok(false);
    }

    tris[t1 as usize] = new_tri1;
    tris[t2 as usize] = new_tri2;
    remove_from_vert(vert_tris, ev, t1);
    remove_from_vert(vert_tris, eu, t2);
    push(&mut vert_tris[apex2 as usize], t1)?;
    push(&mut vert_tris[apex1 as usize], t2)?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn try_collapse_edge(
    pos: &mut [[f32; 3]],
    uvs: &mut Option<Vec<[f32; 2]>>,
    pos_dirty: &mut [bool],
    uv_dirty: &mut [bool],
    tris: &mut [[VertexId; 3]],
    t_alive: &mut [bool],
    vert_tris: &mut [Vec<u32>],
    u: VertexId,
    v: VertexId,
    nb_u: &mut Vec<VertexId>,
    nb_v: &mut Vec<VertexId>,
    affected: &mut Vec<u32>,
) -> Result<bool> {
    let inc = edge_triangles(vert_tris, tris, t_alive, u, v);
    let adj_len = inc.iter().take_while(|x| x.is_some()).count();
    if adj_len == 0 || adj_len > 2 {
        return Ok(false);
    }

    let (bnd_u, nm_u) = is_boundary_or_nm_vertex(vert_tris, tris, t_alive, u);
    let (bnd_v, nm_v) = is_boundary_or_nm_vertex(vert_tris, tris, t_alive, v);
    if nm_u || nm_v {
        return Ok(false);
    }
    if bnd_u && bnd_v && adj_len != 1 {
        // Collapsing an interior strut connecting two boundaries pinches the loop.
        return Ok(false);
    }

    // Link condition: N(u) intersect N(v) == opposite vertex/vertices of edge (u, v).
    vertex_neighbors(vert_tris, tris, t_alive, u, v, nb_u)?;
    vertex_neighbors(vert_tris, tris, t_alive, v, u, nb_v)?;
    let mut opp = [None; 2];
    for (i, slot) in inc.iter().enumerate().take(adj_len) {
        let t = slot.expect("adj triangle");
        opp[i] = opposite_vertex(tris[t as usize], u, v);
        if opp[i].is_none() {
            return Ok(false);
        }
    }
    let mut common_count = 0usize;
    for &w in nb_u.iter() {
        if nb_v.binary_search(&w).is_ok() {
            if !opp[..adj_len].contains(&Some(w)) {
                return Ok(false);
            }
            common_count += 1;
        }
    }
    let expected_common = if adj_len == 2 && opp[0] == opp[1] {
        1
    } else {
        adj_len
    };
    if common_count != expected_common {
        return Ok(false);
    }

    // Pick survivor vertex & candidate positions.
    type CollapseCandidate = ([f32; 3], [f32; 2], bool);
    let (keep, drop, candidates): (VertexId, VertexId, Vec<CollapseCandidate>) = {
        let uv_u = uvs.as_ref().map_or([0.0; 2], |buf| buf[u as usize]);
        let uv_v = uvs.as_ref().map_or([0.0; 2], |buf| buf[v as usize]);
        let p_u = pos[u as usize];
        let p_v = pos[v as usize];
        let mid_p = [
            0.5 * (p_u[0] + p_v[0]),
            0.5 * (p_u[1] + p_v[1]),
            0.5 * (p_u[2] + p_v[2]),
        ];
        let mid_uv = [0.5 * (uv_u[0] + uv_v[0]), 0.5 * (uv_u[1] + uv_v[1])];
        if bnd_u && !bnd_v {
            (u, v, vec![(p_u, uv_u, false)])
        } else if bnd_v && !bnd_u {
            (v, u, vec![(p_v, uv_v, false)])
        } else if u < v {
            (
                u,
                v,
                vec![(mid_p, mid_uv, true), (p_u, uv_u, false), (p_v, uv_v, true)],
            )
        } else {
            (
                v,
                u,
                vec![(mid_p, mid_uv, true), (p_v, uv_v, false), (p_u, uv_u, true)],
            )
        }
    };

    affected.clear();
    for &t in vert_tris[keep as usize]
        .iter()
        .chain(&vert_tris[drop as usize])
    {
        if !t_alive[t as usize] {
            continue;
        }
        if inc[..adj_len].contains(&Some(t)) {
            continue;
        }
        push(affected, t)?;
    }
    affected.sort_unstable();
    affected.dedup();
    if affected.is_empty() {
        return Ok(false);
    }

    // Prevent folding a closed low-valence cap (like a tetrahedron) into duplicate triangles.
    let mut keys: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut keys, affected.len())?;
    for &t in affected.iter() {
        let mut tri = tris[t as usize];
        for c in &mut tri {
            if *c == drop {
                *c = keep;
            }
        }
        if !is_proper(tri) {
            return Ok(false);
        }
        push(&mut keys, sorted_triple(tri))?;
    }
    keys.sort_unstable();
    if keys.windows(2).any(|w| w[0] == w[1]) {
        return Ok(false);
    }

    let orig_keep_pos = pos[keep as usize];
    let mut chosen: Option<([f32; 3], [f32; 2], bool)> = None;
    for &(cand_p, cand_uv, changed_keep) in &candidates {
        if !cand_p.iter().all(|x| x.is_finite()) {
            continue;
        }
        let mut ok = true;
        for &t in affected.iter() {
            let tri_old = tris[t as usize];
            let Some(n_old) = unit_normal_raw(pos, tri_old) else {
                ok = false;
                break;
            };
            let mut tri_new = tri_old;
            for c in &mut tri_new {
                if *c == drop {
                    *c = keep;
                }
            }
            pos[keep as usize] = cand_p;
            let n_new = unit_normal_raw(pos, tri_new);
            let m_new = tri_metrics(pos, tri_new);
            pos[keep as usize] = orig_keep_pos;
            match (n_new, m_new) {
                (Some(n_after), Some(_)) if dot(n_old, n_after) > NORMAL_DOT_MIN => {}
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            chosen = Some((cand_p, cand_uv, changed_keep));
            break;
        }
    }
    let Some((new_p, new_uv, changed_keep)) = chosen else {
        return Ok(false);
    };

    if changed_keep {
        pos[keep as usize] = new_p;
        pos_dirty[keep as usize] = true;
        if let Some(uv_buf) = uvs.as_mut() {
            uv_buf[keep as usize] = new_uv;
            uv_dirty[keep as usize] = true;
        }
    }
    for slot in inc.iter().take(adj_len) {
        let t = slot.expect("adj triangle");
        t_alive[t as usize] = false;
        let tri = tris[t as usize];
        for v_corner in tri {
            remove_from_vert(vert_tris, v_corner, t);
        }
    }
    for &t in affected.iter() {
        let tri = &mut tris[t as usize];
        let mut replaced = false;
        for c in tri.iter_mut() {
            if *c == drop {
                *c = keep;
                replaced = true;
            }
        }
        if replaced {
            remove_from_vert(vert_tris, drop, t);
            push(&mut vert_tris[keep as usize], t)?;
        }
    }
    Ok(true)
}

/// Cleans up degenerate, needle, and cap sliver triangles on the mesh inside `tx`.
///
/// - Zero-area or repeated-vertex triangles are removed directly.
/// - Needle triangles (shortest edge small relative to longest or violating
///   quality bounds) are resolved by shortest-edge collapse when the
///   topological link condition holds and adjacent normals do not invert
///   (`dot(n_before, n_after) > 0.2`).
/// - Cap triangles (large obtuse angle opposite the longest interior manifold
///   edge) are resolved by flipping that edge when the two-triangle quad is
///   convex, improves the minimum interior angle, and preserves normals.
///
/// Vertices orphaned by cleanup are removed. UVs on collapsed vertices follow
/// midpoint or survivor interpolation cleanly; edge flips preserve UVs
/// automatically.
///
/// # Errors
/// - [`OpsError::InvalidInput`] for invalid thresholds in [`SliverOptions`].
/// - [`OpsError::Store`] when memory runs out or a store write fails.
pub fn cleanup_slivers(tx: &mut Transaction, options: &SliverOptions) -> Result<SliverReport> {
    allocation_allowed()?;
    check_options(options)?;
    let mesh = tx.mesh();
    let v_slots = mesh.vertex_slots();
    let t_slots = mesh.triangle_slots();
    if mesh.triangle_count() == 0 {
        return Ok(SliverReport::default());
    }

    let mut pos: Vec<[f32; 3]> = Vec::new();
    reserve(&mut pos, v_slots)?;
    let raw_pos = mesh.positions();
    for &p in raw_pos.iter().take(v_slots) {
        push(&mut pos, p)?;
    }

    let mut uvs: Option<Vec<[f32; 2]>> = if let Some(src) = mesh.uvs() {
        let mut buf: Vec<[f32; 2]> = Vec::new();
        reserve(&mut buf, v_slots)?;
        for v in 0..v_slots {
            push(&mut buf, [src[2 * v], src[2 * v + 1]])?;
        }
        Some(buf)
    } else {
        None
    };

    let mut orig_tris = filled(t_slots, [0; 3])?;
    let mut tris = filled(t_slots, [0; 3])?;
    let mut t_alive = filled(t_slots, false)?;
    let mut orig_used = filled(v_slots, false)?;
    let mut vert_tris = filled(v_slots, Vec::<u32>::new())?;

    for t in 0..t_slots {
        if let Some(tri) = mesh.triangle(t as TriangleId) {
            orig_tris[t] = tri;
            tris[t] = tri;
            t_alive[t] = true;
            for v in tri {
                orig_used[v as usize] = true;
            }
            push(&mut vert_tris[tri[0] as usize], t as u32)?;
            if tri[1] != tri[0] {
                push(&mut vert_tris[tri[1] as usize], t as u32)?;
            }
            if tri[2] != tri[0] && tri[2] != tri[1] {
                push(&mut vert_tris[tri[2] as usize], t as u32)?;
            }
        }
    }

    let mut pos_dirty = filled(v_slots, false)?;
    let mut uv_dirty = filled(v_slots, false)?;
    let mut nb_u: Vec<VertexId> = Vec::new();
    let mut nb_v: Vec<VertexId> = Vec::new();
    let mut affected: Vec<u32> = Vec::new();

    let mut collapsed_needles = 0usize;
    let mut flipped_caps = 0usize;
    let mut removed_degenerate = 0usize;

    let max_aspect = f64::from(options.max_aspect_ratio);
    let min_angle = f64::from(options.min_angle_deg);
    let max_angle = f64::from(options.max_angle_deg);

    for _iter in 0..options.max_iterations {
        let mut round_changed = false;
        for t in 0..t_slots {
            if !t_alive[t] {
                continue;
            }
            let tri = tris[t];
            let Some(m) = tri_metrics(&pos, tri) else {
                t_alive[t] = false;
                removed_degenerate += 1;
                round_changed = true;
                for v in tri {
                    remove_from_vert(&mut vert_tris, v, t as u32);
                }
                continue;
            };
            let is_sliver =
                m.aspect > max_aspect || m.min_angle_deg < min_angle || m.max_angle_deg > max_angle;
            if !is_sliver {
                continue;
            }

            let looks_like_cap =
                m.max_angle_deg > max_angle || (m.max_angle_deg > 120.0 && m.l_mid < 4.0 * m.l_min);
            if looks_like_cap {
                let (lu, lv) = m.longest_edge;
                if try_flip_cap(&pos, &mut tris, &t_alive, &mut vert_tris, t as u32, lu, lv)? {
                    flipped_caps += 1;
                    round_changed = true;
                    continue;
                }
                if m.l_min < 0.25 * m.l_max
                    && (m.aspect > max_aspect * 10.0 || m.area <= 1e-10 * m.l_max * m.l_max)
                    && try_collapse_edge(
                        &mut pos,
                        &mut uvs,
                        &mut pos_dirty,
                        &mut uv_dirty,
                        &mut tris,
                        &mut t_alive,
                        &mut vert_tris,
                        m.shortest_edge.0,
                        m.shortest_edge.1,
                        &mut nb_u,
                        &mut nb_v,
                        &mut affected,
                    )?
                {
                    collapsed_needles += 1;
                    round_changed = true;
                }
            } else {
                let (su, sv) = m.shortest_edge;
                if try_collapse_edge(
                    &mut pos,
                    &mut uvs,
                    &mut pos_dirty,
                    &mut uv_dirty,
                    &mut tris,
                    &mut t_alive,
                    &mut vert_tris,
                    su,
                    sv,
                    &mut nb_u,
                    &mut nb_v,
                    &mut affected,
                )? {
                    collapsed_needles += 1;
                    round_changed = true;
                }
            }
        }
        if !round_changed {
            break;
        }
    }

    let mut moved_vids: Vec<VertexId> = Vec::new();
    let mut moved_pos: Vec<[f32; 3]> = Vec::new();
    let mut uv_vids: Vec<VertexId> = Vec::new();
    let mut uv_vals: Vec<f32> = Vec::new();
    let mut rewired_ids: Vec<TriangleId> = Vec::new();
    let mut rewired_tris: Vec<[VertexId; 3]> = Vec::new();
    let mut removed_tris: Vec<TriangleId> = Vec::new();
    let mut final_used = filled(v_slots, false)?;

    for t in 0..t_slots {
        if !mesh.is_triangle_alive(t as TriangleId) {
            continue;
        }
        if !t_alive[t] {
            push(&mut removed_tris, t as TriangleId)?;
        } else {
            let tri = tris[t];
            for v in tri {
                final_used[v as usize] = true;
            }
            if tri != orig_tris[t] {
                push(&mut rewired_ids, t as TriangleId)?;
                push(&mut rewired_tris, tri)?;
            }
        }
    }

    let mut removed_verts: Vec<VertexId> = Vec::new();
    for v in 0..v_slots {
        let vid = v as VertexId;
        if orig_used[v] && !final_used[v] && mesh.is_vertex_alive(vid) {
            push(&mut removed_verts, vid)?;
        } else if final_used[v] {
            if pos_dirty[v] {
                push(&mut moved_vids, vid)?;
                push(&mut moved_pos, pos[v])?;
            }
            if uv_dirty[v]
                && let Some(uv_buf) = uvs.as_ref()
            {
                push(&mut uv_vids, vid)?;
                push(&mut uv_vals, uv_buf[v][0])?;
                push(&mut uv_vals, uv_buf[v][1])?;
            }
        }
    }

    if removed_tris.is_empty() && rewired_ids.is_empty() && moved_vids.is_empty() {
        return Ok(SliverReport::default());
    }

    if !moved_vids.is_empty() {
        tx.set_positions(&moved_vids, &moved_pos)?;
    }
    if !uv_vids.is_empty() {
        tx.write_channel(UV_CHANNEL, &uv_vids, &uv_vals)?;
    }
    if !rewired_ids.is_empty() {
        tx.set_triangles(&rewired_ids, &rewired_tris)?;
    }
    if !removed_tris.is_empty() {
        tx.remove_triangles(&removed_tris)?;
    }
    if !removed_verts.is_empty() {
        tx.remove_vertices(&removed_verts)?;
    }

    Ok(SliverReport {
        collapsed_needles,
        flipped_caps,
        removed_degenerate,
        removed_vertices: removed_verts.len(),
    })
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::{ChannelDesc, UV_CHANNEL};

    use super::*;
    use crate::analyze::{AnalyzeOptions, analyze};
    use crate::test_util::{box_mesh, edit, mesh_of, sweep};

    #[test]
    fn clean_box_is_noop() {
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        let (r, empty) = edit(&mut mesh, |tx| {
            cleanup_slivers(tx, &SliverOptions::default())
        });
        assert!(empty);
        assert_eq!(r, SliverReport::default());
    }

    #[test]
    fn collapses_needle_triangle_and_updates_uv() {
        // Two healthy quads flanking a tiny needle strip `(2, 3)` inside a 2x2 grid.
        let mut mesh = mesh_of(
            &[
                0.0, 0.0, 0.0, // 0
                2.0, 0.0, 0.0, // 1
                0.995, 1.0, 0.0, // 2
                1.005, 1.0, 0.0, // 3
                0.0, 2.0, 0.0, // 4
                2.0, 2.0, 0.0, // 5
            ],
            &[
                [0, 1, 2],
                [1, 3, 2], // needle: short top edge (2, 3) length 0.01
                [0, 2, 4],
                [2, 5, 4],
                [2, 3, 5], // needle: short bottom edge (2, 3) length 0.01
                [1, 5, 3],
            ],
        );
        mesh.add_channel(ChannelDesc::uv()).unwrap();
        edit(&mut mesh, |tx| {
            tx.write_channel(
                UV_CHANNEL,
                &[0, 1, 2, 3, 4, 5],
                &[0.0, 0.0, 1.0, 0.0, 0.4, 0.5, 0.6, 0.5, 0.0, 1.0, 1.0, 1.0],
            )?;
            Ok(())
        });

        let (r, _) = edit(&mut mesh, |tx| {
            cleanup_slivers(tx, &SliverOptions::default())
        });
        assert_eq!(r.collapsed_needles, 1);
        assert_eq!(mesh.triangle_count(), 4);
        assert_eq!(mesh.vertex_count(), 5);
        assert_eq!(&mesh.uvs().unwrap()[4..6], &[0.5, 0.5]);
        let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
        assert_eq!(a.non_manifold_edges, 0);
        assert!(a.pinch_vertices.is_empty());
    }

    #[test]
    fn flips_interior_cap_edge() {
        // Convex quad with a flat cap triangle along interior diagonal (0, 2).
        let mut mesh = mesh_of(
            &[
                0.0, 0.0, 0.0, // 0
                1.0, -1.0, 0.0, // 1
                2.0, 0.0, 0.0, // 2
                1.0, 0.01, 0.0, // 3: almost collinear with (0, 2) -> obtuse cap
            ],
            &[[0, 1, 2], [0, 2, 3]],
        );
        let (r, _) = edit(&mut mesh, |tx| {
            cleanup_slivers(tx, &SliverOptions::default())
        });
        assert_eq!(r.flipped_caps, 1);
        assert_eq!(mesh.triangle_count(), 2);
        // Edge flipped from (0, 2) to (1, 3): both new triangles contain vertex 1 and 3.
        let t0 = mesh.triangle(0).unwrap();
        let t1 = mesh.triangle(1).unwrap();
        assert!(t0.contains(&1) && t0.contains(&3));
        assert!(t1.contains(&1) && t1.contains(&3));
    }

    #[test]
    fn oom_writes_nothing() {
        let mesh = mesh_of(
            &[0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.01, 0.0],
            &[[0, 1, 2], [0, 2, 3]],
        );
        let n = sweep(mesh, |tx| {
            cleanup_slivers(tx, &SliverOptions::default()).map(|_| ())
        });
        assert!(n > 0);
    }

    #[test]
    fn rejects_invalid_options() {
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        let mut tx = mesh.begin_edit();
        assert!(
            cleanup_slivers(
                &mut tx,
                &SliverOptions {
                    max_aspect_ratio: 0.5,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            cleanup_slivers(
                &mut tx,
                &SliverOptions {
                    min_angle_deg: -1.0,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            cleanup_slivers(
                &mut tx,
                &SliverOptions {
                    max_angle_deg: 200.0,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
}
