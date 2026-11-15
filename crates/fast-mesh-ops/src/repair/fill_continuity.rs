//! Continuity-selectable hole filling (`Flat`, `TangentG1`, `CurvatureG2`) and
//! island-aware multi-loop patching.

use fast_mesh_store::{Mesh, Transaction, VertexId};

pub use crate::fair::FillContinuity;
use crate::fair::fair_with_continuity;
use crate::fill::{FailReason, Plan, loops_on_seams, merge_sorted};
use crate::fill_dp::{
    best_fit_frame, flat_area, is_coplanar_outer_rim, triangulate_loop, triangulate_with_islands,
    unit_normal,
};
use crate::refine::{RefinedPatch, refine};
use crate::topology::{BoundaryLoop, EdgeMap, VertexFans, boundary_loops, cross, dot, point, sub};
use crate::{OpsError, Result, filled, push, reserve};

/// Default dynamic-programming cutoff passed to [`triangulate_loop`].
const DEFAULT_DP_LIMIT: usize = 200;

/// Options for [`fill_hole`] and [`fill_all_holes`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FillHolesOptions {
    /// Geometric continuity mode used to position interior patch vertices.
    pub continuity: FillContinuity,
    /// Skip (in `fill_all_holes`) or reject (in `fill_hole`) holes whose
    /// perimeter exceeds this threshold. `None` disables the limit.
    pub max_hole_perimeter: Option<f32>,
    /// Skip (in `fill_all_holes`) or reject (in `fill_hole`) holes with more
    /// vertices than this threshold. `None` disables the limit.
    pub max_hole_vertices: Option<usize>,
    /// Isotropically refine interior patch triangles to match surrounding
    /// boundary edge lengths before fairing. Default `true`.
    pub refine_patch: bool,
    /// Number of continuity fairing and Delaunay relaxation passes when
    /// `continuity` is not [`FillContinuity::Flat`]. Default `1`.
    pub smooth_iterations: usize,
}

impl Default for FillHolesOptions {
    fn default() -> Self {
        Self {
            continuity: FillContinuity::CurvatureG2,
            max_hole_perimeter: None,
            max_hole_vertices: None,
            refine_patch: true,
            smooth_iterations: 1,
        }
    }
}

impl FillHolesOptions {
    /// A builder for [`FillHolesOptions`].
    pub fn builder() -> FillHolesOptionsBuilder {
        FillHolesOptionsBuilder::default()
    }
}

/// Builder for [`FillHolesOptions`].
#[derive(Debug, Clone, Copy, Default)]
pub struct FillHolesOptionsBuilder {
    options: FillHolesOptions,
}

impl FillHolesOptionsBuilder {
    /// Sets the continuity mode (`Flat`, `TangentG1`, or `CurvatureG2`).
    pub fn continuity(mut self, continuity: FillContinuity) -> Self {
        self.options.continuity = continuity;
        self
    }

    /// Sets `max_hole_perimeter`.
    pub fn max_hole_perimeter(mut self, max: Option<f32>) -> Self {
        self.options.max_hole_perimeter = max;
        self
    }

    /// Sets `max_hole_vertices`.
    pub fn max_hole_vertices(mut self, max: Option<usize>) -> Self {
        self.options.max_hole_vertices = max;
        self
    }

    /// Sets `refine_patch`.
    pub fn refine_patch(mut self, refine_patch: bool) -> Self {
        self.options.refine_patch = refine_patch;
        self
    }

    /// Sets `smooth_iterations`.
    pub fn smooth_iterations(mut self, smooth_iterations: usize) -> Self {
        self.options.smooth_iterations = smooth_iterations;
        self
    }

    /// Returns the built [`FillHolesOptions`].
    pub fn build(self) -> FillHolesOptions {
        self.options
    }
}

/// Summary statistics for a filled hole patch produced by [`fill_hole`] or
/// [`fill_all_holes`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FillStats {
    /// Index of the filled outer boundary loop at the time of the call.
    pub loop_index: usize,
    /// Number of newly inserted interior vertices in the patch.
    pub filled_vertices: usize,
    /// Number of triangles added to seal the hole.
    pub filled_triangles: usize,
    /// Perimeter of the outer boundary loop before filling.
    pub perimeter: f64,
    /// Maximum dihedral angle in degrees across boundary seam edges and
    /// interior patch edges of the final faired patch.
    pub max_dihedral_angle_deg: f64,
    /// Continuity mode applied to the patch.
    pub continuity: FillContinuity,
    /// Number of inner island boundary loops merged into this hole patch.
    pub island_loops_merged: usize,
    /// Whether the continuity solver converged within tolerance (`true` when
    /// fairing was skipped, e.g. `Flat` or `refine_patch == false`).
    pub converged: bool,
}

/// Fills boundary loop `loop_index` (along with any nested inner island loops
/// contained within it) using the continuity mode and options in `options`.
///
/// The fill is planned and verified before `tx` is mutated, so failed or
/// out-of-memory calls leave the transaction untouched.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when `loop_index` is out of bounds, the loop
///   is open/unclean or exceeds configured bounds, or triangulation/fairing fails.
/// - [`OpsError::Store`] on store / memory exhaustion.
pub fn fill_hole(
    tx: &mut Transaction,
    loop_index: usize,
    options: &FillHolesOptions,
) -> Result<FillStats> {
    let mesh = tx.mesh();
    let loops = boundary_loops(mesh)?;
    let l = loops
        .get(loop_index)
        .ok_or(OpsError::InvalidInput("loop_index out of bounds"))?;
    if !l.closed || !l.clean || l.vertices.len() < 3 {
        return Err(OpsError::InvalidInput(
            "boundary loop must be closed and clean",
        ));
    }
    if !within_bounds(l, options) {
        return Err(OpsError::InvalidInput(
            "boundary loop exceeds configured bounds",
        ));
    }

    let edges = EdgeMap::build(mesh)?;
    let fans = VertexFans::build(mesh)?;
    let outer_rims = classify_outer_rims(mesh, &loops, &edges)?;
    let islands_by_hole = detect_islands(mesh, &loops, &outer_rims)?;
    let mut plan = Plan::default();
    let mut chords: Vec<(VertexId, VertexId)> = Vec::new();
    let stats = match plan_loop_fill(
        mesh,
        &fans,
        &edges,
        &loops,
        loop_index,
        &islands_by_hole[loop_index],
        options,
        &mut plan,
        &mut chords,
    )? {
        Ok(stats) => stats,
        Err(reason) => return Err(OpsError::InvalidInput(fail_message(reason))),
    };
    plan.write(tx)?;
    Ok(stats)
}

/// Fills all eligible holes in `tx.mesh()` using `options`, automatically
/// pairing nested inner island loops with their enclosing hole perimeter and
/// skipping open outer sheet boundaries, UV seams, and out-of-bounds loops.
///
/// # Errors
/// [`OpsError::Store`] when store / memory runs out.
pub fn fill_all_holes(tx: &mut Transaction, options: &FillHolesOptions) -> Result<Vec<FillStats>> {
    let mesh = tx.mesh();
    let loops = boundary_loops(mesh)?;
    if loops.is_empty() {
        return Ok(Vec::new());
    }

    let edges = EdgeMap::build(mesh)?;
    let fans = VertexFans::build(mesh)?;
    let outer_rims = classify_outer_rims(mesh, &loops, &edges)?;
    let islands_by_hole = detect_islands(mesh, &loops, &outer_rims)?;
    let seams = if mesh.uvs().is_some() {
        loops_on_seams(mesh, &loops)?
    } else {
        filled(loops.len(), false)?
    };

    let mut claimed_as_island = filled(loops.len(), false)?;
    for isl_list in &islands_by_hole {
        for &idx in isl_list {
            claimed_as_island[idx] = true;
        }
    }

    let mut plan = Plan::default();
    let mut chords: Vec<(VertexId, VertexId)> = Vec::new();
    let mut stats_out = Vec::new();
    reserve(&mut stats_out, loops.len())?;

    for (idx, l) in loops.iter().enumerate() {
        if claimed_as_island[idx] || outer_rims[idx] || seams[idx] {
            continue;
        }
        if !l.closed || !l.clean || l.vertices.len() < 3 || !within_bounds(l, options) {
            continue;
        }
        if let Ok(stats) = plan_loop_fill(
            mesh,
            &fans,
            &edges,
            &loops,
            idx,
            &islands_by_hole[idx],
            options,
            &mut plan,
            &mut chords,
        )? {
            push(&mut stats_out, stats)?;
        }
    }

    plan.write(tx)?;
    Ok(stats_out)
}

fn within_bounds(l: &BoundaryLoop, options: &FillHolesOptions) -> bool {
    options
        .max_hole_vertices
        .is_none_or(|m| l.vertices.len() <= m)
        && options
            .max_hole_perimeter
            .is_none_or(|m| l.perimeter <= m as f64)
}

fn fail_message(reason: FailReason) -> &'static str {
    match reason {
        FailReason::ProjectionFolds => "hole projection folds or self-intersects",
        FailReason::NoValidTriangulation => "no valid manifold hole triangulation",
        FailReason::Solver => "continuity fairing solver failed to converge",
    }
}

fn classify_outer_rims(mesh: &Mesh, loops: &[BoundaryLoop], edges: &EdgeMap) -> Result<Vec<bool>> {
    let mut rims = filled(loops.len(), false)?;
    for (i, l) in loops.iter().enumerate() {
        if l.closed && l.clean && l.vertices.len() >= 3 {
            rims[i] = is_coplanar_outer_rim(mesh, &l.vertices, edges)?;
        }
    }
    Ok(rims)
}

/// Pairs each inner island loop `j` directly inside the tightest enclosing hole
/// loop `i` (`!outer_rims[i]`, opposing Newell normals, coplanar proximity, and
/// strict 2D containment).
fn detect_islands(
    mesh: &Mesh,
    loops: &[BoundaryLoop],
    outer_rims: &[bool],
) -> Result<Vec<Vec<usize>>> {
    let n = loops.len();
    let mut out = Vec::new();
    reserve(&mut out, n)?;
    out.resize_with(n, Vec::new);
    if n < 2 {
        return Ok(out);
    }

    let mut newells = filled(n, [0.0f64; 3])?;
    for (i, l) in loops.iter().enumerate() {
        if l.closed && l.clean && l.vertices.len() >= 3 {
            let v = loop_newell(mesh, &l.vertices);
            let len = dot(v, v).sqrt();
            if len > 0.0 && len.is_finite() {
                newells[i] = [v[0] / len, v[1] / len, v[2] / len];
            }
        }
    }

    // Precompute best-fit 2D polygons for valid hole candidates.
    struct HoleProj {
        frame: crate::fill_dp::PlaneFrame,
        poly: Vec<[f64; 2]>,
        area: f64,
        diam: f64,
    }
    let mut holes: Vec<Option<HoleProj>> = Vec::new();
    reserve(&mut holes, n)?;
    for (i, l) in loops.iter().enumerate() {
        if !l.closed || !l.clean || l.vertices.len() < 3 || outer_rims[i] {
            holes.push(None);
            continue;
        }
        let frame = best_fit_frame(mesh, &l.vertices)?;
        let mut poly = Vec::new();
        reserve(&mut poly, l.vertices.len())?;
        let (mut min_u, mut max_u) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_w, mut max_w) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in &l.vertices {
            let p2 = frame.project(point(mesh, v));
            min_u = min_u.min(p2[0]);
            max_u = max_u.max(p2[0]);
            min_w = min_w.min(p2[1]);
            max_w = max_w.max(p2[1]);
            poly.push(p2);
        }
        let area = flat_area(&poly);
        let diam = ((max_u - min_u).powi(2) + (max_w - min_w).powi(2)).sqrt();
        if area.is_finite() && area > 0.0 && diam.is_finite() && diam > 0.0 {
            holes.push(Some(HoleProj {
                frame,
                poly,
                area,
                diam,
            }));
        } else {
            holes.push(None);
        }
    }

    let mut isl_proj = Vec::new();
    for (j, lj) in loops.iter().enumerate() {
        if !lj.closed || !lj.clean || lj.vertices.len() < 3 || newells[j] == [0.0; 3] {
            continue;
        }
        let mut best_host: Option<(f64, usize)> = None;
        for (i, hole_opt) in holes.iter().enumerate() {
            if i == j {
                continue;
            }
            let Some(h) = hole_opt else {
                continue;
            };
            if dot(newells[i], newells[j]) >= -0.5 {
                continue;
            }
            isl_proj.clear();
            reserve(&mut isl_proj, lj.vertices.len())?;
            let max_depth = 0.6 * h.diam;
            let mut inside = true;
            for &v in &lj.vertices {
                let p3 = point(mesh, v);
                let depth = dot(sub(p3, h.frame.centre), h.frame.normal).abs();
                if !depth.is_finite() || depth > max_depth {
                    inside = false;
                    break;
                }
                let p2 = h.frame.project(p3);
                if !point_in_polygon_2d(p2, &h.poly) {
                    inside = false;
                    break;
                }
                isl_proj.push(p2);
            }
            if !inside {
                continue;
            }
            let isl_area = flat_area(&isl_proj).abs();
            if !(isl_area.is_finite() && isl_area > 0.0 && isl_area < 0.999 * h.area) {
                continue;
            }
            let replace = match best_host {
                None => true,
                Some((best_area, best_i)) => {
                    h.area < best_area || (h.area == best_area && i < best_i)
                }
            };
            if replace {
                best_host = Some((h.area, i));
            }
        }
        if let Some((_, host_i)) = best_host {
            push(&mut out[host_i], j)?;
        }
    }
    for list in &mut out {
        list.sort_unstable();
    }
    Ok(out)
}

fn loop_newell(mesh: &Mesh, verts: &[VertexId]) -> [f64; 3] {
    let mut sum = [0.0; 3];
    for i in 0..verts.len() {
        let a = point(mesh, verts[i]);
        let b = point(mesh, verts[(i + 1) % verts.len()]);
        let c = cross(a, b);
        sum[0] += c[0];
        sum[1] += c[1];
        sum[2] += c[2];
    }
    sum
}

/// Ray-casting winding test in 2D; vertices on edges count as outside so an
/// overlapping loop is not misclassified as strictly nested.
fn point_in_polygon_2d(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = poly.len();
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        if ((a[1] > p[1]) != (b[1] > p[1]))
            && (p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0])
        {
            inside = !inside;
        }
    }
    inside
}

#[allow(clippy::too_many_arguments)]
fn plan_loop_fill(
    mesh: &Mesh,
    fans: &VertexFans,
    edges: &EdgeMap,
    loops: &[BoundaryLoop],
    loop_index: usize,
    island_indices: &[usize],
    options: &FillHolesOptions,
    plan: &mut Plan,
    chords: &mut Vec<(VertexId, VertexId)>,
) -> Result<std::result::Result<FillStats, FailReason>> {
    let outer = &loops[loop_index];
    let mut island_slices: Vec<&[VertexId]> = Vec::new();
    reserve(&mut island_slices, island_indices.len())?;
    for &idx in island_indices {
        island_slices.push(loops[idx].vertices.as_slice());
    }

    let base_tris = if island_slices.is_empty() {
        match triangulate_loop(mesh, &outer.vertices, edges, DEFAULT_DP_LIMIT, None)? {
            Ok(t) => t,
            Err(r) => return Ok(Err(r)),
        }
    } else {
        match triangulate_with_islands(mesh, &outer.vertices, &island_slices, edges)? {
            Ok(t) => t,
            Err(r) => return Ok(Err(r)),
        }
    };

    let total_bnd_verts = island_slices
        .iter()
        .fold(outer.vertices.len(), |s, isl| s + isl.len());
    let mut all_boundary: Vec<VertexId> = Vec::new();
    reserve(&mut all_boundary, total_bnd_verts)?;
    all_boundary.extend_from_slice(&outer.vertices);
    for isl in &island_slices {
        all_boundary.extend_from_slice(isl);
    }

    let mut patch = if options.refine_patch {
        refine(mesh, fans, edges, &all_boundary, &base_tris, None)?
    } else {
        unrefined_patch(mesh, &all_boundary, &base_tris)?
    };

    let old_len = patch.old.len() as u32;
    let mut new_chords: Vec<(VertexId, VertexId)> = Vec::new();
    for t in &patch.triangles {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            if a >= old_len || b >= old_len {
                continue;
            }
            let (va, vb) = (patch.old[a as usize], patch.old[b as usize]);
            if edges.kind(va, vb).is_none() {
                push(&mut new_chords, if va < vb { (va, vb) } else { (vb, va) })?;
            }
        }
    }
    new_chords.sort_unstable();
    let manifold = new_chords.chunk_by(|a, b| a == b).all(|r| r.len() == 2);
    new_chords.dedup();
    if !manifold || new_chords.iter().any(|c| chords.binary_search(c).is_ok()) {
        return Ok(Err(FailReason::NoValidTriangulation));
    }

    let converged = match fair_with_continuity(
        mesh,
        fans,
        edges,
        &mut patch,
        options.continuity,
        options.smooth_iterations,
    )? {
        Ok(r) => r.converged,
        Err(reason) => return Ok(Err(reason)),
    };

    let max_dihedral_angle_deg = patch_max_dihedral_deg(mesh, edges, &patch)?;
    merge_sorted(chords, &new_chords)?;
    plan.add(&patch)?;

    Ok(Ok(FillStats {
        loop_index,
        filled_vertices: patch.new_vertices(),
        filled_triangles: patch.triangles.len(),
        perimeter: outer.perimeter,
        max_dihedral_angle_deg,
        continuity: options.continuity,
        island_loops_merged: island_indices.len(),
        converged,
    }))
}

fn unrefined_patch(
    mesh: &Mesh,
    boundary_vertices: &[VertexId],
    tris: &[[VertexId; 3]],
) -> Result<RefinedPatch> {
    let mut old: Vec<VertexId> = Vec::new();
    reserve(&mut old, boundary_vertices.len())?;
    old.extend_from_slice(boundary_vertices);
    old.sort_unstable();
    old.dedup();

    let mut points: Vec<[f64; 3]> = Vec::new();
    reserve(&mut points, old.len())?;
    points.extend(old.iter().map(|&v| point(mesh, v)));

    let mut local_tris: Vec<[u32; 3]> = Vec::new();
    reserve(&mut local_tris, tris.len())?;
    local_tris.extend(tris.iter().map(|t| {
        t.map(|v| {
            old.binary_search(&v)
                .expect("triangle vertex is on boundary") as u32
        })
    }));

    Ok(RefinedPatch {
        old,
        points,
        parents: Vec::new(),
        triangles: local_tris,
        capped: false,
    })
}

/// Computes the maximum dihedral angle (in degrees) across interior patch
/// edges and seam edges shared with the existing mesh.
fn patch_max_dihedral_deg(mesh: &Mesh, edges: &EdgeMap, patch: &RefinedPatch) -> Result<f64> {
    let nt = patch.triangles.len();
    let mut tri_normals: Vec<[f64; 3]> = Vec::new();
    reserve(&mut tri_normals, nt)?;
    for &t in &patch.triangles {
        let [a, b, c] = t.map(|i| patch.points[i as usize]);
        tri_normals.push(unit_normal(a, b, c));
    }

    let mut edge_recs: Vec<(u32, u32, u32)> = Vec::new();
    reserve(&mut edge_recs, nt * 3)?;
    for (ti, &t) in patch.triangles.iter().enumerate() {
        for (u, v) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edge_recs.push((u.min(v), u.max(v), ti as u32));
        }
    }
    edge_recs.sort_unstable();

    let mut max_deg = 0.0f64;
    let old_len = patch.old.len();
    let angle_deg = |n1: [f64; 3], n2: [f64; 3]| -> f64 {
        if n1 == [0.0; 3] || n2 == [0.0; 3] {
            return 0.0;
        }
        dot(n1, n2).clamp(-1.0, 1.0).acos().to_degrees()
    };

    let mut i = 0usize;
    while i < edge_recs.len() {
        let (u, v) = (edge_recs[i].0, edge_recs[i].1);
        let mut end = i + 1;
        while end < edge_recs.len() && edge_recs[end].0 == u && edge_recs[end].1 == v {
            end += 1;
        }
        if end - i == 2 {
            let n1 = tri_normals[edge_recs[i].2 as usize];
            let n2 = tri_normals[edge_recs[i + 1].2 as usize];
            max_deg = max_deg.max(angle_deg(n1, n2));
        } else if end - i == 1 && (u as usize) < old_len && (v as usize) < old_len {
            let n_patch = tri_normals[edge_recs[i].2 as usize];
            for &mesh_t in edges.triangles(patch.old[u as usize], patch.old[v as usize]) {
                let tri = mesh.triangles()[mesh_t as usize];
                let n_mesh = unit_normal(
                    point(mesh, tri[0]),
                    point(mesh, tri[1]),
                    point(mesh, tri[2]),
                );
                max_deg = max_deg.max(angle_deg(n_patch, n_mesh));
            }
        }
        i = end;
    }
    Ok(max_deg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::{AnalyzeOptions, analyze};
    use crate::test_util::{edit, grid_without, mesh_of, parts, sweep};

    #[test]
    fn fill_hole_respects_continuity_options_and_dihedral_stats() {
        let cells: Vec<(u32, u32)> = (2..6).flat_map(|y| (2..6).map(move |x| (x, y))).collect();
        let (mut spos, stris) = parts(&grid_without(8, &cells));
        let radius = 5.5f32;
        for p in spos.chunks_mut(3) {
            let (x, y) = (p[0] - 4.0, p[1] - 4.0);
            p[2] = (radius * radius - x * x - y * y).sqrt();
        }
        let base_cap = mesh_of(&spos, &stris);
        let loops = boundary_loops(&base_cap).unwrap();
        let hole_idx = loops.iter().position(|l| l.vertices.len() == 16).unwrap();

        let mut m_flat = mesh_of(&spos, &stris);
        let s_flat = edit(&mut m_flat, |tx| {
            fill_hole(
                tx,
                hole_idx,
                &FillHolesOptions::builder()
                    .continuity(FillContinuity::Flat)
                    .refine_patch(false)
                    .build(),
            )
        })
        .0;
        assert_eq!(s_flat.filled_vertices, 0);
        assert_eq!(s_flat.filled_triangles, 14);

        let mut m_g1 = mesh_of(&spos, &stris);
        let s_g1 = edit(&mut m_g1, |tx| {
            fill_hole(
                tx,
                hole_idx,
                &FillHolesOptions::builder()
                    .continuity(FillContinuity::TangentG1)
                    .refine_patch(true)
                    .build(),
            )
        })
        .0;
        assert!(s_g1.filled_vertices > 0 && s_g1.converged);

        let mut m_g2 = base_cap;
        let s_g2 = edit(&mut m_g2, |tx| {
            fill_hole(
                tx,
                hole_idx,
                &FillHolesOptions::builder()
                    .continuity(FillContinuity::CurvatureG2)
                    .smooth_iterations(2)
                    .build(),
            )
        })
        .0;
        assert!(s_g2.filled_vertices > 0 && s_g2.converged);
        assert!(
            s_g2.max_dihedral_angle_deg < s_flat.max_dihedral_angle_deg * 0.65,
            "G2 dihedral {} should be far below Flat {}",
            s_g2.max_dihedral_angle_deg,
            s_flat.max_dihedral_angle_deg
        );
        assert!(
            s_g2.max_dihedral_angle_deg < s_g1.max_dihedral_angle_deg * 0.65,
            "G2 dihedral {} should be lower than G1 {}",
            s_g2.max_dihedral_angle_deg,
            s_g1.max_dihedral_angle_deg
        );
    }

    #[test]
    fn fill_all_holes_seals_island_in_hole_and_skips_outer_rim() {
        let mut moat = Vec::new();
        for y in 1..7u32 {
            for x in 1..7u32 {
                if !((3..5).contains(&x) && (3..5).contains(&y)) {
                    moat.push((x, y));
                }
            }
        }
        let mut mesh = grid_without(8, &moat);
        let opts = FillHolesOptions::builder()
            .continuity(FillContinuity::CurvatureG2)
            .refine_patch(true)
            .build();
        let (stats, _) = edit(&mut mesh, |tx| fill_all_holes(tx, &opts));
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].island_loops_merged, 1);

        let after_loops = boundary_loops(&mesh).unwrap();
        assert_eq!(after_loops.len(), 1, "only outer sheet border remains");
        let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
        assert_eq!(a.non_manifold_edges, 0);
        assert_eq!(a.orientation_conflicts, 0);
    }

    #[test]
    fn fill_hole_sweep_keeps_atomicity_on_oom() {
        let cells: Vec<(u32, u32)> = (2..6).flat_map(|y| (2..6).map(move |x| (x, y))).collect();
        let mesh = grid_without(8, &cells);
        let loops = boundary_loops(&mesh).unwrap();
        let hole_idx = loops.iter().position(|l| l.vertices.len() == 16).unwrap();
        let opts = FillHolesOptions::builder()
            .continuity(FillContinuity::CurvatureG2)
            .refine_patch(true)
            .build();
        let points = sweep(mesh, |tx| {
            fill_hole(tx, hole_idx, &opts)?;
            Ok(())
        });
        assert!(points > 0);
    }
}
