//! Per-triangle constrained 2D sub-triangulation for mesh booleans.
//!
//! Projects a cut triangle onto its dominant normal axis with positive 2D
//! counter-clockwise orientation, resolves collinear point-on-segment splits
//! and strict 2D segment-segment crossings using exact `robust::orient2d`
//! predicates, inserts all split boundary and interior vertices, and recovers
//! every constraint segment via cavity ear-clipping.

use std::cmp::Ordering;

use fast_mesh_store::StoreError;
use robust::{Coord, orient2d};

use crate::boolean::arrangement::{ArrOrigin, ArrVertexId, TriConstraints};
use crate::{OpsError, Result, filled, push, reserve};

/// Coordinate-plane projection dropping the triangle's dominant normal axis,
/// ordered so the parent triangle `[c0, c1, c2]` has `orient2d > 0`.
#[derive(Clone, Copy)]
struct Proj2D {
    u_axis: usize,
    v_axis: usize,
}

impl Proj2D {
    fn from_corners(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> Result<Self> {
        let mut best: Option<(usize, f64)> = None;
        for axis in 0..3 {
            let u = (axis + 1) % 3;
            let v = (axis + 2) % 3;
            let det = orient2d(
                Coord { x: p0[u], y: p0[v] },
                Coord { x: p1[u], y: p1[v] },
                Coord { x: p2[u], y: p2[v] },
            );
            if det == 0.0 {
                continue;
            }
            if best.is_none_or(|(_, b)| det.abs() > b.abs()) {
                best = Some((axis, det));
            }
        }
        let Some((axis, det)) = best else {
            return Err(OpsError::InvalidInput(
                "boolean input has degenerate triangles",
            ));
        };
        let (u_axis, v_axis) = if det > 0.0 {
            ((axis + 1) % 3, (axis + 2) % 3)
        } else {
            ((axis + 2) % 3, (axis + 1) % 3)
        };
        Ok(Self { u_axis, v_axis })
    }

    fn project(self, p: [f32; 3]) -> Coord<f64> {
        Coord {
            x: f64::from(p[self.u_axis]) + 0.0,
            y: f64::from(p[self.v_axis]) + 0.0,
        }
    }
}

fn edge_key_arr(u: ArrVertexId, v: ArrVertexId) -> (ArrVertexId, ArrVertexId) {
    if u < v { (u, v) } else { (v, u) }
}

fn edge_key_local(u: usize, v: usize) -> (usize, usize) {
    if u < v { (u, v) } else { (v, u) }
}

fn coords_eq(a: Coord<f64>, b: Coord<f64>) -> bool {
    a.x == b.x && a.y == b.y
}

fn dist2(a: Coord<f64>, b: Coord<f64>) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    dx * dx + dy * dy
}

/// For `p` collinear with `a` and `b` (`orient2d(a, b, p) == 0`), whether `p`
/// lies strictly between `a` and `b`.
fn strictly_between_collinear(a: Coord<f64>, b: Coord<f64>, p: Coord<f64>) -> bool {
    if coords_eq(a, p) || coords_eq(b, p) {
        return false;
    }
    a.x.min(b.x) <= p.x && p.x <= a.x.max(b.x) && a.y.min(b.y) <= p.y && p.y <= a.y.max(b.y)
}

/// For `p` collinear with `a` and `b`, whether `p` lies on the closed segment `a..b`.
fn on_segment_collinear(a: Coord<f64>, b: Coord<f64>, p: Coord<f64>) -> bool {
    a.x.min(b.x) <= p.x && p.x <= a.x.max(b.x) && a.y.min(b.y) <= p.y && p.y <= a.y.max(b.y)
}

/// Exact test for whether closed 2D segments `a..b` and `c..d` share any point.
fn segments_intersect_2d(a: Coord<f64>, b: Coord<f64>, c: Coord<f64>, d: Coord<f64>) -> bool {
    let o1 = orient2d(a, b, c);
    let o2 = orient2d(a, b, d);
    let o3 = orient2d(c, d, a);
    let o4 = orient2d(c, d, b);
    let straddle_ab = (o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0);
    let straddle_cd = (o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0);
    (straddle_ab && straddle_cd)
        || (o1 == 0.0 && on_segment_collinear(a, b, c))
        || (o2 == 0.0 && on_segment_collinear(a, b, d))
        || (o3 == 0.0 && on_segment_collinear(c, d, a))
        || (o4 == 0.0 && on_segment_collinear(c, d, b))
}

/// Whether `p` lies in or on the closed counter-clockwise triangle `(a, b, c)`.
fn in_closed_ccw_tri(a: Coord<f64>, b: Coord<f64>, c: Coord<f64>, p: Coord<f64>) -> bool {
    orient2d(a, b, p) >= 0.0 && orient2d(b, c, p) >= 0.0 && orient2d(c, a, p) >= 0.0
}

fn closest_point_on_seg_2d(p: Coord<f64>, u: Coord<f64>, v: Coord<f64>) -> Coord<f64> {
    let dx = v.x - u.x;
    let dy = v.y - u.y;
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 {
        ((p.x - u.x) * dx + (p.y - u.y) * dy) / len2
    } else {
        0.0
    }
    .clamp(0.0, 1.0);
    Coord {
        x: u.x + t * dx,
        y: u.y + t * dy,
    }
}

fn closest_point_on_tri_2d(
    p: Coord<f64>,
    a: Coord<f64>,
    b: Coord<f64>,
    c: Coord<f64>,
) -> Coord<f64> {
    if in_closed_ccw_tri(a, b, c, p) {
        return p;
    }
    let q0 = closest_point_on_seg_2d(p, a, b);
    let q1 = closest_point_on_seg_2d(p, b, c);
    let q2 = closest_point_on_seg_2d(p, c, a);
    let d0 = dist2(p, q0);
    let d1 = dist2(p, q1);
    let d2 = dist2(p, q2);
    if d0 <= d1 && d0 <= d2 {
        q0
    } else if d1 <= d2 {
        q1
    } else {
        q2
    }
}

/// Picks the smallest dyadic `step` in `[2^-40, 0.5 / (count + 1)]` (falling back
/// to uniform `1 / (count + 1)`) such that all adjacent pairs along
/// `[p_tie, p_tie + 1*step*(target_pt - p_tie), ..., p_tie + count*step*(target_pt - p_tie), target_pt]`
/// are strictly distinct in `f64`.
fn select_tie_nudge_step(p_tie: Coord<f64>, target_pt: Coord<f64>, count: usize) -> f64 {
    let max_step = 0.5 / ((count + 1) as f64);
    let dx = target_pt.x - p_tie.x;
    let dy = target_pt.y - p_tie.y;
    let is_distinct_chain = |step: f64| -> bool {
        let mut prev = p_tie;
        for j in 1..=count {
            let frac = (j as f64) * step;
            let curr = Coord {
                x: p_tie.x + frac * dx,
                y: p_tie.y + frac * dy,
            };
            if coords_eq(prev, curr) {
                return false;
            }
            prev = curr;
        }
        !coords_eq(prev, target_pt)
    };

    for exp in -40..=-1 {
        let step = 2.0_f64.powi(exp).min(max_step);
        if is_distinct_chain(step) {
            return step;
        }
    }
    1.0 / ((count + 1) as f64)
}

fn read_pos_f64(positions: &[[f32; 3]], v: ArrVertexId) -> Result<[f64; 3]> {
    let Some(&p) = positions.get(v as usize) else {
        return Err(OpsError::InvalidInput(
            "boolean arrangement vertex out of bounds",
        ));
    };
    if !p.iter().all(|c| c.is_finite()) {
        return Err(OpsError::InvalidInput("boolean point leaves f32 range"));
    }
    Ok([
        f64::from(p[0]) + 0.0,
        f64::from(p[1]) + 0.0,
        f64::from(p[2]) + 0.0,
    ])
}

/// Sub-triangulates a single mesh triangle `[c0, c1, c2]` subject to split
/// boundary vertices, interior vertices, and constraint segments.
pub(super) fn triangulate_constrained(
    corners: [ArrVertexId; 3],
    constraints: &TriConstraints,
    positions: &mut Vec<[f32; 3]>,
    origins: &mut Vec<ArrOrigin>,
    seam_edges: &mut Vec<(ArrVertexId, ArrVertexId)>,
) -> Result<Vec<[ArrVertexId; 3]>> {
    if corners[0] == corners[1] || corners[1] == corners[2] || corners[0] == corners[2] {
        return Err(OpsError::InvalidInput(
            "boolean input has degenerate triangles",
        ));
    }
    let p0 = read_pos_f64(positions, corners[0])?;
    let p1 = read_pos_f64(positions, corners[1])?;
    let p2 = read_pos_f64(positions, corners[2])?;
    let proj = Proj2D::from_corners(p0, p1, p2)?;

    if constraints.edge_splits.iter().all(Vec::is_empty)
        && constraints.inner_verts.is_empty()
        && constraints.segments.is_empty()
    {
        let mut out = Vec::new();
        push(&mut out, corners)?;
        return Ok(out);
    }

    let orig_pos_len = positions.len();
    let orig_orig_len = origins.len();
    let res = (|| {
        let mut cdt = Cdt::build(corners, constraints, positions, proj)?;
        cdt.resolve_segments(positions, origins, seam_edges)?;
        cdt.insert_interior_vertices()?;
        cdt.recover_constraints()?;
        cdt.emit()
    })();
    if res.is_err() {
        positions.truncate(orig_pos_len);
        origins.truncate(orig_orig_len);
    }
    res
}

struct Cdt {
    proj: Proj2D,
    verts: Vec<ArrVertexId>,
    pts: Vec<Coord<f64>>,
    num_boundary: usize,
    ring: Vec<usize>,
    side_chains: [Vec<usize>; 3],
    segs: Vec<(usize, usize)>,
    tris: Vec<[usize; 3]>,
}

impl Cdt {
    fn find_or_insert(&mut self, v: ArrVertexId, positions: &[[f32; 3]]) -> Result<usize> {
        if let Some(idx) = self.verts.iter().position(|&x| x == v) {
            return Ok(idx);
        }
        let _ = read_pos_f64(positions, v)?;
        let p2 = self.proj.project(positions[v as usize]);
        let idx = self.verts.len();
        push(&mut self.verts, v)?;
        push(&mut self.pts, p2)?;
        Ok(idx)
    }

    fn build(
        corners: [ArrVertexId; 3],
        constraints: &TriConstraints,
        positions: &[[f32; 3]],
        proj: Proj2D,
    ) -> Result<Self> {
        let mut cdt = Self {
            proj,
            verts: Vec::new(),
            pts: Vec::new(),
            num_boundary: 0,
            ring: Vec::new(),
            side_chains: [Vec::new(), Vec::new(), Vec::new()],
            segs: Vec::new(),
            tris: Vec::new(),
        };

        // 0, 1, 2 are always the parent corners.
        for &c in &corners {
            let _ = cdt.find_or_insert(c, positions)?;
        }

        // Register boundary split vertices per edge in their given split order,
        // building the CCW outer split ring and per-side chains.
        let mut edge_local: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (edge_i, splits) in constraints.edge_splits.iter().enumerate() {
            for &v in splits {
                if cdt.verts.contains(&v) {
                    continue;
                }
                let idx = cdt.find_or_insert(v, positions)?;
                push(&mut edge_local[edge_i], idx)?;
            }
        }
        cdt.num_boundary = cdt.verts.len();

        reserve(&mut cdt.ring, cdt.num_boundary)?;
        for (edge_i, splits) in edge_local.iter().enumerate() {
            push(&mut cdt.ring, edge_i)?;
            reserve(&mut cdt.side_chains[edge_i], splits.len() + 2)?;
            cdt.side_chains[edge_i].push(edge_i);
            for &idx in splits {
                push(&mut cdt.ring, idx)?;
                cdt.side_chains[edge_i].push(idx);
            }
            cdt.side_chains[edge_i].push((edge_i + 1) % 3);
        }

        // Guard against two distinct boundary vertices projecting to the exact
        // same 2D point when differing by 1 ULP along the dropped normal axis.
        for edge_i in 0..3 {
            let chain = &cdt.side_chains[edge_i];
            let pts = &mut cdt.pts;
            let m = chain.len();
            if m <= 2 {
                continue;
            }
            let mut l = 0usize;
            while l < m {
                let p_tie = pts[chain[l]];
                let mut r = l;
                while r + 1 < m && coords_eq(pts[chain[r + 1]], p_tie) {
                    r += 1;
                }
                if r > l {
                    let count = r - l;
                    if r < m - 1 {
                        // Forward pass: keep `chain[l]` (including `c_start` when `l == 0`)
                        // at `p_tie` and nudge `chain[l + 1..=r]` slightly toward `next_pt`.
                        let next_pt = pts[chain[r + 1]];
                        let step = select_tie_nudge_step(p_tie, next_pt, count);
                        for (k, &v_idx) in chain[(l + 1)..=r].iter().enumerate() {
                            let frac = ((k + 1) as f64) * step;
                            pts[v_idx] = Coord {
                                x: p_tie.x + frac * (next_pt.x - p_tie.x),
                                y: p_tie.y + frac * (next_pt.y - p_tie.y),
                            };
                        }
                    } else if l > 0 {
                        // Backward pass: keep `chain[m - 1]` (`c_end`) at `p_tie` and nudge
                        // `chain[l..m - 1]` slightly backward toward `before_pt`.
                        let before_pt = pts[chain[l - 1]];
                        let step = select_tie_nudge_step(p_tie, before_pt, count);
                        for (k, &v_idx) in chain[l..(m - 1)].iter().enumerate() {
                            let back = ((count - k) as f64) * step;
                            pts[v_idx] = Coord {
                                x: p_tie.x + back * (before_pt.x - p_tie.x),
                                y: p_tie.y + back * (before_pt.y - p_tie.y),
                            };
                        }
                    }
                }
                l = r + 1;
            }
        }

        // Triangulate the outer split ring immediately; this forms the exact
        // partition of the polygon interior before any interior vertices enter.
        ear_clip_polygon(&cdt.ring, &cdt.pts, &mut cdt.tris)?;

        // Register interior vertices and any segment endpoints not yet listed.
        for &v in &constraints.inner_verts {
            let _ = cdt.find_or_insert(v, positions)?;
        }
        for &(ua, va) in &constraints.segments {
            let u = cdt.find_or_insert(ua, positions)?;
            let v = cdt.find_or_insert(va, positions)?;
            if u != v {
                push(&mut cdt.segs, edge_key_local(u, v))?;
            }
        }
        cdt.segs.sort_unstable();
        cdt.segs.dedup();

        // Ensure every interior vertex lies strictly inside the outer split
        // polygon (handling 1-ULP f32 rounding onto/outside the boundary).
        for idx in cdt.num_boundary..cdt.verts.len() {
            cdt.ensure_strictly_interior_point(idx);
        }

        Ok(cdt)
    }

    fn is_outer_ring_edge(&self, a: usize, b: usize) -> bool {
        if a >= self.num_boundary || b >= self.num_boundary {
            return false;
        }
        let n = self.ring.len();
        for i in 0..n {
            let u = self.ring[i];
            let v = self.ring[(i + 1) % n];
            if (u == a && v == b) || (u == b && v == a) {
                return true;
            }
        }
        false
    }

    /// True when `p` lies strictly inside the outer split ring (either strictly
    /// inside an initial triangle, or strictly on an interior diagonal of `self.tris`),
    /// and does not coincide in 2D with any earlier vertex `0..limit`.
    fn is_valid_interior_coord(&self, p: Coord<f64>, limit: usize) -> bool {
        if self.pts[..limit].iter().any(|&q| coords_eq(p, q)) {
            return false;
        }
        for &[a, b, c] in &self.tris {
            let o0 = orient2d(self.pts[a], self.pts[b], p);
            let o1 = orient2d(self.pts[b], self.pts[c], p);
            let o2 = orient2d(self.pts[c], self.pts[a], p);
            if o0 > 0.0 && o1 > 0.0 && o2 > 0.0 {
                return true;
            }
            if o0 == 0.0 && o1 > 0.0 && o2 > 0.0 && !self.is_outer_ring_edge(a, b) {
                return true;
            }
            if o1 == 0.0 && o2 > 0.0 && o0 > 0.0 && !self.is_outer_ring_edge(b, c) {
                return true;
            }
            if o2 == 0.0 && o0 > 0.0 && o1 > 0.0 && !self.is_outer_ring_edge(c, a) {
                return true;
            }
        }
        false
    }

    fn ensure_strictly_interior_point(&mut self, idx: usize) {
        let orig = self.pts[idx];
        if self.is_valid_interior_coord(orig, idx) {
            return;
        }
        // Clamp `orig` onto the closest initial triangle in `self.tris` first so
        // tangential position along thin slivers is preserved before stepping inward.
        let mut best_tri = self.tris[0];
        let mut best_clamped = orig;
        let mut best_d2 = f64::INFINITY;
        for &[a, b, c] in &self.tris {
            let clamped = closest_point_on_tri_2d(orig, self.pts[a], self.pts[b], self.pts[c]);
            let d2 = dist2(orig, clamped);
            if d2 < best_d2 {
                best_d2 = d2;
                best_tri = [a, b, c];
                best_clamped = clamped;
            }
        }
        let [ta, tb, tc] = best_tri;
        let cen = Coord {
            x: (self.pts[ta].x + self.pts[tb].x + self.pts[tc].x) / 3.0,
            y: (self.pts[ta].y + self.pts[tb].y + self.pts[tc].y) / 3.0,
        };
        if !coords_eq(best_clamped, cen) {
            for exp in -40..=-1 {
                let alpha = 2.0_f64.powi(exp);
                let cand = Coord {
                    x: best_clamped.x + alpha * (cen.x - best_clamped.x),
                    y: best_clamped.y + alpha * (cen.y - best_clamped.y),
                };
                if self.is_valid_interior_coord(cand, idx) {
                    self.pts[idx] = cand;
                    return;
                }
            }
        }
        if self.is_valid_interior_coord(cen, idx) {
            self.pts[idx] = cen;
            return;
        }
        for exp in -40..=-1 {
            let beta = 2.0_f64.powi(exp);
            for &corner in &[self.pts[ta], self.pts[tb], self.pts[tc]] {
                let cand = Coord {
                    x: cen.x + beta * (corner.x - cen.x),
                    y: cen.y + beta * (corner.y - cen.y),
                };
                if self.is_valid_interior_coord(cand, idx) {
                    self.pts[idx] = cand;
                    return;
                }
            }
        }
    }

    /// Resolves collinear vertex-on-segment splits and strict 2D crossings,
    /// updating `positions`, `origins`, and `seam_edges` when segments split.
    fn resolve_segments(
        &mut self,
        positions: &mut Vec<[f32; 3]>,
        origins: &mut Vec<ArrOrigin>,
        seam_edges: &mut Vec<(ArrVertexId, ArrVertexId)>,
    ) -> Result<()> {
        if self.segs.is_empty() {
            return Ok(());
        }
        let mut seam_modified = self.stabilize_segments(seam_edges)?;

        let max_crossings = self
            .segs
            .len()
            .saturating_mul(self.segs.len())
            .saturating_add(64);
        let mut crossings_resolved = 0usize;
        while let Some((i, j, t_ab, u_cd)) = self.find_first_strict_crossing() {
            if crossings_resolved >= max_crossings {
                return Err(OpsError::InvalidInput(
                    "boolean arrangement has too many segment crossings",
                ));
            }
            crossings_resolved += 1;

            let (a, b) = self.segs[i];
            let (c, d) = self.segs[j];
            let pa3 = read_pos_f64(positions, self.verts[a])?;
            let pb3 = read_pos_f64(positions, self.verts[b])?;
            let pc3 = read_pos_f64(positions, self.verts[c])?;
            let pd3 = read_pos_f64(positions, self.verts[d])?;

            let mut pos_f32 = [0.0f32; 3];
            for k in 0..3 {
                let on_ab = pa3[k] + t_ab * (pb3[k] - pa3[k]);
                let on_cd = pc3[k] + u_cd * (pd3[k] - pc3[k]);
                pos_f32[k] = (0.5 * (on_ab + on_cd)) as f32;
            }
            if !pos_f32.iter().all(|x| x.is_finite()) {
                return Err(OpsError::InvalidInput("boolean point leaves f32 range"));
            }

            let p2 = self.proj.project(pos_f32);
            let mid = if let Some(existing) = self.pts.iter().position(|&q| coords_eq(q, p2)) {
                existing
            } else {
                let new_arr = u32::try_from(positions.len())
                    .map_err(|_| OpsError::Store(StoreError::OutOfMemory))?;
                let prev_pos_len = positions.len();
                push(positions, pos_f32)?;
                if let Err(e) = push(origins, ArrOrigin::Fresh) {
                    positions.truncate(prev_pos_len);
                    return Err(e);
                }
                let idx = self.verts.len();
                push(&mut self.verts, new_arr)?;
                push(&mut self.pts, p2)?;
                self.ensure_strictly_interior_point(idx);
                idx
            };

            // Remove j first (j > i), then i, and add the up to four split halves.
            self.segs.remove(j);
            self.segs.remove(i);
            for (u, v) in [(a, mid), (mid, b), (c, mid), (mid, d)] {
                if u != v {
                    push(&mut self.segs, edge_key_local(u, v))?;
                }
            }
            self.segs.sort_unstable();
            self.segs.dedup();

            let arr_mid = self.verts[mid];
            if replace_seam_segment(
                seam_edges,
                self.verts[a],
                self.verts[b],
                &[self.verts[a], arr_mid, self.verts[b]],
            )? {
                seam_modified = true;
            }
            if replace_seam_segment(
                seam_edges,
                self.verts[c],
                self.verts[d],
                &[self.verts[c], arr_mid, self.verts[d]],
            )? {
                seam_modified = true;
            }

            if self.stabilize_segments(seam_edges)? {
                seam_modified = true;
            }
        }

        if seam_modified {
            seam_edges.sort_unstable();
            seam_edges.dedup();
        }
        Ok(())
    }

    fn stabilize_segments(
        &mut self,
        seam_edges: &mut Vec<(ArrVertexId, ArrVertexId)>,
    ) -> Result<bool> {
        let mut seam_modified = false;
        let max_passes = self
            .verts
            .len()
            .saturating_mul(self.verts.len())
            .saturating_add(16);
        for _ in 0..max_passes {
            if self.split_same_boundary_side_segments(seam_edges)? {
                seam_modified = true;
            }
            if self.split_all_collinear(seam_edges)? {
                seam_modified = true;
            }
            let (grazing_changed, grazing_seam) =
                self.snap_boundary_grazing_segments(seam_edges)?;
            if grazing_seam {
                seam_modified = true;
            }
            if !grazing_changed {
                break;
            }
        }
        Ok(seam_modified)
    }

    fn is_ring_vertex_reflex(&self, ri: usize) -> bool {
        let n = self.ring.len();
        let prev = self.ring[(ri + n - 1) % n];
        let curr = self.ring[ri];
        let next = self.ring[(ri + 1) % n];
        orient2d(self.pts[prev], self.pts[curr], self.pts[next]) < 0.0
    }

    /// Splits any segment `(u, v)` whose endpoints both belong to the same
    /// parent side chain across any intervening edge-split vertices.
    fn split_same_boundary_side_segments(
        &mut self,
        seam_edges: &mut Vec<(ArrVertexId, ArrVertexId)>,
    ) -> Result<bool> {
        let mut seam_modified = false;
        let mut next_segs: Vec<(usize, usize)> = Vec::new();
        reserve(&mut next_segs, self.segs.len())?;

        for &(u, v) in &self.segs {
            let mut matched_chain: Option<(usize, usize, usize)> = None;
            if u < self.num_boundary && v < self.num_boundary {
                for (side_i, chain) in self.side_chains.iter().enumerate() {
                    if chain.len() < 3 {
                        continue;
                    }
                    if let (Some(pu), Some(pv)) = (
                        chain.iter().position(|&x| x == u),
                        chain.iter().position(|&x| x == v),
                    ) {
                        let (lo, hi) = if pu < pv { (pu, pv) } else { (pv, pu) };
                        if hi - lo >= 2 {
                            matched_chain = Some((side_i, lo, hi));
                            break;
                        }
                    }
                }
            }
            let Some((side_i, lo, hi)) = matched_chain else {
                push(&mut next_segs, (u, v))?;
                continue;
            };
            let chain = &self.side_chains[side_i];
            let mut chain_arr: Vec<ArrVertexId> = Vec::new();
            reserve(&mut chain_arr, hi - lo + 1)?;
            for k in lo..hi {
                push(&mut next_segs, edge_key_local(chain[k], chain[k + 1]))?;
                chain_arr.push(self.verts[chain[k]]);
            }
            chain_arr.push(self.verts[chain[hi]]);
            if replace_seam_segment(seam_edges, self.verts[u], self.verts[v], &chain_arr)? {
                seam_modified = true;
            }
        }

        next_segs.sort_unstable();
        next_segs.dedup();
        self.segs = next_segs;
        Ok(seam_modified)
    }

    /// If an interior constraint segment grazes across a 1-ULP inward-bent
    /// boundary split vertex `s` on the outer split ring, snaps `(u, v)` through
    /// `s` into `(u, s)` and `(s, v)`. Returns `(segs_changed, seam_modified)`.
    fn snap_boundary_grazing_segments(
        &mut self,
        seam_edges: &mut Vec<(ArrVertexId, ArrVertexId)>,
    ) -> Result<(bool, bool)> {
        let mut segs_changed = false;
        let mut seam_modified = false;
        let ring_len = self.ring.len();
        if ring_len <= 3 {
            return Ok((false, false));
        }
        loop {
            let mut split_target: Option<(usize, usize)> = None;
            for (si, &(u, v)) in self.segs.iter().enumerate() {
                if self.is_outer_ring_edge(u, v) {
                    continue;
                }
                let (pu, pv) = (self.pts[u], self.pts[v]);
                for ri in 0..ring_len {
                    let next_ri = (ri + 1) % ring_len;
                    let ra = self.ring[ri];
                    let rb = self.ring[next_ri];
                    if ra == u || ra == v || rb == u || rb == v {
                        continue;
                    }
                    if segments_intersect_2d(pu, pv, self.pts[ra], self.pts[rb]) {
                        // Prefer a reflex (inward-bent) non-corner split vertex so
                        // snapping pulls the chord around the inward notch rather than
                        // onto an adjacent outward-bent split vertex.
                        let ra_reflex = ra >= 3 && self.is_ring_vertex_reflex(ri);
                        let rb_reflex = rb >= 3 && self.is_ring_vertex_reflex(next_ri);
                        let pick = if ra_reflex && !rb_reflex {
                            ra
                        } else if rb_reflex && !ra_reflex {
                            rb
                        } else if ra >= 3 && rb >= 3 {
                            if orient2d(pu, pv, self.pts[ra]).abs()
                                <= orient2d(pu, pv, self.pts[rb]).abs()
                            {
                                ra
                            } else {
                                rb
                            }
                        } else if ra >= 3 {
                            ra
                        } else {
                            rb
                        };
                        split_target = Some((si, pick));
                        break;
                    }
                }
                if split_target.is_some() {
                    break;
                }
            }
            let Some((si, mid)) = split_target else {
                break;
            };
            segs_changed = true;
            let (u, v) = self.segs.remove(si);
            if u != mid {
                push(&mut self.segs, edge_key_local(u, mid))?;
            }
            if mid != v {
                push(&mut self.segs, edge_key_local(mid, v))?;
            }
            self.segs.sort_unstable();
            self.segs.dedup();
            if replace_seam_segment(
                seam_edges,
                self.verts[u],
                self.verts[v],
                &[self.verts[u], self.verts[mid], self.verts[v]],
            )? {
                seam_modified = true;
            }
        }
        Ok((segs_changed, seam_modified))
    }

    fn find_first_strict_crossing(&self) -> Option<(usize, usize, f64, f64)> {
        for i in 0..self.segs.len() {
            let (a, b) = self.segs[i];
            let (pa, pb) = (self.pts[a], self.pts[b]);
            for j in (i + 1)..self.segs.len() {
                let (c, d) = self.segs[j];
                if a == c || a == d || b == c || b == d {
                    continue;
                }
                let (pc, pd) = (self.pts[c], self.pts[d]);
                let o1 = orient2d(pa, pb, pc);
                let o2 = orient2d(pa, pb, pd);
                let straddle_ab = (o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0);
                if !straddle_ab {
                    continue;
                }
                let o3 = orient2d(pc, pd, pa);
                let o4 = orient2d(pc, pd, pb);
                let straddle_cd = (o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0);
                if !straddle_cd {
                    continue;
                }
                let t_ab = (o3 / (o3 - o4)).clamp(0.0, 1.0);
                let u_cd = (o1 / (o1 - o2)).clamp(0.0, 1.0);
                return Some((i, j, t_ab, u_cd));
            }
        }
        None
    }

    /// Splits every constraint segment at any local vertex lying strictly on its
    /// interior (`orient2d == 0`). Returns whether `seam_edges` was modified.
    fn split_all_collinear(
        &mut self,
        seam_edges: &mut Vec<(ArrVertexId, ArrVertexId)>,
    ) -> Result<bool> {
        let mut seam_modified = false;
        loop {
            let mut changed = false;
            let mut next_segs: Vec<(usize, usize)> = Vec::new();
            reserve(&mut next_segs, self.segs.len())?;

            for &(u, v) in &self.segs {
                let pu = self.pts[u];
                let pv = self.pts[v];
                let mut mids: Vec<usize> = Vec::new();
                for w in 0..self.verts.len() {
                    if w == u || w == v {
                        continue;
                    }
                    let pw = self.pts[w];
                    if orient2d(pu, pv, pw) == 0.0 && strictly_between_collinear(pu, pv, pw) {
                        push(&mut mids, w)?;
                    }
                }
                if mids.is_empty() {
                    push(&mut next_segs, (u, v))?;
                    continue;
                }
                changed = true;
                mids.sort_unstable_by(|&x, &y| {
                    compare_along_collinear(
                        pu,
                        pv,
                        self.pts[x],
                        self.pts[y],
                        self.verts[x],
                        self.verts[y],
                    )
                });

                let mut chain_arr: Vec<ArrVertexId> = Vec::new();
                reserve(&mut chain_arr, mids.len() + 2)?;
                push(&mut chain_arr, self.verts[u])?;

                let mut prev = u;
                for m in mids {
                    if m != prev && !coords_eq(self.pts[m], self.pts[prev]) {
                        push(&mut next_segs, edge_key_local(prev, m))?;
                        push(&mut chain_arr, self.verts[m])?;
                        prev = m;
                    }
                }
                if prev != v {
                    push(&mut next_segs, edge_key_local(prev, v))?;
                    push(&mut chain_arr, self.verts[v])?;
                }

                if replace_seam_segment(seam_edges, self.verts[u], self.verts[v], &chain_arr)? {
                    seam_modified = true;
                }
            }

            next_segs.sort_unstable();
            next_segs.dedup();
            self.segs = next_segs;
            if !changed {
                break;
            }
        }
        Ok(seam_modified)
    }

    /// Inserts all interior vertices (`num_boundary..verts.len()`) into `self.tris`
    /// via 1-to-3 triangle splits or 2-to-4 shared-edge splits.
    fn insert_interior_vertices(&mut self) -> Result<()> {
        for p in self.num_boundary..self.verts.len() {
            let pp = self.pts[p];
            let mut located_in: Option<usize> = None;
            let mut located_on_edge: Option<(usize, usize, usize, usize)> = None;

            for (ti, &[a, b, c]) in self.tris.iter().enumerate() {
                let o0 = orient2d(self.pts[a], self.pts[b], pp);
                let o1 = orient2d(self.pts[b], self.pts[c], pp);
                let o2 = orient2d(self.pts[c], self.pts[a], pp);
                if o0 > 0.0 && o1 > 0.0 && o2 > 0.0 {
                    located_in = Some(ti);
                    break;
                }
                if o0 == 0.0 && o1 > 0.0 && o2 > 0.0 {
                    located_on_edge = Some((ti, a, b, c));
                    break;
                }
                if o1 == 0.0 && o2 > 0.0 && o0 > 0.0 {
                    located_on_edge = Some((ti, b, c, a));
                    break;
                }
                if o2 == 0.0 && o0 > 0.0 && o1 > 0.0 {
                    located_on_edge = Some((ti, c, a, b));
                    break;
                }
            }

            let mut flip_stack: Vec<(usize, usize)> = Vec::new();
            if let Some(ti) = located_in {
                let [a, b, c] = self.tris[ti];
                reserve(&mut self.tris, 2)?;
                self.tris[ti] = [a, b, p];
                self.tris.push([b, c, p]);
                self.tris.push([c, a, p]);
                reserve(&mut flip_stack, 3)?;
                flip_stack.push(edge_key_local(a, b));
                flip_stack.push(edge_key_local(b, c));
                flip_stack.push(edge_key_local(c, a));
            } else if let Some((ti, u, v, c)) = located_on_edge {
                // Find the twin triangle carrying directed half-edge `v -> u`.
                let Some((tj, d)) = self.find_half_edge_tri(v, u) else {
                    return Err(OpsError::InvalidInput(
                        "boolean interior vertex fell on boundary edge",
                    ));
                };
                reserve(&mut self.tris, 2)?;
                self.tris[ti] = [u, p, c];
                self.tris[tj] = [v, p, d];
                self.tris.push([p, v, c]);
                self.tris.push([p, u, d]);
                reserve(&mut flip_stack, 4)?;
                flip_stack.push(edge_key_local(u, c));
                flip_stack.push(edge_key_local(c, v));
                flip_stack.push(edge_key_local(v, d));
                flip_stack.push(edge_key_local(d, u));
            } else {
                return Err(OpsError::InvalidInput(
                    "boolean interior vertex outside triangle",
                ));
            }

            self.legalize_edges(&mut flip_stack, false)?;
        }
        Ok(())
    }

    /// Finds the triangle containing directed half-edge `u -> v` and returns
    /// `(tri_idx, opposite_vert)`.
    fn find_half_edge_tri(&self, u: usize, v: usize) -> Option<(usize, usize)> {
        for (ti, &[a, b, c]) in self.tris.iter().enumerate() {
            if a == u && b == v {
                return Some((ti, c));
            }
            if b == u && c == v {
                return Some((ti, a));
            }
            if c == u && a == v {
                return Some((ti, b));
            }
        }
        None
    }

    fn has_undirected_edge(&self, u: usize, v: usize) -> bool {
        self.find_half_edge_tri(u, v).is_some() || self.find_half_edge_tri(v, u).is_some()
    }

    fn is_constrained_segment(&self, u: usize, v: usize) -> bool {
        self.segs.binary_search(&edge_key_local(u, v)).is_ok()
    }

    /// Flips non-boundary (and when `respect_constraints` is true, non-constrained)
    /// interior diagonals that strictly improve minimum triangle quality while
    /// maintaining exact `orient2d > 0` on both triangles.
    fn legalize_edges(
        &mut self,
        stack: &mut Vec<(usize, usize)>,
        respect_constraints: bool,
    ) -> Result<()> {
        let max_flips = self.tris.len().saturating_mul(4).saturating_add(16);
        let mut flips = 0usize;
        while let Some((u, v)) = stack.pop() {
            if flips >= max_flips {
                break;
            }
            if self.is_outer_ring_edge(u, v) {
                continue;
            }
            if respect_constraints && self.is_constrained_segment(u, v) {
                continue;
            }
            let Some((t1, c)) = self.find_half_edge_tri(u, v) else {
                continue;
            };
            let Some((t2, d)) = self.find_half_edge_tri(v, u) else {
                continue;
            };
            if c == d || self.has_undirected_edge(c, d) {
                continue;
            }
            // In quad (u, d, v, c), flipping u-v to d-c forms [d, c, u] and [c, d, v].
            let o_new1 = orient2d(self.pts[d], self.pts[c], self.pts[u]);
            let o_new2 = orient2d(self.pts[c], self.pts[d], self.pts[v]);
            if o_new1 <= 0.0 || o_new2 <= 0.0 {
                continue;
            }
            let cur_q = tri_quality(self.pts[u], self.pts[v], self.pts[c]).min(tri_quality(
                self.pts[v],
                self.pts[u],
                self.pts[d],
            ));
            let new_q = tri_quality(self.pts[d], self.pts[c], self.pts[u]).min(tri_quality(
                self.pts[c],
                self.pts[d],
                self.pts[v],
            ));
            if new_q > cur_q + 1e-12 {
                self.tris[t1] = [d, c, u];
                self.tris[t2] = [c, d, v];
                flips += 1;
                reserve(stack, 4)?;
                stack.push(edge_key_local(u, d));
                stack.push(edge_key_local(d, v));
                stack.push(edge_key_local(v, c));
                stack.push(edge_key_local(c, u));
            }
        }
        Ok(())
    }

    /// Enforces every constraint segment `(u, v)` in `self.segs` by walking the
    /// crossed triangle cavity and ear-clipping the lower and upper cavity polygons.
    fn recover_constraints(&mut self) -> Result<()> {
        let mut work: Vec<(usize, usize)> = Vec::new();
        reserve(&mut work, self.segs.len())?;
        for &seg in &self.segs {
            work.push(seg);
        }

        let mut cursor = 0usize;
        while cursor < work.len() {
            let (u, v) = work[cursor];
            cursor += 1;
            if u == v || self.has_undirected_edge(u, v) {
                continue;
            }
            if let Some(mid) = self.recover_single_segment(u, v)? {
                // Defensive self-healing if a vertex ever lay exactly on u..v.
                let s1 = edge_key_local(u, mid);
                let s2 = edge_key_local(mid, v);
                push(&mut self.segs, s1)?;
                push(&mut self.segs, s2)?;
                self.segs.sort_unstable();
                self.segs.dedup();
                push(&mut work, s1)?;
                push(&mut work, s2)?;
            }
        }
        Ok(())
    }

    /// Recovers segment `u -> v`. Returns `Ok(Some(mid))` only if an unexpected
    /// collinear vertex `mid` was encountered on `u..v` so the caller splits it.
    fn recover_single_segment(&mut self, u: usize, v: usize) -> Result<Option<usize>> {
        let pu = self.pts[u];
        let pv = self.pts[v];

        // Find the unique triangle `[u, r0, l0]` around `u` straddling `u -> v`.
        let mut start: Option<(usize, usize, usize)> = None;
        for (ti, &[a, b, c]) in self.tris.iter().enumerate() {
            let (x, y) = if a == u {
                (b, c)
            } else if b == u {
                (c, a)
            } else if c == u {
                (a, b)
            } else {
                continue;
            };
            let ox = orient2d(pu, pv, self.pts[x]);
            let oy = orient2d(pu, pv, self.pts[y]);
            if ox == 0.0 && strictly_between_collinear(pu, pv, self.pts[x]) {
                return Ok(Some(x));
            }
            if oy == 0.0 && strictly_between_collinear(pu, pv, self.pts[y]) {
                return Ok(Some(y));
            }
            if ox < 0.0 && oy > 0.0 {
                start = Some((ti, x, y));
                break;
            }
        }
        let Some((t0, mut r, mut l)) = start else {
            return Err(OpsError::InvalidInput(
                "boolean constraint start sector not found",
            ));
        };

        let mut in_cavity = filled(self.tris.len(), false)?;
        in_cavity[t0] = true;

        let mut lower: Vec<usize> = Vec::new();
        let mut upper: Vec<usize> = Vec::new();
        push(&mut lower, u)?;
        push(&mut lower, r)?;
        push(&mut upper, u)?;
        push(&mut upper, l)?;

        let mut steps = 0usize;
        let max_steps = self.tris.len();
        loop {
            if steps >= max_steps {
                return Err(OpsError::InvalidInput(
                    "boolean cavity walk did not terminate",
                ));
            }
            steps += 1;

            // Crossed edge in current triangle is `r -> l`, so adjacent has `l -> r`.
            let Some((tnext, w)) = self.find_half_edge_tri(l, r) else {
                return Err(OpsError::InvalidInput(
                    "boolean constraint crosses triangle boundary",
                ));
            };
            if in_cavity[tnext] {
                return Err(OpsError::InvalidInput(
                    "boolean cavity walk revisited triangle",
                ));
            }
            in_cavity[tnext] = true;

            if w == v {
                push(&mut lower, v)?;
                break;
            }
            let ow = orient2d(pu, pv, self.pts[w]);
            if ow == 0.0 {
                return Ok(Some(w));
            } else if ow < 0.0 {
                push(&mut lower, w)?;
                r = w;
            } else {
                push(&mut upper, w)?;
                l = w;
            }
        }

        // Build `poly_upper = [u, v, l_last, ..., l_0]` (CCW, with first edge `u -> v`).
        let mut poly_upper: Vec<usize> = Vec::new();
        reserve(&mut poly_upper, upper.len() + 1)?;
        poly_upper.push(u);
        poly_upper.push(v);
        for &lv in upper[1..].iter().rev() {
            poly_upper.push(lv);
        }

        let mut new_tris: Vec<[usize; 3]> = Vec::new();
        ear_clip_polygon(&lower, &self.pts, &mut new_tris)?;
        ear_clip_polygon(&poly_upper, &self.pts, &mut new_tris)?;

        // Compact remaining triangles + append new cavity triangles deterministically.
        let kept_count = in_cavity.iter().filter(|&&c| !c).count();
        let mut next_tris: Vec<[usize; 3]> = Vec::new();
        reserve(&mut next_tris, kept_count + new_tris.len())?;
        for (ti, &t) in self.tris.iter().enumerate() {
            if !in_cavity[ti] {
                next_tris.push(t);
            }
        }
        for &t in &new_tris {
            next_tris.push(t);
        }
        self.tris = next_tris;

        // Legalize newly created interior diagonals inside the cavity (preserving all constraints).
        let mut flip_stack: Vec<(usize, usize)> = Vec::new();
        reserve(&mut flip_stack, new_tris.len() * 3)?;
        for &[a, b, c] in &new_tris {
            for (x, y) in [(a, b), (b, c), (c, a)] {
                let k = edge_key_local(x, y);
                if k != edge_key_local(u, v) {
                    flip_stack.push(k);
                }
            }
        }
        flip_stack.sort_unstable();
        flip_stack.dedup();
        self.legalize_edges(&mut flip_stack, true)?;

        Ok(None)
    }

    fn emit(&self) -> Result<Vec<[ArrVertexId; 3]>> {
        let mut out: Vec<[ArrVertexId; 3]> = Vec::new();
        reserve(&mut out, self.tris.len())?;
        for &[a, b, c] in &self.tris {
            debug_assert!(a != b && b != c && c != a);
            debug_assert!(orient2d(self.pts[a], self.pts[b], self.pts[c]) > 0.0);
            out.push([self.verts[a], self.verts[b], self.verts[c]]);
        }
        Ok(out)
    }
}

fn compare_along_collinear(
    pu: Coord<f64>,
    pv: Coord<f64>,
    px: Coord<f64>,
    py: Coord<f64>,
    vx: ArrVertexId,
    vy: ArrVertexId,
) -> Ordering {
    let ord = if pu.x < pv.x {
        px.x.total_cmp(&py.x)
    } else if pu.x > pv.x {
        py.x.total_cmp(&px.x)
    } else if pu.y < pv.y {
        px.y.total_cmp(&py.y)
    } else {
        py.y.total_cmp(&px.y)
    };
    ord.then_with(|| vx.cmp(&vy))
}

fn replace_seam_segment(
    seam_edges: &mut Vec<(ArrVertexId, ArrVertexId)>,
    old_u: ArrVertexId,
    old_v: ArrVertexId,
    chain: &[ArrVertexId],
) -> Result<bool> {
    if chain.len() < 3 {
        return Ok(false);
    }
    let old_key = edge_key_arr(old_u, old_v);
    if !seam_edges.contains(&old_key) {
        return Ok(false);
    }
    seam_edges.retain(|&e| e != old_key);
    reserve(seam_edges, chain.len() - 1)?;
    for w in chain.windows(2) {
        if w[0] != w[1] {
            seam_edges.push(edge_key_arr(w[0], w[1]));
        }
    }
    Ok(true)
}

fn tri_quality(a: Coord<f64>, b: Coord<f64>, c: Coord<f64>) -> f64 {
    let twice_area = orient2d(a, b, c);
    let denom = dist2(a, b) + dist2(b, c) + dist2(c, a);
    if denom > 0.0 { twice_area / denom } else { 0.0 }
}

fn polygon_twice_area(ring: &[usize], pts: &[Coord<f64>]) -> f64 {
    let Some(&first) = ring.first() else {
        return 0.0;
    };
    let o = pts[first];
    let mut sum = 0.0;
    for i in 0..ring.len() {
        let p = pts[ring[i]];
        let q = pts[ring[(i + 1) % ring.len()]];
        sum += (p.x - o.x) * (q.y - o.y) - (q.x - o.x) * (p.y - o.y);
    }
    sum
}

/// Exact `orient2d` ear-clipping for simple and weakly-simple (pinched / slit)
/// counter-clockwise polygons. Preserves every `180°` collinear boundary vertex
/// without dropping sliver triangles.
fn ear_clip_polygon(poly: &[usize], pts: &[Coord<f64>], out: &mut Vec<[usize; 3]>) -> Result<()> {
    if poly.len() < 3 {
        return Err(OpsError::InvalidInput("boolean cavity polygon too small"));
    }
    reserve(out, poly.len() - 2)?;

    let mut stack: Vec<Vec<usize>> = Vec::new();
    let mut initial: Vec<usize> = Vec::new();
    reserve(&mut initial, poly.len())?;
    initial.extend_from_slice(poly);
    push(&mut stack, initial)?;

    while let Some(mut ring) = stack.pop() {
        // Strip consecutive equal indices if any.
        let mut write = 0usize;
        for i in 0..ring.len() {
            if write == 0 || ring[i] != ring[write - 1] {
                ring[write] = ring[i];
                write += 1;
            }
        }
        ring.truncate(write);
        while ring.len() > 1 && ring.first() == ring.last() {
            ring.pop();
        }
        if ring.len() < 3 {
            continue;
        }

        // If two positive-area sub-loops touch at a repeated pinch vertex, split
        // them into two independent positive-area rings (keeping slits and CW
        // islands intact in the parent ring).
        if let Some((m1, m2)) = find_positive_pinch_split(&ring, pts) {
            let mut sub1: Vec<usize> = Vec::new();
            reserve(&mut sub1, m2 - m1)?;
            sub1.extend_from_slice(&ring[m1..m2]);

            let mut sub2: Vec<usize> = Vec::new();
            reserve(&mut sub2, ring.len() - (m2 - m1))?;
            sub2.extend_from_slice(&ring[..m1]);
            sub2.extend_from_slice(&ring[m2..]);

            push(&mut stack, sub2)?;
            push(&mut stack, sub1)?;
            continue;
        }

        while ring.len() > 3 {
            let mut best_ear: Option<(usize, f64)> = None;
            let len = ring.len();
            for b in 0..len {
                if !is_valid_ear(&ring, b, pts) {
                    continue;
                }
                let a = (b + len - 1) % len;
                let c = (b + 1) % len;
                let q = tri_quality(pts[ring[a]], pts[ring[b]], pts[ring[c]]);
                if best_ear.is_none_or(|(_, bq)| q > bq) {
                    best_ear = Some((b, q));
                }
            }
            let Some((b, _)) = best_ear else {
                return Err(OpsError::InvalidInput(
                    "boolean cavity triangulation failed",
                ));
            };
            let len = ring.len();
            let va = ring[(b + len - 1) % len];
            let vb = ring[b];
            let vc = ring[(b + 1) % len];
            push(out, [va, vb, vc])?;
            ring.remove(b);
        }

        let [va, vb, vc] = [ring[0], ring[1], ring[2]];
        if va == vb || vb == vc || vc == va || orient2d(pts[va], pts[vb], pts[vc]) <= 0.0 {
            return Err(OpsError::InvalidInput(
                "boolean cavity final triangle degenerate",
            ));
        }
        push(out, [va, vb, vc])?;
    }

    Ok(())
}

fn find_positive_pinch_split(ring: &[usize], pts: &[Coord<f64>]) -> Option<(usize, usize)> {
    let n = ring.len();
    if n < 6 {
        return None;
    }
    for m1 in 0..n {
        for m2 in (m1 + 3)..n {
            if n - (m2 - m1) < 3 {
                continue;
            }
            if ring[m1] != ring[m2] {
                continue;
            }
            if polygon_twice_area(&ring[m1..m2], pts) > 0.0 {
                // Verify complement also has positive signed area.
                let o = pts[ring[0]];
                let mut rem_area = 0.0;
                let rem_len = n - (m2 - m1);
                let at = |k: usize| {
                    if k < m1 { ring[k] } else { ring[k + (m2 - m1)] }
                };
                for k in 0..rem_len {
                    let p = pts[at(k)];
                    let q = pts[at((k + 1) % rem_len)];
                    rem_area += (p.x - o.x) * (q.y - o.y) - (q.x - o.x) * (p.y - o.y);
                }
                if rem_area > 0.0 {
                    return Some((m1, m2));
                }
            }
        }
    }
    None
}

fn is_valid_ear(ring: &[usize], b: usize, pts: &[Coord<f64>]) -> bool {
    let len = ring.len();
    let a = (b + len - 1) % len;
    let c = (b + 1) % len;
    let (va, vb, vc) = (ring[a], ring[b], ring[c]);
    if va == vb || vb == vc || vc == va {
        return false;
    }
    let (pa, pb, pc) = (pts[va], pts[vb], pts[vc]);
    if orient2d(pa, pb, pc) <= 0.0 {
        return false;
    }

    // Locally inside check at `a` (for convex or 180-degree straight `a`; reflex
    // and 360-degree hairpin `ap == vb` already satisfy `orient2d(pa, pb, pc) > 0`).
    let ap = ring[(a + len - 1) % len];
    if ap != vb && orient2d(pts[ap], pa, pb) >= 0.0 && vc != ap && orient2d(pts[ap], pa, pc) <= 0.0
    {
        return false;
    }

    // Locally inside check at `c`.
    let cn = ring[(c + 1) % len];
    if cn != vb && orient2d(pb, pc, pts[cn]) >= 0.0 && va != cn && orient2d(pc, pts[cn], pa) <= 0.0
    {
        return false;
    }

    // Point-in-triangle check for all distinct vertices in the ring.
    for &vp in ring {
        if vp == va || vp == vb || vp == vc {
            continue;
        }
        if in_closed_ccw_tri(pa, pb, pc, pts[vp]) {
            return false;
        }
    }

    // Edge-crossing and duplicate half-edge check against the new diagonal `va -> vc`.
    for q in 0..len {
        let vp = ring[q];
        let vq = ring[(q + 1) % len];
        if vp == va && vq == vc {
            return false;
        }
        if vp != va
            && vp != vc
            && vq != va
            && vq != vc
            && segments_intersect_2d(pa, pc, pts[vp], pts[vq])
        {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn tri_normal_and_area(positions: &[[f32; 3]], [a, b, c]: [ArrVertexId; 3]) -> ([f64; 3], f64) {
        let pa = positions[a as usize].map(f64::from);
        let pb = positions[b as usize].map(f64::from);
        let pc = positions[c as usize].map(f64::from);
        let n = cross3(sub3(pb, pa), sub3(pc, pa));
        let len = dot3(n, n).sqrt();
        ([n[0] / len, n[1] / len, n[2] / len], 0.5 * len)
    }

    fn edge_key(u: ArrVertexId, v: ArrVertexId) -> (ArrVertexId, ArrVertexId) {
        (u.min(v), u.max(v))
    }

    fn assert_valid_subtriangulation(
        corners: [ArrVertexId; 3],
        edge_splits: &[Vec<ArrVertexId>; 3],
        positions: &[[f32; 3]],
        tris: &[[ArrVertexId; 3]],
        expected_constraints: &[(ArrVertexId, ArrVertexId)],
    ) {
        assert!(!tris.is_empty(), "sub-triangulation must not be empty");
        let (parent_n, parent_area) = tri_normal_and_area(positions, corners);

        let mut ring = Vec::new();
        for i in 0..3 {
            ring.push(corners[i]);
            for &v in &edge_splits[i] {
                ring.push(v);
            }
        }
        let mut outer_directed = Vec::new();
        for i in 0..ring.len() {
            outer_directed.push((ring[i], ring[(i + 1) % ring.len()]));
        }

        let mut half_edges: BTreeMap<(ArrVertexId, ArrVertexId), usize> = BTreeMap::new();
        let mut undirected_edges: BTreeMap<(ArrVertexId, ArrVertexId), usize> = BTreeMap::new();
        let mut total_area = 0.0;
        let mut total_projected_area = 0.0;

        for &[a, b, c] in tris {
            assert!(
                a != b && b != c && c != a,
                "degenerate repeated-vertex sub-triangle [{a}, {b}, {c}]"
            );
            let pa = positions[a as usize].map(f64::from);
            let pb = positions[b as usize].map(f64::from);
            let pc = positions[c as usize].map(f64::from);
            let n = cross3(sub3(pb, pa), sub3(pc, pa));
            let signed_twice_area = dot3(n, parent_n);
            assert!(
                signed_twice_area > 0.0,
                "sub-triangle [{a}, {b}, {c}] has non-positive or flipped winding: {signed_twice_area}"
            );
            total_projected_area += 0.5 * signed_twice_area;
            total_area += 0.5 * dot3(n, n).sqrt();

            for (u, v) in [(a, b), (b, c), (c, a)] {
                *half_edges.entry((u, v)).or_insert(0) += 1;
                *undirected_edges.entry(edge_key(u, v)).or_insert(0) += 1;
            }
        }

        let max_coord = positions
            .iter()
            .flatten()
            .map(|&c| f64::from(c).abs())
            .fold(1.0_f64, f64::max);
        let area_tol = (1e-5 * parent_area.max(1.0)).max(1e-8 * max_coord);
        assert!(
            (total_projected_area - parent_area).abs() <= 1e-5 * parent_area.max(1.0),
            "projected area mismatch: got {total_projected_area}, expected {parent_area}"
        );
        assert!(
            (total_area - parent_area).abs() <= area_tol,
            "area mismatch: got {total_area}, expected {parent_area} (tol {area_tol})"
        );

        for &(u, v) in &outer_directed {
            assert_eq!(
                half_edges.get(&(u, v)).copied().unwrap_or(0),
                1,
                "outer boundary half-edge ({u} -> {v}) must appear once"
            );
            assert_eq!(
                half_edges.get(&(v, u)).copied().unwrap_or(0),
                0,
                "reverse of outer boundary half-edge ({v} -> {u}) must not appear"
            );
        }

        for (&(u, v), &count) in &half_edges {
            assert_eq!(count, 1, "duplicate directed half-edge ({u} -> {v})");
            if !outer_directed.contains(&(u, v)) {
                assert_eq!(
                    half_edges.get(&(v, u)).copied().unwrap_or(0),
                    1,
                    "unmatched interior half-edge ({u} -> {v})"
                );
            }
        }

        for &(u, v) in expected_constraints {
            assert!(
                undirected_edges.contains_key(&edge_key(u, v)),
                "missing constraint edge ({u}, {v}) in output triangles"
            );
        }
    }

    #[test]
    fn single_interior_point_splits_into_three_ccw_triangles() {
        let mut positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.0],
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
        ];
        let mut seam_edges = Vec::new();
        let constraints = TriConstraints {
            edge_splits: [Vec::new(), Vec::new(), Vec::new()],
            inner_verts: vec![3],
            segments: Vec::new(),
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        assert_eq!(tris.len(), 3);
        assert_valid_subtriangulation([0, 1, 2], &constraints.edge_splits, &positions, &tris, &[]);
        let (_, area0) = tri_normal_and_area(&positions, tris[0]);
        let (_, area1) = tri_normal_and_area(&positions, tris[1]);
        let (_, area2) = tri_normal_and_area(&positions, tris[2]);
        assert!((area0 + area1 + area2 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn polyline_chain_across_triangle_preserves_all_constraints_and_boundary() {
        // 4-segment chain from split on edge 0 (v3) through 3 interior points (v5, v6, v7)
        // to split on edge 1 (v4), zig-zagging across interior diagonals.
        let mut positions = vec![
            [0.0, 0.0, 0.0],   // 0: corner 0
            [1.0, 0.0, 0.0],   // 1: corner 1
            [0.0, 1.0, 0.0],   // 2: corner 2
            [0.4, 0.0, 0.0],   // 3: split on edge 0 (0 -> 1)
            [0.4, 0.6, 0.0],   // 4: split on edge 1 (1 -> 2)
            [0.15, 0.15, 0.0], // 5: inner 1
            [0.55, 0.22, 0.0], // 6: inner 2
            [0.20, 0.45, 0.0], // 7: inner 3
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let chain = vec![(3, 5), (5, 6), (6, 7), (7, 4)];
        let mut seam_edges: Vec<_> = chain.iter().map(|&(u, v)| edge_key(u, v)).collect();
        seam_edges.sort_unstable();
        let constraints = TriConstraints {
            edge_splits: [vec![3], vec![4], Vec::new()],
            inner_verts: vec![5, 6, 7],
            segments: chain.clone(),
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        // Euler check: B = 5 boundary vertices, I = 3 interior vertices => T = B + 2*I - 2 = 9.
        assert_eq!(tris.len(), 9);
        assert_valid_subtriangulation(
            [0, 1, 2],
            &constraints.edge_splits,
            &positions,
            &tris,
            &chain,
        );
    }

    #[test]
    fn crossing_segments_insert_intersection_and_recover_both() {
        let mut positions = vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [0.0, 1.0, 0.0], // 2
            [0.1, 0.1, 0.0], // 3
            [0.6, 0.3, 0.0], // 4
            [0.1, 0.4, 0.0], // 5
            [0.5, 0.1, 0.0], // 6
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let mut seam_edges = vec![edge_key(3, 4), edge_key(5, 6)];
        seam_edges.sort_unstable();
        let constraints = TriConstraints {
            edge_splits: [Vec::new(), Vec::new(), Vec::new()],
            inner_verts: vec![3, 4, 5, 6],
            segments: vec![(3, 4), (5, 6)],
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        assert_eq!(
            positions.len(),
            8,
            "must allocate 1 Fresh intersection vertex"
        );
        assert_eq!(origins.len(), 8);
        assert_eq!(origins[7], ArrOrigin::Fresh);

        let new_v = 7;
        let expected_halves = vec![
            edge_key(3, new_v),
            edge_key(new_v, 4),
            edge_key(5, new_v),
            edge_key(new_v, 6),
        ];
        let mut expected_seam = expected_halves.clone();
        expected_seam.sort_unstable();
        assert_eq!(seam_edges, expected_seam);

        assert_valid_subtriangulation(
            [0, 1, 2],
            &constraints.edge_splits,
            &positions,
            &tris,
            &expected_halves,
        );
    }

    #[test]
    fn near_collinear_edge_and_inner_points_leave_no_holes() {
        // Interior point 1e-6 from a split boundary edge (edge 0: y = 0).
        let mut positions = vec![
            [0.0, 0.0, 0.0],    // 0
            [1.0, 0.0, 0.0],    // 1
            [0.0, 1.0, 0.0],    // 2
            [0.3, 0.0, 0.0],    // 3: split on edge 0
            [0.7, 0.0, 0.0],    // 4: split on edge 0
            [0.5, 1.0e-6, 0.0], // 5: interior sliver point 1e-6 above edge 0
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let mut seam_edges = vec![edge_key(3, 5), edge_key(5, 4)];
        seam_edges.sort_unstable();
        let constraints = TriConstraints {
            edge_splits: [vec![3, 4], Vec::new(), Vec::new()],
            inner_verts: vec![5],
            segments: vec![(3, 5), (5, 4)],
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        // B = 5, I = 1 => T = 5 + 2 - 2 = 5.
        assert_eq!(tris.len(), 5);
        assert_valid_subtriangulation(
            [0, 1, 2],
            &constraints.edge_splits,
            &positions,
            &tris,
            &[(3, 5), (5, 4)],
        );
    }

    #[test]
    fn tilted_3d_negative_dominant_axis_preserves_parent_3d_winding() {
        // Dominant normal is along -X: corners in YZ plane with negative X normal.
        let mut positions = vec![
            [2.0, 0.0, 0.0],   // 0
            [2.0, 0.0, 1.0],   // 1
            [2.0, 1.0, 0.0],   // 2 -> (p1 - p0) x (p2 - p0) = (0,0,1) x (0,1,0) = (-1, 0, 0)
            [2.0, 0.0, 0.5],   // 3: split on edge 0
            [2.0, 0.25, 0.25], // 4: inner
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let mut seam_edges = vec![edge_key(3, 4), edge_key(4, 2)];
        seam_edges.sort_unstable();
        let constraints = TriConstraints {
            edge_splits: [vec![3], Vec::new(), Vec::new()],
            inner_verts: vec![4],
            segments: vec![(3, 4), (4, 2)],
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        assert_valid_subtriangulation(
            [0, 1, 2],
            &constraints.edge_splits,
            &positions,
            &tris,
            &[(3, 4), (4, 2)],
        );
    }

    #[test]
    fn collinear_points_and_duplicate_segments_split_cleanly() {
        // Segment (3 -> 6) passes collinearly through inner points 4 and 5,
        // plus duplicate and overlapping sub-segments in input.
        let mut positions = vec![
            [0.0, 0.0, 0.0], // 0
            [2.0, 0.0, 0.0], // 1
            [0.0, 2.0, 0.0], // 2
            [0.2, 0.2, 0.0], // 3
            [0.5, 0.5, 0.0], // 4 (collinear on 3..6)
            [0.8, 0.8, 0.0], // 5 (collinear on 3..6)
            [1.0, 1.0, 0.0], // 6: split on edge 1 (1 -> 2)
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let mut seam_edges = vec![edge_key(3, 6)];
        let constraints = TriConstraints {
            edge_splits: [Vec::new(), vec![6], Vec::new()],
            inner_verts: vec![3, 4, 5],
            segments: vec![(3, 6), (6, 3), (3, 5)],
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        let expected = vec![edge_key(3, 4), edge_key(4, 5), edge_key(5, 6)];
        let mut expected_seam = expected.clone();
        expected_seam.sort_unstable();
        assert_eq!(seam_edges, expected_seam);

        assert_valid_subtriangulation(
            [0, 1, 2],
            &constraints.edge_splits,
            &positions,
            &tris,
            &expected,
        );
    }

    #[test]
    fn constraint_segment_matching_boundary_sub_edge_is_handled_gracefully() {
        let mut positions = vec![
            [0.0, 0.0, 0.0],   // 0
            [1.0, 0.0, 0.0],   // 1
            [0.0, 1.0, 0.0],   // 2
            [0.3, 0.0, 0.0],   // 3: split on edge 0
            [0.7, 0.0, 0.0],   // 4: split on edge 0
            [0.25, 0.25, 0.0], // 5: inner
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let mut seam_edges = vec![edge_key(0, 1), edge_key(3, 4), edge_key(4, 5)];
        seam_edges.sort_unstable();
        // Includes both a direct boundary sub-edge (3, 4) and full parent edge (0, 1)!
        let constraints = TriConstraints {
            edge_splits: [vec![3, 4], Vec::new(), Vec::new()],
            inner_verts: vec![5],
            segments: vec![(3, 4), (0, 1), (4, 5)],
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        assert_valid_subtriangulation(
            [0, 1, 2],
            &constraints.edge_splits,
            &positions,
            &tris,
            &[(0, 3), (3, 4), (4, 1), (4, 5)],
        );
    }

    #[test]
    fn cavity_with_reflex_corners_and_slit_retriangulates_manifold() {
        // Large parent triangle enclosing u=(0,0) -> v=(10,0), with reflex/slit points
        // x=(5,-4), z=(5,-1), l1=(2,2), l2=(8,2), plus extra reflex blockers.
        let mut positions = vec![
            [-5.0, -10.0, 0.0], // 0: corner 0
            [20.0, -10.0, 0.0], // 1: corner 1
            [5.0, 15.0, 0.0],   // 2: corner 2
            [0.0, 0.0, 0.0],    // 3: u
            [10.0, 0.0, 0.0],   // 4: v
            [5.0, -4.0, 0.0],   // 5: x
            [5.0, -1.0, 0.0],   // 6: z
            [2.0, 2.0, 0.0],    // 7: l1
            [8.0, 2.0, 0.0],    // 8: l2
            [3.0, -0.2, 0.0],   // 9: reflex near u..v
            [7.0, 0.2, 0.0],    // 10: reflex near u..v
        ];
        let mut origins = vec![ArrOrigin::Fresh; positions.len()];
        origins[0] = ArrOrigin::LiveA(0);
        origins[1] = ArrOrigin::LiveA(1);
        origins[2] = ArrOrigin::LiveA(2);
        let mut seam_edges = vec![edge_key(3, 4)];
        let constraints = TriConstraints {
            edge_splits: [Vec::new(), Vec::new(), Vec::new()],
            inner_verts: vec![3, 4, 5, 6, 7, 8, 9, 10],
            segments: vec![(5, 6), (3, 4)],
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        assert_valid_subtriangulation(
            [0, 1, 2],
            &constraints.edge_splits,
            &positions,
            &tris,
            &[(5, 6), (3, 4)],
        );
    }

    #[test]
    fn interior_point_on_interior_diagonal_splits_cleanly() {
        // Point 3 at (0.2, 0.2) creates interior diagonals (0, 3), (1, 3), (2, 3);
        // Point 4 at (0.1, 0.1) lies strictly on the interior edge (0, 3)!
        let mut positions = vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [0.0, 1.0, 0.0], // 2
            [0.2, 0.2, 0.0], // 3
            [0.1, 0.1, 0.0], // 4 (exact midpoint of 0..3!)
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let mut seam_edges = Vec::new();
        let constraints = TriConstraints {
            edge_splits: [Vec::new(), Vec::new(), Vec::new()],
            inner_verts: vec![3, 4],
            segments: Vec::new(),
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        // B = 3, I = 2 => T = 3 + 4 - 2 = 5.
        assert_eq!(tris.len(), 5);
        assert_valid_subtriangulation([0, 1, 2], &constraints.edge_splits, &positions, &tris, &[]);
    }

    #[test]
    fn oom_fault_injection_propagates_cleanly() {
        let base_positions = vec![
            [0.0, 0.0, 0.0],  // 0
            [1.0, 0.0, 0.0],  // 1
            [0.0, 1.0, 0.0],  // 2
            [0.4, 0.0, 0.0],  // 3: split on edge 0
            [0.1, 0.1, 0.0],  // 4
            [0.6, 0.3, 0.0],  // 5
            [0.1, 0.4, 0.0],  // 6
            [0.5, 0.1, 0.0],  // 7
            [0.1, 0.6, 0.0],  // 8: multi-collinear start
            [0.15, 0.6, 0.0], // 9: collinear mid 1
            [0.2, 0.6, 0.0],  // 10: collinear mid 2
            [0.25, 0.6, 0.0], // 11: multi-collinear end
        ];
        let base_origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let constraints = TriConstraints {
            edge_splits: [vec![3], Vec::new(), Vec::new()],
            inner_verts: vec![4, 5, 6, 7, 8, 9, 10, 11],
            segments: vec![(4, 5), (6, 7), (8, 11)],
        };

        let mut succeeded = false;
        for step in 0..256 {
            let mut positions = base_positions.clone();
            let mut origins = base_origins.clone();
            let mut seam_edges = vec![edge_key(4, 5), edge_key(6, 7), edge_key(8, 11)];
            crate::fail_allocations_after(Some(step));
            let res = triangulate_constrained(
                [0, 1, 2],
                &constraints,
                &mut positions,
                &mut origins,
                &mut seam_edges,
            );
            crate::fail_allocations_after(None);
            match res {
                Ok(tris) => {
                    assert!(step > 0, "must allocate at least once");
                    assert!(!tris.is_empty());
                    succeeded = true;
                    break;
                }
                Err(e) => {
                    assert_eq!(e, OpsError::Store(StoreError::OutOfMemory));
                }
            }
        }
        assert!(succeeded, "triangulate_constrained must eventually succeed");
    }

    #[test]
    fn production_cdt_uses_no_untracked_allocating_sorts() {
        let prod_src = include_str!("cdt.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production section must exist");
        assert!(
            !prod_src.contains(".sort(")
                && !prod_src.contains(".sort_by(")
                && !prod_src.contains(".sort_by_key(")
                && !prod_src.contains("vec!["),
            "production cdt.rs must not use allocating stdlib sorts or vec! macros"
        );
    }

    #[test]
    fn tilted_boundary_split_projecting_to_end_corner_is_not_teleported() {
        // 45-degree tilted triangle where dominant normal axis is X, so Proj2D projects onto (Z, Y).
        // Split 3 lies within < 0.5 ULP of corner 1 on Z (`11.1`), differing on X (`1.0 - 2e-7`).
        let mut positions = vec![
            [0.1, 0.0, 10.0],        // 0: corner 0
            [1.0, 0.0, 11.1],        // 1: corner 1
            [0.1, 1.0, 10.0],        // 2: corner 2
            [1.0 - 2e-7, 0.0, 11.1], // 3: split on edge 0 projecting to c1 in (Z, Y)
            [0.64, 0.02, 10.66],     // 4: inner left
            [0.82, 0.005, 10.88],    // 5: inner right
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let mut seam_edges = vec![edge_key(3, 5), edge_key(5, 4)];
        seam_edges.sort_unstable();
        let constraints = TriConstraints {
            edge_splits: [vec![3], Vec::new(), Vec::new()],
            inner_verts: vec![4, 5],
            segments: vec![(3, 5), (5, 4)],
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        assert_eq!(positions.len(), 6, "must not fabricate false crossings");
        assert_valid_subtriangulation(
            [0, 1, 2],
            &constraints.edge_splits,
            &positions,
            &tris,
            &[(3, 5), (5, 4)],
        );
    }

    #[test]
    fn tilted_end_tied_split_with_nearby_preceding_split_does_not_underflow_f64() {
        // Split 3 at z = 11.0995 precedes split 4 tying with c1 at z = 11.1 (`dz = -0.0005`),
        // where fixed `1e-12 * (-0.0005) = -5e-16` underflows `0.5 * ULP_f64(11.1)`.
        let mut positions = vec![
            [0.1, 0.0, 10.0],            // 0: corner 0
            [1.0, 0.0, 11.1],            // 1: corner 1
            [0.1, 1.0, 10.0],            // 2: corner 2
            [0.999_590_9, 0.0, 11.0995], // 3: split ~500 ULP_f32 before c1
            [1.0 - 2e-7, 0.0, 11.1],     // 4: split projecting to c1 in (Z, Y)
            [0.64, 0.02, 10.66],         // 5: inner left
            [0.82, 0.005, 10.88],        // 6: inner right
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let mut seam_edges = vec![edge_key(4, 6), edge_key(6, 5)];
        seam_edges.sort_unstable();
        let constraints = TriConstraints {
            edge_splits: [vec![3, 4], Vec::new(), Vec::new()],
            inner_verts: vec![5, 6],
            segments: vec![(4, 6), (6, 5)],
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        assert_eq!(positions.len(), 7, "must not fabricate false crossings");
        assert_valid_subtriangulation(
            [0, 1, 2],
            &constraints.edge_splits,
            &positions,
            &tris,
            &[(4, 6), (6, 5)],
        );
    }

    #[test]
    fn tilted_start_tied_split_with_nearby_following_split_does_not_underflow_f64() {
        // Split 3 ties with c0 at z = 2.0, followed by split 4 at z = 2.0001 (`dz = +1e-4`),
        // where fixed `1e-12 * 1e-4 = 1e-16` underflows `0.5 * ULP_f64(2.0)`.
        let mut positions = vec![
            [0.1, 0.0, 2.0],             // 0: corner 0
            [1.0, 0.0, 3.1],             // 1: corner 1
            [0.1, 1.0, 2.0],             // 2: corner 2
            [0.1 + 2e-7, 0.0, 2.0],      // 3: split projecting to c0 in (Z, Y)
            [0.100_081_82, 0.0, 2.0001], // 4: split 1e-4 ahead on Z
            [0.55, 0.2, 2.55],           // 5: inner
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let mut seam_edges = vec![edge_key(3, 5)];
        let constraints = TriConstraints {
            edge_splits: [vec![3, 4], Vec::new(), Vec::new()],
            inner_verts: vec![5],
            segments: vec![(3, 5)],
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        assert_eq!(positions.len(), 6, "must not fabricate false crossings");
        assert_valid_subtriangulation(
            [0, 1, 2],
            &constraints.edge_splits,
            &positions,
            &tris,
            &[(3, 5)],
        );
    }

    #[test]
    fn tilted_boundary_split_at_large_world_coordinates_does_not_underflow_f64() {
        // Translated by +20_000.0 on Z where `ULP_f64(20_011.1) ≈ 3.64e-12`, so fixed
        // `1e-12 * (-1.1)` underflows `0.5 * ULP_f64` even with a single split.
        let mut positions = vec![
            [0.1, 0.0, 20_010.0],        // 0
            [1.0, 0.0, 20_011.1],        // 1
            [0.1, 1.0, 20_010.0],        // 2
            [1.0 - 2e-7, 0.0, 20_011.1], // 3: split projecting to c1 in (Z, Y)
            [0.64, 0.02, 20_010.66],     // 4: inner left
            [0.82, 0.005, 20_010.88],    // 5: inner right
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let mut seam_edges = vec![edge_key(3, 5), edge_key(5, 4)];
        seam_edges.sort_unstable();
        let constraints = TriConstraints {
            edge_splits: [vec![3], Vec::new(), Vec::new()],
            inner_verts: vec![4, 5],
            segments: vec![(3, 5), (5, 4)],
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        assert_eq!(positions.len(), 6, "must not fabricate false crossings");
        assert_valid_subtriangulation(
            [0, 1, 2],
            &constraints.edge_splits,
            &positions,
            &tris,
            &[(3, 5), (5, 4)],
        );
    }

    #[test]
    fn grazing_segment_across_outward_then_inward_diagonal_splits_recovers() {
        // Hypotenuse 1=(1,0) -> 2=(0,1) has outward split 3=(0.8, 0.2 + 5e-8) and inward
        // split 4=(0.5, 0.5 - 2e-7). Segment (5, 2) from 5=(0.85, 0.15 - 4e-8) crosses 3->4
        // closer to outward 3 than inward 4; must snap/split cleanly without leaving an
        // unrecoverable exterior chord (3, 2).
        let mut positions = vec![
            [0.0, 0.0, 0.0],           // 0
            [1.0, 0.0, 0.0],           // 1
            [0.0, 1.0, 0.0],           // 2
            [0.8, 0.200_000_05, 0.0],  // 3: outward split (x + y > 1)
            [0.5, 0.499_999_8, 0.0],   // 4: inward reflex split (x + y < 1)
            [0.85, 0.149_999_96, 0.0], // 5: interior near 3
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let mut seam_edges = vec![edge_key(5, 2)];
        let constraints = TriConstraints {
            edge_splits: [Vec::new(), vec![3, 4], Vec::new()],
            inner_verts: vec![5],
            segments: vec![(5, 2)],
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        // B = 5, I = 1 => T = 5 + 2 - 2 = 5.
        assert_eq!(tris.len(), 5);
        assert_valid_subtriangulation(
            [0, 1, 2],
            &constraints.edge_splits,
            &positions,
            &tris,
            &seam_edges,
        );
    }

    fn assert_manifold_euler_topology(
        corners: [ArrVertexId; 3],
        edge_splits: &[Vec<ArrVertexId>; 3],
        expected_tris: usize,
        tris: &[[ArrVertexId; 3]],
        expected_constraints: &[(ArrVertexId, ArrVertexId)],
    ) {
        assert_eq!(tris.len(), expected_tris);
        let mut ring = Vec::new();
        for i in 0..3 {
            ring.push(corners[i]);
            for &v in &edge_splits[i] {
                ring.push(v);
            }
        }
        let mut outer_directed = Vec::new();
        for i in 0..ring.len() {
            outer_directed.push((ring[i], ring[(i + 1) % ring.len()]));
        }
        let mut half_edges: BTreeMap<(ArrVertexId, ArrVertexId), usize> = BTreeMap::new();
        let mut undirected_edges: BTreeMap<(ArrVertexId, ArrVertexId), usize> = BTreeMap::new();
        for &[a, b, c] in tris {
            assert!(a != b && b != c && c != a);
            for (u, v) in [(a, b), (b, c), (c, a)] {
                *half_edges.entry((u, v)).or_insert(0) += 1;
                *undirected_edges.entry(edge_key(u, v)).or_insert(0) += 1;
            }
        }
        for &(u, v) in &outer_directed {
            assert_eq!(half_edges.get(&(u, v)).copied().unwrap_or(0), 1);
            assert_eq!(half_edges.get(&(v, u)).copied().unwrap_or(0), 0);
        }
        for (&(u, v), &count) in &half_edges {
            assert_eq!(count, 1);
            if !outer_directed.contains(&(u, v)) {
                assert_eq!(half_edges.get(&(v, u)).copied().unwrap_or(0), 1);
            }
        }
        for &(u, v) in expected_constraints {
            assert!(undirected_edges.contains_key(&edge_key(u, v)));
        }
    }

    #[test]
    fn thin_sliver_with_two_f32_rounded_interior_vertices_does_not_collide() {
        // Sliver height is 2 * ULP_f32(1.0) (< 4 * f32::EPSILON); two interior vertices
        // rounded 1 ULP below y = 1.0 must clamp tangentially and not collide at best_centroid.
        let ulp = f32::EPSILON;
        let mut positions = vec![
            [0.0, 1.0, 0.0],             // 0
            [1.0, 1.0, 0.0],             // 1
            [0.5, 1.0 + 2.0 * ulp, 0.0], // 2
            [0.25, 1.0 - ulp, 0.0],      // 3: 1 ULP outside sliver at x = 0.25
            [0.75, 1.0 - ulp, 0.0],      // 4: 1 ULP outside sliver at x = 0.75
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let mut seam_edges = vec![edge_key(3, 4)];
        let constraints = TriConstraints {
            edge_splits: [Vec::new(), Vec::new(), Vec::new()],
            inner_verts: vec![3, 4],
            segments: vec![(3, 4)],
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        // B = 3, I = 2 => T = 3 + 4 - 2 = 5.
        assert_manifold_euler_topology([0, 1, 2], &constraints.edge_splits, 5, &tris, &[(3, 4)]);
    }

    #[test]
    fn two_interior_vertices_projecting_to_centroid_do_not_collide() {
        // Two interior vertices differ by 1 ULP along dropped normal axis Z and both project
        // onto the exact 2D centroid (1.0, 1.0).
        let mut positions = vec![
            [0.0, 0.0, 0.0],          // 0
            [3.0, 0.0, 0.0],          // 1
            [0.0, 3.0, 0.0],          // 2 -> centroid (1.0, 1.0)
            [1.0, 1.0, 0.0],          // 3 -> projects to (1.0, 1.0)
            [1.0, 1.0, f32::EPSILON], // 4 -> projects to (1.0, 1.0)
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let mut seam_edges = vec![edge_key(3, 4)];
        let constraints = TriConstraints {
            edge_splits: [Vec::new(), Vec::new(), Vec::new()],
            inner_verts: vec![3, 4],
            segments: vec![(3, 4)],
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        // B = 3, I = 2 => T = 3 + 4 - 2 = 5.
        assert_manifold_euler_topology([0, 1, 2], &constraints.edge_splits, 5, &tris, &[(3, 4)]);
    }

    #[test]
    fn f32_quantized_diagonal_edge_splits_and_boundary_segment() {
        // On tilted hypotenuse (1,0,0) -> (0,1,0):
        // (1/3, 2/3) in f32 sums to > 1.0 (+3e-8 outward), while (0.1, 0.9) sums to < 1.0 (-2e-8 inward).
        // A constraint segment (1, 2) along that split hypotenuse must resolve into (1, 3), (3, 4), (4, 2)
        // without failing cavity walk across the 1-ULP bent boundary.
        let mut positions = vec![
            [0.0, 0.0, 0.0],             // 0
            [1.0, 0.0, 0.0],             // 1
            [0.0, 1.0, 0.0],             // 2
            [2.0 / 3.0, 1.0 / 3.0, 0.0], // 3: f32 sum > 1.0
            [0.1, 0.9, 0.0],             // 4: f32 sum < 1.0
            [0.25, 0.25, 0.0],           // 5: inner
        ];
        let mut origins = vec![
            ArrOrigin::LiveA(0),
            ArrOrigin::LiveA(1),
            ArrOrigin::LiveA(2),
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
            ArrOrigin::Fresh,
        ];
        let mut seam_edges = vec![edge_key(1, 2), edge_key(3, 5)];
        seam_edges.sort_unstable();
        let constraints = TriConstraints {
            edge_splits: [Vec::new(), vec![3, 4], Vec::new()],
            inner_verts: vec![5],
            segments: vec![(1, 2), (3, 5)],
        };

        let tris = triangulate_constrained(
            [0, 1, 2],
            &constraints,
            &mut positions,
            &mut origins,
            &mut seam_edges,
        )
        .unwrap();

        let expected = vec![
            edge_key(1, 3),
            edge_key(3, 4),
            edge_key(4, 2),
            edge_key(3, 5),
        ];
        let mut expected_seam = expected.clone();
        expected_seam.sort_unstable();
        assert_eq!(seam_edges, expected_seam);

        assert_valid_subtriangulation(
            [0, 1, 2],
            &constraints.edge_splits,
            &positions,
            &tris,
            &expected,
        );
    }

    #[test]
    fn deterministic_random_arrangements_satisfy_euler_and_manifold_invariants() {
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next_f64 = || -> f64 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
        };

        for case_idx in 0..200 {
            // Random non-degenerate 3D parent triangle.
            let c0 = [
                (next_f64() * 4.0 - 2.0) as f32,
                (next_f64() * 4.0 - 2.0) as f32,
                (next_f64() * 4.0 - 2.0) as f32,
            ];
            let c1 = [
                c0[0] + (0.8 + next_f64()) as f32,
                c0[1] + (next_f64() * 0.4 - 0.2) as f32,
                c0[2] + (next_f64() * 0.6 - 0.3) as f32,
            ];
            let c2 = [
                c0[0] + (next_f64() * 0.4 - 0.2) as f32,
                c0[1] + (0.8 + next_f64()) as f32,
                c0[2] + (next_f64() * 0.6 - 0.3) as f32,
            ];

            let mut positions = vec![c0, c1, c2];
            let mut origins = vec![
                ArrOrigin::LiveA(0),
                ArrOrigin::LiveA(1),
                ArrOrigin::LiveA(2),
            ];
            let mut edge_splits: [Vec<ArrVertexId>; 3] = [Vec::new(), Vec::new(), Vec::new()];

            for (ei, splits_out) in edge_splits.iter_mut().enumerate() {
                let n_splits = (case_idx + ei) % 3;
                let mut ts = Vec::new();
                for k in 0..n_splits {
                    let t = 0.15 + 0.7 * ((k as f64 + 0.2 + 0.6 * next_f64()) / (n_splits as f64));
                    ts.push(t);
                }
                ts.sort_by(|a, b| a.total_cmp(b));
                let pa = positions[ei];
                let pb = positions[(ei + 1) % 3];
                for t in ts {
                    let id = positions.len() as ArrVertexId;
                    positions.push([
                        (f64::from(pa[0]) + t * f64::from(pb[0] - pa[0])) as f32,
                        (f64::from(pa[1]) + t * f64::from(pb[1] - pa[1])) as f32,
                        (f64::from(pa[2]) + t * f64::from(pb[2] - pa[2])) as f32,
                    ]);
                    origins.push(ArrOrigin::Fresh);
                    splits_out.push(id);
                }
            }

            let mut inner_verts = Vec::new();
            let n_inner = 1 + (case_idx % 6);
            for _ in 0..n_inner {
                let mut u = next_f64();
                let mut v = next_f64();
                if u + v > 1.0 {
                    u = 1.0 - u;
                    v = 1.0 - v;
                }
                u = 0.04 + 0.90 * u;
                v = 0.04 + 0.90 * v;
                let w = 1.0 - u - v;
                let id = positions.len() as ArrVertexId;
                positions.push([
                    (w * f64::from(c0[0]) + u * f64::from(c1[0]) + v * f64::from(c2[0])) as f32,
                    (w * f64::from(c0[1]) + u * f64::from(c1[1]) + v * f64::from(c2[1])) as f32,
                    (w * f64::from(c0[2]) + u * f64::from(c1[2]) + v * f64::from(c2[2])) as f32,
                ]);
                origins.push(ArrOrigin::Fresh);
                inner_verts.push(id);
            }

            let total_v = positions.len() as u32;
            let mut segments = Vec::new();
            let n_segs = case_idx % 5;
            for _ in 0..n_segs {
                let a = ((next_f64() * (total_v as f64)) as u32).min(total_v - 1);
                let b = ((next_f64() * (total_v as f64)) as u32).min(total_v - 1);
                if a != b {
                    segments.push((a, b));
                }
            }

            let constraints = TriConstraints {
                edge_splits,
                inner_verts,
                segments: segments.clone(),
            };
            let mut seam1: Vec<_> = segments.iter().map(|&(u, v)| edge_key(u, v)).collect();
            seam1.sort_unstable();
            seam1.dedup();

            let mut pos2 = positions.clone();
            let mut orig2 = origins.clone();
            let mut seam2 = seam1.clone();

            let tris1 = triangulate_constrained(
                [0, 1, 2],
                &constraints,
                &mut positions,
                &mut origins,
                &mut seam1,
            )
            .unwrap();
            let tris2 =
                triangulate_constrained([0, 1, 2], &constraints, &mut pos2, &mut orig2, &mut seam2)
                    .unwrap();

            // Bit-identical across runs.
            assert_eq!(tris1, tris2);
            assert_eq!(positions, pos2);
            assert_eq!(origins, orig2);
            assert_eq!(seam1, seam2);

            assert_valid_subtriangulation(
                [0, 1, 2],
                &constraints.edge_splits,
                &positions,
                &tris1,
                &seam1,
            );
        }
    }
}
