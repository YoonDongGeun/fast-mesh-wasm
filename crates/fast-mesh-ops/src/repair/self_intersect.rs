//! Self-intersection detection and resolution: hole removal+fill,
//! local cut-and-stitch remeshing, and volumetric Marching Cubes healing.

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_fields::{
    DEFAULT_MAX_VOXELS, FieldsError, GridSpec, SignData, SignMode, bvh_signed_grid,
    fh_unsigned_grid,
};
use fast_mesh_query::self_intersecting_pairs;
use fast_mesh_store::{Mesh, Transaction, TriangleId, UV_CHANNEL, VertexId};

use super::manifold::{ManifoldOptions, make_manifold};
use super::norm64;
use crate::cleanup::{DegenerateOptions, remove_degenerate, unused_vertices};
use crate::fill::{FillOptions, fill_holes};
use crate::marching_cubes::marching_cubes;
use crate::orient::orient;
use crate::topology::{UnionFind, VertexFans, cross, dot, point, sub};
use crate::{OpsError, Result, allocation_allowed, filled, push, reserve};

const GRID_PAD_VOXELS: usize = 3;
const AUTO_VOXEL_RES: usize = 48;

/// Strategy used by [`resolve_self_intersections()`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelfIntersectionMode {
    /// Detect self-intersecting pairs and return the count without editing the mesh.
    None,
    /// Remove intersecting triangles (optionally expanded by topological rings),
    /// make boundaries manifold, and patch resulting holes with [`fill_holes`].
    #[default]
    DeleteAndFill,
    /// Split colliding triangles along intersection segments, stitch shared seam
    /// vertices, cull enclosed internal overlap patches, and fall back locally to
    /// [`SelfIntersectionMode::DeleteAndFill`] on coplanar or degenerate pairs.
    CutAndStitch,
    /// Rebuild the surface from a generalized-winding signed distance grid via
    /// Marching Cubes (`iso = 0.0`). Drops [`UV_CHANNEL`](fast_mesh_store::UV_CHANNEL).
    VoxelHeal,
}

/// Options for [`resolve_self_intersections()`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelfIntersectOptions {
    /// Resolution strategy. Default [`SelfIntersectionMode::DeleteAndFill`].
    pub mode: SelfIntersectionMode,
    /// Number of 1-ring topological neighbors around intersecting faces to
    /// delete in [`SelfIntersectionMode::DeleteAndFill`] (and local fallback).
    /// Default `1`.
    pub ring_expansion: u32,
    /// Explicit voxel pitch in local units for [`SelfIntersectionMode::VoxelHeal`].
    /// `None` chooses automatically from the bounding box.
    pub voxel_size: Option<f32>,
    /// Maximum voxel count allowed for [`SelfIntersectionMode::VoxelHeal`].
    /// Default [`DEFAULT_MAX_VOXELS`].
    pub max_voxels: usize,
    /// Optional hole boundary edge cap passed to [`fill_holes`]. Default `None`.
    pub max_hole_edges: Option<usize>,
    /// Whether hole patches in `DeleteAndFill` are faired. Default `true`.
    pub fair: bool,
    /// Tolerance for welding shared cut vertices in `CutAndStitch`. Default `1e-5`.
    pub stitch_tolerance: f32,
}

impl Default for SelfIntersectOptions {
    fn default() -> Self {
        Self {
            mode: SelfIntersectionMode::DeleteAndFill,
            ring_expansion: 1,
            voxel_size: None,
            max_voxels: DEFAULT_MAX_VOXELS,
            max_hole_edges: None,
            fair: true,
            stitch_tolerance: 1e-5,
        }
    }
}

impl SelfIntersectOptions {
    /// A builder for [`SelfIntersectOptions`].
    pub fn builder() -> SelfIntersectOptionsBuilder {
        SelfIntersectOptionsBuilder::default()
    }
}

/// Builder for [`SelfIntersectOptions`].
#[derive(Debug, Clone, Default)]
pub struct SelfIntersectOptionsBuilder {
    options: SelfIntersectOptions,
}

impl SelfIntersectOptionsBuilder {
    /// Resolution mode.
    pub fn mode(mut self, mode: SelfIntersectionMode) -> Self {
        self.options.mode = mode;
        self
    }

    /// Ring expansion around intersecting triangles for deletion modes.
    pub fn ring_expansion(mut self, ring_expansion: u32) -> Self {
        self.options.ring_expansion = ring_expansion;
        self
    }

    /// Explicit voxel size for [`SelfIntersectionMode::VoxelHeal`].
    pub fn voxel_size(mut self, voxel_size: Option<f32>) -> Self {
        self.options.voxel_size = voxel_size;
        self
    }

    /// Maximum voxel budget for [`SelfIntersectionMode::VoxelHeal`].
    pub fn max_voxels(mut self, max_voxels: usize) -> Self {
        self.options.max_voxels = max_voxels;
        self
    }

    /// Maximum hole loop edge count to fill in [`SelfIntersectionMode::DeleteAndFill`].
    pub fn max_hole_edges(mut self, max_hole_edges: Option<usize>) -> Self {
        self.options.max_hole_edges = max_hole_edges;
        self
    }

    /// Whether to fair hole fill patches.
    pub fn fair(mut self, fair: bool) -> Self {
        self.options.fair = fair;
        self
    }

    /// Spatial stitch tolerance for [`SelfIntersectionMode::CutAndStitch`].
    pub fn stitch_tolerance(mut self, stitch_tolerance: f32) -> Self {
        self.options.stitch_tolerance = stitch_tolerance;
        self
    }

    /// The options as set.
    pub fn build(self) -> SelfIntersectOptions {
        self.options
    }
}

/// What [`resolve_self_intersections()`] observed and changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SelfIntersectReport {
    /// Self-intersecting triangle pairs present before resolution.
    pub initial_pairs: usize,
    /// Self-intersecting triangle pairs remaining after resolution.
    pub remaining_pairs: usize,
    /// Triangles removed during resolution.
    pub removed_triangles: usize,
    /// Triangles added during resolution.
    pub added_triangles: usize,
    /// Resolution mode executed.
    pub mode_used: SelfIntersectionMode,
    /// Whether `UV_CHANNEL` was dropped (e.g. by [`SelfIntersectionMode::VoxelHeal`]).
    pub uvs_dropped: bool,
}

fn validate_options(o: &SelfIntersectOptions) -> Result<()> {
    if let Some(h) = o.voxel_size
        && (!h.is_finite() || h <= 0.0)
    {
        return Err(OpsError::InvalidInput(
            "voxel_size must be positive and finite",
        ));
    }
    if o.max_voxels < 64 {
        return Err(OpsError::InvalidInput("max_voxels must be at least 64"));
    }
    if !o.stitch_tolerance.is_finite() || o.stitch_tolerance < 0.0 {
        return Err(OpsError::InvalidInput(
            "stitch_tolerance must be finite and non-negative",
        ));
    }
    Ok(())
}

pub(crate) fn find_self_intersecting_pairs(
    mesh: &Mesh,
    bvh: Option<&Bvh>,
) -> Result<Vec<(TriangleId, TriangleId)>> {
    let mut pairs = Vec::new();
    if let Some(b) = bvh {
        self_intersecting_pairs(mesh, b, &mut pairs)?;
        return Ok(pairs);
    }
    if mesh.triangle_count() == 0 {
        return Ok(pairs);
    }
    let built = Bvh::build(mesh, &BuildOptions::default())?;
    self_intersecting_pairs(mesh, &built, &mut pairs)?;
    Ok(pairs)
}

#[inline]
fn f64p(mesh: &Mesh, v: VertexId) -> [f64; 3] {
    point(mesh, v)
}

#[inline]
fn add64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale64(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn lerp64(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn expand_rings(mesh: &Mesh, seed: &[TriangleId], ring_expansion: u32) -> Result<Vec<TriangleId>> {
    let slots = mesh.triangle_slots();
    let total_alive = mesh.triangle_count();
    let mut selected = filled(slots, false)?;
    for &t in seed {
        selected[t as usize] = true;
    }
    let mut selected_count = seed.len();
    if ring_expansion == 0 || total_alive.saturating_sub(selected_count) < 3 {
        let mut out = Vec::new();
        reserve(&mut out, selected_count)?;
        for (i, &s) in selected.iter().enumerate() {
            if s {
                push(&mut out, i as TriangleId)?;
            }
        }
        return Ok(out);
    }

    let fans = VertexFans::build(mesh)?;
    let mut next_ring: Vec<TriangleId> = Vec::new();
    for _ in 0..ring_expansion {
        next_ring.clear();
        for (t, &s) in selected.iter().enumerate() {
            if !s {
                continue;
            }
            let Some(tri) = mesh.triangle(t as TriangleId) else {
                continue;
            };
            for v in tri {
                for &nb in fans.fan(v) {
                    if !selected[nb as usize] {
                        push(&mut next_ring, nb)?;
                    }
                }
            }
        }
        next_ring.sort_unstable();
        next_ring.dedup();
        if next_ring.is_empty() || total_alive.saturating_sub(selected_count + next_ring.len()) < 3
        {
            break;
        }
        for &nb in &next_ring {
            selected[nb as usize] = true;
        }
        selected_count += next_ring.len();
    }

    let mut out = Vec::new();
    reserve(&mut out, selected_count)?;
    for (i, &s) in selected.iter().enumerate() {
        if s {
            push(&mut out, i as TriangleId)?;
        }
    }
    Ok(out)
}

fn delete_and_fill_seed(
    tx: &mut Transaction,
    seed_tris: &[TriangleId],
    ring_expansion: u32,
    max_hole_edges: Option<usize>,
    fair: bool,
) -> Result<(usize, usize)> {
    if seed_tris.is_empty() {
        return Ok((0, 0));
    }
    let to_remove = expand_rings(tx.mesh(), seed_tris, ring_expansion)?;
    if to_remove.is_empty() {
        return Ok((0, 0));
    }
    let unused = unused_vertices(tx.mesh(), &to_remove, &[], &[])?;
    let mut removed = to_remove.len();
    tx.remove_triangles(&to_remove)?;
    if !unused.is_empty() {
        tx.remove_vertices(&unused)?;
    }

    let mf = make_manifold(tx, &ManifoldOptions::default())?;
    removed += mf.removed_duplicate_triangles + mf.pruned_fin_triangles;
    let _ = orient(tx)?;
    let fill = fill_holes(
        tx,
        &FillOptions {
            max_edges: Some(max_hole_edges.unwrap_or(usize::MAX)),
            allow_unclean: false,
            fair,
            ..FillOptions::default()
        },
    )?;
    let added: usize = fill.filled.iter().map(|f| f.triangles_added).sum();
    Ok((removed, added))
}

enum PairClip {
    DisjointOrPoint,
    Coplanar,
    Segment([f64; 3], [f64; 3]),
}

fn plane_line_interval(
    tri: [[f64; 3]; 3],
    d: [f64; 3],
    dir: [f64; 3],
    tol: f64,
) -> Option<(f64, f64, [f64; 3])> {
    let mut hits: Vec<(f64, [f64; 3])> = Vec::with_capacity(6);
    for i in 0..3 {
        if d[i].abs() <= tol {
            hits.push((dot(tri[i], dir), tri[i]));
        }
        let j = (i + 1) % 3;
        let di = d[i];
        let dj = d[j];
        if (di > tol && dj < -tol) || (di < -tol && dj > tol) {
            let t = di / (di - dj);
            let p = lerp64(tri[i], tri[j], t);
            hits.push((dot(p, dir), p));
        }
    }
    if hits.len() < 2 {
        return None;
    }
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    let first = hits[0];
    let last = *hits.last()?;
    Some((first.0, last.0, first.1))
}

fn clip_tri_tri_segment(a: [[f64; 3]; 3], b: [[f64; 3]; 3], tol: f64) -> PairClip {
    let na_raw = cross(sub(a[1], a[0]), sub(a[2], a[0]));
    let nb_raw = cross(sub(b[1], b[0]), sub(b[2], b[0]));
    let la = norm64(na_raw);
    let lb = norm64(nb_raw);
    if la <= 1e-18 || lb <= 1e-18 {
        return PairClip::Coplanar;
    }
    let na = scale64(na_raw, 1.0 / la);
    let nb = scale64(nb_raw, 1.0 / lb);

    let dir_raw = cross(na, nb);
    let ldir = norm64(dir_raw);
    if ldir <= 1e-6 {
        return PairClip::Coplanar;
    }
    let dir = scale64(dir_raw, 1.0 / ldir);

    let da = [
        dot(na, sub(b[0], a[0])),
        dot(na, sub(b[1], a[0])),
        dot(na, sub(b[2], a[0])),
    ];
    let db = [
        dot(nb, sub(a[0], b[0])),
        dot(nb, sub(a[1], b[0])),
        dot(nb, sub(a[2], b[0])),
    ];

    if da.iter().all(|&x| x.abs() <= tol) || db.iter().all(|&x| x.abs() <= tol) {
        return PairClip::Coplanar;
    }

    let Some((sa0, sa1, p_ref_a)) = plane_line_interval(a, db, dir, tol) else {
        return PairClip::DisjointOrPoint;
    };
    let Some((sb0, sb1, _p_ref_b)) = plane_line_interval(b, da, dir, tol) else {
        return PairClip::DisjointOrPoint;
    };

    let t0 = sa0.max(sb0);
    let t1 = sa1.min(sb1);
    if t1 - t0 <= tol {
        return PairClip::DisjointOrPoint;
    }
    let s_ref = dot(p_ref_a, dir);
    let p0 = add64(p_ref_a, scale64(dir, t0 - s_ref));
    let p1 = add64(p_ref_a, scale64(dir, t1 - s_ref));
    PairClip::Segment(p0, p1)
}

#[derive(Clone, Copy)]
struct SubTri {
    verts: [u32; 3],
}

fn barycentric_on_tri(
    p: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    tol: f64,
) -> Option<[f64; 3]> {
    let v0 = sub(b, a);
    let v1 = sub(c, a);
    let v2 = sub(p, a);
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() <= 1e-24 {
        return None;
    }
    let inv = 1.0 / denom;
    let v = (d11 * d20 - d01 * d21) * inv;
    let w = (d00 * d21 - d01 * d20) * inv;
    let u = 1.0 - v - w;
    if u < -tol || v < -tol || w < -tol {
        return None;
    }
    let u = u.max(0.0);
    let v = v.max(0.0);
    let w = w.max(0.0);
    let s = u + v + w;
    if s <= 0.0 {
        return None;
    }
    Some([u / s, v / s, w / s])
}

#[derive(Clone, Copy)]
struct LocalCutTri {
    orig_verts: [VertexId; 3],
    // Projection axes (dropping dominant normal coordinate) with CCW parent triangle.
    u_axis: usize,
    v_axis: usize,
}

impl LocalCutTri {
    fn from_mesh(mesh: &Mesh, t: TriangleId) -> Option<Self> {
        let tri = mesh.triangle(t)?;
        let p = [f64p(mesh, tri[0]), f64p(mesh, tri[1]), f64p(mesh, tri[2])];
        let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        let ax = [n[0].abs(), n[1].abs(), n[2].abs()];
        let drop = if ax[0] >= ax[1] && ax[0] >= ax[2] {
            0
        } else if ax[1] >= ax[2] {
            1
        } else {
            2
        };
        let u0 = (drop + 1) % 3;
        let v0 = (drop + 2) % 3;
        let area2 = (p[1][u0] - p[0][u0]) * (p[2][v0] - p[0][v0])
            - (p[1][v0] - p[0][v0]) * (p[2][u0] - p[0][u0]);
        let (u_axis, v_axis) = if area2 >= 0.0 { (u0, v0) } else { (v0, u0) };
        Some(Self {
            orig_verts: tri,
            u_axis,
            v_axis,
        })
    }

    #[inline]
    fn proj(&self, p: [f64; 3]) -> [f64; 2] {
        [p[self.u_axis], p[self.v_axis]]
    }
}

#[inline]
fn orient2(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn seg_cross_param(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> Option<f64> {
    let s1 = orient2(a, b, c);
    let s2 = orient2(a, b, d);
    let s3 = orient2(c, d, a);
    let s4 = orient2(c, d, b);
    if ((s1 > 1e-12 && s2 < -1e-12) || (s1 < -1e-12 && s2 > 1e-12))
        && ((s3 > 1e-12 && s4 < -1e-12) || (s3 < -1e-12 && s4 > 1e-12))
    {
        let denom = s3 - s4;
        if denom.abs() > 1e-18 {
            // Parameter t along edge `c -> d`: Wait! Notice s3, s4 test `a, b` against `c -> d`.
            // Intersection along `c -> d` is given by `s1 / (s1 - s2)`!
            let t = s1 / (s1 - s2);
            if t > 1e-7 && t < 1.0 - 1e-7 {
                return Some(t);
            }
        }
    }
    None
}

fn point_to_seg_param(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> (f64, f64) {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let apx = p[0] - a[0];
    let apy = p[1] - a[1];
    let len2 = abx * abx + aby * aby;
    if len2 <= 1e-24 {
        return (0.0, (apx * apx + apy * apy).sqrt());
    }
    let t = ((apx * abx + apy * aby) / len2).clamp(0.0, 1.0);
    let dx = p[0] - (a[0] + abx * t);
    let dy = p[1] - (a[1] + aby * t);
    (t, (dx * dx + dy * dy).sqrt())
}

fn subtriangulate_triangle(
    cut_info: &LocalCutTri,
    cut_points: &[u32],
    cut_segs: &[(u32, u32)],
    node_pos: &mut Vec<[f64; 3]>,
    node_parents: &mut Vec<[(VertexId, f32); 3]>,
    tol: f64,
    out_tris: &mut Vec<[u32; 3]>,
) -> Result<()> {
    let mut sub_tris = vec![SubTri {
        verts: [
            cut_info.orig_verts[0],
            cut_info.orig_verts[1],
            cut_info.orig_verts[2],
        ],
    }];

    // Insert all explicit points lying inside or on edges of `sub_tris`.
    for &pt in cut_points {
        insert_point_into_subtris(&mut sub_tris, cut_info, pt, node_pos, tol)?;
    }

    // Enforce every constraint segment `(u, v)` by splitting any crossed sub-triangle edge.
    let mut queue: Vec<(u32, u32)> = Vec::new();
    reserve(&mut queue, cut_segs.len())?;
    for &(a, b) in cut_segs {
        if a != b {
            push(&mut queue, (a, b))?;
        }
    }
    let mut steps = 0usize;
    let max_steps = queue.len().saturating_mul(64).saturating_add(128);
    while let Some((sa, sb)) = queue.pop() {
        if steps >= max_steps {
            break;
        }
        steps += 1;
        if sa == sb {
            continue;
        }
        let already_edge = sub_tris.iter().any(|st| {
            let [x, y, z] = st.verts;
            (x == sa && (y == sb || z == sb))
                || (y == sa && (x == sb || z == sb))
                || (z == sa && (x == sb || y == sb))
        });
        if already_edge {
            continue;
        }
        let pa = cut_info.proj(node_pos[sa as usize]);
        let pb = cut_info.proj(node_pos[sb as usize]);

        // Find closest crossed interior/edge of any sub-triangle along `sa -> sb`.
        let mut hit_edge: Option<(u32, u32, f64)> = None;
        for st in &sub_tris {
            for k in 0..3 {
                let u = st.verts[k];
                let v = st.verts[(k + 1) % 3];
                if u == sa || u == sb || v == sa || v == sb {
                    continue;
                }
                let pu = cut_info.proj(node_pos[u as usize]);
                let pv = cut_info.proj(node_pos[v as usize]);
                if let Some(t_edge) = seg_cross_param(pa, pb, pu, pv) {
                    let pt = lerp64(node_pos[u as usize], node_pos[v as usize], t_edge);
                    let p2 = cut_info.proj(pt);
                    let (t_seg, _) = point_to_seg_param(p2, pa, pb);
                    if hit_edge.is_none_or(|(_, _, best_t)| t_seg < best_t) {
                        hit_edge = Some((u, v, t_edge));
                    }
                }
            }
        }
        if let Some((eu, ev, t_edge)) = hit_edge {
            let new_p = lerp64(node_pos[eu as usize], node_pos[ev as usize], t_edge);
            let a3 = node_pos[cut_info.orig_verts[0] as usize];
            let b3 = node_pos[cut_info.orig_verts[1] as usize];
            let c3 = node_pos[cut_info.orig_verts[2] as usize];
            let Some(bary) = barycentric_on_tri(new_p, a3, b3, c3, 1e-3) else {
                continue;
            };
            let new_id = u32::try_from(node_pos.len())
                .map_err(|_| OpsError::InvalidInput("vertex count exceeds u32::MAX"))?;
            push(node_pos, new_p)?;
            push(
                node_parents,
                [
                    (cut_info.orig_verts[0], bary[0] as f32),
                    (cut_info.orig_verts[1], bary[1] as f32),
                    (cut_info.orig_verts[2], bary[2] as f32),
                ],
            )?;
            split_subtri_edge(&mut sub_tris, eu, ev, new_id);
            push(&mut queue, (sa, new_id))?;
            push(&mut queue, (new_id, sb))?;
        }
    }

    for st in sub_tris {
        let [a, b, c] = st.verts;
        if a == b || b == c || c == a {
            continue;
        }
        let area2 = orient2(
            cut_info.proj(node_pos[a as usize]),
            cut_info.proj(node_pos[b as usize]),
            cut_info.proj(node_pos[c as usize]),
        );
        if area2.abs() <= 1e-18 {
            continue;
        }
        let ordered = if area2 > 0.0 { [a, b, c] } else { [a, c, b] };
        push(out_tris, ordered)?;
    }
    Ok(())
}

fn split_subtri_edge(sub_tris: &mut Vec<SubTri>, u: u32, v: u32, mid: u32) {
    let mut i = 0usize;
    while i < sub_tris.len() {
        let [a, b, c] = sub_tris[i].verts;
        let matched = if (a == u && b == v) || (a == v && b == u) {
            Some((a, b, c))
        } else if (b == u && c == v) || (b == v && c == u) {
            Some((b, c, a))
        } else if (c == u && a == v) || (c == v && a == u) {
            Some((c, a, b))
        } else {
            None
        };
        if let Some((e0, e1, opp)) = matched
            && mid != e0
            && mid != e1
            && mid != opp
        {
            sub_tris[i] = SubTri {
                verts: [e0, mid, opp],
            };
            sub_tris.push(SubTri {
                verts: [mid, e1, opp],
            });
            i += 2;
            continue;
        }
        i += 1;
    }
}

fn insert_point_into_subtris(
    sub_tris: &mut Vec<SubTri>,
    cut_info: &LocalCutTri,
    pt: u32,
    node_pos: &[[f64; 3]],
    tol: f64,
) -> Result<()> {
    let p2 = cut_info.proj(node_pos[pt as usize]);
    let tol2 = tol * tol;

    for st in sub_tris.iter() {
        for v in st.verts {
            if v == pt {
                return Ok(());
            }
            let pv = cut_info.proj(node_pos[v as usize]);
            let dx = p2[0] - pv[0];
            let dy = p2[1] - pv[1];
            if dx * dx + dy * dy <= tol2 {
                return Ok(());
            }
        }
    }

    // Check if `pt` lies on an existing sub-triangle edge.
    let mut on_edge: Option<(u32, u32)> = None;
    for st in sub_tris.iter() {
        for k in 0..3 {
            let u = st.verts[k];
            let v = st.verts[(k + 1) % 3];
            let pu = cut_info.proj(node_pos[u as usize]);
            let pv = cut_info.proj(node_pos[v as usize]);
            let (t, dist) = point_to_seg_param(p2, pu, pv);
            if t > 1e-4 && t < 1.0 - 1e-4 && dist <= tol * 2.0 {
                on_edge = Some((u, v));
                break;
            }
        }
        if on_edge.is_some() {
            break;
        }
    }
    if let Some((u, v)) = on_edge {
        split_subtri_edge(sub_tris, u, v, pt);
        return Ok(());
    }

    // Otherwise locate containing sub-triangle and split 1 -> 3.
    let mut hit_idx: Option<usize> = None;
    for (idx, st) in sub_tris.iter().enumerate() {
        let p0 = cut_info.proj(node_pos[st.verts[0] as usize]);
        let p1 = cut_info.proj(node_pos[st.verts[1] as usize]);
        let p2v = cut_info.proj(node_pos[st.verts[2] as usize]);
        let area = orient2(p0, p1, p2v);
        if area.abs() <= 1e-18 {
            continue;
        }
        let s = if area > 0.0 { 1.0 } else { -1.0 };
        let o0 = orient2(p0, p1, p2) * s;
        let o1 = orient2(p1, p2v, p2) * s;
        let o2 = orient2(p2v, p0, p2) * s;
        let eps = -tol * area.abs().sqrt().max(1.0) * 1e-2;
        if o0 >= eps && o1 >= eps && o2 >= eps {
            hit_idx = Some(idx);
            break;
        }
    }
    if let Some(idx) = hit_idx {
        let [a, b, c] = sub_tris[idx].verts;
        sub_tris[idx] = SubTri { verts: [a, b, pt] };
        push(sub_tris, SubTri { verts: [b, c, pt] })?;
        push(sub_tris, SubTri { verts: [c, a, pt] })?;
    }
    Ok(())
}

fn cull_internal_seam_patches(tx: &mut Transaction, diag: f64) -> Result<usize> {
    let mesh = tx.mesh();
    let t_slots = mesh.triangle_slots();
    if mesh.triangle_count() < 4 {
        return Ok(0);
    }

    // Build edge incidence directly over alive proper triangles.
    let mut edge_refs: Vec<((VertexId, VertexId), TriangleId)> = Vec::new();
    for t in 0..t_slots as TriangleId {
        let Some([a, b, c]) = mesh.triangle(t) else {
            continue;
        };
        if a == b || b == c || c == a {
            continue;
        }
        for (u, v) in [(a, b), (b, c), (c, a)] {
            let key = if u < v { (u, v) } else { (v, u) };
            push(&mut edge_refs, (key, t))?;
        }
    }
    edge_refs.sort_unstable();

    let mut has_nm_edge = false;
    let mut uf = UnionFind::new();
    uf.reset(t_slots)?;
    let mut on_nm_edge = filled(t_slots, false)?;
    let mut i = 0usize;
    while i < edge_refs.len() {
        let key = edge_refs[i].0;
        let mut j = i + 1;
        while j < edge_refs.len() && edge_refs[j].0 == key {
            j += 1;
        }
        let len = j - i;
        if len == 2 {
            uf.union(edge_refs[i].1, edge_refs[i + 1].1);
        } else if len > 2 {
            has_nm_edge = true;
            for item in &edge_refs[i..j] {
                on_nm_edge[item.1 as usize] = true;
            }
        }
        i = j;
    }
    if !has_nm_edge {
        return Ok(0);
    }

    let mut work = Mesh::from_buffers(
        &mesh
            .positions()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        &(0..t_slots as TriangleId)
            .filter_map(|t| mesh.triangle(t))
            .flatten()
            .collect::<Vec<_>>(),
    )?;
    let work_bvh = Bvh::build(&work, &BuildOptions::default())?;
    let sign_data = SignData::build(&mut work, &work_bvh, SignMode::Winding)?;

    // Pick one interior sample per patch (`uf` root) and evaluate outside winding number.
    let offset = (1e-3 * diag).max(1e-5) as f32;
    let mut roots: Vec<u32> = Vec::new();
    let mut samples: Vec<[f32; 3]> = Vec::new();
    for t in 0..t_slots as TriangleId {
        let Some([a, b, c]) = mesh.triangle(t) else {
            continue;
        };
        let r = uf.find(t);
        if roots.contains(&r) {
            continue;
        }
        // Prefer an interior triangle away from non-manifold seams if available.
        let p0 = f64p(mesh, a);
        let p1 = f64p(mesh, b);
        let p2 = f64p(mesh, c);
        let n = cross(sub(p1, p0), sub(p2, p0));
        let l = norm64(n);
        if l <= 1e-18 {
            continue;
        }
        let center = [
            ((p0[0] + p1[0] + p2[0]) / 3.0) as f32,
            ((p0[1] + p1[1] + p2[1]) / 3.0) as f32,
            ((p0[2] + p1[2] + p2[2]) / 3.0) as f32,
        ];
        let pt = [
            center[0] + offset * (n[0] / l) as f32,
            center[1] + offset * (n[1] / l) as f32,
            center[2] + offset * (n[2] / l) as f32,
        ];
        push(&mut roots, r)?;
        push(&mut samples, pt)?;
    }

    let mut drop_roots: Vec<u32> = Vec::new();
    for (&r, &pt) in roots.iter().zip(&samples) {
        let w = sign_data.winding_number(&work, &work_bvh, pt);
        if w > 0.75 {
            push(&mut drop_roots, r)?;
        }
    }
    if drop_roots.is_empty() || drop_roots.len() == roots.len() {
        return Ok(0);
    }

    let mut to_remove: Vec<TriangleId> = Vec::new();
    for t in 0..t_slots as TriangleId {
        if mesh.is_triangle_alive(t) && drop_roots.contains(&uf.find(t)) {
            push(&mut to_remove, t)?;
        }
    }
    if to_remove.is_empty() || mesh.triangle_count().saturating_sub(to_remove.len()) < 4 {
        return Ok(0);
    }
    let unused = unused_vertices(mesh, &to_remove, &[], &[])?;
    let removed = to_remove.len();
    tx.remove_triangles(&to_remove)?;
    if !unused.is_empty() {
        tx.remove_vertices(&unused)?;
    }
    Ok(removed)
}

fn cut_and_stitch_pass(
    tx: &mut Transaction,
    pairs: &[(TriangleId, TriangleId)],
    options: &SelfIntersectOptions,
) -> Result<(usize, usize)> {
    let mesh = tx.mesh();
    let v_slots = mesh.vertex_slots();
    let t_slots = mesh.triangle_slots();
    let tol = f64::from(options.stitch_tolerance.max(1e-6));

    // Seed node table with existing mesh vertices (`0..v_slots`).
    let mut node_pos: Vec<[f64; 3]> = Vec::new();
    reserve(&mut node_pos, v_slots)?;
    let mut node_parents: Vec<[(VertexId, f32); 3]> = Vec::new();
    reserve(&mut node_parents, v_slots)?;
    for v in 0..v_slots as VertexId {
        push(&mut node_pos, f64p(mesh, v))?;
        push(&mut node_parents, [(v, 1.0), (v, 0.0), (v, 0.0)])?;
    }

    let mut coplanar_tris: Vec<TriangleId> = Vec::new();
    let mut raw_segs: Vec<(TriangleId, TriangleId, [f64; 3], [f64; 3])> = Vec::new();

    for &(ta, tb) in pairs {
        let Some(va) = mesh.triangle(ta) else {
            continue;
        };
        let Some(vb) = mesh.triangle(tb) else {
            continue;
        };
        let a = va.map(|v| f64p(mesh, v));
        let b = vb.map(|v| f64p(mesh, v));
        match clip_tri_tri_segment(a, b, tol) {
            PairClip::DisjointOrPoint => {}
            PairClip::Coplanar => {
                push(&mut coplanar_tris, ta)?;
                push(&mut coplanar_tris, tb)?;
            }
            PairClip::Segment(p0, p1) => {
                push(&mut raw_segs, (ta, tb, p0, p1))?;
            }
        }
    }

    // Helper: allocate or reuse a cut vertex on `tri`, snapping to corners / existing cut nodes within `tol`.
    let mut cut_tri_points: Vec<(TriangleId, u32)> = Vec::new();
    let mut cut_tri_segs: Vec<(TriangleId, u32, u32)> = Vec::new();
    let mut alloc_or_find_cut_point =
        |pt: [f64; 3], tri_id: TriangleId, tri_v: [VertexId; 3]| -> Result<Option<u32>> {
            let tol2 = tol * tol;
            // Check original vertices of triangle first.
            for &v in &tri_v {
                let d = sub(pt, node_pos[v as usize]);
                if dot(d, d) <= tol2 {
                    return Ok(Some(v));
                }
            }
            // Check already allocated cut nodes within tolerance.
            for (idx, &np) in node_pos.iter().enumerate().skip(v_slots) {
                let d = sub(pt, np);
                if dot(d, d) <= tol2 {
                    return Ok(Some(idx as u32));
                }
            }
            let a = f64p(mesh, tri_v[0]);
            let b = f64p(mesh, tri_v[1]);
            let c = f64p(mesh, tri_v[2]);
            let Some(bary) = barycentric_on_tri(pt, a, b, c, 1e-3) else {
                return Ok(None);
            };
            let id = u32::try_from(node_pos.len())
                .map_err(|_| OpsError::InvalidInput("vertex count exceeds u32::MAX"))?;
            push(&mut node_pos, pt)?;
            push(
                &mut node_parents,
                [
                    (tri_v[0], bary[0] as f32),
                    (tri_v[1], bary[1] as f32),
                    (tri_v[2], bary[2] as f32),
                ],
            )?;
            let _ = tri_id;
            Ok(Some(id))
        };

    for &(ta, tb, p0, p1) in &raw_segs {
        let Some(va) = mesh.triangle(ta) else {
            continue;
        };
        let Some(vb) = mesh.triangle(tb) else {
            continue;
        };
        let Some(id0) = alloc_or_find_cut_point(p0, ta, va)? else {
            continue;
        };
        let Some(id1) = alloc_or_find_cut_point(p1, tb, vb)? else {
            continue;
        };
        if id0 == id1 {
            continue;
        }
        push(&mut cut_tri_points, (ta, id0))?;
        push(&mut cut_tri_points, (ta, id1))?;
        push(&mut cut_tri_segs, (ta, id0, id1))?;
        push(&mut cut_tri_points, (tb, id0))?;
        push(&mut cut_tri_points, (tb, id1))?;
        push(&mut cut_tri_segs, (tb, id0, id1))?;
    }

    // Record cut nodes landing along existing triangle edges so neighboring uncut triangles split seamlessly.
    let mut edge_splits: Vec<((VertexId, VertexId), u32)> = Vec::new();
    let new_node_count = node_pos.len() - v_slots;
    if new_node_count > 0 {
        let mut touched_tris: Vec<TriangleId> = cut_tri_points.iter().map(|&(t, _)| t).collect();
        touched_tris.sort_unstable();
        touched_tris.dedup();

        // Check every cut point inserted on each touched triangle's 3 edges.
        for &(t, pt_id) in &cut_tri_points {
            if (pt_id as usize) < v_slots {
                continue;
            }
            let Some([v0, v1, v2]) = mesh.triangle(t) else {
                continue;
            };
            let pt = node_pos[pt_id as usize];
            for (u, v) in [(v0, v1), (v1, v2), (v2, v0)] {
                let pu = node_pos[u as usize];
                let pv = node_pos[v as usize];
                let ab = sub(pv, pu);
                let len2 = dot(ab, ab);
                if len2 <= 1e-24 {
                    continue;
                }
                let t_param = dot(sub(pt, pu), ab) / len2;
                if t_param > 1e-4 && t_param < 1.0 - 1e-4 {
                    let proj = add64(pu, scale64(ab, t_param));
                    let d = sub(pt, proj);
                    if dot(d, d) <= (tol * 2.0) * (tol * 2.0) {
                        let key = if u < v { (u, v) } else { (v, u) };
                        push(&mut edge_splits, (key, pt_id))?;
                    }
                }
            }
        }
    }
    edge_splits.sort_unstable();
    edge_splits.dedup();

    // Propagate edge split points to every alive triangle sharing that edge.
    if !edge_splits.is_empty() {
        for t in 0..t_slots as TriangleId {
            let Some([a, b, c]) = mesh.triangle(t) else {
                continue;
            };
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let key = if u < v { (u, v) } else { (v, u) };
                let start = edge_splits.partition_point(|&(k, _)| k < key);
                for &(k, pt_id) in &edge_splits[start..] {
                    if k != key {
                        break;
                    }
                    push(&mut cut_tri_points, (t, pt_id))?;
                }
            }
        }
    }

    cut_tri_points.sort_unstable();
    cut_tri_points.dedup();
    cut_tri_segs.sort_unstable();
    cut_tri_segs.dedup();

    let mut cut_tris: Vec<TriangleId> = cut_tri_points.iter().map(|&(t, _)| t).collect();
    cut_tris.sort_unstable();
    cut_tris.dedup();

    let mut replaced_tris: Vec<TriangleId> = Vec::new();
    let mut new_raw_tris: Vec<[u32; 3]> = Vec::new();
    let mut pts_buf: Vec<u32> = Vec::new();
    let mut segs_buf: Vec<(u32, u32)> = Vec::new();

    for &t in &cut_tris {
        let Some(info) = LocalCutTri::from_mesh(mesh, t) else {
            continue;
        };
        pts_buf.clear();
        let p_start = cut_tri_points.partition_point(|&(tid, _)| tid < t);
        for &(tid, pt) in &cut_tri_points[p_start..] {
            if tid != t {
                break;
            }
            push(&mut pts_buf, pt)?;
        }
        segs_buf.clear();
        let s_start = cut_tri_segs.partition_point(|&(tid, _, _)| tid < t);
        for &(tid, a, b) in &cut_tri_segs[s_start..] {
            if tid != t {
                break;
            }
            push(&mut segs_buf, (a, b))?;
        }
        if pts_buf.is_empty() && segs_buf.is_empty() {
            continue;
        }
        push(&mut replaced_tris, t)?;
        subtriangulate_triangle(
            &info,
            &pts_buf,
            &segs_buf,
            &mut node_pos,
            &mut node_parents,
            tol,
            &mut new_raw_tris,
        )?;
    }

    let mut removed_count = 0usize;
    let mut added_count = 0usize;

    if !replaced_tris.is_empty() && !new_raw_tris.is_empty() {
        let added_nodes = node_pos.len() - v_slots;
        let mut node_to_vid = filled(added_nodes, 0 as VertexId)?;
        for (i, slot) in (v_slots..node_pos.len()).enumerate() {
            let p = node_pos[slot].map(|x| x as f32);
            let parents = node_parents[slot];
            node_to_vid[i] = tx.add_vertex_with_parents(p, &parents)?;
        }
        let mut mapped_tris: Vec<[VertexId; 3]> = Vec::new();
        reserve(&mut mapped_tris, new_raw_tris.len())?;
        for &tri in &new_raw_tris {
            let m = tri.map(|id| {
                if (id as usize) >= v_slots {
                    node_to_vid[(id as usize) - v_slots]
                } else {
                    id
                }
            });
            push(&mut mapped_tris, m)?;
        }
        let _ = tx.add_triangles(&mapped_tris)?;
        added_count += mapped_tris.len();
        removed_count += replaced_tris.len();
        tx.remove_triangles(&replaced_tris)?;
        let unused = unused_vertices(tx.mesh(), &[], &[], &[])?;
        if !unused.is_empty() {
            tx.remove_vertices(&unused)?;
        }
    }

    // Compute bounds diagonal for patch winding culling.
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in tx.mesh().positions() {
        for k in 0..3 {
            let v = f64::from(p[k]);
            if v.is_finite() {
                min[k] = min[k].min(v);
                max[k] = max[k].max(v);
            }
        }
    }
    let diag = norm64(sub(max, min)).max(1.0);
    removed_count += cull_internal_seam_patches(tx, diag)?;

    // Clean any degenerate cut slivers and split any remaining seam non-manifold junctions.
    let deg = remove_degenerate(tx, &DegenerateOptions::default())?;
    removed_count += deg.removed_triangles;
    let mf = make_manifold(tx, &ManifoldOptions::default())?;
    removed_count += mf.removed_duplicate_triangles + mf.pruned_fin_triangles;

    // Verify if any coplanar / unsplit / touch intersections remain; fall back locally to DeleteAndFill.
    let remaining = find_self_intersecting_pairs(tx.mesh(), None)?;
    if !remaining.is_empty() || !coplanar_tris.is_empty() {
        let mut fallback_seed: Vec<TriangleId> = Vec::new();
        for (a, b) in remaining {
            push(&mut fallback_seed, a)?;
            push(&mut fallback_seed, b)?;
        }
        for t in coplanar_tris {
            if tx.mesh().is_triangle_alive(t) {
                push(&mut fallback_seed, t)?;
            }
        }
        fallback_seed.sort_unstable();
        fallback_seed.dedup();
        if !fallback_seed.is_empty() {
            let (r, a) = delete_and_fill_seed(
                tx,
                &fallback_seed,
                options.ring_expansion,
                options.max_hole_edges,
                options.fair,
            )?;
            removed_count += r;
            added_count += a;
        }
    }

    Ok((removed_count, added_count))
}

fn choose_heal_grid_spec(
    min: [f32; 3],
    max: [f32; 3],
    voxel_size: Option<f32>,
    max_voxels: usize,
) -> Result<GridSpec> {
    if let Some(h) = voxel_size {
        return Ok(GridSpec::around_bounds(
            min,
            max,
            h,
            0.0,
            GRID_PAD_VOXELS,
            max_voxels,
        )?);
    }
    let dx = f64::from(max[0]) - f64::from(min[0]);
    let dy = f64::from(max[1]) - f64::from(min[1]);
    let dz = f64::from(max[2]) - f64::from(min[2]);
    let diag = (dx * dx + dy * dy + dz * dz).sqrt();
    let mut h = if diag > 0.0 {
        ((diag / AUTO_VOXEL_RES as f64) as f32).max(1e-4)
    } else {
        1e-2
    };
    let max_span = dx.max(dy).max(dz);
    let axis_budget = (max_voxels as f64).cbrt() - (2 * GRID_PAD_VOXELS + 1) as f64;
    if axis_budget > 0.0 && max_span > 0.0 {
        let lower = ((max_span / axis_budget) * 1.01) as f32;
        if lower.is_finite() && lower > h {
            h = lower;
        }
    }
    for _ in 0..32 {
        match GridSpec::around_bounds(min, max, h, 0.0, GRID_PAD_VOXELS, max_voxels) {
            Ok(spec) => return Ok(spec),
            Err(FieldsError::InvalidInput("grid exceeds max_voxels")) => {
                let next = h * 1.25;
                if !next.is_finite() {
                    break;
                }
                h = next;
            }
            Err(e) => return Err(e.into()),
        }
    }
    Ok(GridSpec::around_bounds(
        min,
        max,
        h,
        0.0,
        GRID_PAD_VOXELS,
        max_voxels,
    )?)
}

pub(crate) fn voxel_heal_surface(
    tx: &mut Transaction,
    voxel_size: Option<f32>,
    max_voxels: usize,
) -> Result<(usize, usize, bool)> {
    let mesh = tx.mesh();
    let had_uvs = mesh.uvs().is_some();
    let mut remap = filled(mesh.vertex_slots(), u32::MAX)?;
    let mut work_positions: Vec<[f32; 3]> = Vec::new();
    let mut work_indices: Vec<u32> = Vec::new();
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some([a, b, c]) = mesh.triangle(t) else {
            continue;
        };
        if a == b || b == c || c == a {
            continue;
        }
        let tri_verts = [a, b, c];
        let raw_pos = mesh.positions();
        for v in tri_verts {
            let p = raw_pos[v as usize];
            if !p.iter().all(|x| x.is_finite()) {
                return Err(OpsError::InvalidInput(
                    "mesh contains non-finite coordinates",
                ));
            }
            let idx = if remap[v as usize] == u32::MAX {
                let next = u32::try_from(work_positions.len())
                    .map_err(|_| OpsError::InvalidInput("vertex count exceeds u32::MAX"))?;
                for k in 0..3 {
                    min[k] = min[k].min(p[k]);
                    max[k] = max[k].max(p[k]);
                }
                push(&mut work_positions, p)?;
                remap[v as usize] = next;
                next
            } else {
                remap[v as usize]
            };
            push(&mut work_indices, idx)?;
        }
    }
    if work_indices.is_empty() {
        return Err(OpsError::InvalidInput(
            "voxel heal requires at least one proper triangle",
        ));
    }

    let flat_pos: Vec<f32> = work_positions.iter().flatten().copied().collect();
    let mut work = Mesh::from_buffers(&flat_pos, &work_indices)?;
    let work_bvh = Bvh::build(&work, &BuildOptions::default())?;
    let spec = choose_heal_grid_spec(min, max, voxel_size, max_voxels)?;
    let sign_data = SignData::build(&mut work, &work_bvh, SignMode::Winding)?;
    let narrow_band = 3.0 * spec.voxel_size;
    let signed_grid = bvh_signed_grid(&work, &work_bvh, &sign_data, spec, narrow_band, max_voxels)?;
    let mut iso = marching_cubes(&signed_grid, 0.0)?;
    if iso.triangles.is_empty() {
        let ugrid = fh_unsigned_grid(&work, spec, max_voxels)?;
        iso = marching_cubes(&ugrid, 0.5 * spec.voxel_size)?;
    }

    // Deduplicate identical-triple or zero-area triangles on reconstructed surface.
    let mut clean_mesh = Mesh::from_buffers(
        &iso.positions.iter().flatten().copied().collect::<Vec<_>>(),
        &iso.triangles.iter().flatten().copied().collect::<Vec<_>>(),
    )?;
    {
        let mut c_tx = clean_mesh.begin_edit();
        let _ = remove_degenerate(&mut c_tx, &DegenerateOptions::default())?;
        let _ = make_manifold(&mut c_tx, &ManifoldOptions::default())?;
        let _ = orient(&mut c_tx)?;
        c_tx.commit()?;
    }

    let mut new_pos: Vec<[f32; 3]> = Vec::new();
    let mut new_remap = filled(clean_mesh.vertex_slots(), u32::MAX)?;
    let mut new_tris: Vec<[VertexId; 3]> = Vec::new();
    let new_raw_pos = clean_mesh.positions();
    for t in 0..clean_mesh.triangle_slots() as TriangleId {
        let Some([a, b, c]) = clean_mesh.triangle(t) else {
            continue;
        };
        let mut mapped = [0; 3];
        for (k, v) in [a, b, c].into_iter().enumerate() {
            mapped[k] = if new_remap[v as usize] == u32::MAX {
                let next = u32::try_from(new_pos.len())
                    .map_err(|_| OpsError::InvalidInput("vertex count exceeds u32::MAX"))?;
                push(&mut new_pos, new_raw_pos[v as usize])?;
                new_remap[v as usize] = next;
                next
            } else {
                new_remap[v as usize]
            };
        }
        push(&mut new_tris, mapped)?;
    }

    let mut old_tris: Vec<TriangleId> = Vec::new();
    for t in 0..mesh.triangle_slots() as TriangleId {
        if mesh.is_triangle_alive(t) {
            push(&mut old_tris, t)?;
        }
    }
    let mut old_verts: Vec<VertexId> = Vec::new();
    for v in 0..mesh.vertex_slots() as VertexId {
        if mesh.is_vertex_alive(v) {
            push(&mut old_verts, v)?;
        }
    }

    let removed_triangles = old_tris.len();
    let added_triangles = new_tris.len();

    if !old_tris.is_empty() {
        tx.remove_triangles(&old_tris)?;
    }
    if !old_verts.is_empty() {
        tx.remove_vertices(&old_verts)?;
    }
    if had_uvs {
        tx.drop_channel(UV_CHANNEL)?;
    }
    if !new_pos.is_empty() {
        let new_vids = tx.add_vertices(&new_pos)?;
        let mut final_tris = filled(new_tris.len(), [0; 3])?;
        for (i, &[a, b, c]) in new_tris.iter().enumerate() {
            final_tris[i] = [
                new_vids[a as usize],
                new_vids[b as usize],
                new_vids[c as usize],
            ];
        }
        tx.add_triangles(&final_tris)?;
    }

    Ok((removed_triangles, added_triangles, had_uvs))
}

/// Detects and resolves self-intersecting non-adjacent triangle pairs in `tx`.
///
/// Reuses `bvh` if provided (verifying it matches `tx.mesh()`) or builds one on
/// demand.
///
/// - [`SelfIntersectionMode::None`]: reports `initial_pairs` without mutating `tx`.
/// - [`SelfIntersectionMode::DeleteAndFill`]: removes intersecting triangles (plus
///   `ring_expansion`), heals boundaries with [`make_manifold`], and seals clean
///   boundary loops via [`fill_holes`].
/// - [`SelfIntersectionMode::CutAndStitch`]: cuts intersecting triangles along
///   intersection segments, splits adjacent shared edges, culls enclosed
///   interior overlap patches, and falls back locally to `DeleteAndFill` on
///   coplanar/degenerate intersections.
/// - [`SelfIntersectionMode::VoxelHeal`]: rebuilds a watertight surface from the
///   generalized-winding signed distance field (`iso = 0.0`), dropping `UV_CHANNEL`.
///
/// # Errors
/// - [`OpsError::InvalidInput`] for invalid [`SelfIntersectOptions`] or empty/non-finite
///   meshes under `VoxelHeal`.
/// - [`OpsError::Query`] if a caller-supplied `bvh` does not match `tx.mesh()`.
/// - [`OpsError::Store`] when memory runs out or a store write fails.
pub fn resolve_self_intersections(
    tx: &mut Transaction,
    bvh: Option<&Bvh>,
    options: &SelfIntersectOptions,
) -> Result<SelfIntersectReport> {
    allocation_allowed()?;
    validate_options(options)?;

    let pairs = find_self_intersecting_pairs(tx.mesh(), bvh)?;
    let initial_pairs = pairs.len();
    if options.mode == SelfIntersectionMode::None || pairs.is_empty() {
        return Ok(SelfIntersectReport {
            initial_pairs,
            remaining_pairs: initial_pairs,
            removed_triangles: 0,
            added_triangles: 0,
            mode_used: options.mode,
            uvs_dropped: false,
        });
    }

    let (removed_triangles, added_triangles, uvs_dropped) = match options.mode {
        SelfIntersectionMode::None => (0, 0, false),
        SelfIntersectionMode::DeleteAndFill => {
            let mut seed: Vec<TriangleId> = Vec::new();
            reserve(&mut seed, pairs.len() * 2)?;
            for (a, b) in pairs {
                push(&mut seed, a)?;
                push(&mut seed, b)?;
            }
            seed.sort_unstable();
            seed.dedup();
            let (r, a) = delete_and_fill_seed(
                tx,
                &seed,
                options.ring_expansion,
                options.max_hole_edges,
                options.fair,
            )?;
            (r, a, false)
        }
        SelfIntersectionMode::CutAndStitch => {
            let (r, a) = cut_and_stitch_pass(tx, &pairs, options)?;
            (r, a, false)
        }
        SelfIntersectionMode::VoxelHeal => {
            voxel_heal_surface(tx, options.voxel_size, options.max_voxels)?
        }
    };

    let remaining_pairs = find_self_intersecting_pairs(tx.mesh(), None)?.len();
    Ok(SelfIntersectReport {
        initial_pairs,
        remaining_pairs,
        removed_triangles,
        added_triangles,
        mode_used: options.mode,
        uvs_dropped,
    })
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::{ChannelDesc, UV_CHANNEL};

    use super::*;
    use crate::analyze::{AnalyzeOptions, analyze};
    use crate::test_util::{crossed_boxes, edit, mesh_of, uv_sphere};

    #[test]
    fn clean_sphere_is_noop() {
        let mut mesh = uv_sphere(8, 12, 1.0);
        let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
        let (r, empty) = edit(&mut mesh, |tx| {
            resolve_self_intersections(tx, Some(&bvh), &SelfIntersectOptions::default())
        });
        assert!(empty);
        assert_eq!(r.initial_pairs, 0);
        assert_eq!(r.remaining_pairs, 0);
    }

    #[test]
    fn none_mode_reports_without_mutating() {
        let mut mesh = crossed_boxes();
        let (r, empty) = edit(&mut mesh, |tx| {
            resolve_self_intersections(
                tx,
                None,
                &SelfIntersectOptions {
                    mode: SelfIntersectionMode::None,
                    ..Default::default()
                },
            )
        });
        assert!(empty);
        assert!(r.initial_pairs > 0);
        assert_eq!(r.remaining_pairs, r.initial_pairs);
    }

    #[test]
    fn delete_and_fill_resolves_poked_sphere() {
        let mut mesh = uv_sphere(12, 16, 1.0);
        // Poke two adjacent poles deep across the interior to create self-intersections.
        edit(&mut mesh, |tx| {
            tx.set_positions(&[0], &[[0.0, -2.0, 0.0]])?;
            Ok(())
        });
        let (r, _) = edit(&mut mesh, |tx| {
            resolve_self_intersections(
                tx,
                None,
                &SelfIntersectOptions {
                    mode: SelfIntersectionMode::DeleteAndFill,
                    ring_expansion: 1,
                    fair: false,
                    ..Default::default()
                },
            )
        });
        assert!(r.initial_pairs > 0);
        assert_eq!(r.remaining_pairs, 0);
        assert!(r.removed_triangles > 0);
        let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
        assert_eq!(a.non_manifold_edges, 0);
        assert!(a.pinch_vertices.is_empty());
    }

    #[test]
    fn cut_and_stitch_resolves_crossed_boxes_and_coplanar_safely() {
        let mut mesh = crossed_boxes();
        let (r, _) = edit(&mut mesh, |tx| {
            resolve_self_intersections(
                tx,
                None,
                &SelfIntersectOptions {
                    mode: SelfIntersectionMode::CutAndStitch,
                    ring_expansion: 0,
                    fair: false,
                    ..Default::default()
                },
            )
        });
        assert!(r.initial_pairs > 0);
        assert_eq!(r.remaining_pairs, 0);
        let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
        assert_eq!(a.non_manifold_edges, 0);
        assert!(a.pinch_vertices.is_empty());

        // Coplanar overlapping triangles fallback safely without panicking or breaking manifoldness.
        let mut coplanar = mesh_of(
            &[
                0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, // tri 0
                0.2, 0.2, 0.0, 2.2, 0.2, 0.0, 0.2, 2.2, 0.0, // tri 1 coplanar overlapping
            ],
            &[[0, 1, 2], [3, 4, 5]],
        );
        let (rc, _) = edit(&mut coplanar, |tx| {
            resolve_self_intersections(
                tx,
                None,
                &SelfIntersectOptions {
                    mode: SelfIntersectionMode::CutAndStitch,
                    ..Default::default()
                },
            )
        });
        assert!(rc.initial_pairs > 0);
        assert_eq!(rc.remaining_pairs, 0);
    }

    #[test]
    fn voxel_heal_reconstructs_closed_manifold_and_drops_uv() {
        let mut mesh = crossed_boxes();
        mesh.add_channel(ChannelDesc::uv()).unwrap();
        let (r, _) = edit(&mut mesh, |tx| {
            resolve_self_intersections(
                tx,
                None,
                &SelfIntersectOptions {
                    mode: SelfIntersectionMode::VoxelHeal,
                    voxel_size: Some(0.15),
                    ..Default::default()
                },
            )
        });
        assert!(r.initial_pairs > 0);
        assert_eq!(r.remaining_pairs, 0);
        assert!(r.uvs_dropped);
        assert!(mesh.uvs().is_none());
        let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
        assert_eq!(a.non_manifold_edges, 0);
        assert!(a.pinch_vertices.is_empty());
        assert!(a.loops.is_empty());
    }

    #[test]
    fn stale_or_mismatched_bvh_fails() {
        let mut mesh = crossed_boxes();
        let other = Bvh::build(&uv_sphere(6, 8, 1.0), &BuildOptions::default()).unwrap();
        let mut tx = mesh.begin_edit();
        assert!(
            resolve_self_intersections(&mut tx, Some(&other), &SelfIntersectOptions::default())
                .is_err()
        );
    }

    #[test]
    fn uv_present_delete_and_fill_keeps_uv_channel() {
        let mut mesh = uv_sphere(10, 12, 1.0);
        mesh.add_channel(ChannelDesc::uv()).unwrap();
        edit(&mut mesh, |tx| {
            tx.set_positions(&[0], &[[0.0, -2.0, 0.0]])?;
            let ids: Vec<u32> = (0..tx.mesh().vertex_slots() as u32).collect();
            let mut uvs = Vec::new();
            for _ in &ids {
                uvs.push(0.5);
                uvs.push(0.5);
            }
            tx.write_channel(UV_CHANNEL, &ids, &uvs)?;
            Ok(())
        });
        let (r, _) = edit(&mut mesh, |tx| {
            resolve_self_intersections(tx, None, &SelfIntersectOptions::default())
        });
        assert!(!r.uvs_dropped);
        assert!(mesh.uvs().is_some());
    }
}
