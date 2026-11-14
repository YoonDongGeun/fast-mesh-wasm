//! Boolean operations (union, subtract, intersect) between two closed meshes.
//!
//! [`boolean_plan`] builds a canonical simplex-keyed intersection arrangement
//! between mesh A and the placed tool B, sub-triangulates cut faces via
//! constrained 2D triangulation, classifies connected seam-bounded patches
//! using exact coplanar predicates and generalized winding numbers, cancels
//! opposite duplicate interfaces, de-pinches balanced non-manifold edges and
//! multi-fan vertices into a strict 2-manifold, and verifies watertightness
//! before returning [`BooleanPlan`]. [`apply_boolean`] writes the plan
//! atomically into A's transaction. The result stays on A with A's placement:
//! for world-space booleans the caller maps B into A's frame first (see
//! `fast-mesh-scene`).
//!
//! Inputs must be closed, 2-manifold, outward wound (`signed_volume > 0`), and
//! free of degenerate, duplicate, and self-intersecting triangles; anything
//! else returns [`OpsError::InvalidInput`]. Point touches, edge touches, and
//! coplanar area overlaps (flush or opposing faces) are handled directly by the
//! arrangement and de-pinching pipeline, producing closed, consistently wound,
//! 2-manifold outputs free of pinch vertices. Disjoint and nested inputs skip
//! sub-triangulation and classify whole connected components.
//!
//! New vertices take channel defaults; only positions and triangles cross over,
//! and A's UV channel is dropped when present ([`BooleanReport::uvs_dropped`]).

mod arrangement;
mod batch;
mod cdt;
mod classify;
mod perturbation;
mod voxel;
mod winding;

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::Rigid;
use fast_mesh_store::{Mesh, StoreError, Transaction, TriangleId, UV_CHANNEL, VertexId};

use crate::analyze::{AREA_EPSILON, analyze};
use crate::{OpsError, Result, allocation_allowed, filled, push, reserve};

pub use crate::cut::{CutOptions, CutOutput, CutReport, cut_by_plane};
pub use crate::offset::{
    OffsetAlgorithm, OffsetOptions, OffsetOptionsBuilder, OffsetReport, ShellOptions,
    ShellOptionsBuilder, ShellReport, offset, offset_with_bvh, shell,
};
pub use batch::{CsgNode, boolean_batch};

/// Backend strategy for CSG boolean evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BooleanMethod {
    /// Exact symbolic-perturbation mesh arrangement and CDT classification.
    #[default]
    Manifold,
    /// Volumetric BVH signed distance field boolean with Marching Cubes extraction.
    BvhSdf,
    /// Volumetric Felzenszwalb-Huttenlocher exact EDT boolean with Marching Cubes extraction.
    FhEdt,
}

/// Configuration options for [`boolean`] and [`apply_boolean_with_options`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BooleanOptions {
    /// CSG operation to perform.
    pub op: BooleanOp,
    /// Evaluation backend (`Manifold`, `BvhSdf`, or `FhEdt`).
    pub method: BooleanMethod,
    /// Coplanar and seam tolerance scale (local units). Default `1e-5`.
    pub coplanar_tolerance: f64,
    /// Whether to collapse short/needle triangles along boolean seam edges in `Manifold` mode.
    pub snap_slivers: bool,
    /// Optional explicit voxel size for volumetric modes (`BvhSdf` / `FhEdt`).
    pub voxel_pitch: Option<f32>,
    /// Optional upper bound on output triangle count via bounded quadric simplification.
    pub max_triangles: Option<u32>,
}

impl Default for BooleanOptions {
    fn default() -> Self {
        Self {
            op: BooleanOp::Union,
            method: BooleanMethod::Manifold,
            coplanar_tolerance: 1e-5,
            snap_slivers: true,
            voxel_pitch: None,
            max_triangles: None,
        }
    }
}

pub(super) fn validate_options(options: &BooleanOptions) -> Result<()> {
    if !options.coplanar_tolerance.is_finite() || options.coplanar_tolerance < 0.0 {
        return Err(OpsError::InvalidInput(
            "coplanar_tolerance must be finite and non-negative",
        ));
    }
    if let Some(pitch) = options.voxel_pitch
        && (!pitch.is_finite() || pitch <= 0.0)
    {
        return Err(OpsError::InvalidInput(
            "voxel_pitch must be finite and positive",
        ));
    }
    Ok(())
}

/// Which boolean to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    /// Whatever is in either mesh.
    Union,
    /// Whatever is in A but not in B.
    Subtract,
    /// Whatever is in both meshes.
    Intersect,
}

/// What [`apply_boolean`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BooleanReport {
    /// Alive triangles of the result.
    pub kept_triangles: usize,
    /// Triangles added (split pieces and the tool's kept pieces).
    pub triangles_added: usize,
    /// Mesh A's UVs were dropped: the result mixes split pieces and the tool's
    /// surface, which have no UVs in A's space. Undo restores them.
    pub uvs_dropped: bool,
}

/// The plan of [`boolean_plan`]: what to drop, rewrite and add on mesh A.
/// Built read-only from both meshes; [`apply_boolean`] writes it.
#[derive(Debug, Clone, Default)]
pub struct BooleanPlan {
    pub(super) drop_a: Vec<TriangleId>,
    pub(super) rewrite: Vec<(TriangleId, Vec<[PRef; 3]>)>,
    pub(super) fresh: Vec<[f32; 3]>,
    pub(super) tool: Vec<[PRef; 3]>,
}

/// A vertex of a planned triangle: a live vertex of A or a fresh position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PRef {
    Live(VertexId),
    Fresh(u32),
}

/// Plans the boolean `op` of A with B (B placed in A's frame by `xf`): the
/// result replaces A. Reads only; [`apply_boolean`] writes the plan.
///
/// `bvh_a` and `bvh_b` must be current for `a` and `b`.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when an input is not closed, manifold, outward
///   wound, or free of degenerate, duplicate, and self-intersecting triangles,
///   when a mapped tool position leaves `f32` range, or when an arrangement
///   cannot form a valid 2-manifold.
/// - [`OpsError::Query`] or [`OpsError::Store`] from BVH/transform checks or
///   fallible allocations. Nothing is written: the plan owns every allocation.
pub fn boolean_plan(
    a: &Mesh,
    bvh_a: &Bvh,
    b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    op: BooleanOp,
) -> Result<BooleanPlan> {
    check_input(a, bvh_a)?;
    check_input(b, bvh_b)?;
    if std::ptr::eq(a, b) {
        return Err(OpsError::InvalidInput("cannot combine a mesh with itself"));
    }
    xf.validate()?;

    // The tool in A's frame, slot for slot: positions mapped through `xf`,
    // triangles untouched.
    let tool = map_tool(b, xf)?;
    let tool_bvh = Bvh::build(&tool, &BuildOptions::default()).map_err(|e| match e {
        fast_mesh_bvh::BvhError::OutOfMemory => OpsError::Store(StoreError::OutOfMemory),
        _ => OpsError::InvalidInput("boolean tool BVH failed to build"),
    })?;
    if *xf != Rigid::IDENTITY {
        check_input(&tool, &tool_bvh)?;
    }

    let arr = arrangement::build_arrangement(a, bvh_a, &tool, &tool_bvh)?;
    classify::classify_and_build_plan(a, bvh_a, &tool, &tool_bvh, arr, op)
}

/// Applies `plan` to A's transaction: drops, rewrites, and adds triangles and
/// removes orphaned vertices. A failure rolls the transaction back, so the call
/// stays atomic through [`Scene::edit`](https://docs.rs/fast-mesh-scene)-style
/// callers.
pub fn apply_boolean(tx: &mut Transaction, plan: &BooleanPlan) -> Result<BooleanReport> {
    let (report, _) = apply_boolean_core(tx, plan, false)?;
    Ok(report)
}

fn apply_boolean_core(
    tx: &mut Transaction,
    plan: &BooleanPlan,
    track_seams: bool,
) -> Result<(BooleanReport, Vec<bool>)> {
    // Pre-compute orphaned live vertices and reserve all temporary buffers
    // BEFORE issuing any write to `tx`, keeping OOM failures strictly atomic.
    let (unused, mut set_ids, mut set_tris, mut add_tris, total_added, v_slots_before) = {
        let mesh = tx.mesh();
        let v_slots = mesh.vertex_slots();
        let t_slots = mesh.triangle_slots();
        let mut used = filled(v_slots, false)?;
        let mut touched_t = filled(t_slots, false)?;

        for &t in &plan.drop_a {
            if (t as usize) < t_slots {
                touched_t[t as usize] = true;
            }
        }
        for &(t, ref pieces) in &plan.rewrite {
            if (t as usize) < t_slots {
                touched_t[t as usize] = true;
            }
            for tri in pieces {
                for &r in tri {
                    if let PRef::Live(v) = r
                        && (v as usize) < v_slots
                    {
                        used[v as usize] = true;
                    }
                }
            }
        }
        for tri in &plan.tool {
            for &r in tri {
                if let PRef::Live(v) = r
                    && (v as usize) < v_slots
                {
                    used[v as usize] = true;
                }
            }
        }
        for t in 0..t_slots as TriangleId {
            if !touched_t[t as usize]
                && let Some(tri) = mesh.triangle(t)
            {
                for v in tri {
                    used[v as usize] = true;
                }
            }
        }

        let mut unused: Vec<VertexId> = Vec::new();
        for v in 0..v_slots as VertexId {
            if mesh.is_vertex_alive(v) && !used[v as usize] {
                push(&mut unused, v)?;
            }
        }

        let extra_rewrite: usize = plan
            .rewrite
            .iter()
            .map(|(_, pieces)| pieces.len().saturating_sub(1))
            .sum();
        let total_added = extra_rewrite.saturating_add(plan.tool.len());

        let mut set_ids: Vec<TriangleId> = Vec::new();
        let mut set_tris: Vec<[VertexId; 3]> = Vec::new();
        let mut add_tris: Vec<[VertexId; 3]> = Vec::new();
        reserve(&mut set_ids, plan.rewrite.len())?;
        reserve(&mut set_tris, plan.rewrite.len())?;
        reserve(&mut add_tris, total_added)?;

        (unused, set_ids, set_tris, add_tris, total_added, v_slots)
    };

    let mut seam_verts: Vec<bool> = if track_seams {
        filled(v_slots_before + plan.fresh.len(), false)?
    } else {
        Vec::new()
    };

    let uvs_dropped = tx.mesh().uvs().is_some();
    if uvs_dropped {
        tx.drop_channel(UV_CHANNEL)?;
    }

    let fresh = if plan.fresh.is_empty() {
        Vec::new()
    } else {
        tx.add_vertices(&plan.fresh)?
    };
    if track_seams {
        let needed = fresh
            .iter()
            .copied()
            .fold(v_slots_before, |acc, v| acc.max(v as usize + 1));
        if needed > seam_verts.len() {
            let grow = needed - seam_verts.len();
            reserve(&mut seam_verts, grow)?;
            seam_verts.resize(needed, false);
        }
        for &v in &fresh {
            seam_verts[v as usize] = true;
        }
        for tri in &plan.tool {
            for &r in tri {
                if let PRef::Live(v) = r
                    && (v as usize) < seam_verts.len()
                {
                    seam_verts[v as usize] = true;
                }
            }
        }
    }
    let at = |r: PRef| match r {
        PRef::Live(v) => v,
        PRef::Fresh(i) => fresh[i as usize],
    };

    for (t, pieces) in &plan.rewrite {
        set_ids.push(*t);
        set_tris.push(pieces[0].map(at));
        for piece in &pieces[1..] {
            add_tris.push(piece.map(at));
        }
    }
    for tri in &plan.tool {
        add_tris.push(tri.map(at));
    }

    if !set_ids.is_empty() {
        tx.set_triangles(&set_ids, &set_tris)?;
    }
    if !plan.drop_a.is_empty() {
        tx.remove_triangles(&plan.drop_a)?;
    }
    if !add_tris.is_empty() {
        tx.add_triangles(&add_tris)?;
    }
    if !unused.is_empty() {
        tx.remove_vertices(&unused)?;
    }

    Ok((
        BooleanReport {
            kept_triangles: tx.mesh().triangle_count(),
            triangles_added: total_added,
            uvs_dropped,
        },
        seam_verts,
    ))
}

/// Computes CSG boolean `options.op` between `mesh_a` and `mesh_b` (with `mesh_b` placed
/// in `mesh_a`'s frame via `b_in_a`), returning a new mesh and [`BooleanReport`].
pub fn boolean(
    mesh_a: &Mesh,
    mesh_b: &Mesh,
    b_in_a: Rigid,
    options: &BooleanOptions,
) -> Result<(Mesh, BooleanReport)> {
    validate_options(options)?;
    b_in_a.validate()?;

    match options.method {
        BooleanMethod::Manifold => {
            allocation_allowed()?;
            let mut out = clone_mesh_fallible(mesh_a)?;
            let mut tx = out.begin_edit();
            let report = apply_boolean_with_options(&mut tx, mesh_b, b_in_a, options)?;
            let _ = tx.commit()?;
            Ok((out, report))
        }
        BooleanMethod::BvhSdf | BooleanMethod::FhEdt => {
            let (mut out, mut report) = voxel::boolean_voxel(mesh_a, mesh_b, &b_in_a, options)?;
            if let Some(cap) = options.max_triangles {
                cap_mesh_triangles(&mut out, cap)?;
                report.kept_triangles = out.triangle_count();
            }
            Ok((out, report))
        }
    }
}

/// Applies CSG boolean `options.op` between `tx.mesh()` and `mesh_b` inside an active
/// [`Transaction`], optionally snapping seam slivers and simplifying to `max_triangles`.
pub fn apply_boolean_with_options(
    tx: &mut Transaction<'_>,
    mesh_b: &Mesh,
    b_in_a: Rigid,
    options: &BooleanOptions,
) -> Result<BooleanReport> {
    validate_options(options)?;
    b_in_a.validate()?;

    match options.method {
        BooleanMethod::Manifold => {
            let plan = {
                let mesh_a = tx.mesh();
                let bvh_a = Bvh::build(mesh_a, &BuildOptions::default())?;
                let bvh_b = Bvh::build(mesh_b, &BuildOptions::default())?;
                boolean_plan(mesh_a, &bvh_a, mesh_b, &bvh_b, &b_in_a, options.op)?
            };
            apply_manifold_plan_with_options(tx, &plan, options)
        }
        BooleanMethod::BvhSdf | BooleanMethod::FhEdt => {
            let (mut out, mut report) = voxel::boolean_voxel(tx.mesh(), mesh_b, &b_in_a, options)?;
            if let Some(cap) = options.max_triangles {
                cap_mesh_triangles(&mut out, cap)?;
                report.kept_triangles = out.triangle_count();
            }
            replace_tx_with_mesh(tx, &out)?;
            Ok(report)
        }
    }
}

fn apply_manifold_plan_with_options(
    tx: &mut Transaction<'_>,
    plan: &BooleanPlan,
    options: &BooleanOptions,
) -> Result<BooleanReport> {
    if !options.snap_slivers && options.max_triangles.is_none() {
        return apply_boolean_core(tx, plan, false).map(|(r, _)| r);
    }
    let mut scratch = clone_mesh_fallible(tx.mesh())?;
    let mut report = {
        let mut stx = scratch.begin_edit();
        let (mut report, seam_verts) = apply_boolean_core(&mut stx, plan, options.snap_slivers)?;
        if options.snap_slivers {
            snap_seam_slivers(&mut stx, &seam_verts, options.coplanar_tolerance)?;
            report.kept_triangles = stx.mesh().triangle_count();
        }
        if let Some(cap) = options.max_triangles {
            cap_tx_triangles(&mut stx, cap)?;
            report.kept_triangles = stx.mesh().triangle_count();
        }
        let _ = stx.commit()?;
        report
    };
    replace_tx_with_mesh(tx, &scratch)?;
    report.kept_triangles = tx.mesh().triangle_count();
    Ok(report)
}

pub(super) fn cap_mesh_triangles(mesh: &mut Mesh, cap: u32) -> Result<()> {
    if mesh.triangle_count() <= cap as usize {
        return Ok(());
    }
    let mut tx = mesh.begin_edit();
    cap_tx_triangles(&mut tx, cap)?;
    let _ = tx.commit()?;
    Ok(())
}

fn cap_tx_triangles(tx: &mut Transaction<'_>, cap: u32) -> Result<()> {
    if tx.mesh().triangle_count() <= cap as usize {
        return Ok(());
    }
    let _ = crate::simplify::simplify(
        tx,
        &crate::simplify::SimplifyOptions {
            target_triangles: cap as usize,
            preserve_boundary: true,
        },
    )?;
    Ok(())
}

fn replace_tx_with_mesh(tx: &mut Transaction<'_>, src: &Mesh) -> Result<()> {
    let mut drop_tris: Vec<TriangleId> = Vec::new();
    let mut drop_verts: Vec<VertexId> = Vec::new();
    {
        let mesh = tx.mesh();
        reserve(&mut drop_tris, mesh.triangle_count())?;
        for t in 0..mesh.triangle_slots() as TriangleId {
            if mesh.is_triangle_alive(t) {
                push(&mut drop_tris, t)?;
            }
        }
        reserve(&mut drop_verts, mesh.vertex_count())?;
        for v in 0..mesh.vertex_slots() as VertexId {
            if mesh.is_vertex_alive(v) {
                push(&mut drop_verts, v)?;
            }
        }
    }

    let mut src_remap: Vec<u32> = filled(src.vertex_slots(), u32::MAX)?;
    let mut add_pos: Vec<[f32; 3]> = Vec::new();
    let mut add_tris: Vec<[u32; 3]> = Vec::new();
    let src_pos = src.positions();
    for t in 0..src.triangle_slots() as TriangleId {
        let Some(tri) = src.triangle(t) else {
            continue;
        };
        let mut mapped = [0u32; 3];
        for (k, v) in tri.into_iter().enumerate() {
            let idx = if src_remap[v as usize] != u32::MAX {
                src_remap[v as usize]
            } else {
                let next = u32::try_from(add_pos.len())
                    .map_err(|_| OpsError::Store(StoreError::OutOfMemory))?;
                push(&mut add_pos, src_pos[v as usize])?;
                src_remap[v as usize] = next;
                next
            };
            mapped[k] = idx;
        }
        push(&mut add_tris, mapped)?;
    }

    if tx.mesh().uvs().is_some() {
        tx.drop_channel(UV_CHANNEL)?;
    }
    if !drop_tris.is_empty() {
        tx.remove_triangles(&drop_tris)?;
    }
    if !drop_verts.is_empty() {
        tx.remove_vertices(&drop_verts)?;
    }
    if !add_pos.is_empty() {
        let added_verts = tx.add_vertices(&add_pos)?;
        for tri in &mut add_tris {
            *tri = tri.map(|i| added_verts[i as usize]);
        }
        if !add_tris.is_empty() {
            tx.add_triangles(&add_tris)?;
        }
    }
    Ok(())
}

/// Rejects inputs no boolean can consume.
pub(super) fn check_input(mesh: &Mesh, bvh: &Bvh) -> Result<()> {
    let options = crate::AnalyzeOptions {
        weld_tolerance: 0.0,
        self_intersection: Some(bvh),
        max_listed: 1,
    };
    let r = analyze(mesh, &options)?;
    if !r.loops.is_empty() {
        return Err(OpsError::InvalidInput("boolean needs closed meshes"));
    }
    if r.non_manifold_edges > 0 || !r.pinch_vertices.is_empty() {
        return Err(OpsError::InvalidInput("boolean needs manifold meshes"));
    }
    if r.degenerate > 0 {
        return Err(OpsError::InvalidInput(
            "boolean input has degenerate triangles",
        ));
    }
    if r.duplicate > 0 {
        return Err(OpsError::InvalidInput(
            "boolean input has duplicate triangles",
        ));
    }
    if r.orientation_conflicts > 0 {
        return Err(OpsError::InvalidInput(
            "boolean input is inconsistently wound",
        ));
    }
    if r.self_intersects == Some(true) {
        return Err(OpsError::InvalidInput("boolean input self-intersects"));
    }
    if mesh.triangle_count() > 0 && winding::signed_volume6(mesh) <= 0.0 {
        return Err(OpsError::InvalidInput("boolean input is wound inward"));
    }
    Ok(())
}

/// Copies a mesh slot-for-slot (positions, triangles, dead slots, and UVs) with fallible allocations.
fn clone_mesh_fallible(src: &Mesh) -> Result<Mesh> {
    let mut positions: Vec<f32> = Vec::new();
    reserve(&mut positions, src.vertex_slots() * 3)?;
    for p in src.positions() {
        positions.extend_from_slice(p);
    }
    let mut indices: Vec<u32> = Vec::new();
    reserve(&mut indices, src.triangle_slots() * 3)?;
    for t in src.triangles() {
        indices.extend_from_slice(t);
    }
    let mut out = Mesh::from_buffers(&positions, &indices)?;
    if let Some(uvs) = src.uvs() {
        out.add_uvs(uvs)?;
    }
    let mut dead_tris: Vec<TriangleId> = Vec::new();
    for t in 0..src.triangle_slots() as TriangleId {
        if src.triangle(t).is_none() {
            push(&mut dead_tris, t)?;
        }
    }
    let mut dead_verts: Vec<VertexId> = Vec::new();
    for v in 0..src.vertex_slots() as VertexId {
        if !src.is_vertex_alive(v) {
            push(&mut dead_verts, v)?;
        }
    }
    if !dead_tris.is_empty() || !dead_verts.is_empty() {
        let mut tx = out.begin_edit();
        if !dead_tris.is_empty() {
            tx.remove_triangles(&dead_tris)?;
        }
        if !dead_verts.is_empty() {
            tx.remove_vertices(&dead_verts)?;
        }
        let _ = tx.commit()?;
    }
    Ok(out)
}

/// Mesh B mapped into A's frame, slot for slot.
pub(super) fn map_tool(b: &Mesh, xf: &Rigid) -> Result<Mesh> {
    let mut positions: Vec<f32> = Vec::new();
    reserve(&mut positions, b.vertex_slots() * 3)?;
    for p in b.positions() {
        let q = xf.apply(*p);
        let f = [q[0] as f32, q[1] as f32, q[2] as f32];
        if !f.iter().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput("boolean tool leaves f32 range"));
        }
        positions.extend_from_slice(&f);
    }
    let mut indices: Vec<u32> = Vec::new();
    reserve(&mut indices, b.triangle_slots() * 3)?;
    for t in b.triangles() {
        indices.extend_from_slice(t);
    }
    let mut mapped = Mesh::from_buffers(&positions, &indices)?;
    // `from_buffers` wakes every slot; put the dead ones back to sleep so
    // queries see the same triangles as `b`.
    let mut dead_tris: Vec<TriangleId> = Vec::new();
    for t in 0..b.triangle_slots() as TriangleId {
        if b.triangle(t).is_none() {
            push(&mut dead_tris, t)?;
        }
    }
    let mut dead_verts: Vec<VertexId> = Vec::new();
    for v in 0..b.vertex_slots() as VertexId {
        if !b.is_vertex_alive(v) {
            push(&mut dead_verts, v)?;
        }
    }
    if !dead_tris.is_empty() || !dead_verts.is_empty() {
        let mut tx = mapped.begin_edit();
        if !dead_tris.is_empty() {
            tx.remove_triangles(&dead_tris)?;
        }
        if !dead_verts.is_empty() {
            tx.remove_vertices(&dead_verts)?;
        }
        let _ = tx.commit()?;
    }
    Ok(mapped)
}

#[inline]
fn sub3_f64(a: [f32; 3], b: [f32; 3]) -> [f64; 3] {
    [
        f64::from(a[0]) - f64::from(b[0]),
        f64::from(a[1]) - f64::from(b[1]),
        f64::from(a[2]) - f64::from(b[2]),
    ]
}

#[inline]
fn dot3_f64(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3_f64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn tri_geom(pos: &[[f32; 3]], tri: [VertexId; 3]) -> Option<([f64; 3], f64, f64)> {
    let [pa, pb, pc] = tri.map(|v| pos[v as usize]);
    let ab = sub3_f64(pb, pa);
    let ac = sub3_f64(pc, pa);
    let bc = sub3_f64(pc, pb);
    let max_len2 = dot3_f64(ab, ab).max(dot3_f64(ac, ac)).max(dot3_f64(bc, bc));
    let n = cross3_f64(ab, ac);
    let n_len = dot3_f64(n, n).sqrt();
    if n_len <= 0.0 || !n_len.is_finite() {
        return None;
    }
    let area = 0.5 * n_len;
    Some(([n[0] / n_len, n[1] / n_len, n[2] / n_len], area, max_len2))
}

fn sorted_triple(mut tri: [VertexId; 3]) -> [VertexId; 3] {
    tri.sort_unstable();
    tri
}

fn is_closed_manifold_vertex(
    vert_tris: &[Vec<u32>],
    tris: &[[VertexId; 3]],
    t_alive: &[bool],
    v: VertexId,
    links: &mut Vec<(VertexId, VertexId)>,
) -> bool {
    links.clear();
    for &t in &vert_tris[v as usize] {
        if !t_alive[t as usize] {
            continue;
        }
        let tri = tris[t as usize];
        let Some(pos_v) = tri.iter().position(|&c| c == v) else {
            return false;
        };
        let next = tri[(pos_v + 1) % 3];
        let prev = tri[(pos_v + 2) % 3];
        if next == prev || next == v || prev == v {
            return false;
        }
        links.push((next, prev));
    }
    let fan_len = links.len();
    if fan_len < 3 {
        return false;
    }
    let start = links[0].0;
    let mut cur = links[0].1;
    let mut visited = 1usize;
    while cur != start {
        let mut found = None;
        for &(a, b) in links.iter().skip(1) {
            if a == cur {
                if found.is_some() {
                    return false;
                }
                found = Some(b);
            }
        }
        let Some(nxt) = found else {
            return false;
        };
        cur = nxt;
        visited += 1;
        if visited > fan_len {
            return false;
        }
    }
    visited == fan_len
}

fn fill_neighbors(
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
        for &c in &tris[t as usize] {
            if c != v && c != exclude {
                push(out, c)?;
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(())
}

fn edge_incident_tris(
    vert_tris: &[Vec<u32>],
    tris: &[[VertexId; 3]],
    t_alive: &[bool],
    u: VertexId,
    v: VertexId,
) -> Option<[u32; 2]> {
    let mut found = [u32::MAX; 2];
    let mut count = 0usize;
    for &t in &vert_tris[u as usize] {
        if !t_alive[t as usize] {
            continue;
        }
        if tris[t as usize].contains(&v) {
            if count >= 2 {
                return None;
            }
            found[count] = t;
            count += 1;
        }
    }
    (count == 2).then_some(found)
}

fn opp_vertex(tri: [VertexId; 3], u: VertexId, v: VertexId) -> Option<VertexId> {
    tri.into_iter().find(|&c| c != u && c != v)
}

#[allow(clippy::too_many_arguments)]
fn try_collapse_seam_edge(
    pos: &mut [[f32; 3]],
    pos_dirty: &mut [bool],
    tris: &mut [[VertexId; 3]],
    t_alive: &mut [bool],
    t_dirty: &mut [bool],
    v_removed: &mut [bool],
    vert_tris: &mut [Vec<u32>],
    is_seam_vert: &[bool],
    u: VertexId,
    v: VertexId,
    links: &mut Vec<(VertexId, VertexId)>,
    nb_u: &mut Vec<VertexId>,
    nb_v: &mut Vec<VertexId>,
    affected: &mut Vec<u32>,
    keys: &mut Vec<[VertexId; 3]>,
) -> Result<bool> {
    let Some([t0, t1]) = edge_incident_tris(vert_tris, tris, t_alive, u, v) else {
        return Ok(false);
    };
    let (Some(opp0), Some(opp1)) = (
        opp_vertex(tris[t0 as usize], u, v),
        opp_vertex(tris[t1 as usize], u, v),
    ) else {
        return Ok(false);
    };
    if opp0 == opp1 {
        return Ok(false);
    }

    if !is_closed_manifold_vertex(vert_tris, tris, t_alive, u, links)
        || !is_closed_manifold_vertex(vert_tris, tris, t_alive, v, links)
    {
        return Ok(false);
    }

    fill_neighbors(vert_tris, tris, t_alive, u, v, nb_u)?;
    fill_neighbors(vert_tris, tris, t_alive, v, u, nb_v)?;
    let mut common = 0usize;
    for &w in nb_u.iter() {
        if nb_v.binary_search(&w).is_ok() {
            if w != opp0 && w != opp1 {
                return Ok(false);
            }
            common += 1;
        }
    }
    if common != 2 {
        return Ok(false);
    }

    affected.clear();
    for &t in vert_tris[u as usize].iter().chain(&vert_tris[v as usize]) {
        if t_alive[t as usize] && t != t0 && t != t1 {
            push(affected, t)?;
        }
    }
    affected.sort_unstable();
    affected.dedup();
    if affected.is_empty() {
        return Ok(false);
    }

    let seam_u = (u as usize) < is_seam_vert.len() && is_seam_vert[u as usize];
    let seam_v = (v as usize) < is_seam_vert.len() && is_seam_vert[v as usize];
    let (keep, drop) = if !seam_u && seam_v {
        (u, v)
    } else if !seam_v && seam_u {
        (v, u)
    } else if u < v {
        (u, v)
    } else {
        (v, u)
    };

    keys.clear();
    reserve(keys, affected.len())?;
    for &t in affected.iter() {
        let mut tri = tris[t as usize];
        for c in &mut tri {
            if *c == drop {
                *c = keep;
            }
        }
        if tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0] {
            return Ok(false);
        }
        keys.push(sorted_triple(tri));
    }
    keys.sort_unstable();
    if keys.windows(2).any(|w| w[0] == w[1]) {
        return Ok(false);
    }

    let p_keep = pos[keep as usize];
    let p_drop = pos[drop as usize];
    let mid = [
        0.5 * (p_keep[0] + p_drop[0]),
        0.5 * (p_keep[1] + p_drop[1]),
        0.5 * (p_keep[2] + p_drop[2]),
    ];
    let candidates = [(p_keep, false), (mid, true), (p_drop, true)];
    let mut chosen: Option<([f32; 3], bool)> = None;

    for &(cand_p, changed_pos) in &candidates {
        if !cand_p.iter().all(|c| c.is_finite()) {
            continue;
        }
        let mut ok = true;
        for &t in affected.iter() {
            let tri_old = tris[t as usize];
            if !changed_pos && !tri_old.contains(&drop) {
                continue;
            }
            let Some((n_old, _, _)) = tri_geom(pos, tri_old) else {
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
            let after = tri_geom(pos, tri_new);
            pos[keep as usize] = p_keep;
            match after {
                Some((n_new, area_new, len2_new))
                    if dot3_f64(n_old, n_new) > 0.5 && area_new > AREA_EPSILON * len2_new => {}
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            chosen = Some((cand_p, changed_pos));
            break;
        }
    }

    let Some((new_p, changed_pos)) = chosen else {
        return Ok(false);
    };

    if changed_pos {
        pos[keep as usize] = new_p;
        pos_dirty[keep as usize] = true;
    }
    t_alive[t0 as usize] = false;
    t_alive[t1 as usize] = false;
    v_removed[drop as usize] = true;
    for &dead_t in &[t0, t1] {
        for vc in tris[dead_t as usize] {
            if let Some(idx) = vert_tris[vc as usize].iter().position(|&x| x == dead_t) {
                vert_tris[vc as usize].swap_remove(idx);
            }
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
            t_dirty[t as usize] = true;
            if !vert_tris[keep as usize].contains(&t) {
                push(&mut vert_tris[keep as usize], t)?;
            }
        }
    }
    vert_tris[drop as usize].clear();

    Ok(true)
}

/// Collapses needle and short triangles along boolean seam edges when strictly
/// manifold-preserving (closed degree-2 vertices, link condition, no fold or
/// duplicate triangles).
fn snap_seam_slivers(
    tx: &mut Transaction,
    is_seam_vert: &[bool],
    coplanar_tolerance: f64,
) -> Result<()> {
    let mesh = tx.mesh();
    let v_slots = mesh.vertex_slots();
    let t_slots = mesh.triangle_slots();
    if mesh.triangle_count() == 0 {
        return Ok(());
    }

    let mut pos: Vec<[f32; 3]> = Vec::new();
    reserve(&mut pos, v_slots)?;
    pos.extend_from_slice(mesh.positions());
    let mut pos_dirty = filled(v_slots, false)?;
    let mut tris: Vec<[VertexId; 3]> = filled(t_slots, [0; 3])?;
    let mut t_alive = filled(t_slots, false)?;
    let mut t_dirty = filled(t_slots, false)?;
    let mut v_removed = filled(v_slots, false)?;
    let mut vert_tris: Vec<Vec<u32>> = filled(v_slots, Vec::new())?;

    let mut bmin = [f64::INFINITY; 3];
    let mut bmax = [f64::NEG_INFINITY; 3];
    for t in 0..t_slots {
        let Some(tri) = mesh.triangle(t as TriangleId) else {
            continue;
        };
        tris[t] = tri;
        t_alive[t] = true;
        for v in tri {
            push(&mut vert_tris[v as usize], t as u32)?;
            let p = pos[v as usize];
            for k in 0..3 {
                let c = f64::from(p[k]);
                bmin[k] = bmin[k].min(c);
                bmax[k] = bmax[k].max(c);
            }
        }
    }

    let diag =
        ((bmax[0] - bmin[0]).powi(2) + (bmax[1] - bmin[1]).powi(2) + (bmax[2] - bmin[2]).powi(2))
            .sqrt();
    if !diag.is_finite() || diag <= 0.0 {
        return Ok(());
    }

    let short_edge_tol = (1e-6 * diag).max(coplanar_tolerance).min(5e-3 * diag);
    let short_edge_tol2 = short_edge_tol * short_edge_tol;
    let sliver_edge_cap = (5e-3 * diag).max(10.0 * short_edge_tol).min(2e-2 * diag);
    let sliver_edge_cap2 = sliver_edge_cap * sliver_edge_cap;
    const SLIVER_AREA_RATIO: f64 = 1e-4;

    let mut candidates: Vec<(f64, VertexId, VertexId)> = Vec::new();
    for t in 0..t_slots {
        if !t_alive[t] {
            continue;
        }
        let tri = tris[t];
        let e_len2 = [0, 1, 2].map(|i| {
            let d = sub3_f64(pos[tri[i] as usize], pos[tri[(i + 1) % 3] as usize]);
            dot3_f64(d, d)
        });
        let max_len2 = e_len2[0].max(e_len2[1]).max(e_len2[2]);
        let min_idx = if e_len2[1] < e_len2[0] && e_len2[1] <= e_len2[2] {
            1
        } else if e_len2[2] < e_len2[0] && e_len2[2] < e_len2[1] {
            2
        } else {
            0
        };
        let is_sliver = match tri_geom(&pos, tri) {
            Some((_, area, _)) => area <= SLIVER_AREA_RATIO * max_len2,
            None => true,
        };
        for i in 0..3 {
            let u = tri[i];
            let v = tri[(i + 1) % 3];
            let seam_u = (u as usize) < is_seam_vert.len() && is_seam_vert[u as usize];
            let seam_v = (v as usize) < is_seam_vert.len() && is_seam_vert[v as usize];
            if !seam_u && !seam_v {
                continue;
            }
            let l2 = e_len2[i];
            let want =
                l2 <= short_edge_tol2 || (is_sliver && i == min_idx && l2 <= sliver_edge_cap2);
            if want {
                let (lo, hi) = if u < v { (u, v) } else { (v, u) };
                push(&mut candidates, (l2, lo, hi))?;
            }
        }
    }

    if candidates.is_empty() {
        return Ok(());
    }
    candidates.sort_unstable_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    let mut links: Vec<(VertexId, VertexId)> = Vec::new();
    let mut nb_u: Vec<VertexId> = Vec::new();
    let mut nb_v: Vec<VertexId> = Vec::new();
    let mut affected: Vec<u32> = Vec::new();
    let mut keys: Vec<[VertexId; 3]> = Vec::new();
    let mut any_collapsed = false;

    for &(_, lo, hi) in &candidates {
        if v_removed[lo as usize] || v_removed[hi as usize] {
            continue;
        }
        if try_collapse_seam_edge(
            &mut pos,
            &mut pos_dirty,
            &mut tris,
            &mut t_alive,
            &mut t_dirty,
            &mut v_removed,
            &mut vert_tris,
            is_seam_vert,
            lo,
            hi,
            &mut links,
            &mut nb_u,
            &mut nb_v,
            &mut affected,
            &mut keys,
        )? {
            any_collapsed = true;
        }
    }

    if !any_collapsed {
        return Ok(());
    }

    let mut mod_v_ids: Vec<VertexId> = Vec::new();
    let mut mod_v_pos: Vec<[f32; 3]> = Vec::new();
    let mut rem_v_ids: Vec<VertexId> = Vec::new();
    for v in 0..v_slots {
        if v_removed[v] {
            push(&mut rem_v_ids, v as VertexId)?;
        } else if pos_dirty[v] {
            push(&mut mod_v_ids, v as VertexId)?;
            push(&mut mod_v_pos, pos[v])?;
        }
    }

    let mut mod_t_ids: Vec<TriangleId> = Vec::new();
    let mut mod_t_tris: Vec<[VertexId; 3]> = Vec::new();
    let mut rem_t_ids: Vec<TriangleId> = Vec::new();
    for t in 0..t_slots {
        if !t_alive[t] && mesh.is_triangle_alive(t as TriangleId) {
            push(&mut rem_t_ids, t as TriangleId)?;
        } else if t_alive[t] && t_dirty[t] {
            push(&mut mod_t_ids, t as TriangleId)?;
            push(&mut mod_t_tris, tris[t])?;
        }
    }

    if !mod_v_ids.is_empty() {
        tx.set_positions(&mod_v_ids, &mod_v_pos)?;
    }
    if !mod_t_ids.is_empty() {
        tx.set_triangles(&mod_t_ids, &mod_t_tris)?;
    }
    if !rem_t_ids.is_empty() {
        tx.remove_triangles(&rem_t_ids)?;
    }
    if !rem_v_ids.is_empty() {
        tx.remove_vertices(&rem_v_ids)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{box_mesh, edit, grid, uv_sphere, with_tetra};
    use fast_mesh_query::Rigid;

    /// A closed unit cube, outward wound.
    fn cube() -> Mesh {
        box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    }

    /// Six times the signed volume: positive for outward-wound meshes.
    fn volume6(mesh: &Mesh) -> f64 {
        let mut v = 0.0;
        for t in 0..mesh.triangle_slots() as TriangleId {
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };
            let [a, b, c] = tri.map(|x| mesh.position(x).unwrap().map(f64::from));
            v += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        v
    }

    fn closed(mesh: &Mesh) -> bool {
        crate::boundary_loops(mesh).unwrap().is_empty()
    }

    fn run(a: &mut Mesh, b: &Mesh, op: BooleanOp) -> BooleanReport {
        let bvh_a = Bvh::build(a, &BuildOptions::default()).unwrap();
        let bvh_b = Bvh::build(b, &BuildOptions::default()).unwrap();
        let (report, _) = edit(a, |tx| {
            let plan = boolean_plan(tx.mesh(), &bvh_a, b, &bvh_b, &Rigid::IDENTITY, op)?;
            apply_boolean(tx, &plan)
        });
        report
    }

    #[test]
    fn union_overlapping_cubes() {
        // B shifted half a unit in x: overlap 0.5 x 1 x 1.
        let mut a = cube();
        let b = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);
        let report = run(&mut a, &b, BooleanOp::Union);
        assert!(closed(&a), "union must be closed");
        assert!(
            (volume6(&a) - 6.0 * 1.5).abs() < 1e-3,
            "vol {}",
            volume6(&a)
        );
        assert_eq!(report.kept_triangles, a.triangle_count());
    }

    #[test]
    fn subtract_overlapping_cubes() {
        let mut a = cube();
        let b = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);
        run(&mut a, &b, BooleanOp::Subtract);
        assert!(closed(&a), "subtract must be closed");
        assert!(
            (volume6(&a) - 6.0 * 0.5).abs() < 1e-3,
            "vol {}",
            volume6(&a)
        );
    }

    #[test]
    fn intersect_overlapping_cubes() {
        let mut a = cube();
        let b = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);
        run(&mut a, &b, BooleanOp::Intersect);
        assert!(closed(&a), "intersect must be closed");
        assert!(
            (volume6(&a) - 6.0 * 0.5).abs() < 1e-3,
            "vol {}",
            volume6(&a)
        );
    }

    #[test]
    fn asymmetric_overlap_needs_interior_points() {
        // B offset in every axis: B's edges pierce A's faces away from
        // their diagonals, so triangles gain interior segment endpoints.
        // Overlap 0.5 x 0.7 x 0.3.
        let mut a = cube();
        let b = box_mesh([0.5, 0.3, 0.7], [1.5, 1.3, 1.7]);
        run(&mut a, &b, BooleanOp::Subtract);
        assert!(closed(&a), "subtract must be closed");
        assert!(
            (volume6(&a) - 6.0 * 0.895).abs() < 1e-3,
            "vol {}",
            volume6(&a)
        );
        let mut u = cube();
        run(&mut u, &b, BooleanOp::Union);
        assert!(closed(&u), "union must be closed");
        assert!(
            (volume6(&u) - 6.0 * 1.895).abs() < 1e-3,
            "vol {}",
            volume6(&u)
        );
    }

    #[test]
    fn stacked_cubes_share_a_face() {
        // Coplanar area overlap: the shared face drops out of the union.
        let mut a = cube();
        let b = box_mesh([0.0, 0.0, 1.0], [1.0, 1.0, 2.0]);
        run(&mut a, &b, BooleanOp::Union);
        assert!(closed(&a), "union must be closed");
        assert!((volume6(&a) - 12.0).abs() < 1e-3, "vol {}", volume6(&a));
        assert_eq!(a.triangle_count(), 20);
    }

    #[test]
    fn nested_spheres_classify_whole() {
        let outer = uv_sphere(8, 12, 1.0);
        let inner = uv_sphere(6, 10, 0.5);
        let (vo, vi) = (volume6(&outer), volume6(&inner));
        let mut u = uv_sphere(8, 12, 1.0);
        run(&mut u, &inner, BooleanOp::Union);
        assert!(closed(&u));
        assert!((volume6(&u) - vo).abs() < 1e-3 * vo.abs());
        let mut i = uv_sphere(8, 12, 1.0);
        run(&mut i, &inner, BooleanOp::Intersect);
        assert!(closed(&i));
        assert!((volume6(&i) - vi).abs() < 1e-3 * vi.abs());
        let mut s = uv_sphere(8, 12, 1.0);
        run(&mut s, &inner, BooleanOp::Subtract);
        assert!(closed(&s));
        assert!((volume6(&s) - (vo - vi)).abs() < 1e-3 * vo.abs());
    }

    #[test]
    fn overlapping_spheres() {
        // Two unit spheres a unit apart: every triangle near the intersection
        // is split, on both sides.
        let sphere_at = |x: f32| {
            let (mut positions, triangles) = crate::test_util::parts(&uv_sphere(8, 12, 1.0));
            for p in positions.as_chunks_mut::<3>().0 {
                p[0] += x;
            }
            let indices: Vec<u32> = triangles.iter().flatten().copied().collect();
            Mesh::from_buffers(&positions, &indices).unwrap()
        };
        // Analytic lens volume for r = 1, d = 1: 5π/12 ≈ 1.3090 (×6 below).
        let lens = 6.0 * 5.0 * std::f64::consts::PI / 12.0;
        let single = volume6(&sphere_at(0.0));
        let mut u = sphere_at(0.0);
        run(&mut u, &sphere_at(1.0), BooleanOp::Union);
        assert!(closed(&u), "union must be closed");
        assert!(
            (volume6(&u) - (2.0 * single - lens)).abs() < 0.05 * single.abs(),
            "union vol {}",
            volume6(&u)
        );
        assert!((volume6(&u) - (2.0 * single - lens)).abs() < 0.05 * single.abs());
        let mut i = sphere_at(0.0);
        run(&mut i, &sphere_at(1.0), BooleanOp::Intersect);
        assert!(closed(&i), "intersect must be closed");
        assert!((volume6(&i) - lens).abs() < 0.05 * single.abs());
        let mut s = sphere_at(0.0);
        run(&mut s, &sphere_at(1.0), BooleanOp::Subtract);
        assert!(closed(&s), "subtract must be closed");
        assert!((volume6(&s) - (single - lens)).abs() < 0.05 * single.abs());
    }

    #[test]
    fn disjoint_cubes() {
        let b = box_mesh([5.0, 0.0, 0.0], [6.0, 1.0, 1.0]);
        let mut u = cube();
        run(&mut u, &b, BooleanOp::Union);
        assert!(closed(&u));
        assert_eq!(u.triangle_count(), 24);
        assert!((volume6(&u) - 12.0).abs() < 1e-9);
        let mut i = cube();
        let report = run(&mut i, &b, BooleanOp::Intersect);
        assert_eq!(i.triangle_count(), 0);
        assert_eq!(report.kept_triangles, 0);
        let mut s = cube();
        run(&mut s, &b, BooleanOp::Subtract);
        assert_eq!(s.triangle_count(), 12);
        assert!((volume6(&s) - 6.0).abs() < 1e-9);
    }

    #[test]
    fn mesh_with_itself_is_rejected() {
        let mut a = cube();
        let bvh_a = Bvh::build(&a, &BuildOptions::default()).unwrap();
        // The scene short-circuits self-booleans; the op itself refuses: a
        // mesh against itself is all interface, which no rule resolves.
        assert_eq!(
            boolean_plan(&a, &bvh_a, &a, &bvh_a, &Rigid::IDENTITY, BooleanOp::Union).unwrap_err(),
            OpsError::InvalidInput("cannot combine a mesh with itself")
        );
        let _ = &mut a;
    }

    #[test]
    fn rejects_bad_inputs() {
        // Open.
        let open = grid(2);
        let a = cube();
        let bvh_a = Bvh::build(&a, &BuildOptions::default()).unwrap();
        let bvh_o = Bvh::build(&open, &BuildOptions::default()).unwrap();
        assert_eq!(
            boolean_plan(
                &a,
                &bvh_a,
                &open,
                &bvh_o,
                &Rigid::IDENTITY,
                BooleanOp::Union
            )
            .unwrap_err(),
            OpsError::InvalidInput("boolean needs closed meshes")
        );
        // Non-manifold edge: a tetrahedron on a cube edge gives edge
        // (0, 1) four triangles, without opening the mesh.
        let spike = with_tetra(&a, 0, 1, [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]);
        let bvh_s = Bvh::build(&spike, &BuildOptions::default()).unwrap();
        assert_eq!(
            boolean_plan(
                &spike,
                &bvh_s,
                &a,
                &bvh_a,
                &Rigid::IDENTITY,
                BooleanOp::Union
            )
            .unwrap_err(),
            OpsError::InvalidInput("boolean needs manifold meshes")
        );
        // A degenerate triangle with its own vertices trips neither the
        // boundary nor the manifold check, but the degenerate one.
        let mut needle = cube();
        let n = edit(&mut needle, |tx| {
            tx.add_vertices(&[[9.0, 9.0, 9.0], [9.0, 9.0, 9.0], [9.0, 9.0, 8.0]])?;
            Ok(())
        });
        let _ = n;
        let base = needle.vertex_slots() as u32 - 3;
        edit(&mut needle, |tx| {
            tx.add_triangles(&[[base, base, base + 1]])?;
            Ok(())
        });
        let bvh_n = Bvh::build(&needle, &BuildOptions::default()).unwrap();
        assert_eq!(
            boolean_plan(
                &needle,
                &bvh_n,
                &a,
                &bvh_a,
                &Rigid::IDENTITY,
                BooleanOp::Union
            )
            .unwrap_err(),
            OpsError::InvalidInput("boolean input has degenerate triangles")
        );
        // Two identical triangles with their own vertices: closed and
        // manifold, but duplicates.
        let dup = Mesh::from_buffers(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[0, 1, 2, 0, 1, 2],
        )
        .unwrap();
        let bvh_d = Bvh::build(&dup, &BuildOptions::default()).unwrap();
        assert_eq!(
            boolean_plan(&dup, &bvh_d, &a, &bvh_a, &Rigid::IDENTITY, BooleanOp::Union).unwrap_err(),
            OpsError::InvalidInput("boolean input has duplicate triangles")
        );
    }

    #[test]
    fn rejects_inward_wound_cube() {
        let a = cube();
        let all: Vec<TriangleId> = (0..a.triangle_count() as TriangleId).collect();
        let inv = crate::test_util::flip(&a, &all);
        let b = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);
        let bvh_a = Bvh::build(&a, &BuildOptions::default()).unwrap();
        let bvh_inv = Bvh::build(&inv, &BuildOptions::default()).unwrap();
        let bvh_b = Bvh::build(&b, &BuildOptions::default()).unwrap();

        assert_eq!(
            boolean_plan(
                &inv,
                &bvh_inv,
                &b,
                &bvh_b,
                &Rigid::IDENTITY,
                BooleanOp::Union
            )
            .unwrap_err(),
            OpsError::InvalidInput("boolean input is wound inward")
        );
        assert_eq!(
            boolean_plan(
                &a,
                &bvh_a,
                &inv,
                &bvh_inv,
                &Rigid::IDENTITY,
                BooleanOp::Union
            )
            .unwrap_err(),
            OpsError::InvalidInput("boolean input is wound inward")
        );
    }

    #[test]
    fn stacked_cubes_intersect_is_empty_and_subtract_keeps_a() {
        let b = box_mesh([0.0, 0.0, 1.0], [1.0, 1.0, 2.0]);

        let mut inter = cube();
        run(&mut inter, &b, BooleanOp::Intersect);
        assert_eq!(
            inter.triangle_count(),
            0,
            "face-touching intersect must be empty, not an open sheet"
        );
        let m_inter = crate::measure(&inter).unwrap();
        assert!(m_inter.closed);
        assert_eq!(m_inter.orientation_conflicts, 0);

        let mut sub_mesh = cube();
        run(&mut sub_mesh, &b, BooleanOp::Subtract);
        let m_sub = crate::measure(&sub_mesh).unwrap();
        assert!(m_sub.closed, "subtract must be closed: {m_sub:?}");
        assert_eq!(m_sub.orientation_conflicts, 0);
        assert!(
            (volume6(&sub_mesh) - 6.0).abs() < 1e-4,
            "vol {}",
            volume6(&sub_mesh)
        );
    }

    #[test]
    fn edge_and_vertex_touching_cubes_are_strictly_manifold() {
        let edge_touch = box_mesh([1.0, 1.0, 0.0], [2.0, 2.0, 1.0]);
        let vert_touch = box_mesh([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);

        for b in [&edge_touch, &vert_touch] {
            for op in [BooleanOp::Union, BooleanOp::Subtract, BooleanOp::Intersect] {
                let mut a = cube();
                run(&mut a, b, op);
                let m = crate::measure(&a).unwrap();
                if a.triangle_count() > 0 {
                    assert!(m.closed, "{op:?}: expected closed, got {m:?}");
                }
                assert_eq!(m.non_manifold_edges, 0, "{op:?}: non-manifold edges");
                assert_eq!(m.orientation_conflicts, 0, "{op:?}: orientation conflicts");
                let an = crate::analyze(&a, &crate::AnalyzeOptions::default()).unwrap();
                assert!(
                    an.pinch_vertices.is_empty(),
                    "{op:?}: expected zero pinch vertices, got {:?}",
                    an.pinch_vertices
                );
            }
        }
    }

    #[test]
    fn hollow_shell_boolean_succeeds() {
        let make_hollow = || {
            let mut hollow = uv_sphere(12, 16, 1.0);
            let _ = edit(&mut hollow, |tx| {
                crate::shell(tx, &crate::ShellOptions { thickness: 0.3 })
            });
            hollow
        };
        let cutter = box_mesh([0.0, -0.6, -0.6], [1.5, 0.6, 0.6]);

        for op in [BooleanOp::Intersect, BooleanOp::Subtract] {
            let mut a = make_hollow();
            run(&mut a, &cutter, op);
            assert!(a.triangle_count() > 0);
            let m = crate::measure(&a).unwrap();
            assert!(m.closed, "{op:?}: {m:?}");
            assert_eq!(m.non_manifold_edges, 0);
            assert_eq!(m.orientation_conflicts, 0);
            assert!(
                m.signed_volume > 0.0,
                "{op:?}: signed_volume {}",
                m.signed_volume
            );
        }
    }

    #[test]
    fn rotated_tool_boolean_matches_analytic_volume() {
        let b = cube();
        let bvh_b = Bvh::build(&b, &BuildOptions::default()).unwrap();

        let len = (1.0_f64 + 4.0 + 9.0).sqrt();
        let (ux, uy, uz) = (1.0 / len, 2.0 / len, 3.0 / len);
        let (s, c) = 37.0_f64.to_radians().sin_cos();
        let t = 1.0 - c;
        let xf = Rigid {
            rotation: [
                [t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy],
                [t * uy * ux + s * uz, t * uy * uy + c, t * uy * uz - s * ux],
                [t * uz * ux - s * uy, t * uz * uy + s * ux, t * uz * uz + c],
            ],
            translation: [0.4, 0.3, 0.2],
        };
        assert_eq!(xf.validate(), Ok(()));

        let run_xf = |op: BooleanOp| -> f64 {
            let mut a = cube();
            let bvh_a = Bvh::build(&a, &BuildOptions::default()).unwrap();
            let (_report, _) = edit(&mut a, |tx| {
                let plan = boolean_plan(tx.mesh(), &bvh_a, &b, &bvh_b, &xf, op)?;
                apply_boolean(tx, &plan)
            });
            let m = crate::measure(&a).unwrap();
            assert!(m.closed, "{op:?}: {m:?}");
            assert_eq!(m.non_manifold_edges, 0);
            assert_eq!(m.orientation_conflicts, 0);
            m.volume.unwrap()
        };

        let v_union = run_xf(BooleanOp::Union);
        let v_sub = run_xf(BooleanOp::Subtract);
        let v_inter = run_xf(BooleanOp::Intersect);
        assert!(
            v_inter > 0.05,
            "expected non-trivial overlap, got {v_inter}"
        );
        assert!(
            (v_union + v_inter - 2.0).abs() < 1e-4,
            "V(A U B)={v_union}, V(A n B)={v_inter}"
        );
        assert!(
            (v_sub + v_inter - 1.0).abs() < 1e-4,
            "V(A \\ B)={v_sub}, V(A n B)={v_inter}"
        );
    }

    #[test]
    fn boolean_oom_sweep_is_atomic() {
        let a = uv_sphere(6, 10, 1.0);
        let b = uv_sphere(6, 10, 1.0);
        let bvh_a = Bvh::build(&a, &BuildOptions::default()).unwrap();
        let bvh_b = Bvh::build(&b, &BuildOptions::default()).unwrap();
        let xf = Rigid {
            translation: [0.45, 0.2, 0.15],
            ..Rigid::IDENTITY
        };
        let steps = crate::test_util::sweep(a, |tx| {
            let plan = boolean_plan(tx.mesh(), &bvh_a, &b, &bvh_b, &xf, BooleanOp::Union)?;
            apply_boolean(tx, &plan)?;
            Ok(())
        });
        assert!(steps > 0);
    }

    #[test]
    fn flush_coplanar_boxes_classify_cleanly() {
        // Two boxes sharing flush coplanar faces on z = 0 and z = 1:
        // A = [0, 2] x [0, 2] x [0, 1], B = [0.5, 1.5] x [0.5, 1.5] x [0, 1].
        let make_a = || box_mesh([0.0, 0.0, 0.0], [2.0, 2.0, 1.0]);
        let b = box_mesh([0.5, 0.5, 0.0], [1.5, 1.5, 1.0]);

        for (op, want_vol) in [
            (BooleanOp::Union, 4.0),
            (BooleanOp::Subtract, 3.0),
            (BooleanOp::Intersect, 1.0),
        ] {
            let mut a = make_a();
            run(&mut a, &b, op);
            let m = crate::measure(&a).unwrap();
            assert!(m.closed, "{op:?}: {m:?}");
            assert_eq!(m.non_manifold_edges, 0);
            assert_eq!(m.orientation_conflicts, 0);
            let an = crate::analyze(&a, &crate::AnalyzeOptions::default()).unwrap();
            assert!(
                an.pinch_vertices.is_empty(),
                "{op:?}: {:?}",
                an.pinch_vertices
            );
            assert!(
                (m.volume.unwrap() - want_vol).abs() < 1e-4,
                "{op:?}: got {:?}, want {want_vol}",
                m.volume
            );
        }
    }

    #[test]
    fn kissing_blind_cavities_are_strictly_manifold() {
        // Subtract two edge-kissing inner cubes sequentially from a larger outer block:
        // (Big \ C1) then \ C2, exercising void-wedge dihedral edge de-pinching.
        let mut big = box_mesh([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
        let c1 = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let c2 = box_mesh([-1.0, -1.0, 0.0], [0.0, 0.0, 1.0]);

        run(&mut big, &c1, BooleanOp::Subtract);
        run(&mut big, &c2, BooleanOp::Subtract);

        let m = crate::measure(&big).unwrap();
        assert!(m.closed, "expected closed: {m:?}");
        assert_eq!(m.non_manifold_edges, 0);
        assert_eq!(m.orientation_conflicts, 0);
        let an = crate::analyze(&big, &crate::AnalyzeOptions::default()).unwrap();
        assert!(
            an.pinch_vertices.is_empty(),
            "expected zero pinch vertices, got {:?}",
            an.pinch_vertices
        );
        assert!(
            (m.volume.unwrap() - 62.0).abs() < 1e-4,
            "vol {:?}",
            m.volume
        );
    }

    #[test]
    fn boolean_options_and_batch_oom_sweep_are_atomic() {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                crate::fail_allocations_after(None);
            }
        }
        let _reset = Reset;

        let a = cube();
        let b = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);
        let c = box_mesh([0.0, 0.5, 0.0], [1.0, 1.5, 1.0]);

        for method in [
            BooleanMethod::Manifold,
            BooleanMethod::BvhSdf,
            BooleanMethod::FhEdt,
        ] {
            let opts = BooleanOptions {
                op: BooleanOp::Union,
                method,
                voxel_pitch: Some(0.25),
                ..BooleanOptions::default()
            };
            let steps = crate::test_util::sweep(clone_mesh_fallible(&a).unwrap(), |tx| {
                apply_boolean_with_options(tx, &b, Rigid::IDENTITY, &opts).map(|_| ())
            });
            assert!(steps > 0, "{method:?} should test at least one allocation");
        }

        let tree = CsgNode::reduce_all(BooleanOp::Union, 3).unwrap();
        let batch_opts = BooleanOptions {
            op: BooleanOp::Union,
            method: BooleanMethod::FhEdt,
            voxel_pitch: Some(0.25),
            ..BooleanOptions::default()
        };
        for after in 0..60 {
            crate::fail_allocations_after(Some(after));
            let res = boolean_batch(
                &[
                    (&a, Rigid::IDENTITY),
                    (&b, Rigid::IDENTITY),
                    (&c, Rigid::IDENTITY),
                ],
                &tree,
                &batch_opts,
            );
            crate::fail_allocations_after(None);
            match res {
                Ok(out) => {
                    assert!(out.triangle_count() > 0);
                    break;
                }
                Err(OpsError::Store(StoreError::OutOfMemory)) => {}
                Err(other) => panic!("batch after={after}: unexpected error {other:?}"),
            }
        }
    }

    #[test]
    fn production_boolean_uses_no_untracked_allocs_or_hash_collections() {
        for (name, src) in [
            ("boolean.rs", include_str!("boolean.rs")),
            ("classify.rs", include_str!("boolean/classify.rs")),
            ("perturbation.rs", include_str!("boolean/perturbation.rs")),
            ("voxel.rs", include_str!("boolean/voxel.rs")),
            ("batch.rs", include_str!("boolean/batch.rs")),
        ] {
            let prod = src.split("#[cfg(test)]").next().unwrap();
            for forbidden in [
                ".sort(",
                ".sort_by(",
                ".sort_by_key(",
                ".collect",
                "HashMap",
                "HashSet",
            ] {
                assert!(
                    !prod.contains(forbidden),
                    "{name} production code must not contain `{forbidden}`"
                );
            }
        }
    }
}
