//! Boolean operations (union, subtract, intersect) between two closed meshes.
//!
//! [`boolean_plan`] splits every triangle of mesh A that mesh B crosses (and
//! vice versa) so that each piece lies strictly on one side of the other
//! mesh, and classifies the pieces; [`apply_boolean`] writes the plan into
//! A's transaction. The result stays on A with A's placement, like a cut
//! that keeps the tool: for world-space booleans the caller maps B into A's
//! frame first (see `fast-mesh-scene`).
//!
//! Inputs must be closed, manifold, consistently wound, and free of
//! degenerate, duplicate and self-intersecting triangles; anything else is
//! [`OpsError::InvalidInput`]. Touches at a point or along an edge are fine,
//! and so are coplanar area overlaps (flush faces): a piece whose centroid
//! lies on the tool surface is decided by the surface just outside it.
//! Disjoint inputs need no splitting (union concatenates, intersect is
//! empty, subtract keeps A); nested inputs classify whole.
//!
//! Each side is split independently against the other's surface, so the two
//! rims along the intersection curve generally carry different vertices.
//! The apply step stitches them without any winding assumption: every rim
//! edge is split at the other rim's nearby vertices (reusing their ids), and
//! a final [`weld`] merges what is left within tolerance. The output is
//! closed when the inputs are; point and edge touches leave pinch vertices.
//!
//! New vertices take channel defaults; only positions and triangles cross
//! over. Tolerances scale with the inputs (`1e-6` of the coordinate
//! magnitude); meshes with features near that scale may weld together.
//!
//! [`weld`]: crate::weld

use std::collections::{BTreeMap, BTreeSet};

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::{
    ClosestOptions, Cull, PairSegment, Ray, RayOptions, Rigid, closest_point,
    intersecting_segments, raycast_all,
};
use fast_mesh_store::{Mesh, StoreError, Transaction, TriangleId, VertexId};

use crate::analyze::analyze;
use crate::{OpsError, Result, push, reserve};

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
    /// Triangles added (split pieces and the tool's kept pieces), minus the
    /// ones the cleanup merged or removed again.
    pub triangles_added: usize,
}

/// The plan of [`boolean_plan`]: what to drop, rewrite and add on mesh A.
/// Built read-only from both meshes; [`apply_boolean`] writes it.
#[derive(Debug, Clone, Default)]
pub struct BooleanPlan {
    drop_a: Vec<TriangleId>,
    rewrite: Vec<(TriangleId, Vec<[PRef; 3]>)>,
    fresh: Vec<[f32; 3]>,
    tool: Vec<[PRef; 3]>,
    tol: f32,
}

/// A vertex of a planned triangle: a live vertex of A or a fresh position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PRef {
    Live(VertexId),
    Fresh(u32),
}

fn oom() -> OpsError {
    OpsError::Store(StoreError::OutOfMemory)
}

type F3 = [f32; 3];
type D3 = [f64; 3];

fn sub(a: D3, b: D3) -> D3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: D3, b: D3) -> D3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: D3, b: D3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn area2(a: D3, b: D3, c: D3) -> f64 {
    let n = cross(sub(b, a), sub(c, a));
    dot(n, n)
}

fn max_edge2(t: &[D3; 3]) -> f64 {
    let mut m = 0.0f64;
    for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
        let d = sub(b, a);
        m = m.max(dot(d, d));
    }
    m
}

/// A piece smaller than this times its longest edge squared is dropped. The
/// same rule as [`remove_degenerate`](crate::remove_degenerate)'s default.
const AREA_EPSILON: f64 = 1e-12;

/// Plans the boolean `op` of A with B (B placed in A's frame by `xf`): the
/// result replaces A. Reads only; [`apply_boolean`] writes the plan.
///
/// `bvh_a` and `bvh_b` must be current for `a` and `b` (the plan queries
/// them for segments, parity and closest points).
///
/// # Errors
/// - [`OpsError::InvalidInput`] when an input is not closed, manifold,
///   consistently wound or free of degenerate, duplicate and
///   self-intersecting triangles, when a mapped tool position leaves `f32`
///   range, or when an intersection is too complex (colinearly overlapping
///   segments that meet in no endpoint).
/// - [`OpsError::Query`] or [`OpsError::Store`] from the queries and the
///   temporary builds. Nothing is written: the plan owns every allocation.
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
    // The tool in A's frame, slot for slot (ids only matter for queries):
    // positions mapped through `xf`, triangles untouched.
    let tool = map_tool(b, xf)?;
    let tool_bvh = Bvh::build(&tool, &BuildOptions::default()).map_err(|e| match e {
        fast_mesh_bvh::BvhError::OutOfMemory => OpsError::Store(StoreError::OutOfMemory),
        // A fresh build from the mapped mesh cannot mismatch; anything else
        // is a bug, reported without a dedicated variant.
        _ => OpsError::InvalidInput("boolean tool BVH failed to build"),
    })?;
    let tol = tolerance(a, &tool);
    let mut plan = BooleanPlan {
        tol,
        ..BooleanPlan::default()
    };
    let mut fresh_of: BTreeMap<[u32; 3], u32> = BTreeMap::new();
    let a_tool = ToolRef {
        mesh: &tool,
        bvh: &tool_bvh,
    };
    let a_live = ToolRef {
        mesh: a,
        bvh: bvh_a,
    };
    let pieces_a = split_side(a, bvh_a, &Side::A, &a_tool, &mut plan, &mut fresh_of)?;
    let pieces_b = split_side(
        &tool,
        &tool_bvh,
        &Side::B,
        &a_live,
        &mut plan,
        &mut fresh_of,
    )?;
    classify_side(
        a,
        &pieces_a,
        &Side::A,
        &a_tool,
        op,
        true,
        &mut plan,
        &mut fresh_of,
    )?;
    classify_side(
        &tool,
        &pieces_b,
        &Side::B,
        &a_live,
        op,
        false,
        &mut plan,
        &mut fresh_of,
    )?;
    Ok(plan)
}

/// Applies `plan` to A's transaction: drops, rewrites and adds triangles,
/// stitches the two rims, welds and cleans up. A failure rolls the
/// transaction back, so the call stays atomic through
/// [`Scene::edit`](https://docs.rs/fast-mesh-scene)-style callers.
pub fn apply_boolean(tx: &mut Transaction, plan: &BooleanPlan) -> Result<BooleanReport> {
    if !plan.drop_a.is_empty() {
        tx.remove_triangles(&plan.drop_a)?;
    }
    let fresh = if plan.fresh.is_empty() {
        Vec::new()
    } else {
        tx.add_vertices(&plan.fresh)?
    };
    let at = |r: PRef| match r {
        PRef::Live(v) => v,
        PRef::Fresh(i) => fresh[i as usize],
    };
    let mut added = 0usize;
    for (t, pieces) in &plan.rewrite {
        // `pieces` is never empty (a rewritten triangle keeps a piece).
        let tris: Vec<[VertexId; 3]> = pieces.iter().map(|p| p.map(at)).collect();
        tx.set_triangles(&[*t], &[tris[0]])?;
        if tris.len() > 1 {
            tx.add_triangles(&tris[1..])?;
            added += tris.len() - 1;
        }
    }
    let mut tool_ids = Vec::new();
    if !plan.tool.is_empty() {
        let tris: Vec<[VertexId; 3]> = plan.tool.iter().map(|p| p.map(at)).collect();
        tool_ids = tx.add_triangles(&tris)?;
        added += tris.len();
    }
    added += stitch(tx, &tool_ids, plan.tol)?;
    let welded = crate::weld(tx, plan.tol)?;
    let cleaned = crate::remove_degenerate(tx, &crate::DegenerateOptions::default())?;
    added = added.saturating_sub(welded.removed_triangles + cleaned.removed_triangles);
    Ok(BooleanReport {
        kept_triangles: tx.mesh().triangle_count(),
        triangles_added: added,
    })
}

/// The tolerance derived from the coordinate magnitude both meshes live at
/// (origin-relative `f32`): it merges seam duplicates from different
/// rounding paths while keeping test-scale features apart.
fn tolerance(a: &Mesh, tool: &Mesh) -> f32 {
    let mut m = 1.0f32;
    for mesh in [a, tool] {
        for p in mesh.positions() {
            for c in p {
                m = m.max(c.abs());
            }
        }
    }
    (1e-6 * m).max(f32::MIN_POSITIVE)
}

/// Rejects inputs no boolean can consume.
fn check_input(mesh: &Mesh, bvh: &Bvh) -> Result<()> {
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
    Ok(())
}

/// Mesh B mapped into A's frame, slot for slot.
fn map_tool(b: &Mesh, xf: &Rigid) -> Result<Mesh> {
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
    let dead_tris: Vec<TriangleId> = (0..b.triangle_slots() as TriangleId)
        .filter(|&t| b.triangle(t).is_none())
        .collect();
    let dead_verts: Vec<VertexId> = (0..b.vertex_slots() as VertexId)
        .filter(|&v| !b.is_vertex_alive(v))
        .collect();
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

/// One side of the split.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    /// Mesh A itself: corners reuse live vertex ids.
    A,
    /// The mapped tool: every vertex is fresh.
    B,
}

/// The other mesh, already in A's frame, with its BVH.
#[derive(Clone, Copy)]
struct ToolRef<'m> {
    mesh: &'m Mesh,
    bvh: &'m Bvh,
}

/// Corner positions of triangle `t` in `f64` (from stored `f32`, so queries
/// and splitting agree).
fn tri_pos(mesh: &Mesh, t: TriangleId) -> [D3; 3] {
    mesh.triangle(t)
        .expect("split of a live triangle")
        .map(|v| mesh.position(v).expect("live corner").map(f64::from))
}

/// A fresh vertex for `p` (exact-`f32` deduplicated across the whole plan).
fn fresh(p: D3, plan: &mut BooleanPlan, fresh_of: &mut BTreeMap<[u32; 3], u32>) -> Result<PRef> {
    let f = [p[0] as f32, p[1] as f32, p[2] as f32];
    if !f.iter().all(|c| c.is_finite()) {
        return Err(OpsError::InvalidInput("boolean point leaves f32 range"));
    }
    let bits = f.map(f32::to_bits);
    if let Some(&i) = fresh_of.get(&bits) {
        return Ok(PRef::Fresh(i));
    }
    let i = u32::try_from(plan.fresh.len()).map_err(|_| oom())?;
    reserve(&mut plan.fresh, 1)?;
    plan.fresh.push(f);
    fresh_of.insert(bits, i);
    Ok(PRef::Fresh(i))
}

/// Distance from `p` to the segment `a..b`, with the projection parameter.
fn point_seg(p: D3, a: D3, b: D3) -> (f64, f64) {
    let ab = sub(b, a);
    let len2 = dot(ab, ab);
    if len2 == 0.0 {
        let d = sub(p, a);
        return (dot(d, d).sqrt(), 0.0);
    }
    let t = dot(sub(p, a), ab) / len2;
    let c = std::array::from_fn(|k| a[k] + t.clamp(0.0, 1.0) * ab[k]);
    let d = sub(p, c);
    (dot(d, d).sqrt(), t)
}

/// The id edges are keyed by.
fn key(a: VertexId, b: VertexId) -> (VertexId, VertexId) {
    (a.min(b), b.max(a))
}

/// A work point: its position and how it materialises.
#[derive(Debug, Clone, Copy)]
struct WPt {
    pos: D3,
    v: PRef,
}

/// A convex piece of a triangle being split: its boundary ring, interior
/// points and segments, all as flat point indices.
#[derive(Debug, Clone)]
struct Piece {
    ring: Vec<usize>,
    inner: Vec<usize>,
    segs: Vec<(usize, usize)>,
}

/// A triangle's corners, crossings and segments, ready to split.
struct TriWork {
    corners: [D3; 3],
    corner_v: [PRef; 3],
    /// Boundary points per edge (corner `i` to corner `(i+1)%3`), sorted.
    edge: [Vec<WPt>; 3],
    /// Interior points.
    inner: Vec<WPt>,
    /// Segments as flat point indices (corners, then edge points per edge
    /// in order, then interior points).
    segs: Vec<(usize, usize)>,
}

/// Split pieces of one triangle, and whether any split follows a shared
/// edge (so the rewrite must materialise even when every piece is kept).
struct SplitTris {
    pieces: Vec<[PRef; 3]>,
    edge_split: bool,
}

/// Splits every triangle of `mesh` that the tool crosses — and every
/// triangle with a crossing on a shared edge, so splits never leave
/// T-junctions within the side — returning the pieces per touched triangle.
/// Untouched triangles are absent; the caller classifies them whole.
fn split_side(
    mesh: &Mesh,
    bvh: &Bvh,
    side: &Side,
    tool: &ToolRef<'_>,
    plan: &mut BooleanPlan,
    fresh_of: &mut BTreeMap<[u32; 3], u32>,
) -> Result<BTreeMap<TriangleId, SplitTris>> {
    let mut out: BTreeMap<TriangleId, SplitTris> = BTreeMap::new();
    let mut segs: Vec<PairSegment> = Vec::new();
    intersecting_segments(mesh, bvh, tool.mesh, tool.bvh, &Rigid::IDENTITY, &mut segs)?;
    if segs.is_empty() {
        return Ok(out);
    }
    // Group the segments by triangle of this side, deduplicated. The mesh
    // being split is always the first mesh of the pair query, so `s.a`
    // (with endpoints already in its frame) is its triangle in both cases.
    let mut by_tri: BTreeMap<TriangleId, Vec<(F3, F3)>> = BTreeMap::new();
    for s in &segs {
        let (t, from, to) = (s.a, s.from, s.to);
        if mesh.triangle(t).is_none() {
            continue;
        }
        let e = by_tri.entry(t).or_default();
        if !e
            .iter()
            .any(|&(a, b)| (a == from && b == to) || (a == to && b == from))
        {
            push(e, (from, to))?;
        }
    }
    // Crossing parameters per mesh edge, shared by both adjacent
    // triangles, so splits never leave T-junctions within the side. Each
    // entry is `(t from the smaller id, fresh index)`.
    let mut edge_pts: BTreeMap<(VertexId, VertexId), Vec<(f64, u32)>> = BTreeMap::new();
    // Pass 1: locate every segment's endpoints, filling the shared edge
    // map. Pass 2 then assembles and splits every triangle with a segment
    // or a shared edge crossing (edge-only triangles still need their ring
    // split, or the splits would leave T-junctions within the side). Rings
    // assemble from the complete map only, so no triangle misses a
    // neighbour's crossing.
    let mut ends: BTreeMap<TriangleId, Vec<(PtLoc, PtLoc)>> = BTreeMap::new();
    for (t, ss) in &by_tri {
        let e = locate_ends(mesh, side, *t, ss, &mut edge_pts, plan, fresh_of)?;
        ends.insert(*t, e);
    }
    let mut todo: BTreeSet<TriangleId> = by_tri.keys().copied().collect();
    for t in 0..mesh.triangle_slots() as TriangleId {
        if mesh.triangle(t).is_none() {
            continue;
        }
        let tri = mesh.triangle(t).expect("live");
        let touched = [0, 1, 2].iter().any(|&i| {
            edge_pts
                .get(&key(tri[i], tri[(i + 1) % 3]))
                .is_some_and(|l| !l.is_empty())
        });
        // Triangles touched only through a shared edge have no segments of
        // their own.
        if touched {
            todo.insert(t);
        }
    }
    for t in todo {
        let e = ends.remove(&t).unwrap_or_default();
        let work = assemble(mesh, side, t, &e, &edge_pts, plan, fresh_of)?;
        let edge_split = work.edge.iter().any(|e| !e.is_empty());
        let tris = split_triangle(work, plan, fresh_of)?;
        if !tris.is_empty() {
            out.insert(
                t,
                SplitTris {
                    pieces: tris,
                    edge_split,
                },
            );
        }
    }
    Ok(out)
}

/// A located segment endpoint: a corner, an edge point or an interior one.
#[derive(Debug, Clone, Copy)]
enum PtLoc {
    Corner(usize),
    Edge {
        edge: usize,
        t: f64,
        pos: D3,
        v: PRef,
    },
    Inner {
        pos: D3,
        v: PRef,
    },
}

/// Locates every segment endpoint of triangle `t` (corner, edge or
/// interior), filing edge crossings in the side's shared edge map. Runs for
/// every segment triangle before any ring assembles, so no triangle misses
/// a neighbour's crossing.
#[allow(clippy::too_many_arguments)]
fn locate_ends(
    mesh: &Mesh,
    _side: &Side,
    t: TriangleId,
    ss: &[(F3, F3)],
    edge_pts: &mut BTreeMap<(VertexId, VertexId), Vec<(f64, u32)>>,
    plan: &mut BooleanPlan,
    fresh_of: &mut BTreeMap<[u32; 3], u32>,
) -> Result<Vec<(PtLoc, PtLoc)>> {
    let corners = tri_pos(mesh, t);
    let c32: [F3; 3] = corners.map(|p| p.map(|c| c as f32));
    let elen: [f64; 3] = std::array::from_fn(|i| {
        let d = sub(corners[(i + 1) % 3], corners[i]);
        dot(d, d).sqrt()
    });
    let etol = |i: usize| 1e-6 * elen[i].max(1e-12);
    let mut ends: Vec<(PtLoc, PtLoc)> = Vec::new();
    for (from, to) in ss {
        let mut pair: Vec<PtLoc> = Vec::new();
        for e in [from, to] {
            if let Some(c) = c32.iter().position(|c| *c == *e) {
                pair.push(PtLoc::Corner(c));
                continue;
            }
            let p = e.map(f64::from);
            // Snap to a corner first (rounding may miss the exact match).
            let mut corner = None;
            for (c, cp) in corners.iter().enumerate() {
                let d = sub(p, *cp);
                if dot(d, d).sqrt() <= etol(c).max(etol((c + 2) % 3)) {
                    corner = Some(c);
                    break;
                }
            }
            if let Some(c) = corner {
                pair.push(PtLoc::Corner(c));
                continue;
            }
            // Nearest edge within tolerance, else interior.
            let mut best: Option<(usize, f64)> = None;
            for (i, (a, b)) in [
                (corners[0], corners[1]),
                (corners[1], corners[2]),
                (corners[2], corners[0]),
            ]
            .into_iter()
            .enumerate()
            {
                let (d, tpar) = point_seg(p, a, b);
                if d <= etol(i) && best.is_none_or(|(_, bt)| tpar < bt) {
                    best = Some((i, tpar));
                }
            }
            if let Some((i, tpar)) = best {
                let tri = mesh.triangle(t).expect("live");
                let ekey = key(tri[i], tri[(i + 1) % 3]);
                let tkey = if tri[i] < tri[(i + 1) % 3] {
                    tpar
                } else {
                    1.0 - tpar
                };
                let list = edge_pts.entry(ekey).or_default();
                let mut found = None;
                for &(pt, v) in list.iter() {
                    if (pt - tkey).abs() <= 1e-9 {
                        found = Some(v);
                        break;
                    }
                }
                let v = match found {
                    Some(v) => PRef::Fresh(v),
                    None => {
                        let v = fresh(p, plan, fresh_of)?;
                        let PRef::Fresh(idx) = v else {
                            unreachable!("fresh is fresh")
                        };
                        push(list, (tkey, idx))?;
                        // Keep parameters sorted for determinism.
                        list.sort_by(|x, y| x.0.total_cmp(&y.0));
                        v
                    }
                };
                pair.push(PtLoc::Edge {
                    edge: i,
                    t: tpar,
                    pos: p,
                    v,
                });
                continue;
            }
            let v = fresh(p, plan, fresh_of)?;
            pair.push(PtLoc::Inner { pos: p, v });
        }
        push(&mut ends, (pair[0], pair[1]))?;
    }
    Ok(ends)
}

/// Assembles triangle `t`'s work from its located ends and the complete
/// shared edge map: corners, edge points in ring order, interior points.
#[allow(clippy::too_many_arguments)]
fn assemble(
    mesh: &Mesh,
    side: &Side,
    t: TriangleId,
    ends: &[(PtLoc, PtLoc)],
    edge_pts: &BTreeMap<(VertexId, VertexId), Vec<(f64, u32)>>,
    plan: &mut BooleanPlan,
    fresh_of: &mut BTreeMap<[u32; 3], u32>,
) -> Result<TriWork> {
    let corners = tri_pos(mesh, t);
    let corner_v: [PRef; 3] = match side {
        Side::A => mesh.triangle(t).expect("live").map(PRef::Live),
        Side::B => {
            let mut v = [PRef::Fresh(0); 3];
            for (k, p) in corners.iter().enumerate() {
                v[k] = fresh(*p, plan, fresh_of)?;
            }
            v
        }
    };
    // Flat points: corners, then edge points in ring order, then interior.
    let mut pts: Vec<WPt> = Vec::new();
    for (k, p) in corners.iter().enumerate() {
        push(
            &mut pts,
            WPt {
                pos: *p,
                v: corner_v[k],
            },
        )?;
    }
    let tri = mesh.triangle(t).expect("live");
    let mut edge_idx: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let edge_frames = [
        (0, corners[0], corners[1], tri[0], tri[1]),
        (1, corners[1], corners[2], tri[1], tri[2]),
        (2, corners[2], corners[0], tri[2], tri[0]),
    ];
    for ((i, c0, c1, t0, t1), edge) in edge_frames.into_iter().zip(edge_idx.iter_mut()) {
        // This triangle's edge points on edge `i`: its own plus the ones
        // the neighbour filed (they split this edge too).
        let mut ordered: Vec<(f64, WPt)> = Vec::new();
        for loc in ends.iter().flat_map(|(a, b)| [*a, *b]) {
            if let PtLoc::Edge { edge, t, pos, v } = loc
                && edge == i
                && !ordered.iter().any(|&(_, w): &(f64, WPt)| w.v == v)
            {
                ordered.push((t, WPt { pos, v }));
            }
        }
        let ekey = key(t0, t1);
        if let Some(list) = edge_pts.get(&ekey) {
            let axis_len = dot(sub(c1, c0), sub(c1, c0)).sqrt().max(1e-12);
            for &(_, idx) in list {
                let v = PRef::Fresh(idx);
                if ordered.iter().any(|&(_, w): &(f64, WPt)| w.v == v) {
                    continue;
                }
                let pos = plan.fresh[idx as usize].map(f64::from);
                let (d, tpar) = point_seg(pos, c0, c1);
                if d <= 1e-6 * axis_len {
                    ordered.push((tpar, WPt { pos, v }));
                }
            }
        }
        ordered.sort_by(|x, y| x.0.total_cmp(&y.0));
        for (_, w) in ordered {
            edge.push(pts.len());
            push(&mut pts, w)?;
        }
    }
    // Interior points (deduplicated by vertex).
    let mut inner_idx: Vec<usize> = Vec::new();
    for loc in ends.iter().flat_map(|(a, b)| [*a, *b]) {
        if let PtLoc::Inner { pos, v } = loc
            && !pts.iter().any(|w| w.v == v)
        {
            inner_idx.push(pts.len());
            push(&mut pts, WPt { pos, v })?;
        }
    }
    // Segments as flat indices.
    let mut segs: Vec<(usize, usize)> = Vec::new();
    for (a, b) in ends {
        let (ia, ib) = (loc_index(*a, &pts), loc_index(*b, &pts));
        if ia != ib {
            push(&mut segs, (ia.min(ib), ia.max(ib)))?;
        }
    }
    segs.sort_unstable();
    segs.dedup();
    // The ring is implicit: corners + edge points in edge order.
    let mut edge_out: [Vec<WPt>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (src, dst) in edge_idx.iter().zip(edge_out.iter_mut()) {
        for &at in src {
            push(dst, pts[at])?;
        }
    }
    let mut inner_out: Vec<WPt> = Vec::new();
    for &at in &inner_idx {
        push(&mut inner_out, pts[at])?;
    }
    Ok(TriWork {
        corners,
        corner_v,
        edge: edge_out,
        inner: inner_out,
        segs,
    })
}

/// Flat index of a located endpoint (always present: every endpoint was
/// added to `pts` above).
fn loc_index(loc: PtLoc, pts: &[WPt]) -> usize {
    match loc {
        PtLoc::Corner(c) => c,
        PtLoc::Edge { v, .. } | PtLoc::Inner { v, .. } => {
            pts.iter().position(|w| w.v == v).expect("located point")
        }
    }
}

/// Splits the work triangle into convex pieces and emits non-degenerate
/// triangles with the parent's winding.
fn split_triangle(
    work: TriWork,
    plan: &mut BooleanPlan,
    fresh_of: &mut BTreeMap<[u32; 3], u32>,
) -> Result<Vec<[PRef; 3]>> {
    // Flat points, exactly as `locate` laid them out: corners, then edge
    // points per edge in order, then interior points. Segments are already
    // flat indices into this layout.
    let mut pts: Vec<WPt> = Vec::new();
    for (k, p) in work.corners.iter().enumerate() {
        push(
            &mut pts,
            WPt {
                pos: *p,
                v: work.corner_v[k],
            },
        )?;
    }
    let mut edge_idx: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (src, dst) in work.edge.iter().zip(edge_idx.iter_mut()) {
        for w in src {
            dst.push(pts.len());
            push(&mut pts, *w)?;
        }
    }
    let mut inner_idx: Vec<usize> = Vec::new();
    for w in &work.inner {
        inner_idx.push(pts.len());
        push(&mut pts, *w)?;
    }
    let mut segs: Vec<(usize, usize)> = Vec::new();
    for &(a, b) in &work.segs {
        if a != b {
            push(&mut segs, (a.min(b), a.max(b)))?;
        }
    }
    segs.sort_unstable();
    segs.dedup();
    // The 2D frame: the triangle with its dominant axis dropped.
    let n = cross(
        sub(work.corners[1], work.corners[0]),
        sub(work.corners[2], work.corners[0]),
    );
    let axis = [n[0].abs(), n[1].abs(), n[2].abs()]
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.total_cmp(y.1))
        .map(|(i, _)| i)
        .unwrap_or(2);
    let flat2 = |p: D3| -> [f64; 2] { [p[(axis + 1) % 3], p[(axis + 2) % 3]] };
    // Split segments at interior crossings and colinear endpoints.
    split_crossings(&mut pts, &mut segs, &mut inner_idx, flat2, plan, fresh_of)?;
    // Ring: corners with edge points in order.
    let mut ring = Vec::new();
    for (i, edge) in edge_idx.iter().enumerate() {
        push(&mut ring, i)?;
        ring.extend_from_slice(edge);
    }
    // Interior points created by `split_crossings` are appended to `pts`
    // and `inner_idx` already; collect any others (none: all interior
    // points are segment endpoints by construction).
    let mut pieces = vec![Piece {
        ring,
        inner: inner_idx,
        segs,
    }];
    // Split until every segment is a piece edge: full segments split convex
    // pieces in two, interior endpoints fan their piece first. Progress is
    // lexicographic (interior points, then segments): a fan consumes an
    // interior point, a full split consumes a segment.
    loop {
        let before: (usize, usize) = pieces.iter().fold((0, 0), |(ni, ns), p| {
            (ni + p.inner.len(), ns + p.segs.len())
        });
        // Drop boundary-hugging segments first: both endpoints on the ring
        // with a colinear walk between them. They follow a ring edge (a
        // coplanar edge overlap), so the ring already preserves them;
        // splitting along them would only make slivers.
        let mut hugged: Option<Vec<Piece>> = None;
        for (pi, piece) in pieces.iter().enumerate() {
            if let Some(seg) = piece
                .segs
                .iter()
                .find(|&&(a, b)| hugs_ring(a, b, piece, &pts, flat2))
            {
                let mut out = Vec::new();
                for (qi, q) in pieces.iter().enumerate() {
                    if qi == pi {
                        let mut p = q.clone();
                        p.segs.retain(|&s| s != *seg);
                        push(&mut out, p)?;
                    } else {
                        push(&mut out, q.clone())?;
                    }
                }
                hugged = Some(out);
                break;
            }
        }
        if let Some(out) = hugged {
            pieces = out;
            continue;
        }
        let mut next: Option<Vec<Piece>> = None;
        for (pi, piece) in pieces.iter().enumerate() {
            if let Some((p1, p2)) = full_split(piece, &pts, flat2)? {
                let mut out = Vec::new();
                for (qi, q) in pieces.iter().enumerate() {
                    if qi == pi {
                        push(&mut out, p1.clone())?;
                        push(&mut out, p2.clone())?;
                    } else {
                        push(&mut out, q.clone())?;
                    }
                }
                next = Some(out);
                break;
            }
            if let Some(&q) = piece
                .inner
                .iter()
                .find(|q| piece.segs.iter().any(|&(a, b)| a == **q || b == **q))
            {
                let fanned = fan_split(piece, q, &mut pts, flat2, plan, fresh_of)?;
                let mut out = Vec::new();
                for (qi, qpiece) in pieces.iter().enumerate() {
                    if qi == pi {
                        for f in &fanned {
                            push(&mut out, f.clone())?;
                        }
                    } else {
                        push(&mut out, qpiece.clone())?;
                    }
                }
                next = Some(out);
                break;
            }
        }
        match next {
            Some(out) => pieces = out,
            None => break,
        }
        let after: (usize, usize) = pieces.iter().fold((0, 0), |(ni, ns), p| {
            (ni + p.inner.len(), ns + p.segs.len())
        });
        if after.0 > before.0 || (after.0 == before.0 && after.1 >= before.1) {
            // No progress: a segment spans pieces (rounding at a shared
            // boundary). Fail rather than loop or leave a hole.
            return Err(OpsError::InvalidInput("boolean intersection too complex"));
        }
    }
    // Triangulate each convex piece without losing ring vertices: collapse
    // colinear runs, ear-clip the corners, then split the collapsed edges
    // back open. Every ring vertex survives in a non-degenerate triangle.
    let mut tris: Vec<[PRef; 3]> = Vec::new();
    for piece in &pieces {
        if piece.ring.len() < 3 {
            continue;
        }
        // Collapse colinear runs, remembering the dropped points per edge.
        // A vertex is a corner unless it sits straight between its ring
        // neighbours (orient2 within 1e-9 relative); runs between
        // consecutive corners are straight by transitivity.
        let n = piece.ring.len();
        // Triples of consecutive ring vertices, wrapping around.
        let ext: Vec<usize> = piece
            .ring
            .iter()
            .chain(piece.ring.iter().take(2))
            .copied()
            .collect();
        let mut is_corner = vec![true; n];
        for (k, w) in ext.windows(3).enumerate() {
            let (a, b, c) = (
                flat2(pts[w[0]].pos),
                flat2(pts[w[1]].pos),
                flat2(pts[w[2]].pos),
            );
            if colinear3(a, b, c) {
                // Window k tests vertex k+1.
                is_corner[(k + 1) % n] = false;
            }
        }
        let mut corners: Vec<usize> = Vec::new();
        for (k, &x) in piece.ring.iter().enumerate() {
            if is_corner[k] {
                push(&mut corners, x)?;
            }
        }
        if corners.len() < 3 {
            continue;
        }
        let mut runs: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        for (ci, &c) in corners.iter().enumerate() {
            let cn = corners[(ci + 1) % corners.len()];
            let mut mid = Vec::new();
            let mut k = piece.ring.iter().position(|&x| x == c).unwrap();
            loop {
                k = (k + 1) % n;
                if piece.ring[k] == cn {
                    break;
                }
                push(&mut mid, piece.ring[k])?;
            }
            if !mid.is_empty() {
                runs.insert((c, cn), mid);
            }
        }
        // Ear-clip the convex corners (every consecutive triple is a valid
        // ear): clip from the front until one triangle is left.
        let mut out: Vec<[usize; 3]> = Vec::new();
        let mut ring = corners.clone();
        while ring.len() > 3 {
            let tri = [ring[ring.len() - 1], ring[0], ring[1]];
            push(&mut out, tri)?;
            ring.remove(0);
        }
        push(&mut out, [ring[0], ring[1], ring[2]])?;
        for ((a, b), qs) in &runs {
            let at = out
                .iter()
                .position(|tri| tri.contains(a) && tri.contains(b))
                .expect("collapsed edge of a live piece");
            let tri = out.remove(at);
            split_chain(tri, *a, *b, qs, &mut out)?;
        }
        // Emit with PRefs, dropping zero-area pieces.
        for tri in out {
            let t = [pts[tri[0]].v, pts[tri[1]].v, pts[tri[2]].v];
            if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                continue;
            }
            let p = [pts[tri[0]].pos, pts[tri[1]].pos, pts[tri[2]].pos];
            if area2(p[0], p[1], p[2]) > AREA_EPSILON * max_edge2(&p).max(1e-300) {
                push(&mut tris, t)?;
            }
        }
    }
    Ok(tris)
}

/// 2D orientation (twice the signed area).
fn orient2(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Whether `b` sits straight between `a` and `c`: orientation within 1e-6
/// relative to the edge lengths. Consistent with the stitch and weld
/// tolerance (features below `1e-6` of the magnitude merge), so collapsed
/// points always come back through the run splits below.
fn colinear3(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let o = orient2(a, b, c).abs();
    let m = (b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2);
    let m = m.max((c[0] - b[0]).powi(2) + (c[1] - b[1]).powi(2));
    o * o <= 1e-12 * m * m.max(1e-300)
}

/// Where the segments `p..q` and `r..s` cross, if strictly interior to
/// both. Colinear cases report `Cross::Colinear`.
enum Cross {
    Point(f64, f64),
    Colinear,
    None,
}

fn seg_cross(p: [f64; 2], q: [f64; 2], r: [f64; 2], s: [f64; 2]) -> Cross {
    let d = orient2(p, q, r);
    let e = orient2(p, q, s);
    let f = orient2(r, s, p);
    let g = orient2(r, s, q);
    if d == 0.0 && e == 0.0 && f == 0.0 && g == 0.0 {
        return Cross::Colinear;
    }
    if d * e >= 0.0 || f * g >= 0.0 {
        return Cross::None;
    }
    let denom = (q[0] - p[0]) * (s[1] - r[1]) - (q[1] - p[1]) * (s[0] - r[0]);
    if denom == 0.0 {
        return Cross::None;
    }
    let t = ((r[0] - p[0]) * (s[1] - r[1]) - (r[1] - p[1]) * (s[0] - r[0])) / denom;
    let u = ((r[0] - p[0]) * (q[1] - p[1]) - (r[1] - p[1]) * (q[0] - p[0])) / denom;
    if t > 1e-9 && t < 1.0 - 1e-9 && u > 1e-9 && u < 1.0 - 1e-9 {
        Cross::Point(t, u)
    } else {
        Cross::None
    }
}

/// Point `p` strictly inside segment `a..b` (2D): exact orientation with
/// a relative distance tolerance.
fn on_seg_tol(p: [f64; 2], a: [f64; 2], b: [f64; 2], tol: f64) -> bool {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1];
    if len2 == 0.0 {
        return false;
    }
    let t = ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / len2;
    if t <= 1e-9 || t >= 1.0 - 1e-9 {
        return false;
    }
    let c = [a[0] + t * ab[0], a[1] + t * ab[1]];
    let d = (p[0] - c[0]) * (p[0] - c[0]) + (p[1] - c[1]) * (p[1] - c[1]);
    d <= tol * tol * len2
}

/// Point `p` strictly inside segment `a..b` (2D, tight tolerance for
/// exactly-computed crossings).
fn on_seg(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> bool {
    on_seg_tol(p, a, b, 1e-9)
}

/// Splits every segment at interior crossings and colinear endpoints,
/// adding the crossing points (as fresh vertices) to `inner`. Colinear
/// partial overlaps fail.
#[allow(clippy::too_many_arguments)]
fn split_crossings(
    pts: &mut Vec<WPt>,
    segs: &mut Vec<(usize, usize)>,
    inner: &mut Vec<usize>,
    flat2: impl Copy + Fn(D3) -> [f64; 2],
    plan: &mut BooleanPlan,
    fresh_of: &mut BTreeMap<[u32; 3], u32>,
) -> Result<()> {
    // Each pass resolves one crossing; cap the passes (segments only grow
    // by splitting, so a bound far above the input size means rounding is
    // fighting the predicates).
    let mut passes = segs.len() * 4 + 64;
    loop {
        if passes == 0 {
            return Err(OpsError::InvalidInput("boolean intersection too complex"));
        }
        passes -= 1;
        // Loose pass first: an endpoint within tolerance of a colinear
        // segment splits it there. Endpoints from different pairs agree
        // only up to rounding, so the exact predicates below would miss the
        // touch. Transverse near-misses are left alone (splitting them
        // would distort the curve more than rounding does).
        let mut snapped: Option<(usize, usize)> = None;
        for (i, &(a, b)) in segs.iter().enumerate() {
            for (j, &(c, d)) in segs.iter().enumerate().take(i) {
                if a == c || a == d || b == c || b == d {
                    continue;
                }
                let (pa, pb) = (flat2(pts[a].pos), flat2(pts[b].pos));
                let dab = [(pb[0] - pa[0]), (pb[1] - pa[1])];
                let len2 = dab[0] * dab[0] + dab[1] * dab[1];
                for (e, (o, oo), si) in [
                    (a, (c, d), j),
                    (b, (c, d), j),
                    (c, (a, b), i),
                    (d, (a, b), i),
                ] {
                    let pe = flat2(pts[e].pos);
                    // Colinear within tolerance: the endpoint is on the
                    // segment's line.
                    let o2 = orient2(pa, pb, pe);
                    if o2 * o2 > 1e-12 * len2 * len2 {
                        continue;
                    }
                    if on_seg_tol(pe, flat2(pts[o].pos), flat2(pts[oo].pos), 1e-6) {
                        snapped = Some((si, e));
                        break;
                    }
                }
                if snapped.is_some() {
                    break;
                }
            }
            if snapped.is_some() {
                break;
            }
        }
        if let Some((si, e)) = snapped {
            let (x, y) = segs[si];
            segs[si] = (x.min(e), x.max(e));
            push(segs, (e.min(y), e.max(y)))?;
            segs.sort_unstable();
            segs.dedup();
            continue;
        }
        let mut crossing: Option<(usize, usize, f64)> = None;
        let mut colinear: Option<(usize, usize)> = None;
        for (i, &(a, b)) in segs.iter().enumerate() {
            for (j, &(c, d)) in segs.iter().enumerate().take(i) {
                if a == c || a == d || b == c || b == d {
                    continue;
                }
                let (pa, pb, pc, pd) = (
                    flat2(pts[a].pos),
                    flat2(pts[b].pos),
                    flat2(pts[c].pos),
                    flat2(pts[d].pos),
                );
                match seg_cross(pa, pb, pc, pd) {
                    Cross::Point(t, _) => {
                        crossing = Some((i, j, t));
                        break;
                    }
                    Cross::Colinear => {
                        for (e, si) in [(a, j), (b, j), (c, i), (d, i)] {
                            let (o, oo) = if si == j { (c, d) } else { (a, b) };
                            if on_seg(flat2(pts[e].pos), flat2(pts[o].pos), flat2(pts[oo].pos)) {
                                colinear = Some((si, e));
                                break;
                            }
                        }
                        if colinear.is_some() {
                            break;
                        }
                        return Err(OpsError::InvalidInput("boolean intersection too complex"));
                    }
                    Cross::None => {}
                }
            }
            if crossing.is_some() || colinear.is_some() {
                break;
            }
        }
        if let Some((si, e)) = colinear {
            // An endpoint strictly inside another segment: split there (no
            // new point). The endpoint is interior, so both halves are
            // sorted pairs.
            let (x, y) = segs[si];
            segs[si] = (x.min(e), x.max(e));
            push(segs, (e.min(y), e.max(y)))?;
        } else if let Some((i, j, t)) = crossing {
            let (a, b) = segs[i];
            let (c, d) = segs[j];
            let p = std::array::from_fn(|k| pts[a].pos[k] + t * (pts[b].pos[k] - pts[a].pos[k]));
            let v = fresh(p, plan, fresh_of)?;
            let at = pts.len();
            push(pts, WPt { pos: p, v })?;
            push(inner, at)?;
            segs[i] = (a.min(at), a.max(at));
            push(segs, (at.min(b), at.max(b)))?;
            segs[j] = (c.min(at), c.max(at));
            push(segs, (at.min(d), at.max(d)))?;
        } else {
            break;
        }
        segs.sort_unstable();
        segs.dedup();
    }
    Ok(())
}

/// Whether segment `(a, b)` hugs the piece's ring: both endpoints on it
/// with a colinear walk between them (a coplanar edge overlap, already
/// preserved by the ring itself).
fn hugs_ring(
    a: usize,
    b: usize,
    piece: &Piece,
    pts: &[WPt],
    flat2: impl Copy + Fn(D3) -> [f64; 2],
) -> bool {
    let pa = piece.ring.iter().position(|&x| x == a);
    let pb = piece.ring.iter().position(|&x| x == b);
    let (Some(ia), Some(ib)) = (pa, pb) else {
        return false;
    };
    if ia == ib {
        return true;
    }
    let (pa, pb) = (flat2(pts[a].pos), flat2(pts[b].pos));
    let dab = [(pb[0] - pa[0]), (pb[1] - pa[1])];
    let len2 = dab[0] * dab[0] + dab[1] * dab[1];
    let tol = 1e-7 * len2.max(1e-300);
    let (lo, hi) = if ia < ib { (ia, ib) } else { (ib, ia) };
    let colinear = |range: &[usize]| {
        range.iter().all(|&x| {
            let p = flat2(pts[x].pos);
            orient2(pa, pb, p).abs() <= tol
        })
    };
    colinear(&piece.ring[lo + 1..hi])
        || colinear(&piece.ring[hi + 1..]) && colinear(&piece.ring[..lo])
}

/// Splits `piece` along a full segment (both endpoints on its ring) into
/// two convex pieces, or `None` when it has none.
fn full_split(
    piece: &Piece,
    pts: &[WPt],
    flat2: impl Copy + Fn(D3) -> [f64; 2],
) -> Result<Option<(Piece, Piece)>> {
    for &(a, b) in &piece.segs {
        let pa = piece.ring.iter().position(|&x| x == a);
        let pb = piece.ring.iter().position(|&x| x == b);
        let (Some(ia), Some(ib)) = (pa, pb) else {
            continue;
        };
        if ia == ib {
            continue;
        }
        // Walk the ring both ways.
        let (lo, hi) = if ia < ib { (ia, ib) } else { (ib, ia) };
        let r1: Vec<usize> = piece.ring[lo..=hi].to_vec();
        let mut r2: Vec<usize> = piece.ring[hi..].to_vec();
        r2.extend_from_slice(&piece.ring[..=lo]);
        if r1.len() < 3 || r2.len() < 3 {
            continue;
        }
        let mut p1 = Piece {
            ring: r1,
            inner: Vec::new(),
            segs: Vec::new(),
        };
        let mut p2 = Piece {
            ring: r2,
            inner: Vec::new(),
            segs: Vec::new(),
        };
        // Distribute the other segments (boundary goes to the first piece)
        // and interior points.
        for &(c, d) in &piece.segs {
            if (c == a && d == b) || (c == b && d == a) {
                continue;
            }
            let in1 = |x: usize| point_in_piece(flat2(pts[x].pos), &p1, pts, flat2);
            if in1(c) && in1(d) {
                push(&mut p1.segs, (c, d))?;
            } else {
                push(&mut p2.segs, (c, d))?;
            }
        }
        for &q in &piece.inner {
            if point_in_piece(flat2(pts[q].pos), &p1, pts, flat2) {
                push(&mut p1.inner, q)?;
            } else {
                push(&mut p2.inner, q)?;
            }
        }
        return Ok(Some((p1, p2)));
    }
    Ok(None)
}

/// Whether `p` is in the (convex) piece, boundary included.
fn point_in_piece(
    p: [f64; 2],
    piece: &Piece,
    pts: &[WPt],
    flat2: impl Copy + Fn(D3) -> [f64; 2],
) -> bool {
    let n = piece.ring.len();
    if n < 3 {
        return false;
    }
    // Convex fan test from the first vertex, with tolerance.
    let p0 = flat2(pts[piece.ring[0]].pos);
    for k in 1..n - 1 {
        let a = flat2(pts[piece.ring[k]].pos);
        let b = flat2(pts[piece.ring[k + 1]].pos);
        let o = orient2(p0, a, b);
        let inside = if o >= 0.0 {
            orient2(p0, a, p) >= -1e-9 && orient2(a, b, p) >= -1e-9 && orient2(b, p0, p) >= -1e-9
        } else {
            orient2(p0, a, p) <= 1e-9 && orient2(a, b, p) <= 1e-9 && orient2(b, p0, p) <= 1e-9
        };
        if inside {
            return true;
        }
    }
    false
}

/// Fans the convex `piece` from interior point `q` (a segment endpoint),
/// splitting crossing segments at the spokes. Returns the fan triangles as
/// pieces with redistributed segments and interior points.
#[allow(clippy::too_many_arguments)]
fn fan_split(
    piece: &Piece,
    q: usize,
    pts: &mut Vec<WPt>,
    flat2: impl Copy + Fn(D3) -> [f64; 2],
    plan: &mut BooleanPlan,
    fresh_of: &mut BTreeMap<[u32; 3], u32>,
) -> Result<Vec<Piece>> {
    let n = piece.ring.len();
    // New points where segments cross the spokes `q..ring[i]`.
    let mut spoke_pts: Vec<Vec<usize>> = vec![Vec::new(); n];
    // Collect first, then split: a segment crossing several spokes is cut
    // at each of them.
    let mut cuts: Vec<(usize, usize, D3)> = Vec::new();
    for &(a, b) in &piece.segs {
        if a == q || b == q {
            continue;
        }
        for (i, &r) in piece.ring.iter().enumerate() {
            let pa = flat2(pts[a].pos);
            let pb = flat2(pts[b].pos);
            let pq = flat2(pts[q].pos);
            let pr = flat2(pts[r].pos);
            if let Cross::Point(t, u) = seg_cross(pa, pb, pq, pr)
                && t > 1e-9
                && t < 1.0 - 1e-9
                && u > 1e-9
                && u < 1.0 - 1e-9
            {
                let p =
                    std::array::from_fn(|k| pts[a].pos[k] + t * (pts[b].pos[k] - pts[a].pos[k]));
                push(&mut cuts, (i, a, p))?;
            }
        }
    }
    // Group cuts per spoke, ordered from q outward.
    let mut per_spoke: Vec<Vec<(f64, D3)>> = vec![Vec::new(); n];
    for (i, _, p) in &cuts {
        // Parameter along the spoke for ordering.
        let (_, u) = point_seg2(flat2(*p), flat2(pts[q].pos), flat2(pts[piece.ring[*i]].pos));
        per_spoke[*i].push((u, *p));
    }
    for sp in per_spoke.iter_mut() {
        sp.sort_by(|x, y| x.0.total_cmp(&y.0));
    }
    // Split the segments at the spoke points.
    let mut seg_list = piece.segs.clone();
    for (i, sp) in per_spoke.iter().enumerate() {
        for &(_, p) in sp {
            let v = fresh(p, plan, fresh_of)?;
            let at = pts.len();
            push(pts, WPt { pos: p, v })?;
            // Split every segment passing through `p` (it lies on a spoke,
            // shared by two future fan triangles).
            let mut k = 0;
            while k < seg_list.len() {
                let (x, y) = seg_list[k];
                if x == at || y == at {
                    k += 1;
                    continue;
                }
                if on_seg(flat2(p), flat2(pts[x].pos), flat2(pts[y].pos)) {
                    seg_list[k] = (x.min(at), x.max(at));
                    push(&mut seg_list, (at.min(y), at.max(y)))?;
                }
                k += 1;
            }
            spoke_pts[i].push(at);
        }
    }
    seg_list.sort_unstable();
    seg_list.dedup();
    // Fan triangles, with spokes split at their crossings.
    let mut out: Vec<Piece> = Vec::new();
    // Spoke chains from q outward (spoke point indices, then the ring end).
    let mut chains: Vec<Vec<usize>> = Vec::new();
    for (spoke, &r) in spoke_pts.iter().zip(piece.ring.iter()) {
        let mut chain = vec![q];
        chain.extend_from_slice(spoke);
        chain.push(r);
        chains.push(chain);
    }
    for i in 0..n {
        let (c0, c1) = (&chains[i], &chains[(i + 1) % n]);
        let rows = c0.len().max(c1.len());
        for r in 0..rows - 1 {
            let a0 = c0[r.min(c0.len() - 1)];
            let a1 = c0[(r + 1).min(c0.len() - 1)];
            let b0 = c1[r.min(c1.len() - 1)];
            let b1 = c1[(r + 1).min(c1.len() - 1)];
            for tri in [[a0, a1, b1], [a0, b1, b0]] {
                if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
                    continue;
                }
                let p = [pts[tri[0]].pos, pts[tri[1]].pos, pts[tri[2]].pos];
                if area2(p[0], p[1], p[2]) <= AREA_EPSILON * max_edge2(&p).max(1e-300) {
                    continue;
                }
                push(
                    &mut out,
                    Piece {
                        ring: vec![tri[0], tri[1], tri[2]],
                        inner: Vec::new(),
                        segs: Vec::new(),
                    },
                )?;
            }
        }
    }
    // Distribute the split segments by midpoint; every sub-seg was cut at
    // each spoke it crosses, so each lies in one fan. Complete any endpoint
    // that is not a fan vertex yet (strictly inside goes to inner, on a
    // boundary edge splices into the ring).
    for (a, b) in &seg_list {
        let mid = std::array::from_fn(|k| 0.5 * (pts[*a].pos[k] + pts[*b].pos[k]));
        let mut at = None;
        for (fi, fan) in out.iter().enumerate() {
            if fan_contains_pos(fan, flat2(mid), pts, flat2) {
                at = Some(fi);
                break;
            }
        }
        let Some(fi) = at else {
            return Err(OpsError::InvalidInput("boolean intersection too complex"));
        };
        for &x in [a, b] {
            complete_endpoint(&mut out[fi], x, pts, flat2)?;
        }
        push(&mut out[fi].segs, (*a, *b))?;
    }
    for &p in &piece.inner {
        if p == q {
            continue;
        }
        let mut placed = false;
        for fan in out.iter_mut() {
            if fan_contains(fan, p, pts, flat2) {
                if !fan.ring.contains(&p) && !fan.inner.contains(&p) {
                    push(&mut fan.inner, p)?;
                }
                placed = true;
                break;
            }
        }
        if !placed {
            return Err(OpsError::InvalidInput("boolean intersection too complex"));
        }
    }
    Ok(out)
}

/// Splits triangle `tri` along its edge `u..v` at `qs` (in order from `u`
/// to `v`, strictly interior), appending the chain to `out`. The pieces
/// follow the parent's winding and are never degenerate.
fn split_chain<T: Copy + PartialEq>(
    tri: [T; 3],
    u: T,
    v: T,
    qs: &[T],
    out: &mut Vec<[T; 3]>,
) -> Result<()> {
    let iu = tri.iter().position(|&x| x == u).expect("edge of triangle");
    // Normalize to the forward direction (cyclic order).
    let (a, b, o, qs): (T, T, T, Vec<T>) = if tri[(iu + 1) % 3] == v {
        (u, v, tri[(iu + 2) % 3], qs.to_vec())
    } else {
        debug_assert!(tri[(iu + 1) % 3] != u && tri[(iu + 2) % 3] == v);
        let o = tri[(iu + 1) % 3];
        let mut r = qs.to_vec();
        r.reverse();
        (v, u, o, r)
    };
    let mut prev = a;
    for q in qs.iter().chain(std::iter::once(&b)) {
        push(out, [prev, *q, o])?;
        prev = *q;
    }
    Ok(())
}

/// Whether the piece contains the 2D point `p` (boundary included).
fn fan_contains_pos(
    fan: &Piece,
    p: [f64; 2],
    pts: &[WPt],
    flat2: impl Copy + Fn(D3) -> [f64; 2],
) -> bool {
    if fan.ring.len() != 3 {
        return point_in_piece(p, fan, pts, flat2);
    }
    let t = [
        flat2(pts[fan.ring[0]].pos),
        flat2(pts[fan.ring[1]].pos),
        flat2(pts[fan.ring[2]].pos),
    ];
    let o = orient2(t[0], t[1], t[2]);
    let (s0, s1, s2) = (
        orient2(t[0], t[1], p),
        orient2(t[1], t[2], p),
        orient2(t[2], t[0], p),
    );
    if o >= 0.0 {
        s0 >= -1e-9 && s1 >= -1e-9 && s2 >= -1e-9
    } else {
        s0 <= 1e-9 && s1 <= 1e-9 && s2 <= 1e-9
    }
}

/// Completes an endpoint `x` of a segment assigned to `fan`: already a ring
/// vertex (done), strictly inside (goes to inner) or on a boundary edge
/// (splices into the ring in order).
fn complete_endpoint(
    fan: &mut Piece,
    x: usize,
    pts: &[WPt],
    flat2: impl Copy + Fn(D3) -> [f64; 2],
) -> Result<()> {
    if fan.ring.contains(&x) || fan.inner.contains(&x) {
        return Ok(());
    }
    let p = flat2(pts[x].pos);
    // Strictly inside: not within tolerance of any ring edge.
    let mut on_edge = None;
    for i in 0..fan.ring.len() {
        let (a, b) = (
            flat2(pts[fan.ring[i]].pos),
            flat2(pts[fan.ring[(i + 1) % fan.ring.len()]].pos),
        );
        let (d, t) = point_seg2(p, a, b);
        let len = (b[0] - a[0]).hypot(b[1] - a[1]).max(1e-300);
        if d <= 1e-9 * len && t > 1e-9 && t < 1.0 - 1e-9 {
            on_edge = Some(i);
            break;
        }
    }
    if let Some(i) = on_edge {
        // Splice into the ring after `i`, ordered along the edge.
        let mut at = i + 1;
        while at < fan.ring.len() + i {
            let prev = fan.ring[at % fan.ring.len()];
            let (d, _) = point_seg2(p, flat2(pts[fan.ring[i]].pos), flat2(pts[prev].pos));
            // Stop once past `x` along the edge.
            let prev_p = flat2(pts[prev].pos);
            let edge_p = flat2(pts[fan.ring[i]].pos);
            let along = (p[0] - edge_p[0]) * (prev_p[0] - edge_p[0])
                + (p[1] - edge_p[1]) * (prev_p[1] - edge_p[1]);
            let _ = d;
            if along <= 0.0 {
                break;
            }
            at += 1;
            if at - i - 1 > fan.ring.len() {
                break;
            }
        }
        fan.ring.insert(at % (fan.ring.len() + 1), x);
        return Ok(());
    }
    push(&mut fan.inner, x)
}
fn fan_contains(fan: &Piece, p: usize, pts: &[WPt], flat2: impl Copy + Fn(D3) -> [f64; 2]) -> bool {
    if fan.ring.len() != 3 {
        return point_in_piece(flat2(pts[p].pos), fan, pts, flat2);
    }
    let q = flat2(pts[p].pos);
    let t = [
        flat2(pts[fan.ring[0]].pos),
        flat2(pts[fan.ring[1]].pos),
        flat2(pts[fan.ring[2]].pos),
    ];
    let o = orient2(t[0], t[1], t[2]);
    let (s0, s1, s2) = (
        orient2(t[0], t[1], q),
        orient2(t[1], t[2], q),
        orient2(t[2], t[0], q),
    );
    if o >= 0.0 {
        s0 >= -1e-9 && s1 >= -1e-9 && s2 >= -1e-9
    } else {
        s0 <= 1e-9 && s1 <= 1e-9 && s2 <= 1e-9
    }
}

/// 2D distance from `p` to segment `a..b`, with the parameter.
fn point_seg2(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> (f64, f64) {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1];
    if len2 == 0.0 {
        return (((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2)).sqrt(), 0.0);
    }
    let t = ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / len2;
    let c = [
        a[0] + t.clamp(0.0, 1.0) * ab[0],
        a[1] + t.clamp(0.0, 1.0) * ab[1],
    ];
    (((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt(), t)
}

/// Classifies every triangle of `mesh` against the tool and records what to
/// keep: A drops and rewrites, tool adds.
#[allow(clippy::too_many_arguments)]
fn classify_side(
    mesh: &Mesh,
    pieces: &BTreeMap<TriangleId, SplitTris>,
    side: &Side,
    tool: &ToolRef<'_>,
    op: BooleanOp,
    is_a: bool,
    plan: &mut BooleanPlan,
    fresh_of: &mut BTreeMap<[u32; 3], u32>,
) -> Result<()> {
    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        let whole: [PRef; 3] = match side {
            Side::A => tri.map(PRef::Live),
            Side::B => {
                let mut v = [PRef::Fresh(0); 3];
                for (k, p) in tri_pos(mesh, t).iter().enumerate() {
                    v[k] = fresh(*p, plan, fresh_of)?;
                }
                v
            }
        };
        let owned;
        let (candidates, edge_split): (&[[PRef; 3]], bool) = match pieces.get(&t) {
            Some(split) => (&split.pieces, split.edge_split),
            None => {
                owned = vec![whole];
                (&owned, false)
            }
        };
        // Classify into a fresh vector first.
        let mut kept: Vec<[PRef; 3]> = Vec::new();
        for c in candidates {
            let (centroid, normal) = piece_geom(mesh, c, plan)?;
            if keep_piece(tool, centroid, normal, op, is_a, plan.tol)? {
                push(&mut kept, *c)?;
            }
        }
        match side {
            Side::A => {
                if kept.is_empty() {
                    push(&mut plan.drop_a, t)?;
                } else if pieces.contains_key(&t) && (kept.len() < candidates.len() || edge_split) {
                    // Materialise the splits: a dropped piece exposes them,
                    // and shared edge splits keep the neighbour consistent.
                    push(&mut plan.rewrite, (t, kept))?;
                }
                // Else the whole triangle stays: untouched, or split with
                // every piece kept and no shared edge split.
            }
            Side::B => {
                for mut k in kept {
                    if op == BooleanOp::Subtract {
                        // The cavity wall faces into the removed volume.
                        k.swap(1, 2);
                    }
                    push(&mut plan.tool, k)?;
                }
            }
        }
    }
    Ok(())
}

/// Centroid and (unnormalized) normal of a planned piece.
fn piece_geom(mesh: &Mesh, tri: &[PRef; 3], plan: &BooleanPlan) -> Result<(D3, D3)> {
    let mut p = [[0.0; 3]; 3];
    for (k, r) in tri.iter().enumerate() {
        p[k] = match r {
            PRef::Live(v) => mesh
                .position(*v)
                .ok_or(OpsError::InvalidInput("boolean triangle lost a vertex"))?
                .map(f64::from),
            PRef::Fresh(i) => plan
                .fresh
                .get(*i as usize)
                .ok_or(OpsError::InvalidInput("boolean plan is not ready"))?
                .map(f64::from),
        };
    }
    let c = std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0);
    Ok((c, cross(sub(p[1], p[0]), sub(p[2], p[0]))))
}

/// Whether a piece with centroid `c` and normal `n` survives `op`.
fn keep_piece(
    tool: &ToolRef<'_>,
    c: D3,
    n: D3,
    op: BooleanOp,
    is_a: bool,
    tol: f32,
) -> Result<bool> {
    let tol = tol as f64;
    let (inside, on_surface) = parity(tool, c, tol)?;
    if !on_surface {
        return Ok(match (op, is_a) {
            (BooleanOp::Union, _) => !inside,
            (BooleanOp::Subtract, true) => !inside,
            (BooleanOp::Subtract, false) => inside,
            (BooleanOp::Intersect, _) => inside,
        });
    }
    // Coplanar: the centroid lies on the tool surface. Flush overlaps
    // (same winding) probe behind the piece, into the solid; opposing
    // overlaps (face-to-face touches) probe in front of it.
    let len = dot(n, n).sqrt();
    if len == 0.0 {
        return Err(OpsError::InvalidInput("boolean piece lost its normal"));
    }
    let u = n.map(|x| x / len);
    let eps = 64.0 * tol;
    let nt = tool_normal(tool, c)?;
    let flush = dot(u, nt) > 0.0;
    let p_out = std::array::from_fn(|k| c[k] + eps * u[k]);
    let (out_inside, _) = parity(tool, p_out, tol)?;
    let p_in = std::array::from_fn(|k| c[k] - eps * u[k]);
    let (in_inside, _) = parity(tool, p_in, tol)?;
    Ok(match (op, is_a) {
        // B's coplanar copies always go: A's copy covers the patch (or the
        // patch is interior and both go). A survives when the tool does not
        // cover its front: a flush overlap is still surface (void in
        // front), while a face-to-face contact is interior.
        (BooleanOp::Union, true) => !out_inside,
        (BooleanOp::Union, false) => false,
        (BooleanOp::Subtract, true) => {
            if flush {
                !in_inside
            } else {
                true
            }
        }
        (BooleanOp::Subtract, false) => false,
        (BooleanOp::Intersect, true) => true,
        (BooleanOp::Intersect, false) => false,
    })
}

/// Inside/outside of `p` against the tool, by majority vote over three
/// axes, plus whether any axis starts on the tool surface.
fn parity(tool: &ToolRef<'_>, p: D3, tol: f64) -> Result<(bool, bool)> {
    let origin = [p[0] as f32, p[1] as f32, p[2] as f32];
    if !origin.iter().all(|c| c.is_finite()) {
        return Err(OpsError::InvalidInput("boolean point leaves f32 range"));
    }
    let mut inside_votes = 0;
    let mut on_surface = false;
    let mut hits: Vec<fast_mesh_query::RayHit> = Vec::new();
    for dir in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
        hits.clear();
        let ray = Ray { origin, dir };
        let opts = RayOptions {
            min_t: 0.0,
            max_t: f32::INFINITY,
            cull: Cull::None,
        };
        raycast_all(tool.mesh, tool.bvh, &ray, &opts, &mut hits)?;
        // Merge hits at the same surface (one face is two triangles).
        hits.sort_by(|x, y| x.t.total_cmp(&y.t));
        let mut groups = 0;
        let mut first = f32::INFINITY;
        let mut prev = f32::NEG_INFINITY;
        for h in &hits {
            if (h.t - prev) as f64 > 1e-6 * (1.0 + h.t as f64) {
                groups += 1;
                if first.is_infinite() {
                    first = h.t;
                }
            }
            prev = h.t;
        }
        if groups % 2 == 1 {
            inside_votes += 1;
        }
        if (first as f64) <= tol {
            on_surface = true;
        }
    }
    Ok((inside_votes >= 2, on_surface))
}

/// The tool's outward normal at the surface nearest `p`.
fn tool_normal(tool: &ToolRef<'_>, p: D3) -> Result<D3> {
    let q = [p[0] as f32, p[1] as f32, p[2] as f32];
    let hit = closest_point(
        tool.mesh,
        tool.bvh,
        q,
        &ClosestOptions {
            max_distance: f32::INFINITY,
        },
    )?
    .ok_or(OpsError::InvalidInput("boolean tool has no triangles"))?;
    let tri = tool
        .mesh
        .triangle(hit.triangle)
        .ok_or(OpsError::InvalidInput("boolean tool lost a triangle"))?;
    let pos = tri.map(|v| {
        tool.mesh
            .position(v)
            .expect("tool triangle is live")
            .map(f64::from)
    });
    let n = cross(sub(pos[1], pos[0]), sub(pos[2], pos[0]));
    if dot(n, n) == 0.0 {
        return Err(OpsError::InvalidInput("boolean tool lost its normal"));
    }
    Ok(n)
}

/// Stitches the rims: every boundary edge is split at the other side's rim
/// vertices within `tol` (reusing their ids), closing the curve on shared
/// vertices. Returns the triangles added by the splits.
fn stitch(tx: &mut Transaction, tool_ids: &[TriangleId], tol: f32) -> Result<usize> {
    let mesh = tx.mesh();
    let loops = crate::boundary_loops(mesh)?;
    if loops.is_empty() {
        return Ok(0);
    }
    let tool_set: BTreeMap<TriangleId, ()> = tool_ids.iter().map(|&t| (t, ())).collect();
    // Boundary edges with their triangle and side (tool or A).
    let mut edges: Vec<((VertexId, VertexId), TriangleId, bool)> = Vec::new();
    for l in &loops {
        let n = l.vertices.len();
        if n < 2 {
            continue;
        }
        let end = if l.closed { n } else { n - 1 };
        for i in 0..end {
            let (u, v) = (l.vertices[i], l.vertices[(i + 1) % n]);
            // The boundary edge's single triangle.
            let mut user = None;
            for t in 0..mesh.triangle_slots() as TriangleId {
                if let Some(tri) = mesh.triangle(t)
                    && tri.contains(&u)
                    && tri.contains(&v)
                {
                    user = Some(t);
                    break;
                }
            }
            let Some(t) = user else { continue };
            push(&mut edges, ((u, v), t, tool_set.contains_key(&t)))?;
        }
    }
    // Rim vertices per side.
    let mut rim_a: Vec<VertexId> = Vec::new();
    let mut rim_b: Vec<VertexId> = Vec::new();
    for ((u, v), _, is_tool) in &edges {
        for w in [*u, *v] {
            let rim = if *is_tool { &mut rim_b } else { &mut rim_a };
            if !rim.contains(&w) {
                push(rim, w)?;
            }
        }
    }
    // Positions of every rim vertex, read once so the borrow ends before
    // the writes below.
    let mut positions: BTreeMap<VertexId, D3> = BTreeMap::new();
    for w in rim_a.iter().chain(rim_b.iter()) {
        if let Some(p) = mesh.position(*w) {
            positions.insert(*w, p.map(f64::from));
        }
    }
    let pos = |v: VertexId| positions[&v];
    // Splits per triangle.
    let mut splits: BTreeMap<TriangleId, Vec<(VertexId, VertexId, VertexId)>> = BTreeMap::new();
    for ((u, v), t, is_tool) in &edges {
        let others = if *is_tool { &rim_a } else { &rim_b };
        let (pu, pv) = (pos(*u), pos(*v));
        for &w in others {
            // Vertices at the edge's ends merge in the weld below.
            let (du, dv) = (sub(pos(w), pu), sub(pos(w), pv));
            if dot(du, du).sqrt() <= tol as f64 || dot(dv, dv).sqrt() <= tol as f64 {
                continue;
            }
            let (d, _) = point_seg(pos(w), pu, pv);
            if d <= tol as f64 {
                splits.entry(*t).or_default().push((*u, *v, w));
            }
        }
    }
    if splits.is_empty() {
        return Ok(0);
    }
    let mut added = 0usize;
    for (t, ss) in splits {
        let tri = tx.mesh().triangle(t).expect("split of a live triangle");
        let mut tris = vec![tri];
        // Group splits per edge, ordered along it.
        let mut by_edge: BTreeMap<(VertexId, VertexId), Vec<VertexId>> = BTreeMap::new();
        for (u, v, w) in ss {
            by_edge.entry((u, v)).or_default().push(w);
        }
        for ((u, v), mut ws) in by_edge {
            let (pu, pv) = (pos(u), pos(v));
            ws.sort_by(|&x, &y| {
                let tx_ = dot(sub(pos(x), pu), sub(pv, pu));
                let ty_ = dot(sub(pos(y), pu), sub(pv, pu));
                tx_.total_cmp(&ty_)
            });
            ws.dedup();
            // Split the one triangle holding the edge.
            let at = tris
                .iter()
                .position(|tri| tri.contains(&u) && tri.contains(&v))
                .expect("split edge of a live triangle");
            let tri = tris.remove(at);
            split_chain(tri, u, v, &ws, &mut tris)?;
        }
        if tris.len() <= 1 {
            continue;
        }
        tx.set_triangles(&[t], &[tris[0]])?;
        tx.add_triangles(&tris[1..])?;
        added += tris.len() - 1;
    }
    Ok(added)
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
}
