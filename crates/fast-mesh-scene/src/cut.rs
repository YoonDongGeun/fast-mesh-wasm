//! Cutting a mesh by a world plane into two capped meshes.
//!
//! [`Scene::cut_by_plane`] maps the world plane into the mesh's local frame
//! (like a section, but without needing a BVH), splits the mesh with
//! [`ops::cut_by_plane`](fast_mesh_ops::cut_by_plane) and adds the removed
//! side as a new mesh with the same placement and update mode. The original
//! handle keeps the `+` side; the report's `added` is the `-` side.
//!
//! Only positions and triangles cross over (see the ops docs); the new mesh
//! has no normals, colouring, fields or section entries. Undo restores the
//! cut mesh in one step; the added mesh stays, like the removed source of
//! [`Scene::join`].

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use fast_mesh_query::{ClosestOptions, closest_point};
use fast_mesh_store::{Transaction, TriangleId, VertexId};

use crate::fields::baked_world_mesh;
use crate::ops::{CutOptions as OpsCutOptions, cut_by_plane as ops_cut_by_plane};
use crate::placement::{local_to_world, world_to_local};
use crate::{MeshId, MeshOptions, Plane, Result, Scene, SceneError};

/// What to cut: a world plane and whether to cap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutOptions {
    /// The cutting plane, in world coordinates. The `+` side
    /// (`signed_distance >= 0`) stays on the original mesh.
    pub plane: Plane,
    /// Close both openings with flat, unfaired patches, across UV seams on a
    /// mesh with UVs. Without it both
    /// parts keep the open rim the cut leaves.
    pub cap: bool,
}

/// What [`Scene::cut_by_plane`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CutReport {
    /// The new mesh holding the `-` side.
    pub added: MeshId,
    /// Alive triangles of the kept (`+`) mesh, cap included.
    pub kept_triangles: usize,
    /// Alive triangles of the added (`-`) mesh, cap included.
    pub removed_triangles: usize,
    /// Cap triangles added on both sides together.
    pub cap_triangles: usize,
}

/// What polyline to trace and how to trim it in [`Scene::trim_by_curve`].
#[derive(Debug, Clone, PartialEq)]
pub struct TrimCurveOptions {
    /// World-space points defining the trim polyline. When `closed` is `true`,
    /// a closing segment connects the last point back to the first.
    pub points: Vec<[f64; 3]>,
    /// Whether the polyline forms a closed loop whose enclosed side is removed.
    pub closed: bool,
    /// When `closed` is `true`, removes the enclosed interior side (`true`) or
    /// the surrounding exterior side (`false`). Ignored when `closed` is `false`.
    pub delete_inside: bool,
    /// World-space snap tolerance onto nearby mesh vertices (`>= 0.0`).
    pub snap_tolerance: f64,
}

impl Default for TrimCurveOptions {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            closed: true,
            delete_inside: true,
            snap_tolerance: 0.0,
        }
    }
}

/// Summary of what [`Scene::trim_by_curve`] changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrimReport {
    /// New live triangles remaining after splitting and trimming.
    pub triangles_added: usize,
    /// Original live triangles that were split or trimmed away.
    pub triangles_removed: usize,
    /// New live vertices created along the curve and seam splits.
    pub vertices_added: usize,
    /// Distinct undirected mesh edges forming the traced seam.
    pub seam_edges: usize,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    a.map(|c| c / n)
}

/// The world plane in the mesh's stored local frame (origin excluded), as
/// for a section: the axes are the world `(u, v)` mapped through the
/// placement's linear inverse, then re-orthonormalised, so that the local
/// plane's zero set is the preimage of the world's, with matching sides.
/// `s >= 0` then means the same side in both frames.
fn local_plane(
    placement: &crate::Placement,
    mesh_origin: [f64; 3],
    plane: &Plane,
) -> Result<Plane> {
    let origin = world_to_local(placement, mesh_origin, plane.origin());
    let u = unit(placement.inverse_direction(plane.u()));
    let v = placement.inverse_direction(plane.v());
    let d = dot(v, u);
    let v = unit(std::array::from_fn(|k| v[k] - d * u[k]));
    Ok(Plane::with_axes(origin, u, v)?)
}

impl Scene {
    /// Cuts the mesh with a world plane: the `+` side
    /// (`signed_distance >= 0`) stays on the mesh and the `-` side is added
    /// as a new mesh, both with the mesh's placement, so the world shape is
    /// kept. With `options.cap` (the usual case) both openings are closed
    /// with flat, unfaired patches; without it both parts keep the open rim.
    /// Cutting a closed mesh with `cap` yields two closed meshes. On a mesh
    /// with UVs, closed means closed by exact position, so seams count as
    /// closed. A rim that runs into a real boundary stays open; on a mesh
    /// without UVs, split vertices are such a boundary, so weld them first.
    ///
    /// The cut and the add are one undo step (the add is undoable, see
    /// [`MeshOptions::undoable`]): undo restores the mesh and removes the
    /// added one, redo cuts again and brings it back under the same id.
    /// Only positions and triangles cross over: the added mesh has no
    /// normals, colouring, fields or section entries, and new vertices take
    /// channel defaults (crossing vertices on the kept side interpolate
    /// their parents). UVs carry over to both sides, interpolated at the
    /// crossing vertices and across the caps.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id;
    /// [`SceneError::InvalidInput`] when the plane does not cut the mesh, or
    /// a rim cannot be capped; [`SceneError::Ops`] or [`SceneError::Store`]
    /// from the cut or the add. A failed cut changes nothing. If adding the
    /// removed side fails (out of memory), the cut of the original stays
    /// recorded and the added mesh is missing.
    pub fn cut_by_plane(&mut self, id: MeshId, options: &CutOptions) -> Result<CutReport> {
        self.drop_field_unit();
        let (local, placement, mode) = {
            let entry = self.entry(id)?;
            (
                local_plane(&entry.placement, entry.mesh.origin(), &options.plane)?,
                entry.placement,
                entry.mode,
            )
        };
        // The edit and the add are one step, or part of the caller's open
        // one: from here to `end_step` nothing returns early.
        self.begin_step();
        let cut = self
            .edit(id, |tx| {
                ops_cut_by_plane(tx, &local, &OpsCutOptions { cap: options.cap })
            })
            .and_then(|output| {
                let added = self.add_mesh(
                    output.removed,
                    MeshOptions {
                        placement,
                        mode,
                        undoable: true,
                        ..MeshOptions::default()
                    },
                )?;
                Ok(CutReport {
                    added,
                    kept_triangles: output.report.kept_triangles,
                    removed_triangles: output.report.removed_triangles,
                    cap_triangles: output.report.cap_triangles,
                })
            });
        self.end_step()?;
        cut
    }

    /// Conformatively imprints a world-space polyline (`options.points`) onto
    /// mesh `id`, optionally closing the loop and trimming away the interior
    /// (`options.delete_inside == true`) or exterior side in a single undoable
    /// edit step.
    ///
    /// Waypoints project onto the closest points on the mesh surface, subdivide
    /// triangles/shared edges conformally (interpolating vertex attributes via
    /// transaction parents), march/trace across crossed edges with a Dijkstra
    /// fallback on sharp creases, and flood-fill non-seam edges to delete the
    /// selected side when `options.closed` is `true`.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] if `id` is stale, [`SceneError::InvalidInput`]
    /// for non-finite inputs or fewer than 2 points (3 when `closed`), or
    /// [`SceneError::StaleBvh`] under `UpdateMode::Manual` when stale.
    pub fn trim_by_curve(&mut self, id: MeshId, options: &TrimCurveOptions) -> Result<TrimReport> {
        self.drop_field_unit();
        validate_trim_options(options)?;
        self.entry(id)?;
        self.prepare(Some(&[id]))?;

        let (waypoints_init, placement, origin, local_snap_tol) = {
            let entry = self.entry(id)?;
            if entry.mesh.triangle_count() == 0 {
                return Err(SceneError::InvalidInput("mesh has no triangles to trim"));
            }
            let uniform = entry.placement.uniform_scale();
            let hits = if uniform.is_some() {
                let mut hits = Vec::with_capacity(options.points.len());
                for &world_pt in &options.points {
                    let local = world_to_local(&entry.placement, entry.mesh.origin(), world_pt)
                        .map(|c| c as f32);
                    let hit =
                        closest_point(&entry.mesh, &entry.bvh, local, &ClosestOptions::default())?
                            .ok_or(SceneError::InvalidInput(
                                "failed to project trim curve onto mesh",
                            ))?;
                    hits.push((hit.triangle, local, hit.barycentric));
                }
                hits
            } else {
                let (baked, baked_bvh) = baked_world_mesh(entry)?;
                let mut hits = Vec::with_capacity(options.points.len());
                for &world_pt in &options.points {
                    let wp = world_pt.map(|c| c as f32);
                    let hit = closest_point(&baked, &baked_bvh, wp, &ClosestOptions::default())?
                        .ok_or(SceneError::InvalidInput(
                            "failed to project trim curve onto mesh",
                        ))?;
                    let local = world_to_local(&entry.placement, entry.mesh.origin(), world_pt)
                        .map(|c| c as f32);
                    hits.push((hit.triangle, local, hit.barycentric));
                }
                hits
            };
            let min_scale = entry.placement.scale[0]
                .min(entry.placement.scale[1])
                .min(entry.placement.scale[2]);
            let local_snap = if min_scale > 0.0 {
                options.snap_tolerance / min_scale
            } else {
                0.0
            };
            (hits, entry.placement, entry.mesh.origin(), local_snap)
        };

        let closed = options.closed;
        let delete_inside = options.delete_inside;
        self.edit(id, move |tx| {
            run_trim_in_tx(
                tx,
                &waypoints_init,
                closed,
                delete_inside,
                local_snap_tol,
                &placement,
                origin,
                options.snap_tolerance,
            )
        })
    }
}

fn validate_trim_options(options: &TrimCurveOptions) -> Result<()> {
    let min_pts = if options.closed { 3 } else { 2 };
    if options.points.len() < min_pts {
        return Err(SceneError::InvalidInput(if options.closed {
            "closed trim curve requires at least 3 points"
        } else {
            "open trim curve requires at least 2 points"
        }));
    }
    if !options
        .points
        .iter()
        .all(|p| p.iter().all(|c| c.is_finite()))
    {
        return Err(SceneError::InvalidInput("trim curve points must be finite"));
    }
    if !options.snap_tolerance.is_finite() || options.snap_tolerance < 0.0 {
        return Err(SceneError::InvalidInput(
            "snap_tolerance must be finite and non-negative",
        ));
    }
    Ok(())
}

#[inline]
fn edge_key(a: VertexId, b: VertexId) -> (VertexId, VertexId) {
    if a < b { (a, b) } else { (b, a) }
}

struct TrimMesh<'a, 'b> {
    tx: &'a mut Transaction<'b>,
    tris: Vec<Option<([VertexId; 3], TriangleId)>>,
    orig_to_logical: Vec<Option<usize>>,
    children: HashMap<usize, Vec<usize>>,
    edge_tris: HashMap<(VertexId, VertexId), Vec<usize>>,
    vert_tris: HashMap<VertexId, Vec<usize>>,
    new_verts: Vec<VertexId>,
    init_logical_len: usize,
    orig_tri_count: usize,
}

#[derive(Clone, Copy)]
struct SnapContext<'a> {
    local_tol: f64,
    world_tol: f64,
    placement: &'a crate::Placement,
    origin: [f64; 3],
}

impl<'a, 'b> TrimMesh<'a, 'b> {
    fn new(tx: &'a mut Transaction<'b>) -> Self {
        let slots = tx.mesh().triangle_slots();
        let mut tris = Vec::with_capacity(slots);
        let mut orig_to_logical = vec![None; slots];
        let mut edge_tris: HashMap<(VertexId, VertexId), Vec<usize>> = HashMap::new();
        let mut vert_tris: HashMap<VertexId, Vec<usize>> = HashMap::new();

        for (s, slot) in orig_to_logical.iter_mut().enumerate() {
            let tid = s as TriangleId;
            if let Some(verts) = tx.mesh().triangle(tid) {
                let lt = tris.len();
                tris.push(Some((verts, tid)));
                *slot = Some(lt);
                for k in 0..3 {
                    let a = verts[k];
                    let b = verts[(k + 1) % 3];
                    edge_tris.entry(edge_key(a, b)).or_default().push(lt);
                    vert_tris.entry(a).or_default().push(lt);
                }
            }
        }
        let init_logical_len = tris.len();
        Self {
            tx,
            tris,
            orig_to_logical,
            children: HashMap::new(),
            edge_tris,
            vert_tris,
            new_verts: Vec::new(),
            init_logical_len,
            orig_tri_count: init_logical_len,
        }
    }

    #[inline]
    fn pos(&self, v: VertexId) -> [f64; 3] {
        self.tx.mesh().positions()[v as usize].map(f64::from)
    }

    #[inline]
    fn has_edge(&self, a: VertexId, b: VertexId) -> bool {
        self.edge_tris
            .get(&edge_key(a, b))
            .is_some_and(|list| !list.is_empty())
    }

    fn detach_logical(&mut self, lt: usize) -> Option<([VertexId; 3], TriangleId)> {
        let (verts, store_id) = self.tris[lt].take()?;
        for k in 0..3 {
            let a = verts[k];
            let b = verts[(k + 1) % 3];
            let key = edge_key(a, b);
            if let Some(list) = self.edge_tris.get_mut(&key) {
                list.retain(|&id| id != lt);
                if list.is_empty() {
                    self.edge_tris.remove(&key);
                }
            }
            if let Some(list) = self.vert_tris.get_mut(&a) {
                list.retain(|&id| id != lt);
            }
        }
        Some((verts, store_id))
    }

    fn push_child(
        &mut self,
        parent_lt: usize,
        verts: [VertexId; 3],
    ) -> std::result::Result<usize, SceneError> {
        let store_id = self.tx.add_triangles(&[verts])?[0];
        let lt = self.tris.len();
        self.tris.push(Some((verts, store_id)));
        for k in 0..3 {
            let a = verts[k];
            let b = verts[(k + 1) % 3];
            self.edge_tris.entry(edge_key(a, b)).or_default().push(lt);
            self.vert_tris.entry(a).or_default().push(lt);
        }
        self.children.entry(parent_lt).or_default().push(lt);
        Ok(lt)
    }

    fn split_edge(
        &mut self,
        a: VertexId,
        b: VertexId,
        t: f64,
        snap: SnapContext<'_>,
    ) -> std::result::Result<VertexId, SceneError> {
        let t = t.clamp(0.0, 1.0);
        let pa = self.pos(a);
        let pb = self.pos(b);
        let p = [
            pa[0] + (pb[0] - pa[0]) * t,
            pa[1] + (pb[1] - pa[1]) * t,
            pa[2] + (pb[2] - pa[2]) * t,
        ];
        if let Some(snap_v) = check_vertex_snap(&[a, b], &[pa, pb], p, &[1.0 - t, t], 1e-5, snap) {
            return Ok(snap_v);
        }

        let new_v = self.tx.add_vertex_with_parents(
            p.map(|c| c as f32),
            &[(a, (1.0 - t) as f32), (b, t as f32)],
        )?;
        self.new_verts.push(new_v);

        let incident = self
            .edge_tris
            .get(&edge_key(a, b))
            .cloned()
            .unwrap_or_default();
        for lt in incident {
            let Some((verts, store_id)) = self.detach_logical(lt) else {
                continue;
            };
            self.tx.remove_triangles(&[store_id])?;
            let (x, y, z) = orient_edge_in_triangle(verts, a, b);
            self.push_child(lt, [x, new_v, z])?;
            self.push_child(lt, [new_v, y, z])?;
        }
        Ok(new_v)
    }

    fn split_triangle_interior(
        &mut self,
        lt: usize,
        bary: [f64; 3],
        snap: SnapContext<'_>,
    ) -> std::result::Result<VertexId, SceneError> {
        let [v0, v1, v2] = self.tris[lt].ok_or(SceneError::Corrupted)?.0;
        let ps = [self.pos(v0), self.pos(v1), self.pos(v2)];
        let sum = (bary[0] + bary[1] + bary[2]).max(1e-30);
        let w = [
            (bary[0] / sum).max(0.0),
            (bary[1] / sum).max(0.0),
            (bary[2] / sum).max(0.0),
        ];
        let p = [
            ps[0][0] * w[0] + ps[1][0] * w[1] + ps[2][0] * w[2],
            ps[0][1] * w[0] + ps[1][1] * w[1] + ps[2][1] * w[2],
            ps[0][2] * w[0] + ps[1][2] * w[1] + ps[2][2] * w[2],
        ];
        if let Some(snap_v) = check_vertex_snap(&[v0, v1, v2], &ps, p, &w, 1e-5, snap) {
            return Ok(snap_v);
        }

        // If the point lies on one of the triangle edges, split the edge conformally.
        let edge_eps = 1e-5;
        if w[2] <= edge_eps && w[0] + w[1] > 1e-12 {
            return self.split_edge(v0, v1, w[1] / (w[0] + w[1]), snap);
        }
        if w[0] <= edge_eps && w[1] + w[2] > 1e-12 {
            return self.split_edge(v1, v2, w[2] / (w[1] + w[2]), snap);
        }
        if w[1] <= edge_eps && w[2] + w[0] > 1e-12 {
            return self.split_edge(v2, v0, w[0] / (w[2] + w[0]), snap);
        }

        let new_v = self.tx.add_vertex_with_parents(
            p.map(|c| c as f32),
            &[(v0, w[0] as f32), (v1, w[1] as f32), (v2, w[2] as f32)],
        )?;
        self.new_verts.push(new_v);
        let (_, store_id) = self.detach_logical(lt).ok_or(SceneError::Corrupted)?;
        self.tx.remove_triangles(&[store_id])?;
        self.push_child(lt, [v0, v1, new_v])?;
        self.push_child(lt, [v1, v2, new_v])?;
        self.push_child(lt, [v2, v0, new_v])?;
        Ok(new_v)
    }

    fn collect_live_descendants(&self, root_lt: usize, out: &mut Vec<usize>) {
        let mut stack = vec![root_lt];
        while let Some(lt) = stack.pop() {
            if self.tris[lt].is_some() {
                out.push(lt);
            } else if let Some(ch) = self.children.get(&lt) {
                stack.extend_from_slice(ch);
            }
        }
    }
}

fn check_vertex_snap(
    verts: &[VertexId],
    ps: &[[f64; 3]],
    p: [f64; 3],
    weights: &[f64],
    bary_eps: f64,
    snap: SnapContext<'_>,
) -> Option<VertexId> {
    for (&v, &w) in verts.iter().zip(weights.iter()) {
        if w >= 1.0 - bary_eps {
            return Some(v);
        }
    }
    if snap.local_tol <= 0.0 && snap.world_tol <= 0.0 {
        return None;
    }
    let pw = local_to_world(snap.placement, snap.origin, p);
    let mut best = None;
    let mut best_d2 = snap.world_tol * snap.world_tol;
    for (&v, &vp) in verts.iter().zip(ps.iter()) {
        let vw = local_to_world(snap.placement, snap.origin, vp);
        let d2 = dist_sq(vw, pw);
        if d2 <= best_d2 {
            best_d2 = d2;
            best = Some(v);
        }
    }
    best
}

fn orient_edge_in_triangle(
    tri: [VertexId; 3],
    a: VertexId,
    b: VertexId,
) -> (VertexId, VertexId, VertexId) {
    for i in 0..3 {
        let u = tri[i];
        let v = tri[(i + 1) % 3];
        let w = tri[(i + 2) % 3];
        if (u == a && v == b) || (u == b && v == a) {
            return (u, v, w);
        }
    }
    (tri[0], tri[1], tri[2])
}

#[inline]
fn dist_sq(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

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

/// Closest barycentric coordinates on triangle `(a, b, c)` to point `p`.
fn closest_barycentric_on_triangle(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    p: [f64; 3],
) -> (f64, [f64; 3]) {
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    let ap = sub3(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (dist_sq(p, a), [1.0, 0.0, 0.0]);
    }
    let bp = sub3(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (dist_sq(p, b), [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let q = [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]];
        return (dist_sq(p, q), [1.0 - v, v, 0.0]);
    }
    let cp = sub3(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (dist_sq(p, c), [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let q = [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]];
        return (dist_sq(p, q), [1.0 - w, 0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = sub3(c, b);
        let q = [b[0] + w * bc[0], b[1] + w * bc[1], b[2] + w * bc[2]];
        return (dist_sq(p, q), [0.0, 1.0 - w, w]);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let u = 1.0 - v - w;
    let q = [
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    ];
    (dist_sq(p, q), [u, v, w])
}

#[allow(clippy::too_many_arguments)]
fn run_trim_in_tx(
    tx: &mut Transaction<'_>,
    waypoints_init: &[(TriangleId, [f32; 3], [f32; 3])],
    closed: bool,
    delete_inside: bool,
    local_snap_tol: f64,
    placement: &crate::Placement,
    origin: [f64; 3],
    world_snap_tol: f64,
) -> std::result::Result<TrimReport, SceneError> {
    let mut tm = TrimMesh::new(tx);
    let snap = SnapContext {
        local_tol: local_snap_tol,
        world_tol: world_snap_tol,
        placement,
        origin,
    };
    let mut curve_verts: Vec<VertexId> = Vec::with_capacity(waypoints_init.len());

    // Step 1: insert/snap all waypoints onto the mesh.
    for &(init_tri, target_local, init_bary) in waypoints_init {
        let root_lt = tm.orig_to_logical[init_tri as usize].ok_or(SceneError::Corrupted)?;
        let (target_lt, bary) = if tm.tris[root_lt].is_some() {
            (root_lt, init_bary.map(f64::from))
        } else {
            let mut cands = Vec::new();
            tm.collect_live_descendants(root_lt, &mut cands);
            let target_f64 = target_local.map(f64::from);
            let mut best = None;
            let mut best_d2 = f64::INFINITY;
            for lt in cands {
                let Some((verts, _)) = tm.tris[lt] else {
                    continue;
                };
                let (d2, b) = closest_barycentric_on_triangle(
                    tm.pos(verts[0]),
                    tm.pos(verts[1]),
                    tm.pos(verts[2]),
                    target_f64,
                );
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = Some((lt, b));
                }
            }
            best.ok_or(SceneError::Corrupted)?
        };
        let v = tm.split_triangle_interior(target_lt, bary, snap)?;
        if curve_verts.last() != Some(&v) {
            curve_verts.push(v);
        }
    }

    if closed && curve_verts.len() >= 2 && curve_verts.first() == curve_verts.last() {
        curve_verts.pop();
    }

    // Step 2: trace conformal segments between consecutive curve vertices.
    let mut seam_set: HashSet<(VertexId, VertexId)> = HashSet::new();
    let seg_count = if closed {
        curve_verts.len()
    } else {
        curve_verts.len().saturating_sub(1)
    };
    if curve_verts.len() >= 2 {
        for s in 0..seg_count {
            let a = curve_verts[s];
            let b = curve_verts[(s + 1) % curve_verts.len()];
            if a != b {
                trace_curve_segment(&mut tm, a, b, snap, &mut seam_set)?;
            }
        }
    }

    // Step 3: when closed, flood-fill non-seam edges and remove the chosen side.
    if closed && !seam_set.is_empty() {
        remove_closed_region(&mut tm, &seam_set, delete_inside)?;
    }

    // Prune vertices newly created/touched that ended up isolated after triangle removal.
    prune_orphaned_vertices(&mut tm)?;

    let mut seam_alive = 0usize;
    for &(a, b) in &seam_set {
        if tm.has_edge(a, b) {
            seam_alive += 1;
        }
    }

    let orig_survived = (0..tm.init_logical_len)
        .filter(|&lt| tm.tris[lt].is_some())
        .count();
    let final_live = tm.tris.iter().filter(|t| t.is_some()).count();
    let triangles_removed = tm.orig_tri_count - orig_survived;
    let triangles_added = final_live - orig_survived;
    let vertices_added = tm
        .new_verts
        .iter()
        .filter(|&&v| tm.tx.mesh().is_vertex_alive(v))
        .count();

    Ok(TrimReport {
        triangles_added,
        triangles_removed,
        vertices_added,
        seam_edges: seam_alive,
    })
}

fn trace_curve_segment(
    tm: &mut TrimMesh<'_, '_>,
    v_start: VertexId,
    v_end: VertexId,
    snap: SnapContext<'_>,
    seam_set: &mut HashSet<(VertexId, VertexId)>,
) -> std::result::Result<(), SceneError> {
    let mut cur = v_start;
    let mut prev: Option<VertexId> = None;
    let mut visited = HashSet::new();
    visited.insert(cur);
    let max_march = tm.tris.len().saturating_mul(2).clamp(64, 8192);

    for _ in 0..max_march {
        if cur == v_end {
            return Ok(());
        }
        if tm.has_edge(cur, v_end) {
            seam_set.insert(edge_key(cur, v_end));
            return Ok(());
        }

        let p_cur = tm.pos(cur);
        let p_end = tm.pos(v_end);
        let d = sub3(p_end, p_cur);
        let cur_dist_sq = dot(d, d);
        if cur_dist_sq <= 1e-24 {
            return Ok(());
        }

        let Some(step) = find_march_step(tm, cur, prev, p_cur, p_end, d, cur_dist_sq) else {
            break;
        };
        let next_v = match step {
            MarchStep::SplitEdge(ea, eb, t) => tm.split_edge(ea, eb, t, snap)?,
            MarchStep::ExistingVertex(v) => v,
        };
        if next_v == cur || visited.contains(&next_v) {
            break;
        }
        if !tm.has_edge(cur, next_v) {
            break;
        }
        seam_set.insert(edge_key(cur, next_v));
        visited.insert(next_v);
        prev = Some(cur);
        cur = next_v;
    }

    if cur != v_end {
        dijkstra_seam_fallback(tm, cur, v_end, seam_set)?;
    }
    Ok(())
}

enum MarchStep {
    SplitEdge(VertexId, VertexId, f64),
    ExistingVertex(VertexId),
}

fn find_march_step(
    tm: &TrimMesh<'_, '_>,
    cur: VertexId,
    prev: Option<VertexId>,
    p_cur: [f64; 3],
    p_end: [f64; 3],
    d: [f64; 3],
    cur_dist_sq: f64,
) -> Option<MarchStep> {
    let fan = tm.vert_tris.get(&cur)?;
    let d_len = cur_dist_sq.sqrt();
    let mut best_score = f64::NEG_INFINITY;
    let mut best_step = None;

    for &lt in fan {
        let Some((verts, _)) = tm.tris[lt] else {
            continue;
        };
        let (b, c) = if verts[0] == cur {
            (verts[1], verts[2])
        } else if verts[1] == cur {
            (verts[2], verts[0])
        } else {
            (verts[0], verts[1])
        };
        if prev == Some(b) || prev == Some(c) {
            continue;
        }
        let pb = tm.pos(b);
        let pc = tm.pos(c);
        let e1 = sub3(pb, p_cur);
        let e2 = sub3(pc, p_cur);
        let n = cross3(e1, e2);
        let nn = dot(n, n);
        if nn <= 1e-24 {
            continue;
        }
        let dn = dot(d, n) / nn;
        let d_proj = [d[0] - dn * n[0], d[1] - dn * n[1], d[2] - dn * n[2]];
        let g00 = dot(e1, e1);
        let g01 = dot(e1, e2);
        let g11 = dot(e2, e2);
        let b0 = dot(d_proj, e1);
        let b1 = dot(d_proj, e2);
        let u = (b0 * g11 - b1 * g01) / nn;
        let v = (g00 * b1 - g01 * b0) / nn;
        let uv = u + v;
        if u < -1e-4 || v < -1e-4 || uv <= 1e-12 {
            continue;
        }
        let t = (v / uv).clamp(0.0, 1.0);
        let p_split = [
            pb[0] + (pc[0] - pb[0]) * t,
            pb[1] + (pc[1] - pb[1]) * t,
            pb[2] + (pc[2] - pb[2]) * t,
        ];
        if dist_sq(p_split, p_end) >= cur_dist_sq {
            continue;
        }
        let step_dir = sub3(p_split, p_cur);
        let step_len = dot(step_dir, step_dir).sqrt();
        if step_len <= 1e-12 {
            continue;
        }
        let align = dot(step_dir, d) / (step_len * d_len);
        if align > best_score {
            best_score = align;
            best_step = Some(if t <= 1e-5 {
                MarchStep::ExistingVertex(b)
            } else if t >= 1.0 - 1e-5 {
                MarchStep::ExistingVertex(c)
            } else {
                MarchStep::SplitEdge(b, c, t)
            });
        }
    }
    best_step
}

#[derive(Copy, Clone, PartialEq)]
struct HeapNode {
    cost: f64,
    v: VertexId,
}

impl Eq for HeapNode {}

impl Ord for HeapNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.v.cmp(&other.v))
    }
}

impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn dijkstra_seam_fallback(
    tm: &TrimMesh<'_, '_>,
    src: VertexId,
    dst: VertexId,
    seam_set: &mut HashSet<(VertexId, VertexId)>,
) -> std::result::Result<(), SceneError> {
    if src == dst {
        return Ok(());
    }
    let mut dist: HashMap<VertexId, f64> = HashMap::new();
    let mut prev: HashMap<VertexId, VertexId> = HashMap::new();
    let mut heap = BinaryHeap::new();
    dist.insert(src, 0.0);
    heap.push(HeapNode { cost: 0.0, v: src });

    while let Some(HeapNode { cost, v }) = heap.pop() {
        if v == dst {
            break;
        }
        if dist.get(&v).is_some_and(|&best| cost > best) {
            continue;
        }
        let pv = tm.pos(v);
        let Some(fan) = tm.vert_tris.get(&v) else {
            continue;
        };
        for &lt in fan {
            let Some((verts, _)) = tm.tris[lt] else {
                continue;
            };
            for &nb in &verts {
                if nb == v {
                    continue;
                }
                let next_cost = cost + dist_sq(pv, tm.pos(nb)).sqrt();
                if dist.get(&nb).is_none_or(|&best| next_cost < best) {
                    dist.insert(nb, next_cost);
                    prev.insert(nb, v);
                    heap.push(HeapNode {
                        cost: next_cost,
                        v: nb,
                    });
                }
            }
        }
    }

    let mut cur = dst;
    while let Some(&p) = prev.get(&cur) {
        seam_set.insert(edge_key(p, cur));
        if p == src {
            return Ok(());
        }
        cur = p;
    }
    Ok(())
}

struct FillComponent {
    tris: Vec<usize>,
    touches_boundary: bool,
    area: f64,
}

fn remove_closed_region(
    tm: &mut TrimMesh<'_, '_>,
    seam_set: &HashSet<(VertexId, VertexId)>,
    delete_inside: bool,
) -> std::result::Result<(), SceneError> {
    let mut seen = vec![false; tm.tris.len()];
    let mut components: Vec<FillComponent> = Vec::new();
    let mut queue = VecDeque::new();

    for start in 0..tm.tris.len() {
        if seen[start] || tm.tris[start].is_none() {
            continue;
        }
        seen[start] = true;
        queue.push_back(start);
        let mut comp_tris = Vec::new();
        let mut touches_boundary = false;
        let mut area = 0.0;

        while let Some(lt) = queue.pop_front() {
            let Some((verts, _)) = tm.tris[lt] else {
                continue;
            };
            comp_tris.push(lt);
            let p0 = tm.pos(verts[0]);
            let p1 = tm.pos(verts[1]);
            let p2 = tm.pos(verts[2]);
            let n = cross3(sub3(p1, p0), sub3(p2, p0));
            area += 0.5 * dot(n, n).sqrt();

            for k in 0..3 {
                let key = edge_key(verts[k], verts[(k + 1) % 3]);
                if seam_set.contains(&key) {
                    continue;
                }
                let Some(adj) = tm.edge_tris.get(&key) else {
                    continue;
                };
                if adj.len() < 2 {
                    touches_boundary = true;
                }
                for &nb in adj {
                    if !seen[nb] && tm.tris[nb].is_some() {
                        seen[nb] = true;
                        queue.push_back(nb);
                    }
                }
            }
        }

        components.push(FillComponent {
            tris: comp_tris,
            touches_boundary,
            area,
        });
    }

    if components.len() < 2 {
        return Ok(());
    }

    let has_interior = components.iter().any(|c| !c.touches_boundary);
    let has_boundary = components.iter().any(|c| c.touches_boundary);
    let remove_flags: Vec<bool> = if has_interior && has_boundary {
        components
            .iter()
            .map(|c| {
                if delete_inside {
                    !c.touches_boundary
                } else {
                    c.touches_boundary
                }
            })
            .collect()
    } else {
        let smallest = components
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.area.total_cmp(&b.area))
            .map(|(i, _)| i)
            .unwrap_or(0);
        (0..components.len())
            .map(|i| {
                if delete_inside {
                    i == smallest
                } else {
                    i != smallest
                }
            })
            .collect()
    };

    let mut to_remove = Vec::new();
    for (i, comp) in components.into_iter().enumerate() {
        if remove_flags[i] {
            for lt in comp.tris {
                if let Some((_, store_id)) = tm.detach_logical(lt) {
                    to_remove.push(store_id);
                }
            }
        }
    }
    if !to_remove.is_empty() {
        tm.tx.remove_triangles(&to_remove)?;
    }
    Ok(())
}

fn prune_orphaned_vertices(tm: &mut TrimMesh<'_, '_>) -> std::result::Result<(), SceneError> {
    let slots = tm.tx.mesh().vertex_slots();
    let mut referenced = vec![false; slots];
    for t in 0..tm.tx.mesh().triangle_slots() as u32 {
        if let Some([a, b, c]) = tm.tx.mesh().triangle(t) {
            referenced[a as usize] = true;
            referenced[b as usize] = true;
            referenced[c as usize] = true;
        }
    }
    let mut orphan_verts = Vec::new();
    for v in 0..slots as u32 {
        if tm.tx.mesh().is_vertex_alive(v)
            && !referenced[v as usize]
            && tm.vert_tris.contains_key(&v)
        {
            orphan_verts.push(v);
        }
    }
    if !orphan_verts.is_empty() {
        tm.tx.remove_vertices(&orphan_verts)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::Mesh;

    use crate::ops::OpsError;
    use crate::{
        CutOptions, MeshOptions, Placement, Plane, Scene, SceneError, SceneOptions,
        TrimCurveOptions,
    };

    /// A closed unit cube, outward wound.
    fn cube() -> Mesh {
        Mesh::from_buffers(
            &[
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                1.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, //
                1.0, 0.0, 1.0, //
                1.0, 1.0, 1.0, //
                0.0, 1.0, 1.0,
            ],
            &[
                0, 2, 1, 0, 3, 2, //
                4, 5, 6, 4, 6, 7, //
                0, 1, 5, 0, 5, 4, //
                2, 3, 7, 2, 7, 6, //
                1, 2, 6, 1, 6, 5, //
                3, 0, 4, 3, 4, 7, //
            ],
        )
        .unwrap()
    }

    fn plane(origin: [f64; 3], normal: [f64; 3]) -> Plane {
        Plane::new(origin, normal).unwrap()
    }

    #[test]
    fn cut_keeps_placement_and_undoes_in_one_step() {
        let mut scene = Scene::new(SceneOptions::default());
        // A unit cube moved to [10, 0, 0]: the world plane x = 10.5 cuts its
        // middle, whatever the local coordinates say.
        let placement = Placement {
            translation: [10.0, 0.0, 0.0],
            ..Placement::IDENTITY
        };
        let id = scene
            .add_mesh(
                cube(),
                MeshOptions {
                    placement,
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        scene.take_changes();
        let before = scene.mesh(id).unwrap().triangle_count();

        let report = scene
            .cut_by_plane(
                id,
                &CutOptions {
                    plane: plane([10.5, 0.0, 0.0], [1.0, 0.0, 0.0]),
                    cap: true,
                },
            )
            .unwrap();
        // Both parts closed, both placed where the cube was.
        assert!(
            crate::ops::boundary_loops(scene.mesh(id).unwrap())
                .unwrap()
                .is_empty()
        );
        assert!(
            crate::ops::boundary_loops(scene.mesh(report.added).unwrap())
                .unwrap()
                .is_empty()
        );
        assert_eq!(scene.placement(report.added), Ok(placement));
        assert_eq!(scene.mode(report.added), Ok(scene.mode(id).unwrap()));
        assert_ne!(report.added, id);
        // The kept side is the + one: its world box starts at the plane.
        let kept_world: Vec<[f64; 3]> = (0..scene.mesh(id).unwrap().vertex_slots() as u32)
            .filter(|&v| scene.mesh(id).unwrap().is_vertex_alive(v))
            .map(|v| {
                placement.apply([
                    scene.mesh(id).unwrap().origin()[0]
                        + f64::from(scene.mesh(id).unwrap().position(v).unwrap()[0]),
                    scene.mesh(id).unwrap().origin()[1]
                        + f64::from(scene.mesh(id).unwrap().position(v).unwrap()[1]),
                    scene.mesh(id).unwrap().origin()[2]
                        + f64::from(scene.mesh(id).unwrap().position(v).unwrap()[2]),
                ])
            })
            .collect();
        assert!(kept_world.iter().all(|p| p[0] >= 10.5 - 1e-6));
        assert_eq!(
            report.kept_triangles,
            scene.mesh(id).unwrap().triangle_count()
        );
        assert_eq!(
            report.removed_triangles,
            scene.mesh(report.added).unwrap().triangle_count()
        );

        // One undo step restores the cut mesh and removes the added part
        // (#269); redo brings it back under the same id.
        assert!(scene.can_undo());
        assert!(scene.undo().unwrap());
        assert_eq!(scene.mesh(id).unwrap().triangle_count(), before);
        assert!(scene.mesh(report.added).is_err());
        assert!(!scene.can_undo());
        assert!(scene.redo().unwrap());
        assert_ne!(scene.mesh(id).unwrap().triangle_count(), before);
        assert_eq!(
            report.removed_triangles,
            scene.mesh(report.added).unwrap().triangle_count()
        );
    }

    #[test]
    fn scaled_cut_follows_the_scale() {
        // The unit cube scaled by [2, 1, 1] and moved to [0, 3, 0]: the
        // world shape is x in [0, 2]. The world plane x = 1 cuts its middle.
        let mut scene = Scene::new(SceneOptions::default());
        let placement = Placement {
            translation: [0.0, 3.0, 0.0],
            scale: [2.0, 1.0, 1.0],
            ..Placement::IDENTITY
        };
        let id = scene
            .add_mesh(
                cube(),
                MeshOptions {
                    placement,
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        scene.take_changes();
        let report = scene
            .cut_by_plane(
                id,
                &CutOptions {
                    plane: plane([1.0, 3.0, 0.0], [1.0, 0.0, 0.0]),
                    cap: true,
                },
            )
            .unwrap();
        assert!(
            crate::ops::boundary_loops(scene.mesh(id).unwrap())
                .unwrap()
                .is_empty()
        );
        assert!(
            crate::ops::boundary_loops(scene.mesh(report.added).unwrap())
                .unwrap()
                .is_empty()
        );
        assert_eq!(scene.placement(report.added), Ok(placement));
        // The kept world box is the + half: x in [1, 2], y in [3, 4].
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mesh = scene.mesh(id).unwrap();
        for v in 0..mesh.vertex_slots() as u32 {
            let Some(p) = mesh.position(v) else {
                continue;
            };
            let w = placement.apply([
                mesh.origin()[0] + f64::from(p[0]),
                mesh.origin()[1] + f64::from(p[1]),
                mesh.origin()[2] + f64::from(p[2]),
            ]);
            for k in 0..3 {
                lo[k] = lo[k].min(w[k]);
                hi[k] = hi[k].max(w[k]);
            }
        }
        assert!((lo[0] - 1.0).abs() < 1e-4, "{lo:?}");
        assert!((hi[0] - 2.0).abs() < 1e-4, "{hi:?}");
        assert!((lo[1] - 3.0).abs() < 1e-4, "{lo:?}");
        assert!((hi[1] - 4.0).abs() < 1e-4, "{hi:?}");
    }

    #[test]
    fn missing_and_uncapped_cuts() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(cube(), MeshOptions::default()).unwrap();
        scene.take_changes();
        // A plane past the cube misses it: nothing changes.
        assert_eq!(
            scene.cut_by_plane(
                id,
                &CutOptions {
                    plane: plane([5.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                    cap: true,
                },
            ),
            Err(SceneError::Ops(OpsError::InvalidInput(
                "plane does not cut the mesh",
            )))
        );
        assert_eq!(scene.mesh(id).unwrap().triangle_count(), 12);
        assert!(!scene.can_undo());
        // Without cap both parts stay open.
        let report = scene
            .cut_by_plane(
                id,
                &CutOptions {
                    plane: plane([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]),
                    cap: false,
                },
            )
            .unwrap();
        assert_eq!(report.cap_triangles, 0);
        assert!(
            !crate::ops::boundary_loops(scene.mesh(id).unwrap())
                .unwrap()
                .is_empty()
        );
        assert!(
            !crate::ops::boundary_loops(scene.mesh(report.added).unwrap())
                .unwrap()
                .is_empty()
        );
    }

    fn grid_sheet(n: usize, size: f32) -> Mesh {
        let mut pos = Vec::with_capacity((n + 1) * (n + 1) * 3);
        for j in 0..=n {
            for i in 0..=n {
                pos.push((i as f32 / n as f32) * size);
                pos.push((j as f32 / n as f32) * size);
                pos.push(0.0);
            }
        }
        let mut indices = Vec::with_capacity(n * n * 6);
        for j in 0..n {
            for i in 0..n {
                let v00 = (j * (n + 1) + i) as u32;
                let v10 = (j * (n + 1) + i + 1) as u32;
                let v01 = ((j + 1) * (n + 1) + i) as u32;
                let v11 = ((j + 1) * (n + 1) + i + 1) as u32;
                indices.extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
            }
        }
        Mesh::from_buffers(&pos, &indices).unwrap()
    }

    #[test]
    fn trim_by_curve_deletes_interior_on_open_sheet_and_supports_undo() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene
            .add_mesh(grid_sheet(4, 4.0), MeshOptions::default())
            .unwrap();
        let initial_tris = scene.mesh(id).unwrap().triangle_count();

        let report = scene
            .trim_by_curve(
                id,
                &TrimCurveOptions {
                    points: vec![
                        [1.2, 1.2, 0.0],
                        [2.8, 1.2, 0.0],
                        [2.8, 2.8, 0.0],
                        [1.2, 2.8, 0.0],
                    ],
                    closed: true,
                    delete_inside: true,
                    snap_tolerance: 0.0,
                },
            )
            .unwrap();
        assert!(report.seam_edges >= 4, "{report:?}");
        assert!(report.vertices_added > 0, "{report:?}");
        assert!(report.triangles_removed > 0, "{report:?}");
        let trimmed_tris = scene.mesh(id).unwrap().triangle_count();
        assert_eq!(
            initial_tris + report.triangles_added - report.triangles_removed,
            trimmed_tris
        );
        // Should now have at least 2 boundary loops (outer border + punched hole).
        let loops = crate::ops::boundary_loops(scene.mesh(id).unwrap()).unwrap();
        assert!(loops.len() >= 2, "{loops:?}");

        // Atomic undo & redo.
        assert!(scene.can_undo());
        scene.undo().unwrap();
        assert_eq!(scene.mesh(id).unwrap().triangle_count(), initial_tris);
        scene.redo().unwrap();
        assert_eq!(scene.mesh(id).unwrap().triangle_count(), trimmed_tris);
    }

    #[test]
    fn trim_by_curve_open_curve_imprints_without_deleting_and_rejects_bad_input() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene
            .add_mesh(grid_sheet(2, 2.0), MeshOptions::default())
            .unwrap();
        let before = scene.mesh(id).unwrap().triangle_count();
        let rep = scene
            .trim_by_curve(
                id,
                &TrimCurveOptions {
                    points: vec![[0.3, 0.3, 0.0], [1.7, 1.7, 0.0]],
                    closed: false,
                    delete_inside: true,
                    snap_tolerance: 0.0,
                },
            )
            .unwrap();
        assert!(rep.triangles_added > rep.triangles_removed);
        assert!(scene.mesh(id).unwrap().triangle_count() >= before);

        assert!(matches!(
            scene.trim_by_curve(
                id,
                &TrimCurveOptions {
                    points: vec![[0.0, 0.0, 0.0]],
                    closed: false,
                    delete_inside: true,
                    snap_tolerance: 0.0,
                },
            ),
            Err(SceneError::InvalidInput(_))
        ));
    }
}
