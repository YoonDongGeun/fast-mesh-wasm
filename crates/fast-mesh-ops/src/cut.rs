//! Cutting a mesh with a plane into two capped parts.
//!
//! [`cut_by_plane`] splits the mesh of a transaction in two along `plane`:
//! triangles on the `+` side (`signed_distance >= 0`, the same rule as
//! [`slice`](fast_mesh_query::slice)) stay in the mesh, triangles on the `-`
//! side move into a removed mesh with the same origin, and triangles crossing
//! the plane are split along it. With [`CutOptions::cap`] (the default), both
//! openings are closed with flat, unfaired [`fill_holes`] patches, so cutting
//! a closed mesh yields two closed meshes.
//!
//! Cap choice: the caps are flat triangulations of the cut rim, refined to
//! the surrounding density but not faired (`fair: false`). They follow the
//! cut plane exactly instead of continuing the surrounding curvature.
//!
//! Only positions and triangles cross over: the removed mesh takes no
//! channels, and new vertices (on both sides) take channel defaults, except
//! that crossing vertices added to the kept mesh interpolate their parents.
//!
//! [`fill_holes`]: crate::fill_holes

use std::collections::{BTreeMap, BTreeSet};

use fast_mesh_query::Plane;
use fast_mesh_store::{Mesh, StoreError, Transaction, TriangleId, VertexId};

use crate::fill::FillOptionsBuilder;
use crate::fill_holes;
use crate::topology::{BoundaryLoop, LoopId, boundary_loops};
use crate::{OpsError, Result, filled, push, reserve};

/// Options for [`cut_by_plane`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutOptions {
    /// Close both openings with flat, unfaired patches. Without it both
    /// parts keep the open rim the cut leaves.
    pub cap: bool,
}

impl Default for CutOptions {
    fn default() -> Self {
        CutOptions { cap: true }
    }
}

/// What [`cut_by_plane`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CutReport {
    /// Alive triangles of the kept (`+`) mesh, cap included.
    pub kept_triangles: usize,
    /// Alive triangles of the removed (`-`) mesh, cap included.
    pub removed_triangles: usize,
    /// Cap triangles added on both sides together.
    pub cap_triangles: usize,
}

/// The `-` side of [`cut_by_plane`]: a mesh with the input's origin, ready
/// to add to the scene with the same placement.
#[derive(Debug)]
pub struct CutOutput {
    /// The removed mesh, capped when [`CutOptions::cap`] is set.
    pub removed: Mesh,
    /// What the cut did.
    pub report: CutReport,
}

fn oom() -> OpsError {
    OpsError::Store(StoreError::OutOfMemory)
}

/// Splits the mesh of `tx` with `plane`: the `+` side stays, the `-` side is
/// returned as a mesh with the same origin. Triangles crossing the plane are
/// split along it; with `options.cap` both openings are closed with flat,
/// unfaired [`fill_holes`](crate::fill_holes) patches over the cut rim.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when the plane does not cut the mesh (every
///   triangle on one side), when a cut point leaves `f32` range, or, with
///   `cap`, when a rim loop cannot be triangulated.
/// - [`OpsError::StaleLoops`] when a rim loop vanished between listing and
///   filling (defensive: both come from the same mesh state).
/// - [`OpsError::Store`] when memory runs out. Nothing is written then: the
///   plan is read from the untouched mesh before the first write.
pub fn cut_by_plane(
    tx: &mut Transaction,
    plane: &Plane,
    options: &CutOptions,
) -> Result<CutOutput> {
    let mesh = tx.mesh();
    let origin = mesh.origin();
    let slots = mesh.vertex_slots();
    let tri_slots = mesh.triangle_slots();

    // Sides of every triangle slot, in slot order, and the crossing points of
    // every crossed edge: read-only over the untouched mesh.
    let mut points: Vec<[f64; 3]> = Vec::new();
    reserve(&mut points, slots)?;
    for v in 0..slots as VertexId {
        points.push(mesh.position(v).map_or([f64::NAN; 3], |p| p.map(f64::from)));
    }
    let mut side: Vec<[bool; 3]> = Vec::new();
    reserve(&mut side, tri_slots)?;
    let mut straddling: Vec<TriangleId> = Vec::new();
    for t in 0..tri_slots as TriangleId {
        let Some(tri) = mesh.triangle(t) else {
            side.push([true; 3]);
            continue;
        };
        let s = tri.map(|v| plane.signed_distance(points[v as usize]) >= 0.0);
        if s[0] != s[1] || s[1] != s[2] {
            push(&mut straddling, t)?;
        }
        push(&mut side, s)?;
    }
    if straddling.is_empty() {
        return Err(OpsError::InvalidInput("plane does not cut the mesh"));
    }
    // One crossing per crossed edge, in id order so that both sides agree.
    // An endpoint exactly on the plane is the crossing: no new vertex.
    let mut at: BTreeMap<(VertexId, VertexId), VertexId> = BTreeMap::new();
    let mut point_of: BTreeMap<(VertexId, VertexId), [f64; 3]> = BTreeMap::new();
    for &t in &straddling {
        // `side` holds one entry per triangle slot, in order.
        let s = side[t as usize];
        let tri = mesh
            .triangle(t)
            .ok_or(OpsError::InvalidInput("plane does not cut the mesh"))?;
        let (e1, e2) = crossed_edges(tri, lone(s));
        for e in [e1, e2] {
            if at.contains_key(&e) {
                continue;
            }
            let (a, b) = e;
            let (pa, pb) = (points[a as usize], points[b as usize]);
            let (sa, sb) = (plane.signed_distance(pa), plane.signed_distance(pb));
            debug_assert!((sa >= 0.0) != (sb >= 0.0), "edge {e:?} is not crossed");
            if sa == 0.0 {
                at.insert(e, a);
            } else if sb == 0.0 {
                at.insert(e, b);
            } else {
                let f = sa / (sa - sb);
                point_of.insert(e, std::array::from_fn(|k| pa[k] + f * (pb[k] - pa[k])));
            }
        }
    }

    // The write phase: add the new crossing vertices first, interpolating
    // their parents so channels blend across the cut.
    let mut added_points: BTreeMap<VertexId, [f64; 3]> = BTreeMap::new();
    for (&e, &p) in &point_of {
        let (a, b) = e;
        let q = p.map(|c| c as f32);
        if !q.iter().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput("a cut point leaves f32 range"));
        }
        let (sa, sb) = (
            plane.signed_distance(points[a as usize]),
            plane.signed_distance(points[b as usize]),
        );
        let f = (sa / (sa - sb)) as f32;
        let id = tx.add_vertex_with_parents(q, &[(a, 1.0 - f), (b, f)])?;
        at.insert(e, id);
        added_points.insert(id, p);
    }

    // Split every straddling triangle: the first kept polygon triangle
    // replaces it, the rest are added; the removed side collects into fresh
    // buffers in its own indexing.
    let mut rem_index: BTreeMap<VertexId, u32> = BTreeMap::new();
    let mut rem_positions: Vec<[f64; 3]> = Vec::new();
    let mut rem_tris: Vec<[u32; 3]> = Vec::new();
    let mut kept_cut: BTreeSet<(VertexId, VertexId)> = BTreeSet::new();
    let mut rem_cut: BTreeSet<(u32, u32)> = BTreeSet::new();
    let mut removed_tris: Vec<TriangleId> = Vec::new();
    for &t in &straddling {
        let tri = tx
            .mesh()
            .triangle(t)
            .ok_or(OpsError::InvalidInput("plane does not cut the mesh"))?;
        let s = side[t as usize];
        let i = lone(s);
        let (e1, e2) = crossed_edges(tri, i);
        let (x1, x2) = (at[&e1], at[&e2]);
        if x1 != x2 {
            kept_cut.insert(key(x1, x2));
        }
        let (kept_poly, removed_poly) = if s[i] {
            (
                vec![tri[i], x1, x2],
                vec![x1, tri[(i + 1) % 3], tri[(i + 2) % 3], x2],
            )
        } else {
            (
                vec![x1, tri[(i + 1) % 3], tri[(i + 2) % 3], x2],
                vec![tri[i], x1, x2],
            )
        };
        write_poly(tx, t, &kept_poly)?;
        // The removed-side index of `v`: original vertices reuse their
        // stored position, new crossing vertices their planned point.
        fn rem_id(
            v: VertexId,
            points: &[[f64; 3]],
            added_points: &BTreeMap<VertexId, [f64; 3]>,
            rem_index: &mut BTreeMap<VertexId, u32>,
            rem_positions: &mut Vec<[f64; 3]>,
        ) -> Result<u32> {
            if let Some(&idx) = rem_index.get(&v) {
                return Ok(idx);
            }
            let p = match added_points.get(&v) {
                Some(&p) => p,
                None => points[v as usize],
            };
            if !p.iter().all(|c| c.is_finite()) {
                return Err(OpsError::InvalidInput("a removed vertex is not finite"));
            }
            let idx = u32::try_from(rem_positions.len()).map_err(|_| oom())?;
            reserve(rem_positions, 1)?;
            rem_positions.push(p);
            rem_index.insert(v, idx);
            Ok(idx)
        }
        let mut quad: Vec<u32> = Vec::with_capacity(removed_poly.len());
        for v in removed_poly {
            quad.push(rem_id(
                v,
                &points,
                &added_points,
                &mut rem_index,
                &mut rem_positions,
            )?);
        }
        let before = rem_tris.len();
        emit_fan(&mut rem_tris, quad)?;
        if rem_tris.len() > before && x1 != x2 {
            match (rem_index.get(&x1), rem_index.get(&x2)) {
                (Some(&r1), Some(&r2)) if r1 != r2 => {
                    rem_cut.insert((r1.min(r2), r1.max(r2)));
                }
                _ => {}
            }
        }
    }
    // Triangles fully on the `-` side leave the kept mesh and join the
    // removed buffers.
    for t in 0..tri_slots as TriangleId {
        if side[t as usize] != [false; 3] {
            continue;
        }
        let tri = tx
            .mesh()
            .triangle(t)
            .ok_or(OpsError::InvalidInput("plane does not cut the mesh"))?;
        let mut idx = [0; 3];
        for (k, v) in tri.iter().enumerate() {
            // Original vertices reuse their stored position.
            if !rem_index.contains_key(v) {
                let p = points[*v as usize];
                let n = u32::try_from(rem_positions.len()).map_err(|_| oom())?;
                reserve(&mut rem_positions, 1)?;
                rem_positions.push(p);
                rem_index.insert(*v, n);
            }
            idx[k] = rem_index[v];
        }
        push(&mut rem_tris, idx)?;
        push(&mut removed_tris, t)?;
    }
    if !removed_tris.is_empty() {
        tx.remove_triangles(&removed_tris)?;
    }
    // Vertices the cut orphaned (used only by the removed side) go.
    let mut orphans: Vec<VertexId> = Vec::new();
    for v in 0..tx.mesh().vertex_slots() as VertexId {
        if tx.mesh().is_vertex_alive(v) && tx.mesh().corner_count(v) == 0 {
            push(&mut orphans, v)?;
        }
    }
    if !orphans.is_empty() {
        tx.remove_vertices(&orphans)?;
    }

    // Close both openings with flat, unfaired patches over the cut rim.
    let mut cap_triangles = 0;
    if options.cap {
        cap_triangles += cap_rim(tx, &kept_cut)?;
    }
    let kept_triangles = tx.mesh().triangle_count();

    // The removed side, with the input's origin so the same placement fits.
    let mut flat_positions: Vec<f64> = Vec::new();
    reserve(&mut flat_positions, rem_positions.len() * 3)?;
    for p in &rem_positions {
        flat_positions.extend_from_slice(&[origin[0] + p[0], origin[1] + p[1], origin[2] + p[2]]);
    }
    let mut flat_tris: Vec<u32> = Vec::new();
    reserve(&mut flat_tris, rem_tris.len() * 3)?;
    for t in &rem_tris {
        flat_tris.extend_from_slice(t);
    }
    let mut removed = Mesh::from_buffers_f64_with_origin(&flat_positions, &flat_tris, origin)?;
    // The removed side keeps its UVs, crossing vertices included: their slots
    // still hold the interpolated values.
    if let Some(uvs) = tx.mesh().uvs() {
        let mut rem_uvs = filled(2 * rem_positions.len(), 0.0f32)?;
        for (&v, &i) in &rem_index {
            rem_uvs[2 * i as usize] = uvs[2 * v as usize];
            rem_uvs[2 * i as usize + 1] = uvs[2 * v as usize + 1];
        }
        removed.add_uvs(&rem_uvs)?;
    }
    if options.cap {
        let loops = boundary_loops(&removed)?;
        let mut rim: Vec<LoopId> = Vec::new();
        for l in &loops {
            if is_cut_rim(l, &rem_cut) {
                push(&mut rim, l.id)?;
            }
        }
        if !rim.is_empty() {
            let mut rtx = removed.begin_edit();
            let before = rtx.mesh().triangle_count();
            let report = fill_holes(
                &mut rtx,
                &FillOptionsBuilder::default()
                    .loops(rim)
                    .fair(false)
                    .allow_unclean(true)
                    .build(),
            )?;
            if !report.skipped.is_empty() || !report.failed.is_empty() {
                return Err(OpsError::InvalidInput("a cut rim cannot be capped"));
            }
            rtx.commit()?;
            cap_triangles += removed.triangle_count() - before;
        }
    }
    let removed_triangles = removed.triangle_count();

    Ok(CutOutput {
        removed,
        report: CutReport {
            kept_triangles,
            removed_triangles,
            cap_triangles,
        },
    })
}

/// The index of the lone vertex, whose side differs from the other two.
fn lone(s: [bool; 3]) -> usize {
    if s[0] == s[1] {
        2
    } else if s[0] == s[2] {
        1
    } else {
        0
    }
}

fn key(a: VertexId, b: VertexId) -> (VertexId, VertexId) {
    (a.min(b), a.max(b))
}

/// The crossed edges of a straddling triangle with lone vertex `i`: the
/// edge out of it and the edge into it, in id order.
fn crossed_edges(tri: [VertexId; 3], i: usize) -> ((VertexId, VertexId), (VertexId, VertexId)) {
    let (a, b, c) = (tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]);
    (key(a, b), key(c, a))
}

/// Writes a kept-side polygon: dedupe drops the vertices the plane passes
/// through twice, and a fan covers the quad.
fn write_poly(tx: &mut Transaction, t: TriangleId, poly: &[VertexId]) -> Result<()> {
    let mut dedup: Vec<VertexId> = Vec::with_capacity(poly.len());
    for &v in poly {
        if dedup.last() != Some(&v) {
            dedup.push(v);
        }
    }
    while dedup.len() > 1 && dedup[0] == *dedup.last().unwrap() {
        dedup.pop();
    }
    match dedup.len() {
        0..=2 => tx.remove_triangles(&[t]).map_err(OpsError::from),
        3 => tx
            .set_triangles(&[t], &[dedup.as_slice().try_into().unwrap()])
            .map_err(OpsError::from),
        _ => {
            tx.set_triangles(&[t], &[[dedup[0], dedup[1], dedup[2]]])?;
            tx.add_triangles(&[[dedup[0], dedup[2], dedup[3]]])?;
            Ok(())
        }
    }
}

/// Emits a removed-side polygon as a fan, dropping degenerate vertices.
fn emit_fan(tris: &mut Vec<[u32; 3]>, poly: Vec<u32>) -> Result<()> {
    let mut dedup: Vec<u32> = Vec::with_capacity(poly.len());
    for v in poly {
        if dedup.last() != Some(&v) {
            dedup.push(v);
        }
    }
    while dedup.len() > 1 && dedup[0] == *dedup.last().unwrap() {
        dedup.pop();
    }
    if dedup.len() == 4 {
        reserve(tris, 2)?;
        tris.push([dedup[0], dedup[1], dedup[2]]);
        tris.push([dedup[0], dedup[2], dedup[3]]);
    } else if dedup.len() == 3 {
        push(tris, [dedup[0], dedup[1], dedup[2]])?;
    }
    Ok(())
}

/// Whether a boundary loop is the cut rim: every edge is a cut edge.
fn is_cut_rim(l: &BoundaryLoop, cut: &BTreeSet<(u32, u32)>) -> bool {
    l.closed && loop_edges(&l.vertices).iter().all(|e| cut.contains(e))
}

/// The edges of a loop as unordered pairs, closing back to the first vertex.
fn loop_edges(v: &[VertexId]) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(v.len());
    for (k, &a) in v.iter().enumerate() {
        let b = v[(k + 1) % v.len()];
        out.push((a.min(b), a.max(b)));
    }
    out
}

/// Fills the rim loops of the kept mesh (every edge a cut edge) with a
/// flat, unfaired patch. Returns the cap triangles added.
fn cap_rim(tx: &mut Transaction, cut: &BTreeSet<(VertexId, VertexId)>) -> Result<usize> {
    let loops = boundary_loops(tx.mesh())?;
    let mut rim: Vec<LoopId> = Vec::new();
    for l in &loops {
        if l.closed && loop_edges(&l.vertices).iter().all(|e| cut.contains(e)) {
            push(&mut rim, l.id)?;
        }
    }
    if rim.is_empty() {
        return Ok(0);
    }
    let before = tx.mesh().triangle_count();
    let report = fill_holes(
        tx,
        &FillOptionsBuilder::default()
            .loops(rim)
            .fair(false)
            .allow_unclean(true)
            .build(),
    )?;
    if !report.skipped.is_empty() || !report.failed.is_empty() {
        return Err(OpsError::InvalidInput("a cut rim cannot be capped"));
    }
    Ok(tx.mesh().triangle_count() - before)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{edit, grid_without};
    use fast_mesh_store::Mesh;

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
                0, 2, 1, 0, 3, 2, // bottom (-z)
                4, 5, 6, 4, 6, 7, // top (+z)
                0, 1, 5, 0, 5, 4, // front (-y)
                2, 3, 7, 2, 7, 6, // back (+y)
                1, 2, 6, 1, 6, 5, // right (+x)
                3, 0, 4, 3, 4, 7, // left (-x)
            ],
        )
        .unwrap()
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

    fn plane(origin: [f64; 3], normal: [f64; 3]) -> Plane {
        Plane::new(origin, normal).unwrap()
    }

    fn cut_kept(mesh: &mut Mesh, plane: &Plane, cap: bool) -> CutOutput {
        let (out, _) = edit(mesh, |tx| cut_by_plane(tx, plane, &CutOptions { cap }));
        out
    }

    /// The local x of a kept-mesh position.
    fn kept_x(mesh: &Mesh, v: VertexId) -> f64 {
        mesh.position(v).unwrap()[0] as f64 + mesh.origin()[0]
    }

    #[test]
    fn cube_cut_in_half_is_two_closed_halves() {
        let mut mesh = cube();
        let origin = mesh.origin();
        let out = cut_kept(&mut mesh, &plane([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]), true);
        // Closed on both sides: the caps are there.
        assert!(boundary_loops(&mesh).unwrap().is_empty());
        assert!(boundary_loops(&out.removed).unwrap().is_empty());
        // The volumes add up to the cube's, both wound outward.
        let (vk, vr) = (volume6(&mesh), volume6(&out.removed));
        assert!(vk > 0.0 && vr > 0.0, "{vk} {vr}");
        assert!((vk + vr - 6.0).abs() < 1e-9, "{vk} {vr}");
        assert!((vk - vr).abs() < 1e-6, "{vk} {vr}");
        // Every vertex stays on its side.
        assert!((0..mesh.vertex_slots() as VertexId).all(|v| {
            mesh.position(v)
                .is_none_or(|_| kept_x(&mesh, v) >= 0.5 - 1e-6)
        }));
        assert!(
            out.removed
                .positions()
                .iter()
                .all(|p| p[0] as f64 + out.removed.origin()[0] <= 0.5 + 1e-6)
        );
        assert_eq!(out.report.kept_triangles, mesh.triangle_count());
        assert_eq!(out.report.removed_triangles, out.removed.triangle_count());
        assert!(out.report.cap_triangles >= 4, "{:?}", out.report);
        assert_eq!(out.removed.origin(), origin);
    }

    #[test]
    fn cap_off_leaves_both_sides_open() {
        let mut mesh = cube();
        let out = cut_kept(&mut mesh, &plane([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]), false);
        assert!(!boundary_loops(&mesh).unwrap().is_empty());
        assert!(!boundary_loops(&out.removed).unwrap().is_empty());
        assert_eq!(out.report.cap_triangles, 0);
        // The cut rim vertices lie exactly on the plane.
        assert!(
            mesh.positions()
                .iter()
                .any(|p| (kept_x_of(&mesh, p) - 0.5).abs() < 1e-6)
        );
    }

    fn kept_x_of(mesh: &Mesh, p: &[f32; 3]) -> f64 {
        p[0] as f64 + mesh.origin()[0]
    }

    #[test]
    fn plane_missing_the_mesh_is_rejected_and_changes_nothing() {
        let mut mesh = cube();
        let mut tx = mesh.begin_edit();
        let r = cut_by_plane(
            &mut tx,
            &plane([5.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            &CutOptions::default(),
        );
        assert_eq!(
            r.unwrap_err(),
            OpsError::InvalidInput("plane does not cut the mesh")
        );
        tx.rollback();
        assert_eq!(mesh.triangle_count(), 12);
    }

    #[test]
    fn open_grid_cut_keeps_origin_and_sides() {
        // An open 8x8 grid cut along its middle without cap.
        let mut mesh = grid_without(8, &[]);
        let origin = mesh.origin();
        let out = cut_kept(&mut mesh, &plane([3.5, 0.0, 0.0], [1.0, 0.0, 0.0]), false);
        assert_eq!(out.removed.origin(), origin);
        let alive = |m: &Mesh, v: VertexId| m.is_vertex_alive(v);
        assert!(
            (0..mesh.vertex_slots() as VertexId)
                .filter(|&v| alive(&mesh, v))
                .all(|v| kept_x(&mesh, v) >= 3.5 - 1e-6)
        );
        assert!(
            out.removed
                .positions()
                .iter()
                .all(|p| p[0] as f64 + out.removed.origin()[0] <= 3.5 + 1e-6)
        );
    }
}
