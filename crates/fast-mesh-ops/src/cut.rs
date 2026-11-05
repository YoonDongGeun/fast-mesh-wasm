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
//! Only positions, triangles and UVs cross over: the removed mesh takes no
//! other channels. Crossing vertices interpolate their parents.
//!
//! UV seams: on a mesh with UVs a rim is read by position, so a seam that
//! crosses the plane does not stop it from closing. The rim's vertices stay
//! the walls': where a seam splits a rim position in two the cap takes one of
//! the two, and the cap's edge next to it closes against the other side's wall
//! by position only, like any seam. So the cap's UVs are the wall's along the
//! rim, one side's at a seam, and the weighted mean of the rim inside: the
//! patch blends across the seam, and it is no island of its own. A rim that
//! closes by position but cannot be capped (its walls wound against each
//! other, or two rims or more than two sheets at one position) is an error.
//! A mesh without UVs has no seams: it is read by vertex index only, and split
//! vertices on it are a real boundary (weld it first).
//!
//! [`fill_holes`]: crate::fill_holes

use std::collections::{BTreeMap, BTreeSet};

use fast_mesh_query::Plane;
use fast_mesh_store::{Mesh, StoreError, Transaction, TriangleId, VertexId};

use crate::fill::{FillOptions, FillOptionsBuilder, Plan};
use crate::fill_dp::triangulate_loop;
use crate::fill_holes;
use crate::refine::refine;
use crate::topology::{BoundaryLoop, EdgeMap, LoopId, VertexFans, boundary_loops, point};
use crate::weld::weld_roots;
use crate::{OpsError, Result, filled, push, reserve};

/// Options for [`cut_by_plane`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutOptions {
    /// Close both openings with flat, unfaired patches, across UV seams on a
    /// mesh with UVs. Without it both parts keep the open rim the cut leaves.
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
/// A rim is capped when it closes: by vertex index, or, on a mesh with UVs,
/// by exact position across seams (see the module docs for the cap's UVs).
/// One that runs into a real boundary of an open mesh stays open. On a mesh
/// without UVs split vertices are such a boundary: weld them first.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when the plane does not cut the mesh (every
///   triangle on one side), when a cut point leaves `f32` range, or, with
///   `cap`, when a rim loop cannot be triangulated, or closes by position
///   with its walls wound against each other, or two rims or more than two
///   sheets at one position.
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
    // On a mesh with UVs, rims close by exact position.
    let by_position = mesh.uvs().is_some();
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
                point_of.insert(e, crossing_point(pa, pb, sa, sb, by_position));
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
        let rims = find_rims(tx.mesh(), &kept_cut, plane)?;
        cap_triangles += cap_rims(tx, &rims)?;
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
        let rims = find_rims(&removed, &rem_cut, plane)?;
        if !rims.whole.is_empty() || !rims.across_seams.is_empty() {
            let mut rtx = removed.begin_edit();
            cap_triangles += cap_rims(&mut rtx, &rims)?;
            rtx.commit()?;
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

/// Where the plane crosses the edge `a`-`b`, whose ends have signed distances
/// `sa` and `sb`. With `by_position`, the result does not depend on which end
/// comes first: the ends are taken in position order, so the two sides of a
/// seam, which hold the same edge under different ids, get the same point.
fn crossing_point(a: [f64; 3], b: [f64; 3], sa: f64, sb: f64, by_position: bool) -> [f64; 3] {
    let swap = by_position && b < a;
    let (a, b, sa, sb) = if swap { (b, a, sb, sa) } else { (a, b, sa, sb) };
    let f = sa / (sa - sb);
    std::array::from_fn(|k| a[k] + f * (b[k] - a[k]))
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

/// The openings a cut left, ready to cap.
struct Rims {
    /// Boundary loops of nothing but cut edges.
    whole: Vec<LoopId>,
    /// On a mesh with UVs, the other boundary edges that lie along the cut,
    /// each in its boundary loop's order.
    across_seams: Vec<(VertexId, VertexId)>,
}

/// The rims of `mesh` after a cut along `plane` with cut edges `cut`. A side
/// that keeps an edge in the plane marks it cut only if its neighbour across
/// the edge straddles the plane, and the neighbour past a seam is another edge.
/// So on a mesh with UVs a boundary edge with both ends exactly in the plane
/// counts too.
fn find_rims(mesh: &Mesh, cut: &BTreeSet<(VertexId, VertexId)>, plane: &Plane) -> Result<Rims> {
    let loops = boundary_loops(mesh)?;
    let seamed = mesh.uvs().is_some();
    let in_plane = |v: VertexId| plane.signed_distance(point(mesh, v)) == 0.0;
    let mut rims = Rims {
        whole: Vec::new(),
        across_seams: Vec::new(),
    };
    for l in &loops {
        if is_cut_rim(l, cut) {
            push(&mut rims.whole, l.id)?;
        } else if seamed {
            for e in loop_edges(&l.vertices) {
                if cut.contains(&e) || (in_plane(e.0) && in_plane(e.1)) {
                    push(&mut rims.across_seams, e)?;
                }
            }
        }
    }
    Ok(rims)
}

/// Caps `rims` with flat, unfaired patches. Returns the triangles added.
fn cap_rims(tx: &mut Transaction, rims: &Rims) -> Result<usize> {
    let before = tx.mesh().triangle_count();
    if !rims.whole.is_empty() {
        let report = fill_holes(
            tx,
            &FillOptionsBuilder::default()
                .loops(rims.whole.clone())
                .fair(false)
                .allow_unclean(true)
                .build(),
        )?;
        if !report.skipped.is_empty() || !report.failed.is_empty() {
            return Err(OpsError::InvalidInput("a cut rim cannot be capped"));
        }
    }
    if !rims.across_seams.is_empty() {
        cap_across_seams(tx, &rims.across_seams)?;
    }
    Ok(tx.mesh().triangle_count() - before)
}

/// A rim that closes by position: at each step, the position's weld root and
/// the vertex whose wall edge leaves it.
type SeamRim = Vec<(VertexId, VertexId)>;

/// The closed rims among boundary `edges` of `mesh`, reading vertices at one
/// exact position (`roots`) as one. An edge group with an odd position runs
/// into a real boundary and has no rim. One with only even positions that
/// cannot be walked as a single cycle in its triangles' direction is an error.
/// Two edges at one pair of positions, running opposite ways, already close
/// against each other, like a seam in the plane: they are no part of a rim.
fn seam_rims(
    mesh: &Mesh,
    edges: &[(VertexId, VertexId)],
    roots: &[VertexId],
) -> Result<Vec<SeamRim>> {
    let map = EdgeMap::build(mesh)?;
    // The directed edges between positions, each with its `from` vertices.
    let mut directed: BTreeMap<(VertexId, VertexId), Vec<VertexId>> = BTreeMap::new();
    for &(a, b) in edges {
        // Only a true boundary edge has one triangle to take a direction from.
        let [t] = map.triangles(a, b) else {
            continue;
        };
        let [x, y, z] = mesh.triangles()[*t as usize];
        let forward = [(x, y), (y, z), (z, x)].contains(&(a, b));
        let (from, to) = if forward { (a, b) } else { (b, a) };
        let (ra, rb) = (roots[from as usize], roots[to as usize]);
        if ra != rb {
            directed.entry((ra, rb)).or_default().push(from);
        }
    }
    // Per position: the edges leaving it, how many arrive, and its neighbours.
    let mut leaving: BTreeMap<VertexId, Vec<(VertexId, VertexId)>> = BTreeMap::new();
    let mut arriving: BTreeMap<VertexId, usize> = BTreeMap::new();
    let mut neighbours: BTreeMap<VertexId, Vec<VertexId>> = BTreeMap::new();
    for (&(ra, rb), froms) in &directed {
        let closed = directed.get(&(rb, ra)).map_or(0, Vec::len);
        for &from in froms.iter().skip(closed) {
            leaving.entry(ra).or_default().push((from, rb));
            *arriving.entry(rb).or_default() += 1;
            neighbours.entry(ra).or_default().push(rb);
            neighbours.entry(rb).or_default().push(ra);
        }
    }
    let mut rims: Vec<SeamRim> = Vec::new();
    let mut seen: BTreeSet<VertexId> = BTreeSet::new();
    let mut group: Vec<VertexId> = Vec::new();
    for &start in neighbours.keys() {
        if !seen.insert(start) {
            continue;
        }
        group.clear();
        push(&mut group, start)?;
        let mut next = 0;
        while let Some(&r) = group.get(next) {
            next += 1;
            for &n in &neighbours[&r] {
                if seen.insert(n) {
                    push(&mut group, n)?;
                }
            }
        }
        if group.iter().any(|r| neighbours[r].len() % 2 == 1) {
            continue;
        }
        let single = |r: &VertexId| {
            leaving.get(r).is_some_and(|l| l.len() == 1) && arriving.get(r) == Some(&1)
        };
        if !group.iter().all(single) {
            return Err(OpsError::InvalidInput("a cut rim cannot be capped"));
        }
        let mut rim: SeamRim = Vec::new();
        reserve(&mut rim, group.len())?;
        let mut r = start;
        loop {
            let (from, to) = leaving[&r][0];
            rim.push((r, from));
            r = to;
            if r == start {
                break;
            }
        }
        rims.push(rim);
    }
    Ok(rims)
}

/// Caps the rims that close by position among `edges`. The triangulation and
/// refinement of [`fill_holes`] read a mesh whose rim closes by index, so they
/// run on a scratch mesh: the triangles at the rims, with coincident vertices
/// merged. The patches then go to the real mesh on the walls' rim vertices.
fn cap_across_seams(tx: &mut Transaction, edges: &[(VertexId, VertexId)]) -> Result<()> {
    let mesh = tx.mesh();
    let roots = weld_roots(mesh, 0.0)?;
    let rims = seam_rims(mesh, edges, &roots)?;
    if rims.is_empty() {
        return Ok(());
    }
    let mut on_rim = filled(mesh.vertex_slots(), false)?;
    for rim in &rims {
        for &(r, _) in rim {
            on_rim[r as usize] = true;
        }
    }
    const NONE: u32 = u32::MAX;
    let mut local = filled(mesh.vertex_slots(), NONE)?;
    let mut positions: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        let tri = tri.map(|v| roots[v as usize]);
        let distinct = tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0];
        if !distinct || !tri.iter().any(|&r| on_rim[r as usize]) {
            continue;
        }
        reserve(&mut indices, 3)?;
        for r in tri {
            if local[r as usize] == NONE {
                local[r as usize] = (positions.len() / 3) as u32;
                reserve(&mut positions, 3)?;
                positions.extend(mesh.positions()[r as usize]);
            }
            indices.push(local[r as usize]);
        }
    }
    // No origin: the scratch mesh holds the very local positions of `mesh`.
    let scratch = Mesh::from_buffers(&positions, &indices)?;
    let map = EdgeMap::build(&scratch)?;
    let fans = VertexFans::build(&scratch)?;
    let defaults = FillOptions::default();
    let mut plan = Plan::default();
    let mut wall = filled(positions.len() / 3, NONE)?;
    let mut order: Vec<VertexId> = Vec::new();
    for rim in &rims {
        order.clear();
        reserve(&mut order, rim.len())?;
        for &(r, from) in rim {
            order.push(local[r as usize]);
            wall[local[r as usize] as usize] = from;
        }
        let tris = triangulate_loop(&scratch, &order, &map, defaults.dp_limit, None)?
            .map_err(|_| OpsError::InvalidInput("a cut rim cannot be capped"))?;
        let mut patch = refine(
            &scratch,
            &fans,
            &map,
            &order,
            &tris,
            defaults.max_refine_vertices,
        )?;
        for v in &mut patch.old {
            *v = wall[*v as usize];
        }
        plan.add(&patch)?;
    }
    plan.write(tx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{edit, grid_without, mesh_of};
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

    // ---- caps across UV seams

    /// A tube around `z` of `slices` columns and `rows` rows, wound outward,
    /// with a UV seam: column `slices` repeats column 0 exactly, `u` runs
    /// along the columns and `v` up the rows. With `lids` both ends are
    /// closed by a fan around a centre vertex, so the mesh is closed by
    /// position, though its boundary loop by index runs along the seam.
    fn seamed_tube(slices: u32, rows: u32, lids: bool) -> Mesh {
        let (mut positions, mut uvs, mut triangles) = (Vec::new(), Vec::new(), Vec::new());
        let id = |i: u32, j: u32| i * (slices + 1) + j;
        for i in 0..=rows {
            for j in 0..=slices {
                let a = std::f32::consts::TAU * (j % slices) as f32 / slices as f32;
                positions.extend([a.cos(), a.sin(), i as f32]);
                uvs.extend([j as f32 / slices as f32, i as f32 / rows as f32]);
            }
        }
        for i in 0..rows {
            for j in 0..slices {
                let (a, b) = (id(i, j), id(i, j + 1));
                let (c, d) = (id(i + 1, j + 1), id(i + 1, j));
                triangles.extend([[a, b, c], [a, c, d]]);
            }
        }
        if lids {
            let bottom = (positions.len() / 3) as u32;
            positions.extend([0.0, 0.0, 0.0, 0.0, 0.0, rows as f32]);
            uvs.extend([0.5, 0.0, 0.5, 1.0]);
            for j in 0..slices {
                triangles.push([bottom, id(0, j + 1), id(0, j)]);
                triangles.push([bottom + 1, id(rows, j), id(rows, j + 1)]);
            }
        }
        let mut mesh = mesh_of(&positions, &triangles);
        mesh.add_uvs(&uvs).unwrap();
        mesh
    }

    /// A unit cube with every triangle its own UV island: 36 vertices, every
    /// edge a seam.
    fn cube_of_islands() -> Mesh {
        let whole = cube();
        let (mut positions, mut uvs, mut triangles) = (Vec::new(), Vec::new(), Vec::new());
        for (t, tri) in whole.triangles().iter().enumerate() {
            let island = t as f32;
            let base = (positions.len() / 3) as u32;
            for (k, &v) in tri.iter().enumerate() {
                positions.extend(whole.position(v).unwrap());
                uvs.extend([island / 12.0, k as f32 / 3.0]);
            }
            triangles.push([base, base + 1, base + 2]);
        }
        let mut mesh = mesh_of(&positions, &triangles);
        mesh.add_uvs(&uvs).unwrap();
        mesh
    }

    /// `mesh` with exactly coincident vertices merged and no channels.
    fn welded_by_position(mesh: &Mesh) -> Mesh {
        let roots = crate::weld::weld_roots(mesh, 0.0).unwrap();
        let positions: Vec<f32> = mesh.positions().iter().flatten().copied().collect();
        let triangles: Vec<[u32; 3]> = (0..mesh.triangle_slots() as TriangleId)
            .filter_map(|t| mesh.triangle(t))
            .map(|t| t.map(|v| roots[v as usize]))
            .collect();
        mesh_of(&positions, &triangles)
    }

    /// Every edge, read by position, is used once in each direction.
    fn assert_closed_by_position(mesh: &Mesh) {
        let welded = welded_by_position(mesh);
        let mut uses: BTreeMap<(u32, u32), i32> = BTreeMap::new();
        for t in 0..welded.triangle_slots() as TriangleId {
            let Some([a, b, c]) = welded.triangle(t) else {
                continue;
            };
            // A plane within rounding of a vertex can leave a triangle without
            // area: it spans its two positions both ways, which closes nothing.
            if a == b || b == c || c == a {
                continue;
            }
            for (x, y) in [(a, b), (b, c), (c, a)] {
                *uses.entry((x, y)).or_default() += 1;
            }
        }
        for (&(a, b), &n) in &uses {
            assert_eq!(n, 1, "edge {a}-{b}");
            assert_eq!(uses.get(&(b, a)), Some(&1), "edge {b}-{a} is open");
        }
    }

    /// Cuts `mesh` with a cap and checks what any capped cut of a mesh that
    /// is closed by position owes: both sides closed by position and wound
    /// outward, volume kept, an honest report, aligned UVs. Returns the kept
    /// side and the output.
    fn assert_cut_closes(mut mesh: Mesh, plane: &Plane) -> (Mesh, CutOutput) {
        assert_closed_by_position(&mesh);
        let whole = volume6(&mesh);
        let triangles = mesh.triangle_count();
        let out = cut_kept(&mut mesh, plane, true);
        assert_closed_by_position(&mesh);
        assert_closed_by_position(&out.removed);
        let (vk, vr) = (volume6(&mesh), volume6(&out.removed));
        assert!(vk > 0.0 && vr > 0.0, "{vk} {vr}");
        assert!(
            (vk + vr - whole).abs() < 1e-4 * whole,
            "{vk} + {vr} != {whole}"
        );
        assert_eq!(out.report.kept_triangles, mesh.triangle_count());
        assert_eq!(out.report.removed_triangles, out.removed.triangle_count());
        assert!(out.report.cap_triangles > 0, "{:?}", out.report);
        assert!(mesh.triangle_count() + out.removed.triangle_count() > triangles);
        for m in [&mesh, &out.removed] {
            assert_eq!(m.uvs().unwrap().len(), 2 * m.vertex_slots());
            assert!(m.uvs().unwrap().iter().all(|u| u.is_finite()));
        }
        (mesh, out)
    }

    #[test]
    fn a_seam_across_the_plane_is_capped_on_both_sides() {
        let plane = plane([0.0, 0.0, 1.5], [0.0, 0.0, 1.0]);
        let (kept, out) = assert_cut_closes(seamed_tube(16, 3, true), &plane);
        // The seam stays: its sides are apart by index on both parts, and
        // the cap closes against them by position.
        assert!(!boundary_loops(&kept).unwrap().is_empty());
        assert!(!boundary_loops(&out.removed).unwrap().is_empty());
        // Rim vertices keep the wall's `v` (half way up), so every cap vertex
        // blends only values with that `v`.
        for m in [&kept, &out.removed] {
            let uvs = m.uvs().unwrap();
            let on_plane = (0..m.vertex_slots() as VertexId)
                .filter(|&v| {
                    m.position(v)
                        .is_some_and(|p| (f64::from(p[2]) + m.origin()[2] - 1.5).abs() < 1e-5)
                })
                .collect::<Vec<_>>();
            assert!(on_plane.len() >= 17, "{}", on_plane.len());
            for v in on_plane {
                assert!((uvs[2 * v as usize + 1] - 0.5).abs() < 1e-5, "vertex {v}");
                assert!((0.0..=1.0).contains(&uvs[2 * v as usize]), "vertex {v}");
            }
        }
    }

    #[test]
    fn every_edge_a_seam_is_capped_on_both_sides() {
        for p in [
            plane([0.5, 0.5, 0.5], [1.0, 0.0, 0.0]),
            plane([0.25, 0.5, 0.5], [1.0, 0.25, 0.125]),
            plane([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
        ] {
            assert_cut_closes(cube_of_islands(), &p);
        }
    }

    #[test]
    fn oblique_cuts_across_a_seam_close() {
        for p in [
            plane([0.0, 0.0, 1.5], [0.3, 0.2, 1.0]),
            plane([0.1, 0.0, 1.2], [1.0, 0.1, 1.0]),
            // Through the axis and across the seam at an angle.
            plane([0.0, 0.0, 1.5], [0.2, 1.0, 0.1]),
        ] {
            assert_cut_closes(seamed_tube(24, 3, true), &p);
        }
    }

    #[test]
    fn a_plane_through_the_seam_vertices_closes() {
        // Through a row of vertices, the two sides of the seam included.
        assert_cut_closes(
            seamed_tube(16, 4, true),
            &plane([0.0, 0.0, 2.0], [0.0, 0.0, 1.0]),
        );
        assert_cut_closes(
            seamed_tube(16, 4, true),
            &plane([0.0, 0.0, 2.0], [0.0, 0.0, -1.0]),
        );
    }

    #[test]
    fn a_plane_along_the_seam_closes() {
        // The seam itself lies in the plane: one side keeps its edges and the
        // other side's triangles leave.
        assert_cut_closes(
            seamed_tube(16, 3, true),
            &plane([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        );
        assert_cut_closes(
            seamed_tube(16, 3, true),
            &plane([0.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        );
    }

    #[test]
    fn a_plane_on_a_face_of_islands_closes_like_a_plain_cube() {
        // Only that face stays: a flat sheet, closed against its own cap.
        let on_face = plane([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
        let mut plain = cube();
        let mut islands = cube_of_islands();
        let (a, b) = (
            cut_kept(&mut plain, &on_face, true),
            cut_kept(&mut islands, &on_face, true),
        );
        assert!(a.report.cap_triangles > 0 && b.report.cap_triangles > 0);
        assert_closed_by_position(&islands);
        assert_closed_by_position(&b.removed);
    }

    #[test]
    fn an_open_tube_with_a_seam_gets_its_cut_capped_and_its_ends_left() {
        let mut mesh = seamed_tube(16, 3, false);
        let out = cut_kept(&mut mesh, &plane([0.0, 0.0, 1.5], [0.0, 0.0, 1.0]), true);
        assert!(out.report.cap_triangles > 0);
        // Only the far ends stay open by position.
        for m in [&mesh, &out.removed] {
            let open = boundary_loops(&welded_by_position(m)).unwrap();
            assert_eq!(open.len(), 1, "{open:?}");
            assert_eq!(open[0].vertices.len(), 16);
        }
    }

    #[test]
    fn an_open_sheet_with_a_seam_stays_open_without_an_error() {
        // The cut runs from border to border: no rim closes, so no cap is due.
        let mut mesh = seamed_tube(16, 3, false);
        let half = plane([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        // Half a tube first, without a cap: an open sheet with no seam left
        // on one side, cut again across the rows.
        let out = cut_kept(&mut mesh, &half, false);
        assert_eq!(out.report.cap_triangles, 0);
        let across = plane([0.0, 0.0, 1.5], [0.0, 0.0, 1.0]);
        let again = cut_kept(&mut mesh, &across, true);
        assert_eq!(again.report.cap_triangles, 0);
        assert_eq!(boundary_loops(&mesh).unwrap().len(), 1);
    }

    #[test]
    fn a_rim_at_a_seam_wound_against_itself_is_an_error_and_writes_nothing() {
        // The cube's islands, with one crossed face flipped: closed by
        // position, but no patch can agree with both neighbours.
        let whole = cube_of_islands();
        let positions: Vec<f32> = whole.positions().iter().flatten().copied().collect();
        let mut triangles = whole.triangles().to_vec();
        // Triangles 4 and 5 are the front face (-y), crossed by `x = 0.5`.
        triangles[4].swap(1, 2);
        triangles[5].swap(1, 2);
        let mut mesh = mesh_of(&positions, &triangles);
        mesh.add_uvs(whole.uvs().unwrap()).unwrap();
        let before = crate::test_util::parts(&mesh);
        let mut tx = mesh.begin_edit();
        let r = cut_by_plane(
            &mut tx,
            &plane([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]),
            &CutOptions::default(),
        );
        assert_eq!(
            r.unwrap_err(),
            OpsError::InvalidInput("a cut rim cannot be capped")
        );
        tx.rollback();
        assert_eq!(crate::test_util::parts(&mesh), before);
    }

    #[test]
    fn overlaid_seamed_sheets_are_an_error_rather_than_a_silent_gap() {
        let one = seamed_tube(16, 3, true);
        let (p, t) = crate::test_util::parts(&one);
        let base = (p.len() / 3) as u32;
        let positions: Vec<f32> = p.iter().chain(&p).copied().collect();
        let triangles: Vec<[u32; 3]> = t
            .iter()
            .copied()
            .chain(t.iter().map(|x| x.map(|v| v + base)))
            .collect();
        let uvs: Vec<f32> = one.uvs().unwrap().repeat(2);
        let mut mesh = mesh_of(&positions, &triangles);
        mesh.add_uvs(&uvs).unwrap();
        let mut tx = mesh.begin_edit();
        let r = cut_by_plane(
            &mut tx,
            &plane([0.0, 0.0, 1.5], [0.0, 0.0, 1.0]),
            &CutOptions::default(),
        );
        assert_eq!(
            r.unwrap_err(),
            OpsError::InvalidInput("a cut rim cannot be capped")
        );
        tx.rollback();
    }

    #[test]
    fn cap_off_leaves_a_seamed_mesh_open() {
        let mut mesh = seamed_tube(16, 3, true);
        let out = cut_kept(&mut mesh, &plane([0.0, 0.0, 1.5], [0.0, 0.0, 1.0]), false);
        assert_eq!(out.report.cap_triangles, 0);
        assert!(
            !boundary_loops(&welded_by_position(&mesh))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn without_uvs_split_vertices_are_read_by_index_as_before() {
        // No UV channel, no seams: duplicate vertices are a real boundary, as
        // for every other operation that reads such a mesh. Weld it first.
        let seamed = seamed_tube(16, 3, true);
        let (p, t) = crate::test_util::parts(&seamed);
        let mut mesh = mesh_of(&p, &t);
        let out = cut_kept(&mut mesh, &plane([0.0, 0.0, 1.5], [0.0, 0.0, 1.0]), true);
        assert_eq!(out.report.cap_triangles, 0);
        assert!(
            !boundary_loops(&welded_by_position(&mesh))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_seamless_mesh_with_uvs_caps_exactly_as_one_without() {
        let p = plane([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let mut plain = cube();
        let mut with_uvs = cube();
        with_uvs.add_uvs(&[0.25; 16]).unwrap();
        let (a, b) = (
            cut_kept(&mut plain, &p, true),
            cut_kept(&mut with_uvs, &p, true),
        );
        assert_eq!(a.report, b.report);
        assert_eq!(
            crate::test_util::parts(&plain),
            crate::test_util::parts(&with_uvs)
        );
        assert_eq!(
            crate::test_util::parts(&a.removed),
            crate::test_util::parts(&b.removed)
        );
    }

    #[test]
    fn a_crossing_is_the_same_point_from_either_end_when_asked() {
        let p = plane([0.1, 0.2, 0.3], [0.3, -0.7, 0.9]);
        // A fixed walk over pairs on opposite sides, far from round numbers.
        let at = |i: u32, s: f64| {
            let t = f64::from(i);
            [
                s * (1.0 + (t * 0.37).sin()) / 3.0,
                s * (t * 0.11).cos() / 7.0,
                s * (2.0 + (t * 0.73).cos()) / 11.0,
            ]
        };
        let mut differ = 0;
        for i in 0..1000 {
            let (a, b) = (at(i, 1.0), at(i + 1, -1.0));
            let (sa, sb) = (p.signed_distance(a), p.signed_distance(b));
            if (sa >= 0.0) == (sb >= 0.0) {
                continue;
            }
            assert_eq!(
                crossing_point(a, b, sa, sb, true),
                crossing_point(b, a, sb, sa, true)
            );
            differ += usize::from(
                crossing_point(a, b, sa, sb, false) != crossing_point(b, a, sb, sa, false),
            );
        }
        // Otherwise the test shows nothing: id order does move the point.
        assert!(differ > 0);
    }

    /// `mesh` welded by position, with no UV channel: the same solid, seamless.
    fn seamless(mesh: &Mesh) -> Mesh {
        let welded = welded_by_position(mesh);
        let (p, t) = crate::test_util::parts(&welded);
        mesh_of(&p, &t)
    }

    #[test]
    fn a_sweep_of_planes_never_leaves_a_seamed_mesh_open() {
        let meshes: [fn() -> Mesh; 3] = [
            || seamed_tube(12, 3, true),
            || seamed_tube(7, 2, true),
            cube_of_islands,
        ];
        // Round values hit rows, columns and the seam; the others miss them.
        let origins = [0.0, 0.25, 0.5, 1.0, 1.5, 0.3137];
        let normals = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.3, -0.5, 0.8],
            [-0.7, 0.2, 0.1],
        ];
        let mut cuts = 0;
        for make in meshes {
            for &o in &origins {
                for &n in &normals {
                    let p = plane(n.map(|c: f64| c * o), n);
                    let mut mesh = make();
                    let whole = volume6(&mesh);
                    let mut tx = mesh.begin_edit();
                    let r = cut_by_plane(&mut tx, &p, &CutOptions::default());
                    let Ok(out) = r else {
                        let err = r.unwrap_err();
                        tx.rollback();
                        // What may fail is what fails on the seamless solid.
                        let mut plain = seamless(&make());
                        let mut tx = plain.begin_edit();
                        let same = cut_by_plane(&mut tx, &p, &CutOptions::default());
                        assert_eq!(same.map(|o| o.report), Err(err), "{o} {n:?}");
                        tx.rollback();
                        continue;
                    };
                    tx.commit().unwrap();
                    cuts += 1;
                    assert_closed_by_position(&mesh);
                    assert_closed_by_position(&out.removed);
                    let sum = volume6(&mesh) + volume6(&out.removed);
                    assert!(
                        (sum - whole).abs() < 1e-4 * whole,
                        "{o} {n:?}: {sum} {whole}"
                    );
                }
            }
        }
        assert!(cuts > 80, "{cuts}");
    }

    #[test]
    fn a_rim_above_the_search_limit_caps_across_a_seam_too() {
        // More rim edges than `dp_limit`: the projection fallback, which needs
        // the walls to flare away from the plane on both sides, so an
        // hourglass cut at its waist.
        let slices = FillOptions::default().dp_limit as u32 + 40;
        let tube = seamed_tube(slices, 2, true);
        let (mut p, t) = crate::test_util::parts(&tube);
        for c in p.as_chunks_mut::<3>().0 {
            if c[2] != 1.0 {
                c[0] *= 2.0;
                c[1] *= 2.0;
            }
        }
        let mut hourglass = mesh_of(&p, &t);
        hourglass.add_uvs(tube.uvs().unwrap()).unwrap();
        let waist = plane([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
        let (kept, out) = assert_cut_closes(hourglass, &waist);
        assert!(kept.triangle_count() > 2 * slices as usize);
        assert!(out.removed.triangle_count() > 2 * slices as usize);
    }
}
