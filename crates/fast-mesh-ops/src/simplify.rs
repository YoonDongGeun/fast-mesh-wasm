//! Mesh simplification by quadric-error edge collapse.
//!
//! [`simplify`] shrinks a mesh toward [`SimplifyOptions::target_triangles`]
//! by repeatedly collapsing the edge whose collapse adds the least quadric
//! error (Garland-Heckbert): each vertex holds the sum of the squared
//! distances to its incident face planes, an edge collapses to the point
//! minimizing the summed quadrics, and the vertex quadrics merge. Boundary
//! vertices stay put when [`SimplifyOptions::preserve_boundary`] is set, so
//! open rims keep their shape. Collapses that would break the link condition
//! or fold the fan are skipped. The whole plan is computed on working copies
//! first and committed through one [`Transaction`](fast_mesh_store::Transaction),
//! which keeps the call atomic under sweep: either every collapse lands or
//! nothing does.
//!
//! UVs ([`Mesh::uvs`](fast_mesh_store::Mesh::uvs)) follow: a survivor takes the
//! UV at its new position's parameter along the collapsed edge. The quadrics
//! ignore UVs, so textures can stretch. With `preserve_boundary`, seams (which
//! are boundaries) stay exactly as they are.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use fast_mesh_store::{Transaction, TriangleId, UV_CHANNEL, VertexId};

use crate::{OpsError, Result, filled, push, reserve};

/// The parameter along `a`-`b` of the point nearest to `p`, in `[0, 1]`; the
/// middle of a zero-length edge.
fn edge_param(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f32 {
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    if len2 > 0.0 {
        let along = (p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1] + (p[2] - a[2]) * d[2];
        (along / len2).clamp(0.0, 1.0) as f32
    } else {
        0.5
    }
}

/// Options for [`simplify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimplifyOptions {
    /// Stop once this many triangles are alive. Must be at least 1; a target
    /// at or above the live count leaves the mesh alone.
    pub target_triangles: usize,
    /// Keep boundary vertices in place: no edge touching a boundary vertex
    /// collapses. Default `true`, so open rims keep their shape.
    pub preserve_boundary: bool,
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        SimplifyOptions {
            target_triangles: 0,
            preserve_boundary: true,
        }
    }
}

impl SimplifyOptions {
    /// A builder for [`SimplifyOptions`].
    pub fn builder() -> SimplifyOptionsBuilder {
        SimplifyOptionsBuilder::default()
    }
}

/// Builder for [`SimplifyOptions`].
#[derive(Debug, Clone, Default)]
pub struct SimplifyOptionsBuilder {
    options: SimplifyOptions,
}

impl SimplifyOptionsBuilder {
    /// Stop once this many triangles are alive. Required: the default 0 is
    /// rejected by [`simplify`].
    pub fn target_triangles(mut self, target: usize) -> Self {
        self.options.target_triangles = target;
        self
    }

    /// Keep boundary vertices in place.
    pub fn preserve_boundary(mut self, preserve: bool) -> Self {
        self.options.preserve_boundary = preserve;
        self
    }

    /// The options as set.
    pub fn build(self) -> SimplifyOptions {
        self.options
    }
}

/// Outcome of [`simplify`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SimplifyReport {
    /// Edge collapses performed.
    pub collapsed: usize,
    /// Triangles removed (one per boundary collapse, two per interior one).
    pub triangles_removed: usize,
    /// Vertices removed (one per collapse).
    pub vertices_removed: usize,
}

/// A working triangle: `id` is its slot in the store.
struct WorkTri {
    id: TriangleId,
    corners: [u32; 3],
    alive: bool,
}

/// A collapse candidate, ordered cheapest-first in the max-heap.
struct HeapEntry {
    cost: f64,
    a: u32,
    b: u32,
    ver_a: u32,
    ver_b: u32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
            && self.a == other.a
            && self.b == other.b
            && self.ver_a == other.ver_a
            && self.ver_b == other.ver_b
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` pops the greatest: rank cheaper edges greater, with
        // the vertex ids and versions breaking ties deterministically.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.a.cmp(&self.a))
            .then_with(|| other.b.cmp(&self.b))
            .then_with(|| other.ver_a.cmp(&self.ver_a))
            .then_with(|| other.ver_b.cmp(&self.ver_b))
    }
}

/// A symmetric 4x4 quadric as its upper triangle:
/// `q00 q01 q02 q03 q11 q12 q13 q22 q23 q33`.
type Quadric = [f64; 10];

fn f64v(p: &[f32; 3]) -> [f64; 3] {
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The quadric of one face plane: squared distance to it.
fn face_quadric(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> Quadric {
    let n = cross3(sub3(p1, p0), sub3(p2, p0));
    let len = dot3(n, n).sqrt();
    if !len.is_finite() || len == 0.0 {
        return [0.0; 10];
    }
    let n = [n[0] / len, n[1] / len, n[2] / len];
    if !n.iter().all(|c| c.is_finite()) {
        return [0.0; 10];
    }
    let d = -dot3(n, p0);
    if !d.is_finite() {
        return [0.0; 10];
    }
    let v = [n[0], n[1], n[2], d];
    let mut q = [0.0; 10];
    let mut k = 0;
    for i in 0..4 {
        for j in i..4 {
            q[k] = v[i] * v[j];
            k += 1;
        }
    }
    q
}

/// The quadric error of `p`: `p^T A p + 2 b^T p + q33`.
fn quad_error(q: &Quadric, p: [f64; 3]) -> f64 {
    let [x, y, z] = p;
    x * x * q[0]
        + 2.0 * x * y * q[1]
        + 2.0 * x * z * q[2]
        + 2.0 * x * q[3]
        + y * y * q[4]
        + 2.0 * y * z * q[5]
        + 2.0 * y * q[6]
        + z * z * q[7]
        + 2.0 * z * q[8]
        + q[9]
}

/// The point minimizing `q` and its error: the linear solve, or the midpoint
/// when the 3x3 block is singular or the solve is not finite.
fn optimal_position(q: &Quadric, a: [f64; 3], b: [f64; 3]) -> ([f64; 3], f64) {
    let (a00, a01, a02, b0) = (q[0], q[1], q[2], q[3]);
    let (a11, a12, b1) = (q[4], q[5], q[6]);
    let (a22, b2) = (q[7], q[8]);
    let det = a00 * (a11 * a22 - a12 * a12) - a01 * (a01 * a22 - a12 * a02)
        + a02 * (a01 * a12 - a11 * a02);
    if det.is_finite() && det.abs() > 1e-12 {
        // Adjugate over the determinant (Cramer on `A x = -b`).
        let inv = [
            (a11 * a22 - a12 * a12) / det,
            (a02 * a12 - a01 * a22) / det,
            (a01 * a12 - a11 * a02) / det,
            (a02 * a12 - a01 * a22) / det,
            (a00 * a22 - a02 * a02) / det,
            (a02 * a01 - a00 * a12) / det,
            (a01 * a12 - a11 * a02) / det,
            (a02 * a01 - a00 * a12) / det,
            (a00 * a11 - a01 * a01) / det,
        ];
        // Symmetric: inv01 == inv10 and so on; index the upper triangle.
        let x = -(inv[0] * b0 + inv[1] * b1 + inv[2] * b2);
        let y = -(inv[1] * b0 + inv[4] * b1 + inv[5] * b2);
        let z = -(inv[2] * b0 + inv[5] * b1 + inv[8] * b2);
        let p = [x, y, z];
        if p.iter().all(|c| c.is_finite()) {
            let cost = quad_error(q, p);
            if cost.is_finite() && cost >= -1e-9 * (1.0 + q[9].abs()) {
                return (p, cost.max(0.0));
            }
        }
    }
    let mid = [
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
        (a[2] + b[2]) / 2.0,
    ];
    let cost = quad_error(q, mid);
    if cost.is_finite() {
        (mid, cost.max(0.0))
    } else {
        // Degenerate quadrics (huge coordinates): fall back to the squared
        // edge length so the ordering stays finite and shortest-first.
        let d = sub3(a, b);
        (mid, dot3(d, d))
    }
}

/// The undirected key of the edge `a`-`b`.
fn key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// Whether `corners` carries both `a` and `b`.
fn carries(corners: &[u32; 3], a: u32, b: u32) -> bool {
    corners.contains(&a) && corners.contains(&b)
}

/// The corner of a triangle carrying `(a, b)` that is neither endpoint.
fn apex_of(corners: &[u32; 3], a: u32, b: u32) -> u32 {
    if corners[0] != a && corners[0] != b {
        corners[0]
    } else if corners[1] != a && corners[1] != b {
        corners[1]
    } else {
        corners[2]
    }
}

/// Reusable scratch buffers for edge collapses inside [`simplify`].
#[derive(Default)]
struct CollapseScratch {
    owners: Vec<usize>,
    na: Vec<u32>,
    nb: Vec<u32>,
    apexes: Vec<u32>,
    common: Vec<u32>,
    fan: Vec<usize>,
    v2t_drop: Vec<u32>,
    ring: Vec<u32>,
}

/// Fills `owners` with the sorted, unique alive working triangles carrying `(a, b)`.
fn fill_owners(
    a: u32,
    b: u32,
    v2t: &[Vec<u32>],
    tris: &[WorkTri],
    owners: &mut Vec<usize>,
) -> Result<()> {
    owners.clear();
    reserve(owners, 4)?;
    for &i in &v2t[a as usize] {
        let t = &tris[i as usize];
        if t.alive && carries(&t.corners, a, b) {
            push(owners, i as usize)?;
        }
    }
    owners.sort_unstable();
    owners.dedup();
    Ok(())
}

/// Fills `out` with the sorted, unique alive vertices sharing an alive triangle with `v`.
fn fill_neighbors(v: u32, v2t: &[Vec<u32>], tris: &[WorkTri], out: &mut Vec<u32>) -> Result<()> {
    out.clear();
    reserve(out, 16)?;
    for &i in &v2t[v as usize] {
        let t = &tris[i as usize];
        if !t.alive || !t.corners.contains(&v) {
            continue;
        }
        for &w in &t.corners {
            if w != v {
                push(out, w)?;
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(())
}

fn tri_normal(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> [f64; 3] {
    cross3(sub3(p1, p0), sub3(p2, p0))
}

/// Shrinks the mesh toward `options.target_triangles` by quadric-error edge
/// collapses (see the module docs). Boundary vertices stay put when
/// `preserve_boundary` is set. Stops early when no valid collapse is left,
/// which can leave more triangles than the target on meshes with locked or
/// non-manifold edges.
///
/// A target at or above the live count, and an empty mesh, succeed without
/// writing. Otherwise at least one collapse runs and the patch is non-empty.
///
/// # Errors
/// [`OpsError::InvalidInput`] for a zero target; [`OpsError::Store`] when
/// memory runs out.
pub fn simplify(tx: &mut Transaction, options: &SimplifyOptions) -> Result<SimplifyReport> {
    if options.target_triangles == 0 {
        return Err(OpsError::InvalidInput(
            "simplify target_triangles must be at least 1",
        ));
    }
    let mesh = tx.mesh();
    let base = mesh.positions();
    let mut positions: Vec<[f32; 3]> = Vec::new();
    reserve(&mut positions, base.len())?;
    positions.extend_from_slice(base);
    let slots = positions.len();
    // The working UVs, on a mesh with them.
    let mut uvs: Option<Vec<f32>> = None;
    if let Some(from) = mesh.uvs() {
        let mut copy = Vec::new();
        reserve(&mut copy, from.len())?;
        copy.extend_from_slice(from);
        uvs = Some(copy);
    }
    let mut tris: Vec<WorkTri> = Vec::new();
    reserve(&mut tris, mesh.triangle_slots())?;
    for t in 0..mesh.triangle_slots() as u32 {
        let Some(corners) = mesh.triangle(t) else {
            continue;
        };
        if corners[0] == corners[1] || corners[1] == corners[2] || corners[2] == corners[0] {
            continue;
        }
        push(
            &mut tris,
            WorkTri {
                id: t,
                corners,
                alive: true,
            },
        )?;
    }
    let mut live = tris.len();
    if live <= options.target_triangles {
        return Ok(SimplifyReport::default());
    }

    let mut quadrics: Vec<Quadric> = Vec::new();
    reserve(&mut quadrics, slots)?;
    quadrics.resize(slots, [0.0; 10]);
    for tri in &tris {
        let c = tri.corners;
        let q = face_quadric(
            f64v(&positions[c[0] as usize]),
            f64v(&positions[c[1] as usize]),
            f64v(&positions[c[2] as usize]),
        );
        for &v in &c {
            for k in 0..10 {
                quadrics[v as usize][k] += q[k];
            }
        }
    }

    // Boundary vertices: endpoints of edges one live triangle uses.
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    reserve(&mut pairs, 3 * tris.len())?;
    for tri in &tris {
        let c = tri.corners;
        for k in 0..3 {
            push(&mut pairs, key(c[k], c[(k + 1) % 3]))?;
        }
    }
    pairs.sort_unstable();
    let mut is_boundary = filled(slots, false)?;
    {
        let mut start = 0;
        while start < pairs.len() {
            let mut end = start + 1;
            while end < pairs.len() && pairs[end] == pairs[start] {
                end += 1;
            }
            if end - start == 1 {
                is_boundary[pairs[start].0 as usize] = true;
                is_boundary[pairs[start].1 as usize] = true;
            }
            start = end;
        }
    }
    pairs.dedup();

    let mut v2t: Vec<Vec<u32>> = Vec::new();
    reserve(&mut v2t, slots)?;
    v2t.resize_with(slots, Vec::new);
    for (i, tri) in tris.iter().enumerate() {
        for &v in &tri.corners {
            push(&mut v2t[v as usize], i as u32)?;
        }
    }

    let mut vlive = filled(slots, true)?;
    let mut moved = filled(slots, false)?;
    let mut v_ver = filled(slots, 0u32)?;
    let mut heap = BinaryHeap::new();
    heap.try_reserve(pairs.len())
        .map_err(|_| OpsError::Store(fast_mesh_store::StoreError::OutOfMemory))?;
    for &(a, b) in &pairs {
        if options.preserve_boundary && (is_boundary[a as usize] || is_boundary[b as usize]) {
            continue;
        }
        let mut q = [0.0; 10];
        for k in 0..10 {
            q[k] = quadrics[a as usize][k] + quadrics[b as usize][k];
        }
        let (_, cost) = optimal_position(
            &q,
            f64v(&positions[a as usize]),
            f64v(&positions[b as usize]),
        );
        heap.push(HeapEntry {
            cost,
            a,
            b,
            ver_a: 0,
            ver_b: 0,
        });
    }

    let mut collapsed = 0usize;
    let mut sc = CollapseScratch::default();

    while live > options.target_triangles {
        let Some(top) = heap.pop() else {
            break;
        };
        let (a, b) = (top.a, top.b);
        if !vlive[a as usize]
            || !vlive[b as usize]
            || v_ver[a as usize] != top.ver_a
            || v_ver[b as usize] != top.ver_b
        {
            continue;
        }
        fill_owners(a, b, &v2t, &tris, &mut sc.owners)?;
        if sc.owners.is_empty() || sc.owners.len() > 2 {
            continue;
        }
        // Link condition: the common neighbours must be exactly the apexes
        // of the owners, or the merge would duplicate triangles.
        fill_neighbors(a, &v2t, &tris, &mut sc.na)?;
        fill_neighbors(b, &v2t, &tris, &mut sc.nb)?;
        sc.apexes.clear();
        reserve(&mut sc.apexes, sc.owners.len())?;
        for &o in &sc.owners {
            push(&mut sc.apexes, apex_of(&tris[o].corners, a, b))?;
        }
        sc.apexes.sort_unstable();
        sc.apexes.dedup();
        if sc.apexes.len() != sc.owners.len() {
            continue;
        }
        sc.common.clear();
        reserve(&mut sc.common, sc.na.len().min(sc.nb.len()))?;
        for &w in &sc.na {
            if w != a && w != b && sc.nb.binary_search(&w).is_ok() {
                push(&mut sc.common, w)?;
            }
        }
        if sc.common != sc.apexes {
            continue;
        }
        let mut qsum = [0.0; 10];
        for k in 0..10 {
            qsum[k] = quadrics[a as usize][k] + quadrics[b as usize][k];
        }
        let pa = f64v(&positions[a as usize]);
        let pb = f64v(&positions[b as usize]);
        let (pnew, _cost) = optimal_position(&qsum, pa, pb);
        // Keep the smaller id so repeated runs agree on the survivor.
        let (keep, drop) = if a < b { (a, b) } else { (b, a) };
        if !pnew.iter().all(|c| c.is_finite()) {
            continue;
        }
        let pnew32 = [pnew[0] as f32, pnew[1] as f32, pnew[2] as f32];
        if !pnew32.iter().all(|c| c.is_finite()) {
            continue;
        }
        // No folded fan triangle: every repointed triangle must keep the
        // sign of its normal and a non-zero area.
        sc.fan.clear();
        reserve(&mut sc.fan, sc.na.len() + sc.nb.len())?;
        for &v in &[a, b] {
            for &i in &v2t[v as usize] {
                let idx = i as usize;
                if !tris[idx].alive || sc.owners.contains(&idx) {
                    continue;
                }
                push(&mut sc.fan, idx)?;
            }
        }
        sc.fan.sort_unstable();
        sc.fan.dedup();
        let keep_pos = [
            f64::from(pnew32[0]),
            f64::from(pnew32[1]),
            f64::from(pnew32[2]),
        ];
        let mut folds = false;
        for &i in &sc.fan {
            let old = tris[i].corners;
            let pts = [
                f64v(&positions[old[0] as usize]),
                f64v(&positions[old[1] as usize]),
                f64v(&positions[old[2] as usize]),
            ];
            let before = tri_normal(pts[0], pts[1], pts[2]);
            if dot3(before, before) == 0.0 {
                folds = true;
                break;
            }
            let new = old.map(|v| {
                if v == keep || v == drop {
                    keep_pos
                } else {
                    f64v(&positions[v as usize])
                }
            });
            let after = tri_normal(new[0], new[1], new[2]);
            if dot3(after, after) == 0.0 || dot3(after, before) <= 0.0 {
                folds = true;
                break;
            }
        }
        if folds {
            continue;
        }

        if let Some(uv) = &mut uvs {
            let f = edge_param(keep_pos, pa, pb);
            let (ia, ib, ik) = (2 * a as usize, 2 * b as usize, 2 * keep as usize);
            let mixed = [0, 1].map(|c| uv[ia + c] + f * (uv[ib + c] - uv[ia + c]));
            uv[ik] = mixed[0];
            uv[ik + 1] = mixed[1];
        }
        quadrics[keep as usize] = qsum;
        positions[keep as usize] = pnew32;
        moved[keep as usize] = true;
        for &o in &sc.owners {
            tris[o].alive = false;
        }
        live -= sc.owners.len();
        // Prune dead triangles from `keep`'s adjacency, then transfer only the
        // surviving triangles from `drop` into `keep` without duplicates.
        v2t[keep as usize].retain(|&i| tris[i as usize].alive);
        sc.v2t_drop.clear();
        reserve(&mut sc.v2t_drop, v2t[drop as usize].len())?;
        sc.v2t_drop.extend_from_slice(&v2t[drop as usize]);
        v2t[drop as usize].clear();
        reserve(&mut v2t[keep as usize], sc.v2t_drop.len())?;
        for &idx in &sc.v2t_drop {
            let t = &mut tris[idx as usize];
            if !t.alive {
                continue;
            }
            t.corners = t.corners.map(|v| if v == drop { keep } else { v });
            push(&mut v2t[keep as usize], idx)?;
        }
        v2t[keep as usize].sort_unstable();
        v2t[keep as usize].dedup();
        vlive[drop as usize] = false;
        v_ver[keep as usize] = v_ver[keep as usize].wrapping_add(1);
        collapsed += 1;

        // Re-arm the survivor's edges with current quadrics and vertex versions.
        sc.ring.clear();
        reserve(&mut sc.ring, 16)?;
        for &i in &v2t[keep as usize] {
            let t = &tris[i as usize];
            if !t.alive || !t.corners.contains(&keep) {
                continue;
            }
            for &w in &t.corners {
                if w != keep && vlive[w as usize] {
                    push(&mut sc.ring, w)?;
                }
            }
        }
        sc.ring.sort_unstable();
        sc.ring.dedup();
        if heap.capacity() - heap.len() < sc.ring.len() {
            heap.try_reserve(sc.ring.len())
                .map_err(|_| OpsError::Store(fast_mesh_store::StoreError::OutOfMemory))?;
        }
        for &w in &sc.ring {
            if options.preserve_boundary && (is_boundary[keep as usize] || is_boundary[w as usize])
            {
                continue;
            }
            let (lo, hi) = key(keep, w);
            let mut q = [0.0; 10];
            for k in 0..10 {
                q[k] = quadrics[lo as usize][k] + quadrics[hi as usize][k];
            }
            let (_, cost) = optimal_position(
                &q,
                f64v(&positions[lo as usize]),
                f64v(&positions[hi as usize]),
            );
            heap.push(HeapEntry {
                cost,
                a: lo,
                b: hi,
                ver_a: v_ver[lo as usize],
                ver_b: v_ver[hi as usize],
            });
        }
    }

    #[cfg(test)]
    for list in &v2t {
        assert!(
            list.windows(2).all(|w| w[0] < w[1]),
            "duplicate or unsorted entry in v2t: {list:?}"
        );
    }

    if collapsed == 0 {
        return Ok(SimplifyReport::default());
    }

    // Reserve everything before the first write; past this point only
    // infallible pushes into reserved capacity remain.
    let mut set_p: Vec<VertexId> = Vec::new();
    let mut set_p_to: Vec<[f32; 3]> = Vec::new();
    let mut set_t: Vec<TriangleId> = Vec::new();
    let mut set_t_to: Vec<[VertexId; 3]> = Vec::new();
    let mut del_t: Vec<TriangleId> = Vec::new();
    let mut del_v: Vec<VertexId> = Vec::new();
    let mut set_uv: Vec<VertexId> = Vec::new();
    let mut set_uv_to: Vec<f32> = Vec::new();
    if let Some(uv) = &uvs {
        for v in 0..slots as u32 {
            if vlive[v as usize] && moved[v as usize] {
                push(&mut set_uv, v)?;
                let i = 2 * v as usize;
                reserve(&mut set_uv_to, 2)?;
                set_uv_to.extend_from_slice(&uv[i..i + 2]);
            }
        }
    }
    let mut n_moved = 0;
    let mut n_rewired = 0;
    let mut n_dead_tri = 0;
    let mut n_dead_vert = 0;
    for v in 0..slots as u32 {
        if !vlive[v as usize] {
            n_dead_vert += 1;
        } else if moved[v as usize] && positions[v as usize] != base[v as usize] {
            n_moved += 1;
        }
    }
    for tri in &tris {
        if !tri.alive {
            n_dead_tri += 1;
        } else {
            let current = mesh
                .triangle(tri.id)
                .expect("simplify: working id must exist");
            if current != tri.corners {
                n_rewired += 1;
            }
        }
    }
    reserve(&mut set_p, n_moved)?;
    reserve(&mut set_p_to, n_moved)?;
    reserve(&mut set_t, n_rewired)?;
    reserve(&mut set_t_to, n_rewired)?;
    reserve(&mut del_t, n_dead_tri)?;
    reserve(&mut del_v, n_dead_vert)?;
    for v in 0..slots as u32 {
        if !vlive[v as usize] {
            del_v.push(v);
        } else if moved[v as usize] && positions[v as usize] != base[v as usize] {
            set_p.push(v);
            set_p_to.push(positions[v as usize]);
        }
    }
    for tri in &tris {
        if !tri.alive {
            del_t.push(tri.id);
        } else {
            let current = mesh
                .triangle(tri.id)
                .expect("simplify: working id must exist");
            if current != tri.corners {
                set_t.push(tri.id);
                set_t_to.push(tri.corners);
            }
        }
    }
    tx.set_positions(&set_p, &set_p_to)?;
    if uvs.is_some() {
        tx.write_channel(UV_CHANNEL, &set_uv, &set_uv_to)?;
    }
    tx.set_triangles(&set_t, &set_t_to)?;
    tx.remove_triangles(&del_t)?;
    tx.remove_vertices(&del_v)?;

    let live_now = tris.iter().filter(|t| t.alive).count();
    let started = live_now + del_t.len();
    Ok(SimplifyReport {
        collapsed,
        triangles_removed: started - live_now,
        vertices_removed: del_v.len(),
    })
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::{Mesh, Transaction, VertexId};

    use super::*;
    use crate::test_util::{box_mesh, edit, grid, mesh_of, parts, sweep, uv_sphere};

    fn live_count(mesh: &Mesh) -> usize {
        (0..mesh.triangle_slots() as u32)
            .filter(|&t| mesh.triangle(t).is_some())
            .count()
    }

    fn assert_no_degenerate(mesh: &Mesh) {
        for t in 0..mesh.triangle_slots() as u32 {
            if let Some(c) = mesh.triangle(t) {
                assert!(
                    c[0] != c[1] && c[1] != c[2] && c[2] != c[0],
                    "degenerate: {c:?}"
                );
                for &v in &c {
                    assert!(mesh.is_vertex_alive(v as VertexId));
                }
            }
        }
    }

    #[test]
    fn decimate_cube_to_target_ratio() {
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        assert_eq!(live_count(&mesh), 12);
        let options = SimplifyOptions::builder().target_triangles(6).build();
        let (report, empty) = edit(&mut mesh, |tx| simplify(tx, &options));
        assert!(!empty);
        assert!(report.collapsed > 0, "expected collapses, got {report:?}");
        let now = live_count(&mesh);
        assert!(now <= 6, "expected at most 6 triangles, got {now}");
        assert!(now > 0, "must keep some triangles");
        assert_eq!(report.triangles_removed, 12 - now);
        assert_eq!(report.vertices_removed, report.collapsed);
        assert_no_degenerate(&mesh);
    }

    #[test]
    fn decimate_sphere_to_target_ratio() {
        let mut mesh = uv_sphere(8, 16, 1.0);
        let before = live_count(&mesh);
        assert!(before > 100, "fixture too small: {before}");
        let target = before / 2;
        let options = SimplifyOptions::builder().target_triangles(target).build();
        let (report, empty) = edit(&mut mesh, |tx| simplify(tx, &options));
        assert!(!empty);
        assert!(report.collapsed > 0, "expected collapses, got {report:?}");
        let now = live_count(&mesh);
        assert!(
            now <= target,
            "expected at most {target} triangles, got {now}"
        );
        assert!(now > 0, "must keep some triangles");
        assert_no_degenerate(&mesh);
    }

    #[test]
    fn boundary_preserved_when_enabled() {
        // A 4x4 grid: 25 vertices, 16 on the rim. Locking the rim keeps
        // every rim vertex alive at its position.
        let mut mesh = grid(4);
        let rim: Vec<[f32; 3]> = mesh.positions().to_vec();
        let options = SimplifyOptions::builder()
            .target_triangles(16)
            .preserve_boundary(true)
            .build();
        let (report, empty) = edit(&mut mesh, |tx| simplify(tx, &options));
        assert!(!empty);
        assert!(report.collapsed > 0, "expected collapses, got {report:?}");
        assert!(live_count(&mesh) <= 16);
        let n = 4 + 1;
        for y in 0..=4 {
            for x in 0..=4 {
                if x == 0 || y == 0 || x == 4 || y == 4 {
                    let v = y * n + x;
                    assert!(
                        mesh.is_vertex_alive(v as VertexId),
                        "rim vertex {v} removed"
                    );
                    assert_eq!(mesh.positions()[v as usize], rim[v as usize]);
                }
            }
        }
        assert_no_degenerate(&mesh);
    }

    #[test]
    fn boundary_option_changes_locked_quad() {
        // Two triangles, every vertex on the rim: locked, nothing collapses;
        // unlocked, the diagonal collapses.
        let quad = || {
            mesh_of(
                &[0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                &[[0, 1, 2], [0, 2, 3]],
            )
        };
        let mut locked = quad();
        let before = parts(&locked);
        let (report, empty) = edit(&mut locked, |tx| {
            simplify(
                tx,
                &SimplifyOptions::builder()
                    .target_triangles(1)
                    .preserve_boundary(true)
                    .build(),
            )
        });
        assert_eq!((report.collapsed, empty), (0, true));
        assert_eq!(parts(&locked), before);

        let mut open = quad();
        let (report, empty) = edit(&mut open, |tx| {
            simplify(
                tx,
                &SimplifyOptions::builder()
                    .target_triangles(1)
                    .preserve_boundary(false)
                    .build(),
            )
        });
        assert!(!empty);
        assert!(report.collapsed > 0, "expected collapses, got {report:?}");
        assert!(live_count(&open) < 2);
    }

    #[test]
    fn invalid_target_and_noop_target() {
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        {
            let mut tx = mesh.begin_edit();
            assert_eq!(
                simplify(&mut tx, &SimplifyOptions::builder().build()),
                Err(OpsError::InvalidInput(
                    "simplify target_triangles must be at least 1"
                ))
            );
        }
        // Target at/above the count succeeds without writing.
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        let before = parts(&mesh);
        let (report, empty) = edit(&mut mesh, |tx| {
            simplify(tx, &SimplifyOptions::builder().target_triangles(12).build())
        });
        assert_eq!((report.collapsed, empty), (0, true));
        assert_eq!(parts(&mesh), before);
        let (report, empty) = edit(&mut mesh, |tx| {
            simplify(
                tx,
                &SimplifyOptions::builder().target_triangles(10_000).build(),
            )
        });
        assert_eq!((report.collapsed, empty), (0, true));
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn simplify_sweep_keeps_atomicity() {
        let n = sweep(grid(4), |tx: &mut Transaction| {
            simplify(tx, &SimplifyOptions::builder().target_triangles(16).build())?;
            Ok(())
        });
        assert!(n > 0);
    }

    #[test]
    fn simplify_large_mesh_without_v2t_bloat() {
        let mut mesh = uv_sphere(160, 320, 1.0);
        let before = live_count(&mesh);
        assert!(before >= 100_000, "fixture too small: {before}");

        let started = std::time::Instant::now();
        let options = SimplifyOptions::builder().target_triangles(10_000).build();
        let (report, empty) = edit(&mut mesh, |tx| simplify(tx, &options));
        let elapsed = started.elapsed();

        assert!(!empty);
        assert!(report.collapsed > 0, "expected collapses, got {report:?}");
        let now = live_count(&mesh);
        assert!(now <= 10_000, "expected at most 10_000 tris, got {now}");
        assert!(now > 0);
        assert!(
            elapsed.as_secs_f64() < 10.0,
            "100k->10k simplify took too long: {elapsed:?}"
        );
        assert_no_degenerate(&mesh);

        let analysis = crate::analyze(&mesh, &crate::AnalyzeOptions::default()).unwrap();
        assert_eq!(analysis.non_manifold_edges, 0);
        assert!(analysis.non_manifold_vertices.is_empty());
        assert_eq!(analysis.duplicate, 0);
        assert_eq!(analysis.degenerate, 0);
        assert_eq!(analysis.orientation_conflicts, 0);
        assert!(analysis.loops.is_empty());
        assert!(analysis.pinch_vertices.is_empty());
    }
}
