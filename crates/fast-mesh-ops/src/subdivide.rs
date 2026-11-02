//! Mesh subdivision by Loop's scheme.
//!
//! [`subdivide`] refines every triangle into four, [`SubdivideOptions::levels`]
//! times, smoothing the surface toward the limit: each level moves the old
//! vertices with Loop's vertex stencil and inserts one vertex per edge with
//! the edge stencil, then splits every triangle at its three edge vertices.
//! The whole plan is computed on working copies first and committed through
//! one [`Transaction`](fast_mesh_store::Transaction), which keeps the call
//! atomic under sweep: either every level lands or nothing does.
//!
//! # Boundary and crease rule
//!
//! Standard Loop with boundary handling (no crease tags):
//!
//! - An edge two triangles share is interior: its new vertex is
//!   `3/8 (a + b) + 1/8 (c + d)` from its endpoints `a`, `b` and the two
//!   opposite vertices `c`, `d`. Any other edge (one owner, or three or more
//!   on a non-manifold rim) is a crease: its new vertex is the midpoint.
//! - A vertex on no boundary edge is interior: Loop's vertex stencil over
//!   all its neighbours, with the exact weight
//!   `beta = (5/8 - (3/8 + cos(2 pi / n) / 4)^2) / n` for valence `n`.
//! - A vertex on boundary edges with exactly two boundary neighbours is a
//!   smooth boundary vertex: `3/4 v + 1/8 (b0 + b1)`, so rims relax along
//!   their own curve and sharp corners round off.
//! - A vertex on any other number of boundary edges (one, or three or more
//!   at a pinch or non-manifold corner) is pinned: it stays where it is.
//! - Isolated vertices, and triangles with a repeated vertex id, pass
//!   through untouched.
//!
//! The stencils are convex combinations, so finite input stays finite, and
//! every list is sorted explicitly, so results do not depend on hash
//! iteration order.

use fast_mesh_store::{Transaction, VertexId};

use crate::{OpsError, Result, push, reserve};

/// Which subdivision scheme to run. Only Loop's scheme exists so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubdivideMethod {
    /// Loop's scheme for triangle meshes, with the boundary rule from the
    /// module docs.
    #[default]
    Loop,
}

/// Options for [`subdivide`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubdivideOptions {
    /// Which scheme to run.
    pub method: SubdivideMethod,
    /// How many times to split every triangle. Must be at least 1; each
    /// level quadruples the triangle count.
    pub levels: u32,
}

impl Default for SubdivideOptions {
    fn default() -> Self {
        SubdivideOptions {
            method: SubdivideMethod::Loop,
            levels: 1,
        }
    }
}

impl SubdivideOptions {
    /// A builder for [`SubdivideOptions`].
    pub fn builder() -> SubdivideOptionsBuilder {
        SubdivideOptionsBuilder::default()
    }
}

/// Builder for [`SubdivideOptions`].
#[derive(Debug, Clone, Default)]
pub struct SubdivideOptionsBuilder {
    options: SubdivideOptions,
}

impl SubdivideOptionsBuilder {
    /// Which scheme to run.
    pub fn method(mut self, method: SubdivideMethod) -> Self {
        self.options.method = method;
        self
    }

    /// How many times to split every triangle.
    pub fn levels(mut self, levels: u32) -> Self {
        self.options.levels = levels;
        self
    }

    /// The options as set.
    pub fn build(self) -> SubdivideOptions {
        self.options
    }
}

/// Outcome of [`subdivide`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SubdivideReport {
    /// Vertices added, one per subdivided edge per level.
    pub vertices_added: usize,
    /// Triangles added: three per subdivided triangle per level.
    pub triangles_added: usize,
}

/// A working triangle: `slot` is its store slot when it reuses one, `None`
/// for an added child. Degenerate input triangles are not clean and ride
/// through untouched.
struct WorkTri {
    slot: Option<u32>,
    corners: [u32; 3],
    clean: bool,
}

fn f64v(p: &[f32; 3]) -> [f64; 3] {
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

/// The undirected key of the edge `a`-`b`.
fn key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// Loop's vertex weight for valence `n`.
fn loop_beta(n: usize) -> f64 {
    use std::f64::consts::PI;
    let n = n as f64;
    (5.0 / 8.0 - (3.0 / 8.0 + 0.25 * (2.0 * PI / n).cos()).powi(2)) / n
}

/// One Loop level over the working mesh: smooths every live vertex and
/// splits every clean triangle into four. Returns whether any triangle was
/// subdivided.
fn step(pos: &mut Vec<[f32; 3]>, vlive: &mut Vec<bool>, tris: &mut Vec<WorkTri>) -> Result<bool> {
    let clean = tris.iter().filter(|t| t.clean).count();
    if clean == 0 {
        return Ok(false);
    }
    let slots = pos.len();

    // Edge entries `(edge, triangle, apex)`, sorted by edge so each group is
    // one unique edge with its owners.
    let mut entries: Vec<((u32, u32), u32, u32)> = Vec::new();
    reserve(
        &mut entries,
        clean.checked_mul(3).ok_or(crate::OpsError::Store(
            fast_mesh_store::StoreError::OutOfMemory,
        ))?,
    )?;
    for (i, tri) in tris.iter().enumerate() {
        if !tri.clean {
            continue;
        }
        let [a, b, c] = tri.corners;
        push(&mut entries, (key(a, b), i as u32, c))?;
        push(&mut entries, (key(b, c), i as u32, a))?;
        push(&mut entries, (key(c, a), i as u32, b))?;
    }
    entries.sort_unstable();

    // Unique edges in order, with their owner count and apexes.
    let mut edges: Vec<(u32, u32)> = Vec::new();
    reserve(&mut edges, entries.len())?;
    let mut owners: Vec<usize> = Vec::new();
    reserve(&mut owners, entries.len())?;
    let mut apexes: Vec<[u32; 2]> = Vec::new();
    reserve(&mut apexes, entries.len())?;
    {
        let mut start = 0;
        while start < entries.len() {
            let mut end = start + 1;
            while end < entries.len() && entries[end].0 == entries[start].0 {
                end += 1;
            }
            let (a, b) = entries[start].0;
            push(&mut edges, (a, b))?;
            push(&mut owners, end - start)?;
            let mut pair = [entries[start].2, entries[start].2];
            if end - start == 2 {
                pair[1] = entries[start + 1].2;
            }
            push(&mut apexes, pair)?;
            start = end;
        }
    }

    // Neighbours from clean triangles, and boundary neighbours from edges
    // one clean triangle uses.
    let mut nbs: Vec<Vec<u32>> = Vec::new();
    reserve(&mut nbs, slots)?;
    nbs.resize_with(slots, Vec::new);
    let mut bnd: Vec<Vec<u32>> = Vec::new();
    reserve(&mut bnd, slots)?;
    bnd.resize_with(slots, Vec::new);
    for tri in tris.iter() {
        if !tri.clean {
            continue;
        }
        let [a, b, c] = tri.corners;
        for (v, w) in [(a, b), (a, c), (b, a), (b, c), (c, a), (c, b)] {
            push(&mut nbs[v as usize], w)?;
        }
    }
    for (k, &(a, b)) in edges.iter().enumerate() {
        if owners[k] == 1 {
            push(&mut bnd[a as usize], b)?;
            push(&mut bnd[b as usize], a)?;
        }
    }
    for v in 0..slots {
        if vlive[v] {
            nbs[v].sort_unstable();
            nbs[v].dedup();
            bnd[v].sort_unstable();
            bnd[v].dedup();
        }
    }

    // Smoothed positions for the old vertices.
    let mut upd: Vec<[f32; 3]> = Vec::new();
    reserve(&mut upd, slots)?;
    upd.extend_from_slice(pos);
    for v in 0..slots as u32 {
        if !vlive[v as usize] {
            continue;
        }
        let p = f64v(&pos[v as usize]);
        let b = &bnd[v as usize];
        if !b.is_empty() {
            // Smooth boundary vertices relax along their curve; corners
            // with any other boundary-neighbour count stay pinned.
            if b.len() == 2 {
                let q0 = f64v(&pos[b[0] as usize]);
                let q1 = f64v(&pos[b[1] as usize]);
                let q = [0, 1, 2].map(|k| 0.75 * p[k] + 0.125 * (q0[k] + q1[k]));
                upd[v as usize] = [q[0] as f32, q[1] as f32, q[2] as f32];
            }
            continue;
        }
        let n = nbs[v as usize].len();
        if n == 0 {
            continue;
        }
        let beta = loop_beta(n);
        let mut q = [0, 1, 2].map(|k| (1.0 - n as f64 * beta) * p[k]);
        for &w in &nbs[v as usize] {
            let r = f64v(&pos[w as usize]);
            for k in 0..3 {
                q[k] += beta * r[k];
            }
        }
        upd[v as usize] = [q[0] as f32, q[1] as f32, q[2] as f32];
    }

    // One new vertex per unique edge, in edge order.
    let mut edge_point: Vec<[f32; 3]> = Vec::new();
    reserve(&mut edge_point, edges.len())?;
    for (k, &(a, b)) in edges.iter().enumerate() {
        let pa = f64v(&pos[a as usize]);
        let pb = f64v(&pos[b as usize]);
        let q = if owners[k] == 2 {
            let pc = f64v(&pos[apexes[k][0] as usize]);
            let pd = f64v(&pos[apexes[k][1] as usize]);
            [0, 1, 2].map(|i| 0.375 * (pa[i] + pb[i]) + 0.125 * (pc[i] + pd[i]))
        } else {
            [0, 1, 2].map(|i| 0.5 * (pa[i] + pb[i]))
        };
        push(&mut edge_point, [q[0] as f32, q[1] as f32, q[2] as f32])?;
    }

    // Split every clean triangle into four; the first child reuses its slot.
    let degenerate = tris.len() - clean;
    let mut next: Vec<WorkTri> = Vec::new();
    reserve(
        &mut next,
        clean
            .checked_mul(4)
            .and_then(|n| n.checked_add(degenerate))
            .ok_or(crate::OpsError::Store(
                fast_mesh_store::StoreError::OutOfMemory,
            ))?,
    )?;
    // `edges` is sorted: the new id of `(a, b)` via binary search.
    let edge_id = |a: u32, b: u32| -> u32 {
        let k = edges.binary_search(&key(a, b)).expect("edge must exist");
        slots as u32 + k as u32
    };
    for tri in tris.iter() {
        if !tri.clean {
            push(
                &mut next,
                WorkTri {
                    slot: tri.slot,
                    corners: tri.corners,
                    clean: false,
                },
            )?;
            continue;
        }
        let [a, b, c] = tri.corners;
        let (ab, bc, ca) = (edge_id(a, b), edge_id(b, c), edge_id(c, a));
        let children = [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]];
        for (j, corners) in children.into_iter().enumerate() {
            push(
                &mut next,
                WorkTri {
                    slot: if j == 0 { tri.slot } else { None },
                    corners,
                    clean: true,
                },
            )?;
        }
    }

    *pos = upd;
    reserve(pos, edge_point.len())?;
    pos.extend_from_slice(&edge_point);
    reserve(vlive, edge_point.len())?;
    vlive.resize(vlive.len() + edge_point.len(), true);
    *tris = next;
    Ok(true)
}

/// Subdivides the mesh [`SubdivideOptions::levels`] times by Loop's scheme
/// (see the module docs). Each level quadruples the clean triangle count:
/// every triangle becomes four, with one new vertex per subdivided edge.
/// Degenerate triangles and isolated vertices pass through untouched.
///
/// A mesh with no clean triangle succeeds without writing. Otherwise the
/// patch is non-empty and the report counts what the levels added.
///
/// # Errors
/// [`OpsError::InvalidInput`] for zero levels; [`OpsError::Store`] when
/// memory runs out.
pub fn subdivide(tx: &mut Transaction, options: &SubdivideOptions) -> Result<SubdivideReport> {
    if options.levels == 0 {
        return Err(OpsError::InvalidInput(
            "subdivide levels must be at least 1",
        ));
    }
    let _ = options.method;
    let mesh = tx.mesh();
    let slots = mesh.vertex_slots();
    let mut pos: Vec<[f32; 3]> = Vec::new();
    reserve(&mut pos, slots)?;
    pos.extend_from_slice(mesh.positions());
    let mut vlive: Vec<bool> = Vec::new();
    reserve(&mut vlive, slots)?;
    vlive.extend((0..slots as u32).map(|v| mesh.is_vertex_alive(v as VertexId)));
    let mut tris: Vec<WorkTri> = Vec::new();
    reserve(&mut tris, mesh.triangle_slots())?;
    for t in 0..mesh.triangle_slots() as u32 {
        let Some(corners) = mesh.triangle(t) else {
            continue;
        };
        push(
            &mut tris,
            WorkTri {
                slot: Some(t),
                corners,
                clean: corners[0] != corners[1]
                    && corners[1] != corners[2]
                    && corners[2] != corners[0],
            },
        )?;
    }

    let mut subdivided = false;
    for _ in 0..options.levels {
        if !step(&mut pos, &mut vlive, &mut tris)? {
            break;
        }
        subdivided = true;
    }
    if !subdivided {
        return Ok(SubdivideReport::default());
    }

    // Plan the writes: changed originals, added vertices, reused slots, and
    // added triangles with synthetic ids past `slots`.
    let base = mesh.positions();
    let mut set_p: Vec<VertexId> = Vec::new();
    let mut set_p_to: Vec<[f32; 3]> = Vec::new();
    let mut new_p: Vec<[f32; 3]> = Vec::new();
    let mut set_t: Vec<u32> = Vec::new();
    let mut set_t_to: Vec<[u32; 3]> = Vec::new();
    let mut add_t: Vec<[u32; 3]> = Vec::new();
    let mut n_moved = 0;
    for v in 0..slots as u32 {
        if vlive[v as usize] && (v as usize) < pos.len() && pos[v as usize] != base[v as usize] {
            n_moved += 1;
        }
    }
    for v in slots..pos.len() {
        if vlive[v] {
            reserve(&mut new_p, 1)?;
            push(&mut new_p, pos[v])?;
        }
    }
    reserve(&mut set_p, n_moved)?;
    reserve(&mut set_p_to, n_moved)?;
    let mut n_set_t = 0;
    let mut n_add_t = 0;
    for tri in &tris {
        if tri.slot.is_some() {
            n_set_t += 1;
        } else {
            n_add_t += 1;
        }
    }
    reserve(&mut set_t, n_set_t)?;
    reserve(&mut set_t_to, n_set_t)?;
    reserve(&mut add_t, n_add_t)?;
    for v in 0..slots as u32 {
        if vlive[v as usize] && pos[v as usize] != base[v as usize] {
            set_p.push(v as VertexId);
            set_p_to.push(pos[v as usize]);
        }
    }
    for tri in &tris {
        if let Some(slot) = tri.slot {
            set_t.push(slot);
            set_t_to.push(tri.corners);
        } else {
            add_t.push(tri.corners);
        }
    }
    // Every reservation the commit needs is made before the first write, so
    // a later failure cannot leave a partial patch behind.
    let mut real: Vec<u32> = Vec::new();
    reserve(&mut real, pos.len())?;
    let mut set_mapped: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut set_mapped, set_t_to.len())?;
    let mut add_mapped: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut add_mapped, add_t.len())?;
    tx.set_positions(&set_p, &set_p_to)?;
    // Added vertices take the next free slots, which need not follow the
    // synthetic order when the mesh has dead slots: map them over. Every
    // vector below already holds its capacity, so filling them allocates
    // nothing.
    let added = tx.add_vertices(&new_p)?;
    real.extend(0..pos.len() as u32);
    {
        let mut k = 0;
        for v in slots..pos.len() {
            if vlive[v] {
                real[v] = added[k];
                k += 1;
            }
        }
    }
    let map = |c: [u32; 3]| c.map(|v| real[v as usize] as VertexId);
    set_mapped.extend(set_t_to.iter().map(|&c| map(c)));
    add_mapped.extend(add_t.iter().map(|&c| map(c)));
    tx.set_triangles(&set_t, &set_mapped)?;
    tx.add_triangles(&add_mapped)?;
    Ok(SubdivideReport {
        vertices_added: added.len(),
        triangles_added: add_mapped.len(),
    })
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::{Mesh, Transaction, VertexId};

    use super::*;
    use crate::test_util::{box_mesh, edit, grid, mesh_of, parts, sweep};

    fn live_count(mesh: &Mesh) -> usize {
        (0..mesh.triangle_slots() as u32)
            .filter(|&t| mesh.triangle(t).is_some())
            .count()
    }

    fn live_vertices(mesh: &Mesh) -> usize {
        (0..mesh.vertex_slots() as u32)
            .filter(|&v| mesh.is_vertex_alive(v as VertexId))
            .count()
    }

    /// Unique-edge and boundary-edge counts over the live triangles.
    fn edge_counts(mesh: &Mesh) -> (usize, usize) {
        let mut pairs = Vec::new();
        for t in 0..mesh.triangle_slots() as u32 {
            if let Some(c) = mesh.triangle(t) {
                if c[0] == c[1] || c[1] == c[2] || c[2] == c[0] {
                    continue;
                }
                for k in 0..3 {
                    pairs.push(key(c[k], c[(k + 1) % 3]));
                }
            }
        }
        pairs.sort_unstable();
        let mut unique = 0;
        let mut boundary = 0;
        let mut start = 0;
        while start < pairs.len() {
            let mut end = start + 1;
            while end < pairs.len() && pairs[end] == pairs[start] {
                end += 1;
            }
            unique += 1;
            if end - start == 1 {
                boundary += 1;
            }
            start = end;
        }
        (unique, boundary)
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
    fn cube_quadruples_closed() {
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        assert_eq!(live_count(&mesh), 12);
        let (unique, boundary) = edge_counts(&mesh);
        assert_eq!(boundary, 0);
        let options = SubdivideOptions::builder().levels(1).build();
        let (report, empty) = edit(&mut mesh, |tx| subdivide(tx, &options));
        assert!(!empty);
        assert_eq!(live_count(&mesh), 48);
        assert_eq!(live_vertices(&mesh), 8 + unique);
        assert_eq!(report.vertices_added, unique);
        assert_eq!(report.triangles_added, 36);
        assert_no_degenerate(&mesh);
    }

    #[test]
    fn plane_quadruples_and_stays_planar() {
        // A 2x2 grid: 8 triangles, 8 boundary edges. Loop is affine
        // invariant, so the flat sheet stays at z = 0; each boundary edge
        // splits in two.
        let mut mesh = grid(2);
        assert_eq!(live_count(&mesh), 8);
        let (unique, boundary) = edge_counts(&mesh);
        assert_eq!(boundary, 8);
        let options = SubdivideOptions::builder().levels(1).build();
        let (report, empty) = edit(&mut mesh, |tx| subdivide(tx, &options));
        assert!(!empty);
        assert_eq!(live_count(&mesh), 32);
        assert_eq!(live_vertices(&mesh), 9 + unique);
        assert_eq!(report.vertices_added, unique);
        assert_eq!(report.triangles_added, 24);
        for v in 0..mesh.vertex_slots() as u32 {
            if mesh.is_vertex_alive(v as VertexId) {
                assert_eq!(mesh.positions()[v as usize][2], 0.0);
            }
        }
        assert_eq!(edge_counts(&mesh).1, 16);
        assert_no_degenerate(&mesh);
    }

    #[test]
    fn two_levels_multiply_by_sixteen() {
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        let options = SubdivideOptions::builder().levels(2).build();
        let (report, empty) = edit(&mut mesh, |tx| subdivide(tx, &options));
        assert!(!empty);
        assert_eq!(live_count(&mesh), 12 * 16);
        assert_eq!(live_vertices(&mesh), 8 + report.vertices_added);
        assert_eq!(report.triangles_added, 12 * 16 - 12);
        assert_no_degenerate(&mesh);
    }

    #[test]
    fn invalid_levels_and_noop_mesh() {
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        {
            let mut tx = mesh.begin_edit();
            assert_eq!(
                subdivide(&mut tx, &SubdivideOptions::builder().levels(0).build()),
                Err(OpsError::InvalidInput(
                    "subdivide levels must be at least 1"
                ))
            );
        }
        // An empty mesh succeeds without writing.
        let mut empty_mesh = mesh_of(&[], &[]);
        let before = parts(&empty_mesh);
        let (report, empty) = edit(&mut empty_mesh, |tx| {
            subdivide(tx, &SubdivideOptions::builder().levels(1).build())
        });
        assert_eq!(
            (report.vertices_added, report.triangles_added, empty),
            (0, 0, true)
        );
        assert_eq!(parts(&empty_mesh), before);
        // Degenerate triangles ride through untouched.
        let mut degenerate = mesh_of(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[[0, 0, 1]]);
        let before = parts(&degenerate);
        let (report, empty) = edit(&mut degenerate, |tx| {
            subdivide(tx, &SubdivideOptions::builder().levels(1).build())
        });
        assert_eq!(
            (report.vertices_added, report.triangles_added, empty),
            (0, 0, true)
        );
        assert_eq!(parts(&degenerate), before);
    }

    #[test]
    fn subdivide_sweep_keeps_atomicity() {
        let n = sweep(grid(2), |tx: &mut Transaction| {
            subdivide(tx, &SubdivideOptions::builder().levels(1).build())?;
            Ok(())
        });
        assert!(n > 0);
    }
}
