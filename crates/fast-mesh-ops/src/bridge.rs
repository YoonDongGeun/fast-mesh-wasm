//! Bridging two boundary loops with a triangle tube.
//!
//! [`bridge_loops`] stitches a tube between two boundary loops identified by
//! [`LoopId`](crate::LoopId). The correspondence is a minimum-area dynamic
//! program over the loop-by-loop grid with a full start search, so loops of
//! different lengths zip without twist. The loops must wind in opposite
//! directions (their Newell vectors must oppose); same-winding loops admit
//! no outward tube that is manifold at both rims and are rejected. The whole
//! strip is planned first and committed through one
//! [`Transaction`](fast_mesh_store::Transaction), so the call is atomic.

use std::cmp::Ordering;

use fast_mesh_store::{Mesh, Transaction, VertexId};

use crate::topology::{BoundaryLoop, EdgeMap, boundary_loops, cross, dot, point, sub};
use crate::{LoopId, OpsError, Result, push, reserve};

/// Default cap on each loop's length in vertices. Explicit `max_edges`
/// overrides it; the dynamic program costs O(starts x cells), so raising
/// the bound past a few hundred gets slow before it gets big.
pub const BRIDGE_DEFAULT_MAX_EDGES: usize = 128;

/// Options for [`bridge_loops`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BridgeOptions {
    /// The two loops to connect. Order is insignificant: the smaller id is
    /// always traversed first internally.
    pub loops: (LoopId, LoopId),
    /// Maximum vertices per loop. `None` means [`BRIDGE_DEFAULT_MAX_EDGES`].
    pub max_edges: Option<usize>,
}

impl BridgeOptions {
    /// A builder for the two loops to connect.
    pub fn builder(a: LoopId, b: LoopId) -> BridgeOptionsBuilder {
        BridgeOptionsBuilder {
            loops: (a, b),
            max_edges: None,
        }
    }
}

/// Builder for [`BridgeOptions`].
#[derive(Debug, Clone, Copy)]
pub struct BridgeOptionsBuilder {
    loops: (LoopId, LoopId),
    max_edges: Option<usize>,
}

impl BridgeOptionsBuilder {
    /// Maximum vertices per loop.
    pub fn max_edges(mut self, max: usize) -> Self {
        self.max_edges = Some(max);
        self
    }

    /// The options as set. Loop validity is checked by [`bridge_loops`].
    pub fn build(self) -> BridgeOptions {
        BridgeOptions {
            loops: self.loops,
            max_edges: self.max_edges,
        }
    }
}

/// Outcome of [`bridge_loops`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BridgeReport {
    /// Triangles added between the loops.
    pub triangles_added: usize,
}

/// Stitches a tube between the two loops in `options`.
///
/// The first loop (smaller id) is traversed backward and the second
/// forward; every emitted triangle uses its rim edges opposite to the rim
/// triangles, so the strip is manifold at both rims, and the minimum-area
/// correspondence keeps it untwisted. New diagonals must not already exist
/// in the mesh. Either every triangle lands or nothing does.
pub fn bridge_loops(tx: &mut Transaction, options: &BridgeOptions) -> Result<BridgeReport> {
    let (ida, idb) = options.loops;
    if ida == idb {
        return Err(OpsError::InvalidInput(
            "bridge loops must be two distinct loops",
        ));
    }
    let mesh = tx.mesh();
    let loops = boundary_loops(mesh)?;
    let la = resolve(&loops, &ida)?;
    let lb = resolve(&loops, &idb)?;
    let (first, second) = if loop_key(&ida) <= loop_key(&idb) {
        (la, lb)
    } else {
        (lb, la)
    };
    for l in [first, second] {
        if !l.closed || !l.clean || l.vertices.len() < 3 {
            return Err(OpsError::InvalidInput(
                "bridge loops must be closed and clean",
            ));
        }
    }
    let bound = options.max_edges.unwrap_or(BRIDGE_DEFAULT_MAX_EDGES);
    if first.vertices.len() > bound || second.vertices.len() > bound {
        return Err(OpsError::InvalidInput("bridge loop exceeds max_edges"));
    }
    for &v in &first.vertices {
        if second.vertices.contains(&v) {
            return Err(OpsError::InvalidInput(
                "bridge loops must not share vertices",
            ));
        }
    }
    for l in [first, second] {
        for &v in &l.vertices {
            if !point(mesh, v).iter().all(|c| c.is_finite()) {
                return Err(OpsError::InvalidInput(
                    "bridge loop positions must be finite",
                ));
            }
        }
    }
    let na = newell(mesh, &first.vertices);
    let nb = newell(mesh, &second.vertices);
    if !matches!(dot(na, nb).partial_cmp(&0.0), Some(Ordering::Less)) {
        return Err(OpsError::InvalidInput(
            "bridge loops must wind in opposite directions",
        ));
    }
    let strip = best_strip(mesh, &first.vertices, &second.vertices)?;
    for t in &strip {
        let area2 = tri_area2(&point(mesh, t[0]), &point(mesh, t[1]), &point(mesh, t[2]));
        if area2 == 0.0 {
            return Err(OpsError::InvalidInput("bridge triangulation is degenerate"));
        }
    }
    let edges = EdgeMap::build(mesh)?;
    let mut chords = chords_of(&strip, &first.vertices)?;
    chords.sort_unstable();
    chords.dedup();
    for &(a, b) in &chords {
        if edges.kind(a, b).is_some() {
            return Err(OpsError::InvalidInput("bridge diagonal already exists"));
        }
    }
    let report = BridgeReport {
        triangles_added: strip.len(),
    };
    tx.add_triangles(&strip)?;
    Ok(report)
}

fn loop_key(id: &LoopId) -> (u32, u32, u64) {
    (id.min_vertex, id.len, id.hash)
}

/// Finds a loop by id in a [`boundary_loops`] vec, like fill does.
fn resolve<'a>(loops: &'a [BoundaryLoop], id: &LoopId) -> Result<&'a BoundaryLoop> {
    let at = loops.partition_point(|l| l.id.min_vertex < id.min_vertex);
    loops
        .get(at)
        .filter(|l| l.id == *id)
        .ok_or(OpsError::StaleLoops)
}

/// The Newell vector of a loop: its direction is the loop's winding normal.
fn newell(mesh: &Mesh, verts: &[VertexId]) -> [f64; 3] {
    let mut n = [0.0; 3];
    for k in 0..verts.len() {
        let c = cross(
            point(mesh, verts[k]),
            point(mesh, verts[(k + 1) % verts.len()]),
        );
        n[0] += c[0];
        n[1] += c[1];
        n[2] += c[2];
    }
    n
}

/// Twice the area of a triangle over f64 positions.
fn tri_area2(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    let n = cross(sub(*b, *a), sub(*c, *a));
    dot(n, n).sqrt()
}

/// The minimum-area strip over all start offsets of the second loop. The
/// first loop is traversed backward, the second forward; see the
/// [`bridge_loops`] docs for why that combination is manifold.
fn best_strip(mesh: &Mesh, first: &[VertexId], second: &[VertexId]) -> Result<Vec<[VertexId; 3]>> {
    let n = first.len();
    let m = second.len();
    // X walks the first loop backward from its start vertex; Y walks the
    // second loop forward, doubled so every start offset is a plain slice.
    let mut xa: Vec<VertexId> = Vec::new();
    reserve(&mut xa, n + 1)?;
    for i in 0..=n {
        push(&mut xa, first[(n - i) % n])?;
    }
    let mut px: Vec<[f64; 3]> = Vec::new();
    reserve(&mut px, n + 1)?;
    for &v in &xa {
        push(&mut px, point(mesh, v))?;
    }
    let mut py: Vec<[f64; 3]> = Vec::new();
    reserve(&mut py, 2 * m + 1)?;
    for k in 0..=2 * m {
        push(&mut py, point(mesh, second[k % m]))?;
    }
    let cells = (n + 1) * (m + 1);
    let mut cost: Vec<f64> = Vec::new();
    reserve(&mut cost, cells)?;
    cost.extend(std::iter::repeat_n(f64::INFINITY, cells));
    let mut back: Vec<u8> = Vec::new();
    reserve(&mut back, cells)?;
    back.extend(std::iter::repeat_n(0, cells));
    let mut best = f64::INFINITY;
    let mut best_s = 0;
    for s in 0..m {
        let total = dp_fill(&px, &py[s..s + m + 1], &mut cost, &mut back);
        if total < best {
            best = total;
            best_s = s;
        }
    }
    dp_fill(&px, &py[best_s..best_s + m + 1], &mut cost, &mut back);
    // Backtrace the staircase. Rim states are forced along their only
    // incoming edge so garbage table entries cannot underflow the walk.
    let mut strip: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut strip, n + m)?;
    let (mut i, mut k) = (n, m);
    while i > 0 || k > 0 {
        let step_a = if k == 0 {
            true
        } else if i == 0 {
            false
        } else {
            back[i * (m + 1) + k] == 0
        };
        if step_a {
            push(&mut strip, [xa[i - 1], xa[i], second[(best_s + k) % m]])?;
            i -= 1;
        } else {
            push(
                &mut strip,
                [
                    xa[i],
                    second[(best_s + k) % m],
                    second[(best_s + k - 1) % m],
                ],
            )?;
            k -= 1;
        }
    }
    strip.reverse();
    Ok(strip)
}

/// Fills the correspondence table for one start offset and returns the
/// minimum total doubled area. Moving along X emits (x, x', y); moving
/// along Y emits (x, y', y); ties prefer X. Areas only, no ids here.
fn dp_fill(px: &[[f64; 3]], yb: &[[f64; 3]], cost: &mut [f64], back: &mut [u8]) -> f64 {
    let n = px.len() - 1;
    let m = yb.len() - 1;
    for c in cost.iter_mut() {
        *c = f64::INFINITY;
    }
    cost[0] = 0.0;
    for i in 0..=n {
        for k in 0..=m {
            if i == 0 && k == 0 {
                continue;
            }
            let mut best = f64::INFINITY;
            let mut dir = 0u8;
            if i > 0 {
                let c = cost[(i - 1) * (m + 1) + k] + tri_area2(&px[i - 1], &px[i], &yb[k]);
                if c <= best {
                    best = c;
                    dir = 0;
                }
            }
            if k > 0 {
                let c = cost[i * (m + 1) + k - 1] + tri_area2(&px[i], &yb[k], &yb[k - 1]);
                if c < best {
                    best = c;
                    dir = 1;
                }
            }
            cost[i * (m + 1) + k] = best;
            back[i * (m + 1) + k] = dir;
        }
    }
    cost[n * (m + 1) + m]
}

/// The strip's mixed first/second-loop edges: rim edges stay within one
/// loop, so any edge the strip shares with the mesh must be one of these.
/// Every strip vertex belongs to a loop, so "not first" means "second".
fn chords_of(strip: &[[VertexId; 3]], first: &[VertexId]) -> Result<Vec<(VertexId, VertexId)>> {
    let mut a_sorted: Vec<VertexId> = Vec::new();
    reserve(&mut a_sorted, first.len())?;
    a_sorted.extend_from_slice(first);
    a_sorted.sort_unstable();
    let mut chords: Vec<(VertexId, VertexId)> = Vec::new();
    reserve(&mut chords, 2 * strip.len())?;
    for t in strip {
        for (p, q) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let pa = a_sorted.binary_search(&p).is_ok();
            let qa = a_sorted.binary_search(&q).is_ok();
            if pa != qa {
                push(&mut chords, (p.min(q), p.max(q)))?;
            }
        }
    }
    Ok(chords)
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::{Mesh, Transaction};

    use super::*;
    use crate::test_util::{edit, mesh_of, parts, sweep};
    use crate::topology::EdgeMap;
    use crate::{OpsError, analyze, boundary_loops};

    /// Quad A at z=0 facing -z, quad B at z=1 facing +z: both rims face
    /// away from the gap, so an outward tube is manifold at both rims.
    fn two_quads() -> Mesh {
        mesh_of(
            &[
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            ],
            &[[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]],
        )
    }

    /// Triangle at z=0 facing -z, quad at z=1 facing +z.
    fn tri_square() -> Mesh {
        mesh_of(
            &[
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            ],
            &[[0, 2, 1], [3, 4, 5], [3, 5, 6]],
        )
    }

    fn loop_of(mesh: &Mesh, min_vertex: u32) -> crate::LoopId {
        boundary_loops(mesh)
            .unwrap()
            .iter()
            .find(|l| l.id.min_vertex == min_vertex)
            .unwrap()
            .id
    }

    /// Every triangle added past `before` faces away from the tube axis.
    fn assert_outward(mesh: &Mesh, before: usize) {
        let (pos, tris) = parts(mesh);
        let p = |v: u32| -> [f64; 3] {
            let i = 3 * v as usize;
            [
                f64::from(pos[i]),
                f64::from(pos[i + 1]),
                f64::from(pos[i + 2]),
            ]
        };
        // Axis through the loop centroids: A is verts 0..4, B the rest.
        let mut ca = [0.0; 3];
        let mut cb = [0.0; 3];
        for v in 0..4u32 {
            let q = p(v);
            for k in 0..3 {
                ca[k] += q[k] / 4.0;
            }
        }
        for v in 4..8u32 {
            let q = p(v);
            for k in 0..3 {
                cb[k] += q[k] / 4.0;
            }
        }
        assert!(tris.len() > before);
        for t in &tris[before..] {
            let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
            let n = [
                (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
                (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
                (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
            ];
            let m = [
                (a[0] + b[0] + c[0]) / 3.0 - (ca[0] + cb[0]) / 2.0,
                (a[1] + b[1] + c[1]) / 3.0 - (ca[1] + cb[1]) / 2.0,
                0.0,
            ];
            let d = n[0] * m[0] + n[1] * m[1] + n[2] * m[2];
            assert!(d > 0.0, "inward or degenerate: {t:?} d={d}");
        }
    }

    #[test]
    fn bridge_zips_two_square_loops() {
        let mut mesh = two_quads();
        let before = parts(&mesh).1.len();
        let a = loop_of(&mesh, 0);
        let b = loop_of(&mesh, 4);
        let options = BridgeOptions::builder(a, b).build();
        let (report, _) = edit(&mut mesh, |tx| bridge_loops(tx, &options));
        assert_eq!(report.triangles_added, 8);
        assert_outward(&mesh, before);
        // Every rim edge is interior now, and so is every new diagonal.
        let edges = EdgeMap::build(&mesh).unwrap();
        let (_, tris) = parts(&mesh);
        for t in &tris[before..] {
            for (x, y) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                assert_eq!(edges.triangles(x, y).len(), 2, "edge ({x}, {y})");
            }
        }
        let conflicts = analyze(&mesh, &Default::default())
            .unwrap()
            .orientation_conflicts;
        assert_eq!(conflicts, 0);
    }

    #[test]
    fn bridge_handles_mismatched_loop_sizes() {
        let mut mesh = tri_square();
        let before = parts(&mesh).1.len();
        let a = loop_of(&mesh, 0);
        let b = loop_of(&mesh, 3);
        let options = BridgeOptions::builder(a, b).build();
        let (report, _) = edit(&mut mesh, |tx| bridge_loops(tx, &options));
        assert_eq!(report.triangles_added, 7);
        let edges = EdgeMap::build(&mesh).unwrap();
        let (_, tris) = parts(&mesh);
        for t in &tris[before..] {
            for (x, y) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                assert_eq!(edges.triangles(x, y).len(), 2, "edge ({x}, {y})");
            }
        }
        let conflicts = analyze(&mesh, &Default::default())
            .unwrap()
            .orientation_conflicts;
        assert_eq!(conflicts, 0);
    }

    #[test]
    fn bridge_rejects_same_winding() {
        // Both quads face +z (both rims face into the gap on one side):
        // no outward tube can be manifold at both rims.
        let mut mesh = mesh_of(
            &[
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            ],
            &[[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
        );
        let before = parts(&mesh);
        let a = loop_of(&mesh, 0);
        let b = loop_of(&mesh, 4);
        let options = BridgeOptions::builder(a, b).build();
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            bridge_loops(&mut tx, &options),
            Err(OpsError::InvalidInput(_))
        ));
        drop(tx);
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn bridge_rejects_bad_selection() {
        let mut mesh = two_quads();
        let before = parts(&mesh);
        let a = loop_of(&mesh, 0);
        let b = loop_of(&mesh, 4);
        // Same loop twice.
        let options = BridgeOptions::builder(a, a).build();
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            bridge_loops(&mut tx, &options),
            Err(OpsError::InvalidInput(_))
        ));
        drop(tx);
        // Unknown loop id.
        let fake = crate::LoopId {
            min_vertex: 99,
            len: 4,
            hash: 7,
        };
        let options = BridgeOptions::builder(a, fake).build();
        let mut tx = mesh.begin_edit();
        assert_eq!(bridge_loops(&mut tx, &options), Err(OpsError::StaleLoops));
        drop(tx);
        // Over the per-loop bound.
        let options = BridgeOptions::builder(a, b).max_edges(3).build();
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            bridge_loops(&mut tx, &options),
            Err(OpsError::InvalidInput(_))
        ));
        drop(tx);
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn bridge_sweep_keeps_atomicity() {
        let mesh = two_quads();
        let a = loop_of(&mesh, 0);
        let b = loop_of(&mesh, 4);
        let options = BridgeOptions::builder(a, b).build();
        let n = sweep(mesh, |tx: &mut Transaction| {
            bridge_loops(tx, &options)?;
            Ok(())
        });
        assert!(n > 0);
    }
}
