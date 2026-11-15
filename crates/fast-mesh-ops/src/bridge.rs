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

/// Options for [`bridge_partial_arcs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PartialBridgeOptions {
    /// Boundary loop index (in [`boundary_loops`] order) containing the first arc.
    pub loop_a_index: usize,
    /// Inclusive vertex-index range `(start, end)` along `loop_a_index` (`start != end`;
    /// wraps modulo loop length when `start > end`).
    pub arc_a_range: (usize, usize),
    /// Boundary loop index (in [`boundary_loops`] order) containing the second arc.
    /// May equal `loop_a_index` to split a single hole loop into smaller loops.
    pub loop_b_index: usize,
    /// Inclusive vertex-index range `(start, end)` along `loop_b_index`.
    pub arc_b_range: (usize, usize),
    /// Cyclic vertex shift applied along `loop_b_index` to `arc_b_range`.
    pub twist_offset: isize,
    /// Optional number of interior span subdivisions across the bridge width
    /// (`None` or `Some(0)` adds no interior vertices).
    pub subdivisions: Option<usize>,
}

impl Default for PartialBridgeOptions {
    fn default() -> Self {
        Self {
            loop_a_index: 0,
            arc_a_range: (0, 1),
            loop_b_index: 0,
            arc_b_range: (2, 3),
            twist_offset: 0,
            subdivisions: None,
        }
    }
}

impl PartialBridgeOptions {
    /// Starts a builder targeting `(loop_a_index, arc_a_range)` and
    /// `(loop_b_index, arc_b_range)`.
    pub fn builder(
        loop_a_index: usize,
        arc_a_range: (usize, usize),
        loop_b_index: usize,
        arc_b_range: (usize, usize),
    ) -> PartialBridgeOptionsBuilder {
        PartialBridgeOptionsBuilder {
            options: Self {
                loop_a_index,
                arc_a_range,
                loop_b_index,
                arc_b_range,
                twist_offset: 0,
                subdivisions: None,
            },
        }
    }
}

/// Builder for [`PartialBridgeOptions`].
#[derive(Debug, Clone, Copy, Default)]
pub struct PartialBridgeOptionsBuilder {
    options: PartialBridgeOptions,
}

impl PartialBridgeOptionsBuilder {
    /// Sets `loop_a_index`.
    pub fn loop_a_index(mut self, index: usize) -> Self {
        self.options.loop_a_index = index;
        self
    }

    /// Sets `arc_a_range`.
    pub fn arc_a_range(mut self, range: (usize, usize)) -> Self {
        self.options.arc_a_range = range;
        self
    }

    /// Sets `loop_b_index`.
    pub fn loop_b_index(mut self, index: usize) -> Self {
        self.options.loop_b_index = index;
        self
    }

    /// Sets `arc_b_range`.
    pub fn arc_b_range(mut self, range: (usize, usize)) -> Self {
        self.options.arc_b_range = range;
        self
    }

    /// Sets `twist_offset`.
    pub fn twist_offset(mut self, twist_offset: isize) -> Self {
        self.options.twist_offset = twist_offset;
        self
    }

    /// Sets `subdivisions`.
    pub fn subdivisions(mut self, subdivisions: Option<usize>) -> Self {
        self.options.subdivisions = subdivisions;
        self
    }

    /// Builds [`PartialBridgeOptions`]. Validation runs inside [`bridge_partial_arcs`].
    pub fn build(self) -> PartialBridgeOptions {
        self.options
    }
}

/// Summary of a [`bridge_partial_arcs`] operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct BridgeStats {
    /// Triangles added to stitch the two arcs.
    pub triangles_added: usize,
    /// Interior bridge vertices added (non-zero when `subdivisions` is set).
    pub vertices_added: usize,
    /// Number of boundary loops on the mesh after bridging.
    pub resulting_loops: usize,
    /// Vertex count on `arc_a`.
    pub arc_a_vertices: usize,
    /// Vertex count on `arc_b`.
    pub arc_b_vertices: usize,
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

/// Maximum interior subdivisions permitted along a partial bridge.
const MAX_PARTIAL_SUBDIVISIONS: usize = 64;

/// Bridges two boundary sub-arcs with a manifold, minimum-area triangle strip,
/// splitting a single loop into smaller loops or partially connecting two loops.
///
/// The sub-arcs may lie on the same boundary loop (splitting complex saddle
/// holes into two simpler closed loops) or on distinct loops. `twist_offset`
/// cyclically rotates `arc_b_range` along `loop_b_index`, and optional
/// `subdivisions` inserts linear barycentric vertex rows across the bridge with
/// smooth UV interpolation via [`Transaction::add_vertex_with_parents`].
///
/// All geometry, chord, and allocation checks complete before `tx` is mutated.
///
/// # Errors
/// - [`OpsError::InvalidInput`] for out-of-bounds loop indices, invalid or
///   overlapping arc ranges, non-finite coordinates, degenerate triangles, or
///   existing mesh edge collisions.
/// - [`OpsError::Store`] on store / memory exhaustion.
pub fn bridge_partial_arcs(
    tx: &mut Transaction,
    options: &PartialBridgeOptions,
) -> Result<BridgeStats> {
    let subdivisions = options.subdivisions.unwrap_or(0);
    if subdivisions > MAX_PARTIAL_SUBDIVISIONS {
        return Err(OpsError::InvalidInput(
            "bridge subdivisions exceeds supported limit",
        ));
    }

    let mesh = tx.mesh();
    let loops = boundary_loops(mesh)?;
    let la = loops
        .get(options.loop_a_index)
        .ok_or(OpsError::InvalidInput("loop_a_index out of bounds"))?;
    let lb = loops
        .get(options.loop_b_index)
        .ok_or(OpsError::InvalidInput("loop_b_index out of bounds"))?;
    for l in [la, lb] {
        if !l.closed || !l.clean || l.vertices.len() < 3 {
            return Err(OpsError::InvalidInput(
                "bridge loops must be closed and clean",
            ));
        }
    }

    let (arc_a, a_start, a_end) = extract_arc(&la.vertices, options.arc_a_range, 0)?;
    let (arc_b_fwd, b_start, b_end) =
        extract_arc(&lb.vertices, options.arc_b_range, options.twist_offset)?;

    // Ensure the two arcs share no vertices so triangles never collapse or pinch.
    let mut a_sorted: Vec<VertexId> = Vec::new();
    reserve(&mut a_sorted, arc_a.len())?;
    a_sorted.extend_from_slice(&arc_a);
    a_sorted.sort_unstable();
    for &v in &arc_b_fwd {
        if a_sorted.binary_search(&v).is_ok() {
            return Err(OpsError::InvalidInput(
                "partial bridge arcs must not share vertices",
            ));
        }
    }

    for &v in arc_a.iter().chain(&arc_b_fwd) {
        if !point(mesh, v).iter().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput(
                "bridge arc positions must be finite",
            ));
        }
    }

    // Walk arc_a forward and arc_b backward so triangles wind opposite to
    // both rim loops (agreeing with adjacent mesh faces on both sides).
    let k = arc_a.len();
    let m = arc_b_fwd.len();
    let mut arc_b_rev: Vec<VertexId> = Vec::new();
    reserve(&mut arc_b_rev, m)?;
    arc_b_rev.extend(arc_b_fwd.iter().copied().rev());

    let mut pu: Vec<[f64; 3]> = Vec::new();
    reserve(&mut pu, k)?;
    for &v in &arc_a {
        pu.push(point(mesh, v));
    }
    let mut pv: Vec<[f64; 3]> = Vec::new();
    reserve(&mut pv, m)?;
    for &v in &arc_b_rev {
        pv.push(point(mesh, v));
    }

    let cells = k
        .checked_mul(m)
        .ok_or(OpsError::InvalidInput("bridge arc DP table overflow"))?;
    let mut cost: Vec<f64> = Vec::new();
    reserve(&mut cost, cells)?;
    cost.extend(std::iter::repeat_n(f64::INFINITY, cells));
    let mut back: Vec<u8> = Vec::new();
    reserve(&mut back, cells)?;
    back.extend(std::iter::repeat_n(0, cells));

    let total = dp_fill(&pu, &pv, &mut cost, &mut back);
    if !total.is_finite() {
        return Err(OpsError::InvalidInput("bridge triangulation cost invalid"));
    }

    // Recover staircase steps (0 = step along u, 1 = step along v) from (0, 0) to (k-1, m-1).
    let steps = (k - 1) + (m - 1);
    let mut path_steps: Vec<u8> = Vec::new();
    reserve(&mut path_steps, steps)?;
    let (mut iu, mut jv) = (k - 1, m - 1);
    while iu > 0 || jv > 0 {
        let step_u = if jv == 0 {
            true
        } else if iu == 0 {
            false
        } else {
            back[iu * m + jv] == 0
        };
        if step_u {
            path_steps.push(0);
            iu -= 1;
        } else {
            path_steps.push(1);
            jv -= 1;
        }
    }
    path_steps.reverse();

    // Record the (u_idx, v_idx) rung vertices along the staircase and validate every triangle/chord.
    let mut rungs: Vec<(usize, usize)> = Vec::new();
    reserve(&mut rungs, steps + 1)?;
    let (mut ci, mut cj) = (0usize, 0usize);
    rungs.push((ci, cj));
    for &s in &path_steps {
        let tri = if s == 0 {
            ci += 1;
            [arc_a[ci], arc_a[ci - 1], arc_b_rev[cj]]
        } else {
            cj += 1;
            [arc_a[ci], arc_b_rev[cj - 1], arc_b_rev[cj]]
        };
        let area2 = tri_area2(
            &point(mesh, tri[0]),
            &point(mesh, tri[1]),
            &point(mesh, tri[2]),
        );
        if !area2.is_finite() || area2 == 0.0 {
            return Err(OpsError::InvalidInput("bridge triangulation is degenerate"));
        }
        rungs.push((ci, cj));
    }

    let same_loop = options.loop_a_index == options.loop_b_index;
    let n_a = la.vertices.len();
    let gap1_edges = if same_loop {
        (b_start + n_a - a_end) % n_a
    } else {
        0
    };
    let gap2_edges = if same_loop {
        (a_start + n_a - b_end) % n_a
    } else {
        0
    };

    let edges = EdgeMap::build(mesh)?;
    for (r_idx, &(ui, vj)) in rungs.iter().enumerate() {
        let (p, q) = (arc_a[ui], arc_b_rev[vj]);
        let incident = edges.triangles(p, q);
        if r_idx > 0 && r_idx < steps {
            if !incident.is_empty() {
                return Err(OpsError::InvalidInput("bridge diagonal already exists"));
            }
        } else if !incident.is_empty() {
            let closes_single_gap = same_loop
                && incident.len() == 1
                && ((r_idx == 0 && gap2_edges == 1) || (r_idx == steps && gap1_edges == 1));
            if !closes_single_gap {
                return Err(OpsError::InvalidInput("bridge diagonal already exists"));
            }
            if subdivisions > 0 {
                return Err(OpsError::InvalidInput(
                    "cannot subdivide across an existing boundary edge",
                ));
            }
        }
    }

    let resulting_loops = if same_loop {
        loops.len() - 1 + usize::from(gap1_edges >= 2) + usize::from(gap2_edges >= 2)
    } else {
        loops.len() - 1
    };

    // Reserve all output memory BEFORE mutating tx so OOM leaves tx untouched.
    let tris_per_step = 2 * subdivisions + 1;
    let total_tris = steps
        .checked_mul(tris_per_step)
        .ok_or(OpsError::InvalidInput("bridge triangle count overflow"))?;
    let total_new_verts = (steps + 1)
        .checked_mul(subdivisions)
        .ok_or(OpsError::InvalidInput("bridge vertex count overflow"))?;

    type BridgeVertexSpec = ([f32; 3], [(VertexId, f32); 2]);
    let mut vert_specs: Vec<BridgeVertexSpec> = Vec::new();
    reserve(&mut vert_specs, total_new_verts)?;
    if subdivisions > 0 {
        let rows = (subdivisions + 1) as f64;
        for &(ui, vj) in &rungs {
            let (ua, vb) = (arc_a[ui], arc_b_rev[vj]);
            let (pa, pb) = (point(mesh, ua), point(mesh, vb));
            for row in 1..=subdivisions {
                let t = row as f64 / rows;
                let omt = 1.0 - t;
                let pos = [
                    (omt * pa[0] + t * pb[0]) as f32,
                    (omt * pa[1] + t * pb[1]) as f32,
                    (omt * pa[2] + t * pb[2]) as f32,
                ];
                vert_specs.push((pos, [(ua, omt as f32), (vb, t as f32)]));
            }
        }
    }

    let mut new_verts: Vec<VertexId> = Vec::new();
    reserve(&mut new_verts, total_new_verts)?;
    let mut out_tris: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut out_tris, total_tris)?;

    // All fallible fast-mesh-ops reservations are complete; now commit vertices and triangles.
    for (pos, parents) in &vert_specs {
        let id = tx.add_vertex_with_parents(*pos, parents)?;
        new_verts.push(id);
    }

    let rung_vertex = |r_idx: usize, row: usize| -> VertexId {
        if row == 0 {
            arc_a[rungs[r_idx].0]
        } else if row == subdivisions + 1 {
            arc_b_rev[rungs[r_idx].1]
        } else {
            new_verts[r_idx * subdivisions + (row - 1)]
        }
    };

    for (r, &s) in path_steps.iter().enumerate() {
        if subdivisions == 0 {
            let tri = if s == 0 {
                [
                    arc_a[rungs[r + 1].0],
                    arc_a[rungs[r].0],
                    arc_b_rev[rungs[r].1],
                ]
            } else {
                [
                    arc_a[rungs[r].0],
                    arc_b_rev[rungs[r].1],
                    arc_b_rev[rungs[r + 1].1],
                ]
            };
            out_tris.push(tri);
            continue;
        }
        if s == 0 {
            for row in 0..subdivisions {
                let (p0, p1) = (rung_vertex(r, row), rung_vertex(r, row + 1));
                let (q0, q1) = (rung_vertex(r + 1, row), rung_vertex(r + 1, row + 1));
                out_tris.push([q0, p0, p1]);
                out_tris.push([q0, p1, q1]);
            }
            let ps = rung_vertex(r, subdivisions);
            let qs = rung_vertex(r + 1, subdivisions);
            let tip = rung_vertex(r, subdivisions + 1);
            out_tris.push([qs, ps, tip]);
        } else {
            let u_tip = rung_vertex(r, 0);
            let p1 = rung_vertex(r, 1);
            let q1 = rung_vertex(r + 1, 1);
            out_tris.push([u_tip, p1, q1]);
            for row in 1..=subdivisions {
                let (p0, p1) = (rung_vertex(r, row), rung_vertex(r, row + 1));
                let (q0, q1) = (rung_vertex(r + 1, row), rung_vertex(r + 1, row + 1));
                out_tris.push([q0, p0, p1]);
                out_tris.push([q0, p1, q1]);
            }
        }
    }

    tx.add_triangles(&out_tris)?;
    Ok(BridgeStats {
        triangles_added: out_tris.len(),
        vertices_added: new_verts.len(),
        resulting_loops,
        arc_a_vertices: k,
        arc_b_vertices: m,
    })
}

/// Extracts an inclusive vertex arc from `loop_verts` after shifting `(start, end)`
/// by `twist_offset` modulo `loop_verts.len()`, returning `(arc, shifted_start, shifted_end)`.
fn extract_arc(
    loop_verts: &[VertexId],
    range: (usize, usize),
    twist_offset: isize,
) -> Result<(Vec<VertexId>, usize, usize)> {
    let n = loop_verts.len();
    let (raw_s, raw_e) = range;
    if raw_s >= n || raw_e >= n || raw_s == raw_e {
        return Err(OpsError::InvalidInput("invalid bridge arc range"));
    }
    let shift =
        |idx: usize| -> usize { ((idx as isize + twist_offset).rem_euclid(n as isize)) as usize };
    let start = shift(raw_s);
    let end = shift(raw_e);
    let steps = (end + n - start) % n;
    let mut arc: Vec<VertexId> = Vec::new();
    reserve(&mut arc, steps + 1)?;
    for step in 0..=steps {
        arc.push(loop_verts[(start + step) % n]);
    }
    Ok((arc, start, end))
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

    #[test]
    fn partial_bridge_splits_hole_into_two_independent_loops() {
        use crate::test_util::grid_without;
        // 8x8 grid missing a 4x4 interior hole (2 boundary loops: outer grid border + 16-vertex hole).
        let hole_cells: Vec<(u32, u32)> =
            (2..6).flat_map(|y| (2..6).map(move |x| (x, y))).collect();
        let mut mesh = grid_without(8, &hole_cells);
        let loops = boundary_loops(&mesh).unwrap();
        assert_eq!(loops.len(), 2);
        let hole_idx = loops
            .iter()
            .position(|l| l.vertices.len() == 16)
            .expect("16-vertex hole");
        let n = loops[hole_idx].vertices.len();

        // Take two separated 3-vertex arcs on opposite sides of the 16-vertex hole.
        let options = PartialBridgeOptions::builder(hole_idx, (1, 3), hole_idx, (9, 11)).build();
        let (stats, _) = edit(&mut mesh, |tx| bridge_partial_arcs(tx, &options));
        assert_eq!(stats.arc_a_vertices, 3);
        assert_eq!(stats.arc_b_vertices, 3);
        assert_eq!(stats.vertices_added, 0);
        assert_eq!(stats.triangles_added, 4);
        assert_eq!(stats.resulting_loops, 3);

        let after_loops = boundary_loops(&mesh).unwrap();
        assert_eq!(after_loops.len(), 3);
        assert!(
            after_loops
                .iter()
                .all(|l| l.closed && l.clean && (l.vertices.len() < n || l.vertices.len() == 32))
        );
        let edges = EdgeMap::build(&mesh).unwrap();
        for (e, tris) in edges.edges() {
            assert!(tris.len() <= 2, "non-manifold edge after split: {e:?}");
        }
    }

    #[test]
    fn partial_bridge_cross_loops_with_twist_subdivisions_and_uvs() {
        // Two detached 2x1 strips in z = 0 facing +Z separated along Y, with UV channels.
        let pos = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0,
            0.0, 0.0, 3.0, 0.0, 1.0, 3.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 0.0, 1.0, 4.0, 0.0, 2.0,
            4.0, 0.0,
        ];
        let tris = vec![
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [6, 7, 10],
            [6, 10, 9],
            [7, 8, 11],
            [7, 11, 10],
        ];
        let mut mesh = mesh_of(&pos, &tris);
        let uvs: Vec<f32> = (0..12)
            .flat_map(|i| [pos[3 * i] * 0.5, pos[3 * i + 1] * 0.25])
            .collect();
        mesh.add_uvs(&uvs).unwrap();

        let loops = boundary_loops(&mesh).unwrap();
        assert_eq!(loops.len(), 2);
        let find_idx =
            |loop_verts: &[VertexId], v: VertexId| loop_verts.iter().position(|&x| x == v).unwrap();
        let l0 = loops.iter().position(|l| l.vertices.contains(&4)).unwrap();
        let l1 = 1 - l0;
        let a_s = find_idx(&loops[l0].vertices, 5);
        let a_e = find_idx(&loops[l0].vertices, 3);
        let b_s_target = find_idx(&loops[l1].vertices, 6);
        let b_e_target = find_idx(&loops[l1].vertices, 8);
        let len1 = loops[l1].vertices.len();
        // Shift raw range by -2 and cancel with twist_offset = +2.
        let b_s_raw = (b_s_target + len1 - 2) % len1;
        let b_e_raw = (b_e_target + len1 - 2) % len1;

        let options = PartialBridgeOptions::builder(l0, (a_s, a_e), l1, (b_s_raw, b_e_raw))
            .twist_offset(2)
            .subdivisions(Some(2))
            .build();
        let (stats, _) = edit(&mut mesh, |tx| bridge_partial_arcs(tx, &options));
        assert_eq!(stats.arc_a_vertices, 3);
        assert_eq!(stats.arc_b_vertices, 3);
        assert_eq!(stats.vertices_added, 5 * 2);
        assert_eq!(stats.triangles_added, 4 * 5);
        assert_eq!(stats.resulting_loops, 1);

        let uv_after = mesh.uvs().unwrap();
        for vi in 12..mesh.vertex_slots() {
            let p = mesh.position(vi as VertexId).unwrap();
            let u = uv_after[2 * vi];
            let v = uv_after[2 * vi + 1];
            assert!((u - p[0] * 0.5).abs() < 1e-5, "uv mismatch x: {u} vs {p:?}");
            assert!(
                (v - p[1] * 0.25).abs() < 1e-5,
                "uv mismatch y: {v} vs {p:?}"
            );
        }
        let edges = EdgeMap::build(&mesh).unwrap();
        for (_e, tris) in edges.edges() {
            assert!(tris.len() <= 2);
        }
    }

    #[test]
    fn partial_bridge_rejects_invalid_inputs_and_sweep_is_atomic() {
        use crate::test_util::grid_without;
        let hole_cells: Vec<(u32, u32)> =
            (2..6).flat_map(|y| (2..6).map(move |x| (x, y))).collect();
        let mut mesh = grid_without(8, &hole_cells);
        let before = parts(&mesh);
        let loops = boundary_loops(&mesh).unwrap();
        let hole_idx = loops.iter().position(|l| l.vertices.len() == 16).unwrap();

        let bad_cases = [
            PartialBridgeOptions::builder(99, (0, 2), hole_idx, (8, 10)).build(),
            PartialBridgeOptions::builder(hole_idx, (3, 3), hole_idx, (8, 10)).build(),
            PartialBridgeOptions::builder(hole_idx, (0, 5), hole_idx, (4, 8)).build(),
            PartialBridgeOptions::builder(hole_idx, (0, 2), hole_idx, (3, 5))
                .subdivisions(Some(1))
                .build(),
        ];
        for opts in &bad_cases {
            let mut tx = mesh.begin_edit();
            assert!(
                matches!(
                    bridge_partial_arcs(&mut tx, opts),
                    Err(OpsError::InvalidInput(_))
                ),
                "expected InvalidInput for {opts:?}"
            );
            drop(tx);
        }
        assert_eq!(parts(&mesh), before);

        let valid = PartialBridgeOptions::builder(hole_idx, (1, 3), hole_idx, (9, 11))
            .subdivisions(Some(2))
            .build();
        let points = sweep(mesh, |tx| {
            bridge_partial_arcs(tx, &valid)?;
            Ok(())
        });
        assert!(points > 0);
    }
}
