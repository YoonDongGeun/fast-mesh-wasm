//! Adaptive splitting: triangles a deformation stretched split on their own,
//! and nothing else changes.
//!
//! [`capture_rest_sizes`] records each vertex's *rest size*, the longest edge
//! on it, in the [`REST_SIZE_CHANNEL`] channel, before the deformation starts.
//! [`split_stretched`] then splits an edge only when it has grown past
//! [`StretchOptions::limit`] times the smaller rest size of its two endpoints.
//!
//! An edge's rest length is at most either endpoint's rest size, so an edge
//! that did not stretch can never reach the limit, however long it is next to
//! the rest of the mesh. A mesh with fine and coarse areas keeps both: each
//! area is measured against itself, not against a mesh-wide mean.
//!
//! # What changes
//!
//! - Existing vertices keep their ids and positions, so anything that stores
//!   vertex ids stays valid.
//! - The triangles on a split edge are bisected. A triangle no split edge
//!   touches keeps its slot and its corners: the only unstretched triangles
//!   that change are neighbours of a stretched edge, to stay conforming.
//! - Nothing collapses and nothing flips.
//!
//! The longest edge splits first, which keeps the bisected triangles from
//! degrading into slivers.
//!
//! # Where new vertices go
//!
//! Existing vertices cannot move, which rules out Loop's edge rule: its
//! `3/8 (a + b) + 1/8 (c + d)` relies on the vertex rule moving `a` and `b`
//! too. Alone it pulls a new vertex below the midpoint on a convex surface and
//! measures no better than a midpoint against the true normal. A new vertex
//! takes the middle of the cubic Hermite curve between the edge's endpoints
//! and their normals instead, which needs only the edge
//! and is exact to second order on a smooth surface:
//! `(a + b) / 2 - (((b - a) . na) na + ((a - b) . nb) nb) / 8`. An endpoint's
//! normal averages only the triangles around it that face like the edge's own
//! (within 60 degrees), so a flat face stays flat next to a hard edge.
//!
//! An exact midpoint is used on a boundary or non-manifold edge, on a sharp
//! edge (owners more than 60 degrees apart), and wherever the curved point
//! would fold a child triangle. The boundary rule is what keeps a UV seam
//! closed: its two sides are separate vertices at one position, and both must
//! compute the same new point.
//!
//! # Seams
//!
//! Two vertices at exactly one position take the larger of their rest sizes,
//! so both sides of a seam use one limit and split together. Boundary edges
//! also split before any interior edge and are exempt from
//! [`StretchOptions::max_splits`], so a seam cannot open at the cap.
//!
//! # Cost
//!
//! One pass tests the vertices against the region and then touches only the
//! edges around vertices inside it, using the mesh's vertex index
//! ([`Mesh::ensure_incidence`]). It never copies or sorts the whole mesh.
//! Without a cached index (inside a transaction that has already changed
//! topology, for example) it builds a private one first, which costs a pass
//! over every triangle.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use fast_mesh_store::{
    ChannelDesc, Incidence, Mesh, StoreError, Transaction, TriangleId, VertexId,
};

use crate::remesh::{
    NewVert, Region, apex_of, blend_parents, carries, cross3, degenerate, dot3, f64v, midpoint,
    runs_directed, sub3,
};
use crate::{OpsError, Result, allocation_allowed, filled, push, reserve};

/// The source channel [`capture_rest_sizes`] writes: one value per vertex
/// slot, interpolated for new vertices. Zero means no rest size is known.
pub const REST_SIZE_CHANNEL: &str = "rest_edge_size";
/// Default for [`StretchOptions::limit`].
pub const DEFAULT_STRETCH_LIMIT: f32 = 4.0 / 3.0;
/// Default for [`StretchOptions::max_splits`].
pub const DEFAULT_MAX_SPLITS: u32 = 1 << 20;
/// Cosine of the angle (60 degrees) past which two faces count as separated
/// by a sharp edge.
const SHARP_COS: f64 = 0.5;

/// Options for [`split_stretched`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StretchOptions {
    /// Vertices inside this region have their edges checked. Edges with no
    /// endpoint inside are left alone.
    pub region: Region,
    /// An edge splits past this multiple of the smaller rest size of its
    /// endpoints. Must be finite and above 1: at 1 or below, an edge that did
    /// not stretch could split.
    pub limit: f32,
    /// Most interior splits in one call; the longest go first and
    /// [`StretchReport::remaining`] counts the rest, for a later call.
    /// Boundary edges do not count, see the module docs.
    pub max_splits: u32,
    /// New interior vertices follow the surface's curvature. `false` puts
    /// every new vertex at its edge's midpoint.
    pub smooth: bool,
}

impl Default for StretchOptions {
    fn default() -> Self {
        StretchOptions {
            region: Region::default(),
            limit: DEFAULT_STRETCH_LIMIT,
            max_splits: DEFAULT_MAX_SPLITS,
            smooth: true,
        }
    }
}

/// Outcome of [`split_stretched`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StretchReport {
    /// Edge splits performed.
    pub splits: usize,
    /// Edges still past the limit because [`StretchOptions::max_splits`] ran
    /// out.
    pub remaining: usize,
    /// `[new, a, b]` per split, in order: the new vertex and its edge's
    /// endpoints, which may be new vertices from earlier in this list. For a
    /// caller that holds per-vertex state of its own and has to carry it over.
    pub vertices: Vec<[VertexId; 3]>,
}

/// Records every vertex's rest size for [`split_stretched`]: the longest edge
/// on it, or on any vertex at exactly its position. Replaces an earlier
/// capture, and builds the mesh's vertex index if needed.
///
/// Call it before the deformation starts, and keep it across strokes that
/// should share one density: capturing again accepts what earlier strokes
/// stretched as the new rest shape, which [`ensure_rest_sizes`] avoids. This is
/// not an edit: no undo step, no journal entry, and the mesh's revision does not
/// change. It costs one pass over the triangles and a sort of the vertices.
///
/// The sizes last as long as they are true: a topology edit other than
/// [`split_stretched`] drops them (undoing that edit brings them back), and so
/// does undoing or redoing an edit from before the capture.
/// [`split_stretched`] then fails until they are taken again. Moving vertices
/// does not drop them, whichever tool does it: that is the stretch they measure.
///
/// # Errors
/// [`OpsError::Store`] when memory runs out; the mesh keeps any earlier
/// capture unless the failure comes after it is replaced.
pub fn capture_rest_sizes(mesh: &mut Mesh) -> Result<()> {
    let positions = mesh.positions();
    let mut sizes: Vec<f32> = filled(mesh.vertex_slots(), 0.0)?;
    for t in 0..mesh.triangle_slots() as u32 {
        let Some(c) = mesh.triangle(t) else { continue };
        if degenerate(&c) {
            continue;
        }
        for k in 0..3 {
            let (a, b) = (c[k], c[(k + 1) % 3]);
            let len = length(&positions[a as usize], &positions[b as usize]) as f32;
            if len.is_finite() {
                sizes[a as usize] = sizes[a as usize].max(len);
                sizes[b as usize] = sizes[b as usize].max(len);
            }
        }
    }
    let mut order: Vec<u32> = Vec::new();
    reserve(&mut order, sizes.len())?;
    order.extend((0..sizes.len() as u32).filter(|&v| sizes[v as usize] > 0.0));
    // `+ 0.0` folds -0 onto 0: seams are recognised by exact coincidence.
    let key = |v: u32| positions[v as usize].map(|x| (x + 0.0).to_bits());
    order.sort_unstable_by_key(|&v| key(v));
    let mut start = 0;
    while start < order.len() {
        let group = key(order[start]);
        let mut end = start + 1;
        let mut size = sizes[order[start] as usize];
        while end < order.len() && key(order[end]) == group {
            size = size.max(sizes[order[end] as usize]);
            end += 1;
        }
        for &v in &order[start..end] {
            sizes[v as usize] = size;
        }
        start = end;
    }
    if mesh.channel(REST_SIZE_CHANNEL).is_some() {
        mesh.remove_channel(REST_SIZE_CHANNEL)?;
    }
    mesh.add_channel_with_values(
        ChannelDesc::source(REST_SIZE_CHANNEL, 1).bound_to_topology(),
        &sizes,
    )?;
    mesh.ensure_incidence()?;
    Ok(())
}

/// [`capture_rest_sizes`] unless the mesh holds valid rest sizes. Returns
/// whether it captured.
///
/// # Errors
/// As for [`capture_rest_sizes`].
pub fn ensure_rest_sizes(mesh: &mut Mesh) -> Result<bool> {
    if mesh.channel(REST_SIZE_CHANNEL).is_some() {
        return Ok(false);
    }
    capture_rest_sizes(mesh)?;
    Ok(true)
}

/// Splits the edges under `options.region` that have stretched past
/// [`StretchOptions::limit`], until none is left or
/// [`StretchOptions::max_splits`] runs out. See the [module docs](self) for
/// what changes and where new vertices go. New vertices interpolate source
/// channels, so UVs follow. The plan is computed first and committed through
/// `tx` at the end: either every split lands or nothing does.
///
/// # Errors
/// [`OpsError::InvalidInput`] for an invalid region or limit, or when the mesh
/// holds no valid rest sizes: see [`capture_rest_sizes`].
pub fn split_stretched(tx: &mut Transaction, options: &StretchOptions) -> Result<StretchReport> {
    let region = &options.region;
    if !region.radius.is_finite() || region.radius <= 0.0 {
        return Err(OpsError::InvalidInput(
            "stretch region radius must be positive and finite",
        ));
    }
    if region.center.iter().any(|c| !c.is_finite()) {
        return Err(OpsError::InvalidInput(
            "stretch region center must be finite",
        ));
    }
    if !options.limit.is_finite() || options.limit <= 1.0 {
        return Err(OpsError::InvalidInput(
            "stretch limit must be finite and above 1",
        ));
    }
    let mesh = tx.mesh();
    let Some(sizes) = mesh.channel(REST_SIZE_CHANNEL) else {
        return Err(OpsError::InvalidInput(
            "split_stretched needs capture_rest_sizes first",
        ));
    };
    let rows = Rows::of(mesh)?;
    let mut work = Work {
        mesh,
        rows: &rows,
        slots: mesh.vertex_slots(),
        sizes,
        limit: f64::from(options.limit),
        new_sizes: Vec::new(),
        edges: Vec::new(),
        added: Vec::new(),
        tris: Vec::new(),
        tri_of: HashMap::new(),
        row_of: HashMap::new(),
    };
    let mut heap = work.initial_edges(&options.region)?;
    let mut report = StretchReport::default();
    let mut interior = 0;
    while let Some(edge) = heap.peek().copied() {
        if !edge.boundary && interior >= options.max_splits {
            break;
        }
        heap.pop();
        let owners = work.owners(edge.a, edge.b)?;
        if owners.is_empty() {
            continue;
        }
        let boundary = owners.len() != 2;
        let mid = midpoint(&work.pos(edge.a), &work.pos(edge.b));
        let pos = if options.smooth && !boundary {
            let curved = work.curved_point(edge.a, edge.b, &owners);
            if work.shortens(edge, curved) {
                curved
            } else {
                mid
            }
        } else {
            mid
        };
        // Too short to split at float precision: it stays as it is.
        if !work.shortens(edge, pos) {
            continue;
        }
        if !boundary {
            interior += 1;
        }
        work.split(edge.a, edge.b, pos, &owners, boundary, &mut heap)?;
        report.splits += 1;
    }
    report.remaining = heap.len();
    if report.splits > 0 {
        let slots = work.slots;
        let edges = work.edges;
        let plan = Plan {
            slots,
            tris: work.tris,
            added: work.added,
        };
        let ids = tx.keeping_bound_channels(|tx| plan.commit(tx))?;
        let resolve = |v: u32| {
            if (v as usize) < slots {
                v
            } else {
                ids[v as usize - slots]
            }
        };
        reserve(&mut report.vertices, ids.len())?;
        for (&id, &(a, b)) in ids.iter().zip(&edges) {
            report.vertices.push([id, resolve(a), resolve(b)]);
        }
    }
    Ok(report)
}

/// An edge past its limit. Boundary edges order first, then longer ones; the
/// endpoints break ties so the order never depends on hashing.
#[derive(Clone, Copy, PartialEq)]
struct Edge {
    boundary: bool,
    len: f64,
    a: u32,
    b: u32,
}

impl Eq for Edge {}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.boundary
            .cmp(&other.boundary)
            .then(self.len.total_cmp(&other.len))
            .then(other.a.cmp(&self.a))
            .then(other.b.cmp(&self.b))
    }
}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The triangles of each vertex: the mesh's cached index, or a private one.
enum Rows<'m> {
    Cached(&'m Incidence),
    Built { offsets: Vec<u32>, ids: Vec<u32> },
}

impl<'m> Rows<'m> {
    fn of(mesh: &'m Mesh) -> Result<Self> {
        if let Some(index) = mesh.incidence() {
            return Ok(Rows::Cached(index));
        }
        let mut offsets: Vec<u32> = filled(mesh.vertex_slots() + 1, 0)?;
        for t in 0..mesh.triangle_slots() as u32 {
            if let Some(c) = mesh.triangle(t)
                && !degenerate(&c)
            {
                for v in c {
                    offsets[v as usize + 1] += 1;
                }
            }
        }
        for v in 1..offsets.len() {
            offsets[v] += offsets[v - 1];
        }
        let mut ids: Vec<u32> = filled(offsets[offsets.len() - 1] as usize, 0)?;
        let mut next: Vec<u32> = Vec::new();
        reserve(&mut next, offsets.len())?;
        next.extend_from_slice(&offsets);
        for t in 0..mesh.triangle_slots() as u32 {
            if let Some(c) = mesh.triangle(t)
                && !degenerate(&c)
            {
                for v in c {
                    ids[next[v as usize] as usize] = t;
                    next[v as usize] += 1;
                }
            }
        }
        Ok(Rows::Built { offsets, ids })
    }

    fn row(&self, v: u32) -> &[TriangleId] {
        match self {
            Rows::Cached(index) => index.triangles(v),
            Rows::Built { offsets, ids } => {
                &ids[offsets[v as usize] as usize..offsets[v as usize + 1] as usize]
            }
        }
    }
}

/// A triangle the plan has read or made: `id` is `None` for one a split made.
struct WorkTri {
    id: Option<TriangleId>,
    corners: [u32; 3],
}

/// The plan's state, sparse over the mesh: vertices past `slots` are new, and
/// only triangles around a split edge are ever loaded.
///
/// A vertex's row is loaded before anything it holds changes, and a split only
/// changes the rows of its edge's endpoints and apexes, which it loads first.
/// So a row not loaded yet still matches the mesh.
struct Work<'a> {
    mesh: &'a Mesh,
    rows: &'a Rows<'a>,
    slots: usize,
    sizes: &'a [f32],
    limit: f64,
    new_sizes: Vec<f32>,
    /// The edge each entry of `added` split.
    edges: Vec<(u32, u32)>,
    added: Vec<NewVert>,
    tris: Vec<WorkTri>,
    tri_of: HashMap<TriangleId, usize>,
    row_of: HashMap<u32, Vec<usize>>,
}

impl Work<'_> {
    fn pos(&self, v: u32) -> [f32; 3] {
        let i = v as usize;
        if i < self.slots {
            self.mesh.positions()[i]
        } else {
            self.added[i - self.slots].pos
        }
    }

    fn size(&self, v: u32) -> f32 {
        let i = v as usize;
        if i < self.slots {
            self.sizes[i]
        } else {
            self.new_sizes[i - self.slots]
        }
    }

    /// The length past which `(a, b)` splits; `None` when neither endpoint
    /// has a rest size, so nothing says the edge stretched.
    fn edge_limit(&self, a: u32, b: u32) -> Option<f64> {
        let usable = |s: f32| s.is_finite() && s > 0.0;
        let (sa, sb) = (self.size(a), self.size(b));
        let size = match (usable(sa), usable(sb)) {
            (true, true) => sa.min(sb),
            (true, false) => sa,
            (false, true) => sb,
            (false, false) => return None,
        };
        Some(f64::from(size) * self.limit)
    }

    /// Whether a vertex at `pos` makes both halves of `edge` shorter than the
    /// edge. It is what ends the pass: without it, a midpoint that rounds onto
    /// an endpoint leaves a half as long as the edge, past its limit for ever.
    fn shortens(&self, edge: Edge, pos: [f32; 3]) -> bool {
        pos.iter().all(|x| x.is_finite())
            && length(&self.pos(edge.a), &pos) < edge.len
            && length(&pos, &self.pos(edge.b)) < edge.len
    }

    /// The edge as a heap entry when it is past its limit.
    fn over(&self, a: u32, b: u32, boundary: bool) -> Option<Edge> {
        let limit = self.edge_limit(a, b)?;
        let len = length(&self.pos(a), &self.pos(b));
        (len > limit).then_some(Edge {
            boundary,
            len,
            a: a.min(b),
            b: a.max(b),
        })
    }

    /// Every edge past its limit with an endpoint in the region. The list is
    /// deduplicated first, so an edge seen from both ends enters the heap once.
    fn initial_edges(&mut self, region: &Region) -> Result<BinaryHeap<Edge>> {
        let positions = self.mesh.positions();
        let center = f64v(&region.center);
        let radius = f64::from(region.radius);
        let mut found: Vec<(u32, u32)> = Vec::new();
        for (v, p) in positions.iter().enumerate() {
            let v = v as u32;
            // Squared: this runs for every vertex, and `hypot` is slow on wasm.
            let d = sub3(f64v(p), center);
            if dot3(d, d) >= radius * radius || !self.mesh.is_vertex_alive(v) {
                continue;
            }
            for &t in self.rows.row(v) {
                let Some(c) = self.mesh.triangle(t) else {
                    continue;
                };
                if degenerate(&c) {
                    continue;
                }
                for other in c {
                    if other != v && self.over(v, other, false).is_some() {
                        push(&mut found, (v.min(other), v.max(other)))?;
                    }
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        let mut heap = BinaryHeap::new();
        allocation_allowed()?;
        heap.try_reserve(found.len()).map_err(|_| oom())?;
        for (a, b) in found {
            let owners = self.owners(a, b)?;
            if let Some(edge) = self.over(a, b, owners.len() != 2) {
                heap.push(edge);
            }
        }
        Ok(heap)
    }

    /// Makes sure `v`'s row is in `row_of`.
    fn load(&mut self, v: u32) -> Result<()> {
        if self.row_of.contains_key(&v) {
            return Ok(());
        }
        let base = if (v as usize) < self.slots {
            self.rows.row(v)
        } else {
            &[]
        };
        let mut row: Vec<usize> = Vec::new();
        reserve(&mut row, base.len())?;
        for &t in base {
            let i = match self.tri_of.get(&t) {
                Some(&i) => i,
                None => {
                    let Some(corners) = self.mesh.triangle(t) else {
                        continue;
                    };
                    let i = self.tris.len();
                    push(
                        &mut self.tris,
                        WorkTri {
                            id: Some(t),
                            corners,
                        },
                    )?;
                    reserve_map(&mut self.tri_of)?;
                    self.tri_of.insert(t, i);
                    i
                }
            };
            row.push(i);
        }
        reserve_map(&mut self.row_of)?;
        self.row_of.insert(v, row);
        Ok(())
    }

    /// The triangles on `(a, b)`, in the order of `a`'s row. Empty once the
    /// edge has split. Degenerate triangles do not count, and keep the edge.
    fn owners(&mut self, a: u32, b: u32) -> Result<Vec<usize>> {
        self.load(a)?;
        self.load(b)?;
        let mut owners: Vec<usize> = Vec::new();
        for &i in &self.row_of[&a] {
            let c = &self.tris[i].corners;
            if !degenerate(c) && carries(c, a, b) {
                push(&mut owners, i)?;
            }
        }
        Ok(owners)
    }

    fn normal(&self, corners: &[u32; 3]) -> [f64; 3] {
        let [p0, p1, p2] = corners.map(|v| f64v(&self.pos(v)));
        cross3(sub3(p1, p0), sub3(p2, p0))
    }

    /// The endpoint's normal over the triangles facing like `facing`, a unit
    /// vector, weighted by area. `None` when they cancel out.
    fn vertex_normal(&self, v: u32, facing: [f64; 3]) -> Option<[f64; 3]> {
        let mut sum = [0.0; 3];
        for &i in &self.row_of[&v] {
            let c = &self.tris[i].corners;
            if degenerate(c) {
                continue;
            }
            let n = self.normal(c);
            let Some(direction) = unit(n) else { continue };
            if dot3(direction, facing) >= SHARP_COS {
                sum = [sum[0] + n[0], sum[1] + n[1], sum[2] + n[2]];
            }
        }
        unit(sum)
    }

    /// The new vertex of an interior edge: see the module docs. Falls back to
    /// the midpoint on a sharp edge, and when a child would fold.
    fn curved_point(&self, a: u32, b: u32, owners: &[usize]) -> [f32; 3] {
        let (pa, pb) = (self.pos(a), self.pos(b));
        let mid = midpoint(&pa, &pb);
        let n0 = unit(self.normal(&self.tris[owners[0]].corners));
        let n1 = unit(self.normal(&self.tris[owners[1]].corners));
        let (Some(n0), Some(n1)) = (n0, n1) else {
            return mid;
        };
        if dot3(n0, n1) < SHARP_COS {
            return mid;
        }
        let Some(facing) = unit([n0[0] + n1[0], n0[1] + n1[1], n0[2] + n1[2]]) else {
            return mid;
        };
        let (Some(na), Some(nb)) = (self.vertex_normal(a, facing), self.vertex_normal(b, facing))
        else {
            return mid;
        };
        let (fa, fb) = (f64v(&pa), f64v(&pb));
        let wa = dot3(sub3(fb, fa), na);
        let wb = dot3(sub3(fa, fb), nb);
        let curved: [f64; 3] =
            std::array::from_fn(|k| (fa[k] + fb[k]) / 2.0 - (wa * na[k] + wb * nb[k]) / 8.0);
        let pos = curved.map(|x| x as f32);
        if pos.iter().any(|x| !x.is_finite()) {
            return mid;
        }
        let m = f64v(&pos);
        for &i in owners {
            let c = self.tris[i].corners;
            let apex = f64v(&self.pos(apex_of(&c, a, b)));
            let parent = self.normal(&c);
            let (first, second) = if runs_directed(&c, a, b) {
                ((fa, m), (m, fb))
            } else {
                ((fb, m), (m, fa))
            };
            let n1 = cross3(sub3(first.1, first.0), sub3(apex, first.0));
            let n2 = cross3(sub3(second.1, second.0), sub3(apex, second.0));
            if dot3(n1, parent) <= 0.0 || dot3(n2, parent) <= 0.0 {
                return mid;
            }
        }
        pos
    }

    /// Splits `(a, b)` at `pos`: each owner becomes two triangles that follow
    /// its winding, and the edges this made join the heap if past their limit.
    fn split(
        &mut self,
        a: u32,
        b: u32,
        pos: [f32; 3],
        owners: &[usize],
        boundary: bool,
        heap: &mut BinaryHeap<Edge>,
    ) -> Result<()> {
        let m = (self.slots + self.added.len()) as u32;
        let parents = blend_parents(a, b, self.slots, &self.added)?;
        let size = self.size(a) / 2.0 + self.size(b) / 2.0;
        push(&mut self.new_sizes, size)?;
        push(&mut self.edges, (a, b))?;
        push(&mut self.added, NewVert { pos, parents })?;
        reserve_map(&mut self.row_of)?;
        self.row_of.insert(m, Vec::new());
        for &i in owners {
            let corners = self.tris[i].corners;
            let apex = apex_of(&corners, a, b);
            self.load(apex)?;
            let (first, second, moved) = if runs_directed(&corners, a, b) {
                ([a, m, apex], [m, b, apex], b)
            } else {
                ([b, m, apex], [m, a, apex], a)
            };
            self.tris[i].corners = first;
            let child = self.tris.len();
            push(
                &mut self.tris,
                WorkTri {
                    id: None,
                    corners: second,
                },
            )?;
            let row = self.row_of.get_mut(&moved).expect("endpoints are loaded");
            row.retain(|&t| t != i);
            push(row, child)?;
            let row = self.row_of.get_mut(&apex).expect("apex is loaded");
            push(row, child)?;
            let row = self.row_of.get_mut(&m).expect("the new row is there");
            push(row, i)?;
            push(row, child)?;
            if let Some(edge) = self.over(m, apex, false) {
                reserve_heap(heap)?;
                heap.push(edge);
            }
        }
        for end in [a, b] {
            if let Some(edge) = self.over(end, m, boundary) {
                reserve_heap(heap)?;
                heap.push(edge);
            }
        }
        Ok(())
    }
}

/// What a finished [`Work`] writes.
struct Plan {
    slots: usize,
    tris: Vec<WorkTri>,
    added: Vec<NewVert>,
}

impl Plan {
    /// Everything is reserved before the first write, so an allocation failure
    /// leaves an empty patch.
    fn commit(self, tx: &mut Transaction) -> Result<Vec<VertexId>> {
        let mut ids: Vec<VertexId> = Vec::new();
        reserve(&mut ids, self.added.len())?;
        let mut set_ids: Vec<TriangleId> = Vec::new();
        reserve(&mut set_ids, self.tris.len())?;
        let mut set_tris: Vec<[VertexId; 3]> = Vec::new();
        reserve(&mut set_tris, self.tris.len())?;
        let mut new_tris: Vec<[VertexId; 3]> = Vec::new();
        reserve(&mut new_tris, self.tris.len())?;
        for nv in &self.added {
            ids.push(tx.add_vertex_with_parents(nv.pos, &nv.parents)?);
        }
        let slots = self.slots;
        let resolve = |v: u32| {
            if (v as usize) < slots {
                v
            } else {
                ids[v as usize - slots]
            }
        };
        let mesh = tx.mesh();
        for tri in &self.tris {
            let current = tri.corners.map(resolve);
            match tri.id {
                Some(id) => {
                    if mesh.triangle(id) != Some(current) {
                        set_ids.push(id);
                        set_tris.push(current);
                    }
                }
                None => new_tris.push(current),
            }
        }
        tx.set_triangles(&set_ids, &set_tris)?;
        tx.add_triangles(&new_tris)?;
        Ok(ids)
    }
}

fn oom() -> OpsError {
    OpsError::Store(StoreError::OutOfMemory)
}

fn reserve_map<K: Eq + std::hash::Hash, V>(map: &mut HashMap<K, V>) -> Result<()> {
    allocation_allowed()?;
    map.try_reserve(1).map_err(|_| oom())
}

fn reserve_heap(heap: &mut BinaryHeap<Edge>) -> Result<()> {
    allocation_allowed()?;
    heap.try_reserve(1).map_err(|_| oom())
}

fn length(a: &[f32; 3], b: &[f32; 3]) -> f64 {
    let d = sub3(f64v(a), f64v(b));
    dot3(d, d).sqrt()
}

fn unit(v: [f64; 3]) -> Option<[f64; 3]> {
    let len = dot3(v, v).sqrt();
    (len.is_finite() && len > 0.0).then(|| v.map(|x| x / len))
}

#[cfg(test)]
#[path = "stretch_tests.rs"]
mod tests;
