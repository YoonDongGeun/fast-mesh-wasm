//! Sign data: what decides whether a point lies in front of or behind a mesh.
//!
//! Pseudonormals (Bærentzen and Aanæs, "Signed distance computation using the
//! angle weighted pseudonormal", 2005): the sign of `dot(p − closest, n)`,
//! where `n` is the normal of the feature the closest point lies on:
//!
//! - a face uses its own normal;
//! - an edge uses the sum of the unit normals of the faces on it (only its own
//!   face on a boundary edge);
//! - a vertex uses the angle-weighted sum of the unit normals of the faces
//!   around it.
//!
//! The sign is exact for closed, consistently oriented manifold meshes. On an
//! open sheet it tells the side of the surface, so "behind the surface" reads
//! as inside.
//!
//! **Zero-area triangles** (exactly collinear or coincident corners) have no
//! normal. They are skipped when normals are summed, and an edge whose
//! neighbor across is a zero-area sliver looks through it (up to
//! [`SLIVER_HOPS`] slivers deep) for the face on the other side. When the
//! closest point is reported on a zero-area triangle, the sign comes from the
//! nearest feature of the non-degenerate triangles that share a vertex with
//! it (ties go to the smallest triangle id).
//!
//! **Ties:** a point exactly on the plane of its feature's normal (the dot
//! product is exactly zero, or the normal is zero) counts as outside, so its
//! distance is positive. A point on the surface (distance zero) gets zero.

use std::ops::Range;

use fast_mesh_bvh::Bvh;
use fast_mesh_query::{ClosestHit, Feature, QueryError};
use fast_mesh_store::{
    Adjacency, AdjacencyBuild, EdgeKind, Incidence, IncidenceBuild, Mesh, TriangleId, VertexId, par,
};

use crate::geom::{
    add, closest_on_triangle, cross, dot, narrow, norm, scale, sub, triangle_normal, unit, widen,
};
use crate::winding::{Moments, MomentsBuild};
use crate::{FieldsError, Result};

/// How many zero-area slivers an edge normal looks through to find the face
/// on the other side.
const SLIVER_HOPS: u32 = 4;

/// Elements (vertex slots, triangle slots, corners or BVH leaves) from which
/// a pass of [`SignData::build`] runs in parallel chunks with the `parallel`
/// feature. Measured on the i5-13600K with every pass parallel: a
/// pseudonormal build of a 4k-vertex sheet breaks even (1 ms), of 1k vertices
/// is slower in parallel, of 8k vertices twice as fast. Each pass decides on
/// its own length, so a 4k-vertex sheet runs its triangle and corner passes
/// in parallel and its vertex pass on the calling thread.
const PAR_MIN_SIGN_ELEMS: usize = 8 * 1024;

/// What a vertex pseudonormal counts for in a [`SignBuild::step`] budget,
/// against one element of the cheaper passes. Measured on the i5-13600K at
/// 1M triangles, single-threaded, in steps of 131,072 elements: about 70 ns
/// per vertex normal, 10 to 15 ns per edge normal and per triangle of the
/// adjacency build, and 5 ns or less per element of the incidence, face
/// normal and topology passes.
const VERTEX_NORMAL_WEIGHT: usize = 12;

/// What an edge normal counts for in a [`SignBuild::step`] budget (see
/// [`VERTEX_NORMAL_WEIGHT`]).
const EDGE_NORMAL_WEIGHT: usize = 2;

/// What a triangle or vertex slot of the adjacency build counts for in a
/// [`SignBuild::step`] budget (see [`VERTEX_NORMAL_WEIGHT`]).
const ADJACENCY_WEIGHT: usize = 3;

/// Elements per parallel chunk of a [`SignData::build`] pass.
const SIGN_CHUNK: usize = 2048;

/// How the sign of a distance is decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SignMode {
    /// The side of the surface the point lies on, from the angle-weighted
    /// pseudonormal at the closest feature. Exact for closed, consistently
    /// oriented manifold meshes; on an open mesh "inside" means behind the
    /// surface.
    Pseudonormal,
    /// The generalized winding number (see [`SignData::winding_number`]):
    /// inside where it is at least one half. Robust to self-intersections and
    /// inconsistent orientation, but it never reaches one half behind an open
    /// sheet.
    Winding,
    /// [`SignMode::Pseudonormal`], except for a closed mesh whose orientation
    /// is inconsistent, which needs [`SignMode::Winding`].
    #[default]
    Auto,
}

/// What [`SignData::build`] found out about a mesh, so that an application
/// can warn when signs are doubtful.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignStatus {
    /// The mode in use: never [`SignMode::Auto`].
    pub mode: SignMode,
    /// True when no edge is a boundary (used by one triangle) or non-manifold
    /// (used by three or more). Zero-length edges of triangles that repeat a
    /// vertex are not counted as edges.
    pub closed: bool,
    /// True when every edge shared by exactly two triangles is traversed in
    /// opposite directions by them.
    pub consistent_orientation: bool,
    /// Alive vertices whose triangles do not form a single fan (see
    /// [`Adjacency::is_non_manifold_vertex`]).
    pub non_manifold_vertices: usize,
    /// Alive triangles with exactly zero area.
    pub degenerate_triangles: usize,
}

/// Per-mesh data that decides the sign of a distance, built by
/// [`SignData::build`] for one revision of a mesh and one state of its BVH.
///
/// In [`SignMode::Pseudonormal`] it holds a unit pseudonormal per vertex
/// slot, an edge normal per triangle corner (for the edge opposite that
/// corner) and, for every zero-area triangle, the non-degenerate triangles
/// that share a vertex with it. Face normals are recomputed from the mesh when
/// needed.
///
/// In [`SignMode::Winding`] it holds instead the order-1 dipole moments of
/// every BVH node (area-weighted normal sum, area-weighted centroid and a
/// bounding radius), for the fast winding number of Barill et al. 2018.
#[derive(Debug, Clone)]
pub struct SignData {
    mesh_uid: u64,
    epoch: u32,
    revision: u64,
    bvh_generation: u64,
    status: SignStatus,
    vertex_normals: Vec<[f32; 3]>,
    /// Indexed by corner `3 * t + i`: the normal of the edge opposite corner `i`.
    edge_normals: Vec<[f32; 3]>,
    /// Alive zero-area triangles, in ascending id order.
    degenerate: Vec<TriangleId>,
    /// `candidates[offsets[i]..offsets[i + 1]]` are the non-degenerate
    /// triangles sharing a vertex with `degenerate[i]`, in ascending id order.
    offsets: Vec<u32>,
    candidates: Vec<TriangleId>,
    /// Winding moments per BVH node; empty unless the mode is `Winding`.
    pub(crate) moments: Moments,
}

/// A vector of `len` copies of `value`, or `OutOfMemory`.
pub(crate) fn try_vec<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| FieldsError::OutOfMemory)?;
    v.resize(len, value);
    Ok(v)
}

/// Pushes with a fallible reservation.
fn try_push<T>(v: &mut Vec<T>, value: T) -> Result<()> {
    v.try_reserve(1).map_err(|_| FieldsError::OutOfMemory)?;
    v.push(value);
    Ok(())
}

/// `f(i)` for every `i` in `span`, in order. With the `parallel` feature, a
/// span of `PAR_MIN_SIGN_ELEMS` or more runs in chunks of [`SIGN_CHUNK`] on
/// the pool; every element is computed alone, so the result is the same
/// either way, and collecting a range in spans gives the same elements as
/// all at once.
pub(crate) fn collect_span<T: Send>(
    span: Range<usize>,
    f: impl Fn(usize) -> T + Sync,
) -> Result<Vec<T>> {
    let mut v = Vec::new();
    let total = span.len();
    extend_span(&mut v, span, total, f)?;
    Ok(v)
}

/// Appends `f(i)` for every `i` in `span` to `v`, in order, in parallel
/// under the rule of [`collect_span`]. `v` is reserved for `total` elements
/// in all at once, so that filling it in spans never copies it.
fn extend_span<T: Send>(
    v: &mut Vec<T>,
    span: Range<usize>,
    total: usize,
    f: impl Fn(usize) -> T + Sync,
) -> Result<()> {
    let oom = || FieldsError::OutOfMemory;
    let (start, len) = (span.start, span.len());
    let room = total.max(v.len() + len) - v.len();
    v.try_reserve_exact(room).map_err(|_| oom())?;
    if len < PAR_MIN_SIGN_ELEMS || !par::enabled() || par::force_sequential() {
        v.extend(span.map(f));
        return Ok(());
    }
    let parts = par::map_range(len, SIGN_CHUNK, PAR_MIN_SIGN_ELEMS, oom, |r| {
        let mut part = Vec::new();
        part.try_reserve_exact(r.len()).map_err(|_| oom())?;
        part.extend((r.start + start..r.end + start).map(&f));
        Ok(part)
    })?;
    parts.into_iter().for_each(|p| v.extend(p));
    Ok(())
}

/// `f` over `span` cut into chunks of [`SIGN_CHUNK`], results in chunk
/// order, in parallel under the same rule as [`collect_span`]. For
/// reductions whose combine does not depend on the order (counts, flags).
fn reduce_span<R: Send>(
    span: Range<usize>,
    f: impl Fn(Range<usize>) -> R + Sync,
) -> Result<Vec<R>> {
    let oom = || FieldsError::OutOfMemory;
    let start = span.start;
    par::map_range(span.len(), SIGN_CHUNK, PAR_MIN_SIGN_ELEMS, oom, |r| {
        Ok(f(r.start + start..r.end + start))
    })
}

fn corners(mesh: &Mesh, tri: [VertexId; 3]) -> [[f32; 3]; 3] {
    tri.map(|v| mesh.positions()[v as usize])
}

/// Topology facts about the triangles in `span`: whether none has a boundary
/// or non-manifold edge, whether every interior edge is traversed in
/// opposite directions, and how many have zero area.
fn topology_span(
    mesh: &Mesh,
    adj: &Adjacency,
    face: &[[f64; 3]],
    span: Range<usize>,
) -> Result<(bool, bool, usize)> {
    let tris = mesh.triangles();
    let parts = reduce_span(span, |range| {
        let (mut closed, mut consistent, mut degenerate) = (true, true, 0);
        for t in range {
            let tri = tris[t];
            if !mesh.is_triangle_alive(t as TriangleId) {
                continue;
            }
            if face[t] == [0.0; 3] {
                degenerate += 1;
            }
            for i in 0..3 {
                let (u, w) = (tri[(i + 1) % 3], tri[(i + 2) % 3]);
                if u == w {
                    continue;
                }
                let c = (3 * t + i) as u32;
                match adj.edge_kind(c) {
                    EdgeKind::Boundary | EdgeKind::NonManifold => closed = false,
                    EdgeKind::Interior => {
                        if let Some(o) = adj.opposite(c) {
                            let (s, j) = ((o / 3) as usize, (o % 3) as usize);
                            let other = tris[s];
                            if (other[(j + 1) % 3], other[(j + 2) % 3]) != (w, u) {
                                consistent = false;
                            }
                        }
                    }
                }
            }
        }
        (closed, consistent, degenerate)
    })?;
    Ok(parts
        .into_iter()
        .fold((true, true, 0), |(a, b, n), (c, d, m)| {
            (a && c, b && d, n + m)
        }))
}

/// The alive non-manifold vertices among the vertex slots in `span`.
fn non_manifold_span(mesh: &Mesh, adj: &Adjacency, span: Range<usize>) -> Result<usize> {
    Ok(reduce_span(span, |range| {
        range
            .filter(|&v| {
                let v = v as VertexId;
                mesh.is_vertex_alive(v) && adj.is_non_manifold_vertex(v)
            })
            .count()
    })?
    .into_iter()
    .sum())
}

/// The unit normal of triangle slot `t`; zero for a dead or zero-area one.
fn face_normal(mesh: &Mesh, t: usize) -> [f64; 3] {
    mesh.triangle(t as TriangleId)
        .map_or([0.0; 3], |tri| unit(triangle_normal(corners(mesh, tri))))
}

/// The angle-weighted pseudonormal of vertex slot `v`, from the
/// non-degenerate triangles around it.
fn vertex_normal(mesh: &Mesh, inc: &Incidence, face: &[[f64; 3]], v: usize) -> [f32; 3] {
    let mut sum = [0.0; 3];
    for &t in inc.triangles(v as VertexId) {
        let n = face[t as usize];
        if n == [0.0; 3] {
            continue;
        }
        let tri = mesh.triangles()[t as usize];
        let Some(i) = tri.iter().position(|&x| x as usize == v) else {
            continue;
        };
        let p = corners(mesh, tri).map(widen);
        let (e1, e2) = (sub(p[(i + 1) % 3], p[i]), sub(p[(i + 2) % 3], p[i]));
        let angle = norm(cross(e1, e2)).atan2(dot(e1, e2));
        sum = add(sum, scale(n, angle));
    }
    narrow(unit(sum))
}

/// The corner of triangle `s` opposite its longest edge (the first on ties).
fn longest_edge_corner(mesh: &Mesh, s: usize) -> u32 {
    let p = corners(mesh, mesh.triangles()[s]).map(widen);
    let len2 = |j: usize| {
        let e = sub(p[(j + 2) % 3], p[(j + 1) % 3]);
        dot(e, e)
    };
    let mut best = 0;
    for j in 1..3 {
        if len2(j) > len2(best) {
            best = j;
        }
    }
    best as u32
}

/// The unit normal of the face across the edge opposite corner `c`, looking
/// through up to `hops` zero-area slivers, or zero when there is none.
///
/// Entering a sliver through one of its short edges, the surface continues
/// across its longest edge. Entering through the longest edge, it continues
/// across both short edges, and their normals are averaged.
fn across(mesh: &Mesh, adj: &Adjacency, face: &[[f64; 3]], c: u32, hops: u32) -> [f64; 3] {
    let Some(o) = adj.opposite(c) else {
        return [0.0; 3];
    };
    let s = (o / 3) as usize;
    if face[s] != [0.0; 3] {
        return face[s];
    }
    if hops == 0 {
        return [0.0; 3];
    }
    let long = longest_edge_corner(mesh, s);
    let base = 3 * s as u32;
    if o % 3 != long {
        across(mesh, adj, face, base + long, hops - 1)
    } else {
        let sum = (0..3)
            .filter(|&j| j != long)
            .map(|j| across(mesh, adj, face, base + j, hops - 1))
            .fold([0.0; 3], add);
        unit(sum)
    }
}

/// The edge normal of corner `c`: the sum of the unit normals of the faces
/// on the edge opposite it.
fn edge_normal(
    mesh: &Mesh,
    adj: &Adjacency,
    inc: &Incidence,
    face: &[[f64; 3]],
    c: usize,
) -> [f32; 3] {
    let tris = mesh.triangles();
    let (t, i) = (c / 3, c % 3);
    if !mesh.is_triangle_alive(t as TriangleId) {
        return [0.0f32; 3];
    }
    let tri = tris[t];
    let mut sum = face[t];
    match adj.edge_kind(c as u32) {
        EdgeKind::Interior => sum = add(sum, across(mesh, adj, face, c as u32, SLIVER_HOPS)),
        EdgeKind::Boundary => {}
        EdgeKind::NonManifold => {
            let (u, w) = (tri[(i + 1) % 3], tri[(i + 2) % 3]);
            for &s in inc.triangles(u) {
                if s as usize != t && tris[s as usize].contains(&w) {
                    sum = add(sum, face[s as usize]);
                }
            }
        }
    }
    narrow(sum)
}

/// The pass a [`SignBuild`] is in, with its cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// Building the mesh's adjacency, unless cached.
    Adjacency,
    /// Building the mesh's incidence, unless cached.
    Incidence,
    /// The unit normals of the triangle slots, from this one on.
    Faces(usize),
    /// The topology facts of the triangle slots, from this one on.
    Topology(usize),
    /// The non-manifold count over the vertex slots, from this one on.
    NonManifold(usize),
    /// The pseudonormals of the vertex slots, from this one on.
    VertexNormals(usize),
    /// The edge normals of the corners, from this one on.
    EdgeNormals(usize),
    /// The zero-area triangles and their neighbours, from this slot on.
    Degenerate(usize),
    /// The winding moments, leaf by leaf.
    Moments,
    /// Finished: [`SignBuild::finish`] hands out the data.
    Done,
}

/// [`SignData::build`] cut into steps, so that a caller can yield between
/// them: the mesh's adjacency and incidence (when not cached), then each pass
/// of the build, a bounded number of elements per step.
///
/// Every element of a pass is computed on its own, and the counts and flags
/// combine in any order, so the finished data is the same, bit for bit, as
/// from [`SignData::build`] (which runs one of these to the end), with or
/// without the `parallel` feature: within a step, a long enough run of
/// elements still runs in parallel chunks. A build is tied to the mesh state
/// and BVH generation it started from ([`SignBuild::is_for`]); stepping it on
/// another state fails.
#[derive(Debug)]
pub struct SignBuild {
    data: SignData,
    requested: SignMode,
    stage: Stage,
    adjacency: AdjacencyBuild,
    incidence: IncidenceBuild,
    face: Vec<[f64; 3]>,
    moments: Option<MomentsBuild>,
}

impl SignBuild {
    /// A build of `mesh`'s sign data in `mode`, against `bvh`, with nothing
    /// done yet.
    ///
    /// # Errors
    /// [`FieldsError::Query`] with `MeshMismatch` or `StaleBvh` when `bvh`
    /// does not match the current `mesh`.
    pub fn new(mesh: &Mesh, bvh: &Bvh, mode: SignMode) -> Result<SignBuild> {
        if bvh.mesh_uid() != mesh.uid() {
            return Err(QueryError::MeshMismatch.into());
        }
        if !bvh.is_current(mesh) {
            return Err(QueryError::StaleBvh.into());
        }
        Ok(SignBuild {
            data: SignData {
                mesh_uid: mesh.uid(),
                epoch: mesh.epoch(),
                revision: mesh.revision(),
                bvh_generation: bvh.generation(),
                status: SignStatus {
                    mode: SignMode::Pseudonormal,
                    closed: true,
                    consistent_orientation: true,
                    non_manifold_vertices: 0,
                    degenerate_triangles: 0,
                },
                vertex_normals: Vec::new(),
                edge_normals: Vec::new(),
                degenerate: Vec::new(),
                offsets: Vec::new(),
                candidates: Vec::new(),
                moments: Moments::default(),
            },
            requested: mode,
            stage: Stage::Adjacency,
            adjacency: AdjacencyBuild::new(mesh),
            incidence: IncidenceBuild::new(mesh),
            face: Vec::new(),
            moments: None,
        })
    }

    /// Whether the build started from `mesh` and `bvh` as they are now (see
    /// [`SignData::is_current`]).
    pub fn is_for(&self, mesh: &Mesh, bvh: &Bvh) -> bool {
        self.data.is_current(mesh, bvh)
    }

    /// Whether every pass is done.
    pub fn is_done(&self) -> bool {
        self.stage == Stage::Done
    }

    /// Runs the build over up to `budget` more elements (at least one,
    /// unless it is done): triangle, vertex or corner slots of a pass, or
    /// leaves of the winding moments, a vertex pseudonormal counting as 12,
    /// an edge normal as 2 and a triangle of the adjacency build as 3, for
    /// what they cost. A step that finishes the mesh's adjacency or
    /// incidence ends there. Returns whether the build is done.
    ///
    /// # Errors
    /// [`FieldsError::InvalidInput`] when `mesh` and `bvh` are not the state
    /// the build started from ([`SignBuild::is_for`]);
    /// [`FieldsError::OutOfMemory`] or [`FieldsError::Store`] when an array
    /// cannot be allocated. The build is unusable after an error.
    pub fn step(&mut self, mesh: &mut Mesh, bvh: &Bvh, budget: usize) -> Result<bool> {
        if !self.is_for(mesh, bvh) {
            return Err(FieldsError::InvalidInput(
                "the sign data build started from another mesh or BVH state",
            ));
        }
        let mut left = budget.max(1);
        while left > 0 {
            match self.stage {
                Stage::Adjacency => {
                    let budget = (left / ADJACENCY_WEIGHT).max(1);
                    if mesh.step_adjacency(&mut self.adjacency, budget)? {
                        self.stage = Stage::Incidence;
                    }
                    // A partial or finishing adjacency step used the budget.
                    return Ok(false);
                }
                Stage::Incidence => {
                    if mesh.step_incidence(&mut self.incidence, left)? {
                        self.stage = Stage::Faces(0);
                    }
                    return Ok(false);
                }
                _ => {}
            }
            let mesh: &Mesh = mesh;
            let missing = FieldsError::InvalidInput("mesh adjacency or incidence unavailable");
            let adj = mesh.adjacency().ok_or(missing.clone())?;
            let inc = mesh.incidence().ok_or(missing)?;
            let span =
                |from: usize, len: usize, left: usize| from..len.min(from.saturating_add(left));
            match self.stage {
                Stage::Adjacency | Stage::Incidence => unreachable!("handled above"),
                Stage::Faces(from) => {
                    let r = span(from, mesh.triangle_slots(), left);
                    let (to, n) = (r.end, r.len());
                    let total = mesh.triangle_slots();
                    extend_span(&mut self.face, r, total, |t| face_normal(mesh, t))?;
                    left -= n.min(left);
                    self.stage = if to == mesh.triangle_slots() {
                        Stage::Topology(0)
                    } else {
                        Stage::Faces(to)
                    };
                }
                Stage::Topology(from) => {
                    let r = span(from, mesh.triangles().len(), left);
                    let (to, n) = (r.end, r.len());
                    let (closed, consistent, degenerate) = topology_span(mesh, adj, &self.face, r)?;
                    let st = &mut self.data.status;
                    st.closed &= closed;
                    st.consistent_orientation &= consistent;
                    st.degenerate_triangles += degenerate;
                    left -= n.min(left);
                    self.stage = if to == mesh.triangles().len() {
                        Stage::NonManifold(0)
                    } else {
                        Stage::Topology(to)
                    };
                }
                Stage::NonManifold(from) => {
                    let r = span(from, mesh.vertex_slots(), left);
                    let (to, n) = (r.end, r.len());
                    self.data.status.non_manifold_vertices += non_manifold_span(mesh, adj, r)?;
                    left -= n.min(left);
                    if to < mesh.vertex_slots() {
                        self.stage = Stage::NonManifold(to);
                        continue;
                    }
                    let st = &mut self.data.status;
                    st.mode = match self.requested {
                        SignMode::Pseudonormal => SignMode::Pseudonormal,
                        SignMode::Auto if !(st.closed && !st.consistent_orientation) => {
                            SignMode::Pseudonormal
                        }
                        SignMode::Auto | SignMode::Winding => SignMode::Winding,
                    };
                    if st.mode == SignMode::Winding {
                        self.moments = Some(MomentsBuild::new(bvh)?);
                        self.stage = Stage::Moments;
                    } else {
                        self.stage = Stage::VertexNormals(0);
                    }
                }
                Stage::Moments => {
                    let moments = self.moments.as_mut().ok_or(FieldsError::OutOfMemory)?;
                    let n = moments.left().min(left);
                    let done = moments.step(mesh, bvh, left)?;
                    left -= n.min(left);
                    if done {
                        let moments = self.moments.take().ok_or(FieldsError::OutOfMemory)?;
                        self.data.moments = moments.finish();
                        self.face = Vec::new();
                        self.stage = Stage::Done;
                    }
                }
                Stage::VertexNormals(from) => {
                    let w = VERTEX_NORMAL_WEIGHT;
                    let r = span(from, mesh.vertex_slots(), (left / w).max(1));
                    let (to, n) = (r.end, r.len());
                    let face = &self.face;
                    let total = mesh.vertex_slots();
                    extend_span(&mut self.data.vertex_normals, r, total, |v| {
                        vertex_normal(mesh, inc, face, v)
                    })?;
                    left -= n.saturating_mul(w).min(left);
                    self.stage = if to == mesh.vertex_slots() {
                        Stage::EdgeNormals(0)
                    } else {
                        Stage::VertexNormals(to)
                    };
                }
                Stage::EdgeNormals(from) => {
                    let w = EDGE_NORMAL_WEIGHT;
                    let r = span(from, 3 * mesh.triangles().len(), (left / w).max(1));
                    let (to, n) = (r.end, r.len());
                    let face = &self.face;
                    let total = 3 * mesh.triangles().len();
                    extend_span(&mut self.data.edge_normals, r, total, |c| {
                        edge_normal(mesh, adj, inc, face, c)
                    })?;
                    left -= n.saturating_mul(w).min(left);
                    if to < 3 * mesh.triangles().len() {
                        self.stage = Stage::EdgeNormals(to);
                        continue;
                    }
                    try_push(&mut self.data.offsets, 0)?;
                    self.stage = Stage::Degenerate(0);
                }
                Stage::Degenerate(from) => {
                    let r = span(from, mesh.triangle_slots(), left);
                    let (to, n) = (r.end, r.len());
                    let SignData {
                        degenerate,
                        offsets,
                        candidates,
                        ..
                    } = &mut self.data;
                    let face = &self.face;
                    for t in r {
                        let Some(tri) = mesh.triangle(t as TriangleId) else {
                            continue;
                        };
                        if face[t] != [0.0; 3] {
                            continue;
                        }
                        let start = candidates.len();
                        for v in tri {
                            for &s in inc.triangles(v) {
                                if face[s as usize] != [0.0; 3] {
                                    try_push(candidates, s)?;
                                }
                            }
                        }
                        candidates[start..].sort_unstable();
                        let mut kept = start;
                        for i in start..candidates.len() {
                            if kept == start || candidates[i] != candidates[kept - 1] {
                                candidates[kept] = candidates[i];
                                kept += 1;
                            }
                        }
                        candidates.truncate(kept);
                        try_push(degenerate, t as TriangleId)?;
                        try_push(offsets, candidates.len() as u32)?;
                    }
                    left -= n.min(left);
                    if to == mesh.triangle_slots() {
                        self.face = Vec::new();
                        self.stage = Stage::Done;
                    } else {
                        self.stage = Stage::Degenerate(to);
                    }
                }
                Stage::Done => return Ok(true),
            }
            if self.stage == Stage::Done {
                return Ok(true);
            }
        }
        Ok(self.stage == Stage::Done)
    }

    /// The sign data, once [`SignBuild::step`] returned `true`; `None`
    /// before.
    pub fn finish(self) -> Option<SignData> {
        (self.stage == Stage::Done).then_some(self.data)
    }
}

impl SignData {
    /// Builds the sign data of `mesh` in `mode`, building the mesh's adjacency
    /// and incidence first if they are not cached. `Auto` is resolved here
    /// (to `Winding` for a closed mesh with inconsistent orientation, to
    /// `Pseudonormal` otherwise); [`SignData::status`] reports the outcome.
    ///
    /// # Errors
    /// - [`FieldsError::Query`] with `MeshMismatch` or `StaleBvh` when `bvh`
    ///   does not match the current `mesh`.
    /// - [`FieldsError::OutOfMemory`] or [`FieldsError::Store`] when the
    ///   adjacency, incidence, normals or winding moments cannot be built.
    pub fn build(mesh: &mut Mesh, bvh: &Bvh, mode: SignMode) -> Result<SignData> {
        let mut build = SignBuild::new(mesh, bvh, mode)?;
        while !build.step(mesh, bvh, usize::MAX)? {}
        build.finish().ok_or(FieldsError::OutOfMemory)
    }

    /// The generalized winding number of `mesh` at `p`: the sum of the solid
    /// angles of its triangles seen from `p`, over 4π. It is 1 inside a
    /// closed, outward-facing mesh, 0 outside it, and in between near holes.
    ///
    /// In [`SignMode::Winding`] this is the fast approximation of Barill et
    /// al. 2018, order 1: a BVH node whose centroid is more than `beta = 8`
    /// times its radius away from `p` contributes its dipole term, and nearer
    /// leaves sum their triangles' exact solid angles (Van Oosterom–Strackee).
    /// It is within 1e-3 of the exact value on the closed test meshes at
    /// points farther than 1% of the bounding-box diagonal from the surface;
    /// on a flat terrain the error is larger (about 1.2e-3, tested against
    /// 2e-3), since the errors of a flat sheet add up. In
    /// [`SignMode::Pseudonormal`] no moments are stored and every triangle is
    /// summed exactly, which is slow on large meshes.
    ///
    /// Returns NaN when this data is not current for `mesh` and `bvh` (see
    /// [`SignData::is_current`]). Nothing is allocated.
    pub fn winding_number(&self, mesh: &Mesh, bvh: &Bvh, p: [f32; 3]) -> f64 {
        if !self.is_current(mesh, bvh) {
            return f64::NAN;
        }
        self.moments.winding(mesh, bvh, p).0
    }

    /// Whether `p`, whose closest point on `mesh` is `hit`, counts as inside
    /// (negative distance). `mesh` and `bvh` must be those this data is
    /// current for.
    pub(crate) fn is_inside(&self, mesh: &Mesh, bvh: &Bvh, p: [f32; 3], hit: &ClosestHit) -> bool {
        match self.status.mode {
            SignMode::Winding => self.moments.winding(mesh, bvh, p).0 >= 0.5,
            _ => self.side(mesh, p, hit) < 0.0,
        }
    }

    /// What the build found out about the mesh, and the resolved mode.
    pub fn status(&self) -> SignStatus {
        self.status
    }

    /// The resolved mode: never [`SignMode::Auto`].
    pub fn mode(&self) -> SignMode {
        self.status.mode
    }

    /// True when this data was built for `mesh` at its current revision and
    /// epoch, and for `bvh` at its current generation.
    pub fn is_current(&self, mesh: &Mesh, bvh: &Bvh) -> bool {
        mesh.uid() == self.mesh_uid
            && bvh.mesh_uid() == self.mesh_uid
            && mesh.epoch() == self.epoch
            && mesh.revision() == self.revision
            && bvh.generation() == self.bvh_generation
    }

    /// Which side of the surface `p` lies on, given its closest point `hit`:
    /// negative behind the surface, positive in front, zero when undecided.
    /// `mesh` must be the mesh this data is current for.
    pub(crate) fn side(&self, mesh: &Mesh, p: [f32; 3], hit: &ClosestHit) -> f64 {
        let q = widen(p);
        let t = hit.triangle;
        let tri = mesh.triangles()[t as usize];
        let n = triangle_normal(corners(mesh, tri));
        if n != [0.0; 3] {
            return self.side_at(q, widen(hit.point), t, tri, n, hit.feature);
        }
        // A zero-area triangle: use the nearest non-degenerate feature around it.
        let Ok(i) = self.degenerate.binary_search(&t) else {
            return 0.0;
        };
        let range = self.offsets[i] as usize..self.offsets[i + 1] as usize;
        let mut best: Option<(f64, TriangleId, [f64; 3], Feature)> = None;
        for &s in &self.candidates[range] {
            let p = corners(mesh, mesh.triangles()[s as usize]).map(widen);
            let (d2, x, f) = closest_on_triangle(q, p);
            // Candidates ascend, so a tie keeps the smaller id.
            if best.is_none_or(|b| d2 < b.0) {
                best = Some((d2, s, x, f));
            }
        }
        let Some((_, s, x, f)) = best else {
            return 0.0;
        };
        let stri = mesh.triangles()[s as usize];
        let n = triangle_normal(corners(mesh, stri));
        self.side_at(q, x, s, stri, n, f)
    }

    /// `dot(q − x, n)` with `n` the pseudonormal of `feature` of triangle `t`
    /// (vertices `tri`, unnormalized face normal `face`).
    fn side_at(
        &self,
        q: [f64; 3],
        x: [f64; 3],
        t: TriangleId,
        tri: [VertexId; 3],
        face: [f64; 3],
        feature: Feature,
    ) -> f64 {
        let n = match feature {
            Feature::Face => face,
            Feature::Edge(k) => {
                // Edge k joins corners k and k + 1, so it is opposite corner k + 2.
                widen(self.edge_normals[3 * t as usize + (k as usize + 2) % 3])
            }
            Feature::Vertex(k) => widen(self.vertex_normals[tri[k as usize] as usize]),
        };
        dot(sub(q, x), n)
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::par;

    use super::*;
    use crate::test_util::*;

    fn status_of(mut mesh: Mesh, mode: SignMode) -> Result<SignStatus> {
        let bvh = build_bvh(&mesh);
        SignData::build(&mut mesh, &bvh, mode).map(|s| s.status())
    }

    #[test]
    fn status_reports_topology() {
        let closed = SignStatus {
            mode: SignMode::Pseudonormal,
            closed: true,
            consistent_orientation: true,
            non_manifold_vertices: 0,
            degenerate_triangles: 0,
        };
        assert_eq!(status_of(cube(), SignMode::Auto), Ok(closed));
        assert_eq!(status_of(cube(), SignMode::Pseudonormal), Ok(closed));
        assert_eq!(status_of(torus(), SignMode::Auto), Ok(closed));

        // Open: a cube with its top face removed, and a terrain sheet.
        let (p, mut t) = cube_buffers();
        t.drain(2..4);
        let open = SignStatus {
            closed: false,
            ..closed
        };
        assert_eq!(status_of(mesh(&p, &t), SignMode::Auto), Ok(open));
        assert_eq!(status_of(terrain(4), SignMode::Auto), Ok(open));

        // Two cubes meeting at one vertex: closed, one non-manifold vertex.
        let (mut p, mut t) = cube_buffers();
        let (p2, t2) = cube_buffers();
        // Corner 0 of the second cube lands on corner 7 of the first.
        p.extend(p2[1..].iter().map(|q| q.map(|c| c + 2.0)));
        t.extend(
            t2.iter()
                .map(|tri| tri.map(|v| if v == 0 { 7 } else { v + 7 })),
        );
        assert_eq!(
            status_of(mesh(&p, &t), SignMode::Auto),
            Ok(SignStatus {
                non_manifold_vertices: 1,
                ..closed
            })
        );

        // A zero-area triangle with a repeated vertex: it makes edge 0-1 non-manifold.
        let (p, mut t) = cube_buffers();
        t.push([0, 0, 1]);
        assert_eq!(
            status_of(mesh(&p, &t), SignMode::Auto),
            Ok(SignStatus {
                closed: false,
                non_manifold_vertices: 2,
                degenerate_triangles: 1,
                ..closed
            })
        );

        // A cube with one triangle flipped: open meshes keep pseudonormals,
        // closed ones fall back to winding numbers.
        let (p, mut t) = cube_buffers();
        t[0].swap(1, 2);
        let flipped = SignStatus {
            consistent_orientation: false,
            ..closed
        };
        assert_eq!(status_of(mesh(&p, &t), SignMode::Pseudonormal), Ok(flipped));
        assert_eq!(
            status_of(mesh(&p, &t), SignMode::Auto),
            Ok(SignStatus {
                mode: SignMode::Winding,
                ..flipped
            })
        );
        t.truncate(11);
        assert_eq!(
            status_of(mesh(&p, &t), SignMode::Auto),
            Ok(SignStatus {
                closed: false,
                ..flipped
            })
        );
        assert_eq!(
            status_of(cube(), SignMode::Winding),
            Ok(SignStatus {
                mode: SignMode::Winding,
                ..closed
            })
        );
    }

    /// The status and every number the sign data holds, as bits.
    fn fingerprint(s: &SignData) -> (SignStatus, Vec<u64>) {
        let normals = s.vertex_normals.iter().chain(&s.edge_normals).flatten();
        let ids = s.degenerate.iter().chain(&s.offsets).chain(&s.candidates);
        let v = normals
            .map(|x| u64::from(x.to_bits()))
            .chain(ids.map(|&i| u64::from(i)))
            .chain(s.moments.bits())
            .collect();
        (s.status, v)
    }

    #[test]
    fn sign_build_bit_identical() {
        // The forced-sequential run is the code a feature-off build runs.
        let mut mesh = terrain(317);
        assert!(mesh.triangle_count() >= 200_000);
        let bvh = build_bvh(&mesh);
        for mode in [SignMode::Pseudonormal, SignMode::Winding] {
            let build = |mesh: &mut Mesh| fingerprint(&SignData::build(mesh, &bvh, mode).unwrap());
            par::set_force_sequential(true);
            let reference = build(&mut mesh);
            par::set_force_sequential(false);
            assert!(reference.1.len() > 200_000, "{mode:?}: nothing built");
            if !par::enabled() {
                // Every build below would be this one again.
                continue;
            }
            for k in [1, 2, 8] {
                let got = par::with_threads(k, || build(&mut mesh));
                assert!(got == reference, "{mode:?}: {k} threads");
            }
            let before = par::parallel_entries();
            assert!(build(&mut mesh) == reference, "{mode:?}: global pool");
            assert!(
                par::parallel_entries() > before,
                "{mode:?}: ran sequentially"
            );
        }
    }

    #[test]
    fn small_inputs_stay_sequential() {
        let before = par::parallel_entries();
        for mode in [SignMode::Pseudonormal, SignMode::Winding] {
            for mesh in [cube(), icosphere(), torus(), terrain(16)] {
                status_of(mesh, mode).unwrap();
            }
        }
        assert_eq!(par::parallel_entries(), before);
    }

    #[test]
    fn is_current_tracks_revision_and_bvh_generation() {
        let mut mesh = cube();
        let mut bvh = build_bvh(&mesh);
        let sign = SignData::build(&mut mesh, &bvh, SignMode::Auto).unwrap();
        assert!(sign.is_current(&mesh, &bvh));
        assert_eq!(sign.mode(), SignMode::Pseudonormal);

        // Another mesh, or its BVH.
        let other = cube();
        let other_bvh = build_bvh(&other);
        assert!(!sign.is_current(&other, &other_bvh));
        assert!(!sign.is_current(&mesh, &other_bvh));

        // An edit changes the revision; updating the BVH does not help.
        let mut tx = mesh.begin_edit();
        tx.set_positions(&[0], &[[-1.0, -1.0, -1.5]]).unwrap();
        let patch = tx.commit().unwrap();
        assert!(!sign.is_current(&mesh, &bvh));
        bvh.update(&mut mesh, patch.changes()).unwrap();
        assert!(!sign.is_current(&mesh, &bvh));
        let sign = SignData::build(&mut mesh, &bvh, SignMode::Auto).unwrap();
        assert!(sign.is_current(&mesh, &bvh));

        // Restructuring the BVH changes its generation.
        bvh.rebuild(&mesh).unwrap();
        assert!(!sign.is_current(&mesh, &bvh));
        let sign = SignData::build(&mut mesh, &bvh, SignMode::Auto).unwrap();
        assert!(sign.is_current(&mesh, &bvh));

        // Compaction changes the epoch.
        let mut tx = mesh.begin_edit();
        tx.remove_triangles(&[0]).unwrap();
        tx.commit().unwrap();
        mesh.compact().unwrap();
        let bvh = build_bvh(&mesh);
        assert!(!sign.is_current(&mesh, &bvh));
        let sign = SignData::build(&mut mesh, &bvh, SignMode::Auto).unwrap();
        assert!(sign.is_current(&mesh, &bvh));
    }

    /// The meshes and modes of the stepped-build tests: both modes, slivers,
    /// a repeated vertex, an inconsistent closed mesh (`Auto` resolves to
    /// winding) and dead triangles.
    fn stepped_cases() -> Vec<(&'static str, Mesh, SignMode)> {
        let (p, mut t) = cube_buffers();
        t[0].swap(1, 2);
        let flipped = mesh(&p, &t);
        let flipped_pn = mesh(&p, &t);
        let (p, mut t) = cube_buffers();
        t.push([0, 0, 1]);
        let repeated = mesh(&p, &t);
        // A terrain with a collinear sliver and some triangles removed.
        let mut sliver = terrain(24);
        let mut tx = sliver.begin_edit();
        let v = tx
            .add_vertices(&[[0.5, 0.0, terrain_height(0.5, 0.0)]])
            .unwrap()[0];
        tx.add_triangles(&[[0, v, 1]]).unwrap();
        tx.remove_triangles(&[30, 31, 200]).unwrap();
        tx.commit().unwrap();
        vec![
            ("torus, auto", torus(), SignMode::Auto),
            ("flipped cube, auto", flipped, SignMode::Auto),
            (
                "flipped cube, pseudonormal",
                flipped_pn,
                SignMode::Pseudonormal,
            ),
            ("repeated vertex", repeated, SignMode::Pseudonormal),
            (
                "sliver terrain, pseudonormal",
                sliver,
                SignMode::Pseudonormal,
            ),
            ("terrain, winding", terrain(20), SignMode::Winding),
        ]
    }

    /// A copy of `m` with nothing cached (no adjacency, no incidence) and
    /// the same slots, dead ones included.
    fn fresh_copy(m: &Mesh) -> Mesh {
        let mut c = mesh(m.positions(), m.triangles());
        let dead: Vec<TriangleId> = (0..m.triangle_slots() as TriangleId)
            .filter(|&t| !m.is_triangle_alive(t))
            .collect();
        if !dead.is_empty() {
            let mut tx = c.begin_edit();
            tx.remove_triangles(&dead).unwrap();
            tx.commit().unwrap();
        }
        c
    }

    /// Steps `build` to the end with `budget`; returns the number of steps.
    fn step_all(build: &mut SignBuild, mesh: &mut Mesh, bvh: &Bvh, budget: usize) -> usize {
        let mut steps = 1;
        while !build.step(mesh, bvh, budget).unwrap() {
            steps += 1;
        }
        steps
    }

    #[test]
    fn stepped_build_matches_one_call() {
        for (name, base, mode) in stepped_cases() {
            // The adjacency and incidence are built inside the stepped
            // builds: each starts from a fresh copy of the mesh.
            let mut m = fresh_copy(&base);
            let bvh = build_bvh(&m);
            let reference = fingerprint(&SignData::build(&mut m, &bvh, mode).unwrap());
            for budget in [1, 2, 7, 100, 5000, usize::MAX] {
                let mut m = fresh_copy(&base);
                let bvh = build_bvh(&m);
                let mut build = SignBuild::new(&m, &bvh, mode).unwrap();
                let steps = step_all(&mut build, &mut m, &bvh, budget);
                assert!(build.is_done(), "{name}");
                let got = fingerprint(&build.finish().unwrap());
                assert!(got == reference, "{name}: budget {budget}");
                if budget == 1 {
                    assert!(steps > m.triangle_slots(), "{name}: {steps} steps of one");
                }
            }
        }
    }

    #[test]
    fn stepped_build_refuses_another_state() {
        let mut m = terrain(8);
        let bvh = build_bvh(&m);
        let mut build = SignBuild::new(&m, &bvh, SignMode::Pseudonormal).unwrap();
        assert!(!build.step(&mut m, &bvh, 3).unwrap());
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[0.0, 0.0, 1.0]]).unwrap();
        tx.commit().unwrap();
        assert!(!build.is_for(&m, &bvh));
        assert!(matches!(
            build.step(&mut m, &bvh, 3),
            Err(FieldsError::InvalidInput(_))
        ));
        assert!(build.finish().is_none());
        let other = terrain(8);
        assert!(matches!(
            SignBuild::new(&other, &bvh, SignMode::Auto),
            Err(FieldsError::Query(QueryError::MeshMismatch))
        ));
    }

    #[test]
    fn stepped_build_bit_identical_in_parallel() {
        // Steps long enough to run their passes on the pool, and short ones.
        let base = terrain(317);
        for mode in [SignMode::Pseudonormal, SignMode::Winding] {
            par::set_force_sequential(true);
            let mut m = fresh_copy(&base);
            let bvh = build_bvh(&m);
            let reference = fingerprint(&SignData::build(&mut m, &bvh, mode).unwrap());
            par::set_force_sequential(false);
            for budget in [3_000, 40_000, 1_000_000] {
                for k in [1, 8] {
                    let got = par::with_threads(k, || {
                        let mut m = fresh_copy(&base);
                        let bvh = build_bvh(&m);
                        let mut build = SignBuild::new(&m, &bvh, mode).unwrap();
                        step_all(&mut build, &mut m, &bvh, budget);
                        fingerprint(&build.finish().unwrap())
                    });
                    assert!(got == reference, "{mode:?}: budget {budget}, {k} threads");
                }
            }
        }
    }
}
