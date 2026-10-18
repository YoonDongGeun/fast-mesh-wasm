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

use fast_mesh_bvh::Bvh;
use fast_mesh_query::{ClosestHit, Feature, QueryError};
use fast_mesh_store::{Adjacency, EdgeKind, Incidence, Mesh, TriangleId, VertexId};

use crate::geom::{
    add, closest_on_triangle, cross, dot, narrow, norm, scale, sub, triangle_normal, unit, widen,
};
use crate::winding::Moments;
use crate::{FieldsError, Result};

/// How many zero-area slivers an edge normal looks through to find the face
/// on the other side.
const SLIVER_HOPS: u32 = 4;

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

fn corners(mesh: &Mesh, tri: [VertexId; 3]) -> [[f32; 3]; 3] {
    tri.map(|v| mesh.positions()[v as usize])
}

/// Topology facts about a mesh.
fn topology(mesh: &Mesh, adj: &Adjacency, face: &[[f64; 3]]) -> (bool, bool, usize, usize) {
    let tris = mesh.triangles();
    let (mut closed, mut consistent, mut degenerate) = (true, true, 0);
    for (t, tri) in tris.iter().enumerate() {
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
    let non_manifold = (0..mesh.vertex_slots() as VertexId)
        .filter(|&v| mesh.is_vertex_alive(v) && adj.is_non_manifold_vertex(v))
        .count();
    (closed, consistent, non_manifold, degenerate)
}

/// Unit normals of the alive triangles; zero for dead and zero-area ones.
fn face_normals(mesh: &Mesh) -> Result<Vec<[f64; 3]>> {
    let mut face = try_vec(mesh.triangle_slots(), [0.0; 3])?;
    for (t, n) in face.iter_mut().enumerate() {
        if let Some(tri) = mesh.triangle(t as TriangleId) {
            *n = unit(triangle_normal(corners(mesh, tri)));
        }
    }
    Ok(face)
}

/// Angle-weighted pseudonormals of the vertex slots, from the non-degenerate
/// triangles around each vertex.
fn vertex_normals(mesh: &Mesh, inc: &Incidence, face: &[[f64; 3]]) -> Result<Vec<[f32; 3]>> {
    let mut normals = try_vec(mesh.vertex_slots(), [0.0f32; 3])?;
    for (v, out) in normals.iter_mut().enumerate() {
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
        *out = narrow(unit(sum));
    }
    Ok(normals)
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

/// Edge normals per corner: the sum of the unit normals of the faces on the
/// edge opposite each corner.
fn edge_normals(
    mesh: &Mesh,
    adj: &Adjacency,
    inc: &Incidence,
    face: &[[f64; 3]],
) -> Result<Vec<[f32; 3]>> {
    let tris = mesh.triangles();
    let mut normals = try_vec(3 * tris.len(), [0.0f32; 3])?;
    for (t, tri) in tris.iter().enumerate() {
        if !mesh.is_triangle_alive(t as TriangleId) {
            continue;
        }
        for i in 0..3 {
            let c = (3 * t + i) as u32;
            let mut sum = face[t];
            match adj.edge_kind(c) {
                EdgeKind::Interior => sum = add(sum, across(mesh, adj, face, c, SLIVER_HOPS)),
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
            normals[c as usize] = narrow(sum);
        }
    }
    Ok(normals)
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
        if bvh.mesh_uid() != mesh.uid() {
            return Err(QueryError::MeshMismatch.into());
        }
        if !bvh.is_current(mesh) {
            return Err(QueryError::StaleBvh.into());
        }
        mesh.ensure_adjacency()?;
        mesh.ensure_incidence()?;
        let mesh: &Mesh = mesh;
        let missing = FieldsError::InvalidInput("mesh adjacency or incidence unavailable");
        let adj = mesh.adjacency().ok_or(missing.clone())?;
        let inc = mesh.incidence().ok_or(missing)?;

        let face = face_normals(mesh)?;
        let (closed, consistent, non_manifold_vertices, degenerate_triangles) =
            topology(mesh, adj, &face);
        let resolved = match mode {
            SignMode::Pseudonormal => SignMode::Pseudonormal,
            SignMode::Auto if !(closed && !consistent) => SignMode::Pseudonormal,
            SignMode::Auto | SignMode::Winding => SignMode::Winding,
        };
        let mut data = SignData {
            mesh_uid: mesh.uid(),
            epoch: mesh.epoch(),
            revision: mesh.revision(),
            bvh_generation: bvh.generation(),
            status: SignStatus {
                mode: resolved,
                closed,
                consistent_orientation: consistent,
                non_manifold_vertices,
                degenerate_triangles,
            },
            vertex_normals: Vec::new(),
            edge_normals: Vec::new(),
            degenerate: Vec::new(),
            offsets: Vec::new(),
            candidates: Vec::new(),
            moments: Moments::default(),
        };
        if resolved == SignMode::Winding {
            data.moments = Moments::build(mesh, bvh)?;
            return Ok(data);
        }

        data.vertex_normals = vertex_normals(mesh, inc, &face)?;
        data.edge_normals = edge_normals(mesh, adj, inc, &face)?;

        let SignData {
            degenerate,
            offsets,
            candidates,
            ..
        } = &mut data;
        try_push(offsets, 0)?;
        for t in 0..mesh.triangle_slots() {
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
        Ok(data)
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
}
