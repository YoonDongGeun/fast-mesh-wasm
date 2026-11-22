//! Convex hulls of point sets, meshes and point clouds.
//!
//! [`convex_hull`] is a quickhull in which every decision is the sign of an
//! exact predicate (`robust::orient3d`, and `orient2d` for collinearity),
//! with no epsilon anywhere. So the hull is exactly right for the given `f64`
//! coordinates:
//!
//! - every input point is inside the hull or on its boundary;
//! - the hull is convex: every vertex is on or behind every face's plane;
//! - it is closed and 2-manifold: every edge has exactly two faces, which use
//!   it in opposite directions;
//! - every face is wound counter-clockwise seen from outside, so its normal
//!   points out and the enclosed volume is positive;
//! - no face is degenerate (every face has a non-zero area).
//!
//! The predicates are exact while their products neither overflow nor
//! underflow: coordinates beyond `1e100` in magnitude are rejected, and
//! points closer together than about `1e-90` are not told apart reliably.
//!
//! Duplicate points, points inside, and points on faces or edges are fine. A
//! point set without volume (all points coincident, collinear or coplanar) is
//! [`OpsError::InvalidInput`], with a message that says which.
//!
//! Hull vertices are input points, and every corner of the hull is one. A
//! point that is on the final boundary but not a corner (on a flat face or
//! along an edge) is usually left out; it stays a vertex only when it was
//! outside the hull built so far at the moment it was added, and then the
//! flat region around it is triangulated with it. Its faces are coplanar with
//! their neighbours, so the shape is the same.
//!
//! The result depends only on the input: the points' coordinates and order.
//! With the `parallel` feature, large batches of points are classified
//! against new faces on rayon's pool, in fixed chunks, so the hull is
//! bit-identical with and without the feature, for any thread count.
//!
//! [`convex_hull_mesh`], [`convex_hull_meshes`] and
//! [`convex_hull_point_cloud`] wrap it for meshes and point clouds and return
//! a new [`Mesh`].

use fast_mesh_store::{Mesh, StoreError, VertexId, par};
use robust::{Coord, Coord3D, orient2d, orient3d};

use crate::append::{AppendSource, append_mesh};
use crate::inspect::PointCloud;
use crate::{OpsError, Result, filled, push, reserve};

/// No face or point.
const NONE: u32 = u32::MAX;
/// Points per chunk when a batch is classified against new faces.
const CHUNK: usize = 4096;
/// The smallest batch classified, or point set scanned, on rayon's pool.
const PAR_MIN: usize = 1 << 15;
/// Points per chunk when the whole point set is scanned.
const SCAN_CHUNK: usize = 1 << 14;
/// The largest coordinate magnitude accepted: the predicates' intermediate
/// products stay finite, which their exactness needs.
const MAX_COORD: f64 = 1e100;

/// The convex hull of a point set, as indices into it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConvexHull {
    /// The input index of each hull vertex, ascending.
    pub vertices: Vec<u32>,
    /// The faces, indexed into [`ConvexHull::vertices`], each wound
    /// counter-clockwise seen from outside.
    pub triangles: Vec<[u32; 3]>,
}

impl ConvexHull {
    /// The centre of the box around the hull's vertices among `points` (the
    /// point set the hull was built from).
    pub fn center(&self, points: &[[f64; 3]]) -> [f64; 3] {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for &v in &self.vertices {
            let p = points[v as usize];
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        if self.vertices.is_empty() {
            return [0.0; 3];
        }
        std::array::from_fn(|k| 0.5 * (min[k] + max[k]))
    }

    /// The hull as geometry to append to a mesh whose local frame has its
    /// zero at `origin`: each vertex at `points[v] - origin`, as `f32`. With
    /// `origin` zero and `f32` input points, the positions are exact.
    ///
    /// # Errors
    /// [`OpsError::InvalidInput`] when an index is out of range or a position
    /// does not fit an `f32`; [`OpsError::Store`] when out of memory.
    pub fn to_append_source(&self, points: &[[f64; 3]], origin: [f64; 3]) -> Result<AppendSource> {
        let mut positions = Vec::new();
        reserve(&mut positions, self.vertices.len())?;
        for &v in &self.vertices {
            let p = points
                .get(v as usize)
                .ok_or(OpsError::InvalidInput("convex hull vertex is out of range"))?;
            let local: [f32; 3] = std::array::from_fn(|k| (p[k] - origin[k]) as f32);
            if !local.iter().all(|c| c.is_finite()) {
                return Err(OpsError::InvalidInput(
                    "convex hull vertex does not fit an f32 position",
                ));
            }
            positions.push(local);
        }
        let mut triangles = Vec::new();
        reserve(&mut triangles, self.triangles.len())?;
        triangles.extend_from_slice(&self.triangles);
        Ok(AppendSource {
            positions,
            triangles,
        })
    }

    /// The hull as a new mesh with origin `origin`, its vertices at
    /// `points[v] - origin` locally (see [`ConvexHull::to_append_source`]).
    ///
    /// # Errors
    /// As for [`ConvexHull::to_append_source`], and when `origin` is not
    /// finite.
    pub fn to_mesh(&self, points: &[[f64; 3]], origin: [f64; 3]) -> Result<Mesh> {
        let source = self.to_append_source(points, origin)?;
        mesh_with_origin(&source, origin)
    }
}

/// A new mesh with origin `origin` holding exactly `source`'s local
/// positions and triangles.
fn mesh_with_origin(source: &AppendSource, origin: [f64; 3]) -> Result<Mesh> {
    let mut mesh = Mesh::from_buffers_f64_with_origin(&[], &[], origin)?;
    let mut tx = mesh.begin_edit();
    append_mesh(&mut tx, source)?;
    tx.commit()?;
    Ok(mesh)
}

/// Appends to `out`, mapped by `map`, the local position of every live vertex
/// of `mesh` that a live triangle uses, in vertex slot order: the points a
/// mesh contributes to a hull. Returns how many were appended.
///
/// # Errors
/// [`OpsError::Store`] when out of memory.
pub fn push_hull_points(
    mesh: &Mesh,
    mut map: impl FnMut([f32; 3]) -> [f64; 3],
    out: &mut Vec<[f64; 3]>,
) -> Result<usize> {
    let before = out.len();
    reserve(out, mesh.vertex_count())?;
    for (v, p) in mesh.positions().iter().enumerate() {
        let v = v as VertexId;
        if mesh.is_vertex_alive(v) && mesh.corner_count(v) > 0 {
            out.push(map(*p));
        }
    }
    Ok(out.len() - before)
}

/// The convex hull of `mesh`'s vertices (the live vertices a live triangle
/// uses) as a new mesh with the same origin. Its vertices have exactly the
/// mesh's local positions.
///
/// # Errors
/// As for [`convex_hull`].
pub fn convex_hull_mesh(mesh: &Mesh) -> Result<Mesh> {
    convex_hull_meshes(&[mesh])
}

/// The convex hull of the vertices of `meshes` (the live vertices a live
/// triangle uses), which share one frame: a vertex is at its mesh's origin
/// plus its local position.
///
/// When every mesh has the same origin, the hull is computed on the local
/// positions, and the new mesh has that origin and exactly those positions.
/// Otherwise it is computed on origin plus local position in `f64`, and the
/// new mesh's origin is the centre of its vertices' box.
///
/// # Errors
/// [`OpsError::InvalidInput`] for an empty list, and as for [`convex_hull`].
pub fn convex_hull_meshes(meshes: &[&Mesh]) -> Result<Mesh> {
    let Some(first) = meshes.first() else {
        return Err(OpsError::InvalidInput(
            "convex hull needs at least one mesh",
        ));
    };
    let origin = first.origin();
    let shared = meshes.iter().all(|m| m.origin() == origin);
    let mut points = Vec::new();
    for mesh in meshes {
        let o = mesh.origin();
        if shared {
            push_hull_points(mesh, |p| p.map(f64::from), &mut points)?;
        } else {
            push_hull_points(
                mesh,
                |p| std::array::from_fn(|k| o[k] + f64::from(p[k])),
                &mut points,
            )?;
        }
    }
    let hull = convex_hull(&points)?;
    if shared {
        mesh_with_origin(&hull.to_append_source(&points, [0.0; 3])?, origin)
    } else {
        let center = hull.center(&points);
        hull.to_mesh(&points, center)
    }
}

/// The convex hull of a point cloud's positions as a new mesh at the origin,
/// its vertices at exactly the cloud's positions.
///
/// # Errors
/// As for [`convex_hull`].
pub fn convex_hull_point_cloud(cloud: &PointCloud) -> Result<Mesh> {
    let mut points = Vec::new();
    reserve(&mut points, cloud.positions.len())?;
    points.extend(cloud.positions.iter().map(|p| p.map(f64::from)));
    convex_hull(&points)?.to_mesh(&points, [0.0; 3])
}

/// The convex hull of `points`: a closed, outward-wound triangle mesh with
/// input points as vertices (see the [module docs](self) for what is
/// guaranteed).
///
/// # Errors
/// [`OpsError::InvalidInput`] with fewer than four points, `u32::MAX` points
/// or more, a coordinate that is not finite or exceeds `1e100` in magnitude,
/// or a point set that has no volume: every point the same, on one line, or
/// on one plane. [`OpsError::Store`] when out of memory.
pub fn convex_hull(points: &[[f64; 3]]) -> Result<ConvexHull> {
    if points.len() < 4 {
        return Err(OpsError::InvalidInput(
            "convex hull needs at least four points",
        ));
    }
    if points.len() >= NONE as usize {
        return Err(OpsError::InvalidInput("convex hull has too many points"));
    }
    let ok = scan(points, |r| {
        points[r]
            .iter()
            .flatten()
            .all(|c| c.is_finite() && c.abs() <= MAX_COORD)
    })?;
    if !ok.iter().all(|&ok| ok) {
        return Err(OpsError::InvalidInput(
            "convex hull coordinates must be finite and at most 1e100 in magnitude",
        ));
    }
    let simplex = initial_simplex(points)?;
    let mut hull = Builder::new(points, simplex)?;
    hull.run()?;
    hull.finish()
}

#[inline]
fn coord(p: &[f64; 3]) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

/// Exact: negative when `d` is strictly above the plane of `(a, b, c)` (on
/// the side from which they appear counter-clockwise), zero on it.
#[inline]
fn side(a: Coord3D<f64>, b: Coord3D<f64>, c: Coord3D<f64>, d: &[f64; 3]) -> f64 {
    orient3d(a, b, c, coord(d))
}

fn oom() -> OpsError {
    OpsError::Store(StoreError::OutOfMemory)
}

/// `f` over `points` cut into fixed chunks (on rayon's pool when there are
/// many), in chunk order.
fn scan<R: Send>(
    points: &[[f64; 3]],
    f: impl Fn(std::ops::Range<usize>) -> R + Sync,
) -> Result<Vec<R>> {
    crate::allocation_allowed()?;
    par::map_range(points.len(), SCAN_CHUNK, PAR_MIN, oom, |r| Ok(f(r)))
}

/// The lowest index with the largest positive `key`, with its key, or `None`
/// when no key is positive. The same as one pass in order, however chunked.
fn argmax(
    points: &[[f64; 3]],
    key: impl Fn(&[f64; 3]) -> f64 + Sync,
) -> Result<Option<(usize, f64)>> {
    let better = |best: Option<(usize, f64)>, k: f64| k > best.map_or(0.0, |b| b.1);
    let chunks = scan(points, |r| {
        let mut best = None;
        for i in r {
            let k = key(&points[i]);
            if better(best, k) {
                best = Some((i, k));
            }
        }
        best
    })?;
    let mut best = None;
    for (i, k) in chunks.into_iter().flatten() {
        if better(best, k) {
            best = Some((i, k));
        }
    }
    Ok(best)
}

fn dist2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]) * (a[k] - b[k])).sum()
}

/// Exact: whether `a`, `b` and `c` are on one line (every projection onto a
/// coordinate plane is collinear).
fn collinear(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> bool {
    [(0, 1), (1, 2), (2, 0)].iter().all(|&(i, j)| {
        let p = |q: &[f64; 3]| Coord { x: q[i], y: q[j] };
        orient2d(p(a), p(b), p(c)) == 0.0
    })
}

/// Four points spanning a tetrahedron, far apart: the two farthest of the
/// axis extremes, the point farthest from their line, and the point farthest
/// from the plane of the three. Ties go to the lower index.
fn initial_simplex(pts: &[[f64; 3]]) -> Result<[u32; 4]> {
    // Lowest and highest point on each axis: per chunk, then over the
    // chunks in order, so ties go to the lowest index either way.
    let extremes = |ext: &mut [usize; 6], i: usize| {
        let p = &pts[i];
        for k in 0..3 {
            if p[k] < pts[ext[2 * k]][k] {
                ext[2 * k] = i;
            }
            if p[k] > pts[ext[2 * k + 1]][k] {
                ext[2 * k + 1] = i;
            }
        }
    };
    let chunks = scan(pts, |r| {
        let mut ext = [r.start; 6];
        for i in r {
            extremes(&mut ext, i);
        }
        ext
    })?;
    let mut ext = [0usize; 6];
    for chunk in chunks {
        for (k, &i) in chunk.iter().enumerate() {
            let axis = k / 2;
            let (p, q) = (pts[i][axis], pts[ext[k]][axis]);
            if (k % 2 == 0 && p < q) || (k % 2 == 1 && p > q) {
                ext[k] = i;
            }
        }
    }
    let mut pair = (ext[0], ext[1]);
    let mut best = -1.0;
    for i in 0..6 {
        for j in i + 1..6 {
            let d = dist2(&pts[ext[i]], &pts[ext[j]]);
            if d > best {
                best = d;
                pair = (ext[i], ext[j]);
            }
        }
    }
    if pts[pair.0] == pts[pair.1] {
        // Every squared distance was zero: either every point is the same,
        // or the differences are too small to square in f64.
        match (0..3).find(|&k| pts[ext[2 * k]][k] < pts[ext[2 * k + 1]][k]) {
            Some(k) => pair = (ext[2 * k], ext[2 * k + 1]),
            None => return Err(OpsError::InvalidInput("convex hull points all coincide")),
        }
    }
    let (a, b) = pair;
    let (pa, pb) = (&pts[a], &pts[b]);

    // Farthest from the line, by the cross product's approximate length,
    // checked exactly; when rounding hid every off-line point, the first
    // exactly off-line point.
    let ab: [f64; 3] = std::array::from_fn(|k| pb[k] - pa[k]);
    let far = argmax(pts, |q| {
        let aq: [f64; 3] = std::array::from_fn(|k| q[k] - pa[k]);
        let x = [
            ab[1] * aq[2] - ab[2] * aq[1],
            ab[2] * aq[0] - ab[0] * aq[2],
            ab[0] * aq[1] - ab[1] * aq[0],
        ];
        x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
    })?;
    let mut c = far.map_or(a, |(i, _)| i);
    if collinear(pa, pb, &pts[c]) {
        c = (0..pts.len())
            .find(|&i| !collinear(pa, pb, &pts[i]))
            .ok_or(OpsError::InvalidInput("convex hull points are collinear"))?;
    }
    let pc = &pts[c];

    // Farthest from the plane, by the exact predicate's approximate value.
    let (ca, cb, cc) = (coord(pa), coord(pb), coord(pc));
    let (d, _) = argmax(pts, |q| side(ca, cb, cc, q).abs())?
        .ok_or(OpsError::InvalidInput("convex hull points are coplanar"))?;
    Ok([a as u32, b as u32, c as u32, d as u32])
}

/// A face of the hull being built.
#[derive(Debug, Clone, Copy)]
struct Face {
    /// Point indices, counter-clockwise seen from outside.
    v: [u32; 3],
    /// `adj[i]` is the face across the edge `v[i] -> v[(i + 1) % 3]`.
    adj: [u32; 3],
    /// The first point of the outside set (linked through `Builder::next`).
    head: u32,
    /// The outside point farthest from the plane.
    far: u32,
    /// The predicate's magnitude for `far`.
    far_d: f64,
    /// Visibility stamp of the current step.
    mark: u32,
    alive: bool,
}

impl Face {
    fn new(v: [u32; 3]) -> Face {
        Face {
            v,
            adj: [NONE; 3],
            head: NONE,
            far: NONE,
            far_d: -1.0,
            mark: 0,
            alive: true,
        }
    }
}

/// A point of a batch and the face it was classified to.
#[derive(Debug, Clone, Copy)]
struct Pick {
    point: u32,
    /// The first candidate face the point is strictly outside, or `NONE`.
    face: u32,
    dist: f64,
}

/// Classifies every pick against `candidates` (face ids with their corner
/// coordinates), in order: the first face the point is strictly outside.
/// Pure per pick, so chunking cannot change the result.
fn classify(pts: &[[f64; 3]], candidates: &[(u32, [Coord3D<f64>; 3])], picks: &mut [Pick]) {
    par::for_each_chunk_mut(picks, CHUNK, PAR_MIN, |_, chunk| {
        for pick in chunk {
            let q = &pts[pick.point as usize];
            pick.face = NONE;
            for &(f, [a, b, c]) in candidates {
                let o = side(a, b, c, q);
                if o < 0.0 {
                    pick.face = f;
                    pick.dist = -o;
                    break;
                }
            }
        }
    });
}

struct Builder<'a> {
    pts: &'a [[f64; 3]],
    faces: Vec<Face>,
    /// Dead face slots, reused last in first out.
    free: Vec<u32>,
    /// Outside-set links, per point.
    next: Vec<u32>,
    /// Faces that may have outside points.
    stack: Vec<u32>,
    stamp: u32,
    // Scratch, kept between steps.
    visible: Vec<u32>,
    /// `(a, b, hidden face, visible face)` per horizon edge `a -> b` of a
    /// visible face.
    horizon: Vec<(u32, u32, u32, u32)>,
    new_faces: Vec<u32>,
    /// Per point: the new face whose horizon edge starts there.
    start: Vec<u32>,
    picks: Vec<Pick>,
    candidates: Vec<(u32, [Coord3D<f64>; 3])>,
}

impl<'a> Builder<'a> {
    fn new(pts: &'a [[f64; 3]], s: [u32; 4]) -> Result<Builder<'a>> {
        let [a, mut b, mut c, d] = s;
        let corner = |i: u32| coord(&pts[i as usize]);
        if side(corner(a), corner(b), corner(c), &pts[d as usize]) < 0.0 {
            std::mem::swap(&mut b, &mut c);
        }
        // `d` is now below `(a, b, c)`, so these four are wound outward, and
        // `adj` names the face across each edge.
        let tets = [
            ([a, b, c], [1, 2, 3]),
            ([a, d, b], [3, 2, 0]),
            ([b, d, c], [1, 3, 0]),
            ([c, d, a], [2, 1, 0]),
        ];
        let mut faces = Vec::new();
        reserve(&mut faces, 4)?;
        faces.extend(tets.map(|(v, adj)| Face {
            adj,
            ..Face::new(v)
        }));
        let n = pts.len();
        let mut hull = Builder {
            pts,
            faces,
            free: Vec::new(),
            next: filled(n, NONE)?,
            stack: Vec::new(),
            stamp: 0,
            visible: Vec::new(),
            horizon: Vec::new(),
            new_faces: Vec::new(),
            start: filled(n, NONE)?,
            picks: Vec::new(),
            candidates: Vec::new(),
        };
        reserve(&mut hull.picks, n)?;
        hull.picks.extend((0..n as u32).map(|point| Pick {
            point,
            face: NONE,
            dist: 0.0,
        }));
        reserve(&mut hull.new_faces, 4)?;
        hull.new_faces.extend([0, 1, 2, 3]);
        hull.assign()?;
        Ok(hull)
    }

    /// Classifies `picks` against `new_faces`, links each outside point
    /// into its face's outside set (in pick order), and stacks the new faces
    /// that got points.
    fn assign(&mut self) -> Result<()> {
        self.candidates.clear();
        reserve(&mut self.candidates, self.new_faces.len())?;
        for &f in &self.new_faces {
            let v = self.faces[f as usize].v;
            self.candidates
                .push((f, v.map(|i| coord(&self.pts[i as usize]))));
        }
        classify(self.pts, &self.candidates, &mut self.picks);
        for pick in &self.picks {
            if pick.face == NONE {
                continue;
            }
            let face = &mut self.faces[pick.face as usize];
            self.next[pick.point as usize] = face.head;
            face.head = pick.point;
            if pick.dist > face.far_d {
                face.far = pick.point;
                face.far_d = pick.dist;
            }
        }
        self.picks.clear();
        reserve(&mut self.stack, self.new_faces.len())?;
        for &f in &self.new_faces {
            if self.faces[f as usize].head != NONE {
                self.stack.push(f);
            }
        }
        Ok(())
    }

    fn run(&mut self) -> Result<()> {
        while let Some(f) = self.stack.pop() {
            let face = &self.faces[f as usize];
            if face.alive && face.head != NONE {
                self.add_point(f)?;
            }
        }
        Ok(())
    }

    /// Adds the farthest outside point of face `f0` to the hull: removes the
    /// faces it sees, cones the horizon to it and redistributes the removed
    /// faces' outside points.
    fn add_point(&mut self, f0: u32) -> Result<()> {
        let pts = self.pts;
        let p = self.faces[f0 as usize].far;
        let pp = &pts[p as usize];
        if self.stamp >= u32::MAX - 2 {
            for face in &mut self.faces {
                face.mark = 0;
            }
            self.stamp = 0;
        }
        self.stamp += 2;
        let (seen, hidden) = (self.stamp - 1, self.stamp);

        // The faces `p` is strictly above, from `f0` across edges, and the
        // edges where they meet the others.
        self.visible.clear();
        self.horizon.clear();
        self.faces[f0 as usize].mark = seen;
        push(&mut self.visible, f0)?;
        let mut i = 0;
        while i < self.visible.len() {
            let f = self.visible[i];
            i += 1;
            for e in 0..3 {
                let face = self.faces[f as usize];
                let g = face.adj[e];
                let mark = self.faces[g as usize].mark;
                if mark == seen {
                    continue;
                }
                if mark != hidden {
                    let gv = self.faces[g as usize].v.map(|v| coord(&pts[v as usize]));
                    if side(gv[0], gv[1], gv[2], pp) < 0.0 {
                        self.faces[g as usize].mark = seen;
                        push(&mut self.visible, g)?;
                        continue;
                    }
                    self.faces[g as usize].mark = hidden;
                }
                push(&mut self.horizon, (face.v[e], face.v[(e + 1) % 3], g, f))?;
            }
        }

        // Their outside points, to redistribute.
        for &f in &self.visible {
            let mut q = self.faces[f as usize].head;
            while q != NONE {
                if q != p {
                    push(
                        &mut self.picks,
                        Pick {
                            point: q,
                            face: NONE,
                            dist: 0.0,
                        },
                    )?;
                }
                q = self.next[q as usize];
            }
        }

        // One new face per horizon edge, linked to the hidden face across it
        // and, through `start`, to the next new face around `p`.
        reserve(&mut self.faces, self.horizon.len())?;
        self.new_faces.clear();
        reserve(&mut self.new_faces, self.horizon.len())?;
        for &(a, b, g, f) in &self.horizon {
            let mut face = Face::new([a, b, p]);
            face.adj[0] = g;
            let nf = match self.free.pop() {
                Some(slot) => {
                    self.faces[slot as usize] = face;
                    slot
                }
                None => {
                    self.faces.push(face);
                    (self.faces.len() - 1) as u32
                }
            };
            let hidden_face = &mut self.faces[g as usize];
            for j in 0..3 {
                if hidden_face.adj[j] == f {
                    hidden_face.adj[j] = nf;
                }
            }
            debug_assert_eq!(
                self.start[a as usize], NONE,
                "horizon is not a simple cycle"
            );
            self.start[a as usize] = nf;
            self.new_faces.push(nf);
        }
        for &nf in &self.new_faces {
            let b = self.faces[nf as usize].v[1];
            let next = self.start[b as usize];
            debug_assert_ne!(next, NONE, "horizon is not closed");
            self.faces[nf as usize].adj[1] = next;
            self.faces[next as usize].adj[2] = nf;
        }
        for &(a, ..) in &self.horizon {
            self.start[a as usize] = NONE;
        }

        // The visible faces are gone; their slots are free after this step.
        reserve(&mut self.free, self.visible.len())?;
        for &f in &self.visible {
            let face = &mut self.faces[f as usize];
            face.alive = false;
            face.head = NONE;
            self.free.push(f);
        }
        self.assign()
    }

    fn finish(self) -> Result<ConvexHull> {
        let mut index = self.next;
        index.fill(NONE);
        let mut count = 0usize;
        for face in self.faces.iter().filter(|f| f.alive) {
            for &v in &face.v {
                index[v as usize] = 0;
            }
            count += 1;
        }
        let mut vertices = Vec::new();
        for (i, slot) in index.iter_mut().enumerate() {
            if *slot != NONE {
                *slot = vertices.len() as u32;
                push(&mut vertices, i as u32)?;
            }
        }
        let mut triangles = Vec::new();
        reserve(&mut triangles, count)?;
        for face in self.faces.iter().filter(|f| f.alive) {
            triangles.push(face.v.map(|v| index[v as usize]));
        }
        Ok(ConvexHull {
            vertices,
            triangles,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degenerate_inputs_are_errors() {
        let few = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(
            convex_hull(&few),
            Err(OpsError::InvalidInput(
                "convex hull needs at least four points"
            ))
        );
        let same = [[1.0, 2.0, 3.0]; 5];
        assert_eq!(
            convex_hull(&same),
            Err(OpsError::InvalidInput("convex hull points all coincide"))
        );
        let line: Vec<[f64; 3]> = (0..6)
            .map(|i| [f64::from(i), 2.0 * f64::from(i), 1.0])
            .collect();
        assert_eq!(
            convex_hull(&line),
            Err(OpsError::InvalidInput("convex hull points are collinear"))
        );
        let plane: Vec<[f64; 3]> = (0..16)
            .map(|i| {
                let (x, y) = (f64::from(i % 4), f64::from(i / 4));
                [x, y, x + y]
            })
            .collect();
        assert_eq!(
            convex_hull(&plane),
            Err(OpsError::InvalidInput("convex hull points are coplanar"))
        );
        let nan = [
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, f64::NAN],
        ];
        assert!(matches!(convex_hull(&nan), Err(OpsError::InvalidInput(_))));
        let huge = [
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1e200],
        ];
        assert!(matches!(convex_hull(&huge), Err(OpsError::InvalidInput(_))));
        assert!(matches!(
            convex_hull_meshes(&[]),
            Err(OpsError::InvalidInput(_))
        ));
    }

    #[test]
    fn running_out_of_memory_is_an_error() {
        let pts: Vec<[f64; 3]> = (0..200)
            .map(|i| {
                let t = f64::from(i) * 0.37;
                [t.cos() * (1.0 + 0.1 * t.sin()), t.sin(), (t * 0.5).cos()]
            })
            .collect();
        let full = convex_hull(&pts).unwrap();
        let mut failed = 0;
        for n in 0..60 {
            crate::fail_allocations_after(Some(n));
            let r = convex_hull(&pts);
            crate::fail_allocations_after(None);
            match r {
                Err(OpsError::Store(StoreError::OutOfMemory)) => failed += 1,
                Ok(h) => assert_eq!(h, full),
                Err(e) => panic!("{e}"),
            }
        }
        assert!(failed > 10, "{failed}");
    }
}
