//! Cutting a mesh with a plane.
//!
//! A vertex is on the plane's `+` side when its signed distance is zero or
//! more, so no vertex lies on the plane and each triangle with vertices on
//! both sides gives exactly one segment, between crossings strictly inside two
//! of its edges. Crossings are named by their edge key `(min id, max id)`.

use fast_mesh_bvh::{Aabb, Bvh, NodeKind};
use fast_mesh_store::{Mesh, TriangleId, VertexId};

use crate::plane::{Plane, dot};
use crate::{Appender, QueryError, Result, allocation_allowed, check};

/// The piece of the cut inside one triangle, from the crossing on edge `from`
/// to the crossing on edge `to`.
///
/// It is oriented by the triangle's winding: walking from `from` to `to`, seen
/// from the plane normal, the triangle's outward side (the one its normal
/// points to) is on the right. For a closed, outward-wound mesh this makes
/// outer contours counter-clockwise and holes clockwise in the plane's frame,
/// and the `to` of one segment is the `from` of the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Segment {
    pub(crate) tri: TriangleId,
    pub(crate) from: (VertexId, VertexId),
    pub(crate) to: (VertexId, VertexId),
}

fn key(a: VertexId, b: VertexId) -> (VertexId, VertexId) {
    (a.min(b), a.max(b))
}

fn position(mesh: &Mesh, v: VertexId) -> [f64; 3] {
    mesh.positions()[v as usize].map(f64::from)
}

/// The point where `plane` crosses the edge `key`, computed from the edge's
/// endpoints in id order so that it is the same whichever triangle asks.
///
/// The endpoints must be on opposite sides. For an edge with both endpoints
/// on the same side the result is meaningless, possibly NaN.
pub(crate) fn crossing(mesh: &Mesh, plane: &Plane, key: (VertexId, VertexId)) -> [f64; 3] {
    let (a, b) = (
        position(mesh, key.0.min(key.1)),
        position(mesh, key.0.max(key.1)),
    );
    let (sa, sb) = (plane.signed_distance(a), plane.signed_distance(b));
    debug_assert!((sa >= 0.0) != (sb >= 0.0), "edge {key:?} is not crossed");
    let t = sa / (sa - sb);
    std::array::from_fn(|k| a[k] + t * (b[k] - a[k]))
}

/// The segment of triangle `tri`, if the plane crosses it.
fn segment(mesh: &Mesh, plane: &Plane, tri: TriangleId) -> Option<Segment> {
    let t = mesh.triangles()[tri as usize];
    let plus = t.map(|v| plane.signed_distance(position(mesh, v)) >= 0.0);
    // The lone vertex, whose side differs from the other two.
    let i = if plus[0] == plus[1] {
        if plus[1] == plus[2] {
            return None;
        }
        2
    } else if plus[0] == plus[2] {
        1
    } else {
        0
    };
    // In winding order (a, b, c) from the lone vertex, the crossed edges are
    // (a, b) and (c, a). With `a` on the `+` side the cut runs from (a, b) to
    // (c, a), keeping the outward side on the right.
    let (a, b, c) = (t[i], t[(i + 1) % 3], t[(i + 2) % 3]);
    let (ab, ca) = (key(a, b), key(c, a));
    let (from, to) = if plus[i] { (ab, ca) } else { (ca, ab) };
    Some(Segment { tri, from, to })
}

/// Whether the plane may cross a triangle inside `b`: the signed distances
/// of the box corners, padded for rounding, take both signs.
fn straddles(plane: &Plane, b: &Aabb) -> bool {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut magnitude = 0.0f64;
    for i in 0..8 {
        let p: [f64; 3] =
            std::array::from_fn(|k| f64::from(if i >> k & 1 == 1 { b.max[k] } else { b.min[k] }));
        let s = plane.signed_distance(p);
        lo = lo.min(s);
        hi = hi.max(s);
        magnitude = magnitude.max(p.iter().fold(0.0, |m, c| m.max(c.abs())));
    }
    let offset = dot(plane.normal(), plane.origin()).abs();
    let pad = 8.0 * f64::from(f32::EPSILON) * (magnitude + offset);
    lo <= pad && hi >= -pad
}

/// Appends to `out` the segment of every alive triangle the plane crosses, in
/// an unspecified order. Elements already in `out` are kept.
///
/// # Errors
/// [`QueryError::MeshMismatch`](crate::QueryError::MeshMismatch) or
/// [`QueryError::StaleBvh`](crate::QueryError::StaleBvh) when `bvh` does not
/// match the current `mesh`, and
/// [`QueryError::OutOfMemory`](crate::QueryError::OutOfMemory) when `out`
/// cannot grow. On any error `out` keeps its original length.
pub(crate) fn segments(
    mesh: &Mesh,
    bvh: &Bvh,
    plane: &Plane,
    out: &mut Vec<Segment>,
) -> Result<()> {
    check(mesh, bvh)?;
    let mut ap = Appender::new(out);
    let mut failed = None;
    bvh.traverse(|_, node| {
        if failed.is_some() || !straddles(plane, &node.bounds) {
            return false;
        }
        if let NodeKind::Leaf { triangles } = node.kind {
            for &id in triangles {
                if let Some(s) = segment(mesh, plane, id)
                    && let Err(e) = ap.push(s)
                {
                    failed = Some(e);
                    return false;
                }
            }
        }
        true
    });
    failed.map_or(Ok(()), Err)
}

/// The cut of a mesh by a plane, as contours.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Slice {
    /// The contours, sorted by their first edge key.
    pub contours: Vec<Contour>,
}

/// One chain of the cut: a closed loop or an open polyline.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Contour {
    /// The crossing point of each edge in `edges`, in the mesh's local
    /// coordinates (the `f32` positions widened to `f64`, without the mesh
    /// origin), not in the plane's `(u, v)` frame: see
    /// [`Plane::to_uv`](crate::Plane::to_uv).
    pub points: Vec<[f64; 3]>,
    /// The crossed edge `(min id, max id)` of each point.
    pub edges: Vec<(VertexId, VertexId)>,
    /// The triangle of each segment: `triangles[i]` joins point `i` to point
    /// `i + 1`, and the last one of a closed contour joins the last point to
    /// the first. One fewer than the points when open, as many when closed.
    pub triangles: Vec<TriangleId>,
    /// Whether the last point joins back to the first.
    pub closed: bool,
    /// Whether every segment runs the same way along the contour, so that
    /// the triangles agree on their winding.
    pub oriented: bool,
    /// Whether no end of the contour is a split at a non-manifold edge.
    pub manifold: bool,
}

/// An end of a segment that no other end joins: the edge is a boundary.
const BOUNDARY: usize = usize::MAX;
/// An end of a segment at an edge crossed by more than two triangles.
const SPLIT: usize = usize::MAX - 1;

/// Cuts `mesh` with `plane` and writes the contours to `out`, replacing what
/// it held and reusing its allocations. The plane and the points are in the
/// mesh's stored local coordinates (the `f32` positions widened to `f64`,
/// without the mesh origin).
///
/// The triangles' segments are joined at shared crossed edges:
/// - An edge crossed by one triangle ends an open contour.
/// - An edge crossed by more than two triangles splits the cut: every piece
///   ending there is open, with `manifold == false`.
/// - Where two triangles disagree on their winding the pieces still join,
///   and the contour has `oriented == false`.
///
/// On a closed, outward-wound mesh the contours are closed, oriented loops,
/// with the solid's inside on their left seen from the plane normal: outer
/// loops run counter-clockwise in `(u, v)` and holes clockwise. An oriented
/// open contour runs the same way. A plane through vertices gives
/// coincident points joined by zero-length segments; they are kept.
///
/// A face lying exactly in the plane is cut or not depending on the sign of
/// the normal, because its vertices are all on the `+` side (`s ≥ 0`): a
/// cube cut at its top face with the normal `+z` gives the square of that
/// face (the side faces cross from `−` to `+` there), while with `−z` all of
/// the cube is on the `+` side and nothing is cut. Offset the plane slightly
/// to choose.
///
/// A zero-area triangle `[a, a, b]` crossed between `a` and `b` gives a
/// segment from edge `(a, b)` back to itself. It has no length, so it is
/// dropped rather than counted as a third triangle at that edge.
///
/// The order is deterministic: a closed contour starts at its smallest edge
/// key, an open one at its first end (its `from` end when oriented), and
/// the contours are sorted by their first edge key.
///
/// # Errors
/// [`QueryError::MeshMismatch`](crate::QueryError::MeshMismatch) or
/// [`QueryError::StaleBvh`](crate::QueryError::StaleBvh) when `bvh` does not
/// match the current `mesh`, and
/// [`QueryError::OutOfMemory`](crate::QueryError::OutOfMemory) when an
/// allocation fails. On any error `out` is left empty.
pub fn slice(mesh: &Mesh, bvh: &Bvh, plane: &Plane, out: &mut Slice) -> Result<()> {
    let r = chain(mesh, bvh, plane, &mut out.contours);
    if r.is_err() {
        out.contours.clear();
    }
    r
}

fn chain(mesh: &Mesh, bvh: &Bvh, plane: &Plane, out: &mut Vec<Contour>) -> Result<()> {
    let mut segs = Vec::new();
    segments(mesh, bvh, plane, &mut segs)?;
    segs.retain(|s| s.from != s.to);
    // By triangle, so that the result does not depend on the BVH.
    segs.sort_unstable();

    // The segment ends (see `end_key`). Sorted by edge key, with `from`s
    // first, the ends at each crossed edge are a run.
    let ends_len = 2 * segs.len();
    let key_of = |e| end_key(&segs, e);
    let mut ends = Vec::new();
    reserve(&mut ends, ends_len)?;
    ends.extend(0..ends_len);
    ends.sort_unstable_by_key(|&e| (key_of(e), e & 1, e));

    // The end each end joins, or why it joins none.
    let mut link = Vec::new();
    reserve(&mut link, ends_len)?;
    link.resize(ends_len, BOUNDARY);
    for run in ends.chunk_by(|&a, &b| key_of(a) == key_of(b)) {
        match *run {
            [_] => {}
            [a, b] => {
                link[a] = b;
                link[b] = a;
            }
            _ => run.iter().for_each(|&e| link[e] = SPLIT),
        }
    }

    let mut used = Vec::new();
    reserve(&mut used, segs.len())?;
    used.resize(segs.len(), false);
    let mut count = 0;
    // Open contours first, from their ends in ascending key order, then the
    // closed loops, each from its smallest key: a loop's segments are all
    // unused when the scan reaches that key, and its `from` end comes first.
    for open in [true, false] {
        for &start in &ends {
            if used[start / 2] || (open && link[start] < SPLIT) {
                continue;
            }
            if count == out.len() {
                push(out, Contour::default())?;
            }
            let c = &mut out[count];
            count += 1;
            c.points.clear();
            c.edges.clear();
            c.triangles.clear();
            walk(&segs, &link, &mut used, start, c)?;
            reserve(&mut c.points, c.edges.len())?;
            c.points
                .extend(c.edges.iter().map(|&e| crossing(mesh, plane, e)));
        }
    }
    out.truncate(count);
    // A triangle is in one contour only, so the order is total.
    out.sort_unstable_by_key(|c| (c.edges[0], c.triangles[0]));
    Ok(())
}

/// Walks the chain entered at end `start`, filling `c`'s edges, triangles
/// and flags and marking its segments used.
fn walk(
    segs: &[Segment],
    link: &[usize],
    used: &mut [bool],
    start: usize,
    c: &mut Contour,
) -> Result<()> {
    let key_of = |e| end_key(segs, e);
    c.oriented = true;
    let mut entry = start;
    loop {
        let s = entry / 2;
        used[s] = true;
        push(&mut c.edges, key_of(entry))?;
        push(&mut c.triangles, segs[s].tri)?;
        // Oriented while every segment is entered by the same kind of end.
        c.oriented &= entry & 1 == start & 1;
        let exit = entry ^ 1;
        match link[exit] {
            next if next == start => {
                c.closed = true;
                c.manifold = true;
                return Ok(());
            }
            next if next >= SPLIT => {
                push(&mut c.edges, key_of(exit))?;
                c.closed = false;
                c.manifold = link[start] != SPLIT && next != SPLIT;
                if c.oriented && start & 1 == 1 {
                    // Walked against the segments: turn it to run along them.
                    c.edges.reverse();
                    c.triangles.reverse();
                }
                return Ok(());
            }
            next => entry = next,
        }
    }
}

/// The edge key at end `e`: end `2i` is the `from` of segment `i`, end
/// `2i + 1` its `to`.
fn end_key(segs: &[Segment], e: usize) -> (VertexId, VertexId) {
    let s = &segs[e / 2];
    if e & 1 == 0 { s.from } else { s.to }
}

/// Reserves room for `additional` more elements, failing with `OutOfMemory`.
fn reserve<T>(v: &mut Vec<T>, additional: usize) -> Result<()> {
    allocation_allowed()?;
    v.try_reserve(additional)
        .map_err(|_| QueryError::OutOfMemory)
}

fn push<T>(v: &mut Vec<T>, value: T) -> Result<()> {
    reserve(v, 1)?;
    v.push(value);
    Ok(())
}

/// [`segments`] by testing every alive triangle, for the tests.
#[cfg(test)]
pub(crate) fn segments_brute(mesh: &Mesh, plane: &Plane, out: &mut Vec<Segment>) {
    out.extend(
        (0..mesh.triangle_slots() as TriangleId)
            .filter(|&id| mesh.is_triangle_alive(id))
            .filter_map(|id| segment(mesh, plane, id)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plane::{cross, dot};
    use crate::test_util::{
        Lcg, box_mesh, build, fin, random_normal, signed_volume6, torus, uv_sphere, wavy,
        with_degenerates,
    };
    use crate::{QueryError, fail_allocations_after};
    use fast_mesh_bvh::Quality;
    use std::collections::HashMap;

    fn sorted(mut s: Vec<Segment>) -> Vec<Segment> {
        s.sort_unstable();
        s
    }

    fn brute(mesh: &Mesh, plane: &Plane) -> Vec<Segment> {
        let mut out = Vec::new();
        segments_brute(mesh, plane, &mut out);
        sorted(out)
    }

    fn fast(mesh: &Mesh, bvh: &Bvh, plane: &Plane) -> Vec<Segment> {
        let mut out = Vec::new();
        segments(mesh, bvh, plane, &mut out).unwrap();
        sorted(out)
    }

    /// 200 random planes through the mesh's bounding box, 50 through a random
    /// vertex and 50 containing a random edge.
    fn planes(mesh: &Mesh, seed: u64) -> Vec<Plane> {
        let mut rng = Lcg(seed);
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in mesh.positions() {
            for k in 0..3 {
                lo[k] = lo[k].min(f64::from(p[k]));
                hi[k] = hi[k].max(f64::from(p[k]));
            }
        }
        let mut out = Vec::new();
        for _ in 0..200 {
            let o = std::array::from_fn(|k| lo[k] + f64::from(rng.next()) * (hi[k] - lo[k]));
            out.push(Plane::new(o, random_normal(&mut rng)).unwrap());
        }
        let tris = mesh.triangles();
        let pick = |rng: &mut Lcg| tris[(rng.next() * tris.len() as f32) as usize % tris.len()];
        for _ in 0..50 {
            let t = pick(&mut rng);
            let o = position(mesh, t[(rng.next() * 3.0) as usize % 3]);
            out.push(Plane::new(o, random_normal(&mut rng)).unwrap());
        }
        for _ in 0..50 {
            let t = pick(&mut rng);
            let (a, b) = (position(mesh, t[0]), position(mesh, t[1]));
            let e = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let n = cross(e, random_normal(&mut rng));
            out.push(Plane::new(a, n).unwrap());
        }
        out
    }

    #[test]
    fn fixtures_are_closed_and_outward() {
        for mesh in [
            uv_sphere(32, 64, 1.0),
            torus(2.0, 0.5, 48, 24),
            box_mesh([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]),
        ] {
            assert!(signed_volume6(&mesh) > 0.0);
            // Every edge is used once in each direction.
            let mut directed = HashMap::new();
            for t in mesh.triangles() {
                for i in 0..3 {
                    *directed.entry((t[i], t[(i + 1) % 3])).or_insert(0) += 1;
                }
            }
            assert!(
                directed
                    .iter()
                    .all(|(&(a, b), &n)| n == 1 && directed.get(&(b, a)) == Some(&1))
            );
        }
        assert_eq!(box_mesh([0.0; 3], [1.0; 3]).triangles().len(), 12);
        assert_eq!(fin().triangles().len(), 6);
        let d = with_degenerates(box_mesh([0.0; 3], [1.0; 3]));
        assert_eq!(d.triangles().len(), 12 + 3 * 3);
        assert_eq!(
            signed_volume6(&d),
            signed_volume6(&box_mesh([0.0; 3], [1.0; 3]))
        );
    }

    #[test]
    fn segments_match_brute_force() {
        let meshes = [wavy(40), uv_sphere(32, 64, 1.0), torus(2.0, 0.5, 48, 24)];
        for (i, mesh) in meshes.iter().enumerate() {
            for quality in [Quality::Sah, Quality::Fast] {
                let bvh = build(mesh, quality);
                let mut nonempty = 0;
                for plane in planes(mesh, 0x5e9 + i as u64) {
                    let expected = brute(mesh, &plane);
                    assert_eq!(fast(mesh, &bvh, &plane), expected, "{plane:?}");
                    nonempty += usize::from(!expected.is_empty());
                }
                assert!(nonempty > 150, "only {nonempty} planes cut mesh {i}");
            }
        }
    }

    #[test]
    fn segment_orientation_rule() {
        // Each segment runs along `n × N` (N the triangle's winding normal),
        // which puts the solid's inside on the left seen from `n`, whichever
        // way the triangle faces.
        let mesh = uv_sphere(32, 64, 1.0);
        let bvh = build(&mesh, Quality::Sah);
        let cases = [
            ([0.0, 0.0, 0.3], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, -0.3], [0.3, -0.2, 0.9]),
            ([0.0, 0.0, -0.3], [-0.3, 0.2, -0.9]),
        ];
        let mut facing_away = 0;
        for (origin, normal) in cases {
            let plane = Plane::new(origin, normal).unwrap();
            let found = fast(&mesh, &bvh, &plane);
            assert!(found.len() > 100);
            for s in &found {
                let from = plane.to_uv(crossing(&mesh, &plane, s.from));
                let to = plane.to_uv(crossing(&mesh, &plane, s.to));
                let [a, b, c] = mesh.triangles()[s.tri as usize].map(|v| position(&mesh, v));
                let tn = cross(
                    std::array::from_fn(|k| b[k] - a[k]),
                    std::array::from_fn(|k| c[k] - a[k]),
                );
                let n = plane.normal();
                facing_away += usize::from(dot(n, tn) < 0.0);
                let w = cross(n, tn);
                let w = [dot(w, plane.u()), dot(w, plane.v())];
                let d = [to[0] - from[0], to[1] - from[1]];
                assert!(d[0] * w[0] + d[1] * w[1] > 0.0, "{plane:?} {s:?}");
            }
        }
        assert!(facing_away > 50, "{facing_away}");
    }

    #[test]
    fn outer_contours_wind_counter_clockwise() {
        // Twice the signed area of the cut, summed over the segments, is
        // positive from either side of the plane for a closed outward mesh.
        let meshes = [
            uv_sphere(32, 64, 1.0),
            box_mesh([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]),
            torus(2.0, 0.5, 48, 24),
        ];
        let normals = [[0.0, 0.0, 1.0], [0.3, -0.2, 0.9], [1.0, 0.0, 0.0]];
        for mesh in &meshes {
            let bvh = build(mesh, Quality::Sah);
            for z in [-0.3, 0.0, 0.2] {
                for n in normals {
                    for n in [n, n.map(|c: f64| -c)] {
                        let plane = Plane::new([0.0, 0.0, z], n).unwrap();
                        let area2: f64 = fast(mesh, &bvh, &plane)
                            .iter()
                            .map(|s| {
                                let a = plane.to_uv(crossing(mesh, &plane, s.from));
                                let b = plane.to_uv(crossing(mesh, &plane, s.to));
                                a[0] * b[1] - a[1] * b[0]
                            })
                            .sum();
                        assert!(area2 > 0.0, "{plane:?}: {area2}");
                    }
                }
            }
        }
    }

    #[test]
    fn segments_chain_head_to_tail_on_closed_meshes() {
        // On a closed, consistently wound mesh every crossed edge key ends one
        // segment and starts another.
        for mesh in [uv_sphere(32, 64, 1.0), torus(2.0, 0.5, 48, 24)] {
            let bvh = build(&mesh, Quality::Sah);
            for plane in planes(&mesh, 0xc4a1) {
                let found = fast(&mesh, &bvh, &plane);
                let mut balance: HashMap<(VertexId, VertexId), i32> = HashMap::new();
                for s in &found {
                    *balance.entry(s.from).or_default() += 1;
                    *balance.entry(s.to).or_default() -= 1;
                }
                assert!(balance.values().all(|&b| b == 0), "{plane:?}");
                assert_eq!(balance.len(), found.len(), "{plane:?}");
            }
        }
    }

    #[test]
    fn crossing_is_symmetric_in_triangle_order() {
        for mesh in [uv_sphere(32, 64, 1.0), torus(2.0, 0.5, 48, 24)] {
            let bvh = build(&mesh, Quality::Sah);
            for plane in planes(&mesh, 0x7b2) {
                // The bits of each edge's crossing, and how many segment ends
                // (one per incident triangle) produced it.
                let mut seen: HashMap<(VertexId, VertexId), ([u64; 3], u32)> = HashMap::new();
                for s in fast(&mesh, &bvh, &plane) {
                    for key in [s.from, s.to] {
                        assert!(key.0 < key.1);
                        let p = crossing(&mesh, &plane, key);
                        let bits = p.map(f64::to_bits);
                        assert_eq!(
                            crossing(&mesh, &plane, (key.1, key.0)).map(f64::to_bits),
                            bits
                        );
                        let entry = seen.entry(key).or_insert((bits, 0));
                        assert_eq!(entry.0, bits, "{key:?}");
                        entry.1 += 1;
                        // The point lies on the edge, between its endpoints
                        // up to rounding.
                        let (a, b) = (position(&mesh, key.0), position(&mesh, key.1));
                        let tol = 1e-12;
                        assert!(
                            (0..3)
                                .all(|k| a[k].min(b[k]) - tol <= p[k]
                                    && p[k] <= a[k].max(b[k]) + tol)
                        );
                        assert!(plane.signed_distance(p).abs() < 1e-12);
                    }
                }
                assert!(seen.values().all(|&(_, n)| n == 2), "{plane:?}");
            }
        }
    }

    #[test]
    fn vertex_on_plane_counts_as_plus() {
        // The plane z = 0 passes through vertices 0, 1, 2 and 3 of the box
        // [0, 1]^3, which are all `+`: no segment.
        let mesh = box_mesh([0.0; 3], [1.0; 3]);
        let bvh = build(&mesh, Quality::Sah);
        let plane = Plane::new([0.0; 3], [0.0, 0.0, 1.0]).unwrap();
        assert!(fast(&mesh, &bvh, &plane).is_empty());
        // Flipped, the bottom corners are still `+` and the top ones `-`.
        let plane = Plane::new([0.0; 3], [0.0, 0.0, -1.0]).unwrap();
        let found = fast(&mesh, &bvh, &plane);
        assert_eq!(found.len(), 8);
        for s in found {
            for key in [s.from, s.to] {
                assert_eq!(crossing(&mesh, &plane, key)[2], 0.0);
            }
        }
    }

    #[test]
    fn degenerate_and_non_manifold_triangles_do_not_panic() {
        let plane = Plane::new([0.1, 0.2, 0.5], [0.1, 0.2, 1.0]).unwrap();
        for mesh in [with_degenerates(uv_sphere(8, 16, 1.0)), fin()] {
            let bvh = build(&mesh, Quality::Sah);
            assert_eq!(fast(&mesh, &bvh, &plane), brute(&mesh, &plane));
        }
        // A triangle [a, a, b] crossed between a and b gives a segment from
        // the edge back to itself; [a, a, a] gives none.
        let p = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mesh = Mesh::from_buffers(&p, &[0, 0, 1, 1, 1, 1]).unwrap();
        let bvh = build(&mesh, Quality::Sah);
        let found = fast(
            &mesh,
            &bvh,
            &Plane::new([0.0; 3], [0.0, 0.0, -1.0]).unwrap(),
        );
        assert_eq!(
            found,
            vec![Segment {
                tri: 0,
                from: (0, 1),
                to: (0, 1)
            }]
        );
        // The fin's shared edge is crossed by all three quads.
        let mesh = fin();
        let bvh = build(&mesh, Quality::Sah);
        let found = fast(
            &mesh,
            &bvh,
            &Plane::new([0.0, 0.0, 0.5], [0.0, 0.0, 1.0]).unwrap(),
        );
        assert_eq!(found.len(), 6);
        let shared = found
            .iter()
            .filter(|s| s.from == (0, 1) || s.to == (0, 1))
            .count();
        assert_eq!(shared, 3);
    }

    #[test]
    fn removed_triangles_are_skipped() {
        let mut mesh = uv_sphere(8, 16, 1.0);
        let mut bvh = build(&mesh, Quality::Sah);
        let plane = Plane::new([0.0; 3], [0.0, 0.0, 1.0]).unwrap();
        let before = fast(&mesh, &bvh, &plane);
        let gone = before[0].tri;
        let mut tx = mesh.begin_edit();
        tx.remove_triangles(&[gone]).unwrap();
        let p = tx.commit().unwrap();
        bvh.update(&mut mesh, p.changes()).unwrap();
        let after = fast(&mesh, &bvh, &plane);
        assert_eq!(after, brute(&mesh, &plane));
        assert_eq!(after.len(), before.len() - 1);
        assert!(after.iter().all(|s| s.tri != gone));
    }

    #[test]
    fn stale_mismatch_and_oom() {
        let mut mesh = uv_sphere(8, 16, 1.0);
        let other = uv_sphere(8, 16, 1.0);
        let bvh = build(&mesh, Quality::Sah);
        let plane = Plane::new([0.0; 3], [0.0, 0.0, 1.0]).unwrap();
        let mut out = vec![Segment {
            tri: 9,
            from: (1, 2),
            to: (3, 4),
        }];
        let keep = out.clone();
        assert_eq!(
            segments(&other, &bvh, &plane, &mut out),
            Err(QueryError::MeshMismatch)
        );
        fail_allocations_after(Some(3));
        let r = segments(&mesh, &bvh, &plane, &mut out);
        fail_allocations_after(None);
        assert_eq!(r, Err(QueryError::OutOfMemory));
        assert_eq!(out, keep);
        // Results are appended after what `out` held.
        segments(&mesh, &bvh, &plane, &mut out).unwrap();
        assert_eq!(out[0], keep[0]);
        assert_eq!(sorted(out[1..].to_vec()), brute(&mesh, &plane));
        let mut tx = mesh.begin_edit();
        tx.set_positions(&[0], &[[0.0, 0.0, 2.0]]).unwrap();
        tx.commit().unwrap();
        assert_eq!(
            segments(&mesh, &bvh, &plane, &mut out),
            Err(QueryError::StaleBvh)
        );
    }

    fn cut(mesh: &Mesh, plane: &Plane) -> Slice {
        let bvh = build(mesh, Quality::Sah);
        let mut out = Slice::default();
        slice(mesh, &bvh, plane, &mut out).unwrap();
        for c in &out.contours {
            check_contour(mesh, plane, c);
        }
        out
    }

    /// The arrays are parallel and every point is its edge's crossing.
    fn check_contour(mesh: &Mesh, plane: &Plane, c: &Contour) {
        assert_eq!(c.points.len(), c.edges.len());
        let segments = c.points.len() - usize::from(!c.closed);
        assert_eq!(c.triangles.len(), segments);
        assert!(!c.triangles.is_empty());
        for (p, &e) in c.points.iter().zip(&c.edges) {
            assert!(e.0 < e.1);
            assert_eq!(
                p.map(f64::to_bits),
                crossing(mesh, plane, e).map(f64::to_bits)
            );
        }
        // Each triangle's segment joins its two points; when oriented, it
        // runs from the first to the second.
        let n = c.edges.len();
        for (i, &t) in c.triangles.iter().enumerate() {
            let s = segment(mesh, plane, t).unwrap();
            let (a, b) = (c.edges[i], c.edges[(i + 1) % n]);
            if c.oriented {
                assert_eq!((s.from, s.to), (a, b), "{c:?}");
            } else {
                assert!((s.from, s.to) == (a, b) || (s.from, s.to) == (b, a));
            }
        }
    }

    /// The signed area of a closed contour in the plane's `(u, v)` frame.
    fn area(plane: &Plane, c: &Contour) -> f64 {
        let q: Vec<[f64; 2]> = c.points.iter().map(|&p| plane.to_uv(p)).collect();
        (0..q.len())
            .map(|i| {
                let (a, b) = (q[i], q[(i + 1) % q.len()]);
                a[0] * b[1] - a[1] * b[0]
            })
            .sum::<f64>()
            / 2.0
    }

    fn close(value: f64, expected: f64, rel: f64) -> bool {
        (value - expected).abs() <= rel * expected.abs()
    }

    #[test]
    fn sphere_cut_is_one_closed_ccw_loop() {
        use std::f64::consts::PI;
        let mesh = uv_sphere(64, 128, 1.0);
        let plane = Plane::new([0.0, 0.0, 0.3], [0.0, 0.0, 1.0]).unwrap();
        let out = cut(&mesh, &plane);
        assert_eq!(out.contours.len(), 1);
        let c = &out.contours[0];
        assert!(c.closed && c.oriented && c.manifold);
        let a = area(&plane, c);
        assert!(a > 0.0 && close(a, PI * (1.0 - 0.09), 0.01), "{a}");
    }

    #[test]
    fn torus_horizontal_cut_is_outer_and_hole() {
        use std::f64::consts::PI;
        let mesh = torus(2.0, 0.5, 96, 48);
        let plane = Plane::new([0.0; 3], [0.0, 0.0, 1.0]).unwrap();
        let out = cut(&mesh, &plane);
        assert_eq!(out.contours.len(), 2);
        let mut areas: Vec<f64> = out
            .contours
            .iter()
            .map(|c| {
                assert!(c.closed && c.oriented && c.manifold);
                area(&plane, c)
            })
            .collect();
        areas.sort_by(f64::total_cmp);
        assert!(close(areas[0], -PI * 1.5 * 1.5, 0.01), "{areas:?}");
        assert!(close(areas[1], PI * 2.5 * 2.5, 0.01), "{areas:?}");
    }

    #[test]
    fn cube_face_flush_plane_is_one_closed_loop() {
        let mesh = box_mesh([-1.0; 3], [1.0; 3]);
        for z in [1.0, -1.0] {
            for n in [1.0, -1.0] {
                let out = cut(&mesh, &Plane::new([0.0, 0.0, z], [0.0, 0.0, n]).unwrap());
                assert!(out.contours.len() <= 1, "z = {z}, n = {n}");
                assert!(out.contours.iter().all(|c| c.closed && c.manifold));
            }
        }
        let out = cut(&mesh, &Plane::new([0.0; 3], [1.0, 0.0, 0.0]).unwrap());
        assert_eq!(out.contours.len(), 1);
        let c = &out.contours[0];
        assert!(c.closed && c.oriented && c.manifold);
        assert_eq!(c.points.len(), 8);
    }

    #[test]
    fn terrain_grid_row_plane_is_open_chains() {
        let mesh = wavy(20);
        let mut incident: HashMap<(VertexId, VertexId), usize> = HashMap::new();
        for t in mesh.triangles() {
            for i in 0..3 {
                *incident.entry(key(t[i], t[(i + 1) % 3])).or_default() += 1;
            }
        }
        // Both normals, so that some chain is walked from its `to` end and
        // turned around.
        for normal in [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]] {
            let plane = Plane::new([0.0, 7.0, 0.0], normal).unwrap();
            let out = cut(&mesh, &plane);
            assert!(!out.contours.is_empty());
            for c in &out.contours {
                assert!(!c.closed && c.oriented && c.manifold);
                assert_eq!(incident[&c.edges[0]], 1);
                assert_eq!(incident[c.edges.last().unwrap()], 1);
            }
            // Every segment is used exactly once.
            let mut used: Vec<TriangleId> = out
                .contours
                .iter()
                .flat_map(|c| c.triangles.iter().copied())
                .collect();
            used.sort_unstable();
            let expected: Vec<TriangleId> = brute(&mesh, &plane).iter().map(|s| s.tri).collect();
            assert_eq!(used, expected);
        }
    }

    #[test]
    fn nonmanifold_fin_splits_chains() {
        let out = cut(
            &fin(),
            &Plane::new([0.0, 0.0, 0.5], [0.0, 0.0, 1.0]).unwrap(),
        );
        assert!(out.contours.len() >= 3);
        for c in &out.contours {
            assert!(!c.closed && !c.manifold);
            // Each piece ends at the shared edge.
            assert!(c.edges[0] == (0, 1) || *c.edges.last().unwrap() == (0, 1));
        }
    }

    #[test]
    fn degenerate_triangles_do_not_break_chains() {
        let mesh = with_degenerates(uv_sphere(16, 32, 1.0));
        let mut rng = Lcg(0xde9e);
        for _ in 0..50 {
            let plane = Plane::new([0.0; 3], random_normal(&mut rng)).unwrap();
            let out = cut(&mesh, &plane);
            assert_eq!(out.contours.len(), 1, "{plane:?}");
            let c = &out.contours[0];
            assert!(c.closed && c.oriented && c.manifold, "{plane:?}");
        }
    }

    #[test]
    fn collinear_sliver_with_distinct_ids_crossing_plane() {
        // A box plus one zero-area triangle: three distinct, collinear
        // vertices straddling the plane. The sliver's two crossings are
        // both where its line meets the plane, so it forms its own open
        // chain of two near-coincident points, leaving the box loop
        // exactly as cut without it.
        let flat = box_mesh([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let n = flat.positions().len() as u32;
        let positions: Vec<f32> = flat
            .positions()
            .iter()
            .flatten()
            .copied()
            .chain([2.0, 0.0, -1.0, 2.0, 0.0, 0.5, 2.0, 0.0, 1.0])
            .collect();
        let mut indices: Vec<u32> = flat.triangles().iter().flatten().copied().collect();
        indices.extend([n, n + 1, n + 2]);
        let mesh = Mesh::from_buffers(&positions, &indices).unwrap();
        let plane = Plane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        let out = cut(&mesh, &plane);
        assert_eq!(out.contours.len(), 2);
        let bare = cut(&flat, &plane);
        assert_eq!(bare.contours.len(), 1);
        let closed = out.contours.iter().find(|c| c.closed).unwrap();
        assert_eq!(closed, &bare.contours[0]);
        let sliver = out.contours.iter().find(|c| !c.closed).unwrap();
        assert_eq!(sliver.triangles, [flat.triangle_slots() as TriangleId]);
        assert_eq!(sliver.points.len(), 2);
        for p in &sliver.points {
            assert!(
                (p[0] - 2.0).abs() < 1e-9 && p[1].abs() < 1e-9 && p[2].abs() < 1e-9,
                "{p:?}"
            );
        }
    }

    #[test]
    fn flipped_patch_reports_unoriented() {
        let sphere = uv_sphere(16, 32, 1.0);
        let plane = Plane::new([0.0, 0.0, 0.3], [0.0, 0.0, 1.0]).unwrap();
        let flipped = brute(&sphere, &plane)[3].tri;
        let positions: Vec<f32> = sphere.positions().iter().flatten().copied().collect();
        let mut tris = sphere.triangles().to_vec();
        tris[flipped as usize].swap(1, 2);
        let indices: Vec<u32> = tris.iter().flatten().copied().collect();
        let mesh = Mesh::from_buffers(&positions, &indices).unwrap();
        let out = cut(&mesh, &plane);
        assert_eq!(out.contours.len(), 1);
        let c = &out.contours[0];
        assert!(c.triangles.contains(&flipped));
        assert!(c.closed && !c.oriented && c.manifold);
        // Unflipped, the same cut is oriented.
        assert!(cut(&sphere, &plane).contours[0].oriented);
    }

    #[test]
    fn deterministic_order() {
        let meshes = [torus(2.0, 0.5, 48, 24), wavy(20), fin()];
        let mut checked = 0;
        for (i, mesh) in meshes.iter().enumerate() {
            let sah = build(mesh, Quality::Sah);
            let quick = build(mesh, Quality::Fast);
            for plane in planes(mesh, 0x0bde + i as u64).into_iter().step_by(5) {
                let mut a = Slice::default();
                let mut b = Slice::default();
                slice(mesh, &sah, &plane, &mut a).unwrap();
                slice(mesh, &sah, &plane, &mut b).unwrap();
                assert_eq!(a, b, "{plane:?}");
                // Another BVH, and reusing a filled output, give the same value.
                slice(mesh, &quick, &plane, &mut b).unwrap();
                assert_eq!(a, b, "{plane:?}");
                for c in &a.contours {
                    check_contour(mesh, &plane, c);
                    if c.closed {
                        assert_eq!(c.edges[0], *c.edges.iter().min().unwrap());
                    }
                }
                assert!(
                    a.contours
                        .windows(2)
                        .all(|w| w[0].edges[0] <= w[1].edges[0])
                );
                checked += usize::from(a.contours.len() > 1);
            }
        }
        assert!(checked > 5, "{checked}");
    }

    #[test]
    fn slice_rejects_stale_bvh() {
        let mut mesh = uv_sphere(8, 16, 1.0);
        let bvh = build(&mesh, Quality::Sah);
        let plane = Plane::new([0.0; 3], [0.0, 0.0, 1.0]).unwrap();
        let mut out = Slice::default();
        slice(&mesh, &bvh, &plane, &mut out).unwrap();
        assert!(!out.contours.is_empty());
        assert_eq!(
            slice(&uv_sphere(8, 16, 1.0), &bvh, &plane, &mut out),
            Err(QueryError::MeshMismatch)
        );
        let mut tx = mesh.begin_edit();
        tx.set_positions(&[0], &[[0.0, 0.0, 2.0]]).unwrap();
        tx.commit().unwrap();
        assert_eq!(
            slice(&mesh, &bvh, &plane, &mut out),
            Err(QueryError::StaleBvh)
        );
    }

    #[test]
    fn slice_out_of_memory_clears_output() {
        let mesh = torus(2.0, 0.5, 24, 12);
        let bvh = build(&mesh, Quality::Sah);
        let plane = Plane::new([0.0; 3], [0.0, 0.0, 1.0]).unwrap();
        let mut expected = Slice::default();
        slice(&mesh, &bvh, &plane, &mut expected).unwrap();
        assert_eq!(expected.contours.len(), 2);
        let mut failures = 0;
        for n in 0..400 {
            let mut out = expected.clone();
            fail_allocations_after(Some(n));
            let r = slice(&mesh, &bvh, &plane, &mut out);
            fail_allocations_after(None);
            match r {
                Ok(()) => assert_eq!(out, expected),
                Err(e) => {
                    assert_eq!(e, QueryError::OutOfMemory);
                    assert!(out.contours.is_empty());
                    failures += 1;
                }
            }
        }
        assert!(failures > 10, "{failures}");
    }
}
