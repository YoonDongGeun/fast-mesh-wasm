//! Cutting a mesh with a plane.
//!
//! A vertex is on the plane's `+` side when its signed distance is zero or
//! more, so no vertex lies on the plane and each triangle with vertices on
//! both sides gives exactly one segment, between crossings strictly inside two
//! of its edges. Crossings are named by their edge key `(min id, max id)`.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "the segments are chained by `slice`, added next")
)]

use fast_mesh_bvh::{Aabb, Bvh, NodeKind};
use fast_mesh_store::{Mesh, TriangleId, VertexId};

use crate::plane::{Plane, dot};
use crate::{Appender, Result, check};

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
/// endpoints in id order so that it is the same whichever triangle asks. The
/// endpoints should be on opposite sides.
pub(crate) fn crossing(mesh: &Mesh, plane: &Plane, key: (VertexId, VertexId)) -> [f64; 3] {
    let (a, b) = (
        position(mesh, key.0.min(key.1)),
        position(mesh, key.0.max(key.1)),
    );
    let (sa, sb) = (plane.signed_distance(a), plane.signed_distance(b));
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
        Lcg, box_mesh, build, fin, signed_volume6, torus, uv_sphere, wavy, with_degenerates,
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

    fn position(mesh: &Mesh, v: VertexId) -> [f64; 3] {
        mesh.positions()[v as usize].map(f64::from)
    }

    fn random_normal(rng: &mut Lcg) -> [f64; 3] {
        loop {
            let n: [f64; 3] = std::array::from_fn(|_| f64::from(rng.next()) * 2.0 - 1.0);
            if dot(n, n) > 0.01 {
                return n;
            }
        }
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
        let mesh = uv_sphere(32, 64, 1.0);
        let bvh = build(&mesh, Quality::Sah);
        let plane = Plane::new([0.0, 0.0, 0.3], [0.0, 0.0, 1.0]).unwrap();
        let found = fast(&mesh, &bvh, &plane);
        assert_eq!(found.len(), 2 * 64);
        let mut checked = 0;
        for s in &found {
            let from = plane.to_uv(crossing(&mesh, &plane, s.from));
            let to = plane.to_uv(crossing(&mesh, &plane, s.to));
            let t = mesh.triangles()[s.tri as usize].map(|v| position(&mesh, v));
            let centroid: [f64; 3] = std::array::from_fn(|k| (t[0][k] + t[1][k] + t[2][k]) / 3.0);
            let side = plane.signed_distance(centroid);
            if side.abs() <= 1e-9 {
                continue;
            }
            let c = plane.to_uv(centroid);
            let d = [to[0] - from[0], to[1] - from[1]];
            let w = [c[0] - from[0], c[1] - from[1]];
            let turn = d[0] * w[1] - d[1] * w[0];
            assert!(turn != 0.0 && (turn > 0.0) == (side > 0.0), "{s:?}");
            checked += 1;
        }
        assert!(checked > 100);
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
}
