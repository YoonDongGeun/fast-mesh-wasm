//! Box and sphere region queries.

use fast_mesh_bvh::{Aabb, Bvh, NodeKind};
use fast_mesh_store::{Mesh, TriangleId, VertexId};

use crate::ray::points;
use crate::tri::{box_triangle, sphere_triangle};
use crate::{Appender, QueryError, Result, check};

/// A region of space, in mesh-local coordinates. Both shapes are closed:
/// points on the boundary are inside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Region {
    /// An axis-aligned box. An empty box ([`Aabb::EMPTY`]) contains nothing.
    Box(Aabb),
    /// A ball. The center must be finite and the radius finite and not negative.
    Sphere {
        /// The center of the ball.
        center: [f32; 3],
        /// The radius of the ball.
        radius: f32,
    },
}

impl Region {
    fn validate(&self) -> Result<()> {
        if let Region::Sphere { center, radius } = self {
            if !center.iter().all(|c| c.is_finite()) {
                return Err(QueryError::InvalidInput("sphere center must be finite"));
            }
            if !(radius.is_finite() && *radius >= 0.0) {
                return Err(QueryError::InvalidInput(
                    "sphere radius must be finite and not negative",
                ));
            }
        }
        Ok(())
    }

    /// A box that contains the whole region.
    ///
    /// The `f32` containment test `d <= r * r` can accept points a few ulp
    /// outside `center +- radius`, so the sphere's box is padded outward: the
    /// radius grows by a relative `4 * EPSILON` plus a tiny absolute amount,
    /// and each bound then moves one more ulp out. Pruning is then
    /// conservative with respect to the containment tests, whatever the tree
    /// layout.
    fn bounds(&self) -> Aabb {
        match *self {
            Region::Box(b) => b,
            Region::Sphere { center, radius } => {
                let r = radius * (1.0 + 4.0 * f32::EPSILON) + f32::MIN_POSITIVE;
                Aabb::new(
                    center.map(|c| (c - r).next_down()),
                    center.map(|c| (c + r).next_up()),
                )
            }
        }
    }

    fn contains_point(&self, q: [f32; 3]) -> bool {
        match *self {
            Region::Box(b) => (0..3).all(|k| b.min[k] <= q[k] && q[k] <= b.max[k]),
            Region::Sphere { center, radius } => {
                let d: f32 = (0..3)
                    .map(|k| (q[k] - center[k]) * (q[k] - center[k]))
                    .sum();
                d <= radius * radius
            }
        }
    }

    fn meets_triangle(&self, p: [[f32; 3]; 3]) -> bool {
        match *self {
            Region::Box(b) => box_triangle(&b, p),
            Region::Sphere { center, radius } => sphere_triangle(center, radius, p),
        }
    }
}

/// Calls `leaf` for every triangle of every leaf whose bounds meet the region's
/// bounding box, until `leaf` reports an error.
fn candidates(
    bvh: &Bvh,
    region: &Region,
    mut leaf: impl FnMut(TriangleId) -> Result<()>,
) -> Result<()> {
    let bounds = region.bounds();
    let mut failed = None;
    bvh.traverse(|_, node| {
        if failed.is_some() || !node.bounds.overlaps(&bounds) {
            return false;
        }
        if let NodeKind::Leaf { triangles } = node.kind {
            for &id in triangles {
                if let Err(e) = leaf(id) {
                    failed = Some(e);
                    return false;
                }
            }
        }
        true
    });
    failed.map_or(Ok(()), Err)
}

/// Appends to `out` every alive triangle that touches `region` (shares a point
/// with it), and returns how many were appended. The order is unspecified.
/// Elements already in `out` are kept.
///
/// The test is exact for the triangle's surface, not its bounding box. Sphere
/// and box overlap are both closed, so a triangle that only touches the region
/// counts.
///
/// # Errors
/// [`QueryError::InvalidInput`] for a sphere with a non-finite center or a
/// radius that is negative or not finite; then [`QueryError::MeshMismatch`] or
/// [`QueryError::StaleBvh`] when `bvh` does not match the current `mesh`; and
/// [`QueryError::OutOfMemory`] when `out` cannot grow. On any error `out` keeps
/// its original length.
pub fn triangles_in(
    mesh: &Mesh,
    bvh: &Bvh,
    region: &Region,
    out: &mut Vec<TriangleId>,
) -> Result<usize> {
    region.validate()?;
    check(mesh, bvh)?;
    let mut ap = Appender::new(out);
    candidates(bvh, region, |id| {
        if region.meets_triangle(points(mesh, id)) {
            ap.push(id)?;
        }
        Ok(())
    })?;
    Ok(ap.count())
}

/// Appends to `out`, in ascending order and without duplicates, every vertex
/// inside `region` that belongs to an alive triangle, and returns how many were
/// appended. Elements already in `out` are kept and stay in front; only the
/// appended part is sorted.
///
/// A vertex that no triangle uses is not found: the search runs over triangles.
///
/// Before deduplication, `out` can temporarily hold each vertex once per
/// incident candidate triangle, about 6 times the result for a typical mesh,
/// and `out` keeps that capacity afterwards.
///
/// # Errors
/// As for [`triangles_in`].
pub fn vertices_in(
    mesh: &Mesh,
    bvh: &Bvh,
    region: &Region,
    out: &mut Vec<VertexId>,
) -> Result<usize> {
    region.validate()?;
    check(mesh, bvh)?;
    let mut ap = Appender::new(out);
    let positions = mesh.positions();
    candidates(bvh, region, |id| {
        for v in mesh.triangles()[id as usize] {
            if region.contains_point(positions[v as usize]) {
                ap.push(v)?;
            }
        }
        Ok(())
    })?;
    let found = ap.appended();
    found.sort_unstable();
    // Compact the sorted run in place, keeping the first of each value.
    let mut kept = 0;
    for i in 0..found.len() {
        if kept == 0 || found[i] != found[kept - 1] {
            found[kept] = found[i];
            kept += 1;
        }
    }
    ap.truncate_appended(kept);
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray::points;
    use crate::test_util::{Lcg, build, wavy};
    use crate::tri::{box_triangle, sphere_triangle};
    use crate::{QueryError, fail_allocations_after};
    use fast_mesh_bvh::{Aabb, Bvh, Quality};
    use fast_mesh_store::{Mesh, TriangleId, VertexId};

    fn contains(region: &Region, q: [f32; 3]) -> bool {
        match region {
            Region::Box(b) => (0..3).all(|k| b.min[k] <= q[k] && q[k] <= b.max[k]),
            Region::Sphere { center, radius } => {
                let d: f32 = (0..3)
                    .map(|k| (q[k] - center[k]) * (q[k] - center[k]))
                    .sum();
                d <= radius * radius
            }
        }
    }

    fn brute_triangles(mesh: &Mesh, region: &Region) -> Vec<TriangleId> {
        (0..mesh.triangle_slots() as TriangleId)
            .filter(|&id| mesh.is_triangle_alive(id))
            .filter(|&id| match region {
                Region::Box(b) => box_triangle(b, points(mesh, id)),
                Region::Sphere { center, radius } => {
                    sphere_triangle(*center, *radius, points(mesh, id))
                }
            })
            .collect()
    }

    fn brute_vertices(mesh: &Mesh, region: &Region) -> Vec<VertexId> {
        let mut v: Vec<VertexId> = (0..mesh.triangle_slots() as TriangleId)
            .filter(|&id| mesh.is_triangle_alive(id))
            .flat_map(|id| mesh.triangles()[id as usize])
            .filter(|&v| mesh.is_vertex_alive(v) && contains(region, mesh.positions()[v as usize]))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// 200 boxes and 200 spheres over and around a `wavy(16)` grid.
    fn regions() -> Vec<Region> {
        let mut rng = Lcg(0x4e61);
        let mut out = Vec::new();
        for _ in 0..200 {
            let c = [
                rng.next() * 18.0 - 1.0,
                rng.next() * 18.0 - 1.0,
                rng.next() * 4.0 - 1.0,
            ];
            let h = [
                0.1 + rng.next() * 3.0,
                0.1 + rng.next() * 3.0,
                0.1 + rng.next() * 2.0,
            ];
            out.push(Region::Box(Aabb::new(
                std::array::from_fn(|k| c[k] - h[k]),
                std::array::from_fn(|k| c[k] + h[k]),
            )));
            out.push(Region::Sphere {
                center: c,
                radius: 0.1 + rng.next() * 3.0,
            });
        }
        out
    }

    fn tri_mesh() -> (Mesh, Bvh) {
        let p = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mesh = Mesh::from_buffers(&p, &[0, 1, 2]).unwrap();
        let bvh = build(&mesh, Quality::Sah);
        (mesh, bvh)
    }

    #[test]
    fn triangles_match_brute_force() {
        let mesh = wavy(16);
        for quality in [Quality::Sah, Quality::Fast] {
            let bvh = build(&mesh, quality);
            let mut nonempty = 0;
            for region in regions() {
                let mut out = Vec::new();
                let n = triangles_in(&mesh, &bvh, &region, &mut out).unwrap();
                assert_eq!(n, out.len());
                out.sort_unstable();
                assert_eq!(out, brute_triangles(&mesh, &region), "{region:?}");
                nonempty += usize::from(n > 0);
            }
            assert!(nonempty > 100, "only {nonempty} regions hit");
        }
    }

    #[test]
    fn bounding_box_overlap_alone_is_not_enough() {
        let (mesh, bvh) = tri_mesh();
        // Both regions overlap the triangle's bounding box but miss the
        // triangle, which lies below the line x + y = 1.
        let regions = [
            Region::Box(Aabb::new([0.6, 0.6, -1.0], [1.0, 1.0, 1.0])),
            Region::Sphere {
                center: [0.9, 0.9, 0.0],
                radius: 0.5,
            },
        ];
        for region in regions {
            let mut out = Vec::new();
            assert_eq!(triangles_in(&mesh, &bvh, &region, &mut out), Ok(0));
            assert!(out.is_empty());
        }
    }

    #[test]
    fn vertices_match_brute_force() {
        let mesh = wavy(16);
        for quality in [Quality::Sah, Quality::Fast] {
            let bvh = build(&mesh, quality);
            let mut nonempty = 0;
            for region in regions() {
                let mut out = Vec::new();
                let n = vertices_in(&mesh, &bvh, &region, &mut out).unwrap();
                assert_eq!(n, out.len());
                assert!(out.windows(2).all(|w| w[0] < w[1]), "{region:?}");
                assert_eq!(out, brute_vertices(&mesh, &region), "{region:?}");
                nonempty += usize::from(n > 0);
            }
            assert!(nonempty > 100, "only {nonempty} regions hit");
        }
    }

    #[test]
    fn sphere_pruning_box_covers_f32_containment_at_ulp_scale() {
        // Rounding in `q - center` and in the squares lets the containment test
        // accept a point a few ulp beyond `center + radius`. Search seeded
        // values for such a point, a few ulp outward along +x.
        let mut rng = Lcg(0x51de);
        let mut found = None;
        'search: for _ in 0..20_000 {
            let center = [rng.next() * 10.0 - 5.0, 0.0, 0.0];
            let radius = 0.1 + rng.next() * 3.0;
            let region = Region::Sphere { center, radius };
            let plain_hi = center[0] + radius;
            let mut x = plain_hi;
            for _ in 0..4 {
                x = x.next_up();
                if contains(&region, [x, 0.0, 0.0]) {
                    found = Some((region, x));
                    break 'search;
                }
            }
        }
        let (region, x) = found.expect("no point just outside the unpadded box is contained");
        let Region::Sphere { center, radius } = region else {
            unreachable!()
        };
        assert!(x > center[0] + radius);
        assert!(contains(&region, [x, 0.0, 0.0]));
        assert!(region.bounds().max[0] >= x);

        // A triangle whose bounds start at that vertex, so only the padded box
        // reaches it.
        let p = [x, 0.0, 0.0, x + 1.0, 5.0, 0.0, x + 1.0, 0.0, 5.0];
        let mesh = Mesh::from_buffers(&p, &[0, 1, 2]).unwrap();
        for quality in [Quality::Sah, Quality::Fast] {
            let bvh = build(&mesh, quality);
            let mut out = Vec::new();
            assert_eq!(vertices_in(&mesh, &bvh, &region, &mut out), Ok(1));
            assert_eq!(out, vec![0]);
        }
    }

    #[test]
    fn vertices_in_sorts_only_the_appended_range() {
        let mesh = wavy(4);
        let bvh = build(&mesh, Quality::Sah);
        let region = Region::Box(Aabb::new([-1.0; 3], [5.0; 3]));
        let mut out = vec![999];
        let n = vertices_in(&mesh, &bvh, &region, &mut out).unwrap();
        assert_eq!(n, 25);
        assert_eq!(out[0], 999);
        assert_eq!(out[1..], (0..25).collect::<Vec<_>>()[..]);
        // The first element is larger than the rest, so a sort of all of
        // `out` would have moved it.
    }

    #[test]
    fn isolated_vertex_is_not_found() {
        let (mut mesh, mut bvh) = tri_mesh();
        let mut tx = mesh.begin_edit();
        let ids = tx.add_vertices(&[[0.5, 0.25, 0.0]]).unwrap();
        let p = tx.commit().unwrap();
        bvh.update(&mut mesh, p.changes()).unwrap();
        let region = Region::Box(Aabb::new([0.4, 0.2, -1.0], [0.6, 0.3, 1.0]));
        let mut out = Vec::new();
        assert_eq!(vertices_in(&mesh, &bvh, &region, &mut out), Ok(0));
        assert!(!out.contains(&ids[0]));
        // Its triangle neighbors are found, though.
        let region = Region::Box(Aabb::new([-1.0; 3], [2.0; 3]));
        assert_eq!(vertices_in(&mesh, &bvh, &region, &mut out), Ok(3));
        assert_eq!(out, vec![0, 1, 2]);
    }

    #[test]
    fn empty_box_returns_nothing() {
        let mesh = wavy(4);
        let bvh = build(&mesh, Quality::Sah);
        let region = Region::Box(Aabb::EMPTY);
        let mut out = Vec::new();
        assert_eq!(triangles_in(&mesh, &bvh, &region, &mut out), Ok(0));
        let mut verts = Vec::new();
        assert_eq!(vertices_in(&mesh, &bvh, &region, &mut verts), Ok(0));
        assert!(out.is_empty() && verts.is_empty());
    }

    #[test]
    fn empty_mesh_returns_nothing() {
        let check_empty = |mesh: &Mesh, bvh: &Bvh| {
            for region in [
                Region::Box(Aabb::new([-9.0; 3], [9.0; 3])),
                Region::Sphere {
                    center: [0.0; 3],
                    radius: 9.0,
                },
            ] {
                let mut t = Vec::new();
                assert_eq!(triangles_in(mesh, bvh, &region, &mut t), Ok(0));
                let mut v = Vec::new();
                assert_eq!(vertices_in(mesh, bvh, &region, &mut v), Ok(0));
                assert!(t.is_empty() && v.is_empty());
            }
        };
        let mesh = Mesh::new();
        let bvh = build(&mesh, Quality::Sah);
        check_empty(&mesh, &bvh);

        let mut mesh = wavy(4);
        let mut bvh = build(&mesh, Quality::Sah);
        let all: Vec<TriangleId> = (0..mesh.triangle_slots() as TriangleId).collect();
        let mut tx = mesh.begin_edit();
        tx.remove_triangles(&all).unwrap();
        let p = tx.commit().unwrap();
        bvh.update(&mut mesh, p.changes()).unwrap();
        check_empty(&mesh, &bvh);
    }

    #[test]
    fn invalid_input() {
        let (mesh, bvh) = tri_mesh();
        let bad = [
            Region::Sphere {
                center: [0.0; 3],
                radius: -1.0,
            },
            Region::Sphere {
                center: [0.0; 3],
                radius: f32::NAN,
            },
            Region::Sphere {
                center: [f32::NAN, 0.0, 0.0],
                radius: 1.0,
            },
        ];
        for region in bad {
            let mut t = vec![5];
            assert!(matches!(
                triangles_in(&mesh, &bvh, &region, &mut t),
                Err(QueryError::InvalidInput(_))
            ));
            let mut v = vec![5];
            assert!(matches!(
                vertices_in(&mesh, &bvh, &region, &mut v),
                Err(QueryError::InvalidInput(_))
            ));
            assert_eq!((t, v), (vec![5], vec![5]));
        }
    }

    #[test]
    fn stale_and_mismatch_are_rejected() {
        let (mut mesh, bvh) = tri_mesh();
        let (other, _) = tri_mesh();
        let region = Region::Box(Aabb::new([-1.0; 3], [2.0; 3]));
        let mut out = Vec::new();
        assert_eq!(
            triangles_in(&other, &bvh, &region, &mut out),
            Err(QueryError::MeshMismatch)
        );
        let mut tx = mesh.begin_edit();
        tx.set_positions(&[0], &[[0.0, 0.0, 1.0]]).unwrap();
        tx.commit().unwrap();
        assert_eq!(
            triangles_in(&mesh, &bvh, &region, &mut out),
            Err(QueryError::StaleBvh)
        );
        let mut verts = Vec::new();
        assert_eq!(
            vertices_in(&mesh, &bvh, &region, &mut verts),
            Err(QueryError::StaleBvh)
        );
    }

    #[test]
    fn oom_restores_out() {
        let (mesh, bvh) = tri_mesh();
        let region = Region::Box(Aabb::new([-1.0; 3], [2.0; 3]));
        let mut t = vec![7];
        fail_allocations_after(Some(0));
        let r = triangles_in(&mesh, &bvh, &region, &mut t);
        fail_allocations_after(None);
        assert_eq!(r, Err(QueryError::OutOfMemory));
        assert_eq!(t, vec![7]);

        let mut v = vec![7];
        fail_allocations_after(Some(0));
        let r = vertices_in(&mesh, &bvh, &region, &mut v);
        fail_allocations_after(None);
        assert_eq!(r, Err(QueryError::OutOfMemory));
        assert_eq!(v, vec![7]);
    }
}
