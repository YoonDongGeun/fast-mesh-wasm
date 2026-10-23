//! Signed distances from points to a mesh.

use fast_mesh_bvh::Bvh;
use fast_mesh_query::{ClosestHit, ClosestOptions, QueryError, closest_point};
use fast_mesh_store::{Mesh, par};

use crate::{FieldsError, Result, SignData};

/// Points from which [`signed_distances`] runs in parallel chunks with the
/// `parallel` feature. A point costs about a microsecond (more in winding
/// mode). Measured on the i5-13600K against a 1M-triangle mesh: 256 points
/// break even (0.18 ms either way), 512 run 2.3 times faster in parallel.
pub(crate) const PAR_MIN_POINTS: usize = 512;

/// Points per parallel chunk of [`signed_distances`].
pub(crate) const POINT_CHUNK: usize = 128;

/// Signed distances from `points_in_b` (in mesh B's local coordinates) to
/// mesh B: `out[i]` gets the value for `points_in_b[i]`.
///
/// - The magnitude is the distance to the closest point of B, from
///   [`fast_mesh_query::closest_point`].
/// - The sign comes from `sign` (see [`SignData`]): negative behind the
///   surface (inside), positive in front of it (outside). A point exactly on
///   the surface gets `+0.0`; a point whose side is undecided (exactly in the
///   tangent plane of its closest feature) counts as outside.
/// - A point farther than `max_distance` from the surface (inclusive bound)
///   gets `f32::INFINITY`, meaning far, with no sign. That includes points
///   deeper inside than `max_distance`. An infinite `max_distance` computes
///   every point.
///
/// Nothing is allocated per point. With the `parallel` feature, large
/// batches run in chunks on rayon's pool, each chunk into its own buffer;
/// the values are the same, bit for bit.
///
/// # Errors
/// Everything is checked before anything is written, so on error `out` is
/// unchanged:
/// - [`FieldsError::InvalidInput`] when the slices differ in length, a point
///   is not finite (NaN or infinite), or `max_distance` is negative or NaN;
/// - [`FieldsError::Query`] with `MeshMismatch` or `StaleBvh` when `bvh_b`
///   does not match the current `mesh_b`;
/// - [`FieldsError::InvalidInput`] when `sign` is not current for `mesh_b`
///   and `bvh_b` (see [`SignData::is_current`]).
pub fn signed_distances(
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    sign: &SignData,
    points_in_b: &[[f32; 3]],
    max_distance: f32,
    out: &mut [f32],
) -> Result<()> {
    if points_in_b.len() != out.len() {
        return Err(FieldsError::InvalidInput(
            "points and out must have the same length",
        ));
    }
    if max_distance.is_nan() || max_distance < 0.0 {
        return Err(FieldsError::InvalidInput(
            "max_distance must be zero or greater, and not NaN",
        ));
    }
    if !points_in_b.iter().flatten().all(|c| c.is_finite()) {
        return Err(FieldsError::InvalidInput("points must be finite"));
    }
    if bvh_b.mesh_uid() != mesh_b.uid() {
        return Err(QueryError::MeshMismatch.into());
    }
    if !bvh_b.is_current(mesh_b) {
        return Err(QueryError::StaleBvh.into());
    }
    if !sign.is_current(mesh_b, bvh_b) {
        return Err(FieldsError::InvalidInput(
            "sign data is not current for this mesh and BVH",
        ));
    }
    let value = |p: [f32; 3]| point_value(mesh_b, bvh_b, sign, p, max_distance);
    if points_in_b.len() < PAR_MIN_POINTS || !par::enabled() || par::force_sequential() {
        for (&p, v) in points_in_b.iter().zip(out.iter_mut()) {
            *v = value(p)?;
        }
        return Ok(());
    }
    // Every point is computed alone, so chunking changes no value.
    let oom = || FieldsError::OutOfMemory;
    let parts = par::map_range(points_in_b.len(), POINT_CHUNK, PAR_MIN_POINTS, oom, |r| {
        let mut part = Vec::new();
        part.try_reserve_exact(r.len()).map_err(|_| oom())?;
        for &p in &points_in_b[r] {
            part.push(value(p)?);
        }
        Ok(part)
    })?;
    for (chunk, part) in out.chunks_mut(POINT_CHUNK).zip(parts) {
        chunk.copy_from_slice(&part);
    }
    Ok(())
}

/// The signed distance of one point, as [`signed_distances`] computes it
/// (the inputs already checked).
pub(crate) fn point_value(
    mesh: &Mesh,
    bvh: &Bvh,
    sign: &SignData,
    p: [f32; 3],
    max_distance: f32,
) -> Result<f32> {
    let opts = ClosestOptions { max_distance };
    let hit = closest_point(mesh, bvh, p, &opts)?;
    Ok(hit_value(mesh, bvh, sign, p, hit.as_ref()))
}

/// The signed distance of `p` given its closest point `hit` (`None`: far).
pub(crate) fn hit_value(
    mesh: &Mesh,
    bvh: &Bvh,
    sign: &SignData,
    p: [f32; 3],
    hit: Option<&ClosestHit>,
) -> f32 {
    match hit {
        None => f32::INFINITY,
        Some(hit) if hit.distance == 0.0 => 0.0,
        Some(hit) if sign.is_inside(mesh, bvh, p, hit) => -hit.distance,
        Some(hit) => hit.distance,
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_query::{ClosestOptions, Feature, QueryError, closest_point};
    use fast_mesh_store::{Mesh, par};

    use super::*;
    use crate::test_util::*;
    use crate::{FieldsError, SignMode};

    fn distances(mesh: &mut Mesh, points: &[[f32; 3]], max_distance: f32) -> (SignData, Vec<f32>) {
        let bvh = build_bvh(mesh);
        let sign = SignData::build(mesh, &bvh, SignMode::Auto).unwrap();
        let mut out = vec![f32::NAN; points.len()];
        signed_distances(mesh, &bvh, &sign, points, max_distance, &mut out).unwrap();
        (sign, out)
    }

    #[test]
    fn closed_meshes_match_exact_sign() {
        let mut rng = Lcg(7);
        let meshes = [
            ("cube", cube()),
            ("icosphere", icosphere()),
            ("torus", torus()),
            ("tetrahedron", tetrahedron()),
            ("star prism", star_prism()),
        ];
        for (name, mut mesh) in meshes {
            let near = diagonal(&mesh) as f32 / 20.0;
            let points = sample_points(&mesh, &mut rng, 2000, near);
            let (sign, out) = distances(&mut mesh, &points, f32::INFINITY);
            assert_eq!(sign.status().mode, SignMode::Pseudonormal, "{name}");
            assert!(sign.status().closed, "{name}");
            let checked = check_against_exact(&mesh, &points, &out);
            assert!(checked > 1900, "{name}: only {checked} signs checked");
            assert!(out.iter().any(|&v| v < 0.0), "{name}: no point inside");
            assert!(out.iter().any(|&v| v > 0.0), "{name}: no point outside");
        }
    }

    #[test]
    fn closed_meshes_match_exact_sign_at_vertices_and_edges() {
        // Points pushed off vertices and edge midpoints, where the closest
        // feature is a vertex or an edge rather than a face.
        let mut rng = Lcg(11);
        let meshes = [
            ("cube", cube()),
            ("icosphere", icosphere()),
            ("torus", torus()),
            ("tetrahedron", tetrahedron()),
            ("star prism", star_prism()),
        ];
        for (name, mut mesh) in meshes {
            let d = diagonal(&mesh) as f32;
            let mut points = Vec::new();
            for t in 0..mesh.triangle_slots() as u32 {
                let tri = mesh
                    .triangle(t)
                    .unwrap()
                    .map(|v| mesh.positions()[v as usize]);
                for k in 0..3 {
                    let (a, b) = (tri[k], tri[(k + 1) % 3]);
                    let mid: [f32; 3] = std::array::from_fn(|i| (a[i] + b[i]) / 2.0);
                    for base in [a, mid] {
                        let off = rng.point([-0.05 * d; 3], [0.05 * d; 3]);
                        points.push(std::array::from_fn(|i| base[i] + off[i]));
                    }
                }
            }
            let (_, out) = distances(&mut mesh, &points, f32::INFINITY);
            let checked = check_against_exact(&mesh, &points, &out);
            assert!(
                checked * 10 > points.len() * 9,
                "{name}: only {checked} of {} signs checked",
                points.len()
            );
        }
    }

    #[test]
    fn open_sheet_sign_past_the_rim_uses_boundary_edges() {
        // Points just past the rim y = 0 of a terrain sheet, above and below
        // it: the closest feature is a boundary edge, whose normal is that of
        // its single face.
        let n = 16;
        let mut mesh = terrain(n);
        let mut rng = Lcg(13);
        let mut points = Vec::new();
        let mut below = Vec::new();
        for i in 0..1000 {
            let x = 1.0 + (n - 2) as f32 * rng.next();
            let y = -(0.02 + 0.3 * rng.next());
            let h = 0.02 + 0.3 * rng.next();
            let z = terrain_surface(&mesh, n, f64::from(x), 0.0) as f32;
            let under = i % 2 == 0;
            points.push([x, y, if under { z - h } else { z + h }]);
            below.push(under);
        }
        let (_, out) = distances(&mut mesh, &points, f32::INFINITY);
        let bvh = build_bvh(&mesh);
        let mut on_edge = 0;
        for ((&p, &value), &under) in points.iter().zip(&out).zip(&below) {
            let hit = closest_point(&mesh, &bvh, p, &ClosestOptions::default());
            if matches!(hit.unwrap().unwrap().feature, Feature::Edge(_)) {
                on_edge += 1;
            }
            assert_eq!(value < 0.0, under, "sign at {p:?}: value {value}");
        }
        assert!(on_edge > 900, "only {on_edge} closest points on edges");
    }

    #[test]
    fn open_sheet_penetration_is_negative() {
        let n = 16;
        let mut mesh = terrain(n);
        let mut rng = Lcg(3);
        let mut points = Vec::new();
        let mut below = Vec::new();
        for i in 0..2000 {
            let (x, y) = (4.0 + 8.0 * rng.next(), 4.0 + 8.0 * rng.next());
            let h = 0.01 + 1.5 * rng.next();
            let z = terrain_surface(&mesh, n, f64::from(x), f64::from(y)) as f32;
            let under = i % 2 == 0;
            points.push([x, y, if under { z - h } else { z + h }]);
            below.push(under);
        }
        let (sign, out) = distances(&mut mesh, &points, f32::INFINITY);
        let status = sign.status();
        assert_eq!(status.mode, SignMode::Pseudonormal);
        assert!(!status.closed);
        assert!(status.consistent_orientation);
        let tol = 1e-5 * diagonal(&mesh);
        for ((&p, &value), &under) in points.iter().zip(&out).zip(&below) {
            let brute = brute_distance(&mesh, p);
            assert!(
                (f64::from(value.abs()) - brute).abs() <= tol,
                "magnitude at {p:?}"
            );
            if brute > tol {
                assert_eq!(value < 0.0, under, "sign at {p:?}: value {value}");
            }
        }
    }

    /// Splits the edge of triangle `t` that starts at corner `k` at its
    /// midpoint `m`: `(a, b, c)` becomes `(a, m, c)` and `(m, b, c)`, and the
    /// zero-area triangle `(a, b, m)` fills the gap against the neighbor
    /// across `a`–`b`, so the mesh stays closed and consistently oriented.
    /// Returns the zero-area triangle and the midpoint.
    fn split_with_sliver(
        positions: &mut Vec<[f32; 3]>,
        tris: &mut Vec<[u32; 3]>,
        t: usize,
        k: usize,
    ) -> ([u32; 3], [f32; 3]) {
        let tri = tris[t];
        let (a, b, c) = (tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]);
        let (pa, pb) = (positions[a as usize], positions[b as usize]);
        let pm = std::array::from_fn(|i| (pa[i] + pb[i]) / 2.0);
        positions.push(pm);
        let m = (positions.len() - 1) as u32;
        tris[t] = [a, m, c];
        tris.push([m, b, c]);
        ([a, b, m], pm)
    }

    #[test]
    fn degenerate_faces_do_not_break_signs() {
        let cases = [
            // A face diagonal, then two cube edges (convex, 90 degrees).
            (cube_buffers(), vec![(0, 2), (2, 0), (6, 1)]),
            // Two opposite edges of a tetrahedron, sharp enough that one
            // face normal alone gives wrong signs next to them.
            (tetrahedron_buffers(), vec![(0, 0), (3, 1)]),
        ];
        for ((mut positions, mut tris), splits) in cases {
            let mut slivers = Vec::new();
            let mut mids = Vec::new();
            for (t, k) in splits {
                let (s, m) = split_with_sliver(&mut positions, &mut tris, t, k);
                slivers.push(s);
                mids.push(m);
            }
            let count = slivers.len();
            // The slivers get the smallest ids, so they win distance ties.
            slivers.extend(tris);
            let mut mesh = mesh(&positions, &slivers);
            let bvh = build_bvh(&mesh);
            let sign = SignData::build(&mut mesh, &bvh, SignMode::Auto).unwrap();
            let status = sign.status();
            assert_eq!(status.degenerate_triangles, count);
            assert!(status.closed);
            assert!(status.consistent_orientation);
            assert_eq!(status.mode, SignMode::Pseudonormal);

            let mut rng = Lcg(5);
            let mut points = sample_points(&mesh, &mut rng, 1000, 0.2);
            for m in &mids {
                for _ in 0..300 {
                    let off = rng.point([-0.4; 3], [0.4; 3]);
                    points.push(std::array::from_fn(|i| m[i] + off[i]));
                }
                // Straight out from the midpoint (the shapes are centred on
                // the origin): the closest point is the midpoint itself,
                // shared by a sliver.
                points.push(m.map(|c| c * 1.5));
                points.push(m.map(|c| c * 0.5));
            }
            let mut out = vec![f32::NAN; points.len()];
            signed_distances(&mesh, &bvh, &sign, &points, f32::INFINITY, &mut out).unwrap();
            let checked = check_against_exact(&mesh, &points, &out);
            assert!(
                checked * 10 > points.len() * 9,
                "only {checked} of {} signs checked",
                points.len()
            );

            // The fallback for a reported zero-area triangle really ran.
            let on_sliver = points
                .iter()
                .filter(|&&p| {
                    let hit = closest_point(&mesh, &bvh, p, &ClosestOptions::default());
                    (hit.unwrap().unwrap().triangle as usize) < count
                })
                .count();
            assert!(on_sliver > 0);
        }
    }

    #[test]
    fn far_points_are_infinite() {
        let mut mesh = cube();
        let bvh = build_bvh(&mesh);
        let sign = SignData::build(&mut mesh, &bvh, SignMode::Pseudonormal).unwrap();
        let points = [
            [3.0, 0.0, 0.0],   // 2 outside: far
            [1.25, 0.0, 0.0],  // 0.25 outside
            [0.0, 0.0, 0.0],   // 1 inside: deeper than max_distance, far
            [0.0, 0.0, -0.75], // 0.25 inside
            [1.0, 0.5, 0.25],  // on the surface
            [1.5, 0.0, 0.0],   // exactly at max_distance: kept
        ];
        let mut out = [f32::NAN; 6];
        signed_distances(&mesh, &bvh, &sign, &points, 0.5, &mut out).unwrap();
        assert_eq!(out[0], f32::INFINITY);
        assert_eq!(out[1], 0.25);
        assert_eq!(out[2], f32::INFINITY);
        assert_eq!(out[3], -0.25);
        assert_eq!(out[4].to_bits(), 0.0f32.to_bits());
        assert_eq!(out[5], 0.5);

        // An infinite max_distance computes every point.
        signed_distances(&mesh, &bvh, &sign, &points, f32::INFINITY, &mut out).unwrap();
        assert_eq!(out[0], 2.0);
        assert_eq!(out[2], -1.0);

        // Invalid input is rejected before anything is written.
        let mut out = [7.0f32; 6];
        let mut bad = points;
        bad[3][1] = f32::NAN;
        let invalid = |r: Result<()>| matches!(r, Err(FieldsError::InvalidInput(_)));
        assert!(invalid(signed_distances(
            &mesh, &bvh, &sign, &bad, 0.5, &mut out
        )));
        bad[3][1] = f32::INFINITY;
        assert!(invalid(signed_distances(
            &mesh, &bvh, &sign, &bad, 0.5, &mut out
        )));
        assert!(invalid(signed_distances(
            &mesh,
            &bvh,
            &sign,
            &points,
            f32::NAN,
            &mut out
        )));
        assert!(invalid(signed_distances(
            &mesh, &bvh, &sign, &points, -1.0, &mut out
        )));
        assert!(invalid(signed_distances(
            &mesh,
            &bvh,
            &sign,
            &points[..5],
            0.5,
            &mut out
        )));
        assert_eq!(out, [7.0; 6]);

        // A mesh with no triangles is far from everything.
        let mut empty = Mesh::new();
        let ebvh = build_bvh(&empty);
        let esign = SignData::build(&mut empty, &ebvh, SignMode::Auto).unwrap();
        let mut out = [0.0f32; 2];
        signed_distances(&empty, &ebvh, &esign, &points[..2], f32::INFINITY, &mut out).unwrap();
        assert_eq!(out, [f32::INFINITY; 2]);
    }

    /// Thread counts the parallel paths are compared under.
    const POOLS: [usize; 3] = [1, 2, 8];

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn signed_distances_bit_identical() {
        // The forced-sequential run is the code a feature-off build runs.
        let mut mesh = terrain(317);
        assert!(mesh.triangle_count() >= 200_000);
        let bvh = build_bvh(&mesh);
        // Still dozens of chunks each (`POINT_CHUNK`), at a fraction of the
        // debug-build time of larger batches.
        let points = sample_points(&mesh, &mut Lcg(31), 30_000, 2.0);
        for (mode, count) in [(SignMode::Pseudonormal, 30_000), (SignMode::Winding, 8_000)] {
            let sign = SignData::build(&mut mesh, &bvh, mode).unwrap();
            let points = &points[..count];
            for max_distance in [f32::INFINITY, 0.5] {
                let run = || {
                    let mut out = vec![f32::NAN; points.len()];
                    signed_distances(&mesh, &bvh, &sign, points, max_distance, &mut out).unwrap();
                    bits(&out)
                };
                par::set_force_sequential(true);
                let reference = run();
                par::set_force_sequential(false);
                if !par::enabled() {
                    // Every run below would be this one again.
                    continue;
                }
                for k in POOLS {
                    let got = par::with_threads(k, run);
                    assert!(got == reference, "{mode:?}, {max_distance}: {k} threads");
                }
                let before = par::parallel_entries();
                assert!(run() == reference, "{mode:?}, {max_distance}: global pool");
                assert!(par::parallel_entries() > before, "ran sequentially");
            }
        }
    }

    #[test]
    fn small_inputs_stay_sequential() {
        let mut mesh = icosphere();
        let before = par::parallel_entries();
        let points = sample_points(&mesh, &mut Lcg(2), 400, 0.2);
        for mode in [SignMode::Pseudonormal, SignMode::Winding] {
            let bvh = build_bvh(&mesh);
            let sign = SignData::build(&mut mesh, &bvh, mode).unwrap();
            let mut out = vec![f32::NAN; points.len()];
            signed_distances(&mesh, &bvh, &sign, &points, f32::INFINITY, &mut out).unwrap();
        }
        assert_eq!(par::parallel_entries(), before);
    }

    #[test]
    fn stale_or_foreign_inputs_are_rejected() {
        let mut mesh = cube();
        let mut bvh = build_bvh(&mesh);
        let sign = SignData::build(&mut mesh, &bvh, SignMode::Auto).unwrap();
        let points = [[2.0, 0.0, 0.0]];
        let mut out = [7.0f32];

        let other = cube();
        let other_bvh = build_bvh(&other);
        assert_eq!(
            signed_distances(&mesh, &other_bvh, &sign, &points, 1.0, &mut out),
            Err(FieldsError::Query(QueryError::MeshMismatch))
        );
        assert_eq!(
            signed_distances(&other, &other_bvh, &sign, &points, 1.0, &mut out),
            Err(FieldsError::InvalidInput(
                "sign data is not current for this mesh and BVH"
            ))
        );

        let mut tx = mesh.begin_edit();
        tx.set_positions(&[0], &[[-1.0, -1.0, -1.5]]).unwrap();
        let patch = tx.commit().unwrap();
        assert_eq!(
            signed_distances(&mesh, &bvh, &sign, &points, 1.0, &mut out),
            Err(FieldsError::Query(QueryError::StaleBvh))
        );
        assert_eq!(
            SignData::build(&mut mesh, &bvh, SignMode::Auto).err(),
            Some(FieldsError::Query(QueryError::StaleBvh))
        );
        bvh.update(&mut mesh, patch.changes()).unwrap();
        assert_eq!(
            signed_distances(&mesh, &bvh, &sign, &points, 1.0, &mut out),
            Err(FieldsError::InvalidInput(
                "sign data is not current for this mesh and BVH"
            ))
        );
        assert_eq!(out, [7.0]);
    }
}
