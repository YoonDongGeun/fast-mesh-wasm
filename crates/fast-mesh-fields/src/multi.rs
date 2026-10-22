//! Signed distances from points to the nearest of several meshes.

use fast_mesh_bvh::Bvh;
use fast_mesh_query::{QueryError, Rigid};
use fast_mesh_store::{Mesh, par};

use crate::distance::{PAR_MIN_POINTS, POINT_CHUNK, point_value};
use crate::{FieldsError, Result, SignData};

/// One mesh of [`signed_distances_multi`], with what measuring to it reads.
#[derive(Debug, Clone, Copy)]
pub struct Target<'a> {
    /// The mesh.
    pub mesh: &'a Mesh,
    /// Its BVH, current for `mesh`.
    pub bvh: &'a Bvh,
    /// Its sign data, current for `mesh` and `bvh`.
    pub sign: &'a SignData,
    /// From the points' space into the mesh's local space.
    pub to_local: Rigid,
}

/// The smallest `f32` not below `x` (`x` not NaN).
fn round_up(x: f64) -> f32 {
    let r = x as f32;
    if f64::from(r) < x { r.next_up() } else { r }
}

/// Signed distances from `points` to the nearest of several `targets`, with
/// the target each value came from: `out[i]` and `nearest[i]` get the value
/// and the target index for `points[i]`.
///
/// A target's value at a point is what [`signed_distances`](crate::signed_distances)
/// gives for that point moved into the target's local space by
/// [`Target::to_local`] (`Rigid::apply` in `f64`, then rounded to `f32`),
/// with `max_distance` rounded up to `f32`, and made `+∞` when its magnitude
/// is above `max_distance` (compared in `f64`). The point's value is the
/// minimum over the targets, so the nearest surface outside every target, or
/// the deepest penetration: the same bits as taking the minimum of separate
/// single-target calls, with ties going to the lowest target index.
/// `nearest[i]` is that index, or `u32::MAX` when the value is `+∞` (no
/// target within `max_distance`, or no targets).
///
/// Every target is searched with the full `max_distance`. The best value so
/// far cannot bound the next target's search: a target the point is inside
/// gives a negative value however far its surface is, and only the closest
/// point decides that side.
///
/// With the `parallel` feature, large batches run in chunks on rayon's
/// pool; the values and indices are the same, bit for bit.
///
/// # Errors
/// Everything is checked before anything is written, so on error `out` and
/// `nearest` are unchanged:
/// - [`FieldsError::InvalidInput`] when the slices differ in length, a point
///   is not finite (in the points' space or moved into a target's),
///   `max_distance` is negative or NaN, or a target's sign data is not
///   current, or there are `u32::MAX` targets or more;
/// - [`FieldsError::Query`] with `MeshMismatch` or `StaleBvh` when a target's
///   BVH does not match its current mesh.
pub fn signed_distances_multi(
    targets: &[Target<'_>],
    points: &[[f32; 3]],
    max_distance: f64,
    out: &mut [f32],
    nearest: &mut [u32],
) -> Result<()> {
    if points.len() != out.len() || points.len() != nearest.len() {
        return Err(FieldsError::InvalidInput(
            "points, out and nearest must have the same length",
        ));
    }
    if max_distance.is_nan() || max_distance < 0.0 {
        return Err(FieldsError::InvalidInput(
            "max_distance must be zero or greater, and not NaN",
        ));
    }
    if targets.len() >= u32::MAX as usize {
        return Err(FieldsError::InvalidInput("too many targets"));
    }
    if !points.iter().flatten().all(|c| c.is_finite()) {
        return Err(FieldsError::InvalidInput("points must be finite"));
    }
    for t in targets {
        if t.bvh.mesh_uid() != t.mesh.uid() {
            return Err(QueryError::MeshMismatch.into());
        }
        if !t.bvh.is_current(t.mesh) {
            return Err(QueryError::StaleBvh.into());
        }
        if !t.sign.is_current(t.mesh, t.bvh) {
            return Err(FieldsError::InvalidInput(
                "sign data is not current for this mesh and BVH",
            ));
        }
        let moved = |&p: &[f32; 3]| t.to_local.apply(p).map(|c| c as f32);
        if !points.iter().flat_map(moved).all(f32::is_finite) {
            return Err(FieldsError::InvalidInput("points must be finite"));
        }
    }
    let bound = round_up(max_distance);
    let value = |p: [f32; 3]| -> Result<(f32, u32)> {
        let (mut best, mut at) = (f32::INFINITY, u32::MAX);
        for (k, t) in targets.iter().enumerate() {
            let q = t.to_local.apply(p).map(|c| c as f32);
            let mut v = point_value(t.mesh, t.bvh, t.sign, q, bound)?;
            if f64::from(v.abs()) > max_distance {
                v = f32::INFINITY;
            }
            // Strictly smaller: a tie stays with the lower index.
            if v < best {
                (best, at) = (v, k as u32);
            }
        }
        Ok((best, at))
    };
    let work = points.len().saturating_mul(targets.len());
    if work < PAR_MIN_POINTS || !par::enabled() || par::force_sequential() {
        for ((&p, v), n) in points.iter().zip(out.iter_mut()).zip(nearest.iter_mut()) {
            (*v, *n) = value(p)?;
        }
        return Ok(());
    }
    // Every point is computed alone, so chunking changes nothing.
    let oom = || FieldsError::OutOfMemory;
    let parts = par::map_range(points.len(), POINT_CHUNK, 0, oom, |r| {
        let mut part = Vec::new();
        part.try_reserve_exact(r.len()).map_err(|_| oom())?;
        for &p in &points[r] {
            part.push(value(p)?);
        }
        Ok(part)
    })?;
    let chunks = out
        .chunks_mut(POINT_CHUNK)
        .zip(nearest.chunks_mut(POINT_CHUNK));
    for ((vs, ns), part) in chunks.zip(parts) {
        for ((v, n), (x, k)) in vs.iter_mut().zip(ns.iter_mut()).zip(part) {
            (*v, *n) = (x, k);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use fast_mesh_query::{QueryError, Rigid};
    use fast_mesh_store::par;

    use super::*;
    use crate::test_util::*;
    use crate::{SignMode, signed_distances};

    /// A turn of `deg` degrees about the axis `axis` (0, 1 or 2), then a step
    /// by `t`.
    fn turned(axis: usize, deg: f64, t: [f64; 3]) -> Rigid {
        let (s, c) = deg.to_radians().sin_cos();
        let (i, j) = ((axis + 1) % 3, (axis + 2) % 3);
        let mut rotation = [[0.0; 3]; 3];
        rotation[axis][axis] = 1.0;
        rotation[i][i] = c;
        rotation[i][j] = -s;
        rotation[j][i] = s;
        rotation[j][j] = c;
        Rigid {
            rotation,
            translation: t,
        }
    }

    /// The reference: separate single-target calls, each made `+∞` past
    /// `max_distance`, then the minimum with ties to the lowest index.
    fn reference(
        targets: &[Target<'_>],
        points: &[[f32; 3]],
        max_distance: f64,
    ) -> (Vec<f32>, Vec<u32>) {
        let mut best = vec![f32::INFINITY; points.len()];
        let mut at = vec![u32::MAX; points.len()];
        for (k, t) in targets.iter().enumerate() {
            let moved: Vec<[f32; 3]> = points
                .iter()
                .map(|&p| t.to_local.apply(p).map(|c| c as f32))
                .collect();
            let mut out = vec![f32::NAN; points.len()];
            let bound = round_up(max_distance);
            signed_distances(t.mesh, t.bvh, t.sign, &moved, bound, &mut out).unwrap();
            for i in 0..points.len() {
                let v = if f64::from(out[i].abs()) > max_distance {
                    f32::INFINITY
                } else {
                    out[i]
                };
                if v < best[i] {
                    best[i] = v;
                    at[i] = k as u32;
                }
            }
        }
        (best, at)
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|x| x.to_bits()).collect()
    }

    /// Three overlapping closed shapes and an open sheet, and points around
    /// them, in a shared space.
    struct Setup {
        meshes: Vec<Mesh>,
        bvhs: Vec<Bvh>,
        places: Vec<Rigid>,
        points: Vec<[f32; 3]>,
    }

    fn setup() -> Setup {
        let mut meshes = vec![cube(), icosphere(), torus(), terrain(12)];
        let places = vec![
            turned(2, 20.0, [0.3, 0.0, 0.0]),
            turned(0, 35.0, [-0.6, 0.4, 0.2]),
            turned(1, -50.0, [0.0, -0.5, 0.4]),
            // The sheet's corner far off, its middle through the shapes.
            turned(2, 0.0, [-6.0, -6.0, -2.0]),
        ];
        let bvhs = meshes.iter_mut().map(|m| build_bvh(m)).collect();
        let mut rng = Lcg(91);
        let points = (0..3000)
            .map(|_| rng.point([-2.5, -2.5, -2.5], [2.5, 2.5, 2.5]))
            .collect();
        Setup {
            meshes,
            bvhs,
            places,
            points,
        }
    }

    fn signs(s: &mut Setup, mode: SignMode) -> Vec<SignData> {
        s.meshes
            .iter_mut()
            .zip(&s.bvhs)
            .map(|(m, b)| SignData::build(m, b, mode).unwrap())
            .collect()
    }

    fn targets<'a>(s: &'a Setup, signs: &'a [SignData], order: &[usize]) -> Vec<Target<'a>> {
        order
            .iter()
            .map(|&k| Target {
                mesh: &s.meshes[k],
                bvh: &s.bvhs[k],
                sign: &signs[k],
                // The places map each mesh into the shared space.
                to_local: s.places[k].inverse(),
            })
            .collect()
    }

    fn run(targets: &[Target<'_>], points: &[[f32; 3]], md: f64) -> (Vec<f32>, Vec<u32>) {
        let mut out = vec![f32::NAN; points.len()];
        let mut nearest = vec![7; points.len()];
        signed_distances_multi(targets, points, md, &mut out, &mut nearest).unwrap();
        (out, nearest)
    }

    #[test]
    fn matches_min_over_single_targets() {
        let mut s = setup();
        for mode in [SignMode::Pseudonormal, SignMode::Winding] {
            let signs = signs(&mut s, mode);
            for order in [&[0, 1, 2, 3][..], &[3, 1, 0], &[2], &[]] {
                let targets = targets(&s, &signs, order);
                for md in [f64::INFINITY, 0.3, 0.1 + 1e-9, 0.0] {
                    let (want, want_at) = reference(&targets, &s.points, md);
                    let (got, at) = run(&targets, &s.points, md);
                    assert_eq!(bits(&got), bits(&want), "{mode:?} {order:?} {md}");
                    assert_eq!(at, want_at, "{mode:?} {order:?} {md}");
                }
            }
            // Penetration, outside values and far points all came up.
            let (got, at) = run(&targets(&s, &signs, &[0, 1, 2, 3]), &s.points, 0.3);
            assert!(got.iter().any(|&v| v < 0.0));
            assert!(got.iter().any(|&v| v > 0.0 && v.is_finite()));
            assert!(got.contains(&f32::INFINITY));
            for k in 0..4 {
                assert!(at.contains(&k), "{mode:?}: target {k} never nearest");
            }
            assert!(
                got.iter()
                    .zip(&at)
                    .all(|(v, &k)| v.is_finite() == (k != u32::MAX))
            );
        }
    }

    #[test]
    fn ties_go_to_the_lowest_index() {
        let mut s = setup();
        let signs = signs(&mut s, SignMode::Auto);
        // The same mesh twice in the same place: every value ties.
        let targets = targets(&s, &signs, &[1, 1, 0]);
        let (got, at) = run(&targets, &s.points, f64::INFINITY);
        let (want, want_at) = reference(&targets, &s.points, f64::INFINITY);
        assert_eq!(bits(&got), bits(&want));
        assert_eq!(at, want_at);
        assert!(!at.contains(&1));
        assert!(at.contains(&0) && at.contains(&2));
    }

    /// Thread counts the parallel path is compared under.
    const POOLS: [usize; 3] = [1, 2, 8];

    #[test]
    fn bit_identical_in_parallel() {
        let mut s = setup();
        let mut rng = Lcg(5);
        s.points = (0..20_000)
            .map(|_| rng.point([-2.5, -2.5, -2.5], [2.5, 2.5, 2.5]))
            .collect();
        for mode in [SignMode::Pseudonormal, SignMode::Winding] {
            let signs = signs(&mut s, mode);
            let targets = targets(&s, &signs, &[0, 1, 2, 3]);
            for md in [f64::INFINITY, 0.3] {
                let go = || {
                    let (v, n) = run(&targets, &s.points, md);
                    (bits(&v), n)
                };
                par::set_force_sequential(true);
                let reference = go();
                par::set_force_sequential(false);
                if !par::enabled() {
                    continue;
                }
                for k in POOLS {
                    assert!(par::with_threads(k, go) == reference, "{mode:?} {md}: {k}");
                }
                let before = par::parallel_entries();
                assert!(go() == reference, "{mode:?} {md}: global pool");
                assert!(par::parallel_entries() > before, "ran sequentially");
            }
        }
    }

    #[test]
    fn invalid_input_writes_nothing() {
        let mut s = setup();
        let signs = signs(&mut s, SignMode::Auto);
        let targets = targets(&s, &signs, &[0, 1]);
        let points = &s.points[..4];
        let mut out = [7.0f32; 4];
        let mut nearest = [7u32; 4];
        let invalid = |r: Result<()>| matches!(r, Err(FieldsError::InvalidInput(_)));
        let mut call = |t: &[Target<'_>], p: &[[f32; 3]], md: f64, n: usize| {
            signed_distances_multi(t, p, md, &mut out[..n], &mut nearest[..n])
        };
        assert!(invalid(call(&targets, points, f64::NAN, 4)));
        assert!(invalid(call(&targets, points, -1.0, 4)));
        assert!(invalid(call(&targets, &points[..3], 1.0, 4)));
        let mut bad = points.to_vec();
        bad[2][0] = f32::NAN;
        assert!(invalid(call(&targets, &bad, 1.0, 4)));
        // Finite here, but not once moved into the second target's space.
        let mut far = targets.clone();
        far[1].to_local.translation = [f64::from(f32::MAX) * 2.0, 0.0, 0.0];
        assert!(invalid(call(&far, points, 1.0, 4)));
        let mut foreign = targets.clone();
        foreign[1].bvh = &s.bvhs[0];
        assert_eq!(
            call(&foreign, points, 1.0, 4),
            Err(FieldsError::Query(QueryError::MeshMismatch))
        );
        let mut wrong_sign = targets.clone();
        wrong_sign[0].sign = &signs[1];
        assert!(invalid(call(&wrong_sign, points, 1.0, 4)));
        assert_eq!(out, [7.0; 4]);
        assert_eq!(nearest, [7; 4]);

        // No targets: everything is far.
        signed_distances_multi(&[], points, 1.0, &mut out, &mut nearest).unwrap();
        assert_eq!(out, [f32::INFINITY; 4]);
        assert_eq!(nearest, [u32::MAX; 4]);
    }
}
