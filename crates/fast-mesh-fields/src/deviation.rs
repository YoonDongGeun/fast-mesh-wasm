//! Surface deviation measurement from a source mesh to a target mesh.
//!
//! [`mesh_deviation`] transforms each sampled vertex of `source` into `target`'s
//! local coordinate space via a rigid transform, computes unsigned or signed
//! closest-point distance against `target`, writes per-vertex-slot values into
//! `out`, and returns summary statistics over the finite samples.

use fast_mesh_bvh::Bvh;
use fast_mesh_query::{ClosestHit, ClosestOptions, QueryError, Rigid, closest_point_many};
use fast_mesh_store::{Mesh, par};

use crate::distance::{PAR_MIN_POINTS, POINT_CHUNK};
use crate::geom::{dot, triangle_normal};
use crate::grid::{filled, push, reserve_exact};
use crate::{FieldsError, Result, SignData, signed_distances};

/// Options for [`mesh_deviation`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviationOptions {
    /// When `true`, distances are signed using `target_sign` (negative inside,
    /// positive outside). When `false`, distances are unsigned magnitudes.
    pub signed: bool,
    /// Maximum distance to search (inclusive); vertices farther than this get
    /// `f32::INFINITY` and are excluded from [`DeviationStats`].
    pub max_distance: f32,
}

impl Default for DeviationOptions {
    fn default() -> Self {
        Self {
            signed: false,
            max_distance: f32::INFINITY,
        }
    }
}

/// Summary statistics of finite sampled vertex deviations from [`mesh_deviation`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviationStats {
    /// Minimum finite sampled deviation (`0.0` if `sample_count == 0`).
    pub min: f32,
    /// Maximum finite sampled deviation (one-sided Hausdorff deviation for
    /// unsigned mode; `0.0` if `sample_count == 0`).
    pub max: f32,
    /// Arithmetic mean of finite sampled deviations (`0.0` if `sample_count == 0`).
    pub mean: f32,
    /// Root mean square of finite sampled deviations (`0.0` if `sample_count == 0`).
    pub rms: f32,
    /// 95th percentile of sorted finite sampled deviations (`0.0` if
    /// `sample_count == 0`).
    pub p95: f32,
    /// Number of finite sampled vertices included in the statistics.
    pub sample_count: usize,
}

/// Measures per-vertex deviation from `source` to `target`.
///
/// Every live vertex used by at least one live non-degenerate triangle of
/// `source` is transformed by `source_to_target` (`Rigid::apply`, cast to
/// `f32`) into `target`'s local space and queried against `target` / `target_bvh`.
/// Dead vertex slots and vertices unused by live non-degenerate triangles are
/// written as `0.0` in `out` and excluded from [`DeviationStats`]. Sampled
/// vertices farther than `options.max_distance` receive `f32::INFINITY` in `out`
/// and are also excluded from [`DeviationStats`].
///
/// With the `parallel` feature, large vertex batches run in fixed chunks on
/// rayon's pool and remain bit-identical across thread counts.
///
/// # Errors
/// - [`FieldsError::InvalidInput`] if `options.max_distance` is negative or NaN,
///   `options.signed` is `true` without current `target_sign`, or any sampled
///   source vertex (or transformed point) is non-finite.
/// - [`FieldsError::Query`] if `source_to_target` is invalid or `target_bvh` is
///   mismatched or stale.
/// - [`FieldsError::OutOfMemory`] if internal or output buffers cannot be reserved.
///
/// On error `out` is left unchanged.
pub fn mesh_deviation(
    source: &Mesh,
    target: &Mesh,
    target_bvh: &Bvh,
    target_sign: Option<&SignData>,
    source_to_target: Rigid,
    options: &DeviationOptions,
    out: &mut Vec<f32>,
) -> Result<DeviationStats> {
    if options.max_distance.is_nan() || options.max_distance < 0.0 {
        return Err(FieldsError::InvalidInput(
            "max_distance must be zero or greater, and not NaN",
        ));
    }
    source_to_target.validate()?;
    if target_bvh.mesh_uid() != target.uid() {
        return Err(QueryError::MeshMismatch.into());
    }
    if !target_bvh.is_current(target) {
        return Err(QueryError::StaleBvh.into());
    }
    if options.signed {
        let Some(sign) = target_sign else {
            return Err(FieldsError::InvalidInput(
                "signed deviation requires target_sign",
            ));
        };
        if !sign.is_current(target, target_bvh) {
            return Err(FieldsError::InvalidInput(
                "target_sign is not current for target mesh and BVH",
            ));
        }
    }

    let slots = source.vertex_slots();
    let pos = source.positions();
    let mut sampled = filled(slots, false)?;
    for t in 0..source.triangle_slots() as u32 {
        let Some(tri) = source.triangle(t) else {
            continue;
        };
        let corners = tri.map(|v| pos[v as usize]);
        if !corners.iter().flatten().all(|c| c.is_finite()) {
            return Err(FieldsError::InvalidInput(
                "source vertex position must be finite",
            ));
        }
        let n = triangle_normal(corners);
        if dot(n, n) > 0.0 {
            for v in tri {
                if source.is_vertex_alive(v) {
                    sampled[v as usize] = true;
                }
            }
        }
    }

    let sample_total = sampled.iter().filter(|&&b| b).count();
    let mut sample_indices: Vec<usize> = Vec::new();
    let mut sample_points: Vec<[f32; 3]> = Vec::new();
    reserve_exact(&mut sample_indices, sample_total)?;
    reserve_exact(&mut sample_points, sample_total)?;
    for (v, &is_sampled) in sampled.iter().enumerate() {
        if !is_sampled {
            continue;
        }
        let moved = source_to_target.apply(pos[v]).map(|c| c as f32);
        if !moved.iter().all(|c| c.is_finite()) {
            return Err(FieldsError::InvalidInput(
                "transformed source vertex must be finite",
            ));
        }
        push(&mut sample_indices, v)?;
        push(&mut sample_points, moved)?;
    }

    let mut sample_vals = filled(sample_points.len(), 0.0f32)?;
    if !sample_points.is_empty() {
        if options.signed {
            let sign = target_sign.expect("validated above");
            signed_distances(
                target,
                target_bvh,
                sign,
                &sample_points,
                options.max_distance,
                &mut sample_vals,
            )?;
        } else {
            let opts = ClosestOptions {
                max_distance: options.max_distance,
            };
            let mut hits: Vec<Option<ClosestHit>> = filled(sample_points.len(), None)?;
            let sequential =
                sample_points.len() < PAR_MIN_POINTS || !par::enabled() || par::force_sequential();
            if sequential {
                par::set_force_sequential(true);
            }
            let found = closest_point_many(target, target_bvh, &sample_points, &opts, &mut hits);
            if sequential {
                par::set_force_sequential(false);
            }
            found?;

            if sequential {
                for (val, hit) in sample_vals.iter_mut().zip(&hits) {
                    *val = hit.map_or(f32::INFINITY, |h| h.distance);
                }
            } else {
                let oom = || FieldsError::OutOfMemory;
                let parts =
                    par::map_range(sample_points.len(), POINT_CHUNK, PAR_MIN_POINTS, oom, |r| {
                        let mut part = Vec::new();
                        part.try_reserve_exact(r.len()).map_err(|_| oom())?;
                        for i in r {
                            part.push(hits[i].map_or(f32::INFINITY, |h| h.distance));
                        }
                        Ok(part)
                    })?;
                for (chunk, part) in sample_vals.chunks_mut(POINT_CHUNK).zip(parts) {
                    chunk.copy_from_slice(&part);
                }
            }
        }
    }

    let mut finite_vals: Vec<f32> = Vec::new();
    reserve_exact(&mut finite_vals, sample_vals.len())?;
    for &d in &sample_vals {
        if d.is_finite() {
            finite_vals.push(d);
        }
    }

    if out.capacity() < slots {
        reserve_exact(out, slots - out.len())?;
    }

    let stats = compute_stats(&mut finite_vals);

    out.clear();
    out.resize(slots, 0.0);
    for (&v, &val) in sample_indices.iter().zip(&sample_vals) {
        out[v] = val;
    }

    Ok(stats)
}

fn compute_stats(finite_vals: &mut [f32]) -> DeviationStats {
    let n = finite_vals.len();
    if n == 0 {
        return DeviationStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            rms: 0.0,
            p95: 0.0,
            sample_count: 0,
        };
    }

    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for &d in finite_vals.iter() {
        if d < min_v {
            min_v = d;
        }
        if d > max_v {
            max_v = d;
        }
        let df = f64::from(d);
        sum += df;
        sum_sq += df * df;
    }

    finite_vals.sort_unstable_by(f32::total_cmp);
    let p95 = if n == 1 {
        finite_vals[0]
    } else {
        let pos = 0.95 * (n - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let t = pos - lo as f64;
        ((1.0 - t) * f64::from(finite_vals[lo]) + t * f64::from(finite_vals[hi])) as f32
    };

    DeviationStats {
        min: min_v,
        max: max_v,
        mean: (sum / n as f64) as f32,
        rms: (sum_sq / n as f64).sqrt() as f32,
        p95,
        sample_count: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SignMode;
    use crate::grid::fail_allocations_after;
    use crate::test_util::*;

    #[test]
    fn zero_deviation_on_same_mesh() {
        let mut m = icosphere();
        let bvh = build_bvh(&m);
        let sign = SignData::build(&mut m, &bvh, SignMode::Auto).unwrap();
        let mut out = Vec::new();
        let stats = mesh_deviation(
            &m,
            &m,
            &bvh,
            Some(&sign),
            Rigid::IDENTITY,
            &DeviationOptions {
                signed: true,
                max_distance: f32::INFINITY,
            },
            &mut out,
        )
        .unwrap();
        assert_eq!(stats.sample_count, m.vertex_count());
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.rms, 0.0);
        assert_eq!(stats.p95, 0.0);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn oom_preserves_output() {
        let mut m = cube();
        let bvh = build_bvh(&m);
        let sign = SignData::build(&mut m, &bvh, SignMode::Auto).unwrap();
        for n in 0..8 {
            let mut out = vec![4.2f32, 9.1];
            fail_allocations_after(Some(n));
            let res = mesh_deviation(
                &m,
                &m,
                &bvh,
                Some(&sign),
                Rigid::IDENTITY,
                &DeviationOptions {
                    signed: true,
                    max_distance: f32::INFINITY,
                },
                &mut out,
            );
            fail_allocations_after(None);
            match res {
                Ok(_) => assert_eq!(out.len(), m.vertex_slots()),
                Err(e) => {
                    assert_eq!(e, FieldsError::OutOfMemory);
                    assert_eq!(out, vec![4.2f32, 9.1]);
                }
            }
        }
    }
}
