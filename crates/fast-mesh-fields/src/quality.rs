//! Triangle mesh quality metrics and sliver detection.
//!
//! [`triangle_quality`] evaluates each live triangle under one of
//! [`QualityMetric`], summarizes min/max/mean and sliver counts, and records the
//! worst incident triangle quality per vertex slot for visualization.

use fast_mesh_store::Mesh;

use crate::geom::{cross, dot, norm, sub, widen};
use crate::grid::{filled, reserve_exact};
use crate::{FieldsError, Result};

/// Quality metric evaluated per live triangle by [`triangle_quality`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityMetric {
    /// Twice the inradius divided by the circumradius:
    /// `8 * A^2 / (s * a * b * c)` in `[0.0, 1.0]`, where `1.0` is equilateral
    /// and `0.0` is degenerate. A triangle is a sliver when `q < sliver_threshold`.
    RadiusRatio,
    /// Minimum interior angle in degrees in `[0.0, 60.0]`, where `60.0` is
    /// equilateral and `0.0` is degenerate. A triangle is a sliver when
    /// `q < sliver_threshold`.
    MinAngleDeg,
    /// Normalized edge-to-inradius ratio:
    /// `longest * s / (2 * sqrt(3) * A)` in `[1.0, +inf)`, where `1.0` is
    /// equilateral and zero-area degenerate triangles evaluate to `+inf`. A
    /// triangle is a sliver when `q > sliver_threshold`.
    AspectRatio,
}

/// Summary report returned by [`triangle_quality`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityReport {
    /// Minimum quality across live triangles (`0.0` if `live_triangle_count == 0`).
    pub min: f32,
    /// Maximum quality across live triangles (`0.0` if `live_triangle_count == 0`).
    pub max: f32,
    /// Arithmetic mean quality across live triangles (`0.0` if
    /// `live_triangle_count == 0`).
    pub mean: f32,
    /// Number of live triangles classified as slivers relative to `sliver_threshold`.
    pub sliver_triangle_count: usize,
    /// Total number of live triangles evaluated.
    pub live_triangle_count: usize,
}

/// Evaluates `metric` across every live triangle of `mesh` and writes the worst
/// incident triangle quality to each vertex slot in `vertex_quality`.
///
/// For `RadiusRatio` and `MinAngleDeg` (higher is better), each vertex receives
/// the minimum quality among its incident live triangles; for `AspectRatio`
/// (lower is better), each vertex receives the maximum quality among its
/// incident live triangles. Dead or unused vertex slots receive `0.0`.
///
/// # Errors
/// - [`FieldsError::InvalidInput`] if `sliver_threshold` is negative or non-finite,
///   or a live triangle references a non-finite vertex position.
/// - [`FieldsError::OutOfMemory`] if internal or output buffers cannot be reserved.
///
/// On error `vertex_quality` is left unchanged.
pub fn triangle_quality(
    mesh: &Mesh,
    metric: QualityMetric,
    sliver_threshold: f32,
    vertex_quality: &mut Vec<f32>,
) -> Result<QualityReport> {
    if !sliver_threshold.is_finite() || sliver_threshold < 0.0 {
        return Err(FieldsError::InvalidInput(
            "sliver_threshold must be finite and non-negative",
        ));
    }

    let slots = mesh.vertex_slots();
    let mut has_incident = filled(slots, false)?;
    let mut vert_q = filled(slots, 0.0f64)?;
    if vertex_quality.capacity() < slots {
        reserve_exact(vertex_quality, slots - vertex_quality.len())?;
    }

    let pos = mesh.positions();
    let thresh = f64::from(sliver_threshold);
    let sqrt3_2 = 2.0 * 3.0f64.sqrt();

    let mut live_triangle_count = 0usize;
    let mut sliver_triangle_count = 0usize;
    let mut min_q = f64::INFINITY;
    let mut max_q = f64::NEG_INFINITY;
    let mut sum_q = 0.0f64;

    for t in 0..mesh.triangle_slots() as u32 {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        let corners = tri.map(|v| pos[v as usize]);
        if !corners.iter().flatten().all(|c| c.is_finite()) {
            return Err(FieldsError::InvalidInput(
                "mesh vertex position must be finite",
            ));
        }

        let [fa, fb, fc] = corners.map(widen);
        let ab = sub(fb, fa);
        let ac = sub(fc, fa);
        let bc = sub(fc, fb);
        let a = norm(bc);
        let b = norm(ac);
        let c = norm(ab);
        let s = 0.5 * (a + b + c);
        let area2 = norm(cross(ab, ac));
        let area = 0.5 * area2;

        let q = match metric {
            QualityMetric::RadiusRatio => {
                if area <= 0.0 || s <= 0.0 || a <= 0.0 || b <= 0.0 || c <= 0.0 {
                    0.0
                } else {
                    (8.0 * area * area / (s * a * b * c)).clamp(0.0, 1.0)
                }
            }
            QualityMetric::MinAngleDeg => {
                if area <= 0.0 || a <= 0.0 || b <= 0.0 || c <= 0.0 {
                    0.0
                } else {
                    let cos_a = dot(ab, ac) / (c * b);
                    let cos_b = dot([-ab[0], -ab[1], -ab[2]], bc) / (c * a);
                    let cos_c = dot(ac, bc) / (b * a);
                    let max_cos = cos_a.max(cos_b).max(cos_c).clamp(-1.0, 1.0);
                    max_cos.acos().to_degrees().clamp(0.0, 60.0)
                }
            }
            QualityMetric::AspectRatio => {
                let longest = a.max(b).max(c);
                if area <= 0.0 || longest <= 0.0 {
                    f64::INFINITY
                } else {
                    (longest * s / (sqrt3_2 * area)).max(1.0)
                }
            }
        };

        let is_sliver = match metric {
            QualityMetric::RadiusRatio | QualityMetric::MinAngleDeg => q < thresh,
            QualityMetric::AspectRatio => q > thresh,
        };

        live_triangle_count += 1;
        if is_sliver {
            sliver_triangle_count += 1;
        }
        if q < min_q {
            min_q = q;
        }
        if q > max_q {
            max_q = q;
        }
        sum_q += q;

        for v in tri {
            let idx = v as usize;
            if !has_incident[idx] {
                has_incident[idx] = true;
                vert_q[idx] = q;
            } else {
                match metric {
                    QualityMetric::RadiusRatio | QualityMetric::MinAngleDeg => {
                        vert_q[idx] = vert_q[idx].min(q);
                    }
                    QualityMetric::AspectRatio => {
                        vert_q[idx] = vert_q[idx].max(q);
                    }
                }
            }
        }
    }

    vertex_quality.clear();
    vertex_quality.resize(slots, 0.0);
    for (v, q_out) in vertex_quality.iter_mut().enumerate() {
        if mesh.is_vertex_alive(v as u32) && has_incident[v] {
            *q_out = vert_q[v] as f32;
        }
    }

    if live_triangle_count == 0 {
        return Ok(QualityReport {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            sliver_triangle_count: 0,
            live_triangle_count: 0,
        });
    }

    Ok(QualityReport {
        min: min_q as f32,
        max: max_q as f32,
        mean: (sum_q / live_triangle_count as f64) as f32,
        sliver_triangle_count,
        live_triangle_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::fail_allocations_after;
    use crate::test_util::mesh;

    #[test]
    fn equilateral_and_degenerate_metrics() {
        let eq = mesh(
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 3.0f32.sqrt() * 0.5, 0.0],
            ],
            &[[0, 1, 2]],
        );
        let mut vq = Vec::new();
        let rr = triangle_quality(&eq, QualityMetric::RadiusRatio, 0.5, &mut vq).unwrap();
        assert!((rr.min - 1.0).abs() < 1e-6);
        assert_eq!(rr.sliver_triangle_count, 0);
        let ma = triangle_quality(&eq, QualityMetric::MinAngleDeg, 20.0, &mut vq).unwrap();
        assert!((ma.min - 60.0).abs() < 1e-4);
        let ar = triangle_quality(&eq, QualityMetric::AspectRatio, 3.0, &mut vq).unwrap();
        assert!((ar.min - 1.0).abs() < 1e-6);

        let degen = mesh(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            &[[0, 1, 2]],
        );
        let rr_d = triangle_quality(&degen, QualityMetric::RadiusRatio, 0.2, &mut vq).unwrap();
        assert_eq!(rr_d.min, 0.0);
        assert_eq!(rr_d.sliver_triangle_count, 1);
        let ar_d = triangle_quality(&degen, QualityMetric::AspectRatio, 5.0, &mut vq).unwrap();
        assert_eq!(ar_d.max, f32::INFINITY);
        assert_eq!(ar_d.sliver_triangle_count, 1);
    }

    #[test]
    fn oom_preserves_output() {
        let m = mesh(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            &[[0, 1, 2]],
        );
        for n in 0..4 {
            let mut vq = vec![1.5f32];
            fail_allocations_after(Some(n));
            let res = triangle_quality(&m, QualityMetric::RadiusRatio, 0.2, &mut vq);
            fail_allocations_after(None);
            match res {
                Ok(_) => assert_eq!(vq.len(), m.vertex_slots()),
                Err(e) => {
                    assert_eq!(e, FieldsError::OutOfMemory);
                    assert_eq!(vq, vec![1.5f32]);
                }
            }
        }
    }
}
