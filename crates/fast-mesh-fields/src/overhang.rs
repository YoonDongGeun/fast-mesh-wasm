//! Printability overhang analysis for triangle meshes.
//!
//! [`overhangs`] measures how far each triangle's outward normal tilts past
//! horizontal relative to a given build direction, reports faces exceeding
//! `critical_angle_deg`, and computes an area-weighted per-vertex overhang angle.

use fast_mesh_store::Mesh;

use crate::geom::{cross, dot, norm, scale, sub, widen};
use crate::grid::{filled, reserve_exact};
use crate::{FieldsError, Result};

/// Configuration for [`overhangs`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverhangOptions {
    /// Upward build direction in mesh-local coordinates (need not be unit length).
    pub build_direction: [f32; 3],
    /// Overhang threshold angle in degrees (`0.0..=180.0`), measured downward past
    /// horizontal: faces strictly above this angle require support.
    pub critical_angle_deg: f32,
}

impl Default for OverhangOptions {
    fn default() -> Self {
        Self {
            build_direction: [0.0, 0.0, 1.0],
            critical_angle_deg: 45.0,
        }
    }
}

/// Summary statistics produced by [`overhangs`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverhangReport {
    /// Total surface area of live triangles whose overhang angle exceeds
    /// `critical_angle_deg`.
    pub total_support_area: f64,
    /// Number of live triangles whose overhang angle exceeds `critical_angle_deg`.
    pub overhang_face_count: usize,
    /// Maximum overhang angle in degrees (`0.0..=90.0`) across all live
    /// non-degenerate triangles.
    pub max_overhang_angle_deg: f32,
}

/// Computes per-triangle overhang angles and area-weighted per-vertex overhang
/// angles (in degrees, `[0.0, 90.0]`) for `mesh`.
///
/// For each live non-degenerate triangle with unit outward normal `n_hat` and
/// unit build vector `b_hat`, let `angle_from_up_deg = acos(clamp(dot(n_hat, b_hat), -1, 1))`.
/// The overhang angle is `overhang_deg = max(angle_from_up_deg - 90.0, 0.0)`:
/// - upward faces (`n_hat == +b_hat`) read `0.0°`,
/// - vertical walls (`n_hat ⟂ b_hat`) read `0.0°`,
/// - `45°` downward slopes read `45.0°`,
/// - flat downward ceilings (`n_hat == -b_hat`) read `90.0°`.
///
/// Triangles with `overhang_deg > options.critical_angle_deg` contribute to
/// [`OverhangReport::total_support_area`] and [`OverhangReport::overhang_face_count`].
/// `vertex_angles_deg` is sized to `mesh.vertex_slots()`, with `0.0` for dead,
/// unused, or degenerate-only vertex slots.
///
/// # Errors
/// - [`FieldsError::InvalidInput`] if `build_direction` is non-finite or zero,
///   or `critical_angle_deg` is non-finite or outside `0.0..=180.0`.
/// - [`FieldsError::OutOfMemory`] if scratch or output buffers cannot be reserved.
///
/// On error `vertex_angles_deg` is left unchanged.
pub fn overhangs(
    mesh: &Mesh,
    options: &OverhangOptions,
    vertex_angles_deg: &mut Vec<f32>,
) -> Result<OverhangReport> {
    if !options.build_direction.iter().all(|c| c.is_finite()) {
        return Err(FieldsError::InvalidInput("build_direction must be finite"));
    }
    let build_f64 = widen(options.build_direction);
    let build_len = norm(build_f64);
    if !build_len.is_finite() || build_len <= 0.0 {
        return Err(FieldsError::InvalidInput(
            "build_direction must be non-zero",
        ));
    }
    let unit_build = scale(build_f64, 1.0 / build_len);

    if !options.critical_angle_deg.is_finite()
        || !(0.0..=180.0).contains(&options.critical_angle_deg)
    {
        return Err(FieldsError::InvalidInput(
            "critical_angle_deg must be finite and in 0.0..=180.0",
        ));
    }
    let critical_f64 = f64::from(options.critical_angle_deg);

    let slots = mesh.vertex_slots();
    let mut weighted_sum = filled(slots, 0.0f64)?;
    let mut area_sum = filled(slots, 0.0f64)?;
    if vertex_angles_deg.capacity() < slots {
        reserve_exact(vertex_angles_deg, slots - vertex_angles_deg.len())?;
    }

    let pos = mesh.positions();
    let mut total_support_area = 0.0f64;
    let mut overhang_face_count = 0usize;
    let mut max_overhang_angle_deg = 0.0f64;

    for t in 0..mesh.triangle_slots() as u32 {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        let corners = tri.map(|v| pos[v as usize]);
        if !corners.iter().flatten().all(|c| c.is_finite()) {
            continue;
        }
        let [fa, fb, fc] = corners.map(widen);
        let n = cross(sub(fb, fa), sub(fc, fa));
        let area2 = norm(n);
        if !area2.is_finite() || area2 <= 0.0 {
            continue;
        }
        let area = 0.5 * area2;
        let n_hat = scale(n, 1.0 / area2);
        let cos_up = dot(n_hat, unit_build).clamp(-1.0, 1.0);
        let angle_from_up_deg = cos_up.acos().to_degrees();
        let overhang_deg = (angle_from_up_deg - 90.0).max(0.0);

        if overhang_deg > critical_f64 {
            total_support_area += area;
            overhang_face_count += 1;
        }
        if overhang_deg > max_overhang_angle_deg {
            max_overhang_angle_deg = overhang_deg;
        }

        let weighted = area * overhang_deg;
        for v in tri {
            let idx = v as usize;
            weighted_sum[idx] += weighted;
            area_sum[idx] += area;
        }
    }

    vertex_angles_deg.clear();
    vertex_angles_deg.resize(slots, 0.0);
    for (v, angle_out) in vertex_angles_deg.iter_mut().enumerate() {
        if mesh.is_vertex_alive(v as u32) && area_sum[v] > 0.0 {
            *angle_out = (weighted_sum[v] / area_sum[v]) as f32;
        }
    }

    Ok(OverhangReport {
        total_support_area,
        overhang_face_count,
        max_overhang_angle_deg: max_overhang_angle_deg as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::fail_allocations_after;
    use crate::test_util::*;

    #[test]
    fn cube_overhang_report_and_angles() {
        let m = cube();
        let mut angles = Vec::new();
        let report = overhangs(&m, &OverhangOptions::default(), &mut angles).unwrap();
        assert_eq!(angles.len(), m.vertex_slots());
        // Bottom face [-1, 1]^2 at z = -1 has 2 triangles of total area 4.0 and 90° overhang.
        assert_eq!(report.overhang_face_count, 2);
        assert!((report.total_support_area - 4.0).abs() < 1e-12);
        assert!((report.max_overhang_angle_deg - 90.0).abs() < 1e-5);
        // Bottom corners have 1 bottom face (area 4 split over 4 corners, plus 2 vertical faces):
        // top corners have 0.0°.
        for (v, &angle) in angles.iter().enumerate().take(8) {
            let z = m.positions()[v][2];
            if z > 0.0 {
                assert_eq!(angle, 0.0);
            } else {
                assert!(angle > 0.0 && angle < 90.0);
            }
        }
    }

    #[test]
    fn invalid_inputs_leave_output_unchanged() {
        let m = cube();
        let mut out = vec![12.5f32, 33.0];
        let orig = out.clone();
        for bad in [
            OverhangOptions {
                build_direction: [0.0, 0.0, 0.0],
                critical_angle_deg: 45.0,
            },
            OverhangOptions {
                build_direction: [f32::NAN, 0.0, 1.0],
                critical_angle_deg: 45.0,
            },
            OverhangOptions {
                build_direction: [0.0, 0.0, 1.0],
                critical_angle_deg: -1.0,
            },
            OverhangOptions {
                build_direction: [0.0, 0.0, 1.0],
                critical_angle_deg: 181.0,
            },
            OverhangOptions {
                build_direction: [0.0, 0.0, 1.0],
                critical_angle_deg: f32::NAN,
            },
        ] {
            assert!(matches!(
                overhangs(&m, &bad, &mut out),
                Err(FieldsError::InvalidInput(_))
            ));
            assert_eq!(out, orig);
        }
    }

    #[test]
    fn oom_preserves_output() {
        let m = cube();
        for n in 0..4 {
            let mut out = vec![7.0f32, 8.0];
            fail_allocations_after(Some(n));
            let res = overhangs(&m, &OverhangOptions::default(), &mut out);
            fail_allocations_after(None);
            match res {
                Ok(_) => assert_eq!(out.len(), m.vertex_slots()),
                Err(e) => {
                    assert_eq!(e, FieldsError::OutOfMemory);
                    assert_eq!(out, vec![7.0f32, 8.0]);
                }
            }
        }
    }
}
