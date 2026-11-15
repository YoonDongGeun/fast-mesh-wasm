//! Free-form deformation, brush deformation, and vertex protection domain.
//!
//! Includes 3D Sederberg-Parry [`ffd`], As-Rigid-As-Possible ([`arap`]),
//! regularized elastostatic [`kelvinlet`]s, Extended Position-Based Dynamics
//! ([`xpbd`]) with self-collision avoidance, alongside brush algorithms
//! (`crate::sculpt`) and vertex protection (`crate::protect`).

pub mod arap;
pub mod ffd;
pub mod kelvinlet;
pub mod xpbd;

pub use crate::protect::{check_protection, with_protection, with_protection_band};
pub use crate::sculpt::*;
pub use arap::{
    ArapAnchor, ArapHandle, ArapOptions, ArapReport, ArapSolver, ArapWeights, apply_arap,
};
pub use ffd::{
    FfdBasis, FfdFrame, FfdLattice, FfdOptions, FfdOutsideMode, apply_ffd, apply_ffd_protected,
};
pub use kelvinlet::{
    KelvinletLoad, KelvinletOptions, KelvinletScale, apply_kelvinlet, apply_kelvinlet_protected,
};
pub use xpbd::{XpbdOptions, XpbdReport, apply_xpbd};

use crate::Result;
use fast_mesh_store::{Transaction, VertexId};

/// Unified summary report returned by [`apply_ffd`], [`apply_kelvinlet`], and [`apply_freeform`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FreeformReport {
    /// Number of active/simulated vertices evaluated.
    pub vertices_evaluated: usize,
    /// Number of vertices whose world positions changed.
    pub vertices_moved: usize,
    /// Maximum Euclidean displacement across moved vertices.
    pub max_displacement: f32,
    /// Solver iterations or substeps executed (`1` for closed-form analytic operators).
    pub iterations: usize,
    /// Whether an iterative solver converged within tolerance (`true` for analytic operators).
    pub converged: bool,
    /// Number of narrow-phase self-collision contacts resolved (`0` unless enabled in XPBD).
    pub self_collision_contacts_resolved: usize,
}

impl FreeformReport {
    /// Returns an empty no-op report (`0` evaluated, `0` moved, `converged: true`).
    #[inline]
    pub fn empty() -> Self {
        Self {
            vertices_evaluated: 0,
            vertices_moved: 0,
            max_displacement: 0.0,
            iterations: 0,
            converged: true,
            self_collision_contacts_resolved: 0,
        }
    }
}

impl From<ArapReport> for FreeformReport {
    #[inline]
    fn from(r: ArapReport) -> Self {
        Self {
            vertices_evaluated: r.vertices_evaluated,
            vertices_moved: r.vertices_moved,
            max_displacement: r.max_displacement,
            iterations: r.iterations_run,
            converged: r.converged,
            self_collision_contacts_resolved: 0,
        }
    }
}

impl From<XpbdReport> for FreeformReport {
    #[inline]
    fn from(r: XpbdReport) -> Self {
        Self {
            vertices_evaluated: r.vertices_evaluated,
            vertices_moved: r.vertices_moved,
            max_displacement: r.max_displacement,
            iterations: r.total_iterations,
            converged: true,
            self_collision_contacts_resolved: r.self_collision_contacts_resolved,
        }
    }
}

/// Unified free-form deformation operator dispatched by [`apply_freeform`].
#[derive(Clone, Debug, PartialEq)]
pub enum FreeformOp {
    /// 3D Sederberg-Parry Free-Form Deformation cage.
    Ffd(FfdOptions),
    /// As-Rigid-As-Possible surface deformation.
    Arap(ArapOptions),
    /// Regularized elastostatic Kelvinlet load.
    Kelvinlet(KelvinletOptions),
    /// Extended Position-Based Dynamics solver.
    Xpbd(XpbdOptions),
}

/// Applies any [`FreeformOp`] variant atomically to `tx`.
pub fn apply_freeform(tx: &mut Transaction, op: &FreeformOp) -> Result<FreeformReport> {
    match op {
        FreeformOp::Ffd(opts) => apply_ffd(tx, opts),
        FreeformOp::Arap(opts) => apply_arap(tx, opts).map(FreeformReport::from),
        FreeformOp::Kelvinlet(opts) => apply_kelvinlet(tx, opts),
        FreeformOp::Xpbd(opts) => apply_xpbd(tx, opts).map(FreeformReport::from),
    }
}

/// Applies any [`FreeformOp`] variant modulated by per-vertex protection weights
/// (`0` leaves vertex frozen at rest bits, `1` keeps full deformation).
pub fn apply_freeform_protected(
    tx: &mut Transaction,
    op: &FreeformOp,
    protected_vertices: &[VertexId],
    protection_weights: &[f32],
) -> Result<FreeformReport> {
    with_protection(tx, protected_vertices, protection_weights, |tx| {
        apply_freeform(tx, op)
    })
}

#[cfg(test)]
mod tests {
    use fast_mesh::compute_vertex_normals;
    use fast_mesh_store::{Mesh, Transaction, VertexId};

    use super::*;
    use crate::sculpt::{brush_weight, falloff_value};
    use crate::test_util::{edit, mesh_of, parts, sweep};
    use crate::{OpsError, Result, fail_allocations_after};

    /// A 3x3 flat grid in `z = 0`, spacing 1, centered at the origin; vertex
    /// `(x, y)` has id `y * 3 + x`, wound counter-clockwise from `+z`.
    fn centered_grid() -> Mesh {
        let mut positions = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                positions.extend([x as f32 - 1.0, y as f32 - 1.0, 0.0]);
            }
        }
        let mut triangles = Vec::new();
        for y in 0..2u32 {
            for x in 0..2u32 {
                let a = y * 3 + x;
                triangles.extend([[a, a + 1, a + 4], [a, a + 4, a + 3]]);
            }
        }
        mesh_of(&positions, &triangles)
    }

    /// The smooth normal of the center vertex (id 4), straight from
    /// `compute_vertex_normals`, whatever its sign convention.
    fn center_normal(mesh: &Mesh) -> [f32; 3] {
        let (positions, triangles) = parts(mesh);
        let indices: Vec<u32> = triangles.iter().flatten().copied().collect();
        let normals = compute_vertex_normals(&positions, &indices).unwrap();
        [normals[4 * 3], normals[4 * 3 + 1], normals[4 * 3 + 2]]
    }

    fn options(center: [f32; 3], radius: f32, falloff: Falloff, strength: f32) -> InflateOptions {
        InflateOptions::builder()
            .center(center)
            .radius(radius)
            .falloff(falloff)
            .strength(strength)
            .build()
    }

    /// Displacement length of vertex `id` between two flat position buffers.
    fn displacement(old: &[f32], new: &[f32], id: usize) -> f32 {
        (0..3)
            .map(|k| (new[id * 3 + k] - old[id * 3 + k]).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn falloff_exact_values() {
        assert!((falloff_value(Falloff::Linear, 0.5) - 0.5).abs() < 1e-6);
        assert!((falloff_value(Falloff::Smooth, 0.0) - 1.0).abs() < 1e-6);
        assert!((falloff_value(Falloff::Smooth, 1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn falloff_boundary_values() {
        assert!((falloff_value(Falloff::Linear, 0.0) - 1.0).abs() < 1e-6);
        assert!((falloff_value(Falloff::Linear, 1.0) - 0.0).abs() < 1e-6);
        assert!((falloff_value(Falloff::Smooth, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn inflate_moves_center_along_normal() {
        let mut mesh = centered_grid();
        let old = parts(&mesh).0;
        let n = center_normal(&mesh);
        let options = options([0.0; 3], 10.0, Falloff::Linear, 2.0);
        let (report, empty) = edit(&mut mesh, |tx| inflate(tx, &options));
        assert!(!empty);
        assert_eq!(report.moved, 9);
        assert!((report.max_displacement - 2.0).abs() < 1e-4);
        let new = parts(&mesh).0;
        for k in 0..3 {
            assert!(
                (new[4 * 3 + k] - (old[4 * 3 + k] + 2.0 * n[k])).abs() < 1e-4,
                "center follows strength * normal"
            );
        }
        assert!(displacement(&old, &new, 0) < displacement(&old, &new, 4));
    }

    #[test]
    fn inflate_negative_strength_deflates() {
        let mut mesh = centered_grid();
        let old = parts(&mesh).0;
        let n = center_normal(&mesh);
        let options = options([0.0; 3], 10.0, Falloff::Linear, -2.0);
        let (report, empty) = edit(&mut mesh, |tx| inflate(tx, &options));
        assert!(!empty);
        assert_eq!(report.moved, 9);
        assert!((report.max_displacement - 2.0).abs() < 1e-4);
        let new = parts(&mesh).0;
        for k in 0..3 {
            assert!(
                (new[4 * 3 + k] - (old[4 * 3 + k] - 2.0 * n[k])).abs() < 1e-4,
                "negative strength pulls along the normal"
            );
        }
    }

    #[test]
    fn inflate_empty_region_moves_nothing() {
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = options([100.0, 0.0, 0.0], 1.0, Falloff::Linear, 2.0);
        let mut tx = mesh.begin_edit();
        let report = inflate(&mut tx, &options).unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn inflate_zero_radius_rejected() {
        let mut mesh = centered_grid();
        let options = options([0.0; 3], 0.0, Falloff::Linear, 2.0);
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            inflate(&mut tx, &options),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(tx.commit().unwrap().is_empty());
    }

    #[test]
    fn inflate_negative_radius_rejected() {
        let mut mesh = centered_grid();
        let options = options([0.0; 3], -1.0, Falloff::Linear, 2.0);
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            inflate(&mut tx, &options),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(tx.commit().unwrap().is_empty());
    }

    #[test]
    fn inflate_nan_strength_rejected() {
        let mut mesh = centered_grid();
        let options = options([0.0; 3], 10.0, Falloff::Linear, f32::NAN);
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            inflate(&mut tx, &options),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(tx.commit().unwrap().is_empty());
    }

    fn smooth_options(
        center: [f32; 3],
        radius: f32,
        falloff: Falloff,
        strength: f32,
        cycles: u32,
    ) -> SmoothOptions {
        SmoothOptions::builder()
            .center(center)
            .radius(radius)
            .falloff(falloff)
            .strength(strength)
            .cycles(cycles)
            .build()
    }

    #[test]
    fn taubin_pyramid_exact_after_one_cycle() {
        assert_eq!(TAUBIN_LAMBDA, 0.33);
        assert_eq!(TAUBIN_MU, -0.34);
        let positions = [
            0.0, 0.0, 0.0, // A
            1.0, 0.0, 0.0, // B
            1.0, 1.0, 0.0, // C
            0.0, 1.0, 0.0, // D
            0.5, 0.5, 1.0, // E
        ];
        let triangles = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]];
        let mut mesh = mesh_of(&positions, &triangles);
        let options = smooth_options([0.5, 0.5, 1.0], 0.1, Falloff::Linear, 1.0, 1);
        let (report, empty) = edit(&mut mesh, |tx| smooth(tx, &options));
        assert!(!empty);
        assert_eq!(report.moved, 1);
        let (flat, _) = parts(&mesh);
        assert!((flat[4 * 3 + 2] - 0.8978).abs() < 1e-4);
        assert_eq!(flat[4 * 3], 0.5);
        assert_eq!(flat[4 * 3 + 1], 0.5);
        for v in 0..4 {
            assert_eq!(&flat[v * 3..v * 3 + 3], &positions[v * 3..v * 3 + 3]);
        }
        assert!((report.max_displacement - (1.0 - 0.8978)).abs() < 1e-4);
    }

    #[test]
    fn laplacian_pyramid_melts_to_centroid() {
        assert_eq!(
            SmoothOptions::builder().build().method,
            SmoothMethod::Taubin
        );
        let positions = [
            0.0, 0.0, 0.0, // A
            1.0, 0.0, 0.0, // B
            1.0, 1.0, 0.0, // C
            0.0, 1.0, 0.0, // D
            0.5, 0.5, 1.0, // E
        ];
        let triangles = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]];
        let mut mesh = mesh_of(&positions, &triangles);
        let options = SmoothOptions::builder()
            .center([0.5, 0.5, 1.0])
            .radius(0.1)
            .falloff(Falloff::Linear)
            .strength(1.0)
            .cycles(1)
            .method(SmoothMethod::Laplacian)
            .build();
        let (report, empty) = edit(&mut mesh, |tx| smooth(tx, &options));
        assert!(!empty);
        assert_eq!(report.moved, 1);
        let (flat, _) = parts(&mesh);
        assert_eq!(&flat[4 * 3..4 * 3 + 3], &[0.5, 0.5, 0.0]);
        assert!((report.max_displacement - 1.0).abs() < 1e-6);
        let mut mesh = mesh_of(&positions, &triangles);
        let options = SmoothOptions::builder()
            .center([0.5, 0.5, 1.0])
            .radius(0.1)
            .falloff(Falloff::Linear)
            .strength(0.5)
            .cycles(2)
            .method(SmoothMethod::Laplacian)
            .build();
        edit(&mut mesh, |tx| smooth(tx, &options).map(drop));
        let (flat, _) = parts(&mesh);
        assert!((flat[4 * 3 + 2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn smooth_zero_cycles_moves_nothing() {
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = smooth_options([0.0; 3], 10.0, Falloff::Linear, 1.0, 0);
        let mut tx = mesh.begin_edit();
        let report = smooth(&mut tx, &options).unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn smooth_empty_region_moves_nothing() {
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = smooth_options([100.0, 0.0, 0.0], 1.0, Falloff::Linear, 1.0, 1);
        let mut tx = mesh.begin_edit();
        let report = smooth(&mut tx, &options).unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn smooth_invalid_brush_rejected() {
        let mut mesh = centered_grid();
        let cases = [
            smooth_options([0.0; 3], 0.0, Falloff::Linear, 1.0, 1),
            smooth_options([0.0; 3], -1.0, Falloff::Linear, 1.0, 1),
            smooth_options([0.0; 3], f32::NAN, Falloff::Linear, 1.0, 1),
            smooth_options([0.0; 3], 10.0, Falloff::Linear, f32::NAN, 1),
            smooth_options([f32::INFINITY, 0.0, 0.0], 10.0, Falloff::Linear, 1.0, 1),
        ];
        for options in cases {
            let mut tx = mesh.begin_edit();
            assert!(matches!(
                smooth(&mut tx, &options),
                Err(OpsError::InvalidInput(_))
            ));
            assert!(tx.commit().unwrap().is_empty());
        }
    }

    #[test]
    fn smooth_nonmanifold_fan_moves_toward_centroid() {
        let positions = [
            0.0, 0.0, 1.0, // E
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        ];
        let triangles = [[0, 1, 2], [0, 3, 4]];
        let mut mesh = mesh_of(&positions, &triangles);
        let options = smooth_options([0.0, 0.0, 1.0], 0.1, Falloff::Linear, 1.0, 1);
        let (report, _) = edit(&mut mesh, |tx| smooth(tx, &options));
        assert_eq!(report.moved, 1);
        let (flat, _) = parts(&mesh);
        let dist = |p: &[f32]| ((p[0] - 0.5).powi(2) + (p[1] - 0.5).powi(2) + p[2].powi(2)).sqrt();
        assert!(dist(&flat[0..3]) < dist(&positions[0..3]));
    }

    #[test]
    fn inflate_nonfinite_center_rejected() {
        let mut mesh = centered_grid();
        let options = options([f32::INFINITY, 0.0, 0.0], 10.0, Falloff::Linear, 2.0);
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            inflate(&mut tx, &options),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(tx.commit().unwrap().is_empty());
    }

    #[test]
    fn inflate_empty_region_skips_temporaries() {
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = options([100.0, 0.0, 0.0], 1.0, Falloff::Linear, 2.0);
        let mut tx = mesh.begin_edit();
        fail_allocations_after(Some(0));
        let report = inflate(&mut tx, &options);
        fail_allocations_after(None);
        let report = report.unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn smooth_empty_region_skips_temporaries() {
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = smooth_options([100.0, 0.0, 0.0], 1.0, Falloff::Linear, 1.0, 1);
        let mut tx = mesh.begin_edit();
        fail_allocations_after(Some(0));
        let report = smooth(&mut tx, &options);
        fail_allocations_after(None);
        let report = report.unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn inflate_out_of_memory_is_reported_and_writes_nothing() {
        let options = options([0.0; 3], 10.0, Falloff::Linear, 2.0);
        assert!(sweep(centered_grid(), |tx| inflate(tx, &options).map(drop)) > 5);
    }

    #[test]
    fn smooth_out_of_memory_is_reported_and_writes_nothing() {
        let options = smooth_options([0.0; 3], 10.0, Falloff::Linear, 1.0, 1);
        assert!(sweep(centered_grid(), |tx| smooth(tx, &options).map(drop)) > 5);
    }

    fn grab_options(
        center: [f32; 3],
        radius: f32,
        falloff: Falloff,
        strength: f32,
        direction: [f32; 3],
    ) -> GrabOptions {
        GrabOptions::builder()
            .center(center)
            .radius(radius)
            .falloff(falloff)
            .strength(strength)
            .direction(direction)
            .build()
    }

    #[test]
    fn grab_moves_region_along_direction() {
        let mut mesh = centered_grid();
        let old = parts(&mesh).0;
        let options = grab_options([0.0; 3], 10.0, Falloff::Linear, 1.0, [2.0, 0.0, 0.0]);
        let (report, empty) = edit(&mut mesh, |tx| grab(tx, &options));
        assert!(!empty);
        assert_eq!(report.moved, 9);
        assert!((report.max_displacement - 2.0).abs() < 1e-4);
        let new = parts(&mesh).0;
        for k in 0..3 {
            let want = [2.0, 0.0, 0.0][k];
            assert!(
                (new[4 * 3 + k] - (old[4 * 3 + k] + want)).abs() < 1e-4,
                "center follows strength * direction"
            );
        }
        assert!(displacement(&old, &new, 0) < displacement(&old, &new, 4));
    }

    #[test]
    fn grab_strength_scales_the_drag() {
        let mut mesh = centered_grid();
        let old = parts(&mesh).0;
        let options = grab_options([0.0; 3], 10.0, Falloff::Linear, 0.5, [2.0, 0.0, 0.0]);
        let (report, _) = edit(&mut mesh, |tx| grab(tx, &options));
        assert_eq!(report.moved, 9);
        let new = parts(&mesh).0;
        assert!((new[4 * 3] - (old[4 * 3] + 1.0)).abs() < 1e-4);
    }

    #[test]
    fn grab_zero_direction_moves_nothing() {
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = grab_options([0.0; 3], 10.0, Falloff::Linear, 1.0, [0.0; 3]);
        let mut tx = mesh.begin_edit();
        let report = grab(&mut tx, &options).unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn grab_empty_region_moves_nothing() {
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = grab_options(
            [100.0, 0.0, 0.0],
            1.0,
            Falloff::Linear,
            1.0,
            [2.0, 0.0, 0.0],
        );
        let mut tx = mesh.begin_edit();
        let report = grab(&mut tx, &options).unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn grab_invalid_brush_rejected() {
        let mut mesh = centered_grid();
        let cases = [
            grab_options([0.0; 3], 0.0, Falloff::Linear, 1.0, [1.0, 0.0, 0.0]),
            grab_options([0.0; 3], -1.0, Falloff::Linear, 1.0, [1.0, 0.0, 0.0]),
            grab_options([0.0; 3], f32::NAN, Falloff::Linear, 1.0, [1.0, 0.0, 0.0]),
            grab_options([0.0; 3], 10.0, Falloff::Linear, f32::NAN, [1.0, 0.0, 0.0]),
            grab_options(
                [f32::INFINITY, 0.0, 0.0],
                10.0,
                Falloff::Linear,
                1.0,
                [1.0, 0.0, 0.0],
            ),
            grab_options([0.0; 3], 10.0, Falloff::Linear, 1.0, [f32::NAN, 0.0, 0.0]),
            grab_options(
                [0.0; 3],
                10.0,
                Falloff::Linear,
                1.0,
                [0.0, f32::INFINITY, 0.0],
            ),
        ];
        for options in cases {
            let mut tx = mesh.begin_edit();
            assert!(matches!(
                grab(&mut tx, &options),
                Err(OpsError::InvalidInput(_))
            ));
            assert!(tx.commit().unwrap().is_empty());
        }
    }

    #[test]
    fn grab_empty_region_skips_temporaries() {
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = grab_options(
            [100.0, 0.0, 0.0],
            1.0,
            Falloff::Linear,
            1.0,
            [2.0, 0.0, 0.0],
        );
        let mut tx = mesh.begin_edit();
        fail_allocations_after(Some(0));
        let report = grab(&mut tx, &options);
        fail_allocations_after(None);
        let report = report.unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn grab_out_of_memory_is_reported_and_writes_nothing() {
        let options = grab_options([0.0; 3], 10.0, Falloff::Linear, 1.0, [2.0, 0.0, 0.0]);
        assert!(sweep(centered_grid(), |tx| grab(tx, &options).map(drop)) > 5);
    }

    /// A wavy grid, `n` quads a side, spacing 1, with a dead triangle.
    fn wavy_grid(n: u32) -> Mesh {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                let (fx, fy) = (x as f32, y as f32);
                positions.extend([fx, fy, (fx * 0.7).sin() * (fy * 0.5).cos()]);
            }
        }
        let mut triangles = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                triangles.extend([[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]);
            }
        }
        let mut mesh = mesh_of(&positions, &triangles);
        let mut tx = mesh.begin_edit();
        tx.remove_triangles(&[10]).unwrap();
        tx.commit().unwrap();
        mesh.ensure_incidence().unwrap();
        mesh
    }

    /// Every vertex slot, the widest legal region.
    fn every_slot(mesh: &Mesh) -> Vec<VertexId> {
        (0..mesh.vertex_slots() as VertexId).collect()
    }

    /// The positions after `f`, with its report.
    fn after<R>(mesh: &mut Mesh, f: impl FnOnce(&mut Transaction) -> Result<R>) -> (R, Vec<f32>) {
        let (r, _) = edit(mesh, f);
        (r, parts(mesh).0)
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn region_functions_match_the_whole_mesh_ones_bit_for_bit() {
        let brush = Brush {
            center: [5.3, 4.1, 0.0],
            radius: 3.2,
            falloff: Falloff::Smooth,
            strength: 0.4,
        };
        let near = |mesh: &Mesh| -> Vec<VertexId> {
            (0..mesh.vertex_slots() as VertexId)
                .filter(|&v| brush_weight(&brush, &mesh.positions()[v as usize]).is_some())
                .collect()
        };
        let inflate_options = InflateOptions {
            brush,
            ..Default::default()
        };
        let (mut a, mut b, mut c) = (wavy_grid(12), wavy_grid(12), wavy_grid(12));
        let (ra, pa) = after(&mut a, |tx| inflate(tx, &inflate_options));
        let region = near(&b);
        let (rb, pb) = after(&mut b, |tx| inflate_region(tx, &inflate_options, &region));
        let region = every_slot(&c);
        let (rc, pc) = after(&mut c, |tx| inflate_region(tx, &inflate_options, &region));
        assert_eq!(ra, rb);
        assert_eq!(ra, rc);
        assert_eq!(bits(&pa), bits(&pb));
        assert_eq!(bits(&pa), bits(&pc));

        for method in [SmoothMethod::Taubin, SmoothMethod::Laplacian] {
            let smooth_options = SmoothOptions {
                brush,
                cycles: 3,
                method,
                ..Default::default()
            };
            let (mut a, mut b) = (wavy_grid(12), wavy_grid(12));
            let (ra, pa) = after(&mut a, |tx| smooth(tx, &smooth_options));
            let region = near(&b);
            let (rb, pb) = after(&mut b, |tx| smooth_region(tx, &smooth_options, &region));
            assert_eq!(ra, rb);
            assert_eq!(bits(&pa), bits(&pb));
        }

        let grab_options = GrabOptions {
            brush,
            direction: [0.3, -0.2, 0.9],
            ..Default::default()
        };
        let (mut a, mut b) = (wavy_grid(12), wavy_grid(12));
        let (ra, pa) = after(&mut a, |tx| grab(tx, &grab_options));
        let region = near(&b);
        let (rb, pb) = after(&mut b, |tx| grab_region(tx, &grab_options, &region));
        assert_eq!(ra, rb);
        assert_eq!(bits(&pa), bits(&pb));
    }

    #[test]
    fn region_functions_ignore_candidates_outside_the_brush() {
        let mut mesh = wavy_grid(6);
        let options = GrabOptions {
            brush: Brush {
                center: [0.0, 0.0, 0.0],
                radius: 1.5,
                falloff: Falloff::Smooth,
                strength: 1.0,
            },
            direction: [0.0, 0.0, 1.0],
            ..Default::default()
        };
        let region = every_slot(&mesh);
        let (report, _) = after(&mut mesh, |tx| grab_region(tx, &options, &region));
        assert!(report.moved > 0 && report.moved < 8);
    }

    #[test]
    fn region_functions_need_the_incidence_index_and_validate() {
        let mut mesh = centered_grid();
        let region = every_slot(&mesh);
        let good = InflateOptions::default();
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            inflate_region(&mut tx, &good, &region),
            Err(OpsError::InvalidInput(_))
        ));
        drop(tx);
        mesh.ensure_incidence().unwrap();
        let bad = InflateOptions::builder().radius(0.0).build();
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            inflate_region(&mut tx, &bad, &region),
            Err(OpsError::InvalidInput(_))
        ));
        let bad = SmoothOptions::builder().radius(-1.0).cycles(1).build();
        assert!(matches!(
            smooth_region(&mut tx, &bad, &region),
            Err(OpsError::InvalidInput(_))
        ));
        let bad = GrabOptions::builder().direction([f32::NAN; 3]).build();
        assert!(matches!(
            grab_region(&mut tx, &bad, &region),
            Err(OpsError::InvalidInput(_))
        ));
    }

    fn max_normal_step(mesh: &Mesh, n: u32, center: f32, radius: f32, band: (f32, f32)) -> f32 {
        let (positions, triangles) = parts(mesh);
        let indices: Vec<u32> = triangles.iter().flatten().copied().collect();
        let normals = compute_vertex_normals(&positions, &indices).unwrap();
        let at = |x: u32, y: u32| ((y * (n + 1) + x) * 3) as usize;
        let mut worst = 0.0f32;
        for y in 1..n {
            for x in 1..n {
                let t = (x as f32 - center).hypot(y as f32 - center) / radius;
                if t < band.0 || t > band.1 {
                    continue;
                }
                let a = at(x, y);
                for b in [at(x + 1, y), at(x, y + 1)] {
                    let dot: f32 = (0..3).map(|k| normals[a + k] * normals[b + k]).sum();
                    worst = worst.max(dot.clamp(-1.0, 1.0).acos());
                }
            }
        }
        worst
    }

    fn flat_dabbed(n: u32, radius: f32, falloff: Falloff) -> Mesh {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                positions.extend([x as f32, y as f32, 0.0]);
            }
        }
        let mut triangles = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                triangles.extend([[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]);
            }
        }
        let mut mesh = mesh_of(&positions, &triangles);
        let c = n as f32 / 2.0;
        let options = InflateOptions::builder()
            .center([c, c, 0.0])
            .radius(radius)
            .falloff(falloff)
            .strength(radius * 0.25)
            .build();
        edit(&mut mesh, |tx| inflate(tx, &options));
        mesh
    }

    #[test]
    fn smooth_falloff_has_no_crease_at_the_rim() {
        let rim_step = |n: u32, falloff: Falloff| {
            let radius = n as f32 / 4.0;
            let band = (1.0 - 1.5 / radius, 1.0 + 1.5 / radius);
            let mesh = flat_dabbed(n, radius, falloff);
            max_normal_step(&mesh, n, n as f32 / 2.0, radius, band)
        };
        let (smooth_coarse, smooth_fine) = (
            rim_step(160, Falloff::Smooth),
            rim_step(320, Falloff::Smooth),
        );
        let (linear_coarse, linear_fine) = (
            rim_step(160, Falloff::Linear),
            rim_step(320, Falloff::Linear),
        );
        assert!(smooth_coarse < 0.06, "smooth rim step {smooth_coarse}");
        assert!(
            smooth_fine < 0.6 * smooth_coarse,
            "smooth rim step shrinks with the edge: {smooth_coarse} to {smooth_fine}"
        );
        assert!(linear_coarse > 0.15, "linear rim step {linear_coarse}");
        assert!(
            linear_fine > 0.9 * linear_coarse,
            "linear rim step stays: {linear_coarse} to {linear_fine}"
        );
    }

    #[test]
    fn freeform_oom_leaves_transaction_untouched() {
        use crate::test_util::{grid, uv_sphere};

        let cloth = grid(4);
        let mut lattice =
            FfdLattice::with_mesh(&cloth, [3, 3, 3], FfdBasis::Bernstein, 0.1).unwrap();
        lattice.set_offset(2, 2, 2, [0.5, 0.2, 0.0]).unwrap();
        let ffd_opts = FfdOptions::new(lattice);
        let ffd_points = sweep(grid(4), |tx| apply_ffd(tx, &ffd_opts).map(|_| ()));
        assert!(ffd_points > 0);

        let v0: VertexId = 0;
        let v_last: VertexId = cloth.vertex_slots() as u32 - 1;
        let arap_opts = ArapOptions::new()
            .with_anchors(vec![ArapAnchor::pin(v0)])
            .with_handles(vec![
                ArapHandle::from_displacement(&cloth, v_last, [0.0, 0.0, 0.5]).unwrap(),
            ])
            .with_iterations(3);
        let arap_points = sweep(grid(4), |tx| apply_arap(tx, &arap_opts).map(|_| ()));
        assert!(arap_points > 0);

        let sphere = uv_sphere(4, 6, 1.0);
        let kel_opts = KelvinletOptions::new(
            [0.0, 0.0, 0.0],
            KelvinletLoad::Grab {
                force: [0.0, 1.0, 0.0],
            },
        );
        let kel_points = sweep(sphere, |tx| apply_kelvinlet(tx, &kel_opts).map(|_| ()));
        assert!(kel_points > 0);

        let xpbd_opts = XpbdOptions::new()
            .with_substeps(2)
            .with_iterations(2)
            .with_pins(vec![(v0, cloth.positions()[0])])
            .with_gravity([0.0, -1.0, 0.0])
            .with_self_collision(true, 0.05);
        let xpbd_points = sweep(grid(4), |tx| apply_xpbd(tx, &xpbd_opts).map(|_| ()));
        assert!(xpbd_points > 0);
    }

    #[test]
    fn freeform_parallel_determinism_unit() {
        use crate::test_util::{assert_identical_across_threads, assert_parallel_identical, grid};

        let to_bits = |m: &Mesh| -> Vec<[u32; 3]> {
            m.positions()
                .iter()
                .map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
                .collect()
        };

        // grid(48) has 49*49 = 2401 vertices (> PAR_THRESHOLD = 2048).
        assert_parallel_identical("ffd", || {
            let mut m = grid(48);
            let mut lat =
                FfdLattice::with_mesh(&m, [3, 2, 3], FfdBasis::CubicBSpline, 0.1).unwrap();
            lat.set_offset(2, 1, 2, [0.3, 0.7, -0.2]).unwrap();
            let opts = FfdOptions::new(lat);
            let mut tx = m.begin_edit();
            apply_ffd(&mut tx, &opts).unwrap();
            tx.commit().unwrap();
            to_bits(&m)
        });

        assert_parallel_identical("kelvinlet", || {
            let mut m = grid(48);
            let opts = KelvinletOptions::new(
                [24.0, 0.0, 24.0],
                KelvinletLoad::Grab {
                    force: [0.0, 1.5, 0.0],
                },
            )
            .with_epsilon(4.0);
            let mut tx = m.begin_edit();
            apply_kelvinlet(&mut tx, &opts).unwrap();
            tx.commit().unwrap();
            to_bits(&m)
        });

        assert_parallel_identical("arap", || {
            let mut m = grid(48);
            let opts = ArapOptions::new()
                .with_anchors(vec![ArapAnchor::pin(0), ArapAnchor::pin(48)])
                .with_handles(vec![ArapHandle::new(2400, [48.0, 5.0, 48.0])])
                .with_solver(ArapSolver::Pcg)
                .with_iterations(4);
            let mut tx = m.begin_edit();
            apply_arap(&mut tx, &opts).unwrap();
            tx.commit().unwrap();
            to_bits(&m)
        });

        assert_identical_across_threads("xpbd", || {
            let mut m = grid(8);
            let opts = XpbdOptions::new()
                .with_substeps(3)
                .with_iterations(4)
                .with_pins(vec![(0, [0.0, 0.0, 0.0]), (8, [8.0, 0.0, 0.0])])
                .with_gravity([0.0, -5.0, 0.0])
                .with_self_collision(true, 0.1);
            let mut tx = m.begin_edit();
            apply_xpbd(&mut tx, &opts).unwrap();
            tx.commit().unwrap();
            to_bits(&m)
        });
    }
}
