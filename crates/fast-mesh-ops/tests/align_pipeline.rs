//! Integration tests for the multi-stage alignment pipeline in `fast_mesh_ops::align`.

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_ops::align::{
    CoarseAlignMethod, IcpObjective, LandmarkPair, RegistrationOptions, align_landmarks,
    detect_symmetry_plane, register_multistage, sample_surface,
};
use fast_mesh_ops::{IcpStatus, OpsError};
use fast_mesh_store::{Mesh, par};

fn rot_axis_angle(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let k = [axis[0] / len, axis[1] / len, axis[2] / len];
    let (s, c) = angle.sin_cos();
    let v = 1.0 - c;
    [
        [
            c + k[0] * k[0] * v,
            k[0] * k[1] * v - k[2] * s,
            k[0] * k[2] * v + k[1] * s,
        ],
        [
            k[1] * k[0] * v + k[2] * s,
            c + k[1] * k[1] * v,
            k[1] * k[2] * v - k[0] * s,
        ],
        [
            k[2] * k[0] * v - k[1] * s,
            k[2] * k[1] * v + k[0] * s,
            c + k[2] * k[2] * v,
        ],
    ]
}

fn mat3_det(r: [[f64; 3]; 3]) -> f64 {
    r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
}

fn transform_pt(r: [[f64; 3]; 3], t: [f64; 3], scale: f64, p: [f64; 3]) -> [f64; 3] {
    [
        scale * (r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2]) + t[0],
        scale * (r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2]) + t[1],
        scale * (r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2]) + t[2],
    ]
}

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn mesh_from_indexed(positions: &[[f64; 3]], triangles: &[[u32; 3]]) -> Mesh {
    let flat_pos: Vec<f32> = positions
        .iter()
        .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    let flat_idx: Vec<u32> = triangles.iter().flat_map(|t| *t).collect();
    Mesh::from_buffers(&flat_pos, &flat_idx).expect("valid test mesh")
}

/// Generates an asymmetric lumpy closed surface with distinct principal axes
/// `(ax, ay, az)` and a keyed feature breaking 90/180 deg symmetries.
fn build_asymmetric_blob(
    stacks: usize,
    slices: usize,
    radii: [f64; 3],
    r_pose: [[f64; 3]; 3],
    t_pose: [f64; 3],
) -> Mesh {
    let point = |x: f64, y: f64, z: f64| {
        let key = if x > 0.2 && y > 0.0 {
            0.35 * ((x - 0.2) * (y + 0.1)).max(0.0)
        } else {
            0.0
        };
        let asym = 1.0 + 0.18 * (2.5 * x + 0.7).sin() + 0.14 * (2.0 * y).cos() * (z + 0.3) + key;
        let local = [
            radii[0] * asym * x,
            radii[1] * asym * y,
            radii[2] * asym * z,
        ];
        transform_pt(r_pose, t_pose, 1.0, local)
    };
    let mut pos = Vec::new();
    pos.push(point(0.0, 0.0, 1.0));
    for i in 1..stacks {
        let phi = std::f64::consts::PI * (i as f64) / (stacks as f64);
        for j in 0..slices {
            let a = std::f64::consts::TAU * (j as f64) / (slices as f64);
            pos.push(point(phi.sin() * a.cos(), phi.sin() * a.sin(), phi.cos()));
        }
    }
    pos.push(point(0.0, 0.0, -1.0));
    let south = (1 + (stacks - 1) * slices) as u32;
    let ring = |i: usize, j: usize| (1 + (i - 1) * slices + (j % slices)) as u32;
    let mut tris = Vec::new();
    for j in 0..slices {
        tris.push([0, ring(1, j), ring(1, j + 1)]);
    }
    for i in 1..stacks - 1 {
        for j in 0..slices {
            let (a, b, c, d) = (
                ring(i, j),
                ring(i + 1, j),
                ring(i + 1, j + 1),
                ring(i, j + 1),
            );
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    for j in 0..slices {
        tris.push([ring(stacks - 1, j), south, ring(stacks - 1, j + 1)]);
    }
    mesh_from_indexed(&pos, &tris)
}

/// Builds a smooth open doubly-curved cylindrical/undulating patch for objective comparison.
fn build_curved_patch(
    u_steps: usize,
    v_steps: usize,
    r_pose: [[f64; 3]; 3],
    t_pose: [f64; 3],
) -> Mesh {
    let mut pos = Vec::with_capacity((u_steps + 1) * (v_steps + 1));
    for iv in 0..=v_steps {
        let z = -1.2 + 2.4 * (iv as f64 / v_steps as f64);
        for iu in 0..=u_steps {
            let theta = -0.95 + 1.9 * (iu as f64 / u_steps as f64);
            let rad =
                2.2 + 0.45 * (2.0 * theta).cos() + 0.35 * (2.0 * z).cos() + 0.15 * z * theta.sin();
            let p = [rad * theta.cos(), rad * theta.sin(), z];
            pos.push(transform_pt(r_pose, t_pose, 1.0, p));
        }
    }
    let idx = |iu: usize, iv: usize| (iv * (u_steps + 1) + iu) as u32;
    let mut tris = Vec::with_capacity(u_steps * v_steps * 2);
    for iv in 0..v_steps {
        for iu in 0..u_steps {
            let (a, b, c, d) = (
                idx(iu, iv),
                idx(iu + 1, iv),
                idx(iu + 1, iv + 1),
                idx(iu, iv + 1),
            );
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    mesh_from_indexed(&pos, &tris)
}

/// Appends an axis-aligned box `[min[0]..max[0], min[1]..max[1], min[2]..max[2]]`
/// transformed by `(r_pose, t_pose)` to `(pos, tris)`.
fn push_box(
    pos: &mut Vec<[f64; 3]>,
    tris: &mut Vec<[u32; 3]>,
    min: [f64; 3],
    max: [f64; 3],
    r_pose: [[f64; 3]; 3],
    t_pose: [f64; 3],
) {
    let base = pos.len() as u32;
    let corners = [
        [min[0], min[1], min[2]],
        [max[0], min[1], min[2]],
        [max[0], max[1], min[2]],
        [min[0], max[1], min[2]],
        [min[0], min[1], max[2]],
        [max[0], min[1], max[2]],
        [max[0], max[1], max[2]],
        [min[0], max[1], max[2]],
    ];
    for c in corners {
        pos.push(transform_pt(r_pose, t_pose, 1.0, c));
    }
    let box_tris = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ];
    for t in box_tris {
        tris.push([base + t[0], base + t[1], base + t[2]]);
    }
}

#[test]
fn umeyama_landmark_exact_recovery_rigid_and_scaled() {
    let r_true = rot_axis_angle([1.0, -2.0, 0.7], 2.01);
    let t_true = [3.2, -1.7, 5.4];
    let src_pts = [
        [0.0, 0.0, 0.0],
        [2.5, 0.2, -0.4],
        [-0.8, 3.1, 0.9],
        [1.1, -1.4, 2.7],
        [0.4, 0.8, -2.2],
    ];

    let rigid_pairs: Vec<LandmarkPair> = src_pts
        .iter()
        .map(|&p| LandmarkPair {
            source: p,
            target: transform_pt(r_true, t_true, 1.0, p),
        })
        .collect();
    let res_rigid = align_landmarks(&rigid_pairs, false).expect("rigid landmarks succeed");
    assert!(res_rigid.rmse < 1e-10, "rigid rmse {}", res_rigid.rmse);
    assert!((res_rigid.scale - 1.0).abs() < 1e-12);
    assert!((mat3_det(res_rigid.rotation_matrix()) - 1.0).abs() < 1e-10);
    for pair in &rigid_pairs {
        assert!(dist3(res_rigid.apply(pair.source), pair.target) < 1e-10);
    }

    let s_true = 2.75;
    let scaled_pairs: Vec<LandmarkPair> = src_pts
        .iter()
        .map(|&p| LandmarkPair {
            source: p,
            target: transform_pt(r_true, t_true, s_true, p),
        })
        .collect();
    let res_scaled = align_landmarks(&scaled_pairs, true).expect("scaled landmarks succeed");
    assert!(res_scaled.rmse < 1e-10, "scaled rmse {}", res_scaled.rmse);
    assert!((res_scaled.scale - s_true).abs() < 1e-10);
    assert!((mat3_det(res_scaled.rotation_matrix()) - 1.0).abs() < 1e-10);
    let m4 = res_scaled.transform_matrix();
    for pair in &scaled_pairs {
        let p = pair.source;
        let mapped = [
            m4[0] * p[0] + m4[4] * p[1] + m4[8] * p[2] + m4[12],
            m4[1] * p[0] + m4[5] * p[1] + m4[9] * p[2] + m4[13],
            m4[2] * p[0] + m4[6] * p[1] + m4[10] * p[2] + m4[14],
        ];
        assert!(dist3(mapped, pair.target) < 1e-10);
    }

    // Rank-2 planar landmark set (4 coplanar points rotated cleanly in 3D).
    let planar = [
        [0.0, 0.0, 1.5],
        [3.0, 0.0, 1.5],
        [3.0, 2.0, 1.5],
        [0.5, 2.5, 1.5],
    ];
    let planar_pairs: Vec<LandmarkPair> = planar
        .iter()
        .map(|&p| LandmarkPair {
            source: p,
            target: transform_pt(r_true, [-1.0, 2.0, 0.5], 1.6, p),
        })
        .collect();
    let res_planar = align_landmarks(&planar_pairs, true).expect("planar landmarks succeed");
    assert!(res_planar.rmse < 1e-10, "planar rmse {}", res_planar.rmse);
    assert!((res_planar.scale - 1.6).abs() < 1e-10);
    assert!((mat3_det(res_planar.rotation_matrix()) - 1.0).abs() < 1e-10);
}

#[test]
fn umeyama_prevents_mirror_reflection_and_rejects_invalid_inputs() {
    // Asymmetric 3D landmarks reflected across the X=0 plane (det = -1).
    let src = [
        [1.0, 0.2, 0.3],
        [2.5, 1.7, 0.4],
        [0.6, 2.4, 2.1],
        [3.1, -0.9, 1.8],
    ];
    let reflected_pairs: Vec<LandmarkPair> = src
        .iter()
        .map(|&p| LandmarkPair {
            source: p,
            target: [-p[0] + 4.0, p[1] - 1.0, p[2] + 2.0],
        })
        .collect();
    for allow_scaling in [false, true] {
        let res = align_landmarks(&reflected_pairs, allow_scaling).expect("reflected solves");
        let det = mat3_det(res.rotation_matrix());
        assert!(
            (det - 1.0).abs() < 1e-10,
            "proper rotation violated: det={det}"
        );
        assert!(res.scale > 0.0 && res.rmse.is_finite() && res.rmse > 0.1);
    }

    assert!(matches!(
        align_landmarks(&reflected_pairs[..2], false),
        Err(OpsError::InvalidInput(_))
    ));
    let nan_pairs = [
        LandmarkPair {
            source: [0.0, 0.0, 0.0],
            target: [1.0, 0.0, 0.0],
        },
        LandmarkPair {
            source: [1.0, f64::NAN, 0.0],
            target: [0.0, 1.0, 0.0],
        },
        LandmarkPair {
            source: [0.0, 0.0, 1.0],
            target: [0.0, 0.0, 1.0],
        },
    ];
    assert!(matches!(
        align_landmarks(&nan_pairs, false),
        Err(OpsError::InvalidInput(_))
    ));
    let zero_var = [LandmarkPair {
        source: [2.0, 3.0, 4.0],
        target: [1.0, 0.0, 0.0],
    }; 3];
    assert!(matches!(
        align_landmarks(&zero_var, true),
        Err(OpsError::InvalidInput(_))
    ));
}

#[test]
fn coarse_pca_and_obb_recover_180_and_90_misaligned_scans() {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    // 1) 180-deg flip + large offset: CoarseAlignMethod::None gets stuck, PcaCentroid recovers.
    let target_180 = build_asymmetric_blob(24, 36, [2.2, 1.4, 0.8], identity, [0.0; 3]);
    let target_180_bvh = Bvh::build(&target_180, &BuildOptions::default()).unwrap();
    let r_180 = rot_axis_angle([0.0, 0.0, 1.0], std::f64::consts::PI);
    let source_180 = build_asymmetric_blob(24, 36, [2.2, 1.4, 0.8], r_180, [14.0, -9.0, 5.0]);

    let fail_none = register_multistage(
        &source_180,
        &target_180,
        &target_180_bvh,
        None,
        &RegistrationOptions {
            coarse_method: CoarseAlignMethod::None,
            objective: IcpObjective::SymmetricNormal,
            max_iterations: 40,
            sample_count: 3000,
            ..RegistrationOptions::default()
        },
    )
    .unwrap();
    assert!(
        fail_none.final_rmse > 0.05,
        "expected CoarseAlignMethod::None to fail on distant 180-deg scan, got rmse {}",
        fail_none.final_rmse
    );

    let pass_pca = register_multistage(
        &source_180,
        &target_180,
        &target_180_bvh,
        None,
        &RegistrationOptions {
            coarse_method: CoarseAlignMethod::PcaCentroid,
            objective: IcpObjective::SymmetricNormal,
            max_iterations: 50,
            sample_count: 3000,
            tolerance: 1e-7,
            target_rmse: 1e-7,
            ..RegistrationOptions::default()
        },
    )
    .unwrap();
    assert!(pass_pca.converged, "PCA status {:?}", pass_pca.status);
    assert!(
        pass_pca.final_rmse < 1e-4,
        "PCA rmse {}",
        pass_pca.final_rmse
    );
    assert!(
        fail_none.final_rmse > pass_pca.final_rmse * 100.0,
        "PCA did not vastly outperform None: pca={:.3e} none={:.3e}",
        pass_pca.final_rmse,
        fail_none.final_rmse
    );

    // 2) 90-deg rotation with close principal radii: OBB tests all 24 octahedral permutations.
    let target_90 = build_asymmetric_blob(24, 36, [2.0, 1.92, 0.9], identity, [0.0; 3]);
    let target_90_bvh = Bvh::build(&target_90, &BuildOptions::default()).unwrap();
    let r_90 = rot_axis_angle([0.3, 0.7, 1.0], std::f64::consts::FRAC_PI_2);
    let source_90 = build_asymmetric_blob(24, 36, [2.0, 1.92, 0.9], r_90, [-11.0, 13.0, -6.0]);

    let fail_90_none = register_multistage(
        &source_90,
        &target_90,
        &target_90_bvh,
        None,
        &RegistrationOptions {
            coarse_method: CoarseAlignMethod::None,
            objective: IcpObjective::SymmetricNormal,
            max_iterations: 40,
            sample_count: 3000,
            ..RegistrationOptions::default()
        },
    )
    .unwrap();
    assert!(
        fail_90_none.final_rmse > 0.05,
        "expected None to fail on distant 90-deg scan, got rmse {}",
        fail_90_none.final_rmse
    );

    let pass_obb = register_multistage(
        &source_90,
        &target_90,
        &target_90_bvh,
        None,
        &RegistrationOptions {
            coarse_method: CoarseAlignMethod::OrientedBoundingBox,
            objective: IcpObjective::SymmetricNormal,
            max_iterations: 60,
            sample_count: 3000,
            tolerance: 1e-7,
            target_rmse: 1e-7,
            ..RegistrationOptions::default()
        },
    )
    .unwrap();
    assert!(pass_obb.converged, "OBB status {:?}", pass_obb.status);
    assert!(
        pass_obb.final_rmse < 1e-4,
        "OBB rmse {}",
        pass_obb.final_rmse
    );
}

#[test]
fn symmetric_normal_outperforms_point_to_point_on_curved_patch() {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let target = build_curved_patch(24, 16, identity, [0.0; 3]);
    let target_bvh = Bvh::build(&target, &BuildOptions::default()).unwrap();

    // Rotate curved elliptical patch ~22 deg around its curvature axis + shift.
    let r_off = rot_axis_angle([0.05, 0.08, 1.0], 0.38);
    let source = build_curved_patch(24, 16, r_off, [0.15, -0.12, 0.08]);

    let budget = 8;
    let p2p = register_multistage(
        &source,
        &target,
        &target_bvh,
        None,
        &RegistrationOptions {
            coarse_method: CoarseAlignMethod::None,
            objective: IcpObjective::PointToPoint,
            max_iterations: budget,
            tolerance: 1e-10,
            sample_count: 1500,
            anderson_window: 0,
            ..RegistrationOptions::default()
        },
    )
    .unwrap();

    let sym = register_multistage(
        &source,
        &target,
        &target_bvh,
        None,
        &RegistrationOptions {
            coarse_method: CoarseAlignMethod::None,
            objective: IcpObjective::SymmetricNormal,
            max_iterations: budget,
            tolerance: 1e-10,
            sample_count: 1500,
            anderson_window: 0,
            ..RegistrationOptions::default()
        },
    )
    .unwrap();

    assert!(
        sym.final_rmse < p2p.final_rmse * 0.25 && sym.final_rmse < 1e-3,
        "expected SymmetricNormal ({:.3e}) to substantially outperform PointToPoint ({:.3e}) on curved patch",
        sym.final_rmse,
        p2p.final_rmse
    );

    // With Anderson acceleration and full budget, SymmetricNormal converges tightly.
    let full = register_multistage(
        &source,
        &target,
        &target_bvh,
        None,
        &RegistrationOptions {
            coarse_method: CoarseAlignMethod::None,
            objective: IcpObjective::SymmetricNormal,
            max_iterations: 40,
            tolerance: 1e-7,
            target_rmse: 1e-7,
            sample_count: 1500,
            anderson_window: 3,
            ..RegistrationOptions::default()
        },
    )
    .unwrap();
    assert!(full.converged, "full status {:?}", full.status);
    assert!(full.final_rmse < 1e-5, "full rmse {}", full.final_rmse);
}

#[test]
fn registration_supports_point_to_plane_zero_iters_and_partial_overlap() {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let target = build_asymmetric_blob(24, 32, [2.0, 1.3, 0.9], identity, [0.0; 3]);
    let target_bvh = Bvh::build(&target, &BuildOptions::default()).unwrap();
    let r_off = rot_axis_angle([0.2, 1.0, -0.3], 0.2);
    let source = build_asymmetric_blob(24, 32, [2.0, 1.3, 0.9], r_off, [0.1, -0.15, 0.05]);

    // 0 iterations evaluates pose without converging.
    let zero = register_multistage(
        &source,
        &target,
        &target_bvh,
        None,
        &RegistrationOptions {
            coarse_method: CoarseAlignMethod::None,
            objective: IcpObjective::PointToPlane,
            max_iterations: 0,
            sample_count: 1000,
            ..RegistrationOptions::default()
        },
    )
    .unwrap();
    assert_eq!(zero.iterations, 0);
    assert_eq!(zero.status, IcpStatus::MaxIterations);
    assert!(!zero.converged);
    assert!(zero.final_rmse > 0.05);

    // PointToPlane + Anderson converges tightly.
    let p2pl = register_multistage(
        &source,
        &target,
        &target_bvh,
        None,
        &RegistrationOptions {
            coarse_method: CoarseAlignMethod::None,
            objective: IcpObjective::PointToPlane,
            max_iterations: 50,
            tolerance: 1e-6,
            overlap: 0.85,
            target_rmse: 1e-7,
            sample_count: 2000,
            anderson_window: 3,
            ..RegistrationOptions::default()
        },
    )
    .unwrap();
    assert!(p2pl.converged, "p2pl status {:?}", p2pl.status);
    assert!(p2pl.final_rmse < 1e-5, "p2pl rmse {}", p2pl.final_rmse);
}

#[test]
fn detect_symmetry_plane_symmetric_vs_asymmetric_bracket() {
    // 1) Symmetric bilateral T-wing bracket translated and tilted to an arbitrary diagonal plane.
    let mut sym_pos = Vec::new();
    let mut sym_tris = Vec::new();
    let r_tilt = rot_axis_angle([1.0, 2.0, -0.5], 0.61);
    let center = [4.0, -3.0, 2.5];
    // Symmetric across local X=0 plane ONLY (`normal = r_tilt * [1, 0, 0]`); break Z symmetry.
    push_box(
        &mut sym_pos,
        &mut sym_tris,
        [-0.4, -2.0, -0.4],
        [0.4, 2.0, 0.4],
        r_tilt,
        center,
    );
    push_box(
        &mut sym_pos,
        &mut sym_tris,
        [-2.2, 0.8, -0.2],
        [-0.4, 2.0, 0.9],
        r_tilt,
        center,
    );
    push_box(
        &mut sym_pos,
        &mut sym_tris,
        [0.4, 0.8, -0.2],
        [2.2, 2.0, 0.9],
        r_tilt,
        center,
    );
    let sym_mesh = mesh_from_indexed(&sym_pos, &sym_tris);
    let sym_bvh = Bvh::build(&sym_mesh, &BuildOptions::default()).unwrap();
    let sym_report = detect_symmetry_plane(&sym_mesh, &sym_bvh).unwrap();
    assert!(
        sym_report.symmetry_score > 0.99,
        "symmetric score={} rms={}",
        sym_report.symmetry_score,
        sym_report.symmetry_error_rms
    );
    let true_n = [r_tilt[0][0], r_tilt[1][0], r_tilt[2][0]];
    let align_dot = (sym_report.plane_normal[0] * true_n[0]
        + sym_report.plane_normal[1] * true_n[1]
        + sym_report.plane_normal[2] * true_n[2])
        .abs();
    assert!(align_dot > 0.999, "plane normal alignment dot={align_dot}");

    // 2) Chiral 3D asymmetric L-bracket (unequal arms + asymmetric Z lug breaks all 3 mirror planes).
    let mut asym_pos = Vec::new();
    let mut asym_tris = Vec::new();
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    push_box(
        &mut asym_pos,
        &mut asym_tris,
        [0.0, 0.0, 0.0],
        [3.2, 0.5, 0.5],
        identity,
        [0.0; 3],
    );
    push_box(
        &mut asym_pos,
        &mut asym_tris,
        [0.0, 0.5, 0.0],
        [0.5, 1.7, 0.5],
        identity,
        [0.0; 3],
    );
    push_box(
        &mut asym_pos,
        &mut asym_tris,
        [2.0, 0.0, 0.5],
        [3.2, 0.5, 1.8],
        identity,
        [0.0; 3],
    );
    let asym_mesh = mesh_from_indexed(&asym_pos, &asym_tris);
    let asym_bvh = Bvh::build(&asym_mesh, &BuildOptions::default()).unwrap();
    let asym_report = detect_symmetry_plane(&asym_mesh, &asym_bvh).unwrap();
    assert!(
        asym_report.symmetry_score < 0.85,
        "expected asymmetric bracket score < 0.85, got {} (rms={})",
        asym_report.symmetry_score,
        asym_report.symmetry_error_rms
    );
}

#[test]
fn multistage_and_symmetry_parallel_bit_identical() {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    // >= 4096 samples to exercise `PAR_MIN_DISTANCES` parallel paths when `parallel` is enabled.
    let target = build_asymmetric_blob(56, 64, [2.0, 1.4, 0.9], identity, [0.0; 3]);
    let target_bvh = Bvh::build(&target, &BuildOptions::default()).unwrap();
    let r_off = rot_axis_angle([0.4, -0.5, 1.0], 0.8);
    let source = build_asymmetric_blob(56, 64, [2.0, 1.4, 0.9], r_off, [3.0, -2.0, 1.5]);

    let opts = RegistrationOptions {
        coarse_method: CoarseAlignMethod::PcaCentroid,
        objective: IcpObjective::SymmetricNormal,
        max_iterations: 20,
        tolerance: 1e-6,
        sample_count: 5000,
        anderson_window: 3,
        ..RegistrationOptions::default()
    };

    let run_all = || {
        let reg = register_multistage(&source, &target, &target_bvh, None, &opts).unwrap();
        let sym = detect_symmetry_plane(&target, &target_bvh).unwrap();
        let sampled = sample_surface(&source, 5000).unwrap();
        assert!(sampled.len() >= 4096);
        (reg, sym)
    };

    par::set_force_sequential(true);
    let seq = run_all();
    par::set_force_sequential(false);
    if par::enabled() {
        for threads in [1, 4] {
            let par_res = par::with_threads(threads, run_all);
            assert_eq!(
                par_res, seq,
                "parallel bit-identity mismatch for {threads} threads"
            );
        }
        assert_eq!(run_all(), seq, "global pool bit-identity mismatch");
    }
}
