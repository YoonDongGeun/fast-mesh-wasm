use std::f64::consts::{PI, TAU};

use fast_mesh_ops::inspect::{
    AnalyzeOptions, FitPrimitiveOptions, FittedPrimitive, PointCloud, PrimitiveKind,
    ReconstructMethod, SurfaceReconstructOptions, analyze, contours_to_csv, contours_to_json,
    contours_to_svg, datum_alignment_rigid, estimate_normals, fit_primitive_mesh,
    fit_primitive_points, measure, parse_pcd_ascii, parse_ply_points, parse_xyz,
    radius_outlier_removal, reconstruct_surface, slice_contour_stack, slice_contours,
    statistical_outlier_removal, voxel_downsample,
};
use fast_mesh_store::Mesh;

struct TestRng(u64);

impl TestRng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn signed(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}

fn unit3(v: [f64; 3]) -> [f64; 3] {
    let n = norm3(v);
    [v[0] / n, v[1] / n, v[2] / n]
}

fn ortho_frame(axis: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let ref_v = if axis[0].abs() < 0.8 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = unit3(cross3(axis, ref_v));
    let v = unit3(cross3(axis, u));
    (u, v)
}

fn build_uv_sphere(center: [f64; 3], radius: f64, rings: usize, sectors: usize) -> Mesh {
    let mut positions = Vec::new();
    let mut indices = Vec::new();

    positions.extend_from_slice(&[
        center[0] as f32,
        (center[1]) as f32,
        (center[2] + radius) as f32,
    ]);
    for r in 1..rings {
        let theta = PI * (r as f64) / (rings as f64);
        let st = theta.sin();
        let ct = theta.cos();
        for s in 0..sectors {
            let phi = TAU * (s as f64) / (sectors as f64);
            positions.push((center[0] + radius * st * phi.cos()) as f32);
            positions.push((center[1] + radius * st * phi.sin()) as f32);
            positions.push((center[2] + radius * ct) as f32);
        }
    }
    let south_idx = (1 + (rings - 1) * sectors) as u32;
    positions.extend_from_slice(&[
        center[0] as f32,
        center[1] as f32,
        (center[2] - radius) as f32,
    ]);

    for s in 0..sectors {
        let next = (s + 1) % sectors;
        indices.extend_from_slice(&[0, 1 + s as u32, 1 + next as u32]);
    }
    for r in 0..(rings - 2) {
        let row0 = 1 + r * sectors;
        let row1 = row0 + sectors;
        for s in 0..sectors {
            let next = (s + 1) % sectors;
            let a = (row0 + s) as u32;
            let b = (row0 + next) as u32;
            let c = (row1 + s) as u32;
            let d = (row1 + next) as u32;
            indices.extend_from_slice(&[a, c, b]);
            indices.extend_from_slice(&[b, c, d]);
        }
    }
    let last_row = 1 + (rings - 2) * sectors;
    for s in 0..sectors {
        let next = (s + 1) % sectors;
        indices.extend_from_slice(&[(last_row + s) as u32, south_idx, (last_row + next) as u32]);
    }

    Mesh::from_buffers(&positions, &indices).expect("valid uv sphere")
}

fn build_box_mesh(hx: f32, hy: f32, hz: f32) -> Mesh {
    let corners = [
        [-hx, -hy, -hz],
        [hx, -hy, -hz],
        [hx, hy, -hz],
        [-hx, hy, -hz],
        [-hx, -hy, hz],
        [hx, -hy, hz],
        [hx, hy, hz],
        [-hx, hy, hz],
    ];
    let tris: [[u32; 3]; 12] = [
        [0, 2, 1],
        [0, 3, 2], // -Z
        [4, 5, 6],
        [4, 6, 7], // +Z
        [0, 1, 5],
        [0, 5, 4], // -Y
        [2, 3, 7],
        [2, 7, 6], // +Y
        [0, 4, 7],
        [0, 7, 3], // -X
        [1, 2, 6],
        [1, 6, 5], // +X
    ];
    let pos: Vec<f32> = corners.into_iter().flatten().collect();
    let idx: Vec<u32> = tris.into_iter().flatten().collect();
    Mesh::from_buffers(&pos, &idx).expect("valid box mesh")
}

fn build_closed_cylinder_mesh(radius: f64, height: f64, sectors: usize) -> Mesh {
    let half_h = 0.5 * height;
    let mut pos = Vec::new();
    let mut idx = Vec::new();
    for s in 0..sectors {
        let phi = TAU * (s as f64) / (sectors as f64);
        let x = (radius * phi.cos()) as f32;
        let y = (radius * phi.sin()) as f32;
        pos.extend_from_slice(&[x, y, -half_h as f32]);
        pos.extend_from_slice(&[x, y, half_h as f32]);
    }
    let bot_center = (sectors * 2) as u32;
    let top_center = bot_center + 1;
    pos.extend_from_slice(&[0.0, 0.0, -half_h as f32]);
    pos.extend_from_slice(&[0.0, 0.0, half_h as f32]);

    for s in 0..sectors {
        let next = (s + 1) % sectors;
        let b0 = (2 * s) as u32;
        let t0 = b0 + 1;
        let b1 = (2 * next) as u32;
        let t1 = b1 + 1;
        idx.extend_from_slice(&[b0, b1, t0]);
        idx.extend_from_slice(&[t0, b1, t1]);
        idx.extend_from_slice(&[bot_center, b1, b0]);
        idx.extend_from_slice(&[top_center, t0, t1]);
    }

    Mesh::from_buffers(&pos, &idx).expect("valid cylinder mesh")
}

fn fibonacci_sphere_points(n: usize, radius: f64, center: [f64; 3]) -> Vec<[f32; 3]> {
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - (2.0 * (i as f64) + 1.0) / (n as f64);
            let r = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden * (i as f64);
            [
                (center[0] + radius * r * theta.cos()) as f32,
                (center[1] + radius * y) as f32,
                (center[2] + radius * r * theta.sin()) as f32,
            ]
        })
        .collect()
}

#[test]
fn test_fit_noisy_primitives_and_auto_classification() {
    let mut rng = TestRng(0xCAFE_BEEF_0234);
    let opts = FitPrimitiveOptions::default();

    // 1. Noisy Plane + 15% outliers
    let p_origin = [1.5, -0.8, 2.2];
    let p_normal = unit3([0.3, -0.45, 0.84]);
    let (pu, pv) = ortho_frame(p_normal);
    let mut plane_pts = Vec::new();
    for _ in 0..200 {
        let su = rng.signed() * 4.0;
        let sv = rng.signed() * 4.0;
        let jitter = rng.signed() * 0.01;
        let p = [
            p_origin[0] + su * pu[0] + sv * pv[0] + jitter * p_normal[0],
            p_origin[1] + su * pu[1] + sv * pv[1] + jitter * p_normal[1],
            p_origin[2] + su * pu[2] + sv * pv[2] + jitter * p_normal[2],
        ];
        plane_pts.push([p[0] as f32, p[1] as f32, p[2] as f32]);
    }
    for _ in 0..35 {
        let off = if rng.next_f64() > 0.5 { 1.8 } else { -1.8 } + rng.signed() * 0.6;
        let su = rng.signed() * 4.0;
        let sv = rng.signed() * 4.0;
        plane_pts.push([
            (p_origin[0] + su * pu[0] + sv * pv[0] + off * p_normal[0]) as f32,
            (p_origin[1] + su * pu[1] + sv * pv[1] + off * p_normal[1]) as f32,
            (p_origin[2] + su * pu[2] + sv * pv[2] + off * p_normal[2]) as f32,
        ]);
    }
    let plane_rep =
        fit_primitive_points(&plane_pts, None, PrimitiveKind::Plane, &opts).expect("plane fit");
    match plane_rep.primitive {
        FittedPrimitive::Plane { origin, normal } => {
            assert!(dot3(normal, p_normal).abs() > 0.995);
            assert!(dot3(sub3(origin, p_origin), p_normal).abs() < 0.02);
        }
        other => panic!("expected Plane, got {other:?}"),
    }
    assert!((plane_rep.inlier_ratio - 200.0 / 235.0).abs() < 0.06);
    assert_eq!(plane_rep.residuals.len(), plane_pts.len());
    let auto_plane =
        fit_primitive_points(&plane_pts, None, PrimitiveKind::Auto, &opts).expect("auto plane fit");
    assert!(matches!(
        auto_plane.primitive,
        FittedPrimitive::Plane { .. }
    ));

    // 2. Noisy Sphere + 15% outliers
    let s_center = [1.4, -2.1, 0.7];
    let s_radius = 2.8;
    let mut sphere_pts = Vec::new();
    for i in 0..240 {
        let base = fibonacci_sphere_points(240, 1.0, [0.0; 3])[i];
        let r = s_radius + rng.signed() * 0.015;
        sphere_pts.push([
            (s_center[0] + r * f64::from(base[0])) as f32,
            (s_center[1] + r * f64::from(base[1])) as f32,
            (s_center[2] + r * f64::from(base[2])) as f32,
        ]);
    }
    for _ in 0..42 {
        let dir = unit3([rng.signed(), rng.signed(), rng.signed()]);
        let r = s_radius + (if rng.next_f64() > 0.5 { 1.2 } else { -1.2 });
        sphere_pts.push([
            (s_center[0] + r * dir[0]) as f32,
            (s_center[1] + r * dir[1]) as f32,
            (s_center[2] + r * dir[2]) as f32,
        ]);
    }
    let sphere_rep =
        fit_primitive_points(&sphere_pts, None, PrimitiveKind::Auto, &opts).expect("auto sphere");
    match sphere_rep.primitive {
        FittedPrimitive::Sphere { center, radius } => {
            assert!(norm3(sub3(center, s_center)) < 0.04);
            assert!((radius - s_radius).abs() < 0.03);
        }
        other => panic!("expected Sphere, got {other:?}"),
    }

    // 3. Noisy Cylinder + 15% outliers + Datum alignment
    let c_origin = [0.8, -1.2, 0.6];
    let c_axis = unit3([0.35, 0.65, 0.67]);
    let (cu, cv) = ortho_frame(c_axis);
    let c_radius = 1.4;
    let c_height = 4.2;
    let mut cyl_pts = Vec::new();
    for i in 0..280 {
        let z = ((i as f64) / 279.0 - 0.5) * c_height;
        let phi = TAU * ((i as f64) * 0.6180339887);
        let r = c_radius + rng.signed() * 0.012;
        let radial = [
            r * (phi.cos() * cu[0] + phi.sin() * cv[0]),
            r * (phi.cos() * cu[1] + phi.sin() * cv[1]),
            r * (phi.cos() * cu[2] + phi.sin() * cv[2]),
        ];
        cyl_pts.push([
            (c_origin[0] + z * c_axis[0] + radial[0]) as f32,
            (c_origin[1] + z * c_axis[1] + radial[1]) as f32,
            (c_origin[2] + z * c_axis[2] + radial[2]) as f32,
        ]);
    }
    for _ in 0..49 {
        let z = rng.signed() * 3.0;
        let phi = rng.next_f64() * TAU;
        let r = c_radius + (if rng.next_f64() > 0.5 { 1.1 } else { -0.8 });
        cyl_pts.push([
            (c_origin[0] + z * c_axis[0] + r * (phi.cos() * cu[0] + phi.sin() * cv[0])) as f32,
            (c_origin[1] + z * c_axis[1] + r * (phi.cos() * cu[1] + phi.sin() * cv[1])) as f32,
            (c_origin[2] + z * c_axis[2] + r * (phi.cos() * cu[2] + phi.sin() * cv[2])) as f32,
        ]);
    }
    let cyl_rep =
        fit_primitive_points(&cyl_pts, None, PrimitiveKind::Auto, &opts).expect("auto cylinder");
    match &cyl_rep.primitive {
        FittedPrimitive::Cylinder {
            axis_origin,
            axis_dir,
            radius,
            height,
        } => {
            assert!(dot3(*axis_dir, c_axis).abs() > 0.995);
            assert!((*radius - c_radius).abs() < 0.03);
            assert!((*height - c_height).abs() < 0.15);
            let off = sub3(*axis_origin, c_origin);
            let perp = sub3(
                off,
                [
                    c_axis[0] * dot3(off, c_axis),
                    c_axis[1] * dot3(off, c_axis),
                    c_axis[2] * dot3(off, c_axis),
                ],
            );
            assert!(norm3(perp) < 0.04);
        }
        other => panic!("expected Cylinder, got {other:?}"),
    }

    let rigid = datum_alignment_rigid(&cyl_rep.primitive, [0.0, 0.0, 1.0], [1.0, 2.0, 3.0])
        .expect("datum alignment rigid");
    assert!(rigid.validate().is_ok());
    if let FittedPrimitive::Cylinder {
        axis_origin,
        axis_dir,
        ..
    } = cyl_rep.primitive
    {
        let r = &rigid.rotation;
        let t = rigid.translation;
        let mapped_origin = [
            dot3(r[0], axis_origin) + t[0],
            dot3(r[1], axis_origin) + t[1],
            dot3(r[2], axis_origin) + t[2],
        ];
        assert!(norm3(sub3(mapped_origin, [1.0, 2.0, 3.0])) < 1e-6);
        let mapped_dir = [
            dot3(r[0], axis_dir),
            dot3(r[1], axis_dir),
            dot3(r[2], axis_dir),
        ];
        assert!(dot3(mapped_dir, [0.0, 0.0, 1.0]) > 1.0 - 1e-6);
    }

    // 4. Noisy Cone + 15% outliers
    let k_apex = [0.4, -0.5, -1.2];
    let k_axis = unit3([0.2, -0.25, 0.947]);
    let (ku, kv) = ortho_frame(k_axis);
    let k_angle: f64 = 0.42;
    let k_height = 3.2;
    let mut cone_pts = Vec::new();
    for i in 0..280 {
        let z = 0.35 + ((i as f64) / 279.0) * (k_height - 0.35);
        let phi = TAU * ((i as f64) * 0.6180339887);
        let r = z * k_angle.tan() + rng.signed() * 0.008;
        cone_pts.push([
            (k_apex[0] + z * k_axis[0] + r * (phi.cos() * ku[0] + phi.sin() * kv[0])) as f32,
            (k_apex[1] + z * k_axis[1] + r * (phi.cos() * ku[1] + phi.sin() * kv[1])) as f32,
            (k_apex[2] + z * k_axis[2] + r * (phi.cos() * ku[2] + phi.sin() * kv[2])) as f32,
        ]);
    }
    for _ in 0..49 {
        let z = 0.4 + rng.next_f64() * 2.8;
        let phi = rng.next_f64() * TAU;
        let r = z * k_angle.tan() + (if rng.next_f64() > 0.5 { 0.8 } else { -0.6 });
        cone_pts.push([
            (k_apex[0] + z * k_axis[0] + r * (phi.cos() * ku[0] + phi.sin() * kv[0])) as f32,
            (k_apex[1] + z * k_axis[1] + r * (phi.cos() * ku[1] + phi.sin() * kv[1])) as f32,
            (k_apex[2] + z * k_axis[2] + r * (phi.cos() * ku[2] + phi.sin() * kv[2])) as f32,
        ]);
    }
    let cone_rep =
        fit_primitive_points(&cone_pts, None, PrimitiveKind::Auto, &opts).expect("auto cone fit");
    match cone_rep.primitive {
        FittedPrimitive::Cone {
            apex,
            axis_dir,
            half_angle_rad,
            ..
        } => {
            assert!(norm3(sub3(apex, k_apex)) < 0.08);
            assert!(dot3(axis_dir, k_axis) > 0.99);
            assert!((half_angle_rad - k_angle).abs() < 0.025);
        }
        other => panic!("expected Cone, got {other:?}"),
    }

    // 5. Mesh fitting API check
    let sphere_mesh = build_uv_sphere([1.0, 2.0, -3.0], 2.5, 16, 24);
    let mesh_rep = fit_primitive_mesh(&sphere_mesh, None, PrimitiveKind::Auto, &opts)
        .expect("fit sphere mesh");
    assert_eq!(mesh_rep.residuals.len(), sphere_mesh.vertex_slots());
    assert!(
        matches!(mesh_rep.primitive, FittedPrimitive::Sphere { radius, .. } if (radius - 2.5).abs() < 0.02)
    );
}

#[test]
fn test_section_contours_metrology_and_exports() {
    // 1. Sphere cross-section circle checks
    let sphere = build_uv_sphere([0.0, 0.0, 0.0], 2.0, 32, 64);
    let z_cut = 1.0_f32;
    let r_expected = (4.0_f64 - 1.0_f64).sqrt();
    let s_slice =
        slice_contours(&sphere, [0.0, 0.0, z_cut], [0.0, 0.0, 1.0]).expect("slice sphere contour");
    assert_eq!(s_slice.loops.len(), 1);
    let sl = &s_slice.loops[0];
    assert!(sl.closed);
    assert!((sl.perimeter - TAU * r_expected).abs() < 0.05);
    assert!((sl.enclosed_area - PI * r_expected * r_expected).abs() < 0.08);
    assert!((sl.min_caliper_diameter - 2.0 * r_expected).abs() < 0.04);
    assert!((sl.max_caliper_diameter - 2.0 * r_expected).abs() < 0.04);
    assert!(norm3(sub3(sl.centroid_3d, [0.0, 0.0, 1.0])) < 1e-3);

    // 2. Box cross-section rectangle (2.0 x 4.0)
    let box_mesh = build_box_mesh(1.0, 2.0, 3.0);
    let b_slice =
        slice_contours(&box_mesh, [0.0, 0.0, 0.5], [0.0, 0.0, 1.0]).expect("slice box contour");
    assert_eq!(b_slice.loops.len(), 1);
    let bl = &b_slice.loops[0];
    assert!(bl.closed);
    assert!((bl.perimeter - 12.0).abs() < 1e-4);
    assert!((bl.enclosed_area - 8.0).abs() < 1e-4);
    assert!((bl.min_caliper_diameter - 2.0).abs() < 1e-4);
    assert!((bl.max_caliper_diameter - (20.0_f64).sqrt()).abs() < 1e-4);

    // 3. Cylinder stack & SVG/JSON/CSV formatting
    let cyl = build_closed_cylinder_mesh(1.5, 4.0, 72);
    let stack = slice_contour_stack(&cyl, [0.0, 0.0, -1.5], [0.0, 0.0, 1.0], 1.0, 4)
        .expect("cylinder contour stack");
    assert_eq!(stack.len(), 4);
    for slc in &stack {
        assert_eq!(slc.loops.len(), 1);
        assert!((slc.total_perimeter - TAU * 1.5).abs() < 0.05);
        assert!((slc.total_area - PI * 1.5 * 1.5).abs() < 0.08);
    }

    let svg = contours_to_svg(&stack);
    assert!(svg.contains("<svg") && svg.contains("<polygon") && svg.contains("viewBox="));
    let json = contours_to_json(&stack);
    assert!(
        json.starts_with("{\"slices\":[")
            && json.contains("\"enclosed_area\":")
            && json.contains("\"min_caliper_diameter\":")
    );
    let csv = contours_to_csv(&stack);
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 5);
    assert!(lines[0].starts_with("slice_index,loop_index,closed"));

    // 4. Empty mesh returns 0 loops cleanly
    let empty = Mesh::from_buffers(&[], &[]).expect("empty mesh");
    let empty_slice = slice_contours(&empty, [0.0; 3], [0.0, 0.0, 1.0]).expect("empty slice");
    assert!(empty_slice.loops.is_empty());
    assert_eq!(empty_slice.total_area, 0.0);
}

#[test]
fn test_point_cloud_pipeline_and_surface_reconstruction() {
    // 1. Parsers (XYZ, PCD ASCII, PLY binary little-endian & ASCII)
    let xyz_str = "0.0 0.0 1.0 0.0 0.0 1.0\n1.0 0.0 0.0 1.0 0.0 0.0\n0.0 1.0 0.0 0.0 1.0 0.0\n";
    let xyz_cloud = parse_xyz(xyz_str).expect("parse xyz");
    assert_eq!(xyz_cloud.positions.len(), 3);
    assert!(xyz_cloud.normals.is_some());

    let pcd_str = "\
# .PCD v0.7
VERSION 0.7
FIELDS x y z r g b
SIZE 4 4 4 1 1 1
TYPE F F F U U U
COUNT 1 1 1 1 1 1
WIDTH 2
HEIGHT 1
POINTS 2
DATA ascii
0.0 0.0 0.0 255 0 0
1.0 2.0 3.0 0 255 128
";
    let pcd_cloud = parse_pcd_ascii(pcd_str).expect("parse pcd");
    assert_eq!(pcd_cloud.len(), 2);
    assert_eq!(pcd_cloud.colors.as_ref().unwrap()[0], [1.0, 0.0, 0.0]);

    let mut bin_ply = Vec::from(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            .as_slice(),
    );
    for (p, c) in [
        ([1.0_f32, 2.0, 3.0], [255_u8, 128, 0]),
        ([-1.0_f32, 0.5, 2.0], [0_u8, 0, 255]),
    ] {
        for v in p {
            bin_ply.extend_from_slice(&v.to_le_bytes());
        }
        bin_ply.extend_from_slice(&c);
    }
    let ply_cloud = parse_ply_points(&bin_ply).expect("parse binary ply");
    assert_eq!(ply_cloud.positions[1], [-1.0, 0.5, 2.0]);
    assert!(ply_cloud.colors.is_some());

    // 2. Dense sphere cloud with jitter + statistical outliers -> SOR -> Voxel -> Normals -> Reconstruct
    let mut rng = TestRng(0xDEAD_BEEF_0099);
    let r_sphere = 2.0;
    let base_pts = fibonacci_sphere_points(700, r_sphere, [0.0; 3]);
    let mut positions: Vec<[f32; 3]> = base_pts
        .iter()
        .map(|&p| {
            [
                p[0] + (rng.signed() * 0.004) as f32,
                p[1] + (rng.signed() * 0.004) as f32,
                p[2] + (rng.signed() * 0.004) as f32,
            ]
        })
        .collect();
    for _ in 0..45 {
        let dir = unit3([rng.signed(), rng.signed(), rng.signed()]);
        let dist = 4.5 + rng.next_f64() * 3.0;
        positions.push([
            (dir[0] * dist) as f32,
            (dir[1] * dist) as f32,
            (dir[2] * dist) as f32,
        ]);
    }

    let raw_cloud = PointCloud::from_positions(positions);
    let (sor_cloud, sor_rep) =
        statistical_outlier_removal(&raw_cloud, 10, 1.5).expect("sor filter");
    assert!(sor_rep.removed_count >= 40);
    assert!(sor_cloud.len() >= 660);
    for p in &sor_cloud.positions {
        let r = norm3([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
        assert!(r < 2.2, "outlier leaked past SOR: radius {r}");
    }

    // Radius outlier removal keeps dense inliers too
    let (ror_cloud, ror_rep) = radius_outlier_removal(&raw_cloud, 0.6, 5).expect("ror filter");
    assert!(ror_rep.removed_count >= 45);
    assert!(ror_cloud.len() >= 680);

    // Voxel downsample preserves spherical shell cleanly
    let mut down = voxel_downsample(&sor_cloud, 0.22).expect("voxel downsample");
    assert!(down.len() < sor_cloud.len());
    assert!(down.len() >= 250);

    // Estimate consistent outward normals without camera hint
    estimate_normals(&mut down, 12, None).expect("estimate normals");
    let normals = down.normals.as_ref().expect("normals populated");
    for (&p, &n) in down.positions.iter().zip(normals.iter()) {
        let pn = unit3([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
        let nv = [f64::from(n[0]), f64::from(n[1]), f64::from(n[2])];
        assert!(
            dot3(pn, nv) > 0.92,
            "normal not oriented outward: dot={}",
            dot3(pn, nv)
        );
    }

    // Camera-hint normal estimation on a flat horizontal slab
    let mut sheet_pts = Vec::new();
    for iy in 0..8 {
        for ix in 0..8 {
            sheet_pts.push([ix as f32 * 0.2, iy as f32 * 0.2, 0.0]);
        }
    }
    let mut sheet = PointCloud::from_positions(sheet_pts);
    estimate_normals(&mut sheet, 8, Some([0.7, 0.7, -5.0])).expect("camera hint normals");
    for &n in sheet.normals.as_ref().unwrap() {
        assert!(n[2] < -0.95);
    }

    // 3. Reconstruct via ImplicitMlsMarchingCubes
    let mc_mesh = reconstruct_surface(
        &down,
        &SurfaceReconstructOptions {
            method: ReconstructMethod::ImplicitMlsMarchingCubes,
            voxel_size: 0.0,
            ball_radius: 0.0,
            k_neighbors: 12,
        },
    )
    .expect("mls marching cubes");
    assert!(mc_mesh.triangle_count() > 200);
    let mc_analysis = analyze(&mc_mesh, &AnalyzeOptions::default()).expect("analyze mc");
    assert!(mc_analysis.loops.is_empty());
    assert_eq!(mc_analysis.non_manifold_edges, 0);
    assert!(mc_analysis.non_manifold_vertices.is_empty());
    assert_eq!(mc_analysis.orientation_conflicts, 0);
    let mc_meas = measure(&mc_mesh).expect("measure mc");
    let expected_vol = (4.0 / 3.0) * PI * r_sphere.powi(3);
    assert!(mc_meas.signed_volume > 0.0);
    let mc_vol = mc_meas.volume.expect("watertight mc volume");
    assert!((mc_vol - expected_vol).abs() / expected_vol < 0.08);

    // 4. Reconstruct via BallPivoting on uniform downsampled sphere
    let bpa_mesh = reconstruct_surface(
        &down,
        &SurfaceReconstructOptions {
            method: ReconstructMethod::BallPivoting,
            voxel_size: 0.0,
            ball_radius: 0.0,
            k_neighbors: 12,
        },
    )
    .expect("ball pivoting reconstruction");
    assert!(bpa_mesh.triangle_count() > 300);
    let bpa_analysis = analyze(&bpa_mesh, &AnalyzeOptions::default()).expect("analyze bpa");
    assert_eq!(bpa_analysis.degenerate, 0);
    assert_eq!(bpa_analysis.duplicate, 0);
    assert_eq!(bpa_analysis.non_manifold_edges, 0);
    assert_eq!(bpa_analysis.orientation_conflicts, 0);

    // Verify BPA triangles lie on sphere and wind outward.
    for ti in 0..bpa_mesh.triangle_slots() {
        let Some([a, b, c]) = bpa_mesh.triangle(ti as u32) else {
            continue;
        };
        let pa = bpa_mesh.world_position(a).unwrap();
        let pb = bpa_mesh.world_position(b).unwrap();
        let pc = bpa_mesh.world_position(c).unwrap();
        let tn = unit3(cross3(sub3(pb, pa), sub3(pc, pa)));
        let center = unit3([
            (pa[0] + pb[0] + pc[0]) / 3.0,
            (pa[1] + pb[1] + pc[1]) / 3.0,
            (pa[2] + pb[2] + pc[2]) / 3.0,
        ]);
        assert!(dot3(tn, center) > 0.75);
    }
}

#[test]
fn test_edge_cases_subset_fit_antiparallel_rigid_open_and_nested_contours() {
    // 1. ASCII PLY parser with normals + colors
    let ascii_ply = "\
ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
end_header
1.0 2.0 3.0 0.0 0.0 1.0 255 0 128
4.0 5.0 6.0 0.0 1.0 0.0 0 255 0
";
    let cloud = parse_ply_points(ascii_ply.as_bytes()).expect("parse ascii ply");
    assert_eq!(cloud.len(), 2);
    assert_eq!(cloud.normals.as_ref().unwrap()[1], [0.0, 1.0, 0.0]);
    assert_eq!(cloud.colors.as_ref().unwrap()[1], [0.0, 1.0, 0.0]);

    // 2. Partial triangle-selection fit leaves unselected vertex slots at 0.0
    let box_m = build_box_mesh(2.0, 2.0, 2.0);
    // Triangles 2 and 3 form the +Z cap plane (z = +2.0, corners 4,5,6,7).
    let top_fit = fit_primitive_mesh(
        &box_m,
        Some(&[2, 3]),
        PrimitiveKind::Plane,
        &FitPrimitiveOptions::default(),
    )
    .expect("subset plane fit");
    assert_eq!(top_fit.residuals.len(), box_m.vertex_slots());
    for slot in 0..box_m.vertex_slots() {
        assert!(top_fit.residuals[slot].abs() < 1e-5);
    }
    match top_fit.primitive {
        FittedPrimitive::Plane { origin, normal } => {
            assert!((origin[2] - 2.0).abs() < 1e-5);
            assert!(normal[2].abs() > 0.999);
        }
        other => panic!("expected Plane, got {other:?}"),
    }

    // 3. Antiparallel datum alignment rotation (180-deg flip)
    let anti = datum_alignment_rigid(
        &FittedPrimitive::Plane {
            origin: [1.0, 2.0, 3.0],
            normal: [0.0, 0.0, -1.0],
        },
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    )
    .expect("antiparallel rigid");
    assert!(anti.validate().is_ok());
    let rot_z = [
        dot3(anti.rotation[0], [0.0, 0.0, -1.0]),
        dot3(anti.rotation[1], [0.0, 0.0, -1.0]),
        dot3(anti.rotation[2], [0.0, 0.0, -1.0]),
    ];
    assert!(dot3(rot_z, [0.0, 0.0, 1.0]) > 1.0 - 1e-6);

    // 4. Open mesh slice produces open chain with enclosed_area == 0.0 and <polyline> SVG
    let open_wall_pos = [
        -1.0_f32, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
    ];
    let open_wall_idx = [0_u32, 1, 2, 0, 2, 3];
    let open_mesh = Mesh::from_buffers(&open_wall_pos, &open_wall_idx).expect("open quad mesh");
    let open_slice =
        slice_contours(&open_mesh, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).expect("slice open mesh");
    assert_eq!(open_slice.loops.len(), 1);
    assert!(!open_slice.loops[0].closed);
    assert_eq!(open_slice.loops[0].enclosed_area, 0.0);
    assert!((open_slice.loops[0].perimeter - 2.0).abs() < 1e-4);
    let open_svg = contours_to_svg(&[open_slice]);
    assert!(open_svg.contains("<polyline"));

    // 5. Nested two-shell mesh (annular cross section) subtracts inner hole area cleanly
    let outer = build_uv_sphere([0.0, 0.0, 0.0], 2.0, 24, 48);
    let inner = build_uv_sphere([0.0, 0.0, 0.0], 1.0, 24, 48);
    let mut ring_pos: Vec<f32> = Vec::new();
    let mut ring_idx: Vec<u32> = Vec::new();
    for &p in outer.positions() {
        ring_pos.extend_from_slice(&p);
    }
    for &t in outer.triangles() {
        ring_idx.extend_from_slice(&t);
    }
    let v_off = outer.vertex_slots() as u32;
    for &p in inner.positions() {
        ring_pos.extend_from_slice(&p);
    }
    for &[a, b, c] in inner.triangles() {
        ring_idx.extend_from_slice(&[a + v_off, c + v_off, b + v_off]);
    }
    let hollow = Mesh::from_buffers(&ring_pos, &ring_idx).expect("hollow sphere shell");
    let hollow_slice =
        slice_contours(&hollow, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).expect("slice hollow sphere");
    assert_eq!(hollow_slice.loops.len(), 2);
    let expected_annulus = PI * (2.0_f64 * 2.0 - 1.0_f64 * 1.0);
    assert!(
        (hollow_slice.total_area - expected_annulus).abs() < 0.15,
        "got total_area={}, expected={}, loops={:?}",
        hollow_slice.total_area,
        expected_annulus,
        hollow_slice
            .loops
            .iter()
            .map(|l| (l.closed, l.points_2d.len(), l.enclosed_area, l.centroid_2d))
            .collect::<Vec<_>>()
    );
}
