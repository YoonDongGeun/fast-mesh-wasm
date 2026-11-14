use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_fields::{
    DeviationOptions, FieldsError, GeodesicHeatSolver, GeodesicMethod, GeodesicOptions,
    OverhangOptions, QualityMetric, SignData, SignMode, geodesic_distances, mesh_deviation,
    overhangs, triangle_quality,
};
use fast_mesh_query::Rigid;
use fast_mesh_store::{Mesh, par};

fn build_bvh(mesh: &Mesh) -> Bvh {
    Bvh::build(mesh, &BuildOptions::default()).unwrap()
}

fn box_mesh(lo: [f32; 3], hi: [f32; 3]) -> Mesh {
    let mut pos = Vec::new();
    for i in 0..8 {
        pos.push(if i & 1 != 0 { hi[0] } else { lo[0] });
        pos.push(if i & 2 != 0 { hi[1] } else { lo[1] });
        pos.push(if i & 4 != 0 { hi[2] } else { lo[2] });
    }
    let idx: [u32; 36] = [
        0, 2, 3, 0, 3, 1, // z = lo (-Z)
        4, 5, 7, 4, 7, 6, // z = hi (+Z)
        0, 1, 5, 0, 5, 4, // y = lo (-Y)
        2, 6, 7, 2, 7, 3, // y = hi (+Y)
        0, 4, 6, 0, 6, 2, // x = lo (-X)
        1, 3, 7, 1, 7, 5, // x = hi (+X)
    ];
    Mesh::from_buffers(&pos, &idx).unwrap()
}

fn grid_mesh(nx: u32, ny: u32, scale: f32) -> Mesh {
    let mut pos = Vec::with_capacity(((nx + 1) * (ny + 1)) as usize * 3);
    for y in 0..=ny {
        for x in 0..=nx {
            pos.extend_from_slice(&[x as f32 * scale, y as f32 * scale, 0.0]);
        }
    }
    let mut idx = Vec::with_capacity((nx * ny * 6) as usize);
    let stride = nx + 1;
    for y in 0..ny {
        for x in 0..nx {
            let a = y * stride + x;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            idx.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
    Mesh::from_buffers(&pos, &idx).unwrap()
}

fn icosphere_unit(subdivisions: u32) -> Mesh {
    use std::collections::HashMap;
    let t = (1.0 + 5.0f64.sqrt()) * 0.5;
    let raw = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    let mut verts: Vec<[f64; 3]> = raw
        .into_iter()
        .map(|p| {
            let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            [p[0] / l, p[1] / l, p[2] / l]
        })
        .collect();
    let mut tris: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..subdivisions {
        let mut mid = HashMap::<(u32, u32), u32>::new();
        let mut next = Vec::with_capacity(tris.len() * 4);
        let mut get_mid = |u: u32, v: u32, vs: &mut Vec<[f64; 3]>| -> u32 {
            let key = if u < v { (u, v) } else { (v, u) };
            if let Some(&idx) = mid.get(&key) {
                return idx;
            }
            let a = vs[u as usize];
            let b = vs[v as usize];
            let m = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
            let l = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
            let idx = vs.len() as u32;
            vs.push([m[0] / l, m[1] / l, m[2] / l]);
            mid.insert(key, idx);
            idx
        };
        for [a, b, c] in tris {
            let ab = get_mid(a, b, &mut verts);
            let bc = get_mid(b, c, &mut verts);
            let ca = get_mid(c, a, &mut verts);
            next.push([a, ab, ca]);
            next.push([b, bc, ab]);
            next.push([c, ca, bc]);
            next.push([ab, bc, ca]);
        }
        tris = next;
    }
    let pos: Vec<f32> = verts
        .into_iter()
        .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    let idx: Vec<u32> = tris.into_iter().flatten().collect();
    Mesh::from_buffers(&pos, &idx).unwrap()
}

/// Planar U-shaped subdivided strip in XY: top arm (y in [2,3]), vertical left bridge (x in [0,1]),
/// bottom arm (y in [0,1]). Vertex indices of arm tips have tiny Euclidean gap (y=1 to y=2 at x=6)
/// while intrinsic path must detour left around x=0.5.
fn u_strip_mesh() -> (Mesh, u32, u32) {
    // Coarsely subdivided integer grid `nx = 12, ny = 6` on `[0..=12] x [0..=6]`, omitting cells in
    // the middle notch `x >= 2 && y in 2..4` so arms remain connected only at `x in 0..2`.
    let nx = 12u32;
    let ny = 6u32;
    let mut pos = Vec::new();
    for y in 0..=ny {
        for x in 0..=nx {
            pos.extend_from_slice(&[x as f32, y as f32, 0.0]);
        }
    }
    let mut idx = Vec::new();
    let stride = nx + 1;
    for y in 0..ny {
        for x in 0..nx {
            if x >= 2 && (2..4).contains(&y) {
                continue;
            }
            let a = y * stride + x;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            idx.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
    let m = Mesh::from_buffers(&pos, &idx).unwrap();
    let bottom_tip = nx; // (12, 0, 0)
    let top_tip = ny * stride + nx; // (12, 6, 0) - Euclidean dist = 6, intrinsic detour > 22
    (m, bottom_tip, top_tip)
}

#[test]
fn overhang_angle_conventions_and_support_area() {
    // 60° downward-facing single triangle: normal = (0, sin(60°), -cos(60°)) -> dot(+Z) = -0.5 -> 120° from up -> 30° overhang?
    // Wait: angle_from_up = acos(-cos(30°)) = 150° -> overhang = 60°!
    // Build triangle whose unit normal has z = -0.5 (120° from +Z => overhang = 30°) and one with z = -sqrt(3)/2 (150° from +Z => overhang = 60°).
    let s = 3.0f32.sqrt() * 0.5;
    let pos = [
        0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5,
        s, // upward-sloping (n = (0, -s, 0.5)) -> nz > 0 -> 0°
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // vertical wall (n = (1, 0, 0)) -> 0°
    ];
    let _ = pos;

    // Cube with sides length 2: bottom face area = 4 at 90°, top/side faces at 0°.
    let cube = box_mesh([-1.0; 3], [1.0; 3]);
    let mut angles = Vec::new();
    let rep = overhangs(&cube, &OverhangOptions::default(), &mut angles).unwrap();
    assert_eq!(rep.overhang_face_count, 2);
    assert!((rep.total_support_area - 4.0).abs() < 1e-10);
    assert!((rep.max_overhang_angle_deg - 90.0).abs() < 1e-4);

    // Custom -Y build direction: +Y face (area 4) now overhangs by 90°.
    let custom = OverhangOptions {
        build_direction: [0.0, -5.0, 0.0],
        critical_angle_deg: 89.0,
    };
    let rep_y = overhangs(&cube, &custom, &mut angles).unwrap();
    assert_eq!(rep_y.overhang_face_count, 2);
    assert!((rep_y.total_support_area - 4.0).abs() < 1e-10);
}

#[test]
fn deviation_unsigned_and_signed_measurements() {
    let mut target = box_mesh([0.0; 3], [2.0; 3]);
    let target_bvh = build_bvh(&target);
    let target_sign = SignData::build(&mut target, &target_bvh, SignMode::Pseudonormal).unwrap();

    let source_same = box_mesh([0.0; 3], [2.0; 3]);
    let mut out = Vec::new();
    let stats_zero = mesh_deviation(
        &source_same,
        &target,
        &target_bvh,
        None,
        Rigid::IDENTITY,
        &DeviationOptions::default(),
        &mut out,
    )
    .unwrap();
    assert_eq!(stats_zero.sample_count, 8);
    assert!(stats_zero.max < 1e-5);
    assert!(stats_zero.rms < 1e-5);

    // Source strictly smaller box [0.25, 1.75]^3 inside target [0, 2]^3 -> every vertex 0.25 inside (-0.25).
    let inner = box_mesh([0.25; 3], [1.75; 3]);
    let signed_opts = DeviationOptions {
        signed: true,
        max_distance: f32::INFINITY,
    };
    let stats_in = mesh_deviation(
        &inner,
        &target,
        &target_bvh,
        Some(&target_sign),
        Rigid::IDENTITY,
        &signed_opts,
        &mut out,
    )
    .unwrap();
    assert_eq!(stats_in.sample_count, 8);
    assert!((stats_in.min - (-0.25)).abs() < 1e-4);
    assert!((stats_in.max - (-0.25)).abs() < 1e-4);
    assert!((stats_in.mean - (-0.25)).abs() < 1e-4);
    assert!((stats_in.rms - 0.25).abs() < 1e-4);

    // Missing SignData in signed mode fails cleanly.
    assert!(matches!(
        mesh_deviation(
            &inner,
            &target,
            &target_bvh,
            None,
            Rigid::IDENTITY,
            &signed_opts,
            &mut out
        ),
        Err(FieldsError::InvalidInput(_))
    ));
}

#[test]
fn quality_metrics_distinguish_equilateral_and_sliver() {
    let eq_pos = [
        0.0f32,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.5,
        3.0f32.sqrt() * 0.5,
        0.0,
    ];
    let eq = Mesh::from_buffers(&eq_pos, &[0, 1, 2]).unwrap();
    let mut vq = Vec::new();
    let r_rad = triangle_quality(&eq, QualityMetric::RadiusRatio, 0.3, &mut vq).unwrap();
    assert!((r_rad.min - 1.0).abs() < 1e-5);
    assert_eq!(r_rad.sliver_triangle_count, 0);

    let r_ang = triangle_quality(&eq, QualityMetric::MinAngleDeg, 20.0, &mut vq).unwrap();
    assert!((r_ang.min - 60.0).abs() < 1e-4);
    assert_eq!(r_ang.sliver_triangle_count, 0);

    let r_asp = triangle_quality(&eq, QualityMetric::AspectRatio, 3.0, &mut vq).unwrap();
    assert!((r_asp.min - 1.0).abs() < 1e-5);
    assert_eq!(r_asp.sliver_triangle_count, 0);

    // Combine equilateral + needle sliver.
    let mixed_pos = [
        0.0f32,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.5,
        3.0f32.sqrt() * 0.5,
        0.0,
        10.0,
        0.01,
        0.0,
    ];
    let mixed = Mesh::from_buffers(&mixed_pos, &[0, 1, 2, 0, 1, 3]).unwrap();
    let mr = triangle_quality(&mixed, QualityMetric::RadiusRatio, 0.3, &mut vq).unwrap();
    assert_eq!(mr.sliver_triangle_count, 1);
    assert!(vq[3] < 0.05);
    let ma = triangle_quality(&mixed, QualityMetric::AspectRatio, 5.0, &mut vq).unwrap();
    assert_eq!(ma.sliver_triangle_count, 1);
    assert!(vq[3] > 100.0);
}

#[test]
fn heat_geodesic_sphere_accuracy_and_concave_barrier() {
    // Accuracy on unit icosphere(4) (2562 vertices): Crane heat method error relative to
    // exact great-circle distance is well under 3% (both mean relative error and away from
    // the immediate 1-ring source discretization).
    let sphere = icosphere_unit(4);
    let mut d_heat = Vec::new();
    geodesic_distances(&sphere, &[0], &GeodesicOptions::default(), &mut d_heat).unwrap();
    let p0 = sphere.positions()[0].map(f64::from);
    let mut max_rel_err = 0.0f64;
    let mut rel_sum = 0.0f64;
    let mut rel_count = 0usize;
    let mut max_abs_err = 0.0f64;
    for (v, &d) in d_heat.iter().enumerate() {
        let pv = sphere.positions()[v].map(f64::from);
        let dot_val = (p0[0] * pv[0] + p0[1] * pv[1] + p0[2] * pv[2]).clamp(-1.0, 1.0);
        let exact = dot_val.acos();
        let abs_err = (f64::from(d) - exact).abs();
        if abs_err > max_abs_err {
            max_abs_err = abs_err;
        }
        if exact > 0.25 {
            let rel = abs_err / exact;
            rel_sum += rel;
            rel_count += 1;
            if exact > 0.5 && rel > max_rel_err {
                max_rel_err = rel;
            }
        }
    }
    let mean_rel = rel_sum / (rel_count as f64);
    assert!(
        mean_rel < 0.02,
        "sphere mean relative error {mean_rel:.4} >= 2%"
    );
    assert!(
        max_rel_err < 0.03,
        "sphere max relative error (theta > 0.5) {max_rel_err:.4} >= 3%"
    );
    assert!(
        max_abs_err / std::f64::consts::PI < 0.025,
        "sphere max normalized error {:.4} >= 2.5%",
        max_abs_err / std::f64::consts::PI
    );

    // U-shape barrier: intrinsic distance >> Euclidean chord distance (6.0).
    let (u_mesh, tip_a, tip_b) = u_strip_mesh();
    for method in [GeodesicMethod::Dijkstra, GeodesicMethod::HeatMethod] {
        let opts = GeodesicOptions {
            method,
            ..Default::default()
        };
        let mut out = Vec::new();
        geodesic_distances(&u_mesh, &[tip_a], &opts, &mut out).unwrap();
        let d_b = out[tip_b as usize];
        assert!(
            d_b > 20.0,
            "{method:?}: expected detour > 20.0 across U-notch, got {d_b}"
        );
    }

    // Deleted vertex & triangle slots behave cleanly across inspect/geodesic fields.
    let mut edited = grid_mesh(2, 2, 1.0);
    // Corner vertex 8 `(2, 2)` is used only by triangles 6 and 7 `(x=1, y=1)` in a 2x2 grid.
    {
        let mut tx = edited.begin_edit();
        tx.remove_triangles(&[6, 7]).unwrap();
        tx.remove_vertices(&[8]).unwrap();
        tx.commit().unwrap();
    }
    let solver = GeodesicHeatSolver::build(&edited, 1.0).unwrap();
    let mut out = vec![123.0];
    assert!(matches!(
        solver.solve(&[8], None, &mut out),
        Err(FieldsError::InvalidInput(_))
    ));
    assert_eq!(out, vec![123.0]);

    solver.solve(&[0], None, &mut out).unwrap();
    assert_eq!(out.len(), edited.vertex_slots());
    assert!(out[8].is_infinite());

    let rep = overhangs(&edited, &OverhangOptions::default(), &mut out).unwrap();
    assert_eq!(out.len(), edited.vertex_slots());
    assert_eq!(out[8], 0.0);
    assert_eq!(rep.overhang_face_count, 0);

    let qrep = triangle_quality(&edited, QualityMetric::RadiusRatio, 0.2, &mut out).unwrap();
    assert_eq!(qrep.live_triangle_count, 6);
    assert_eq!(out[8], 0.0);
}

#[test]
fn large_mesh_parallel_bit_identity() {
    // 130 x 130 grid = 17,161 vertices >= PAR_MIN_ROWS (16,384).
    let mesh = grid_mesh(130, 130, 0.25);
    assert!(mesh.vertex_slots() >= 16_384);

    let run_heat = || -> Vec<u32> {
        let mut out = Vec::new();
        geodesic_distances(&mesh, &[0, 8_580], &GeodesicOptions::default(), &mut out).unwrap();
        out.into_iter().map(f32::to_bits).collect()
    };

    par::set_force_sequential(true);
    let seq_bits = run_heat();
    par::set_force_sequential(false);

    for threads in [1usize, 2, 8] {
        let par_bits = par::with_threads(threads, run_heat);
        assert_eq!(
            par_bits, seq_bits,
            "heat geodesic differs with {threads} threads"
        );
    }

    let bvh = build_bvh(&mesh);
    let run_dev = || -> Vec<u32> {
        let mut out = Vec::new();
        mesh_deviation(
            &mesh,
            &mesh,
            &bvh,
            None,
            Rigid {
                rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                translation: [0.0, 0.0, 0.1],
            },
            &DeviationOptions::default(),
            &mut out,
        )
        .unwrap();
        out.into_iter().map(f32::to_bits).collect()
    };

    par::set_force_sequential(true);
    let dev_seq = run_dev();
    par::set_force_sequential(false);
    for threads in [1usize, 2, 8] {
        let dev_par = par::with_threads(threads, run_dev);
        assert_eq!(dev_par, dev_seq, "deviation differs with {threads} threads");
    }
}
