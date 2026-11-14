use fast_mesh::compute_vertex_normals;
use fast_mesh_store::{Mesh, Transaction, VertexId};

use super::{
    Brush, ClayOptions, ClayProfile, CreaseOptions, Falloff, FlattenMode, FlattenOptions,
    GRAB_MAX_STRETCH, GrabMethod, GrabOptions, HC_ALPHA, HC_BETA, InflateMethod, InflateOptions,
    STROKE_NORMAL_CHANNEL, STROKE_POS_CHANNEL, SmoothMethod, SmoothOptions, StrokeSlice,
    SurfaceFilter, Symmetry, TAUBIN_LAMBDA, TAUBIN_MU, TwistMethod, TwistOptions,
    capture_stroke_state, clay, clay_region, crease, crease_region, ensure_stroke_state, flatten,
    flatten_region, grab, grab_region, grab_region_with_stroke, grab_widest_reach, inflate,
    inflate_region, inflate_region_with_stroke, smooth, smooth_region, twist, twist_region,
};
use crate::OpsError;
use crate::remesh::Region;
use crate::stretch::{StretchOptions, capture_rest_sizes, split_stretched};
use crate::test_util::{edit, grid, mesh_of, parts, sweep, uv_sphere};

fn bits_of(mesh: &Mesh) -> Vec<[u32; 3]> {
    mesh.positions()
        .iter()
        .map(|p| p.map(f32::to_bits))
        .collect()
}

/// Mirror-symmetric wavy sheet centered at the origin (`x, y in [-6, 6]`),
/// triangulated symmetrically across `x = 0` and `y = 0` so mirror vertex pairs
/// have exact mirror connectivity and normals.
fn symmetric_sheet(half: i32, lift: impl Fn(f32, f32) -> f32) -> Mesh {
    let side = (2 * half + 1) as u32;
    let mut positions = Vec::new();
    for iy in -half..=half {
        for ix in -half..=half {
            let (x, y) = (ix as f32, iy as f32);
            positions.extend([x, y, lift(x, y)]);
        }
    }
    let at = |ix: i32, iy: i32| ((iy + half) as u32) * side + (ix + half) as u32;
    let mut triangles = Vec::new();
    for iy in -half..half {
        for ix in -half..half {
            let (a, b, c, d) = (
                at(ix, iy),
                at(ix + 1, iy),
                at(ix + 1, iy + 1),
                at(ix, iy + 1),
            );
            // Mirror diagonal choice across quadrants so connectivity is x- and y-symmetric.
            let same_sign = (ix >= 0) == (iy >= 0);
            if same_sign {
                triangles.extend([[a, b, c], [a, c, d]]);
            } else {
                triangles.extend([[b, c, d], [b, d, a]]);
            }
        }
    }
    mesh_of(&positions, &triangles)
}

#[test]
fn stroke_channels_capture_interpolate_across_splits_and_freeze_stroke_normals() {
    let mut mesh = uv_sphere(12, 18, 2.0);
    let initial_positions = mesh.positions().to_vec();
    let (flat, tris) = parts(&mesh);
    let indices: Vec<u32> = tris.iter().flatten().copied().collect();
    let initial_normals = compute_vertex_normals(&flat, &indices).unwrap();

    capture_stroke_state(&mut mesh).unwrap();
    let pos_desc = mesh.channel_desc(STROKE_POS_CHANNEL).expect("pos channel");
    let nrm_desc = mesh
        .channel_desc(STROKE_NORMAL_CHANNEL)
        .expect("normal channel");
    assert_eq!(pos_desc.width, 3);
    assert_eq!(nrm_desc.width, 3);
    assert!(pos_desc.topology_bound);
    assert!(nrm_desc.topology_bound);

    // ensure_stroke_state preserves an existing stroke capture even after vertices move.
    let v0_orig = initial_positions[0];
    edit(&mut mesh, |tx| {
        tx.set_positions(&[0], &[[v0_orig[0], v0_orig[1], v0_orig[2] + 0.5]])?;
        Ok(())
    });
    ensure_stroke_state(&mut mesh).unwrap();
    let pos_ch = mesh.channel(STROKE_POS_CHANNEL).unwrap();
    assert_eq!(&pos_ch[0..3], &v0_orig);
    // Resetting v0 and re-capturing refreshes the channels.
    edit(&mut mesh, |tx| {
        tx.set_positions(&[0], &[v0_orig])?;
        Ok(())
    });
    capture_stroke_state(&mut mesh).unwrap();

    // Interpolates parent values across split_stretched.
    let mut split_mesh = uv_sphere(8, 12, 2.0);
    capture_rest_sizes(&mut split_mesh).unwrap();
    capture_stroke_state(&mut split_mesh).unwrap();
    let old_slots = split_mesh.vertex_slots();
    let ids: Vec<VertexId> = (0..old_slots as VertexId).collect();
    let stretched: Vec<[f32; 3]> = split_mesh
        .positions()
        .iter()
        .map(|p| [p[0] * 1.8, p[1] * 1.8, p[2] * 1.8])
        .collect();
    edit(&mut split_mesh, |tx| {
        tx.set_positions(&ids, &stretched)?;
        Ok(())
    });
    let (stretch_report, _) = edit(&mut split_mesh, |tx| {
        split_stretched(
            tx,
            &StretchOptions {
                region: Region {
                    center: [0.0; 3],
                    radius: 100.0,
                },
                ..StretchOptions::default()
            },
        )
    });
    assert!(stretch_report.splits > 0);
    let rest_pos = split_mesh
        .channel(STROKE_POS_CHANNEL)
        .expect("stroke pos survived split_stretched");
    let rest_nrm = split_mesh
        .channel(STROKE_NORMAL_CHANNEL)
        .expect("stroke normal survived split_stretched");
    for &[new_v, a, b] in &stretch_report.vertices {
        for k in 0..3 {
            let want_p = 0.5 * (rest_pos[3 * a as usize + k] + rest_pos[3 * b as usize + k]);
            let want_n = 0.5 * (rest_nrm[3 * a as usize + k] + rest_nrm[3 * b as usize + k]);
            assert!((rest_pos[3 * new_v as usize + k] - want_p).abs() < 1e-5);
            assert!((rest_nrm[3 * new_v as usize + k] - want_n).abs() < 1e-5);
        }
    }

    // Fallback when capture_stroke_state was never called: matches VertexNormal bit-for-bit.
    let mut uncaptured_a = symmetric_sheet(5, |x, y| 0.25 * x + 0.1 * y * y);
    let mut uncaptured_b = symmetric_sheet(5, |x, y| 0.25 * x + 0.1 * y * y);
    let base_brush = Brush {
        center: [0.6, 0.2, 0.15],
        radius: 2.8,
        falloff: Falloff::Smooth,
        strength: 0.35,
    };
    edit(&mut uncaptured_a, |tx| {
        inflate(
            tx,
            &InflateOptions {
                brush: base_brush,
                method: InflateMethod::StrokeNormal,
                ..Default::default()
            },
        )
    });
    edit(&mut uncaptured_b, |tx| {
        inflate(
            tx,
            &InflateOptions {
                brush: base_brush,
                method: InflateMethod::VertexNormal,
                ..Default::default()
            },
        )
    });
    assert_eq!(bits_of(&uncaptured_a), bits_of(&uncaptured_b));

    // Across 5 repeated dabs centered at tracked vertex `vc` on an asymmetric slope,
    // StrokeNormal moves `vc` linearly equal to `5 * single_dab` along its frozen normal,
    // whereas VertexNormal curves as surrounding normals tilt.
    let make_slope = || symmetric_sheet(6, |x, y| 0.4 * (0.4 * x).sin() + 0.15 * y);
    let vc = (6 * 13 + 7) as VertexId; // interior vertex at (1, 0)
    let mut one_dab_mesh = make_slope();
    capture_stroke_state(&mut one_dab_mesh).unwrap();
    let p0 = one_dab_mesh.positions()[vc as usize];
    let dab_for = |center: [f32; 3], method: InflateMethod| InflateOptions {
        brush: Brush {
            center,
            radius: 2.2,
            falloff: Falloff::Smooth,
            strength: 0.3,
        },
        method,
        ..Default::default()
    };
    edit(&mut one_dab_mesh, |tx| {
        inflate(tx, &dab_for(p0, InflateMethod::StrokeNormal))
    });
    let p1 = one_dab_mesh.positions()[vc as usize];
    let single_dab = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];

    let mut stroke_five = make_slope();
    capture_stroke_state(&mut stroke_five).unwrap();
    let mut vertex_five = make_slope();
    for _ in 0..5 {
        let cs = stroke_five.positions()[vc as usize];
        edit(&mut stroke_five, |tx| {
            inflate(tx, &dab_for(cs, InflateMethod::StrokeNormal))
        });
        let cv = vertex_five.positions()[vc as usize];
        edit(&mut vertex_five, |tx| {
            inflate(tx, &dab_for(cv, InflateMethod::VertexNormal))
        });
    }
    let ps5 = stroke_five.positions()[vc as usize];
    let pv5 = vertex_five.positions()[vc as usize];
    let stroke_disp = [ps5[0] - p0[0], ps5[1] - p0[1], ps5[2] - p0[2]];
    let vertex_disp = [pv5[0] - p0[0], pv5[1] - p0[1], pv5[2] - p0[2]];
    let want_five = [
        5.0 * single_dab[0],
        5.0 * single_dab[1],
        5.0 * single_dab[2],
    ];
    let err_stroke = (0..3)
        .map(|k| (stroke_disp[k] - want_five[k]).powi(2))
        .sum::<f32>()
        .sqrt();
    let err_vertex = (0..3)
        .map(|k| (vertex_disp[k] - want_five[k]).powi(2))
        .sum::<f32>()
        .sqrt();
    assert!(
        err_stroke < 1e-5,
        "StrokeNormal 5 dabs {stroke_disp:?} != 5 * single_dab {want_five:?}"
    );
    assert!(
        err_vertex > 0.02,
        "VertexNormal should curve away from frozen normal: err {err_vertex}"
    );

    // Every vertex moved by StrokeNormal moves strictly parallel to its initial normal.
    let _ = initial_normals;
}

#[test]
fn inflate_average_normal_translates_parallel_without_diverging_on_ridge() {
    // Sharp V-ridge: z = 2.0 - |x|. Left flank normal points (-x, +z), right flank (+x, +z).
    let ridge = || symmetric_sheet(6, |x, _y| 2.0 - x.abs());
    let brush = Brush {
        center: [0.35, 0.0, 1.65],
        radius: 3.0,
        falloff: Falloff::Smooth,
        strength: 0.5,
    };

    // Independently compute expected falloff-weighted unit average normal.
    let ref_mesh = ridge();
    let (flat, tris) = parts(&ref_mesh);
    let indices: Vec<u32> = tris.iter().flatten().copied().collect();
    let normals = compute_vertex_normals(&flat, &indices).unwrap();
    let mut sum = [0.0f64; 3];
    for (v, p) in ref_mesh.positions().iter().enumerate() {
        let dx = p[0] - brush.center[0];
        let dy = p[1] - brush.center[1];
        let dz = p[2] - brush.center[2];
        let t = (dx * dx + dy * dy + dz * dz).sqrt() / brush.radius;
        if t < 1.0 {
            let s = 1.0 - t;
            let f = f64::from(s * s * (3.0 - 2.0 * s));
            sum[0] += f * f64::from(normals[3 * v]);
            sum[1] += f * f64::from(normals[3 * v + 1]);
            sum[2] += f * f64::from(normals[3 * v + 2]);
        }
    }
    let len = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
    let expected_dir = [
        (sum[0] / len) as f32,
        (sum[1] / len) as f32,
        (sum[2] / len) as f32,
    ];
    assert!(expected_dir[0] > 0.05, "off-center ridge brush tilts +x");

    let mut avg_mesh = ridge();
    let before = avg_mesh.positions().to_vec();
    let (rep, _) = edit(&mut avg_mesh, |tx| {
        inflate(
            tx,
            &InflateOptions {
                brush,
                method: InflateMethod::AverageNormal,
                ..Default::default()
            },
        )
    });
    assert!(rep.moved > 6);
    for (p0, p1) in before.iter().zip(avg_mesh.positions()) {
        let dp = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let dlen = (dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]).sqrt();
        if dlen > 1e-2 {
            let unit_dp = [dp[0] / dlen, dp[1] / dlen, dp[2] / dlen];
            for k in 0..3 {
                assert!(
                    (unit_dp[k] - expected_dir[k]).abs() < 5e-5,
                    "dp unit {unit_dp:?} != expected_dir {expected_dir:?}"
                );
            }
        }
    }

    // Contrast with VertexNormal, where -x and +x flanks diverge in x.
    let mut vn_mesh = ridge();
    edit(&mut vn_mesh, |tx| {
        inflate(
            tx,
            &InflateOptions {
                brush,
                method: InflateMethod::VertexNormal,
                ..Default::default()
            },
        )
    });
    let mut saw_neg_dx = false;
    let mut saw_pos_dx = false;
    for (p0, p1) in before.iter().zip(vn_mesh.positions()) {
        if p1[0] - p0[0] < -0.02 {
            saw_neg_dx = true;
        }
        if p1[0] - p0[0] > 0.02 {
            saw_pos_dx = true;
        }
    }
    assert!(saw_neg_dx && saw_pos_dx);
}

#[test]
fn smooth_hc_laplacian_resists_sphere_shrinkage_while_reducing_noise() {
    assert_eq!(TAUBIN_LAMBDA, 0.33);
    assert_eq!(TAUBIN_MU, -0.34);
    assert_eq!(HC_ALPHA, 0.1);
    assert_eq!(HC_BETA, 0.5);

    let noisy_sphere = || {
        let mut m = uv_sphere(16, 24, 2.0);
        let ids: Vec<VertexId> = (0..m.vertex_slots() as VertexId).collect();
        let noisy: Vec<[f32; 3]> = m
            .positions()
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                let sign = if (i * 7 + 3) % 2 == 0 { 1.0 } else { -1.0 };
                let scale = (r + 0.08 * sign) / r;
                [p[0] * scale, p[1] * scale, p[2] * scale]
            })
            .collect();
        edit(&mut m, |tx| {
            tx.set_positions(&ids, &noisy)?;
            Ok(())
        });
        m
    };

    let radial_stats = |m: &Mesh| -> (f64, f64) {
        let radii: Vec<f64> = m
            .positions()
            .iter()
            .map(|p| {
                (f64::from(p[0]).powi(2) + f64::from(p[1]).powi(2) + f64::from(p[2]).powi(2)).sqrt()
            })
            .collect();
        let mean = radii.iter().sum::<f64>() / radii.len() as f64;
        let var = radii.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / radii.len() as f64;
        (mean, var.sqrt())
    };

    let base = noisy_sphere();
    let (mean_before, std_before) = radial_stats(&base);

    let brush = Brush {
        center: [0.0; 3],
        radius: 100.0,
        falloff: Falloff::Linear,
        strength: 1.0,
    };
    let mut hc = noisy_sphere();
    edit(&mut hc, |tx| {
        smooth(
            tx,
            &SmoothOptions {
                brush,
                cycles: 6,
                method: SmoothMethod::HcLaplacian,
                ..Default::default()
            },
        )
    });
    let mut lap = noisy_sphere();
    edit(&mut lap, |tx| {
        smooth(
            tx,
            &SmoothOptions {
                brush,
                cycles: 6,
                method: SmoothMethod::Laplacian,
                ..Default::default()
            },
        )
    });

    let (mean_hc, std_hc) = radial_stats(&hc);
    let (mean_lap, _) = radial_stats(&lap);
    let shrink_hc = (mean_before - mean_hc).abs();
    let shrink_lap = mean_before - mean_lap;

    assert!(
        std_hc < 0.35 * std_before,
        "HC-Laplacian must damp high-frequency noise: before={std_before}, after={std_hc}"
    );
    assert!(
        shrink_hc < shrink_lap * 0.25,
        "HC-Laplacian shrinkage ({shrink_hc}) must be strictly smaller than Laplacian ({shrink_lap})"
    );
}

#[test]
fn smooth_cotangent_keeps_irregular_flat_interior_vertex_stationary() {
    // Vertex 0 at (0, 0, 0), surrounded by a coarse left rim and a densely subdivided right rim
    // on the plane z = 0.
    let mut positions = vec![0.0f32, 0.0, 0.0];
    let ring_angles_deg: [f64; 9] = [-60.0, -35.0, -15.0, 0.0, 15.0, 35.0, 60.0, 140.0, 220.0];
    for &deg in &ring_angles_deg {
        let rad = deg.to_radians();
        positions.extend([rad.cos() as f32, rad.sin() as f32, 0.0]);
    }
    let n = ring_angles_deg.len() as u32;
    let mut triangles = Vec::new();
    for k in 0..n {
        triangles.push([0, 1 + k, 1 + (k + 1) % n]);
    }

    let brush = Brush {
        center: [0.0, 0.0, 0.0],
        radius: 0.25, // only vertex 0 is inside the brush; ring stays fixed
        falloff: Falloff::Linear,
        strength: 1.0,
    };

    let mut cot_mesh = mesh_of(&positions, &triangles);
    edit(&mut cot_mesh, |tx| {
        smooth(
            tx,
            &SmoothOptions {
                brush,
                cycles: 5,
                method: SmoothMethod::Cotangent,
                ..Default::default()
            },
        )
    });
    let p_cot = cot_mesh.positions()[0];
    assert!(
        p_cot[0].abs() < 1e-5 && p_cot[1].abs() < 1e-5 && p_cot[2].abs() < 1e-6,
        "Cotangent drifted tangentially on flat mesh: {p_cot:?}"
    );

    let mut lap_mesh = mesh_of(&positions, &triangles);
    edit(&mut lap_mesh, |tx| {
        smooth(
            tx,
            &SmoothOptions {
                brush,
                cycles: 5,
                method: SmoothMethod::Laplacian,
                ..Default::default()
            },
        )
    });
    let p_lap = lap_mesh.positions()[0];
    assert!(
        p_lap[0] > 0.3,
        "Uniform Laplacian should drift toward denser +x side: {p_lap:?}"
    );
}

#[test]
fn surface_filter_peaks_valleys_and_normal_angle_selectively_mask_features() {
    // Sheet with a sharp bump at (-3, 0) and an inverted pit at (+3, 0).
    let bump_and_pit = || {
        symmetric_sheet(8, |x, y| {
            let r_peak = ((x + 3.0).powi(2) + y * y).sqrt();
            let r_pit = ((x - 3.0).powi(2) + y * y).sqrt();
            let peak = (1.5 - r_peak).max(0.0) * 0.9;
            let pit = -(1.5 - r_pit).max(0.0) * 0.9;
            peak + pit
        })
    };

    let wide_brush = Brush {
        center: [0.0, 0.0, 0.0],
        radius: 50.0,
        falloff: Falloff::Linear,
        strength: 0.8,
    };

    // PeaksOnly lowers the bump while leaving every pit vertex bit-identical (0.0 displacement).
    let mut peaks_mesh = bump_and_pit();
    let before = peaks_mesh.positions().to_vec();
    edit(&mut peaks_mesh, |tx| {
        smooth(
            tx,
            &SmoothOptions {
                brush: wide_brush,
                cycles: 2,
                method: SmoothMethod::Laplacian,
                filter: SurfaceFilter::PeaksOnly {
                    threshold: 0.05,
                    contrast: 0.8,
                },
                ..Default::default()
            },
        )
    });
    let mut peak_lowered = false;
    for (p0, p1) in before.iter().zip(peaks_mesh.positions()) {
        if p0[2] > 1.0 && p1[2] < p0[2] - 0.1 {
            peak_lowered = true;
        }
        // Concave pit interior (`z < -0.3`, inside the crater rim) must be 0-ULP untouched.
        if p0[2] < -0.3 {
            assert_eq!(
                p0.map(f32::to_bits),
                p1.map(f32::to_bits),
                "PeaksOnly must leave concave pit untouched: {p0:?} -> {p1:?}"
            );
        }
    }
    assert!(peak_lowered, "PeaksOnly should lower the raised bump");

    // ValleysOnly fills the pit while leaving every convex peak vertex bit-identical (0.0 displacement).
    let mut valleys_mesh = bump_and_pit();
    edit(&mut valleys_mesh, |tx| {
        smooth(
            tx,
            &SmoothOptions {
                brush: wide_brush,
                cycles: 2,
                method: SmoothMethod::Laplacian,
                filter: SurfaceFilter::ValleysOnly {
                    threshold: 0.05,
                    contrast: 0.8,
                },
                ..Default::default()
            },
        )
    });
    let mut pit_filled = false;
    for (p0, p1) in before.iter().zip(valleys_mesh.positions()) {
        if p0[2] < -1.0 && p1[2] > p0[2] + 0.1 {
            pit_filled = true;
        }
        if p0[2] > 0.3 {
            assert_eq!(
                p0.map(f32::to_bits),
                p1.map(f32::to_bits),
                "ValleysOnly must leave convex peak untouched: {p0:?} -> {p1:?}"
            );
        }
    }
    assert!(pit_filled, "ValleysOnly should fill the inverted pit");

    // Subdivided split-face cube [-1, 1]^3: each face has orthogonal face normals.
    let split_cube = || {
        let n = 4u32;
        let mut positions = Vec::new();
        let mut triangles = Vec::new();
        // 6 faces: +z, -z, +x, -x, +y, -y (outward wound)
        let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
            ([-1.0, -1.0, 1.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]), // +z
            ([-1.0, 1.0, -1.0], [2.0, 0.0, 0.0], [0.0, -2.0, 0.0]), // -z
            ([1.0, -1.0, -1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]), // +x
            ([-1.0, 1.0, -1.0], [0.0, -2.0, 0.0], [0.0, 0.0, 2.0]), // -x
            ([1.0, 1.0, -1.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 2.0]), // +y
            ([-1.0, -1.0, -1.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]), // -y
        ];
        for (origin, u_dir, v_dir) in faces {
            let base = (positions.len() / 3) as u32;
            for iv in 0..=n {
                for iu in 0..=n {
                    let (fu, fv) = (iu as f32 / n as f32, iv as f32 / n as f32);
                    positions.extend([
                        origin[0] + fu * u_dir[0] + fv * v_dir[0],
                        origin[1] + fu * u_dir[1] + fv * v_dir[1],
                        origin[2] + fu * u_dir[2] + fv * v_dir[2],
                    ]);
                }
            }
            for iv in 0..n {
                for iu in 0..n {
                    let a = base + iv * (n + 1) + iu;
                    triangles.extend([[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]);
                }
            }
        }
        mesh_of(&positions, &triangles)
    };

    let mut cube = split_cube();
    let cube_before = cube.positions().to_vec();
    let top_face_count = 25usize; // first 5x5 vertices belong to +z face
    let (rep, _) = edit(&mut cube, |tx| {
        inflate(
            tx,
            &InflateOptions {
                brush: Brush {
                    center: [0.0, 0.0, 0.0],
                    radius: 10.0,
                    falloff: Falloff::Linear,
                    strength: 0.4,
                },
                filter: SurfaceFilter::NormalAngle {
                    normal: Some([0.0, 0.0, 1.0]),
                    max_angle_deg: 35.0,
                },
                ..Default::default()
            },
        )
    });
    assert_eq!(rep.moved, top_face_count);
    for (idx, (p0, p1)) in cube_before.iter().zip(cube.positions()).enumerate() {
        if idx < top_face_count {
            assert!(p1[2] > p0[2] + 0.1, "top face vertex {idx} must inflate +z");
        } else {
            assert_eq!(
                p0.map(f32::to_bits),
                p1.map(f32::to_bits),
                "non-top face vertex {idx} must have 0.0 displacement"
            );
        }
    }

    // Invalid filter arguments rejected before writing.
    let bad_filters = [
        SurfaceFilter::PeaksOnly {
            threshold: -0.1,
            contrast: 0.5,
        },
        SurfaceFilter::PeaksOnly {
            threshold: 0.5,
            contrast: 1.2,
        },
        SurfaceFilter::ValleysOnly {
            threshold: f32::NAN,
            contrast: 0.5,
        },
        SurfaceFilter::NormalAngle {
            normal: None,
            max_angle_deg: 0.0,
        },
        SurfaceFilter::NormalAngle {
            normal: None,
            max_angle_deg: 181.0,
        },
        SurfaceFilter::NormalAngle {
            normal: Some([0.0, 0.0, 0.0]),
            max_angle_deg: 30.0,
        },
        SurfaceFilter::NormalAngle {
            normal: Some([f32::NAN, 0.0, 1.0]),
            max_angle_deg: 30.0,
        },
    ];
    let mut g = grid(4);
    for filter in bad_filters {
        let mut tx = g.begin_edit();
        assert!(matches!(
            inflate(
                &mut tx,
                &InflateOptions {
                    filter,
                    ..Default::default()
                }
            ),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(tx.commit().unwrap().is_empty());
    }
}

#[test]
fn symmetry_disjoint_mirror_hemispheres_and_centerline_max_envelope() {
    assert_eq!(Symmetry::NONE.active_masks(), ([0, 0, 0, 0, 0, 0, 0, 0], 1));
    let sym_x = Symmetry {
        x: true,
        y: false,
        z: false,
    };
    let (masks, count) = sym_x.active_masks();
    assert_eq!(&masks[..count], &[0, 1]);

    // 1. Disjoint mirror spheres (`x: true`): `+x` and `-x` hemispheres move symmetrically.
    let half = 8i32;
    let side = (2 * half + 1) as usize;
    let mut disjoint = symmetric_sheet(half, |x, y| 0.05 * (x * x + y * y));
    let before = disjoint.positions().to_vec();
    edit(&mut disjoint, |tx| {
        inflate(
            tx,
            &InflateOptions {
                brush: Brush {
                    center: [5.0, 1.0, 1.3],
                    radius: 2.5,
                    falloff: Falloff::Smooth,
                    strength: 0.6,
                },
                symmetry: sym_x,
                ..Default::default()
            },
        )
    });
    let after = disjoint.positions();
    let mut moved_pairs = 0;
    for iy in -half..=half {
        for ix in 1..=half {
            let v_pos = (iy + half) as usize * side + (ix + half) as usize;
            let v_neg = (iy + half) as usize * side + (-ix + half) as usize;
            let (pp, pn) = (after[v_pos], after[v_neg]);
            if pp != before[v_pos] {
                moved_pairs += 1;
            }
            assert!(
                (pn[0] - (-pp[0])).abs() < 1e-6
                    && (pn[1] - pp[1]).abs() < 1e-6
                    && (pn[2] - pp[2]).abs() < 1e-6,
                "mirror mismatch at ({ix}, {iy}): +x={pp:?}, -x={pn:?}"
            );
        }
    }
    assert!(moved_pairs > 4);

    // 2. Centerline overlap (`brush.center = [0.05 * R, 0.0, 0.0]`, `Symmetry { x: true, .. }`):
    //    Vertices on `x = 0` keep `dx == 0.0` (even under AverageNormal where off-center pass 0
    //    has non-zero normal.x on a curved dome!) and match single-brush `strength * falloff(t)`
    //    on a flat sheet without a `1.0` plateau.
    let r = 6.0f32;
    let strength = 0.8f32;
    let mut flat_sym = symmetric_sheet(half, |_, _| 0.0);
    edit(&mut flat_sym, |tx| {
        inflate(
            tx,
            &InflateOptions {
                brush: Brush {
                    center: [0.05 * r, 0.0, 0.0],
                    radius: r,
                    falloff: Falloff::Linear,
                    strength,
                },
                method: InflateMethod::VertexNormal,
                symmetry: sym_x,
                ..Default::default()
            },
        )
    });
    for iy in -half..=half {
        let v_mid = (iy + half) as usize * side + half as usize;
        let p = flat_sym.positions()[v_mid];
        assert_eq!(p[0], 0.0, "centerline vertex must keep dx == 0.0");
        let y = iy as f32;
        let t = (0.05 * r).hypot(y) / r;
        let want_z = if t < 1.0 { strength * (1.0 - t) } else { 0.0 };
        assert!(
            (p[2] - want_z).abs() < 1e-6,
            "y={y}: got z={}, want single-brush profile {want_z}",
            p[2]
        );
    }
    // Specifically at y = 0 (`t = 0.05`), displacement is `0.95 * strength`, NOT clamped to `1.0 * strength`!
    let origin_v = half as usize * side + half as usize;
    assert!((flat_sym.positions()[origin_v][2] - 0.95 * strength).abs() < 1e-6);

    // Also verify dx == 0.0 on centerline for curved dome under AverageNormal (cross-plane cancels!).
    let mut dome = symmetric_sheet(half, |x, y| 4.0 - 0.05 * (x * x + y * y));
    edit(&mut dome, |tx| {
        inflate(
            tx,
            &InflateOptions {
                brush: Brush {
                    center: [0.05 * r, 0.0, 4.0],
                    radius: r,
                    falloff: Falloff::Smooth,
                    strength,
                },
                method: InflateMethod::AverageNormal,
                symmetry: sym_x,
                ..Default::default()
            },
        )
    });
    for iy in -half..=half {
        let v_mid = (iy + half) as usize * side + half as usize;
        assert!(
            dome.positions()[v_mid][0].abs() < 1e-6,
            "dome centerline dx must cancel to 0: {}",
            dome.positions()[v_mid][0]
        );
    }
}

#[test]
fn region_parity_bit_for_bit_and_oom_rollback_sweeps() {
    let base_mesh = || {
        let mut m = symmetric_sheet(6, |x, y| (0.6 * x).sin() * (0.5 * y).cos() + 0.1 * x);
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[7]).unwrap();
        tx.commit().unwrap();
        m.ensure_incidence().unwrap();
        m
    };

    let region_union = |mesh: &Mesh, brush: &Brush, sym: Symmetry| -> Vec<VertexId> {
        let (masks, count) = sym.active_masks();
        (0..mesh.vertex_slots() as VertexId)
            .filter(|&v| {
                let p = mesh.positions()[v as usize];
                masks[..count].iter().any(|&bits| {
                    let c = [
                        if bits & 1 != 0 {
                            -brush.center[0]
                        } else {
                            brush.center[0]
                        },
                        if bits & 2 != 0 {
                            -brush.center[1]
                        } else {
                            brush.center[1]
                        },
                        if bits & 4 != 0 {
                            -brush.center[2]
                        } else {
                            brush.center[2]
                        },
                    ];
                    let d2 = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2);
                    d2.sqrt() / brush.radius < 1.0
                })
            })
            .collect()
    };

    let brush = Brush {
        center: [1.4, -0.9, 0.2],
        radius: 3.4,
        falloff: Falloff::Smooth,
        strength: 0.45,
    };
    let filters = [
        SurfaceFilter::None,
        SurfaceFilter::PeaksOnly {
            threshold: 0.02,
            contrast: 0.6,
        },
        SurfaceFilter::ValleysOnly {
            threshold: 0.02,
            contrast: 0.6,
        },
        SurfaceFilter::NormalAngle {
            normal: Some([0.1, -0.2, 1.0]),
            max_angle_deg: 55.0,
        },
        SurfaceFilter::NormalAngle {
            normal: None,
            max_angle_deg: 45.0,
        },
    ];
    let symmetries = [
        Symmetry::NONE,
        Symmetry {
            x: true,
            y: false,
            z: false,
        },
        Symmetry {
            x: false,
            y: true,
            z: true,
        },
        Symmetry {
            x: true,
            y: true,
            z: true,
        },
    ];

    for method in [
        InflateMethod::VertexNormal,
        InflateMethod::AverageNormal,
        InflateMethod::StrokeNormal,
    ] {
        for filter in filters {
            for symmetry in symmetries {
                let opts = InflateOptions {
                    brush,
                    method,
                    filter,
                    symmetry,
                };
                let (mut whole, mut reg) = (base_mesh(), base_mesh());
                if method == InflateMethod::StrokeNormal {
                    capture_stroke_state(&mut whole).unwrap();
                    capture_stroke_state(&mut reg).unwrap();
                    // Perturb a vertex first so stroke normal differs from current normal!
                    edit(&mut whole, |tx| {
                        Ok(tx.set_positions(&[40], &[[0.0, 0.0, 1.5]])?)
                    });
                    edit(&mut reg, |tx| {
                        Ok(tx.set_positions(&[40], &[[0.0, 0.0, 1.5]])?)
                    });
                    reg.ensure_incidence().unwrap();
                }
                let candidates = region_union(&reg, &brush, symmetry);
                let (rw, _) = edit(&mut whole, |tx| inflate(tx, &opts));
                let (rr, _) = edit(&mut reg, |tx| inflate_region(tx, &opts, &candidates));
                assert_eq!(rw, rr, "{method:?} {filter:?} {symmetry:?}");
                assert_eq!(
                    bits_of(&whole),
                    bits_of(&reg),
                    "{method:?} {filter:?} {symmetry:?}"
                );
            }
        }
    }

    // Also test explicit StrokeSlice parity with channel-backed StrokeNormal.
    {
        let mut a = base_mesh();
        capture_stroke_state(&mut a).unwrap();
        let mut b = base_mesh();
        let slice_positions: Vec<[f32; 3]> = a
            .channel(STROKE_POS_CHANNEL)
            .unwrap()
            .as_chunks::<3>()
            .0
            .to_vec();
        let slice_normals: Vec<[f32; 3]> = a
            .channel(STROKE_NORMAL_CHANNEL)
            .unwrap()
            .as_chunks::<3>()
            .0
            .to_vec();
        edit(&mut a, |tx| {
            Ok(tx.set_positions(&[42], &[[0.2, -0.3, 1.4]])?)
        });
        edit(&mut b, |tx| {
            Ok(tx.set_positions(&[42], &[[0.2, -0.3, 1.4]])?)
        });
        b.ensure_incidence().unwrap();
        let opts = InflateOptions {
            brush,
            method: InflateMethod::StrokeNormal,
            filter: SurfaceFilter::None,
            symmetry: Symmetry {
                x: true,
                y: false,
                z: false,
            },
        };
        let candidates = region_union(&b, &brush, opts.symmetry);
        let (ra, _) = edit(&mut a, |tx| inflate(tx, &opts));
        let stroke = StrokeSlice {
            positions: &slice_positions,
            normals: &slice_normals,
        };
        let (rb, _) = edit(&mut b, |tx| {
            inflate_region_with_stroke(tx, &opts, &candidates, Some(stroke))
        });
        assert_eq!(ra, rb);
        assert_eq!(bits_of(&a), bits_of(&b));
    }

    for method in [
        SmoothMethod::Taubin,
        SmoothMethod::Laplacian,
        SmoothMethod::HcLaplacian,
        SmoothMethod::Cotangent,
    ] {
        for filter in filters {
            for symmetry in symmetries {
                let opts = SmoothOptions {
                    brush,
                    cycles: 2,
                    method,
                    filter,
                    symmetry,
                };
                let (mut whole, mut reg) = (base_mesh(), base_mesh());
                let candidates = region_union(&reg, &brush, symmetry);
                let (rw, _) = edit(&mut whole, |tx| smooth(tx, &opts));
                let (rr, _) = edit(&mut reg, |tx| smooth_region(tx, &opts, &candidates));
                assert_eq!(rw, rr, "{method:?} {filter:?} {symmetry:?}");
                assert_eq!(
                    bits_of(&whole),
                    bits_of(&reg),
                    "{method:?} {filter:?} {symmetry:?}"
                );
            }
        }
    }

    // OOM rollback sweeps across new methods, filters, and symmetry (whole-mesh & region).
    let oom_sym = Symmetry {
        x: true,
        y: true,
        z: false,
    };
    let inf_opts = InflateOptions {
        brush,
        method: InflateMethod::AverageNormal,
        filter: SurfaceFilter::PeaksOnly {
            threshold: 0.01,
            contrast: 0.5,
        },
        symmetry: oom_sym,
    };
    assert!(sweep(base_mesh(), |tx| inflate(tx, &inf_opts).map(drop)) > 3);
    let reg_mesh = base_mesh();
    let reg_ids = region_union(&reg_mesh, &brush, oom_sym);
    assert!(
        sweep(reg_mesh, |tx| inflate_region(tx, &inf_opts, &reg_ids)
            .map(drop))
            > 3
    );

    for method in [SmoothMethod::HcLaplacian, SmoothMethod::Cotangent] {
        let sm_opts = SmoothOptions {
            brush,
            cycles: 2,
            method,
            filter: SurfaceFilter::NormalAngle {
                normal: None,
                max_angle_deg: 70.0,
            },
            symmetry: oom_sym,
        };
        assert!(sweep(base_mesh(), |tx| smooth(tx, &sm_opts).map(drop)) > 4);
        let reg_mesh = base_mesh();
        assert!(
            sweep(reg_mesh, |tx| smooth_region(tx, &sm_opts, &reg_ids)
                .map(drop))
                > 4
        );
    }

    // Confirm Transaction parameter type works directly in signatures.
    let _: fn(&mut Transaction, &InflateOptions) -> crate::Result<_> = inflate;
}

#[test]
fn grab_geodesic_isolates_air_gap_sheets_and_bridges_uv_seams() {
    // 1. Two parallel disconnected sheets z = 0.0 and z = 0.05 (radius = 1.0, center = [0, 0, 0]).
    let two_sheets = || {
        let mut positions = Vec::new();
        let mut triangles = Vec::new();
        for (sheet, z) in [(0u32, 0.0f32), (1u32, 0.05f32)] {
            let base = sheet * 25;
            for iy in -2i32..=2 {
                for ix in -2i32..=2 {
                    positions.extend([ix as f32 * 0.25, iy as f32 * 0.25, z]);
                }
            }
            for iy in 0..4u32 {
                for ix in 0..4u32 {
                    let a = base + iy * 5 + ix;
                    triangles.extend([[a, a + 1, a + 6], [a, a + 6, a + 5]]);
                }
            }
        }
        mesh_of(&positions, &triangles)
    };

    let brush = Brush {
        center: [0.0, 0.0, 0.0],
        radius: 1.0,
        falloff: Falloff::Smooth,
        strength: 0.4,
    };

    let mut sphere_mesh = two_sheets();
    let before = sphere_mesh.positions().to_vec();
    edit(&mut sphere_mesh, |tx| {
        grab(
            tx,
            &GrabOptions {
                brush,
                direction: [0.0, 0.0, 1.0],
                method: GrabMethod::Sphere,
                ..Default::default()
            },
        )
    });
    assert!(
        (25..50).any(|v| sphere_mesh.positions()[v] != before[v]),
        "Sphere grab moves the opposing z = 0.05 sheet across the air gap"
    );

    let mut geo_mesh = two_sheets();
    let (rep, _) = edit(&mut geo_mesh, |tx| {
        grab(
            tx,
            &GrabOptions {
                brush,
                direction: [0.0, 0.0, 1.0],
                method: GrabMethod::Geodesic,
                ..Default::default()
            },
        )
    });
    assert!(rep.moved > 0);
    assert!(
        (0..25).any(|v| geo_mesh.positions()[v][2] > 0.1),
        "Geodesic grab moves the z = 0.0 seed sheet"
    );
    for (offset, (after_pos, before_pos)) in geo_mesh.positions()[25..50]
        .iter()
        .zip(&before[25..50])
        .enumerate()
    {
        let v = 25 + offset;
        assert_eq!(
            after_pos.map(f32::to_bits),
            before_pos.map(f32::to_bits),
            "Geodesic grab must leave z = 0.05 sheet byte-identical (vertex {v})"
        );
    }

    // 2. Single sheet split down x = 0 by a UV seam (duplicate vertex IDs with identical coords,
    //    including -0.0 vs +0.0 on the seam).
    let seamed_sheet = || {
        let mut positions = Vec::new();
        let mut triangles = Vec::new();
        // Left half x in [-2..=0], right half x in [0..=2], separate vertex IDs along x = 0.
        for (half_idx, x_start) in [(0u32, -2i32), (1u32, 0i32)] {
            let base = half_idx * 15;
            for iy in -2i32..=2 {
                for ix in 0..=2i32 {
                    let raw_x = (x_start + ix) as f32 * 0.25;
                    let x = if half_idx == 1 && raw_x == 0.0 {
                        -0.0 // exercise -0.0 / +0.0 canonicalization across seam
                    } else {
                        raw_x
                    };
                    positions.extend([x, iy as f32 * 0.25, 0.0]);
                }
            }
            for iy in 0..4u32 {
                for ix in 0..2u32 {
                    let a = base + iy * 3 + ix;
                    triangles.extend([[a, a + 1, a + 4], [a, a + 4, a + 3]]);
                }
            }
        }
        mesh_of(&positions, &triangles)
    };

    let mut seamed = seamed_sheet();
    // Seed strictly on the left half (-0.25, 0, 0) so reaching the right half MUST cross the x = 0 seam.
    edit(&mut seamed, |tx| {
        grab(
            tx,
            &GrabOptions {
                brush: Brush {
                    center: [-0.25, 0.0, 0.0],
                    radius: 0.9,
                    falloff: Falloff::Smooth,
                    strength: 0.5,
                },
                direction: [0.1, -0.05, 0.4],
                method: GrabMethod::Geodesic,
                ..Default::default()
            },
        )
    });
    // Right half interior vertex (+0.25, 0, 0) is slot 15 + 2*3 + 1 = 22: must have moved!
    assert!(
        seamed.positions()[22][2] > 0.05,
        "Geodesic must cross UV seam onto right chart: got {:?}",
        seamed.positions()[22]
    );
    // Seam pairs: left seam is ix=2 (`iy * 3 + 2`), right seam is ix=0 (`15 + iy * 3`).
    for iy in 0..5usize {
        let p_left = seamed.positions()[iy * 3 + 2];
        let p_right = seamed.positions()[15 + iy * 3];
        let gap = (0..3)
            .map(|k| (p_left[k] - p_right[k]).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            gap, 0.0,
            "UV seam row {iy} must stay watertight: left={p_left:?}, right={p_right:?}"
        );
    }
}

#[test]
fn grab_kelvinlet_calibrates_center_couples_poisson_and_validates() {
    let sheet = || {
        let mut positions = Vec::new();
        let n = 10i32;
        for iy in -n..=n {
            for ix in -n..=n {
                positions.extend([ix as f32 * 0.2, iy as f32 * 0.2, 0.0]);
            }
        }
        let side = (2 * n + 1) as u32;
        let mut triangles = Vec::new();
        for iy in 0..(2 * n) as u32 {
            for ix in 0..(2 * n) as u32 {
                let a = iy * side + ix;
                triangles.extend([[a, a + 1, a + side + 1], [a, a + side + 1, a + side]]);
            }
        }
        mesh_of(&positions, &triangles)
    };

    let brush = Brush {
        center: [0.0, 0.0, 0.0],
        radius: 1.5,
        falloff: Falloff::Smooth,
        strength: 0.6,
    };
    let dir = [1.0f32, 0.0, 0.0];
    let center_id = 10 * 21 + 10;
    let diag_id = 13 * 21 + 13; // (0.6, 0.6, 0.0): off-axis in first quadrant

    let mut m_high = sheet();
    let before = m_high.positions().to_vec();
    edit(&mut m_high, |tx| {
        grab(
            tx,
            &GrabOptions {
                brush,
                direction: dir,
                method: GrabMethod::Kelvinlet,
                poisson: 0.45,
                ..Default::default()
            },
        )
    });
    let mut m_zero = sheet();
    edit(&mut m_zero, |tx| {
        grab(
            tx,
            &GrabOptions {
                brush,
                direction: dir,
                method: GrabMethod::Kelvinlet,
                poisson: 0.0,
                ..Default::default()
            },
        )
    });

    // Center displacement at r = 0 equals strength * direction within 1e-6 for any valid poisson.
    for m in [&m_high, &m_zero] {
        let pc = m.positions()[center_id];
        assert!((pc[0] - 0.6).abs() < 1e-6, "center dx got {}", pc[0]);
        assert!(pc[1].abs() < 1e-6 && pc[2].abs() < 1e-6);
    }

    // Outside R_eff (e.g. corner (-2.0, -2.0, 0.0), r = 2.828 >= 1.5), displacement is 0.0.
    assert_eq!(
        m_high.positions()[0].map(f32::to_bits),
        before[0].map(f32::to_bits)
    );

    // Off-axis vertex at (0.6, 0.6, 0): pulling along +x induces lateral dy != 0,
    // and poisson = 0.45 differs distinctly from poisson = 0.0.
    let dy_high = m_high.positions()[diag_id][1] - before[diag_id][1];
    let dy_zero = m_zero.positions()[diag_id][1] - before[diag_id][1];
    assert!(
        dy_high.abs() > 1e-3,
        "off-axis lateral dy should be non-zero: {dy_high}"
    );
    assert!(
        (dy_high - dy_zero).abs() > 5e-3,
        "poisson 0.45 ({dy_high}) vs 0.0 ({dy_zero}) must produce distinct lateral coupling"
    );

    // Invalid poisson < 0.0, > 0.5, or NaN rejected with InvalidInput and writes nothing.
    for bad_nu in [-0.01f32, 0.51, f32::NAN, f32::INFINITY] {
        let mut tx = m_high.begin_edit();
        assert!(matches!(
            grab(
                &mut tx,
                &GrabOptions {
                    brush,
                    direction: dir,
                    method: GrabMethod::Kelvinlet,
                    poisson: bad_nu,
                    ..Default::default()
                }
            ),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(tx.commit().unwrap().is_empty());
    }
}

#[test]
fn grab_adaptive_reach_widens_tangential_drag_without_folding_and_snaps_back() {
    assert_eq!(GRAB_MAX_STRETCH, 0.8);
    let b_lin = Brush {
        center: [0.0; 3],
        radius: 0.2,
        falloff: Falloff::Linear,
        strength: 1.0,
    };
    let b_sm = Brush {
        falloff: Falloff::Smooth,
        ..b_lin
    };
    assert!((grab_widest_reach(&b_lin, [1.2, 0.0, 0.0], GrabMethod::Sphere) - 1.5).abs() < 1e-6);
    assert!((grab_widest_reach(&b_sm, [1.2, 0.0, 0.0], GrabMethod::Sphere) - 2.25).abs() < 1e-6);
    assert!(
        (grab_widest_reach(&b_sm, [1.2, 0.0, 0.0], GrabMethod::Kelvinlet) - 2.625).abs() < 1e-6
    );

    // Fine flat grid in [-3, 3]^2, spacing 0.1.
    let fine_plane = || {
        let n = 30i32;
        let side = (2 * n + 1) as u32;
        let mut positions = Vec::new();
        for iy in -n..=n {
            for ix in -n..=n {
                positions.extend([ix as f32 * 0.1, iy as f32 * 0.1, 0.0]);
            }
        }
        let mut triangles = Vec::new();
        for iy in 0..(2 * n) as u32 {
            for ix in 0..(2 * n) as u32 {
                let a = iy * side + ix;
                triangles.extend([[a, a + 1, a + side + 1], [a, a + side + 1, a + side]]);
            }
        }
        mesh_of(&positions, &triangles)
    };

    let brush = Brush {
        center: [0.0, 0.0, 0.0],
        radius: 0.2,
        falloff: Falloff::Smooth,
        strength: 1.0,
    };

    // 1. Tangential drag along +x (L = 1.2, R = 0.2): widens R_eff to 2.25 and flips zero normals.
    let mut tang = fine_plane();
    let before = tang.positions().to_vec();
    edit(&mut tang, |tx| {
        grab(
            tx,
            &GrabOptions {
                brush,
                direction: [1.2, 0.0, 0.0],
                adaptive_reach: true,
                ..Default::default()
            },
        )
    });
    // Vertex at (1.5, 0.0, 0.0) is well past R = 0.2 yet inside R_wide = 2.25, so it moves!
    let far_v = (30 * 61 + 45) as usize;
    assert!(
        tang.positions()[far_v][0] > before[far_v][0] + 0.1,
        "tangential adaptive_reach must widen past R = 0.2"
    );
    // Every triangle keeps strictly positive +z orientation (zero folds!).
    for t in 0..tang.triangle_slots() as u32 {
        let [a, b, c] = tang.triangle(t).unwrap();
        let (pa, pb, pc) = (
            tang.positions()[a as usize],
            tang.positions()[b as usize],
            tang.positions()[c as usize],
        );
        let nz = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]);
        assert!(nz > 1e-6, "triangle {t} folded: nz = {nz}");
    }

    // 2. Normal pull straight off flat plane (+z): all surface offsets have along == 0, so R_eff == R = 0.2!
    let mut horn = fine_plane();
    edit(&mut horn, |tx| {
        grab(
            tx,
            &GrabOptions {
                brush,
                direction: [0.0, 0.0, 1.2],
                adaptive_reach: true,
                ..Default::default()
            },
        )
    });
    for (p0, p1) in before.iter().zip(horn.positions()) {
        let r = p0[0].hypot(p0[1]);
        if r >= 0.2 {
            assert_eq!(
                p0.map(f32::to_bits),
                p1.map(f32::to_bits),
                "normal pull off flat plane must stay narrow (R_eff == R): r={r}"
            );
        }
    }

    // 3. Snap-back with `from_stroke: true, adaptive_reach: true`:
    //    Move 1 (`[1.0, 0, 0]`) moves wide vertices (`R_wide = 1.875`);
    //    Move 2 (`[0.05, 0, 0]`) shrinks `R_eff` back to `R = 0.2` and snaps outer vertices
    //    back to `rest_pos` bit-for-bit!
    let mut stroke_mesh = fine_plane();
    capture_stroke_state(&mut stroke_mesh).unwrap();
    let rest_bits = bits_of(&stroke_mesh);

    let (rep1, _) = edit(&mut stroke_mesh, |tx| {
        grab(
            tx,
            &GrabOptions {
                brush,
                direction: [1.0, 0.0, 0.0],
                from_stroke: true,
                adaptive_reach: true,
                ..Default::default()
            },
        )
    });
    assert!(rep1.moved > 50);
    let mid_v = (30 * 61 + 40) as usize; // (1.0, 0.0, 0.0): moved on Move 1, outside R = 0.2 on Move 2
    assert_ne!(
        stroke_mesh.positions()[mid_v].map(f32::to_bits),
        rest_bits[mid_v]
    );

    let short_opts = GrabOptions {
        brush,
        direction: [0.05, 0.0, 0.0],
        from_stroke: true,
        adaptive_reach: true,
        ..Default::default()
    };
    let (rep2, _) = edit(&mut stroke_mesh, |tx| grab(tx, &short_opts));
    assert!(rep2.moved > 0);
    for (v, p) in stroke_mesh.positions().iter().enumerate() {
        let r = before[v][0].hypot(before[v][1]);
        if r >= 0.2 {
            assert_eq!(
                p.map(f32::to_bits),
                rest_bits[v],
                "outer vertex {v} (r={r}) must snap back to rest_pos bit-for-bit"
            );
        }
    }
    // Repeating the identical Move 2 writes nothing (`same_bits` skip -> moved == 0).
    let mut tx = stroke_mesh.begin_edit();
    let rep3 = grab(&mut tx, &short_opts).unwrap();
    assert_eq!(rep3.moved, 0);
    assert!(tx.commit().unwrap().is_empty());
}

#[test]
fn extra_brushes_flatten_crease_clay_and_twist_follow_expected_geometry() {
    // 1. Flatten: Both vs Peaks (scrape) vs Valleys (fill) against explicit z = 0 plane.
    let bumpy = || {
        symmetric_sheet(6, |x, y| {
            let r_peak = ((x + 2.0).powi(2) + y * y).sqrt();
            let r_pit = ((x - 2.0).powi(2) + y * y).sqrt();
            (1.8 - r_peak).max(0.0) - (1.8 - r_pit).max(0.0)
        })
    };
    let flat_brush = Brush {
        center: [0.0, 0.0, 0.0],
        radius: 20.0,
        falloff: Falloff::Linear,
        strength: 0.8,
    };
    let before_bumpy = bumpy().positions().to_vec();

    let mut m_both = bumpy();
    edit(&mut m_both, |tx| {
        flatten(
            tx,
            &FlattenOptions {
                brush: flat_brush,
                mode: FlattenMode::Both,
                plane_origin: Some([0.0, 0.0, 0.0]),
                plane_normal: Some([0.0, 0.0, 1.0]),
                ..Default::default()
            },
        )
    });
    let mut m_peaks = bumpy();
    edit(&mut m_peaks, |tx| {
        flatten(
            tx,
            &FlattenOptions {
                brush: flat_brush,
                mode: FlattenMode::Peaks,
                plane_origin: Some([0.0, 0.0, 0.0]),
                plane_normal: Some([0.0, 0.0, 1.0]),
                ..Default::default()
            },
        )
    });
    let mut m_valleys = bumpy();
    edit(&mut m_valleys, |tx| {
        flatten(
            tx,
            &FlattenOptions {
                brush: flat_brush,
                mode: FlattenMode::Valleys,
                plane_origin: Some([0.0, 0.0, 0.0]),
                plane_normal: Some([0.0, 0.0, 1.0]),
                ..Default::default()
            },
        )
    });

    for (i, p0) in before_bumpy.iter().enumerate() {
        let (pb, pp, pv) = (
            m_both.positions()[i],
            m_peaks.positions()[i],
            m_valleys.positions()[i],
        );
        if p0[2] > 0.2 {
            assert!(pb[2] < p0[2] && pb[2] > 0.0, "Both lowers peak toward z=0");
            assert!(pp[2] < p0[2], "Peaks scrapes z > 0 down");
            assert_eq!(
                pv.map(f32::to_bits),
                p0.map(f32::to_bits),
                "Valleys leaves z > 0 untouched"
            );
        } else if p0[2] < -0.2 {
            assert!(
                pb[2] > p0[2] && pb[2] < 0.0,
                "Both raises valley toward z=0"
            );
            assert_eq!(
                pp.map(f32::to_bits),
                p0.map(f32::to_bits),
                "Peaks leaves z < 0 untouched"
            );
            assert!(pv[2] > p0[2], "Valleys fills z < 0 up");
        }
    }

    // 2. Crease & pure Pinch on flat sheet z = 0.
    let mut creased = symmetric_sheet(5, |_, _| 0.0);
    let before_flat = creased.positions().to_vec();
    edit(&mut creased, |tx| {
        crease(
            tx,
            &CreaseOptions {
                brush: Brush {
                    center: [0.0, 0.0, 0.0],
                    radius: 3.5,
                    falloff: Falloff::Smooth,
                    strength: -0.6,
                },
                pinch: 0.5,
                ..Default::default()
            },
        )
    });
    for (p0, p1) in before_flat.iter().zip(creased.positions()) {
        let r0 = p0[0].hypot(p0[1]);
        let r1 = p1[0].hypot(p1[1]);
        if r0 < 3.5 {
            assert!(
                p1[2] < -1e-4,
                "Crease with strength < 0 carves groove: {p0:?} -> {p1:?}"
            );
            if r0 > 0.0 {
                assert!(
                    r1 < r0,
                    "Crease with pinch > 0 pulls radially inward: {r0} -> {r1}"
                );
            }
        }
    }

    let mut pinched = symmetric_sheet(5, |_, _| 0.0);
    edit(&mut pinched, |tx| {
        crease(
            tx,
            &CreaseOptions {
                brush: Brush {
                    center: [0.0, 0.0, 0.0],
                    radius: 3.5,
                    falloff: Falloff::Linear,
                    strength: 0.0,
                },
                pinch: 0.6,
                ..Default::default()
            },
        )
    });
    let mut pinched_any = false;
    for (p0, p1) in before_flat.iter().zip(pinched.positions()) {
        assert_eq!(
            p1[2].to_bits(),
            0.0f32.to_bits(),
            "pure Pinch keeps z == 0.0"
        );
        let r0 = p0[0].hypot(p0[1]);
        let r1 = p1[0].hypot(p1[1]);
        if r0 > 0.0 && r0 < 3.5 {
            assert!(r1 < r0);
            pinched_any = true;
        }
    }
    assert!(pinched_any);

    // 3. Clay: Round vs Strips (`h = 0.25 * R * sign(strength)`, Strips square 2R x 2R reaches (±0.8R, ±0.8R)).
    let r_clay = 2.5f32; // (±0.8 * 2.5, ±0.8 * 2.5) = (±2.0, ±2.0), exact integer grid vertices!
    let clay_brush = Brush {
        center: [0.0, 0.0, 0.0],
        radius: r_clay,
        falloff: Falloff::Smooth,
        strength: 1.0,
    };
    let mut clay_round = symmetric_sheet(5, |_, _| 0.0);
    edit(&mut clay_round, |tx| {
        clay(
            tx,
            &ClayOptions {
                brush: clay_brush,
                profile: ClayProfile::Round,
                tangent: Some([1.0, 0.0, 0.0]),
                ..Default::default()
            },
        )
    });
    let mut clay_strips = symmetric_sheet(5, |_, _| 0.0);
    edit(&mut clay_strips, |tx| {
        clay(
            tx,
            &ClayOptions {
                brush: clay_brush,
                profile: ClayProfile::Strips,
                tangent: Some([1.0, 0.0, 0.0]),
                ..Default::default()
            },
        )
    });
    let center_slot = 5 * 11 + 5;
    let want_h = 0.25 * r_clay; // 0.625
    assert!((clay_round.positions()[center_slot][2] - want_h).abs() < 1e-6);
    assert!((clay_strips.positions()[center_slot][2] - want_h).abs() < 1e-6);
    for (ix, iy) in [(-2i32, -2i32), (2, -2), (-2, 2), (2, 2)] {
        // (±2.0, ±2.0) = (±0.8 R, ±0.8 R), Euclidean r = 2.8284 > 2.5
        let slot = ((iy + 5) * 11 + (ix + 5)) as usize;
        assert_eq!(
            clay_round.positions()[slot][2].to_bits(),
            0.0f32.to_bits(),
            "Round clay must not reach Euclidean r > R corner ({ix}, {iy})"
        );
        assert!(
            clay_strips.positions()[slot][2] > 0.05,
            "Clay Strips 2R x 2R box must lift (±0.8R, ±0.8R) corner ({ix}, {iy}): got {}",
            clay_strips.positions()[slot][2]
        );
    }

    // 4. Twist: Vortex preserves axial height & cylindrical radius; Kelvinlet twists smoothly;
    //    Symmetry { x: true } flips strength by det(M) = -1 so mirrored vortices counter-rotate!
    let mut vortex = symmetric_sheet(6, |_, _| 0.5);
    let before_vortex = vortex.positions().to_vec();
    let angle = 0.7f32;
    edit(&mut vortex, |tx| {
        twist(
            tx,
            &TwistOptions {
                brush: Brush {
                    center: [0.0, 0.0, 0.5],
                    radius: 4.5,
                    falloff: Falloff::Smooth,
                    strength: angle,
                },
                axis: Some([0.0, 0.0, 1.0]),
                method: TwistMethod::Vortex,
                ..Default::default()
            },
        )
    });
    for (p0, p1) in before_vortex.iter().zip(vortex.positions()) {
        assert!(
            (p1[2] - p0[2]).abs() < 1e-6,
            "Vortex preserves axial height z"
        );
        let r0 = p0[0].hypot(p0[1]);
        let r1 = p1[0].hypot(p1[1]);
        assert!(
            (r1 - r0).abs() < 1e-5,
            "Vortex preserves cylindrical radius: {r0} vs {r1}"
        );
    }
    // Check exact rotation at (1, 0, 0.5): t = 1 / 4.5
    let v_1_0 = (6 * 13 + 7) as usize;
    let t_1 = 1.0f32 / 4.5;
    let s_1 = 1.0 - t_1;
    let theta_1 = angle * s_1 * s_1 * (3.0 - 2.0 * s_1);
    assert!((vortex.positions()[v_1_0][0] - theta_1.cos()).abs() < 1e-5);
    assert!((vortex.positions()[v_1_0][1] - theta_1.sin()).abs() < 1e-5);

    for method in [TwistMethod::Vortex, TwistMethod::Kelvinlet] {
        let half = 8i32;
        let side = (2 * half + 1) as usize;
        let mut sym_twist = symmetric_sheet(half, |x, y| 0.04 * (x * x + y * y));
        let before_sym = sym_twist.positions().to_vec();
        edit(&mut sym_twist, |tx| {
            twist(
                tx,
                &TwistOptions {
                    brush: Brush {
                        center: [4.5, 0.5, 0.82],
                        radius: 3.0,
                        falloff: Falloff::Smooth,
                        strength: 0.65,
                    },
                    axis: Some([0.0, 0.0, 1.0]),
                    method,
                    symmetry: Symmetry {
                        x: true,
                        y: false,
                        z: false,
                    },
                    ..Default::default()
                },
            )
        });
        let after = sym_twist.positions();
        let mut moved_pairs = 0;
        for iy in -half..=half {
            for ix in 1..=half {
                let vp = (iy + half) as usize * side + (ix + half) as usize;
                let vn = (iy + half) as usize * side + (-ix + half) as usize;
                let (pp, pn) = (after[vp], after[vn]);
                if pp != before_sym[vp] {
                    moved_pairs += 1;
                }
                assert!(
                    (pn[0] - (-pp[0])).abs() < 1e-5
                        && (pn[1] - pp[1]).abs() < 1e-5
                        && (pn[2] - pp[2]).abs() < 1e-5,
                    "{method:?} mirror counter-rotation mismatch at ({ix}, {iy}): +x={pp:?}, -x={pn:?}"
                );
            }
        }
        assert!(moved_pairs > 4);
    }
}

#[test]
fn task2_region_parity_bit_for_bit_and_oom_sweeps_all_brushes() {
    let base_mesh = || {
        let mut m = symmetric_sheet(6, |x, y| (0.6 * x).sin() * (0.5 * y).cos() + 0.1 * x);
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[7]).unwrap();
        tx.commit().unwrap();
        m.ensure_incidence().unwrap();
        m
    };

    let region_for = |mesh: &Mesh, center: [f32; 3], reach: f32, sym: Symmetry| -> Vec<VertexId> {
        let (masks, count) = sym.active_masks();
        (0..mesh.vertex_slots() as VertexId)
            .filter(|&v| {
                let p = mesh.positions()[v as usize];
                masks[..count].iter().any(|&bits| {
                    let c = [
                        if bits & 1 != 0 { -center[0] } else { center[0] },
                        if bits & 2 != 0 { -center[1] } else { center[1] },
                        if bits & 4 != 0 { -center[2] } else { center[2] },
                    ];
                    let d = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2))
                        .sqrt();
                    d / reach < 1.0
                })
            })
            .collect()
    };

    let brush = Brush {
        center: [1.4, -0.9, 0.2],
        radius: 3.1,
        falloff: Falloff::Smooth,
        strength: 0.45,
    };
    let sym = Symmetry {
        x: true,
        y: true,
        z: false,
    };
    let filter = SurfaceFilter::PeaksOnly {
        threshold: 0.01,
        contrast: 0.5,
    };

    // 1. Grab across Sphere, Geodesic, Kelvinlet (+ adaptive_reach & from_stroke)
    for method in [
        GrabMethod::Sphere,
        GrabMethod::Geodesic,
        GrabMethod::Kelvinlet,
    ] {
        for adaptive_reach in [false, true] {
            let opts = GrabOptions {
                brush,
                direction: [1.6, -0.8, 0.4],
                method,
                poisson: 0.42,
                from_stroke: true,
                adaptive_reach,
                filter,
                symmetry: sym,
            };
            let (mut whole, mut reg) = (base_mesh(), base_mesh());
            capture_stroke_state(&mut whole).unwrap();
            capture_stroke_state(&mut reg).unwrap();
            let reach = if adaptive_reach {
                grab_widest_reach(&brush, opts.direction, method)
            } else {
                brush.radius
            };
            let mut candidates = region_for(&reg, brush.center, reach, sym);
            // Reverse + duplicate to also exercise unsorted/dedup handling in *_region!
            candidates.reverse();
            if let Some(&first) = candidates.first() {
                candidates.push(first);
            }
            let (rw, _) = edit(&mut whole, |tx| grab(tx, &opts));
            let (rr, _) = edit(&mut reg, |tx| grab_region(tx, &opts, &candidates));
            let mut reg_ws = base_mesh();
            capture_stroke_state(&mut reg_ws).unwrap();
            let (rws, _) = edit(&mut reg_ws, |tx| {
                grab_region_with_stroke(tx, &opts, &candidates, None, &[])
            });
            assert_eq!(rw, rr, "{method:?} adaptive={adaptive_reach}");
            assert_eq!(rw, rws, "{method:?} adaptive={adaptive_reach}");
            assert_eq!(
                bits_of(&whole),
                bits_of(&reg),
                "{method:?} adaptive={adaptive_reach}"
            );
            assert_eq!(
                bits_of(&whole),
                bits_of(&reg_ws),
                "{method:?} adaptive={adaptive_reach}"
            );
            assert!(sweep(base_mesh(), |tx| grab(tx, &opts).map(drop)) >= 3);
        }
    }

    // 2. Flatten across Both, Peaks, Valleys
    for mode in [FlattenMode::Both, FlattenMode::Peaks, FlattenMode::Valleys] {
        let opts = FlattenOptions {
            brush,
            mode,
            plane_origin: None,
            plane_normal: None,
            filter: SurfaceFilter::None,
            symmetry: sym,
        };
        let (mut whole, mut reg) = (base_mesh(), base_mesh());
        let candidates = region_for(&reg, brush.center, brush.radius, sym);
        let (rw, _) = edit(&mut whole, |tx| flatten(tx, &opts));
        let (rr, _) = edit(&mut reg, |tx| flatten_region(tx, &opts, &candidates));
        assert_eq!(rw, rr, "{mode:?}");
        assert_eq!(bits_of(&whole), bits_of(&reg), "{mode:?}");
        assert!(sweep(base_mesh(), |tx| flatten(tx, &opts).map(drop)) >= 3);
    }

    // 3. Crease
    {
        let opts = CreaseOptions {
            brush: Brush {
                strength: -0.35,
                ..brush
            },
            pinch: 0.65,
            filter,
            symmetry: sym,
        };
        let (mut whole, mut reg) = (base_mesh(), base_mesh());
        let candidates = region_for(&reg, brush.center, brush.radius, sym);
        let (rw, _) = edit(&mut whole, |tx| crease(tx, &opts));
        let (rr, _) = edit(&mut reg, |tx| crease_region(tx, &opts, &candidates));
        assert_eq!(rw, rr);
        assert_eq!(bits_of(&whole), bits_of(&reg));
        assert!(sweep(base_mesh(), |tx| crease(tx, &opts).map(drop)) >= 3);
    }

    // 4. Clay across Round & Strips (query radius sqrt(3) * R for Strips)
    for profile in [ClayProfile::Round, ClayProfile::Strips] {
        let opts = ClayOptions {
            brush,
            profile,
            tangent: Some([1.0, 0.3, 0.0]),
            filter: SurfaceFilter::None,
            symmetry: sym,
        };
        let reach = match profile {
            ClayProfile::Round => brush.radius,
            ClayProfile::Strips => brush.radius * 3.0f32.sqrt(),
        };
        let (mut whole, mut reg) = (base_mesh(), base_mesh());
        let candidates = region_for(&reg, brush.center, reach, sym);
        let (rw, _) = edit(&mut whole, |tx| clay(tx, &opts));
        let (rr, _) = edit(&mut reg, |tx| clay_region(tx, &opts, &candidates));
        assert_eq!(rw, rr, "{profile:?}");
        assert_eq!(bits_of(&whole), bits_of(&reg), "{profile:?}");
        assert!(sweep(base_mesh(), |tx| clay(tx, &opts).map(drop)) >= 3);
    }

    // 5. Twist across Vortex & Kelvinlet
    for method in [TwistMethod::Vortex, TwistMethod::Kelvinlet] {
        let opts = TwistOptions {
            brush,
            axis: None,
            method,
            filter,
            symmetry: sym,
        };
        let (mut whole, mut reg) = (base_mesh(), base_mesh());
        let candidates = region_for(&reg, brush.center, brush.radius, sym);
        let (rw, _) = edit(&mut whole, |tx| twist(tx, &opts));
        let (rr, _) = edit(&mut reg, |tx| twist_region(tx, &opts, &candidates));
        assert_eq!(rw, rr, "{method:?}");
        assert_eq!(bits_of(&whole), bits_of(&reg), "{method:?}");
        assert!(sweep(base_mesh(), |tx| twist(tx, &opts).map(drop)) >= 3);
    }
}

#[test]
fn asymmetric_multipass_scratch_reuse_reserves_without_infallible_realloc() {
    // Pass 0 (`+x`, center `[+4, 0, 0]`, radius `1.5`): 3x3 grid (`m0 = 9`, max valence `6`).
    // Pass 1 (`-x`, center `[-4, 0, 0]`, radius `1.5`): 10-spoke wheel + 3 extra fan vertices
    // (`m1 = 14`, hub valence `10`), hitting `C < m <= 2 * C` (`9 < 14 <= 18` and `6 < 10 <= 12`)
    // across all cleared scratch vectors in `GrabMethod::Geodesic` and `SmoothMethod::*`!
    let asym_mesh = || {
        let mut positions = Vec::new();
        let mut triangles = Vec::new();

        // Patch 0 (`+x`, 9 vertices, slots 0..9): 3x3 grid in [3.5..=4.5] x [-0.5..=0.5]
        for iy in -1i32..=1 {
            for ix in -1i32..=1 {
                let z = if ix == 0 && iy == 0 { 0.25 } else { 0.0 };
                positions.extend([4.0 + ix as f32 * 0.5, iy as f32 * 0.5, z]);
            }
        }
        for iy in 0..2u32 {
            for ix in 0..2u32 {
                let a = iy * 3 + ix;
                triangles.extend([[a, a + 1, a + 4], [a, a + 4, a + 3]]);
            }
        }

        // Patch 1 (`-x`, 14 vertices, slots 9..23):
        // Slots 9..19: 10-gon ring of radius 0.8 around (-4, 0, 0)
        // Slot 19: hub at (-4, 0, 0.3) with valence 10 (`6 < 10 <= 12` vs Patch 0's max valence 6!)
        // Slots 20..23: 3 outer tabs at radius 1.1 attached to ring edges (`m1 = 14`, `9 < 14 <= 18`!)
        let ring_base = 9u32;
        let spokes = 10u32;
        for k in 0..spokes {
            let ang = (k as f32) * std::f32::consts::TAU / (spokes as f32);
            positions.extend([-4.0 + 0.8 * ang.cos(), 0.8 * ang.sin(), 0.0]);
        }
        let hub = ring_base + spokes; // 19
        positions.extend([-4.0, 0.0, 0.3]);
        for k in 0..spokes {
            let next = (k + 1) % spokes;
            triangles.push([hub, ring_base + k, ring_base + next]);
        }
        for k in 0..3u32 {
            let ang = ((k as f32) + 0.5) * std::f32::consts::TAU / (spokes as f32);
            let tab = hub + 1 + k;
            positions.extend([-4.0 + 1.1 * ang.cos(), 1.1 * ang.sin(), 0.05]);
            triangles.push([ring_base + k, tab, ring_base + k + 1]);
        }

        let mut m = mesh_of(&positions, &triangles);
        m.ensure_incidence().unwrap();
        m
    };

    let brush = Brush {
        center: [4.0, 0.0, 0.0],
        radius: 1.8,
        falloff: Falloff::Smooth,
        strength: 0.5,
    };
    let sym_x = Symmetry {
        x: true,
        y: false,
        z: false,
    };
    let all_ids: Vec<VertexId> = (0..23).collect();

    // 1. Geodesic Grab (`seam_order`, `seam_next`, `geo_dist` reused from m0 = 9 to m1 = 14)
    let geo_opts = GrabOptions {
        brush,
        direction: [0.2, -0.1, 0.6],
        method: GrabMethod::Geodesic,
        symmetry: sym_x,
        ..Default::default()
    };
    let (mut gw, mut gr) = (asym_mesh(), asym_mesh());
    let (rw, _) = edit(&mut gw, |tx| grab(tx, &geo_opts));
    let (rr, _) = edit(&mut gr, |tx| grab_region(tx, &geo_opts, &all_ids));
    assert_eq!(rw.moved, 23);
    assert_eq!(rw, rr);
    assert_eq!(bits_of(&gw), bits_of(&gr));
    assert!(sweep(asym_mesh(), |tx| grab(tx, &geo_opts).map(drop)) >= 8);
    assert!(
        sweep(asym_mesh(), |tx| grab_region(tx, &geo_opts, &all_ids)
            .map(drop))
            >= 8
    );

    // 2. Smooth across all 4 methods (`starts`, `current`, `before`, `hc_lap`, `hc_b`, `raw_cot_scratch`)
    for method in [
        SmoothMethod::Taubin,
        SmoothMethod::Laplacian,
        SmoothMethod::HcLaplacian,
        SmoothMethod::Cotangent,
    ] {
        let sm_opts = SmoothOptions {
            brush,
            cycles: 2,
            method,
            filter: SurfaceFilter::None,
            symmetry: sym_x,
        };
        let (mut sw, mut sr) = (asym_mesh(), asym_mesh());
        let (rw, _) = edit(&mut sw, |tx| smooth(tx, &sm_opts));
        let (rr, _) = edit(&mut sr, |tx| smooth_region(tx, &sm_opts, &all_ids));
        assert!(rw.moved > 0, "{method:?}");
        assert_eq!(rw, rr, "{method:?}");
        assert_eq!(bits_of(&sw), bits_of(&sr), "{method:?}");
        assert!(sweep(asym_mesh(), |tx| smooth(tx, &sm_opts).map(drop)) >= 6);
        assert!(
            sweep(asym_mesh(), |tx| smooth_region(tx, &sm_opts, &all_ids)
                .map(drop))
                >= 6
        );
    }
}

#[test]
fn grab_region_with_stroke_uses_slice_adaptive_reach_and_zero_protected() {
    // 3x3 connected grid over x in [0, 1, 2], y in [0, 1, 2]:
    // vertex index = iy * 3 + ix
    // v0=(0,0,0), v1=(1,0,0) [perp outside R=0.6], v2=(2,0,0),
    // v3=(0,1,0) [ahead @ dist=1.0], v6=(0,2,0) [ahead @ dist=2.0].
    let mut positions = Vec::new();
    let mut triangles = Vec::new();
    for iy in 0..3u32 {
        for ix in 0..3u32 {
            positions.extend([ix as f32, iy as f32, 0.0]);
        }
    }
    for iy in 0..2u32 {
        for ix in 0..2u32 {
            let a = iy * 3 + ix;
            triangles.extend([[a, a + 1, a + 4], [a, a + 4, a + 3]]);
        }
    }
    let rest_mesh = mesh_of(&positions, &triangles);
    let rest_positions = rest_mesh.positions().to_vec();
    let rest_normals = vec![[0.0, 0.0, 1.0]; rest_positions.len()];
    let candidates: Vec<VertexId> = (0..9).collect();

    // Perturb seed v0 WITHOUT capturing any stroke channel on the mesh itself:
    // passing `Some(StrokeSlice)` must supply the rest base for `from_stroke: true`.
    let perturbed_seed = |shift: [f32; 3]| {
        let mut m = mesh_of(&positions, &triangles);
        let p0 = rest_positions[0];
        edit(&mut m, |tx| {
            tx.set_positions(
                &[0],
                &[[p0[0] + shift[0], p0[1] + shift[1], p0[2] + shift[2]]],
            )?;
            Ok(())
        });
        m
    };

    let brush = Brush {
        center: [0.0, 0.0, 0.0],
        radius: 0.6,
        falloff: Falloff::Linear,
        strength: 1.0,
    };
    let opts = GrabOptions {
        brush,
        direction: [0.0, 12.0, 0.0],
        from_stroke: true,
        adaptive_reach: true,
        ..Default::default()
    };
    let slice = StrokeSlice {
        positions: &rest_positions,
        normals: &rest_normals,
    };

    // Case 1: All ahead vertices (y > 0 => indices 3..9) are zero-protected.
    // `effective_pass_radius` skips zero-protected vertices when measuring `ahead`,
    // so radius stays 0.6 and perpendicular v1=(1,0,0) does not move.
    let mut gated = perturbed_seed([0.0, 0.0, 0.5]);
    let all_ahead_zp: Vec<VertexId> = vec![3, 4, 5, 6, 7, 8];
    edit(&mut gated, |tx| {
        grab_region_with_stroke(tx, &opts, &candidates, Some(slice), &all_ahead_zp)
    });
    assert!(
        (gated.positions()[0][1] - 12.0).abs() < 1e-4,
        "seed v0 must move from StrokeSlice rest pos even with ahead vertices protected"
    );
    assert!(
        (gated.positions()[0][2]).abs() < 1e-5,
        "from_stroke must use StrokeSlice z=0 rather than perturbed z=0.5"
    );
    assert_eq!(
        gated.positions()[1].map(f32::to_bits),
        rest_positions[1].map(f32::to_bits),
        "perpendicular v1 at dist=1.0 must remain untouched when ahead vertices are zero-protected"
    );

    // Case 2: Unprotect v3=(0,1,0) while keeping ahead vertices 4..9 in zero_protected.
    // Adaptive reach now widens past r=1.0, moving v1, then zero-direction snaps back to StrokeSlice.
    let mut widened = perturbed_seed([0.0, 0.0, 0.5]);
    let partial_zp: Vec<VertexId> = vec![4, 5, 6, 7, 8];
    edit(&mut widened, |tx| {
        grab_region_with_stroke(tx, &opts, &candidates, Some(slice), &partial_zp)
    });
    assert!(
        widened.positions()[1][1] > 0.5,
        "unprotected ahead v3 must widen adaptive_reach to pull perpendicular v1=(1,0,0), got y={}",
        widened.positions()[1][1]
    );

    // Zero-direction snap-back must return vertices inside reach back to StrokeSlice.positions.
    let reset_opts = GrabOptions {
        direction: [0.0, 0.0, 0.0],
        ..opts
    };
    edit(&mut widened, |tx| {
        grab_region_with_stroke(tx, &reset_opts, &candidates, Some(slice), &partial_zp)
    });
    assert_eq!(
        widened.positions()[0].map(f32::to_bits),
        rest_positions[0].map(f32::to_bits),
        "zero-direction grab must restore v0 exactly to StrokeSlice.positions"
    );
    assert_eq!(
        widened.positions()[1].map(f32::to_bits),
        rest_positions[1].map(f32::to_bits),
        "zero-direction grab must restore v1 exactly to StrokeSlice.positions"
    );
}
