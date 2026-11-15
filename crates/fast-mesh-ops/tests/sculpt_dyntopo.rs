use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_ops::sculpt::{
    Brush, DEFAULT_DYNTOPO_MAX_RATIO, DEFAULT_DYNTOPO_MIN_RATIO, DyntopoMode, DyntopoOptions,
    DyntopoPatch, Falloff, GrabMethod, GrabOptions, InflateMethod, InflateOptions,
    STROKE_NORMAL_CHANNEL, STROKE_POS_CHANNEL, apply_dyntopo, apply_dyntopo_in_roi,
    apply_dyntopo_protected, apply_dyntopo_with_bvh, capture_stroke_state, grab, inflate,
};
use fast_mesh_ops::{AnalyzeOptions, OpsError, analyze};
use fast_mesh_store::{ChannelDesc, Mesh, UV_CHANNEL, VertexId};

fn flat_grid(n: u32, spacing: f32) -> Mesh {
    let half = (n as f32 * spacing) * 0.5;
    let side = n + 1;
    let mut positions = Vec::with_capacity((side * side * 3) as usize);
    for y in 0..=n {
        for x in 0..=n {
            positions.extend([x as f32 * spacing - half, y as f32 * spacing - half, 0.0]);
        }
    }
    let mut indices = Vec::with_capacity((n * n * 6) as usize);
    for y in 0..n {
        for x in 0..n {
            let a = y * side + x;
            let b = a + 1;
            let c = a + side + 1;
            let d = a + side;
            indices.extend([a, b, c, a, c, d]);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn uv_sphere(stacks: u32, slices: u32, r: f32) -> Mesh {
    use std::f64::consts::PI;
    let mut positions = vec![0.0f32, 0.0, r];
    for i in 1..stacks {
        let theta = PI * f64::from(i) / f64::from(stacks);
        for j in 0..slices {
            let phi = 2.0 * PI * f64::from(j) / f64::from(slices);
            let rs = f64::from(r) * theta.sin();
            let rc = f64::from(r) * theta.cos();
            positions.extend([(rs * phi.cos()) as f32, (rs * phi.sin()) as f32, rc as f32]);
        }
    }
    positions.extend([0.0, 0.0, -r]);

    let ring = |i: u32, j: u32| 1 + (i - 1) * slices + (j % slices);
    let south = 1 + (stacks - 1) * slices;
    let mut indices = Vec::new();
    for j in 0..slices {
        indices.extend([0, ring(1, j), ring(1, j + 1)]);
        for i in 1..(stacks - 1) {
            let a = ring(i, j);
            let b = ring(i + 1, j);
            let c = ring(i + 1, j + 1);
            let d = ring(i, j + 1);
            indices.extend([a, b, c, a, c, d]);
        }
        indices.extend([south, ring(stacks - 1, j + 1), ring(stacks - 1, j)]);
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f64 {
    let dx = f64::from(a[0]) - f64::from(b[0]);
    let dy = f64::from(a[1]) - f64::from(b[1]);
    let dz = f64::from(a[2]) - f64::from(b[2]);
    dx * dx + dy * dy + dz * dz
}

fn assert_manifold_clean(mesh: &Mesh, allow_boundary_loops: bool) {
    let report = analyze(mesh, &AnalyzeOptions::default()).unwrap();
    assert_eq!(report.non_manifold_edges, 0, "non-manifold edges found");
    assert!(
        report.non_manifold_vertices.is_empty(),
        "non-manifold vertices: {:?}",
        report.non_manifold_vertices
    );
    assert!(
        report.pinch_vertices.is_empty(),
        "pinch vertices: {:?}",
        report.pinch_vertices
    );
    assert_eq!(report.duplicate, 0, "duplicate triangles created");
    assert_eq!(report.degenerate, 0, "degenerate triangles created");
    assert_eq!(
        report.orientation_conflicts, 0,
        "orientation conflicts created"
    );
    if !allow_boundary_loops {
        assert!(report.loops.is_empty(), "unexpected holes opened");
    }
}

fn max_edge_inside_sphere(mesh: &Mesh, center: [f32; 3], radius: f32) -> f64 {
    let r2 = f64::from(radius) * f64::from(radius);
    let pos = mesh.positions();
    let mut best = 0.0f64;
    for t in 0..mesh.triangle_slots() {
        if let Some([a, b, c]) = mesh.triangle(t as u32) {
            let pts = [pos[a as usize], pos[b as usize], pos[c as usize]];
            if pts.iter().all(|&p| dist2(p, center) <= r2) {
                for (u, v) in [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])] {
                    best = best.max(dist2(u, v).sqrt());
                }
            }
        }
    }
    best
}

fn count_interior_edges_shorter_than(
    mesh: &Mesh,
    center: [f32; 3],
    radius: f32,
    thresh: f64,
) -> usize {
    let r2 = f64::from(radius) * f64::from(radius);
    let thresh2 = thresh * thresh;
    let pos = mesh.positions();
    let mut edges = Vec::new();
    for t in 0..mesh.triangle_slots() {
        if let Some([a, b, c]) = mesh.triangle(t as u32) {
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let (eu, ev) = if u < v { (u, v) } else { (v, u) };
                if dist2(pos[eu as usize], center) <= r2 && dist2(pos[ev as usize], center) <= r2 {
                    edges.push((eu, ev));
                }
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    edges
        .into_iter()
        .filter(|&(u, v)| dist2(pos[u as usize], pos[v as usize]) < thresh2)
        .count()
}

#[test]
fn defaults_and_invalid_inputs_preserve_mesh() {
    let opts = DyntopoOptions::default();
    assert_eq!(opts.min_detail_ratio, DEFAULT_DYNTOPO_MIN_RATIO);
    assert_eq!(opts.max_detail_ratio, DEFAULT_DYNTOPO_MAX_RATIO);
    assert!((opts.min_detail_ratio - 0.4).abs() < 1e-6);
    assert!((opts.max_detail_ratio - 1.33).abs() < 1e-6);

    let mut mesh = flat_grid(4, 1.0);
    let orig_verts = mesh.vertex_count();
    let orig_tris = mesh.triangle_count();

    let bad_cases = [
        ([f32::NAN, 0.0, 0.0], 1.0, opts),
        ([0.0, 0.0, 0.0], 0.0, opts),
        ([0.0, 0.0, 0.0], -1.0, opts),
        ([0.0, 0.0, 0.0], f32::INFINITY, opts),
        (
            [0.0, 0.0, 0.0],
            1.0,
            DyntopoOptions::new(DyntopoMode::SplitCollapse, 0.0),
        ),
        (
            [0.0, 0.0, 0.0],
            1.0,
            DyntopoOptions::new(DyntopoMode::SplitCollapse, -0.2),
        ),
        (
            [0.0, 0.0, 0.0],
            1.0,
            DyntopoOptions::new(DyntopoMode::SplitCollapse, 0.5).with_ratios(-0.1, 1.3),
        ),
        (
            [0.0, 0.0, 0.0],
            1.0,
            DyntopoOptions::new(DyntopoMode::SplitCollapse, 0.5).with_ratios(1.5, 1.3),
        ),
    ];

    for (c, r, o) in bad_cases {
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            apply_dyntopo(&mut tx, c, r, &o),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(tx.commit().unwrap().is_empty());
    }

    // Dead pinned vertex rejected.
    {
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            apply_dyntopo_protected(&mut tx, [0.0, 0.0, 0.0], 1.0, &opts, &[9999]),
            Err(OpsError::InvalidInput(_))
        ));
    }
    // Dead ROI triangle rejected.
    {
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            apply_dyntopo_in_roi(&mut tx, [0.0, 0.0, 0.0], 1.0, &opts, &[9999], &[]),
            Err(OpsError::InvalidInput(_))
        ));
    }
    assert_eq!(mesh.vertex_count(), orig_verts);
    assert_eq!(mesh.triangle_count(), orig_tris);
}

#[test]
fn split_only_and_localization_leave_outside_roi_bit_identical() {
    let mut mesh = flat_grid(10, 1.0); // [-5, 5]^2
    let center = [0.0, 0.0, 0.0];
    let radius = 1.6f32;
    let r2 = f64::from(radius) * f64::from(radius);

    let orig_slots = mesh.vertex_slots();
    let outside_verts: Vec<(VertexId, [u32; 3])> = (0..orig_slots as VertexId)
        .filter_map(|v| {
            let p = mesh.positions()[v as usize];
            (dist2(p, center) > r2).then_some((v, p.map(f32::to_bits)))
        })
        .collect();
    let far_tris: Vec<(u32, [VertexId; 3])> = (0..mesh.triangle_slots() as u32)
        .filter_map(|t| {
            let corners = mesh.triangle(t)?;
            let all_far = corners.iter().all(|&v| {
                dist2(mesh.positions()[v as usize], center) > (radius + 1.5).powi(2) as f64
            });
            all_far.then_some((t, corners))
        })
        .collect();

    // First make some edges short too; SplitOnly must NOT collapse them.
    let opts = DyntopoOptions::new(DyntopoMode::SplitOnly, 0.45).with_equalize_valence(true);
    let report = {
        let mut tx = mesh.begin_edit();
        let r = apply_dyntopo(&mut tx, center, radius, &opts).unwrap();
        tx.commit().unwrap();
        r
    };
    assert!(report.splits > 0);
    assert_eq!(report.collapses, 0);
    assert!(report.removed_vertices.is_empty());
    assert!(max_edge_inside_sphere(&mesh, center, radius) <= 0.45 * 1.33 + 1e-4);

    // Every outside vertex keeps exact bit pattern and aliveness.
    for (v, bits) in outside_verts {
        assert!(mesh.is_vertex_alive(v));
        assert_eq!(mesh.positions()[v as usize].map(f32::to_bits), bits);
    }
    // Every triangle completely clear of the sphere stays unmodified.
    for (t, corners) in far_tris {
        assert_eq!(mesh.triangle(t), Some(corners));
    }
    assert_manifold_clean(&mesh, true);

    // Run SplitOnly on an ultra-dense mesh (edge = 0.25) with detail_size = 2.0 -> 0 splits, 0 collapses.
    let mut dense = flat_grid(8, 0.25);
    let pre_v = dense.vertex_count();
    let pre_t = dense.triangle_count();
    let opts_big = DyntopoOptions::new(DyntopoMode::SplitOnly, 2.0);
    let r = {
        let mut tx = dense.begin_edit();
        let r = apply_dyntopo(&mut tx, [0.0, 0.0, 0.0], 0.8, &opts_big).unwrap();
        tx.commit().unwrap();
        r
    };
    assert_eq!(r.collapses, 0);
    assert_eq!(r.splits, 0);
    assert_eq!(dense.vertex_count(), pre_v);
    assert_eq!(dense.triangle_count(), pre_t);
}

#[test]
fn short_edge_collapse_clears_sliver_clusters_without_non_manifold_or_boundary_tears() {
    let mut mesh = flat_grid(12, 0.25); // [-1.5, 1.5]^2, fine mesh spacing 0.25
    let pre_tris = mesh.triangle_count();

    // Target detail_size = 1.2 => d_min = 0.48 > diag 0.353, so interior short
    // edges inside radius 1.1 collapse heavily while boundary stays locked.
    let boundary_verts: Vec<(VertexId, [u32; 3])> = (0..mesh.vertex_slots() as VertexId)
        .filter_map(|v| {
            let p = mesh.positions()[v as usize];
            (p[0].abs() > 1.49 || p[1].abs() > 1.49).then_some((v, p.map(f32::to_bits)))
        })
        .collect();

    let opts = DyntopoOptions::new(DyntopoMode::SplitCollapse, 1.2)
        .with_equalize_valence(true)
        .with_smooth_tangent(true);
    let report = {
        let mut tx = mesh.begin_edit();
        let r = apply_dyntopo(&mut tx, [0.0, 0.0, 0.0], 1.1, &opts).unwrap();
        tx.commit().unwrap();
        r
    };

    assert!(report.collapses > 0);
    assert!(!report.removed_vertices.is_empty());
    assert!(!report.removed_triangles.is_empty());
    assert!(mesh.triangle_count() < pre_tris);
    assert_eq!(
        count_interior_edges_shorter_than(&mesh, [0.0, 0.0, 0.0], 0.8, 0.3),
        0,
        "clustered fine edges inside center ROI should collapse"
    );

    // Every boundary vertex remained alive and un-moved.
    for (v, bits) in boundary_verts {
        assert!(mesh.is_vertex_alive(v));
        assert_eq!(mesh.positions()[v as usize].map(f32::to_bits), bits);
    }
    assert_manifold_clean(&mesh, true);
}

#[test]
fn constant_detail_vs_brush_relative_modes() {
    let center = [0.0, 0.0, 0.0];
    // Identical 8x8 coarse grid spacing = 1.0:
    // BrushRelative with detail_size = 0.25 and radius = 2.0 gives target L = 0.5 (d_max = 0.665 -> splits!).
    // BrushRelative with detail_size = 0.25 and radius = 8.0 gives target L = 2.0 (d_max = 2.66 -> no split, collapses short!).
    let mut m_small_r = flat_grid(10, 1.0);
    let opts_rel = DyntopoOptions::new(DyntopoMode::BrushRelative, 0.25);
    let r_small = {
        let mut tx = m_small_r.begin_edit();
        let r = apply_dyntopo(&mut tx, center, 2.0, &opts_rel).unwrap();
        tx.commit().unwrap();
        r
    };
    assert!(
        r_small.splits > 0,
        "radius=2 * 0.25 = 0.5 splits coarse spacing=1"
    );

    let mut m_large_r = flat_grid(10, 1.0);
    let r_large = {
        let mut tx = m_large_r.begin_edit();
        let r = apply_dyntopo(&mut tx, center, 6.0, &opts_rel).unwrap();
        tx.commit().unwrap();
        r
    };
    assert_eq!(
        r_large.splits, 0,
        "radius=6 * 0.25 = 1.5 does not split 1.41 diagonals"
    );

    // ConstantDetail ignores radius scaling of target edge length.
    let mut m_const_r2 = flat_grid(10, 1.0);
    let mut m_const_r3 = flat_grid(10, 1.0);
    let opts_const = DyntopoOptions::new(DyntopoMode::ConstantDetail, 0.45);
    let rc2 = {
        let mut tx = m_const_r2.begin_edit();
        let r = apply_dyntopo(&mut tx, center, 2.0, &opts_const).unwrap();
        tx.commit().unwrap();
        r
    };
    let rc3 = {
        let mut tx = m_const_r3.begin_edit();
        let r = apply_dyntopo(&mut tx, center, 3.0, &opts_const).unwrap();
        tx.commit().unwrap();
        r
    };
    assert!(rc2.splits > 0 && rc3.splits > rc2.splits);
    assert!(max_edge_inside_sphere(&m_const_r2, center, 2.0) <= 0.45 * 1.33 + 1e-4);
    assert!(max_edge_inside_sphere(&m_const_r3, center, 3.0) <= 0.45 * 1.33 + 1e-4);
}

#[test]
fn link_condition_guards_tetrahedron_from_collapsing_to_double_face() {
    // Minimal closed tetrahedron with tiny edges: link condition passes
    // naively because N(a) ∩ N(b) == {c, d}, yet collapse would crush the
    // volume to duplicate back-to-back faces.
    let positions = vec![
        0.0f32, 0.0, 0.0, // 0
        0.05, 0.0, 0.0, // 1
        0.025, 0.05, 0.0, // 2
        0.025, 0.02, 0.05, // 3
    ];
    let indices = vec![
        0, 2, 1, // bottom
        0, 1, 3, // front
        1, 2, 3, // right
        2, 0, 3, // left
    ];
    let mut tet = Mesh::from_buffers(&positions, &indices).unwrap();
    assert_manifold_clean(&tet, false);

    let opts = DyntopoOptions::new(DyntopoMode::SplitCollapse, 1.0);
    let report = {
        let mut tx = tet.begin_edit();
        let r = apply_dyntopo(&mut tx, [0.025, 0.02, 0.02], 1.0, &opts).unwrap();
        tx.commit().unwrap();
        r
    };
    assert_eq!(report.collapses, 0);
    assert_eq!(tet.triangle_count(), 4);
    assert_manifold_clean(&tet, false);
}

#[test]
fn valence_equalization_flips_and_tangent_smooth_keep_manifold_on_closed_sphere() {
    let mut sphere = uv_sphere(12, 18, 2.0);
    // Target north pole region with fine SplitCollapse + valence flips + tangent relaxation.
    let opts = DyntopoOptions::new(DyntopoMode::SplitCollapse, 0.35)
        .with_equalize_valence(true)
        .with_smooth_tangent(true);
    let mut total_flips = 0usize;
    let mut total_smoothed = 0usize;

    // Two sequential dabs simulate interactive movement across surface curvature.
    for c in [[0.0, 0.0, 2.0], [0.4, 0.2, 1.95]] {
        let r = {
            let mut tx = sphere.begin_edit();
            let r = apply_dyntopo(&mut tx, c, 1.1, &opts).unwrap();
            tx.commit().unwrap();
            r
        };
        total_flips += r.flips;
        total_smoothed += r.smoothed;
        assert_manifold_clean(&sphere, false);
    }
    assert!(total_flips > 0, "expected valence/Delaunay flips on sphere");
    assert!(
        total_smoothed > 0,
        "expected tangential relaxation displacement"
    );
}

#[test]
fn pinned_vertices_stay_alive_and_stationary_across_collapse_flip_and_smooth() {
    let mut mesh = flat_grid(10, 0.25); // dense interior ready to collapse/smooth
    let center_v = 5 * 11 + 5; // vertex (0.0, 0.0, 0.0)
    let neighbor_v = center_v + 1;
    let pinned = [center_v, neighbor_v];
    let expected_bits = [
        mesh.positions()[center_v as usize].map(f32::to_bits),
        mesh.positions()[neighbor_v as usize].map(f32::to_bits),
    ];

    let opts = DyntopoOptions::new(DyntopoMode::SplitCollapse, 1.2)
        .with_equalize_valence(true)
        .with_smooth_tangent(true);
    let r = {
        let mut tx = mesh.begin_edit();
        let r = apply_dyntopo_protected(&mut tx, [0.0, 0.0, 0.0], 1.1, &opts, &pinned).unwrap();
        tx.commit().unwrap();
        r
    };
    assert!(r.collapses > 0);
    for (v, bits) in pinned.into_iter().zip(expected_bits) {
        assert!(mesh.is_vertex_alive(v), "pinned vertex {v} must survive");
        assert_eq!(mesh.positions()[v as usize].map(f32::to_bits), bits);
    }
    assert_manifold_clean(&mesh, true);
}

#[test]
fn linear_interpolation_preserves_uv_and_stroke_channels_across_splits_and_collapses() {
    let mut mesh = flat_grid(8, 0.5); // [-2, 2]^2
    // Add planar UVs `(u, v) = (x * 0.25 + 0.5, y * 0.25 + 0.5)`.
    let uvs: Vec<f32> = mesh
        .positions()
        .iter()
        .flat_map(|p| [p[0] * 0.25 + 0.5, p[1] * 0.25 + 0.5])
        .collect();
    mesh.add_channel_with_values(ChannelDesc::source(UV_CHANNEL, 2), &uvs)
        .unwrap();
    capture_stroke_state(&mut mesh).unwrap();

    // Also pull vertex 40 slightly in +Z before dyntopo so rest pos differs from current pos.
    {
        let mut tx = mesh.begin_edit();
        let p40 = tx.mesh().positions()[40];
        tx.set_positions(&[40], &[[p40[0], p40[1], 0.3]]).unwrap();
        tx.commit().unwrap();
    }

    // Step 1: fine SplitOnly on coarse grid creates new vertices via edge splits.
    let split_opts = DyntopoOptions::new(DyntopoMode::SplitOnly, 0.25).with_equalize_valence(true);
    let r_split = {
        let mut tx = mesh.begin_edit();
        let r = apply_dyntopo(&mut tx, [0.0, 0.0, 0.0], 1.2, &split_opts).unwrap();
        tx.commit().unwrap();
        r
    };
    assert!(r_split.splits > 0);

    // Step 2: coarser SplitCollapse triggers collapses inside center ROI.
    let col_opts = DyntopoOptions::new(DyntopoMode::SplitCollapse, 1.1).with_equalize_valence(true);
    let r_col = {
        let mut tx = mesh.begin_edit();
        let r = apply_dyntopo(&mut tx, [0.0, 0.0, 0.0], 1.0, &col_opts).unwrap();
        tx.commit().unwrap();
        r
    };
    assert!(r_col.collapses > 0);

    // Bound stroke channels and UV channel must stay intact and finite for all alive vertices.
    let stroke_pos = mesh
        .channel(STROKE_POS_CHANNEL)
        .expect("stroke pos channel preserved");
    let stroke_nrm = mesh
        .channel(STROKE_NORMAL_CHANNEL)
        .expect("stroke normal channel preserved");
    let uv = mesh.channel(UV_CHANNEL).expect("UV channel preserved");
    for v in 0..mesh.vertex_slots() {
        if !mesh.is_vertex_alive(v as VertexId) {
            continue;
        }
        let rp = &stroke_pos[v * 3..v * 3 + 3];
        let rn = &stroke_nrm[v * 3..v * 3 + 3];
        let uv_v = &uv[v * 2..v * 2 + 2];
        assert!(rp.iter().all(|x| x.is_finite()));
        assert!(rn.iter().all(|x| x.is_finite()));
        assert!(uv_v.iter().all(|x| x.is_finite()));
        // On a flat grid z_rest == 0 and linear UVs satisfy `u == x_rest * 0.25 + 0.5`.
        assert!(rp[2].abs() < 1e-5, "rest z interpolated flat");
        assert!((uv_v[0] - (rp[0] * 0.25 + 0.5)).abs() < 1e-4);
        assert!((uv_v[1] - (rp[1] * 0.25 + 0.5)).abs() < 1e-4);
    }

    // Grab(from_stroke: true) and Inflate(StrokeNormal) execute cleanly over dyntopo result.
    let brush = Brush {
        center: [0.0, 0.0, 0.0],
        radius: 1.0,
        strength: 0.4,
        falloff: Falloff::Smooth,
    };
    {
        let mut tx = mesh.begin_edit();
        let g = grab(
            &mut tx,
            &GrabOptions {
                brush,
                direction: [0.0, 0.0, 0.5],
                from_stroke: true,
                method: GrabMethod::Geodesic,
                ..GrabOptions::default()
            },
        )
        .unwrap();
        assert!(g.moved > 0);
        let inf = inflate(
            &mut tx,
            &InflateOptions {
                brush,
                method: InflateMethod::StrokeNormal,
                ..InflateOptions::default()
            },
        )
        .unwrap();
        assert!(inf.moved > 0);
        tx.commit().unwrap();
    }
    assert_manifold_clean(&mesh, true);
}

#[test]
fn bvh_roi_matches_whole_mesh_scan_bit_for_bit() {
    let mut scan_mesh = uv_sphere(12, 16, 2.0);
    let mut bvh_mesh = uv_sphere(12, 16, 2.0);
    bvh_mesh.ensure_incidence().unwrap();
    let bvh = Bvh::build(&bvh_mesh, &BuildOptions::default()).unwrap();

    let center = [0.0, 0.0, 2.0];
    let radius = 1.25;
    let opts = DyntopoOptions::new(DyntopoMode::SplitCollapse, 0.35)
        .with_equalize_valence(true)
        .with_smooth_tangent(true);

    let r_scan: DyntopoPatch = {
        let mut tx = scan_mesh.begin_edit();
        let r = apply_dyntopo(&mut tx, center, radius, &opts).unwrap();
        tx.commit().unwrap();
        r
    };
    let r_bvh: DyntopoPatch = {
        let mut tx = bvh_mesh.begin_edit();
        let r = apply_dyntopo_with_bvh(&mut tx, &bvh, center, radius, &opts, &[]).unwrap();
        tx.commit().unwrap();
        r
    };

    assert_eq!(r_scan.splits, r_bvh.splits);
    assert_eq!(r_scan.collapses, r_bvh.collapses);
    assert_eq!(r_scan.flips, r_bvh.flips);
    assert_eq!(r_scan.smoothed, r_bvh.smoothed);
    assert_eq!(scan_mesh.vertex_count(), bvh_mesh.vertex_count());
    assert_eq!(scan_mesh.triangle_count(), bvh_mesh.triangle_count());

    for v in 0..scan_mesh.vertex_slots() {
        assert_eq!(
            scan_mesh.is_vertex_alive(v as VertexId),
            bvh_mesh.is_vertex_alive(v as VertexId)
        );
        if scan_mesh.is_vertex_alive(v as VertexId) {
            assert_eq!(
                scan_mesh.positions()[v].map(f32::to_bits),
                bvh_mesh.positions()[v].map(f32::to_bits)
            );
        }
    }
}
