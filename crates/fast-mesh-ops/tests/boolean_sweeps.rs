//! Issue #165 regression sweeps for canonical boolean arrangement:
//! - 6-bit face-diagonal unit-cube sweeps (>=576 + 144 combinations)
//! - Curved manifold mesh matrix (spheres, heterogeneous spheres, torus vs sphere,
//!   axis-aligned & generic offsets, identical coincident spheres)
//! - 50k-triangle overlapping UV spheres
//! - Coplanar, nested, disjoint, and edge/vertex-touching manifold regressions

use std::f64::consts::PI;

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_ops::{
    AnalyzeOptions, BooleanOp, BooleanReport, analyze, apply_boolean, boolean_plan, measure,
};
use fast_mesh_query::Rigid;
use fast_mesh_store::Mesh;

/// Unit cube `[0, 1]^3` shifted by `at`, matching the exact 6-bit face-diagonal
/// convention of `masked_cube_obj` in `crates/fast-mesh-cli/tests/cli.rs`.
fn masked_cube(mask: u32, at: [f64; 3]) -> Mesh {
    let mut positions = Vec::with_capacity(24);
    for x in [0.0, 1.0] {
        for y in [0.0, 1.0] {
            for z in [0.0, 1.0] {
                positions.extend([(x + at[0]) as f32, (y + at[1]) as f32, (z + at[2]) as f32]);
            }
        }
    }
    const QUADS: [[u32; 4]; 6] = [
        [0, 1, 3, 2],
        [4, 6, 7, 5],
        [0, 4, 5, 1],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 5, 7, 3],
    ];
    let mut indices = Vec::with_capacity(36);
    for (i, [a, b, c, d]) in QUADS.into_iter().enumerate() {
        let triangles = if mask & (1 << i) != 0 {
            [[a, b, c], [a, c, d]]
        } else {
            [[b, c, d], [b, d, a]]
        };
        for t in triangles {
            indices.extend_from_slice(&t);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// Outward-wound closed UV sphere of radius `r` around the origin with `stacks`
/// polar bands and `slices` azimuth segments (`2 * (stacks - 1) * slices` tris).
fn sphere(r: f32, stacks: u32, slices: u32) -> Mesh {
    let mut positions = Vec::with_capacity(((1 + (stacks - 1) * slices + 1) * 3) as usize);
    positions.extend([0.0, 0.0, r]);
    for i in 1..stacks {
        let theta = PI * f64::from(i) / f64::from(stacks);
        let (rs, rc) = (f64::from(r) * theta.sin(), f64::from(r) * theta.cos());
        for j in 0..slices {
            let phi = 2.0 * PI * f64::from(j) / f64::from(slices);
            positions.extend([(rs * phi.cos()) as f32, (rs * phi.sin()) as f32, rc as f32]);
        }
    }
    let south = (positions.len() / 3) as u32;
    positions.extend([0.0, 0.0, -r]);

    let ring = |i: u32, j: u32| 1 + (i - 1) * slices + j % slices;
    let mut indices = Vec::with_capacity((2 * (stacks - 1) * slices * 3) as usize);
    for j in 0..slices {
        indices.extend([0, ring(1, j), ring(1, j + 1)]);
        for i in 1..stacks - 1 {
            let (a, b, c, d) = (
                ring(i, j),
                ring(i + 1, j),
                ring(i + 1, j + 1),
                ring(i, j + 1),
            );
            indices.extend([a, b, c, a, c, d]);
        }
        indices.extend([ring(stacks - 1, j), south, ring(stacks - 1, j + 1)]);
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// Outward-wound closed torus around the z-axis (`2 * n * m` triangles).
fn torus(major: f32, minor: f32, n: u32, m: u32) -> Mesh {
    let mut positions = Vec::with_capacity((n * m * 3) as usize);
    for i in 0..n {
        let phi = 2.0 * PI * f64::from(i) / f64::from(n);
        for j in 0..m {
            let theta = 2.0 * PI * f64::from(j) / f64::from(m);
            let w = f64::from(major) + f64::from(minor) * theta.cos();
            positions.extend([
                (w * phi.cos()) as f32,
                (w * phi.sin()) as f32,
                (f64::from(minor) * theta.sin()) as f32,
            ]);
        }
    }
    let at = |i: u32, j: u32| (i % n) * m + j % m;
    let mut indices = Vec::with_capacity((2 * n * m * 3) as usize);
    for i in 0..n {
        for j in 0..m {
            let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            indices.extend([a, b, c, a, c, d]);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// Rebuilds an identical fresh `Mesh` with the same vertex/triangle slots as `src`.
fn clone_mesh(src: &Mesh) -> Mesh {
    let positions: Vec<f32> = src.positions().iter().flatten().copied().collect();
    let indices: Vec<u32> = src.triangles().iter().flatten().copied().collect();
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// Plans `op` on `(a, bvh_a)` and `(b, bvh_b)` and commits it on a fresh copy of `a`.
fn run_boolean(
    a: &Mesh,
    bvh_a: &Bvh,
    b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    op: BooleanOp,
    ctx: &str,
) -> (Mesh, BooleanReport) {
    let plan = boolean_plan(a, bvh_a, b, bvh_b, xf, op)
        .unwrap_or_else(|e| panic!("{ctx} ({op:?}): boolean_plan failed: {e:?}"));
    let mut res = clone_mesh(a);
    let mut tx = res.begin_edit();
    let report = apply_boolean(&mut tx, &plan)
        .unwrap_or_else(|e| panic!("{ctx} ({op:?}): apply_boolean failed: {e:?}"));
    tx.commit()
        .unwrap_or_else(|e| panic!("{ctx} ({op:?}): commit failed: {e:?}"));
    assert_eq!(
        report.kept_triangles,
        res.triangle_count(),
        "{ctx} ({op:?}): kept_triangles mismatch"
    );
    (res, report)
}

/// Asserts strict 2-manifold closedness, outward orientation, and zero pinch vertices.
fn assert_strict_manifold(res: &Mesh, ctx: &str) -> f64 {
    let m = measure(res).unwrap_or_else(|e| panic!("{ctx}: measure failed: {e:?}"));
    let an = analyze(res, &AnalyzeOptions::default())
        .unwrap_or_else(|e| panic!("{ctx}: analyze failed: {e:?}"));
    assert_eq!(m.non_manifold_edges, 0, "{ctx}: non-manifold edges: {m:?}");
    assert_eq!(
        m.orientation_conflicts, 0,
        "{ctx}: orientation conflicts: {m:?}"
    );
    assert!(
        an.pinch_vertices.is_empty(),
        "{ctx}: pinch vertices: {:?}",
        an.pinch_vertices
    );
    if res.triangle_count() > 0 {
        assert!(m.closed, "{ctx}: open boolean output: {m:?}");
        assert!(
            m.signed_volume > 0.0,
            "{ctx}: non-positive signed volume {}: {m:?}",
            m.signed_volume
        );
        m.volume.unwrap()
    } else {
        assert!(m.closed, "{ctx}: empty mesh must report closed: {m:?}");
        0.0
    }
}

/// Analytic intersection lens volume of two spheres of radii `r1`, `r2` separated by `d`.
fn sphere_lens_volume(r1: f64, r2: f64, d: f64) -> f64 {
    if d >= r1 + r2 {
        return 0.0;
    }
    if d <= (r1 - r2).abs() {
        let r_min = r1.min(r2);
        return 4.0 / 3.0 * PI * r_min.powi(3);
    }
    PI * (r1 + r2 - d).powi(2) * (d * d + 2.0 * d * (r1 + r2) - 3.0 * (r1 - r2).powi(2))
        / (12.0 * d)
}

#[test]
fn cube_diagonal_and_offset_sweep_576() {
    let opts = BuildOptions::default();
    let cubes_a: Vec<(Mesh, Bvh)> = (0..64)
        .map(|mask| {
            let m = masked_cube(mask, [0.0; 3]);
            let vol = assert_strict_manifold(&m, &format!("input cube A mask={mask}"));
            assert!(
                (vol - 1.0).abs() < 1e-6,
                "input cube A mask={mask} volume={vol}"
            );
            let bvh = Bvh::build(&m, &opts).unwrap();
            (m, bvh)
        })
        .collect();
    let cubes_b: Vec<(Mesh, Bvh)> = (0..64)
        .map(|mask| {
            let m = masked_cube(mask, [0.0; 3]);
            let bvh = Bvh::build(&m, &opts).unwrap();
            (m, bvh)
        })
        .collect();

    let check_case = |mask_a: usize, mask_b: usize, offset: [f64; 3], op: BooleanOp| {
        let (ref a, ref bvh_a) = cubes_a[mask_a];
        let (ref b, ref bvh_b) = cubes_b[mask_b];
        let xf = Rigid {
            translation: offset,
            ..Rigid::IDENTITY
        };
        let overlap = (1.0 - offset[0].abs()).max(0.0)
            * (1.0 - offset[1].abs()).max(0.0)
            * (1.0 - offset[2].abs()).max(0.0);
        let expected_volume = match op {
            BooleanOp::Union => 2.0 - overlap,
            BooleanOp::Subtract => 1.0 - overlap,
            BooleanOp::Intersect => overlap,
        };
        let ctx = format!("cube sweep mask_a={mask_a} mask_b={mask_b} offset={offset:?} op={op:?}");
        let (res, _) = run_boolean(a, bvh_a, b, bvh_b, &xf, op, &ctx);
        let vol = assert_strict_manifold(&res, &ctx);
        if expected_volume > 0.0 {
            assert!(res.triangle_count() > 0, "{ctx}: unexpectedly empty");
        }
        assert!(
            (vol - expected_volume).abs() < 1e-5,
            "{ctx}: got volume {vol}, expected {expected_volume}"
        );
    };

    let primary_offsets = [[0.5, 0.25, 0.125], [0.37, 0.41, 0.53], [0.5, 0.0, 0.0]];
    let ops = [BooleanOp::Union, BooleanOp::Subtract, BooleanOp::Intersect];

    // Full 64 x 3 x 3 = 576 sweep varying both A's mask and B's mask via a
    // bijective affine permutation mod 64, plus 576 runs against mask_b = 0
    // (covering the exact Issue #165 reproducers such as mask_a = 0 and 36).
    let mut count_permuted = 0usize;
    let mut count_vs_zero = 0usize;
    for mask in 0..64usize {
        let mask_b = (mask * 29 + 17) & 63;
        for offset in primary_offsets {
            for op in ops {
                check_case(mask, mask_b, offset, op);
                count_permuted += 1;
                check_case(mask, 0, offset, op);
                count_vs_zero += 1;
            }
        }
    }
    assert_eq!(count_permuted, 576);
    assert_eq!(count_vs_zero, 576);

    // Explicit 12 x 4 x 3 = 144 sweep across boundary/complementary masks and [0.25, 0.25, 0.25].
    let mask_pairs: [(usize, usize); 12] = [
        (0, 0),
        (63, 63),
        (0, 63),
        (63, 0),
        (0b010101, 0b010101),
        (0b101010, 0b101010),
        (0b010101, 0b101010),
        (0b101010, 0b010101),
        (36, 0),
        (0, 36),
        (0b000111, 0b111000),
        (0b001100, 0b110011),
    ];
    let four_offsets = [
        [0.5, 0.25, 0.125],
        [0.37, 0.41, 0.53],
        [0.5, 0.0, 0.0],
        [0.25, 0.25, 0.25],
    ];
    let mut count_144 = 0usize;
    for (ma, mb) in mask_pairs {
        for offset in four_offsets {
            for op in ops {
                check_case(ma, mb, offset, op);
                count_144 += 1;
            }
        }
    }
    assert_eq!(count_144, 144);
}

#[test]
fn issue_165_curved_mesh_matrix() {
    let opts = BuildOptions::default();

    let s_1_32 = sphere(1.0, 32, 32);
    let bvh_s_1_32 = Bvh::build(&s_1_32, &opts).unwrap();
    let v_s_1_32 = assert_strict_manifold(&s_1_32, "sphere(1.0, 32, 32)");

    let s_09_39_43 = sphere(0.9, 39, 43);
    let bvh_s_09 = Bvh::build(&s_09_39_43, &opts).unwrap();
    let v_s_09 = assert_strict_manifold(&s_09_39_43, "sphere(0.9, 39, 43)");

    let tor = torus(1.0, 0.4, 48, 24);
    let bvh_tor = Bvh::build(&tor, &opts).unwrap();
    let v_tor = assert_strict_manifold(&tor, "torus(1.0, 0.4, 48, 24)");

    let check_overlap_pair = |name: &str,
                              a: &Mesh,
                              bvh_a: &Bvh,
                              v_a: f64,
                              b: &Mesh,
                              bvh_b: &Bvh,
                              v_b: f64,
                              offset: [f64; 3],
                              analytic_lens: Option<f64>| {
        let xf = Rigid {
            translation: offset,
            ..Rigid::IDENTITY
        };
        let (u_mesh, _) = run_boolean(a, bvh_a, b, bvh_b, &xf, BooleanOp::Union, name);
        let (s_mesh, _) = run_boolean(a, bvh_a, b, bvh_b, &xf, BooleanOp::Subtract, name);
        let (i_mesh, _) = run_boolean(a, bvh_a, b, bvh_b, &xf, BooleanOp::Intersect, name);

        assert!(u_mesh.triangle_count() > 0, "{name}: Union empty");
        assert!(s_mesh.triangle_count() > 0, "{name}: Subtract empty");
        assert!(i_mesh.triangle_count() > 0, "{name}: Intersect empty");

        let v_union = assert_strict_manifold(&u_mesh, &format!("{name} Union"));
        let v_sub = assert_strict_manifold(&s_mesh, &format!("{name} Subtract"));
        let v_inter = assert_strict_manifold(&i_mesh, &format!("{name} Intersect"));

        assert!(
            ((v_union + v_inter) - (v_a + v_b)).abs() < 1e-4,
            "{name}: V_union ({v_union}) + V_inter ({v_inter}) != V_a ({v_a}) + V_b ({v_b})"
        );
        assert!(
            ((v_sub + v_inter) - v_a).abs() < 1e-4,
            "{name}: V_sub ({v_sub}) + V_inter ({v_inter}) != V_a ({v_a})"
        );

        if let Some(lens) = analytic_lens {
            let rel_err = ((v_inter - lens) / lens).abs();
            assert!(
                rel_err < 0.015,
                "{name}: V_inter ({v_inter}) vs analytic lens ({lens}) rel_err={rel_err}"
            );
        }
    };

    let s_1_32_b = sphere(1.0, 32, 32);
    let bvh_s_1_32_b = Bvh::build(&s_1_32_b, &opts).unwrap();

    let d_generic = (0.31_f64.powi(2) + 0.17_f64.powi(2) + 0.23_f64.powi(2)).sqrt();

    // 1. sphere(1.0, 32, 32) vs sphere(1.0, 32, 32) at (0.31, 0.17, 0.23)
    check_overlap_pair(
        "pair1_equal_spheres_generic",
        &s_1_32,
        &bvh_s_1_32,
        v_s_1_32,
        &s_1_32_b,
        &bvh_s_1_32_b,
        v_s_1_32,
        [0.31, 0.17, 0.23],
        Some(sphere_lens_volume(1.0, 1.0, d_generic)),
    );

    // 2. sphere(1.0, 32, 32) vs sphere(0.9, 39, 43) at (0.31, 0.17, 0.23)
    check_overlap_pair(
        "pair2_hetero_spheres_generic",
        &s_1_32,
        &bvh_s_1_32,
        v_s_1_32,
        &s_09_39_43,
        &bvh_s_09,
        v_s_09,
        [0.31, 0.17, 0.23],
        Some(sphere_lens_volume(1.0, 0.9, d_generic)),
    );

    // 3. torus(1.0, 0.4, 48, 24) vs sphere(0.9, 39, 43) at (0.31, 0.17, 0.23)
    check_overlap_pair(
        "pair3_torus_vs_sphere",
        &tor,
        &bvh_tor,
        v_tor,
        &s_09_39_43,
        &bvh_s_09,
        v_s_09,
        [0.31, 0.17, 0.23],
        None,
    );

    // 4. sphere(1.0, 32, 32) vs sphere(1.0, 32, 32) at (0.31, 0.0, 0.0)
    check_overlap_pair(
        "pair4_equal_spheres_x_offset",
        &s_1_32,
        &bvh_s_1_32,
        v_s_1_32,
        &s_1_32_b,
        &bvh_s_1_32_b,
        v_s_1_32,
        [0.31, 0.0, 0.0],
        Some(sphere_lens_volume(1.0, 1.0, 0.31)),
    );

    // 5. Identical sphere(1.0, 32, 32) vs sphere(1.0, 32, 32) at (0.0, 0.0, 0.0)
    let (u_ident, _) = run_boolean(
        &s_1_32,
        &bvh_s_1_32,
        &s_1_32_b,
        &bvh_s_1_32_b,
        &Rigid::IDENTITY,
        BooleanOp::Union,
        "pair5_identical_spheres",
    );
    let v_u_ident = assert_strict_manifold(&u_ident, "pair5_identical_spheres Union");
    assert!(
        (v_u_ident - v_s_1_32).abs() < 1e-5,
        "identical Union volume {v_u_ident} != {v_s_1_32}"
    );

    let (i_ident, _) = run_boolean(
        &s_1_32,
        &bvh_s_1_32,
        &s_1_32_b,
        &bvh_s_1_32_b,
        &Rigid::IDENTITY,
        BooleanOp::Intersect,
        "pair5_identical_spheres",
    );
    let v_i_ident = assert_strict_manifold(&i_ident, "pair5_identical_spheres Intersect");
    assert!(
        (v_i_ident - v_s_1_32).abs() < 1e-5,
        "identical Intersect volume {v_i_ident} != {v_s_1_32}"
    );

    let (sub_ident, _) = run_boolean(
        &s_1_32,
        &bvh_s_1_32,
        &s_1_32_b,
        &bvh_s_1_32_b,
        &Rigid::IDENTITY,
        BooleanOp::Subtract,
        "pair5_identical_spheres",
    );
    assert_eq!(
        sub_ident.triangle_count(),
        0,
        "identical Subtract must yield 0 triangles"
    );
    assert_strict_manifold(&sub_ident, "pair5_identical_spheres Subtract");
}

#[test]
fn curved_50k_overlap() {
    let opts = BuildOptions::default();
    let a = sphere(1.0, 158, 158);
    let b = sphere(0.9, 165, 169);
    assert_eq!(a.triangle_count(), 49_612);
    assert_eq!(b.triangle_count(), 55_432);

    let bvh_a = Bvh::build(&a, &opts).unwrap();
    let bvh_b = Bvh::build(&b, &opts).unwrap();
    let v_a = assert_strict_manifold(&a, "50k input A");
    let v_b = assert_strict_manifold(&b, "50k input B");

    let xf = Rigid {
        translation: [0.31, 0.17, 0.23],
        ..Rigid::IDENTITY
    };

    let (u_mesh, _) = run_boolean(&a, &bvh_a, &b, &bvh_b, &xf, BooleanOp::Union, "50k");
    let (s_mesh, _) = run_boolean(&a, &bvh_a, &b, &bvh_b, &xf, BooleanOp::Subtract, "50k");
    let (i_mesh, _) = run_boolean(&a, &bvh_a, &b, &bvh_b, &xf, BooleanOp::Intersect, "50k");

    let v_union = assert_strict_manifold(&u_mesh, "50k Union");
    let v_sub = assert_strict_manifold(&s_mesh, "50k Subtract");
    let v_inter = assert_strict_manifold(&i_mesh, "50k Intersect");

    assert!(
        ((v_union + v_inter) - (v_a + v_b)).abs() < 1e-4,
        "50k conservation: V_union ({v_union}) + V_inter ({v_inter}) != V_a ({v_a}) + V_b ({v_b})"
    );
    assert!(
        ((v_sub + v_inter) - v_a).abs() < 1e-4,
        "50k conservation: V_sub ({v_sub}) + V_inter ({v_inter}) != V_a ({v_a})"
    );

    let d = (0.31_f64.powi(2) + 0.17_f64.powi(2) + 0.23_f64.powi(2)).sqrt();
    let lens = sphere_lens_volume(1.0, 0.9, d);
    assert!(
        ((v_inter - lens) / lens).abs() < 0.005,
        "50k V_inter ({v_inter}) vs analytic lens ({lens})"
    );
}

#[test]
fn coplanar_nested_disjoint_and_touch_regressions() {
    let opts = BuildOptions::default();
    let cube_a = masked_cube(0, [0.0; 3]);
    let bvh_ca = Bvh::build(&cube_a, &opts).unwrap();
    let cube_b = masked_cube(63, [0.0; 3]);
    let bvh_cb = Bvh::build(&cube_b, &opts).unwrap();

    // Stacked face-touching cubes ([0, 0, 1]), edge-touching ([1, 1, 0]), and vertex-touching ([1, 1, 1]).
    for (label, offset) in [
        ("stacked_face_touch", [0.0, 0.0, 1.0]),
        ("edge_touch", [1.0, 1.0, 0.0]),
        ("vertex_touch", [1.0, 1.0, 1.0]),
    ] {
        let xf = Rigid {
            translation: offset,
            ..Rigid::IDENTITY
        };
        let (u, _) = run_boolean(
            &cube_a,
            &bvh_ca,
            &cube_b,
            &bvh_cb,
            &xf,
            BooleanOp::Union,
            label,
        );
        let (s, _) = run_boolean(
            &cube_a,
            &bvh_ca,
            &cube_b,
            &bvh_cb,
            &xf,
            BooleanOp::Subtract,
            label,
        );
        let (i, _) = run_boolean(
            &cube_a,
            &bvh_ca,
            &cube_b,
            &bvh_cb,
            &xf,
            BooleanOp::Intersect,
            label,
        );

        let v_u = assert_strict_manifold(&u, &format!("{label} Union"));
        let v_s = assert_strict_manifold(&s, &format!("{label} Subtract"));
        assert_eq!(i.triangle_count(), 0, "{label} Intersect must be empty");
        assert_strict_manifold(&i, &format!("{label} Intersect"));
        assert!((v_u - 2.0).abs() < 1e-5, "{label} Union vol {v_u}");
        assert!((v_s - 1.0).abs() < 1e-5, "{label} Subtract vol {v_s}");
    }

    // Concentric nested spheres: outer R=1.0, inner R=0.5.
    let outer = sphere(1.0, 24, 24);
    let inner = sphere(0.5, 20, 20);
    let bvh_outer = Bvh::build(&outer, &opts).unwrap();
    let bvh_inner = Bvh::build(&inner, &opts).unwrap();
    let v_outer = assert_strict_manifold(&outer, "nested outer");
    let v_inner = assert_strict_manifold(&inner, "nested inner");

    let (u_nest, _) = run_boolean(
        &outer,
        &bvh_outer,
        &inner,
        &bvh_inner,
        &Rigid::IDENTITY,
        BooleanOp::Union,
        "nested",
    );
    let (s_nest, _) = run_boolean(
        &outer,
        &bvh_outer,
        &inner,
        &bvh_inner,
        &Rigid::IDENTITY,
        BooleanOp::Subtract,
        "nested",
    );
    let (i_nest, _) = run_boolean(
        &outer,
        &bvh_outer,
        &inner,
        &bvh_inner,
        &Rigid::IDENTITY,
        BooleanOp::Intersect,
        "nested",
    );
    assert!((assert_strict_manifold(&u_nest, "nested Union") - v_outer).abs() < 1e-5);
    assert!(
        (assert_strict_manifold(&s_nest, "nested Subtract") - (v_outer - v_inner)).abs() < 1e-5
    );
    assert!((assert_strict_manifold(&i_nest, "nested Intersect") - v_inner).abs() < 1e-5);

    // Disjoint spheres separated by 3.0 along +x.
    let xf_disjoint = Rigid {
        translation: [3.0, 0.0, 0.0],
        ..Rigid::IDENTITY
    };
    let (u_dis, _) = run_boolean(
        &outer,
        &bvh_outer,
        &inner,
        &bvh_inner,
        &xf_disjoint,
        BooleanOp::Union,
        "disjoint",
    );
    let (s_dis, _) = run_boolean(
        &outer,
        &bvh_outer,
        &inner,
        &bvh_inner,
        &xf_disjoint,
        BooleanOp::Subtract,
        "disjoint",
    );
    let (i_dis, _) = run_boolean(
        &outer,
        &bvh_outer,
        &inner,
        &bvh_inner,
        &xf_disjoint,
        BooleanOp::Intersect,
        "disjoint",
    );
    assert!((assert_strict_manifold(&u_dis, "disjoint Union") - (v_outer + v_inner)).abs() < 1e-5);
    assert!((assert_strict_manifold(&s_dis, "disjoint Subtract") - v_outer).abs() < 1e-5);
    assert_eq!(i_dis.triangle_count(), 0);
    assert_strict_manifold(&i_dis, "disjoint Intersect");
}
