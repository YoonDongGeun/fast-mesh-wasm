//! Integration tests for continuity-selectable hole filling (`Flat`,
//! `TangentG1`, `CurvatureG2`), island-in-hole multi-loop patching, partial arc
//! bridging, UV + undo/redo preservation, and 200-vertex G2 performance.

use std::time::Instant;

use fast_mesh_ops::OpsError;
use fast_mesh_ops::boundary_loops;
use fast_mesh_ops::repair::{
    AnalyzeOptions, BridgeStats, FillContinuity, FillHolesOptions, FillStats, PartialBridgeOptions,
    analyze, bridge_partial_arcs, fill_all_holes, fill_hole,
};
use fast_mesh_store::{History, Mesh, Transaction, VertexId};

fn edit<R>(mesh: &mut Mesh, f: impl FnOnce(&mut Transaction) -> fast_mesh_ops::Result<R>) -> R {
    let mut tx = mesh.begin_edit();
    let out = f(&mut tx).unwrap();
    tx.commit().unwrap();
    out
}

fn grid_without_mapped(
    n: u32,
    skip_cell: impl Fn(u32, u32) -> bool,
    map_pos: impl Fn(f32, f32) -> [f32; 3],
) -> Mesh {
    let stride = n + 1;
    let mut positions = Vec::with_capacity((stride * stride * 3) as usize);
    for y in 0..=n {
        for x in 0..=n {
            positions.extend_from_slice(&map_pos(x as f32, y as f32));
        }
    }
    let mut indices = Vec::new();
    for y in 0..n {
        for x in 0..n {
            if skip_cell(x, y) {
                continue;
            }
            let v00 = y * stride + x;
            let v10 = v00 + 1;
            let v01 = v00 + stride;
            let v11 = v01 + 1;
            indices.extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// Generates a multi-ring annulus around a circular inner hole of `n_hole` vertices
/// at radius `r_inner`, lifted onto `(x, y) -> map_pos(x, y)`.
fn circular_hole_mesh(
    n_hole: usize,
    rings: usize,
    r_inner: f64,
    dr: f64,
    map_pos: impl Fn(f64, f64) -> [f32; 3],
) -> Mesh {
    let mut positions = Vec::with_capacity((rings + 1) * n_hole * 3);
    for ring in 0..=rings {
        let r = r_inner + dr * ring as f64;
        for i in 0..n_hole {
            let theta = std::f64::consts::TAU * i as f64 / n_hole as f64;
            positions.extend_from_slice(&map_pos(r * theta.cos(), r * theta.sin()));
        }
    }
    let mut indices = Vec::new();
    for ring in 0..rings {
        let a_base = (ring * n_hole) as u32;
        let b_base = ((ring + 1) * n_hole) as u32;
        for i in 0..n_hole {
            let j = (i + 1) % n_hole;
            let a0 = a_base + i as u32;
            let a1 = a_base + j as u32;
            let b0 = b_base + i as u32;
            let b1 = b_base + j as u32;
            indices.extend_from_slice(&[a0, a1, b1, a0, b1, b0]);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

#[test]
fn flat_plane_hole_is_exact_for_all_continuities() {
    let plane_z = |x: f32, y: f32| 0.25 * x - 0.15 * y + 1.0;
    for continuity in [
        FillContinuity::Flat,
        FillContinuity::TangentG1,
        FillContinuity::CurvatureG2,
    ] {
        let mut mesh = grid_without_mapped(
            8,
            |x, y| (2..6).contains(&x) && (2..6).contains(&y),
            |x, y| [x, y, plane_z(x, y)],
        );
        let loops = boundary_loops(&mesh).unwrap();
        let hole_idx = loops.iter().position(|l| l.vertices.len() == 16).unwrap();

        let stats = edit(&mut mesh, |tx| {
            fill_hole(
                tx,
                hole_idx,
                &FillHolesOptions::builder()
                    .continuity(continuity)
                    .refine_patch(true)
                    .smooth_iterations(2)
                    .build(),
            )
        });
        assert!(stats.converged);
        assert!(
            stats.max_dihedral_angle_deg < 1e-3,
            "planar dihedral angle not zero for {continuity:?}: {}",
            stats.max_dihedral_angle_deg
        );
        for v in 0..mesh.vertex_slots() as VertexId {
            if let Some(p) = mesh.position(v) {
                assert!(
                    (p[2] - plane_z(p[0], p[1])).abs() < 1e-4,
                    "off-plane vertex {p:?} for {continuity:?}"
                );
            }
        }
        let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
        assert_eq!(a.non_manifold_edges, 0);
        assert_eq!(a.orientation_conflicts, 0);
        assert_eq!(boundary_loops(&mesh).unwrap().len(), 1);
    }
}

#[test]
fn sphere_hole_g2_matches_curvature_and_smooths_seam() {
    let radius = 5.5f32;
    let map_sphere = |x: f32, y: f32| -> [f32; 3] {
        let dx = x - 4.0;
        let dy = y - 4.0;
        [x, y, (radius * radius - dx * dx - dy * dy).sqrt()]
    };
    let build = || {
        grid_without_mapped(
            8,
            |x, y| (2..6).contains(&x) && (2..6).contains(&y),
            map_sphere,
        )
    };

    let mut mesh_flat = build();
    let hole_idx = boundary_loops(&mesh_flat)
        .unwrap()
        .iter()
        .position(|l| l.vertices.len() == 16)
        .unwrap();

    let s_flat = edit(&mut mesh_flat, |tx| {
        fill_hole(
            tx,
            hole_idx,
            &FillHolesOptions::builder()
                .continuity(FillContinuity::Flat)
                .refine_patch(false)
                .build(),
        )
    });

    let mut mesh_g1 = build();
    let s_g1 = edit(&mut mesh_g1, |tx| {
        fill_hole(
            tx,
            hole_idx,
            &FillHolesOptions::builder()
                .continuity(FillContinuity::TangentG1)
                .refine_patch(true)
                .build(),
        )
    });

    let mut mesh_g2 = build();
    let initial_verts = mesh_g2.vertex_slots();
    let s_g2 = edit(&mut mesh_g2, |tx| {
        fill_hole(
            tx,
            hole_idx,
            &FillHolesOptions::builder()
                .continuity(FillContinuity::CurvatureG2)
                .refine_patch(true)
                .smooth_iterations(2)
                .build(),
        )
    });

    assert!(s_g2.converged);
    assert!(s_g2.filled_vertices > 0);
    assert!(
        s_g2.max_dihedral_angle_deg < s_flat.max_dihedral_angle_deg * 0.65,
        "G2 seam dihedral {} should be far below Flat {}",
        s_g2.max_dihedral_angle_deg,
        s_flat.max_dihedral_angle_deg
    );
    assert!(
        s_g2.max_dihedral_angle_deg < s_g1.max_dihedral_angle_deg * 0.65,
        "G2 seam dihedral {} should be below G1 {}",
        s_g2.max_dihedral_angle_deg,
        s_g1.max_dihedral_angle_deg
    );

    // Every G2 interior vertex conforms tightly to sphere radius.
    let mut max_radial_err = 0.0f32;
    for vi in initial_verts..mesh_g2.vertex_slots() {
        let p = mesh_g2.position(vi as VertexId).unwrap();
        let r = ((p[0] - 4.0).powi(2) + (p[1] - 4.0).powi(2) + p[2] * p[2]).sqrt();
        max_radial_err = max_radial_err.max((r - radius).abs());
    }
    assert!(
        max_radial_err < 0.085,
        "sphere G2 max radial error too high: {max_radial_err}"
    );

    let a = analyze(&mesh_g2, &AnalyzeOptions::default()).unwrap();
    assert_eq!(a.non_manifold_edges, 0);
    assert_eq!(a.orientation_conflicts, 0);
}

#[test]
fn cylinder_hole_g2_matches_cylindrical_radius() {
    let radius = 6.0f32;
    let mut mesh = grid_without_mapped(
        8,
        |x, y| (2..6).contains(&x) && (2..6).contains(&y),
        |x, y| {
            let dx = x - 4.0;
            [x, y, (radius * radius - dx * dx).sqrt()]
        },
    );
    let initial_verts = mesh.vertex_slots();
    let hole_idx = boundary_loops(&mesh)
        .unwrap()
        .iter()
        .position(|l| l.vertices.len() == 16)
        .unwrap();

    let stats: FillStats = edit(&mut mesh, |tx| {
        fill_hole(
            tx,
            hole_idx,
            &FillHolesOptions::builder()
                .continuity(FillContinuity::CurvatureG2)
                .smooth_iterations(2)
                .build(),
        )
    });
    assert!(stats.converged && stats.filled_vertices > 0);
    for vi in initial_verts..mesh.vertex_slots() {
        let p = mesh.position(vi as VertexId).unwrap();
        let r = ((p[0] - 4.0).powi(2) + p[2] * p[2]).sqrt();
        assert!(
            (r - radius).abs() < 0.06,
            "cylinder G2 radial error too large at {p:?}: r={r}"
        );
    }
    let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
    assert_eq!(a.non_manifold_edges, 0);
    assert_eq!(a.orientation_conflicts, 0);
}

#[test]
fn saddle_hole_split_via_partial_bridge_and_fill_no_folds() {
    // 10x10 hyperbolic saddle grid with a large 6x6 central hole.
    let mut mesh = grid_without_mapped(
        10,
        |x, y| (2..8).contains(&x) && (2..8).contains(&y),
        |x, y| {
            let (dx, dy) = (x - 5.0, y - 5.0);
            [x, y, 0.12 * (dx * dx - dy * dy)]
        },
    );
    let initial_verts = mesh.vertex_slots();
    let loops = boundary_loops(&mesh).unwrap();
    assert_eq!(loops.len(), 2);
    let hole_idx = loops.iter().position(|l| l.vertices.len() == 24).unwrap();
    assert_eq!(loops[hole_idx].vertices.len(), 24);
    let (mut hole_z_min, mut hole_z_max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &loops[hole_idx].vertices {
        let z = mesh.position(v).unwrap()[2];
        hole_z_min = hole_z_min.min(z);
        hole_z_max = hole_z_max.max(z);
    }

    // Split the 24-vertex saddle hole across opposite 3-vertex arcs into 2 loops.
    let bridge_stats: BridgeStats = edit(&mut mesh, |tx| {
        bridge_partial_arcs(
            tx,
            &PartialBridgeOptions::builder(hole_idx, (2, 4), hole_idx, (14, 16))
                .subdivisions(Some(2))
                .build(),
        )
    });
    assert_eq!(bridge_stats.resulting_loops, 3);
    assert_eq!(bridge_stats.vertices_added, 5 * 2);
    assert_eq!(boundary_loops(&mesh).unwrap().len(), 3);

    // Fill the two smaller saddle sub-holes smoothly without folds or non-manifold edges.
    let filled = edit(&mut mesh, |tx| {
        fill_all_holes(
            tx,
            &FillHolesOptions::builder()
                .continuity(FillContinuity::TangentG1)
                .refine_patch(true)
                .max_hole_vertices(Some(20))
                .build(),
        )
    });
    assert_eq!(filled.len(), 2);
    let after_loops = boundary_loops(&mesh).unwrap();
    assert_eq!(after_loops.len(), 1);
    let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
    assert_eq!(a.non_manifold_edges, 0);
    assert_eq!(a.orientation_conflicts, 0);
    // All newly created bridge and G1 patch vertices respect the discrete maximum principle.
    assert!(mesh.vertex_slots() > initial_verts);
    for vi in initial_verts..mesh.vertex_slots() {
        if let Some(p) = mesh.position(vi as VertexId) {
            assert!(
                p[2] >= hole_z_min - 1e-4 && p[2] <= hole_z_max + 1e-4,
                "saddle fold/overshoot at {p:?}: expected within [{hole_z_min}, {hole_z_max}]"
            );
        }
    }
}

#[test]
fn island_in_hole_planar_and_spherical_multi_loop_patching() {
    // 1. Planar moat around a floating 2x2 center island.
    let mut flat_moat = grid_without_mapped(
        8,
        |x, y| {
            (1..7).contains(&x)
                && (1..7).contains(&y)
                && !((3..5).contains(&x) && (3..5).contains(&y))
        },
        |x, y| [x, y, 0.0],
    );
    let stats = edit(&mut flat_moat, |tx| {
        fill_all_holes(
            tx,
            &FillHolesOptions::builder()
                .continuity(FillContinuity::CurvatureG2)
                .refine_patch(true)
                .build(),
        )
    });
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].island_loops_merged, 1);
    assert_eq!(boundary_loops(&flat_moat).unwrap().len(), 1);
    let a = analyze(&flat_moat, &AnalyzeOptions::default()).unwrap();
    assert_eq!(a.non_manifold_edges, 0);
    assert_eq!(a.orientation_conflicts, 0);

    // 2. Curved spherical dome moat around a floating polar island cap at R = 10.
    let radius = 10.0f32;
    let mut sphere_moat = grid_without_mapped(
        8,
        |x, y| {
            (1..7).contains(&x)
                && (1..7).contains(&y)
                && !((3..5).contains(&x) && (3..5).contains(&y))
        },
        |x, y| {
            let (dx, dy) = (x - 4.0, y - 4.0);
            [x, y, (radius * radius - dx * dx - dy * dy).sqrt()]
        },
    );
    let init_verts = sphere_moat.vertex_slots();
    let sstats = edit(&mut sphere_moat, |tx| {
        fill_all_holes(
            tx,
            &FillHolesOptions::builder()
                .continuity(FillContinuity::CurvatureG2)
                .refine_patch(true)
                .smooth_iterations(2)
                .build(),
        )
    });
    assert_eq!(sstats.len(), 1);
    assert_eq!(sstats[0].island_loops_merged, 1);
    assert!(sstats[0].converged);
    for vi in init_verts..sphere_moat.vertex_slots() {
        let p = sphere_moat.position(vi as VertexId).unwrap();
        let r = ((p[0] - 4.0).powi(2) + (p[1] - 4.0).powi(2) + p[2] * p[2]).sqrt();
        assert!(
            (r - radius).abs() < 0.08,
            "spherical island patch off radius: r={r} at {p:?}"
        );
    }
    let sa = analyze(&sphere_moat, &AnalyzeOptions::default()).unwrap();
    assert_eq!(sa.non_manifold_edges, 0);
    assert_eq!(sa.orientation_conflicts, 0);
}

#[test]
fn partial_bridge_uv_interpolation_twist_and_undo_redo() {
    let pos = vec![
        0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0,
        0.0, 0.0, 3.0, 0.0, 1.0, 3.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 0.0, 1.0, 4.0, 0.0, 2.0, 4.0,
        0.0,
    ];
    let tris = vec![
        0u32, 1, 4, 0, 4, 3, 1, 2, 5, 1, 5, 4, 6, 7, 10, 6, 10, 9, 7, 8, 11, 7, 11, 10,
    ];
    let mut mesh = Mesh::from_buffers(&pos, &tris).unwrap();
    let uvs: Vec<f32> = (0..12)
        .flat_map(|i| [pos[3 * i] * 0.5, pos[3 * i + 1] * 0.25])
        .collect();
    mesh.add_uvs(&uvs).unwrap();

    let orig_verts = mesh.vertex_count();
    let orig_tris = mesh.triangle_count();
    let loops = boundary_loops(&mesh).unwrap();
    assert_eq!(loops.len(), 2);

    let find_idx =
        |loop_verts: &[VertexId], v: VertexId| loop_verts.iter().position(|&x| x == v).unwrap();
    let l0 = loops.iter().position(|l| l.vertices.contains(&4)).unwrap();
    let l1 = 1 - l0;
    let a_s = find_idx(&loops[l0].vertices, 5);
    let a_e = find_idx(&loops[l0].vertices, 3);
    let b_s = find_idx(&loops[l1].vertices, 6);
    let b_e = find_idx(&loops[l1].vertices, 8);
    let len1 = loops[l1].vertices.len();

    let opts = PartialBridgeOptions::builder(
        l0,
        (a_s, a_e),
        l1,
        ((b_s + len1 - 1) % len1, (b_e + len1 - 1) % len1),
    )
    .twist_offset(1)
    .subdivisions(Some(3))
    .build();

    let mut history = History::new(1 << 24);
    let mut tx = mesh.begin_edit();
    let stats = bridge_partial_arcs(&mut tx, &opts).unwrap();
    let patch = tx.commit().unwrap();
    history.push(&mut mesh, patch).unwrap();

    assert_eq!(stats.resulting_loops, 1);
    assert_eq!(stats.vertices_added, 5 * 3);
    assert_eq!(stats.triangles_added, 4 * 7);
    let uv_after = mesh.uvs().unwrap();
    for vi in orig_verts..mesh.vertex_slots() {
        let p = mesh.position(vi as VertexId).unwrap();
        assert!((uv_after[2 * vi] - p[0] * 0.5).abs() < 1e-5);
        assert!((uv_after[2 * vi + 1] - p[1] * 0.25).abs() < 1e-5);
    }

    assert!(history.undo(&mut mesh).unwrap().is_some());
    assert_eq!(mesh.vertex_count(), orig_verts);
    assert_eq!(mesh.triangle_count(), orig_tris);
    assert_eq!(&mesh.uvs().unwrap()[..uvs.len()], uvs.as_slice());

    assert!(history.redo(&mut mesh).unwrap().is_some());
    assert_eq!(mesh.triangle_count(), orig_tris + stats.triangles_added);
    let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
    assert_eq!(a.non_manifold_edges, 0);
}

#[test]
fn invalid_inputs_reject_without_mutation() {
    let mut mesh = grid_without_mapped(
        6,
        |x, y| (2..4).contains(&x) && (2..4).contains(&y),
        |x, y| [x, y, 0.0],
    );
    let orig_verts = mesh.vertex_count();
    let orig_tris = mesh.triangle_count();

    let mut tx = mesh.begin_edit();
    assert!(matches!(
        fill_hole(&mut tx, 99, &FillHolesOptions::default()),
        Err(OpsError::InvalidInput(_))
    ));
    assert!(matches!(
        fill_hole(
            &mut tx,
            0,
            &FillHolesOptions::builder()
                .max_hole_vertices(Some(2))
                .build()
        ),
        Err(OpsError::InvalidInput(_))
    ));
    assert!(matches!(
        bridge_partial_arcs(
            &mut tx,
            &PartialBridgeOptions::builder(0, (0, 2), 0, (1, 3)).build()
        ),
        Err(OpsError::InvalidInput(_))
    ));
    drop(tx);

    assert_eq!(mesh.vertex_count(), orig_verts);
    assert_eq!(mesh.triangle_count(), orig_tris);
}

#[test]
fn two_hundred_vertex_hole_g2_quality_and_timing() {
    let radius = 30.0f64;
    let r_inner = 4.0f64;
    let mut mesh = circular_hole_mesh(200, 3, r_inner, 0.25, |x, y| {
        let z = (radius * radius - x * x - y * y).sqrt();
        [x as f32, y as f32, z as f32]
    });
    let loops = boundary_loops(&mesh).unwrap();
    let hole_idx = loops.iter().position(|l| l.vertices.len() == 200).unwrap();
    let init_verts = mesh.vertex_slots();

    let start = Instant::now();
    let stats = edit(&mut mesh, |tx| {
        fill_hole(
            tx,
            hole_idx,
            &FillHolesOptions::builder()
                .continuity(FillContinuity::CurvatureG2)
                .refine_patch(true)
                .smooth_iterations(1)
                .build(),
        )
    });
    let elapsed = start.elapsed();

    assert!(stats.converged);
    assert!(stats.filled_vertices > 0);
    assert!(stats.filled_triangles > 200);
    // Check timing margin (note unoptimized debug builds include bounds checks;
    // release/Wasm must run well under 80ms).
    let budget_ms: u128 = if cfg!(debug_assertions) { 1200 } else { 80 };
    assert!(
        elapsed.as_millis() < budget_ms,
        "200-vertex G2 hole fill took {}ms (budget {}ms, release target 80ms)",
        elapsed.as_millis(),
        budget_ms
    );
    for vi in init_verts..mesh.vertex_slots() {
        let p = mesh.position(vi as VertexId).unwrap();
        let r = ((p[0] as f64).powi(2) + (p[1] as f64).powi(2) + (p[2] as f64).powi(2)).sqrt();
        assert!((r - radius).abs() < 0.2, "radial err on 200-vert cap: {r}");
    }
    let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
    assert_eq!(a.non_manifold_edges, 0);
    assert_eq!(a.orientation_conflicts, 0);
}
