use std::time::Instant;

use fast_mesh::compute_vertex_normals;
use fast_mesh_scene::ops::sculpt::{
    Brush, DyntopoMode, DyntopoOptions, Falloff, GrabMethod, GrabOptions, InflateMethod,
    InflateOptions, Symmetry,
};
use fast_mesh_scene::ops::{AnalyzeOptions, analyze};
use fast_mesh_scene::{Dab, MeshOptions, Scene, SceneOptions, UpdateMode};
use fast_mesh_store::{ChannelDesc, Mesh, UV_CHANNEL, VertexId};

fn terrain_height(fx: f32, fy: f32) -> f32 {
    (fx * 0.12).cos() * (fy * 0.12).cos() * 1.8 + (fx * 0.65).sin() * (fy * 0.65).cos() * 0.45
}

fn terrain(n: u32) -> Mesh {
    let half = n as f32 * 0.5;
    let mut positions = Vec::with_capacity(((n + 1) * (n + 1) * 3) as usize);
    for y in 0..=n {
        for x in 0..=n {
            let fx = x as f32 - half;
            let fy = y as f32 - half;
            positions.extend([fx, fy, terrain_height(fx, fy)]);
        }
    }
    let mut indices = Vec::with_capacity((n * n * 6) as usize);
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

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

fn snapshot_mesh(mesh: &Mesh) -> (Vec<[u32; 3]>, Vec<[VertexId; 3]>, Vec<u32>) {
    let pos_bits: Vec<[u32; 3]> = mesh
        .positions()
        .iter()
        .enumerate()
        .filter(|(i, _)| mesh.is_vertex_alive(*i as VertexId))
        .map(|(_, p)| p.map(f32::to_bits))
        .collect();
    let tris: Vec<[VertexId; 3]> = (0..mesh.triangle_slots() as u32)
        .filter_map(|t| mesh.triangle(t))
        .collect();
    let uv_bits: Vec<u32> = mesh
        .channel(UV_CHANNEL)
        .unwrap_or(&[])
        .iter()
        .enumerate()
        .filter(|(i, _)| mesh.is_vertex_alive((i / 2) as VertexId))
        .map(|(_, v)| v.to_bits())
        .collect();
    (pos_bits, tris, uv_bits)
}

#[test]
fn scene_apply_dyntopo_refits_bvh_normals_and_compact_undo_redo() {
    let mut mesh = terrain(16); // [-8, 8]^2
    let uvs: Vec<f32> = mesh
        .positions()
        .iter()
        .flat_map(|p| [p[0] * 0.05 + 0.5, p[1] * 0.05 + 0.5])
        .collect();
    mesh.add_channel_with_values(ChannelDesc::source(UV_CHANNEL, 2), &uvs)
        .unwrap();
    let pre = snapshot_mesh(&mesh);

    let mut scene = Scene::new(SceneOptions::default());
    let id = scene
        .add_mesh(
            mesh,
            MeshOptions {
                mode: UpdateMode::Eager,
                ..MeshOptions::default()
            },
        )
        .unwrap();
    scene.set_normals(id, true).unwrap();

    let center = [0.0, 0.0, terrain_height(0.0, 0.0)];
    let opts = DyntopoOptions::new(DyntopoMode::SplitCollapse, 0.45)
        .with_equalize_valence(true)
        .with_smooth_tangent(true);

    let patch = scene.apply_dyntopo(id, center, 2.5, &opts).unwrap();
    assert!(patch.splits > 0);
    assert!(scene.is_current(id).unwrap());

    // Incremental vertex normals match fresh recomputation across modified ROI.
    let m = scene.mesh(id).unwrap();
    let flat_pos: Vec<f32> = m.positions().iter().flatten().copied().collect();
    let tri_indices: Vec<u32> = (0..m.triangle_slots() as u32)
        .filter_map(|t| m.triangle(t))
        .flatten()
        .collect();
    let expected_nrm = compute_vertex_normals(&flat_pos, &tri_indices).unwrap();
    let scene_nrm = scene.normals(id).unwrap().unwrap();
    for v in 0..m.vertex_slots() {
        if m.is_vertex_alive(v as VertexId) {
            let a = scene_nrm[v];
            let b = &expected_nrm[v * 3..v * 3 + 3];
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 1e-4, "normal mismatch at v={v}");
            }
        }
    }

    // Subsequent BVH query sees newly subdivided vertices inside sphere (midpoints
    // of edges that merely skim the sphere boundary can sit slightly outside radius).
    let positions_snapshot = scene.mesh(id).unwrap().positions().to_vec();
    let mut inside_new = 0usize;
    let queried = scene.vertices_in_sphere(id, center, 2.5).unwrap();
    for &new_v in &patch.new_vertices {
        let p = positions_snapshot[new_v as usize];
        let d2 = (p[0] - center[0]) * (p[0] - center[0])
            + (p[1] - center[1]) * (p[1] - center[1])
            + (p[2] - center[2]) * (p[2] - center[2]);
        if d2 <= 2.499 * 2.499 {
            inside_new += 1;
            assert!(queried.contains(&new_v));
        }
    }
    assert!(inside_new > 0);

    let post = snapshot_mesh(scene.mesh(id).unwrap());

    // Exact bit-for-bit Undo & Redo including UV channel.
    assert!(scene.undo().unwrap());
    assert_eq!(snapshot_mesh(scene.mesh(id).unwrap()), pre);

    assert!(scene.redo().unwrap());
    assert_eq!(snapshot_mesh(scene.mesh(id).unwrap()), post);
}

#[test]
fn scene_dyntopo_dab_with_symmetry_protection_and_stroke_snapshot() {
    let mesh = flat_grid(8, 1.0); // [-4, 4]^2 coarse grid
    let mut scene = Scene::new(SceneOptions::default());
    let id = scene.add_mesh(mesh, MeshOptions::default()).unwrap();

    // Protect center column x == 0 vertices at weight 0.0.
    let protected: Vec<VertexId> = (0..scene.mesh(id).unwrap().vertex_slots() as VertexId)
        .filter(|&v| scene.mesh(id).unwrap().positions()[v as usize][0].abs() < 1e-5)
        .collect();
    let weights = vec![0.0f32; protected.len()];
    let prot_bits: Vec<[u32; 3]> = protected
        .iter()
        .map(|&v| scene.mesh(id).unwrap().positions()[v as usize].map(f32::to_bits))
        .collect();

    // Begin an active stroke snapshot & combine symmetry + Grab(from_stroke: true) + dyntopo.
    scene.capture_stroke_state(id, false).unwrap();

    let sym_x = Symmetry {
        x: true,
        y: false,
        z: false,
    };
    let dy_opts = DyntopoOptions::new(DyntopoMode::BrushRelative, 0.3).with_equalize_valence(true);
    let dab = Dab::Inflate(InflateOptions {
        brush: Brush {
            center: [2.0, 0.0, 0.0],
            radius: 1.5,
            strength: 0.35,
            falloff: Falloff::Smooth,
        },
        method: InflateMethod::StrokeNormal,
        symmetry: sym_x,
        ..InflateOptions::default()
    });
    let (dy1, r1) = scene
        .dyntopo_dab_protected(id, &dab, &dy_opts, &protected, &weights)
        .unwrap();
    assert!(dy1.splits > 0);
    assert!(r1.moved > 0);

    // Follow up with a Grab(from_stroke: true) dab sliding across the subdivided dome.
    let grab_dab = Dab::Grab(GrabOptions {
        brush: Brush {
            center: [2.0, 0.0, 0.0],
            radius: 1.5,
            strength: 0.8,
            falloff: Falloff::Smooth,
        },
        direction: [0.0, 0.0, 0.9],
        from_stroke: true,
        method: GrabMethod::Geodesic,
        symmetry: sym_x,
        ..GrabOptions::default()
    });
    let (_dy2, r2) = scene
        .dyntopo_dab_protected(id, &grab_dab, &dy_opts, &protected, &weights)
        .unwrap();
    assert!(r2.moved > 0);

    // Verify protected vertices remained completely untouched & alive.
    for (&v, &bits) in protected.iter().zip(&prot_bits) {
        assert!(scene.mesh(id).unwrap().is_vertex_alive(v));
        assert_eq!(
            scene.mesh(id).unwrap().positions()[v as usize].map(f32::to_bits),
            bits
        );
    }

    // Verify clean 2-manifold topology.
    let report = analyze(scene.mesh(id).unwrap(), &AnalyzeOptions::default()).unwrap();
    assert_eq!(report.non_manifold_edges, 0);
    assert!(report.pinch_vertices.is_empty());
    assert_eq!(report.orientation_conflicts, 0);

    // One undo reverts the whole second `dyntopo_dab`, another reverts the first!
    assert!(scene.undo().unwrap());
    assert!(scene.undo().unwrap());
    assert_eq!(scene.mesh(id).unwrap().triangle_count(), 2 * 8 * 8);
}

#[test]
fn scene_apply_dyntopo_handles_manual_lazy_and_eager_modes() {
    for mode in [UpdateMode::Lazy, UpdateMode::Eager, UpdateMode::Manual] {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene
            .add_mesh(
                terrain(10),
                MeshOptions {
                    mode,
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        if mode == UpdateMode::Manual {
            // Make BVH stale deliberately before apply_dyntopo to verify safe fallback.
            scene
                .edit(id, |tx| tx.set_positions(&[0], &[[-5.0, -5.0, 0.1]]))
                .unwrap();
            assert!(!scene.is_current(id).unwrap());
        }
        let opts = DyntopoOptions::new(DyntopoMode::ConstantDetail, 0.4);
        let p = scene
            .apply_dyntopo(id, [0.0, 0.0, terrain_height(0.0, 0.0)], 2.0, &opts)
            .unwrap();
        assert!(p.splits > 0);
        if mode == UpdateMode::Eager {
            assert!(scene.is_current(id).unwrap());
        }
    }
}

#[test]
fn one_million_triangle_localized_dyntopo_stays_sub_8ms_and_compact() {
    // `terrain(708)` has 2 * 708 * 708 = 1,002,528 triangles (~502,681 vertices).
    let mesh = terrain(708);
    let initial_tris = mesh.triangle_count();
    assert!(initial_tris >= 1_000_000);

    // Tight 256 KB undo budget: a full-mesh clone (> 24 MB) would exceed it and be
    // immediately evicted (`can_undo() == false`), proving compact delta patches.
    let mut scene = Scene::new(SceneOptions {
        history_budget: 256 * 1024,
    });
    let id = scene
        .add_mesh(
            mesh,
            MeshOptions {
                mode: UpdateMode::Lazy,
                ..MeshOptions::default()
            },
        )
        .unwrap();
    scene.set_normals(id, true).unwrap();
    scene.ensure_incidence(id).unwrap();
    scene.update(id).unwrap();

    let radius = 3.5f32;
    let opts = DyntopoOptions::new(DyntopoMode::SplitCollapse, 0.55)
        .with_equalize_valence(true)
        .with_smooth_tangent(true);

    // Warm up one localized dab off-center so any initial lazy state is settled.
    let warmup_center = [-20.0, -20.0, terrain_height(-20.0, -20.0)];
    let warmup = scene
        .apply_dyntopo(id, warmup_center, radius, &opts)
        .unwrap();
    assert!(warmup.splits > 0 || warmup.flips > 0);

    // Measure 5 localized dabs moving along X across the 1M-triangle terrain.
    let mut timings_ms = Vec::new();
    for step in 0..5 {
        let fx = (step as f32) * 3.0;
        let fy = 4.0;
        let c = [fx, fy, terrain_height(fx, fy)];
        let t0 = Instant::now();
        let report = scene.apply_dyntopo(id, c, radius, &opts).unwrap();
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        timings_ms.push(dt_ms);
        assert!(
            report.splits > 0 || report.collapses > 0 || report.flips > 0 || report.smoothed > 0
        );
    }

    timings_ms.sort_by(f64::total_cmp);
    let median_ms = timings_ms[timings_ms.len() / 2];
    let max_ms = *timings_ms.last().unwrap();
    eprintln!(
        "1M-tri dyntopo dab timings (ms): sorted={timings_ms:?}, median={median_ms:.3}ms, max={max_ms:.3}ms"
    );

    #[cfg(not(debug_assertions))]
    {
        assert!(
            median_ms < 8.0 && max_ms < 8.0,
            "1M-tri dyntopo dab exceeded 8ms budget: median={median_ms:.3}ms, max={max_ms:.3}ms"
        );
    }

    // All 5 measured dabs + warmup must fit within 256 KB undo budget and revert cleanly!
    for _ in 0..6 {
        assert!(scene.undo().unwrap(), "undo step should not be evicted");
    }
    assert_eq!(scene.mesh(id).unwrap().triangle_count(), initial_tris);
}
