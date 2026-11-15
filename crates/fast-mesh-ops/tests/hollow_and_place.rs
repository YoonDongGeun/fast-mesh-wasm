//! Integration tests for Issue #236: SDF inward hollowing, chamfered/tapered
//! drain holes + plug generation, and surface-aligned `place_mesh`.

use std::f64::consts::{FRAC_PI_2, PI};

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_ops::boolean::{
    BooleanMethod, DrainHoleOptions, DrainHoleSpec, HollowBackend, HollowOptions, PlaceMeshParams,
    PlaceMode, PlaceUpAxis, align_up_axis_rotation, compute_place_transform, hollow_mesh,
    place_mesh, punch_drain_hole, punch_drain_holes,
};
use fast_mesh_ops::{AnalyzeOptions, OpsError, analyze, measure};
use fast_mesh_store::{Mesh, Patch, TriangleId, VertexId};

/// Outward-wound closed axis-aligned box `[min[0]..max[0], min[1]..max[1], min[2]..max[2]]`.
fn box_mesh(min: [f64; 3], max: [f64; 3]) -> Mesh {
    let mut positions = Vec::with_capacity(24);
    for &x in &[min[0], max[0]] {
        for &y in &[min[1], max[1]] {
            for &z in &[min[2], max[2]] {
                positions.extend([x as f32, y as f32, z as f32]);
            }
        }
    }
    #[rustfmt::skip]
    let indices: Vec<u32> = vec![
        0, 1, 3, 0, 3, 2, // -x
        4, 6, 7, 4, 7, 5, // +x
        0, 4, 5, 0, 5, 1, // -y
        2, 3, 7, 2, 7, 6, // +y
        0, 2, 6, 0, 6, 4, // -z
        1, 5, 7, 1, 7, 3, // +z
    ];
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// Outward-wound closed subdivided box grid `[min..max]` with `nx * ny * nz` cells on faces.
fn subdivided_box_mesh(min: [f64; 3], max: [f64; 3], nx: usize, ny: usize, nz: usize) -> Mesh {
    let mut positions: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let mut add_quad_grid =
        |u_steps: usize, v_steps: usize, point_at: &dyn Fn(f64, f64) -> [f64; 3], flip: bool| {
            let base = (positions.len() / 3) as u32;
            for j in 0..=v_steps {
                let v = j as f64 / v_steps as f64;
                for i in 0..=u_steps {
                    let u = i as f64 / u_steps as f64;
                    let p = point_at(u, v);
                    positions.extend([p[0] as f32, p[1] as f32, p[2] as f32]);
                }
            }
            let stride = (u_steps + 1) as u32;
            for j in 0..v_steps as u32 {
                for i in 0..u_steps as u32 {
                    let a = base + j * stride + i;
                    let b = a + 1;
                    let c = a + stride + 1;
                    let d = a + stride;
                    if flip {
                        indices.extend([a, d, c, a, c, b]);
                    } else {
                        indices.extend([a, b, c, a, c, d]);
                    }
                }
            }
        };

    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    // +Z and -Z
    add_quad_grid(
        nx,
        ny,
        &|u, v| [lerp(min[0], max[0], u), lerp(min[1], max[1], v), max[2]],
        false,
    );
    add_quad_grid(
        nx,
        ny,
        &|u, v| [lerp(min[0], max[0], u), lerp(min[1], max[1], v), min[2]],
        true,
    );
    // +Y and -Y
    add_quad_grid(
        nx,
        nz,
        &|u, v| [lerp(min[0], max[0], u), max[1], lerp(min[2], max[2], v)],
        true,
    );
    add_quad_grid(
        nx,
        nz,
        &|u, v| [lerp(min[0], max[0], u), min[1], lerp(min[2], max[2], v)],
        false,
    );
    // +X and -X
    add_quad_grid(
        ny,
        nz,
        &|u, v| [max[0], lerp(min[1], max[1], u), lerp(min[2], max[2], v)],
        false,
    );
    add_quad_grid(
        ny,
        nz,
        &|u, v| [min[0], lerp(min[1], max[1], u), lerp(min[2], max[2], v)],
        true,
    );

    // Weld shared corners/edges deterministically so it is closed manifold.
    let mut raw = Mesh::from_buffers(&positions, &indices).unwrap();
    let mut tx = raw.begin_edit();
    fast_mesh_ops::weld(&mut tx, 1e-5).unwrap();
    tx.commit().unwrap();
    raw
}

/// Outward-wound closed UV sphere centered at the origin.
fn sphere_mesh(r: f32, stacks: u32, slices: u32) -> Mesh {
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
        indices.extend([south, ring(stacks - 1, j + 1), ring(stacks - 1, j)]);
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn attach_planar_uvs(mesh: &mut Mesh) {
    let uvs: Vec<f32> = mesh
        .positions()
        .iter()
        .flat_map(|p| [p[0] * 0.1 + 0.5, p[1] * 0.1 + 0.5])
        .collect();
    mesh.add_uvs(&uvs).unwrap();
}

fn assert_watertight_manifold(mesh: &Mesh, expected_components: usize, context: &str) {
    let bvh = Bvh::build(mesh, &BuildOptions::default()).unwrap();
    let rep = analyze(
        mesh,
        &AnalyzeOptions {
            weld_tolerance: 0.0,
            self_intersection: Some(&bvh),
            max_listed: 16,
        },
    )
    .unwrap();
    assert!(
        rep.loops.is_empty(),
        "{context}: expected closed mesh, got {} boundary loops",
        rep.loops.len()
    );
    assert_eq!(
        rep.non_manifold_edges, 0,
        "{context}: expected 0 non-manifold edges, got {}",
        rep.non_manifold_edges
    );
    assert!(
        rep.pinch_vertices.is_empty(),
        "{context}: expected 0 pinch vertices, got {}",
        rep.pinch_vertices.len()
    );
    assert_eq!(
        rep.orientation_conflicts, 0,
        "{context}: expected 0 orientation conflicts, got {}",
        rep.orientation_conflicts
    );
    assert_eq!(
        rep.self_intersects,
        Some(false),
        "{context}: mesh must not self-intersect"
    );
    assert_eq!(
        rep.components.len(),
        expected_components,
        "{context}: expected {expected_components} components, got {}",
        rep.components.len()
    );

    let m = measure(mesh).unwrap();
    assert!(
        m.signed_volume > 0.0,
        "{context}: expected positive outward volume, got {}",
        m.signed_volume
    );
}

#[test]
fn hollow_box_bvh_and_fh_edt_reduce_volume_cleanly() {
    for backend in [HollowBackend::BvhSdf, HollowBackend::FhEdt] {
        let mut mesh = box_mesh([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]);
        let orig_vol = measure(&mesh).unwrap().volume.unwrap();
        assert!((orig_vol - 1000.0).abs() < 1e-3);

        let mut tx = mesh.begin_edit();
        let report = hollow_mesh(
            &mut tx,
            &HollowOptions {
                wall_thickness: 1.5,
                backend,
                voxel_size: Some(0.35),
                ..HollowOptions::default()
            },
        )
        .unwrap();
        tx.commit().unwrap();

        assert_eq!(report.cavities_created, 1);
        assert!(report.cavity_triangles > 0);
        // Expected cavity is approx [-3.5, 3.5]^3 -> 7^3 = 343 (rounded edges slightly smaller).
        assert!(
            report.cavity_volume > 260.0 && report.cavity_volume < 360.0,
            "unexpected cavity_volume {} for {backend:?}",
            report.cavity_volume
        );

        assert_watertight_manifold(&mesh, 2, &format!("hollow box {backend:?}"));

        let new_meas = measure(&mesh).unwrap();
        let new_vol = new_meas.volume.unwrap();
        assert!(
            (new_vol - (orig_vol - report.cavity_volume)).abs() < 1e-2,
            "volume conservation mismatch: new_vol={new_vol}, expected={}",
            orig_vol - report.cavity_volume
        );
        assert!(new_vol < orig_vol * 0.78);
    }
}

#[test]
fn hollow_sphere_with_cap_and_oversized_wall_and_undo() {
    let mut sphere = sphere_mesh(6.0, 20, 32);
    attach_planar_uvs(&mut sphere);
    let orig_meas = measure(&sphere).unwrap();
    let orig_vol = orig_meas.volume.unwrap();
    let orig_tris = sphere.triangle_count();

    // Oversized wall thickness returns 0 cavities cleanly and leaves topology & UVs untouched.
    let mut tx_over = sphere.begin_edit();
    let over_report = hollow_mesh(
        &mut tx_over,
        &HollowOptions {
            wall_thickness: 6.5,
            backend: HollowBackend::BvhSdf,
            voxel_size: Some(0.4),
            ..HollowOptions::default()
        },
    )
    .unwrap();
    tx_over.commit().unwrap();
    assert_eq!(over_report.cavities_created, 0);
    assert!(!over_report.uvs_dropped);
    assert_eq!(sphere.triangle_count(), orig_tris);
    assert!(sphere.uvs().is_some());

    // Valid hollow with triangle cap and undo check.
    let mut tx = sphere.begin_edit();
    let report = hollow_mesh(
        &mut tx,
        &HollowOptions {
            wall_thickness: 1.5,
            backend: HollowBackend::BvhSdf,
            voxel_size: Some(0.35),
            max_cavity_triangles: Some(250),
            min_cavity_volume: 10.0,
            ..HollowOptions::default()
        },
    )
    .unwrap();
    assert_eq!(report.cavities_created, 1);
    assert!(report.uvs_dropped);
    assert!(report.cavity_triangles <= 250);
    let patch: Patch = tx.commit().unwrap();

    assert!(sphere.uvs().is_none());
    assert_watertight_manifold(&sphere, 2, "hollow capped sphere");
    assert!(measure(&sphere).unwrap().volume.unwrap() < orig_vol * 0.7);

    // Undo restores exact original triangle count, volume, and UVs.
    patch.revert(&mut sphere).unwrap();
    assert_eq!(sphere.triangle_count(), orig_tris);
    assert!(sphere.uvs().is_some());
    let restored_vol = measure(&sphere).unwrap().volume.unwrap();
    assert!((restored_vol - orig_vol).abs() < 1e-6);
}

#[test]
fn hollow_with_wall_modulation_weights_thickens_top() {
    let base_box = subdivided_box_mesh([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], 6, 6, 6);

    let mut plain = box_mesh([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]);
    let mut tx_p = plain.begin_edit();
    let rep_plain = hollow_mesh(
        &mut tx_p,
        &HollowOptions {
            wall_thickness: 1.0,
            backend: HollowBackend::BvhSdf,
            voxel_size: Some(0.3),
            ..HollowOptions::default()
        },
    )
    .unwrap();
    tx_p.commit().unwrap();

    let mut modulated = base_box;
    let mut weights: Vec<(VertexId, f32)> = Vec::new();
    for v in 0..modulated.vertex_slots() as VertexId {
        if let Some(p) = modulated.position(v)
            && p[2] > 4.5
        {
            // Double local wall thickness on the +Z top face (1.0 -> 2.0).
            weights.push((v, 1.0));
        }
    }
    assert!(!weights.is_empty());

    let mut tx_m = modulated.begin_edit();
    let rep_mod = hollow_mesh(
        &mut tx_m,
        &HollowOptions {
            wall_thickness: 1.0,
            backend: HollowBackend::BvhSdf,
            voxel_size: Some(0.3),
            wall_modulation_weights: Some(weights),
            ..HollowOptions::default()
        },
    )
    .unwrap();
    tx_m.commit().unwrap();

    assert_eq!(rep_mod.cavities_created, 1);
    assert!(
        rep_mod.cavity_volume < rep_plain.cavity_volume - 20.0,
        "modulated cavity volume {} should be noticeably smaller than plain {}",
        rep_mod.cavity_volume,
        rep_plain.cavity_volume
    );
    assert_watertight_manifold(&modulated, 2, "modulated hollow box");
}

#[test]
fn punch_drain_hole_connects_box_and_sphere_cavities_and_builds_plugs() {
    // 1. Hollowed box with 1 chamfered, tapered hole + plug
    let mut mesh = box_mesh([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]);
    let mut tx = mesh.begin_edit();
    let hrep = hollow_mesh(
        &mut tx,
        &HollowOptions {
            wall_thickness: 1.5,
            backend: HollowBackend::BvhSdf,
            voxel_size: Some(0.35),
            ..HollowOptions::default()
        },
    )
    .unwrap();
    assert_eq!(hrep.cavities_created, 1);
    tx.commit().unwrap();
    assert_eq!(
        analyze(&mesh, &AnalyzeOptions::default())
            .unwrap()
            .components
            .len(),
        2
    );

    let hole1 = DrainHoleSpec {
        surface_point: [0.0, 0.0, 5.0],
        surface_normal: [0.0, 0.0, 1.0],
        diameter: 3.0,
        depth: None,
        outer_chamfer: 0.35,
        taper_angle_deg: 5.0,
        segments: 24,
    };
    let hole_opts = DrainHoleOptions {
        boolean_method: BooleanMethod::Manifold,
        fallback_bvh_sdf: false,
        generate_plug: true,
        plug_clearance: 0.08,
    };

    let mut tx_h1 = mesh.begin_edit();
    let drep1 = punch_drain_hole(&mut tx_h1, &hole1, &hole_opts).unwrap();
    let patch_hole1 = tx_h1.commit().unwrap();
    assert_eq!(drep1.holes_punched, 1);
    assert_eq!(drep1.plug_meshes.len(), 1);
    assert_eq!(drep1.method_used, BooleanMethod::Manifold);
    assert!(!drep1.used_fallback);

    // Outer shell and inner cavity now form ONE continuous watertight 2-manifold!
    assert_watertight_manifold(&mesh, 1, "box after 1 drain hole");
    assert_watertight_manifold(&drep1.plug_meshes[0], 1, "box drain plug 1");

    // Verify plug clearance shrinks plug radius strictly below nominal hole radius (1.5 + 0.35 = 1.85).
    let mut max_plug_r = 0.0f32;
    for v in 0..drep1.plug_meshes[0].vertex_slots() as VertexId {
        if let Some(p) = drep1.plug_meshes[0].position(v) {
            max_plug_r = max_plug_r.max(p[0].hypot(p[1]));
        }
    }
    assert!(
        max_plug_r < 1.85 - 0.05 && max_plug_r > 1.70,
        "plug rim radius should match nominal+chamfer minus clearance, got {max_plug_r}"
    );

    // Undo restores the 2 disconnected shells.
    patch_hole1.revert(&mut mesh).unwrap();
    assert_watertight_manifold(&mesh, 2, "box after drain hole undo");

    // 2. Punch TWO chamfered/tapered drain holes (+Z and -Z) in one call.
    let hole2 = DrainHoleSpec {
        surface_point: [0.0, 0.0, -5.0],
        surface_normal: [0.0, 0.0, -1.0],
        diameter: 2.6,
        depth: None,
        outer_chamfer: 0.25,
        taper_angle_deg: 4.0,
        segments: 20,
    };
    let mut tx_h2 = mesh.begin_edit();
    let drep2 = punch_drain_holes(&mut tx_h2, &[hole1, hole2], &hole_opts).unwrap();
    tx_h2.commit().unwrap();
    assert_eq!(drep2.holes_punched, 2);
    assert_eq!(drep2.plug_meshes.len(), 2);
    assert_watertight_manifold(&mesh, 1, "box after 2 drain holes");
    assert_watertight_manifold(&drep2.plug_meshes[1], 1, "box drain plug 2");

    // 3. Curved hollowed sphere with drain hole using auto-raycast depth (rim sagitta check!).
    let mut sph = sphere_mesh(6.0, 20, 32);
    let mut tx_sh = sph.begin_edit();
    hollow_mesh(
        &mut tx_sh,
        &HollowOptions {
            wall_thickness: 1.4,
            backend: HollowBackend::BvhSdf,
            voxel_size: Some(0.35),
            ..HollowOptions::default()
        },
    )
    .unwrap();
    tx_sh.commit().unwrap();
    assert_watertight_manifold(&sph, 2, "hollow sphere before hole");

    let mut tx_sd = sph.begin_edit();
    let sph_hole = DrainHoleSpec {
        surface_point: [0.0, 0.0, 6.0],
        surface_normal: [0.0, 0.0, 1.0],
        diameter: 2.8,
        depth: None,
        outer_chamfer: 0.25,
        taper_angle_deg: 3.0,
        segments: 24,
    };
    let sd_rep = punch_drain_hole(&mut tx_sd, &sph_hole, &hole_opts).unwrap();
    tx_sd.commit().unwrap();
    assert_eq!(sd_rep.holes_punched, 1);
    assert_watertight_manifold(&sph, 1, "hollow sphere after 1 drain hole");
}

#[test]
fn drain_hole_fallback_bvh_sdf_handles_self_intersecting_input() {
    // Build a deliberately self-intersecting closed solid (two overlapping boxes) so
    // exact Manifold boolean rejects with InvalidInput and fallback_bvh_sdf kicks in.
    let a = box_mesh([-5.0, -5.0, -5.0], [3.0, 5.0, 5.0]);
    let b = box_mesh([-1.0, -3.0, -3.0], [5.0, 3.0, 3.0]);
    let mut mesh = Mesh::new();
    let mut append_into = |src: &Mesh| {
        let pos: Vec<[f32; 3]> = src.positions().to_vec();
        let tris: Vec<[u32; 3]> = (0..src.triangle_slots() as TriangleId)
            .filter_map(|t| src.triangle(t))
            .collect();
        let mut tx = mesh.begin_edit();
        fast_mesh_ops::append_mesh(
            &mut tx,
            &fast_mesh_ops::AppendSource {
                positions: pos,
                triangles: tris,
            },
        )
        .unwrap();
        tx.commit().unwrap();
    };
    append_into(&a);
    append_into(&b);

    let hole = DrainHoleSpec {
        surface_point: [0.0, 0.0, 5.0],
        surface_normal: [0.0, 0.0, 1.0],
        diameter: 2.5,
        depth: Some(4.0),
        outer_chamfer: 0.2,
        taper_angle_deg: 0.0,
        segments: 16,
    };

    // Without fallback, Manifold rejects self-intersecting input cleanly without mutating mesh.
    let orig_tris = mesh.triangle_count();
    {
        let mut tx_strict = mesh.begin_edit();
        let err = punch_drain_hole(
            &mut tx_strict,
            &hole,
            &DrainHoleOptions {
                boolean_method: BooleanMethod::Manifold,
                fallback_bvh_sdf: false,
                generate_plug: false,
                plug_clearance: 0.0,
            },
        )
        .unwrap_err();
        assert!(matches!(err, OpsError::InvalidInput(_)));
    }
    assert_eq!(mesh.triangle_count(), orig_tris);

    // With fallback_bvh_sdf: true, BvhSdf succeeds cleanly and returns closed 2-manifold.
    let mut tx_fb = mesh.begin_edit();
    let rep = punch_drain_hole(
        &mut tx_fb,
        &hole,
        &DrainHoleOptions {
            boolean_method: BooleanMethod::Manifold,
            fallback_bvh_sdf: true,
            generate_plug: false,
            plug_clearance: 0.0,
        },
    )
    .unwrap();
    tx_fb.commit().unwrap();
    assert_eq!(rep.method_used, BooleanMethod::BvhSdf);
    assert!(rep.used_fallback);
    assert_watertight_manifold(&mesh, 1, "fallback_bvh_sdf drain hole");
}

#[test]
fn place_mesh_transform_up_axes_modes_conform_and_undo() {
    // Verify align_up_axis_rotation across all 4 PlaceUpAxis variants onto tilted & inverted normals.
    let tilted_raw = [1.0f32, 2.0, 3.0];
    let len = (1.0f64 + 4.0 + 9.0).sqrt();
    let tilted = [1.0 / len, 2.0 / len, 3.0 / len];

    for axis in [
        PlaceUpAxis::PosZ,
        PlaceUpAxis::NegZ,
        PlaceUpAxis::PosY,
        PlaceUpAxis::NegY,
    ] {
        for n_target in [tilted_raw, [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]] {
            let r = align_up_axis_rotation(axis, n_target).unwrap();
            let u = axis.unit_vector();
            let mapped = [
                r[0][0] * u[0] + r[0][1] * u[1] + r[0][2] * u[2],
                r[1][0] * u[0] + r[1][1] * u[1] + r[1][2] * u[2],
                r[2][0] * u[0] + r[2][1] * u[1] + r[2][2] * u[2],
            ];
            let nt_len = (f64::from(n_target[0]).powi(2)
                + f64::from(n_target[1]).powi(2)
                + f64::from(n_target[2]).powi(2))
            .sqrt();
            for k in 0..3 {
                assert!(
                    (mapped[k] - f64::from(n_target[k]) / nt_len).abs() < 1e-10,
                    "axis {axis:?} did not align to {n_target:?}"
                );
            }
        }
    }

    // Verify twist + scale + embed_depth in compute_place_transform.
    let params = PlaceMeshParams {
        hit_point: [0.0, 0.0, 5.0],
        hit_normal: [0.0, 0.0, 1.0],
        up_axis: PlaceUpAxis::PosZ,
        twist_angle_rad: FRAC_PI_2 as f32,
        uniform_scale: 2.0,
        embed_depth: 0.5,
        mode: PlaceMode::KeepSeparate,
        conform_to_surface: false,
        boolean_method: BooleanMethod::Manifold,
    };
    let xf = compute_place_transform(&params).unwrap();
    let p_mapped = xf.apply([1.0, 0.0, 1.0]);
    // Scale 2 -> (2, 0, 2); twist +90deg around +Z -> (0, 2, 2); translate (0, 0, 4.5) -> (0, 2, 6.5).
    assert!((p_mapped[0] - 0.0).abs() < 1e-6);
    assert!((p_mapped[1] - 2.0).abs() < 1e-6);
    assert!((p_mapped[2] - 6.5).abs() < 1e-6);
    let m4 = xf.to_matrix4();
    assert!((m4[14] - 4.5).abs() < 1e-12);
    let _ = tilted;

    // 1. KeepSeparate leaves target untouched.
    let mut target_box = box_mesh([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]);
    attach_planar_uvs(&mut target_box);
    let stamp = box_mesh([-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]);
    let orig_box_vol = measure(&target_box).unwrap().volume.unwrap();
    let orig_box_tris = target_box.triangle_count();

    {
        let mut tx_sep = target_box.begin_edit();
        let rep_sep = place_mesh(&mut tx_sep, &stamp, &params).unwrap();
        tx_sep.commit().unwrap();
        assert!(!rep_sep.uvs_dropped);
        assert_eq!(rep_sep.kept_triangles, orig_box_tris);
        let placed = rep_sep
            .placed_mesh
            .expect("KeepSeparate must return placed_mesh");
        assert_watertight_manifold(&placed, 1, "KeepSeparate placed stamp");
        assert_eq!(target_box.triangle_count(), orig_box_tris);
        assert!(target_box.uvs().is_some());
    }

    // 2. EmbedUnion increases box volume, drops UVs, stays 2-manifold, and undoes cleanly.
    {
        let mut tx_union = target_box.begin_edit();
        let rep_u = place_mesh(
            &mut tx_union,
            &stamp,
            &PlaceMeshParams {
                hit_point: [0.0, 0.0, 5.0],
                hit_normal: [0.0, 0.0, 1.0],
                up_axis: PlaceUpAxis::PosZ,
                twist_angle_rad: 0.4,
                uniform_scale: 1.5,
                embed_depth: 0.5,
                mode: PlaceMode::EmbedUnion,
                conform_to_surface: false,
                boolean_method: BooleanMethod::Manifold,
            },
        )
        .unwrap();
        assert!(rep_u.placed_mesh.is_none());
        assert!(rep_u.uvs_dropped);
        let patch = tx_union.commit().unwrap();

        assert_watertight_manifold(&target_box, 1, "EmbedUnion on box");
        let vol_after = measure(&target_box).unwrap().volume.unwrap();
        assert!(
            vol_after > orig_box_vol + 10.0,
            "EmbedUnion should increase volume: {orig_box_vol} -> {vol_after}"
        );
        assert!(target_box.uvs().is_none());

        patch.revert(&mut target_box).unwrap();
        assert!(target_box.uvs().is_some());
        assert!((measure(&target_box).unwrap().volume.unwrap() - orig_box_vol).abs() < 1e-6);
    }

    // 2b. AppendShell appends stamp as second disconnected shell, drops UVs, and undoes cleanly.
    {
        let mut tx_app = target_box.begin_edit();
        let rep_app = place_mesh(
            &mut tx_app,
            &stamp,
            &PlaceMeshParams {
                hit_point: [0.0, 0.0, 6.0],
                hit_normal: [0.0, 0.0, 1.0],
                up_axis: PlaceUpAxis::PosZ,
                twist_angle_rad: 0.0,
                uniform_scale: 1.0,
                embed_depth: 0.0,
                mode: PlaceMode::AppendShell,
                conform_to_surface: false,
                boolean_method: BooleanMethod::Manifold,
            },
        )
        .unwrap();
        assert!(rep_app.placed_mesh.is_none());
        assert!(rep_app.uvs_dropped);
        let patch = tx_app.commit().unwrap();
        assert_watertight_manifold(&target_box, 2, "AppendShell on box");
        patch.revert(&mut target_box).unwrap();
        assert!(target_box.uvs().is_some());
        assert!((measure(&target_box).unwrap().volume.unwrap() - orig_box_vol).abs() < 1e-6);
    }

    // 3. EngraveSubtract decreases box volume and stays watertight 2-manifold.
    {
        let mut tx_sub = target_box.begin_edit();
        let rep_s = place_mesh(
            &mut tx_sub,
            &stamp,
            &PlaceMeshParams {
                hit_point: [0.0, 0.0, 5.0],
                hit_normal: [0.0, 0.0, 1.0],
                up_axis: PlaceUpAxis::PosZ,
                twist_angle_rad: 0.0,
                uniform_scale: 1.0,
                embed_depth: 1.0,
                mode: PlaceMode::EngraveSubtract,
                conform_to_surface: false,
                boolean_method: BooleanMethod::Manifold,
            },
        )
        .unwrap();
        assert!(rep_s.placed_mesh.is_none());
        let patch = tx_sub.commit().unwrap();

        assert_watertight_manifold(&target_box, 1, "EngraveSubtract on box");
        let vol_after = measure(&target_box).unwrap().volume.unwrap();
        assert!(
            (vol_after - (orig_box_vol - 4.0)).abs() < 0.25,
            "expected box volume {} after 2x2x1 engraving, got {vol_after}",
            orig_box_vol - 4.0
        );

        patch.revert(&mut target_box).unwrap();
        assert!((measure(&target_box).unwrap().volume.unwrap() - orig_box_vol).abs() < 1e-6);
    }

    // 4. conform_to_surface on a curved sphere bends stamp vertices onto sphere curvature!
    let mut sph = sphere_mesh(8.0, 24, 36);
    let subdivided_plate = subdivided_box_mesh([-3.0, -3.0, 0.0], [3.0, 3.0, 1.0], 6, 6, 1);

    {
        let mut tx_cf = sph.begin_edit();
        let rep_cf = place_mesh(
            &mut tx_cf,
            &subdivided_plate,
            &PlaceMeshParams {
                hit_point: [0.0, 0.0, 8.0],
                hit_normal: [0.0, 0.0, 1.0],
                up_axis: PlaceUpAxis::PosZ,
                twist_angle_rad: 0.0,
                uniform_scale: 1.0,
                embed_depth: 0.0,
                mode: PlaceMode::KeepSeparate,
                conform_to_surface: true,
                boolean_method: BooleanMethod::Manifold,
            },
        )
        .unwrap();
        tx_cf.commit().unwrap();

        let conformed = rep_cf.placed_mesh.unwrap();
        assert_watertight_manifold(&conformed, 1, "conformed KeepSeparate stamp");

        // Verify corner base vertices dropped down to match sphere surface z = sqrt(8^2 - r^2).
        let mut checked_outer_base = false;
        for v in 0..conformed.vertex_slots() as VertexId {
            if let Some(p) = conformed.position(v) {
                let r_xy = p[0].hypot(p[1]);
                let expected_z_sphere = (64.0f32 - r_xy * r_xy).sqrt();
                // Vertices on base face (local z = 0) conform closely to the sphere surface.
                if r_xy > 2.0 && (p[2] - expected_z_sphere).abs() < 0.15 {
                    checked_outer_base = true;
                }
            }
        }
        assert!(
            checked_outer_base,
            "conformed outer vertices did not hug the sphere surface curvature"
        );
    }

    // 5. conform_to_surface + EmbedUnion merges conformed stamp cleanly into curved sphere!
    let orig_sph_vol = measure(&sph).unwrap().volume.unwrap();
    {
        let mut tx_su = sph.begin_edit();
        place_mesh(
            &mut tx_su,
            &subdivided_plate,
            &PlaceMeshParams {
                hit_point: [0.0, 0.0, 8.0],
                hit_normal: [0.0, 0.0, 1.0],
                up_axis: PlaceUpAxis::PosZ,
                twist_angle_rad: 0.0,
                uniform_scale: 1.0,
                embed_depth: 0.4,
                mode: PlaceMode::EmbedUnion,
                conform_to_surface: true,
                boolean_method: BooleanMethod::Manifold,
            },
        )
        .unwrap();
        tx_su.commit().unwrap();
    }
    assert_watertight_manifold(&sph, 1, "conformed EmbedUnion on sphere");
    assert!(measure(&sph).unwrap().volume.unwrap() > orig_sph_vol);
}

#[test]
fn hollow_and_drain_validation_min_volume_and_uv_undo() {
    let mut b = box_mesh([-4.0, -4.0, -4.0], [4.0, 4.0, 4.0]);
    attach_planar_uvs(&mut b);
    let orig_vol = measure(&b).unwrap().volume.unwrap();
    let orig_tris = b.triangle_count();

    // min_cavity_volume above actual cavity rejects cavity gracefully without mutating UVs.
    {
        let mut tx = b.begin_edit();
        let rep = hollow_mesh(
            &mut tx,
            &HollowOptions {
                wall_thickness: 1.5,
                backend: HollowBackend::FhEdt,
                voxel_size: Some(0.4),
                min_cavity_volume: 1000.0,
                ..HollowOptions::default()
            },
        )
        .unwrap();
        assert_eq!(rep.cavities_created, 0);
        assert!(!rep.uvs_dropped);
        tx.commit().unwrap();
    }
    assert!(b.uvs().is_some());
    assert_eq!(b.triangle_count(), orig_tris);

    // Invalid hollow options reject cleanly.
    for bad_wall in [0.0f32, -1.0, f32::NAN] {
        let mut tx = b.begin_edit();
        assert!(matches!(
            hollow_mesh(
                &mut tx,
                &HollowOptions {
                    wall_thickness: bad_wall,
                    ..HollowOptions::default()
                }
            ),
            Err(OpsError::InvalidInput(_))
        ));
    }
    {
        let mut tx = b.begin_edit();
        assert!(matches!(
            hollow_mesh(
                &mut tx,
                &HollowOptions {
                    max_cavity_triangles: Some(0),
                    ..HollowOptions::default()
                }
            ),
            Err(OpsError::InvalidInput(_))
        ));
    }
    {
        let mut tx = b.begin_edit();
        assert!(matches!(
            hollow_mesh(
                &mut tx,
                &HollowOptions {
                    wall_modulation_weights: Some(vec![(9999, 0.5)]),
                    ..HollowOptions::default()
                }
            ),
            Err(OpsError::InvalidInput(_))
        ));
    }

    // Empty drain hole slice is a clean no-op preserving UVs.
    {
        let mut tx = b.begin_edit();
        let rep = punch_drain_holes(&mut tx, &[], &DrainHoleOptions::default()).unwrap();
        assert_eq!(rep.holes_punched, 0);
        assert!(!rep.uvs_dropped);
        tx.commit().unwrap();
    }
    assert!(b.uvs().is_some());

    // Drain hole punch on UV-equipped mesh drops UVs cleanly and revert restores them.
    {
        let mut tx = b.begin_edit();
        let rep = punch_drain_hole(
            &mut tx,
            &DrainHoleSpec {
                surface_point: [0.0, 0.0, 4.0],
                surface_normal: [0.0, 0.0, 1.0],
                diameter: 2.0,
                depth: Some(4.0),
                outer_chamfer: 0.2,
                taper_angle_deg: 0.0,
                segments: 16,
            },
            &DrainHoleOptions {
                boolean_method: BooleanMethod::FhEdt,
                fallback_bvh_sdf: false,
                generate_plug: true,
                plug_clearance: 0.05,
            },
        )
        .unwrap();
        assert!(rep.uvs_dropped);
        assert_eq!(rep.plug_meshes.len(), 1);
        let patch = tx.commit().unwrap();
        assert!(b.uvs().is_none());
        assert_watertight_manifold(&b, 1, "FhEdt drain hole with UV drop");
        patch.revert(&mut b).unwrap();
        assert!(b.uvs().is_some());
        assert!((measure(&b).unwrap().volume.unwrap() - orig_vol).abs() < 1e-6);
    }

    // Drain hole spec validation checks.
    {
        let mut tx = b.begin_edit();
        assert!(matches!(
            punch_drain_hole(
                &mut tx,
                &DrainHoleSpec {
                    surface_normal: [0.0, 0.0, 0.0],
                    ..DrainHoleSpec::default()
                },
                &DrainHoleOptions::default()
            ),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(matches!(
            punch_drain_hole(
                &mut tx,
                &DrainHoleSpec {
                    depth: Some(0.2),
                    outer_chamfer: 0.5,
                    ..DrainHoleSpec::default()
                },
                &DrainHoleOptions::default()
            ),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(matches!(
            punch_drain_hole(
                &mut tx,
                &DrainHoleSpec {
                    diameter: 2.0,
                    segments: 6,
                    ..DrainHoleSpec::default()
                },
                &DrainHoleOptions::default()
            ),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(matches!(
            punch_drain_hole(
                &mut tx,
                &DrainHoleSpec {
                    diameter: 2.0,
                    ..DrainHoleSpec::default()
                },
                &DrainHoleOptions {
                    generate_plug: true,
                    plug_clearance: 1.0,
                    ..DrainHoleOptions::default()
                }
            ),
            Err(OpsError::InvalidInput(_))
        ));
    }
}

#[test]
fn place_mesh_volumetric_backends_and_validation() {
    let mut target = box_mesh([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]);
    let stamp = box_mesh([-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]);
    let orig_vol = measure(&target).unwrap().volume.unwrap();

    // Validation errors on zero normal, non-positive scale, empty stamp.
    {
        let mut tx = target.begin_edit();
        assert!(matches!(
            place_mesh(
                &mut tx,
                &stamp,
                &PlaceMeshParams {
                    hit_normal: [0.0, 0.0, 0.0],
                    ..PlaceMeshParams::default()
                }
            ),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(matches!(
            place_mesh(
                &mut tx,
                &stamp,
                &PlaceMeshParams {
                    uniform_scale: 0.0,
                    ..PlaceMeshParams::default()
                }
            ),
            Err(OpsError::InvalidInput(_))
        ));
        let empty = Mesh::new();
        assert!(matches!(
            place_mesh(&mut tx, &empty, &PlaceMeshParams::default()),
            Err(OpsError::InvalidInput(_))
        ));
    }

    // BvhSdf union followed by FhEdt subtract.
    {
        let mut tx = target.begin_edit();
        place_mesh(
            &mut tx,
            &stamp,
            &PlaceMeshParams {
                hit_point: [0.0, 0.0, 5.0],
                hit_normal: [0.0, 0.0, 1.0],
                up_axis: PlaceUpAxis::PosZ,
                twist_angle_rad: 0.0,
                uniform_scale: 1.5,
                embed_depth: 0.5,
                mode: PlaceMode::EmbedUnion,
                conform_to_surface: false,
                boolean_method: BooleanMethod::BvhSdf,
            },
        )
        .unwrap();
        let patch = tx.commit().unwrap();
        assert_watertight_manifold(&target, 1, "BvhSdf EmbedUnion");
        assert!(measure(&target).unwrap().volume.unwrap() > orig_vol + 10.0);
        patch.revert(&mut target).unwrap();
    }

    {
        let mut tx = target.begin_edit();
        place_mesh(
            &mut tx,
            &stamp,
            &PlaceMeshParams {
                hit_point: [0.0, 0.0, 5.0],
                hit_normal: [0.0, 0.0, 1.0],
                up_axis: PlaceUpAxis::PosZ,
                twist_angle_rad: 0.0,
                uniform_scale: 1.5,
                embed_depth: 1.0,
                mode: PlaceMode::EngraveSubtract,
                conform_to_surface: false,
                boolean_method: BooleanMethod::FhEdt,
            },
        )
        .unwrap();
        let patch = tx.commit().unwrap();
        assert_watertight_manifold(&target, 1, "FhEdt EngraveSubtract");
        assert!(measure(&target).unwrap().volume.unwrap() < orig_vol - 5.0);
        patch.revert(&mut target).unwrap();
    }
}
