use std::f64::consts::{FRAC_PI_2, PI};

use fast_mesh_ops::boolean::{
    DrainHoleOptions, DrainHoleSpec, HollowBackend, HollowOptions, PlaceMeshParams, PlaceMode,
    PlaceUpAxis,
};
use fast_mesh_ops::deform::{
    ArapAnchor, ArapHandle, ArapOptions, FfdBasis, FfdOptions, FfdOutsideMode, FreeformOp,
    KelvinletLoad, KelvinletOptions, XpbdOptions,
};
use fast_mesh_ops::inspect::{
    FitPrimitiveOptions, FittedPrimitive, PointCloud, PrimitiveKind, ReconstructMethod,
    SurfaceReconstructOptions,
};
use fast_mesh_ops::repair::{AdjacencyKind, ScreenSelectOptions, ScreenSelectionShape};
use fast_mesh_scene::{
    CollisionMode, CollisionRule, MeshId, MeshOptions, Placement, Scene, SceneBooleanOps,
    SceneDeformOps, SceneInspectOps, SceneOptions, SceneRepairOps, UpdateMode,
};
use fast_mesh_store::Mesh as OpsMesh;

fn build_box_mesh(min: [f64; 3], max: [f64; 3], origin: [f64; 3]) -> OpsMesh {
    let [x0, y0, z0] = min;
    let [x1, y1, z1] = max;
    #[rustfmt::skip]
    let pos = [
        x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0,
        x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1,
    ];
    #[rustfmt::skip]
    let tris = [
        0, 2, 1, 0, 3, 2,
        4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4,
        1, 2, 6, 1, 6, 5,
        2, 3, 7, 2, 7, 6,
        3, 0, 4, 3, 4, 7,
    ];
    OpsMesh::from_buffers_f64_with_origin(&pos, &tris, origin).unwrap()
}

fn build_subdivided_grid(nx: usize, ny: usize, size: f64) -> OpsMesh {
    let mut pos = Vec::new();
    for iy in 0..=ny {
        for ix in 0..=nx {
            pos.push((ix as f64 / nx as f64 - 0.5) * size);
            pos.push((iy as f64 / ny as f64 - 0.5) * size);
            pos.push(0.0);
        }
    }
    let mut tris = Vec::new();
    let stride = (nx + 1) as u32;
    for iy in 0..ny as u32 {
        for ix in 0..nx as u32 {
            let a = iy * stride + ix;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            tris.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
    OpsMesh::from_buffers_f64(&pos, &tris).unwrap()
}

fn build_uv_sphere(radius: f64, center: [f64; 3], lat_steps: usize, lon_steps: usize) -> OpsMesh {
    let mut pos = Vec::new();
    for ilat in 0..=lat_steps {
        let theta = PI * (ilat as f64) / (lat_steps as f64);
        let st = theta.sin();
        let ct = theta.cos();
        for ilon in 0..lon_steps {
            let phi = 2.0 * PI * (ilon as f64) / (lon_steps as f64);
            pos.push(center[0] + radius * st * phi.cos());
            pos.push(center[1] + radius * st * phi.sin());
            pos.push(center[2] + radius * ct);
        }
    }
    let mut tris = Vec::new();
    let ring = lon_steps as u32;
    for ilat in 0..lat_steps as u32 {
        for ilon in 0..ring {
            let next = (ilon + 1) % ring;
            let a = ilat * ring + ilon;
            let b = ilat * ring + next;
            let c = (ilat + 1) * ring + ilon;
            let d = (ilat + 1) * ring + next;
            if ilat > 0 {
                tris.extend_from_slice(&[a, b, d]);
            }
            if ilat + 1 < lat_steps as u32 {
                tris.extend_from_slice(&[a, d, c]);
            }
        }
    }
    OpsMesh::from_buffers_f64(&pos, &tris).unwrap()
}

fn rot_z(angle_rad: f64) -> [[f64; 3]; 3] {
    let (s, c) = angle_rad.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

fn world_positions_set(scene: &Scene, id: MeshId) -> Vec<[f64; 3]> {
    let mesh = scene.mesh(id).unwrap();
    let pos = mesh.positions();
    let mut pts = Vec::new();
    for (v, &p) in pos.iter().enumerate().take(mesh.vertex_slots()) {
        if mesh.is_vertex_alive(v as u32) {
            pts.push(scene.local_to_world(id, p.map(f64::from)).unwrap());
        }
    }
    pts
}

fn trans(translation: [f64; 3]) -> Placement {
    Placement {
        translation,
        ..Placement::IDENTITY
    }
}

#[test]
fn selection_ops_world_coords_and_split_extract_undo() {
    let mut scene = Scene::new(SceneOptions::default());
    let mesh = build_box_mesh([9.0, 1.0, -1.0], [11.0, 3.0, 1.0], [10.0, 2.0, 0.0]);
    let placement = Placement {
        rotation: rot_z(FRAC_PI_2),
        scale: [2.0, 2.0, 2.0],
        translation: [5.0, -3.0, 1.5],
    };
    let id = scene
        .add_mesh(
            mesh,
            MeshOptions {
                placement,
                mode: UpdateMode::Lazy,
                ..MeshOptions::default()
            },
        )
        .unwrap();

    // Local box is around origin [10, 2, 0] -> world center = rot_z(pi/2)*[20, 4, 0] + [5, -3, 1.5] = [1, 17, 1.5].
    // Top (+Z) face lies at local z = +1, world z = 1.5 + 2*1 = 3.5.
    let top_tris = SceneRepairOps::select_faces_in_sphere_world(
        &mut scene,
        id,
        [1.0, 17.0, 3.5],
        3.0,
        Some(([0.0, 0.0, 1.0], 0.2)),
    )
    .unwrap();
    assert_eq!(top_tris.len(), 2);

    // Ortho projection looking straight down -Z onto world cube centered around (1, 17).
    // Map [-4, 6] x [12, 22] x [-5, 5] -> [-1, 1]^3 column-major.
    let ortho_vp: [[f32; 4]; 4] = [
        [0.2, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0],
        [-0.2, -3.4, 0.0, 1.0],
    ];
    let screen_top = scene
        .select_faces_screen(
            id,
            &ScreenSelectionShape::Rect {
                min_ndc: [-1.0, -1.0],
                max_ndc: [1.0, 1.0],
            },
            &ScreenSelectOptions {
                view_proj: ortho_vp,
                camera_origin: [1.0, 17.0, 20.0],
                camera_forward: [0.0, 0.0, -1.0],
                is_orthographic: true,
                visible_only: true,
                reject_backfaces: true,
                include_touching: true,
            },
        )
        .unwrap();
    assert_eq!(screen_top, top_tris);

    let crease = scene.select_by_crease_angle(id, top_tris[0], 0.1).unwrap();
    assert_eq!(crease, top_tris);
    let shell = scene.select_connected_shell(id, top_tris[0]).unwrap();
    assert_eq!(shell.len(), 12);

    let grown = scene
        .grow_face_selection(id, &top_tris, 1, AdjacencyKind::Edge)
        .unwrap();
    assert!(grown.len() > top_tris.len());
    let shrunk = scene
        .shrink_face_selection(id, &grown, 1, AdjacencyKind::Edge)
        .unwrap();
    assert_eq!(shrunk, top_tris);

    let v_shrunk = scene
        .shrink_face_selection(id, &grown, 1, AdjacencyKind::Vertex)
        .unwrap();
    assert!(v_shrunk.len() < grown.len());

    let smoothed = scene.smooth_selection_boundary(id, &grown, 1).unwrap();
    assert!(!smoothed.is_empty());
    let filled = scene.fill_selection_holes(id, &shell, 4).unwrap();
    assert_eq!(filled, shell);

    let mask = scene.face_selection_to_vertex_mask(id, &top_tris).unwrap();
    assert_eq!(mask.len(), scene.mesh(id).unwrap().vertex_slots());
    let back_all = scene
        .vertex_mask_to_face_selection(id, &mask, 0.5, true)
        .unwrap();
    assert_eq!(back_all, top_tris);
    let back_any = scene
        .vertex_mask_to_face_selection(id, &mask, 0.5, false)
        .unwrap();
    assert!(back_any.len() > back_all.len());

    // Extract preserves exact world coordinates even with non-zero mesh.origin + placement.
    let extracted_id = scene.extract_selected_faces(id, &top_tris).unwrap();
    assert_eq!(scene.mesh(extracted_id).unwrap().triangle_count(), 2);
    let ext_world = world_positions_set(&scene, extracted_id);
    for wp in &ext_world {
        assert!((wp[2] - 3.5).abs() < 1e-4, "expected z=3.5, got {:?}", wp);
    }

    // Small island detection on extracted 2-triangle island.
    let islands = scene
        .select_small_face_islands(extracted_id, 4, None)
        .unwrap();
    assert_eq!(islands.len(), 2);

    // Split + undo restores original 12 triangles on `id`.
    let (split_id, split_rep) = scene.split_selected_faces(id, &top_tris).unwrap();
    assert_eq!(split_rep.removed_triangles, 2);
    assert_eq!(scene.mesh(id).unwrap().triangle_count(), 10);
    assert_eq!(scene.mesh(split_id).unwrap().triangle_count(), 2);
    assert!(scene.undo().unwrap());
    assert_eq!(scene.mesh(id).unwrap().triangle_count(), 12);
    assert!(scene.redo().unwrap());
    assert_eq!(scene.mesh(id).unwrap().triangle_count(), 10);

    // Delete faces + undo.
    let rem_shell = scene.select_connected_shell(split_id, 0).unwrap();
    let del_rep = scene
        .delete_selected_faces(split_id, &rem_shell[..1])
        .unwrap();
    assert_eq!(del_rep.removed_triangles, 1);
    assert_eq!(scene.mesh(split_id).unwrap().triangle_count(), 1);
    assert!(scene.undo().unwrap());
    assert_eq!(scene.mesh(split_id).unwrap().triangle_count(), 2);
}

#[test]
fn freeform_deform_ffd_arap_kelvinlet_xpbd_and_undo() {
    let mut scene = Scene::new(SceneOptions::default());
    let grid = build_subdivided_grid(4, 4, 4.0);
    let id = scene.add_mesh(grid, MeshOptions::default()).unwrap();

    // FFD
    let mut lattice = scene
        .build_ffd_lattice(id, [2, 2, 2], FfdBasis::Bernstein, 0.1)
        .unwrap();
    lattice.set_offset(1, 1, 1, [0.0, 0.0, 1.0]).unwrap();
    let ffd_opts = FfdOptions::new(lattice).with_outside(FfdOutsideMode::Clamp);
    let rep = SceneDeformOps::apply_freeform(&mut scene, id, &FreeformOp::Ffd(ffd_opts)).unwrap();
    assert!(rep.vertices_moved > 0 && rep.max_displacement > 0.1);
    assert!(scene.undo().unwrap());

    // Kelvinlet + protected vertex 0 stays fixed.
    let kel = KelvinletOptions::new(
        [0.0, 0.0, 0.0],
        KelvinletLoad::Grab {
            force: [0.0, 0.0, 2.0],
        },
    );
    let p0_before = scene.mesh(id).unwrap().positions()[0];
    let k_rep = scene
        .apply_freeform_protected(id, &FreeformOp::Kelvinlet(kel), &[0], None)
        .unwrap();
    assert!(k_rep.vertices_moved > 0);
    assert_eq!(scene.mesh(id).unwrap().positions()[0], p0_before);
    assert!(scene.undo().unwrap());

    // ARAP
    let arap = ArapOptions {
        anchors: vec![ArapAnchor {
            vertex: 0,
            position: Some(scene.mesh(id).unwrap().positions()[0]),
        }],
        handles: vec![ArapHandle {
            vertex: 24,
            target: [2.0, 2.0, 1.0],
        }],
        iterations: 5,
        ..ArapOptions::default()
    };
    let a_rep = scene.apply_freeform(id, &FreeformOp::Arap(arap)).unwrap();
    assert!(a_rep.vertices_moved > 0);
    assert!(scene.undo().unwrap());

    // XPBD
    let xpbd = XpbdOptions {
        substeps: 2,
        iterations: 3,
        gravity: [0.0, 0.0, -9.8],
        pins: vec![(0, scene.mesh(id).unwrap().positions()[0])],
        ..XpbdOptions::default()
    };
    let x_rep = scene.apply_freeform(id, &FreeformOp::Xpbd(xpbd)).unwrap();
    assert!(x_rep.vertices_moved > 0);
}

#[test]
fn collision_response_query_helper() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = scene
        .add_mesh(
            build_box_mesh([0.0; 3], [1.0; 3], [0.0; 3]),
            MeshOptions::default(),
        )
        .unwrap();
    let b = scene
        .add_mesh(
            build_box_mesh([5.0; 3], [6.0; 3], [0.0; 3]),
            MeshOptions::default(),
        )
        .unwrap();
    assert!(!scene.has_active_collision_response(a));

    scene
        .set_collision_rule(
            a,
            b,
            CollisionRule {
                mode: CollisionMode::Rigid,
                ..CollisionRule::default()
            },
        )
        .unwrap();
    assert!(scene.has_active_collision_response(a));
    assert!(scene.has_active_collision_response(b));

    scene.clear_collision_rule(a, b).unwrap();
    assert!(!scene.has_active_collision_response(a));

    scene
        .set_default_collision_rule(CollisionRule {
            mode: CollisionMode::Cut,
            ..CollisionRule::default()
        })
        .unwrap();
    assert!(scene.has_active_collision_response(a));
}

#[test]
fn metrology_fit_align_contours_and_point_cloud() {
    let mut scene = Scene::new(SceneOptions::default());
    let sphere = build_uv_sphere(2.0, [0.0; 3], 12, 24);
    let placement = trans([3.0, -4.0, 5.0]);
    let sid = scene
        .add_mesh(
            sphere,
            MeshOptions {
                placement,
                ..MeshOptions::default()
            },
        )
        .unwrap();

    let fit = SceneInspectOps::fit_primitive_world(
        &scene,
        sid,
        None,
        Some(PrimitiveKind::Sphere),
        &FitPrimitiveOptions::default(),
    )
    .unwrap();
    match fit.primitive {
        FittedPrimitive::Sphere { center, radius } => {
            assert!((radius - 2.0).abs() < 0.05);
            assert!((center[0] - 3.0).abs() < 0.05);
            assert!((center[1] + 4.0).abs() < 0.05);
            assert!((center[2] - 5.0).abs() < 0.05);
        }
        other => panic!("expected sphere primitive, got {:?}", other),
    }

    let new_pl = scene
        .align_to_fitted_primitive(sid, &fit.primitive, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
        .unwrap();
    assert!(new_pl.translation[0].abs() < 0.1 && new_pl.translation[1].abs() < 0.1);
    assert!(scene.undo().unwrap());

    let slice = scene
        .slice_contours_world(sid, [3.0, -4.0, 5.0], [0.0, 0.0, 1.0])
        .unwrap();
    assert!(!slice.loops.is_empty());
    let stack = scene
        .slice_contour_stack_world(sid, [3.0, -4.0, 4.5], [0.0, 0.0, 1.0], 0.5, 3)
        .unwrap();
    assert_eq!(stack.len(), 3);

    // Point cloud surface reconstruction
    let mut pts = Vec::new();
    let mut nrms = Vec::new();
    for ilat in 1..12 {
        let theta = PI * (ilat as f64) / 12.0;
        for ilon in 0..24 {
            let phi = 2.0 * PI * (ilon as f64) / 24.0;
            let n = [
                (theta.sin() * phi.cos()) as f32,
                (theta.sin() * phi.sin()) as f32,
                theta.cos() as f32,
            ];
            pts.push([n[0] * 1.5, n[1] * 1.5, n[2] * 1.5]);
            nrms.push(n);
        }
    }
    let cloud = PointCloud {
        positions: pts,
        normals: Some(nrms),
        colors: None,
    };
    let recon_id = scene
        .reconstruct_point_cloud(
            &cloud,
            &SurfaceReconstructOptions {
                method: ReconstructMethod::ImplicitMlsMarchingCubes,
                voxel_size: 0.35,
                ..SurfaceReconstructOptions::default()
            },
        )
        .unwrap();
    assert!(scene.mesh(recon_id).unwrap().triangle_count() > 0);
}

#[test]
fn hollow_drain_hole_and_place_mesh_world() {
    let mut scene = Scene::new(SceneOptions::default());
    let box_mesh = build_box_mesh([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], [0.0; 3]);
    let box_pl = trans([10.0, 0.0, 0.0]);
    let id = scene
        .add_mesh(
            box_mesh,
            MeshOptions {
                placement: box_pl,
                ..MeshOptions::default()
            },
        )
        .unwrap();

    let h_rep = SceneBooleanOps::hollow_mesh(
        &mut scene,
        id,
        &HollowOptions {
            wall_thickness: 1.5,
            backend: HollowBackend::BvhSdf,
            voxel_size: Some(0.5),
            ..HollowOptions::default()
        },
    )
    .unwrap();
    assert_eq!(h_rep.cavities_created, 1);

    // Punch a drain hole at the world top face (x = 10, y = 0, z = 5) with a plug mesh.
    let (dh_rep, plug_ids) = scene
        .punch_drain_hole_world(
            id,
            &DrainHoleSpec {
                surface_point: [10.0, 0.0, 5.0],
                surface_normal: [0.0, 0.0, 1.0],
                diameter: 2.0,
                depth: Some(3.0),
                outer_chamfer: 0.2,
                taper_angle_deg: 2.0,
                segments: 16,
            },
            &DrainHoleOptions {
                generate_plug: true,
                plug_clearance: 0.05,
                ..DrainHoleOptions::default()
            },
        )
        .unwrap();
    assert_eq!(dh_rep.holes_punched, 1);
    assert_eq!(plug_ids.len(), 1);
    let plug_bounds = scene.bounds(plug_ids[0]).unwrap().unwrap();
    let plug_pl = scene.placement(plug_ids[0]).unwrap();
    let plug_world_center = plug_pl.apply([
        f64::from(plug_bounds.min[0] + plug_bounds.max[0]) * 0.5,
        f64::from(plug_bounds.min[1] + plug_bounds.max[1]) * 0.5,
        f64::from(plug_bounds.min[2] + plug_bounds.max[2]) * 0.5,
    ]);
    assert!(
        (plug_world_center[0] - 10.0).abs() < 0.1
            && plug_world_center[1].abs() < 0.1
            && plug_world_center[2] > 2.0,
        "plug world center mismatch: {:?}",
        plug_world_center
    );

    // Place mesh: KeepSeparate + EmbedUnion on a fresh box.
    let target = scene
        .add_mesh(
            build_box_mesh([-4.0, -4.0, -4.0], [4.0, 4.0, 4.0], [0.0; 3]),
            MeshOptions {
                placement: trans([0.0, 20.0, 0.0]),
                ..MeshOptions::default()
            },
        )
        .unwrap();
    let stamp = scene
        .add_mesh(
            build_box_mesh([-0.5, -0.5, 0.0], [0.5, 0.5, 1.0], [0.0; 3]),
            MeshOptions::default(),
        )
        .unwrap();

    let (_, sep_id_opt) = scene
        .place_mesh_world(
            target,
            stamp,
            [0.0, 20.0, 4.0],
            [0.0, 0.0, 1.0],
            &PlaceMeshParams {
                up_axis: PlaceUpAxis::PosZ,
                mode: PlaceMode::KeepSeparate,
                ..PlaceMeshParams::default()
            },
        )
        .unwrap();
    let sep_id = sep_id_opt.unwrap();
    let sep_pts = world_positions_set(&scene, sep_id);
    assert!(
        sep_pts
            .iter()
            .any(|p| (p[1] - 20.0).abs() < 1.0 && p[2] > 4.5)
    );

    let pre_tris = scene.mesh(target).unwrap().triangle_count();
    let (union_rep, none_id) = scene
        .place_mesh_world(
            target,
            stamp,
            [0.0, 20.0, 4.0],
            [0.0, 0.0, 1.0],
            &PlaceMeshParams {
                up_axis: PlaceUpAxis::PosZ,
                embed_depth: 0.25,
                mode: PlaceMode::EmbedUnion,
                ..PlaceMeshParams::default()
            },
        )
        .unwrap();
    assert!(none_id.is_none());
    assert!(union_rep.kept_triangles > pre_tris);
    assert!(scene.undo().unwrap());
    assert_eq!(scene.mesh(target).unwrap().triangle_count(), pre_tris);
}

/// Undoes the one step that added `added`, checks that it is gone, and redoes
/// it: the same mesh is back, under the same id.
fn assert_add_undoes(scene: &mut Scene, added: MeshId) {
    let shape = |scene: &Scene, id: MeshId| {
        let mesh = scene.mesh(id).unwrap();
        let tris: Vec<_> = (0..mesh.triangle_slots() as u32)
            .filter_map(|t| mesh.triangle(t))
            .collect();
        let placement = scene.placement(id).unwrap();
        (mesh.positions().to_vec(), tris, mesh.origin(), placement)
    };
    let before = shape(scene, added);
    assert!(!before.1.is_empty());
    let live: Vec<_> = scene.ids().collect();
    scene.take_changes();
    assert!(scene.undo().unwrap());
    assert!(scene.mesh(added).is_err());
    assert!(scene.is_restorable(added));
    assert!(
        scene
            .take_changes()
            .iter()
            .any(|c| c.id == added && c.removed)
    );
    assert!(scene.redo().unwrap());
    assert_eq!(shape(scene, added), before);
    assert_eq!(scene.ids().collect::<Vec<_>>(), live);
    assert!(
        scene
            .take_changes()
            .iter()
            .any(|c| c.id == added && c.added)
    );
}

#[test]
fn ops_that_add_a_mesh_undo_to_nothing_and_redo_it() {
    let mut scene = Scene::new(SceneOptions::default());
    let id = scene
        .add_mesh(
            build_box_mesh([9.0, 1.0, -1.0], [11.0, 3.0, 1.0], [10.0, 2.0, 0.0]),
            MeshOptions {
                placement: Placement {
                    rotation: rot_z(FRAC_PI_2),
                    scale: [2.0, 2.0, 2.0],
                    translation: [5.0, -3.0, 1.5],
                },
                mode: UpdateMode::Lazy,
                ..MeshOptions::default()
            },
        )
        .unwrap();
    scene.clear_history().unwrap();

    let extracted = scene.extract_selected_faces(id, &[0, 1]).unwrap();
    assert_add_undoes(&mut scene, extracted);
    assert_eq!(scene.mesh(id).unwrap().triangle_count(), 12);

    let (detached, report) = scene.split_selected_faces(id, &[0, 1]).unwrap();
    assert_eq!(report.removed_triangles, 2);
    assert_add_undoes(&mut scene, detached);
    assert_eq!(scene.mesh(id).unwrap().triangle_count(), 10);
    assert!(scene.undo().unwrap());
    assert_eq!(scene.mesh(id).unwrap().triangle_count(), 12);
    assert!(scene.mesh(detached).is_err());
    assert!(scene.undo().unwrap());
    assert!(scene.mesh(extracted).is_err());
    assert!(!scene.can_undo(), "one step per op");
    scene.clear_history().unwrap();

    let mut cloud = PointCloud {
        positions: Vec::new(),
        normals: Some(Vec::new()),
        colors: None,
    };
    for ilat in 1..12 {
        let theta = PI * f64::from(ilat) / 12.0;
        for ilon in 0..24 {
            let phi = 2.0 * PI * f64::from(ilon) / 24.0;
            let n = [
                (theta.sin() * phi.cos()) as f32,
                (theta.sin() * phi.sin()) as f32,
                theta.cos() as f32,
            ];
            cloud.positions.push(n.map(|c| c * 1.5));
            cloud.normals.as_mut().unwrap().push(n);
        }
    }
    let surface = scene
        .reconstruct_point_cloud(
            &cloud,
            &SurfaceReconstructOptions {
                method: ReconstructMethod::ImplicitMlsMarchingCubes,
                voxel_size: 0.35,
                ..SurfaceReconstructOptions::default()
            },
        )
        .unwrap();
    assert_add_undoes(&mut scene, surface);
    assert!(scene.undo().unwrap());
    assert!(!scene.can_undo());

    let target = scene
        .add_mesh(
            build_box_mesh([-4.0, -4.0, -4.0], [4.0, 4.0, 4.0], [0.0; 3]),
            MeshOptions {
                placement: trans([0.0, 20.0, 0.0]),
                ..MeshOptions::default()
            },
        )
        .unwrap();
    let stamp = scene
        .add_mesh(
            build_box_mesh([-0.5, -0.5, 0.0], [0.5, 0.5, 1.0], [0.0; 3]),
            MeshOptions::default(),
        )
        .unwrap();
    for conform_to_surface in [false, true] {
        let (_, placed) = scene
            .place_mesh_world(
                target,
                stamp,
                [0.0, 20.0, 4.0],
                [0.0, 0.0, 1.0],
                &PlaceMeshParams {
                    up_axis: PlaceUpAxis::PosZ,
                    mode: PlaceMode::KeepSeparate,
                    conform_to_surface,
                    ..PlaceMeshParams::default()
                },
            )
            .unwrap();
        assert_add_undoes(&mut scene, placed.unwrap());
        assert!(scene.undo().unwrap());
        assert!(!scene.can_undo());
    }
}

#[test]
fn a_drain_hole_and_its_plug_are_one_undo_step() {
    let spec = DrainHoleSpec {
        surface_point: [10.0, 0.0, 5.0],
        surface_normal: [0.0, 0.0, 1.0],
        diameter: 2.0,
        depth: Some(3.0),
        outer_chamfer: 0.2,
        taper_angle_deg: 2.0,
        segments: 16,
    };
    let options = DrainHoleOptions {
        generate_plug: true,
        plug_clearance: 0.05,
        ..DrainHoleOptions::default()
    };
    // With one scale the hole is punched in local space, else in the world.
    for scale in [[1.0; 3], [1.0, 1.0, 2.0]] {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene
            .add_mesh(
                build_box_mesh([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], [0.0; 3]),
                MeshOptions {
                    placement: Placement {
                        translation: [10.0, 0.0, 5.0 - 5.0 * scale[2]],
                        scale,
                        ..Placement::IDENTITY
                    },
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        let whole = scene.mesh(id).unwrap().triangle_count();
        let (_, plugs) = scene.punch_drain_hole_world(id, &spec, &options).unwrap();
        assert_eq!(plugs.len(), 1);
        let punched = scene.mesh(id).unwrap().triangle_count();
        assert!(punched > whole);
        assert_add_undoes(&mut scene, plugs[0]);
        assert_eq!(scene.mesh(id).unwrap().triangle_count(), punched);
        assert!(scene.undo().unwrap());
        assert_eq!(scene.mesh(id).unwrap().triangle_count(), whole);
        assert!(scene.mesh(plugs[0]).is_err());
        assert!(!scene.can_undo(), "the hole and its plug are one step");

        // A failed punch leaves the caller's step open, with what it has.
        let moved = trans([1.0, 2.0, 3.0]);
        scene.begin_step();
        scene.set_placement(id, moved).unwrap();
        for bad in [
            DrainHoleSpec {
                diameter: -1.0,
                ..spec
            },
            DrainHoleSpec {
                surface_point: [f32::NAN; 3],
                ..spec
            },
        ] {
            assert!(scene.punch_drain_hole_world(id, &bad, &options).is_err());
        }
        assert!(scene.split_selected_faces(id, &[u32::MAX]).is_err());
        assert_eq!(scene.placement(id).unwrap(), moved);
        scene.end_step().unwrap();
        assert!(scene.undo().unwrap());
        assert!(!scene.can_undo());
    }
}
