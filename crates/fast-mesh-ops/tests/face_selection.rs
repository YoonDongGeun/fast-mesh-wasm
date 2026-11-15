//! Integration tests for `fast_mesh_ops::repair::selection` (#233).

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_ops::OpsError;
use fast_mesh_ops::repair::{
    AdjacencyKind, FaceSelection, ScreenSelectOptions, ScreenSelectionShape, SelectionCombineMode,
    delete_selected_faces, extract_selected_faces, face_selection_to_vertex_mask,
    fill_selection_holes, grow_face_selection, grow_face_selection_with, select_by_crease_angle,
    select_connected_shell, select_faces_in_sphere, select_faces_screen, select_small_face_islands,
    shrink_face_selection, smooth_selection_boundary, split_selected_faces,
    vertex_mask_to_face_selection,
};
use fast_mesh_store::{History, Mesh, TriangleId, VertexId};

// ---------------------------------------------------------------------------
// Test mesh and camera builders
// ---------------------------------------------------------------------------

/// Builds an axis-aligned planar grid in the XY plane at depth `z`, covering
/// `x in [x_bounds[0], x_bounds[1]]`, `y in [y_bounds[0], y_bounds[1]]`
/// with `nx * ny` quads (`2 * nx * ny` tris, normals along `+Z`).
fn make_plane_grid(nx: usize, ny: usize, x_bounds: [f32; 2], y_bounds: [f32; 2], z: f32) -> Mesh {
    let mut positions = Vec::with_capacity((nx + 1) * (ny + 1) * 3);
    for iy in 0..=ny {
        let ty = iy as f32 / ny as f32;
        let y = y_bounds[0] + (y_bounds[1] - y_bounds[0]) * ty;
        for ix in 0..=nx {
            let tx = ix as f32 / nx as f32;
            let x = x_bounds[0] + (x_bounds[1] - x_bounds[0]) * tx;
            positions.extend_from_slice(&[x, y, z]);
        }
    }

    let stride = nx + 1;
    let mut indices = Vec::with_capacity(nx * ny * 6);
    for iy in 0..ny {
        for ix in 0..nx {
            let a = (iy * stride + ix) as u32;
            let b = a + 1;
            let c = ((iy + 1) * stride + ix) as u32;
            let d = c + 1;
            indices.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }

    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// Builds a closed watertight cube `[-half, half]^3` where each of the 6 faces
/// is subdivided into `subdiv x subdiv` quads (`6 * 2 * subdiv^2` triangles).
fn make_subdivided_cube(half: f32, subdiv: usize) -> Mesh {
    let mut positions: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let eps = 1e-5_f32;

    let mut add_vert = |p: [f32; 3]| -> u32 {
        for i in 0..(positions.len() / 3) {
            let base = i * 3;
            if (positions[base] - p[0]).abs() < eps
                && (positions[base + 1] - p[1]).abs() < eps
                && (positions[base + 2] - p[2]).abs() < eps
            {
                return i as u32;
            }
        }
        let id = (positions.len() / 3) as u32;
        positions.extend_from_slice(&p);
        id
    };

    // Each face defined by origin + u_axis + v_axis oriented so cross(u_axis, v_axis) points outward.
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        (
            [-half, -half, half],
            [2.0 * half, 0.0, 0.0],
            [0.0, 2.0 * half, 0.0],
        ), // +Z
        (
            [half, -half, -half],
            [-2.0 * half, 0.0, 0.0],
            [0.0, 2.0 * half, 0.0],
        ), // -Z
        (
            [half, -half, half],
            [0.0, 0.0, -2.0 * half],
            [0.0, 2.0 * half, 0.0],
        ), // +X
        (
            [-half, -half, -half],
            [0.0, 0.0, 2.0 * half],
            [0.0, 2.0 * half, 0.0],
        ), // -X
        (
            [-half, half, half],
            [2.0 * half, 0.0, 0.0],
            [0.0, 0.0, -2.0 * half],
        ), // +Y
        (
            [-half, -half, -half],
            [2.0 * half, 0.0, 0.0],
            [0.0, 0.0, 2.0 * half],
        ), // -Y
    ];

    for (orig, u_ax, v_ax) in faces {
        let mut grid_verts = vec![0u32; (subdiv + 1) * (subdiv + 1)];
        for iv in 0..=subdiv {
            let tv = iv as f32 / subdiv as f32;
            for iu in 0..=subdiv {
                let tu = iu as f32 / subdiv as f32;
                let p = [
                    orig[0] + u_ax[0] * tu + v_ax[0] * tv,
                    orig[1] + u_ax[1] * tu + v_ax[1] * tv,
                    orig[2] + u_ax[2] * tu + v_ax[2] * tv,
                ];
                grid_verts[iv * (subdiv + 1) + iu] = add_vert(p);
            }
        }
        let s = subdiv + 1;
        for iv in 0..subdiv {
            for iu in 0..subdiv {
                let a = grid_verts[iv * s + iu];
                let b = grid_verts[iv * s + iu + 1];
                let c = grid_verts[(iv + 1) * s + iu];
                let d = grid_verts[(iv + 1) * s + iu + 1];
                indices.extend_from_slice(&[a, b, d, a, d, c]);
            }
        }
    }

    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// Concatenates disjoint meshes into a single `Mesh`.
fn merge_meshes(parts: &[&Mesh]) -> Mesh {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    for m in parts {
        let base = (positions.len() / 3) as u32;
        for v in 0..m.vertex_slots() {
            let p = m.position(v as VertexId).unwrap();
            positions.extend_from_slice(&p);
        }
        for t in 0..m.triangle_slots() {
            let tri = m.triangle(t as TriangleId).unwrap();
            indices.extend_from_slice(&[base + tri[0], base + tri[1], base + tri[2]]);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn tri_normal(mesh: &Mesh, t: TriangleId) -> [f32; 3] {
    let [a, b, c] = mesh.triangle(t).unwrap();
    let pa = mesh.position(a).unwrap();
    let pb = mesh.position(b).unwrap();
    let pc = mesh.position(c).unwrap();
    let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
    let n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    [n[0] / len, n[1] / len, n[2] / len]
}

fn tri_centroid(mesh: &Mesh, t: TriangleId) -> [f32; 3] {
    let [a, b, c] = mesh.triangle(t).unwrap();
    let pa = mesh.position(a).unwrap();
    let pb = mesh.position(b).unwrap();
    let pc = mesh.position(c).unwrap();
    [
        (pa[0] + pb[0] + pc[0]) / 3.0,
        (pa[1] + pb[1] + pc[1]) / 3.0,
        (pa[2] + pb[2] + pc[2]) / 3.0,
    ]
}

/// Column-major orthographic view-projection looking along `-Z` from `eye = [0, 0, eye_z]`
/// mapping `x in [-half_w, half_w]`, `y in [-half_h, half_h]`, `z in [z_far, z_near]` to NDC.
fn ortho_vp_neg_z(half_w: f32, half_h: f32, near: f32, far: f32, eye_z: f32) -> [[f32; 4]; 4] {
    // World z_view = z - eye_z, looking along -Z so near is at -near, far is at -far.
    let sz = -2.0 / (far - near);
    let tz = -(far + near) / (far - near) - sz * eye_z;
    [
        [1.0 / half_w, 0.0, 0.0, 0.0],
        [0.0, 1.0 / half_h, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [0.0, 0.0, tz, 1.0],
    ]
}

/// Column-major OpenGL-style perspective view-projection from `eye = [0, 0, eye_z]`
/// looking down `-Z`.
fn perspective_vp_neg_z(
    fovy_rad: f32,
    aspect: f32,
    near: f32,
    far: f32,
    eye_z: f32,
) -> [[f32; 4]; 4] {
    let f = 1.0 / (0.5 * fovy_rad).tan();
    let nf = 1.0 / (near - far);
    // Column-major: M = Proj * Translate(0, 0, -eye_z)
    // clip = [x * f/aspect, y * f, (z - eye_z) * (far + near)*nf + 2*far*near*nf, -(z - eye_z)]
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * nf, -1.0],
        [
            0.0,
            0.0,
            2.0 * far * near * nf - eye_z * (far + near) * nf,
            eye_z,
        ],
    ]
}

// ---------------------------------------------------------------------------
// 1. FaceSelection & Boolean Combinators
// ---------------------------------------------------------------------------

#[test]
fn face_selection_set_ops_and_dead_slot_invert() {
    let mut mesh = make_plane_grid(2, 2, [0.0, 2.0], [0.0, 2.0], 0.0);
    assert_eq!(mesh.triangle_count(), 8);

    let mut a = FaceSelection::from_triangles(&mesh, &[0, 2, 7, 2]).unwrap();
    assert_eq!(a.len(), 3);
    assert!(!a.is_empty());
    assert!(a.contains(0));
    assert!(a.contains(2));
    assert!(a.contains(7));
    assert!(!a.contains(1));
    assert_eq!(a.to_triangles(&mesh), vec![0, 2, 7]);

    assert!(!a.insert(2));
    assert!(a.insert(100));
    assert_eq!(a.len(), 4);
    assert!(a.remove(100));
    assert_eq!(a, FaceSelection::from_triangles(&mesh, &[0, 2, 7]).unwrap());

    let b = FaceSelection::from_triangles(&mesh, &[2, 3, 5]).unwrap();

    let mut u = a.clone();
    u.apply_mode(&b, SelectionCombineMode::Add);
    assert_eq!(u.to_triangles(&mesh), vec![0, 2, 3, 5, 7]);

    let mut diff = a.clone();
    diff.apply_mode(&b, SelectionCombineMode::Subtract);
    assert_eq!(diff.to_triangles(&mesh), vec![0, 7]);

    let mut inter = a.clone();
    inter.apply_mode(&b, SelectionCombineMode::Intersect);
    assert_eq!(inter.to_triangles(&mesh), vec![2]);

    let mut rep = a.clone();
    rep.apply_mode(&b, SelectionCombineMode::Replace);
    assert_eq!(rep, b);

    // Dead slots must remain excluded after invert().
    let mut tx = mesh.begin_edit();
    tx.remove_triangles(&[1, 4]).unwrap();
    tx.commit().unwrap();

    let mut inv = FaceSelection::from_triangles(&mesh, &[0, 2]).unwrap();
    inv.invert(&mesh);
    assert_eq!(inv.len(), 4);
    assert_eq!(inv.to_triangles(&mesh), vec![3, 5, 6, 7]);
    assert!(!inv.contains(1));
    assert!(!inv.contains(4));

    assert!(matches!(
        FaceSelection::from_triangles(&mesh, &[1]),
        Err(OpsError::InvalidInput(_))
    ));
}

// ---------------------------------------------------------------------------
// 2. Crease Angle Flood-Fill
// ---------------------------------------------------------------------------

#[test]
fn crease_flood_fill_on_subdivided_cube() {
    let cube = make_subdivided_cube(1.0, 3);
    assert_eq!(cube.triangle_count(), 108); // 6 faces * 9 quads * 2

    // Pick center triangle of the +Z face (normal [0, 0, 1]).
    let seed = (0..cube.triangle_slots())
        .map(|i| i as TriangleId)
        .find(|&t| {
            let n = tri_normal(&cube, t);
            let c = tri_centroid(&cube, t);
            n[2] > 0.99 && c[0].abs() < 0.35 && c[1].abs() < 0.35
        })
        .unwrap();

    // At 45 deg, propagation must stop at all 90 deg cube creases -> exactly 1 cube face (18 tris).
    let face_45 = select_by_crease_angle(&cube, &[seed], 45.0).unwrap();
    assert_eq!(face_45.len(), 18);
    for t in face_45.to_triangles(&cube) {
        let n = tri_normal(&cube, t);
        assert!(n[2] > 0.999, "leaked past 90 deg crease onto normal {n:?}");
    }

    // At 0 deg, all coplanar triangles on the +Z face still flood cleanly.
    let face_0 = select_by_crease_angle(&cube, &[seed], 0.0).unwrap();
    assert_eq!(face_0, face_45);

    // At 95 deg, flood-fill wraps over every 90 deg crease to the full cube.
    let whole_cube = select_by_crease_angle(&cube, &[seed], 95.0).unwrap();
    assert_eq!(whole_cube.len(), 108);

    // Validation errors.
    assert!(matches!(
        select_by_crease_angle(&cube, &[seed], -1.0),
        Err(OpsError::InvalidInput(_))
    ));
    assert!(matches!(
        select_by_crease_angle(&cube, &[seed], 181.0),
        Err(OpsError::InvalidInput(_))
    ));
    assert!(matches!(
        select_by_crease_angle(&cube, &[9999], 45.0),
        Err(OpsError::InvalidInput(_))
    ));
}

// ---------------------------------------------------------------------------
// 3. Connected Shell & Small Face Islands
// ---------------------------------------------------------------------------

#[test]
fn connected_shell_and_small_face_islands() {
    let cube = make_subdivided_cube(1.0, 1); // 12 tris, area = 24.0
    let small_quad = make_plane_grid(1, 1, [5.0, 5.5], [0.0, 0.5], 0.0); // 2 tris, area = 0.25
    let strip = make_plane_grid(2, 1, [10.0, 12.0], [0.0, 2.0], 0.0); // 4 tris, area = 4.0

    let scene = merge_meshes(&[&cube, &small_quad, &strip]);
    assert_eq!(scene.triangle_count(), 18);

    let seed_small = 12 as TriangleId;
    let seed_strip = 14 as TriangleId;

    let shell_small = select_connected_shell(&scene, &[seed_small]).unwrap();
    assert_eq!(shell_small.to_triangles(&scene), vec![12, 13]);

    let both_small = select_connected_shell(&scene, &[seed_small, seed_strip]).unwrap();
    assert_eq!(
        both_small.to_triangles(&scene),
        vec![12, 13, 14, 15, 16, 17]
    );

    // Filter strictly by small count & area.
    let tiny = select_small_face_islands(&scene, 2, 0.5).unwrap();
    assert_eq!(tiny.to_triangles(&scene), vec![12, 13]);

    let small_and_strip = select_small_face_islands(&scene, 4, 5.0).unwrap();
    assert_eq!(
        small_and_strip.to_triangles(&scene),
        vec![12, 13, 14, 15, 16, 17]
    );

    assert!(matches!(
        select_small_face_islands(&scene, 4, -0.1),
        Err(OpsError::InvalidInput(_))
    ));
}

// ---------------------------------------------------------------------------
// 4. 3D Sphere & Normal Cone Picker
// ---------------------------------------------------------------------------

#[test]
fn sphere_and_normal_cone_selection() {
    let cube = make_subdivided_cube(1.0, 2); // 48 tris
    let bvh = Bvh::build(&cube, &BuildOptions::default()).unwrap();

    // Sphere overlapping the +Z / +X / +Y corner [1, 1, 1].
    let without_cone_temp =
        select_faces_in_sphere(&cube, None, [1.0, 1.0, 1.0], 0.6, None).unwrap();
    let without_cone_cached =
        select_faces_in_sphere(&cube, Some(&bvh), [1.0, 1.0, 1.0], 0.6, None).unwrap();
    assert_eq!(without_cone_temp, without_cone_cached);
    assert!(without_cone_temp.len() >= 3);

    // Filter with cone centered on +Z (45 deg): excludes +X and +Y corner faces.
    let z_cone = select_faces_in_sphere(
        &cube,
        Some(&bvh),
        [1.0, 1.0, 1.0],
        0.6,
        Some(([0.0, 0.0, 1.0], 30.0)),
    )
    .unwrap();
    assert!(!z_cone.is_empty());
    assert!(z_cone.len() < without_cone_temp.len());
    for t in z_cone.to_triangles(&cube) {
        assert!(tri_normal(&cube, t)[2] > 0.99);
    }

    assert!(matches!(
        select_faces_in_sphere(&cube, Some(&bvh), [0.0, 0.0, 0.0], -1.0, None),
        Err(OpsError::InvalidInput(_))
    ));
    assert!(matches!(
        select_faces_in_sphere(
            &cube,
            Some(&bvh),
            [0.0, 0.0, 0.0],
            1.0,
            Some(([0.0, 0.0, 0.0], 30.0))
        ),
        Err(OpsError::InvalidInput(_))
    ));
}

// ---------------------------------------------------------------------------
// 5. Screen-Space Selection (Rect & Concave Lasso, Ortho & Perspective, Occlusion)
// ---------------------------------------------------------------------------

#[test]
fn screen_select_rect_and_concave_lasso_occlusion_ortho_and_perspective() {
    // Front layer: 4x4 grid at z = 0 spanning [-0.6, 0.6]^2 (32 tris, ids 0..32).
    // Back layer:  6x6 grid at z = -2 spanning [-0.9, 0.9]^2 (72 tris, ids 32..104).
    // Center region [-0.5, 0.5]^2 is occluded on the back layer by the front layer;
    // outer rim of back layer remains unoccluded.
    let front = make_plane_grid(4, 4, [-0.6, 0.6], [-0.6, 0.6], 0.0);
    let back = make_plane_grid(6, 6, [-0.9, 0.9], [-0.9, 0.9], -2.0);
    let scene = merge_meshes(&[&front, &back]);
    let bvh = Bvh::build(&scene, &BuildOptions::default()).unwrap();
    let front_tris = front.triangle_count();

    let count_layers = |sel: &FaceSelection| -> (usize, usize) {
        let mut f = 0;
        let mut b = 0;
        for t in sel.to_triangles(&scene) {
            if (t as usize) < front_tris {
                f += 1;
            } else {
                b += 1;
            }
        }
        (f, b)
    };

    // Concave U-shaped lasso entirely inside the central occluded zone [-0.55, 0.55]^2
    // leaving an unselected concave notch around (0.0, 0.25).
    let concave_lasso = ScreenSelectionShape::Lasso {
        polygon_ndc: vec![
            [-0.45, -0.45],
            [0.45, -0.45],
            [0.45, 0.45],
            [0.18, 0.45],
            [0.18, 0.05],
            [-0.18, 0.05],
            [-0.18, 0.45],
            [-0.45, 0.45],
        ],
    };

    let center_rect = ScreenSelectionShape::Rect {
        min_ndc: [-0.45, -0.45],
        max_ndc: [0.45, 0.45],
    };

    // --- Orthographic tests ---
    let ortho_vp = ortho_vp_neg_z(1.0, 1.0, 0.5, 20.0, 5.0);
    let mut ortho_opts = ScreenSelectOptions {
        view_proj: ortho_vp,
        camera_origin: [0.0, 0.0, 5.0],
        camera_forward: [0.0, 0.0, -1.0],
        is_orthographic: true,
        visible_only: false,
        reject_backfaces: true,
        include_touching: true,
    };

    // Ortho Rect: visible_only=false selects BOTH layers; visible_only=true selects ONLY front layer.
    let o_rect_all = select_faces_screen(&scene, None, &center_rect, &ortho_opts).unwrap();
    let (f_all, b_all) = count_layers(&o_rect_all);
    assert!(
        f_all > 0 && b_all > 0,
        "expected both layers, got ({f_all}, {b_all})"
    );

    ortho_opts.visible_only = true;
    let o_rect_vis = select_faces_screen(&scene, None, &center_rect, &ortho_opts).unwrap();
    let o_rect_vis_cached =
        select_faces_screen(&scene, Some(&bvh), &center_rect, &ortho_opts).unwrap();
    assert_eq!(o_rect_vis, o_rect_vis_cached);
    let (f_vis, b_vis) = count_layers(&o_rect_vis);
    assert_eq!(f_vis, f_all);
    assert_eq!(b_vis, 0, "occluded back layer must not be selected");

    // Ortho Concave Lasso: visible_only=false selects BOTH layers; visible_only=true selects ONLY front layer,
    // AND triangle centered inside the concave bay (x ~ 0, y ~ 0.25) is excluded.
    ortho_opts.visible_only = false;
    let o_lasso_all = select_faces_screen(&scene, Some(&bvh), &concave_lasso, &ortho_opts).unwrap();
    let (lf_all, lb_all) = count_layers(&o_lasso_all);
    assert!(lf_all > 0 && lb_all > 0);

    ortho_opts.visible_only = true;
    let o_lasso_vis = select_faces_screen(&scene, Some(&bvh), &concave_lasso, &ortho_opts).unwrap();
    let (lf_vis, lb_vis) = count_layers(&o_lasso_vis);
    assert_eq!(lf_vis, lf_all);
    assert_eq!(lb_vis, 0);

    // Ensure no triangle strictly inside the concave notch around (0.0, 0.25) was selected.
    ortho_opts.include_touching = false;
    let o_lasso_strict =
        select_faces_screen(&scene, Some(&bvh), &concave_lasso, &ortho_opts).unwrap();
    for t in o_lasso_strict.to_triangles(&scene) {
        let c = tri_centroid(&scene, t);
        let in_notch = c[0].abs() < 0.14 && c[1] > 0.10;
        assert!(
            !in_notch,
            "concave lasso notch triangle {t} was wrongly selected"
        );
    }

    // --- Perspective tests ---
    // Camera at z = 2.5, fov = 60 deg.
    // Front at z = 0 (dist 2.5): x=0.6 maps to ndc = 0.6 / (2.5 * tan(30)) ~= 0.415.
    // Back  at z = -2 (dist 4.5): x=0.9 maps to ndc = 0.9 / (4.5 * tan(30)) ~= 0.346 (smaller!).
    //Wait: make sure rect [-0.28, 0.28] is completely inside front's NDC footprint [-0.41, 0.41]!
    let persp_vp = perspective_vp_neg_z(std::f32::consts::FRAC_PI_3, 1.0, 0.2, 20.0, 2.5);
    let persp_rect = ScreenSelectionShape::Rect {
        min_ndc: [-0.28, -0.28],
        max_ndc: [0.28, 0.28],
    };
    let persp_lasso = ScreenSelectionShape::Lasso {
        polygon_ndc: vec![
            [-0.28, -0.28],
            [0.28, -0.28],
            [0.28, 0.28],
            [0.10, 0.28],
            [0.10, 0.02],
            [-0.10, 0.02],
            [-0.10, 0.28],
            [-0.28, 0.28],
        ],
    };

    let mut persp_opts = ScreenSelectOptions {
        view_proj: persp_vp,
        camera_origin: [0.0, 0.0, 2.5],
        camera_forward: [0.0, 0.0, -1.0],
        is_orthographic: false,
        visible_only: false,
        reject_backfaces: true,
        include_touching: true,
    };

    let p_rect_all = select_faces_screen(&scene, Some(&bvh), &persp_rect, &persp_opts).unwrap();
    let (pf_all, pb_all) = count_layers(&p_rect_all);
    assert!(
        pf_all > 0 && pb_all > 0,
        "expected both perspective layers, got ({pf_all}, {pb_all})"
    );

    persp_opts.visible_only = true;
    let p_rect_vis = select_faces_screen(&scene, None, &persp_rect, &persp_opts).unwrap();
    let (pf_vis, pb_vis) = count_layers(&p_rect_vis);
    assert_eq!(pf_vis, pf_all);
    assert_eq!(pb_vis, 0);

    persp_opts.visible_only = false;
    let p_lasso_all = select_faces_screen(&scene, Some(&bvh), &persp_lasso, &persp_opts).unwrap();
    let (plf_all, plb_all) = count_layers(&p_lasso_all);
    assert!(plf_all > 0 && plb_all > 0);

    persp_opts.visible_only = true;
    let p_lasso_vis = select_faces_screen(&scene, Some(&bvh), &persp_lasso, &persp_opts).unwrap();
    let (plf_vis, plb_vis) = count_layers(&p_lasso_vis);
    assert_eq!(plf_vis, plf_all);
    assert_eq!(plb_vis, 0);

    // Wide ortho rect confirms unoccluded back triangles ARE still selected when visible_only=true.
    let wide_rect = ScreenSelectionShape::Rect {
        min_ndc: [-0.95, -0.95],
        max_ndc: [0.95, 0.95],
    };
    ortho_opts.include_touching = true;
    ortho_opts.visible_only = true;
    let wide_vis = select_faces_screen(&scene, Some(&bvh), &wide_rect, &ortho_opts).unwrap();
    let (wf, wb) = count_layers(&wide_vis);
    assert_eq!(wf, front_tris);
    assert!(
        wb > 0,
        "unoccluded outer rim of back layer must be selected"
    );
}

#[test]
fn screen_select_backfaces_and_touching_flag() {
    // Closed cube in front of camera: 12 triangles (half front-facing, half back-facing or perpendicular).
    let cube = make_subdivided_cube(0.5, 1);
    let persp_vp = perspective_vp_neg_z(std::f32::consts::FRAC_PI_2, 1.0, 0.2, 20.0, 3.0);
    let full_rect = ScreenSelectionShape::Rect {
        min_ndc: [-1.0, -1.0],
        max_ndc: [1.0, 1.0],
    };

    let mut opts = ScreenSelectOptions {
        view_proj: persp_vp,
        camera_origin: [0.0, 0.0, 3.0],
        camera_forward: [0.0, 0.0, -1.0],
        is_orthographic: false,
        visible_only: false,
        reject_backfaces: false,
        include_touching: true,
    };

    let all = select_faces_screen(&cube, None, &full_rect, &opts).unwrap();
    assert_eq!(all.len(), 12);

    opts.reject_backfaces = true;
    let front_only = select_faces_screen(&cube, None, &full_rect, &opts).unwrap();
    // From [0, 0, 3], only the +Z face (2 triangles) faces toward the camera; side walls and -Z face away.
    assert_eq!(front_only.len(), 2);

    // include_touching false vs true on a single huge quad spanning [-1, 1]^2 when rect is [-0.2, 0.2]^2.
    let big_quad = make_plane_grid(1, 1, [-1.0, 1.0], [-1.0, 1.0], 0.0);
    let small_rect = ScreenSelectionShape::Rect {
        min_ndc: [-0.2, -0.2],
        max_ndc: [0.2, 0.2],
    };
    let mut ortho = ScreenSelectOptions {
        view_proj: ortho_vp_neg_z(1.0, 1.0, 0.5, 10.0, 2.0),
        camera_origin: [0.0, 0.0, 2.0],
        camera_forward: [0.0, 0.0, -1.0],
        is_orthographic: true,
        visible_only: true,
        reject_backfaces: true,
        include_touching: false,
    };
    let strict = select_faces_screen(&big_quad, None, &small_rect, &ortho).unwrap();
    assert_eq!(strict.len(), 0);

    ortho.include_touching = true;
    let touching = select_faces_screen(&big_quad, None, &small_rect, &ortho).unwrap();
    assert_eq!(touching.len(), 2);
}

// ---------------------------------------------------------------------------
// 6. Topological Boundary Operations
// ---------------------------------------------------------------------------

#[test]
fn grow_shrink_fill_holes_and_smooth_boundary() {
    // 6x6 planar grid = 72 triangles.
    let grid = make_plane_grid(6, 6, [0.0, 6.0], [0.0, 6.0], 0.0);
    // Quad at (ix=2, iy=2) has triangles 2*(2*6 + 2) = 28 and 29.
    let center_quad = FaceSelection::from_triangles(&grid, &[28, 29]).unwrap();

    let grown_edge = grow_face_selection_with(&grid, &center_quad, 1, AdjacencyKind::Edge).unwrap();
    assert_eq!(grown_edge.len(), 6);

    let grown_vert = grow_face_selection(&grid, &center_quad, 1).unwrap();
    assert!(grown_vert.len() > grown_edge.len());

    // Shrinking a 1-ring-grown patch removes the outer rim triangles.
    let shrunk = shrink_face_selection(&grid, &grown_vert, 1).unwrap();
    assert!(shrunk.len() < grown_vert.len());
    assert!(shrunk.contains(28) && shrunk.contains(29));

    // Selecting every triangle on an open grid and shrinking peels the boundary rim.
    let mut all_open = FaceSelection::new();
    all_open.invert(&grid);
    let peeled_open = shrink_face_selection(&grid, &all_open, 1).unwrap();
    assert!(peeled_open.len() < grid.triangle_count());

    // Selecting every triangle on a closed cube and shrinking peels nothing (no boundary).
    let cube = make_subdivided_cube(1.0, 1);
    let mut all_cube = FaceSelection::new();
    all_cube.invert(&cube);
    assert_eq!(
        shrink_face_selection(&cube, &all_cube, 2).unwrap().len(),
        12
    );

    // Fill enclosed hole: build a 3x3 inner block minus the center 1x1 quad (2-tri hole),
    // while leaving the outer grid ring unselected and bordering the mesh boundary.
    let two_rings = grow_face_selection(&grid, &center_quad, 1).unwrap();
    let mut ring_with_hole = two_rings.clone();
    ring_with_hole.apply_mode(&center_quad, SelectionCombineMode::Subtract);
    assert!(!ring_with_hole.contains(28));
    assert!(!ring_with_hole.contains(29));

    let filled = fill_selection_holes(&grid, &ring_with_hole, 4).unwrap();
    assert_eq!(
        filled, two_rings,
        "only the 2-tri enclosed hole should be filled"
    );

    // Boundary smoothing: remove a single protruding boundary triangle and heal a 1-tri pinhole.
    let mut noisy = grown_vert.clone();
    // Grow a 2x2 block first so interior boundary has 2 selected neighbors cleanly.
    let mut patch_4x4 = FaceSelection::new();
    for iy in 1..=4 {
        for ix in 1..=4 {
            let base = (iy * 6 + ix) * 2;
            patch_4x4.insert(base as TriangleId);
            patch_4x4.insert((base + 1) as TriangleId);
        }
    }
    noisy.apply_mode(&patch_4x4, SelectionCombineMode::Replace);
    // Poke a 1-triangle hole inside (all 3 neighbors selected -> heals!).
    let pinhole = (2 * 6 + 2) * 2;
    noisy.remove(pinhole as TriangleId);
    // Add an isolated 1-triangle spike far outside touching nothing (0 neighbors -> smoothed off!).
    let spike = 0; // corner quad tri 0 has only 2 neighbors (1 outside, 1 outside).
    noisy.insert(spike);

    let smoothed = smooth_selection_boundary(&grid, &noisy, 1).unwrap();
    assert!(
        smoothed.contains(pinhole as TriangleId),
        "isolated pinhole must heal"
    );
    assert!(
        !smoothed.contains(spike),
        "isolated corner spike must be removed"
    );
}

// ---------------------------------------------------------------------------
// 7. Delete, Extract, Split, UV_CHANNEL & History Undo/Redo
// ---------------------------------------------------------------------------

#[test]
fn delete_extract_split_preserves_uvs_and_undo_redo() {
    let base_mesh = make_plane_grid(4, 4, [0.0, 4.0], [0.0, 4.0], 0.0);
    let mut pos64 = Vec::with_capacity(base_mesh.vertex_slots() * 3);
    let mut uvs = Vec::with_capacity(base_mesh.vertex_slots() * 2);
    let origin = [1000.0, 2000.0, -500.0];
    for v in 0..base_mesh.vertex_slots() {
        let p = base_mesh.position(v as VertexId).unwrap();
        pos64.extend_from_slice(&[
            origin[0] + f64::from(p[0]),
            origin[1] + f64::from(p[1]),
            origin[2] + f64::from(p[2]),
        ]);
        uvs.extend_from_slice(&[p[0] * 0.25, p[1] * 0.25]);
    }
    let mut tris_u32 = Vec::with_capacity(base_mesh.triangle_count() * 3);
    for t in 0..base_mesh.triangle_slots() {
        tris_u32.extend_from_slice(&base_mesh.triangle(t as TriangleId).unwrap());
    }
    let build_uv_mesh = || {
        let mut m = Mesh::from_buffers_f64_with_origin(&pos64, &tris_u32, origin).unwrap();
        m.add_uvs(&uvs).unwrap();
        m
    };
    let mesh = build_uv_mesh();

    // Select middle 2x2 quads (ix in 1..3, iy in 1..3) -> 8 triangles, 1 interior vertex (12).
    let mut sel = FaceSelection::new();
    for iy in 1..3 {
        for ix in 1..3 {
            let base = (iy * 4 + ix) * 2;
            sel.insert(base as TriangleId);
            sel.insert((base + 1) as TriangleId);
        }
    }
    assert_eq!(sel.len(), 8);

    // 1) extract_selected_faces leaves source untouched and preserves UVs + origin.
    let extracted = extract_selected_faces(&mesh, &sel).unwrap();
    assert_eq!(extracted.triangle_count(), 8);
    assert_eq!(extracted.vertex_count(), 9);
    assert_eq!(extracted.origin(), origin);
    let ext_uvs = extracted.uvs().expect("extracted UVs must be preserved");
    assert_eq!(ext_uvs.len(), 9 * 2);
    for v in 0..extracted.vertex_slots() {
        let p = extracted.position(v as VertexId).unwrap();
        assert!((ext_uvs[2 * v] - p[0] * 0.25).abs() < 1e-6);
        assert!((ext_uvs[2 * v + 1] - p[1] * 0.25).abs() < 1e-6);
    }
    assert_eq!(mesh.triangle_count(), 32);

    // 2) delete_selected_faces + History undo/redo.
    let mut del_mesh = build_uv_mesh();
    let mut history = History::new(1 << 24);
    let mut tx = del_mesh.begin_edit();
    let report = delete_selected_faces(&mut tx, &sel).unwrap();
    let patch = tx.commit().unwrap();
    history.push(&mut del_mesh, patch).unwrap();

    assert_eq!(report.removed_triangles, 8);
    assert_eq!(report.removed_vertices, 1);
    assert_eq!(del_mesh.triangle_count(), 24);
    assert_eq!(del_mesh.vertex_count(), 24);
    let rem_uvs = del_mesh.uvs().expect("surviving UVs must remain attached");
    for v in 0..del_mesh.vertex_slots() {
        let vid = v as VertexId;
        if del_mesh.is_vertex_alive(vid) {
            let p = del_mesh.position(vid).unwrap();
            assert!((rem_uvs[2 * v] - p[0] * 0.25).abs() < 1e-6);
            assert!((rem_uvs[2 * v + 1] - p[1] * 0.25).abs() < 1e-6);
        }
    }

    // Undo must restore all 32 triangles, 25 vertices, and exact UVs.
    assert!(history.undo(&mut del_mesh).unwrap().is_some());
    assert_eq!(del_mesh.triangle_count(), 32);
    assert_eq!(del_mesh.vertex_count(), 25);
    assert_eq!(del_mesh.uvs().unwrap(), uvs.as_slice());

    // Redo must cleanly re-delete.
    assert!(history.redo(&mut del_mesh).unwrap().is_some());
    assert_eq!(del_mesh.triangle_count(), 24);
    assert_eq!(del_mesh.vertex_count(), 24);

    // 3) split_selected_faces atomically extracts + deletes and supports undo.
    let mut split_mesh = build_uv_mesh();
    let mut split_hist = History::new(1 << 24);
    let mut tx2 = split_mesh.begin_edit();
    let (split_part, split_report) = split_selected_faces(&mut tx2, &sel).unwrap();
    let patch2 = tx2.commit().unwrap();
    split_hist.push(&mut split_mesh, patch2).unwrap();

    assert_eq!(split_report.removed_triangles, 8);
    assert_eq!(split_report.removed_vertices, 1);
    assert_eq!(split_part.triangle_count(), 8);
    assert_eq!(split_part.vertex_count(), 9);
    assert_eq!(split_part.uvs().unwrap().len(), 18);
    assert_eq!(split_mesh.triangle_count(), 24);
    assert_eq!(split_mesh.vertex_count(), 24);

    assert!(split_hist.undo(&mut split_mesh).unwrap().is_some());
    assert_eq!(split_mesh.triangle_count(), 32);
    assert_eq!(split_mesh.vertex_count(), 25);
    assert_eq!(split_mesh.uvs().unwrap(), uvs.as_slice());
}

// ---------------------------------------------------------------------------
// 8. Face Selection <-> Vertex Mask Interop
// ---------------------------------------------------------------------------

#[test]
fn face_selection_and_vertex_mask_round_trip_with_feather() {
    let grid = make_plane_grid(4, 4, [0.0, 4.0], [0.0, 4.0], 0.0);
    let mut sel = FaceSelection::new();
    for iy in 1..3 {
        for ix in 1..3 {
            let base = (iy * 4 + ix) * 2;
            sel.insert(base as TriangleId);
            sel.insert((base + 1) as TriangleId);
        }
    }

    // Hard mask round-trip.
    let hard_mask = face_selection_to_vertex_mask(&grid, &sel, 0).unwrap();
    assert_eq!(hard_mask.len(), 9);
    for &(_, w) in &hard_mask {
        assert!((w - 1.0).abs() < 1e-6);
    }
    let recovered = vertex_mask_to_face_selection(&grid, &hard_mask, 0.5).unwrap();
    assert_eq!(recovered, sel);

    // Feathered mask emits all 25 vertices across 1-ring boundary out to edge of 4x4 grid.
    let feathered = face_selection_to_vertex_mask(&grid, &sel, 2).unwrap();
    assert_eq!(feathered.len(), 25);

    // Center vertex 12 has weight 1.0; outermost corner vertex 0 is at Chebyshev distance 1?
    // Grid vertices 0..25: core is rows 1..=3, cols 1..=3. Vertex (0,0) = 0 is 1 ring away!
    let w_v12 = feathered.iter().find(|&&(v, _)| v == 12).unwrap().1;
    let w_v0 = feathered.iter().find(|&&(v, _)| v == 0).unwrap().1;
    assert!((w_v12 - 1.0).abs() < 1e-6);
    assert!((w_v0 - (2.0 / 3.0)).abs() < 1e-5);

    // High threshold recovers core selection; medium threshold expands across 1-ring boundary.
    let core_from_feather = vertex_mask_to_face_selection(&grid, &feathered, 0.9).unwrap();
    assert_eq!(core_from_feather, sel);

    let expanded_ring1 = vertex_mask_to_face_selection(&grid, &feathered, 0.5).unwrap();
    assert_eq!(expanded_ring1.len(), 30);

    let expanded_ring2 = vertex_mask_to_face_selection(&grid, &feathered, 0.25).unwrap();
    assert_eq!(expanded_ring2.len(), 32);

    // Validation.
    assert!(matches!(
        vertex_mask_to_face_selection(&grid, &[(0, 1.5)], 0.5),
        Err(OpsError::InvalidInput(_))
    ));
    assert!(matches!(
        vertex_mask_to_face_selection(&grid, &[(999, 0.5)], 0.5),
        Err(OpsError::InvalidInput(_))
    ));
    assert!(matches!(
        vertex_mask_to_face_selection(&grid, &hard_mask, f32::NAN),
        Err(OpsError::InvalidInput(_))
    ));
}

#[test]
fn crease_exact_dihedral_threshold_and_empty_stale_guards() {
    // Central horizontal triangle flanked by a 30-deg tilted wing and a 60-deg tilted wing.
    let rad30 = 30.0_f32.to_radians();
    let rad60 = 60.0_f32.to_radians();
    let positions = [
        // Central triangle [0, 1, 2] in XY plane, normal +Z.
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        // 30 deg wing across edge (1, 0): bends down into -Z along -Y.
        0.5,
        -rad30.cos(),
        -rad30.sin(),
        // 60 deg wing across edge (0, 2): bends down into -Z along -X.
        -rad60.cos(),
        0.5,
        -rad60.sin(),
    ];
    let indices = [
        0, 1, 2, // center
        1, 0, 3, // 30 deg neighbor
        0, 2, 4, // 60 deg neighbor
    ];
    let wedge = Mesh::from_buffers(&positions, &indices).unwrap();

    assert_eq!(
        select_by_crease_angle(&wedge, &[0], 10.0)
            .unwrap()
            .to_triangles(&wedge),
        vec![0]
    );
    assert_eq!(
        select_by_crease_angle(&wedge, &[0], 35.0)
            .unwrap()
            .to_triangles(&wedge),
        vec![0, 1]
    );
    assert_eq!(
        select_by_crease_angle(&wedge, &[0], 65.0)
            .unwrap()
            .to_triangles(&wedge),
        vec![0, 1, 2]
    );

    // Empty mesh behavior across all operations.
    let empty = Mesh::from_buffers(&[], &[]).unwrap();
    let empty_sel = FaceSelection::new();
    assert!(
        select_by_crease_angle(&empty, &[], 45.0)
            .unwrap()
            .is_empty()
    );
    assert!(select_connected_shell(&empty, &[]).unwrap().is_empty());
    assert!(
        select_small_face_islands(&empty, 10, 10.0)
            .unwrap()
            .is_empty()
    );
    assert!(
        select_faces_in_sphere(&empty, None, [0.0; 3], 1.0, None)
            .unwrap()
            .is_empty()
    );
    assert!(
        grow_face_selection(&empty, &empty_sel, 2)
            .unwrap()
            .is_empty()
    );
    assert!(
        shrink_face_selection(&empty, &empty_sel, 2)
            .unwrap()
            .is_empty()
    );
    assert!(
        fill_selection_holes(&empty, &empty_sel, 5)
            .unwrap()
            .is_empty()
    );
    assert!(
        smooth_selection_boundary(&empty, &empty_sel, 2)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        extract_selected_faces(&empty, &empty_sel)
            .unwrap()
            .triangle_count(),
        0
    );
    assert!(
        face_selection_to_vertex_mask(&empty, &empty_sel, 2)
            .unwrap()
            .is_empty()
    );

    // Stale BVH must be rejected cleanly by select_faces_screen and select_faces_in_sphere.
    let mut grid = make_plane_grid(1, 1, [-0.5, 0.5], [-0.5, 0.5], 0.0);
    let stale_bvh = Bvh::build(&grid, &BuildOptions::default()).unwrap();
    let mut tx = grid.begin_edit();
    tx.remove_triangles(&[0]).unwrap();
    tx.commit().unwrap();

    let rect = ScreenSelectionShape::Rect {
        min_ndc: [-1.0, -1.0],
        max_ndc: [1.0, 1.0],
    };
    let opts = ScreenSelectOptions {
        view_proj: ortho_vp_neg_z(1.0, 1.0, 0.1, 10.0, 2.0),
        camera_origin: [0.0, 0.0, 2.0],
        camera_forward: [0.0, 0.0, -1.0],
        is_orthographic: true,
        visible_only: false,
        reject_backfaces: true,
        include_touching: true,
    };
    assert!(select_faces_screen(&grid, Some(&stale_bvh), &rect, &opts).is_err());
    assert!(select_faces_in_sphere(&grid, Some(&stale_bvh), [0.0; 3], 1.0, None).is_err());
}
