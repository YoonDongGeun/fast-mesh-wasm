//! Issue #206 regression and feature tests for manifold-grade booleans,
//! volumetric fallbacks (`BvhSdf`, `FhEdt`), seam sliver collapse, and N-ary
//! CSG batch evaluation.

use std::f64::consts::PI;

use fast_mesh_ops::boolean::{
    BooleanMethod, BooleanOp, BooleanOptions, CsgNode, apply_boolean_with_options, boolean,
    boolean_batch,
};
use fast_mesh_ops::{AnalyzeOptions, OpsError, analyze, measure};
use fast_mesh_query::Rigid;
use fast_mesh_store::{Mesh, TriangleId, par};

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

/// Outward-wound closed UV sphere around the origin.
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

/// Outward-wound closed cylinder centered at origin along Z (`[-half_h, +half_h]`).
fn cylinder(r: f32, half_h: f32, slices: u32) -> Mesh {
    let mut positions = Vec::with_capacity(((2 + 2 * slices) * 3) as usize);
    positions.extend([0.0, 0.0, -half_h]); // 0: bottom center
    positions.extend([0.0, 0.0, half_h]); // 1: top center
    for j in 0..slices {
        let phi = 2.0 * PI * f64::from(j) / f64::from(slices);
        let (x, y) = (f64::from(r) * phi.cos(), f64::from(r) * phi.sin());
        positions.extend([x as f32, y as f32, -half_h]);
    }
    for j in 0..slices {
        let phi = 2.0 * PI * f64::from(j) / f64::from(slices);
        let (x, y) = (f64::from(r) * phi.cos(), f64::from(r) * phi.sin());
        positions.extend([x as f32, y as f32, half_h]);
    }
    let bot = |j: u32| 2 + (j % slices);
    let top = |j: u32| 2 + slices + (j % slices);
    let mut indices = Vec::with_capacity((4 * slices * 3) as usize);
    for j in 0..slices {
        // Bottom cap (-z normal)
        indices.extend([0, bot(j + 1), bot(j)]);
        // Top cap (+z normal)
        indices.extend([1, top(j), top(j + 1)]);
        // Side wall (outward radial normal)
        let (b0, b1, t0, t1) = (bot(j), bot(j + 1), top(j), top(j + 1));
        indices.extend([b0, b1, t1, b0, t1, t0]);
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// Open box missing its `-x` face (valid single-boundary open surface).
fn open_box_missing_side(min: [f64; 3], max: [f64; 3]) -> Mesh {
    let mut m = box_mesh(min, max);
    let mut tx = m.begin_edit();
    tx.remove_triangles(&[0, 1]).unwrap();
    let _ = tx.commit().unwrap();
    m
}

/// Dirty self-intersecting mesh containing two overlapping closed shells.
fn self_intersecting_boxes() -> Mesh {
    let a = box_mesh([0.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
    let b = box_mesh([0.5, -0.4, -0.4], [1.5, 0.6, 0.6]);
    let mut positions = Vec::with_capacity((a.vertex_count() + b.vertex_count()) * 3);
    for p in a.positions() {
        positions.extend_from_slice(p);
    }
    for p in b.positions() {
        positions.extend_from_slice(p);
    }
    let offset = a.vertex_count() as u32;
    let mut indices = Vec::with_capacity((a.triangle_count() + b.triangle_count()) * 3);
    for t in a.triangles() {
        indices.extend_from_slice(t);
    }
    for t in b.triangles() {
        indices.extend(t.map(|v| v + offset));
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn rot_xyz_rigid(rx: f64, ry: f64, rz: f64, translation: [f64; 3]) -> Rigid {
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    let (sz, cz) = rz.sin_cos();
    Rigid {
        rotation: [
            [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
            [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
            [-sy, cy * sx, cx * cy],
        ],
        translation,
    }
}

fn clone_mesh(src: &Mesh) -> Mesh {
    let positions: Vec<f32> = src.positions().iter().flatten().copied().collect();
    let indices: Vec<u32> = src.triangles().iter().flatten().copied().collect();
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn assert_strict_manifold(mesh: &Mesh, ctx: &str) -> f64 {
    let m = measure(mesh).unwrap_or_else(|e| panic!("{ctx}: measure failed: {e:?}"));
    let an = analyze(mesh, &AnalyzeOptions::default())
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
    assert!(m.closed, "{ctx}: expected closed mesh: {m:?}");
    if mesh.triangle_count() > 0 {
        assert!(
            m.signed_volume > 0.0,
            "{ctx}: expected positive signed volume, got {}: {m:?}",
            m.signed_volume
        );
        m.volume.unwrap()
    } else {
        0.0
    }
}

fn min_aspect_ratio(mesh: &Mesh) -> f64 {
    let pos = mesh.positions();
    let mut min_ar = f64::INFINITY;
    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some([a, b, c]) = mesh.triangle(t) else {
            continue;
        };
        let p = [a, b, c].map(|v| pos[v as usize].map(f64::from));
        let e0 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
        let e1 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
        let e2 = [p[2][0] - p[1][0], p[2][1] - p[1][1], p[2][2] - p[1][2]];
        let cx = e0[1] * e1[2] - e0[2] * e1[1];
        let cy = e0[2] * e1[0] - e0[0] * e1[2];
        let cz = e0[0] * e1[1] - e0[1] * e1[0];
        let area2 = (cx * cx + cy * cy + cz * cz).sqrt();
        let sum_sq = (e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2])
            + (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2])
            + (e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
        if sum_sq > 0.0 {
            // Normalized in [0, 1]: 2 * sqrt(3) * area2 / sum_sq = 1.0 for equilateral.
            let q = 3.0_f64.sqrt() * area2 / sum_sq;
            min_ar = min_ar.min(q);
        }
    }
    min_ar
}

fn assert_meshes_bit_identical(a: &Mesh, b: &Mesh, ctx: &str) {
    assert_eq!(
        a.vertex_count(),
        b.vertex_count(),
        "{ctx}: vertex count mismatch"
    );
    assert_eq!(
        a.triangle_count(),
        b.triangle_count(),
        "{ctx}: triangle count mismatch"
    );
    assert_eq!(a.vertex_slots(), b.vertex_slots(), "{ctx}: vertex slots");
    assert_eq!(
        a.triangle_slots(),
        b.triangle_slots(),
        "{ctx}: triangle slots"
    );
    for (i, (pa, pb)) in a.positions().iter().zip(b.positions()).enumerate() {
        assert_eq!(
            pa.map(f32::to_bits),
            pb.map(f32::to_bits),
            "{ctx}: position bits differ at vertex {i}: {pa:?} vs {pb:?}"
        );
    }
    for (i, (ta, tb)) in a.triangles().iter().zip(b.triangles()).enumerate() {
        assert_eq!(ta, tb, "{ctx}: triangle differs at {i}");
    }
}

#[test]
fn coplanar_overlapping_and_congruent_boxes_are_strictly_manifold() {
    // 1. Overlapping coplanar axis-aligned slabs on z in [0, 1].
    let a = box_mesh([0.0, 0.0, 0.0], [2.0, 2.0, 1.0]);
    let b = box_mesh([1.0, 0.0, 0.0], [3.0, 2.0, 1.0]);
    for (op, want) in [
        (BooleanOp::Union, 6.0),
        (BooleanOp::Subtract, 2.0),
        (BooleanOp::Intersect, 2.0),
    ] {
        let opts = BooleanOptions {
            op,
            ..BooleanOptions::default()
        };
        let (out, rep) = boolean(&a, &b, Rigid::IDENTITY, &opts).unwrap();
        assert_eq!(rep.kept_triangles, out.triangle_count());
        let vol = assert_strict_manifold(&out, &format!("aligned coplanar {op:?}"));
        assert!((vol - want).abs() < 1e-4, "{op:?}: got {vol}, want {want}");
    }

    // 2. Identical congruent unit cubes.
    let c = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    for (op, want) in [
        (BooleanOp::Union, 1.0),
        (BooleanOp::Subtract, 0.0),
        (BooleanOp::Intersect, 1.0),
    ] {
        let opts = BooleanOptions {
            op,
            ..BooleanOptions::default()
        };
        let (out, _) = boolean(&c, &c, Rigid::IDENTITY, &opts).unwrap();
        let vol = assert_strict_manifold(&out, &format!("congruent {op:?}"));
        if want == 0.0 {
            assert_eq!(out.triangle_count(), 0);
        } else {
            assert!((vol - want).abs() < 1e-4, "{op:?}: got {vol}, want {want}");
        }
    }

    // 3. Coplanar slabs rotated 45 deg about Z around center (two pairs of flush coplanar faces,
    //    angled crossing edges on z = 0 and z = 1).
    let slab_a = box_mesh([-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]); // vol = 4.0
    let slab_b = box_mesh([-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]); // vol = 4.0
    let rot45 = rot_xyz_rigid(0.0, 0.0, PI / 4.0, [0.0, 0.0, 0.0]);
    // Intersection of two centered 2x2 squares rotated 45 deg has octagon area 8*(sqrt(2)-1) ~ 3.3137085.
    let inter_want = 8.0 * (2.0_f64.sqrt() - 1.0);
    for (op, want) in [
        (BooleanOp::Union, 8.0 - inter_want),
        (BooleanOp::Subtract, 4.0 - inter_want),
        (BooleanOp::Intersect, inter_want),
    ] {
        let opts = BooleanOptions {
            op,
            ..BooleanOptions::default()
        };
        let (out, _) = boolean(&slab_a, &slab_b, rot45, &opts).unwrap();
        let vol = assert_strict_manifold(&out, &format!("rotated coplanar {op:?}"));
        assert!((vol - want).abs() < 1e-3, "{op:?}: got {vol}, want {want}");
    }
}

#[test]
fn concentric_and_kissing_spheres_are_strictly_manifold() {
    // 1. Concentric spheres with differing subdivisions + 3D rotation (grazing near-coincident faces).
    let s1 = sphere(1.0, 10, 18);
    let s2 = sphere(1.0, 12, 20);
    let vol1 = measure(&s1).unwrap().volume.unwrap();
    let xf_rot = rot_xyz_rigid(0.31, 0.47, 0.23, [0.0, 0.0, 0.0]);
    for op in [BooleanOp::Union, BooleanOp::Intersect, BooleanOp::Subtract] {
        let opts = BooleanOptions {
            op,
            ..BooleanOptions::default()
        };
        let (out, _) = boolean(&s1, &s2, xf_rot, &opts).unwrap();
        let vol = assert_strict_manifold(&out, &format!("concentric same-r rotated {op:?}"));
        if op == BooleanOp::Union || op == BooleanOp::Intersect {
            assert!(
                (vol - vol1).abs() < 0.15,
                "concentric rotated {op:?}: vol={vol}, s1={vol1}"
            );
        }
    }

    // 2. Concentric nested spheres (r=1.0 vs r=0.5).
    let outer = sphere(1.0, 12, 24);
    let inner = sphere(0.5, 10, 18);
    let v_out = measure(&outer).unwrap().volume.unwrap();
    let v_in = measure(&inner).unwrap().volume.unwrap();
    for (op, want) in [
        (BooleanOp::Union, v_out),
        (BooleanOp::Intersect, v_in),
        (BooleanOp::Subtract, v_out - v_in),
    ] {
        let opts = BooleanOptions {
            op,
            ..BooleanOptions::default()
        };
        let (out, _) = boolean(&outer, &inner, Rigid::IDENTITY, &opts).unwrap();
        let vol = assert_strict_manifold(&out, &format!("nested concentric {op:?}"));
        assert!((vol - want).abs() < 1e-4, "{op:?}: got {vol}, want {want}");
    }

    // 3. Externally kissing unit spheres (offset by exact 2.0 along x).
    let kiss = Rigid {
        translation: [2.0, 0.0, 0.0],
        ..Rigid::IDENTITY
    };
    let s_ref = sphere(1.0, 10, 18);
    let v_ref = measure(&s_ref).unwrap().volume.unwrap();
    for op in [BooleanOp::Union, BooleanOp::Subtract, BooleanOp::Intersect] {
        let opts = BooleanOptions {
            op,
            ..BooleanOptions::default()
        };
        let (out, _) = boolean(&s_ref, &s_ref, kiss, &opts).unwrap();
        let vol = assert_strict_manifold(&out, &format!("kissing spheres {op:?}"));
        match op {
            BooleanOp::Union => assert!((vol - 2.0 * v_ref).abs() < 1e-3),
            BooleanOp::Subtract => assert!((vol - v_ref).abs() < 1e-3),
            BooleanOp::Intersect => assert!(vol < 1e-3),
        }
    }
}

#[test]
fn kissing_tangent_cylinders_and_boxes_are_strictly_manifold() {
    // 1. Boxes sharing an edge diagonally: [0,1]^3 and [1,2]x[1,2]x[0,1].
    let b1 = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b2 = box_mesh([1.0, 1.0, 0.0], [2.0, 2.0, 1.0]);
    for (op, want) in [
        (BooleanOp::Union, 2.0),
        (BooleanOp::Subtract, 1.0),
        (BooleanOp::Intersect, 0.0),
    ] {
        let opts = BooleanOptions {
            op,
            ..BooleanOptions::default()
        };
        let (out, _) = boolean(&b1, &b2, Rigid::IDENTITY, &opts).unwrap();
        let vol = assert_strict_manifold(&out, &format!("edge-kissing boxes {op:?}"));
        assert!((vol - want).abs() < 1e-4, "{op:?}: got {vol}, want {want}");
    }

    // 2. Internally tangent cylinders: R=1.0 at (0,0) vs R=0.5 shifted by +0.5 along X
    //    (kissing along vertical tangent generator line x=1.0, y=0 + flush coplanar caps z = +/-1).
    let c_big = cylinder(1.0, 1.0, 24);
    let c_small = cylinder(0.5, 1.0, 20);
    let v_big = measure(&c_big).unwrap().volume.unwrap();
    let v_small = measure(&c_small).unwrap().volume.unwrap();
    let xf_tangent_in = Rigid {
        translation: [0.5, 0.0, 0.0],
        ..Rigid::IDENTITY
    };
    for (op, want) in [
        (BooleanOp::Union, v_big),
        (BooleanOp::Intersect, v_small),
        (BooleanOp::Subtract, v_big - v_small),
    ] {
        let opts = BooleanOptions {
            op,
            ..BooleanOptions::default()
        };
        let (out, _) = boolean(&c_big, &c_small, xf_tangent_in, &opts).unwrap();
        let vol = assert_strict_manifold(&out, &format!("internally tangent cylinders {op:?}"));
        assert!(
            (vol - want).abs() < 0.05,
            "internally tangent cylinders {op:?}: got {vol}, want {want}"
        );
    }

    // 3. Externally kissing tangent cylinders along x=0.5, rotated around X to avoid trivial identical mesh alignment.
    let c_a = cylinder(0.5, 1.0, 20);
    let c_b = cylinder(0.5, 1.0, 24);
    let xf_tangent_out = Rigid {
        translation: [1.0, 0.0, 0.0],
        ..Rigid::IDENTITY
    };
    let v_a = measure(&c_a).unwrap().volume.unwrap();
    let v_b = measure(&c_b).unwrap().volume.unwrap();
    for op in [BooleanOp::Union, BooleanOp::Subtract, BooleanOp::Intersect] {
        let opts = BooleanOptions {
            op,
            ..BooleanOptions::default()
        };
        let (out, _) = boolean(&c_a, &c_b, xf_tangent_out, &opts).unwrap();
        let vol = assert_strict_manifold(&out, &format!("externally tangent cylinders {op:?}"));
        match op {
            BooleanOp::Union => assert!((vol - (v_a + v_b)).abs() < 0.05),
            BooleanOp::Subtract => assert!((vol - v_a).abs() < 0.05),
            BooleanOp::Intersect => assert!(vol < 0.05),
        }
    }
}

#[test]
fn volumetric_fallbacks_handle_dirty_open_and_self_intersecting_meshes() {
    let clean = box_mesh([0.5, -0.25, -0.25], [2.5, 1.25, 1.25]);
    let open = open_box_missing_side([0.0, 0.0, 0.0], [1.5, 1.0, 1.0]);
    let self_int = self_intersecting_boxes();

    // Manifold method strictly rejects open / self-intersecting inputs.
    let manifold_opts = BooleanOptions::default();
    assert!(matches!(
        boolean(&open, &clean, Rigid::IDENTITY, &manifold_opts),
        Err(OpsError::InvalidInput(_))
    ));
    assert!(matches!(
        boolean(&self_int, &clean, Rigid::IDENTITY, &manifold_opts),
        Err(OpsError::InvalidInput(_))
    ));

    // Both volumetric fallbacks (BvhSdf and FhEdt) produce closed 2-manifold outputs.
    for method in [BooleanMethod::BvhSdf, BooleanMethod::FhEdt] {
        for op in [BooleanOp::Union, BooleanOp::Subtract, BooleanOp::Intersect] {
            let opts = BooleanOptions {
                op,
                method,
                voxel_pitch: Some(0.1),
                ..BooleanOptions::default()
            };
            let (out_open, rep_open) = boolean(&open, &clean, Rigid::IDENTITY, &opts).unwrap();
            assert_eq!(rep_open.kept_triangles, out_open.triangle_count());
            let vol_open = assert_strict_manifold(&out_open, &format!("{method:?} open {op:?}"));
            assert!(vol_open > 0.0, "{method:?} open {op:?} empty volume");

            let (out_si, rep_si) = boolean(&self_int, &clean, Rigid::IDENTITY, &opts).unwrap();
            assert_eq!(rep_si.kept_triangles, out_si.triangle_count());
            let vol_si =
                assert_strict_manifold(&out_si, &format!("{method:?} self-intersecting {op:?}"));
            assert!(
                vol_si > 0.0,
                "{method:?} self-intersecting {op:?} empty volume"
            );
        }

        // Disjoint Intersect and enclosing Subtract yield valid closed empty meshes.
        let b0 = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b_far = box_mesh([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]);
        let b_big = box_mesh([-1.0, -1.0, -1.0], [2.0, 2.0, 2.0]);

        let (disjoint, _) = boolean(
            &b0,
            &b_far,
            Rigid::IDENTITY,
            &BooleanOptions {
                op: BooleanOp::Intersect,
                method,
                voxel_pitch: Some(0.15),
                ..BooleanOptions::default()
            },
        )
        .unwrap();
        assert_eq!(disjoint.triangle_count(), 0);
        assert_strict_manifold(&disjoint, &format!("{method:?} disjoint intersect"));

        let (encl_sub, _) = boolean(
            &b0,
            &b_big,
            Rigid::IDENTITY,
            &BooleanOptions {
                op: BooleanOp::Subtract,
                method,
                voxel_pitch: Some(0.15),
                ..BooleanOptions::default()
            },
        )
        .unwrap();
        assert_eq!(encl_sub.triangle_count(), 0);
        assert_strict_manifold(&encl_sub, &format!("{method:?} enclosed subtract"));
    }
}

#[test]
fn snap_slivers_and_max_triangles_cap_work_on_seam_slivers() {
    // Shift mesh B by a thin sliver offset (2e-4) in X/Y relative to A to generate
    // needle/sliver triangles along the boolean seam.
    let a = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = box_mesh([0.0, 0.0, 0.5], [1.0, 1.0, 1.5]);
    let sliver_xf = Rigid {
        translation: [2e-4, 2e-4, 0.0],
        ..Rigid::IDENTITY
    };

    let (no_snap, _) = boolean(
        &a,
        &b,
        sliver_xf,
        &BooleanOptions {
            op: BooleanOp::Union,
            method: BooleanMethod::Manifold,
            coplanar_tolerance: 1e-3,
            snap_slivers: false,
            ..BooleanOptions::default()
        },
    )
    .unwrap();
    let (snapped, _) = boolean(
        &a,
        &b,
        sliver_xf,
        &BooleanOptions {
            op: BooleanOp::Union,
            method: BooleanMethod::Manifold,
            coplanar_tolerance: 1e-3,
            snap_slivers: true,
            ..BooleanOptions::default()
        },
    )
    .unwrap();

    let vol_no_snap = assert_strict_manifold(&no_snap, "sliver no_snap");
    let vol_snapped = assert_strict_manifold(&snapped, "sliver snapped");
    assert!((vol_snapped - vol_no_snap).abs() < 1e-2);
    assert!(
        min_aspect_ratio(&snapped) >= min_aspect_ratio(&no_snap)
            || snapped.triangle_count() <= no_snap.triangle_count(),
        "snap_slivers did not improve aspect or reduce sliver triangles: min_ar {} -> {}, tris {} -> {}",
        min_aspect_ratio(&no_snap),
        min_aspect_ratio(&snapped),
        no_snap.triangle_count(),
        snapped.triangle_count()
    );

    // Test max_triangles cap via both Manifold and BvhSdf on spheres.
    let s1 = sphere(1.0, 14, 24);
    let s2 = sphere(1.0, 14, 24);
    let xf = Rigid {
        translation: [0.6, 0.0, 0.0],
        ..Rigid::IDENTITY
    };
    for method in [BooleanMethod::Manifold, BooleanMethod::BvhSdf] {
        let cap = 250;
        let opts = BooleanOptions {
            op: BooleanOp::Union,
            method,
            voxel_pitch: Some(0.12),
            max_triangles: Some(cap),
            ..BooleanOptions::default()
        };
        let (capped, rep) = boolean(&s1, &s2, xf, &opts).unwrap();
        assert_eq!(rep.kept_triangles, capped.triangle_count());
        assert!(
            capped.triangle_count() <= cap as usize,
            "{method:?}: got {} tris > cap {cap}",
            capped.triangle_count()
        );
        let vol = assert_strict_manifold(&capped, &format!("{method:?} max_triangles={cap}"));
        assert!(vol > 4.0, "{method:?} capped volume unexpected: {vol}");

        // Also verify apply_boolean_with_options honours cap cleanly.
        let mut target = clone_mesh(&s1);
        let mut tx = target.begin_edit();
        let rep_tx = apply_boolean_with_options(&mut tx, &s2, xf, &opts).unwrap();
        let _ = tx.commit().unwrap();
        assert_eq!(rep_tx.kept_triangles, target.triangle_count());
        assert!(target.triangle_count() <= cap as usize);
        assert_strict_manifold(&target, &format!("{method:?} tx max_triangles={cap}"));
    }
}

#[test]
fn csg_batch_nary_matches_pairwise_topology_and_volume() {
    assert!(CsgNode::reduce_all(BooleanOp::Union, 0).is_none());
    assert!(matches!(
        CsgNode::reduce_all(BooleanOp::Union, 1),
        Some(CsgNode::Leaf(0))
    ));

    // 4 axis-aligned unit boxes in an overlapping chain `[k*0.5..k*0.5+1] x [0..1] x [0..1]`.
    // Union volume = 1 + 3 * 0.5 = 2.5.
    let b = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let inputs = [
        (
            &b,
            Rigid {
                translation: [0.0, 0.0, 0.0],
                ..Rigid::IDENTITY
            },
        ),
        (
            &b,
            Rigid {
                translation: [0.5, 0.0, 0.0],
                ..Rigid::IDENTITY
            },
        ),
        (
            &b,
            Rigid {
                translation: [1.0, 0.0, 0.0],
                ..Rigid::IDENTITY
            },
        ),
        (
            &b,
            Rigid {
                translation: [1.5, 0.0, 0.0],
                ..Rigid::IDENTITY
            },
        ),
    ];

    let tree_union = CsgNode::reduce_all(BooleanOp::Union, 4).unwrap();
    for method in [
        BooleanMethod::Manifold,
        BooleanMethod::BvhSdf,
        BooleanMethod::FhEdt,
    ] {
        let opts = BooleanOptions {
            method,
            voxel_pitch: Some(0.08),
            ..BooleanOptions::default()
        };
        let out = boolean_batch(&inputs, &tree_union, &opts).unwrap();
        let vol = assert_strict_manifold(&out, &format!("batch 4-union {method:?}"));
        let tol = if method == BooleanMethod::Manifold {
            1e-4
        } else {
            0.35
        };
        assert!(
            (vol - 2.5).abs() < tol,
            "batch 4-union {method:?}: vol={vol}, want 2.5"
        );
    }

    // Left-heavy Subtract reduction vs sequential pairwise: B0 \ B1 \ B2.
    // B0 = [0..2, 0..1, 0..1] (vol 2.0), B1 = [0..0.5], B2 = [1.5..2.0] -> remaining [0.5..1.5], vol = 1.0.
    let base = box_mesh([0.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
    let cut1 = box_mesh([0.0, 0.0, 0.0], [0.5, 1.0, 1.0]);
    let cut2 = box_mesh([1.5, 0.0, 0.0], [2.0, 1.0, 1.0]);
    // Also exercise repeated Leaf(1) reference `(base \ cut1) \ (cut2 | cut1)`.
    let mixed_tree = CsgNode::Op {
        op: BooleanOp::Subtract,
        left: Box::new(CsgNode::Op {
            op: BooleanOp::Subtract,
            left: Box::new(CsgNode::Leaf(0)),
            right: Box::new(CsgNode::Leaf(1)),
        }),
        right: Box::new(CsgNode::Op {
            op: BooleanOp::Union,
            left: Box::new(CsgNode::Leaf(2)),
            right: Box::new(CsgNode::Leaf(1)),
        }),
    };
    let sub_inputs = [
        (&base, Rigid::IDENTITY),
        (&cut1, Rigid::IDENTITY),
        (&cut2, Rigid::IDENTITY),
    ];
    for method in [
        BooleanMethod::Manifold,
        BooleanMethod::BvhSdf,
        BooleanMethod::FhEdt,
    ] {
        let opts = BooleanOptions {
            method,
            voxel_pitch: Some(0.08),
            ..BooleanOptions::default()
        };
        let out = boolean_batch(&sub_inputs, &mixed_tree, &opts).unwrap();
        let vol = assert_strict_manifold(&out, &format!("batch mixed {method:?}"));
        let tol = if method == BooleanMethod::Manifold {
            1e-4
        } else {
            0.3
        };
        assert!(
            (vol - 1.0).abs() < tol,
            "batch mixed {method:?}: vol={vol}, want 1.0"
        );
    }

    // Validation errors: out-of-range leaf index & non-finite transform.
    let bad_tree = CsgNode::Leaf(99);
    assert!(matches!(
        boolean_batch(&sub_inputs, &bad_tree, &BooleanOptions::default()),
        Err(OpsError::InvalidInput(_))
    ));
    let bad_inputs = [(
        &base,
        Rigid {
            translation: [f64::NAN, 0.0, 0.0],
            ..Rigid::IDENTITY
        },
    )];
    assert!(matches!(
        boolean_batch(&bad_inputs, &CsgNode::Leaf(0), &BooleanOptions::default()),
        Err(OpsError::Query(_) | OpsError::InvalidInput(_))
    ));
}

#[test]
fn parallel_and_sequential_boolean_are_bit_identical() {
    struct SeqGuard(bool);
    impl Drop for SeqGuard {
        fn drop(&mut self) {
            if self.0 {
                par::set_force_sequential(false);
            }
        }
    }

    let s1 = sphere(1.0, 12, 20);
    let s2 = sphere(0.9, 12, 20);
    let xf1 = Rigid {
        translation: [0.55, 0.15, -0.1],
        ..Rigid::IDENTITY
    };
    let xf2 = Rigid {
        translation: [-0.45, -0.2, 0.15],
        ..Rigid::IDENTITY
    };

    for method in [
        BooleanMethod::Manifold,
        BooleanMethod::BvhSdf,
        BooleanMethod::FhEdt,
    ] {
        let opts = BooleanOptions {
            op: BooleanOp::Union,
            method,
            voxel_pitch: Some(0.14),
            ..BooleanOptions::default()
        };

        let (par_out, _) = par::with_threads(4, || boolean(&s1, &s2, xf1, &opts)).unwrap();
        let (seq_out, _) = par::with_threads(1, || {
            par::set_force_sequential(true);
            let _guard = SeqGuard(true);
            boolean(&s1, &s2, xf1, &opts)
        })
        .unwrap();
        assert_meshes_bit_identical(&par_out, &seq_out, &format!("binary {method:?}"));

        let tree = CsgNode::reduce_all(BooleanOp::Union, 3).unwrap();
        let batch_inputs = [(&s1, Rigid::IDENTITY), (&s2, xf1), (&s2, xf2)];
        let par_batch =
            par::with_threads(4, || boolean_batch(&batch_inputs, &tree, &opts)).unwrap();
        let seq_batch = par::with_threads(1, || {
            par::set_force_sequential(true);
            let _guard = SeqGuard(true);
            boolean_batch(&batch_inputs, &tree, &opts)
        })
        .unwrap();
        assert_meshes_bit_identical(&par_batch, &seq_batch, &format!("batch {method:?}"));
    }
}
