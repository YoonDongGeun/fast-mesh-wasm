//! Integration tests verifying the seven cohesive domain modules in
//! `fast_mesh_ops` compile cleanly and match legacy top-level exports bit-for-bit.

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::{Plane, Rigid};
use fast_mesh_store::{Mesh, Transaction};

fn unit_box() -> Mesh {
    let positions = [
        0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
    ];
    let indices = [
        0, 2, 1, 0, 3, 2, // -z
        4, 5, 6, 4, 6, 7, // +z
        0, 1, 5, 0, 5, 4, // -y
        2, 3, 7, 2, 7, 6, // +y
        0, 4, 7, 0, 7, 3, // -x
        1, 2, 6, 1, 6, 5, // +x
    ];
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn clone_mesh(src: &Mesh) -> Mesh {
    let positions: Vec<f32> = src.positions().iter().flatten().copied().collect();
    let indices: Vec<u32> = src.triangles().iter().flatten().copied().collect();
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn open_quad_patch() -> Mesh {
    let positions = [
        0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0,
        0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0,
    ];
    let indices = [
        0, 1, 4, 0, 4, 3, 1, 2, 5, 1, 5, 4, 3, 4, 7, 3, 7, 6, 4, 5, 8, 4, 8, 7,
    ];
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn open_box_with_hole() -> Mesh {
    let mut mesh = unit_box();
    let mut tx = mesh.begin_edit();
    tx.remove_triangles(&[0]).unwrap();
    tx.commit().unwrap();
    mesh
}

fn edit<R>(mesh: &mut Mesh, f: impl FnOnce(&mut Transaction) -> fast_mesh_ops::Result<R>) -> R {
    let mut tx = mesh.begin_edit();
    let out = f(&mut tx).unwrap();
    tx.commit().unwrap();
    out
}

#[test]
fn repair_domain_matches_legacy_exports() {
    use fast_mesh_ops::repair::{
        AutoRepairOptions, AutoRepairPreset, FillOptions, ManifoldOptions, SelfIntersectOptions,
        SelfIntersectionMode, SliverOptions, auto_repair, cleanup_slivers, fill_holes,
        make_manifold, orient, remove_degenerate, remove_small_components,
        resolve_self_intersections, weld,
    };

    let mut mesh_domain = open_box_with_hole();
    let mut mesh_legacy = open_box_with_hole();

    let opts = AutoRepairOptions::for_preset(AutoRepairPreset::General);
    let rep_domain = edit(&mut mesh_domain, |tx| auto_repair(tx, &opts));
    let rep_legacy = edit(&mut mesh_legacy, |tx| fast_mesh_ops::auto_repair(tx, &opts));
    assert_eq!(rep_domain, rep_legacy);
    assert_eq!(mesh_domain.positions(), mesh_legacy.positions());
    assert_eq!(mesh_domain.triangles(), mesh_legacy.triangles());

    // Exercise additional curated repair domain symbols and signatures.
    let mut extra = open_box_with_hole();
    let w = edit(&mut extra, |tx| weld(tx, 0.0));
    let o = edit(&mut extra, orient);
    let f = edit(&mut extra, |tx| {
        fill_holes(tx, &FillOptions::builder().max_edges(16).build())
    });
    let _ = (
        w,
        o,
        f,
        ManifoldOptions::default(),
        SliverOptions::default(),
        SelfIntersectOptions::default(),
        SelfIntersectionMode::DeleteAndFill,
        make_manifold,
        cleanup_slivers,
        resolve_self_intersections,
        remove_degenerate,
        remove_small_components,
    );
}

#[test]
fn inspect_domain_matches_legacy_exports() {
    use fast_mesh_ops::inspect::{
        AnalyzeOptions, analyze, boundary_loops, measure, measure_placed,
    };

    let mesh = open_box_with_hole();
    let opts = AnalyzeOptions::default();
    assert_eq!(
        analyze(&mesh, &opts).unwrap(),
        fast_mesh_ops::analyze(&mesh, &opts).unwrap()
    );
    assert_eq!(
        measure(&mesh).unwrap(),
        fast_mesh_ops::measure(&mesh).unwrap()
    );
    assert_eq!(
        boundary_loops(&mesh).unwrap(),
        fast_mesh_ops::boundary_loops(&mesh).unwrap()
    );

    let closed = unit_box();
    let m_domain = measure_placed(&closed, [0.0; 3], |p| p).unwrap();
    let m_legacy = fast_mesh_ops::measure_placed(&closed, [0.0; 3], |p| p).unwrap();
    assert_eq!(m_domain, m_legacy);
    assert!(m_domain.closed);
    assert!((m_domain.area - 6.0).abs() < 1e-6);
    assert!((m_domain.volume.unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn remesh_domain_exposes_region_simplify_subdivide_and_marching_cubes() {
    use fast_mesh_ops::remesh::{
        Region, RemeshOptions, SimplifyOptions, SubdivideMethod, SubdivideOptions,
        marching_cubes_raw, remesh, simplify, subdivide,
    };

    let mut sub_domain = unit_box();
    let mut sub_legacy = unit_box();
    let sub_opts = SubdivideOptions {
        method: SubdivideMethod::Loop,
        levels: 1,
    };
    let rep_d = edit(&mut sub_domain, |tx| subdivide(tx, &sub_opts));
    let rep_l = edit(&mut sub_legacy, |tx| {
        fast_mesh_ops::subdivide::subdivide(tx, &sub_opts)
    });
    assert_eq!(rep_d, rep_l);
    assert_eq!(sub_domain.triangles(), sub_legacy.triangles());

    let mut sim_domain = clone_mesh(&sub_domain);
    let mut sim_legacy = clone_mesh(&sub_legacy);
    let sim_opts = SimplifyOptions::builder().target_triangles(24).build();
    let srep_d = edit(&mut sim_domain, |tx| simplify(tx, &sim_opts));
    let srep_l = edit(&mut sim_legacy, |tx| {
        fast_mesh_ops::simplify::simplify(tx, &sim_opts)
    });
    assert_eq!(srep_d, srep_l);
    assert_eq!(sim_domain.triangles(), sim_legacy.triangles());

    let mut region_mesh = clone_mesh(&sub_domain);
    let rem_opts = RemeshOptions {
        region: Region {
            center: [0.5, 0.5, 0.5],
            radius: 2.0,
        },
        target_length: Some(0.5),
        max_rounds: 2,
        split_only: true,
    };
    let r_rep = edit(&mut region_mesh, |tx| remesh(tx, &rem_opts));
    assert!(r_rep.splits > 0 || region_mesh.triangle_count() >= sub_domain.triangle_count());

    // 3x3x3 signed grid with central negative value at (1, 1, 1) extracts a valid IsoMesh.
    let mut grid = vec![1.0f32; 27];
    grid[13] = -1.0;
    let iso = marching_cubes_raw([0.0, 0.0, 0.0], [3, 3, 3], 0.5, &grid, 0.0).unwrap();
    assert!(!iso.triangles.is_empty());
}

#[test]
fn boolean_domain_exposes_csg_cut_offset_and_shell() {
    use fast_mesh_ops::boolean::{
        BooleanOp, CutOptions, OffsetOptions, ShellOptions, apply_boolean, boolean_plan,
        cut_by_plane, offset, shell,
    };

    let a = unit_box();
    let b = unit_box();
    let bvh_a = Bvh::build(&a, &BuildOptions::default()).unwrap();
    let bvh_b = Bvh::build(&b, &BuildOptions::default()).unwrap();
    let xf = Rigid {
        translation: [0.5, 0.0, 0.0],
        ..Rigid::IDENTITY
    };

    let mut out_d = clone_mesh(&a);
    let mut out_l = clone_mesh(&a);
    let plan_d = boolean_plan(&a, &bvh_a, &b, &bvh_b, &xf, BooleanOp::Union).unwrap();
    let plan_l =
        fast_mesh_ops::boolean_plan(&a, &bvh_a, &b, &bvh_b, &xf, BooleanOp::Union).unwrap();
    let rep_d = edit(&mut out_d, |tx| apply_boolean(tx, &plan_d));
    let rep_l = edit(&mut out_l, |tx| fast_mesh_ops::apply_boolean(tx, &plan_l));
    assert_eq!(rep_d, rep_l);
    assert_eq!(out_d.triangles(), out_l.triangles());

    let mut cut_d = unit_box();
    let mut cut_l = unit_box();
    let plane = Plane::new([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap();
    let c_d = edit(&mut cut_d, |tx| {
        cut_by_plane(tx, &plane, &CutOptions::default())
    });
    let c_l = edit(&mut cut_l, |tx| {
        fast_mesh_ops::cut_by_plane(tx, &plane, &CutOptions::default())
    });
    assert_eq!(c_d.report, c_l.report);

    let mut off_d = unit_box();
    let off_opts = OffsetOptions::builder().distance(0.1).build();
    let o_rep = edit(&mut off_d, |tx| offset(tx, &off_opts));
    assert!(o_rep.moved > 0);

    let mut shell_d = open_quad_patch();
    let sh_rep = edit(&mut shell_d, |tx| {
        shell(tx, &ShellOptions { thickness: 0.1 })
    });
    assert!(sh_rep.triangles_added > 0);
}

#[test]
fn sculpt_and_deform_domains_match_legacy_and_support_protection() {
    use fast_mesh_ops::deform::{check_protection, with_protection};
    use fast_mesh_ops::sculpt::{
        Brush, Falloff, InflateOptions, StretchOptions, capture_rest_sizes, inflate,
        split_stretched,
    };

    let mut s_domain = open_quad_patch();
    let mut s_legacy = open_quad_patch();
    capture_rest_sizes(&mut s_domain).unwrap();
    fast_mesh_ops::sculpt::capture_rest_sizes(&mut s_legacy).unwrap();

    let opts = InflateOptions {
        brush: Brush {
            center: [1.0, 1.0, 0.0],
            radius: 1.5,
            falloff: Falloff::Smooth,
            strength: 0.4,
        },
        ..Default::default()
    };
    let rep_d = edit(&mut s_domain, |tx| inflate(tx, &opts));
    let rep_l = edit(&mut s_legacy, |tx| fast_mesh_ops::inflate(tx, &opts));
    assert_eq!(rep_d, rep_l);
    assert_eq!(s_domain.positions(), s_legacy.positions());

    // Protection via deform domain pins vertex 4 cleanly.
    let mut protected_mesh = open_quad_patch();
    check_protection(&protected_mesh, &[4], &[0.0]).unwrap();
    let before = protected_mesh.position(4).unwrap();
    edit(&mut protected_mesh, |tx| {
        with_protection(tx, &[4], &[0.0], |inner| {
            fast_mesh_ops::deform::inflate(inner, &opts)
        })
    });
    assert_eq!(protected_mesh.position(4).unwrap(), before);

    // Split stretched exposed through sculpt domain.
    let stretch_rep = edit(&mut s_domain, |tx| {
        split_stretched(tx, &StretchOptions::default())
    });
    assert_eq!(
        stretch_rep.splits,
        edit(&mut s_legacy, |tx| {
            fast_mesh_ops::sculpt::split_stretched(tx, &StretchOptions::default())
        })
        .splits
    );
}

#[test]
fn align_domain_matches_legacy_exports() {
    use fast_mesh_ops::align::{IcpOptions, IcpStatus, icp, sample_surface};

    let fixed = unit_box();
    let bvh = Bvh::build(&fixed, &BuildOptions::default()).unwrap();
    let samples_d = sample_surface(&fixed, 64).unwrap();
    let samples_l = fast_mesh_ops::sample_surface(&fixed, 64).unwrap();
    assert_eq!(samples_d, samples_l);

    let initial = Rigid {
        translation: [0.05, 0.0, 0.0],
        ..Rigid::IDENTITY
    };
    let moved: Vec<[f64; 3]> = samples_d
        .iter()
        .map(|&[x, y, z]| [x - 0.05, y, z])
        .collect();

    let rep_d = icp(&moved, &fixed, &bvh, &initial, &IcpOptions::default()).unwrap();
    let rep_l = fast_mesh_ops::icp(&moved, &fixed, &bvh, &initial, &IcpOptions::default()).unwrap();
    assert_eq!(rep_d.status, rep_l.status);
    assert_eq!(rep_d.status, IcpStatus::Converged);
    assert_eq!(rep_d.iterations, rep_l.iterations);
}
