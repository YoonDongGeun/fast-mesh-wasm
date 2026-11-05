//! Tests of the table in `docs/uv-channel.md`: one per row, each asserting
//! what the row says about UVs. Scene-level rows (`join`, `cut_by_plane`,
//! `boolean`, undo) are tested in `fast-mesh-scene`.

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::{Plane, Rigid};
use fast_mesh_store::{Mesh, VertexId};

use crate::bridge::BridgeOptions;
use crate::deform::{Brush, Falloff, GrabOptions, InflateOptions, SmoothMethod, SmoothOptions};
use crate::remesh::{Region, RemeshOptions};
use crate::simplify::SimplifyOptions;
use crate::subdivide::{SubdivideMethod, SubdivideOptions};
use crate::test_util::{box_mesh, edit, grid, grid_without, hemisphere, mesh_of, uv_sphere};
use crate::{
    AppendSource, BooleanOp, ComponentOptions, CutOptions, DegenerateOptions, FillOptions,
    OffsetOptions, ShellOptions, append_mesh, append_mesh_with_uvs, apply_boolean, boolean_plan,
    boundary_loops, bridge_loops, cut_by_plane, fill_holes, inflate, offset, orient,
    remove_degenerate, remove_small_components, shell, weld,
};

/// A planar UV of a position: linear, so anything interpolated stays on it.
fn planar(p: [f32; 3]) -> [f32; 2] {
    [0.25 * p[0] + 0.5, 0.25 * p[1] - 0.125]
}

/// `mesh` with [`planar`] UVs.
fn with_planar(mut mesh: Mesh) -> Mesh {
    let uvs: Vec<f32> = mesh.positions().iter().flat_map(|&p| planar(p)).collect();
    mesh.add_uvs(&uvs).unwrap();
    mesh
}

fn uv_of(mesh: &Mesh, v: VertexId) -> [f32; 2] {
    let u = mesh.uvs().unwrap();
    [u[2 * v as usize], u[2 * v as usize + 1]]
}

fn alive(mesh: &Mesh) -> Vec<VertexId> {
    (0..mesh.vertex_slots() as VertexId)
        .filter(|&v| mesh.is_vertex_alive(v))
        .collect()
}

/// Every live vertex's UV is [`planar`] of its current position, within `tol`:
/// the invariant of an operation that interpolates a linear mapping and does
/// not move what it keeps.
fn assert_planar(mesh: &Mesh, tol: f32) {
    assert_eq!(mesh.uvs().unwrap().len(), 2 * mesh.vertex_slots());
    for v in alive(mesh) {
        let want = planar(mesh.position(v).unwrap());
        let got = uv_of(mesh, v);
        assert!(
            (want[0] - got[0]).abs() <= tol && (want[1] - got[1]).abs() <= tol,
            "vertex {v}: uv {got:?}, want {want:?}"
        );
    }
}

/// The UVs of the vertices that existed before, unchanged.
fn assert_kept(mesh: &Mesh, before: &Mesh) {
    for v in alive(mesh) {
        if (v as usize) < before.vertex_slots() && before.is_vertex_alive(v) {
            assert_eq!(uv_of(mesh, v), uv_of(before, v), "vertex {v}");
        }
    }
}

/// A copy of `mesh`, for comparing (positions, triangles and UVs).
fn copy(mesh: &Mesh) -> Mesh {
    let positions: Vec<f32> = mesh.positions().iter().flatten().copied().collect();
    let triangles: Vec<[u32; 3]> = (0..mesh.triangle_slots() as u32)
        .filter_map(|t| mesh.triangle(t))
        .collect();
    let mut out = mesh_of(&positions, &triangles);
    out.add_uvs(mesh.uvs().unwrap()).unwrap();
    out
}

/// Two unit quads side by side along `x = 1`, with split vertices along the
/// shared edge: slots 4 and 5 duplicate slots 1 and 2. With `seam` the two
/// sides' UVs differ there, as at a UV seam.
fn seam_quads(seam: bool) -> Mesh {
    let mut m = mesh_of(
        &[
            0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., // left
            1., 0., 0., 1., 1., 0., 2., 0., 0., 2., 1., 0., // right
        ],
        &[[0, 1, 2], [0, 2, 3], [4, 6, 7], [4, 7, 5]],
    );
    let u = if seam { 0.75 } else { 0.5 };
    m.add_uvs(&[
        0., 0., 0.5, 0., 0.5, 1., 0., 1., //
        u, 0., u, 1., 1., 0., 1., 1.,
    ])
    .unwrap();
    m
}

// ---- weld

#[test]
fn weld_keeps_a_uv_seam_apart() {
    let mut m = seam_quads(true);
    let (r, _) = edit(&mut m, |tx| weld(tx, 0.01));
    assert_eq!(r.merged_vertices, 0);
    assert_eq!(m.vertex_count(), 8);
    assert_eq!(uv_of(&m, 1), [0.5, 0.]);
    assert_eq!(uv_of(&m, 4), [0.75, 0.]);
}

#[test]
fn weld_merges_coincident_vertices_with_equal_uvs() {
    let mut m = seam_quads(false);
    let (r, _) = edit(&mut m, |tx| weld(tx, 0.01));
    assert_eq!(r.merged_vertices, 2);
    assert_eq!(m.vertex_count(), 6);
    assert_eq!(uv_of(&m, 1), [0.5, 0.]);
    assert_eq!(m.uvs().unwrap().len(), 2 * m.vertex_slots());
}

#[test]
fn weld_keeps_the_seam_along_a_position_chain() {
    // Three coincident vertices: the outer two share a UV, the middle one
    // does not. Only the matching pair merges.
    let mut m = mesh_of(
        &[
            0., 0., 0., 0., 0., 0., 0., 0., 0., // the three
            1., 0., 0., 0., 1., 0., 1., 1., 0.,
        ],
        &[[0, 3, 4], [1, 3, 5], [2, 4, 5]],
    );
    m.add_uvs(&[0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0., 0., 0., 0., 0., 0.])
        .unwrap();
    let (r, _) = edit(&mut m, |tx| weld(tx, 0.0));
    assert_eq!(r.merged_vertices, 1);
    assert!(m.is_vertex_alive(0) && m.is_vertex_alive(1) && !m.is_vertex_alive(2));
}

#[test]
fn weld_without_uvs_is_unchanged() {
    let seam = seam_quads(true);
    let positions: Vec<f32> = seam.positions().iter().flatten().copied().collect();
    let triangles: Vec<[u32; 3]> = (0..4u32).filter_map(|t| seam.triangle(t)).collect();
    let mut m = mesh_of(&positions, &triangles);
    let (r, _) = edit(&mut m, |tx| weld(tx, 0.01));
    assert_eq!(r.merged_vertices, 2);
    assert_eq!(m.vertex_count(), 6);
}

#[test]
fn analyze_still_sees_seam_vertices_as_coincident() {
    let m = seam_quads(true);
    let a = crate::analyze(
        &m,
        &crate::AnalyzeOptions {
            weld_tolerance: 0.01,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(a.coincident_groups, 2);
}

// ---- operations that remove or rewrite only

#[test]
fn remove_degenerate_keeps_the_survivors_uvs() {
    let mut m = with_planar(grid(4));
    // A zero-area triangle on three distinct vertices would need a duplicate;
    // repeat a corner instead.
    edit(&mut m, |tx| {
        tx.add_triangles(&[[0, 0, 1]]).map_err(Into::into)
    });
    let before = copy(&m);
    let (r, _) = edit(&mut m, |tx| {
        remove_degenerate(tx, &DegenerateOptions::default())
    });
    assert_eq!(r.removed_triangles, 1);
    assert_kept(&m, &before);
}

#[test]
fn remove_small_components_keeps_the_survivors_uvs() {
    let mut m = with_planar(grid(4));
    edit(&mut m, |tx| {
        let v = tx.add_vertices(&[[8., 8., 0.], [9., 8., 0.], [8., 9., 0.]])?;
        tx.add_triangles(&[[v[0], v[1], v[2]]])?;
        tx.write_channel("uv", &v, &[0.; 6])?;
        Ok(())
    });
    let before = copy(&m);
    let options = ComponentOptions {
        min_triangles: Some(2),
        ..Default::default()
    };
    let (r, _) = edit(&mut m, |tx| remove_small_components(tx, &options));
    assert_eq!(r.removed_components, 1);
    assert_kept(&m, &before);
}

#[test]
fn orient_keeps_uvs() {
    let mut m = with_planar(crate::test_util::flip(&grid(4), &[0, 3]));
    let before = copy(&m);
    let (r, _) = edit(&mut m, orient);
    assert!(r.flipped_triangles > 0);
    assert_kept(&m, &before);
}

// ---- operations that add vertices

#[test]
fn fill_holes_interpolates_new_vertices() {
    let cells: Vec<(u32, u32)> = (2..6).flat_map(|y| (2..6).map(move |x| (x, y))).collect();
    let mut m = with_planar(grid_without(8, &cells));
    let before = copy(&m);
    let options = FillOptions {
        max_edges: Some(20),
        ..FillOptions::default()
    };
    let (r, _) = edit(&mut m, |tx| fill_holes(tx, &options));
    assert!(r.filled[0].vertices_added > 0);
    assert_kept(&m, &before);
    // Flat grid, linear UVs: interpolated values lie on the mapping.
    assert_planar(&m, 1e-4);
}

#[test]
fn remesh_interpolates_splits_and_collapses() {
    let mut m = with_planar(grid(6));
    let options = RemeshOptions {
        region: Region {
            center: [3., 3., 0.],
            radius: 2.5,
        },
        target_length: Some(0.4),
        ..RemeshOptions::default()
    };
    let (r, _) = edit(&mut m, |tx| crate::remesh::remesh(tx, &options));
    assert!(r.splits > 0);
    assert_planar(&m, 1e-4);

    let mut m = with_planar(grid(6));
    let options = RemeshOptions {
        region: Region {
            center: [3., 3., 0.],
            radius: 2.5,
        },
        target_length: Some(2.0),
        ..RemeshOptions::default()
    };
    let (r, _) = edit(&mut m, |tx| crate::remesh::remesh(tx, &options));
    assert!(r.collapses > 0);
    assert_planar(&m, 1e-4);
}

#[test]
fn subdivide_takes_edge_midpoints() {
    let mut m = with_planar(grid(3));
    let before = copy(&m);
    let options = SubdivideOptions {
        method: SubdivideMethod::Loop,
        levels: 2,
    };
    let (r, _) = edit(&mut m, |tx| crate::subdivide::subdivide(tx, &options));
    assert!(r.vertices_added > 0);
    // Old vertices keep their UV, even those Loop moved.
    assert_kept(&m, &before);
    assert_eq!(m.uvs().unwrap().len(), 2 * m.vertex_slots());
    // A flat grid is a fixed surface of Loop in the interior, and the first
    // level's edge vertices are UV midpoints of their edge: all finite, none
    // left at the default.
    for v in alive(&m) {
        let uv = uv_of(&m, v);
        assert!((0.5..=1.25).contains(&uv[0]) && (-0.125..=0.625).contains(&uv[1]));
    }
    let mut m = mesh_of(&[0., 0., 0., 4., 0., 0., 0., 4., 0.], &[[0, 1, 2]]);
    m.add_uvs(&[0., 0., 1., 0., 0., 1.]).unwrap();
    let options = SubdivideOptions {
        method: SubdivideMethod::Loop,
        levels: 1,
    };
    edit(&mut m, |tx| crate::subdivide::subdivide(tx, &options));
    let mut new: Vec<[f32; 2]> = (3..6).map(|v| uv_of(&m, v)).collect();
    new.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(new, [[0., 0.5], [0.5, 0.], [0.5, 0.5]]);
}

#[test]
fn subdivide_reuses_dead_slots_with_the_right_uvs() {
    let mut m = with_planar(grid(3));
    // Free some slots: a removed triangle's unused corner.
    edit(&mut m, |tx| {
        let v = tx.add_vertices(&[[9., 9., 0.]])?;
        tx.remove_vertices(&v)?;
        Ok(())
    });
    let options = SubdivideOptions {
        method: SubdivideMethod::Loop,
        levels: 1,
    };
    let dead = m.free_vertex_slots()[0];
    let old = m.uvs().unwrap().to_vec();
    edit(&mut m, |tx| crate::subdivide::subdivide(tx, &options));
    assert_eq!(
        &m.uvs().unwrap()[..2 * dead as usize],
        &old[..2 * dead as usize]
    );
    // The reused slot holds its edge's midpoint: on a flat grid both stencils
    // give the position's midpoint, so the linear UV matches the position.
    assert!(m.is_vertex_alive(dead));
    let (want, got) = (planar(m.position(dead).unwrap()), uv_of(&m, dead));
    assert!((want[0] - got[0]).abs() < 1e-5 && (want[1] - got[1]).abs() < 1e-5);
}

#[test]
fn simplify_interpolates_survivors_along_the_collapsed_edge() {
    let mut m = with_planar(grid(12));
    let options = SimplifyOptions {
        target_triangles: 60,
        preserve_boundary: true,
    };
    let (r, _) = edit(&mut m, |tx| crate::simplify::simplify(tx, &options));
    assert!(r.collapsed > 0);
    // Flat: a survivor stays in the plane, and the linear UV follows it.
    assert_planar(&m, 1e-3);
}

#[test]
fn simplify_with_preserved_boundary_keeps_seam_vertices() {
    let mut m = with_planar(hemisphere(8, 16, 1.0));
    let boundary: Vec<u32> = boundary_loops(&m)
        .unwrap()
        .iter()
        .flat_map(|l| l.vertices.clone())
        .collect();
    let before = copy(&m);
    let options = SimplifyOptions {
        target_triangles: 40,
        preserve_boundary: true,
    };
    let (r, _) = edit(&mut m, |tx| crate::simplify::simplify(tx, &options));
    assert!(r.collapsed > 0);
    for v in boundary {
        assert_eq!(uv_of(&m, v), uv_of(&before, v));
    }
}

#[test]
fn simplify_undo_restores_uvs() {
    let mut m = with_planar(grid(8));
    let before = m.uvs().unwrap().to_vec();
    let mut tx = m.begin_edit();
    let options = SimplifyOptions {
        target_triangles: 20,
        preserve_boundary: true,
    };
    crate::simplify::simplify(&mut tx, &options).unwrap();
    let patch = tx.commit().unwrap();
    assert_ne!(m.uvs().unwrap(), &before[..]);
    patch.revert(&mut m).unwrap();
    assert_eq!(&m.uvs().unwrap()[..before.len()], &before[..]);
}

#[test]
fn shell_copies_uvs_to_the_inner_surface() {
    let mut m = with_planar(hemisphere(4, 8, 1.0));
    let n = m.vertex_slots();
    let before = copy(&m);
    let options = ShellOptions { thickness: 0.05 };
    let (r, _) = edit(&mut m, |tx| shell(tx, &options));
    assert_eq!(r.vertices_added, n);
    assert_kept(&m, &before);
    for k in 0..n as u32 {
        assert_eq!(uv_of(&m, n as u32 + k), uv_of(&before, k));
    }
}

#[test]
fn bridge_keeps_uvs() {
    // Two parallel quads, facing each other.
    let mut m = mesh_of(
        &[
            0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., //
            0., 0., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1.,
        ],
        &[[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]],
    );
    m.add_uvs(&[0.5; 16]).unwrap();
    let before = copy(&m);
    let loops = boundary_loops(&m).unwrap();
    let options = BridgeOptions {
        loops: (loops[0].id, loops[1].id),
        max_edges: None,
    };
    edit(&mut m, |tx| bridge_loops(tx, &options));
    assert_eq!(m.vertex_count(), 8, "no vertex is created");
    assert_kept(&m, &before);
}

// ---- operations that only move vertices

fn brush() -> Brush {
    Brush {
        center: [3., 3., 0.],
        radius: 2.,
        falloff: Falloff::Smooth,
        strength: 0.3,
    }
}

#[test]
fn inflate_smooth_grab_and_offset_keep_uvs_on_their_vertices() {
    let check = |run: &dyn Fn(&mut fast_mesh_store::Transaction) -> crate::Result<()>| {
        let mut m = with_planar(uv_sphere(8, 12, 3.));
        let before = copy(&m);
        edit(&mut m, |tx| run(tx));
        assert_kept(&m, &before);
    };
    check(&|tx| {
        let options = InflateOptions { brush: brush() };
        inflate(tx, &options).map(|_| ())
    });
    check(&|tx| {
        let options = SmoothOptions {
            brush: brush(),
            cycles: 2,
            method: SmoothMethod::default(),
        };
        crate::deform::smooth(tx, &options).map(|_| ())
    });
    check(&|tx| {
        let options = GrabOptions {
            brush: brush(),
            direction: [0., 0., 1.],
        };
        crate::deform::grab(tx, &options).map(|_| ())
    });
    check(&|tx| offset(tx, &OffsetOptions { distance: 0.1 }).map(|_| ()));
}

// ---- cut

#[test]
fn cut_interpolates_crossing_vertices_on_both_sides() {
    let mut m = with_planar(grid(4));
    let plane = Plane::new([1.5, 0., 0.], [1., 0., 0.]).unwrap();
    let (out, _) = edit(&mut m, |tx| {
        cut_by_plane(tx, &plane, &CutOptions { cap: false })
    });
    assert_planar(&m, 1e-5);
    assert_planar(&out.removed, 1e-5);
    assert!(out.removed.triangle_count() > 0);
}

#[test]
fn cut_caps_keep_uvs_aligned_on_both_sides() {
    let mut m = with_planar(box_mesh([0., 0., 0.], [2., 2., 2.]));
    let plane = Plane::new([1., 0., 0.], [1., 0., 0.]).unwrap();
    let (out, _) = edit(&mut m, |tx| {
        cut_by_plane(tx, &plane, &CutOptions { cap: true })
    });
    assert!(out.report.cap_triangles > 0);
    for mesh in [&m, &out.removed] {
        assert_eq!(mesh.uvs().unwrap().len(), 2 * mesh.vertex_slots());
    }
    assert_planar(&m, 1e-4);
    assert_planar(&out.removed, 1e-4);
}

// ---- boolean

#[test]
fn boolean_drops_uvs_and_undo_restores_them() {
    let mut a = with_planar(box_mesh([0., 0., 0.], [1., 1., 1.]));
    let b = box_mesh([0.5, 0.25, 0.25], [1.5, 0.75, 0.75]);
    let options = BuildOptions::default();
    let (ba, bb) = (
        Bvh::build(&a, &options).unwrap(),
        Bvh::build(&b, &options).unwrap(),
    );
    let plan = boolean_plan(&a, &ba, &b, &bb, &Rigid::IDENTITY, BooleanOp::Union).unwrap();
    let before = a.uvs().unwrap().to_vec();
    let mut tx = a.begin_edit();
    let report = apply_boolean(&mut tx, &plan).unwrap();
    assert!(report.uvs_dropped);
    assert!(tx.mesh().uvs().is_none());
    let patch = tx.commit().unwrap();
    assert!(a.uvs().is_none(), "never stale: the channel is gone");
    patch.revert(&mut a).unwrap();
    assert_eq!(&a.uvs().unwrap()[..before.len()], &before[..]);
    patch.apply(&mut a).unwrap();
    assert!(a.uvs().is_none());
}

#[test]
fn boolean_without_uvs_reports_no_drop() {
    let mut a = box_mesh([0., 0., 0.], [1., 1., 1.]);
    let b = box_mesh([0.5, 0.25, 0.25], [1.5, 0.75, 0.75]);
    let options = BuildOptions::default();
    let (ba, bb) = (
        Bvh::build(&a, &options).unwrap(),
        Bvh::build(&b, &options).unwrap(),
    );
    let plan = boolean_plan(&a, &ba, &b, &bb, &Rigid::IDENTITY, BooleanOp::Union).unwrap();
    let (r, _) = edit(&mut a, |tx| apply_boolean(tx, &plan));
    assert!(!r.uvs_dropped);
}

// ---- append

fn triangle_source() -> AppendSource {
    AppendSource {
        positions: vec![[5., 5., 0.], [6., 5., 0.], [5., 6., 0.]],
        triangles: vec![[0, 1, 2]],
    }
}

#[test]
fn append_with_uvs_on_both_sides_keeps_them() {
    let mut m = with_planar(grid(2));
    let (r, _) = edit(&mut m, |tx| {
        append_mesh_with_uvs(
            tx,
            &triangle_source(),
            Some(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        )
    });
    assert!(!r.uvs_dropped);
    assert_eq!(uv_of(&m, r.vertex_map[1]), [0.3, 0.4]);
}

#[test]
fn append_without_source_uvs_drops_the_targets() {
    let mut m = with_planar(grid(2));
    let (r, _) = edit(&mut m, |tx| append_mesh(tx, &triangle_source()));
    assert!(r.uvs_dropped);
    assert!(m.uvs().is_none());
}

#[test]
fn append_onto_a_target_without_uvs_reports_the_sources_lost() {
    let mut m = grid(2);
    let (r, _) = edit(&mut m, |tx| {
        append_mesh_with_uvs(tx, &triangle_source(), Some(&[0.; 6]))
    });
    assert!(r.uvs_dropped);
    assert!(m.uvs().is_none());

    let mut m = grid(2);
    let (r, _) = edit(&mut m, |tx| append_mesh(tx, &triangle_source()));
    assert!(!r.uvs_dropped);
}

#[test]
fn append_rejects_a_wrong_uv_length() {
    let mut m = with_planar(grid(2));
    let mut tx = m.begin_edit();
    assert!(append_mesh_with_uvs(&mut tx, &triangle_source(), Some(&[0.; 5])).is_err());
}

// ---- parallel

#[test]
fn a_large_seamed_weld_is_identical_in_parallel() {
    // Past the parallel threshold: a soup of coincident vertices along a
    // line, with a seam every other group.
    let n = crate::weld::PAR_MIN_WELD_VERTS as u32 + 1024;
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    for i in 0..n {
        positions.extend([(i / 2) as f32, 0., 0.]);
        let side = if i % 4 == 3 { 0.5 } else { 0.25 };
        uvs.extend([side, 0.]);
    }
    let run = || {
        let mut m = mesh_of(&positions, &[[0, 1, 2]]);
        m.add_uvs(&uvs).unwrap();
        crate::weld::weld_roots_keeping_seams(&m, 0.01, m.uvs()).unwrap()
    };
    crate::test_util::assert_parallel_identical("seamed weld", run);
}
