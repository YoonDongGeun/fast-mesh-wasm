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
    AppendSource, AutoRepairOptions, BooleanOp, ComponentOptions, CutOptions, DegenerateOptions,
    FillOptions, OffsetAlgorithm, OffsetOptions, ShellOptions, SkipReason, append_mesh,
    append_mesh_with_uvs, apply_boolean, auto_repair, boolean_plan, boundary_loops, bridge_loops,
    cut_by_plane, fill_holes, inflate, offset, orient, remove_degenerate, remove_small_components,
    shell, weld,
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

#[test]
fn remove_small_components_counts_uv_islands_as_one_surface() {
    let mut m = two_charts(None);
    let triangles = m.triangle_count();
    let options = ComponentOptions {
        min_triangles: Some(triangles),
        ..Default::default()
    };
    let (r, _) = edit(&mut m, |tx| remove_small_components(tx, &options));
    assert_eq!(r.removed_components, 0, "islands of one surface");
    assert_eq!(m.triangle_count(), triangles);
}

// ---- auto_repair

/// A flat `8 x 4` sheet of two charts split along `x = 4` with a UV seam, the
/// right chart's UVs offset. `hole`, a cell of the left chart, is left out.
fn two_charts(hole: Option<(u32, u32)>) -> Mesh {
    let (mut positions, mut uvs, mut triangles) = (Vec::new(), Vec::new(), Vec::new());
    for (chart, from) in [(0u32, 0u32), (1, 4)] {
        let base = (positions.len() / 3) as u32;
        let at = |x: u32, y: u32| base + y * 5 + x;
        for y in 0..5u32 {
            for x in 0..5u32 {
                positions.extend([(from + x) as f32, y as f32, 0.0]);
                uvs.extend([chart as f32 + x as f32 / 8.0, y as f32 / 8.0]);
            }
        }
        for y in 0..4 {
            for x in 0..4 {
                if chart == 0 && hole == Some((x, y)) {
                    continue;
                }
                triangles.push([at(x, y), at(x + 1, y), at(x + 1, y + 1)]);
                triangles.push([at(x, y), at(x + 1, y + 1), at(x, y + 1)]);
            }
        }
    }
    let mut m = mesh_of(&positions, &triangles);
    m.add_uvs(&uvs).unwrap();
    m
}

#[test]
fn auto_repair_keeps_seams_and_islands_and_fills_only_real_holes() {
    let mut m = two_charts(Some((1, 1)));
    let before = copy(&m);
    let triangles = m.triangle_count();
    let n = m.vertex_slots();
    let options = AutoRepairOptions {
        fair: false,
        ..AutoRepairOptions::default()
    };
    let (r, _) = edit(&mut m, |tx| auto_repair(tx, &options));
    assert_eq!(r.weld.merged_vertices, 0, "the seam stays");
    assert_eq!(r.components.removed_components, 0, "both islands stay");
    assert_eq!(r.fill.filled.len(), 1, "only the hole");
    let seams = r
        .fill
        .skipped
        .iter()
        .filter(|(_, why)| *why == SkipReason::UvSeam);
    assert_eq!(seams.count(), 2, "each chart's rim holds the seam");
    assert_eq!(
        m.triangle_count(),
        triangles + r.fill.filled[0].triangles_added
    );
    assert_kept(&m, &before);
    assert_eq!(m.uvs().unwrap().len(), 2 * m.vertex_slots());
    assert!(m.vertex_slots() >= n);
    // The hole's patch stays in its chart's UV range: never a stale default.
    for v in n as u32..m.vertex_slots() as u32 {
        if m.is_vertex_alive(v) {
            let want = m.position(v).unwrap().map(|c| c / 8.0);
            let got = uv_of(&m, v);
            assert!((want[0] - got[0]).abs() < 1e-5 && (want[1] - got[1]).abs() < 1e-5);
        }
    }
}

#[test]
fn auto_repair_undoes_with_uvs_in_one_patch() {
    let mut m = two_charts(Some((1, 1)));
    let uvs = m.uvs().unwrap().to_vec();
    let mut tx = m.begin_edit();
    auto_repair(&mut tx, &AutoRepairOptions::default()).unwrap();
    let patch = tx.commit().unwrap();
    patch.revert(&mut m).unwrap();
    assert_eq!(&m.uvs().unwrap()[..uvs.len()], &uvs[..]);
    assert!(m.uvs().is_some());
}

#[test]
fn auto_repair_still_welds_equal_uvs_and_ignores_seams_without_uvs() {
    let mut m = seam_quads(false);
    let (r, _) = edit(&mut m, |tx| auto_repair(tx, &AutoRepairOptions::default()));
    assert_eq!(r.weld.merged_vertices, 2);
    let seam = seam_quads(true);
    let positions: Vec<f32> = seam.positions().iter().flatten().copied().collect();
    let triangles: Vec<[u32; 3]> = (0..4u32).filter_map(|t| seam.triangle(t)).collect();
    let mut plain = mesh_of(&positions, &triangles);
    let (r, _) = edit(&mut plain, |tx| {
        auto_repair(tx, &AutoRepairOptions::default())
    });
    assert_eq!(r.weld.merged_vertices, 2);
}

#[test]
fn fill_skips_seam_loops_unless_they_are_listed() {
    let mut m = two_charts(None);
    let options = FillOptions {
        max_edges: Some(64),
        fair: false,
        ..FillOptions::default()
    };
    let (r, _) = edit(&mut m, |tx| fill_holes(tx, &options));
    assert!(r.filled.is_empty());
    assert_eq!(r.skipped.len(), 2);
    assert!(r.skipped.iter().all(|(_, why)| *why == SkipReason::UvSeam));

    let mut listed = two_charts(None);
    let id = boundary_loops(&listed).unwrap()[0].id;
    let options = FillOptions {
        loops: Some(vec![id]),
        allow_unclean: true,
        fair: false,
        ..FillOptions::default()
    };
    let (r, _) = edit(&mut listed, |tx| fill_holes(tx, &options));
    assert!(!r.skipped.iter().any(|(_, why)| *why == SkipReason::UvSeam));
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
fn split_stretched_interpolates_and_keeps_old_uvs() {
    use crate::stretch::{StretchOptions, capture_rest_sizes, split_stretched};
    let mut m = with_planar(grid(6));
    let before = m.uvs().unwrap().to_vec();
    capture_rest_sizes(&mut m).unwrap();
    let ids: Vec<VertexId> = (0..m.vertex_slots() as VertexId).collect();
    let doubled: Vec<[f32; 3]> = m.positions().iter().map(|p| p.map(|x| x * 2.0)).collect();
    edit(&mut m, |tx| Ok(tx.set_positions(&ids, &doubled)?));
    let options = StretchOptions {
        region: Region {
            center: [0.0; 3],
            radius: 1e6,
        },
        ..StretchOptions::default()
    };
    let (r, _) = edit(&mut m, |tx| split_stretched(tx, &options));
    assert!(r.splits > 0);
    let uvs = m.uvs().unwrap();
    assert_eq!(uvs[..before.len()], before[..]);
    // The flat doubled grid: new positions lie on the doubled mapping.
    for v in ids.len()..m.vertex_slots() {
        let p = m.positions()[v];
        let uv = uv_of(&m, v as VertexId);
        let want = planar([p[0] / 2.0, p[1] / 2.0, 0.0]);
        assert!((uv[0] - want[0]).abs() < 1e-4 && (uv[1] - want[1]).abs() < 1e-4);
    }
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
        let options = InflateOptions {
            brush: brush(),
            ..Default::default()
        };
        inflate(tx, &options).map(|_| ())
    });
    check(&|tx| {
        let options = SmoothOptions {
            brush: brush(),
            cycles: 2,
            method: SmoothMethod::default(),
            ..Default::default()
        };
        crate::deform::smooth(tx, &options).map(|_| ())
    });
    check(&|tx| {
        let options = GrabOptions {
            brush: brush(),
            direction: [0., 0., 1.],
            ..Default::default()
        };
        crate::deform::grab(tx, &options).map(|_| ())
    });
    check(&|tx| {
        let options = GrabOptions {
            brush: brush(),
            direction: [0., 0., 1.],
            method: crate::sculpt::GrabMethod::Geodesic,
            ..Default::default()
        };
        crate::sculpt::grab(tx, &options).map(|_| ())
    });
    check(&|tx| {
        let options = GrabOptions {
            brush: brush(),
            direction: [0., 0., 1.],
            method: crate::sculpt::GrabMethod::Kelvinlet,
            ..Default::default()
        };
        crate::sculpt::grab(tx, &options).map(|_| ())
    });
    check(&|tx| {
        crate::sculpt::flatten(
            tx,
            &crate::sculpt::FlattenOptions {
                brush: brush(),
                ..Default::default()
            },
        )
        .map(|_| ())
    });
    check(&|tx| {
        crate::sculpt::crease(
            tx,
            &crate::sculpt::CreaseOptions {
                brush: brush(),
                ..Default::default()
            },
        )
        .map(|_| ())
    });
    check(&|tx| {
        crate::sculpt::clay(
            tx,
            &crate::sculpt::ClayOptions {
                brush: brush(),
                profile: crate::sculpt::ClayProfile::Strips,
                tangent: Some([1., 0., 0.]),
                ..Default::default()
            },
        )
        .map(|_| ())
    });
    check(&|tx| {
        crate::sculpt::twist(
            tx,
            &crate::sculpt::TwistOptions {
                brush: brush(),
                ..Default::default()
            },
        )
        .map(|_| ())
    });
    check(&|tx| {
        offset(
            tx,
            &OffsetOptions {
                distance: 0.1,
                ..Default::default()
            },
        )
        .map(|_| ())
    });
}

// ---- what moving vertices does to a seam

/// Two quads with a UV seam along `x = 1`, its corners at slots 1, 2 and 4, 5.
/// `bend` lifts the far sides, so the two sides of the seam have different
/// normals.
fn seam_with_bend(bend: f32) -> Mesh {
    let mut m = mesh_of(
        &[
            0., 0., bend, 1., 0., 0., 1., 1., 0., 0., 1., bend, //
            1., 0., 0., 1., 1., 0., 2., 0., bend, 2., 1., bend,
        ],
        &[[0, 1, 2], [0, 2, 3], [4, 6, 7], [4, 7, 5]],
    );
    m.add_uvs(&[
        0., 0., 0.4, 0., 0.4, 1., 0., 1., 0.6, 0., 0.6, 1., 1., 0., 1., 1.,
    ])
    .unwrap();
    m
}

/// The distance between the two sides of `seam_with_bend`'s seam.
fn seam_gap(m: &Mesh) -> f32 {
    let (a, b) = (m.position(1).unwrap(), m.position(4).unwrap());
    (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f32::max)
}

#[test]
fn offset_opens_a_bent_seam_and_keeps_a_flat_one() {
    let options = OffsetOptions {
        distance: 0.1,
        ..Default::default()
    };
    let mut bent = seam_with_bend(1.);
    edit(&mut bent, |tx| offset(tx, &options));
    assert!(seam_gap(&bent) > 0.05, "{}", seam_gap(&bent));
    let mut flat = seam_with_bend(0.);
    edit(&mut flat, |tx| offset(tx, &options));
    assert_eq!(seam_gap(&flat), 0.);
}

#[test]
fn brushes_open_a_bent_seam_unless_the_move_is_the_callers() {
    let brush = Brush {
        center: [1., 0.5, 0.],
        radius: 3.,
        falloff: Falloff::Linear,
        strength: 0.5,
    };
    let mut inflated = seam_with_bend(1.);
    edit(&mut inflated, |tx| {
        inflate(
            tx,
            &InflateOptions {
                brush,
                ..Default::default()
            },
        )
    });
    assert!(seam_gap(&inflated) > 0.01, "{}", seam_gap(&inflated));
    for method in [SmoothMethod::Taubin, SmoothMethod::Laplacian] {
        let options = SmoothOptions {
            brush,
            cycles: 1,
            method,
            ..Default::default()
        };
        let mut smoothed = seam_with_bend(1.);
        // Lopsided, so the two sides have different neighbours.
        let mut tx = smoothed.begin_edit();
        tx.set_positions(&[0], &[[-1., 0., 2.]]).unwrap();
        crate::deform::smooth(&mut tx, &options).unwrap();
        tx.commit().unwrap();
        assert!(
            seam_gap(&smoothed) > 0.01,
            "{method:?}: {}",
            seam_gap(&smoothed)
        );
    }
    let mut grabbed = seam_with_bend(1.);
    edit(&mut grabbed, |tx| {
        let options = GrabOptions {
            brush,
            direction: [0., 0., 1.],
            ..Default::default()
        };
        crate::deform::grab(tx, &options)
    });
    assert_eq!(seam_gap(&grabbed), 0.);
    for method in [
        crate::sculpt::GrabMethod::Geodesic,
        crate::sculpt::GrabMethod::Kelvinlet,
    ] {
        let mut m = seam_with_bend(1.);
        edit(&mut m, |tx| {
            crate::sculpt::grab(
                tx,
                &GrabOptions {
                    brush,
                    direction: [0., 0., 1.],
                    method,
                    ..Default::default()
                },
            )
        });
        assert_eq!(seam_gap(&m), 0., "{method:?}");
    }
    let mut creased = seam_with_bend(1.);
    edit(&mut creased, |tx| {
        crate::sculpt::crease(
            tx,
            &crate::sculpt::CreaseOptions {
                brush,
                pinch: 0.5,
                ..Default::default()
            },
        )
    });
    assert!(seam_gap(&creased) > 0.01, "{}", seam_gap(&creased));
}

#[test]
fn shell_walls_a_seam_and_opens_its_inner_surface() {
    let options = ShellOptions { thickness: 0.1 };
    // Each island has 4 border edges, with 2 wall triangles on each, and 2
    // triangles copied. One welded sheet would have 6 border edges.
    let mut flat = seam_with_bend(0.);
    edit(&mut flat, |tx| shell(tx, &options));
    assert_eq!(flat.triangle_count(), 4 + 4 + 2 * 8);
    // Two of those walls stand along the seam itself, one for each island.
    let on_seam = (0..flat.triangle_slots() as u32)
        .filter_map(|t| flat.triangle(t))
        .filter(|t| t.iter().all(|&v| flat.position(v).unwrap()[0] == 1.))
        .count();
    assert_eq!(on_seam, 4);
    // Bent, the copies of the two sides move apart. Slots 8 + k copy slots k.
    let mut bent = seam_with_bend(1.);
    edit(&mut bent, |tx| shell(tx, &options));
    let (a, b) = (bent.position(9).unwrap(), bent.position(12).unwrap());
    assert!((a[0] - b[0]).abs() > 0.05, "{a:?} {b:?}");
}

/// A box with every face its own UV island: three vertices at each corner.
fn box_of_faces() -> Mesh {
    let whole = box_mesh([0., 0., 0.], [1., 1., 1.]);
    let (mut positions, mut uvs, mut triangles) = (Vec::new(), Vec::new(), Vec::new());
    // Each face is a quad, wound as [a, b, c], [a, c, d].
    for (face, pair) in whole.triangles().as_chunks::<2>().0.iter().enumerate() {
        let base = (positions.len() / 3) as u32;
        let [a, b, c] = pair[0];
        for (k, v) in [a, b, c, pair[1][2]].into_iter().enumerate() {
            positions.extend(whole.position(v).unwrap());
            uvs.extend([face as f32 / 6., k as f32 / 4.]);
        }
        triangles.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]]);
    }
    let mut m = mesh_of(&positions, &triangles);
    m.add_uvs(&uvs).unwrap();
    m
}

/// How many positions the mesh has, exactly.
fn distinct_positions(m: &Mesh) -> usize {
    let mut p: Vec<[u32; 3]> = alive(m)
        .iter()
        .map(|&v| m.position(v).unwrap().map(f32::to_bits))
        .collect();
    p.sort_unstable();
    p.dedup();
    p.len()
}

#[test]
fn subdivide_keeps_a_seam_shut_along_it_but_opens_where_the_borders_part() {
    let options = SubdivideOptions {
        method: SubdivideMethod::Loop,
        levels: 1,
    };
    let gap = |m: &Mesh, a: VertexId, b: VertexId| {
        let (p, q) = (m.position(a).unwrap(), m.position(b).unwrap());
        (0..3).map(|k| (p[k] - q[k]).abs()).fold(0.0, f32::max)
    };
    // Slot `5 y + 4` of the left chart and `25 + 5 y` of the right are one point.
    let mut charts = two_charts(None);
    edit(&mut charts, |tx| crate::subdivide::subdivide(tx, &options));
    // Along the seam both sides follow the same boundary rule. Where it ends
    // on the sheet's border, each side relaxes along its own: 3/4 v + 1/8 of
    // neighbours that differ by 2 in x.
    assert_eq!(gap(&charts, 14, 35), 0.);
    assert_eq!(gap(&charts, 4, 25), 0.25);
    // A corner shared by three islands: each copy follows its own island's
    // border. The whole box has 18 edges (with face diagonals), so 26
    // positions after one level if nothing opens.
    let mut faces = box_of_faces();
    assert_eq!(distinct_positions(&faces), 8);
    edit(&mut faces, |tx| crate::subdivide::subdivide(tx, &options));
    assert!(
        distinct_positions(&faces) > 26,
        "{}",
        distinct_positions(&faces)
    );
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

#[test]
fn volumetric_offset_drops_uvs_merges_seams_and_reverts_cleanly() {
    let live_snapshot = |m: &Mesh| {
        let verts: Vec<(VertexId, [f32; 3])> = alive(m)
            .into_iter()
            .map(|v| (v, m.position(v).unwrap()))
            .collect();
        let tris: Vec<[VertexId; 3]> = (0..m.triangle_slots() as u32)
            .filter_map(|t| m.triangle(t))
            .collect();
        (verts, tris)
    };

    for alg in [
        OffsetAlgorithm::FhEdt {
            voxel_size: Some(0.08),
        },
        OffsetAlgorithm::BvhSdf {
            voxel_size: Some(0.08),
        },
    ] {
        for distance in [-0.15f32, 0.15f32] {
            let mut m = box_of_faces();
            let orig_snap = live_snapshot(&m);
            let orig_uvs = m.uvs().unwrap().to_vec();
            assert!(!boundary_loops(&m).unwrap().is_empty(), "seams start open");

            let opts = OffsetOptions::builder()
                .distance(distance)
                .algorithm(alg)
                .build();
            let mut tx = m.begin_edit();
            let r = offset(&mut tx, &opts).unwrap();
            assert!(r.uvs_dropped);
            assert!(tx.mesh().uvs().is_none());
            let patch = tx.commit().unwrap();

            assert!(m.uvs().is_none());
            assert!(
                boundary_loops(&m).unwrap().is_empty(),
                "alg={alg:?} d={distance} must weld seams into a closed solid"
            );
            let vol = crate::measure(&m).unwrap().volume.unwrap();
            if distance < 0.0 {
                assert!(vol < 0.8 && vol > 0.15, "shrunk volume {vol}");
            } else {
                assert!(vol > 1.3, "expanded volume {vol}");
            }

            let rebuilt_snap = live_snapshot(&m);
            patch.revert(&mut m).unwrap();
            assert_eq!(live_snapshot(&m), orig_snap);
            assert_eq!(&m.uvs().unwrap()[..orig_uvs.len()], &orig_uvs[..]);

            patch.apply(&mut m).unwrap();
            assert!(m.uvs().is_none());
            assert_eq!(live_snapshot(&m), rebuilt_snap);
        }
    }
}

#[test]
fn make_manifold_duplicates_and_preserves_uvs_with_undo() {
    let mut m = mesh_of(
        &[
            0.0, 0.0, 0.0, // pinch vertex
            -1.0, 1.0, 0.0, // 1
            1.0, 1.0, 0.0, // 2
            -1.0, -1.0, 0.0, // 3
            1.0, -1.0, 0.0, // 4
        ],
        &[[0, 2, 1], [0, 3, 4]],
    );
    let orig_uvs = vec![0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    m.add_uvs(&orig_uvs).unwrap();
    let mut tx = m.begin_edit();
    let r = crate::make_manifold(&mut tx, &crate::ManifoldOptions::default()).unwrap();
    assert_eq!(r.split_pinch_vertices, 1);
    assert_eq!(r.vertices_added, 1);
    let patch = tx.commit().unwrap();
    assert_eq!(m.vertex_count(), 6);
    assert_eq!(&m.uvs().unwrap()[10..12], &[0.5, 0.5]);

    patch.revert(&mut m).unwrap();
    assert_eq!(m.vertex_count(), 5);
    assert_eq!(&m.uvs().unwrap()[..orig_uvs.len()], &orig_uvs[..]);
}

#[test]
fn cleanup_slivers_preserves_interpolated_uvs_and_undoes() {
    let mut m = with_planar(mesh_of(
        &[
            0.0, 0.0, 0.0, // 0
            2.0, 0.0, 0.0, // 1
            0.995, 1.0, 0.0, // 2
            1.005, 1.0, 0.0, // 3
            0.0, 2.0, 0.0, // 4
            2.0, 2.0, 0.0, // 5
        ],
        &[
            [0, 1, 2],
            [1, 3, 2], // needle: short edge (2, 3) length 0.01
            [0, 2, 4],
            [2, 5, 4],
            [2, 3, 5], // needle: short edge (2, 3) length 0.01
            [1, 5, 3],
        ],
    ));
    let orig_uvs = m.uvs().unwrap().to_vec();
    let orig_verts = m.vertex_count();
    let mut tx = m.begin_edit();
    let r = crate::cleanup_slivers(&mut tx, &crate::SliverOptions::default()).unwrap();
    assert_eq!(r.collapsed_needles, 1);
    let patch = tx.commit().unwrap();
    for v in alive(&m) {
        let p = m.position(v).unwrap();
        let got = uv_of(&m, v);
        let want = planar(p);
        assert!(
            (got[0] - want[0]).abs() < 1e-4 && (got[1] - want[1]).abs() < 1e-4,
            "uv drifted at v={v}: got={got:?} want={want:?}"
        );
    }

    patch.revert(&mut m).unwrap();
    assert_eq!(m.vertex_count(), orig_verts);
    assert_eq!(&m.uvs().unwrap()[..orig_uvs.len()], &orig_uvs[..]);
}

#[test]
fn resolve_self_intersections_uv_contract_and_undo() {
    // 1) DeleteAndFill preserves UVs & undoes.
    let mut poked = uv_sphere(12, 8, 1.0);
    // Invert a pole region into the sphere so opposite triangles intersect.
    edit(&mut poked, |tx| {
        let mut moved = Vec::new();
        for v in 0..tx.mesh().vertex_slots() as u32 {
            if let Some(p) = tx.mesh().position(v)
                && p[2] > 0.65
            {
                moved.push((v, [p[0], p[1], -p[2] * 1.4]));
            }
        }
        let ids: Vec<u32> = moved.iter().map(|(v, _)| *v).collect();
        let pos: Vec<[f32; 3]> = moved.iter().map(|(_, p)| *p).collect();
        tx.set_positions(&ids, &pos)?;
        Ok(())
    });
    let mut poked = with_planar(poked);
    let orig_uvs = poked.uvs().unwrap().to_vec();
    let mut tx = poked.begin_edit();
    let r = crate::resolve_self_intersections(
        &mut tx,
        None,
        &crate::SelfIntersectOptions::builder()
            .mode(crate::SelfIntersectionMode::DeleteAndFill)
            .build(),
    )
    .unwrap();
    assert!(!r.uvs_dropped);
    assert!(r.initial_pairs > 0);
    assert!(tx.mesh().uvs().is_some());
    let patch = tx.commit().unwrap();
    patch.revert(&mut poked).unwrap();
    assert_eq!(&poked.uvs().unwrap()[..orig_uvs.len()], &orig_uvs[..]);

    // 2) CutAndStitch keeps/interpolates planar UVs on split vertices.
    let mut crossed = with_planar(mesh_of(
        &[
            // Tri A in z = 0
            -1.0, -1.0, 0.0, // 0
            1.0, -1.0, 0.0, // 1
            0.0, 1.0, 0.0, // 2
            // Tri B crossing through z = 0 along y = 0, x in [-0.6, 0.6]
            -0.6, 0.0, -0.5, // 3
            0.6, 0.0, -0.5, // 4
            0.0, 0.0, 0.5, // 5
        ],
        &[[0, 1, 2], [3, 4, 5]],
    ));
    let orig_crossed_uvs = crossed.uvs().unwrap().to_vec();
    let mut tx = crossed.begin_edit();
    let r = crate::resolve_self_intersections(
        &mut tx,
        None,
        &crate::SelfIntersectOptions::builder()
            .mode(crate::SelfIntersectionMode::CutAndStitch)
            .build(),
    )
    .unwrap();
    assert!(!r.uvs_dropped);
    assert!(r.initial_pairs > 0);
    let patch = tx.commit().unwrap();
    assert!(crossed.uvs().is_some());
    patch.revert(&mut crossed).unwrap();
    assert_eq!(
        &crossed.uvs().unwrap()[..orig_crossed_uvs.len()],
        &orig_crossed_uvs[..]
    );

    // 3) VoxelHeal drops UVs explicitly and restores them on undo.
    let mut b = box_of_faces();
    let orig_box_uvs = b.uvs().unwrap().to_vec();
    let mut tx = b.begin_edit();
    let r = crate::resolve_self_intersections(
        &mut tx,
        None,
        &crate::SelfIntersectOptions::builder()
            .mode(crate::SelfIntersectionMode::VoxelHeal)
            .voxel_size(Some(0.12))
            .build(),
    )
    .unwrap();
    assert!(r.uvs_dropped);
    assert!(tx.mesh().uvs().is_none());
    let patch = tx.commit().unwrap();
    assert!(b.uvs().is_none());
    patch.revert(&mut b).unwrap();
    assert_eq!(&b.uvs().unwrap()[..orig_box_uvs.len()], &orig_box_uvs[..]);
}

#[test]
fn auto_repair_presets_uv_contract() {
    // Scan3D & CadImport preserve UVs on clean/seamed meshes and undo cleanly.
    for preset in [
        crate::AutoRepairPreset::Scan3D,
        crate::AutoRepairPreset::CadImport,
    ] {
        let mut m = two_charts(Some((1, 1)));
        let orig_uvs = m.uvs().unwrap().to_vec();
        let mut tx = m.begin_edit();
        let r = auto_repair(
            &mut tx,
            &AutoRepairOptions::builder()
                .preset(preset)
                .fair(false)
                .build(),
        )
        .unwrap();
        assert!(
            !r.self_intersections
                .as_ref()
                .is_some_and(|si| si.uvs_dropped)
        );
        let patch = tx.commit().unwrap();
        assert!(m.uvs().is_some());
        patch.revert(&mut m).unwrap();
        assert_eq!(&m.uvs().unwrap()[..orig_uvs.len()], &orig_uvs[..]);
    }

    // Print3D explicitly records uvs_dropped when VoxelHeal runs (directly or on fallback).
    let mut b = box_of_faces();
    let orig_uvs = b.uvs().unwrap().to_vec();
    let mut tx = b.begin_edit();
    let r = auto_repair(
        &mut tx,
        &AutoRepairOptions::builder()
            .preset(crate::AutoRepairPreset::Print3D)
            .self_intersection_mode(Some(crate::SelfIntersectionMode::VoxelHeal))
            .voxel_size(Some(0.12))
            .build(),
    )
    .unwrap();
    assert!(
        r.self_intersections
            .as_ref()
            .is_some_and(|si| si.uvs_dropped)
    );
    let patch = tx.commit().unwrap();
    assert!(b.uvs().is_none());
    patch.revert(&mut b).unwrap();
    assert_eq!(&b.uvs().unwrap()[..orig_uvs.len()], &orig_uvs[..]);
}
