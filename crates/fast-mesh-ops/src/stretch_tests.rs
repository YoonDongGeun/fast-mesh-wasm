//! Tests of [`super::split_stretched`]. Each stroke is a real grab brush over a
//! mesh with both fine and coarse areas, and the expected facts come from
//! comparing the buffers from before the stroke.

use fast_mesh_store::{History, Mesh, Transaction, VertexId};

use super::*;
use crate::deform::{Brush, Falloff, GrabOptions, grab};
use crate::remesh::{RemeshOptions, remesh};
use crate::test_util::{box_mesh, edit, fin, grid, mesh_of, parts, sweep, uv_sphere};
use crate::topology::EdgeMap;

/// A grid with cells from nearly zero at one corner to `3` at the other: both
/// fine and coarse areas, and a mean edge length that fits neither.
fn graded(n: u32) -> Mesh {
    let g = grid(n);
    let warp = |v: f32| n as f32 * (v / n as f32).powi(3);
    let positions: Vec<f32> = g
        .positions()
        .iter()
        .flat_map(|p| [warp(p[0]), warp(p[1]), 0.0])
        .collect();
    mesh_of(&positions, &parts(&g).1)
}

fn all() -> Region {
    Region {
        center: [0.0; 3],
        radius: 1e6,
    }
}

fn pass(mesh: &mut Mesh, region: Region) -> StretchReport {
    edit(mesh, |tx| {
        split_stretched(
            tx,
            &StretchOptions {
                region,
                ..StretchOptions::default()
            },
        )
    })
    .0
}

fn dab(mesh: &mut Mesh, center: [f32; 3], radius: f32, strength: f32) {
    edit(mesh, |tx| {
        grab(
            tx,
            &GrabOptions {
                brush: Brush {
                    center,
                    radius,
                    falloff: Falloff::Smooth,
                    strength,
                },
                direction: [0.0, 0.0, 1.0],
            },
        )
    });
}

/// Every edge against its limit: the rest sizes of its endpoints as they are
/// in the channel, which is what a later pass would read.
fn longest_over_limit(mesh: &Mesh) -> f64 {
    let sizes = mesh.channel(REST_SIZE_CHANNEL).unwrap();
    let edges = EdgeMap::build(mesh).unwrap();
    edges
        .edges()
        .map(|((a, b), _)| {
            let len = length(&mesh.positions()[a as usize], &mesh.positions()[b as usize]);
            let size = sizes[a as usize].min(sizes[b as usize]);
            len / (f64::from(size) * f64::from(DEFAULT_STRETCH_LIMIT))
        })
        .fold(0.0, f64::max)
}

/// The vertices a stroke moved, as a mask over the slots of `before`.
fn stroke_on_graded(
    mesh: &mut Mesh,
    dabs: u32,
    mut after_dab: impl FnMut(&mut Mesh, [f32; 3], u32),
) -> Vec<bool> {
    let slots = mesh.vertex_slots();
    let mut moved = vec![false; slots];
    for k in 0..dabs {
        let center = [30.0 + 0.4 * k as f32, 30.0, 0.0];
        let before = mesh.positions().to_vec();
        dab(mesh, center, 6.0, 1.0);
        for v in 0..slots {
            if mesh.positions()[v] != before[v] {
                moved[v] = true;
            }
        }
        after_dab(mesh, center, k);
    }
    moved
}

#[test]
fn a_stroke_leaves_unstretched_triangles_and_vertices_identical() {
    let mut mesh = graded(60);
    capture_rest_sizes(&mut mesh).unwrap();
    let before = mesh.positions().to_vec();
    let tris_before: Vec<[u32; 3]> = (0..mesh.triangle_slots() as u32)
        .map(|t| mesh.triangle(t).unwrap())
        .collect();
    let mut splits = 0;
    let moved = stroke_on_graded(&mut mesh, 16, |m, center, k| {
        if k % 4 == 3 {
            splits += pass(
                m,
                Region {
                    center,
                    radius: 18.0,
                },
            )
            .splits;
        }
    });
    assert!(splits > 0, "the stroke stretched edges");
    let mut unmoved_tris = 0;
    for (t, tri) in tris_before.iter().enumerate() {
        if tri.iter().all(|&v| !moved[v as usize]) {
            unmoved_tris += 1;
            assert_eq!(
                mesh.triangle(t as u32),
                Some(*tri),
                "triangle {t} rewritten"
            );
        }
    }
    assert!(unmoved_tris > 7000);
    for v in 0..before.len() as VertexId {
        assert!(
            mesh.is_vertex_alive(v) && mesh.corner_count(v) > 0,
            "{v} died"
        );
        if !moved[v as usize] {
            assert_eq!(
                mesh.positions()[v as usize].map(f32::to_bits),
                before[v as usize].map(f32::to_bits),
                "vertex {v} moved"
            );
        }
    }
}

/// The damage this module exists to avoid, on the same stroke: a mesh-wide
/// target rewrites triangles the stroke never touched and deletes vertices,
/// whose ids anything holding them (a mask) then loses.
#[test]
fn the_mean_target_remesh_does_damage_the_same_stroke() {
    let mut mesh = graded(60);
    let tris_before: Vec<[u32; 3]> = (0..mesh.triangle_slots() as u32)
        .map(|t| mesh.triangle(t).unwrap())
        .collect();
    let slots = mesh.vertex_slots();
    let moved = stroke_on_graded(&mut mesh, 16, |_, _, _| {});
    let options = RemeshOptions::builder()
        .center([33.0, 30.0, 0.0])
        .radius(15.0)
        .target_length(1.2)
        .max_rounds(4)
        .build();
    let (report, _) = edit(&mut mesh, |tx| remesh(tx, &options));
    assert!(report.collapses > 0);
    let rewritten = tris_before
        .iter()
        .enumerate()
        .filter(|(t, tri)| {
            tri.iter().all(|&v| !moved[v as usize]) && mesh.triangle(*t as u32) != Some(**tri)
        })
        .count();
    let deleted = (0..slots as u32)
        .filter(|&v| !moved[v as usize] && !mesh.is_vertex_alive(v))
        .count();
    assert!(rewritten > 0 && deleted > 0, "{rewritten} {deleted}");
}

#[test]
fn an_unstretched_mesh_never_splits_however_uneven() {
    let mut mesh = graded(30);
    capture_rest_sizes(&mut mesh).unwrap();
    let before = parts(&mesh);
    let (report, empty) = edit(&mut mesh, |tx| {
        split_stretched(
            tx,
            &StretchOptions {
                region: all(),
                ..StretchOptions::default()
            },
        )
    });
    assert_eq!(report, StretchReport::default());
    assert!(empty);
    assert_eq!(parts(&mesh), before);
}

#[test]
fn edges_stay_under_the_limit_through_a_long_stroke() {
    let mut mesh = graded(60);
    capture_rest_sizes(&mut mesh).unwrap();
    let start = mesh.triangle_count();
    stroke_on_graded(&mut mesh, 60, |m, center, _| {
        pass(
            m,
            Region {
                center,
                radius: 18.0,
            },
        );
        assert!(longest_over_limit(m) <= 1.0 + 1e-6);
    });
    assert!(mesh.triangle_count() > start);
}

#[test]
fn repeated_strokes_do_not_ratchet_the_size_up() {
    let mut mesh = grid(20);
    capture_rest_sizes(&mut mesh).unwrap();
    for _ in 0..12 {
        dab(&mut mesh, [10.0, 10.0, 0.0], 5.0, 0.3);
        pass(&mut mesh, all());
    }
    assert!(longest_over_limit(&mesh) <= 1.0 + 1e-6);
}

fn unit_normal(mesh: &Mesh, t: u32) -> [f64; 3] {
    let [a, b, c] = mesh
        .triangle(t)
        .unwrap()
        .map(|v| f64v(&mesh.positions()[v as usize]));
    unit(cross3(sub3(b, a), sub3(c, a))).unwrap()
}

/// Root mean square and largest angle between the faces on an edge touching a
/// new vertex. Not the plain mean: midpoints leave most new edges inside a flat
/// face at zero and keep the whole crease on the old ones, which is what a
/// facet looks like, and a plain mean rewards it.
fn dihedrals_at_new_vertices(mesh: &Mesh, base_slots: usize) -> (f64, f64) {
    let edges = EdgeMap::build(mesh).unwrap();
    let (mut sum, mut max, mut n) = (0.0_f64, 0.0_f64, 0);
    for ((a, b), owners) in edges.edges() {
        if (a as usize) < base_slots && (b as usize) < base_slots {
            continue;
        }
        if let [o0, o1] = owners {
            let angle = dot3(unit_normal(mesh, *o0), unit_normal(mesh, *o1))
                .clamp(-1.0, 1.0)
                .acos();
            sum += angle * angle;
            max = max.max(angle);
            n += 1;
        }
    }
    assert!(n > 0);
    ((sum / f64::from(n)).sqrt(), max)
}

/// Length-weighted root mean square dihedral over every edge, and the largest:
/// what a facet shows, wherever it is.
fn dihedrals_everywhere(mesh: &Mesh) -> (f64, f64) {
    let edges = EdgeMap::build(mesh).unwrap();
    let (mut sum, mut weight, mut max) = (0.0_f64, 0.0_f64, 0.0_f64);
    for ((a, b), owners) in edges.edges() {
        if let [o0, o1] = owners {
            let angle = dot3(unit_normal(mesh, *o0), unit_normal(mesh, *o1))
                .clamp(-1.0, 1.0)
                .acos();
            let len = length(&mesh.positions()[a as usize], &mesh.positions()[b as usize]);
            sum += len * angle * angle;
            weight += len;
            max = max.max(angle);
        }
    }
    ((sum / weight).sqrt(), max)
}

/// Area-weighted root mean square angle between each face and the normal of
/// the sphere around the origin that the mesh samples.
fn shading_error(mesh: &Mesh) -> f64 {
    let (mut sum, mut weight) = (0.0, 0.0);
    for t in 0..mesh.triangle_slots() as u32 {
        let Some(c) = mesh.triangle(t) else { continue };
        let [a, b, d] = c.map(|v| f64v(&mesh.positions()[v as usize]));
        let n = cross3(sub3(b, a), sub3(d, a));
        let centroid: [f64; 3] = std::array::from_fn(|k| (a[k] + b[k] + d[k]) / 3.0);
        let (Some(face), Some(truth)) = (unit(n), unit(centroid)) else {
            continue;
        };
        let angle = dot3(face, truth).clamp(-1.0, 1.0).acos();
        let area = dot3(n, n).sqrt();
        sum += area * angle * angle;
        weight += area;
    }
    (sum / weight).sqrt()
}

/// A coarse sphere stretched by `scale`, so its edges are past the limit,
/// split with or without the curved rule.
fn stretched_sphere(scale: f32, smooth: bool) -> (Mesh, usize) {
    let mut mesh = uv_sphere(8, 12, 1.0);
    capture_rest_sizes(&mut mesh).unwrap();
    let slots = mesh.vertex_slots();
    let ids: Vec<u32> = (0..slots as u32).collect();
    let far: Vec<[f32; 3]> = mesh
        .positions()
        .iter()
        .map(|p| p.map(|x| scale * x))
        .collect();
    edit(&mut mesh, |tx| Ok(tx.set_positions(&ids, &far)?));
    edit(&mut mesh, |tx| {
        split_stretched(
            tx,
            &StretchOptions {
                region: all(),
                smooth,
                ..StretchOptions::default()
            },
        )
    });
    (mesh, slots)
}

#[test]
fn curved_vertices_are_smoother_than_midpoints_where_every_edge_splits() {
    let (flat, slots) = stretched_sphere(3.0, false);
    let (curved, _) = stretched_sphere(3.0, true);
    let (flat_rms, flat_max) = dihedrals_at_new_vertices(&flat, slots);
    let (curved_rms, curved_max) = dihedrals_at_new_vertices(&curved, slots);
    eprintln!(
        "dihedral at new vertices (degrees): midpoint rms {:.2} max {:.2}, curved rms {:.2} max {:.2}",
        flat_rms.to_degrees(),
        flat_max.to_degrees(),
        curved_rms.to_degrees(),
        curved_max.to_degrees()
    );
    assert!(curved_rms < 0.8 * flat_rms, "{curved_rms} vs {flat_rms}");
    assert!(curved_max < 0.5 * flat_max, "{curved_max} vs {flat_max}");
    // And they lie on the sphere the mesh samples: far closer than midpoints.
    let off = |m: &Mesh| {
        (slots..m.vertex_slots())
            .map(|v| (length(&m.positions()[v], &[0.0; 3]) - 3.0).abs())
            .fold(0.0, f64::max)
    };
    assert!(off(&curved) < 0.5 * off(&flat));
}

/// At a small stretch only some edges split. A lifted vertex then puts a
/// small crease on its own edges where a midpoint leaves none, so the angle
/// at new vertices alone would favour midpoints; over the whole surface, and
/// against the true normal, the curved rule still wins.
#[test]
fn curved_vertices_are_smoother_over_the_whole_surface_at_any_stretch() {
    for scale in [1.5, 2.0, 3.0] {
        let (flat, _) = stretched_sphere(scale, false);
        let (curved, _) = stretched_sphere(scale, true);
        let (flat_rms, flat_max) = dihedrals_everywhere(&flat);
        let (curved_rms, curved_max) = dihedrals_everywhere(&curved);
        let (flat_shade, curved_shade) = (shading_error(&flat), shading_error(&curved));
        eprintln!(
            "x{scale} whole surface (degrees): dihedral rms {:.2} -> {:.2}, max {:.2} -> {:.2}, shading {:.2} -> {:.2}",
            flat_rms.to_degrees(),
            curved_rms.to_degrees(),
            flat_max.to_degrees(),
            curved_max.to_degrees(),
            flat_shade.to_degrees(),
            curved_shade.to_degrees()
        );
        assert!(curved_rms < 0.8 * flat_rms, "x{scale}");
        assert!(curved_max < 0.7 * flat_max, "x{scale}");
        assert!(curved_shade < 0.5 * flat_shade, "x{scale}");
    }
}

#[test]
fn a_flat_face_stays_flat_next_to_a_hard_edge() {
    let mut mesh = box_mesh([-1.0; 3], [1.0; 3]);
    capture_rest_sizes(&mut mesh).unwrap();
    let slots = mesh.vertex_slots();
    let ids: Vec<u32> = (0..slots as u32).collect();
    let far: Vec<[f32; 3]> = mesh
        .positions()
        .iter()
        .map(|p| p.map(|x| 3.0 * x))
        .collect();
    edit(&mut mesh, |tx| Ok(tx.set_positions(&ids, &far)?));
    assert!(pass(&mut mesh, all()).splits > 0);
    for v in slots..mesh.vertex_slots() {
        let p = mesh.positions()[v];
        let on_face = p.iter().any(|x| (x.abs() - 3.0).abs() < 1e-5);
        let inside = p.iter().all(|x| x.abs() <= 3.0 + 1e-5);
        assert!(on_face && inside, "{p:?} left the box");
    }
}

#[test]
fn undo_restores_the_buffers_and_redo_the_split() {
    let mut mesh = graded(30);
    capture_rest_sizes(&mut mesh).unwrap();
    let before = parts(&mesh);
    let mut history = History::new(1 << 26);
    dab(&mut mesh, [8.0, 8.0, 0.0], 5.0, 6.0);
    let after_dab = parts(&mesh);
    let mut tx = mesh.begin_edit();
    let report = split_stretched(
        &mut tx,
        &StretchOptions {
            region: all(),
            ..StretchOptions::default()
        },
    )
    .unwrap();
    assert!(report.splits > 0);
    let patch = tx.commit().unwrap();
    history.push(&mut mesh, patch).unwrap();
    let split = parts(&mesh);
    assert_ne!(split, after_dab);
    history.undo(&mut mesh).unwrap().unwrap();
    assert_eq!(parts(&mesh).1, after_dab.1);
    let slots = before.0.len();
    assert_eq!(parts(&mesh).0[..slots], after_dab.0[..slots]);
    history.redo(&mut mesh).unwrap().unwrap();
    assert_eq!(parts(&mesh), split);
    // Split again after the redo: its vertices kept their rest sizes.
    assert_eq!(pass(&mut mesh, all()).splits, 0);
}

/// A sheet cut along `x = 2` into two UV islands whose sides differ in
/// density, so the seam's two copies have different longest edges.
fn seamed_sheet() -> (Mesh, Vec<(u32, u32)>) {
    let mut positions: Vec<f32> = Vec::new();
    let mut uvs: Vec<f32> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut seam: Vec<(u32, u32)> = Vec::new();
    let (ny, cols_left, cols_right) = (8u32, [0.0f32, 1.0, 1.5, 1.75, 2.0], [2.0f32, 5.0]);
    let mut island = |cols: &[f32], u0: f32| {
        let base = (positions.len() / 3) as u32;
        let width = cols.len() as u32;
        for y in 0..=ny {
            for &x in cols {
                positions.extend([x, y as f32, 0.0]);
                uvs.extend([u0 + x / 10.0, y as f32 / 10.0]);
            }
        }
        for y in 0..ny {
            for x in 0..width - 1 {
                let a = base + y * width + x;
                tris.push([a, a + 1, a + width + 1]);
                tris.push([a, a + width + 1, a + width]);
            }
        }
        base
    };
    let left = island(&cols_left, 0.0);
    let right = island(&cols_right, 0.5);
    for y in 0..=ny {
        seam.push((left + y * 5 + 4, right + y * 2));
    }
    let mut mesh = mesh_of(&positions, &tris);
    mesh.add_uvs(&uvs).unwrap();
    (mesh, seam)
}

#[test]
fn a_seam_splits_on_both_sides_and_stays_closed() {
    let (mut mesh, seam) = seamed_sheet();
    capture_rest_sizes(&mut mesh).unwrap();
    let slots = mesh.vertex_slots();
    // The sides see one rest size, though their longest edges differ.
    let sizes = mesh.channel(REST_SIZE_CHANNEL).unwrap();
    for &(a, b) in &seam {
        assert_eq!(sizes[a as usize], sizes[b as usize]);
    }
    let uv_before = mesh.uvs().unwrap().to_vec();
    // Stretch the seam itself along y, moving each side's copies alike.
    let ids: Vec<u32> = seam.iter().flat_map(|&(a, b)| [a, b]).collect();
    let moved: Vec<[f32; 3]> = ids
        .iter()
        .map(|&v| {
            let p = mesh.positions()[v as usize];
            [p[0], p[1] * 8.0, p[1]]
        })
        .collect();
    edit(&mut mesh, |tx| Ok(tx.set_positions(&ids, &moved)?));
    assert!(pass(&mut mesh, all()).splits > 0);

    // Every boundary edge on the seam line has a twin at the same positions.
    let edges = EdgeMap::build(&mesh).unwrap();
    let on_seam = |v: u32| mesh.positions()[v as usize][0] == 2.0;
    let mut sides: Vec<[[u32; 3]; 2]> = edges
        .edges()
        .filter(|((a, b), owners)| owners.len() == 1 && on_seam(*a) && on_seam(*b))
        .map(|((a, b), _)| {
            let (pa, pb) = (mesh.positions()[a as usize], mesh.positions()[b as usize]);
            let (pa, pb) = (pa.map(f32::to_bits), pb.map(f32::to_bits));
            if pa <= pb { [pa, pb] } else { [pb, pa] }
        })
        .collect();
    assert!(sides.len() > 2 * seam.len());
    sides.sort_unstable();
    for pair in sides.chunks(2) {
        assert_eq!(pair.len(), 2);
        assert_eq!(pair[0], pair[1], "the seam opened");
    }

    // Old UVs are where they were and new ones interpolate along their island.
    let uvs = mesh.uvs().unwrap();
    assert_eq!(&uvs[..2 * slots], &uv_before[..]);
    assert_eq!(uvs.len(), 2 * mesh.vertex_slots());
    for v in slots..mesh.vertex_slots() {
        let u = uvs[2 * v];
        assert!((0.0..=1.0).contains(&u) && (u < 0.21 || u >= 0.5), "{u}");
    }
}

#[test]
fn the_cap_leaves_the_rest_for_a_later_pass_and_splits_the_longest_first() {
    let mut mesh = grid(10);
    capture_rest_sizes(&mut mesh).unwrap();
    let slots = mesh.vertex_slots();
    let ids: Vec<u32> = (0..slots as u32).collect();
    let far: Vec<[f32; 3]> = mesh
        .positions()
        .iter()
        .map(|p| [p[0] * 2.0, p[1] * 2.0, 0.0])
        .collect();
    edit(&mut mesh, |tx| Ok(tx.set_positions(&ids, &far)?));
    let capped = StretchOptions {
        region: all(),
        max_splits: 3,
        ..StretchOptions::default()
    };
    let (first, _) = edit(&mut mesh, |tx| split_stretched(tx, &capped));
    assert!(first.remaining > 0);
    // Every boundary edge splits whatever the cap, then three interior ones.
    assert!(first.splits > 3);
    assert_eq!(pass(&mut mesh, all()).remaining, 0);
    assert!(longest_over_limit(&mesh) <= 1.0 + 1e-6);
}

#[test]
fn edges_outside_the_region_wait() {
    let mut mesh = grid(10);
    capture_rest_sizes(&mut mesh).unwrap();
    let slots = mesh.vertex_slots();
    let ids: Vec<u32> = (0..slots as u32).collect();
    let far: Vec<[f32; 3]> = mesh
        .positions()
        .iter()
        .map(|p| p.map(|x| x * 2.0))
        .collect();
    edit(&mut mesh, |tx| Ok(tx.set_positions(&ids, &far)?));
    let before = parts(&mesh);
    let away = Region {
        center: [1000.0; 3],
        radius: 1.0,
    };
    assert_eq!(pass(&mut mesh, away), StretchReport::default());
    assert_eq!(parts(&mesh), before);
}

#[test]
fn every_owner_of_a_non_manifold_edge_splits() {
    let mut mesh = fin();
    capture_rest_sizes(&mut mesh).unwrap();
    let slots = mesh.vertex_slots();
    let ids: Vec<u32> = (0..slots as u32).collect();
    let far: Vec<[f32; 3]> = mesh
        .positions()
        .iter()
        .map(|p| p.map(|x| x * 4.0))
        .collect();
    edit(&mut mesh, |tx| Ok(tx.set_positions(&ids, &far)?));
    assert!(pass(&mut mesh, all()).splits > 0);
    let sizes = EdgeMap::build(&mesh).unwrap();
    for ((a, b), _) in sizes.edges() {
        assert!(mesh.corner_count(a) > 0 && mesh.corner_count(b) > 0);
    }
    assert!(longest_over_limit(&mesh) <= 1.0 + 1e-6);
}

#[test]
fn it_needs_a_capture_and_valid_options() {
    let mut mesh = grid(2);
    let run = |mesh: &mut Mesh, o: StretchOptions| -> Result<StretchReport> {
        let mut tx = mesh.begin_edit();
        let r = split_stretched(&mut tx, &o);
        tx.rollback();
        r
    };
    let ok = StretchOptions {
        region: all(),
        ..StretchOptions::default()
    };
    assert!(matches!(run(&mut mesh, ok), Err(OpsError::InvalidInput(_))));
    capture_rest_sizes(&mut mesh).unwrap();
    assert!(run(&mut mesh, ok).is_ok());
    for limit in [1.0, 0.5, f32::NAN, f32::INFINITY] {
        let bad = StretchOptions { limit, ..ok };
        assert!(matches!(
            run(&mut mesh, bad),
            Err(OpsError::InvalidInput(_))
        ));
    }
    let bad = StretchOptions {
        region: Region {
            center: [f32::NAN; 3],
            radius: 1.0,
        },
        ..ok
    };
    assert!(matches!(
        run(&mut mesh, bad),
        Err(OpsError::InvalidInput(_))
    ));
    let bad = StretchOptions {
        region: Region {
            center: [0.0; 3],
            radius: 0.0,
        },
        ..ok
    };
    assert!(matches!(
        run(&mut mesh, bad),
        Err(OpsError::InvalidInput(_))
    ));
}

#[test]
fn it_lands_whole_or_not_at_all() {
    let mut mesh = grid(6);
    capture_rest_sizes(&mut mesh).unwrap();
    let slots = mesh.vertex_slots();
    let ids: Vec<u32> = (0..slots as u32).collect();
    let far: Vec<[f32; 3]> = mesh
        .positions()
        .iter()
        .map(|p| p.map(|x| x * 3.0))
        .collect();
    edit(&mut mesh, |tx| Ok(tx.set_positions(&ids, &far)?));
    let failures = sweep(mesh, |tx: &mut Transaction| {
        split_stretched(
            tx,
            &StretchOptions {
                region: all(),
                ..StretchOptions::default()
            },
        )
        .map(|_| ())
    });
    assert!(failures > 0);
}

fn mesh_positions_equal(a: &Mesh, b: &Mesh) -> bool {
    parts(a).0 == parts(b).0
}

#[test]
fn a_private_index_gives_the_same_result_as_the_cached_one() {
    let stretched = || {
        let mut mesh = graded(20);
        capture_rest_sizes(&mut mesh).unwrap();
        dab(&mut mesh, [10.0, 10.0, 0.0], 8.0, 5.0);
        mesh
    };
    let mut cached = stretched();
    assert!(cached.incidence().is_some());
    let a = pass(&mut cached, all());
    // A topology edit earlier in the same transaction leaves the index
    // pending, so the pass has to build its own.
    let mut private = stretched();
    let mut tx = private.begin_edit();
    let t = tx.add_triangles(&[[0, 1, 22]]).unwrap()[0];
    tx.remove_triangles(&[t]).unwrap();
    assert!(tx.mesh().incidence().is_none());
    let b = split_stretched(
        &mut tx,
        &StretchOptions {
            region: all(),
            ..StretchOptions::default()
        },
    )
    .unwrap();
    tx.commit().unwrap();
    assert!(a.splits > 0);
    assert_eq!(a, b);
    assert!(mesh_positions_equal(&cached, &private));
    let sorted = |m: &Mesh| {
        let mut tris = parts(m).1;
        tris.sort_unstable();
        tris
    };
    assert!(sorted(&cached) == sorted(&private));
}

#[test]
fn capture_replaces_the_rest_shape_without_an_edit() {
    let mut mesh = grid(4);
    let revision = mesh.revision();
    capture_rest_sizes(&mut mesh).unwrap();
    assert_eq!(mesh.revision(), revision);
    assert_eq!(mesh.channel(REST_SIZE_CHANNEL).unwrap()[12], 2f32.sqrt());
    let ids: Vec<u32> = (0..mesh.vertex_slots() as u32).collect();
    let far: Vec<[f32; 3]> = mesh
        .positions()
        .iter()
        .map(|p| p.map(|x| x * 2.0))
        .collect();
    edit(&mut mesh, |tx| Ok(tx.set_positions(&ids, &far)?));
    capture_rest_sizes(&mut mesh).unwrap();
    assert_eq!(
        mesh.channel(REST_SIZE_CHANNEL).unwrap()[12],
        2.0 * 2f32.sqrt()
    );
    assert_eq!(pass(&mut mesh, all()).splits, 0);
}

/// Past 64 the float grid doubles: edges one grid step long at 63 land one step
/// long at 66, twice their rest size, and their midpoints round onto an end.
#[test]
fn an_edge_too_short_to_split_at_float_precision_stops_the_pass() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let step = f32::EPSILON * 32.0;
        let mut mesh = mesh_of(
            &[
                63.0 + step,
                0.0,
                0.0,
                63.0 + 2.0 * step,
                0.0,
                0.0,
                63.0 + step,
                step,
                0.0,
            ],
            &[[0, 1, 2]],
        );
        capture_rest_sizes(&mut mesh).unwrap();
        let moved: Vec<[f32; 3]> = mesh
            .positions()
            .iter()
            .map(|p| [p[0] + 3.0, p[1], p[2]])
            .collect();
        edit(&mut mesh, |t| Ok(t.set_positions(&[0, 1, 2], &moved)?));
        let report = pass(&mut mesh, all());
        tx.send((report.splits, mesh.triangle_count())).unwrap();
    });
    let (splits, triangles) = rx
        .recv_timeout(std::time::Duration::from_secs(20))
        .expect("the pass must end");
    assert!(splits < 100, "{splits} splits");
    assert_eq!(triangles, 1 + splits);
}
