//! Tests of [`extrude_faces`] and [`extrude_base`].

use std::collections::{BTreeMap, BTreeSet};

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::self_intersecting_pairs;
use fast_mesh_store::{Mesh, TriangleId, VertexId};

use super::*;
use crate::primitives::{BoxOptions, Primitive, generate};
use crate::test_util::{
    assert_identical_across_threads, band, bowtie_holes, box_mesh, combine, edit, fin, flip, grid,
    grid_without, hemisphere, mesh_of, parts, sweep, uv_sphere,
};
use crate::topology::{EdgeMap, pinch_vertices};
use crate::{OpsError, boundary_loops};

/// How often each directed edge of the alive triangles is used.
fn directed(mesh: &Mesh) -> BTreeMap<(VertexId, VertexId), usize> {
    let mut out = BTreeMap::new();
    for t in 0..mesh.triangle_slots() as TriangleId {
        if let Some([a, b, c]) = mesh.triangle(t) {
            for e in [(a, b), (b, c), (c, a)] {
                *out.entry(e).or_default() += 1;
            }
        }
    }
    out
}

/// Directed edges without their reverse: the open boundary of a consistently
/// wound 2-manifold mesh, which this also asserts `mesh` to be.
fn open_edges(mesh: &Mesh) -> Vec<(VertexId, VertexId)> {
    let edges = directed(mesh);
    assert!(
        edges.values().all(|&n| n == 1),
        "an edge is used twice in one direction"
    );
    let map = EdgeMap::build(mesh).unwrap();
    assert_eq!(
        pinch_vertices(mesh, &map).unwrap(),
        [0u32; 0],
        "pinched vertices"
    );
    edges
        .keys()
        .filter(|&&(a, b)| !edges.contains_key(&(b, a)))
        .copied()
        .collect()
}

/// The number of directed edges without a reverse, without the manifold
/// assertions of [`open_edges`].
fn open_pairs(mesh: &Mesh) -> usize {
    let e = directed(mesh);
    e.keys().filter(|&&(a, b)| !e.contains_key(&(b, a))).count()
}

/// `mesh` is closed, consistently wound, 2-manifold and uses every vertex.
fn assert_closed(mesh: &Mesh, what: &str) {
    assert!(open_edges(mesh).is_empty(), "{what}: open edges");
    for v in 0..mesh.vertex_slots() as VertexId {
        if mesh.is_vertex_alive(v) {
            assert!(mesh.corner_count(v) > 0, "{what}: vertex {v} is unused");
        }
    }
}

/// The signed volume in `f64`.
fn volume(mesh: &Mesh) -> f64 {
    let mut v = 0.0;
    for t in 0..mesh.triangle_slots() as TriangleId {
        if let Some(tri) = mesh.triangle(t) {
            let [a, b, c] = tri.map(|x| point_of(mesh, x));
            v += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
    }
    v / 6.0
}

fn point_of(mesh: &Mesh, v: VertexId) -> [f64; 3] {
    mesh.position(v).unwrap().map(f64::from)
}

fn normal_of(mesh: &Mesh, t: TriangleId) -> [f64; 3] {
    let [a, b, c] = mesh.triangle(t).unwrap().map(|v| point_of(mesh, v));
    let (u, w) = (
        [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
        [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
    );
    [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ]
}

/// What the check should report for `changed` triangles of the written
/// `mesh`, worked out on the whole mesh: the pairs of the engine's own
/// self-intersection search, and the edge neighbours whose normals are
/// opposite.
fn expected_intersections(mesh: &Mesh, changed: &[TriangleId]) -> Vec<TriangleId> {
    let changed: BTreeSet<TriangleId> = changed.iter().copied().collect();
    let bvh = Bvh::build(mesh, &BuildOptions::default()).unwrap();
    let mut pairs = Vec::new();
    self_intersecting_pairs(mesh, &bvh, &mut pairs).unwrap();
    let mut out: BTreeSet<TriangleId> = pairs
        .iter()
        .filter(|(a, b)| changed.contains(a) || changed.contains(b))
        .flat_map(|&(a, b)| [a, b])
        .collect();
    let map = EdgeMap::build(mesh).unwrap();
    for (_, tris) in map.edges() {
        for (i, &a) in tris.iter().enumerate() {
            for &b in &tris[i + 1..] {
                let (na, nb) = (normal_of(mesh, a), normal_of(mesh, b));
                let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
                let size = (dot(na, na) * dot(nb, nb)).sqrt();
                if (changed.contains(&a) || changed.contains(&b))
                    && size > 0.0
                    && dot(na, nb) < -(1.0 - 5e-7) * size
                {
                    out.extend([a, b]);
                }
            }
        }
    }
    out.into_iter().collect()
}

fn out_by(distance: f64) -> ExtrudeFacesOptions {
    ExtrudeFacesOptions {
        distance,
        ..Default::default()
    }
}

/// Extrudes `sel` of `mesh` and returns the report.
fn extrude(
    mesh: &mut Mesh,
    sel: &[TriangleId],
    options: &ExtrudeFacesOptions,
) -> ExtrudeFacesReport {
    edit(mesh, |tx| extrude_faces(tx, sel, options)).0
}

/// Extrudes and checks everything that always holds: the mesh stays 2-manifold
/// and consistently wound with as many open edges as before, the caps keep
/// their ids, and the reported intersections are the expected ones.
fn extrude_checked(
    mesh: &mut Mesh,
    sel: &[TriangleId],
    options: &ExtrudeFacesOptions,
    what: &str,
) -> ExtrudeFacesReport {
    let open = open_edges(mesh).len();
    let (before_tris, before_verts) = (mesh.triangle_count(), mesh.vertex_count());
    let r = extrude(mesh, sel, options);
    assert_eq!(open_edges(mesh).len(), open, "{what}: open edges");
    assert_eq!(
        mesh.triangle_count(),
        before_tris + r.wall_triangles.len(),
        "{what}"
    );
    assert_eq!(
        mesh.vertex_count(),
        before_verts + r.added_vertices,
        "{what}"
    );
    assert!(sel.iter().all(|&t| mesh.is_triangle_alive(t)), "{what}");
    let changed: Vec<TriangleId> = sel.iter().chain(&r.wall_triangles).copied().collect();
    assert_eq!(
        r.self_intersections.as_deref().unwrap(),
        expected_intersections(mesh, &changed),
        "{what}"
    );
    r
}

/// A segmented box and the triangles whose centroid passes `pick`, on its top
/// (`+z`) face.
fn tiled_box(n: u32, pick: impl Fn(f64, f64) -> bool) -> (Mesh, Vec<TriangleId>) {
    let s = generate(&Primitive::Box(BoxOptions {
        size: [f64::from(n), f64::from(n), 2.0],
        segments: [n, n, 1],
    }))
    .unwrap();
    let flat: Vec<f32> = s.positions.iter().flatten().copied().collect();
    let mesh = mesh_of(&flat, &s.triangles);
    let half = f64::from(n) / 2.0;
    let sel = (0..mesh.triangle_slots() as TriangleId)
        .filter(|&t| {
            let p = mesh.triangle(t).unwrap().map(|v| point_of(&mesh, v));
            let c: [f64; 3] = std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0);
            c[2] > 0.999 && pick(c[0] + half, c[1] + half)
        })
        .collect();
    (mesh, sel)
}

fn cell(x: f64, y: f64) -> (u32, u32) {
    (x as u32, y as u32)
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self, below: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % u64::from(below)) as u32
    }
}

// ---- Extrude faces ----

#[test]
fn a_box_face_gains_its_area_times_the_distance() {
    for (direction, distance, gain) in [
        (ExtrudeDirection::RegionNormal, 0.5, 3.0),
        (ExtrudeDirection::VertexNormals, 0.5, 3.0),
        (ExtrudeDirection::Fixed([0.0, 0.0, 3.0]), 0.5, 3.0),
        (ExtrudeDirection::Fixed([0.0, 0.0, -1.0]), -0.5, 3.0),
        // Slanted walls: only the rise counts (a 3-4-5 direction).
        (ExtrudeDirection::Fixed([3.0, 0.0, 4.0]), 5.0, 24.0),
        (ExtrudeDirection::RegionNormal, 8.0, 48.0),
    ] {
        // The top face of a 2 x 3 x 4 box.
        let mut mesh = box_mesh([0.0; 3], [2.0, 3.0, 4.0]);
        let options = ExtrudeFacesOptions {
            distance,
            direction,
            ..Default::default()
        };
        let r = extrude_checked(&mut mesh, &[10, 11], &options, "box face");
        assert_closed(&mesh, "box face");
        assert!(
            (volume(&mesh) - (24.0 + gain)).abs() < 1e-5,
            "{direction:?} by {distance}: {}",
            volume(&mesh)
        );
        assert_eq!(
            (
                r.regions,
                r.cap_triangles,
                r.moved_vertices,
                r.added_vertices
            ),
            (1, 2, 0, 4)
        );
        assert_eq!(r.wall_triangles.len(), 8);
        assert_eq!(r.self_intersections, Some(vec![]));
    }
}

#[test]
fn a_second_extrusion_continues_the_first() {
    let mut mesh = box_mesh([0.0; 3], [2.0, 3.0, 4.0]);
    extrude_checked(&mut mesh, &[10, 11], &out_by(0.5), "first");
    // The selected triangles are the moved caps: select them again.
    extrude_checked(&mut mesh, &[10, 11], &out_by(0.25), "second");
    assert_closed(&mesh, "stacked");
    assert!((volume(&mesh) - 28.5).abs() < 1e-5);
}

#[test]
fn pushing_in_a_part_of_a_face_makes_a_pocket() {
    let (mut mesh, sel) = tiled_box(4, |x, y| {
        let (i, j) = cell(x, y);
        (1..3).contains(&i) && (1..3).contains(&j)
    });
    assert_eq!(sel.len(), 8);
    let r = extrude_checked(&mut mesh, &sel, &out_by(-0.5), "pocket");
    assert_closed(&mesh, "pocket");
    assert!((volume(&mesh) - (32.0 - 2.0)).abs() < 1e-5);
    assert_eq!(r.self_intersections, Some(vec![]));
    // The middle vertex moves; the eight on the rim are copied.
    assert_eq!((r.moved_vertices, r.added_vertices), (1, 8));
}

#[test]
fn a_selection_with_a_hole_gets_walls_around_the_hole() {
    // A ring of 12 cells around a 2 x 2 hole.
    let (mut mesh, sel) = tiled_box(6, |x, y| {
        let (i, j) = cell(x, y);
        (1..5).contains(&i) && (1..5).contains(&j) && !((2..4).contains(&i) && (2..4).contains(&j))
    });
    assert_eq!(sel.len(), 24);
    let r = extrude_checked(&mut mesh, &sel, &out_by(0.75), "ring");
    assert_closed(&mesh, "ring");
    assert_eq!(r.regions, 1);
    // 16 edges outside and 8 around the hole; no vertex is inside.
    assert_eq!(r.wall_triangles.len(), 2 * (16 + 8));
    assert_eq!((r.moved_vertices, r.added_vertices), (0, 24));
    assert!((volume(&mesh) - (72.0 + 12.0 * 0.75)).abs() < 1e-5);
    assert_eq!(r.self_intersections, Some(vec![]));
}

#[test]
fn several_regions_move_along_their_own_normals() {
    // Top and bottom of a box: opposite ways.
    let mut mesh = box_mesh([0.0; 3], [2.0, 3.0, 4.0]);
    let r = extrude_checked(&mut mesh, &[8, 9, 10, 11], &out_by(0.5), "two faces");
    assert_eq!(r.regions, 2);
    assert_closed(&mesh, "two faces");
    assert!((volume(&mesh) - 30.0).abs() < 1e-5);

    // Two cells of one face that share one corner: each gets its own copy of
    // it, and each vertex stays manifold.
    let (mut mesh, sel) = tiled_box(4, |x, y| [(1, 1), (2, 2)].contains(&cell(x, y)));
    assert_eq!(sel.len(), 4);
    let r = extrude_checked(&mut mesh, &sel, &out_by(0.5), "corner to corner");
    assert_eq!((r.regions, r.added_vertices), (2, 8));
    assert_closed(&mesh, "corner to corner");
    assert!((volume(&mesh) - 33.0).abs() < 1e-5);
    // Their coincident corners touch, and that is reported.
    assert!(!r.self_intersections.unwrap().is_empty());
}

#[test]
fn faces_across_an_edge_move_together() {
    // The top and the +x side share an edge, so they are one region and move
    // along their average normal as one piece.
    let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
    let r = extrude_checked(&mut mesh, &[2, 3, 10, 11], &out_by(1.0), "L");
    assert_eq!(r.regions, 1);
    assert_closed(&mesh, "L");
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let p = mesh.triangle(10).unwrap().map(|v| point_of(&mesh, v));
    assert!((p[0][2] - (1.0 + h)).abs() < 1e-6);
    // Each face sweeps its area times the rise along its own normal.
    assert!((volume(&mesh) - (1.0 + 2.0 * h)).abs() < 1e-5);
}

#[test]
fn random_selections_of_a_closed_surface_stay_closed() {
    let mut rng = Lcg(7);
    for round in 0..40 {
        let mesh0 = uv_sphere(10, 14, 1.0);
        let n = mesh0.triangle_slots() as u32;
        // Scattered triangles, then patches grown around seeds.
        let mut picked = BTreeSet::new();
        if round % 2 == 0 {
            for _ in 0..1 + rng.next(n / 3) {
                picked.insert(rng.next(n));
            }
        } else {
            for _ in 0..1 + rng.next(4) {
                let seed = rng.next(n);
                let c = normal_of(&mesh0, seed);
                let reach = 0.2 + 0.1 * f64::from(rng.next(8));
                for t in 0..n {
                    let q = point_of(&mesh0, mesh0.triangle(t).unwrap()[0]);
                    let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                    if (c[0] * q[0] + c[1] * q[1] + c[2] * q[2]) / l > reach {
                        picked.insert(t);
                    }
                }
            }
        }
        let sel: Vec<TriangleId> = picked.into_iter().collect();
        for direction in [
            ExtrudeDirection::RegionNormal,
            ExtrudeDirection::VertexNormals,
            ExtrudeDirection::Fixed([0.3, -0.2, 1.0]),
        ] {
            for distance in [0.05, -0.1, 1.5] {
                let options = ExtrudeFacesOptions {
                    distance,
                    direction,
                    ..Default::default()
                };
                let mut mesh = uv_sphere(10, 14, 1.0);
                let what = format!("round {round}, {direction:?}, {distance}");
                extrude_checked(&mut mesh, &sel, &options, &what);
                assert_closed(&mesh, &what);
            }
        }
    }
}

#[test]
fn an_open_sheet_gets_walls_on_its_border_too() {
    // All of a 4 x 4 grid: the sheet lifts off and its walls' feet are the
    // open border.
    let mut mesh = grid(4);
    let all: Vec<TriangleId> = (0..32).collect();
    let r = extrude_checked(&mut mesh, &all, &out_by(2.0), "sheet");
    assert_eq!((r.moved_vertices, r.added_vertices), (9, 16));
    assert_eq!(r.wall_triangles.len(), 2 * 16);
    let loops = boundary_loops(&mesh).unwrap();
    assert_eq!(loops.len(), 1);
    assert!(
        loops[0]
            .vertices
            .iter()
            .all(|&v| point_of(&mesh, v)[2] == 0.0)
    );
    assert!(mesh.position(12).unwrap()[2] == 2.0);

    // Around two holes that meet at a vertex.
    let mut mesh = bowtie_holes();
    let all: Vec<TriangleId> = (0..mesh.triangle_slots() as TriangleId).collect();
    let before = open_pairs(&mesh);
    let r = extrude(&mut mesh, &all, &out_by(1.0));
    assert_eq!(open_pairs(&mesh), before);
    // 16 on the border, 8 around the holes with the shared corner counted
    // twice: it has two fans, each with its own copy.
    assert_eq!(r.added_vertices, 16 + 8);
}

#[test]
fn vertex_normals_follow_the_surface() {
    // A whole closed sphere moves out along its vertex normals, with no wall.
    let mut mesh = uv_sphere(16, 24, 1.0);
    let all: Vec<TriangleId> = (0..mesh.triangle_slots() as TriangleId).collect();
    let options = ExtrudeFacesOptions {
        distance: 0.5,
        direction: ExtrudeDirection::VertexNormals,
        ..Default::default()
    };
    let r = extrude_checked(&mut mesh, &all, &options, "sphere");
    assert_eq!(
        (r.regions, r.added_vertices, r.wall_triangles.len()),
        (1, 0, 0)
    );
    assert_eq!(r.moved_vertices, mesh.vertex_count());
    for p in mesh.positions() {
        let l = p.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((l - 1.5).abs() < 2e-2, "{l}");
    }
    // Its region normal cancels out.
    let mut mesh = uv_sphere(16, 24, 1.0);
    let mut tx = mesh.begin_edit();
    assert!(matches!(
        extrude_faces(&mut tx, &all, &out_by(1.0)),
        Err(OpsError::InvalidInput(m)) if m.contains("cancel")
    ));
    assert!(tx.commit().unwrap().is_empty());
}

#[test]
fn to_world_measures_distances_and_normals() {
    // A face seen through a scale of 2 along z: a distance of 1 is half a
    // local unit.
    let mut mesh = box_mesh([0.0; 3], [2.0, 3.0, 4.0]);
    let options = ExtrudeFacesOptions {
        to_world: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
        ..out_by(1.0)
    };
    extrude(&mut mesh, &[10, 11], &options);
    assert!((volume(&mesh) - (24.0 + 6.0 * 0.5)).abs() < 1e-5);

    // A slanted face under a rotation and a non-uniform scale, as the linear
    // part of a placement: the caps move along the mapped surface's normal,
    // by the distance, for both normal modes.
    let to_world = [[0.0, -3.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.5]];
    let flat = [0.0, 0.0, 0.0, 4.0, 0.0, 1.0, 0.0, 4.0, 2.0, 0.0, 0.0, -9.0];
    let tris = [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]];
    for direction in [
        ExtrudeDirection::RegionNormal,
        ExtrudeDirection::VertexNormals,
    ] {
        let mut mesh = mesh_of(&flat, &tris);
        let before = mesh.triangle(0).unwrap().map(|v| point_of(&mesh, v));
        let options = ExtrudeFacesOptions {
            distance: 0.25,
            direction,
            to_world,
            ..Default::default()
        };
        extrude(&mut mesh, &[0], &options);
        let after = mesh.triangle(0).unwrap().map(|v| point_of(&mesh, v));
        let w = before.map(|p| apply(&to_world, p));
        let e = |i: usize| [0, 1, 2].map(|k| w[i][k] - w[0][k]);
        let n = crate::topology::cross(e(1), e(2));
        let l = n.iter().map(|x| x * x).sum::<f64>().sqrt();
        for (b, a) in before.iter().zip(&after) {
            let moved = apply(&to_world, std::array::from_fn(|k| a[k] - b[k]));
            for k in 0..3 {
                assert!((moved[k] - 0.25 * n[k] / l).abs() < 1e-5, "{direction:?}");
            }
        }
    }
    // A fixed direction is given in the mapped frame.
    let mut mesh = box_mesh([0.0; 3], [2.0, 3.0, 4.0]);
    let at = |m: &Mesh| m.triangle(10).unwrap().map(|v| point_of(m, v));
    let before = at(&mesh);
    let options = ExtrudeFacesOptions {
        distance: 6.0,
        direction: ExtrudeDirection::Fixed([0.0, 0.0, 10.0]),
        to_world,
        ..Default::default()
    };
    extrude(&mut mesh, &[10, 11], &options);
    for (b, a) in before.iter().zip(&at(&mesh)) {
        assert!((a[2] - b[2] - 12.0).abs() < 1e-5 && (a[0] - b[0]).abs() < 1e-5);
    }
}

#[test]
fn self_intersections_are_reported_and_written() {
    // The top pushed through the bottom of a box.
    let (mut mesh, sel) = tiled_box(4, |x, y| cell(x, y) == (1, 1));
    let r = extrude_checked(&mut mesh, &sel, &out_by(-3.0), "through");
    assert_closed(&mesh, "through");
    let hit = r.self_intersections.unwrap();
    assert!(hit.iter().any(|t| r.wall_triangles.contains(t)));
    // ... and the bottom's own triangles, which did not change.
    assert!(
        hit.iter()
            .any(|t| !r.wall_triangles.contains(t) && !sel.contains(t))
    );

    // A whole face pushed straight in: its walls lie back on the sides, which
    // share their edges.
    let mut mesh = box_mesh([0.0; 3], [2.0, 3.0, 4.0]);
    let r = extrude_checked(&mut mesh, &[10, 11], &out_by(-1.0), "flat fold");
    assert!((volume(&mesh) - 18.0).abs() < 1e-5);
    let hit = r.self_intersections.unwrap();
    assert!(!hit.is_empty() && hit.iter().any(|&t| t < 8));

    // The check can be left out.
    let mut mesh = box_mesh([0.0; 3], [2.0, 3.0, 4.0]);
    let options = ExtrudeFacesOptions {
        check_self_intersections: false,
        ..out_by(-1.0)
    };
    assert_eq!(
        extrude(&mut mesh, &[10, 11], &options).self_intersections,
        None
    );

    // An intersection that was there before and is left alone is not listed.
    let mut mesh = combine(&[
        &box_mesh([0.0; 3], [2.0; 3]),
        &box_mesh([1.0; 3], [3.0; 3]),
        &box_mesh([10.0; 3], [11.0; 3]),
    ]);
    let r = extrude_checked(&mut mesh, &[34, 35], &out_by(0.5), "elsewhere");
    assert_eq!(r.self_intersections, Some(vec![]));
}

#[test]
fn invalid_selections_and_options_write_nothing() {
    let check = |mut mesh: Mesh, sel: &[TriangleId], o: ExtrudeFacesOptions, what: &str| {
        let before = parts(&mesh);
        let mut tx = mesh.begin_edit();
        let e = extrude_faces(&mut tx, sel, &o).unwrap_err();
        assert!(
            matches!(e, OpsError::InvalidInput(m) if m.contains(what)),
            "{e:?}, want {what}"
        );
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    };
    let cube = || box_mesh([0.0; 3], [1.0; 3]);
    check(cube(), &[], out_by(1.0), "at least one");
    check(cube(), &[12], out_by(1.0), "dead or out of range");
    check(cube(), &[10], out_by(0.0), "distance");
    check(cube(), &[10], out_by(f64::NAN), "distance");
    let fixed = |d| ExtrudeFacesOptions {
        direction: ExtrudeDirection::Fixed(d),
        ..Default::default()
    };
    check(cube(), &[10], fixed([0.0; 3]), "direction");
    check(cube(), &[10], fixed([f64::INFINITY, 0.0, 0.0]), "direction");
    check(
        cube(),
        &[10],
        ExtrudeFacesOptions {
            to_world: [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ..Default::default()
        },
        "to_world",
    );
    check(cube(), &[10], out_by(1e39), "f32");
    check(fin(), &[0, 1], out_by(1.0), "three or more");
    check(
        flip(&cube(), &[11]),
        &[10, 11],
        out_by(1.0),
        "wound against",
    );
    let mut dead = cube();
    edit(&mut dead, |tx| Ok(tx.remove_triangles(&[3])?));
    check(dead, &[2, 3], out_by(1.0), "dead");
    let slit = mesh_of(
        &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[[0, 1, 2], [0, 1, 1]],
    );
    check(slit, &[0, 1], out_by(1.0), "repeats a vertex");
}

#[test]
fn duplicates_in_the_selection_count_once_and_undo_restores() {
    let mut mesh = box_mesh([0.0; 3], [2.0, 3.0, 4.0]);
    let before = parts(&mesh);
    let mut tx = mesh.begin_edit();
    let r = extrude_faces(&mut tx, &[11, 10, 11], &out_by(0.5)).unwrap();
    assert_eq!(r.cap_triangles, 2);
    let patch = tx.commit().unwrap();
    assert!(!patch.is_empty());
    patch.revert(&mut mesh).unwrap();
    let after = parts(&mesh);
    assert_eq!(after.1, before.1);
    assert_eq!(after.0[..before.0.len()], before.0[..]);
    assert_eq!(mesh.vertex_count(), 8);
}

#[test]
fn faces_out_of_memory_write_nothing() {
    let (mesh, sel) = tiled_box(4, |x, y| [(1, 1), (2, 2), (2, 1)].contains(&cell(x, y)));
    let points = sweep(mesh, |tx| {
        extrude_faces(tx, &sel, &out_by(-3.0)).map(|_| ())
    });
    assert!(points > 20);
}

// ---- Extrude base ----

fn down(level: BaseLevel) -> ExtrudeBaseOptions {
    ExtrudeBaseOptions::along([0.0, 0.0, -1.0], level)
}

/// Extrudes `mesh`'s base and checks the reported counts and intersections.
fn base_checked(mesh: &mut Mesh, options: &ExtrudeBaseOptions, what: &str) -> ExtrudeBaseReport {
    let (tris, verts) = (mesh.triangle_count(), mesh.vertex_count());
    let old: BTreeSet<TriangleId> = (0..mesh.triangle_slots() as TriangleId)
        .filter(|&t| mesh.is_triangle_alive(t))
        .collect();
    let r = edit(mesh, |tx| extrude_base(tx, options)).0;
    assert_eq!(
        mesh.triangle_count(),
        tris + r.wall_triangles + r.cap_triangles,
        "{what}"
    );
    assert_eq!(
        mesh.vertex_count(),
        verts + r.base_vertices + r.cap_vertices,
        "{what}"
    );
    let new: Vec<TriangleId> = (0..mesh.triangle_slots() as TriangleId)
        .filter(|&t| mesh.is_triangle_alive(t) && !old.contains(&t))
        .collect();
    assert_eq!(
        r.self_intersections.as_deref().unwrap(),
        expected_intersections(mesh, &new),
        "{what}"
    );
    r
}

/// The area of a regular `n`-gon of circumradius 1.
fn ngon_area(n: u32) -> f64 {
    0.5 * f64::from(n) * (std::f64::consts::TAU / f64::from(n)).sin()
}

#[test]
fn a_hemisphere_becomes_a_closed_solid() {
    let open = hemisphere(12, 32, 1.0);
    // The dome above the equator plane, which passes through the origin.
    let dome = volume(&open);
    for refine in [true, false] {
        let mut mesh = hemisphere(12, 32, 1.0);
        let options = ExtrudeBaseOptions {
            refine,
            ..down(BaseLevel::At(0.5))
        };
        let r = base_checked(&mut mesh, &options, "hemisphere");
        assert_closed(&mesh, "hemisphere");
        let want = dome + ngon_area(32) * 0.5;
        assert!(
            (volume(&mesh) - want).abs() < 1e-5,
            "{} vs {want}",
            volume(&mesh)
        );
        // Close to the smooth solid too.
        let smooth = std::f64::consts::PI * (2.0 / 3.0 + 0.5);
        assert!((volume(&mesh) - smooth).abs() < 0.05 * smooth);
        assert_eq!(r.loops.len(), 1);
        assert_eq!((r.level, r.base_vertices, r.wall_triangles), (0.5, 32, 64));
        assert_eq!(r.cap_triangles, 30 + 2 * r.cap_vertices);
        assert_eq!(r.cap_vertices > 0, refine);
        assert!(!r.flipped && !r.capped);
        assert_eq!(r.self_intersections, Some(vec![]));
        // Every new vertex is on the plane.
        for v in open.vertex_slots() as VertexId..mesh.vertex_slots() as VertexId {
            assert!((mesh.position(v).unwrap()[2] + 0.5).abs() < 1e-6);
        }
    }
}

#[test]
fn a_surface_facing_the_base_is_turned_around() {
    let open = hemisphere(12, 32, 1.0);
    let all: Vec<TriangleId> = (0..open.triangle_slots() as TriangleId).collect();
    let mut mesh = flip(&open, &all);
    let r = base_checked(&mut mesh, &down(BaseLevel::At(0.5)), "inward");
    assert!(r.flipped);
    assert_closed(&mesh, "inward");
    assert!((volume(&mesh) - (volume(&open) + ngon_area(32) * 0.5)).abs() < 1e-5);
    // Another surface in the mesh keeps its winding.
    let mut mesh = combine(&[&flip(&open, &all), &box_mesh([5.0; 3], [6.0; 3])]);
    let untouched = mesh.triangle(open.triangle_slots() as TriangleId).unwrap();
    let r = base_checked(&mut mesh, &down(BaseLevel::At(0.5)), "two parts");
    assert!(r.flipped);
    assert_eq!(
        mesh.triangle(open.triangle_slots() as TriangleId).unwrap(),
        untouched
    );
    assert_closed(&mesh, "two parts");
}

#[test]
fn a_rim_with_straight_runs_caps_exactly() {
    // A square sheet: every side of its rim is a run of collinear vertices.
    for refine in [true, false] {
        for direction in [[0.0, 0.0, -1.0], [0.3, 0.2, -1.0]] {
            let mut mesh = grid(8);
            let options = ExtrudeBaseOptions {
                direction,
                normal: [0.0, 0.0, -1.0],
                refine,
                ..down(BaseLevel::Beyond(2.0))
            };
            let r = base_checked(&mut mesh, &options, "sheet");
            assert_closed(&mesh, "sheet");
            // A slanted prism has the same volume as an upright one.
            assert!((volume(&mesh) - 128.0).abs() < 1e-3, "{}", volume(&mesh));
            assert_eq!((r.level, r.base_vertices, r.wall_triangles), (2.0, 32, 64));
            assert_eq!(r.self_intersections, Some(vec![]));
        }
    }
}

#[test]
fn the_normal_may_have_any_length() {
    // Levels are measured along the normal as given: half as far here.
    let mut mesh = grid(4);
    let options = ExtrudeBaseOptions {
        direction: [0.0, 0.0, -0.1],
        normal: [0.0, 0.0, -2.0],
        ..down(BaseLevel::Beyond(2.0))
    };
    let r = base_checked(&mut mesh, &options, "scaled normal");
    assert_eq!(r.level, 2.0);
    assert!((volume(&mesh) - 16.0).abs() < 1e-5);
}

#[test]
fn a_bumpy_rim_reaches_one_plane() {
    // A wavy sheet whose rim is not flat.
    let n = 12u32;
    let mut flat = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            let z = (0.7 * f64::from(x)).sin() * (0.5 * f64::from(y)).cos();
            flat.extend([x as f32, y as f32, z as f32]);
        }
    }
    let mut mesh = mesh_of(&flat, &parts(&grid(n)).1);
    let r = base_checked(&mut mesh, &down(BaseLevel::Beyond(0.5)), "bumpy");
    assert_closed(&mesh, "bumpy");
    assert!(volume(&mesh) > 0.0);
    assert_eq!(r.self_intersections, Some(vec![]));
    // Beyond every vertex, by the offset.
    let low = flat
        .iter()
        .skip(2)
        .step_by(3)
        .fold(f32::MAX, |a, &z| a.min(z));
    assert!((r.level - (0.5 - f64::from(low))).abs() < 1e-6);

    // A plane between the rim and the surface's low point cuts it, reported.
    let mut dented = parts(&grid(6)).0;
    dented[3 * 24 + 2] = -3.0;
    let mut mesh = mesh_of(&dented, &parts(&grid(6)).1);
    let r = base_checked(&mut mesh, &down(BaseLevel::At(1.0)), "cutting plane");
    assert_closed(&mesh, "cutting plane");
    let hit = r.self_intersections.unwrap();
    assert!(hit.iter().any(|&t| t < 72) && hit.iter().any(|&t| t >= 72));
}

#[test]
fn holes_in_the_surface_become_holes_in_the_cap() {
    let hole: Vec<(u32, u32)> = (2..4).flat_map(|y| (2..4).map(move |x| (x, y))).collect();
    for refine in [true, false] {
        // Only the outer rim: the hole stays open.
        let mut mesh = grid_without(6, &hole);
        let options = ExtrudeBaseOptions {
            refine,
            ..down(BaseLevel::Beyond(1.0))
        };
        let r = base_checked(&mut mesh, &options, "largest");
        assert_eq!((r.loops.len(), r.base_vertices), (1, 24));
        let left = boundary_loops(&mesh).unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].vertices.len(), 8);

        // Both: a ring-shaped solid.
        let mut mesh = grid_without(6, &hole);
        // The hole's middle vertex is used by no triangle.
        edit(&mut mesh, |tx| Ok(tx.remove_vertices(&[24])?));
        let options = ExtrudeBaseOptions {
            loops: BaseLoops::All,
            refine,
            ..down(BaseLevel::Beyond(1.0))
        };
        let r = base_checked(&mut mesh, &options, "ring");
        assert_eq!((r.loops.len(), r.base_vertices), (2, 32));
        assert_closed(&mesh, "ring");
        assert!((volume(&mesh) - 32.0).abs() < 1e-4, "{}", volume(&mesh));
        assert_eq!(r.self_intersections, Some(vec![]));
    }

    // By id, each loop in turn.
    let mesh = grid_without(6, &hole);
    let loops = boundary_loops(&mesh).unwrap();
    for l in &loops {
        let mut mesh = grid_without(6, &hole);
        let options = ExtrudeBaseOptions {
            loops: BaseLoops::Ids(vec![l.id]),
            ..down(BaseLevel::Beyond(1.0))
        };
        let r = edit(&mut mesh, |tx| extrude_base(tx, &options)).0;
        assert_eq!(r.loops, [l.id]);
        assert_eq!(r.base_vertices, l.vertices.len());
    }
}

#[test]
fn rings_nest_to_any_depth() {
    // Two flat rings, one inside the other's hole, and a disk inside both.
    let ring = |inner: f64, outer: f64| {
        let at = |r: f64| {
            move |t: f64| {
                let a = std::f64::consts::TAU * t;
                [r * a.cos(), r * a.sin(), 0.0]
            }
        };
        band(24, at(inner), at(outer))
    };
    let mut disk = vec![0.0f32, 0.0, 0.0];
    for i in 0..24u32 {
        let a = std::f64::consts::TAU * f64::from(i) / 24.0;
        disk.extend([a.cos() as f32, a.sin() as f32, 0.0]);
    }
    let fan: Vec<[u32; 3]> = (0..24).map(|i| [0, 1 + i, 1 + (i + 1) % 24]).collect();
    let mut mesh = combine(&[&ring(2.0, 3.0), &ring(4.0, 5.0), &mesh_of(&disk, &fan)]);
    let options = ExtrudeBaseOptions {
        loops: BaseLoops::All,
        ..down(BaseLevel::At(1.0))
    };
    let r = base_checked(&mut mesh, &options, "nested");
    assert_eq!(r.loops.len(), 5);
    assert_closed(&mesh, "nested");
    let want = ngon_area(24) * (25.0 - 16.0 + 9.0 - 4.0 + 1.0);
    assert!((volume(&mesh) - want).abs() < 1e-3, "{}", volume(&mesh));
    assert_eq!(r.self_intersections, Some(vec![]));
}

#[test]
fn several_sheets_each_get_a_base() {
    let shifted = {
        let (mut p, t) = parts(&grid(3));
        for q in p.as_chunks_mut::<3>().0 {
            q[0] += 10.0;
            q[2] += 1.0;
        }
        mesh_of(&p, &t)
    };
    let mut mesh = combine(&[&grid(3), &shifted]);
    let options = ExtrudeBaseOptions {
        loops: BaseLoops::All,
        ..down(BaseLevel::Beyond(1.0))
    };
    let r = base_checked(&mut mesh, &options, "sheets");
    assert_eq!(r.loops.len(), 2);
    assert_closed(&mesh, "sheets");
    assert!((volume(&mesh) - 9.0 * (1.0 + 2.0)).abs() < 1e-4);
}

#[test]
fn invalid_bases_write_nothing() {
    let check = |mut mesh: Mesh, o: ExtrudeBaseOptions, what: &str| {
        let before = parts(&mesh);
        let mut tx = mesh.begin_edit();
        let e = extrude_base(&mut tx, &o).unwrap_err();
        assert!(
            matches!(e, OpsError::InvalidInput(m) if m.contains(what)),
            "{e:?}, want {what}"
        );
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    };
    let ok = || down(BaseLevel::Beyond(1.0));
    check(box_mesh([0.0; 3], [1.0; 3]), ok(), "no boundary loop");
    check(grid(3), down(BaseLevel::At(-1.0)), "beyond every boundary");
    // In the plane of the rim is not beyond it.
    check(
        grid(3),
        down(BaseLevel::Beyond(0.0)),
        "beyond every boundary",
    );
    check(grid(3), down(BaseLevel::At(f64::NAN)), "level");
    check(grid(3), down(BaseLevel::Beyond(f64::INFINITY)), "level");
    check(grid(3), down(BaseLevel::At(1e39)), "f32");
    let way = |direction, normal| ExtrudeBaseOptions {
        direction,
        normal,
        ..ok()
    };
    check(grid(3), way([0.0; 3], [0.0, 0.0, -1.0]), "non-zero");
    check(grid(3), way([0.0, 0.0, -1.0], [0.0; 3]), "non-zero");
    check(
        grid(3),
        way([0.0, 0.0, -1.0], [0.0, 0.0, 1.0]),
        "toward the base plane",
    );
    check(
        grid(3),
        way([1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        "toward the base plane",
    );
    check(
        grid(3),
        ExtrudeBaseOptions {
            loops: BaseLoops::Ids(Vec::new()),
            ..ok()
        },
        "at least one loop",
    );
    // Edge on: the outline has no area.
    check(
        grid(3),
        ExtrudeBaseOptions::along([1.0, 0.0, 0.0], BaseLevel::Beyond(1.0)),
        "no area",
    );
    // A rim that runs over itself seen from above.
    let curl = {
        let (mut p, t) = parts(&grid(8));
        for q in p.as_chunks_mut::<3>().0 {
            let a = f64::from(q[0]) * 1.1;
            (q[0], q[2]) = ((a.sin() * 2.0) as f32, (a.cos() * 2.0) as f32);
        }
        mesh_of(&p, &t)
    };
    check(curl, ok(), "crosses itself");
    // Two sheets above each other: the outlines nest, the same way round.
    let above = {
        let (mut p, t) = parts(&grid(2));
        for q in p.as_chunks_mut::<3>().0 {
            (q[0], q[1], q[2]) = (q[0] + 2.0, q[1] + 2.0, 3.0);
        }
        combine(&[&grid(6), &mesh_of(&p, &t)])
    };
    check(
        above,
        ExtrudeBaseOptions {
            loops: BaseLoops::All,
            ..ok()
        },
        "opposite ways",
    );

    // A loop id that is not there (any more).
    let mut mesh = grid(3);
    let mut id = boundary_loops(&mesh).unwrap()[0].id;
    id.hash ^= 1;
    let o = ExtrudeBaseOptions {
        loops: BaseLoops::Ids(vec![id]),
        ..ok()
    };
    let mut tx = mesh.begin_edit();
    assert_eq!(extrude_base(&mut tx, &o), Err(OpsError::StaleLoops));
    assert!(tx.commit().unwrap().is_empty());
}

#[test]
fn base_out_of_memory_writes_nothing() {
    for (mesh, loops) in [
        (hemisphere(3, 8, 1.0), BaseLoops::Largest),
        (grid_without(3, &[(1, 1)]), BaseLoops::All),
    ] {
        let options = ExtrudeBaseOptions {
            loops,
            ..down(BaseLevel::Beyond(1.0))
        };
        let points = sweep(mesh, |tx| extrude_base(tx, &options).map(|_| ()));
        assert!(points > 20);
    }
}

#[test]
fn results_do_not_depend_on_threads() {
    assert_identical_across_threads("extrude faces", || {
        let mut mesh = uv_sphere(40, 60, 1.0);
        let sel: Vec<TriangleId> = (0..mesh.triangle_slots() as TriangleId)
            .filter(|t| t % 7 < 4)
            .collect();
        let r = extrude(&mut mesh, &sel, &out_by(-0.4));
        (r, parts(&mesh))
    });
    assert_identical_across_threads("extrude base", || {
        let mut mesh = hemisphere(40, 120, 1.0);
        let r = edit(&mut mesh, |tx| extrude_base(tx, &down(BaseLevel::At(0.5)))).0;
        (r, parts(&mesh))
    });
}
