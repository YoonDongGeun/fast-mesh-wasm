//! `trim_by_curve` (#273): what its report counts, and which component of a
//! mesh it cuts.

use std::collections::HashMap;

use fast_mesh_store::{Mesh, TriangleId, par};

use crate::{
    MeshId, MeshOptions, Placement, Scene, SceneError, SceneOptions, TrimCurveOptions, TrimReport,
};

type Buffers = (Vec<f32>, Vec<u32>);

/// A closed ball of `32 x 16` quads, 960 triangles.
fn ball(c: [f32; 3], r: f32) -> Buffers {
    let (rings, segs) = (16u32, 32u32);
    let mut p = vec![c[0], c[1] + r, c[2]];
    for i in 1..rings {
        let (s, h) = (std::f32::consts::PI * i as f32 / rings as f32).sin_cos();
        for j in 0..segs {
            let (st, ct) = (std::f32::consts::TAU * j as f32 / segs as f32).sin_cos();
            p.extend([c[0] + r * s * ct, c[1] + r * h, c[2] + r * s * st]);
        }
    }
    p.extend([c[0], c[1] - r, c[2]]);
    let south = 1 + (rings - 1) * segs;
    let ring = |i: u32, j: u32| 1 + (i - 1) * segs + j % segs;
    let mut ix = Vec::new();
    for j in 0..segs {
        ix.extend([0, ring(1, j + 1), ring(1, j)]);
        ix.extend([south, ring(rings - 1, j), ring(rings - 1, j + 1)]);
    }
    for i in 1..rings - 1 {
        for j in 0..segs {
            let (a, b) = (ring(i, j), ring(i, j + 1));
            let (c, d) = (ring(i + 1, j), ring(i + 1, j + 1));
            ix.extend([a, b, c, b, d, c]);
        }
    }
    (p, ix)
}

/// A torus around y (major radius 1, minor 0.35) of `48 x 24` quads.
fn torus() -> Buffers {
    let (nu, nv) = (48u32, 24u32);
    let mut p = Vec::new();
    for i in 0..nu {
        let (sa, ca) = (std::f32::consts::TAU * i as f32 / nu as f32).sin_cos();
        for j in 0..nv {
            let (sb, cb) = (std::f32::consts::TAU * j as f32 / nv as f32).sin_cos();
            let r = 1.0 + 0.35 * cb;
            p.extend([r * ca, 0.35 * sb, r * sa]);
        }
    }
    let at = |i: u32, j: u32| (i % nu) * nv + j % nv;
    let mut ix = Vec::new();
    for i in 0..nu {
        for j in 0..nv {
            let (a, b) = (at(i, j), at(i, j + 1));
            let (c, d) = (at(i + 1, j), at(i + 1, j + 1));
            ix.extend([a, b, c, b, d, c]);
        }
    }
    (p, ix)
}

/// A flat open sheet of `n x n` quads over `[0, size]` squared, at `at`.
fn sheet(n: u32, size: f32, at: [f32; 3]) -> Buffers {
    let mut p = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let (x, y) = (i as f32 / n as f32 * size, j as f32 / n as f32 * size);
            p.extend([at[0] + x, at[1] + y, at[2]]);
        }
    }
    let mut ix = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            ix.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
        }
    }
    (p, ix)
}

/// Every part's triangles in one mesh, sharing no vertex.
fn merged(parts: &[Buffers]) -> Mesh {
    let (mut p, mut ix) = (Vec::new(), Vec::new());
    for (positions, indices) in parts {
        let at = (p.len() / 3) as u32;
        p.extend_from_slice(positions);
        ix.extend(indices.iter().map(|v| v + at));
    }
    Mesh::from_buffers(&p, &ix).unwrap()
}

/// A loop around the bottom of the ball at `c`, `polar` radians from its pole.
fn around_pole(c: [f64; 3], r: f64, polar: f64) -> Vec<[f64; 3]> {
    (0..8)
        .map(|k| {
            let (st, ct) = (f64::from(k) * std::f64::consts::TAU / 8.0 + 0.1).sin_cos();
            let (s, h) = polar.sin_cos();
            [c[0] + r * s * ct, c[1] - r * h, c[2] + r * s * st]
        })
        .collect()
}

/// A loop once around the torus's tube, off its mesh lines: it has one side.
fn around_tube() -> Vec<[f64; 3]> {
    [0.0, 0.3, 0.0, -0.3, 0.2]
        .iter()
        .enumerate()
        .map(|(k, a): (usize, &f64)| {
            let t = k as f64 * std::f64::consts::TAU / 5.0;
            let r = 1.0 + 0.35 * t.cos();
            [r * a.cos(), 0.35 * t.sin(), r * a.sin()]
        })
        .collect()
}

/// A square loop on the plane `z`.
fn square([x0, y0]: [f64; 2], [x1, y1]: [f64; 2], z: f64) -> Vec<[f64; 3]> {
    vec![[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]
}

fn closed(points: &[[f64; 3]], delete_inside: bool) -> TrimCurveOptions {
    TrimCurveOptions {
        points: points.to_vec(),
        closed: true,
        delete_inside,
        snap_tolerance: 0.0,
    }
}

fn add(mesh: Mesh) -> (Scene, MeshId) {
    add_at(mesh, Placement::IDENTITY)
}

fn add_at(mesh: Mesh, placement: Placement) -> (Scene, MeshId) {
    let mut scene = Scene::new(SceneOptions::default());
    let options = MeshOptions {
        placement,
        ..MeshOptions::default()
    };
    let id = scene.add_mesh(mesh, options).unwrap();
    (scene, id)
}

/// The area of the triangles `of` picks, under a scale.
fn area_of(mesh: &Mesh, scale: [f64; 3], of: impl Fn(TriangleId) -> bool) -> f64 {
    let mut total = 0.0;
    for t in (0..mesh.triangle_slots() as TriangleId).filter(|&t| of(t)) {
        let Some(corners) = mesh.triangle(t) else {
            continue;
        };
        let p = corners.map(|v| {
            let p = mesh.position(v).unwrap();
            [0, 1, 2].map(|k| f64::from(p[k]) * scale[k])
        });
        let e = |i: usize| [0, 1, 2].map(|k| p[i][k] - p[0][k]);
        let (a, b) = (e(1), e(2));
        let n = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        total += 0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    }
    total
}

fn area(scene: &Scene, id: MeshId) -> f64 {
    let scale = scene.placement(id).unwrap().scale;
    area_of(scene.mesh(id).unwrap(), scale, |_| true)
}

/// Every alive triangle's corner ids and positions, by slot.
fn triangles(mesh: &Mesh) -> HashMap<TriangleId, [(u32, [f32; 3]); 3]> {
    (0..mesh.triangle_slots() as TriangleId)
        .filter_map(|t| {
            let corners = mesh.triangle(t)?;
            Some((t, corners.map(|v| (v, mesh.position(v).unwrap()))))
        })
        .collect()
}

/// Trims, and checks the report against the mesh before and after.
fn trim(scene: &mut Scene, id: MeshId, options: &TrimCurveOptions) -> TrimReport {
    let before = (scene.mesh(id).unwrap().triangle_count(), area(scene, id));
    let r = scene.trim_by_curve(id, options).unwrap();
    let after = (scene.mesh(id).unwrap().triangle_count(), area(scene, id));
    assert_eq!(
        r.triangles_removed,
        r.imprint_triangles + r.cut_triangles,
        "{r:?}"
    );
    assert_eq!(
        before.0 + r.triangles_added - r.triangles_removed,
        after.0,
        "{r:?}"
    );
    assert_eq!(r.cut_triangles == 0, r.removed_area == 0.0, "{r:?}");
    let lost = before.1 - after.1;
    assert!(
        (r.removed_area - lost).abs() < 1e-6 * before.1,
        "{r:?}: lost {lost}"
    );
    r
}

#[test]
fn a_loop_that_separates_nothing_reports_no_cut() {
    let other = ball([4.0, 0.0, 0.0], 0.3);
    for parts in [
        vec![torus()],
        vec![torus(), other.clone()],
        vec![other, torus()],
    ] {
        for delete_inside in [true, false] {
            let (mut scene, id) = add(merged(&parts));
            let before = triangles(scene.mesh(id).unwrap());
            let r = trim(&mut scene, id, &closed(&around_tube(), delete_inside));
            assert_eq!((r.cut_triangles, r.removed_area), (0, 0.0), "{r:?}");
            assert!(r.imprint_triangles > 40, "{r:?}");
            assert_eq!(r.triangles_removed, r.imprint_triangles);
            assert!(r.triangles_added > r.triangles_removed, "{r:?}");
            // It is imprinted all the same, in one undo step.
            assert!(r.seam_edges > 40, "{r:?}");
            assert_ne!(triangles(scene.mesh(id).unwrap()), before);
            assert!(scene.undo().unwrap());
            assert_eq!(triangles(scene.mesh(id).unwrap()), before);
            assert!(!scene.can_undo());
        }
    }
}

#[test]
fn an_open_curve_reports_no_cut() {
    let (mut scene, id) = add(merged(&[sheet(8, 4.0, [0.0; 3])]));
    let mut options = closed(&square([0.6, 0.7], [3.1, 2.9], 0.0), true);
    options.closed = false;
    let r = trim(&mut scene, id, &options);
    assert!(r.imprint_triangles > 0 && r.cut_triangles == 0, "{r:?}");
    assert_eq!(r.removed_area, 0.0);
}

#[test]
fn a_cut_reports_the_triangles_that_lost_surface_and_the_area() {
    let points = square([0.7, 0.8], [3.1, 2.9], 0.0);
    let whole = 4.0 * 4.0;
    let hole = (3.1 - 0.7) * (2.9 - 0.8);
    for placement in [
        Placement::IDENTITY,
        Placement {
            scale: [2.0; 3],
            translation: [1.0, -2.0, 0.5],
            ..Placement::IDENTITY
        },
        Placement {
            scale: [1.0, 3.0, 0.5],
            rotation: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ..Placement::IDENTITY
        },
    ] {
        let world: Vec<_> = points.iter().map(|&p| placement.apply(p)).collect();
        let unit = placement.scale[0] * placement.scale[1];
        for (delete_inside, removed) in [(true, hole), (false, whole - hole)] {
            let (mut scene, id) = add_at(merged(&[sheet(16, 4.0, [0.0; 3])]), placement);
            let r = trim(&mut scene, id, &closed(&world, delete_inside));
            assert!(
                (r.removed_area - unit * removed).abs() < 1e-4 * unit,
                "{r:?}: want {}",
                unit * removed
            );
            // A triangle along the curve lost part of its surface or was
            // only split beside it; those inside the removed side went whole.
            let cells = removed / (4.0 * 4.0 / 512.0);
            assert!((r.cut_triangles as f64) > 0.9 * cells, "{r:?}");
            assert!((r.cut_triangles as f64) < 1.5 * cells, "{r:?}");
        }
    }
}

#[test]
fn a_loop_inside_one_triangle_cuts_that_one() {
    let points = [[0.4, 0.1, 0.0], [0.6, 0.1, 0.0], [0.6, 0.2, 0.0]];
    let (mut scene, id) = add(merged(&[sheet(1, 1.0, [0.0; 3])]));
    let r = trim(&mut scene, id, &closed(&points, true));
    assert_eq!((r.cut_triangles, r.imprint_triangles), (1, 0), "{r:?}");
    assert!((r.removed_area - 0.01).abs() < 1e-6, "{r:?}");
}

/// The positions of the triangles `of` picks, sorted.
fn soup(mesh: &Mesh, of: impl Fn(TriangleId) -> bool) -> Vec<[[u32; 3]; 3]> {
    let mut out: Vec<_> = triangles(mesh)
        .into_iter()
        .filter(|&(t, _)| of(t))
        .map(|(_, corners)| corners.map(|(_, p)| p.map(f32::to_bits)))
        .collect();
    out.sort_unstable();
    out
}

/// Both sides of a loop on one component of a mesh: the others stay exactly
/// as they are, and it is cut as it is without them.
#[test]
fn only_the_component_the_curve_lies_on_is_cut() {
    let big = || ball([0.0; 3], 1.0);
    let small = || ball([3.0, 0.0, 0.0], 0.3);
    let flat = || sheet(16, 4.0, [6.0, 0.0, 0.0]);
    let on_big = around_pole([0.0; 3], 1.0, 0.7);
    let on_small = around_pole([3.0, 0.0, 0.0], 0.3, 0.7);
    let on_flat = square([6.7, 0.8], [9.1, 2.9], 0.0);
    // The parts, which one the loop is on, and the loop. The "inside" of all
    // the components at once would be another of them in each.
    let cases = [
        (
            "two balls, on the big one",
            vec![big(), small()],
            0,
            &on_big,
        ),
        (
            "two balls, on the small one",
            vec![big(), small()],
            1,
            &on_small,
        ),
        (
            "a ball and a sheet, on the sheet",
            vec![small(), flat()],
            1,
            &on_flat,
        ),
        (
            "a ball and a sheet, on the ball",
            vec![big(), flat()],
            0,
            &on_big,
        ),
        (
            "three parts, on the middle one",
            vec![big(), flat(), small()],
            1,
            &on_flat,
        ),
    ];
    for (name, parts, on, points) in cases {
        let first: usize = parts[..on].iter().map(|p| p.1.len() / 3).sum();
        let own = first..first + parts[on].1.len() / 3;
        let mesh = merged(&parts);
        let before = triangles(&mesh);
        let component = area_of(&mesh, [1.0; 3], |t| own.contains(&(t as usize)));
        let mut sides = Vec::new();
        for delete_inside in [true, false] {
            let (mut scene, id) = add(merged(&parts));
            let r = trim(&mut scene, id, &closed(points, delete_inside));
            sides.push(r.removed_area);
            let after = triangles(scene.mesh(id).unwrap());
            for (t, corners) in &before {
                if !own.contains(&(*t as usize)) {
                    assert_eq!(after.get(t), Some(corners), "{name}: triangle {t}");
                }
            }
            let (mut alone, part) = add(merged(std::slice::from_ref(&parts[on])));
            let same = trim(&mut alone, part, &closed(points, delete_inside));
            assert_eq!(r, same, "{name}");
            let is_own = |t| !before.contains_key(&t) || own.contains(&(t as usize));
            assert_eq!(
                soup(scene.mesh(id).unwrap(), is_own),
                soup(alone.mesh(part).unwrap(), |_| true),
                "{name}"
            );
            assert!(scene.undo().unwrap());
            assert_eq!(triangles(scene.mesh(id).unwrap()), before, "{name}");
            assert!(!scene.can_undo());
        }
        let total = sides[0] + sides[1];
        assert!(
            (total - component).abs() < 1e-4 * component,
            "{name}: {sides:?} of {component}"
        );
        assert!(0.05 * component < sides[0], "{name}: {sides:?}");
        // The smaller side of a closed ball, the one off the sheet's border.
        assert!(sides[0] < sides[1], "{name}: {sides:?}");
    }
}

#[test]
fn the_inside_is_the_side_away_from_the_components_own_border() {
    // Most of the sheet: more than the outside, and than the ball, whose
    // triangles come first and would have been "inside" too.
    let parts = [ball([0.0; 3], 0.5), sheet(16, 4.0, [6.0, 0.0, 0.0])];
    let (mut scene, id) = add(merged(&parts));
    let points = square([6.2, 0.2], [9.8, 3.8], 0.0);
    let r = trim(&mut scene, id, &closed(&points, true));
    assert!((r.removed_area - 3.6 * 3.6).abs() < 1e-3, "{r:?}");
}

#[test]
fn the_smaller_side_is_the_smaller_one_in_the_world() {
    // A bent bar: an arm along x from x = -0.5 to 1, and one from there up z,
    // cut where they meet. The first has less surface (7 to 9.2), unless x is
    // stretched.
    let p: [[f32; 3]; 12] = [
        [-0.5, 0.0, 0.0],
        [-0.5, 1.0, 0.0],
        [-0.5, 1.0, 1.0],
        [-0.5, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [2.0, 0.0, 3.0],
        [2.0, 1.0, 3.0],
        [1.0, 1.0, 3.0],
        [1.0, 0.0, 3.0],
    ];
    let mut ix = vec![0u32, 3, 2, 0, 2, 1];
    for ring in [0u32, 4] {
        for k in 0..4 {
            let (a, b) = (ring + k, ring + (k + 1) % 4);
            ix.extend([a, b, b + 4, a, b + 4, a + 4]);
        }
    }
    ix.extend([8, 9, 10, 8, 10, 11]);
    let arm = |scale: [f64; 3], up: bool| {
        let mesh = merged(&[(p.concat(), ix.clone())]);
        area_of(&mesh, scale, |t| (t >= 10) == up)
    };
    for (scale, up) in [([1.0; 3], false), ([20.0, 1.0, 1.0], true)] {
        assert_eq!(arm(scale, true) < arm(scale, false), up);
        let placement = Placement {
            scale,
            ..Placement::IDENTITY
        };
        let (mut scene, id) = add_at(merged(&[(p.concat(), ix.clone())]), placement);
        let joint: Vec<_> = p[4..8]
            .iter()
            .map(|q| placement.apply(q.map(f64::from)))
            .collect();
        let r = trim(&mut scene, id, &closed(&joint, true));
        let want = arm(scale, up);
        assert!((r.removed_area - want).abs() < 1e-6 * want, "{r:?}: {want}");
    }
}

#[test]
fn a_curve_on_several_components_is_refused() {
    let parts = [sheet(4, 2.0, [0.0; 3]), sheet(4, 2.0, [3.0, 0.0, 0.0])];
    for closed in [true, false] {
        for delete_inside in [true, false] {
            let (mut scene, id) = add(merged(&parts));
            let before = triangles(scene.mesh(id).unwrap());
            scene.take_changes();
            let options = TrimCurveOptions {
                points: square([0.5, 0.5], [4.5, 1.5], 0.0),
                closed,
                delete_inside,
                snap_tolerance: 0.0,
            };
            assert_eq!(
                scene.trim_by_curve(id, &options),
                Err(SceneError::InvalidInput(
                    "trim curve crosses several components of the mesh"
                ))
            );
            assert_eq!(triangles(scene.mesh(id).unwrap()), before);
            assert!(!scene.can_undo());
            assert!(scene.take_changes().is_empty());
        }
    }
}

#[test]
fn components_that_only_share_a_vertex_are_separate() {
    // Two sheets touching at one corner, (2, 2).
    let (mut p, mut ix) = sheet(4, 2.0, [0.0; 3]);
    let (p2, ix2) = sheet(4, 2.0, [2.0, 2.0, 0.0]);
    let at = (p.len() / 3) as u32;
    p.extend(p2);
    // The second sheet's corner 0 is the first's last vertex.
    ix.extend(ix2.iter().map(|&v| if v == 0 { at - 1 } else { v + at }));
    let mesh = || Mesh::from_buffers(&p, &ix).unwrap();
    let before = triangles(&mesh());
    for delete_inside in [true, false] {
        let (mut scene, id) = add(mesh());
        // One waypoint on the corner both use, nearest to both.
        let points = [[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [2.0, 2.0, 0.0]];
        let r = trim(&mut scene, id, &closed(&points, delete_inside));
        assert!(r.cut_triangles > 0 && r.removed_area < 4.0, "{r:?}");
        let after = triangles(scene.mesh(id).unwrap());
        for t in 32..64 {
            assert_eq!(after.get(&t), before.get(&t), "triangle {t}");
        }
    }
    let (mut scene, id) = add(mesh());
    let points = [[0.5, 0.5, 0.0], [2.5, 2.5, 0.0], [0.5, 1.5, 0.0]];
    assert!(scene.trim_by_curve(id, &closed(&points, true)).is_err());
}

#[test]
fn a_curve_where_components_touch_is_on_the_first_that_has_all_of_it() {
    // A sheet touched from above, at waypoints of the loop, by tetrahedra
    // whose triangles come first: equally near, and first by id.
    let spike = |x: f32, y: f32| -> Buffers {
        (
            vec![
                x,
                y,
                0.0,
                x - 0.1,
                y,
                0.5,
                x + 0.1,
                y - 0.1,
                0.5,
                x,
                y + 0.1,
                0.5,
            ],
            vec![0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2],
        )
    };
    let flat = || sheet(16, 4.0, [0.0; 3]);
    let points = square([0.7, 0.8], [3.1, 2.9], 0.0);
    let hole = (3.1 - 0.7) * (2.9 - 0.8);
    for spikes in [
        vec![spike(0.7, 0.8)],
        vec![spike(3.1, 0.8)],
        vec![
            spike(0.7, 0.8),
            spike(3.1, 0.8),
            spike(3.1, 2.9),
            spike(0.7, 2.9),
        ],
    ] {
        let spiked = spikes.len();
        let mut parts = spikes;
        parts.push(flat());
        let before = triangles(&merged(&parts));
        let (mut scene, id) = add(merged(&parts));
        let r = trim(&mut scene, id, &closed(&points, true));
        assert!((r.removed_area - hole).abs() < 1e-4, "{r:?}");
        let after = triangles(scene.mesh(id).unwrap());
        for t in 0..4 * spiked as TriangleId {
            assert_eq!(after.get(&t), before.get(&t), "triangle {t}");
        }
    }
    // Two sheets in one place: the first, and only that one.
    let (mut scene, id) = add(merged(&[flat(), flat()]));
    let r = trim(&mut scene, id, &closed(&points, true));
    assert!((r.removed_area - hole).abs() < 1e-4, "{r:?}");
    let after = triangles(scene.mesh(id).unwrap());
    let before = triangles(&merged(&[flat(), flat()]));
    for t in 512..1024 {
        assert_eq!(after.get(&t), before.get(&t), "triangle {t}");
    }
}

#[test]
fn triangles_with_a_repeated_vertex_belong_to_no_component() {
    let (p, mut ix) = sheet(4, 2.0, [0.0; 3]);
    ix.splice(0..0, [6, 6, 7]);
    let (mut scene, id) = add(Mesh::from_buffers(&p, &ix).unwrap());
    // Vertex 6 is on the loop and vertex 7 inside it.
    let points = square([0.5, 0.5], [1.7, 1.7], 0.0);
    let r = trim(&mut scene, id, &closed(&points, true));
    assert!((r.removed_area - 1.2 * 1.2).abs() < 1e-5, "{r:?}");
    assert_eq!(scene.mesh(id).unwrap().triangle(0), Some([6, 6, 7]));
}

/// A closed tube around y of `n x rows` quads with fan ends, 1 high and of
/// radius 1, whose texture wraps once around: a UV seam runs up it and around
/// both ends. With `uvs` off the same triangles share their vertices.
fn tube(n: u32, rows: u32, uvs: bool) -> Mesh {
    let (mut p, mut uv, mut ix) = (Vec::new(), Vec::new(), Vec::new());
    let vertex = |p: &mut Vec<f32>, uv: &mut Vec<f32>, at: [f32; 3], t: [f32; 2]| -> u32 {
        // Off, a vertex is shared by position.
        if !uvs {
            let same = |q: &[f32]| q == at;
            if let Some(k) = p.chunks(3).position(same) {
                return k as u32;
            }
        }
        p.extend(at);
        uv.extend(t);
        (p.len() / 3 - 1) as u32
    };
    let rim = |j: u32| {
        // The last column is at exactly the first one's positions.
        let (s, c) = (std::f32::consts::TAU * (j % n) as f32 / n as f32).sin_cos();
        [c, s]
    };
    let mut wall = Vec::new();
    for i in 0..=rows {
        for j in 0..=n {
            let [c, s] = rim(j);
            let h = i as f32 / rows as f32;
            wall.push(vertex(&mut p, &mut uv, [c, h, s], [j as f32 / n as f32, h]));
        }
    }
    let at = |i: u32, j: u32| wall[(i * (n + 1) + j) as usize];
    for i in 0..rows {
        for j in 0..n {
            let (a, b, c, d) = (at(i, j), at(i, j + 1), at(i + 1, j), at(i + 1, j + 1));
            ix.extend([a, c, b, b, c, d]);
        }
    }
    for (h, top) in [(0.0, false), (1.0, true)] {
        let mid = vertex(&mut p, &mut uv, [0.0, h, 0.0], [4.0, h]);
        let ring: Vec<u32> = (0..n)
            .map(|j| {
                let [c, s] = rim(j);
                vertex(&mut p, &mut uv, [c, h, s], [4.0 + c, h + s])
            })
            .collect();
        for j in 0..n as usize {
            let (a, b) = (ring[j], ring[(j + 1) % n as usize]);
            ix.extend(if top { [mid, b, a] } else { [mid, a, b] });
        }
    }
    let mut mesh = Mesh::from_buffers(&p, &ix).unwrap();
    if uvs {
        mesh.add_uvs(&uv).unwrap();
    }
    mesh
}

/// The open edges of the mesh by exact position: each with one triangle.
fn open_edges(mesh: &Mesh) -> usize {
    let mut uses: HashMap<[[u32; 3]; 2], usize> = HashMap::new();
    for t in 0..mesh.triangle_slots() as TriangleId {
        if let Some(corners) = mesh.triangle(t) {
            let p = corners.map(|v| mesh.position(v).unwrap().map(|x| (x + 0.0).to_bits()));
            for k in 0..3 {
                let mut edge = [p[k], p[(k + 1) % 3]];
                edge.sort_unstable();
                *uses.entry(edge).or_default() += 1;
            }
        }
    }
    assert!(uses.values().all(|&n| n <= 2));
    uses.values().filter(|&&n| n == 1).count()
}

#[test]
fn a_trim_crosses_uv_seams_and_splits_both_sides() {
    // Around the seam that runs up the tube, over the one around its end and
    // back, and around the tube once.
    let turn = |t: f64, h: f64| [t.cos(), h, t.sin()];
    let patch: Vec<_> = [
        turn(0.4, 0.3),
        turn(0.15, 0.25),
        turn(-0.15, 0.3),
        turn(-0.45, 0.35),
        turn(-0.4, 0.7),
        turn(0.02, 0.75),
        turn(0.45, 0.7),
    ]
    .into();
    let over_the_end: Vec<_> = [
        turn(0.4, 0.6),
        turn(-0.35, 0.65),
        turn(-0.35, 1.0),
        [0.4, 1.0, -0.1],
        [0.45, 1.0, 0.12],
        turn(0.4, 1.0),
    ]
    .into();
    let around: Vec<_> = (0..24)
        .map(|k| turn(f64::from(k) * std::f64::consts::TAU / 24.0 + 0.07, 0.4))
        .collect();
    for (name, points) in [("patch", patch), ("end", over_the_end), ("around", around)] {
        for delete_inside in [true, false] {
            let options = closed(&points, delete_inside);
            let (mut plain, plain_id) = add(tube(24, 6, false));
            let want = trim(&mut plain, plain_id, &options);
            let (mut scene, id) = add(tube(24, 6, true));
            assert_eq!(open_edges(scene.mesh(id).unwrap()), 0);
            let before = triangles(scene.mesh(id).unwrap());
            let r = trim(&mut scene, id, &options);
            // The same surface goes as from the mesh without seams.
            assert!(r.cut_triangles > 0, "{name}: {r:?}");
            assert!(
                (r.removed_area - want.removed_area).abs() < 1e-4,
                "{name}: {r:?}, {want:?}"
            );
            assert_eq!(r.triangles_removed, want.triangles_removed, "{name}");
            // Closed by position but for the curve: no seam came open.
            let mesh = scene.mesh(id).unwrap();
            assert_eq!(open_edges(mesh), r.seam_edges, "{name}: {r:?}");
            assert_eq!(open_edges(plain.mesh(plain_id).unwrap()), want.seam_edges);
            // New vertices take the UVs of the island they are on: on the
            // wall `u` is the angle's turn, 0 and 1 on the seam's two sides.
            let uvs = mesh.uvs().unwrap();
            let mut on_seam = [false; 2];
            for v in 0..mesh.vertex_slots() as u32 {
                let (Some([x, y, z]), [u, t]) = (
                    mesh.position(v),
                    [uvs[2 * v as usize], uvs[2 * v as usize + 1]],
                ) else {
                    continue;
                };
                if u > 1.0 {
                    continue; // on an end
                }
                assert!((t - y).abs() < 1e-5, "{name}: vertex {v}");
                let angle = (z.atan2(x) / std::f32::consts::TAU).rem_euclid(1.0);
                let off = (angle - u).abs();
                assert!(
                    off.min(1.0 - off) < 2e-3,
                    "{name}: vertex {v}: {u}, {angle}"
                );
                let new = y > 0.01 && (y * 6.0 - (y * 6.0).round()).abs() > 1e-4;
                if new && z == 0.0 && x > 0.0 {
                    on_seam[usize::from(u > 0.5)] = true;
                }
            }
            if name == "patch" {
                assert_eq!(on_seam, [true; 2], "{name}");
            }
            assert!(scene.undo().unwrap());
            assert_eq!(triangles(scene.mesh(id).unwrap()), before);
        }
    }
}

#[test]
fn the_uv_islands_of_one_surface_are_one_component() {
    let points = [[0.1, 1.0, 0.1], [0.5, 1.0, 0.1], [0.3, 1.0, 0.5]];
    // On the end, off its seam: the rest of the tube is the outside too.
    let (mut scene, id) = add(tube(24, 6, true));
    let whole = area(&scene, id);
    let inside = trim(&mut scene, id, &closed(&points, true)).removed_area;
    assert!((inside - 0.08).abs() < 1e-4, "{inside}");
    scene.undo().unwrap();
    let outside = trim(&mut scene, id, &closed(&points, false)).removed_area;
    assert!((inside + outside - whole).abs() < 1e-4);
    // Without UVs, split vertices are a border: the end is a component.
    let mut split = tube(24, 6, true);
    split.remove_channel("uv").unwrap();
    let (mut scene, id) = add(split);
    let end = trim(&mut scene, id, &closed(&points, false)).removed_area;
    assert!((inside + end - std::f64::consts::PI).abs() < 0.05, "{end}");
}

#[test]
fn a_sliver_along_a_uv_seam_splits_with_it() {
    // Two quads side by side, split along x = 1 with a triangle of no area
    // in the seam: two of its corners are at (1, 0).
    let p = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, // left
        1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 0.0, // right
    ];
    let ix = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 1, 4, 2];
    let mut mesh = Mesh::from_buffers(&p, &ix).unwrap();
    mesh.add_uvs(&[0.0; 16]).unwrap();
    let (mut scene, id) = add(mesh);
    let points = square([0.5, 0.4], [1.5, 0.8], 0.0);
    let r = trim(&mut scene, id, &closed(&points, true));
    assert!((r.removed_area - 0.4).abs() < 1e-5, "{r:?}");
    let (mut scene, id) = {
        let mut mesh = Mesh::from_buffers(&p, &ix).unwrap();
        mesh.add_uvs(&[0.0; 16]).unwrap();
        add(mesh)
    };
    let r = trim(&mut scene, id, &closed(&points, false));
    assert!((r.removed_area - 1.6).abs() < 1e-5, "{r:?}");
}

#[test]
fn trims_do_not_depend_on_the_parallel_feature() {
    let parts = [
        ball([0.0; 3], 1.0),
        torus(),
        sheet(60, 4.0, [6.0, 0.0, 0.0]),
    ];
    let run = || {
        let mut out = Vec::new();
        for scale in [[1.0; 3], [1.0, 2.0, 0.5]] {
            let placement = Placement {
                scale,
                ..Placement::IDENTITY
            };
            let loops = [
                square([6.7, 0.8], [9.1, 2.9], 0.0),
                around_pole([0.0; 3], 1.0, 0.7),
            ];
            for points in loops {
                let (mut scene, id) = add_at(merged(&parts), placement);
                let world: Vec<_> = points.iter().map(|&p| placement.apply(p)).collect();
                let r = trim(&mut scene, id, &closed(&world, true));
                let mesh = scene.mesh(id).unwrap();
                let bits: Vec<_> = mesh
                    .positions()
                    .iter()
                    .flatten()
                    .map(|x| x.to_bits())
                    .collect();
                out.push((r, r.removed_area.to_bits(), mesh.triangles().to_vec(), bits));
            }
        }
        out
    };
    par::set_force_sequential(true);
    let sequential = run();
    par::set_force_sequential(false);
    assert_eq!(run(), sequential);
}
