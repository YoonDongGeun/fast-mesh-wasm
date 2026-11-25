//! Wrap properties (#263), checked on every result: closed (no boundary
//! loop), 2-manifold (no non-manifold edge, no pinch vertex), consistently
//! and outward wound, free of degenerate and duplicate triangles, of
//! coincident vertices and of self-intersections (with the engine for
//! triangles that share no vertex, and with exact predicates for those that
//! do); every result vertex within the closing radius (half the gap, plus
//! half a voxel) of the input, so within the gap; and every input vertex
//! inside the result or within a voxel's diagonal of it.
//!
//! The inputs are what a wrap is for: a sphere with holes, touching and
//! overlapping parts, parts a small gap apart, a solid with a cavity, broken
//! input (flipped, duplicated and crossing triangles) and loose triangle
//! soups.

use std::collections::BTreeSet;

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_fields::{SignData, SignMode, signed_distances};
use fast_mesh_ops::append::AppendSource;
use fast_mesh_ops::primitives::{
    BoxOptions, IcosphereOptions, TorusOptions, cuboid, icosphere, torus,
};
use fast_mesh_ops::wrap::{WRAP_BYTES_PER_VOXEL, WrapOptions, WrapReport, wrap, wrap_plan};
use fast_mesh_ops::{AnalyzeOptions, OpsError, analyze, wrap_meshes};
use fast_mesh_query::{ClosestOptions, closest_point};
use fast_mesh_store::Mesh;
use robust::{Coord, Coord3D, orient2d, orient3d};

fn moved(src: AppendSource, by: [f64; 3]) -> AppendSource {
    AppendSource {
        positions: src
            .positions
            .iter()
            .map(|p| std::array::from_fn(|k| (f64::from(p[k]) + by[k]) as f32))
            .collect(),
        triangles: src.triangles,
    }
}

fn merged(parts: &[AppendSource]) -> AppendSource {
    let mut out = AppendSource::default();
    for p in parts {
        let base = out.positions.len() as u32;
        out.positions.extend_from_slice(&p.positions);
        out.triangles
            .extend(p.triangles.iter().map(|t| t.map(|v| v + base)));
    }
    out
}

fn sphere(radius: f64, subdivisions: u32) -> AppendSource {
    icosphere(&IcosphereOptions {
        radius,
        subdivisions,
    })
    .unwrap()
}

fn cube(size: [f64; 3], at: [f64; 3]) -> AppendSource {
    let src = cuboid(&BoxOptions {
        size,
        segments: [4; 3],
    })
    .unwrap();
    // Centred on the origin: move its min corner to `at`.
    moved(src, std::array::from_fn(|k| at[k] + 0.5 * size[k]))
}

/// `src` turned by `angle` about the axis (1, 2, 3).
fn turned(src: AppendSource, angle: f64) -> AppendSource {
    let l = 14f64.sqrt();
    let u = [1.0 / l, 2.0 / l, 3.0 / l];
    let (s, c) = angle.sin_cos();
    AppendSource {
        positions: src
            .positions
            .iter()
            .map(|p| {
                let p = p.map(f64::from);
                let d = u[0] * p[0] + u[1] * p[1] + u[2] * p[2];
                let x = [
                    u[1] * p[2] - u[2] * p[1],
                    u[2] * p[0] - u[0] * p[2],
                    u[0] * p[1] - u[1] * p[0],
                ];
                std::array::from_fn(|k| (c * p[k] + s * x[k] + (1.0 - c) * d * u[k]) as f32)
            })
            .collect(),
        triangles: src.triangles,
    }
}

/// `src` without the triangles whose centroid is within `radius` of a centre.
fn holed(src: AppendSource, centers: &[[f64; 3]], radius: f64) -> AppendSource {
    let p = |v: u32| src.positions[v as usize].map(f64::from);
    let triangles = src
        .triangles
        .iter()
        .copied()
        .filter(|t| {
            let c: [f64; 3] =
                std::array::from_fn(|k| t.iter().map(|&v| p(v)[k]).sum::<f64>() / 3.0);
            centers
                .iter()
                .all(|o| (0..3).map(|k| (c[k] - o[k]).powi(2)).sum::<f64>() > radius * radius)
        })
        .collect();
    AppendSource {
        positions: src.positions,
        triangles,
    }
}

fn mesh_of(src: &AppendSource) -> Mesh {
    Mesh::from_buffers(src.positions.as_flattened(), src.triangles.as_flattened()).unwrap()
}

fn bvh_of(mesh: &Mesh) -> Bvh {
    Bvh::build(mesh, &BuildOptions::default()).unwrap()
}

fn c3(p: [f64; 3]) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

/// Exact: whether the closed segment `p q` meets the closed triangle
/// `(a, b, c)`, not lying in its plane.
fn crosses([p, q]: [[f64; 3]; 2], [a, b, c]: [[f64; 3]; 3]) -> bool {
    let o = |a, b, c, d| orient3d(c3(a), c3(b), c3(c), c3(d));
    let (s, t) = (o(a, b, c, p), o(a, b, c, q));
    if (s > 0.0 && t > 0.0) || (s < 0.0 && t < 0.0) || (s == 0.0 && t == 0.0) {
        return false;
    }
    let w = [o(p, q, a, b), o(p, q, b, c), o(p, q, c, a)];
    w.iter().all(|&x| x >= 0.0) || w.iter().all(|&x| x <= 0.0)
}

/// Exact: whether two coplanar triangles overlap in more than their borders.
fn overlap_in_plane(t: [[f64; 3]; 3], u: [[f64; 3]; 3]) -> bool {
    // Drop an axis along which neither is degenerate.
    for (x, y) in [(0, 1), (1, 2), (2, 0)] {
        let flat = |t: [[f64; 3]; 3]| {
            let f = t.map(|p| Coord { x: p[x], y: p[y] });
            let o = orient2d(f[0], f[1], f[2]);
            (if o < 0.0 { [f[0], f[2], f[1]] } else { f }, o != 0.0)
        };
        let ((t, t_ok), (u, u_ok)) = (flat(t), flat(u));
        if t_ok && u_ok {
            let apart = |t: [Coord<f64>; 3], u: [Coord<f64>; 3]| {
                (0..3).any(|i| u.iter().all(|&q| orient2d(t[i], t[(i + 1) % 3], q) <= 0.0))
            };
            return !apart(t, u) && !apart(u, t);
        }
    }
    false
}

/// Exact: the pairs of triangles that share a vertex and also meet elsewhere
/// (beyond the shared vertex or edge), which the engine's self-intersection
/// queries leave out.
fn crossing_neighbours(out: &AppendSource) -> usize {
    let p = |v: u32| out.positions[v as usize].map(f64::from);
    let mut fans = vec![Vec::new(); out.positions.len()];
    for (t, tri) in out.triangles.iter().enumerate() {
        for &v in tri {
            fans[v as usize].push(t);
        }
    }
    let mut found = 0;
    for (v, fan) in fans.iter().enumerate() {
        let v = v as u32;
        for (i, &a) in fan.iter().enumerate() {
            for &b in &fan[i + 1..] {
                let (ta, tb) = (out.triangles[a], out.triangles[b]);
                let shared: Vec<u32> = ta.into_iter().filter(|x| tb.contains(x)).collect();
                // A pair on one edge comes up at both its ends; keep one.
                if shared.iter().any(|&x| x < v) {
                    continue;
                }
                let rest = |t: [u32; 3]| -> Vec<[f64; 3]> {
                    t.into_iter()
                        .filter(|x| !shared.contains(x))
                        .map(p)
                        .collect()
                };
                let (pa, pb) = (ta.map(p), tb.map(p));
                let (ra, rb) = (rest(ta), rest(tb));
                let coplanar = rb
                    .iter()
                    .all(|&x| orient3d(c3(pa[0]), c3(pa[1]), c3(pa[2]), c3(x)) == 0.0);
                let bad = if coplanar {
                    overlap_in_plane(pa, pb)
                } else if shared.len() == 1 {
                    crosses([ra[0], ra[1]], pb) || crosses([rb[0], rb[1]], pa)
                } else {
                    false
                };
                found += usize::from(bad);
            }
        }
    }
    found
}

/// Checks that `out` is closed, manifold, consistently and outward wound and
/// free of self-intersections; returns its component count and volume.
fn check_surface(out: &AppendSource, what: &str) -> (usize, f64) {
    let mesh = mesh_of(out);
    let bvh = bvh_of(&mesh);
    let a = analyze(
        &mesh,
        &AnalyzeOptions {
            self_intersection: Some(&bvh),
            ..AnalyzeOptions::default()
        },
    )
    .unwrap();
    assert!(
        a.loops.is_empty(),
        "{what}: {} boundary loops",
        a.loops.len()
    );
    assert_eq!(a.non_manifold_edges, 0, "{what}: non-manifold edges");
    assert!(a.pinch_vertices.is_empty(), "{what}: pinch vertices");
    assert_eq!(a.orientation_conflicts, 0, "{what}: orientation conflicts");
    assert_eq!(a.degenerate, 0, "{what}: degenerate triangles");
    assert_eq!(a.duplicate, 0, "{what}: duplicate triangles");
    assert_eq!(a.coincident_groups, 0, "{what}: coincident vertices");
    assert_eq!(a.self_intersects, Some(false), "{what}: self-intersects");
    assert_eq!(
        crossing_neighbours(out),
        0,
        "{what}: neighbours that cross each other"
    );
    let mut volume = 0.0;
    for t in &out.triangles {
        let [p, q, s] = t.map(|v| out.positions[v as usize].map(f64::from));
        volume += (p[0] * (q[1] * s[2] - q[2] * s[1]) - p[1] * (q[0] * s[2] - q[2] * s[0])
            + p[2] * (q[0] * s[1] - q[1] * s[0]))
            / 6.0;
    }
    assert!(volume > 0.0, "{what}: wound inward");
    (a.components.len(), volume)
}

/// Checks that every vertex of `out` is within the closing radius (plus a
/// voxel) of `input`, and so within the gap (or two voxels); returns the
/// largest and the mean distance.
fn check_distance(
    input: &AppendSource,
    out: &AppendSource,
    report: &WrapReport,
    gap: f32,
    what: &str,
) -> (f64, f64) {
    let h = f64::from(report.plan.voxel_size);
    let r = f64::from(report.plan.radius);
    let mesh = mesh_of(input);
    let bvh = bvh_of(&mesh);
    let (mut max, mut sum) = (0.0f64, 0.0f64);
    for &p in &out.positions {
        let hit = closest_point(&mesh, &bvh, p, &ClosestOptions::default())
            .unwrap()
            .unwrap();
        max = max.max(f64::from(hit.distance));
        sum += f64::from(hit.distance);
    }
    assert!(
        max <= r + h && max <= f64::from(gap).max(2.0 * h),
        "{what}: a vertex {max} from the input (radius {r}, voxel {h}, gap {gap})"
    );
    (max, sum / out.positions.len() as f64)
}

/// What a checked wrap came out as.
struct Checked {
    components: usize,
    volume: f64,
    max_deviation: f64,
    mean_deviation: f64,
}

/// Wraps `input` and checks every property.
fn wrap_checked(input: &AppendSource, options: &WrapOptions, what: &str) -> (WrapReport, Checked) {
    let (out, report) = wrap(input, options).unwrap();
    assert_eq!(report.vertices, out.positions.len(), "{what}");
    assert_eq!(report.triangles, out.triangles.len(), "{what}");
    let (components, volume) = check_surface(&out, what);
    let (max_deviation, mean_deviation) = check_distance(input, &out, &report, options.gap, what);

    // Every input vertex is inside the result or within a voxel's diagonal
    // of it (marching cubes cuts sharp corners).
    let h = f64::from(report.plan.voxel_size);
    let mut result = mesh_of(&out);
    let bvh = bvh_of(&result);
    let sign = SignData::build(&mut result, &bvh, SignMode::Auto).unwrap();
    let mut d = vec![0.0f32; input.positions.len()];
    signed_distances(
        &result,
        &bvh,
        &sign,
        &input.positions,
        f32::INFINITY,
        &mut d,
    )
    .unwrap();
    let used: BTreeSet<u32> = input.triangles.iter().flatten().copied().collect();
    for &v in &used {
        assert!(
            f64::from(d[v as usize]) <= 3f64.sqrt() * h,
            "{what}: input vertex {v} is {} outside (voxel {h})",
            d[v as usize]
        );
    }
    let checked = Checked {
        components,
        volume,
        max_deviation,
        mean_deviation,
    };
    (report, checked)
}

fn options(gap: f32, detail: u32, project: bool) -> WrapOptions {
    WrapOptions::builder()
        .gap(gap)
        .detail(detail)
        .project(project)
        .build()
}

#[test]
fn a_sphere_with_holes_wraps_closed_within_the_gap() {
    // Three holes about 0.3 across in a unit sphere, closed by a 0.4 gap.
    let input = holed(
        sphere(1.0, 5),
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -0.6, 0.8]],
        0.15,
    );
    let (plain, a) = wrap_checked(&input, &options(0.4, 48, false), "holed sphere");
    assert_eq!(a.components, 1);
    assert_eq!(plain.projected, 0);
    // Filled: close to the unit ball's volume.
    let ball = 4.0 / 3.0 * std::f64::consts::PI;
    assert!((a.volume - ball).abs() < 0.01 * ball, "volume {}", a.volume);

    let (projected, b) = wrap_checked(&input, &options(0.4, 48, true), "projected");
    assert_eq!(b.components, 1);
    assert!(
        projected.projected * 10 > projected.vertices * 9,
        "{projected:?}"
    );
    assert!(
        b.mean_deviation < 0.8 * a.mean_deviation,
        "{} vs {}",
        b.mean_deviation,
        a.mean_deviation
    );
    assert!(b.max_deviation <= a.max_deviation + 1e-6);

    // Holes wider than the gap let the outside in: only the thin wall is
    // left, which encloses nothing.
    assert_eq!(
        wrap(&input, &options(0.2, 48, false)).map(|_| ()),
        Err(OpsError::InvalidInput(
            "wrap encloses no volume at this gap and voxel size"
        ))
    );
}

#[test]
fn touching_meshes_wrap_into_one() {
    // Two boxes sharing a face, and a third touching them along an edge.
    let input = merged(&[
        cube([1.0; 3], [0.0; 3]),
        cube([1.0; 3], [1.0, 0.0, 0.0]),
        cube([1.0; 3], [2.0, 1.0, 0.0]),
    ]);
    for project in [false, true] {
        let (_, c) = wrap_checked(&input, &options(0.2, 48, project), "touching boxes");
        assert_eq!(c.components, 1);
        assert!((c.volume - 3.0).abs() < 0.06, "volume {}", c.volume);
    }
    // The same at an angle to the grid.
    let input = turned(input, 0.7);
    for project in [false, true] {
        let (_, c) = wrap_checked(&input, &options(0.2, 48, project), "turned boxes");
        assert_eq!(c.components, 1);
        assert!((c.volume - 3.0).abs() < 0.06, "volume {}", c.volume);
    }

    // Two spheres touching at a point, and two overlapping.
    let touching = merged(&[sphere(0.5, 4), moved(sphere(0.5, 4), [1.0, 0.0, 0.0])]);
    let overlapping = merged(&[sphere(0.5, 4), moved(sphere(0.5, 4), [0.6, 0.0, 0.0])]);
    for (input, what) in [
        (touching, "touching spheres"),
        (overlapping, "overlapping spheres"),
    ] {
        for project in [false, true] {
            let (_, c) = wrap_checked(&input, &options(0.2, 48, project), what);
            assert_eq!(c.components, 1, "{what}");
        }
    }
}

#[test]
fn gaps_under_the_setting_close_and_wider_ones_stay_open() {
    // Two boxes 0.1 apart.
    let input = merged(&[cube([1.0; 3], [0.0; 3]), cube([1.0; 3], [1.1, 0.0, 0.0])]);
    let (_, bridged) = wrap_checked(&input, &options(0.3, 48, false), "bridged boxes");
    assert_eq!(bridged.components, 1);
    // Bridged: about the volume of the 2.1-long box.
    assert!(
        bridged.volume > 2.05 && bridged.volume < 2.12,
        "volume {}",
        bridged.volume
    );
    let (_, apart) = wrap_checked(&input, &options(0.06, 96, false), "separate boxes");
    assert_eq!(apart.components, 2);
    assert!((apart.volume - 2.0).abs() < 0.02, "volume {}", apart.volume);
}

#[test]
fn cavities_are_filled_and_broken_input_is_fine() {
    // A box with an inner cavity: the wrap is the outer envelope alone.
    let mut inner = cube([0.5; 3], [0.25; 3]);
    for t in &mut inner.triangles {
        t.swap(1, 2);
    }
    let hollow = merged(&[cube([1.0; 3], [0.0; 3]), inner]);
    let (_, c) = wrap_checked(&hollow, &options(0.1, 40, false), "box with a cavity");
    assert_eq!(c.components, 1);
    assert!((c.volume - 1.0).abs() < 0.02, "volume {}", c.volume);

    // Crossing tori, every other triangle flipped, some duplicated, and one
    // a non-manifold fin: still one clean envelope.
    let a = torus(&TorusOptions::default()).unwrap();
    let mut b = moved(a.clone(), [0.4, 0.0, 0.0]);
    for (i, t) in b.triangles.iter_mut().enumerate() {
        if i % 2 == 0 {
            t.swap(1, 2);
        }
    }
    let mut broken = merged(&[a, b]);
    let dup: Vec<[u32; 3]> = broken.triangles.iter().step_by(7).copied().collect();
    broken.triangles.extend(dup);
    let [x, y, _] = broken.triangles[0];
    let fin = broken.positions.len() as u32;
    // Inside the first tube, so the fin is inside the envelope.
    broken.positions.push([0.4, 0.0, 0.0]);
    broken.triangles.push([x, y, fin]);
    for project in [false, true] {
        let (_, c) = wrap_checked(&broken, &options(0.1, 48, project), "broken tori");
        assert_eq!(c.components, 1);
    }
}

#[test]
fn the_plan_comes_first_and_big_grids_are_refused() {
    let input = cube([2.0, 1.0, 1.0], [0.0; 3]);
    let opts = WrapOptions::builder()
        .gap(0.5)
        .voxel_size(Some(0.0625))
        .build();
    let plan = wrap_plan([0.0; 3], [2.0, 1.0, 1.0], &opts).unwrap();
    // 2 / 0.0625 = 32 cells, plus the 0.25 radius and three voxels per side.
    assert_eq!(plan.dims, [32 + 2 * (4 + 3) + 1, 16 + 15, 16 + 15]);
    assert_eq!(plan.bytes, plan.voxels * WRAP_BYTES_PER_VOXEL);
    let (_, report) = wrap(&input, &opts).unwrap();
    assert_eq!(report.plan, plan);

    let limited = WrapOptions {
        max_voxels: plan.voxels as usize - 1,
        ..opts
    };
    assert!(!wrap_plan([0.0; 3], [2.0, 1.0, 1.0], &limited).unwrap().fits);
    assert_eq!(
        wrap(&input, &limited).map(|_| ()),
        Err(OpsError::InvalidInput("wrap grid exceeds max_voxels"))
    );
}

#[test]
fn wrap_meshes_applies_origins_and_centres_the_result() {
    let part = cube([1.0; 3], [0.0; 3]);
    let a = Mesh::from_buffers_f64_with_origin(
        &part
            .positions
            .iter()
            .flat_map(|p| p.map(f64::from))
            .collect::<Vec<_>>(),
        part.triangles.as_flattened(),
        [10.0, 0.0, 0.0],
    )
    .unwrap();
    let opts = options(0.1, 32, false);
    let (one, report) = wrap_meshes(&[&a], &opts).unwrap();
    assert_eq!(one.origin(), [0.5; 3]);
    assert_eq!(one.triangle_count(), report.triangles);
    // The same shape as without the origin.
    let (out, _) = wrap(&moved(part, [-0.5; 3]), &opts).unwrap();
    assert_eq!(one.positions(), out.positions);
    let b = mesh_of(&cube([1.0; 3], [1.0, 0.0, 0.0]));
    let (two, _) = wrap_meshes(&[&a, &b], &opts).unwrap();
    assert_eq!(two.origin(), [1.0, 0.5, 0.5]);
    for meshes in [&[][..], &[&Mesh::new()][..]] {
        assert!(matches!(
            wrap_meshes(meshes, &opts),
            Err(OpsError::InvalidInput(_))
        ));
    }
}

#[test]
fn far_from_the_origin_the_voxels_must_stay_resolvable() {
    let near = cube([1.0; 3], [0.0; 3]);
    let (_, at_zero) = wrap_checked(&near, &options(0.1, 32, false), "near");
    // At 100 an `f32` still resolves a 1/32 voxel, and the wrap is as clean.
    let far = cube([1.0; 3], [100.0, -50.0, 0.0]);
    for project in [false, true] {
        let (_, c) = wrap_checked(&far, &options(0.1, 32, project), "far");
        assert_eq!(c.components, 1);
        assert!((c.volume - at_zero.volume).abs() < 0.02, "{}", c.volume);
    }
    let lost = cube([1.0; 3], [10_000.0, 0.0, 0.0]);
    assert_eq!(
        wrap(&lost, &options(0.1, 128, false)).map(|_| ()),
        Err(OpsError::InvalidInput(
            "wrap voxel size is too small for f32 positions this far from the origin"
        ))
    );
}

/// A deterministic linear congruential generator.
struct Lcg(u64);

impl Lcg {
    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn triangle_soups_wrap_clean() {
    // Loose, crossing triangles of every size: nothing about them is closed,
    // manifold or oriented. What they enclose at the gap still wraps clean.
    let mut wrapped = 0;
    for seed in 0..24 {
        let mut r = Lcg(seed);
        let mut soup = AppendSource::default();
        for _ in 0..60 {
            let size = 0.1 + 0.9 * r.unit();
            let at = [0, 1, 2].map(|_| r.unit());
            let base = soup.positions.len() as u32;
            for _ in 0..3 {
                soup.positions
                    .push(at.map(|c| (c + size * (r.unit() - 0.5)) as f32));
            }
            soup.triangles.push([base, base + 1, base + 2]);
        }
        let gap = (0.05 + 0.4 * r.unit()) as f32;
        for project in [false, true] {
            let what = format!("soup {seed}, projected {project}");
            match wrap(&soup, &options(gap, 24, project)) {
                // Sheets that stick out are not covered, so only the
                // surface itself is checked.
                Ok((out, report)) => {
                    check_surface(&out, &what);
                    check_distance(&soup, &out, &report, gap, &what);
                    wrapped += 1;
                }
                Err(e) => assert_eq!(
                    e,
                    OpsError::InvalidInput("wrap encloses no volume at this gap and voxel size")
                ),
            }
        }
    }
    assert!(wrapped >= 24, "{wrapped} soups wrapped");
}

#[test]
fn the_neighbour_check_sees_crossings() {
    let soup = |positions: &[[f32; 3]], triangles: &[[u32; 3]]| AppendSource {
        positions: positions.to_vec(),
        triangles: triangles.to_vec(),
    };
    // One shared corner: apart, and crossing.
    let mut p = [
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];
    assert_eq!(crossing_neighbours(&soup(&p, &[[0, 1, 2], [0, 3, 4]])), 1);
    p[3] = [-1.0, -1.0, 1.0];
    p[4] = [-1.0, -1.0, -1.0];
    assert_eq!(crossing_neighbours(&soup(&p, &[[0, 1, 2], [0, 3, 4]])), 0);
    // One shared edge: flat, and folded back onto the other.
    p[3] = [1.0, 1.0, 0.0];
    assert_eq!(crossing_neighbours(&soup(&p, &[[0, 1, 2], [2, 1, 3]])), 0);
    p[3] = [0.1, 0.1, 0.0];
    assert_eq!(crossing_neighbours(&soup(&p, &[[0, 1, 2], [2, 1, 3]])), 1);
}
