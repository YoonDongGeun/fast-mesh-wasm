//! Wrap properties (#263), checked on every result: closed (no boundary
//! loop), 2-manifold (no non-manifold edge, no pinch vertex), consistently
//! and outward wound, free of degenerate and duplicate triangles, of
//! coincident vertices and of self-intersections (any two triangles that
//! share no vertex and intersect or touch); every result vertex within the
//! closing radius (half the gap, plus a voxel) of the input, so within the
//! gap; and every input vertex inside the result or within two voxels of it.
//!
//! The inputs are what a wrap is for: a sphere with holes, touching and
//! overlapping parts, parts a small gap apart, a solid with a cavity, and
//! broken input (flipped, duplicated and crossing triangles).

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

/// What a checked wrap came out as.
struct Checked {
    components: usize,
    volume: f64,
    max_deviation: f64,
    mean_deviation: f64,
}

/// Wraps `input` and checks every property; `tolerance` is how far past the
/// closing radius a result vertex may be, in voxels.
fn wrap_checked(input: &AppendSource, options: &WrapOptions, what: &str) -> (WrapReport, Checked) {
    let (out, report) = wrap(input, options).unwrap();
    assert_eq!(report.vertices, out.positions.len(), "{what}");
    assert_eq!(report.triangles, out.triangles.len(), "{what}");
    let h = f64::from(report.plan.voxel_size);
    let r = f64::from(report.plan.radius);

    let mesh = mesh_of(&out);
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
    let mut volume = 0.0;
    for t in &out.triangles {
        let [p, q, s] = t.map(|v| out.positions[v as usize].map(f64::from));
        volume += (p[0] * (q[1] * s[2] - q[2] * s[1]) - p[1] * (q[0] * s[2] - q[2] * s[0])
            + p[2] * (q[0] * s[1] - q[1] * s[0]))
            / 6.0;
    }
    assert!(volume > 0.0, "{what}: wound inward");

    // Every result vertex is within the closing radius of the input.
    let input_mesh = mesh_of(input);
    let input_bvh = bvh_of(&input_mesh);
    let (mut max_deviation, mut sum) = (0.0f64, 0.0f64);
    for &p in &out.positions {
        let hit = closest_point(&input_mesh, &input_bvh, p, &ClosestOptions::default())
            .unwrap()
            .unwrap();
        max_deviation = max_deviation.max(f64::from(hit.distance));
        sum += f64::from(hit.distance);
    }
    assert!(
        max_deviation <= r + 1.5 * h,
        "{what}: a vertex {max_deviation} from the input (radius {r}, voxel {h})"
    );
    let gap = f64::from(options.gap).max(2.0 * h);
    assert!(max_deviation <= gap, "{what}: past the gap");

    // Every input vertex is inside the result or within two voxels of it
    // (marching cubes cuts sharp convex corners by up to `sqrt(3)` voxels).
    let mut result = mesh_of(&out);
    let result_bvh = bvh_of(&result);
    let sign = SignData::build(&mut result, &result_bvh, SignMode::Auto).unwrap();
    let mut d = vec![0.0f32; input.positions.len()];
    signed_distances(
        &result,
        &result_bvh,
        &sign,
        &input.positions,
        f32::INFINITY,
        &mut d,
    )
    .unwrap();
    let used: std::collections::BTreeSet<u32> = input.triangles.iter().flatten().copied().collect();
    for &v in &used {
        assert!(
            f64::from(d[v as usize]) <= 2.0 * h,
            "{what}: input vertex {v} is {} outside",
            d[v as usize]
        );
    }

    let checked = Checked {
        components: a.components.len(),
        volume,
        max_deviation,
        mean_deviation: sum / out.positions.len() as f64,
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
    // Close to the unit ball's volume.
    let ball = 4.0 / 3.0 * std::f64::consts::PI;
    assert!((a.volume - ball).abs() < 0.05 * ball, "volume {}", a.volume);

    let (projected, b) = wrap_checked(&input, &options(0.4, 48, true), "holed sphere, projected");
    assert_eq!(b.components, 1);
    assert!(
        projected.projected > projected.vertices / 2,
        "{projected:?}"
    );
    assert!(
        b.mean_deviation < 0.8 * a.mean_deviation,
        "{} vs {}",
        b.mean_deviation,
        a.mean_deviation
    );
    assert!(b.max_deviation <= a.max_deviation + 1e-6);
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
        assert!((c.volume - 3.0).abs() < 0.15, "volume {}", c.volume);
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
    assert!(bridged.volume > 2.05, "volume {}", bridged.volume);
    let (_, apart) = wrap_checked(&input, &options(0.05, 64, false), "separate boxes");
    assert_eq!(apart.components, 2);
    assert!((apart.volume - 2.0).abs() < 0.1, "volume {}", apart.volume);
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
    assert!((c.volume - 1.0).abs() < 0.08, "volume {}", c.volume);

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
        .voxel_size(Some(0.05))
        .build();
    let plan = wrap_plan([0.0; 3], [2.0, 1.0, 1.0], &opts).unwrap();
    // 2 / 0.05 = 40 cells, plus the 0.25 radius and three voxels per side.
    assert_eq!(plan.dims, [40 + 2 * (5 + 3) + 1, 20 + 17, 20 + 17]);
    assert_eq!(plan.bytes, plan.voxels * WRAP_BYTES_PER_VOXEL);
    let (_, report) = wrap(&input, &opts).unwrap();
    assert_eq!(report.plan, plan);

    let limited = WrapOptions {
        max_voxels: plan.voxels - 1,
        ..opts
    };
    assert!(!wrap_plan([0.0; 3], [2.0, 1.0, 1.0], &limited).unwrap().fits);
    assert_eq!(
        wrap(&input, &limited).map(|_| ()),
        Err(OpsError::InvalidInput("wrap grid exceeds max_voxels"))
    );
}

#[test]
fn wrap_meshes_keeps_a_shared_origin_and_centres_others() {
    let a = Mesh::from_buffers_f64_with_origin(
        &cube([1.0; 3], [0.0; 3])
            .positions
            .iter()
            .flat_map(|p| p.map(f64::from))
            .collect::<Vec<_>>(),
        cube([1.0; 3], [0.0; 3]).triangles.as_flattened(),
        [10.0, 0.0, 0.0],
    )
    .unwrap();
    let opts = options(0.1, 32, false);
    let (one, report) = wrap_meshes(&[&a], &opts).unwrap();
    assert_eq!(one.origin(), [10.0, 0.0, 0.0]);
    assert_eq!(one.triangle_count(), report.triangles);
    let b = mesh_of(&cube([1.0; 3], [1.0, 0.0, 0.0]));
    let (two, _) = wrap_meshes(&[&a, &b], &opts).unwrap();
    // [0, 2] along x in the shared frame.
    assert_eq!(two.origin()[0], 1.0);
    assert_eq!(
        wrap_meshes(&[], &opts).map(|_| ()),
        Err(OpsError::InvalidInput("wrap needs at least one mesh"))
    );
}
