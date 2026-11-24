//! Booleans, cuts, hulls, wraps, hollows and placed meshes far from the world
//! origin (#278). Whatever mix of a mesh's `f64` origin and its placement puts
//! it there, the results keep the `f32` precision of small local coordinates:
//! world coordinates in `f32` would be off by 0.004 at 1e5.

use fast_mesh_ops::boolean::{
    DrainHoleOptions, DrainHoleSpec, HollowBackend, HollowOptions, PlaceMeshParams, PlaceMode,
};
use fast_mesh_ops::wrap::WrapOptions;
use fast_mesh_scene::{
    BooleanMethod, BooleanOp, BooleanOptions, CsgNode, CutOptions, MeshId, MeshOptions, Placement,
    Plane, Scene, SceneOptions,
};
use fast_mesh_store::Mesh;

type V3 = [f64; 3];

/// About eight `f32` steps at the sizes below (up to 2.2 units).
const TOL: f64 = 2e-6;

const FAR: [V3; 3] = [
    [1.0e5, -2.5e5, 7.3e4],
    [148_000.3, -77_000.7, 91_234.1],
    [6.0e6, -3.5e6, 1.25e6],
];

/// Box A is `[0, SIZE]`, box B the same moved by `SHIFT`.
const SIZE: V3 = [1.1, 0.9, 0.7];
const SHIFT: V3 = [0.4, 0.3, 0.2];

fn add(a: V3, b: V3) -> V3 {
    std::array::from_fn(|k| a[k] + b[k])
}

fn sub(a: V3, b: V3) -> V3 {
    std::array::from_fn(|k| a[k] - b[k])
}

/// How far `d` reaches along the unit vector `n`, and across it.
fn along_and_across(d: V3, n: V3) -> (f64, f64) {
    let along: f64 = (0..3).map(|k| d[k] * n[k]).sum();
    let across: f64 = (0..3).map(|k| (d[k] - along * n[k]).powi(2)).sum();
    (along, across.sqrt())
}

/// A rotation about z, then about x.
fn turn(z: f64, x: f64) -> [[f64; 3]; 3] {
    let ((sz, cz), (sx, cx)) = (z.sin_cos(), x.sin_cos());
    [
        [cz, -sz, 0.0],
        [cx * sz, cx * cz, -sx],
        [sx * sz, sx * cz, cx],
    ]
}

/// A closed box with its centre as its origin.
fn box_mesh(lo: V3, hi: V3) -> Mesh {
    let ([x0, y0, z0], [x1, y1, z1]) = (lo, hi);
    #[rustfmt::skip]
    let pos = [
        x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0,
        x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1,
    ];
    #[rustfmt::skip]
    let tris = [
        0, 2, 1, 0, 3, 2,
        4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4,
        1, 2, 6, 1, 6, 5,
        2, 3, 7, 2, 7, 6,
        3, 0, 4, 3, 4, 7,
    ];
    Mesh::from_buffers_f64(&pos, &tris).unwrap()
}

/// How a mesh modelled around zero gets to its place in the world: its
/// positions (and so its origin) moved by `geometry`, then a placement.
#[derive(Debug, Clone, Copy)]
struct Pose {
    geometry: V3,
    placement: Placement,
}

impl Pose {
    const NEAR: Pose = Pose {
        geometry: [0.0; 3],
        placement: Placement::IDENTITY,
    };

    fn to_world(self, p: V3) -> V3 {
        self.placement.apply(add(self.geometry, p))
    }

    fn to_model(self, w: V3) -> V3 {
        sub(self.placement.inverse_apply(w), self.geometry)
    }

    /// [`TOL`] in the model frame.
    fn tol(self) -> f64 {
        TOL / self.placement.scale.iter().fold(1.0, |m, s| s.min(m))
    }

    fn add_box(self, scene: &mut Scene, lo: V3, hi: V3) -> MeshId {
        let mesh = box_mesh(add(self.geometry, lo), add(self.geometry, hi));
        let options = MeshOptions {
            placement: self.placement,
            ..MeshOptions::default()
        };
        scene.add_mesh(mesh, options).unwrap()
    }

    fn add_a(self, scene: &mut Scene) -> MeshId {
        self.add_box(scene, [0.0; 3], SIZE)
    }

    fn add_b(self, scene: &mut Scene) -> MeshId {
        self.add_box(scene, SHIFT, add(SIZE, SHIFT))
    }
}

fn moved(translation: V3) -> Placement {
    Placement {
        translation,
        ..Placement::IDENTITY
    }
}

/// Rigid ways to get to `far` (and one to come back).
fn rigid_poses(far: V3) -> [Pose; 4] {
    [
        // An origin far out.
        Pose {
            geometry: far,
            placement: Placement::IDENTITY,
        },
        // A placement far out.
        Pose {
            geometry: [0.0; 3],
            placement: moved(far),
        },
        // Both, and a rotation.
        Pose {
            geometry: far,
            placement: Placement {
                rotation: turn(0.6, -0.35),
                ..moved([7.0, -42.5, 300.0])
            },
        },
        // A far origin that the placement takes back to the world origin.
        Pose {
            geometry: far,
            placement: moved(far.map(|c| -c)),
        },
    ]
}

fn scaled_pose(far: V3, scale: V3) -> Pose {
    Pose {
        geometry: far,
        placement: Placement {
            rotation: turn(0.6, -0.35),
            translation: [7.0, -42.5, 300.0],
            scale,
        },
    }
}

fn poses(far: V3) -> Vec<Pose> {
    let mut out = rigid_poses(far).to_vec();
    out.push(scaled_pose(far, [1.5; 3]));
    out.push(scaled_pose(far, [1.5, 0.5, 2.0]));
    out
}

/// The corners of the live triangles in the model frame of `pose`.
fn model_triangles(scene: &Scene, id: MeshId, pose: Pose) -> Vec<[V3; 3]> {
    let mesh = scene.mesh(id).unwrap();
    (0..mesh.triangle_slots() as u32)
        .filter_map(|t| mesh.triangle(t))
        .map(|tri| {
            tri.map(|v| {
                let p = mesh.position(v).unwrap().map(f64::from);
                pose.to_model(scene.local_to_world(id, p).unwrap())
            })
        })
        .collect()
}

fn volume(triangles: &[[V3; 3]]) -> f64 {
    let six: f64 = triangles
        .iter()
        .map(|[a, b, c]| {
            a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
        })
        .sum();
    six / 6.0
}

fn bounds(triangles: &[[V3; 3]]) -> (V3, V3) {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in triangles.iter().flatten() {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

fn is_close(got: V3, want: V3, tol: f64) -> bool {
    (0..3).all(|k| (got[k] - want[k]).abs() <= tol)
}

fn assert_close(got: V3, want: V3, tol: f64, what: &str) {
    assert!(is_close(got, want, tol), "{what}: {got:?} is not {want:?}");
}

/// Every corner is on an edge or a corner of the boxes whose planes `planes`
/// lists per axis: two of its coordinates are there (an edge of one box that
/// runs into another splits a face diagonal too, so the third may be any).
fn assert_on_edges(triangles: &[[V3; 3]], planes: [&[f64]; 3], tol: f64, what: &str) {
    assert!(!triangles.is_empty(), "{what}: empty");
    for p in triangles.iter().flatten() {
        let mut off: V3 = std::array::from_fn(|k| {
            planes[k]
                .iter()
                .fold(f64::INFINITY, |m, a| m.min((p[k] - a).abs()))
        });
        off.sort_by(f64::total_cmp);
        assert!(off[1] <= tol, "{what}: {p:?} is off by {:e}", off[1]);
    }
}

fn assert_ab_edges(triangles: &[[V3; 3]], tol: f64, what: &str) {
    let planes: [[f64; 4]; 3] =
        std::array::from_fn(|k| [0.0, SIZE[k], SHIFT[k], SIZE[k] + SHIFT[k]]);
    assert_on_edges(triangles, [&planes[0], &planes[1], &planes[2]], tol, what);
}

/// The corners of box A.
fn a_corners() -> [V3; 8] {
    std::array::from_fn(|i| std::array::from_fn(|k| if i >> k & 1 == 1 { SIZE[k] } else { 0.0 }))
}

fn assert_keeps_a_corners(triangles: &[[V3; 3]], tol: f64, what: &str) {
    for corner in a_corners() {
        assert!(
            triangles
                .iter()
                .flatten()
                .any(|&p| is_close(p, corner, tol)),
            "{what}: lost corner {corner:?}"
        );
    }
}

const OPS: [BooleanOp; 3] = [BooleanOp::Union, BooleanOp::Subtract, BooleanOp::Intersect];

fn ops_op(op: BooleanOp) -> fast_mesh_ops::BooleanOp {
    match op {
        BooleanOp::Union => fast_mesh_ops::BooleanOp::Union,
        BooleanOp::Subtract => fast_mesh_ops::BooleanOp::Subtract,
        BooleanOp::Intersect => fast_mesh_ops::BooleanOp::Intersect,
    }
}

/// The volume of `op` on boxes A and B.
fn expected_volume(op: BooleanOp) -> f64 {
    let a: f64 = SIZE.iter().product();
    let both: f64 = (0..3).map(|k| SIZE[k] - SHIFT[k]).product();
    match op {
        BooleanOp::Union => 2.0 * a - both,
        BooleanOp::Subtract => a - both,
        BooleanOp::Intersect => both,
    }
}

fn manifold(op: BooleanOp) -> BooleanOptions {
    BooleanOptions {
        op: ops_op(op),
        method: BooleanMethod::Manifold,
        ..BooleanOptions::default()
    }
}

fn new_scene() -> Scene {
    Scene::new(SceneOptions::default())
}

#[test]
fn boolean_batch_keeps_local_precision_far_away() {
    for far in FAR {
        for pose in poses(far) {
            for op in OPS {
                let what = format!("{op:?} of {pose:?}");
                let mut scene = new_scene();
                let (a, b) = (pose.add_a(&mut scene), pose.add_b(&mut scene));
                let tree = CsgNode::reduce_all(ops_op(op), 2).unwrap();
                let out = scene.boolean_batch(&[a, b], &tree, &manifold(op)).unwrap();
                assert_eq!(scene.placement(out).unwrap(), Placement::IDENTITY);
                let got = model_triangles(&scene, out, pose);
                assert_ab_edges(&got, pose.tol(), &what);
                let (v, want) = (volume(&got), expected_volume(op));
                assert!((v - want).abs() < 1e-5, "{what}: volume {v}");
                // The result's origin is by its inputs.
                let o = pose.to_model(scene.mesh(out).unwrap().origin());
                assert!(
                    (0..3).all(|k| (-3.0..4.0).contains(&o[k])),
                    "{what}: origin {o:?}"
                );
            }
        }
    }
}

#[test]
fn boolean_batch_grids_far_away_as_near() {
    let run = |pose: Pose, method| {
        let mut scene = new_scene();
        let (a, b) = (pose.add_a(&mut scene), pose.add_b(&mut scene));
        let tree = CsgNode::reduce_all(fast_mesh_ops::BooleanOp::Union, 2).unwrap();
        let options = BooleanOptions {
            method,
            voxel_pitch: Some(0.1),
            ..BooleanOptions::default()
        };
        let out = scene.boolean_batch(&[a, b], &tree, &options).unwrap();
        model_triangles(&scene, out, pose)
    };
    for method in [BooleanMethod::BvhSdf, BooleanMethod::FhEdt] {
        for pose in rigid_poses(FAR[1]) {
            let what = format!("{method:?} {pose:?}");
            // As turned, but by the world origin.
            let near = run(
                Pose {
                    geometry: [0.0; 3],
                    placement: Placement {
                        translation: [0.0; 3],
                        ..pose.placement
                    },
                },
                method,
            );
            let got = run(pose, method);
            let ((lo, hi), (near_lo, near_hi)) = (bounds(&got), bounds(&near));
            assert_close(lo, near_lo, 1e-5, &what);
            assert_close(hi, near_hi, 1e-5, &what);
            let off = (volume(&got) - volume(&near)).abs();
            assert!(off < 1e-4, "{what}: volume off by {off}");
        }
    }
}

#[test]
fn boolean_batch_origin_follows_the_meshes_its_tree_uses() {
    let far = FAR[1];
    let pose = rigid_poses(far)[0];
    let mut scene = new_scene();
    let (a, b) = (pose.add_a(&mut scene), pose.add_b(&mut scene));
    // Neither a mesh elsewhere that the tree leaves out nor an empty one
    // moves the frame.
    let elsewhere = Pose::NEAR.add_a(&mut scene);
    let empty = scene.add_mesh(Mesh::new(), MeshOptions::default()).unwrap();
    let tree = CsgNode::reduce_all(fast_mesh_ops::BooleanOp::Union, 2).unwrap();
    let out = scene
        .boolean_batch(
            &[a, b, elsewhere, empty],
            &tree,
            &manifold(BooleanOp::Union),
        )
        .unwrap();
    let mid: V3 = std::array::from_fn(|k| far[k] + 0.5 * (SIZE[k] + SHIFT[k]));
    assert_close(scene.mesh(out).unwrap().origin(), mid, TOL, "origin");
    let before = model_triangles(&scene, out, pose);
    assert_ab_edges(&before, TOL, "result");
    // Undo and redo keep it.
    assert!(scene.undo().unwrap());
    assert!(scene.mesh(out).is_err());
    assert!(scene.redo().unwrap());
    assert_eq!(model_triangles(&scene, out, pose), before);
}

#[test]
fn boolean_batch_of_nothing_stays_at_the_world_origin() {
    let mut scene = new_scene();
    let empty = scene.add_mesh(Mesh::new(), MeshOptions::default()).unwrap();
    let options = BooleanOptions {
        method: BooleanMethod::BvhSdf,
        ..BooleanOptions::default()
    };
    let out = scene
        .boolean_batch(&[empty], &CsgNode::Leaf(0), &options)
        .unwrap();
    assert_eq!(scene.mesh(out).unwrap().triangle_count(), 0);
    assert_eq!(scene.mesh(out).unwrap().origin(), [0.0; 3]);
}

#[test]
fn two_mesh_booleans_keep_local_precision_far_away() {
    for far in FAR {
        for pose in poses(far) {
            for op in OPS {
                for with_options in [false, true] {
                    let what = format!("{op:?} of {pose:?}, options {with_options}");
                    let mut scene = new_scene();
                    let (a, b) = (pose.add_a(&mut scene), pose.add_b(&mut scene));
                    let origin = scene.mesh(a).unwrap().origin();
                    if with_options {
                        scene.boolean_with_options(a, b, &manifold(op)).unwrap();
                    } else {
                        scene.boolean(a, b, op).unwrap();
                    }
                    assert_eq!(scene.mesh(a).unwrap().origin(), origin, "{what}");
                    let got = model_triangles(&scene, a, pose);
                    assert_ab_edges(&got, pose.tol(), &what);
                    let (v, want) = (volume(&got), expected_volume(op));
                    assert!((v - want).abs() < 1e-5, "{what}: volume {v}");
                }
            }
        }
        // A pair in one place by different means.
        let [by_origin, by_placement, ..] = rigid_poses(far);
        let mut scene = new_scene();
        let (a, b) = (by_origin.add_a(&mut scene), by_placement.add_b(&mut scene));
        scene.boolean(a, b, BooleanOp::Union).unwrap();
        assert_ab_edges(&model_triangles(&scene, a, by_origin), TOL, "mixed pair");
    }
}

#[test]
fn plane_cuts_keep_local_precision_far_away() {
    for far in FAR {
        for pose in poses(far) {
            let what = format!("cut of {pose:?}");
            let mut scene = new_scene();
            let a = pose.add_a(&mut scene);
            let plane = Plane::new(
                pose.to_world([0.4, 0.3, 0.2]),
                pose.placement.normal_to_world([1.0, 0.0, 0.0]),
            )
            .unwrap();
            let cut = scene
                .cut_by_plane(a, &CutOptions { plane, cap: true })
                .unwrap();
            for (id, x) in [(a, [0.4, SIZE[0]]), (cut.added, [0.0, 0.4])] {
                let got = model_triangles(&scene, id, pose);
                let allowed: [&[f64]; 3] = [&x, &[0.0, SIZE[1]], &[0.0, SIZE[2]]];
                assert_on_edges(&got, allowed, pose.tol(), &what);
                let want = (x[1] - x[0]) * SIZE[1] * SIZE[2];
                assert!((volume(&got) - want).abs() < 1e-5, "{what}");
            }
        }
    }
}

#[test]
fn convex_hulls_keep_local_precision_far_away() {
    for far in FAR {
        for pose in poses(far) {
            let what = format!("hull of {pose:?}");
            let mut scene = new_scene();
            let (a, b) = (pose.add_a(&mut scene), pose.add_b(&mut scene));
            let hull = scene.convex_hull(&[a, b]).unwrap();
            let got = model_triangles(&scene, hull.added, pose);
            assert_ab_edges(&got, pose.tol(), &what);
            // Across placements: in the world, to the size of the hull.
            let other = rigid_poses(far)[0];
            let c = other.add_a(&mut scene);
            let mixed = scene.convex_hull(&[c, a]).unwrap();
            let corners: Vec<V3> = [other, pose]
                .into_iter()
                .flat_map(|p| a_corners().map(|c| p.to_world(c)))
                .collect();
            let (lo, hi) = bounds(&[[corners[0]; 3], [corners[8]; 3]]);
            let size = (0..3).fold(1.0, |m: f64, k| m.max(hi[k] - lo[k]));
            for &p in model_triangles(&scene, mixed.added, Pose::NEAR)
                .iter()
                .flatten()
            {
                assert!(
                    corners.iter().any(|&c| is_close(p, c, size * TOL)),
                    "{what}: {p:?} is no corner"
                );
            }
        }
    }
}

#[test]
fn wraps_keep_local_precision_far_away() {
    let options = WrapOptions::builder().gap(0.1).detail(24).build();
    let run = |pose: Pose| {
        let mut scene = new_scene();
        let (a, b) = (pose.add_a(&mut scene), pose.add_b(&mut scene));
        let wrap = scene.wrap(&[a, b], &options).unwrap();
        model_triangles(&scene, wrap.added, pose)
    };
    let near = run(Pose::NEAR);
    let (near_lo, near_hi) = bounds(&near);
    for far in FAR {
        for pose in rigid_poses(far) {
            let what = format!("wrap of {pose:?}");
            let got = run(pose);
            let (lo, hi) = bounds(&got);
            assert_close(lo, near_lo, 1e-5, &what);
            assert_close(hi, near_hi, 1e-5, &what);
            let off = (volume(&got) - volume(&near)).abs();
            assert!(off < 1e-4, "{what}: volume off by {off}");
        }
    }
}

#[test]
fn hollows_keep_local_precision_far_away() {
    let options = HollowOptions {
        wall_thickness: 0.1,
        backend: HollowBackend::BvhSdf,
        voxel_size: Some(0.05),
        ..HollowOptions::default()
    };
    let run = |pose: Pose| {
        let mut scene = new_scene();
        let a = pose.add_a(&mut scene);
        let origin = scene.mesh(a).unwrap().origin();
        scene.hollow_mesh(a, &options).unwrap();
        assert_eq!(scene.mesh(a).unwrap().origin(), origin);
        model_triangles(&scene, a, pose)
    };
    let near = run(Pose::NEAR);
    for far in FAR {
        for pose in rigid_poses(far) {
            let what = format!("hollow of {pose:?}");
            let got = run(pose);
            assert_keeps_a_corners(&got, TOL, &what);
            let off = (volume(&got) - volume(&near)).abs();
            assert!(off < 1e-4, "{what}: volume off by {off}");
        }
    }
}

/// The middle of the top of box A.
const TOP: V3 = [0.55, 0.45, 0.7];

/// A box to place: 0.2 wide and 0.3 high, standing on its own zero.
fn add_stamp(scene: &mut Scene) -> MeshId {
    Pose::NEAR.add_box(scene, [-0.1, -0.1, 0.0], [0.1, 0.1, 0.3])
}

/// From the stamp's axis to its corners.
fn stamp_reach() -> f64 {
    0.02f64.sqrt()
}

/// `TOP` and the normal there, in the world.
fn top_of(pose: Pose) -> (V3, V3) {
    let normal = pose.placement.normal_to_world([0.0, 0.0, 1.0]);
    (pose.to_world(TOP), normal)
}

#[test]
fn meshes_placed_apart_keep_local_precision_far_away() {
    for far in FAR {
        for pose in poses(far) {
            for conform_to_surface in [false, true] {
                let what = format!("placed on {pose:?}, conform {conform_to_surface}");
                let mut scene = new_scene();
                let (a, stamp) = (pose.add_a(&mut scene), add_stamp(&mut scene));
                let (hit, normal) = top_of(pose);
                let params = PlaceMeshParams {
                    mode: PlaceMode::KeepSeparate,
                    conform_to_surface,
                    ..PlaceMeshParams::default()
                };
                let (report, placed) = scene
                    .place_mesh_world(a, stamp, hit, normal, &params)
                    .unwrap();
                let placed = placed.unwrap();
                assert_eq!(scene.placement(placed).unwrap(), Placement::IDENTITY);
                assert_close(report.transform.rigid.translation, hit, 1e-9, &what);
                let mut ends = [0usize; 2];
                for &p in model_triangles(&scene, placed, Pose::NEAR).iter().flatten() {
                    let (along, across) = along_and_across(sub(p, hit), normal);
                    let end = usize::from(along > 0.15);
                    ends[end] += 1;
                    let want = 0.3 * end as f64;
                    assert!((along - want).abs() < 1e-5, "{what}: {along} up");
                    assert!(
                        (across - stamp_reach()).abs() < 1e-5,
                        "{what}: {across} off the axis"
                    );
                }
                assert!(ends[0] > 0 && ends[1] > 0, "{what}");
            }
        }
    }
}

#[test]
fn meshes_placed_into_a_target_keep_local_precision_far_away() {
    for far in FAR {
        for pose in poses(far) {
            for mode in [PlaceMode::AppendShell, PlaceMode::EngraveSubtract] {
                let what = format!("{mode:?} on {pose:?}");
                let mut scene = new_scene();
                let (a, stamp) = (pose.add_a(&mut scene), add_stamp(&mut scene));
                let (hit, normal) = top_of(pose);
                let params = PlaceMeshParams {
                    mode,
                    embed_depth: 0.1,
                    ..PlaceMeshParams::default()
                };
                let (report, _) = scene
                    .place_mesh_world(a, stamp, hit, normal, &params)
                    .unwrap();
                let base: V3 = std::array::from_fn(|k| hit[k] - 0.1 * normal[k]);
                // The depth is an f32.
                assert_close(report.transform.rigid.translation, base, 1e-7, &what);
                let got = model_triangles(&scene, a, pose);
                assert!(got.len() > 12, "{what}");
                assert_keeps_a_corners(&got, pose.tol(), &what);
                // The stamp's foot, 0.1 below the surface.
                let foot: Vec<(f64, f64)> = got
                    .iter()
                    .flatten()
                    .map(|&p| along_and_across(sub(pose.to_world(p), hit), normal))
                    .filter(|(along, across)| along.abs() < 0.15 && *across < 0.2)
                    .filter(|(along, _)| *along < -0.05)
                    .collect();
                assert!(foot.len() >= 4, "{what}");
                for (along, across) in foot {
                    assert!((along + 0.1).abs() < 1e-5, "{what}: {along} up");
                    assert!(
                        (across - stamp_reach()).abs() < 1e-5,
                        "{what}: {across} off the axis"
                    );
                }
            }
        }
    }
}

#[test]
fn drain_holes_keep_local_precision_far_away() {
    const RADIUS: f64 = 0.15;
    for far in FAR {
        for pose in poses(far) {
            let what = format!("hole in {pose:?}");
            let mut scene = new_scene();
            let a = pose.add_a(&mut scene);
            let (hit, normal) = top_of(pose);
            let spec = DrainHoleSpec {
                surface_normal: normal.map(|c| c as f32),
                diameter: 2.0 * RADIUS as f32,
                depth: Some(0.2),
                segments: 16,
                ..DrainHoleSpec::default()
            };
            let options = DrainHoleOptions {
                generate_plug: true,
                ..DrainHoleOptions::default()
            };
            let (report, plugs) = scene.punch_drain_hole_at(a, hit, &spec, &options).unwrap();
            assert_eq!((report.holes_punched, plugs.len()), (1, 1), "{what}");
            let got = model_triangles(&scene, a, pose);
            assert_keeps_a_corners(&got, pose.tol(), &what);
            // The rim on the top: 16 corners on the circle around the hit,
            // and points on its sides between them.
            let inner = RADIUS * (std::f64::consts::PI / 16.0).cos();
            let mut rim = 0;
            for &p in got.iter().flatten() {
                let (along, across) = along_and_across(sub(pose.to_world(p), hit), normal);
                if along.abs() < 1e-4 && across < 0.2 {
                    rim += 1;
                    assert!(
                        (inner - 1e-5..RADIUS + 1e-5).contains(&across),
                        "{what}: rim at {across}"
                    );
                }
            }
            assert!(rim >= 16, "{what}: {rim} corners on the rim");
            let reach = model_triangles(&scene, plugs[0], Pose::NEAR)
                .iter()
                .flatten()
                .map(|&p| along_and_across(sub(p, hit), normal).1)
                .fold(0.0, f64::max);
            assert!((reach - RADIUS).abs() < 1e-5, "{what}: plug to {reach}");
        }
    }
}
