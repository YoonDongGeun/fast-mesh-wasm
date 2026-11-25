//! Booleans, cuts, hulls, wraps, hollows and placed meshes (#278), contours,
//! fits, point clouds, trims, selections and surface paths (#280) far from the
//! world origin. Whatever mix of a mesh's `f64` origin and its placement puts
//! it there, the results keep the `f32` precision of small local coordinates:
//! world coordinates in `f32` would be off by 0.004 at 1e5.

use fast_mesh_ops::boolean::{
    DrainHoleOptions, DrainHoleSpec, HollowBackend, HollowOptions, PlaceMeshParams, PlaceMode,
    PlaceUpAxis,
};
use fast_mesh_ops::inspect::{
    ContourSlice, FitPrimitiveOptions, FitPrimitiveReport, FittedPrimitive, PointCloud,
    PrimitiveKind, ReconstructMethod, SurfaceReconstructOptions,
};
use fast_mesh_ops::wrap::WrapOptions;
use fast_mesh_scene::{
    BooleanMethod, BooleanOp, BooleanOptions, CsgNode, CutOptions, GeodesicMethod, GeodesicOptions,
    MeshId, MeshOptions, Placement, Plane, Scene, SceneOptions, TrimCurveOptions,
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

    /// The pose without what puts this one far away: the model as turned and
    /// scaled, its zero at the world's.
    fn near(self) -> Pose {
        Pose {
            geometry: [0.0; 3],
            placement: Placement {
                translation: [0.0; 3],
                ..self.placement
            },
        }
    }

    /// From where [`Pose::near`] puts a point of the model to where this does.
    fn shift(self) -> V3 {
        self.to_world([0.0; 3])
    }

    fn add_mesh(self, scene: &mut Scene, mesh: Mesh) -> MeshId {
        let options = MeshOptions {
            placement: self.placement,
            ..MeshOptions::default()
        };
        scene.add_mesh(mesh, options).unwrap()
    }

    fn add_box(self, scene: &mut Scene, lo: V3, hi: V3) -> MeshId {
        let mesh = box_mesh(add(self.geometry, lo), add(self.geometry, hi));
        self.add_mesh(scene, mesh)
    }

    fn add_model(self, scene: &mut Scene, model: &Model) -> MeshId {
        let moved = model.positions.iter().map(|&p| add(self.geometry, p));
        let positions: Vec<f64> = moved.flatten().collect();
        let mesh = Mesh::from_buffers_f64(&positions, &model.triangles).unwrap();
        self.add_mesh(scene, mesh)
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

/// Positions around zero, and triangles.
struct Model {
    positions: Vec<V3>,
    triangles: Vec<u32>,
}

/// The surface that the profile `(radius, z)` sweeps around the z axis, in
/// `segments` steps, around [`MID`].
fn revolved(profile: &[(f64, f64)], segments: u32) -> Model {
    let positions = profile
        .iter()
        .flat_map(|&(r, z)| {
            (0..segments).map(move |j| {
                let turn = f64::from(j) / f64::from(segments) * std::f64::consts::TAU;
                add(MID, [r * turn.cos(), r * turn.sin(), z])
            })
        })
        .collect();
    let mut triangles = Vec::new();
    for i in 0..profile.len() as u32 - 1 {
        for j in 0..segments {
            let (a, b) = (i * segments + j, i * segments + (j + 1) % segments);
            let (c, d) = (a + segments, b + segments);
            triangles.extend([a, b, d, a, d, c]);
        }
    }
    Model {
        positions,
        triangles,
    }
}

const MID: V3 = [0.55, 0.45, 0.35];
const RADIUS: f64 = 0.6;
const RINGS: u32 = 24;
const SEGMENTS: u32 = 48;

/// A ball around [`MID`] with small holes at its poles: triangles of about
/// 0.08.
fn ball() -> Model {
    let profile: Vec<(f64, f64)> = (0..=RINGS)
        .map(|i| {
            let polar = std::f64::consts::PI * (0.04 + 0.92 * f64::from(i) / f64::from(RINGS));
            (RADIUS * polar.sin(), -RADIUS * polar.cos())
        })
        .collect();
    revolved(&profile, SEGMENTS)
}

/// How far `got` is from `want` moved by `shift`, at most: infinite for
/// another number of points.
fn off(got: &[V3], want: &[V3], shift: V3) -> f64 {
    if got.len() != want.len() {
        return f64::INFINITY;
    }
    let pairs = got.iter().zip(want);
    pairs.fold(0.0, |m: f64, (g, w)| {
        (0..3).fold(m, |m, k| m.max((g[k] - (w[k] + shift[k])).abs()))
    })
}

/// [`off`] for plain numbers.
fn off_by(got: &[f64], want: &[f64]) -> f64 {
    if got.len() != want.len() {
        return f64::INFINITY;
    }
    let pairs = got.iter().zip(want);
    pairs.fold(0.0, |m: f64, (g, w)| m.max((g - w).abs()))
}

/// Compares what `run` makes of each of the `poses` of every far place with
/// what it makes of it near, and the shift between the two.
fn far_and_near<T>(
    poses: impl Fn(V3) -> Vec<Pose>,
    run: impl Fn(Pose) -> T,
    compare: impl Fn(Pose, T, &T),
) {
    for (i, pose) in poses(FAR[0]).into_iter().enumerate() {
        let near = run(pose.near());
        for far in FAR {
            let pose = poses(far)[i];
            compare(pose, run(pose), &near);
        }
    }
}

/// [`TOL`] in the world, at the largest scale of [`poses`].
const WORLD_TOL: f64 = 2.0 * TOL;

fn check(off: f64, tol: f64, what: &str) {
    assert!(off <= tol, "{what}: off by {off:e}");
}

/// What a slice has in its plane, which does not depend on where that is.
fn in_plane(slice: &ContourSlice) -> Vec<f64> {
    let mut out = vec![slice.total_area, slice.total_perimeter];
    for l in &slice.loops {
        out.extend([l.perimeter, l.enclosed_area]);
        out.extend([l.min_caliper_diameter, l.max_caliper_diameter]);
        out.extend(l.centroid_2d);
        out.extend(l.points_2d.iter().flatten());
    }
    out
}

/// The points of a slice in the world: its plane's origin, then the loops'.
fn in_world(slice: &ContourSlice) -> Vec<V3> {
    let mut out = vec![slice.plane_origin];
    for l in &slice.loops {
        out.push(l.centroid_3d);
        out.extend(&l.points_3d);
    }
    out
}

/// The points of a slice from its axes and `(u, v)` coordinates.
fn from_plane(slice: &ContourSlice) -> Vec<V3> {
    let o = slice.plane_origin;
    let points = slice.loops.iter().flat_map(|l| &l.points_2d);
    points
        .map(|[u, v]| std::array::from_fn(|k| o[k] + u * slice.u_axis[k] + v * slice.v_axis[k]))
        .collect()
}

const NORMALS: [V3; 2] = [[0.0, 0.0, 1.0], [0.36, 0.48, 0.8]];

#[test]
fn contours_keep_local_precision_far_away() {
    for normal in NORMALS {
        let run = |pose: Pose| {
            let mut scene = new_scene();
            let (a, b) = (pose.add_a(&mut scene), pose.add_model(&mut scene, &ball()));
            let o = pose.to_world(MID);
            let mut out = Vec::new();
            for id in [a, b] {
                out.push(scene.slice_contours_world(id, o, normal).unwrap());
                let stack = scene.slice_contour_stack_world(id, o, normal, 0.05, 4);
                out.extend(stack.unwrap());
            }
            out
        };
        far_and_near(poses, run, |pose, got, near| {
            let what = format!("contours of {pose:?} along {normal:?}");
            assert!(near.iter().all(|s| !s.loops.is_empty()), "{what}");
            for (slice, want) in got.iter().zip(near) {
                let off_world = off(&in_world(slice), &in_world(want), pose.shift());
                check(off_world, WORLD_TOL, &what);
                let off_plane = off_by(&in_plane(slice), &in_plane(want));
                check(off_plane, 4.0 * WORLD_TOL, &what);
            }
        });
    }
}

#[test]
fn contours_on_a_plane_given_by_a_far_point() {
    // A point of the plane 1e5 from the mesh along it: the world's origin,
    // say, for a plane through a mesh far away.
    for normal in NORMALS {
        let run = |pose: Pose| {
            let mut scene = new_scene();
            let id = pose.add_model(&mut scene, &ball());
            let plane = Plane::new([0.0; 3], normal).unwrap();
            let (mid, u, v) = (pose.to_world(MID), plane.u(), plane.v());
            let is_near = [pose.geometry, pose.placement.translation] == [[0.0; 3]; 2];
            let along = if is_near { 0.0 } else { 1.0e5 };
            let o = std::array::from_fn(|k| mid[k] + along * (u[k] - 0.5 * v[k]));
            scene.slice_contours_world(id, o, normal).unwrap()
        };
        let points = |s: &ContourSlice| in_world(s).split_off(1);
        far_and_near(poses, run, |pose, got, near| {
            let what = format!("contours of {pose:?} along {normal:?}");
            let off_world = off(&points(&got), &points(near), pose.shift());
            check(off_world, WORLD_TOL, &what);
            let off_plane = off(&from_plane(&got), &from_plane(near), pose.shift());
            check(off_plane, WORLD_TOL, &what);
        });
    }
}

/// The primitive's point, and its other numbers.
fn fitted(report: &FitPrimitiveReport) -> (V3, Vec<f64>) {
    let mut rest = vec![report.rms_error, report.max_error, report.inlier_ratio];
    let point = match report.primitive {
        FittedPrimitive::Plane { origin, normal } => {
            rest.extend(normal);
            origin
        }
        FittedPrimitive::Sphere { center, radius } => {
            rest.push(radius);
            center
        }
        FittedPrimitive::Cylinder {
            axis_origin,
            axis_dir,
            radius,
            height,
        } => {
            rest.extend(axis_dir);
            rest.extend([radius, height]);
            axis_origin
        }
        FittedPrimitive::Cone {
            apex,
            axis_dir,
            half_angle_rad,
            height,
        } => {
            rest.extend(axis_dir);
            rest.extend([half_angle_rad, height]);
            apex
        }
    };
    (point, rest)
}

#[test]
fn fitted_primitives_keep_local_precision_far_away() {
    let line = |r0: f64, r1: f64| -> Vec<(f64, f64)> {
        (0..=8)
            .map(|i| {
                let t = f64::from(i) / 8.0;
                (r0 + t * (r1 - r0), 0.9 * t - 0.45)
            })
            .collect()
    };
    let shapes = [
        (PrimitiveKind::Sphere, ball()),
        (PrimitiveKind::Cylinder, revolved(&line(0.4, 0.4), 32)),
        (PrimitiveKind::Cone, revolved(&line(0.1, 0.55), 32)),
        (PrimitiveKind::Auto, revolved(&line(0.4, 0.4), 32)),
    ];
    let poses = |far| {
        let mut all = rigid_poses(far).to_vec();
        all.push(scaled_pose(far, [1.5; 3]));
        all.push(scaled_pose(far, [1.5, 1.5, 2.0]));
        all
    };
    for (kind, model) in &shapes {
        let run = |pose: Pose| {
            let mut scene = new_scene();
            let id = pose.add_model(&mut scene, model);
            let fit = |scene: &Scene| {
                let options = FitPrimitiveOptions::default();
                scene.fit_primitive_world(id, None, Some(*kind), &options)
            };
            let report = fit(&scene).unwrap();
            // Aligning it takes the mesh to the target.
            scene
                .align_to_fitted_primitive(id, &report.primitive, [0.0, 0.0, 1.0], [0.0; 3])
                .unwrap();
            let aligned = fitted(&fit(&scene).unwrap()).0;
            (report, aligned)
        };
        far_and_near(poses, run, |pose, (got, aligned), (near, _)| {
            let what = format!("{kind:?} fitted to {pose:?}");
            let ((point, rest), (near_point, near_rest)) = (fitted(&got), fitted(near));
            check(off(&[point], &[near_point], pose.shift()), 1e-5, &what);
            check(off_by(&rest, &near_rest), 1e-5, &what);
            let residuals = |r: &FitPrimitiveReport| -> Vec<f64> {
                r.residuals.iter().map(|&x| f64::from(x)).collect()
            };
            check(off_by(&residuals(&got), &residuals(near)), 1e-5, &what);
            check(off(&[aligned], &[[0.0; 3]], [0.0; 3]), 1e-5, &what);
        });
    }
}

/// Points on the ball around a point of it, `angle` from it, in the model.
fn around_on_ball(angle: f64, count: u32) -> Vec<V3> {
    (0..count)
        .map(|k| {
            let turn = f64::from(k) / f64::from(count) * std::f64::consts::TAU + 0.05;
            let (s, c) = angle.sin_cos();
            let d = [c, s * turn.cos(), s * turn.sin()];
            add(MID, d.map(|x| RADIUS * x))
        })
        .collect()
}

/// The vertices of a mesh in the model, and its triangles.
fn model_buffers(scene: &Scene, id: MeshId, pose: Pose) -> (Vec<V3>, Vec<[u32; 3]>) {
    let mesh = scene.mesh(id).unwrap();
    let mut slots = vec![u32::MAX; mesh.vertex_slots()];
    let mut positions = Vec::new();
    for v in 0..mesh.vertex_slots() as u32 {
        if let Some(p) = mesh.position(v) {
            slots[v as usize] = positions.len() as u32;
            let w = scene.local_to_world(id, p.map(f64::from)).unwrap();
            positions.push(pose.to_model(w));
        }
    }
    let triangles = (0..mesh.triangle_slots() as u32)
        .filter_map(|t| mesh.triangle(t))
        .map(|tri| tri.map(|v| slots[v as usize]))
        .collect();
    (positions, triangles)
}

#[test]
fn trims_keep_local_precision_far_away() {
    for closed in [true, false] {
        let run = |pose: Pose| {
            let mut scene = new_scene();
            let id = pose.add_model(&mut scene, &ball());
            let options = TrimCurveOptions {
                points: around_on_ball(0.5, 12)
                    .into_iter()
                    .map(|p| pose.to_world(p))
                    .collect(),
                closed,
                delete_inside: true,
                snap_tolerance: 1e-3,
            };
            let report = scene.trim_by_curve(id, &options).unwrap();
            (report, model_buffers(&scene, id, pose))
        };
        far_and_near(poses, run, |pose, (got, mesh), (near, near_mesh)| {
            let what = format!("trim of {pose:?}, closed {closed}");
            assert_eq!(near.removed_area > 0.1, closed, "{what}");
            let off_area = (got.removed_area - near.removed_area).abs();
            check(off_area, 1e-5, &what);
            let off_mesh = off(&mesh.0, &near_mesh.0, [0.0; 3]);
            check(off_mesh, 4.0 * pose.tol(), &what);
            assert_eq!(mesh.1, near_mesh.1, "{what}");
        });
    }
}

#[test]
fn selections_far_away_are_those_near() {
    let run = |pose: Pose| {
        let mut scene = new_scene();
        let id = pose.add_model(&mut scene, &ball());
        let center = pose.to_world(around_on_ball(0.2, 5)[1]);
        let axis = pose.placement.normal_to_world([1.0, 0.1, 0.2]);
        let mut found = Vec::new();
        for radius in [0.04, 0.17, 0.43] {
            for cone in [None, Some((axis, 0.5))] {
                let faces = scene
                    .select_faces_in_sphere_world(id, center, radius, cone)
                    .unwrap();
                found.push((format!("sphere of {radius}, cone {cone:?}"), faces));
            }
        }
        for angle in [0.05, 0.1, 0.125, 0.15, 0.25, 0.4] {
            let faces = scene.select_by_crease_angle(id, 1_200, angle).unwrap();
            found.push((format!("crease of {angle}"), faces));
        }
        let a = pose.add_a(&mut scene);
        for id in [id, a] {
            for area in [1.0, 2.0, 4.0, 5.1, 8.0, 12.0] {
                let faces = scene
                    .select_small_face_islands(id, 1_000_000, Some(area))
                    .unwrap();
                found.push((format!("islands of at most {area}"), faces));
            }
        }
        found
    };
    far_and_near(poses, run, |pose, got, near| {
        for ((what, got), (_, want)) in got.iter().zip(near) {
            let (n, m) = (got.len(), want.len());
            assert!(got == want, "{what} on {pose:?}: {n} faces, not {m}");
        }
    });
}

#[test]
fn surface_paths_keep_local_precision_far_away() {
    let (from, to) = (4 * SEGMENTS + 3, 20 * SEGMENTS + 26);
    let run = |pose: Pose| -> Vec<V3> {
        let what = format!("path on {pose:?}");
        let mut scene = new_scene();
        let id = pose.add_model(&mut scene, &ball());
        let mut distances = Vec::new();
        let options = GeodesicOptions {
            method: GeodesicMethod::Dijkstra,
            ..GeodesicOptions::default()
        };
        scene
            .geodesic_distances(id, &[from], &options, &mut distances)
            .unwrap();
        let mut path = Vec::new();
        scene
            .shortest_surface_path(id, &distances, &[from], to, &mut path)
            .unwrap();
        // Every point is where its barycentric coordinates say.
        let mesh = scene.mesh(id).unwrap();
        let corner = |v: u32| mesh.position(v).unwrap().map(f64::from);
        for p in &path {
            let [a, b, c] = mesh.triangle(p.triangle).unwrap().map(corner);
            let [wa, wb, wc] = p.barycentric.map(f64::from);
            let at = std::array::from_fn(|k| wa * a[k] + wb * b[k] + wc * c[k]);
            let on = scene.local_to_world(id, at).unwrap();
            check(off(&[p.position], &[on], [0.0; 3]), WORLD_TOL, &what);
        }
        path.iter().map(|p| p.position).collect()
    };
    far_and_near(poses, run, |pose, got, near| {
        assert!(near.len() > 20);
        let what = format!("path on {pose:?}");
        check(off(&got, near, pose.shift()), WORLD_TOL, &what);
    });
}

/// The vertices of a mesh in the world.
fn world_positions(scene: &Scene, id: MeshId) -> Vec<V3> {
    model_buffers(scene, id, Pose::NEAR).0
}

#[test]
fn reconstructed_clouds_keep_local_precision_far_away() {
    // As a file has them: text, with every digit.
    let cloud_at = |at: V3| {
        let rows: Vec<String> = around_on_ball(0.3, 24)
            .into_iter()
            .chain(around_on_ball(0.2, 18))
            .chain(around_on_ball(0.1, 9))
            .chain([add(MID, [RADIUS, 0.0, 0.0])])
            .map(|p| {
                let ([x, y, z], n) = (add(at, p), sub(p, MID).map(|c| c / RADIUS));
                format!("{x:?} {y:?} {z:?} {:?} {:?} {:?}", n[0], n[1], n[2])
            })
            .collect();
        PointCloud::parse_xyz_centered(&rows.join("\n")).unwrap()
    };
    for method in [
        ReconstructMethod::ImplicitMlsMarchingCubes,
        ReconstructMethod::BallPivoting,
    ] {
        let options = SurfaceReconstructOptions {
            method,
            ..SurfaceReconstructOptions::default()
        };
        let run = |at: V3| {
            let mut scene = new_scene();
            let (cloud, origin) = cloud_at(at);
            let id = scene
                .reconstruct_point_cloud_at(&cloud, origin, &options)
                .unwrap();
            assert_eq!(scene.mesh(id).unwrap().origin(), origin);
            // One step, which redo brings back as it was.
            let made = world_positions(&scene, id);
            assert!(scene.undo().unwrap());
            assert!(scene.redo().unwrap());
            assert_eq!(world_positions(&scene, id), made);
            made
        };
        let near = run([0.0; 3]);
        for far in FAR {
            let what = format!("{method:?} at {far:?}");
            check(off(&run(far), &near, far), 4.0 * TOL, &what);
        }
    }
    let mut scene = new_scene();
    let (cloud, _) = cloud_at([0.0; 3]);
    let options = SurfaceReconstructOptions::default();
    for x in [f64::NAN, f64::INFINITY] {
        let made = scene.reconstruct_point_cloud_at(&cloud, [0.0, x, 0.0], &options);
        assert!(made.is_err());
    }
}

/// Places `stamp` on the top of box A in `pose`, and returns what that made:
/// the stamp itself, or the box with it. With `anchor` by that point of it.
fn placed_on_top(
    pose: Pose,
    add_stamp: impl Fn(&mut Scene) -> MeshId,
    anchor: Option<V3>,
    params: &PlaceMeshParams,
) -> Vec<V3> {
    let mut scene = new_scene();
    let (a, stamp) = (pose.add_a(&mut scene), add_stamp(&mut scene));
    let (hit, normal) = top_of(pose);
    let (report, added) = match anchor {
        None => scene.place_mesh_world(a, stamp, hit, normal, params),
        Some(p) => scene.place_mesh_anchored(a, stamp, p, hit, normal, params),
    }
    .unwrap();
    let placed = world_positions(&scene, added.unwrap_or(a));
    // The transform takes the stamp's own coordinates there.
    if let Some(added) = added
        && !params.conform_to_surface
    {
        let t = &report.transform;
        let scale = scene.placement(stamp).unwrap().scale;
        let mesh = scene.mesh(stamp).unwrap();
        let by_transform: Vec<V3> = (0..mesh.vertex_slots() as u32)
            .map(|v| {
                let p = mesh.world_position(v).unwrap();
                let s: V3 = std::array::from_fn(|k| t.scale * scale[k] * p[k]);
                std::array::from_fn(|k| {
                    let r = t.rigid.rotation[k];
                    r[0] * s[0] + r[1] * s[1] + r[2] * s[2] + t.rigid.translation[k]
                })
            })
            .collect();
        for p in &placed {
            let nearest = by_transform
                .iter()
                .map(|&q| off(&[*p], &[q], [0.0; 3]))
                .fold(f64::INFINITY, f64::min);
            check(nearest, 1e-5, "transform");
        }
        assert_eq!(scene.placement(added).unwrap(), Placement::IDENTITY);
    }
    placed
}

/// [`add_stamp`] with its geometry moved by `at`, and lying on another side
/// for another up axis.
fn add_stamp_at(scene: &mut Scene, at: V3, up: PlaceUpAxis) -> MeshId {
    let (lo, hi) = match up {
        PlaceUpAxis::PosZ => ([-0.1, -0.1, 0.0], [0.1, 0.1, 0.3]),
        PlaceUpAxis::NegZ => ([-0.1, -0.1, -0.3], [0.1, 0.1, 0.0]),
        PlaceUpAxis::PosY => ([-0.1, 0.0, -0.1], [0.1, 0.3, 0.1]),
        PlaceUpAxis::NegY => ([-0.1, -0.3, -0.1], [0.1, 0.0, 0.1]),
    };
    let mesh = box_mesh(add(at, lo), add(at, hi));
    let options = MeshOptions {
        // What counts of it is the scale.
        placement: Placement {
            scale: [1.0, 1.25, 0.8],
            ..moved([40.0, 50.0, 60.0])
        },
        ..MeshOptions::default()
    };
    scene.add_mesh(mesh, options).unwrap()
}

#[test]
fn stamps_far_from_their_own_zero_stand_on_the_hit_point() {
    for far in FAR {
        for pose in [Pose::NEAR, rigid_poses(far)[2], scaled_pose(far, [1.5; 3])] {
            for up_axis in [
                PlaceUpAxis::PosZ,
                PlaceUpAxis::NegZ,
                PlaceUpAxis::PosY,
                PlaceUpAxis::NegY,
            ] {
                for (mode, conform_to_surface) in [
                    (PlaceMode::KeepSeparate, false),
                    (PlaceMode::KeepSeparate, true),
                    (PlaceMode::AppendShell, false),
                    (PlaceMode::EmbedUnion, false),
                ] {
                    let what = format!("{up_axis:?} {mode:?} stamp of {far:?} on {pose:?}");
                    let params = PlaceMeshParams {
                        mode,
                        conform_to_surface,
                        up_axis,
                        twist_angle_rad: 0.4,
                        uniform_scale: 0.9,
                        embed_depth: 0.05,
                        ..PlaceMeshParams::default()
                    };
                    let at = |p: V3| move |s: &mut Scene| add_stamp_at(s, p, up_axis);
                    let near = placed_on_top(pose, at([0.0; 3]), None, &params);
                    let got = placed_on_top(pose, at(far), None, &params);
                    check(off(&got, &near, [0.0; 3]), 1e-5, &what);
                    // Or with the point given.
                    let got = placed_on_top(pose, at(far), Some(far), &params);
                    check(off(&got, &near, [0.0; 3]), 1e-5, &what);
                    let moved = [0.02, -0.03, 0.01];
                    let near = placed_on_top(pose, at([0.0; 3]), Some(moved), &params);
                    let got = placed_on_top(pose, at(far), Some(add(far, moved)), &params);
                    check(off(&got, &near, [0.0; 3]), 1e-5, &what);
                }
            }
        }
    }
}

#[test]
fn stamps_near_their_own_zero_are_placed_by_it() {
    let params = PlaceMeshParams {
        mode: PlaceMode::KeepSeparate,
        ..PlaceMeshParams::default()
    };
    let (up, zero) = (PlaceUpAxis::PosZ, Some([0.0; 3]));
    // The stamp is 0.3 large: in reach of its zero up to 307.2 from it.
    for (x, in_reach) in [(0.5, true), (300.0, true), (-300.0, true), (310.0, false)] {
        let at = |s: &mut Scene| add_stamp_at(s, [x, 0.0, 0.0], up);
        let by_default = placed_on_top(Pose::NEAR, at, None, &params);
        let by_zero = placed_on_top(Pose::NEAR, at, zero, &params);
        assert_eq!(by_default == by_zero, in_reach, "{x}");
        let underside = placed_on_top(Pose::NEAR, at, Some([x, 0.0, 0.0]), &params);
        let off_underside = off(&by_default, &underside, [0.0; 3]);
        assert_eq!(off_underside < 1e-6, !in_reach, "{x}");
    }
}

#[test]
fn a_stamp_by_its_far_zero_keeps_local_precision() {
    let params = PlaceMeshParams {
        mode: PlaceMode::KeepSeparate,
        twist_angle_rad: 0.4,
        ..PlaceMeshParams::default()
    };
    let up = PlaceUpAxis::PosZ;
    for far in FAR {
        let pose = rigid_poses(far)[2];
        let near = placed_on_top(pose, |s| add_stamp_at(s, [0.0; 3], up), None, &params);
        let zero = Some([0.0; 3]);
        let got = placed_on_top(pose, |s| add_stamp_at(s, far, up), zero, &params);
        // Off the hit point by its own distance from zero: everywhere the same.
        let shift = sub(got[0], near[0]);
        let length = |v: V3| (0..3).map(|k| v[k] * v[k]).sum::<f64>().sqrt();
        let scaled = [far[0], 1.25 * far[1], 0.8 * far[2]];
        check((length(shift) - length(scaled)).abs(), 1e-5, "distance");
        check(off(&got, &near, shift), 1e-5, &format!("stamp of {far:?}"));
    }
}
