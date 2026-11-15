//! Integration tests for configurable inter-object collision responses (#194).

use std::collections::BTreeMap;

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::self_intersecting_pairs;
use fast_mesh_scene::{
    BooleanMethod, CollisionDistanceBackend, CollisionMode, CollisionRole, CollisionRule,
    CutTrigger, MaterialMemory, MeshOptions, Placement, Scene, SceneClosestOptions, SceneError,
    SceneOptions, SoftDeformSolver, ordered_pair,
};
use fast_mesh_store::Mesh;

fn box_mesh(min: [f32; 3], max: [f32; 3]) -> Mesh {
    subdivided_box(min, max, 1)
}

/// Builds a closed, outward-wound, watertight 2-manifold box subdivided `n` times per edge.
fn subdivided_box(min: [f32; 3], max: [f32; 3], n: usize) -> Mesh {
    assert!(n >= 1);
    let mut map: BTreeMap<(usize, usize, usize), u32> = BTreeMap::new();
    let mut positions: Vec<f32> = Vec::new();
    let mut vert = |ix: usize, iy: usize, iz: usize| -> u32 {
        if let Some(&id) = map.get(&(ix, iy, iz)) {
            return id;
        }
        let id = (positions.len() / 3) as u32;
        let fx = ix as f32 / n as f32;
        let fy = iy as f32 / n as f32;
        let fz = iz as f32 / n as f32;
        positions.push(min[0] + fx * (max[0] - min[0]));
        positions.push(min[1] + fy * (max[1] - min[1]));
        positions.push(min[2] + fz * (max[2] - min[2]));
        map.insert((ix, iy, iz), id);
        id
    };

    let mut tris: Vec<u32> = Vec::new();
    let mut push_quad = |a: u32, b: u32, c: u32, d: u32| {
        tris.extend_from_slice(&[a, b, c, a, c, d]);
    };

    for u in 0..n {
        for v in 0..n {
            // +Z (ix = u, iy = v, iz = n)
            let a = vert(u, v, n);
            let b = vert(u + 1, v, n);
            let c = vert(u + 1, v + 1, n);
            let d = vert(u, v + 1, n);
            push_quad(a, b, c, d);

            // -Z (ix = u, iy = v, iz = 0)
            let a = vert(u, v, 0);
            let b = vert(u, v + 1, 0);
            let c = vert(u + 1, v + 1, 0);
            let d = vert(u + 1, v, 0);
            push_quad(a, b, c, d);

            // +X (ix = n, iy = u, iz = v)
            let a = vert(n, u, v);
            let b = vert(n, u + 1, v);
            let c = vert(n, u + 1, v + 1);
            let d = vert(n, u, v + 1);
            push_quad(a, b, c, d);

            // -X (ix = 0, iy = u, iz = v)
            let a = vert(0, u, v);
            let b = vert(0, u, v + 1);
            let c = vert(0, u + 1, v + 1);
            let d = vert(0, u + 1, v);
            push_quad(a, b, c, d);

            // +Y (ix = v, iy = n, iz = u)
            let a = vert(v, n, u);
            let b = vert(v, n, u + 1);
            let c = vert(v + 1, n, u + 1);
            let d = vert(v + 1, n, u);
            push_quad(a, b, c, d);

            // -Y (ix = u, iy = 0, iz = v)
            let a = vert(u, 0, v);
            let b = vert(u + 1, 0, v);
            let c = vert(u + 1, 0, v + 1);
            let d = vert(u, 0, v + 1);
            push_quad(a, b, c, d);
        }
    }

    Mesh::from_buffers(&positions, &tris).unwrap()
}

/// Builds a closed outward-wound subdivided octahedron sphere (`subdivisions` 2 => 128 tris).
fn icosphere(center: [f32; 3], radius: f32, subdivisions: usize) -> Mesh {
    let mut verts: Vec<[f64; 3]> = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let mut tris: Vec<[u32; 3]> = vec![
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];

    for _ in 0..subdivisions {
        let mut mid_cache: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut midpoint = |a: u32, b: u32| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&idx) = mid_cache.get(&key) {
                return idx;
            }
            let va = verts[a as usize];
            let vb = verts[b as usize];
            let m = [
                0.5 * (va[0] + vb[0]),
                0.5 * (va[1] + vb[1]),
                0.5 * (va[2] + vb[2]),
            ];
            let len = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
            let u = [m[0] / len, m[1] / len, m[2] / len];
            let idx = verts.len() as u32;
            verts.push(u);
            mid_cache.insert(key, idx);
            idx
        };

        let mut next = Vec::with_capacity(tris.len() * 4);
        for [a, b, c] in tris {
            let ab = midpoint(a, b);
            let bc = midpoint(b, c);
            let ca = midpoint(c, a);
            next.push([a, ab, ca]);
            next.push([b, bc, ab]);
            next.push([c, ca, bc]);
            next.push([ab, bc, ca]);
        }
        tris = next;
    }

    let mut positions = Vec::with_capacity(verts.len() * 3);
    for v in verts {
        positions.push(center[0] + (v[0] as f32) * radius);
        positions.push(center[1] + (v[1] as f32) * radius);
        positions.push(center[2] + (v[2] as f32) * radius);
    }
    let indices: Vec<u32> = tris.into_iter().flatten().collect();
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn translate(t: [f64; 3]) -> Placement {
    Placement {
        translation: t,
        ..Placement::IDENTITY
    }
}

fn ry_deg(deg: f64) -> [[f64; 3]; 3] {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

fn max_vertex_drift(a: &[[f32; 3]], b: &[[f32; 3]]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(p, q)| {
            let dx = f64::from(p[0]) - f64::from(q[0]);
            let dy = f64::from(p[1]) - f64::from(q[1]);
            let dz = f64::from(p[2]) - f64::from(q[2]);
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(0.0, f64::max)
}

#[test]
fn registry_ordered_pair_defaults_validation_and_removal_cleanup() {
    let mut scene = Scene::new(SceneOptions::default());
    let a = scene
        .add_mesh(box_mesh([0.0; 3], [1.0; 3]), MeshOptions::default())
        .unwrap();
    let b = scene
        .add_mesh(
            box_mesh([2.0, 0.0, 0.0], [3.0, 1.0, 1.0]),
            MeshOptions::default(),
        )
        .unwrap();
    let c = scene
        .add_mesh(
            box_mesh([4.0, 0.0, 0.0], [5.0, 1.0, 1.0]),
            MeshOptions::default(),
        )
        .unwrap();

    assert_eq!(ordered_pair(a, b), ordered_pair(b, a));
    assert_eq!(
        scene.default_collision_rule().mode,
        CollisionMode::PassThrough
    );

    let default_rigid = CollisionRule {
        mode: CollisionMode::Rigid,
        contact_margin: 0.02,
        ..CollisionRule::default()
    };
    scene.set_default_collision_rule(default_rigid).unwrap();
    assert_eq!(*scene.default_collision_rule(), default_rigid);
    assert_eq!(scene.collision_rule(a, c).unwrap(), default_rigid);

    let pair_cut = CollisionRule {
        mode: CollisionMode::Cut,
        role: CollisionRole::MovingIsCutter,
        cut_trigger: CutTrigger::LiveStep,
        ..CollisionRule::default()
    };
    scene.set_collision_rule(b, a, pair_cut).unwrap();
    assert_eq!(scene.collision_rule(a, b).unwrap(), pair_cut);
    assert_eq!(scene.collision_rule(b, a).unwrap(), pair_cut);

    assert!(matches!(
        scene.set_collision_rule(a, a, pair_cut),
        Err(SceneError::InvalidInput(_))
    ));
    for bad in [
        CollisionRule {
            compliance: -0.1,
            ..CollisionRule::default()
        },
        CollisionRule {
            stiffness: f32::NAN,
            ..CollisionRule::default()
        },
        CollisionRule {
            recovery_rate: 1.1,
            ..CollisionRule::default()
        },
        CollisionRule {
            contact_margin: -0.01,
            ..CollisionRule::default()
        },
    ] {
        assert!(matches!(
            scene.set_default_collision_rule(bad),
            Err(SceneError::InvalidInput(_))
        ));
    }

    assert!(scene.clear_collision_rule(a, b).unwrap());
    assert!(!scene.clear_collision_rule(a, b).unwrap());
    assert_eq!(scene.collision_rule(a, b).unwrap(), default_rigid);

    scene.set_collision_rule(a, b, pair_cut).unwrap();
    scene.remove_mesh(b).unwrap();
    assert_eq!(scene.collision_rule(a, b), Err(SceneError::UnknownMesh));
    assert_eq!(
        scene.clear_collision_rule(a, b),
        Err(SceneError::UnknownMesh)
    );
}

#[test]
fn passthrough_moves_freely_and_reports_margin_and_intersection() {
    let mut scene = Scene::new(SceneOptions::default());
    let anchor = scene
        .add_mesh(box_mesh([0.0; 3], [1.0; 3]), MeshOptions::default())
        .unwrap();
    let mover = scene
        .add_mesh(
            box_mesh([0.0; 3], [1.0; 3]),
            MeshOptions {
                placement: translate([2.0, 0.0, 0.0]),
                ..MeshOptions::default()
            },
        )
        .unwrap();

    scene
        .set_collision_rule(
            anchor,
            mover,
            CollisionRule {
                mode: CollisionMode::PassThrough,
                contact_margin: 0.25,
                ..CollisionRule::default()
            },
        )
        .unwrap();

    // Near within contact margin without intersection.
    let near = translate([1.15, 0.0, 0.0]);
    let rep_near = scene.move_with_collision(mover, near, false).unwrap();
    assert_eq!(rep_near.final_placement, near);
    assert_eq!(rep_near.active_pairs.len(), 1);
    assert!(!rep_near.active_pairs[0].intersecting);
    assert!((rep_near.active_pairs[0].min_distance - 0.15).abs() < 1e-4);
    assert!((rep_near.min_clearance - 0.15).abs() < 1e-4);

    // Move straight through anchor: passes through and reports intersection.
    let inside = translate([0.4, 0.0, 0.0]);
    let rep_in = scene.move_with_collision(mover, inside, true).unwrap();
    assert_eq!(rep_in.final_placement, inside);
    assert_eq!(rep_in.active_pairs.len(), 1);
    assert!(rep_in.active_pairs[0].intersecting);
    assert_eq!(rep_in.active_pairs[0].min_distance, 0.0);
    assert_eq!(rep_in.min_clearance, 0.0);
    assert_eq!(scene.intersects(anchor, mover), Ok(true));
}

#[test]
fn rigid_stops_cleanly_without_interpenetration_or_tunneling() {
    let mut scene = Scene::new(SceneOptions::default());
    let wall = scene
        .add_mesh(
            box_mesh([1.0, -2.0, -2.0], [1.2, 2.0, 2.0]),
            MeshOptions::default(),
        )
        .unwrap();
    let block = scene
        .add_mesh(
            box_mesh([0.0, -0.5, -0.5], [1.0, 0.5, 0.5]),
            MeshOptions {
                placement: translate([-1.0, 0.0, 0.0]),
                ..MeshOptions::default()
            },
        )
        .unwrap();

    scene
        .set_collision_rule(
            wall,
            block,
            CollisionRule {
                mode: CollisionMode::Rigid,
                rigid_slide: false,
                ..CollisionRule::default()
            },
        )
        .unwrap();

    // Target is far past the opposite side of the thin wall; must not tunnel through!
    let target = translate([5.0, 0.0, 0.0]);
    let rep = scene.move_with_collision(block, target, true).unwrap();
    assert_eq!(scene.intersects(wall, block), Ok(false));
    assert!(
        rep.final_placement.translation[0] <= 0.001,
        "stopped in front of wall at x=1: got {:?}",
        rep.final_placement.translation
    );
    assert!(
        rep.final_placement.translation[0] > -0.05,
        "advanced up to the wall: got {:?}",
        rep.final_placement.translation
    );
    assert_eq!(rep.active_pairs.len(), 1);
    assert!(rep.active_pairs[0].contact_blocked);
    assert!(!rep.active_pairs[0].intersecting);
    assert!(rep.min_clearance >= 0.0 && rep.min_clearance < 0.01);

    // Undo restores the pre-move placement cleanly.
    assert_eq!(scene.undo(), Ok(true));
    assert_eq!(scene.placement(block).unwrap(), translate([-1.0, 0.0, 0.0]));
}

#[test]
fn rigid_slides_tangentially_along_angled_surface() {
    let mut scene = Scene::new(SceneOptions::default());
    // Flat slab rotated 45 deg about Y: top normal points (+X, 0, +Z) / sqrt(2).
    let ramp = scene
        .add_mesh(
            box_mesh([-3.0, -3.0, -0.5], [3.0, 3.0, 0.0]),
            MeshOptions {
                placement: Placement {
                    rotation: ry_deg(45.0),
                    translation: [0.0, 0.0, 0.0],
                    scale: [1.0; 3],
                },
                ..MeshOptions::default()
            },
        )
        .unwrap();

    let start = translate([0.0, 0.0, 2.0]);
    let sphere = scene
        .add_mesh(
            icosphere([0.0, 0.0, 0.0], 0.4, 2),
            MeshOptions {
                placement: start,
                ..MeshOptions::default()
            },
        )
        .unwrap();

    // Compare slide vs no-slide from the exact same drop.
    scene
        .set_collision_rule(
            ramp,
            sphere,
            CollisionRule {
                mode: CollisionMode::Rigid,
                rigid_slide: false,
                ..CollisionRule::default()
            },
        )
        .unwrap();
    let drop_target = translate([0.0, 0.0, -2.0]);
    let rep_no_slide = scene
        .move_with_collision(sphere, drop_target, true)
        .unwrap();
    assert_eq!(scene.intersects(ramp, sphere), Ok(false));
    assert!(rep_no_slide.final_placement.translation[0].abs() < 1e-6);

    scene.undo().unwrap();
    assert_eq!(scene.placement(sphere).unwrap(), start);

    scene
        .set_collision_rule(
            ramp,
            sphere,
            CollisionRule {
                mode: CollisionMode::Rigid,
                rigid_slide: true,
                ..CollisionRule::default()
            },
        )
        .unwrap();
    let rep_slide = scene
        .move_with_collision(sphere, drop_target, true)
        .unwrap();
    assert_eq!(scene.intersects(ramp, sphere), Ok(false));
    assert_eq!(rep_slide.active_pairs.len(), 1);
    assert!(rep_slide.active_pairs[0].contact_blocked);
    assert!(rep_slide.active_pairs[0].slidden);
    assert!(
        rep_slide.final_placement.translation[0] > 0.4,
        "expected significant +X tangential slide down 45 deg ramp, got {:?}",
        rep_slide.final_placement.translation
    );
    assert!(
        rep_slide.final_placement.translation[2]
            < rep_no_slide.final_placement.translation[2] - 0.3,
        "expected sphere to slide further down slope in Z"
    );
}

#[test]
fn cut_on_release_live_step_backends_roles_and_undo() {
    // 1. OnRelease with Manifold + commit_collision_interaction + undo/redo.
    {
        let mut scene = Scene::new(SceneOptions::default());
        let block = scene
            .add_mesh(box_mesh([-1.0; 3], [1.0; 3]), MeshOptions::default())
            .unwrap();
        let cutter = scene
            .add_mesh(
                box_mesh([-0.5; 3], [0.5; 3]),
                MeshOptions {
                    placement: translate([3.0, 0.0, 0.0]),
                    ..MeshOptions::default()
                },
            )
            .unwrap();

        let tris_before = scene.mesh(block).unwrap().triangle_count();
        scene
            .set_collision_rule(
                block,
                cutter,
                CollisionRule {
                    mode: CollisionMode::Cut,
                    role: CollisionRole::MovingIsCutter,
                    cut_method: BooleanMethod::Manifold,
                    cut_trigger: CutTrigger::OnRelease,
                    ..CollisionRule::default()
                },
            )
            .unwrap();

        // Non-release drag moves into block and previews without carving.
        let rep_drag = scene
            .move_with_collision(cutter, translate([0.0, 0.0, 0.0]), false)
            .unwrap();
        assert_eq!(rep_drag.cuts_executed, 0);
        assert_eq!(scene.mesh(block).unwrap().triangle_count(), tris_before);
        assert_eq!(rep_drag.active_pairs.len(), 1);
        assert!(rep_drag.active_pairs[0].preview_only);
        assert!(!rep_drag.active_pairs[0].cut_executed);

        // Commit executes subtraction cleanly.
        let rep_commit = scene.commit_collision_interaction(cutter).unwrap();
        assert_eq!(rep_commit.cuts_executed, 1);
        assert!(rep_commit.active_pairs.iter().any(|p| p.cut_executed));
        assert!(scene.mesh(block).unwrap().triangle_count() > tris_before);

        // Center [0, 0, 0] is now a hollow cavity whose nearest wall lies on [-0.5, 0.5]^3 face (dist ~ 0.5).
        let center_hit = scene
            .closest_point_on(block, [0.0, 0.0, 0.0], &SceneClosestOptions::default())
            .unwrap()
            .unwrap();
        assert!((center_hit.distance - 0.5).abs() < 1e-4);

        // Undo restores pre-cut solid cleanly.
        assert_eq!(scene.undo(), Ok(true));
        assert_eq!(scene.mesh(block).unwrap().triangle_count(), tris_before);
        assert_eq!(scene.redo(), Ok(true));
        assert!(scene.mesh(block).unwrap().triangle_count() > tris_before);
    }

    // 2. LiveStep with MovingIsTarget role.
    {
        let mut scene = Scene::new(SceneOptions::default());
        let target = scene
            .add_mesh(
                box_mesh([-1.0; 3], [1.0; 3]),
                MeshOptions {
                    placement: translate([3.0, 0.0, 0.0]),
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        let static_cutter = scene
            .add_mesh(
                box_mesh([-0.5, -0.5, -2.0], [0.5, 0.5, 2.0]),
                MeshOptions::default(),
            )
            .unwrap();
        let cutter_tris = scene.mesh(static_cutter).unwrap().triangle_count();
        let target_tris = scene.mesh(target).unwrap().triangle_count();

        scene
            .set_collision_rule(
                target,
                static_cutter,
                CollisionRule {
                    mode: CollisionMode::Cut,
                    role: CollisionRole::MovingIsTarget,
                    cut_method: BooleanMethod::Manifold,
                    cut_trigger: CutTrigger::LiveStep,
                    ..CollisionRule::default()
                },
            )
            .unwrap();

        let rep = scene
            .move_with_collision(target, translate([0.0, 0.0, 0.0]), false)
            .unwrap();
        assert_eq!(rep.cuts_executed, 1);
        assert_eq!(
            scene.mesh(static_cutter).unwrap().triangle_count(),
            cutter_tris
        );
        assert!(scene.mesh(target).unwrap().triangle_count() > target_tris);
    }

    // 3. Volumetric Cut methods: BvhSdf & FhEdt with undo restore.
    for method in [BooleanMethod::BvhSdf, BooleanMethod::FhEdt] {
        let mut scene = Scene::new(SceneOptions::default());
        let block = scene
            .add_mesh(box_mesh([-1.0; 3], [1.0; 3]), MeshOptions::default())
            .unwrap();
        let cutter = scene
            .add_mesh(
                box_mesh([-0.4; 3], [0.4; 3]),
                MeshOptions {
                    placement: translate([3.0, 0.0, 0.0]),
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        let pre_tris = scene.mesh(block).unwrap().triangle_count();
        let pre_pos = scene.mesh(block).unwrap().positions().to_vec();

        scene
            .set_collision_rule(
                block,
                cutter,
                CollisionRule {
                    mode: CollisionMode::Cut,
                    role: CollisionRole::MovingIsCutter,
                    cut_method: method,
                    cut_trigger: CutTrigger::OnRelease,
                    ..CollisionRule::default()
                },
            )
            .unwrap();

        let rep = scene
            .move_with_collision(cutter, translate([0.0, 0.0, 0.0]), true)
            .unwrap();
        assert_eq!(rep.cuts_executed, 1, "failed cut for {method:?}");
        assert!(scene.mesh(block).unwrap().triangle_count() > pre_tris);
        let center_hit = scene
            .closest_point_on(block, [0.0, 0.0, 0.0], &SceneClosestOptions::default())
            .unwrap()
            .unwrap();
        assert!(
            center_hit.distance > 0.25,
            "cavity missed for {method:?}: {}",
            center_hit.distance
        );

        assert_eq!(scene.undo(), Ok(true));
        let m = scene.mesh(block).unwrap();
        assert_eq!(m.triangle_count(), pre_tris);
        for (idx, &orig) in pre_pos.iter().enumerate() {
            let vid = idx as u32;
            assert!(m.is_vertex_alive(vid));
            assert_eq!(m.position(vid), Some(orig));
        }
    }
}

#[test]
fn soft_deform_elastic_and_plastic_across_all_solvers_and_backends() {
    let solvers = [
        SoftDeformSolver::SdfLaplacian,
        SoftDeformSolver::Kelvinlet,
        SoftDeformSolver::Xpbd,
    ];
    let backends = [
        CollisionDistanceBackend::BvhSdf,
        CollisionDistanceBackend::FhEdt,
    ];

    for solver in solvers {
        for backend in backends {
            // Elastic: indent on contact, smoothly recover to rest geometry on separation.
            {
                let mut scene = Scene::new(SceneOptions::default());
                let block = scene
                    .add_mesh(
                        subdivided_box([-1.0, -1.0, 0.0], [1.0, 1.0, 1.0], 8),
                        MeshOptions::default(),
                    )
                    .unwrap();
                let sphere = scene
                    .add_mesh(
                        icosphere([0.0, 0.0, 0.0], 0.55, 2),
                        MeshOptions {
                            placement: translate([0.0, 0.0, 2.2]),
                            ..MeshOptions::default()
                        },
                    )
                    .unwrap();

                let rest_pos = scene.mesh(block).unwrap().positions().to_vec();
                let indenter_rest = scene.mesh(sphere).unwrap().positions().to_vec();

                scene
                    .set_collision_rule(
                        block,
                        sphere,
                        CollisionRule {
                            mode: CollisionMode::SoftDeform,
                            role: CollisionRole::MovingIsCutter,
                            distance_backend: backend,
                            soft_solver: solver,
                            material_memory: MaterialMemory::Elastic,
                            compliance: 0.08,
                            stiffness: 0.4,
                            recovery_rate: 1.0,
                            self_collision_guard: true,
                            ..CollisionRule::default()
                        },
                    )
                    .unwrap();

                // Push sphere down so bottom cap penetrates block top face (z=1.0) down to z=0.65.
                let push_rep = scene
                    .move_with_collision(sphere, translate([0.0, 0.0, 1.20]), false)
                    .unwrap();
                assert!(
                    push_rep.soft_vertices_deformed > 0,
                    "{solver:?}/{backend:?}: expected deformed vertices"
                );
                assert_eq!(
                    scene.mesh(sphere).unwrap().positions(),
                    indenter_rest.as_slice()
                );

                // Verify center top area indented below z = 0.85.
                let min_top_z = scene
                    .mesh(block)
                    .unwrap()
                    .positions()
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| {
                        rest_pos[*i][2] > 0.99
                            && rest_pos[*i][0].abs() < 0.3
                            && rest_pos[*i][1].abs() < 0.3
                    })
                    .map(|(_, p)| p[2])
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    min_top_z < 0.85,
                    "{solver:?}/{backend:?}: expected top center dent < 0.85, got {min_top_z}"
                );

                // Separate indenter: elastic memory recovers rest geometry cleanly.
                let sep_rep = scene
                    .move_with_collision(sphere, translate([0.0, 0.0, 2.5]), true)
                    .unwrap();
                assert!(
                    sep_rep.elastic_recovery_count > 0,
                    "{solver:?}/{backend:?}: expected recovery updates"
                );
                let drift = max_vertex_drift(scene.mesh(block).unwrap().positions(), &rest_pos);
                assert!(
                    drift < 1e-4,
                    "{solver:?}/{backend:?}: expected elastic recovery to rest, max drift {drift}"
                );
            }

            // Plastic: retains permanent dent after separation and reverts on undo().
            {
                let mut scene = Scene::new(SceneOptions::default());
                let block = scene
                    .add_mesh(
                        subdivided_box([-1.0, -1.0, 0.0], [1.0, 1.0, 1.0], 8),
                        MeshOptions::default(),
                    )
                    .unwrap();
                let sphere = scene
                    .add_mesh(
                        icosphere([0.0, 0.0, 0.0], 0.55, 2),
                        MeshOptions {
                            placement: translate([0.0, 0.0, 2.2]),
                            ..MeshOptions::default()
                        },
                    )
                    .unwrap();

                let rest_pos = scene.mesh(block).unwrap().positions().to_vec();
                scene
                    .set_collision_rule(
                        block,
                        sphere,
                        CollisionRule {
                            mode: CollisionMode::SoftDeform,
                            role: CollisionRole::MovingIsCutter,
                            distance_backend: backend,
                            soft_solver: solver,
                            material_memory: MaterialMemory::Plastic,
                            compliance: 0.08,
                            stiffness: 0.4,
                            self_collision_guard: true,
                            ..CollisionRule::default()
                        },
                    )
                    .unwrap();

                let push_rep = scene
                    .move_with_collision(sphere, translate([0.0, 0.0, 1.20]), true)
                    .unwrap();
                assert!(push_rep.soft_vertices_deformed > 0);
                let after_push_pos = scene.mesh(block).unwrap().positions().to_vec();

                // Separate: no recovery occurs; dent remains permanently.
                let sep_rep = scene
                    .move_with_collision(sphere, translate([0.0, 0.0, 2.5]), true)
                    .unwrap();
                assert_eq!(sep_rep.elastic_recovery_count, 0);
                let drift_after_sep =
                    max_vertex_drift(scene.mesh(block).unwrap().positions(), &after_push_pos);
                assert!(drift_after_sep < 1e-6);
                let drift_from_rest =
                    max_vertex_drift(scene.mesh(block).unwrap().positions(), &rest_pos);
                assert!(
                    drift_from_rest > 0.15,
                    "{solver:?}/{backend:?}: expected permanent plastic dent > 0.15, got {drift_from_rest}"
                );

                // Undo separation and undo indentation: reverts to pre-dent rest positions.
                assert_eq!(scene.undo(), Ok(true));
                assert_eq!(scene.undo(), Ok(true));
                let drift_undone =
                    max_vertex_drift(scene.mesh(block).unwrap().positions(), &rest_pos);
                assert!(
                    drift_undone < 1e-6,
                    "{solver:?}/{backend:?}: expected undo to restore rest, got {drift_undone}"
                );
            }
        }
    }
}

#[test]
fn soft_deform_partial_elastic_recovery_and_symmetric_and_target_roles() {
    // 1. Gradual multi-step elastic recovery with recovery_rate < 1.0.
    {
        let mut scene = Scene::new(SceneOptions::default());
        let block = scene
            .add_mesh(
                subdivided_box([-1.0, -1.0, 0.0], [1.0, 1.0, 1.0], 6),
                MeshOptions::default(),
            )
            .unwrap();
        let indenter = scene
            .add_mesh(
                icosphere([0.0, 0.0, 0.0], 0.55, 2),
                MeshOptions {
                    placement: translate([0.0, 0.0, 2.2]),
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        let rest_pos = scene.mesh(block).unwrap().positions().to_vec();

        scene
            .set_collision_rule(
                block,
                indenter,
                CollisionRule {
                    mode: CollisionMode::SoftDeform,
                    role: CollisionRole::MovingIsCutter,
                    soft_solver: SoftDeformSolver::Xpbd,
                    material_memory: MaterialMemory::Elastic,
                    recovery_rate: 0.5,
                    ..CollisionRule::default()
                },
            )
            .unwrap();

        scene
            .move_with_collision(indenter, translate([0.0, 0.0, 1.2]), false)
            .unwrap();
        let dented_drift = max_vertex_drift(scene.mesh(block).unwrap().positions(), &rest_pos);
        assert!(dented_drift > 0.15);

        scene
            .move_with_collision(indenter, translate([0.0, 0.0, 2.5]), false)
            .unwrap();
        let half_drift = max_vertex_drift(scene.mesh(block).unwrap().positions(), &rest_pos);
        assert!(
            half_drift > 0.05 && half_drift < dented_drift * 0.65,
            "expected partial 50% recovery, got dented={dented_drift}, half={half_drift}"
        );

        for _ in 0..12 {
            scene.commit_collision_interaction(indenter).unwrap();
        }
        let settled_drift = max_vertex_drift(scene.mesh(block).unwrap().positions(), &rest_pos);
        assert!(settled_drift < 1e-3, "settled_drift={settled_drift}");
    }

    // 2. MovingIsTarget and Symmetric roles.
    {
        let mut scene = Scene::new(SceneOptions::default());
        let soft_ball = scene
            .add_mesh(
                icosphere([0.0, 0.0, 0.0], 0.6, 2),
                MeshOptions {
                    placement: translate([0.0, 0.0, 2.0]),
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        let rigid_floor = scene
            .add_mesh(
                box_mesh([-2.0, -2.0, -1.0], [2.0, 2.0, 0.0]),
                MeshOptions::default(),
            )
            .unwrap();
        let floor_rest = scene.mesh(rigid_floor).unwrap().positions().to_vec();
        let ball_rest = scene.mesh(soft_ball).unwrap().positions().to_vec();

        scene
            .set_collision_rule(
                soft_ball,
                rigid_floor,
                CollisionRule {
                    mode: CollisionMode::SoftDeform,
                    role: CollisionRole::MovingIsTarget,
                    soft_solver: SoftDeformSolver::Kelvinlet,
                    material_memory: MaterialMemory::Elastic,
                    ..CollisionRule::default()
                },
            )
            .unwrap();

        let rep = scene
            .move_with_collision(soft_ball, translate([0.0, 0.0, 0.3]), false)
            .unwrap();
        assert!(rep.soft_vertices_deformed > 0);
        assert_eq!(
            scene.mesh(rigid_floor).unwrap().positions(),
            floor_rest.as_slice()
        );
        assert!(max_vertex_drift(scene.mesh(soft_ball).unwrap().positions(), &ball_rest) > 0.15);

        scene
            .move_with_collision(soft_ball, translate([0.0, 0.0, 2.0]), true)
            .unwrap();
        assert!(max_vertex_drift(scene.mesh(soft_ball).unwrap().positions(), &ball_rest) < 1e-4);
    }

    // 3. Symmetric mutual SoftDeform between two soft spheres.
    {
        let mut scene = Scene::new(SceneOptions::default());
        let s1 = scene
            .add_mesh(icosphere([0.0, 0.0, 0.0], 0.6, 2), MeshOptions::default())
            .unwrap();
        let s2 = scene
            .add_mesh(
                icosphere([0.0, 0.0, 0.0], 0.6, 2),
                MeshOptions {
                    placement: translate([2.0, 0.0, 0.0]),
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        let r1 = scene.mesh(s1).unwrap().positions().to_vec();
        let r2 = scene.mesh(s2).unwrap().positions().to_vec();

        scene
            .set_collision_rule(
                s1,
                s2,
                CollisionRule {
                    mode: CollisionMode::SoftDeform,
                    role: CollisionRole::Symmetric,
                    soft_solver: SoftDeformSolver::SdfLaplacian,
                    material_memory: MaterialMemory::Elastic,
                    ..CollisionRule::default()
                },
            )
            .unwrap();

        let rep = scene
            .move_with_collision(s2, translate([0.7, 0.0, 0.0]), false)
            .unwrap();
        assert!(rep.soft_vertices_deformed > 0);
        assert!(max_vertex_drift(scene.mesh(s1).unwrap().positions(), &r1) > 0.05);
        assert!(max_vertex_drift(scene.mesh(s2).unwrap().positions(), &r2) > 0.05);

        scene
            .move_with_collision(s2, translate([2.0, 0.0, 0.0]), true)
            .unwrap();
        assert!(max_vertex_drift(scene.mesh(s1).unwrap().positions(), &r1) < 1e-4);
        assert!(max_vertex_drift(scene.mesh(s2).unwrap().positions(), &r2) < 1e-4);
    }
}

#[test]
fn self_collision_guard_prevents_self_intersection_under_deep_xpbd_push() {
    let mut scene = Scene::new(SceneOptions::default());
    // Thin compliant slab: z in [0.0, 0.20]. A wide punch whose bottom plane sits below z = 0
    // drives top and bottom faces onto the same plane; the guard separates intersecting folds cleanly.
    let slab = scene
        .add_mesh(
            subdivided_box([-1.0, -1.0, 0.0], [1.0, 1.0, 0.20], 8),
            MeshOptions::default(),
        )
        .unwrap();
    let indenter = scene
        .add_mesh(
            box_mesh([-0.80, -0.80, -0.55], [0.80, 0.80, 0.55]),
            MeshOptions {
                placement: translate([0.0, 0.0, 1.5]),
                ..MeshOptions::default()
            },
        )
        .unwrap();

    scene
        .set_collision_rule(
            slab,
            indenter,
            CollisionRule {
                mode: CollisionMode::SoftDeform,
                role: CollisionRole::MovingIsCutter,
                soft_solver: SoftDeformSolver::Xpbd,
                material_memory: MaterialMemory::Plastic,
                compliance: 0.5,
                stiffness: 0.1,
                self_collision_guard: true,
                ..CollisionRule::default()
            },
        )
        .unwrap();

    // Place punch center at z = 0.50 so its span is z in [-0.05, +1.05]: interior 3x3 patches on
    // both top (0.20) and bottom (0.0) faces project onto bottom plane -0.05 and collide unless guarded.
    let rep = scene
        .move_with_collision(indenter, translate([0.0, 0.0, 0.50]), true)
        .unwrap();
    assert!(rep.soft_vertices_deformed > 0);
    assert!(
        rep.self_intersection_guard_triggered,
        "expected deep push compressing thin slab to trigger self-collision guard"
    );

    let slab_mesh = scene.mesh(slab).unwrap();
    let slab_bvh = Bvh::build(slab_mesh, &BuildOptions::default()).unwrap();
    let mut self_pairs = Vec::new();
    self_intersecting_pairs(slab_mesh, &slab_bvh, &mut self_pairs).unwrap();
    assert!(
        self_pairs.is_empty(),
        "expected self-collision guard to leave slab free of self-intersections, got {self_pairs:?}"
    );
}
