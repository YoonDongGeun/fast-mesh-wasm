//! Moves with collision responses, in every mode.

use fast_mesh_scene::{
    CollisionDistanceBackend, CollisionMode, CollisionRole, CollisionRule, CollisionStepReport,
    MaterialMemory, MeshId, SoftDeformSolver,
};

use crate::*;

/// Mesh `id` moved by `d` in the world.
fn moved(s: &Scene, id: MeshId, d: V3) -> Placement {
    let p = s.placement(id).unwrap();
    Placement {
        translation: add(p.translation, d),
        ..p
    }
}

fn numbers(r: &CollisionStepReport) -> Vec<f64> {
    let mut out = vec![r.min_clearance, r.cuts_executed as f64];
    for p in &r.active_pairs {
        out.push(p.min_distance);
        out.extend(
            [
                p.intersecting,
                p.contact_blocked,
                p.slidden,
                p.cut_executed,
                p.preview_only,
            ]
            .map(|b| f64::from(u8::from(b))),
        );
    }
    out
}

fn wall() -> Shape {
    block([1.6, -1.4, -1.5], [2.2, 1.3, 1.6])
}

#[test]
fn passing_through() {
    let rule = CollisionRule {
        contact_margin: 0.2,
        ..CollisionRule::default()
    };
    agree_pairs("pass through", 2e-5, |pa, pb| {
        let mut s = scene();
        let (a, b) = (pa.add(&mut s, &lump()), pb.add(&mut s, &wall()));
        s.set_collision_rule(a, b, rule).unwrap();
        let mut out = Vec::new();
        for d in [[-0.3, 0.0, 0.0], [0.5, 0.1, 0.0], [0.3, 0.0, 0.1]] {
            let to = moved(&s, a, d);
            let r = s.move_with_collision(a, to, false).unwrap();
            out.extend(numbers(&r));
            out.extend(solid(&s, a, pa));
        }
        out
    });
}

#[test]
fn rigid() {
    for rigid_slide in [false, true] {
        let rule = CollisionRule {
            mode: CollisionMode::Rigid,
            rigid_slide,
            ..CollisionRule::default()
        };
        // It stops within its skin of 1e-4 of the meshes' size, wherever the
        // search for it ends.
        agree_pairs("rigid", 3e-3, |pa, pb| {
            let mut s = scene();
            let (a, b) = (pa.add(&mut s, &lump()), pb.add(&mut s, &wall()));
            s.set_collision_rule(a, b, rule).unwrap();
            let mut out = Vec::new();
            for (x, d) in [(a, [1.1, 0.31, 0.2]), (b, [-0.6, 0.1, 0.0])] {
                let to = moved(&s, x, d);
                let r = s.move_with_collision(x, to, true).unwrap();
                out.extend(numbers(&r));
                let put = if x == a { pa } else { pb };
                out.extend(&solid(&s, x, put)[2..8]);
                assert!(!s.intersects(a, b).unwrap());
            }
            s.undo().unwrap();
            out.extend(solid(&s, b, pb));
            s.undo().unwrap();
            out.extend(solid(&s, a, pa));
            out
        });
    }
}

#[test]
fn cutting() {
    for role in [CollisionRole::MovingIsCutter, CollisionRole::MovingIsTarget] {
        let rule = CollisionRule {
            mode: CollisionMode::Cut,
            role,
            ..CollisionRule::default()
        };
        agree_pairs("cut", 2e-5, |pa, pb| {
            let mut s = scene();
            let (a, b) = (pa.add(&mut s, &lump()), pb.add(&mut s, &wall()));
            s.set_collision_rule(a, b, rule).unwrap();
            let to = moved(&s, a, [0.9, 0.1, 0.2]);
            let preview = s.move_with_collision(a, to, false).unwrap();
            let mut out = numbers(&preview)[1..].to_vec();
            let done = s.commit_collision_interaction(a).unwrap();
            assert_eq!(done.cuts_executed, 1);
            out.extend(solid(&s, a, pa));
            out.extend(solid(&s, b, pb));
            s.undo().unwrap();
            out.extend(solid(&s, a, pa));
            out.extend(solid(&s, b, pb));
            out
        });
    }
}

/// A block fine enough to be dented.
fn pad() -> Shape {
    let n = 14;
    let mut shape = sheet(n);
    let count = shape.points.len() as u32;
    // Its top and, 0.8 below, its bottom the other way round.
    let below: Vec<V3> = shape
        .points
        .iter()
        .map(|p| [p[0], p[1], p[2] - 0.8])
        .collect();
    shape.points.extend(below);
    let top = shape.triangles.clone();
    shape.triangles.extend(
        top.iter()
            .map(|t| [t[0] + count, t[2] + count, t[1] + count]),
    );
    let at = |i: u32, j: u32| j * (n + 1) + i;
    let mut rim: Vec<u32> = (0..n).map(|i| at(i, 0)).collect();
    rim.extend((0..n).map(|j| at(n, j)));
    rim.extend((0..n).map(|i| at(n - i, n)));
    rim.extend((0..n).map(|j| at(0, n - j)));
    for (k, &a) in rim.iter().enumerate() {
        let b = rim[(k + 1) % rim.len()];
        shape.triangles.push([a, a + count, b + count]);
        shape.triangles.push([a, b + count, b]);
    }
    shape
}

fn soft(
    backend: CollisionDistanceBackend,
    solver: SoftDeformSolver,
    role: CollisionRole,
    margin: f64,
) -> CollisionRule {
    CollisionRule {
        mode: CollisionMode::SoftDeform,
        role,
        distance_backend: backend,
        soft_solver: solver,
        material_memory: MaterialMemory::Elastic,
        contact_margin: margin,
        recovery_rate: 0.6,
        ..CollisionRule::default()
    }
}

fn dented(pa: Put, pb: Put, rule: CollisionRule) -> Vec<f64> {
    let mut s = scene();
    let a = pa.add(&mut s, &lump().moved([0.1, 0.05, 1.45]));
    let b = pb.add(&mut s, &pad());
    s.set_collision_rule(a, b, rule).unwrap();
    let mut out = Vec::new();
    for d in [[0.0, 0.0, -0.3], [0.1, 0.0, -0.1], [0.0, 0.0, 0.9]] {
        let to = moved(&s, a, d);
        let r = s.move_with_collision(a, to, false).unwrap();
        out.push(r.soft_vertices_deformed as f64);
        out.extend(positions(&s, b, pb));
        out.extend(positions(&s, a, pa));
    }
    s.undo().unwrap();
    out.extend(positions(&s, b, pb));
    out
}

#[test]
fn soft_by_the_bvh() {
    for solver in [
        SoftDeformSolver::SdfLaplacian,
        SoftDeformSolver::Kelvinlet,
        SoftDeformSolver::Xpbd,
    ] {
        for (role, margin) in [
            (CollisionRole::MovingIsCutter, 0.0),
            (CollisionRole::MovingIsTarget, 0.02),
            (CollisionRole::Symmetric, 0.02),
        ] {
            let rule = soft(CollisionDistanceBackend::BvhSdf, solver, role, margin);
            agree_few_pairs(&format!("soft {solver:?} {role:?}"), 2e-4, |pa, pb| {
                dented(pa, pb, rule)
            });
        }
    }
}

/// On a grid of 32 voxels along the indenter's own axes: about the same.
#[test]
fn soft_by_a_grid() {
    let rule = soft(
        CollisionDistanceBackend::FhEdt,
        SoftDeformSolver::SdfLaplacian,
        CollisionRole::MovingIsCutter,
        0.0,
    );
    agree_few_pairs("soft on a grid", 0.12, |pa, pb| {
        dented(pa, pb, rule)
            .into_iter()
            // Not how many vertices.
            .filter(|v| v.abs() < 10.0)
            .collect()
    });
}
