//! Fields on a mesh's vertices.

use fast_mesh_scene::{
    CurvatureField, DistanceField, FieldBudget, FieldId, SignMode, StepState, ThicknessField,
};

use crate::*;

const TOL: f64 = 8e-6;

fn apart() -> Shape {
    lump().moved([3.4, 0.37, -0.21])
}

fn crossing() -> Shape {
    block([0.6, -0.31, -0.42], [2.1, 0.53, 0.37])
}

fn values(s: &Scene, id: FieldId) -> Vec<f64> {
    f64s(s.field_values(id).unwrap())
}

#[test]
fn distances() {
    for sign in [SignMode::Pseudonormal, SignMode::Winding] {
        for max in [f64::INFINITY, 0.7] {
            agree_pairs("distance field", TOL, |pa, pb| {
                let mut s = scene();
                let a = pa.add(&mut s, &lump());
                let mut out = Vec::new();
                for (name, shape) in [("hit", crossing()), ("far", apart())] {
                    let b = pb.add(&mut s, &shape);
                    let f = s.add_distance_field(DistanceField::new(name, a, b, max, sign));
                    let g = s.add_distance_field(DistanceField::new("back", b, a, max, sign));
                    s.update_fields().unwrap();
                    out.extend(values(&s, f.unwrap()));
                    out.extend(values(&s, g.unwrap()));
                }
                out
            });
        }
    }
}

#[test]
fn distances_to_several_and_which_is_nearest() {
    agree_pairs("field_nearest", TOL, |pa, pb| {
        let mut s = scene();
        let a = pa.add(&mut s, &lump());
        // Each target is put in another way.
        let to = vec![
            pb.add(&mut s, &crossing()),
            pa.add(&mut s, &lump().moved([-2.9, -0.4, 0.3])),
            Put { at: pb.at, ..PLAIN }.add(&mut s, &lump().moved([0.2, 2.8, 0.1])),
            pb.add(&mut s, &lump().moved([0.3, -0.2, 2.9])),
        ];
        let f = s
            .add_distance_field(DistanceField {
                name: "near".into(),
                on: a,
                to,
                max_distance: 2.0,
                sign: SignMode::Auto,
            })
            .unwrap();
        let mut out = values(&s, f);
        let nearest = s.field_nearest(f).unwrap();
        out.extend(nearest.iter().map(|&n| f64::from(n)));
        out
    });
}

/// A field of meshes that move, are edited and go back through the history:
/// what a new one gives, all the same.
#[test]
fn distances_that_follow() {
    agree_pairs("distance field after moves", TOL, |pa, pb| {
        let mut s = scene();
        let (a, b) = (pa.add(&mut s, &lump()), pb.add(&mut s, &crossing()));
        let f = s
            .add_distance_field(DistanceField::new("d", a, b, 0.8, SignMode::Auto))
            .unwrap();
        let mut out = Vec::new();
        let home = s.placement(b).unwrap();
        let mut there = home;
        for (k, d) in [0.23, -0.17, 0.4].into_iter().enumerate() {
            there.translation[k] += d;
        }
        s.set_placement(b, there).unwrap();
        out.extend(values(&s, f));
        // A vertex of the lump, out by 0.3 in the world.
        let mut p = s.mesh(a).unwrap().position(40).unwrap().map(f64::from);
        p = s.local_to_world(a, p).unwrap();
        p[1] += 0.3;
        let q = s.world_to_local(a, p).unwrap().map(|c| c as f32);
        s.edit(a, |tx| tx.set_positions(&[40], &[q])).unwrap();
        out.extend(values(&s, f));
        for _ in 0..2 {
            s.undo().unwrap();
            out.extend(values(&s, f));
        }
        for _ in 0..2 {
            s.redo().unwrap();
            out.extend(values(&s, f));
        }
        out
    });
}

#[test]
fn distances_in_steps() {
    agree_pairs("field_step", TOL, |pa, pb| {
        let mut s = scene();
        let (a, b) = (pa.add(&mut s, &lump()), pb.add(&mut s, &crossing()));
        let f = s
            .add_distance_field_deferred(DistanceField::new("d", a, b, 0.8, SignMode::Auto))
            .unwrap();
        while s.field_step_field(f, FieldBudget::Points(30)).unwrap() == StepState::Pending {}
        values(&s, f)
    });
}

/// After a drag with the interaction tier on, and its end.
#[test]
fn distances_through_an_interaction() {
    agree_pairs("field interaction", TOL, |pa, pb| {
        let mut s = scene();
        let (a, b) = (pa.add(&mut s, &lump()), pb.add(&mut s, &crossing()));
        let f = s
            .add_distance_field(DistanceField::new("d", a, b, 0.8, SignMode::Auto))
            .unwrap();
        s.set_field_interaction(Some(40)).unwrap();
        let mut there = s.placement(b).unwrap();
        for d in [0.11, 0.07, -0.05] {
            there.translation[0] += d;
            there.translation[2] -= d;
            s.set_placement(b, there).unwrap();
        }
        s.set_field_interaction(None).unwrap();
        s.update_fields().unwrap();
        values(&s, f)
    });
}

#[test]
fn thicknesses() {
    for max_distance in [f64::INFINITY, 1.5] {
        agree("thickness field", TOL, |put| {
            let mut s = scene();
            let on = put.add(&mut s, &lump());
            let name = "t".into();
            let f = s
                .add_thickness_field(ThicknessField {
                    name,
                    on,
                    max_distance,
                })
                .unwrap();
            values(&s, f)
        });
    }
}

#[test]
fn curvatures() {
    agree("curvature field", 1e-4, |put| {
        let mut s = scene();
        let on = put.add(&mut s, &lump());
        let name = "c".into();
        let f = s.add_curvature_field(CurvatureField { name, on }).unwrap();
        let mut out = values(&s, f);
        // A scale comes and goes.
        let p = s.placement(on).unwrap();
        let twice = fast_mesh_scene::Placement {
            scale: p.scale.map(|c| 2.0 * c),
            ..p
        };
        s.set_placement(on, twice).unwrap();
        out.extend(values(&s, f));
        s.undo().unwrap();
        out.extend(values(&s, f));
        out
    });
}
