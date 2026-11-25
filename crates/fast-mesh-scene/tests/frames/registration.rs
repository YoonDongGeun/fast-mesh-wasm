//! Registrations and alignments. (Symmetry planes: `measures.rs`.)

use fast_mesh_ops::inspect::FittedPrimitive;
use fast_mesh_scene::{
    CoarseAlignMethod, LandmarkPair, MeshId, RegisterOptions, RegistrationOptions,
};

use crate::*;

/// A small turn and a shift about the model's zero.
fn nudge(p: V3) -> V3 {
    let r = turn(0.07, -0.05);
    let q: V3 = std::array::from_fn(|i| dot(r[i], p));
    add(q, [0.06, -0.04, 0.05])
}

fn nudged() -> Shape {
    let mut shape = lump();
    for p in &mut shape.points {
        *p = nudge(*p);
    }
    shape
}

fn history(s: &mut Scene, id: MeshId, put: Put) -> Vec<f64> {
    let mut out = positions(s, id, put);
    s.undo().unwrap();
    out.extend(positions(s, id, put));
    s.redo().unwrap();
    out.extend(positions(s, id, put));
    out
}

#[test]
fn icp() {
    agree_pairs("register", 2e-4, |pa, pb| {
        let mut s = scene();
        let (moving, fixed) = (pa.add(&mut s, &nudged()), pb.add(&mut s, &lump()));
        let r = s
            .register(moving, fixed, &RegisterOptions::default())
            .unwrap();
        let mut out = vec![r.rmse, f64::from(u8::from(r.applied)), r.max_rmse];
        out.extend(history(&mut s, moving, pa));
        out
    });
}

#[test]
fn multistage() {
    let options = RegistrationOptions {
        coarse_method: CoarseAlignMethod::None,
        ..RegistrationOptions::default()
    };
    agree_pairs("register_multistage", 1e-3, |pa, pb| {
        let mut s = scene();
        let (moving, fixed) = (pa.add(&mut s, &nudged()), pb.add(&mut s, &lump()));
        let before = positions(&s, moving, pa);
        let r = s.register_multistage(moving, fixed, &options).unwrap();
        let mut out = history(&mut s, moving, pa);
        // The matrix it reports takes the one to the other.
        let m = r.transform;
        for p in before.chunks(3).take(20) {
            let w = pa.world([p[0], p[1], p[2]]);
            out.extend(pa.model(std::array::from_fn(|i| {
                m[i] * w[0] + m[4 + i] * w[1] + m[8 + i] * w[2] + m[12 + i]
            })));
        }
        out
    });
}

#[test]
fn landmarks() {
    let points = [
        [1.0, 0.1, 0.2],
        [-0.8, 0.5, -0.3],
        [0.1, -0.7, 0.6],
        [0.2, 0.3, -0.9],
    ];
    for allow_scaling in [false, true] {
        agree_pairs("align_by_landmarks", 2e-5, |pa, pb| {
            let mut s = scene();
            let (moving, fixed) = (pa.add(&mut s, &nudged()), pb.add(&mut s, &lump()));
            let grown = |p: V3| {
                if allow_scaling { p.map(|c| 1.2 * c) } else { p }
            };
            let pairs: Vec<LandmarkPair> = points
                .iter()
                .map(|&p| LandmarkPair {
                    source: pa.world(nudge(p)),
                    target: pb.world(grown(p)),
                })
                .collect();
            let r = s
                .align_by_landmarks(moving, fixed, &pairs, allow_scaling)
                .unwrap();
            let mut out = vec![r.rmse, r.scale];
            out.extend(history(&mut s, moving, pa));
            out
        });
    }
}

#[test]
fn fitted_primitives() {
    agree("align_to_fitted_primitive", 2e-5, |put| {
        let mut s = scene();
        let id = put.add(&mut s, &lump());
        let plane = FittedPrimitive::Plane {
            origin: put.world([0.1, 0.2, -0.9]),
            normal: unit([0.1, -0.2, -0.95]),
        };
        let to = put.world([0.5, -0.3, 0.2]);
        s.align_to_fitted_primitive(id, &plane, [0.0, 0.0, -1.0], to)
            .unwrap();
        history(&mut s, id, put)
    });
}
