//! Queries between two meshes. (The same near the world origin, with every
//! put: `tests/scaled_pairs.rs`.)

use fast_mesh_scene::{ContactFilter, DeviationOptions, SceneError};

use crate::*;

const TOL: f64 = 6e-6;

/// A block 2 units from the lump, and one that runs into it.
fn apart() -> Shape {
    lump().moved([3.4, 0.37, -0.21])
}

fn crossing() -> Shape {
    block([0.6, -0.31, -0.42], [2.1, 0.53, 0.37])
}

#[test]
fn distances_and_their_points() {
    agree_pairs("mesh_distance", TOL, |pa, pb| {
        let mut s = scene();
        let (a, b) = (pa.add(&mut s, &lump()), pb.add(&mut s, &apart()));
        let mut out = Vec::new();
        for (x, y) in [(a, b), (b, a)] {
            let d = s.mesh_distance(x, y, f64::INFINITY).unwrap().unwrap();
            out.push(d.distance);
            out.extend(pa.model(d.point_a));
            out.extend(pa.model(d.point_b));
            out.extend([d.triangle_a, d.triangle_b].map(f64::from));
            let within = s.mesh_distance(x, y, d.distance + 1e-3).unwrap();
            let without = s.mesh_distance(x, y, d.distance - 1e-3).unwrap();
            out.extend([within.is_some(), without.is_some()].map(|b| f64::from(u8::from(b))));
        }
        out
    });
}

#[test]
fn intersections() {
    agree_pairs("intersects", 0.0, |pa, pb| {
        let mut s = scene();
        let (a, hit) = (pa.add(&mut s, &lump()), pb.add(&mut s, &crossing()));
        let far = pb.add(&mut s, &apart());
        [(a, hit), (hit, a), (a, far), (far, a), (hit, hit)]
            .map(|(x, y)| f64::from(u8::from(s.intersects(x, y).unwrap())))
            .to_vec()
    });
}

#[test]
fn contacts_and_triangles_near() {
    agree_pairs("find_contacts", TOL, |pa, pb| {
        let mut s = scene();
        let a = pa.add(&mut s, &lump());
        let (hit, far) = (pb.add(&mut s, &crossing()), pb.add(&mut s, &apart()));
        let mut out = Vec::new();
        for margin in [0.0, 0.43, 1.7] {
            for pairs in [None, Some(vec![(far, a), (a, hit)])] {
                let mut found = Vec::new();
                let filter = ContactFilter { pairs };
                s.find_contacts(margin, &filter, &mut found).unwrap();
                for c in found {
                    out.extend([c.distance, f64::from(u8::from(c.intersecting))]);
                    out.extend(c.triangles_a.into_iter().map(f64::from));
                    out.extend(c.triangles_b.into_iter().map(f64::from));
                }
                out.push(f64::NAN);
            }
            let (mut ta, mut tb) = (Vec::new(), Vec::new());
            s.triangles_near(a, far, margin, &mut ta, &mut tb).unwrap();
            s.triangles_near(hit, a, margin, &mut ta, &mut tb).unwrap();
            out.extend(ta.into_iter().chain(tb).map(f64::from));
        }
        out
    });
}

#[test]
fn self_intersections() {
    // Two blocks in one mesh that run into each other.
    let mut shape = block([-1.0, -0.8, -0.7], [0.4, 0.6, 0.5]);
    let other = block([0.1, 0.1, 0.1], [1.2, 1.1, 0.9]);
    let n = shape.points.len() as u32;
    shape.points.extend(other.points);
    shape
        .triangles
        .extend(other.triangles.iter().map(|t| t.map(|v| v + n)));
    agree("self_intersecting_triangles", 0.0, |put| {
        let mut s = scene();
        let id = put.add(&mut s, &shape);
        let mut out = Vec::new();
        s.self_intersecting_triangles(id, &mut out).unwrap();
        out.into_iter().map(f64::from).collect()
    });
}

#[test]
fn deviations() {
    for signed in [false, true] {
        for max_distance in [f32::INFINITY, 0.6] {
            let options = DeviationOptions {
                signed,
                max_distance,
            };
            agree_pairs("mesh_deviation", TOL, |pa, pb| {
                let mut s = scene();
                let (a, b) = (pa.add(&mut s, &lump()), pb.add(&mut s, &crossing()));
                let far = pb.add(&mut s, &apart());
                let mut out = Vec::new();
                for (x, y) in [(a, b), (b, a), (a, far)] {
                    let mut values = Vec::new();
                    let r = s.mesh_deviation(x, y, &options, &mut values).unwrap();
                    out.extend(f64s(&values));
                    out.extend([r.min, r.max, r.mean, r.rms, r.p95].map(f64::from));
                    out.push(r.sample_count as f64);
                }
                out
            });
        }
    }
}

/// The map from one mesh's local space to another's: documented as for rigid
/// pairs only.
#[test]
fn relative_transforms() {
    agree_pairs("relative_transform", 1e-9, |pa, pb| {
        let mut s = scene();
        let (a, b) = (pa.add(&mut s, &lump()), pb.add(&mut s, &apart()));
        let scaled = pa.scale != [1.0; 3] || pb.scale != [1.0; 3];
        let xf = match s.relative_transform(a, b) {
            Ok(xf) => xf,
            Err(e) => {
                assert!(scaled && matches!(e, SceneError::InvalidInput(_)), "{e:?}");
                // Nothing to compare: what the plain ones give.
                return vec![0.0; 24];
            }
        };
        assert!(!scaled);
        let mut out = Vec::new();
        for i in 0..8u32 {
            let p = [i & 1, i >> 1 & 1, i >> 2 & 1].map(f64::from);
            let through = put_back(&s, a, xf.apply(p.map(|c| c as f32)), pa);
            let direct = put_back(&s, b, p, pb);
            out.extend(sub(through, direct));
        }
        out
    });
}

fn put_back(s: &Scene, id: MeshId, p: V3, put: Put) -> V3 {
    put.model(s.local_to_world(id, p).unwrap())
}
