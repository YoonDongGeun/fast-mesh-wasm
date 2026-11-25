//! Rays and nearest points.

use fast_mesh_query::Cull;
use fast_mesh_scene::{MeshId, Scene, SceneClosestOptions, SceneRayOptions};

use crate::*;

const TOL: f64 = 6e-6;

/// Points all around the shapes, some inside them.
fn around() -> Vec<V3> {
    let mut out = Vec::new();
    for i in 0..60 {
        let (a, b) = (f64::from(i) * 2.399_963, f64::from(i) / 59.0 * 2.0 - 1.0);
        let r = [0.31, 1.7, 3.1, 0.77][i as usize % 4];
        let s = (1.0 - b * b).sqrt();
        out.push([
            r * s * a.cos() + 0.013,
            r * s * a.sin() - 0.021,
            r * b + 0.017,
        ]);
    }
    out
}

#[test]
fn raycasts() {
    let options = [
        SceneRayOptions::default(),
        SceneRayOptions {
            cull: Cull::Back,
            ..SceneRayOptions::default()
        },
        SceneRayOptions {
            min_t: 1.3,
            max_t: 2.6,
            ..SceneRayOptions::default()
        },
    ];
    agree("raycast", TOL, |put| {
        let mut s = scene();
        let a = put.add(&mut s, &lump());
        put.add(&mut s, &block([1.9, -0.6, -0.5], [2.3, 0.7, 0.8]));
        let mut out = Vec::new();
        for p in around() {
            // At the lump's middle or past it.
            let to = [0.11 + 0.2 * p[1], -0.07, 0.3 * p[0]];
            for o in options {
                match s.raycast(put.world(p), sub(to, p), &o).unwrap() {
                    Some(h) => {
                        out.extend([h.t, f64::from(u8::from(h.mesh == a))]);
                        out.push(f64::from(h.hit.triangle));
                        out.extend(put.model(h.point));
                        out.extend(h.hit.barycentric.map(f64::from));
                    }
                    None => out.push(f64::NAN),
                }
            }
        }
        out
    });
}

#[test]
fn nearest_points() {
    for bound in [f64::INFINITY, 0.9] {
        let options = SceneClosestOptions {
            max_distance: bound,
        };
        agree("closest_point", TOL, |put| {
            let mut s = scene();
            let a = put.add(&mut s, &lump());
            put.add(&mut s, &block([1.9, -0.6, -0.5], [2.3, 0.7, 0.8]));
            let mut out = Vec::new();
            for p in around() {
                let w = put.world(p);
                let all = s.closest_point(w, &options).unwrap();
                let on = s.closest_point_on(a, w, &options).unwrap();
                for hit in [all, on] {
                    match hit {
                        Some(h) => {
                            out.push(h.distance);
                            out.extend(put.model(h.point));
                            out.push(f64::from(u8::from(h.mesh == a)));
                            // Where it says it is on its own mesh.
                            let local = h.hit.point.map(f64::from);
                            out.extend(put.model(s.local_to_world(h.mesh, local).unwrap()));
                            out.push(at_barycentric(&s, h.mesh, &h.hit, put));
                        }
                        None => out.push(f64::NAN),
                    }
                }
            }
            out
        });
    }
}

/// How far the hit's point is from where its barycentric coordinates are.
fn at_barycentric(s: &Scene, id: MeshId, hit: &fast_mesh_query::ClosestHit, put: Put) -> f64 {
    let mesh = s.mesh(id).unwrap();
    let tri = mesh.triangle(hit.triangle).unwrap();
    let mut p = [0.0; 3];
    for (&v, b) in tri.iter().zip(hit.barycentric) {
        let q = mesh.position(v).unwrap().map(f64::from);
        let w = put.model(s.local_to_world(id, q).unwrap());
        p = add(p, w.map(|c| c * f64::from(b)));
    }
    let local = hit.point.map(f64::from);
    norm(sub(p, put.model(s.local_to_world(id, local).unwrap())))
}

#[test]
fn local_to_world_and_back() {
    agree("world_to_local", 1e-9, |put| {
        let mut s = scene();
        let id = put.add(&mut s, &lump());
        let mut out = Vec::new();
        for p in around() {
            let l = s.world_to_local(id, put.world(p)).unwrap();
            out.extend(put.model(s.local_to_world(id, l).unwrap()));
        }
        out
    });
}
