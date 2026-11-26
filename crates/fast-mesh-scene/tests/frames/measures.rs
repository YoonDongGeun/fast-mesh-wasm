//! What is measured of a mesh as placed.

use fast_mesh_ops::heightfield::HeightfieldOptions;
use fast_mesh_ops::inspect::{FitPrimitiveOptions, FittedPrimitive, PrimitiveKind};
use fast_mesh_scene::{
    GeodesicMethod, GeodesicOptions, HeightMapOptions, OverhangOptions, QualityMetric,
};

use crate::*;

const TOL: f64 = 1e-5;

#[test]
fn overhangs() {
    // Degrees: a normal's error is the positions' over the triangles' size.
    agree("overhangs", 5e-4, |put| {
        let mut s = scene();
        let id = put.add(&mut s, &lump());
        let mut out = Vec::new();
        for build_direction in [[0.0, 0.0, 1.0], [0.3, -0.5, 0.8]] {
            let options = OverhangOptions {
                build_direction,
                critical_angle_deg: 37.0,
            };
            let mut angles = Vec::new();
            let r = s.overhangs(id, &options, &mut angles).unwrap();
            out.extend(f64s(&angles));
            out.extend([r.total_support_area, f64::from(r.max_overhang_angle_deg)]);
            out.push(r.overhang_face_count as f64);
        }
        out
    });
}

#[test]
fn areas_and_volumes() {
    agree("measure_placed", 2e-6, |put| {
        let mut s = scene();
        let mut out = Vec::new();
        for shape in [lump(), sheet(5)] {
            let id = put.add(&mut s, &shape);
            let m = s.measure_placed(id).unwrap();
            out.extend([m.area, m.volume.unwrap_or(f64::NAN)]);
        }
        out
    });
}

#[test]
fn triangle_qualities() {
    for metric in [
        QualityMetric::RadiusRatio,
        QualityMetric::MinAngleDeg,
        QualityMetric::AspectRatio,
    ] {
        agree("triangle_quality", 1e-3, |put| {
            let mut s = scene();
            let id = put.add(&mut s, &lump());
            let mut values = Vec::new();
            let r = s.triangle_quality(id, metric, 0.5, &mut values).unwrap();
            let mut out = f64s(&values);
            out.extend([r.min, r.max, r.mean].map(f64::from));
            out.push(r.sliver_triangle_count as f64);
            out
        });
    }
}

#[test]
fn geodesics_and_paths() {
    // The heat method cuts a mesh this coarse finer first, the largest
    // triangles first, so `f32` rounding decides between equal ones: its
    // distances agree to much less than their own error (3 %), not to
    // rounding, and the vertices near the bound may fall on either side.
    for (method, tol, near_bound) in [
        (GeodesicMethod::HeatMethod, 1e-2, 0.05),
        (GeodesicMethod::Dijkstra, 1e-4, 0.0),
    ] {
        agree("geodesic_distances", tol, |put| {
            let mut s = scene();
            let id = put.add(&mut s, &lump());
            let mut out = Vec::new();
            for max_distance in [None, Some(1.1)] {
                let options = GeodesicOptions {
                    method,
                    max_distance,
                    ..GeodesicOptions::default()
                };
                let mut d = Vec::new();
                s.geodesic_distances(id, &[17], &options, &mut d).unwrap();
                match max_distance {
                    Some(max) if near_bound > 0.0 => {
                        let most = f64::from(max) - near_bound;
                        out.extend(f64s(&d).iter().map(|x| x.min(most)));
                    }
                    _ => out.extend(f64s(&d)),
                }
                if max_distance.is_none() {
                    let mut path = Vec::new();
                    s.shortest_surface_path(id, &d, &[17], 97, &mut path)
                        .unwrap();
                    let ends = [path[0], path[path.len() - 1]];
                    out.extend(ends.iter().flat_map(|p| put.model(p.position)));
                    let length = path.windows(2);
                    out.push(
                        length
                            .map(|w| norm(sub(w[0].position, w[1].position)))
                            .sum(),
                    );
                }
            }
            out
        });
    }
}

/// The operators kept for a mesh's geodesics are those of the scale it has.
#[test]
fn geodesics_after_a_change_of_scale() {
    let options = GeodesicOptions::default();
    for at in [NEAR, FAR] {
        for put in some_puts_at(at) {
            let mut s = scene();
            let id = put.add(&mut s, &lump());
            let distances = |s: &Scene, id| {
                let mut d = Vec::new();
                s.geodesic_distances(id, &[17], &options, &mut d).unwrap();
                f64s(&d)
            };
            let before = distances(&s, id);
            let p = s.placement(id).unwrap();
            for scale in [[1.0; 3], [0.5, 1.0, 1.5], [3.0; 3], [1.5, 1.0, 0.5]] {
                s.set_placement(id, Placement { scale, ..p }).unwrap();
                let mut fresh = scene();
                let (mesh, _) = put.mesh(&lump());
                let placement = Placement { scale, ..p };
                let options = MeshOptions {
                    placement,
                    ..MeshOptions::default()
                };
                let new = fresh.add_mesh(mesh, options).unwrap();
                let d = off(&distances(&s, id), &distances(&fresh, new));
                assert!(d < 1e-6, "{put:?} {scale:?}: {d}");
            }
            while s.undo().unwrap() {}
            assert!(off(&distances(&s, id), &before) < 1e-6, "{put:?}");
        }
    }
}

#[test]
fn height_maps() {
    agree("height_map", TOL, |put| {
        let mut s = scene();
        let a = put.add(&mut s, &lump());
        let b = put.add(&mut s, &block([1.4, -0.6, -0.5], [1.9, 0.7, 0.8]));
        let mut out = Vec::new();
        for direction in [[0.0, 0.0, -1.0], [0.4, 0.3, -0.8]] {
            let options = HeightMapOptions {
                direction,
                resolution: 37,
                margin: 0.01,
                ..HeightMapOptions::default()
            };
            let frame = s.height_map_frame(&[a, b], &options).unwrap();
            out.extend([f64::from(frame.width), f64::from(frame.height)]);
            out.extend([frame.pixel_size, frame.range]);
            out.extend(put.model(frame.point(0.0, 0.0, 0.0)));
            let h = s.height_map(&[a, b], &frame, 0..frame.height).unwrap();
            // A hit by a hair is one for some and not for others.
            let inside: Vec<f64> = f64s(&h).into_iter().map(|v| v.max(0.0)).collect();
            out.push(inside.iter().sum::<f64>() / inside.len() as f64);
            let relief = s
                .add_heightfield(
                    &h.iter().map(|v| v.max(0.0)).collect::<Vec<_>>(),
                    frame.width,
                    frame.height,
                    &HeightfieldOptions {
                        pixel_size: [frame.pixel_size; 2],
                        ..HeightfieldOptions::default()
                    },
                    frame.relief_placement(),
                )
                .unwrap();
            out.extend(&solid(&s, relief.added, put)[2..8]);
            s.remove_mesh(relief.added).unwrap();
        }
        out
    });
}

#[test]
fn fits() {
    let options = FitPrimitiveOptions::default();
    agree("fit_primitive_world", 2e-4, |put| {
        let mut s = scene();
        let id = put.add(&mut s, &ball(1.3));
        let r = s
            .fit_primitive_world(id, None, Some(PrimitiveKind::Sphere), &options)
            .unwrap();
        let FittedPrimitive::Sphere { center, radius } = r.primitive else {
            panic!("{:?}", r.primitive);
        };
        let mut out = vec![radius];
        out.extend(put.model(center));
        out
    });
}

#[test]
fn symmetry_planes() {
    // Symmetric in x only.
    let shape = revolved(
        &[
            (0.0, -1.0),
            (0.7, -0.8),
            (1.0, -0.2),
            (0.6, 0.6),
            (0.0, 1.0),
        ],
        16,
        |p| {
            [
                1.4 * p[0],
                p[1] + 0.2 * p[1] * p[1] * p[1],
                p[2] + 0.3 * p[1],
            ]
        },
    );
    agree("detect_symmetry_plane", 1e-3, |put| {
        let mut s = scene();
        let id = put.add(&mut s, &shape);
        let r = s.detect_symmetry_plane(id).unwrap();
        let n = r.plane_normal;
        vec![n[0].abs(), n[1], n[2], dot(n, put.model(r.plane_point))]
    });
}
