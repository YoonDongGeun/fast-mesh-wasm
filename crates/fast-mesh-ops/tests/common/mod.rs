//! Shapes and surface checks shared by the integration tests.

// Not every test uses all of it.
#![allow(dead_code)]

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_ops::append::AppendSource;
use fast_mesh_ops::primitives::{BoxOptions, IcosphereOptions, cuboid, icosphere};
use fast_mesh_ops::{AnalyzeOptions, analyze};
use fast_mesh_store::Mesh;
use robust::{Coord, Coord3D, orient2d, orient3d};

pub fn moved(src: AppendSource, by: [f64; 3]) -> AppendSource {
    AppendSource {
        positions: src
            .positions
            .iter()
            .map(|p| std::array::from_fn(|k| (f64::from(p[k]) + by[k]) as f32))
            .collect(),
        triangles: src.triangles,
    }
}

pub fn merged(parts: &[AppendSource]) -> AppendSource {
    let mut out = AppendSource::default();
    for p in parts {
        let base = out.positions.len() as u32;
        out.positions.extend_from_slice(&p.positions);
        out.triangles
            .extend(p.triangles.iter().map(|t| t.map(|v| v + base)));
    }
    out
}

pub fn sphere(radius: f64, subdivisions: u32) -> AppendSource {
    icosphere(&IcosphereOptions {
        radius,
        subdivisions,
    })
    .unwrap()
}

pub fn cube(size: [f64; 3], at: [f64; 3]) -> AppendSource {
    let src = cuboid(&BoxOptions {
        size,
        segments: [4; 3],
    })
    .unwrap();
    // Centred on the origin: move its min corner to `at`.
    moved(src, std::array::from_fn(|k| at[k] + 0.5 * size[k]))
}

/// `src` turned by `angle` about the axis (1, 2, 3).
pub fn turned(src: AppendSource, angle: f64) -> AppendSource {
    let l = 14f64.sqrt();
    let u = [1.0 / l, 2.0 / l, 3.0 / l];
    let (s, c) = angle.sin_cos();
    AppendSource {
        positions: src
            .positions
            .iter()
            .map(|p| {
                let p = p.map(f64::from);
                let d = u[0] * p[0] + u[1] * p[1] + u[2] * p[2];
                let x = [
                    u[1] * p[2] - u[2] * p[1],
                    u[2] * p[0] - u[0] * p[2],
                    u[0] * p[1] - u[1] * p[0],
                ];
                std::array::from_fn(|k| (c * p[k] + s * x[k] + (1.0 - c) * d * u[k]) as f32)
            })
            .collect(),
        triangles: src.triangles,
    }
}

/// `src` without the triangles whose centroid is within `radius` of a centre.
pub fn holed(src: AppendSource, centers: &[[f64; 3]], radius: f64) -> AppendSource {
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

pub fn mesh_of(src: &AppendSource) -> Mesh {
    Mesh::from_buffers(src.positions.as_flattened(), src.triangles.as_flattened()).unwrap()
}

pub fn bvh_of(mesh: &Mesh) -> Bvh {
    Bvh::build(mesh, &BuildOptions::default()).unwrap()
}

fn c3(p: [f64; 3]) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

/// Exact: whether the closed segment `p q` meets the closed triangle
/// `(a, b, c)`, not lying in its plane.
fn crosses([p, q]: [[f64; 3]; 2], [a, b, c]: [[f64; 3]; 3]) -> bool {
    let o = |a, b, c, d| orient3d(c3(a), c3(b), c3(c), c3(d));
    let (s, t) = (o(a, b, c, p), o(a, b, c, q));
    if (s > 0.0 && t > 0.0) || (s < 0.0 && t < 0.0) || (s == 0.0 && t == 0.0) {
        return false;
    }
    let w = [o(p, q, a, b), o(p, q, b, c), o(p, q, c, a)];
    w.iter().all(|&x| x >= 0.0) || w.iter().all(|&x| x <= 0.0)
}

/// Exact: whether two coplanar triangles overlap in more than their borders.
fn overlap_in_plane(t: [[f64; 3]; 3], u: [[f64; 3]; 3]) -> bool {
    // Drop an axis along which neither is degenerate.
    for (x, y) in [(0, 1), (1, 2), (2, 0)] {
        let flat = |t: [[f64; 3]; 3]| {
            let f = t.map(|p| Coord { x: p[x], y: p[y] });
            let o = orient2d(f[0], f[1], f[2]);
            (if o < 0.0 { [f[0], f[2], f[1]] } else { f }, o != 0.0)
        };
        let ((t, t_ok), (u, u_ok)) = (flat(t), flat(u));
        if t_ok && u_ok {
            let apart = |t: [Coord<f64>; 3], u: [Coord<f64>; 3]| {
                (0..3).any(|i| u.iter().all(|&q| orient2d(t[i], t[(i + 1) % 3], q) <= 0.0))
            };
            return !apart(t, u) && !apart(u, t);
        }
    }
    false
}

/// Exact: the pairs of triangles that share a vertex and also meet elsewhere
/// (beyond the shared vertex or edge), which the engine's self-intersection
/// queries leave out.
pub fn crossing_neighbours(out: &AppendSource) -> usize {
    let p = |v: u32| out.positions[v as usize].map(f64::from);
    let mut fans = vec![Vec::new(); out.positions.len()];
    for (t, tri) in out.triangles.iter().enumerate() {
        for &v in tri {
            fans[v as usize].push(t);
        }
    }
    let mut found = 0;
    for (v, fan) in fans.iter().enumerate() {
        let v = v as u32;
        for (i, &a) in fan.iter().enumerate() {
            for &b in &fan[i + 1..] {
                let (ta, tb) = (out.triangles[a], out.triangles[b]);
                let shared: Vec<u32> = ta.into_iter().filter(|x| tb.contains(x)).collect();
                // A pair on one edge comes up at both its ends; keep one.
                if shared.iter().any(|&x| x < v) {
                    continue;
                }
                let rest = |t: [u32; 3]| -> Vec<[f64; 3]> {
                    t.into_iter()
                        .filter(|x| !shared.contains(x))
                        .map(p)
                        .collect()
                };
                let (pa, pb) = (ta.map(p), tb.map(p));
                let (ra, rb) = (rest(ta), rest(tb));
                let coplanar = rb
                    .iter()
                    .all(|&x| orient3d(c3(pa[0]), c3(pa[1]), c3(pa[2]), c3(x)) == 0.0);
                let bad = if coplanar {
                    overlap_in_plane(pa, pb)
                } else if shared.len() == 1 {
                    crosses([ra[0], ra[1]], pb) || crosses([rb[0], rb[1]], pa)
                } else {
                    false
                };
                found += usize::from(bad);
            }
        }
    }
    found
}

/// Checks that `out` is closed, manifold, consistently and outward wound and
/// free of self-intersections; returns its component count and volume.
pub fn check_surface(out: &AppendSource, what: &str) -> (usize, f64) {
    let mesh = mesh_of(out);
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
    assert_eq!(
        crossing_neighbours(out),
        0,
        "{what}: neighbours that cross each other"
    );
    let mut volume = 0.0;
    for t in &out.triangles {
        let [p, q, s] = t.map(|v| out.positions[v as usize].map(f64::from));
        volume += (p[0] * (q[1] * s[2] - q[2] * s[1]) - p[1] * (q[0] * s[2] - q[2] * s[0])
            + p[2] * (q[0] * s[1] - q[1] * s[0]))
            / 6.0;
    }
    assert!(volume > 0.0, "{what}: wound inward");
    (a.components.len(), volume)
}
