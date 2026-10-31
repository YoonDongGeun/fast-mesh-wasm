//! Per-vertex signed mean curvature of a mesh.
//!
//! [`curvatures`] writes, for every vertex slot, the discrete signed mean
//! curvature at that vertex: positive on domes (convex as seen from the
//! normal side), negative in bowls and dents, zero on flat spots. On a
//! sphere of radius `r` with outward normals every vertex reads about
//! `1 / r`; on a plane every vertex reads 0; a saddle reads the sign of
//! its dominant bend (zero where the two bends cancel, as at the middle of
//! a symmetric saddle).
//!
//! The estimate is the cotangent Laplace one (Meyer et al. 2003): with
//! `w` the half-cotangent edge weights, `A` the barycentric area (a third
//! of the adjacent triangles' area) and `n` the area-weighted smooth normal,
//! `H = -(sum_j w_ij (x_j - x_i)) . n / (2 A)`. The sign follows the
//! normals: flipping every triangle negates the values.
//! The `. n` projection keeps flat regions at exactly 0 however irregular
//! their triangulation is (the raw Laplace vector has a spurious tangential
//! part on irregular flat meshes).
//!
//! Units are 1/local-unit: a mesh scaled by `s` reads `H / s` (the scene
//! applies this for uniform placements, and bakes non-uniform ones into a
//! world-frame copy first, so its fields read 1/world-unit).
//!
//! Boundary vertices use their one-sided fan like any other vertex. Dead
//! slots, alive vertices no alive triangle uses, vertices with a non-finite
//! position, and degenerate stencils (zero area or zero normal) read NaN.
//!
//! Nothing is allocated per vertex, and the neighbour sums run in ascending
//! vertex order, so the values are bit-identical however they are threaded.

use fast_mesh_store::Mesh;

use crate::Result;
use crate::FieldsError;

/// Per-vertex signed mean curvature of `mesh`: `out[v]` gets the value for
/// vertex slot `v` (see the module docs), in 1/local-units. `out` must hold
/// one entry per vertex slot.
///
/// # Errors
/// [`FieldsError::InvalidInput`] when `out` is not one entry per vertex
/// slot. On error `out` is unchanged.
pub fn curvatures(mesh: &Mesh, out: &mut [f32]) -> Result<()> {
    if out.len() != mesh.vertex_slots() {
        return Err(FieldsError::InvalidInput(
            "out must hold one entry per vertex slot",
        ));
    }
    let slots = mesh.vertex_slots();
    // Half-cotangent weights by edge, kept per endpoint in ascending
    // neighbour order (deterministic sums); barycentric areas; smooth
    // (area-weighted) normals.
    let mut neighbours: Vec<Vec<(u32, f64)>> = Vec::new();
    neighbours.try_reserve_exact(slots).map_err(|_| FieldsError::OutOfMemory)?;
    neighbours.resize(slots, Vec::new());
    let mut areas = vec![0.0f64; slots];
    let mut normals = vec![[0.0f64; 3]; slots];
    for t in 0..mesh.triangle_slots() as u32 {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        if !mesh.is_triangle_alive(t) {
            continue;
        }
        let p = mesh.positions();
        let [a, b, c] = tri.map(|v| p[v as usize]);
        if ![a, b, c].iter().all(|p| p.iter().all(|c| c.is_finite())) {
            continue;
        }
        let fa = a.map(f64::from);
        let fb = b.map(f64::from);
        let fc = c.map(f64::from);
        let (ab, ac, bc) = (sub(fb, fa), sub(fc, fa), sub(fc, fb));
        let n = cross(ab, ac);
        let area2 = len(n);
        // Zero-area (or empty) triangles contribute nothing; positions are
        // finite, so areas are finite too and `<=` is exact here.
        if area2 <= 0.0 {
            continue;
        }
        for v in tri {
            areas[v as usize] += area2 / 6.0;
            for k in 0..3 {
                normals[v as usize][k] += n[k];
            }
        }
        // Cotangents at a, b, c; each weighs the opposite edge by half.
        let ca = dot(ab, ac) / area2;
        let cb = dot([-ab[0], -ab[1], -ab[2]], bc) / area2;
        let cc = dot([-ac[0], -ac[1], -ac[2]], [-bc[0], -bc[1], -bc[2]]) / area2;
        add(&mut neighbours, tri[0], tri[1], cc / 2.0)?;
        add(&mut neighbours, tri[1], tri[2], ca / 2.0)?;
        add(&mut neighbours, tri[2], tri[0], cb / 2.0)?;
    }
    for list in &mut neighbours {
        list.sort_by_key(|&(v, _)| v);
    }
    let pos = mesh.positions();
    for (v, o) in out.iter_mut().enumerate() {
        let alive = mesh.is_vertex_alive(v as u32)
            && mesh.corner_count(v as u32) > 0
            && pos[v].iter().all(|c| c.is_finite());
        let area = areas[v];
        let n = normals[v];
        let nlen = len(n);
        if !alive || area <= 0.0 || nlen <= 0.0 {
            *o = f32::NAN;
            continue;
        }
        let x = pos[v].map(f64::from);
        let mut lap = [0.0; 3];
        for &(u, w) in &neighbours[v] {
            let y = pos[u as usize].map(f64::from);
            for k in 0..3 {
                lap[k] += w * (y[k] - x[k]);
            }
        }
        let proj = (lap[0] * n[0] + lap[1] * n[1] + lap[2] * n[2]) / nlen;
        *o = (-proj / (2.0 * area)) as f32;
    }
    Ok(())
}

/// Adds `w` to the weights between `a` and `b`, both ways round.
fn add(neighbours: &mut [Vec<(u32, f64)>], a: u32, b: u32, w: f64) -> Result<()> {
    for (v, u) in [(a, b), (b, a)] {
        let list = &mut neighbours[v as usize];
        if let Some(slot) = list.iter_mut().find(|(x, _)| *x == u) {
            slot.1 += w;
        } else {
            list.try_reserve(1).map_err(|_| FieldsError::OutOfMemory)?;
            list.push((u, w));
        }
    }
    Ok(())
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn len(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::*;

    fn run(mesh: &Mesh) -> Vec<f32> {
        let mut out = vec![f32::NAN; mesh.vertex_slots()];
        curvatures(mesh, &mut out).unwrap();
        out
    }

    /// A flat `(n+1) x (n+1)` grid in `z = 0`, `h` apart.
    fn plane(n: u32, h: f32) -> Mesh {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                positions.push([x as f32 * h, y as f32 * h, 0.0]);
            }
        }
        let mut tris = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                tris.extend([[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]);
            }
        }
        mesh(&positions, &tris)
    }

    /// A saddle `z = (2 x^2 - y^2) / s` over `[-1, 1]^2`, `n` cells a side,
    /// facing up: smooth mean curvature `-1 / s` at the middle (the bend
    /// toward the normal wins).
    fn saddle(n: u32, s: f32) -> Mesh {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                let (fx, fy) = (x as f32 / n as f32 * 2.0 - 1.0, y as f32 / n as f32 * 2.0 - 1.0);
                positions.push([fx, fy, (2.0 * fx * fx - fy * fy) / s]);
            }
        }
        let mut tris = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                tris.extend([[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]);
            }
        }
        mesh(&positions, &tris)
    }

    #[test]
    fn sphere_reads_one_over_r() {
        let m = icosphere();
        let out = run(&m);
        assert_eq!(out.len(), m.vertex_slots());
        for &v in &out {
            assert!(v.is_finite(), "{out:?}");
            assert!((v - 1.0).abs() < 0.2, "{v}");
        }
        // Twice the radius reads half the curvature, bit for bit up to the
        // float rounding of the doubled positions.
        let (p, t) = (m.positions().to_vec(), m.triangles().to_vec());
        let big = mesh(
            &p.iter().map(|q| [q[0] * 2.0, q[1] * 2.0, q[2] * 2.0]).collect::<Vec<_>>(),
            &t,
        );
        for (&v, &w) in out.iter().zip(run(&big).iter()) {
            assert!((w - v / 2.0).abs() < 1e-6, "{w} vs {v}");
        }
    }

    #[test]
    fn plane_reads_zero() {
        let m = plane(4, 0.25);
        let out = run(&m);
        for &v in &out {
            assert!(v.is_finite(), "{out:?}");
            assert!(v.abs() < 1e-5, "{v}");
        }
    }

    #[test]
    fn saddle_reads_negative() {
        // The middle reads about -1/4 (the asymmetric bend wins); the rim
        // uses its one-sided fan and stays finite.
        let m = saddle(32, 4.0);
        let c = (16 * 33 + 16) as usize;
        let out = run(&m);
        assert!((out[c] + 0.25).abs() < 0.1, "{:?}", out[c]);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|&v| v < -0.1));
    }

    #[test]
    fn cube_corners_read_positive() {
        // Sharp convex corners read positive (domes from outside).
        let m = cube();
        for &v in &run(&m) {
            assert!(v.is_finite(), "{v}");
            assert!(v > 0.3, "{v}");
        }
    }

    #[test]
    fn flipped_winding_negates() {
        // The sign follows the normals: flipping every triangle negates it.
        let m = icosphere();
        let t = m.triangles().iter().map(|&[a, b, c]| [a, c, b]).collect::<Vec<_>>();
        let flipped = mesh(m.positions(), &t);
        let (a, b) = (run(&m), run(&flipped));
        for (&x, &y) in a.iter().zip(&b) {
            assert!((x + y).abs() < 1e-6, "{x} vs {y}");
        }
    }

    #[test]
    fn dead_and_unused_slots_read_nan() {
        let (mut positions, tris) = cube_buffers();
        positions.push([5.0, 5.0, 5.0]);
        let m = mesh(&positions, &tris);
        assert_eq!(m.vertex_slots(), 9);
        assert!(m.is_vertex_alive(8));
        assert_eq!(m.corner_count(8), 0);
        let mut out = vec![7.0f32; m.vertex_slots()];
        curvatures(&m, &mut out).unwrap();
        assert!(out[8].is_nan(), "{out:?}");
        assert!(out[..8].iter().all(|v| v.is_finite()), "{out:?}");
    }

    #[test]
    fn invalid_input_writes_nothing() {
        let m = cube();
        let mut out = vec![7.0f32; m.vertex_slots()];
        assert!(matches!(
            curvatures(&m, &mut out[..4]),
            Err(FieldsError::InvalidInput(_))
        ));
        assert_eq!(out, vec![7.0; m.vertex_slots()]);
    }
}
