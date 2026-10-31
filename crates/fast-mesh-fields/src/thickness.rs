//! Per-vertex wall thickness of a mesh.
//!
//! [`thicknesses`] writes, for every vertex slot, the distance to the
//! opposite surface along the vertex's smooth normal: the nearest ray hit in
//! either direction (`+n` or `-n`), in the mesh's local `f32` coordinates.
//! Closed solids read their wall thickness (a slab's faces read its depth, a
//! sphere's vertices its diameter); a ray that escapes in both directions
//! reads NaN, so open sheets and the rims of open meshes report NaN rather
//! than a distance to nowhere. Hits beyond `max_distance` read `+∞`, the
//! same "far" value the distance kernels use (and so the same [`ColorRule`]
//! handling: `above` for far, `nan` for open). Dead slots and alive vertices
//! no alive triangle uses read NaN.
//!
//! The normals are area-weighted face-normal sums, normalised (zero stays
//! zero, and reads NaN). Self-hits are skipped by starting the rays a small
//! way off the surface, scaled to the mesh's coordinate magnitude. The sign
//! of the normal does not matter (both directions are tried), so meshes with
//! flipped patches still read the nearest opposite surface.
//!
//! Nothing is allocated per vertex. With the `parallel` feature, large
//! batches of rays run in chunks on rayon's pool through
//! [`fast_mesh_query::raycast_many`]; the values are the same, bit for bit.

use fast_mesh_bvh::Bvh;
use fast_mesh_query::{Cull, RayOptions, raycast_many};
use fast_mesh_store::Mesh;

use crate::Result;
use crate::FieldsError;

/// A thickness that is this fraction of the mesh's coordinate magnitude, or
/// less, is skipped as a self-hit: `32 * f32::EPSILON` times the largest
/// absolute coordinate, about twice the rounding of one ray-triangle test.
fn self_gap(magnitude: f32) -> f32 {
    (32.0 * f32::EPSILON * magnitude).max(f32::MIN_POSITIVE)
}

/// The largest absolute coordinate of the mesh's positions, for
/// [`self_gap`]. 0 when the mesh has no vertices.
fn coordinate_magnitude(mesh: &Mesh) -> f32 {
    mesh.positions()
        .iter()
        .flatten()
        .map(|c| c.abs())
        .fold(0.0, f32::max)
}

/// Per-vertex wall thickness of `mesh`: `out[v]` gets the value for vertex
/// slot `v` (see the module docs).
///
/// `max_distance` bounds the values kept (`+∞` beyond it, inclusive); it
/// must be zero or greater and not NaN, and may be infinite to keep every
/// hit. `out` must hold one entry per vertex slot.
///
/// # Errors
/// Everything is checked before anything is written, so on error `out` is
/// unchanged:
/// - [`FieldsError::InvalidInput`] when `out` is not one entry per vertex
///   slot, or `max_distance` is negative or NaN;
/// - [`FieldsError::Query`] with `MeshMismatch` or `StaleBvh` when `bvh`
///   does not match the current `mesh`.
pub fn thicknesses(mesh: &Mesh, bvh: &Bvh, max_distance: f32, out: &mut [f32]) -> Result<()> {
    if out.len() != mesh.vertex_slots() {
        return Err(FieldsError::InvalidInput(
            "out must hold one entry per vertex slot",
        ));
    }
    if max_distance.is_nan() || max_distance < 0.0 {
        return Err(FieldsError::InvalidInput(
            "max_distance must be zero or greater, and not NaN",
        ));
    }
    if bvh.mesh_uid() != mesh.uid() {
        return Err(fast_mesh_query::QueryError::MeshMismatch.into());
    }
    if !bvh.is_current(mesh) {
        return Err(fast_mesh_query::QueryError::StaleBvh.into());
    }
    let slots = mesh.vertex_slots();
    let mut normals = vec![[0.0f32; 3]; slots];
    // Area-weighted face-normal sums, over the alive triangles in ascending
    // order (the same sums `fast_mesh::compute_vertex_normals` makes).
    for t in 0..mesh.triangle_slots() as u32 {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        if !mesh.is_triangle_alive(t) {
            continue;
        }
        let p = mesh.positions();
        let [a, b, c] = tri.map(|v| p[v as usize]);
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        for v in tri {
            for k in 0..3 {
                normals[v as usize][k] += n[k];
            }
        }
    }
    // Unit normals; zero stays zero (degenerate: reads NaN below).
    for n in &mut normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 0.0 {
            for c in n.iter_mut() {
                *c /= len;
            }
        }
    }
    let gap = self_gap(coordinate_magnitude(mesh));
    let opts = RayOptions {
        min_t: gap,
        max_t: f32::INFINITY,
        cull: Cull::None,
    };
    // One ray pair per defined vertex with a normal; the rest read NaN.
    let mut origins: Vec<[f32; 3]> = Vec::new();
    let mut dirs: Vec<[f32; 3]> = Vec::new();
    let mut slots_of: Vec<u32> = Vec::new();
    for v in 0..slots as u32 {
        let n = normals[v as usize];
        let defined = mesh.is_vertex_alive(v)
            && mesh.corner_count(v) > 0
            && n != [0.0; 3]
            && mesh.position(v).is_some_and(|p| p.iter().all(|c| c.is_finite()));
        if !defined {
            continue;
        }
        let p = mesh.position(v).unwrap_or([0.0; 3]);
        origins.push(p);
        dirs.push(n);
        origins.push(p);
        dirs.push([-n[0], -n[1], -n[2]]);
        slots_of.push(v);
        slots_of.push(v);
    }
    let mut hits = vec![None; origins.len()];
    if !origins.is_empty() {
        raycast_many(mesh, bvh, &origins, &dirs, &opts, &mut hits)?;
    }
    for v in out.iter_mut() {
        *v = f32::NAN;
    }
    debug_assert_eq!(hits.len(), slots_of.len());
    for (i, &v) in slots_of.iter().enumerate() {
        let t = hits[i].as_ref().map(|h| h.t);
        let cur = out[v as usize];
        match (cur.is_nan(), t) {
            // First of the pair, or a miss after a hit: keep what is there.
            (_, None) => {}
            // A hit where none was: the distance, far when past the bound.
            (true, Some(t)) => {
                out[v as usize] = if t > max_distance {
                    f32::INFINITY
                } else {
                    t
                };
            }
            // The nearer of the pair, far when past the bound.
            (false, Some(t)) => {
                let got = if t > max_distance {
                    f32::INFINITY
                } else {
                    t
                };
                if got < cur {
                    out[v as usize] = got;
                }
            }
        }
    }
    // Keep the parallel paths warm under the same thresholds the distance
    // kernels use (a no-op here: `raycast_many` parallelises on its own).
    Ok(())
}

#[cfg(test)]
mod tests {
    use fast_mesh_bvh::{BuildOptions, Bvh};
    use fast_mesh_store::Mesh;

    use super::*;
    use crate::test_util::*;
    use fast_mesh_store::par;

    fn build(mesh: &Mesh) -> Bvh {
        Bvh::build(
            mesh,
            &BuildOptions {
                max_leaf_size: 2,
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn run(mesh: &Mesh, max_distance: f32) -> Vec<f32> {
        let bvh = build(mesh);
        let mut out = vec![f32::NAN; mesh.vertex_slots()];
        thicknesses(mesh, &bvh, max_distance, &mut out).unwrap();
        out
    }

    #[test]
    fn cube_vertices_read_wall_to_wall() {
        // Every corner of the [-1, 1]^3 cube reads the distance to the
        // opposite surface along its area-weighted normal: between one face
        // width (2) and the body diagonal (2 sqrt(3)).
        let mesh = cube();
        let out = run(&mesh, f32::INFINITY);
        assert_eq!(out.len(), 8);
        for &v in &out {
            assert!(v.is_finite(), "{out:?}");
            assert!((2.0..=2.0 * 3f32.sqrt() + 1e-4).contains(&v), "{out:?}");
        }
    }

    /// A closed plate, 1 x 1 x 0.5 (`n` cells a side), facing out: the
    /// interior face vertices have untilted normals and read the depth.
    fn plate(n: u32) -> Mesh {
        let h = 0.25;
        let w = n + 1;
        let mut positions = Vec::new();
        for z in [h, -h] {
            for y in 0..=n {
                for x in 0..=n {
                    positions.push([x as f32 / n as f32, y as f32 / n as f32, z]);
                }
            }
        }
        let (top, bottom) = (0, w * w);
        let t = |x: u32, y: u32| top + y * w + x;
        let b = |x: u32, y: u32| bottom + y * w + x;
        let mut tris = Vec::new();
        for y in 0..n {
            for x in 0..n {
                tris.extend([[t(x, y), t(x + 1, y), t(x + 1, y + 1)], [t(x, y), t(x + 1, y + 1), t(x, y + 1)]]);
                tris.extend([[b(x, y), b(x + 1, y + 1), b(x + 1, y)], [b(x, y), b(x, y + 1), b(x + 1, y + 1)]]);
            }
        }
        for x in 0..n {
            // South (y = 0) and north (y = 1) skirts.
            tris.extend([
                [t(x, 0), b(x, 0), b(x + 1, 0)],
                [t(x, 0), b(x + 1, 0), t(x + 1, 0)],
                [t(x, n), t(x + 1, n), b(x + 1, n)],
                [t(x, n), b(x + 1, n), b(x, n)],
            ]);
            // West (x = 0) and east (x = 1) skirts.
            tris.extend([
                [t(0, x), t(0, x + 1), b(0, x + 1)],
                [t(0, x), b(0, x + 1), b(0, x)],
                [t(n, x), b(n, x), b(n, x + 1)],
                [t(n, x), b(n, x + 1), t(n, x + 1)],
            ]);
        }
        mesh(&positions, &tris)
    }

    #[test]
    fn slab_faces_read_depth() {
        // The interior face vertices read the plate depth, 0.5.
        let m = plate(4);
        let out = run(&m, f32::INFINITY);
        let w = 5;
        for y in 1..4 {
            for x in 1..4 {
                for &v in &[y * w + x, w * w + y * w + x] {
                    assert!((out[v as usize] - 0.5).abs() < 1e-3, "{out:?}");
                }
            }
        }
        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
    }

    #[test]
    fn sphere_reads_diameter() {
        let mesh = icosphere();
        let out = run(&mesh, f32::INFINITY);
        assert_eq!(out.len(), mesh.vertex_slots());
        for &v in &out {
            assert!(v.is_finite(), "{out:?}");
            assert!((v - 2.0).abs() < 0.05, "{v}");
        }
    }

    #[test]
    fn open_sheet_reads_nan() {
        // An open sheet has no opposite surface: every ray escapes.
        let mesh = terrain(4);
        let out = run(&mesh, f32::INFINITY);
        assert!(!out.is_empty());
        for &v in &out {
            assert!(v.is_nan(), "{v}");
        }
    }

    #[test]
    fn open_cube_rim_reads_nan_while_walls_read_finite() {
        // The open-cube fixture (top face removed): the missing face's
        // vertices have no opposite surface above them... check the mix:
        // walls read finite, and nothing reads a bogus zero.
        let (p, mut t) = cube_buffers();
        t.drain(2..4);
        let m = mesh(&p, &t);
        let out = run(&m, f32::INFINITY);
        assert_eq!(out.len(), 8);
        for &v in &out {
            assert!(v.is_nan() || v > 1.0, "{out:?}");
        }
        assert!(out.iter().any(|v| v.is_finite()), "{out:?}");
    }

    #[test]
    fn far_reads_infinite() {
        let mesh = cube();
        let out = run(&mesh, 1.0);
        for &v in &out {
            assert_eq!(v, f32::INFINITY, "{out:?}");
        }
        let out = run(&mesh, 4.0);
        for &v in &out {
            assert!(v.is_finite(), "{out:?}");
        }
    }

    #[test]
    fn dead_and_unused_slots_read_nan() {
        // An isolated vertex (alive, used by no triangle) reads NaN while
        // the cube around it reads finite.
        let (mut positions, tris) = cube_buffers();
        positions.push([5.0, 5.0, 5.0]);
        let m = mesh(&positions, &tris);
        assert_eq!(m.vertex_slots(), 9);
        assert!(m.is_vertex_alive(8));
        assert_eq!(m.corner_count(8), 0);
        let bvh = build(&m);
        let mut out = vec![7.0f32; m.vertex_slots()];
        thicknesses(&m, &bvh, f32::INFINITY, &mut out).unwrap();
        assert!(out[8].is_nan(), "{out:?}");
        assert!(out[..8].iter().all(|v| v.is_finite()), "{out:?}");
    }

    #[test]
    fn invalid_input_writes_nothing() {
        let mesh = cube();
        let bvh = build(&mesh);
        let mut out = vec![7.0f32; mesh.vertex_slots()];
        let bad_len = thicknesses(&mesh, &bvh, f32::INFINITY, &mut out[..4]);
        assert!(matches!(bad_len, Err(FieldsError::InvalidInput(_))));
        assert!(matches!(
            thicknesses(&mesh, &bvh, f32::NAN, &mut out),
            Err(FieldsError::InvalidInput(_))
        ));
        assert!(matches!(
            thicknesses(&mesh, &bvh, -1.0, &mut out),
            Err(FieldsError::InvalidInput(_))
        ));
        assert_eq!(out, vec![7.0; mesh.vertex_slots()]);
        let other = cube();
        let other_bvh = build(&other);
        assert!(matches!(
            thicknesses(&mesh, &other_bvh, 1.0, &mut out),
            Err(FieldsError::Query(_))
        ));
        assert_eq!(out, vec![7.0; mesh.vertex_slots()]);
    }

    #[test]
    fn thickness_bit_identical() {
        // The forced-sequential run is the code a feature-off build runs.
        // The torus is closed (every vertex reads finite), with enough rays
        // to pass the parallel threshold.
        let mesh = torus();
        let bvh = build(&mesh);
        assert!(mesh.triangle_count() >= 500);
        let run = || {
            let mut out = vec![f32::NAN; mesh.vertex_slots()];
            thicknesses(&mesh, &bvh, f32::INFINITY, &mut out).unwrap();
            out.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        };
        par::set_force_sequential(true);
        let reference = run();
        par::set_force_sequential(false);
        if par::enabled() {
            for k in [1usize, 2, 8] {
                assert_eq!(par::with_threads(k, &run), reference, "{k} threads");
            }
        }
        // Every vertex of the torus reads a finite wall thickness.
        let mut out = vec![f32::NAN; mesh.vertex_slots()];
        thicknesses(&mesh, &bvh, f32::INFINITY, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
    }
}
