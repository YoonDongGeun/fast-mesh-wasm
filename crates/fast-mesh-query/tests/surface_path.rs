use fast_mesh_query::{QueryError, SurfacePathPoint, shortest_surface_path};
use fast_mesh_store::Mesh;

fn grid_mesh(nx: u32, ny: u32) -> Mesh {
    let mut pos = Vec::new();
    for y in 0..=ny {
        for x in 0..=nx {
            pos.extend_from_slice(&[x as f32, y as f32, 0.0]);
        }
    }
    let mut idx = Vec::new();
    let stride = nx + 1;
    for y in 0..ny {
        for x in 0..nx {
            let a = y * stride + x;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            idx.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
    Mesh::from_buffers(&pos, &idx).unwrap()
}

fn icosphere_unit(subdivisions: u32) -> Mesh {
    use std::collections::HashMap;
    let t = (1.0 + 5.0f64.sqrt()) * 0.5;
    let raw = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    let mut verts: Vec<[f64; 3]> = raw
        .into_iter()
        .map(|p| {
            let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            [p[0] / l, p[1] / l, p[2] / l]
        })
        .collect();
    let mut tris: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..subdivisions {
        let mut mid = HashMap::<(u32, u32), u32>::new();
        let mut next = Vec::with_capacity(tris.len() * 4);
        let mut get_mid = |u: u32, v: u32, vs: &mut Vec<[f64; 3]>| -> u32 {
            let key = if u < v { (u, v) } else { (v, u) };
            if let Some(&idx) = mid.get(&key) {
                return idx;
            }
            let a = vs[u as usize];
            let b = vs[v as usize];
            let m = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
            let l = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
            let idx = vs.len() as u32;
            vs.push([m[0] / l, m[1] / l, m[2] / l]);
            mid.insert(key, idx);
            idx
        };
        for [a, b, c] in tris {
            let ab = get_mid(a, b, &mut verts);
            let bc = get_mid(b, c, &mut verts);
            let ca = get_mid(c, a, &mut verts);
            next.push([a, ab, ca]);
            next.push([b, bc, ab]);
            next.push([c, ca, bc]);
            next.push([ab, bc, ca]);
        }
        tris = next;
    }
    let pos: Vec<f32> = verts
        .into_iter()
        .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    let idx: Vec<u32> = tris.into_iter().flatten().collect();
    Mesh::from_buffers(&pos, &idx).unwrap()
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f64 {
    let dx = f64::from(a[0] - b[0]);
    let dy = f64::from(a[1] - b[1]);
    let dz = f64::from(a[2] - b[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn path_len(path: &[SurfacePathPoint]) -> f64 {
    path.windows(2)
        .map(|w| dist3(w[0].position, w[1].position))
        .sum()
}

fn assert_path_invariants(mesh: &Mesh, path: &[SurfacePathPoint], sources: &[u32], target: u32) {
    assert!(!path.is_empty());
    let pos = mesh.positions();
    for pt in path {
        let tri = mesh
            .triangle(pt.triangle)
            .expect("point triangle must be alive");
        let w = pt.barycentric;
        for wi in w {
            assert!(
                (-1e-6..=1.0 + 1e-6).contains(&wi),
                "invalid barycentric coordinate {w:?}"
            );
        }
        let sum = w[0] + w[1] + w[2];
        assert!((sum - 1.0).abs() < 1e-4, "barycentric sum = {sum}");
        let recon = [0, 1, 2].map(|c| {
            w[0] * pos[tri[0] as usize][c]
                + w[1] * pos[tri[1] as usize][c]
                + w[2] * pos[tri[2] as usize][c]
        });
        assert!(
            dist3(pt.position, recon) < 1e-4,
            "position mismatch: {:?} vs {recon:?}",
            pt.position
        );
    }

    let starts_at_source = sources
        .iter()
        .any(|&s| dist3(path[0].position, pos[s as usize]) < 1e-5);
    assert!(starts_at_source, "path did not start at any source");
    assert!(
        dist3(path.last().unwrap().position, pos[target as usize]) < 1e-5,
        "path did not end at target"
    );
}

#[test]
fn target_equal_to_source_produces_single_point() {
    let m = grid_mesh(2, 2);
    let d = vec![0.0f32; m.vertex_slots()];
    let mut out = Vec::new();
    shortest_surface_path(&m, &d, &[4], 4, &mut out).unwrap();
    assert_eq!(out.len(), 1);
    assert_path_invariants(&m, &out, &[4], 4);
}

#[test]
fn non_diagonal_grid_and_multisource() {
    let m = grid_mesh(6, 6);
    let stride = 7u32;
    let s1 = 0u32; // (0, 0)
    let s2 = 6u32; // (6, 0)
    let target = 4 * stride + 5; // (5, 4): closer to s2=(6,0) [dist sqrt(17)=4.123] than s1 [dist sqrt(41)=6.403]

    let d_single: Vec<f32> = m
        .positions()
        .iter()
        .map(|p| (f64::from(p[0]).hypot(f64::from(p[1]))) as f32)
        .collect();
    let mut out = Vec::new();
    shortest_surface_path(&m, &d_single, &[s1], target, &mut out).unwrap();
    assert_path_invariants(&m, &out, &[s1], target);
    let exact = 5.0f64.hypot(4.0);
    assert!(
        (path_len(&out) - exact).abs() < 0.08,
        "expected {exact}, got {}",
        path_len(&out)
    );

    // Multi-source chooses closer source s2.
    let d_multi: Vec<f32> = m
        .positions()
        .iter()
        .map(|p| {
            let a = f64::from(p[0]).hypot(f64::from(p[1]));
            let b = f64::from(p[0] - 6.0).hypot(f64::from(p[1]));
            a.min(b) as f32
        })
        .collect();
    shortest_surface_path(&m, &d_multi, &[s1, s2], target, &mut out).unwrap();
    assert_path_invariants(&m, &out, &[s1, s2], target);
    assert!(dist3(out[0].position, m.positions()[s2 as usize]) < 1e-5);
}

#[test]
fn unit_sphere_great_circle_surface_path() {
    let m = icosphere_unit(2);
    let p0 = m.positions()[0].map(f64::from);
    let mut target = 0u32;
    let mut best_ang = 0.0f64;
    let d: Vec<f32> = m
        .positions()
        .iter()
        .enumerate()
        .map(|(i, pv)| {
            let q = pv.map(f64::from);
            let ang = (p0[0] * q[0] + p0[1] * q[1] + p0[2] * q[2])
                .clamp(-1.0, 1.0)
                .acos();
            if ang < 2.3 && ang > best_ang {
                best_ang = ang;
                target = i as u32;
            }
            ang as f32
        })
        .collect();

    let mut path = Vec::new();
    shortest_surface_path(&m, &d, &[0], target, &mut path).unwrap();
    assert_path_invariants(&m, &path, &[0], target);
    let len = path_len(&path);
    assert!(
        (len - best_ang).abs() / best_ang < 0.05,
        "sphere path len {len} vs arc {best_ang}"
    );
    for pt in &path {
        let r = dist3(pt.position, [0.0, 0.0, 0.0]);
        assert!((r - 1.0).abs() < 0.04, "left unit sphere radius={r}");
    }
}

#[test]
fn u_shape_concave_detour_and_noisy_escape() {
    let nx = 10u32;
    let ny = 4u32;
    let mut pos = Vec::new();
    for y in 0..=ny {
        for x in 0..=nx {
            pos.extend_from_slice(&[x as f32, y as f32, 0.0]);
        }
    }
    let mut idx = Vec::new();
    let stride = nx + 1;
    for y in 0..ny {
        for x in 0..nx {
            if x >= 2 && (1..3).contains(&y) {
                continue;
            }
            let a = y * stride + x;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            idx.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
    let m = Mesh::from_buffers(&pos, &idx).unwrap();
    let tip_a = nx; // (10, 0, 0)
    let tip_b = ny * stride + nx; // (10, 4, 0)

    // Flat/constant distance field except 0 at source forces cycle-free escape without infinite loop.
    let mut flat = vec![1.0f32; m.vertex_slots()];
    flat[tip_a as usize] = 0.0;
    let mut path = Vec::new();
    shortest_surface_path(&m, &flat, &[tip_a], tip_b, &mut path).unwrap();
    assert_path_invariants(&m, &path, &[tip_a], tip_b);
    assert!(
        path_len(&path) > 16.0,
        "detour must go around U-notch, got len {}",
        path_len(&path)
    );
}

#[test]
fn disconnected_and_invalid_inputs_preserve_output() {
    let pos = [
        0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // comp 0
        5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 5.0, 1.0, 0.0, // comp 1
    ];
    let idx = [0u32, 1, 2, 3, 4, 5];
    let m = Mesh::from_buffers(&pos, &idx).unwrap();
    let d = vec![0.0, 1.0, 1.0, 2.0, 3.0, 3.0];

    let sentinel = vec![SurfacePathPoint {
        position: [9.0, 9.0, 9.0],
        triangle: 0,
        barycentric: [1.0, 0.0, 0.0],
    }];
    let mut out = sentinel.clone();

    assert!(matches!(
        shortest_surface_path(&m, &d, &[0], 4, &mut out),
        Err(QueryError::InvalidInput(_))
    ));
    assert_eq!(out, sentinel);

    assert!(matches!(
        shortest_surface_path(&m, &d[..3], &[0], 1, &mut out),
        Err(QueryError::InvalidInput(_))
    ));
    assert_eq!(out, sentinel);

    assert!(matches!(
        shortest_surface_path(&m, &d, &[], 1, &mut out),
        Err(QueryError::InvalidInput(_))
    ));
    assert_eq!(out, sentinel);
}

#[test]
fn dead_vertex_rejected_and_remaining_mesh_succeeds() {
    let mut m = grid_mesh(2, 2);
    {
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[6, 7]).unwrap();
        tx.remove_vertices(&[8]).unwrap();
        tx.commit().unwrap();
    }
    let d: Vec<f32> = m
        .positions()
        .iter()
        .map(|p| (f64::from(p[0]).hypot(f64::from(p[1]))) as f32)
        .collect();
    let mut out = Vec::new();
    assert!(matches!(
        shortest_surface_path(&m, &d, &[0], 8, &mut out),
        Err(QueryError::InvalidInput(_))
    ));
    assert!(matches!(
        shortest_surface_path(&m, &d, &[8], 0, &mut out),
        Err(QueryError::InvalidInput(_))
    ));
    shortest_surface_path(&m, &d, &[0], 7, &mut out).unwrap();
    assert_path_invariants(&m, &out, &[0], 7);
}
