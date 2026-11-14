//! Integration tests for #208 (Botsch-Kobbelt surface remeshing) and
//! #211 (attribute-aware, error-bounded, and crease-locked simplification).

use fast_mesh_ops::remesh::{
    SurfaceRemeshMode, SurfaceRemeshOptions, remesh_surface, remesh_surface_with_mask,
};
use fast_mesh_ops::{
    AnalyzeOptions, DecimateOptions, SimplifyExtendedOptions, SimplifyOptions, analyze,
    boundary_loops, simplify, simplify_with,
};
use fast_mesh_store::{Mesh, VertexId};

fn edit<R>(
    mesh: &mut Mesh,
    f: impl FnOnce(&mut fast_mesh_store::Transaction<'_>) -> fast_mesh_ops::Result<R>,
) -> R {
    let mut tx = mesh.begin_edit();
    let report = f(&mut tx).unwrap();
    tx.commit().unwrap();
    report
}

fn live_triangles(mesh: &Mesh) -> usize {
    (0..mesh.triangle_slots() as u32)
        .filter(|&t| mesh.triangle(t).is_some())
        .count()
}

fn sphere_mesh(stacks: u32, slices: u32, radius: f32) -> Mesh {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    positions.extend_from_slice(&[0.0, radius, 0.0]);
    for i in 1..stacks {
        let theta = std::f32::consts::PI * (i as f32 / stacks as f32);
        let y = radius * theta.cos();
        let r = radius * theta.sin();
        for j in 0..slices {
            let phi = 2.0 * std::f32::consts::PI * (j as f32 / slices as f32);
            positions.extend_from_slice(&[r * phi.cos(), y, r * phi.sin()]);
        }
    }
    positions.extend_from_slice(&[0.0, -radius, 0.0]);
    let south = 1 + (stacks - 1) * slices;
    let ring = |i: u32, j: u32| 1 + (i - 1) * slices + (j % slices);

    for j in 0..slices {
        indices.extend_from_slice(&[0, ring(1, j + 1), ring(1, j)]);
    }
    for i in 1..(stacks - 1) {
        for j in 0..slices {
            let a = ring(i, j);
            let b = ring(i, j + 1);
            let c = ring(i + 1, j + 1);
            let d = ring(i + 1, j);
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    for j in 0..slices {
        indices.extend_from_slice(&[south, ring(stacks - 1, j), ring(stacks - 1, j + 1)]);
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn subdivided_grid(n: u32, extent: f32) -> Mesh {
    let mut positions = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let x = extent * (i as f32 / n as f32);
            let y = extent * (j as f32 / n as f32);
            positions.extend_from_slice(&[x, y, 0.0]);
        }
    }
    let idx = |i: u32, j: u32| j * (n + 1) + i;
    let mut indices = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let a = idx(i, j);
            let b = idx(i + 1, j);
            let c = idx(i + 1, j + 1);
            let d = idx(i, j + 1);
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn sharp_box(min: [f32; 3], max: [f32; 3]) -> Mesh {
    let [x0, y0, z0] = min;
    let [x1, y1, z1] = max;
    let positions = [
        x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1,
        z1,
    ];
    let indices = [
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6,
        3, 0, 4, 3, 4, 7,
    ];
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// 6-face grid box whose flat interiors and subdivided edges meet at 90-degree corners.
fn subdivided_box(n: u32) -> Mesh {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    let mut push_face = |origin: [f32; 3], u: [f32; 3], v: [f32; 3]| {
        let base = (positions.len() / 3) as u32;
        for j in 0..=n {
            let tv = j as f32 / n as f32;
            for i in 0..=n {
                let tu = i as f32 / n as f32;
                positions.extend_from_slice(&[
                    origin[0] + tu * u[0] + tv * v[0],
                    origin[1] + tu * u[1] + tv * v[1],
                    origin[2] + tu * u[2] + tv * v[2],
                ]);
            }
        }
        for j in 0..n {
            for i in 0..n {
                let a = base + j * (n + 1) + i;
                let b = a + 1;
                let c = a + n + 2;
                let d = a + n + 1;
                indices.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }
    };
    push_face([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    push_face([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    push_face([1.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
    push_face([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
    push_face([0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
    push_face([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);

    let mut m = Mesh::from_buffers(&positions, &indices).unwrap();
    edit(&mut m, |tx| fast_mesh_ops::weld(tx, 1e-6));
    m
}

fn edge_lengths(mesh: &Mesh) -> Vec<f64> {
    let mut edges = Vec::new();
    for t in 0..mesh.triangle_slots() as u32 {
        if let Some(c) = mesh.triangle(t) {
            for k in 0..3 {
                let a = c[k];
                let b = c[(k + 1) % 3];
                edges.push(if a < b { (a, b) } else { (b, a) });
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    edges
        .into_iter()
        .map(|(a, b)| {
            let pa = mesh.position(a).unwrap();
            let pb = mesh.position(b).unwrap();
            let dx = f64::from(pb[0] - pa[0]);
            let dy = f64::from(pb[1] - pa[1]);
            let dz = f64::from(pb[2] - pa[2]);
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .collect()
}

#[test]
fn isotropic_surface_remesh_produces_uniform_manifold_sphere() {
    let mut mesh = sphere_mesh(12, 24, 1.0);
    let opts = SurfaceRemeshOptions::builder()
        .mode(SurfaceRemeshMode::Isotropic)
        .target_edge_length(0.22)
        .iterations(5)
        .lock_creases(false)
        .build();
    let report = edit(&mut mesh, |tx| remesh_surface(tx, &opts));
    assert!(report.splits + report.collapses > 0);
    assert_eq!(report.iterations_run, 5);
    assert!(report.final_triangles > 0);

    let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
    assert_eq!(a.non_manifold_edges, 0);
    assert!(a.non_manifold_vertices.is_empty());
    assert_eq!(a.orientation_conflicts, 0);
    assert!(a.loops.is_empty());

    // Vertices remain projected onto the unit sphere within small tolerance.
    for v in 0..mesh.vertex_slots() as VertexId {
        if mesh.is_vertex_alive(v) {
            let p = mesh.position(v).unwrap();
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((r - 1.0).abs() < 0.06, "radial drift too large: {r}");
        }
    }

    // Edge lengths should cluster near target_edge_length = 0.22.
    let lens = edge_lengths(&mesh);
    let mean = lens.iter().sum::<f64>() / lens.len() as f64;
    assert!(
        (mean - 0.22).abs() < 0.09,
        "unexpected mean edge len {mean}"
    );
}

#[test]
fn adaptive_surface_remesh_allocates_smaller_edges_at_high_curvature() {
    // Stretched ellipsoid: sharp nose at |x| ~= 2.5, flatter sides near x ~= 0.
    let mut mesh = sphere_mesh(12, 24, 1.0);
    let ids: Vec<VertexId> = (0..mesh.vertex_slots() as VertexId).collect();
    let scaled: Vec<[f32; 3]> = mesh
        .positions()
        .iter()
        .map(|p| [p[0] * 2.5, p[1] * 0.7, p[2] * 0.7])
        .collect();
    {
        let mut tx = mesh.begin_edit();
        tx.set_positions(&ids, &scaled).unwrap();
        tx.commit().unwrap();
    }

    let opts = SurfaceRemeshOptions::builder()
        .mode(SurfaceRemeshMode::Adaptive)
        .target_edge_length(0.28)
        .min_edge_length(0.11)
        .max_edge_length(0.48)
        .iterations(5)
        .lock_creases(false)
        .build();
    let rep = edit(&mut mesh, |tx| remesh_surface(tx, &opts));
    assert!(rep.final_triangles > 0);
    let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
    assert_eq!(a.non_manifold_edges, 0);
    assert_eq!(a.orientation_conflicts, 0);

    // Count triangle density around sharp tips (|x| > 2.0) vs broad belly (|x| < 0.5).
    let mut tip_edges = Vec::new();
    let mut belly_edges = Vec::new();
    for t in 0..mesh.triangle_slots() as u32 {
        if let Some([i0, i1, i2]) = mesh.triangle(t) {
            let p0 = mesh.position(i0).unwrap();
            let p1 = mesh.position(i1).unwrap();
            let p2 = mesh.position(i2).unwrap();
            let cx = ((p0[0] + p1[0] + p2[0]) / 3.0).abs();
            let avg_e = ((((p1[0] - p0[0]).powi(2)
                + (p1[1] - p0[1]).powi(2)
                + (p1[2] - p0[2]).powi(2))
            .sqrt()
                + ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2) + (p2[2] - p1[2]).powi(2))
                    .sqrt()
                + ((p0[0] - p2[0]).powi(2) + (p0[1] - p2[1]).powi(2) + (p0[2] - p2[2]).powi(2))
                    .sqrt())
                / 3.0) as f64;
            if cx > 1.9 {
                tip_edges.push(avg_e);
            } else if cx < 0.6 {
                belly_edges.push(avg_e);
            }
        }
    }
    assert!(!tip_edges.is_empty() && !belly_edges.is_empty());
    let tip_mean = tip_edges.iter().sum::<f64>() / tip_edges.len() as f64;
    let belly_mean = belly_edges.iter().sum::<f64>() / belly_edges.len() as f64;
    assert!(
        tip_mean < belly_mean,
        "expected smaller adaptive triangles at high-curvature tips ({tip_mean}) than belly ({belly_mean})"
    );
}

#[test]
fn surface_remesh_with_mask_keeps_masked_vertices_pinned() {
    let mut mesh = subdivided_grid(4, 2.0);
    let pin: Vec<VertexId> = boundary_loops(&mesh)
        .unwrap()
        .into_iter()
        .flat_map(|l| l.vertices)
        .collect();
    let pinned_pos: Vec<(VertexId, [f32; 3])> = pin
        .iter()
        .map(|&v| (v, mesh.position(v).unwrap()))
        .collect();
    let opts = SurfaceRemeshOptions::builder()
        .target_edge_length(0.25)
        .iterations(3)
        .lock_boundaries(false)
        .build();
    let rep = edit(&mut mesh, |tx| remesh_surface_with_mask(tx, &opts, &pin));
    assert!(rep.splits > 0);
    for (v, expected) in pinned_pos {
        assert!(mesh.is_vertex_alive(v));
        assert_eq!(mesh.position(v).unwrap(), expected);
    }
}

#[test]
fn simplify_and_simplify_with_backcompat_and_new_features() {
    let mut m1 = sphere_mesh(12, 24, 1.0);
    let mut m2 = Mesh::from_buffers(
        &m1.positions().iter().flatten().copied().collect::<Vec<_>>(),
        &m1.triangles().iter().flatten().copied().collect::<Vec<_>>(),
    )
    .unwrap();
    let legacy = SimplifyOptions {
        target_triangles: 200,
        preserve_boundary: true,
    };
    let ext = SimplifyExtendedOptions::from(legacy);
    let r1 = edit(&mut m1, |tx| simplify(tx, &legacy));
    let r2 = edit(&mut m2, |tx| simplify_with(tx, &ext));
    assert_eq!(r1, r2);
    assert_eq!(m1.positions(), m2.positions());
    assert_eq!(m1.triangles(), m2.triangles());

    // Error-bounded decimation stops within max_deviation and reports it accurately.
    let mut m3 = sphere_mesh(16, 32, 1.0);
    let max_dev = 0.015;
    let rep = edit(&mut m3, |tx| {
        simplify_with(
            tx,
            &DecimateOptions::builder()
                .target_triangles(20)
                .max_deviation(max_dev)
                .build(),
        )
    });
    assert!(rep.max_deviation_achieved <= max_dev + 1e-12);
    assert!(live_triangles(&m3) > 20);
}

#[test]
fn decimate_lock_creases_preserves_subdivided_box_silhouette() {
    let mut box_sub = subdivided_box(5);
    assert_eq!(live_triangles(&box_sub), 300);
    let opts = DecimateOptions::builder()
        .target_triangles(12)
        .lock_creases(true)
        .crease_angle_deg(30.0)
        .build();
    let rep = edit(&mut box_sub, |tx| simplify_with(tx, &opts));
    assert!(rep.collapsed > 0);
    assert!(live_triangles(&box_sub) <= 32);

    // Measure box extents remain intact [0, 1]^3 with clean manifold topology.
    let a = analyze(&box_sub, &AnalyzeOptions::default()).unwrap();
    assert_eq!(a.non_manifold_edges, 0);
    assert_eq!(a.orientation_conflicts, 0);
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in 0..box_sub.vertex_slots() as VertexId {
        if box_sub.is_vertex_alive(v) {
            let p = box_sub.position(v).unwrap();
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
    }
    for k in 0..3 {
        assert!(min[k].abs() < 1e-5, "min[{k}] shifted: {}", min[k]);
        assert!((max[k] - 1.0).abs() < 1e-5, "max[{k}] shifted: {}", max[k]);
    }

    // Minimal sharp box stays untouched when creases are locked.
    let mut sharp = sharp_box([0.0; 3], [1.0; 3]);
    let sharp_rep = edit(&mut sharp, |tx| {
        simplify_with(
            tx,
            &DecimateOptions::builder()
                .target_triangles(4)
                .lock_creases(true)
                .crease_angle_deg(30.0)
                .build(),
        )
    });
    assert_eq!(sharp_rep.collapsed, 0);
    assert_eq!(live_triangles(&sharp), 12);
}

#[test]
fn decimate_undo_and_determinism() {
    let base = sphere_mesh(12, 24, 1.0);
    let mut a = Mesh::from_buffers(
        &base
            .positions()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        &base
            .triangles()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut b = Mesh::from_buffers(
        &base
            .positions()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        &base
            .triangles()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let opts = DecimateOptions::builder()
        .target_ratio(0.35)
        .max_deviation(0.25)
        .lock_creases(true)
        .build();

    let mut tx_a = a.begin_edit();
    let rep_a = simplify_with(&mut tx_a, &opts).unwrap();
    let patch_a = tx_a.commit().unwrap();
    let rep_b = edit(&mut b, |tx| simplify_with(tx, &opts));

    assert_eq!(rep_a, rep_b);
    assert_eq!(a.positions(), b.positions());
    assert_eq!(a.triangles(), b.triangles());

    patch_a.revert(&mut a).unwrap();
    assert_eq!(a.positions(), base.positions());
    assert_eq!(a.triangles(), base.triangles());
}
