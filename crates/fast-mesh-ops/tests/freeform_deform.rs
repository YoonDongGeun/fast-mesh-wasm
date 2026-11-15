//! Integration tests for `fast_mesh_ops::deform` free-form deformation algorithms:
//! FFD (`ffd`), ARAP (`arap`), Elastostatic Kelvinlets (`kelvinlet`), and XPBD (`xpbd`)
//! including BVH narrow-phase self-collision avoidance and parallel determinism.

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_ops::deform::{
    ArapAnchor, ArapHandle, ArapOptions, ArapSolver, ArapWeights, FfdBasis, FfdFrame, FfdLattice,
    FfdOptions, FfdOutsideMode, FreeformOp, KelvinletLoad, KelvinletOptions, KelvinletScale,
    Symmetry, XpbdOptions, apply_arap, apply_ffd, apply_ffd_protected, apply_freeform,
    apply_freeform_protected, apply_kelvinlet, apply_kelvinlet_protected, apply_xpbd,
};
use fast_mesh_ops::remesh::Region;
use fast_mesh_ops::{AnalyzeOptions, OpsError, analyze, measure};
use fast_mesh_query::self_intersecting_pairs;
use fast_mesh_store::{Mesh, VertexId};
use std::f32::consts::{FRAC_PI_4, PI};

fn build_mesh_with_uvs(positions: Vec<f32>, indices: Vec<u32>, uvs: Vec<f32>) -> Mesh {
    let mut m = Mesh::from_buffers(&positions, &indices).unwrap();
    m.add_uvs(&uvs).unwrap();
    m
}

fn make_grid(nx: u32, nz: u32, width: f32, depth: f32) -> Mesh {
    let mut positions = Vec::with_capacity((nx * nz * 3) as usize);
    let mut uvs = Vec::with_capacity((nx * nz * 2) as usize);
    for iz in 0..nz {
        let vz = iz as f32 / (nz - 1) as f32;
        for ix in 0..nx {
            let ux = ix as f32 / (nx - 1) as f32;
            positions.push(ux * width);
            positions.push(0.0);
            positions.push(vz * depth);
            uvs.push(ux);
            uvs.push(vz);
        }
    }
    let mut indices = Vec::with_capacity(((nx - 1) * (nz - 1) * 6) as usize);
    for iz in 0..(nz - 1) {
        for ix in 0..(nx - 1) {
            let v00 = iz * nx + ix;
            let v10 = v00 + 1;
            let v01 = v00 + nx;
            let v11 = v01 + 1;
            indices.extend_from_slice(&[v00, v01, v10, v10, v01, v11]);
        }
    }
    build_mesh_with_uvs(positions, indices, uvs)
}

/// Builds a watertight subdivided rectangular bar `[0..lx] x [-ry..ry] x [-rz..rz]`.
fn make_bar(nx: u32, ny: u32, nz: u32, lx: f32, hy: f32, hz: f32) -> (Mesh, Vec<(usize, u32)>) {
    // Emit a regular lattice of boundary grid vertices mapped onto the 6 box faces via shared IDs.
    use std::collections::BTreeMap;
    let mut coord_to_id: BTreeMap<(u32, u32, u32), u32> = BTreeMap::new();
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut x_ring_map = Vec::new();

    let mut get_vid = |ix: u32, iy: u32, iz: u32| -> u32 {
        if let Some(&id) = coord_to_id.get(&(ix, iy, iz)) {
            return id;
        }
        let id = (positions.len() / 3) as u32;
        let x = (ix as f32 / nx as f32) * lx;
        let y = (iy as f32 / ny as f32 * 2.0 - 1.0) * hy;
        let z = (iz as f32 / nz as f32 * 2.0 - 1.0) * hz;
        positions.extend_from_slice(&[x, y, z]);
        uvs.push(x / lx);
        uvs.push(0.5 * (y / hy + 1.0));
        coord_to_id.insert((ix, iy, iz), id);
        x_ring_map.push((ix as usize, id));
        id
    };

    let mut indices = Vec::new();
    let mut add_quad = |a: u32, b: u32, c: u32, d: u32| {
        indices.extend_from_slice(&[a, b, c, a, c, d]);
    };

    // -Y and +Y faces
    for ix in 0..nx {
        for iz in 0..nz {
            let a = get_vid(ix, 0, iz);
            let b = get_vid(ix + 1, 0, iz);
            let c = get_vid(ix + 1, 0, iz + 1);
            let d = get_vid(ix, 0, iz + 1);
            add_quad(a, b, c, d);

            let e = get_vid(ix, ny, iz);
            let f = get_vid(ix + 1, ny, iz);
            let g = get_vid(ix + 1, ny, iz + 1);
            let h = get_vid(ix, ny, iz + 1);
            add_quad(e, h, g, f);
        }
    }
    // -Z and +Z faces
    for ix in 0..nx {
        for iy in 0..ny {
            let a = get_vid(ix, iy, 0);
            let b = get_vid(ix, iy + 1, 0);
            let c = get_vid(ix + 1, iy + 1, 0);
            let d = get_vid(ix + 1, iy, 0);
            add_quad(a, b, c, d);

            let e = get_vid(ix, iy, nz);
            let f = get_vid(ix, iy + 1, nz);
            let g = get_vid(ix + 1, iy + 1, nz);
            let h = get_vid(ix + 1, iy, nz);
            add_quad(e, h, g, f);
        }
    }
    // -X and +X endcaps
    for iy in 0..ny {
        for iz in 0..nz {
            let a = get_vid(0, iy, iz);
            let b = get_vid(0, iy, iz + 1);
            let c = get_vid(0, iy + 1, iz + 1);
            let d = get_vid(0, iy + 1, iz);
            add_quad(a, b, c, d);

            let e = get_vid(nx, iy, iz);
            let f = get_vid(nx, iy, iz + 1);
            let g = get_vid(nx, iy + 1, iz + 1);
            let h = get_vid(nx, iy + 1, iz);
            add_quad(e, h, g, f);
        }
    }

    (build_mesh_with_uvs(positions, indices, uvs), x_ring_map)
}

/// Builds a closed outward-oriented UV sphere with UV channel.
fn make_uv_sphere(stacks: u32, slices: u32, r: f32) -> Mesh {
    let mut positions = vec![0.0, r, 0.0];
    let mut uvs = vec![0.5, 0.0];
    for st in 1..stacks {
        let phi = PI * (st as f32) / (stacks as f32);
        let (sp, cp) = phi.sin_cos();
        for sl in 0..slices {
            let theta = 2.0 * PI * (sl as f32) / (slices as f32);
            let (st_th, ct_th) = theta.sin_cos();
            positions.extend_from_slice(&[r * sp * ct_th, r * cp, r * sp * st_th]);
            uvs.push(sl as f32 / slices as f32);
            uvs.push(st as f32 / stacks as f32);
        }
    }
    let south = (positions.len() / 3) as u32;
    positions.extend_from_slice(&[0.0, -r, 0.0]);
    uvs.extend_from_slice(&[0.5, 1.0]);

    let mut indices = Vec::new();
    for sl in 0..slices {
        let next = (sl + 1) % slices;
        indices.extend_from_slice(&[0, 1 + next, 1 + sl]);
    }
    for st in 0..(stacks - 2) {
        let r0 = 1 + st * slices;
        let r1 = 1 + (st + 1) * slices;
        for sl in 0..slices {
            let next = (sl + 1) % slices;
            indices.extend_from_slice(&[r0 + sl, r0 + next, r1 + next]);
            indices.extend_from_slice(&[r0 + sl, r1 + next, r1 + sl]);
        }
    }
    let last_ring = 1 + (stacks - 2) * slices;
    for sl in 0..slices {
        let next = (sl + 1) % slices;
        indices.extend_from_slice(&[last_ring + sl, last_ring + next, south]);
    }

    build_mesh_with_uvs(positions, indices, uvs)
}

/// Builds a connected U-folded strip: bottom shelf `[0..1]` at `y = 0`, tight bend to top flap hovering at `y = 0.22`.
fn make_folded_u_sheet(nx_arm: u32, nz: u32) -> (Mesh, Vec<u32>, Vec<u32>) {
    // Cross-section parameter s along the U shape (bottom arm -> semicircle bend -> top overhanging flap).
    let n_bend = 6u32;
    let total_cols = nx_arm + n_bend + nx_arm;
    let arm_len = 1.4f32;
    let gap_r = 0.14f32;
    let width_z = 1.0f32;

    let mut profile = Vec::with_capacity(total_cols as usize);
    for i in 0..nx_arm {
        let t = i as f32 / nx_arm as f32;
        profile.push((t * arm_len, 0.0f32));
    }
    for i in 0..=n_bend {
        let ang = -std::f32::consts::FRAC_PI_2 + PI * (i as f32 / n_bend as f32);
        let x = arm_len + gap_r * ang.cos();
        let y = gap_r + gap_r * ang.sin();
        profile.push((x, y));
    }
    for i in 1..nx_arm {
        let t = i as f32 / (nx_arm - 1) as f32;
        let x = arm_len - t * (arm_len - 0.15);
        profile.push((x, 2.0 * gap_r));
    }

    let n_prof = profile.len() as u32;
    let mut positions = Vec::with_capacity((n_prof * nz * 3) as usize);
    let mut uvs = Vec::with_capacity((n_prof * nz * 2) as usize);
    let mut bottom_pins = Vec::new();
    let mut flap_tip = Vec::new();

    for iz in 0..nz {
        let v = iz as f32 / (nz - 1) as f32;
        let z = v * width_z;
        for (ip, &(x, y)) in profile.iter().enumerate() {
            let vid = (positions.len() / 3) as u32;
            positions.extend_from_slice(&[x, y, z]);
            uvs.push(ip as f32 / (n_prof - 1) as f32);
            uvs.push(v);
            if ip < nx_arm as usize {
                bottom_pins.push(vid);
            }
            if ip == (n_prof - 1) as usize {
                flap_tip.push(vid);
            }
        }
    }

    let mut indices = Vec::new();
    for iz in 0..(nz - 1) {
        for ip in 0..(n_prof - 1) {
            let v00 = iz * n_prof + ip;
            let v10 = v00 + 1;
            let v01 = v00 + n_prof;
            let v11 = v01 + 1;
            indices.extend_from_slice(&[v00, v01, v10, v10, v01, v11]);
        }
    }

    (
        build_mesh_with_uvs(positions, indices, uvs),
        bottom_pins,
        flap_tip,
    )
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn mean_edge_length_rel_error(before: &Mesh, after: &Mesh) -> f32 {
    let p0 = before.positions();
    let p1 = after.positions();
    let mut rel_sum = 0.0f64;
    let mut count = 0usize;
    for t in 0..before.triangle_slots() as u32 {
        let Some([a, b, c]) = before.triangle(t) else {
            continue;
        };
        for &(u, v) in &[(a, b), (b, c), (c, a)] {
            let l0 = dist3(p0[u as usize], p0[v as usize]) as f64;
            if l0 > 1e-8 {
                let l1 = dist3(p1[u as usize], p1[v as usize]) as f64;
                rel_sum += ((l1 - l0).abs()) / l0;
                count += 1;
            }
        }
    }
    (rel_sum / count.max(1) as f64) as f32
}

fn count_self_intersections(mesh: &Mesh) -> (usize, Option<bool>) {
    let bvh = Bvh::build(mesh, &BuildOptions::default()).unwrap();
    let mut pairs = Vec::new();
    self_intersecting_pairs(mesh, &bvh, &mut pairs).unwrap();
    let analysis = analyze(
        mesh,
        &AnalyzeOptions::builder().self_intersection(&bvh).build(),
    )
    .unwrap();
    (pairs.len(), analysis.self_intersects)
}

#[test]
fn ffd_bernstein_and_bspline_identity_translation_and_uvs() {
    for basis in [FfdBasis::Bernstein, FfdBasis::CubicBSpline] {
        let mut mesh = make_grid(6, 6, 2.0, 2.0);
        let rest_pos = mesh.positions().to_vec();
        let uvs_before = mesh.uvs().unwrap().to_vec();

        // Identity cage
        let lattice_id = FfdLattice::with_mesh(&mesh, [5, 4, 5], basis, 0.1).unwrap();
        let opts_id = FfdOptions::new(lattice_id);
        let rep_id = {
            let mut tx = mesh.begin_edit();
            let r = apply_ffd(&mut tx, &opts_id).unwrap();
            tx.commit().unwrap();
            r
        };
        assert_eq!(rep_id.vertices_moved, 0);
        assert_eq!(mesh.positions(), rest_pos.as_slice());

        // Uniform translation reproduces shift identically for partition-of-unity bases.
        let mut lattice_tr = FfdLattice::with_mesh(&mesh, [5, 4, 5], basis, 0.1).unwrap();
        let delta = [0.4f32, -0.25, 0.75];
        for off in lattice_tr.offsets_mut() {
            *off = delta;
        }
        let opts_tr = FfdOptions::new(lattice_tr);
        let rep_tr = {
            let mut tx = mesh.begin_edit();
            let r = apply_ffd(&mut tx, &opts_tr).unwrap();
            tx.commit().unwrap();
            r
        };
        assert_eq!(rep_tr.vertices_moved, mesh.vertex_count());
        assert!((rep_tr.max_displacement - dist3([0.0; 3], delta)).abs() < 1e-4);
        for (before, after) in rest_pos.iter().zip(mesh.positions().iter()) {
            assert!(
                dist3(
                    [
                        before[0] + delta[0],
                        before[1] + delta[1],
                        before[2] + delta[2]
                    ],
                    *after
                ) < 1e-4
            );
        }
        assert_eq!(mesh.uvs().unwrap(), uvs_before.as_slice());
    }
}

#[test]
fn ffd_bend_custom_frame_region_protection_and_validation() {
    let mut mesh = make_grid(8, 8, 2.0, 2.0);
    let uvs_before = mesh.uvs().unwrap().to_vec();
    let rest_pos = mesh.positions().to_vec();

    // Rotated 45-deg custom orthonormal frame around Y.
    let (s, c) = FRAC_PI_4.sin_cos();
    let frame = FfdFrame::new(
        [1.0, 0.0, 1.0],
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        [2.0, 0.5, 2.0],
    )
    .unwrap();
    let mut lattice = FfdLattice::new([4, 2, 4], frame, FfdBasis::Bernstein).unwrap();
    // Push far i=3 plane upward cleanly along +Y -> monotonic ramp across the rotated axis.
    for j in 0..2 {
        for k in 0..4 {
            lattice.set_offset(3, j, k, [0.0, 1.2, 0.0]).unwrap();
        }
    }

    // Protection semantics matching crate::protect::with_protection: 0.0 = frozen, 1.0 = full move.
    let protected = vec![0 as VertexId, (mesh.vertex_count() - 1) as VertexId];
    let weights = vec![0.0f32, 1.0f32];
    let opts = FfdOptions::new(lattice).with_outside(FfdOutsideMode::Clamp);
    let rep = {
        let mut tx = mesh.begin_edit();
        let r = apply_ffd_protected(&mut tx, &opts, &protected, &weights).unwrap();
        tx.commit().unwrap();
        r
    };
    assert!(rep.vertices_moved > 0);
    assert_eq!(mesh.positions()[0], rest_pos[0]);
    assert!(mesh.positions()[mesh.vertex_count() - 1][1] > 0.1);
    assert_eq!(mesh.uvs().unwrap(), uvs_before.as_slice());

    // Region scoping & B-spline local support check
    let mut mesh_bs = make_grid(8, 8, 2.0, 2.0);
    let mut bs_lat =
        FfdLattice::with_mesh(&mesh_bs, [6, 3, 6], FfdBasis::CubicBSpline, 0.05).unwrap();
    for j in 0..3 {
        let p_target = bs_lat.rest_position(5, j, 5).unwrap();
        bs_lat
            .set_control_position(5, j, 5, [p_target[0], p_target[1] + 1.0, p_target[2]])
            .unwrap();
    }
    let bs_opts = FfdOptions::new(bs_lat).with_region(Region {
        center: [2.0, 0.0, 2.0],
        radius: 1.0,
    });
    {
        let mut tx = mesh_bs.begin_edit();
        apply_ffd(&mut tx, &bs_opts).unwrap();
        tx.commit().unwrap();
    }
    assert_eq!(mesh_bs.positions()[0], [0.0, 0.0, 0.0]);
    assert!(mesh_bs.positions()[mesh_bs.vertex_count() - 1][1] > 0.4);

    // Invalid inputs must reject cleanly and leave tx untouched.
    assert!(matches!(
        FfdFrame::from_aabb([1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        Err(OpsError::InvalidInput(_))
    ));
    assert!(matches!(
        FfdLattice::with_mesh(&mesh, [1, 2, 2], FfdBasis::Bernstein, 0.1),
        Err(OpsError::InvalidInput(_))
    ));
    let mut bad_lat = FfdLattice::with_mesh(&mesh, [2, 2, 2], FfdBasis::Bernstein, 0.1).unwrap();
    assert!(matches!(
        bad_lat.set_offset(0, 0, 0, [f32::NAN, 0.0, 0.0]),
        Err(OpsError::InvalidInput(_))
    ));
}

#[test]
fn arap_bar_bend_and_twist_preserves_detail_and_uvs() {
    let (mut bar_bend, x_ring) = make_bar(12, 2, 2, 4.0, 0.25, 0.25);
    let rest_bar = make_bar(12, 2, 2, 4.0, 0.25, 0.25).0;
    let uvs_before = bar_bend.uvs().unwrap().to_vec();

    let mut anchors = Vec::new();
    let mut handles = Vec::new();
    for &(ix, vid) in &x_ring {
        if ix == 0 {
            anchors.push(ArapAnchor::pin(vid));
        } else if ix == 12 {
            handles.push(
                ArapHandle::from_rigid(
                    &bar_bend,
                    vid,
                    [2.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    0.55,
                    [0.0, 0.0, 0.0],
                )
                .unwrap(),
            );
        }
    }

    let opts = ArapOptions::new()
        .with_anchors(anchors)
        .with_handles(handles.clone())
        .with_weights(ArapWeights::Cotangent)
        .with_solver(ArapSolver::Auto)
        .with_iterations(120)
        .with_tolerance(1e-5);
    let rep = {
        let mut tx = bar_bend.begin_edit();
        let r = apply_arap(&mut tx, &opts).unwrap();
        tx.commit().unwrap();
        r
    };
    assert!(
        rep.converged,
        "ARAP bend did not converge in {} iters, residual={}",
        rep.iterations_run, rep.final_residual
    );
    assert!(rep.free_vertices > 0);
    for h in &handles {
        assert!(dist3(bar_bend.positions()[h.vertex as usize], h.target) < 1e-4);
    }
    let bend_rel = mean_edge_length_rel_error(&rest_bar, &bar_bend);
    assert!(
        bend_rel < 0.06,
        "ARAP bend edge distortion too large: {bend_rel}"
    );
    assert_eq!(bar_bend.uvs().unwrap(), uvs_before.as_slice());

    // Twist test around X axis with PCG solver verification
    let (mut bar_twist, x_ring_tw) = make_bar(12, 2, 2, 4.0, 0.25, 0.25);
    let mut tw_anchors = Vec::new();
    let mut tw_handles = Vec::new();
    for &(ix, vid) in &x_ring_tw {
        if ix == 0 {
            tw_anchors.push(ArapAnchor::pin(vid));
        } else if ix == 12 {
            tw_handles.push(
                ArapHandle::from_axis_angle(
                    &bar_twist,
                    vid,
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    PI / 3.0,
                )
                .unwrap(),
            );
        }
    }
    let tw_opts = ArapOptions::new()
        .with_anchors(tw_anchors)
        .with_handles(tw_handles.clone())
        .with_solver(ArapSolver::Pcg)
        .with_iterations(220)
        .with_tolerance(1e-5);
    let tw_rep = {
        let mut tx = bar_twist.begin_edit();
        let r = apply_arap(&mut tx, &tw_opts).unwrap();
        tx.commit().unwrap();
        r
    };
    assert!(
        tw_rep.converged,
        "ARAP twist did not converge in {} iters, residual={}",
        tw_rep.iterations_run, tw_rep.final_residual
    );
    let tw_rel = mean_edge_length_rel_error(&rest_bar, &bar_twist);
    assert!(
        tw_rel < 0.07,
        "ARAP twist edge distortion too large: {tw_rel}"
    );
    // Mid-span vertex rotated smoothly (non-zero Y/Z rotation).
    let mid_pos: Vec<[f32; 3]> = x_ring_tw
        .iter()
        .filter(|&&(ix, _)| ix == 6)
        .map(|&(_, v)| bar_twist.positions()[v as usize])
        .collect();
    assert!(!mid_pos.is_empty());
    assert!(
        mid_pos
            .iter()
            .zip(x_ring_tw.iter().filter(|&&(ix, _)| ix == 6))
            .any(|(&p, &(_, v))| dist3(p, rest_bar.positions()[v as usize]) > 0.03)
    );
}

#[test]
fn arap_error_checks_isolated_unpinned_and_pinned() {
    let mut mesh = make_grid(4, 4, 1.0, 1.0);
    let snap = mesh.positions().to_vec();

    // No anchors/handles
    {
        let mut tx = mesh.begin_edit();
        let err = apply_arap(&mut tx, &ArapOptions::new());
        assert!(matches!(err, Err(OpsError::InvalidInput(_))));
        assert!(tx.commit().unwrap().is_empty());
    }

    // Two components (grid + isolated unpinned vertex) -> must fail without touching tx.
    let (mut pos_iso, tris_iso) = {
        let p = mesh
            .positions()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let t: Vec<u32> = (0..mesh.triangle_slots() as u32)
            .filter_map(|i| mesh.triangle(i))
            .flatten()
            .collect();
        (p, t)
    };
    let iso_vid = (pos_iso.len() / 3) as u32;
    pos_iso.extend_from_slice(&[5.0, 5.0, 5.0]);
    let mut mesh_iso = Mesh::from_buffers(&pos_iso, &tris_iso).unwrap();

    let opts_unpinned_iso = ArapOptions::new()
        .with_anchors(vec![ArapAnchor::pin(0)])
        .with_handles(vec![ArapHandle::new(15, [1.0, 0.5, 1.0])]);
    {
        let mut tx = mesh_iso.begin_edit();
        let err = apply_arap(&mut tx, &opts_unpinned_iso);
        assert!(matches!(err, Err(OpsError::InvalidInput(_))));
        assert!(tx.commit().unwrap().is_empty());
    }

    // When isolated vertex IS pinned, ARAP succeeds cleanly and leaves isolated vertex anchored.
    let opts_pinned_iso = ArapOptions::new()
        .with_anchors(vec![ArapAnchor::pin(0), ArapAnchor::pin(iso_vid)])
        .with_handles(vec![ArapHandle::new(15, [1.0, 0.5, 1.0])]);
    {
        let mut tx = mesh_iso.begin_edit();
        let rep = apply_arap(&mut tx, &opts_pinned_iso).unwrap();
        tx.commit().unwrap();
        assert_eq!(mesh_iso.positions()[iso_vid as usize], [5.0, 5.0, 5.0]);
        assert!(rep.vertices_moved > 0);
    }

    // Conflicting anchor & handle on same vertex -> rejected cleanly.
    let opts_conflict = ArapOptions::new()
        .with_anchors(vec![ArapAnchor::pin(0)])
        .with_handles(vec![ArapHandle::new(0, [0.0, 2.0, 0.0])]);
    {
        let mut tx = mesh.begin_edit();
        let err = apply_arap(&mut tx, &opts_conflict);
        assert!(matches!(err, Err(OpsError::InvalidInput(_))));
        assert!(tx.commit().unwrap().is_empty());
    }
    assert_eq!(mesh.positions(), snap.as_slice());
}

#[test]
fn kelvinlet_loads_scales_volume_preservation_symmetry_and_uvs() {
    // 1. Incompressible nu = 0.5 volume preservation under strong Pinch + Twist + Grab
    let base_sphere = make_uv_sphere(18, 24, 1.0);
    let vol0 = measure(&base_sphere).unwrap().volume.unwrap();
    let uvs0 = base_sphere.uvs().unwrap().to_vec();

    // Scale nu=0.0 expands vs shrinks, whereas nu=0.5 is zero-displacement exact.
    let mut sp_expand = make_uv_sphere(18, 24, 1.0);
    {
        let opts = KelvinletOptions::new([0.0, 0.0, 0.0], KelvinletLoad::Scale { magnitude: 1.5 })
            .with_epsilon(0.8)
            .with_nu(0.0)
            .with_mu(1.0);
        let mut tx = sp_expand.begin_edit();
        apply_kelvinlet(&mut tx, &opts).unwrap();
        tx.commit().unwrap();
    }
    assert!(measure(&sp_expand).unwrap().volume.unwrap() > vol0 * 1.05);

    let mut sp_incomp_scale = make_uv_sphere(18, 24, 1.0);
    {
        let opts = KelvinletOptions::new([0.0, 0.0, 0.0], KelvinletLoad::Scale { magnitude: 1.5 })
            .with_epsilon(0.8)
            .with_nu(0.5);
        let mut tx = sp_incomp_scale.begin_edit();
        let rep = apply_kelvinlet(&mut tx, &opts).unwrap();
        tx.commit().unwrap();
        assert_eq!(rep.vertices_moved, 0);
    }

    // Pinch + Grab at nu = 0.5 preserve enclosed volume tightly despite large deformation.
    let mut sp_pinch = make_uv_sphere(18, 24, 1.0);
    let rep_pinch = {
        let opts = KelvinletOptions::new(
            [0.0, 0.2, 0.0],
            KelvinletLoad::Pinch {
                force_dipole: [[1.2, 0.0, 0.0], [0.0, -0.6, 0.0], [0.0, 0.0, -0.6]],
            },
        )
        .with_epsilon(0.6)
        .with_nu(0.5)
        .with_mu(1.0);
        let mut tx = sp_pinch.begin_edit();
        let r = apply_kelvinlet(&mut tx, &opts).unwrap();
        tx.commit().unwrap();
        r
    };
    assert!(rep_pinch.max_displacement > 0.05);
    let vol_pinch = measure(&sp_pinch).unwrap().volume.unwrap();
    let rel_vol_err = ((vol_pinch - vol0) / vol0).abs();
    assert!(
        rel_vol_err < 0.015,
        "incompressible nu=0.5 Pinch volume drift too large: {rel_vol_err}"
    );
    assert_eq!(sp_pinch.uvs().unwrap(), uvs0.as_slice());

    // Twist swirl direction and near-zero radial volume drift
    let mut sp_twist = make_uv_sphere(14, 18, 1.0);
    {
        let opts = KelvinletOptions::new(
            [0.0, 0.0, 0.0],
            KelvinletLoad::Twist {
                torque: [0.0, 1.5, 0.0],
            },
        )
        .with_epsilon(0.8)
        .with_nu(0.5);
        let mut tx = sp_twist.begin_edit();
        let r = apply_kelvinlet(&mut tx, &opts).unwrap();
        tx.commit().unwrap();
        assert!(r.max_displacement > 0.05);
    }
    let vol_twist = measure(&sp_twist).unwrap().volume.unwrap();
    assert!(((vol_twist - vol0) / vol0).abs() < 0.02);

    // BiScale / TriScale localized far-field falloff vs Single Grab
    let far_probe = [3.0f32, 0.0, 0.0];
    let eval_disp_at = |scale: KelvinletScale| -> f32 {
        let mut pts = Mesh::from_buffers(&[0.0, 0.0, 0.0, far_probe[0], 0.0, 0.0], &[]).unwrap();
        let opts = KelvinletOptions::new(
            [0.0, 0.0, 0.0],
            KelvinletLoad::Grab {
                force: [0.0, 1.0, 0.0],
            },
        )
        .with_epsilon(0.5)
        .with_scale(scale);
        let mut tx = pts.begin_edit();
        apply_kelvinlet(&mut tx, &opts).unwrap();
        tx.commit().unwrap();
        dist3(far_probe, pts.positions()[1])
    };
    let d_single = eval_disp_at(KelvinletScale::Single);
    let d_bi = eval_disp_at(KelvinletScale::BiScale { alpha: 1.5 });
    let d_tri = eval_disp_at(KelvinletScale::TriScale);
    assert!(
        d_bi < 0.3 * d_single,
        "BiScale {d_bi} should decay faster than Single {d_single}"
    );
    assert!(
        d_tri < d_bi,
        "TriScale {d_tri} should decay faster than BiScale {d_bi}"
    );

    // Symmetry + Protection
    let mut grid_sym = make_grid(5, 5, 2.0, 2.0);
    // Shift grid so x in [-1, 1]
    {
        let ids: Vec<VertexId> = (0..grid_sym.vertex_slots() as u32).collect();
        let shifted: Vec<[f32; 3]> = grid_sym
            .positions()
            .iter()
            .map(|p| [p[0] - 1.0, p[1], p[2] - 1.0])
            .collect();
        let mut tx = grid_sym.begin_edit();
        tx.set_positions(&ids, &shifted).unwrap();
        tx.commit().unwrap();
    }
    let mut sym = Symmetry::NONE;
    sym.x = true;
    let opts_sym = KelvinletOptions::new(
        [0.5, 0.0, 0.0],
        KelvinletLoad::Grab {
            force: [0.0, 1.0, 0.0],
        },
    )
    .with_epsilon(0.5)
    .with_symmetry(sym);
    {
        let mut tx = grid_sym.begin_edit();
        // Protection weight 0.0 freezes vertex 0 cleanly at its rest position.
        apply_kelvinlet_protected(&mut tx, &opts_sym, &[0], &[0.0]).unwrap();
        tx.commit().unwrap();
    }
    // For iz = 2 (center row): vertex idx 10 is x = -1, vertex idx 14 is x = +1
    assert!((grid_sym.positions()[10][1] - grid_sym.positions()[14][1]).abs() < 1e-5);
    assert_eq!(grid_sym.positions()[0], [-1.0, 0.0, -1.0]);
    // Also verify partial 0.5 weight cuts displacement in half vs un-protected symmetric partner:
    // iz=0: vertex 0 (x=-1, protected 0.0), vertex 4 (x=+1, free).
    assert!(grid_sym.positions()[4][1] > 0.01);

    // Validation failure
    let mut dummy = make_grid(2, 2, 1.0, 1.0);
    let mut tx = dummy.begin_edit();
    assert!(matches!(
        apply_kelvinlet(
            &mut tx,
            &KelvinletOptions::new([0.0; 3], KelvinletLoad::Grab { force: [0.0; 3] }).with_nu(0.8)
        ),
        Err(OpsError::InvalidInput(_))
    ));
    assert!(tx.commit().unwrap().is_empty());
}

#[test]
fn xpbd_cloth_sag_and_closed_volume_preservation() {
    let mut cloth = make_grid(7, 7, 1.0, 1.0);
    let uvs_before = cloth.uvs().unwrap().to_vec();
    let pins: Vec<(VertexId, [f32; 3])> = (0..7)
        .map(|ix| (ix, cloth.positions()[ix as usize]))
        .collect();

    let opts = XpbdOptions::new()
        .with_substeps(10)
        .with_iterations(8)
        .with_dt(0.08)
        .with_stretch_compliance(0.0)
        .with_bend_compliance(0.05)
        .with_gravity([0.0, -9.81, 0.0])
        .with_pins(pins.clone());
    let rep = {
        let mut tx = cloth.begin_edit();
        let r = apply_xpbd(&mut tx, &opts).unwrap();
        tx.commit().unwrap();
        r
    };
    assert!(rep.vertices_moved > 0);
    for &(v, tgt) in &pins {
        assert_eq!(cloth.positions()[v as usize], tgt);
    }
    // Free edge at z = 1 sags downward noticeably under gravity.
    assert!(cloth.positions()[45][1] < -0.01);
    assert_eq!(cloth.uvs().unwrap(), uvs_before.as_slice());

    // Closed sphere with volume compliance preserves enclosed volume under handle pull.
    let mut sphere = make_uv_sphere(8, 12, 1.0);
    let v0 = measure(&sphere).unwrap().volume.unwrap();
    let v_opts = XpbdOptions::new()
        .with_substeps(8)
        .with_iterations(8)
        .with_dt(0.05)
        .with_stretch_compliance(1e-3)
        .with_volume_compliance(0.0)
        .with_handles(vec![(0, [0.0, 1.3, 0.0], 0.0)]);
    {
        let mut tx = sphere.begin_edit();
        apply_xpbd(&mut tx, &v_opts).unwrap();
        tx.commit().unwrap();
    }
    let v1 = measure(&sphere).unwrap().volume.unwrap();
    assert!(((v1 - v0) / v0).abs() < 0.03);
}

#[test]
fn xpbd_self_collision_prevents_folding_intersections() {
    let (base, bottom_pins, flap_tip) = make_folded_u_sheet(8, 6);
    assert_eq!(count_self_intersections(&base).0, 0);

    let pins: Vec<(VertexId, [f32; 3])> = bottom_pins
        .iter()
        .map(|&v| (v, base.positions()[v as usize]))
        .collect();
    // Push the upper flap downward across the bottom shelf plane (y = -0.18).
    let handles: Vec<(VertexId, [f32; 3], f32)> = flap_tip
        .iter()
        .map(|&v| {
            let p = base.positions()[v as usize];
            (v, [p[0], -0.18, p[2]], 1e-4)
        })
        .collect();

    // Run 1: self_collision = false -> flap penetrates straight through the resting sheet!
    let (mut without_sc, _, _) = make_folded_u_sheet(8, 6);
    let opts_no_sc = XpbdOptions::new()
        .with_substeps(20)
        .with_iterations(10)
        .with_dt(0.1)
        .with_stretch_compliance(0.0)
        .with_bend_compliance(0.05)
        .with_damping(0.08)
        .with_gravity([0.0, -25.0, 0.0])
        .with_pins(pins.clone())
        .with_handles(handles.clone())
        .with_self_collision(false, 0.05);
    let rep_no_sc = {
        let mut tx = without_sc.begin_edit();
        let r = apply_xpbd(&mut tx, &opts_no_sc).unwrap();
        tx.commit().unwrap();
        r
    };
    let (pairs_no_sc, hit_no_sc) = count_self_intersections(&without_sc);
    assert_eq!(rep_no_sc.self_collision_contacts_resolved, 0);
    assert!(
        pairs_no_sc > 0 && hit_no_sc == Some(true),
        "expected self-intersections without collision avoidance, got {pairs_no_sc}"
    );

    // Run 2: self_collision = true -> BVH narrow-phase prevents flap from crossing resting sheet!
    let (mut with_sc, _, _) = make_folded_u_sheet(8, 6);
    let opts_sc = XpbdOptions {
        self_collision: true,
        collision_thickness: 0.05,
        ..opts_no_sc
    };
    let rep_sc = {
        let mut tx = with_sc.begin_edit();
        let r = apply_xpbd(&mut tx, &opts_sc).unwrap();
        tx.commit().unwrap();
        r
    };
    let (pairs_sc, hit_sc) = count_self_intersections(&with_sc);
    assert!(rep_sc.self_collision_contacts_resolved > 0);
    assert_eq!(
        pairs_sc, 0,
        "self_collision: true should prevent self-intersections (had {pairs_sc}, without sc had {pairs_no_sc})"
    );
    assert_eq!(hit_sc, Some(false));
}

#[test]
fn freeform_unified_and_parallel_determinism() {
    // 48x48 grid = 2304 vertices (> PAR_THRESHOLD = 2048) triggers parallel chunks when `parallel` is enabled.
    let run_all = || -> [Vec<[f32; 3]>; 4] {
        let mut m_ffd = make_grid(48, 48, 4.0, 4.0);
        let mut lat =
            FfdLattice::with_mesh(&m_ffd, [3, 2, 3], FfdBasis::CubicBSpline, 0.1).unwrap();
        lat.set_offset(2, 1, 2, [0.2, 0.5, 0.1]).unwrap();
        {
            let mut tx = m_ffd.begin_edit();
            apply_freeform(&mut tx, &FreeformOp::Ffd(FfdOptions::new(lat))).unwrap();
            tx.commit().unwrap();
        }

        let mut m_kel = make_grid(48, 48, 4.0, 4.0);
        {
            let op = FreeformOp::Kelvinlet(
                KelvinletOptions::new(
                    [2.0, 0.0, 2.0],
                    KelvinletLoad::Grab {
                        force: [0.0, 0.8, 0.0],
                    },
                )
                .with_epsilon(0.9),
            );
            let mut tx = m_kel.begin_edit();
            apply_freeform_protected(&mut tx, &op, &[0], &[1.0]).unwrap();
            tx.commit().unwrap();
        }

        // Smaller mesh for ARAP + XPBD repeated bit-identical determinism check
        let mut m_arap = make_grid(12, 12, 2.0, 2.0);
        {
            let op = FreeformOp::Arap(
                ArapOptions::new()
                    .with_anchors(vec![ArapAnchor::pin(0), ArapAnchor::pin(11)])
                    .with_handles(vec![ArapHandle::new(143, [2.0, 0.6, 2.0])])
                    .with_iterations(8),
            );
            let mut tx = m_arap.begin_edit();
            apply_freeform(&mut tx, &op).unwrap();
            tx.commit().unwrap();
        }

        let mut m_xpbd = make_grid(10, 10, 1.0, 1.0);
        {
            let op = FreeformOp::Xpbd(
                XpbdOptions::new()
                    .with_substeps(4)
                    .with_iterations(4)
                    .with_pins(vec![(0, [0.0, 0.0, 0.0]), (9, [1.0, 0.0, 0.0])])
                    .with_gravity([0.0, -5.0, 0.0])
                    .with_self_collision(true, 0.03),
            );
            let mut tx = m_xpbd.begin_edit();
            apply_freeform(&mut tx, &op).unwrap();
            tx.commit().unwrap();
        }

        [
            m_ffd.positions().to_vec(),
            m_kel.positions().to_vec(),
            m_arap.positions().to_vec(),
            m_xpbd.positions().to_vec(),
        ]
    };

    let r1 = run_all();
    let r2 = run_all();
    for (v1, v2) in r1.iter().zip(r2.iter()) {
        assert_eq!(v1.len(), v2.len());
        for (p, q) in v1.iter().zip(v2.iter()) {
            assert_eq!(
                [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()],
                [q[0].to_bits(), q[1].to_bits(), q[2].to_bits()]
            );
        }
    }
}
