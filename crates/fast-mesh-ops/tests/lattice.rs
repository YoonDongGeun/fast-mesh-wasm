//! Lattice infill properties (#265), checked on every result: closed (no
//! boundary loop), 2-manifold (no non-manifold edge, no pinch vertex),
//! consistently and outward wound, free of degenerate and duplicate
//! triangles, of coincident vertices and of self-intersections (with the
//! engine for triangles that share no vertex, and with exact predicates for
//! those that do); the lattice stays inside the input; and the share of the
//! region its walls fill is what the minimal surface of its kind gives.
//!
//! For walls of thickness `t` in cells of edge `a`, that share is
//! `A x + (pi / 6) chi x³` with `x = t / a` (the tube formula), where the
//! published areas per cell `A` are 3.0915 (gyroid), 2.3451 (Schwarz P) and
//! 3.8378 (diamond), and the Euler characteristics `chi` are -8, -4 and -16.

use std::f64::consts::PI;

use fast_mesh_fields::{SignData, SignMode, signed_distances};
use fast_mesh_ops::append::AppendSource;
use fast_mesh_ops::boolean::{HollowOptions, hollow_mesh};
use fast_mesh_ops::lattice::{
    LATTICE_BYTES_PER_TRIANGLE, LATTICE_BYTES_PER_VOXEL, LatticeKind, LatticeOptions,
    LatticeOptionsBuilder, LatticeRegion, LatticeReport, lattice_fill, lattice_plan,
    lattice_preview,
};
use fast_mesh_ops::primitives::{TorusOptions, torus};
use fast_mesh_ops::{OpsError, measure};
use fast_mesh_query::{ClosestOptions, closest_point};
use fast_mesh_store::{Mesh, Transaction, VertexId};

mod common;
use common::{bvh_of, check_surface, cube, merged, mesh_of, moved, sphere, turned};

/// The live geometry of `mesh`, compact, in its stored frame.
fn soup(mesh: &Mesh) -> AppendSource {
    let mut remap = vec![u32::MAX; mesh.vertex_slots()];
    let mut out = AppendSource::default();
    for t in 0..mesh.triangle_slots() as u32 {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        out.triangles.push(tri.map(|v| {
            if remap[v as usize] == u32::MAX {
                remap[v as usize] = out.positions.len() as u32;
                out.positions.push(mesh.positions()[v as usize]);
            }
            remap[v as usize]
        }));
    }
    out
}

fn edit<R>(mesh: &mut Mesh, f: impl FnOnce(&mut Transaction) -> R) -> R {
    let mut tx = mesh.begin_edit();
    let r = f(&mut tx);
    tx.commit().unwrap();
    r
}

fn fill(mesh: &mut Mesh, options: &LatticeOptions) -> Result<LatticeReport, OpsError> {
    edit(mesh, |tx| lattice_fill(tx, options))
}

fn options(kind: LatticeKind, cell: f32, wall: f32) -> LatticeOptionsBuilder {
    LatticeOptions::builder()
        .kind(kind)
        .cell_size(cell)
        .wall_thickness(wall)
}

/// The published volume fraction of walls of thickness `t` in cells `a` long.
fn expected(kind: LatticeKind, a: f32, t: f32) -> f64 {
    let x = f64::from(t) / f64::from(a);
    let (area, chi) = match kind {
        LatticeKind::Gyroid => (3.0915, -8.0),
        LatticeKind::SchwarzP => (2.3451, -4.0),
        LatticeKind::Diamond => (3.8378, -16.0),
    };
    area * x + PI / 6.0 * chi * x * x * x
}

/// The signed distances of `points` to `input`, positive outside.
fn outside(input: &AppendSource, points: &[[f32; 3]]) -> Vec<f32> {
    let mut mesh = mesh_of(input);
    let bvh = bvh_of(&mesh);
    let sign = SignData::build(&mut mesh, &bvh, SignMode::Auto).unwrap();
    let mut d = vec![0.0f32; points.len()];
    signed_distances(&mesh, &bvh, &sign, points, f32::INFINITY, &mut d).unwrap();
    d
}

fn deepest_and_farthest(input: &AppendSource, points: &[[f32; 3]]) -> (f32, f32) {
    outside(input, points)
        .into_iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), d| {
            (lo.min(d), hi.max(d))
        })
}

#[test]
fn the_volume_fraction_is_that_of_the_minimal_surface() {
    // Two cells along each side of a box: whole periods, so the fraction does
    // not depend on where the cells start.
    let part = cube([2.0; 3], [0.0; 3]);
    for kind in LatticeKind::ALL {
        for (wall, tolerance) in [(0.1, 0.02), (0.15, 0.02), (0.22, 0.02)] {
            let mut mesh = mesh_of(&part);
            let opts = options(kind, 1.0, wall).min_fragment(0.0).build();
            let r = fill(&mut mesh, &opts).unwrap();
            let out = soup(&mesh);
            let what = format!("{kind:?} {wall}");
            let (_, volume) = check_surface(&out, &what);
            let want = expected(kind, 1.0, wall);
            let got = volume / 8.0;
            assert!((got / want - 1.0).abs() < tolerance, "{what}: {got} {want}");
            assert!((r.solid_volume - volume).abs() < 1e-3, "{what}");
            assert!((r.plan.region_volume - 8.0).abs() < 1e-3);
            assert!((r.plan.solid_fraction - want).abs() < 1e-6);
            assert_eq!(r.triangles, out.triangles.len());
            assert_eq!(r.vertices, out.positions.len());
            // The estimate the plan makes of the triangles.
            let estimate = r.plan.triangles as f64 / r.triangles as f64;
            assert!((0.8..1.25).contains(&estimate), "{what}: {estimate}");
            // Inside, and up to the surface.
            let h = r.plan.voxel_size;
            let (deep, far) = deepest_and_farthest(&part, &out.positions);
            assert!(far < 0.01 * h, "{what}: a vertex {far} outside");
            assert!(deep < -0.8, "{what}: {deep}");
        }
    }
}

/// An L-shaped prism: convex and reentrant edges.
fn ell() -> AppendSource {
    let outline = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]];
    let mut out = AppendSource::default();
    for z in 0..2 {
        out.positions
            .extend(outline.map(|[x, y]| [x as f32, y as f32, z as f32]));
    }
    for t in [[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]] {
        out.triangles.push([t[0], t[2], t[1]]);
        out.triangles.push(t.map(|v| v + 6));
    }
    for i in 0..6u32 {
        let j = (i + 1) % 6;
        out.triangles.push([i, j, j + 6]);
        out.triangles.push([i, j + 6, i + 6]);
    }
    out
}

fn ring() -> AppendSource {
    torus(&TorusOptions {
        major_radius: 0.8,
        minor_radius: 0.4,
        ..TorusOptions::default()
    })
    .unwrap()
}

/// Shapes about 2 across: smooth, sharp, reentrant, thin, off the grid.
fn shapes() -> Vec<(&'static str, AppendSource)> {
    vec![
        ("sphere", sphere(1.0, 4)),
        ("box", cube([2.0, 1.5, 1.2], [0.3, -0.2, 0.1])),
        ("turned box", turned(cube([1.6; 3], [-0.8; 3]), 0.7)),
        ("torus", ring()),
        ("ell", ell()),
        ("turned ell", turned(ell(), 1.1)),
        ("plate", cube([2.0, 1.5, 0.15], [0.0; 3])),
        (
            "two parts",
            merged(&[sphere(0.8, 3), moved(sphere(0.6, 3), [1.6, 0.0, 0.0])]),
        ),
    ]
}

#[test]
fn open_lattices_are_clean_and_stay_inside() {
    for (n, (name, part)) in shapes().into_iter().enumerate() {
        for (i, kind) in LatticeKind::ALL.into_iter().enumerate() {
            let cell = [0.6, 0.6, 0.8][i];
            let wall = [0.1, 0.13, 0.1][i];
            // Auto, or as coarse as it goes: two voxels to the wall.
            let voxel = ((n + i) % 2 == 0).then_some(wall / 2.0);
            let mut mesh = mesh_of(&part);
            let opts = options(kind, cell, wall).voxel_size(voxel).build();
            let what = format!("{name} {kind:?} {voxel:?}");
            let r = fill(&mut mesh, &opts).unwrap_or_else(|e| panic!("{what}: {e}"));
            let out = soup(&mesh);
            let (_, volume) = check_surface(&out, &what);
            assert!((r.solid_volume - volume).abs() < 1e-3 * volume, "{what}");
            let h = r.plan.voxel_size;
            let (_, far) = deepest_and_farthest(&part, &out.positions);
            assert!(far <= 0.02 * h, "{what}: a vertex {} voxels out", far / h);
            if ["sphere", "box", "turned box"].contains(&name) {
                // A few cells thick: close to the fraction of whole cells.
                let share = volume / r.plan.region_volume;
                let want = expected(kind, cell, wall);
                assert!((share / want - 1.0).abs() < 0.2, "{what}: {share} {want}");
            }
        }
    }
}

#[test]
fn a_shell_keeps_the_surface_and_is_one_solid_with_the_walls() {
    for (n, (name, part)) in shapes().into_iter().enumerate() {
        if name == "plate" {
            // Thinner than two shells.
            continue;
        }
        for (i, kind) in LatticeKind::ALL.into_iter().enumerate() {
            let cell = [0.6, 0.6, 0.8][i];
            let wall = [0.1, 0.13, 0.1][i];
            let shell: f32 = [0.12, 0.1, 0.16][i];
            let voxel = ((n + i) % 2 == 0).then_some(shell.min(wall) / 2.0);
            let mut mesh = mesh_of(&part);
            let opts = options(kind, cell, wall)
                .shell_thickness(shell)
                .voxel_size(voxel)
                .build();
            let what = format!("{name} {kind:?} {voxel:?}");
            let r = fill(&mut mesh, &opts).unwrap_or_else(|e| panic!("{what}: {e}"));
            // The surface itself is as it was, bit for bit.
            let (nv, nt) = (part.positions.len(), part.triangles.len());
            assert_eq!(mesh.positions()[..nv], part.positions[..], "{what}");
            for (t, tri) in part.triangles.iter().enumerate() {
                assert_eq!(mesh.triangle(t as u32), Some(*tri), "{what}");
            }
            assert_eq!(mesh.triangle_count(), nt + r.triangles, "{what}");
            let out = soup(&mesh);
            let (_, volume) = check_surface(&out, &what);
            assert!((r.solid_volume - volume).abs() < 1e-3 * volume, "{what}");
            // The void keeps the shell's thickness from it.
            let h = r.plan.voxel_size;
            let (_, far) = deepest_and_farthest(&part, &mesh.positions()[nv..]);
            assert!(
                far <= -shell + 0.02 * h,
                "{what}: {far} from the surface, shell {shell}"
            );
            assert!(volume < r.plan.region_volume, "{what}");
        }
    }
}

#[test]
fn a_shelled_sphere_weighs_what_shell_and_walls_do() {
    let part = sphere(1.0, 4);
    let solid = measure(&mesh_of(&part)).unwrap().signed_volume;
    for kind in LatticeKind::ALL {
        let mut mesh = mesh_of(&part);
        let opts = options(kind, 0.4, 0.06).shell_thickness(0.12).build();
        let r = fill(&mut mesh, &opts).unwrap();
        let inside = 4.0 / 3.0 * PI * 0.88f64.powi(3);
        let want = solid - inside + inside * expected(kind, 0.4, 0.06);
        let got = measure(&mesh).unwrap().signed_volume;
        assert!((got / want - 1.0).abs() < 0.02, "{kind:?}: {got} {want}");
        assert!((r.solid_volume / got - 1.0).abs() < 1e-3);
    }
}

#[test]
fn the_cavity_of_a_hollowed_mesh_fills_from_its_shell() {
    let part = sphere(1.0, 4);
    for kind in LatticeKind::ALL {
        let mut mesh = mesh_of(&part);
        let wall = 0.12;
        let h = edit(&mut mesh, |tx| {
            hollow_mesh(
                tx,
                &HollowOptions {
                    wall_thickness: wall,
                    voxel_size: Some(0.03),
                    ..HollowOptions::default()
                },
            )
        })
        .unwrap();
        assert_eq!(h.cavities_created, 1);
        let hollow = measure(&mesh).unwrap().signed_volume;
        let opts = options(kind, 0.4, 0.06).build();
        let plan = lattice_plan(&mesh, &opts).unwrap();
        assert_eq!(plan.region, LatticeRegion::Cavity);
        assert!((plan.region_volume / h.cavity_volume - 1.0).abs() < 1e-3);
        let r = fill(&mut mesh, &opts).unwrap();
        assert_eq!(r.plan, plan);
        let what = format!("{kind:?} cavity");
        let out = soup(&mesh);
        let (_, volume) = check_surface(&out, &what);
        assert!((volume - hollow - r.solid_volume).abs() < 1e-3 * volume);
        // The outer surface is as it was; the cavity's is gone.
        let (nv, nt) = (part.positions.len(), part.triangles.len());
        assert_eq!(mesh.positions()[..nv], part.positions[..]);
        assert_eq!(out.triangles.len(), nt + r.triangles);
        for (t, tri) in part.triangles.iter().enumerate() {
            assert_eq!(mesh.triangle(t as u32), Some(*tri));
        }
        // The shell is as thick as it was, to a fraction of a voxel.
        let added: Vec<[f32; 3]> = (nv..mesh.vertex_slots())
            .filter(|&v| mesh.is_vertex_alive(v as VertexId))
            .map(|v| mesh.positions()[v])
            .collect();
        assert_eq!(added.len(), r.vertices);
        let (_, far) = deepest_and_farthest(&part, &added);
        assert!(far < -wall + 0.02, "{far}");
        // A whole number of cells would give exactly the fraction; here the
        // cavity cuts them.
        let share = r.solid_volume / r.plan.region_volume;
        let want = expected(kind, 0.4, 0.06);
        assert!((share / want - 1.0).abs() < 0.12, "{what}: {share} {want}");

        // The shell in one go is the same solid.
        let mut direct = mesh_of(&part);
        let opts = options(kind, 0.4, 0.06).shell_thickness(wall).build();
        fill(&mut direct, &opts).unwrap();
        let at_once = measure(&direct).unwrap().signed_volume;
        assert!((at_once / volume - 1.0).abs() < 0.02, "{at_once} {volume}");
    }
}

#[test]
fn cavities_with_solids_in_them_and_several_parts_fill() {
    let reversed = |mut s: AppendSource| {
        for t in &mut s.triangles {
            t.swap(1, 2);
        }
        s
    };
    // Two hollow balls and a solid one; in the first cavity floats a ball.
    let ball = |r: f64, x: f64| moved(sphere(r, 3), [x, 0.0, 0.0]);
    let part = merged(&[
        ball(1.0, 0.0),
        reversed(ball(0.8, 0.0)),
        ball(0.2, 0.0),
        ball(1.0, 3.0),
        reversed(ball(0.8, 3.0)),
        ball(0.5, 5.0),
    ]);
    let mut mesh = mesh_of(&part);
    let opts = options(LatticeKind::Gyroid, 0.5, 0.08).build();
    let before = measure(&mesh).unwrap().signed_volume;
    let r = fill(&mut mesh, &opts).unwrap();
    assert_eq!(r.plan.region, LatticeRegion::Cavity);
    let (big, small) = (
        measure(&mesh_of(&ball(0.8, 0.0))).unwrap().signed_volume,
        measure(&mesh_of(&ball(0.2, 0.0))).unwrap().signed_volume,
    );
    assert!((r.plan.region_volume - (2.0 * big - small)).abs() < 1e-3);
    let (_, volume) = check_surface(&soup(&mesh), "nested balls");
    assert!((volume - (before + r.solid_volume)).abs() < 2e-3 * volume);
    let n = ball(1.0, 0.0).positions.len();
    // The solid ball in the first cavity and the cavities' surfaces are gone,
    // the three outer surfaces are not.
    let alive = |v: usize| mesh.is_vertex_alive(v as VertexId);
    assert!((0..n).all(alive));
    assert!(!(n..3 * n).any(alive));
    assert!((3 * n..4 * n).all(alive));
    assert!(!(4 * n..5 * n).any(alive));
    assert!((5 * n..6 * n).all(alive));

    // The inside of the same mesh: its material, with the cavities left.
    let mut mesh = mesh_of(&part);
    let opts = options(LatticeKind::Diamond, 0.3, 0.05)
        .region(LatticeRegion::Inside)
        .build();
    let r = fill(&mut mesh, &opts).unwrap();
    assert_eq!(r.plan.region, LatticeRegion::Inside);
    check_surface(&soup(&mesh), "the inside of nested balls");

    // A loose part wound inward is not a cavity.
    let part = merged(&[ball(1.0, 0.0), reversed(ball(0.2, 4.0))]);
    let plan = lattice_plan(&mesh_of(&part), &opts).unwrap();
    assert_eq!(plan.region, LatticeRegion::Inside);
    let cavity = LatticeOptions {
        region: LatticeRegion::Cavity,
        ..opts
    };
    assert_eq!(
        lattice_plan(&mesh_of(&part), &cavity).map(|_| ()),
        Err(OpsError::InvalidInput(
            "lattice found no cavity enclosed by the mesh"
        ))
    );
}

#[test]
fn a_thin_wall_around_a_cavity_needs_fine_voxels() {
    let mut inner = sphere(0.98, 4);
    for t in &mut inner.triangles {
        t.swap(1, 2);
    }
    let part = merged(&[sphere(1.0, 4), inner]);
    let coarse = options(LatticeKind::SchwarzP, 0.5, 0.08);
    let mut mesh = mesh_of(&part);
    assert_eq!(
        fill(&mut mesh, &coarse.clone().build()).map(|_| ()),
        Err(OpsError::InvalidInput(
            "lattice voxels are too coarse for the wall around the cavity; use a smaller voxel \
             size"
        ))
    );
    assert_eq!(mesh.triangle_count(), part.triangles.len());
    // A shell on top keeps the void away from the wall.
    let shelled = coarse.clone().shell_thickness(0.06).build();
    fill(&mut mesh, &shelled).unwrap();
    check_surface(&soup(&mesh), "shelled cavity");
    // And so do voxels finer than the wall.
    let mut mesh = mesh_of(&part);
    let fine = coarse.voxel_size(Some(0.012)).build();
    fill(&mut mesh, &fine).unwrap();
    check_surface(&soup(&mesh), "fine cavity");
}

#[test]
fn lengths_are_measured_after_the_scale() {
    // Stored half as large along x and ten times along z.
    let part = cube([1.5, 3.0, 0.3], [2.0, -1.0, 0.0]);
    let opts = options(LatticeKind::Gyroid, 1.0, 0.12)
        .scale([2.0, 1.0, 10.0])
        .min_fragment(0.0)
        .build();
    let mut mesh = mesh_of(&part);
    let r = fill(&mut mesh, &opts).unwrap();
    assert!((r.plan.region_volume - 27.0).abs() < 1e-3);
    let mut out = soup(&mesh);
    let (_, far) = deepest_and_farthest(&part, &out.positions);
    assert!(far < 1e-3, "{far}");
    for p in &mut out.positions {
        *p = [2.0 * p[0], p[1], 10.0 * p[2]];
    }
    let (_, volume) = check_surface(&out, "scaled");
    let want = 27.0 * expected(LatticeKind::Gyroid, 1.0, 0.12);
    assert!((volume / want - 1.0).abs() < 0.02, "{volume} {want}");
    assert!((r.solid_volume - volume).abs() < 1e-2);

    // The same lattice as the scaled part has without a scale.
    let mut plain = mesh_of(&cube([3.0; 3], [0.0; 3]));
    let unscaled = LatticeOptions {
        scale: [1.0; 3],
        ..opts
    };
    let s = fill(&mut plain, &unscaled).unwrap();
    assert_eq!((s.vertices, s.triangles), (r.vertices, r.triangles));
}

#[test]
fn far_from_the_origin_the_voxels_must_stay_resolvable() {
    let opts = options(LatticeKind::Gyroid, 0.5, 0.08).build();
    let mut mesh = mesh_of(&cube([1.0; 3], [100.0, -50.0, 0.0]));
    fill(&mut mesh, &opts).unwrap();
    check_surface(&soup(&mesh), "far");
    let mut lost = mesh_of(&cube([1.0; 3], [10_000.0, 0.0, 0.0]));
    assert_eq!(
        fill(&mut lost, &opts).map(|_| ()),
        Err(OpsError::InvalidInput(
            "lattice voxel size is too small for f32 positions this far from the origin"
        ))
    );
}

#[test]
fn the_plan_comes_first_and_big_grids_are_refused() {
    let part = cube([2.0, 1.0, 1.0], [0.0; 3]);
    let mesh = mesh_of(&part);
    let opts = options(LatticeKind::Gyroid, 0.5, 0.0625)
        .voxel_size(Some(0.03125))
        .build();
    let plan = lattice_plan(&mesh, &opts).unwrap();
    // 2 / 0.03125 = 64 cells, and two empty voxels on each side.
    assert_eq!(plan.dims, [64 + 4 + 1, 32 + 5, 32 + 5]);
    assert_eq!(plan.voxels, 69 * 37 * 37);
    assert_eq!(
        plan.bytes,
        plan.voxels * LATTICE_BYTES_PER_VOXEL + plan.triangles * LATTICE_BYTES_PER_TRIANGLE
    );
    assert!(plan.fits && !plan.cropped);
    assert_eq!(plan.region, LatticeRegion::Inside);
    assert_eq!(plan.wall_thickness, 0.0625);
    let r = fill(&mut mesh_of(&part), &opts).unwrap();
    assert_eq!(r.plan, plan);

    let limited = LatticeOptions {
        max_voxels: plan.voxels as usize - 1,
        ..opts
    };
    let plan = lattice_plan(&mesh, &limited).unwrap();
    assert!(!plan.fits);
    let mut same = mesh_of(&part);
    assert_eq!(
        fill(&mut same, &limited).map(|_| ()),
        Err(OpsError::InvalidInput("lattice grid exceeds max_voxels"))
    );
    assert_eq!(soup_in_order(&same), part);

    // Without a voxel size: 2.5 voxels to a wall, 16 to a cell, 2 to a shell.
    let auto = |o: LatticeOptionsBuilder| lattice_plan(&mesh, &o.build()).unwrap().voxel_size;
    assert_eq!(auto(options(LatticeKind::Gyroid, 0.5, 0.05)), 0.02);
    assert_eq!(auto(options(LatticeKind::Gyroid, 0.5, 0.15)), 0.03125);
    let shelled = options(LatticeKind::Gyroid, 0.5, 0.05).shell_thickness(0.03);
    assert_eq!(auto(shelled), 0.015);
}

#[test]
fn previews_are_small_and_leave_the_mesh_alone() {
    let part = sphere(1.0, 4);
    let mesh = mesh_of(&part);
    let opts = options(LatticeKind::Gyroid, 0.4, 0.03).build();
    let full = lattice_plan(&mesh, &opts).unwrap();

    // The full grid, if it fits.
    let all = lattice_preview(&mesh, &opts, full.voxels as usize).unwrap();
    let mut filled = mesh_of(&part);
    fill(&mut filled, &opts).unwrap();
    assert_eq!(all.walls, soup_in_order(&filled));
    assert!(!all.plan.cropped);

    // A coarser one draws the walls two voxels thick.
    let small = lattice_preview(&mesh, &opts, 200_000).unwrap();
    assert!(small.plan.voxels <= 200_000 && !small.plan.cropped);
    assert!(small.plan.voxel_size > full.voxel_size);
    assert_eq!(small.plan.wall_thickness, 2.0 * small.plan.voxel_size);
    check_surface(&small.walls, "small preview");
    let (_, far) = deepest_and_farthest(&part, &small.walls.positions);
    assert!(far < 0.02 * small.plan.voxel_size);
    assert!((bounds(&small.walls)[0] + 1.0).abs() < 0.1);

    // And one too small for the cells shows the middle.
    let middle = lattice_preview(&mesh, &opts, 30_000).unwrap();
    assert!(middle.plan.voxels <= 30_000 && middle.plan.cropped && middle.plan.fits);
    assert!((middle.plan.voxel_size / 0.04 - 1.0).abs() < 1e-6);
    check_surface(&middle.walls, "cropped preview");
    assert!(bounds(&middle.walls)[0] > -0.8);

    // With a shell, the walls start at its inner face.
    let shelled = LatticeOptions {
        shell_thickness: 0.2,
        ..opts
    };
    let inner = lattice_preview(&mesh, &shelled, 200_000).unwrap();
    check_surface(&inner.walls, "shelled preview");
    let (_, far) = deepest_and_farthest(&part, &inner.walls.positions);
    assert!((far + 0.2).abs() < 0.1 * inner.plan.voxel_size, "{far}");
    // Nothing is left: an empty preview.
    let thick = LatticeOptions {
        shell_thickness: 1.5,
        ..opts
    };
    let none = lattice_preview(&mesh, &thick, 200_000).unwrap();
    assert!(none.walls.triangles.is_empty());
    assert_eq!(soup_in_order(&mesh), part);
}

/// The live geometry in vertex order.
fn soup_in_order(mesh: &Mesh) -> AppendSource {
    let mut remap = vec![u32::MAX; mesh.vertex_slots()];
    let mut out = AppendSource::default();
    for v in 0..mesh.vertex_slots() {
        if mesh.is_vertex_alive(v as VertexId) {
            remap[v] = out.positions.len() as u32;
            out.positions.push(mesh.positions()[v]);
        }
    }
    out.triangles = (0..mesh.triangle_slots() as u32)
        .filter_map(|t| mesh.triangle(t))
        .map(|t| t.map(|v| remap[v as usize]))
        .collect();
    out
}

/// The smallest coordinate along each axis.
fn bounds(s: &AppendSource) -> [f32; 3] {
    s.positions.iter().fold([f32::INFINITY; 3], |m, p| {
        std::array::from_fn(|k| m[k].min(p[k]))
    })
}

#[test]
fn walls_are_as_thick_as_asked() {
    // Straight through the walls of an open lattice from points on the
    // middle surface: the distance to the result is half the thickness.
    let part = cube([2.0; 3], [-1.0; 3]);
    for kind in LatticeKind::ALL {
        for wall in [0.1, 0.25] {
            let mut mesh = mesh_of(&part);
            let opts = options(kind, 1.0, wall).voxel_size(Some(0.025)).build();
            fill(&mut mesh, &opts).unwrap();
            let out = mesh_of(&soup(&mesh));
            let bvh = bvh_of(&out);
            let k = 2.0 * PI;
            let f = |p: [f64; 3]| {
                let [x, y, z] = p.map(|c| k * c);
                match kind {
                    LatticeKind::Gyroid => {
                        x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos()
                    }
                    LatticeKind::SchwarzP => x.cos() + y.cos() + z.cos(),
                    LatticeKind::Diamond => {
                        x.sin() * y.sin() * z.sin()
                            + x.sin() * y.cos() * z.cos()
                            + x.cos() * y.sin() * z.cos()
                            + x.cos() * y.cos() * z.sin()
                    }
                }
            };
            let (mut sum, mut n, mut worst) = (0.0, 0, 0.0f64);
            for i in 0..400u32 {
                // On a vertical line, the first crossing.
                let (x, y) = (
                    0.7 * (f64::from(i) * 0.618_034).fract() - 0.35,
                    0.7 * (f64::from(i) * 0.754_878).fract() - 0.35,
                );
                let (mut a, mut b) = (-0.35, -0.35 + 0.01);
                while b < 0.35 && (f([x, y, a]) < 0.0) == (f([x, y, b]) < 0.0) {
                    (a, b) = (b, b + 0.01);
                }
                if b >= 0.35 {
                    continue;
                }
                for _ in 0..40 {
                    let m = 0.5 * (a + b);
                    if (f([x, y, a]) < 0.0) == (f([x, y, m]) < 0.0) {
                        a = m;
                    } else {
                        b = m;
                    }
                }
                let p = [x, y, a].map(|c| c as f32);
                let hit = closest_point(&out, &bvh, p, &ClosestOptions::default())
                    .unwrap()
                    .unwrap();
                let off = f64::from(hit.distance) / (0.5 * f64::from(wall)) - 1.0;
                worst = worst.max(off.abs());
                sum += off;
                n += 1;
            }
            assert!(n > 100, "{kind:?}: {n}");
            assert!(
                (sum / f64::from(n)).abs() < 0.02,
                "{kind:?} {wall}: {sum} {n}"
            );
            assert!(worst < 0.08, "{kind:?} {wall}: {worst}");
        }
    }
}

#[test]
fn a_uv_seam_does_not_open_the_result() {
    // A box whose faces have their own corners, set apart by their UVs.
    let part = cube([2.0; 3], [0.0; 3]);
    let mut split = AppendSource::default();
    let mut uvs: Vec<f32> = Vec::new();
    for (t, tri) in part.triangles.iter().enumerate() {
        let base = split.positions.len() as u32;
        for (k, &v) in tri.iter().enumerate() {
            split.positions.push(part.positions[v as usize]);
            uvs.extend([t as f32, k as f32]);
        }
        split.triangles.push([base, base + 1, base + 2]);
    }
    for shell in [0.0, 0.15] {
        let mut mesh = mesh_of(&split);
        mesh.add_uvs(&uvs).unwrap();
        let opts = options(LatticeKind::SchwarzP, 0.5, 0.08)
            .shell_thickness(shell)
            .build();
        let r = fill(&mut mesh, &opts).unwrap();
        assert!(r.uvs_dropped && mesh.uvs().is_none());
        check_surface(&soup(&mesh), "seams");
        let mut welded = mesh_of(&part);
        let s = fill(&mut welded, &opts).unwrap();
        assert_eq!(s.triangles, r.triangles);
        assert!(!s.uvs_dropped);
    }
}

#[test]
fn bad_input_is_refused_and_leaves_the_mesh() {
    let part = sphere(1.0, 3);
    let opts = options(LatticeKind::Gyroid, 0.4, 0.06);
    let refused = |part: &AppendSource, opts: LatticeOptions, msg: &'static str| {
        let mut mesh = mesh_of(part);
        assert_eq!(
            fill(&mut mesh, &opts).map(|_| ()),
            Err(OpsError::InvalidInput(msg))
        );
        assert_eq!(soup_in_order(&mesh), *part, "{msg}");
    };
    let mut holed = part.clone();
    holed.triangles.pop();
    let mut reversed = part.clone();
    for t in &mut reversed.triangles {
        t.swap(1, 2);
    }
    let mut flipped = part.clone();
    flipped.triangles[7].swap(0, 1);
    let closed = "lattice needs a closed, consistently wound manifold mesh";
    refused(&holed, opts.clone().build(), closed);
    refused(&flipped, opts.clone().build(), closed);
    refused(
        &reversed,
        opts.clone().build(),
        "lattice needs a closed mesh that encloses a volume and is wound outward",
    );
    refused(
        &AppendSource::default(),
        opts.clone().build(),
        "lattice needs a closed mesh with triangles",
    );
    refused(
        &part,
        opts.clone().region(LatticeRegion::Cavity).build(),
        "lattice found no cavity enclosed by the mesh",
    );
    refused(
        &part,
        opts.clone().shell_thickness(1.2).build(),
        "lattice leaves no open space; use thinner walls, a thinner shell or larger cells",
    );
    refused(
        &part,
        opts.clone().min_fragment(1e6).build(),
        "lattice leaves no walls; use smaller cells or thicker walls",
    );
    refused(
        &part,
        opts.clone().voxel_size(Some(0.04)).build(),
        "lattice walls are thinner than two voxels; use a smaller voxel size or thicker walls",
    );
    refused(
        &part,
        options(LatticeKind::Gyroid, 0.2, 0.06)
            .voxel_size(Some(0.03))
            .build(),
        "lattice cells are shorter than eight voxels; use a smaller voxel size or larger cells",
    );
    refused(
        &part,
        opts.clone()
            .voxel_size(Some(0.03))
            .shell_thickness(0.05)
            .build(),
        "lattice shell is thinner than two voxels; use a smaller voxel size or a thicker shell",
    );
    let d = LatticeOptions::default();
    for (opts, msg) in [
        (
            LatticeOptions {
                cell_size: 0.0,
                ..d
            },
            "lattice cell size must be positive and finite",
        ),
        (
            LatticeOptions {
                wall_thickness: f32::NAN,
                ..d
            },
            "lattice wall thickness must be positive and finite",
        ),
        (
            LatticeOptions {
                wall_thickness: 3.4,
                ..d
            },
            "lattice walls must not be thicker than a third of the cell size",
        ),
        (
            LatticeOptions {
                shell_thickness: -1.0,
                ..d
            },
            "lattice shell thickness must be finite and not negative",
        ),
        (
            LatticeOptions {
                voxel_size: Some(0.0),
                ..d
            },
            "lattice voxel size must be positive and finite",
        ),
        (
            LatticeOptions {
                max_voxels: 10,
                ..d
            },
            "lattice max_voxels must be at least 64",
        ),
        (
            LatticeOptions {
                min_fragment: -0.1,
                ..d
            },
            "lattice min_fragment must be finite and not negative",
        ),
        (
            LatticeOptions {
                scale: [1.0, 0.0, 1.0],
                ..d
            },
            "lattice scale must be positive and finite",
        ),
    ] {
        refused(&part, opts, msg);
        assert!(lattice_plan(&mesh_of(&part), &opts).is_err());
    }
}

#[test]
fn loose_pieces_are_left_out() {
    // A coarse lattice in a small part leaves bits of wall at its surface.
    let part = turned(cube([1.0, 0.8, 0.9], [-0.5; 3]), 0.5);
    let count = |min: f32| {
        let mut mesh = mesh_of(&part);
        let opts = options(LatticeKind::SchwarzP, 0.45, 0.05)
            .min_fragment(min)
            .build();
        let r = fill(&mut mesh, &opts).unwrap();
        let (parts, _) = check_surface(&soup(&mesh), "pieces");
        (parts, r.fragments)
    };
    let (all, none) = count(0.0);
    let (kept, dropped) = count(0.05);
    assert_eq!(none, 0);
    assert_eq!(all, kept + dropped);
    assert!(kept >= 1);
}

#[test]
fn coarse_random_lattices_are_clean() {
    // As coarse as the options allow, on parts at random angles: many cells
    // with several sheets through one voxel.
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut unit = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for run in 0..24 {
        let kind = LatticeKind::ALL[run % 3];
        let cell = 0.3 + 0.3 * unit() as f32;
        let wall = cell * (0.25 + 0.08 * unit() as f32);
        let voxel = (cell / 8.0).min(wall / 2.0);
        let part = turned(
            match run % 4 {
                0 => cube([1.4, 1.0, 1.2], [-0.6; 3]),
                1 => sphere(0.8, 3),
                2 => ell(),
                _ => ring(),
            },
            6.0 * unit(),
        );
        let shell = if run % 2 == 0 { 0.0 } else { 2.0 * voxel };
        let opts = options(kind, cell, wall)
            .voxel_size(Some(voxel))
            .shell_thickness(shell)
            .min_fragment(if run % 5 == 0 { 0.0 } else { 0.01 })
            .build();
        let mut mesh = mesh_of(&part);
        match fill(&mut mesh, &opts) {
            Ok(_) => {
                check_surface(&soup(&mesh), &format!("random {run}: {opts:?}"));
            }
            Err(e) => assert!(
                matches!(e, OpsError::InvalidInput(m) if m.starts_with("lattice leaves no")),
                "{run}: {e}"
            ),
        }
    }
}
