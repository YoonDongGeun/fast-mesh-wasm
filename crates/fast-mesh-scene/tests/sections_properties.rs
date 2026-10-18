//! Property test for sections: after any sequence of plane moves, placement
//! changes, vertex moves near and far from the plane, triangle additions,
//! removals and edge flips, undo, redo, cancelled drags, mode switches,
//! compactions, selection changes and fill toggles, every section entry must
//! be current after `update_sections` and equal a from-scratch slice of the
//! same mesh with the same plane and placement. Results that changed must be
//! reported in `sections_changed`.
//!
//! The scene holds a closed UV sphere and an open wavy terrain, built from
//! `f64` coordinates around 1000 so that both have a non-zero origin and the
//! section path's origin handling is exercised. Edits that cannot reach the
//! plane leave entries current without a recompute, so this test checks that
//! rule against slicing from scratch.
//!
//! The reference uses public API only: it maps the world plane into the
//! mesh's local frame the way the scene documents (the plane origin through
//! `Scene::world_to_local`, the axes through the transposed rotation and
//! re-orthonormalised), slices the scene's own mesh with a freshly built
//! BVH, and maps the points back through `Scene::local_to_world`. Topology
//! (edge keys, triangles, flags, fill triangles) is compared exactly, and
//! coordinates within `1e-9` of their magnitude.
//!
//! Seeds: `PROPTEST_RNG_SEED` is honoured; without it the seed is fixed, so
//! that CI runs are deterministic.

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::{Slice, contour_length, signed_area};
use fast_mesh_scene::{
    Fill, MeshId, MeshOptions, Placement, Plane, Scene, SceneError, SceneOptions, SectionContour,
    SectionDesc, SectionId, SectionResult, SectionStatus, UpdateMode,
};
use fast_mesh_store::{Mesh, TriangleId, VertexId};
use proptest::prelude::*;
use proptest::test_runner::RngSeed;

/// The seed used when `PROPTEST_RNG_SEED` is not set.
const DEFAULT_SEED: u64 = 0x5ec7_1015;

/// `PROPTEST_CASES` (else `cases`) and `PROPTEST_RNG_SEED` (else a fixed
/// seed). `with_cases` alone would ignore the environment variables.
fn config(cases: u32) -> ProptestConfig {
    let base = ProptestConfig::default();
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(cases);
    let rng_seed = match base.rng_seed {
        RngSeed::Random => RngSeed::Fixed(DEFAULT_SEED),
        fixed => fixed,
    };
    ProptestConfig {
        cases,
        rng_seed,
        ..base
    }
}

/// Coordinate tolerance, relative to the coordinates' magnitude (about
/// 1000).
const TOL: f64 = 1e-9 * 1100.0;

/// Where the meshes sit in the world.
const SPHERE_CENTRE: [f64; 3] = [1000.2, -999.8, 1000.1];
const TERRAIN_CORNER: [f64; 3] = [999.6, -1000.4, 1000.0];
/// A point between them, near which section planes pass.
const MIDDLE: [f64; 3] = [1000.05, -999.95, 1000.05];

/// A closed UV sphere of radius 0.5, outward wound.
fn sphere() -> Mesh {
    let (stacks, slices, r) = (10u32, 20u32, 0.5);
    let c = SPHERE_CENTRE;
    let mut pos = vec![c[0], c[1], c[2] + r];
    for i in 1..stacks {
        let theta = std::f64::consts::PI * f64::from(i) / f64::from(stacks);
        for j in 0..slices {
            let phi = std::f64::consts::TAU * f64::from(j) / f64::from(slices);
            let (rs, rc) = (r * theta.sin(), r * theta.cos());
            pos.extend([c[0] + rs * phi.cos(), c[1] + rs * phi.sin(), c[2] + rc]);
        }
    }
    pos.extend([c[0], c[1], c[2] - r]);
    let south = 1 + (stacks - 1) * slices;
    let ring = |i: u32, j: u32| 1 + (i - 1) * slices + j % slices;
    let mut idx = Vec::new();
    for j in 0..slices {
        idx.extend([0, ring(1, j), ring(1, j + 1)]);
        for i in 1..stacks - 1 {
            let (a, b, c, d) = (
                ring(i, j),
                ring(i + 1, j),
                ring(i + 1, j + 1),
                ring(i, j + 1),
            );
            idx.extend([a, b, c, a, c, d]);
        }
        idx.extend([ring(stacks - 1, j), south, ring(stacks - 1, j + 1)]);
    }
    Mesh::from_buffers_f64(&pos, &idx).unwrap()
}

/// An open grid, 12 x 12 squares of 0.1, with a wave in z.
fn terrain() -> Mesh {
    let n = 12u32;
    let o = TERRAIN_CORNER;
    let mut pos = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            let (fx, fy) = (f64::from(x) * 0.1, f64::from(y) * 0.1);
            let z = 0.08 * (fx * 4.0).sin() * (fy * 3.0).cos();
            pos.extend([o[0] + fx, o[1] + fy, o[2] + z]);
        }
    }
    let mut idx = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            let c = a + n + 1;
            idx.extend([a, a + 1, c, a + 1, c + 1, c]);
        }
    }
    Mesh::from_buffers_f64(&pos, &idx).unwrap()
}

#[derive(Debug, Clone)]
enum Op {
    /// Moves a section's plane: a normal and an offset from `MIDDLE`.
    Plane(bool, [i8; 3], [i8; 3]),
    /// A placement with a small turn about z and a small translation.
    Place(bool, i8, [i8; 3]),
    /// Moves one of the vertices closest to section 0's plane.
    MoveNear(bool, u32, [i8; 3]),
    /// Moves any vertex a little.
    MoveFar(bool, u32, [i8; 3]),
    RemoveTriangle(bool, u32),
    /// Splits a triangle at its centroid.
    Split(bool, u32),
    /// Flips the edge between a triangle and a neighbour.
    Flip(bool, u32, u8),
    /// Adds a vertex and a triangle on two alive vertices.
    AddTriangle(bool, u32, u32, [i8; 3]),
    /// A drag with a vertex edit, cancelled.
    CancelledDrag(bool, u32),
    Undo,
    Redo,
    SetMode(bool, u8),
    Compact(bool),
    /// Selects meshes by bit mask (bit 0 the sphere, bit 1 the terrain).
    Select(bool, u8),
    Fill(bool, bool),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        2 => (any::<bool>(), any::<[i8; 3]>(), any::<[i8; 3]>())
            .prop_map(|(s, n, o)| Op::Plane(s, n, o)),
        2 => (any::<bool>(), any::<i8>(), any::<[i8; 3]>())
            .prop_map(|(m, d, t)| Op::Place(m, d, t)),
        4 => (any::<bool>(), any::<u32>(), any::<[i8; 3]>())
            .prop_map(|(m, v, d)| Op::MoveNear(m, v, d)),
        3 => (any::<bool>(), any::<u32>(), any::<[i8; 3]>())
            .prop_map(|(m, v, d)| Op::MoveFar(m, v, d)),
        1 => (any::<bool>(), any::<u32>()).prop_map(|(m, t)| Op::RemoveTriangle(m, t)),
        1 => (any::<bool>(), any::<u32>()).prop_map(|(m, t)| Op::Split(m, t)),
        2 => (any::<bool>(), any::<u32>(), any::<u8>()).prop_map(|(m, t, e)| Op::Flip(m, t, e)),
        1 => (any::<bool>(), any::<u32>(), any::<u32>(), any::<[i8; 3]>())
            .prop_map(|(m, p, q, x)| Op::AddTriangle(m, p, q, x)),
        1 => (any::<bool>(), any::<u32>()).prop_map(|(m, v)| Op::CancelledDrag(m, v)),
        2 => Just(Op::Undo),
        1 => Just(Op::Redo),
        1 => (any::<bool>(), 0u8..3).prop_map(|(m, k)| Op::SetMode(m, k)),
        1 => any::<bool>().prop_map(Op::Compact),
        1 => (any::<bool>(), 0u8..4).prop_map(|(s, k)| Op::Select(s, k)),
        1 => (any::<bool>(), any::<bool>()).prop_map(|(s, f)| Op::Fill(s, f)),
    ]
}

fn update_mode(k: u8) -> UpdateMode {
    [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual][usize::from(k) % 3]
}

/// `i8` as a fraction in `[-1, 1]`.
fn unit(x: i8) -> f64 {
    f64::from(x) / 127.0
}

/// A rotation by `deg` degrees about z, then a translation.
fn turned(deg: f64, t: [f64; 3]) -> Placement {
    let (s, c) = deg.to_radians().sin_cos();
    Placement {
        rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        translation: t,
    }
}

/// A plane through `MIDDLE + offset` with normal `n` (z when `n` is zero).
fn plane_of(n: [i8; 3], offset: [i8; 3]) -> Plane {
    let origin = std::array::from_fn(|k| MIDDLE[k] + unit(offset[k]) * 0.4);
    let normal = if n == [0; 3] {
        [0.0, 0.0, 1.0]
    } else {
        n.map(f64::from)
    };
    Plane::new(origin, normal).unwrap()
}

fn alive_vertices(scene: &Scene, m: MeshId) -> Vec<VertexId> {
    let mesh = scene.mesh(m).unwrap();
    (0..mesh.vertex_slots() as VertexId)
        .filter(|&v| mesh.is_vertex_alive(v))
        .collect()
}

fn alive_triangles(scene: &Scene, m: MeshId) -> Vec<TriangleId> {
    let mesh = scene.mesh(m).unwrap();
    (0..mesh.triangle_slots() as TriangleId)
        .filter(|&t| mesh.is_triangle_alive(t))
        .collect()
}

/// Picks one of `items` by `k`, if there are any.
fn pick<T: Copy>(items: &[T], k: u32) -> Option<T> {
    (!items.is_empty()).then(|| items[k as usize % items.len()])
}

fn move_vertex(scene: &mut Scene, m: MeshId, v: VertexId, d: [f32; 3]) {
    scene
        .edit(m, |tx| {
            let p = tx.mesh().position(v).unwrap();
            tx.set_positions(&[v], &[std::array::from_fn(|k| p[k] + d[k])])
        })
        .unwrap();
}

/// The scene, its two meshes and two sections, and what the test asked each
/// section to be (the scene has no getter for it).
struct World {
    scene: Scene,
    meshes: [MeshId; 2],
    sections: [SectionId; 2],
    descs: [SectionDesc; 2],
}

/// Applies one operation. Operations that do not apply (nothing to undo, no
/// such neighbour) are skipped; a mesh never drops below five triangles.
fn apply(w: &mut World, op: &Op) {
    let meshes = w.meshes;
    let which = |m: &bool| if *m { meshes[0] } else { meshes[1] };
    let section = |s: &bool| usize::from(!*s);
    match op {
        Op::Plane(s, n, o) => {
            let s = section(s);
            let plane = plane_of(*n, *o);
            w.scene.set_section_plane(w.sections[s], plane).unwrap();
            w.descs[s].plane = plane;
        }
        Op::Place(m, deg, t) => {
            let t = t.map(|x| unit(x) * 0.1);
            w.scene
                .set_placement(which(m), turned(unit(*deg) * 20.0, t))
                .unwrap();
        }
        Op::MoveNear(m, k, d) => {
            let m = which(m);
            let plane = w.descs[0].plane;
            let mesh = w.scene.mesh(m).unwrap();
            let mut near: Vec<(f64, VertexId)> = alive_vertices(&w.scene, m)
                .into_iter()
                .map(|v| {
                    let p = mesh.position(v).unwrap().map(f64::from);
                    let q = w.scene.local_to_world(m, p).unwrap();
                    (plane.signed_distance(q).abs(), v)
                })
                .collect();
            near.sort_by(|a, b| a.partial_cmp(b).unwrap());
            near.truncate(12);
            let near: Vec<VertexId> = near.into_iter().map(|(_, v)| v).collect();
            if let Some(v) = pick(&near, *k) {
                move_vertex(&mut w.scene, m, v, d.map(|x| (unit(x) * 0.05) as f32));
            }
        }
        Op::MoveFar(m, v, d) => {
            let m = which(m);
            if let Some(v) = pick(&alive_vertices(&w.scene, m), *v) {
                move_vertex(&mut w.scene, m, v, d.map(|x| (unit(x) * 0.01) as f32));
            }
        }
        Op::RemoveTriangle(m, t) => {
            let m = which(m);
            let tris = alive_triangles(&w.scene, m);
            if tris.len() > 4
                && let Some(t) = pick(&tris, *t)
            {
                w.scene.edit(m, |tx| tx.remove_triangles(&[t])).unwrap();
            }
        }
        Op::Split(m, t) => {
            let m = which(m);
            if let Some(t) = pick(&alive_triangles(&w.scene, m), *t) {
                w.scene
                    .edit(m, |tx| {
                        let [i, j, k] = tx.mesh().triangle(t).unwrap();
                        let ps = [i, j, k].map(|v| tx.mesh().position(v).unwrap());
                        let c = std::array::from_fn(|d| (ps[0][d] + ps[1][d] + ps[2][d]) / 3.0);
                        let v = tx.add_vertices(&[c])?[0];
                        tx.remove_triangles(&[t])?;
                        tx.add_triangles(&[[i, j, v], [j, k, v], [k, i, v]])
                    })
                    .unwrap();
            }
        }
        Op::Flip(m, t, e) => {
            let m = which(m);
            let Some(t) = pick(&alive_triangles(&w.scene, m), *t) else {
                return;
            };
            let mesh = w.scene.mesh(m).unwrap();
            let tri = mesh.triangle(t).unwrap();
            let r = usize::from(*e % 3);
            // Triangle (a, b, c) and its neighbour (b, a, d) across edge ab
            // become (a, d, c) and (d, b, c).
            let (a, b, c) = (tri[r], tri[(r + 1) % 3], tri[(r + 2) % 3]);
            let other = alive_triangles(&w.scene, m).into_iter().find_map(|u| {
                let o = mesh.triangle(u).unwrap();
                (0..3).find_map(|s| {
                    (u != t && o[s] == b && o[(s + 1) % 3] == a).then_some((u, o[(s + 2) % 3]))
                })
            });
            let Some((u, d)) = other else { return };
            if d == c || a == b || a == c || b == c || d == a || d == b {
                return;
            }
            w.scene
                .edit(m, |tx| tx.set_triangles(&[t, u], &[[a, d, c], [d, b, c]]))
                .unwrap();
        }
        Op::AddTriangle(m, p, q, x) => {
            let m = which(m);
            let alive = alive_vertices(&w.scene, m);
            let (Some(p), Some(q)) = (pick(&alive, *p), pick(&alive, *q)) else {
                return;
            };
            if p == q {
                return;
            }
            w.scene
                .edit(m, |tx| {
                    let (pp, pq) = (
                        tx.mesh().position(p).unwrap(),
                        tx.mesh().position(q).unwrap(),
                    );
                    let x =
                        std::array::from_fn(|k| (pp[k] + pq[k]) * 0.5 + (unit(x[k]) * 0.1) as f32);
                    let v = tx.add_vertices(&[x])?;
                    tx.add_triangles(&[[v[0], p, q]])
                })
                .unwrap();
        }
        Op::CancelledDrag(m, v) => {
            let m = which(m);
            w.scene.begin_step();
            w.scene
                .set_placement(m, turned(3.0, [0.03, 0.02, -0.04]))
                .unwrap();
            if let Some(v) = pick(&alive_vertices(&w.scene, m), *v) {
                move_vertex(&mut w.scene, m, v, [0.0, 0.0, 0.1]);
            }
            w.scene.cancel_step().unwrap();
        }
        Op::Undo => {
            w.scene.undo().unwrap();
        }
        Op::Redo => {
            w.scene.redo().unwrap();
        }
        Op::SetMode(m, k) => {
            w.scene.set_mode(which(m), update_mode(*k)).unwrap();
        }
        Op::Compact(m) => {
            w.scene.compact(which(m)).unwrap();
        }
        Op::Select(s, k) => {
            let s = section(s);
            let meshes: Vec<MeshId> = (0..2)
                .filter(|b| k >> b & 1 == 1)
                .map(|b| w.meshes[b])
                .collect();
            w.scene
                .set_section_meshes(w.sections[s], meshes.clone())
                .unwrap();
            w.descs[s].meshes = meshes;
        }
        Op::Fill(s, f) => {
            let s = section(s);
            w.scene.set_section_fill(w.sections[s], *f).unwrap();
            w.descs[s].fill = *f;
        }
    }
}

/// `update_sections`, bringing stale Manual BVHs up to date when it asks.
fn update_all(scene: &mut Scene) {
    for _ in 0..8 {
        match scene.update_sections() {
            Ok(()) => return,
            Err(SceneError::StaleBvh(id)) => scene.update(id).unwrap(),
            Err(e) => panic!("update_sections failed: {e}"),
        }
    }
    panic!("update_sections kept failing");
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit_vec(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    a.map(|c| c / n)
}

/// The world plane in mesh `m`'s local frame, as the scene documents it.
fn local_plane(scene: &Scene, m: MeshId, plane: &Plane) -> Plane {
    let r = scene.placement(m).unwrap().rotation;
    let back = |d: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|i| r[0][i] * d[0] + r[1][i] * d[1] + r[2][i] * d[2])
    };
    let origin = scene.world_to_local(m, plane.origin()).unwrap();
    let u = unit_vec(back(plane.u()));
    let v = back(plane.v());
    let d = dot(v, u);
    let v = unit_vec(std::array::from_fn(|k| v[k] - d * u[k]));
    Plane::with_axes(origin, u, v).unwrap()
}

/// Mesh `m`'s section by `plane`, computed from scratch.
fn reference(scene: &Scene, m: MeshId, plane: &Plane, fill: bool) -> SectionResult {
    let mesh = scene.mesh(m).unwrap();
    let bvh = Bvh::build(mesh, &BuildOptions::default()).unwrap();
    let mut cut = Slice::default();
    fast_mesh_query::slice(mesh, &bvh, &local_plane(scene, m, plane), &mut cut).unwrap();
    let mut out = SectionResult::default();
    for c in cut.contours {
        let points: Vec<[f64; 3]> = c
            .points
            .iter()
            .map(|&p| scene.local_to_world(m, p).unwrap())
            .collect();
        let uv: Vec<[f64; 2]> = points.iter().map(|&p| plane.to_uv(p)).collect();
        let length = contour_length(&uv, c.closed);
        let area = if c.closed { signed_area(&uv) } else { 0.0 };
        out.length += length;
        if c.closed && c.oriented {
            out.area += area;
        }
        out.contours.push(SectionContour {
            points,
            uv,
            edges: c.edges,
            triangles: c.triangles,
            closed: c.closed,
            oriented: c.oriented,
            manifold: c.manifold,
            length,
            area,
        });
    }
    if fill {
        let closed: Vec<(&[[f64; 2]], bool)> = out
            .contours
            .iter()
            .filter(|c| c.closed)
            .map(|c| (c.uv.as_slice(), c.oriented))
            .collect();
        let mut f = Fill::default();
        out.fill_ok = fast_mesh_query::fill(&closed, &mut f).unwrap();
        out.fill = Some(f);
    }
    out
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() <= TOL
}

fn near_all<const N: usize>(a: &[[f64; N]], b: &[[f64; N]]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(p, q)| (0..N).all(|k| near(p[k], q[k])))
}

/// `got` equals `want`: topology exactly, coordinates within `TOL`.
fn check_equal(got: &SectionResult, want: &SectionResult, at: &str) -> Result<(), TestCaseError> {
    prop_assert_eq!(got.contours.len(), want.contours.len(), "{}", at);
    for (g, w) in got.contours.iter().zip(&want.contours) {
        prop_assert_eq!(&g.edges, &w.edges, "{}", at);
        prop_assert_eq!(&g.triangles, &w.triangles, "{}", at);
        prop_assert_eq!(
            (g.closed, g.oriented, g.manifold),
            (w.closed, w.oriented, w.manifold),
            "{}",
            at
        );
        prop_assert!(near_all(&g.points, &w.points), "{}: points", at);
        prop_assert!(near_all(&g.uv, &w.uv), "{}: uv", at);
        prop_assert!(near(g.length, w.length) && near(g.area, w.area), "{}", at);
    }
    prop_assert!(near(got.length, want.length), "{}: length", at);
    prop_assert!(near(got.area, want.area), "{}: area", at);
    prop_assert_eq!(got.fill_ok, want.fill_ok, "{}", at);
    match (&got.fill, &want.fill) {
        (None, None) => {}
        (Some(g), Some(w)) => {
            prop_assert_eq!(&g.triangles, &w.triangles, "{}: fill", at);
            prop_assert!(near_all(&g.points, &w.points), "{}: fill points", at);
        }
        _ => prop_assert!(false, "{}: fill presence differs", at),
    }
    Ok(())
}

proptest! {
    #![proptest_config(config(64))]

    /// See the module docs.
    #[test]
    fn sections_match_from_scratch(
        modes in prop::array::uniform2(0u8..3),
        planes in prop::array::uniform2((any::<[i8; 3]>(), any::<[i8; 3]>())),
        fills in any::<[bool; 2]>(),
        batches in prop::collection::vec(prop::collection::vec(op(), 1..6), 1..12),
    ) {
        let mut scene = Scene::new(SceneOptions::default());
        let options = |k: u8| MeshOptions {
            mode: update_mode(k),
            ..MeshOptions::default()
        };
        let s = scene.add_mesh(sphere(), options(modes[0])).unwrap();
        let t = scene.add_mesh(terrain(), options(modes[1])).unwrap();
        let descs = [
            SectionDesc {
                plane: plane_of(planes[0].0, planes[0].1),
                meshes: vec![s, t],
                fill: fills[0],
            },
            SectionDesc {
                plane: plane_of(planes[1].0, planes[1].1),
                meshes: vec![s],
                fill: fills[1],
            },
        ];
        let ids = [
            scene.add_section(descs[0].clone()).unwrap(),
            scene.add_section(descs[1].clone()).unwrap(),
        ];
        let mut w = World { scene, meshes: [s, t], sections: ids, descs };
        update_all(&mut w.scene);
        w.scene.take_changes();

        let results = |w: &World| -> Vec<(usize, MeshId, SectionResult)> {
            (0..2)
                .flat_map(|k| w.descs[k].meshes.iter().map(move |&m| (k, m)))
                .map(|(k, m)| (k, m, w.scene.section(w.sections[k], m).unwrap().clone()))
                .collect()
        };
        let mut before = results(&w);
        for (i, batch) in batches.iter().enumerate() {
            for o in batch {
                apply(&mut w, o);
            }
            update_all(&mut w.scene);
            let changes = w.scene.take_changes();
            for k in 0..2 {
                let d = &w.descs[k];
                for &m in &d.meshes {
                    let at = format!("batch {i}, section {k}, mesh {m:?}");
                    prop_assert_eq!(
                        w.scene.section_status(w.sections[k], m),
                        Ok(SectionStatus::Current),
                        "{}",
                        at
                    );
                    let got = w.scene.section(w.sections[k], m).unwrap();
                    let want = reference(&w.scene, m, &d.plane, d.fill);
                    check_equal(got, &want, &at)?;
                }
            }
            // A result that changed is reported.
            let after = results(&w);
            for (k, m, r) in &after {
                let old = before.iter().find(|(k2, m2, _)| k2 == k && m2 == m);
                if let Some((_, _, old)) = old
                    && old != r
                {
                    let reported = changes
                        .iter()
                        .find(|c| c.id == *m)
                        .is_some_and(|c| c.sections_changed.contains(&w.sections[*k]));
                    prop_assert!(reported, "batch {}: section {} on {:?} not reported", i, k, m);
                }
            }
            before = after;
        }
    }
}
