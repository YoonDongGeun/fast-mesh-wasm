//! Property test for distance fields and colours: after any sequence of
//! placements, drags, edits, undo, redo, compactions and mode changes, every
//! field value must equal a from-scratch `signed_distances`, every colour
//! must equal `color_of` of its value, and `colors_dirty` must cover every
//! colour that changed.
//!
//! The scene holds two small meshes, each closed or open, built from `f64`
//! coordinates around 1000 so that both have a non-zero origin and the
//! field path's origin handling is exercised. A field runs from each mesh to
//! the other, in a random sign mode and with a random `max_distance`; either
//! only mesh A is coloured (one-sided) or both are.
//!
//! The reference uses public API only. For the mesh the distances are
//! measured to, it builds a shadow `Mesh` from the scene mesh's alive
//! triangles, a fresh BVH and fresh `SignData`, and maps the vertices of the
//! other mesh through `Scene::local_to_world` and `Scene::world_to_local`.
//!
//! One difference from the unit test `field_matches_from_scratch` in
//! `fast-mesh-scene`: that test reuses the production BVH for winding signs,
//! because the fast winding number depends on the shape of the BVH tree. An
//! integration test cannot reach the scene's `&mut Mesh` that
//! `SignData::build` needs for that BVH, so when the shadow's sign data
//! resolves to winding, a sign that differs between the two trees is accepted
//! at points where the shadow's winding number is within `WINDING_MARGIN` of
//! the 0.5 threshold, which the approximation's error (about 1e-3) can flip.
//! Near an open mesh's hole that holds at any distance from the surface.
//! Everything else is compared strictly.

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_fields::{SignData, signed_distances};
use fast_mesh_scene::{
    ColorRule, Coloring, Colormap, DistanceField, MeshId, MeshOptions, OutOfRange, Placement,
    Scene, SceneError, SceneOptions, SignMode, UpdateMode, color_of,
};
use fast_mesh_store::{Mesh, VertexId};
use proptest::prelude::*;
use proptest::test_runner::RngSeed;

/// The case count: `PROPTEST_CASES` when set, else `default`. `with_cases`
/// alone would ignore the environment variable.
fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// The proptest config: a fixed seed, so CI is reproducible (a rare sign
/// flip near the winding tolerance band must not fail CI with no code
/// change). `PROPTEST_RNG_SEED=random` (the nightly job) or a number
/// overrides it.
fn config() -> ProptestConfig {
    let mut c = ProptestConfig::with_cases(cases(64));
    c.rng_seed = match std::env::var("PROPTEST_RNG_SEED").as_deref() {
        Ok("random") => RngSeed::Random,
        Ok(s) => s
            .parse()
            .map(RngSeed::Fixed)
            .unwrap_or(RngSeed::Fixed(0x5EED_F33D)),
        Err(_) => RngSeed::Fixed(0x5EED_F33D),
    };
    c
}

/// Surface tolerance for the reference, whose points take another rounding
/// path (relative to the meshes' unit size).
const TOL: f32 = 1e-5;

/// How close to 0.5 a winding number must be for its sign to differ between
/// two BVH trees.
const WINDING_MARGIN: f64 = 0.02;

/// World positions of the two meshes' corners, around 1000.
const ORIGIN_A: [f64; 3] = [1000.0, -1000.5, 999.7];
const ORIGIN_B: [f64; 3] = [1000.3, -1000.2, 999.6];

/// A grid sheet, 8 x 8 squares of 0.1, in the plane `z = 0` of `origin`.
fn sheet(origin: [f64; 3]) -> Mesh {
    let n = 8u32;
    let mut pos = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            pos.extend_from_slice(&[
                origin[0] + f64::from(x) * 0.1,
                origin[1] + f64::from(y) * 0.1,
                origin[2],
            ]);
        }
    }
    let mut idx = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            let c = a + n + 1;
            idx.extend_from_slice(&[a, a + 1, c, a + 1, c + 1, c]);
        }
    }
    Mesh::from_buffers_f64(&pos, &idx).unwrap()
}

/// A cube of side `size` with its minimum corner at `min`, outward oriented;
/// `open` leaves out the two triangles of one face.
fn cube(min: [f64; 3], size: f64, open: bool) -> Mesh {
    let mut pos = Vec::new();
    for i in 0..8u32 {
        let b = |bit: u32| if i & bit != 0 { size } else { 0.0 };
        pos.extend_from_slice(&[min[0] + b(1), min[1] + b(2), min[2] + b(4)]);
    }
    let mut idx = vec![
        0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6,
        1, 3, 5, 3, 7, 5,
    ];
    if open {
        idx.truncate(idx.len() - 6);
    }
    Mesh::from_buffers_f64(&pos, &idx).unwrap()
}

/// Mesh A: an open sheet, or a closed cube through the sheet's place.
fn mesh_a(closed: bool) -> Mesh {
    if closed {
        cube([ORIGIN_A[0], ORIGIN_A[1], ORIGIN_A[2] - 0.4], 0.8, false)
    } else {
        sheet(ORIGIN_A)
    }
}

/// Mesh B: a cube of side 0.4, closed or with a face missing, cutting A.
fn mesh_b(closed: bool) -> Mesh {
    cube(
        [ORIGIN_B[0] + 0.1, ORIGIN_B[1] + 0.1, ORIGIN_B[2] - 0.2],
        0.4,
        !closed,
    )
}

#[derive(Debug, Clone)]
enum Op {
    /// A placement with a small turn about z and a small translation.
    Place(bool, i8, [i8; 3]),
    /// Moves a vertex by a small amount.
    Move(bool, u32, [i8; 3]),
    /// Scales a vertex of mesh B about the origin of its local space.
    Scale(u32, i8),
    RemoveTriangle(bool, u32),
    /// Splits a triangle at its centroid.
    Split(bool, u32),
    /// Adds a vertex and a triangle on two alive vertices of mesh A.
    AddTriangle(u32, u32, [i8; 2]),
    RemoveUnusedVertex(bool),
    /// Three placements in one step.
    Drag(bool, i8),
    /// A drag with a vertex edit, cancelled.
    CancelledDrag(bool, u32),
    Undo,
    Redo,
    Compact(bool),
    SetMode(bool, u8),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        3 => (any::<bool>(), any::<i8>(), any::<[i8; 3]>())
            .prop_map(|(m, d, t)| Op::Place(m, d, t)),
        3 => (any::<bool>(), any::<u32>(), any::<[i8; 3]>())
            .prop_map(|(m, v, d)| Op::Move(m, v, d)),
        1 => (any::<u32>(), any::<i8>()).prop_map(|(v, s)| Op::Scale(v, s)),
        1 => (any::<bool>(), any::<u32>()).prop_map(|(m, t)| Op::RemoveTriangle(m, t)),
        1 => (any::<bool>(), any::<u32>()).prop_map(|(m, t)| Op::Split(m, t)),
        1 => (any::<u32>(), any::<u32>(), any::<[i8; 2]>())
            .prop_map(|(p, q, x)| Op::AddTriangle(p, q, x)),
        1 => any::<bool>().prop_map(Op::RemoveUnusedVertex),
        1 => (any::<bool>(), any::<i8>()).prop_map(|(m, d)| Op::Drag(m, d)),
        1 => (any::<bool>(), any::<u32>()).prop_map(|(m, v)| Op::CancelledDrag(m, v)),
        2 => Just(Op::Undo),
        1 => Just(Op::Redo),
        1 => any::<bool>().prop_map(Op::Compact),
        1 => (any::<bool>(), 0u8..3).prop_map(|(m, k)| Op::SetMode(m, k)),
    ]
}

fn update_mode(k: u8) -> UpdateMode {
    [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual][usize::from(k) % 3]
}

fn sign_mode(k: u8) -> SignMode {
    [SignMode::Pseudonormal, SignMode::Winding, SignMode::Auto][usize::from(k) % 3]
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

fn alive_vertices(scene: &Scene, m: MeshId) -> Vec<VertexId> {
    let mesh = scene.mesh(m).unwrap();
    (0..mesh.vertex_slots() as VertexId)
        .filter(|&v| mesh.is_vertex_alive(v))
        .collect()
}

fn alive_triangles(scene: &Scene, m: MeshId) -> Vec<u32> {
    let mesh = scene.mesh(m).unwrap();
    (0..mesh.triangle_slots() as u32)
        .filter(|&t| mesh.is_triangle_alive(t))
        .collect()
}

/// Picks one of `items` by `k`, if there are any.
fn pick<T: Copy>(items: &[T], k: u32) -> Option<T> {
    (!items.is_empty()).then(|| items[k as usize % items.len()])
}

/// Applies one operation to the scene. Operations that do not apply (nothing
/// to undo, no such vertex) are skipped; a mesh is never left with fewer
/// than five triangles by a removal, so that no mesh becomes empty.
fn apply(scene: &mut Scene, a: MeshId, b: MeshId, op: &Op) {
    let which = |m: &bool| if *m { a } else { b };
    match op {
        Op::Place(m, deg, t) => {
            let t = [unit(t[0]) * 0.1, unit(t[1]) * 0.1, unit(t[2]) * 0.1];
            scene
                .set_placement(which(m), turned(unit(*deg) * 25.0, t))
                .unwrap();
        }
        Op::Move(m, v, d) => {
            let m = which(m);
            if let Some(v) = pick(&alive_vertices(scene, m), *v) {
                let d = d.map(|x| (unit(x) * 0.1) as f32);
                scene
                    .edit(m, |tx| {
                        let p = tx.mesh().position(v).unwrap();
                        tx.set_positions(&[v], &[[p[0] + d[0], p[1] + d[1], p[2] + d[2]]])
                    })
                    .unwrap();
            }
        }
        Op::Scale(v, s) => {
            if let Some(v) = pick(&alive_vertices(scene, b), *v) {
                let s = 1.0 + (unit(*s) * 0.05) as f32;
                scene
                    .edit(b, |tx| {
                        let p = tx.mesh().position(v).unwrap();
                        tx.set_positions(&[v], &[p.map(|c| c * s)])
                    })
                    .unwrap();
            }
        }
        Op::RemoveTriangle(m, t) => {
            let m = which(m);
            let tris = alive_triangles(scene, m);
            if tris.len() > 4
                && let Some(t) = pick(&tris, *t)
            {
                scene.edit(m, |tx| tx.remove_triangles(&[t])).unwrap();
            }
        }
        Op::Split(m, t) => {
            let m = which(m);
            if let Some(t) = pick(&alive_triangles(scene, m), *t) {
                scene
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
        Op::AddTriangle(p, q, x) => {
            let alive = alive_vertices(scene, a);
            let (Some(p), Some(q)) = (pick(&alive, *p), pick(&alive, *q)) else {
                return;
            };
            if p == q {
                return;
            }
            scene
                .edit(a, |tx| {
                    let (pp, pq) = (
                        tx.mesh().position(p).unwrap(),
                        tx.mesh().position(q).unwrap(),
                    );
                    let x = [
                        (pp[0] + pq[0]) * 0.5 + (unit(x[0]) * 0.1) as f32,
                        (pp[1] + pq[1]) * 0.5 + (unit(x[1]) * 0.1) as f32,
                        (pp[2] + pq[2]) * 0.5 + 0.02,
                    ];
                    let v = tx.add_vertices(&[x])?;
                    tx.add_triangles(&[[v[0], p, q]])
                })
                .unwrap();
        }
        Op::RemoveUnusedVertex(m) => {
            let m = which(m);
            let mesh = scene.mesh(m).unwrap();
            let unused = (0..mesh.vertex_slots() as VertexId)
                .find(|&v| mesh.is_vertex_alive(v) && mesh.corner_count(v) == 0);
            if let Some(v) = unused {
                scene.edit(m, |tx| tx.remove_vertices(&[v])).unwrap();
            }
        }
        Op::Drag(m, d) => {
            let m = which(m);
            scene.begin_step();
            for k in 1..=3 {
                let d = 0.01 * f64::from(k) * (1.0 + unit(*d));
                scene
                    .set_placement(m, turned(d * 50.0, [d, -d, d]))
                    .unwrap();
            }
            scene.end_step().unwrap();
        }
        Op::CancelledDrag(m, v) => {
            let m = which(m);
            scene.begin_step();
            scene
                .set_placement(m, turned(0.0, [0.03, 0.02, -0.04]))
                .unwrap();
            if let Some(v) = pick(&alive_vertices(scene, a), *v) {
                scene
                    .edit(a, |tx| {
                        let p = tx.mesh().position(v).unwrap();
                        tx.set_positions(&[v], &[[p[0], p[1], p[2] + 0.1]])
                    })
                    .unwrap();
            }
            scene.cancel_step().unwrap();
        }
        Op::Undo => {
            scene.undo().unwrap();
        }
        Op::Redo => {
            scene.redo().unwrap();
        }
        Op::Compact(m) => {
            scene.compact(which(m)).unwrap();
        }
        Op::SetMode(m, k) => {
            scene.set_mode(which(m), update_mode(*k)).unwrap();
        }
    }
}

/// `update_fields`, bringing stale Manual BVHs up to date when it asks.
fn update_all(scene: &mut Scene) {
    for _ in 0..8 {
        match scene.update_fields() {
            Ok(()) => return,
            Err(SceneError::StaleBvh(id)) => scene.update(id).unwrap(),
            Err(e) => panic!("update_fields failed: {e}"),
        }
    }
    panic!("update_fields kept failing");
}

/// The `f32` bound for `signed_distances` that keeps every distance up to
/// `md` (rounded up; infinite stays infinite).
fn bound_of(md: f64) -> f32 {
    let x = md as f32;
    if f64::from(x) < md {
        f32::from_bits(x.to_bits() + 1)
    } else {
        x
    }
}

/// Whether `got` (the scene's value) and `want` (the reference's) agree.
///
/// - Equal bits, or both NaN.
/// - Both finite, within `TOL` (or, with `loose_sign`, equal in magnitude
///   within `TOL`: a flipped winding sign).
/// - One finite and one infinite: the finite one sits within `TOL` of
///   `max_distance`. Both infinite: equal, so that `+∞` never matches `-∞`.
fn agree(got: f32, want: f32, md: f64, loose_sign: bool) -> bool {
    if got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()) {
        return true;
    }
    if got.is_nan() || want.is_nan() {
        return false;
    }
    match (got.is_finite(), want.is_finite()) {
        (true, true) => {
            (got - want).abs() <= TOL || (loose_sign && (got.abs() - want.abs()).abs() <= TOL)
        }
        (false, false) => got == want,
        _ => {
            let finite = if got.is_finite() { got } else { want };
            let infinite = if got.is_finite() { want } else { got };
            infinite == f32::INFINITY && f64::from(finite.abs()) >= md - f64::from(TOL)
        }
    }
}

/// The field on `on`, to `to`, computed from scratch (see the module docs).
fn check_values(
    scene: &Scene,
    field: fast_mesh_scene::FieldId,
    desc: &DistanceField,
    at: &str,
) -> Result<(), TestCaseError> {
    let target = scene.mesh(desc.to[0]).unwrap();
    let positions: Vec<f32> = target.positions().iter().flatten().copied().collect();
    let indices: Vec<u32> = (0..target.triangle_slots() as u32)
        .filter(|&t| target.is_triangle_alive(t))
        .flat_map(|t| target.triangle(t).unwrap())
        .collect();
    let mut shadow = Mesh::from_buffers(&positions, &indices).unwrap();
    let bvh = Bvh::build(&shadow, &BuildOptions::default()).unwrap();
    let sign = SignData::build(&mut shadow, &bvh, desc.sign).unwrap();
    let winding = sign.mode() == SignMode::Winding;

    let source = scene.mesh(desc.on).unwrap();
    let slots = source.vertex_slots();
    let defined: Vec<bool> = (0..slots as VertexId)
        .map(|v| source.is_vertex_alive(v) && source.corner_count(v) > 0)
        .collect();
    let points: Vec<[f32; 3]> = (0..slots as VertexId)
        .filter(|&v| defined[v as usize])
        .map(|v| {
            let p = source.position(v).unwrap().map(f64::from);
            let w = scene.local_to_world(desc.on, p).unwrap();
            scene
                .world_to_local(desc.to[0], w)
                .unwrap()
                .map(|c| c as f32)
        })
        .collect();
    let mut out = vec![0.0; points.len()];
    signed_distances(
        &shadow,
        &bvh,
        &sign,
        &points,
        bound_of(desc.max_distance),
        &mut out,
    )
    .unwrap();
    let mut wants = out.into_iter().map(|v| {
        if f64::from(v.abs()) > desc.max_distance {
            f32::INFINITY
        } else {
            v
        }
    });
    let want: Vec<f32> = defined
        .iter()
        .map(|&d| if d { wants.next().unwrap() } else { f32::NAN })
        .collect();

    // Where the reference's winding number is near the threshold.
    let mut near_threshold = defined.iter().scan(points.iter(), |ps, &d| {
        Some(
            d && {
                let w = sign.winding_number(&shadow, &bvh, *ps.next().unwrap());
                winding && (w - 0.5).abs() < WINDING_MARGIN
            },
        )
    });
    let loose: Vec<bool> = near_threshold.by_ref().collect();

    let got = scene.field_values(field).unwrap();
    prop_assert_eq!(got.len(), want.len(), "{}", at);
    for (v, (g, w)) in got.iter().zip(&want).enumerate() {
        prop_assert!(
            agree(*g, *w, desc.max_distance, loose[v]),
            "{}: vertex {}: got {}, want {}",
            at,
            v,
            g,
            w
        );
    }
    prop_assert!(scene.field_status(field).unwrap().current, "{}", at);
    Ok(())
}

/// The colours of `mesh` are `color_of` of its field's values, one per
/// vertex slot, and every colour that differs from `before` is reported.
fn check_colors(
    scene: &Scene,
    mesh: MeshId,
    field: fast_mesh_scene::FieldId,
    rule: &ColorRule,
    before: &[[u8; 4]],
    dirty: Option<&Vec<VertexId>>,
    at: &str,
) -> Result<(), TestCaseError> {
    let values = scene.field_values(field).unwrap();
    let colors = scene.colors(mesh).unwrap().unwrap();
    prop_assert_eq!(colors.len(), scene.mesh(mesh).unwrap().vertex_slots());
    prop_assert_eq!(colors.len(), values.len());
    for (v, (c, &x)) in colors.iter().zip(values).enumerate() {
        prop_assert_eq!(*c, color_of(rule, x), "{}: vertex {}, value {}", at, v, x);
        if let Some(dirty) = dirty
            && before.get(v) != Some(c)
        {
            prop_assert!(
                dirty.binary_search(&(v as VertexId)).is_ok(),
                "{}: vertex {} changed colour but is not in colors_dirty",
                at,
                v
            );
        }
    }
    Ok(())
}

fn contact_rule(d: f32) -> ColorRule {
    ColorRule {
        map: Colormap::Turbo,
        range: (-d, 0.0),
        below: OutOfRange::Clamp,
        above: OutOfRange::Base,
        center: None,
        base: [200, 200, 200, 255],
        nan: [255, 0, 255, 255],
    }
}

fn distance_rule(d: f32, g: f32) -> ColorRule {
    ColorRule {
        map: Colormap::Coolwarm,
        range: (-d, g),
        below: OutOfRange::Clamp,
        above: OutOfRange::Base,
        center: Some(0.0),
        base: [200, 200, 200, 255],
        nan: [255, 0, 255, 255],
    }
}

/// A `max_distance` from a seed: infinite for 0, else 0.03 to 0.25.
fn max_distance(k: u8) -> f64 {
    if k == 0 {
        f64::INFINITY
    } else {
        0.03 + f64::from(k - 1) / 254.0 * 0.22
    }
}

proptest! {
    #![proptest_config(config())]

    /// See the module docs.
    #[test]
    fn fields_match_from_scratch(
        closed in any::<[bool; 2]>(),
        modes in prop::array::uniform2(0u8..3),
        signs in prop::array::uniform2(0u8..3),
        mds in prop::array::uniform2(any::<u8>()),
        both_sides in any::<bool>(),
        batches in prop::collection::vec(prop::collection::vec(op(), 1..5), 1..14),
    ) {
        let mut scene = Scene::new(SceneOptions::default());
        let options = |k: u8| MeshOptions {
            mode: update_mode(k),
            ..MeshOptions::default()
        };
        let a = scene.add_mesh(mesh_a(closed[0]), options(modes[0])).unwrap();
        let b = scene.add_mesh(mesh_b(closed[1]), options(modes[1])).unwrap();
        // Field B-to-A is computed against A's BVH, which a Manual mesh
        // may not have current yet.
        scene.update(a).unwrap();
        scene.update(b).unwrap();
        let descs = [
            DistanceField {
                name: "d".to_owned(),
                on: a,
                to: vec![b],
                max_distance: max_distance(mds[0]),
                sign: sign_mode(signs[0]),
            },
            DistanceField {
                name: "d".to_owned(),
                on: b,
                to: vec![a],
                max_distance: max_distance(mds[1]),
                sign: sign_mode(signs[1]),
            },
        ];
        let fa = scene.add_distance_field(descs[0].clone()).unwrap();
        let fb = scene.add_distance_field(descs[1].clone()).unwrap();
        let fields = [fa, fb];
        let rules = [distance_rule(0.2, 0.1), contact_rule(0.1)];
        let meshes = [a, b];
        let coloured = if both_sides { 2 } else { 1 };
        for k in 0..coloured {
            scene
                .set_coloring(
                    meshes[k],
                    Some(Coloring { field: fields[k], rule: rules[k].clone() }),
                )
                .unwrap();
        }
        update_all(&mut scene);
        scene.take_changes();
        if !both_sides {
            prop_assert!(scene.colors(b).unwrap().is_none());
        }
        let mut mode_a = modes[0];
        for (i, batch) in batches.iter().enumerate() {
            let at = format!("batch {i}");
            let before: Vec<Vec<[u8; 4]>> = (0..coloured)
                .map(|k| scene.colors(meshes[k]).unwrap().unwrap().to_vec())
                .collect();
            // An Eager mesh sees one operation at a time, keyed on its
            // current mode (a batch may change it).
            let ops = if mode_a == 0 { &batch[..1] } else { &batch[..] };
            for o in ops {
                apply(&mut scene, a, b, o);
                if let Op::SetMode(true, k) = o {
                    mode_a = k % 3;
                }
                for &m in &meshes[..coloured] {
                    let n = scene.mesh(m).unwrap().vertex_slots();
                    prop_assert_eq!(scene.colors(m).unwrap().unwrap().len(), n);
                }
            }
            update_all(&mut scene);
            let changes = scene.take_changes();
            for k in 0..2 {
                check_values(&scene, fields[k], &descs[k], &at)?;
            }
            for k in 0..coloured {
                let dirty = changes
                    .iter()
                    .find(|c| c.id == meshes[k])
                    .map_or(Some(Vec::new()), |c| c.colors_dirty.clone());
                check_colors(
                    &scene, meshes[k], fields[k], &rules[k], &before[k], dirty.as_ref(), &at,
                )?;
            }
            if !both_sides {
                prop_assert!(scene.colors(b).unwrap().is_none());
            }
        }
    }
}
