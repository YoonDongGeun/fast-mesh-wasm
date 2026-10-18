//! Repair operations run through `Scene::edit`: the edit is one undoable
//! step, Eager BVHs, fields and sections follow it like any other edit, and
//! a repair error is a `SceneError::Ops` that leaves the scene unchanged.

use fast_mesh_scene::ops::{self, FillOptions, OpsError};
use fast_mesh_scene::{
    DistanceField, MeshId, MeshOptions, Plane, Scene, SceneError, SceneOptions, SectionDesc,
    SignMode, UpdateMode,
};
use fast_mesh_store::{Mesh, TriangleId, VertexId};

const STACKS: u32 = 6;
const SLICES: u32 = 16;

/// The upper unit hemisphere, outward wound, open at the rim (16 edges).
/// `holes` are left out (by index into the full list) and `flips` are wound
/// the other way.
fn hemisphere(holes: &[usize], flips: &[usize]) -> Mesh {
    let mut pos = vec![0.0f32, 0.0, 1.0];
    for i in 1..=STACKS {
        let theta = std::f64::consts::FRAC_PI_2 * f64::from(i) / f64::from(STACKS);
        for j in 0..SLICES {
            let phi = std::f64::consts::TAU * f64::from(j) / f64::from(SLICES);
            let (s, c) = theta.sin_cos();
            pos.extend([(s * phi.cos()) as f32, (s * phi.sin()) as f32, c as f32]);
        }
    }
    let ring = |i: u32, j: u32| 1 + (i - 1) * SLICES + j % SLICES;
    let mut tris = Vec::new();
    for j in 0..SLICES {
        tris.push([0, ring(1, j), ring(1, j + 1)]);
    }
    for i in 1..STACKS {
        for j in 0..SLICES {
            let (a, b, c, d) = (
                ring(i, j),
                ring(i + 1, j),
                ring(i + 1, j + 1),
                ring(i, j + 1),
            );
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    for &f in flips {
        tris[f].swap(1, 2);
    }
    let idx: Vec<u32> = tris
        .iter()
        .enumerate()
        .filter(|(k, _)| !holes.contains(k))
        .flat_map(|(_, t)| *t)
        .collect();
    Mesh::from_buffers(&pos, &idx).unwrap()
}

/// The index of quad `(i, j)`'s first triangle (rings `i` and `i + 1`).
fn quad(i: u32, j: u32) -> usize {
    (SLICES + 2 * ((i - 1) * SLICES + j)) as usize
}

/// A pole-fan hole (3 edges) and a hole of 3 x 3 quads between rings 2 and
/// 5 (12 edges, four vertices left unused), with a few flipped triangles
/// away from them.
fn damaged() -> Mesh {
    let mut holes = vec![10];
    for i in 2..5 {
        for j in 3..6 {
            holes.extend([quad(i, j), quad(i, j) + 1]);
        }
    }
    hemisphere(&holes, &[2, quad(1, 7), quad(3, 12) + 1, quad(4, 0)])
}

/// A small closed octahedron inside the dome.
fn probe() -> Mesh {
    let (c, r) = ([0.1f32, -0.05, 0.55], 0.2f32);
    let pos = [
        c[0] + r,
        c[1],
        c[2], //
        c[0] - r,
        c[1],
        c[2], //
        c[0],
        c[1] + r,
        c[2], //
        c[0],
        c[1] - r,
        c[2], //
        c[0],
        c[1],
        c[2] + r, //
        c[0],
        c[1],
        c[2] - r,
    ];
    let idx = [
        0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5,
    ];
    Mesh::from_buffers(&pos, &idx).unwrap()
}

fn eager() -> MeshOptions {
    MeshOptions {
        mode: UpdateMode::Eager,
        ..MeshOptions::default()
    }
}

/// Bounds that fill both holes but keep the rim.
fn fill_options() -> FillOptions {
    FillOptions {
        max_edges: Some(12),
        ..FillOptions::default()
    }
}

/// Positions and triangles of every slot.
fn snapshot(scene: &Scene, id: MeshId) -> (Vec<[f32; 3]>, Vec<[VertexId; 3]>) {
    let m = scene.mesh(id).unwrap();
    (m.positions().to_vec(), m.triangles().to_vec())
}

/// Alive vertices with their positions and alive triangles, by slot.
type Alive = (Vec<(VertexId, [f32; 3])>, Vec<(TriangleId, [VertexId; 3])>);

/// The alive part of a snapshot.
fn alive(scene: &Scene, id: MeshId) -> Alive {
    let m = scene.mesh(id).unwrap();
    let vs = (0..m.vertex_slots() as VertexId)
        .filter(|&v| m.is_vertex_alive(v))
        .map(|v| (v, m.position(v).unwrap()))
        .collect();
    let ts = (0..m.triangle_slots() as TriangleId)
        .filter(|&t| m.is_triangle_alive(t))
        .map(|t| (t, m.triangle(t).unwrap()))
        .collect();
    (vs, ts)
}

#[test]
fn repair_via_edit_is_undoable() {
    let mut scene = Scene::new(SceneOptions::default());
    let id = scene.add_mesh(damaged(), eager()).unwrap();
    let before = alive(&scene, id);
    let slots = {
        let m = scene.mesh(id).unwrap();
        (m.vertex_slots(), m.triangle_slots())
    };
    let opts = fill_options();

    let report = scene
        .edit(id, |tx| {
            let oriented = ops::orient(tx)?;
            assert!(oriented.flipped_triangles > 0);
            ops::fill_holes(tx, &opts)
        })
        .unwrap();
    assert_eq!(report.filled.len(), 2, "{report:?}");
    assert!(
        report.filled.iter().any(|f| f.vertices_added > 0),
        "{report:?}"
    );
    assert_eq!(scene.is_current(id), Ok(true));
    let repaired = snapshot(&scene, id);
    assert_ne!(alive(&scene, id), before);
    let analysis = ops::analyze(scene.mesh(id).unwrap(), &Default::default()).unwrap();
    assert_eq!(analysis.loops.len(), 1, "only the rim stays open");
    assert_eq!(analysis.orientation_conflicts, 0);

    // Undo leaves the slots the fill added dead; the alive content is exact.
    assert_eq!(scene.undo(), Ok(true));
    assert_eq!(alive(&scene, id), before);
    let m = scene.mesh(id).unwrap();
    assert!(m.vertex_slots() > slots.0 && m.triangle_slots() > slots.1);
    assert!((slots.0..m.vertex_slots()).all(|v| !m.is_vertex_alive(v as VertexId)));
    assert!((slots.1..m.triangle_slots()).all(|t| !m.is_triangle_alive(t as TriangleId)));
    assert_eq!(scene.is_current(id), Ok(true));
    assert_eq!(scene.redo(), Ok(true));
    assert_eq!(snapshot(&scene, id), repaired);
    assert_eq!(scene.is_current(id), Ok(true));
}

/// One field each way between the hemisphere and the probe, and a section
/// through the quad hole.
fn decorate(
    scene: &mut Scene,
    hemi: MeshId,
    probe: MeshId,
) -> (Vec<fast_mesh_scene::FieldId>, fast_mesh_scene::SectionId) {
    let on_hemi = scene
        .add_distance_field(DistanceField {
            name: "to_probe".into(),
            on: hemi,
            to: probe,
            max_distance: f64::INFINITY,
            sign: SignMode::Pseudonormal,
        })
        .unwrap();
    let on_probe = scene
        .add_distance_field(DistanceField {
            name: "to_hemi".into(),
            on: probe,
            to: hemi,
            max_distance: f64::INFINITY,
            sign: SignMode::Pseudonormal,
        })
        .unwrap();
    // Through the big hole, between rings 2 and 3.
    let z = (2.5 * std::f64::consts::FRAC_PI_2 / f64::from(STACKS)).cos();
    let section = scene
        .add_section(SectionDesc {
            plane: Plane::new([0.0, 0.0, z], [0.05, 0.02, 1.0]).unwrap(),
            meshes: vec![hemi],
            fill: true,
        })
        .unwrap();
    (vec![on_hemi, on_probe], section)
}

#[test]
fn repair_updates_fields_and_sections() {
    let mut scene = Scene::new(SceneOptions::default());
    let hemi = scene.add_mesh(damaged(), eager()).unwrap();
    let pr = scene.add_mesh(probe(), eager()).unwrap();
    let (fields, section) = decorate(&mut scene, hemi, pr);
    scene.update_fields().unwrap();
    scene.update_sections().unwrap();
    let section_before = scene.section(section, hemi).unwrap().clone();

    let opts = fill_options();
    let report = scene
        .edit(hemi, |tx| {
            ops::orient(tx)?;
            ops::fill_holes(tx, &opts)
        })
        .unwrap();
    assert_eq!(report.filled.len(), 2, "{report:?}");
    assert!(
        report
            .filled
            .iter()
            .all(|f| f.vertices_added > 0 || f.triangles_added > 0)
    );

    // Eager: everything follows the edit without being asked.
    for &f in &fields {
        assert!(scene.field_status(f).unwrap().current);
    }
    assert_eq!(
        scene.section_status(section, hemi),
        Ok(fast_mesh_scene::SectionStatus::Current)
    );
    assert_ne!(
        scene.section(section, hemi).unwrap(),
        &section_before,
        "the section crosses a filled hole"
    );

    // From scratch: a new scene on a copy of the repaired mesh, slot for slot.
    let (positions, triangles) = snapshot(&scene, hemi);
    let m = scene.mesh(hemi).unwrap();
    assert_eq!(m.vertex_count(), m.vertex_slots());
    assert_eq!(m.triangle_count(), m.triangle_slots());
    let flat: Vec<f32> = positions.iter().flatten().copied().collect();
    let idx: Vec<u32> = triangles.iter().flatten().copied().collect();
    let mut fresh = Scene::new(SceneOptions::default());
    let fresh_hemi = fresh
        .add_mesh(Mesh::from_buffers(&flat, &idx).unwrap(), eager())
        .unwrap();
    let fresh_probe = fresh.add_mesh(probe(), eager()).unwrap();
    let (fresh_fields, fresh_section) = decorate(&mut fresh, fresh_hemi, fresh_probe);
    fresh.update_fields().unwrap();
    fresh.update_sections().unwrap();

    for (&f, &g) in fields.iter().zip(&fresh_fields) {
        let (got, want) = (
            scene.field_values(f).unwrap(),
            fresh.field_values(g).unwrap(),
        );
        assert_eq!(got.len(), want.len());
        for (v, (a, b)) in got.iter().zip(want).enumerate() {
            // Bit-exact: the same positions and meshes give the same values
            // (unused vertices hold NaN in both).
            let same = a.to_bits() == b.to_bits();
            assert!(same, "vertex {v}: {a} vs {b}");
        }
    }
    assert_eq!(
        scene.section(section, hemi).unwrap(),
        fresh.section(fresh_section, fresh_hemi).unwrap()
    );
}

#[test]
fn ops_error_maps_to_scene_error() {
    let mut scene = Scene::new(SceneOptions::default());
    let id = scene.add_mesh(damaged(), eager()).unwrap();
    scene.take_changes();
    let revision = scene.mesh(id).unwrap().revision();
    let before = snapshot(&scene, id);

    scene.begin_step();
    // `orient` writes before `fill_holes` fails: the whole edit rolls back.
    let got = scene.edit(id, |tx| {
        ops::orient(tx)?;
        ops::fill_holes(tx, &FillOptions::default())
    });
    assert!(
        matches!(got, Err(SceneError::Ops(OpsError::InvalidInput(_)))),
        "{got:?}"
    );
    assert_eq!(scene.mesh(id).unwrap().revision(), revision);
    assert_eq!(snapshot(&scene, id), before);
    assert!(scene.take_changes().is_empty());
    assert_eq!(scene.is_current(id), Ok(true));
    scene.end_step().unwrap();
    assert_eq!(scene.undo(), Ok(false), "nothing was recorded");

    let e = SceneError::Ops(OpsError::StaleLoops);
    assert_eq!(
        e.to_string(),
        "repair operation failed: a requested boundary loop no longer matches the mesh"
    );
    let source = std::error::Error::source(&e).unwrap();
    assert_eq!(
        source.downcast_ref::<OpsError>(),
        Some(&OpsError::StaleLoops)
    );
    assert_eq!(SceneError::from(OpsError::StaleLoops), e);
}
