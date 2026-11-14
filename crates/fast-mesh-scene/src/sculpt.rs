//! Brush dabs whose cost follows the brush region, not the mesh.

use fast_mesh_ops::deform::{
    Brush, GrabOptions, InflateOptions, SmoothOptions, grab, grab_region, inflate, inflate_region,
    smooth, smooth_region,
};
use fast_mesh_ops::{check_protection, with_protection};
use fast_mesh_query::{Region, vertices_in};
use fast_mesh_store::VertexId;

use crate::scene::Entry;
use crate::{MeshId, Result, Scene, SceneError, UpdateMode};

/// One brush dab for [`Scene::dab`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dab {
    /// [`inflate`](fast_mesh_ops::deform::inflate): along the smooth normals.
    Inflate(InflateOptions),
    /// [`smooth`](fast_mesh_ops::deform::smooth): toward the neighbors' centroid.
    Smooth(SmoothOptions),
    /// [`grab`](fast_mesh_ops::deform::grab): along one vector.
    Grab(GrabOptions),
}

impl Dab {
    fn brush(&self) -> &Brush {
        match self {
            Dab::Inflate(o) => &o.brush,
            Dab::Smooth(o) => &o.brush,
            Dab::Grab(o) => &o.brush,
        }
    }
}

/// What [`Scene::dab`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DabReport {
    /// How many vertices moved (displacement above 0).
    pub moved: usize,
    /// The largest displacement applied, in local units; 0 when nothing moved.
    pub max_displacement: f32,
}

/// The ascending vertices under `brush`, through the mesh's BVH, with the
/// incidence index the region deforms read. `None` when the brush cannot be
/// searched for (an invalid one, which the whole-mesh deform then reports) or
/// the BVH cannot serve: stale under Manual, or a failed update.
fn region_of(entry: &mut Entry, brush: &Brush) -> Option<Vec<VertexId>> {
    let valid = brush.radius.is_finite()
        && brush.radius > 0.0
        && brush.center.iter().all(|c| c.is_finite())
        && brush.strength.is_finite();
    if !valid || (entry.mode == UpdateMode::Manual && !entry.is_current()) {
        return None;
    }
    entry.update_bvh().ok()?;
    entry.mesh.ensure_incidence().ok()?;
    let mut region = Vec::new();
    let sphere = Region::Sphere {
        center: brush.center,
        radius: brush.radius,
    };
    vertices_in(&entry.mesh, &entry.bvh, &sphere, &mut region).ok()?;
    Some(region)
}

/// Whether every vertex of the ascending `region` is listed at weight 0.
fn fully_held(region: &[VertexId], vertices: &[VertexId], weights: &[f32]) -> bool {
    let mut left = region.len();
    let mut seen = vec![false; region.len()];
    for (v, w) in vertices.iter().zip(weights) {
        if *w == 0.0
            && let Ok(k) = region.binary_search(v)
            && !seen[k]
        {
            seen[k] = true;
            left -= 1;
        }
    }
    left == 0
}

impl Scene {
    /// Applies one brush dab as one edit: the same history, reports and
    /// failure behaviour as [`Scene::edit`] with the matching
    /// [`deform`](fast_mesh_ops::deform) function, and the same result, bit
    /// for bit, except that a vertex no triangle uses is out of the brush's
    /// reach. The cost follows the vertices under the brush, plus a walk down
    /// the BVH, however large the mesh is; where the BVH cannot serve (stale
    /// under [`UpdateMode::Manual`]) it scans the mesh instead.
    ///
    /// The first dab on a mesh builds its vertex-to-triangles index, once.
    ///
    /// # Errors
    /// As for [`Scene::edit`] with the deform function:
    /// [`SceneError::UnknownMesh`] for a stale id, [`SceneError::Ops`] for an
    /// invalid brush, [`SceneError::Store`] when memory runs out. Nothing
    /// changes in any of these cases.
    pub fn dab(&mut self, id: MeshId, dab: &Dab) -> Result<DabReport> {
        self.dab_protected(id, dab, &[], &[])
    }

    /// [`Scene::dab`] under [`with_protection`]: each listed vertex's move is
    /// scaled by its weight, at a cost that follows the brush region plus the
    /// list. List only what the brush can reach.
    ///
    /// A dab with every vertex under its brush held fully (weight 0) moves
    /// nothing, so it edits nothing either: it reports 0 and leaves no undo step.
    ///
    /// # Errors
    /// As for [`Scene::dab`], and as [`check_protection`] for the list.
    /// Nothing changes in any of these cases.
    pub fn dab_protected(
        &mut self,
        id: MeshId,
        dab: &Dab,
        vertices: &[VertexId],
        weights: &[f32],
    ) -> Result<DabReport> {
        self.drop_field_unit();
        let entry = self.entries.get_mut(id).ok_or(SceneError::UnknownMesh)?;
        let region = region_of(entry, dab.brush());
        check_protection(&entry.mesh, vertices, weights)?;
        if let Some(r) = &region
            && !vertices.is_empty()
            && fully_held(r, vertices, weights)
        {
            return Ok(DabReport::default());
        }
        self.edit(id, |tx| -> std::result::Result<DabReport, SceneError> {
            let (moved, max_displacement) = with_protection(tx, vertices, weights, |tx| {
                Ok(match (dab, &region) {
                    (Dab::Inflate(o), Some(r)) => {
                        let r = inflate_region(tx, o, r)?;
                        (r.moved, r.max_displacement)
                    }
                    (Dab::Inflate(o), None) => {
                        let r = inflate(tx, o)?;
                        (r.moved, r.max_displacement)
                    }
                    (Dab::Smooth(o), Some(r)) => {
                        let r = smooth_region(tx, o, r)?;
                        (r.moved, r.max_displacement)
                    }
                    (Dab::Smooth(o), None) => {
                        let r = smooth(tx, o)?;
                        (r.moved, r.max_displacement)
                    }
                    (Dab::Grab(o), Some(r)) => {
                        let r = grab_region(tx, o, r)?;
                        (r.moved, r.max_displacement)
                    }
                    (Dab::Grab(o), None) => {
                        let r = grab(tx, o)?;
                        (r.moved, r.max_displacement)
                    }
                })
            })?;
            Ok(DabReport {
                moved,
                max_displacement,
            })
        })
    }
}

#[cfg(test)]
mod protected_tests {
    use fast_mesh_ops::deform::Falloff;
    use fast_mesh_store::Mesh;

    use super::*;
    use crate::{MeshOptions, SceneOptions};

    /// A flat `n x n` grid, with its vertices left of the middle held.
    fn setup(n: u32) -> (Scene, MeshId, Vec<VertexId>, Dab) {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                positions.extend([x as f32, y as f32, 0.0]);
            }
        }
        let mut indices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene
            .add_mesh(
                Mesh::from_buffers(&positions, &indices).unwrap(),
                MeshOptions::default(),
            )
            .unwrap();
        let local = scene.mesh(id).unwrap().positions().to_vec();
        let middle = local[((n / 2) * (n + 1) + n / 2) as usize];
        let held = local
            .iter()
            .enumerate()
            .filter(|(_, p)| p[0] < middle[0])
            .map(|(v, _)| v as VertexId)
            .collect();
        let dab = Dab::Inflate(InflateOptions {
            brush: Brush {
                center: middle,
                radius: n as f32 / 4.0,
                falloff: Falloff::Smooth,
                strength: 0.5,
            },
            ..Default::default()
        });
        (scene, id, held, dab)
    }

    #[test]
    fn holds_listed_vertices_and_matches_the_transaction_form() {
        let (mut a, id_a, held, dab) = setup(20);
        let weights = vec![0.0; held.len()];
        let before = a.mesh(id_a).unwrap().positions().to_vec();
        let report = a.dab_protected(id_a, &dab, &held, &weights).unwrap();
        assert!(report.moved > 0);
        let after = a.mesh(id_a).unwrap().positions().to_vec();
        for &v in &held {
            let (p, q) = (before[v as usize], after[v as usize]);
            assert_eq!(p.map(f32::to_bits), q.map(f32::to_bits));
        }
        assert_ne!(before, after);
        // The same as the whole-mesh deform under the same protection, bit for bit.
        let (mut b, id_b, _, _) = setup(20);
        let Dab::Inflate(o) = dab else { unreachable!() };
        b.edit(id_b, |tx| {
            with_protection(tx, &held, &weights, |tx| inflate(tx, &o))
        })
        .unwrap();
        let other = b.mesh(id_b).unwrap().positions().to_vec();
        assert_eq!(after, other);
        // One undo step, restoring everything.
        a.undo().unwrap();
        assert_eq!(a.mesh(id_a).unwrap().positions(), before.as_slice());
    }

    #[test]
    fn a_fully_held_brush_edits_nothing() {
        let (mut scene, id, _, dab) = setup(20);
        let all: Vec<VertexId> = (0..scene.mesh(id).unwrap().vertex_slots() as u32).collect();
        let weights = vec![0.0; all.len()];
        let report = scene.dab_protected(id, &dab, &all, &weights).unwrap();
        assert_eq!(report, DabReport::default());
        assert!(!scene.can_undo());
    }

    #[test]
    fn a_soft_weight_is_not_fully_held() {
        let (mut scene, id, _, dab) = setup(20);
        let all: Vec<VertexId> = (0..scene.mesh(id).unwrap().vertex_slots() as u32).collect();
        let weights = vec![0.5; all.len()];
        assert!(scene.dab_protected(id, &dab, &all, &weights).unwrap().moved > 0);
        assert!(scene.can_undo());
    }

    #[test]
    fn rejects_a_bad_list_and_changes_nothing() {
        let (mut scene, id, _, dab) = setup(4);
        assert!(scene.dab_protected(id, &dab, &[9999], &[0.0]).is_err());
        assert!(scene.dab_protected(id, &dab, &[0], &[2.0]).is_err());
        assert!(scene.dab_protected(id, &dab, &[0], &[]).is_err());
        assert!(!scene.can_undo());
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_ops::deform::{Falloff, SmoothMethod};
    use fast_mesh_store::Mesh;

    use super::*;
    use crate::{MeshOptions, SceneOptions};

    fn terrain(n: u32) -> Mesh {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                let (fx, fy) = (x as f32, y as f32);
                positions.extend([fx, fy, (fx * 0.3).sin() * (fy * 0.2).cos()]);
            }
        }
        let mut indices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        Mesh::from_buffers(&positions, &indices).unwrap()
    }

    fn scene_with(mode: UpdateMode) -> (Scene, MeshId) {
        let mut scene = Scene::new(SceneOptions::default());
        let options = MeshOptions {
            mode,
            ..MeshOptions::default()
        };
        let id = scene.add_mesh(terrain(24), options).unwrap();
        (scene, id)
    }

    fn brush(center: [f32; 3]) -> Brush {
        Brush {
            center,
            radius: 3.5,
            falloff: Falloff::Smooth,
            strength: 0.4,
        }
    }

    fn dabs() -> Vec<Dab> {
        let mut out = Vec::new();
        for k in 0..12 {
            let b = brush([6.0 + k as f32, 9.0, 0.5]);
            out.push(match k % 3 {
                0 => Dab::Inflate(InflateOptions {
                    brush: b,
                    ..Default::default()
                }),
                1 => Dab::Smooth(SmoothOptions {
                    brush: b,
                    cycles: 2,
                    method: SmoothMethod::Laplacian,
                    ..Default::default()
                }),
                _ => Dab::Grab(GrabOptions {
                    brush: b,
                    direction: [0.2, 0.1, 0.5],
                    ..Default::default()
                }),
            });
        }
        out
    }

    fn whole(scene: &mut Scene, id: MeshId, dab: &Dab) -> DabReport {
        scene
            .edit(id, |tx| -> std::result::Result<_, SceneError> {
                Ok(match dab {
                    Dab::Inflate(o) => {
                        let r = inflate(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                    Dab::Smooth(o) => {
                        let r = smooth(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                    Dab::Grab(o) => {
                        let r = grab(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                })
            })
            .unwrap()
    }

    fn bits(scene: &Scene, id: MeshId) -> Vec<[u32; 3]> {
        scene
            .mesh(id)
            .unwrap()
            .positions()
            .iter()
            .map(|p| p.map(f32::to_bits))
            .collect()
    }

    #[test]
    fn dabs_match_the_whole_mesh_deforms_bit_for_bit_in_every_mode() {
        for mode in [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual] {
            let (mut a, ia) = scene_with(mode);
            let (mut b, ib) = scene_with(mode);
            for dab in dabs() {
                let ra = a.dab(ia, &dab).unwrap();
                let rb = whole(&mut b, ib, &dab);
                assert_eq!(ra, rb, "{mode:?}");
                assert_eq!(bits(&a, ia), bits(&b, ib), "{mode:?}");
            }
        }
    }

    #[test]
    fn a_dab_is_one_undo_step_and_reports_its_change() {
        let (mut scene, id) = scene_with(UpdateMode::Eager);
        let before = bits(&scene, id);
        scene.take_changes();
        let report = scene
            .dab(
                id,
                &Dab::Inflate(InflateOptions {
                    brush: brush([12.0, 12.0, 0.0]),
                    ..Default::default()
                }),
            )
            .unwrap();
        assert!(report.moved > 0);
        assert_eq!(scene.take_changes().len(), 1);
        assert_ne!(bits(&scene, id), before);
        scene.undo().unwrap();
        assert_eq!(bits(&scene, id), before);
        assert!(!scene.can_undo());
    }

    #[test]
    fn dabs_in_an_open_step_undo_together() {
        let (mut scene, id) = scene_with(UpdateMode::Eager);
        let before = bits(&scene, id);
        scene.begin_step();
        for dab in dabs() {
            scene.dab(id, &dab).unwrap();
        }
        scene.end_step().unwrap();
        assert_ne!(bits(&scene, id), before);
        scene.undo().unwrap();
        assert_eq!(bits(&scene, id), before);
    }

    #[test]
    fn an_empty_region_records_nothing_and_a_bad_brush_fails() {
        let (mut scene, id) = scene_with(UpdateMode::Eager);
        let far = Dab::Grab(GrabOptions {
            brush: brush([1000.0, 0.0, 0.0]),
            direction: [0.0, 0.0, 1.0],
            ..Default::default()
        });
        assert_eq!(scene.dab(id, &far).unwrap(), DabReport::default());
        assert!(!scene.can_undo());
        let mut bad = brush([0.0; 3]);
        bad.radius = -1.0;
        assert!(matches!(
            scene.dab(
                id,
                &Dab::Inflate(InflateOptions {
                    brush: bad,
                    ..Default::default()
                })
            ),
            Err(SceneError::Ops(_))
        ));
    }

    #[test]
    fn a_stale_manual_mesh_still_dabs_by_scanning() {
        let (mut scene, id) = scene_with(UpdateMode::Manual);
        let dab = Dab::Inflate(InflateOptions {
            brush: brush([12.0, 12.0, 0.0]),
            ..Default::default()
        });
        scene.dab(id, &dab).unwrap();
        // The BVH is stale now: the next dab cannot use it, and must not fail.
        assert!(scene.dab(id, &dab).unwrap().moved > 0);
    }

    #[test]
    fn a_dab_reaches_vertices_grown_into_the_region_by_earlier_dabs() {
        // Drag a lump far from where the BVH was built, then dab where it is
        // now: the region search must follow the moved surface.
        let (mut scene, id) = scene_with(UpdateMode::Eager);
        let mut at = [12.0, 12.0, 0.0];
        for _ in 0..30 {
            let dab = Dab::Grab(GrabOptions {
                brush: Brush {
                    center: at,
                    radius: 2.0,
                    falloff: Falloff::Smooth,
                    strength: 1.0,
                },
                direction: [0.0, 0.0, 1.0],
                ..Default::default()
            });
            assert!(scene.dab(id, &dab).unwrap().moved > 0);
            at[2] += 1.0;
        }
        let top = scene
            .mesh(id)
            .unwrap()
            .positions()
            .iter()
            .map(|p| p[2])
            .fold(f32::MIN, f32::max);
        assert!(top > 25.0, "the lump followed: {top}");
    }
}
