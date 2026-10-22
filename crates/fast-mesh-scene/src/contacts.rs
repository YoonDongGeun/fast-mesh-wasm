//! Broadphase and contacts between the meshes of a scene.

use fast_mesh_store::{StoreError, TriangleId};

use crate::placement::relative_rigid;
use crate::query::{Aabb64, MARGIN, check_bound, pair_pad, round_up, try_vec, world_box};
use crate::{MeshId, Result, Scene, SceneError};

/// Two meshes within the margin of [`Scene::find_contacts`].
#[derive(Debug, Clone, PartialEq)]
pub struct Contact {
    /// The mesh with the lower index.
    pub a: MeshId,
    /// The mesh with the higher index.
    pub b: MeshId,
    /// Whether the two meshes intersect or touch, decided exactly.
    pub intersecting: bool,
    /// The distance between the two surfaces; 0 when `intersecting`.
    pub distance: f64,
    /// The triangles of `a` within the margin of `b`, ascending.
    pub triangles_a: Vec<TriangleId>,
    /// The triangles of `b` within the margin of `a`, ascending.
    pub triangles_b: Vec<TriangleId>,
}

/// Which mesh pairs [`Scene::find_contacts`] looks at.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ContactFilter {
    /// Only these pairs, in any order and either way round; `None` means
    /// every pair of meshes.
    pub pairs: Option<Vec<(MeshId, MeshId)>>,
}

impl Scene {
    /// Appends a [`Contact`] for every mesh pair (of `filter`, or of the whole
    /// scene) whose surfaces are within `margin` of each other as placed in
    /// the world, sorted by `(a, b)`, and returns how many were appended.
    /// Elements already in `out` stay in front.
    ///
    /// The broadphase keeps the pairs whose world boxes (BVH root boxes
    /// placed by the rigid transforms) are within `margin`, padded so that it
    /// never drops a pair Phase 3b would report. For each, in A's space:
    /// `intersecting` comes from [`fast_mesh_query::intersects`]; `distance`
    /// is 0 when intersecting, else from [`fast_mesh_query::mesh_distance`]
    /// with `margin` rounded up to `f32`, and a pair whose distance exceeds
    /// the exact `margin` is not reported; the triangle lists come from
    /// [`fast_mesh_query::triangles_near`] with the rounded-up margin (see
    /// [`Scene::triangles_near`]).
    ///
    /// With `filter.pairs`, each pair may be given either way round and more
    /// than once; only the listed meshes' BVHs are brought up to date.
    /// Without, every Lazy BVH is (§15.4).
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] for a negative or NaN `margin` or a pair
    /// naming one mesh twice; [`SceneError::UnknownMesh`] for a stale id in
    /// the filter; then a BVH or memory error from a Lazy update,
    /// [`SceneError::StaleBvh`] for the lowest stale Manual mesh involved, or
    /// a memory error while collecting. On any error `out` keeps its length.
    pub fn find_contacts(
        &mut self,
        margin: f64,
        filter: &ContactFilter,
        out: &mut Vec<Contact>,
    ) -> Result<usize> {
        self.drop_field_unit();
        let start = out.len();
        let found = self.find_contacts_into(margin, filter, out);
        if found.is_err() {
            out.truncate(start);
        }
        found
    }

    fn find_contacts_into(
        &mut self,
        margin: f64,
        filter: &ContactFilter,
        out: &mut Vec<Contact>,
    ) -> Result<usize> {
        check_bound(margin, MARGIN)?;
        // The meshes involved, ascending, and the pairs as indices into them.
        let (ids, pairs) = match &filter.pairs {
            Some(list) => {
                let mut pairs = try_vec(list.len())?;
                for &(a, b) in list {
                    self.entry(a)?;
                    self.entry(b)?;
                    if a == b {
                        return Err(SceneError::InvalidInput(
                            "a contact pair needs two different meshes",
                        ));
                    }
                    pairs.push((a.min(b), a.max(b)));
                }
                pairs.sort_unstable();
                pairs.dedup();
                let mut ids = try_vec(2 * pairs.len())?;
                for &(a, b) in &pairs {
                    ids.push(a);
                    ids.push(b);
                }
                ids.sort_unstable();
                ids.dedup();
                self.prepare(Some(&ids))?;
                (ids, Some(pairs))
            }
            None => {
                self.prepare(None)?;
                let mut ids = try_vec(self.entries.iter().count())?;
                ids.extend(self.ids());
                (ids, None)
            }
        };
        let mut boxes = try_vec(ids.len())?;
        for &id in &ids {
            boxes.push(world_box(self.entry(id)?));
        }
        let start = out.len();
        match pairs {
            Some(pairs) => {
                for (a, b) in pairs {
                    // Both are in `ids`, which is sorted.
                    let i = ids.binary_search(&a).map_err(|_| SceneError::Corrupted)?;
                    let j = ids.binary_search(&b).map_err(|_| SceneError::Corrupted)?;
                    self.contact((a, &boxes[i]), (b, &boxes[j]), margin, out)?;
                }
            }
            None => {
                for i in 0..ids.len() {
                    for j in i + 1..ids.len() {
                        self.contact((ids[i], &boxes[i]), (ids[j], &boxes[j]), margin, out)?;
                    }
                }
            }
        }
        Ok(out.len() - start)
    }

    /// Runs the broadphase test and, if it passes, the narrow phase for one
    /// pair (`a` below `b`), appending its contact if within `margin`.
    fn contact(
        &self,
        (a, wa): (MeshId, &Aabb64),
        (b, wb): (MeshId, &Aabb64),
        margin: f64,
        out: &mut Vec<Contact>,
    ) -> Result<()> {
        if wa.is_empty() || wb.is_empty() {
            return Ok(());
        }
        let (ea, eb) = (self.entry(a)?, self.entry(b)?);
        if wa.distance_to_box(wb) > margin + pair_pad(ea, wa, eb, wb) {
            return Ok(());
        }
        let xf = relative_rigid(
            &ea.placement,
            ea.mesh.origin(),
            &eb.placement,
            eb.mesh.origin(),
        );
        let (ma, ba, mb, bb) = (&ea.mesh, &ea.bvh, &eb.mesh, &eb.bvh);
        let m32 = round_up(margin);
        let intersecting = fast_mesh_query::intersects(ma, ba, mb, bb, &xf)?;
        let distance = if intersecting {
            0.0
        } else {
            match fast_mesh_query::mesh_distance(ma, ba, mb, bb, &xf, m32)? {
                Some(d) if f64::from(d.distance) <= margin => f64::from(d.distance),
                _ => return Ok(()),
            }
        };
        let (mut triangles_a, mut triangles_b) = (Vec::new(), Vec::new());
        fast_mesh_query::triangles_near(
            ma,
            ba,
            mb,
            bb,
            &xf,
            m32,
            &mut triangles_a,
            &mut triangles_b,
        )?;
        out.try_reserve(1)
            .map_err(|_| SceneError::Store(StoreError::OutOfMemory))?;
        out.push(Contact {
            a,
            b,
            intersecting,
            distance,
            triangles_a,
            triangles_b,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::relative_rigid;
    use crate::query::tests::{RX90, RZ90, cube, place};
    use crate::{MeshOptions, Placement, SceneOptions, UpdateMode};

    /// Every pair `a < b` by calling Phase 3b directly.
    fn brute(scene: &Scene, margin: f64, pairs: &[(MeshId, MeshId)]) -> Vec<Contact> {
        let m32 = margin as f32;
        assert_eq!(f64::from(m32), margin, "test margins are exact in f32");
        let mut out = Vec::new();
        for &(a, b) in pairs {
            let (ea, eb) = (scene.entries.get(a).unwrap(), scene.entries.get(b).unwrap());
            let xf = relative_rigid(
                &ea.placement,
                ea.mesh.origin(),
                &eb.placement,
                eb.mesh.origin(),
            );
            let (ma, ba, mb, bb) = (&ea.mesh, &ea.bvh, &eb.mesh, &eb.bvh);
            let intersecting = fast_mesh_query::intersects(ma, ba, mb, bb, &xf).unwrap();
            let distance = if intersecting {
                0.0
            } else {
                match fast_mesh_query::mesh_distance(ma, ba, mb, bb, &xf, m32).unwrap() {
                    Some(d) => f64::from(d.distance),
                    None => continue,
                }
            };
            let (mut ta, mut tb) = (Vec::new(), Vec::new());
            fast_mesh_query::triangles_near(ma, ba, mb, bb, &xf, m32, &mut ta, &mut tb).unwrap();
            out.push(Contact {
                a,
                b,
                intersecting,
                distance,
                triangles_a: ta,
                triangles_b: tb,
            });
        }
        out
    }

    fn all_pairs(scene: &Scene) -> Vec<(MeshId, MeshId)> {
        let ids: Vec<MeshId> = scene.ids().collect();
        let mut pairs = Vec::new();
        for (i, &a) in ids.iter().enumerate() {
            for &b in &ids[i + 1..] {
                pairs.push((a, b));
            }
        }
        pairs
    }

    /// A rotation by 45 degrees about z.
    fn rz45() -> [[f64; 3]; 3] {
        let (s, c) = 45f64.to_radians().sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    fn scene() -> (Scene, Vec<MeshId>) {
        let mut scene = Scene::new(SceneOptions::default());
        let identity = Placement::IDENTITY.rotation;
        let ids = vec![
            // [0, 2]^3.
            scene
                .add_mesh(cube([0.0; 3], 2.0), MeshOptions::default())
                .unwrap(),
            // x in [1.5, 2.5]: crosses the first.
            scene
                .add_mesh(cube([0.0; 3], 1.0), place(RZ90, [2.5, 0.5, 0.5]))
                .unwrap(),
            // y in [2.25, 3.25]: 0.25 from the first.
            scene
                .add_mesh(cube([0.0; 3], 1.0), place(RX90, [0.5, 3.25, 1.0]))
                .unwrap(),
            // Far from everything.
            scene
                .add_mesh(cube([0.0; 3], 1.0), place(identity, [10.0, 10.0, 10.0]))
                .unwrap(),
            // Turned 45 degrees, off the first's edge x = 0, y = 2: its box
            // is about 0.41 away, its surface about 0.91.
            scene
                .add_mesh(
                    cube([-0.5, -0.5, 0.0], 1.0),
                    place(rz45(), [-1.0, 3.0, 0.5]),
                )
                .unwrap(),
            // 0.6 beyond the first along +z: outside a 0.5 margin.
            scene
                .add_mesh(cube([0.0; 3], 1.0), place(identity, [0.5, 0.5, 2.6]))
                .unwrap(),
        ];
        (scene, ids)
    }

    #[test]
    fn find_contacts_matches_all_pairs() {
        let (mut scene, ids) = scene();
        let pairs = all_pairs(&scene);
        for margin in [0.0, 0.25, 0.5, 1.0, 4.0, f64::INFINITY] {
            let want = if margin.is_finite() {
                brute(&scene, margin, &pairs)
            } else {
                brute(&scene, f64::from(f32::MAX), &pairs)
            };
            let mut out = Vec::new();
            let n = scene
                .find_contacts(margin, &ContactFilter::default(), &mut out)
                .unwrap();
            assert_eq!(n, out.len());
            assert_eq!(out, want, "margin {margin}");
            if !margin.is_finite() {
                assert_eq!(out.len(), pairs.len());
            }
        }
        let mut out = Vec::new();
        scene
            .find_contacts(0.5, &ContactFilter::default(), &mut out)
            .unwrap();
        let found: Vec<_> = out
            .iter()
            .map(|c| (c.a, c.b, c.intersecting, c.distance))
            .collect();
        // The turned cube passes the broadphase but not the narrow phase.
        let (e0, e4) = (
            scene.entries.get(ids[0]).unwrap(),
            scene.entries.get(ids[4]).unwrap(),
        );
        let box_gap = world_box(e0).distance_to_box(&world_box(e4));
        assert!((0.4..0.45).contains(&box_gap), "{box_gap}");
        assert!(out.iter().all(|c| (c.a, c.b) != (ids[0], ids[4])));
        let d = scene.mesh_distance(ids[0], ids[4], 1.0).unwrap().unwrap();
        assert!((0.9..0.95).contains(&d.distance), "{}", d.distance);
        assert!(found.contains(&(ids[0], ids[1], true, 0.0)));
        assert!(found.contains(&(ids[0], ids[2], false, 0.25)));
        assert!(out.iter().all(|c| c.a != ids[3] && c.b != ids[3]));
        assert!(out.iter().all(|c| c.b != ids[5] || c.a != ids[0]));
        assert!(out.windows(2).all(|w| (w[0].a, w[0].b) < (w[1].a, w[1].b)));
    }

    #[test]
    fn find_contacts_with_a_pair_filter() {
        let (mut scene, ids) = scene();
        let filter = ContactFilter {
            pairs: Some(vec![
                (ids[2], ids[0]),
                (ids[0], ids[2]),
                (ids[3], ids[1]),
                (ids[1], ids[0]),
            ]),
        };
        let mut out = vec![];
        let n = scene.find_contacts(0.5, &filter, &mut out).unwrap();
        let want = brute(&scene, 0.5, &[(ids[0], ids[1]), (ids[0], ids[2])]);
        assert_eq!(n, 2);
        assert_eq!(out, want);
        // Existing elements stay in front.
        let n = scene.find_contacts(0.5, &filter, &mut out).unwrap();
        assert_eq!(n, 2);
        assert_eq!(out[..2], out[2..]);
    }

    #[test]
    fn find_contacts_staleness_and_errors() {
        let (mut scene, ids) = scene();
        let keep = || {
            vec![Contact {
                a: ids[0],
                b: ids[1],
                intersecting: false,
                distance: 9.0,
                triangles_a: vec![],
                triangles_b: vec![],
            }]
        };
        // A stale Manual mesh fails a search over all pairs, and one naming
        // it; `out` is left as it was.
        scene.set_mode(ids[3], UpdateMode::Manual).unwrap();
        scene
            .edit(ids[3], |tx| tx.set_positions(&[0], &[[0.0, 0.0, -0.5]]))
            .unwrap();
        let lazy = ids[4];
        scene.set_mode(lazy, UpdateMode::Lazy).unwrap();
        scene
            .edit(lazy, |tx| tx.set_positions(&[0], &[[-0.5, -0.5, -0.25]]))
            .unwrap();
        let mut out = keep();
        let all = ContactFilter::default();
        assert_eq!(
            scene.find_contacts(0.5, &all, &mut out),
            Err(SceneError::StaleBvh(ids[3]))
        );
        assert_eq!(out, keep());
        let with = ContactFilter {
            pairs: Some(vec![(ids[0], ids[3])]),
        };
        assert_eq!(
            scene.find_contacts(0.5, &with, &mut out),
            Err(SceneError::StaleBvh(ids[3]))
        );
        assert_eq!(out, keep());
        // The search over all pairs still updated the Lazy mesh first.
        assert_eq!(scene.is_current(lazy), Ok(true));
        scene
            .edit(lazy, |tx| tx.set_positions(&[0], &[[-0.5, -0.5, -0.3]]))
            .unwrap();
        // A filter that leaves it out runs, and updates only what it names.
        scene
            .edit(ids[2], |tx| tx.set_positions(&[0], &[[0.0, 0.0, -0.1]]))
            .unwrap();
        scene.set_mode(ids[2], UpdateMode::Lazy).unwrap();
        scene
            .edit(ids[2], |tx| tx.set_positions(&[0], &[[0.0, 0.0, -0.2]]))
            .unwrap();
        let without = ContactFilter {
            pairs: Some(vec![(ids[0], ids[2])]),
        };
        assert_eq!(scene.find_contacts(0.5, &without, &mut out), Ok(1));
        assert_eq!(scene.is_current(ids[2]), Ok(true));
        assert_eq!(scene.is_current(lazy), Ok(false));
        out.truncate(1);
        // Bad inputs.
        let same = ContactFilter {
            pairs: Some(vec![(ids[0], ids[0])]),
        };
        assert!(matches!(
            scene.find_contacts(0.5, &same, &mut out),
            Err(SceneError::InvalidInput(_))
        ));
        for margin in [-1.0, f64::NAN] {
            assert!(matches!(
                scene.find_contacts(margin, &without, &mut out),
                Err(SceneError::InvalidInput(_))
            ));
        }
        let gone = ids[5];
        scene.remove_mesh(gone).unwrap();
        let unknown = ContactFilter {
            pairs: Some(vec![(ids[0], ids[2]), (gone, ids[0])]),
        };
        assert_eq!(
            scene.find_contacts(0.5, &unknown, &mut out),
            Err(SceneError::UnknownMesh)
        );
        assert_eq!(out, keep());
    }
}
