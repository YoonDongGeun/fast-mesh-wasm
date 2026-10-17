//! The scene: a registry of meshes with BVHs and placements.

use fast_mesh_bvh::{BuildOptions, Bvh, UpdatePolicy};
use fast_mesh_store::{ChangeSet, Mesh};

use crate::id::SlotMap;
use crate::{MeshId, Placement, Result, SceneError};

/// When a mesh's BVH is brought up to date after the mesh changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpdateMode {
    /// Right after every commit, undo and redo of that mesh.
    #[default]
    Eager,
    /// Before the next query that needs it, or on `flush` / `update`.
    Lazy,
    /// Only on an explicit `update`; queries that need it fail instead.
    Manual,
}

/// Options for adding a mesh to a [`Scene`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MeshOptions {
    /// How the mesh's BVH is built.
    pub build: BuildOptions,
    /// When the BVH falls back to heavier update strategies.
    pub bvh_policy: UpdatePolicy,
    /// When the BVH is brought up to date.
    pub mode: UpdateMode,
    /// Where the mesh starts out in the world.
    pub placement: Placement,
}

/// Options for a [`Scene`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneOptions {
    /// Most bytes the shared undo history may hold.
    pub history_budget: usize,
}

impl Default for SceneOptions {
    /// A history budget of 256 MiB.
    fn default() -> Self {
        SceneOptions {
            history_budget: 256 << 20,
        }
    }
}

/// What happened to a mesh since the last `take_changes`.
#[derive(Default)]
// Filled in by the editing and reporting code of later tasks.
#[allow(dead_code)]
pub(crate) struct Report {
    /// Merged changes; `None` means "everything".
    pub(crate) changes: Option<ChangeSet>,
    pub(crate) placement_changed: bool,
    pub(crate) added: bool,
    pub(crate) removed: bool,
}

/// One mesh of the scene with everything attached to it.
// Fields other than `mesh`, `bvh`, `placement` and `mode` are used by the
// editing, history and query code of later tasks.
#[allow(dead_code)]
pub(crate) struct Entry {
    pub(crate) mesh: Mesh,
    pub(crate) bvh: Bvh,
    pub(crate) placement: Placement,
    pub(crate) mode: UpdateMode,
    /// Changes not yet applied to the BVH, merged into one set.
    pub(crate) pending: Option<ChangeSet>,
    pub(crate) report: Report,
}

/// Several meshes, each with a BVH and a placement in the world.
pub struct Scene {
    entries: SlotMap<Entry>,
    // Used by the shared history of a later task.
    #[allow(dead_code)]
    history_budget: usize,
}

impl Scene {
    /// An empty scene.
    pub fn new(options: SceneOptions) -> Scene {
        Scene {
            entries: SlotMap::new(),
            history_budget: options.history_budget,
        }
    }

    /// Adds a mesh, building its BVH, and returns its id.
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] for an invalid placement, or a BVH or
    /// memory error from the build. The scene is unchanged on error.
    pub fn add_mesh(&mut self, mesh: Mesh, options: MeshOptions) -> Result<MeshId> {
        options.placement.validate()?;
        let mut bvh = Bvh::build(&mesh, &options.build)?;
        bvh.set_policy(options.bvh_policy)?;
        self.entries.insert(Entry {
            mesh,
            bvh,
            placement: options.placement,
            mode: options.mode,
            pending: None,
            report: Report {
                added: true,
                ..Report::default()
            },
        })
    }

    /// Removes a mesh and returns it. The id never matches again.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale or foreign id.
    pub fn remove_mesh(&mut self, id: MeshId) -> Result<Mesh> {
        self.entries
            .remove(id)
            .map(|e| e.mesh)
            .ok_or(SceneError::UnknownMesh)
    }

    fn entry(&self, id: MeshId) -> Result<&Entry> {
        self.entries.get(id).ok_or(SceneError::UnknownMesh)
    }

    /// The mesh behind `id`.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale or foreign id.
    pub fn mesh(&self, id: MeshId) -> Result<&Mesh> {
        Ok(&self.entry(id)?.mesh)
    }

    /// The ids of all meshes, in index order.
    pub fn ids(&self) -> impl Iterator<Item = MeshId> {
        self.entries.iter().map(|(id, _)| id)
    }

    /// Where the mesh sits in the world.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale or foreign id.
    pub fn placement(&self, id: MeshId) -> Result<Placement> {
        Ok(self.entry(id)?.placement)
    }

    /// Moves the mesh. Only stores the placement; nothing else is touched.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`], or [`SceneError::InvalidInput`] when the
    /// placement is not a proper rotation. The scene is unchanged on error.
    pub fn set_placement(&mut self, id: MeshId, placement: Placement) -> Result<()> {
        placement.validate()?;
        let entry = self.entries.get_mut(id).ok_or(SceneError::UnknownMesh)?;
        entry.placement = placement;
        entry.report.placement_changed = true;
        Ok(())
    }

    /// Whether the mesh's BVH is up to date with the mesh.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale or foreign id.
    pub fn is_current(&self, id: MeshId) -> Result<bool> {
        let e = self.entry(id)?;
        Ok(e.bvh.is_current(&e.mesh))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri() -> Mesh {
        Mesh::from_buffers(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[0, 1, 2]).unwrap()
    }

    #[test]
    fn stale_ids_are_rejected() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(tri(), MeshOptions::default()).unwrap();
        assert!(scene.mesh(a).is_ok());
        assert!(scene.remove_mesh(a).is_ok());
        assert_eq!(scene.mesh(a).err(), Some(SceneError::UnknownMesh));
        assert_eq!(scene.placement(a).err(), Some(SceneError::UnknownMesh));
        assert_eq!(
            scene.set_placement(a, Placement::IDENTITY),
            Err(SceneError::UnknownMesh)
        );
        assert_eq!(scene.is_current(a).err(), Some(SceneError::UnknownMesh));
        assert_eq!(scene.remove_mesh(a).err(), Some(SceneError::UnknownMesh));
        // The slot is reused, but the old id still does not match.
        let b = scene.add_mesh(tri(), MeshOptions::default()).unwrap();
        assert_ne!(a, b);
        assert_eq!(scene.mesh(a).err(), Some(SceneError::UnknownMesh));
        assert!(scene.mesh(b).is_ok());
    }

    #[test]
    fn ids_iterate_in_index_order() {
        let mut scene = Scene::new(SceneOptions::default());
        let ids: Vec<_> = (0..4)
            .map(|_| scene.add_mesh(tri(), MeshOptions::default()).unwrap())
            .collect();
        scene.remove_mesh(ids[1]).unwrap();
        assert_eq!(scene.ids().collect::<Vec<_>>(), [ids[0], ids[2], ids[3]]);
        let again = scene.add_mesh(tri(), MeshOptions::default()).unwrap();
        // Reuses index 1, so it iterates between ids[0] and ids[2].
        assert_eq!(
            scene.ids().collect::<Vec<_>>(),
            [ids[0], again, ids[2], ids[3]]
        );
    }

    #[test]
    fn placement_round_trip_and_validation() {
        let mut scene = Scene::new(SceneOptions::default());
        let bad = Placement {
            translation: [f64::NAN, 0.0, 0.0],
            ..Placement::IDENTITY
        };
        let opts = MeshOptions {
            placement: bad,
            ..MeshOptions::default()
        };
        assert!(matches!(
            scene.add_mesh(tri(), opts),
            Err(SceneError::InvalidInput(_))
        ));
        assert_eq!(scene.ids().count(), 0);
        let id = scene.add_mesh(tri(), MeshOptions::default()).unwrap();
        let moved = Placement {
            translation: [1.0, 2.0, 3.0],
            ..Placement::IDENTITY
        };
        scene.set_placement(id, moved).unwrap();
        assert_eq!(scene.placement(id), Ok(moved));
        assert!(scene.set_placement(id, bad).is_err());
        assert_eq!(scene.placement(id), Ok(moved));
        assert_eq!(scene.is_current(id), Ok(true));
    }
}
