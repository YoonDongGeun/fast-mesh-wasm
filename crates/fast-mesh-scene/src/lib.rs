//! Scenes of several editable meshes, built on [`fast_mesh_store`],
//! [`fast_mesh_bvh`] and [`fast_mesh_query`].
//!
//! A [`Scene`] owns meshes, each with a BVH, an [`UpdateMode`] and a
//! [`Placement`] (a rigid transform into the world). Meshes are addressed by
//! generational [`MeshId`]s: once a mesh is removed, its id is rejected with
//! [`SceneError::UnknownMesh`] for good.
//!
//! - Registry: [`Scene::add_mesh`], [`Scene::remove_mesh`], [`Scene::mesh`]
//!   and [`Scene::ids`] (in index order).
//! - Placements: [`Scene::placement`] and [`Scene::set_placement`]. A
//!   placement is validated like a [`fast_mesh_query::Rigid`] and stored
//!   exactly as given.
//! - BVH state: [`Scene::is_current`] says whether a mesh's BVH is up to date.

mod error;
mod id;
mod placement;
mod scene;

pub use error::{Result, SceneError};
pub use id::MeshId;
pub use placement::Placement;
pub use scene::{MeshOptions, Scene, SceneOptions, UpdateMode};
