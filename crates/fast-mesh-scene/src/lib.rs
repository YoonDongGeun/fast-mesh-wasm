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
//! - Editing: [`Scene::edit`] runs a store transaction on one mesh and
//!   records its patch. All-or-nothing: a failed edit changes nothing.
//! - History: one undo history across all meshes, in steps.
//!   [`Scene::begin_step`] / [`Scene::end_step`] group edits and placement
//!   changes (otherwise each is its own step); [`Scene::undo`],
//!   [`Scene::redo`] and [`Scene::cancel_step`] are all-or-nothing per step.
//!   [`SceneOptions::history_budget`] evicts the oldest steps.
//! - BVH state: each mesh's [`UpdateMode`] says when its BVH follows the
//!   mesh; [`Scene::set_mode`], [`Scene::update`], [`Scene::flush`] and
//!   [`Scene::is_current`].
//! - Change reports: [`Scene::take_changes`] drains one [`MeshChanges`] per
//!   affected mesh.
//! - Maintenance: [`Scene::add_channel`], [`Scene::remove_channel`],
//!   [`Scene::derived_channel_mut`] and [`Scene::compact`].

mod error;
mod history;
mod id;
mod placement;
mod report;
mod scene;

pub use error::{Result, SceneError};
pub use id::MeshId;
pub use placement::Placement;
pub use report::MeshChanges;
pub use scene::{MeshOptions, Scene, SceneOptions, UpdateMode};
