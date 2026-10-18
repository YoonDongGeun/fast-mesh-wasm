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
//!   affected mesh. Further consumers each get their own copy of every
//!   report: [`Scene::subscribe`], [`Scene::take_changes_for`] and
//!   [`Scene::unsubscribe`].
//! - Maintenance: [`Scene::add_channel`], [`Scene::remove_channel`],
//!   [`Scene::derived_channel_mut`] (all three reported, and the `field:`
//!   prefix reserved) and [`Scene::compact`].
//! - Local space: [`Scene::bvh`] (current BVHs only),
//!   [`Scene::relative_transform`], [`Scene::world_to_local`] and
//!   [`Scene::local_to_world`].
//! - World queries, in world `f64` coordinates: [`Scene::raycast`],
//!   [`Scene::closest_point`] and [`Scene::closest_point_on`], and between
//!   two meshes [`Scene::intersects`], [`Scene::mesh_distance`] and
//!   [`Scene::triangles_near`]. Each maps into mesh-local space, runs the
//!   Phase 3 query and maps the result back; bounds are rounded outward to
//!   `f32` and results filtered against the exact bound. Queries over all
//!   meshes update every Lazy BVH and fail with [`SceneError::StaleBvh`] on
//!   a stale Manual one; queries naming meshes touch only those.
//! - Contacts: [`Scene::find_contacts`] finds the mesh pairs within a margin
//!   (broadphase over world boxes, then Phase 3b per pair), optionally
//!   restricted by a [`ContactFilter`].

mod contacts;
mod error;
mod history;
mod id;
mod placement;
mod query;
mod report;
mod scene;

pub use contacts::{Contact, ContactFilter};
pub use error::{Result, SceneError};
pub use id::MeshId;
pub use placement::Placement;
pub use query::{
    SceneClosestHit, SceneClosestOptions, SceneMeshDistance, SceneRayHit, SceneRayOptions,
};
pub use report::{FieldId, MeshChanges, SubscriberId};
pub use scene::{MeshOptions, Scene, SceneOptions, UpdateMode};
