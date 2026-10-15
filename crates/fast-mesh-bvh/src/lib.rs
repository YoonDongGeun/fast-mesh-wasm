//! Incrementally updatable bounding volume hierarchy for meshes stored in
//! [`fast_mesh_store`].
//!
//! - [`Bvh::build`] builds over a mesh's alive triangles in mesh-local space,
//!   with a binned SAH ([`Quality::Sah`]) or Morton-code ([`Quality::Fast`]) split.
//! - [`Bvh::update`] takes the store's [`fast_mesh_store::ChangeSet`] and refits,
//!   rebuilds the affected treelets, or rebuilds everything, whichever is cheapest.
//! - [`Bvh::query_aabb`] finds the triangles whose boxes overlap a box.
//!
//! Every operation is failure-atomic: on error the BVH is left as it was.

mod aabb;
mod arena;
mod build;
mod error;
mod options;
mod tree;

pub use aabb::Aabb;
pub use error::{BvhError, Result};
pub use options::{BuildOptions, Quality, UpdatePolicy};
pub use tree::Bvh;
