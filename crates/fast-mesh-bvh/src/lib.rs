//! Incrementally updatable bounding volume hierarchy for meshes stored in
//! [`fast_mesh_store`].
//!
//! - [`Bvh::build`] builds over a mesh's alive triangles in mesh-local space,
//!   with a binned SAH ([`Quality::Sah`]) or Morton-code ([`Quality::Fast`]) split.
//! - [`Bvh::update`] takes the store's [`fast_mesh_store::ChangeSet`] and refits,
//!   rebuilds the affected treelets, or rebuilds everything, whichever is cheapest.
//! - [`Bvh::query_aabb`] finds the triangles whose boxes overlap a box.
//! - [`Bvh::root`], [`Bvh::node`], [`Bvh::parent`], [`Bvh::traverse`],
//!   [`Bvh::traverse_from`], [`Bvh::traverse_by_key`] and
//!   [`Bvh::traverse_ordered`] expose the tree read-only, for building other
//!   queries on top of it. No traversal allocates on the heap or fails.
//!
//! Every fallible operation is failure-atomic: all memory it needs is reserved
//! before the tree is touched, so on error the BVH is left as it was. The two
//! optional follow-up steps of [`Bvh::update`] (rebuilding degraded treelets and
//! compacting storage) never report errors: if memory runs out they are skipped
//! and the tree stays correct.

mod aabb;
mod arena;
mod build;
mod error;
mod node;
mod options;
mod tree;
mod update;

pub use aabb::Aabb;
pub use error::{BvhError, Result};
pub use node::{NodeId, NodeKind, NodeView, TRAVERSE_STACK};
pub use options::{BuildOptions, Quality, UpdatePolicy};
pub use tree::{Bvh, BvhBuild};
pub use update::{Strategy, UpdateReport};
