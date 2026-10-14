//! Editable triangle mesh storage with transactional undo/redo.
//!
//! - [`Mesh`] stores vertices, triangles and per-vertex channels in slots whose ids
//!   stay stable across edits, undo and redo.
//! - [`Mesh::begin_edit`] opens a [`Transaction`]. Every write goes through it, and
//!   [`Transaction::commit`] returns a [`Patch`] holding only what changed.
//! - [`History`] keeps patches within a memory budget and provides undo/redo.
//! - Every commit, undo and redo reports a [`ChangeSet`] so that dependent data
//!   (BVHs, fields, render buffers) can update just the affected elements.

mod channel;
mod column;
mod error;
mod mesh;

pub use channel::{ChannelDesc, ChannelKind, TopologyPolicy};
pub use error::StoreError;
pub use mesh::Mesh;

/// Id of a vertex slot.
pub type VertexId = u32;
/// Id of a triangle slot.
pub type TriangleId = u32;

pub type Result<T> = std::result::Result<T, StoreError>;
