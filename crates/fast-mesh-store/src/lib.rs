//! Editable triangle mesh storage with transactional undo/redo.
//!
//! - [`Mesh`] stores vertices, triangles and per-vertex channels in slots whose ids
//!   stay stable across edits, undo and redo.
//! - [`Mesh::begin_edit`] opens a [`Transaction`]. Every write goes through it, and
//!   [`Transaction::commit`] returns a [`Patch`] holding only what changed.
//! - [`History`] keeps patches within a memory budget and provides undo/redo.
//! - Every commit, undo and redo reports a [`ChangeSet`] so that dependent data
//!   (BVHs, fields, render buffers) can update just the affected elements.

mod adjacency;
mod channel;
mod column;
mod compact;
mod error;
mod history;
mod incidence;
mod mesh;
#[doc(hidden)]
pub mod par;
mod patch;
pub mod scratch;
mod transaction;

pub use adjacency::{Adjacency, AdjacencyBuild, EdgeKind};
pub use channel::{ChannelDesc, ChannelKind, TopologyPolicy, UV_CHANNEL};
pub use compact::Remap;
pub use error::StoreError;
pub use history::History;
pub use incidence::{Incidence, IncidenceBuild};
pub use mesh::{Mesh, MeshBuilder};
pub use patch::{ChangeSet, Patch, PatchSide, PatchState};
pub use scratch::{
    DEFAULT_MAX_RETAINED_ELEMS, SCRATCH_F32, SCRATCH_F64, SCRATCH_U8, SCRATCH_U32, SCRATCH_U64,
    SCRATCH_USIZE, ScratchF32, ScratchF64, ScratchPool, ScratchU8, ScratchU32, ScratchU64,
    ScratchUsize, ScratchVec,
};
pub use transaction::Transaction;

/// Id of a vertex slot.
pub type VertexId = u32;
/// Id of a triangle slot.
pub type TriangleId = u32;

pub type Result<T> = std::result::Result<T, StoreError>;
