//! Errors reported by the scene.

use std::fmt;

use fast_mesh_bvh::BvhError;
use fast_mesh_fields::FieldsError;
use fast_mesh_query::QueryError;
use fast_mesh_store::StoreError;

use crate::MeshId;

/// Why a scene operation could not run. A failed operation leaves the scene
/// unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SceneError {
    /// The id is stale: its mesh was removed.
    UnknownMesh,
    /// The field id is unknown: the field was removed, by hand or with the
    /// mesh it was on or measured to.
    UnknownField,
    /// The mesh's BVH is out of date and its update mode forbids updating it
    /// implicitly.
    StaleBvh(MeshId),
    /// An argument was invalid (non-finite, not a rotation and so on).
    InvalidInput(&'static str),
    /// There is no open history step to end or cancel.
    StepNotOpen,
    /// An internal invariant was found broken.
    Corrupted,
    /// The mesh store reported an error (including `OutOfMemory`).
    Store(StoreError),
    /// The BVH reported an error (including `OutOfMemory`).
    Bvh(BvhError),
    /// A query reported an error (including `OutOfMemory`).
    Query(QueryError),
    /// A field kernel reported an error (including `OutOfMemory`).
    Fields(FieldsError),
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SceneError::UnknownMesh => f.write_str("unknown or removed mesh id"),
            SceneError::UnknownField => f.write_str("unknown or removed field id"),
            SceneError::StaleBvh(id) => write!(f, "the BVH of mesh {id:?} is out of date"),
            SceneError::InvalidInput(what) => write!(f, "invalid input: {what}"),
            SceneError::StepNotOpen => f.write_str("no history step is open"),
            SceneError::Corrupted => f.write_str("scene state is corrupted"),
            SceneError::Store(e) => write!(f, "mesh store error: {e}"),
            SceneError::Bvh(e) => write!(f, "BVH error: {e}"),
            SceneError::Query(e) => write!(f, "query error: {e}"),
            SceneError::Fields(e) => write!(f, "field error: {e}"),
        }
    }
}

impl std::error::Error for SceneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SceneError::Store(e) => Some(e),
            SceneError::Bvh(e) => Some(e),
            SceneError::Query(e) => Some(e),
            SceneError::Fields(e) => Some(e),
            _ => None,
        }
    }
}

impl From<StoreError> for SceneError {
    fn from(e: StoreError) -> Self {
        SceneError::Store(e)
    }
}

impl From<BvhError> for SceneError {
    fn from(e: BvhError) -> Self {
        SceneError::Bvh(e)
    }
}

impl From<QueryError> for SceneError {
    fn from(e: QueryError) -> Self {
        SceneError::Query(e)
    }
}

impl From<FieldsError> for SceneError {
    fn from(e: FieldsError) -> Self {
        SceneError::Fields(e)
    }
}

/// `Result` with [`SceneError`] as the error type.
pub type Result<T> = std::result::Result<T, SceneError>;
