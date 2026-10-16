//! Errors reported by queries.

use std::fmt;

/// Why a query could not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryError {
    /// The BVH was built for a different mesh.
    MeshMismatch,
    /// The mesh changed since the BVH was last updated.
    StaleBvh,
    /// An argument was invalid (non-finite, negative radius and so on).
    InvalidInput(&'static str),
    /// Memory for the result could not be reserved.
    OutOfMemory,
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryError::MeshMismatch => f.write_str("the BVH was built for a different mesh"),
            QueryError::StaleBvh => f.write_str("the BVH is out of date with the mesh"),
            QueryError::InvalidInput(what) => write!(f, "invalid input: {what}"),
            QueryError::OutOfMemory => f.write_str("out of memory"),
        }
    }
}

impl std::error::Error for QueryError {}

/// `Result` with [`QueryError`] as the error type.
pub type Result<T> = std::result::Result<T, QueryError>;
