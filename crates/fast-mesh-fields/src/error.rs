//! Errors reported by field kernels.

use std::fmt;

use fast_mesh_query::QueryError;
use fast_mesh_store::StoreError;

/// Why a field kernel could not run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldsError {
    /// An argument was invalid (non-finite point, mismatched lengths, stale
    /// sign data and so on); the message says which.
    InvalidInput(&'static str),
    /// Memory for the result could not be reserved.
    OutOfMemory,
    /// The mesh store failed, for example while building adjacency.
    Store(StoreError),
    /// A query failed, for example because the BVH is stale.
    Query(QueryError),
}

impl fmt::Display for FieldsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldsError::InvalidInput(what) => write!(f, "invalid input: {what}"),
            FieldsError::OutOfMemory => f.write_str("out of memory"),
            FieldsError::Store(e) => write!(f, "store error: {e}"),
            FieldsError::Query(e) => write!(f, "query error: {e}"),
        }
    }
}

impl std::error::Error for FieldsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FieldsError::Store(e) => Some(e),
            FieldsError::Query(e) => Some(e),
            _ => None,
        }
    }
}

impl From<StoreError> for FieldsError {
    fn from(e: StoreError) -> Self {
        match e {
            StoreError::OutOfMemory => FieldsError::OutOfMemory,
            e => FieldsError::Store(e),
        }
    }
}

impl From<QueryError> for FieldsError {
    fn from(e: QueryError) -> Self {
        match e {
            QueryError::OutOfMemory => FieldsError::OutOfMemory,
            e => FieldsError::Query(e),
        }
    }
}

/// `Result` with [`FieldsError`] as the error type.
pub type Result<T> = std::result::Result<T, FieldsError>;
