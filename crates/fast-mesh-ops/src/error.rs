//! Errors reported by repair operations.

use std::fmt;

use fast_mesh_query::QueryError;
use fast_mesh_store::StoreError;

/// Why a repair operation could not run.
///
/// Every operation computes and validates its whole plan before writing, so
/// an error leaves the mesh as it was, except where the store's own error says
/// otherwise ([`StoreError::TransactionPoisoned`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpsError {
    /// The store failed, including running out of memory
    /// ([`StoreError::OutOfMemory`]).
    Store(StoreError),
    /// A query used by the operation failed.
    Query(QueryError),
    /// An argument was invalid; the message says which and why.
    InvalidInput(&'static str),
    /// A requested boundary loop no longer exists in the mesh.
    StaleLoops,
}

impl From<StoreError> for OpsError {
    fn from(e: StoreError) -> Self {
        OpsError::Store(e)
    }
}

/// A query running out of memory becomes [`OpsError::Store`] with
/// [`StoreError::OutOfMemory`], like every other allocation failure here.
impl From<QueryError> for OpsError {
    fn from(e: QueryError) -> Self {
        match e {
            QueryError::OutOfMemory => OpsError::Store(StoreError::OutOfMemory),
            e => OpsError::Query(e),
        }
    }
}

/// Field kernel errors map out-of-memory to [`OpsError::Store`] with
/// [`StoreError::OutOfMemory`], store and query errors to their counterparts,
/// and invalid inputs directly.
impl From<fast_mesh_fields::FieldsError> for OpsError {
    fn from(e: fast_mesh_fields::FieldsError) -> Self {
        match e {
            fast_mesh_fields::FieldsError::OutOfMemory => OpsError::Store(StoreError::OutOfMemory),
            fast_mesh_fields::FieldsError::Store(s) => OpsError::Store(s),
            fast_mesh_fields::FieldsError::Query(q) => OpsError::from(q),
            fast_mesh_fields::FieldsError::InvalidInput(msg) => OpsError::InvalidInput(msg),
        }
    }
}

impl fmt::Display for OpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpsError::Store(e) => write!(f, "store error: {e}"),
            OpsError::Query(e) => write!(f, "query error: {e}"),
            OpsError::InvalidInput(what) => write!(f, "invalid input: {what}"),
            OpsError::StaleLoops => {
                f.write_str("a requested boundary loop no longer matches the mesh")
            }
        }
    }
}

impl std::error::Error for OpsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OpsError::Store(e) => Some(e),
            OpsError::Query(e) => Some(e),
            _ => None,
        }
    }
}

/// `Result` with [`OpsError`] as the error type.
pub type Result<T> = std::result::Result<T, OpsError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn conversions_display_and_source() {
        let e: OpsError = StoreError::OutOfMemory.into();
        assert_eq!(e, OpsError::Store(StoreError::OutOfMemory));
        assert_eq!(e.to_string(), "store error: out of memory");
        assert!(e.source().is_some());
        let q: OpsError = QueryError::StaleBvh.into();
        assert_eq!(q, OpsError::Query(QueryError::StaleBvh));
        assert!(q.source().is_some());
        let oom: OpsError = QueryError::OutOfMemory.into();
        assert_eq!(oom, OpsError::Store(StoreError::OutOfMemory));
        assert_eq!(
            OpsError::InvalidInput("no bound").to_string(),
            "invalid input: no bound"
        );
        assert!(OpsError::StaleLoops.source().is_none());

        let f_oom: OpsError = fast_mesh_fields::FieldsError::OutOfMemory.into();
        assert_eq!(f_oom, OpsError::Store(StoreError::OutOfMemory));
        let f_store: OpsError =
            fast_mesh_fields::FieldsError::Store(StoreError::InvalidTriangle(3)).into();
        assert_eq!(f_store, OpsError::Store(StoreError::InvalidTriangle(3)));
        let f_query: OpsError = fast_mesh_fields::FieldsError::Query(QueryError::StaleBvh).into();
        assert_eq!(f_query, OpsError::Query(QueryError::StaleBvh));
        let f_in: OpsError = fast_mesh_fields::FieldsError::InvalidInput("bad grid").into();
        assert_eq!(f_in, OpsError::InvalidInput("bad grid"));
    }
}
