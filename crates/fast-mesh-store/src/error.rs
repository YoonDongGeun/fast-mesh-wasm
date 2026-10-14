use std::fmt;

/// Errors returned by mesh storage operations.
///
/// Validation errors are returned before anything is written, so a transaction
/// stays usable after one. `TransactionPoisoned` and `OutOfMemory` are the exceptions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// The vertex id is out of range or refers to a removed vertex.
    InvalidVertex(u32),
    /// The triangle id is out of range or refers to a removed triangle.
    InvalidTriangle(u32),
    /// The vertex cannot be removed while triangles still use it.
    VertexInUse { vertex: u32, corners: u32 },
    /// Two related inputs have inconsistent lengths.
    LengthMismatch { expected: usize, got: usize },
    /// An argument is invalid; the message says which and why.
    InvalidArgument(&'static str),
    /// No channel with this name exists.
    UnknownChannel(String),
    /// A channel with this name already exists.
    ChannelExists(String),
    /// The operation needs a source channel.
    NotSourceChannel(String),
    /// The operation needs a derived channel.
    NotDerivedChannel(String),
    /// The patch belongs to another mesh or does not match the mesh's current revision.
    PatchMismatch,
    /// The patch was created before `compact()` renumbered the mesh.
    StalePatch,
    /// An earlier operation in this transaction failed after writing; the transaction was rolled back.
    TransactionPoisoned,
    /// A memory allocation failed.
    OutOfMemory,
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVertex(v) => write!(f, "vertex {v} does not exist"),
            Self::InvalidTriangle(t) => write!(f, "triangle {t} does not exist"),
            Self::VertexInUse { vertex, corners } => {
                write!(
                    f,
                    "vertex {vertex} is still used by {corners} triangle corners"
                )
            }
            Self::LengthMismatch { expected, got } => {
                write!(f, "expected length {expected}, got {got}")
            }
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::UnknownChannel(name) => write!(f, "channel \"{name}\" does not exist"),
            Self::ChannelExists(name) => write!(f, "channel \"{name}\" already exists"),
            Self::NotSourceChannel(name) => write!(f, "channel \"{name}\" is not a source channel"),
            Self::NotDerivedChannel(name) => {
                write!(f, "channel \"{name}\" is not a derived channel")
            }
            Self::PatchMismatch => write!(f, "patch does not match this mesh's current revision"),
            Self::StalePatch => write!(f, "patch was created before the mesh was compacted"),
            Self::TransactionPoisoned => {
                write!(
                    f,
                    "transaction was rolled back because an earlier operation failed"
                )
            }
            Self::OutOfMemory => write!(f, "out of memory"),
        }
    }
}

impl std::error::Error for StoreError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_the_problem() {
        assert_eq!(
            StoreError::InvalidVertex(7).to_string(),
            "vertex 7 does not exist"
        );
        assert_eq!(
            StoreError::VertexInUse {
                vertex: 3,
                corners: 2
            }
            .to_string(),
            "vertex 3 is still used by 2 triangle corners"
        );
        assert_eq!(
            StoreError::UnknownChannel("mask".into()).to_string(),
            "channel \"mask\" does not exist"
        );
    }
}
