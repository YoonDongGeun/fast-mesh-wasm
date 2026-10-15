use std::fmt;

use fast_mesh_store::StoreError;

/// Errors returned by BVH operations. On any error the BVH is left exactly as it
/// was before the call: every operation reserves the memory it needs before it
/// changes the tree (best-effort steps that run afterwards are skipped silently
/// instead of failing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BvhError {
    /// The mesh is not the one this BVH was built from.
    MeshMismatch,
    /// An option is out of range; the message says which.
    InvalidOptions(&'static str),
    /// A memory allocation failed.
    OutOfMemory,
    /// The mesh store reported an error.
    Store(StoreError),
}

impl fmt::Display for BvhError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MeshMismatch => write!(f, "mesh is not the one this BVH was built from"),
            Self::InvalidOptions(msg) => write!(f, "invalid BVH options: {msg}"),
            Self::OutOfMemory => write!(f, "out of memory"),
            Self::Store(e) => write!(f, "mesh store error: {e}"),
        }
    }
}

impl std::error::Error for BvhError {}

impl From<StoreError> for BvhError {
    fn from(e: StoreError) -> Self {
        match e {
            StoreError::OutOfMemory => Self::OutOfMemory,
            other => Self::Store(other),
        }
    }
}

pub type Result<T> = std::result::Result<T, BvhError>;

#[cfg(test)]
thread_local! {
    /// Test-only: how many more allocations may succeed (`None` = unlimited).
    static ALLOCATIONS_LEFT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Test-only: make every allocation after the next `n` fail with `OutOfMemory`
/// on this thread (`None` turns it off).
#[cfg(test)]
pub(crate) fn fail_allocations_after(n: Option<usize>) {
    ALLOCATIONS_LEFT.with(|c| c.set(n));
}

#[cfg(test)]
fn allocation_allowed() -> Result<()> {
    ALLOCATIONS_LEFT.with(|c| match c.get() {
        Some(0) => Err(BvhError::OutOfMemory),
        Some(n) => {
            c.set(Some(n - 1));
            Ok(())
        }
        None => Ok(()),
    })
}

#[cfg(not(test))]
fn allocation_allowed() -> Result<()> {
    Ok(())
}

pub(crate) fn try_reserve<T>(v: &mut Vec<T>, additional: usize) -> Result<()> {
    allocation_allowed()?;
    v.try_reserve(additional).map_err(|_| BvhError::OutOfMemory)
}

pub(crate) fn try_reserve_exact<T>(v: &mut Vec<T>, additional: usize) -> Result<()> {
    allocation_allowed()?;
    v.try_reserve_exact(additional)
        .map_err(|_| BvhError::OutOfMemory)
}

pub(crate) fn try_reserve_map<K: Eq + std::hash::Hash, V>(
    m: &mut std::collections::HashMap<K, V>,
    additional: usize,
) -> Result<()> {
    allocation_allowed()?;
    m.try_reserve(additional).map_err(|_| BvhError::OutOfMemory)
}

pub(crate) fn try_vec<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    allocation_allowed()?;
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| BvhError::OutOfMemory)?;
    v.resize(len, value);
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_the_problem() {
        assert_eq!(
            BvhError::InvalidOptions("max_leaf_size must be 1 to 16").to_string(),
            "invalid BVH options: max_leaf_size must be 1 to 16"
        );
        assert_eq!(
            BvhError::from(StoreError::OutOfMemory),
            BvhError::OutOfMemory
        );
        assert_eq!(
            BvhError::from(StoreError::InvalidVertex(3)).to_string(),
            "mesh store error: vertex 3 does not exist"
        );
    }
}
