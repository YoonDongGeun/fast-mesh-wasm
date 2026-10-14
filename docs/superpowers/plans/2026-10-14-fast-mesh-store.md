# fast-mesh-store (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `fast-mesh-store` crate: slot-based editable mesh storage with per-vertex channels, transactions that journal their writes, reversible patches, a memory-budgeted undo/redo history, compaction and a lazily built corner-table adjacency.

**Architecture:** Every undoable array is a `Column<T>` that, during a transaction, records the old value of each index the first time it is written. Committing turns the journals into a `Patch` (sparse or full-copy per column, whichever is smaller) plus a `ChangeSet` describing what changed. `History` stacks patches within a byte budget and decides when dead slots may be reused, so slot ids never collide across undo/redo.

**Tech Stack:** Rust 1.99 (edition 2024, pinned by `rust-toolchain.toml`), no runtime dependencies, `proptest` for property tests.

**Spec:** `docs/superpowers/specs/2026-10-14-fast-mesh-engine-design.md` (section 4 "Storage"; section 14 "Testing", Store row).

## Global Constraints

- Crate path `crates/fast-mesh-store`, package name `fast-mesh-store`, workspace-inherited version/edition/license (`MIT OR Apache-2.0`).
- No runtime dependencies. Dev-dependency `proptest = "1.12.0"` only.
- Must compile for `wasm32-unknown-unknown`: no threads, no filesystem, no OS APIs.
- Positions are stored as `f32` relative to a per-mesh `f64` origin. Vertex and triangle ids are `u32`.
- Large allocations use `try_reserve`/`try_vec` and report `StoreError::OutOfMemory` instead of aborting.
- Validation errors are returned before anything is written.
- Removed triangles read `[0, 0, 0]` in `Mesh::triangles()`.
- `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` must pass at the end (Task 8). Until Task 8, unused-item warnings from not-yet-used internals are expected; only `cargo test` must pass per task.
- Commit messages end with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

### Deviations from the spec (recorded in the spec in Task 8)

1. **Snapshot fallback** picks whichever form (sparse or full copy) is smaller in bytes, instead of a fixed 50% threshold. Same intent, never worse.
2. **Adjacency** is cached and rebuilt lazily after a topology change instead of being maintained incrementally. Incremental maintenance moves to the repair/ops phase, where topology algorithms first need it during a transaction.
3. **`TopologyPolicy::Nearest`** takes the value of the creation parent with the largest weight. A spatial nearest-vertex lookup needs the BVH and arrives with the Scene phase.
4. **Scene-wide history** (one undo step across meshes) is deferred to the Scene phase (spec phase 4).
5. **Benchmarks** for the store are deferred to the BVH phase, where build and refit benchmarks are set up.

## Review Focus

- Undoing a commit that appended a vertex past a 64-slot liveness word must leave that vertex dead; otherwise undone vertices silently reappear (Task 4: `undoing_an_append_past_a_bitset_word_kills_the_slot`).
- A validation error in the middle of a transaction must change nothing, and the transaction must stay usable (Task 4: `validation_errors_leave_the_transaction_usable`).
- A slot removed by a still-undoable patch must not be reused by a later edit, or undo would resurrect a vertex on top of another (Task 5: `removed_slots_are_reserved_until_their_patch_is_evicted`).
- Coordinates far from zero (10 km) given as `f64` must keep sub-millimeter precision (Task 3: `f64_input_keeps_precision_through_the_origin`).
- Non-manifold edges, bowtie vertices and inconsistently wound neighbors must be accepted and classified, not rejected or mis-linked (Task 7: `three_triangles_on_one_edge_are_non_manifold`, `bowtie_vertex_is_non_manifold`, `inconsistent_orientation_still_connects`).

## File Structure

| File | Responsibility |
| --- | --- |
| `crates/fast-mesh-store/Cargo.toml` | Crate manifest |
| `crates/fast-mesh-store/src/lib.rs` | Module wiring, public re-exports, `VertexId`/`TriangleId`/`Result` |
| `src/error.rs` | `StoreError` and its messages |
| `src/column.rs` | `Column<T>` with write journal, `ColumnPatch<T>`, `try_vec` |
| `src/channel.rs` | `ChannelDesc`, `ChannelKind`, `TopologyPolicy`, value initialization for new vertices |
| `src/mesh.rs` | `Mesh`: construction, read API, liveness bitsets, channel management, slot release |
| `src/patch.rs` | `Patch` (apply/revert) and `ChangeSet` |
| `src/transaction.rs` | `Transaction`: validated edits, commit, rollback |
| `src/history.rs` | `History`: undo/redo stacks, byte budget, slot reservation |
| `src/compact.rs` | `Mesh::compact` and `Remap` |
| `src/adjacency.rs` | Corner table, edge/vertex manifoldness, fan traversal |
| `tests/properties.rs` | Property tests across all of the above |

---

### Task 1: Crate scaffold and errors

**Files:**
- Create: `crates/fast-mesh-store/Cargo.toml`
- Create: `crates/fast-mesh-store/src/lib.rs`
- Create: `crates/fast-mesh-store/src/error.rs`

**Interfaces:**
- Consumes: workspace keys in the root `Cargo.toml` (`[workspace.package]`); `members = ["crates/*"]` already picks the crate up.
- Produces: `pub enum StoreError` with variants `InvalidVertex(u32)`, `InvalidTriangle(u32)`, `VertexInUse { vertex: u32, corners: u32 }`, `LengthMismatch { expected: usize, got: usize }`, `InvalidArgument(&'static str)`, `UnknownChannel(String)`, `ChannelExists(String)`, `NotSourceChannel(String)`, `NotDerivedChannel(String)`, `PatchMismatch`, `StalePatch`, `TransactionPoisoned`, `OutOfMemory`; `pub type Result<T> = std::result::Result<T, StoreError>`; `pub type VertexId = u32`; `pub type TriangleId = u32`.

- [ ] **Step 1: Create the manifest**

`crates/fast-mesh-store/Cargo.toml`:

```toml
[package]
name = "fast-mesh-store"
description = "Editable triangle mesh storage with transactional undo/redo (part of fast-mesh-wasm)"
keywords = ["mesh", "geometry", "undo", "3d"]
categories = ["graphics", "data-structures"]
readme = "../../README.md"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
authors.workspace = true

[dependencies]

[dev-dependencies]
proptest = "1.12.0"
```

- [ ] **Step 2: Write the failing test**

Create `crates/fast-mesh-store/src/error.rs` containing only this test module for now:

```rust
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
```

And `crates/fast-mesh-store/src/lib.rs`:

```rust
//! Editable triangle mesh storage with transactional undo/redo.
//!
//! - [`Mesh`] stores vertices, triangles and per-vertex channels in slots whose ids
//!   stay stable across edits, undo and redo.
//! - [`Mesh::begin_edit`] opens a [`Transaction`]. Every write goes through it, and
//!   [`Transaction::commit`] returns a [`Patch`] holding only what changed.
//! - [`History`] keeps patches within a memory budget and provides undo/redo.
//! - Every commit, undo and redo reports a [`ChangeSet`] so that dependent data
//!   (BVHs, fields, render buffers) can update just the affected elements.

mod error;

pub use error::StoreError;

/// Id of a vertex slot.
pub type VertexId = u32;
/// Id of a triangle slot.
pub type TriangleId = u32;

pub type Result<T> = std::result::Result<T, StoreError>;
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test -p fast-mesh-store`
Expected: compile errors, `unresolved import error::StoreError` and `cannot find type StoreError`.

- [ ] **Step 4: Implement `StoreError`**

Insert above the test module in `src/error.rs`:

```rust
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test -p fast-mesh-store`
Expected: `test error::tests::messages_name_the_problem ... ok`.

- [ ] **Step 6: Commit**

```bash
git add crates/fast-mesh-store Cargo.lock
git commit -m "feat(store): add fast-mesh-store crate with StoreError"
```

---

### Task 2: Journaled columns

**Files:**
- Create: `crates/fast-mesh-store/src/column.rs`
- Modify: `crates/fast-mesh-store/src/lib.rs` (add `mod column;`)

**Interfaces:**
- Consumes: `StoreError`, `Result` (Task 1).
- Produces (crate-private): `struct Column<T>` with `from_vec(Vec<T>)`, `len()`, `as_slice()`, `as_mut_slice()` (bypasses the journal), `get(i)`, `begin()`, `set(i, value)`, `push(value) -> Result<usize>`, `grow_to(len, value) -> Result<()>`, `finish() -> ColumnPatch<T>`, `rollback()`; `enum ColumnPatch<T>` with `apply(&mut Column<T>) -> Result<()>`, `revert(&mut Column<T>)`, `is_empty()`, `byte_size()`; `fn try_vec<T: Clone>(len, value) -> Result<Vec<T>>`. All for `T: Copy + Default`.

- [ ] **Step 1: Write the failing tests**

Create `src/column.rs` with this test module, and add `mod column;` to `src/lib.rs` (full file below):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn column(values: &[u32]) -> Column<u32> {
        Column::from_vec(values.to_vec())
    }

    #[test]
    fn journals_only_the_first_write_per_index() {
        let mut c = column(&[1, 2, 3]);
        c.begin();
        c.set(1, 20);
        c.set(1, 200);
        match c.finish() {
            ColumnPatch::Sparse { ids, old, new, .. } => {
                assert_eq!(ids, vec![1]);
                assert_eq!(old, vec![2]);
                assert_eq!(new, vec![200]);
            }
            ColumnPatch::Snapshot { .. } => panic!("expected sparse patch"),
        }
    }

    #[test]
    fn revert_and_apply_round_trip() {
        let mut c = column(&[1, 2, 3]);
        c.begin();
        c.set(0, 10);
        c.push(4).unwrap();
        let patch = c.finish();
        assert_eq!(c.as_slice(), &[10, 2, 3, 4]);
        patch.revert(&mut c);
        assert_eq!(&c.as_slice()[..3], &[1, 2, 3], "old values restored");
        assert_eq!(c.len(), 4, "appended slot stays allocated");
        assert_eq!(c.get(3), 0, "appended slot is reset to the default");
        patch.apply(&mut c).unwrap();
        assert_eq!(c.as_slice(), &[10, 2, 3, 4]);
    }

    #[test]
    fn rollback_restores_values_and_length() {
        let mut c = column(&[1, 2, 3]);
        c.begin();
        c.set(2, 30);
        c.push(4).unwrap();
        c.rollback();
        assert_eq!(c.as_slice(), &[1, 2, 3]);
        // The touched bits were cleared: a new transaction journals index 2 again.
        c.begin();
        c.set(2, 31);
        match c.finish() {
            ColumnPatch::Sparse { ids, old, .. } => {
                assert_eq!(ids, vec![2]);
                assert_eq!(old, vec![3]);
            }
            ColumnPatch::Snapshot { .. } => panic!("expected sparse patch"),
        }
    }

    #[test]
    fn rewriting_most_values_switches_to_a_snapshot() {
        let mut c = column(&[0; 100]);
        c.begin();
        for i in 0..100 {
            c.set(i, 1);
        }
        let patch = c.finish();
        assert!(matches!(patch, ColumnPatch::Snapshot { .. }));
        patch.revert(&mut c);
        assert_eq!(c.as_slice(), &[0; 100]);
        patch.apply(&mut c).unwrap();
        assert_eq!(c.as_slice(), &[1; 100]);
    }

    #[test]
    fn untouched_column_produces_an_empty_patch() {
        let mut c = column(&[1, 2]);
        c.begin();
        let patch = c.finish();
        assert!(patch.is_empty());
        assert_eq!(patch.byte_size(), 0);
    }
}
```

```rust
//! Editable triangle mesh storage with transactional undo/redo.
//!
//! - [`Mesh`] stores vertices, triangles and per-vertex channels in slots whose ids
//!   stay stable across edits, undo and redo.
//! - [`Mesh::begin_edit`] opens a [`Transaction`]. Every write goes through it, and
//!   [`Transaction::commit`] returns a [`Patch`] holding only what changed.
//! - [`History`] keeps patches within a memory budget and provides undo/redo.
//! - Every commit, undo and redo reports a [`ChangeSet`] so that dependent data
//!   (BVHs, fields, render buffers) can update just the affected elements.

mod column;
mod error;

pub use error::StoreError;

/// Id of a vertex slot.
pub type VertexId = u32;
/// Id of a triangle slot.
pub type TriangleId = u32;

pub type Result<T> = std::result::Result<T, StoreError>;
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p fast-mesh-store column`
Expected: compile errors, `cannot find type Column` / `ColumnPatch`.

- [ ] **Step 3: Implement the column**

Insert above the test module in `src/column.rs`:

```rust
//! A growable column of values with an optional write journal.
//!
//! While a journal is active, the first write to each pre-existing index records the
//! old value. `finish()` turns the journal into a [`ColumnPatch`] that can re-apply or
//! revert the changes; `rollback()` restores the column to its state at `begin()`.

use crate::{Result, StoreError};

pub(crate) struct Column<T> {
    data: Vec<T>,
    /// Bitset of indices already journaled in the current transaction. Kept between
    /// transactions and cleared bit by bit, so beginning a transaction costs nothing.
    touched: Vec<u64>,
    journal: Option<Journal<T>>,
}

struct Journal<T> {
    start_len: usize,
    ids: Vec<u32>,
    old: Vec<T>,
}

/// The recorded effect of one transaction on one column.
#[derive(Debug, Clone)]
pub(crate) enum ColumnPatch<T> {
    /// Per-index old and new values, plus values appended past the old length.
    Sparse {
        ids: Vec<u32>,
        old: Vec<T>,
        new: Vec<T>,
        old_len: usize,
        appended: Vec<T>,
    },
    /// Full copies, used when they are smaller than the sparse form.
    Snapshot { old: Vec<T>, new: Vec<T> },
}

pub(crate) fn try_vec<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| StoreError::OutOfMemory)?;
    v.resize(len, value);
    Ok(v)
}

impl<T: Copy + Default> Column<T> {
    pub(crate) fn from_vec(data: Vec<T>) -> Self {
        Self {
            data,
            touched: Vec::new(),
            journal: None,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable access that bypasses the journal. Only for data that is not undoable.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        debug_assert!(
            self.journal.is_none(),
            "unjournaled write during a transaction"
        );
        &mut self.data
    }

    pub(crate) fn get(&self, i: usize) -> T {
        self.data[i]
    }

    pub(crate) fn begin(&mut self) {
        debug_assert!(self.journal.is_none(), "begin() called twice");
        let words = self.data.len().div_ceil(64);
        if self.touched.len() < words {
            self.touched.resize(words, 0);
        }
        self.journal = Some(Journal {
            start_len: self.data.len(),
            ids: Vec::new(),
            old: Vec::new(),
        });
    }

    pub(crate) fn set(&mut self, i: usize, value: T) {
        if let Some(j) = &mut self.journal
            && i < j.start_len
        {
            let (word, bit) = (i / 64, 1u64 << (i % 64));
            if self.touched[word] & bit == 0 {
                self.touched[word] |= bit;
                j.ids.push(i as u32);
                j.old.push(self.data[i]);
            }
        }
        self.data[i] = value;
    }

    /// Appends a value and returns its index. Appended values are not journaled:
    /// rollback truncates them and patches store them as `appended`.
    pub(crate) fn push(&mut self, value: T) -> Result<usize> {
        self.data
            .try_reserve(1)
            .map_err(|_| StoreError::OutOfMemory)?;
        self.data.push(value);
        Ok(self.data.len() - 1)
    }

    /// Grows the column to `len` with `value`. Never shrinks.
    pub(crate) fn grow_to(&mut self, len: usize, value: T) -> Result<()> {
        if self.data.len() < len {
            let extra = len - self.data.len();
            self.data
                .try_reserve(extra)
                .map_err(|_| StoreError::OutOfMemory)?;
            self.data.resize(len, value);
        }
        Ok(())
    }

    pub(crate) fn finish(&mut self) -> ColumnPatch<T> {
        let j = self.journal.take().expect("finish() without begin()");
        self.clear_touched(&j.ids);
        let appended = self.data[j.start_len..].to_vec();
        let size = std::mem::size_of::<T>();
        let sparse_bytes = j.ids.len() * (4 + 2 * size) + appended.len() * size;
        let snapshot_bytes = (j.start_len + self.data.len()) * size;
        if !j.ids.is_empty() && snapshot_bytes < sparse_bytes {
            let mut old = self.data[..j.start_len].to_vec();
            for (k, &i) in j.ids.iter().enumerate() {
                old[i as usize] = j.old[k];
            }
            ColumnPatch::Snapshot {
                old,
                new: self.data.clone(),
            }
        } else {
            let new = j.ids.iter().map(|&i| self.data[i as usize]).collect();
            ColumnPatch::Sparse {
                ids: j.ids,
                old: j.old,
                new,
                old_len: j.start_len,
                appended,
            }
        }
    }

    pub(crate) fn rollback(&mut self) {
        let j = self.journal.take().expect("rollback() without begin()");
        self.clear_touched(&j.ids);
        for (k, &i) in j.ids.iter().enumerate() {
            self.data[i as usize] = j.old[k];
        }
        self.data.truncate(j.start_len);
    }

    fn clear_touched(&mut self, ids: &[u32]) {
        for &i in ids {
            self.touched[i as usize / 64] &= !(1u64 << (i % 64));
        }
    }
}

impl<T: Copy + Default> ColumnPatch<T> {
    pub(crate) fn apply(&self, column: &mut Column<T>) -> Result<()> {
        match self {
            Self::Sparse {
                ids,
                new,
                old_len,
                appended,
                ..
            } => {
                let end = old_len + appended.len();
                column.grow_to(end, T::default())?;
                for (k, &i) in ids.iter().enumerate() {
                    column.data[i as usize] = new[k];
                }
                column.data[*old_len..end].copy_from_slice(appended);
            }
            Self::Snapshot { new, .. } => {
                column.grow_to(new.len(), T::default())?;
                column.data[..new.len()].copy_from_slice(new);
            }
        }
        Ok(())
    }

    /// Restores the old values. The column keeps its length: slots appended by the
    /// transaction stay allocated so their ids are never handed out twice, and are
    /// reset to `T::default()` (dead liveness bits, `[0, 0, 0]` triangles).
    pub(crate) fn revert(&self, column: &mut Column<T>) {
        let (old_len, new_len) = match self {
            Self::Sparse {
                ids,
                old,
                old_len,
                appended,
                ..
            } => {
                for (k, &i) in ids.iter().enumerate() {
                    column.data[i as usize] = old[k];
                }
                (*old_len, old_len + appended.len())
            }
            Self::Snapshot { old, new } => {
                column.data[..old.len()].copy_from_slice(old);
                (old.len(), new.len())
            }
        };
        column.data[old_len..new_len].fill(T::default());
    }

    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Sparse { ids, appended, .. } => ids.is_empty() && appended.is_empty(),
            Self::Snapshot { .. } => false,
        }
    }

    pub(crate) fn byte_size(&self) -> usize {
        let size = std::mem::size_of::<T>();
        match self {
            Self::Sparse {
                ids,
                old,
                new,
                appended,
                ..
            } => ids.len() * 4 + (old.len() + new.len() + appended.len()) * size,
            Self::Snapshot { old, new } => (old.len() + new.len()) * size,
        }
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p fast-mesh-store column`
Expected: 5 tests pass (`journals_only_the_first_write_per_index`, `revert_and_apply_round_trip`, `rollback_restores_values_and_length`, `rewriting_most_values_switches_to_a_snapshot`, `untouched_column_produces_an_empty_patch`).

- [ ] **Step 5: Commit**

```bash
git add crates/fast-mesh-store
git commit -m "feat(store): add journaled columns with sparse and snapshot patches"
```

---

### Task 3: Mesh storage and channels

**Files:**
- Create: `crates/fast-mesh-store/src/channel.rs`
- Create: `crates/fast-mesh-store/src/mesh.rs`
- Modify: `crates/fast-mesh-store/src/lib.rs`

**Interfaces:**
- Consumes: `Column`, `try_vec` (Task 2); `StoreError`, `Result`, `VertexId`, `TriangleId` (Task 1).
- Produces:
  - `pub enum ChannelKind { Source, Derived }`, `pub enum TopologyPolicy { Interpolate, Default, Nearest }`, `pub struct ChannelDesc { name: String, width: usize, kind, policy, default: f32 }` with `source(name, width)`, `derived(name, width)`, `with_policy`, `with_default`.
  - crate-private `struct Channel { desc, data: Column<f32> }` with `initial_value(&self, parents: &[(u32, f32)], k: usize) -> f32`.
  - `pub struct Mesh` with `new()`, `from_buffers(&[f32], &[u32])`, `from_buffers_f64(&[f64], &[u32])`, `uid()`, `revision()`, `origin()`, `vertex_slots()`, `triangle_slots()`, `vertex_count()`, `triangle_count()`, `is_vertex_alive(v)`, `is_triangle_alive(t)`, `position(v) -> Option<[f32; 3]>`, `world_position(v) -> Option<[f64; 3]>`, `positions()`, `triangle(t) -> Option<[u32; 3]>`, `triangles()`, `corner_count(v) -> u32`, `free_vertex_slots()`, `free_triangle_slots()`, `add_channel`, `remove_channel`, `channel_desc`, `channel(name) -> Option<&[f32]>`, `derived_channel_mut(name) -> Result<&mut [f32]>`.
  - crate-private: fields listed in the struct, `bit`, `set_bit`, `full_bits`, `Mesh::from_parts`, `channel_index`, `grow_channels`, `release_vertices`, `release_triangles`, `on_topology_changed` (empty until Task 7).

- [ ] **Step 1: Write the failing tests**

Create `src/channel.rs` with:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn channel(desc: ChannelDesc, values: &[f32]) -> Channel {
        Channel {
            desc,
            data: Column::from_vec(values.to_vec()),
        }
    }

    #[test]
    fn interpolate_uses_normalized_weights() {
        let c = channel(ChannelDesc::source("m", 1), &[10.0, 20.0]);
        assert_eq!(c.initial_value(&[(0, 1.0), (1, 3.0)], 0), 17.5);
    }

    #[test]
    fn nearest_takes_the_heaviest_parent() {
        let c = channel(
            ChannelDesc::source("m", 1).with_policy(TopologyPolicy::Nearest),
            &[10.0, 20.0],
        );
        assert_eq!(c.initial_value(&[(0, 0.9), (1, 0.1)], 0), 10.0);
    }

    #[test]
    fn no_parents_or_zero_weights_use_the_default() {
        let c = channel(ChannelDesc::source("m", 1).with_default(-1.0), &[10.0]);
        assert_eq!(c.initial_value(&[], 0), -1.0);
        assert_eq!(c.initial_value(&[(0, 0.0)], 0), -1.0);
    }

    #[test]
    fn derived_channels_always_start_at_the_default() {
        let c = channel(ChannelDesc::derived("d", 1), &[10.0]);
        assert!(c.initial_value(&[(0, 1.0)], 0).is_nan());
    }

    #[test]
    fn components_are_read_per_width() {
        let c = channel(
            ChannelDesc::source("rgb", 3),
            &[1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
        );
        let parents = [(0, 1.0), (1, 1.0)];
        assert_eq!(
            [0, 1, 2].map(|k| c.initial_value(&parents, k)),
            [3.0, 4.0, 5.0]
        );
    }
}
```

Create `src/mesh.rs` with:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> Mesh {
        Mesh::from_buffers(
            &[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap()
    }

    #[test]
    fn builds_from_flat_buffers() {
        let m = quad();
        assert_eq!((m.vertex_count(), m.triangle_count()), (4, 2));
        assert_eq!(m.position(2), Some([1., 1., 0.]));
        assert_eq!(m.triangle(1), Some([0, 2, 3]));
        assert_eq!(m.corner_count(0), 2);
        assert_eq!(m.corner_count(1), 1);
    }

    #[test]
    fn rejects_bad_buffers() {
        assert!(matches!(
            Mesh::from_buffers(&[0., 0.], &[]),
            Err(StoreError::InvalidArgument(_))
        ));
        assert_eq!(
            Mesh::from_buffers(&[0., 0., 0.], &[0, 0, 5]).err(),
            Some(StoreError::InvalidVertex(5))
        );
    }

    #[test]
    fn f64_input_keeps_precision_through_the_origin() {
        let far = 1.0e7;
        let m = Mesh::from_buffers_f64(&[far, far, 0.0, far + 0.001, far, 0.0], &[]).unwrap();
        let o = m.origin();
        assert!((o[0] - (far + 0.0005)).abs() < 1e-6 && o[1] == far && o[2] == 0.0);
        let p = m.world_position(1).unwrap();
        assert!((p[0] - (far + 0.001)).abs() < 1e-6, "got {}", p[0]);
    }

    #[test]
    fn out_of_range_ids_read_as_dead() {
        let m = quad();
        assert!(!m.is_vertex_alive(99));
        assert_eq!(m.position(99), None);
        assert_eq!(m.triangle(99), None);
    }

    #[test]
    fn channels_are_slot_aligned_and_unique() {
        let mut m = quad();
        m.add_channel(ChannelDesc::source("mask", 1).with_default(0.5))
            .unwrap();
        assert_eq!(m.channel("mask"), Some(&[0.5; 4][..]));
        assert_eq!(
            m.add_channel(ChannelDesc::source("mask", 1)),
            Err(StoreError::ChannelExists("mask".into()))
        );
        assert!(matches!(
            m.add_channel(ChannelDesc::source("wide", 5)),
            Err(StoreError::InvalidArgument(_))
        ));
        assert_eq!(
            m.derived_channel_mut("mask").err(),
            Some(StoreError::NotDerivedChannel("mask".into()))
        );
        m.add_channel(ChannelDesc::derived("dist", 1)).unwrap();
        m.derived_channel_mut("dist").unwrap()[3] = 7.0;
        assert_eq!(m.channel("dist").unwrap()[3], 7.0);
        assert_eq!(m.remove_channel("dist").unwrap().name, "dist");
        assert_eq!(m.channel("dist"), None);
    }

    #[test]
    fn every_mesh_gets_a_distinct_uid() {
        assert_ne!(Mesh::new().uid(), Mesh::new().uid());
    }
}
```

Replace `src/lib.rs` with:

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p fast-mesh-store`
Expected: compile errors, unresolved imports of `ChannelDesc`, `Mesh` and friends.

- [ ] **Step 3: Implement channels**

Insert above the test module in `src/channel.rs`:

```rust
use crate::column::Column;

/// Whether a channel's values are undoable input or recomputable output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelKind {
    /// Values provided by the user (e.g. a painted mask). Journaled and undoable.
    Source,
    /// Values computed from other data (e.g. distances, colors). Not journaled;
    /// recomputed after edits, undo and redo.
    Derived,
}

/// How a source channel gets values for vertices created by an edit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyPolicy {
    /// Weighted average of the creation parents' values.
    Interpolate,
    /// The channel's default value.
    Default,
    /// The value of the creation parent with the largest weight.
    Nearest,
}

/// Describes a per-vertex attribute channel.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelDesc {
    pub name: String,
    /// Values per vertex, 1 to 4.
    pub width: usize,
    pub kind: ChannelKind,
    pub policy: TopologyPolicy,
    /// Value for new vertices without usable parents, and for derived channels.
    pub default: f32,
}

impl ChannelDesc {
    /// A source channel that interpolates values for new vertices and defaults to 0.
    pub fn source(name: impl Into<String>, width: usize) -> Self {
        Self {
            name: name.into(),
            width,
            kind: ChannelKind::Source,
            policy: TopologyPolicy::Interpolate,
            default: 0.0,
        }
    }

    /// A derived channel whose new values are NaN until computed.
    pub fn derived(name: impl Into<String>, width: usize) -> Self {
        Self {
            name: name.into(),
            width,
            kind: ChannelKind::Derived,
            policy: TopologyPolicy::Default,
            default: f32::NAN,
        }
    }

    pub fn with_policy(mut self, policy: TopologyPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn with_default(mut self, default: f32) -> Self {
        self.default = default;
        self
    }
}

pub(crate) struct Channel {
    pub(crate) desc: ChannelDesc,
    /// `width` values per vertex slot.
    pub(crate) data: Column<f32>,
}

impl Channel {
    /// Value for a new vertex created from `parents` (vertex id, weight) for component `k`.
    pub(crate) fn initial_value(&self, parents: &[(u32, f32)], k: usize) -> f32 {
        let d = &self.desc;
        if d.kind == ChannelKind::Derived || parents.is_empty() {
            return d.default;
        }
        let at = |v: u32| self.data.get(v as usize * d.width + k);
        match d.policy {
            TopologyPolicy::Default => d.default,
            TopologyPolicy::Nearest => {
                let (v, _) = parents
                    .iter()
                    .copied()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .expect("parents is not empty");
                at(v)
            }
            TopologyPolicy::Interpolate => {
                let total: f32 = parents.iter().map(|p| p.1).sum();
                if total == 0.0 {
                    d.default
                } else {
                    parents.iter().map(|&(v, w)| at(v) * w).sum::<f32>() / total
                }
            }
        }
    }
}
```

- [ ] **Step 4: Implement the mesh**

Insert above the test module in `src/mesh.rs`:

```rust
use std::sync::atomic::{AtomicU64, Ordering};

use crate::channel::{Channel, ChannelDesc, ChannelKind};
use crate::column::{Column, try_vec};
use crate::{Result, StoreError, TriangleId, VertexId};

static NEXT_MESH_UID: AtomicU64 = AtomicU64::new(1);

/// An editable triangle mesh stored in slots.
///
/// Vertices and triangles are addressed by slot ids that stay stable across edits,
/// undo and redo. Removed slots are dead: `positions()` and `triangles()` still contain
/// them (removed triangles read `[0, 0, 0]`, which renderers draw as nothing), and
/// `is_vertex_alive` / `is_triangle_alive` tell them apart. Only [`Mesh::compact`]
/// renumbers slots.
pub struct Mesh {
    pub(crate) uid: u64,
    pub(crate) epoch: u32,
    pub(crate) revision: u64,
    pub(crate) next_revision: u64,
    pub(crate) origin: [f64; 3],
    pub(crate) positions: Column<[f32; 3]>,
    pub(crate) triangles: Column<[u32; 3]>,
    pub(crate) vertex_live: Column<u64>,
    pub(crate) triangle_live: Column<u64>,
    /// Number of triangle corners referencing each vertex.
    pub(crate) vertex_refs: Column<u32>,
    pub(crate) channels: Vec<Channel>,
    pub(crate) vertex_free: Vec<VertexId>,
    pub(crate) triangle_free: Vec<TriangleId>,
    pub(crate) live_vertices: usize,
    pub(crate) live_triangles: usize,
}

pub(crate) fn bit(bits: &Column<u64>, i: usize) -> bool {
    let word = i / 64;
    word < bits.len() && (bits.get(word) >> (i % 64)) & 1 == 1
}

pub(crate) fn set_bit(bits: &mut Column<u64>, i: usize, on: bool) -> Result<()> {
    let word = i / 64;
    bits.grow_to(word + 1, 0)?;
    let old = bits.get(word);
    let mask = 1u64 << (i % 64);
    let new = if on { old | mask } else { old & !mask };
    if new != old {
        bits.set(word, new);
    }
    Ok(())
}

pub(crate) fn full_bits(len: usize) -> Result<Vec<u64>> {
    let mut words = try_vec(len / 64, u64::MAX)?;
    if !len.is_multiple_of(64) {
        words.push((1u64 << (len % 64)) - 1);
    }
    Ok(words)
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

impl Mesh {
    /// An empty mesh with its origin at zero.
    pub fn new() -> Self {
        Self::from_parts([0.0; 3], Vec::new(), Vec::new(), Vec::new())
    }

    /// Builds a mesh from flat `[x, y, z, ...]` positions and `[a, b, c, ...]` indices.
    pub fn from_buffers(positions: &[f32], indices: &[u32]) -> Result<Self> {
        if !positions.len().is_multiple_of(3) {
            return Err(StoreError::InvalidArgument(
                "positions length must be a multiple of 3",
            ));
        }
        let mut points = try_vec(positions.len() / 3, [0.0f32; 3])?;
        for (p, c) in points.iter_mut().zip(positions.as_chunks::<3>().0) {
            *p = *c;
        }
        Self::build([0.0; 3], points, indices)
    }

    /// Builds a mesh from double-precision positions. The origin is set to the center
    /// of the bounding box and positions are stored as `f32` offsets from it, which
    /// keeps precision for coordinates far from zero.
    pub fn from_buffers_f64(positions: &[f64], indices: &[u32]) -> Result<Self> {
        if !positions.len().is_multiple_of(3) {
            return Err(StoreError::InvalidArgument(
                "positions length must be a multiple of 3",
            ));
        }
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for c in positions.as_chunks::<3>().0 {
            for k in 0..3 {
                min[k] = min[k].min(c[k]);
                max[k] = max[k].max(c[k]);
            }
        }
        let origin = if positions.is_empty() {
            [0.0; 3]
        } else {
            [0, 1, 2].map(|k| (min[k] + max[k]) * 0.5)
        };
        let mut points = try_vec(positions.len() / 3, [0.0f32; 3])?;
        for (p, c) in points.iter_mut().zip(positions.as_chunks::<3>().0) {
            *p = [0, 1, 2].map(|k| (c[k] - origin[k]) as f32);
        }
        Self::build(origin, points, indices)
    }

    fn build(origin: [f64; 3], points: Vec<[f32; 3]>, indices: &[u32]) -> Result<Self> {
        if !indices.len().is_multiple_of(3) {
            return Err(StoreError::InvalidArgument(
                "indices length must be a multiple of 3",
            ));
        }
        let vertex_count = points.len();
        let mut refs = try_vec(vertex_count, 0u32)?;
        let mut tris = try_vec(indices.len() / 3, [0u32; 3])?;
        for (t, c) in tris.iter_mut().zip(indices.as_chunks::<3>().0) {
            for &v in c {
                if v as usize >= vertex_count {
                    return Err(StoreError::InvalidVertex(v));
                }
                refs[v as usize] += 1;
            }
            *t = *c;
        }
        Ok(Self::from_parts(origin, points, tris, refs))
    }

    pub(crate) fn from_parts(
        origin: [f64; 3],
        points: Vec<[f32; 3]>,
        tris: Vec<[u32; 3]>,
        refs: Vec<u32>,
    ) -> Self {
        let (v, t) = (points.len(), tris.len());
        Self {
            uid: NEXT_MESH_UID.fetch_add(1, Ordering::Relaxed),
            epoch: 0,
            revision: 0,
            next_revision: 0,
            origin,
            positions: Column::from_vec(points),
            triangles: Column::from_vec(tris),
            vertex_live: Column::from_vec(full_bits(v).expect("bitset of existing vertices")),
            triangle_live: Column::from_vec(full_bits(t).expect("bitset of existing triangles")),
            vertex_refs: Column::from_vec(refs),
            channels: Vec::new(),
            vertex_free: Vec::new(),
            triangle_free: Vec::new(),
            live_vertices: v,
            live_triangles: t,
        }
    }

    /// Process-unique id of this mesh. Patches only apply to the mesh that created them.
    pub fn uid(&self) -> u64 {
        self.uid
    }

    /// Changes on every commit, undo and redo. Equal revisions mean equal content.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// World-space origin that positions are relative to.
    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }

    /// Number of vertex slots, alive or dead. Render buffers use this length.
    pub fn vertex_slots(&self) -> usize {
        self.positions.len()
    }

    /// Number of triangle slots, alive or dead.
    pub fn triangle_slots(&self) -> usize {
        self.triangles.len()
    }

    /// Number of alive vertices.
    pub fn vertex_count(&self) -> usize {
        self.live_vertices
    }

    /// Number of alive triangles.
    pub fn triangle_count(&self) -> usize {
        self.live_triangles
    }

    pub fn is_vertex_alive(&self, v: VertexId) -> bool {
        bit(&self.vertex_live, v as usize)
    }

    pub fn is_triangle_alive(&self, t: TriangleId) -> bool {
        bit(&self.triangle_live, t as usize)
    }

    /// Position relative to [`Mesh::origin`], or `None` for a dead or missing vertex.
    pub fn position(&self, v: VertexId) -> Option<[f32; 3]> {
        self.is_vertex_alive(v)
            .then(|| self.positions.get(v as usize))
    }

    /// Position in world space (origin added in `f64`).
    pub fn world_position(&self, v: VertexId) -> Option<[f64; 3]> {
        self.position(v)
            .map(|p| [0, 1, 2].map(|k| self.origin[k] + p[k] as f64))
    }

    /// Slot-aligned positions, including dead slots.
    pub fn positions(&self) -> &[[f32; 3]] {
        self.positions.as_slice()
    }

    /// Vertex ids of a triangle, or `None` for a dead or missing triangle.
    pub fn triangle(&self, t: TriangleId) -> Option<[VertexId; 3]> {
        self.is_triangle_alive(t)
            .then(|| self.triangles.get(t as usize))
    }

    /// Slot-aligned triangles; dead slots read `[0, 0, 0]`.
    pub fn triangles(&self) -> &[[VertexId; 3]] {
        self.triangles.as_slice()
    }

    /// How many triangle corners use vertex `v` (a vertex repeated in one triangle counts twice).
    pub fn corner_count(&self, v: VertexId) -> u32 {
        if (v as usize) < self.vertex_refs.len() {
            self.vertex_refs.get(v as usize)
        } else {
            0
        }
    }

    /// Dead vertex slots ready for reuse.
    pub fn free_vertex_slots(&self) -> &[VertexId] {
        &self.vertex_free
    }

    /// Dead triangle slots ready for reuse.
    pub fn free_triangle_slots(&self) -> &[TriangleId] {
        &self.triangle_free
    }

    /// Adds a channel filled with its default value. Adding a channel is not undoable.
    pub fn add_channel(&mut self, desc: ChannelDesc) -> Result<()> {
        if !(1..=4).contains(&desc.width) {
            return Err(StoreError::InvalidArgument("channel width must be 1 to 4"));
        }
        if self.channel_index(&desc.name).is_some() {
            return Err(StoreError::ChannelExists(desc.name));
        }
        let data = try_vec(self.vertex_slots() * desc.width, desc.default)?;
        self.channels.push(Channel {
            desc,
            data: Column::from_vec(data),
        });
        Ok(())
    }

    /// Removes a channel. Not undoable; patches that recorded it skip it from now on.
    pub fn remove_channel(&mut self, name: &str) -> Result<ChannelDesc> {
        let i = self
            .channel_index(name)
            .ok_or_else(|| StoreError::UnknownChannel(name.into()))?;
        Ok(self.channels.remove(i).desc)
    }

    pub fn channel_desc(&self, name: &str) -> Option<&ChannelDesc> {
        self.channel_index(name).map(|i| &self.channels[i].desc)
    }

    /// Slot-aligned values, `width` per vertex.
    pub fn channel(&self, name: &str) -> Option<&[f32]> {
        self.channel_index(name)
            .map(|i| self.channels[i].data.as_slice())
    }

    /// Writable values of a derived channel. Writes are not recorded for undo.
    pub fn derived_channel_mut(&mut self, name: &str) -> Result<&mut [f32]> {
        let i = self
            .channel_index(name)
            .ok_or_else(|| StoreError::UnknownChannel(name.into()))?;
        if self.channels[i].desc.kind != ChannelKind::Derived {
            return Err(StoreError::NotDerivedChannel(name.into()));
        }
        Ok(self.channels[i].data.as_mut_slice())
    }

    pub(crate) fn channel_index(&self, name: &str) -> Option<usize> {
        self.channels.iter().position(|c| c.desc.name == name)
    }

    /// Makes every channel cover every vertex slot.
    pub(crate) fn grow_channels(&mut self) -> Result<()> {
        let slots = self.vertex_slots();
        for c in &mut self.channels {
            c.data.grow_to(slots * c.desc.width, c.desc.default)?;
        }
        Ok(())
    }

    pub(crate) fn release_vertices(&mut self, ids: &[VertexId]) {
        for &v in ids {
            debug_assert!(!self.is_vertex_alive(v), "releasing alive vertex {v}");
            self.vertex_free.push(v);
        }
    }

    pub(crate) fn release_triangles(&mut self, ids: &[TriangleId]) {
        for &t in ids {
            debug_assert!(!self.is_triangle_alive(t), "releasing alive triangle {t}");
            self.triangle_free.push(t);
        }
    }

    /// Called whenever triangles were added, removed or rewritten.
    pub(crate) fn on_topology_changed(&mut self) {
        // The adjacency cache added in Task 7 is invalidated here.
    }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p fast-mesh-store`
Expected: all channel and mesh tests pass, including `f64_input_keeps_precision_through_the_origin`.

- [ ] **Step 6: Commit**

```bash
git add crates/fast-mesh-store
git commit -m "feat(store): add slot-based Mesh with f64 origin and attribute channels"
```

---

### Task 4: Transactions and patches

**Files:**
- Create: `crates/fast-mesh-store/src/patch.rs`
- Create: `crates/fast-mesh-store/src/transaction.rs`
- Modify: `crates/fast-mesh-store/src/lib.rs`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces:
  - `pub struct ChangeSet { vertices_added, vertices_removed, vertices_moved, triangles_added, triangles_removed, triangles_changed: Vec<u32>, channels: Vec<(String, Vec<u32>)> }` with `topology_changed()` and `is_empty()`. Lists are sorted and deduplicated.
  - `pub struct Patch` with `changes() -> &ChangeSet`, `is_empty()`, `base_revision()`, `revision()`, `byte_size()`, `apply(&mut Mesh) -> Result<ChangeSet>`, `revert(&mut Mesh) -> Result<ChangeSet>`; crate-visible fields `mesh_uid`, `epoch`, `base_revision`, `revision`, `changes` (used by `History` in Task 5).
  - `Mesh::begin_edit(&mut self) -> Transaction<'_>`.
  - `pub struct Transaction<'m>` with `mesh()`, `add_vertices(&[[f32; 3]]) -> Result<Vec<u32>>`, `add_vertex_with_parents([f32; 3], &[(u32, f32)]) -> Result<u32>`, `set_positions(&[u32], &[[f32; 3]])`, `remove_vertices(&[u32])`, `add_triangles(&[[u32; 3]]) -> Result<Vec<u32>>`, `remove_triangles(&[u32])`, `set_triangles(&[u32], &[[u32; 3]])`, `write_channel(&str, &[u32], &[f32])`, `commit(self) -> Result<Patch>`, `rollback(self)`; dropping without commit rolls back.

`patch.rs` has no inline tests; its behavior is tested through transactions.

- [ ] **Step 1: Write the failing tests**

Create `src/transaction.rs` with:

```rust
#[cfg(test)]
mod tests {
    use crate::{ChannelDesc, Mesh, StoreError, TopologyPolicy};

    fn quad() -> Mesh {
        Mesh::from_buffers(
            &[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap()
    }

    #[test]
    fn moving_vertices_commits_a_reversible_patch() {
        let mut m = quad();
        let mut tx = m.begin_edit();
        tx.set_positions(&[2], &[[5., 5., 5.]]).unwrap();
        let patch = tx.commit().unwrap();
        assert_eq!(m.position(2), Some([5., 5., 5.]));
        assert_eq!(patch.changes().vertices_moved, vec![2]);
        assert!(!patch.changes().topology_changed());

        let undo = patch.revert(&mut m).unwrap();
        assert_eq!(m.position(2), Some([1., 1., 0.]));
        assert_eq!(undo.vertices_moved, vec![2]);
        patch.apply(&mut m).unwrap();
        assert_eq!(m.position(2), Some([5., 5., 5.]));
    }

    #[test]
    fn adding_and_removing_topology_round_trips() {
        let mut m = quad();
        let mut tx = m.begin_edit();
        let v = tx.add_vertices(&[[2., 0., 0.]]).unwrap()[0];
        let t = tx.add_triangles(&[[1, v, 2]]).unwrap()[0];
        tx.remove_triangles(&[0]).unwrap();
        let patch = tx.commit().unwrap();
        assert_eq!((v, t), (4, 2));
        assert_eq!((m.vertex_count(), m.triangle_count()), (5, 2));
        assert_eq!(
            m.triangles()[0],
            [0, 0, 0],
            "removed triangles read as degenerate"
        );
        assert_eq!(m.corner_count(1), 1);

        let undo = patch.revert(&mut m).unwrap();
        assert_eq!(undo.triangles_added, vec![0]);
        assert_eq!(undo.triangles_removed, vec![2]);
        assert_eq!(undo.vertices_removed, vec![4]);
        assert_eq!((m.vertex_count(), m.triangle_count()), (4, 2));
        assert_eq!(m.triangle(0), Some([0, 1, 2]));
        assert!(!m.is_vertex_alive(4));
        assert_eq!(m.corner_count(1), 1);
    }

    #[test]
    fn validation_errors_leave_the_transaction_usable() {
        let mut m = quad();
        let mut tx = m.begin_edit();
        assert_eq!(
            tx.set_positions(&[9], &[[0.; 3]]),
            Err(StoreError::InvalidVertex(9))
        );
        assert_eq!(
            tx.remove_vertices(&[0]),
            Err(StoreError::VertexInUse {
                vertex: 0,
                corners: 2
            })
        );
        assert!(matches!(
            tx.remove_triangles(&[0, 0]),
            Err(StoreError::InvalidArgument(_))
        ));
        tx.set_positions(&[0], &[[1., 2., 3.]]).unwrap();
        tx.commit().unwrap();
        assert_eq!(m.position(0), Some([1., 2., 3.]));
    }

    #[test]
    fn rollback_and_drop_discard_everything() {
        let mut m = quad();
        let before = m.revision();
        let mut tx = m.begin_edit();
        tx.add_vertices(&[[9.; 3]]).unwrap();
        tx.remove_triangles(&[1]).unwrap();
        tx.rollback();
        assert_eq!(
            (m.vertex_slots(), m.vertex_count(), m.triangle_count()),
            (4, 4, 2)
        );
        assert_eq!(m.triangle(1), Some([0, 2, 3]));
        assert_eq!(m.revision(), before);

        {
            let mut tx = m.begin_edit();
            tx.set_positions(&[0], &[[7.; 3]]).unwrap();
        }
        assert_eq!(m.position(0), Some([0., 0., 0.]));
    }

    #[test]
    fn empty_transactions_keep_the_revision() {
        let mut m = quad();
        let patch = m.begin_edit().commit().unwrap();
        assert!(patch.is_empty());
        assert_eq!(m.revision(), 0);
    }

    #[test]
    fn new_vertices_take_channel_values_from_parents() {
        let mut m = quad();
        m.add_channel(ChannelDesc::source("mask", 1)).unwrap();
        m.add_channel(ChannelDesc::source("id", 1).with_policy(TopologyPolicy::Nearest))
            .unwrap();
        let mut tx = m.begin_edit();
        tx.write_channel("mask", &[0, 1], &[2.0, 4.0]).unwrap();
        tx.write_channel("id", &[0, 1], &[10.0, 20.0]).unwrap();
        let v = tx
            .add_vertex_with_parents([0.5, 0., 0.], &[(0, 0.75), (1, 0.25)])
            .unwrap();
        let patch = tx.commit().unwrap();
        assert_eq!(m.channel("mask").unwrap()[v as usize], 2.5);
        assert_eq!(m.channel("id").unwrap()[v as usize], 10.0);
        assert_eq!(
            patch.changes().channels,
            vec![
                ("mask".to_string(), vec![0, 1]),
                ("id".to_string(), vec![0, 1])
            ]
        );
        patch.revert(&mut m).unwrap();
        assert_eq!(&m.channel("mask").unwrap()[..2], &[0.0, 0.0]);
    }

    #[test]
    fn writing_a_derived_channel_in_a_transaction_is_rejected() {
        let mut m = quad();
        m.add_channel(ChannelDesc::derived("dist", 1)).unwrap();
        let mut tx = m.begin_edit();
        assert_eq!(
            tx.write_channel("dist", &[0], &[1.0]),
            Err(StoreError::NotSourceChannel("dist".into()))
        );
    }

    #[test]
    fn slots_created_and_removed_in_one_transaction_are_reused() {
        let mut m = quad();
        let mut tx = m.begin_edit();
        let v = tx.add_vertices(&[[0.; 3]]).unwrap()[0];
        tx.remove_vertices(&[v]).unwrap();
        let patch = tx.commit().unwrap();
        assert!(patch.is_empty());
        assert_eq!(m.free_vertex_slots(), &[v]);
        let mut tx = m.begin_edit();
        assert_eq!(tx.add_vertices(&[[1.; 3]]).unwrap(), vec![v]);
    }

    #[test]
    fn undoing_an_append_past_a_bitset_word_kills_the_slot() {
        // 64 vertices fill exactly one liveness word; vertex 64 starts a new word.
        let positions = vec![0.0f32; 64 * 3];
        let mut m = Mesh::from_buffers(&positions, &[]).unwrap();
        let mut tx = m.begin_edit();
        let v = tx.add_vertices(&[[1.; 3]]).unwrap()[0];
        let t = tx.add_triangles(&[[0, 1, v]]).unwrap()[0];
        let patch = tx.commit().unwrap();
        assert_eq!(v, 64);
        patch.revert(&mut m).unwrap();
        assert!(!m.is_vertex_alive(64));
        assert!(!m.is_triangle_alive(t));
        assert_eq!(m.triangles()[t as usize], [0, 0, 0]);
        assert_eq!(m.vertex_count(), 64);
        patch.apply(&mut m).unwrap();
        assert!(m.is_vertex_alive(64));
        assert_eq!(m.triangle(t), Some([0, 1, 64]));
    }

    #[test]
    fn patches_only_apply_at_the_matching_revision() {
        let mut m = quad();
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[1.; 3]]).unwrap();
        let first = tx.commit().unwrap();
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[2.; 3]]).unwrap();
        let _second = tx.commit().unwrap();
        assert_eq!(first.revert(&mut m).err(), Some(StoreError::PatchMismatch));
        let mut other = quad();
        assert_eq!(
            first.apply(&mut other).err(),
            Some(StoreError::PatchMismatch)
        );
    }
}
```

Replace `src/lib.rs` with:

```rust
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
mod patch;
mod transaction;

pub use channel::{ChannelDesc, ChannelKind, TopologyPolicy};
pub use error::StoreError;
pub use mesh::Mesh;
pub use patch::{ChangeSet, Patch};
pub use transaction::Transaction;

/// Id of a vertex slot.
pub type VertexId = u32;
/// Id of a triangle slot.
pub type TriangleId = u32;

pub type Result<T> = std::result::Result<T, StoreError>;
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p fast-mesh-store transaction`
Expected: compile errors, `file not found for module patch` and `no method named begin_edit`.

- [ ] **Step 3: Implement patches**

Create `src/patch.rs`:

```rust
use crate::column::ColumnPatch;
use crate::{Mesh, Result, StoreError, TriangleId, VertexId};

/// What an edit, undo or redo changed. All lists are sorted and free of duplicates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChangeSet {
    pub vertices_added: Vec<VertexId>,
    pub vertices_removed: Vec<VertexId>,
    /// Existing vertices whose position changed.
    pub vertices_moved: Vec<VertexId>,
    pub triangles_added: Vec<TriangleId>,
    pub triangles_removed: Vec<TriangleId>,
    /// Existing triangles whose vertex ids changed.
    pub triangles_changed: Vec<TriangleId>,
    /// Source channels written for existing vertices: (channel name, vertex ids).
    pub channels: Vec<(String, Vec<VertexId>)>,
}

impl ChangeSet {
    /// True when triangles were added, removed or rewritten.
    pub fn topology_changed(&self) -> bool {
        !(self.triangles_added.is_empty()
            && self.triangles_removed.is_empty()
            && self.triangles_changed.is_empty())
    }

    pub fn is_empty(&self) -> bool {
        !self.topology_changed()
            && self.vertices_added.is_empty()
            && self.vertices_removed.is_empty()
            && self.vertices_moved.is_empty()
            && self.channels.iter().all(|(_, v)| v.is_empty())
    }

    fn inverted(mut self) -> Self {
        std::mem::swap(&mut self.vertices_added, &mut self.vertices_removed);
        std::mem::swap(&mut self.triangles_added, &mut self.triangles_removed);
        self
    }
}

/// The recorded effect of one committed transaction. Re-apply it with
/// [`Patch::apply`] (redo) or undo it with [`Patch::revert`].
///
/// A patch only applies to the mesh that produced it, at the matching revision:
/// `revert` needs the mesh exactly as the commit left it, `apply` needs it exactly as
/// it was before the commit.
#[derive(Debug, Clone)]
pub struct Patch {
    pub(crate) mesh_uid: u64,
    pub(crate) epoch: u32,
    pub(crate) base_revision: u64,
    pub(crate) revision: u64,
    pub(crate) positions: ColumnPatch<[f32; 3]>,
    pub(crate) triangles: ColumnPatch<[u32; 3]>,
    pub(crate) vertex_live: ColumnPatch<u64>,
    pub(crate) triangle_live: ColumnPatch<u64>,
    pub(crate) vertex_refs: ColumnPatch<u32>,
    pub(crate) channel_data: Vec<(String, ColumnPatch<f32>)>,
    pub(crate) changes: ChangeSet,
}

impl Patch {
    /// The changes this patch makes when applied.
    pub fn changes(&self) -> &ChangeSet {
        &self.changes
    }

    /// True when the transaction changed nothing.
    pub fn is_empty(&self) -> bool {
        self.base_revision == self.revision
    }

    /// Mesh revision before the commit.
    pub fn base_revision(&self) -> u64 {
        self.base_revision
    }

    /// Mesh revision after the commit.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Approximate heap memory held by this patch.
    pub fn byte_size(&self) -> usize {
        let c = &self.changes;
        let ids = c.vertices_added.len()
            + c.vertices_removed.len()
            + c.vertices_moved.len()
            + c.triangles_added.len()
            + c.triangles_removed.len()
            + c.triangles_changed.len()
            + c.channels.iter().map(|(_, v)| v.len()).sum::<usize>();
        self.positions.byte_size()
            + self.triangles.byte_size()
            + self.vertex_live.byte_size()
            + self.triangle_live.byte_size()
            + self.vertex_refs.byte_size()
            + self
                .channel_data
                .iter()
                .map(|(n, p)| n.len() + p.byte_size())
                .sum::<usize>()
            + ids * 4
    }

    /// Re-applies the patch (redo). The mesh must be at [`Patch::base_revision`].
    pub fn apply(&self, mesh: &mut Mesh) -> Result<ChangeSet> {
        self.check(mesh, self.base_revision)?;
        self.positions.apply(&mut mesh.positions)?;
        self.triangles.apply(&mut mesh.triangles)?;
        self.vertex_live.apply(&mut mesh.vertex_live)?;
        self.triangle_live.apply(&mut mesh.triangle_live)?;
        self.vertex_refs.apply(&mut mesh.vertex_refs)?;
        for (name, p) in &self.channel_data {
            if let Some(i) = mesh.channel_index(name) {
                p.apply(&mut mesh.channels[i].data)?;
            }
        }
        mesh.grow_channels()?;
        let c = &self.changes;
        mesh.live_vertices = mesh.live_vertices + c.vertices_added.len() - c.vertices_removed.len();
        mesh.live_triangles =
            mesh.live_triangles + c.triangles_added.len() - c.triangles_removed.len();
        mesh.revision = self.revision;
        if c.topology_changed() {
            mesh.on_topology_changed();
        }
        Ok(self.changes.clone())
    }

    /// Undoes the patch. The mesh must be at [`Patch::revision`].
    pub fn revert(&self, mesh: &mut Mesh) -> Result<ChangeSet> {
        self.check(mesh, self.revision)?;
        self.positions.revert(&mut mesh.positions);
        self.triangles.revert(&mut mesh.triangles);
        self.vertex_live.revert(&mut mesh.vertex_live);
        self.triangle_live.revert(&mut mesh.triangle_live);
        self.vertex_refs.revert(&mut mesh.vertex_refs);
        for (name, p) in &self.channel_data {
            if let Some(i) = mesh.channel_index(name) {
                p.revert(&mut mesh.channels[i].data);
            }
        }
        let c = &self.changes;
        mesh.live_vertices = mesh.live_vertices + c.vertices_removed.len() - c.vertices_added.len();
        mesh.live_triangles =
            mesh.live_triangles + c.triangles_removed.len() - c.triangles_added.len();
        mesh.revision = self.base_revision;
        if c.topology_changed() {
            mesh.on_topology_changed();
        }
        Ok(self.changes.clone().inverted())
    }

    fn check(&self, mesh: &Mesh, expected_revision: u64) -> Result<()> {
        if self.mesh_uid != mesh.uid {
            return Err(StoreError::PatchMismatch);
        }
        if self.epoch != mesh.epoch {
            return Err(StoreError::StalePatch);
        }
        if mesh.revision != expected_revision {
            return Err(StoreError::PatchMismatch);
        }
        Ok(())
    }
}
```

- [ ] **Step 4: Implement transactions**

Insert above the test module in `src/transaction.rs`:

```rust
use crate::channel::ChannelKind;
use crate::mesh::set_bit;
use crate::patch::{ChangeSet, Patch};
use crate::{Mesh, Result, StoreError, TriangleId, VertexId};

/// An open edit on a [`Mesh`]. All writes go through it.
///
/// Every method validates its whole input before writing, so a returned validation
/// error leaves the transaction unchanged and usable. If a write itself fails (out of
/// memory), the transaction is poisoned and `commit` rolls it back.
///
/// Dropping a transaction without committing rolls it back.
pub struct Transaction<'m> {
    mesh: &'m mut Mesh,
    allocated_vertices: Vec<VertexId>,
    removed_vertices: Vec<VertexId>,
    moved_vertices: Vec<VertexId>,
    allocated_triangles: Vec<TriangleId>,
    removed_triangles: Vec<TriangleId>,
    changed_triangles: Vec<TriangleId>,
    /// Slots taken from the free lists, in order, so rollback can return them.
    popped_vertices: Vec<VertexId>,
    popped_triangles: Vec<TriangleId>,
    /// Per channel (same order as `mesh.channels`): vertices written by `write_channel`.
    channel_writes: Vec<Vec<VertexId>>,
    live_at_start: (usize, usize),
    poisoned: bool,
    finished: bool,
}

impl Mesh {
    /// Opens a transaction. The mesh cannot be used otherwise until it ends.
    pub fn begin_edit(&mut self) -> Transaction<'_> {
        self.positions.begin();
        self.triangles.begin();
        self.vertex_live.begin();
        self.triangle_live.begin();
        self.vertex_refs.begin();
        for c in &mut self.channels {
            c.data.begin();
        }
        Transaction {
            channel_writes: vec![Vec::new(); self.channels.len()],
            live_at_start: (self.live_vertices, self.live_triangles),
            mesh: self,
            allocated_vertices: Vec::new(),
            removed_vertices: Vec::new(),
            moved_vertices: Vec::new(),
            allocated_triangles: Vec::new(),
            removed_triangles: Vec::new(),
            changed_triangles: Vec::new(),
            popped_vertices: Vec::new(),
            popped_triangles: Vec::new(),
            poisoned: false,
            finished: false,
        }
    }
}

fn sorted(mut ids: Vec<u32>) -> Vec<u32> {
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn without(ids: Vec<u32>, a: &[u32], b: &[u32]) -> Vec<u32> {
    ids.into_iter()
        .filter(|i| a.binary_search(i).is_err() && b.binary_search(i).is_err())
        .collect()
}

fn check_unique(ids: &[u32], what: &'static str) -> Result<()> {
    let mut s = ids.to_vec();
    s.sort_unstable();
    if s.windows(2).any(|w| w[0] == w[1]) {
        return Err(StoreError::InvalidArgument(what));
    }
    Ok(())
}

impl Transaction<'_> {
    /// Read access to the mesh as edited so far.
    pub fn mesh(&self) -> &Mesh {
        self.mesh
    }

    fn check_vertex(&self, v: VertexId) -> Result<()> {
        if self.mesh.is_vertex_alive(v) {
            Ok(())
        } else {
            Err(StoreError::InvalidVertex(v))
        }
    }

    fn check_triangle(&self, t: TriangleId) -> Result<()> {
        if self.mesh.is_triangle_alive(t) {
            Ok(())
        } else {
            Err(StoreError::InvalidTriangle(t))
        }
    }

    /// Runs the write phase of an operation; any failure poisons the transaction.
    fn write<R>(&mut self, f: impl FnOnce(&mut Self) -> Result<R>) -> Result<R> {
        if self.poisoned {
            return Err(StoreError::TransactionPoisoned);
        }
        let r = f(self);
        if r.is_err() {
            self.poisoned = true;
        }
        r
    }

    fn alloc_vertex(
        &mut self,
        position: [f32; 3],
        parents: &[(VertexId, f32)],
    ) -> Result<VertexId> {
        let m = &mut *self.mesh;
        let id = match m.vertex_free.pop() {
            Some(id) => {
                self.popped_vertices.push(id);
                debug_assert!(!m.is_vertex_alive(id), "free list held alive vertex {id}");
                m.positions.set(id as usize, position);
                id
            }
            None => {
                let id = m.positions.push(position)?;
                m.vertex_refs.push(0)?;
                for c in &mut m.channels {
                    for _ in 0..c.desc.width {
                        c.data.push(c.desc.default)?;
                    }
                }
                u32::try_from(id).map_err(|_| StoreError::OutOfMemory)?
            }
        };
        for c in &mut m.channels {
            let w = c.desc.width;
            for k in 0..w {
                let value = c.initial_value(parents, k);
                c.data.set(id as usize * w + k, value);
            }
        }
        set_bit(&mut m.vertex_live, id as usize, true)?;
        m.live_vertices += 1;
        self.allocated_vertices.push(id);
        Ok(id)
    }

    fn alloc_triangle(&mut self, tri: [VertexId; 3]) -> Result<TriangleId> {
        let m = &mut *self.mesh;
        let id = match m.triangle_free.pop() {
            Some(id) => {
                self.popped_triangles.push(id);
                debug_assert!(
                    !m.is_triangle_alive(id),
                    "free list held alive triangle {id}"
                );
                m.triangles.set(id as usize, tri);
                id
            }
            None => {
                let id = m.triangles.push(tri)?;
                u32::try_from(id).map_err(|_| StoreError::OutOfMemory)?
            }
        };
        for v in tri {
            let r = m.vertex_refs.get(v as usize);
            m.vertex_refs.set(v as usize, r + 1);
        }
        set_bit(&mut m.triangle_live, id as usize, true)?;
        m.live_triangles += 1;
        self.allocated_triangles.push(id);
        Ok(id)
    }

    fn release_corners(&mut self, tri: [VertexId; 3]) {
        let m = &mut *self.mesh;
        for v in tri {
            let r = m.vertex_refs.get(v as usize);
            m.vertex_refs.set(v as usize, r - 1);
        }
    }

    /// Adds vertices without creation parents; source channels get their default value.
    pub fn add_vertices(&mut self, positions: &[[f32; 3]]) -> Result<Vec<VertexId>> {
        self.write(|tx| positions.iter().map(|&p| tx.alloc_vertex(p, &[])).collect())
    }

    /// Adds a vertex created from existing vertices, e.g. an edge split creates a vertex
    /// with parents `[(a, 0.5), (b, 0.5)]`. Source channels derive its values from the
    /// parents according to their [`crate::TopologyPolicy`].
    pub fn add_vertex_with_parents(
        &mut self,
        position: [f32; 3],
        parents: &[(VertexId, f32)],
    ) -> Result<VertexId> {
        for &(v, w) in parents {
            self.check_vertex(v)?;
            if !w.is_finite() || w < 0.0 {
                return Err(StoreError::InvalidArgument(
                    "parent weights must be finite and non-negative",
                ));
            }
        }
        self.write(|tx| tx.alloc_vertex(position, parents))
    }

    /// Sets positions of existing vertices.
    pub fn set_positions(&mut self, ids: &[VertexId], positions: &[[f32; 3]]) -> Result<()> {
        if ids.len() != positions.len() {
            return Err(StoreError::LengthMismatch {
                expected: ids.len(),
                got: positions.len(),
            });
        }
        for &v in ids {
            self.check_vertex(v)?;
        }
        self.write(|tx| {
            for (&v, &p) in ids.iter().zip(positions) {
                tx.mesh.positions.set(v as usize, p);
            }
            tx.moved_vertices.extend_from_slice(ids);
            Ok(())
        })
    }

    /// Removes vertices that no triangle uses.
    pub fn remove_vertices(&mut self, ids: &[VertexId]) -> Result<()> {
        check_unique(ids, "duplicate vertex id")?;
        for &v in ids {
            self.check_vertex(v)?;
            let corners = self.mesh.corner_count(v);
            if corners > 0 {
                return Err(StoreError::VertexInUse { vertex: v, corners });
            }
        }
        self.write(|tx| {
            for &v in ids {
                set_bit(&mut tx.mesh.vertex_live, v as usize, false)?;
                tx.mesh.live_vertices -= 1;
            }
            tx.removed_vertices.extend_from_slice(ids);
            Ok(())
        })
    }

    /// Adds triangles between existing vertices.
    pub fn add_triangles(&mut self, triangles: &[[VertexId; 3]]) -> Result<Vec<TriangleId>> {
        for t in triangles {
            for &v in t {
                self.check_vertex(v)?;
            }
        }
        self.write(|tx| triangles.iter().map(|&t| tx.alloc_triangle(t)).collect())
    }

    /// Removes triangles. Their vertices stay; remove them separately if unused.
    pub fn remove_triangles(&mut self, ids: &[TriangleId]) -> Result<()> {
        check_unique(ids, "duplicate triangle id")?;
        for &t in ids {
            self.check_triangle(t)?;
        }
        self.write(|tx| {
            for &t in ids {
                let tri = tx.mesh.triangles.get(t as usize);
                tx.release_corners(tri);
                tx.mesh.triangles.set(t as usize, [0; 3]);
                set_bit(&mut tx.mesh.triangle_live, t as usize, false)?;
                tx.mesh.live_triangles -= 1;
            }
            tx.removed_triangles.extend_from_slice(ids);
            Ok(())
        })
    }

    /// Replaces the vertex ids of existing triangles (e.g. an edge flip).
    pub fn set_triangles(&mut self, ids: &[TriangleId], triangles: &[[VertexId; 3]]) -> Result<()> {
        if ids.len() != triangles.len() {
            return Err(StoreError::LengthMismatch {
                expected: ids.len(),
                got: triangles.len(),
            });
        }
        check_unique(ids, "duplicate triangle id")?;
        for (&t, tri) in ids.iter().zip(triangles) {
            self.check_triangle(t)?;
            for &v in tri {
                self.check_vertex(v)?;
            }
        }
        self.write(|tx| {
            for (&t, &tri) in ids.iter().zip(triangles) {
                let old = tx.mesh.triangles.get(t as usize);
                tx.release_corners(old);
                for v in tri {
                    let r = tx.mesh.vertex_refs.get(v as usize);
                    tx.mesh.vertex_refs.set(v as usize, r + 1);
                }
                tx.mesh.triangles.set(t as usize, tri);
            }
            tx.changed_triangles.extend_from_slice(ids);
            Ok(())
        })
    }

    /// Writes source channel values: `values` holds `width` values per vertex in `ids`.
    pub fn write_channel(&mut self, name: &str, ids: &[VertexId], values: &[f32]) -> Result<()> {
        let ci = self
            .mesh
            .channel_index(name)
            .ok_or_else(|| StoreError::UnknownChannel(name.into()))?;
        let desc = &self.mesh.channels[ci].desc;
        if desc.kind != ChannelKind::Source {
            return Err(StoreError::NotSourceChannel(name.into()));
        }
        let w = desc.width;
        if values.len() != ids.len() * w {
            return Err(StoreError::LengthMismatch {
                expected: ids.len() * w,
                got: values.len(),
            });
        }
        for &v in ids {
            self.check_vertex(v)?;
        }
        self.write(|tx| {
            let data = &mut tx.mesh.channels[ci].data;
            for (n, &v) in ids.iter().enumerate() {
                for k in 0..w {
                    data.set(v as usize * w + k, values[n * w + k]);
                }
            }
            tx.channel_writes[ci].extend_from_slice(ids);
            Ok(())
        })
    }

    /// Ends the transaction and returns its patch. An empty transaction returns an
    /// empty patch and leaves the revision unchanged.
    pub fn commit(mut self) -> Result<Patch> {
        self.finished = true;
        if self.poisoned {
            self.rollback_inner();
            return Err(StoreError::TransactionPoisoned);
        }
        let m = &mut *self.mesh;

        // Slots allocated and removed within this transaction were never visible:
        // they go straight back to the free lists.
        let allocated_v = sorted(std::mem::take(&mut self.allocated_vertices));
        let (created_v, transient_v): (Vec<_>, Vec<_>) =
            allocated_v.iter().partition(|&&v| m.is_vertex_alive(v));
        let allocated_t = sorted(std::mem::take(&mut self.allocated_triangles));
        let (created_t, transient_t): (Vec<_>, Vec<_>) =
            allocated_t.iter().partition(|&&t| m.is_triangle_alive(t));
        m.vertex_free.extend_from_slice(&transient_v);
        m.triangle_free.extend_from_slice(&transient_t);

        let freed_v = without(
            sorted(std::mem::take(&mut self.removed_vertices)),
            &allocated_v,
            &[],
        );
        let freed_t = without(
            sorted(std::mem::take(&mut self.removed_triangles)),
            &allocated_t,
            &[],
        );
        let moved = without(
            sorted(std::mem::take(&mut self.moved_vertices)),
            &allocated_v,
            &freed_v,
        );
        let changed_t = without(
            sorted(std::mem::take(&mut self.changed_triangles)),
            &allocated_t,
            &freed_t,
        );

        let mut channels = Vec::new();
        let mut channel_data = Vec::new();
        for (ci, c) in m.channels.iter_mut().enumerate() {
            let patch = c.data.finish();
            if c.desc.kind == ChannelKind::Source {
                let written = std::mem::take(&mut self.channel_writes[ci]);
                let written = without(sorted(written), &allocated_v, &freed_v);
                if !written.is_empty() {
                    channels.push((c.desc.name.clone(), written));
                }
                if !patch.is_empty() {
                    channel_data.push((c.desc.name.clone(), patch));
                }
            }
        }

        let changes = ChangeSet {
            vertices_added: created_v,
            vertices_removed: freed_v,
            vertices_moved: moved,
            triangles_added: created_t,
            triangles_removed: freed_t,
            triangles_changed: changed_t,
            channels,
        };
        let base_revision = m.revision;
        if !changes.is_empty() {
            m.next_revision += 1;
            m.revision = m.next_revision;
        }
        let patch = Patch {
            mesh_uid: m.uid,
            epoch: m.epoch,
            base_revision,
            revision: m.revision,
            positions: m.positions.finish(),
            triangles: m.triangles.finish(),
            vertex_live: m.vertex_live.finish(),
            triangle_live: m.triangle_live.finish(),
            vertex_refs: m.vertex_refs.finish(),
            channel_data,
            changes,
        };
        if patch.changes.topology_changed() {
            m.on_topology_changed();
        }
        Ok(patch)
    }

    /// Discards every change made in this transaction.
    pub fn rollback(mut self) {
        self.finished = true;
        self.rollback_inner();
    }

    fn rollback_inner(&mut self) {
        let m = &mut *self.mesh;
        m.positions.rollback();
        m.triangles.rollback();
        m.vertex_live.rollback();
        m.triangle_live.rollback();
        m.vertex_refs.rollback();
        for c in &mut m.channels {
            c.data.rollback();
        }
        while let Some(v) = self.popped_vertices.pop() {
            m.vertex_free.push(v);
        }
        while let Some(t) = self.popped_triangles.pop() {
            m.triangle_free.push(t);
        }
        (m.live_vertices, m.live_triangles) = self.live_at_start;
    }
}

impl Drop for Transaction<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.rollback_inner();
        }
    }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p fast-mesh-store`
Expected: all tests pass, including the 10 transaction tests (`moving_vertices_commits_a_reversible_patch`, `adding_and_removing_topology_round_trips`, `validation_errors_leave_the_transaction_usable`, `rollback_and_drop_discard_everything`, `empty_transactions_keep_the_revision`, `new_vertices_take_channel_values_from_parents`, `writing_a_derived_channel_in_a_transaction_is_rejected`, `slots_created_and_removed_in_one_transaction_are_reused`, `undoing_an_append_past_a_bitset_word_kills_the_slot`, `patches_only_apply_at_the_matching_revision`).

- [ ] **Step 6: Commit**

```bash
git add crates/fast-mesh-store
git commit -m "feat(store): add transactions with journaled commits and reversible patches"
```

---

### Task 5: History

**Files:**
- Create: `crates/fast-mesh-store/src/history.rs`
- Create: `crates/fast-mesh-store/src/compact.rs` (only `Remap`, needed by `History::compact`; `Mesh::compact` comes in Task 6)
- Modify: `crates/fast-mesh-store/src/lib.rs`

**Interfaces:**
- Consumes: `Patch` fields `mesh_uid`, `revision`, `changes` and `Patch::{apply, revert, is_empty, byte_size}` (Task 4); `Mesh::{release_vertices, release_triangles}` (Task 3).
- Produces: `pub struct History` with `new(budget_bytes)`, `push(&mut Mesh, Patch) -> Result<()>`, `undo(&mut Mesh) -> Result<Option<ChangeSet>>`, `redo(&mut Mesh) -> Result<Option<ChangeSet>>`, `clear(&mut Mesh)`, `compact(&mut Mesh) -> Result<Remap>`, `set_budget(&mut Mesh, usize)`, `can_undo`, `can_redo`, `undo_len`, `redo_len`, `bytes`, `budget`.

Because `History::compact` calls `Mesh::compact`, this task creates `compact.rs` with the full Task 6 implementation but without its tests; Task 6 adds the tests. (Splitting them would leave `History::compact` uncompilable.)

- [ ] **Step 1: Write the failing tests**

Create `src/history.rs` with:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> Mesh {
        Mesh::from_buffers(
            &[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap()
    }

    fn move_vertex(m: &mut Mesh, h: &mut History, v: u32, x: f32) {
        let mut tx = m.begin_edit();
        tx.set_positions(&[v], &[[x, 0., 0.]]).unwrap();
        let p = tx.commit().unwrap();
        h.push(m, p).unwrap();
    }

    #[test]
    fn undo_and_redo_walk_the_stack() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        move_vertex(&mut m, &mut h, 0, 5.0);
        move_vertex(&mut m, &mut h, 0, 6.0);
        assert_eq!(h.undo(&mut m).unwrap().unwrap().vertices_moved, vec![0]);
        assert_eq!(m.position(0).unwrap()[0], 5.0);
        h.undo(&mut m).unwrap();
        assert_eq!(m.position(0).unwrap()[0], 0.0);
        assert!(h.undo(&mut m).unwrap().is_none());
        h.redo(&mut m).unwrap();
        h.redo(&mut m).unwrap();
        assert_eq!(m.position(0).unwrap()[0], 6.0);
        assert!(h.redo(&mut m).unwrap().is_none());
    }

    #[test]
    fn a_new_commit_discards_redo() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        move_vertex(&mut m, &mut h, 0, 5.0);
        h.undo(&mut m).unwrap();
        move_vertex(&mut m, &mut h, 1, 7.0);
        assert!(!h.can_redo());
        assert_eq!(h.undo_len(), 1);
    }

    #[test]
    fn empty_commits_keep_redo() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        move_vertex(&mut m, &mut h, 0, 5.0);
        h.undo(&mut m).unwrap();
        let empty = m.begin_edit().commit().unwrap();
        h.push(&mut m, empty).unwrap();
        assert!(h.can_redo());
    }

    #[test]
    fn budget_evicts_the_oldest_patches() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        move_vertex(&mut m, &mut h, 0, 1.0);
        let one = h.bytes();
        h.set_budget(&mut m, one * 2);
        move_vertex(&mut m, &mut h, 0, 2.0);
        move_vertex(&mut m, &mut h, 0, 3.0);
        assert_eq!(h.undo_len(), 2);
        assert!(h.bytes() <= h.budget());
        h.undo(&mut m).unwrap();
        h.undo(&mut m).unwrap();
        assert_eq!(
            m.position(0).unwrap()[0],
            1.0,
            "the first edit can no longer be undone"
        );
    }

    #[test]
    fn removed_slots_are_reserved_until_their_patch_is_evicted() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[1]).unwrap();
        tx.remove_vertices(&[3]).unwrap();
        let p = tx.commit().unwrap();
        h.push(&mut m, p).unwrap();
        assert!(m.free_vertex_slots().is_empty(), "reserved while undoable");

        let mut tx = m.begin_edit();
        assert_eq!(
            tx.add_vertices(&[[0.; 3]]).unwrap(),
            vec![4],
            "slot 3 is not reused"
        );
        tx.rollback();

        h.clear(&mut m);
        assert_eq!(m.free_vertex_slots(), &[3]);
        assert_eq!(m.free_triangle_slots(), &[1]);
    }

    #[test]
    fn created_slots_of_discarded_redo_are_released() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        let mut tx = m.begin_edit();
        tx.add_vertices(&[[0.; 3]]).unwrap();
        let p = tx.commit().unwrap();
        h.push(&mut m, p).unwrap();
        h.undo(&mut m).unwrap();
        assert!(m.free_vertex_slots().is_empty(), "reserved for redo");
        move_vertex(&mut m, &mut h, 0, 1.0);
        assert_eq!(m.free_vertex_slots(), &[4]);
    }

    #[test]
    fn rejects_patches_that_are_not_the_latest_commit() {
        let mut m = quad();
        let mut h = History::new(1 << 20);
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[1.; 3]]).unwrap();
        let first = tx.commit().unwrap();
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[2.; 3]]).unwrap();
        let _second = tx.commit().unwrap();
        assert_eq!(h.push(&mut m, first).err(), Some(StoreError::PatchMismatch));
    }
}
```

Replace `src/lib.rs` with:

```rust
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
mod compact;
mod error;
mod history;
mod mesh;
mod patch;
mod transaction;

pub use channel::{ChannelDesc, ChannelKind, TopologyPolicy};
pub use compact::Remap;
pub use error::StoreError;
pub use history::History;
pub use mesh::Mesh;
pub use patch::{ChangeSet, Patch};
pub use transaction::Transaction;

/// Id of a vertex slot.
pub type VertexId = u32;
/// Id of a triangle slot.
pub type TriangleId = u32;

pub type Result<T> = std::result::Result<T, StoreError>;
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p fast-mesh-store history`
Expected: compile errors, `cannot find type History` and `file not found for module compact`.

- [ ] **Step 3: Implement compaction support**

Create `src/compact.rs`:

```rust
use crate::column::{Column, try_vec};
use crate::mesh::full_bits;
use crate::{Mesh, Result, TriangleId, VertexId};

/// Old slot id → new id mapping produced by [`Mesh::compact`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Remap {
    /// Indexed by old vertex slot; [`Remap::REMOVED`] for dead slots.
    pub vertices: Vec<VertexId>,
    /// Indexed by old triangle slot; [`Remap::REMOVED`] for dead slots.
    pub triangles: Vec<TriangleId>,
}

impl Remap {
    pub const REMOVED: u32 = u32::MAX;

    pub fn vertex(&self, old: VertexId) -> Option<VertexId> {
        self.vertices
            .get(old as usize)
            .copied()
            .filter(|&v| v != Self::REMOVED)
    }

    pub fn triangle(&self, old: TriangleId) -> Option<TriangleId> {
        self.triangles
            .get(old as usize)
            .copied()
            .filter(|&t| t != Self::REMOVED)
    }
}

impl Mesh {
    /// Removes dead slots and renumbers alive vertices and triangles in their current
    /// order. All existing patches become stale (`StalePatch`); use
    /// [`crate::History::compact`] to clear a history at the same time.
    pub fn compact(&mut self) -> Result<Remap> {
        let mut vmap = try_vec(self.vertex_slots(), Remap::REMOVED)?;
        let mut alive_v = Vec::with_capacity(self.vertex_count());
        for v in 0..self.vertex_slots() as u32 {
            if self.is_vertex_alive(v) {
                vmap[v as usize] = alive_v.len() as u32;
                alive_v.push(v as usize);
            }
        }
        let mut tmap = try_vec(self.triangle_slots(), Remap::REMOVED)?;
        let mut tris = Vec::with_capacity(self.triangle_count());
        for t in 0..self.triangle_slots() as u32 {
            if let Some(tri) = self.triangle(t) {
                tmap[t as usize] = tris.len() as u32;
                tris.push(tri.map(|v| vmap[v as usize]));
            }
        }

        let positions = alive_v.iter().map(|&v| self.positions.get(v)).collect();
        let refs = alive_v.iter().map(|&v| self.vertex_refs.get(v)).collect();
        for c in &mut self.channels {
            let w = c.desc.width;
            let data = c.data.as_slice();
            let compacted = alive_v
                .iter()
                .flat_map(|&v| data[v * w..(v + 1) * w].iter().copied())
                .collect();
            c.data = Column::from_vec(compacted);
        }
        self.vertex_live = Column::from_vec(full_bits(alive_v.len())?);
        self.triangle_live = Column::from_vec(full_bits(tris.len())?);
        self.positions = Column::from_vec(positions);
        self.vertex_refs = Column::from_vec(refs);
        self.triangles = Column::from_vec(tris);
        self.vertex_free.clear();
        self.triangle_free.clear();
        self.epoch += 1;
        self.next_revision += 1;
        self.revision = self.next_revision;
        self.on_topology_changed();
        Ok(Remap {
            vertices: vmap,
            triangles: tmap,
        })
    }
}
```

- [ ] **Step 4: Implement the history**

Insert above the test module in `src/history.rs`:

```rust
use std::collections::VecDeque;

use crate::compact::Remap;
use crate::{ChangeSet, Mesh, Patch, Result, StoreError};

/// Undo/redo stacks of [`Patch`]es for one mesh, limited by a memory budget.
///
/// Every commit on the mesh must be pushed here (or the history cleared): undo and
/// redo only work when the mesh revision matches the top patch.
///
/// The history also decides when removed slots may be reused. A slot freed by a
/// patch stays reserved while that patch can still be undone, and a slot created by
/// an undone patch stays reserved while it can be redone. Evicting or discarding the
/// patch releases the slot. Patches kept outside a `History` never release their
/// slots; those are reclaimed by [`Mesh::compact`].
pub struct History {
    undo: VecDeque<Patch>,
    redo: Vec<Patch>,
    budget: usize,
    bytes: usize,
}

impl History {
    /// A history that keeps at most `budget_bytes` of patches (undo and redo combined).
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            undo: VecDeque::new(),
            redo: Vec::new(),
            budget: budget_bytes,
            bytes: 0,
        }
    }

    /// Records a just-committed patch. Empty patches are ignored and keep the redo
    /// stack. Otherwise the redo stack is discarded and the oldest patches are evicted
    /// until the budget is met (possibly including this one, if it alone is too large).
    pub fn push(&mut self, mesh: &mut Mesh, patch: Patch) -> Result<()> {
        if patch.mesh_uid != mesh.uid || patch.revision != mesh.revision {
            return Err(StoreError::PatchMismatch);
        }
        if patch.is_empty() {
            return Ok(());
        }
        for p in self.redo.drain(..) {
            self.bytes -= p.byte_size();
            mesh.release_vertices(&p.changes.vertices_added);
            mesh.release_triangles(&p.changes.triangles_added);
        }
        self.bytes += patch.byte_size();
        self.undo.push_back(patch);
        self.evict(mesh);
        Ok(())
    }

    /// Undoes the latest patch. Returns `None` when there is nothing to undo.
    pub fn undo(&mut self, mesh: &mut Mesh) -> Result<Option<ChangeSet>> {
        let Some(patch) = self.undo.pop_back() else {
            return Ok(None);
        };
        match patch.revert(mesh) {
            Ok(changes) => {
                self.redo.push(patch);
                Ok(Some(changes))
            }
            Err(e) => {
                self.undo.push_back(patch);
                Err(e)
            }
        }
    }

    /// Redoes the latest undone patch. Returns `None` when there is nothing to redo.
    pub fn redo(&mut self, mesh: &mut Mesh) -> Result<Option<ChangeSet>> {
        let Some(patch) = self.redo.pop() else {
            return Ok(None);
        };
        match patch.apply(mesh) {
            Ok(changes) => {
                self.undo.push_back(patch);
                Ok(Some(changes))
            }
            Err(e) => {
                self.redo.push(patch);
                Err(e)
            }
        }
    }

    /// Drops all patches and releases the slots they reserved.
    pub fn clear(&mut self, mesh: &mut Mesh) {
        for p in self.undo.drain(..) {
            mesh.release_vertices(&p.changes.vertices_removed);
            mesh.release_triangles(&p.changes.triangles_removed);
        }
        for p in self.redo.drain(..) {
            mesh.release_vertices(&p.changes.vertices_added);
            mesh.release_triangles(&p.changes.triangles_added);
        }
        self.bytes = 0;
    }

    /// Clears the history and compacts the mesh (see [`Mesh::compact`]).
    pub fn compact(&mut self, mesh: &mut Mesh) -> Result<Remap> {
        self.undo.clear();
        self.redo.clear();
        self.bytes = 0;
        mesh.compact()
    }

    /// Changes the budget, evicting old patches if needed.
    pub fn set_budget(&mut self, mesh: &mut Mesh, budget_bytes: usize) {
        self.budget = budget_bytes;
        self.evict(mesh);
    }

    fn evict(&mut self, mesh: &mut Mesh) {
        while self.bytes > self.budget {
            let Some(p) = self.undo.pop_front() else {
                break;
            };
            self.bytes -= p.byte_size();
            mesh.release_vertices(&p.changes.vertices_removed);
            mesh.release_triangles(&p.changes.triangles_removed);
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    /// Memory held by all patches, in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn budget(&self) -> usize {
        self.budget
    }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p fast-mesh-store`
Expected: all tests pass, including the 7 history tests.

- [ ] **Step 6: Commit**

```bash
git add crates/fast-mesh-store
git commit -m "feat(store): add budgeted undo/redo history with slot reservation"
```

---

### Task 6: Compaction tests

**Files:**
- Modify: `crates/fast-mesh-store/src/compact.rs` (append tests)

**Interfaces:**
- Consumes: `Mesh::compact`, `Remap`, `History::compact` (Task 5).
- Produces: nothing new; locks in compaction behavior.

- [ ] **Step 1: Write the tests**

Append to `src/compact.rs`:

```rust
#[cfg(test)]
mod tests {
    use crate::{ChannelDesc, History, Mesh, StoreError};

    #[test]
    fn compact_renumbers_alive_elements_in_order() {
        let mut m = Mesh::from_buffers(
            &[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap();
        m.add_channel(ChannelDesc::source("mask", 1)).unwrap();
        let mut tx = m.begin_edit();
        tx.write_channel("mask", &[0, 1, 2, 3], &[0., 1., 2., 3.])
            .unwrap();
        tx.remove_triangles(&[0]).unwrap();
        tx.remove_vertices(&[1]).unwrap();
        let patch = tx.commit().unwrap();

        let remap = m.compact().unwrap();
        assert_eq!(remap.vertices, vec![0, u32::MAX, 1, 2]);
        assert_eq!(remap.triangle(1), Some(0));
        assert_eq!(remap.triangle(0), None);
        assert_eq!((m.vertex_slots(), m.triangle_slots()), (3, 1));
        assert_eq!(m.triangle(0), Some([0, 1, 2]));
        assert_eq!(m.position(1), Some([1., 1., 0.]));
        assert_eq!(m.channel("mask").unwrap(), &[0., 2., 3.]);
        assert_eq!(m.corner_count(0), 1);
        assert_eq!(patch.revert(&mut m).err(), Some(StoreError::StalePatch));
    }

    #[test]
    fn history_compact_clears_the_stacks() {
        let mut m = Mesh::from_buffers(&[0., 0., 0.], &[]).unwrap();
        let mut h = History::new(1 << 20);
        let mut tx = m.begin_edit();
        tx.remove_vertices(&[0]).unwrap();
        let p = tx.commit().unwrap();
        h.push(&mut m, p).unwrap();
        h.compact(&mut m).unwrap();
        assert!(!h.can_undo());
        assert_eq!(m.vertex_slots(), 0);
        assert!(m.free_vertex_slots().is_empty());
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p fast-mesh-store compact`
Expected: `compact_renumbers_alive_elements_in_order` and `history_compact_clears_the_stacks` pass. If either fails, fix `Mesh::compact` before continuing; the expected values in the tests are correct.

- [ ] **Step 3: Commit**

```bash
git add crates/fast-mesh-store
git commit -m "test(store): cover mesh compaction and history reset"
```

---

### Task 7: Corner-table adjacency

**Files:**
- Create: `crates/fast-mesh-store/src/adjacency.rs`
- Modify: `crates/fast-mesh-store/src/mesh.rs` (adjacency cache field and invalidation)
- Modify: `crates/fast-mesh-store/src/lib.rs`

**Interfaces:**
- Consumes: `Mesh` read API, `try_vec`, `Mesh::on_topology_changed` (called on every topology change by commit, apply, revert and compact).
- Produces: `pub enum EdgeKind { Interior, Boundary, NonManifold }`; `pub struct Adjacency` with `opposite(corner) -> Option<u32>`, `edge_kind(corner) -> EdgeKind`, `vertex_corner(v) -> Option<u32>`, `is_non_manifold_vertex(v) -> bool`, `triangles_around(&Mesh, v) -> Vec<u32>`; `Mesh::ensure_adjacency(&mut self) -> Result<()>` and `Mesh::adjacency(&self) -> Option<&Adjacency>`.

- [ ] **Step 1: Write the failing tests**

Create `src/adjacency.rs` with:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Two triangles sharing edge 0-2: 0-1-2 and 0-2-3.
    fn quad() -> Mesh {
        Mesh::from_buffers(
            &[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap()
    }

    #[test]
    fn finds_the_shared_edge() {
        let mut m = quad();
        m.ensure_adjacency().unwrap();
        let a = m.adjacency().unwrap();
        // Corner 1 (vertex 1 of triangle 0) faces edge 2-0, shared with triangle 1,
        // whose corner facing that edge is vertex 3 (corner 5).
        assert_eq!(a.opposite(1), Some(5));
        assert_eq!(a.opposite(5), Some(1));
        assert_eq!(a.edge_kind(0), EdgeKind::Boundary);
        assert_eq!(a.edge_kind(1), EdgeKind::Interior);
        assert!(!a.is_non_manifold_vertex(0));
        let mut around = a.triangles_around(&m, 0);
        around.sort();
        assert_eq!(around, vec![0, 1]);
    }

    #[test]
    fn three_triangles_on_one_edge_are_non_manifold() {
        let mut m = Mesh::from_buffers(
            &[0., 0., 0., 1., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 1.],
            &[0, 1, 2, 0, 1, 3, 0, 1, 4],
        )
        .unwrap();
        m.ensure_adjacency().unwrap();
        let a = m.adjacency().unwrap();
        assert_eq!(a.edge_kind(2), EdgeKind::NonManifold);
        assert_eq!(a.opposite(2), None);
        assert!(a.is_non_manifold_vertex(0));
        assert!(!a.is_non_manifold_vertex(2));
    }

    #[test]
    fn bowtie_vertex_is_non_manifold() {
        // Two triangles touching only at vertex 0.
        let mut m = Mesh::from_buffers(
            &[
                0., 0., 0., 1., 0., 0., 1., 1., 0., -1., 0., 0., -1., -1., 0.,
            ],
            &[0, 1, 2, 0, 3, 4],
        )
        .unwrap();
        m.ensure_adjacency().unwrap();
        let a = m.adjacency().unwrap();
        assert!(a.is_non_manifold_vertex(0));
        assert_eq!(
            a.triangles_around(&m, 0).len(),
            1,
            "only one fan is reachable"
        );
    }

    #[test]
    fn inconsistent_orientation_still_connects() {
        // Same quad, second triangle wound the other way.
        let mut m = Mesh::from_buffers(
            &[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
            &[0, 1, 2, 0, 3, 2],
        )
        .unwrap();
        m.ensure_adjacency().unwrap();
        assert_eq!(m.adjacency().unwrap().opposite(1), Some(4));
    }

    #[test]
    fn topology_changes_drop_the_cache_but_moves_keep_it() {
        let mut m = quad();
        m.ensure_adjacency().unwrap();
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[0., 0., 1.]]).unwrap();
        tx.commit().unwrap();
        assert!(m.adjacency().is_some());

        let mut tx = m.begin_edit();
        tx.remove_triangles(&[1]).unwrap();
        let patch = tx.commit().unwrap();
        assert!(m.adjacency().is_none());
        m.ensure_adjacency().unwrap();
        assert_eq!(m.adjacency().unwrap().edge_kind(1), EdgeKind::Boundary);

        patch.revert(&mut m).unwrap();
        assert!(m.adjacency().is_none());
    }
}
```

Replace `src/lib.rs` with:

```rust
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
mod mesh;
mod patch;
mod transaction;

pub use adjacency::{Adjacency, EdgeKind};
pub use channel::{ChannelDesc, ChannelKind, TopologyPolicy};
pub use compact::Remap;
pub use error::StoreError;
pub use history::History;
pub use mesh::Mesh;
pub use patch::{ChangeSet, Patch};
pub use transaction::Transaction;

/// Id of a vertex slot.
pub type VertexId = u32;
/// Id of a triangle slot.
pub type TriangleId = u32;

pub type Result<T> = std::result::Result<T, StoreError>;
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p fast-mesh-store adjacency`
Expected: compile errors, unresolved import `adjacency::Adjacency` and `no method named ensure_adjacency`.

- [ ] **Step 3: Add the cache to `Mesh`**

In `src/mesh.rs`:

1. Add the import at the top, before `use crate::channel::...`:

```rust
use crate::adjacency::Adjacency;
```

2. Add the last field of `pub struct Mesh`, after `pub(crate) live_triangles: usize,`:

```rust
    /// Cached corner table; dropped on every topology change.
    pub(crate) adjacency: Option<Adjacency>,
```

3. In `Mesh::from_parts`, add after `live_triangles: t,`:

```rust
            adjacency: None,
```

4. Replace the body of `on_topology_changed`:

```rust
    pub(crate) fn on_topology_changed(&mut self) {
        self.adjacency = None;
    }
```

- [ ] **Step 4: Implement the adjacency table**

Insert above the test module in `src/adjacency.rs`:

```rust
//! Corner-table adjacency.
//!
//! Corner `c = 3 * t + i` is the `i`-th corner of triangle `t`. The edge opposite a
//! corner joins the triangle's other two vertices. For every corner the table stores
//! the corner across that edge in the neighboring triangle, or marks the edge as a
//! boundary (no neighbor) or non-manifold (more than one neighbor). Orientation is not
//! assumed to be consistent.

use crate::column::try_vec;
use crate::{Mesh, Result, TriangleId, VertexId};

const NONE: u32 = u32::MAX;
const NON_MANIFOLD: u32 = u32::MAX - 1;

/// Kind of the edge opposite a corner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// Shared by exactly two triangles.
    Interior,
    /// Used by one triangle only.
    Boundary,
    /// Shared by three or more triangles.
    NonManifold,
}

pub struct Adjacency {
    opposite: Vec<u32>,
    vertex_corner: Vec<u32>,
    non_manifold_vertices: Vec<u64>,
}

impl Adjacency {
    pub(crate) fn build(mesh: &Mesh) -> Result<Self> {
        let tris = mesh.triangles();
        let alive = |t: usize| mesh.is_triangle_alive(t as TriangleId);
        let vslots = mesh.vertex_slots();

        // Incident corners per vertex, in compressed rows.
        let mut offsets = try_vec(vslots + 1, 0u32)?;
        for (t, tri) in tris.iter().enumerate() {
            if alive(t) {
                for &v in tri {
                    offsets[v as usize + 1] += 1;
                }
            }
        }
        for v in 0..vslots {
            offsets[v + 1] += offsets[v];
        }
        let mut incident = try_vec(offsets[vslots] as usize, 0u32)?;
        let mut cursor = offsets.clone();
        for (t, tri) in tris.iter().enumerate() {
            if alive(t) {
                for (i, &v) in tri.iter().enumerate() {
                    incident[cursor[v as usize] as usize] = (3 * t + i) as u32;
                    cursor[v as usize] += 1;
                }
            }
        }
        let corners_of =
            |v: u32| &incident[offsets[v as usize] as usize..offsets[v as usize + 1] as usize];

        let mut opposite = try_vec(3 * tris.len(), NONE)?;
        for (t, tri) in tris.iter().enumerate() {
            if !alive(t) {
                continue;
            }
            for i in 0..3 {
                let (u, w) = (tri[(i + 1) % 3], tri[(i + 2) % 3]);
                if u == w {
                    continue;
                }
                let (mut count, mut found, mut last) = (0, NONE, NONE);
                for &d in corners_of(u) {
                    let s = d / 3;
                    if s as usize == t || s == last {
                        continue;
                    }
                    last = s;
                    let other = tris[s as usize];
                    if !other.contains(&w) {
                        continue;
                    }
                    count += 1;
                    found = (0..3)
                        .find(|&j| other[j] != u && other[j] != w)
                        .map_or(NON_MANIFOLD, |j| 3 * s + j as u32);
                }
                opposite[3 * t + i] = match count {
                    0 => NONE,
                    1 => found,
                    _ => NON_MANIFOLD,
                };
            }
        }

        let mut vertex_corner = try_vec(vslots, NONE)?;
        let mut non_manifold_vertices = try_vec(vslots.div_ceil(64), 0u64)?;
        for v in 0..vslots as u32 {
            let corners = corners_of(v);
            if let Some(&c) = corners.first() {
                vertex_corner[v as usize] = c;
            }
            if fan_count(corners, &opposite) != 1 && !corners.is_empty() {
                non_manifold_vertices[v as usize / 64] |= 1 << (v % 64);
            }
        }
        Ok(Self {
            opposite,
            vertex_corner,
            non_manifold_vertices,
        })
    }

    /// The corner across the edge opposite `corner`, if that edge is interior.
    pub fn opposite(&self, corner: u32) -> Option<u32> {
        self.opposite
            .get(corner as usize)
            .copied()
            .filter(|&o| o < NON_MANIFOLD)
    }

    /// Kind of the edge opposite `corner`. Corners of dead triangles read as boundary.
    pub fn edge_kind(&self, corner: u32) -> EdgeKind {
        match self.opposite.get(corner as usize).copied().unwrap_or(NONE) {
            NONE => EdgeKind::Boundary,
            NON_MANIFOLD => EdgeKind::NonManifold,
            _ => EdgeKind::Interior,
        }
    }

    /// One corner at vertex `v`, or `None` if no triangle uses it.
    pub fn vertex_corner(&self, v: VertexId) -> Option<u32> {
        self.vertex_corner
            .get(v as usize)
            .copied()
            .filter(|&c| c != NONE)
    }

    /// True when the triangles around `v` do not form a single fan: two surfaces meet
    /// only at `v` (a "bowtie"), or an edge at `v` is non-manifold.
    pub fn is_non_manifold_vertex(&self, v: VertexId) -> bool {
        self.non_manifold_vertices
            .get(v as usize / 64)
            .is_some_and(|w| (w >> (v % 64)) & 1 == 1)
    }

    /// Triangles in the fan of `v` reached from [`Adjacency::vertex_corner`] through
    /// interior edges. For a non-manifold vertex this is one fan, not every triangle.
    pub fn triangles_around(&self, mesh: &Mesh, v: VertexId) -> Vec<TriangleId> {
        let Some(start) = self.vertex_corner(v) else {
            return Vec::new();
        };
        let tris = mesh.triangles();
        let mut fan = vec![start / 3];
        let mut stack = vec![start];
        while let Some(c) = stack.pop() {
            let (t, i) = (c / 3, c % 3);
            for e in [(i + 1) % 3, (i + 2) % 3] {
                let Some(o) = self.opposite(3 * t + e) else {
                    continue;
                };
                let s = o / 3;
                if fan.contains(&s) {
                    continue;
                }
                if let Some(j) = tris[s as usize].iter().position(|&x| x == v) {
                    fan.push(s);
                    stack.push(3 * s + j as u32);
                }
            }
        }
        fan
    }
}

/// Number of fans the corners at one vertex form when joined across interior edges.
/// A non-manifold edge at the vertex makes the result 0 (never "one fan").
fn fan_count(corners: &[u32], opposite: &[u32]) -> usize {
    let mut parent: Vec<usize> = (0..corners.len()).collect();
    fn root(parent: &mut [usize], mut a: usize) -> usize {
        while parent[a] != a {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        a
    }
    for (a, &c) in corners.iter().enumerate() {
        let (t, i) = (c / 3, c % 3);
        for e in [(i + 1) % 3, (i + 2) % 3] {
            match opposite[(3 * t + e) as usize] {
                NON_MANIFOLD => return 0,
                NONE => {}
                o => {
                    if let Some(b) = corners.iter().position(|&x| x / 3 == o / 3) {
                        let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                        parent[ra] = rb;
                    }
                }
            }
        }
    }
    (0..corners.len())
        .filter(|&a| root(&mut parent, a) == a)
        .count()
}

impl Mesh {
    /// Builds the adjacency table if it is not cached. It is cached until the next
    /// topology change (commit, undo or redo that adds, removes or rewrites triangles).
    pub fn ensure_adjacency(&mut self) -> Result<()> {
        if self.adjacency.is_none() {
            self.adjacency = Some(Adjacency::build(self)?);
        }
        Ok(())
    }

    /// The cached adjacency table, if [`Mesh::ensure_adjacency`] built it and no
    /// topology change has happened since.
    pub fn adjacency(&self) -> Option<&Adjacency> {
        self.adjacency.as_ref()
    }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p fast-mesh-store`
Expected: all tests pass, including the 5 adjacency tests.

- [ ] **Step 6: Commit**

```bash
git add crates/fast-mesh-store
git commit -m "feat(store): add lazily built corner-table adjacency"
```

---

### Task 8: Property tests, spec update and full verification

**Files:**
- Create: `crates/fast-mesh-store/tests/properties.rs`
- Modify: `docs/superpowers/specs/2026-10-14-fast-mesh-engine-design.md` (section 4)
- Modify: `CONTRIBUTING.md` (project layout)

**Interfaces:**
- Consumes: the whole public API of the crate.
- Produces: no new API.

- [ ] **Step 1: Write the property tests**

Create `crates/fast-mesh-store/tests/properties.rs`:

```rust
//! Property tests: random edit sequences must keep the store consistent, and
//! undo/redo/rollback must restore earlier states exactly.

use fast_mesh_store::{ChannelDesc, History, Mesh, TopologyPolicy, Transaction};
use proptest::prelude::*;

#[derive(Debug, Clone)]
enum Op {
    Move(u32, [i8; 3]),
    AddVertex(u32, u32, u8),
    AddTriangle([u32; 3]),
    RemoveTriangle(u32),
    RemoveVertex(u32),
    SetTriangle(u32, [u32; 3]),
    Paint(u32, i8),
}

#[derive(Debug, Clone)]
enum Step {
    Commit(Vec<Op>),
    Rollback(Vec<Op>),
    Undo,
    Redo,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        3 => (any::<u32>(), any::<[i8; 3]>()).prop_map(|(v, d)| Op::Move(v, d)),
        2 => (any::<u32>(), any::<u32>(), any::<u8>()).prop_map(|(a, b, w)| Op::AddVertex(a, b, w)),
        2 => any::<[u32; 3]>().prop_map(Op::AddTriangle),
        2 => any::<u32>().prop_map(Op::RemoveTriangle),
        1 => any::<u32>().prop_map(Op::RemoveVertex),
        1 => (any::<u32>(), any::<[u32; 3]>()).prop_map(|(t, v)| Op::SetTriangle(t, v)),
        2 => (any::<u32>(), any::<i8>()).prop_map(|(v, x)| Op::Paint(v, x)),
    ]
}

fn ops() -> impl Strategy<Value = Vec<Op>> {
    prop::collection::vec(op(), 1..6)
}

fn step() -> impl Strategy<Value = Step> {
    prop_oneof![
        4 => ops().prop_map(Step::Commit),
        1 => ops().prop_map(Step::Rollback),
        2 => Just(Step::Undo),
        2 => Just(Step::Redo),
    ]
}

/// An n×n grid of quads with two source channels and one derived channel.
/// `grid(8)` has 81 vertices, so edits cross a 64-bit liveness word boundary.
fn grid(n: u32) -> Mesh {
    let mut positions = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            positions.extend([x as f32, y as f32, 0.0]);
        }
    }
    let mut indices = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            let (b, c, d) = (a + 1, a + n + 2, a + n + 1);
            indices.extend([a, b, c, a, c, d]);
        }
    }
    let mut m = Mesh::from_buffers(&positions, &indices).unwrap();
    m.add_channel(ChannelDesc::source("mask", 1)).unwrap();
    m.add_channel(ChannelDesc::source("label", 2).with_policy(TopologyPolicy::Nearest))
        .unwrap();
    m.add_channel(ChannelDesc::derived("dist", 1)).unwrap();
    m
}

fn alive_vertices(m: &Mesh) -> Vec<u32> {
    (0..m.vertex_slots() as u32)
        .filter(|&v| m.is_vertex_alive(v))
        .collect()
}

fn alive_triangles(m: &Mesh) -> Vec<u32> {
    (0..m.triangle_slots() as u32)
        .filter(|&t| m.is_triangle_alive(t))
        .collect()
}

fn pick(ids: &[u32], seed: u32) -> Option<u32> {
    (!ids.is_empty()).then(|| ids[seed as usize % ids.len()])
}

/// Applies one random op. Validation errors are expected and ignored; they must not
/// change anything.
fn run(tx: &mut Transaction<'_>, op: &Op) {
    let vs = alive_vertices(tx.mesh());
    let ts = alive_triangles(tx.mesh());
    let v = |s: u32| pick(&vs, s);
    match *op {
        Op::Move(s, d) => {
            if let Some(v) = v(s) {
                let p = tx.mesh().position(v).unwrap();
                let q = [0, 1, 2].map(|k| p[k] + d[k] as f32 * 0.25);
                tx.set_positions(&[v], &[q]).unwrap();
            }
        }
        Op::AddVertex(a, b, w) => {
            if let (Some(a), Some(b)) = (v(a), v(b)) {
                let w = w as f32 / 255.0;
                tx.add_vertex_with_parents([0.5; 3], &[(a, w), (b, 1.0 - w)])
                    .unwrap();
            }
        }
        Op::AddTriangle(s) => {
            if let (Some(a), Some(b), Some(c)) = (v(s[0]), v(s[1]), v(s[2])) {
                tx.add_triangles(&[[a, b, c]]).unwrap();
            }
        }
        Op::RemoveTriangle(s) => {
            if let Some(t) = pick(&ts, s) {
                tx.remove_triangles(&[t]).unwrap();
            }
        }
        Op::RemoveVertex(s) => {
            if let Some(v) = v(s) {
                let _ = tx.remove_vertices(&[v]); // fails while in use
            }
        }
        Op::SetTriangle(t, s) => {
            if let (Some(t), Some(a), Some(b), Some(c)) = (pick(&ts, t), v(s[0]), v(s[1]), v(s[2]))
            {
                tx.set_triangles(&[t], &[[a, b, c]]).unwrap();
            }
        }
        Op::Paint(s, x) => {
            if let Some(v) = v(s) {
                tx.write_channel("mask", &[v], &[x as f32]).unwrap();
                tx.write_channel("label", &[v], &[x as f32, -(x as f32)])
                    .unwrap();
            }
        }
    }
}

/// Everything observable about alive elements, compared bit for bit.
#[derive(Debug, PartialEq)]
struct State {
    vertices: Vec<(u32, [u32; 3], Vec<u32>)>,
    triangles: Vec<(u32, [u32; 3])>,
    counts: (usize, usize),
}

fn state(m: &Mesh) -> State {
    let channel = |name: &str, v: u32, w: usize| -> Vec<u32> {
        m.channel(name).unwrap()[v as usize * w..(v as usize + 1) * w]
            .iter()
            .map(|x| x.to_bits())
            .collect()
    };
    State {
        vertices: alive_vertices(m)
            .into_iter()
            .map(|v| {
                let mut values = channel("mask", v, 1);
                values.extend(channel("label", v, 2));
                (v, m.position(v).unwrap().map(f32::to_bits), values)
            })
            .collect(),
        triangles: alive_triangles(m)
            .into_iter()
            .map(|t| (t, m.triangle(t).unwrap()))
            .collect(),
        counts: (m.vertex_count(), m.triangle_count()),
    }
}

fn check_invariants(m: &Mesh) {
    let vs = alive_vertices(m);
    let ts = alive_triangles(m);
    assert_eq!(m.vertex_count(), vs.len());
    assert_eq!(m.triangle_count(), ts.len());
    let mut corners = vec![0u32; m.vertex_slots()];
    for &t in &ts {
        for v in m.triangle(t).unwrap() {
            assert!(m.is_vertex_alive(v), "triangle {t} uses dead vertex {v}");
            corners[v as usize] += 1;
        }
    }
    for v in 0..m.vertex_slots() as u32 {
        let expected = if m.is_vertex_alive(v) {
            corners[v as usize]
        } else {
            0
        };
        assert_eq!(m.corner_count(v), expected, "corner count of vertex {v}");
    }
    for t in 0..m.triangle_slots() as u32 {
        if !m.is_triangle_alive(t) {
            assert_eq!(m.triangles()[t as usize], [0, 0, 0]);
        }
    }
    let mut free_v = m.free_vertex_slots().to_vec();
    free_v.sort_unstable();
    assert!(
        free_v.windows(2).all(|w| w[0] != w[1]),
        "duplicate free vertex slot"
    );
    assert!(
        free_v.iter().all(|&v| !m.is_vertex_alive(v)),
        "alive vertex on free list"
    );
    let mut free_t = m.free_triangle_slots().to_vec();
    free_t.sort_unstable();
    assert!(
        free_t.windows(2).all(|w| w[0] != w[1]),
        "duplicate free triangle slot"
    );
    assert!(
        free_t.iter().all(|&t| !m.is_triangle_alive(t)),
        "alive triangle on free list"
    );
    for name in ["mask", "dist"] {
        assert_eq!(m.channel(name).unwrap().len(), m.vertex_slots());
    }
    assert_eq!(m.channel("label").unwrap().len(), m.vertex_slots() * 2);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn undo_all_and_redo_all_restore_exact_states(commits in prop::collection::vec(ops(), 1..12)) {
        let mut m = grid(8);
        let mut h = History::new(usize::MAX);
        let initial = state(&m);
        for ops in &commits {
            let mut tx = m.begin_edit();
            for op in ops {
                run(&mut tx, op);
            }
            let patch = tx.commit().unwrap();
            h.push(&mut m, patch).unwrap();
            check_invariants(&m);
        }
        let last = state(&m);
        while h.undo(&mut m).unwrap().is_some() {
            check_invariants(&m);
        }
        prop_assert_eq!(state(&m), initial);
        while h.redo(&mut m).unwrap().is_some() {
            check_invariants(&m);
        }
        prop_assert_eq!(state(&m), last);
    }

    #[test]
    fn interleaved_steps_match_a_model(steps in prop::collection::vec(step(), 1..30), budget in 200usize..4000) {
        let mut m = grid(8);
        let mut h = History::new(budget);
        // States before each patch on the undo stack, and after each patch on the redo stack.
        let mut undo_states: Vec<State> = Vec::new();
        let mut redo_states: Vec<State> = Vec::new();
        for step in &steps {
            match step {
                Step::Commit(ops) => {
                    let before = state(&m);
                    let mut tx = m.begin_edit();
                    for op in ops {
                        run(&mut tx, op);
                    }
                    let patch = tx.commit().unwrap();
                    let empty = patch.is_empty();
                    h.push(&mut m, patch).unwrap();
                    if !empty {
                        undo_states.push(before);
                        redo_states.clear();
                    }
                    let evicted = undo_states.len() - h.undo_len();
                    undo_states.drain(..evicted);
                    prop_assert!(h.bytes() <= h.budget());
                }
                Step::Rollback(ops) => {
                    let before = state(&m);
                    let revision = m.revision();
                    let mut tx = m.begin_edit();
                    for op in ops {
                        run(&mut tx, op);
                    }
                    tx.rollback();
                    prop_assert_eq!(state(&m), before);
                    prop_assert_eq!(m.revision(), revision);
                }
                Step::Undo => {
                    let current = state(&m);
                    if h.undo(&mut m).unwrap().is_some() {
                        prop_assert_eq!(&state(&m), undo_states.last().unwrap());
                        undo_states.pop();
                        redo_states.push(current);
                    } else {
                        prop_assert!(undo_states.is_empty());
                    }
                }
                Step::Redo => {
                    let current = state(&m);
                    if h.redo(&mut m).unwrap().is_some() {
                        prop_assert_eq!(&state(&m), redo_states.last().unwrap());
                        redo_states.pop();
                        undo_states.push(current);
                    } else {
                        prop_assert!(redo_states.is_empty());
                    }
                }
            }
            check_invariants(&m);
        }
    }

    #[test]
    fn compact_preserves_alive_geometry(commits in prop::collection::vec(ops(), 1..8)) {
        let mut m = grid(8);
        let mut h = History::new(usize::MAX);
        for ops in &commits {
            let mut tx = m.begin_edit();
            for op in ops {
                run(&mut tx, op);
            }
            let patch = tx.commit().unwrap();
            h.push(&mut m, patch).unwrap();
        }
        let before = state(&m);
        let remap = h.compact(&mut m).unwrap();
        check_invariants(&m);
        prop_assert!(m.free_vertex_slots().is_empty());
        prop_assert_eq!(m.vertex_slots(), m.vertex_count());
        prop_assert_eq!(m.triangle_slots(), m.triangle_count());
        for (v, pos, values) in &before.vertices {
            let nv = remap.vertex(*v).unwrap();
            prop_assert_eq!(m.position(nv).unwrap().map(f32::to_bits), *pos);
            prop_assert_eq!(m.channel("mask").unwrap()[nv as usize].to_bits(), values[0]);
        }
        for (t, tri) in &before.triangles {
            let nt = remap.triangle(*t).unwrap();
            prop_assert_eq!(m.triangle(nt).unwrap(), tri.map(|v| remap.vertex(v).unwrap()));
        }
    }
}
```

- [ ] **Step 2: Run them**

Run: `cargo test -p fast-mesh-store --test properties`
Expected: 3 tests pass. Then run a longer pass:

Run: `PROPTEST_CASES=3000 cargo test -p fast-mesh-store --release --test properties`
Expected: 3 tests pass. If proptest reports a failure, it prints a minimal failing input; fix the store, not the test.

- [ ] **Step 3: Record the deviations in the spec**

In `docs/superpowers/specs/2026-10-14-fast-mesh-engine-design.md`:

1. In 4.1, replace the bullet starting `- **Adjacency is built lazily**` with:

```markdown
- **Adjacency is built lazily**, the first time an algorithm or field needs neighbors, and cached until the next topology change. (Phase 1 rebuilds it after topology changes; incremental maintenance is added with the topology operations that need it inside a transaction.) A user who only moves vertices never pays for it.
```

2. In 4.3, replace `` `interpolate` (weighted by creation parents), `default`, or `nearest`. `` with:

```markdown
`interpolate` (weighted by creation parents), `default`, or `nearest` (the creation parent with the largest weight; a spatial nearest-vertex lookup arrives with the Scene).
```

3. In 4.4, replace the bullet starting `- **Snapshot fallback:**` with:

```markdown
- **Snapshot fallback:** for each column the patch stores either per-slot entries or full copies, whichever is smaller in bytes. Large rewrites (e.g. remeshing) therefore never cost more than two copies of the column.
```

4. In 4.5, replace the bullet starting `- Per-mesh history or scene-wide history` with:

```markdown
- Per-mesh history in the store. Scene-wide history (one undo step can span several meshes) is built on top of it in the Scene phase.
```

- [ ] **Step 4: Update the contributor guide**

In `CONTRIBUTING.md`, in the "Project layout" code block, add this line after the `crates/fast-mesh-wasm/` line:

```
crates/fast-mesh-store/  Editable mesh storage: slots, channels, transactions, undo/redo history.
```

- [ ] **Step 5: Run the full verification**

Run: `cargo fmt --all --check`
Expected: no output.

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: `Finished` with no warnings.

Run: `cargo test --workspace`
Expected: all tests pass (existing `fast-mesh` tests plus `fast-mesh-store`).

Run: `cargo build -p fast-mesh-store --target wasm32-unknown-unknown --release`
Expected: `Finished`. This confirms the crate builds for WebAssembly.

Run: `cargo doc -p fast-mesh-store --no-deps`
Expected: no warnings.

- [ ] **Step 6: Commit**

```bash
git add crates/fast-mesh-store docs/superpowers/specs CONTRIBUTING.md
git commit -m "test(store): add property tests and record phase-1 design decisions"
```
