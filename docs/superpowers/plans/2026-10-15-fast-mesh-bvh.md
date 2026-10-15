# fast-mesh-bvh (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a vertex → triangles incidence index to `fast-mesh-store`, and a new crate `fast-mesh-bvh` that builds a BVH over a mesh, keeps it current from the store's `ChangeSet` (refit, treelet rebuild or full rebuild), answers box queries, and ships with benchmarks.

**Architecture:** Nodes live in an arena with parent links, and every traversal walks those links, so traversals never allocate. Subtrees are built top-down (binned SAH or Morton/LBVH) over an index permutation. The tree is cut into treelets of at most `treelet_size` triangles: topology edits rebuild only the affected treelets, appending new nodes and leaving old ones as garbage that is compacted later. Every fallible allocation happens before the tree is modified.

**Tech Stack:** Rust 1.99 (edition 2024, MSRV 1.88), `fast-mesh-store` (path dependency), `proptest` and `criterion` (dev only).

**Spec:** `docs/superpowers/specs/2026-10-15-fast-mesh-bvh-design.md` (parent: `docs/superpowers/specs/2026-10-14-fast-mesh-engine-design.md` §5).

## Global Constraints

- New crate `crates/fast-mesh-bvh`, package `fast-mesh-bvh`, workspace-inherited metadata. Runtime dependency: `fast-mesh-store` only.
- Must compile for `wasm32-unknown-unknown` (no threads, no OS APIs).
- Boxes are `f32`, in mesh-local space. The tree holds every alive triangle exactly once and never references dead slots.
- Node layout: 32 bytes (`Aabb` + two `u32`). Leaf: `a` = first ref, `b` = `LEAF_BIT | count`.
- Large allocations are fallible (`BvhError::OutOfMemory`). `update` leaves the BVH exactly as before on error. The two optional follow-up steps (degradation rebuild, storage compaction) are skipped silently if memory runs out.
- Builds are deterministic.
- Defaults: `max_leaf_size` 8, `sah_bins` 16, `treelet_size` 4,096, `full_rebuild_fraction` 0.30, `degradation_factor` 1.5. Compaction runs when garbage exceeds 50% of the live nodes or refs.
- `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` must pass at the end (Task 7). Until then, unused-item warnings from not-yet-used internals are expected; only `cargo test` must pass per task.
- Commit messages end with a `Co-Authored-By: Claude <model> <noreply@anthropic.com>` trailer naming the model that wrote the commit.

### Deviations from the Phase 2 spec (recorded in the spec in Task 7)

1. **`treelet_size` defaults to 4,096, not 65,536.** Benchmarks at 1M triangles: replacing 200 triangles took 29.6 ms with 65,536, 17.6 ms with 16,384, and 8.9 ms with 4,096. Vertex-move updates were unchanged at about 2 ms.
2. **`update` takes `&mut Mesh`.** It may build the mesh's incidence index, which is a cache stored on the mesh.
3. **Manual `refit` and `rebuild_region` do not change the recorded revision.** `update`, `build` and `rebuild` do.
4. **Full builds allocate nodes on demand.** They start from an estimate of about 3 × triangles / leaf size instead of the worst case 2n − 1. Treelet rebuilds reserve the worst case for their own size before mutating, with at least 1/8 headroom. Measured memory: 21.6 MB for 1M triangles.

## Review Focus

- A vertex at a non-manifold (bowtie) junction moves: every triangle around it, in every fan, must be refit (Task 1: `lists_every_triangle_of_every_fan`; Task 6 property test).
- Memory runs out in the middle of `update`: the tree must be bit-for-bit what it was before the call (Task 6: `failed_partial_updates_leave_the_tree_unchanged`, `failed_full_rebuilds_leave_the_tree_unchanged`).
- Degenerate, NaN or huge (`1e30`) coordinates: building and querying must not panic and must still match brute force (Task 5: `degenerate_and_non_finite_triangles_do_not_break_the_build`).
- The inverted `ChangeSet` from undo or redo is fed to `update`: the tree must stay exact (Task 6: `undo_and_redo_change_sets_keep_the_tree_correct`).
- Every triangle is removed, then restored by undo: the tree must become empty and then refill (Task 6: `emptying_and_refilling_the_mesh`).

## File Structure

| File | Responsibility |
| --- | --- |
| `crates/fast-mesh-store/src/incidence.rs` | `Incidence` (vertex → triangles, compressed rows), `Mesh::ensure_incidence` / `Mesh::incidence` |
| `crates/fast-mesh-bvh/Cargo.toml` | Crate manifest (bench target added in Task 7) |
| `src/lib.rs` | Module wiring and re-exports |
| `src/error.rs` | `BvhError`, `Result`, fallible allocation helpers, test-only allocation-failure switch |
| `src/options.rs` | `BuildOptions`, `Quality`, `UpdatePolicy` and their validation |
| `src/aabb.rs` | `Aabb` |
| `src/arena.rs` | `Node`, `Arena`, stackless traversal, refit, subtree size, SAH cost |
| `src/build.rs` | `Prims`, binned SAH and Morton builders, reservation helpers |
| `src/tree.rs` | `Bvh`: build, treelets, `query_aabb`, inspection, compaction, `validate` |
| `src/update.rs` | `update`, `refit`, `rebuild_region`, `rebuild`, `UpdateReport`, `Strategy` |
| `tests/properties.rs` | Random edit/undo/redo sequences checked against brute force |
| `benches/bvh.rs` | Criterion benchmarks (1M; 10M with `FAST_MESH_BENCH_LARGE=1`) |
| `docs/benchmarks.md` | Recorded results and the defaults they justify |

---

### Task 1: Store incidence index

**Files:**
- Create: `crates/fast-mesh-store/src/incidence.rs`
- Modify: `crates/fast-mesh-store/src/mesh.rs`, `crates/fast-mesh-store/src/lib.rs`, `crates/fast-mesh-store/tests/properties.rs`

**Interfaces:**
- Consumes: `Mesh::{triangles, vertex_slots, is_triangle_alive}`, `crate::column::try_vec`, `Mesh::on_topology_changed` (already called on every topology change).
- Produces: `pub struct Incidence` with `triangles(&self, v: VertexId) -> &[TriangleId]` (ascending, each once, all fans); `Mesh::ensure_incidence(&mut self) -> Result<()>`; `Mesh::incidence(&self) -> Option<&Incidence>`.

- [ ] **Step 1: Write the failing tests**

Create `crates/fast-mesh-store/src/incidence.rs` with only this test module:

```rust
#[cfg(test)]
mod tests {
    use crate::Mesh;

    /// Three triangles share vertex 0; a fourth touches it only at vertex 0
    /// (a bowtie), so the vertex is non-manifold.
    fn fan_and_bowtie() -> Mesh {
        Mesh::from_buffers(
            &[
                0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., -1., 0., 0., -1., -1., 0.,
            ],
            &[0, 1, 2, 0, 2, 3, 3, 2, 1, 0, 4, 5],
        )
        .unwrap()
    }

    #[test]
    fn lists_every_triangle_of_every_fan() {
        let mut m = fan_and_bowtie();
        m.ensure_incidence().unwrap();
        let inc = m.incidence().unwrap();
        assert_eq!(inc.triangles(0), &[0, 1, 3]);
        assert_eq!(inc.triangles(2), &[0, 1, 2]);
        assert_eq!(inc.triangles(5), &[3]);
        assert!(inc.triangles(99).is_empty());
    }

    #[test]
    fn repeated_vertices_are_listed_once() {
        let mut m = Mesh::from_buffers(&[0., 0., 0., 1., 0., 0.], &[0, 0, 1]).unwrap();
        m.ensure_incidence().unwrap();
        assert_eq!(m.incidence().unwrap().triangles(0), &[0]);
        assert_eq!(m.incidence().unwrap().triangles(1), &[0]);
    }

    #[test]
    fn moves_keep_the_index_and_topology_changes_drop_it() {
        let mut m = fan_and_bowtie();
        m.ensure_incidence().unwrap();
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[0., 0., 1.]]).unwrap();
        tx.commit().unwrap();
        assert!(m.incidence().is_some());

        let mut tx = m.begin_edit();
        tx.remove_triangles(&[1]).unwrap();
        let patch = tx.commit().unwrap();
        assert!(m.incidence().is_none());
        m.ensure_incidence().unwrap();
        assert_eq!(m.incidence().unwrap().triangles(0), &[0, 3]);

        patch.revert(&mut m).unwrap();
        assert!(m.incidence().is_none());
    }
}
```

In `crates/fast-mesh-store/src/lib.rs` add `mod incidence;` after `mod history;` and `pub use incidence::Incidence;` after `pub use history::History;`.

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p fast-mesh-store incidence`
Expected: compile errors, `cannot find type Incidence` / `no method named ensure_incidence`.

- [ ] **Step 3: Implement the index**

Insert above the test module in `incidence.rs`:

```rust
//! Vertex → triangles incidence index.
//!
//! Lists, for every vertex slot, all alive triangles that use it. Unlike
//! [`crate::Adjacency::triangles_around`] it covers every fan of a non-manifold
//! vertex, which is what BVH refits need after vertices move.

use crate::column::try_vec;
use crate::{Mesh, Result, TriangleId, VertexId};

/// Every alive triangle using each vertex, stored in compressed rows.
#[derive(Debug)]
pub struct Incidence {
    /// `offsets[v]..offsets[v + 1]` is the range of `triangles` for vertex `v`.
    offsets: Vec<u32>,
    triangles: Vec<TriangleId>,
}

impl Incidence {
    pub(crate) fn build(mesh: &Mesh) -> Result<Self> {
        let tris = mesh.triangles();
        let slots = mesh.vertex_slots();
        // A triangle repeating a vertex ([a, a, b]) is listed once for it.
        let distinct = |tri: &[VertexId; 3]| {
            let [a, b, c] = *tri;
            [
                Some(a),
                (b != a).then_some(b),
                (c != a && c != b).then_some(c),
            ]
        };
        let mut offsets = try_vec(slots + 1, 0u32)?;
        for (t, tri) in tris.iter().enumerate() {
            if mesh.is_triangle_alive(t as TriangleId) {
                for v in distinct(tri).into_iter().flatten() {
                    offsets[v as usize + 1] += 1;
                }
            }
        }
        for v in 0..slots {
            offsets[v + 1] += offsets[v];
        }
        let mut triangles = try_vec(offsets[slots] as usize, 0 as TriangleId)?;
        let mut cursor = try_vec(slots, 0u32)?;
        cursor.copy_from_slice(&offsets[..slots]);
        // Triangles are visited in ascending id order, so every row comes out sorted.
        for (t, tri) in tris.iter().enumerate() {
            if mesh.is_triangle_alive(t as TriangleId) {
                for v in distinct(tri).into_iter().flatten() {
                    triangles[cursor[v as usize] as usize] = t as TriangleId;
                    cursor[v as usize] += 1;
                }
            }
        }
        Ok(Self { offsets, triangles })
    }

    /// Every alive triangle using `v`, each once, in ascending id order. Empty for
    /// unused, dead or out-of-range vertices.
    pub fn triangles(&self, v: VertexId) -> &[TriangleId] {
        let v = v as usize;
        if v + 1 >= self.offsets.len() {
            return &[];
        }
        &self.triangles[self.offsets[v] as usize..self.offsets[v + 1] as usize]
    }
}

impl Mesh {
    /// Builds the vertex → triangles index if it is not cached. It stays cached
    /// while only positions or channels change and is dropped on every topology
    /// change, like the adjacency table.
    pub fn ensure_incidence(&mut self) -> Result<()> {
        if self.incidence.is_none() {
            self.incidence = Some(Incidence::build(self)?);
        }
        Ok(())
    }

    /// The cached index, if [`Mesh::ensure_incidence`] built it and no topology
    /// change has happened since.
    pub fn incidence(&self) -> Option<&Incidence> {
        self.incidence.as_ref()
    }
}
```

In `crates/fast-mesh-store/src/mesh.rs`:

1. After `use crate::column::{Column, try_reserve, try_vec};` add `use crate::incidence::Incidence;`.
2. After the `adjacency` field of `pub struct Mesh` add:

```rust
    /// Cached vertex → triangles index; dropped on every topology change.
    pub(crate) incidence: Option<Incidence>,
```

3. In `assemble`, after `adjacency: None,` add `incidence: None,`.
4. In `on_topology_changed`, after `self.adjacency = None;` add `self.incidence = None;`.

- [ ] **Step 4: Add the property-test invariant**

In `crates/fast-mesh-store/tests/properties.rs`, add this function right before `proptest! {`:

```rust
/// The incidence index must list, per vertex, exactly the alive triangles using it.
fn check_incidence(m: &mut Mesh) {
    m.ensure_incidence().unwrap();
    let inc = m.incidence().unwrap();
    for v in 0..m.vertex_slots() as u32 {
        let expected: Vec<u32> = alive_triangles(m)
            .into_iter()
            .filter(|&t| m.triangle(t).unwrap().contains(&v))
            .collect();
        assert_eq!(inc.triangles(v), &expected[..], "incidence of vertex {v}");
    }
}
```

and in `interleaved_steps_match_a_model`, right after the `check_invariants(&m);` at the end of the loop body, add `check_incidence(&mut m);`.

- [ ] **Step 5: Run the tests**

Run: `cargo test -p fast-mesh-store`
Expected: all pass (60 unit tests including the 3 new ones, and 3 property tests).

- [ ] **Step 6: Commit**

```bash
git add crates/fast-mesh-store
git commit -m "feat(store): add a vertex to triangles incidence index"
```

---

### Task 2: BVH crate scaffold: errors, options, boxes

**Files:**
- Modify: `Cargo.toml` (workspace dependencies)
- Create: `crates/fast-mesh-bvh/Cargo.toml`, `src/lib.rs`, `src/error.rs`, `src/options.rs`, `src/aabb.rs`

**Interfaces:**
- Produces:
  - `pub enum BvhError { MeshMismatch, InvalidOptions(&'static str), OutOfMemory, Store(StoreError) }` with `From<StoreError>`, and `pub type Result<T>`.
  - crate-private `try_reserve`, `try_reserve_exact`, `try_reserve_map`, `try_vec`; test-only `fail_allocations_after(Option<usize>)`.
  - `pub enum Quality { Sah, Fast }`.
  - `pub struct BuildOptions { quality, max_leaf_size: u32, sah_bins: u32, treelet_size: u32 }` and `pub struct UpdatePolicy { full_rebuild_fraction: f32, degradation_factor: f32 }`, each with `Default` and crate-private `validate`.
  - `pub struct Aabb { min, max }` with `EMPTY`, `new`, `of_triangle`, `grow_point`, `grow`, `union`, `is_empty`, `contains`, `overlaps`, `surface_area`, `center`.

- [ ] **Step 1: Manifests**

In the root `Cargo.toml`, under `[workspace.dependencies]`, add after the `fast-mesh` line:

```toml
fast-mesh-store = { path = "crates/fast-mesh-store", version = "0.1.0" }
```

Create `crates/fast-mesh-bvh/Cargo.toml`:

```toml
[package]
name = "fast-mesh-bvh"
description = "Incrementally updatable bounding volume hierarchy for editable triangle meshes (part of fast-mesh-wasm)"
keywords = ["bvh", "mesh", "geometry", "collision", "3d"]
categories = ["graphics", "data-structures"]
readme = "../../README.md"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
authors.workspace = true

[dependencies]
fast-mesh-store.workspace = true

[dev-dependencies]
proptest = "1.12.0"
```

- [ ] **Step 2: Write the failing tests**

Create `src/error.rs`, `src/options.rs` and `src/aabb.rs`, each containing only its test module:

`src/error.rs`:

```rust
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
```

`src/options.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        assert!(BuildOptions::default().validate().is_ok());
        assert!(UpdatePolicy::default().validate().is_ok());
    }

    #[test]
    fn out_of_range_options_are_rejected() {
        let bad_leaf = BuildOptions {
            max_leaf_size: 0,
            ..Default::default()
        };
        assert!(matches!(
            bad_leaf.validate(),
            Err(BvhError::InvalidOptions(_))
        ));
        let bad_bins = BuildOptions {
            sah_bins: 65,
            ..Default::default()
        };
        assert!(bad_bins.validate().is_err());
        let bad_treelet = BuildOptions {
            treelet_size: 4,
            ..Default::default()
        };
        assert!(bad_treelet.validate().is_err());
        let nan = UpdatePolicy {
            degradation_factor: f32::NAN,
            ..Default::default()
        };
        assert!(nan.validate().is_err());
        let fraction = UpdatePolicy {
            full_rebuild_fraction: 1.5,
            ..Default::default()
        };
        assert!(fraction.validate().is_err());
    }
}
```

`src/aabb.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_box_contains_and_overlaps_nothing() {
        let unit = Aabb::new([0.0; 3], [1.0; 3]);
        assert!(Aabb::EMPTY.is_empty());
        assert!(!Aabb::EMPTY.overlaps(&unit));
        assert!(unit.contains(&Aabb::EMPTY));
        assert!(!Aabb::EMPTY.contains(&unit));
        assert_eq!(Aabb::EMPTY.surface_area(), 0.0);
    }

    #[test]
    fn touching_boxes_overlap() {
        let a = Aabb::new([0.0; 3], [1.0; 3]);
        let b = Aabb::new([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
        assert!(a.overlaps(&b));
        let c = Aabb::new([1.5, 0.0, 0.0], [2.0, 1.0, 1.0]);
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn triangle_box_and_area() {
        let positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 1.0]];
        let b = Aabb::of_triangle(&positions, [0, 1, 2]);
        assert_eq!(b, Aabb::new([0.0, 0.0, 0.0], [2.0, 3.0, 1.0]));
        assert_eq!(b.surface_area(), 2.0 * (6.0 + 3.0 + 2.0));
        assert_eq!(b.center(), [1.0, 1.5, 0.5]);
    }

    #[test]
    fn nan_coordinates_never_widen_a_box() {
        let positions = [[f32::NAN, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, f32::NAN, 2.0]];
        let b = Aabb::of_triangle(&positions, [0, 1, 2]);
        assert_eq!(b, Aabb::new([1.0, 0.0, 0.0], [2.0, 1.0, 2.0]));
        let all_nan = Aabb::of_triangle(&[[f32::NAN; 3]], [0, 0, 0]);
        assert!(all_nan.is_empty());
        assert_eq!(all_nan.center(), [0.0; 3]);
    }
}
```

`src/lib.rs`:

```rust
//! Incrementally updatable bounding volume hierarchy for meshes stored in
//! [`fast_mesh_store`].
//!
//! - [`Bvh::build`] builds over a mesh's alive triangles in mesh-local space,
//!   with a binned SAH ([`Quality::Sah`]) or Morton-code ([`Quality::Fast`]) split.
//! - [`Bvh::update`] takes the store's [`fast_mesh_store::ChangeSet`] and refits,
//!   rebuilds the affected treelets, or rebuilds everything, whichever is cheapest.
//! - [`Bvh::query_aabb`] finds the triangles whose boxes overlap a box.
//!
//! Every operation is failure-atomic: on error the BVH is left as it was.

mod aabb;
mod error;
mod options;

pub use aabb::Aabb;
pub use error::{BvhError, Result};
pub use options::{BuildOptions, Quality, UpdatePolicy};
```

- [ ] **Step 3: Run them to verify they fail**

Run: `cargo test -p fast-mesh-bvh`
Expected: compile errors (unresolved imports of `Aabb`, `BvhError`, `BuildOptions`, ...).

- [ ] **Step 4: Implement**

Insert above the test module of each file:

`src/error.rs`:

```rust
use std::fmt;

use fast_mesh_store::StoreError;

/// Errors returned by BVH operations. On any error the BVH is left exactly as it
/// was before the call.
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
```

`src/options.rs`:

```rust
use crate::{BvhError, Result};

/// How the tree is split when it is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// Binned surface-area heuristic: slower to build, faster to query.
    Sah,
    /// Morton-code (LBVH) build: faster to build, somewhat slower to query.
    Fast,
}

/// Options for building (and rebuilding parts of) a BVH.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuildOptions {
    pub quality: Quality,
    /// Most triangles per leaf, 1 to 16.
    pub max_leaf_size: u32,
    /// Bins per axis for the SAH build, 4 to 64.
    pub sah_bins: u32,
    /// Most triangles in a treelet, the unit of partial rebuilds. At least `max_leaf_size`.
    pub treelet_size: u32,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            quality: Quality::Sah,
            max_leaf_size: 8,
            sah_bins: 16,
            treelet_size: 4096,
        }
    }
}

impl BuildOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if !(1..=16).contains(&self.max_leaf_size) {
            return Err(BvhError::InvalidOptions("max_leaf_size must be 1 to 16"));
        }
        if !(4..=64).contains(&self.sah_bins) {
            return Err(BvhError::InvalidOptions("sah_bins must be 4 to 64"));
        }
        if self.treelet_size < self.max_leaf_size {
            return Err(BvhError::InvalidOptions(
                "treelet_size must be at least max_leaf_size",
            ));
        }
        Ok(())
    }
}

/// When `update` falls back to heavier strategies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpdatePolicy {
    /// Rebuild everything when more than this fraction of alive triangles is
    /// dirty (added, removed, rewritten or moved), 0 to 1.
    pub full_rebuild_fraction: f32,
    /// Rebuild a refit treelet when its SAH cost exceeds this factor times its
    /// cost when it was built. At least 1.
    pub degradation_factor: f32,
}

impl Default for UpdatePolicy {
    fn default() -> Self {
        Self {
            full_rebuild_fraction: 0.30,
            degradation_factor: 1.5,
        }
    }
}

impl UpdatePolicy {
    pub(crate) fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.full_rebuild_fraction) {
            return Err(BvhError::InvalidOptions(
                "full_rebuild_fraction must be 0 to 1",
            ));
        }
        if self.degradation_factor.is_nan() || self.degradation_factor < 1.0 {
            return Err(BvhError::InvalidOptions(
                "degradation_factor must be at least 1",
            ));
        }
        Ok(())
    }
}
```

`src/aabb.rs`:

```rust
use std::cmp::Ordering;

/// Axis-aligned bounding box in mesh-local coordinates.
///
/// [`Aabb::EMPTY`] (min = +∞, max = −∞) contains nothing and overlaps nothing.
/// Non-finite coordinates are tolerated: `f32::min`/`max` ignore NaN, so a NaN
/// coordinate never widens a box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };

    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Box of one triangle, given slot-aligned positions.
    pub fn of_triangle(positions: &[[f32; 3]], tri: [u32; 3]) -> Self {
        let mut b = Self::EMPTY;
        for v in tri {
            b.grow_point(positions[v as usize]);
        }
        b
    }

    pub fn grow_point(&mut self, p: [f32; 3]) {
        for (k, c) in p.into_iter().enumerate() {
            self.min[k] = self.min[k].min(c);
            self.max[k] = self.max[k].max(c);
        }
    }

    pub fn grow(&mut self, other: &Aabb) {
        for k in 0..3 {
            self.min[k] = self.min[k].min(other.min[k]);
            self.max[k] = self.max[k].max(other.max[k]);
        }
    }

    pub fn union(mut self, other: &Aabb) -> Aabb {
        self.grow(other);
        self
    }

    /// True when any axis has `min > max` (or a NaN bound).
    pub fn is_empty(&self) -> bool {
        (0..3).any(|k| {
            !matches!(
                self.min[k].partial_cmp(&self.max[k]),
                Some(Ordering::Less | Ordering::Equal)
            )
        })
    }

    /// True when `other` lies inside this box. An empty `other` is always contained.
    pub fn contains(&self, other: &Aabb) -> bool {
        other.is_empty()
            || (0..3).all(|k| self.min[k] <= other.min[k] && other.max[k] <= self.max[k])
    }

    /// True when the boxes share at least one point (touching counts).
    pub fn overlaps(&self, other: &Aabb) -> bool {
        !self.is_empty()
            && !other.is_empty()
            && (0..3).all(|k| self.min[k] <= other.max[k] && other.min[k] <= self.max[k])
    }

    /// Surface area; 0 for an empty box.
    pub fn surface_area(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let d = [0, 1, 2].map(|k| self.max[k] - self.min[k]);
        2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
    }

    /// Box center; components that are not finite (empty or infinite boxes) read 0.
    pub fn center(&self) -> [f32; 3] {
        [0, 1, 2].map(|k| {
            let c = (self.min[k] + self.max[k]) * 0.5;
            if c.is_finite() { c } else { 0.0 }
        })
    }
}
```

- [ ] **Step 5: Run the tests**

Run: `cargo test -p fast-mesh-bvh`
Expected: 7 tests pass.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock crates/fast-mesh-bvh
git commit -m "feat(bvh): add fast-mesh-bvh crate with errors, options and boxes"
```

---

### Task 3: Node arena and stackless traversal

**Files:**
- Create: `crates/fast-mesh-bvh/src/arena.rs`
- Modify: `crates/fast-mesh-bvh/src/lib.rs`

**Interfaces:**
- Consumes: `Aabb` (Task 2).
- Produces (crate-private):
  - `NONE`, `LEAF_BIT`;
  - `Node { bounds, a, b }` with `leaf`, `is_leaf`, `leaf_range`;
  - `Arena { nodes, parents, refs, tri_leaf }` with:
    - `traverse(root, FnMut(u32, &Node) -> bool)` (pre-order; return false to skip children);
    - `refit_leaf(leaf, positions, triangles)`;
    - `refit_ancestors(node) -> usize` (stops when a box is unchanged);
    - `subtree_size(root) -> (nodes, refs)`;
    - `sah_cost(root) -> f32`.

- [ ] **Step 1: Write the failing tests**

Create `src/arena.rs` with only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// root(0) -> [1, 2], 1 -> [3, 4]; leaves 2, 3, 4 hold triangles 0, 1, 2.
    fn small() -> Arena {
        let leaf = |t: u32, x: f32| Node::leaf(Aabb::new([x, 0.0, 0.0], [x + 1.0, 1.0, 0.0]), t, 1);
        let inner = |a, b| Node {
            bounds: Aabb::EMPTY,
            a,
            b,
        };
        Arena {
            nodes: vec![
                inner(1, 2),
                inner(3, 4),
                leaf(0, 0.0),
                leaf(1, 1.0),
                leaf(2, 2.0),
            ],
            parents: vec![NONE, 0, 0, 1, 1],
            refs: vec![0, 1, 2],
            tri_leaf: vec![2, 3, 4],
        }
    }

    #[test]
    fn traverses_in_pre_order_and_can_prune() {
        let a = small();
        let mut seen = Vec::new();
        a.traverse(0, |i, _| {
            seen.push(i);
            true
        });
        assert_eq!(seen, vec![0, 1, 3, 4, 2]);

        let mut pruned = Vec::new();
        a.traverse(0, |i, _| {
            pruned.push(i);
            i != 1
        });
        assert_eq!(pruned, vec![0, 1, 2]);

        let mut sub = Vec::new();
        a.traverse(1, |i, _| {
            sub.push(i);
            true
        });
        assert_eq!(
            sub,
            vec![1, 3, 4],
            "a subtree traversal stays inside the subtree"
        );
    }

    #[test]
    fn refits_ancestors_and_counts_subtrees() {
        let mut a = small();
        a.refit_ancestors(3);
        assert_eq!(
            a.nodes[1].bounds,
            Aabb::new([1.0, 0.0, 0.0], [3.0, 1.0, 0.0])
        );
        assert_eq!(
            a.nodes[0].bounds,
            Aabb::new([0.0, 0.0, 0.0], [3.0, 1.0, 0.0])
        );
        assert_eq!(a.subtree_size(0), (5, 3));
        assert_eq!(a.subtree_size(1), (3, 2));
    }
}
```

Replace `src/lib.rs` with:

```rust
//! Incrementally updatable bounding volume hierarchy for meshes stored in
//! [`fast_mesh_store`].
//!
//! - [`Bvh::build`] builds over a mesh's alive triangles in mesh-local space,
//!   with a binned SAH ([`Quality::Sah`]) or Morton-code ([`Quality::Fast`]) split.
//! - [`Bvh::update`] takes the store's [`fast_mesh_store::ChangeSet`] and refits,
//!   rebuilds the affected treelets, or rebuilds everything, whichever is cheapest.
//! - [`Bvh::query_aabb`] finds the triangles whose boxes overlap a box.
//!
//! Every operation is failure-atomic: on error the BVH is left as it was.

mod aabb;
mod arena;
mod error;
mod options;

pub use aabb::Aabb;
pub use error::{BvhError, Result};
pub use options::{BuildOptions, Quality, UpdatePolicy};
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p fast-mesh-bvh arena`
Expected: compile errors (`cannot find type Arena`).

- [ ] **Step 3: Implement**

Insert above the test module:

```rust
//! Node storage shared by building, updating and querying.
//!
//! Every traversal walks parent links instead of using a stack, so no traversal
//! allocates and none can fail, however deep or unbalanced the tree is.

use crate::Aabb;

pub(crate) const NONE: u32 = u32::MAX;
pub(crate) const LEAF_BIT: u32 = 1 << 31;

/// 32-byte node. Inner: `a` = left child, `b` = right child.
/// Leaf: `a` = first index into `refs`, `b` = `LEAF_BIT | count`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Node {
    pub(crate) bounds: Aabb,
    pub(crate) a: u32,
    pub(crate) b: u32,
}

const _: () = assert!(std::mem::size_of::<Node>() == 32);

impl Node {
    pub(crate) fn leaf(bounds: Aabb, first: u32, count: u32) -> Self {
        Self {
            bounds,
            a: first,
            b: LEAF_BIT | count,
        }
    }

    pub(crate) fn is_leaf(&self) -> bool {
        self.b & LEAF_BIT != 0
    }

    pub(crate) fn leaf_range(&self) -> std::ops::Range<usize> {
        let first = self.a as usize;
        first..first + (self.b & !LEAF_BIT) as usize
    }
}

/// Nodes, parent links, leaf triangle references and the triangle → leaf map.
pub(crate) struct Arena {
    pub(crate) nodes: Vec<Node>,
    pub(crate) parents: Vec<u32>,
    pub(crate) refs: Vec<u32>,
    /// Indexed by triangle slot: the leaf holding it, or `NONE`.
    pub(crate) tri_leaf: Vec<u32>,
}

impl Arena {
    /// Visits the subtree rooted at `root` in depth-first pre-order. `visit`
    /// returns whether to descend into the node's children (ignored for leaves).
    pub(crate) fn traverse(&self, root: u32, mut visit: impl FnMut(u32, &Node) -> bool) {
        if root == NONE {
            return;
        }
        let stop = self.parents[root as usize];
        let mut cur = root;
        let mut from = stop;
        loop {
            let node = &self.nodes[cur as usize];
            let next;
            if from == self.parents[cur as usize] {
                // First arrival from above.
                if visit(cur, node) && !node.is_leaf() {
                    next = node.a;
                } else {
                    next = self.parents[cur as usize];
                }
            } else if from == node.a {
                next = node.b;
            } else {
                next = self.parents[cur as usize];
            }
            if cur == root && next == stop {
                return;
            }
            from = cur;
            cur = next;
        }
    }

    /// Recomputes a leaf's box from its triangles.
    pub(crate) fn refit_leaf(&mut self, leaf: u32, positions: &[[f32; 3]], triangles: &[[u32; 3]]) {
        let range = self.nodes[leaf as usize].leaf_range();
        let mut b = Aabb::EMPTY;
        for &t in &self.refs[range] {
            b.grow(&Aabb::of_triangle(positions, triangles[t as usize]));
        }
        self.nodes[leaf as usize].bounds = b;
    }

    /// Recomputes the boxes of `node`'s ancestors, stopping early once a box
    /// does not change. Returns how many nodes were rewritten.
    pub(crate) fn refit_ancestors(&mut self, node: u32) -> usize {
        let mut written = 0;
        let mut cur = self.parents[node as usize];
        while cur != NONE {
            let n = self.nodes[cur as usize];
            let b = self.nodes[n.a as usize]
                .bounds
                .union(&self.nodes[n.b as usize].bounds);
            if b == n.bounds {
                break;
            }
            self.nodes[cur as usize].bounds = b;
            written += 1;
            cur = self.parents[cur as usize];
        }
        written
    }

    /// Number of nodes and of triangle references in a subtree.
    pub(crate) fn subtree_size(&self, root: u32) -> (usize, usize) {
        let (mut nodes, mut refs) = (0, 0);
        self.traverse(root, |_, n| {
            nodes += 1;
            if n.is_leaf() {
                refs += n.leaf_range().len();
            }
            true
        });
        (nodes, refs)
    }

    /// Surface-area-heuristic cost of a subtree, relative to its root's area:
    /// inner nodes cost their area, leaves their area times their triangle count.
    pub(crate) fn sah_cost(&self, root: u32) -> f32 {
        let root_area = self.nodes[root as usize].bounds.surface_area();
        let mut cost = 0.0f64;
        self.traverse(root, |_, n| {
            let area = n.bounds.surface_area() as f64;
            cost += if n.is_leaf() {
                area * n.leaf_range().len() as f64
            } else {
                area
            };
            true
        });
        if root_area > 0.0 {
            (cost / root_area as f64) as f32
        } else {
            0.0
        }
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p fast-mesh-bvh arena`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fast-mesh-bvh
git commit -m "feat(bvh): add node arena with allocation-free traversal and refit"
```

---

### Task 4: Subtree builders (binned SAH and Morton)

**Files:**
- Create: `crates/fast-mesh-bvh/src/build.rs`
- Modify: `crates/fast-mesh-bvh/src/lib.rs`

**Interfaces:**
- Consumes: `Arena`, `Node`, `NONE`, `LEAF_BIT` (Task 3); `try_reserve`, `try_reserve_exact`, `Result` (Task 2); `Aabb`, `BuildOptions`, `Quality`.
- Produces (crate-private):
  - `Prims` with `with_capacity(n) -> Result<Prims>`, `clear`, `push(id, Aabb)`, `len`, and public-in-crate fields `ids`, `boxes`, `centers`;
  - `build_subtree(&mut Arena, &mut Prims, &BuildOptions, parent: u32) -> Result<u32>`. It cannot fail when `max_nodes(n)` nodes and parents and `n` refs are reserved;
  - `max_nodes(n)`, `expected_nodes(n, leaf)`, `reserve_headroom(&mut Vec<T>, needed) -> Result<()>`.

- [ ] **Step 1: Write the failing tests**

Create `src/build.rs` with only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn arena_for(n: usize, tris: usize) -> Arena {
        let mut a = Arena {
            nodes: Vec::new(),
            parents: Vec::new(),
            refs: Vec::new(),
            tri_leaf: vec![NONE; tris],
        };
        a.nodes.reserve(max_nodes(n));
        a.parents.reserve(max_nodes(n));
        a.refs.reserve(n);
        a
    }

    /// `n` unit triangles spaced along x.
    fn row(n: u32) -> Prims {
        let mut p = Prims::with_capacity(n as usize).unwrap();
        for i in 0..n {
            let x = i as f32 * 2.0;
            p.push(i, Aabb::new([x, 0.0, 0.0], [x + 1.0, 1.0, 0.0]));
        }
        p
    }

    fn leaves(a: &Arena, root: u32) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        a.traverse(root, |_, n| {
            if n.is_leaf() {
                out.push(a.refs[n.leaf_range()].to_vec());
            }
            true
        });
        out
    }

    #[test]
    fn spread_interleaves_bits() {
        assert_eq!(spread(0b11), 0b1001);
        assert_eq!(spread(0x3ff), 0x09249249);
    }

    #[test]
    fn both_builders_cover_every_triangle_once_within_leaf_size() {
        for quality in [Quality::Sah, Quality::Fast] {
            let opts = BuildOptions {
                quality,
                max_leaf_size: 4,
                ..Default::default()
            };
            let mut prims = row(37);
            let mut arena = arena_for(37, 37);
            let root = build_subtree(&mut arena, &mut prims, &opts, NONE).unwrap();
            let mut all: Vec<u32> = leaves(&arena, root).into_iter().flatten().collect();
            all.sort();
            assert_eq!(all, (0..37).collect::<Vec<_>>(), "{quality:?}");
            assert!(leaves(&arena, root).iter().all(|l| l.len() <= 4));
            for t in 0..37 {
                let leaf = arena.tri_leaf[t as usize];
                assert!(arena.refs[arena.nodes[leaf as usize].leaf_range()].contains(&t));
            }
            assert_eq!(
                arena.nodes[root as usize].bounds,
                Aabb::new([0.0, 0.0, 0.0], [73.0, 1.0, 0.0])
            );
        }
    }

    #[test]
    fn sah_separates_spatial_clusters() {
        let mut p = Prims::with_capacity(8).unwrap();
        for i in 0..8u32 {
            let x = if i % 2 == 0 { 0.0 } else { 100.0 } + i as f32 * 0.01;
            p.push(i, Aabb::new([x, 0.0, 0.0], [x + 0.5, 0.5, 0.0]));
        }
        let mut arena = arena_for(8, 8);
        let opts = BuildOptions {
            max_leaf_size: 4,
            ..Default::default()
        };
        let root = build_subtree(&mut arena, &mut p, &opts, NONE).unwrap();
        let mut groups = leaves(&arena, root);
        groups.iter_mut().for_each(|g| g.sort());
        groups.sort();
        assert_eq!(groups, vec![vec![0, 2, 4, 6], vec![1, 3, 5, 7]]);
    }

    #[test]
    fn identical_centers_fall_back_to_halving() {
        for quality in [Quality::Sah, Quality::Fast] {
            let mut p = Prims::with_capacity(9).unwrap();
            for i in 0..9 {
                p.push(i, Aabb::new([0.0; 3], [1.0; 3]));
            }
            let mut arena = arena_for(9, 9);
            let opts = BuildOptions {
                quality,
                max_leaf_size: 2,
                ..Default::default()
            };
            let root = build_subtree(&mut arena, &mut p, &opts, NONE).unwrap();
            assert_eq!(leaves(&arena, root).iter().map(Vec::len).sum::<usize>(), 9);
            assert!(leaves(&arena, root).iter().all(|l| l.len() <= 2));
        }
    }

    #[test]
    fn empty_input_builds_one_empty_leaf() {
        let mut p = Prims::with_capacity(0).unwrap();
        let mut arena = arena_for(0, 0);
        let root = build_subtree(&mut arena, &mut p, &BuildOptions::default(), NONE).unwrap();
        assert!(arena.nodes[root as usize].is_leaf());
        assert!(arena.nodes[root as usize].bounds.is_empty());
        assert_eq!(arena.nodes[root as usize].leaf_range().len(), 0);
    }

    #[test]
    fn builds_are_deterministic() {
        let shape = |arena: &Arena| arena.nodes.iter().map(|n| (n.a, n.b)).collect::<Vec<_>>();
        for quality in [Quality::Sah, Quality::Fast] {
            let opts = BuildOptions {
                quality,
                max_leaf_size: 3,
                ..Default::default()
            };
            let (mut a1, mut a2) = (arena_for(50, 50), arena_for(50, 50));
            build_subtree(&mut a1, &mut row(50), &opts, NONE).unwrap();
            build_subtree(&mut a2, &mut row(50), &opts, NONE).unwrap();
            assert_eq!(shape(&a1), shape(&a2));
            assert_eq!(a1.refs, a2.refs);
        }
    }
}
```

Replace `src/lib.rs` with:

```rust
//! Incrementally updatable bounding volume hierarchy for meshes stored in
//! [`fast_mesh_store`].
//!
//! - [`Bvh::build`] builds over a mesh's alive triangles in mesh-local space,
//!   with a binned SAH ([`Quality::Sah`]) or Morton-code ([`Quality::Fast`]) split.
//! - [`Bvh::update`] takes the store's [`fast_mesh_store::ChangeSet`] and refits,
//!   rebuilds the affected treelets, or rebuilds everything, whichever is cheapest.
//! - [`Bvh::query_aabb`] finds the triangles whose boxes overlap a box.
//!
//! Every operation is failure-atomic: on error the BVH is left as it was.

mod aabb;
mod arena;
mod build;
mod error;
mod options;

pub use aabb::Aabb;
pub use error::{BvhError, Result};
pub use options::{BuildOptions, Quality, UpdatePolicy};
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p fast-mesh-bvh build`
Expected: compile errors (`cannot find type Prims`).

- [ ] **Step 3: Implement**

Insert above the test module:

```rust
//! Top-down subtree builders: binned SAH and Morton-code (LBVH).
//!
//! Both work on an index permutation (`Prims::order`) so the primitive arrays are
//! never moved. All memory is reserved up front by the caller, so building never
//! allocates and cannot fail.

use crate::arena::{Arena, LEAF_BIT, NONE, Node};
use crate::error::{Result, try_reserve, try_reserve_exact};
use crate::{Aabb, BuildOptions, Quality};

/// Triangles to build a subtree from, plus scratch space.
pub(crate) struct Prims {
    pub(crate) ids: Vec<u32>,
    pub(crate) boxes: Vec<Aabb>,
    pub(crate) centers: Vec<[f32; 3]>,
    keys: Vec<u32>,
    order: Vec<u32>,
    scratch: Vec<u32>,
}

/// Most nodes a subtree over `n` triangles can need.
pub(crate) fn max_nodes(n: usize) -> usize {
    if n == 0 { 1 } else { 2 * n - 1 }
}

/// Typical node count for `n` triangles with leaves of up to `leaf` triangles,
/// used as the initial reservation of full builds (they grow if needed).
pub(crate) fn expected_nodes(n: usize, leaf: u32) -> usize {
    (3 * n.div_ceil(leaf as usize) + 1).min(max_nodes(n))
}

/// Ensures room for `needed` more elements, reserving at least an eighth of the
/// current length extra so repeated small rebuilds do not copy the whole array
/// each time, while keeping slack far below the 2x of doubling.
pub(crate) fn reserve_headroom<T>(v: &mut Vec<T>, needed: usize) -> Result<()> {
    if v.capacity() - v.len() < needed {
        try_reserve_exact(v, needed.max(v.len() / 8))?;
    }
    Ok(())
}

impl Prims {
    /// Empty set with room for `n` triangles.
    pub(crate) fn with_capacity(n: usize) -> Result<Self> {
        let mut p = Self {
            ids: Vec::new(),
            boxes: Vec::new(),
            centers: Vec::new(),
            keys: Vec::new(),
            order: Vec::new(),
            scratch: Vec::new(),
        };
        try_reserve(&mut p.ids, n)?;
        try_reserve(&mut p.boxes, n)?;
        try_reserve(&mut p.centers, n)?;
        try_reserve(&mut p.keys, n)?;
        try_reserve(&mut p.order, n)?;
        try_reserve(&mut p.scratch, n)?;
        Ok(p)
    }

    pub(crate) fn clear(&mut self) {
        self.ids.clear();
        self.boxes.clear();
        self.centers.clear();
    }

    /// Adds a triangle. The caller stays within the reserved capacity.
    pub(crate) fn push(&mut self, id: u32, bounds: Aabb) {
        debug_assert!(
            self.ids.len() < self.ids.capacity(),
            "Prims capacity exceeded"
        );
        self.ids.push(id);
        self.boxes.push(bounds);
        self.centers.push(bounds.center());
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    fn range_bounds(&self, start: usize, end: usize) -> (Aabb, Aabb) {
        let mut bounds = Aabb::EMPTY;
        let mut centers = Aabb::EMPTY;
        for &i in &self.order[start..end] {
            bounds.grow(&self.boxes[i as usize]);
            centers.grow_point(self.centers[i as usize]);
        }
        (bounds, centers)
    }
}

/// Builds a subtree over all of `prims`, appending to `arena`, and returns its root.
/// `tri_leaf` must cover every triangle id in `prims`. Nodes are allocated as
/// needed; when the caller has already reserved `max_nodes(n)` nodes and parents and
/// `n` refs (as treelet rebuilds do), this cannot fail.
pub(crate) fn build_subtree(
    arena: &mut Arena,
    prims: &mut Prims,
    opts: &BuildOptions,
    parent: u32,
) -> Result<u32> {
    let n = prims.len();
    if arena.refs.capacity() - arena.refs.len() < n {
        try_reserve(&mut arena.refs, n)?;
    }
    prims.order.clear();
    prims.order.extend(0..n as u32);
    if opts.quality == Quality::Fast && n > 1 {
        morton_sort(prims);
    }

    let root = push_node(arena, parent)?;
    // Continue with the smaller child and push the larger: the stack never holds
    // more than log2(n) entries, so a fixed array is enough.
    let mut stack = [(0u32, 0usize, 0usize); 64];
    let mut depth = 0;
    let mut task = (root, 0, n);
    loop {
        let (node, start, end) = task;
        let (bounds, centers) = prims.range_bounds(start, end);
        if end - start <= opts.max_leaf_size as usize {
            let first = arena.refs.len() as u32;
            for &i in &prims.order[start..end] {
                let id = prims.ids[i as usize];
                arena.refs.push(id);
                arena.tri_leaf[id as usize] = node;
            }
            arena.nodes[node as usize] = Node::leaf(bounds, first, (end - start) as u32);
            if depth == 0 {
                return Ok(root);
            }
            depth -= 1;
            task = stack[depth];
            continue;
        }
        let mid = match opts.quality {
            Quality::Sah => sah_split(prims, start, end, &centers, opts.sah_bins as usize),
            Quality::Fast => morton_split(prims, start, end),
        };
        let left = push_node(arena, node)?;
        let right = push_node(arena, node)?;
        arena.nodes[node as usize] = Node {
            bounds,
            a: left,
            b: right,
        };
        let (small, large) = if mid - start <= end - mid {
            ((left, start, mid), (right, mid, end))
        } else {
            ((right, mid, end), (left, start, mid))
        };
        stack[depth] = large;
        depth += 1;
        task = small;
    }
}

fn push_node(arena: &mut Arena, parent: u32) -> Result<u32> {
    if arena.nodes.len() == arena.nodes.capacity() {
        let more = arena.nodes.len().max(16);
        try_reserve(&mut arena.nodes, more)?;
    }
    if arena.parents.len() == arena.parents.capacity() {
        let more = arena.parents.len().max(16);
        try_reserve(&mut arena.parents, more)?;
    }
    let index = arena.nodes.len() as u32;
    arena.nodes.push(Node {
        bounds: Aabb::EMPTY,
        a: NONE,
        b: LEAF_BIT,
    });
    arena.parents.push(parent);
    Ok(index)
}

/// Chooses the cheapest binned SAH split and partitions `order[start..end]`.
/// Falls back to the middle when no split separates the centers.
fn sah_split(prims: &mut Prims, start: usize, end: usize, centers: &Aabb, bins: usize) -> usize {
    let mid = start + (end - start) / 2;
    let mut best: Option<(f32, usize, usize)> = None; // (cost, axis, last left bin)
    for axis in 0..3 {
        let lo = centers.min[axis];
        let extent = centers.max[axis] - lo;
        if !extent.is_finite() || extent <= 0.0 {
            continue;
        }
        let bin_of =
            |c: f32| (((c - lo) / extent) * bins as f32).clamp(0.0, (bins - 1) as f32) as usize;
        let mut counts = [0u32; 64];
        let mut boxes = [Aabb::EMPTY; 64];
        for &i in &prims.order[start..end] {
            let b = bin_of(prims.centers[i as usize][axis]);
            counts[b] += 1;
            boxes[b].grow(&prims.boxes[i as usize]);
        }
        // Sweep from the right to get the right-hand side areas and counts.
        let mut right_area = [0.0f32; 64];
        let mut right_count = [0u32; 64];
        let mut acc = Aabb::EMPTY;
        let mut cnt = 0;
        for b in (1..bins).rev() {
            acc.grow(&boxes[b]);
            cnt += counts[b];
            right_area[b] = acc.surface_area();
            right_count[b] = cnt;
        }
        let mut left = Aabb::EMPTY;
        let mut left_count = 0;
        for b in 0..bins - 1 {
            left.grow(&boxes[b]);
            left_count += counts[b];
            if left_count == 0 || right_count[b + 1] == 0 {
                continue;
            }
            let cost = left.surface_area() * left_count as f32
                + right_area[b + 1] * right_count[b + 1] as f32;
            if best.is_none_or(|(c, _, _)| cost < c) {
                best = Some((cost, axis, b));
            }
        }
    }
    let Some((_, axis, split_bin)) = best else {
        return mid;
    };
    let lo = centers.min[axis];
    let extent = centers.max[axis] - lo;
    let in_left = |c: f32| {
        (((c - lo) / extent) * bins as f32).clamp(0.0, (bins - 1) as f32) as usize <= split_bin
    };
    let (mut i, mut j) = (start, end);
    while i < j {
        if in_left(prims.centers[prims.order[i] as usize][axis]) {
            i += 1;
        } else {
            j -= 1;
            prims.order.swap(i, j);
        }
    }
    if i == start || i == end { mid } else { i }
}

/// Splits a Morton-sorted range at the highest bit where its codes differ.
fn morton_split(prims: &Prims, start: usize, end: usize) -> usize {
    let key = |i: usize| prims.keys[prims.order[i] as usize];
    let (first, last) = (key(start), key(end - 1));
    if first == last {
        return start + (end - start) / 2;
    }
    let bit = 31 - (first ^ last).leading_zeros();
    let mask = 1u32 << bit;
    // Codes are sorted, so those with the bit set form a suffix.
    let (mut lo, mut hi) = (start, end - 1);
    while lo < hi {
        let m = lo + (hi - lo) / 2;
        if key(m) & mask != 0 {
            hi = m
        } else {
            lo = m + 1
        }
    }
    lo
}

/// 30-bit Morton codes of the centers within their bounds, then a stable LSD
/// radix sort of `order` by code (ties keep id order, so builds are deterministic).
fn morton_sort(prims: &mut Prims) {
    let mut bounds = Aabb::EMPTY;
    for &c in &prims.centers {
        bounds.grow_point(c);
    }
    prims.keys.clear();
    for &c in &prims.centers {
        let q = [0, 1, 2].map(|k| {
            let extent = bounds.max[k] - bounds.min[k];
            let t = if extent > 0.0 && extent.is_finite() {
                (c[k] - bounds.min[k]) / extent
            } else {
                0.0
            };
            (t.clamp(0.0, 1.0) * 1023.0) as u32
        });
        prims
            .keys
            .push((spread(q[0]) << 2) | (spread(q[1]) << 1) | spread(q[2]));
    }
    let n = prims.order.len();
    prims.scratch.clear();
    prims.scratch.resize(n, 0);
    for shift in [0, 8, 16, 24] {
        let mut counts = [0usize; 257];
        for &i in &prims.order {
            counts[((prims.keys[i as usize] >> shift) & 0xff) as usize + 1] += 1;
        }
        for b in 0..256 {
            counts[b + 1] += counts[b];
        }
        for &i in &prims.order {
            let b = ((prims.keys[i as usize] >> shift) & 0xff) as usize;
            prims.scratch[counts[b]] = i;
            counts[b] += 1;
        }
        std::mem::swap(&mut prims.order, &mut prims.scratch);
    }
}

/// Spreads the low 10 bits of `x` so that there are two zero bits between each.
fn spread(mut x: u32) -> u32 {
    x &= 0x3ff;
    x = (x | (x << 16)) & 0x030000ff;
    x = (x | (x << 8)) & 0x0300f00f;
    x = (x | (x << 4)) & 0x030c30c3;
    x = (x | (x << 2)) & 0x09249249;
    x
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p fast-mesh-bvh build`
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fast-mesh-bvh
git commit -m "feat(bvh): add binned SAH and Morton subtree builders"
```

---

### Task 5: The `Bvh` type: build, treelets, query, compaction, validation

**Files:**
- Create: `crates/fast-mesh-bvh/src/tree.rs`
- Modify: `crates/fast-mesh-bvh/src/lib.rs`

**Interfaces:**
- Consumes: everything from Tasks 2–4; `fast_mesh_store::{Mesh, TriangleId}`.
- Produces:
  - `pub struct Bvh` with:
    - `build(&Mesh, &BuildOptions) -> Result<Bvh>`;
    - `set_policy(UpdatePolicy) -> Result<()>`, `options()`, `policy()`, `is_current(&Mesh) -> bool`;
    - `query_aabb(&Mesh, min, max, &mut Vec<TriangleId>)`;
    - `node_count()`, `treelet_count()`, `root_bounds() -> Option<Aabb>`, `depth()`, `byte_size()`;
    - `validate(&Mesh) -> Result<(), String>`.
  - crate-private:
    - `Treelet { root, build_cost }`;
    - `Bvh` fields `arena`, `root`, `treelets`, `treelet_of_root: HashMap<u32, usize>`, `garbage_nodes`, `garbage_refs`, `mesh_uid`, `revision`, `options`, `policy`;
    - `Bvh::assemble(&Mesh, &BuildOptions, &UpdatePolicy) -> Result<Bvh>`, `treelet_of(node) -> usize`, `compact_storage() -> Result<()>`;
    - test-only `fingerprint() -> String`.

- [ ] **Step 1: Write the failing tests**

Create `src/tree.rs` with only:

```rust
#[cfg(test)]
impl Bvh {
    /// Test-only: everything that defines the tree, to compare before and after
    /// a failed operation. `tri_leaf` entries are compared as (triangle, leaf)
    /// pairs so growing it with `NONE` does not count as a change.
    pub(crate) fn fingerprint(&self) -> String {
        let a = &self.arena;
        let leaves: Vec<(usize, u32)> = a
            .tri_leaf
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l != NONE)
            .map(|(t, &l)| (t, l))
            .collect();
        let mut treelets: Vec<(u32, usize)> =
            self.treelet_of_root.iter().map(|(&r, &i)| (r, i)).collect();
        treelets.sort();
        format!(
            "{:?}|{:?}|{:?}|{:?}|{}|{:?}|{}|{}|{}",
            a.nodes
                .iter()
                .map(|n| (n.bounds, n.a, n.b))
                .collect::<Vec<_>>(),
            a.parents,
            a.refs,
            leaves,
            self.root,
            treelets,
            self.garbage_nodes,
            self.garbage_refs,
            self.revision,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BvhError, Quality};

    /// n×n grid of quads in the XY plane.
    pub(crate) fn grid(n: u32) -> Mesh {
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
                indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        Mesh::from_buffers(&positions, &indices).unwrap()
    }

    fn brute(mesh: &Mesh, q: &Aabb) -> Vec<u32> {
        (0..mesh.triangle_slots() as u32)
            .filter(|&t| {
                mesh.triangle(t)
                    .is_some_and(|tri| Aabb::of_triangle(mesh.positions(), tri).overlaps(q))
            })
            .collect()
    }

    #[test]
    fn builds_and_queries_both_qualities() {
        let mesh = grid(20);
        for quality in [Quality::Sah, Quality::Fast] {
            let opts = BuildOptions {
                quality,
                treelet_size: 64,
                ..Default::default()
            };
            let bvh = Bvh::build(&mesh, &opts).unwrap();
            bvh.validate(&mesh).unwrap();
            assert!(bvh.treelet_count() > 1);
            assert_eq!(
                bvh.root_bounds(),
                Some(Aabb::new([0.0; 3], [20.0, 20.0, 0.0]))
            );
            let q = Aabb::new([3.5, 3.5, -1.0], [7.0, 5.0, 1.0]);
            let mut got = Vec::new();
            bvh.query_aabb(&mesh, q.min, q.max, &mut got);
            got.sort();
            assert_eq!(got, brute(&mesh, &q), "{quality:?}");
        }
    }

    #[test]
    fn empty_mesh_and_single_triangle() {
        let empty = Mesh::new();
        let bvh = Bvh::build(&empty, &BuildOptions::default()).unwrap();
        bvh.validate(&empty).unwrap();
        assert_eq!(
            (bvh.node_count(), bvh.depth(), bvh.root_bounds()),
            (0, 0, None)
        );
        let mut out = Vec::new();
        bvh.query_aabb(&empty, [-1.0; 3], [1.0; 3], &mut out);
        assert!(out.is_empty());

        let one = Mesh::from_buffers(&[0., 0., 0., 1., 0., 0., 0., 1., 0.], &[0, 1, 2]).unwrap();
        let bvh = Bvh::build(&one, &BuildOptions::default()).unwrap();
        bvh.validate(&one).unwrap();
        assert_eq!(
            (bvh.node_count(), bvh.depth(), bvh.treelet_count()),
            (1, 1, 1)
        );
    }

    #[test]
    fn dead_slots_are_skipped() {
        let mut mesh = grid(4);
        let mut tx = mesh.begin_edit();
        tx.remove_triangles(&[0, 5, 9]).unwrap();
        tx.commit().unwrap();
        let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
        bvh.validate(&mesh).unwrap();
        let mut out = Vec::new();
        bvh.query_aabb(&mesh, [-1.0; 3], [10.0; 3], &mut out);
        assert_eq!(out.len(), mesh.triangle_count());
        assert!(!out.contains(&5));
    }

    #[test]
    fn degenerate_and_non_finite_triangles_do_not_break_the_build() {
        let mesh = Mesh::from_buffers(
            &[
                0.,
                0.,
                0.,
                1.,
                0.,
                0.,
                f32::NAN,
                1.,
                0.,
                1e30,
                1e30,
                1e30,
                2.,
                2.,
                2.,
            ],
            &[0, 0, 0, 0, 1, 2, 1, 3, 4, 4, 4, 4, 0, 1, 4],
        )
        .unwrap();
        for quality in [Quality::Sah, Quality::Fast] {
            let opts = BuildOptions {
                quality,
                max_leaf_size: 1,
                ..Default::default()
            };
            let bvh = Bvh::build(&mesh, &opts).unwrap();
            bvh.validate(&mesh).unwrap();
            let mut out = Vec::new();
            let q = Aabb::new([0.5, -1.0, -1.0], [1.5, 1.0, 1.0]);
            bvh.query_aabb(&mesh, q.min, q.max, &mut out);
            out.sort();
            assert_eq!(out, brute(&mesh, &q));
        }
    }

    #[test]
    fn invalid_options_are_rejected() {
        let opts = BuildOptions {
            max_leaf_size: 17,
            ..Default::default()
        };
        assert!(matches!(
            Bvh::build(&grid(1), &opts),
            Err(BvhError::InvalidOptions(_))
        ));
    }

    #[test]
    fn compaction_preserves_the_tree() {
        let mesh = grid(10);
        let mut bvh = Bvh::build(
            &mesh,
            &BuildOptions {
                treelet_size: 16,
                ..Default::default()
            },
        )
        .unwrap();
        let before = bvh.node_count();
        bvh.compact_storage().unwrap();
        bvh.validate(&mesh).unwrap();
        assert_eq!(bvh.node_count(), before);
    }
}
```

Replace `src/lib.rs` with:

```rust
//! Incrementally updatable bounding volume hierarchy for meshes stored in
//! [`fast_mesh_store`].
//!
//! - [`Bvh::build`] builds over a mesh's alive triangles in mesh-local space,
//!   with a binned SAH ([`Quality::Sah`]) or Morton-code ([`Quality::Fast`]) split.
//! - [`Bvh::update`] takes the store's [`fast_mesh_store::ChangeSet`] and refits,
//!   rebuilds the affected treelets, or rebuilds everything, whichever is cheapest.
//! - [`Bvh::query_aabb`] finds the triangles whose boxes overlap a box.
//!
//! Every operation is failure-atomic: on error the BVH is left as it was.

mod aabb;
mod arena;
mod build;
mod error;
mod options;
mod tree;

pub use aabb::Aabb;
pub use error::{BvhError, Result};
pub use options::{BuildOptions, Quality, UpdatePolicy};
pub use tree::Bvh;
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p fast-mesh-bvh tree`
Expected: compile errors (`cannot find type Bvh`).

- [ ] **Step 3: Implement**

Insert above the `#[cfg(test)] impl Bvh` block:

```rust
use std::collections::HashMap;

use fast_mesh_store::{Mesh, TriangleId};

use crate::arena::{Arena, NONE, Node};
use crate::build::{Prims, build_subtree, expected_nodes};
use crate::error::{Result, try_reserve, try_reserve_exact, try_reserve_map, try_vec};
use crate::{Aabb, BuildOptions, UpdatePolicy};

/// A subtree that is rebuilt as one unit.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Treelet {
    pub(crate) root: u32,
    /// SAH cost when the treelet was last built.
    pub(crate) build_cost: f32,
}

/// Bounding volume hierarchy over the alive triangles of one [`Mesh`], in
/// mesh-local coordinates. Keep it current with [`Bvh::update`].
pub struct Bvh {
    pub(crate) arena: Arena,
    pub(crate) root: u32,
    pub(crate) treelets: Vec<Treelet>,
    /// Treelet root node → index into `treelets`.
    pub(crate) treelet_of_root: HashMap<u32, usize>,
    /// Nodes and refs no longer reachable from the root.
    pub(crate) garbage_nodes: usize,
    pub(crate) garbage_refs: usize,
    pub(crate) mesh_uid: u64,
    pub(crate) revision: u64,
    pub(crate) options: BuildOptions,
    pub(crate) policy: UpdatePolicy,
}

impl std::fmt::Debug for Bvh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bvh")
            .field("nodes", &self.node_count())
            .field("treelets", &self.treelets.len())
            .field("depth", &self.depth())
            .field("bytes", &self.byte_size())
            .finish()
    }
}

/// Collects every alive triangle of `mesh` into `prims`.
pub(crate) fn alive_prims(mesh: &Mesh) -> Result<Prims> {
    let mut prims = Prims::with_capacity(mesh.triangle_count())?;
    let (positions, triangles) = (mesh.positions(), mesh.triangles());
    for (t, &tri) in triangles.iter().enumerate() {
        if mesh.is_triangle_alive(t as TriangleId) {
            prims.push(t as u32, Aabb::of_triangle(positions, tri));
        }
    }
    Ok(prims)
}

impl Bvh {
    /// Builds a BVH over the alive triangles of `mesh`.
    pub fn build(mesh: &Mesh, options: &BuildOptions) -> Result<Bvh> {
        options.validate()?;
        let mut bvh = Self::assemble(mesh, options, &UpdatePolicy::default())?;
        bvh.revision = mesh.revision();
        Ok(bvh)
    }

    /// Replaces the update policy (see [`Bvh::update`]).
    pub fn set_policy(&mut self, policy: UpdatePolicy) -> Result<()> {
        policy.validate()?;
        self.policy = policy;
        Ok(())
    }

    pub fn options(&self) -> &BuildOptions {
        &self.options
    }

    pub fn policy(&self) -> &UpdatePolicy {
        &self.policy
    }

    /// True when the BVH matches the mesh's current revision.
    pub fn is_current(&self, mesh: &Mesh) -> bool {
        self.mesh_uid == mesh.uid() && self.revision == mesh.revision()
    }

    /// Builds a complete new tree. Nothing outside the returned value is touched,
    /// so callers can swap it in only on success.
    pub(crate) fn assemble(
        mesh: &Mesh,
        options: &BuildOptions,
        policy: &UpdatePolicy,
    ) -> Result<Bvh> {
        let mut prims = alive_prims(mesh)?;
        let n = prims.len();
        let mut arena = Arena {
            nodes: Vec::new(),
            parents: Vec::new(),
            refs: Vec::new(),
            tri_leaf: try_vec(mesh.triangle_slots(), NONE)?,
        };
        let expected = expected_nodes(n, options.max_leaf_size);
        try_reserve_exact(&mut arena.nodes, expected)?;
        try_reserve_exact(&mut arena.parents, expected)?;
        try_reserve_exact(&mut arena.refs, n)?;
        let root = if n == 0 {
            NONE
        } else {
            build_subtree(&mut arena, &mut prims, options, NONE)?
        };
        let mut bvh = Bvh {
            arena,
            root,
            treelets: Vec::new(),
            treelet_of_root: HashMap::new(),
            garbage_nodes: 0,
            garbage_refs: 0,
            mesh_uid: mesh.uid(),
            revision: mesh.revision(),
            options: *options,
            policy: *policy,
        };
        bvh.assign_treelets()?;
        Ok(bvh)
    }

    /// Marks the topmost subtrees holding at most `treelet_size` triangles (or
    /// single leaves) as treelets. Requires a tree without garbage.
    fn assign_treelets(&mut self) -> Result<()> {
        self.treelets.clear();
        self.treelet_of_root.clear();
        if self.root == NONE {
            return Ok(());
        }
        // Children always have larger indices than their parents, so a reverse
        // sweep sees children first.
        let nodes = &self.arena.nodes;
        let mut counts = try_vec(nodes.len(), 0u32)?;
        for i in (0..nodes.len()).rev() {
            let n = &nodes[i];
            counts[i] = if n.is_leaf() {
                n.leaf_range().len() as u32
            } else {
                counts[n.a as usize] + counts[n.b as usize]
            };
        }
        let limit = self.options.treelet_size;
        let is_treelet_root = |i: u32, n: &Node| n.is_leaf() || counts[i as usize] <= limit;
        let mut found = 0;
        self.arena.traverse(self.root, |i, n| {
            let root = is_treelet_root(i, n);
            found += usize::from(root);
            !root
        });
        let mut roots = Vec::new();
        try_reserve(&mut roots, found)?;
        try_reserve_map(&mut self.treelet_of_root, found)?;
        try_reserve(&mut self.treelets, found)?;
        self.arena.traverse(self.root, |i, n| {
            let root = is_treelet_root(i, n);
            if root {
                roots.push(i);
            }
            !root
        });
        for root in roots {
            self.treelet_of_root.insert(root, self.treelets.len());
            self.treelets.push(Treelet {
                root,
                build_cost: self.arena.sah_cost(root),
            });
        }
        Ok(())
    }

    /// Treelet containing `node` (a node inside or at the root of a treelet).
    pub(crate) fn treelet_of(&self, mut node: u32) -> usize {
        loop {
            if let Some(&t) = self.treelet_of_root.get(&node) {
                return t;
            }
            node = self.arena.parents[node as usize];
            debug_assert!(node != NONE, "node is above every treelet");
        }
    }

    /// Appends every alive triangle whose bounding box overlaps the box
    /// `[min, max]` (touching counts), in no particular order.
    pub fn query_aabb(&self, mesh: &Mesh, min: [f32; 3], max: [f32; 3], out: &mut Vec<TriangleId>) {
        let query = Aabb::new(min, max);
        let (positions, triangles) = (mesh.positions(), mesh.triangles());
        self.arena.traverse(self.root, |_, node| {
            if !node.bounds.overlaps(&query) {
                return false;
            }
            if node.is_leaf() {
                for &t in &self.arena.refs[node.leaf_range()] {
                    if Aabb::of_triangle(positions, triangles[t as usize]).overlaps(&query) {
                        out.push(t);
                    }
                }
            }
            true
        });
    }

    /// Reachable nodes.
    pub fn node_count(&self) -> usize {
        self.arena.nodes.len() - self.garbage_nodes
    }

    pub fn treelet_count(&self) -> usize {
        self.treelets.len()
    }

    /// Bounds of all triangles, or `None` for an empty tree.
    pub fn root_bounds(&self) -> Option<Aabb> {
        (self.root != NONE).then(|| self.arena.nodes[self.root as usize].bounds)
    }

    /// Longest root-to-leaf path, counting nodes (0 for an empty tree).
    pub fn depth(&self) -> usize {
        let mut deepest = 0;
        self.arena.traverse(self.root, |i, n| {
            if n.is_leaf() {
                let mut d = 1;
                let mut cur = i;
                while self.arena.parents[cur as usize] != NONE {
                    cur = self.arena.parents[cur as usize];
                    d += 1;
                }
                deepest = deepest.max(d);
            }
            true
        });
        deepest
    }

    /// Heap memory held by the BVH, garbage included.
    pub fn byte_size(&self) -> usize {
        let a = &self.arena;
        a.nodes.capacity() * size_of::<Node>()
            + (a.parents.capacity() + a.refs.capacity() + a.tri_leaf.capacity()) * 4
            + self.treelets.capacity() * size_of::<Treelet>()
            + self.treelet_of_root.capacity() * 16
    }

    /// Copies the reachable nodes and refs into fresh storage in depth-first
    /// order, dropping garbage. On allocation failure nothing changes.
    pub(crate) fn compact_storage(&mut self) -> Result<()> {
        let live_nodes = self.node_count();
        let live_refs = self.arena.refs.len() - self.garbage_refs;
        let mut new_index = try_vec(self.arena.nodes.len(), NONE)?;
        let mut order = Vec::new();
        try_reserve(&mut order, live_nodes)?;
        let mut nodes = Vec::new();
        try_reserve(&mut nodes, live_nodes)?;
        let mut parents = Vec::new();
        try_reserve(&mut parents, live_nodes)?;
        let mut refs = Vec::new();
        try_reserve(&mut refs, live_refs)?;
        let mut tri_leaf = try_vec(self.arena.tri_leaf.len(), NONE)?;
        let mut roots = HashMap::new();
        try_reserve_map(&mut roots, self.treelets.len())?;

        // Phase 2: nothing below allocates.
        self.arena.traverse(self.root, |i, _| {
            new_index[i as usize] = order.len() as u32;
            order.push(i);
            true
        });
        for &old in &order {
            let n = self.arena.nodes[old as usize];
            let me = nodes.len() as u32;
            let parent = self.arena.parents[old as usize];
            parents.push(if parent == NONE {
                NONE
            } else {
                new_index[parent as usize]
            });
            if n.is_leaf() {
                let first = refs.len() as u32;
                for &t in &self.arena.refs[n.leaf_range()] {
                    refs.push(t);
                    tri_leaf[t as usize] = me;
                }
                nodes.push(Node::leaf(n.bounds, first, n.leaf_range().len() as u32));
            } else {
                nodes.push(Node {
                    bounds: n.bounds,
                    a: new_index[n.a as usize],
                    b: new_index[n.b as usize],
                });
            }
        }
        for (i, t) in self.treelets.iter_mut().enumerate() {
            t.root = new_index[t.root as usize];
            roots.insert(t.root, i);
        }
        if self.root != NONE {
            self.root = new_index[self.root as usize];
        }
        self.arena = Arena {
            nodes,
            parents,
            refs,
            tri_leaf,
        };
        self.treelet_of_root = roots;
        self.garbage_nodes = 0;
        self.garbage_refs = 0;
        Ok(())
    }

    /// Checks every structural invariant against `mesh`. Meant for tests and
    /// debugging; returns a description of the first violation.
    pub fn validate(&self, mesh: &Mesh) -> std::result::Result<(), String> {
        let a = &self.arena;
        let (positions, triangles) = (mesh.positions(), mesh.triangles());
        if self.root != NONE && a.parents[self.root as usize] != NONE {
            return Err("root has a parent".into());
        }
        let mut error: Option<String> = None;
        let (mut reachable, mut refs_seen) = (0usize, 0usize);
        a.traverse(self.root, |i, n| {
            if error.is_some() {
                return false;
            }
            reachable += 1;
            if n.is_leaf() {
                for &t in &a.refs[n.leaf_range()] {
                    refs_seen += 1;
                    if !mesh.is_triangle_alive(t) {
                        error = Some(format!("leaf {i} references dead triangle {t}"));
                    } else if a.tri_leaf.get(t as usize) != Some(&i) {
                        error = Some(format!(
                            "tri_leaf of triangle {t} does not point at leaf {i}"
                        ));
                    } else if !n
                        .bounds
                        .contains(&Aabb::of_triangle(positions, triangles[t as usize]))
                    {
                        error = Some(format!("leaf {i} box does not contain triangle {t}"));
                    }
                }
            } else {
                for c in [n.a, n.b] {
                    if a.parents[c as usize] != i {
                        error = Some(format!("child {c} of node {i} has the wrong parent"));
                    } else if !n.bounds.contains(&a.nodes[c as usize].bounds) {
                        error = Some(format!("node {i} box does not contain child {c}"));
                    }
                }
            }
            true
        });
        if let Some(e) = error {
            return Err(e);
        }
        if refs_seen != mesh.triangle_count() {
            return Err(format!(
                "tree holds {refs_seen} triangles, mesh has {}",
                mesh.triangle_count()
            ));
        }
        if reachable + self.garbage_nodes != a.nodes.len() {
            return Err(format!(
                "{reachable} reachable + {} garbage nodes != {} stored",
                self.garbage_nodes,
                a.nodes.len()
            ));
        }
        if refs_seen + self.garbage_refs != a.refs.len() {
            return Err("reachable + garbage refs != stored refs".into());
        }
        for (i, t) in self.treelets.iter().enumerate() {
            if self.treelet_of_root.get(&t.root) != Some(&i) {
                return Err(format!("treelet {i} is not indexed by its root"));
            }
        }
        if self.root != NONE {
            // Every leaf must sit inside exactly one treelet.
            let mut leaves_in_treelets = 0;
            for t in &self.treelets {
                a.traverse(t.root, |_, n| {
                    if n.is_leaf() {
                        leaves_in_treelets += 1;
                    }
                    true
                });
            }
            let mut leaves = 0;
            a.traverse(self.root, |_, n| {
                if n.is_leaf() {
                    leaves += 1;
                }
                true
            });
            if leaves != leaves_in_treelets {
                return Err(format!(
                    "{leaves} leaves but {leaves_in_treelets} inside treelets"
                ));
            }
        }
        Ok(())
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p fast-mesh-bvh`
Expected: all pass (6 new tree tests).

- [ ] **Step 5: Commit**

```bash
git add crates/fast-mesh-bvh
git commit -m "feat(bvh): add Bvh build, treelets, box query, compaction and validation"
```

---

### Task 6: Incremental update

**Files:**
- Create: `crates/fast-mesh-bvh/src/update.rs`
- Modify: `crates/fast-mesh-bvh/src/lib.rs`

**Interfaces:**
- Consumes: Tasks 1–5. That includes `Mesh::ensure_incidence` / `Mesh::incidence` (Task 1), `fast_mesh_store::{ChangeSet, History}` and `fail_allocations_after` (Task 2).
- Produces:
  - `pub enum Strategy { Unchanged, Refit, Partial, Full }`;
  - `pub struct UpdateReport { strategy, refit_nodes, rebuilt_treelets, compacted }`;
  - `Bvh::update(&mut self, &mut Mesh, &ChangeSet) -> Result<UpdateReport>`;
  - `Bvh::refit(&mut self, &Mesh, &[TriangleId]) -> Result<()>`;
  - `Bvh::rebuild_region(&mut self, &Mesh, &[TriangleId]) -> Result<()>`;
  - `Bvh::rebuild(&mut self, &Mesh) -> Result<()>`.

- [ ] **Step 1: Write the failing tests**

Create `src/update.rs` with only the two test modules:

```rust
#[cfg(test)]
mod tests {
    use fast_mesh_store::{History, Mesh};

    use crate::{Aabb, BuildOptions, Bvh, BvhError, Strategy, UpdatePolicy};

    /// n×n grid of quads in the XY plane.
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
                indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        Mesh::from_buffers(&positions, &indices).unwrap()
    }

    fn opts() -> BuildOptions {
        BuildOptions {
            max_leaf_size: 4,
            treelet_size: 32,
            ..Default::default()
        }
    }

    fn hits(bvh: &Bvh, mesh: &Mesh, min: [f32; 3], max: [f32; 3]) -> Vec<u32> {
        let mut out = Vec::new();
        bvh.query_aabb(mesh, min, max, &mut out);
        out.sort();
        out
    }

    #[test]
    fn moving_a_vertex_refits() {
        let mut m = grid(16);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[0.0, 0.0, 5.0]]).unwrap();
        let p = tx.commit().unwrap();
        let report = bvh.update(&mut m, p.changes()).unwrap();
        assert_eq!(report.strategy, Strategy::Refit);
        assert!(report.refit_nodes > 0);
        bvh.validate(&m).unwrap();
        assert!(bvh.is_current(&m));
        assert_eq!(
            hits(&bvh, &m, [-1.0, -1.0, 4.0], [0.5, 0.5, 6.0]),
            vec![0, 1]
        );
    }

    #[test]
    fn local_topology_edits_rebuild_only_their_treelets() {
        let mut m = grid(16);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let treelets = bvh.treelet_count();
        assert!(treelets > 4);
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[0, 1]).unwrap();
        let v = tx.add_vertices(&[[0.5, 0.5, 2.0]]).unwrap()[0];
        tx.add_triangles(&[[0, 1, v]]).unwrap();
        let p = tx.commit().unwrap();
        let report = bvh.update(&mut m, p.changes()).unwrap();
        assert_eq!(report.strategy, Strategy::Partial);
        assert!(report.rebuilt_treelets >= 1 && report.rebuilt_treelets < treelets);
        bvh.validate(&m).unwrap();
        assert_eq!(
            hits(&bvh, &m, [0.0, 0.0, 1.5], [1.0, 1.0, 2.5]),
            p.changes().triangles_added
        );
    }

    #[test]
    fn large_edits_rebuild_everything() {
        let mut m = grid(8);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let all: Vec<u32> = (0..m.vertex_slots() as u32).collect();
        let lifted: Vec<[f32; 3]> = all
            .iter()
            .map(|&v| {
                let p = m.position(v).unwrap();
                [p[0], p[1], 1.0]
            })
            .collect();
        let mut tx = m.begin_edit();
        tx.set_positions(&all, &lifted).unwrap();
        let p = tx.commit().unwrap();
        let report = bvh.update(&mut m, p.changes()).unwrap();
        assert_eq!(report.strategy, Strategy::Full);
        bvh.validate(&m).unwrap();
    }

    #[test]
    fn empty_changes_only_record_the_revision() {
        let mut m = grid(2);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut tx = m.begin_edit();
        tx.add_vertices(&[[9.0; 3]]).unwrap();
        let p = tx.commit().unwrap();
        assert!(!bvh.is_current(&m));
        let report = bvh.update(&mut m, p.changes()).unwrap();
        assert_eq!(report.strategy, Strategy::Unchanged);
        assert!(bvh.is_current(&m));
    }

    #[test]
    fn undo_and_redo_change_sets_keep_the_tree_correct() {
        let mut m = grid(12);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut h = History::new(1 << 20);
        let mut tx = m.begin_edit();
        tx.remove_triangles(&[10, 11, 12]).unwrap();
        tx.set_positions(&[40], &[[3.0, 3.0, 3.0]]).unwrap();
        let p = tx.commit().unwrap();
        bvh.update(&mut m, p.changes()).unwrap();
        h.push(&mut m, p).unwrap();
        bvh.validate(&m).unwrap();

        let undo = h.undo(&mut m).unwrap().unwrap();
        bvh.update(&mut m, &undo).unwrap();
        bvh.validate(&m).unwrap();
        let redo = h.redo(&mut m).unwrap().unwrap();
        bvh.update(&mut m, &redo).unwrap();
        bvh.validate(&m).unwrap();
    }

    #[test]
    fn emptying_and_refilling_the_mesh() {
        let mut m = grid(2);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let all: Vec<u32> = (0..8).collect();
        let mut tx = m.begin_edit();
        tx.remove_triangles(&all).unwrap();
        let p = tx.commit().unwrap();
        bvh.update(&mut m, p.changes()).unwrap();
        bvh.validate(&m).unwrap();
        assert_eq!(bvh.root_bounds(), None);
        let undo = p.revert(&mut m).unwrap();
        bvh.update(&mut m, &undo).unwrap();
        bvh.validate(&m).unwrap();
        assert_eq!(
            bvh.root_bounds(),
            Some(Aabb::new([0.0; 3], [2.0, 2.0, 0.0]))
        );
    }

    #[test]
    fn another_mesh_is_rejected() {
        let m = grid(2);
        let mut other = grid(2);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut tx = other.begin_edit();
        tx.set_positions(&[0], &[[1.0; 3]]).unwrap();
        let p = tx.commit().unwrap();
        assert_eq!(
            bvh.update(&mut other, p.changes()),
            Err(BvhError::MeshMismatch)
        );
        assert_eq!(bvh.refit(&other, &[0]), Err(BvhError::MeshMismatch));
        assert_eq!(bvh.rebuild(&other), Err(BvhError::MeshMismatch));
        bvh.validate(&m).unwrap();
    }

    #[test]
    fn degradation_triggers_a_treelet_rebuild() {
        let mut m = grid(16);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        bvh.set_policy(UpdatePolicy {
            degradation_factor: 1.0,
            full_rebuild_fraction: 1.0,
        })
        .unwrap();
        // Scatter a few vertices far away: refits blow up the boxes they touch.
        let ids = [0u32, 50, 100, 150];
        let far: Vec<[f32; 3]> = ids
            .iter()
            .map(|&v| [v as f32, 0.0, 100.0 + v as f32])
            .collect();
        let mut tx = m.begin_edit();
        tx.set_positions(&ids, &far).unwrap();
        let p = tx.commit().unwrap();
        let report = bvh.update(&mut m, p.changes()).unwrap();
        assert!(report.rebuilt_treelets > 0, "{report:?}");
        bvh.validate(&m).unwrap();
    }

    #[test]
    fn manual_refit_and_region_rebuild() {
        let mut m = grid(8);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut tx = m.begin_edit();
        tx.set_positions(&[0], &[[0.0, 0.0, 2.0]]).unwrap();
        tx.commit().unwrap();
        bvh.refit(&m, &[0, 1]).unwrap();
        bvh.validate(&m).unwrap();
        bvh.rebuild_region(&m, &[0]).unwrap();
        bvh.validate(&m).unwrap();
        bvh.rebuild(&m).unwrap();
        bvh.validate(&m).unwrap();
        assert!(bvh.is_current(&m));
    }

    #[test]
    fn garbage_is_compacted_away() {
        let mut m = grid(16);
        let mut bvh = Bvh::build(&m, &opts()).unwrap();
        let mut compacted = false;
        for round in 0..40u32 {
            let t = (round * 37) % m.triangle_slots() as u32;
            let Some(tri) = m.triangle(t) else { continue };
            let mut tx = m.begin_edit();
            tx.remove_triangles(&[t]).unwrap();
            tx.add_triangles(&[tri]).unwrap();
            let p = tx.commit().unwrap();
            compacted |= bvh.update(&mut m, p.changes()).unwrap().compacted;
            bvh.validate(&m).unwrap();
        }
        assert!(compacted);
    }
}

#[cfg(test)]
mod oom_tests {
    use fast_mesh_store::Mesh;

    use crate::error::fail_allocations_after;
    use crate::{BuildOptions, Bvh, BvhError};

    fn wavy(n: u32) -> Mesh {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                positions.extend([x as f32, y as f32, ((x + y) % 3) as f32]);
            }
        }
        let mut indices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        Mesh::from_buffers(&positions, &indices).unwrap()
    }

    /// Fails the k-th allocation of an update for every k until the update
    /// succeeds; each failure must leave the tree exactly as it was.
    fn assert_atomic(edit: impl Fn(&mut Mesh) -> fast_mesh_store::ChangeSet) {
        let opts = BuildOptions {
            max_leaf_size: 2,
            treelet_size: 8,
            ..Default::default()
        };
        for k in 0.. {
            let mut m = wavy(8);
            let mut bvh = Bvh::build(&m, &opts).unwrap();
            m.ensure_incidence().unwrap();
            let changes = edit(&mut m);
            let before = bvh.fingerprint();
            fail_allocations_after(Some(k));
            let result = bvh.update(&mut m, &changes);
            fail_allocations_after(None);
            match result {
                Err(BvhError::OutOfMemory) => {
                    assert_eq!(bvh.fingerprint(), before, "allocation {k}")
                }
                Err(e) => panic!("unexpected error {e}"),
                Ok(_) => {
                    bvh.validate(&m).unwrap();
                    assert!(k > 0, "the update must allocate at least once");
                    return;
                }
            }
        }
    }

    #[test]
    fn failed_partial_updates_leave_the_tree_unchanged() {
        assert_atomic(|m| {
            let mut tx = m.begin_edit();
            tx.remove_triangles(&[3, 40]).unwrap();
            tx.add_triangles(&[[0, 10, 20]]).unwrap();
            tx.set_positions(&[30], &[[3.0, 3.0, 4.0]]).unwrap();
            tx.commit().unwrap().changes().clone()
        });
    }

    #[test]
    fn failed_full_rebuilds_leave_the_tree_unchanged() {
        assert_atomic(|m| {
            let all: Vec<u32> = (0..m.triangle_slots() as u32).step_by(2).collect();
            let mut tx = m.begin_edit();
            tx.remove_triangles(&all).unwrap();
            tx.commit().unwrap().changes().clone()
        });
    }
}
```

Replace `src/lib.rs` with:

```rust
//! Incrementally updatable bounding volume hierarchy for meshes stored in
//! [`fast_mesh_store`].
//!
//! - [`Bvh::build`] builds over a mesh's alive triangles in mesh-local space,
//!   with a binned SAH ([`Quality::Sah`]) or Morton-code ([`Quality::Fast`]) split.
//! - [`Bvh::update`] takes the store's [`fast_mesh_store::ChangeSet`] and refits,
//!   rebuilds the affected treelets, or rebuilds everything, whichever is cheapest.
//! - [`Bvh::query_aabb`] finds the triangles whose boxes overlap a box.
//!
//! Every operation is failure-atomic: on error the BVH is left as it was.

mod aabb;
mod arena;
mod build;
mod error;
mod options;
mod tree;
mod update;

pub use aabb::Aabb;
pub use error::{BvhError, Result};
pub use options::{BuildOptions, Quality, UpdatePolicy};
pub use tree::Bvh;
pub use update::{Strategy, UpdateReport};
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p fast-mesh-bvh update`
Expected: compile errors (`no method named update`, unresolved `Strategy`).

- [ ] **Step 3: Implement**

Insert above the test modules:

```rust
//! Incremental updates driven by the store's `ChangeSet`.
//!
//! Every fallible allocation happens in a preparation phase before the tree is
//! touched; the mutation phase cannot fail. Two follow-up steps — rebuilding
//! degraded treelets and compacting storage — are optimizations: if memory runs
//! out there they are skipped and the tree stays correct.

use fast_mesh_store::{ChangeSet, Mesh, TriangleId};

use crate::Aabb;
use crate::arena::NONE;
use crate::build::{Prims, build_subtree, max_nodes, reserve_headroom};
use crate::error::{BvhError, Result, try_reserve, try_vec};
use crate::tree::Bvh;

/// How `update` brought the tree up to date.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Nothing relevant changed.
    Unchanged,
    /// Only boxes were recomputed.
    Refit,
    /// Some treelets were rebuilt (and boxes refit).
    Partial,
    /// The whole tree was rebuilt.
    Full,
}

/// What `update` did, for diagnostics and benchmarks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpdateReport {
    pub strategy: Strategy,
    /// Inner node boxes rewritten while propagating refits.
    pub refit_nodes: usize,
    /// Treelets rebuilt (topology changes plus degradation).
    pub rebuilt_treelets: usize,
    /// Whether garbage was compacted away.
    pub compacted: bool,
}

impl UpdateReport {
    fn new(strategy: Strategy) -> Self {
        Self {
            strategy,
            refit_nodes: 0,
            rebuilt_treelets: 0,
            compacted: false,
        }
    }
}

/// Sorted union of two sorted, deduplicated id lists.
fn merge(a: &[u32], b: &[u32]) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    try_reserve(&mut out, a.len() + b.len())?;
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

impl Bvh {
    /// Brings the BVH up to date with `mesh`, given the [`ChangeSet`]s of every
    /// commit, undo and redo since the BVH last matched (merge them when there are
    /// several). Builds the mesh's incidence index when vertices moved.
    ///
    /// Chooses the cheapest correct strategy: refit for moves, treelet rebuilds for
    /// local topology changes, a full rebuild when more than
    /// `UpdatePolicy::full_rebuild_fraction` of the triangles are dirty. On error the
    /// BVH is unchanged.
    pub fn update(&mut self, mesh: &mut Mesh, changes: &ChangeSet) -> Result<UpdateReport> {
        if mesh.uid() != self.mesh_uid {
            return Err(BvhError::MeshMismatch);
        }
        if !changes.vertices_moved.is_empty() {
            mesh.ensure_incidence()?;
        }
        let mesh = &*mesh;
        let removed = merge(&changes.triangles_removed, &changes.triangles_changed)?;
        let added = merge(&changes.triangles_added, &changes.triangles_changed)?;
        let mut moved = Vec::new();
        if let Some(inc) = mesh
            .incidence()
            .filter(|_| !changes.vertices_moved.is_empty())
        {
            let total: usize = changes
                .vertices_moved
                .iter()
                .map(|&v| inc.triangles(v).len())
                .sum();
            try_reserve(&mut moved, total)?;
            for &v in &changes.vertices_moved {
                moved.extend_from_slice(inc.triangles(v));
            }
            moved.sort_unstable();
            moved.dedup();
            // Triangles that are rebuilt anyway need no refit.
            moved.retain(|t| removed.binary_search(t).is_err() && added.binary_search(t).is_err());
        }

        let dirty = removed.len().max(added.len()) + moved.len();
        if removed.is_empty() && added.is_empty() && moved.is_empty() {
            self.revision = mesh.revision();
            return Ok(UpdateReport::new(Strategy::Unchanged));
        }
        let alive = mesh.triangle_count().max(1);
        if self.root == NONE || dirty as f32 > self.policy.full_rebuild_fraction * alive as f32 {
            *self = Bvh::assemble(mesh, &self.options, &self.policy)?;
            return Ok(UpdateReport::new(Strategy::Full));
        }

        let mut report = UpdateReport::new(if removed.is_empty() && added.is_empty() {
            Strategy::Refit
        } else {
            Strategy::Partial
        });
        let mut touched = self.rebuild_treelets(mesh, &removed, &added)?;
        report.rebuilt_treelets = touched.rebuilt;
        report.refit_nodes += touched.refit_nodes;

        // Refit leaves of moved triangles and propagate upwards.
        let (positions, triangles) = (mesh.positions(), mesh.triangles());
        for &t in &moved {
            let leaf = self.arena.tri_leaf[t as usize];
            if leaf == NONE {
                continue;
            }
            self.arena.refit_leaf(leaf, positions, triangles);
            report.refit_nodes += self.arena.refit_ancestors(leaf);
            touched.mark(self.treelet_of(leaf));
        }

        // Best effort: rebuild treelets whose quality degraded.
        for i in 0..self.treelets.len() {
            if !touched.refit[i] || touched.rebuilt_now[i] {
                continue;
            }
            let t = self.treelets[i];
            if self.arena.sah_cost(t.root) > self.policy.degradation_factor * t.build_cost
                && self.rebuild_one(mesh, i, &[], &[]).is_ok()
            {
                report.rebuilt_treelets += 1;
                report.strategy = Strategy::Partial;
            }
        }

        // Best effort: drop garbage once it exceeds half of the live tree.
        let live_refs = self.arena.refs.len() - self.garbage_refs;
        if (2 * self.garbage_nodes > self.node_count() || 2 * self.garbage_refs > live_refs)
            && self.compact_storage().is_ok()
        {
            report.compacted = true;
        }
        self.revision = mesh.revision();
        Ok(report)
    }

    /// Recomputes the boxes of the leaves holding `triangles` and their ancestors.
    pub fn refit(&mut self, mesh: &Mesh, triangles: &[TriangleId]) -> Result<()> {
        if mesh.uid() != self.mesh_uid {
            return Err(BvhError::MeshMismatch);
        }
        for &t in triangles {
            let leaf = self.arena.tri_leaf.get(t as usize).copied().unwrap_or(NONE);
            if leaf != NONE {
                self.arena
                    .refit_leaf(leaf, mesh.positions(), mesh.triangles());
                self.arena.refit_ancestors(leaf);
            }
        }
        Ok(())
    }

    /// Rebuilds the treelets containing `triangles` from their current contents.
    pub fn rebuild_region(&mut self, mesh: &Mesh, triangles: &[TriangleId]) -> Result<()> {
        if mesh.uid() != self.mesh_uid {
            return Err(BvhError::MeshMismatch);
        }
        let mut which = Vec::new();
        try_reserve(&mut which, triangles.len())?;
        for &t in triangles {
            let leaf = self.arena.tri_leaf.get(t as usize).copied().unwrap_or(NONE);
            if leaf != NONE {
                which.push(self.treelet_of(leaf));
            }
        }
        which.sort_unstable();
        which.dedup();
        for i in which {
            self.rebuild_one(mesh, i, &[], &[])?;
        }
        Ok(())
    }

    /// Rebuilds the whole tree from `mesh`. On error the BVH is unchanged.
    pub fn rebuild(&mut self, mesh: &Mesh) -> Result<()> {
        if mesh.uid() != self.mesh_uid {
            return Err(BvhError::MeshMismatch);
        }
        *self = Bvh::assemble(mesh, &self.options, &self.policy)?;
        Ok(())
    }

    /// Removes `removed` from the tree and inserts `added`, rebuilding each
    /// affected treelet. All allocation happens before the first write.
    fn rebuild_treelets(&mut self, mesh: &Mesh, removed: &[u32], added: &[u32]) -> Result<Touched> {
        let mut touched = Touched::new(self.treelets.len())?;
        if removed.is_empty() && added.is_empty() {
            return Ok(touched);
        }
        // tri_leaf must cover new slots; growing it with NONE changes nothing else.
        if self.arena.tri_leaf.len() < mesh.triangle_slots() {
            let extra = mesh.triangle_slots() - self.arena.tri_leaf.len();
            reserve_headroom(&mut self.arena.tri_leaf, extra)?;
            self.arena.tri_leaf.resize(mesh.triangle_slots(), NONE);
        }
        let positions = mesh.positions();
        let triangles = mesh.triangles();
        // Which treelet each removal and insertion belongs to.
        let mut queued: Vec<Vec<u32>> = Vec::new();
        try_reserve(&mut queued, self.treelets.len())?;
        queued.resize_with(self.treelets.len(), Vec::new);
        for &t in removed {
            let leaf = self.arena.tri_leaf[t as usize];
            if leaf != NONE {
                touched.rebuild[self.treelet_of(leaf)] = true;
            }
        }
        for &t in added {
            if !mesh.is_triangle_alive(t) {
                continue;
            }
            let i = self.choose_treelet(&Aabb::of_triangle(positions, triangles[t as usize]));
            touched.rebuild[i] = true;
            try_reserve(&mut queued[i], 1)?;
            queued[i].push(t);
        }
        // Reserve for the largest rebuild and for all appended nodes and refs.
        let (mut max_n, mut total_nodes, mut total_refs) = (0usize, 0usize, 0usize);
        for (i, q) in queued.iter().enumerate() {
            if touched.rebuild[i] {
                let (_, refs) = self.arena.subtree_size(self.treelets[i].root);
                let n = refs + q.len();
                max_n = max_n.max(n);
                total_nodes += max_nodes(n);
                total_refs += n;
            }
        }
        let mut prims = Prims::with_capacity(max_n)?;
        reserve_headroom(&mut self.arena.nodes, total_nodes)?;
        reserve_headroom(&mut self.arena.parents, total_nodes)?;
        reserve_headroom(&mut self.arena.refs, total_refs)?;

        // Mutation phase: nothing below can fail. Clear removed triangles first so
        // a rebuilt treelet never overwrites the leaf of a triangle re-added elsewhere.
        for &t in removed {
            self.arena.tri_leaf[t as usize] = NONE;
        }
        for (i, q) in queued.iter().enumerate() {
            if touched.rebuild[i] {
                touched.refit_nodes += self.replace_treelet(mesh, i, removed, q, &mut prims);
                touched.rebuilt += 1;
                touched.rebuilt_now[i] = true;
            }
        }
        Ok(touched)
    }

    /// Rebuilds one treelet with its own allocation (used for degradation and
    /// `rebuild_region`). On error the tree is unchanged.
    fn rebuild_one(&mut self, mesh: &Mesh, i: usize, removed: &[u32], added: &[u32]) -> Result<()> {
        let (_, refs) = self.arena.subtree_size(self.treelets[i].root);
        let n = refs + added.len();
        let mut prims = Prims::with_capacity(n)?;
        reserve_headroom(&mut self.arena.nodes, max_nodes(n))?;
        reserve_headroom(&mut self.arena.parents, max_nodes(n))?;
        reserve_headroom(&mut self.arena.refs, n)?;
        self.replace_treelet(mesh, i, removed, added, &mut prims);
        Ok(())
    }

    /// Builds a new subtree for treelet `i` from its triangles that are still alive
    /// and not in `removed`, plus `added`, links it in place of the old one and
    /// refits the ancestors. Capacity must already be reserved, and `tri_leaf` of
    /// removed triangles already cleared. Returns how many ancestor boxes changed.
    fn replace_treelet(
        &mut self,
        mesh: &Mesh,
        i: usize,
        removed: &[u32],
        added: &[u32],
        prims: &mut Prims,
    ) -> usize {
        let old_root = self.treelets[i].root;
        let (positions, triangles) = (mesh.positions(), mesh.triangles());
        prims.clear();
        let (mut old_nodes, mut old_refs) = (0usize, 0usize);
        let arena = &self.arena;
        arena.traverse(old_root, |_, n| {
            old_nodes += 1;
            if n.is_leaf() {
                old_refs += n.leaf_range().len();
                for &t in &arena.refs[n.leaf_range()] {
                    if mesh.is_triangle_alive(t) && removed.binary_search(&t).is_err() {
                        prims.push(t, Aabb::of_triangle(positions, triangles[t as usize]));
                    }
                }
            }
            true
        });
        for &t in added {
            prims.push(t, Aabb::of_triangle(positions, triangles[t as usize]));
        }
        let parent = self.arena.parents[old_root as usize];
        let new_root = build_subtree(&mut self.arena, prims, &self.options, parent)
            .expect("capacity for the rebuild was reserved before mutating");
        if parent == NONE {
            self.root = new_root;
        } else {
            let p = &mut self.arena.nodes[parent as usize];
            if p.a == old_root {
                p.a = new_root;
            } else {
                p.b = new_root;
            }
        }
        self.garbage_nodes += old_nodes;
        self.garbage_refs += old_refs;
        self.treelet_of_root.remove(&old_root);
        self.treelet_of_root.insert(new_root, i);
        self.treelets[i].root = new_root;
        self.treelets[i].build_cost = self.arena.sah_cost(new_root);
        self.arena.refit_ancestors(new_root)
    }

    /// Descends the top tree, at each node taking the child whose box grows least,
    /// until reaching a treelet root.
    fn choose_treelet(&self, b: &Aabb) -> usize {
        let mut node = self.root;
        loop {
            if let Some(&i) = self.treelet_of_root.get(&node) {
                return i;
            }
            let n = &self.arena.nodes[node as usize];
            let growth = |c: u32| {
                let cb = &self.arena.nodes[c as usize].bounds;
                cb.union(b).surface_area() - cb.surface_area()
            };
            node = if growth(n.b) < growth(n.a) { n.b } else { n.a };
        }
    }
}

/// Per-treelet bookkeeping for one update.
struct Touched {
    rebuild: Vec<bool>,
    rebuilt_now: Vec<bool>,
    refit: Vec<bool>,
    rebuilt: usize,
    refit_nodes: usize,
}

impl Touched {
    fn new(n: usize) -> Result<Self> {
        Ok(Self {
            rebuild: try_vec(n, false)?,
            rebuilt_now: try_vec(n, false)?,
            refit: try_vec(n, false)?,
            rebuilt: 0,
            refit_nodes: 0,
        })
    }

    fn mark(&mut self, treelet: usize) {
        self.refit[treelet] = true;
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p fast-mesh-bvh`
Expected: all pass (33 unit tests in the crate).

- [ ] **Step 5: Commit**

```bash
git add crates/fast-mesh-bvh
git commit -m "feat(bvh): add incremental updates driven by store change sets"
```

---

### Task 7: Property tests, benchmarks, docs and full verification

**Files:**
- Create: `crates/fast-mesh-bvh/tests/properties.rs`, `crates/fast-mesh-bvh/benches/bvh.rs`, `docs/benchmarks.md`
- Modify: `crates/fast-mesh-bvh/Cargo.toml`, `.github/workflows/ci.yml`, `CONTRIBUTING.md`, `docs/superpowers/specs/2026-10-15-fast-mesh-bvh-design.md`

**Interfaces:**
- Consumes: the public API of both crates.

- [ ] **Step 1: Property tests**

Create `crates/fast-mesh-bvh/tests/properties.rs`:

```rust
//! Property tests: after any sequence of edits, undos and redos, an updated BVH
//! must satisfy every invariant and answer box queries exactly like brute force
//! and like a freshly built tree.

use fast_mesh_bvh::{Aabb, BuildOptions, Bvh, Quality, UpdatePolicy};
use fast_mesh_store::{History, Mesh, Transaction};
use proptest::prelude::*;

#[derive(Debug, Clone)]
enum Op {
    Move(u32, [i8; 3]),
    AddVertex(u32, u32, u8),
    AddTriangle([u32; 3]),
    RemoveTriangle(u32),
    SetTriangle(u32, [u32; 3]),
}

#[derive(Debug, Clone)]
enum Step {
    Commit(Vec<Op>),
    Undo,
    Redo,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        4 => (any::<u32>(), any::<[i8; 3]>()).prop_map(|(v, d)| Op::Move(v, d)),
        1 => (any::<u32>(), any::<u32>(), any::<u8>()).prop_map(|(a, b, w)| Op::AddVertex(a, b, w)),
        2 => any::<[u32; 3]>().prop_map(Op::AddTriangle),
        2 => any::<u32>().prop_map(Op::RemoveTriangle),
        1 => (any::<u32>(), any::<[u32; 3]>()).prop_map(|(t, v)| Op::SetTriangle(t, v)),
    ]
}

fn step() -> impl Strategy<Value = Step> {
    prop_oneof![
        5 => prop::collection::vec(op(), 1..8).prop_map(Step::Commit),
        1 => Just(Step::Undo),
        1 => Just(Step::Redo),
    ]
}

fn config() -> impl Strategy<Value = (BuildOptions, UpdatePolicy)> {
    (
        prop_oneof![Just(Quality::Sah), Just(Quality::Fast)],
        1u32..=8,
        8u32..=64,
        0.05f32..=1.0,
        1.0f32..=2.0,
    )
        .prop_map(|(quality, leaf, treelet, fraction, factor)| {
            (
                BuildOptions {
                    quality,
                    max_leaf_size: leaf,
                    treelet_size: treelet.max(leaf),
                    ..Default::default()
                },
                UpdatePolicy {
                    full_rebuild_fraction: fraction,
                    degradation_factor: factor,
                },
            )
        })
}

/// A wavy n×n grid, so boxes are not all coplanar.
fn grid(n: u32) -> Mesh {
    let mut positions = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            positions.extend([x as f32, y as f32, ((x * 7 + y * 3) % 5) as f32 * 0.3]);
        }
    }
    let mut indices = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

fn alive(m: &Mesh, vertices: bool) -> Vec<u32> {
    let n = if vertices {
        m.vertex_slots()
    } else {
        m.triangle_slots()
    } as u32;
    (0..n)
        .filter(|&i| {
            if vertices {
                m.is_vertex_alive(i)
            } else {
                m.is_triangle_alive(i)
            }
        })
        .collect()
}

fn pick(ids: &[u32], seed: u32) -> Option<u32> {
    (!ids.is_empty()).then(|| ids[seed as usize % ids.len()])
}

fn run(tx: &mut Transaction<'_>, op: &Op) {
    let vs = alive(tx.mesh(), true);
    let ts = alive(tx.mesh(), false);
    let v = |s: u32| pick(&vs, s);
    match *op {
        Op::Move(s, d) => {
            if let Some(v) = v(s) {
                let p = tx.mesh().position(v).unwrap();
                let q = [0, 1, 2].map(|k| p[k] + d[k] as f32 * 0.05);
                tx.set_positions(&[v], &[q]).unwrap();
            }
        }
        Op::AddVertex(a, b, w) => {
            if let (Some(a), Some(b)) = (v(a), v(b)) {
                let (pa, pb) = (
                    tx.mesh().position(a).unwrap(),
                    tx.mesh().position(b).unwrap(),
                );
                let t = w as f32 / 255.0;
                let p = [0, 1, 2].map(|k| pa[k] * (1.0 - t) + pb[k] * t);
                tx.add_vertex_with_parents(p, &[(a, 1.0 - t), (b, t)])
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
        Op::SetTriangle(t, s) => {
            if let (Some(t), Some(a), Some(b), Some(c)) = (pick(&ts, t), v(s[0]), v(s[1]), v(s[2]))
            {
                tx.set_triangles(&[t], &[[a, b, c]]).unwrap();
            }
        }
    }
}

fn query(bvh: &Bvh, mesh: &Mesh, q: &Aabb) -> Vec<u32> {
    let mut out = Vec::new();
    bvh.query_aabb(mesh, q.min, q.max, &mut out);
    out.sort();
    out
}

fn brute(mesh: &Mesh, q: &Aabb) -> Vec<u32> {
    alive(mesh, false)
        .into_iter()
        .filter(|&t| Aabb::of_triangle(mesh.positions(), mesh.triangle(t).unwrap()).overlaps(q))
        .collect()
}

/// Boxes spread over the grid, from tiny to covering everything.
fn probe_boxes(n: u32) -> Vec<Aabb> {
    let n = n as f32;
    let mut boxes = vec![Aabb::new([-1.0; 3], [n + 1.0, n + 1.0, 5.0])];
    for i in 0..6 {
        let c = i as f32 * n / 6.0;
        let r = 0.3 + i as f32 * 0.7;
        boxes.push(Aabb::new([c - r, c - r, -1.0], [c + r, c + r, 2.0]));
        boxes.push(Aabb::new([c, n - c, 0.0], [c + 0.1, n - c + 0.1, 0.4]));
    }
    boxes
}

fn check(bvh: &Bvh, mesh: &Mesh, options: &BuildOptions, n: u32) -> Result<(), TestCaseError> {
    bvh.validate(mesh).map_err(TestCaseError::fail)?;
    prop_assert!(bvh.is_current(mesh));
    let fresh = Bvh::build(mesh, options).unwrap();
    for q in probe_boxes(n) {
        let got = query(bvh, mesh, &q);
        prop_assert_eq!(&got, &brute(mesh, &q));
        prop_assert_eq!(&got, &query(&fresh, mesh, &q));
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn updated_trees_match_brute_force(
        (options, policy) in config(),
        n in 3u32..10,
        steps in prop::collection::vec(step(), 1..25),
    ) {
        let mut m = grid(n);
        let mut bvh = Bvh::build(&m, &options).unwrap();
        bvh.set_policy(policy).unwrap();
        let mut h = History::new(usize::MAX);
        for step in &steps {
            let changes = match step {
                Step::Commit(ops) => {
                    let mut tx = m.begin_edit();
                    for op in ops {
                        run(&mut tx, op);
                    }
                    let patch = tx.commit().unwrap();
                    let changes = patch.changes().clone();
                    h.push(&mut m, patch).unwrap();
                    Some(changes)
                }
                Step::Undo => h.undo(&mut m).unwrap(),
                Step::Redo => h.redo(&mut m).unwrap(),
            };
            if let Some(changes) = changes {
                bvh.update(&mut m, &changes).unwrap();
            }
            check(&bvh, &m, &options, n)?;
        }
    }
}
```

Run: `cargo test -p fast-mesh-bvh --test properties`
Expected: 1 test passes (128 cases). If proptest reports a failure, fix the BVH, not the test.

- [ ] **Step 2: Benchmarks**

Replace `crates/fast-mesh-bvh/Cargo.toml` with:

```toml
[package]
name = "fast-mesh-bvh"
description = "Incrementally updatable bounding volume hierarchy for editable triangle meshes (part of fast-mesh-wasm)"
keywords = ["bvh", "mesh", "geometry", "collision", "3d"]
categories = ["graphics", "data-structures"]
readme = "../../README.md"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
authors.workspace = true

[dependencies]
fast-mesh-store.workspace = true

[dev-dependencies]
proptest = "1.12.0"
criterion = { version = "0.8.2", default-features = false }

[[bench]]
name = "bvh"
harness = false
```

Create `crates/fast-mesh-bvh/benches/bvh.rs`:

```rust
//! Build and update benchmarks on synthetic meshes.
//!
//! `cargo bench -p fast-mesh-bvh` runs 1M triangles; set `FAST_MESH_BENCH_LARGE=1`
//! to add 10M. Each size prints the BVH's node count, depth and memory once.
//!
//! Update benchmarks measure the steady state of an interactive session: one mesh
//! and one BVH are edited over and over. The first edit after loading a mesh is
//! slower, because the store's and the BVH's arrays grow once; that one-time
//! cost is printed separately.

use std::hint::black_box;
use std::time::Instant;

use criterion::{Criterion, criterion_group, criterion_main};
use fast_mesh_bvh::{BuildOptions, Bvh, Quality};
use fast_mesh_store::Mesh;

/// Height field over an n×n grid (2n² triangles), so boxes are not all coplanar.
fn terrain(n: u32) -> Mesh {
    let mut positions = Vec::with_capacity(((n + 1) * (n + 1) * 3) as usize);
    for y in 0..=n {
        for x in 0..=n {
            let (fx, fy) = (x as f32, y as f32);
            positions.extend([fx, fy, (fx * 0.05).sin() * (fy * 0.07).cos() * 4.0]);
        }
    }
    let mut indices = Vec::with_capacity((n * n * 6) as usize);
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// Ids of the vertices in a square patch of `side`×`side` grid points.
fn patch(n: u32, side: u32) -> Vec<u32> {
    let start = (n / 2 - side / 2) * (n + 1) + n / 2 - side / 2;
    (0..side)
        .flat_map(|dy| (0..side).map(move |dx| start + dy * (n + 1) + dx))
        .collect()
}

/// Moves `ids` up or down by 0.5 and updates the BVH.
fn move_patch(m: &mut Mesh, bvh: &mut Bvh, ids: &[u32], up: bool) {
    let dz = if up { 0.5 } else { -0.5 };
    let moved: Vec<[f32; 3]> = ids
        .iter()
        .map(|&v| {
            let p = m.position(v).unwrap();
            [p[0], p[1], p[2] + dz]
        })
        .collect();
    let mut tx = m.begin_edit();
    tx.set_positions(ids, &moved).unwrap();
    let patch = tx.commit().unwrap();
    black_box(bvh.update(m, patch.changes()).unwrap());
}

/// Removes 200 alive triangles starting near `first` and adds them back.
fn replace_triangles(m: &mut Mesh, bvh: &mut Bvh, first: u32) {
    let doomed: Vec<u32> = (first..m.triangle_slots() as u32)
        .filter(|&t| m.is_triangle_alive(t))
        .take(200)
        .collect();
    let replacements: Vec<[u32; 3]> = doomed.iter().map(|&t| m.triangle(t).unwrap()).collect();
    let mut tx = m.begin_edit();
    tx.remove_triangles(&doomed).unwrap();
    tx.add_triangles(&replacements).unwrap();
    let patch = tx.commit().unwrap();
    black_box(bvh.update(m, patch.changes()).unwrap());
}

fn bench_size(c: &mut Criterion, label: &str, n: u32) {
    let mesh = terrain(n);
    let sah = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
    eprintln!(
        "{label}: {} triangles, {} nodes, depth {}, {} treelets, {:.1} MB",
        mesh.triangle_count(),
        sah.node_count(),
        sah.depth(),
        sah.treelet_count(),
        sah.byte_size() as f64 / 1e6
    );
    {
        let mut m = terrain(n);
        let mut bvh = Bvh::build(&m, &BuildOptions::default()).unwrap();
        let start = Instant::now();
        replace_triangles(&mut m, &mut bvh, (n / 2) * n * 2);
        eprintln!(
            "{label}: first topology edit after loading took {:?} (one-time growth)",
            start.elapsed()
        );
    }

    let mut group = c.benchmark_group(label);
    group.sample_size(10);
    for quality in [Quality::Sah, Quality::Fast] {
        let opts = BuildOptions {
            quality,
            ..Default::default()
        };
        group.bench_function(format!("build {quality:?}"), |b| {
            b.iter(|| black_box(Bvh::build(&mesh, &opts).unwrap()))
        });
    }

    let mut m = terrain(n);
    m.ensure_incidence().unwrap();
    let mut bvh = Bvh::build(&m, &BuildOptions::default()).unwrap();
    for side in [22u32, 224] {
        let ids = patch(n, side);
        let mut up = true;
        group.bench_function(format!("move {} vertices + update", ids.len()), |b| {
            b.iter(|| {
                move_patch(&mut m, &mut bvh, &ids, up);
                up = !up;
            })
        });
    }
    let mut first = (n / 2) * n * 2;
    group.bench_function("remove + add 200 triangles + update", |b| {
        b.iter(|| {
            replace_triangles(&mut m, &mut bvh, first);
            first = (first + 997) % (n * n * 2);
        })
    });
    group.finish();
    assert!(
        bvh.validate(&m).is_ok(),
        "the benchmarked BVH must stay valid"
    );
}

fn benches(c: &mut Criterion) {
    bench_size(c, "1M", 708);
    if std::env::var_os("FAST_MESH_BENCH_LARGE").is_some() {
        bench_size(c, "10M", 2237);
    }
}

criterion_group!(bvh, benches);
criterion_main!(bvh);
```

Run: `cargo bench -p fast-mesh-bvh --no-run`
Expected: compiles. (Running the full benchmarks takes a few minutes; `docs/benchmarks.md` below records the reference results.)

- [ ] **Step 3: CI keeps the benchmarks compiling**

In `.github/workflows/ci.yml`, job `rust`, add after the `cargo test --workspace` step:

```yaml
      - run: cargo bench -p fast-mesh-bvh --no-run
```

- [ ] **Step 4: Documentation**

Create `docs/benchmarks.md`:

```markdown
# Benchmarks

Reference results for `cargo bench -p fast-mesh-bvh`. Times are criterion medians and depend on the machine, so compare runs only on the same hardware.

- **Machine:** Intel Core i5-13600K, 32 GB RAM, Windows 11, single thread
- **Toolchain:** Rust 1.99.0, release profile (LTO, `codegen-units = 1`)
- **Date:** 2026-10-15
- **Meshes:** synthetic height fields (`terrain(n)` in `benches/bvh.rs`): 1,002,528 and 10,008,338 triangles
- **Options:** defaults (`max_leaf_size` 8, `sah_bins` 16, `treelet_size` 4,096)

## BVH (fast-mesh-bvh)

| | 1M triangles | 10M triangles |
| --- | --- | --- |
| Nodes / depth / treelets | 338,389 / 19 / 270 | 3,146,083 / 23 / 4,096 |
| Memory (`byte_size`) | 21.6 MB | 215.3 MB |
| Build, `Quality::Sah` | 439 ms | 5.23 s |
| Build, `Quality::Fast` | 115 ms | 1.70 s |
| Move 484 vertices, commit + update (refit) | 0.16 ms | 0.14 ms |
| Move 50,176 vertices, commit + update (refit) | 15.7 ms | 17.3 ms |
| Remove + re-add 200 triangles, commit + update (treelet rebuild) | 6.3 ms | 2.8 ms |
| First topology edit after loading (one-time array growth) | 6.5 ms | 55.8 ms |

Update times are steady state: one mesh and one BVH are edited repeatedly, as in an interactive session. Small edits cost about the same at 1M and 10M because they touch only the affected leaves and treelets. The first edit after loading a mesh is slower once: the store's columns and the BVH's arrays grow from their exact initial size, which copies them one time.

## How the defaults were chosen

`treelet_size` sets the cost of a topology edit, because every edit rebuilds the treelets it touches. Measured on 1M triangles (first-edit timing, before the steady-state benchmark existed, so absolute numbers are higher than the table above):

| `treelet_size` | Remove + re-add 200 triangles | Move 484 vertices |
| --- | --- | --- |
| 65,536 | 29.6 ms | 2.4 ms |
| 16,384 | 17.6 ms | 2.1 ms |
| 4,096 | 8.9 ms | 2.1 ms |

The default is **4,096**: topology edits cost about a third of the 65,536 setting, while moves and builds are unaffected. Even at 10M triangles the top tree stays small (4,096 treelets).

`full_rebuild_fraction` (0.30) and `degradation_factor` (1.5) are kept. Moving 5% of the vertices of a 1M mesh refits in about 16 ms, against 439 ms for a full SAH build, so falling back to a full rebuild earlier than 30% dirty would be slower.

## Reproducing

```sh
cargo bench -p fast-mesh-bvh                          # 1M triangles
FAST_MESH_BENCH_LARGE=1 cargo bench -p fast-mesh-bvh  # adds 10M (needs ~2 GB RAM, several minutes)
```
```

In `CONTRIBUTING.md`, in the "Project layout" code block, add after the `crates/fast-mesh-store/` line:

```
crates/fast-mesh-bvh/    BVH over a store mesh: build, incremental update, box query.
```

and in the "Everyday commands" table add a row:

```
| `cargo bench -p fast-mesh-bvh` | BVH benchmarks on 1M triangles (`FAST_MESH_BENCH_LARGE=1` adds 10M); results go in `docs/benchmarks.md` |
```

- [ ] **Step 5: Record the deviations in the spec**

In `docs/superpowers/specs/2026-10-15-fast-mesh-bvh-design.md`:

1. In §4, replace `whose subtree holds at most `treelet_size` triangles (default 65,536)` with `whose subtree holds at most `treelet_size` triangles (default 4,096, chosen from benchmarks: see `docs/benchmarks.md`)`.
2. In §5, replace the line `bvh.update(&mesh, &changes) -> Result<UpdateReport, BvhError>` with `bvh.update(&mut mesh, &changes) -> Result<UpdateReport, BvhError>   // &mut: may build the mesh's incidence cache`.
3. At the end of §5, add the paragraph: `Manual `refit` and `rebuild_region` do not change the revision the BVH records; `update`, `build` and `rebuild` do. Full builds reserve an estimated node count and grow on demand; treelet rebuilds reserve their worst case (with at least 1/8 headroom) before mutating.`

- [ ] **Step 6: Full verification**

Run: `cargo fmt --all --check`
Expected: no output.

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: no warnings.

Run: `cargo test --workspace`
Expected: all pass (fast-mesh 4, store 60 + 3 property, bvh 33 + 1 property).

Run: `cargo build -p fast-mesh-bvh --target wasm32-unknown-unknown --release`
Expected: `Finished`.

Run: `cargo doc -p fast-mesh-bvh -p fast-mesh-store --no-deps`
Expected: no warnings.

- [ ] **Step 7: Commit**

```bash
git add crates/fast-mesh-bvh docs .github CONTRIBUTING.md
git commit -m "test(bvh): add property tests, benchmarks and docs"
```
