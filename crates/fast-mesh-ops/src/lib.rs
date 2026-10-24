//! Repair operations on meshes stored in [`fast_mesh_store`].
//!
//! Operations read the mesh through a transaction's `tx.mesh()`, build the
//! maps they need from its triangles, compute and validate their whole plan,
//! and only then write. Positions are read as the stored `f32` local
//! coordinates and all geometry is computed in `f64`, without the origin.
//! Every growable temporary is reserved fallibly, so that running out of
//! memory is [`OpsError::Store`] with [`StoreError::OutOfMemory`]. Results do
//! not depend on hash iteration order: every list is sorted explicitly.
//!
//! - Edges: an undirected edge is keyed `(min, max)`. It is a boundary edge
//!   when one triangle uses it, interior with two and non-manifold with three
//!   or more ([`EdgeKind`]). Triangles with a repeated vertex id are skipped.
//! - Components: triangles connect through shared edges of any kind
//!   ([`Component`]).
//! - Boundary loops: [`boundary_loops`] chains the boundary edges into
//!   [`BoundaryLoop`]s by walking each vertex's triangle fan, so that two holes
//!   touching at one vertex come out as two loops. Each loop has a content
//!   fingerprint, a [`LoopId`], that callers use to pick loops to fill.
//! - Analysis: [`analyze`] reports a mesh's defects without changing it, as
//!   an [`Analysis`]: loops, coincident vertices, degenerate and duplicate
//!   triangles, non-manifold edges and vertices, pinch vertices, components,
//!   orientation conflicts and, given a BVH, self-intersection.
//! - Orientation: [`orient()`] makes the winding consistent patch by patch
//!   and turns closed components outward.
//! - Hole filling: [`fill_holes`] closes the boundary loops chosen by size
//!   or by id with triangle patches that agree with their neighbours' winding
//!   and never make an edge non-manifold, refined to the surrounding density
//!   and faired to continue the surrounding curvature. Run [`orient()`]
//!   first on meshes that may contain flipped triangles: a hole whose rim
//!   crosses an orientation seam is not clean and is skipped.

mod analyze;
mod cleanup;
mod error;
mod fair;
mod fill;
mod fill_dp;
mod orient;
mod refine;
#[cfg(test)]
mod test_util;
mod topology;
mod weld;

pub use analyze::{Analysis, AnalyzeOptions, AnalyzeOptionsBuilder, analyze};
pub use cleanup::{
    ComponentOptions, ComponentOptionsBuilder, DegenerateOptions, DegenerateOptionsBuilder,
    DegenerateReport, RemoveReport, WeldReport, remove_degenerate, remove_small_components, weld,
};
pub use error::{OpsError, Result};
pub use fill::{
    FailReason, FairWeights, FillOptions, FillOptionsBuilder, FillReport, FilledLoop, SkipReason,
    fill_holes,
};
pub use orient::{OrientReport, orient};
pub use topology::{BoundaryLoop, Component, EdgeKind, LoopId, boundary_loops};

use fast_mesh_store::StoreError;

#[cfg(test)]
thread_local! {
    /// Test-only fault injection: number of fallible reservations that still succeed.
    static ALLOCATIONS_LEFT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Makes the `n`-th next fallible reservation on this thread (0-based) and every one
/// after it fail with `OutOfMemory`. `None` turns failures off. While armed,
/// parallel work runs on this thread so the hook sees every reservation.
#[cfg(test)]
pub(crate) fn fail_allocations_after(n: Option<usize>) {
    // Only on arming or disarming: the count must see one `true` per arm.
    let was = ALLOCATIONS_LEFT.with(|c| c.replace(n)).is_some();
    if was != n.is_some() {
        fast_mesh_store::par::set_force_sequential(n.is_some());
    }
}

/// Fails with `OutOfMemory` when the test fault hook says so; always succeeds
/// outside tests.
pub(crate) fn allocation_allowed() -> Result<()> {
    #[cfg(test)]
    {
        if let Some(n) = ALLOCATIONS_LEFT.with(|c| c.get()) {
            if n == 0 {
                return Err(OpsError::Store(StoreError::OutOfMemory));
            }
            ALLOCATIONS_LEFT.with(|c| c.set(Some(n - 1)));
        }
    }
    Ok(())
}

/// Reserves room for `additional` more elements, honouring the test fault hook.
pub(crate) fn reserve<T>(v: &mut Vec<T>, additional: usize) -> Result<()> {
    allocation_allowed()?;
    v.try_reserve(additional)
        .map_err(|_| OpsError::Store(StoreError::OutOfMemory))
}

/// Pushes with a fallible reservation.
pub(crate) fn push<T>(v: &mut Vec<T>, value: T) -> Result<()> {
    reserve(v, 1)?;
    v.push(value);
    Ok(())
}

/// A vector of `len` copies of `value`, reserved fallibly.
pub(crate) fn filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut v = Vec::new();
    reserve(&mut v, len)?;
    v.resize(len, value);
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{damaged_sphere, edit, grid_without};

    #[test]
    fn fault_hook_fails_the_nth_reservation() {
        let mut v: Vec<u32> = Vec::new();
        fail_allocations_after(Some(1));
        assert_eq!(push(&mut v, 1), Ok(()));
        assert_eq!(
            push(&mut v, 2),
            Err(OpsError::Store(StoreError::OutOfMemory))
        );
        fail_allocations_after(None);
        assert_eq!(v, vec![1]);
        assert_eq!(filled(3, 7u8), Ok(vec![7, 7, 7]));
    }

    #[test]
    /// Small inputs to every parallel site of this crate (analyze, weld,
    /// fairing) take no parallel path.
    fn small_inputs_stay_sequential() {
        use fast_mesh_bvh::{BuildOptions, Bvh};
        use fast_mesh_store::par;
        let before = par::parallel_entries();
        let damaged = damaged_sphere();
        let bvh = Bvh::build(&damaged, &BuildOptions::default()).unwrap();
        let options = AnalyzeOptions {
            weld_tolerance: 0.01,
            self_intersection: Some(&bvh),
            ..Default::default()
        };
        analyze(&damaged, &options).unwrap();
        let mut mesh = damaged_sphere();
        edit(&mut mesh, |tx| weld(tx, 0.01));
        // A 4 x 4 hole, faired.
        let cells: Vec<(u32, u32)> = (2..6).flat_map(|y| (2..6).map(move |x| (x, y))).collect();
        let mut mesh = grid_without(8, &cells);
        let options = FillOptions {
            max_edges: Some(20),
            ..FillOptions::default()
        };
        let (r, _) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert!(r.filled.len() == 1 && r.filled[0].weights.is_some());
        assert_eq!(par::parallel_entries(), before);
    }
}
