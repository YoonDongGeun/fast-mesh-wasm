//! Queries on meshes stored in [`fast_mesh_store`], accelerated by [`fast_mesh_bvh`].
//!
//! Every query takes a mesh and a BVH built for it, works in the mesh's local
//! `f32` coordinates, and fails with [`QueryError::StaleBvh`] when the mesh was
//! edited after the BVH was last updated. Queries allocate only into the
//! output vector they are given.
//!
//! - Closest point: [`closest_point`] and [`closest_point_many`], with the
//!   [`Feature`] of the triangle the nearest point lies on.
//! - Raycasts: [`raycast`] (nearest hit), [`raycast_any`] (occlusion),
//!   [`raycast_all`] (every hit, sorted) and [`raycast_many`] (a batch). The
//!   ray-triangle test is watertight: a ray through a shared edge or vertex
//!   never slips between the triangles meeting there. Ties go to the smallest
//!   triangle id.
//! - Region queries: [`triangles_in`] and [`vertices_in`] find the triangles
//!   that touch, or the vertices inside, a [`Region`] (a box or a sphere). The
//!   box is an [`Aabb`], re-exported from `fast_mesh_bvh` so that callers need
//!   not depend on that crate to build a [`Region::Box`].
//! - Relative placement: a [`Rigid`] transform (rotation and translation, no
//!   scale) maps one mesh's local coordinates into another's, for queries
//!   between two meshes. Whether two triangles intersect or touch is decided
//!   exactly, by exact orientation predicates on the transformed `f64`
//!   coordinates.
//! - Mesh-vs-mesh queries, with mesh B placed in mesh A's space by a
//!   [`Rigid`]: [`intersects`], [`mesh_distance`] (a [`MeshDistance`] with
//!   both points in A's space), [`triangles_near`] (the triangles of each
//!   mesh within a margin of the other) and [`intersecting_pairs`]. Results
//!   are in A's space; passing the same mesh twice means two coincident
//!   copies.
//! - Cutting planes: a [`Plane`] (origin and unit normal, in `f64` mesh-local
//!   coordinates) with an orthonormal 2D frame `(u, v)`. A point is on its
//!   `+` side when its signed distance is zero or more.
//! - Sections: [`slice()`] cuts a mesh with a [`Plane`] into a [`Slice`] of
//!   [`Contour`]s, chained by crossed edge, with flags for open, unoriented
//!   and non-manifold pieces.
//! - Polygon fill: [`triangulate::triangulate`] splits a 2D polygon with
//!   holes into counter-clockwise triangles by ear clipping.

mod closest;
mod error;
mod pair;
mod pairs;
mod plane;
mod ray;
mod region;
mod rigid;
mod slice;
#[cfg(test)]
mod test_util;
mod tri;
pub mod triangulate;
mod tritri;

use fast_mesh_bvh::Bvh;
use fast_mesh_store::Mesh;

pub use closest::*;
pub use error::{QueryError, Result};
/// The axis-aligned box type of [`Region::Box`], re-exported from `fast_mesh_bvh`.
pub use fast_mesh_bvh::Aabb;
pub use pairs::*;
pub use plane::Plane;
pub use ray::*;
pub use region::*;
pub use rigid::Rigid;
pub use slice::{Contour, Slice, slice};
pub use tri::{Cull, Feature};

/// Checks that `bvh` was built for `mesh` and is up to date with it.
pub(crate) fn check(mesh: &Mesh, bvh: &Bvh) -> Result<()> {
    if bvh.mesh_uid() != mesh.uid() {
        Err(QueryError::MeshMismatch)
    } else if !bvh.is_current(mesh) {
        Err(QueryError::StaleBvh)
    } else {
        Ok(())
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only fault injection: number of fallible reservations that still succeed.
    static ALLOCATIONS_LEFT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Makes the `n`-th next fallible reservation on this thread (0-based) and every one
/// after it fail with `OutOfMemory`. `None` turns failures off.
#[cfg(test)]
pub(crate) fn fail_allocations_after(n: Option<usize>) {
    ALLOCATIONS_LEFT.with(|c| c.set(n));
}

fn allocation_allowed() -> Result<()> {
    #[cfg(test)]
    {
        if let Some(n) = ALLOCATIONS_LEFT.with(|c| c.get()) {
            if n == 0 {
                return Err(QueryError::OutOfMemory);
            }
            ALLOCATIONS_LEFT.with(|c| c.set(Some(n - 1)));
        }
    }
    Ok(())
}

/// Appends to a caller's vector; on failure it restores the vector's original length.
pub(crate) struct Appender<'a, T> {
    out: &'a mut Vec<T>,
    start: usize,
}

impl<'a, T> Appender<'a, T> {
    pub(crate) fn new(out: &'a mut Vec<T>) -> Self {
        let start = out.len();
        Appender { out, start }
    }

    /// Pushes with a fallible reservation. On failure everything appended so far is
    /// dropped and `OutOfMemory` is returned.
    pub(crate) fn push(&mut self, v: T) -> Result<()> {
        let reserved = allocation_allowed()
            .and_then(|()| self.out.try_reserve(1).map_err(|_| QueryError::OutOfMemory));
        if let Err(e) = reserved {
            self.out.truncate(self.start);
            return Err(e);
        }
        self.out.push(v);
        Ok(())
    }

    /// The elements appended through this appender.
    pub(crate) fn appended(&mut self) -> &mut [T] {
        &mut self.out[self.start..]
    }

    /// Drops all but the first `n` appended elements.
    pub(crate) fn truncate_appended(&mut self, n: usize) {
        self.out.truncate(self.start + n);
    }

    /// How many elements were appended.
    pub(crate) fn count(&self) -> usize {
        self.out.len() - self.start
    }

    /// Sorts the appended elements and removes duplicates among them, in
    /// place; elements from before stay as they are. Returns how many remain.
    pub(crate) fn sort_dedup_appended(&mut self) -> usize
    where
        T: Ord + Copy,
    {
        let found = self.appended();
        found.sort_unstable();
        // Compact the sorted run in place, keeping the first of each value.
        let mut kept = 0;
        for i in 0..found.len() {
            if kept == 0 || found[i] != found[kept - 1] {
                found[kept] = found[i];
                kept += 1;
            }
        }
        self.truncate_appended(kept);
        kept
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fast_mesh_bvh::{BuildOptions, Bvh};
    use fast_mesh_store::Mesh;

    fn mesh() -> Mesh {
        Mesh::from_buffers(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[0, 1, 2]).unwrap()
    }

    #[test]
    fn check_reports_mismatch_then_staleness() {
        let mut a = mesh();
        let b = mesh();
        let mut bvh = Bvh::build(&a, &BuildOptions::default()).unwrap();
        assert_eq!(check(&b, &bvh), Err(QueryError::MeshMismatch));
        assert_eq!(check(&a, &bvh), Ok(()));
        let mut tx = a.begin_edit();
        tx.set_positions(&[0], &[[0.0, 0.0, 1.0]]).unwrap();
        let p = tx.commit().unwrap();
        assert_eq!(check(&a, &bvh), Err(QueryError::StaleBvh));
        bvh.update(&mut a, p.changes()).unwrap();
        assert_eq!(check(&a, &bvh), Ok(()));
    }

    #[test]
    fn appender_restores_length_on_oom() {
        let mut out = vec![7u32];
        {
            let mut ap = Appender::new(&mut out);
            fail_allocations_after(Some(1));
            assert_eq!(ap.push(1), Ok(()));
            assert_eq!(ap.count(), 1);
            assert_eq!(ap.appended(), &[1]);
            assert_eq!(ap.push(2), Err(QueryError::OutOfMemory));
            fail_allocations_after(None);
        }
        assert_eq!(out, vec![7]);
    }
}
