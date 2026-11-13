//! Field kernels for meshes stored in [`fast_mesh_store`]: signed distances
//! from points to a mesh, accelerated by its [`fast_mesh_bvh`] BVH and the
//! closest-point query of [`fast_mesh_query`].
//!
//! - [`SignData`] holds what decides the sign of a distance for one mesh. It is
//!   built once per mesh revision by [`SignData::build`] in a [`SignMode`], and
//!   [`SignData::status`] reports the mesh's topology (closed or open,
//!   consistently oriented or not, non-manifold vertices, zero-area
//!   triangles) so that an application can warn about doubtful signs.
//!   [`SignData::winding_number`] gives the generalized winding number, fast
//!   in [`SignMode::Winding`] (Barill et al. 2018).
//! - [`signed_distances`] writes, for every point, its distance to the mesh:
//!   positive outside (in front of the surface), negative inside (behind it),
//!   zero on it, and `+∞` beyond `max_distance`.
//! - [`signed_distances_multi`] does the same against several [`Target`]
//!   meshes, each with its own placement, and keeps per point the smallest
//!   value and the target it came from.
//!
//! - [`color_of`] turns a field value into an RGBA8 colour under a
//!   [`ColorRule`] ([`Colormap`], range, optional centre, out-of-range rules).
//!
//! Everything works in the mesh's local `f32` coordinates. Kernels allocate
//! only while building [`SignData`], never per point (with the `parallel`
//! feature, large batches of points also get one buffer per chunk).
//!
//! With the `parallel` feature, [`SignData::build`] and large batches of
//! [`signed_distances`] and [`signed_distances_multi`] run on rayon's pool.
//! Every value is the same, bit for bit, with the feature on or off and for
//! any thread count.

mod color;
mod curvature;
mod distance;
mod edt;
mod error;
mod geom;
mod grid;
mod multi;
mod sign;
#[cfg(test)]
mod test_util;
mod thickness;
mod winding;

pub use color::{ColorRule, Colormap, OutOfRange, color_of};
pub use curvature::curvatures;
pub use distance::signed_distances;
pub use edt::{edt_3d_mask, edt_3d_sq, fh_signed_grid, fh_unsigned_grid};
pub use error::{FieldsError, Result};
pub use grid::{
    DEFAULT_AUTO_GRID_RES, DEFAULT_MAX_VOXELS, GridSpec, ScalarGrid, bvh_signed_grid,
    bvh_unsigned_grid,
};
pub use multi::{Target, signed_distances_multi};
pub use sign::{SignBuild, SignData, SignMode, SignStatus};
pub use thickness::thicknesses;
