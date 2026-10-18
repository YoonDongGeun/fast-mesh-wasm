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
//!
//! Everything works in the mesh's local `f32` coordinates. Kernels allocate
//! only while building [`SignData`], never per point.

mod distance;
mod error;
mod geom;
mod sign;
#[cfg(test)]
mod test_util;
mod winding;

pub use distance::signed_distances;
pub use error::{FieldsError, Result};
pub use sign::{SignData, SignMode, SignStatus};
