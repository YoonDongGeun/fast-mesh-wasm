//! Topological, geometric, and volumetric mesh repair operations and domain presets.
//!
//! - [`manifold::make_manifold`] repairs duplicate triangles, dangling fins,
//!   non-manifold edges, and pinch vertices.
//! - [`sliver::cleanup_slivers`] collapses needle triangles and flips cap
//!   triangles without inverting surface normals.
//! - [`self_intersect::resolve_self_intersections`] eliminates self-intersecting
//!   pairs by hole replacement, local cut-and-stitch remeshing, or voxel
//!   reconstruction.
//! - [`presets::AutoRepairPreset`] configures domain-tailored pipelines for
//!   [`auto_repair`](crate::auto_repair).

pub mod manifold;
pub mod presets;
pub mod self_intersect;
pub mod sliver;

pub use crate::analyze::{Analysis, AnalyzeOptions, AnalyzeOptionsBuilder, analyze};
pub use crate::auto_repair::{
    AutoRepairOptions, AutoRepairOptionsBuilder, AutoRepairReport, AutoRepairStage, auto_repair,
};
pub use crate::bridge::{
    BRIDGE_DEFAULT_MAX_EDGES, BridgeOptions, BridgeOptionsBuilder, BridgeReport, bridge_loops,
};
pub use crate::cleanup::{
    ComponentOptions, ComponentOptionsBuilder, DegenerateOptions, DegenerateOptionsBuilder,
    DegenerateReport, RemoveReport, WeldReport, remove_degenerate, remove_small_components, weld,
};
pub use crate::fill::{
    FailReason, FairWeights, FillOptions, FillOptionsBuilder, FillReport, FilledLoop, SkipReason,
    fill_holes,
};
pub use crate::orient::{OrientReport, orient};
pub use manifold::{ManifoldOptions, ManifoldOptionsBuilder, ManifoldReport, make_manifold};
pub use presets::AutoRepairPreset;
pub use self_intersect::{
    SelfIntersectOptions, SelfIntersectOptionsBuilder, SelfIntersectReport, SelfIntersectionMode,
    resolve_self_intersections,
};
pub use sliver::{SliverOptions, SliverOptionsBuilder, SliverReport, cleanup_slivers};

/// Euclidean length of a 3D `f64` vector.
#[inline]
pub(super) fn norm64(v: [f64; 3]) -> f64 {
    crate::topology::dot(v, v).sqrt()
}
