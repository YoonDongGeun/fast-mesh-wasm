//! Mesh inspection domain: read-only defect analysis, scalar measurements, and topology queries.
//!
//! - [`analyze`] inspects a mesh for boundary loops, coincident vertices,
//!   degenerate/duplicate triangles, non-manifold edges/vertices, pinch vertices,
//!   orientation conflicts, and optional BVH self-intersections.
//! - [`measure`] and [`measure_placed`] compute surface area, signed volume,
//!   and watertight enclosed volume.
//! - [`boundary_loops`] extracts topological boundary loops and open chains.

pub use crate::analyze::{Analysis, AnalyzeOptions, AnalyzeOptionsBuilder, analyze};
pub use crate::measure::{MeasureReport, VolumeReason, measure, measure_placed};
pub use crate::topology::{BoundaryLoop, Component, EdgeKind, LoopId, boundary_loops};
