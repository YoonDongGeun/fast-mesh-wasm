//! Mesh inspection domain: read-only defect analysis, scalar measurements,
//! topology queries, primitive fitting, section contours, and point clouds.
//!
//! - [`analyze`] inspects a mesh for boundary loops, coincident vertices,
//!   degenerate/duplicate triangles, non-manifold edges/vertices, pinch vertices,
//!   orientation conflicts, and optional BVH self-intersections.
//! - [`measure`] and [`measure_placed`] compute surface area, signed volume,
//!   and watertight enclosed volume.
//! - [`boundary_loops`] extracts topological boundary loops and open chains.
//! - [`primitive_fit`] provides RANSAC + Levenberg-Marquardt fitting for planes,
//!   spheres, cylinders, and cones, plus CAD datum alignment transforms.
//! - [`contours`] extracts planar section loops, 2D caliper/area metrology,
//!   and SVG/JSON/CSV exports.
//! - [`point_cloud`] provides XYZ/PCD/PLY parsers, outlier filters, voxel
//!   downsampling, consistent PCA+MST normal estimation, and surface reconstruction.

pub mod contours;
pub mod point_cloud;
pub mod primitive_fit;

pub use crate::analyze::{Analysis, AnalyzeOptions, AnalyzeOptionsBuilder, analyze};
pub use crate::measure::{MeasureReport, VolumeReason, measure, measure_placed};
pub use crate::topology::{BoundaryLoop, Component, EdgeKind, LoopId, boundary_loops};

pub use contours::{
    ContourLoop, ContourSlice, contours_to_csv, contours_to_json, contours_to_svg,
    slice_contour_stack, slice_contours,
};
pub use point_cloud::{
    OutlierReport, PointCloud, ReconstructMethod, SurfaceReconstructOptions, estimate_normals,
    parse_pcd_ascii, parse_ply_points, parse_xyz, radius_outlier_removal, reconstruct_surface,
    statistical_outlier_removal, voxel_downsample,
};
pub use primitive_fit::{
    FitPrimitiveOptions, FitPrimitiveReport, FittedPrimitive, PrimitiveKind, datum_alignment_rigid,
    fit_primitive_mesh, fit_primitive_points,
};
