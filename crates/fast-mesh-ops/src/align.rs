//! Mesh alignment domain: rigid point-to-plane ICP registration and surface sampling.
//!
//! - [`icp`] aligns a moving point cloud against a target mesh and its BVH via
//!   trimmed point-to-plane iterative closest point.
//! - [`sample_surface`] draws deterministic stratified surface samples from a mesh.

pub use crate::icp::{IcpMotion, IcpOptions, IcpReport, IcpStatus, icp, sample_surface};
