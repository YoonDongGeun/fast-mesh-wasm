//! Domain-specific repair presets for [`auto_repair`](crate::auto_repair).

use fast_mesh_fields::DEFAULT_MAX_VOXELS;

use super::self_intersect::SelfIntersectionMode;
use crate::auto_repair::AutoRepairOptions;

/// Domain-tailored repair pipeline configuration for [`auto_repair`](crate::auto_repair).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AutoRepairPreset {
    /// Backward-compatible general pipeline: weld, degenerate cleanup, small
    /// component removal, consistent orientation, and faired hole filling.
    #[default]
    General,
    /// Tuned for raw 3D scans: tight welding, degenerate removal, non-manifold
    /// healing, outlier removal, orientation, sliver cleanup, and faired hole
    /// filling.
    Scan3D,
    /// Tuned for watertight 3D printing: manifold healing, full hole filling,
    /// orientation, and self-intersection resolution (`DeleteAndFill` with
    /// automatic `VoxelHeal` fallback if intersections or open boundaries remain).
    Print3D,
    /// Tuned for CAD imports: ultra-tight welding, duplicate/degenerate cleanup,
    /// manifold healing, orientation, and sliver cleanup while preserving sharp
    /// features (no fairing; boundary openings left open unless bounded by `max_edges`).
    CadImport,
}

impl AutoRepairPreset {
    /// Default [`AutoRepairOptions`] configured for this domain preset.
    pub fn default_options(self) -> AutoRepairOptions {
        match self {
            Self::General => AutoRepairOptions {
                weld_tolerance: 1e-5,
                min_triangles: 500,
                max_edges: Some(64),
                fair: true,
                preset: Self::General,
                self_intersection_mode: None,
                voxel_size: None,
                max_voxels: DEFAULT_MAX_VOXELS,
            },
            Self::Scan3D => AutoRepairOptions {
                weld_tolerance: 1e-5,
                min_triangles: 500,
                max_edges: Some(128),
                fair: true,
                preset: Self::Scan3D,
                self_intersection_mode: None,
                voxel_size: None,
                max_voxels: DEFAULT_MAX_VOXELS,
            },
            Self::Print3D => AutoRepairOptions {
                weld_tolerance: 1e-5,
                min_triangles: 500,
                max_edges: None,
                fair: true,
                preset: Self::Print3D,
                self_intersection_mode: Some(SelfIntersectionMode::DeleteAndFill),
                voxel_size: None,
                max_voxels: DEFAULT_MAX_VOXELS,
            },
            Self::CadImport => AutoRepairOptions {
                weld_tolerance: 1e-7,
                min_triangles: 0,
                max_edges: None,
                fair: false,
                preset: Self::CadImport,
                self_intersection_mode: None,
                voxel_size: None,
                max_voxels: DEFAULT_MAX_VOXELS,
            },
        }
    }
}
