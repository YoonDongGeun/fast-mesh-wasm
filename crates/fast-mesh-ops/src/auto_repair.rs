//! The standard cleanup pipeline as one call.

use fast_mesh_fields::DEFAULT_MAX_VOXELS;
use fast_mesh_store::{Mesh, Transaction};

use crate::repair::manifold::{ManifoldOptions, ManifoldReport, make_manifold};
use crate::repair::presets::AutoRepairPreset;
use crate::repair::self_intersect::{
    SelfIntersectOptions, SelfIntersectReport, SelfIntersectionMode, find_self_intersecting_pairs,
    resolve_self_intersections, voxel_heal_surface,
};
use crate::repair::sliver::{SliverOptions, SliverReport, cleanup_slivers};
use crate::topology::{EdgeMap, pinch_vertices};
use crate::{
    ComponentOptions, DegenerateOptions, DegenerateReport, FillOptions, FillReport, OrientReport,
    RemoveReport, Result, WeldReport, push,
};

/// Options for [`auto_repair()`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoRepairOptions {
    /// Vertices at most this far apart are welded, in local units. Default
    /// `1e-5`.
    pub weld_tolerance: f32,
    /// Components with fewer triangles are removed; the largest component
    /// always stays. Default `500`. (`0` skips small-component pruning.)
    pub min_triangles: usize,
    /// Fill only holes with at most this many edges; `None` fills any size,
    /// an open mesh's outer boundary included (under [`AutoRepairPreset::CadImport`],
    /// `None` leaves boundaries open). Default `Some(64)`.
    pub max_edges: Option<usize>,
    /// Refine and fair each fill so that it continues the surrounding
    /// curvature ([`FillOptions::fair`]). Default `true`.
    pub fair: bool,
    /// Domain preset configuring pipeline stages and default parameters.
    /// Default [`AutoRepairPreset::General`].
    pub preset: AutoRepairPreset,
    /// Optional self-intersection resolution override. When `None`, follows
    /// the preset default (`None` for `General`/`Scan3D`/`CadImport`,
    /// `DeleteAndFill` with `VoxelHeal` fallback for `Print3D`).
    pub self_intersection_mode: Option<SelfIntersectionMode>,
    /// Optional explicit voxel pitch for [`SelfIntersectionMode::VoxelHeal`]
    /// (including `Print3D` volumetric fallback).
    pub voxel_size: Option<f32>,
    /// Maximum voxel budget for volumetric healing. Default [`DEFAULT_MAX_VOXELS`].
    pub max_voxels: usize,
}

impl Default for AutoRepairOptions {
    fn default() -> Self {
        Self {
            weld_tolerance: 1e-5,
            min_triangles: 500,
            max_edges: Some(64),
            fair: true,
            preset: AutoRepairPreset::General,
            self_intersection_mode: None,
            voxel_size: None,
            max_voxels: DEFAULT_MAX_VOXELS,
        }
    }
}

impl AutoRepairOptions {
    /// Options initialized with the defaults of `preset`.
    pub fn for_preset(preset: AutoRepairPreset) -> Self {
        preset.default_options()
    }

    /// A builder for [`AutoRepairOptions`]: every option unset, as
    /// [`Default`], until set.
    pub fn builder() -> AutoRepairOptionsBuilder {
        AutoRepairOptionsBuilder::default()
    }
}

/// A builder for [`AutoRepairOptions`], see [`AutoRepairOptions::builder`].
#[derive(Debug, Clone, Copy, Default)]
pub struct AutoRepairOptionsBuilder {
    options: AutoRepairOptions,
    weld_set: bool,
    min_set: bool,
    max_set: bool,
    fair_set: bool,
    si_set: bool,
    voxel_set: bool,
    max_voxels_set: bool,
}

impl AutoRepairOptionsBuilder {
    /// Selects a domain [`AutoRepairPreset`], applying its defaults to any
    /// unset option while preserving previously overridden fields.
    pub fn preset(mut self, preset: AutoRepairPreset) -> Self {
        let mut base = preset.default_options();
        if self.weld_set {
            base.weld_tolerance = self.options.weld_tolerance;
        }
        if self.min_set {
            base.min_triangles = self.options.min_triangles;
        }
        if self.max_set {
            base.max_edges = self.options.max_edges;
        }
        if self.fair_set {
            base.fair = self.options.fair;
        }
        if self.si_set {
            base.self_intersection_mode = self.options.self_intersection_mode;
        }
        if self.voxel_set {
            base.voxel_size = self.options.voxel_size;
        }
        if self.max_voxels_set {
            base.max_voxels = self.options.max_voxels;
        }
        self.options = base;
        self
    }

    /// Vertices at most this far apart are welded, in local units.
    pub fn weld_tolerance(mut self, weld_tolerance: f32) -> Self {
        self.options.weld_tolerance = weld_tolerance;
        self.weld_set = true;
        self
    }

    /// Components with fewer triangles are removed.
    pub fn min_triangles(mut self, min_triangles: usize) -> Self {
        self.options.min_triangles = min_triangles;
        self.min_set = true;
        self
    }

    /// Fill only holes with at most this many edges; `None` fills any size.
    pub fn max_edges(mut self, max_edges: Option<usize>) -> Self {
        self.options.max_edges = max_edges;
        self.max_set = true;
        self
    }

    /// Whether to refine and fair each fill.
    pub fn fair(mut self, fair: bool) -> Self {
        self.options.fair = fair;
        self.fair_set = true;
        self
    }

    /// Override self-intersection resolution mode.
    pub fn self_intersection_mode(mut self, mode: Option<SelfIntersectionMode>) -> Self {
        self.options.self_intersection_mode = mode;
        self.si_set = true;
        self
    }

    /// Explicit voxel size for [`SelfIntersectionMode::VoxelHeal`].
    pub fn voxel_size(mut self, voxel_size: Option<f32>) -> Self {
        self.options.voxel_size = voxel_size;
        self.voxel_set = true;
        self
    }

    /// Maximum voxel budget for [`SelfIntersectionMode::VoxelHeal`].
    pub fn max_voxels(mut self, max_voxels: usize) -> Self {
        self.options.max_voxels = max_voxels;
        self.max_voxels_set = true;
        self
    }

    /// The options as set.
    pub fn build(self) -> AutoRepairOptions {
        self.options
    }
}

/// What [`auto_repair()`] did, stage by stage.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AutoRepairReport {
    /// The weld stage.
    pub weld: WeldReport,
    /// The remove-degenerate stage.
    pub degenerate: DegenerateReport,
    /// The orient stage.
    pub orient: OrientReport,
    /// The remove-small-components stage.
    pub components: RemoveReport,
    /// The fill-holes stage.
    pub fill: FillReport,
    /// Non-manifold healing stage, when run.
    pub manifold: Option<ManifoldReport>,
    /// Sliver cleanup stage, when run.
    pub slivers: Option<SliverReport>,
    /// Self-intersection resolution stage, when run.
    pub self_intersections: Option<SelfIntersectReport>,
}

/// One stage's report in an [`AutoRepairReport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoRepairStage<'a> {
    /// The weld stage.
    Weld(&'a WeldReport),
    /// The remove-degenerate stage.
    Degenerate(&'a DegenerateReport),
    /// The orient stage.
    Orient(&'a OrientReport),
    /// The remove-small-components stage.
    Components(&'a RemoveReport),
    /// The fill-holes stage.
    Fill(&'a FillReport),
    /// The make-manifold stage.
    Manifold(&'a ManifoldReport),
    /// The cleanup-slivers stage.
    Slivers(&'a SliverReport),
    /// The resolve-self-intersections stage.
    SelfIntersections(&'a SelfIntersectReport),
}

impl AutoRepairReport {
    /// The core five stages' reports in legacy pipeline order.
    pub fn stages(&self) -> [AutoRepairStage<'_>; 5] {
        [
            AutoRepairStage::Weld(&self.weld),
            AutoRepairStage::Degenerate(&self.degenerate),
            AutoRepairStage::Orient(&self.orient),
            AutoRepairStage::Components(&self.components),
            AutoRepairStage::Fill(&self.fill),
        ]
    }

    /// Every stage report that ran for `options`, in execution order.
    pub fn executed_stages(&self, options: &AutoRepairOptions) -> Result<Vec<AutoRepairStage<'_>>> {
        let mut out = Vec::new();
        push(&mut out, AutoRepairStage::Weld(&self.weld))?;
        push(&mut out, AutoRepairStage::Degenerate(&self.degenerate))?;
        match options.preset {
            AutoRepairPreset::General => {
                if let Some(m) = &self.manifold {
                    push(&mut out, AutoRepairStage::Manifold(m))?;
                }
                push(&mut out, AutoRepairStage::Orient(&self.orient))?;
                push(&mut out, AutoRepairStage::Components(&self.components))?;
                if let Some(s) = &self.slivers {
                    push(&mut out, AutoRepairStage::Slivers(s))?;
                }
                push(&mut out, AutoRepairStage::Fill(&self.fill))?;
                if let Some(si) = &self.self_intersections {
                    push(&mut out, AutoRepairStage::SelfIntersections(si))?;
                }
            }
            AutoRepairPreset::Scan3D => {
                if let Some(m) = &self.manifold {
                    push(&mut out, AutoRepairStage::Manifold(m))?;
                }
                push(&mut out, AutoRepairStage::Components(&self.components))?;
                push(&mut out, AutoRepairStage::Orient(&self.orient))?;
                if let Some(s) = &self.slivers {
                    push(&mut out, AutoRepairStage::Slivers(s))?;
                }
                push(&mut out, AutoRepairStage::Fill(&self.fill))?;
                if let Some(si) = &self.self_intersections {
                    push(&mut out, AutoRepairStage::SelfIntersections(si))?;
                }
            }
            AutoRepairPreset::Print3D => {
                if let Some(m) = &self.manifold {
                    push(&mut out, AutoRepairStage::Manifold(m))?;
                }
                push(&mut out, AutoRepairStage::Components(&self.components))?;
                push(&mut out, AutoRepairStage::Orient(&self.orient))?;
                push(&mut out, AutoRepairStage::Fill(&self.fill))?;
                if let Some(si) = &self.self_intersections {
                    push(&mut out, AutoRepairStage::SelfIntersections(si))?;
                }
            }
            AutoRepairPreset::CadImport => {
                if let Some(m) = &self.manifold {
                    push(&mut out, AutoRepairStage::Manifold(m))?;
                }
                push(&mut out, AutoRepairStage::Orient(&self.orient))?;
                if let Some(s) = &self.slivers {
                    push(&mut out, AutoRepairStage::Slivers(s))?;
                }
                if options.max_edges.is_some()
                    || !self.fill.filled.is_empty()
                    || !self.fill.skipped.is_empty()
                    || !self.fill.failed.is_empty()
                {
                    push(&mut out, AutoRepairStage::Fill(&self.fill))?;
                }
                if let Some(si) = &self.self_intersections {
                    push(&mut out, AutoRepairStage::SelfIntersections(si))?;
                }
            }
        }
        Ok(out)
    }
}

fn is_closed_manifold(mesh: &Mesh) -> Result<bool> {
    if mesh.triangle_count() == 0 {
        return Ok(false);
    }
    let edges = EdgeMap::build(mesh)?;
    if edges.len() == 0 {
        return Ok(false);
    }
    for (_, tris) in edges.edges() {
        if tris.len() != 2 {
            return Ok(false);
        }
    }
    Ok(pinch_vertices(mesh, &edges)?.is_empty())
}

fn run_small_components(tx: &mut Transaction, min_triangles: usize) -> Result<RemoveReport> {
    if min_triangles == 0 {
        return Ok(RemoveReport::default());
    }
    crate::remove_small_components(
        tx,
        &ComponentOptions {
            min_triangles: Some(min_triangles),
            ..ComponentOptions::default()
        },
    )
}

fn run_fill(tx: &mut Transaction, max_edges: Option<usize>, fair: bool) -> Result<FillReport> {
    crate::fill_holes(
        tx,
        &FillOptions {
            // The fill needs a bound: "any size" is one that every loop meets.
            max_edges: Some(max_edges.unwrap_or(usize::MAX)),
            fair,
            ..FillOptions::default()
        },
    )
}

fn run_self_intersection_stage(
    tx: &mut Transaction,
    options: &AutoRepairOptions,
    mode: SelfIntersectionMode,
) -> Result<SelfIntersectReport> {
    resolve_self_intersections(
        tx,
        None,
        &SelfIntersectOptions {
            mode,
            voxel_size: options.voxel_size,
            max_voxels: options.max_voxels,
            max_hole_edges: options.max_edges,
            fair: options.fair,
            ..SelfIntersectOptions::default()
        },
    )
}

/// Repairs a mesh with either the backward-compatible cleanup pipeline
/// ([`AutoRepairPreset::General`]) or a domain-tailored preset (`Scan3D`,
/// `Print3D`, `CadImport`), all in `tx`, so committing it makes one undo step.
///
/// The order matters: welding stitches coincident vertices and can leave
/// duplicate triangles for the next stage, orienting makes hole rims clean
/// for filling, and small components go before filling so that their holes
/// are not filled for nothing. Holes past `max_edges` are skipped and holes
/// that cannot be filled are listed in [`FillReport`]; neither is an error.
///
/// On a mesh with UVs a seam stays a seam: the weld keeps its two sides apart,
/// UV islands of one surface count as one component, and loops with a seam
/// edge are skipped with [`SkipReason::UvSeam`](crate::SkipReason::UvSeam).
/// New fill vertices interpolate their rim's UVs.
///
/// ```
/// use fast_mesh_ops::{AutoRepairOptions, auto_repair};
/// use fast_mesh_store::Mesh;
///
/// // Two triangles of a unit square, with the shared corners duplicated.
/// let positions = [
///     0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, // first triangle
///     0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, // second triangle
/// ];
/// let mut mesh = Mesh::from_buffers(&positions, &[0, 1, 2, 3, 4, 5]).unwrap();
///
/// let mut tx = mesh.begin_edit();
/// let report = auto_repair(&mut tx, &AutoRepairOptions::default()).unwrap();
/// tx.commit().unwrap();
/// assert_eq!(report.weld.merged_vertices, 2);
/// ```
///
/// # Errors
/// [`OpsError::InvalidInput`](crate::OpsError::InvalidInput) when
/// `weld_tolerance` is negative, NaN or infinite;
/// [`OpsError::Store`](crate::OpsError::Store) when memory runs out. Earlier
/// stages have written to `tx` by then: drop it, which rolls all of them
/// back, instead of committing.
pub fn auto_repair(tx: &mut Transaction, options: &AutoRepairOptions) -> Result<AutoRepairReport> {
    let weld = crate::weld(tx, options.weld_tolerance)?;
    let degenerate = crate::remove_degenerate(tx, &DegenerateOptions::default())?;

    match options.preset {
        AutoRepairPreset::General => {
            let orient = crate::orient(tx)?;
            let components = run_small_components(tx, options.min_triangles)?;
            let fill = run_fill(tx, options.max_edges, options.fair)?;
            let self_intersections = match options.self_intersection_mode {
                Some(mode) if mode != SelfIntersectionMode::None => {
                    Some(run_self_intersection_stage(tx, options, mode)?)
                }
                _ => None,
            };
            Ok(AutoRepairReport {
                weld,
                degenerate,
                orient,
                components,
                fill,
                manifold: None,
                slivers: None,
                self_intersections,
            })
        }
        AutoRepairPreset::Scan3D => {
            let manifold = make_manifold(tx, &ManifoldOptions::default())?;
            let components = run_small_components(tx, options.min_triangles)?;
            let orient = crate::orient(tx)?;
            let slivers = cleanup_slivers(tx, &SliverOptions::default())?;
            let fill = run_fill(tx, options.max_edges, options.fair)?;
            let self_intersections = match options.self_intersection_mode {
                Some(mode) if mode != SelfIntersectionMode::None => {
                    Some(run_self_intersection_stage(tx, options, mode)?)
                }
                _ => None,
            };
            Ok(AutoRepairReport {
                weld,
                degenerate,
                orient,
                components,
                fill,
                manifold: Some(manifold),
                slivers: Some(slivers),
                self_intersections,
            })
        }
        AutoRepairPreset::Print3D => {
            let manifold = make_manifold(tx, &ManifoldOptions::default())?;
            let components = run_small_components(tx, options.min_triangles)?;
            let mut orient_r = crate::orient(tx)?;
            let fill = run_fill(tx, options.max_edges, options.fair)?;
            let post_fill_orient = crate::orient(tx)?;
            orient_r.flipped_triangles += post_fill_orient.flipped_triangles;
            orient_r.flipped_outward_components += post_fill_orient.flipped_outward_components;

            let mode = options
                .self_intersection_mode
                .unwrap_or(SelfIntersectionMode::DeleteAndFill);
            let self_intersections = if mode == SelfIntersectionMode::None {
                Some(run_self_intersection_stage(tx, options, mode)?)
            } else {
                let mut si = run_self_intersection_stage(tx, options, mode)?;
                let needs_voxel_fallback = mode != SelfIntersectionMode::VoxelHeal
                    && (si.remaining_pairs > 0 || !is_closed_manifold(tx.mesh())?)
                    && tx.mesh().triangle_count() > 0;
                if needs_voxel_fallback {
                    let (r_vox, a_vox, dropped) =
                        voxel_heal_surface(tx, options.voxel_size, options.max_voxels)?;
                    si.removed_triangles += r_vox;
                    si.added_triangles += a_vox;
                    si.uvs_dropped |= dropped;
                    si.mode_used = SelfIntersectionMode::VoxelHeal;
                    si.remaining_pairs = find_self_intersecting_pairs(tx.mesh(), None)?.len();
                }
                let final_orient = crate::orient(tx)?;
                orient_r.flipped_triangles += final_orient.flipped_triangles;
                orient_r.flipped_outward_components += final_orient.flipped_outward_components;
                Some(si)
            };

            Ok(AutoRepairReport {
                weld,
                degenerate,
                orient: orient_r,
                components,
                fill,
                manifold: Some(manifold),
                slivers: None,
                self_intersections,
            })
        }
        AutoRepairPreset::CadImport => {
            let manifold = make_manifold(tx, &ManifoldOptions::default())?;
            let components = run_small_components(tx, options.min_triangles)?;
            let orient = crate::orient(tx)?;
            let slivers = cleanup_slivers(tx, &SliverOptions::default())?;
            let fill = if let Some(limit) = options.max_edges {
                run_fill(tx, Some(limit), false)?
            } else {
                FillReport::default()
            };
            let self_intersections = match options.self_intersection_mode {
                Some(mode) if mode != SelfIntersectionMode::None => {
                    Some(run_self_intersection_stage(tx, options, mode)?)
                }
                _ => None,
            };
            Ok(AutoRepairReport {
                weld,
                degenerate,
                orient,
                components,
                fill,
                manifold: Some(manifold),
                slivers: Some(slivers),
                self_intersections,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{damaged_sphere, edit, grid_without, mesh_of, parts};
    use crate::{OpsError, fail_allocations_after, weld};
    use fast_mesh_store::StoreError;

    /// Small enough that the sphere stays and the box floater goes.
    fn options() -> AutoRepairOptions {
        AutoRepairOptions {
            weld_tolerance: 1e-4,
            min_triangles: 10,
            ..AutoRepairOptions::default()
        }
    }

    #[test]
    fn defaults_and_builder() {
        let d = AutoRepairOptions::default();
        assert_eq!(d.weld_tolerance, 1e-5);
        assert_eq!(d.min_triangles, 500);
        assert_eq!(d.max_edges, Some(64));
        assert!(d.fair);
        assert_eq!(AutoRepairOptions::builder().build(), d);
        assert_eq!(
            AutoRepairOptions::builder()
                .weld_tolerance(0.5)
                .min_triangles(3)
                .max_edges(None)
                .fair(false)
                .build(),
            AutoRepairOptions {
                weld_tolerance: 0.5,
                min_triangles: 3,
                max_edges: None,
                fair: false,
                ..AutoRepairOptions::default()
            }
        );
    }

    #[test]
    fn matches_the_ops_run_by_hand_in_order() {
        let o = options();
        let (mut piped, mut by_hand) = (damaged_sphere(), damaged_sphere());
        let (r, _) = edit(&mut piped, |tx| auto_repair(tx, &o));
        let (expected, _) = edit(&mut by_hand, |tx| {
            Ok(AutoRepairReport {
                weld: weld(tx, o.weld_tolerance)?,
                degenerate: crate::remove_degenerate(tx, &DegenerateOptions::default())?,
                orient: crate::orient(tx)?,
                components: crate::remove_small_components(
                    tx,
                    &ComponentOptions {
                        min_triangles: Some(o.min_triangles),
                        keep_largest: true,
                        min_area: None,
                    },
                )?,
                fill: crate::fill_holes(
                    tx,
                    &FillOptions {
                        max_edges: Some(64),
                        ..FillOptions::default()
                    },
                )?,
                ..AutoRepairReport::default()
            })
        });
        assert_eq!(r, expected);
        assert_eq!(parts(&piped), parts(&by_hand));
        // Every stage had work on this mesh.
        assert!(r.weld.merged_vertices > 0);
        assert!(r.degenerate.removed_triangles > 0);
        assert!(r.orient.flipped_triangles > 0);
        assert_eq!(r.components.removed_components, 1);
        assert!(!r.fill.filled.is_empty());
    }

    #[test]
    fn weld_runs_before_remove_degenerate() {
        // The same triangle twice on separate vertices: a duplicate only once welded.
        let corners = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let positions: Vec<f32> = corners.iter().chain(&corners).copied().collect();
        let mut mesh = mesh_of(&positions, &[[0, 1, 2], [3, 4, 5]]);
        let (r, _) = edit(&mut mesh, |tx| auto_repair(tx, &options()));
        assert_eq!(r.weld.merged_vertices, 3);
        assert_eq!(r.degenerate.removed_triangles, 1);
    }

    #[test]
    fn stages_come_in_run_order() {
        let r = AutoRepairReport::default();
        assert_eq!(
            r.stages(),
            [
                AutoRepairStage::Weld(&r.weld),
                AutoRepairStage::Degenerate(&r.degenerate),
                AutoRepairStage::Orient(&r.orient),
                AutoRepairStage::Components(&r.components),
                AutoRepairStage::Fill(&r.fill),
            ]
        );
    }

    #[test]
    fn repaired_mesh_needs_nothing_more() {
        let mut mesh = damaged_sphere();
        edit(&mut mesh, |tx| auto_repair(tx, &options()));
        let (again, empty) = edit(&mut mesh, |tx| auto_repair(tx, &options()));
        assert!(empty);
        assert_eq!(again.weld, WeldReport::default());
        assert_eq!(again.degenerate, DegenerateReport::default());
        assert_eq!(again.orient.flipped_triangles, 0);
        assert_eq!(again.components, RemoveReport::default());
        assert_eq!(again.fill, FillReport::default());
    }

    #[test]
    fn max_edges_bounds_the_fill_and_none_means_any_size() {
        // One 4 x 4 hole (16 edges) plus the grid's outer boundary (32 edges).
        let cells: Vec<(u32, u32)> = (2..6).flat_map(|y| (2..6).map(move |x| (x, y))).collect();
        let run = |max_edges| {
            let mut mesh = grid_without(8, &cells);
            let o = AutoRepairOptions {
                max_edges,
                fair: false,
                ..AutoRepairOptions::default()
            };
            edit(&mut mesh, |tx| auto_repair(tx, &o)).0.fill
        };
        let (smallest, inner, any) = (run(Some(4)), run(Some(16)), run(None));
        assert_eq!((smallest.filled.len(), smallest.skipped.len()), (0, 2));
        assert_eq!((inner.filled.len(), inner.skipped.len()), (1, 1));
        // None selects the outer boundary too (its collinear rim may not fill).
        assert!(any.skipped.is_empty());
        assert_eq!(any.filled.len() + any.failed.len(), 2);
    }

    #[test]
    fn min_triangles_keeps_the_largest_component() {
        // Every component is below the cutoff: the largest still stays.
        let o = AutoRepairOptions {
            min_triangles: 1_000_000,
            ..options()
        };
        let mut mesh = damaged_sphere();
        let (r, _) = edit(&mut mesh, |tx| auto_repair(tx, &o));
        assert_eq!(r.components.removed_components, 1);
        assert!(mesh.triangle_count() > 2);
    }

    #[test]
    fn invalid_tolerance_writes_nothing() {
        for bad in [-1.0, f32::NAN, f32::INFINITY] {
            let mut mesh = damaged_sphere();
            let before = parts(&mesh);
            let mut tx = mesh.begin_edit();
            let o = AutoRepairOptions {
                weld_tolerance: bad,
                ..options()
            };
            assert!(matches!(
                auto_repair(&mut tx, &o),
                Err(OpsError::InvalidInput(_))
            ));
            assert!(tx.commit().unwrap().is_empty());
            assert_eq!(parts(&mesh), before);
        }
    }

    #[test]
    fn a_failure_at_any_stage_rolls_back_when_dropped() {
        let mut mesh = damaged_sphere();
        let before = parts(&mesh);
        let mut n = 0;
        loop {
            let mut tx = mesh.begin_edit();
            fail_allocations_after(Some(n));
            let r = auto_repair(&mut tx, &options());
            fail_allocations_after(None);
            if let Ok(report) = &r {
                // Past every failure point: the sweep reached the last stage.
                assert!(report.fill.filled.len() > 1);
                assert!(n > 5);
                break;
            }
            assert_eq!(r, Err(OpsError::Store(StoreError::OutOfMemory)));
            drop(tx);
            assert_eq!(parts(&mesh), before, "failure point {n}");
            n += 1;
        }
    }

    #[test]
    fn preset_scan3d_heals_manifold_slivers_and_holes() {
        let mut mesh = damaged_sphere();
        // Also attach a dangling non-manifold fin to verify Scan3D invokes make_manifold.
        edit(&mut mesh, |tx| {
            let v = tx.add_vertices(&[[0.0, 3.0, 0.0]])?;
            tx.add_triangles(&[[10, 11, v[0]]])?;
            Ok(())
        });
        let opts = AutoRepairOptions::for_preset(AutoRepairPreset::Scan3D);
        let (r, _) = edit(&mut mesh, |tx| auto_repair(tx, &opts));
        assert!(r.manifold.is_some());
        assert!(r.slivers.is_some());
        assert!(r.self_intersections.is_none());
        let a = crate::analyze::analyze(&mesh, &crate::analyze::AnalyzeOptions::default()).unwrap();
        assert_eq!(a.non_manifold_edges, 0);
        assert!(a.pinch_vertices.is_empty());
        assert!(a.loops.is_empty());
    }

    #[test]
    fn preset_print3d_produces_watertight_self_intersection_free_mesh() {
        let mut mesh = crate::test_util::crossed_boxes();
        let opts = AutoRepairOptions::builder()
            .preset(AutoRepairPreset::Print3D)
            .voxel_size(Some(0.15))
            .build();
        let (r, _) = edit(&mut mesh, |tx| auto_repair(tx, &opts));
        assert!(r.manifold.is_some());
        let si = r
            .self_intersections
            .expect("Print3D resolves self-intersections");
        assert_eq!(si.remaining_pairs, 0);
        let bvh =
            fast_mesh_bvh::Bvh::build(&mesh, &fast_mesh_bvh::BuildOptions::default()).unwrap();
        let a = crate::analyze::analyze(
            &mesh,
            &crate::analyze::AnalyzeOptions {
                self_intersection: Some(&bvh),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(a.non_manifold_edges, 0);
        assert!(a.pinch_vertices.is_empty());
        assert!(a.loops.is_empty());
        assert_eq!(a.self_intersects, Some(false));
    }

    #[test]
    fn preset_cad_import_preserves_open_boundaries_and_respects_max_edges() {
        // Open box (top face removed = 4-edge boundary loop) + dangling fin.
        let cells: Vec<(u32, u32)> = (1..3).flat_map(|y| (1..3).map(move |x| (x, y))).collect();
        let mut open_grid = grid_without(6, &cells);
        let opts_open = AutoRepairOptions::for_preset(AutoRepairPreset::CadImport);
        let (r_open, _) = edit(&mut open_grid, |tx| auto_repair(tx, &opts_open));
        assert!(r_open.manifold.is_some());
        assert!(r_open.slivers.is_some());
        assert!(r_open.fill.filled.is_empty());
        // Both inner 8-edge hole and outer 24-edge boundary remain open.
        assert_eq!(
            crate::analyze::analyze(&open_grid, &crate::analyze::AnalyzeOptions::default())
                .unwrap()
                .loops
                .len(),
            2
        );

        // Bounded max_edges fills only the small inner hole, leaving outer boundary open.
        let opts_bounded = AutoRepairOptions::builder()
            .preset(AutoRepairPreset::CadImport)
            .max_edges(Some(10))
            .build();
        let (r_bounded, _) = edit(&mut open_grid, |tx| auto_repair(tx, &opts_bounded));
        assert_eq!(r_bounded.fill.filled.len(), 1);
        assert_eq!(
            crate::analyze::analyze(&open_grid, &crate::analyze::AnalyzeOptions::default())
                .unwrap()
                .loops
                .len(),
            1
        );
    }
}
