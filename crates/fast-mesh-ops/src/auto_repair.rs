//! The standard cleanup pipeline as one call.

use fast_mesh_store::Transaction;

use crate::{
    ComponentOptions, DegenerateOptions, DegenerateReport, FillOptions, FillReport, OrientReport,
    RemoveReport, Result, WeldReport,
};

/// Options for [`auto_repair()`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoRepairOptions {
    /// Vertices at most this far apart are welded, in local units. Default
    /// `1e-5`.
    pub weld_tolerance: f32,
    /// Components with fewer triangles are removed; the largest component
    /// always stays. Default `500`.
    pub min_triangles: usize,
    /// Fill only holes with at most this many edges; `None` fills any size,
    /// an open mesh's outer boundary included. Default `Some(64)`.
    pub max_edges: Option<usize>,
    /// Refine and fair each fill so that it continues the surrounding
    /// curvature ([`FillOptions::fair`]). Default `true`.
    pub fair: bool,
}

impl Default for AutoRepairOptions {
    fn default() -> Self {
        Self {
            weld_tolerance: 1e-5,
            min_triangles: 500,
            max_edges: Some(64),
            fair: true,
        }
    }
}

impl AutoRepairOptions {
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
}

impl AutoRepairOptionsBuilder {
    /// Vertices at most this far apart are welded, in local units.
    pub fn weld_tolerance(mut self, weld_tolerance: f32) -> Self {
        self.options.weld_tolerance = weld_tolerance;
        self
    }

    /// Components with fewer triangles are removed.
    pub fn min_triangles(mut self, min_triangles: usize) -> Self {
        self.options.min_triangles = min_triangles;
        self
    }

    /// Fill only holes with at most this many edges; `None` fills any size.
    pub fn max_edges(mut self, max_edges: Option<usize>) -> Self {
        self.options.max_edges = max_edges;
        self
    }

    /// Whether to refine and fair each fill.
    pub fn fair(mut self, fair: bool) -> Self {
        self.options.fair = fair;
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
}

impl AutoRepairReport {
    /// The stages' reports in the order they ran, for showing them without
    /// repeating that order.
    pub fn stages(&self) -> [AutoRepairStage<'_>; 5] {
        [
            AutoRepairStage::Weld(&self.weld),
            AutoRepairStage::Degenerate(&self.degenerate),
            AutoRepairStage::Orient(&self.orient),
            AutoRepairStage::Components(&self.components),
            AutoRepairStage::Fill(&self.fill),
        ]
    }
}

/// Repairs a mesh with the standard cleanup pipeline: [`weld()`](crate::weld)
/// → [`remove_degenerate()`](crate::remove_degenerate) →
/// [`orient()`](crate::orient) →
/// [`remove_small_components()`](crate::remove_small_components) →
/// [`fill_holes()`](crate::fill_holes), all in `tx`, so committing it makes
/// one undo step.
///
/// The order matters: welding stitches coincident vertices and can leave
/// duplicate triangles for the next stage, orienting makes hole rims clean
/// for filling, and small components go before filling so that their holes
/// are not filled for nothing. Holes past `max_edges` are skipped and holes
/// that cannot be filled are listed in [`FillReport`]; neither is an error.
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
    let orient = crate::orient(tx)?;
    let components = crate::remove_small_components(
        tx,
        &ComponentOptions {
            min_triangles: Some(options.min_triangles),
            ..ComponentOptions::default()
        },
    )?;
    let fill = crate::fill_holes(
        tx,
        &FillOptions {
            // The fill needs a bound: "any size" is one that every loop meets.
            max_edges: Some(options.max_edges.unwrap_or(usize::MAX)),
            fair: options.fair,
            ..FillOptions::default()
        },
    )?;
    Ok(AutoRepairReport {
        weld,
        degenerate,
        orient,
        components,
        fill,
    })
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
            if r.is_ok() {
                // Past every failure point: the sweep reached the last stage.
                assert!(r.unwrap().fill.filled.len() > 1);
                assert!(n > 5);
                break;
            }
            assert_eq!(r, Err(OpsError::Store(StoreError::OutOfMemory)));
            drop(tx);
            assert_eq!(parts(&mesh), before, "failure point {n}");
            n += 1;
        }
    }
}
