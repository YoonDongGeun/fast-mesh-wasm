//! Welding vertices and removing degenerate triangles and small components.
//!
//! Each operation reads its whole plan from `tx.mesh()`, then writes in this
//! order: rewired triangles, removed triangles, removed vertices. A second run
//! on the result finds nothing to do and writes nothing.

use fast_mesh_store::{Mesh, Transaction, TriangleId, VertexId};

use crate::analyze::{AREA_EPSILON, duplicate_triangles, is_degenerate};
use crate::topology::{EdgeMap, components};
use crate::weld::weld_roots;
use crate::{OpsError, Result, filled, push, reserve};

/// What [`weld()`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WeldReport {
    /// Vertices merged into the smallest id of their group.
    pub merged_vertices: usize,
    /// Triangles removed because two of their corners merged.
    pub removed_triangles: usize,
}

/// Merges vertices at most `tolerance` apart (local units).
///
/// Vertices are grouped through chains of pairs within the tolerance; each
/// group keeps its smallest id at its original position. Triangles are
/// rewritten to the kept ids, those that thereby get a repeated id are
/// removed, and so are all vertices no triangle uses afterwards, including
/// ones that were unused before. Tolerance 0 merges exact duplicates only. No
/// vertex is created.
///
/// Rewiring can make two triangles share a vertex set, or give an edge a
/// third triangle; run [`remove_degenerate()`] afterwards to drop duplicates.
///
/// # Errors
/// [`OpsError::InvalidInput`] when `tolerance` is negative, NaN or infinite;
/// [`OpsError::Store`] when memory runs out.
pub fn weld(tx: &mut Transaction, tolerance: f32) -> Result<WeldReport> {
    let mesh = tx.mesh();
    let roots = weld_roots(mesh, tolerance)?;
    let mut report = WeldReport::default();
    for (v, &r) in roots.iter().enumerate() {
        if r as usize != v {
            report.merged_vertices += 1;
        }
    }
    let repeats = |[a, b, c]: [VertexId; 3]| a == b || b == c || a == c;
    let (mut rewired, mut rewired_to, mut collapsed) = (Vec::new(), Vec::new(), Vec::new());
    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        let new = tri.map(|v| roots[v as usize]);
        if new == tri {
            continue;
        }
        if repeats(new) && !repeats(tri) {
            push(&mut collapsed, t)?;
        } else {
            push(&mut rewired, t)?;
            push(&mut rewired_to, new)?;
        }
    }
    report.removed_triangles = collapsed.len();
    let unused = unused_vertices(mesh, &collapsed, &rewired, &rewired_to)?;
    tx.set_triangles(&rewired, &rewired_to)?;
    tx.remove_triangles(&collapsed)?;
    tx.remove_vertices(&unused)?;
    Ok(report)
}

/// Options for [`remove_degenerate()`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DegenerateOptions {
    /// A triangle is degenerate when its area is at most this times its
    /// longest edge squared. Default `1e-12`.
    pub area_epsilon: f64,
    /// Whether to remove the vertices no triangle uses afterwards. Default
    /// `true`.
    pub remove_unused_vertices: bool,
}

impl Default for DegenerateOptions {
    fn default() -> Self {
        Self {
            area_epsilon: AREA_EPSILON,
            remove_unused_vertices: true,
        }
    }
}

/// What [`remove_degenerate()`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DegenerateReport {
    /// Triangles removed.
    pub removed_triangles: usize,
    /// Vertices removed.
    pub removed_vertices: usize,
}

/// Removes triangles with a repeated vertex id, triangles whose area is at
/// most `area_epsilon` times their longest edge squared, and duplicates: of
/// triangles with the same vertex set, all but the one with the smallest id.
///
/// # Errors
/// [`OpsError::InvalidInput`] when `area_epsilon` is negative, NaN or
/// infinite; [`OpsError::Store`] when memory runs out.
pub fn remove_degenerate(
    tx: &mut Transaction,
    options: &DegenerateOptions,
) -> Result<DegenerateReport> {
    if !(options.area_epsilon >= 0.0 && options.area_epsilon.is_finite()) {
        return Err(OpsError::InvalidInput(
            "area_epsilon must be finite and not negative",
        ));
    }
    let mesh = tx.mesh();
    let mut doomed: Vec<TriangleId> = Vec::new();
    for t in 0..mesh.triangle_slots() as TriangleId {
        if mesh
            .triangle(t)
            .is_some_and(|tri| is_degenerate(mesh, tri, options.area_epsilon))
        {
            push(&mut doomed, t)?;
        }
    }
    for t in duplicate_triangles(mesh)? {
        push(&mut doomed, t)?;
    }
    doomed.sort_unstable();
    doomed.dedup();
    let unused = if options.remove_unused_vertices {
        unused_vertices(mesh, &doomed, &[], &[])?
    } else {
        Vec::new()
    };
    tx.remove_triangles(&doomed)?;
    tx.remove_vertices(&unused)?;
    let removed_vertices = unused.len();
    Ok(DegenerateReport {
        removed_triangles: doomed.len(),
        removed_vertices,
    })
}

/// Options for [`remove_small_components()`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComponentOptions {
    /// Remove components with fewer triangles than this.
    pub min_triangles: Option<usize>,
    /// Remove components with less area than this (local units squared).
    pub min_area: Option<f64>,
    /// Keep the largest component even when it is below a bound. Largest
    /// means most area; ties go to the most triangles, then the smallest
    /// triangle id. Default `true`.
    pub keep_largest: bool,
}

impl Default for ComponentOptions {
    fn default() -> Self {
        Self {
            min_triangles: None,
            min_area: None,
            keep_largest: true,
        }
    }
}

/// What [`remove_small_components()`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RemoveReport {
    /// Components removed.
    pub removed_components: usize,
    /// Triangles removed.
    pub removed_triangles: usize,
    /// Vertices removed: those no triangle uses afterwards.
    pub removed_vertices: usize,
}

/// Removes the connected components (triangles joined through shared edges
/// of any kind) that are below any bound that is set, and the vertices left
/// unused. Triangles with a repeated vertex id belong to no component and stay.
///
/// # Errors
/// [`OpsError::InvalidInput`] when no bound is set or `min_area` is negative,
/// NaN or infinite; [`OpsError::Store`] when memory runs out.
pub fn remove_small_components(
    tx: &mut Transaction,
    options: &ComponentOptions,
) -> Result<RemoveReport> {
    if options.min_triangles.is_none() && options.min_area.is_none() {
        return Err(OpsError::InvalidInput(
            "remove_small_components needs min_triangles or min_area",
        ));
    }
    if let Some(a) = options.min_area
        && !(a >= 0.0 && a.is_finite())
    {
        return Err(OpsError::InvalidInput(
            "min_area must be finite and not negative",
        ));
    }
    let mesh = tx.mesh();
    let edges = EdgeMap::build(mesh)?;
    let comps = components(mesh, &edges)?;
    // Components come sorted by their smallest triangle id, so the first of
    // equals wins the final tie.
    let mut largest: Option<usize> = None;
    for (i, c) in comps.iter().enumerate() {
        let better = largest.is_none_or(|j| {
            let l = &comps[j];
            (c.area, c.triangles.len()) > (l.area, l.triangles.len())
        });
        if better {
            largest = Some(i);
        }
    }
    let mut doomed: Vec<TriangleId> = Vec::new();
    let mut removed_components = 0;
    for (i, c) in comps.iter().enumerate() {
        if options.keep_largest && largest == Some(i) {
            continue;
        }
        let small = options.min_triangles.is_some_and(|n| c.triangles.len() < n)
            || options.min_area.is_some_and(|a| c.area < a);
        if small {
            removed_components += 1;
            reserve(&mut doomed, c.triangles.len())?;
            doomed.extend_from_slice(&c.triangles);
        }
    }
    doomed.sort_unstable();
    let unused = unused_vertices(mesh, &doomed, &[], &[])?;
    tx.remove_triangles(&doomed)?;
    tx.remove_vertices(&unused)?;
    let removed_vertices = unused.len();
    Ok(RemoveReport {
        removed_components,
        removed_triangles: doomed.len(),
        removed_vertices,
    })
}

/// The alive vertices no triangle uses once the planned edits are applied:
/// triangles `removed` (ascending) are gone, and `rewired[i]` (ascending) has
/// the corners `rewired_to[i]`. Ascending.
fn unused_vertices(
    mesh: &Mesh,
    removed: &[TriangleId],
    rewired: &[TriangleId],
    rewired_to: &[[VertexId; 3]],
) -> Result<Vec<VertexId>> {
    let mut used = filled(mesh.vertex_slots(), false)?;
    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some(mut tri) = mesh.triangle(t) else {
            continue;
        };
        if removed.binary_search(&t).is_ok() {
            continue;
        }
        if let Ok(i) = rewired.binary_search(&t) {
            tri = rewired_to[i];
        }
        for v in tri {
            used[v as usize] = true;
        }
    }
    let mut unused = Vec::new();
    for v in 0..mesh.vertex_slots() as VertexId {
        if mesh.is_vertex_alive(v) && !used[v as usize] {
            push(&mut unused, v)?;
        }
    }
    Ok(unused)
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::{History, Mesh};

    use super::*;
    use crate::analyze::{AnalyzeOptions, analyze};
    use crate::test_util::*;

    #[test]
    fn weld_merges_duplicates_and_drops_collapsed() {
        let mut mesh = damaged_sphere();
        let (r, _) = edit(&mut mesh, |tx| weld(tx, 0.0));
        assert_eq!(
            r,
            WeldReport {
                merged_vertices: 4,
                removed_triangles: 0
            }
        );
        assert_eq!(mesh.vertex_count(), 90);
        assert_eq!(mesh.triangle_count(), 170);
        let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
        assert_eq!(a.coincident_groups, 0);
        // The slits are sewn shut: only the three holes and the square remain.
        let mut lens: Vec<usize> = a.loops.iter().map(|l| l.vertices.len()).collect();
        lens.sort_unstable();
        assert_eq!(lens, [3, 3, 3, 4]);
    }

    #[test]
    fn weld_removes_triangles_that_collapse() {
        // Vertices 0 and 1 are 1e-5 apart; triangle 0 uses both.
        let mut mesh = Mesh::from_buffers(
            &[0.0, 0.0, 0.0, 1e-5, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            &[0, 1, 2, 1, 3, 2],
        )
        .unwrap();
        let (r, _) = edit(&mut mesh, |tx| weld(tx, 1e-4));
        assert_eq!(
            r,
            WeldReport {
                merged_vertices: 1,
                removed_triangles: 1
            }
        );
        assert_eq!(mesh.triangle_count(), 1);
        assert_eq!(mesh.triangle(1), Some([0, 3, 2]));
        assert!(!mesh.is_vertex_alive(1));
    }

    #[test]
    fn weld_tolerance_merges_near() {
        let positions = [0.0, 0.0, 0.0, 1e-5, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let mut mesh = Mesh::from_buffers(&positions, &[0, 3, 2, 1, 3, 2]).unwrap();
        let (r, empty) = edit(&mut mesh, |tx| weld(tx, 1e-6));
        assert_eq!((r.merged_vertices, empty), (0, true));
        let (r, _) = edit(&mut mesh, |tx| weld(tx, 1e-4));
        assert_eq!(r.merged_vertices, 1);
        assert_eq!(mesh.triangle(1), Some([0, 3, 2]));
    }

    #[test]
    fn weld_rejects_bad_tolerance() {
        let mut mesh = damaged_sphere();
        let mut tx = mesh.begin_edit();
        for t in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(weld(&mut tx, t), Err(OpsError::InvalidInput(_))));
        }
    }

    #[test]
    fn degenerate_and_duplicates_removed() {
        let mut mesh = damaged_sphere();
        let (r, _) = edit(&mut mesh, |tx| {
            remove_degenerate(tx, &DegenerateOptions::default())
        });
        // 166 and 167 repeat an id, 165 is a copy; no vertex is left unused.
        assert_eq!(
            r,
            DegenerateReport {
                removed_triangles: 3,
                removed_vertices: 0
            }
        );
        assert_eq!(mesh.triangle_count(), 167);
        assert!(!mesh.is_triangle_alive(165));
        assert!(!mesh.is_triangle_alive(166) && !mesh.is_triangle_alive(167));
        let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
        assert_eq!((a.degenerate, a.duplicate, a.non_manifold_edges), (0, 0, 0));
    }

    #[test]
    fn degenerate_removes_slivers_and_unused_vertices() {
        // Triangle 1 lies on a line; vertices 3 and 4 are used only by it.
        let positions = [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0,
        ];
        let mut mesh = Mesh::from_buffers(&positions, &[0, 1, 2, 1, 3, 4]).unwrap();
        let keep = DegenerateOptions {
            remove_unused_vertices: false,
            ..Default::default()
        };
        let (r, _) = edit(&mut mesh, |tx| remove_degenerate(tx, &keep));
        assert_eq!((r.removed_triangles, r.removed_vertices), (1, 0));
        let (r, _) = edit(&mut mesh, |tx| {
            remove_degenerate(tx, &DegenerateOptions::default())
        });
        assert_eq!((r.removed_triangles, r.removed_vertices), (0, 2));
        assert_eq!((mesh.vertex_count(), mesh.triangle_count()), (3, 1));
    }

    #[test]
    fn degenerate_rejects_bad_epsilon() {
        let mut mesh = damaged_sphere();
        let mut tx = mesh.begin_edit();
        let o = DegenerateOptions {
            area_epsilon: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            remove_degenerate(&mut tx, &o),
            Err(OpsError::InvalidInput(_))
        ));
    }

    #[test]
    fn small_components_removed_largest_kept() {
        let mut mesh = damaged_sphere();
        let o = ComponentOptions {
            min_triangles: Some(10),
            ..Default::default()
        };
        let (r, _) = edit(&mut mesh, |tx| remove_small_components(tx, &o));
        assert_eq!(
            r,
            RemoveReport {
                removed_components: 1,
                removed_triangles: 2,
                removed_vertices: 4
            }
        );
        assert_eq!(mesh.triangle_count(), 168);
        let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
        assert_eq!(a.components.len(), 1);

        // Only small pieces: the largest survives, unless keep_largest is off.
        let pieces = || combine(&[&grid(1), &grid(2), &grid(3)]);
        let o = ComponentOptions {
            min_triangles: Some(100),
            ..Default::default()
        };
        let mut m = pieces();
        let (r, _) = edit(&mut m, |tx| remove_small_components(tx, &o));
        assert_eq!((r.removed_components, m.triangle_count()), (2, 18));
        let mut m = pieces();
        let o = ComponentOptions {
            keep_largest: false,
            ..o
        };
        let (r, _) = edit(&mut m, |tx| remove_small_components(tx, &o));
        assert_eq!((r.removed_components, m.triangle_count()), (3, 0));
        assert_eq!(m.vertex_count(), 0);
    }

    #[test]
    fn small_components_by_area_and_largest_tie() {
        // Two identical unit squares: equal area and size, the first is kept.
        let mut m = combine(&[&grid(1), &grid(1)]);
        let o = ComponentOptions {
            min_area: Some(5.0),
            ..Default::default()
        };
        let (r, _) = edit(&mut m, |tx| remove_small_components(tx, &o));
        assert_eq!(r.removed_components, 1);
        assert!(m.is_triangle_alive(0) && !m.is_triangle_alive(2));
    }

    #[test]
    fn no_bound_is_invalid_input() {
        let mut mesh = damaged_sphere();
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            remove_small_components(&mut tx, &ComponentOptions::default()),
            Err(OpsError::InvalidInput(_))
        ));
        let o = ComponentOptions {
            min_area: Some(f64::NAN),
            ..Default::default()
        };
        assert!(matches!(
            remove_small_components(&mut tx, &o),
            Err(OpsError::InvalidInput(_))
        ));
    }

    #[test]
    fn idempotent() {
        let mut mesh = damaged_sphere();
        let comp = ComponentOptions {
            min_triangles: Some(10),
            ..Default::default()
        };
        let dg = DegenerateOptions::default();
        edit(&mut mesh, |tx| weld(tx, 0.0));
        edit(&mut mesh, |tx| remove_degenerate(tx, &dg));
        edit(&mut mesh, |tx| remove_small_components(tx, &comp));
        let (r, empty) = edit(&mut mesh, |tx| weld(tx, 0.0));
        assert_eq!((r, empty), (WeldReport::default(), true));
        let (r, empty) = edit(&mut mesh, |tx| remove_degenerate(tx, &dg));
        assert_eq!((r, empty), (DegenerateReport::default(), true));
        let (r, empty) = edit(&mut mesh, |tx| remove_small_components(tx, &comp));
        assert_eq!((r, empty), (RemoveReport::default(), true));
    }

    #[test]
    fn undo_restores() {
        let mut mesh = damaged_sphere();
        let before = parts(&mesh);
        let mut history = History::new(1 << 24);
        let comp = ComponentOptions {
            min_triangles: Some(10),
            ..Default::default()
        };
        for step in 0..3 {
            let mut tx = mesh.begin_edit();
            match step {
                0 => drop(weld(&mut tx, 0.0).unwrap()),
                1 => drop(remove_degenerate(&mut tx, &DegenerateOptions::default()).unwrap()),
                _ => drop(remove_small_components(&mut tx, &comp).unwrap()),
            }
            let patch = tx.commit().unwrap();
            assert!(!patch.is_empty());
            history.push(&mut mesh, patch).unwrap();
        }
        assert_ne!(parts(&mesh), before);
        while history.undo(&mut mesh).unwrap().is_some() {}
        assert_eq!(parts(&mesh), before);
        assert_eq!(mesh.vertex_count(), 94);
    }

    #[test]
    fn weld_into_duplicates_then_remove_degenerate() {
        // Triangles 0 and 1 differ only in vertex 0 vs its copy 4.
        let positions = [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0,
        ];
        let mut mesh = Mesh::from_buffers(&positions, &[0, 1, 2, 4, 1, 2]).unwrap();
        let (r, _) = edit(&mut mesh, |tx| weld(tx, 0.0));
        assert_eq!(r.merged_vertices, 1);
        assert_eq!(mesh.triangle_count(), 2);
        // The stray vertex 3 was unused before and goes too.
        assert_eq!(mesh.vertex_count(), 3);
        let (r, _) = edit(&mut mesh, |tx| {
            remove_degenerate(tx, &DegenerateOptions::default())
        });
        assert_eq!(r.removed_triangles, 1);
        assert!(mesh.is_triangle_alive(0) && !mesh.is_triangle_alive(1));
    }

    #[test]
    fn weld_chains_near_points_and_keeps_the_smallest_in_place() {
        // A, B, C in a row 0.7e-4 apart: |AC| = 1.4e-4 > 1e-4, one group.
        let positions = [
            0.0, 0.0, 0.0, 0.7e-4, 0.0, 0.0, 1.4e-4, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        ];
        let mut mesh = Mesh::from_buffers(&positions, &[0, 3, 4, 1, 3, 4, 2, 3, 4]).unwrap();
        let kept = mesh.positions()[0];
        let (r, _) = edit(&mut mesh, |tx| weld(tx, 1e-4));
        assert_eq!(r.merged_vertices, 2);
        assert_eq!(mesh.positions()[0], kept);
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.triangle(2), Some([0, 3, 4]));
    }

    #[test]
    fn weld_removes_preexisting_unused_vertices() {
        let positions = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 9.0, 9.0, 9.0];
        let mut mesh = Mesh::from_buffers(&positions, &[0, 1, 2]).unwrap();
        let (r, empty) = edit(&mut mesh, |tx| weld(tx, 0.0));
        assert_eq!((r, empty), (WeldReport::default(), false));
        assert_eq!(mesh.vertex_count(), 3);
        let (_, empty) = edit(&mut mesh, |tx| weld(tx, 0.0));
        assert!(empty);
    }

    #[test]
    fn out_of_memory_is_reported_and_writes_nothing() {
        let comp = ComponentOptions {
            min_triangles: Some(10),
            ..Default::default()
        };
        assert!(sweep(damaged_sphere(), |tx| weld(tx, 0.0).map(drop)) > 5);
        assert!(
            sweep(damaged_sphere(), |tx| remove_degenerate(
                tx,
                &DegenerateOptions::default()
            )
            .map(drop))
                > 5
        );
        assert!(
            sweep(damaged_sphere(), |tx| remove_small_components(tx, &comp)
                .map(drop))
                > 5
        );
    }
}
