//! Filling boundary loops with triangle patches.
//!
//! [`fill_holes`] recomputes the loops from `tx.mesh()`, selects them by size
//! or by id, triangulates every selected loop (see `fill_dp`) and
//! only then writes the successful patches, in one `add_triangles` call.

use fast_mesh_store::{Transaction, VertexId};

use crate::fill_dp::triangulate_loop;
use crate::topology::{BoundaryLoop, EdgeMap, LoopId, loops_with};
use crate::{OpsError, Result, filled, push, reserve};

/// Options for [`fill_holes`].
#[derive(Debug, Clone, PartialEq)]
pub struct FillOptions {
    /// Fill only loops with at most this many edges.
    pub max_edges: Option<usize>,
    /// Fill only loops whose perimeter ([`BoundaryLoop::perimeter`]) is at
    /// most this, in local units.
    pub max_perimeter: Option<f64>,
    /// Fill only loops whose vector area ([`BoundaryLoop::area`]) is at most
    /// this, in local units squared.
    pub max_area: Option<f64>,
    /// Fill only these loops, as [`boundary_loops`](crate::boundary_loops)
    /// identified them. Every id must still match a loop of the mesh.
    pub loops: Option<Vec<LoopId>>,
    /// Also fill loops that are not [`clean`](BoundaryLoop::clean). Open
    /// chains are never filled.
    pub allow_unclean: bool,
    /// Fair the patch to continue the surrounding curvature. Accepted but
    /// not applied yet: patches are the bare triangulation.
    pub fair: bool,
    /// Loops with more edges than this are triangulated by projection onto
    /// their best-fit plane instead of the `O(n³)` minimum-weight search.
    pub dp_limit: usize,
}

impl Default for FillOptions {
    fn default() -> Self {
        FillOptions {
            max_edges: None,
            max_perimeter: None,
            max_area: None,
            loops: None,
            allow_unclean: false,
            fair: true,
            dp_limit: 200,
        }
    }
}

/// Why a loop was left out of the fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// It exceeds a bound that is set.
    Bounds,
    /// It is not clean (and unclean loops were not allowed), or it is an
    /// open chain.
    Unclean,
}

/// Why a selected loop could not be filled; it is left open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailReason {
    /// The loop is too long for the minimum-weight search and its
    /// projection onto the best-fit plane folds: the projected polygon is
    /// not simple, or a neighbouring triangle lands inside it.
    ProjectionFolds,
    /// No triangulation avoids degenerate input, edges the mesh already has
    /// and triangles it already has.
    NoValidTriangulation,
    /// The fairing solver failed.
    Solver,
}

/// A loop that was filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilledLoop {
    /// The loop, as it was before the fill.
    pub id: LoopId,
    /// Triangles added.
    pub triangles_added: usize,
    /// Vertices added (by refinement).
    pub vertices_added: usize,
    /// Whether fairing converged; `true` when it did not run.
    pub converged: bool,
}

/// What [`fill_holes`] did, each list in loop order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FillReport {
    /// The loops filled.
    pub filled: Vec<FilledLoop>,
    /// The loops left out, and why.
    pub skipped: Vec<(LoopId, SkipReason)>,
    /// The selected loops that could not be filled, and why.
    pub failed: Vec<(LoopId, FailReason)>,
}

/// Fills boundary loops with triangle patches, chosen by size or by id.
///
/// - Selection: the loops of `tx.mesh()` (as [`boundary_loops`] finds them)
///   that meet every bound that is set: `max_edges`, `max_perimeter` and
///   `max_area` (against [`BoundaryLoop::area`]). With `loops`, only those
///   loops are candidates, still subject to any bound that is set; the ids
///   are matched by content, so an id whose loop changed or vanished, also
///   through an earlier operation in the same transaction, fails the call.
///   Loops that are open chains, or not clean unless `allow_unclean`, are
///   skipped. An open mesh's intended border (a hemisphere's rim) is kept by
///   a bound below its size.
/// - Each selected loop is triangulated on its own: up to `dp_limit` edges
///   by Liepa's minimum-weight search (the worst dihedral angle to the
///   neighbouring triangles first, then area), above it by projection onto
///   the loop's best-fit plane and ear clipping, which fails with
///   [`FailReason::ProjectionFolds`] when the projection folds. No new edge
///   duplicates one the mesh has, so the patch never makes an edge
///   non-manifold, and the patch triangles are wound to agree with their
///   neighbours. A loop whose patch would reuse a
///   diagonal planned for an earlier loop of the same call fails too. Two
///   holes sharing a vertex are two loops, each filled.
/// - Every selected loop is planned before the first write; the successful
///   patches are then written in loop order in one `add_triangles` call.
///   A failed loop is reported and left open.
///
/// Refinement and fairing are not applied yet: `fair` is accepted, every
/// patch is the bare triangulation, `vertices_added` is 0 and `converged`
/// is `true`.
///
/// [`boundary_loops`]: crate::boundary_loops
///
/// # Errors
/// - [`OpsError::InvalidInput`] when no bound and no `loops` are given, or
///   a bound is NaN or negative.
/// - [`OpsError::StaleLoops`] when an id in `loops` matches no loop of the
///   mesh.
/// - [`OpsError::Store`] when memory runs out.
///
/// Nothing is written in any of these cases.
pub fn fill_holes(tx: &mut Transaction, options: &FillOptions) -> Result<FillReport> {
    let bounds = [options.max_perimeter, options.max_area];
    if options.max_edges.is_none() && bounds.iter().all(Option::is_none) && options.loops.is_none()
    {
        return Err(OpsError::InvalidInput(
            "fill_holes needs a bound or a list of loops",
        ));
    }
    if bounds.iter().flatten().any(|b| b.is_nan() || *b < 0.0) {
        return Err(OpsError::InvalidInput(
            "fill bounds must not be NaN or negative",
        ));
    }
    let mesh = tx.mesh();
    let edges = EdgeMap::build(mesh)?;
    let loops = loops_with(mesh, &edges)?;
    let mut selected = filled(loops.len(), options.loops.is_none())?;
    if let Some(ids) = &options.loops {
        for id in ids {
            // The loops are sorted by smallest vertex first.
            let from = loops.partition_point(|l| l.id.min_vertex < id.min_vertex);
            let found = loops[from..]
                .iter()
                .take_while(|l| l.id.min_vertex == id.min_vertex)
                .position(|l| l.id == *id)
                .ok_or(OpsError::StaleLoops)?;
            selected[from + found] = true;
        }
    }

    let mut report = FillReport::default();
    let mut patch: Vec<[VertexId; 3]> = Vec::new();
    // The diagonals of the patches planned so far, sorted.
    let mut chords: Vec<(VertexId, VertexId)> = Vec::new();
    let mut new_chords: Vec<(VertexId, VertexId)> = Vec::new();
    for (l, _) in loops.iter().zip(&selected).filter(|(_, s)| **s) {
        if !within_bounds(l, options) {
            push(&mut report.skipped, (l.id, SkipReason::Bounds))?;
            continue;
        }
        if !l.closed || !(l.clean || options.allow_unclean) {
            push(&mut report.skipped, (l.id, SkipReason::Unclean))?;
            continue;
        }
        let tris = match triangulate_loop(mesh, &l.vertices, &edges, options.dp_limit)? {
            Ok(tris) => tris,
            Err(reason) => {
                push(&mut report.failed, (l.id, reason))?;
                continue;
            }
        };
        new_chords.clear();
        for t in &tris {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                if edges.kind(a, b).is_none() {
                    push(&mut new_chords, if a < b { (a, b) } else { (b, a) })?;
                }
            }
        }
        // Each diagonal is in exactly two patch triangles, unless a loop that
        // repeats a vertex mapped two diagonals onto one edge. It must not be
        // a diagonal of an earlier patch either.
        new_chords.sort_unstable();
        let manifold = new_chords.chunk_by(|a, b| a == b).all(|r| r.len() == 2);
        new_chords.dedup();
        if !manifold || new_chords.iter().any(|c| chords.binary_search(c).is_ok()) {
            push(&mut report.failed, (l.id, FailReason::NoValidTriangulation))?;
            continue;
        }
        reserve(&mut chords, new_chords.len())?;
        chords.extend_from_slice(&new_chords);
        chords.sort_unstable();
        reserve(&mut patch, tris.len())?;
        patch.extend_from_slice(&tris);
        push(
            &mut report.filled,
            FilledLoop {
                id: l.id,
                triangles_added: tris.len(),
                vertices_added: 0,
                converged: true,
            },
        )?;
    }
    if !patch.is_empty() {
        tx.add_triangles(&patch)?;
    }
    Ok(report)
}

/// Whether `l` meets every bound of `options` that is set.
fn within_bounds(l: &BoundaryLoop, options: &FillOptions) -> bool {
    options.max_edges.is_none_or(|m| l.vertices.len() <= m)
        && options.max_perimeter.is_none_or(|m| l.perimeter <= m)
        && options.max_area.is_none_or(|m| l.area <= m)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use fast_mesh_store::Mesh;

    use super::*;
    use crate::analyze::{Analysis, AnalyzeOptions, analyze};
    use crate::test_util::*;
    use crate::{boundary_loops, weld};

    fn analysis(mesh: &Mesh) -> Analysis {
        analyze(mesh, &AnalyzeOptions::default()).unwrap()
    }

    fn unfaired() -> FillOptions {
        FillOptions {
            fair: false,
            ..FillOptions::default()
        }
    }

    /// No loop left but those in `kept`, no non-manifold edge and no
    /// orientation conflict.
    fn assert_sound(mesh: &Mesh, kept: usize) {
        let a = analysis(mesh);
        assert_eq!(a.loops.len(), kept);
        assert_eq!(a.non_manifold_edges, 0);
        assert_eq!(a.orientation_conflicts, 0);
    }

    #[test]
    fn sphere_holes_filled_closed_and_consistent() {
        let centres = [[1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [-0.6, -0.48, -0.64]];
        let mut mesh = punch_holes(&uv_sphere(64, 128, 1.0), &centres, 0.2);
        let loops = boundary_loops(&mesh).unwrap();
        assert_eq!(loops.len(), 3);
        let options = FillOptions {
            max_edges: Some(500),
            ..unfaired()
        };
        let (r, _) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert!(r.skipped.is_empty() && r.failed.is_empty());
        let expected: Vec<FilledLoop> = loops
            .iter()
            .map(|l| FilledLoop {
                id: l.id,
                triangles_added: l.vertices.len() - 2,
                vertices_added: 0,
                converged: true,
            })
            .collect();
        assert_eq!(r.filled, expected);
        assert_sound(&mesh, 0);
        let a = analysis(&mesh);
        assert_eq!(a.components.len(), 1);
        assert!(a.components.iter().all(|c| c.closed));
    }

    #[test]
    fn bowtie_holes_both_filled() {
        let mut mesh = bowtie_holes();
        let loops = boundary_loops(&mesh).unwrap();
        let options = FillOptions {
            max_edges: Some(4),
            ..unfaired()
        };
        let (r, _) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        let filled: Vec<LoopId> = r.filled.iter().map(|f| f.id).collect();
        let small: Vec<LoopId> = loops
            .iter()
            .filter(|l| l.vertices.len() == 4)
            .map(|l| l.id)
            .collect();
        assert_eq!((filled.len(), filled), (2, small));
        let outer = loops.iter().find(|l| l.vertices.len() == 16).unwrap();
        assert_eq!(r.skipped, vec![(outer.id, SkipReason::Bounds)]);
        assert_sound(&mesh, 1);
        assert!(analysis(&mesh).pinch_vertices.is_empty());
        assert_eq!(mesh.triangle_count(), 32);
    }

    #[test]
    fn hemisphere_rim_kept_by_perimeter_bound() {
        let c2 = [0.8f32.sin(), 0.0, 0.8f32.cos()];
        let mut mesh = punch_holes(&hemisphere(16, 64, 1.0), &[[0.0, 0.0, 1.0], c2], 0.1);
        let loops = boundary_loops(&mesh).unwrap();
        assert_eq!(loops.len(), 3);
        let options = FillOptions {
            max_perimeter: Some(1.0),
            ..unfaired()
        };
        let (r, _) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert_eq!(r.filled.len(), 2);
        assert_eq!(r.skipped.len(), 1);
        assert_sound(&mesh, 1);
        let rim = &boundary_loops(&mesh).unwrap()[0];
        assert_eq!(rim.vertices.len(), 64);
        assert!((rim.perimeter - TAU).abs() < 0.01);
        assert_eq!(r.skipped, vec![(rim.id, SkipReason::Bounds)]);
    }

    /// [`bowtie_holes`] with vertex 6, a corner of the first hole, moved
    /// next to vertex 7.
    fn bowtie_near_weld() -> Mesh {
        let (mut positions, triangles) = parts(&bowtie_holes());
        positions[3 * 6] = 1.999;
        mesh_of(&positions, &triangles)
    }

    #[test]
    fn stale_loop_id_after_weld_in_same_tx_is_rejected() {
        let mut mesh = bowtie_near_weld();
        let ids: Vec<LoopId> = boundary_loops(&mesh)
            .unwrap()
            .iter()
            .map(|l| l.id)
            .collect();
        let mut tx = mesh.begin_edit();
        assert_eq!(weld(&mut tx, 0.01).unwrap().merged_vertices, 1);
        let options = FillOptions {
            loops: Some(ids),
            ..unfaired()
        };
        assert_eq!(fill_holes(&mut tx, &options), Err(OpsError::StaleLoops));
        let patch = tx.commit().unwrap();

        // The same as a weld alone: same slots, positions and triangles.
        let mut reference = bowtie_near_weld();
        let mut tx = reference.begin_edit();
        weld(&mut tx, 0.01).unwrap();
        let weld_only = tx.commit().unwrap();
        assert!(!patch.is_empty());
        assert_eq!(patch.is_empty(), weld_only.is_empty());
        assert_eq!(parts(&mesh), parts(&reference));
        assert_eq!(mesh.triangle_slots(), reference.triangle_slots());
        assert_eq!(mesh.vertex_count(), reference.vertex_count());
    }

    /// [`grid_without`]`(4, (1, 1))` with vertex 12 moved to `(1.3, 1.3)`, so
    /// that the hole `6, 7, 12, 11` is reflex at 12 and only its diagonal
    /// 6-12 lies inside it; with a closed tetrahedron on that diagonal when
    /// `tetra` is set.
    fn bent_hole(tetra: bool) -> Mesh {
        let (mut positions, triangles) = parts(&grid_without(4, &[(1, 1)]));
        positions[3 * 12] = 1.3;
        positions[3 * 12 + 1] = 1.3;
        let mesh = mesh_of(&positions, &triangles);
        if tetra {
            with_tetra(&mesh, 6, 12, [[1.6, 1.2, -1.0], [1.3, 1.8, -0.9]])
        } else {
            mesh
        }
    }

    fn uses_edge(tris: &[[VertexId; 3]], a: VertexId, b: VertexId) -> bool {
        tris.iter().any(|t| t.contains(&a) && t.contains(&b))
    }

    #[test]
    fn fill_avoids_existing_edges() {
        // Without the tetrahedron, the search prefers diagonal 6-12.
        let free = bent_hole(false);
        let loops = boundary_loops(&free).unwrap();
        let l = loops.iter().find(|l| l.vertices.len() == 4).unwrap();
        let edges = EdgeMap::build(&free).unwrap();
        let tris = triangulate_loop(&free, &l.vertices, &edges, 200)
            .unwrap()
            .unwrap();
        assert!(uses_edge(&tris, 6, 12));

        let mut mesh = bent_hole(true);
        let loops = boundary_loops(&mesh).unwrap();
        let hole = loops.iter().find(|l| l.vertices.len() == 4).unwrap();
        assert!(hole.clean);
        let options = FillOptions {
            loops: Some(vec![hole.id]),
            ..unfaired()
        };
        let (r, _) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert_eq!(r.filled.len(), 1);
        assert_eq!(analysis(&mesh).non_manifold_edges, 0);
        let last: Vec<[VertexId; 3]> = (mesh.triangle_slots() as u32 - 2..)
            .take(2)
            .map(|t| mesh.triangle(t).unwrap())
            .collect();
        assert!(uses_edge(&last, 7, 11) && !uses_edge(&last, 6, 12));
    }

    /// An annulus of `n` quads in the plane `z = 0`, radii 1 and 1.2, with
    /// a small wobble in `z`.
    fn wobbly_annulus(n: u32) -> Mesh {
        let at = |r: f64| {
            move |t: f64| {
                let a = TAU * t;
                [r * a.cos(), r * a.sin(), 0.02 * (3.0 * a).sin()]
            }
        };
        band(n, at(1.0), at(1.2))
    }

    #[test]
    fn large_loop_uses_projection_fallback() {
        let mut mesh = wobbly_annulus(400);
        let loops = boundary_loops(&mesh).unwrap();
        let inner = loops.iter().find(|l| l.id.min_vertex == 0).unwrap();
        assert_eq!(inner.vertices.len(), 400);
        let options = FillOptions {
            max_perimeter: Some(7.0),
            dp_limit: 200,
            ..unfaired()
        };
        let (r, _) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert_eq!(r.filled.len(), 1);
        assert_eq!(r.filled[0].id, inner.id);
        assert_eq!(r.filled[0].triangles_added, 398);
        assert_sound(&mesh, 1);
    }

    /// A band around a curve that winds twice around the `z` axis, so its
    /// loops project onto their best-fit plane (`z = 0`) as curves that
    /// cross themselves.
    fn double_wound_band(n: u32) -> Mesh {
        let at = |rho: f64| {
            move |t: f64| {
                let (major, minor) = (2.0 * TAU * t, TAU * t);
                let r = 2.0 + rho * minor.cos();
                [r * major.cos(), r * major.sin(), rho * minor.sin()]
            }
        };
        band(n, at(0.5), at(0.8))
    }

    #[test]
    fn folded_projection_fails_and_writes_nothing() {
        let mut mesh = double_wound_band(300);
        let loops = boundary_loops(&mesh).unwrap();
        assert_eq!(loops.len(), 2);
        let inner = loops[0].id;
        let options = FillOptions {
            loops: Some(vec![inner]),
            ..unfaired()
        };
        let (r, empty) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert!(empty);
        assert_eq!(r.failed, vec![(inner, FailReason::ProjectionFolds)]);
        assert!(r.filled.is_empty());
        assert_eq!(boundary_loops(&mesh).unwrap(), loops);
    }

    #[test]
    fn no_bound_invalid_input() {
        let mut mesh = bowtie_holes();
        let mut tx = mesh.begin_edit();
        let r = fill_holes(&mut tx, &FillOptions::default());
        assert!(matches!(r, Err(OpsError::InvalidInput(_))));
        let nan = FillOptions {
            max_area: Some(f64::NAN),
            ..FillOptions::default()
        };
        assert!(matches!(
            fill_holes(&mut tx, &nan),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(tx.commit().unwrap().is_empty());
    }

    /// [`grid_without`]`(4, (1, 1))` with a closed tetrahedron on edge
    /// 12-13, which makes that edge non-manifold and the hole unclean.
    fn unclean_hole() -> Mesh {
        with_tetra(
            &grid_without(4, &[(1, 1)]),
            12,
            13,
            [[2.5, 1.8, 1.0], [2.5, 2.4, 0.8]],
        )
    }

    #[test]
    fn unclean_loop_skipped_unless_allowed() {
        let mut mesh = unclean_hole();
        let loops = boundary_loops(&mesh).unwrap();
        let hole = loops.iter().find(|l| l.vertices.len() == 4).unwrap();
        assert!(hole.closed && !hole.clean);
        let mut options = FillOptions {
            max_edges: Some(4),
            ..unfaired()
        };
        let (r, empty) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert!(empty && r.filled.is_empty());
        assert!(r.skipped.contains(&(hole.id, SkipReason::Unclean)));

        options.allow_unclean = true;
        let (r, _) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert_eq!(r.filled.len(), 1);
        assert_eq!(r.filled[0].id, hole.id);
        assert!(
            boundary_loops(&mesh)
                .unwrap()
                .iter()
                .all(|l| l.id != hole.id)
        );
    }

    #[test]
    fn open_chains_are_never_filled() {
        let mut mesh = fin();
        let loops = boundary_loops(&mesh).unwrap();
        let options = FillOptions {
            max_edges: Some(100),
            allow_unclean: true,
            ..unfaired()
        };
        let (r, _) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        for l in loops.iter().filter(|l| !l.closed) {
            assert!(r.skipped.contains(&(l.id, SkipReason::Unclean)));
        }
    }

    #[test]
    fn single_triangle_is_not_doubled() {
        let mut mesh = mesh_of(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[[0, 1, 2]]);
        let options = FillOptions {
            max_edges: Some(3),
            ..unfaired()
        };
        let (r, empty) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert!(empty);
        assert_eq!(r.failed.len(), 1);
        assert_eq!(r.failed[0].1, FailReason::NoValidTriangulation);
    }

    #[test]
    fn concave_planar_hole_gets_no_folded_triangle() {
        // An L-shaped hole: every patch triangle must face +z like the grid.
        let mut mesh = grid_without(4, &[(1, 1), (2, 1), (1, 2)]);
        let options = FillOptions {
            max_edges: Some(8),
            ..unfaired()
        };
        let (r, _) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert_eq!(r.filled.len(), 1);
        assert_sound(&mesh, 1);
        for t in mesh.triangle_slots() as u32 - 6..mesh.triangle_slots() as u32 {
            let [a, b, c] = mesh
                .triangle(t)
                .unwrap()
                .map(|v| mesh.positions()[v as usize]);
            let z = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            assert!(z > 0.0, "triangle {t} faces down");
        }
    }

    #[test]
    fn out_of_memory_is_reported_and_writes_nothing() {
        let options = FillOptions {
            max_edges: Some(4),
            ..unfaired()
        };
        assert!(sweep(bowtie_holes(), |tx| fill_holes(tx, &options).map(drop)) > 5);
        let options = FillOptions {
            max_perimeter: Some(7.0),
            ..unfaired()
        };
        assert!(sweep(wobbly_annulus(400), |tx| fill_holes(tx, &options).map(drop)) > 5);
    }
}
