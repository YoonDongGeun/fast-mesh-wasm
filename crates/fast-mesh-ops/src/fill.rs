//! Filling boundary loops with triangle patches.
//!
//! [`fill_holes`] recomputes the loops from `tx.mesh()`, selects them by size
//! or by id, triangulates every selected loop (see `fill_dp`), refines the
//! patch to the surrounding density (`refine`), fairs it (`fair`) and only
//! then writes the successful patches: their new vertices, then their
//! triangles.

use fast_mesh_store::{StoreError, Transaction, VertexId};

use crate::fair::fair;
use crate::fill_dp::triangulate_loop;
use crate::refine::{RefinedPatch, refine};
use crate::topology::{BoundaryLoop, EdgeMap, LoopId, VertexFans, loops_with};
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
    /// Fair the patch so that it continues the surrounding curvature;
    /// otherwise the refined patch keeps its new vertices at the centroids
    /// they were split at.
    pub fair: bool,
    /// Loops with more edges than this are triangulated by projection onto
    /// their best-fit plane instead of the `O(n³)` minimum-weight search.
    /// Below 3, every loop goes to the projection.
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
    /// The fairing solver gave no finite solution, with either weights.
    Solver,
}

/// A loop that was filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilledLoop {
    /// The loop, as it was before the fill.
    pub id: LoopId,
    /// Triangles added.
    pub triangles_added: usize,
    /// Vertices added by refinement; each adds two triangles.
    pub vertices_added: usize,
    /// Whether fairing converged; `true` when it did not run.
    pub converged: bool,
    /// The Laplacian weights fairing used; `None` when it did not run.
    pub weights: Option<FairWeights>,
}

/// The edge weights of the Laplacian used to fair a patch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FairWeights {
    /// Cotangent weights, clamped to stay positive.
    Cotangent,
    /// Uniform weights, the fallback when the cotangent system fails.
    Uniform,
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
/// - The patch is refined to the density of the mesh around the loop
///   (Liepa's rule: centroid splits while a triangle is coarse for the mean
///   edge length at its corners, with Delaunay edge flips between rounds of
///   splits). Each new vertex has its triangle's corners as creation parents,
///   1/3 each.
/// - With `fair`, the new vertices are moved to minimise bending, holding
///   the loop and the ring of mesh vertices around it fixed: the discrete
///   bi-Laplacian, cotangent weights falling back to uniform ones
///   ([`FilledLoop::weights`]), solved by preconditioned conjugate gradient.
///   A solve that does not reach its tolerance keeps its best iterate and
///   reports `converged: false`; one that is not finite fails the loop with
///   [`FailReason::Solver`].
/// - Every selected loop is planned before the first write; the new
///   vertices of the successful patches are then added in creation order,
///   with their parents, and the triangles in one `add_triangles` call.
///   A failed loop is reported and writes nothing.
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

    let fans = VertexFans::build(mesh)?;

    let mut report = FillReport::default();
    let mut plan = Plan::default();
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
        let mut patch = refine(mesh, &fans, &edges, &l.vertices, &tris)?;
        let old = patch.old.len() as u32;
        new_chords.clear();
        for t in &patch.triangles {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                if a >= old || b >= old {
                    continue;
                }
                let (a, b) = (patch.old[a as usize], patch.old[b as usize]);
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
        let (mut converged, mut weights) = (true, None);
        if options.fair && patch.new_vertices() > 0 {
            match fair(mesh, &fans, &mut patch)? {
                Ok(r) => (converged, weights) = (r.converged, Some(r.weights)),
                Err(reason) => {
                    push(&mut report.failed, (l.id, reason))?;
                    continue;
                }
            }
        }
        merge_sorted(&mut chords, &new_chords)?;
        plan.add(&patch)?;
        push(
            &mut report.filled,
            FilledLoop {
                id: l.id,
                triangles_added: patch.triangles.len(),
                vertices_added: patch.new_vertices(),
                converged,
                weights,
            },
        )?;
    }
    plan.write(tx)?;
    Ok(report)
}

/// A patch corner: a mesh vertex, or a vertex the plan adds (by its index
/// among the plan's new vertices).
#[derive(Clone, Copy)]
enum Corner {
    Old(VertexId),
    New(u32),
}

/// A planned vertex: its position and its creation parents with weights.
type NewVertex = ([f32; 3], [(Corner, f32); 3]);

/// The vertices and triangles of every patch planned, ready to write.
#[derive(Default)]
struct Plan {
    /// New vertices in creation order; a new parent is always an earlier
    /// new vertex.
    vertices: Vec<NewVertex>,
    triangles: Vec<[Corner; 3]>,
}

impl Plan {
    /// Appends `patch`, translating its local indices.
    fn add(&mut self, patch: &RefinedPatch) -> Result<()> {
        let base = self.vertices.len();
        let old = patch.old.len();
        if u32::try_from(base + patch.new_vertices()).is_err() {
            return Err(OpsError::Store(StoreError::OutOfMemory));
        }
        let corner = |v: u32| match patch.old.get(v as usize) {
            Some(&id) => Corner::Old(id),
            None => Corner::New((base + v as usize - old) as u32),
        };
        reserve(&mut self.vertices, patch.new_vertices())?;
        for (p, parents) in patch.points[old..].iter().zip(&patch.parents) {
            let position = p.map(|x| x as f32);
            self.vertices
                .push((position, parents.map(|(v, w)| (corner(v), w))));
        }
        reserve(&mut self.triangles, patch.triangles.len())?;
        self.triangles
            .extend(patch.triangles.iter().map(|t| t.map(corner)));
        Ok(())
    }

    /// Writes the new vertices in creation order, with
    /// `add_vertex_with_parents`, then every triangle. Everything is reserved
    /// before the first write.
    fn write(&self, tx: &mut Transaction) -> Result<()> {
        if self.triangles.is_empty() {
            return Ok(());
        }
        let mut ids: Vec<VertexId> = Vec::new();
        reserve(&mut ids, self.vertices.len())?;
        let mut triangles: Vec<[VertexId; 3]> = Vec::new();
        reserve(&mut triangles, self.triangles.len())?;
        let id = |ids: &[VertexId], c: Corner| match c {
            Corner::Old(v) => v,
            Corner::New(i) => ids[i as usize],
        };
        for (position, parents) in &self.vertices {
            let parents = parents.map(|(c, w)| (id(&ids, c), w));
            ids.push(tx.add_vertex_with_parents(*position, &parents)?);
        }
        triangles.extend(self.triangles.iter().map(|t| t.map(|c| id(&ids, c))));
        tx.add_triangles(&triangles)?;
        Ok(())
    }
}

/// Merges the sorted `extra` into the sorted `into`, from the back.
fn merge_sorted(
    into: &mut Vec<(VertexId, VertexId)>,
    extra: &[(VertexId, VertexId)],
) -> Result<()> {
    reserve(into, extra.len())?;
    let (mut i, mut j) = (into.len(), extra.len());
    into.extend_from_slice(extra);
    for k in (0..into.len()).rev() {
        if j == 0 {
            break;
        }
        if i > 0 && into[i - 1] > extra[j - 1] {
            into[k] = into[i - 1];
            i -= 1;
        } else {
            into[k] = extra[j - 1];
            j -= 1;
        }
    }
    Ok(())
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

    use fast_mesh_store::{ChannelDesc, History, Mesh, TopologyPolicy};

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
        // Each centroid split adds one vertex and two triangles.
        let expected: Vec<FilledLoop> = loops
            .iter()
            .zip(&r.filled)
            .map(|(l, f)| FilledLoop {
                id: l.id,
                triangles_added: l.vertices.len() - 2 + 2 * f.vertices_added,
                vertices_added: f.vertices_added,
                converged: true,
                weights: None,
            })
            .collect();
        assert_eq!(r.filled, expected);
        assert!(r.filled.iter().all(|f| f.vertices_added > 0));
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
        let f = r.filled[0];
        assert_eq!(f.triangles_added, 400 - 2 + 2 * f.vertices_added);
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
    fn merge_keeps_order() {
        let mut v = vec![(1, 2), (3, 4), (7, 8)];
        merge_sorted(&mut v, &[(0, 1), (3, 5), (9, 9)]).unwrap();
        assert_eq!(v, [(0, 1), (1, 2), (3, 4), (3, 5), (7, 8), (9, 9)]);
        merge_sorted(&mut v, &[]).unwrap();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn collinear_hole_filled_by_projection() {
        // An 8-edge square hole with a collinear midpoint on each side.
        let mut mesh = grid_without(4, &[(1, 1), (2, 1), (1, 2), (2, 2)]);
        let options = FillOptions {
            max_edges: Some(8),
            dp_limit: 3,
            ..unfaired()
        };
        let (r, _) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert_eq!(r.failed, vec![]);
        assert_eq!(r.filled.len(), 1);
        let f = r.filled[0];
        assert_eq!(f.triangles_added, 8 - 2 + 2 * f.vertices_added);
        assert_sound(&mesh, 1);
    }

    #[test]
    fn large_flat_grid_hole_filled_at_default_limit() {
        // A square hole of 53 x 53 cells: 212 edges, mostly collinear.
        let cells: Vec<(u32, u32)> = (4..57).flat_map(|y| (4..57).map(move |x| (x, y))).collect();
        let mut mesh = grid_without(60, &cells);
        let options = FillOptions {
            max_edges: Some(220),
            ..unfaired()
        };
        let (r, _) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert_eq!(r.failed, vec![]);
        assert_eq!(r.filled.len(), 1);
        let f = r.filled[0];
        assert_eq!(f.triangles_added, 212 - 2 + 2 * f.vertices_added);
        assert_sound(&mesh, 1);
    }

    #[test]
    fn degenerate_only_triangulation_fails() {
        // A three-vertex loop a -> c -> b on collinear a, b, c: its only
        // triangle has no area.
        let positions = [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0, 0.0,
        ];
        let mut mesh = mesh_of(&positions, &[[0, 2, 3], [0, 4, 1], [1, 4, 2]]);
        let loops = boundary_loops(&mesh).unwrap();
        let sliver = loops.iter().find(|l| l.vertices == [0, 2, 1]).unwrap();
        let options = FillOptions {
            loops: Some(vec![sliver.id]),
            allow_unclean: true,
            ..unfaired()
        };
        let (r, empty) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert!(empty);
        assert_eq!(
            r.failed,
            vec![(sliver.id, FailReason::NoValidTriangulation)]
        );
    }

    /// Unit-sphere hole centres, at polar and azimuthal angles (degrees)
    /// away from the poles and from each other.
    fn hole_centres() -> [[f64; 3]; 3] {
        [(80.0, 10.0), (105.0, 130.0), (70.0, 250.0)].map(|(theta, phi): (f64, f64)| {
            let (t, p) = (theta.to_radians(), phi.to_radians());
            [t.sin() * p.cos(), t.sin() * p.sin(), t.cos()]
        })
    }

    fn angle_between(a: [f64; 3], b: [f64; 3]) -> f64 {
        let d: f64 = (0..3).map(|k| a[k] * b[k]).sum();
        let n = |v: [f64; 3]| (0..3).map(|k| v[k] * v[k]).sum::<f64>().sqrt();
        (d / (n(a) * n(b))).clamp(-1.0, 1.0).acos()
    }

    /// [`uv_sphere`]`(64, 128, 1)` with holes of angular radius 10, 20 and 30
    /// degrees around [`hole_centres`]: every triangle whose centroid lies
    /// within that angle is removed (as a chord, `2 sin(angle / 2)`).
    fn sphere_with_angular_holes() -> Mesh {
        let sphere = uv_sphere(64, 128, 1.0);
        let mut mesh = sphere;
        for (c, deg) in hole_centres().iter().zip([10.0f64, 20.0, 30.0]) {
            let chord = 2.0 * (deg.to_radians() / 2.0).sin();
            mesh = punch_holes(&mesh, &[c.map(|x| x as f32)], chord as f32);
        }
        mesh
    }

    fn vertex(mesh: &Mesh, v: VertexId) -> [f64; 3] {
        mesh.positions()[v as usize].map(f64::from)
    }

    /// The largest `| |p| - 1 |` over the vertices from `first` on, for each
    /// hole (by nearest centre), and how many vertices each hole got.
    fn radius_errors(mesh: &Mesh, first: usize) -> ([f64; 3], [usize; 3]) {
        let (mut worst, mut count) = ([0.0f64; 3], [0usize; 3]);
        for v in first..mesh.vertex_slots() {
            let p = vertex(mesh, v as VertexId);
            assert!(p.iter().all(|x| x.is_finite()));
            let h = (0..3)
                .min_by(|&a, &b| {
                    angle_between(p, hole_centres()[a])
                        .total_cmp(&angle_between(p, hole_centres()[b]))
                })
                .unwrap();
            let r = (0..3).map(|k| p[k] * p[k]).sum::<f64>().sqrt();
            worst[h] = worst[h].max((r - 1.0).abs());
            count[h] += 1;
        }
        (worst, count)
    }

    fn fill_sphere_holes(fair: bool) -> (Mesh, FillReport, usize) {
        let mut mesh = sphere_with_angular_holes();
        let first = mesh.vertex_slots();
        let options = FillOptions {
            max_edges: Some(1000),
            fair,
            ..FillOptions::default()
        };
        let (r, _) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert_eq!(r.filled.len(), 3);
        assert_sound(&mesh, 0);
        (mesh, r, first)
    }

    #[test]
    fn sphere_holes_are_punched_at_their_angular_radius() {
        let mesh = sphere_with_angular_holes();
        let loops = boundary_loops(&mesh).unwrap();
        assert_eq!(loops.len(), 3);
        // One stack is 180 / 64 degrees; the loop runs along the cut.
        let cell = (180.0f64 / 64.0).to_radians();
        for (c, deg) in hole_centres().iter().zip([10.0f64, 20.0, 30.0]) {
            let l = loops
                .iter()
                .find(|l| angle_between(vertex(&mesh, l.vertices[0]), *c) < 0.7)
                .unwrap();
            for &v in &l.vertices {
                let a = angle_between(vertex(&mesh, v), *c);
                assert!((a - deg.to_radians()).abs() < cell, "{deg}: {a}");
            }
        }
    }

    #[test]
    fn faired_patch_follows_sphere() {
        let (mesh, r, first) = fill_sphere_holes(true);
        let (worst, count) = radius_errors(&mesh, first);
        assert!(count.iter().all(|&c| c > 0), "{count:?}");
        for (h, e) in worst.iter().enumerate() {
            assert!(*e < 0.02, "hole {h}: radius error {e}");
        }
        let added: usize = r.filled.iter().map(|f| f.vertices_added).sum();
        assert_eq!(added, mesh.vertex_slots() - first);
        assert!(r.filled.iter().all(|f| f.weights.is_some()));
    }

    #[test]
    fn unfaired_patch_is_flatter() {
        let (faired, _, first) = fill_sphere_holes(true);
        let (flat, r, _) = fill_sphere_holes(false);
        let (with, _) = radius_errors(&faired, first);
        let (without, _) = radius_errors(&flat, first);
        for h in 0..3 {
            assert!(
                without[h] > with[h],
                "hole {h}: {} vs {}",
                without[h],
                with[h]
            );
        }
        // The 30-degree cap is 1 - cos 30 = 0.134 high; flat it misses by most of that.
        assert!(without[2] > 0.05);
        assert!(r.filled.iter().all(|f| f.weights.is_none() && f.converged));
    }

    /// The mean length of the edges of `tris`.
    fn mean_edge(mesh: &Mesh, tris: &[[VertexId; 3]]) -> f64 {
        let mut edges: Vec<(VertexId, VertexId)> = tris
            .iter()
            .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
            .map(|(a, b)| (a.min(b), a.max(b)))
            .collect();
        edges.sort_unstable();
        edges.dedup();
        let len = |(a, b): (VertexId, VertexId)| {
            let (p, q) = (vertex(mesh, a), vertex(mesh, b));
            (0..3).map(|k| (p[k] - q[k]).powi(2)).sum::<f64>().sqrt()
        };
        edges.iter().map(|&e| len(e)).sum::<f64>() / edges.len() as f64
    }

    #[test]
    fn density_matches_surroundings() {
        let mut mesh = sphere_with_angular_holes();
        let loops = boundary_loops(&mesh).unwrap();
        let slots = mesh.triangle_slots() as u32;
        for l in &loops {
            let around: Vec<[VertexId; 3]> = (0..slots)
                .filter_map(|t| mesh.triangle(t))
                .filter(|t| t.iter().any(|v| l.vertices.contains(v)))
                .collect();
            let target = mean_edge(&mesh, &around);
            let mut m = sphere_with_angular_holes();
            let options = FillOptions {
                loops: Some(vec![l.id]),
                ..FillOptions::default()
            };
            let (r, _) = edit(&mut m, |tx| fill_holes(tx, &options));
            assert!(r.filled[0].vertices_added > 0);
            let patch: Vec<[VertexId; 3]> = (slots..m.triangle_slots() as u32)
                .map(|t| m.triangle(t).unwrap())
                .collect();
            let ratio = mean_edge(&m, &patch) / target;
            assert!((0.5..=1.5).contains(&ratio), "ratio {ratio}");
        }
        // The loops' own mesh is untouched by the per-loop fills above.
        assert_eq!(boundary_loops(&mesh).unwrap(), loops);
        let options = FillOptions {
            max_edges: Some(1000),
            ..FillOptions::default()
        };
        edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert_sound(&mesh, 0);
    }

    #[test]
    fn fairing_thin_hole_no_nan() {
        // A sliver hole two cells high and sixty long in a flat grid.
        let cells: Vec<(u32, u32)> = (2..62).flat_map(|x| (3..5).map(move |y| (x, y))).collect();
        let mut mesh = grid_without(64, &cells);
        let first = mesh.vertex_slots();
        let options = FillOptions {
            max_edges: Some(200),
            ..FillOptions::default()
        };
        let (r, _) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        assert_eq!(r.filled.len(), 1, "{r:?}");
        let f = r.filled[0];
        assert!(f.vertices_added > 0);
        assert_eq!(f.vertices_added, mesh.vertex_slots() - first);
        assert_sound(&mesh, 1);
        for v in first..mesh.vertex_slots() {
            let p = vertex(&mesh, v as VertexId);
            assert!(p.iter().all(|x| x.is_finite()));
            // A flat surrounding gives a flat patch, inside the hole.
            assert!(p[2].abs() < 1e-5, "{p:?}");
            assert!(
                (2.0..=62.0).contains(&p[0]) && (3.0..=5.0).contains(&p[1]),
                "{p:?}"
            );
        }
        // Converged within the cap on this well-posed system, and said so.
        assert!(f.converged);
        assert!(f.weights.is_some());
    }

    #[test]
    fn new_vertices_interpolate_channels() {
        // A source channel holding x: every new vertex's value is a convex
        // combination of the loop's, so its parents were valid.
        let mut mesh = sphere_with_angular_holes();
        mesh.add_channel(ChannelDesc::source("x", 1).with_policy(TopologyPolicy::Interpolate))
            .unwrap();
        let ids: Vec<VertexId> = (0..mesh.vertex_slots() as VertexId).collect();
        let xs: Vec<f32> = ids
            .iter()
            .map(|&v| mesh.positions()[v as usize][0])
            .collect();
        let mut tx = mesh.begin_edit();
        tx.write_channel("x", &ids, &xs).unwrap();
        tx.commit().unwrap();
        let first = mesh.vertex_slots();
        let options = FillOptions {
            max_edges: Some(1000),
            ..FillOptions::default()
        };
        edit(&mut mesh, |tx| fill_holes(tx, &options));
        let values = mesh.channel("x").unwrap();
        assert!(mesh.vertex_slots() > first);
        for x in &values[first..] {
            assert!((-1.0..=1.0).contains(x), "{x}");
        }
    }

    #[test]
    fn undo_restores_after_fill() {
        let mut mesh = sphere_with_angular_holes();
        let before = parts(&mesh);
        let (vertices, triangles) = (mesh.vertex_count(), mesh.triangle_count());
        let mut history = History::new(1 << 26);
        let mut tx = mesh.begin_edit();
        let options = FillOptions {
            max_edges: Some(1000),
            ..FillOptions::default()
        };
        let r = fill_holes(&mut tx, &options).unwrap();
        assert!(r.filled.iter().any(|f| f.vertices_added > 0));
        let patch = tx.commit().unwrap();
        history.push(&mut mesh, patch).unwrap();
        assert!(mesh.vertex_count() > vertices);
        assert!(history.undo(&mut mesh).unwrap().is_some());
        assert_eq!(
            (mesh.vertex_count(), mesh.triangle_count()),
            (vertices, triangles)
        );
        let (positions, tris) = parts(&mesh);
        assert_eq!(&positions[..before.0.len()], &before.0[..]);
        assert_eq!(tris, before.1);
        assert!(boundary_loops(&mesh).unwrap().len() == 3);
    }

    #[test]
    fn out_of_memory_is_reported_and_writes_nothing() {
        let options = FillOptions {
            max_edges: Some(4),
            ..unfaired()
        };
        assert!(sweep(bowtie_holes(), |tx| fill_holes(tx, &options).map(drop)) > 5);
        // The projection path.
        let options = FillOptions {
            max_perimeter: Some(7.0),
            dp_limit: 8,
            ..unfaired()
        };
        assert!(sweep(wobbly_annulus(64), |tx| fill_holes(tx, &options).map(drop)) > 5);
        // Refinement, fairing and new vertices: two refined holes.
        let options = FillOptions {
            max_edges: Some(16),
            ..FillOptions::default()
        };
        let r = edit(&mut refined_holes(), |tx| fill_holes(tx, &options)).0;
        assert_eq!(r.filled.len(), 2);
        assert!(
            r.filled
                .iter()
                .all(|f| f.vertices_added > 0 && f.weights.is_some())
        );
        assert!(sweep(refined_holes(), |tx| fill_holes(tx, &options).map(drop)) > 20);
    }

    /// [`grid`]`(12)` without two 4 x 4 blocks of cells, wavy in `z`.
    fn refined_holes() -> Mesh {
        let cells: Vec<(u32, u32)> = (2..6)
            .flat_map(|y| (1..5).chain(7..11).map(move |x| (x, y)))
            .collect();
        let (mut positions, triangles) = parts(&grid_without(12, &cells));
        for p in positions.chunks_mut(3) {
            p[2] = (0.5 * p[0]).sin() * (0.3 * p[1]).cos();
        }
        mesh_of(&positions, &triangles)
    }

    #[test]
    fn solver_failure_writes_nothing() {
        // A NaN next to the first hole fails its fairing; the second is filled.
        let (mut positions, triangles) = parts(&refined_holes());
        positions[3 * 13 + 2] = f32::NAN; // vertex (0, 1), next to corner (1, 2)
        let mut mesh = mesh_of(&positions, &triangles);
        let loops = boundary_loops(&mesh).unwrap();
        let holes: Vec<&BoundaryLoop> = loops.iter().filter(|l| l.vertices.len() == 16).collect();
        let options = FillOptions {
            max_edges: Some(16),
            ..FillOptions::default()
        };
        let slots = (mesh.vertex_slots(), mesh.triangle_slots());
        let (r, _) = edit(&mut mesh, |tx| fill_holes(tx, &options));
        let failed = holes.iter().find(|l| l.vertices.contains(&27)).unwrap();
        assert_eq!(r.failed, vec![(failed.id, FailReason::Solver)]);
        assert_eq!(r.filled.len(), 1);
        let f = r.filled[0];
        assert_eq!(
            (mesh.vertex_slots(), mesh.triangle_slots()),
            (slots.0 + f.vertices_added, slots.1 + f.triangles_added)
        );
        let left = boundary_loops(&mesh).unwrap();
        assert!(left.iter().any(|l| l.id == failed.id));
        assert_eq!(left.len(), 2);
    }
}
