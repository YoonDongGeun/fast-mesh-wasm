//! A read-only report of what is wrong with a mesh.

use std::fmt;

use fast_mesh_bvh::Bvh;
use fast_mesh_store::{Mesh, TriangleId, VertexId, par};

use fast_mesh_query::self_intersecting_pairs;

use crate::topology::{
    BoundaryLoop, Component, EdgeMap, components, loops_with, pinch_vertices, point, triangle_area,
};
use crate::weld::{check_tolerance, weld_roots};
use crate::{Result, filled, push, reserve};

/// Options for [`analyze`].
#[derive(Clone, Copy)]
pub struct AnalyzeOptions<'a> {
    /// Vertices at most this far apart (local units) count as coincident.
    /// 0, the default, counts exact duplicates only.
    pub weld_tolerance: f32,
    /// A BVH of the mesh, to check for self-intersection; `None`, the
    /// default, skips the check.
    pub self_intersection: Option<&'a Bvh>,
    /// The most entries kept in each list that has a count beside it; the
    /// counts are always exact. Default 1000.
    pub max_listed: usize,
}

impl Default for AnalyzeOptions<'_> {
    fn default() -> Self {
        AnalyzeOptions {
            weld_tolerance: 0.0,
            self_intersection: None,
            max_listed: 1000,
        }
    }
}

impl fmt::Debug for AnalyzeOptions<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnalyzeOptions")
            .field("weld_tolerance", &self.weld_tolerance)
            .field("self_intersection", &self.self_intersection.map(|_| "Bvh"))
            .field("max_listed", &self.max_listed)
            .finish()
    }
}

/// What [`analyze`] found.
#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    /// The boundary loops, as [`boundary_loops`](crate::boundary_loops)
    /// returns them.
    pub loops: Vec<BoundaryLoop>,
    /// How many groups of two or more coincident vertices there are.
    pub coincident_groups: usize,
    /// The groups, by smallest id, each ascending.
    pub coincident: Vec<Vec<VertexId>>,
    /// How many triangles are degenerate.
    pub degenerate: usize,
    /// The degenerate triangles, ascending.
    pub degenerate_list: Vec<TriangleId>,
    /// How many triangles repeat the vertex set of a triangle with a smaller id.
    pub duplicate: usize,
    /// Those triangles, ascending.
    pub duplicate_list: Vec<TriangleId>,
    /// How many edges have three or more triangles.
    pub non_manifold_edges: usize,
    /// Those edges as `(min, max)`, ascending.
    pub non_manifold_edge_list: Vec<(VertexId, VertexId)>,
    /// The vertices on non-manifold edges, ascending.
    pub non_manifold_vertices: Vec<VertexId>,
    /// The vertices whose triangle fan splits into several pieces, ascending.
    pub pinch_vertices: Vec<VertexId>,
    /// The connected components, by smallest triangle id.
    pub components: Vec<Component>,
    /// How many interior edges are used in the same direction by both their
    /// triangles.
    pub orientation_conflicts: usize,
    /// Whether two triangles that share no vertex intersect; `None` when no
    /// BVH was given.
    pub self_intersects: Option<bool>,
}

/// A triangle is degenerate by area when its area is at most this times its
/// longest edge squared (the default of `remove_degenerate`).
pub(crate) const AREA_EPSILON: f64 = 1e-12;

/// [`analyze`] runs its passes side by side from this many triangles, with
/// the `parallel` feature.
///
/// Measured on the i5-13600K on terrains (no defects, no BVH): at 512
/// triangles parallel is 1.1 to 1.2 times slower, at 1,152 1.1 to 1.3 times
/// faster, and from 4k on steadily 1.25 times (the edge map's sort, which
/// the other passes wait for, stays sequential).
pub(crate) const PAR_MIN_ANALYZE_TRIS: usize = 2048;

/// Reports the defects of `mesh` without changing it.
///
/// - Loops, components, edge kinds, pinch vertices and orientation conflicts
///   are computed from the triangles with three distinct vertex ids; the
///   others are skipped there, as everywhere in this crate.
/// - Coincident vertices: groups of alive vertices linked through pairs at
///   most `weld_tolerance` apart, as a weld would merge them.
/// - Degenerate triangles: a repeated vertex id, or an area of at most
///   `1e-12` times the longest edge squared.
/// - Duplicate triangles: every alive triangle but the smallest id of each
///   set of triangles on the same three vertex ids, in any order or winding.
/// - Non-manifold vertices are the ends of non-manifold edges. Pinch
///   vertices, whose fan splits into pieces (the corner where two holes
///   touch), are listed apart.
/// - Orientation conflicts: interior edges whose two triangles both run
///   along them in the same direction.
/// - Self-intersection ([`self_intersecting_pairs`]): two alive triangles
///   that share no vertex id, neither degenerate, intersect or touch. Pairs
///   sharing a vertex are not tested.
///
/// The lists with a count beside them (`coincident`, `degenerate_list`,
/// `duplicate_list` and `non_manifold_edge_list`) keep their first
/// `options.max_listed` entries; the counts and the other lists are complete.
///
/// # Errors
/// [`OpsError::InvalidInput`] when `weld_tolerance` is negative, NaN or
/// infinite; [`OpsError::Query`] when the BVH does not match the mesh; and
/// [`OpsError::Store`] with [`StoreError::OutOfMemory`] when a temporary
/// cannot grow.
///
/// [`StoreError::OutOfMemory`]: fast_mesh_store::StoreError::OutOfMemory
/// [`OpsError::InvalidInput`]: crate::OpsError::InvalidInput
/// [`OpsError::Query`]: crate::OpsError::Query
/// [`OpsError::Store`]: crate::OpsError::Store
pub fn analyze(mesh: &Mesh, options: &AnalyzeOptions) -> Result<Analysis> {
    let max = options.max_listed;
    // Checked first, so a bad tolerance fails before any pass runs.
    check_tolerance(options.weld_tolerance)?;
    // The four passes read the mesh only; on large meshes they run side by
    // side. Their errors are returned in pass order, so the first error is
    // the same however they ran.
    let parallel = mesh.triangle_count() >= PAR_MIN_ANALYZE_TRIS;
    let ((coincident, topology), (scans, self_intersects)) = par::join(
        parallel,
        || {
            par::join(
                parallel,
                || groups(&weld_roots(mesh, options.weld_tolerance)?, max),
                || topology_pass(mesh, max, parallel),
            )
        },
        || {
            par::join(
                parallel,
                || scan_pass(mesh, max),
                || self_intersection_pass(mesh, options.self_intersection),
            )
        },
    );
    let (coincident_groups, coincident) = coincident?;
    let topology = topology?;
    let (degenerate, degenerate_list, duplicate, duplicate_list) = scans?;
    let self_intersects = self_intersects?;

    Ok(Analysis {
        loops: topology.loops,
        coincident_groups,
        coincident,
        degenerate,
        degenerate_list,
        duplicate,
        duplicate_list,
        non_manifold_edges: topology.non_manifold_edges,
        non_manifold_edge_list: topology.non_manifold_edge_list,
        non_manifold_vertices: topology.non_manifold_vertices,
        pinch_vertices: topology.pinch_vertices,
        components: topology.components,
        orientation_conflicts: topology.orientation_conflicts,
        self_intersects,
    })
}

/// What [`analyze`] reads from the edge map.
struct Topology {
    loops: Vec<BoundaryLoop>,
    non_manifold_edges: usize,
    non_manifold_edge_list: Vec<(VertexId, VertexId)>,
    non_manifold_vertices: Vec<VertexId>,
    pinch_vertices: Vec<VertexId>,
    components: Vec<Component>,
    orientation_conflicts: usize,
}

/// The edge map and everything [`analyze`] reads from it: the loops, the
/// components and the rest side by side when `parallel`, errors in that
/// order.
fn topology_pass(mesh: &Mesh, max: usize, parallel: bool) -> Result<Topology> {
    let edges = EdgeMap::build(mesh)?;
    let (loops, (components, rest)) = par::join(
        parallel,
        || loops_with(mesh, &edges),
        || {
            par::join(
                parallel,
                || components(mesh, &edges),
                || edge_kinds(mesh, &edges, max),
            )
        },
    );
    let (loops, components) = (loops?, components?);
    let (pinch_vertices, non_manifold, orientation_conflicts) = rest?;
    let (non_manifold_edges, non_manifold_edge_list, non_manifold_vertices) = non_manifold;
    Ok(Topology {
        loops,
        non_manifold_edges,
        non_manifold_edge_list,
        non_manifold_vertices,
        pinch_vertices,
        components,
        orientation_conflicts,
    })
}

/// The non-manifold edge count, edge list and vertices.
type NonManifold = (usize, Vec<(VertexId, VertexId)>, Vec<VertexId>);

/// The pinch vertices, the non-manifold edges and vertices, and the
/// orientation conflicts.
fn edge_kinds(
    mesh: &Mesh,
    edges: &EdgeMap,
    max: usize,
) -> Result<(Vec<VertexId>, NonManifold, usize)> {
    let pinch_vertices = pinch_vertices(mesh, edges)?;
    let mut non_manifold_edges = 0;
    let mut non_manifold_edge_list = Vec::new();
    let mut on_non_manifold = filled(mesh.vertex_slots(), false)?;
    let mut orientation_conflicts = 0;
    for ((a, b), tris) in edges.edges() {
        match *tris {
            [_] => {}
            [s, t] => {
                if runs_forward(mesh, s, a, b) == runs_forward(mesh, t, a, b) {
                    orientation_conflicts += 1;
                }
            }
            _ => {
                non_manifold_edges += 1;
                if non_manifold_edge_list.len() < max {
                    push(&mut non_manifold_edge_list, (a, b))?;
                }
                on_non_manifold[a as usize] = true;
                on_non_manifold[b as usize] = true;
            }
        }
    }
    let mut non_manifold_vertices = Vec::new();
    for (v, &on) in on_non_manifold.iter().enumerate() {
        if on {
            push(&mut non_manifold_vertices, v as VertexId)?;
        }
    }
    let non_manifold = (
        non_manifold_edges,
        non_manifold_edge_list,
        non_manifold_vertices,
    );
    Ok((pinch_vertices, non_manifold, orientation_conflicts))
}

/// The degenerate count and list, then the duplicate count and list.
type Scans = (usize, Vec<TriangleId>, usize, Vec<TriangleId>);

/// The degenerate and duplicate triangle scans of [`analyze`].
fn scan_pass(mesh: &Mesh, max: usize) -> Result<Scans> {
    let mut degenerate = 0;
    let mut degenerate_list = Vec::new();
    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        if is_degenerate(mesh, tri, AREA_EPSILON) {
            degenerate += 1;
            if degenerate_list.len() < max {
                push(&mut degenerate_list, t)?;
            }
        }
    }
    let mut duplicate_list = duplicate_triangles(mesh)?;
    let duplicate = duplicate_list.len();
    duplicate_list.truncate(max);
    Ok((degenerate, degenerate_list, duplicate, duplicate_list))
}

/// Whether two triangles of `mesh` intersect, given a BVH.
fn self_intersection_pass(mesh: &Mesh, bvh: Option<&Bvh>) -> Result<Option<bool>> {
    let Some(bvh) = bvh else {
        return Ok(None);
    };
    let mut pairs = Vec::new();
    self_intersecting_pairs(mesh, bvh, &mut pairs)?;
    Ok(Some(!pairs.is_empty()))
}

/// The alive triangles that repeat the vertex set of a triangle with a smaller
/// id, ascending.
pub(crate) fn duplicate_triangles(mesh: &Mesh) -> Result<Vec<TriangleId>> {
    // Each alive triangle's sorted vertex ids, then its id: sorting brings
    // each set together, smallest id first.
    let mut sets: Vec<([VertexId; 3], TriangleId)> = Vec::new();
    reserve(&mut sets, mesh.triangle_count())?;
    for t in 0..mesh.triangle_slots() as TriangleId {
        if let Some(mut key) = mesh.triangle(t) {
            key.sort_unstable();
            push(&mut sets, (key, t))?;
        }
    }
    sets.sort_unstable();
    let mut out = Vec::new();
    for w in sets.windows(2) {
        if w[0].0 == w[1].0 {
            push(&mut out, w[1].1)?;
        }
    }
    out.sort_unstable();
    Ok(out)
}

/// Whether triangle `t` runs along its edge `a`-`b` from `a` to `b`.
fn runs_forward(mesh: &Mesh, t: TriangleId, a: VertexId, b: VertexId) -> bool {
    let [x, y, z] = mesh.triangles()[t as usize];
    (x, y) == (a, b) || (y, z) == (a, b) || (z, x) == (a, b)
}

/// A repeated vertex id, or an area of at most `area_epsilon` times the
/// longest edge squared.
pub(crate) fn is_degenerate(mesh: &Mesh, [a, b, c]: [VertexId; 3], area_epsilon: f64) -> bool {
    if a == b || b == c || a == c {
        return true;
    }
    let [p, q, r] = [a, b, c].map(|v| point(mesh, v));
    let d2 = |u: [f64; 3], v: [f64; 3]| (0..3).map(|k| (u[k] - v[k]).powi(2)).sum::<f64>();
    let longest = d2(p, q).max(d2(q, r)).max(d2(r, p));
    triangle_area(mesh, [a, b, c]) <= area_epsilon * longest
}

/// Marks a vertex whose group is not numbered yet, in [`groups`].
const UNNUMBERED: usize = usize::MAX;

/// The number of weld groups of two or more vertices, and the first `max` of
/// them by smallest id, each ascending. `roots[v]` is the smallest id of
/// `v`'s group.
fn groups(roots: &[VertexId], max: usize) -> Result<(usize, Vec<Vec<VertexId>>)> {
    // A group's root is its smallest member, so visiting ids in order meets
    // each group at its second-smallest member first, after its root.
    let mut index = filled(roots.len(), UNNUMBERED)?;
    let mut count = 0;
    let mut out: Vec<Vec<VertexId>> = Vec::new();
    for (v, &r) in roots.iter().enumerate() {
        let r = r as usize;
        if r == v {
            continue;
        }
        if index[r] == UNNUMBERED {
            index[r] = count;
            count += 1;
            if out.len() < max {
                let mut group = Vec::new();
                push(&mut group, r as VertexId)?;
                push(&mut out, group)?;
            }
        }
        if let Some(group) = out.get_mut(index[r]) {
            push(group, v as VertexId)?;
        }
    }
    Ok((count, out))
}

#[cfg(test)]
mod tests {
    use fast_mesh_bvh::BuildOptions;
    use fast_mesh_store::StoreError;

    use super::*;
    use crate::test_util::*;
    use crate::{OpsError, fail_allocations_after};

    fn run(mesh: &Mesh) -> Analysis {
        analyze(mesh, &AnalyzeOptions::default()).unwrap()
    }

    #[test]
    fn analyze_counts_on_damaged_fixture() {
        let mesh = damaged_sphere();
        assert_eq!(mesh.triangle_count(), 170);
        assert_eq!(mesh.vertex_count(), 94);
        let a = run(&mesh);

        // 3 holes, 4 slits and the square's border.
        let mut lens: Vec<usize> = a.loops.iter().map(|l| l.vertices.len()).collect();
        lens.sort_unstable();
        assert_eq!(lens, [3, 3, 3, 4, 4, 4, 4, 4]);
        assert!(a.loops.iter().all(|l| l.clean && l.closed));
        let seqs: Vec<&[VertexId]> = a.loops.iter().map(|l| l.vertices.as_slice()).collect();
        // The hole of quad (1, 0), triangle [1, 13, 14], walked against it.
        assert!(seqs.contains(&&[1, 14, 13][..]));
        // The slit at vertex 10: triangle [10, 22, 23] now uses copy 86.
        assert!(seqs.contains(&&[10, 23, 86, 22][..]));
        assert!(seqs.contains(&&[90, 91, 92, 93][..]));

        assert_eq!(a.coincident_groups, 4);
        assert_eq!(a.coincident, [[10, 86], [25, 87], [31, 88], [52, 89]]);
        assert_eq!(a.degenerate, 2);
        assert_eq!(a.degenerate_list, [166, 167]);
        assert_eq!(a.duplicate, 1);
        assert_eq!(a.duplicate_list, [165]);
        assert_eq!(a.non_manifold_edges, 3);
        assert_eq!(a.non_manifold_edge_list, [(77, 78), (77, 85), (78, 85)]);
        assert_eq!(a.non_manifold_vertices, [77, 78, 85]);
        assert!(a.pinch_vertices.is_empty());
        // The sphere with the copy (165 triangles plus one), and the square.
        assert_eq!(a.components.len(), 2);
        assert_eq!(a.components[0].triangles, (0..166).collect::<Vec<_>>());
        assert_eq!(a.components[1].triangles, [168, 169]);
        assert!(a.components.iter().all(|c| !c.closed));
        assert!((a.components[1].area - 1.0).abs() < 1e-12);
        // 5 flips, each against its 3 neighbours.
        assert_eq!(a.orientation_conflicts, 15);
        assert_eq!(a.self_intersects, None);
    }

    #[test]
    fn analyze_clean_sphere_reports_nothing() {
        let a = run(&uv_sphere(8, 12, 1.0));
        assert!(a.loops.is_empty());
        assert_eq!((a.coincident_groups, a.degenerate, a.duplicate), (0, 0, 0));
        assert_eq!((a.non_manifold_edges, a.orientation_conflicts), (0, 0));
        assert!(a.non_manifold_vertices.is_empty() && a.pinch_vertices.is_empty());
        assert_eq!(a.components.len(), 1);
        assert!(a.components[0].closed);
    }

    #[test]
    fn analyze_bowtie_reports_pinch_and_two_clean_holes() {
        let a = run(&bowtie_holes());
        assert_eq!(a.pinch_vertices, [12]);
        assert!(a.non_manifold_vertices.is_empty());
        assert_eq!(a.non_manifold_edges, 0);
        let seqs: Vec<&[VertexId]> = a.loops.iter().map(|l| l.vertices.as_slice()).collect();
        assert_eq!(seqs.len(), 3);
        assert_eq!(&seqs[1..], [&[6, 11, 12, 7][..], &[12, 17, 18, 13][..]]);
        assert!(a.loops.iter().all(|l| l.clean && l.closed));
        assert_eq!(a.components.len(), 1);
        assert_eq!(a.orientation_conflicts, 0);
    }

    #[test]
    fn analyze_fin_reports_non_manifold_vertices_not_pinches() {
        let a = run(&fin());
        assert_eq!(a.non_manifold_edges, 1);
        assert_eq!(a.non_manifold_edge_list, [(0, 1)]);
        assert_eq!(a.non_manifold_vertices, [0, 1]);
        assert!(a.pinch_vertices.is_empty());
        assert!(a.loops.iter().all(|l| !l.clean));
    }

    #[test]
    fn analyze_does_not_modify() {
        let mesh = damaged_sphere();
        let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
        let (revision, before) = (mesh.revision(), parts(&mesh));
        let options = AnalyzeOptions {
            self_intersection: Some(&bvh),
            ..Default::default()
        };
        let first = analyze(&mesh, &options).unwrap();
        assert_eq!(mesh.revision(), revision);
        assert_eq!(parts(&mesh), before);
        assert_eq!(analyze(&mesh, &options).unwrap(), first);
    }

    #[test]
    fn analyze_self_intersection_flag() {
        let boxes = crossed_boxes();
        let sphere = uv_sphere(8, 12, 1.0);
        for (mesh, want) in [(&boxes, true), (&sphere, false)] {
            let bvh = Bvh::build(mesh, &BuildOptions::default()).unwrap();
            let options = AnalyzeOptions {
                self_intersection: Some(&bvh),
                ..Default::default()
            };
            assert_eq!(analyze(mesh, &options).unwrap().self_intersects, Some(want));
            assert_eq!(run(mesh).self_intersects, None);
        }
        // A BVH of another mesh is the query's error.
        let bvh = Bvh::build(&sphere, &BuildOptions::default()).unwrap();
        let options = AnalyzeOptions {
            self_intersection: Some(&bvh),
            ..Default::default()
        };
        assert_eq!(
            analyze(&boxes, &options),
            Err(OpsError::Query(fast_mesh_query::QueryError::MeshMismatch))
        );
    }

    #[test]
    fn analyze_degenerate_by_area_and_duplicates_in_any_order() {
        // The unit square's triangle [0, 1, 2], its reversed copy, a sliver
        // through the midpoint 4 of edge 0-1 (exact in f32), a needle far
        // below the area threshold, and a repeated id.
        let positions = [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.5, 1e-7,
            0.0,
        ];
        let tris = [
            [0, 1, 2],
            [0, 2, 3],
            [2, 1, 0], // duplicate of 0, reversed
            [0, 4, 1], // collinear
            [0, 5, 1], // area 5e-8 against L^2 = 1: kept
            [3, 3, 2], // repeated id
            [1, 2, 0], // duplicate of 0, rotated
        ];
        let indices: Vec<u32> = tris.iter().flatten().copied().collect();
        let mesh = Mesh::from_buffers(&positions, &indices).unwrap();
        let a = run(&mesh);
        assert_eq!(a.degenerate_list, [3, 5]);
        assert_eq!(a.duplicate_list, [2, 6]);
        assert_eq!((a.degenerate, a.duplicate), (2, 2));
    }

    #[test]
    fn analyze_coincident_with_tolerance() {
        // Vertices 0 to 3 on the x axis at 0, 0.1, 0.2 and 0.5; 4 at 0 too.
        let positions = [
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.5, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 1.0,
            0.0,
        ];
        let indices = [0, 1, 5, 1, 2, 5, 2, 3, 5, 3, 4, 5];
        let mesh = Mesh::from_buffers(&positions, &indices).unwrap();
        let with = |weld_tolerance| {
            let options = AnalyzeOptions {
                weld_tolerance,
                ..Default::default()
            };
            analyze(&mesh, &options).unwrap().coincident
        };
        assert_eq!(with(0.0), [vec![0, 4]]);
        // Chained: 0-1 and 1-2 are within 0.15, 0-2 is not.
        assert_eq!(with(0.15), [vec![0, 1, 2, 4]]);
        assert_eq!(with(0.05), [vec![0, 4]]);
        assert_eq!(with(1.5), [vec![0, 1, 2, 3, 4, 5]]);
        for bad in [-1.0, f32::NAN, f32::INFINITY] {
            let options = AnalyzeOptions {
                weld_tolerance: bad,
                ..Default::default()
            };
            assert!(matches!(
                analyze(&mesh, &options),
                Err(OpsError::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn analyze_truncates_lists_but_not_counts() {
        let mesh = damaged_sphere();
        let options = AnalyzeOptions {
            max_listed: 1,
            ..Default::default()
        };
        let a = analyze(&mesh, &options).unwrap();
        assert_eq!((a.coincident_groups, a.coincident), (4, vec![vec![10, 86]]));
        assert_eq!((a.degenerate, a.degenerate_list), (2, vec![166]));
        assert_eq!((a.duplicate, a.duplicate_list), (1, vec![165]));
        assert_eq!(
            (a.non_manifold_edges, a.non_manifold_edge_list),
            (3, vec![(77, 78)])
        );
        assert_eq!(a.non_manifold_vertices.len(), 3);
        assert_eq!(a.loops.len(), 8);
        let none = AnalyzeOptions {
            max_listed: 0,
            ..Default::default()
        };
        let a = analyze(&mesh, &none).unwrap();
        assert_eq!((a.degenerate, a.degenerate_list.len()), (2, 0));
    }

    /// A damaged mesh of over [`PAR_MIN_ANALYZE_TRIS`] triangles: a sphere
    /// with holes, flipped triangles, a duplicate, a repeated id and a slit
    /// (vertex 500 copied, with one triangle moved onto the copy), beside two
    /// crossed boxes.
    fn large_damaged() -> Mesh {
        let sphere = uv_sphere(96, 192, 1.0);
        let holed = punch_holes(&sphere, &[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], 0.2);
        let every_97th: Vec<TriangleId> = (0..holed.triangle_count() as u32).step_by(97).collect();
        let (mut positions, mut triangles) =
            parts(&combine(&[&flip(&holed, &every_97th), &crossed_boxes()]));
        let copy = (positions.len() / 3) as VertexId;
        positions.extend_from_within(1500..1503);
        let t = triangles.iter().position(|t| t.contains(&500)).unwrap();
        triangles[t] = triangles[t].map(|v| if v == 500 { copy } else { v });
        triangles.extend([triangles[10], [5, 5, 6]]);
        mesh_of(&positions, &triangles)
    }

    #[test]
    fn analyze_bit_identical() {
        let mesh = large_damaged();
        assert!(mesh.triangle_count() >= PAR_MIN_ANALYZE_TRIS);
        let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
        for weld_tolerance in [0.0, 0.02] {
            let options = AnalyzeOptions {
                weld_tolerance,
                self_intersection: Some(&bvh),
                ..Default::default()
            };
            let a = assert_parallel_identical("analyze", || analyze(&mesh, &options).unwrap());
            assert_eq!(a.self_intersects, Some(true));
            assert!(a.loops.len() >= 2 && a.orientation_conflicts > 0);
            assert_eq!((a.duplicate, a.degenerate), (1, 1));
            assert!(a.coincident_groups > 0);
        }
        // A pass's error is the same however the passes ran.
        let other = Bvh::build(&uv_sphere(8, 12, 1.0), &BuildOptions::default()).unwrap();
        let mut bad = AnalyzeOptions {
            self_intersection: Some(&other),
            ..Default::default()
        };
        let e = assert_parallel_identical("analyze error", || analyze(&mesh, &bad));
        assert!(matches!(e, Err(OpsError::Query(_))));
        // A bad tolerance fails before any pass runs.
        bad.weld_tolerance = -1.0;
        let before = par::parallel_entries();
        assert!(matches!(
            analyze(&mesh, &bad),
            Err(OpsError::InvalidInput(_))
        ));
        assert_eq!(par::parallel_entries(), before);
    }

    #[test]
    fn analyze_oom_is_clean() {
        let mesh = damaged_sphere();
        let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
        let options = AnalyzeOptions {
            self_intersection: Some(&bvh),
            ..Default::default()
        };
        let expected = analyze(&mesh, &options).unwrap();
        let mut n = 0;
        loop {
            fail_allocations_after(Some(n));
            let got = analyze(&mesh, &options);
            fail_allocations_after(None);
            match got {
                Ok(a) => {
                    assert_eq!(a, expected);
                    break;
                }
                Err(e) => assert_eq!(e, OpsError::Store(StoreError::OutOfMemory)),
            }
            n += 1;
        }
        assert!(n > 10, "only {n} fallible reservations");
    }
}
