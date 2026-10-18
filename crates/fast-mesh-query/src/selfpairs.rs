//! Self-intersection: the triangle pairs of one mesh that intersect each
//! other.

use fast_mesh_bvh::{Aabb, Bvh, NodeId, NodeKind};
use fast_mesh_store::{Mesh, TriangleId, par};

use crate::pair::{
    PAIR_STACK, PAR_PARTS, Step, oom, split_pair, traverse_subtree_pairs_with, try_push,
};
use crate::ray::points;
use crate::tritri::{P3, collinear, tri_tri_intersect};
use crate::{Appender, Result, Rigid, check};

/// Appends to `out` every pair `(a, b)` with `a < b` of alive triangles of
/// `mesh` that intersect or touch and share no vertex id. The appended pairs
/// are sorted; elements already in `out` are kept and stay in front.
///
/// The test is exact on the `f64` coordinates, as for
/// [`intersecting_pairs`](crate::intersecting_pairs). Pairs that share a
/// vertex id are not tested at all: neighbours always touch along their
/// shared edge or corner, so such a pair is never reported, even when the two
/// triangles also cross elsewhere (a fold back onto a neighbour). Degenerate
/// triangles, whose corners are collinear or repeat a vertex id, are skipped
/// too. Distinct vertices at the same position are distinct vertices: two
/// triangles meeting only through such a pair of vertices touch, and are
/// reported.
///
/// The BVH is walked against itself so that each unordered pair of nodes is
/// visited once, and each pair of triangles tested once.
///
/// # Errors
/// [`QueryError::MeshMismatch`](crate::QueryError::MeshMismatch) or
/// [`QueryError::StaleBvh`](crate::QueryError::StaleBvh) when `bvh` does not
/// match `mesh`, and [`QueryError::OutOfMemory`](crate::QueryError::OutOfMemory)
/// when `out` cannot grow. On any error `out` keeps its original length.
pub fn self_intersecting_pairs(
    mesh: &Mesh,
    bvh: &Bvh,
    out: &mut Vec<(TriangleId, TriangleId)>,
) -> Result<()> {
    check(mesh, bvh)?;
    let mut ap = Appender::new(out);
    let Some(root) = bvh.root() else {
        return Ok(());
    };
    if par::enabled() && !par::force_sequential() && mesh.triangle_count() >= PAR_MIN_SELF_TRIS {
        // Each part's pairs in part order, sorted below.
        let parts = split(bvh, root)?;
        let found = par::map_chunks(&parts, 1, 0, oom, |_, part| {
            let mut pairs = Vec::new();
            let mut failed = None;
            part[0].search(bvh, &mut |a, b| {
                if let Some(pair) = intersecting(mesh, a, b)
                    && let Err(e) = try_push(&mut pairs, pair)
                {
                    failed = Some(e);
                    return false;
                }
                true
            });
            failed.map_or(Ok(pairs), Err)
        })?;
        for pairs in &found {
            ap.extend(pairs)?;
        }
    } else {
        let mut failed = None;
        Part::Within(root).search(bvh, &mut |a, b| {
            if let Some(pair) = intersecting(mesh, a, b)
                && let Err(e) = ap.push(pair)
            {
                failed = Some(e);
                return false;
            }
            true
        });
        if let Some(e) = failed {
            return Err(e);
        }
    }
    ap.appended().sort_unstable();
    Ok(())
}

/// Triangles from which [`self_intersecting_pairs`] splits its search into
/// [`Part`]s searched in parallel, with the `parallel` feature.
///
/// Measured on the i5-13600K on terrains (no pair intersects): at 128
/// triangles parallel is 1.2 to 1.5 times slower, at 512 between 1.3 times
/// slower and 1.3 times faster from run to run, at 1,058 three times faster
/// and at 32k nine times. Meshes that do cross themselves gain more.
const PAR_MIN_SELF_TRIS: usize = 1024;

/// A part of the self-intersection search.
#[derive(Debug, Clone, Copy)]
enum Part {
    /// Every pair of triangles under one node.
    Within(NodeId),
    /// Every pair of a triangle under the first node and one under the
    /// second, two disjoint subtrees whose boxes overlap.
    Across(NodeId, NodeId),
}

impl Part {
    /// Calls `test(a, b)` on every pair of triangles of the part that may
    /// intersect, once each, until it returns `false`.
    fn search(self, bvh: &Bvh, test: &mut impl FnMut(TriangleId, TriangleId) -> bool) {
        match self {
            Part::Within(root) => within(bvh, root, test),
            Part::Across(p, q) => {
                across(bvh, p, q, test);
            }
        }
    }
}

/// Cuts the search of the tree below `root` into disjoint [`Part`]s, one
/// level at a time, until there are [`PAR_PARTS`] or more, or none can be
/// cut further. A node's `Within` becomes its children's two `Within`s and,
/// when their boxes overlap, their `Across`; an `Across` becomes the
/// `Across`es of its child pairs whose boxes overlap ([`split_pair`]). Each
/// pair of triangles the whole search tests is in exactly one part, and in
/// none of the dropped ones. The parts come in a fixed order for a given
/// tree.
fn split(bvh: &Bvh, root: NodeId) -> Result<Vec<Part>> {
    let overlap = |p: &Aabb, q: &Aabb| p.overlaps(q).then_some(());
    let mut parts = Vec::new();
    try_push(&mut parts, Part::Within(root))?;
    while parts.len() < PAR_PARTS {
        let mut next = Vec::new();
        let mut cut = false;
        for &part in &parts {
            match part {
                Part::Within(n) => match bvh.node(n).kind {
                    NodeKind::Leaf { .. } => try_push(&mut next, part)?,
                    NodeKind::Inner { left, right } => {
                        cut = true;
                        try_push(&mut next, Part::Within(left))?;
                        try_push(&mut next, Part::Within(right))?;
                        if bvh.node(left).bounds.overlaps(&bvh.node(right).bounds) {
                            try_push(&mut next, Part::Across(left, right))?;
                        }
                    }
                },
                Part::Across(p, q) => {
                    match split_pair(bvh, p, bvh, q, &Rigid::IDENTITY, &overlap) {
                        None => try_push(&mut next, part)?,
                        Some(children) => {
                            cut = true;
                            for (p, q) in children.into_iter().flatten() {
                                try_push(&mut next, Part::Across(p, q))?;
                            }
                        }
                    }
                }
            }
        }
        parts = next;
        if !cut {
            break;
        }
    }
    Ok(parts)
}

/// Calls `test` on the pairs under `root` that may intersect, until it
/// returns `false`.
fn within(bvh: &Bvh, root: NodeId, test: &mut impl FnMut(TriangleId, TriangleId) -> bool) {
    let mut ok = true;
    // Each unordered pair of distinct leaves has one lowest common ancestor,
    // with one leaf under each child; each pair inside one leaf is that
    // leaf's. So walking every node once, pairing its two children's
    // subtrees or a leaf's triangles among themselves, tests every pair of
    // triangles once.
    bvh.traverse_from(root, |_, view| {
        if !ok {
            return false;
        }
        match view.kind {
            NodeKind::Leaf { triangles } => {
                for (i, &a) in triangles.iter().enumerate() {
                    for &b in &triangles[i + 1..] {
                        if !test(a, b) {
                            ok = false;
                            return false;
                        }
                    }
                }
            }
            NodeKind::Inner { left, right } => {
                if bvh.node(left).bounds.overlaps(&bvh.node(right).bounds) {
                    ok = across(bvh, left, right, test);
                }
            }
        }
        ok
    });
}

/// Calls `test` on the pairs of a triangle under `p` and one under `q` that
/// may intersect, until it returns `false`; returns `false` then.
fn across(
    bvh: &Bvh,
    p: NodeId,
    q: NodeId,
    test: &mut impl FnMut(TriangleId, TriangleId) -> bool,
) -> bool {
    let mut ok = true;
    traverse_subtree_pairs_with::<PAIR_STACK, f64>(
        bvh,
        p,
        bvh,
        q,
        &Rigid::IDENTITY,
        |p, q| p.overlaps(q).then_some(0.0),
        |va, vb, _| {
            let (NodeKind::Leaf { triangles: ta }, NodeKind::Leaf { triangles: tb }) =
                (va.kind, vb.kind)
            else {
                return Step::Descend;
            };
            for &a in ta {
                for &b in tb {
                    if !test(a, b) {
                        ok = false;
                        return Step::Stop;
                    }
                }
            }
            Step::Skip
        },
    );
    ok
}

/// `(min, max)` of `a` and `b` when they share no vertex id, neither is
/// degenerate, and they intersect.
fn intersecting(mesh: &Mesh, a: TriangleId, b: TriangleId) -> Option<(TriangleId, TriangleId)> {
    let (va, vb) = (mesh.triangles()[a as usize], mesh.triangles()[b as usize]);
    if va.iter().any(|v| vb.contains(v)) {
        return None;
    }
    let (pa, pb) = (tri(mesh, a), tri(mesh, b));
    let apart = (0..3).any(|k| {
        let lo = |p: &[P3; 3]| p[0][k].min(p[1][k]).min(p[2][k]);
        let hi = |p: &[P3; 3]| p[0][k].max(p[1][k]).max(p[2][k]);
        lo(&pa) > hi(&pb) || lo(&pb) > hi(&pa)
    });
    // A repeated vertex id makes the corners collinear. Degeneracy only
    // matters for pairs that intersect, so it is decided last.
    (!apart && tri_tri_intersect(pa, pb) && !collinear(&pa) && !collinear(&pb))
        .then_some((a.min(b), a.max(b)))
}

/// Triangle `t` in `f64`.
fn tri(mesh: &Mesh, t: TriangleId) -> [P3; 3] {
    points(mesh, t).map(|p| p.map(f64::from))
}

#[cfg(test)]
mod tests {
    use fast_mesh_bvh::Quality;

    use super::*;
    use crate::test_util::{build, wavy};
    use crate::tritri::{collinear, tri_tri_intersect};
    use crate::{QueryError, fail_allocations_after};

    /// A closed UV sphere of radius 1 with outward winding.
    fn uv_sphere(stacks: u32, slices: u32) -> Mesh {
        use std::f64::consts::PI;
        let mut positions = vec![0.0f32, 0.0, 1.0];
        for i in 1..stacks {
            let theta = PI * f64::from(i) / f64::from(stacks);
            for j in 0..slices {
                let phi = 2.0 * PI * f64::from(j) / f64::from(slices);
                positions.extend([
                    (theta.sin() * phi.cos()) as f32,
                    (theta.sin() * phi.sin()) as f32,
                    theta.cos() as f32,
                ]);
            }
        }
        let south = (positions.len() / 3) as u32;
        positions.extend([0.0, 0.0, -1.0]);
        let ring = |i: u32, j: u32| 1 + (i - 1) * slices + j % slices;
        let mut indices = Vec::new();
        for j in 0..slices {
            indices.extend([0, ring(1, j), ring(1, j + 1)]);
            for i in 1..stacks - 1 {
                let (a, b, c, d) = (
                    ring(i, j),
                    ring(i + 1, j),
                    ring(i + 1, j + 1),
                    ring(i, j + 1),
                );
                indices.extend([a, b, c, a, c, d]);
            }
            indices.extend([ring(stacks - 1, j), south, ring(stacks - 1, j + 1)]);
        }
        Mesh::from_buffers(&positions, &indices).unwrap()
    }

    /// Two closed boxes, `[0, 2]^3` and `[1, 3]^3`, that interpenetrate
    /// without sharing a vertex.
    fn crossed_boxes() -> Mesh {
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        for (base, lo, hi) in [(0u32, 0.0f32, 2.0f32), (8, 1.0, 3.0)] {
            for i in 0..8 {
                positions.extend((0..3).map(|k| if i >> k & 1 == 1 { hi } else { lo }));
            }
            let faces: [[u32; 3]; 12] = [
                [0, 4, 6],
                [0, 6, 2],
                [1, 3, 7],
                [1, 7, 5],
                [0, 1, 5],
                [0, 5, 4],
                [2, 6, 7],
                [2, 7, 3],
                [0, 2, 3],
                [0, 3, 1],
                [4, 5, 7],
                [4, 7, 6],
            ];
            indices.extend(faces.iter().flatten().map(|v| v + base));
        }
        Mesh::from_buffers(&positions, &indices).unwrap()
    }

    /// Every qualifying pair by brute force over all triangle pairs.
    fn reference(mesh: &Mesh) -> Vec<(TriangleId, TriangleId)> {
        let tri = |t: TriangleId| {
            let [a, b, c] = mesh.triangles()[t as usize];
            let p = |v: u32| mesh.positions()[v as usize].map(f64::from);
            ([a, b, c], [p(a), p(b), p(c)])
        };
        let n = mesh.triangle_slots() as TriangleId;
        let mut out = Vec::new();
        for a in (0..n).filter(|&t| mesh.is_triangle_alive(t)) {
            for b in (a + 1..n).filter(|&t| mesh.is_triangle_alive(t)) {
                let ((va, pa), (vb, pb)) = (tri(a), tri(b));
                let shared = va.iter().any(|v| vb.contains(v));
                if !shared && !collinear(&pa) && !collinear(&pb) && tri_tri_intersect(pa, pb) {
                    out.push((a, b));
                }
            }
        }
        out
    }

    fn pairs(mesh: &Mesh, bvh: &Bvh) -> Vec<(TriangleId, TriangleId)> {
        let mut out = Vec::new();
        self_intersecting_pairs(mesh, bvh, &mut out).unwrap();
        out
    }

    #[test]
    fn self_pairs_crossed_boxes_nonempty_and_sphere_empty() {
        let boxes = crossed_boxes();
        let sphere = uv_sphere(8, 12);
        for q in [Quality::Sah, Quality::Fast] {
            let got = pairs(&boxes, &build(&boxes, q));
            assert!(!got.is_empty());
            assert_eq!(got, reference(&boxes));
            assert!(got.iter().all(|&(a, b)| a < 12 && b >= 12));
            assert!(pairs(&sphere, &build(&sphere, q)).is_empty());
        }
        // Default leaves hold several triangles: pairs inside one leaf count.
        let bvh = Bvh::build(&boxes, &Default::default()).unwrap();
        assert_eq!(pairs(&boxes, &bvh), reference(&boxes));
    }

    #[test]
    fn self_pairs_skip_vertex_sharing_neighbours() {
        // Every triangle of a closed sphere touches its neighbours, through
        // shared vertex ids only.
        let sphere = uv_sphere(16, 24);
        let bvh = build(&sphere, Quality::Sah);
        assert_eq!(pairs(&sphere, &bvh), vec![]);
        // Two wavy sheets, one shifted into the other: they cross, and only
        // the pairs between the sheets, which share no vertex, are reported.
        let (sheet, lifted) = (wavy(4), wavy(4));
        let mut positions: Vec<f32> = sheet.positions().iter().flatten().copied().collect();
        let base = (positions.len() / 3) as u32;
        positions.extend(
            lifted
                .positions()
                .iter()
                .flat_map(|p| [p[0] + 0.25, p[1] + 0.5, p[2] + 0.5]),
        );
        let mut indices: Vec<u32> = sheet.triangles().iter().flatten().copied().collect();
        indices.extend(lifted.triangles().iter().flatten().map(|v| v + base));
        let mesh = Mesh::from_buffers(&positions, &indices).unwrap();
        for q in [Quality::Sah, Quality::Fast] {
            let got = pairs(&mesh, &build(&mesh, q));
            assert!(!got.is_empty());
            assert_eq!(got, reference(&mesh));
        }
    }

    #[test]
    fn self_pairs_skip_degenerate_and_keep_coincident_vertices() {
        // A triangle crossing a flat square, as one with a repeated id, as a
        // collinear one, and as a proper one.
        #[rustfmt::skip]
        let positions = [
            0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0, // 0-3 square
            1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, // 4-6 crossing
            1.0, 1.0, 0.0, // 7, collinear with 4 and 5
            0.0, 0.0, 0.0, // 8, at vertex 0's position
            -1.0, 0.0, 1.0, 0.0, -1.0, 1.0, // 9-10, above the square's corner
        ];
        #[rustfmt::skip]
        let indices = [
            0, 1, 2, 0, 2, 3, // triangles 0-1: the square
            4, 5, 5, // 2: a repeated id
            4, 5, 7, // 3: collinear
            4, 5, 6, // 4: proper, through the square's diagonal
            8, 9, 10, // 5: touching the square only at vertex 8
        ];
        let mesh = Mesh::from_buffers(&positions, &indices).unwrap();
        let bvh = build(&mesh, Quality::Sah);
        // Triangles 2 and 3 cross the square too but are degenerate. Triangle
        // 4 meets the diagonal shared by 0 and 1; triangle 5 touches both at
        // the corner where vertex 8 coincides with vertex 0, sharing no id.
        assert_eq!(pairs(&mesh, &bvh), vec![(0, 4), (0, 5), (1, 4), (1, 5)]);
        assert_eq!(pairs(&mesh, &bvh), reference(&mesh));
    }

    #[test]
    fn self_pairs_appends_sorted_and_checks_the_bvh() {
        let boxes = crossed_boxes();
        let bvh = build(&boxes, Quality::Sah);
        let mut out = vec![(99, 99)];
        self_intersecting_pairs(&boxes, &bvh, &mut out).unwrap();
        assert_eq!(out[0], (99, 99));
        assert_eq!(&out[1..], reference(&boxes).as_slice());

        let other = crossed_boxes();
        assert_eq!(
            self_intersecting_pairs(&other, &bvh, &mut out),
            Err(QueryError::MeshMismatch)
        );
        let mut edited = crossed_boxes();
        let stale = build(&edited, Quality::Sah);
        let mut tx = edited.begin_edit();
        tx.set_positions(&[0], &[[0.5, 0.0, 0.0]]).unwrap();
        tx.commit().unwrap();
        assert_eq!(
            self_intersecting_pairs(&edited, &stale, &mut out),
            Err(QueryError::StaleBvh)
        );
    }

    #[test]
    fn self_pairs_oom_keeps_out() {
        let boxes = crossed_boxes();
        let bvh = build(&boxes, Quality::Sah);
        let want = reference(&boxes);
        for n in 0..want.len() {
            let mut out = vec![(7, 7)];
            fail_allocations_after(Some(n));
            let got = self_intersecting_pairs(&boxes, &bvh, &mut out);
            fail_allocations_after(None);
            assert_eq!(got, Err(QueryError::OutOfMemory));
            assert_eq!(out, vec![(7, 7)]);
        }
    }

    #[test]
    fn self_pairs_bit_identical() {
        use crate::test_util::{assert_parallel_identical, crossed_ripples};
        let mesh = crossed_ripples(60);
        assert!(mesh.triangle_count() >= PAR_MIN_SELF_TRIS);
        for quality in [Quality::Sah, Quality::Fast] {
            let bvh = build(&mesh, quality);
            let found = assert_parallel_identical("self pairs", || pairs(&mesh, &bvh));
            assert!(found.len() > 500, "{quality:?}: {} pairs", found.len());
        }
    }
}
