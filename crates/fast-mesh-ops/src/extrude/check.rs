//! The self-intersection check of a planned extrusion, run before writing.

use fast_mesh_bvh::{Aabb, BuildOptions, Bvh, NodeKind};
use fast_mesh_query::{Rigid, intersecting_pairs, self_intersecting_pairs};
use fast_mesh_store::{Mesh, TriangleId, VertexId};
use robust::{Coord, orient2d};

use crate::topology::{dot, proper_triangle, sub};
use crate::{Result, push, reserve};

/// Two triangles on one edge count as folded onto each other when the angle
/// between them has a cosine above this: less than about 1e-3 radians.
const FOLD_COS: f64 = 1.0 - 5e-7;

/// Triangles an operation is about to write, over planned vertex ids: an id
/// below `mesh.vertex_slots()` is an existing vertex (at its `moved` position
/// when listed there), and `slots + i` is the new vertex `added[i]`.
pub(crate) struct Planned<'a> {
    /// The mesh before the write.
    pub(crate) mesh: &'a Mesh,
    /// Existing vertices that move, with their new positions, by id. Only
    /// `replaced` triangles use them.
    pub(crate) moved: &'a [(VertexId, [f32; 3])],
    /// The positions of the new vertices.
    pub(crate) added: &'a [[f32; 3]],
    /// The new or rewritten triangles.
    pub(crate) triangles: &'a [[VertexId; 3]],
    /// Existing triangles that `triangles` replace, ascending: left out of
    /// the check, since they will not exist as they are.
    pub(crate) replaced: &'a [TriangleId],
}

/// What [`intersections`] found: the planned triangles (indices into
/// [`Planned::triangles`]) and the existing ones that take part in an
/// intersecting pair, each list ascending without repeats.
#[derive(Default)]
pub(crate) struct Found {
    planned: Vec<u32>,
    existing: Vec<TriangleId>,
}

impl Found {
    /// How many ids [`Found::write_final`] writes at most: reserve that much
    /// before the write, so that nothing is allocated after it.
    pub(crate) fn len(&self) -> usize {
        self.planned.len() + self.existing.len()
    }

    /// Fills `out` (reserved for [`Found::len`] ids) with the final ids of
    /// what was found, ascending: `ids[i]` is the id planned triangle `i`
    /// got.
    pub(crate) fn write_final(&self, ids: &[TriangleId], out: &mut Vec<TriangleId>) {
        debug_assert!(out.capacity() - out.len() >= self.len());
        out.extend(self.planned.iter().map(|&i| ids[i as usize]));
        out.extend_from_slice(&self.existing);
        out.sort_unstable();
        out.dedup();
    }

    fn finish(mut self) -> Self {
        self.planned.sort_unstable();
        self.planned.dedup();
        self.existing.sort_unstable();
        self.existing.dedup();
        self
    }
}

/// Whether the three corners are collinear (or coincide), decided exactly:
/// all three axis projections have zero orientation.
fn collinear(p: [[f32; 3]; 3]) -> bool {
    (0..3).all(|k| {
        let (x, y) = ((k + 1) % 3, (k + 2) % 3);
        let c = |q: [f32; 3]| Coord {
            x: f64::from(q[x]),
            y: f64::from(q[y]),
        };
        orient2d(c(p[0]), c(p[1]), c(p[2])) == 0.0
    })
}

/// The triangles that would intersect after the write, with at least one
/// planned triangle in every pair:
///
/// - the pairs of [`self_intersecting_pairs`] on the written mesh that
///   involve a planned triangle: triangles that touch or cross and share no
///   vertex id, with the same exact test, degenerate triangles skipped;
/// - triangles sharing an edge that fold flat onto each other (past
///   [`FOLD_COS`]), which that test leaves out: a wall pushed straight back
///   along the face next to it, say.
///
/// The planned triangles go into a small mesh of their own; the existing
/// triangles whose boxes touch one of theirs into another, and the two are
/// searched against each other and the first against itself. So the cost
/// follows the size of the changed region, not of the mesh, past one scan
/// of the mesh's triangle boxes.
///
/// # Errors
/// [`OpsError::Store`](crate::OpsError::Store) when memory runs out.
pub(crate) fn intersections(p: &Planned) -> Result<Found> {
    let mut found = Found::default();
    if p.triangles.is_empty() {
        return Ok(found);
    }
    let mesh = p.mesh;
    let slots = mesh.vertex_slots() as VertexId;
    let position = |v: VertexId| -> [f32; 3] {
        if v >= slots {
            p.added[(v - slots) as usize]
        } else if let Ok(i) = p.moved.binary_search_by_key(&v, |&(id, _)| id) {
            p.moved[i].1
        } else {
            mesh.positions()[v as usize]
        }
    };
    let planned = local_mesh(p.triangles, position)?;
    let planned_bvh = Bvh::build(&planned, &BuildOptions::default())?;
    let mut pairs: Vec<(TriangleId, TriangleId)> = Vec::new();
    self_intersecting_pairs(&planned, &planned_bvh, &mut pairs)?;
    reserve(&mut found.planned, 2 * pairs.len())?;
    found
        .planned
        .extend(pairs.iter().flat_map(|&(a, b)| [a, b]));

    let near = near(p, &planned, &planned_bvh)?;
    if !near.is_empty() {
        crossings(p, &near, &planned, &planned_bvh, &mut found)?;
    }
    folds(p, &near, position, &mut found)?;
    Ok(found.finish())
}

/// `triangles` as a mesh of their own, with one vertex per distinct id.
fn local_mesh(
    triangles: &[[VertexId; 3]],
    position: impl Fn(VertexId) -> [f32; 3],
) -> Result<Mesh> {
    let mut ids: Vec<VertexId> = Vec::new();
    reserve(&mut ids, 3 * triangles.len())?;
    ids.extend(triangles.iter().flatten());
    ids.sort_unstable();
    ids.dedup();
    let mut flat: Vec<f32> = Vec::new();
    reserve(&mut flat, 3 * ids.len())?;
    flat.extend(ids.iter().flat_map(|&v| position(v)));
    let mut index: Vec<u32> = Vec::new();
    reserve(&mut index, 3 * triangles.len())?;
    let local = |v: &VertexId| ids.binary_search(v).expect("listed above") as u32;
    index.extend(triangles.iter().flatten().map(local));
    Ok(Mesh::from_buffers(&flat, &index)?)
}

/// The existing triangles, not replaced, whose box touches the box of a
/// planned one, ascending.
fn near(p: &Planned, planned: &Mesh, planned_bvh: &Bvh) -> Result<Vec<TriangleId>> {
    let mesh = p.mesh;
    let mut near: Vec<TriangleId> = Vec::new();
    let Some(root) = planned_bvh.root() else {
        return Ok(near);
    };
    let reach = planned_bvh.node(root).bounds;
    let mut replaced = p.replaced.iter().copied().peekable();
    for t in 0..mesh.triangle_slots() as TriangleId {
        if replaced.next_if_eq(&t).is_some() {
            continue;
        }
        let Some(tri) = proper_triangle(mesh, t) else {
            continue;
        };
        let bounds = Aabb::of_triangle(mesh.positions(), tri);
        if !reach.overlaps(&bounds) {
            continue;
        }
        let mut hit = false;
        planned_bvh.traverse(|_, node| {
            if hit || !node.bounds.overlaps(&bounds) {
                return false;
            }
            if let NodeKind::Leaf { triangles } = node.kind {
                hit = triangles.iter().any(|&x| {
                    Aabb::of_triangle(planned.positions(), planned.triangles()[x as usize])
                        .overlaps(&bounds)
                });
            }
            true
        });
        if hit {
            push(&mut near, t)?;
        }
    }
    Ok(near)
}

/// Appends the pairs of a planned triangle and one of `near` that intersect
/// and share no vertex.
fn crossings(
    p: &Planned,
    near: &[TriangleId],
    planned: &Mesh,
    planned_bvh: &Bvh,
    found: &mut Found,
) -> Result<()> {
    let mesh = p.mesh;
    let mut tris: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut tris, near.len())?;
    tris.extend(near.iter().map(|&t| mesh.triangles()[t as usize]));
    let existing = local_mesh(&tris, |v| mesh.positions()[v as usize])?;
    let existing_bvh = Bvh::build(&existing, &BuildOptions::default())?;
    let mut pairs: Vec<(TriangleId, TriangleId)> = Vec::new();
    intersecting_pairs(
        planned,
        planned_bvh,
        &existing,
        &existing_bvh,
        &Rigid::IDENTITY,
        &mut pairs,
    )?;
    let corners =
        |m: &Mesh, t: TriangleId| m.triangles()[t as usize].map(|v| m.positions()[v as usize]);
    for &(a, b) in &pairs {
        let (ours, theirs) = (p.triangles[a as usize], tris[b as usize]);
        if ours.iter().any(|v| theirs.contains(v))
            || collinear(corners(planned, a))
            || collinear(corners(&existing, b))
        {
            continue;
        }
        push(&mut found.planned, a)?;
        push(&mut found.existing, near[b as usize])?;
    }
    Ok(())
}

/// Appends the pairs of triangles on one edge that fold flat onto each
/// other: planned ones, or a planned one and one of `near`.
fn folds(
    p: &Planned,
    near: &[TriangleId],
    position: impl Fn(VertexId) -> [f32; 3],
    found: &mut Found,
) -> Result<()> {
    let ours = p.triangles.len() as u32;
    // Every edge with the triangle on it (`near` past the planned ones) and
    // the vertex across it, grouped by edge, planned triangles first.
    let mut uses: Vec<([VertexId; 2], u32, VertexId)> = Vec::new();
    reserve(&mut uses, 3 * (p.triangles.len() + near.len()))?;
    let theirs = near.iter().map(|&t| p.mesh.triangles()[t as usize]);
    for (i, tri) in p.triangles.iter().copied().chain(theirs).enumerate() {
        for k in 0..3 {
            let (a, b, apex) = (tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]);
            uses.push(([a.min(b), a.max(b)], i as u32, apex));
        }
    }
    uses.sort_unstable();
    for group in uses.chunk_by(|x, y| x.0 == y.0) {
        let [from, to] = group[0].0;
        let corners = |apex: VertexId| [from, to, apex].map(&position);
        for (i, &(_, a, apex)) in group.iter().enumerate() {
            if a >= ours {
                break;
            }
            for &(_, b, other) in &group[i + 1..] {
                if !folded(corners(apex), corners(other)) {
                    continue;
                }
                push(&mut found.planned, a)?;
                if b < ours {
                    push(&mut found.planned, b)?;
                } else {
                    push(&mut found.existing, near[(b - ours) as usize])?;
                }
            }
        }
    }
    Ok(())
}

/// Whether two triangles, each given as the same edge and then its own third
/// corner, fold flat onto each other.
fn folded(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> bool {
    if collinear(a) || collinear(b) {
        return false;
    }
    let [a, b] = [a, b].map(|t| t.map(|q| q.map(f64::from)));
    let e = sub(a[1], a[0]);
    // The apexes, seen along the edge.
    let across = |apex: [f64; 3]| {
        let d = sub(apex, a[0]);
        let t = dot(d, e) / dot(e, e);
        sub(d, e.map(|x| t * x))
    };
    let (pa, pb) = (across(a[2]), across(b[2]));
    dot(pa, pb) > FOLD_COS * (dot(pa, pa) * dot(pb, pb)).sqrt()
}
