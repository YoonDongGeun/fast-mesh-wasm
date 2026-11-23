//! The self-intersection check of a planned extrusion, run before writing.

use fast_mesh_bvh::{Aabb, BuildOptions, Bvh};
use fast_mesh_query::self_intersecting_pairs_where;
use fast_mesh_store::{Mesh, TriangleId, VertexId};
use robust::{Coord, orient2d};

use crate::topology::{dot, proper_triangle, sub};
use crate::{Result, filled, push, reserve};

/// A planned triangle's group when it has none.
pub(crate) const NO_GROUP: u32 = u32::MAX;

/// Cells per axis of the grid that finds the triangles near the planned ones.
const GRID: usize = 64;

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
    /// The group of each of `triangles`, or [`NO_GROUP`]. Triangles of one
    /// group do not move relative to each other, or tile a planar polygon,
    /// so they get no new intersections among themselves.
    pub(crate) groups: &'a [u32],
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
/// Pairs within one of [`Planned::groups`] are left out.
///
/// The search runs on a mesh of the planned triangles and the existing ones
/// whose boxes touch one of theirs, so past one scan of the mesh's triangle
/// boxes its cost follows the size of the changed region.
///
/// # Errors
/// [`OpsError::Store`](crate::OpsError::Store) when memory runs out.
///
/// [`self_intersecting_pairs`]: fast_mesh_query::self_intersecting_pairs
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
    let near = near(p, position)?;
    // The planned triangles, then the ones near them.
    let ours = p.triangles.len();
    let mut triangles: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut triangles, ours + near.len())?;
    triangles.extend_from_slice(p.triangles);
    triangles.extend(near.iter().map(|&t| mesh.triangles()[t as usize]));
    debug_assert_eq!(p.groups.len(), ours);
    let keep = |a: TriangleId, b: TriangleId| {
        let group = |t: TriangleId| p.groups.get(t as usize);
        match (group(a), group(b)) {
            (None, None) => false,
            (Some(x), Some(y)) => x != y || *x == NO_GROUP,
            _ => true,
        }
    };
    let mut note = |t: TriangleId| match (t as usize).checked_sub(ours) {
        Some(i) => push(&mut found.existing, near[i]),
        None => push(&mut found.planned, t),
    };

    let around = local_mesh(&triangles, position)?;
    let bvh = Bvh::build(&around, &BuildOptions::default())?;
    let mut pairs: Vec<(TriangleId, TriangleId)> = Vec::new();
    self_intersecting_pairs_where(&around, &bvh, keep, &mut pairs)?;
    for &(a, b) in &pairs {
        note(a)?;
        note(b)?;
    }
    folds(&triangles, keep, position, |a, b| {
        note(a)?;
        note(b)
    })?;
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

/// The existing triangles, not replaced, whose box may touch the box of a
/// planned one, ascending: those whose box reaches a cell of a coarse grid
/// that a planned box reaches too.
fn near(p: &Planned, position: impl Fn(VertexId) -> [f32; 3]) -> Result<Vec<TriangleId>> {
    let bounds_of = |tri: [VertexId; 3]| {
        let mut b = Aabb::EMPTY;
        for v in tri {
            b.grow_point(position(v));
        }
        b
    };
    let mut reach = Aabb::EMPTY;
    for &tri in p.triangles {
        reach.grow(&bounds_of(tri));
    }
    // Never decreasing along an axis, so boxes that share a point share a cell.
    let scale: [f64; 3] = std::array::from_fn(|k| {
        let extent = f64::from(reach.max[k]) - f64::from(reach.min[k]);
        if extent > 0.0 {
            GRID as f64 / extent
        } else {
            0.0
        }
    });
    let cell = |x: f32, k: usize| {
        let c = (f64::from(x) - f64::from(reach.min[k])) * scale[k];
        (c.max(0.0) as usize).min(GRID - 1)
    };
    // One bit per cell along x; a row per `(y, z)`.
    let rows = |b: &Aabb| {
        let [x0, y0, z0] = [0, 1, 2].map(|k| cell(b.min[k], k));
        let [x1, y1, z1] = [0, 1, 2].map(|k| cell(b.max[k], k));
        let bits = (u64::MAX << x0) & (u64::MAX >> (GRID - 1 - x1));
        let rows = (z0..=z1).flat_map(move |z| (y0..=y1).map(move |y| z * GRID + y));
        (bits, rows)
    };
    let mut grid = filled(GRID * GRID, 0u64)?;
    for &tri in p.triangles {
        let (bits, rows) = rows(&bounds_of(tri));
        for row in rows {
            grid[row] |= bits;
        }
    }
    let mesh = p.mesh;
    let mut near: Vec<TriangleId> = Vec::new();
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
        let (bits, mut rows) = rows(&bounds);
        if rows.any(|row| grid[row] & bits != 0) {
            push(&mut near, t)?;
        }
    }
    Ok(near)
}

/// Calls `fold` on every pair of `triangles` that `keep` accepts, shares an
/// edge and folds flat onto each other.
fn folds(
    triangles: &[[VertexId; 3]],
    keep: impl Fn(TriangleId, TriangleId) -> bool,
    position: impl Fn(VertexId) -> [f32; 3],
    mut fold: impl FnMut(TriangleId, TriangleId) -> Result<()>,
) -> Result<()> {
    // Every edge with the triangle on it and the vertex across it, by edge.
    let mut uses: Vec<([VertexId; 2], u32, VertexId)> = Vec::new();
    reserve(&mut uses, 3 * triangles.len())?;
    for (i, tri) in triangles.iter().enumerate() {
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
            for &(_, b, other) in &group[i + 1..] {
                if keep(a, b) && folded(corners(apex), corners(other)) {
                    fold(a, b)?;
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
