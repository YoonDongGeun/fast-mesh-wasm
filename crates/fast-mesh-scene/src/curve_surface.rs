//! The component of a mesh a curve lies on, for
//! [`Scene::trim_by_curve`](crate::Scene::trim_by_curve).
//!
//! A component is a set of triangles joined through shared edges of any kind,
//! as in [`remove_small_components`](fast_mesh_ops::remove_small_components):
//! on a mesh with UVs an edge also joins across a seam (two edges at exactly
//! the same positions), so the islands of one surface are one component.
//! Triangles with a repeated vertex id belong to no component.

use fast_mesh_bvh::Bvh;
use fast_mesh_query::{Region, triangles_in};
use fast_mesh_store::{Mesh, StoreError, TriangleId, VertexId};

use crate::cut::closest_barycentric_on_triangle;
use crate::{Result, SceneError};

const NONE: u32 = u32::MAX;

/// How far a curve point may be from a component beyond its distance to the
/// mesh and still lie on it, in units of the mesh's largest coordinate: a few
/// `f32` steps, enough for a point mapped to the world and back.
const ON_SURFACE: f64 = 1e-6;

fn oom() -> SceneError {
    SceneError::Store(StoreError::OutOfMemory)
}

/// `n` copies of `value`.
fn filled<T: Copy>(n: usize, value: T) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(|_| oom())?;
    out.resize(n, value);
    Ok(out)
}

/// The vertices of triangle `t` when it is alive and has three distinct ones.
fn proper_triangle(mesh: &Mesh, t: TriangleId) -> Option<[VertexId; 3]> {
    let [a, b, c] = mesh.triangle(t)?;
    (a != b && b != c && a != c).then_some([a, b, c])
}

/// Which vertices count as one where triangles are joined: on a mesh with UVs,
/// those at exactly one position, the two sides of a seam. Dead and later
/// vertices stand for themselves until they are joined.
pub(crate) struct Weld(Option<Vec<VertexId>>);

impl Weld {
    fn of(mesh: &Mesh) -> Result<Weld> {
        if mesh.uvs().is_none() {
            return Ok(Weld(None));
        }
        let mut keyed: Vec<([u32; 3], VertexId)> = Vec::new();
        keyed
            .try_reserve_exact(mesh.vertex_count())
            .map_err(|_| oom())?;
        for v in 0..mesh.vertex_slots() as VertexId {
            if let Some(p) = mesh.position(v)
                && p.iter().all(|x| x.is_finite())
            {
                // Adding 0.0 turns -0.0 into 0.0.
                keyed.push((p.map(|x| (x + 0.0).to_bits()), v));
            }
        }
        keyed.sort_unstable();
        let mut roots = filled(mesh.vertex_slots(), 0)?;
        for (v, root) in roots.iter_mut().enumerate() {
            *root = v as VertexId;
        }
        for pair in keyed.windows(2) {
            if pair[0].0 == pair[1].0 {
                roots[pair[1].1 as usize] = roots[pair[0].1 as usize];
            }
        }
        Ok(Weld(Some(roots)))
    }

    /// The vertex standing for `v`.
    #[inline]
    pub(crate) fn root(&self, v: VertexId) -> VertexId {
        match &self.0 {
            Some(roots) => roots.get(v as usize).copied().unwrap_or(v),
            None => v,
        }
    }

    /// Makes `root` stand for `v`, a new vertex at its position.
    pub(crate) fn join(&mut self, v: VertexId, root: VertexId) {
        if let Some(roots) = &mut self.0 {
            roots.extend(roots.len() as VertexId..=v);
            roots[v as usize] = root;
        }
    }
}

/// For every vertex, the triangles using one welded to it, ascending.
pub(crate) struct Fans {
    /// Vertex `v`'s triangles are `tris[starts[v]..starts[v + 1]]`.
    starts: Vec<u32>,
    tris: Vec<u32>,
}

impl Fans {
    /// The fans of triangles `0..slots`, whose vertices `corners` gives
    /// (`None` leaves a triangle out), over `vertices` vertex slots.
    pub(crate) fn build(
        vertices: usize,
        slots: usize,
        weld: &Weld,
        corners: impl Fn(usize) -> Option<[VertexId; 3]>,
    ) -> Result<Fans> {
        if slots > (NONE / 3) as usize {
            return Err(SceneError::InvalidInput(
                "mesh has too many triangles to trim",
            ));
        }
        let roots = |c: [VertexId; 3]| distinct(c.map(|v| weld.root(v)));
        let mut starts = filled(vertices + 1, 0u32)?;
        for t in 0..slots {
            for v in corners(t).into_iter().flat_map(roots) {
                starts[v as usize + 1] += 1;
            }
        }
        // Prefix sums give each fan's end; filling backwards from it leaves
        // every fan ascending and `starts` at the fans' beginnings.
        for v in 0..vertices {
            starts[v + 1] += starts[v];
        }
        let total = starts[vertices];
        let mut tris = filled(total as usize, 0u32)?;
        for t in (0..slots).rev() {
            for v in corners(t).into_iter().flat_map(roots) {
                let end = &mut starts[v as usize + 1];
                *end -= 1;
                tris[*end as usize] = t as u32;
            }
        }
        starts.copy_within(1.., 0);
        starts[vertices] = total;
        Ok(Fans { starts, tris })
    }

    /// The triangles using a vertex welded to `root`, ascending. None for a
    /// vertex that is not a root or came later.
    #[inline]
    pub(crate) fn around(&self, root: VertexId) -> &[u32] {
        let v = root as usize;
        match (self.starts.get(v), self.starts.get(v + 1)) {
            (Some(&a), Some(&b)) => &self.tris[a as usize..b as usize],
            _ => &[],
        }
    }
}

/// A triangle's vertices, each once.
pub(crate) fn distinct([a, b, c]: [VertexId; 3]) -> impl Iterator<Item = VertexId> {
    [
        Some(a),
        (b != a).then_some(b),
        (c != a && c != b).then_some(c),
    ]
    .into_iter()
    .flatten()
}

/// A curve point where it lands on the mesh.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CurvePoint {
    /// The point in the query mesh's coordinates, and its distance to it.
    pub(crate) query: [f32; 3],
    pub(crate) distance: f32,
    /// The point in the mesh's local coordinates.
    pub(crate) local: [f32; 3],
    /// Where on the mesh it lands.
    pub(crate) triangle: TriangleId,
    pub(crate) barycentric: [f32; 3],
}

/// The component a curve lies on.
pub(crate) struct CurveSurface {
    pub(crate) weld: Weld,
    /// Every proper triangle of the mesh, whatever its component.
    pub(crate) fans: Fans,
    label: Vec<u32>,
    component: u32,
}

impl CurveSurface {
    /// Whether triangle `t` is part of the component.
    pub(crate) fn contains(&self, t: TriangleId) -> bool {
        self.label.get(t as usize) == Some(&self.component)
    }

    /// Finds the component every point lies on, and moves the points that
    /// landed on another one touching it there. When there are several, it is
    /// the one the first point landed on, or else the one with the smallest
    /// triangle id among those at the first point. `query` is the mesh the
    /// points were projected on (`mesh`, or a copy of it in world coordinates
    /// with the same triangles).
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] when no component has all the points.
    pub(crate) fn locate(
        mesh: &Mesh,
        query: (&Mesh, &Bvh),
        points: &mut [CurvePoint],
    ) -> Result<CurveSurface> {
        let slots = mesh.triangle_slots();
        let weld = Weld::of(mesh)?;
        let fans = Fans::build(mesh.vertex_slots(), slots, &weld, |t| {
            proper_triangle(mesh, t as TriangleId)
        })?;
        let mut surface = CurveSurface {
            weld,
            fans,
            label: filled(slots, NONE)?,
            component: 0,
        };
        if let Some(first) = points.first().copied()
            && let Some(on) = surface.moved_onto_one_at(mesh, query, &first, points)?
        {
            points.copy_from_slice(&on);
            return Ok(surface);
        }
        Err(SceneError::InvalidInput(
            "trim curve crosses several components of the mesh",
        ))
    }

    /// The points on the first component at point `at` that has them all,
    /// which becomes this one.
    fn moved_onto_one_at(
        &mut self,
        mesh: &Mesh,
        query: (&Mesh, &Bvh),
        at: &CurvePoint,
        points: &[CurvePoint],
    ) -> Result<Option<Vec<CurvePoint>>> {
        let mut number = 0;
        let mut seeds = vec![at.triangle];
        for first in [true, false] {
            if !first {
                // Not the component it landed on: the others touching it.
                seeds.clear();
                triangles_in(query.0, query.1, &near(query.1, at), &mut seeds)?;
                seeds.sort_unstable();
            }
            for &seed in &seeds {
                if self.label[seed as usize] != NONE || proper_triangle(mesh, seed).is_none() {
                    continue;
                }
                self.flood(mesh, seed, number)?;
                self.component = number;
                number += 1;
                if let Some(on) = self.moved_onto(query, points)? {
                    return Ok(Some(on));
                }
            }
        }
        Ok(None)
    }

    /// The points on the component, or `None` when one does not lie on it.
    fn moved_onto(
        &self,
        (mesh, bvh): (&Mesh, &Bvh),
        points: &[CurvePoint],
    ) -> Result<Option<Vec<CurvePoint>>> {
        let mut around = Vec::new();
        let mut out = Vec::with_capacity(points.len());
        for p in points {
            if self.contains(p.triangle) {
                out.push(*p);
                continue;
            }
            around.clear();
            triangles_in(mesh, bvh, &near(bvh, p), &mut around)?;
            let q = p.query.map(f64::from);
            let best = around
                .iter()
                .filter(|&&t| self.contains(t))
                .map(|&t| {
                    let [a, b, c] = mesh.triangles()[t as usize]
                        .map(|v| mesh.positions()[v as usize].map(f64::from));
                    let (d2, barycentric) = closest_barycentric_on_triangle(a, b, c, q);
                    (d2, t, barycentric)
                })
                .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let Some((_, triangle, barycentric)) = best else {
                return Ok(None);
            };
            out.push(CurvePoint {
                triangle,
                barycentric: barycentric.map(|w| w as f32),
                ..*p
            });
        }
        Ok(Some(out))
    }

    /// Labels every triangle connected to `seed`, itself included, `number`.
    fn flood(&mut self, mesh: &Mesh, seed: TriangleId, number: u32) -> Result<()> {
        let (weld, label) = (&self.weld, &mut self.label);
        let mut stack = vec![seed];
        label[seed as usize] = number;
        while let Some(t) = stack.pop() {
            let corners = mesh.triangles()[t as usize].map(|v| weld.root(v));
            for k in 0..3 {
                let (a, b) = (corners[k], corners[(k + 1) % 3]);
                if a == b {
                    continue;
                }
                for &nb in self.fans.around(a) {
                    let uses_b = || {
                        let corners = mesh.triangles()[nb as usize];
                        corners.iter().any(|&v| weld.root(v) == b)
                    };
                    if label[nb as usize] == NONE && uses_b() {
                        label[nb as usize] = number;
                        stack.try_reserve(1).map_err(|_| oom())?;
                        stack.push(nb);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Where the triangles a point lies on are: its distance around it, and
/// [`ON_SURFACE`].
fn near(bvh: &Bvh, p: &CurvePoint) -> Region {
    let bounds = bvh.root().map(|root| bvh.node(root).bounds);
    let corners = bounds.map_or([[0.0; 3]; 2], |b| [b.min, b.max]);
    let coordinates = corners.iter().flatten().chain(&p.query);
    let largest = coordinates.fold(0.0f32, |m, x| m.max(x.abs()));
    Region::Sphere {
        center: p.query,
        radius: (f64::from(p.distance) + ON_SURFACE * f64::from(largest)) as f32,
    }
}
