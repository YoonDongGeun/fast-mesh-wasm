//! [`extrude_base`]: boundary loops run along a direction to a plane and
//! capped there.

use fast_mesh_store::{Mesh, Transaction, TriangleId, VertexId};

use super::check::{Found, Planned, intersections};
use super::{to_f32, unit};
use crate::fill_dp::{triangulate_flat_loop, triangulate_with_islands};
use crate::refine::refine;
use crate::topology::{
    BoundaryLoop, EdgeMap, LoopId, VertexFans, components, cross, dot, loops_with, point,
    proper_triangle, sub,
};
use crate::{OpsError, Result, filled, push, reserve};

const NONE: u32 = u32::MAX;

/// The most new vertices the refinement of one cap adds. A base of a few
/// thousand rim edges reaches the density of its walls within it; a longer
/// one is refined part of the way, as for the caps of
/// [`cut_by_plane`](crate::cut_by_plane).
const CAP_VERTEX_BUDGET: usize = 20_000;

/// The smallest cosine between the direction and the plane normal: walls
/// nearly parallel to the plane would run off to great lengths.
const MIN_COS: f64 = 1e-3;

/// Where [`extrude_base`] puts the base plane, `dot(p, normal) = level` in
/// the mesh's local coordinates (origin excluded), with
/// [`ExtrudeBaseOptions::normal`] as given (not scaled to unit length).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BaseLevel {
    /// This `level`.
    At(f64),
    /// `offset` past the vertex the mesh's triangles use that lies furthest
    /// along the normal: `level = max dot(p, normal) + offset`. With a unit
    /// normal, `offset` is a distance.
    Beyond(f64),
}

/// Which boundary loops [`extrude_base`] runs to the base.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum BaseLoops {
    /// The loop with the longest perimeter (the first of equals): the outer
    /// rim of an open surface, leaving smaller holes as they are. Default.
    #[default]
    Largest,
    /// Every boundary loop. Outlines nested on the plane (a surface with
    /// holes) become holes in their cap.
    All,
    /// These loops, by [`BoundaryLoop::id`] (see
    /// [`boundary_loops`](crate::boundary_loops)).
    Ids(Vec<LoopId>),
}

/// Options for [`extrude_base`].
#[derive(Debug, Clone, PartialEq)]
pub struct ExtrudeBaseOptions {
    /// The direction the walls run in, from the boundary toward the base
    /// (any length), in local coordinates.
    pub direction: [f64; 3],
    /// The base plane's normal (any length), pointing the way the walls go:
    /// `dot(direction, normal) > 0`. Usually `direction` itself.
    pub normal: [f64; 3],
    /// Where the plane is along `normal`. Every boundary vertex extruded
    /// must lie strictly before it.
    pub level: BaseLevel,
    /// The loops to extrude. Default [`BaseLoops::Largest`].
    pub loops: BaseLoops,
    /// Refine each cap to the density of its walls (up to 20,000 new
    /// vertices per cap); without it the cap has only the rim's vertices,
    /// with its diagonals relaxed toward Delaunay. Default `true`.
    pub refine: bool,
    /// Check the result for self-intersections before writing it (see the
    /// [module docs](super)). Default `true`.
    pub check_self_intersections: bool,
}

impl ExtrudeBaseOptions {
    /// Walls along `direction` to the plane perpendicular to it at `level`,
    /// with the other options at their defaults.
    pub fn along(direction: [f64; 3], level: BaseLevel) -> Self {
        Self {
            direction,
            normal: unit(direction).unwrap_or(direction),
            level,
            loops: BaseLoops::Largest,
            refine: true,
            check_self_intersections: true,
        }
    }
}

/// What [`extrude_base`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtrudeBaseReport {
    /// The loops extruded, in [`boundary_loops`](crate::boundary_loops)
    /// order, as they were before.
    pub loops: Vec<LoopId>,
    /// The base plane's level: `dot(p, normal)` for its points.
    pub level: f64,
    /// New vertices on the plane, one per vertex of each loop.
    pub base_vertices: usize,
    /// New vertices inside the caps, from refinement.
    pub cap_vertices: usize,
    /// New wall triangles, two per loop edge.
    pub wall_triangles: usize,
    /// New cap triangles.
    pub cap_triangles: usize,
    /// Whether a cap's refinement stopped at its vertex budget.
    pub capped: bool,
    /// Whether the surfaces were turned around: the solid came out wound
    /// inward (the surface faced the way the walls go), so every triangle
    /// of the extruded surfaces and the new ones was reversed.
    pub flipped: bool,
    /// The triangles of every pair that touches, crosses or folds flat after
    /// the extrusion with a wall or cap among them (see the
    /// [module docs](super)), ascending: a plane cutting through the surface
    /// gives some. `None` when
    /// [`ExtrudeBaseOptions::check_self_intersections`] is off.
    pub self_intersections: Option<Vec<TriangleId>>,
}

/// Extrudes boundary loops to a base: runs every vertex of each loop along
/// [`direction`](ExtrudeBaseOptions::direction) to the plane of
/// [`normal`](ExtrudeBaseOptions::normal) and
/// [`level`](ExtrudeBaseOptions::level), joins the loop to its copy on the
/// plane with a wall of two triangles per edge, and closes the copy with a
/// flat cap. An open surface whose only boundary is the loop becomes a
/// closed solid. A solid that came out wound inward is reversed, surfaces
/// included, so it faces outward ([`ExtrudeBaseReport::flipped`]).
///
/// The base vertices take their loop vertex's channel values (UVs
/// included); cap vertices from refinement interpolate them.
///
/// The result is checked for self-intersections before it is written and
/// they are reported, not refused: a plane through the surface, or a rim
/// that overhangs its own surface along the direction, gives a closed mesh
/// that crosses itself.
///
/// # Errors
/// [`OpsError::InvalidInput`] when the mesh has no boundary loop; a loop to
/// extrude is open, not clean or shorter than three vertices; the direction
/// or normal is zero or not finite, or they are more than about 89.9° apart;
/// the level or offset is not finite; a loop vertex does not lie strictly
/// before the plane; a base position does not fit `f32`; an outline on the
/// plane has no area or crosses itself; or nested outlines run the same way.
/// [`OpsError::StaleLoops`] for a loop id that matches no loop.
/// [`OpsError::Store`] when out of memory. Nothing is written on error.
pub fn extrude_base(
    tx: &mut Transaction,
    options: &ExtrudeBaseOptions,
) -> Result<ExtrudeBaseReport> {
    let plan = plan(tx.mesh(), options)?;
    plan.write(tx)
}

/// One loop on the scratch meshes: its top vertices are `start..start + n`
/// and its base vertices the `n` after them.
struct ScratchLoop {
    start: u32,
    n: u32,
}

impl ScratchLoop {
    fn base(&self, i: u32) -> u32 {
        self.start + self.n + i
    }
}

struct Plan {
    loops: Vec<LoopId>,
    level: f64,
    vertex_slots: u32,
    /// New vertices: the base vertices (one parent each, weight 1), then
    /// the cap vertices (three weighted parents over planned ids).
    added: Vec<[f32; 3]>,
    base_parents: Vec<VertexId>,
    cap_parents: Vec<[(VertexId, f32); 3]>,
    /// Walls, then caps, over planned ids, wound as before any flip.
    triangles: Vec<[VertexId; 3]>,
    walls: usize,
    capped: bool,
    /// Existing triangles to reverse, when the solid is flipped.
    reverse: Vec<TriangleId>,
    flipped: bool,
    found: Option<Found>,
}

/// Whether `p` is inside the closed polygon `poly` (even-odd rule).
fn inside(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    let mut odd = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
            if p[0] < x {
                odd = !odd;
            }
        }
        j = i;
    }
    odd
}

fn shoelace(poly: &[[f64; 2]]) -> f64 {
    let mut s = 0.0;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        s += a[0] * b[1] - a[1] * b[0];
    }
    0.5 * s
}

/// The loops `which` picks, in `loops` order.
fn chosen<'a>(loops: &'a [BoundaryLoop], which: &BaseLoops) -> Result<Vec<&'a BoundaryLoop>> {
    let mut out: Vec<&BoundaryLoop> = Vec::new();
    match which {
        BaseLoops::Largest => {
            let mut best: Option<&BoundaryLoop> = None;
            for l in loops {
                if best.is_none_or(|b| l.perimeter > b.perimeter) {
                    best = Some(l);
                }
            }
            push(&mut out, best.expect("at least one loop"))?;
        }
        BaseLoops::All => {
            reserve(&mut out, loops.len())?;
            out.extend(loops);
        }
        BaseLoops::Ids(ids) => {
            if ids.is_empty() {
                return Err(OpsError::InvalidInput(
                    "extrude base needs at least one loop",
                ));
            }
            for id in ids {
                if !loops.iter().any(|l| l.id == *id) {
                    return Err(OpsError::StaleLoops);
                }
            }
            for l in loops {
                if ids.contains(&l.id) {
                    push(&mut out, l)?;
                }
            }
        }
    }
    for l in &out {
        if !l.closed || !l.clean || l.vertices.len() < 3 {
            return Err(OpsError::InvalidInput(
                "a boundary loop to extrude is open or not clean; orient or repair the mesh first",
            ));
        }
    }
    Ok(out)
}

fn plan(mesh: &Mesh, o: &ExtrudeBaseOptions) -> Result<Plan> {
    let bad_way =
        OpsError::InvalidInput("extrude base direction and normal must be finite and non-zero");
    let d_unit = unit(o.direction).ok_or(bad_way.clone())?;
    let n_unit = unit(o.normal).ok_or(bad_way)?;
    if dot(d_unit, n_unit) < MIN_COS {
        return Err(OpsError::InvalidInput(
            "extrude base direction must run toward the base plane, along its normal",
        ));
    }
    let along = dot(o.direction, o.normal);
    let edges = EdgeMap::build(mesh)?;
    let loops = loops_with(mesh, &edges)?;
    if loops.is_empty() {
        return Err(OpsError::InvalidInput(
            "the mesh has no boundary loop to extrude",
        ));
    }
    let picked = chosen(&loops, &o.loops)?;
    let level = match o.level {
        BaseLevel::At(c) => c,
        BaseLevel::Beyond(offset) => {
            let mut far = f64::NEG_INFINITY;
            for t in 0..mesh.triangle_slots() as TriangleId {
                if let Some(tri) = proper_triangle(mesh, t) {
                    for v in tri {
                        far = far.max(dot(point(mesh, v), o.normal));
                    }
                }
            }
            far + offset
        }
    };
    if !level.is_finite() {
        return Err(OpsError::InvalidInput("extrude base level must be finite"));
    }

    // The scratch meshes hold each loop and its copy on the plane, at their
    // local positions: bare for the triangulation (no walls to check against,
    // since the walls stand on the outline), walled for the refinement (the
    // walls give the density).
    let total: usize = picked.iter().map(|l| l.vertices.len()).sum();
    let slots = mesh.vertex_slots();
    if slots + 2 * total >= NONE as usize {
        return Err(OpsError::InvalidInput("too many vertices for an extrusion"));
    }
    let mut flat: Vec<f32> = Vec::new();
    reserve(&mut flat, 6 * total)?;
    let mut added: Vec<[f32; 3]> = Vec::new();
    reserve(&mut added, total)?;
    let mut base_parents: Vec<VertexId> = Vec::new();
    reserve(&mut base_parents, total)?;
    let mut scratch_loops: Vec<ScratchLoop> = Vec::new();
    reserve(&mut scratch_loops, picked.len())?;
    let mut wall_index: Vec<u32> = Vec::new();
    reserve(&mut wall_index, 6 * total)?;
    let mut triangles: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut triangles, 2 * total)?;
    for l in &picked {
        let n = l.vertices.len() as u32;
        let sl = ScratchLoop {
            start: (flat.len() / 3) as u32,
            n,
        };
        let first_base = added.len();
        for &v in &l.vertices {
            flat.extend(mesh.positions()[v as usize]);
        }
        for &v in &l.vertices {
            let p = point(mesh, v);
            let h = level - dot(p, o.normal);
            if h.is_nan() || h <= 0.0 {
                return Err(OpsError::InvalidInput(
                    "the base plane must lie beyond every boundary vertex, along the direction",
                ));
            }
            let t = h / along;
            let q = to_f32(std::array::from_fn(|k| p[k] + t * o.direction[k]))
                .ok_or(OpsError::InvalidInput("a base position does not fit f32"))?;
            flat.extend(q);
            added.push(q);
            base_parents.push(v);
        }
        // The loop runs along each edge the way its triangle uses it, so the
        // wall uses it backwards: the quad v(i+1) -> v(i) -> b(i) -> b(i+1),
        // split along its shorter diagonal.
        let length = |a: [f32; 3], b: [f32; 3]| {
            let d = sub(a.map(f64::from), b.map(f64::from));
            dot(d, d)
        };
        for i in 0..n {
            let j = (i + 1) % n;
            let (top_i, top_j) = (sl.start + i, sl.start + j);
            let (base_i, base_j) = (sl.base(i), sl.base(j));
            let (vi, vj) = (l.vertices[i as usize], l.vertices[j as usize]);
            let (bi, bj) = (
                (slots + first_base + i as usize) as VertexId,
                (slots + first_base + j as usize) as VertexId,
            );
            let p = |s: u32| -> [f32; 3] {
                let s = 3 * s as usize;
                [flat[s], flat[s + 1], flat[s + 2]]
            };
            if length(p(top_j), p(base_i)) <= length(p(top_i), p(base_j)) {
                wall_index.extend([top_j, top_i, base_i, top_j, base_i, base_j]);
                triangles.extend([[vj, vi, bi], [vj, bi, bj]]);
            } else {
                wall_index.extend([top_j, top_i, base_j, top_i, base_i, base_j]);
                triangles.extend([[vj, vi, bj], [vi, bi, bj]]);
            }
        }
        scratch_loops.push(sl);
    }
    let walls = triangles.len();
    let bare = Mesh::from_buffers(&flat, &[])?;
    let bare_edges = EdgeMap::build(&bare)?;
    let walled = Mesh::from_buffers(&flat, &wall_index)?;
    let walled_edges = EdgeMap::build(&walled)?;
    let fans = VertexFans::build(&walled)?;

    // The outlines on the plane, how they nest (even-odd), and which are
    // holes of which.
    let helper = if n_unit[0].abs() < 0.6 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = unit(cross(helper, n_unit)).expect("helper is not along the normal");
    let w = cross(n_unit, u);
    let mut outlines: Vec<Vec<[f64; 2]>> = Vec::new();
    reserve(&mut outlines, scratch_loops.len())?;
    let mut areas: Vec<f64> = Vec::new();
    reserve(&mut areas, scratch_loops.len())?;
    for sl in &scratch_loops {
        let mut ring = Vec::new();
        reserve(&mut ring, sl.n as usize)?;
        ring.extend((0..sl.n).map(|i| {
            let q = point(&bare, sl.base(i));
            [dot(q, u), dot(q, w)]
        }));
        let area = shoelace(&ring);
        if !(area.is_finite() && area != 0.0) {
            return Err(OpsError::InvalidInput(
                "a base outline has no area on the plane; choose another direction",
            ));
        }
        areas.push(area);
        outlines.push(ring);
    }
    let count = scratch_loops.len();
    let mut parent = filled(count, NONE)?;
    let mut depth = filled(count, 0usize)?;
    for i in 0..count {
        for j in 0..count {
            if i != j && inside(outlines[i][0], &outlines[j]) {
                depth[i] += 1;
                if parent[i] == NONE || areas[j].abs() < areas[parent[i] as usize].abs() {
                    parent[i] = j as u32;
                }
            }
        }
    }

    let budget = if o.refine { CAP_VERTEX_BUDGET } else { 0 };
    let mut capped = false;
    let mut cap_parents: Vec<[(VertexId, f32); 3]> = Vec::new();
    // A scratch base vertex's planned id.
    let mut planned_base = filled(flat.len() / 3, NONE)?;
    let mut next_base = slots as u32;
    for sl in &scratch_loops {
        for i in 0..sl.n {
            planned_base[sl.base(i) as usize] = next_base;
            next_base += 1;
        }
    }
    let mut holes: Vec<usize> = Vec::new();
    let mut base_ids: Vec<VertexId> = Vec::new();
    let mut island_ids: Vec<Vec<VertexId>> = Vec::new();
    let mut ring_vertices: Vec<VertexId> = Vec::new();
    for outer in 0..count {
        if depth[outer] % 2 == 1 {
            continue;
        }
        holes.clear();
        for i in 0..count {
            if depth[i] % 2 == 1 && parent[i] == outer as u32 {
                if areas[i] * areas[outer] >= 0.0 {
                    return Err(OpsError::InvalidInput(
                        "nested base outlines must run opposite ways, as a surface with holes does",
                    ));
                }
                push(&mut holes, i)?;
            }
        }
        let sl = &scratch_loops[outer];
        base_ids.clear();
        reserve(&mut base_ids, sl.n as usize)?;
        base_ids.extend((0..sl.n).map(|i| sl.base(i)));
        island_ids.clear();
        reserve(&mut island_ids, holes.len())?;
        for &h in &holes {
            let hl = &scratch_loops[h];
            let mut ids = Vec::new();
            reserve(&mut ids, hl.n as usize)?;
            ids.extend((0..hl.n).map(|i| hl.base(i)));
            island_ids.push(ids);
        }
        let tris = if holes.is_empty() {
            triangulate_flat_loop(&bare, &base_ids, &bare_edges, n_unit)?
        } else {
            let mut islands: Vec<&[VertexId]> = Vec::new();
            reserve(&mut islands, island_ids.len())?;
            islands.extend(island_ids.iter().map(Vec::as_slice));
            triangulate_with_islands(&bare, &base_ids, &islands, &bare_edges)?
        }
        .map_err(|_| {
            OpsError::InvalidInput(
                "a base outline crosses itself on the plane; choose another direction",
            )
        })?;
        ring_vertices.clear();
        reserve(&mut ring_vertices, base_ids.len())?;
        ring_vertices.extend_from_slice(&base_ids);
        for ids in &island_ids {
            reserve(&mut ring_vertices, ids.len())?;
            ring_vertices.extend_from_slice(ids);
        }
        let patch = refine(
            &walled,
            &fans,
            &walled_edges,
            &ring_vertices,
            &tris,
            Some(budget),
        )?;
        capped |= o.refine && patch.capped;
        // Patch-local vertex `x`: an old (base) vertex, or a new cap vertex.
        let first_new = slots + added.len();
        let old = patch.old.len();
        let planned = |x: u32| -> VertexId {
            if (x as usize) < old {
                planned_base[patch.old[x as usize] as usize]
            } else {
                (first_new + x as usize - old) as VertexId
            }
        };
        if first_new + patch.new_vertices() >= NONE as usize {
            return Err(OpsError::InvalidInput("too many vertices for an extrusion"));
        }
        reserve(&mut added, patch.new_vertices())?;
        reserve(&mut cap_parents, patch.new_vertices())?;
        for (q, parents) in patch.points[old..].iter().zip(&patch.parents) {
            added.push(
                to_f32(*q).ok_or(OpsError::InvalidInput("a base position does not fit f32"))?,
            );
            cap_parents.push(parents.map(|(x, wt)| (planned(x), wt)));
        }
        reserve(&mut triangles, patch.triangles.len())?;
        triangles.extend(patch.triangles.iter().map(|t| t.map(planned)));
    }

    // Which way the solid faces: the signed volume of the surfaces the loops
    // bound plus the walls and caps, about the base's centre.
    let components = components(mesh, &edges)?;
    let mut component_of = filled(mesh.triangle_slots(), NONE)?;
    for (c, comp) in components.iter().enumerate() {
        for &t in &comp.triangles {
            component_of[t as usize] = c as u32;
        }
    }
    let mut extruded = filled(components.len(), false)?;
    for l in &picked {
        let t = edges.triangles(l.vertices[0], l.vertices[1])[0];
        extruded[component_of[t as usize] as usize] = true;
    }
    let base_count = base_parents.len();
    let mut centre = [0.0f64; 3];
    for q in &added[..base_count] {
        for k in 0..3 {
            centre[k] += f64::from(q[k]) / base_count as f64;
        }
    }
    let at = |v: VertexId| -> [f64; 3] {
        let q = if v as usize >= slots {
            added[v as usize - slots].map(f64::from)
        } else {
            point(mesh, v)
        };
        sub(q, centre)
    };
    let volume6 = |[a, b, c]: [VertexId; 3]| dot(at(a), cross(at(b), at(c)));
    let mut volume = 0.0;
    for (c, comp) in components.iter().enumerate() {
        if extruded[c] {
            volume += comp
                .triangles
                .iter()
                .map(|&t| volume6(mesh.triangles()[t as usize]))
                .sum::<f64>();
        }
    }
    volume += triangles.iter().map(|&t| volume6(t)).sum::<f64>();
    let flipped = volume < 0.0;
    let mut reverse: Vec<TriangleId> = Vec::new();
    if flipped {
        for (c, comp) in components.iter().enumerate() {
            if extruded[c] {
                reserve(&mut reverse, comp.triangles.len())?;
                reverse.extend_from_slice(&comp.triangles);
            }
        }
    }

    let found = if o.check_self_intersections {
        Some(intersections(&Planned {
            mesh,
            moved: &[],
            added: &added,
            triangles: &triangles,
            replaced: &[],
        })?)
    } else {
        None
    };
    let mut ids: Vec<LoopId> = Vec::new();
    reserve(&mut ids, picked.len())?;
    ids.extend(picked.iter().map(|l| l.id));
    Ok(Plan {
        loops: ids,
        level,
        vertex_slots: slots as u32,
        added,
        base_parents,
        cap_parents,
        triangles,
        walls,
        capped,
        reverse,
        flipped,
        found,
    })
}

impl Plan {
    /// Writes the plan; everything the report needs is reserved first.
    fn write(self, tx: &mut Transaction) -> Result<ExtrudeBaseReport> {
        let mut new_ids: Vec<VertexId> = Vec::new();
        reserve(&mut new_ids, self.added.len())?;
        let mut tris: Vec<[VertexId; 3]> = Vec::new();
        reserve(&mut tris, self.triangles.len())?;
        let mut reversed: Vec<[VertexId; 3]> = Vec::new();
        reserve(&mut reversed, self.reverse.len())?;
        let mut crossing: Vec<TriangleId> = Vec::new();
        if let Some(found) = &self.found {
            reserve(&mut crossing, found.len())?;
        }

        let slots = self.vertex_slots;
        let base = self.base_parents.len();
        for (&v, &q) in self.base_parents.iter().zip(&self.added) {
            new_ids.push(tx.add_vertex_with_parents(q, &[(v, 1.0)])?);
        }
        for (parents, &q) in self.cap_parents.iter().zip(&self.added[base..]) {
            // Every parent is a base vertex or an earlier cap vertex.
            let parents = parents.map(|(v, w)| {
                let v = if v >= slots {
                    new_ids[(v - slots) as usize]
                } else {
                    v
                };
                (v, w)
            });
            new_ids.push(tx.add_vertex_with_parents(q, &parents)?);
        }
        let real = |v: VertexId| {
            if v >= slots {
                new_ids[(v - slots) as usize]
            } else {
                v
            }
        };
        let flip = |[a, b, c]: [VertexId; 3]| if self.flipped { [a, c, b] } else { [a, b, c] };
        tris.extend(self.triangles.iter().map(|t| flip(t.map(real))));
        let added = tx.add_triangles(&tris)?;
        if self.flipped {
            let old = tx.mesh().triangles();
            reversed.extend(self.reverse.iter().map(|&t| {
                let [a, b, c] = old[t as usize];
                [a, c, b]
            }));
            tx.set_triangles(&self.reverse, &reversed)?;
        }
        let self_intersections = self.found.as_ref().map(|found| {
            found.write_final(&added, &mut crossing);
            crossing
        });
        Ok(ExtrudeBaseReport {
            loops: self.loops,
            level: self.level,
            base_vertices: base,
            cap_vertices: self.added.len() - base,
            wall_triangles: self.walls,
            cap_triangles: self.triangles.len() - self.walls,
            capped: self.capped,
            flipped: self.flipped,
            self_intersections,
        })
    }
}
