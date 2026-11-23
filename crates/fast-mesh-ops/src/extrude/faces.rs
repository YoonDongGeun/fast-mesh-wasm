//! [`extrude_faces`]: a face selection moved out, with walls along its
//! boundary.

use fast_mesh_store::{Mesh, Transaction, TriangleId, VertexId};

use super::check::{Found, Planned, intersections};
use super::{IDENTITY, apply, cofactor, inverse, to_f32, unit};
use crate::topology::{UnionFind, cross, dot, point, proper_triangle, sub};
use crate::{OpsError, Result, filled, push, reserve};

const NONE: u32 = u32::MAX;

/// Which way [`extrude_faces`] moves the selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtrudeDirection {
    /// Each region of the selection (its triangles connected through shared
    /// edges) moves as one rigid piece along its area-weighted average
    /// normal: the caps keep their shape and a flat region gains exactly its
    /// area times the distance in volume. A region whose normals cancel out
    /// (a whole closed surface, say) has no direction and is an error.
    RegionNormal,
    /// Each vertex moves along the angle-weighted average normal of its
    /// selected triangles, so the caps follow the surface like an offset
    /// (and bulge or shrink with its curvature).
    VertexNormals,
    /// Every vertex moves along this vector, scaled to unit length in the
    /// frame of [`ExtrudeFacesOptions::to_world`].
    Fixed([f64; 3]),
}

/// Options for [`extrude_faces`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtrudeFacesOptions {
    /// How far the selection moves; negative moves it against its normals
    /// (or against a [`ExtrudeDirection::Fixed`] vector), pushing it in.
    /// Must be finite and non-zero. Default `1`.
    pub distance: f64,
    /// Which way it moves. Default [`ExtrudeDirection::RegionNormal`].
    pub direction: ExtrudeDirection,
    /// The linear map (row-major) from the mesh's local coordinates to the
    /// frame that normals, directions and the distance are measured in,
    /// such as the linear part of the mesh's placement in a scene. It must
    /// be invertible. Default the identity: everything in local units.
    pub to_world: [[f64; 3]; 3],
    /// Check the result for self-intersections before writing it (see the
    /// [module docs](super)). Default `true`.
    pub check_self_intersections: bool,
}

impl Default for ExtrudeFacesOptions {
    fn default() -> Self {
        Self {
            distance: 1.0,
            direction: ExtrudeDirection::RegionNormal,
            to_world: IDENTITY,
            check_self_intersections: true,
        }
    }
}

/// What [`extrude_faces`] did.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExtrudeFacesReport {
    /// Regions of the selection: its triangles connected through shared
    /// edges. Each is extruded as one piece with walls of its own.
    pub regions: usize,
    /// Selected triangles, deduplicated. They keep their ids and form the
    /// moved caps.
    pub cap_triangles: usize,
    /// Vertices moved in place: those whose triangles are all selected and
    /// that lie on no wall.
    pub moved_vertices: usize,
    /// New vertices: the caps' copies of the vertices on the selection's
    /// boundary. The originals stay where they were, as the walls' feet.
    pub added_vertices: usize,
    /// The new wall triangles, two per boundary edge of the selection
    /// (including edges on the mesh's own open boundary), ascending.
    pub wall_triangles: Vec<TriangleId>,
    /// The triangles of every pair that touches, crosses or folds flat after
    /// the extrusion with a cap or wall among them (see the
    /// [module docs](super)), ascending. `None` when
    /// [`ExtrudeFacesOptions::check_self_intersections`] is off.
    pub self_intersections: Option<Vec<TriangleId>>,
}

/// Extrudes the selected `triangles`: moves them by
/// [`distance`](ExtrudeFacesOptions::distance) along
/// [`direction`](ExtrudeFacesOptions::direction) and builds a wall of two
/// triangles on every boundary edge of the selection, between the edge
/// where it was and where it went.
///
/// The selected triangles keep their ids (and their winding) and become
/// the caps; triangles that are not selected keep their vertices, which
/// stay put. A vertex whose triangles are all selected simply moves. A
/// vertex on the selection's boundary stays as a wall's foot, and the caps
/// get a new copy of it for each fan of selected triangles around it, so
/// regions that touch at one vertex separate there and stay manifold. The
/// walls are wound like the caps, so a closed, consistently wound mesh
/// stays closed and consistently wound. A selection with holes gets walls
/// around them; selected edges on the mesh's open boundary get walls too,
/// whose feet become the new open boundary.
///
/// Wall quads are split along their shorter diagonal (measured through
/// [`ExtrudeFacesOptions::to_world`]). New vertices take their parent's
/// channel values (UVs included).
///
/// The result is checked for self-intersections before it is written and
/// they are reported, not refused: pushing a region through the opposite
/// side of a thin part, say, gives a closed mesh that crosses itself.
///
/// # Errors
/// [`OpsError::InvalidInput`] for an empty selection, a dead or degenerate
/// selected triangle, a distance that is zero or not finite, a direction
/// that is zero or not finite, a `to_world` that is not invertible, a
/// selected edge shared by three or more triangles, two selected triangles
/// wound against each other across an edge, a region (or a vertex, for
/// [`ExtrudeDirection::VertexNormals`]) whose normals cancel out, or a
/// moved position that does not fit `f32`; [`OpsError::Store`] when out of
/// memory. Nothing is written on error.
pub fn extrude_faces(
    tx: &mut Transaction,
    triangles: &[TriangleId],
    options: &ExtrudeFacesOptions,
) -> Result<ExtrudeFacesReport> {
    let plan = plan(tx.mesh(), triangles, options)?;
    plan.write(tx)
}

/// Everything [`extrude_faces`] writes, over planned vertex ids (see
/// [`Planned`]).
struct Plan {
    sel: Vec<TriangleId>,
    regions: usize,
    vertex_slots: u32,
    /// Vertices moved in place, by id.
    moved: Vec<(VertexId, [f32; 3])>,
    /// The new vertices' positions and parents.
    added: Vec<[f32; 3]>,
    parents: Vec<VertexId>,
    /// Every cap, in selection order, over planned ids.
    caps: Vec<[VertexId; 3]>,
    /// The walls, over planned ids.
    walls: Vec<[VertexId; 3]>,
    found: Option<Found>,
}

/// The triangles of the sorted fans `a` and `b` that use both their
/// vertices: at most three are kept, and the count is exact.
fn shared(a: &[TriangleId], b: &[TriangleId]) -> ([TriangleId; 3], usize) {
    let (mut i, mut j) = (0, 0);
    let mut out = [NONE; 3];
    let mut n = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                if n < 3 {
                    out[n] = a[i];
                }
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    (out, n)
}

fn plan(mesh: &Mesh, triangles: &[TriangleId], o: &ExtrudeFacesOptions) -> Result<Plan> {
    if !o.distance.is_finite() || o.distance == 0.0 {
        return Err(OpsError::InvalidInput(
            "extrude distance must be finite and non-zero",
        ));
    }
    let to_local = inverse(&o.to_world).ok_or(OpsError::InvalidInput(
        "extrude to_world must be a finite, invertible linear map",
    ))?;
    let fixed = match o.direction {
        ExtrudeDirection::Fixed(d) => Some(unit(d).ok_or(OpsError::InvalidInput(
            "extrude direction must be finite and non-zero",
        ))?),
        _ => None,
    };
    let cof = cofactor(&o.to_world);

    let mut sel: Vec<TriangleId> = Vec::new();
    reserve(&mut sel, triangles.len())?;
    sel.extend_from_slice(triangles);
    sel.sort_unstable();
    sel.dedup();
    if sel.is_empty() {
        return Err(OpsError::InvalidInput(
            "extrude needs at least one selected triangle",
        ));
    }
    for &t in &sel {
        match mesh.triangle(t) {
            None => {
                return Err(OpsError::InvalidInput(
                    "a selected triangle is dead or out of range",
                ));
            }
            Some([a, b, c]) if a == b || b == c || a == c => {
                return Err(OpsError::InvalidInput(
                    "a selected triangle repeats a vertex",
                ));
            }
            Some(_) => {}
        }
    }
    let n = sel.len();
    if 3 * n >= NONE as usize {
        return Err(OpsError::InvalidInput("too many selected triangles"));
    }
    let tris = mesh.triangles();
    let mut in_sel = filled(mesh.triangle_slots(), false)?;
    for &t in &sel {
        in_sel[t as usize] = true;
    }

    // The selected vertices, with a local index each.
    let mut verts: Vec<VertexId> = Vec::new();
    reserve(&mut verts, 3 * n)?;
    verts.extend(sel.iter().flat_map(|&t| tris[t as usize]));
    verts.sort_unstable();
    verts.dedup();
    let nv = verts.len();
    let mut local = filled(mesh.vertex_slots(), NONE)?;
    for (i, &v) in verts.iter().enumerate() {
        local[v as usize] = i as u32;
    }

    // Every proper triangle at a selected vertex, per vertex, ascending: one
    // scan of the mesh, the only part that grows with its size.
    let mut starts = filled(nv + 1, 0u32)?;
    let mut corners = 0usize;
    for t in 0..mesh.triangle_slots() as TriangleId {
        if let Some(tri) = proper_triangle(mesh, t) {
            for v in tri {
                let l = local[v as usize];
                if l != NONE {
                    starts[l as usize + 1] += 1;
                    corners += 1;
                }
            }
        }
    }
    if corners >= NONE as usize {
        return Err(OpsError::InvalidInput(
            "too many triangles at the selection",
        ));
    }
    for l in 0..nv {
        starts[l + 1] += starts[l];
    }
    let mut next = filled(nv, 0u32)?;
    next.copy_from_slice(&starts[..nv]);
    let mut fans = filled(corners, 0 as TriangleId)?;
    for t in 0..mesh.triangle_slots() as TriangleId {
        if let Some(tri) = proper_triangle(mesh, t) {
            for v in tri {
                let l = local[v as usize];
                if l != NONE {
                    fans[next[l as usize] as usize] = t;
                    next[l as usize] += 1;
                }
            }
        }
    }
    let fan = |v: VertexId| {
        let l = local[v as usize] as usize;
        &fans[starts[l] as usize..starts[l + 1] as usize]
    };

    // Regions (selected triangles joined across selected edges), corner
    // groups (corners of one vertex joined across selected edges at it) and
    // the boundary edges that get walls.
    let mut regions = UnionFind::new();
    regions.reset(n)?;
    let mut groups = UnionFind::new();
    groups.reset(3 * n)?;
    let mut wall_edges: Vec<(u32, u32)> = Vec::new();
    let mut on_wall = filled(nv, false)?;
    for (i, &t) in sel.iter().enumerate() {
        let tri = tris[t as usize];
        for k in 0..3 {
            let (u, w) = (tri[k], tri[(k + 1) % 3]);
            let (users, count) = shared(fan(u), fan(w));
            if count > 2 {
                return Err(OpsError::InvalidInput(
                    "a selected edge is shared by three or more triangles",
                ));
            }
            let other = if users[0] == t { users[1] } else { users[0] };
            if count == 2 && in_sel[other as usize] {
                if other < t {
                    continue;
                }
                let o_tri = tris[other as usize];
                let at = |x: VertexId| o_tri.iter().position(|&y| y == x).expect("shared vertex");
                let (ou, ow) = (at(u), at(w));
                if (ow + 1) % 3 != ou {
                    return Err(OpsError::InvalidInput(
                        "selected triangles are wound against each other",
                    ));
                }
                let j = sel.binary_search(&other).expect("selected") as u32;
                let i = i as u32;
                regions.union(i, j);
                groups.union(3 * i + k as u32, 3 * j + ou as u32);
                groups.union(3 * i + ((k + 1) % 3) as u32, 3 * j + ow as u32);
                continue;
            }
            push(&mut wall_edges, (i as u32, k as u32))?;
            on_wall[local[u as usize] as usize] = true;
            on_wall[local[w as usize] as usize] = true;
        }
    }

    // Region numbers, in order of their first triangle.
    let mut region = filled(n, NONE)?;
    let mut region_count = 0usize;
    for i in 0..n {
        let r = regions.find(i as u32) as usize;
        if region[r] == NONE {
            region[r] = region_count as u32;
            region_count += 1;
        }
        region[i] = region[r];
    }

    // One group per fan of corners; its cap vertex is the vertex itself when
    // the vertex has nothing else (no unselected triangle, no wall, no
    // earlier group), else a new copy.
    let mut group_of = filled(3 * n, NONE)?;
    let mut group_vertex: Vec<VertexId> = Vec::new();
    for c in 0..3 * n {
        let r = groups.find(c as u32) as usize;
        if r == c {
            group_of[c] = group_vertex.len() as u32;
            push(&mut group_vertex, tris[sel[c / 3] as usize][c % 3])?;
        } else {
            group_of[c] = group_of[r];
        }
    }
    let g_count = group_vertex.len();
    let slots = mesh.vertex_slots();
    let mut taken = filled(nv, false)?;
    let mut cap_vertex = filled(g_count, NONE)?;
    let mut parents: Vec<VertexId> = Vec::new();
    for (g, &v) in group_vertex.iter().enumerate() {
        let l = local[v as usize] as usize;
        let alone = !on_wall[l] && fan(v).iter().all(|&x| in_sel[x as usize]);
        if alone && !taken[l] {
            taken[l] = true;
            cap_vertex[g] = v;
        } else {
            let id = slots + parents.len();
            if id >= NONE as usize {
                return Err(OpsError::InvalidInput("too many vertices for an extrusion"));
            }
            cap_vertex[g] = id as VertexId;
            push(&mut parents, v)?;
        }
    }

    // Where each group goes, in the measuring frame, then mapped back.
    let area_vector = |t: TriangleId| {
        let [a, b, c] = tris[t as usize].map(|v| point(mesh, v));
        cross(sub(b, a), sub(c, a))
    };
    let mut world_move = filled(g_count, [0.0f64; 3])?;
    match (o.direction, fixed) {
        (_, Some(d)) => world_move.fill(d.map(|x| o.distance * x)),
        (ExtrudeDirection::RegionNormal, _) => {
            let mut sum = filled(region_count, [0.0f64; 3])?;
            let mut size = filled(region_count, 0.0f64)?;
            for (i, &t) in sel.iter().enumerate() {
                let nw = apply(&cof, area_vector(t));
                let r = region[i] as usize;
                sum[r] = [sum[r][0] + nw[0], sum[r][1] + nw[1], sum[r][2] + nw[2]];
                size[r] += dot(nw, nw).sqrt();
            }
            let mut way = filled(region_count, [0.0f64; 3])?;
            for r in 0..region_count {
                let len = dot(sum[r], sum[r]).sqrt();
                if !(len.is_finite() && len > 1e-12 * size[r]) {
                    return Err(OpsError::InvalidInput(
                        "a selected region's normals cancel out; give a direction",
                    ));
                }
                way[r] = sum[r].map(|x| o.distance * x / len);
            }
            for c in 0..3 * n {
                world_move[group_of[c] as usize] = way[region[c / 3] as usize];
            }
        }
        (_, None) => {
            let mut sum = filled(g_count, [0.0f64; 3])?;
            for c in 0..3 * n {
                let tri = tris[sel[c / 3] as usize];
                let k = c % 3;
                let p = [0, 1, 2].map(|d| apply(&o.to_world, point(mesh, tri[(k + d) % 3])));
                let (e1, e2) = (sub(p[1], p[0]), sub(p[2], p[0]));
                let nw = cross(e1, e2);
                let len = dot(nw, nw).sqrt();
                if !(len.is_finite() && len > 0.0) {
                    continue;
                }
                let angle = len.atan2(dot(e1, e2));
                let g = group_of[c] as usize;
                sum[g] = std::array::from_fn(|x| sum[g][x] + angle * nw[x] / len);
            }
            for g in 0..g_count {
                let d = unit(sum[g]).ok_or(OpsError::InvalidInput(
                    "a selected vertex's normals cancel out; give a direction",
                ))?;
                world_move[g] = d.map(|x| o.distance * x);
            }
        }
    }

    let mut moved: Vec<(VertexId, [f32; 3])> = Vec::new();
    let mut added = filled(parents.len(), [0.0f32; 3])?;
    for g in 0..g_count {
        let v = group_vertex[g];
        let step = apply(&to_local, world_move[g]);
        let p = point(mesh, v);
        let q = to_f32([p[0] + step[0], p[1] + step[1], p[2] + step[2]]).ok_or(
            OpsError::InvalidInput("an extruded position does not fit f32"),
        )?;
        let id = cap_vertex[g];
        if id == v {
            push(&mut moved, (v, q))?;
        } else {
            added[id as usize - slots] = q;
        }
    }
    moved.sort_unstable_by_key(|&(v, _)| v);

    let position = |v: VertexId| -> [f64; 3] {
        if v as usize >= slots {
            added[v as usize - slots].map(f64::from)
        } else if let Ok(i) = moved.binary_search_by_key(&v, |&(id, _)| id) {
            moved[i].1.map(f64::from)
        } else {
            point(mesh, v)
        }
    };
    let mut caps: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut caps, n)?;
    for i in 0..n {
        caps.push([0, 1, 2].map(|k| cap_vertex[group_of[3 * i + k] as usize]));
    }
    let mut walls: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut walls, 2 * wall_edges.len())?;
    for &(i, k) in &wall_edges {
        let (i, k) = (i as usize, k as usize);
        let tri = tris[sel[i] as usize];
        let (u, w) = (tri[k], tri[(k + 1) % 3]);
        let (u2, w2) = (caps[i][k], caps[i][(k + 1) % 3]);
        // The quad u -> w -> w2 -> u2, split along its shorter diagonal.
        let length = |a: VertexId, b: VertexId| {
            let d = apply(&o.to_world, sub(position(a), position(b)));
            dot(d, d)
        };
        if length(u, w2) <= length(w, u2) {
            walls.extend([[u, w, w2], [u, w2, u2]]);
        } else {
            walls.extend([[u, w, u2], [w, w2, u2]]);
        }
    }

    let found = if o.check_self_intersections {
        let mut planned: Vec<[VertexId; 3]> = Vec::new();
        reserve(&mut planned, caps.len() + walls.len())?;
        planned.extend_from_slice(&caps);
        planned.extend_from_slice(&walls);
        Some(intersections(&Planned {
            mesh,
            moved: &moved,
            added: &added,
            triangles: &planned,
            replaced: &sel,
        })?)
    } else {
        None
    };
    Ok(Plan {
        sel,
        regions: region_count,
        vertex_slots: slots as u32,
        moved,
        added,
        parents,
        caps,
        walls,
        found,
    })
}

impl Plan {
    /// Writes the plan. Everything the report needs is reserved first, so
    /// nothing but the store's own writes can fail after the first one.
    fn write(self, tx: &mut Transaction) -> Result<ExtrudeFacesReport> {
        let n = self.sel.len();
        let mut new_ids: Vec<VertexId> = Vec::new();
        reserve(&mut new_ids, self.parents.len())?;
        let mut moved_ids: Vec<VertexId> = Vec::new();
        reserve(&mut moved_ids, self.moved.len())?;
        let mut moved_at: Vec<[f32; 3]> = Vec::new();
        reserve(&mut moved_at, self.moved.len())?;
        let mut rewired: Vec<TriangleId> = Vec::new();
        reserve(&mut rewired, n)?;
        let mut rewired_to: Vec<[VertexId; 3]> = Vec::new();
        reserve(&mut rewired_to, n)?;
        let mut walls: Vec<[VertexId; 3]> = Vec::new();
        reserve(&mut walls, self.walls.len())?;
        let mut final_ids: Vec<TriangleId> = Vec::new();
        let mut crossing: Vec<TriangleId> = Vec::new();
        if let Some(found) = &self.found {
            reserve(&mut final_ids, n + self.walls.len())?;
            reserve(&mut crossing, found.len())?;
        }
        let mut wall_ids: Vec<TriangleId> = Vec::new();
        reserve(&mut wall_ids, self.walls.len())?;

        for (&v, &p) in self.parents.iter().zip(&self.added) {
            new_ids.push(tx.add_vertex_with_parents(p, &[(v, 1.0)])?);
        }
        let slots = self.vertex_slots;
        let real = |v: VertexId| {
            if v >= slots {
                new_ids[(v - slots) as usize]
            } else {
                v
            }
        };
        moved_ids.extend(self.moved.iter().map(|&(v, _)| v));
        moved_at.extend(self.moved.iter().map(|&(_, p)| p));
        tx.set_positions(&moved_ids, &moved_at)?;
        let old = tx.mesh().triangles();
        for (&t, cap) in self.sel.iter().zip(&self.caps) {
            let cap = cap.map(real);
            if cap != old[t as usize] {
                rewired.push(t);
                rewired_to.push(cap);
            }
        }
        tx.set_triangles(&rewired, &rewired_to)?;
        walls.extend(self.walls.iter().map(|w| w.map(real)));
        let added = tx.add_triangles(&walls)?;
        wall_ids.extend_from_slice(&added);
        wall_ids.sort_unstable();
        let self_intersections = self.found.as_ref().map(|found| {
            final_ids.extend_from_slice(&self.sel);
            final_ids.extend_from_slice(&added);
            found.write_final(&final_ids, &mut crossing);
            crossing
        });
        Ok(ExtrudeFacesReport {
            regions: self.regions,
            cap_triangles: n,
            moved_vertices: self.moved.len(),
            added_vertices: self.parents.len(),
            wall_triangles: wall_ids,
            self_intersections,
        })
    }
}
