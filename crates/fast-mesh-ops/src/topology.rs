//! Edge maps, connected components and boundary loops.
//!
//! Everything here is built from the triangles alone, never from the store's
//! cached adjacency, which a transaction does not keep current. Triangles
//! with a repeated vertex id are skipped throughout.

use fast_mesh_store::{Mesh, TriangleId, VertexId};

use crate::{OpsError, Result, filled, push, reserve};

/// How many triangles share an edge. Re-exported from `fast_mesh_store`,
/// whose corner table uses the same three kinds with the same meaning, so
/// that callers of both crates see one type.
pub use fast_mesh_store::EdgeKind;

const NONE: u32 = u32::MAX;

/// The undirected key of the edge `a`-`b`: `(min, max)`.
fn key(a: VertexId, b: VertexId) -> (VertexId, VertexId) {
    if a < b { (a, b) } else { (b, a) }
}

/// The vertices of triangle `t` when it is alive and has three distinct ones.
pub(crate) fn proper_triangle(mesh: &Mesh, t: TriangleId) -> Option<[VertexId; 3]> {
    let [a, b, c] = mesh.triangle(t)?;
    (a != b && b != c && a != c).then_some([a, b, c])
}

/// The local position of `v` in `f64`, origin excluded.
pub(crate) fn point(mesh: &Mesh, v: VertexId) -> [f64; 3] {
    mesh.positions()[v as usize].map(f64::from)
}

pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// The area of triangle `[a, b, c]`.
pub(crate) fn triangle_area(mesh: &Mesh, [a, b, c]: [VertexId; 3]) -> f64 {
    let p = point(mesh, a);
    0.5 * norm(cross(sub(point(mesh, b), p), sub(point(mesh, c), p)))
}

/// Every undirected edge of the proper triangles, sorted by key, with the
/// triangles using it in ascending id order.
pub(crate) struct EdgeMap {
    keys: Vec<(VertexId, VertexId)>,
    /// Edge `i` is used by `tris[starts[i]..starts[i + 1]]`.
    starts: Vec<u32>,
    tris: Vec<TriangleId>,
}

impl EdgeMap {
    /// Builds the map from the alive triangles, skipping those with a
    /// repeated vertex id.
    pub(crate) fn build(mesh: &Mesh) -> Result<EdgeMap> {
        let mut pairs: Vec<((VertexId, VertexId), TriangleId)> = Vec::new();
        reserve(&mut pairs, 3 * mesh.triangle_count())?;
        for t in 0..mesh.triangle_slots() as TriangleId {
            if let Some([a, b, c]) = proper_triangle(mesh, t) {
                for (u, v) in [(a, b), (b, c), (c, a)] {
                    push(&mut pairs, (key(u, v), t))?;
                }
            }
        }
        if pairs.len() >= NONE as usize {
            return Err(OpsError::InvalidInput("too many triangles for an edge map"));
        }
        pairs.sort_unstable();
        let distinct = 1 + pairs.windows(2).filter(|w| w[0].0 != w[1].0).count();
        let distinct = if pairs.is_empty() { 0 } else { distinct };
        let mut map = EdgeMap {
            keys: Vec::new(),
            starts: Vec::new(),
            tris: Vec::new(),
        };
        reserve(&mut map.keys, distinct)?;
        reserve(&mut map.starts, distinct + 1)?;
        reserve(&mut map.tris, pairs.len())?;
        for (i, &(k, t)) in pairs.iter().enumerate() {
            if map.keys.last() != Some(&k) {
                map.keys.push(k);
                map.starts.push(i as u32);
            }
            map.tris.push(t);
        }
        map.starts.push(pairs.len() as u32);
        Ok(map)
    }

    /// The index of edge `a`-`b`, in either direction.
    fn index(&self, a: VertexId, b: VertexId) -> Option<usize> {
        self.keys.binary_search(&key(a, b)).ok()
    }

    fn at(&self, i: usize) -> &[TriangleId] {
        &self.tris[self.starts[i] as usize..self.starts[i + 1] as usize]
    }

    /// The number of edges.
    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }

    /// The kind of edge `a`-`b`, or `None` when no proper triangle uses it.
    #[allow(dead_code, reason = "used by the repair operations of later tasks")]
    pub(crate) fn kind(&self, a: VertexId, b: VertexId) -> Option<EdgeKind> {
        self.index(a, b).map(|i| kind_of(self.at(i).len()))
    }

    /// The triangles using edge `a`-`b` in ascending id order; empty when
    /// there are none.
    pub(crate) fn triangles(&self, a: VertexId, b: VertexId) -> &[TriangleId] {
        self.index(a, b).map_or(&[], |i| self.at(i))
    }

    /// Every edge in key order, with its triangles.
    pub(crate) fn edges(&self) -> impl Iterator<Item = ((VertexId, VertexId), &[TriangleId])> {
        (0..self.keys.len()).map(|i| (self.keys[i], self.at(i)))
    }
}

fn kind_of(triangles: usize) -> EdgeKind {
    match triangles {
        1 => EdgeKind::Boundary,
        2 => EdgeKind::Interior,
        _ => EdgeKind::NonManifold,
    }
}

/// For every vertex, the proper triangles using it, in ascending id order.
pub(crate) struct VertexFans {
    /// Vertex `v`'s fan is `tris[starts[v]..starts[v + 1]]`.
    starts: Vec<u32>,
    tris: Vec<TriangleId>,
}

impl VertexFans {
    pub(crate) fn build(mesh: &Mesh) -> Result<VertexFans> {
        let slots = mesh.vertex_slots();
        let mut starts = filled(slots + 1, 0u32)?;
        let mut corners = 0usize;
        for t in 0..mesh.triangle_slots() as TriangleId {
            if let Some(tri) = proper_triangle(mesh, t) {
                for v in tri {
                    starts[v as usize + 1] += 1;
                }
                corners += 3;
            }
        }
        if corners >= NONE as usize {
            return Err(OpsError::InvalidInput("too many triangles for vertex fans"));
        }
        // Prefix sums give each fan's end; filling backwards from the end with
        // descending triangle ids leaves every fan ascending and `starts` at
        // the fans' beginnings.
        for v in 0..slots {
            starts[v + 1] += starts[v];
        }
        let mut tris = filled(corners, 0)?;
        for t in (0..mesh.triangle_slots() as TriangleId).rev() {
            if let Some(tri) = proper_triangle(mesh, t) {
                for v in tri {
                    let end = &mut starts[v as usize + 1];
                    *end -= 1;
                    tris[*end as usize] = t;
                }
            }
        }
        // Each fan's end moved back to its start: shift the starts down one slot.
        starts.copy_within(1.., 0);
        starts[slots] = corners as u32;
        Ok(VertexFans { starts, tris })
    }

    /// The proper triangles using `v`, ascending.
    pub(crate) fn fan(&self, v: VertexId) -> &[TriangleId] {
        &self.tris[self.starts[v as usize] as usize..self.starts[v as usize + 1] as usize]
    }
}

/// A union-find over `0..n` whose roots are always the smallest member.
pub(crate) struct UnionFind {
    parent: Vec<u32>,
}

impl UnionFind {
    pub(crate) fn new() -> Self {
        UnionFind { parent: Vec::new() }
    }

    /// Resets to `n` singletons, reusing the storage.
    pub(crate) fn reset(&mut self, n: usize) -> Result<()> {
        self.parent.clear();
        reserve(&mut self.parent, n)?;
        self.parent.extend(0..n as u32);
        Ok(())
    }

    pub(crate) fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let up = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = up;
            x = up;
        }
        x
    }

    /// Joins the sets of `a` and `b`; `true` when they were separate.
    pub(crate) fn union(&mut self, a: u32, b: u32) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        self.parent[hi as usize] = lo;
        true
    }
}

/// A set of triangles connected through shared edges of any kind.
#[derive(Debug, Clone, PartialEq)]
pub struct Component {
    /// The triangles, ascending.
    pub triangles: Vec<TriangleId>,
    /// The total triangle area, in local units squared.
    pub area: f64,
    /// `true` when the component has no boundary and no non-manifold edge.
    pub closed: bool,
}

/// The connected components of the proper triangles, sorted by their
/// smallest triangle id.
pub(crate) fn components(mesh: &Mesh, edges: &EdgeMap) -> Result<Vec<Component>> {
    let slots = mesh.triangle_slots();
    let mut uf = UnionFind::new();
    uf.reset(slots)?;
    for (_, tris) in edges.edges() {
        for &t in &tris[1..] {
            uf.union(tris[0], t);
        }
    }
    // The roots are the smallest triangle ids, so visiting triangles in
    // ascending order numbers the components in the required order.
    let mut index = filled(slots, NONE)?;
    let mut out: Vec<Component> = Vec::new();
    for t in 0..slots as TriangleId {
        let Some(tri) = proper_triangle(mesh, t) else {
            continue;
        };
        let root = uf.find(t) as usize;
        if index[root] == NONE {
            index[root] = out.len() as u32;
            push(
                &mut out,
                Component {
                    triangles: Vec::new(),
                    area: 0.0,
                    closed: true,
                },
            )?;
        }
        let c = &mut out[index[root] as usize];
        push(&mut c.triangles, t)?;
        c.area += triangle_area(mesh, tri);
    }
    for (_, tris) in edges.edges() {
        if tris.len() != 2 {
            let root = uf.find(tris[0]) as usize;
            out[index[root] as usize].closed = false;
        }
    }
    Ok(out)
}

/// The vertices whose fan of proper triangles splits into more than one
/// piece, ascending. Triangles around a vertex `v` are in one piece when they
/// are linked through edges at `v`, of any kind; a bowtie corner is a pinch,
/// a fin's spine is not.
pub(crate) fn pinch_vertices(mesh: &Mesh, edges: &EdgeMap) -> Result<Vec<VertexId>> {
    let fans = VertexFans::build(mesh)?;
    let mut uf = UnionFind::new();
    let mut out = Vec::new();
    for v in 0..mesh.vertex_slots() as VertexId {
        let fan = fans.fan(v);
        if fan.len() < 2 {
            continue;
        }
        uf.reset(fan.len())?;
        let mut pieces = fan.len();
        for (i, &t) in fan.iter().enumerate() {
            for x in mesh.triangles()[t as usize] {
                if x == v {
                    continue;
                }
                for &u in edges.triangles(v, x) {
                    if u > t {
                        let j = fan.binary_search(&u).expect("edge triangle in fan");
                        if uf.union(i as u32, j as u32) {
                            pieces -= 1;
                        }
                    }
                }
            }
        }
        if pieces > 1 {
            push(&mut out, v)?;
        }
    }
    Ok(out)
}

/// A content fingerprint of a boundary loop, valid across edits and
/// transactions for as long as the loop's vertex sequence is unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoopId {
    /// The smallest vertex id on the loop.
    pub min_vertex: VertexId,
    /// The number of vertices on the loop.
    pub len: u32,
    /// FNV-1a 64 over the little-endian bytes of the vertex ids in
    /// [`BoundaryLoop::vertices`] order: for a closed loop, starting at
    /// `min_vertex`; for an open chain, from its first end.
    pub hash: u64,
}

/// A chain of boundary edges, closed unless the walk could not close it.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryLoop {
    /// The loop's content fingerprint, which callers pass back to pick this
    /// loop (for example to fill it). It matches as long as the loop's vertex
    /// sequence is unchanged, across edits and transactions.
    pub id: LoopId,
    /// The vertices in loop order. The loop runs along each boundary edge in
    /// the direction its one triangle uses it, so a hole in an outward-facing
    /// surface runs clockwise seen from outside. A closed loop starts at
    /// `id.min_vertex` and its polygon closes from the last vertex back to the
    /// first. An open chain runs from one end to the other, unrotated.
    pub vertices: Vec<VertexId>,
    /// The length of the polygon closed from the last vertex back to the
    /// first, in local units (for an open chain, including that closing step).
    pub perimeter: f64,
    /// The vector area `|sum p_i x p_(i+1)| / 2` of the same closed polygon.
    pub area: f64,
    /// `true` when the walk came back to its first edge. `false` for an open
    /// chain, which ends at a vertex where no boundary edge continues it (for
    /// example at a non-manifold edge). An open chain is never clean, so it is
    /// never filled.
    pub closed: bool,
    /// `false` when the edges at some vertex of the loop could not be ordered
    /// by the fan walk: the vertex touches a non-manifold edge, or it is a
    /// pinch whose fan pieces are wound against each other, so the walk fell
    /// back to pairing by edge key. The loop may then be open or join openings
    /// that are not one hole. This fails safe: unclean loops are skipped by
    /// `fill_holes` unless allowed, and orienting first makes a wound-against
    /// pinch clean.
    pub clean: bool,
}

/// FNV-1a 64 over the little-endian bytes of `ids`.
pub(crate) fn fnv1a(ids: &[VertexId]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for id in ids {
        for byte in id.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The boundary loops of `mesh`, sorted by smallest vertex, then by vertex
/// sequence (for a closed loop, the vertex after the smallest).
///
/// Each loop follows the boundary edges (edges of exactly one proper
/// triangle). At each vertex `b`, entered along `a -> b`, the walk rotates
/// through `b`'s triangle fan from the incoming edge's triangle across
/// interior edges to the far side of that fan piece. At an ordinary vertex
/// that far side is the next edge. At a pinch vertex, where the fan splits
/// into pieces, the far side borders the other opening, so the next edge is
/// taken across the gap from another piece instead (by turning angle when
/// there are several): two holes sharing one vertex are two loops. Where the
/// fan reaches a non-manifold edge, the outgoing boundary edges at that
/// vertex are paired in key order instead, and loops through a vertex on a
/// non-manifold edge are marked unclean.
///
/// Walks first start at chain heads, vertices with more outgoing than
/// incoming boundary half-edges, so that an open chain is walked whole from
/// its first end; the closed loops are walked after.
pub fn boundary_loops(mesh: &Mesh) -> Result<Vec<BoundaryLoop>> {
    let edges = EdgeMap::build(mesh)?;
    loops_with(mesh, &edges)
}

/// A boundary edge traversed `from -> to`; `tri` is its one triangle and
/// `edge` its index in the edge map.
#[derive(Clone, Copy)]
struct Step {
    from: VertexId,
    to: VertexId,
    tri: TriangleId,
    edge: usize,
}

/// [`boundary_loops`] with a prebuilt edge map.
pub(crate) fn loops_with(mesh: &Mesh, edges: &EdgeMap) -> Result<Vec<BoundaryLoop>> {
    let mut non_manifold = filled(mesh.vertex_slots(), false)?;
    // Boundary half-edges, in the direction their triangle uses them, sorted
    // by `(from, to)`.
    let mut half: Vec<Step> = Vec::new();
    for (i, ((a, b), tris)) in edges.edges().enumerate() {
        match tris.len() {
            1 => {
                let t = tris[0];
                let [x, y, z] = mesh.triangles()[t as usize];
                let forward = (x, y) == (a, b) || (y, z) == (a, b) || (z, x) == (a, b);
                let (from, to) = if forward { (a, b) } else { (b, a) };
                push(
                    &mut half,
                    Step {
                        from,
                        to,
                        tri: t,
                        edge: i,
                    },
                )?;
            }
            2 => {}
            _ => {
                non_manifold[a as usize] = true;
                non_manifold[b as usize] = true;
            }
        }
    }
    half.sort_unstable_by_key(|s| (s.from, s.to));
    // Outgoing minus incoming boundary half-edges per vertex: chain heads are
    // the vertices where it is positive.
    let mut excess = filled(mesh.vertex_slots(), 0i64)?;
    for s in &half {
        excess[s.from as usize] += 1;
        excess[s.to as usize] -= 1;
    }
    let mut visited = filled(edges.len(), false)?;
    let mut loops: Vec<BoundaryLoop> = Vec::new();
    let mut verts: Vec<VertexId> = Vec::new();
    let order = (0..2).flat_map(|pass| (0..half.len()).map(move |h| (pass, h)));
    for (pass, h) in order {
        let start = half[h];
        if visited[start.edge] || (pass == 0 && excess[start.from as usize] <= 0) {
            continue;
        }
        let mut closed = true;
        verts.clear();
        let mut clean = true;
        let mut cur = start;
        loop {
            visited[cur.edge] = true;
            push(&mut verts, cur.from)?;
            let b = cur.to;
            let lo = half.partition_point(|s| s.from < b);
            let hi = lo + half[lo..].iter().take_while(|s| s.from == b).count();
            let leaving = &half[lo..hi];
            // The fan walk crosses the incoming edge's fan piece to its far
            // side. At a pinch the next edge of the same hole lies across the
            // gap, in another piece; otherwise it is the far side itself.
            let walked = fan_walk(mesh, edges, cur.tri, cur.from, b).and_then(|(c, t)| {
                let e = edges.index(b, c).expect("walked edge in map");
                let s = across_gap(mesh, leaving, e, cur).unwrap_or(Step {
                    from: b,
                    to: c,
                    tri: t,
                    edge: e,
                });
                (s.edge == start.edge || !visited[s.edge]).then_some(s)
            });
            let next = match walked {
                Some(s) => Some(s),
                None => {
                    // Pair by key order: the first outgoing boundary half-edge at
                    // `b` that is unvisited or closes this loop.
                    clean = false;
                    leaving
                        .iter()
                        .find(|s| s.edge == start.edge || !visited[s.edge])
                        .copied()
                }
            };
            match next {
                Some(s) if s.edge == start.edge => break,
                Some(s) => cur = s,
                None => {
                    // Nothing leaves `b`: an open chain ending at `b`. Only the
                    // key-order fallback can get here, so it is already unclean.
                    push(&mut verts, b)?;
                    closed = false;
                    clean = false;
                    break;
                }
            }
        }
        if verts.iter().any(|&v| non_manifold[v as usize]) {
            clean = false;
        }
        push(&mut loops, make_loop(mesh, &verts, closed, clean)?)?;
    }
    loops.sort_unstable_by(|a, b| {
        (a.id.min_vertex, &a.vertices).cmp(&(b.id.min_vertex, &b.vertices))
    });
    Ok(loops)
}

/// At a pinch vertex `b`, the boundary half-edge that continues the hole
/// entered along `cur` (`a -> b`): one leaving `b` in another fan piece than
/// `cur`'s, whose far side at `b` is the edge `walked`. With one such edge it
/// is the answer. With several, it is the first met turning from `b -> a`
/// counter-clockwise around `b`'s local normal (into the gap beside `cur`'s
/// piece), ties to edge-key order. `None` when no other piece has an
/// outgoing boundary edge, as at an ordinary vertex.
fn across_gap(mesh: &Mesh, leaving: &[Step], walked: usize, cur: Step) -> Option<Step> {
    // `cur` itself leaves `b` when the walk runs against its triangle's
    // direction, past an orientation flip.
    let other = |s: &&Step| s.edge != walked && s.edge != cur.edge;
    let mut others = leaving.iter().filter(other);
    let first = *others.next()?;
    if others.clone().next().is_none() {
        return Some(first);
    }
    let normal = |t: TriangleId| {
        let [x, y, z] = mesh.triangles()[t as usize].map(|v| point(mesh, v));
        cross(sub(y, x), sub(z, x))
    };
    let mut n = normal(cur.tri);
    for s in leaving.iter().filter(other) {
        let m = normal(s.tri);
        n = [n[0] + m[0], n[1] + m[1], n[2] + m[2]];
    }
    let pb = point(mesh, cur.to);
    let nn = dot(n, n);
    let flat = |v: VertexId| {
        let d = sub(point(mesh, v), pb);
        let k = dot(n, d) / nn;
        [d[0] - k * n[0], d[1] - k * n[1], d[2] - k * n[2]]
    };
    let d0 = flat(cur.from);
    let turn = |s: &Step| {
        let d = flat(s.to);
        let a = dot(n, cross(d0, d)).atan2(dot(d0, d) * norm(n));
        let a = if a <= 0.0 {
            a + std::f64::consts::TAU
        } else {
            a
        };
        if a.is_finite() { a } else { f64::INFINITY }
    };
    let mut best = (turn(&first), first);
    for s in others {
        let a = turn(s);
        if a < best.0 {
            best = (a, *s);
        }
    }
    Some(best.1)
}

/// Rotates around vertex `b` from triangle `t`, entered across its edge
/// `a`-`b`, through interior edges until a boundary edge `b`-`c` is reached.
/// Returns `c` and that edge's triangle, or `None` at a non-manifold edge or
/// when the walk takes more steps than `b` has triangles.
fn fan_walk(
    mesh: &Mesh,
    edges: &EdgeMap,
    t: TriangleId,
    a: VertexId,
    b: VertexId,
) -> Option<(VertexId, TriangleId)> {
    let (mut tri, mut prev) = (t, a);
    for _ in 0..=mesh.corner_count(b) {
        let c = mesh.triangles()[tri as usize]
            .into_iter()
            .find(|&v| v != b && v != prev)?;
        match edges.triangles(b, c) {
            [_] => return Some((c, tri)),
            &[x, y] => {
                tri = if x == tri { y } else { x };
                prev = c;
            }
            _ => return None,
        }
    }
    None
}

/// The loop through `verts`: a closed one rotated to its canonical start, an
/// open chain kept as walked so that its ends stay first and last.
fn make_loop(mesh: &Mesh, verts: &[VertexId], closed: bool, clean: bool) -> Result<BoundaryLoop> {
    let n = verts.len();
    let min = *verts.iter().min().expect("a loop has vertices");
    // Among the occurrences of the smallest vertex, start at the one giving
    // the lexicographically smallest rotation, so the start does not depend
    // on where the walk began.
    let rotation = |i: usize| verts[i..].iter().chain(&verts[..i]);
    let mut best = 0;
    if closed {
        best = verts
            .iter()
            .position(|&v| v == min)
            .expect("min is present");
        for (i, &v) in verts.iter().enumerate().skip(best + 1) {
            if v == min && rotation(i).cmp(rotation(best)).is_lt() {
                best = i;
            }
        }
    }
    let mut vertices = Vec::new();
    reserve(&mut vertices, n)?;
    vertices.extend(rotation(best));
    let (mut perimeter, mut vector) = (0.0, [0.0; 3]);
    for i in 0..n {
        let (p, q) = (point(mesh, vertices[i]), point(mesh, vertices[(i + 1) % n]));
        perimeter += norm(sub(q, p));
        let c = cross(p, q);
        vector = [vector[0] + c[0], vector[1] + c[1], vector[2] + c[2]];
    }
    Ok(BoundaryLoop {
        id: LoopId {
            min_vertex: min,
            // Fewer than `u32::MAX` edges exist (checked by `EdgeMap::build`).
            len: n as u32,
            hash: fnv1a(&vertices),
        },
        vertices,
        perimeter,
        area: 0.5 * norm(vector),
        closed,
        clean,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::*;
    use crate::{OpsError, fail_allocations_after};
    use fast_mesh_store::{Mesh, StoreError};

    fn kinds(mesh: &Mesh) -> Vec<((VertexId, VertexId), EdgeKind)> {
        let map = EdgeMap::build(mesh).unwrap();
        map.edges()
            .map(|(k, _)| (k, map.kind(k.0, k.1).unwrap()))
            .collect()
    }

    fn count(mesh: &Mesh, kind: EdgeKind) -> usize {
        kinds(mesh).iter().filter(|(_, k)| *k == kind).count()
    }

    #[test]
    fn edge_kinds_on_fixtures() {
        let sphere = uv_sphere(8, 12, 1.0);
        assert_eq!(count(&sphere, EdgeKind::Boundary), 0);
        assert_eq!(count(&sphere, EdgeKind::NonManifold), 0);
        assert_eq!(
            count(&sphere, EdgeKind::Interior),
            3 * sphere.triangle_count() / 2
        );
        assert_eq!(count(&grid(4), EdgeKind::Boundary), 16);
        assert_eq!(count(&grid(4), EdgeKind::NonManifold), 0);
        let fin = fin();
        assert_eq!(count(&fin, EdgeKind::NonManifold), 1);
        let map = EdgeMap::build(&fin).unwrap();
        assert_eq!(map.kind(1, 0), Some(EdgeKind::NonManifold));
        assert_eq!(map.triangles(1, 0), &[1, 3, 5]);
        assert_eq!(map.kind(0, 3), Some(EdgeKind::Interior));
        assert_eq!(map.triangles(3, 0), &[0, 1]);
        assert_eq!(map.kind(2, 4), None);
        assert_eq!(map.triangles(2, 4), &[] as &[TriangleId]);
    }

    #[test]
    fn components_count_and_closed() {
        let two = combine(&[&uv_sphere(8, 12, 1.0), &uv_sphere(6, 8, 0.25)]);
        let map = EdgeMap::build(&two).unwrap();
        let comps = components(&two, &map).unwrap();
        assert_eq!(comps.len(), 2);
        assert!(comps.iter().all(|c| c.closed));
        assert_eq!(comps[0].triangles.len(), 2 * 12 * 7);
        assert_eq!(comps[1].triangles.len(), 2 * 8 * 5);
        assert_eq!(comps[1].triangles[0], 2 * 12 * 7);
        assert!(comps[0].area > comps[1].area);
        // A UV sphere's area approaches 4 pi r^2 from below.
        assert!(comps[0].area < 4.0 * std::f64::consts::PI && comps[0].area > 3.5);

        let hemi = hemisphere(4, 16, 1.0);
        let map = EdgeMap::build(&hemi).unwrap();
        let comps = components(&hemi, &map).unwrap();
        assert_eq!(comps.len(), 1);
        assert!(!comps[0].closed);
        assert_eq!(comps[0].triangles.len(), hemi.triangle_count());

        // The fin is one component (shared edges of any kind connect), not closed.
        let fin = fin();
        let comps = components(&fin, &EdgeMap::build(&fin).unwrap()).unwrap();
        assert_eq!(comps.len(), 1);
        assert!(!comps[0].closed);
    }

    #[test]
    fn hemisphere_has_one_loop() {
        let r = 2.0;
        let loops = boundary_loops(&hemisphere(8, 64, r)).unwrap();
        assert_eq!(loops.len(), 1);
        let l = &loops[0];
        assert_eq!(l.vertices.len(), 64);
        assert_eq!(l.id.len, 64);
        assert!(l.clean);
        let circumference = 2.0 * std::f64::consts::PI * f64::from(r);
        assert!((l.perimeter - circumference).abs() < 0.01 * circumference);
        // The rim is a flat circle: vector area is about pi r^2.
        let disc = std::f64::consts::PI * f64::from(r * r);
        assert!((l.area - disc).abs() < 0.01 * disc);
        // Ring 8 starts at vertex 1 + 7 * 64. With outward winding the rim
        // runs counter-clockwise from +z, in ring order.
        let first = 1 + 7 * 64;
        assert_eq!(l.id.min_vertex, first);
        assert_eq!(l.vertices[0], first);
        assert_eq!(l.vertices[1], first + 1);
    }

    #[test]
    fn bowtie_holes_are_two_clean_loops() {
        let mesh = bowtie_holes();
        let loops = boundary_loops(&mesh).unwrap();
        let seqs: Vec<&[VertexId]> = loops.iter().map(|l| l.vertices.as_slice()).collect();
        assert_eq!(
            seqs,
            [
                // The grid border, counter-clockwise from +z.
                &[0, 1, 2, 3, 4, 9, 14, 19, 24, 23, 22, 21, 20, 15, 10, 5][..],
                // Cell (1, 1): its corners 6, 7, 12, 11 walked clockwise.
                &[6, 11, 12, 7][..],
                // Cell (2, 2): its corners 12, 13, 18, 17 walked clockwise.
                &[12, 17, 18, 13][..],
            ]
        );
        assert!(loops.iter().all(|l| l.clean && l.closed));
        assert_eq!(loops[1].id.len, 4);
        assert_eq!(loops[2].id.len, 4);
        assert_eq!(loops[1].id.min_vertex, 6);
        assert_eq!(loops[2].id.min_vertex, 12);
        for l in &loops[1..] {
            assert!((l.perimeter - 4.0).abs() < 1e-12);
            assert!((l.area - 1.0).abs() < 1e-12);
        }
        assert!((loops[0].perimeter - 16.0).abs() < 1e-12);
        assert!((loops[0].area - 16.0).abs() < 1e-12);
        let map = EdgeMap::build(&mesh).unwrap();
        assert_eq!(pinch_vertices(&mesh, &map).unwrap(), vec![12]);
        // A plain grid has no pinch.
        let g = grid(4);
        assert!(
            pinch_vertices(&g, &EdgeMap::build(&g).unwrap())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn three_holes_at_one_vertex_are_three_loops() {
        // Three wedges meet at vertex 0, so each hole's next edge there is
        // chosen by turning into the gap; key order alone would mix them.
        let mesh = tri_pinch();
        let loops = boundary_loops(&mesh).unwrap();
        let seqs: Vec<&[VertexId]> = loops.iter().map(|l| l.vertices.as_slice()).collect();
        assert_eq!(
            seqs,
            [
                &[0, 1, 6][..],
                &[0, 3, 2][..],
                &[0, 5, 4][..],
                &[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18][..],
            ]
        );
        assert!(loops.iter().all(|l| l.clean));
        let map = EdgeMap::build(&mesh).unwrap();
        assert_eq!(pinch_vertices(&mesh, &map).unwrap(), vec![0]);
    }

    #[test]
    fn fin_loops_unclean() {
        let mesh = fin();
        let loops = boundary_loops(&mesh).unwrap();
        // Each quad's boundary runs 0 -> p -> q -> 1 and would close over the
        // non-manifold edge (1, 0); vertex 1 has no outgoing boundary edge, so
        // each comes out as an open chain, unclean.
        let seqs: Vec<&[VertexId]> = loops.iter().map(|l| l.vertices.as_slice()).collect();
        assert_eq!(
            seqs,
            [&[0, 2, 3, 1][..], &[0, 4, 5, 1][..], &[0, 6, 7, 1][..]]
        );
        assert!(loops.iter().all(|l| !l.clean && !l.closed));
        assert_eq!(loops[0].id.hash, fnv1a(&[0, 2, 3, 1]));
        // The non-manifold edge does not make the fin's vertices pinches.
        let map = EdgeMap::build(&mesh).unwrap();
        assert!(pinch_vertices(&mesh, &map).unwrap().is_empty());
    }

    #[test]
    fn fin_chains_whole_whatever_the_numbering() {
        // The fin with the spine relabelled (1, 5): every chain's first edge
        // no longer sorts first, so walks must start at chain heads to keep
        // each chain whole, and the chains must keep their ends.
        let (old, _) = parts(&fin());
        // New id of each old vertex 0..8; new vertex 0 is unused.
        let new = [5, 1, 7, 3, 2, 4, 6, 8];
        let mut positions = vec![0.0f32; 27];
        for (o, &n) in new.iter().enumerate() {
            positions[3 * n..3 * n + 3].copy_from_slice(&old[3 * o..3 * o + 3]);
        }
        let tris = [
            [5, 7, 3],
            [5, 3, 1],
            [5, 2, 4],
            [5, 4, 1],
            [5, 6, 8],
            [5, 8, 1],
        ];
        let indices: Vec<u32> = tris.iter().flatten().copied().collect();
        let mesh = Mesh::from_buffers(&positions, &indices).unwrap();
        let loops = boundary_loops(&mesh).unwrap();
        let seqs: Vec<&[VertexId]> = loops.iter().map(|l| l.vertices.as_slice()).collect();
        assert_eq!(
            seqs,
            [&[5, 2, 4, 1][..], &[5, 6, 8, 1][..], &[5, 7, 3, 1][..]]
        );
        assert!(loops.iter().all(|l| !l.clean && !l.closed));
        assert!(loops.iter().all(|l| l.id.min_vertex == 1 && l.id.len == 4));
        assert_eq!(loops[0].id.hash, fnv1a(&[5, 2, 4, 1]));
    }

    #[test]
    fn three_holes_at_one_vertex_out_of_plane() {
        // The tri_pinch wedges curved into a bowl, twisted alternately up and
        // down (a monkey saddle), and folded 39 degrees along the y axis
        // through vertex 0; all well under 90 degrees.
        let bowl: fn(f64, f64) -> f64 = |r, _| 0.3 * r * r;
        let saddle: fn(f64, f64) -> f64 = |r, deg| 0.7 * r * (3.0 * deg.to_radians()).cos();
        let fold: fn(f64, f64) -> f64 = |r, deg| (0.8 * r * deg.to_radians().cos()).max(0.0);
        for lift in [bowl, saddle, fold] {
            let mesh = tri_pinch_lifted(lift);
            let loops = boundary_loops(&mesh).unwrap();
            let seqs: Vec<&[VertexId]> = loops.iter().map(|l| l.vertices.as_slice()).collect();
            assert_eq!(
                seqs,
                [
                    &[0, 1, 6][..],
                    &[0, 3, 2][..],
                    &[0, 5, 4][..],
                    &[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18][..],
                ]
            );
            assert!(loops.iter().all(|l| l.clean && l.closed));
        }
    }

    #[test]
    fn loop_id_is_rotation_invariant_and_stable() {
        let mesh = bowtie_holes();
        let before = boundary_loops(&mesh).unwrap();
        // Re-add the same vertices with the triangles in another order and each
        // triangle's corners rotated, so the walk starts elsewhere.
        let (positions, triangles) = parts(&mesh);
        let mut shuffled: Vec<u32> = Vec::new();
        for (i, t) in triangles.iter().enumerate().rev() {
            let r = i % 3;
            shuffled.extend([t[r], t[(r + 1) % 3], t[(r + 2) % 3]]);
        }
        let again = Mesh::from_buffers(&positions, &shuffled).unwrap();
        let after = boundary_loops(&again).unwrap();
        let ids = |ls: &[BoundaryLoop]| ls.iter().map(|l| l.id).collect::<Vec<_>>();
        assert_eq!(ids(&before), ids(&after));
        assert_eq!(before[1].vertices, after[1].vertices);
        // Different loops, different hashes.
        let mut hashes: Vec<u64> = before.iter().map(|l| l.id.hash).collect();
        hashes.sort_unstable();
        hashes.dedup();
        assert_eq!(hashes.len(), 3);
        // The hash is FNV-1a 64 over the rotated sequence.
        assert_eq!(before[1].id.hash, fnv1a(&[6, 11, 12, 7]));
        assert_ne!(fnv1a(&[6, 11, 12, 7]), fnv1a(&[6, 7, 12, 11]));
    }

    #[test]
    fn degenerate_triangles_ignored_by_maps() {
        for mesh in [bowtie_holes(), fin(), hemisphere(3, 8, 1.0)] {
            let (_, tris) = parts(&mesh);
            let extra: Vec<[u32; 3]> = tris
                .iter()
                .step_by(3)
                .flat_map(|&[a, b, c]| [[a, a, b], [b, c, c], [c, c, c]])
                .collect();
            let dirty = with_triangles(&mesh, &extra);
            assert_eq!(kinds(&mesh), kinds(&dirty));
            let (l0, l1) = (
                boundary_loops(&mesh).unwrap(),
                boundary_loops(&dirty).unwrap(),
            );
            assert_eq!(l0, l1);
            let m0 = EdgeMap::build(&mesh).unwrap();
            let m1 = EdgeMap::build(&dirty).unwrap();
            assert_eq!(
                pinch_vertices(&mesh, &m0).unwrap(),
                pinch_vertices(&dirty, &m1).unwrap()
            );
            let c1 = components(&dirty, &m1).unwrap();
            assert_eq!(components(&mesh, &m0).unwrap(), c1);
        }
    }

    #[test]
    fn fixtures_sanity() {
        // Möbius: one component, one boundary curve of 2n edges, walked across
        // the orientation seam.
        let m = mobius(12);
        let map = EdgeMap::build(&m).unwrap();
        assert_eq!(components(&m, &map).unwrap().len(), 1);
        let loops = boundary_loops(&m).unwrap();
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].vertices.len(), 24);
        assert!(loops[0].clean);
        // Two punched holes in a sphere give two loops in one component.
        let punched = punch_holes(
            &uv_sphere(16, 16, 1.0),
            &[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            0.3,
        );
        let map = EdgeMap::build(&punched).unwrap();
        assert_eq!(components(&punched, &map).unwrap().len(), 1);
        assert_eq!(boundary_loops(&punched).unwrap().len(), 2);
        // Flipping a triangle changes no edge kind and keeps one border loop.
        let g = grid(3);
        let flipped = flip(&g, &[4]);
        assert_eq!(kinds(&g), kinds(&flipped));
        let loops = boundary_loops(&flipped).unwrap();
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].vertices.len(), 12);
        // Crossed boxes: two closed components.
        let b = crossed_boxes();
        let comps = components(&b, &EdgeMap::build(&b).unwrap()).unwrap();
        assert_eq!(comps.len(), 2);
        assert!(comps.iter().all(|c| c.closed && c.triangles.len() == 12));
        assert!((comps[0].area - 24.0).abs() < 1e-9);
    }

    #[test]
    fn boundary_loops_oom_is_clean() {
        let mesh = bowtie_holes();
        let expected = boundary_loops(&mesh).unwrap();
        let mut n = 0;
        loop {
            fail_allocations_after(Some(n));
            let got = boundary_loops(&mesh);
            fail_allocations_after(None);
            match got {
                Ok(loops) => {
                    assert_eq!(loops, expected);
                    break;
                }
                Err(e) => assert_eq!(e, OpsError::Store(StoreError::OutOfMemory)),
            }
            n += 1;
        }
        assert!(n > 5, "only {n} fallible reservations");
    }
}
