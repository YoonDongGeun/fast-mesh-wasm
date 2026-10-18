//! Triangulation of 2D polygons with holes.
//!
//! [`triangulate`] is ear clipping in the manner of Mapbox's earcut: the
//! polygon is a doubly linked ring of vertices, each hole is bridged into the
//! outer ring by a pair of coincident edges, and ears are clipped until two
//! vertices remain. It is meant for section fills and hole filling, where the
//! input is usually simple but may not be, so it never panics and reports
//! whether it finished instead of failing.

use std::cmp::Ordering;

use crate::{QueryError, Result, allocation_allowed};

/// Triangulates the polygon `outer` minus the polygons in `holes`, appending
/// the triangles to `out`.
///
/// Vertices are numbered `outer` first, then each hole in order; a triangle
/// is three such numbers. Triangles are counter-clockwise (positive area in
/// a right-handed `(x, y)` frame) when the input is a simple polygon with
/// simple holes inside it.
///
/// - Either orientation is accepted: the outer ring is made counter-clockwise
///   and the holes clockwise internally.
/// - Consecutive equal points, including a last point equal to the first,
///   count as one. The repeats keep their numbers but never appear in a
///   triangle. Collinear vertices may also be left out of every triangle.
/// - A ring with fewer than three distinct points has nothing to fill: such
///   an outer gives no triangles and `Ok(true)`, and such a hole is ignored.
///   An outer whose points are all collinear also gives no triangles and
///   `Ok(true)`.
/// - Any non-finite coordinate (NaN or infinite), in the outer ring or a
///   hole, gives no triangles and `Ok(false)`.
///
/// Returns `Ok(false)` when ear clipping could not finish (typically a
/// self-intersecting ring) or a hole could not be bridged; the triangles found
/// so far are kept in `out`. It never panics.
///
/// # Errors
/// - [`QueryError::OutOfMemory`] when working memory or room for the
///   triangles could not be reserved; `out` is then unchanged.
/// - [`QueryError::InvalidInput`] when there are more points in all than a
///   `u32` can number.
pub fn triangulate(
    outer: &[[f64; 2]],
    holes: &[&[[f64; 2]]],
    out: &mut Vec<[u32; 3]>,
) -> Result<bool> {
    let total = holes
        .iter()
        .try_fold(outer.len(), |n, h| n.checked_add(h.len()))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or(QueryError::InvalidInput("too many polygon points"))?;
    let finite = |r: &[[f64; 2]]| r.iter().flatten().all(|c| c.is_finite());
    if !finite(outer) || !holes.iter().all(|h| finite(h)) {
        return Ok(false);
    }
    // Each bridge duplicates two vertices, and every triangle removes at
    // least one vertex from the ring until two remain.
    // Saturating, so that an impossible size fails to reserve instead.
    let capacity = total.saturating_add(holes.len().saturating_mul(2));
    let mut nodes = Vec::new();
    reserve(&mut nodes, capacity)?;
    let mut queue = Vec::new();
    reserve(&mut queue, holes.len())?;
    reserve(out, capacity.saturating_sub(2))?;

    let mut e = Earcut {
        nodes,
        out,
        ok: true,
    };
    let Some(mut start) = e.ring(outer, 0, true) else {
        return Ok(true);
    };
    if e.next(start) == e.prev(start) {
        return Ok(true);
    }
    let mut offset = outer.len();
    for hole in holes {
        if let Some(h) = e.ring(hole, offset, false) {
            // A hole of one or two distinct points has no area to remove.
            if e.next(h) != e.prev(h) {
                queue.push(e.leftmost(h));
            }
        }
        offset += hole.len();
    }
    // Bridge holes from left to right, so that each bridge sees the ones
    // bridged before it as part of the outer ring.
    queue.sort_by(|&a, &b| e.compare_xy_slope(a, b));
    for &h in &queue {
        start = e.eliminate_hole(h, start);
    }
    let finished = e.clip(e.live(start));
    Ok(finished && e.ok)
}

/// Reserves room for `n` more elements, honouring the test fault hook.
fn reserve<T>(v: &mut Vec<T>, n: usize) -> Result<()> {
    allocation_allowed()?;
    v.try_reserve(n).map_err(|_| QueryError::OutOfMemory)
}

/// A vertex of the ring: its input number, position and neighbours.
#[derive(Debug, Clone, Copy)]
struct Node {
    i: u32,
    x: f64,
    y: f64,
    prev: usize,
    next: usize,
    /// Unlinked from its ring. Its `prev` and `next` still name the
    /// neighbours it had when it was removed.
    removed: bool,
}

/// Twice the signed area of the turn `p → q → r`, negated: negative when it
/// turns counter-clockwise. The sign convention is earcut's, so that the
/// predicates below read as there.
fn area(p: &Node, q: &Node, r: &Node) -> f64 {
    (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
}

fn equals(a: &Node, b: &Node) -> bool {
    a.x == b.x && a.y == b.y
}

/// Whether `(px, py)` is inside or on the counter-clockwise triangle `abc`.
#[allow(clippy::too_many_arguments)]
fn point_in_triangle(
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    cx: f64,
    cy: f64,
    px: f64,
    py: f64,
) -> bool {
    (cx - px) * (ay - py) >= (ax - px) * (cy - py)
        && (ax - px) * (by - py) >= (bx - px) * (ay - py)
        && (bx - px) * (cy - py) >= (cx - px) * (by - py)
}

fn sign(v: f64) -> i8 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

/// Whether `q` lies in the bounding box of `p` and `r` (`q` collinear with them).
fn on_segment(p: &Node, q: &Node, r: &Node) -> bool {
    q.x <= p.x.max(r.x) && q.x >= p.x.min(r.x) && q.y <= p.y.max(r.y) && q.y >= p.y.min(r.y)
}

/// Whether the segments `p1 q1` and `p2 q2` intersect, touching included.
fn intersects(p1: &Node, q1: &Node, p2: &Node, q2: &Node) -> bool {
    let o1 = sign(area(p1, q1, p2));
    let o2 = sign(area(p1, q1, q2));
    let o3 = sign(area(p2, q2, p1));
    let o4 = sign(area(p2, q2, q1));
    (o1 != o2 && o3 != o4)
        || (o1 == 0 && on_segment(p1, p2, q1))
        || (o2 == 0 && on_segment(p1, q2, q1))
        || (o3 == 0 && on_segment(p2, p1, q2))
        || (o4 == 0 && on_segment(p2, q1, q2))
}

/// The ring under construction and the triangles found so far.
///
/// Removed nodes stay in `nodes`, unlinked. Every loop below either removes
/// a node or walks once around the ring, so all terminate, whatever the
/// coordinates.
struct Earcut<'a> {
    nodes: Vec<Node>,
    out: &'a mut Vec<[u32; 3]>,
    /// False once a hole could not be bridged.
    ok: bool,
}

impl Earcut<'_> {
    fn n(&self, k: usize) -> &Node {
        &self.nodes[k]
    }

    fn next(&self, k: usize) -> usize {
        self.nodes[k].next
    }

    fn prev(&self, k: usize) -> usize {
        self.nodes[k].prev
    }

    /// `area` of the turn at three nodes.
    fn turn(&self, p: usize, q: usize, r: usize) -> f64 {
        area(self.n(p), self.n(q), self.n(r))
    }

    fn same(&self, a: usize, b: usize) -> bool {
        equals(self.n(a), self.n(b))
    }

    /// Adds a node after `last` (or a ring of one when there is none).
    fn insert(&mut self, i: u32, p: [f64; 2], last: Option<usize>) -> usize {
        let k = self.nodes.len();
        // Adding zero turns -0.0 into 0.0, so that sorting treats them alike.
        let (x, y) = (p[0] + 0.0, p[1] + 0.0);
        let (prev, next) = match last {
            Some(l) => (l, self.next(l)),
            None => (k, k),
        };
        // Capacity was reserved for every node up front.
        self.nodes.push(Node {
            i,
            x,
            y,
            prev,
            next,
            removed: false,
        });
        if let Some(l) = last {
            self.nodes[next].prev = k;
            self.nodes[l].next = k;
        }
        k
    }

    fn remove(&mut self, k: usize) {
        let Node { prev, next, .. } = self.nodes[k];
        self.nodes[next].prev = prev;
        self.nodes[prev].next = next;
        self.nodes[k].removed = true;
    }

    /// `k`, or the first node still in a ring after it when `k` was removed.
    ///
    /// A node handed back by one pass may be removed by the next (filtering
    /// can cascade backwards past it); walking a ring from a removed node
    /// would never come back to it. Following `next` from a removed node
    /// always reaches a node removed later or not at all, so this ends.
    fn live(&self, mut k: usize) -> usize {
        for _ in 0..self.nodes.len() {
            if !self.nodes[k].removed {
                break;
            }
            k = self.next(k);
        }
        k
    }

    /// Links `points` (numbered from `offset`) into a ring, counter-clockwise
    /// when `ccw` and clockwise otherwise, skipping consecutive repeats.
    /// Returns its last node, or `None` for no points.
    fn ring(&mut self, points: &[[f64; 2]], offset: usize, ccw: bool) -> Option<usize> {
        let n = points.len();
        // Twice the shoelace area: positive when counter-clockwise.
        let twice_area: f64 = (0..n)
            .map(|j| {
                let (p, q) = (points[j], points[(j + 1) % n]);
                p[0] * q[1] - q[0] * p[1]
            })
            .sum();
        let forward = (twice_area > 0.0) == ccw;
        let mut last: Option<usize> = None;
        for j in 0..n {
            let j = if forward { j } else { n - 1 - j };
            let p = points[j];
            if last.is_some_and(|l| self.n(l).x == p[0] && self.n(l).y == p[1]) {
                continue;
            }
            // `j + offset` is below the total, which fits a `u32`.
            last = Some(self.insert((offset + j) as u32, p, last));
        }
        let l = last?;
        if l != self.next(l) && self.same(l, self.next(l)) {
            let first = self.next(l);
            self.remove(l);
            return Some(first);
        }
        Some(l)
    }

    /// The leftmost node of the ring through `start`, lowest on ties.
    fn leftmost(&self, start: usize) -> usize {
        let (mut p, mut best) = (start, start);
        loop {
            let (a, b) = (self.n(p), self.n(best));
            if a.x < b.x || (a.x == b.x && a.y < b.y) {
                best = p;
            }
            p = self.next(p);
            if p == start {
                return best;
            }
        }
    }

    /// Orders hole start nodes by `x`, then `y`, then the slope of their
    /// first edge. A total order, so that sorting cannot panic.
    fn compare_xy_slope(&self, a: usize, b: usize) -> Ordering {
        let slope = |k: usize| {
            let (p, q) = (self.n(k), self.n(self.next(k)));
            (q.y - p.y) / (q.x - p.x)
        };
        let (pa, pb) = (self.n(a), self.n(b));
        pa.x.total_cmp(&pb.x)
            .then(pa.y.total_cmp(&pb.y))
            .then_with(|| slope(a).total_cmp(&slope(b)))
    }

    /// Removes repeated and collinear vertices from `start` around to `end`
    /// (the whole ring when `end` is `None`), returning a node still in it.
    fn filter_points(&mut self, start: usize, end: Option<usize>) -> usize {
        let start = self.live(start);
        let mut end = end.map_or(start, |e| self.live(e));
        let mut p = start;
        loop {
            let (prev, next) = (self.prev(p), self.next(p));
            let again = if self.same(p, next) || self.turn(prev, p, next) == 0.0 {
                self.remove(p);
                p = prev;
                end = prev;
                if p == self.next(p) {
                    break;
                }
                true
            } else {
                p = next;
                false
            };
            if !again && p == end {
                break;
            }
        }
        end
    }

    /// Whether the vertex `ear` and its neighbours form a triangle that can
    /// be cut off: convex, with no reflex vertex of the ring inside it.
    fn is_ear(&self, ear: usize) -> bool {
        let (a, b, c) = (self.prev(ear), ear, self.next(ear));
        if self.turn(a, b, c) >= 0.0 {
            return false;
        }
        let (na, nb, nc) = (self.n(a), self.n(b), self.n(c));
        let (x0, x1) = (na.x.min(nb.x).min(nc.x), na.x.max(nb.x).max(nc.x));
        let (y0, y1) = (na.y.min(nb.y).min(nc.y), na.y.max(nb.y).max(nc.y));
        let mut p = self.next(c);
        while p != a {
            let np = self.n(p);
            if np.x >= x0
                && np.x <= x1
                && np.y >= y0
                && np.y <= y1
                && !equals(na, np)
                && point_in_triangle(na.x, na.y, nb.x, nb.y, nc.x, nc.y, np.x, np.y)
                && self.turn(self.prev(p), p, self.next(p)) >= 0.0
            {
                return false;
            }
            p = self.next(p);
        }
        true
    }

    fn emit(&mut self, a: usize, b: usize, c: usize) {
        // Room for every triangle was reserved up front.
        self.out.push([self.n(a).i, self.n(b).i, self.n(c).i]);
    }

    /// Clips ears from the ring through `ear` until two vertices remain.
    /// When a full turn finds no ear it first drops repeated and collinear
    /// vertices, then cures local self-intersections, then gives up and
    /// returns false.
    fn clip(&mut self, mut ear: usize) -> bool {
        let mut pass = 0;
        let mut stop = ear;
        while self.prev(ear) != self.next(ear) {
            let (prev, next) = (self.prev(ear), self.next(ear));
            if self.is_ear(ear) {
                self.emit(prev, ear, next);
                self.remove(ear);
                ear = self.next(next);
                stop = ear;
                continue;
            }
            ear = next;
            if ear == stop {
                ear = match pass {
                    0 => self.filter_points(ear, None),
                    1 => {
                        let start = self.filter_points(ear, None);
                        self.cure_local_intersections(start)
                    }
                    _ => return false,
                };
                pass += 1;
                stop = ear;
            }
        }
        true
    }

    /// Cuts off a triangle wherever the edges before and after a vertex pair
    /// cross (`a p` and `p.next b`), which removes that local twist.
    fn cure_local_intersections(&mut self, start: usize) -> usize {
        let (mut start, mut p) = (start, start);
        loop {
            let a = self.prev(p);
            let pn = self.next(p);
            let b = self.next(pn);
            // Needs four distinct vertices `a p pn b`.
            if b != a
                && b != p
                && !self.same(a, b)
                && intersects(self.n(a), self.n(p), self.n(pn), self.n(b))
                && self.locally_inside(a, b)
                && self.locally_inside(b, a)
            {
                self.emit(a, p, b);
                self.remove(p);
                self.remove(pn);
                p = b;
                start = b;
            }
            p = self.next(p);
            if p == start {
                break;
            }
        }
        self.filter_points(p, None)
    }

    /// Whether the diagonal from `a` to `b` starts into the polygon's
    /// interior at `a`.
    fn locally_inside(&self, a: usize, b: usize) -> bool {
        let (ap, an) = (self.prev(a), self.next(a));
        if self.turn(ap, a, an) < 0.0 {
            self.turn(a, b, an) >= 0.0 && self.turn(a, ap, b) >= 0.0
        } else {
            self.turn(a, b, ap) < 0.0 || self.turn(a, an, b) < 0.0
        }
    }

    /// Whether the sector of `m` contains the sector of `p` (for two
    /// coincident bridge candidates).
    fn sector_contains_sector(&self, m: usize, p: usize) -> bool {
        self.turn(self.prev(m), m, self.prev(p)) < 0.0
            && self.turn(self.next(p), m, self.prev(m)) < 0.0
    }

    /// Joins the hole ring at `hole` (its leftmost node) into the ring
    /// through `outer` and returns a node of the joined ring. A hole with no
    /// visible outer vertex is left out and recorded in `ok`.
    fn eliminate_hole(&mut self, hole: usize, outer: usize) -> usize {
        let outer = self.live(outer);
        let Some(bridge) = self.find_hole_bridge(hole, outer) else {
            self.ok = false;
            return outer;
        };
        let reverse = self.split(bridge, hole);
        let after = self.next(reverse);
        self.filter_points(reverse, Some(after));
        let bridge = self.live(bridge);
        let after = self.next(bridge);
        self.filter_points(bridge, Some(after))
    }

    /// The outer vertex to bridge `hole` to: cast a ray left from the hole's
    /// leftmost point, take the nearest edge it hits, then the visible vertex
    /// in the triangle it spans with the smallest angle to the ray.
    fn find_hole_bridge(&self, hole: usize, outer: usize) -> Option<usize> {
        let (hx, hy) = (self.n(hole).x, self.n(hole).y);
        let mut qx = f64::NEG_INFINITY;
        let mut m = None;
        let mut p = outer;
        if self.same(hole, p) {
            return Some(p);
        }
        loop {
            let pn = self.next(p);
            if self.same(hole, pn) {
                return Some(pn);
            }
            let (a, b) = (self.n(p), self.n(pn));
            if hy <= a.y && hy >= b.y && b.y != a.y {
                let x = a.x + (hy - a.y) * (b.x - a.x) / (b.y - a.y);
                if x <= hx && x > qx {
                    qx = x;
                    let near = if a.x < b.x { p } else { pn };
                    m = Some(near);
                    if x == hx {
                        return m;
                    }
                }
            }
            p = pn;
            if p == outer {
                break;
            }
        }
        let mut m = m?;
        let stop = m;
        let (mx, my) = (self.n(m).x, self.n(m).y);
        let mut tan_min = f64::INFINITY;
        p = m;
        loop {
            let np = self.n(p);
            let (lx, rx) = if hy < my { (hx, qx) } else { (qx, hx) };
            if hx >= np.x
                && np.x >= mx
                && hx != np.x
                && point_in_triangle(lx, hy, mx, my, rx, hy, np.x, np.y)
            {
                let tan = (hy - np.y).abs() / (hx - np.x);
                let nm = self.n(m);
                if self.locally_inside(p, hole)
                    && (tan < tan_min
                        || (tan == tan_min
                            && (np.x > nm.x
                                || (np.x == nm.x && self.sector_contains_sector(m, p)))))
                {
                    m = p;
                    tan_min = tan;
                }
            }
            p = self.next(p);
            if p == stop {
                break;
            }
        }
        Some(m)
    }

    /// Links `a` to `b` with a pair of coincident edges, splitting the ring
    /// in two if they are on one ring or joining two rings into one. Returns
    /// the copy of `b`.
    fn split(&mut self, a: usize, b: usize) -> usize {
        let (an, bp) = (self.next(a), self.prev(b));
        let (na, nb) = (*self.n(a), *self.n(b));
        let a2 = self.nodes.len();
        let b2 = a2 + 1;
        // Capacity for two copies per hole was reserved up front.
        self.nodes.push(Node {
            next: an,
            prev: b2,
            ..na
        });
        self.nodes.push(Node {
            next: a2,
            prev: bp,
            ..nb
        });
        self.nodes[a].next = b;
        self.nodes[b].prev = a;
        self.nodes[an].prev = a2;
        self.nodes[bp].next = b2;
        b2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QueryError, fail_allocations_after};
    use proptest::prelude::*;

    /// The points of `outer` then of each hole, as numbered by [`triangulate`].
    fn all_points(outer: &[[f64; 2]], holes: &[&[[f64; 2]]]) -> Vec<[f64; 2]> {
        let mut pts = outer.to_vec();
        for h in holes {
            pts.extend_from_slice(h);
        }
        pts
    }

    fn signed_area(pts: &[[f64; 2]], t: [u32; 3]) -> f64 {
        let [a, b, c] = t.map(|i| pts[i as usize]);
        0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    }

    fn shoelace(ring: &[[f64; 2]]) -> f64 {
        let n = ring.len();
        (0..n)
            .map(|i| {
                let (p, q) = (ring[i], ring[(i + 1) % n]);
                p[0] * q[1] - q[0] * p[1]
            })
            .sum::<f64>()
            * 0.5
    }

    /// Triangulates, checks that it finished with counter-clockwise triangles
    /// and returns them with their total area.
    fn fill(outer: &[[f64; 2]], holes: &[&[[f64; 2]]]) -> (Vec<[u32; 3]>, f64) {
        let mut out = Vec::new();
        assert_eq!(triangulate(outer, holes, &mut out), Ok(true));
        let pts = all_points(outer, holes);
        let mut area = 0.0;
        for &t in &out {
            let a = signed_area(&pts, t);
            assert!(a > 0.0, "triangle {t:?} has area {a}");
            area += a;
        }
        (out, area)
    }

    fn square(x: f64, y: f64, s: f64) -> Vec<[f64; 2]> {
        vec![[x, y], [x + s, y], [x + s, y + s], [x, y + s]]
    }

    fn inside(ring: &[[f64; 2]], p: [f64; 2]) -> bool {
        let mut c = false;
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            if (a[1] > p[1]) != (b[1] > p[1])
                && p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            {
                c = !c;
            }
        }
        c
    }

    #[test]
    fn square_gives_two_triangles() {
        let (tris, area) = fill(&square(0.0, 0.0, 1.0), &[]);
        assert_eq!(tris.len(), 2);
        assert!((area - 1.0).abs() < 1e-12);
    }

    #[test]
    fn concave_l_shape_area_exact() {
        let l = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let (tris, area) = fill(&l, &[]);
        assert_eq!(tris.len(), 4);
        assert!((area - 3.0).abs() < 1e-12, "area {area}");
    }

    #[test]
    fn square_with_square_hole() {
        let hole = square(1.0, 1.0, 2.0);
        let (_, area) = fill(&square(0.0, 0.0, 4.0), &[&hole]);
        assert!((area - 12.0).abs() < 1e-12, "area {area}");
    }

    #[test]
    fn two_holes() {
        let outer = square(0.0, 0.0, 10.0);
        let h1 = square(1.0, 1.0, 3.0);
        let h2 = square(6.0, 5.0, 2.0);
        let (tris, area) = fill(&outer, &[&h1, &h2]);
        assert!((area - (100.0 - 9.0 - 4.0)).abs() < 1e-12, "area {area}");
        let pts = all_points(&outer, &[&h1, &h2]);
        for t in tris {
            let [a, b, c] = t.map(|i| pts[i as usize]);
            let g = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0];
            assert!(
                !inside(&h1, g) && !inside(&h2, g),
                "centroid {g:?} in a hole"
            );
        }
    }

    #[test]
    fn duplicate_consecutive_points_ignored() {
        // Duplicates in the middle and across the wrap (last equals first).
        let outer = [
            [0.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [2.0, 2.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ];
        let hole = [[0.5, 0.5], [0.5, 1.0], [0.5, 1.0], [1.0, 1.0], [1.0, 0.5]];
        let (tris, area) = fill(&outer, &[&hole]);
        assert!((area - 3.75).abs() < 1e-12, "area {area}");
        // The duplicates: outer 1, 4, 5, 7 (equal to 0) and hole 2 (8 + 2).
        for t in tris {
            for i in t {
                assert!(![1, 4, 5, 7, 10].contains(&i), "duplicate {i} in {t:?}");
            }
        }
    }

    #[test]
    fn collinear_points_ok() {
        let outer = [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [3.0, 2.0],
            [1.5, 2.0],
            [0.0, 2.0],
            [0.0, 1.0],
        ];
        let (_, area) = fill(&outer, &[]);
        assert!((area - 6.0).abs() < 1e-12, "area {area}");
    }

    #[test]
    fn cw_input_is_normalised() {
        let mut outer = square(0.0, 0.0, 4.0);
        outer.reverse();
        // A counter-clockwise hole, also to be normalised.
        let hole = square(1.0, 1.0, 1.0);
        let (_, area) = fill(&outer, &[&hole]);
        assert!((area - 15.0).abs() < 1e-12, "area {area}");
    }

    #[test]
    fn star_polygon_100_vertices_area() {
        let star: Vec<[f64; 2]> = (0..100)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 100.0;
                let r = if i % 2 == 0 { 10.0 } else { 4.0 };
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        let expected = shoelace(&star);
        let (tris, area) = fill(&star, &[]);
        assert_eq!(tris.len(), 98);
        assert!(
            ((area - expected) / expected).abs() < 1e-9,
            "{area} vs {expected}"
        );
    }

    #[test]
    fn degenerate_outer_has_nothing_to_fill() {
        let mut out = vec![[9, 9, 9]];
        assert_eq!(triangulate(&[], &[], &mut out), Ok(true));
        assert_eq!(triangulate(&[[1.0, 1.0]; 5], &[], &mut out), Ok(true));
        assert_eq!(
            triangulate(&[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], &[], &mut out),
            Ok(true)
        );
        // Collinear: zero area, nothing to fill.
        let line = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 0.0]];
        assert_eq!(triangulate(&line, &[], &mut out), Ok(true));
        assert_eq!(out, vec![[9, 9, 9]]);
    }

    #[test]
    fn degenerate_holes_are_ignored() {
        let outer = square(0.0, 0.0, 2.0);
        let (_, area) = fill(&outer, &[&[], &[[1.0, 1.0]], &[[0.5, 0.5], [1.0, 1.0]]]);
        assert!((area - 4.0).abs() < 1e-12, "area {area}");
    }

    #[test]
    fn non_finite_input_returns_false() {
        let mut out = vec![[9, 9, 9]];
        let mut bad = square(0.0, 0.0, 1.0);
        bad[2][0] = f64::NAN;
        assert_eq!(triangulate(&bad, &[], &mut out), Ok(false));
        bad[2][0] = f64::INFINITY;
        assert_eq!(triangulate(&bad, &[], &mut out), Ok(false));
        let hole = [[0.2, 0.2], [0.2, f64::NEG_INFINITY], [0.5, 0.5]];
        assert_eq!(
            triangulate(&square(0.0, 0.0, 1.0), &[&hole], &mut out),
            Ok(false)
        );
        assert_eq!(out, vec![[9, 9, 9]]);
    }

    #[test]
    fn appends_after_existing_triangles() {
        let mut out = vec![[9, 9, 9]];
        assert_eq!(triangulate(&square(0.0, 0.0, 1.0), &[], &mut out), Ok(true));
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], [9, 9, 9]);
    }

    #[test]
    fn out_of_memory_leaves_out_unchanged() {
        let outer = square(0.0, 0.0, 4.0);
        let hole = square(1.0, 1.0, 2.0);
        // Every fallible reservation the call makes, failing in turn.
        for n in 0..3 {
            let mut out = vec![[9, 9, 9]];
            fail_allocations_after(Some(n));
            let r = triangulate(&outer, &[&hole], &mut out);
            fail_allocations_after(None);
            assert_eq!(r, Err(QueryError::OutOfMemory), "reservation {n}");
            assert_eq!(out, vec![[9, 9, 9]]);
        }
        let mut out = Vec::new();
        fail_allocations_after(Some(3));
        let r = triangulate(&outer, &[&hole], &mut out);
        fail_allocations_after(None);
        assert_eq!(r, Ok(true));
    }

    /// Proptest cases: `default`, or `PROPTEST_CASES` when set.
    fn cases(default: u32) -> u32 {
        std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }

    fn ring(max: usize) -> impl Strategy<Value = Vec<[f64; 2]>> {
        // A small grid, so that duplicates, collinear runs and crossings are common.
        prop::collection::vec(
            (-4i32..=4, -4i32..=4).prop_map(|(x, y)| [f64::from(x), f64::from(y)]),
            0..=max,
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(cases(512)))]

        #[test]
        fn proptest_never_panics(
            outer in ring(64),
            holes in prop::collection::vec(ring(16), 0..4),
            scale in prop::sample::select(vec![1.0, 1e-3, 1e150]),
        ) {
            let outer: Vec<[f64; 2]> = outer.iter().map(|p| p.map(|c| c * scale)).collect();
            let holes: Vec<Vec<[f64; 2]>> = holes
                .iter()
                .map(|h| h.iter().map(|p| p.map(|c| c * scale)).collect())
                .collect();
            let hole_refs: Vec<&[[f64; 2]]> = holes.iter().map(Vec::as_slice).collect();
            let total = outer.len() + holes.iter().map(Vec::len).sum::<usize>();
            let mut out = Vec::new();
            prop_assert!(triangulate(&outer, &hole_refs, &mut out).is_ok());
            for t in out {
                prop_assert!(t.iter().all(|&i| (i as usize) < total), "{:?}", t);
            }
        }

        #[test]
        fn convex_polygons_fill_exactly(n in 3usize..40, r in 0.1f64..100.0) {
            let poly: Vec<[f64; 2]> = (0..n)
                .map(|i| {
                    let a = i as f64 * std::f64::consts::TAU / n as f64;
                    [r * a.cos(), r * a.sin()]
                })
                .collect();
            let mut out = Vec::new();
            prop_assert_eq!(triangulate(&poly, &[], &mut out), Ok(true));
            let area: f64 = out.iter().map(|&t| signed_area(&poly, t)).sum();
            let expected = shoelace(&poly);
            prop_assert!(((area - expected) / expected).abs() < 1e-9);
        }

        /// A random star-shaped (so simple) polygon around a square hole
        /// well inside it: the fill finishes with the exact area.
        #[test]
        fn star_shaped_with_hole_fills_exactly(
            radii in prop::collection::vec(2.0f64..10.0, 8..80),
            hole_size in 0.1f64..1.0,
        ) {
            let n = radii.len();
            let outer: Vec<[f64; 2]> = radii
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let a = i as f64 * std::f64::consts::TAU / n as f64;
                    [r * a.cos(), r * a.sin()]
                })
                .collect();
            let h = hole_size / 2.0;
            let hole = square(-h, -h, hole_size);
            // With at least 8 spokes of radius 2 to 10 every edge stays more
            // than 1.6 from the centre, clear of the hole's corners (< 0.71).
            let mut out = Vec::new();
            prop_assert_eq!(triangulate(&outer, &[&hole], &mut out), Ok(true));
            let pts = all_points(&outer, &[&hole]);
            let mut area = 0.0;
            for &t in &out {
                let a = signed_area(&pts, t);
                prop_assert!(a > 0.0);
                area += a;
            }
            let expected = shoelace(&outer) - hole_size * hole_size;
            prop_assert!(((area - expected) / expected).abs() < 1e-9, "{} vs {}", area, expected);
        }
    }
}
