//! Measurements on section contours, in the plane's `(u, v)` frame: length,
//! signed area, the filled cross-section and the in-plane gap between two
//! sets of contours.
//!
//! Everything here is pure 2D on polylines of `(u, v)` points, as made by
//! [`Plane::to_uv`](crate::Plane::to_uv) from the points of a [`slice()`](crate::slice()).

use crate::triangulate::triangulate;
use crate::tritri::{orient2d_approx, side2};
use crate::{QueryError, Result, allocation_allowed};

type P2 = [f64; 2];

/// A segment between two points.
type Seg = [P2; 2];

/// The filled cross-section made by [`fill`]: triangles over `(u, v)` points.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Fill {
    /// The points of every filled contour, in `(u, v)`: each outer contour
    /// followed by its holes, in input order.
    pub points: Vec<[f64; 2]>,
    /// Triangles numbering `points`, counter-clockwise for simple contours.
    pub triangles: Vec<[u32; 3]>,
}

/// The closest pair between two sets of polylines, found by [`contour_gap`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gap2 {
    /// The distance between `a` and `b`; 0 when the polylines cross or touch.
    pub distance: f64,
    /// The point on the first set.
    pub a: [f64; 2],
    /// The point on the second set.
    pub b: [f64; 2],
}

/// The length of a polyline: the sum of its segment lengths. A closed one
/// also counts the segment from its last point back to its first.
pub fn contour_length(points: &[[f64; 2]], closed: bool) -> f64 {
    let open: f64 = points.windows(2).map(|w| dist(w[0], w[1])).sum();
    match points {
        [first, .., last] if closed => open + dist(*last, *first),
        _ => open,
    }
}

/// The signed area of the closed polygon `points`, by the shoelace formula:
/// positive when it runs counter-clockwise. Fewer than three points have no
/// area.
pub fn signed_area(points: &[[f64; 2]]) -> f64 {
    let Some(&o) = points.first() else {
        return 0.0;
    };
    // A fan from the first point, which keeps the products small when the
    // polygon is far from the origin.
    let twice: f64 = points
        .windows(2)
        .skip(1)
        .map(|w| cross(sub(w[0], o), sub(w[1], o)))
        .sum();
    0.5 * twice
}

/// Fills the closed contours `contours` (each with whether it is oriented)
/// and writes the triangles to `out`, replacing what it held.
///
/// Contours are nested into outers and holes:
/// - When every contour is oriented, counter-clockwise contours are outers,
///   and each clockwise contour is a hole of the smallest outer containing
///   it. A clockwise contour inside no outer is not filled. When the summed
///   signed area is negative (a closed mesh wound inward), the orientation is
///   taken as reversed throughout: clockwise contours are the outers.
/// - When any contour is unoriented, winding is ignored: a contour inside an
///   even number of others is an outer, and one inside an odd number is a
///   hole of the smallest such outer containing it.
///
/// Containment tests one point of the inner contour with the even-odd ray
/// rule. Contours with zero area are not filled. Each outer and its holes are
/// triangulated by [`triangulate`]; `out.points` holds their points, each
/// outer followed by its holes.
///
/// Returns whether the fill is good (`fill_ok`): `false` when triangulation
/// could not finish, or when the fill's signed area differs from the
/// contours' summed area (signed when oriented, with the sign flipped when
/// reversed; by nesting depth otherwise)
/// by more than `1e-6 · max(1, Σ |area|)`. This is best effort: there is no
/// separate self-intersection test, so a bad fill can go unreported.
///
/// # Errors
/// - [`QueryError::OutOfMemory`] when memory could not be reserved; `out` is
///   then empty.
/// - [`QueryError::InvalidInput`] when there are more points in all than a
///   `u32` can number.
pub fn fill(contours: &[(&[[f64; 2]], bool)], out: &mut Fill) -> Result<bool> {
    out.points.clear();
    out.triangles.clear();
    let r = fill_into(contours, out);
    if r.is_err() {
        out.points.clear();
        out.triangles.clear();
    }
    r
}

/// What a contour is in the fill.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Role {
    /// Not filled.
    Skip,
    Outer,
    /// A hole of the outer with this index.
    Hole(usize),
}

/// A contour's area and bounding box.
struct Info {
    area: f64,
    lo: P2,
    hi: P2,
}

fn fill_into(contours: &[(&[[f64; 2]], bool)], out: &mut Fill) -> Result<bool> {
    let total = contours
        .iter()
        .try_fold(0usize, |n, c| n.checked_add(c.0.len()))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or(QueryError::InvalidInput("too many contour points"))?;
    let n = contours.len();
    let mut info = Vec::new();
    reserve(&mut info, n)?;
    for &(pts, _) in contours {
        let (lo, hi) = bounds(pts);
        info.push(Info {
            area: signed_area(pts),
            lo,
            hi,
        });
    }
    // Whether contour `j` contains a point of contour `i`.
    let holds = |j: usize, i: usize| {
        let c = &info[j];
        j != i
            && c.area != 0.0
            && contours[i]
                .0
                .first()
                .is_some_and(|&p| in_box(p, c.lo, c.hi) && contains(contours[j].0, p))
    };
    // The smallest contour that is `allowed` and contains contour `i`.
    let container = |i: usize, allowed: &dyn Fn(usize) -> bool| {
        (0..n)
            .filter(|&j| allowed(j) && holds(j, i))
            .min_by(|&a, &b| info[a].area.abs().total_cmp(&info[b].area.abs()))
    };

    let mut roles = Vec::new();
    reserve(&mut roles, n)?;
    let mut expected = 0.0;
    let mut scale = 0.0;
    if contours.iter().all(|c| c.1) {
        // A closed mesh wound inward gives clockwise outers and
        // counter-clockwise holes: a negative total means the orientation is
        // reversed throughout.
        let sign = if info.iter().map(|c| c.area).sum::<f64>() < 0.0 {
            -1.0
        } else {
            1.0
        };
        for (i, c) in info.iter().enumerate() {
            let area = sign * c.area;
            roles.push(if area > 0.0 {
                Role::Outer
            } else if area < 0.0 {
                container(i, &|j| sign * info[j].area > 0.0).map_or(Role::Skip, Role::Hole)
            } else {
                Role::Skip
            });
            expected += area;
            scale += area.abs();
        }
    } else {
        let mut depth = Vec::new();
        reserve(&mut depth, n)?;
        for i in 0..n {
            depth.push((0..n).filter(|&j| holds(j, i)).count());
        }
        for (i, c) in info.iter().enumerate() {
            let a = c.area.abs();
            roles.push(if c.area == 0.0 {
                Role::Skip
            } else if depth[i] % 2 == 0 {
                expected += a;
                Role::Outer
            } else {
                expected -= a;
                container(i, &|j| depth[j] % 2 == 0).map_or(Role::Skip, Role::Hole)
            });
            scale += a;
        }
    }

    reserve(&mut out.points, total)?;
    let mut holes: Vec<&[[f64; 2]]> = Vec::new();
    reserve(&mut holes, n)?;
    let mut ok = true;
    for (o, &(outer, _)) in contours.iter().enumerate() {
        if roles[o] != Role::Outer {
            continue;
        }
        holes.clear();
        holes.extend(
            (0..n)
                .filter(|&i| roles[i] == Role::Hole(o))
                .map(|i| contours[i].0),
        );
        // Fits: every point counted in `total` is at most `u32::MAX`.
        let base = out.points.len() as u32;
        out.points.extend_from_slice(outer);
        for h in &holes {
            out.points.extend_from_slice(h);
        }
        let start = out.triangles.len();
        ok &= triangulate(outer, &holes, &mut out.triangles)?;
        for t in &mut out.triangles[start..] {
            *t = t.map(|k| k + base);
        }
    }

    let area: f64 = out
        .triangles
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|k| out.points[k as usize]);
            0.5 * cross(sub(b, a), sub(c, a))
        })
        .sum();
    // Written so that NaN fails.
    let matches = (area - expected).abs() <= 1e-6 * f64::max(1.0, scale);
    Ok(ok && matches)
}

/// Whether `p` is inside the polygon `poly`, by the even-odd ray rule.
fn contains(poly: &[P2], p: P2) -> bool {
    let mut inside = false;
    let mut j = poly.len().wrapping_sub(1);
    for (i, &a) in poly.iter().enumerate() {
        let b = poly[j];
        if (a[1] > p[1]) != (b[1] > p[1])
            && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// The smallest box containing `pts`; inverted (`lo > hi`) when empty.
fn bounds(pts: &[P2]) -> (P2, P2) {
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in pts {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

fn in_box(p: P2, lo: P2, hi: P2) -> bool {
    (lo[0]..=hi[0]).contains(&p[0]) && (lo[1]..=hi[1]).contains(&p[1])
}

/// The minimum distance between the segments of the polylines `a` and those
/// of `b`, each with whether it is closed, with the closest pair of points.
///
/// An open polyline of `n` points has `n − 1` segments, and a closed one also
/// joins its last point to its first. When segments of the two sides cross
/// or touch, the distance is 0 and both points are the same point where
/// they meet. Returns `None` when either side has no segments.
///
/// The segments of `b` are binned in a uniform grid with about one cell per
/// segment, and each segment of `a` searches outward through it ring by ring
/// until no unvisited cell can be closer. The segments of `a` are taken
/// nearest `b`'s bounding box first, and the search stops at the first one
/// whose box is at least the best distance from it. The result is the exact
/// minimum of the pairwise distances, not an approximation.
///
/// Known limits: a segment of `a` starts from every cell its box overlaps,
/// so a very long one still visits many cells on its first ring (mesh
/// sections have short segments). Coordinates beyond about `1e154` overflow
/// the squared distances.
///
/// # Errors
/// - [`QueryError::InvalidInput`] when a point is not finite.
/// - [`QueryError::OutOfMemory`] when working memory could not be reserved.
pub fn contour_gap(a: &[(&[[f64; 2]], bool)], b: &[(&[[f64; 2]], bool)]) -> Result<Option<Gap2>> {
    let mut sa = Vec::new();
    collect_segments(a, &mut sa)?;
    let mut sb = Vec::new();
    collect_segments(b, &mut sb)?;
    if sa.is_empty() || sb.is_empty() {
        return Ok(None);
    }
    let grid = Grid::build(&sb)?;
    // The segments of `a` nearest `b`'s box first: the first searches find a
    // close pair, and the rest stop as soon as their box gap, a lower bound
    // on their distance to every segment of `b`, reaches it. Without this the
    // cost depended on where contour `a` happened to start.
    let to_b = |s: &Seg| {
        let (l, h) = bounds(s);
        box_gap(l, h, grid.lo, grid.hi)
    };
    sa.sort_unstable_by(|p, q| to_b(p).total_cmp(&to_b(q)));
    let mut seen = Vec::new();
    reserve(&mut seen, sb.len())?;
    seen.resize(sb.len(), 0usize);
    let mut best = segment_gap(sa[0], sb[0]);
    for (k, &s) in sa.iter().enumerate() {
        if best.0 == 0.0 || to_b(&s) >= best.0 {
            break;
        }
        grid.search(s, &sb, &mut seen, k + 1, &mut best);
    }
    Ok(Some(Gap2 {
        distance: best.0.sqrt(),
        a: best.1,
        b: best.2,
    }))
}

/// Appends the segments of `lines` to `out`.
fn collect_segments(lines: &[(&[[f64; 2]], bool)], out: &mut Vec<Seg>) -> Result<()> {
    for &(pts, closed) in lines {
        if !pts.iter().flatten().all(|c| c.is_finite()) {
            return Err(QueryError::InvalidInput("non-finite contour point"));
        }
        let closing = match pts {
            [first, .., last] if closed => Some([*last, *first]),
            _ => None,
        };
        let count = pts.len().saturating_sub(1) + usize::from(closing.is_some());
        reserve(out, count)?;
        out.extend(pts.windows(2).map(|w| [w[0], w[1]]));
        out.extend(closing);
    }
    Ok(())
}

/// The squared distance between segments `p` and `q`, with the closest point
/// on each. Crossing or touching segments give 0 and the point where they
/// meet, decided by exact orientation tests.
fn segment_gap(p: Seg, q: Seg) -> (f64, P2, P2) {
    #[cfg(test)]
    PAIRS_TESTED.with(|c| c.set(c.get() + 1));
    let [a, b] = p;
    let [c, d] = q;
    let (s1, s2) = (side2::<false>(a, b, c), side2::<false>(a, b, d));
    let (s3, s4) = (side2::<false>(c, d, a), side2::<false>(c, d, b));
    if s1 * s2 < 0 && s3 * s4 < 0 {
        // A proper crossing: where `ab` meets the line through `cd`.
        let (o3, o4) = (orient2d_approx(c, d, a), orient2d_approx(c, d, b));
        let t = o3 / (o3 - o4);
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let x = lerp(a, b, t);
        return (0.0, x, x);
    }
    // An endpoint on the other segment.
    for (s, pt, [e, f]) in [(s1, c, p), (s2, d, p), (s3, a, q), (s4, b, q)] {
        let (lo, hi) = bounds(&[e, f]);
        if s == 0 && in_box(pt, lo, hi) {
            return (0.0, pt, pt);
        }
    }
    // Apart: the closest pair has an endpoint of one segment.
    let candidates = [
        (closest_on(c, a, b), c),
        (closest_on(d, a, b), d),
        (a, closest_on(a, c, d)),
        (b, closest_on(b, c, d)),
    ];
    let mut best = (f64::INFINITY, a, c);
    for (x, y) in candidates {
        let d2 = dist2(x, y);
        if d2 < best.0 {
            best = (d2, x, y);
        }
    }
    best
}

#[cfg(test)]
thread_local! {
    /// Test-only count of [`segment_gap`] calls on this thread.
    static PAIRS_TESTED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Test-only count of rings searched by [`Grid::search`] on this thread.
    static RINGS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The point of segment `ab` closest to `p`.
fn closest_on(p: P2, a: P2, b: P2) -> P2 {
    let ab = sub(b, a);
    let len2 = dot(ab, ab);
    if len2 == 0.0 {
        return a;
    }
    let t = dot(sub(p, a), ab) / len2;
    if t <= 0.0 {
        a
    } else if t >= 1.0 {
        b
    } else {
        lerp(a, b, t)
    }
}

/// A uniform grid over segments: each segment is listed in the cells it
/// passes through (see [`Grid::visit_cells`]), so every point of a segment
/// is in one of its cells or next to one, which the search's one-cell slack
/// allows for.
struct Grid {
    lo: P2,
    hi: P2,
    /// The cell size along each axis.
    size: P2,
    nx: usize,
    ny: usize,
    /// The segments of cell `x + y·nx` are `items[start[c]..start[c + 1]]`.
    start: Vec<usize>,
    items: Vec<usize>,
}

impl Grid {
    fn build(segs: &[Seg]) -> Result<Grid> {
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for s in segs {
            let (l, h) = bounds(s);
            for k in 0..2 {
                lo[k] = lo[k].min(l[k]);
                hi[k] = hi[k].max(h[k]);
            }
        }
        let (w, h) = (hi[0] - lo[0], hi[1] - lo[1]);
        let n = segs.len().max(1);
        let nf = n as f64;
        // About one cell per segment, square where the extent allows; a flat
        // extent gets one row or column, and an overflowing one a single cell.
        let c = (w * h / nf).sqrt();
        let (nx, ny) = if !(w.is_finite() && h.is_finite()) {
            (1, 1)
        } else if w > 0.0 && h > 0.0 {
            if c > 0.0 && c.is_finite() {
                let along = |e: f64| ((e / c).ceil() as usize).clamp(1, n);
                (along(w), along(h))
            } else {
                (1, 1)
            }
        } else {
            (if w > 0.0 { n } else { 1 }, if h > 0.0 { n } else { 1 })
        };
        let size_of = |e: f64, m: usize| {
            let s = e / m as f64;
            if s > 0.0 && s.is_finite() { s } else { 1.0 }
        };
        let mut grid = Grid {
            lo,
            hi,
            size: [size_of(w, nx), size_of(h, ny)],
            nx,
            ny,
            start: Vec::new(),
            items: Vec::new(),
        };
        let cells = nx * ny;
        reserve(&mut grid.start, cells + 1)?;
        grid.start.resize(cells + 1, 0);
        let (lo, size) = (grid.lo, grid.size);
        let mut total = 0usize;
        for s in segs {
            let mut count = 0;
            Grid::visit_cells(lo, size, nx, ny, s, |c| {
                grid.start[c + 1] += 1;
                count += 1;
            });
            total = total.saturating_add(count);
        }
        for c in 0..cells {
            grid.start[c + 1] += grid.start[c];
        }
        reserve(&mut grid.items, total)?;
        grid.items.resize(total, 0);
        // Fill each cell from its end down, leaving `start` as it was.
        let mut end = Vec::new();
        reserve(&mut end, cells)?;
        end.extend_from_slice(&grid.start[1..]);
        let items = &mut grid.items;
        for (j, s) in segs.iter().enumerate() {
            Grid::visit_cells(lo, size, nx, ny, s, |c| {
                end[c] -= 1;
                items[end[c]] = j;
            });
        }
        Ok(grid)
    }

    /// The cell coordinate of `x` along axis `k`, clamped to `-1..=n` (one
    /// past either edge of the grid).
    fn coord(&self, x: f64, k: usize) -> i64 {
        let n = [self.nx, self.ny][k] as f64;
        ((x - self.lo[k]) / self.size[k]).floor().clamp(-1.0, n) as i64
    }

    /// Calls `f` with every cell the segment `s` passes through, from its
    /// start to its end (Amanatides & Woo's grid traversal), so that a long
    /// diagonal costs cells along its length rather than its box. Each step
    /// moves one cell towards the end cell, so every cell comes up once and
    /// the walk ends after at most `nx + ny` steps, whatever the rounding;
    /// a step lands at most one cell from the segment, which the search's
    /// one-cell slack allows for.
    fn visit_cells(lo: P2, size: P2, nx: usize, ny: usize, s: &Seg, mut f: impl FnMut(usize)) {
        let coord = |p: f64, k: usize| {
            let n = [nx, ny][k] as f64;
            (((p - lo[k]) / size[k]).floor().clamp(-1.0, n) as i64).clamp(0, [nx, ny][k] as i64 - 1)
        };
        let g = |p: f64, k: usize| (p - lo[k]) / size[k];
        let [mut x, mut y] = [coord(s[0][0], 0), coord(s[0][1], 1)];
        let [ex, ey] = [coord(s[1][0], 0), coord(s[1][1], 1)];
        let (dx, dy) = (g(s[1][0], 0) - g(s[0][0], 0), g(s[1][1], 1) - g(s[0][1], 1));
        let (sx, sy) = ((ex - x).signum(), (ey - y).signum());
        // The fraction of the segment to the first grid line along each
        // axis, and per cell after that (`inf` past an axis already at the
        // end, which then never steps).
        let t_max = |d: f64, c: i64, e: i64, g0: f64| {
            if e == c {
                (f64::INFINITY, f64::INFINITY)
            } else if e > c {
                ((c as f64 + 1.0 - g0) / d, 1.0 / d)
            } else {
                ((g0 - c as f64) / -d, -1.0 / d)
            }
        };
        let (mut t_max_x, t_delta_x) = t_max(dx, x, ex, g(s[0][0], 0));
        let (mut t_max_y, t_delta_y) = t_max(dy, y, ey, g(s[0][1], 1));
        loop {
            f((x + y * nx as i64) as usize);
            if x == ex && y == ey {
                break;
            }
            // Step the nearer grid line; ties step `x` first, visiting the
            // corner's neighbour too. An axis at the end never steps.
            if x != ex && (y == ey || t_max_x <= t_max_y) {
                x += sx;
                t_max_x += t_delta_x;
            } else {
                y += sy;
                t_max_y += t_delta_y;
            }
        }
    }

    /// Tests `s` against the grid's segments that could be closer than
    /// `best`, improving it. Cells are visited in rings around the cells of
    /// `s`'s box. After each ring the unvisited cells form up to four strips
    /// of the grid box beyond the ring's sides, each grown by one cell toward
    /// `s` (slack for rounding in [`Grid::coord`]); the search stops once the
    /// nearest strip is at least `best` away.
    fn search(
        &self,
        s: Seg,
        segs: &[Seg],
        seen: &mut [usize],
        stamp: usize,
        best: &mut (f64, P2, P2),
    ) {
        let (lo, hi) = bounds(&s);
        // Every segment of the grid lies within its box.
        let to_grid = box_gap(lo, hi, self.lo, self.hi);
        if to_grid >= best.0 {
            return;
        }
        let (x0, x1) = (self.coord(lo[0], 0), self.coord(hi[0], 0));
        let (y0, y1) = (self.coord(lo[1], 1), self.coord(hi[1], 1));
        let (nx, ny) = (self.nx as i64, self.ny as i64);
        // The edge of cell `i` along axis `k`.
        let cell = |i: i64, k: usize| self.lo[k] + i as f64 * self.size[k];
        let mut visit = |x: i64, y: i64, best: &mut (f64, P2, P2)| {
            if !(0..nx).contains(&x) || !(0..ny).contains(&y) {
                return;
            }
            // The cell grown by a cell on every side (slack for rounding):
            // its segments lie within it, unless they reach a nearer cell.
            let near = box_gap(
                lo,
                hi,
                [cell(x - 1, 0), cell(y - 1, 1)],
                [cell(x + 2, 0), cell(y + 2, 1)],
            );
            if near >= best.0 {
                return;
            }
            let c = (x + y * nx) as usize;
            for &j in &self.items[self.start[c]..self.start[c + 1]] {
                if seen[j] != stamp {
                    seen[j] = stamp;
                    let (l, h) = bounds(&segs[j]);
                    if box_gap(lo, hi, l, h) >= best.0 {
                        continue;
                    }
                    let g = segment_gap(s, segs[j]);
                    if g.0 < best.0 {
                        *best = g;
                    }
                }
            }
        };
        let mut r = 0i64;
        loop {
            #[cfg(test)]
            RINGS.with(|c| c.set(c.get() + 1));
            let (rx0, rx1, ry0, ry1) = (x0 - r, x1 + r, y0 - r, y1 + r);
            for y in ry0.max(0)..=ry1.min(ny - 1) {
                if r == 0 || y == ry0 || y == ry1 {
                    for x in rx0.max(0)..=rx1.min(nx - 1) {
                        visit(x, y, best);
                    }
                } else {
                    visit(rx0, y, best);
                    visit(rx1, y, best);
                }
            }
            // The unvisited strips left, right, below and above the ring,
            // where the grid extends past it.
            let (glo, ghi) = (self.lo, self.hi);
            let strips = [
                (rx0 > 0).then(|| (glo, [cell(rx0 + 1, 0), ghi[1]])),
                (rx1 < nx - 1).then(|| ([cell(rx1, 0), glo[1]], ghi)),
                (ry0 > 0).then(|| (glo, [ghi[0], cell(ry0 + 1, 1)])),
                (ry1 < ny - 1).then(|| ([glo[0], cell(ry1, 1)], ghi)),
            ];
            let nearest = strips
                .into_iter()
                .flatten()
                .map(|(l, h)| box_gap(lo, hi, l, h))
                .fold(f64::INFINITY, f64::min);
            if best.0 == 0.0 || nearest.max(to_grid) >= best.0 {
                return;
            }
            r += 1;
        }
    }
}

/// The squared distance between the boxes `[alo, ahi]` and `[blo, bhi]`.
fn box_gap(alo: P2, ahi: P2, blo: P2, bhi: P2) -> f64 {
    let gap = |k: usize| (blo[k] - ahi[k]).max(alo[k] - bhi[k]).max(0.0);
    let d = [gap(0), gap(1)];
    dot(d, d)
}

/// Reserves room for `n` more elements, honouring the test fault hook.
fn reserve<T>(v: &mut Vec<T>, n: usize) -> Result<()> {
    allocation_allowed()?;
    v.try_reserve(n).map_err(|_| QueryError::OutOfMemory)
}

fn sub(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot(a: P2, b: P2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn cross(a: P2, b: P2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn lerp(a: P2, b: P2, t: f64) -> P2 {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

fn dist2(a: P2, b: P2) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

fn dist(a: P2, b: P2) -> f64 {
    let d = sub(a, b);
    d[0].hypot(d[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QueryError, fail_allocations_after};
    use proptest::prelude::*;
    use std::f64::consts::PI;

    fn square(c: [f64; 2], half: f64, ccw: bool) -> Vec<[f64; 2]> {
        let mut s = vec![
            [c[0] - half, c[1] - half],
            [c[0] + half, c[1] - half],
            [c[0] + half, c[1] + half],
            [c[0] - half, c[1] + half],
        ];
        if !ccw {
            s.reverse();
        }
        s
    }

    fn circle(r: f64, n: usize, ccw: bool) -> Vec<[f64; 2]> {
        let mut s: Vec<_> = (0..n)
            .map(|i| {
                let a = 2.0 * PI * i as f64 / n as f64;
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        if !ccw {
            s.reverse();
        }
        s
    }

    fn fill_area(f: &Fill) -> f64 {
        f.triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| f.points[i as usize]);
                0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            })
            .sum()
    }

    #[test]
    fn length_and_area_of_unit_square() {
        let mut s = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert_eq!(contour_length(&s, true), 4.0);
        assert_eq!(contour_length(&s, false), 3.0);
        assert_eq!(signed_area(&s), 1.0);
        s.reverse();
        assert_eq!(signed_area(&s), -1.0);
        assert_eq!(contour_length(&[], true), 0.0);
        assert_eq!(contour_length(&[[1.0, 2.0]], true), 0.0);
        assert_eq!(signed_area(&[[0.0, 0.0], [1.0, 0.0]]), 0.0);
    }

    #[test]
    fn fill_ring_area() {
        let outer = circle(2.0, 256, true);
        let inner = circle(1.0, 256, false);
        let mut out = Fill::default();
        assert_eq!(fill(&[(&outer, true), (&inner, true)], &mut out), Ok(true));
        let area = fill_area(&out);
        assert!((area - 3.0 * PI).abs() < 3.0 * PI * 1e-3, "{area}");
        assert_eq!(out.points.len(), 512);
        // Every triangle is counter-clockwise.
        let single = Fill {
            points: out.points.clone(),
            triangles: Vec::new(),
        };
        for &t in &out.triangles {
            let one = Fill {
                triangles: vec![t],
                ..single.clone()
            };
            assert!(fill_area(&one) >= 0.0);
        }
    }

    #[test]
    fn fill_nested_island() {
        let (a, b, c) = (
            square([0.0, 0.0], 5.0, true),
            square([0.0, 0.0], 3.0, false),
            square([0.0, 0.0], 1.0, true),
        );
        let mut out = Fill::default();
        // The order of the contours does not matter.
        for set in [[&c, &a, &b], [&a, &b, &c], [&b, &c, &a]] {
            let input: Vec<_> = set.iter().map(|s| (s.as_slice(), true)).collect();
            assert_eq!(fill(&input, &mut out), Ok(true));
            assert!((fill_area(&out) - (100.0 - 36.0 + 4.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn fill_unoriented_even_odd() {
        let mut out = Fill::default();
        for (wa, wb, wc) in [
            (true, true, true),
            (false, true, false),
            (true, false, true),
        ] {
            let (a, b, c) = (
                square([0.0, 0.0], 5.0, wa),
                square([0.5, 0.0], 3.0, wb),
                square([0.5, 0.5], 1.0, wc),
            );
            let input = [(&a[..], false), (&b[..], true), (&c[..], true)];
            assert_eq!(fill(&input, &mut out), Ok(true));
            assert!((fill_area(&out) - (100.0 - 36.0 + 4.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn fill_inside_out_contours() {
        // A ring, then the same ring wound inward (clockwise outer,
        // counter-clockwise hole): the same fill.
        let outer = square([0.0, 0.0], 3.0, true);
        let hole = square([0.5, 0.0], 1.0, false);
        let mut out = Fill::default();
        assert_eq!(fill(&[(&outer, true), (&hole, true)], &mut out), Ok(true));
        let area = fill_area(&out);
        assert!((area - 32.0).abs() < 1e-9, "{area}");
        let (outer_cw, hole_ccw) = (
            square([0.0, 0.0], 3.0, false),
            square([0.5, 0.0], 1.0, true),
        );
        assert_eq!(
            fill(&[(&outer_cw, true), (&hole_ccw, true)], &mut out),
            Ok(true)
        );
        assert_eq!(fill_area(&out), area);
        // A lone clockwise contour is an inside-out outer.
        assert_eq!(fill(&[(&outer_cw, true)], &mut out), Ok(true));
        assert!((fill_area(&out) - 36.0).abs() < 1e-9);
    }

    #[test]
    fn fill_hole_touching_outer_contour() {
        // A hole with an edge on the outer's edge (starting on it), and
        // one touching at a single apex: both still nest as holes and
        // fill outer minus hole.
        let outer = square([0.0, 0.0], 2.0, true);
        let flush = [[-2.0, 1.0], [-1.0, 1.0], [-1.0, 0.0], [-2.0, 0.0]];
        let apex = [[-1.0, -1.0], [1.0, -1.0], [0.0, -2.0]];
        for hole in [&flush[..], &apex[..]] {
            let mut out = Fill::default();
            assert_eq!(fill(&[(&outer, true), (hole, true)], &mut out), Ok(true));
            let area = fill_area(&out);
            assert!((area - 15.0).abs() < 1e-9, "{area}");
        }
    }

    #[test]
    fn fill_orphan_hole_and_reused_output() {
        // A clockwise contour inside no outer is not filled, so the area check
        // fails (and only that outer is filled).
        let outer = square([10.0, 0.0], 3.0, true);
        let hole = square([0.0, 0.0], 1.0, false);
        let mut out = Fill {
            points: vec![[9.0, 9.0]],
            triangles: vec![[0, 0, 0]],
        };
        assert_eq!(fill(&[(&outer, true), (&hole, true)], &mut out), Ok(false));
        assert_eq!(out.points, outer);
        assert!((fill_area(&out) - 36.0).abs() < 1e-9);
        // Collinear and tiny contours fill nothing and pass.
        let line = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        assert_eq!(fill(&[(&line, true), (&[], false)], &mut out), Ok(true));
        assert!(out.triangles.is_empty());
        assert_eq!(fill(&[], &mut out), Ok(true));
    }

    #[test]
    fn fill_non_finite_is_not_ok() {
        let mut s = square([0.0, 0.0], 1.0, true);
        s[1][0] = f64::NAN;
        let mut out = Fill::default();
        assert_eq!(fill(&[(&s, true)], &mut out), Ok(false));
        assert_eq!(fill(&[(&s, false)], &mut out), Ok(false));
    }

    #[test]
    fn fill_self_intersecting_reports_false_or_ok_area() {
        let bow = [[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]];
        let twisted = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [1.0, -1.0], [0.0, 4.0]];
        let mut out = Fill::default();
        for ring in [&bow[..], &twisted[..]] {
            for oriented in [true, false] {
                let ok = fill(&[(ring, oriented)], &mut out).unwrap();
                // One ring is an outer whichever way it turns.
                let expected = signed_area(ring).abs();
                if ok {
                    assert!((fill_area(&out) - expected).abs() <= 1e-6);
                }
            }
        }
    }

    #[test]
    fn fill_out_of_memory_leaves_output_empty() {
        let (a, b, c) = (
            square([0.0, 0.0], 5.0, true),
            square([0.0, 0.0], 3.0, false),
            square([0.0, 0.0], 1.0, true),
        );
        let input = [(&a[..], true), (&b[..], true), (&c[..], true)];
        let mut expected = Fill::default();
        assert_eq!(fill(&input, &mut expected), Ok(true));
        let mut n = 0;
        loop {
            let mut out = Fill {
                points: vec![[1.0, 1.0]],
                triangles: vec![[0, 0, 0]],
            };
            fail_allocations_after(Some(n));
            let r = fill(&input, &mut out);
            fail_allocations_after(None);
            match r {
                Err(QueryError::OutOfMemory) => assert_eq!(out, Fill::default()),
                Ok(ok) => {
                    assert!(ok);
                    assert_eq!(out, expected);
                    break;
                }
                Err(e) => panic!("{e}"),
            }
            n += 1;
        }
        assert!(n > 0);
    }

    #[test]
    fn gap_concentric_circles() {
        let a = circle(1.0, 512, true);
        let b = circle(3.0, 512, false);
        let g = contour_gap(&[(&a, true)], &[(&b, true)]).unwrap().unwrap();
        assert!((g.distance - 2.0).abs() < 1e-3, "{g:?}");
        let r = |p: [f64; 2]| p[0].hypot(p[1]);
        assert!((r(g.a) - 1.0).abs() < 1e-3);
        assert!((r(g.b) - 3.0).abs() < 1e-3);
        assert!(((g.a[0] - g.b[0]).hypot(g.a[1] - g.b[1]) - g.distance).abs() < 1e-12);
    }

    #[test]
    fn gap_is_not_all_pairs() {
        let a = circle(1.0, 2048, true);
        let b = circle(3.0, 2048, false);
        PAIRS_TESTED.with(|c| c.set(0));
        let g = contour_gap(&[(&a, true)], &[(&b, true)]).unwrap().unwrap();
        let tested = PAIRS_TESTED.with(|c| c.get());
        assert!((g.distance - 2.0).abs() < 1e-5);
        // About 45 pairs per segment, against 2048 for all pairs.
        assert!(tested < 2048 * 64, "{tested}");
    }

    #[test]
    fn gap_near_flat_side_is_not_quadratic() {
        // Side b is one grid row thinner than its segment spacing (a 1e-6
        // spike at one end); side a hangs over its other half.
        let n = 5000;
        let mut b: Vec<_> = (0..=n).map(|i| [10.0 * i as f64 / n as f64, 0.0]).collect();
        b[1][1] = 1e-6;
        for height in [1.0, 0.01] {
            let a: Vec<_> = (0..=n)
                .map(|i| [5.0 + 5.0 * i as f64 / n as f64, height])
                .collect();
            RINGS.with(|c| c.set(0));
            let g = contour_gap(&[(&a, false)], &[(&b, false)])
                .unwrap()
                .unwrap();
            let rings = RINGS.with(|c| c.get());
            assert_eq!(g.distance, height);
            // A few rings per segment, against about `n` each before.
            assert!(rings < 8 * n, "{rings}");
        }
    }

    #[test]
    fn gap_long_diagonal_lists_cells_along_its_length() {
        // 4096 short segments on a grid, plus one diagonal across it all:
        // listing the diagonal in every cell its box overlaps fills the
        // whole grid, while binning it along its length visits one row of
        // cells.
        let mut segs = Vec::new();
        for i in 0..4096 {
            let (x, y) = (f64::from(i % 64) + 0.5, f64::from(i / 64) + 0.5);
            segs.push([[x, y], [x + 0.01, y]]);
        }
        segs.push([[0.0, 0.0], [63.0, 63.0]]);
        let grid = Grid::build(&segs).unwrap();
        assert!((grid.nx * grid.ny) >= 4096, "{}x{}", grid.nx, grid.ny);
        let listed = grid.items.iter().filter(|&&j| j == 4096).count();
        assert!(listed < 512, "{listed}");
        // And the gap through the grid is still right.
        let b: Vec<_> = segs.iter().map(|s| (s.to_vec(), false)).collect();
        let rb: Vec<_> = b.iter().map(|(l, c)| (l.as_slice(), *c)).collect();
        let far = [[200.0, 200.0], [201.0, 200.0]];
        let g = contour_gap(&[(&far[..], false)], &rb).unwrap().unwrap();
        // Nearest: the end (63.51, 63.5) of the corner short segment.
        assert!(
            (g.distance - (200.0f64 - 63.51).hypot(200.0 - 63.5)).abs() < 1e-9,
            "{g:?}"
        );
        let brute = brute_gap(&[(&far[..], false)], &rb).unwrap();
        assert!((g.distance - brute).abs() <= 1e-12);
        // A long query segment crossing the grid agrees with brute force.
        let query = [[-50.0, 70.0], [50.0, -10.0]];
        let g = contour_gap(&[(&query[..], false)], &rb).unwrap().unwrap();
        let brute = brute_gap(&[(&query[..], false)], &rb).unwrap();
        assert!((g.distance - brute).abs() <= 1e-12);
    }

    #[test]
    fn gap_cost_does_not_depend_on_where_a_starts() {
        // Two circles 30 apart; contour a starts at its far side, then at
        // its point closest to b.
        let b: Vec<_> = circle(100.0, 2072, true)
            .into_iter()
            .map(|p| [p[0] + 230.0, p[1]])
            .collect();
        let mut tested = Vec::new();
        for start in [1036, 0] {
            let mut a = circle(100.0, 2072, true);
            a.rotate_left(start);
            PAIRS_TESTED.with(|c| c.set(0));
            let g = contour_gap(&[(&a, true)], &[(&b, true)]).unwrap().unwrap();
            assert!((g.distance - 30.0).abs() < 1e-9, "{g:?}");
            assert_eq!(g.a, [100.0, 0.0]);
            tested.push(PAIRS_TESTED.with(|c| c.get()));
        }
        // A handful of segments on each side, either way.
        assert!(tested.iter().all(|&t| t < 64), "{tested:?}");
    }

    #[test]
    fn gap_crossing_is_zero() {
        let a = [[0.0, 0.0], [2.0, 2.0]];
        let b = [[0.0, 2.0], [2.0, 0.0]];
        let g = contour_gap(&[(&a, false)], &[(&b, false)])
            .unwrap()
            .unwrap();
        assert_eq!(g.distance, 0.0);
        assert_eq!(g.a, [1.0, 1.0]);
        assert_eq!(g.b, [1.0, 1.0]);
        // Two squares overlapping at a corner.
        let s = square([0.0, 0.0], 1.0, true);
        let t = square([1.5, 1.5], 1.0, true);
        let g = contour_gap(&[(&s, true)], &[(&t, true)]).unwrap().unwrap();
        assert_eq!(g.distance, 0.0);
        assert_eq!(g.a, g.b);
        // Touching end to end, and overlapping collinear segments.
        let c = [[2.0, 2.0], [3.0, 5.0]];
        let g = contour_gap(&[(&a, false)], &[(&c, false)])
            .unwrap()
            .unwrap();
        assert_eq!((g.distance, g.a, g.b), (0.0, [2.0, 2.0], [2.0, 2.0]));
        let d = [[1.0, 1.0], [3.0, 3.0]];
        let g = contour_gap(&[(&a, false)], &[(&d, false)])
            .unwrap()
            .unwrap();
        assert_eq!(g.distance, 0.0);
        assert_eq!(g.a, g.b);
    }

    #[test]
    fn gap_of_closing_segment_and_far_apart_sets() {
        // Only the closing segment of the open-looking triangle comes near.
        let tri = [[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let p = [[-1.0, 5.0], [-2.0, 5.0]];
        let open = contour_gap(&[(&tri, false)], &[(&p, false)])
            .unwrap()
            .unwrap();
        let closed = contour_gap(&[(&tri, true)], &[(&p, false)])
            .unwrap()
            .unwrap();
        assert!(closed.distance < open.distance);
        assert_eq!(
            (closed.distance, closed.a, closed.b),
            (1.0, [0.0, 5.0], [-1.0, 5.0])
        );
        let far = [[1000.0, -3.0], [1000.0, 17.0]];
        let g = contour_gap(&[(&far, false)], &[(&tri, true)])
            .unwrap()
            .unwrap();
        assert_eq!(
            (g.distance, g.a, g.b),
            (990.0, [1000.0, 10.0], [10.0, 10.0])
        );
    }

    #[test]
    fn gap_none_when_either_empty() {
        let s = square([0.0, 0.0], 1.0, true);
        let one = [[5.0, 5.0]];
        assert_eq!(contour_gap(&[], &[(&s, true)]), Ok(None));
        assert_eq!(contour_gap(&[(&s, true)], &[]), Ok(None));
        assert_eq!(contour_gap(&[(&one, true)], &[(&s, true)]), Ok(None));
        assert_eq!(contour_gap(&[(&s, false)], &[(&[], true)]), Ok(None));
    }

    #[test]
    fn gap_degenerate_and_non_finite() {
        // Every point of the second side in one place: one grid cell.
        let s = square([0.0, 0.0], 1.0, true);
        let dot = [[3.0, 0.0], [3.0, 0.0]];
        let g = contour_gap(&[(&s, true)], &[(&dot, false)])
            .unwrap()
            .unwrap();
        assert_eq!((g.distance, g.a, g.b), (2.0, [1.0, 0.0], [3.0, 0.0]));
        let bad = [[0.0, 0.0], [f64::INFINITY, 0.0]];
        assert!(matches!(
            contour_gap(&[(&s, true)], &[(&bad, false)]),
            Err(QueryError::InvalidInput(_))
        ));
        assert!(matches!(
            contour_gap(&[(&bad, false)], &[(&s, true)]),
            Err(QueryError::InvalidInput(_))
        ));
    }

    #[test]
    fn gap_out_of_memory() {
        let a = circle(1.0, 64, true);
        let b = circle(3.0, 64, false);
        let expected = contour_gap(&[(&a, true)], &[(&b, true)]).unwrap();
        let mut n = 0;
        loop {
            fail_allocations_after(Some(n));
            let r = contour_gap(&[(&a, true)], &[(&b, true)]);
            fail_allocations_after(None);
            match r {
                Err(QueryError::OutOfMemory) => n += 1,
                r => {
                    assert_eq!(r, Ok(expected));
                    break;
                }
            }
        }
        assert!(n > 0);
    }

    /// The all-pairs minimum, through the same segment test.
    fn brute_gap(a: &[(&[[f64; 2]], bool)], b: &[(&[[f64; 2]], bool)]) -> Option<f64> {
        let mut sa = Vec::new();
        let mut sb = Vec::new();
        collect_segments(a, &mut sa).unwrap();
        collect_segments(b, &mut sb).unwrap();
        let mut best: Option<f64> = None;
        for p in &sa {
            for q in &sb {
                let d = segment_gap(*p, *q).0;
                best = Some(best.map_or(d, |b| b.min(d)));
            }
        }
        best.map(f64::sqrt)
    }

    fn cases(default: u32) -> u32 {
        std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }

    fn polylines() -> impl Strategy<Value = Vec<(Vec<[f64; 2]>, bool)>> {
        // A coarse lattice now and then, so that crossings and touches are common.
        let coord = prop_oneof![(-20i32..20).prop_map(f64::from), -20.0f64..20.0];
        let line = (
            prop::collection::vec([coord.clone(), coord], 0..=51),
            any::<bool>(),
        );
        prop::collection::vec(line, 2..=5)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(cases(256)))]

        #[test]
        fn gap_matches_brute_force(a in polylines(), b in polylines(), shift in 0.0f64..60.0) {
            let b: Vec<_> = b
                .into_iter()
                .map(|(l, c)| (l.into_iter().map(|p| [p[0] + shift, p[1]]).collect::<Vec<_>>(), c))
                .collect();
            let ra: Vec<_> = a.iter().map(|(l, c)| (l.as_slice(), *c)).collect();
            let rb: Vec<_> = b.iter().map(|(l, c)| (l.as_slice(), *c)).collect();
            let got = contour_gap(&ra, &rb).unwrap();
            let want = brute_gap(&ra, &rb);
            match (got, want) {
                (None, None) => {}
                (Some(g), Some(w)) => {
                    prop_assert!((g.distance - w).abs() <= 1e-12, "{} vs {}", g.distance, w);
                    let d = (g.a[0] - g.b[0]).hypot(g.a[1] - g.b[1]);
                    prop_assert!((d - g.distance).abs() <= 1e-9);
                }
                (g, w) => prop_assert!(false, "{:?} vs {:?}", g, w),
            }
        }
    }
}
