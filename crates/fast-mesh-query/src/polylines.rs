//! Intersection polylines between two meshes: the outline of the region
//! where one mesh penetrates the other.
//!
//! Every pair of [`intersecting_pairs`](crate::intersecting_pairs) is clipped
//! to its intersection segment, and the segments are chained into polylines
//! with welded endpoints. Which pairs meet, and whether a pair crosses,
//! touches, is coplanar or involves a degenerate triangle, is decided by
//! exact predicates; the segment endpoints are `f64` interpolations, stored
//! as `f32`.

use fast_mesh_bvh::Bvh;
use fast_mesh_store::Mesh;

use crate::pair::PAIR_STACK;
use crate::pairs::{for_each_pair, overlap_key, validate};
use crate::tritri::{
    P3, collinear, in_tri_2d, normal_axis, orient2d_approx, side, side2, tri_tri_intersect,
};
use crate::{QueryError, Result, Rigid, allocation_allowed};

/// One chain of the intersection: a closed loop or an open polyline, in A's
/// space (B placed by the query's [`Rigid`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Polyline {
    /// The chain's points. A closed polyline's last point joins back to the
    /// first (not repeated); an open one's has one fewer segment than points.
    pub points: Vec<[f32; 3]>,
    /// Whether the last point joins back to the first.
    pub closed: bool,
    /// Whether no end of the polyline is a junction where three or more
    /// segments meet. Chains always end at junctions; a closed polyline is
    /// manifold.
    pub manifold: bool,
}

/// Appends to `out` the polylines where `mesh_a` and `mesh_b` (placed in A's
/// space by `xf`) intersect, and returns how many were appended. Elements
/// already in `out` are kept and stay in front.
///
/// A pair of triangles that only touch at a point contributes no segment;
/// coplanar pairs contribute only area overlaps, as their boundary
/// (interior edges shared by two pairs cancel; a touch along an edge or at
/// a point contributes nothing); a degenerate triangle contributes the
/// overlap of its longest edge. Segments are welded by exact `f32` equality:
/// endpoints computed from the same edge and plane are bit-identical, and
/// endpoints on a shared vertex snap to it exactly, so loops close.
///
/// The order is deterministic: open polylines first, each from its smaller
/// end, then closed loops, each from its smallest point, and the polylines
/// sorted by their first point.
///
/// # Errors
/// [`QueryError::InvalidInput`] for an invalid `xf` (see [`Rigid`]); then
/// [`QueryError::MeshMismatch`] or [`QueryError::StaleBvh`] when a BVH does
/// not match its mesh; and [`QueryError::OutOfMemory`] when `out` or a
/// temporary cannot grow. On any error `out` keeps its original length.
pub fn intersection_polylines(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    out: &mut Vec<Polyline>,
) -> Result<usize> {
    validate(mesh_a, bvh_a, mesh_b, bvh_b, xf, None)?;
    let start = out.len();
    let r = polylines(mesh_a, bvh_a, mesh_b, bvh_b, xf, out);
    if r.is_err() {
        out.truncate(start);
    }
    r
}

fn polylines(
    mesh_a: &Mesh,
    bvh_a: &Bvh,
    mesh_b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    out: &mut Vec<Polyline>,
) -> Result<usize> {
    // Every box-overlapping pair is classified exactly; intersecting ones
    // are clipped.
    let mut segs: Vec<Seg64> = Vec::new();
    let mut failed = false;
    let mut buf = [Seg64::EMPTY; 6];
    for_each_pair::<PAIR_STACK>(
        mesh_a,
        bvh_a,
        mesh_b,
        bvh_b,
        xf,
        None,
        overlap_key,
        |_, pa, _, pb| {
            if failed {
                return false;
            }
            if !tri_tri_intersect(pa, pb) {
                return true;
            }
            let n = clip_pair(pa, pb, &mut buf);
            for s in &buf[..n] {
                if s.from == s.to {
                    continue;
                }
                if push(&mut segs, *s).is_err() {
                    failed = true;
                    return false;
                }
            }
            true
        },
    );
    if failed {
        return Err(QueryError::OutOfMemory);
    }
    if segs.is_empty() {
        return Ok(0);
    }
    // Interior coplanar edges shared by two pairs cancel; the rest chains.
    cancel_dupes(&mut segs);
    if segs.is_empty() {
        return Ok(0);
    }
    let mut segs32: Vec<Seg32> = Vec::new();
    for s in &segs {
        push(
            &mut segs32,
            Seg32 {
                from: s.from.map(|x| x as f32),
                to: s.to.map(|x| x as f32),
            },
        )?;
    }
    segs32.retain(|s| s.from != s.to);
    if segs32.is_empty() {
        return Ok(0);
    }
    chain(&segs32, out)
}

/// An intersection segment after the `f32` cast, before welding.
#[derive(Debug, Clone, Copy)]
struct Seg32 {
    from: [f32; 3],
    to: [f32; 3],
}

/// Drops exact duplicate segments: reversed pairs (the interior edge of a
/// coplanar overlap, emitted once per side) cancel, same-direction ones
/// (coincident triangles) collapse to one. Anything else is kept: only
/// bit-identical endpoints merge, so T-junction stubs stay.
fn cancel_dupes(segs: &mut Vec<Seg64>) {
    segs.sort_unstable_by_key(|s| {
        let (a, b) = (s.from.map(f64::to_bits), s.to.map(f64::to_bits));
        if a <= b { (a, b) } else { (b, a) }
    });
    let key = |s: &Seg64| {
        let (a, b) = (s.from.map(f64::to_bits), s.to.map(f64::to_bits));
        if a <= b { (a, b) } else { (b, a) }
    };
    let mut w = 0;
    let mut i = 0;
    while i < segs.len() {
        let k0 = key(&segs[i]);
        let (mut fwd, mut bwd) = (None, None);
        let (mut nf, mut nb) = (0, 0);
        while i + nf + nb < segs.len() && key(&segs[i + nf + nb]) == k0 {
            let s = segs[i + nf + nb];
            if s.from.map(f64::to_bits) == k0.0 {
                nf += 1;
                fwd = Some(s);
            } else {
                nb += 1;
                bwd = Some(s);
            }
        }
        // Reversed pairs cancel; at most one of the remainder survives.
        let cancel = nf.min(nb);
        if nf - cancel > 0 {
            segs[w] = fwd.unwrap();
            w += 1;
        } else if nb - cancel > 0 {
            segs[w] = bwd.unwrap();
            w += 1;
        }
        i += nf + nb;
    }
    segs.truncate(w);
}

/// End links, as in [`crate::slice`]: a unique point ends a chain, a pair
/// links through, three or more split it.
const BOUNDARY: usize = usize::MAX;
const SPLIT: usize = usize::MAX - 1;

/// Chains `segs` into `out` by exact `f32` equality of endpoints: open
/// polylines from their smaller end first, then closed loops from their
/// smallest point, and the polylines sorted by their first point. Returns
/// how many were appended.
fn chain(segs: &[Seg32], out: &mut Vec<Polyline>) -> Result<usize> {
    let start = out.len();
    let key_of = |e: usize| {
        let s = &segs[e / 2];
        let p = if e.is_multiple_of(2) { s.from } else { s.to };
        p.map(f32::to_bits)
    };
    let mut ends: Vec<usize> = Vec::new();
    reserve(&mut ends, 2 * segs.len())?;
    ends.extend(0..2 * segs.len());
    ends.sort_unstable_by_key(|&e| (key_of(e), e & 1, e));
    // End to its position in `ends`.
    let mut index = Vec::new();
    reserve(&mut index, ends.len())?;
    index.resize(ends.len(), 0);
    for (pos, &e) in ends.iter().enumerate() {
        index[e] = pos;
    }
    let mut link = Vec::new();
    reserve(&mut link, ends.len())?;
    link.resize(ends.len(), BOUNDARY);
    for group in ends.chunk_by(|a, b| key_of(*a) == key_of(*b)) {
        match group.len() {
            1 => link[index[group[0]]] = BOUNDARY,
            2 => {
                link[index[group[0]]] = group[1];
                link[index[group[1]]] = group[0];
            }
            _ => {
                for &e in group {
                    link[index[e]] = SPLIT;
                }
            }
        }
    }
    let mut used = Vec::new();
    reserve(&mut used, segs.len())?;
    used.resize(segs.len(), false);
    // Open chains from boundary or split ends, then closed loops.
    for closed_pass in [false, true] {
        for (pos, &e) in ends.iter().enumerate() {
            if used[e / 2] || (!closed_pass && link[pos] < SPLIT) {
                continue;
            }
            let first = e;
            let mut points = Vec::new();
            let mut entry = e;
            let next = loop {
                used[entry / 2] = true;
                let s = &segs[entry / 2];
                push(
                    &mut points,
                    if entry.is_multiple_of(2) {
                        s.from
                    } else {
                        s.to
                    },
                )?;
                let exit = entry ^ 1;
                let l = link[index[exit]];
                if l >= SPLIT || l == first {
                    break l;
                }
                entry = l;
            };
            let closed = next == first;
            if !closed {
                // The last point joins nothing further, so the walk did not
                // push it; a closed loop repeats nothing.
                let s = &segs[entry / 2];
                push(
                    &mut points,
                    if entry.is_multiple_of(2) {
                        s.to
                    } else {
                        s.from
                    },
                )?;
            }
            let manifold = link[pos] != SPLIT && next != SPLIT;
            push(
                out,
                Polyline {
                    points,
                    closed,
                    manifold,
                },
            )?;
        }
    }
    out[start..].sort_unstable_by_key(|c| c.points[0].map(f32::to_bits));
    Ok(out.len() - start)
}

/// Reserves `n` more elements, charging the fallible-allocation hook first.
fn reserve<T>(v: &mut Vec<T>, n: usize) -> Result<()> {
    allocation_allowed()?;
    v.try_reserve(n).map_err(|_| QueryError::OutOfMemory)
}

/// Pushes `x`, charging the fallible-allocation hook first.
fn push<T>(v: &mut Vec<T>, x: T) -> Result<()> {
    reserve(v, 1)?;
    v.push(x);
    Ok(())
}

/// A clipped intersection segment in A's space (`f64`, before welding).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Seg64 {
    from: [f64; 3],
    to: [f64; 3],
}

impl Seg64 {
    const EMPTY: Seg64 = Seg64 {
        from: [0.0; 3],
        to: [0.0; 3],
    };
}

/// Clips the intersecting pair `pa`, `pb` (both in A's space) to its
/// intersection segments, writing at most 6 into `segs` and returning how
/// many. A pair that only touches at a point gives none.
///
/// Every branch is decided by exact signs of the input coordinates; only
/// the endpoints are interpolated (`f64`). Endpoints that coincide with an
/// input corner snap to it exactly, so segments from adjacent pairs weld.
fn clip_pair(pa: [P3; 3], pb: [P3; 3], segs: &mut [Seg64; 6]) -> usize {
    let (da, db) = (collinear::<false>(&pa), collinear::<false>(&pb));
    if da || db {
        return clip_degenerate(pa, da, pb, db, segs);
    }
    let sa = pa.map(|p| side::<false>(pb[0], pb[1], pb[2], p));
    let sb = pb.map(|p| side::<false>(pa[0], pa[1], pa[2], p));
    if sa == [0; 3] && sb == [0; 3] {
        return clip_coplanar(pa, pb, segs);
    }
    clip_crossing(pa, pb, segs)
}

/// The non-coplanar case of [`clip_pair`]: each triangle's edges are crossed
/// with the other's plane, crossings inside the other triangle are kept, and
/// the extremes along the planes' line are the segment.
fn clip_crossing(pa: [P3; 3], pb: [P3; 3], segs: &mut [Seg64; 6]) -> usize {
    let (na, nb) = (normal(pa), normal(pb));
    let mut pts = [[0.0; 3]; 12];
    let mut n = 0;
    for (edges, plane, nrm) in [(pa, pb, nb), (pb, pa, na)] {
        let axis = normal_axis::<false>(&plane).expect("non-degenerate triangle");
        let plane2 = plane.map(|p| drop_axis(p, axis));
        for i in 0..3 {
            let (u, v) = (edges[i], edges[(i + 1) % 3]);
            let (su, sv) = (
                side::<false>(plane[0], plane[1], plane[2], u),
                side::<false>(plane[0], plane[1], plane[2], v),
            );
            if su * sv > 0 {
                continue;
            }
            if su == 0 && sv == 0 {
                // The edge lies in the plane: each end in the triangle counts.
                for &x in &[u, v] {
                    if in_tri_2d::<false>(drop_axis(x, axis), &plane2) && !pts[..n].contains(&x) {
                        pts[n] = x;
                        n += 1;
                    }
                }
                continue;
            }
            let x = if su == 0 {
                u
            } else if sv == 0 {
                v
            } else {
                edge_plane_cross(u, v, plane[0], nrm)
            };
            if !finite(x) {
                continue;
            }
            if in_tri_2d::<false>(drop_axis(x, axis), &plane2) && !pts[..n].contains(&x) {
                pts[n] = x;
                n += 1;
            }
        }
    }
    if n < 2 {
        return 0;
    }
    // The extremes along the planes' line. All kept points lie on it up to
    // rounding, and the intersection is convex, so they span the segment.
    let d = cross(na, nb);
    let mut lo = 0;
    let mut hi = 0;
    for i in 1..n {
        if dot(pts[i], d) < dot(pts[lo], d) {
            lo = i;
        }
        if dot(pts[i], d) > dot(pts[hi], d) {
            hi = i;
        }
    }
    if pts[lo] == pts[hi] {
        return 0;
    }
    segs[0] = Seg64 {
        from: pts[lo],
        to: pts[hi],
    };
    1
}

/// Where the straddling edge `u..v` meets the plane through `o` with normal
/// `n`: `u` and `v` are strictly on opposite sides. The endpoints are ordered
/// by bits first, so the same edge crossed from adjacent pairs interpolates
/// bit-identically.
fn edge_plane_cross(u: P3, v: P3, o: P3, n: P3) -> P3 {
    let (lo, hi) = order(u, v);
    let hlo = dot(n, sub(lo, o));
    let hhi = dot(n, sub(hi, o));
    let t = hlo / (hlo - hhi);
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.5
    };
    std::array::from_fn(|k| lo[k] + t * (hi[k] - lo[k]))
}

/// Whether every coordinate of `p` is finite. Interpolations at absurd
/// coordinate scales can overflow; such crossings are dropped rather than
/// fed to the exact predicates.
fn finite(p: P3) -> bool {
    p.iter().all(|x| x.is_finite())
}

/// The endpoints ordered by bits, for direction-independent interpolation.
fn order(a: P3, b: P3) -> (P3, P3) {
    if a.map(f64::to_bits) <= b.map(f64::to_bits) {
        (a, b)
    } else {
        (b, a)
    }
}

fn sub(a: P3, b: P3) -> P3 {
    std::array::from_fn(|k| a[k] - b[k])
}

fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normal(t: [P3; 3]) -> P3 {
    cross(sub(t[1], t[0]), sub(t[2], t[0]))
}

/// Drops `axis` from `p`, projecting onto a coordinate plane.
fn drop_axis(p: P3, axis: usize) -> [f64; 2] {
    [p[(axis + 1) % 3], p[(axis + 2) % 3]]
}

/// One Sutherland–Hodgman stage: `poly` clipped by the half-plane
/// through `e0..e1` holding `s_in`, into `next`. Returns the vertex count,
/// or 0 when the polygon is empty or would exceed `next`'s 6 vertices:
/// exact signs on rounded crossings can weave a non-convex pattern (a
/// pentagon with four crossings emits 7) that no convex input produces, so
/// the stage bails out instead of writing out of bounds.
fn clip_stage(
    poly: &[[f64; 2]],
    e0: [f64; 2],
    e1: [f64; 2],
    s_in: i8,
    next: &mut [[f64; 2]; 6],
) -> usize {
    let n = poly.len();
    let mut m = 0;
    for i in 0..n {
        let (p, q) = (poly[i], poly[(i + 1) % n]);
        let (sp, sq) = (side2::<false>(e0, e1, p), side2::<false>(e0, e1, q));
        let pin = sp == 0 || sp == s_in;
        let qin = sq == 0 || sq == s_in;
        if pin {
            if m == 6 {
                return 0;
            }
            next[m] = p;
            m += 1;
        }
        if pin != qin {
            if m == 6 {
                return 0;
            }
            next[m] = if sp == 0 {
                p
            } else if sq == 0 {
                q
            } else {
                cross_2d(p, q, e0, e1)
            };
            m += 1;
        }
    }
    m
}

/// The coplanar case of [`clip_pair`]: `pa` clipped by `pb`'s half-planes,
/// whose boundary is the segments.
fn clip_coplanar(pa: [P3; 3], pb: [P3; 3], segs: &mut [Seg64; 6]) -> usize {
    let axis = normal_axis::<false>(&pa).expect("non-degenerate triangle");
    let a = pa.map(|p| drop_axis(p, axis));
    let b = pb.map(|p| drop_axis(p, axis));
    let mut poly = [a[0], a[1], a[2], a[0], a[0], a[0]];
    let mut n = 3;
    // Sutherland–Hodgman: inside tests are exact signs, crossings are
    // interpolated with exact endpoints snapped.
    for j in 0..3 {
        let s_in = side2::<false>(b[j], b[(j + 1) % 3], b[(j + 2) % 3]);
        debug_assert!(s_in != 0);
        let mut next = [[0.0; 2]; 6];
        n = clip_stage(&poly[..n], b[j], b[(j + 1) % 3], s_in, &mut next);
        poly = next;
        if n == 0 {
            return 0;
        }
    }
    // Consecutive duplicates (touching at a point) collapse.
    let mut q = [[0.0; 2]; 6];
    let mut m = 0;
    for p in poly.iter().take(n) {
        if !p.iter().all(|x| x.is_finite()) {
            continue;
        }
        if m == 0 || *p != q[m - 1] {
            q[m] = *p;
            m += 1;
        }
    }
    if m > 1 && q[m - 1] == q[0] {
        m -= 1;
    }
    if m < 3 {
        // A point or segment touch: coplanar pairs contribute only area
        // overlaps, whose boundary follows. A one-dimensional overlap is
        // either interior to a neighboring area overlap, whose boundary
        // already covers it, or a tangential touch; emitting it would break
        // the reversed-pair cancellation counts.
        return 0;
    }
    if (0..m).all(|i| side2::<false>(q[i], q[(i + 1) % m], q[(i + 2) % m]) == 0) {
        // Zero area: all on one line.
        return 0;
    }
    // A vertex that is an input corner of either triangle snaps to it
    // exactly, so pairs sharing the corner weld.
    let corners = [
        (drop_axis(pa[0], axis), pa[0]),
        (drop_axis(pa[1], axis), pa[1]),
        (drop_axis(pa[2], axis), pa[2]),
        (drop_axis(pb[0], axis), pb[0]),
        (drop_axis(pb[1], axis), pb[1]),
        (drop_axis(pb[2], axis), pb[2]),
    ];
    let mut pts = [[0.0; 3]; 6];
    for i in 0..m {
        pts[i] = lift(q[i], axis, pa, &corners);
    }
    for i in 0..m {
        segs[i] = Seg64 {
            from: pts[i],
            to: pts[(i + 1) % m],
        };
    }
    m
}

/// Where the straddling segment `p..q` meets the line through `e0..e1`,
/// with endpoints ordered by bits so adjacent pairs agree bit-identically.
fn cross_2d(p: [f64; 2], q: [f64; 2], e0: [f64; 2], e1: [f64; 2]) -> [f64; 2] {
    let (lo, hi) = if p.map(f64::to_bits) <= q.map(f64::to_bits) {
        (p, q)
    } else {
        (q, p)
    };
    let o3 = orient2d_approx(e0, e1, lo);
    let o4 = orient2d_approx(e0, e1, hi);
    let t = o3 / (o3 - o4);
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.5
    };
    [lo[0] + t * (hi[0] - lo[0]), lo[1] + t * (hi[1] - lo[1])]
}

/// The degenerate case of [`clip_pair`]: `da`/`db` say which of `pa`/`pb`
/// is collinear. A degenerate triangle is the union of its edges, covered
/// by its longest one; only an overlap along a line gives a segment, a
/// point contact gives none.
fn clip_degenerate(pa: [P3; 3], da: bool, pb: [P3; 3], db: bool, segs: &mut [Seg64; 6]) -> usize {
    let (sa, sb) = (longest_edge(pa), longest_edge(pb));
    if da && db {
        return seg_seg_overlap(sa, sb, segs);
    }
    let (s, t) = if da { (sa, pb) } else { (sb, pa) };
    seg_in_plane_overlap(s, t, segs)
}

/// The longest edge of `t`, for a degenerate triangle covering the union of
/// its edges. Exact endpoints from the inputs.
fn longest_edge(t: [P3; 3]) -> (P3, P3) {
    let mut best = (t[0], t[1]);
    let mut len = dist2(t[0], t[1]);
    for i in 1..3 {
        let (u, v) = (t[i], t[(i + 1) % 3]);
        if dist2(u, v) > len {
            len = dist2(u, v);
            best = (u, v);
        }
    }
    best
}

fn dist2(a: P3, b: P3) -> f64 {
    (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])
}

/// The overlap of the segment `s` with the non-degenerate triangle `t`: a
/// segment only when `s` lies in `t`'s plane.
fn seg_in_plane_overlap(s: (P3, P3), t: [P3; 3], segs: &mut [Seg64; 6]) -> usize {
    let (su, sv) = (
        side::<false>(t[0], t[1], t[2], s.0),
        side::<false>(t[0], t[1], t[2], s.1),
    );
    if su != 0 || sv != 0 {
        // Straddling, or off the plane: at most one contact point.
        return 0;
    }
    let axis = normal_axis::<false>(&t).expect("non-degenerate triangle");
    let t2 = t.map(|p| drop_axis(p, axis));
    let (mut p, mut q) = (drop_axis(s.0, axis), drop_axis(s.1, axis));
    // Liang–Barsky over the three half-planes: exact classification,
    // interpolated crossings, input corners snapped.
    for j in 0..3 {
        let s_in = side2::<false>(t2[j], t2[(j + 1) % 3], t2[(j + 2) % 3]);
        debug_assert!(s_in != 0);
        let (sp, sq) = (
            side2::<false>(t2[j], t2[(j + 1) % 3], p),
            side2::<false>(t2[j], t2[(j + 1) % 3], q),
        );
        let pin = sp == 0 || sp == s_in;
        let qin = sq == 0 || sq == s_in;
        if !pin && !qin {
            return 0;
        }
        if pin && qin {
            continue;
        }
        let x = if sp == 0 {
            p
        } else if sq == 0 {
            q
        } else {
            cross_2d(p, q, t2[j], t2[(j + 1) % 3])
        };
        if pin {
            q = x;
        } else {
            p = x;
        }
    }
    if p == q {
        return 0;
    }
    let corners = [
        (drop_axis(s.0, axis), s.0),
        (drop_axis(s.1, axis), s.1),
        (t2[0], t[0]),
        (t2[1], t[1]),
        (t2[2], t[2]),
    ];
    let (a, b) = (lift(p, axis, t, &corners), lift(q, axis, t, &corners));
    if a == b {
        return 0;
    }
    segs[0] = Seg64 { from: a, to: b };
    1
}

/// The overlap of two segments in 3D: their shared line portion, or nothing
/// for a point contact or skew lines. Overlap endpoints are input endpoints.
fn seg_seg_overlap(a: (P3, P3), b: (P3, P3), segs: &mut [Seg64; 6]) -> usize {
    let (p, q, r, s) = (a.0, a.1, b.0, b.1);
    if side::<false>(p, q, r, s) != 0 {
        return 0;
    }
    // Coplanar: project along the normal of a non-collinear triple, where
    // both segments project injectively.
    for t in [[p, q, r], [p, q, s], [r, s, p], [r, s, q]] {
        if let Some(k) = normal_axis::<false>(&t) {
            let (p2, q2, r2, s2) = (
                drop_axis(p, k),
                drop_axis(q, k),
                drop_axis(r, k),
                drop_axis(s, k),
            );
            if side2::<false>(p2, q2, r2) != 0 || side2::<false>(p2, q2, s2) != 0 {
                // A proper crossing or touch: one point.
                return 0;
            }
            return collinear_overlap_2d(
                p2,
                q2,
                r2,
                s2,
                [(p, p2), (q, q2), (r, r2), (s, s2)],
                segs,
            );
        }
    }
    // One line: overlap along the widest-spread axis.
    let mut k = 0;
    for j in 1..3 {
        let spread = |k: usize| {
            [p[k], q[k], r[k], s[k]]
                .into_iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
                    (lo.min(x), hi.max(x))
                })
        };
        let (lo0, hi0) = spread(k);
        let (lo1, hi1) = spread(j);
        if hi1 - lo1 > hi0 - lo0 {
            k = j;
        }
    }
    line_overlap(p, q, r, s, k, segs)
}

/// The overlap of two collinear 2D segments, lifted to the 3D input
/// endpoints `pts` (each 3D point with its projection).
fn collinear_overlap_2d(
    p: [f64; 2],
    q: [f64; 2],
    r: [f64; 2],
    s: [f64; 2],
    pts: [(P3, [f64; 2]); 4],
    segs: &mut [Seg64; 6],
) -> usize {
    // Order along the widest-spread projected axis.
    let k = if (p[0] - q[0]).abs().max((r[0] - s[0]).abs())
        >= (p[1] - q[1]).abs().max((r[1] - s[1]).abs())
    {
        0
    } else {
        1
    };
    let key = |x: [f64; 2]| x[k];
    let (lo, hi) = (
        key(p).min(key(q)).max(key(r).min(key(s))),
        key(p).max(key(q)).min(key(r).max(key(s))),
    );
    if lo >= hi {
        return 0;
    }
    // The overlap bounds are input endpoints.
    let at = |t: f64| pts.iter().find(|(_, x)| key(*x) == t).map(|(p, _)| *p);
    match (at(lo), at(hi)) {
        (Some(a), Some(b)) if a != b => {
            segs[0] = Seg64 { from: a, to: b };
            1
        }
        _ => 0,
    }
}

/// The overlap of four collinear 3D points' two segments along axis `k`.
fn line_overlap(p: P3, q: P3, r: P3, s: P3, k: usize, segs: &mut [Seg64; 6]) -> usize {
    let (lo, hi) = (
        p[k].min(q[k]).max(r[k].min(s[k])),
        p[k].max(q[k]).min(r[k].max(s[k])),
    );
    if lo >= hi {
        return 0;
    }
    let at = |t: f64| [p, q, r, s].into_iter().find(|x| x[k] == t);
    match (at(lo), at(hi)) {
        (Some(a), Some(b)) if a != b => {
            segs[0] = Seg64 { from: a, to: b };
            1
        }
        _ => 0,
    }
}

/// Lifts the projected point `x` back to 3D: an input corner snaps to its
/// exact coordinates, anything else interpolates the dropped coordinate off
/// `plane` (exact at `plane`'s first corner).
fn lift(x: [f64; 2], axis: usize, plane: [P3; 3], corners: &[([f64; 2], P3)]) -> P3 {
    if let Some((_, c)) = corners.iter().find(|(q, _)| *q == x) {
        return *c;
    }
    // The dropped coordinate of the plane at `x`, by linear interpolation
    // over the projected triangle.
    let p = plane.map(|q| drop_axis(q, axis));
    let d = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
    let c = if d == 0.0 {
        plane[0][axis]
    } else {
        let u =
            ((x[0] - p[0][0]) * (p[2][1] - p[0][1]) - (x[1] - p[0][1]) * (p[2][0] - p[0][0])) / d;
        let v =
            ((x[1] - p[0][1]) * (p[1][0] - p[0][0]) - (x[0] - p[0][0]) * (p[1][1] - p[0][1])) / d;
        plane[0][axis]
            + u * (plane[1][axis] - plane[0][axis])
            + v * (plane[2][axis] - plane[0][axis])
    };
    let mut out = [c; 3];
    out[(axis + 1) % 3] = x[0];
    out[(axis + 2) % 3] = x[1];
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{box_mesh, build, fin};
    use fast_mesh_bvh::Quality;
    use fast_mesh_store::Mesh;

    /// The segments of `clip_pair` as a sorted set of endpoint pairs.
    fn clip_set(pa: [P3; 3], pb: [P3; 3]) -> Vec<([u64; 3], [u64; 3])> {
        let mut buf = [Seg64::EMPTY; 6];
        let n = clip_pair(pa, pb, &mut buf);
        assert!(n <= 6);
        let mut out: Vec<_> = buf[..n]
            .iter()
            .map(|s| {
                let (a, b) = (s.from.map(f64::to_bits), s.to.map(f64::to_bits));
                if a <= b { (a, b) } else { (b, a) }
            })
            .collect();
        out.sort_unstable();
        out
    }

    fn bits(p: [f64; 3]) -> [u64; 3] {
        p.map(f64::to_bits)
    }

    const A: [P3; 3] = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];

    #[test]
    fn clip_crossing_pair() {
        // B's plane is x = 0.5; the intersection runs along it.
        let b = [[0.5, -1.0, -1.0], [0.5, -1.0, 1.0], [0.5, 2.0, 0.0]];
        assert!(tri_tri_intersect(A, b));
        assert_eq!(
            clip_set(A, b),
            vec![(bits([0.5, 0.0, 0.0]), bits([0.5, 1.5, 0.0]))]
        );
        assert_eq!(clip_set(b, A), clip_set(A, b));
    }

    #[test]
    fn clip_coplanar_overlap() {
        // Overlap is the triangle (1,0), (2,0), (1,1) at z = 0.
        let b = [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 2.0, 0.0]];
        assert!(tri_tri_intersect(A, b));
        let (p, q, r) = (
            bits([1.0, 0.0, 0.0]),
            bits([2.0, 0.0, 0.0]),
            bits([1.0, 1.0, 0.0]),
        );
        assert_eq!(clip_set(A, b), vec![(p, r), (p, q), (r, q)]);
    }

    #[test]
    fn clip_stage_bails_out_past_six_vertices() {
        // A concave pentagon weaving across the clip line four times emits
        // 3 kept vertices plus 4 crossings: 7 outputs, past the 6-vertex
        // buffer. Exact signs on rounded crossings can produce such a
        // pattern from ulp-crafted meshes; the stage bails out with 0
        // instead of writing out of bounds.
        let poly = [[0.0, 1.0], [1.0, -1.0], [2.0, 1.0], [3.0, -1.0], [4.0, 1.0]];
        let mut next = [[0.0; 2]; 6];
        assert_eq!(
            clip_stage(&poly, [-10.0, 0.0], [10.0, 0.0], 1, &mut next),
            0
        );
    }

    #[test]
    fn clip_point_touch_gives_nothing() {
        let b = [[2.0, 0.0, 0.0], [3.0, 0.0, 1.0], [3.0, 1.0, 1.0]];
        assert!(tri_tri_intersect(A, b));
        assert!(clip_set(A, b).is_empty());
    }

    #[test]
    fn clip_degenerate_pairs() {
        // A segment piercing A meets it at one point: nothing.
        let pierce = [[0.5, 0.5, -1.0], [0.5, 0.5, 1.0], [0.5, 0.5, 1.0]];
        assert!(tri_tri_intersect(A, pierce));
        assert!(clip_set(A, pierce).is_empty());
        // A segment lying in A gives the overlap.
        let flat = [[0.25, 0.25, 0.0], [0.75, 0.75, 0.0], [0.5, 0.5, 0.0]];
        assert!(tri_tri_intersect(A, flat));
        assert_eq!(
            clip_set(A, flat),
            vec![(bits([0.25, 0.25, 0.0]), bits([0.75, 0.75, 0.0]))]
        );
    }

    /// Meshes A and B plus the identity, for the end-to-end tests.
    fn meshes(pa: &[f32], ia: &[u32], pb: &[f32], ib: &[u32]) -> (Mesh, Bvh, Mesh, Bvh) {
        let ma = Mesh::from_buffers(pa, ia).unwrap();
        let mb = Mesh::from_buffers(pb, ib).unwrap();
        let (ba, bb) = (build(&ma, Quality::Sah), build(&mb, Quality::Sah));
        (ma, ba, mb, bb)
    }

    fn run(ma: &Mesh, ba: &Bvh, mb: &Mesh, bb: &Bvh, xf: &Rigid) -> Vec<Polyline> {
        let mut out = Vec::new();
        let n = intersection_polylines(ma, ba, mb, bb, xf, &mut out).unwrap();
        assert_eq!(n, out.len());
        out
    }

    #[test]
    fn box_cut_by_sheet_is_a_closed_loop() {
        let ma = box_mesh([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let mb = Mesh::from_buffers(
            &[
                -2.0, -2.0, 0.0, 2.0, -2.0, 0.0, 2.0, 2.0, 0.0, -2.0, 2.0, 0.0,
            ],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap();
        let (ma_bvh, mb_bvh) = (build(&ma, Quality::Sah), build(&mb, Quality::Sah));
        let out = run(&ma, &ma_bvh, &mb, &mb_bvh, &Rigid::IDENTITY);
        assert_eq!(out.len(), 1);
        assert!(out[0].closed && out[0].manifold);
        // The four corners plus the midpoint of each side, where the sheet's
        // two triangles split the crossing: one loop proves they all welded.
        let mut pts = out[0].points.clone();
        pts.sort_by_key(|p| p.map(f32::to_bits));
        assert_eq!(
            pts,
            vec![
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0],
            ]
        );
    }

    #[test]
    fn coplanar_overlap_chains_its_boundary() {
        // Two overlapping quads in z = 0; the overlap [1,2]^2 chains to its
        // 4 boundary edges, the interior ones cancelling.
        let (ma, ba, mb, bb) = meshes(
            &[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0],
            &[0, 1, 2, 0, 2, 3],
            &[1.0, 1.0, 0.0, 3.0, 1.0, 0.0, 3.0, 3.0, 0.0, 1.0, 3.0, 0.0],
            &[0, 1, 2, 0, 2, 3],
        );
        let out = run(&ma, &ba, &mb, &bb, &Rigid::IDENTITY);
        assert_eq!(out.len(), 1);
        assert!(out[0].closed && out[0].manifold);
        let mut pts = out[0].points.clone();
        pts.sort_by_key(|p| p.map(f32::to_bits));
        assert_eq!(
            pts,
            vec![
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
            ]
        );
    }

    #[test]
    fn nonmanifold_junction_splits_chains() {
        // Three quads sharing an edge, cut by a sheet at z = 0.5: three open
        // chains meeting at the junction (0, 0, 0.5).
        let ma = fin();
        let mb = Mesh::from_buffers(
            &[
                -2.0, -2.0, 0.5, 2.0, -2.0, 0.5, 2.0, 2.0, 0.5, -2.0, 2.0, 0.5,
            ],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap();
        let (ba, bb) = (build(&ma, Quality::Sah), build(&mb, Quality::Sah));
        let out = run(&ma, &ba, &mb, &bb, &Rigid::IDENTITY);
        assert_eq!(out.len(), 3);
        for c in &out {
            assert!(!c.closed && !c.manifold);
            assert_eq!(c.points[0], [0.0, 0.0, 0.5]);
            assert_eq!(c.points.len(), 3);
        }
    }

    #[test]
    fn cancel_dupes_keeps_one_per_direction_group() {
        let (u, v, w) = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let seg = |from, to| Seg64 { from, to };
        let mut segs = vec![
            seg(u, v),
            seg(v, u), // reversed pair cancels
            seg(u, w),
            seg(u, w), // same direction collapses to one
            seg(v, w), // survivor
        ];
        cancel_dupes(&mut segs);
        let mut got: Vec<_> = segs.iter().map(|s| (bits(s.from), bits(s.to))).collect();
        got.sort_unstable();
        assert_eq!(got, vec![(bits(u), bits(w)), (bits(v), bits(w))]);
    }

    #[test]
    fn coplanar_edge_touch_gives_nothing() {
        // Two coplanar triangles sharing only an edge: a one-dimensional
        // overlap contributes nothing (only area overlaps do).
        let b = [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]];
        assert!(tri_tri_intersect(A, b));
        assert!(clip_set(A, b).is_empty());
    }

    #[test]
    fn segments_match_brute_force_and_lie_on_both_meshes() {
        use crate::test_util::Lcg;
        use crate::tri::closest_on_triangle;
        let ma = box_mesh([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let mut rng = Lcg(0x24);
        let mut total = 0;
        for round in 0..6 {
            // A small box crossing A, the last two rounds rotated and moved
            // by the transform instead of in mesh space.
            let xf = if round < 4 {
                Rigid::IDENTITY
            } else {
                Rigid {
                    rotation: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    translation: [0.5, -0.25, 0.0],
                }
            };
            let c = if round < 4 {
                [
                    (rng.next() * 2.0 - 1.0) * 0.8,
                    (rng.next() * 2.0 - 1.0) * 0.8,
                    (rng.next() * 2.0 - 1.0) * 0.8,
                ]
            } else {
                [0.0, 0.0, 0.0]
            };
            let mb = box_mesh(
                [c[0] - 0.4, c[1] - 0.4, c[2] - 0.4],
                [c[0] + 0.4, c[1] + 0.4, c[2] + 0.4],
            );
            let (ma_bvh, mb_bvh) = (build(&ma, Quality::Sah), build(&mb, Quality::Sah));
            let out = run(&ma, &ma_bvh, &mb, &mb_bvh, &xf);
            // Brute force over all pairs.
            let (pta, pta_t, ptb, ptb_t) = (
                ma.positions(),
                ma.triangles(),
                mb.positions(),
                mb.triangles(),
            );
            let mut brute: Vec<Seg64> = Vec::new();
            let mut buf = [Seg64::EMPTY; 6];
            for ta in pta_t {
                let pa = ta.map(|v| {
                    let p = pta[v as usize];
                    p.map(f64::from)
                });
                for tb in ptb_t {
                    let pb = tb.map(|v| xf.apply(ptb[v as usize]));
                    if !tri_tri_intersect(pa, pb) {
                        continue;
                    }
                    let n = clip_pair(pa, pb, &mut buf);
                    brute.extend(buf[..n].iter().filter(|s| s.from != s.to).copied());
                }
            }
            cancel_dupes(&mut brute);
            let mut want: Vec<_> = brute
                .iter()
                .map(|s| {
                    let (a, b) = (
                        s.from.map(|x| (x as f32).to_bits()),
                        s.to.map(|x| (x as f32).to_bits()),
                    );
                    if a <= b { (a, b) } else { (b, a) }
                })
                .filter(|(a, b)| a != b)
                .collect();
            want.sort_unstable();
            let mut got = Vec::new();
            for c in &out {
                // Every segment midpoint lies on both meshes.
                let n = c.points.len();
                let end = if c.closed { n } else { n - 1 };
                for i in 0..end {
                    let (p, q) = (c.points[i], c.points[(i + 1) % n]);
                    let (a, b) = (p.map(f32::to_bits), q.map(f32::to_bits));
                    got.push(if a <= b { (a, b) } else { (b, a) });
                    let m = std::array::from_fn(|k| f64::from(p[k] + q[k]) / 2.0);
                    let on_a = pta_t.iter().any(|t| {
                        let tri = t.map(|v| pta[v as usize].map(f64::from));
                        closest_on_triangle(m, tri).dist2 < 1e-10
                    });
                    let on_b = ptb_t.iter().any(|t| {
                        let tri = t.map(|v| xf.apply(ptb[v as usize]));
                        closest_on_triangle(m, tri).dist2 < 1e-10
                    });
                    assert!(on_a, "midpoint {m:?} off A");
                    assert!(on_b, "midpoint {m:?} off B");
                }
            }
            got.sort_unstable();
            assert_eq!(got, want);
            total += got.len();
        }
        assert!(total > 0, "no segments over all rounds");
    }

    #[test]
    fn vertex_touch_has_no_polylines() {
        let (ma, ba, mb, bb) = meshes(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[0, 1, 2],
            &[0.25, 0.25, 0.0, 1.0, 0.25, 1.0, 0.25, 1.0, 1.0],
            &[0, 1, 2],
        );
        assert!(run(&ma, &ba, &mb, &bb, &Rigid::IDENTITY).is_empty());
    }

    #[test]
    fn validation_errors_leave_out_alone() {
        let (ma, ba, mb, bb) = meshes(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[0, 1, 2],
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[0, 1, 2],
        );
        let bad = Rigid {
            rotation: [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0; 3],
        };
        let mut out = vec![Polyline::default()];
        let is_invalid = |e: QueryError| matches!(e, QueryError::InvalidInput(_));
        assert!(is_invalid(
            intersection_polylines(&ma, &ba, &mb, &bb, &bad, &mut out).unwrap_err()
        ));
        assert_eq!(out.len(), 1);
        // Validation comes before the mesh check.
        assert!(is_invalid(
            intersection_polylines(&ma, &bb, &mb, &bb, &bad, &mut out).unwrap_err()
        ));
        assert_eq!(
            intersection_polylines(&mb, &ba, &ma, &ba, &Rigid::IDENTITY, &mut out),
            Err(QueryError::MeshMismatch)
        );
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn out_of_memory_restores_out() {
        use crate::fail_allocations_after;
        let ma = box_mesh([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let mb = box_mesh([0.4, -0.5, -0.5], [1.4, 0.5, 0.5]);
        let (ba, bb) = (build(&ma, Quality::Sah), build(&mb, Quality::Sah));
        // Sanity: this shape intersects in a loop.
        assert!(!run(&ma, &ba, &mb, &bb, &Rigid::IDENTITY).is_empty());
        fail_allocations_after(Some(0));
        let mut out = Vec::new();
        assert_eq!(
            intersection_polylines(&ma, &ba, &mb, &bb, &Rigid::IDENTITY, &mut out),
            Err(QueryError::OutOfMemory)
        );
        assert!(out.is_empty());
        fail_allocations_after(None);
    }
}
