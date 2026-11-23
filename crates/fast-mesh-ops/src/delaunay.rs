//! Delaunay edge flips on the nearly flat parts of a surface.
//!
//! [`delaunay_flips`] flips an edge shared by two triangles to the other
//! diagonal of their quad when
//!
//! - the two triangles are within [`DelaunayOptions::dihedral_limit_deg`]
//!   of coplanar, and so are the two new ones;
//! - the original triangles the quad covers (earlier flips included) are
//!   within the limit of each other and of both new triangles;
//! - the edge is not Delaunay (the two angles facing it add up to more than
//!   180 degrees) and the new one is;
//! - the smallest of the quad's six angles gets strictly larger;
//! - the new edge does not exist yet, so the mesh stays manifold, and
//!   neither new triangle turns over.
//!
//! So the limit holds against the original surface, however many flips
//! follow each other: a new triangle's normal is within it of the normal of
//! every original triangle it replaces, and an edge that was sharper than
//! it at the start (a crease, a corner) never flips. Each triangle carries
//! a cone that bounds the original normals it covers, so the check costs
//! the same however deep the flips go.
//!
//! Vertices never move, so the triangles only change shape: towards the
//! Delaunay triangulation of each flat patch, which has the largest
//! smallest angle. Every flip raises the smallest angle of its quad, so the
//! smallest angle of the region never drops, and the flips come to an end.
//!
//! The work runs in passes. A pass checks the edges in question (every
//! edge first, then those of the triangles the last pass flipped), in
//! parallel with the `parallel` feature, then flips the ones that qualify
//! in edge order, skipping any whose triangles an earlier flip of the pass
//! already changed (the next pass looks at it again). The result is
//! bit-identical with and without the feature.
//!
//! Only edges of exactly two triangles that wind it in opposite directions
//! flip, and only around vertices whose triangles form one fan. The flips
//! are one [`Transaction`] edit of the changed triangles; positions and the
//! UV channel are kept.

use std::f64::consts::PI;

use fast_mesh_store::{Transaction, TriangleId, VertexId, par};

use crate::denoise::{
    CHUNK, NONE, PAR_MIN, SurfaceStats, angle, concat, corners_of, in_mask, min_angle, oom, points,
    region_mask, stats, twins, unit_normal,
};
use crate::remesh::{cross3, dot3, sub3};
use crate::topology::VertexFans;
use crate::{OpsError, Result, allocation_allowed, filled, push, reserve};

/// How far past 180 degrees the facing angles must add up (as a sine)
/// before an edge counts as not Delaunay; cocircular quads stay as they are.
const DELAUNAY_EPS: f64 = 1e-9;
/// Slack on the dihedral limit, in radians, for rounding.
const LIMIT_EPS: f64 = 1e-9;
/// The most passes [`delaunay_flips`] runs.
pub const MAX_DELAUNAY_PASSES: usize = 1000;

/// Options for [`delaunay_flips`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DelaunayOptions {
    /// How far, in degrees in `[0, 90]`, flips may take the surface from
    /// the original: the largest angle across a flipped edge, and between
    /// a new triangle's normal and those of the original triangles it
    /// replaces (see the [module docs](self)). Default 10.
    pub dihedral_limit_deg: f32,
}

impl Default for DelaunayOptions {
    fn default() -> Self {
        Self {
            dihedral_limit_deg: 10.0,
        }
    }
}

/// Outcome of [`delaunay_flips`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DelaunayReport {
    /// Edges flipped.
    pub flips: usize,
    /// Passes that flipped at least one edge.
    pub passes: usize,
    /// `false` when [`MAX_DELAUNAY_PASSES`] ran out with flips left.
    pub converged: bool,
    /// The region's [`SurfaceStats`] before.
    pub before: SurfaceStats,
    /// The region's [`SurfaceStats`] after.
    pub after: SurfaceStats,
}

/// Flips the edges of the surface in `tx` towards a Delaunay triangulation
/// where its triangles are nearly coplanar (see the [module docs](self)),
/// on the edges between two triangles of `region` (triangle ids; `None`
/// for the whole mesh).
///
/// # Errors
/// [`OpsError::InvalidInput`] for a dihedral limit outside `[0, 90]` or a
/// dead or out-of-range triangle in `region`; [`OpsError::Store`] when out
/// of memory. Nothing is written on error.
pub fn delaunay_flips(
    tx: &mut Transaction<'_>,
    options: &DelaunayOptions,
    region: Option<&[TriangleId]>,
) -> Result<DelaunayReport> {
    let limit = options.dihedral_limit_deg;
    if !(0.0..=90.0).contains(&limit) {
        return Err(OpsError::InvalidInput(
            "delaunay dihedral limit must be in [0, 90] degrees",
        ));
    }
    let mesh = tx.mesh();
    let mask = region_mask(mesh, region)?;
    let mask = mask.as_deref();
    let positions = mesh.positions();
    let mut corners = corners_of(mesh)?;
    let (mut twin, single) = {
        let fans = VertexFans::build(mesh)?;
        let twin = twins(&corners, &fans)?;
        let single = single_fans(&corners, &twin, &fans, positions.len())?;
        (twin, single)
    };
    let before = stats(positions, &corners, &twin, mask)?;
    let mut original = Vec::new();
    reserve(&mut original, corners.len())?;
    original.extend_from_slice(&corners);
    let mut cones = filled(corners.len(), Cone::default())?;
    par::for_each_chunk_mut(&mut cones, CHUNK, PAR_MIN, |i, chunk| {
        for (j, cone) in chunk.iter_mut().enumerate() {
            let c = corners[i * CHUNK + j];
            if c[0] != NONE
                && let Some(axis) = unit_normal(points(positions, c))
            {
                *cone = Cone { axis, half: 0.0 };
            }
        }
    });

    let mut state = Flips {
        positions,
        corners: &mut corners,
        twin: &mut twin,
        cones: &mut cones,
        mask,
        single: &single,
        limit: f64::from(limit).to_radians() + LIMIT_EPS,
    };
    let mut report = DelaunayReport {
        before,
        converged: true,
        ..DelaunayReport::default()
    };
    let mut stamp = filled(original.len(), 0u32)?;
    let mut frontier: Vec<u32> = Vec::new();
    for pass in 0..=MAX_DELAUNAY_PASSES {
        let candidates = if pass == 0 {
            state.scan_all()?
        } else {
            state.scan_frontier(&frontier)?
        };
        frontier.clear();
        let mut flipped = 0;
        let mark = pass as u32 + 1;
        for h in candidates {
            let h = h as usize;
            let t = h / 3;
            let u = state.twin[h] as usize / 3;
            // Changed earlier in this pass: the next pass checks it again.
            if stamp[t] == mark || stamp[u] == mark {
                continue;
            }
            if state.flip_exists(h) {
                // A flip may yet take that edge away: look again next pass.
                push(&mut frontier, t as u32)?;
                continue;
            }
            if pass == MAX_DELAUNAY_PASSES {
                report.converged = false;
                break;
            }
            state.flip(h);
            stamp[t] = mark;
            stamp[u] = mark;
            push(&mut frontier, t as u32)?;
            push(&mut frontier, u as u32)?;
            flipped += 1;
        }
        if flipped == 0 {
            break;
        }
        report.flips += flipped;
        report.passes += 1;
    }

    report.after = stats(positions, state.corners, state.twin, mask)?;
    let changed = (0..original.len())
        .filter(|&t| !same_triangle(corners[t], original[t]))
        .count();
    let mut ids: Vec<TriangleId> = Vec::new();
    let mut tris: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut ids, changed)?;
    reserve(&mut tris, changed)?;
    for t in 0..original.len() {
        if !same_triangle(corners[t], original[t]) {
            ids.push(t as TriangleId);
            tris.push(corners[t]);
        }
    }
    if !ids.is_empty() {
        tx.set_triangles(&ids, &tris)?;
    }
    Ok(report)
}

/// Whether `x` is `y`, possibly starting from another corner.
fn same_triangle(x: [u32; 3], y: [u32; 3]) -> bool {
    x == y || x == [y[1], y[2], y[0]] || x == [y[2], y[0], y[1]]
}

/// Whether each vertex's triangles form one fan across twinned edges, so
/// that walking around it from any of them reaches them all.
fn single_fans(
    corners: &[[u32; 3]],
    twin: &[u32],
    fans: &VertexFans,
    slots: usize,
) -> Result<Vec<bool>> {
    let mut single = filled(slots, false)?;
    par::for_each_chunk_mut(&mut single, CHUNK, PAR_MIN, |i, chunk| {
        for (j, s) in chunk.iter_mut().enumerate() {
            let v = (i * CHUNK + j) as VertexId;
            let fan = fans.fan(v);
            if let Some(&first) = fan.first() {
                let (count, stopped) = around(corners, twin, first as usize, v, |_| false);
                *s = !stopped && count == fan.len();
            }
        }
    });
    Ok(single)
}

/// Visits the triangles around `v` reachable from `start` (which uses `v`)
/// across twinned edges, `start` first, until `f` returns `true`. Returns
/// how many it visited and whether it stopped early.
fn around(
    corners: &[[u32; 3]],
    twin: &[u32],
    start: usize,
    v: u32,
    mut f: impl FnMut(usize) -> bool,
) -> (usize, bool) {
    if f(start) {
        return (1, true);
    }
    let cap = corners.len();
    let mut count = 1;
    let k0 = (0..3).find(|&k| corners[start][k] == v).unwrap_or(0);
    // Forward, across the edge leaving `v` in each triangle.
    let (mut t, mut k) = (start, k0);
    loop {
        let tw = twin[3 * t + k];
        if tw == NONE {
            break;
        }
        let (u, kk) = (tw as usize / 3, tw as usize % 3);
        if u == start {
            return (count, false);
        }
        count += 1;
        if count > cap || f(u) {
            return (count, true);
        }
        (t, k) = (u, (kk + 1) % 3);
    }
    // Backward, across the edge entering `v`.
    let (mut t, mut k) = (start, k0);
    loop {
        let tw = twin[3 * t + (k + 2) % 3];
        if tw == NONE {
            return (count, false);
        }
        let (u, kk) = (tw as usize / 3, tw as usize % 3);
        count += 1;
        if count > cap || f(u) {
            return (count, true);
        }
        (t, k) = (u, kk);
    }
}

/// The directions within `half` radians of the unit `axis`: a bound on the
/// original normals a working triangle covers.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Cone {
    axis: [f64; 3],
    half: f64,
}

impl Cone {
    /// The smallest cone that holds both.
    fn merge(self, other: Cone) -> Cone {
        let between = angle(self.axis, other.axis);
        if between + other.half <= self.half {
            return self;
        }
        if between + self.half <= other.half {
            return other;
        }
        // This axis, turned towards the other by what the angle grew.
        let half = 0.5 * (between + self.half + other.half);
        let turn = half - self.half;
        let sin = between.sin();
        if sin <= 0.0 {
            return Cone { half, ..self };
        }
        let (x, y) = ((between - turn).sin() / sin, turn.sin() / sin);
        let mixed = [0, 1, 2].map(|k| x * self.axis[k] + y * other.axis[k]);
        let len = dot3(mixed, mixed).sqrt();
        Cone {
            axis: mixed.map(|m| m / len),
            half,
        }
    }

    /// Whether the cone's directions are within `limit` of each other and
    /// of `n`.
    fn within(self, n: [f64; 3], limit: f64) -> bool {
        2.0 * self.half <= limit && angle(self.axis, n) + self.half <= limit
    }
}

/// The working triangulation.
struct Flips<'a> {
    positions: &'a [[f32; 3]],
    corners: &'a mut [[u32; 3]],
    twin: &'a mut [u32],
    /// Per triangle, the original normals it covers.
    cones: &'a mut [Cone],
    mask: Option<&'a [bool]>,
    single: &'a [bool],
    /// The dihedral limit in radians, with its slack.
    limit: f64,
}

impl Flips<'_> {
    /// The quad of half-edge `h` (`a -> b` in its triangle, apex `c`, and
    /// `d` across): `[a, b, c, d]`.
    fn quad(&self, h: usize) -> [u32; 4] {
        let (t, k) = (h / 3, h % 3);
        let tw = self.twin[h] as usize;
        let c = self.corners[t];
        [
            c[k],
            c[(k + 1) % 3],
            c[(k + 2) % 3],
            self.corners[tw / 3][(tw % 3 + 2) % 3],
        ]
    }

    /// Whether half-edge `h` (with a twin) passes every test but the new
    /// edge's existence.
    fn wants(&self, h: usize) -> bool {
        let (t, u) = (h / 3, self.twin[h] as usize / 3);
        if !in_mask(self.mask, t) || !in_mask(self.mask, u) {
            return false;
        }
        let [a, b, c, d] = self.quad(h);
        if c == d || !self.single[c as usize] {
            return false;
        }
        let p = [a, b, c, d].map(|v| self.positions[v as usize].map(f64::from));
        improves(p, self.limit, || self.cones[t].merge(self.cones[u]))
    }

    /// Every edge that [`Flips::wants`] to flip, by its lower half-edge.
    fn scan_all(&self) -> Result<Vec<u32>> {
        allocation_allowed()?;
        let parts = par::map_range(self.corners.len(), CHUNK, PAR_MIN, oom, |r| {
            let mut out: Vec<u32> = Vec::new();
            for h in 3 * r.start..3 * r.end {
                let tw = self.twin[h];
                if tw != NONE && tw as usize > h && self.wants(h) {
                    push(&mut out, h as u32)?;
                }
            }
            Ok(out)
        })?;
        concat(parts)
    }

    /// The edges of the triangles in `frontier` that [`Flips::wants`] to
    /// flip, by their lower half-edges, ascending.
    fn scan_frontier(&self, frontier: &[u32]) -> Result<Vec<u32>> {
        let mut edges: Vec<u32> = Vec::new();
        reserve(&mut edges, 3 * frontier.len())?;
        for &t in frontier {
            for h in 3 * t..3 * t + 3 {
                let tw = self.twin[h as usize];
                if tw != NONE {
                    edges.push(h.min(tw));
                }
            }
        }
        edges.sort_unstable();
        edges.dedup();
        allocation_allowed()?;
        let parts = par::map_chunks(&edges, CHUNK, PAR_MIN, oom, |_, hs| {
            let mut out: Vec<u32> = Vec::new();
            for &h in hs {
                if self.wants(h as usize) {
                    push(&mut out, h)?;
                }
            }
            Ok(out)
        })?;
        concat(parts)
    }

    /// Whether the edge a flip of `h` would make already exists: some
    /// triangle around `c` uses `d`.
    fn flip_exists(&self, h: usize) -> bool {
        let [_, _, c, d] = self.quad(h);
        let corners = &*self.corners;
        around(corners, self.twin, h / 3, c, |t| corners[t].contains(&d)).1
    }

    /// Flips half-edge `h` (`a -> b`, apex `c`, `d` across): its triangles
    /// become `[a, d, c]` and `[d, b, c]`, and the twins follow.
    fn flip(&mut self, h: usize) {
        let (t, k) = (h / 3, h % 3);
        let tw = self.twin[h] as usize;
        let (u, kk) = (tw / 3, tw % 3);
        let [a, b, c, d] = self.quad(h);
        let x_bc = self.twin[3 * t + (k + 1) % 3];
        let x_ca = self.twin[3 * t + (k + 2) % 3];
        let x_ad = self.twin[3 * u + (kk + 1) % 3];
        let x_db = self.twin[3 * u + (kk + 2) % 3];
        let cone = self.cones[t].merge(self.cones[u]);
        self.cones[t] = cone;
        self.cones[u] = cone;
        self.corners[t] = [a, d, c];
        self.corners[u] = [d, b, c];
        let (t3, u3) = (3 * t as u32, 3 * u as u32);
        self.twin[3 * t..3 * t + 3].copy_from_slice(&[x_ad, u3 + 2, x_ca]);
        self.twin[3 * u..3 * u + 3].copy_from_slice(&[x_db, x_bc, t3 + 1]);
        for (x, to) in [(x_ad, t3), (x_ca, t3 + 2), (x_db, u3), (x_bc, u3 + 1)] {
            if x != NONE {
                self.twin[x as usize] = to;
            }
        }
    }
}

/// The geometric tests of the [module docs](self) for flipping the edge
/// `p[0] -> p[1]` of triangle `[p0, p1, p2]`, with `p[3]` across; `cone`
/// bounds the original normals its two triangles cover.
fn improves(p: [[f64; 3]; 4], limit: f64, cone: impl FnOnce() -> Cone) -> bool {
    let [a, b, c, d] = p;
    if !facing_exceeds(p) {
        return false;
    }
    let (Some(n0), Some(n1)) = (unit_normal([a, b, c]), unit_normal([b, a, d])) else {
        return false;
    };
    if angle(n0, n1) > limit {
        return false;
    }
    let (Some(m0), Some(m1)) = (unit_normal([a, d, c]), unit_normal([d, b, c])) else {
        return false;
    };
    // A quad that is not convex turns one of the new triangles over.
    if [(m0, n0), (m0, n1), (m1, n0), (m1, n1)]
        .iter()
        .any(|&(m, n)| dot3(m, n) <= 0.0)
        || angle(m0, m1) > limit
    {
        return false;
    }
    let cone = cone();
    if !cone.within(m0, limit) || !cone.within(m1, limit) {
        return false;
    }
    if angle(sub3(d, a), sub3(c, a)) + angle(sub3(d, b), sub3(c, b)) > PI {
        return false;
    }
    let old = min_angle([a, b, c]).min(min_angle([b, a, d]));
    let new = min_angle([a, d, c]).min(min_angle([d, b, c]));
    new > old
}

/// Whether the angles at `c` and `d` facing the edge `a`-`b` add up to more
/// than 180 degrees: the sine of their sum is negative.
fn facing_exceeds([a, b, c, d]: [[f64; 3]; 4]) -> bool {
    let (ca, cb, da, db) = (sub3(a, c), sub3(b, c), sub3(a, d), sub3(b, d));
    let (nc, nd) = (cross3(ca, cb), cross3(da, db));
    let (sc, kc) = (dot3(nc, nc).sqrt(), dot3(ca, cb));
    let (sd, kd) = (dot3(nd, nd).sqrt(), dot3(da, db));
    let scale = sc.hypot(kc) * sd.hypot(kd);
    sc * kd + kc * sd < -DELAUNAY_EPS * scale
}

#[cfg(test)]
mod tests;
