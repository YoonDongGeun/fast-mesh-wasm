//! The triangulation the heat method is solved on: the mesh's own, or an
//! intrinsic Delaunay triangulation of the same surface.
//!
//! A cotangent Laplacian is as good as the angles of its triangles. Across an
//! edge whose opposite angles sum to more than π its weight is negative: the
//! diffusion (which needs an M-matrix, for positive heat) has to clamp it,
//! and the Poisson step then answers for another surface. Triangles that are
//! large next to the time step spoil the heat's direction instead: a fan
//! around one vertex is too coarse to tell directions apart across it.
//!
//! Neither is a property of the surface, only of how it is cut into
//! triangles, so where the mesh has either it is cut again, without moving
//! any of it (Sharp, Soliman & Crane 2019, *Navigating Intrinsic
//! Triangulations*): an edge is not a chord any more but a straight line over
//! the faces, of which only the length is kept.
//!
//! - Edges are flipped until the triangulation is Delaunay (Bobenko &
//!   Springborn 2007): the weights are then positive, with nothing clamped.
//! - The edges that cannot be flipped, for there are not two triangles at
//!   them, are kept; one that a corner across from it sees at an obtuse angle
//!   is split at the foot of that corner.
//! - A triangle whose circumcircle is larger than the size asked for gets a
//!   vertex at its centre (Delaunay refinement, Chew 1993), the largest ones
//!   first, as long as the part it is in may grow.
//!
//! Triangles are laid out flat one at a time ([`Layout`]); those that still
//! are the mesh's own need none. All of it runs in order on one thread, but
//! for the passes that only look: a mesh that needs none of it, such as an
//! evenly triangulated one, is only looked at.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use fast_mesh_store::par;

use super::heat::{CHUNK, Corners, Flat, Numbered, bounds, by_first_corner, morton};
use crate::geom::{dot, norm, sub, triangle_normal, widen};
use crate::grid::{allocation_allowed, filled, push, reserve_exact};
use crate::{FieldsError, Result};

/// A negative weight counts from this share of its two cotangents on.
const TOLERANCE: f64 = 1e-5;
/// A coordinate's rounding error, relative to the largest of its part: edges
/// that rounding alone could have made negative are left alone (the diagonals
/// of rectangles, say).
const ROUNDING: f64 = 1.0 / (1u64 << 25) as f64;
/// Flips at most, per triangle. (A sphere drawn out twenty times takes two.)
const FLIPS: usize = 64;
/// A flip leaves each of its two triangles at least this share of their area.
const SLIVER: f64 = 1e-12;
/// A vertex this near an edge, in barycentric coordinates, goes on it.
const ON_EDGE: f64 = 1e-6;
/// Triangles on the way to a circumcentre at most.
const WALK: usize = 1000;
/// Sides looked at, or vertices put in, per step.
const STEP: usize = 1 << 16;
/// What is not there.
const NONE: u32 = u32::MAX;

const OOM: fn() -> FieldsError = || FieldsError::OutOfMemory;

/// A triangle laid out flat: side 0, from corner 0 to corner 1, along x from
/// the origin, and corner 2 at `(x, y)`, `y > 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Layout {
    pub(super) length: f64,
    pub(super) x: f64,
    pub(super) y: f64,
}

impl Layout {
    pub(super) fn of(corners: [[f32; 3]; 3]) -> Layout {
        let [a, b, c] = corners.map(widen);
        let ab = sub(b, a);
        let length = norm(ab);
        Layout {
            length,
            x: dot(sub(c, a), ab) / length,
            y: norm(triangle_normal(corners)) / length,
        }
    }

    /// The same triangle with the numbers of its corners moved on by `side`.
    fn turned(self, side: usize) -> Layout {
        let Layout { length, x, y } = self;
        match side {
            1 => {
                let l = (x - length).hypot(y);
                Layout {
                    length: l,
                    x: length * (length - x) / l,
                    y: length * y / l,
                }
            }
            2 => {
                let l = x.hypot(y);
                Layout {
                    length: l,
                    x: (x * (x - length) + y * y) / l,
                    y: length * y / l,
                }
            }
            _ => self,
        }
    }

    /// Half the cotangent of the angle at each corner.
    pub(super) fn half_cotangents(self) -> [f64; 3] {
        let Layout { length, x, y } = self;
        [0.5 * x / y, 0.5 * (length - x) / y, self.half_cotangent()]
    }

    /// Half the cotangent of the angle at corner 2, across from side 0.
    fn half_cotangent(self) -> f64 {
        let Layout { length, x, y } = self;
        0.5 * (x * (x - length) + y * y) / (length * y)
    }

    fn circumradius(self) -> f64 {
        let Layout { length, x, y } = self;
        0.5 * (x - length).hypot(y) * x.hypot(y) / y
    }

    fn circumcentre(self) -> [f64; 2] {
        let Layout { length, x, y } = self;
        [0.5 * length, 0.5 * (x * (x - length) + y * y) / y]
    }

    /// How far from zero the half cotangent across from side 0 still counts
    /// as zero, when the corners are off by about `noise`.
    fn slack(self, noise: f64) -> f64 {
        let Layout { length, x, y } = self;
        let c = self.half_cotangent().abs();
        TOLERANCE * c + noise * (length + x.abs() + y) / (length * y) * (1.0 + 2.0 * c)
    }
}

/// Whether the weight of an edge is negative by more than rounding. `own`
/// and `other` see it as side 0.
fn negative(own: Layout, other: Option<Layout>, noise: f64) -> bool {
    let (mut weight, mut slack) = (own.half_cotangent(), TOLERANCE + own.slack(noise));
    if let Some(other) = other {
        weight += other.half_cotangent();
        slack += other.slack(noise);
    }
    weight < -slack
}

/// What the triangles of a mesh are like.
#[derive(Debug, Clone, Copy)]
pub(super) struct Survey {
    pub(super) area: f64,
    /// The largest circumradius.
    pub(super) radius: f64,
    /// Every weight is positive, up to rounding.
    pub(super) delaunay: bool,
}

/// The largest coordinate of each part, times the rounding error.
fn noise(mesh: &Numbered) -> Result<Vec<f64>> {
    let mut noise = filled(mesh.parts(), 0.0f64)?;
    for (k, n) in noise.iter_mut().enumerate() {
        let part = mesh.starts[k] as usize..mesh.starts[k + 1] as usize;
        let largest = mesh.positions[part]
            .iter()
            .flatten()
            .fold(0.0f32, |m, c| m.max(c.abs()));
        *n = ROUNDING * f64::from(largest);
    }
    Ok(noise)
}

/// Looks at the triangles of a mesh, whose half cotangents are `cots`.
pub(super) fn survey(mesh: &Numbered, corners: &Corners, cots: &[[f64; 3]]) -> Result<Survey> {
    let noise = noise(mesh)?;
    allocation_allowed()?;
    let parts = par::map_range(mesh.triangles.len(), CHUNK, 2 * CHUNK, OOM, |range| {
        let mut s = Survey {
            area: 0.0,
            radius: 0.0,
            delaunay: true,
        };
        for t in range {
            let tri = mesh.triangles[t];
            let at = tri.map(|v| mesh.positions[v as usize]);
            s.area += 0.5 * norm(triangle_normal(at));
            let e = sub(widen(at[2]), widen(at[1]));
            let c = cots[t][0];
            s.radius = s.radius.max(dot(e, e) * (1.0 + 4.0 * c * c));
            if !s.delaunay {
                continue;
            }
            // Only an angle above a right one can make a weight negative.
            for k in 0..3 {
                if cots[t][k] >= 0.0 {
                    continue;
                }
                let side = 3 * t as u32 + (k as u32 + 1) % 3;
                let own = mesh.layout(t).turned((k + 1) % 3);
                let ring = corners.ring(&mesh.triangles, side);
                let other = match ring {
                    Ring::Two(other) if opposite(&mesh.triangles, side, other) => {
                        Some(mesh.layout(other as usize / 3).turned(other as usize % 3))
                    }
                    _ => None,
                };
                let noise = noise[mesh.part_of(tri[0])];
                if negative(own, other, noise) {
                    s.delaunay = false;
                }
            }
        }
        Ok::<Survey, FieldsError>(s)
    })?;
    let mut all = Survey {
        area: 0.0,
        radius: 0.0,
        delaunay: true,
    };
    for s in parts {
        all.area += s.area;
        all.radius = all.radius.max(s.radius);
        all.delaunay &= s.delaunay;
    }
    all.radius = 0.5 * all.radius.sqrt();
    Ok(all)
}

/// The other sides at the edge of a side.
enum Ring {
    /// None: the edge is a boundary.
    One,
    Two(u32),
    More,
}

impl Corners {
    /// Calls `f` with every side at the edge of `side`, itself among them, in
    /// ascending order.
    fn sides(&self, triangles: &[[u32; 3]], side: u32, mut f: impl FnMut(u32)) {
        let tri = triangles[side as usize / 3];
        let k = side as usize % 3;
        let (mut at, mut to) = (tri[k], tri[(k + 1) % 3]);
        if self.around(to).len() < self.around(at).len() {
            std::mem::swap(&mut at, &mut to);
        }
        for &corner in self.around(at) {
            let (t, j) = (corner as usize / 3, corner as usize % 3);
            if triangles[t][(j + 1) % 3] == to {
                f(corner);
            }
            if triangles[t][(j + 2) % 3] == to {
                f((3 * t + (j + 2) % 3) as u32);
            }
        }
    }

    fn ring(&self, triangles: &[[u32; 3]], side: u32) -> Ring {
        let mut ring = Ring::One;
        self.sides(triangles, side, |other| {
            if other != side {
                ring = match ring {
                    Ring::One => Ring::Two(other),
                    _ => Ring::More,
                };
            }
        });
        ring
    }

    /// The side after `side` at its edge: the next larger one, or else the
    /// smallest.
    fn next(&self, triangles: &[[u32; 3]], side: u32) -> u32 {
        let (mut first, mut next) = (side, NONE);
        self.sides(triangles, side, |other| {
            first = first.min(other);
            if other > side {
                next = next.min(other);
            }
        });
        if next == NONE { first } else { next }
    }
}

/// Whether two sides at an edge run along it in opposite directions.
fn opposite(triangles: &[[u32; 3]], a: u32, b: u32) -> bool {
    let (ta, tb) = (triangles[a as usize / 3], triangles[b as usize / 3]);
    let (ka, kb) = (a as usize % 3, b as usize % 3);
    ta[ka] == tb[(kb + 1) % 3] && ta[(ka + 1) % 3] == tb[kb]
}

/// A triangle that is too large.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Large {
    radius: f64,
    triangle: u32,
}

impl Eq for Large {}

impl Ord for Large {
    fn cmp(&self, other: &Self) -> Ordering {
        self.radius
            .total_cmp(&other.radius)
            .then(other.triangle.cmp(&self.triangle))
    }
}

impl PartialOrd for Large {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Where a walk ended.
enum Place {
    /// In a triangle, at these barycentric coordinates.
    Inside(usize, [f64; 3]),
    /// At a side that has nothing to walk on to behind it.
    Edge(u32),
    Nowhere,
}

/// A triangulation being made Delaunay.
///
/// A side is numbered `3 t + k`: it runs from corner `k` of triangle `t` to
/// the next, and corner `k + 2` is across from it.
#[derive(Debug)]
pub(super) struct Intrinsic {
    triangles: Vec<[u32; 3]>,
    /// Per side, the next one at the same edge, around in a ring.
    next: Vec<[u32; 3]>,
    /// Per triangle, its place in `layouts`, or [`NONE`] as long as it is one
    /// of the mesh.
    layout_of: Vec<u32>,
    layouts: Vec<Layout>,
    /// Where the vertices are; those put in only about.
    positions: Vec<[f32; 3]>,
    /// The mesh's vertex per vertex, part after part, and where each part's
    /// start. Those put in come after.
    vertices: Vec<u32>,
    starts: Vec<u32>,
    /// The part of every vertex put in.
    part_of_new: Vec<u32>,
    /// Per part, its rounding error.
    noise: Vec<f64>,
    /// Per part, how many triangles it may still gain.
    left: Vec<u32>,
    flips_left: usize,
    /// Sides whose weight may be negative.
    todo: Vec<u32>,
    stage: Stage,
    /// The triangles changed since the last vertex was put in.
    changed: Vec<u32>,
    large: BinaryHeap<Large>,
    /// What [`Intrinsic::renumber`] works in.
    links: Vec<[u32; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Stage {
    Flip,
    /// Put vertices into the triangles above this size.
    Refine(f64),
}

impl Intrinsic {
    /// The triangulation of a mesh as it is, whose half cotangents are
    /// `cots` and whose area is `area`. Each part may gain half its
    /// triangles, and `extra` ones shared out between the parts by area.
    pub(super) fn new(
        mesh: Numbered,
        corners: &Corners,
        cots: &[[f64; 3]],
        area: f64,
        extra: f64,
    ) -> Result<Intrinsic> {
        let count = mesh.triangles.len();
        let mut next = filled(count, [0u32; 3])?;
        par::for_each_chunk_mut(&mut next, CHUNK, 2 * CHUNK, |c, part| {
            for (i, out) in part.iter_mut().enumerate() {
                let t = 3 * (c * CHUNK + i) as u32;
                *out = [t, t + 1, t + 2].map(|side| corners.next(&mesh.triangles, side));
            }
        });

        let mut todo = Vec::new();
        let mut areas = filled(mesh.parts(), 0.0f64)?;
        let mut left = filled(mesh.parts(), 0u32)?;
        for (t, (tri, cots)) in mesh.triangles.iter().zip(cots).enumerate() {
            let part = mesh.part_of(tri[0]);
            left[part] += 1;
            if areas.len() > 1 {
                let at = tri.map(|v| mesh.positions[v as usize]);
                areas[part] += 0.5 * norm(triangle_normal(at));
            }
            for (k, &c) in cots.iter().enumerate() {
                // Only an angle above a right one can make a weight negative.
                if c < 0.0 {
                    push(&mut todo, (3 * t + (k + 1) % 3) as u32)?;
                }
            }
        }
        if let [all] = &mut areas[..] {
            *all = area;
        }
        let mut gain = 0usize;
        for (left, area_of) in left.iter_mut().zip(&areas) {
            *left = (f64::from(*left) / 2.0 + extra * area_of / area) as u32;
            gain += *left as usize;
        }
        Ok(Intrinsic {
            next,
            layout_of: filled(count, NONE)?,
            layouts: Vec::new(),
            part_of_new: Vec::new(),
            noise: noise(&mesh)?,
            left,
            flips_left: FLIPS.saturating_mul(count + gain),
            todo,
            stage: Stage::Flip,
            changed: Vec::new(),
            large: BinaryHeap::new(),
            links: Vec::new(),
            vertices: mesh.vertices,
            starts: mesh.starts,
            triangles: mesh.triangles,
            positions: mesh.positions,
        })
    }

    /// The mean length of the edges, as long as they are the mesh's.
    pub(super) fn mean_edge(&self) -> Result<f64> {
        allocation_allowed()?;
        let parts = par::map_range(self.triangles.len(), CHUNK, 2 * CHUNK, OOM, |range| {
            let (mut sum, mut count) = (0.0, 0usize);
            for side in 3 * range.start..3 * range.end {
                if self.is_first(side as u32) {
                    let ends = self.ends(side as u32);
                    let [a, b] = ends.map(|v| widen(self.positions[v as usize]));
                    sum += norm(sub(b, a));
                    count += 1;
                }
            }
            Ok::<(f64, usize), FieldsError>((sum, count))
        })?;
        let (mut sum, mut count) = (0.0, 0usize);
        for (s, c) in parts {
            sum += s;
            count += c;
        }
        Ok(if count > 0 { sum / count as f64 } else { 0.0 })
    }

    /// Whether a side is the smallest at its edge.
    fn is_first(&self, side: u32) -> bool {
        let mut at = self.next_of(side);
        while at > side {
            at = self.next_of(at);
        }
        at == side
    }

    /// Flips and splits about [`STEP`] edges; whether the triangulation is
    /// Delaunay.
    pub(super) fn flip(&mut self) -> Result<bool> {
        self.settle(STEP)
    }

    /// Starts to put vertices into the triangles with circumcircles larger
    /// than `size`.
    pub(super) fn begin_refining(&mut self, size: f64) -> Result<()> {
        self.stage = Stage::Refine(size);
        for t in 0..self.triangles.len() {
            self.consider(t as u32, size)?;
        }
        Ok(())
    }

    /// Puts in about [`STEP`] more; whether that was all.
    pub(super) fn refine(&mut self) -> Result<bool> {
        let Stage::Refine(size) = self.stage else {
            return Ok(true);
        };
        for _ in 0..STEP / 16 {
            while !self.settle(STEP)? {}
            while let Some(t) = self.changed.pop() {
                self.consider(t, size)?;
            }
            let Some(Large { radius, triangle }) = self.large.pop() else {
                return Ok(true);
            };
            let t = triangle as usize;
            if self.layout(t).circumradius() == radius && self.left[self.part(t)] >= 2 {
                self.insert_centre(t, size)?;
            }
        }
        Ok(false)
    }

    /// Queues a triangle if it is too large.
    fn consider(&mut self, t: u32, size: f64) -> Result<()> {
        let radius = self.layout(t as usize).circumradius();
        if radius > size && radius.is_finite() {
            if self.large.len() == self.large.capacity() {
                allocation_allowed()?;
                self.large.try_reserve(1).map_err(|_| OOM())?;
            }
            self.large.push(Large {
                radius,
                triangle: t,
            });
        }
        Ok(())
    }

    /// The triangulation, numbered like a mesh. The vertices are numbered
    /// again if there are new ones: `local_of` then changes.
    pub(super) fn finish(mut self, local_of: &mut [u32]) -> Result<Numbered> {
        self.next = Vec::new();
        self.todo = Vec::new();
        let (old, n) = (self.vertices.len(), self.positions.len());
        // The new number of every vertex.
        let mut to = Vec::new();
        if n > old {
            let (lo, hi) = bounds(&self.positions);
            let mut keys: Vec<(u64, u32)> = Vec::new();
            reserve_exact(&mut keys, n)?;
            keys.extend((0..n as u32).map(|v| (morton(self.positions[v as usize], lo, hi), v)));
            keys.sort_unstable();
            let part_of = |v: u32| match v.checked_sub(old as u32) {
                Some(new) => self.part_of_new[new as usize] as usize,
                None => self.starts.partition_point(|&s| s <= v) - 1,
            };
            let mut starts = filled(self.starts.len(), 0u32)?;
            for v in 0..n as u32 {
                starts[part_of(v) + 1] += 1;
            }
            for k in 1..starts.len() {
                starts[k] += starts[k - 1];
            }
            let mut at = filled(starts.len(), 0u32)?;
            at.copy_from_slice(&starts);
            to = filled(n, 0u32)?;
            let mut vertices = filled(n, NONE)?;
            let mut positions = filled(n, [0.0f32; 3])?;
            for (_, v) in keys {
                let part = part_of(v);
                let new = at[part];
                at[part] += 1;
                to[v as usize] = new;
                positions[new as usize] = self.positions[v as usize];
                if let Some(&slot) = self.vertices.get(v as usize) {
                    vertices[new as usize] = slot;
                    local_of[slot as usize] = new - starts[part];
                }
            }
            (self.vertices, self.starts, self.positions) = (vertices, starts, positions);
        }
        let new = |tri: [u32; 3]| match to.len() {
            0 => tri,
            _ => tri.map(|v| to[v as usize]),
        };
        let mut plain: Vec<[u32; 3]> = Vec::new();
        reserve_exact(&mut plain, self.triangles.len() - self.layouts.len())?;
        let mut flat: Vec<Flat> = Vec::new();
        reserve_exact(&mut flat, self.layouts.len())?;
        for (&tri, &layout) in self.triangles.iter().zip(&self.layout_of) {
            match self.layouts.get(layout as usize) {
                Some(&layout) => flat.push(Flat::new(new(tri), layout)),
                None => plain.push(new(tri)),
            }
        }
        Ok(Numbered {
            triangles: by_first_corner(&plain, n, |t| *t)?,
            flat: by_first_corner(&flat, n, |f| f.corners)?,
            vertices: self.vertices,
            starts: self.starts,
            positions: self.positions,
        })
    }

    fn layout(&self, t: usize) -> Layout {
        match self.layouts.get(self.layout_of[t] as usize) {
            Some(&layout) => layout,
            None => Layout::of(self.triangles[t].map(|v| self.positions[v as usize])),
        }
    }

    /// A triangle as seen from one of its sides.
    fn layout_from(&self, side: u32) -> Layout {
        self.layout(side as usize / 3).turned(side as usize % 3)
    }

    fn set_layout(&mut self, t: usize, layout: Layout) -> Result<()> {
        if let Some(own) = self.layouts.get_mut(self.layout_of[t] as usize) {
            *own = layout;
        } else {
            self.layout_of[t] = self.layouts.len() as u32;
            push(&mut self.layouts, layout)?;
        }
        Ok(())
    }

    fn add_triangle(&mut self, corners: [u32; 3], layout: Layout) -> Result<u32> {
        let t = self.triangles.len();
        if 3 * t as u64 + 2 >= u64::from(NONE) {
            return Err(FieldsError::OutOfMemory);
        }
        push(&mut self.triangles, corners)?;
        push(&mut self.next, [0; 3])?;
        push(&mut self.layout_of, NONE)?;
        self.set_layout(t, layout)?;
        Ok(t as u32)
    }

    fn add_vertex(&mut self, position: [f64; 3], part: usize) -> Result<u32> {
        let v = self.positions.len() as u32;
        push(&mut self.positions, position.map(|c| c as f32))?;
        push(&mut self.part_of_new, part as u32)?;
        Ok(v)
    }

    /// The part a triangle is in.
    fn part(&self, t: usize) -> usize {
        let v = self.triangles[t][0];
        match v.checked_sub(self.starts[self.starts.len() - 1]) {
            Some(new) => self.part_of_new[new as usize] as usize,
            None => self.starts.partition_point(|&s| s <= v) - 1,
        }
    }

    fn next_of(&self, side: u32) -> u32 {
        self.next[side as usize / 3][side as usize % 3]
    }

    fn set_next(&mut self, side: u32, to: u32) {
        self.next[side as usize / 3][side as usize % 3] = to;
    }

    /// The side before `side` at its edge.
    fn before(&self, side: u32) -> u32 {
        let mut at = side;
        loop {
            let to = self.next_of(at);
            if to == side {
                return at;
            }
            at = to;
        }
    }

    /// The vertices a side runs from and to.
    fn ends(&self, side: u32) -> [u32; 2] {
        let tri = self.triangles[side as usize / 3];
        let k = side as usize % 3;
        [tri[k], tri[(k + 1) % 3]]
    }

    /// The side across an edge that can be flipped or walked over: there are
    /// two triangles at it, next to each other.
    fn across(&self, side: u32) -> Option<u32> {
        let other = self.next_of(side);
        (other != side && self.next_of(other) == side && opposite(&self.triangles, side, other))
            .then_some(other)
    }

    /// Looks at up to `count` sides that may have negative weights, and
    /// flips or splits them; whether none are left.
    fn settle(&mut self, count: usize) -> Result<bool> {
        for _ in 0..count {
            let Some(side) = self.todo.pop() else {
                return Ok(true);
            };
            let own = self.layout_from(side);
            let noise = self.noise[self.part(side as usize / 3)];
            if let Some(other) = self.across(side) {
                if other / 3 != side / 3
                    && self.flips_left > 0
                    && negative(own, Some(self.layout_from(other)), noise)
                {
                    self.flip_edge(side, other)?;
                }
            } else if negative(own, None, noise) {
                // The foot of the corner across from it.
                self.split(side, own.x / own.length)?;
            }
        }
        Ok(self.todo.is_empty())
    }

    /// Queues the sides of a triangle that has changed.
    fn touch(&mut self, t: u32) -> Result<()> {
        for k in 0..3 {
            push(&mut self.todo, 3 * t + k)?;
        }
        if self.stage != Stage::Flip {
            push(&mut self.changed, t)?;
        }
        Ok(())
    }

    /// Gives sides new numbers, in the rings they are in: from the first to
    /// the second of each pair.
    fn renumber(&mut self, sides: &[(u32, u32)]) -> Result<()> {
        let new = |side: u32| {
            let found = sides.iter().find(|&&(old, _)| old == side);
            found.map_or(side, |&(_, new)| new)
        };
        let mut links = std::mem::take(&mut self.links);
        links.clear();
        for &(old, _) in sides {
            push(&mut links, [new(self.before(old)), new(self.next_of(old))])?;
        }
        for (&[before, next], &(_, to)) in links.iter().zip(sides) {
            self.set_next(before, to);
            self.set_next(to, next);
        }
        self.links = links;
        Ok(())
    }

    /// Flips the edge of two sides, unless that would leave a triangle
    /// without an area.
    fn flip_edge(&mut self, side: u32, other: u32) -> Result<()> {
        let (t, o) = (side / 3, other / 3);
        let turn = |s: u32, by: u32| 3 * (s / 3) + (s % 3 + by) % 3;
        // `a` to `b`, with `c` above and `d` below.
        let [a, b] = self.ends(side);
        let c = self.ends(turn(side, 2))[0];
        let d = self.ends(turn(other, 2))[0];
        let up = self.layout_from(side);
        let down = self.layout_from(other);
        let length = up.length;
        let (cx, cy) = (up.x, up.y);
        let (dx, dy) = (length - down.x, down.y);
        let left = cx * dy + dx * cy;
        let right = (length - cx) * dy + (length - dx) * cy;
        if !(left > SLIVER * (left + right) && right > SLIVER * (left + right)) {
            return Ok(());
        }
        self.flips_left -= 1;
        self.renumber(&[
            (turn(side, 1), 3 * o + 1),
            (turn(side, 2), 3 * t),
            (turn(other, 1), 3 * t + 1),
            (turn(other, 2), 3 * o),
        ])?;
        self.set_next(3 * t + 2, 3 * o + 2);
        self.set_next(3 * o + 2, 3 * t + 2);
        self.triangles[t as usize] = [c, a, d];
        self.triangles[o as usize] = [d, b, c];
        let ca = cx.hypot(cy);
        self.set_layout(
            t as usize,
            Layout {
                length: ca,
                x: (cx * (cx - dx) + cy * (cy + dy)) / ca,
                y: left / ca,
            },
        )?;
        let db = (length - dx).hypot(dy);
        self.set_layout(
            o as usize,
            Layout {
                length: db,
                x: ((length - dx) * (cx - dx) + dy * (cy + dy)) / db,
                y: right / db,
            },
        )?;
        self.touch(t)?;
        self.touch(o)
    }

    /// Splits the edge of a side at `at` of its length, and the triangles
    /// there, unless their part may not grow that much.
    fn split(&mut self, side: u32, at: f64) -> Result<()> {
        if !(at > 0.0 && at < 1.0) {
            return Ok(());
        }
        // Every side at the edge, and whether it runs the other way.
        let mut ring: Vec<(u32, bool)> = Vec::new();
        let paired = self.across(side);
        let from = self.ends(side)[0];
        let mut s = side;
        loop {
            let back = if paired.is_some() {
                s != side
            } else {
                self.ends(s)[0] != from
            };
            push(&mut ring, (s, back))?;
            s = self.next_of(s);
            if s == side {
                break;
            }
        }
        let part = self.part(side as usize / 3);
        let count = ring.len() as u32;
        // Triangles that meet themselves at the edge cannot be cut.
        let mut members = filled(ring.len(), 0u32)?;
        for (m, &(s, _)) in members.iter_mut().zip(&ring) {
            *m = s / 3;
        }
        members.sort_unstable();
        if self.left[part] < count || members.windows(2).any(|w| w[0] == w[1]) {
            return Ok(());
        }
        self.left[part] -= count;

        let [p, q] = self.ends(side).map(|v| widen(self.positions[v as usize]));
        let middle = std::array::from_fn(|k| p[k] + at * (q[k] - p[k]));
        let m = self.add_vertex(middle, part)?;
        // `(a, b, c)` becomes `(a, m, c)`, in its place, and `(m, b, c)`.
        let mut renamed: Vec<(u32, u32)> = Vec::new();
        reserve_exact(&mut renamed, 2 * ring.len())?;
        // The sides from the edge's start to `m`, and from there on.
        let mut halves: Vec<[u32; 2]> = Vec::new();
        reserve_exact(&mut halves, ring.len())?;
        for &(s, back) in &ring {
            let (t, k) = (s / 3, s % 3);
            let [a, b] = self.ends(s);
            let c = self.ends(3 * t + (k + 2) % 3)[0];
            let Layout { length, x, y } = self.layout_from(s);
            let cut = length * if back { 1.0 - at } else { at };
            let u = self.add_triangle(
                [m, b, c],
                Layout {
                    length: length - cut,
                    x: x - cut,
                    y,
                },
            )?;
            push(&mut renamed, (3 * t + (k + 1) % 3, 3 * u + 1))?;
            push(&mut renamed, (3 * t + (k + 2) % 3, 3 * t + 2))?;
            push(
                &mut halves,
                if back { [3 * u, 3 * t] } else { [3 * t, 3 * u] },
            )?;
            self.triangles[t as usize] = [a, m, c];
            self.set_layout(t as usize, Layout { length: cut, x, y })?;
        }
        self.renumber(&renamed)?;
        for (i, &(s, _)) in ring.iter().enumerate() {
            let (t, u) = (s / 3, renamed[2 * i].1 / 3);
            self.set_next(3 * t + 1, 3 * u + 2);
            self.set_next(3 * u + 2, 3 * t + 1);
            for half in 0..2 {
                self.set_next(halves[i][half], halves[(i + 1) % halves.len()][half]);
            }
            self.touch(t)?;
            self.touch(u)?;
        }
        Ok(())
    }

    /// Puts a vertex at the circumcentre of a triangle, or splits the edge
    /// in the way.
    fn insert_centre(&mut self, t: usize, size: f64) -> Result<()> {
        match self.walk(t) {
            Place::Inside(t, at) => {
                if let Some(k) = (0..3).find(|&k| at[k] < ON_EDGE) {
                    let (a, b) = (at[(k + 1) % 3], at[(k + 2) % 3]);
                    return self.split((3 * t + (k + 1) % 3) as u32, b / (a + b));
                }
                self.insert(t, at)
            }
            Place::Edge(side) if self.layout_from(side).length > size => self.split(side, 0.5),
            _ => Ok(()),
        }
    }

    /// Walks from a triangle to its circumcentre.
    fn walk(&self, t: usize) -> Place {
        let mut at = self.layout(t).circumcentre();
        // The side whose frame `at` is in, that it came in by from the second
        // triangle on.
        let mut side = 3 * t as u32;
        for step in 0..WALK {
            let t = side / 3;
            let Layout { length, x, y } = self.layout_from(side);
            let c = at[1] / y;
            let b = (at[0] - c * x) / length;
            let a = 1.0 - b - c;
            if !(a.is_finite() && b.is_finite()) {
                return Place::Nowhere;
            }
            // Its sides, by the corners across from them.
            let mut out = (-1e-12, NONE);
            for (k, inside) in [(0, c), (1, a), (2, b)] {
                if inside < out.0 && (k > 0 || step == 0) {
                    out = (inside, k);
                }
            }
            let k = side as usize % 3;
            if out.1 == NONE {
                let mut corners = [0.0; 3];
                for (j, w) in [a, b, c].into_iter().enumerate() {
                    corners[(k + j) % 3] = w.max(0.0);
                }
                return Place::Inside(t as usize, corners);
            }
            let exit = 3 * t + (side % 3 + out.1) % 3;
            let Some(other) = self.across(exit) else {
                return Place::Edge(exit);
            };
            // From the frame of `side` to that of `exit`, and across.
            let (from, to) = match out.1 {
                0 => ([0.0, 0.0], [length, 0.0]),
                1 => ([length, 0.0], [x, y]),
                _ => ([x, y], [0.0, 0.0]),
            };
            let e = [to[0] - from[0], to[1] - from[1]];
            let l = e[0].hypot(e[1]);
            let d = [at[0] - from[0], at[1] - from[1]];
            at = [
                l - (d[0] * e[0] + d[1] * e[1]) / l,
                (d[0] * e[1] - d[1] * e[0]) / l,
            ];
            side = other;
        }
        Place::Nowhere
    }

    /// Puts a vertex into a triangle.
    fn insert(&mut self, t: usize, at: [f64; 3]) -> Result<()> {
        let part = self.part(t);
        if self.left[part] < 2 {
            return Ok(());
        }
        self.left[part] -= 2;
        let corners = self.triangles[t];
        let p = corners.map(|v| widen(self.positions[v as usize]));
        let position = std::array::from_fn(|k| at[0] * p[0][k] + at[1] * p[1][k] + at[2] * p[2][k]);
        let m = self.add_vertex(position, part)?;
        let old = self.layout(t);
        // The triangle of side `k` and the vertex.
        let layouts = [0, 1, 2].map(|k| {
            let Layout { length, x, y } = old.turned(k);
            let (b, c) = (at[(k + 1) % 3], at[(k + 2) % 3]);
            Layout {
                length,
                x: b * length + c * x,
                y: c * y,
            }
        });
        let first = t as u32;
        let second = self.add_triangle([corners[1], corners[2], m], layouts[1])?;
        let third = self.add_triangle([corners[2], corners[0], m], layouts[2])?;
        self.triangles[t] = [corners[0], corners[1], m];
        self.set_layout(t, layouts[0])?;
        let all = [first, second, third];
        self.renumber(&[
            (3 * first, 3 * first),
            (3 * first + 1, 3 * second),
            (3 * first + 2, 3 * third),
        ])?;
        for k in 0..3 {
            let (one, two) = (3 * all[k] + 1, 3 * all[(k + 1) % 3] + 2);
            self.set_next(one, two);
            self.set_next(two, one);
        }
        for t in all {
            self.touch(t)?;
        }
        Ok(())
    }
}
