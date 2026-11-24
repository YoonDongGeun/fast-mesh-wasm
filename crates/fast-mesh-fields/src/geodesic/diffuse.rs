//! The diffusion step of the heat method, solved outwards from the sources.
//!
//! Heat falls off by about `e` per `√t`, so on a large mesh it spans hundreds
//! of orders of magnitude, and past `f64`'s range on a very large one. A
//! solver that stops at a residual relative to the sources leaves nothing but
//! noise beyond a few dozen edges, and the method needs the heat's direction
//! everywhere. So the mesh is cut into bands by distance from the sources,
//! each solved from the heat that arrives from the band before it, to a
//! residual relative to that heat and in its own power-of-two scale.
//!
//! A band is solved with [`OVERLAP`] more `√t` beyond it, of which only the
//! band itself is kept: what is cut off further out comes back damped twice
//! over the overlap.
//!
//! The distance that cuts the bands is an estimate of the heat's fall along
//! the mesh edges. It only decides what is solved together, not what is
//! solved, but where the estimate is far off (a fan of long triangles carries
//! heat much further than its size suggests) a band misses heat that comes
//! back from a later one. So the solution is checked at every vertex, and
//! unless it holds the bands are solved again, now cut by the heat found and
//! with what the rest of the mesh sends back, until it does.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use fast_mesh_store::par;

use super::amg::{BLOCK, Matrix, PAR_MIN_ROWS, Rows, dot};
use crate::geom::{norm, sub, widen};
use crate::grid::{allocation_allowed, filled, push, reserve};
use crate::{FieldsError, Result};

/// Width of a band, in `√t`: heat falls to about `e⁻¹⁰` across it.
const WIDTH: f64 = 10.0;
/// How far past its band each solve reaches, in `√t`.
const OVERLAP: f64 = 6.0;
/// A band's solve ends at this residual, relative to the heat that arrives.
const TOLERANCE: f64 = 1e-12;
/// The heat holds at a vertex when it differs from what its neighbours give
/// it by at most this share: its direction is then off by about as much.
const HOLDS: f64 = 1e-3;
/// A later pass leaves a band alone that differs by at most this share.
const SETTLED: f64 = 1e-4;
/// The heat holds when it does at all but this share of the vertices.
const EXCEPTIONS: f64 = 1e-3;
/// How often the bands are solved at most.
const PASSES: usize = 6;
/// An edge is long from this length on, in `√t`.
const LONG: f64 = 3.0;
/// No place in the band.
const NONE: u32 = u32::MAX;
/// In the band, without a place yet.
const UNPLACED: u32 = u32::MAX - 1;

const OOM: fn() -> FieldsError = || FieldsError::OutOfMemory;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct HeapItem {
    pub(super) dist: f64,
    pub(super) vertex: u32,
}

impl Eq for HeapItem {}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.vertex.cmp(&self.vertex))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(super) fn heap_push(heap: &mut BinaryHeap<HeapItem>, item: HeapItem) -> Result<()> {
    if heap.len() == heap.capacity() {
        allocation_allowed()?;
        heap.try_reserve(1).map_err(|_| FieldsError::OutOfMemory)?;
    }
    heap.push(item);
    Ok(())
}

/// The heat of every vertex as `value · 2^scale`.
#[derive(Debug)]
pub(super) struct Heat {
    pub(super) value: Vec<f64>,
    pub(super) scale: Vec<i32>,
}

impl Heat {
    /// The heat of three vertices over a common power of two.
    pub(super) fn corners(&self, v: [usize; 3]) -> [f64; 3] {
        let s = v.map(|i| self.scale[i]);
        let top = s[0].max(s[1]).max(s[2]);
        if s == [top; 3] {
            return v.map(|i| self.value[i]);
        }
        std::array::from_fn(|k| self.value[v[k]] * power_of_two(s[k] - top))
    }

    /// The natural logarithm of a vertex's heat, `-∞` where there is none.
    pub(super) fn ln(&self, v: usize) -> f64 {
        self.value[v].ln() + f64::from(self.scale[v]) * std::f64::consts::LN_2
    }
}

/// `2^e`, zero below the range of normal numbers.
fn power_of_two(e: i32) -> f64 {
    match e {
        -1022..=1023 => f64::from_bits(((e + 1023) as u64) << 52),
        i32::MIN..-1022 => 0.0,
        _ => f64::INFINITY,
    }
}

/// The `e` with `2^e <= x < 2^(e+1)`, for a positive normal `x`.
fn exponent_of(x: f64) -> i32 {
    ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// What the diffusion runs on.
pub(super) struct Surface<'a> {
    /// The Laplacian.
    pub(super) matrix: &'a Matrix,
    /// The lumped mass per vertex.
    pub(super) mass: &'a [f64],
    pub(super) positions: &'a [[f32; 3]],
    pub(super) time_step: f64,
}

/// The diffusion under way.
#[derive(Debug)]
pub(super) struct Diffusion {
    heat: Heat,
    /// How far the heat has fallen, as a power of `e`: estimated along the
    /// edges at first, then measured.
    reach: Vec<f64>,
    /// The vertices in that order.
    order: Vec<u32>,
    heap: BinaryHeap<HeapItem>,
    /// Whether `order` is complete.
    ordered: bool,
    /// How many vertices of `order` have their heat in this pass.
    done: usize,
    /// How many passes are over.
    passes: usize,
    is_source: Vec<bool>,
    /// Per vertex, its place in the band, or [`NONE`].
    place: Vec<u32>,
    band: Band,
}

impl Diffusion {
    /// Diffusion from `sources` over a surface of `n` vertices.
    pub(super) fn new(n: usize, sources: &[u32]) -> Result<Diffusion> {
        let mut reach = filled(n, f64::INFINITY)?;
        let mut is_source = filled(n, false)?;
        let mut heap = BinaryHeap::new();
        for &s in sources {
            is_source[s as usize] = true;
            if reach[s as usize] > 0.0 {
                reach[s as usize] = 0.0;
                heap_push(
                    &mut heap,
                    HeapItem {
                        dist: 0.0,
                        vertex: s,
                    },
                )?;
            }
        }
        let mut order = Vec::new();
        reserve(&mut order, n)?;
        Ok(Diffusion {
            heat: Heat {
                value: filled(n, 0.0f64)?,
                scale: filled(n, 0i32)?,
            },
            reach,
            order,
            heap,
            ordered: false,
            done: 0,
            passes: 0,
            is_source,
            place: filled(n, NONE)?,
            band: Band::default(),
        })
    }

    /// Orders about `work` more vertices, or solves the next band; how much
    /// work that was, or `None` when every vertex has its heat.
    pub(super) fn step(&mut self, s: &Surface<'_>, work: usize) -> Result<Option<usize>> {
        if !self.ordered {
            let before = self.order.len();
            self.order_vertices(s, work.max(1024))?;
            return Ok(Some(self.order.len() - before));
        }
        if self.done < self.order.len() {
            let size = self.solve_band(s)?;
            return Ok(Some(size.saturating_mul(16)));
        }
        self.passes += 1;
        if self.passes == PASSES || self.holds(s)? {
            return Ok(None);
        }
        self.order_by_heat();
        Ok(Some(self.order.len().saturating_mul(8)))
    }

    #[cfg(test)]
    pub(super) fn passes(&self) -> usize {
        self.passes
    }

    pub(super) fn finish(self) -> Heat {
        self.heat
    }

    /// Whether the heat is the solution, at all but a few vertices.
    fn holds(&self, s: &Surface<'_>) -> Result<bool> {
        let n = self.order.len();
        let off = par::map_range(n, BLOCK, PAR_MIN_ROWS, OOM, |range| {
            let off = range.filter(|&v| self.error_at(s, v) > HOLDS).count();
            Ok::<usize, FieldsError>(off)
        })?;
        let off: usize = off.into_iter().sum();
        Ok(off as f64 <= EXCEPTIONS * n as f64)
    }

    /// How much a vertex's heat differs from what it is given, as a share.
    fn error_at(&self, s: &Surface<'_>, v: usize) -> f64 {
        let own = s.mass[v] + s.time_step * s.matrix.diag[v];
        let scale = self.heat.scale[v];
        let mut given = if self.is_source[v] {
            power_of_two(-scale)
        } else {
            0.0
        };
        for (c, w) in s.matrix.off.row(v) {
            let c = c as usize;
            let at = power_of_two(self.heat.scale[c] - scale);
            if at > 0.0 {
                given -= s.time_step * w * self.heat.value[c] * at;
            }
        }
        let has = own * self.heat.value[v];
        let error = (given - has).abs() / has;
        // (NaN without heat, or when a neighbour has too much more of it.)
        if error.is_finite() {
            error
        } else {
            f64::INFINITY
        }
    }

    /// Orders the vertices by how far their heat has fallen.
    fn order_by_heat(&mut self) {
        for (v, reach) in self.reach.iter_mut().enumerate() {
            let fall = -self.heat.ln(v);
            *reach = if fall.is_nan() { f64::INFINITY } else { fall };
        }
        let reach = &self.reach;
        self.order.sort_unstable_by(|&a, &b| {
            reach[a as usize]
                .total_cmp(&reach[b as usize])
                .then(a.cmp(&b))
        });
        self.done = 0;
    }

    /// Puts up to `count` more vertices in order, from the sources on.
    fn order_vertices(&mut self, s: &Surface<'_>, count: usize) -> Result<()> {
        let per_length = 1.0 / s.time_step.sqrt();
        let end = self.order.len().saturating_add(count);
        while let Some(HeapItem { dist, vertex }) = self.heap.pop() {
            let v = vertex as usize;
            if dist > self.reach[v] {
                continue;
            }
            push(&mut self.order, vertex)?;
            let p = widen(s.positions[v]);
            for (c, w) in s.matrix.off.row(v) {
                let mut fall = norm(sub(widen(s.positions[c as usize]), p)) * per_length;
                if fall > LONG {
                    // Along an edge much longer than `√t` the heat falls at
                    // most by what this neighbour alone passes on.
                    let c = c as usize;
                    let own = s.mass[c] + s.time_step * s.matrix.diag[c];
                    fall = fall.min(-(-s.time_step * w / own).ln()).max(LONG);
                }
                let next = dist + fall;
                if next < self.reach[c as usize] {
                    self.reach[c as usize] = next;
                    heap_push(
                        &mut self.heap,
                        HeapItem {
                            dist: next,
                            vertex: c,
                        },
                    )?;
                }
            }
            if self.order.len() >= end {
                return Ok(());
            }
        }
        self.ordered = true;
        self.heap = BinaryHeap::new();
        Ok(())
    }

    /// Solves the band that starts at the first vertex not done in this
    /// pass; how many vertices that took.
    fn solve_band(&mut self, s: &Surface<'_>) -> Result<usize> {
        let from = self.done;
        let start = self.reach[self.order[from] as usize];
        let ends = |at: f64| {
            self.order[from..].partition_point(|&v| self.reach[v as usize] <= start + at) + from
        };
        let (keep, outer) = (ends(WIDTH), ends(WIDTH + OVERLAP));
        if self.passes > 0
            && self.order[from..keep]
                .iter()
                .all(|&v| self.error_at(s, v as usize) <= SETTLED)
        {
            self.done = keep;
            return Ok((keep - from) / 16);
        }

        let band = &mut self.band;
        band.clear();
        band.number(s, &self.order[from..outer], &mut self.place)?;
        band.assemble(s, &self.place, &self.heat)?;
        let scale = band.arrive(&self.heat, &self.is_source)?;
        // The overlap of the band before, or the pass before, is a first guess.
        for (x, &v) in band.x.iter_mut().zip(&band.members) {
            let v = v as usize;
            let guess = self.heat.value[v] * power_of_two(self.heat.scale[v] - scale);
            // (NaN without heat, which counts as scale 0.)
            *x = if guess.is_finite() { guess } else { 0.0 };
        }
        band.solve()?;
        // One sweep outwards: it gives every vertex some of the heat of the
        // one before it on its path, so none is left without any.
        for &v in &self.order[from..outer] {
            let k = self.place[v as usize] as usize;
            let mut sum = band.b[k];
            for (c, a) in band.off.row(k) {
                sum -= a * band.x[c as usize].max(0.0);
            }
            band.x[k] = sum / band.diag[k];
        }
        for (&x, &v) in band.x.iter().zip(&band.members) {
            let v = v as usize;
            self.heat.value[v] = x;
            self.heat.scale[v] = scale;
            self.place[v] = NONE;
        }
        self.done = keep;
        Ok(outer - from)
    }
}

/// The linear system of one band, and the buffers of its solve.
#[derive(Debug, Default)]
struct Band {
    members: Vec<u32>,
    /// The members in the surface's order.
    sorted: Vec<u32>,
    /// One row of `off`: column, weight.
    row: Vec<(u32, f64)>,
    off: Rows,
    /// Per row, where the columns above the diagonal start.
    upper: Vec<u32>,
    diag: Vec<f64>,
    /// Couplings to vertices with heat outside: row, vertex, weight.
    arrivals: Vec<(u32, u32, f64)>,
    b: Vec<f64>,
    /// The incomplete factor's entries, at their places in `off`.
    low: Vec<f64>,
    inv: Vec<f64>,
    /// One row of the factor spread out, zero between rows.
    spread: Vec<f64>,
    x: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
}

fn zeroed(v: &mut Vec<f64>, n: usize) -> Result<()> {
    v.clear();
    if v.capacity() < n {
        reserve(v, n)?;
    }
    v.resize(n, 0.0);
    Ok(())
}

impl Band {
    fn clear(&mut self) {
        self.members.clear();
        self.sorted.clear();
        self.off.ptr.clear();
        self.off.col.clear();
        self.off.val.clear();
        self.upper.clear();
        self.diag.clear();
        self.arrivals.clear();
    }

    /// Numbers the members in chains along their strongest couplings: the
    /// incomplete factor is nearly exact along a chain, so that triangles
    /// with one short side, whose ends couple by far the most, take no more
    /// iterations than others.
    fn number(&mut self, s: &Surface<'_>, vertices: &[u32], place: &mut [u32]) -> Result<()> {
        reserve(&mut self.sorted, vertices.len())?;
        reserve(&mut self.members, vertices.len())?;
        self.sorted.extend_from_slice(vertices);
        // The chains start like the surface, whose order keeps neighbours together.
        self.sorted.sort_unstable();
        for &v in &self.sorted {
            place[v as usize] = UNPLACED;
        }
        for &start in &self.sorted {
            let mut v = start;
            while place[v as usize] == UNPLACED {
                place[v as usize] = self.members.len() as u32;
                self.members.push(v);
                let mut strongest = 0.0;
                for (c, w) in s.matrix.off.row(v as usize) {
                    if -w > strongest && place[c as usize] == UNPLACED {
                        strongest = -w;
                        v = c;
                    }
                }
            }
        }
        Ok(())
    }

    /// The rows of `mass + t · Laplacian` between the members.
    fn assemble(&mut self, s: &Surface<'_>, place: &[u32], heat: &Heat) -> Result<()> {
        let n = self.members.len();
        let entries: usize = self
            .members
            .iter()
            .map(|&v| s.matrix.off.range(v as usize).len())
            .sum();
        reserve(&mut self.off.ptr, n + 1)?;
        reserve(&mut self.off.col, entries)?;
        reserve(&mut self.off.val, entries)?;
        reserve(&mut self.upper, n)?;
        reserve(&mut self.diag, n)?;
        self.off.ptr.push(0);
        for (k, &v) in self.members.iter().enumerate() {
            let v = v as usize;
            self.diag.push(s.mass[v] + s.time_step * s.matrix.diag[v]);
            self.row.clear();
            for (c, w) in s.matrix.off.row(v) {
                let w = s.time_step * w;
                let at = place[c as usize];
                if at != NONE {
                    push(&mut self.row, (at, w))?;
                } else if heat.value[c as usize] > 0.0 {
                    push(&mut self.arrivals, (k as u32, c, -w))?;
                }
            }
            self.row.sort_unstable_by_key(|e| e.0);
            let below = self.row.partition_point(|e| (e.0 as usize) < k);
            self.upper.push((self.off.col.len() + below) as u32);
            for &(c, w) in &self.row {
                self.off.col.push(c);
                self.off.val.push(w);
            }
            self.off.ptr.push(self.off.col.len() as u32);
        }
        Ok(())
    }

    /// Fills `b` with the heat of the sources and the heat arriving from
    /// outside, over the power of two that brings the largest to about 1,
    /// which it returns; sizes the other vectors.
    fn arrive(&mut self, heat: &Heat, is_source: &[bool]) -> Result<i32> {
        let n = self.members.len();
        for v in [
            &mut self.b,
            &mut self.x,
            &mut self.r,
            &mut self.z,
            &mut self.p,
            &mut self.ap,
            &mut self.inv,
            &mut self.spread,
        ] {
            zeroed(v, n)?;
        }
        zeroed(&mut self.low, self.off.col.len())?;
        let scale = self
            .arrivals
            .iter()
            .map(|&(_, v, w)| {
                heat.scale[v as usize] + exponent_of(heat.value[v as usize]) + exponent_of(w)
            })
            .chain(
                // A source gives off 1.
                self.members
                    .iter()
                    .filter(|&&v| is_source[v as usize])
                    .map(|_| 0),
            )
            .max()
            .unwrap_or(0);
        for (b, &v) in self.b.iter_mut().zip(&self.members) {
            if is_source[v as usize] {
                *b = power_of_two(-scale);
            }
        }
        for &(k, v, w) in &self.arrivals {
            let v = v as usize;
            self.b[k as usize] += w * heat.value[v] * power_of_two(heat.scale[v] - scale);
        }
        Ok(scale)
    }

    /// The incomplete Cholesky factor on the matrix's own pattern. It exists
    /// for every symmetric M-matrix (Meijerink & van der Vorst 1977), which
    /// this is.
    fn factor(&mut self) -> Result<()> {
        for i in 0..self.diag.len() {
            let row = self.off.ptr[i] as usize..self.upper[i] as usize;
            let mut d = self.diag[i];
            for e in row.clone() {
                let j = self.off.col[e] as usize;
                let mut v = self.off.val[e];
                for f in self.off.ptr[j] as usize..self.upper[j] as usize {
                    v -= self.low[f] * self.spread[self.off.col[f] as usize];
                }
                v *= self.inv[j];
                self.low[e] = v;
                self.spread[j] = v;
                d -= v * v;
            }
            for e in row {
                self.spread[self.off.col[e] as usize] = 0.0;
            }
            // Rounding can undo what holds exactly: fall back to the diagonal.
            let d = if d > 1e-14 * self.diag[i] {
                d
            } else {
                self.diag[i]
            };
            self.inv[i] = 1.0 / d.sqrt();
        }
        Ok(())
    }

    /// `z = (L Lᵀ)⁻¹ r`.
    fn precondition(&mut self) {
        let n = self.z.len();
        for i in 0..n {
            let row = self.off.ptr[i] as usize..self.upper[i] as usize;
            let mut sum = self.r[i];
            for (&c, &l) in self.off.col[row.clone()].iter().zip(&self.low[row]) {
                sum -= l * self.z[c as usize];
            }
            self.z[i] = sum * self.inv[i];
        }
        for i in (0..n).rev() {
            let row = self.off.ptr[i] as usize..self.upper[i] as usize;
            let zi = self.z[i] * self.inv[i];
            self.z[i] = zi;
            for (&c, &l) in self.off.col[row.clone()].iter().zip(&self.low[row]) {
                self.z[c as usize] -= l * zi;
            }
        }
    }

    /// `out = A from`, for `(p, ap)` or the residual's `(x, r)`.
    fn multiply(off: &Rows, diag: &[f64], from: &[f64], out: &mut [f64]) {
        par::for_each_chunk_mut(out, BLOCK, PAR_MIN_ROWS, |c, ys| {
            for (k, y) in ys.iter_mut().enumerate() {
                let i = c * BLOCK + k;
                let mut sum = diag[i] * from[i];
                for (c, v) in off.row(i) {
                    sum += v * from[c as usize];
                }
                *y = sum;
            }
        });
    }

    /// Preconditioned conjugate gradients from `x`.
    fn solve(&mut self) -> Result<()> {
        let goal = TOLERANCE * TOLERANCE * dot(&self.b, &self.b)?;
        Self::multiply(&self.off, &self.diag, &self.x, &mut self.r);
        for (r, b) in self.r.iter_mut().zip(&self.b) {
            *r = b - *r;
        }
        if dot(&self.r, &self.r)? <= goal {
            return Ok(());
        }
        self.factor()?;
        self.precondition();
        self.p.copy_from_slice(&self.z);
        let mut rz = dot(&self.r, &self.z)?;
        for _ in 0..self.x.len().saturating_mul(2).clamp(200, 20_000) {
            if !rz.is_finite() || rz <= 0.0 {
                break;
            }
            Self::multiply(&self.off, &self.diag, &self.p, &mut self.ap);
            let curvature = dot(&self.p, &self.ap)?;
            if !curvature.is_finite() || curvature <= 0.0 {
                break;
            }
            let alpha = rz / curvature;
            for (x, p) in self.x.iter_mut().zip(&self.p) {
                *x += alpha * p;
            }
            for (r, ap) in self.r.iter_mut().zip(&self.ap) {
                *r -= alpha * ap;
            }
            if dot(&self.r, &self.r)? <= goal {
                break;
            }
            self.precondition();
            let next = dot(&self.r, &self.z)?;
            let beta = next / rz;
            for (p, z) in self.p.iter_mut().zip(&self.z) {
                *p = z + beta * *p;
            }
            rz = next;
        }
        Ok(())
    }
}
