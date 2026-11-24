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
//! The distance that cuts the bands is the shortest one along mesh edges. It
//! only decides what is solved together, not what is solved.

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
/// No place in the band.
const NONE: u32 = u32::MAX;

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
    sources: Vec<u32>,
    heat: Heat,
    /// Distance along edges from the nearest source, over `√t`.
    reach: Vec<f64>,
    /// The vertices by that distance.
    order: Vec<u32>,
    heap: BinaryHeap<HeapItem>,
    /// Whether `order` is complete.
    ordered: bool,
    /// How many vertices of `order` have their heat.
    done: usize,
    solved: Vec<bool>,
    /// Per vertex, its place in the band, or [`NONE`].
    place: Vec<u32>,
    band: Band,
}

impl Diffusion {
    /// Diffusion from `sources` over a surface of `n` vertices.
    pub(super) fn new(n: usize, sources: Vec<u32>) -> Result<Diffusion> {
        let mut reach = filled(n, f64::INFINITY)?;
        let mut heap = BinaryHeap::new();
        for &s in &sources {
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
            sources,
            heat: Heat {
                value: filled(n, 0.0f64)?,
                scale: filled(n, 0i32)?,
            },
            reach,
            order,
            heap,
            ordered: false,
            done: 0,
            solved: filled(n, false)?,
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
        if self.done == self.order.len() {
            return Ok(None);
        }
        let size = self.solve_band(s)?;
        Ok(Some(size.saturating_mul(16)))
    }

    pub(super) fn finish(self) -> Heat {
        self.heat
    }

    /// Settles up to `count` more vertices in order of their distance.
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
            for (c, _) in s.matrix.off.row(v) {
                let len = norm(sub(widen(s.positions[c as usize]), p));
                let next = dist + len * per_length;
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

    /// Solves the band that starts at the first vertex without heat; how
    /// many vertices that took.
    fn solve_band(&mut self, s: &Surface<'_>) -> Result<usize> {
        let from = self.done;
        let start = self.reach[self.order[from] as usize];
        let ends = |at: f64| {
            self.order[from..].partition_point(|&v| self.reach[v as usize] <= start + at) + from
        };
        let (keep, outer) = (ends(WIDTH), ends(WIDTH + OVERLAP));

        let band = &mut self.band;
        band.clear();
        reserve(&mut band.members, outer - from)?;
        band.members.extend_from_slice(&self.order[from..outer]);
        // Numbered like the surface, which keeps neighbours together.
        band.members.sort_unstable();
        for (k, &v) in band.members.iter().enumerate() {
            self.place[v as usize] = k as u32;
        }
        band.assemble(s, &self.place, &self.solved, &self.heat)?;
        let scale = band.arrive(&self.heat)?;
        if from == 0 {
            for &v in &self.sources {
                let k = self.place[v as usize] as usize;
                band.b[k] = 1.0;
            }
        }
        // The overlap of the band before is a first guess.
        for (x, &v) in band.x.iter_mut().zip(&band.members) {
            let v = v as usize;
            if self.heat.value[v] > 0.0 {
                *x = self.heat.value[v] * power_of_two(self.heat.scale[v] - scale);
            }
        }
        band.factor()?;
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
        for &v in &self.order[from..keep] {
            self.solved[v as usize] = true;
        }
        self.done = keep;
        Ok(outer - from)
    }
}

/// The linear system of one band, and the buffers of its solve.
#[derive(Debug, Default)]
struct Band {
    members: Vec<u32>,
    off: Rows,
    /// Per row, where the columns above the diagonal start.
    upper: Vec<u32>,
    diag: Vec<f64>,
    /// Couplings to solved vertices: row, vertex, weight.
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
        self.off.ptr.clear();
        self.off.col.clear();
        self.off.val.clear();
        self.upper.clear();
        self.diag.clear();
        self.arrivals.clear();
    }

    /// The rows of `mass + t · Laplacian` between the members.
    fn assemble(
        &mut self,
        s: &Surface<'_>,
        place: &[u32],
        solved: &[bool],
        heat: &Heat,
    ) -> Result<()> {
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
            let mut upper = None;
            for (c, w) in s.matrix.off.row(v) {
                let w = s.time_step * w;
                let at = place[c as usize];
                if at != NONE {
                    if at as usize > k && upper.is_none() {
                        upper = Some(self.off.col.len());
                    }
                    self.off.col.push(at);
                    self.off.val.push(w);
                } else if solved[c as usize] && heat.value[c as usize] > 0.0 {
                    push(&mut self.arrivals, (k as u32, c, -w))?;
                }
            }
            self.upper.push(upper.unwrap_or(self.off.col.len()) as u32);
            self.off.ptr.push(self.off.col.len() as u32);
        }
        Ok(())
    }

    /// Fills `b` with the heat arriving from solved vertices, over the power
    /// of two that brings the largest to about 1, which it returns; sizes the
    /// other vectors.
    fn arrive(&mut self, heat: &Heat) -> Result<i32> {
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
            .max()
            .unwrap_or(0);
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
