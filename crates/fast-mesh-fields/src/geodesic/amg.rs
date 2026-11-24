//! Conjugate gradients preconditioned by smoothed-aggregation algebraic
//! multigrid (Vaněk, Mandel & Brezina 1996), for the Poisson step of the heat
//! method.
//!
//! The iteration count hardly grows with the mesh, unlike that of an
//! incomplete factorisation, and memory stays within about half again that
//! of the matrix, unlike that of a complete one.
//!
//! The matrices are positive semidefinite graph Laplacians: the constants are
//! their null space, and the right-hand sides sum to zero. Every sweep and
//! product runs over fixed blocks of [`BLOCK`] rows and every row is computed
//! on its own, so results are the same, bit for bit, with and without the
//! `parallel` feature and for any thread count.

use std::ops::Range;

use fast_mesh_store::par;

use crate::grid::{allocation_allowed, filled, push, reserve, reserve_exact};
use crate::{FieldsError, Result};

/// Row count from which row loops run on the pool.
pub(super) const PAR_MIN_ROWS: usize = 16_384;
/// Rows per chunk of the row loops, and per block of the smoother: a block
/// is swept in order (Gauss-Seidel), the blocks against each other's old
/// values.
pub(super) const BLOCK: usize = 1024;
/// A level of at most this many rows is solved directly.
const COARSE_ROWS: usize = 200;
/// Levels at most: a mesh of `u32` vertices needs about ten.
const MAX_LEVELS: usize = 24;
/// Coarsening stops once a level keeps more than this share of the rows.
const STALL: f64 = 0.85;
/// A coupling counts as strong from this share of its rows' diagonals.
const STRONG: f64 = 0.08;
/// Damping of the interpolation's smoothing step: `4/3` over the largest
/// eigenvalue of the filtered matrix scaled by its diagonal, which is at
/// most 2 (its rows sum to at least zero and only its diagonal is positive).
const DAMPING: f64 = 2.0 / 3.0;
/// No entry, no aggregate.
const NONE: u32 = u32::MAX;

const OOM: fn() -> FieldsError = || FieldsError::OutOfMemory;

/// A sparse matrix by rows, columns ascending.
#[derive(Debug, Clone, Default)]
pub(super) struct Rows {
    pub(super) ptr: Vec<u32>,
    pub(super) col: Vec<u32>,
    pub(super) val: Vec<f64>,
}

impl Rows {
    pub(super) fn range(&self, row: usize) -> Range<usize> {
        self.ptr[row] as usize..self.ptr[row + 1] as usize
    }

    /// The entries of a row, as `(column, value)`.
    pub(super) fn row(&self, row: usize) -> impl Iterator<Item = (u32, f64)> + '_ {
        let r = self.range(row);
        self.col[r.clone()]
            .iter()
            .copied()
            .zip(self.val[r].iter().copied())
    }

    fn rows(&self) -> usize {
        self.ptr.len().saturating_sub(1)
    }

    fn byte_size(&self) -> usize {
        self.ptr.capacity() * size_of::<u32>()
            + self.col.capacity() * size_of::<u32>()
            + self.val.capacity() * size_of::<f64>()
    }

    /// `y = self x`.
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        par::for_each_chunk_mut(y, BLOCK, PAR_MIN_ROWS, |c, ys| {
            for (k, out) in ys.iter_mut().enumerate() {
                *out = self.dot_row(c * BLOCK + k, x);
            }
        });
    }

    /// `y += self x`.
    fn add_apply(&self, x: &[f64], y: &mut [f64]) {
        par::for_each_chunk_mut(y, BLOCK, PAR_MIN_ROWS, |c, ys| {
            for (k, out) in ys.iter_mut().enumerate() {
                *out += self.dot_row(c * BLOCK + k, x);
            }
        });
    }

    fn dot_row(&self, row: usize, x: &[f64]) -> f64 {
        let mut sum = 0.0;
        for (c, v) in self.row(row) {
            sum += v * x[c as usize];
        }
        sum
    }

    fn transposed(&self, columns: usize) -> Result<Rows> {
        let mut ptr = filled(columns + 1, 0u32)?;
        for &c in &self.col {
            ptr[c as usize + 1] += 1;
        }
        for i in 0..columns {
            ptr[i + 1] += ptr[i];
        }
        let mut at = filled(columns, 0u32)?;
        at.copy_from_slice(&ptr[..columns]);
        let mut col = filled(self.col.len(), 0u32)?;
        let mut val = filled(self.val.len(), 0.0f64)?;
        for row in 0..self.rows() {
            for (c, v) in self.row(row) {
                let slot = &mut at[c as usize];
                col[*slot as usize] = row as u32;
                val[*slot as usize] = v;
                *slot += 1;
            }
        }
        Ok(Rows { ptr, col, val })
    }
}

/// Sums the entries of one row by column.
struct Accumulator {
    /// Per column, where its entry is in `entries`, or [`NONE`].
    slot: Vec<u32>,
    entries: Vec<(u32, f64)>,
}

impl Accumulator {
    fn add(&mut self, col: u32, val: f64) -> Result<()> {
        let slot = &mut self.slot[col as usize];
        if *slot == NONE {
            *slot = self.entries.len() as u32;
            push(&mut self.entries, (col, val))
        } else {
            self.entries[*slot as usize].1 += val;
            Ok(())
        }
    }
}

/// A `rows` by `columns` matrix whose row `i` is what `fill(i, _)` adds up.
fn build_rows(
    rows: usize,
    columns: usize,
    fill: impl Fn(usize, &mut Accumulator) -> Result<()> + Sync,
) -> Result<Rows> {
    // A bounded number of chunks, for each allocates a slot per column.
    let chunk = rows.div_ceil(64).max(4096);
    allocation_allowed()?;
    let parts = par::map_range(rows, chunk, PAR_MIN_ROWS, OOM, |range| {
        let mut acc = Accumulator {
            slot: filled(columns, NONE)?,
            entries: Vec::new(),
        };
        let mut part = Rows::default();
        reserve_exact(&mut part.ptr, range.len())?;
        for row in range {
            fill(row, &mut acc)?;
            acc.entries.sort_unstable_by_key(|e| e.0);
            reserve(&mut part.col, acc.entries.len())?;
            reserve(&mut part.val, acc.entries.len())?;
            for (c, v) in acc.entries.drain(..) {
                acc.slot[c as usize] = NONE;
                part.col.push(c);
                part.val.push(v);
            }
            part.ptr.push(part.col.len() as u32);
        }
        Ok(part)
    })?;
    let total: usize = parts.iter().map(|p| p.col.len()).sum();
    if u32::try_from(total).is_err() {
        return Err(FieldsError::OutOfMemory);
    }
    let mut out = Rows::default();
    reserve_exact(&mut out.ptr, rows + 1)?;
    reserve_exact(&mut out.col, total)?;
    reserve_exact(&mut out.val, total)?;
    out.ptr.push(0);
    for part in parts {
        let base = out.col.len() as u32;
        out.ptr.extend(part.ptr.iter().map(|p| p + base));
        out.col.extend_from_slice(&part.col);
        out.val.extend_from_slice(&part.val);
    }
    Ok(out)
}

/// A symmetric matrix: its off-diagonal entries by rows, its diagonal, and
/// what the smoother needs of both.
#[derive(Debug, Clone, Default)]
pub(super) struct Matrix {
    pub(super) off: Rows,
    pub(super) diag: Vec<f64>,
    /// Per row, where the columns above the diagonal start.
    upper: Vec<u32>,
    /// Per row, what the smoother multiplies a residual by.
    relax: Vec<f64>,
}

impl Matrix {
    pub(super) fn new(off: Rows, diag: Vec<f64>) -> Result<Matrix> {
        let n = diag.len();
        let mut upper = filled(n, 0u32)?;
        let mut relax = filled(n, 0.0f64)?;
        for (i, up) in upper.iter_mut().enumerate() {
            let r = off.range(i);
            *up = (r.start + off.col[r].partition_point(|&c| (c as usize) < i)) as u32;
        }
        for (i, out) in relax.iter_mut().enumerate() {
            // Sweeping the blocks against each other's old values converges
            // in the energy norm from this diagonal on (a half of the l1
            // scaling of Baker et al. 2011, which leaves the diagonal alone
            // unless the other blocks' entries outweigh it).
            let base = i - i % BLOCK;
            let others: f64 = off
                .row(i)
                .filter(|&(c, _)| !(base..base + BLOCK).contains(&(c as usize)))
                .map(|(_, v)| v.abs())
                .sum();
            let d = diag[i].max(0.5 * (diag[i] + others));
            *out = if d > 0.0 && d.is_finite() {
                1.0 / d
            } else {
                0.0
            };
        }
        Ok(Matrix {
            off,
            diag,
            upper,
            relax,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.diag.len()
    }

    fn byte_size(&self) -> usize {
        self.off.byte_size()
            + self.upper.capacity() * size_of::<u32>()
            + (self.diag.capacity() + self.relax.capacity()) * size_of::<f64>()
    }

    /// `y = self x`.
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        par::for_each_chunk_mut(y, BLOCK, PAR_MIN_ROWS, |c, ys| {
            for (k, out) in ys.iter_mut().enumerate() {
                let i = c * BLOCK + k;
                *out = self.diag[i] * x[i] + self.off.dot_row(i, x);
            }
        });
    }

    /// `r = b - self x`.
    fn residual(&self, b: &[f64], x: &[f64], r: &mut [f64]) {
        par::for_each_chunk_mut(r, BLOCK, PAR_MIN_ROWS, |c, rs| {
            for (k, out) in rs.iter_mut().enumerate() {
                let i = c * BLOCK + k;
                *out = b[i] - self.diag[i] * x[i] - self.off.dot_row(i, x);
            }
        });
    }

    /// `x = M⁻¹ b`: one forward sweep from zero.
    fn sweep_forward(&self, b: &[f64], x: &mut [f64]) {
        par::for_each_chunk_mut(x, BLOCK, PAR_MIN_ROWS, |c, xs| {
            let base = c * BLOCK;
            for k in 0..xs.len() {
                let i = base + k;
                let r = self.off.ptr[i] as usize..self.upper[i] as usize;
                let mut sum = b[i];
                for (&c, &v) in self.off.col[r.clone()].iter().zip(&self.off.val[r]) {
                    if let Some(xj) = xs.get((c as usize).wrapping_sub(base)) {
                        sum -= v * xj;
                    }
                }
                xs[k] = sum * self.relax[i];
            }
        });
    }

    /// `y = x + M⁻ᵀ (b - self x)`: one backward sweep from `x`.
    fn sweep_backward(&self, b: &[f64], x: &[f64], y: &mut [f64]) {
        par::for_each_chunk_mut(y, BLOCK, PAR_MIN_ROWS, |c, ys| {
            let base = c * BLOCK;
            ys.copy_from_slice(&x[base..base + ys.len()]);
            for k in (0..ys.len()).rev() {
                let i = base + k;
                let mut sum = b[i] - self.diag[i] * ys[k];
                for (c, v) in self.off.row(i) {
                    let c = c as usize;
                    let xj = match ys.get(c.wrapping_sub(base)) {
                        Some(yj) => *yj,
                        None => x[c],
                    };
                    sum -= v * xj;
                }
                ys[k] += sum * self.relax[i];
            }
        });
    }
}

/// `LDLᵀ` of a small positive semidefinite matrix. A pivot that vanishes
/// (the constants of a connected part) fixes its unknown to zero.
#[derive(Debug, Clone)]
struct Dense {
    n: usize,
    /// The unit lower factor by rows, full.
    low: Vec<f64>,
    /// Per row, one over its pivot, or zero.
    inv: Vec<f64>,
}

impl Dense {
    fn new(a: &Matrix) -> Result<Dense> {
        let n = a.len();
        let mut low = filled(n * n, 0.0f64)?;
        let mut inv = filled(n, 0.0f64)?;
        for i in 0..n {
            for (c, v) in a.off.row(i) {
                low[i * n + c as usize] = v;
            }
            low[i * n + i] = a.diag[i];
        }
        let mut pivot = filled(n, 0.0f64)?;
        for i in 0..n {
            for j in 0..i {
                let mut v = low[i * n + j];
                for k in 0..j {
                    v -= low[i * n + k] * pivot[k] * low[j * n + k];
                }
                low[i * n + j] = v * inv[j];
            }
            let mut d = low[i * n + i];
            for k in 0..i {
                d -= low[i * n + k] * low[i * n + k] * pivot[k];
            }
            if d.is_finite() && d > 1e-10 * a.diag[i] {
                pivot[i] = d;
                inv[i] = 1.0 / d;
            }
        }
        Ok(Dense { n, low, inv })
    }

    fn solve(&self, b: &[f64], x: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            let mut v = b[i];
            for (j, xj) in x[..i].iter().enumerate() {
                v -= self.low[i * n + j] * xj;
            }
            // An unknown without a pivot stays zero, and so adds nothing.
            x[i] = if self.inv[i] > 0.0 { v } else { 0.0 };
        }
        for (xi, inv) in x.iter_mut().zip(&self.inv) {
            *xi *= inv;
        }
        for i in (0..n).rev() {
            let xi = x[i];
            for (j, xj) in x[..i].iter_mut().enumerate() {
                if self.inv[j] > 0.0 {
                    *xj -= self.low[i * n + j] * xi;
                }
            }
        }
    }
}

/// A coarser level: how it interpolates to the one above, and its matrix.
#[derive(Debug, Clone)]
struct Level {
    /// From this level's unknowns to the finer level's.
    prolong: Rows,
    /// Its transpose.
    restrict: Rows,
    matrix: Matrix,
}

/// The multigrid levels of one matrix.
#[derive(Debug, Clone)]
pub(super) struct Amg {
    fine: Matrix,
    levels: Vec<Level>,
    /// The last level's factors; without any (a matrix small enough as it
    /// is, or one that stopped coarsening) it is smoothed only.
    coarsest: Option<Dense>,
}

/// The vectors a cycle works in, per level.
#[derive(Debug, Default)]
struct LevelWork {
    b: Vec<f64>,
    x: Vec<f64>,
    r: Vec<f64>,
}

/// [`Amg`] under construction, a level per step.
#[derive(Debug)]
pub(super) struct AmgBuild {
    amg: Amg,
    stage: Stage,
}

#[derive(Debug)]
enum Stage {
    /// Group the last level's rows and interpolate between the groups.
    Interpolate,
    /// Multiply the next level's matrix from the interpolation.
    Multiply {
        prolong: Rows,
        restrict: Rows,
    },
    /// Add its diagonal and factor it if it is the last.
    Finish {
        prolong: Rows,
        restrict: Rows,
        product: Rows,
    },
    Done,
}

impl AmgBuild {
    pub(super) fn new(fine: Matrix) -> AmgBuild {
        AmgBuild {
            amg: Amg {
                fine,
                levels: Vec::new(),
                coarsest: None,
            },
            stage: Stage::Interpolate,
        }
    }

    /// Runs the next pass; whether the levels are done.
    pub(super) fn step(&mut self) -> Result<bool> {
        let last = match self.amg.levels.last() {
            Some(level) => &level.matrix,
            None => &self.amg.fine,
        };
        match std::mem::replace(&mut self.stage, Stage::Done) {
            Stage::Interpolate => {
                let n = last.len();
                if n <= COARSE_ROWS {
                    if !self.amg.levels.is_empty() {
                        self.amg.coarsest = Some(Dense::new(last)?);
                    }
                    return Ok(true);
                }
                if self.amg.levels.len() + 1 >= MAX_LEVELS {
                    return Ok(true);
                }
                let (aggregate, count) = aggregates(last)?;
                if count == 0 || count as f64 > STALL * n as f64 {
                    return Ok(true);
                }
                let prolong = interpolation(last, &aggregate, count)?;
                let restrict = prolong.transposed(count)?;
                self.stage = Stage::Multiply { prolong, restrict };
            }
            Stage::Multiply { prolong, restrict } => {
                let columns = restrict.rows();
                let product = build_rows(last.len(), columns, |i, acc| {
                    for (c, p) in prolong.row(i) {
                        acc.add(c, last.diag[i] * p)?;
                    }
                    for (j, a) in last.off.row(i) {
                        for (c, p) in prolong.row(j as usize) {
                            acc.add(c, a * p)?;
                        }
                    }
                    Ok(())
                })?;
                self.stage = Stage::Finish {
                    prolong,
                    restrict,
                    product,
                };
            }
            Stage::Finish {
                prolong,
                restrict,
                product,
            } => {
                let n = restrict.rows();
                let full = build_rows(n, n, |i, acc| {
                    for (f, r) in restrict.row(i) {
                        for (c, v) in product.row(f as usize) {
                            acc.add(c, r * v)?;
                        }
                    }
                    Ok(())
                })?;
                let matrix = without_diagonal(full)?;
                push(
                    &mut self.amg.levels,
                    Level {
                        prolong,
                        restrict,
                        matrix,
                    },
                )?;
                self.stage = Stage::Interpolate;
            }
            Stage::Done => return Ok(true),
        }
        Ok(false)
    }

    pub(super) fn finish(self) -> Amg {
        self.amg
    }
}

/// Splits the diagonal off a square matrix.
fn without_diagonal(mut full: Rows) -> Result<Matrix> {
    let n = full.rows();
    let mut diag = filled(n, 0.0f64)?;
    let mut at = 0usize;
    let mut from = 0usize;
    for (i, d) in diag.iter_mut().enumerate() {
        let to = full.ptr[i + 1] as usize;
        for e in from..to {
            let c = full.col[e];
            if c as usize == i {
                *d = full.val[e];
            } else {
                full.col[at] = c;
                full.val[at] = full.val[e];
                at += 1;
            }
        }
        from = to;
        full.ptr[i + 1] = at as u32;
    }
    full.col.truncate(at);
    full.val.truncate(at);
    Matrix::new(full, diag)
}

/// Whether row `i`'s entry `v` in column `c` couples the two strongly.
fn strong(a: &Matrix, i: usize, c: u32, v: f64) -> bool {
    v < 0.0 && v * v >= STRONG * STRONG * a.diag[i] * a.diag[c as usize]
}

/// Groups the rows into aggregates of strongly coupled neighbours: per row
/// its aggregate ([`NONE`] for a row coupled to nothing), and how many.
fn aggregates(a: &Matrix) -> Result<(Vec<u32>, usize)> {
    let n = a.len();
    let mut of = filled(n, NONE)?;
    let mut count = 0u32;
    // Rows none of whose neighbours is taken, with their neighbours.
    for i in 0..n {
        if of[i] != NONE {
            continue;
        }
        let mut free = true;
        let mut any = false;
        for (c, v) in a.off.row(i) {
            if strong(a, i, c, v) {
                any = true;
                if of[c as usize] != NONE {
                    free = false;
                    break;
                }
            }
        }
        if !free || !any {
            continue;
        }
        of[i] = count;
        for (c, v) in a.off.row(i) {
            if strong(a, i, c, v) {
                of[c as usize] = count;
            }
        }
        count += 1;
    }
    // The rest join the aggregate they couple to most, if any.
    let mut join: Vec<(u32, u32)> = Vec::new();
    for i in 0..n {
        if of[i] != NONE {
            continue;
        }
        let mut best = (0.0, NONE);
        for (c, v) in a.off.row(i) {
            if v < 0.0 && of[c as usize] != NONE && -v > best.0 {
                best = (-v, of[c as usize]);
            }
        }
        if best.1 != NONE {
            push(&mut join, (i as u32, best.1))?;
        }
    }
    for (i, g) in join {
        of[i as usize] = g;
    }
    // What is left couples to no aggregate: rows around one another.
    for i in 0..n {
        if of[i] != NONE {
            continue;
        }
        let mut any = false;
        for (c, v) in a.off.row(i) {
            if v < 0.0 && of[c as usize] == NONE {
                of[c as usize] = count;
                any = true;
            }
        }
        if any {
            of[i] = count;
            count += 1;
        }
    }
    Ok((of, count as usize))
}

/// Piecewise constant over the aggregates, smoothed by one damped Jacobi
/// step of `a` without its weak couplings (which go to the diagonal, so that
/// the rows keep their sums).
fn interpolation(a: &Matrix, aggregate: &[u32], count: usize) -> Result<Rows> {
    build_rows(a.len(), count, |i, acc| {
        let own = aggregate[i];
        if own == NONE {
            return Ok(());
        }
        let mut d = a.diag[i];
        for (c, v) in a.off.row(i) {
            if !strong(a, i, c, v) {
                d += v;
            }
        }
        if !(d > 0.0 && d.is_finite()) {
            return acc.add(own, 1.0);
        }
        acc.add(own, 1.0 - DAMPING)?;
        for (c, v) in a.off.row(i) {
            let g = aggregate[c as usize];
            if g != NONE && strong(a, i, c, v) {
                acc.add(g, -v * DAMPING / d)?;
            }
        }
        Ok(())
    })
}

impl Amg {
    pub(super) fn fine(&self) -> &Matrix {
        &self.fine
    }

    pub(super) fn byte_size(&self) -> usize {
        let levels: usize = self
            .levels
            .iter()
            .map(|l| l.prolong.byte_size() + l.restrict.byte_size() + l.matrix.byte_size())
            .sum();
        let coarsest = self.coarsest.as_ref().map_or(0, |d| {
            (d.low.capacity() + d.inv.capacity()) * size_of::<f64>()
        });
        self.fine.byte_size() + levels + coarsest
    }

    fn work(&self) -> Result<Vec<LevelWork>> {
        let mut work = Vec::new();
        reserve_exact(&mut work, self.levels.len() + 1)?;
        for k in 0..=self.levels.len() {
            let n = self.matrix(k).len();
            work.push(LevelWork {
                // The finest right-hand side is the caller's.
                b: if k == 0 { Vec::new() } else { filled(n, 0.0)? },
                x: filled(n, 0.0)?,
                r: filled(n, 0.0)?,
            });
        }
        Ok(work)
    }

    fn matrix(&self, level: usize) -> &Matrix {
        match level {
            0 => &self.fine,
            k => &self.levels[k - 1].matrix,
        }
    }

    /// One symmetric V-cycle on `b`; the result is in `work[0].x`.
    fn cycle(&self, b: &[f64], work: &mut [LevelWork]) {
        let last = self.levels.len();
        for k in 0..last {
            let (head, tail) = work.split_at_mut(k + 1);
            let w = &mut head[k];
            let b = if k == 0 { b } else { &w.b[..] };
            let a = self.matrix(k);
            a.sweep_forward(b, &mut w.x);
            a.residual(b, &w.x, &mut w.r);
            self.levels[k].restrict.apply(&w.r, &mut tail[0].b);
        }
        {
            let w = &mut work[last];
            let b = if last == 0 { b } else { &w.b[..] };
            match &self.coarsest {
                Some(dense) => dense.solve(b, &mut w.x),
                None => {
                    let a = self.matrix(last);
                    a.sweep_forward(b, &mut w.x);
                    a.sweep_backward(b, &w.x, &mut w.r);
                    std::mem::swap(&mut w.x, &mut w.r);
                }
            }
        }
        for k in (0..last).rev() {
            let (head, tail) = work.split_at_mut(k + 1);
            let w = &mut head[k];
            self.levels[k].prolong.add_apply(&tail[0].x, &mut w.x);
            let b = if k == 0 { b } else { &w.b[..] };
            self.matrix(k).sweep_backward(b, &w.x, &mut w.r);
            std::mem::swap(&mut w.x, &mut w.r);
        }
    }
}

/// A conjugate gradient solve of the finest matrix under way, an iteration
/// per step.
#[derive(Debug)]
pub(super) struct Solve {
    x: Vec<f64>,
    r: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
    work: Vec<LevelWork>,
    /// The squared energy norm of the error, as the preconditioner sees it.
    rz: f64,
    /// Where `rz` is small enough.
    goal: f64,
    left: usize,
}

impl Solve {
    /// Solves `amg x = b` from `x`, until the squared energy norm of the
    /// error is about `goal`. `b` must sum to zero.
    pub(super) fn new(amg: &Amg, x: Vec<f64>, b: &[f64], goal: f64) -> Result<Solve> {
        let n = amg.fine.len();
        let mut r = filled(n, 0.0f64)?;
        amg.fine.residual(b, &x, &mut r);
        let mut work = amg.work()?;
        amg.cycle(&r, &mut work);
        let mut p = filled(n, 0.0f64)?;
        p.copy_from_slice(&work[0].x);
        let rz = dot(&r, &work[0].x)?;
        Ok(Solve {
            x,
            r,
            p,
            ap: filled(n, 0.0f64)?,
            work,
            rz,
            goal,
            left: n.saturating_mul(2).clamp(200, 2000),
        })
    }

    /// Runs one iteration; whether the solution is close enough (or can get
    /// no closer).
    pub(super) fn step(&mut self, amg: &Amg) -> Result<bool> {
        if self.left == 0 || !self.rz.is_finite() || self.rz <= self.goal {
            return Ok(true);
        }
        self.left -= 1;
        amg.fine.apply(&self.p, &mut self.ap);
        let curvature = dot(&self.p, &self.ap)?;
        if !curvature.is_finite() || curvature <= 0.0 {
            return Ok(true);
        }
        let alpha = self.rz / curvature;
        add_scaled(alpha, &self.p, &mut self.x);
        add_scaled(-alpha, &self.ap, &mut self.r);
        amg.cycle(&self.r, &mut self.work);
        let z = &self.work[0].x;
        let rz = dot(&self.r, z)?;
        if !rz.is_finite() || rz <= self.goal {
            return Ok(true);
        }
        let beta = rz / self.rz;
        par::for_each_chunk_mut(&mut self.p, BLOCK, PAR_MIN_ROWS, |c, ps| {
            for (pi, zi) in ps.iter_mut().zip(&z[c * BLOCK..]) {
                *pi = zi + beta * *pi;
            }
        });
        self.rz = rz;
        Ok(false)
    }

    pub(super) fn finish(self) -> Vec<f64> {
        self.x
    }
}

/// `y += alpha x`.
fn add_scaled(alpha: f64, x: &[f64], y: &mut [f64]) {
    par::for_each_chunk_mut(y, BLOCK, PAR_MIN_ROWS, |c, ys| {
        for (yi, xi) in ys.iter_mut().zip(&x[c * BLOCK..]) {
            *yi += alpha * xi;
        }
    });
}

/// The dot product, summed by chunks in order.
pub(super) fn dot(a: &[f64], b: &[f64]) -> Result<f64> {
    sum_range(a.len(), |i| a[i] * b[i])
}

/// The sum of `term(0..len)`, by chunks in order.
pub(super) fn sum_range(len: usize, term: impl Fn(usize) -> f64 + Sync) -> Result<f64> {
    let part = |r: Range<usize>| {
        let mut sum = 0.0;
        for i in r {
            sum += term(i);
        }
        sum
    };
    if len < PAR_MIN_ROWS {
        // The same chunks, without a vector of their sums.
        let mut total = 0.0;
        let mut at = 0;
        while at < len {
            let end = len.min(at + BLOCK);
            total += part(at..end);
            at = end;
        }
        return Ok(total);
    }
    let parts = par::map_range(len, BLOCK, PAR_MIN_ROWS, OOM, |r| {
        Ok::<f64, FieldsError>(part(r))
    })?;
    Ok(parts.into_iter().sum())
}
