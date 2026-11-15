//! Sparse direct solver for the fairing normal equations.
//!
//! [`factor_ordered`] factors `BᵀB`, for a sparse `B` in compressed rows,
//! as `Pᵀ L D Lᵀ P`: the caller-given permutation `P`, a unit lower `L` in
//! compressed columns and a diagonal `D`. [`Factor::solve`] then solves for
//! a right-hand side. In-tree and dependency-free: the symbolic
//! factorisation and the numeric LDLᵀ are all here.
//!
//! The pattern of `L` comes from the elimination tree: each row's pattern
//! is found by walking the tree up from the row's entries in `BᵀB`, and the
//! columns are the transpose. [`nested_dissection`] orders a patch's free
//! vertices by recursive geometric splitting, so the factor stays sparse.
//! Every tie breaks to the smallest index, so the factorisation and the
//! solution are deterministic.

use fast_mesh_store::StoreError;

use crate::{OpsError, Result, filled, push, reserve};

/// No parent, in an elimination tree.
const NONE: u32 = u32::MAX;

/// `BᵀB` factored as `Pᵀ L D Lᵀ P`.
pub(crate) struct Factor {
    /// Column `j` holds rows `row_idx[col_starts[j]..col_starts[j + 1]]`,
    /// all past `j` and ascending, with `vals` alongside. The diagonal of
    /// `L` is an implicit unit.
    col_starts: Vec<u32>,
    row_idx: Vec<u32>,
    vals: Vec<f64>,
    /// The diagonal `D`.
    diag: Vec<f64>,
    /// `order[p]` is the original index eliminated at position `p`.
    order: Vec<u32>,
}

/// Factors `BᵀB`, where `B` has `n` columns and row `k` holds
/// `cols[starts[k]..starts[k + 1]]` with `b_vals` alongside, eliminating in
/// `order` (`order[p]` is the original index at position `p`).
///
/// `None` when the product is not numerically positive definite (a zero,
/// negative or non-finite pivot).
///
/// # Errors
/// [`OpsError::Store`] when memory runs out.
pub(crate) fn factor_ordered(
    starts: &[u32],
    cols: &[u32],
    b_vals: &[f64],
    n: usize,
    order: &[u32],
) -> Result<Option<Factor>> {
    if n > u32::MAX as usize {
        return Err(OpsError::Store(StoreError::OutOfMemory));
    }
    let mut inv = filled(n, 0u32)?;
    for (p, &k) in order.iter().enumerate() {
        inv[k as usize] = p as u32;
    }
    // The strict upper triangle of the permuted BᵀB, in compressed columns:
    // every pair a B row joins, as (min, max) of positions.
    let mut pairs: Vec<u64> = Vec::new();
    for k in 0..starts.len() - 1 {
        let range = starts[k] as usize..starts[k + 1] as usize;
        let width = range.len();
        reserve(&mut pairs, width * width / 2)?;
        for (i, &a) in cols[range.clone()].iter().enumerate() {
            for &b in &cols[range.start + i + 1..range.end] {
                let (p, q) = (inv[a as usize], inv[b as usize]);
                if p != q {
                    pairs.push((p.max(q) as u64) << 32 | p.min(q) as u64);
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    // Compressed columns of the strict upper triangle: pairs sort by max
    // first, so each column's entries (its lesser ends, ascending) come as
    // one run.
    let mut up_starts: Vec<u32> = Vec::new();
    reserve(&mut up_starts, n + 1)?;
    up_starts.push(0);
    let mut up_rows: Vec<u32> = Vec::new();
    reserve(&mut up_rows, pairs.len())?;
    let mut at = 0usize;
    for k in 0..n {
        while at < pairs.len() && (pairs[at] >> 32) as u32 == k as u32 {
            push(&mut up_rows, pairs[at] as u32)?;
            at += 1;
        }
        up_starts.push(up_rows.len() as u32);
    }
    let parent = etree(n, &up_starts, &up_rows)?;
    let (col_starts, row_idx) = patterns(n, &up_starts, &up_rows, &parent)?;
    let total = row_idx.len();
    let mut vals = filled(total, 0.0f64)?;
    let mut diag = filled(n, 0.0f64)?;
    // Scatter B's rows into A: a row with (position, value) pairs adds
    // `v * w` at (max, min), the diagonal included.
    let mut perm: Vec<(u32, f64)> = Vec::new();
    for k in 0..starts.len() - 1 {
        let range = starts[k] as usize..starts[k + 1] as usize;
        perm.clear();
        reserve(&mut perm, range.len())?;
        perm.extend(
            cols[range.clone()]
                .iter()
                .zip(&b_vals[range.clone()])
                .map(|(&c, &v)| (inv[c as usize], v)),
        );
        perm.sort_by_key(|&(p, _)| p);
        for (i, &(p, v)) in perm.iter().enumerate() {
            diag[p as usize] += v * v;
            for &(q, w) in &perm[i + 1..] {
                if q == p {
                    diag[p as usize] += v * w;
                    continue;
                }
                let col = col_starts[p as usize] as usize..col_starts[p as usize + 1] as usize;
                let at = row_idx[col.clone()]
                    .binary_search(&q)
                    .expect("A entry is in L")
                    + col.start;
                vals[at] += v * w;
            }
        }
    }
    numeric_ldlt(n, col_starts, row_idx, vals, diag, order)
}

/// Factors a symmetric positive definite matrix `A` given as unique or
/// duplicate triangle entries `(row, col, val)` (with each off-diagonal
/// undirected pair supplied once, or duplicate entries summed), eliminating in
/// `order` (`order[p]` is the original index at position `p`).
///
/// `None` when the matrix is not numerically positive definite (a zero,
/// negative, or non-finite pivot).
///
/// # Errors
/// [`OpsError::Store`] when memory runs out.
pub(crate) fn factor_spd_ordered(
    n: usize,
    entries: &[(u32, u32, f64)],
    order: &[u32],
) -> Result<Option<Factor>> {
    if n > u32::MAX as usize {
        return Err(OpsError::Store(StoreError::OutOfMemory));
    }
    let mut inv = filled(n, 0u32)?;
    for (p, &k) in order.iter().enumerate() {
        inv[k as usize] = p as u32;
    }
    let mut diag = filled(n, 0.0f64)?;
    let mut pairs: Vec<(u64, f64)> = Vec::new();
    reserve(&mut pairs, entries.len())?;
    for &(u, v, w) in entries {
        let (p, q) = (inv[u as usize], inv[v as usize]);
        if p == q {
            diag[p as usize] += w;
        } else {
            pairs.push(((p.max(q) as u64) << 32 | p.min(q) as u64, w));
        }
    }
    pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut up_starts: Vec<u32> = Vec::new();
    reserve(&mut up_starts, n + 1)?;
    up_starts.push(0);
    let mut up_rows: Vec<u32> = Vec::new();
    let mut up_vals: Vec<f64> = Vec::new();
    reserve(&mut up_rows, pairs.len())?;
    reserve(&mut up_vals, pairs.len())?;
    let mut cur_col = 0usize;
    let mut i = 0usize;
    while i < pairs.len() {
        let key = pairs[i].0;
        let col = (key >> 32) as usize;
        let row = key as u32;
        let mut sum = pairs[i].1;
        i += 1;
        while i < pairs.len() && pairs[i].0 == key {
            sum += pairs[i].1;
            i += 1;
        }
        while cur_col < col {
            up_starts.push(up_rows.len() as u32);
            cur_col += 1;
        }
        up_rows.push(row);
        up_vals.push(sum);
    }
    while cur_col < n {
        up_starts.push(up_rows.len() as u32);
        cur_col += 1;
    }
    let parent = etree(n, &up_starts, &up_rows)?;
    let (col_starts, row_idx) = patterns(n, &up_starts, &up_rows, &parent)?;
    let mut vals = filled(row_idx.len(), 0.0f64)?;
    for k in 0..n {
        let k_u32 = k as u32;
        for t in up_starts[k] as usize..up_starts[k + 1] as usize {
            let p = up_rows[t] as usize;
            let col = col_starts[p] as usize..col_starts[p + 1] as usize;
            let at = row_idx[col.clone()]
                .binary_search(&k_u32)
                .expect("A entry is in L")
                + col.start;
            vals[at] += up_vals[t];
        }
    }
    numeric_ldlt(n, col_starts, row_idx, vals, diag, order)
}

/// Shared left-looking numeric LDLᵀ over the symbolic pattern `(col_starts, row_idx)`.
fn numeric_ldlt(
    n: usize,
    col_starts: Vec<u32>,
    row_idx: Vec<u32>,
    mut vals: Vec<f64>,
    mut diag: Vec<f64>,
    order: &[u32],
) -> Result<Option<Factor>> {
    let total = row_idx.len();
    // The row lists: for each row, the (column, value slot) pairs above it,
    // in column order.
    let mut row_counts = filled(n, 0u32)?;
    for &r in &row_idx {
        row_counts[r as usize] += 1;
    }
    let mut row_starts: Vec<u32> = Vec::new();
    reserve(&mut row_starts, n + 1)?;
    row_starts.push(0);
    for c in &row_counts {
        row_starts.push(row_starts.last().expect("nonempty") + c);
    }
    let mut row_cols: Vec<(u32, u32)> = Vec::new();
    reserve(&mut row_cols, total)?;
    row_cols.resize(total, (0, 0));
    let mut cursor: Vec<u32> = Vec::new();
    reserve(&mut cursor, n + 1)?;
    cursor.extend_from_slice(&row_starts);
    for p in 0..n {
        for t in col_starts[p] as usize..col_starts[p + 1] as usize {
            let at = cursor[row_idx[t] as usize] as usize;
            row_cols[at] = (p as u32, t as u32);
            cursor[row_idx[t] as usize] += 1;
        }
    }
    // Numeric LDLᵀ, left-looking: column j gathers the earlier columns
    // that reach it, scattered through the marked rows of its own pattern.
    let mut mark = filled(n, 0u32)?;
    let mut acc = filled(n, 0.0f64)?;
    for j in 0..n {
        let stamp = j as u32 + 1;
        let krow = row_starts[j] as usize..row_starts[j + 1] as usize;
        let mut d = diag[j];
        for &(k, s) in &row_cols[krow.clone()] {
            let l = vals[s as usize];
            d -= l * l * diag[k as usize];
        }
        if !(d.is_finite() && d > 0.0) {
            return Ok(None);
        }
        diag[j] = d;
        let col = col_starts[j] as usize..col_starts[j + 1] as usize;
        if col.is_empty() {
            continue;
        }
        for t in col.clone() {
            mark[row_idx[t] as usize] = stamp;
            acc[row_idx[t] as usize] = vals[t];
        }
        for &(k, s) in &row_cols[krow] {
            let f = vals[s as usize] * diag[k as usize];
            let kend = col_starts[k as usize + 1] as usize;
            for t in s as usize + 1..kend {
                let i = row_idx[t];
                if mark[i as usize] == stamp {
                    acc[i as usize] -= vals[t] * f;
                }
            }
        }
        for t in col {
            let v = acc[row_idx[t] as usize] / d;
            if !v.is_finite() {
                return Ok(None);
            }
            vals[t] = v;
        }
    }
    let mut kept = Vec::new();
    reserve(&mut kept, n)?;
    kept.extend_from_slice(order);
    Ok(Some(Factor {
        col_starts,
        row_idx,
        vals,
        diag,
        order: kept,
    }))
}

/// Result of a Preconditioned Conjugate Gradient solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PcgResult {
    pub(crate) converged: bool,
    pub(crate) iterations: usize,
    pub(crate) residual: f64,
}

/// Borrowed Compressed Sparse Row (CSR) view of a square symmetric matrix.
pub(crate) struct SymCsr<'a> {
    pub(crate) n: usize,
    pub(crate) row_starts: &'a [u32],
    pub(crate) cols: &'a [u32],
    pub(crate) vals: &'a [f64],
}

/// Solves a symmetric positive definite CSR system `A x = rhs` via
/// Jacobi-preconditioned Conjugate Gradient.
///
/// # Errors
/// [`OpsError::Store`] when memory runs out.
pub(crate) fn solve_pcg(
    csr: &SymCsr<'_>,
    rhs: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iters: usize,
) -> Result<PcgResult> {
    let SymCsr {
        n,
        row_starts,
        cols,
        vals,
    } = *csr;
    if n == 0 {
        return Ok(PcgResult {
            converged: true,
            iterations: 0,
            residual: 0.0,
        });
    }
    let mut inv_diag = filled(n, 0.0f64)?;
    for i in 0..n {
        let mut diag = 0.0;
        for t in row_starts[i] as usize..row_starts[i + 1] as usize {
            if cols[t] as usize == i {
                diag += vals[t];
            }
        }
        if !(diag.is_finite() && diag > 0.0) {
            return Ok(PcgResult {
                converged: false,
                iterations: 0,
                residual: f64::INFINITY,
            });
        }
        inv_diag[i] = 1.0 / diag;
    }

    let mut r = filled(n, 0.0f64)?;
    let mut z = filled(n, 0.0f64)?;
    let mut p = filled(n, 0.0f64)?;
    let mut ap = filled(n, 0.0f64)?;

    let rhs_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut r_norm2 = 0.0;
    for i in 0..n {
        let mut ax = 0.0;
        for t in row_starts[i] as usize..row_starts[i + 1] as usize {
            ax += vals[t] * x[cols[t] as usize];
        }
        r[i] = rhs[i] - ax;
        r_norm2 += r[i] * r[i];
    }
    let ref_norm = if rhs_norm > 0.0 {
        rhs_norm
    } else {
        r_norm2.sqrt()
    };
    if !r_norm2.is_finite() {
        return Ok(PcgResult {
            converged: false,
            iterations: 0,
            residual: f64::INFINITY,
        });
    }
    if ref_norm == 0.0 || r_norm2.sqrt() <= tol * ref_norm {
        return Ok(PcgResult {
            converged: true,
            iterations: 0,
            residual: if ref_norm > 0.0 {
                r_norm2.sqrt() / ref_norm
            } else {
                0.0
            },
        });
    }

    let mut rz = 0.0;
    for i in 0..n {
        z[i] = inv_diag[i] * r[i];
        p[i] = z[i];
        rz += r[i] * z[i];
    }

    let mut iterations = 0usize;
    let mut converged = false;
    let mut rel_res = r_norm2.sqrt() / ref_norm;

    for iter in 0..max_iters {
        iterations = iter + 1;
        let mut pap = 0.0;
        for i in 0..n {
            let mut sum = 0.0;
            for t in row_starts[i] as usize..row_starts[i + 1] as usize {
                sum += vals[t] * p[cols[t] as usize];
            }
            ap[i] = sum;
            pap += p[i] * sum;
        }
        if !(pap.is_finite() && pap > 0.0) {
            break;
        }
        let alpha = rz / pap;
        if !alpha.is_finite() {
            break;
        }
        r_norm2 = 0.0;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
            r_norm2 += r[i] * r[i];
        }
        if !r_norm2.is_finite() {
            return Ok(PcgResult {
                converged: false,
                iterations,
                residual: f64::INFINITY,
            });
        }
        rel_res = r_norm2.sqrt() / ref_norm;
        if rel_res <= tol {
            converged = true;
            break;
        }
        let mut rz_next = 0.0;
        for i in 0..n {
            z[i] = inv_diag[i] * r[i];
            rz_next += r[i] * z[i];
        }
        if !(rz_next.is_finite() && rz != 0.0) {
            break;
        }
        let beta = rz_next / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_next;
    }

    Ok(PcgResult {
        converged,
        iterations,
        residual: rel_res,
    })
}

/// The elimination tree of the permuted `BᵀB`, from its strict upper
/// triangle in compressed columns: `parent[j]` is the least row past `j`
/// reached from `j` through lesser vertices, [`NONE`] at a root.
fn etree(n: usize, up_starts: &[u32], up_rows: &[u32]) -> Result<Vec<u32>> {
    let mut parent = filled(n, NONE)?;
    let mut ancestor = filled(n, NONE)?;
    for k in 0..n {
        let column = up_starts[k] as usize..up_starts[k + 1] as usize;
        for &row in &up_rows[column] {
            let mut i = row;
            while i != NONE && i < k as u32 {
                let next = ancestor[i as usize];
                ancestor[i as usize] = k as u32;
                if next == NONE {
                    parent[i as usize] = k as u32;
                }
                i = next;
            }
        }
    }
    Ok(parent)
}

/// The strict columns of `L` in compressed columns, from the strict upper
/// triangle of `BᵀB` and its elimination tree: each row's earlier pattern
/// is found by walking the tree up from the row's entries, and the columns
/// are the transpose, each ascending.
fn patterns(
    n: usize,
    up_starts: &[u32],
    up_rows: &[u32],
    parent: &[u32],
) -> Result<(Vec<u32>, Vec<u32>)> {
    let mut mark = filled(n, 0u32)?;
    let mut row_starts: Vec<u32> = Vec::new();
    reserve(&mut row_starts, n + 1)?;
    row_starts.push(0);
    let mut row_cols: Vec<u32> = Vec::new();
    for r in 0..n {
        let stamp = r as u32 + 1;
        mark[r] = stamp;
        let column = up_starts[r] as usize..up_starts[r + 1] as usize;
        for &row in &up_rows[column] {
            let mut i = row;
            while mark[i as usize] != stamp {
                mark[i as usize] = stamp;
                push(&mut row_cols, i)?;
                i = parent[i as usize];
                if i == NONE {
                    break;
                }
            }
        }
        row_starts.push(row_cols.len() as u32);
    }
    if row_cols.len() > u32::MAX as usize {
        return Err(OpsError::Store(StoreError::OutOfMemory));
    }
    let mut counts = filled(n, 0u32)?;
    for &c in &row_cols {
        counts[c as usize] += 1;
    }
    let mut col_starts: Vec<u32> = Vec::new();
    reserve(&mut col_starts, n + 1)?;
    col_starts.push(0);
    for c in &counts {
        col_starts.push(col_starts.last().expect("nonempty") + c);
    }
    let mut row_idx: Vec<u32> = Vec::new();
    reserve(&mut row_idx, row_cols.len())?;
    row_idx.resize(row_cols.len(), 0);
    let mut cursor: Vec<u32> = Vec::new();
    reserve(&mut cursor, n + 1)?;
    cursor.extend_from_slice(&col_starts);
    for r in 0..n {
        let entries = row_starts[r] as usize..row_starts[r + 1] as usize;
        for &c in &row_cols[entries] {
            let at = cursor[c as usize] as usize;
            row_idx[at] = r as u32;
            cursor[c as usize] += 1;
        }
    }
    Ok((col_starts, row_idx))
}

/// Sets this small are ordered as they come, not split further.
const ND_LEAF: usize = 32;

/// A nested-dissection elimination order of `n` vertices with `positions`
/// and undirected `edges` (each once, either direction): `order[p]` is the
/// original index eliminated at position `p`.
///
/// The vertices split recursively along the longest axis of their bounding
/// box, at the median; the vertices adjacent to both halves (which the two
/// halves minus them disconnect) order last, after the halves. Small sets
/// keep their arrival order. Deterministic: every tie breaks to the
/// smallest index.
///
/// # Errors
/// [`OpsError::Store`] when memory runs out.
pub(crate) fn nested_dissection(
    n: usize,
    positions: &[[f64; 3]],
    edges: &[(u32, u32)],
) -> Result<Vec<u32>> {
    let mut adj: Vec<Vec<u32>> = Vec::new();
    reserve(&mut adj, n)?;
    adj.resize(n, Vec::new());
    for &(a, b) in edges {
        push(&mut adj[a as usize], b)?;
        push(&mut adj[b as usize], a)?;
    }
    let mut members: Vec<u32> = Vec::new();
    reserve(&mut members, n)?;
    members.extend(0..n as u32);
    let mut marks = filled(n, 0u32)?;
    let mut order: Vec<u32> = Vec::new();
    reserve(&mut order, n)?;
    split(members, positions, &adj, &mut marks, &mut 1, &mut order)?;
    Ok(order)
}

/// Appends the nested-dissection order of `members` to `out`.
fn split(
    mut members: Vec<u32>,
    positions: &[[f64; 3]],
    adj: &[Vec<u32>],
    marks: &mut [u32],
    stamp: &mut u32,
    out: &mut Vec<u32>,
) -> Result<()> {
    if members.len() <= ND_LEAF {
        reserve(out, members.len())?;
        out.extend_from_slice(&members);
        return Ok(());
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &v in &members {
        for k in 0..3 {
            let c = positions[v as usize][k];
            lo[k] = lo[k].min(c);
            hi[k] = hi[k].max(c);
        }
    }
    let mut axis = 0;
    for k in 1..3 {
        if hi[k] - lo[k] > hi[axis] - lo[axis] {
            axis = k;
        }
    }
    members.sort_by(|&a, &b| {
        positions[a as usize][axis]
            .total_cmp(&positions[b as usize][axis])
            .then(a.cmp(&b))
    });
    let mid = members.len() / 2;
    let right = members.split_off(mid);
    let left = members;
    // The separator: the left vertices adjacent to the right half, then the
    // right vertices adjacent to what is left of the left half. What
    // remains on each side is disconnected from the other side.
    let (stamp_r, stamp_l) = (*stamp, *stamp + 1);
    *stamp += 2;
    for &v in &right {
        marks[v as usize] = stamp_r;
    }
    let mut sep: Vec<u32> = Vec::new();
    let mut left2: Vec<u32> = Vec::new();
    for &v in &left {
        if adj[v as usize]
            .iter()
            .any(|&u| marks[u as usize] == stamp_r)
        {
            push(&mut sep, v)?;
        } else {
            push(&mut left2, v)?;
        }
    }
    for &v in &left2 {
        marks[v as usize] = stamp_l;
    }
    let mut right2: Vec<u32> = Vec::new();
    for &v in &right {
        if adj[v as usize]
            .iter()
            .any(|&u| marks[u as usize] == stamp_l)
        {
            push(&mut sep, v)?;
        } else {
            push(&mut right2, v)?;
        }
    }
    split(left2, positions, adj, marks, stamp, out)?;
    split(right2, positions, adj, marks, stamp, out)?;
    reserve(out, sep.len())?;
    out.extend_from_slice(&sep);
    Ok(())
}

impl Factor {
    /// Solves `BᵀB x = rhs`, in the original numbering.
    ///
    /// # Errors
    /// [`OpsError::Store`] when memory runs out.
    pub(crate) fn solve(&self, rhs: &[f64], x: &mut [f64]) -> Result<()> {
        let n = self.order.len();
        let mut y = filled(n, 0.0)?;
        for (i, &k) in self.order.iter().enumerate() {
            y[i] = rhs[k as usize];
        }
        for k in 0..n {
            let yk = y[k];
            for t in self.col_starts[k] as usize..self.col_starts[k + 1] as usize {
                y[self.row_idx[t] as usize] -= self.vals[t] * yk;
            }
        }
        for (yk, d) in y.iter_mut().zip(self.diag.iter()) {
            *yk /= *d;
        }
        for k in (0..n).rev() {
            let mut s = y[k];
            for t in self.col_starts[k] as usize..self.col_starts[k + 1] as usize {
                s -= self.vals[t] * y[self.row_idx[t] as usize];
            }
            y[k] = s;
        }
        for (i, &k) in self.order.iter().enumerate() {
            x[k as usize] = y[i];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `B` in compressed rows, as [`factor_ordered`] takes it.
    fn assemble(rows: &[Vec<(u32, f64)>]) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
        let mut starts = vec![0u32];
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for r in rows {
            for &(c, v) in r {
                cols.push(c);
                vals.push(v);
            }
            starts.push(cols.len() as u32);
        }
        (starts, cols, vals)
    }

    /// `|A x - rhs| / |rhs|` with `A = BᵀB` formed densely.
    fn residual(
        starts: &[u32],
        cols: &[u32],
        vals: &[f64],
        n: usize,
        x: &[f64],
        rhs: &[f64],
    ) -> f64 {
        let mut a = vec![0.0; n * n];
        for k in 0..starts.len() - 1 {
            let range = starts[k] as usize..starts[k + 1] as usize;
            for (i, &c) in cols[range.clone()].iter().enumerate() {
                for (j, &d) in cols[range.clone()].iter().enumerate() {
                    a[c as usize * n + d as usize] += vals[range.start + i] * vals[range.start + j];
                }
            }
        }
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut r = rhs.to_vec();
        for i in 0..n {
            for j in 0..n {
                r[i] -= a[i * n + j] * x[j];
            }
        }
        norm(&r) / norm(rhs)
    }

    #[test]
    fn small_system_matches_known_solution() {
        // B = [[2, 0], [1, 1], [0, 3]]: A = BᵀB = [[5, 1], [1, 10]].
        let (starts, cols, vals) =
            assemble(&[vec![(0, 2.0)], vec![(0, 1.0), (1, 1.0)], vec![(1, 3.0)]]);
        let f = factor_ordered(&starts, &cols, &vals, 2, &[0, 1])
            .unwrap()
            .unwrap();
        // A [1, 2] = [7, 21].
        let mut x = [0.0; 2];
        f.solve(&[7.0, 21.0], &mut x).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12);
        assert!(residual(&starts, &cols, &vals, 2, &x, &[7.0, 21.0]) < 1e-14);
    }

    #[test]
    fn band_system_matches_dense_elimination() {
        // A 50-column B with a band plus closing corners (a ring): each row
        // couples neighbours, so A is sparse but far from diagonal.
        let n = 50usize;
        let mut lcg = 0x1234_5678_9abcu64;
        let mut next = || {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (lcg >> 33) as f64 / u32::MAX as f64 - 0.5
        };
        let mut rows: Vec<Vec<(u32, f64)>> = Vec::new();
        for k in 0..80 {
            let c = (k % n) as u32;
            rows.push(vec![
                (c, 2.0 + next()),
                ((k + 1) as u32 % n as u32, next()),
                ((k + 7) as u32 % n as u32, 0.5 * next()),
            ]);
        }
        let (starts, cols, vals) = assemble(&rows);
        let order: Vec<u32> = (0..n as u32).collect();
        let f = factor_ordered(&starts, &cols, &vals, n, &order)
            .unwrap()
            .unwrap();
        let rhs: Vec<f64> = (0..n).map(|_| next()).collect();
        let mut x = vec![0.0; n];
        f.solve(&rhs, &mut x).unwrap();
        assert!(residual(&starts, &cols, &vals, n, &x, &rhs) < 1e-12);
    }

    #[test]
    fn singular_system_gives_none() {
        // Column 1 is empty: BᵀB is singular.
        let (starts, cols, vals) = assemble(&[vec![(0, 1.0)], vec![(0, 2.0)]]);
        assert!(
            factor_ordered(&starts, &cols, &vals, 2, &[0, 1])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn empty_system_solves_trivially() {
        let f = factor_ordered(&[0], &[], &[], 0, &[]).unwrap().unwrap();
        f.solve(&[], &mut []).unwrap();
    }

    #[test]
    fn etree_and_patterns_match_hand_worked_cases() {
        // A path 0-1-2-3: no fill; the tree is the path itself.
        let up = (vec![0, 0, 1, 2, 3], vec![0, 1, 2]);
        let parent = etree(4, &up.0, &up.1).unwrap();
        assert_eq!(parent, vec![1, 2, 3, NONE]);
        let (starts, rows) = patterns(4, &up.0, &up.1, &parent).unwrap();
        assert_eq!((starts, rows), (vec![0, 1, 2, 3, 3], vec![1, 2, 3]));
        // A star at 0: eliminating 0 first fills the triangle on 1, 2, 3.
        let up = (vec![0, 0, 1, 2, 3], vec![0, 0, 0]);
        let parent = etree(4, &up.0, &up.1).unwrap();
        assert_eq!(parent, vec![1, 2, 3, NONE]);
        let (starts, rows) = patterns(4, &up.0, &up.1, &parent).unwrap();
        assert_eq!(
            (starts, rows),
            (vec![0, 3, 5, 6, 6], vec![1, 2, 3, 2, 3, 3])
        );
    }

    #[test]
    fn nested_dissection_orders_a_grid_deterministically() {
        // An 8x8 grid, row-major: the order is a permutation, run twice the
        // same, with a middle separator ordered last.
        let n = 8u32;
        let positions: Vec<[f64; 3]> = (0..n * n)
            .map(|v| [(v % n) as f64, (v / n) as f64, 0.0])
            .collect();
        let mut edges = Vec::new();
        for y in 0..n {
            for x in 0..n {
                if x + 1 < n {
                    edges.push((y * n + x, y * n + x + 1));
                }
                if y + 1 < n {
                    edges.push((y * n + x, (y + 1) * n + x));
                }
            }
        }
        let order = nested_dissection((n * n) as usize, &positions, &edges).unwrap();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n * n).collect::<Vec<_>>());
        assert_eq!(
            order,
            nested_dissection((n * n) as usize, &positions, &edges).unwrap()
        );
        // The first split is vertical (x and y tie, x wins): the last
        // vertices are the middle columns' separator.
        let tail: Vec<u32> = order[order.len() - 8..].to_vec();
        assert!(tail.iter().all(|&v| v % n == 3 || v % n == 4));
    }

    #[test]
    fn factor_spd_ordered_solves_merged_triplets_and_rejects_singular() {
        // 3x3 SPD system with duplicate off-diagonal and diagonal entries:
        // [4 -1  0]   [1]   [2]
        // [-1 4 -1] * [2] = [4]
        // [0 -1  4]   [3]   [10]
        let entries = [
            (0, 0, 2.5),
            (0, 0, 1.5),
            (0, 1, -0.4),
            (1, 0, -0.6),
            (1, 1, 4.0),
            (2, 1, -1.0),
            (2, 2, 4.0),
        ];
        let order = [2, 0, 1];
        let factor = factor_spd_ordered(3, &entries, &order)
            .expect("ok")
            .expect("spd");
        let rhs = [2.0, 4.0, 10.0];
        let mut x = [0.0; 3];
        factor.solve(&rhs, &mut x).expect("solve ok");
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
        assert!((x[2] - 3.0).abs() < 1e-12);

        // Singular / indefinite matrix returns Ok(None).
        let singular = [(0, 0, 1.0), (1, 1, 1.0), (0, 1, 1.0)];
        assert!(
            factor_spd_ordered(2, &singular, &[0, 1])
                .expect("ok")
                .is_none()
        );

        // Empty system succeeds cleanly.
        let empty = factor_spd_ordered(0, &[], &[])
            .expect("ok")
            .expect("empty spd");
        let mut empty_x = [];
        empty.solve(&[], &mut empty_x).expect("empty solve ok");
    }

    #[test]
    fn solve_pcg_converges_and_handles_edge_cases() {
        // 4x4 1D Laplacian with Dirichlet endpoints: diag=2, offdiag=-1.
        let row_starts = [0u32, 2, 5, 8, 10];
        let cols = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let vals = [2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
        // True x = [1, 2, 3, 4] -> A x = [0, 0, 0, 5]
        let rhs = [0.0, 0.0, 0.0, 5.0];
        let csr = SymCsr {
            n: 4,
            row_starts: &row_starts,
            cols: &cols,
            vals: &vals,
        };
        let mut x = [0.0; 4];
        let res = solve_pcg(&csr, &rhs, &mut x, 1e-12, 64).expect("ok");
        assert!(res.converged);
        assert!(res.residual < 1e-11);
        for (i, &xi) in x.iter().enumerate() {
            assert!((xi - (i + 1) as f64).abs() < 1e-10, "x[{i}] = {xi}");
        }

        // Zero RHS with zero initial guess converges in 0 iterations.
        let mut zero_x = [0.0; 4];
        let zres = solve_pcg(&csr, &[0.0; 4], &mut zero_x, 1e-12, 16).expect("ok");
        assert!(zres.converged);
        assert_eq!(zres.iterations, 0);

        // Zero diagonal stops cleanly with converged = false.
        let bad_vals = [0.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
        let bad_csr = SymCsr {
            n: 4,
            row_starts: &row_starts,
            cols: &cols,
            vals: &bad_vals,
        };
        let mut bx = [0.0; 4];
        let bres = solve_pcg(&bad_csr, &rhs, &mut bx, 1e-12, 16).expect("ok");
        assert!(!bres.converged);
    }
}
