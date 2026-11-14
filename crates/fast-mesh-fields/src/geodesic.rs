//! Intrinsic geodesic distance fields on triangle meshes.
//!
//! Supports two complementary methods:
//! - [`GeodesicMethod::Dijkstra`]: exact graph shortest paths over triangle
//!   edges with Euclidean edge weights.
//! - [`GeodesicMethod::HeatMethod`]: Crane, Weischedel & Wardetzky (2013),
//!   *Geodesics in Heat*, solving a backward-Euler heat diffusion step on the
//!   clamped cotangent Laplacian followed by unit-gradient normalization and a
//!   Poisson distance reconstruction.
//!
//! For repeated queries on the same mesh with varying source sets, build
//! [`GeodesicHeatSolver`] once and call [`GeodesicHeatSolver::solve`].

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use fast_mesh_store::{Mesh, par};

use crate::geom::{add, cross, dot, norm, scale, sub, triangle_normal, widen};
use crate::grid::{allocation_allowed, filled, push, reserve_exact};
use crate::{FieldsError, Result};

/// Row threshold above which PCG vector/matrix kernels dispatch to rayon chunks.
const PAR_MIN_ROWS: usize = 16_384;
/// Fixed row/element chunk size for deterministic parallel PCG operations.
const PAR_CHUNK: usize = 1024;
/// Minimum positive off-diagonal conductivity (`w_ij`) for clamped cotangent edges
/// so the positive-semidefinite Laplacian remains an irreducible symmetric M-matrix.
const MIN_EDGE_WEIGHT: f64 = 1e-6;

/// Algorithm used by [`geodesic_distances`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeodesicMethod {
    /// Multi-source Dijkstra along mesh edges weighted by Euclidean edge length.
    Dijkstra,
    /// Crane et al. (2013) *Geodesics in Heat* via clamped cotangent Laplacian
    /// and IC(0)-preconditioned Conjugate Gradient.
    HeatMethod,
}

/// Configuration for [`geodesic_distances`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodesicOptions {
    /// Distance computation method. Default: [`GeodesicMethod::HeatMethod`].
    pub method: GeodesicMethod,
    /// Optional distance cutoff; vertices farther than this bound (or on
    /// unreachable components) receive `f32::INFINITY`.
    pub max_distance: Option<f32>,
    /// Multiplier `m` for the heat diffusion time step `t = m * h_mean^2`.
    /// Must be finite and strictly positive. Default: `1.0`.
    pub time_step_scale: f64,
}

impl Default for GeodesicOptions {
    fn default() -> Self {
        Self {
            method: GeodesicMethod::HeatMethod,
            max_distance: None,
            time_step_scale: 1.0,
        }
    }
}

/// Computes single- or multi-source geodesic distances from `sources` to every
/// vertex slot of `mesh`, writing `mesh.vertex_slots()` values into `out`.
///
/// Unreached vertices (disconnected components without a source, dead or unused
/// vertex slots, and values exceeding `options.max_distance`) receive
/// `f32::INFINITY`.
///
/// With the `parallel` feature, large meshes accelerate PCG operations across
/// fixed chunks with bit-identical results across thread counts.
///
/// # Errors
/// - [`FieldsError::InvalidInput`] if `sources` is empty, any source index is
///   out of bounds or dead, `options.time_step_scale` is non-positive/non-finite,
///   `options.max_distance` is negative or non-finite, or a live vertex position
///   referenced by live triangles or `sources` is non-finite.
/// - [`FieldsError::OutOfMemory`] if any internal or output allocation fails.
///
/// On error `out` is left unchanged.
pub fn geodesic_distances(
    mesh: &Mesh,
    sources: &[u32],
    options: &GeodesicOptions,
    out: &mut Vec<f32>,
) -> Result<()> {
    validate_options(options.time_step_scale, options.max_distance)?;
    validate_sources_against_mesh(mesh, sources)?;

    match options.method {
        GeodesicMethod::Dijkstra => dijkstra_distances(mesh, sources, options.max_distance, out),
        GeodesicMethod::HeatMethod => {
            let solver = GeodesicHeatSolver::build(mesh, options.time_step_scale)?;
            solver.solve(sources, options.max_distance, out)
        }
    }
}

fn validate_options(time_step_scale: f64, max_distance: Option<f32>) -> Result<()> {
    if !time_step_scale.is_finite() || time_step_scale <= 0.0 {
        return Err(FieldsError::InvalidInput(
            "time_step_scale must be finite and strictly positive",
        ));
    }
    if let Some(d) = max_distance
        && (!d.is_finite() || d < 0.0)
    {
        return Err(FieldsError::InvalidInput(
            "max_distance must be finite and non-negative",
        ));
    }
    Ok(())
}

fn validate_sources_against_mesh(mesh: &Mesh, sources: &[u32]) -> Result<()> {
    if sources.is_empty() {
        return Err(FieldsError::InvalidInput("sources must not be empty"));
    }
    let slots = mesh.vertex_slots();
    let pos = mesh.positions();
    for &s in sources {
        let idx = s as usize;
        if idx >= slots || !mesh.is_vertex_alive(s) {
            return Err(FieldsError::InvalidInput(
                "source vertex index is out of range or dead",
            ));
        }
        if !pos[idx].iter().all(|c| c.is_finite()) {
            return Err(FieldsError::InvalidInput(
                "source vertex position must be finite",
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
struct HeapItem {
    dist: f64,
    vertex: u32,
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

fn heap_push(heap: &mut BinaryHeap<HeapItem>, item: HeapItem) -> Result<()> {
    if heap.len() == heap.capacity() {
        allocation_allowed()?;
        heap.try_reserve(1).map_err(|_| FieldsError::OutOfMemory)?;
    }
    heap.push(item);
    Ok(())
}

fn dijkstra_distances(
    mesh: &Mesh,
    sources: &[u32],
    max_distance: Option<f32>,
    out: &mut Vec<f32>,
) -> Result<()> {
    let slots = mesh.vertex_slots();
    let pos = mesh.positions();

    let edge_cap = mesh.triangle_count().saturating_mul(6);
    let mut raw_edges: Vec<(u32, u32)> = Vec::new();
    reserve_exact(&mut raw_edges, edge_cap)?;

    for t in 0..mesh.triangle_slots() as u32 {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        let corners = tri.map(|v| pos[v as usize]);
        if !corners.iter().flatten().all(|c| c.is_finite()) {
            return Err(FieldsError::InvalidInput(
                "mesh vertex position must be finite",
            ));
        }
        for k in 0..3 {
            let a = tri[k];
            let b = tri[(k + 1) % 3];
            if a != b {
                push(&mut raw_edges, (a, b))?;
                push(&mut raw_edges, (b, a))?;
            }
        }
    }

    raw_edges.sort_unstable();
    raw_edges.dedup();

    let mut offsets = filled(slots + 1, 0u32)?;
    for &(u, _) in &raw_edges {
        offsets[u as usize + 1] += 1;
    }
    for i in 0..slots {
        offsets[i + 1] += offsets[i];
    }

    let mut neighbors: Vec<(u32, f64)> = Vec::new();
    reserve_exact(&mut neighbors, raw_edges.len())?;
    for &(u, v) in &raw_edges {
        let len = norm(sub(widen(pos[v as usize]), widen(pos[u as usize])));
        neighbors.push((v, len));
    }
    drop(raw_edges);

    let mut dist = filled(slots, f64::INFINITY)?;
    let mut heap = BinaryHeap::new();
    for &s in sources {
        if dist[s as usize] > 0.0 {
            dist[s as usize] = 0.0;
            heap_push(
                &mut heap,
                HeapItem {
                    dist: 0.0,
                    vertex: s,
                },
            )?;
        }
    }

    let max_d = max_distance.map(f64::from).unwrap_or(f64::INFINITY);
    while let Some(HeapItem { dist: d, vertex: u }) = heap.pop() {
        if d > max_d {
            break;
        }
        if d > dist[u as usize] {
            continue;
        }
        let start = offsets[u as usize] as usize;
        let end = offsets[u as usize + 1] as usize;
        for &(v, w) in &neighbors[start..end] {
            let nd = d + w;
            if nd <= max_d && nd < dist[v as usize] {
                dist[v as usize] = nd;
                heap_push(
                    &mut heap,
                    HeapItem {
                        dist: nd,
                        vertex: v,
                    },
                )?;
            }
        }
    }

    if out.capacity() < slots {
        reserve_exact(out, slots - out.len())?;
    }
    out.clear();
    out.resize(slots, f32::INFINITY);
    for (v, dst) in out.iter_mut().enumerate() {
        if dist[v].is_finite() {
            let val = dist[v] as f32;
            *dst = if let Some(limit) = max_distance {
                if val > limit { f32::INFINITY } else { val }
            } else {
                val
            };
        }
    }
    Ok(())
}

/// Triangle geometry precomputed in a component's local vertex numbering.
#[derive(Debug, Clone)]
struct LocalTri {
    verts: [u32; 3],
    area: f64,
    grad_vecs: [[f64; 3]; 3],
}

/// Incomplete Cholesky `L L^T` preconditioner (`Lower`) or diagonal Jacobi
/// fallback (`Diagonal`) for a symmetric SPD sparse matrix.
#[derive(Debug, Clone)]
enum Preconditioner {
    IncompleteCholesky {
        row_ptr: Vec<usize>,
        col_idx: Vec<u32>,
        values: Vec<f64>,
        inv_diag: Vec<f64>,
    },
    Diagonal {
        inv_diag: Vec<f64>,
    },
}

/// Pre-assembled linear operators and preconditioners for one connected
/// triangle-mesh component.
#[derive(Debug, Clone)]
struct HeatComponent {
    global_verts: Vec<u32>,
    avg_mass: f64,
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    heat_vals: Vec<f64>,
    poisson_vals: Vec<f64>,
    heat_precond: Preconditioner,
    poisson_precond: Preconditioner,
    tris: Vec<LocalTri>,
}

/// Precomputed solver for Crane et al. (2013) *Geodesics in Heat* on a fixed
/// mesh and diffusion time step.
///
/// Building the solver assembles the lumped mass matrix, clamped cotangent
/// Laplacian, and IC(0) preconditioners per connected component so subsequent
/// [`GeodesicHeatSolver::solve`] calls for different source sets avoid
/// rebuilding the operators.
#[derive(Debug, Clone)]
pub struct GeodesicHeatSolver {
    vertex_slots: usize,
    source_valid: Vec<bool>,
    component_of: Vec<u32>,
    local_of: Vec<u32>,
    components: Vec<HeatComponent>,
}

impl GeodesicHeatSolver {
    /// Builds a reusable heat-method solver for `mesh` using diffusion time step
    /// `t = time_step_scale * h_mean^2`, where `h_mean` is the mean edge length
    /// across unique edges of live non-degenerate triangles.
    ///
    /// # Errors
    /// - [`FieldsError::InvalidInput`] if `time_step_scale` is non-finite or
    ///   `<= 0.0`, or if any vertex referenced by a live triangle has a
    ///   non-finite coordinate.
    /// - [`FieldsError::OutOfMemory`] if any internal buffer cannot be reserved.
    pub fn build(mesh: &Mesh, time_step_scale: f64) -> Result<Self> {
        validate_options(time_step_scale, None)?;

        let slots = mesh.vertex_slots();
        let pos = mesh.positions();

        let mut source_valid = filled(slots, false)?;
        for (v, flag) in source_valid.iter_mut().enumerate() {
            if mesh.is_vertex_alive(v as u32) && pos[v].iter().all(|c| c.is_finite()) {
                *flag = true;
            }
        }

        let mut active = filled(slots, false)?;
        let mut vertex_mass = filled(slots, 0.0f64)?;
        let mut tris_global: Vec<(LocalTri, [u32; 3])> = Vec::new();
        reserve_exact(&mut tris_global, mesh.triangle_count())?;
        let mut raw_contribs: Vec<(u32, u32, f64)> = Vec::new();
        reserve_exact(&mut raw_contribs, mesh.triangle_count().saturating_mul(3))?;

        for t in 0..mesh.triangle_slots() as u32 {
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };
            let corners = tri.map(|v| pos[v as usize]);
            if !corners.iter().flatten().all(|c| c.is_finite()) {
                return Err(FieldsError::InvalidInput(
                    "mesh vertex position must be finite",
                ));
            }
            if tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0] {
                continue;
            }

            let [pa, pb, pc] = corners.map(widen);
            let n = triangle_normal(corners);
            let area2 = norm(n);
            let e0 = sub(pc, pb);
            let e1 = sub(pa, pc);
            let e2 = sub(pb, pa);
            let max_e_sq = dot(e0, e0).max(dot(e1, e1)).max(dot(e2, e2));
            if !area2.is_finite() || area2 <= 1e-24 * max_e_sq || area2 == 0.0 {
                continue;
            }

            let area = 0.5 * area2;
            let n_hat = scale(n, 1.0 / area2);
            let inv_area2 = 1.0 / area2;
            let grad_vecs = [
                scale(cross(n_hat, e0), inv_area2),
                scale(cross(n_hat, e1), inv_area2),
                scale(cross(n_hat, e2), inv_area2),
            ];
            push(
                &mut tris_global,
                (
                    LocalTri {
                        verts: [0; 3],
                        area,
                        grad_vecs,
                    },
                    tri,
                ),
            )?;

            let tri_mass = area / 3.0;
            for &v in &tri {
                active[v as usize] = true;
                vertex_mass[v as usize] += tri_mass;
            }

            // Half-cotangent contributions for the three edges of the triangle.
            let c0 = 0.5 * dot(sub(pb, pa), sub(pc, pa)) * inv_area2;
            let c1 = 0.5 * dot(sub(pc, pb), sub(pa, pb)) * inv_area2;
            let c2 = 0.5 * dot(sub(pa, pc), sub(pb, pc)) * inv_area2;
            for ((u, v), c) in [
                ((tri[1], tri[2]), c0),
                ((tri[2], tri[0]), c1),
                ((tri[0], tri[1]), c2),
            ] {
                let (lo, hi) = if u < v { (u, v) } else { (v, u) };
                push(&mut raw_contribs, (lo, hi, c))?;
            }
        }

        let mut component_of = filled(slots, u32::MAX)?;
        let mut local_of = filled(slots, 0u32)?;
        if raw_contribs.is_empty() {
            return Ok(Self {
                vertex_slots: slots,
                source_valid,
                component_of,
                local_of,
                components: Vec::new(),
            });
        }

        raw_contribs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut unique_edges: Vec<(u32, u32, f64)> = Vec::new();
        reserve_exact(&mut unique_edges, raw_contribs.len())?;

        let mut edge_len_sum = 0.0f64;
        let mut idx = 0usize;
        while idx < raw_contribs.len() {
            let (u, v, mut c_sum) = raw_contribs[idx];
            idx += 1;
            while idx < raw_contribs.len() && raw_contribs[idx].0 == u && raw_contribs[idx].1 == v {
                c_sum += raw_contribs[idx].2;
                idx += 1;
            }
            let w = c_sum.max(MIN_EDGE_WEIGHT);
            push(&mut unique_edges, (u, v, w))?;
            edge_len_sum += norm(sub(widen(pos[v as usize]), widen(pos[u as usize])));
        }
        drop(raw_contribs);

        let h_mean = edge_len_sum / (unique_edges.len() as f64);
        let time_step = time_step_scale * h_mean * h_mean;

        // Build global undirected neighbor CSR over active vertices to find components.
        let mut deg = filled(slots + 1, 0usize)?;
        for &(u, v, _) in &unique_edges {
            deg[u as usize + 1] += 1;
            deg[v as usize + 1] += 1;
        }
        for i in 0..slots {
            deg[i + 1] += deg[i];
        }
        let mut adj_fill = filled(slots, 0usize)?;
        let mut adj = filled(deg[slots], 0u32)?;
        for &(u, v, _) in &unique_edges {
            let pu = deg[u as usize] + adj_fill[u as usize];
            adj[pu] = v;
            adj_fill[u as usize] += 1;
            let pv = deg[v as usize] + adj_fill[v as usize];
            adj[pv] = u;
            adj_fill[v as usize] += 1;
        }
        drop(adj_fill);

        let mut queue: Vec<u32> = Vec::new();
        let mut comp_verts_all: Vec<Vec<u32>> = Vec::new();
        for start in 0..slots {
            if !active[start] || component_of[start] != u32::MAX {
                continue;
            }
            let comp_id = comp_verts_all.len() as u32;
            let mut verts: Vec<u32> = Vec::new();
            queue.clear();
            component_of[start] = comp_id;
            push(&mut queue, start as u32)?;
            let mut qh = 0usize;
            while qh < queue.len() {
                let u = queue[qh];
                qh += 1;
                push(&mut verts, u)?;
                for &v in &adj[deg[u as usize]..deg[u as usize + 1]] {
                    if component_of[v as usize] == u32::MAX {
                        component_of[v as usize] = comp_id;
                        push(&mut queue, v)?;
                    }
                }
            }
            verts.sort_unstable();
            for (li, &gv) in verts.iter().enumerate() {
                local_of[gv as usize] = li as u32;
            }
            if comp_verts_all.len() == comp_verts_all.capacity() {
                allocation_allowed()?;
                comp_verts_all
                    .try_reserve(1)
                    .map_err(|_| FieldsError::OutOfMemory)?;
            }
            comp_verts_all.push(verts);
        }
        drop(queue);
        drop(adj);
        drop(deg);

        let num_comps = comp_verts_all.len();
        let mut comp_edges: Vec<Vec<(u32, u32, f64)>> = Vec::new();
        let mut comp_tris: Vec<Vec<LocalTri>> = Vec::new();
        reserve_exact(&mut comp_edges, num_comps)?;
        reserve_exact(&mut comp_tris, num_comps)?;
        for _ in 0..num_comps {
            comp_edges.push(Vec::new());
            comp_tris.push(Vec::new());
        }

        for &(u, v, w) in &unique_edges {
            let cid = component_of[u as usize] as usize;
            push(
                &mut comp_edges[cid],
                (local_of[u as usize], local_of[v as usize], w),
            )?;
        }
        drop(unique_edges);

        for (mut ltri, gtri) in tris_global {
            let cid = component_of[gtri[0] as usize] as usize;
            ltri.verts = [
                local_of[gtri[0] as usize],
                local_of[gtri[1] as usize],
                local_of[gtri[2] as usize],
            ];
            push(&mut comp_tris[cid], ltri)?;
        }

        let mut components: Vec<HeatComponent> = Vec::new();
        reserve_exact(&mut components, num_comps)?;
        for ((gverts, edges), tris) in comp_verts_all.into_iter().zip(comp_edges).zip(comp_tris) {
            components.push(build_component(
                gverts,
                edges,
                tris,
                &vertex_mass,
                time_step,
            )?);
        }

        Ok(Self {
            vertex_slots: slots,
            source_valid,
            component_of,
            local_of,
            components,
        })
    }

    /// Number of vertex slots on the mesh this solver was built for.
    #[must_use]
    pub fn vertex_slots(&self) -> usize {
        self.vertex_slots
    }

    /// Computes Crane heat-method geodesic distances from `sources`, writing
    /// `self.vertex_slots()` distances into `out`.
    ///
    /// Vertices in components containing no source, isolated alive vertices not
    /// in `sources`, dead slots, and distances strictly greater than
    /// `max_distance` receive `f32::INFINITY`.
    ///
    /// # Errors
    /// - [`FieldsError::InvalidInput`] if `sources` is empty, any source is out
    ///   of range / dead / non-finite, or `max_distance` is negative or non-finite.
    /// - [`FieldsError::OutOfMemory`] if working or output buffers cannot be reserved.
    ///
    /// On error `out` is left unchanged.
    pub fn solve(
        &self,
        sources: &[u32],
        max_distance: Option<f32>,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        if let Some(d) = max_distance
            && (!d.is_finite() || d < 0.0)
        {
            return Err(FieldsError::InvalidInput(
                "max_distance must be finite and non-negative",
            ));
        }
        if sources.is_empty() {
            return Err(FieldsError::InvalidInput("sources must not be empty"));
        }
        for &s in sources {
            let idx = s as usize;
            if idx >= self.vertex_slots || !self.source_valid[idx] {
                return Err(FieldsError::InvalidInput(
                    "source vertex index is out of range, dead, or non-finite",
                ));
            }
        }

        let mut comp_sources: Vec<Vec<u32>> = Vec::new();
        reserve_exact(&mut comp_sources, self.components.len())?;
        for _ in 0..self.components.len() {
            comp_sources.push(Vec::new());
        }
        for &s in sources {
            let cid = self.component_of[s as usize];
            if cid != u32::MAX {
                push(&mut comp_sources[cid as usize], self.local_of[s as usize])?;
            }
        }

        let mut result = filled(self.vertex_slots, f32::INFINITY)?;
        for &s in sources {
            if self.component_of[s as usize] == u32::MAX {
                result[s as usize] = 0.0;
            }
        }

        for (comp, local_sources) in self.components.iter().zip(comp_sources.iter()) {
            if local_sources.is_empty() {
                continue;
            }
            let phi = solve_component(comp, local_sources)?;
            for (&gv, &val) in comp.global_verts.iter().zip(phi.iter()) {
                let d = val as f32;
                result[gv as usize] = if let Some(limit) = max_distance {
                    if d > limit { f32::INFINITY } else { d }
                } else {
                    d
                };
            }
        }

        if out.capacity() < self.vertex_slots {
            reserve_exact(out, self.vertex_slots - out.len())?;
        }
        out.clear();
        out.extend_from_slice(&result);
        Ok(())
    }
}

fn build_component(
    global_verts: Vec<u32>,
    edges: Vec<(u32, u32, f64)>,
    tris: Vec<LocalTri>,
    vertex_mass: &[f64],
    time_step: f64,
) -> Result<HeatComponent> {
    let n = global_verts.len();
    let total_area: f64 = tris.iter().map(|t| t.area).sum();
    let avg_mass = if n > 0 { total_area / (n as f64) } else { 1.0 };
    let avg_mass = if avg_mass > 0.0 && avg_mass.is_finite() {
        avg_mass
    } else {
        1.0
    };

    let mut row_counts = filled(n + 1, 0usize)?;
    for &(u, v, _) in &edges {
        row_counts[u as usize + 1] += 1;
        row_counts[v as usize + 1] += 1;
    }
    for i in 0..n {
        row_counts[i + 1] += 1; // diagonal entry
        row_counts[i + 1] += row_counts[i];
    }

    let nnz = row_counts[n];
    let mut pairs: Vec<(u32, f64)> = filled(nnz, (0u32, 0.0f64))?;
    let mut fill = filled(n, 0usize)?;
    let mut diag_w = filled(n, 0.0f64)?;

    for &(u, v, w) in &edges {
        let pu = row_counts[u as usize] + fill[u as usize];
        pairs[pu] = (v, -w);
        fill[u as usize] += 1;

        let pv = row_counts[v as usize] + fill[v as usize];
        pairs[pv] = (u, -w);
        fill[v as usize] += 1;

        diag_w[u as usize] += w;
        diag_w[v as usize] += w;
    }
    for i in 0..n {
        let pi = row_counts[i] + fill[i];
        pairs[pi] = (i as u32, diag_w[i]);
        pairs[row_counts[i]..row_counts[i + 1]].sort_unstable_by_key(|e| e.0);
    }
    drop(fill);
    drop(diag_w);

    let mut col_idx: Vec<u32> = Vec::new();
    let mut lap_vals: Vec<f64> = Vec::new();
    reserve_exact(&mut col_idx, nnz)?;
    reserve_exact(&mut lap_vals, nnz)?;
    for &(c, val) in &pairs {
        col_idx.push(c);
        lap_vals.push(val);
    }
    drop(pairs);

    let mut diag_pos = filled(n, 0usize)?;
    for i in 0..n {
        let start = row_counts[i];
        let end = row_counts[i + 1];
        let rel = col_idx[start..end]
            .binary_search(&(i as u32))
            .map_err(|_| FieldsError::InvalidInput("missing Laplacian diagonal"))?;
        diag_pos[i] = start + rel;
    }

    let mut heat_vals = filled(nnz, 0.0f64)?;
    for i in 0..n {
        let start = row_counts[i];
        let end = row_counts[i + 1];
        for p in start..end {
            heat_vals[p] = if p == diag_pos[i] {
                vertex_mass[global_verts[i] as usize] + time_step * lap_vals[p]
            } else {
                time_step * lap_vals[p]
            };
        }
    }

    let mut poisson_vals = lap_vals;
    if n >= 2 {
        poisson_vals[row_counts[0]..row_counts[1]].fill(0.0);
        poisson_vals[diag_pos[0]] = 1.0;
        for i in 1..n {
            if col_idx[row_counts[i]] == 0 {
                poisson_vals[row_counts[i]] = 0.0;
            }
        }
    } else if nnz > 0 {
        poisson_vals[0] = 1.0;
    }

    let heat_precond = build_preconditioner(n, &row_counts, &col_idx, &diag_pos, &heat_vals)?;
    let poisson_precond = build_preconditioner(n, &row_counts, &col_idx, &diag_pos, &poisson_vals)?;

    Ok(HeatComponent {
        global_verts,
        avg_mass,
        row_ptr: row_counts,
        col_idx,
        heat_vals,
        poisson_vals,
        heat_precond,
        poisson_precond,
        tris,
    })
}

fn build_preconditioner(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[u32],
    diag_pos: &[usize],
    vals: &[f64],
) -> Result<Preconditioner> {
    let mut ic_row_ptr = filled(n + 1, 0usize)?;
    for i in 0..n {
        let lower_len = diag_pos[i] - row_ptr[i] + 1;
        ic_row_ptr[i + 1] = ic_row_ptr[i] + lower_len;
    }
    let ic_nnz = ic_row_ptr[n];
    let mut ic_col_idx: Vec<u32> = Vec::new();
    reserve_exact(&mut ic_col_idx, ic_nnz)?;
    for i in 0..n {
        ic_col_idx.extend_from_slice(&col_idx[row_ptr[i]..=diag_pos[i]]);
    }

    let mut ic_vals = filled(ic_nnz, 0.0f64)?;
    let mut inv_diag = filled(n, 0.0f64)?;

    for i in 0..n {
        let i_start = ic_row_ptr[i];
        let i_diag = ic_row_ptr[i + 1] - 1;
        let orig_diag = vals[diag_pos[i]];
        let mut row_sq_sum = 0.0f64;

        for p in i_start..i_diag {
            let j = ic_col_idx[p] as usize;
            let a_ij = vals[row_ptr[i] + (p - i_start)];

            let mut pi = i_start;
            let mut pj = ic_row_ptr[j];
            let pj_end = ic_row_ptr[j + 1] - 1;
            let mut dot_sum = 0.0f64;
            while pi < p && pj < pj_end {
                match ic_col_idx[pi].cmp(&ic_col_idx[pj]) {
                    Ordering::Less => pi += 1,
                    Ordering::Greater => pj += 1,
                    Ordering::Equal => {
                        dot_sum += ic_vals[pi] * ic_vals[pj];
                        pi += 1;
                        pj += 1;
                    }
                }
            }
            let l_ij = (a_ij - dot_sum) * inv_diag[j];
            ic_vals[p] = l_ij;
            row_sq_sum += l_ij * l_ij;
        }

        let d = orig_diag - row_sq_sum;
        if !d.is_finite() || d <= 1e-14 * orig_diag.abs().max(1e-30) {
            let mut fallback = filled(n, 0.0f64)?;
            for r in 0..n {
                let diag_v = vals[diag_pos[r]];
                fallback[r] = if diag_v > 0.0 && diag_v.is_finite() {
                    1.0 / diag_v
                } else {
                    1.0
                };
            }
            return Ok(Preconditioner::Diagonal { inv_diag: fallback });
        }

        let l_ii = d.sqrt();
        ic_vals[i_diag] = l_ii;
        inv_diag[i] = 1.0 / l_ii;
    }

    Ok(Preconditioner::IncompleteCholesky {
        row_ptr: ic_row_ptr,
        col_idx: ic_col_idx,
        values: ic_vals,
        inv_diag,
    })
}

fn apply_preconditioner(precond: &Preconditioner, r: &[f64], z: &mut [f64]) {
    let n = r.len();
    match precond {
        Preconditioner::Diagonal { inv_diag } => {
            for i in 0..n {
                z[i] = r[i] * inv_diag[i];
            }
        }
        Preconditioner::IncompleteCholesky {
            row_ptr,
            col_idx,
            values,
            inv_diag,
        } => {
            for i in 0..n {
                let start = row_ptr[i];
                let end = row_ptr[i + 1] - 1;
                let mut sum = r[i];
                for p in start..end {
                    sum -= values[p] * z[col_idx[p] as usize];
                }
                z[i] = sum * inv_diag[i];
            }
            for i in (0..n).rev() {
                let start = row_ptr[i];
                let end = row_ptr[i + 1] - 1;
                let zi = z[i] * inv_diag[i];
                z[i] = zi;
                for p in start..end {
                    z[col_idx[p] as usize] -= values[p] * zi;
                }
            }
        }
    }
}

fn solve_component(comp: &HeatComponent, local_sources: &[u32]) -> Result<Vec<f64>> {
    let n = comp.global_verts.len();
    if n <= 1 {
        return filled(n, 0.0f64);
    }

    let mut rhs = filled(n, 0.0f64)?;
    for &ls in local_sources {
        rhs[ls as usize] = comp.avg_mass;
    }

    let mut u = filled(n, 0.0f64)?;
    let mut r = filled(n, 0.0f64)?;
    let mut z = filled(n, 0.0f64)?;
    let mut p = filled(n, 0.0f64)?;
    let mut ap = filled(n, 0.0f64)?;

    solve_pcg(
        &comp.row_ptr,
        &comp.col_idx,
        &comp.heat_vals,
        &comp.heat_precond,
        &rhs,
        &mut u,
        &mut r,
        &mut z,
        &mut p,
        &mut ap,
    )?;

    // Normalized negative heat gradient and integrated divergence RHS for Poisson.
    rhs.fill(0.0);
    for tri in &comp.tris {
        let [v0, v1, v2] = tri.verts.map(|v| v as usize);
        let g = add(
            add(
                scale(tri.grad_vecs[0], u[v0]),
                scale(tri.grad_vecs[1], u[v1]),
            ),
            scale(tri.grad_vecs[2], u[v2]),
        );
        let gn = norm(g);
        if gn.is_finite() && gn > 0.0 {
            let x_t = scale(g, -1.0 / gn);
            for k in 0..3 {
                rhs[tri.verts[k] as usize] += tri.area * dot(tri.grad_vecs[k], x_t);
            }
        }
    }

    let mean = sum_par(&rhs)? / (n as f64);
    for val in &mut rhs {
        *val -= mean;
    }
    rhs[0] = 0.0;

    let mut phi = u;
    phi.fill(0.0);
    solve_pcg(
        &comp.row_ptr,
        &comp.col_idx,
        &comp.poisson_vals,
        &comp.poisson_precond,
        &rhs,
        &mut phi,
        &mut r,
        &mut z,
        &mut p,
        &mut ap,
    )?;

    let mut min_src = f64::INFINITY;
    for &ls in local_sources {
        let v = phi[ls as usize];
        if v < min_src {
            min_src = v;
        }
    }
    if min_src.is_finite() {
        for v in &mut phi {
            *v = (*v - min_src).max(0.0);
        }
    }
    for &ls in local_sources {
        phi[ls as usize] = 0.0;
    }

    Ok(phi)
}

#[allow(clippy::too_many_arguments)]
fn solve_pcg(
    row_ptr: &[usize],
    col_idx: &[u32],
    vals: &[f64],
    precond: &Preconditioner,
    b: &[f64],
    x: &mut [f64],
    r: &mut [f64],
    z: &mut [f64],
    p: &mut [f64],
    ap: &mut [f64],
) -> Result<()> {
    let n = b.len();
    x.fill(0.0);
    r.copy_from_slice(b);

    let b_norm2 = dot_par(b, b)?;
    if !b_norm2.is_finite() || b_norm2 == 0.0 {
        return Ok(());
    }
    let tol2 = 1e-20 * b_norm2;

    apply_preconditioner(precond, r, z);
    p.copy_from_slice(z);
    let mut rz = dot_par(r, z)?;
    let max_iters = n.saturating_mul(2).max(200);

    for _ in 0..max_iters {
        if !rz.is_finite() || rz <= 1e-300 * b_norm2 {
            break;
        }
        spmv_par(row_ptr, col_idx, vals, p, ap);
        let p_ap = dot_par(p, ap)?;
        if !p_ap.is_finite() || p_ap <= 1e-300 * rz {
            break;
        }
        let alpha = rz / p_ap;
        axpy_par(alpha, p, x);
        axpy_par(-alpha, ap, r);

        let r_norm2 = dot_par(r, r)?;
        if !r_norm2.is_finite() || r_norm2 <= tol2 {
            break;
        }

        apply_preconditioner(precond, r, z);
        let new_rz = dot_par(r, z)?;
        if !new_rz.is_finite() || new_rz <= 0.0 {
            break;
        }
        let beta = new_rz / rz;
        xpby_par(z, beta, p);
        rz = new_rz;
    }
    Ok(())
}

fn spmv_par(row_ptr: &[usize], col_idx: &[u32], vals: &[f64], x: &[f64], y: &mut [f64]) {
    par::for_each_chunk_mut(y, PAR_CHUNK, PAR_MIN_ROWS, |chunk_idx, y_chunk| {
        let base = chunk_idx * PAR_CHUNK;
        for (k, out) in y_chunk.iter_mut().enumerate() {
            let row = base + k;
            let mut sum = 0.0f64;
            for idx in row_ptr[row]..row_ptr[row + 1] {
                sum += vals[idx] * x[col_idx[idx] as usize];
            }
            *out = sum;
        }
    });
}

fn axpy_par(alpha: f64, x: &[f64], y: &mut [f64]) {
    par::for_each_chunk_mut(y, PAR_CHUNK, PAR_MIN_ROWS, |chunk_idx, y_chunk| {
        let base = chunk_idx * PAR_CHUNK;
        for (k, yi) in y_chunk.iter_mut().enumerate() {
            *yi += alpha * x[base + k];
        }
    });
}

fn xpby_par(z: &[f64], beta: f64, p: &mut [f64]) {
    par::for_each_chunk_mut(p, PAR_CHUNK, PAR_MIN_ROWS, |chunk_idx, p_chunk| {
        let base = chunk_idx * PAR_CHUNK;
        for (k, pi) in p_chunk.iter_mut().enumerate() {
            *pi = z[base + k] + beta * (*pi);
        }
    });
}

fn dot_par(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() >= PAR_MIN_ROWS && par::enabled() && !par::force_sequential() {
        let partials = par::map_range(
            a.len(),
            PAR_CHUNK,
            PAR_MIN_ROWS,
            || FieldsError::OutOfMemory,
            |r| {
                let mut acc = 0.0f64;
                for i in r {
                    acc += a[i] * b[i];
                }
                Ok::<f64, FieldsError>(acc)
            },
        )?;
        let mut total = 0.0f64;
        for s in partials {
            total += s;
        }
        Ok(total)
    } else {
        let mut total = 0.0f64;
        for (ca, cb) in a.chunks(PAR_CHUNK).zip(b.chunks(PAR_CHUNK)) {
            let mut acc = 0.0f64;
            for (&x, &y) in ca.iter().zip(cb) {
                acc += x * y;
            }
            total += acc;
        }
        Ok(total)
    }
}

fn sum_par(a: &[f64]) -> Result<f64> {
    if a.len() >= PAR_MIN_ROWS && par::enabled() && !par::force_sequential() {
        let partials = par::map_range(
            a.len(),
            PAR_CHUNK,
            PAR_MIN_ROWS,
            || FieldsError::OutOfMemory,
            |r| {
                let mut acc = 0.0f64;
                for i in r {
                    acc += a[i];
                }
                Ok::<f64, FieldsError>(acc)
            },
        )?;
        let mut total = 0.0f64;
        for s in partials {
            total += s;
        }
        Ok(total)
    } else {
        let mut total = 0.0f64;
        for c in a.chunks(PAR_CHUNK) {
            let mut acc = 0.0f64;
            for &x in c {
                acc += x;
            }
            total += acc;
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::fail_allocations_after;
    use crate::test_util::*;

    #[test]
    fn dijkstra_cube_exact_edge_distances() {
        let m = cube();
        let mut out = Vec::new();
        let opts = GeodesicOptions {
            method: GeodesicMethod::Dijkstra,
            ..Default::default()
        };
        geodesic_distances(&m, &[0], &opts, &mut out).unwrap();
        assert_eq!(out.len(), 8);
        assert_eq!(out[0], 0.0);
        // Cube vertices in test_util::cube() lie on [-1, 1]^3 (edge = 2, face diag = 2*sqrt(2)).
        assert!((out[1] - 2.0).abs() < 1e-5, "edge neighbor got {}", out[1]);
        assert!(out[6] > 2.0 && out.iter().all(|&d| d.is_finite()));
    }

    #[test]
    fn heat_method_single_and_multi_source_on_icosphere() {
        let m = icosphere();
        let solver = GeodesicHeatSolver::build(&m, 1.0).unwrap();
        let mut d_single = Vec::new();
        solver.solve(&[0], None, &mut d_single).unwrap();

        let opts = GeodesicOptions::default();
        let mut d_oneshot = Vec::new();
        geodesic_distances(&m, &[0], &opts, &mut d_oneshot).unwrap();
        assert_eq!(d_single, d_oneshot);
        assert_eq!(d_single[0], 0.0);
        for (i, &d) in d_single.iter().enumerate() {
            assert!(d.is_finite() && d >= 0.0, "vertex {i} invalid d={d}");
        }

        // Farther geodesic vertex has larger angle on the unit sphere.
        let p0 = widen(m.positions()[0]);
        let far = d_single
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let expected_pi = dot(p0, widen(m.positions()[far])).clamp(-1.0, 1.0).acos();
        assert!(
            (f64::from(d_single[far]) - expected_pi).abs() / expected_pi < 0.05,
            "got {} expected {expected_pi}",
            d_single[far]
        );

        // Multi-source reuse.
        let opp = m
            .positions()
            .iter()
            .enumerate()
            .min_by(|a, b| dot(p0, widen(*a.1)).total_cmp(&dot(p0, widen(*b.1))))
            .unwrap()
            .0 as u32;
        let mut d_multi = Vec::new();
        solver.solve(&[0, opp], None, &mut d_multi).unwrap();
        assert_eq!(d_multi[0], 0.0);
        assert_eq!(d_multi[opp as usize], 0.0);
        assert!(d_multi[far] < d_single[far]);
    }

    #[test]
    fn disconnected_component_and_max_distance_cutoff() {
        let mut verts = cube_buffers().0;
        let mut tris = cube_buffers().1;
        let base = verts.len() as u32;
        for v in cube_buffers().0 {
            verts.push([v[0] + 10.0, v[1], v[2]]);
        }
        for t in cube_buffers().1 {
            tris.push([t[0] + base, t[1] + base, t[2] + base]);
        }
        let m = mesh(&verts, &tris);

        for method in [GeodesicMethod::Dijkstra, GeodesicMethod::HeatMethod] {
            let opts = GeodesicOptions {
                method,
                ..Default::default()
            };
            let mut out = Vec::new();
            geodesic_distances(&m, &[0], &opts, &mut out).unwrap();
            for (v, &d) in out.iter().enumerate().take(8) {
                assert!(d.is_finite(), "reached component missing {v}");
            }
            for &d in out.iter().take(16).skip(8) {
                assert!(d.is_infinite(), "unreached component got {d}");
            }

            let capped = GeodesicOptions {
                method,
                max_distance: Some(1.5),
                time_step_scale: 1.0,
            };
            geodesic_distances(&m, &[0], &capped, &mut out).unwrap();
            assert_eq!(out[0], 0.0);
            assert!(out[1].is_infinite());
        }
    }

    #[test]
    fn invalid_inputs_preserve_output() {
        let m = tetrahedron();
        let sentinel = vec![42.0f32, 99.0];
        let mut out = sentinel.clone();

        assert!(matches!(
            geodesic_distances(&m, &[], &GeodesicOptions::default(), &mut out),
            Err(FieldsError::InvalidInput(_))
        ));
        assert_eq!(out, sentinel);

        assert!(matches!(
            geodesic_distances(&m, &[999], &GeodesicOptions::default(), &mut out),
            Err(FieldsError::InvalidInput(_))
        ));
        assert_eq!(out, sentinel);

        let bad_scale = GeodesicOptions {
            time_step_scale: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            geodesic_distances(&m, &[0], &bad_scale, &mut out),
            Err(FieldsError::InvalidInput(_))
        ));
        assert_eq!(out, sentinel);

        let bad_cap = GeodesicOptions {
            max_distance: Some(-1.0),
            ..Default::default()
        };
        assert!(matches!(
            geodesic_distances(&m, &[0], &bad_cap, &mut out),
            Err(FieldsError::InvalidInput(_))
        ));
        assert_eq!(out, sentinel);
    }

    #[test]
    fn oom_fault_injection_is_clean() {
        let m = icosphere();
        for method in [GeodesicMethod::Dijkstra, GeodesicMethod::HeatMethod] {
            let opts = GeodesicOptions {
                method,
                ..Default::default()
            };
            for fail_at in 0..25 {
                let mut out = vec![7.5f32];
                fail_allocations_after(Some(fail_at));
                let res = geodesic_distances(&m, &[0], &opts, &mut out);
                fail_allocations_after(None);
                match res {
                    Ok(()) => break,
                    Err(FieldsError::OutOfMemory) => assert_eq!(out, vec![7.5f32]),
                    Err(other) => panic!("unexpected error: {other:?}"),
                }
            }
        }
    }
}
