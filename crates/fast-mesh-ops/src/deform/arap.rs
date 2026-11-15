//! As-Rigid-As-Possible (ARAP) surface deformation (Sorkine & Alexa 2007).
//!
//! Preserves local surface detail under large bends and twists via alternating
//! per-vertex 3x3 covariance polar rotations (`SO(3)`) and sparse SPD global Poisson
//! solves using `crate::sparse`.

use fast_mesh_store::{Mesh, StoreError, Transaction, VertexId, par};

use crate::remesh::Region;
use crate::sparse::{SymCsr, factor_spd_ordered, nested_dissection, solve_pcg};
use crate::topology::{UnionFind, proper_triangle};
use crate::{OpsError, Result, allocation_allowed, filled, push, reserve};

const COT_CLAMP_FRAC: f64 = 1e-6;
const PAR_CHUNK: usize = 256;
const PAR_THRESHOLD: usize = 2048;
const PIN_CONFLICT_TOL_SQ: f64 = 1e-10;
const SVD_SWEEPS: usize = 15;

/// Fixed or repositioned anchor constraint for [`apply_arap`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArapAnchor {
    /// Constrained vertex identifier.
    pub vertex: VertexId,
    /// Optional target position (`None` pins the vertex at its current mesh position).
    pub position: Option<[f32; 3]>,
}

impl ArapAnchor {
    /// Pins `vertex` cleanly at its current position.
    #[inline]
    pub fn pin(vertex: VertexId) -> Self {
        Self {
            vertex,
            position: None,
        }
    }

    /// Pins `vertex` at an explicit target world position.
    #[inline]
    pub fn at(vertex: VertexId, position: [f32; 3]) -> Self {
        Self {
            vertex,
            position: Some(position),
        }
    }
}

/// Displaced handle constraint driving ARAP deformation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArapHandle {
    /// Handle vertex identifier.
    pub vertex: VertexId,
    /// Target world position for `vertex`.
    pub target: [f32; 3],
}

impl ArapHandle {
    /// Creates a handle moving `vertex` to `target`.
    #[inline]
    pub fn new(vertex: VertexId, target: [f32; 3]) -> Self {
        Self { vertex, target }
    }

    /// Creates a handle by adding `delta` to `vertex`'s current mesh position.
    ///
    /// # Errors
    /// Returns [`OpsError::InvalidInput`] if `vertex` is not alive or `delta` is non-finite.
    pub fn from_displacement(mesh: &Mesh, vertex: VertexId, delta: [f32; 3]) -> Result<Self> {
        if !mesh.is_vertex_alive(vertex) {
            return Err(OpsError::InvalidInput("ARAP handle vertex is not alive"));
        }
        if !finite3(delta) {
            return Err(OpsError::InvalidInput("ARAP displacement must be finite"));
        }
        let p = mesh.positions()[vertex as usize];
        if !finite3(p) {
            return Err(OpsError::InvalidInput(
                "mesh contains non-finite vertex position",
            ));
        }
        Ok(Self {
            vertex,
            target: [p[0] + delta[0], p[1] + delta[1], p[2] + delta[2]],
        })
    }

    /// Creates a handle by rigidly rotating `vertex` around `pivot` and `axis` by `angle_rad`
    /// and then adding `translation`.
    ///
    /// # Errors
    /// Returns [`OpsError::InvalidInput`] if inputs are non-finite, `vertex` is not alive, or `axis` has zero length.
    pub fn from_rigid(
        mesh: &Mesh,
        vertex: VertexId,
        pivot: [f32; 3],
        axis: [f32; 3],
        angle_rad: f32,
        translation: [f32; 3],
    ) -> Result<Self> {
        if !mesh.is_vertex_alive(vertex) {
            return Err(OpsError::InvalidInput("ARAP handle vertex is not alive"));
        }
        if !finite3(pivot) || !finite3(axis) || !angle_rad.is_finite() || !finite3(translation) {
            return Err(OpsError::InvalidInput(
                "ARAP rigid handle transform parameters must be finite",
            ));
        }
        let ax = [axis[0] as f64, axis[1] as f64, axis[2] as f64];
        let norm = (ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]).sqrt();
        if norm <= 1e-12 {
            return Err(OpsError::InvalidInput(
                "ARAP rigid rotation axis must be nonzero",
            ));
        }
        let k = [ax[0] / norm, ax[1] / norm, ax[2] / norm];
        let p = mesh.positions()[vertex as usize];
        if !finite3(p) {
            return Err(OpsError::InvalidInput(
                "mesh contains non-finite vertex position",
            ));
        }
        let v = [
            p[0] as f64 - pivot[0] as f64,
            p[1] as f64 - pivot[1] as f64,
            p[2] as f64 - pivot[2] as f64,
        ];
        let (s, c) = (angle_rad as f64).sin_cos();
        let k_cross_v = cross3(k, v);
        let k_dot_v = dot3(k, v);
        let mut rot = [0.0f64; 3];
        for i in 0..3 {
            rot[i] = v[i] * c + k_cross_v[i] * s + k[i] * k_dot_v * (1.0 - c);
        }
        Ok(Self {
            vertex,
            target: [
                (pivot[0] as f64 + rot[0] + translation[0] as f64) as f32,
                (pivot[1] as f64 + rot[1] + translation[1] as f64) as f32,
                (pivot[2] as f64 + rot[2] + translation[2] as f64) as f32,
            ],
        })
    }

    /// Convenience alias for [`ArapHandle::from_rigid`] with zero translation.
    pub fn from_axis_angle(
        mesh: &Mesh,
        vertex: VertexId,
        pivot: [f32; 3],
        axis: [f32; 3],
        angle_rad: f32,
    ) -> Result<Self> {
        Self::from_rigid(mesh, vertex, pivot, axis, angle_rad, [0.0, 0.0, 0.0])
    }
}

/// Edge weighting scheme for ARAP local-global energies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ArapWeights {
    /// Positive-clamped cotangent Laplacian weights (`0.5 * (cot alpha + cot beta)`).
    #[default]
    Cotangent,
    /// Unit edge weights (`w_ij = 1.0`).
    Uniform,
}

/// Linear solver backend for the ARAP global step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ArapSolver {
    /// Nested-dissection direct sparse LDLT Cholesky with automatic PCG fallback.
    #[default]
    Auto,
    /// Direct sparse LDLT Cholesky factorization (falls back to PCG only if non-SPD).
    Direct,
    /// Warm-started Jacobi-preconditioned Conjugate Gradient solver.
    Pcg,
}

/// Configuration for ARAP surface deformation via [`apply_arap`].
#[derive(Clone, Debug, PartialEq)]
pub struct ArapOptions {
    /// Stationary or explicitly repositioned pinned anchor vertices.
    pub anchors: Vec<ArapAnchor>,
    /// Driven target handle vertices.
    pub handles: Vec<ArapHandle>,
    /// Optional spherical active region; vertices outside remain fixed and anchor boundaries.
    pub region: Option<Region>,
    /// Optional explicit free/active vertex subset; unselected neighbors remain fixed boundaries.
    pub vertices: Option<Vec<VertexId>>,
    /// Edge weighting mode.
    pub weights: ArapWeights,
    /// Global SPD linear solver mode.
    pub solver: ArapSolver,
    /// Maximum local-global iterations (`>= 1`).
    pub iterations: usize,
    /// Convergence tolerance on normalized RMS vertex update (`>= 0`).
    pub tolerance: f32,
}

impl Default for ArapOptions {
    fn default() -> Self {
        Self {
            anchors: Vec::new(),
            handles: Vec::new(),
            region: None,
            vertices: None,
            weights: ArapWeights::Cotangent,
            solver: ArapSolver::Auto,
            iterations: 10,
            tolerance: 1e-4,
        }
    }
}

impl ArapOptions {
    /// Creates default ARAP options (10 iterations, cotangent weights, tolerance `1e-4`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets stationary/repositioned anchor constraints.
    pub fn with_anchors(mut self, anchors: Vec<ArapAnchor>) -> Self {
        self.anchors = anchors;
        self
    }

    /// Sets driven target handle constraints.
    pub fn with_handles(mut self, handles: Vec<ArapHandle>) -> Self {
        self.handles = handles;
        self
    }

    /// Restricts active vertices to a spherical [`Region`].
    pub fn with_region(mut self, region: Region) -> Self {
        self.region = Some(region);
        self
    }

    /// Restricts active vertices to an explicit vertex subset.
    pub fn with_vertices(mut self, vertices: Vec<VertexId>) -> Self {
        self.vertices = Some(vertices);
        self
    }

    /// Sets the ARAP edge weight formulation.
    pub fn with_weights(mut self, weights: ArapWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Sets the global linear solver strategy.
    pub fn with_solver(mut self, solver: ArapSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Sets the maximum number of local-global iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the convergence residual tolerance.
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }
}

/// Summary report returned by [`apply_arap`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArapReport {
    /// Number of alive vertices participating in the active region/mesh.
    pub vertices_evaluated: usize,
    /// Number of vertices whose world position changed.
    pub vertices_moved: usize,
    /// Number of unconstrained free vertices solved by the linear system.
    pub free_vertices: usize,
    /// Number of constrained (anchor, handle, or boundary-fixed) vertices.
    pub pinned_vertices: usize,
    /// Number of local-global iterations executed.
    pub iterations_run: usize,
    /// Normalized RMS step displacement on the final iteration.
    pub final_residual: f32,
    /// Whether `final_residual <= tolerance` was reached.
    pub converged: bool,
    /// Maximum Euclidean displacement across all moved vertices.
    pub max_displacement: f32,
}

/// Applies As-Rigid-As-Possible surface deformation to `tx`.
///
/// Validates all anchors, handles, weights, and connected-component constraints before
/// modifying `tx`. Preserves `UV_CHANNEL` and leaves `tx` untouched on error or OOM.
///
/// # Errors
/// Returns [`OpsError::InvalidInput`] on empty triangle meshes, invalid options,
/// conflicting/dead constraints, or unpinned connected components/isolated vertices;
/// returns [`OpsError::Store`] if memory allocation fails.
pub fn apply_arap(tx: &mut Transaction, options: &ArapOptions) -> Result<ArapReport> {
    if options.iterations == 0 {
        return Err(OpsError::InvalidInput("ARAP iterations must be >= 1"));
    }
    if !options.tolerance.is_finite() || options.tolerance < 0.0 {
        return Err(OpsError::InvalidInput(
            "ARAP tolerance must be finite and >= 0",
        ));
    }
    if let Some(r) = options.region
        && (!finite3(r.center) || !r.radius.is_finite() || r.radius <= 0.0)
    {
        return Err(OpsError::InvalidInput(
            "ARAP region center must be finite and radius > 0",
        ));
    }
    if options.anchors.is_empty() && options.handles.is_empty() {
        return Err(OpsError::InvalidInput(
            "ARAP requires at least one anchor or handle constraint",
        ));
    }

    let mesh = tx.mesh();
    if mesh.triangle_count() == 0 {
        return Err(OpsError::InvalidInput(
            "ARAP requires at least one triangle",
        ));
    }
    let slots = mesh.vertex_slots();
    if slots == 0 {
        return Err(OpsError::InvalidInput("ARAP mesh has no vertices"));
    }
    let positions = mesh.positions();

    // Validate all alive vertex coordinates once upfront.
    let mut alive = filled(slots, false)?;
    let mut rest_pos = filled(slots, [0.0f64; 3])?;
    let mut bbox_min = [f64::INFINITY; 3];
    let mut bbox_max = [f64::NEG_INFINITY; 3];
    let mut alive_count = 0usize;
    for slot in 0..slots {
        let vid = slot as VertexId;
        if !mesh.is_vertex_alive(vid) {
            continue;
        }
        let p = positions[slot];
        if !finite3(p) {
            return Err(OpsError::InvalidInput(
                "mesh contains non-finite vertex position",
            ));
        }
        alive[slot] = true;
        alive_count += 1;
        let pd = [p[0] as f64, p[1] as f64, p[2] as f64];
        rest_pos[slot] = pd;
        for a in 0..3 {
            if pd[a] < bbox_min[a] {
                bbox_min[a] = pd[a];
            }
            if pd[a] > bbox_max[a] {
                bbox_max[a] = pd[a];
            }
        }
    }
    if alive_count == 0 {
        return Err(OpsError::InvalidInput("ARAP requires alive vertices"));
    }
    let dx = bbox_max[0] - bbox_min[0];
    let dy = bbox_max[1] - bbox_min[1];
    let dz = bbox_max[2] - bbox_min[2];
    let bbox_diag = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);

    // Determine active subset before constraint pinning.
    let mut in_subset = filled(slots, false)?;
    if let Some(verts) = options.vertices.as_deref() {
        for &v in verts {
            if !mesh.is_vertex_alive(v) {
                return Err(OpsError::InvalidInput(
                    "ARAP vertex list contains dead or invalid VertexId",
                ));
            }
            let s = v as usize;
            if in_region(rest_pos[s], options.region) {
                in_subset[s] = true;
            }
        }
    } else {
        for slot in 0..slots {
            if alive[slot] && in_region(rest_pos[slot], options.region) {
                in_subset[slot] = true;
            }
        }
    }

    // Track explicit pin targets and detect conflicts.
    let mut is_pinned = filled(slots, false)?;
    let mut pin_target = filled(slots, [0.0f64; 3])?;

    for anchor in &options.anchors {
        if !mesh.is_vertex_alive(anchor.vertex) {
            return Err(OpsError::InvalidInput(
                "ARAP anchor references non-alive vertex",
            ));
        }
        let s = anchor.vertex as usize;
        let tgt = match anchor.position {
            Some(pos) => {
                if !finite3(pos) {
                    return Err(OpsError::InvalidInput(
                        "ARAP anchor position must be finite",
                    ));
                }
                [pos[0] as f64, pos[1] as f64, pos[2] as f64]
            }
            None => rest_pos[s],
        };
        record_pin(&mut is_pinned, &mut pin_target, s, tgt)?;
    }

    for handle in &options.handles {
        if !mesh.is_vertex_alive(handle.vertex) {
            return Err(OpsError::InvalidInput(
                "ARAP handle references non-alive vertex",
            ));
        }
        if !finite3(handle.target) {
            return Err(OpsError::InvalidInput("ARAP handle target must be finite"));
        }
        let s = handle.vertex as usize;
        let tgt = [
            handle.target[0] as f64,
            handle.target[1] as f64,
            handle.target[2] as f64,
        ];
        record_pin(&mut is_pinned, &mut pin_target, s, tgt)?;
    }

    // Any alive vertex outside the active subset stays implicitly pinned at rest.
    for slot in 0..slots {
        if alive[slot] && !in_subset[slot] && !is_pinned[slot] {
            is_pinned[slot] = true;
            pin_target[slot] = rest_pos[slot];
        }
    }

    // Assemble unique undirected weighted edges from proper triangles.
    let (edges, adj_starts, adj_nbrs, adj_weights) =
        build_edge_graph(mesh, slots, &rest_pos, options.weights)?;

    // Verify every active connected component (including isolated degree-0 vertices) has >= 1 pin.
    let mut uf = UnionFind::new();
    uf.reset(slots)?;
    for &(u, v, _) in &edges {
        uf.union(u, v);
    }
    let mut comp_has_pin = filled(slots, false)?;
    for slot in 0..slots {
        if alive[slot] && is_pinned[slot] {
            let root = uf.find(slot as u32) as usize;
            comp_has_pin[root] = true;
        }
    }
    for slot in 0..slots {
        if alive[slot] && in_subset[slot] {
            let root = uf.find(slot as u32) as usize;
            if !comp_has_pin[root] {
                return Err(OpsError::InvalidInput(
                    "ARAP connected component or isolated vertex has no pinned constraint",
                ));
            }
        }
    }

    // Build free DOF mapping for active non-pinned vertices.
    let mut free_of_slot = filled(slots, u32::MAX)?;
    let mut slot_of_free: Vec<u32> = Vec::new();
    let mut free_rest: Vec<[f64; 3]> = Vec::new();
    let mut pinned_count = 0usize;
    let mut active_count = 0usize;
    for slot in 0..slots {
        if !alive[slot] {
            continue;
        }
        if in_subset[slot] || is_pinned[slot] {
            active_count += 1;
        }
        if is_pinned[slot] {
            if in_subset[slot] {
                pinned_count += 1;
            }
        } else if in_subset[slot] {
            let fidx = slot_of_free.len() as u32;
            free_of_slot[slot] = fidx;
            push(&mut slot_of_free, slot as u32)?;
            push(&mut free_rest, rest_pos[slot])?;
        }
    }
    let n_free = slot_of_free.len();

    let mut curr_pos = rest_pos.clone();
    for slot in 0..slots {
        if alive[slot] && is_pinned[slot] {
            curr_pos[slot] = pin_target[slot];
        }
    }

    if n_free == 0 {
        return finalize_arap(
            tx,
            &alive,
            &rest_pos,
            &curr_pos,
            active_count,
            0,
            pinned_count,
            1,
            0.0,
            true,
        );
    }

    // Assemble symmetric Laplacian L_free for direct factorization and/or SymCsr PCG.
    let mut csr_starts: Vec<u32> = Vec::new();
    let mut csr_cols: Vec<u32> = Vec::new();
    let mut csr_vals: Vec<f64> = Vec::new();
    reserve(&mut csr_starts, n_free + 1)?;
    push(&mut csr_starts, 0)?;

    let mut factor_entries: Vec<(u32, u32, f64)> = Vec::new();
    let mut nd_edges: Vec<(u32, u32)> = Vec::new();
    let mut pin_rhs = filled(n_free, [0.0f64; 3])?;

    for f_i in 0..n_free {
        let s_i = slot_of_free[f_i] as usize;
        let begin = adj_starts[s_i] as usize;
        let end = adj_starts[s_i + 1] as usize;
        let mut diag_w = 0.0f64;
        let mut row_terms: Vec<(u32, f64)> = Vec::new();
        reserve(&mut row_terms, (end - begin) + 1)?;

        for e_idx in begin..end {
            let s_j = adj_nbrs[e_idx] as usize;
            let w = adj_weights[e_idx];
            diag_w += w;
            let f_j = free_of_slot[s_j];
            if f_j != u32::MAX {
                push(&mut row_terms, (f_j, -w))?;
                if (f_i as u32) < f_j {
                    push(&mut factor_entries, (f_i as u32, f_j, -w))?;
                    push(&mut nd_edges, (f_i as u32, f_j))?;
                }
            } else {
                let p_j = pin_target[s_j];
                pin_rhs[f_i][0] += w * p_j[0];
                pin_rhs[f_i][1] += w * p_j[1];
                pin_rhs[f_i][2] += w * p_j[2];
            }
        }
        if !diag_w.is_finite() || diag_w <= 0.0 {
            return Err(OpsError::InvalidInput(
                "ARAP free vertex has zero total edge weight",
            ));
        }
        push(&mut factor_entries, (f_i as u32, f_i as u32, diag_w))?;
        push(&mut row_terms, (f_i as u32, diag_w))?;
        row_terms.sort_unstable_by_key(|t| t.0);
        for (col, val) in row_terms {
            push(&mut csr_cols, col)?;
            push(&mut csr_vals, val)?;
        }
        push(&mut csr_starts, csr_cols.len() as u32)?;
    }

    let csr = SymCsr {
        n: n_free,
        row_starts: &csr_starts,
        cols: &csr_cols,
        vals: &csr_vals,
    };

    let factor = match options.solver {
        ArapSolver::Pcg => None,
        ArapSolver::Auto | ArapSolver::Direct => {
            let order = nested_dissection(n_free, &free_rest, &nd_edges)?;
            factor_spd_ordered(n_free, &factor_entries, &order)?
        }
    };

    let mut x_axis = [
        filled(n_free, 0.0f64)?,
        filled(n_free, 0.0f64)?,
        filled(n_free, 0.0f64)?,
    ];
    for (f_i, &s_i) in slot_of_free.iter().enumerate() {
        let p = curr_pos[s_i as usize];
        x_axis[0][f_i] = p[0];
        x_axis[1][f_i] = p[1];
        x_axis[2][f_i] = p[2];
    }
    let mut b_axis = [
        filled(n_free, 0.0f64)?,
        filled(n_free, 0.0f64)?,
        filled(n_free, 0.0f64)?,
    ];

    let mut rotations = filled(slots, IDENTITY_3)?;

    // Harmonic initialization (R_i = I) provides a smooth starting pose under large handle motion.
    assemble_arap_rhs(
        &slot_of_free,
        &adj_starts,
        &adj_nbrs,
        &adj_weights,
        &rest_pos,
        &rotations,
        &pin_rhs,
        &mut b_axis,
    )?;
    solve_global_axes(factor.as_ref(), &csr, &b_axis, &mut x_axis)?;
    for (f_i, &s_i) in slot_of_free.iter().enumerate() {
        curr_pos[s_i as usize] = [x_axis[0][f_i], x_axis[1][f_i], x_axis[2][f_i]];
    }

    let mut iterations_run = 0usize;
    let mut final_residual = 0.0f32;
    let mut converged = false;

    for iter in 1..=options.iterations {
        iterations_run = iter;

        // Local step: fit nearest SO(3) rotation per vertex via 3x3 Jacobi SVD.
        compute_local_rotations(
            &alive,
            &adj_starts,
            &adj_nbrs,
            &adj_weights,
            &rest_pos,
            &curr_pos,
            &mut rotations,
        )?;

        // Global step: assemble symmetric ARAP RHS and solve sparse SPD system.
        assemble_arap_rhs(
            &slot_of_free,
            &adj_starts,
            &adj_nbrs,
            &adj_weights,
            &rest_pos,
            &rotations,
            &pin_rhs,
            &mut b_axis,
        )?;
        solve_global_axes(factor.as_ref(), &csr, &b_axis, &mut x_axis)?;

        let mut step_sq = 0.0f64;
        for (f_i, &s_i) in slot_of_free.iter().enumerate() {
            let next_p = [x_axis[0][f_i], x_axis[1][f_i], x_axis[2][f_i]];
            let prev_p = curr_pos[s_i as usize];
            let dx = next_p[0] - prev_p[0];
            let dy = next_p[1] - prev_p[1];
            let dz = next_p[2] - prev_p[2];
            step_sq += dx * dx + dy * dy + dz * dz;
            curr_pos[s_i as usize] = next_p;
        }
        let rms = (step_sq / (n_free as f64)).sqrt() / bbox_diag;
        final_residual = rms as f32;
        if rms <= options.tolerance as f64 {
            converged = true;
            break;
        }
    }

    finalize_arap(
        tx,
        &alive,
        &rest_pos,
        &curr_pos,
        active_count,
        n_free,
        pinned_count,
        iterations_run,
        final_residual,
        converged,
    )
}

#[allow(clippy::too_many_arguments)]
fn finalize_arap(
    tx: &mut Transaction,
    alive: &[bool],
    rest_pos: &[[f64; 3]],
    curr_pos: &[[f64; 3]],
    vertices_evaluated: usize,
    free_vertices: usize,
    pinned_vertices: usize,
    iterations_run: usize,
    final_residual: f32,
    converged: bool,
) -> Result<ArapReport> {
    let mut ids: Vec<VertexId> = Vec::new();
    let mut moved: Vec<[f32; 3]> = Vec::new();
    let mut max_disp = 0.0f64;
    for (slot, &is_alive) in alive.iter().enumerate() {
        if !is_alive {
            continue;
        }
        let r = rest_pos[slot];
        let c = curr_pos[slot];
        if !c[0].is_finite() || !c[1].is_finite() || !c[2].is_finite() {
            return Err(OpsError::InvalidInput(
                "ARAP solver produced non-finite position",
            ));
        }
        let old32 = [r[0] as f32, r[1] as f32, r[2] as f32];
        let new32 = [c[0] as f32, c[1] as f32, c[2] as f32];
        if new32 != old32 {
            let dx = c[0] - r[0];
            let dy = c[1] - r[1];
            let dz = c[2] - r[2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d > max_disp {
                max_disp = d;
            }
            push(&mut ids, slot as VertexId)?;
            push(&mut moved, new32)?;
        }
    }

    if !ids.is_empty() {
        allocation_allowed()?;
        tx.set_positions(&ids, &moved)
            .map_err(|_| OpsError::Store(StoreError::OutOfMemory))?;
    }

    Ok(ArapReport {
        vertices_evaluated,
        vertices_moved: ids.len(),
        free_vertices,
        pinned_vertices,
        iterations_run,
        final_residual,
        converged,
        max_displacement: max_disp as f32,
    })
}

fn record_pin(
    is_pinned: &mut [bool],
    pin_target: &mut [[f64; 3]],
    slot: usize,
    target: [f64; 3],
) -> Result<()> {
    if is_pinned[slot] {
        let prev = pin_target[slot];
        let dx = prev[0] - target[0];
        let dy = prev[1] - target[1];
        let dz = prev[2] - target[2];
        if dx * dx + dy * dy + dz * dz > PIN_CONFLICT_TOL_SQ {
            return Err(OpsError::InvalidInput(
                "conflicting ARAP anchor/handle targets on same vertex",
            ));
        }
    }
    is_pinned[slot] = true;
    pin_target[slot] = target;
    Ok(())
}

type EdgeGraph = (Vec<(u32, u32, f64)>, Vec<u32>, Vec<u32>, Vec<f64>);

fn build_edge_graph(
    mesh: &Mesh,
    slots: usize,
    rest_pos: &[[f64; 3]],
    mode: ArapWeights,
) -> Result<EdgeGraph> {
    let tri_slots = mesh.triangle_slots();
    let mut raw_entries: Vec<(u64, f64)> = Vec::new();
    reserve(&mut raw_entries, mesh.triangle_count().saturating_mul(3))?;

    for t in 0..tri_slots {
        let Some([ia, ib, ic]) = proper_triangle(mesh, t as u32) else {
            continue;
        };
        let pa = rest_pos[ia as usize];
        let pb = rest_pos[ib as usize];
        let pc = rest_pos[ic as usize];

        let cot_a = half_cot(pa, pb, pc);
        let cot_b = half_cot(pb, pc, pa);
        let cot_c = half_cot(pc, pa, pb);

        push(&mut raw_entries, (pack_edge(ib, ic), cot_a))?;
        push(&mut raw_entries, (pack_edge(ic, ia), cot_b))?;
        push(&mut raw_entries, (pack_edge(ia, ib), cot_c))?;
    }

    raw_entries.sort_unstable_by_key(|e| e.0);

    let mut edges: Vec<(u32, u32, f64)> = Vec::new();
    let mut deg = filled(slots, 0u32)?;
    let mut pos_sum = 0.0f64;
    let mut pos_count = 0usize;

    let mut idx = 0usize;
    while idx < raw_entries.len() {
        let key = raw_entries[idx].0;
        let mut sum = raw_entries[idx].1;
        idx += 1;
        while idx < raw_entries.len() && raw_entries[idx].0 == key {
            sum += raw_entries[idx].1;
            idx += 1;
        }
        let u = (key >> 32) as u32;
        let v = (key & 0xFFFF_FFFF) as u32;
        if sum > 0.0 && sum.is_finite() {
            pos_sum += sum;
            pos_count += 1;
        }
        deg[u as usize] += 1;
        deg[v as usize] += 1;
        push(&mut edges, (u, v, sum))?;
    }

    let mean_pos = if pos_count > 0 {
        pos_sum / (pos_count as f64)
    } else {
        1.0
    };
    let floor_w = (COT_CLAMP_FRAC * mean_pos).max(1e-8);
    for e in &mut edges {
        e.2 = match mode {
            ArapWeights::Cotangent => e.2.max(floor_w),
            ArapWeights::Uniform => 1.0,
        };
    }

    let mut adj_starts: Vec<u32> = Vec::new();
    reserve(&mut adj_starts, slots + 1)?;
    let mut running = 0u32;
    for &d in &deg {
        push(&mut adj_starts, running)?;
        running += d;
    }
    push(&mut adj_starts, running)?;

    let mut cursor = adj_starts[..slots].to_vec();
    let mut adj_nbrs = filled(running as usize, 0u32)?;
    let mut adj_weights = filled(running as usize, 0.0f64)?;
    for &(u, v, w) in &edges {
        let cu = cursor[u as usize] as usize;
        adj_nbrs[cu] = v;
        adj_weights[cu] = w;
        cursor[u as usize] += 1;

        let cv = cursor[v as usize] as usize;
        adj_nbrs[cv] = u;
        adj_weights[cv] = w;
        cursor[v as usize] += 1;
    }

    Ok((edges, adj_starts, adj_nbrs, adj_weights))
}

fn compute_local_rotations(
    alive: &[bool],
    adj_starts: &[u32],
    adj_nbrs: &[u32],
    adj_weights: &[f64],
    rest_pos: &[[f64; 3]],
    curr_pos: &[[f64; 3]],
    rotations: &mut [[[f64; 3]; 3]],
) -> Result<()> {
    let slots = alive.len();
    if par::enabled() && slots >= PAR_THRESHOLD {
        allocation_allowed()?;
        par::for_each_chunk_mut(rotations, PAR_CHUNK, PAR_THRESHOLD, |chunk_idx, slice| {
            let start = chunk_idx * PAR_CHUNK;
            for (local_i, rot) in slice.iter_mut().enumerate() {
                let s = start + local_i;
                if !alive[s] {
                    continue;
                }
                *rot = vertex_optimal_rotation(
                    s,
                    adj_starts,
                    adj_nbrs,
                    adj_weights,
                    rest_pos,
                    curr_pos,
                );
            }
        });
    } else {
        for (s, rot) in rotations.iter_mut().enumerate() {
            if !alive[s] {
                continue;
            }
            *rot =
                vertex_optimal_rotation(s, adj_starts, adj_nbrs, adj_weights, rest_pos, curr_pos);
        }
    }
    Ok(())
}

#[inline]
fn vertex_optimal_rotation(
    i: usize,
    adj_starts: &[u32],
    adj_nbrs: &[u32],
    adj_weights: &[f64],
    rest_pos: &[[f64; 3]],
    curr_pos: &[[f64; 3]],
) -> [[f64; 3]; 3] {
    let begin = adj_starts[i] as usize;
    let end = adj_starts[i + 1] as usize;
    if begin == end {
        return IDENTITY_3;
    }
    let pi = rest_pos[i];
    let qi = curr_pos[i];
    let mut cov = [[0.0f64; 3]; 3];
    for e in begin..end {
        let j = adj_nbrs[e] as usize;
        let w = adj_weights[e];
        let pj = rest_pos[j];
        let qj = curr_pos[j];
        let e_rest = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]];
        let e_curr = [qi[0] - qj[0], qi[1] - qj[1], qi[2] - qj[2]];
        for r in 0..3 {
            let wr = w * e_rest[r];
            cov[r][0] += wr * e_curr[0];
            cov[r][1] += wr * e_curr[1];
            cov[r][2] += wr * e_curr[2];
        }
    }
    closest_rotation_3x3(cov)
}

#[allow(clippy::too_many_arguments)]
fn assemble_arap_rhs(
    slot_of_free: &[u32],
    adj_starts: &[u32],
    adj_nbrs: &[u32],
    adj_weights: &[f64],
    rest_pos: &[[f64; 3]],
    rotations: &[[[f64; 3]; 3]],
    pin_rhs: &[[f64; 3]],
    b_axis: &mut [Vec<f64>; 3],
) -> Result<()> {
    let n_free = slot_of_free.len();
    if par::enabled() && n_free >= PAR_THRESHOLD {
        allocation_allowed()?;
        let mut packed = filled(n_free, [0.0f64; 3])?;
        par::for_each_chunk_mut(&mut packed, PAR_CHUNK, PAR_THRESHOLD, |chunk_idx, slice| {
            let start = chunk_idx * PAR_CHUNK;
            for (local_i, out) in slice.iter_mut().enumerate() {
                let f_i = start + local_i;
                *out = vertex_rhs(
                    slot_of_free[f_i] as usize,
                    pin_rhs[f_i],
                    adj_starts,
                    adj_nbrs,
                    adj_weights,
                    rest_pos,
                    rotations,
                );
            }
        });
        for (f_i, &v) in packed.iter().enumerate() {
            b_axis[0][f_i] = v[0];
            b_axis[1][f_i] = v[1];
            b_axis[2][f_i] = v[2];
        }
    } else {
        for (f_i, &s_i_u32) in slot_of_free.iter().enumerate() {
            let v = vertex_rhs(
                s_i_u32 as usize,
                pin_rhs[f_i],
                adj_starts,
                adj_nbrs,
                adj_weights,
                rest_pos,
                rotations,
            );
            b_axis[0][f_i] = v[0];
            b_axis[1][f_i] = v[1];
            b_axis[2][f_i] = v[2];
        }
    }
    Ok(())
}

#[inline]
fn vertex_rhs(
    s_i: usize,
    pin_term: [f64; 3],
    adj_starts: &[u32],
    adj_nbrs: &[u32],
    adj_weights: &[f64],
    rest_pos: &[[f64; 3]],
    rotations: &[[[f64; 3]; 3]],
) -> [f64; 3] {
    let begin = adj_starts[s_i] as usize;
    let end = adj_starts[s_i + 1] as usize;
    let pi = rest_pos[s_i];
    let ri = rotations[s_i];
    let mut rhs = pin_term;
    for e in begin..end {
        let s_j = adj_nbrs[e] as usize;
        let half_w = 0.5 * adj_weights[e];
        let pj = rest_pos[s_j];
        let rj = rotations[s_j];
        let diff = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]];
        for row in 0..3 {
            let avg_r0 = ri[row][0] + rj[row][0];
            let avg_r1 = ri[row][1] + rj[row][1];
            let avg_r2 = ri[row][2] + rj[row][2];
            rhs[row] += half_w * (avg_r0 * diff[0] + avg_r1 * diff[1] + avg_r2 * diff[2]);
        }
    }
    rhs
}

fn solve_global_axes(
    factor: Option<&crate::sparse::Factor>,
    csr: &SymCsr<'_>,
    b_axis: &[Vec<f64>; 3],
    x_axis: &mut [Vec<f64>; 3],
) -> Result<()> {
    let max_iters = csr.n.saturating_mul(4).clamp(64, 4096);
    for a in 0..3 {
        if let Some(fac) = factor {
            fac.solve(&b_axis[a], &mut x_axis[a])?;
        } else {
            let res = solve_pcg(csr, &b_axis[a], &mut x_axis[a], 1e-9, max_iters)?;
            if !res.residual.is_finite() {
                return Err(OpsError::InvalidInput("ARAP PCG linear solve diverged"));
            }
        }
    }
    Ok(())
}

/// Computes `R = V U^T in SO(3)` maximizing `Tr(R * cov)` via one-sided 3x3 Jacobi SVD.
fn closest_rotation_3x3(cov: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut w = cov;
    let mut v = IDENTITY_3;

    for _ in 0..SVD_SWEEPS {
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            let g_pp = w[0][p] * w[0][p] + w[1][p] * w[1][p] + w[2][p] * w[2][p];
            let g_qq = w[0][q] * w[0][q] + w[1][q] * w[1][q] + w[2][q] * w[2][q];
            let g_pq = w[0][p] * w[0][q] + w[1][p] * w[1][q] + w[2][p] * w[2][q];
            if g_pq.abs() <= 1e-15 * (g_pp * g_qq).sqrt().max(1e-30) {
                continue;
            }
            let tau = (g_qq - g_pp) / (2.0 * g_pq);
            let t_abs = 1.0 / (tau.abs() + (1.0 + tau * tau).sqrt());
            let t = if tau < 0.0 { -t_abs } else { t_abs };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;
            for r in 0..3 {
                let wp = w[r][p];
                let wq = w[r][q];
                w[r][p] = c * wp - s * wq;
                w[r][q] = s * wp + c * wq;

                let vp = v[r][p];
                let vq = v[r][q];
                v[r][p] = c * vp - s * vq;
                v[r][q] = s * vp + c * vq;
            }
        }
    }

    let mut cols_w = [
        [w[0][0], w[1][0], w[2][0]],
        [w[0][1], w[1][1], w[2][1]],
        [w[0][2], w[1][2], w[2][2]],
    ];
    let mut cols_v = [
        [v[0][0], v[1][0], v[2][0]],
        [v[0][1], v[1][1], v[2][1]],
        [v[0][2], v[1][2], v[2][2]],
    ];
    let mut sig = [
        dot3(cols_w[0], cols_w[0]).sqrt(),
        dot3(cols_w[1], cols_w[1]).sqrt(),
        dot3(cols_w[2], cols_w[2]).sqrt(),
    ];

    // Sort descending by singular value magnitude (deterministic 3-element network).
    for (a, b) in [(0usize, 1usize), (0, 2), (1, 2)] {
        if sig[b] > sig[a] {
            sig.swap(a, b);
            cols_w.swap(a, b);
            cols_v.swap(a, b);
        }
    }

    if sig[0] <= 1e-18 {
        return IDENTITY_3;
    }

    let u0 = normalize3(cols_w[0]).unwrap_or([1.0, 0.0, 0.0]);
    let u1_raw = if sig[1] > 1e-12 * sig[0] {
        let proj = dot3(cols_w[1], u0);
        let orth = [
            cols_w[1][0] - proj * u0[0],
            cols_w[1][1] - proj * u0[1],
            cols_w[1][2] - proj * u0[2],
        ];
        normalize3(orth).unwrap_or_else(|| any_orthogonal_unit(u0))
    } else {
        any_orthogonal_unit(u0)
    };
    let u2 = cross3(u0, u1_raw);

    let v0 = normalize3(cols_v[0]).unwrap_or([1.0, 0.0, 0.0]);
    let v1_orth = {
        let proj = dot3(cols_v[1], v0);
        let orth = [
            cols_v[1][0] - proj * v0[0],
            cols_v[1][1] - proj * v0[1],
            cols_v[1][2] - proj * v0[2],
        ];
        normalize3(orth).unwrap_or_else(|| any_orthogonal_unit(v0))
    };
    // det(U) == +1 by u2 = u0 x u1; set v2 = v0 x v1 so det(V) == +1 and det(V U^T) == +1.
    let v2 = cross3(v0, v1_orth);

    let mut rot = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            rot[r][c] = v0[r] * u0[c] + v1_orth[r] * u1_raw[c] + v2[r] * u2[c];
        }
    }
    rot
}

const IDENTITY_3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

#[inline]
fn normalize3(v: [f64; 3]) -> Option<[f64; 3]> {
    let n2 = dot3(v, v);
    if n2 > 1e-36 {
        let inv = 1.0 / n2.sqrt();
        Some([v[0] * inv, v[1] * inv, v[2] * inv])
    } else {
        None
    }
}

#[inline]
fn any_orthogonal_unit(u: [f64; 3]) -> [f64; 3] {
    let ax = u[0].abs();
    let ay = u[1].abs();
    let az = u[2].abs();
    let basis = if ax <= ay && ax <= az {
        [1.0, 0.0, 0.0]
    } else if ay <= az {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let proj = dot3(basis, u);
    let orth = [
        basis[0] - proj * u[0],
        basis[1] - proj * u[1],
        basis[2] - proj * u[2],
    ];
    normalize3(orth).unwrap_or([0.0, 1.0, 0.0])
}

#[inline]
fn half_cot(apex: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let u = [b[0] - apex[0], b[1] - apex[1], b[2] - apex[2]];
    let w = [c[0] - apex[0], c[1] - apex[1], c[2] - apex[2]];
    let cr = cross3(u, w);
    let area2 = dot3(cr, cr).sqrt();
    if area2 <= 1e-24 {
        0.0
    } else {
        0.5 * dot3(u, w) / area2
    }
}

#[inline]
fn pack_edge(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | (hi as u64)
}

#[inline]
fn in_region(p: [f64; 3], region: Option<Region>) -> bool {
    if let Some(r) = region {
        let dx = p[0] - r.center[0] as f64;
        let dy = p[1] - r.center[1] as f64;
        let dz = p[2] - r.center[2] as f64;
        let rad = r.radius as f64;
        dx * dx + dy * dy + dz * dz <= rad * rad
    } else {
        true
    }
}

#[inline]
fn finite3(v: [f32; 3]) -> bool {
    v[0].is_finite() && v[1].is_finite() && v[2].is_finite()
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
