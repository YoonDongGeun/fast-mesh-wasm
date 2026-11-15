//! Extended Position-Based Dynamics (XPBD) deformable surface and volume solver
//! with narrow-phase BVH self-collision avoidance.
//!
//! Solves stretch, isometric dihedral bending, closed-mesh volume conservation,
//! hard pins, compliant handles, gravity, damping, and non-adjacent Vertex-Triangle /
//! Edge-Edge minimum thickness constraints.

use std::cmp::Ordering;

use fast_mesh_store::{Mesh, StoreError, Transaction, VertexId};

use crate::topology::{UnionFind, proper_triangle};
use crate::{OpsError, Result, allocation_allowed, filled, push, reserve};

/// Configuration for [`apply_xpbd`].
#[derive(Clone, Debug, PartialEq)]
pub struct XpbdOptions {
    /// Number of physics substeps splitting total time `dt` (`>= 1`).
    pub substeps: usize,
    /// Number of Gauss-Seidel constraint projection iterations per substep (`>= 1`).
    pub iterations: usize,
    /// Total integration duration per call (`> 0`), divided into `substeps` steps.
    pub dt: f32,
    /// XPBD inverse stiffness for edge stretching (`>= 0`, `0` = infinitely stiff).
    pub stretch_compliance: f32,
    /// XPBD inverse stiffness for interior dihedral bending (`>= 0`).
    pub bend_compliance: f32,
    /// Optional XPBD inverse stiffness for closed-component volume conservation (`>= 0`).
    pub volume_compliance: Option<f32>,
    /// Linear velocity damping factor applied each substep (`[0.0, 1.0]`).
    pub damping: f32,
    /// Constant world-space gravitational acceleration vector.
    pub gravity: [f32; 3],
    /// Hard-pinned `(vertex, target)` constraints interpolated across substeps.
    pub pins: Vec<(VertexId, [f32; 3])>,
    /// Soft/compliant `(vertex, target, compliance)` handle constraints interpolated across substeps.
    pub handles: Vec<(VertexId, [f32; 3], f32)>,
    /// Enables narrow-phase BVH self-collision detection and thickness resolution.
    pub self_collision: bool,
    /// Minimum separation thickness enforced between non-adjacent surface primitives.
    pub collision_thickness: f32,
}

impl Default for XpbdOptions {
    fn default() -> Self {
        Self {
            substeps: 8,
            iterations: 6,
            dt: 1.0 / 60.0,
            stretch_compliance: 0.0,
            bend_compliance: 0.05,
            volume_compliance: None,
            damping: 0.02,
            gravity: [0.0, 0.0, 0.0],
            pins: Vec::new(),
            handles: Vec::new(),
            self_collision: false,
            collision_thickness: 0.01,
        }
    }
}

impl XpbdOptions {
    /// Creates default XPBD solver options (`8` substeps, `6` iterations, `dt = 1/60`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets substep count (`>= 1`).
    pub fn with_substeps(mut self, substeps: usize) -> Self {
        self.substeps = substeps;
        self
    }

    /// Sets solver iteration count per substep (`>= 1`).
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets total integration timestep (`> 0`).
    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// Sets edge stretch compliance (`>= 0`).
    pub fn with_stretch_compliance(mut self, compliance: f32) -> Self {
        self.stretch_compliance = compliance;
        self
    }

    /// Sets dihedral bend compliance (`>= 0`).
    pub fn with_bend_compliance(mut self, compliance: f32) -> Self {
        self.bend_compliance = compliance;
        self
    }

    /// Enables closed-component volume conservation with the given compliance (`>= 0`).
    pub fn with_volume_compliance(mut self, compliance: f32) -> Self {
        self.volume_compliance = Some(compliance);
        self
    }

    /// Sets linear velocity damping in `[0.0, 1.0]`.
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping;
        self
    }

    /// Sets gravitational acceleration.
    pub fn with_gravity(mut self, gravity: [f32; 3]) -> Self {
        self.gravity = gravity;
        self
    }

    /// Sets hard Dirichlet pin targets.
    pub fn with_pins(mut self, pins: Vec<(VertexId, [f32; 3])>) -> Self {
        self.pins = pins;
        self
    }

    /// Sets compliant handle constraints `(vertex, target, compliance)`.
    pub fn with_handles(mut self, handles: Vec<(VertexId, [f32; 3], f32)>) -> Self {
        self.handles = handles;
        self
    }

    /// Configures narrow-phase self-collision avoidance and minimum contact thickness.
    pub fn with_self_collision(mut self, enabled: bool, thickness: f32) -> Self {
        self.self_collision = enabled;
        self.collision_thickness = thickness;
        self
    }
}

/// Summary report returned by [`apply_xpbd`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct XpbdReport {
    /// Total alive vertices simulated.
    pub vertices_evaluated: usize,
    /// Number of vertices whose final positions differ from rest.
    pub vertices_moved: usize,
    /// Total number of substeps executed.
    pub substeps_run: usize,
    /// Total number of Gauss-Seidel constraint passes executed.
    pub total_iterations: usize,
    /// Maximum Euclidean displacement across moved vertices.
    pub max_displacement: f32,
    /// Number of active non-adjacent self-collision projections resolved.
    pub self_collision_contacts_resolved: usize,
}

#[derive(Clone, Copy)]
struct StretchConstraint {
    a: u32,
    b: u32,
    rest_len: f64,
}

#[derive(Clone, Copy)]
struct BendConstraint {
    p0: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    rest_angle: f64,
}

struct VolumeComponent {
    tris: Vec<[u32; 3]>,
    verts: Vec<u32>,
    rest_volume: f64,
}

#[derive(Clone, Copy)]
struct TriRef {
    v: [u32; 3],
}

#[derive(Clone, Copy)]
struct VtContact {
    q: u32,
    tri: [u32; 3],
    eff_thickness: f64,
}

#[derive(Clone, Copy)]
struct EeContact {
    e1: [u32; 2],
    e2: [u32; 2],
    eff_thickness: f64,
}

#[derive(Clone, Copy, Default)]
struct BvhNode {
    min: [f64; 3],
    max: [f64; 3],
    first_or_left: u32,
    count_or_right: u32,
    is_leaf: bool,
}

#[derive(Clone, Copy)]
struct PrimBox {
    idx: u32,
    min: [f64; 3],
    max: [f64; 3],
    centroid: [f64; 3],
}

/// Executes XPBD physics substeps on `tx` and commits the resulting vertex positions atomically.
///
/// Preserves `UV_CHANNEL`, uses fallible allocations, and leaves `tx` untouched on error or OOM.
///
/// # Errors
/// Returns [`OpsError::InvalidInput`] on invalid parameters or dead pin/handle `VertexId`s;
/// returns [`OpsError::Store`] on memory allocation failure.
pub fn apply_xpbd(tx: &mut Transaction, options: &XpbdOptions) -> Result<XpbdReport> {
    validate_options(options)?;

    let mesh = tx.mesh();
    let slots = mesh.vertex_slots();
    if slots == 0 || mesh.vertex_count() == 0 {
        return Err(OpsError::InvalidInput(
            "XPBD requires at least one alive vertex",
        ));
    }
    let positions = mesh.positions();

    let mut alive = filled(slots, false)?;
    let mut rest_pos = filled(slots, [0.0f64; 3])?;
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
        rest_pos[slot] = [p[0] as f64, p[1] as f64, p[2] as f64];
    }

    // Validate pins & handles against alive vertices and detect duplicate pin targets.
    let mut is_pinned = filled(slots, false)?;
    let mut pin_target = filled(slots, [0.0f64; 3])?;
    for &(v, tgt) in &options.pins {
        if !mesh.is_vertex_alive(v) {
            return Err(OpsError::InvalidInput(
                "XPBD pin references non-alive vertex",
            ));
        }
        if !finite3(tgt) {
            return Err(OpsError::InvalidInput("XPBD pin target must be finite"));
        }
        let s = v as usize;
        let td = [tgt[0] as f64, tgt[1] as f64, tgt[2] as f64];
        if is_pinned[s] {
            let prev = pin_target[s];
            let d2 = dist2(prev, td);
            if d2 > 1e-10 {
                return Err(OpsError::InvalidInput(
                    "conflicting XPBD pins on same vertex",
                ));
            }
        }
        is_pinned[s] = true;
        pin_target[s] = td;
    }

    let mut handles: Vec<(u32, [f64; 3], f64)> = Vec::new();
    reserve(&mut handles, options.handles.len())?;
    for &(v, tgt, comp) in &options.handles {
        if !mesh.is_vertex_alive(v) {
            return Err(OpsError::InvalidInput(
                "XPBD handle references non-alive vertex",
            ));
        }
        if !finite3(tgt) || !comp.is_finite() || comp < 0.0 {
            return Err(OpsError::InvalidInput(
                "XPBD handle target must be finite and compliance >= 0",
            ));
        }
        push(
            &mut handles,
            (
                v,
                [tgt[0] as f64, tgt[1] as f64, tgt[2] as f64],
                comp as f64,
            ),
        )?;
    }

    // Build triangles, lumped area masses, edges, dihedral bends, and closed volume components.
    let (tris, stretches, bends, volume_comps, inv_mass) = build_constraints(
        mesh,
        &alive,
        &is_pinned,
        &rest_pos,
        options.volume_compliance.is_some(),
    )?;

    let mut p_curr = rest_pos.clone();
    let mut p_prev = rest_pos.clone();
    let mut vel = filled(slots, [0.0f64; 3])?;

    let mut stretch_lambda = filled(stretches.len(), 0.0f64)?;
    let mut bend_lambda = filled(bends.len(), 0.0f64)?;
    let mut handle_lambda = filled(handles.len(), 0.0f64)?;
    let mut vol_lambda = filled(volume_comps.len(), 0.0f64)?;
    let mut vol_grads = filled(slots, [0.0f64; 3])?;

    // Pre-allocated collision scratch buffers reused across every substep.
    let mut active_verts: Vec<u32> = Vec::new();
    if options.self_collision {
        reserve(&mut active_verts, alive_count)?;
        for (s, &is_alive) in alive.iter().enumerate() {
            if is_alive {
                push(&mut active_verts, s as u32)?;
            }
        }
    }
    let mut tri_boxes: Vec<PrimBox> = Vec::new();
    let mut tri_bvh: Vec<BvhNode> = Vec::new();
    let mut edge_boxes: Vec<PrimBox> = Vec::new();
    let mut edge_bvh: Vec<BvhNode> = Vec::new();
    let mut vt_contacts: Vec<VtContact> = Vec::new();
    let mut ee_contacts: Vec<EeContact> = Vec::new();
    if options.self_collision {
        reserve(&mut tri_boxes, tris.len())?;
        reserve(&mut tri_bvh, tris.len().saturating_mul(2).max(1))?;
        reserve(&mut edge_boxes, stretches.len())?;
        reserve(&mut edge_bvh, stretches.len().saturating_mul(2).max(1))?;
    }

    let h = (options.dt as f64) / (options.substeps as f64);
    let h2 = h * h;
    let grav = [
        options.gravity[0] as f64,
        options.gravity[1] as f64,
        options.gravity[2] as f64,
    ];
    let damp = (1.0 - options.damping as f64).clamp(0.0, 1.0);
    let stretch_alpha = (options.stretch_compliance as f64) / h2;
    let bend_alpha = (options.bend_compliance as f64) / h2;
    let vol_alpha = options.volume_compliance.map(|c| (c as f64) / h2);
    let thickness = options.collision_thickness as f64;

    let mut contacts_resolved = 0usize;

    for sub in 1..=options.substeps {
        let frac = (sub as f64) / (options.substeps as f64);

        // Save substep initial pose and predict unconstrained motion.
        for s in 0..slots {
            if !alive[s] {
                continue;
            }
            p_prev[s] = p_curr[s];
            if is_pinned[s] {
                let r = rest_pos[s];
                let t = pin_target[s];
                p_curr[s] = [
                    r[0] + frac * (t[0] - r[0]),
                    r[1] + frac * (t[1] - r[1]),
                    r[2] + frac * (t[2] - r[2]),
                ];
                vel[s] = [0.0, 0.0, 0.0];
            } else {
                vel[s][0] = vel[s][0] * damp + h * grav[0];
                vel[s][1] = vel[s][1] * damp + h * grav[1];
                vel[s][2] = vel[s][2] * damp + h * grav[2];
                p_curr[s][0] += h * vel[s][0];
                p_curr[s][1] += h * vel[s][1];
                p_curr[s][2] += h * vel[s][2];
            }
        }

        // Reset Lagrange multipliers once per substep (Macklin et al. 2016).
        stretch_lambda.fill(0.0);
        bend_lambda.fill(0.0);
        handle_lambda.fill(0.0);
        vol_lambda.fill(0.0);

        if options.self_collision && !tris.is_empty() {
            collect_self_collision_candidates(
                &active_verts,
                &tris,
                &stretches,
                &rest_pos,
                &p_prev,
                &p_curr,
                thickness,
                &mut tri_boxes,
                &mut tri_bvh,
                &mut edge_boxes,
                &mut edge_bvh,
                &mut vt_contacts,
                &mut ee_contacts,
            )?;
        }

        for _iter in 0..options.iterations {
            // Handles (linearly interpolated toward target across substeps).
            for (hi, &(v_u32, tgt, comp)) in handles.iter().enumerate() {
                let s = v_u32 as usize;
                let w = inv_mass[s];
                if w <= 0.0 {
                    continue;
                }
                let r = rest_pos[s];
                let des = [
                    r[0] + frac * (tgt[0] - r[0]),
                    r[1] + frac * (tgt[1] - r[1]),
                    r[2] + frac * (tgt[2] - r[2]),
                ];
                let diff = sub3(p_curr[s], des);
                let len = norm3(diff);
                if len <= 1e-12 {
                    continue;
                }
                let alpha_t = comp / h2;
                let denom = w + alpha_t;
                if denom <= 1e-24 {
                    continue;
                }
                let dlambda = (-len - alpha_t * handle_lambda[hi]) / denom;
                handle_lambda[hi] += dlambda;
                let scale = w * dlambda / len;
                p_curr[s][0] += scale * diff[0];
                p_curr[s][1] += scale * diff[1];
                p_curr[s][2] += scale * diff[2];
            }

            // Edge stretch constraints.
            for (ei, st) in stretches.iter().enumerate() {
                let a = st.a as usize;
                let b = st.b as usize;
                let wa = inv_mass[a];
                let wb = inv_mass[b];
                let wsum = wa + wb;
                if wsum <= 0.0 {
                    continue;
                }
                let d = sub3(p_curr[a], p_curr[b]);
                let len = norm3(d);
                if len <= 1e-12 {
                    continue;
                }
                let denom = wsum + stretch_alpha;
                let c = len - st.rest_len;
                let dlambda = (-c - stretch_alpha * stretch_lambda[ei]) / denom;
                stretch_lambda[ei] += dlambda;
                let n = [d[0] / len, d[1] / len, d[2] / len];
                p_curr[a][0] += wa * dlambda * n[0];
                p_curr[a][1] += wa * dlambda * n[1];
                p_curr[a][2] += wa * dlambda * n[2];
                p_curr[b][0] -= wb * dlambda * n[0];
                p_curr[b][1] -= wb * dlambda * n[1];
                p_curr[b][2] -= wb * dlambda * n[2];
            }

            // Isometric dihedral bending constraints.
            for (bi, bd) in bends.iter().enumerate() {
                solve_bend_constraint(bd, bend_alpha, &mut bend_lambda[bi], &inv_mass, &mut p_curr);
            }

            // Closed-component volume conservation constraints.
            if let Some(v_alpha) = vol_alpha {
                for (ci, comp) in volume_comps.iter().enumerate() {
                    solve_volume_constraint(
                        comp,
                        v_alpha,
                        &mut vol_lambda[ci],
                        &inv_mass,
                        &mut vol_grads,
                        &mut p_curr,
                    );
                }
            }

            // Narrow-phase Vertex-Triangle and Edge-Edge thickness constraints.
            if options.self_collision {
                for c in &vt_contacts {
                    if resolve_vt_contact(c, &inv_mass, &p_prev, &mut p_curr) {
                        contacts_resolved += 1;
                    }
                }
                for c in &ee_contacts {
                    if resolve_ee_contact(c, &inv_mass, &p_prev, &mut p_curr) {
                        contacts_resolved += 1;
                    }
                }
            }
        }

        // Update XPBD velocities from corrected positions and enforce hard pins cleanly.
        let inv_h = 1.0 / h;
        for s in 0..slots {
            if !alive[s] {
                continue;
            }
            if is_pinned[s] {
                vel[s] = [0.0, 0.0, 0.0];
            } else {
                vel[s] = [
                    (p_curr[s][0] - p_prev[s][0]) * inv_h,
                    (p_curr[s][1] - p_prev[s][1]) * inv_h,
                    (p_curr[s][2] - p_prev[s][2]) * inv_h,
                ];
            }
        }
    }

    let mut out_ids: Vec<VertexId> = Vec::new();
    let mut out_pos: Vec<[f32; 3]> = Vec::new();
    let mut max_disp = 0.0f64;
    for s in 0..slots {
        if !alive[s] {
            continue;
        }
        let r = rest_pos[s];
        let c = p_curr[s];
        if !c[0].is_finite() || !c[1].is_finite() || !c[2].is_finite() {
            return Err(OpsError::InvalidInput(
                "XPBD produced non-finite vertex position",
            ));
        }
        let old32 = [r[0] as f32, r[1] as f32, r[2] as f32];
        let new32 = [c[0] as f32, c[1] as f32, c[2] as f32];
        if new32 != old32 {
            let d = norm3(sub3(c, r));
            if d > max_disp {
                max_disp = d;
            }
            push(&mut out_ids, s as VertexId)?;
            push(&mut out_pos, new32)?;
        }
    }

    if !out_ids.is_empty() {
        allocation_allowed()?;
        tx.set_positions(&out_ids, &out_pos)
            .map_err(|_| OpsError::Store(StoreError::OutOfMemory))?;
    }

    Ok(XpbdReport {
        vertices_evaluated: alive_count,
        vertices_moved: out_ids.len(),
        substeps_run: options.substeps,
        total_iterations: options.substeps * options.iterations,
        max_displacement: max_disp as f32,
        self_collision_contacts_resolved: contacts_resolved,
    })
}

fn validate_options(options: &XpbdOptions) -> Result<()> {
    if options.substeps == 0 {
        return Err(OpsError::InvalidInput("XPBD substeps must be >= 1"));
    }
    if options.iterations == 0 {
        return Err(OpsError::InvalidInput("XPBD iterations must be >= 1"));
    }
    if !options.dt.is_finite() || options.dt <= 0.0 {
        return Err(OpsError::InvalidInput("XPBD dt must be finite and > 0"));
    }
    if !options.stretch_compliance.is_finite() || options.stretch_compliance < 0.0 {
        return Err(OpsError::InvalidInput(
            "XPBD stretch_compliance must be finite and >= 0",
        ));
    }
    if !options.bend_compliance.is_finite() || options.bend_compliance < 0.0 {
        return Err(OpsError::InvalidInput(
            "XPBD bend_compliance must be finite and >= 0",
        ));
    }
    if let Some(vc) = options.volume_compliance
        && (!vc.is_finite() || vc < 0.0)
    {
        return Err(OpsError::InvalidInput(
            "XPBD volume_compliance must be finite and >= 0",
        ));
    }
    if !options.damping.is_finite() || options.damping < 0.0 || options.damping > 1.0 {
        return Err(OpsError::InvalidInput("XPBD damping must be in [0.0, 1.0]"));
    }
    if !finite3(options.gravity) {
        return Err(OpsError::InvalidInput("XPBD gravity must be finite"));
    }
    if !options.collision_thickness.is_finite()
        || (options.self_collision && options.collision_thickness <= 0.0)
        || (!options.self_collision && options.collision_thickness < 0.0)
    {
        return Err(OpsError::InvalidInput(
            "XPBD collision_thickness must be finite and positive when self_collision is enabled",
        ));
    }
    Ok(())
}

type ConstraintBuild = (
    Vec<TriRef>,
    Vec<StretchConstraint>,
    Vec<BendConstraint>,
    Vec<VolumeComponent>,
    Vec<f64>,
);

fn build_constraints(
    mesh: &Mesh,
    alive: &[bool],
    is_pinned: &[bool],
    rest_pos: &[[f64; 3]],
    want_volume: bool,
) -> Result<ConstraintBuild> {
    let slots = alive.len();
    let tri_slots = mesh.triangle_slots();
    let mut tris: Vec<TriRef> = Vec::new();
    reserve(&mut tris, mesh.triangle_count())?;

    let mut mass = filled(slots, 0.0f64)?;
    let mut half_edges: Vec<(u64, u32)> = Vec::new();
    reserve(&mut half_edges, mesh.triangle_count().saturating_mul(3))?;

    let mut uf = UnionFind::new();
    uf.reset(slots)?;

    for t in 0..tri_slots {
        let Some([a, b, c]) = proper_triangle(mesh, t as u32) else {
            continue;
        };
        push(&mut tris, TriRef { v: [a, b, c] })?;
        uf.union(a, b);
        uf.union(b, c);

        let pa = rest_pos[a as usize];
        let pb = rest_pos[b as usize];
        let pc = rest_pos[c as usize];
        let area = 0.5 * norm3(cross3(sub3(pb, pa), sub3(pc, pa)));
        let m_third = area / 3.0;
        mass[a as usize] += m_third;
        mass[b as usize] += m_third;
        mass[c as usize] += m_third;

        push(&mut half_edges, (pack_edge(a, b), c))?;
        push(&mut half_edges, (pack_edge(b, c), a))?;
        push(&mut half_edges, (pack_edge(c, a), b))?;
    }

    // Normalize lumped masses so average active mass is 1.0 (invariant scale across meshes).
    let mut mass_sum = 0.0f64;
    let mut mass_count = 0usize;
    for (s, &is_alive) in alive.iter().enumerate() {
        if is_alive && mass[s] > 0.0 {
            mass_sum += mass[s];
            mass_count += 1;
        }
    }
    let avg_mass = if mass_count > 0 {
        mass_sum / (mass_count as f64)
    } else {
        1.0
    };
    let mut inv_mass = filled(slots, 0.0f64)?;
    for (s, &is_alive) in alive.iter().enumerate() {
        if !is_alive || is_pinned[s] {
            inv_mass[s] = 0.0;
        } else if mass[s] <= 0.0 {
            inv_mass[s] = 1.0;
        } else {
            let rel_m = (mass[s] / avg_mass).max(1e-3);
            inv_mass[s] = 1.0 / rel_m;
        }
    }

    half_edges.sort_unstable_by_key(|e| e.0);

    let mut stretches: Vec<StretchConstraint> = Vec::new();
    let mut bends: Vec<BendConstraint> = Vec::new();
    let mut comp_closed = filled(slots, true)?;

    let mut idx = 0usize;
    while idx < half_edges.len() {
        let key = half_edges[idx].0;
        let opp_start = idx;
        while idx < half_edges.len() && half_edges[idx].0 == key {
            idx += 1;
        }
        let u = (key >> 32) as u32;
        let v = (key & 0xFFFF_FFFF) as u32;
        let count = idx - opp_start;
        if count != 2 {
            let root = uf.find(u) as usize;
            comp_closed[root] = false;
        }

        let rest_len = norm3(sub3(rest_pos[u as usize], rest_pos[v as usize]));
        if rest_len > 1e-12 {
            push(
                &mut stretches,
                StretchConstraint {
                    a: u,
                    b: v,
                    rest_len,
                },
            )?;
        }

        if count == 2 {
            let p2 = half_edges[opp_start].1;
            let p3 = half_edges[opp_start + 1].1;
            if p2 != p3
                && let Some(rest_angle) = compute_dihedral_angle(
                    rest_pos[u as usize],
                    rest_pos[v as usize],
                    rest_pos[p2 as usize],
                    rest_pos[p3 as usize],
                )
            {
                push(
                    &mut bends,
                    BendConstraint {
                        p0: u,
                        p1: v,
                        p2,
                        p3,
                        rest_angle,
                    },
                )?;
            }
        }
    }

    // Collect closed components with non-zero rest volume when volume conservation is enabled.
    let mut volume_comps: Vec<VolumeComponent> = Vec::new();
    if want_volume && !tris.is_empty() {
        let mut comp_id_map = filled(slots, u32::MAX)?;
        let mut tri_groups: Vec<Vec<[u32; 3]>> = Vec::new();
        for t in &tris {
            let root = uf.find(t.v[0]) as usize;
            if !comp_closed[root] {
                continue;
            }
            if comp_id_map[root] == u32::MAX {
                comp_id_map[root] = tri_groups.len() as u32;
                push(&mut tri_groups, Vec::new())?;
            }
            let gi = comp_id_map[root] as usize;
            push(&mut tri_groups[gi], t.v)?;
        }

        for group in tri_groups {
            let rest_vol = signed_volume(&group, rest_pos);
            if rest_vol.abs() <= 1e-12 {
                continue;
            }
            let mut verts: Vec<u32> = Vec::new();
            reserve(&mut verts, group.len().saturating_mul(3))?;
            for t in &group {
                push(&mut verts, t[0])?;
                push(&mut verts, t[1])?;
                push(&mut verts, t[2])?;
            }
            verts.sort_unstable();
            verts.dedup();
            push(
                &mut volume_comps,
                VolumeComponent {
                    tris: group,
                    verts,
                    rest_volume: rest_vol,
                },
            )?;
        }
    }

    Ok((tris, stretches, bends, volume_comps, inv_mass))
}

#[inline]
fn compute_dihedral_angle(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> Option<f64> {
    let e = sub3(p1, p0);
    let len_e = norm3(e);
    if len_e <= 1e-12 {
        return None;
    }
    let e_hat = [e[0] / len_e, e[1] / len_e, e[2] / len_e];
    let n1 = cross3(e, sub3(p2, p0));
    let n2 = cross3(sub3(p3, p0), e);
    let l1 = norm3(n1);
    let l2 = norm3(n2);
    if l1 <= 1e-12 || l2 <= 1e-12 {
        return None;
    }
    let u1 = [n1[0] / l1, n1[1] / l1, n1[2] / l1];
    let u2 = [n2[0] / l2, n2[1] / l2, n2[2] / l2];
    let cos_t = dot3(u1, u2).clamp(-1.0, 1.0);
    let sin_t = dot3(cross3(u1, u2), e_hat);
    Some(sin_t.atan2(cos_t))
}

fn solve_bend_constraint(
    bd: &BendConstraint,
    alpha: f64,
    lambda: &mut f64,
    inv_mass: &[f64],
    p: &mut [[f64; 3]],
) {
    let i0 = bd.p0 as usize;
    let i1 = bd.p1 as usize;
    let i2 = bd.p2 as usize;
    let i3 = bd.p3 as usize;
    let w0 = inv_mass[i0];
    let w1 = inv_mass[i1];
    let w2 = inv_mass[i2];
    let w3 = inv_mass[i3];
    if w0 + w1 + w2 + w3 <= 0.0 {
        return;
    }

    let p0 = p[i0];
    let p1 = p[i1];
    let p2 = p[i2];
    let p3 = p[i3];

    let e = sub3(p1, p0);
    let len_e = norm3(e);
    if len_e <= 1e-12 {
        return;
    }
    let inv_le2 = 1.0 / (len_e * len_e);
    let e_hat = [e[0] / len_e, e[1] / len_e, e[2] / len_e];

    let n1 = cross3(e, sub3(p2, p0));
    let n2 = cross3(sub3(p3, p0), e);
    let n1_sq = dot3(n1, n1);
    let n2_sq = dot3(n2, n2);
    if n1_sq <= 1e-24 || n2_sq <= 1e-24 {
        return;
    }

    let l1 = n1_sq.sqrt();
    let l2 = n2_sq.sqrt();
    let u1_n = [n1[0] / l1, n1[1] / l1, n1[2] / l1];
    let u2_n = [n2[0] / l2, n2[1] / l2, n2[2] / l2];
    let theta = dot3(cross3(u1_n, u2_n), e_hat).atan2(dot3(u1_n, u2_n).clamp(-1.0, 1.0));
    // Wrapped signed angular difference in (-pi, pi] avoids 2*pi branch discontinuities.
    let diff = theta - bd.rest_angle;
    let c = diff.sin().atan2(diff.cos());

    let s1 = -len_e / n1_sq;
    let s2 = -len_e / n2_sq;
    let g2 = [s1 * n1[0], s1 * n1[1], s1 * n1[2]];
    let g3 = [s2 * n2[0], s2 * n2[1], s2 * n2[2]];

    let u1 = dot3(sub3(p2, p0), e) * inv_le2;
    let v1 = 1.0 - u1;
    let u2 = dot3(sub3(p3, p0), e) * inv_le2;
    let v2 = 1.0 - u2;

    let g0 = [
        -(v1 * g2[0] + v2 * g3[0]),
        -(v1 * g2[1] + v2 * g3[1]),
        -(v1 * g2[2] + v2 * g3[2]),
    ];
    let g1 = [
        -(u1 * g2[0] + u2 * g3[0]),
        -(u1 * g2[1] + u2 * g3[1]),
        -(u1 * g2[2] + u2 * g3[2]),
    ];

    let sum_wg = w0 * dot3(g0, g0) + w1 * dot3(g1, g1) + w2 * dot3(g2, g2) + w3 * dot3(g3, g3);
    let denom = sum_wg + alpha;
    if denom <= 1e-24 {
        return;
    }

    let dlambda = (-c - alpha * (*lambda)) / denom;
    *lambda += dlambda;

    for a in 0..3 {
        p[i0][a] += w0 * dlambda * g0[a];
        p[i1][a] += w1 * dlambda * g1[a];
        p[i2][a] += w2 * dlambda * g2[a];
        p[i3][a] += w3 * dlambda * g3[a];
    }
}

fn solve_volume_constraint(
    comp: &VolumeComponent,
    alpha: f64,
    lambda: &mut f64,
    inv_mass: &[f64],
    grads: &mut [[f64; 3]],
    p: &mut [[f64; 3]],
) {
    for &v in &comp.verts {
        grads[v as usize] = [0.0, 0.0, 0.0];
    }
    let mut vol6 = 0.0f64;
    let sixth = 1.0 / 6.0;
    for &[ia, ib, ic] in &comp.tris {
        let a = p[ia as usize];
        let b = p[ib as usize];
        let c = p[ic as usize];
        let bc = cross3(b, c);
        let ca = cross3(c, a);
        let ab = cross3(a, b);
        vol6 += dot3(a, bc);
        for k in 0..3 {
            grads[ia as usize][k] += sixth * bc[k];
            grads[ib as usize][k] += sixth * ca[k];
            grads[ic as usize][k] += sixth * ab[k];
        }
    }
    let v_curr = vol6 * sixth;
    let c = v_curr - comp.rest_volume;

    let mut sum_wg = 0.0f64;
    for &v in &comp.verts {
        let idx = v as usize;
        let w = inv_mass[idx];
        if w > 0.0 {
            let g = grads[idx];
            sum_wg += w * dot3(g, g);
        }
    }
    let denom = sum_wg + alpha;
    if denom <= 1e-24 {
        return;
    }
    let dlambda = (-c - alpha * (*lambda)) / denom;
    *lambda += dlambda;
    for &v in &comp.verts {
        let idx = v as usize;
        let w = inv_mass[idx];
        if w > 0.0 {
            let g = grads[idx];
            p[idx][0] += w * dlambda * g[0];
            p[idx][1] += w * dlambda * g[1];
            p[idx][2] += w * dlambda * g[2];
        }
    }
}

fn signed_volume(tris: &[[u32; 3]], pos: &[[f64; 3]]) -> f64 {
    let mut sum6 = 0.0f64;
    for &[ia, ib, ic] in tris {
        sum6 += dot3(pos[ia as usize], cross3(pos[ib as usize], pos[ic as usize]));
    }
    sum6 / 6.0
}

#[allow(clippy::too_many_arguments)]
fn collect_self_collision_candidates(
    active_verts: &[u32],
    tris: &[TriRef],
    stretches: &[StretchConstraint],
    rest_pos: &[[f64; 3]],
    p_prev: &[[f64; 3]],
    p_curr: &[[f64; 3]],
    thickness: f64,
    tri_boxes: &mut Vec<PrimBox>,
    tri_bvh: &mut Vec<BvhNode>,
    edge_boxes: &mut Vec<PrimBox>,
    edge_bvh: &mut Vec<BvhNode>,
    vt_contacts: &mut Vec<VtContact>,
    ee_contacts: &mut Vec<EeContact>,
) -> Result<()> {
    vt_contacts.clear();
    ee_contacts.clear();
    tri_boxes.clear();
    tri_bvh.clear();
    edge_boxes.clear();
    edge_bvh.clear();

    let pad = thickness * 1.5;

    for (ti, tri) in tris.iter().enumerate() {
        let (bmin, bmax, cent) = swept_tri_box(tri.v, p_prev, p_curr, 0.0);
        push(
            tri_boxes,
            PrimBox {
                idx: ti as u32,
                min: bmin,
                max: bmax,
                centroid: cent,
            },
        )?;
    }
    if !tri_boxes.is_empty() {
        build_bvh(tri_boxes, tri_bvh)?;
        let mut stack = [0u32; 64];
        for &vq in active_verts {
            let qi = vq as usize;
            let mut qmin = [0.0f64; 3];
            let mut qmax = [0.0f64; 3];
            for a in 0..3 {
                qmin[a] = p_prev[qi][a].min(p_curr[qi][a]) - pad;
                qmax[a] = p_prev[qi][a].max(p_curr[qi][a]) + pad;
            }
            let mut sp = 1usize;
            stack[0] = 0;
            while sp > 0 {
                sp -= 1;
                let ni = stack[sp] as usize;
                let node = &tri_bvh[ni];
                if !aabb_overlap(qmin, qmax, node.min, node.max) {
                    continue;
                }
                if node.is_leaf {
                    let start = node.first_or_left as usize;
                    let end = start + node.count_or_right as usize;
                    for box_item in &tri_boxes[start..end] {
                        let tri = tris[box_item.idx as usize].v;
                        if vq == tri[0] || vq == tri[1] || vq == tri[2] {
                            continue;
                        }
                        if !aabb_overlap(qmin, qmax, box_item.min, box_item.max) {
                            continue;
                        }
                        let (cp_rest, _, _, _) = closest_point_on_triangle(
                            rest_pos[qi],
                            rest_pos[tri[0] as usize],
                            rest_pos[tri[1] as usize],
                            rest_pos[tri[2] as usize],
                        );
                        let rest_dist = norm3(sub3(rest_pos[qi], cp_rest));
                        let eff = thickness.min(0.85 * rest_dist);
                        if eff > 1e-12 {
                            push(
                                vt_contacts,
                                VtContact {
                                    q: vq,
                                    tri,
                                    eff_thickness: eff,
                                },
                            )?;
                        }
                    }
                } else {
                    if sp + 2 <= stack.len() {
                        stack[sp] = node.first_or_left;
                        stack[sp + 1] = node.count_or_right;
                        sp += 2;
                    }
                }
            }
        }
    }

    for (ei, e) in stretches.iter().enumerate() {
        let (bmin, bmax, cent) = swept_edge_box(e.a, e.b, p_prev, p_curr, 0.0);
        push(
            edge_boxes,
            PrimBox {
                idx: ei as u32,
                min: bmin,
                max: bmax,
                centroid: cent,
            },
        )?;
    }
    if !edge_boxes.is_empty() {
        build_bvh(edge_boxes, edge_bvh)?;
        let mut stack = [0u32; 64];
        for (ei, e1) in stretches.iter().enumerate() {
            let e1_u32 = ei as u32;
            let (qmin, qmax, _) = swept_edge_box(e1.a, e1.b, p_prev, p_curr, pad);
            let mut sp = 1usize;
            stack[0] = 0;
            while sp > 0 {
                sp -= 1;
                let ni = stack[sp] as usize;
                let node = &edge_bvh[ni];
                if !aabb_overlap(qmin, qmax, node.min, node.max) {
                    continue;
                }
                if node.is_leaf {
                    let start = node.first_or_left as usize;
                    let end = start + node.count_or_right as usize;
                    for box_item in &edge_boxes[start..end] {
                        if box_item.idx <= e1_u32 {
                            continue;
                        }
                        let e2 = stretches[box_item.idx as usize];
                        if e1.a == e2.a || e1.a == e2.b || e1.b == e2.a || e1.b == e2.b {
                            continue;
                        }
                        if !aabb_overlap(qmin, qmax, box_item.min, box_item.max) {
                            continue;
                        }
                        let (c1, c2, _, _) = closest_points_segments(
                            rest_pos[e1.a as usize],
                            rest_pos[e1.b as usize],
                            rest_pos[e2.a as usize],
                            rest_pos[e2.b as usize],
                        );
                        let rest_dist = norm3(sub3(c1, c2));
                        let eff = thickness.min(0.85 * rest_dist);
                        if eff > 1e-12 {
                            push(
                                ee_contacts,
                                EeContact {
                                    e1: [e1.a, e1.b],
                                    e2: [e2.a, e2.b],
                                    eff_thickness: eff,
                                },
                            )?;
                        }
                    }
                } else if sp + 2 <= stack.len() {
                    stack[sp] = node.first_or_left;
                    stack[sp + 1] = node.count_or_right;
                    sp += 2;
                }
            }
        }
    }

    // Sort candidate constraints so traversal order is strictly invariant to BVH partition order.
    vt_contacts.sort_unstable_by_key(|c| (c.q, c.tri));
    ee_contacts.sort_unstable_by_key(|c| (c.e1, c.e2));
    Ok(())
}

fn resolve_vt_contact(
    c: &VtContact,
    inv_mass: &[f64],
    p_prev: &[[f64; 3]],
    p_curr: &mut [[f64; 3]],
) -> bool {
    let iq = c.q as usize;
    let ia = c.tri[0] as usize;
    let ib = c.tri[1] as usize;
    let ic = c.tri[2] as usize;
    let wq = inv_mass[iq];
    let wa = inv_mass[ia];
    let wb = inv_mass[ib];
    let wc = inv_mass[ic];
    if wq + wa + wb + wc <= 0.0 {
        return false;
    }

    let (cp_prev, _, _, _) =
        closest_point_on_triangle(p_prev[iq], p_prev[ia], p_prev[ib], p_prev[ic]);
    let diff_prev = sub3(p_prev[iq], cp_prev);
    let d_prev = norm3(diff_prev);
    let s_prev = if d_prev > 1e-12 {
        [
            diff_prev[0] / d_prev,
            diff_prev[1] / d_prev,
            diff_prev[2] / d_prev,
        ]
    } else {
        let fn_prev = cross3(sub3(p_prev[ib], p_prev[ia]), sub3(p_prev[ic], p_prev[ia]));
        normalize3(fn_prev).unwrap_or([0.0, 1.0, 0.0])
    };

    let (cp_curr, u, v, w) =
        closest_point_on_triangle(p_curr[iq], p_curr[ia], p_curr[ib], p_curr[ic]);
    let diff_curr = sub3(p_curr[iq], cp_curr);
    let dist = norm3(diff_curr);
    let h = c.eff_thickness;

    let same_side = dot3(diff_curr, s_prev) >= 0.0;
    let (push_dir, pen) = if same_side {
        if dist >= h {
            return false;
        }
        let dir = if dist > 1e-12 {
            [
                diff_curr[0] / dist,
                diff_curr[1] / dist,
                diff_curr[2] / dist,
            ]
        } else {
            s_prev
        };
        (dir, h - dist)
    } else if d_prev < 4.0 * h {
        // Vertex tunneled to the negative side of s_prev during this substep: push back across.
        let dir = if dist > 1e-12 {
            [
                -diff_curr[0] / dist,
                -diff_curr[1] / dist,
                -diff_curr[2] / dist,
            ]
        } else {
            s_prev
        };
        (dir, h + dist)
    } else {
        return false;
    };

    let denom = wq + u * u * wa + v * v * wb + w * w * wc;
    if denom <= 1e-24 || pen <= 0.0 {
        return false;
    }
    let scale = pen / denom;
    for k in 0..3 {
        p_curr[iq][k] += (wq * scale) * push_dir[k];
        p_curr[ia][k] -= (u * wa * scale) * push_dir[k];
        p_curr[ib][k] -= (v * wb * scale) * push_dir[k];
        p_curr[ic][k] -= (w * wc * scale) * push_dir[k];
    }
    true
}

fn resolve_ee_contact(
    c: &EeContact,
    inv_mass: &[f64],
    p_prev: &[[f64; 3]],
    p_curr: &mut [[f64; 3]],
) -> bool {
    let i0 = c.e1[0] as usize;
    let i1 = c.e1[1] as usize;
    let j0 = c.e2[0] as usize;
    let j1 = c.e2[1] as usize;
    let w0 = inv_mass[i0];
    let w1 = inv_mass[i1];
    let w2 = inv_mass[j0];
    let w3 = inv_mass[j1];
    if w0 + w1 + w2 + w3 <= 0.0 {
        return false;
    }

    let (cp1_prev, cp2_prev, _, _) =
        closest_points_segments(p_prev[i0], p_prev[i1], p_prev[j0], p_prev[j1]);
    let diff_prev = sub3(cp1_prev, cp2_prev);
    let d_prev = norm3(diff_prev);

    let (cp1_curr, cp2_curr, s, t) =
        closest_points_segments(p_curr[i0], p_curr[i1], p_curr[j0], p_curr[j1]);
    let diff_curr = sub3(cp1_curr, cp2_curr);
    let dist = norm3(diff_curr);
    let h = c.eff_thickness;

    let same_side = d_prev <= 1e-12 || dot3(diff_curr, diff_prev) >= 0.0;
    let (push_dir, pen) = if same_side {
        if dist >= h {
            return false;
        }
        let dir = if dist > 1e-12 {
            [
                diff_curr[0] / dist,
                diff_curr[1] / dist,
                diff_curr[2] / dist,
            ]
        } else if d_prev > 1e-12 {
            [
                diff_prev[0] / d_prev,
                diff_prev[1] / d_prev,
                diff_prev[2] / d_prev,
            ]
        } else {
            let ex = cross3(sub3(p_curr[i1], p_curr[i0]), sub3(p_curr[j1], p_curr[j0]));
            normalize3(ex).unwrap_or([0.0, 1.0, 0.0])
        };
        (dir, h - dist)
    } else if d_prev < 3.0 * h {
        let dir = if dist > 1e-12 {
            [
                -diff_curr[0] / dist,
                -diff_curr[1] / dist,
                -diff_curr[2] / dist,
            ]
        } else {
            [
                diff_prev[0] / d_prev,
                diff_prev[1] / d_prev,
                diff_prev[2] / d_prev,
            ]
        };
        (dir, h + dist)
    } else {
        return false;
    };

    let b0 = 1.0 - s;
    let b1 = s;
    let b2 = 1.0 - t;
    let b3 = t;
    let denom = b0 * b0 * w0 + b1 * b1 * w1 + b2 * b2 * w2 + b3 * b3 * w3;
    if denom <= 1e-24 || pen <= 0.0 {
        return false;
    }
    let scale = pen / denom;
    for k in 0..3 {
        p_curr[i0][k] += (b0 * w0 * scale) * push_dir[k];
        p_curr[i1][k] += (b1 * w1 * scale) * push_dir[k];
        p_curr[j0][k] -= (b2 * w2 * scale) * push_dir[k];
        p_curr[j1][k] -= (b3 * w3 * scale) * push_dir[k];
    }
    true
}

fn build_bvh(boxes: &mut [PrimBox], nodes: &mut Vec<BvhNode>) -> Result<()> {
    nodes.clear();
    let n = boxes.len();
    if n == 0 {
        return Ok(());
    }
    push(nodes, BvhNode::default())?;
    build_bvh_rec(boxes, 0, n, 0, 0, nodes)?;
    Ok(())
}

fn build_bvh_rec(
    boxes: &mut [PrimBox],
    start: usize,
    end: usize,
    node_idx: usize,
    depth: usize,
    nodes: &mut Vec<BvhNode>,
) -> Result<()> {
    let mut bmin = [f64::INFINITY; 3];
    let mut bmax = [f64::NEG_INFINITY; 3];
    let mut cmin = [f64::INFINITY; 3];
    let mut cmax = [f64::NEG_INFINITY; 3];
    for item in &boxes[start..end] {
        for a in 0..3 {
            bmin[a] = bmin[a].min(item.min[a]);
            bmax[a] = bmax[a].max(item.max[a]);
            cmin[a] = cmin[a].min(item.centroid[a]);
            cmax[a] = cmax[a].max(item.centroid[a]);
        }
    }

    let count = end - start;
    if count <= 4 || depth >= 56 {
        nodes[node_idx] = BvhNode {
            min: bmin,
            max: bmax,
            first_or_left: start as u32,
            count_or_right: count as u32,
            is_leaf: true,
        };
        return Ok(());
    }

    let ex = cmax[0] - cmin[0];
    let ey = cmax[1] - cmin[1];
    let ez = cmax[2] - cmin[2];
    let axis = if ex >= ey && ex >= ez {
        0
    } else if ey >= ez {
        1
    } else {
        2
    };

    let mid = start + count / 2;
    boxes[start..end].select_nth_unstable_by(mid - start, |a, b| {
        a.centroid[axis]
            .partial_cmp(&b.centroid[axis])
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.idx.cmp(&b.idx))
    });

    let left_idx = nodes.len();
    push(nodes, BvhNode::default())?;
    let right_idx = nodes.len();
    push(nodes, BvhNode::default())?;

    nodes[node_idx] = BvhNode {
        min: bmin,
        max: bmax,
        first_or_left: left_idx as u32,
        count_or_right: right_idx as u32,
        is_leaf: false,
    };

    build_bvh_rec(boxes, start, mid, left_idx, depth + 1, nodes)?;
    build_bvh_rec(boxes, mid, end, right_idx, depth + 1, nodes)?;
    Ok(())
}

#[inline]
fn swept_tri_box(
    v: [u32; 3],
    p_prev: &[[f64; 3]],
    p_curr: &[[f64; 3]],
    pad: f64,
) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let mut bmin = [f64::INFINITY; 3];
    let mut bmax = [f64::NEG_INFINITY; 3];
    for &idx in &v {
        let a = p_prev[idx as usize];
        let b = p_curr[idx as usize];
        for k in 0..3 {
            bmin[k] = bmin[k].min(a[k]).min(b[k]);
            bmax[k] = bmax[k].max(a[k]).max(b[k]);
        }
    }
    let mut cent = [0.0f64; 3];
    for k in 0..3 {
        cent[k] = 0.5 * (bmin[k] + bmax[k]);
        bmin[k] -= pad;
        bmax[k] += pad;
    }
    (bmin, bmax, cent)
}

#[inline]
fn swept_edge_box(
    u: u32,
    v: u32,
    p_prev: &[[f64; 3]],
    p_curr: &[[f64; 3]],
    pad: f64,
) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let mut bmin = [f64::INFINITY; 3];
    let mut bmax = [f64::NEG_INFINITY; 3];
    for idx in [u as usize, v as usize] {
        let a = p_prev[idx];
        let b = p_curr[idx];
        for k in 0..3 {
            bmin[k] = bmin[k].min(a[k]).min(b[k]);
            bmax[k] = bmax[k].max(a[k]).max(b[k]);
        }
    }
    let mut cent = [0.0f64; 3];
    for k in 0..3 {
        cent[k] = 0.5 * (bmin[k] + bmax[k]);
        bmin[k] -= pad;
        bmax[k] += pad;
    }
    (bmin, bmax, cent)
}

#[inline]
fn aabb_overlap(amin: [f64; 3], amax: [f64; 3], bmin: [f64; 3], bmax: [f64; 3]) -> bool {
    amin[0] <= bmax[0]
        && amax[0] >= bmin[0]
        && amin[1] <= bmax[1]
        && amax[1] >= bmin[1]
        && amin[2] <= bmax[2]
        && amax[2] >= bmin[2]
}

/// Closest point on triangle `(a, b, c)` to `p`, returning `(point, u, v, w)` (Ericson 2005).
fn closest_point_on_triangle(
    p: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> ([f64; 3], f64, f64, f64) {
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    let ap = sub3(p, a);
    let d1 = dot3(ab, ap);
    let d2 = dot3(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, 1.0, 0.0, 0.0);
    }

    let bp = sub3(p, b);
    let d3 = dot3(ab, bp);
    let d4 = dot3(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, 0.0, 1.0, 0.0);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let u = 1.0 - v;
        return (
            [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]],
            u,
            v,
            0.0,
        );
    }

    let cp = sub3(p, c);
    let d5 = dot3(ab, cp);
    let d6 = dot3(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, 0.0, 0.0, 1.0);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let u = 1.0 - w;
        return (
            [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]],
            u,
            0.0,
            w,
        );
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let v = 1.0 - w;
        return (
            [
                b[0] + w * (c[0] - b[0]),
                b[1] + w * (c[1] - b[1]),
                b[2] + w * (c[2] - b[2]),
            ],
            0.0,
            v,
            w,
        );
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let u = 1.0 - v - w;
    (
        [
            a[0] + ab[0] * v + ac[0] * w,
            a[1] + ab[1] * v + ac[1] * w,
            a[2] + ab[2] * v + ac[2] * w,
        ],
        u,
        v,
        w,
    )
}

/// Closest point pair `(c1, c2, s, t)` between segments `(p1, q1)` and `(p2, q2)` (Ericson 2005).
fn closest_points_segments(
    p1: [f64; 3],
    q1: [f64; 3],
    p2: [f64; 3],
    q2: [f64; 3],
) -> ([f64; 3], [f64; 3], f64, f64) {
    let d1 = sub3(q1, p1);
    let d2 = sub3(q2, p2);
    let r = sub3(p1, p2);
    let a = dot3(d1, d1);
    let e = dot3(d2, d2);
    let f = dot3(d2, r);
    let eps = 1e-18;

    let (s, t) = if a <= eps && e <= eps {
        (0.0, 0.0)
    } else if a <= eps {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = dot3(d1, r);
        if e <= eps {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = dot3(d1, d2);
            let denom = a * e - b * b;
            let mut s_val = if denom != 0.0 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut t_val = (b * s_val + f) / e;
            if t_val < 0.0 {
                t_val = 0.0;
                s_val = (-c / a).clamp(0.0, 1.0);
            } else if t_val > 1.0 {
                t_val = 1.0;
                s_val = ((b - c) / a).clamp(0.0, 1.0);
            }
            (s_val, t_val)
        }
    };

    let c1 = [p1[0] + d1[0] * s, p1[1] + d1[1] * s, p1[2] + d1[2] * s];
    let c2 = [p2[0] + d2[0] * t, p2[1] + d2[1] * t, p2[2] + d2[2] * t];
    (c1, c2, s, t)
}

#[inline]
fn pack_edge(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | (hi as u64)
}

#[inline]
fn finite3(v: [f32; 3]) -> bool {
    v[0].is_finite() && v[1].is_finite() && v[2].is_finite()
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = sub3(a, b);
    dot3(d, d)
}

#[inline]
fn norm3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}

#[inline]
fn normalize3(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = norm3(v);
    if n > 1e-18 {
        Some([v[0] / n, v[1] / n, v[2] / n])
    } else {
        None
    }
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
