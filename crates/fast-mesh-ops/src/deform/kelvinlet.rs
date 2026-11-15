//! Regularized elastostatic Kelvinlets (`Grab`, `Scale`, `Twist`, `Pinch`)
//! with multi-scale falloff control (`Single`, `BiScale`, `TriScale`), reflection symmetry,
//! spherical region scoping, and vertex protection weights (de Goes & James 2017).

use std::f64::consts::PI;

use fast_mesh_store::{StoreError, Transaction, VertexId, par};

use super::FreeformReport;
use crate::protect::with_protection;
use crate::remesh::Region;
use crate::sculpt::Symmetry;
use crate::{OpsError, Result, allocation_allowed, filled, push, reserve};

const PAR_CHUNK: usize = 256;
const PAR_THRESHOLD: usize = 2048;

/// Fundamental elastostatic Kelvinlet load kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KelvinletLoad {
    /// Concentrated regularized point force (`O(1/r)` single-scale decay).
    Grab {
        /// Force vector in world units.
        force: [f32; 3],
    },
    /// Isotropic radial volume change (`0` identically at Poisson ratio `nu = 0.5`).
    Scale {
        /// Scalar dilation strength (`> 0` expands, `< 0` shrinks when `nu < 0.5`).
        magnitude: f32,
    },
    /// Concentrated torsional couple producing divergence-free swirl.
    Twist {
        /// Torque pseudovector (`axis * strength`, right-hand rule).
        torque: [f32; 3],
    },
    /// Concentrated force dipole / stretch-compression tensor (`3x3` matrix).
    ///
    /// Symmetric traceless dipoles produce pure volume-preserving pinch at `nu = 0.5`;
    /// general tensors cleanly decompose into symmetric deviatoric, trace (scale), and skew (twist) modes.
    Pinch {
        /// Row-major `3x3` dipole tensor.
        force_dipole: [[f32; 3]; 3],
    },
}

/// Radial multi-scale falloff formulation for Kelvinlets.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum KelvinletScale {
    /// Standard single-radius regularized Kelvinlet (`epsilon`).
    #[default]
    Single,
    /// Two-scale difference (`epsilon` minus `alpha * epsilon`, `alpha > 1`),
    /// eliminating leading far-field tail while preserving center response.
    BiScale {
        /// Ratio of secondary radius to primary `epsilon` (must be `> 1.0`).
        alpha: f32,
    },
    /// Three-scale difference at `(1, 2, 3) * epsilon`, canceling the first two
    /// far-field tail orders for highly localized sculpting.
    TriScale,
}

/// Configuration for [`apply_kelvinlet`].
#[derive(Clone, Debug, PartialEq)]
pub struct KelvinletOptions {
    /// World-space load application center.
    pub center: [f32; 3],
    /// Kelvinlet load type and vector/tensor parameters.
    pub load: KelvinletLoad,
    /// Regularization radius `epsilon > 0`.
    pub epsilon: f32,
    /// Poisson ratio `nu in [0.0, 0.5]` (`0.5` yields exact incompressibility `div u = 0`).
    pub nu: f32,
    /// Elastic shear modulus `mu > 0`.
    pub mu: f32,
    /// Multi-scale falloff mode.
    pub scale: KelvinletScale,
    /// Optional reflection symmetry planes across world origin axes.
    pub symmetry: Symmetry,
    /// Optional spherical inclusion region; vertices outside are unaffected.
    pub region: Option<Region>,
    /// Optional explicit subset of vertices to deform (all alive vertices if `None`).
    pub vertices: Option<Vec<VertexId>>,
}

impl KelvinletOptions {
    /// Creates Kelvinlet options with physical defaults (`epsilon = 1.0`, `nu = 0.3`, `mu = 1.0`).
    pub fn new(center: [f32; 3], load: KelvinletLoad) -> Self {
        Self {
            center,
            load,
            epsilon: 1.0,
            nu: 0.3,
            mu: 1.0,
            scale: KelvinletScale::Single,
            symmetry: Symmetry::NONE,
            region: None,
            vertices: None,
        }
    }

    /// Sets regularization radius `epsilon > 0`.
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Sets Poisson ratio `nu in [0.0, 0.5]`.
    pub fn with_nu(mut self, nu: f32) -> Self {
        self.nu = nu;
        self
    }

    /// Sets shear modulus `mu > 0`.
    pub fn with_mu(mut self, mu: f32) -> Self {
        self.mu = mu;
        self
    }

    /// Sets multi-scale kernel mode.
    pub fn with_scale(mut self, scale: KelvinletScale) -> Self {
        self.scale = scale;
        self
    }

    /// Sets mirror symmetry flags.
    pub fn with_symmetry(mut self, symmetry: Symmetry) -> Self {
        self.symmetry = symmetry;
        self
    }

    /// Restricts evaluation to vertices inside a spherical [`Region`].
    pub fn with_region(mut self, region: Region) -> Self {
        self.region = Some(region);
        self
    }

    /// Restricts evaluation to an explicit vertex slice.
    pub fn with_vertices(mut self, vertices: Vec<VertexId>) -> Self {
        self.vertices = Some(vertices);
        self
    }
}

/// Applies regularized elastostatic Kelvinlet deformation to `tx`.
///
/// Preserves all mesh channels including `UV_CHANNEL`, uses fallible allocations,
/// and leaves `tx` untouched if validation or allocation fails.
///
/// # Errors
/// Returns [`OpsError::InvalidInput`] if physical parameters (`epsilon`, `nu`, `mu`, `alpha`, load)
/// or region/vertex inputs are invalid; returns [`OpsError::Store`] on OOM.
pub fn apply_kelvinlet(tx: &mut Transaction, options: &KelvinletOptions) -> Result<FreeformReport> {
    validate_options(options)?;

    let mesh = tx.mesh();
    let positions = mesh.positions();
    let active_ids = collect_active_vertices(mesh, options.vertices.as_deref(), options.region)?;
    let n = active_ids.len();
    if n == 0 {
        return Ok(FreeformReport::empty());
    }

    let (masks, mask_count) = options.symmetry.active_masks();
    let a = 1.0 / (4.0 * PI * options.mu as f64);
    let b = a / (4.0 * (1.0 - options.nu as f64));
    let scales = build_scales(options.load, options.epsilon as f64, options.scale);

    let mut pack: Vec<([f32; 3], u8, f32)> = filled(n, ([0.0; 3], 0u8, 0.0f32))?;
    let center = options.center;
    let load = options.load;

    if par::enabled() && n >= PAR_THRESHOLD {
        allocation_allowed()?;
        par::for_each_chunk_mut(&mut pack, PAR_CHUNK, PAR_THRESHOLD, |chunk_idx, slice| {
            let start = chunk_idx * PAR_CHUNK;
            for (local_i, slot) in slice.iter_mut().enumerate() {
                let v = active_ids[start + local_i];
                let old_p = positions[v as usize];
                *slot = evaluate_vertex(old_p, center, load, a, b, &scales, &masks[..mask_count]);
            }
        });
    } else {
        for i in 0..n {
            let old_p = positions[active_ids[i] as usize];
            pack[i] = evaluate_vertex(old_p, center, load, a, b, &scales, &masks[..mask_count]);
        }
    }

    let mut out_ids = Vec::new();
    let mut out_pos = Vec::new();
    let mut vertices_moved = 0usize;
    let mut max_displacement = 0.0f32;
    for (i, &(new_p, moved, mag)) in pack.iter().enumerate() {
        if moved != 0 {
            vertices_moved += 1;
            if mag > max_displacement {
                max_displacement = mag;
            }
            push(&mut out_ids, active_ids[i])?;
            push(&mut out_pos, new_p)?;
        }
    }

    if !out_ids.is_empty() {
        allocation_allowed()?;
        tx.set_positions(&out_ids, &out_pos)
            .map_err(|_| OpsError::Store(StoreError::OutOfMemory))?;
    }

    Ok(FreeformReport {
        vertices_evaluated: n,
        vertices_moved,
        max_displacement,
        iterations: 1,
        converged: true,
        self_collision_contacts_resolved: 0,
    })
}

/// Applies Kelvinlet deformation modulated by per-vertex protection weights
/// (`0` leaves vertex frozen at rest bits, `1` keeps full deformation).
pub fn apply_kelvinlet_protected(
    tx: &mut Transaction,
    options: &KelvinletOptions,
    protected_vertices: &[VertexId],
    protection_weights: &[f32],
) -> Result<FreeformReport> {
    with_protection(tx, protected_vertices, protection_weights, |tx| {
        apply_kelvinlet(tx, options)
    })
}

fn validate_options(options: &KelvinletOptions) -> Result<()> {
    if !finite3(options.center) {
        return Err(OpsError::InvalidInput("Kelvinlet center must be finite"));
    }
    if !options.epsilon.is_finite() || options.epsilon <= 0.0 {
        return Err(OpsError::InvalidInput(
            "Kelvinlet epsilon must be finite and > 0",
        ));
    }
    if !options.mu.is_finite() || options.mu <= 0.0 {
        return Err(OpsError::InvalidInput(
            "Kelvinlet shear modulus mu must be finite and > 0",
        ));
    }
    if !options.nu.is_finite() || options.nu < 0.0 || options.nu > 0.5 {
        return Err(OpsError::InvalidInput(
            "Kelvinlet Poisson ratio nu must be in [0.0, 0.5]",
        ));
    }
    if let KelvinletScale::BiScale { alpha } = options.scale
        && (!alpha.is_finite() || alpha <= 1.0)
    {
        return Err(OpsError::InvalidInput(
            "Kelvinlet BiScale alpha must be finite and > 1.0",
        ));
    }
    match options.load {
        KelvinletLoad::Grab { force } => {
            if !finite3(force) {
                return Err(OpsError::InvalidInput(
                    "Kelvinlet grab force must be finite",
                ));
            }
        }
        KelvinletLoad::Scale { magnitude } => {
            if !magnitude.is_finite() {
                return Err(OpsError::InvalidInput(
                    "Kelvinlet scale magnitude must be finite",
                ));
            }
        }
        KelvinletLoad::Twist { torque } => {
            if !finite3(torque) {
                return Err(OpsError::InvalidInput(
                    "Kelvinlet twist torque must be finite",
                ));
            }
        }
        KelvinletLoad::Pinch { force_dipole } => {
            for row in &force_dipole {
                if !finite3(*row) {
                    return Err(OpsError::InvalidInput(
                        "Kelvinlet force_dipole must be finite",
                    ));
                }
            }
        }
    }
    if let Some(r) = options.region
        && (!finite3(r.center) || !r.radius.is_finite() || r.radius <= 0.0)
    {
        return Err(OpsError::InvalidInput(
            "Kelvinlet region center must be finite and radius > 0",
        ));
    }
    Ok(())
}

fn collect_active_vertices(
    mesh: &fast_mesh_store::Mesh,
    explicit: Option<&[VertexId]>,
    region: Option<Region>,
) -> Result<Vec<VertexId>> {
    let positions = mesh.positions();
    let mut ids = Vec::new();
    let in_region = |p: [f32; 3]| -> bool {
        if let Some(r) = region {
            let dx = p[0] as f64 - r.center[0] as f64;
            let dy = p[1] as f64 - r.center[1] as f64;
            let dz = p[2] as f64 - r.center[2] as f64;
            let rad = r.radius as f64;
            dx * dx + dy * dy + dz * dz <= rad * rad
        } else {
            true
        }
    };

    if let Some(verts) = explicit {
        reserve(&mut ids, verts.len())?;
        for &v in verts {
            if !mesh.is_vertex_alive(v) {
                return Err(OpsError::InvalidInput(
                    "Kelvinlet vertex list contains dead or invalid VertexId",
                ));
            }
            let p = positions[v as usize];
            if !finite3(p) {
                return Err(OpsError::InvalidInput(
                    "mesh contains non-finite vertex position",
                ));
            }
            if in_region(p) {
                push(&mut ids, v)?;
            }
        }
    } else {
        let slots = mesh.vertex_slots();
        reserve(&mut ids, mesh.vertex_count())?;
        for (slot, &p) in positions.iter().enumerate().take(slots) {
            let v = slot as VertexId;
            if !mesh.is_vertex_alive(v) {
                continue;
            }
            if !finite3(p) {
                return Err(OpsError::InvalidInput(
                    "mesh contains non-finite vertex position",
                ));
            }
            if in_region(p) {
                push(&mut ids, v)?;
            }
        }
    }
    Ok(ids)
}

#[derive(Clone, Copy)]
struct ScaleEntry {
    weight: f64,
    eps: f64,
}

fn build_scales(load: KelvinletLoad, eps: f64, mode: KelvinletScale) -> [ScaleEntry; 3] {
    let is_grab = matches!(load, KelvinletLoad::Grab { .. });
    match mode {
        KelvinletScale::Single => [
            ScaleEntry { weight: 1.0, eps },
            ScaleEntry {
                weight: 0.0,
                eps: 1.0,
            },
            ScaleEntry {
                weight: 0.0,
                eps: 1.0,
            },
        ],
        KelvinletScale::BiScale { alpha } => {
            let a = alpha as f64;
            // Normalize center peak (1/eps for Grab) or center gradient (1/eps^3 for dipoles) to match Single.
            let norm = if is_grab {
                a / (a - 1.0)
            } else {
                let a3 = a * a * a;
                a3 / (a3 - 1.0)
            };
            [
                ScaleEntry { weight: norm, eps },
                ScaleEntry {
                    weight: -norm,
                    eps: a * eps,
                },
                ScaleEntry {
                    weight: 0.0,
                    eps: 1.0,
                },
            ]
        }
        KelvinletScale::TriScale => {
            // Ratios (1, 2, 3) cancel both 1/r (or 1/r^3) and the next O(eps^2) tail term.
            if is_grab {
                [
                    ScaleEntry { weight: 2.5, eps },
                    ScaleEntry {
                        weight: -4.0,
                        eps: 2.0 * eps,
                    },
                    ScaleEntry {
                        weight: 1.5,
                        eps: 3.0 * eps,
                    },
                ]
            } else {
                // 45/37 * [1, -1.6, 0.6] keeps sum(w) = 0, sum(w * k^2) = 0, and sum(w / k^3) = 1.
                let s = 45.0 / 37.0;
                [
                    ScaleEntry { weight: s, eps },
                    ScaleEntry {
                        weight: -1.6 * s,
                        eps: 2.0 * eps,
                    },
                    ScaleEntry {
                        weight: 0.6 * s,
                        eps: 3.0 * eps,
                    },
                ]
            }
        }
    }
}

#[inline]
fn evaluate_vertex(
    p: [f32; 3],
    center: [f32; 3],
    load: KelvinletLoad,
    a: f64,
    b: f64,
    scales: &[ScaleEntry; 3],
    masks: &[u8],
) -> ([f32; 3], u8, f32) {
    let mut total_disp = [0.0f64; 3];
    for &bits in masks {
        let c_m = reflect_vec(bits, center);
        let r = [
            p[0] as f64 - c_m[0] as f64,
            p[1] as f64 - c_m[1] as f64,
            p[2] as f64 - c_m[2] as f64,
        ];
        let mut disp_m = [0.0f64; 3];
        for sc in scales {
            if sc.weight == 0.0 {
                continue;
            }
            let du = match load {
                KelvinletLoad::Grab { force } => {
                    let f = reflect_vec(bits, force);
                    eval_grab(r, [f[0] as f64, f[1] as f64, f[2] as f64], sc.eps, a, b)
                }
                KelvinletLoad::Scale { magnitude } => eval_scale(r, magnitude as f64, sc.eps, a, b),
                KelvinletLoad::Twist { torque } => {
                    let t = reflect_pseudovec(bits, torque);
                    eval_twist(r, t, sc.eps, a)
                }
                KelvinletLoad::Pinch { force_dipole } => {
                    let f_m = reflect_tensor(bits, force_dipole);
                    eval_dipole(r, f_m, sc.eps, a, b)
                }
            };
            disp_m[0] += sc.weight * du[0];
            disp_m[1] += sc.weight * du[1];
            disp_m[2] += sc.weight * du[2];
        }
        total_disp[0] += disp_m[0];
        total_disp[1] += disp_m[1];
        total_disp[2] += disp_m[2];
    }

    let new_p = [
        (p[0] as f64 + total_disp[0]) as f32,
        (p[1] as f64 + total_disp[1]) as f32,
        (p[2] as f64 + total_disp[2]) as f32,
    ];
    let mag = (total_disp[0] * total_disp[0]
        + total_disp[1] * total_disp[1]
        + total_disp[2] * total_disp[2])
        .sqrt() as f32;
    let moved = if new_p != p { 1u8 } else { 0u8 };
    (new_p, moved, mag)
}

#[inline]
fn eval_grab(r: [f64; 3], f: [f64; 3], eps: f64, a: f64, b: f64) -> [f64; 3] {
    let e2 = eps * eps;
    let re2 = dot3(r, r) + e2;
    let re = re2.sqrt();
    let re3 = re * re2;
    let k1 = (a - b) / re + (0.5 * a * e2) / re3;
    let k2 = (b / re3) * dot3(r, f);
    [
        k1 * f[0] + k2 * r[0],
        k1 * f[1] + k2 * r[1],
        k1 * f[2] + k2 * r[2],
    ]
}

#[inline]
fn eval_scale(r: [f64; 3], mag: f64, eps: f64, a: f64, b: f64) -> [f64; 3] {
    let c = a - 2.0 * b;
    if c == 0.0 || mag == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let e2 = eps * eps;
    let re2 = dot3(r, r) + e2;
    let re = re2.sqrt();
    let re3 = re * re2;
    let re5 = re3 * re2;
    let factor = mag * c * (1.0 / re3 + (1.5 * e2) / re5);
    [factor * r[0], factor * r[1], factor * r[2]]
}

#[inline]
fn eval_twist(r: [f64; 3], torque: [f64; 3], eps: f64, a: f64) -> [f64; 3] {
    let e2 = eps * eps;
    let re2 = dot3(r, r) + e2;
    let re = re2.sqrt();
    let re3 = re * re2;
    let re5 = re3 * re2;
    let factor = a * (1.0 / re3 + (1.5 * e2) / re5);
    let tr = cross3(torque, r);
    [factor * tr[0], factor * tr[1], factor * tr[2]]
}

#[inline]
fn eval_dipole(r: [f64; 3], f: [[f64; 3]; 3], eps: f64, a: f64, b: f64) -> [f64; 3] {
    // Decompose F = F_sym_dev + (tr/3) * I + F_skew so each component hits its exact
    // incompressible regularization (div u == 0 at nu = 0.5 for any force_dipole).
    let tr = (f[0][0] + f[1][1] + f[2][2]) / 3.0;
    let f_dev = [
        [
            f[0][0] - tr,
            0.5 * (f[0][1] + f[1][0]),
            0.5 * (f[0][2] + f[2][0]),
        ],
        [
            0.5 * (f[1][0] + f[0][1]),
            f[1][1] - tr,
            0.5 * (f[1][2] + f[2][1]),
        ],
        [
            0.5 * (f[2][0] + f[0][2]),
            0.5 * (f[2][1] + f[1][2]),
            f[2][2] - tr,
        ],
    ];
    // F_skew * r = w x r where w = [f21 - f12, f02 - f20, f10 - f01] * 0.5
    let torque = [
        0.5 * (f[2][1] - f[1][2]),
        0.5 * (f[0][2] - f[2][0]),
        0.5 * (f[1][0] - f[0][1]),
    ];

    let e2 = eps * eps;
    let re2 = dot3(r, r) + e2;
    let re = re2.sqrt();
    let re3 = re * re2;
    let re5 = re3 * re2;

    let fr = [
        f_dev[0][0] * r[0] + f_dev[0][1] * r[1] + f_dev[0][2] * r[2],
        f_dev[1][0] * r[0] + f_dev[1][1] * r[1] + f_dev[1][2] * r[2],
        f_dev[2][0] * r[0] + f_dev[2][1] * r[1] + f_dev[2][2] * r[2],
    ];
    let rfr = dot3(r, fr);
    let c_sym = (a - 2.0 * b) / re3 + (1.5 * a * e2) / re5;
    let c_rad = (3.0 * b * rfr) / re5;

    let mut u = [
        c_sym * fr[0] + c_rad * r[0],
        c_sym * fr[1] + c_rad * r[1],
        c_sym * fr[2] + c_rad * r[2],
    ];

    let u_scale = eval_scale(r, tr, eps, a, b);
    let u_twist = eval_twist(r, torque, eps, a);
    u[0] += u_scale[0] + u_twist[0];
    u[1] += u_scale[1] + u_twist[1];
    u[2] += u_scale[2] + u_twist[2];
    u
}

#[inline]
fn reflect_vec(bits: u8, v: [f32; 3]) -> [f32; 3] {
    [
        if bits & 1 != 0 { -v[0] } else { v[0] },
        if bits & 2 != 0 { -v[1] } else { v[1] },
        if bits & 4 != 0 { -v[2] } else { v[2] },
    ]
}

#[inline]
fn reflect_signs(bits: u8) -> [f64; 3] {
    [
        if bits & 1 != 0 { -1.0 } else { 1.0 },
        if bits & 2 != 0 { -1.0 } else { 1.0 },
        if bits & 4 != 0 { -1.0 } else { 1.0 },
    ]
}

#[inline]
fn reflect_pseudovec(bits: u8, torque: [f32; 3]) -> [f64; 3] {
    let s = reflect_signs(bits);
    let det = s[0] * s[1] * s[2];
    [
        det * s[0] * torque[0] as f64,
        det * s[1] * torque[1] as f64,
        det * s[2] * torque[2] as f64,
    ]
}

#[inline]
fn reflect_tensor(bits: u8, f: [[f32; 3]; 3]) -> [[f64; 3]; 3] {
    let s = reflect_signs(bits);
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = s[i] * (f[i][j] as f64) * s[j];
        }
    }
    out
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
