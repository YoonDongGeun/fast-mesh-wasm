//! 3D Sederberg-Parry Free-Form Deformation (FFD) lattice.
//!
//! Supports both global tensor-product [`FfdBasis::Bernstein`] cages and local
//! clamped uniform [`FfdBasis::CubicBSpline`] cages over either an axis-aligned
//! bounding box or a custom right-handed orthonormal [`FfdFrame`].

use fast_mesh_store::{Mesh, StoreError, Transaction, VertexId, par};

use super::FreeformReport;
use crate::protect::with_protection;
use crate::remesh::Region;
use crate::{OpsError, Result, allocation_allowed, filled, push, reserve};

const BOUNDARY_EPS: f64 = 1e-5;
const ORTHONORMAL_TOL: f64 = 1e-3;
const MIN_EXTENT: f64 = 1e-3;
const PAR_CHUNK: usize = 256;
const PAR_THRESHOLD: usize = 2048;

/// Polynomial basis used by an [`FfdLattice`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FfdBasis {
    /// Global tensor-product Bernstein polynomial basis (`Sederberg-Parry 1986`).
    #[default]
    Bernstein,
    /// Clamped uniform tensor-product B-spline basis (degree `min(n - 1, 3)`).
    CubicBSpline,
}

/// Handling of vertices whose normalized `(u, v, w)` coordinates fall outside `[0, 1]^3`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FfdOutsideMode {
    /// Leave vertices outside the lattice box unmodified.
    #[default]
    Ignore,
    /// Clamp `(u, v, w)` coordinates to `[0, 1]^3` before evaluating the cage.
    Clamp,
}

/// Orthonormal bounding frame defining the local `(u, v, w) in [0, 1]^3` parameterization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FfdFrame {
    /// World-space center of the lattice box.
    pub center: [f32; 3],
    /// Right-handed orthonormal basis axes `[x_axis, y_axis, z_axis]`.
    pub axes: [[f32; 3]; 3],
    /// Strictly positive half-extents `[hx, hy, hz]` along each local axis.
    pub half_extents: [f32; 3],
}

impl FfdFrame {
    /// Constructs a validated custom orthonormal FFD frame.
    ///
    /// # Errors
    /// Returns [`OpsError::InvalidInput`] if any field is non-finite, any half-extent is `<= 0`,
    /// or `axes` fails right-handed orthonormality checks.
    pub fn new(center: [f32; 3], axes: [[f32; 3]; 3], half_extents: [f32; 3]) -> Result<Self> {
        let frame = Self {
            center,
            axes,
            half_extents,
        };
        frame.validate()?;
        Ok(frame)
    }

    /// Constructs an axis-aligned frame spanning `[min, max]`.
    ///
    /// # Errors
    /// Returns [`OpsError::InvalidInput`] if coordinates are non-finite or `max[i] <= min[i]`.
    pub fn from_aabb(min: [f32; 3], max: [f32; 3]) -> Result<Self> {
        if !finite3(min) || !finite3(max) {
            return Err(OpsError::InvalidInput("FFD bounds must be finite"));
        }
        let mut center = [0.0f32; 3];
        let mut half = [0.0f32; 3];
        for i in 0..3 {
            let lo = min[i] as f64;
            let hi = max[i] as f64;
            if hi <= lo {
                return Err(OpsError::InvalidInput(
                    "FFD AABB requires max > min on all axes",
                ));
            }
            center[i] = (0.5 * (lo + hi)) as f32;
            half[i] = (0.5 * (hi - lo)) as f32;
            if !center[i].is_finite() || !half[i].is_finite() || half[i] <= 0.0 {
                return Err(OpsError::InvalidInput(
                    "FFD half-extents must be finite > 0",
                ));
            }
        }
        Ok(Self {
            center,
            axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            half_extents: half,
        })
    }

    /// Computes an axis-aligned frame tightly enclosing all alive vertices of `mesh`,
    /// expanded uniformly by `padding` (and floored on degenerate planar axes).
    ///
    /// # Errors
    /// Returns [`OpsError::InvalidInput`] if `mesh` has no alive vertices, has non-finite
    /// positions, or if `padding` is negative / non-finite.
    pub fn from_mesh(mesh: &Mesh, padding: f32) -> Result<Self> {
        if !padding.is_finite() || padding < 0.0 {
            return Err(OpsError::InvalidInput(
                "FFD padding must be finite and >= 0",
            ));
        }
        let positions = mesh.positions();
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        let mut count = 0usize;
        for (slot, pos) in positions.iter().enumerate() {
            if !mesh.is_vertex_alive(slot as VertexId) {
                continue;
            }
            if !finite3(*pos) {
                return Err(OpsError::InvalidInput(
                    "mesh contains non-finite vertex position",
                ));
            }
            count += 1;
            for a in 0..3 {
                let v = pos[a] as f64;
                if v < min[a] {
                    min[a] = v;
                }
                if v > max[a] {
                    max[a] = v;
                }
            }
        }
        if count == 0 {
            return Err(OpsError::InvalidInput(
                "FFD requires at least one alive vertex",
            ));
        }
        let dx = max[0] - min[0];
        let dy = max[1] - min[1];
        let dz = max[2] - min[2];
        let diag = (dx * dx + dy * dy + dz * dz).sqrt();
        let floor_half = 0.5 * (MIN_EXTENT.max(diag * 1e-3)) + padding as f64;
        let mut center = [0.0f32; 3];
        let mut half_extents = [0.0f32; 3];
        for a in 0..3 {
            center[a] = (0.5 * (min[a] + max[a])) as f32;
            let raw_half = 0.5 * (max[a] - min[a]) + padding as f64;
            half_extents[a] = raw_half.max(floor_half) as f32;
        }
        Self::new(
            center,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            half_extents,
        )
    }

    /// Validates that the frame has finite coordinates, strictly positive half-extents,
    /// and a right-handed orthonormal basis.
    pub fn validate(&self) -> Result<()> {
        if !finite3(self.center) || !finite3(self.half_extents) {
            return Err(OpsError::InvalidInput(
                "FFD frame center/extents must be finite",
            ));
        }
        for &h in &self.half_extents {
            if h <= 0.0 {
                return Err(OpsError::InvalidInput("FFD half-extents must be > 0"));
            }
        }
        let mut ax = [[0.0f64; 3]; 3];
        for (i, &v) in self.axes.iter().enumerate() {
            if !finite3(v) {
                return Err(OpsError::InvalidInput("FFD frame axis must be finite"));
            }
            ax[i] = [v[0] as f64, v[1] as f64, v[2] as f64];
            let norm = dot3(ax[i], ax[i]).sqrt();
            if (norm - 1.0).abs() > ORTHONORMAL_TOL {
                return Err(OpsError::InvalidInput("FFD frame axes must be unit length"));
            }
        }
        if dot3(ax[0], ax[1]).abs() > ORTHONORMAL_TOL
            || dot3(ax[1], ax[2]).abs() > ORTHONORMAL_TOL
            || dot3(ax[2], ax[0]).abs() > ORTHONORMAL_TOL
        {
            return Err(OpsError::InvalidInput("FFD frame axes must be orthogonal"));
        }
        let det = dot3(cross3(ax[0], ax[1]), ax[2]);
        if det <= 0.5 {
            return Err(OpsError::InvalidInput(
                "FFD frame axes must form a right-handed basis",
            ));
        }
        Ok(())
    }

    /// Converts a world-space point `p` to normalized local lattice coordinates `(u, v, w)`.
    #[inline]
    pub fn world_to_param(&self, p: [f32; 3]) -> [f64; 3] {
        let rel = [
            p[0] as f64 - self.center[0] as f64,
            p[1] as f64 - self.center[1] as f64,
            p[2] as f64 - self.center[2] as f64,
        ];
        let mut uvw = [0.0f64; 3];
        for (a, slot) in uvw.iter_mut().enumerate() {
            let axis = [
                self.axes[a][0] as f64,
                self.axes[a][1] as f64,
                self.axes[a][2] as f64,
            ];
            let coord = dot3(rel, axis);
            let half = self.half_extents[a] as f64;
            *slot = 0.5 + coord / (2.0 * half);
        }
        uvw
    }

    /// Converts normalized local lattice coordinates `(u, v, w)` in `[0, 1]^3` back to world space.
    #[inline]
    pub fn param_to_world(&self, uvw: [f64; 3]) -> [f32; 3] {
        let mut p = [
            self.center[0] as f64,
            self.center[1] as f64,
            self.center[2] as f64,
        ];
        for (a, &u_a) in uvw.iter().enumerate() {
            let s = (2.0 * u_a - 1.0) * (self.half_extents[a] as f64);
            for (k, pk) in p.iter_mut().enumerate() {
                *pk += s * (self.axes[a][k] as f64);
            }
        }
        [p[0] as f32, p[1] as f32, p[2] as f32]
    }
}

/// 3D FFD control cage consisting of `dims[0] x dims[1] x dims[2]` control point offsets.
#[derive(Clone, Debug, PartialEq)]
pub struct FfdLattice {
    dims: [usize; 3],
    frame: FfdFrame,
    basis: FfdBasis,
    offsets: Vec<[f32; 3]>,
}

impl FfdLattice {
    /// Creates an identity (zero-offset) lattice of size `dims = [nx, ny, nz]`.
    ///
    /// # Errors
    /// Returns [`OpsError::InvalidInput`] if any dimension is `< 2` or `frame` is invalid,
    /// or [`StoreError::OutOfMemory`] if allocation fails.
    pub fn new(dims: [usize; 3], frame: FfdFrame, basis: FfdBasis) -> Result<Self> {
        frame.validate()?;
        let total = validate_dims(dims)?;
        let offsets = filled(total, [0.0f32; 3])?;
        Ok(Self {
            dims,
            frame,
            basis,
            offsets,
        })
    }

    /// Creates a lattice tightly wrapping all alive vertices of `mesh` with optional `padding`.
    pub fn with_mesh(mesh: &Mesh, dims: [usize; 3], basis: FfdBasis, padding: f32) -> Result<Self> {
        let frame = FfdFrame::from_mesh(mesh, padding)?;
        Self::new(dims, frame, basis)
    }

    /// Lattice grid dimensions `[nx, ny, nz]`.
    #[inline]
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Bounding [`FfdFrame`] of this lattice.
    #[inline]
    pub fn frame(&self) -> &FfdFrame {
        &self.frame
    }

    /// Basis polynomial mode of this lattice.
    #[inline]
    pub fn basis(&self) -> FfdBasis {
        self.basis
    }

    /// Flat row-major slice of all `(nx * ny * nz)` control point displacement offsets.
    #[inline]
    pub fn offsets(&self) -> &[[f32; 3]] {
        &self.offsets
    }

    /// Mutable flat row-major slice of all `(nx * ny * nz)` control point displacement offsets.
    #[inline]
    pub fn offsets_mut(&mut self) -> &mut [[f32; 3]] {
        &mut self.offsets
    }

    /// Computes the flat row-major index `(i * ny + j) * nz + k`.
    pub fn index(&self, i: usize, j: usize, k: usize) -> Result<usize> {
        if i >= self.dims[0] || j >= self.dims[1] || k >= self.dims[2] {
            return Err(OpsError::InvalidInput(
                "FFD control point index out of bounds",
            ));
        }
        Ok((i * self.dims[1] + j) * self.dims[2] + k)
    }

    /// Returns the rest-state world position of control point `(i, j, k)`.
    ///
    /// For [`FfdBasis::Bernstein`], parameters are uniformly spaced `i / (n - 1)`;
    /// for [`FfdBasis::CubicBSpline`], Greville abscissae of the clamped uniform knot vector
    /// are used so affine control point motion reproduces affine space transforms.
    pub fn rest_position(&self, i: usize, j: usize, k: usize) -> Result<[f32; 3]> {
        let _ = self.index(i, j, k)?;
        let u = axis_param(i, self.dims[0], self.basis);
        let v = axis_param(j, self.dims[1], self.basis);
        let w = axis_param(k, self.dims[2], self.basis);
        Ok(self.frame.param_to_world([u, v, w]))
    }

    /// Returns the displaced world position `rest_position(i, j, k) + offset(i, j, k)`.
    pub fn control_position(&self, i: usize, j: usize, k: usize) -> Result<[f32; 3]> {
        let idx = self.index(i, j, k)?;
        let rest = self.rest_position(i, j, k)?;
        let off = self.offsets[idx];
        Ok([rest[0] + off[0], rest[1] + off[1], rest[2] + off[2]])
    }

    /// Returns the displacement offset at control point `(i, j, k)`.
    pub fn offset(&self, i: usize, j: usize, k: usize) -> Result<[f32; 3]> {
        let idx = self.index(i, j, k)?;
        Ok(self.offsets[idx])
    }

    /// Sets the finite displacement offset at control point `(i, j, k)`.
    pub fn set_offset(&mut self, i: usize, j: usize, k: usize, offset: [f32; 3]) -> Result<()> {
        if !finite3(offset) {
            return Err(OpsError::InvalidInput(
                "FFD control point offset must be finite",
            ));
        }
        let idx = self.index(i, j, k)?;
        self.offsets[idx] = offset;
        Ok(())
    }

    /// Adds a finite delta displacement to control point `(i, j, k)`.
    pub fn add_offset(&mut self, i: usize, j: usize, k: usize, delta: [f32; 3]) -> Result<()> {
        if !finite3(delta) {
            return Err(OpsError::InvalidInput(
                "FFD control point delta must be finite",
            ));
        }
        let idx = self.index(i, j, k)?;
        let cur = self.offsets[idx];
        let next = [cur[0] + delta[0], cur[1] + delta[1], cur[2] + delta[2]];
        if !finite3(next) {
            return Err(OpsError::InvalidInput("FFD accumulated offset overflowed"));
        }
        self.offsets[idx] = next;
        Ok(())
    }

    /// Sets the displaced world position of control point `(i, j, k)`.
    pub fn set_control_position(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
        position: [f32; 3],
    ) -> Result<()> {
        if !finite3(position) {
            return Err(OpsError::InvalidInput(
                "FFD control point position must be finite",
            ));
        }
        let rest = self.rest_position(i, j, k)?;
        self.set_offset(
            i,
            j,
            k,
            [
                position[0] - rest[0],
                position[1] - rest[1],
                position[2] - rest[2],
            ],
        )
    }

    /// Validates all lattice invariants: dimensions, frame, offset length, and finiteness.
    pub fn validate(&self) -> Result<()> {
        let total = validate_dims(self.dims)?;
        self.frame.validate()?;
        if self.offsets.len() != total {
            return Err(OpsError::InvalidInput(
                "FFD offsets length does not match lattice dimensions",
            ));
        }
        for &off in &self.offsets {
            if !finite3(off) {
                return Err(OpsError::InvalidInput(
                    "FFD offsets contain non-finite value",
                ));
            }
        }
        Ok(())
    }

    /// Evaluates the deformed world-space position of `p` under this lattice.
    ///
    /// Returns `None` if `outside == FfdOutsideMode::Ignore` and `p` lies outside the frame
    /// beyond the boundary tolerance.
    pub fn evaluate_point(&self, p: [f32; 3], outside: FfdOutsideMode) -> Option<[f32; 3]> {
        let mut uvw = self.frame.world_to_param(p);
        match outside {
            FfdOutsideMode::Ignore => {
                if uvw
                    .iter()
                    .any(|&t| t < -BOUNDARY_EPS || t > 1.0 + BOUNDARY_EPS)
                {
                    return None;
                }
            }
            FfdOutsideMode::Clamp => {}
        }
        for t in &mut uvw {
            *t = t.clamp(0.0, 1.0);
        }
        let disp = self.evaluate_displacement(uvw);
        Some([
            (p[0] as f64 + disp[0]) as f32,
            (p[1] as f64 + disp[1]) as f32,
            (p[2] as f64 + disp[2]) as f32,
        ])
    }

    fn evaluate_displacement(&self, uvw: [f64; 3]) -> [f64; 3] {
        let [nx, ny, nz] = self.dims;
        match self.basis {
            FfdBasis::Bernstein => {
                let mut wx = [0.0f64; 64];
                let mut wy = [0.0f64; 64];
                let mut wz = [0.0f64; 64];
                debug_assert!(nx <= 64 && ny <= 64 && nz <= 64);
                bernstein_weights(nx, uvw[0], &mut wx[..nx]);
                bernstein_weights(ny, uvw[1], &mut wy[..ny]);
                bernstein_weights(nz, uvw[2], &mut wz[..nz]);
                let mut acc = [0.0f64; 3];
                for (i, &wxi) in wx[..nx].iter().enumerate() {
                    if wxi == 0.0 {
                        continue;
                    }
                    for (j, &wyj) in wy[..ny].iter().enumerate() {
                        let wxy = wxi * wyj;
                        if wxy == 0.0 {
                            continue;
                        }
                        let base = (i * ny + j) * nz;
                        for (k, &wzk) in wz[..nz].iter().enumerate() {
                            let w = wxy * wzk;
                            if w == 0.0 {
                                continue;
                            }
                            let off = self.offsets[base + k];
                            acc[0] += w * (off[0] as f64);
                            acc[1] += w * (off[1] as f64);
                            acc[2] += w * (off[2] as f64);
                        }
                    }
                }
                acc
            }
            FfdBasis::CubicBSpline => {
                let (ix0, wx) = bspline_local_weights(nx, uvw[0]);
                let (iy0, wy) = bspline_local_weights(ny, uvw[1]);
                let (iz0, wz) = bspline_local_weights(nz, uvw[2]);
                let dx_len = ((nx - 1).min(3)) + 1;
                let dy_len = ((ny - 1).min(3)) + 1;
                let dz_len = ((nz - 1).min(3)) + 1;
                let mut acc = [0.0f64; 3];
                for (lx, &wxi) in wx.iter().enumerate().take(dx_len) {
                    if wxi == 0.0 {
                        continue;
                    }
                    let i = ix0 + lx;
                    for (ly, &wyj) in wy.iter().enumerate().take(dy_len) {
                        let wxy = wxi * wyj;
                        if wxy == 0.0 {
                            continue;
                        }
                        let j = iy0 + ly;
                        let base = (i * ny + j) * nz;
                        for (lz, &wzk) in wz.iter().enumerate().take(dz_len) {
                            let w = wxy * wzk;
                            if w == 0.0 {
                                continue;
                            }
                            let off = self.offsets[base + iz0 + lz];
                            acc[0] += w * (off[0] as f64);
                            acc[1] += w * (off[1] as f64);
                            acc[2] += w * (off[2] as f64);
                        }
                    }
                }
                acc
            }
        }
    }
}

/// Configuration for Free-Form Deformation via [`apply_ffd`].
#[derive(Clone, Debug, PartialEq)]
pub struct FfdOptions {
    /// Control cage lattice and offsets.
    pub lattice: FfdLattice,
    /// Behavior for vertices lying outside the lattice bounds.
    pub outside: FfdOutsideMode,
    /// Optional spherical inclusion region; vertices outside are unaffected.
    pub region: Option<Region>,
    /// Optional explicit subset of vertices to deform (all alive vertices if `None`).
    pub vertices: Option<Vec<VertexId>>,
}

impl FfdOptions {
    /// Creates options from an [`FfdLattice`] with default settings.
    pub fn new(lattice: FfdLattice) -> Self {
        Self {
            lattice,
            outside: FfdOutsideMode::Ignore,
            region: None,
            vertices: None,
        }
    }

    /// Builder-style helper to set [`FfdOutsideMode`].
    pub fn with_outside(mut self, outside: FfdOutsideMode) -> Self {
        self.outside = outside;
        self
    }

    /// Builder-style helper to scope deformation to a spherical [`Region`].
    pub fn with_region(mut self, region: Region) -> Self {
        self.region = Some(region);
        self
    }

    /// Builder-style helper to restrict deformation to an explicit vertex slice.
    pub fn with_vertices(mut self, vertices: Vec<VertexId>) -> Self {
        self.vertices = Some(vertices);
        self
    }
}

/// Applies 3D Sederberg-Parry Free-Form Deformation atomically to `tx`.
///
/// Preserves all vertex attributes including `UV_CHANNEL`, uses fallible allocations,
/// and leaves `tx` completely untouched on validation or out-of-memory errors.
///
/// # Errors
/// Returns [`OpsError::InvalidInput`] on invalid lattice/region/vertex options, or
/// [`OpsError::Store`] if allocation fails.
pub fn apply_ffd(tx: &mut Transaction, options: &FfdOptions) -> Result<FreeformReport> {
    options.lattice.validate()?;
    validate_region(options.region)?;

    let mesh = tx.mesh();
    let active_ids = collect_active_vertices(mesh, options.vertices.as_deref(), options.region)?;
    let n = active_ids.len();
    if n == 0 {
        return Ok(FreeformReport::empty());
    }

    let positions = mesh.positions();
    let mut pack: Vec<([f32; 3], u8, f32)> = filled(n, ([0.0f32; 3], 0u8, 0.0f32))?;

    let lattice = &options.lattice;
    let outside = options.outside;

    let eval_one = |v: VertexId| -> ([f32; 3], u8, f32) {
        let old_p = positions[v as usize];
        if let Some(new_p) = lattice.evaluate_point(old_p, outside) {
            let dx = new_p[0] as f64 - old_p[0] as f64;
            let dy = new_p[1] as f64 - old_p[1] as f64;
            let dz = new_p[2] as f64 - old_p[2] as f64;
            let d = (dx * dx + dy * dy + dz * dz).sqrt() as f32;
            let moved = if new_p != old_p { 1u8 } else { 0u8 };
            (new_p, moved, d)
        } else {
            (old_p, 0u8, 0.0f32)
        }
    };

    if par::enabled() && n >= PAR_THRESHOLD {
        allocation_allowed()?;
        par::for_each_chunk_mut(&mut pack, PAR_CHUNK, PAR_THRESHOLD, |chunk_idx, slice| {
            let start = chunk_idx * PAR_CHUNK;
            for (local_i, slot) in slice.iter_mut().enumerate() {
                *slot = eval_one(active_ids[start + local_i]);
            }
        });
    } else {
        for (i, slot) in pack.iter_mut().enumerate() {
            *slot = eval_one(active_ids[i]);
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

/// Applies Free-Form Deformation subject to per-vertex protection weights
/// (`0` leaves vertex frozen at rest bits, `1` keeps full deformation).
///
/// Leaves `tx` untouched if validation or allocation fails.
pub fn apply_ffd_protected(
    tx: &mut Transaction,
    options: &FfdOptions,
    protected_vertices: &[VertexId],
    protection_weights: &[f32],
) -> Result<FreeformReport> {
    with_protection(tx, protected_vertices, protection_weights, |tx| {
        apply_ffd(tx, options)
    })
}

fn validate_dims(dims: [usize; 3]) -> Result<usize> {
    if dims[0] < 2 || dims[1] < 2 || dims[2] < 2 {
        return Err(OpsError::InvalidInput(
            "FFD lattice requires dimensions >= [2, 2, 2]",
        ));
    }
    if dims[0] > 64 || dims[1] > 64 || dims[2] > 64 {
        return Err(OpsError::InvalidInput(
            "FFD lattice axis dimension must be <= 64",
        ));
    }
    dims[0]
        .checked_mul(dims[1])
        .and_then(|xy| xy.checked_mul(dims[2]))
        .ok_or(OpsError::InvalidInput("FFD lattice dimensions overflow"))
}

fn validate_region(region: Option<Region>) -> Result<()> {
    if let Some(r) = region
        && (!finite3(r.center) || !r.radius.is_finite() || r.radius <= 0.0)
    {
        return Err(OpsError::InvalidInput(
            "FFD region center must be finite and radius > 0",
        ));
    }
    Ok(())
}

fn collect_active_vertices(
    mesh: &Mesh,
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
                    "FFD vertex list contains dead or invalid VertexId",
                ));
            }
            let p = positions[v as usize];
            if !finite3(p) {
                return Err(OpsError::InvalidInput("mesh contains non-finite position"));
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
                return Err(OpsError::InvalidInput("mesh contains non-finite position"));
            }
            if in_region(p) {
                push(&mut ids, v)?;
            }
        }
    }
    Ok(ids)
}

fn bernstein_weights(n: usize, u: f64, out: &mut [f64]) {
    let deg = n - 1;
    let one_minus_u = 1.0 - u;
    out[0] = 1.0;
    for r in 1..=deg {
        out[r] = u * out[r - 1];
        for j in (1..r).rev() {
            out[j] = one_minus_u * out[j] + u * out[j - 1];
        }
        out[0] *= one_minus_u;
    }
}

fn bspline_knot(n: usize, p: usize, idx: usize) -> f64 {
    if idx <= p {
        0.0
    } else if idx >= n {
        1.0
    } else {
        (idx - p) as f64 / (n - p) as f64
    }
}

fn axis_param(i: usize, n: usize, basis: FfdBasis) -> f64 {
    match basis {
        FfdBasis::Bernstein => i as f64 / (n - 1) as f64,
        FfdBasis::CubicBSpline => {
            let p = (n - 1).min(3);
            let mut sum = 0.0f64;
            for offset in 1..=p {
                sum += bspline_knot(n, p, i + offset);
            }
            sum / (p as f64)
        }
    }
}

fn bspline_local_weights(n: usize, u: f64) -> (usize, [f64; 4]) {
    let p = (n - 1).min(3);
    let mut w = [0.0f64; 4];
    if u <= 0.0 {
        w[0] = 1.0;
        return (0, w);
    }
    if u >= 1.0 {
        w[p] = 1.0;
        return (n - 1 - p, w);
    }

    // Find span s in p..n such that t_s <= u < t_{s+1}.
    let mut span = p;
    for s in p..(n - 1) {
        if u < bspline_knot(n, p, s + 1) {
            span = s;
            break;
        }
        span = s + 1;
    }

    // Standard Cox-de Boor Algorithm A2.2 (Piegl & Tiller).
    let mut left = [0.0f64; 4];
    let mut right = [0.0f64; 4];
    w[0] = 1.0;
    for j in 1..=p {
        left[j] = u - bspline_knot(n, p, span + 1 - j);
        right[j] = bspline_knot(n, p, span + j) - u;
        let mut saved = 0.0f64;
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            let temp = if denom > 0.0 { w[r] / denom } else { 0.0 };
            w[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        w[j] = saved;
    }
    (span - p, w)
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
