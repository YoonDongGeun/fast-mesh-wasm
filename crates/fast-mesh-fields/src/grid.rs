//! Regular 3D scalar grids and BVH-accelerated distance grid builders.

#[cfg(test)]
use std::cell::Cell;

use fast_mesh_bvh::Bvh;
use fast_mesh_query::{ClosestHit, ClosestOptions, QueryError, closest_point_many};
use fast_mesh_store::Mesh;

use crate::{FieldsError, Result, SignData, signed_distances};

/// Default safety cap on total voxel count (`256^3`).
pub const DEFAULT_MAX_VOXELS: usize = 16_777_216;

/// Default automatic grid resolution along the bounding-box diagonal.
pub const DEFAULT_AUTO_GRID_RES: usize = 64;

/// Points evaluated per batch in [`bvh_unsigned_grid`] and [`bvh_signed_grid`].
/// Large enough to saturate parallel chunking (`PAR_MIN_POINTS = 512`), small
/// enough to keep temporary query buffers well under a megabyte regardless of
/// total grid resolution.
const GRID_QUERY_BATCH: usize = 16_384;

#[cfg(test)]
thread_local! {
    static ALLOCATIONS_LEFT: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Makes the `n`-th next fallible reservation on this thread (0-based) and every one
/// after it fail with [`FieldsError::OutOfMemory`]. `None` turns failures off.
#[cfg(test)]
pub(crate) fn fail_allocations_after(n: Option<usize>) {
    let was = ALLOCATIONS_LEFT.with(|c| c.replace(n)).is_some();
    if was != n.is_some() {
        fast_mesh_store::par::set_force_sequential(n.is_some());
    }
}

/// Fails with [`FieldsError::OutOfMemory`] when the test fault hook says so.
pub(crate) fn allocation_allowed() -> Result<()> {
    #[cfg(test)]
    {
        if let Some(n) = ALLOCATIONS_LEFT.with(Cell::get) {
            if n == 0 {
                return Err(FieldsError::OutOfMemory);
            }
            ALLOCATIONS_LEFT.with(|c| c.set(Some(n - 1)));
        }
    }
    Ok(())
}

/// Reserves room for at least `additional` more elements, honouring the fault hook.
pub(crate) fn reserve<T>(v: &mut Vec<T>, additional: usize) -> Result<()> {
    allocation_allowed()?;
    v.try_reserve(additional)
        .map_err(|_| FieldsError::OutOfMemory)
}

/// Reserves exact room for `additional` more elements, honouring the fault hook.
pub(crate) fn reserve_exact<T>(v: &mut Vec<T>, additional: usize) -> Result<()> {
    allocation_allowed()?;
    v.try_reserve_exact(additional)
        .map_err(|_| FieldsError::OutOfMemory)
}

/// Pushes one element after a fallible reservation when capacity is full.
pub(crate) fn push<T>(v: &mut Vec<T>, value: T) -> Result<()> {
    if v.len() == v.capacity() {
        reserve(v, 1)?;
    }
    v.push(value);
    Ok(())
}

/// Allocates a vector of `len` copies of `value` via fallible exact reservation.
pub(crate) fn filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut v = Vec::new();
    reserve_exact(&mut v, len)?;
    v.resize(len, value);
    Ok(v)
}

/// Geometry of a regular 3D Cartesian node grid in mesh-local coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridSpec {
    pub origin: [f32; 3],
    pub dims: [usize; 3],
    pub voxel_size: f32,
}

impl GridSpec {
    /// Validates `origin` (finite), `voxel_size` (finite and `> 0.0`), `dims` (each `>= 2`),
    /// checked product `nx * ny * nz <= max_voxels`, and finite upper corner
    /// `origin[k] + (dims[k] - 1) as f32 * voxel_size`. Returns total voxel count.
    pub fn validate(&self, max_voxels: usize) -> Result<usize> {
        if !self.origin.iter().all(|c| c.is_finite()) {
            return Err(FieldsError::InvalidInput("grid origin must be finite"));
        }
        if !self.voxel_size.is_finite() || self.voxel_size <= 0.0 {
            return Err(FieldsError::InvalidInput(
                "voxel_size must be finite and positive",
            ));
        }
        if !self.dims.iter().all(|&d| d >= 2) {
            return Err(FieldsError::InvalidInput(
                "grid dimensions must each be at least 2",
            ));
        }
        let total = self.dims[0]
            .checked_mul(self.dims[1])
            .and_then(|xy| xy.checked_mul(self.dims[2]))
            .ok_or(FieldsError::InvalidInput(
                "grid voxel count overflows usize",
            ))?;
        if total > max_voxels {
            return Err(FieldsError::InvalidInput("grid exceeds max_voxels"));
        }
        for k in 0..3 {
            let span_n = self.dims[k] - 1;
            let upper_f32 = self.origin[k] + span_n as f32 * self.voxel_size;
            let upper_f64 =
                f64::from(self.origin[k]) + (span_n as f64) * f64::from(self.voxel_size);
            if !upper_f32.is_finite() || !upper_f64.is_finite() {
                return Err(FieldsError::InvalidInput(
                    "grid upper corner must be finite",
                ));
            }
        }
        Ok(total)
    }

    /// Builds a grid centered on `(min + max) * 0.5` with padding
    /// `margin = margin_distance.max(0.0) + (pad_voxels.max(2) as f32) * voxel_size`
    /// on all 6 sides and `dims[k] >= 4`, validated against `max_voxels`.
    pub fn around_bounds(
        min: [f32; 3],
        max: [f32; 3],
        voxel_size: f32,
        margin_distance: f32,
        pad_voxels: usize,
        max_voxels: usize,
    ) -> Result<Self> {
        if !min.iter().chain(max.iter()).all(|c| c.is_finite()) {
            return Err(FieldsError::InvalidInput("bounds must be finite"));
        }
        if (0..3).any(|k| min[k] > max[k]) {
            return Err(FieldsError::InvalidInput("bounds min must not exceed max"));
        }
        if !voxel_size.is_finite() || voxel_size <= 0.0 {
            return Err(FieldsError::InvalidInput(
                "voxel_size must be finite and positive",
            ));
        }
        if !margin_distance.is_finite() {
            return Err(FieldsError::InvalidInput("margin_distance must be finite"));
        }

        let h = f64::from(voxel_size);
        let pad_f32 = (pad_voxels.max(2) as f32) * voxel_size;
        let margin_f32 = margin_distance.max(0.0) + pad_f32;
        if !margin_f32.is_finite() {
            return Err(FieldsError::InvalidInput("grid margin must be finite"));
        }
        let margin = f64::from(margin_f32);

        let mut origin = [0.0f32; 3];
        let mut dims = [0usize; 3];
        for k in 0..3 {
            let lo = f64::from(min[k]);
            let hi = f64::from(max[k]);
            let center = 0.5 * (lo + hi);
            let span_needed = (hi - lo) + 2.0 * margin;
            let raw_cells = (span_needed / h).ceil();
            if !raw_cells.is_finite() || raw_cells < 0.0 || raw_cells > max_voxels as f64 {
                return Err(FieldsError::InvalidInput("grid exceeds max_voxels"));
            }
            let dim = ((raw_cells as usize).saturating_add(1)).max(4);
            dims[k] = dim;
            origin[k] = (center - 0.5 * ((dim - 1) as f64) * h) as f32;
        }

        let spec = GridSpec {
            origin,
            dims,
            voxel_size,
        };
        spec.validate(max_voxels)?;
        Ok(spec)
    }

    /// Slice-major flat index `(iz * self.dims[1] + iy) * self.dims[0] + ix`.
    pub fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        debug_assert!(ix < self.dims[0] && iy < self.dims[1] && iz < self.dims[2]);
        (iz * self.dims[1] + iy) * self.dims[0] + ix
    }

    /// Local coordinates `[origin[0] + ix as f32 * h, ...]` of grid node `(ix, iy, iz)`.
    pub fn point(&self, ix: usize, iy: usize, iz: usize) -> [f32; 3] {
        [
            self.origin[0] + ix as f32 * self.voxel_size,
            self.origin[1] + iy as f32 * self.voxel_size,
            self.origin[2] + iz as f32 * self.voxel_size,
        ]
    }
}

/// Scalar values sampled at the nodes of a [`GridSpec`].
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarGrid {
    spec: GridSpec,
    values: Vec<f32>,
}

impl ScalarGrid {
    pub fn new(spec: GridSpec, values: Vec<f32>, max_voxels: usize) -> Result<Self> {
        let total = spec.validate(max_voxels)?;
        if values.len() != total {
            return Err(FieldsError::InvalidInput(
                "grid values length must match dims",
            ));
        }
        Ok(ScalarGrid { spec, values })
    }

    pub fn spec(&self) -> GridSpec {
        self.spec
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn into_values(self) -> Vec<f32> {
        self.values
    }

    /// Trilinear cell coordinates `(ix, iy, iz)` and fractional offsets `(tx, ty, tz)`.
    fn locate_cell(&self, p: [f32; 3]) -> Option<([usize; 3], [f64; 3])> {
        if !p.iter().all(|c| c.is_finite()) {
            return None;
        }
        let h = f64::from(self.spec.voxel_size);
        let mut cell = [0usize; 3];
        let mut frac = [0.0f64; 3];
        for k in 0..3 {
            let origin_k = self.spec.origin[k];
            let upper_k = origin_k + (self.spec.dims[k] - 1) as f32 * self.spec.voxel_size;
            if p[k] < origin_k || p[k] > upper_k {
                return None;
            }
            let max_coord = (self.spec.dims[k] - 1) as f64;
            let u = ((f64::from(p[k]) - f64::from(origin_k)) / h).clamp(0.0, max_coord);
            let i = (u.floor() as usize).min(self.spec.dims[k] - 2);
            cell[k] = i;
            frac[k] = (u - i as f64).clamp(0.0, 1.0);
        }
        Some((cell, frac))
    }

    /// Reads the 8 corner values of cell `(ix, iy, iz)` as `f64`, or `None` if any is non-finite.
    fn cell_corners(&self, [ix, iy, iz]: [usize; 3]) -> Option<[f64; 8]> {
        let corners = [
            self.values[self.spec.index(ix, iy, iz)],
            self.values[self.spec.index(ix + 1, iy, iz)],
            self.values[self.spec.index(ix, iy + 1, iz)],
            self.values[self.spec.index(ix + 1, iy + 1, iz)],
            self.values[self.spec.index(ix, iy, iz + 1)],
            self.values[self.spec.index(ix + 1, iy, iz + 1)],
            self.values[self.spec.index(ix, iy + 1, iz + 1)],
            self.values[self.spec.index(ix + 1, iy + 1, iz + 1)],
        ];
        if !corners.iter().all(|v| v.is_finite()) {
            return None;
        }
        Some(corners.map(f64::from))
    }

    /// Trilinear interpolation inside `[origin, upper]` (inclusive); `None` if `p` is non-finite,
    /// outside the grid bounds, or any of the 8 surrounding cell values is non-finite.
    pub fn sample(&self, p: [f32; 3]) -> Option<f32> {
        let (cell, [tx, ty, tz]) = self.locate_cell(p)?;
        let [c000, c100, c010, c110, c001, c101, c011, c111] = self.cell_corners(cell)?;

        let c00 = c000 * (1.0 - tx) + c100 * tx;
        let c10 = c010 * (1.0 - tx) + c110 * tx;
        let c01 = c001 * (1.0 - tx) + c101 * tx;
        let c11 = c011 * (1.0 - tx) + c111 * tx;

        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;

        let val = (c0 * (1.0 - tz) + c1 * tz) as f32;
        val.is_finite().then_some(val)
    }

    /// Trilinear value + trilinear / central-difference gradient `[df/dx, df/dy, df/dz]` in local units.
    pub fn sample_with_gradient(&self, p: [f32; 3]) -> Option<(f32, [f32; 3])> {
        let (cell, [tx, ty, tz]) = self.locate_cell(p)?;
        let [c000, c100, c010, c110, c001, c101, c011, c111] = self.cell_corners(cell)?;
        let inv_h = 1.0 / f64::from(self.spec.voxel_size);

        let c00 = c000 * (1.0 - tx) + c100 * tx;
        let c10 = c010 * (1.0 - tx) + c110 * tx;
        let c01 = c001 * (1.0 - tx) + c101 * tx;
        let c11 = c011 * (1.0 - tx) + c111 * tx;

        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;

        let val = (c0 * (1.0 - tz) + c1 * tz) as f32;

        let dx0 = (c100 - c000) * (1.0 - ty) + (c110 - c010) * ty;
        let dx1 = (c101 - c001) * (1.0 - ty) + (c111 - c011) * ty;
        let gx = ((dx0 * (1.0 - tz) + dx1 * tz) * inv_h) as f32;
        let gy = (((c10 - c00) * (1.0 - tz) + (c11 - c01) * tz) * inv_h) as f32;
        let gz = ((c1 - c0) * inv_h) as f32;

        let grad = [gx, gy, gz];
        (val.is_finite() && grad.iter().all(|g| g.is_finite())).then_some((val, grad))
    }
}

/// Fills `pts` with the local positions of flat indices `start..end`.
fn fill_grid_points(spec: &GridSpec, start: usize, end: usize, pts: &mut Vec<[f32; 3]>) {
    let [nx, ny, _] = spec.dims;
    let nxy = nx * ny;
    pts.clear();
    for idx in start..end {
        let iz = idx / nxy;
        let rem = idx % nxy;
        let iy = rem / nx;
        let ix = rem % nx;
        pts.push(spec.point(ix, iy, iz));
    }
}

/// Unsigned exact BVH closest-point distance grid using `closest_point_many` with `max_distance = narrow_band`
/// (`narrow_band` must be `> 0.0` and not `NaN`; `None` hits beyond `narrow_band` receive `narrow_band`).
pub fn bvh_unsigned_grid(
    mesh: &Mesh,
    bvh: &Bvh,
    spec: GridSpec,
    narrow_band: f32,
    max_voxels: usize,
) -> Result<ScalarGrid> {
    let total = spec.validate(max_voxels)?;
    if narrow_band.is_nan() || narrow_band <= 0.0 {
        return Err(FieldsError::InvalidInput(
            "narrow_band must be positive and not NaN",
        ));
    }
    if bvh.mesh_uid() != mesh.uid() {
        return Err(QueryError::MeshMismatch.into());
    }
    if !bvh.is_current(mesh) {
        return Err(QueryError::StaleBvh.into());
    }

    let mut values = filled(total, narrow_band)?;
    let batch_cap = total.min(GRID_QUERY_BATCH);
    let mut pts: Vec<[f32; 3]> = Vec::new();
    reserve_exact(&mut pts, batch_cap)?;
    let mut hits: Vec<Option<ClosestHit>> = filled(batch_cap, None)?;
    let opts = ClosestOptions {
        max_distance: narrow_band,
    };

    let mut start = 0;
    while start < total {
        let end = (start + GRID_QUERY_BATCH).min(total);
        let len = end - start;
        fill_grid_points(&spec, start, end, &mut pts);
        for h in &mut hits[..len] {
            *h = None;
        }
        closest_point_many(mesh, bvh, &pts, &opts, &mut hits[..len])?;
        for (slot, hit) in values[start..end].iter_mut().zip(&hits[..len]) {
            *slot = hit.as_ref().map_or(narrow_band, |h| h.distance);
        }
        start = end;
    }

    ScalarGrid::new(spec, values, max_voxels)
}

/// Calls `f` with each in-bounds 6-neighbor flat index of `idx` in deterministic order
/// `[-X, +X, -Y, +Y, -Z, +Z]`.
fn for_each_6_neighbor(dims: [usize; 3], idx: usize, mut f: impl FnMut(usize)) {
    let [nx, ny, nz] = dims;
    let nxy = nx * ny;
    let iz = idx / nxy;
    let rem = idx % nxy;
    let iy = rem / nx;
    let ix = rem % nx;

    if ix > 0 {
        f(idx - 1);
    }
    if ix + 1 < nx {
        f(idx + 1);
    }
    if iy > 0 {
        f(idx - nx);
    }
    if iy + 1 < ny {
        f(idx + nx);
    }
    if iz > 0 {
        f(idx - nxy);
    }
    if iz + 1 < nz {
        f(idx + nxy);
    }
}

/// Signed exact BVH distance grid using `signed_distances` with `max_distance = narrow_band`:
/// - When `narrow_band` is finite, requires `narrow_band >= 2.0 * spec.voxel_size` (else `InvalidInput`).
/// - Resolves `f32::INFINITY` far nodes via deterministic 6-neighbor BFS sign propagation seeded from
///   all finite nodes (`+narrow_band` if nearest finite neighbor `>= 0.0`, `-narrow_band` if `< 0.0`;
///   if zero nodes are finite, fills `+narrow_band`), preventing false interior sign flips inside thick solids.
pub fn bvh_signed_grid(
    mesh: &Mesh,
    bvh: &Bvh,
    sign: &SignData,
    spec: GridSpec,
    narrow_band: f32,
    max_voxels: usize,
) -> Result<ScalarGrid> {
    let total = spec.validate(max_voxels)?;
    if narrow_band.is_nan() || narrow_band <= 0.0 {
        return Err(FieldsError::InvalidInput(
            "narrow_band must be positive and not NaN",
        ));
    }
    if narrow_band.is_finite() && narrow_band < 2.0 * spec.voxel_size {
        return Err(FieldsError::InvalidInput(
            "narrow_band must be at least 2.0 * voxel_size",
        ));
    }
    if bvh.mesh_uid() != mesh.uid() {
        return Err(QueryError::MeshMismatch.into());
    }
    if !bvh.is_current(mesh) {
        return Err(QueryError::StaleBvh.into());
    }
    if !sign.is_current(mesh, bvh) {
        return Err(FieldsError::InvalidInput(
            "sign data is not current for this mesh and BVH",
        ));
    }

    let mut values = filled(total, f32::INFINITY)?;
    let batch_cap = total.min(GRID_QUERY_BATCH);
    let mut pts: Vec<[f32; 3]> = Vec::new();
    reserve_exact(&mut pts, batch_cap)?;

    let mut start = 0;
    while start < total {
        let end = (start + GRID_QUERY_BATCH).min(total);
        fill_grid_points(&spec, start, end, &mut pts);
        signed_distances(mesh, bvh, sign, &pts, narrow_band, &mut values[start..end])?;
        start = end;
    }

    let finite_count = values.iter().filter(|v| v.is_finite()).count();
    if finite_count == 0 {
        values.fill(narrow_band);
        return ScalarGrid::new(spec, values, max_voxels);
    }
    if finite_count < total && narrow_band.is_finite() {
        let mut queue: Vec<usize> = Vec::new();
        reserve_exact(&mut queue, total)?;

        for idx in 0..total {
            if !values[idx].is_finite() {
                continue;
            }
            let mut borders_far = false;
            for_each_6_neighbor(spec.dims, idx, |nb| {
                if !values[nb].is_finite() {
                    borders_far = true;
                }
            });
            if borders_far {
                queue.push(idx);
            }
        }

        let mut head = 0;
        while head < queue.len() {
            let curr = queue[head];
            head += 1;
            let fill_val = if values[curr] >= 0.0 {
                narrow_band
            } else {
                -narrow_band
            };
            for_each_6_neighbor(spec.dims, curr, |nb| {
                if !values[nb].is_finite() {
                    values[nb] = fill_val;
                    debug_assert!(queue.len() < queue.capacity());
                    queue.push(nb);
                }
            });
        }
    }

    ScalarGrid::new(spec, values, max_voxels)
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::Mesh;

    use super::*;
    use crate::SignMode;
    use crate::test_util::{brute_distance, build_bvh, cube};

    #[test]
    fn grid_spec_validation_and_trilinear_gradient() {
        let valid = GridSpec {
            origin: [-1.0, -2.0, 0.5],
            dims: [4, 5, 6],
            voxel_size: 0.5,
        };
        assert_eq!(valid.validate(DEFAULT_MAX_VOXELS), Ok(120));
        assert!(matches!(
            valid.validate(119),
            Err(FieldsError::InvalidInput(_))
        ));

        for bad in [
            GridSpec {
                origin: [f32::NAN, 0.0, 0.0],
                ..valid
            },
            GridSpec {
                origin: [0.0, f32::INFINITY, 0.0],
                ..valid
            },
            GridSpec {
                voxel_size: 0.0,
                ..valid
            },
            GridSpec {
                voxel_size: -0.5,
                ..valid
            },
            GridSpec {
                voxel_size: f32::NAN,
                ..valid
            },
            GridSpec {
                voxel_size: f32::INFINITY,
                ..valid
            },
            GridSpec {
                dims: [1, 4, 4],
                ..valid
            },
            GridSpec {
                dims: [usize::MAX, 2, 2],
                ..valid
            },
            GridSpec {
                origin: [f32::MAX, 0.0, 0.0],
                dims: [4, 4, 4],
                voxel_size: f32::MAX,
            },
        ] {
            assert!(
                matches!(
                    bad.validate(DEFAULT_MAX_VOXELS),
                    Err(FieldsError::InvalidInput(_))
                ),
                "expected InvalidInput for {bad:?}"
            );
        }

        // around_bounds edge cases & rejection
        assert!(matches!(
            GridSpec::around_bounds([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.5, 0.0, 2, 1000),
            Err(FieldsError::InvalidInput(_))
        ));
        assert!(matches!(
            GridSpec::around_bounds([0.0; 3], [1.0; 3], 0.5, f32::NAN, 2, 1000),
            Err(FieldsError::InvalidInput(_))
        ));
        assert!(matches!(
            GridSpec::around_bounds([0.0; 3], [10.0; 3], 0.01, 0.0, 2, 100),
            Err(FieldsError::InvalidInput(_))
        ));

        let around =
            GridSpec::around_bounds([-1.0, 0.0, 2.0], [1.0, 0.0, 4.0], 0.5, 0.25, 1, 10_000)
                .unwrap();
        assert!(around.dims.iter().all(|&d| d >= 4));
        let margin = 0.25 + 2.0 * 0.5;
        let upper = [
            around.origin[0] + (around.dims[0] - 1) as f32 * around.voxel_size,
            around.origin[1] + (around.dims[1] - 1) as f32 * around.voxel_size,
            around.origin[2] + (around.dims[2] - 1) as f32 * around.voxel_size,
        ];
        assert!(((around.origin[0] + upper[0]) * 0.5 - 0.0).abs() < 1e-5);
        assert!(((around.origin[1] + upper[1]) * 0.5 - 0.0).abs() < 1e-5);
        assert!(((around.origin[2] + upper[2]) * 0.5 - 3.0).abs() < 1e-5);
        assert!(around.origin[0] <= -1.0 - margin + 1e-5);
        assert!(upper[0] >= 1.0 + margin - 1e-5);
        assert!(around.origin[1] <= 0.0 - margin + 1e-5);
        assert!(upper[1] >= 0.0 + margin - 1e-5);

        // Negative margin_distance clamps cleanly to 0.0.
        let neg_margin =
            GridSpec::around_bounds([-1.0; 3], [1.0; 3], 0.5, -2.0, 2, 10_000).unwrap();
        let zero_margin =
            GridSpec::around_bounds([-1.0; 3], [1.0; 3], 0.5, 0.0, 2, 10_000).unwrap();
        assert_eq!(neg_margin, zero_margin);

        // ScalarGrid::new rejects wrong values length.
        assert!(matches!(
            ScalarGrid::new(valid, vec![0.0; 10], DEFAULT_MAX_VOXELS),
            Err(FieldsError::InvalidInput(_))
        ));

        // Linear field f(x, y, z) = 2.5*x - 1.25*y + 0.75*z + 3.0
        let total = valid.validate(DEFAULT_MAX_VOXELS).unwrap();
        let mut values = vec![0.0f32; total];
        for iz in 0..valid.dims[2] {
            for iy in 0..valid.dims[1] {
                for ix in 0..valid.dims[0] {
                    let p = valid.point(ix, iy, iz);
                    values[valid.index(ix, iy, iz)] = 2.5 * p[0] - 1.25 * p[1] + 0.75 * p[2] + 3.0;
                }
            }
        }
        let grid = ScalarGrid::new(valid, values, DEFAULT_MAX_VOXELS).unwrap();
        assert_eq!(grid.spec(), valid);
        for p in [
            valid.origin,
            valid.point(3, 4, 5),
            [-0.37, -1.13, 1.81],
            [0.25, -0.5, 2.25],
        ] {
            let want = 2.5 * p[0] - 1.25 * p[1] + 0.75 * p[2] + 3.0;
            let got = grid.sample(p).unwrap();
            let (got2, grad) = grid.sample_with_gradient(p).unwrap();
            assert!((got - want).abs() < 1e-5, "{p:?}: {got} vs {want}");
            assert_eq!(got.to_bits(), got2.to_bits());
            assert!((grad[0] - 2.5).abs() < 1e-5, "{grad:?}");
            assert!((grad[1] - (-1.25)).abs() < 1e-5, "{grad:?}");
            assert!((grad[2] - 0.75).abs() < 1e-5, "{grad:?}");
        }
        assert_eq!(grid.sample([-1.01, -2.0, 0.5]), None);
        assert_eq!(grid.sample([0.51, -2.0, 0.5]), None);
        assert_eq!(grid.sample([f32::NAN, -1.0, 1.0]), None);
        assert_eq!(grid.sample_with_gradient([f32::INFINITY, -1.0, 1.0]), None);

        // Non-linear trilinear field: analytic gradient matches finite difference inside a cell.
        let mut nl_vals = grid.into_values();
        for (i, v) in nl_vals.iter_mut().enumerate() {
            *v = ((i as f32) * 0.73).sin() * 4.0 + (i as f32) * 0.05;
        }
        let nl_grid = ScalarGrid::new(valid, nl_vals.clone(), DEFAULT_MAX_VOXELS).unwrap();
        let p = [-0.31f32, -1.17, 1.62];
        let (val, grad) = nl_grid.sample_with_gradient(p).unwrap();
        assert_eq!(val.to_bits(), nl_grid.sample(p).unwrap().to_bits());
        let eps = 1e-3f32;
        for k in 0..3 {
            let mut pp = p;
            let mut pm = p;
            pp[k] += eps;
            pm[k] -= eps;
            let fd = (nl_grid.sample(pp).unwrap() - nl_grid.sample(pm).unwrap()) / (2.0 * eps);
            assert!((grad[k] - fd).abs() < 2e-3, "axis {k}: {} vs {fd}", grad[k]);
        }

        // Non-finite cell corner makes overlapping cells return None, distant cells Some.
        nl_vals[valid.index(0, 0, 0)] = f32::INFINITY;
        let inf_grid = ScalarGrid::new(valid, nl_vals, DEFAULT_MAX_VOXELS).unwrap();
        assert_eq!(inf_grid.sample([-0.9, -1.9, 0.6]), None);
        assert_eq!(inf_grid.sample_with_gradient([-0.9, -1.9, 0.6]), None);
        assert!(inf_grid.sample([0.2, -0.5, 2.0]).is_some());
    }

    #[test]
    fn bvh_signed_grid_narrow_band_has_no_inner_sign_flip() {
        let mut mesh = cube();
        let bvh = build_bvh(&mesh);
        let sign = SignData::build(&mut mesh, &bvh, SignMode::Auto).unwrap();
        let spec = GridSpec::around_bounds([-1.0; 3], [1.0; 3], 0.125, 0.0, 3, DEFAULT_MAX_VOXELS)
            .unwrap();

        // Rejects narrow_band < 2 * h when finite, or <= 0 / NaN.
        for bad_band in [0.2, 0.0, -1.0, f32::NAN, f32::NEG_INFINITY] {
            assert!(matches!(
                bvh_signed_grid(&mesh, &bvh, &sign, spec, bad_band, DEFAULT_MAX_VOXELS),
                Err(FieldsError::InvalidInput(_))
            ));
        }
        for bad_band in [0.0, -0.5, f32::NAN, f32::NEG_INFINITY] {
            assert!(matches!(
                bvh_unsigned_grid(&mesh, &bvh, spec, bad_band, DEFAULT_MAX_VOXELS),
                Err(FieldsError::InvalidInput(_))
            ));
        }

        let band = 0.3f32;
        let signed = bvh_signed_grid(&mesh, &bvh, &sign, spec, band, DEFAULT_MAX_VOXELS).unwrap();
        let unsigned = bvh_unsigned_grid(&mesh, &bvh, spec, band, DEFAULT_MAX_VOXELS).unwrap();

        let mut deep_inside = 0usize;
        let mut deep_outside = 0usize;
        for iz in 0..spec.dims[2] {
            for iy in 0..spec.dims[1] {
                for ix in 0..spec.dims[0] {
                    let p = spec.point(ix, iy, iz);
                    let s = signed.values()[spec.index(ix, iy, iz)];
                    let u = unsigned.values()[spec.index(ix, iy, iz)];
                    assert!(s.is_finite() && u.is_finite());
                    let d = brute_distance(&mesh, p) as f32;
                    let inside_cube = p.iter().all(|&c| c.abs() < 1.0 - 1e-5);
                    let outside_cube = p.iter().any(|&c| c.abs() > 1.0 + 1e-5);
                    if inside_cube {
                        assert!(s < 0.0, "interior sign flip at {p:?}: {s}");
                        if d > band + 1e-4 {
                            assert_eq!(s, -band, "deep inside at {p:?}: {s}");
                            assert_eq!(u, band);
                            deep_inside += 1;
                        }
                    } else if outside_cube {
                        assert!(s > 0.0, "exterior sign flip at {p:?}: {s}");
                        if d > band + 1e-4 {
                            assert_eq!(s, band, "deep outside at {p:?}: {s}");
                            assert_eq!(u, band);
                            deep_outside += 1;
                        }
                    }
                    if d <= band - 1e-4 {
                        assert!((s.abs() - d).abs() < 1e-4, "near band at {p:?}: {s} vs {d}");
                        assert!(
                            (u - d).abs() < 1e-4,
                            "unsigned near band at {p:?}: {u} vs {d}"
                        );
                    }
                }
            }
        }
        assert!(
            deep_inside > 100,
            "expected many deep-interior nodes, got {deep_inside}"
        );
        assert!(
            deep_outside > 100,
            "expected many deep-exterior nodes, got {deep_outside}"
        );

        // Zero-finite-node fallback fills +narrow_band.
        let mut empty = Mesh::new();
        let ebvh = build_bvh(&empty);
        let esign = SignData::build(&mut empty, &ebvh, SignMode::Auto).unwrap();
        let small = GridSpec {
            origin: [0.0; 3],
            dims: [3, 3, 3],
            voxel_size: 0.5,
        };
        let empty_signed =
            bvh_signed_grid(&empty, &ebvh, &esign, small, 1.5, DEFAULT_MAX_VOXELS).unwrap();
        assert!(empty_signed.values().iter().all(|&v| v == 1.5));
        let empty_unsigned =
            bvh_unsigned_grid(&empty, &ebvh, small, 1.5, DEFAULT_MAX_VOXELS).unwrap();
        assert!(empty_unsigned.values().iter().all(|&v| v == 1.5));
    }
}
