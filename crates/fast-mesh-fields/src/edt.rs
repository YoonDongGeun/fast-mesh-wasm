//! Felzenszwalb-Huttenlocher (2012) 3D Euclidean distance transforms and mesh voxelization grids.

use fast_mesh_store::{Mesh, TriangleId, par};
use robust::{Coord, orient2d};

use crate::geom::{cross, dot, sub, triangle_normal, widen};
use crate::grid::{allocation_allowed, filled, push, reserve_exact};
use crate::{FieldsError, GridSpec, Result, ScalarGrid};

/// Cost cutoff at or above which an [`edt_3d_sq`] entry is treated as an absent site.
const EDT_INF_F32: f32 = 1e30;

/// Grid size above which separable EDT and voxelization passes dispatch to Rayon.
const PAR_MIN_EDT_VOXELS: usize = 16_384;

/// 1D lines grouped per parallel task during [`edt_3d_sq`].
const EDT_LINE_CHUNK: usize = 64;

/// Scratch buffers for a 1D lower parabolic envelope of length up to `cap`.
struct EdtScratch {
    f: Vec<f64>,
    d: Vec<f64>,
    v: Vec<usize>,
    z: Vec<f64>,
}

impl EdtScratch {
    fn new(cap: usize) -> Result<Self> {
        Ok(EdtScratch {
            f: filled(cap, 0.0)?,
            d: filled(cap, 0.0)?,
            v: filled(cap, 0)?,
            z: filled(cap + 1, 0.0)?,
        })
    }
}

/// Exact 1D Felzenszwalb-Huttenlocher squared distance transform:
/// `d[x] = min_{q: f[q].is_finite()} ((x - q)^2 + f[q])`, or `+INFINITY` if none is finite.
fn edt_1d(f: &[f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    debug_assert_eq!(d.len(), n);
    debug_assert!(v.len() >= n);
    debug_assert!(z.len() > n);

    if n == 0 {
        return;
    }
    if n == 1 {
        d[0] = f[0];
        return;
    }

    let mut k = 0usize;
    for (q, &fq) in f.iter().enumerate() {
        if !fq.is_finite() {
            continue;
        }
        if k == 0 {
            v[0] = q;
            z[0] = f64::NEG_INFINITY;
            z[1] = f64::INFINITY;
            k = 1;
            continue;
        }
        let qf = q as f64;
        let s = loop {
            let p = v[k - 1];
            let pf = p as f64;
            let s = ((fq + qf * qf) - (f[p] + pf * pf)) / (2.0 * (qf - pf));
            if s > z[k - 1] {
                break s;
            }
            k -= 1;
            if k == 0 {
                break f64::NEG_INFINITY;
            }
        };
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
        k += 1;
    }

    if k == 0 {
        d.fill(f64::INFINITY);
        return;
    }

    let mut m = 0usize;
    for (x, out) in d.iter_mut().enumerate() {
        let xf = x as f64;
        while z[m + 1] < xf {
            m += 1;
        }
        let dx = xf - v[m] as f64;
        *out = dx * dx + f[v[m]];
    }
}

fn checked_grid_total(dims: [usize; 3]) -> Result<usize> {
    if dims.contains(&0) {
        return Err(FieldsError::InvalidInput(
            "edt dimensions must each be at least 1",
        ));
    }
    dims[0]
        .checked_mul(dims[1])
        .and_then(|xy| xy.checked_mul(dims[2]))
        .ok_or(FieldsError::InvalidInput("edt voxel count overflows usize"))
}

/// Separable 3D Felzenszwalb-Huttenlocher (2012) squared Euclidean distance transform in grid-index units:
/// `out_sq[p] = min_q ((px - qx)^2 + (py - qy)^2 + (pz - qz)^2 + cost[q])`.
/// - `dims` must each be `>= 1` (`checked_mul`), `cost.len() == out_sq.len() == total`.
/// - Rejects `NaN` or negative finite `cost[q]` with `FieldsError::InvalidInput`.
/// - `+INFINITY` (or `>= 1e30`) means "absent site": skipped before parabola intersection arithmetic
///   so `Inf - Inf = NaN` never occurs. If no finite site exists anywhere in `cost`, fills `out_sq`
///   with `f32::INFINITY`.
pub fn edt_3d_sq(dims: [usize; 3], cost: &[f32], out_sq: &mut [f32]) -> Result<()> {
    let total = checked_grid_total(dims)?;
    if cost.len() != total || out_sq.len() != total {
        return Err(FieldsError::InvalidInput(
            "edt cost and out_sq lengths must match dims",
        ));
    }

    let mut any_finite = false;
    for &c in cost {
        if c.is_nan() || c < 0.0 {
            return Err(FieldsError::InvalidInput(
                "edt cost must be non-negative and not NaN",
            ));
        }
        if c.is_finite() && c < EDT_INF_F32 {
            any_finite = true;
        }
    }
    if !any_finite {
        out_sq.fill(f32::INFINITY);
        return Ok(());
    }

    let [nx, ny, nz] = dims;
    let mut buf = Vec::new();
    reserve_exact(&mut buf, total)?;
    for &c in cost {
        buf.push(if c.is_finite() && c < EDT_INF_F32 {
            if c == 0.0 { 0.0 } else { f64::from(c) }
        } else {
            f64::INFINITY
        });
    }

    let parallel = total >= PAR_MIN_EDT_VOXELS && par::enabled() && !par::force_sequential();
    if !parallel {
        let max_dim = nx.max(ny).max(nz);
        let mut sc = EdtScratch::new(max_dim)?;

        if nx > 1 {
            for row in buf.chunks_mut(nx) {
                sc.f[..nx].copy_from_slice(row);
                edt_1d(&sc.f[..nx], row, &mut sc.v[..nx], &mut sc.z[..=nx]);
            }
        }

        if ny > 1 {
            let nxy = nx * ny;
            for iz in 0..nz {
                let z_base = iz * nxy;
                for ix in 0..nx {
                    for iy in 0..ny {
                        sc.f[iy] = buf[z_base + iy * nx + ix];
                    }
                    edt_1d(
                        &sc.f[..ny],
                        &mut sc.d[..ny],
                        &mut sc.v[..ny],
                        &mut sc.z[..=ny],
                    );
                    for iy in 0..ny {
                        buf[z_base + iy * nx + ix] = sc.d[iy];
                    }
                }
            }
        }

        if nz > 1 {
            let nxy = nx * ny;
            for col in 0..nxy {
                for iz in 0..nz {
                    sc.f[iz] = buf[iz * nxy + col];
                }
                edt_1d(
                    &sc.f[..nz],
                    &mut sc.d[..nz],
                    &mut sc.v[..nz],
                    &mut sc.z[..=nz],
                );
                for iz in 0..nz {
                    out_sq[iz * nxy + col] = sc.d[iz] as f32;
                }
            }
        } else {
            for (o, &b) in out_sq.iter_mut().zip(&buf) {
                *o = b as f32;
            }
        }
        return Ok(());
    }

    let oom = || FieldsError::OutOfMemory;

    // Pass 1: X lines (contiguous rows of length nx).
    if nx > 1 {
        let n_rows = ny * nz;
        let batch_rows = (262_144 / nx).max(EDT_LINE_CHUNK * 2);
        let mut row_start = 0;
        while row_start < n_rows {
            let row_end = (row_start + batch_rows).min(n_rows);
            let batch_len = row_end - row_start;
            allocation_allowed()?;
            let parts = {
                let buf_ref = &buf;
                par::map_range(batch_len, EDT_LINE_CHUNK, 0, oom, |r| {
                    let mut sc = EdtScratch::new(nx)?;
                    let mut part = Vec::new();
                    reserve_exact(&mut part, r.len() * nx)?;
                    for local in r {
                        let row = row_start + local;
                        let slice = &buf_ref[row * nx..(row + 1) * nx];
                        edt_1d(slice, &mut sc.d[..nx], &mut sc.v[..nx], &mut sc.z[..=nx]);
                        part.extend_from_slice(&sc.d[..nx]);
                    }
                    Ok(part)
                })?
            };
            let mut cursor = row_start * nx;
            for part in parts {
                buf[cursor..cursor + part.len()].copy_from_slice(&part);
                cursor += part.len();
            }
            row_start = row_end;
        }
    }

    // Pass 2: Y lines (indexed by c = iz * nx + ix).
    if ny > 1 {
        let n_cols = nx * nz;
        let batch_cols = (262_144 / ny).max(EDT_LINE_CHUNK * 2);
        let mut col_start = 0;
        while col_start < n_cols {
            let col_end = (col_start + batch_cols).min(n_cols);
            let batch_len = col_end - col_start;
            allocation_allowed()?;
            let parts = {
                let buf_ref = &buf;
                par::map_range(batch_len, EDT_LINE_CHUNK, 0, oom, |r| {
                    let mut sc = EdtScratch::new(ny)?;
                    let mut part = Vec::new();
                    reserve_exact(&mut part, r.len() * ny)?;
                    for local in r {
                        let c = col_start + local;
                        let iz = c / nx;
                        let ix = c % nx;
                        let z_base = iz * nx * ny;
                        for iy in 0..ny {
                            sc.f[iy] = buf_ref[z_base + iy * nx + ix];
                        }
                        edt_1d(
                            &sc.f[..ny],
                            &mut sc.d[..ny],
                            &mut sc.v[..ny],
                            &mut sc.z[..=ny],
                        );
                        part.extend_from_slice(&sc.d[..ny]);
                    }
                    Ok(part)
                })?
            };
            let mut offset = col_start;
            for part in parts {
                let count = part.len() / ny;
                for i in 0..count {
                    let c = offset + i;
                    let iz = c / nx;
                    let ix = c % nx;
                    let z_base = iz * nx * ny;
                    let src = &part[i * ny..(i + 1) * ny];
                    for (iy, &val) in src.iter().enumerate() {
                        buf[z_base + iy * nx + ix] = val;
                    }
                }
                offset += count;
            }
            col_start = col_end;
        }
    }

    // Pass 3: Z lines (indexed by col = iy * nx + ix).
    if nz > 1 {
        let nxy = nx * ny;
        let batch_cols = (262_144 / nz).max(EDT_LINE_CHUNK * 2);
        let mut col_start = 0;
        while col_start < nxy {
            let col_end = (col_start + batch_cols).min(nxy);
            let batch_len = col_end - col_start;
            allocation_allowed()?;
            let parts = {
                let buf_ref = &buf;
                par::map_range(batch_len, EDT_LINE_CHUNK, 0, oom, |r| {
                    let mut sc = EdtScratch::new(nz)?;
                    let mut part = Vec::new();
                    reserve_exact(&mut part, r.len() * nz)?;
                    for local in r {
                        let col = col_start + local;
                        for iz in 0..nz {
                            sc.f[iz] = buf_ref[iz * nxy + col];
                        }
                        edt_1d(
                            &sc.f[..nz],
                            &mut sc.d[..nz],
                            &mut sc.v[..nz],
                            &mut sc.z[..=nz],
                        );
                        for &val in &sc.d[..nz] {
                            part.push(val as f32);
                        }
                    }
                    Ok(part)
                })?
            };
            let mut offset = col_start;
            for part in parts {
                let count = part.len() / nz;
                for i in 0..count {
                    let col = offset + i;
                    let src = &part[i * nz..(i + 1) * nz];
                    for (iz, &val) in src.iter().enumerate() {
                        out_sq[iz * nxy + col] = val;
                    }
                }
                offset += count;
            }
            col_start = col_end;
        }
    } else {
        for (o, &b) in out_sq.iter_mut().zip(&buf) {
            *o = b as f32;
        }
    }

    Ok(())
}

/// Binary mask 3D Euclidean distance transform in local units:
/// seeds `0.0` where `mask[i]` is `true` and `+INFINITY` where `false`, runs `edt_3d_sq`,
/// and writes `voxel_size * sq.sqrt()` (`f32::INFINITY` if `mask` has no `true` entry).
/// `voxel_size` must be finite and `> 0.0`.
pub fn edt_3d_mask(
    dims: [usize; 3],
    mask: &[bool],
    voxel_size: f32,
    out: &mut [f32],
) -> Result<()> {
    let total = checked_grid_total(dims)?;
    if mask.len() != total || out.len() != total {
        return Err(FieldsError::InvalidInput(
            "edt mask and out lengths must match dims",
        ));
    }
    if !voxel_size.is_finite() || voxel_size <= 0.0 {
        return Err(FieldsError::InvalidInput(
            "voxel_size must be finite and positive",
        ));
    }
    if !mask.iter().any(|&b| b) {
        out.fill(f32::INFINITY);
        return Ok(());
    }

    let mut cost = Vec::new();
    reserve_exact(&mut cost, total)?;
    for &b in mask {
        cost.push(if b { 0.0 } else { f32::INFINITY });
    }
    edt_3d_sq(dims, &cost, out)?;
    for v in out.iter_mut() {
        *v = if v.is_finite() {
            voxel_size * v.sqrt()
        } else {
            f32::INFINITY
        };
    }
    Ok(())
}

/// Conservative index range `[lo, hi]` along axis `k` whose grid nodes lie in `[min_v, max_v]`.
/// Returns `None` if the interval misses the grid axis completely.
fn axis_index_range(spec: &GridSpec, k: usize, min_v: f64, max_v: f64) -> Option<(usize, usize)> {
    let n = spec.dims[k];
    let origin = f64::from(spec.origin[k]);
    let h = f64::from(spec.voxel_size);
    let raw_lo = ((min_v - origin) / h).floor() - 1.0;
    let raw_hi = ((max_v - origin) / h).ceil() + 1.0;
    if raw_hi < 0.0 || raw_lo >= n as f64 {
        return None;
    }
    let lo = if raw_lo <= 0.0 { 0 } else { raw_lo as usize };
    let hi = (raw_hi.max(0.0) as usize).min(n - 1);
    (lo <= hi).then_some((lo, hi))
}

/// Exact `f64` Akenine-Möller (2001) 13-axis SAT overlap test between triangle `tri`
/// and axis-aligned cube `[center - half_h, center + half_h]`.
fn tri_overlaps_voxel(tri: [[f64; 3]; 3], center: [f64; 3], half_h: f64) -> bool {
    let v0 = sub(tri[0], center);
    let v1 = sub(tri[1], center);
    let v2 = sub(tri[2], center);

    // 1. Box face normals (X, Y, Z).
    for k in 0..3 {
        let min_v = v0[k].min(v1[k]).min(v2[k]);
        let max_v = v0[k].max(v1[k]).max(v2[k]);
        if min_v > half_h || max_v < -half_h {
            return false;
        }
    }

    let f0 = sub(v1, v0);
    let f1 = sub(v2, v1);
    let f2 = sub(v0, v2);

    // 2. Triangle plane normal.
    let n = cross(f0, f1);
    let plane_r = half_h * (n[0].abs() + n[1].abs() + n[2].abs());
    if dot(n, v0).abs() > plane_r {
        return false;
    }

    // 3. Nine edge cross-product axes `e_i x f_j`.
    for f in [f0, f1, f2] {
        let [fx, fy, fz] = f;

        // e_x x f = [0, -fz, fy]
        let p0 = -fz * v0[1] + fy * v0[2];
        let p1 = -fz * v1[1] + fy * v1[2];
        let p2 = -fz * v2[1] + fy * v2[2];
        let r = half_h * (fz.abs() + fy.abs());
        if p0.min(p1).min(p2) > r || p0.max(p1).max(p2) < -r {
            return false;
        }

        // e_y x f = [fz, 0, -fx]
        let p0 = fz * v0[0] - fx * v0[2];
        let p1 = fz * v1[0] - fx * v1[2];
        let p2 = fz * v2[0] - fx * v2[2];
        let r = half_h * (fz.abs() + fx.abs());
        if p0.min(p1).min(p2) > r || p0.max(p1).max(p2) < -r {
            return false;
        }

        // e_z x f = [-fy, fx, 0]
        let p0 = -fy * v0[0] + fx * v0[1];
        let p1 = -fy * v1[0] + fx * v1[1];
        let p2 = -fy * v2[0] + fx * v2[1];
        let r = half_h * (fy.abs() + fx.abs());
        if p0.min(p1).min(p2) > r || p0.max(p1).max(p2) < -r {
            return false;
        }
    }

    true
}

/// Collects all live, finite, non-zero-area triangles of `mesh` in `f64`.
fn collect_proper_triangles(mesh: &Mesh) -> Result<Vec<[[f64; 3]; 3]>> {
    let mut tris = Vec::new();
    let positions = mesh.positions();
    for t in 0..mesh.triangle_slots() as TriangleId {
        if !mesh.is_triangle_alive(t) {
            continue;
        }
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        let corners_f32 = tri.map(|v| positions[v as usize]);
        if !corners_f32.iter().flatten().all(|c| c.is_finite()) {
            continue;
        }
        if triangle_normal(corners_f32) == [0.0; 3] {
            continue;
        }
        push(&mut tris, corners_f32.map(widen))?;
    }
    Ok(tris)
}

#[inline]
fn grid_point_f64(spec: &GridSpec, ix: usize, iy: usize, iz: usize) -> [f64; 3] {
    let h = f64::from(spec.voxel_size);
    [
        f64::from(spec.origin[0]) + (ix as f64) * h,
        f64::from(spec.origin[1]) + (iy as f64) * h,
        f64::from(spec.origin[2]) + (iz as f64) * h,
    ]
}

fn rasterize_unsigned_z_slice(
    spec: &GridSpec,
    tris: &[[[f64; 3]; 3]],
    tri_ids: &[u32],
    iz: usize,
    half_h: f64,
    slice_mask: &mut [bool],
) {
    let nx = spec.dims[0];
    for &ti in tri_ids {
        let tri = tris[ti as usize];
        let xmin = tri[0][0].min(tri[1][0]).min(tri[2][0]) - half_h;
        let xmax = tri[0][0].max(tri[1][0]).max(tri[2][0]) + half_h;
        let ymin = tri[0][1].min(tri[1][1]).min(tri[2][1]) - half_h;
        let ymax = tri[0][1].max(tri[1][1]).max(tri[2][1]) + half_h;
        let Some((ix0, ix1)) = axis_index_range(spec, 0, xmin, xmax) else {
            continue;
        };
        let Some((iy0, iy1)) = axis_index_range(spec, 1, ymin, ymax) else {
            continue;
        };
        for iy in iy0..=iy1 {
            for ix in ix0..=ix1 {
                let local = iy * nx + ix;
                if !slice_mask[local]
                    && tri_overlaps_voxel(tri, grid_point_f64(spec, ix, iy, iz), half_h)
                {
                    slice_mask[local] = true;
                }
            }
        }
    }
}

/// Unsigned surface distance grid via Akenine-Möller SAT triangle-voxel overlap (`[p - 0.5*h, p + 0.5*h]`)
/// over all live non-degenerate triangles of `mesh` + `edt_3d_mask`.
pub fn fh_unsigned_grid(mesh: &Mesh, spec: GridSpec, max_voxels: usize) -> Result<ScalarGrid> {
    let total = spec.validate(max_voxels)?;
    let tris = collect_proper_triangles(mesh)?;
    if tris.is_empty() {
        return ScalarGrid::new(spec, filled(total, f32::INFINITY)?, max_voxels);
    }

    let [nx, ny, nz] = spec.dims;
    let nxy = nx * ny;
    let h64 = f64::from(spec.voxel_size);
    let coord_scale = (0..3)
        .map(|k| {
            let lo = f64::from(spec.origin[k]).abs();
            let hi = (f64::from(spec.origin[k]) + ((spec.dims[k] - 1) as f64) * h64).abs();
            lo.max(hi)
        })
        .fold(1.0f64, f64::max);
    let half_h = 0.5 * h64 * (1.0 + 1e-6) + 2.0 * f64::from(f32::EPSILON) * coord_scale;

    let mut slice_tris: Vec<Vec<u32>> = filled(nz, Vec::new())?;
    for (ti, tri) in tris.iter().enumerate() {
        let zmin = tri[0][2].min(tri[1][2]).min(tri[2][2]) - half_h;
        let zmax = tri[0][2].max(tri[1][2]).max(tri[2][2]) + half_h;
        if let Some((iz0, iz1)) = axis_index_range(&spec, 2, zmin, zmax) {
            for bucket in &mut slice_tris[iz0..=iz1] {
                push(bucket, ti as u32)?;
            }
        }
    }

    let parallel = total >= PAR_MIN_EDT_VOXELS && par::enabled() && !par::force_sequential();
    let mask = if !parallel {
        let mut mask = filled(total, false)?;
        for (iz, tri_ids) in slice_tris.iter().enumerate() {
            rasterize_unsigned_z_slice(
                &spec,
                &tris,
                tri_ids,
                iz,
                half_h,
                &mut mask[iz * nxy..(iz + 1) * nxy],
            );
        }
        mask
    } else {
        let oom = || FieldsError::OutOfMemory;
        allocation_allowed()?;
        let parts = par::map_range(nz, 1, 0, oom, |r| {
            let iz = r.start;
            let mut slice_mask = filled(nxy, false)?;
            rasterize_unsigned_z_slice(&spec, &tris, &slice_tris[iz], iz, half_h, &mut slice_mask);
            Ok(slice_mask)
        })?;
        let mut mask = Vec::new();
        reserve_exact(&mut mask, total)?;
        for part in parts {
            mask.extend_from_slice(&part);
        }
        mask
    };

    drop(slice_tris);
    drop(tris);

    let mut out = filled(total, f32::INFINITY)?;
    edt_3d_mask(spec.dims, &mask, spec.voxel_size, &mut out)?;
    ScalarGrid::new(spec, out, max_voxels)
}

#[derive(Clone, Copy)]
struct OrientedZTri {
    p0: [f64; 3],
    p1: [f64; 3],
    p2: [f64; 3],
    area2: f64,
    z_min: f64,
    z_max: f64,
    ix0: usize,
    ix1: usize,
}

#[inline]
fn c2(p: [f64; 3]) -> Coord<f64> {
    Coord { x: p[0], y: p[1] }
}

/// Exact Simulation-of-Simplicity half-plane test for directed XY edge `u -> v` against
/// perturbed ray `R(eps) = (rx + eps^2, ry + eps)` (`0 < eps << 1`).
/// Returns `(is_left, raw_orient2d)`.
#[inline]
fn edge_left_sos(u: [f64; 3], v: [f64; 3], rx: f64, ry: f64) -> (bool, f64) {
    let d = orient2d(c2(u), c2(v), Coord { x: rx, y: ry });
    if d > 0.0 {
        return (true, d);
    }
    if d < 0.0 {
        return (false, d);
    }
    // Exact collinearity: evaluate +eps*(vx - ux) - eps^2*(vy - uy).
    if v[0] > u[0] {
        (true, 0.0)
    } else if v[0] < u[0] {
        (false, 0.0)
    } else {
        (v[1] < u[1], 0.0)
    }
}

/// Evaluates Z crossings and inside/outside classification for all `(ix, iz)` nodes at fixed `iy`.
/// Writes `nx * nz` booleans indexed by `iz * nx + ix` and returns the number of inside nodes.
fn classify_signed_y_row(
    spec: &GridSpec,
    z_tris: &[OrientedZTri],
    tri_ids: &[u32],
    iy: usize,
    row_inside: &mut [bool],
) -> Result<usize> {
    let nx = spec.dims[0];
    let nz = spec.dims[2];
    let ry = f64::from(spec.origin[1] + iy as f32 * spec.voxel_size);

    let mut col_crossings: Vec<Vec<f64>> = filled(nx, Vec::new())?;
    for &ti in tri_ids {
        let zt = &z_tris[ti as usize];
        for (ix, crossings) in col_crossings
            .iter_mut()
            .enumerate()
            .take(zt.ix1 + 1)
            .skip(zt.ix0)
        {
            let rx = f64::from(spec.origin[0] + ix as f32 * spec.voxel_size);
            let (ok01, d01) = edge_left_sos(zt.p0, zt.p1, rx, ry);
            if !ok01 {
                continue;
            }
            let (ok12, _) = edge_left_sos(zt.p1, zt.p2, rx, ry);
            if !ok12 {
                continue;
            }
            let (ok20, d20) = edge_left_sos(zt.p2, zt.p0, rx, ry);
            if !ok20 {
                continue;
            }
            let b1 = d20 / zt.area2;
            let b2 = d01 / zt.area2;
            let z_cross = (zt.p0[2] + b1 * (zt.p1[2] - zt.p0[2]) + b2 * (zt.p2[2] - zt.p0[2]))
                .clamp(zt.z_min, zt.z_max);
            push(crossings, z_cross)?;
        }
    }

    let mut inside_count = 0usize;
    for (ix, crossings) in col_crossings.iter_mut().enumerate() {
        if crossings.is_empty() {
            continue;
        }
        crossings.sort_unstable_by(f64::total_cmp);
        let mut cur = 0usize;
        for iz in 0..nz {
            let z_node = f64::from(spec.origin[2] + iz as f32 * spec.voxel_size);
            while cur < crossings.len() && crossings[cur] <= z_node {
                cur += 1;
            }
            if ((crossings.len() - cur) & 1) == 1 {
                row_inside[iz * nx + ix] = true;
                inside_count += 1;
            }
        }
    }
    Ok(inside_count)
}

/// Signed solid distance grid via Z-column ray-parity solid voxelization + dual `edt_3d_sq` passes:
/// - Classifies `(ix, iy, iz)` as inside/outside by shooting Z-parallel rays `(x0 + ix*h, y0 + iy*h, z)`
///   against live non-degenerate triangles using exact `robust::orient2d` with strict Simulation-of-Simplicity
///   half-open edge tie-breaking so rays hitting shared edges or vertices toggle parity cleanly once.
/// - Runs `edt_3d_sq` on inside-as-0 (`d_in_sq` for outside nodes) and outside-as-0 (`d_out_sq` for inside nodes):
///   outside nodes get `+h * (d_in_sq.sqrt() - 0.5)`, inside nodes get `-h * (d_out_sq.sqrt() - 0.5)`.
/// - If `mesh` has 0 triangles, fills `f32::INFINITY`. If `mesh` has triangles but occupies 0 grid nodes
///   (sub-voxel mesh between rays), falls back to `fh_unsigned_grid` so `d > 0` offsets remain non-empty.
pub fn fh_signed_grid(mesh: &Mesh, spec: GridSpec, max_voxels: usize) -> Result<ScalarGrid> {
    let total = spec.validate(max_voxels)?;
    if mesh.triangle_count() == 0 {
        return ScalarGrid::new(spec, filled(total, f32::INFINITY)?, max_voxels);
    }

    let tris = collect_proper_triangles(mesh)?;
    if tris.is_empty() {
        return ScalarGrid::new(spec, filled(total, f32::INFINITY)?, max_voxels);
    }

    let [nx, ny, nz] = spec.dims;
    let mut z_tris: Vec<OrientedZTri> = Vec::new();
    let mut row_tris: Vec<Vec<u32>> = filled(ny, Vec::new())?;

    for tri in &tris {
        let [a, b, c] = *tri;
        let raw = orient2d(c2(a), c2(b), c2(c));
        if raw == 0.0 {
            continue;
        }
        let (p0, p1, p2, area2) = if raw > 0.0 {
            (a, b, c, raw)
        } else {
            let pos_area = orient2d(c2(a), c2(c), c2(b));
            if pos_area <= 0.0 {
                continue;
            }
            (a, c, b, pos_area)
        };

        let xmin = p0[0].min(p1[0]).min(p2[0]);
        let xmax = p0[0].max(p1[0]).max(p2[0]);
        let ymin = p0[1].min(p1[1]).min(p2[1]);
        let ymax = p0[1].max(p1[1]).max(p2[1]);
        let Some((ix0, ix1)) = axis_index_range(&spec, 0, xmin, xmax) else {
            continue;
        };
        let Some((iy0, iy1)) = axis_index_range(&spec, 1, ymin, ymax) else {
            continue;
        };

        let z_min = p0[2].min(p1[2]).min(p2[2]);
        let z_max = p0[2].max(p1[2]).max(p2[2]);
        let ti = z_tris.len() as u32;
        push(
            &mut z_tris,
            OrientedZTri {
                p0,
                p1,
                p2,
                area2,
                z_min,
                z_max,
                ix0,
                ix1,
            },
        )?;
        for bucket in &mut row_tris[iy0..=iy1] {
            push(bucket, ti)?;
        }
    }
    drop(tris);

    if z_tris.is_empty() {
        return fh_unsigned_grid(mesh, spec, max_voxels);
    }

    let parallel = total >= PAR_MIN_EDT_VOXELS && par::enabled() && !par::force_sequential();
    let (inside, inside_count) = if !parallel {
        let mut inside = filled(total, false)?;
        let mut row_buf = filled(nx * nz, false)?;
        let mut count = 0usize;
        for (iy, tri_ids) in row_tris.iter().enumerate() {
            row_buf.fill(false);
            let c = classify_signed_y_row(&spec, &z_tris, tri_ids, iy, &mut row_buf)?;
            if c > 0 {
                count += c;
                for iz in 0..nz {
                    let dst = (iz * ny + iy) * nx;
                    let src = iz * nx;
                    inside[dst..dst + nx].copy_from_slice(&row_buf[src..src + nx]);
                }
            }
        }
        (inside, count)
    } else {
        let oom = || FieldsError::OutOfMemory;
        allocation_allowed()?;
        let parts = par::map_range(ny, 1, 0, oom, |r| {
            let iy = r.start;
            let mut row_buf = filled(nx * nz, false)?;
            let c = classify_signed_y_row(&spec, &z_tris, &row_tris[iy], iy, &mut row_buf)?;
            Ok((row_buf, c))
        })?;
        let mut inside = filled(total, false)?;
        let mut count = 0usize;
        for (iy, (row_buf, c)) in parts.into_iter().enumerate() {
            if c > 0 {
                count += c;
                for iz in 0..nz {
                    let dst = (iz * ny + iy) * nx;
                    let src = iz * nx;
                    inside[dst..dst + nx].copy_from_slice(&row_buf[src..src + nx]);
                }
            }
        }
        (inside, count)
    };

    drop(row_tris);
    drop(z_tris);

    if inside_count == 0 {
        return fh_unsigned_grid(mesh, spec, max_voxels);
    }

    let mut cost_in = Vec::new();
    reserve_exact(&mut cost_in, total)?;
    for &b in &inside {
        cost_in.push(if b { 0.0 } else { f32::INFINITY });
    }
    let mut d_in_sq = filled(total, 0.0f32)?;
    edt_3d_sq(spec.dims, &cost_in, &mut d_in_sq)?;
    drop(cost_in);

    let mut cost_out = Vec::new();
    reserve_exact(&mut cost_out, total)?;
    for &b in &inside {
        cost_out.push(if b { f32::INFINITY } else { 0.0 });
    }
    let mut d_out_sq = filled(total, 0.0f32)?;
    edt_3d_sq(spec.dims, &cost_out, &mut d_out_sq)?;
    drop(cost_out);

    let h = spec.voxel_size;
    for i in 0..total {
        d_in_sq[i] = if inside[i] {
            -h * (d_out_sq[i].sqrt() - 0.5)
        } else {
            h * (d_in_sq[i].sqrt() - 0.5)
        };
    }

    ScalarGrid::new(spec, d_in_sq, max_voxels)
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::{Mesh, par};

    use super::*;
    use crate::grid::fail_allocations_after;
    use crate::test_util::{Lcg, brute_distance, build_bvh, cube, icosphere, mesh, terrain};
    use crate::{DEFAULT_MAX_VOXELS, SignData, SignMode, bvh_signed_grid, bvh_unsigned_grid};

    #[test]
    fn edt_all_infinite_and_single_site_exact() {
        let dims = [5, 6, 7];
        let total = 5 * 6 * 7;

        // All infinite or >= 1e30 -> all INFINITY, zero NaNs.
        let mut cost = vec![f32::INFINITY; total];
        cost[10] = 1e30;
        cost[25] = 2e30;
        let mut out_sq = vec![0.0f32; total];
        edt_3d_sq(dims, &cost, &mut out_sq).unwrap();
        assert!(out_sq.iter().all(|&v| v == f32::INFINITY));

        let mask = vec![false; total];
        let mut out = vec![0.0f32; total];
        edt_3d_mask(dims, &mask, 0.25, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == f32::INFINITY));

        // Single site at (2, 3, 4) with cost 1.5.
        let idx_of = |x: usize, y: usize, z: usize| (z * dims[1] + y) * dims[0] + x;
        let mut single = vec![f32::INFINITY; total];
        single[idx_of(2, 3, 4)] = 1.5;
        edt_3d_sq(dims, &single, &mut out_sq).unwrap();
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let dx = x as isize - 2;
                    let dy = y as isize - 3;
                    let dz = z as isize - 4;
                    let want = (dx * dx + dy * dy + dz * dz) as f32 + 1.5;
                    let got = out_sq[idx_of(x, y, z)];
                    assert_eq!(got, want, "at ({x}, {y}, {z})");
                }
            }
        }

        // Multi-site random weighted EDT matches brute-force lower envelope everywhere.
        let mut rng = Lcg(99);
        let mut multi = vec![f32::INFINITY; total];
        let mut sites = Vec::new();
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    if rng.next() < 0.12 {
                        let c = ((rng.next() * 8.0).floor()) * 0.25;
                        multi[idx_of(x, y, z)] = c;
                        sites.push((x as isize, y as isize, z as isize, c));
                    }
                }
            }
        }
        assert!(sites.len() >= 10);
        edt_3d_sq(dims, &multi, &mut out_sq).unwrap();
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let want = sites
                        .iter()
                        .map(|&(sx, sy, sz, c)| {
                            let dx = x as isize - sx;
                            let dy = y as isize - sy;
                            let dz = z as isize - sz;
                            (dx * dx + dy * dy + dz * dz) as f32 + c
                        })
                        .fold(f32::INFINITY, f32::min);
                    assert_eq!(out_sq[idx_of(x, y, z)], want);
                }
            }
        }

        // Degenerate 1D / 2D grid dimensions (including >= PAR_MIN_EDT_VOXELS) work cleanly.
        let mut one = [99.0f32];
        edt_3d_sq([1, 1, 1], &[2.5], &mut one).unwrap();
        assert_eq!(one, [2.5]);
        for thin_dims in [[20_000, 1, 1], [1, 20_000, 1], [1, 1, 20_000]] {
            let mut thin_cost = vec![f32::INFINITY; 20_000];
            thin_cost[5_000] = 0.0;
            let mut thin_out = vec![0.0f32; 20_000];
            edt_3d_sq(thin_dims, &thin_cost, &mut thin_out).unwrap();
            assert_eq!(thin_out[5_000], 0.0);
            assert_eq!(thin_out[5_010], 100.0);
            assert_eq!(thin_out[4_995], 25.0);
        }

        // Validation rejects 0 dims, mismatched slices, NaN, negative costs, and bad voxel_size.
        let sentinel = vec![42.0f32; total];
        let mut check_out = sentinel.clone();
        assert!(matches!(
            edt_3d_sq([0, 6, 7], &[], &mut []),
            Err(FieldsError::InvalidInput(_))
        ));
        assert!(matches!(
            edt_3d_sq(dims, &single[..total - 1], &mut check_out),
            Err(FieldsError::InvalidInput(_))
        ));
        single[0] = f32::NAN;
        assert!(matches!(
            edt_3d_sq(dims, &single, &mut check_out),
            Err(FieldsError::InvalidInput(_))
        ));
        single[0] = -0.25;
        assert!(matches!(
            edt_3d_sq(dims, &single, &mut check_out),
            Err(FieldsError::InvalidInput(_))
        ));
        single[0] = f32::NEG_INFINITY;
        assert!(matches!(
            edt_3d_sq(dims, &single, &mut check_out),
            Err(FieldsError::InvalidInput(_))
        ));
        assert!(matches!(
            edt_3d_mask(dims, &mask, 0.0, &mut check_out),
            Err(FieldsError::InvalidInput(_))
        ));
        assert!(matches!(
            edt_3d_mask(dims, &mask, f32::INFINITY, &mut check_out),
            Err(FieldsError::InvalidInput(_))
        ));
        assert_eq!(check_out, sentinel);
    }

    /// Subdivides every triangle into 4 so vertices on faces/edges have valence 6+ and
    /// rays hit horizontal, vertical, and diagonal shared edges simultaneously.
    fn subdivided_cube() -> Mesh {
        let (mut pos, mut tris) = crate::test_util::cube_buffers();
        for _ in 0..2 {
            let mut edges: Vec<((u32, u32), u32)> = Vec::new();
            let mut mid = |a: u32, b: u32, pos: &mut Vec<[f32; 3]>| -> u32 {
                let key = (a.min(b), a.max(b));
                if let Some(&(_, id)) = edges.iter().find(|(k, _)| *k == key) {
                    return id;
                }
                let (pa, pb) = (pos[a as usize], pos[b as usize]);
                pos.push(std::array::from_fn(|k| 0.5 * (pa[k] + pb[k])));
                let id = (pos.len() - 1) as u32;
                edges.push((key, id));
                id
            };
            let mut next = Vec::new();
            for [a, b, c] in tris {
                let ab = mid(a, b, &mut pos);
                let bc = mid(b, c, &mut pos);
                let ca = mid(c, a, &mut pos);
                next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
            }
            tris = next;
        }
        mesh(&pos, &tris)
    }

    #[test]
    fn fh_signed_grid_grazing_rays_on_grid_aligned_cube() {
        for c in [cube(), subdivided_cube()] {
            // Grid where rays hit x = ±1, y = ±1, z = ±1, corners, midpoints, and diagonals!
            let spec = GridSpec {
                origin: [-1.5, -1.5, -1.5],
                dims: [13, 13, 13],
                voxel_size: 0.25,
            };
            let grid = fh_signed_grid(&c, spec, DEFAULT_MAX_VOXELS).unwrap();

            for iz in 0..spec.dims[2] {
                for iy in 0..spec.dims[1] {
                    for ix in 0..spec.dims[0] {
                        let p = spec.point(ix, iy, iz);
                        let v = grid.values()[spec.index(ix, iy, iz)];
                        assert!(v.is_finite());
                        // Half-open SoS membership: [-1.0, 1.0)^3 has 8x8x8 = 512 inside nodes!
                        let want_inside = (-1.0..1.0).contains(&p[0])
                            && (-1.0..1.0).contains(&p[1])
                            && (-1.0..1.0).contains(&p[2]);
                        assert_eq!(
                            v < 0.0,
                            want_inside,
                            "wrong classification at ({ix}, {iy}, {iz}) {p:?}: {v}"
                        );
                    }
                }
            }

            // Center is deep inside (~ -1.0), exterior is positive, half-voxel zero crossing near faces.
            let center_val = grid.sample([0.0, 0.0, 0.0]).unwrap();
            assert!((center_val - (-1.0)).abs() <= 0.26, "center: {center_val}");
            assert!(grid.sample([-1.4, 0.0, 0.0]).unwrap() > 0.2);
            // Between x = -1.25 (+0.5*h) and x = -1.0 (-0.5*h), zero-crossing sits at x = -1.125 (within 0.5*h of -1.0).
            assert!(grid.sample([-1.125, 0.0, 0.0]).unwrap().abs() < 1e-5);
        }

        // Empty mesh -> all INFINITY.
        let spec = GridSpec {
            origin: [-1.5, -1.5, -1.5],
            dims: [6, 6, 6],
            voxel_size: 0.5,
        };
        let empty = Mesh::new();
        let eg = fh_signed_grid(&empty, spec, DEFAULT_MAX_VOXELS).unwrap();
        assert!(eg.values().iter().all(|&v| v == f32::INFINITY));

        // Sub-voxel closed cube strictly between rays -> falls back to fh_unsigned_grid.
        let (mut pos, tris) = crate::test_util::cube_buffers();
        for p in &mut pos {
            *p = [p[0] * 0.03 + 0.12, p[1] * 0.03 + 0.12, p[2] * 0.03 + 0.12];
        }
        let tiny = mesh(&pos, &tris);
        let coarse = GridSpec {
            origin: [-1.0, -1.0, -1.0],
            dims: [6, 6, 6],
            voxel_size: 0.5,
        };
        let tg = fh_signed_grid(&tiny, coarse, DEFAULT_MAX_VOXELS).unwrap();
        let ug = fh_unsigned_grid(&tiny, coarse, DEFAULT_MAX_VOXELS).unwrap();
        assert_eq!(tg.values(), ug.values());
        assert!(tg.values().iter().all(|&v| v.is_finite() && v >= 0.0));
        assert!(tg.values().contains(&0.0));
    }

    #[test]
    fn fh_unsigned_grid_on_open_terrain_sheet() {
        let sheet = terrain(8);
        let h = 0.25f32;
        let spec = GridSpec::around_bounds(
            [0.0, 0.0, -1.0],
            [8.0, 8.0, 1.0],
            h,
            0.0,
            2,
            DEFAULT_MAX_VOXELS,
        )
        .unwrap();
        let grid = fh_unsigned_grid(&sheet, spec, DEFAULT_MAX_VOXELS).unwrap();
        let bound = h * 3.0f32.sqrt() + 1e-4;
        for iz in 0..spec.dims[2] {
            for iy in 0..spec.dims[1] {
                for ix in 0..spec.dims[0] {
                    let p = spec.point(ix, iy, iz);
                    let got = grid.values()[spec.index(ix, iy, iz)];
                    assert!(got.is_finite() && got >= 0.0);
                    let exact = brute_distance(&sheet, p) as f32;
                    assert!(
                        (got - exact).abs() <= bound,
                        "at {p:?}: got {got} vs exact {exact} (bound {bound})"
                    );
                }
            }
        }
    }

    #[test]
    fn edt_and_grids_parallel_bit_identical() {
        let mut sphere = icosphere();
        let bvh = build_bvh(&sphere);
        let sign = SignData::build(&mut sphere, &bvh, SignMode::Auto).unwrap();
        let spec = GridSpec {
            origin: [-1.4, -1.4, -1.4],
            dims: [36, 36, 36],
            voxel_size: 2.8 / 35.0,
        };
        let total = 36 * 36 * 36;
        let mut cost = vec![f32::INFINITY; total];
        for i in (0..total).step_by(97) {
            cost[i] = (i % 7) as f32 * 0.25;
        }
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();

        let run_all = || {
            let mut edt_out = vec![0.0f32; total];
            edt_3d_sq(spec.dims, &cost, &mut edt_out).unwrap();
            let fhs = fh_signed_grid(&sphere, spec, DEFAULT_MAX_VOXELS).unwrap();
            let fhu = fh_unsigned_grid(&sphere, spec, DEFAULT_MAX_VOXELS).unwrap();
            let bvhs =
                bvh_signed_grid(&sphere, &bvh, &sign, spec, 0.35, DEFAULT_MAX_VOXELS).unwrap();
            let bvhu = bvh_unsigned_grid(&sphere, &bvh, spec, 0.35, DEFAULT_MAX_VOXELS).unwrap();
            (
                bits(&edt_out),
                bits(fhs.values()),
                bits(fhu.values()),
                bits(bvhs.values()),
                bits(bvhu.values()),
            )
        };

        par::set_force_sequential(true);
        let reference = run_all();
        par::set_force_sequential(false);
        if par::enabled() {
            for k in [1, 2, 8] {
                let got = par::with_threads(k, run_all);
                assert!(got == reference, "mismatch on {k} threads");
            }
            // Verify each of the 5 kernels enters the parallel pool on 36^3.
            let mut edt_out = vec![0.0f32; total];
            let b0 = par::parallel_entries();
            edt_3d_sq(spec.dims, &cost, &mut edt_out).unwrap();
            let b1 = par::parallel_entries();
            assert!(b1 > b0, "edt_3d_sq did not parallelize");
            fh_signed_grid(&sphere, spec, DEFAULT_MAX_VOXELS).unwrap();
            let b2 = par::parallel_entries();
            assert!(b2 > b1, "fh_signed_grid did not parallelize");
            fh_unsigned_grid(&sphere, spec, DEFAULT_MAX_VOXELS).unwrap();
            let b3 = par::parallel_entries();
            assert!(b3 > b2, "fh_unsigned_grid did not parallelize");
            bvh_signed_grid(&sphere, &bvh, &sign, spec, 0.35, DEFAULT_MAX_VOXELS).unwrap();
            let b4 = par::parallel_entries();
            assert!(b4 > b3, "bvh_signed_grid did not parallelize");
            bvh_unsigned_grid(&sphere, &bvh, spec, 0.35, DEFAULT_MAX_VOXELS).unwrap();
            let b5 = par::parallel_entries();
            assert!(b5 > b4, "bvh_unsigned_grid did not parallelize");

            // Small grids stay sequential.
            let small = GridSpec {
                origin: [-1.5; 3],
                dims: [4, 4, 4],
                voxel_size: 1.0,
            };
            let b_small = par::parallel_entries();
            let mut small_out = [0.0f32; 64];
            edt_3d_sq([4, 4, 4], &[0.0f32; 64], &mut small_out).unwrap();
            fh_signed_grid(&sphere, small, DEFAULT_MAX_VOXELS).unwrap();
            fh_unsigned_grid(&sphere, small, DEFAULT_MAX_VOXELS).unwrap();
            bvh_signed_grid(&sphere, &bvh, &sign, small, 2.5, DEFAULT_MAX_VOXELS).unwrap();
            bvh_unsigned_grid(&sphere, &bvh, small, 2.5, DEFAULT_MAX_VOXELS).unwrap();
            assert_eq!(par::parallel_entries(), b_small);
        }
    }

    #[test]
    fn oom_fault_injection_returns_out_of_memory() {
        struct ResetGuard;
        impl Drop for ResetGuard {
            fn drop(&mut self) {
                fail_allocations_after(None);
            }
        }
        let _guard = ResetGuard;

        let mut c = cube();
        let bvh = build_bvh(&c);
        let sign = SignData::build(&mut c, &bvh, SignMode::Auto).unwrap();
        let spec = GridSpec {
            origin: [-1.5, -1.5, -1.5],
            dims: [7, 7, 7],
            voxel_size: 0.5,
        };
        let total = 7 * 7 * 7;
        let mut mask = vec![false; total];
        mask[total / 2] = true;

        fn exhaust<T>(mut f: impl FnMut() -> Result<T>) {
            let mut saw_oom = false;
            for n in 0..500 {
                fail_allocations_after(Some(n));
                match f() {
                    Err(FieldsError::OutOfMemory) => saw_oom = true,
                    Ok(_) => {
                        fail_allocations_after(None);
                        assert!(saw_oom, "never hit OutOfMemory before succeeding at n={n}");
                        return;
                    }
                    Err(other) => panic!("unexpected error at n={n}: {other:?}"),
                }
            }
            panic!("did not succeed within 500 allocation steps");
        }

        exhaust(|| {
            let mut out = vec![0.0f32; total];
            edt_3d_mask(spec.dims, &mask, 0.5, &mut out)
        });
        exhaust(|| fh_unsigned_grid(&c, spec, DEFAULT_MAX_VOXELS));
        exhaust(|| fh_signed_grid(&c, spec, DEFAULT_MAX_VOXELS));
        exhaust(|| bvh_unsigned_grid(&c, &bvh, spec, 1.0, DEFAULT_MAX_VOXELS));
        exhaust(|| bvh_signed_grid(&c, &bvh, &sign, spec, 1.0, DEFAULT_MAX_VOXELS));
    }

    #[test]
    fn fh_unsigned_grid_captures_flat_sheet_on_half_voxel_boundary() {
        // Even nz (12) places z = 0.0 right on the half-voxel boundary between iz = 5 and iz = 6.
        let sheet = Mesh::from_buffers(
            &[0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 4.0, 0.0],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap();
        let spec = GridSpec::around_bounds(
            [0.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
            0.12,
            0.25,
            3,
            DEFAULT_MAX_VOXELS,
        )
        .unwrap();
        assert_eq!(spec.dims[2], 12);
        let grid = fh_unsigned_grid(&sheet, spec, DEFAULT_MAX_VOXELS).unwrap();
        assert!(grid.values().iter().all(|v| v.is_finite()));
        let mid_x = spec.dims[0] / 2;
        let mid_y = spec.dims[1] / 2;
        assert_eq!(grid.values()[spec.index(mid_x, mid_y, 5)], 0.0);
        assert_eq!(grid.values()[spec.index(mid_x, mid_y, 6)], 0.0);
    }
}
