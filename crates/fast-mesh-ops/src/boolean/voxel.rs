//! Volumetric CSG boolean evaluation via [`BooleanMethod::BvhSdf`] and [`BooleanMethod::FhEdt`].
//!
//! Uses `fast_mesh_fields` signed grids and Marching Cubes (`iso = 0.0`) to evaluate
//! watertight 2-manifold booleans even when input meshes are open, self-intersecting,
//! non-manifold, or inconsistently oriented.

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_fields::{
    DEFAULT_AUTO_GRID_RES, DEFAULT_MAX_VOXELS, FieldsError, GridSpec, ScalarGrid, SignData,
    SignMode, bvh_signed_grid, fh_signed_grid,
};
use fast_mesh_query::Rigid;
use fast_mesh_store::{Mesh, StoreError, TriangleId, VertexId};

use super::{BooleanMethod, BooleanOp, BooleanOptions, BooleanReport};
use crate::cleanup::{DegenerateOptions, DegenerateReport, remove_degenerate};
use crate::marching_cubes::{IsoMesh, marching_cubes};
use crate::{OpsError, Result, allocation_allowed, filled, push};

/// Transforms all proper alive triangles of `mesh` by `xf` into a compact mesh.
///
/// Dead vertex/triangle slots, unused vertices, and repeated-corner triangles
/// are skipped without failing, so dirty meshes flow straight into volumetric
/// reconstruction.
pub(super) fn place_mesh_compact(mesh: &Mesh, xf: &Rigid) -> Result<Mesh> {
    xf.validate()?;
    let positions = mesh.positions();
    let mut remap: Vec<u32> = filled(mesh.vertex_slots(), u32::MAX)?;
    let mut out_positions: Vec<[f32; 3]> = Vec::new();
    let mut out_tris: Vec<[u32; 3]> = Vec::new();

    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some([ra, rb, rc]) = mesh.triangle(t) else {
            continue;
        };
        if ra == rb || rb == rc || ra == rc {
            continue;
        }
        let mut mapped = [0u32; 3];
        for (k, rv) in [ra, rb, rc].into_iter().enumerate() {
            let idx = if remap[rv as usize] != u32::MAX {
                remap[rv as usize]
            } else {
                let fresh = u32::try_from(out_positions.len())
                    .map_err(|_| OpsError::Store(StoreError::OutOfMemory))?;
                let p = positions[rv as usize];
                let q = xf.apply(p);
                let f = [q[0] as f32, q[1] as f32, q[2] as f32];
                if !f.iter().all(|c| c.is_finite()) {
                    return Err(OpsError::InvalidInput("boolean point leaves f32 range"));
                }
                push(&mut out_positions, f)?;
                remap[rv as usize] = fresh;
                fresh
            };
            mapped[k] = idx;
        }
        push(&mut out_tris, mapped)?;
    }

    let mut out = Mesh::new();
    if !out_tris.is_empty() {
        let mut tx = out.begin_edit();
        tx.add_vertices(&out_positions)?;
        tx.add_triangles(&out_tris)?;
        tx.commit()?;
    }
    Ok(out)
}

/// Computes the world bounding box of all vertices used by live triangles in `mesh`.
pub(super) fn live_bounds(mesh: &Mesh) -> Option<([f32; 3], [f32; 3])> {
    let positions = mesh.positions();
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut any = false;

    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some([a, b, c]) = mesh.triangle(t) else {
            continue;
        };
        for v in [a, b, c] {
            let p = positions[v as usize];
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
            any = true;
        }
    }

    any.then_some((min, max))
}

/// Validates and selects a padded [`GridSpec`] enclosing `[min, max]`.
pub(super) fn choose_boolean_grid_spec(
    min: [f32; 3],
    max: [f32; 3],
    voxel_pitch: Option<f32>,
) -> Result<GridSpec> {
    if let Some(h_user) = voxel_pitch {
        if !h_user.is_finite() || h_user <= 0.0 {
            return Err(OpsError::InvalidInput(
                "voxel_pitch must be finite and positive",
            ));
        }
        return Ok(GridSpec::around_bounds(
            min,
            max,
            h_user,
            0.0,
            3,
            DEFAULT_MAX_VOXELS,
        )?);
    }

    let dx = f64::from(max[0]) - f64::from(min[0]);
    let dy = f64::from(max[1]) - f64::from(min[1]);
    let dz = f64::from(max[2]) - f64::from(min[2]);
    let diag = (dx * dx + dy * dy + dz * dz).sqrt();
    let pad_voxels = 3usize;
    let mut h = if diag > 0.0 {
        ((diag / DEFAULT_AUTO_GRID_RES as f64) as f32).max(1e-4)
    } else {
        1e-3
    };

    let max_base = dx.max(dy).max(dz).max(0.0);
    let axis_budget = (DEFAULT_MAX_VOXELS as f64).cbrt() - (2 * pad_voxels + 1) as f64;
    if axis_budget > 0.0 && max_base > 0.0 {
        let h_lower = ((max_base / axis_budget) * 1.01) as f32;
        if h_lower.is_finite() && h_lower > h {
            h = h_lower;
        }
    }

    for _ in 0..32 {
        match GridSpec::around_bounds(min, max, h, 0.0, pad_voxels, DEFAULT_MAX_VOXELS) {
            Ok(spec) => return Ok(spec),
            Err(FieldsError::InvalidInput("grid exceeds max_voxels")) => {
                let next = h * 1.25;
                if !next.is_finite() {
                    break;
                }
                h = next;
            }
            Err(e) => return Err(e.into()),
        }
    }

    Err(OpsError::InvalidInput(
        "could not choose a grid within DEFAULT_MAX_VOXELS",
    ))
}

/// Evaluates the signed distance field of compact mesh `placed` on `spec`.
///
/// `BvhSdf` uses generalized winding signs (`SignMode::Winding`) so open cavities,
/// inconsistent face orientations, and self-intersections retain well-defined interiors.
/// `FhEdt` computes the Felzenszwalb-Huttenlocher exact Euclidean distance transform.
pub(super) fn eval_signed_grid(
    placed: &mut Mesh,
    spec: GridSpec,
    method: BooleanMethod,
) -> Result<ScalarGrid> {
    let total = spec.validate(DEFAULT_MAX_VOXELS)?;
    if placed.triangle_count() == 0 {
        return Ok(ScalarGrid::new(
            spec,
            filled(total, 4.0 * spec.voxel_size)?,
            DEFAULT_MAX_VOXELS,
        )?);
    }

    match method {
        BooleanMethod::BvhSdf => {
            let bvh = Bvh::build(placed, &BuildOptions::default())?;
            let sign = SignData::build(placed, &bvh, SignMode::Winding)?;
            let narrow_band = 4.0 * spec.voxel_size;
            Ok(bvh_signed_grid(
                placed,
                &bvh,
                &sign,
                spec,
                narrow_band,
                DEFAULT_MAX_VOXELS,
            )?)
        }
        BooleanMethod::FhEdt => Ok(fh_signed_grid(placed, spec, DEFAULT_MAX_VOXELS)?),
        BooleanMethod::Manifold => Err(OpsError::InvalidInput(
            "eval_signed_grid requires volumetric BooleanMethod",
        )),
    }
}

/// Combines `lhs` and `rhs` grids voxel-by-voxel in place according to `op`.
pub(super) fn combine_grids_in_place(lhs: &mut [f32], rhs: &[f32], op: BooleanOp) {
    debug_assert_eq!(lhs.len(), rhs.len());
    match op {
        BooleanOp::Union => {
            for (a, &b) in lhs.iter_mut().zip(rhs.iter()) {
                *a = a.min(b);
            }
        }
        BooleanOp::Intersect => {
            for (a, &b) in lhs.iter_mut().zip(rhs.iter()) {
                *a = a.max(b);
            }
        }
        BooleanOp::Subtract => {
            for (a, &b) in lhs.iter_mut().zip(rhs.iter()) {
                *a = a.max(-b);
            }
        }
    }
}

/// Forces the outermost 1-voxel shell to be strictly positive so Marching Cubes
/// produces a closed, boundary-free 2-manifold even when ray parity on an open mesh
/// reaches a grid boundary.
pub(super) fn clamp_outer_border_positive(spec: GridSpec, values: &mut [f32]) {
    let [nx, ny, nz] = spec.dims;
    let border_floor = 0.5 * spec.voxel_size;
    for iz in 0..nz {
        let z_edge = iz == 0 || iz + 1 == nz;
        for iy in 0..ny {
            let y_edge = iy == 0 || iy + 1 == ny;
            let row = (iz * ny + iy) * nx;
            if z_edge || y_edge {
                for v in &mut values[row..row + nx] {
                    *v = v.max(border_floor);
                }
            } else {
                values[row] = values[row].max(border_floor);
                values[row + nx - 1] = values[row + nx - 1].max(border_floor);
            }
        }
    }
}

/// Extracts and cleans the zero-level isosurface from `values` on `spec`.
pub(super) fn extract_isosurface(spec: GridSpec, mut values: Vec<f32>) -> Result<Mesh> {
    clamp_outer_border_positive(spec, &mut values);
    let grid = ScalarGrid::new(spec, values, DEFAULT_MAX_VOXELS)?;
    let iso = marching_cubes(&grid, 0.0)?;
    drop(grid);
    if iso.triangles.is_empty() {
        return Ok(Mesh::new());
    }
    let (cleaned, _) = clean_iso_mesh(iso)?;
    if cleaned.triangles.is_empty() {
        return Ok(Mesh::new());
    }
    let mut out = Mesh::new();
    {
        let mut tx = out.begin_edit();
        tx.add_vertices(&cleaned.positions)?;
        tx.add_triangles(&cleaned.triangles)?;
        tx.commit()?;
    }
    Ok(out)
}

fn clean_iso_mesh(iso: IsoMesh) -> Result<(IsoMesh, DegenerateReport)> {
    if iso.triangles.is_empty() {
        return Ok((
            IsoMesh::default(),
            DegenerateReport {
                removed_triangles: 0,
                removed_vertices: iso.positions.len(),
            },
        ));
    }
    allocation_allowed()?;
    let mut temp = Mesh::new();
    let cleanup = {
        let mut ttx = temp.begin_edit();
        if !iso.positions.is_empty() {
            ttx.add_vertices(&iso.positions)?;
        }
        ttx.add_triangles(&iso.triangles)?;
        let report = remove_degenerate(&mut ttx, &DegenerateOptions::default())?;
        ttx.commit()?;
        report
    };
    if cleanup.removed_triangles == 0 && cleanup.removed_vertices == 0 {
        return Ok((iso, cleanup));
    }
    let mut remap: Vec<u32> = filled(temp.vertex_slots(), u32::MAX)?;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    for (v, &p) in temp.positions().iter().enumerate() {
        if temp.is_vertex_alive(v as VertexId) {
            let idx = positions.len() as u32;
            push(&mut positions, p)?;
            remap[v] = idx;
        }
    }
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    for t in 0..temp.triangle_slots() as TriangleId {
        if let Some([a, b, c]) = temp.triangle(t) {
            push(
                &mut triangles,
                [remap[a as usize], remap[b as usize], remap[c as usize]],
            )?;
        }
    }
    Ok((
        IsoMesh {
            positions,
            triangles,
        },
        cleanup,
    ))
}

/// Evaluates a volumetric CSG boolean (`BvhSdf` or `FhEdt`) between `mesh_a` and `mesh_b`.
pub(super) fn boolean_voxel(
    mesh_a: &Mesh,
    mesh_b: &Mesh,
    b_in_a: &Rigid,
    options: &BooleanOptions,
) -> Result<(Mesh, BooleanReport)> {
    let mut placed_a = place_mesh_compact(mesh_a, &Rigid::IDENTITY)?;
    let mut placed_b = place_mesh_compact(mesh_b, b_in_a)?;

    let bounds_a = live_bounds(&placed_a);
    let bounds_b = live_bounds(&placed_b);

    let (min, max) = match (bounds_a, bounds_b, options.op) {
        (None, None, _)
        | (None, Some(_), BooleanOp::Subtract | BooleanOp::Intersect)
        | (Some(_), None, BooleanOp::Intersect) => {
            if let Some(h) = options.voxel_pitch
                && (!h.is_finite() || h <= 0.0)
            {
                return Err(OpsError::InvalidInput(
                    "voxel_pitch must be finite and positive",
                ));
            }
            return Ok((Mesh::new(), BooleanReport::default()));
        }
        (Some(ba), None, _) => ba,
        (None, Some(bb), BooleanOp::Union) => bb,
        (Some((amin, amax)), Some((bmin, bmax)), _) => (
            [
                amin[0].min(bmin[0]),
                amin[1].min(bmin[1]),
                amin[2].min(bmin[2]),
            ],
            [
                amax[0].max(bmax[0]),
                amax[1].max(bmax[1]),
                amax[2].max(bmax[2]),
            ],
        ),
    };

    let spec = choose_boolean_grid_spec(min, max, options.voxel_pitch)?;
    let mut grid_a = eval_signed_grid(&mut placed_a, spec, options.method)?.into_values();
    let grid_b = eval_signed_grid(&mut placed_b, spec, options.method)?.into_values();
    combine_grids_in_place(&mut grid_a, &grid_b, options.op);
    drop(grid_b);

    let out = extract_isosurface(spec, grid_a)?;
    let count = out.triangle_count();
    Ok((
        out,
        BooleanReport {
            kept_triangles: count,
            triangles_added: count,
            uvs_dropped: mesh_a.uvs().is_some(),
        },
    ))
}
