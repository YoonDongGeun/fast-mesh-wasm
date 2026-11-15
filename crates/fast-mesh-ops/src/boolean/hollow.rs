//! SDF inward hollowing and chamfered/tapered drain holes for closed 2-manifold solids.
//!
//! [`hollow_mesh`] computes a signed distance field (`BvhSdf` or `FhEdt`) for a
//! closed, outward-wound solid, extracts the `-wall_thickness` level-set (with
//! optional per-vertex thickness modulation), flips the inner cavity winding so
//! normals point inward into the void, and appends the cavity cleanly inside one
//! undoable [`Transaction`].
//!
//! [`punch_drain_hole`] and [`punch_drain_holes`] build watertight, outward-wound
//! cylindrical / 45°-chamfered / conical-tapered hole cutters (plus optional
//! clearance-fitted frustum plug meshes) and CSG-subtract them from `tx`,
//! bridging the outer wall and inner cavity into a single continuous 2-manifold.

use std::f64::consts::TAU;

use fast_mesh_bvh::Bvh;
use fast_mesh_fields::{
    DEFAULT_AUTO_GRID_RES, DEFAULT_MAX_VOXELS, FieldsError, GridSpec, ScalarGrid, SignData,
    SignMode, bvh_signed_grid, fh_signed_grid,
};
use fast_mesh_query::{
    ClosestHit, ClosestOptions, Cull, Ray, RayHit, RayOptions, Rigid, closest_point_many,
    raycast_all,
};
use fast_mesh_store::{Mesh, StoreError, Transaction, TriangleId, VertexId};

use super::batch::{CsgNode, boolean_batch};
use super::winding;
use super::{
    BooleanMethod, BooleanOp, BooleanOptions, apply_boolean_with_options, boolean,
    clone_mesh_fallible, replace_tx_with_mesh,
};
use crate::append::{AppendSource, append_mesh};
use crate::offset::{
    build_work_bvh, choose_grid_spec, classify_solid, clean_iso_mesh, compact_work_mesh,
};
use crate::topology::{cross, dot};
use crate::{
    OpsError, Result, SimplifyOptions, allocation_allowed, filled, marching_cubes, push, reserve,
    simplify,
};

/// SDF grid backend used by [`hollow_mesh`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum HollowBackend {
    /// Narrow-band exact BVH closest-triangle signed distance grid.
    #[default]
    BvhSdf,
    /// Exact 3D Felzenszwalb-Huttenlocher Euclidean Distance Transform signed grid.
    FhEdt,
}

/// Configuration options for [`hollow_mesh`].
#[derive(Debug, Clone, PartialEq)]
pub struct HollowOptions {
    /// Target minimum shell wall thickness in local units (must be finite and `> 0.0`).
    pub wall_thickness: f32,
    /// Signed distance field evaluation backend.
    pub backend: HollowBackend,
    /// Explicit voxel grid pitch in local units; `None` chooses automatically within `max_voxels`.
    pub voxel_size: Option<f32>,
    /// Maximum voxel count allowed for the SDF evaluation grid (must be `>= 64`).
    pub max_voxels: usize,
    /// Optional upper bound on extracted cavity triangle count via quadric simplification.
    pub max_cavity_triangles: Option<u32>,
    /// Minimum cavity volume required to keep an extracted void (must be finite and `>= 0.0`).
    pub min_cavity_volume: f64,
    /// Optional per-vertex extra wall thickness factors `(vertex, factor)` where local wall
    /// thickness becomes `wall_thickness * (1.0 + factor)` (`factor >= 0.0`).
    pub wall_modulation_weights: Option<Vec<(VertexId, f32)>>,
}

impl Default for HollowOptions {
    fn default() -> Self {
        Self {
            wall_thickness: 1.0,
            backend: HollowBackend::BvhSdf,
            voxel_size: None,
            max_voxels: DEFAULT_MAX_VOXELS,
            max_cavity_triangles: None,
            min_cavity_volume: 0.0,
            wall_modulation_weights: None,
        }
    }
}

/// Outcome metrics reported by [`hollow_mesh`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HollowReport {
    /// Number of interior cavities appended to the mesh (`0` if solid is too thin).
    pub cavities_created: usize,
    /// Number of vertices added for the cavity surface.
    pub cavity_vertices: usize,
    /// Number of triangles added for the cavity surface.
    pub cavity_triangles: usize,
    /// Enclosed volume of the generated inner cavity in cubic local units.
    pub cavity_volume: f64,
    /// Voxel spacing actually used for SDF extraction.
    pub voxel_size: f32,
    /// Whether `UV_CHANNEL` on the target mesh was dropped to keep UVs consistent.
    pub uvs_dropped: bool,
}

/// Specification of a single cylindrical, chamfered, and/or tapered drain hole.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrainHoleSpec {
    /// Point on the outer surface where the drain hole enters.
    pub surface_point: [f32; 3],
    /// Outward surface normal at `surface_point` (normalized internally; must be non-zero).
    pub surface_normal: [f32; 3],
    /// Nominal cylindrical bore diameter at the bottom of any outer chamfer (`> 0.0`).
    pub diameter: f32,
    /// Explicit inward pierce depth along `-surface_normal`; `None` auto-raycasts into the cavity.
    pub depth: Option<f32>,
    /// 45° outer rim bevel width/depth (`>= 0.0`).
    pub outer_chamfer: f32,
    /// Conical half-angle draft in degrees (`[0.0, 85.0)`) narrowing toward the inner end.
    pub taper_angle_deg: f32,
    /// Radial segment count around the hole circumference (must be `>= 8`).
    pub segments: u32,
}

impl Default for DrainHoleSpec {
    fn default() -> Self {
        Self {
            surface_point: [0.0, 0.0, 0.0],
            surface_normal: [0.0, 0.0, 1.0],
            diameter: 2.0,
            depth: None,
            outer_chamfer: 0.0,
            taper_angle_deg: 0.0,
            segments: 32,
        }
    }
}

/// Options controlling CSG subtraction and optional plug generation in [`punch_drain_hole`]
/// and [`punch_drain_holes`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrainHoleOptions {
    /// CSG boolean evaluation backend.
    pub boolean_method: BooleanMethod,
    /// When `true` and `Manifold` fails with [`OpsError::InvalidInput`], retry using `BvhSdf`.
    pub fallback_bvh_sdf: bool,
    /// Whether to construct matching frustum plug meshes in [`DrainHoleReport::plug_meshes`].
    pub generate_plug: bool,
    /// Uniform radial clearance subtracted from each plug radius (`>= 0.0`).
    pub plug_clearance: f32,
}

impl Default for DrainHoleOptions {
    fn default() -> Self {
        Self {
            boolean_method: BooleanMethod::Manifold,
            fallback_bvh_sdf: false,
            generate_plug: false,
            plug_clearance: 0.0,
        }
    }
}

/// Outcome of [`punch_drain_hole`] or [`punch_drain_holes`].
#[derive(Debug, Default)]
pub struct DrainHoleReport {
    /// Number of drain holes subtracted.
    pub holes_punched: usize,
    /// Triangle count of the target mesh after subtraction.
    pub kept_triangles: usize,
    /// Matching outward-wound manifold plug meshes (empty unless `generate_plug` is `true`).
    pub plug_meshes: Vec<Mesh>,
    /// Boolean backend that produced the final mesh.
    pub method_used: BooleanMethod,
    /// `true` when `Manifold` fell back to `BvhSdf`.
    pub used_fallback: bool,
    /// Whether `UV_CHANNEL` was dropped during the operation.
    pub uvs_dropped: bool,
}

/// Hollows a closed, outward-wound solid inside `tx` by extracting its `-wall_thickness`
/// inward isosurface and appending the reversed-winding cavity shell.
///
/// If `wall_thickness` (including any modulation) exceeds the object's maximum inscribed
/// thickness, returns `cavities_created: 0` without mutating `tx`.
pub fn hollow_mesh(tx: &mut Transaction<'_>, options: &HollowOptions) -> Result<HollowReport> {
    validate_hollow_options(options)?;

    let mesh = tx.mesh();
    if mesh.triangle_count() == 0 {
        return Err(OpsError::InvalidInput(
            "hollow requires a non-empty closed solid",
        ));
    }
    let positions = mesh.positions();
    for (v, p) in positions.iter().enumerate() {
        if mesh.is_vertex_alive(v as VertexId) && !p.iter().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput(
                "hollow needs finite vertex positions",
            ));
        }
    }

    let compact = compact_work_mesh(mesh)?;
    if !compact.has_positive_area {
        return Err(OpsError::InvalidInput(
            "hollow requires at least one non-degenerate triangle",
        ));
    }
    let mut work = compact.mesh;
    let min = compact.min;
    let max = compact.max;

    let (closed_oriented, signed_volume) = classify_solid(&work)?;
    if !closed_oriented {
        return Err(OpsError::InvalidInput(
            "hollow requires a closed manifold mesh",
        ));
    }
    if signed_volume <= 0.0 {
        return Err(OpsError::InvalidInput("hollow input is wound inward"));
    }

    let (work_weights, max_weight) = validate_and_map_weights(
        mesh,
        work.vertex_slots(),
        &compact.vertex_to_work,
        compact.weld_roots.as_deref(),
        options.wall_modulation_weights.as_deref(),
    )?;

    let dx = f64::from(max[0]) - f64::from(min[0]);
    let dy = f64::from(max[1]) - f64::from(min[1]);
    let dz = f64::from(max[2]) - f64::from(min[2]);
    let diag = (dx * dx + dy * dy + dz * dz).sqrt();
    let spec = choose_hollow_grid_spec(
        min,
        max,
        diag,
        options.voxel_size,
        options.wall_thickness,
        options.max_voxels,
    )?;

    let min_extent = dx.min(dy).min(dz);
    if f64::from(options.wall_thickness) >= 0.5 * min_extent {
        return Ok(HollowReport {
            cavities_created: 0,
            cavity_vertices: 0,
            cavity_triangles: 0,
            cavity_volume: 0.0,
            voxel_size: spec.voxel_size,
            uvs_dropped: false,
        });
    }

    let max_wall = options.wall_thickness * (1.0 + max_weight);
    let mut work_bvh_opt: Option<Bvh> = None;

    let grid = match options.backend {
        HollowBackend::BvhSdf => {
            let work_bvh = build_work_bvh(&work)?;
            let sign_data = SignData::build(&mut work, &work_bvh, SignMode::Auto)?;
            let narrow_band = (max_wall + 3.0 * spec.voxel_size).max(3.0 * spec.voxel_size);
            let grid = bvh_signed_grid(
                &work,
                &work_bvh,
                &sign_data,
                spec,
                narrow_band,
                options.max_voxels,
            )?;
            work_bvh_opt = Some(work_bvh);
            grid
        }
        HollowBackend::FhEdt => fh_signed_grid(&work, spec, options.max_voxels)?,
    };

    let iso = if let Some(weights) = work_weights.as_deref().filter(|_| max_weight > 0.0) {
        let work_bvh = match work_bvh_opt {
            Some(bvh) => bvh,
            None => build_work_bvh(&work)?,
        };
        let modulated = build_modulated_field(
            &work,
            &work_bvh,
            &grid,
            options.wall_thickness,
            max_wall,
            weights,
            options.max_voxels,
        )?;
        marching_cubes(&modulated, 0.0)?
    } else {
        marching_cubes(&grid, -options.wall_thickness)?
    };

    if iso.triangles.is_empty() {
        return Ok(HollowReport {
            cavities_created: 0,
            cavity_vertices: 0,
            cavity_triangles: 0,
            cavity_volume: 0.0,
            voxel_size: spec.voxel_size,
            uvs_dropped: false,
        });
    }

    let (cleaned, _) = clean_iso_mesh(iso)?;
    if cleaned.triangles.is_empty() {
        return Ok(HollowReport {
            cavities_created: 0,
            cavity_vertices: 0,
            cavity_triangles: 0,
            cavity_volume: 0.0,
            voxel_size: spec.voxel_size,
            uvs_dropped: false,
        });
    }

    // Build outward-wound temporary mesh to evaluate volume and optional simplification.
    let mut cavity_mesh = build_clean_mesh(&cleaned.positions, &cleaned.triangles)?;
    if let Some(cap) = options.max_cavity_triangles
        && cavity_mesh.triangle_count() > cap as usize
    {
        let mut ctx = cavity_mesh.begin_edit();
        simplify(
            &mut ctx,
            &SimplifyOptions {
                target_triangles: cap as usize,
                preserve_boundary: false,
            },
        )?;
        ctx.commit()?;
    }

    let cavity_volume = winding::signed_volume6(&cavity_mesh) / 6.0;
    if cavity_mesh.triangle_count() == 0
        || !cavity_volume.is_finite()
        || cavity_volume <= 0.0
        || cavity_volume < options.min_cavity_volume
    {
        return Ok(HollowReport {
            cavities_created: 0,
            cavity_vertices: 0,
            cavity_triangles: 0,
            cavity_volume: 0.0,
            voxel_size: spec.voxel_size,
            uvs_dropped: false,
        });
    }

    let source = extract_reversed_source(&cavity_mesh)?;
    let cavity_vertices = source.positions.len();
    let cavity_triangles = source.triangles.len();

    let append_report = append_mesh(tx, &source)?;

    Ok(HollowReport {
        cavities_created: 1,
        cavity_vertices,
        cavity_triangles,
        cavity_volume,
        voxel_size: spec.voxel_size,
        uvs_dropped: append_report.uvs_dropped,
    })
}

/// Punches a single cylindrical, chamfered, and/or tapered drain hole into `tx`.
pub fn punch_drain_hole(
    tx: &mut Transaction<'_>,
    hole: &DrainHoleSpec,
    options: &DrainHoleOptions,
) -> Result<DrainHoleReport> {
    punch_drain_holes(tx, std::slice::from_ref(hole), options)
}

/// Punches one or more drain holes into `tx`, optionally returning matching plug meshes.
///
/// On a hollowed solid, each hole bridges the outer shell and inner cavity into one
/// continuous watertight 2-manifold component when sufficient depth reaches the void.
pub fn punch_drain_holes(
    tx: &mut Transaction<'_>,
    holes: &[DrainHoleSpec],
    options: &DrainHoleOptions,
) -> Result<DrainHoleReport> {
    validate_drain_hole_options(options, holes)?;

    let had_uvs = tx.mesh().uvs().is_some();
    if holes.is_empty() {
        return Ok(DrainHoleReport {
            holes_punched: 0,
            kept_triangles: tx.mesh().triangle_count(),
            plug_meshes: Vec::new(),
            method_used: options.boolean_method,
            used_fallback: false,
            uvs_dropped: false,
        });
    }

    let orig_mesh = tx.mesh();
    if orig_mesh.triangle_count() == 0 {
        return Err(OpsError::InvalidInput(
            "punch_drain_holes requires a non-empty mesh",
        ));
    }

    // Compact / weld UV seams up front so queries and booleans operate on a clean solid.
    let compact = compact_work_mesh(orig_mesh)?;
    if !compact.has_positive_area {
        return Err(OpsError::InvalidInput(
            "punch_drain_holes requires at least one non-degenerate triangle",
        ));
    }
    let work_bvh = build_work_bvh(&compact.mesh)?;
    let target_diag = bounds_diag(compact.min, compact.max);

    let mut cutters: Vec<Mesh> = Vec::new();
    reserve(&mut cutters, holes.len())?;
    let mut plugs: Vec<Mesh> = Vec::new();
    if options.generate_plug {
        reserve(&mut plugs, holes.len())?;
    }
    let mut min_hole_diameter = f32::INFINITY;

    for hole in holes {
        min_hole_diameter = min_hole_diameter.min(hole.diameter);
        let built = build_hole_cutter_and_plug(
            &compact.mesh,
            &work_bvh,
            target_diag,
            hole,
            options.generate_plug,
            options.plug_clearance,
        )?;
        push(&mut cutters, built.cutter)?;
        if let Some(plug) = built.plug {
            push(&mut plugs, plug)?;
        }
    }

    let (result_mesh, method_used, used_fallback) = subtract_cutters(
        &compact.mesh,
        &cutters,
        compact.min,
        compact.max,
        min_hole_diameter,
        options,
    )?;

    replace_tx_with_mesh(tx, &result_mesh)?;

    Ok(DrainHoleReport {
        holes_punched: holes.len(),
        kept_triangles: tx.mesh().triangle_count(),
        plug_meshes: plugs,
        method_used,
        used_fallback,
        uvs_dropped: had_uvs,
    })
}

fn validate_hollow_options(options: &HollowOptions) -> Result<()> {
    if !options.wall_thickness.is_finite() || options.wall_thickness <= 0.0 {
        return Err(OpsError::InvalidInput(
            "wall_thickness must be finite and positive",
        ));
    }
    if options.max_voxels < 64 {
        return Err(OpsError::InvalidInput("max_voxels must be at least 64"));
    }
    if let Some(h) = options.voxel_size
        && (!h.is_finite() || h <= 0.0)
    {
        return Err(OpsError::InvalidInput(
            "voxel_size must be finite and positive",
        ));
    }
    if !options.min_cavity_volume.is_finite() || options.min_cavity_volume < 0.0 {
        return Err(OpsError::InvalidInput(
            "min_cavity_volume must be finite and non-negative",
        ));
    }
    if options.max_cavity_triangles == Some(0) {
        return Err(OpsError::InvalidInput(
            "max_cavity_triangles must be at least 1",
        ));
    }
    Ok(())
}

fn validate_and_map_weights(
    mesh: &Mesh,
    work_vertex_slots: usize,
    vertex_to_work: &[u32],
    weld_roots: Option<&[VertexId]>,
    weights: Option<&[(VertexId, f32)]>,
) -> Result<(Option<Vec<f32>>, f32)> {
    let Some(entries) = weights else {
        return Ok((None, 0.0));
    };
    let mut mapped = filled(work_vertex_slots, 0.0f32)?;
    let mut max_w = 0.0f32;
    for &(vid, w) in entries {
        if !w.is_finite() || w < 0.0 {
            return Err(OpsError::InvalidInput(
                "wall_modulation_weights must be finite and non-negative",
            ));
        }
        if (vid as usize) >= mesh.vertex_slots() || !mesh.is_vertex_alive(vid) {
            return Err(OpsError::InvalidInput(
                "wall_modulation_weights contains an invalid or dead vertex id",
            ));
        }
        let rv = match weld_roots {
            Some(roots) => roots[vid as usize],
            None => vid,
        };
        let wv = vertex_to_work[rv as usize];
        if wv != u32::MAX {
            let slot = &mut mapped[wv as usize];
            if w > *slot {
                *slot = w;
            }
            if w > max_w {
                max_w = w;
            }
        }
    }
    Ok((Some(mapped), max_w))
}

fn choose_hollow_grid_spec(
    min: [f32; 3],
    max: [f32; 3],
    diag: f64,
    voxel_size: Option<f32>,
    wall_thickness: f32,
    max_voxels: usize,
) -> Result<GridSpec> {
    if voxel_size.is_some() {
        return choose_grid_spec(min, max, diag, voxel_size, 0.0, max_voxels);
    }

    // Prefer at least ~2 voxels across the nominal wall thickness while respecting max_voxels.
    let auto_h = if diag > 0.0 {
        ((diag / DEFAULT_AUTO_GRID_RES as f64) as f32).max(1e-4)
    } else {
        (wall_thickness * 0.25).max(1e-3)
    };
    let wall_h = (wall_thickness * 0.45).max(1e-4);
    let mut h = auto_h.min(wall_h);

    let pad_voxels = 3usize;
    let dx = f64::from(max[0]) - f64::from(min[0]);
    let dy = f64::from(max[1]) - f64::from(min[1]);
    let dz = f64::from(max[2]) - f64::from(min[2]);
    let max_base = dx.max(dy).max(dz).max(0.0);
    let axis_budget = (max_voxels as f64).cbrt() - (2 * pad_voxels + 1) as f64;
    if axis_budget > 0.0 && max_base > 0.0 {
        let h_lower = ((max_base / axis_budget) * 1.01) as f32;
        if h_lower.is_finite() && h_lower > h {
            h = h_lower;
        }
    }

    for _ in 0..32 {
        match GridSpec::around_bounds(min, max, h, 0.0, pad_voxels, max_voxels) {
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
    choose_grid_spec(min, max, diag, Some(h), 0.0, max_voxels)
}

fn build_modulated_field(
    work: &Mesh,
    work_bvh: &Bvh,
    sdf_grid: &ScalarGrid,
    wall_thickness: f32,
    max_wall: f32,
    weights: &[f32],
    max_voxels: usize,
) -> Result<ScalarGrid> {
    let spec = sdf_grid.spec();
    let sdf_vals = sdf_grid.values();
    let total = sdf_vals.len();
    let band_limit = -(max_wall + 2.0 * spec.voxel_size);

    let mut out = filled(total, 0.0f32)?;
    let mut query_indices: Vec<usize> = Vec::new();
    let mut query_points: Vec<[f32; 3]> = Vec::new();
    let [nx, ny, _] = spec.dims;
    let nxy = nx * ny;

    for (idx, &sdf) in sdf_vals.iter().enumerate() {
        if sdf >= 0.0 {
            out[idx] = sdf + wall_thickness;
        } else if sdf <= band_limit {
            out[idx] = sdf + max_wall;
        } else {
            let iz = idx / nxy;
            let rem = idx % nxy;
            let iy = rem / nx;
            let ix = rem % nx;
            push(&mut query_indices, idx)?;
            push(&mut query_points, spec.point(ix, iy, iz))?;
        }
    }

    if !query_points.is_empty() {
        let mut hits: Vec<Option<ClosestHit>> = filled(query_points.len(), None)?;
        let opts = ClosestOptions {
            max_distance: f32::INFINITY,
        };
        closest_point_many(work, work_bvh, &query_points, &opts, &mut hits)?;
        for (&idx, hit_opt) in query_indices.iter().zip(hits.iter()) {
            let local_w = if let Some(hit) = hit_opt {
                if let Some([a, b, c]) = work.triangle(hit.triangle) {
                    let bary = hit.barycentric;
                    bary[0] * weights[a as usize]
                        + bary[1] * weights[b as usize]
                        + bary[2] * weights[c as usize]
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let local_wall = wall_thickness * (1.0 + local_w);
            out[idx] = sdf_vals[idx] + local_wall;
        }
    }

    Ok(ScalarGrid::new(spec, out, max_voxels)?)
}

fn build_clean_mesh(positions: &[[f32; 3]], triangles: &[[u32; 3]]) -> Result<Mesh> {
    allocation_allowed()?;
    let mut m = Mesh::new();
    let mut mtx = m.begin_edit();
    mtx.add_vertices(positions)?;
    mtx.add_triangles(triangles)?;
    mtx.commit()?;
    Ok(m)
}

fn extract_reversed_source(mesh: &Mesh) -> Result<AppendSource> {
    let positions = mesh.positions();
    let mut remap: Vec<u32> = filled(mesh.vertex_slots(), u32::MAX)?;
    let mut out_pos: Vec<[f32; 3]> = Vec::new();
    let mut out_tris: Vec<[u32; 3]> = Vec::new();
    reserve(&mut out_pos, mesh.vertex_count())?;
    reserve(&mut out_tris, mesh.triangle_count())?;

    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some([a, b, c]) = mesh.triangle(t) else {
            continue;
        };
        // Reverse [a, b, c] -> [a, c, b] so normals face inward into the cavity void.
        let mut mapped = [0u32; 3];
        for (k, v) in [a, c, b].into_iter().enumerate() {
            let idx = if remap[v as usize] != u32::MAX {
                remap[v as usize]
            } else {
                let fresh = u32::try_from(out_pos.len())
                    .map_err(|_| OpsError::Store(StoreError::OutOfMemory))?;
                push(&mut out_pos, positions[v as usize])?;
                remap[v as usize] = fresh;
                fresh
            };
            mapped[k] = idx;
        }
        push(&mut out_tris, mapped)?;
    }

    Ok(AppendSource {
        positions: out_pos,
        triangles: out_tris,
    })
}

fn validate_drain_hole_options(options: &DrainHoleOptions, holes: &[DrainHoleSpec]) -> Result<()> {
    if !options.plug_clearance.is_finite() || options.plug_clearance < 0.0 {
        return Err(OpsError::InvalidInput(
            "plug_clearance must be finite and non-negative",
        ));
    }
    for hole in holes {
        if !hole.surface_point.iter().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput(
                "drain hole surface_point must be finite",
            ));
        }
        if !hole.surface_normal.iter().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput(
                "drain hole surface_normal must be finite",
            ));
        }
        let nlen2 = dot(to_f64(hole.surface_normal), to_f64(hole.surface_normal));
        if !nlen2.is_finite() || nlen2 <= 1e-24 {
            return Err(OpsError::InvalidInput(
                "drain hole surface_normal must be non-zero",
            ));
        }
        if !hole.diameter.is_finite() || hole.diameter <= 0.0 {
            return Err(OpsError::InvalidInput(
                "drain hole diameter must be finite and positive",
            ));
        }
        if !hole.outer_chamfer.is_finite() || hole.outer_chamfer < 0.0 {
            return Err(OpsError::InvalidInput(
                "drain hole outer_chamfer must be finite and non-negative",
            ));
        }
        if !hole.taper_angle_deg.is_finite()
            || hole.taper_angle_deg < 0.0
            || hole.taper_angle_deg >= 85.0
        {
            return Err(OpsError::InvalidInput(
                "drain hole taper_angle_deg must be in [0, 85)",
            ));
        }
        if hole.segments < 8 {
            return Err(OpsError::InvalidInput(
                "drain hole segments must be at least 8",
            ));
        }
        if let Some(d) = hole.depth {
            if !d.is_finite() || d <= 0.0 {
                return Err(OpsError::InvalidInput(
                    "drain hole depth must be finite and positive",
                ));
            }
            if hole.outer_chamfer > 0.0 && d <= hole.outer_chamfer {
                return Err(OpsError::InvalidInput(
                    "drain hole depth must exceed outer_chamfer",
                ));
            }
        }
        if options.generate_plug && options.plug_clearance >= 0.5 * hole.diameter {
            return Err(OpsError::InvalidInput(
                "plug_clearance must be smaller than drain hole radius",
            ));
        }
    }
    Ok(())
}

struct BuiltHole {
    cutter: Mesh,
    plug: Option<Mesh>,
}

fn to_f64(v: [f32; 3]) -> [f64; 3] {
    [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
}

struct HoleFrame {
    origin: [f64; 3],
    t1: [f64; 3],
    t2: [f64; 3],
    n: [f64; 3],
}

fn orthonormal_frame(origin_f32: [f32; 3], normal_f32: [f32; 3]) -> HoleFrame {
    let n_raw = to_f64(normal_f32);
    let n_len = dot(n_raw, n_raw).sqrt();
    let n = [n_raw[0] / n_len, n_raw[1] / n_len, n_raw[2] / n_len];
    let helper = if n[0].abs() < 0.85 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let t1_raw = cross(helper, n);
    let t1_len = dot(t1_raw, t1_raw).sqrt();
    let t1 = [t1_raw[0] / t1_len, t1_raw[1] / t1_len, t1_raw[2] / t1_len];
    let t2 = cross(n, t1);
    HoleFrame {
        origin: to_f64(origin_f32),
        t1,
        t2,
        n,
    }
}

fn bounds_diag(min: [f32; 3], max: [f32; 3]) -> f64 {
    let dx = f64::from(max[0]) - f64::from(min[0]);
    let dy = f64::from(max[1]) - f64::from(min[1]);
    let dz = f64::from(max[2]) - f64::from(min[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

struct DepthEstimate {
    pierce_depth: f64,
    wall_thickness_est: f64,
}

fn estimate_pierce_depth(
    mesh: &Mesh,
    bvh: &Bvh,
    frame: &HoleFrame,
    r_rim: f64,
    target_diag: f64,
    outer_chamfer: f64,
) -> Result<DepthEstimate> {
    let (origin_p, t1, t2, n) = (frame.origin, frame.t1, frame.t2, frame.n);
    let offset_out = (0.5 * r_rim).max(target_diag * 0.05).max(1e-3);
    let dir_in = [-n[0] as f32, -n[1] as f32, -n[2] as f32];
    let ray_opts = RayOptions {
        min_t: 0.0,
        max_t: f32::INFINITY,
        cull: Cull::None,
    };

    let mut sample_points: Vec<[f64; 3]> = Vec::new();
    reserve(&mut sample_points, 9)?;
    push(&mut sample_points, origin_p)?;
    if r_rim > 0.0 {
        for i in 0..8 {
            let theta = TAU * (i as f64) / 8.0;
            let (s, c) = theta.sin_cos();
            let q = [
                origin_p[0] + 0.95 * r_rim * (c * t1[0] + s * t2[0]),
                origin_p[1] + 0.95 * r_rim * (c * t1[1] + s * t2[1]),
                origin_p[2] + 0.95 * r_rim * (c * t1[2] + s * t2[2]),
            ];
            push(&mut sample_points, q)?;
        }
    }

    let mut hits: Vec<RayHit> = Vec::new();
    let mut max_exit_depth = 0.0f64;
    let mut min_reentry_depth = f64::INFINITY;
    let mut any_reentry = false;
    let mut center_wall = 0.0f64;
    let mut any_exit = false;

    for (sample_idx, sp) in sample_points.iter().enumerate() {
        hits.clear();
        let ray_origin = [
            (sp[0] + offset_out * n[0]) as f32,
            (sp[1] + offset_out * n[1]) as f32,
            (sp[2] + offset_out * n[2]) as f32,
        ];
        let ray = Ray {
            origin: ray_origin,
            dir: dir_in,
        };
        raycast_all(mesh, bvh, &ray, &ray_opts, &mut hits)?;
        if hits.is_empty() {
            continue;
        }
        hits.sort_unstable_by(|a, b| a.t.total_cmp(&b.t));

        // Deduplicate near-identical edge/vertex hits separated by tiny numerical noise.
        let mut depths: Vec<(f64, f64)> = Vec::new();
        for hit in &hits {
            let d = f64::from(hit.t) - offset_out;
            let ndot = dot(to_f64(hit.normal), n);
            if depths
                .last()
                .is_some_and(|&(prev_d, _)| (d - prev_d).abs() < 1e-5)
            {
                continue;
            }
            push(&mut depths, (d, ndot))?;
        }

        // Pair outer entry (Hit 0) with wall exit (Hit 1) and optional cavity reentry (Hit 2).
        if depths.len() >= 2 {
            let d_entry = depths[0].0.max(0.0);
            let d_exit = depths[1].0;
            if d_exit > d_entry + 1e-6 {
                any_exit = true;
                if d_exit > max_exit_depth {
                    max_exit_depth = d_exit;
                }
                if sample_idx == 0 {
                    center_wall = (d_exit - d_entry).max(0.0);
                }
                if depths.len() >= 3 {
                    let d_reentry = depths[2].0;
                    if d_reentry > d_exit + 1e-6 {
                        any_reentry = true;
                        if d_reentry < min_reentry_depth {
                            min_reentry_depth = d_reentry;
                        }
                    }
                }
            }
        }
    }

    if !any_exit {
        // Fallback when rays grazed past the solid: pierce generously relative to diameter/diag.
        let fallback_depth = (2.0 * r_rim)
            .max(target_diag * 0.25)
            .max(outer_chamfer + 1e-2);
        return Ok(DepthEstimate {
            pierce_depth: fallback_depth,
            wall_thickness_est: fallback_depth,
        });
    }

    let wall_est = if center_wall > 0.0 {
        center_wall
    } else {
        max_exit_depth
    };
    let margin = (0.3 * wall_est).max(0.25 * r_rim).max(1e-3);
    let mut pierce_depth = (max_exit_depth + margin).max(outer_chamfer + 1e-3);

    if any_reentry && min_reentry_depth > max_exit_depth {
        let gap = min_reentry_depth - max_exit_depth;
        let safe_cap = max_exit_depth + 0.5 * gap;
        if pierce_depth > safe_cap {
            pierce_depth = safe_cap;
        }
    }

    Ok(DepthEstimate {
        pierce_depth,
        wall_thickness_est: wall_est.max(outer_chamfer + 1e-3),
    })
}

fn estimate_outward_clearance(
    mesh: &Mesh,
    bvh: &Bvh,
    origin_p: [f64; 3],
    n: [f64; 3],
    diameter: f64,
    bevel_above: f64,
) -> Result<f64> {
    let base_out = (0.25 * diameter)
        .max(bevel_above + 0.1 * diameter)
        .max(1e-3);
    let ray_origin = [
        (origin_p[0] + 1e-4 * diameter * n[0]) as f32,
        (origin_p[1] + 1e-4 * diameter * n[1]) as f32,
        (origin_p[2] + 1e-4 * diameter * n[2]) as f32,
    ];
    let ray = Ray {
        origin: ray_origin,
        dir: [n[0] as f32, n[1] as f32, n[2] as f32],
    };
    let mut hits: Vec<RayHit> = Vec::new();
    raycast_all(
        mesh,
        bvh,
        &ray,
        &RayOptions {
            min_t: 0.0,
            max_t: base_out as f32,
            cull: Cull::None,
        },
        &mut hits,
    )?;
    if hits.is_empty() {
        return Ok(base_out);
    }
    hits.sort_unstable_by(|a, b| a.t.total_cmp(&b.t));
    let nearest = f64::from(hits[0].t) + 1e-4 * diameter;
    Ok((0.5 * nearest).max(bevel_above + 1e-4))
}

fn build_hole_cutter_and_plug(
    mesh: &Mesh,
    bvh: &Bvh,
    target_diag: f64,
    hole: &DrainHoleSpec,
    generate_plug: bool,
    plug_clearance: f32,
) -> Result<BuiltHole> {
    let frame = orthonormal_frame(hole.surface_point, hole.surface_normal);
    let r_nom = 0.5 * f64::from(hole.diameter);
    let chamfer = f64::from(hole.outer_chamfer);
    let r_rim = r_nom + chamfer;
    let tan_alpha = f64::from(hole.taper_angle_deg).to_radians().tan();

    let (d_in, plug_depth) = if let Some(d) = hole.depth {
        let d_f64 = f64::from(d);
        (d_f64, d_f64)
    } else {
        let est = estimate_pierce_depth(mesh, bvh, &frame, r_rim, target_diag, chamfer)?;
        (
            est.pierce_depth,
            est.wall_thickness_est.min(est.pierce_depth),
        )
    };

    // Continue any 45° chamfer slightly above z = 0 so the bevel cone slices z = 0 cleanly
    // at r = r_nom + outer_chamfer without placing a creased ring directly on z = 0.
    let bevel_above = if chamfer > 0.0 {
        (0.15 * chamfer + 0.05 * r_nom).min(0.25 * f64::from(hole.diameter))
    } else {
        0.0
    };
    let d_out = estimate_outward_clearance(
        mesh,
        bvh,
        frame.origin,
        frame.n,
        f64::from(hole.diameter),
        bevel_above,
    )?;

    let bore_len = d_in - chamfer;
    let r_bot = (r_nom - bore_len * tan_alpha).max(0.1 * r_nom);

    let mut cutter_rings: Vec<(f64, f64)> = Vec::new();
    if chamfer > 0.0 {
        let r_top = r_rim + bevel_above;
        push(&mut cutter_rings, (d_out, r_top))?;
        push(&mut cutter_rings, (bevel_above, r_top))?;
        push(&mut cutter_rings, (-chamfer, r_nom))?;
        push(&mut cutter_rings, (-d_in, r_bot))?;
    } else {
        push(&mut cutter_rings, (d_out, r_nom))?;
        push(&mut cutter_rings, (-d_in, r_bot))?;
    }

    let cutter = build_revolved_solid(&frame, &cutter_rings, hole.segments)?;

    let plug = if generate_plug {
        let clr = f64::from(plug_clearance);
        let r_min_plug = (0.1 * (r_nom - clr)).max(1e-4);
        let plug_bore_len = (plug_depth - chamfer).max(0.0);
        let r_plug_bot = ((r_nom - plug_bore_len * tan_alpha) - clr).max(r_min_plug);
        let mut plug_rings: Vec<(f64, f64)> = Vec::new();
        if chamfer > 0.0 {
            let c_eff = chamfer.min(plug_depth * 0.85);
            push(
                &mut plug_rings,
                (0.0, (r_nom + c_eff - clr).max(r_min_plug)),
            )?;
            push(&mut plug_rings, (-c_eff, (r_nom - clr).max(r_min_plug)))?;
            push(&mut plug_rings, (-plug_depth, r_plug_bot))?;
        } else {
            push(&mut plug_rings, (0.0, (r_nom - clr).max(r_min_plug)))?;
            push(&mut plug_rings, (-plug_depth, r_plug_bot))?;
        }
        Some(build_revolved_solid(&frame, &plug_rings, hole.segments)?)
    } else {
        None
    };

    Ok(BuiltHole { cutter, plug })
}

/// Builds a closed, outward-wound 2-manifold revolved frustum mesh.
///
/// `rings` must be ordered strictly descending in axial coordinate `z` (top at `rings[0]`,
/// bottom at `rings[last]`) with positive radii.
fn build_revolved_solid(frame: &HoleFrame, rings: &[(f64, f64)], segments: u32) -> Result<Mesh> {
    let (origin_p, t1, t2, n) = (frame.origin, frame.t1, frame.t2, frame.n);
    let seg = segments as usize;
    let ring_count = rings.len();
    let vert_count = 2 + ring_count * seg;
    let tri_count = 2 * ring_count * seg;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    reserve(&mut positions, vert_count)?;
    reserve(&mut tris, tri_count)?;

    // Ring vertices
    for &(z, r) in rings {
        let base = [
            origin_p[0] + z * n[0],
            origin_p[1] + z * n[1],
            origin_p[2] + z * n[2],
        ];
        for i in 0..seg {
            let theta = TAU * (i as f64) / (seg as f64);
            let (s, c) = theta.sin_cos();
            let pos = [
                (base[0] + r * (c * t1[0] + s * t2[0])) as f32,
                (base[1] + r * (c * t1[1] + s * t2[1])) as f32,
                (base[2] + r * (c * t1[2] + s * t2[2])) as f32,
            ];
            if !pos.iter().all(|v| v.is_finite()) {
                return Err(OpsError::InvalidInput(
                    "drain hole vertex position leaves finite range",
                ));
            }
            push(&mut positions, pos)?;
        }
    }

    let top_center_idx = positions.len() as u32;
    let z_top = rings[0].0;
    push(
        &mut positions,
        [
            (origin_p[0] + z_top * n[0]) as f32,
            (origin_p[1] + z_top * n[1]) as f32,
            (origin_p[2] + z_top * n[2]) as f32,
        ],
    )?;

    let bot_center_idx = positions.len() as u32;
    let z_bot = rings[ring_count - 1].0;
    push(
        &mut positions,
        [
            (origin_p[0] + z_bot * n[0]) as f32,
            (origin_p[1] + z_bot * n[1]) as f32,
            (origin_p[2] + z_bot * n[2]) as f32,
        ],
    )?;

    // Top cap (+n outward)
    for i in 0..seg {
        let next = (i + 1) % seg;
        push(&mut tris, [top_center_idx, i as u32, next as u32])?;
    }

    // Side bands between rings (cross(-n, +e_theta) = +e_r outward)
    for k in 0..(ring_count - 1) {
        let r0 = (k * seg) as u32;
        let r1 = ((k + 1) * seg) as u32;
        for i in 0..seg {
            let next = ((i + 1) % seg) as u32;
            let i_u = i as u32;
            let a = r0 + i_u;
            let b = r0 + next;
            let c = r1 + next;
            let d = r1 + i_u;
            push(&mut tris, [a, d, c])?;
            push(&mut tris, [a, c, b])?;
        }
    }

    // Bottom cap (-n outward)
    let last_ring_base = ((ring_count - 1) * seg) as u32;
    for i in 0..seg {
        let next = ((i + 1) % seg) as u32;
        let i_u = i as u32;
        push(
            &mut tris,
            [bot_center_idx, last_ring_base + next, last_ring_base + i_u],
        )?;
    }

    build_clean_mesh(&positions, &tris)
}

fn subtract_cutters(
    orig_work: &Mesh,
    cutters: &[Mesh],
    target_min: [f32; 3],
    target_max: [f32; 3],
    min_hole_diameter: f32,
    options: &DrainHoleOptions,
) -> Result<(Mesh, BooleanMethod, bool)> {
    match options.boolean_method {
        BooleanMethod::Manifold => match subtract_manifold_sequential(orig_work, cutters) {
            Ok(out) => Ok((out, BooleanMethod::Manifold, false)),
            Err(OpsError::InvalidInput(_)) if options.fallback_bvh_sdf => {
                let out = subtract_voxel_batch(
                    orig_work,
                    cutters,
                    target_min,
                    target_max,
                    min_hole_diameter,
                    BooleanMethod::BvhSdf,
                )?;
                Ok((out, BooleanMethod::BvhSdf, true))
            }
            Err(e) => Err(e),
        },
        method @ (BooleanMethod::BvhSdf | BooleanMethod::FhEdt) => {
            let out = subtract_voxel_batch(
                orig_work,
                cutters,
                target_min,
                target_max,
                min_hole_diameter,
                method,
            )?;
            Ok((out, method, false))
        }
    }
}

fn subtract_manifold_sequential(orig_work: &Mesh, cutters: &[Mesh]) -> Result<Mesh> {
    let mut current = clone_mesh_fallible(orig_work)?;
    let opts = BooleanOptions {
        op: BooleanOp::Subtract,
        method: BooleanMethod::Manifold,
        coplanar_tolerance: 1e-5,
        snap_slivers: true,
        voxel_pitch: None,
        max_triangles: None,
    };

    for cutter in cutters {
        let mut cur_tx = current.begin_edit();
        apply_boolean_with_options(&mut cur_tx, cutter, Rigid::IDENTITY, &opts)?;
        cur_tx.commit()?;
    }
    Ok(current)
}

fn subtract_voxel_batch(
    orig_work: &Mesh,
    cutters: &[Mesh],
    target_min: [f32; 3],
    target_max: [f32; 3],
    min_hole_diameter: f32,
    method: BooleanMethod,
) -> Result<Mesh> {
    let diag = bounds_diag(target_min, target_max);
    let default_h = if diag > 0.0 {
        (diag / DEFAULT_AUTO_GRID_RES as f64) as f32
    } else {
        1e-3
    };
    let pitch = if min_hole_diameter.is_finite() && min_hole_diameter * 0.25 < default_h {
        let mut min_b = target_min;
        let mut max_b = target_max;
        for cutter in cutters {
            for v in 0..cutter.vertex_slots() as VertexId {
                if let Some(p) = cutter.position(v) {
                    for k in 0..3 {
                        min_b[k] = min_b[k].min(p[k]);
                        max_b[k] = max_b[k].max(p[k]);
                    }
                }
            }
        }
        let max_span = (0..3)
            .map(|k| f64::from(max_b[k]) - f64::from(min_b[k]))
            .fold(0.0f64, f64::max);
        let axis_budget = (DEFAULT_MAX_VOXELS as f64).cbrt() - 7.0;
        let min_safe_h = if axis_budget > 0.0 && max_span > 0.0 {
            ((max_span / axis_budget) * 1.02) as f32
        } else {
            1e-4
        };
        Some((min_hole_diameter * 0.2).max(min_safe_h))
    } else {
        None
    };

    let opts = BooleanOptions {
        op: BooleanOp::Subtract,
        method,
        coplanar_tolerance: 1e-5,
        snap_slivers: true,
        voxel_pitch: pitch,
        max_triangles: None,
    };

    if cutters.len() == 1 {
        let (out, _) = boolean(orig_work, &cutters[0], Rigid::IDENTITY, &opts)?;
        return Ok(out);
    }

    let mut inputs: Vec<(&Mesh, Rigid)> = Vec::new();
    reserve(&mut inputs, 1 + cutters.len())?;
    push(&mut inputs, (orig_work, Rigid::IDENTITY))?;
    for cutter in cutters {
        push(&mut inputs, (cutter, Rigid::IDENTITY))?;
    }

    let mut tree = CsgNode::Leaf(0);
    for idx in 0..cutters.len() {
        tree = CsgNode::Op {
            op: BooleanOp::Subtract,
            left: Box::new(tree),
            right: Box::new(CsgNode::Leaf(1 + idx)),
        };
    }

    boolean_batch(&inputs, &tree, &opts)
}
