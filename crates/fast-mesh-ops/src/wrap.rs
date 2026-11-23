//! Wrapping: a closed, manifold envelope around one or more meshes.
//!
//! [`wrap`] takes a triangle soup in one frame (one mesh or several meshes
//! appended together; open, holed, overlapping, self-intersecting or
//! non-manifold input is all fine) and returns a new surface that is closed,
//! 2-manifold, wound outward and free of self-intersections, with every gap,
//! slot and hole narrower than [`WrapOptions::gap`] bridged. It is a
//! morphological closing on a voxel grid, meshed with the Manifold Marching
//! Cubes of [`crate::marching_cubes`]:
//!
//! 1. **Unsigned distance.** The exact Felzenszwalb-Huttenlocher distance
//!    transform of the voxelised input ([`fh_unsigned_grid`]), refined to
//!    the exact closest-triangle distance (through a BVH) on the nodes that
//!    can lie next to the offset surface below.
//! 2. **Dilate by the closing radius `r`** (half the gap). The nodes farther
//!    than `r` from the input that are reachable from the grid boundary
//!    (6-connected flood fill) are the outside; everything else, enclosed
//!    cavities included, is filled. Marching cubes extracts the boundary of
//!    the filled region: the outward offset of the input by `r`, with every
//!    gap under `2r` closed.
//! 3. **Erode back by `r`.** The distance from every filled node to that
//!    offset surface, by a second distance transform on the outside nodes,
//!    refined to the exact distance (through a BVH of the offset surface) on
//!    the nodes next to the final surface.
//! 4. **Extract** the nodes deeper than `r` with marching cubes again.
//! 5. **Project** (optional, [`WrapOptions::project`]): every vertex within
//!    one voxel of the input moves to its closest input point, unless that
//!    flips, collapses or folds a triangle or makes the surface intersect
//!    itself; such vertices stay where marching cubes put them.
//!
//! Where the input is smoother than `r`, the result follows it to within
//! the voxel resolution; concave corners get fillets of radius about `r`,
//! and openings narrower than `2r` are spanned. Every result vertex lies
//! within about `r` (plus a voxel) of the input. Interior cavities are
//! filled: the result is the outer envelope. Features thinner than a voxel
//! may vanish or break into pieces, and a surface that encloses no volume
//! at this gap (a single open sheet, say) wraps to nothing, which is an
//! error.
//!
//! The grid is planned before anything is allocated: [`wrap_plan`] reports
//! its dimensions, voxel count and estimated memory for given bounds, and
//! [`wrap`] refuses a grid over [`WrapOptions::max_voxels`]. With the
//! `parallel` feature, the distance transforms, the closest-point batches and
//! marching cubes run on rayon's pool; the result is bit-identical with and
//! without the feature, for any thread count.

use fast_mesh_bvh::{BuildOptions, Bvh, BvhError};
use fast_mesh_fields::{DEFAULT_MAX_VOXELS, GridSpec, edt_3d_sq, fh_unsigned_grid};
use fast_mesh_query::{ClosestHit, ClosestOptions, closest_point_many, self_intersecting_pairs};
use fast_mesh_store::{Mesh, StoreError, TriangleId};

use crate::append::{AppendSource, append_mesh};
use crate::{IsoMesh, OpsError, Result, allocation_allowed, filled, marching_cubes_raw, reserve};

/// Default [`WrapOptions::detail`]: voxels along the longest side of the
/// input's bounds.
pub const DEFAULT_WRAP_DETAIL: u32 = 128;

/// The smallest [`WrapOptions::detail`] accepted.
pub const MIN_WRAP_DETAIL: u32 = 8;

/// The largest [`WrapOptions::detail`] accepted.
pub const MAX_WRAP_DETAIL: u32 = 4096;

/// Bytes per voxel [`WrapPlan::bytes`] counts, at the peak: a distance grid
/// (`f32`, also a distance transform's input), the outside flags (`u8`), and
/// the distance transform's work buffer (`f64`) and output (`f32`).
pub const WRAP_BYTES_PER_VOXEL: usize = 17;

/// Empty voxels around the offset surface on every side of the grid.
const PAD_VOXELS: usize = 3;

/// Grid values closer than this many voxels to the iso level are moved off
/// it, so every marching cubes vertex keeps at least about this fraction of
/// a voxel from the grid nodes and no triangle collapses.
const NUDGE: f32 = 0.01;

/// Grid nodes per closest-point batch.
const QUERY_BATCH: usize = 16_384;

/// Rounds of self-intersection checks after projection before every
/// projected vertex is put back.
const MAX_PROJECT_ROUNDS: usize = 8;

/// Options for [`wrap`] and [`wrap_plan`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WrapOptions {
    /// The widest gap, slot or hole that gets closed, in local units:
    /// openings narrower than this are bridged. The closing radius is half
    /// of it, and at least one voxel. Finite and not negative; default `0`
    /// (close only what the voxels cannot resolve).
    pub gap: f32,
    /// The grid spacing in local units; `None` (the default) derives it
    /// from [`WrapOptions::detail`]. Positive and finite when set.
    pub voxel_size: Option<f32>,
    /// Voxels along the longest side of the input's bounds when
    /// `voxel_size` is `None`; [`MIN_WRAP_DETAIL`] to [`MAX_WRAP_DETAIL`],
    /// default [`DEFAULT_WRAP_DETAIL`].
    pub detail: u32,
    /// Move the result's vertices that lie within one voxel of the input
    /// onto it, where that keeps the surface clean (see the module
    /// documentation). Default `false`.
    pub project: bool,
    /// The largest grid accepted, in voxels (at least 64); a larger plan is
    /// refused before anything is allocated. Default
    /// [`DEFAULT_MAX_VOXELS`] (`256³`).
    pub max_voxels: usize,
}

impl Default for WrapOptions {
    fn default() -> Self {
        WrapOptions {
            gap: 0.0,
            voxel_size: None,
            detail: DEFAULT_WRAP_DETAIL,
            project: false,
            max_voxels: DEFAULT_MAX_VOXELS,
        }
    }
}

impl WrapOptions {
    /// A builder for [`WrapOptions`]: every option as [`Default`] until set.
    pub fn builder() -> WrapOptionsBuilder {
        WrapOptionsBuilder::default()
    }
}

/// A builder for [`WrapOptions`], see [`WrapOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct WrapOptionsBuilder {
    options: WrapOptions,
}

impl WrapOptionsBuilder {
    /// The widest gap, slot or hole that gets closed.
    pub fn gap(mut self, gap: f32) -> Self {
        self.options.gap = gap;
        self
    }

    /// The grid spacing; `None` derives it from the detail.
    pub fn voxel_size(mut self, voxel_size: Option<f32>) -> Self {
        self.options.voxel_size = voxel_size;
        self
    }

    /// Voxels along the longest side of the input's bounds.
    pub fn detail(mut self, detail: u32) -> Self {
        self.options.detail = detail;
        self
    }

    /// Whether to project the result onto the input.
    pub fn project(mut self, project: bool) -> Self {
        self.options.project = project;
        self
    }

    /// The largest grid accepted, in voxels.
    pub fn max_voxels(mut self, max_voxels: usize) -> Self {
        self.options.max_voxels = max_voxels;
        self
    }

    /// The options as set.
    pub fn build(self) -> WrapOptions {
        self.options
    }
}

/// The voxel grid a wrap uses, as [`wrap_plan`] computes it before anything
/// is allocated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WrapPlan {
    /// Grid nodes along x, y and z.
    pub dims: [usize; 3],
    /// The local position of node `(0, 0, 0)`; meaningful when
    /// [`WrapPlan::fits`].
    pub origin: [f32; 3],
    /// The grid spacing in local units.
    pub voxel_size: f32,
    /// The closing radius in local units: half the gap, at least one voxel.
    pub radius: f32,
    /// Grid nodes in all (`usize::MAX` when the count overflows).
    pub voxels: usize,
    /// Estimated peak memory of the grids, in bytes:
    /// [`WRAP_BYTES_PER_VOXEL`] per voxel. The input, the meshes extracted
    /// and their BVHs come on top.
    pub bytes: usize,
    /// The limit the plan was checked against ([`WrapOptions::max_voxels`]).
    pub max_voxels: usize,
    /// Whether `voxels <= max_voxels`; [`wrap`] refuses the plan otherwise.
    pub fits: bool,
}

/// What [`wrap`] did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WrapReport {
    /// The grid used.
    pub plan: WrapPlan,
    /// Vertices of the result.
    pub vertices: usize,
    /// Triangles of the result.
    pub triangles: usize,
    /// Result vertices moved onto the input (`0` without
    /// [`WrapOptions::project`]).
    pub projected: usize,
}

/// Plans the grid of a wrap of geometry within the local bounds `[min, max]`
/// under `options`, without allocating it. [`wrap`] uses exactly this plan
/// for the bounds of the triangles it wraps.
///
/// # Errors
/// [`OpsError::InvalidInput`] for invalid options (see [`WrapOptions`]),
/// bounds that are not finite or have `min > max`, or bounds without extent
/// and no voxel size. A plan over the limit is not an error here: it comes
/// back with `fits == false`.
pub fn wrap_plan(min: [f32; 3], max: [f32; 3], options: &WrapOptions) -> Result<WrapPlan> {
    validate(options)?;
    if !min.iter().chain(&max).all(|c| c.is_finite()) || (0..3).any(|k| min[k] > max[k]) {
        return Err(OpsError::InvalidInput(
            "wrap bounds must be finite with min <= max",
        ));
    }
    let longest = (0..3)
        .map(|k| f64::from(max[k]) - f64::from(min[k]))
        .fold(0.0f64, f64::max);
    let h = match options.voxel_size {
        Some(h) => h,
        None if longest > 0.0 => (longest / f64::from(options.detail)) as f32,
        None => {
            return Err(OpsError::InvalidInput(
                "wrap bounds have no extent; set a voxel size",
            ));
        }
    };
    if !h.is_finite() || h <= 0.0 {
        return Err(OpsError::InvalidInput(
            "wrap voxel size must be positive and finite",
        ));
    }
    let radius = (0.5 * options.gap).max(h);
    let hd = f64::from(h);
    let margin = f64::from(radius) + PAD_VOXELS as f64 * hd;
    let mut dims = [0usize; 3];
    let mut origin = [0.0f32; 3];
    for k in 0..3 {
        let span = f64::from(max[k]) - f64::from(min[k]) + 2.0 * margin;
        let cells = (span / hd).ceil();
        dims[k] = if cells.is_finite() && cells < 1e15 {
            (cells as usize + 1).max(4)
        } else {
            usize::MAX
        };
        let center = 0.5 * (f64::from(min[k]) + f64::from(max[k]));
        origin[k] = (center - 0.5 * dims[k].saturating_sub(1) as f64 * hd) as f32;
    }
    let voxels = dims[0]
        .checked_mul(dims[1])
        .and_then(|xy| xy.checked_mul(dims[2]))
        .unwrap_or(usize::MAX);
    Ok(WrapPlan {
        dims,
        origin,
        voxel_size: h,
        radius,
        voxels,
        bytes: voxels.saturating_mul(WRAP_BYTES_PER_VOXEL),
        max_voxels: options.max_voxels,
        fits: voxels <= options.max_voxels,
    })
}

/// Wraps the triangle soup `source` in a closed, manifold, outward-wound
/// envelope without self-intersections, closing gaps narrower than
/// `options.gap` (see the module documentation). The result is in the same
/// frame as `source`.
///
/// Triangles with a repeated index or no area are ignored, as are positions
/// no other triangle uses.
///
/// # Errors
/// - [`OpsError::InvalidInput`] for invalid options, a triangle index past
///   the positions, a non-finite position on a triangle, no triangle with
///   area, a grid over `options.max_voxels` (`"wrap grid exceeds
///   max_voxels"`; [`wrap_plan`] has the numbers), or an empty result (the
///   input encloses no volume at this gap and voxel size).
/// - [`OpsError::Store`] with [`StoreError::OutOfMemory`] when memory runs
///   out.
pub fn wrap(source: &AppendSource, options: &WrapOptions) -> Result<(AppendSource, WrapReport)> {
    validate(options)?;
    allocation_allowed()?;
    let work = work_mesh(source)?;
    let plan = wrap_plan(work.min, work.max, options)?;
    if !plan.fits {
        return Err(OpsError::InvalidInput("wrap grid exceeds max_voxels"));
    }
    let spec = GridSpec {
        origin: plan.origin,
        dims: plan.dims,
        voxel_size: plan.voxel_size,
    };
    spec.validate(options.max_voxels)?;
    let h = plan.voxel_size;
    let r = plan.radius;
    let delta = NUDGE * h;
    let bvh = build_bvh(&work.mesh)?;

    // 1. The unsigned distance to the input, exact where it can be within
    // about a voxel of `r` (the voxelised transform is off by at most
    // `sqrt(3) / 2` voxels).
    let mut grid = fh_unsigned_grid(&work.mesh, spec, options.max_voxels)?.into_values();
    refine(
        &mut grid,
        &spec,
        |_, d| (d - r).abs() <= 2.0 * h,
        &work.mesh,
        &bvh,
        r + 3.0 * h,
        |d| d,
    )?;

    // 2. Dilate: the outside, and the boundary of everything else.
    let outside = flood_outside(&grid, spec.dims, r)?;
    for (v, &o) in grid.iter_mut().zip(&outside) {
        *v = if o != 0 {
            (*v - r).max(delta)
        } else {
            (*v - r).min(-delta)
        };
    }
    let dilated = marching_cubes_raw(spec.origin, spec.dims, h, &grid, 0.0)?;
    let dilated = mesh_of_iso(dilated)?;
    let dilated_bvh = build_bvh(&dilated)?;

    // 3. Erode: the distance from every filled node to the dilated surface,
    // first in whole voxels to the nearest outside node (at most `sqrt(3)`
    // voxels more than the true distance), then exact near `r`.
    for (v, &o) in grid.iter_mut().zip(&outside) {
        *v = if o != 0 { 0.0 } else { f32::INFINITY };
    }
    {
        let mut sq = filled(grid.len(), 0.0f32)?;
        edt_3d_sq(spec.dims, &grid, &mut sq)?;
        for ((v, &o), &s) in grid.iter_mut().zip(&outside).zip(&sq) {
            *v = if o != 0 { r } else { r - h * s.sqrt() };
        }
    }
    refine(
        &mut grid,
        &spec,
        |i, f| outside[i] == 0 && f >= -2.8 * h && f <= 1.1 * h,
        &dilated,
        &dilated_bvh,
        r + 3.0 * h,
        |e| r - e,
    )?;
    drop(dilated_bvh);
    drop(dilated);
    drop(outside);
    for v in &mut grid {
        if *v > -delta && *v < delta {
            *v = delta;
        }
    }

    // 4. Extract.
    let mut iso = marching_cubes_raw(spec.origin, spec.dims, h, &grid, 0.0)?;
    drop(grid);
    if iso.triangles.is_empty() {
        return Err(OpsError::InvalidInput(
            "wrap encloses no volume at this gap and voxel size",
        ));
    }

    // 5. Project.
    let projected = if options.project {
        project(&mut iso, &work.mesh, &bvh, h)?
    } else {
        0
    };
    let report = WrapReport {
        plan,
        vertices: iso.positions.len(),
        triangles: iso.triangles.len(),
        projected,
    };
    Ok((
        AppendSource {
            positions: iso.positions,
            triangles: iso.triangles,
        },
        report,
    ))
}

/// Wraps `meshes` together into a new mesh (see [`wrap`]). With one shared
/// origin (one mesh, say) the wrap is built in their local frame and keeps
/// that origin; otherwise it is built around the centre of their bounds,
/// which becomes its origin. Lengths in `options` are local units.
///
/// # Errors
/// [`OpsError::InvalidInput`] for an empty list, a local position that does
/// not fit `f32` around the new origin, and as for [`wrap`].
pub fn wrap_meshes(meshes: &[&Mesh], options: &WrapOptions) -> Result<(Mesh, WrapReport)> {
    let Some(first) = meshes.first() else {
        return Err(OpsError::InvalidInput("wrap needs at least one mesh"));
    };
    let origin = first.origin();
    let shared = meshes.iter().all(|m| m.origin() == origin);
    let center = if shared {
        origin
    } else {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for mesh in meshes {
            let o = mesh.origin();
            for (v, p) in mesh.positions().iter().enumerate() {
                if mesh.is_vertex_alive(v as u32) && mesh.corner_count(v as u32) > 0 {
                    for k in 0..3 {
                        let w = o[k] + f64::from(p[k]);
                        lo[k] = lo[k].min(w);
                        hi[k] = hi[k].max(w);
                    }
                }
            }
        }
        if lo[0] > hi[0] {
            origin
        } else {
            std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]))
        }
    };
    let mut soup = AppendSource::default();
    for mesh in meshes {
        let o = mesh.origin();
        let shift: [f64; 3] = std::array::from_fn(|k| o[k] - center[k]);
        push_soup(
            mesh,
            |p| std::array::from_fn(|k| f64::from(p[k]) + shift[k]),
            &mut soup,
        )?;
    }
    let (out, report) = wrap(&soup, options)?;
    let mut mesh = Mesh::from_buffers_f64_with_origin(&[], &[], center)?;
    let mut tx = mesh.begin_edit();
    append_mesh(&mut tx, &out)?;
    tx.commit()?;
    Ok((mesh, report))
}

/// Appends the live triangles of `mesh` to `soup`, with the vertices they
/// use mapped by `map` (from local `f32` positions to the soup's frame).
/// Returns the triangles appended.
///
/// # Errors
/// [`OpsError::InvalidInput`] when a mapped position does not fit `f32`;
/// [`OpsError::Store`] when out of memory.
pub fn push_soup(
    mesh: &Mesh,
    mut map: impl FnMut([f32; 3]) -> [f64; 3],
    soup: &mut AppendSource,
) -> Result<usize> {
    let mut remap: Vec<u32> = filled(mesh.vertex_slots(), u32::MAX)?;
    reserve(&mut soup.triangles, mesh.triangle_count())?;
    let positions = mesh.positions();
    let mut added = 0;
    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        let mut out = [0u32; 3];
        for (slot, v) in out.iter_mut().zip(tri) {
            if remap[v as usize] == u32::MAX {
                let index = u32::try_from(soup.positions.len())
                    .map_err(|_| OpsError::Store(StoreError::OutOfMemory))?;
                let w = map(positions[v as usize]);
                let local: [f32; 3] = std::array::from_fn(|k| w[k] as f32);
                if !local.iter().all(|c| c.is_finite()) {
                    return Err(OpsError::InvalidInput(
                        "wrap position does not fit an f32 position",
                    ));
                }
                if soup.positions.len() == soup.positions.capacity() {
                    let grow = soup.positions.len().max(64);
                    reserve(&mut soup.positions, grow)?;
                }
                soup.positions.push(local);
                remap[v as usize] = index;
            }
            *slot = remap[v as usize];
        }
        soup.triangles.push(out);
        added += 1;
    }
    Ok(added)
}

fn validate(options: &WrapOptions) -> Result<()> {
    if !options.gap.is_finite() || options.gap < 0.0 {
        return Err(OpsError::InvalidInput(
            "wrap gap must be finite and not negative",
        ));
    }
    if let Some(h) = options.voxel_size
        && (!h.is_finite() || h <= 0.0)
    {
        return Err(OpsError::InvalidInput(
            "wrap voxel size must be positive and finite",
        ));
    }
    if !(MIN_WRAP_DETAIL..=MAX_WRAP_DETAIL).contains(&options.detail) {
        return Err(OpsError::InvalidInput("wrap detail must be 8 to 4096"));
    }
    if options.max_voxels < 64 {
        return Err(OpsError::InvalidInput(
            "wrap max_voxels must be at least 64",
        ));
    }
    Ok(())
}

/// The proper triangles of a soup as a compact mesh, with their bounds.
struct Work {
    mesh: Mesh,
    min: [f32; 3],
    max: [f32; 3],
}

fn work_mesh(source: &AppendSource) -> Result<Work> {
    let n = source.positions.len();
    let mut remap: Vec<u32> = filled(n, u32::MAX)?;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    reserve(&mut triangles, source.triangles.len())?;
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for &tri in &source.triangles {
        if tri.iter().any(|&v| v as usize >= n) {
            return Err(OpsError::InvalidInput("wrap triangle index out of range"));
        }
        let [a, b, c] = tri;
        if a == b || b == c || a == c {
            continue;
        }
        let p = tri.map(|v| source.positions[v as usize]);
        if !p.iter().flatten().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput("wrap needs finite vertex positions"));
        }
        let q = p.map(|x| x.map(f64::from));
        let u: [f64; 3] = std::array::from_fn(|k| q[1][k] - q[0][k]);
        let w: [f64; 3] = std::array::from_fn(|k| q[2][k] - q[0][k]);
        let cross = [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ];
        if cross == [0.0; 3] {
            continue;
        }
        let mut out = [0u32; 3];
        for (slot, v) in out.iter_mut().zip(tri) {
            if remap[v as usize] == u32::MAX {
                let pos = source.positions[v as usize];
                for k in 0..3 {
                    min[k] = min[k].min(pos[k]);
                    max[k] = max[k].max(pos[k]);
                }
                remap[v as usize] = positions.len() as u32;
                if positions.len() == positions.capacity() {
                    let grow = positions.len().max(64);
                    reserve(&mut positions, grow)?;
                }
                positions.push(pos);
            }
            *slot = remap[v as usize];
        }
        triangles.push(out);
    }
    if triangles.is_empty() {
        return Err(OpsError::InvalidInput(
            "wrap needs at least one triangle with area",
        ));
    }
    let mesh = Mesh::from_buffers(positions.as_flattened(), triangles.as_flattened())?;
    Ok(Work { mesh, min, max })
}

fn build_bvh(mesh: &Mesh) -> Result<Bvh> {
    Bvh::build(mesh, &BuildOptions::default()).map_err(|e| match e {
        BvhError::OutOfMemory => OpsError::Store(StoreError::OutOfMemory),
        e => e.into(),
    })
}

/// A mesh of the marching cubes output `iso`.
fn mesh_of_iso(iso: IsoMesh) -> Result<Mesh> {
    let mesh = Mesh::from_buffers(iso.positions.as_flattened(), iso.triangles.as_flattened())?;
    Ok(mesh)
}

/// Replaces `values[i]` by `map(distance)` for every node `i` that `select`
/// picks, where `distance` is the exact distance from the node to `mesh`
/// (`max_distance` when nothing is that close).
fn refine(
    values: &mut [f32],
    spec: &GridSpec,
    select: impl Fn(usize, f32) -> bool,
    mesh: &Mesh,
    bvh: &Bvh,
    max_distance: f32,
    map: impl Fn(f32) -> f32,
) -> Result<()> {
    let [nx, ny, _] = spec.dims;
    let nxy = nx * ny;
    let mut nodes: Vec<usize> = Vec::new();
    reserve(&mut nodes, QUERY_BATCH)?;
    let mut points: Vec<[f32; 3]> = Vec::new();
    reserve(&mut points, QUERY_BATCH)?;
    let mut hits: Vec<Option<ClosestHit>> = filled(QUERY_BATCH, None)?;
    let opts = ClosestOptions { max_distance };
    let mut flush =
        |nodes: &mut Vec<usize>, points: &mut Vec<[f32; 3]>, values: &mut [f32]| -> Result<()> {
            let hits = &mut hits[..points.len()];
            closest_point_many(mesh, bvh, points, &opts, hits)?;
            for (&i, hit) in nodes.iter().zip(hits.iter()) {
                values[i] = map(hit.map_or(max_distance, |h| h.distance));
            }
            nodes.clear();
            points.clear();
            Ok(())
        };
    for i in 0..values.len() {
        if !select(i, values[i]) {
            continue;
        }
        let iz = i / nxy;
        let rem = i % nxy;
        nodes.push(i);
        points.push(spec.point(rem % nx, rem / nx, iz));
        if nodes.len() == QUERY_BATCH {
            flush(&mut nodes, &mut points, values)?;
        }
    }
    if !nodes.is_empty() {
        flush(&mut nodes, &mut points, values)?;
    }
    Ok(())
}

/// The nodes farther than `r` that a 6-connected walk from the grid boundary
/// through such nodes reaches, as `1`; every other node `0`.
fn flood_outside(distance: &[f32], dims: [usize; 3], r: f32) -> Result<Vec<u8>> {
    let [nx, ny, nz] = dims;
    let nxy = nx * ny;
    let mut outside: Vec<u8> = filled(distance.len(), 0)?;
    let mut frontier: Vec<usize> = Vec::new();
    let mut next: Vec<usize> = Vec::new();
    let visit = |i: usize, outside: &mut [u8], into: &mut Vec<usize>| -> Result<()> {
        if outside[i] == 0 && distance[i] > r {
            outside[i] = 1;
            if into.len() == into.capacity() {
                let grow = into.len().max(1024);
                reserve(into, grow)?;
            }
            into.push(i);
        }
        Ok(())
    };
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                if ix == 0 || iy == 0 || iz == 0 || ix + 1 == nx || iy + 1 == ny || iz + 1 == nz {
                    visit((iz * ny + iy) * nx + ix, &mut outside, &mut frontier)?;
                }
            }
        }
    }
    while !frontier.is_empty() {
        for &i in &frontier {
            let iz = i / nxy;
            let rem = i % nxy;
            let iy = rem / nx;
            let ix = rem % nx;
            if ix > 0 {
                visit(i - 1, &mut outside, &mut next)?;
            }
            if ix + 1 < nx {
                visit(i + 1, &mut outside, &mut next)?;
            }
            if iy > 0 {
                visit(i - nx, &mut outside, &mut next)?;
            }
            if iy + 1 < ny {
                visit(i + nx, &mut outside, &mut next)?;
            }
            if iz > 0 {
                visit(i - nxy, &mut outside, &mut next)?;
            }
            if iz + 1 < nz {
                visit(i + nxy, &mut outside, &mut next)?;
            }
        }
        std::mem::swap(&mut frontier, &mut next);
        next.clear();
    }
    Ok(outside)
}

/// Twice the area vector of the triangle `[a, b, c]` of `p`, in `f64`.
fn normal(p: &[[f32; 3]], [a, b, c]: [u32; 3]) -> [f64; 3] {
    let [pa, pb, pc] = [a, b, c].map(|v| p[v as usize].map(f64::from));
    let u: [f64; 3] = std::array::from_fn(|k| pb[k] - pa[k]);
    let w: [f64; 3] = std::array::from_fn(|k| pc[k] - pa[k]);
    [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// The cosine of the angle between `a` and `b` (`-1` when either is zero).
fn cosine(a: [f64; 3], b: [f64; 3]) -> f64 {
    let n = norm(a) * norm(b);
    if n > 0.0 { dot(a, b) / n } else { -1.0 }
}

/// The triangles around every vertex, as offsets into one list.
struct Fans {
    offsets: Vec<u32>,
    triangles: Vec<u32>,
}

impl Fans {
    fn build(vertices: usize, triangles: &[[u32; 3]]) -> Result<Fans> {
        let mut offsets: Vec<u32> = filled(vertices + 1, 0)?;
        for tri in triangles {
            for &v in tri {
                offsets[v as usize + 1] += 1;
            }
        }
        for i in 0..vertices {
            offsets[i + 1] += offsets[i];
        }
        let mut fill = filled(vertices, 0u32)?;
        let mut list: Vec<u32> = filled(3 * triangles.len(), 0)?;
        for (t, tri) in triangles.iter().enumerate() {
            for &v in tri {
                let at = offsets[v as usize] + fill[v as usize];
                list[at as usize] = t as u32;
                fill[v as usize] += 1;
            }
        }
        Ok(Fans {
            offsets,
            triangles: list,
        })
    }

    fn of(&self, v: u32) -> &[u32] {
        &self.triangles[self.offsets[v as usize] as usize..self.offsets[v as usize + 1] as usize]
    }

    /// The other triangle on the edge `a -> b` of triangle `t`, if any.
    fn across(&self, triangles: &[[u32; 3]], t: u32, a: u32, b: u32) -> Option<u32> {
        self.of(a)
            .iter()
            .copied()
            .find(|&u| u != t && triangles[u as usize].contains(&b))
    }
}

/// Moves the vertices of `iso` within `max_distance` of `mesh` onto their
/// closest points on it, except where that flips, collapses or folds a
/// triangle (against the unprojected surface) or makes the surface intersect
/// itself. Returns the vertices left moved.
fn project(iso: &mut IsoMesh, mesh: &Mesh, bvh: &Bvh, max_distance: f32) -> Result<usize> {
    let n = iso.positions.len();
    let tris = &iso.triangles;
    let mut original: Vec<[f32; 3]> = Vec::new();
    reserve(&mut original, n)?;
    original.extend_from_slice(&iso.positions);
    let mut moved: Vec<bool> = filled(n, false)?;
    {
        let opts = ClosestOptions { max_distance };
        let mut hits: Vec<Option<ClosestHit>> = filled(QUERY_BATCH.min(n), None)?;
        for (start, chunk) in (0..n)
            .step_by(QUERY_BATCH)
            .zip(original.chunks(QUERY_BATCH))
        {
            let hits = &mut hits[..chunk.len()];
            closest_point_many(mesh, bvh, chunk, &opts, hits)?;
            for (k, hit) in hits.iter().enumerate() {
                if let Some(hit) = hit
                    && hit.point != chunk[k]
                {
                    iso.positions[start + k] = hit.point;
                    moved[start + k] = true;
                }
            }
        }
    }
    if !moved.contains(&true) {
        return Ok(0);
    }
    let fans = Fans::build(n, tris)?;
    let mut queued: Vec<bool> = filled(tris.len(), false)?;
    let mut queue: Vec<u32> = Vec::new();
    let enqueue = |v: u32, queued: &mut [bool], queue: &mut Vec<u32>| -> Result<()> {
        for &t in fans.of(v) {
            if !queued[t as usize] {
                queued[t as usize] = true;
                if queue.len() == queue.capacity() {
                    let grow = queue.len().max(1024);
                    reserve(queue, grow)?;
                }
                queue.push(t);
            }
        }
        Ok(())
    };
    for v in 0..n as u32 {
        if moved[v as usize] {
            enqueue(v, &mut queued, &mut queue)?;
        }
    }
    let revert = |v: u32,
                  positions: &mut [[f32; 3]],
                  moved: &mut [bool],
                  queued: &mut [bool],
                  queue: &mut Vec<u32>|
     -> Result<()> {
        if moved[v as usize] {
            moved[v as usize] = false;
            positions[v as usize] = original[v as usize];
            enqueue(v, queued, queue)?;
        }
        Ok(())
    };
    let mut rounds = 0;
    loop {
        // Flips, collapses and folds, until none is left.
        while let Some(t) = queue.pop() {
            queued[t as usize] = false;
            let tri = tris[t as usize];
            let now = normal(&iso.positions, tri);
            let was = normal(&original, tri);
            let mut bad: Option<u32> = None;
            if cosine(now, was) <= 0.5 || norm(now) < 0.05 * norm(was) {
                bad = Some(t);
            } else {
                for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                    let Some(u) = fans.across(tris, t, a, b) else {
                        continue;
                    };
                    let other = tris[u as usize];
                    let fold_now = cosine(now, normal(&iso.positions, other));
                    let fold_was = cosine(was, normal(&original, other));
                    if fold_now < fold_was.min(0.0) {
                        bad = Some(u);
                        break;
                    }
                }
            }
            if let Some(u) = bad {
                for v in tri.into_iter().chain(tris[u as usize]) {
                    revert(v, &mut iso.positions, &mut moved, &mut queued, &mut queue)?;
                }
            }
        }
        // Then self-intersections.
        let check = Mesh::from_buffers(iso.positions.as_flattened(), tris.as_flattened())?;
        let check_bvh = build_bvh(&check)?;
        let mut pairs = Vec::new();
        self_intersecting_pairs(&check, &check_bvh, &mut pairs)?;
        if pairs.is_empty() {
            break;
        }
        rounds += 1;
        if rounds >= MAX_PROJECT_ROUNDS {
            // The unprojected surface does not intersect itself.
            iso.positions.copy_from_slice(&original);
            return Ok(0);
        }
        for (a, b) in pairs {
            for v in tris[a as usize].into_iter().chain(tris[b as usize]) {
                revert(v, &mut iso.positions, &mut moved, &mut queued, &mut queue)?;
            }
        }
    }
    Ok(moved.iter().filter(|&&m| m).count())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{assert_parallel_identical, box_mesh, punch_holes, uv_sphere};
    use crate::{AnalyzeOptions, analyze};

    fn soup_of(meshes: &[&Mesh]) -> AppendSource {
        let mut soup = AppendSource::default();
        for m in meshes {
            push_soup(m, |p| p.map(f64::from), &mut soup).unwrap();
        }
        soup
    }

    /// Asserts the soup is closed, manifold, consistently and outward wound,
    /// free of degenerate triangles and self-intersections, and returns its
    /// component count.
    fn assert_clean(out: &AppendSource, what: &str) -> usize {
        let mesh =
            Mesh::from_buffers(out.positions.as_flattened(), out.triangles.as_flattened()).unwrap();
        let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
        let a = analyze(
            &mesh,
            &AnalyzeOptions {
                self_intersection: Some(&bvh),
                ..AnalyzeOptions::default()
            },
        )
        .unwrap();
        assert!(
            a.loops.is_empty(),
            "{what}: {} boundary loops",
            a.loops.len()
        );
        assert_eq!(a.non_manifold_edges, 0, "{what}: non-manifold edges");
        assert!(a.pinch_vertices.is_empty(), "{what}: pinch vertices");
        assert_eq!(a.orientation_conflicts, 0, "{what}: orientation conflicts");
        assert_eq!(a.degenerate, 0, "{what}: degenerate triangles");
        assert_eq!(a.duplicate, 0, "{what}: duplicate triangles");
        assert_eq!(a.coincident_groups, 0, "{what}: coincident vertices");
        assert_eq!(a.self_intersects, Some(false), "{what}: self-intersects");
        let mut v6 = 0.0;
        for t in &out.triangles {
            let [p, q, s] = t.map(|v| out.positions[v as usize].map(f64::from));
            v6 += p[0] * (q[1] * s[2] - q[2] * s[1]) - p[1] * (q[0] * s[2] - q[2] * s[0])
                + p[2] * (q[0] * s[1] - q[1] * s[0]);
        }
        assert!(v6 > 0.0, "{what}: wound inward");
        a.components.len()
    }

    #[test]
    fn plan_reports_the_grid_and_refuses_past_the_limit() {
        let options = WrapOptions::builder().gap(0.5).detail(32).build();
        let plan = wrap_plan([0.0; 3], [2.0, 1.0, 1.0], &options).unwrap();
        assert_eq!(plan.voxel_size, 2.0 / 32.0);
        assert_eq!(plan.radius, 0.25);
        // The span plus the radius and three voxels on each side.
        assert_eq!(plan.dims, [32 + 2 * (4 + 3) + 1, 16 + 15, 16 + 15]);
        assert_eq!(plan.voxels, 47 * 31 * 31);
        assert_eq!(plan.bytes, plan.voxels * WRAP_BYTES_PER_VOXEL);
        assert!(plan.fits);

        let small = WrapOptions {
            max_voxels: plan.voxels - 1,
            ..options
        };
        let tight = wrap_plan([0.0; 3], [2.0, 1.0, 1.0], &small).unwrap();
        assert!(!tight.fits);
        let soup = soup_of(&[&box_mesh([0.0; 3], [2.0, 1.0, 1.0])]);
        assert_eq!(
            wrap(&soup, &small).map(|_| ()),
            Err(OpsError::InvalidInput("wrap grid exceeds max_voxels"))
        );

        // A radius under a voxel becomes one voxel.
        let thin = wrap_plan([0.0; 3], [1.0; 3], &WrapOptions::default()).unwrap();
        assert_eq!(thin.radius, thin.voxel_size);
        let huge = WrapOptions::builder().voxel_size(Some(1e-9)).build();
        let plan = wrap_plan([0.0; 3], [1.0; 3], &huge).unwrap();
        assert!(!plan.fits);
    }

    #[test]
    fn invalid_options_and_inputs_are_errors() {
        let soup = soup_of(&[&box_mesh([0.0; 3], [1.0; 3])]);
        for (options, msg) in [
            (
                WrapOptions::builder().gap(-1.0).build(),
                "wrap gap must be finite and not negative",
            ),
            (
                WrapOptions::builder().gap(f32::NAN).build(),
                "wrap gap must be finite and not negative",
            ),
            (
                WrapOptions::builder().voxel_size(Some(0.0)).build(),
                "wrap voxel size must be positive and finite",
            ),
            (
                WrapOptions::builder().detail(4).build(),
                "wrap detail must be 8 to 4096",
            ),
            (
                WrapOptions::builder().max_voxels(10).build(),
                "wrap max_voxels must be at least 64",
            ),
        ] {
            assert_eq!(
                wrap(&soup, &options).map(|_| ()),
                Err(OpsError::InvalidInput(msg))
            );
        }
        let flat = AppendSource {
            positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            triangles: vec![[0, 1, 2], [0, 0, 1]],
        };
        assert_eq!(
            wrap(&flat, &WrapOptions::default()).map(|_| ()),
            Err(OpsError::InvalidInput(
                "wrap needs at least one triangle with area"
            ))
        );
        let bad = AppendSource {
            positions: vec![[0.0; 3]],
            triangles: vec![[0, 1, 2]],
        };
        assert_eq!(
            wrap(&bad, &WrapOptions::default()).map(|_| ()),
            Err(OpsError::InvalidInput("wrap triangle index out of range"))
        );
        let nan = AppendSource {
            positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, f32::NAN, 0.0]],
            triangles: vec![[0, 1, 2]],
        };
        assert_eq!(
            wrap(&nan, &WrapOptions::default()).map(|_| ()),
            Err(OpsError::InvalidInput("wrap needs finite vertex positions"))
        );
    }

    #[test]
    fn an_open_sheet_encloses_nothing() {
        let sheet = AppendSource {
            positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        assert_eq!(
            wrap(&sheet, &WrapOptions::builder().detail(16).build()).map(|_| ()),
            Err(OpsError::InvalidInput(
                "wrap encloses no volume at this gap and voxel size"
            ))
        );
    }

    #[test]
    fn a_holed_sphere_wraps_closed() {
        let sphere = uv_sphere(24, 32, 1.0);
        let holed = punch_holes(&sphere, &[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], 0.2);
        for project in [false, true] {
            let options = WrapOptions::builder()
                .gap(0.5)
                .detail(40)
                .project(project)
                .build();
            let (out, report) = wrap(&soup_of(&[&holed]), &options).unwrap();
            assert_eq!(assert_clean(&out, "holed sphere"), 1);
            assert_eq!(report.vertices, out.positions.len());
            assert_eq!(report.triangles, out.triangles.len());
            assert_eq!(report.projected > 0, project);
        }
    }

    #[test]
    fn wrap_is_identical_with_and_without_parallel() {
        let sphere = uv_sphere(24, 32, 1.0);
        let holed = punch_holes(&sphere, &[[1.0, 0.0, 0.0]], 0.3);
        let other = box_mesh([0.5, -0.5, -0.5], [1.5, 0.5, 0.5]);
        let soup = soup_of(&[&holed, &other]);
        for project in [false, true] {
            let options = WrapOptions::builder()
                .gap(0.4)
                .detail(48)
                .project(project)
                .build();
            let (out, _) = assert_parallel_identical("wrap", || wrap(&soup, &options).unwrap());
            assert_eq!(assert_clean(&out, "sphere and box"), 1);
        }
    }
}
