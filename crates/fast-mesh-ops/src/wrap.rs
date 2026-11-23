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
//!    itself (checked exactly); such vertices stay where marching cubes put
//!    them.
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

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_fields::{DEFAULT_MAX_VOXELS, GridSpec, edt_3d_sq, fh_unsigned_grid};
use fast_mesh_query::{ClosestHit, ClosestOptions, closest_point_many, self_intersecting_pairs};
use fast_mesh_store::{Mesh, StoreError, TriangleId, VertexId};
use robust::{Coord3D, orient3d};

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
pub const WRAP_BYTES_PER_VOXEL: u64 = 17;

/// Empty voxels around the offset surface on every side of the grid.
const PAD_VOXELS: usize = 3;

/// Grid values closer than this many voxels to the iso level are moved off
/// it, so every marching cubes vertex keeps at least about this fraction of
/// a voxel from the grid nodes and no triangle collapses.
const NUDGE: f32 = 0.01;

/// Bound, in voxels, on how far the voxelised distance transform is from the
/// exact distance: `sqrt(3) / 2`, with some slack.
const VOXELISED_ERROR: f32 = 0.9;

/// The voxel size must be at least this many rounding steps of the largest
/// coordinate, so that rounding cannot move a vertex across its lattice edge.
const MIN_VOXEL_ULPS: f64 = 1024.0;

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
    /// Grid nodes in all (`u64::MAX` when the count overflows).
    pub voxels: u64,
    /// Estimated peak memory of the grids, in bytes:
    /// [`WRAP_BYTES_PER_VOXEL`] per voxel. The input, the meshes extracted
    /// and their BVHs come on top.
    pub bytes: u64,
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
/// bounds that are not finite or have `min > max`, bounds without extent and
/// no voxel size, or a voxel size that `f32` positions this far from the
/// origin cannot resolve. A plan over the limit is not an error here: it
/// comes back with `fits == false`.
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
        // Past a billion the plan never fits; kept below overflow.
        dims[k] = (cells.min(1e9) as usize + 1).max(4);
        let center = 0.5 * (f64::from(min[k]) + f64::from(max[k]));
        origin[k] = (center - 0.5 * (dims[k] - 1) as f64 * hd) as f32;
    }
    let voxels = dims.iter().fold(1u64, |n, &d| n.saturating_mul(d as u64));
    let fits = voxels <= options.max_voxels as u64;
    let largest = min
        .iter()
        .chain(&max)
        .fold(0.0f64, |m, &c| m.max(f64::from(c).abs()))
        + margin;
    if fits && hd < MIN_VOXEL_ULPS * f64::from(f32::EPSILON) * largest {
        return Err(OpsError::InvalidInput(
            "wrap voxel size is too small for f32 positions this far from the origin",
        ));
    }
    Ok(WrapPlan {
        dims,
        origin,
        voxel_size: h,
        radius,
        voxels,
        bytes: voxels.saturating_mul(WRAP_BYTES_PER_VOXEL),
        max_voxels: options.max_voxels,
        fits,
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
    let (h, r) = (plan.voxel_size, plan.radius);
    // The work mesh is centred on the middle of its bounds, and so is the grid.
    let spec = GridSpec {
        origin: plan
            .dims
            .map(|n| (-0.5 * (n - 1) as f64 * f64::from(h)) as f32),
        dims: plan.dims,
        voxel_size: h,
    };
    spec.validate(options.max_voxels)?;
    let bvh = build_bvh(&work.mesh)?;

    // 1. and 2. The dilated surface, from the distance to the input.
    let mut grid = fh_unsigned_grid(&work.mesh, spec, options.max_voxels)?.into_values();
    let mut flags: Vec<u8> = filled(grid.len(), 0)?;
    let mut exact = NodeDistances::new(&work.mesh, &bvh, spec, r + 3.0 * h)?;
    flood_outside(&mut grid, &mut flags, r, &mut exact)?;
    drop(exact);
    for (v, &f) in grid.iter_mut().zip(&flags) {
        *v = if f & OUTSIDE != 0 {
            (*v - r).max(NUDGE * h)
        } else {
            (*v - r).min(-NUDGE * h)
        };
    }
    let dilated = marching_cubes_raw(spec.origin, spec.dims, h, &grid, 0.0)?;
    let dilated = Mesh::from_buffers(
        dilated.positions.as_flattened(),
        dilated.triangles.as_flattened(),
    )?;
    let dilated_bvh = build_bvh(&dilated)?;

    // 3. The depth below that surface.
    let mut exact = NodeDistances::new(&dilated, &dilated_bvh, spec, r + 3.0 * h)?;
    erode(&mut grid, &mut flags, r, &mut exact)?;
    drop(exact);
    drop(dilated_bvh);
    drop(dilated);
    drop(flags);

    // 4. Extract.
    let mut iso = marching_cubes_raw(spec.origin, spec.dims, h, &grid, 0.0)?;
    drop(grid);
    if iso.triangles.is_empty() {
        return Err(OpsError::InvalidInput(
            "wrap encloses no volume at this gap and voxel size",
        ));
    }

    // Back to the frame of `source`, before the projection's checks.
    for p in &mut iso.positions {
        *p = std::array::from_fn(|k| (f64::from(p[k]) + work.center[k]) as f32);
    }

    // 5. Project.
    let projected = if options.project {
        project(&mut iso, &work, &bvh, h)?
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

/// Wraps `meshes` together, in one frame with their origins applied, into a
/// new mesh whose origin is the centre of their bounds (see [`wrap`]).
/// Lengths in `options` are local units.
///
/// # Errors
/// [`OpsError::InvalidInput`] for an empty list, a position that does not
/// fit `f32` around the new origin, and as for [`wrap`].
pub fn wrap_meshes(meshes: &[&Mesh], options: &WrapOptions) -> Result<(Mesh, WrapReport)> {
    if meshes.is_empty() {
        return Err(OpsError::InvalidInput("wrap needs at least one mesh"));
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for mesh in meshes {
        let o = mesh.origin();
        extend_bounds(
            mesh,
            |p| std::array::from_fn(|k| o[k] + f64::from(p[k])),
            &mut lo,
            &mut hi,
        );
    }
    if lo[0] > hi[0] {
        return Err(OpsError::InvalidInput(
            "wrap needs at least one triangle with area",
        ));
    }
    let center: [f64; 3] = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
    let mut soup = AppendSource::default();
    for mesh in meshes {
        let o = mesh.origin();
        push_soup(
            mesh,
            |p| std::array::from_fn(|k| o[k] + f64::from(p[k]) - center[k]),
            &mut soup,
        )?;
    }
    let (out, report) = wrap(&soup, options)?;
    drop(soup);
    let mut mesh = Mesh::from_buffers_f64_with_origin(&[], &[], center)?;
    let mut tx = mesh.begin_edit();
    append_mesh(&mut tx, &out)?;
    tx.commit()?;
    Ok((mesh, report))
}

/// Grows the bounds `[lo, hi]` by the positions of the vertices that the
/// live triangles of `mesh` use, mapped by `map`.
pub fn extend_bounds(
    mesh: &Mesh,
    mut map: impl FnMut([f32; 3]) -> [f64; 3],
    lo: &mut [f64; 3],
    hi: &mut [f64; 3],
) {
    for (v, &p) in mesh.positions().iter().enumerate() {
        let v = v as VertexId;
        if mesh.is_vertex_alive(v) && mesh.corner_count(v) > 0 {
            let q = map(p);
            for k in 0..3 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
    }
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

/// The proper triangles of a soup as a compact mesh centred on the middle of
/// their bounds, with those bounds.
struct Work {
    mesh: Mesh,
    min: [f32; 3],
    max: [f32; 3],
    center: [f64; 3],
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
        if !tri
            .iter()
            .flat_map(|&v| source.positions[v as usize])
            .all(f32::is_finite)
        {
            return Err(OpsError::InvalidInput("wrap needs finite vertex positions"));
        }
        if normal(&source.positions, tri) == [0.0; 3] {
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
                grow_push(&mut positions, pos)?;
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
    let center: [f64; 3] = std::array::from_fn(|k| 0.5 * (f64::from(min[k]) + f64::from(max[k])));
    for p in &mut positions {
        *p = std::array::from_fn(|k| (f64::from(p[k]) - center[k]) as f32);
    }
    let mesh = Mesh::from_buffers(positions.as_flattened(), triangles.as_flattened())?;
    Ok(Work {
        mesh,
        min,
        max,
        center,
    })
}

fn build_bvh(mesh: &Mesh) -> Result<Bvh> {
    Ok(Bvh::build(mesh, &BuildOptions::default())?)
}

/// Pushes with a reservation that doubles the capacity.
fn grow_push<T>(v: &mut Vec<T>, value: T) -> Result<()> {
    if v.len() == v.capacity() {
        let grow = v.len().max(64);
        reserve(v, grow)?;
    }
    v.push(value);
    Ok(())
}

/// Exact distances from grid nodes to a mesh, queried in batches.
struct NodeDistances<'a> {
    mesh: &'a Mesh,
    bvh: &'a Bvh,
    spec: GridSpec,
    max_distance: f32,
    points: Vec<[f32; 3]>,
    hits: Vec<Option<ClosestHit>>,
}

impl<'a> NodeDistances<'a> {
    fn new(mesh: &'a Mesh, bvh: &'a Bvh, spec: GridSpec, max_distance: f32) -> Result<Self> {
        let mut points = Vec::new();
        reserve(&mut points, QUERY_BATCH)?;
        Ok(NodeDistances {
            mesh,
            bvh,
            spec,
            max_distance,
            points,
            hits: filled(QUERY_BATCH, None)?,
        })
    }

    /// Calls `write` with each of `nodes`, in order, and its distance to the
    /// mesh (`max_distance` when nothing is that close).
    fn each(&mut self, nodes: &[usize], mut write: impl FnMut(usize, f32)) -> Result<()> {
        let [nx, ny, _] = self.spec.dims;
        let opts = ClosestOptions {
            max_distance: self.max_distance,
        };
        for chunk in nodes.chunks(QUERY_BATCH) {
            self.points.clear();
            for &i in chunk {
                let row = i / nx;
                self.points
                    .push(self.spec.point(i % nx, row % ny, row / ny));
            }
            let hits = &mut self.hits[..chunk.len()];
            closest_point_many(self.mesh, self.bvh, &self.points, &opts, hits)?;
            for (&i, hit) in chunk.iter().zip(hits) {
                write(i, hit.map_or(self.max_distance, |h| h.distance));
            }
        }
        Ok(())
    }
}

/// A node reachable from the grid boundary that is farther than `r` from the
/// input.
const OUTSIDE: u8 = 1;
/// A node whose grid value is exact (or is queued to become so).
const EXACT: u8 = 2;

/// The flat index distance to the next node along each axis.
fn steps([nx, ny, _]: [usize; 3]) -> [usize; 3] {
    [1, nx, nx * ny]
}

/// Calls `f` with the index of every node that is not on the grid boundary,
/// in order.
fn for_each_interior(
    [nx, ny, nz]: [usize; 3],
    mut f: impl FnMut(usize) -> Result<()>,
) -> Result<()> {
    for z in 1..nz - 1 {
        for y in 1..ny - 1 {
            let row = (z * ny + y) * nx;
            for i in row + 1..row + nx - 1 {
                f(i)?;
            }
        }
    }
    Ok(())
}

/// Looks at node `i` from a neighbouring outside node: one that is surely
/// farther than `r` from the input is outside and goes on the stack, one too
/// close to `r` to tell is queued for its exact distance.
fn reach(
    i: usize,
    grid: &[f32],
    flags: &mut [u8],
    (r, band): (f32, f32),
    stack: &mut Vec<usize>,
    queue: &mut Vec<usize>,
) -> Result<()> {
    if flags[i] != 0 {
        return Ok(());
    }
    let d = grid[i];
    if (d - r).abs() <= band {
        flags[i] = EXACT;
        grow_push(queue, i)?;
    } else if d > r {
        flags[i] = OUTSIDE;
        grow_push(stack, i)?;
    }
    Ok(())
}

/// Flags as [`OUTSIDE`] the nodes farther than `r` from the input that a
/// 6-connected walk from the grid boundary through such nodes reaches, and
/// makes `grid` exact at both ends of every lattice edge that leaves them.
///
/// On entry `grid` holds the voxelised distance, off by at most `sqrt(3) / 2`
/// voxels, so only nodes that close to `r` need the exact distance to be told
/// apart, and only those next to an outside node are asked for it.
fn flood_outside(
    grid: &mut [f32],
    flags: &mut [u8],
    r: f32,
    exact: &mut NodeDistances,
) -> Result<()> {
    let dims = exact.spec.dims;
    let h = exact.spec.voxel_size;
    let [nx, ny, nz] = dims;
    let steps = steps(dims);
    let near = (r, VOXELISED_ERROR * h);
    // The boundary is farther than `r + 2 h` from the input.
    for z in 0..nz {
        for y in 0..ny {
            let row = (z * ny + y) * nx;
            if z == 0 || z + 1 == nz || y == 0 || y + 1 == ny {
                flags[row..row + nx].fill(OUTSIDE);
            } else {
                flags[row] = OUTSIDE;
                flags[row + nx - 1] = OUTSIDE;
            }
        }
    }
    let mut stack: Vec<usize> = Vec::new();
    let mut queue: Vec<usize> = Vec::new();
    for_each_interior(dims, |i| {
        let (x, row) = (i % nx, i / nx);
        let (y, z) = (row % ny, row / ny);
        if x == 1 || y == 1 || z == 1 || x + 2 == nx || y + 2 == ny || z + 2 == nz {
            reach(i, grid, flags, near, &mut stack, &mut queue)?;
        }
        Ok(())
    })?;
    loop {
        // Only interior nodes are stacked, so every neighbour exists.
        while let Some(i) = stack.pop() {
            for s in steps {
                reach(i - s, grid, flags, near, &mut stack, &mut queue)?;
                reach(i + s, grid, flags, near, &mut stack, &mut queue)?;
            }
        }
        if queue.is_empty() {
            break;
        }
        queue.sort_unstable();
        exact.each(&queue, |i, d| grid[i] = d)?;
        for &i in &queue {
            if grid[i] > r {
                flags[i] |= OUTSIDE;
                grow_push(&mut stack, i)?;
            }
        }
        queue.clear();
    }
    // The ends of the edges that leave the outside, within `h` of `r`.
    for_each_interior(dims, |i| {
        if flags[i] & OUTSIDE == 0 || grid[i] > r + 2.0 * h {
            return Ok(());
        }
        let mut leaves = false;
        for s in steps {
            for j in [i - s, i + s] {
                if flags[j] & OUTSIDE == 0 {
                    leaves = true;
                    if flags[j] & EXACT == 0 {
                        flags[j] |= EXACT;
                        grow_push(&mut queue, j)?;
                    }
                }
            }
        }
        if leaves && flags[i] & EXACT == 0 {
            flags[i] |= EXACT;
            grow_push(&mut queue, i)?;
        }
        Ok(())
    })?;
    exact.each(&queue, |i, d| grid[i] = d)
}

/// Replaces `grid` by `r` minus the depth below the dilated surface that
/// `exact` measures to: negative deeper than `r`, where the result is. Values
/// keep clear of zero by [`NUDGE`] voxels, and both ends of every lattice edge
/// that changes sign are exact.
fn erode(grid: &mut [f32], flags: &mut [u8], r: f32, exact: &mut NodeDistances) -> Result<()> {
    let dims = exact.spec.dims;
    let h = exact.spec.voxel_size;
    let steps = steps(dims);
    let nudge = NUDGE * h;
    for (v, f) in grid.iter_mut().zip(flags.iter_mut()) {
        *f &= OUTSIDE;
        *v = if *f != 0 { 0.0 } else { f32::INFINITY };
    }
    // The distance to the nearest outside node is at most `sqrt(3)` voxels
    // more than the depth, and never less.
    {
        let mut sq = filled(grid.len(), 0.0f32)?;
        edt_3d_sq(dims, grid, &mut sq)?;
        for ((v, &f), &s) in grid.iter_mut().zip(flags.iter()).zip(&sq) {
            *v = if f != 0 { r } else { r - h * s.sqrt() };
        }
    }
    let mut queue: Vec<usize> = Vec::new();
    for_each_interior(dims, |i| {
        if flags[i] == 0 && grid[i] >= -1.8 * h && grid[i] <= 0.05 * h {
            flags[i] = EXACT;
            grow_push(&mut queue, i)?;
        }
        Ok(())
    })?;
    exact.each(&queue, |i, e| grid[i] = r - e)?;
    queue.clear();
    for v in grid.iter_mut() {
        if *v > -nudge && *v < nudge {
            *v = nudge;
        }
    }
    for_each_interior(dims, |i| {
        if grid[i] > 0.0 || grid[i] < -3.0 * h {
            return Ok(());
        }
        let mut crosses = false;
        for s in steps {
            for j in [i - s, i + s] {
                if grid[j] > 0.0 {
                    crosses = true;
                    if flags[j] == 0 {
                        flags[j] = EXACT;
                        grow_push(&mut queue, j)?;
                    }
                }
            }
        }
        if crosses && flags[i] == 0 {
            flags[i] = EXACT;
            grow_push(&mut queue, i)?;
        }
        Ok(())
    })?;
    exact.each(&queue, |i, e| {
        // Already on its side; the exact value only places the crossing.
        grid[i] = if grid[i] > 0.0 {
            (r - e).max(nudge)
        } else {
            (r - e).min(-nudge)
        };
    })
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

/// Whether two triangles that share exactly one corner also meet elsewhere
/// (exactly, except that coplanar ones are not told).
fn cross_beyond_corner(p: &[[f32; 3]], t: [u32; 3], u: [u32; 3]) -> bool {
    let mut shared = (0..3).filter(|&i| u.contains(&t[i]));
    let (Some(i), None) = (shared.next(), shared.next()) else {
        return false;
    };
    let Some(j) = (0..3).find(|&j| u[j] == t[i]) else {
        return false;
    };
    let at = |v: u32| {
        let [x, y, z] = p[v as usize].map(f64::from);
        Coord3D { x, y, z }
    };
    let (t, u) = (t.map(at), u.map(at));
    // Whether the edge of `a` across from corner `i` meets the triangle `b`.
    let edge_meets = |a: [Coord3D<f64>; 3], i: usize, b: [Coord3D<f64>; 3]| {
        let (q, r) = (a[(i + 1) % 3], a[(i + 2) % 3]);
        let side = [q, r].map(|x| orient3d(b[0], b[1], b[2], x));
        if side[0] * side[1] > 0.0 || side == [0.0; 2] {
            return false;
        }
        let w = [
            orient3d(q, r, b[0], b[1]),
            orient3d(q, r, b[1], b[2]),
            orient3d(q, r, b[2], b[0]),
        ];
        w.iter().all(|&x| x >= 0.0) || w.iter().all(|&x| x <= 0.0)
    };
    edge_meets(t, i, u) || edge_meets(u, j, t)
}

/// Moves the vertices of `iso` (in the frame of the source) within
/// `max_distance` of the input `work` onto their closest points on it, except
/// where that flips, collapses or folds a triangle (against the unprojected
/// surface) or makes the surface intersect itself. Returns the vertices left
/// moved.
fn project(iso: &mut IsoMesh, work: &Work, bvh: &Bvh, max_distance: f32) -> Result<usize> {
    let n = iso.positions.len();
    let tris = &iso.triangles;
    let mut original: Vec<[f32; 3]> = Vec::new();
    reserve(&mut original, n)?;
    original.extend_from_slice(&iso.positions);
    let mut moved: Vec<bool> = filled(n, false)?;
    {
        let opts = ClosestOptions { max_distance };
        let shift = |p: [f32; 3], by: f64| -> [f32; 3] {
            std::array::from_fn(|k| (f64::from(p[k]) + by * work.center[k]) as f32)
        };
        let mut points: Vec<[f32; 3]> = Vec::new();
        reserve(&mut points, QUERY_BATCH.min(n))?;
        let mut hits: Vec<Option<ClosestHit>> = filled(QUERY_BATCH.min(n), None)?;
        for start in (0..n).step_by(QUERY_BATCH) {
            let chunk = &original[start..(start + QUERY_BATCH).min(n)];
            points.clear();
            points.extend(chunk.iter().map(|&p| shift(p, -1.0)));
            let hits = &mut hits[..chunk.len()];
            closest_point_many(&work.mesh, bvh, &points, &opts, hits)?;
            for (k, hit) in hits.iter().enumerate() {
                if let Some(hit) = hit {
                    let to = shift(hit.point, 1.0);
                    if to != chunk[k] {
                        iso.positions[start + k] = to;
                        moved[start + k] = true;
                    }
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
        // Flips, collapses, folds and crossings of triangles that share a
        // corner, until none is left.
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
            if bad.is_none() {
                bad = tri.iter().find_map(|&v| {
                    fans.of(v)
                        .iter()
                        .copied()
                        .find(|&u| cross_beyond_corner(&iso.positions, tri, tris[u as usize]))
                });
            }
            if let Some(u) = bad {
                for v in tri.into_iter().chain(tris[u as usize]) {
                    revert(v, &mut iso.positions, &mut moved, &mut queued, &mut queue)?;
                }
            }
        }
        // Then crossings of triangles that share none.
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
        assert_eq!(plan.origin, [-0.4375, -0.4375, -0.4375]);
        assert_eq!(plan.bytes, plan.voxels * WRAP_BYTES_PER_VOXEL);
        assert!(plan.fits);

        let small = WrapOptions {
            max_voxels: plan.voxels as usize - 1,
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
    fn running_out_of_memory_is_an_error() {
        let soup = soup_of(&[&box_mesh([0.0; 3], [1.0; 3])]);
        let options = WrapOptions::builder()
            .gap(0.3)
            .detail(8)
            .project(true)
            .build();
        let expected = wrap(&soup, &options).unwrap();
        let mut failed = 0;
        loop {
            crate::fail_allocations_after(Some(failed));
            let got = wrap(&soup, &options);
            crate::fail_allocations_after(None);
            match got {
                Err(e) => assert_eq!(e, OpsError::Store(StoreError::OutOfMemory)),
                Ok(got) => {
                    assert_eq!(got, expected);
                    break;
                }
            }
            failed += 1;
        }
        assert!(failed > 10, "{failed} reservations");
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
