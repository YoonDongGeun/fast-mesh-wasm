//! Lattice infill: a periodic minimal-surface lattice inside a closed mesh.
//!
//! [`lattice_fill`] replaces the inside of a closed, outward-wound mesh, or
//! the cavity of a hollowed one, by walls of constant thickness that follow a
//! gyroid, Schwarz P or diamond surface ([`LatticeKind`]):
//!
//! - **Open** (the inside, without a shell): the mesh becomes the walls, cut
//!   off at its surface, so their rims show on the outside.
//! - **In a shell** (the inside, with [`LatticeOptions::shell_thickness`]):
//!   the mesh's own surface is kept as it is, bit for bit, and the space
//!   between the walls is carved out of it from that depth on, the way
//!   [`hollow_mesh`](crate::boolean::hollow_mesh) carves one cavity. Shell and
//!   walls are one solid.
//! - **In a cavity** ([`LatticeRegion::Cavity`]): the surfaces that enclose
//!   the cavities of a hollowed mesh are replaced by those of the space
//!   between the walls, so the walls grow out of the existing shell. The
//!   shell thickness then comes on top of that shell.
//!
//! The result is closed, 2-manifold, consistently wound and free of
//! self-intersections, as far as the surface that is kept is.
//!
//! How it works:
//!
//! 1. **Clip distance.** On a regular grid, the depth minus the distance
//!    inside the region's boundary (signed, through a BVH), exact on the
//!    nodes next to the clip surface. Everywhere else blocks of 8³, 4³ and 2³
//!    nodes are decided by their centres, so the work grows with the surface
//!    and not with the volume.
//! 2. **Walls.** The distance to the implicit surface, from its value,
//!    gradient and second derivative, and near the walls' faces from the
//!    closest point on it. The thickness is uniform, unlike that of a band of
//!    the implicit function's values.
//! 3. **Extract** the intersection of the two with the Manifold Marching Cubes
//!    of [`mod@crate::marching_cubes`], and move the vertices next to the clip
//!    surface along their lattice edges onto it. Marching cubes is free of
//!    self-intersections wherever on their edges the vertices sit.
//! 4. **Leave out** loose pieces that enclose less than
//!    [`LatticeOptions::min_fragment`] of a cell.
//!
//! Lengths are measured after [`LatticeOptions::scale`] is applied to the
//! stored positions, and the cells are aligned with the mesh's own axes,
//! around the middle of its bounds.
//!
//! The grid is planned before anything is allocated: [`lattice_plan`] reports
//! its dimensions, voxel count and estimated triangles and memory, and
//! [`lattice_fill`] refuses a grid over [`LatticeOptions::max_voxels`], and
//! one too coarse for its walls. [`lattice_preview`] builds the walls on a
//! grid as small as asked for, thickening them if it has to. With the
//! `parallel` feature, the distances, the field and marching cubes run on
//! rayon's pool; the result is bit-identical with and without the feature,
//! for any thread count.

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_fields::{DEFAULT_MAX_VOXELS, GridSpec, SignData, SignMode, signed_distances};
use fast_mesh_query::{ClosestHit, ClosestOptions, closest_point_many};
use fast_mesh_store::{Mesh, StoreError, Transaction, TriangleId, UV_CHANNEL, VertexId};

use crate::append::AppendSource;
use crate::offset::classify_solid;
use crate::topology::UnionFind;
use crate::{OpsError, Result, allocation_allowed, filled, reserve};

mod grid;
mod surface;
#[cfg(test)]
mod tests;

use grid::{Boundary, Extract, Job, Walls, drop_fragments, grow_push, volume6};
pub use surface::LatticeKind;

/// Bytes per voxel [`LatticePlan::bytes`] counts: the field (`f32`) and the
/// flags of the nodes next to the clip surface (`u8`).
pub const LATTICE_BYTES_PER_VOXEL: u64 = 5;

/// Bytes per estimated triangle [`LatticePlan::bytes`] counts, for the
/// extraction and the result.
pub const LATTICE_BYTES_PER_TRIANGLE: u64 = 64;

/// The fewest voxels a wall is thick, or a shell around an inside.
pub const MIN_VOXELS_PER_WALL: f32 = 2.0;

/// The fewest voxels a cell is long.
pub const MIN_VOXELS_PER_CELL: f32 = 8.0;

/// Voxels per wall when [`LatticeOptions::voxel_size`] is `None`.
const AUTO_VOXELS_PER_WALL: f32 = 2.5;

/// Voxels per cell, at least, when [`LatticeOptions::voxel_size`] is `None`.
const AUTO_VOXELS_PER_CELL: f32 = 16.0;

/// Voxels per cell a preview goes down to before it shows less of the part.
const PREVIEW_VOXELS_PER_CELL: f32 = 10.0;

/// Empty voxels around the mesh on every side of the grid.
const PAD_VOXELS: f64 = 2.0;

/// The voxel size must be at least this many rounding steps of the largest
/// coordinate, so that rounding cannot move a vertex across its lattice edge.
const MIN_VOXEL_ULPS: f64 = 1024.0;

/// Points per closest-point batch.
const QUERY_BATCH: usize = 16_384;

/// What a lattice fills.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LatticeRegion {
    /// The cavities when the mesh encloses one, and the inside otherwise.
    #[default]
    Auto,
    /// The inside of the mesh: its material.
    Inside,
    /// The cavities the mesh encloses: the space inside its components that
    /// are wound inward, such as those of a hollowed mesh. A cavity that a
    /// hole opens to the outside is not enclosed; fill it before the hole.
    Cavity,
}

/// Options for [`lattice_fill`], [`lattice_plan`] and [`lattice_preview`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatticeOptions {
    /// The surface the walls follow.
    pub kind: LatticeKind,
    /// The edge length of one cell. Positive and finite; default `10`.
    pub cell_size: f32,
    /// The thickness of the walls. Positive, finite and at most a third of
    /// the cell size; default `1`.
    pub wall_thickness: f32,
    /// How thick a solid shell is left inside the region's boundary before
    /// the lattice starts. Finite and not negative; default `0`: none, and
    /// the inside becomes an open lattice.
    pub shell_thickness: f32,
    /// What is filled. Default [`LatticeRegion::Auto`].
    pub region: LatticeRegion,
    /// The grid spacing. Positive and finite when set; `None` (the default)
    /// derives it from the thickness of the walls and the shell and from the
    /// cell size. At least two voxels make a wall and the shell of an
    /// inside, and eight a cell.
    pub voxel_size: Option<f32>,
    /// The largest grid accepted, in voxels (at least 64); a larger plan is
    /// refused before anything is allocated. Default
    /// [`DEFAULT_MAX_VOXELS`] (`256³`).
    pub max_voxels: usize,
    /// Connected pieces of the result that enclose less than this share of a
    /// cell's volume are left out: loose bits of wall at the surface of an
    /// open lattice, and pockets too small to matter in a shell. Finite and
    /// not negative; default `0.01`.
    pub min_fragment: f32,
    /// The scale of the stored positions, per axis, before lengths are
    /// measured. Positive and finite; default `1` each.
    pub scale: [f64; 3],
}

impl Default for LatticeOptions {
    fn default() -> Self {
        LatticeOptions {
            kind: LatticeKind::Gyroid,
            cell_size: 10.0,
            wall_thickness: 1.0,
            shell_thickness: 0.0,
            region: LatticeRegion::Auto,
            voxel_size: None,
            max_voxels: DEFAULT_MAX_VOXELS,
            min_fragment: 0.01,
            scale: [1.0; 3],
        }
    }
}

impl LatticeOptions {
    /// A builder for [`LatticeOptions`]: every option as [`Default`] until
    /// set.
    pub fn builder() -> LatticeOptionsBuilder {
        LatticeOptionsBuilder::default()
    }
}

/// A builder for [`LatticeOptions`], see [`LatticeOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct LatticeOptionsBuilder {
    options: LatticeOptions,
}

impl LatticeOptionsBuilder {
    /// The surface the walls follow.
    pub fn kind(mut self, kind: LatticeKind) -> Self {
        self.options.kind = kind;
        self
    }

    /// The edge length of one cell.
    pub fn cell_size(mut self, cell_size: f32) -> Self {
        self.options.cell_size = cell_size;
        self
    }

    /// The thickness of the walls.
    pub fn wall_thickness(mut self, wall_thickness: f32) -> Self {
        self.options.wall_thickness = wall_thickness;
        self
    }

    /// The thickness of the shell; `0` for none.
    pub fn shell_thickness(mut self, shell_thickness: f32) -> Self {
        self.options.shell_thickness = shell_thickness;
        self
    }

    /// What is filled.
    pub fn region(mut self, region: LatticeRegion) -> Self {
        self.options.region = region;
        self
    }

    /// The grid spacing; `None` derives it from the lengths.
    pub fn voxel_size(mut self, voxel_size: Option<f32>) -> Self {
        self.options.voxel_size = voxel_size;
        self
    }

    /// The largest grid accepted, in voxels.
    pub fn max_voxels(mut self, max_voxels: usize) -> Self {
        self.options.max_voxels = max_voxels;
        self
    }

    /// The smallest piece kept, as a share of a cell's volume.
    pub fn min_fragment(mut self, min_fragment: f32) -> Self {
        self.options.min_fragment = min_fragment;
        self
    }

    /// The scale of the stored positions, per axis.
    pub fn scale(mut self, scale: [f64; 3]) -> Self {
        self.options.scale = scale;
        self
    }

    /// The options as set.
    pub fn build(self) -> LatticeOptions {
        self.options
    }
}

/// The voxel grid a lattice uses and what to expect of it, as
/// [`lattice_plan`] computes it before anything is allocated. Lengths are
/// measured with [`LatticeOptions::scale`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatticePlan {
    /// What is filled: never [`LatticeRegion::Auto`].
    pub region: LatticeRegion,
    /// Grid nodes along x, y and z.
    pub dims: [usize; 3],
    /// The grid spacing.
    pub voxel_size: f32,
    /// The thickness of the walls: [`LatticeOptions::wall_thickness`], except
    /// in a preview on a grid too coarse for it.
    pub wall_thickness: f32,
    /// Grid nodes in all (`u64::MAX` when the count overflows).
    pub voxels: u64,
    /// Estimated triangles of the extracted surface.
    pub triangles: u64,
    /// Estimated peak memory of the grid, the extraction and the result, in
    /// bytes: [`LATTICE_BYTES_PER_VOXEL`] per voxel and
    /// [`LATTICE_BYTES_PER_TRIANGLE`] per triangle. The input and its BVH
    /// come on top.
    pub bytes: u64,
    /// The limit the plan was checked against.
    pub max_voxels: usize,
    /// Whether `voxels <= max_voxels`; [`lattice_fill`] refuses the plan
    /// otherwise.
    pub fits: bool,
    /// The share of space the walls fill
    /// ([`LatticeKind::solid_fraction`]), without any shell.
    pub solid_fraction: f64,
    /// The volume of the region that is filled.
    pub region_volume: f64,
    /// Whether the grid covers only the middle of the mesh (a preview of many
    /// small cells).
    pub cropped: bool,
}

/// What [`lattice_fill`] did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatticeReport {
    /// The grid used.
    pub plan: LatticePlan,
    /// Vertices added.
    pub vertices: usize,
    /// Triangles added.
    pub triangles: usize,
    /// Loose pieces left out ([`LatticeOptions::min_fragment`]).
    pub fragments: usize,
    /// The volume of material in the region now, walls and shell, measured
    /// with [`LatticeOptions::scale`]. Divided by
    /// [`LatticePlan::region_volume`] it is the share that is left.
    pub solid_volume: f64,
    /// Whether the mesh's UVs were dropped.
    pub uvs_dropped: bool,
}

/// What [`lattice_preview`] built.
#[derive(Debug, Clone, PartialEq)]
pub struct LatticePreview {
    /// The walls, closed, in the mesh's stored local frame. Empty when there
    /// are none.
    pub walls: AppendSource,
    /// The grid used, with the thickness the walls are drawn with and whether
    /// it shows the middle of the mesh only.
    pub plan: LatticePlan,
}

/// Plans the grid of [`lattice_fill`] for `mesh` and `options`, without
/// allocating it. Costs a few passes over the vertices and triangles.
///
/// # Errors
/// As for [`lattice_fill`], except that the mesh is not checked for being
/// closed, and that a grid over the limit comes back with
/// [`LatticePlan::fits`] false.
pub fn lattice_plan(mesh: &Mesh, options: &LatticeOptions) -> Result<LatticePlan> {
    validate(options)?;
    allocation_allowed()?;
    let survey = Survey::of(mesh, options)?;
    let h = fine_voxel(options, survey.region)?;
    plan_at(&survey, options, h, options.wall_thickness)
}

/// Fills the inside of the closed, outward-wound mesh in `tx`, or the
/// cavities it encloses, with a lattice (see the module documentation).
///
/// # Errors
/// [`OpsError::InvalidInput`], before anything is written, for invalid
/// options; a mesh that is not closed and consistently wound, or wound
/// inward; [`LatticeRegion::Cavity`] for a mesh that encloses none; a grid
/// over [`LatticeOptions::max_voxels`] (`"lattice grid exceeds max_voxels"`;
/// [`lattice_plan`] has the numbers), too coarse for the walls, the shell or
/// the cells, for the wall around a cavity, or for `f32` positions this far
/// from the origin; and a lattice that leaves nothing, or nothing open.
/// [`OpsError::Store`] with [`StoreError::OutOfMemory`] when memory runs out.
pub fn lattice_fill(tx: &mut Transaction<'_>, options: &LatticeOptions) -> Result<LatticeReport> {
    validate(options)?;
    allocation_allowed()?;
    let survey = Survey::of(tx.mesh(), options)?;
    let h = fine_voxel(options, survey.region)?;
    let plan = plan_at(&survey, options, h, options.wall_thickness)?;
    if !plan.fits {
        return Err(OpsError::InvalidInput("lattice grid exceeds max_voxels"));
    }
    let open = survey.region == LatticeRegion::Inside && options.shell_thickness == 0.0;
    let extract = if open { Extract::Walls } else { Extract::Void };
    let outer = match survey.region {
        LatticeRegion::Cavity => Some(survey.outer()?),
        _ => None,
    };
    let (mut out, at_clip) = survey.extract(options, &plan, extract)?;
    let cell = f64::from(options.cell_size);
    let kept = drop_fragments(&mut out, f64::from(options.min_fragment) * cell.powi(3))?;
    if out.triangles.is_empty() {
        return Err(OpsError::InvalidInput(if open {
            "lattice leaves no walls; use smaller cells or thicker walls"
        } else {
            "lattice leaves no open space; use thinner walls, a thinner shell or larger cells"
        }));
    }
    if let Some(outer) = &outer {
        // A triangle lies within a voxel of its nearest corner, and only
        // corners next to the clip surface can be that close to the outside.
        let mut rim: Vec<[f32; 3]> = Vec::new();
        reserve(&mut rim, at_clip.len())?;
        rim.extend(
            at_clip
                .iter()
                .map(|&v| kept.vertices[v as usize])
                .filter(|&v| v != u32::MAX)
                .map(|v| out.positions[v as usize]),
        );
        check_wall(outer, &rim, plan.voxel_size)?;
    }
    let solid_volume = if open {
        kept.volume
    } else {
        survey.volume - kept.volume
    };
    let mut added = AppendSource {
        positions: out.positions,
        triangles: out.triangles,
    };
    survey.to_stored(&mut added.positions);
    if !open {
        // The surface of the void, seen from the solid around it.
        for t in &mut added.triangles {
            t.swap(1, 2);
        }
    }
    let (vertices, triangles) = (added.positions.len(), added.triangles.len());
    let uvs_dropped = survey.write(tx, added, open)?;
    Ok(LatticeReport {
        plan,
        vertices,
        triangles,
        fragments: kept.dropped,
        solid_volume,
        uvs_dropped,
    })
}

/// The walls of the lattice that [`lattice_fill`] would put into `mesh`, on a
/// grid of at most `max_voxels` voxels (at least 64) instead of
/// [`LatticeOptions::max_voxels`], without changing the mesh.
///
/// The grid is as fine as that allows, and no finer than the fill's. Walls it
/// is too coarse for are drawn two voxels thick, and where even cells of ten
/// voxels do not fit, the preview shows the middle of the mesh only
/// ([`LatticePlan::cropped`]). With a shell, the walls start at its inner
/// face; the shell itself is not part of the preview.
///
/// # Errors
/// As for [`lattice_fill`], except that nothing is checked against
/// [`LatticeOptions::max_voxels`], the wall around a cavity is not checked,
/// and a lattice without walls is an empty preview.
pub fn lattice_preview(
    mesh: &Mesh,
    options: &LatticeOptions,
    max_voxels: usize,
) -> Result<LatticePreview> {
    let options = &LatticeOptions {
        max_voxels,
        ..*options
    };
    validate(options)?;
    allocation_allowed()?;
    let survey = Survey::of(mesh, options)?;
    let plan = preview_plan(&survey, options)?;
    let (mut out, _) = survey.extract(options, &plan, Extract::Walls)?;
    let cell = f64::from(options.cell_size);
    drop_fragments(&mut out, f64::from(options.min_fragment) * cell.powi(3))?;
    survey.to_stored(&mut out.positions);
    Ok(LatticePreview {
        walls: AppendSource {
            positions: out.positions,
            triangles: out.triangles,
        },
        plan,
    })
}

fn validate(o: &LatticeOptions) -> Result<()> {
    let positive = |v: f32| v.is_finite() && v > 0.0;
    if !positive(o.cell_size) {
        return Err(OpsError::InvalidInput(
            "lattice cell size must be positive and finite",
        ));
    }
    if !positive(o.wall_thickness) {
        return Err(OpsError::InvalidInput(
            "lattice wall thickness must be positive and finite",
        ));
    }
    if f64::from(o.wall_thickness) > f64::from(o.cell_size) / 3.0 * (1.0 + 1e-6) {
        return Err(OpsError::InvalidInput(
            "lattice walls must not be thicker than a third of the cell size",
        ));
    }
    if !o.shell_thickness.is_finite() || o.shell_thickness < 0.0 {
        return Err(OpsError::InvalidInput(
            "lattice shell thickness must be finite and not negative",
        ));
    }
    if o.voxel_size.is_some_and(|h| !positive(h)) {
        return Err(OpsError::InvalidInput(
            "lattice voxel size must be positive and finite",
        ));
    }
    if o.max_voxels < 64 {
        return Err(OpsError::InvalidInput(
            "lattice max_voxels must be at least 64",
        ));
    }
    if !o.min_fragment.is_finite() || o.min_fragment < 0.0 {
        return Err(OpsError::InvalidInput(
            "lattice min_fragment must be finite and not negative",
        ));
    }
    if !o.scale.iter().all(|s| s.is_finite() && *s > 0.0) {
        return Err(OpsError::InvalidInput(
            "lattice scale must be positive and finite",
        ));
    }
    Ok(())
}

/// The voxel size of the full lattice.
fn fine_voxel(o: &LatticeOptions, region: LatticeRegion) -> Result<f32> {
    // The shell of an inside keeps the void's surface off the mesh's own.
    let shell =
        (region == LatticeRegion::Inside && o.shell_thickness > 0.0).then_some(o.shell_thickness);
    let Some(h) = o.voxel_size else {
        let h = (o.wall_thickness / AUTO_VOXELS_PER_WALL).min(o.cell_size / AUTO_VOXELS_PER_CELL);
        return Ok(shell.map_or(h, |s| h.min(s / MIN_VOXELS_PER_WALL)));
    };
    let spans = |length: f32, voxels: f32| length >= voxels * h * (1.0 - 1e-4);
    if !spans(o.wall_thickness, MIN_VOXELS_PER_WALL) {
        return Err(OpsError::InvalidInput(
            "lattice walls are thinner than two voxels; use a smaller voxel size or thicker walls",
        ));
    }
    if !spans(o.cell_size, MIN_VOXELS_PER_CELL) {
        return Err(OpsError::InvalidInput(
            "lattice cells are shorter than eight voxels; use a smaller voxel size or larger cells",
        ));
    }
    if shell.is_some_and(|s| !spans(s, MIN_VOXELS_PER_WALL)) {
        return Err(OpsError::InvalidInput(
            "lattice shell is thinner than two voxels; use a smaller voxel size or a thicker shell",
        ));
    }
    Ok(h)
}

/// Grid nodes along each axis for a mesh of these extents.
fn grid_dims(extent: [f64; 3], h: f32) -> [usize; 3] {
    extent.map(|e| {
        let cells = (e / f64::from(h) + 2.0 * PAD_VOXELS).ceil();
        // Past a billion the plan never fits; kept below overflow.
        cells.min(1e9) as usize + 1
    })
}

fn count(dims: [usize; 3]) -> u64 {
    dims.iter().fold(1u64, |n, &d| n.saturating_mul(d as u64))
}

/// The plan for a grid with spacing `h` over the whole mesh.
fn plan_at(survey: &Survey, o: &LatticeOptions, h: f32, wall: f32) -> Result<LatticePlan> {
    let dims = grid_dims(survey.extent, h);
    let voxels = count(dims);
    let fits = voxels <= o.max_voxels as u64;
    if fits && !survey.resolves(h) {
        return Err(OpsError::InvalidInput(
            "lattice voxel size is too small for f32 positions this far from the origin",
        ));
    }
    let (cell, t) = (f64::from(o.cell_size), f64::from(wall));
    let solid_fraction = o.kind.solid_fraction(cell, t);
    // Marching cubes makes about three triangles per voxel face of area. The
    // walls' faces, in what the shell leaves, and the clip surface between
    // them.
    let depth = f64::from(o.shell_thickness);
    let left = (survey.volume - depth * survey.area).max(0.0);
    let open = survey.region == LatticeRegion::Inside && depth == 0.0;
    let share = if open {
        solid_fraction
    } else {
        1.0 - solid_fraction
    };
    let area = o.kind.wall_area(cell, t) * left + share * survey.area;
    let triangles = (3.0 * area / f64::from(h).powi(2)).min(1e18) as u64;
    Ok(LatticePlan {
        region: survey.region,
        dims,
        voxel_size: h,
        wall_thickness: wall,
        voxels,
        triangles,
        bytes: voxels
            .saturating_mul(LATTICE_BYTES_PER_VOXEL)
            .saturating_add(triangles.saturating_mul(LATTICE_BYTES_PER_TRIANGLE)),
        max_voxels: o.max_voxels,
        fits,
        solid_fraction,
        region_volume: survey.volume,
        cropped: false,
    })
}

/// The plan of a preview: as fine as the limit allows, then cropped.
fn preview_plan(survey: &Survey, o: &LatticeOptions) -> Result<LatticePlan> {
    let coarse = o.cell_size / PREVIEW_VOXELS_PER_CELL;
    let fine = match o.voxel_size {
        Some(h) => h,
        None => fine_voxel(o, survey.region)?,
    }
    .min(coarse);
    let fits = |h: f32| count(grid_dims(survey.extent, h)) <= o.max_voxels as u64;
    let mut h = fine;
    while !fits(h) && h < coarse {
        h = (h * 1.05).min(coarse);
    }
    let wall = o.wall_thickness.max(MIN_VOXELS_PER_WALL * h);
    let mut plan = plan_at(survey, o, h, wall)?;
    if !plan.fits {
        if !survey.resolves(h) {
            return Err(OpsError::InvalidInput(
                "lattice voxel size is too small for f32 positions this far from the origin",
            ));
        }
        // The longest sides first, down to a cube.
        let mut lo = 4usize;
        let mut hi = plan.dims.into_iter().max().unwrap_or(4);
        let cut = |n: usize| plan.dims.map(|d| d.min(n));
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if count(cut(mid)) <= o.max_voxels as u64 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let share = count(cut(lo)) as f64 / plan.voxels as f64;
        plan.dims = cut(lo);
        plan.voxels = count(plan.dims);
        plan.triangles = (plan.triangles as f64 * share) as u64;
        plan.bytes = plan.voxels * LATTICE_BYTES_PER_VOXEL
            + plan.triangles.saturating_mul(LATTICE_BYTES_PER_TRIANGLE);
        plan.fits = true;
        plan.cropped = true;
    }
    Ok(plan)
}

/// A connected part of a mesh, in the measured frame.
struct Part {
    /// Its smallest vertex.
    first: u32,
    /// Its signed volume.
    volume: f64,
    lo: [f32; 3],
    hi: [f32; 3],
    /// Whether it bounds a cavity that is filled.
    cavity: bool,
}

impl Part {
    /// Whether the bounds of `other` are within the bounds of this part.
    fn holds(&self, other: &Part) -> bool {
        (0..3).all(|k| self.lo[k] <= other.lo[k] && other.hi[k] <= self.hi[k])
    }
}

/// What a mesh is, in the frame lengths are measured in: its scaled positions
/// around the middle of its bounds.
struct Survey {
    positions: Vec<[f32; 3]>,
    /// The mesh vertex each position is from.
    vertices: Vec<VertexId>,
    triangles: Vec<[u32; 3]>,
    /// The mesh triangle each triangle is from.
    sources: Vec<TriangleId>,
    /// Whether each triangle bounds the region that is filled.
    bounds_region: Vec<bool>,
    /// What is filled: never [`LatticeRegion::Auto`].
    region: LatticeRegion,
    /// The volume and the boundary's area of the region.
    volume: f64,
    area: f64,
    extent: [f64; 3],
    /// The middle of the bounds, in the stored frame.
    center: [f64; 3],
    scale: [f64; 3],
    /// The largest stored coordinate along each axis.
    largest: [f64; 3],
    /// Vertices at the same place (the two sides of a UV seam) are one here.
    welded: bool,
}

impl Survey {
    fn of(mesh: &Mesh, options: &LatticeOptions) -> Result<Survey> {
        let stored = mesh.positions();
        let roots = match mesh.uvs() {
            Some(_) => Some(crate::weld::weld_roots(mesh, 0.0)?),
            None => None,
        };
        let mut remap: Vec<u32> = filled(mesh.vertex_slots(), u32::MAX)?;
        let mut s = Survey {
            positions: Vec::new(),
            vertices: Vec::new(),
            triangles: Vec::new(),
            sources: Vec::new(),
            bounds_region: Vec::new(),
            region: LatticeRegion::Inside,
            volume: 0.0,
            area: 0.0,
            extent: [0.0; 3],
            center: [0.0; 3],
            scale: options.scale,
            largest: [0.0; 3],
            welded: false,
        };
        reserve(&mut s.triangles, mesh.triangle_count())?;
        reserve(&mut s.sources, mesh.triangle_count())?;
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for t in 0..mesh.triangle_slots() as TriangleId {
            let Some(raw) = mesh.triangle(t) else {
                continue;
            };
            let tri = match &roots {
                Some(r) => raw.map(|v| r[v as usize]),
                None => raw,
            };
            s.welded |= tri != raw;
            let [a, b, c] = tri;
            if a == b || b == c || a == c {
                continue;
            }
            let mut out = [0u32; 3];
            for (slot, v) in out.iter_mut().zip(tri) {
                if remap[v as usize] == u32::MAX {
                    let p = stored[v as usize].map(f64::from);
                    if !p.iter().all(|c| c.is_finite()) {
                        return Err(OpsError::InvalidInput(
                            "lattice needs finite vertex positions",
                        ));
                    }
                    for k in 0..3 {
                        lo[k] = lo[k].min(p[k]);
                        hi[k] = hi[k].max(p[k]);
                    }
                    remap[v as usize] = s.vertices.len() as u32;
                    grow_push(&mut s.vertices, v)?;
                }
                *slot = remap[v as usize];
            }
            s.triangles.push(out);
            s.sources.push(t);
        }
        if s.triangles.is_empty() {
            return Err(OpsError::InvalidInput(
                "lattice needs a closed mesh with triangles",
            ));
        }
        s.center = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
        s.extent = std::array::from_fn(|k| (hi[k] - lo[k]) * s.scale[k]);
        s.largest = std::array::from_fn(|k| lo[k].abs().max(hi[k].abs()));
        reserve(&mut s.positions, s.vertices.len())?;
        for &v in &s.vertices {
            let p: [f32; 3] = std::array::from_fn(|k| {
                ((f64::from(stored[v as usize][k]) - s.center[k]) * s.scale[k]) as f32
            });
            if !p.iter().all(|c| c.is_finite()) {
                return Err(OpsError::InvalidInput("lattice scale is too large"));
            }
            s.positions.push(p);
        }
        s.choose_region(options.region)?;
        Ok(s)
    }

    /// Tells the cavities' surfaces from the outer ones, and which are filled.
    fn choose_region(&mut self, wanted: LatticeRegion) -> Result<()> {
        let (part_of, mut parts) = self.parts()?;
        let total: f64 = parts.iter().map(|p| p.volume).sum();
        if total <= 0.0 {
            return Err(OpsError::InvalidInput(
                "lattice needs a closed mesh that encloses a volume and is wound outward",
            ));
        }
        let cavities = if wanted == LatticeRegion::Inside {
            false
        } else {
            self.mark_cavities(&part_of, &mut parts)?
        };
        self.region = match wanted {
            LatticeRegion::Cavity if !cavities => {
                return Err(OpsError::InvalidInput(
                    "lattice found no cavity enclosed by the mesh",
                ));
            }
            _ if cavities => LatticeRegion::Cavity,
            _ => LatticeRegion::Inside,
        };
        reserve(&mut self.bounds_region, self.triangles.len())?;
        for &t in &self.triangles {
            let bounds = !cavities || parts[part_of[t[0] as usize] as usize].cavity;
            self.bounds_region.push(bounds);
            if bounds {
                self.area += triangle_area(&self.positions, t);
            }
        }
        self.volume = if cavities {
            -parts
                .iter()
                .filter(|p| p.cavity)
                .map(|p| p.volume)
                .sum::<f64>()
        } else {
            total
        };
        Ok(())
    }

    /// The part of every vertex, and the connected parts of the mesh.
    fn parts(&self) -> Result<(Vec<u32>, Vec<Part>)> {
        let n = self.positions.len();
        let mut sets = UnionFind::new();
        sets.reset(n)?;
        for &[a, b, c] in &self.triangles {
            sets.union(a, b);
            sets.union(a, c);
        }
        let mut part_of: Vec<u32> = filled(n, u32::MAX)?;
        let mut parts: Vec<Part> = Vec::new();
        for (v, p) in self.positions.iter().enumerate() {
            // A root is its set's smallest vertex, so it comes first.
            let root = sets.find(v as u32) as usize;
            if root == v {
                part_of[v] = parts.len() as u32;
                grow_push(
                    &mut parts,
                    Part {
                        first: v as u32,
                        volume: 0.0,
                        lo: *p,
                        hi: *p,
                        cavity: false,
                    },
                )?;
            } else {
                part_of[v] = part_of[root];
            }
            let part = &mut parts[part_of[v] as usize];
            for (k, &c) in p.iter().enumerate() {
                part.lo[k] = part.lo[k].min(c);
                part.hi[k] = part.hi[k].max(c);
            }
        }
        for &t in &self.triangles {
            parts[part_of[t[0] as usize] as usize].volume += volume6(&self.positions, t) / 6.0;
        }
        Ok((part_of, parts))
    }

    /// Marks the parts that bound cavities: those wound inward inside a solid
    /// part, and the solids that float in one of those. Returns whether there
    /// are any.
    fn mark_cavities(&self, part_of: &[u32], parts: &mut [Part]) -> Result<bool> {
        for i in 0..parts.len() {
            parts[i].cavity =
                parts[i].volume < 0.0 && parts.iter().any(|p| p.volume > 0.0 && p.holds(&parts[i]));
        }
        if !parts.iter().any(|p| p.cavity) {
            return Ok(false);
        }
        let mut solids: Vec<usize> = Vec::new();
        for (i, part) in parts.iter().enumerate() {
            if part.volume > 0.0 && parts.iter().any(|p| p.cavity && p.holds(part)) {
                grow_push(&mut solids, i)?;
            }
        }
        if solids.is_empty() {
            return Ok(true);
        }
        let mut mesh = self.submesh(
            |t| parts[part_of[self.triangles[t][0] as usize] as usize].cavity,
            true,
        )?;
        let bvh = Bvh::build(&mesh, &BuildOptions::default())?;
        let sign = SignData::build(&mut mesh, &bvh, SignMode::Auto)?;
        let mut points: Vec<[f32; 3]> = Vec::new();
        reserve(&mut points, solids.len())?;
        points.extend(
            solids
                .iter()
                .map(|&i| self.positions[parts[i].first as usize]),
        );
        let mut d: Vec<f32> = filled(solids.len(), 0.0)?;
        signed_distances(&mesh, &bvh, &sign, &points, f32::INFINITY, &mut d)?;
        for (&i, &d) in solids.iter().zip(&d) {
            parts[i].cavity = d < 0.0;
        }
        Ok(true)
    }

    /// The triangles `take` picks as a compact mesh in the measured frame.
    fn submesh(&self, take: impl Fn(usize) -> bool, reversed: bool) -> Result<Mesh> {
        let mut remap: Vec<u32> = filled(self.positions.len(), u32::MAX)?;
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut triangles: Vec<[u32; 3]> = Vec::new();
        reserve(&mut triangles, self.triangles.len())?;
        for (i, &t) in self.triangles.iter().enumerate() {
            if !take(i) {
                continue;
            }
            let mut out = [0u32; 3];
            for (slot, v) in out.iter_mut().zip(t) {
                if remap[v as usize] == u32::MAX {
                    remap[v as usize] = positions.len() as u32;
                    grow_push(&mut positions, self.positions[v as usize])?;
                }
                *slot = remap[v as usize];
            }
            if reversed {
                out.swap(1, 2);
            }
            triangles.push(out);
        }
        Ok(Mesh::from_buffers(
            positions.as_flattened(),
            triangles.as_flattened(),
        )?)
    }

    /// The surfaces around the cavities that are filled, which stay.
    fn outer(&self) -> Result<Mesh> {
        let outer = self.submesh(|i| !self.bounds_region[i], false)?;
        if !classify_solid(&outer)?.0 {
            return Err(not_closed());
        }
        Ok(outer)
    }

    /// Whether `f32` positions around the mesh resolve voxels of size `h`.
    fn resolves(&self, h: f32) -> bool {
        (0..3).all(|k| {
            let step = f64::from(f32::EPSILON) * self.largest[k] * self.scale[k];
            f64::from(h) >= MIN_VOXEL_ULPS * step
                && f64::from(h) >= MIN_VOXEL_ULPS * f64::from(f32::EPSILON) * self.extent[k]
        })
    }

    /// Samples the grid of `plan` and extracts the lattice, in the measured
    /// frame, with the vertices next to the clip surface.
    fn extract(
        &self,
        options: &LatticeOptions,
        plan: &LatticePlan,
        extract: Extract,
    ) -> Result<(crate::IsoMesh, Vec<u32>)> {
        let mut mesh = self.submesh(
            |i| self.bounds_region[i],
            self.region == LatticeRegion::Cavity,
        )?;
        let bvh = Bvh::build(&mesh, &BuildOptions::default())?;
        let sign = SignData::build(&mut mesh, &bvh, SignMode::Auto)?;
        let status = sign.status();
        if !status.closed || !status.consistent_orientation {
            return Err(not_closed());
        }
        let h = plan.voxel_size;
        let spec = GridSpec {
            origin: plan
                .dims
                .map(|n| (-0.5 * (n - 1) as f64 * f64::from(h)) as f32),
            dims: plan.dims,
            voxel_size: h,
        };
        spec.validate(options.max_voxels)?;
        let job = Job {
            boundary: Boundary {
                mesh: &mesh,
                bvh: &bvh,
                sign: &sign,
            },
            spec,
            depth: options.shell_thickness,
            walls: Walls {
                kind: options.kind,
                cell_size: f64::from(options.cell_size),
                thickness: f64::from(plan.wall_thickness),
            },
            extract,
            // Around a cavity, every vertex that could come within a voxel
            // of the outside is next to a node this close.
            band: if self.region == LatticeRegion::Cavity {
                1.6
            } else {
                1.0
            },
            cropped: plan.cropped,
        };
        let out = job.run()?;
        Ok((out.iso, out.at_clip))
    }

    /// Maps positions from the measured frame to the stored one.
    fn to_stored(&self, positions: &mut [[f32; 3]]) {
        for p in positions {
            *p = std::array::from_fn(|k| (f64::from(p[k]) / self.scale[k] + self.center[k]) as f32);
        }
    }

    /// Writes the result into `tx`: `added` alone with `replace`, and else in
    /// place of the cavities' surfaces, if any. Returns whether UVs were
    /// dropped.
    fn write(&self, tx: &mut Transaction<'_>, added: AppendSource, replace: bool) -> Result<bool> {
        let mesh = tx.mesh();
        let had_uvs = mesh.uvs().is_some();
        // A surface with UV seams is kept welded: its sides share vertices,
        // and the UVs that set them apart go.
        let rebuild = replace || self.welded;
        let mut drop: Vec<bool> = filled(mesh.triangle_slots(), rebuild)?;
        for (&t, &inner) in self.sources.iter().zip(&self.bounds_region) {
            if inner && self.region == LatticeRegion::Cavity {
                drop[t as usize] = true;
            }
        }
        let mut used: Vec<bool> = filled(mesh.vertex_slots(), false)?;
        let mut touched: Vec<bool> = filled(mesh.vertex_slots(), rebuild)?;
        let mut triangles: Vec<TriangleId> = Vec::new();
        for t in 0..mesh.triangle_slots() as TriangleId {
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };
            let marks = if drop[t as usize] {
                grow_push(&mut triangles, t)?;
                &mut touched
            } else {
                &mut used
            };
            for v in tri {
                marks[v as usize] = true;
            }
        }
        let mut vertices: Vec<VertexId> = Vec::new();
        for v in 0..mesh.vertex_slots() {
            if touched[v] && !used[v] && mesh.is_vertex_alive(v as VertexId) {
                grow_push(&mut vertices, v as VertexId)?;
            }
        }
        let source = if self.welded && !replace {
            self.with_kept(mesh, added)?
        } else {
            added
        };
        let mut mapped: Vec<[VertexId; 3]> = Vec::new();
        reserve(&mut mapped, source.triangles.len())?;
        if had_uvs {
            tx.drop_channel(UV_CHANNEL)?;
        }
        if !triangles.is_empty() {
            tx.remove_triangles(&triangles)?;
        }
        if !vertices.is_empty() {
            tx.remove_vertices(&vertices)?;
        }
        let ids = tx.add_vertices(&source.positions)?;
        mapped.extend(source.triangles.iter().map(|t| t.map(|v| ids[v as usize])));
        tx.add_triangles(&mapped)?;
        Ok(had_uvs)
    }

    /// `added` after the triangles that stay, welded, in the stored frame.
    fn with_kept(&self, mesh: &Mesh, added: AppendSource) -> Result<AppendSource> {
        let mut remap: Vec<u32> = filled(self.positions.len(), u32::MAX)?;
        let mut out = AppendSource::default();
        let kept = match self.region {
            LatticeRegion::Cavity => self.bounds_region.iter().filter(|b| !**b).count(),
            _ => self.triangles.len(),
        };
        reserve(&mut out.triangles, kept + added.triangles.len())?;
        for (i, &t) in self.triangles.iter().enumerate() {
            if self.region == LatticeRegion::Cavity && self.bounds_region[i] {
                continue;
            }
            let mut tri = [0u32; 3];
            for (slot, v) in tri.iter_mut().zip(t) {
                if remap[v as usize] == u32::MAX {
                    remap[v as usize] = out.positions.len() as u32;
                    let p = mesh.positions()[self.vertices[v as usize] as usize];
                    grow_push(&mut out.positions, p)?;
                }
                *slot = remap[v as usize];
            }
            out.triangles.push(tri);
        }
        let base = u32::try_from(out.positions.len() + added.positions.len())
            .map_err(|_| OpsError::Store(StoreError::OutOfMemory))?
            - added.positions.len() as u32;
        reserve(&mut out.positions, added.positions.len())?;
        out.positions.extend_from_slice(&added.positions);
        out.triangles
            .extend(added.triangles.iter().map(|t| t.map(|v| base + v)));
        Ok(out)
    }
}

fn not_closed() -> OpsError {
    OpsError::InvalidInput("lattice needs a closed, consistently wound manifold mesh")
}

/// Checks that no triangle with its corners at `points`, none longer than a
/// voxel's diagonal, can reach `outer`.
fn check_wall(outer: &Mesh, points: &[[f32; 3]], h: f32) -> Result<()> {
    let bvh = Bvh::build(outer, &BuildOptions::default())?;
    let opts = ClosestOptions {
        max_distance: h * 1.01,
    };
    let mut hits: Vec<Option<ClosestHit>> = filled(QUERY_BATCH.min(points.len()), None)?;
    for batch in points.chunks(QUERY_BATCH) {
        let hits = &mut hits[..batch.len()];
        closest_point_many(outer, &bvh, batch, &opts, hits)?;
        if hits.iter().any(Option::is_some) {
            return Err(OpsError::InvalidInput(
                "lattice voxels are too coarse for the wall around the cavity; \
                 use a smaller voxel size",
            ));
        }
    }
    Ok(())
}

fn triangle_area(p: &[[f32; 3]], [a, b, c]: [u32; 3]) -> f64 {
    let [p, q, r] = [a, b, c].map(|v| p[v as usize].map(f64::from));
    let u: [f64; 3] = std::array::from_fn(|k| q[k] - p[k]);
    let w: [f64; 3] = std::array::from_fn(|k| r[k] - p[k]);
    let n = [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ];
    0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
}
