//! Offsetting a surface along its normals or reconstructing an offset volume,
//! and solidifying a surface into a shell.
//!
//! [`offset`] supports three strategies via [`OffsetAlgorithm`]:
//! - [`OffsetAlgorithm::Normal`] (default): displaces every live vertex along
//!   its smooth vertex normal ([`compute_vertex_normals`](fast_mesh::compute_vertex_normals))
//!   by `distance` in local units; zero distance writes nothing.
//! - [`OffsetAlgorithm::FhEdt`] and [`OffsetAlgorithm::BvhSdf`]: evaluate a
//!   signed (closed, consistently outward solids) or unsigned (open /
//!   non-manifold sheets for `distance > 0`) 3D distance grid and extract the
//!   `iso_value = distance` level-set with Manifold Marching Cubes, replacing
//!   the mesh topology in one undoable transaction (dropping `UV_CHANNEL` when
//!   present).
//!
//! [`shell`] solidifies the mesh: it keeps the input surface, adds a copy
//! moved by `thickness` against the smooth normals with reversed winding, and
//! stitches every boundary rim to its copy with a flat wall band, so an open
//! mesh comes out closed. A closed mesh has no rims to stitch; it comes out
//! as two nested closed components. Triangles with a repeated vertex id are
//! not copied: they have no surface to solidify.
//!
//! All operations read and validate their whole plan from the untouched mesh
//! and only then write, so an error leaves the mesh as it was.

use fast_mesh::compute_vertex_normals;
use fast_mesh_bvh::{BuildOptions, Bvh, BvhError};
use fast_mesh_fields::{
    DEFAULT_AUTO_GRID_RES, FieldsError, GridSpec, SignData, SignMode, bvh_signed_grid,
    bvh_unsigned_grid, fh_signed_grid, fh_unsigned_grid,
};
use fast_mesh_query::QueryError;
use fast_mesh_store::{Mesh, StoreError, Transaction, TriangleId, UV_CHANNEL, VertexId};

use crate::topology::{EdgeMap, boundary_loops, cross, dot, point, proper_triangle};
use crate::{
    DegenerateOptions, DegenerateReport, IsoMesh, OpsError, Result, allocation_allowed, filled,
    marching_cubes, push, remove_degenerate, reserve,
};

/// Surface offsetting and reconstruction algorithm for [`offset`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum OffsetAlgorithm {
    /// Displace existing vertices along smooth area-weighted vertex normals.
    #[default]
    Normal,
    /// Voxelise onto a 3D regular grid, compute exact separable 3D
    /// Felzenszwalb-Huttenlocher Euclidean Distance Transform, and extract
    /// the offset isosurface with Manifold Marching Cubes.
    FhEdt {
        /// Grid spacing in local units; `None` selects `bbox_diag / 64.0`
        /// (clamped to fit `max_voxels`).
        voxel_size: Option<f32>,
    },
    /// Evaluate exact BVH closest-triangle distance (with narrow-band sign
    /// propagation on closed solids) on a 3D regular grid, and extract the
    /// offset isosurface with Manifold Marching Cubes.
    BvhSdf {
        /// Grid spacing in local units; `None` selects `bbox_diag / 64.0`
        /// (clamped to fit `max_voxels`).
        voxel_size: Option<f32>,
    },
}

/// Options for [`offset`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OffsetOptions {
    /// Signed offset distance in local units.
    pub distance: f32,
    /// Surface offsetting or volumetric reconstruction algorithm.
    pub algorithm: OffsetAlgorithm,
    /// Safety cap on total voxel count for volumetric algorithms (`FhEdt` and `BvhSdf`).
    pub max_voxels: usize,
}

impl Default for OffsetOptions {
    fn default() -> Self {
        OffsetOptions {
            distance: 0.0,
            algorithm: OffsetAlgorithm::Normal,
            max_voxels: fast_mesh_fields::DEFAULT_MAX_VOXELS,
        }
    }
}

impl OffsetOptions {
    /// A builder for [`OffsetOptions`]: zero distance, [`OffsetAlgorithm::Normal`],
    /// and [`fast_mesh_fields::DEFAULT_MAX_VOXELS`] until set.
    pub fn builder() -> OffsetOptionsBuilder {
        OffsetOptionsBuilder::default()
    }
}

/// A builder for [`OffsetOptions`], see [`OffsetOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct OffsetOptionsBuilder {
    options: OffsetOptions,
}

impl OffsetOptionsBuilder {
    /// Signed offset distance in local units.
    pub fn distance(mut self, distance: f32) -> Self {
        self.options.distance = distance;
        self
    }

    /// Surface offsetting or volumetric reconstruction algorithm.
    pub fn algorithm(mut self, algorithm: OffsetAlgorithm) -> Self {
        self.options.algorithm = algorithm;
        self
    }

    /// Safety cap on total voxel count for volumetric algorithms.
    pub fn max_voxels(mut self, max_voxels: usize) -> Self {
        self.options.max_voxels = max_voxels;
        self
    }

    /// The options as set.
    pub fn build(self) -> OffsetOptions {
        self.options
    }
}

/// What [`offset`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OffsetReport {
    /// How many vertices moved (displacement above 0; `0` for volumetric rebuilds).
    pub moved: usize,
    /// Largest displacement applied, in local units (`distance.abs()` for volumetric rebuilds).
    pub max_displacement: f32,
    /// Degenerate triangles the cleanup pass removed.
    pub degenerate_removed: usize,
    /// Vertices removed (cleanup removals for `Normal`, replaced + cleanup removals for volumetric).
    pub vertices_removed: usize,
    /// Vertices added by volumetric reconstruction (`0` for `Normal`).
    pub vertices_added: usize,
    /// Triangles added by volumetric reconstruction (`0` for `Normal`).
    pub triangles_added: usize,
    /// Existing triangles replaced by volumetric reconstruction (`0` for `Normal`).
    pub triangles_removed: usize,
    /// Grid pitch in local units used by volumetric reconstruction (`0.0` for `Normal`).
    pub voxel_size: f32,
    /// Whether `UV_CHANNEL` was dropped by volumetric reconstruction (`false` for `Normal`).
    pub uvs_dropped: bool,
}

/// Offsets the surface in `tx` according to `options` (without a prebuilt BVH).
pub fn offset(tx: &mut Transaction, options: &OffsetOptions) -> Result<OffsetReport> {
    offset_with_bvh(tx, None, options)
}

/// Offsets the surface in `tx` according to `options`, optionally reusing a current `bvh`
/// of `tx.mesh()`.
pub fn offset_with_bvh(
    tx: &mut Transaction,
    bvh: Option<&Bvh>,
    options: &OffsetOptions,
) -> Result<OffsetReport> {
    allocation_allowed()?;
    let distance = options.distance;
    if !distance.is_finite() {
        return Err(OpsError::InvalidInput("offset distance must be finite"));
    }
    if let Some(b) = bvh {
        if b.mesh_uid() != tx.mesh().uid() {
            return Err(QueryError::MeshMismatch.into());
        }
        if !b.is_current(tx.mesh()) {
            return Err(QueryError::StaleBvh.into());
        }
    }
    match options.algorithm {
        OffsetAlgorithm::Normal => offset_normal(tx, distance),
        OffsetAlgorithm::FhEdt { voxel_size } => {
            offset_volumetric(tx, bvh, options, voxel_size, false)
        }
        OffsetAlgorithm::BvhSdf { voxel_size } => {
            offset_volumetric(tx, bvh, options, voxel_size, true)
        }
    }
}

fn build_work_bvh(work: &Mesh) -> Result<Bvh> {
    Bvh::build(work, &BuildOptions::default()).map_err(|e| match e {
        BvhError::OutOfMemory => OpsError::Store(StoreError::OutOfMemory),
        _ => OpsError::InvalidInput("volumetric offset BVH failed to build"),
    })
}

/// Returns `(closed_and_consistently_oriented, signed_volume)` for compact `work`.
fn classify_solid(work: &Mesh) -> Result<(bool, f64)> {
    let edges = EdgeMap::build(work)?;
    let mut closed_oriented = false;
    let mut any_edge = false;
    let mut all_ok = true;

    let edge_dir = |tri: [VertexId; 3], u: VertexId, v: VertexId| -> bool {
        (tri[0] == u && tri[1] == v) || (tri[1] == u && tri[2] == v) || (tri[2] == u && tri[0] == v)
    };

    for ((u, v), tris) in edges.edges() {
        any_edge = true;
        if tris.len() != 2 {
            all_ok = false;
            break;
        }
        let t0 = work.triangle(tris[0]).expect("live edge triangle");
        let t1 = work.triangle(tris[1]).expect("live edge triangle");
        if edge_dir(t0, u, v) == edge_dir(t1, u, v) {
            all_ok = false;
            break;
        }
    }
    if any_edge && all_ok {
        closed_oriented = true;
    }

    let mut signed_volume = 0.0f64;
    if closed_oriented {
        for t in 0..work.triangle_slots() as TriangleId {
            if let Some([a, b, c]) = work.triangle(t) {
                let pa = point(work, a);
                let pb = point(work, b);
                let pc = point(work, c);
                signed_volume += dot(pa, cross(pb, pc)) / 6.0;
            }
        }
    }
    Ok((closed_oriented, signed_volume))
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

fn choose_grid_spec(
    min: [f32; 3],
    max: [f32; 3],
    diag: f64,
    voxel_size: Option<f32>,
    margin_distance: f32,
    max_voxels: usize,
) -> Result<GridSpec> {
    let pad_voxels = 3usize;
    if let Some(h_user) = voxel_size {
        return Ok(GridSpec::around_bounds(
            min,
            max,
            h_user,
            margin_distance,
            pad_voxels,
            max_voxels,
        )?);
    }

    let mut h = if diag > 0.0 {
        ((diag / DEFAULT_AUTO_GRID_RES as f64) as f32).max(1e-4)
    } else {
        (margin_distance.abs() * 0.25).max(1e-3)
    };
    let max_base = (0..3)
        .map(|k| {
            (f64::from(max[k]) - f64::from(min[k])) + 2.0 * f64::from(margin_distance.max(0.0))
        })
        .fold(0.0f64, f64::max);
    let axis_budget = (max_voxels as f64).cbrt() - (2 * pad_voxels + 1) as f64;
    if axis_budget > 0.0 && max_base > 0.0 {
        let h_lower = ((max_base / axis_budget) * 1.01) as f32;
        if h_lower.is_finite() && h_lower > h {
            h = h_lower;
        }
    }

    for _ in 0..32 {
        match GridSpec::around_bounds(min, max, h, margin_distance, pad_voxels, max_voxels) {
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
    Ok(GridSpec::around_bounds(
        min,
        max,
        h,
        margin_distance,
        pad_voxels,
        max_voxels,
    )?)
}

fn offset_volumetric(
    tx: &mut Transaction,
    bvh: Option<&Bvh>,
    options: &OffsetOptions,
    voxel_size: Option<f32>,
    use_bvh: bool,
) -> Result<OffsetReport> {
    if options.max_voxels < 64 {
        return Err(OpsError::InvalidInput("max_voxels must be at least 64"));
    }
    if let Some(h) = voxel_size
        && (!h.is_finite() || h <= 0.0)
    {
        return Err(OpsError::InvalidInput(
            "voxel_size must be positive and finite",
        ));
    }

    let mesh = tx.mesh();
    if mesh.triangle_count() == 0 {
        return Ok(OffsetReport::default());
    }
    let positions = mesh.positions();
    for (v, p) in positions.iter().enumerate() {
        if mesh.is_vertex_alive(v as VertexId) && !p.iter().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput(
                "offset needs finite vertex positions",
            ));
        }
    }

    // Build compact UV-less working mesh, welding exact 0.0-coincident UV seams when UVs exist.
    let had_uvs = mesh.uvs().is_some();
    let roots = if had_uvs {
        Some(crate::weld::weld_roots(mesh, 0.0)?)
    } else {
        None
    };

    let mut remap: Vec<u32> = filled(mesh.vertex_slots(), u32::MAX)?;
    let mut work_positions: Vec<[f32; 3]> = Vec::new();
    let mut work_tris: Vec<[u32; 3]> = Vec::new();
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut has_positive_area = false;
    let mut had_skipped_tri = false;

    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some(raw) = mesh.triangle(t) else {
            continue;
        };
        let [ra, rb, rc] = match &roots {
            Some(r) => raw.map(|v| r[v as usize]),
            None => raw,
        };
        if ra == rb || rb == rc || ra == rc {
            had_skipped_tri = true;
            continue;
        }
        let pa = positions[ra as usize];
        let pb = positions[rb as usize];
        let pc = positions[rc as usize];
        let ua = [
            f64::from(pb[0]) - f64::from(pa[0]),
            f64::from(pb[1]) - f64::from(pa[1]),
            f64::from(pb[2]) - f64::from(pa[2]),
        ];
        let va = [
            f64::from(pc[0]) - f64::from(pa[0]),
            f64::from(pc[1]) - f64::from(pa[1]),
            f64::from(pc[2]) - f64::from(pa[2]),
        ];
        let n = cross(ua, va);
        if dot(n, n) > 0.0 {
            has_positive_area = true;
        }
        let mut mapped = [0u32; 3];
        for (k, rv) in [ra, rb, rc].into_iter().enumerate() {
            let idx = if remap[rv as usize] != u32::MAX {
                remap[rv as usize]
            } else {
                let fresh = u32::try_from(work_positions.len())
                    .map_err(|_| OpsError::Store(StoreError::OutOfMemory))?;
                let pos = positions[rv as usize];
                for axis in 0..3 {
                    min[axis] = min[axis].min(pos[axis]);
                    max[axis] = max[axis].max(pos[axis]);
                }
                push(&mut work_positions, pos)?;
                remap[rv as usize] = fresh;
                fresh
            };
            mapped[k] = idx;
        }
        push(&mut work_tris, mapped)?;
    }

    if !has_positive_area {
        return Err(OpsError::InvalidInput(
            "volumetric offset requires at least one non-degenerate triangle",
        ));
    }

    let mut work = Mesh::new();
    {
        let mut wtx = work.begin_edit();
        wtx.add_vertices(&work_positions)?;
        wtx.add_triangles(&work_tris)?;
        wtx.commit()?;
    }

    let (closed_oriented, signed_volume) = classify_solid(&work)?;
    if closed_oriented {
        if signed_volume <= 0.0 {
            return Err(OpsError::InvalidInput(
                "volumetric offset input is wound inward",
            ));
        }
    } else if options.distance <= 0.0 {
        return Err(OpsError::InvalidInput(
            "open or non-manifold mesh offset requires distance > 0",
        ));
    }

    let dx = f64::from(max[0]) - f64::from(min[0]);
    let dy = f64::from(max[1]) - f64::from(min[1]);
    let dz = f64::from(max[2]) - f64::from(min[2]);
    let diag = (dx * dx + dy * dy + dz * dz).sqrt();
    let margin_distance = options.distance.max(0.0);
    let spec = choose_grid_spec(
        min,
        max,
        diag,
        voxel_size,
        margin_distance,
        options.max_voxels,
    )?;

    let grid = if closed_oriented {
        if !use_bvh {
            fh_signed_grid(&work, spec, options.max_voxels)?
        } else {
            let work_bvh = build_work_bvh(&work)?;
            let sign_data = SignData::build(&mut work, &work_bvh, SignMode::Auto)?;
            debug_assert!(sign_data.status().closed && sign_data.status().consistent_orientation);
            let narrow_band =
                (options.distance.abs() + 3.0 * spec.voxel_size).max(3.0 * spec.voxel_size);
            bvh_signed_grid(
                &work,
                &work_bvh,
                &sign_data,
                spec,
                narrow_band,
                options.max_voxels,
            )?
        }
    } else if !use_bvh {
        fh_unsigned_grid(&work, spec, options.max_voxels)?
    } else {
        let narrow_band = (options.distance + 3.0 * spec.voxel_size).max(3.0 * spec.voxel_size);
        if !had_uvs
            && !had_skipped_tri
            && let Some(caller_bvh) = bvh
        {
            bvh_unsigned_grid(tx.mesh(), caller_bvh, spec, narrow_band, options.max_voxels)?
        } else {
            let work_bvh = build_work_bvh(&work)?;
            bvh_unsigned_grid(&work, &work_bvh, spec, narrow_band, options.max_voxels)?
        }
    };

    let iso = marching_cubes(&grid, options.distance)?;
    let raw_vertices_added = iso.positions.len();
    let raw_triangles_added = iso.triangles.len();
    let (cleaned, cleanup) = clean_iso_mesh(iso)?;
    let final_positions = cleaned.positions;
    let final_triangles = cleaned.triangles;

    let mesh = tx.mesh();
    let mut old_tris: Vec<TriangleId> = Vec::new();
    for t in 0..mesh.triangle_slots() as TriangleId {
        if mesh.is_triangle_alive(t) {
            push(&mut old_tris, t)?;
        }
    }
    let mut old_verts: Vec<VertexId> = Vec::new();
    for v in 0..mesh.vertex_slots() as VertexId {
        if mesh.is_vertex_alive(v) {
            push(&mut old_verts, v)?;
        }
    }
    let mut mapped_tris: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut mapped_tris, final_triangles.len())?;

    if !old_tris.is_empty() {
        tx.remove_triangles(&old_tris)?;
    }
    if !old_verts.is_empty() {
        tx.remove_vertices(&old_verts)?;
    }
    if had_uvs {
        tx.drop_channel(UV_CHANNEL)?;
    }
    if !final_positions.is_empty() {
        let new_vids = tx.add_vertices(&final_positions)?;
        for &[a, b, c] in &final_triangles {
            mapped_tris.push([
                new_vids[a as usize],
                new_vids[b as usize],
                new_vids[c as usize],
            ]);
        }
        tx.add_triangles(&mapped_tris)?;
    }

    Ok(OffsetReport {
        moved: 0,
        max_displacement: options.distance.abs(),
        degenerate_removed: cleanup.removed_triangles,
        vertices_removed: old_verts.len() + cleanup.removed_vertices,
        vertices_added: raw_vertices_added,
        triangles_added: raw_triangles_added,
        triangles_removed: old_tris.len(),
        voxel_size: spec.voxel_size,
        uvs_dropped: had_uvs,
    })
}

fn offset_normal(tx: &mut Transaction, distance: f32) -> Result<OffsetReport> {
    let mut report = OffsetReport::default();
    if distance == 0.0 {
        return Ok(report);
    }
    let mesh = tx.mesh();
    if mesh.triangle_count() == 0 {
        return Ok(report);
    }
    let normals = smooth_normals(mesh)?;
    let positions = mesh.positions();
    let mut ids: Vec<VertexId> = Vec::new();
    let mut moved_to: Vec<[f32; 3]> = Vec::new();
    for (v, p) in positions.iter().enumerate() {
        let v = v as VertexId;
        if !mesh.is_vertex_alive(v) {
            continue;
        }
        let n = [
            normals[3 * v as usize],
            normals[3 * v as usize + 1],
            normals[3 * v as usize + 2],
        ];
        let displaced = [
            p[0] + distance * n[0],
            p[1] + distance * n[1],
            p[2] + distance * n[2],
        ];
        let length = n.iter().map(|n| (distance * n).powi(2)).sum::<f32>().sqrt();
        if !length.is_finite() || length <= 0.0 {
            continue;
        }
        push(&mut ids, v)?;
        push(&mut moved_to, displaced)?;
        report.max_displacement = report.max_displacement.max(length);
    }
    report.moved = ids.len();
    if !ids.is_empty() {
        tx.set_positions(&ids, &moved_to)?;
    }
    let cleanup = remove_degenerate(tx, &DegenerateOptions::default())?;
    report.degenerate_removed = cleanup.removed_triangles;
    report.vertices_removed = cleanup.removed_vertices;
    Ok(report)
}

/// Options for [`shell`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellOptions {
    /// Wall thickness in local units: the copy sits this far against the
    /// smooth normals. Must be positive and finite.
    pub thickness: f32,
}

impl Default for ShellOptions {
    fn default() -> Self {
        ShellOptions { thickness: 1.0 }
    }
}

impl ShellOptions {
    /// A builder for [`ShellOptions`]: a unit thickness until set.
    pub fn builder() -> ShellOptionsBuilder {
        ShellOptionsBuilder::default()
    }
}

/// A builder for [`ShellOptions`], see [`ShellOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct ShellOptionsBuilder {
    options: ShellOptions,
}

impl ShellOptionsBuilder {
    /// Wall thickness in local units.
    pub fn thickness(mut self, thickness: f32) -> Self {
        self.options.thickness = thickness;
        self
    }

    /// The options as set.
    pub fn build(self) -> ShellOptions {
        self.options
    }
}

/// What [`shell`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShellReport {
    /// Copy vertices added.
    pub vertices_added: usize,
    /// Copy and wall triangles added.
    pub triangles_added: usize,
    /// Boundary rims stitched to their copy. A closed mesh has none.
    pub loops_closed: usize,
    /// Degenerate triangles the cleanup pass removed.
    pub degenerate_removed: usize,
    /// Vertices the cleanup pass removed.
    pub vertices_removed: usize,
}

/// A corner of a planned triangle: a kept live vertex or a copy vertex by
/// its index in the planned copy positions.
#[derive(Debug, Clone, Copy)]
enum Corner {
    Kept(VertexId),
    Copy(usize),
}

/// Solidifies the mesh into a printable shell of wall `thickness`.
///
/// - The input surface stays. Every live vertex gains a copy moved by
///   `thickness` against its smooth normal; every proper triangle gains a
///   reversed copy over the copied vertices, so the copy faces outward from
///   the wall interior.
/// - Every boundary edge (an edge of exactly one proper triangle) gains a
///   flat wall quad over its copy: for the edge the outer triangle uses as
///   `x -> y`, the quad is `(y, x, x')` and `(y, x', y')`, which agrees with
///   both sheets along their shared edges. Rims through non-manifold
///   vertices or orientation seams are stitched edge by edge on a
///   best-effort basis and may stay open.
/// - Writes happen in order: copy vertices, then copy and wall triangles.
///   Then [`remove_degenerate`](crate::remove_degenerate) with the default
///   options drops collapsed triangles (a wall thinner than the local
///   curvature pinches its copy to nothing) and unused vertices, reported as
///   `degenerate_removed` and `vertices_removed`. History is untouched.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when the thickness is not positive and
///   finite, when a vertex position or a copy position is not finite, or
///   when the mesh buffers are not a valid mesh.
/// - [`OpsError::Store`] when memory runs out.
///
/// Nothing is written in any of these cases.
pub fn shell(tx: &mut Transaction, options: &ShellOptions) -> Result<ShellReport> {
    let thickness = options.thickness;
    if !thickness.is_finite() || thickness <= 0.0 {
        return Err(OpsError::InvalidInput(
            "shell thickness must be positive and finite",
        ));
    }
    let mut report = ShellReport::default();
    let mesh = tx.mesh();
    if mesh.triangle_count() == 0 {
        return Ok(report);
    }
    let normals = smooth_normals(mesh)?;
    let positions = mesh.positions();
    // The copy of every live vertex, in ascending id order, with each live
    // vertex's index in it. Planned from the untouched mesh, before writing.
    let mut copied: Vec<[f32; 3]> = Vec::new();
    // On a mesh with UVs, each copy keeps its original's.
    let source_uvs = mesh.uvs();
    let mut copied_uvs: Vec<f32> = Vec::new();
    let mut at: Vec<Option<usize>> = filled(mesh.vertex_slots(), None)?;
    for (v, p) in positions.iter().enumerate() {
        let v = v as VertexId;
        if !mesh.is_vertex_alive(v) {
            continue;
        }
        if !p.iter().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput(
                "shell needs finite vertex positions",
            ));
        }
        let n = [
            normals[3 * v as usize],
            normals[3 * v as usize + 1],
            normals[3 * v as usize + 2],
        ];
        let q = [
            p[0] - thickness * n[0],
            p[1] - thickness * n[1],
            p[2] - thickness * n[2],
        ];
        if !q.iter().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput("shell thickness leaves f32 range"));
        }
        push(&mut copied, q)?;
        if let Some(uv) = source_uvs {
            reserve(&mut copied_uvs, 2)?;
            copied_uvs.extend_from_slice(&uv[2 * v as usize..2 * v as usize + 2]);
        }
        at[v as usize] = Some(copied.len() - 1);
    }
    // The reversed copies and the wall quads, in ascending triangle and edge
    // order. Triangles with a repeated vertex id are not copied.
    let mut planned: Vec<[Corner; 3]> = Vec::new();
    for t in 0..mesh.triangle_slots() as u32 {
        if let Some([a, b, c]) = proper_triangle(mesh, t) {
            let [ia, ib, ic] = [a, b, c].map(|v| at[v as usize].expect("live corner"));
            push(
                &mut planned,
                [Corner::Copy(ic), Corner::Copy(ib), Corner::Copy(ia)],
            )?;
        }
    }
    let edges = EdgeMap::build(mesh)?;
    for ((u, v), tris) in edges.edges() {
        if tris.len() != 1 {
            continue;
        }
        let [a, b, c] = mesh.triangle(tris[0]).expect("mapped triangle is alive");
        let (x, y) = [(a, b), (b, c), (c, a)]
            .into_iter()
            .find(|&(p, q)| (p == u && q == v) || (p == v && q == u))
            .expect("mapped edge uses its triangle");
        let (ix, iy) = (
            at[x as usize].expect("live corner"),
            at[y as usize].expect("live corner"),
        );
        push(
            &mut planned,
            [Corner::Kept(y), Corner::Kept(x), Corner::Copy(ix)],
        )?;
        push(
            &mut planned,
            [Corner::Kept(y), Corner::Copy(ix), Corner::Copy(iy)],
        )?;
    }
    report.loops_closed = boundary_loops(mesh)?.len();
    let has_uvs = source_uvs.is_some();
    let fresh = tx.add_vertices(&copied)?;
    if has_uvs {
        tx.write_channel(UV_CHANNEL, &fresh, &copied_uvs)?;
    }
    let resolve = |c: Corner| match c {
        Corner::Kept(v) => v,
        Corner::Copy(i) => fresh[i],
    };
    let mut added: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut added, planned.len())?;
    for tri in planned {
        added.push(tri.map(resolve));
    }
    tx.add_triangles(&added)?;
    report.vertices_added = fresh.len();
    report.triangles_added = added.len();
    let cleanup = remove_degenerate(tx, &DegenerateOptions::default())?;
    report.degenerate_removed = cleanup.removed_triangles;
    report.vertices_removed = cleanup.removed_vertices;
    Ok(report)
}

/// The smooth vertex normals of `mesh`, slot-aligned, as
/// [`inflate`](crate::deform::inflate) computes them.
fn smooth_normals(mesh: &Mesh) -> Result<Vec<f32>> {
    let positions = mesh.positions();
    let mut flat: Vec<f32> = Vec::new();
    reserve(&mut flat, positions.len() * 3)?;
    flat.extend(positions.iter().flatten().copied());
    let mut indices: Vec<u32> = Vec::new();
    reserve(&mut indices, mesh.triangle_slots() * 3)?;
    for t in 0..mesh.triangle_slots() as u32 {
        if let Some(tri) = mesh.triangle(t) {
            indices.extend(tri);
        }
    }
    compute_vertex_normals(&flat, &indices)
        .map_err(|_| OpsError::InvalidInput("mesh buffers are not a valid mesh"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{box_mesh, edit, grid, mesh_of, parts};

    /// The distance of every live vertex from `c`.
    fn distances(mesh: &Mesh, c: [f32; 3]) -> Vec<f32> {
        let mut out = Vec::new();
        for (v, p) in mesh.positions().iter().enumerate() {
            if !mesh.is_vertex_alive(v as VertexId) {
                continue;
            }
            out.push(
                ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt(),
            );
        }
        out
    }

    #[test]
    fn offset_moves_along_normals_by_distance() {
        let mut mesh = grid(2);
        let old = parts(&mesh).0;
        let options = OffsetOptions::builder().distance(0.5).build();
        let (report, empty) = edit(&mut mesh, |tx| offset(tx, &options));
        assert!(!empty);
        // A flat grid wound from +z: every normal is +z, so every vertex
        // moves exactly 0.5 up.
        assert_eq!(report.moved, 9);
        assert!((report.max_displacement - 0.5).abs() < 1e-6);
        assert_eq!(report.degenerate_removed, 0);
        assert_eq!(report.vertices_removed, 0);
        let new = parts(&mesh).0;
        for v in 0..9 {
            assert!((new[v * 3] - old[v * 3]).abs() < 1e-6);
            assert!((new[v * 3 + 1] - old[v * 3 + 1]).abs() < 1e-6);
            assert!((new[v * 3 + 2] - old[v * 3 + 2] - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn offset_sign_follows_winding() {
        // A closed, outward-wound box: a positive offset moves every corner
        // exactly `distance` along its own (unit) smooth normal, which points
        // outward, so every corner leaves the centroid behind; a negative
        // distance sinks the corners back inward. (Box corners are shared by
        // an uneven triangulation, so their normals are not exact diagonals:
        // only the length and the outward sense are asserted.)
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        let c = [0.5; 3];
        let before = distances(&mesh, c);
        let (report, _) = edit(&mut mesh, |tx| {
            offset(tx, &OffsetOptions::builder().distance(0.25).build())
        });
        assert_eq!(report.moved, 8);
        assert!((report.max_displacement - 0.25).abs() < 1e-6);
        let grown = distances(&mesh, c);
        for (a, b) in before.iter().zip(&grown) {
            assert!(*b > *a, "{a} -> {b}");
            assert!((b - a) <= 0.25 + 1e-6, "{a} -> {b}");
        }
        // Every corner moved exactly 0.25: smooth normals are unit.
        let prev = parts(&box_mesh([0.0; 3], [1.0; 3])).0;
        let next = parts(&mesh).0;
        for v in 0..8 {
            let d = ((next[v * 3] - prev[v * 3]).powi(2)
                + (next[v * 3 + 1] - prev[v * 3 + 1]).powi(2)
                + (next[v * 3 + 2] - prev[v * 3 + 2]).powi(2))
            .sqrt();
            assert!((d - 0.25).abs() < 1e-5, "{d}");
        }
        // And a negative distance sinks the surface back past its start.
        edit(&mut mesh, |tx| {
            offset(tx, &OffsetOptions::builder().distance(-0.5).build()).map(drop)
        });
        let sunk = distances(&mesh, c);
        for (a, b) in before.iter().zip(&sunk) {
            assert!(*b < *a, "{a} -> {b}");
        }
    }

    #[test]
    fn offset_zero_distance_and_empty_mesh_are_noops() {
        let mut mesh = grid(2);
        let before = parts(&mesh);
        let mut tx = mesh.begin_edit();
        let report = offset(&mut tx, &OffsetOptions::builder().distance(0.0).build()).unwrap();
        assert_eq!(report, OffsetReport::default());
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
        // No triangles: nothing to move along, and no cleanup either.
        let mut mesh = Mesh::new();
        let mut tx = mesh.begin_edit();
        let report = offset(&mut tx, &OffsetOptions::builder().distance(1.0).build()).unwrap();
        assert_eq!(report, OffsetReport::default());
        assert!(tx.commit().unwrap().is_empty());
    }

    #[test]
    fn offset_rejects_nonfinite_distance() {
        for distance in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut mesh = grid(2);
            let before = parts(&mesh);
            let mut tx = mesh.begin_edit();
            assert!(matches!(
                offset(
                    &mut tx,
                    &OffsetOptions::builder().distance(distance).build()
                ),
                Err(OpsError::InvalidInput(_))
            ));
            assert!(tx.commit().unwrap().is_empty());
            assert_eq!(parts(&mesh), before);
        }
    }

    #[test]
    fn offset_collapse_without_pinch_stays_valid() {
        // A regular tetrahedron offset inward by its circumradius moves every
        // vertex onto the centroid. The four triangles shrink to specks, but
        // they stay well-shaped, so the cleanup keeps them: only exact
        // pinches and pre-existing degenerates are removed, never valid
        // triangles, however small. (Near-coincident vertices are not
        // merged either; welding stays an explicit operation.)
        let mut mesh = mesh_of(
            &[1., 1., 1., 1., -1., -1., -1., 1., -1., -1., -1., 1.],
            &[[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
        );
        assert_eq!(mesh.vertex_count(), 4);
        let c = [0.0; 3];
        let radius = distances(&mesh, c)[0];
        let (report, _) = edit(&mut mesh, |tx| {
            offset(tx, &OffsetOptions::builder().distance(-radius).build())
        });
        assert_eq!(report.moved, 4);
        assert!((report.max_displacement - radius).abs() < 1e-4);
        assert_eq!(report.degenerate_removed, 0);
        assert_eq!(mesh.triangle_count(), 4);
        for d in distances(&mesh, c) {
            assert!(d < 1e-6, "{d}");
        }
    }

    #[test]
    fn offset_cleans_preexisting_degenerate_triangles() {
        // A grid with an extra collapsed triangle: the offset moves the grid
        // and the cleanup drops the collapsed triangle, reporting it.
        let mut mesh = grid(2);
        let (flat, mut tris) = parts(&mesh);
        tris.push([0, 0, 1]);
        mesh = mesh_of(&flat, &tris);
        assert_eq!(mesh.triangle_count(), 9);
        let options = OffsetOptions::builder().distance(0.5).build();
        let (report, _) = edit(&mut mesh, |tx| offset(tx, &options));
        assert_eq!(report.moved, 9);
        assert_eq!(report.degenerate_removed, 1);
        assert_eq!(mesh.triangle_count(), 8);
    }

    #[test]
    fn shell_stitches_an_open_triangle() {
        // One triangle in z = 0, wound from +z: the copy lands at z = -t and
        // three wall quads stitch the rim.
        let mut mesh = mesh_of(&[0., 0., 0., 1., 0., 0., 0., 1., 0.], &[[0, 1, 2]]);
        let (report, empty) = edit(&mut mesh, |tx| {
            shell(tx, &ShellOptions::builder().thickness(0.5).build())
        });
        assert!(!empty);
        assert_eq!(
            report,
            ShellReport {
                vertices_added: 3,
                triangles_added: 1 + 6,
                loops_closed: 1,
                degenerate_removed: 0,
                vertices_removed: 0,
            }
        );
        assert_eq!(mesh.vertex_count(), 6);
        assert_eq!(mesh.triangle_count(), 8);
        // The outer sheet stayed, the copy sits exactly one thickness below.
        let (flat, _) = parts(&mesh);
        assert_eq!(&flat[0..9], &[0., 0., 0., 1., 0., 0., 0., 1., 0.][..]);
        assert_eq!(
            &flat[9..18],
            &[0., 0., -0.5, 1., 0., -0.5, 0., 1., -0.5][..]
        );
        // Closed and consistently wound: no boundary, no conflicts.
        let analysis = crate::analyze(&mesh, &crate::AnalyzeOptions::default()).unwrap();
        assert!(analysis.loops.is_empty());
        assert_eq!(analysis.orientation_conflicts, 0);
        assert!(analysis.components.iter().all(|c| c.closed));
    }

    #[test]
    fn shell_nests_a_closed_cube() {
        // A closed box has no rims: the reversed copy nests inside it.
        let mut mesh = box_mesh([0.0; 3], [2.0; 3]);
        let (report, _) = edit(&mut mesh, |tx| {
            shell(tx, &ShellOptions::builder().thickness(0.25).build())
        });
        assert_eq!(report.vertices_added, 8);
        assert_eq!(report.triangles_added, 12);
        assert_eq!(report.loops_closed, 0);
        assert_eq!(report.degenerate_removed, 0);
        assert_eq!(mesh.triangle_count(), 24);
        let analysis = crate::analyze(&mesh, &crate::AnalyzeOptions::default()).unwrap();
        assert!(analysis.loops.is_empty());
        assert_eq!(analysis.orientation_conflicts, 0);
        assert_eq!(analysis.components.len(), 2);
        assert!(analysis.components.iter().all(|c| c.closed));
        // The outer sheet stayed exactly where it was.
        let (flat, _) = parts(&mesh);
        assert_eq!(flat.len(), 16 * 3);
        let outer: Vec<f32> = (0..8)
            .flat_map(|i| (0..3).map(move |k| if i >> k & 1 == 1 { 2.0 } else { 0.0 }))
            .collect();
        assert_eq!(&flat[0..24], &outer[..]);
        // The inner copy sits one thickness inside: every copy vertex is
        // exactly `thickness` from its original (unit smooth normals) and
        // strictly inside the box (the thickness is smaller than its
        // half-size, so nothing overshoots the far side).
        for v in 0..8 {
            let (o, q) = (&flat[v * 3..v * 3 + 3], &flat[(v + 8) * 3..(v + 8) * 3 + 3]);
            let d = ((q[0] - o[0]).powi(2) + (q[1] - o[1]).powi(2) + (q[2] - o[2]).powi(2)).sqrt();
            assert!((d - 0.25).abs() < 1e-5, "{d}");
            for k in 0..3 {
                assert!(q[k] > 0.0 && q[k] < 2.0, "{q:?}");
            }
        }
    }

    #[test]
    fn shell_rejects_bad_thickness() {
        for thickness in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut mesh = grid(2);
            let before = parts(&mesh);
            let mut tx = mesh.begin_edit();
            assert!(matches!(
                shell(
                    &mut tx,
                    &ShellOptions::builder().thickness(thickness).build()
                ),
                Err(OpsError::InvalidInput(_))
            ));
            assert!(tx.commit().unwrap().is_empty());
            assert_eq!(parts(&mesh), before);
        }
    }

    #[test]
    fn shell_empty_mesh_is_a_noop() {
        let mut mesh = Mesh::new();
        let mut tx = mesh.begin_edit();
        let report = shell(&mut tx, &ShellOptions::builder().thickness(0.5).build()).unwrap();
        assert_eq!(report, ShellReport::default());
        assert!(tx.commit().unwrap().is_empty());
    }

    #[test]
    fn shell_first_failure_writes_nothing() {
        // The plan is read before the first write: failing the first
        // reservation leaves the mesh untouched.
        let mut mesh = grid(2);
        let before = parts(&mesh);
        let mut tx = mesh.begin_edit();
        crate::fail_allocations_after(Some(0));
        let r = shell(&mut tx, &ShellOptions::builder().thickness(0.5).build());
        crate::fail_allocations_after(None);
        assert_eq!(
            r,
            Err(OpsError::Store(fast_mesh_store::StoreError::OutOfMemory))
        );
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    /// Closed, outward-wound L-shaped solid extruded over `z in [0, 2]` with a thicker
    /// `[0, 2] x [0, 3]` stem and a thinner `[2, 3.5] x [0, 0.8]` arm, plus mid-wall
    /// vertices near the re-entrant `(2.0, 0.8)` corner.
    fn l_shape_solid() -> Mesh {
        let poly: [[f32; 2]; 9] = [
            [0.0, 0.0],
            [2.0, 0.0],
            [3.5, 0.0],
            [3.5, 0.8],
            [2.2, 0.8],
            [2.0, 0.8],
            [2.0, 1.0],
            [2.0, 3.0],
            [0.0, 3.0],
        ];
        let cap_tris: [[u32; 3]; 7] = [
            [1, 2, 3],
            [1, 3, 4],
            [1, 4, 5],
            [0, 1, 5],
            [0, 5, 6],
            [0, 6, 7],
            [0, 7, 8],
        ];
        let mut positions = Vec::new();
        for z in [0.0f32, 1.0, 2.0] {
            for [x, y] in poly {
                positions.extend([x, y, z]);
            }
        }
        let mut tris = Vec::new();
        // Bottom cap at z = 0 (outward -Z).
        for [a, b, c] in cap_tris {
            tris.push([a, c, b]);
        }
        // Top cap at z = 2 (outward +Z).
        for [a, b, c] in cap_tris {
            tris.push([18 + a, 18 + b, 18 + c]);
        }
        // Side bands r = 0..2.
        for r in 0..2u32 {
            let b0 = r * 9;
            let t0 = (r + 1) * 9;
            for i in 0..9u32 {
                let j = (i + 1) % 9;
                tris.push([b0 + i, b0 + j, t0 + j]);
                tris.push([b0 + i, t0 + j, t0 + i]);
            }
        }
        mesh_of(&positions, &tris)
    }

    fn assert_closed_clean_manifold(mesh: &Mesh) {
        use fast_mesh_bvh::BuildOptions;
        let bvh = Bvh::build(mesh, &BuildOptions::default()).unwrap();
        let a = crate::analyze(
            mesh,
            &crate::AnalyzeOptions::builder()
                .self_intersection(&bvh)
                .build(),
        )
        .unwrap();
        assert!(
            a.loops.is_empty(),
            "expected closed mesh, got {:?}",
            a.loops
        );
        assert_eq!(a.non_manifold_edges, 0, "{a:?}");
        assert!(a.non_manifold_vertices.is_empty(), "{a:?}");
        assert!(a.pinch_vertices.is_empty(), "{a:?}");
        assert_eq!(a.orientation_conflicts, 0, "{a:?}");
        assert_eq!(a.degenerate, 0, "{a:?}");
        assert_eq!(a.duplicate, 0, "{a:?}");
        assert_eq!(a.self_intersects, Some(false), "{a:?}");
        let m = crate::measure(mesh).unwrap();
        assert!(m.closed && m.signed_volume > 0.0, "{m:?}");
    }

    #[test]
    fn volumetric_offset_closed_cube_and_l_shape_are_manifold_and_self_intersection_free() {
        use fast_mesh_bvh::BuildOptions;

        let base_l = l_shape_solid();
        assert_closed_clean_manifold(&base_l);
        let base_vol = crate::measure(&base_l).unwrap().volume.unwrap();
        // [0, 2] x [0, 3] x [0, 2] (12.0) + [2, 3.5] x [0, 0.8] x [0, 2] (2.4) = 14.4
        assert!((base_vol - 14.4).abs() < 1e-4, "{base_vol}");

        // Normal offset self-intersects on both outward (+0.45) and inward (-0.5) L-shape offsets.
        for d in [0.45f32, -0.5f32] {
            let mut m = l_shape_solid();
            edit(&mut m, |tx| {
                offset(tx, &OffsetOptions::builder().distance(d).build())
            });
            let bvh = Bvh::build(&m, &BuildOptions::default()).unwrap();
            let a = crate::analyze(
                &m,
                &crate::AnalyzeOptions::builder()
                    .self_intersection(&bvh)
                    .build(),
            )
            .unwrap();
            assert_eq!(
                a.self_intersects,
                Some(true),
                "Normal offset at d={d} should self-intersect on L-shape"
            );
        }

        // Both FhEdt and BvhSdf produce closed, 2-manifold, outward-wound, self-intersection-free solids!
        let h = Some(0.12f32);
        for alg in [
            OffsetAlgorithm::FhEdt { voxel_size: h },
            OffsetAlgorithm::BvhSdf { voxel_size: h },
        ] {
            for d in [0.45f32, -0.5f32] {
                let mut m = l_shape_solid();
                let pre_bvh = Bvh::build(&m, &BuildOptions::default()).unwrap();
                let opts = OffsetOptions::builder().distance(d).algorithm(alg).build();
                let (r, empty) = edit(&mut m, |tx| offset_with_bvh(tx, Some(&pre_bvh), &opts));
                assert!(!empty);
                assert_eq!(r.moved, 0);
                assert!((r.max_displacement - d.abs()).abs() < 1e-6);
                assert!((r.voxel_size - 0.12).abs() < 1e-6);
                assert!(r.vertices_added > 0 && r.triangles_added > 0);
                assert_eq!(r.triangles_removed, base_l.triangle_count());
                assert!(!r.uvs_dropped);
                assert_closed_clean_manifold(&m);

                let vol = crate::measure(&m).unwrap().volume.unwrap();
                if d > 0.0 {
                    assert!(vol > base_vol, "outward volume {vol} <= {base_vol}");
                } else {
                    assert!(vol < base_vol && vol > 0.5, "inward volume {vol}");
                }
            }
        }

        // distance == 0.0: Normal is a no-op, whereas FhEdt and BvhSdf voxel-remesh a closed box
        // to matching volume within voxel tolerance.
        let mut normal_box = box_mesh([0.0; 3], [2.0; 3]);
        let (r_norm, empty_norm) = edit(&mut normal_box, |tx| {
            offset(tx, &OffsetOptions::builder().distance(0.0).build())
        });
        assert!(empty_norm);
        assert_eq!(r_norm, OffsetReport::default());

        for alg in [
            OffsetAlgorithm::FhEdt {
                voxel_size: Some(0.1),
            },
            OffsetAlgorithm::BvhSdf {
                voxel_size: Some(0.1),
            },
        ] {
            let mut remeshed = box_mesh([0.0; 3], [2.0; 3]);
            let opts = OffsetOptions::builder()
                .distance(0.0)
                .algorithm(alg)
                .build();
            let (r, empty) = edit(&mut remeshed, |tx| offset(tx, &opts));
            assert!(!empty);
            assert!(r.triangles_added > 12);
            assert_closed_clean_manifold(&remeshed);
            let vol = crate::measure(&remeshed).unwrap().volume.unwrap();
            assert!(
                (vol - 8.0).abs() < 0.6,
                "alg={alg:?} remeshed box volume {vol} too far from 8.0"
            );
        }
    }

    #[test]
    fn volumetric_offset_open_sheet_capsules_and_rejects_nonpositive() {
        use fast_mesh_bvh::BuildOptions;

        for alg in [
            OffsetAlgorithm::FhEdt {
                voxel_size: Some(0.12),
            },
            OffsetAlgorithm::BvhSdf {
                voxel_size: Some(0.12),
            },
        ] {
            let mut sheet = grid(4);
            let pre_bvh = Bvh::build(&sheet, &BuildOptions::default()).unwrap();
            let opts = OffsetOptions::builder()
                .distance(0.25)
                .algorithm(alg)
                .build();
            let (r, empty) = edit(&mut sheet, |tx| offset_with_bvh(tx, Some(&pre_bvh), &opts));
            assert!(!empty);
            assert!(r.triangles_added > 0 && r.vertices_added > 0);
            assert!(boundary_loops(&sheet).unwrap().is_empty());
            assert_closed_clean_manifold(&sheet);
            let vol = crate::measure(&sheet).unwrap().volume.unwrap();
            // 4 x 4 sheet thickened by radius 0.25 on both sides has volume > 4 * 4 * 0.5 = 8.0.
            assert!(vol > 7.0 && vol < 14.0, "alg={alg:?} capsule volume {vol}");

            for bad_d in [0.0f32, -0.2f32] {
                let mut untouched = grid(4);
                let before = parts(&untouched);
                let mut tx = untouched.begin_edit();
                let res = offset(
                    &mut tx,
                    &OffsetOptions::builder()
                        .distance(bad_d)
                        .algorithm(alg)
                        .build(),
                );
                assert!(matches!(res, Err(OpsError::InvalidInput(_))), "{res:?}");
                assert!(tx.commit().unwrap().is_empty());
                assert_eq!(parts(&untouched), before);
            }

            // Non-manifold fin: rejected at distance <= 0, thickened cleanly at distance > 0.
            let mut fin_bad = crate::test_util::fin();
            let fin_before = parts(&fin_bad);
            let mut tx = fin_bad.begin_edit();
            assert!(matches!(
                offset(
                    &mut tx,
                    &OffsetOptions::builder()
                        .distance(-0.1)
                        .algorithm(alg)
                        .build()
                ),
                Err(OpsError::InvalidInput(_))
            ));
            assert!(tx.commit().unwrap().is_empty());
            assert_eq!(parts(&fin_bad), fin_before);

            let mut fin_ok = crate::test_util::fin();
            edit(&mut fin_ok, |tx| {
                offset(
                    tx,
                    &OffsetOptions::builder()
                        .distance(0.25)
                        .algorithm(alg)
                        .build(),
                )
            });
            assert_closed_clean_manifold(&fin_ok);
        }

        // UV-less split-edge box has no UVs, so split vertices remain open boundaries (not welded).
        let whole = box_mesh([0.0; 3], [1.0; 3]);
        let (mut split_pos, mut split_tris) = (Vec::new(), Vec::new());
        for pair in whole.triangles().as_chunks::<2>().0 {
            let base = (split_pos.len() / 3) as u32;
            let [a, b, c] = pair[0];
            for v in [a, b, c, pair[1][2]] {
                split_pos.extend(whole.position(v).unwrap());
            }
            split_tris.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]]);
        }
        let mut unseamed = mesh_of(&split_pos, &split_tris);
        let mut tx = unseamed.begin_edit();
        assert!(matches!(
            offset(
                &mut tx,
                &OffsetOptions::builder()
                    .distance(-0.1)
                    .algorithm(OffsetAlgorithm::FhEdt {
                        voxel_size: Some(0.1),
                    })
                    .build()
            ),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(tx.commit().unwrap().is_empty());
    }

    #[test]
    fn volumetric_offset_erodes_past_thickness_to_empty_mesh_cleanly() {
        for alg in [
            OffsetAlgorithm::FhEdt {
                voxel_size: Some(0.15),
            },
            OffsetAlgorithm::BvhSdf {
                voxel_size: Some(0.15),
            },
        ] {
            let mut m = box_mesh([0.0; 3], [1.0; 3]);
            let before = parts(&m);
            let mut tx = m.begin_edit();
            let r = offset(
                &mut tx,
                &OffsetOptions::builder()
                    .distance(-2.0)
                    .algorithm(alg)
                    .build(),
            )
            .unwrap();
            let patch = tx.commit().unwrap();
            assert!(!patch.is_empty());
            assert_eq!(r.triangles_added, 0);
            assert_eq!(r.vertices_added, 0);
            assert_eq!(r.triangles_removed, 12);
            assert_eq!(r.vertices_removed, 8);
            assert_eq!(m.triangle_count(), 0);
            assert_eq!(m.vertex_count(), 0);

            patch.revert(&mut m).unwrap();
            assert_eq!(parts(&m), before);
        }
    }

    #[test]
    fn volumetric_offset_rejects_inward_solid_bad_options_and_honours_oom_and_parallel() {
        use crate::test_util::{assert_parallel_identical, flip, sweep, uv_sphere};
        use fast_mesh_bvh::BuildOptions;

        // Default options preserve max_voxels = DEFAULT_MAX_VOXELS.
        assert_eq!(
            OffsetOptions::default(),
            OffsetOptions {
                distance: 0.0,
                algorithm: OffsetAlgorithm::Normal,
                max_voxels: fast_mesh_fields::DEFAULT_MAX_VOXELS,
            }
        );
        assert_eq!(OffsetOptions::builder().build(), OffsetOptions::default());

        // Flipped inward-wound closed box is rejected with exact message.
        for alg in [
            OffsetAlgorithm::FhEdt {
                voxel_size: Some(0.2),
            },
            OffsetAlgorithm::BvhSdf {
                voxel_size: Some(0.2),
            },
        ] {
            let mut m = flip(&box_mesh([0.0; 3], [1.0; 3]), &(0..12).collect::<Vec<_>>());
            let before = parts(&m);
            let mut tx = m.begin_edit();
            let err = offset(
                &mut tx,
                &OffsetOptions::builder()
                    .distance(0.1)
                    .algorithm(alg)
                    .build(),
            );
            assert_eq!(
                err,
                Err(OpsError::InvalidInput(
                    "volumetric offset input is wound inward"
                ))
            );
            assert!(tx.commit().unwrap().is_empty());
            assert_eq!(parts(&m), before);
        }

        // Bad voxel_size / max_voxels / exceeding max_voxels leave mesh untouched.
        for bad_opts in [
            OffsetOptions::builder()
                .distance(0.1)
                .algorithm(OffsetAlgorithm::FhEdt {
                    voxel_size: Some(0.0),
                })
                .build(),
            OffsetOptions::builder()
                .distance(0.1)
                .algorithm(OffsetAlgorithm::BvhSdf {
                    voxel_size: Some(-0.1),
                })
                .build(),
            OffsetOptions::builder()
                .distance(0.1)
                .algorithm(OffsetAlgorithm::FhEdt {
                    voxel_size: Some(f32::NAN),
                })
                .build(),
            OffsetOptions::builder()
                .distance(0.1)
                .algorithm(OffsetAlgorithm::BvhSdf {
                    voxel_size: Some(f32::INFINITY),
                })
                .build(),
            OffsetOptions::builder()
                .distance(0.1)
                .algorithm(OffsetAlgorithm::FhEdt { voxel_size: None })
                .max_voxels(8)
                .build(),
            OffsetOptions::builder()
                .distance(0.1)
                .algorithm(OffsetAlgorithm::BvhSdf {
                    voxel_size: Some(0.01),
                })
                .max_voxels(1_000)
                .build(),
        ] {
            let mut m = box_mesh([0.0; 3], [1.0; 3]);
            let before = parts(&m);
            let mut tx = m.begin_edit();
            assert!(matches!(
                offset(&mut tx, &bad_opts),
                Err(OpsError::InvalidInput(_))
            ));
            assert!(tx.commit().unwrap().is_empty());
            assert_eq!(parts(&m), before);
        }

        // Mismatched BVH is rejected with OpsError::Query.
        let other_mesh = uv_sphere(6, 8, 1.0);
        let wrong_bvh = Bvh::build(&other_mesh, &BuildOptions::default()).unwrap();
        let mut m = box_mesh([0.0; 3], [1.0; 3]);
        let before = parts(&m);
        let mut tx = m.begin_edit();
        assert!(matches!(
            offset_with_bvh(
                &mut tx,
                Some(&wrong_bvh),
                &OffsetOptions::builder()
                    .distance(0.1)
                    .algorithm(OffsetAlgorithm::BvhSdf {
                        voxel_size: Some(0.25),
                    })
                    .build()
            ),
            Err(OpsError::Query(_))
        ));
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&m), before);

        // OOM sweep on both FhEdt and BvhSdf (including with UVs and auto voxel_size clamping)
        // leaves tx completely empty on every failure point.
        for alg in [
            OffsetAlgorithm::FhEdt { voxel_size: None },
            OffsetAlgorithm::BvhSdf { voxel_size: None },
        ] {
            // Empty mesh is a no-op.
            let mut empty = Mesh::new();
            let mut tx = empty.begin_edit();
            assert_eq!(
                offset(
                    &mut tx,
                    &OffsetOptions::builder()
                        .distance(0.2)
                        .algorithm(alg)
                        .build()
                )
                .unwrap(),
                OffsetReport::default()
            );
            assert!(tx.commit().unwrap().is_empty());

            // All-degenerate mesh is rejected cleanly before mutating tx.
            let mut degen = mesh_of(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0], &[[0, 1, 2]]);
            let degen_before = parts(&degen);
            let mut tx = degen.begin_edit();
            assert!(matches!(
                offset(
                    &mut tx,
                    &OffsetOptions::builder()
                        .distance(0.2)
                        .algorithm(alg)
                        .build()
                ),
                Err(OpsError::InvalidInput(_))
            ));
            assert!(tx.commit().unwrap().is_empty());
            assert_eq!(parts(&degen), degen_before);

            let opts = OffsetOptions::builder()
                .distance(0.15)
                .algorithm(alg)
                .max_voxels(4_096)
                .build();
            let mut with_uvs = box_mesh([0.0; 3], [1.0; 3]);
            with_uvs.add_uvs(&[0.25; 16]).unwrap();
            let failures = sweep(with_uvs, |tx| offset(tx, &opts).map(drop));
            assert!(failures >= 5, "alg={alg:?} failures={failures}");
        }

        // Parallel determinism across 1, 2, 8 threads and global pool (auto voxel_size -> >= 64^3 grid).
        for (label, alg) in [
            (
                "volumetric_offset_fh_edt",
                OffsetAlgorithm::FhEdt {
                    voxel_size: Some(0.05),
                },
            ),
            (
                "volumetric_offset_bvh_sdf",
                OffsetAlgorithm::BvhSdf {
                    voxel_size: Some(0.05),
                },
            ),
        ] {
            let opts = OffsetOptions::builder()
                .distance(0.12)
                .algorithm(alg)
                .build();
            let (pos_bits, tris) = assert_parallel_identical(label, || {
                let mut m = uv_sphere(10, 14, 0.8);
                edit(&mut m, |tx| offset(tx, &opts));
                let (flat, tris) = parts(&m);
                let bits: Vec<u32> = flat.into_iter().map(f32::to_bits).collect();
                (bits, tris)
            });
            assert!(!pos_bits.is_empty() && !tris.is_empty());
        }
    }
}
