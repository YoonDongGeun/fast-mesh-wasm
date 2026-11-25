//! The heat method's operators and solves; see [`GeodesicHeatSolver`].

use std::f64::consts::FRAC_1_SQRT_2;
use std::ops::Range;

use fast_mesh_store::{Mesh, par};

use super::amg::{Amg, AmgBuild, Matrix, PAR_MIN_ROWS, Rows, Solve, sum_range};
use super::diffuse::{Diffusion, Heat, Surface};
use super::intrinsic::{Intrinsic, Layout, Survey, hypot, survey};
use super::{GeodesicPhase, MeshState, validate_max_distance, validate_options};
use crate::geom::{add, cross, dot, norm, scale, sub, triangle_normal, widen};
use crate::grid::{allocation_allowed, filled, push, reserve, reserve_exact};
use crate::{FieldsError, Result};

/// Minimum positive off-diagonal conductivity (`w_ij`) for clamped cotangent edges
/// so the positive-semidefinite Laplacian remains an irreducible symmetric M-matrix.
const MIN_EDGE_WEIGHT: f64 = 1e-6;
/// The Poisson solve ends at this error, in the energy norm and relative to
/// that of a distance field (whose gradient is 1 everywhere).
const TOLERANCE: f64 = 1e-7;
/// The diffusion's length is at most that of the sides of so many squares that
/// make up the whole surface: a mesh with fewer vertices gets more.
const RESOLUTION: f64 = 2000.0;
/// Elements per chunk of the assembly's loops.
pub(super) const CHUNK: usize = 8192;
/// Triangles per step of the gradient.
const GRADIENT_STEP: usize = 1 << 17;
/// No component.
const NONE: u32 = u32::MAX;

const OOM: fn() -> FieldsError = || FieldsError::OutOfMemory;

/// One connected part of the mesh, in its own vertex numbering, which
/// follows a space-filling curve: neighbours on the surface are mostly
/// neighbours in memory.
#[derive(Debug, Clone)]
struct Component {
    /// The mesh's vertex per vertex, or [`NONE`] for one it has not got.
    vertices: Vec<u32>,
    positions: Vec<[f32; 3]>,
    /// The triangles with an area, ordered by their first vertex: those of
    /// the mesh, and those that are cut otherwise.
    triangles: Vec<[u32; 3]>,
    flat: Vec<Flat>,
    /// Lumped mass: a third of the area around each vertex.
    mass: Vec<f64>,
    area: f64,
    /// The cotangent Laplacian and its multigrid levels.
    amg: Amg,
}

impl Component {
    fn byte_size(&self) -> usize {
        self.vertices.capacity() * size_of::<u32>()
            + (self.positions.capacity() + self.triangles.capacity()) * size_of::<[u32; 3]>()
            + self.flat.capacity() * size_of::<Flat>()
            + self.mass.capacity() * size_of::<f64>()
            + self.amg.byte_size()
    }
}

/// A triangle that is not the mesh's, laid out flat: the length of its first
/// side, and where its third corner is from there (as in [`Layout`]).
#[derive(Debug, Clone, Copy)]
pub(super) struct Flat {
    pub(super) corners: [u32; 3],
    shape: [f32; 3],
}

impl Flat {
    pub(super) fn new(corners: [u32; 3], layout: Layout) -> Flat {
        let [length, x, y] = [layout.length, layout.x, layout.y].map(|c| c as f32);
        // (Never without an area, however small.)
        Flat {
            corners,
            shape: [length.max(f32::MIN_POSITIVE), x, y.max(f32::MIN_POSITIVE)],
        }
    }

    fn layout(&self) -> Layout {
        let [length, x, y] = self.shape.map(f64::from);
        Layout { length, x, y }
    }

    fn area(&self) -> f64 {
        0.5 * f64::from(self.shape[0]) * f64::from(self.shape[2])
    }
}

/// Precomputed solver for Crane et al. (2013) *Geodesics in Heat* on a fixed
/// mesh.
///
/// Building the solver assembles the lumped mass matrix, the cotangent
/// Laplacian and its multigrid levels per connected component, so subsequent
/// [`GeodesicHeatSolver::solve`] calls for different source sets avoid
/// rebuilding the operators.
///
/// The operators are those of the mesh's own triangles as long as these are
/// good enough, and of an intrinsic Delaunay triangulation of the same
/// surface otherwise: where an edge's opposite angles sum to more than π, or
/// where a triangle is large next to the time step. See
/// [`GeodesicMethod::HeatMethod`](super::GeodesicMethod::HeatMethod) for how
/// far off the distances are.
#[derive(Debug, Clone)]
pub struct GeodesicHeatSolver {
    mesh: MeshState,
    source_valid: Vec<bool>,
    component_of: Vec<u32>,
    local_of: Vec<u32>,
    components: Vec<Component>,
    /// The diffusion's length: its time step is the square of it.
    length: f64,
    time_step_scale: f64,
}

impl GeodesicHeatSolver {
    /// Builds a reusable heat-method solver for `mesh` using diffusion time step
    /// `t = time_step_scale * h^2`, where `h` is the mean edge length across
    /// unique edges of live non-degenerate triangles, or `sqrt(area / 2000)`
    /// where that is smaller (on a mesh of less than about 2000 vertices).
    ///
    /// # Errors
    /// - [`FieldsError::InvalidInput`] if `time_step_scale` is non-finite or
    ///   `<= 0.0`, or if any vertex referenced by a live triangle has a
    ///   non-finite coordinate.
    /// - [`FieldsError::OutOfMemory`] if any internal buffer cannot be reserved.
    pub fn build(mesh: &Mesh, time_step_scale: f64) -> Result<Self> {
        let mut build = SolverBuild::new(mesh, time_step_scale)?;
        while build.step(mesh)?.is_some() {}
        Ok(build.finish())
    }

    /// Number of vertex slots on the mesh this solver was built for.
    #[must_use]
    pub fn vertex_slots(&self) -> usize {
        self.mesh.vertex_slots
    }

    /// Whether the solver was built for `mesh` as it is now.
    #[must_use]
    pub fn is_for(&self, mesh: &Mesh) -> bool {
        self.mesh == MeshState::of(mesh)
    }

    /// Multiplier of the squared mean edge length in the diffusion time step.
    #[must_use]
    pub fn time_step_scale(&self) -> f64 {
        self.time_step_scale
    }

    /// Changes the diffusion time step; nothing needs to be rebuilt.
    ///
    /// # Errors
    /// [`FieldsError::InvalidInput`] if `time_step_scale` is non-finite or `<= 0.0`.
    pub fn set_time_step_scale(&mut self, time_step_scale: f64) -> Result<()> {
        validate_options(time_step_scale, None)?;
        self.time_step_scale = time_step_scale;
        Ok(())
    }

    /// Approximate heap memory the solver holds.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.source_valid.capacity()
            + (self.component_of.capacity() + self.local_of.capacity()) * size_of::<u32>()
            + self
                .components
                .iter()
                .map(|c| size_of::<Component>() + c.byte_size())
                .sum::<usize>()
    }

    /// Computes Crane heat-method geodesic distances from `sources`, writing
    /// `self.vertex_slots()` distances into `out`.
    ///
    /// Vertices in components containing no source, isolated alive vertices not
    /// in `sources`, dead slots, and distances strictly greater than
    /// `max_distance` receive `f32::INFINITY`.
    ///
    /// # Errors
    /// - [`FieldsError::InvalidInput`] if `sources` is empty, any source is out
    ///   of range / dead / non-finite, or `max_distance` is negative or non-finite.
    /// - [`FieldsError::OutOfMemory`] if working or output buffers cannot be reserved.
    ///
    /// On error `out` is left unchanged.
    pub fn solve(
        &self,
        sources: &[u32],
        max_distance: Option<f32>,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        let mut solve = HeatSolve::new(self, sources, max_distance, self.time_step_scale)?;
        while solve.step(self, usize::MAX)?.is_some() {}
        super::write_out(&solve.finish(), out)
    }
}

/// [`GeodesicHeatSolver`] under construction.
#[derive(Debug)]
pub(super) struct SolverBuild {
    solver: GeodesicHeatSolver,
    stage: BuildStage,
    /// Whether the triangles may be cut otherwise.
    recut: bool,
}

#[derive(Debug)]
enum BuildStage {
    /// Collect the triangles with an area.
    Triangles,
    /// Find the connected parts and number their vertices.
    Number {
        triangles: Vec<[u32; 3]>,
    },
    /// Renumber the triangles.
    Renumber {
        triangles: Vec<[u32; 3]>,
        /// The mesh's vertex per vertex, part after part.
        vertices: Vec<u32>,
        /// Where each part starts in `vertices`, and where the last ends.
        starts: Vec<u32>,
    },
    /// Look at the triangles.
    Survey(Numbered),
    /// Assemble the Laplacian of every part.
    Assemble {
        numbered: Numbered,
        corners: Corners,
        cots: Vec<[f64; 3]>,
        /// What the mesh's own triangles are like.
        survey: Option<Survey>,
    },
    /// Flip edges until the triangulation is Delaunay.
    Flip {
        intrinsic: Box<Intrinsic>,
        area: f64,
    },
    /// Put vertices into the triangles that are too large.
    Refine(Box<Intrinsic>),
    /// Cut it into its parts.
    Split {
        numbered: Numbered,
        laplacian: Rows,
    },
    /// Build the multigrid levels of the last component pushed, then of the
    /// parts left.
    Levels {
        parts: std::vec::IntoIter<Part>,
        current: Option<Box<(Part, AmgBuild)>>,
    },
    Done,
}

/// The triangles in the solver's numbering, part after part.
#[derive(Debug)]
pub(super) struct Numbered {
    /// The mesh's vertex per vertex, or [`NONE`] for one it has not got.
    pub(super) vertices: Vec<u32>,
    /// Where each part's vertices start, and where the last part's end.
    pub(super) starts: Vec<u32>,
    pub(super) positions: Vec<[f32; 3]>,
    /// Those of the mesh, ordered by their smallest vertex.
    pub(super) triangles: Vec<[u32; 3]>,
    /// Those cut otherwise, likewise.
    pub(super) flat: Vec<Flat>,
}

impl Numbered {
    pub(super) fn parts(&self) -> usize {
        self.starts.len() - 1
    }

    /// The part a vertex is in.
    pub(super) fn part_of(&self, v: u32) -> usize {
        self.starts.partition_point(|&s| s <= v) - 1
    }

    /// How many triangles there are, flat ones and others.
    fn count(&self) -> usize {
        self.triangles.len() + self.flat.len()
    }

    /// The corners of a triangle, the flat ones numbered after the others.
    fn corners(&self, t: usize) -> [u32; 3] {
        match self.triangles.get(t) {
            Some(&tri) => tri,
            None => self.flat[t - self.triangles.len()].corners,
        }
    }

    pub(super) fn layout(&self, t: usize) -> Layout {
        match self.triangles.get(t) {
            Some(tri) => Layout::of(tri.map(|v| self.positions[v as usize])),
            None => self.flat[t - self.triangles.len()].layout(),
        }
    }
}

/// The triangles around each vertex.
#[derive(Debug)]
pub(super) struct Corners {
    /// Where each vertex's start.
    start: Vec<u32>,
    /// `3 t + k` for corner `k` of triangle `t`, ascending.
    around: Vec<u32>,
}

impl Corners {
    fn of(mesh: &Numbered) -> Result<Corners> {
        let (n, count) = (mesh.vertices.len(), mesh.count());
        if 3 * count as u64 >= u64::from(NONE) {
            return Err(FieldsError::OutOfMemory);
        }
        let mut start = filled(n + 1, 0u32)?;
        for t in 0..count {
            for v in mesh.corners(t) {
                start[v as usize + 1] += 1;
            }
        }
        for v in 0..n {
            start[v + 1] += start[v];
        }
        let mut at = filled(n, 0u32)?;
        at.copy_from_slice(&start[..n]);
        let mut around = filled(3 * count, 0u32)?;
        for t in 0..count {
            for (k, v) in mesh.corners(t).into_iter().enumerate() {
                let slot = &mut at[v as usize];
                around[*slot as usize] = (3 * t + k) as u32;
                *slot += 1;
            }
        }
        Ok(Corners { start, around })
    }

    pub(super) fn around(&self, v: u32) -> &[u32] {
        let v = v as usize;
        &self.around[self.start[v] as usize..self.start[v + 1] as usize]
    }
}

/// A connected part before its multigrid levels are built.
#[derive(Debug)]
struct Part {
    vertices: Vec<u32>,
    positions: Vec<[f32; 3]>,
    triangles: Vec<[u32; 3]>,
    flat: Vec<Flat>,
    mass: Vec<f64>,
    area: f64,
    laplacian: Option<Rows>,
}

impl SolverBuild {
    pub(super) fn new(mesh: &Mesh, time_step_scale: f64) -> Result<SolverBuild> {
        validate_options(time_step_scale, None)?;
        Ok(SolverBuild {
            solver: GeodesicHeatSolver {
                mesh: MeshState::of(mesh),
                source_valid: Vec::new(),
                component_of: Vec::new(),
                local_of: Vec::new(),
                components: Vec::new(),
                length: 0.0,
                time_step_scale,
            },
            stage: BuildStage::Triangles,
            recut: true,
        })
    }

    pub(super) fn phase(&self) -> GeodesicPhase {
        match self.stage {
            BuildStage::Levels { .. } | BuildStage::Done => GeodesicPhase::Factorize,
            _ => GeodesicPhase::Assemble,
        }
    }

    /// Runs the next pass; about how many elements it went over, or `None`
    /// when the solver is built.
    pub(super) fn step(&mut self, mesh: &Mesh) -> Result<Option<usize>> {
        let s = &mut self.solver;
        let work;
        self.stage = match std::mem::replace(&mut self.stage, BuildStage::Done) {
            BuildStage::Triangles => {
                work = mesh.triangle_slots();
                s.source_valid = valid_sources(mesh)?;
                BuildStage::Number {
                    triangles: proper_triangles(mesh)?,
                }
            }
            BuildStage::Number { triangles } => {
                work = triangles.len();
                let (vertices, starts) =
                    number(mesh, &triangles, &mut s.component_of, &mut s.local_of)?;
                BuildStage::Renumber {
                    triangles,
                    vertices,
                    starts,
                }
            }
            BuildStage::Renumber {
                triangles,
                vertices,
                starts,
            } => {
                work = triangles.len();
                let renumber =
                    |v: u32| starts[s.component_of[v as usize] as usize] + s.local_of[v as usize];
                let mut triangles = triangles;
                for tri in &mut triangles {
                    *tri = tri.map(renumber);
                }
                let triangles = by_first_corner(&triangles, vertices.len(), |t| *t)?;
                let mut positions: Vec<[f32; 3]> = Vec::new();
                reserve_exact(&mut positions, vertices.len())?;
                positions.extend(vertices.iter().map(|&v| mesh.positions()[v as usize]));
                BuildStage::Survey(Numbered {
                    vertices,
                    starts,
                    positions,
                    triangles,
                    flat: Vec::new(),
                })
            }
            BuildStage::Survey(numbered) => {
                work = 2 * numbered.count();
                let corners = Corners::of(&numbered)?;
                let cots = half_cotangents(&numbered)?;
                let survey = survey(&numbered, &corners, &cots)?;
                if survey.delaunay || !self.recut {
                    BuildStage::Assemble {
                        numbered,
                        corners,
                        cots,
                        survey: Some(survey),
                    }
                } else {
                    flip_stage(numbered, &corners, &cots, survey.area)?
                }
            }
            BuildStage::Assemble {
                numbered,
                corners,
                cots,
                survey,
            } => {
                work = 2 * numbered.count();
                let (laplacian, mean) = assemble(&numbered, &corners, &cots)?;
                match survey {
                    Some(survey) => {
                        let (length, size) = length_and_size(mean, survey.area);
                        s.length = if self.recut { length } else { mean };
                        if self.recut && survey.radius > size {
                            flip_stage(numbered, &corners, &cots, survey.area)?
                        } else {
                            BuildStage::Split {
                                numbered,
                                laplacian,
                            }
                        }
                    }
                    None => BuildStage::Split {
                        numbered,
                        laplacian,
                    },
                }
            }
            BuildStage::Flip {
                mut intrinsic,
                area,
            } => {
                work = 1 << 18;
                if intrinsic.flip()? {
                    let (length, size) = length_and_size(intrinsic.mean_edge(), area);
                    s.length = length;
                    intrinsic.begin_refining(size)?;
                    BuildStage::Refine(intrinsic)
                } else {
                    BuildStage::Flip { intrinsic, area }
                }
            }
            BuildStage::Refine(mut intrinsic) => {
                work = 1 << 18;
                if intrinsic.refine()? {
                    let numbered = intrinsic.finish(&mut s.local_of)?;
                    let corners = Corners::of(&numbered)?;
                    let cots = half_cotangents(&numbered)?;
                    BuildStage::Assemble {
                        numbered,
                        corners,
                        cots,
                        survey: None,
                    }
                } else {
                    BuildStage::Refine(intrinsic)
                }
            }
            BuildStage::Split {
                numbered,
                laplacian,
            } => {
                work = numbered.count();
                let parts = split(numbered, laplacian)?;
                reserve_exact(&mut s.components, parts.len())?;
                BuildStage::Levels {
                    parts: parts.into_iter(),
                    current: None,
                }
            }
            BuildStage::Levels { mut parts, current } => match current {
                Some(mut building) => {
                    work = 4 * building.0.vertices.len();
                    if building.1.step()? {
                        let (part, amg) = *building;
                        // Reserved with the parts.
                        s.components.push(Component {
                            vertices: part.vertices,
                            positions: part.positions,
                            triangles: part.triangles,
                            flat: part.flat,
                            mass: part.mass,
                            area: part.area,
                            amg: amg.finish(),
                        });
                        BuildStage::Levels {
                            parts,
                            current: None,
                        }
                    } else {
                        BuildStage::Levels {
                            parts,
                            current: Some(building),
                        }
                    }
                }
                None => match parts.next() {
                    Some(mut part) => {
                        work = part.vertices.len();
                        let matrix = laplacian_matrix(part.laplacian.take().unwrap_or_default())?;
                        BuildStage::Levels {
                            parts,
                            current: Some(Box::new((part, AmgBuild::new(matrix)))),
                        }
                    }
                    None => return Ok(None),
                },
            },
            BuildStage::Done => return Ok(None),
        };
        Ok(Some(work.max(1)))
    }

    pub(super) fn finish(self) -> GeodesicHeatSolver {
        self.solver
    }
}

/// The length of the diffusion on a surface whose edges have a mean length of
/// `mean`, and the largest circumradius its triangles may have.
fn length_and_size(mean: f64, area: f64) -> (f64, f64) {
    let side = (area / RESOLUTION).sqrt();
    (mean.min(side), mean.min(side * FRAC_1_SQRT_2))
}

/// Starts to cut the surface otherwise.
fn flip_stage(
    mesh: Numbered,
    corners: &Corners,
    cots: &[[f64; 3]],
    area: f64,
) -> Result<BuildStage> {
    // (About what it takes: triangles of the largest size allowed.)
    let extra = 4.0 * RESOLUTION;
    Ok(BuildStage::Flip {
        intrinsic: Box::new(Intrinsic::new(mesh, corners, cots, area, extra)?),
        area,
    })
}

/// Which vertex slots can be sources: the live ones with a finite position.
fn valid_sources(mesh: &Mesh) -> Result<Vec<bool>> {
    let positions = mesh.positions();
    let mut valid = filled(mesh.vertex_slots(), false)?;
    for (v, flag) in valid.iter_mut().enumerate() {
        *flag = mesh.is_vertex_alive(v as u32) && positions[v].iter().all(|c| c.is_finite());
    }
    Ok(valid)
}

/// The live triangles with three corners and an area, in slot order.
fn proper_triangles(mesh: &Mesh) -> Result<Vec<[u32; 3]>> {
    let positions = mesh.positions();
    allocation_allowed()?;
    let parts = par::map_range(mesh.triangle_slots(), CHUNK, 2 * CHUNK, OOM, |range| {
        let mut kept: Vec<[u32; 3]> = Vec::new();
        reserve_exact(&mut kept, range.len())?;
        for t in range {
            let Some(tri) = mesh.triangle(t as u32) else {
                continue;
            };
            let corners = tri.map(|v| positions[v as usize]);
            if !corners.iter().flatten().all(|c| c.is_finite()) {
                return Err(FieldsError::InvalidInput(
                    "mesh vertex position must be finite",
                ));
            }
            if tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0] {
                continue;
            }
            let [a, b, c] = corners.map(widen);
            let area2 = norm(triangle_normal(corners));
            let longest = [sub(c, b), sub(a, c), sub(b, a)]
                .iter()
                .map(|&e| dot(e, e))
                .fold(0.0, f64::max);
            if area2.is_finite() && area2 > 1e-24 * longest && area2 > 0.0 {
                kept.push(tri);
            }
        }
        Ok(kept)
    })?;
    let mut out = Vec::new();
    reserve_exact(&mut out, parts.iter().map(Vec::len).sum())?;
    for part in parts {
        out.extend_from_slice(&part);
    }
    Ok(out)
}

/// Finds the connected parts of the triangles and numbers the vertices of
/// each along a space-filling curve. Fills `component_of` and `local_of`;
/// returns the vertices part after part, and where each part starts. The
/// parts come in the order of their first vertex.
fn number(
    mesh: &Mesh,
    triangles: &[[u32; 3]],
    component_of: &mut Vec<u32>,
    local_of: &mut Vec<u32>,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let slots = mesh.vertex_slots();
    let positions = mesh.positions();
    // Every vertex's parent: the smallest vertex of a part is its root.
    let mut parent = filled(slots, NONE)?;
    fn root(parent: &mut [u32], mut v: u32) -> u32 {
        while parent[v as usize] != v {
            let up = parent[parent[v as usize] as usize];
            parent[v as usize] = up;
            v = up;
        }
        v
    }
    for tri in triangles {
        for &v in tri {
            if parent[v as usize] == NONE {
                parent[v as usize] = v;
            }
        }
        let a = root(&mut parent, tri[0]);
        let b = root(&mut parent, tri[1]);
        let ab = a.min(b);
        parent[a.max(b) as usize] = ab;
        let c = root(&mut parent, tri[2]);
        parent[ab.max(c) as usize] = ab.min(c);
    }

    *component_of = filled(slots, NONE)?;
    *local_of = filled(slots, 0u32)?;
    let mut sizes: Vec<u32> = Vec::new();
    let (mut lo, mut hi) = NOWHERE;
    let mut live = 0usize;
    for v in 0..slots {
        if parent[v] == NONE {
            continue;
        }
        // A root comes before the rest of its part.
        let r = root(&mut parent, v as u32) as usize;
        if r == v {
            component_of[v] = sizes.len() as u32;
            push(&mut sizes, 0)?;
        }
        let c = component_of[r];
        component_of[v] = c;
        sizes[c as usize] += 1;
        live += 1;
        include(&mut lo, &mut hi, positions[v]);
    }
    drop(parent);

    let mut keys: Vec<(u64, u32)> = Vec::new();
    reserve_exact(&mut keys, live)?;
    for v in 0..slots {
        if component_of[v] != NONE {
            keys.push((morton(positions[v], lo, hi), v as u32));
        }
    }
    keys.sort_unstable();

    let mut starts = filled(sizes.len() + 1, 0u32)?;
    for (k, &size) in sizes.iter().enumerate() {
        starts[k + 1] = starts[k] + size;
    }
    let mut vertices = filled(live, 0u32)?;
    sizes.fill(0);
    for &(_, v) in &keys {
        let c = component_of[v as usize] as usize;
        let local = sizes[c];
        sizes[c] += 1;
        local_of[v as usize] = local;
        vertices[(starts[c] + local) as usize] = v;
    }
    Ok((vertices, starts))
}

/// A box without a point in it.
const NOWHERE: ([f32; 3], [f32; 3]) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);

fn include(lo: &mut [f32; 3], hi: &mut [f32; 3], p: [f32; 3]) {
    for k in 0..3 {
        lo[k] = lo[k].min(p[k]);
        hi[k] = hi[k].max(p[k]);
    }
}

/// The box around some points.
pub(super) fn bounds(points: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let (mut lo, mut hi) = NOWHERE;
    for &p in points {
        include(&mut lo, &mut hi, p);
    }
    (lo, hi)
}

/// A point's place along a Morton curve through the box, 21 bits per axis.
pub(super) fn morton(p: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> u64 {
    let mut code = 0u64;
    for k in 0..3 {
        let (l, h) = (f64::from(lo[k]), f64::from(hi[k]));
        let at = (f64::from(p[k]) - l) / (h - l) * 2_097_151.0;
        // (A box without extent along the axis gives NaN, which casts to 0.)
        let mut x = at as u64 & 0x1f_ffff;
        x = (x | x << 32) & 0x001f_0000_0000_ffff;
        x = (x | x << 16) & 0x001f_0000_ff00_00ff;
        x = (x | x << 8) & 0x100f_00f0_0f00_f00f;
        x = (x | x << 4) & 0x10c3_0c30_c30c_30c3;
        x = (x | x << 2) & 0x1249_2492_4924_9249;
        code |= x << k;
    }
    code
}

/// Triangles ordered by their smallest corner, then as before.
pub(super) fn by_first_corner<T: Copy>(
    triangles: &[T],
    vertices: usize,
    corners: impl Fn(&T) -> [u32; 3],
) -> Result<Vec<T>> {
    let first = |t: &T| {
        let [a, b, c] = corners(t);
        a.min(b).min(c) as usize
    };
    let Some(&any) = triangles.first() else {
        return Ok(Vec::new());
    };
    let mut at = filled(vertices + 1, 0u32)?;
    for t in triangles {
        at[first(t) + 1] += 1;
    }
    for v in 0..vertices {
        at[v + 1] += at[v];
    }
    let mut out = filled(triangles.len(), any)?;
    for t in triangles {
        let slot = &mut at[first(t)];
        out[*slot as usize] = *t;
        *slot += 1;
    }
    Ok(out)
}

/// Half the cotangent at each corner of every triangle.
fn half_cotangents(mesh: &Numbered) -> Result<Vec<[f64; 3]>> {
    let mut cots = filled(mesh.count(), [0.0f64; 3])?;
    par::for_each_chunk_mut(&mut cots, CHUNK, 2 * CHUNK, |c, part| {
        for (i, out) in part.iter_mut().enumerate() {
            let t = c * CHUNK + i;
            *out = match mesh.triangles.get(t) {
                Some(tri) => half_cotangents_of(tri.map(|v| mesh.positions[v as usize])),
                None => mesh.layout(t).half_cotangents(),
            };
        }
    });
    Ok(cots)
}

/// Half the cotangent at each corner of a triangle.
fn half_cotangents_of(corners: [[f32; 3]; 3]) -> [f64; 3] {
    let [a, b, c] = corners.map(widen);
    let inv = 1.0 / norm(triangle_normal(corners));
    [
        0.5 * dot(sub(b, a), sub(c, a)) * inv,
        0.5 * dot(sub(c, b), sub(a, b)) * inv,
        0.5 * dot(sub(a, c), sub(b, c)) * inv,
    ]
}

/// Off-diagonal Laplacian rows of the vertices of a range, columns local to
/// each vertex's part, with the lengths of their edges to later vertices.
#[derive(Default)]
struct RowsPart {
    rows: Rows,
    length: f64,
    edges: usize,
}

/// The off-diagonal entries of the cotangent Laplacian, columns numbered
/// within each part, and the mean length of the edges (if they are straight).
fn assemble(mesh: &Numbered, corners: &Corners, cots: &[[f64; 3]]) -> Result<(Rows, f64)> {
    let n = mesh.vertices.len();
    allocation_allowed()?;
    let parts = par::map_range(n, CHUNK, 2 * CHUNK, OOM, |range| {
        let mut part = RowsPart::default();
        reserve_exact(&mut part.rows.ptr, range.len())?;
        // Column, side, weight: the side keeps the order fixed, and the same
        // in the rows of both its ends.
        let mut entries: Vec<(u32, u32, f64)> = Vec::new();
        let mut of = mesh.part_of(range.start as u32);
        for v in range {
            while mesh.starts[of + 1] as usize <= v {
                of += 1;
            }
            let base = mesh.starts[of];
            entries.clear();
            for &corner in corners.around(v as u32) {
                let (t, k) = (corner as usize / 3, corner as usize % 3);
                let (tri, cot) = (mesh.corners(t), cots[t]);
                let (a, b) = ((k + 1) % 3, (k + 2) % 3);
                // (A side can lead back to where it started.)
                if tri[a] as usize != v {
                    push(&mut entries, (tri[a], corner, cot[b]))?;
                }
                if tri[b] as usize != v {
                    push(&mut entries, (tri[b], (3 * t + b) as u32, cot[a]))?;
                }
            }
            entries.sort_unstable_by_key(|e| (e.0, e.1));
            let p = widen(mesh.positions[v]);
            reserve(&mut part.rows.col, entries.len())?;
            reserve(&mut part.rows.val, entries.len())?;
            let mut e = 0;
            while e < entries.len() {
                let (c, _, mut w) = entries[e];
                e += 1;
                while e < entries.len() && entries[e].0 == c {
                    w += entries[e].2;
                    e += 1;
                }
                part.rows.col.push(c - base);
                part.rows.val.push(-w.max(MIN_EDGE_WEIGHT));
                if c as usize > v {
                    part.length += norm(sub(widen(mesh.positions[c as usize]), p));
                    part.edges += 1;
                }
            }
            part.rows.ptr.push(part.rows.col.len() as u32);
        }
        Ok(part)
    })?;

    let total: usize = parts.iter().map(|p| p.rows.col.len()).sum();
    if u32::try_from(total).is_err() {
        return Err(FieldsError::OutOfMemory);
    }
    let mut rows = Rows::default();
    reserve_exact(&mut rows.ptr, n + 1)?;
    reserve_exact(&mut rows.col, total)?;
    reserve_exact(&mut rows.val, total)?;
    rows.ptr.push(0);
    let (mut length, mut edges) = (0.0, 0usize);
    for part in parts {
        let base = rows.col.len() as u32;
        rows.ptr.extend(part.rows.ptr.iter().map(|p| p + base));
        rows.col.extend_from_slice(&part.rows.col);
        rows.val.extend_from_slice(&part.rows.val);
        length += part.length;
        edges += part.edges;
    }
    let mean = if edges > 0 {
        length / edges as f64
    } else {
        0.0
    };
    Ok((rows, mean))
}

/// Cuts the numbered mesh and its Laplacian into their parts.
fn split(mesh: Numbered, laplacian: Rows) -> Result<Vec<Part>> {
    let count = mesh.starts.len() - 1;
    let mut parts = Vec::new();
    reserve_exact(&mut parts, count)?;
    if count == 1 {
        let (mass, area) = masses(&mesh.positions, &mesh.triangles, &mesh.flat)?;
        parts.push(Part {
            vertices: mesh.vertices,
            positions: mesh.positions,
            triangles: mesh.triangles,
            flat: mesh.flat,
            mass,
            area,
            laplacian: Some(laplacian),
        });
        return Ok(parts);
    }
    fn copied<T: Copy>(from: &[T]) -> Result<Vec<T>> {
        let mut v = Vec::new();
        reserve_exact(&mut v, from.len())?;
        v.extend_from_slice(from);
        Ok(v)
    }
    // (Ordered by their smallest vertex, the triangles come part after part.)
    let (mut t, mut f) = (0, 0);
    for k in 0..count {
        let range = mesh.starts[k] as usize..mesh.starts[k + 1] as usize;
        let (from, from_flat) = (t, f);
        while t < mesh.triangles.len() && (mesh.triangles[t][0] as usize) < range.end {
            t += 1;
        }
        while f < mesh.flat.len() && (mesh.flat[f].corners[0] as usize) < range.end {
            f += 1;
        }
        let mut triangles = copied(&mesh.triangles[from..t])?;
        let mut flat = copied(&mesh.flat[from_flat..f])?;
        let own = triangles.iter_mut().flatten();
        for v in own.chain(flat.iter_mut().flat_map(|f| &mut f.corners)) {
            *v -= range.start as u32;
        }
        let positions = copied(&mesh.positions[range.clone()])?;
        let (mass, area) = masses(&positions, &triangles, &flat)?;
        let entries = laplacian.ptr[range.start] as usize..laplacian.ptr[range.end] as usize;
        let mut ptr = copied(&laplacian.ptr[range.start..=range.end])?;
        for p in &mut ptr {
            *p -= entries.start as u32;
        }
        parts.push(Part {
            vertices: copied(&mesh.vertices[range])?,
            positions,
            triangles,
            flat,
            mass,
            area,
            laplacian: Some(Rows {
                ptr,
                col: copied(&laplacian.col[entries.clone()])?,
                val: copied(&laplacian.val[entries])?,
            }),
        });
    }
    Ok(parts)
}

/// A third of the area around each vertex, and the whole area.
fn masses(
    positions: &[[f32; 3]],
    triangles: &[[u32; 3]],
    flat: &[Flat],
) -> Result<(Vec<f64>, f64)> {
    let mut mass = filled(positions.len(), 0.0f64)?;
    let mut total = 0.0;
    let mut add = |tri: &[u32; 3], area: f64| {
        total += area;
        for &v in tri {
            mass[v as usize] += area / 3.0;
        }
    };
    for tri in triangles {
        let corners = tri.map(|v| positions[v as usize]);
        add(tri, 0.5 * norm(triangle_normal(corners)));
    }
    for f in flat {
        add(&f.corners, f.area());
    }
    Ok((mass, total))
}

/// The Laplacian of its off-diagonal rows: every row sums to zero.
fn laplacian_matrix(off: Rows) -> Result<Matrix> {
    let n = off.ptr.len().saturating_sub(1);
    let mut diag = filled(n, 0.0f64)?;
    for (i, d) in diag.iter_mut().enumerate() {
        *d = -off.val[off.range(i)].iter().sum::<f64>();
    }
    Matrix::new(off, diag)
}

/// A solve under way, for one source set.
#[derive(Debug)]
pub(super) struct HeatSolve {
    max_distance: Option<f32>,
    time_step_scale: f64,
    /// The components with sources, last first, and their sources there.
    todo: Vec<(u32, Vec<u32>)>,
    stage: SolveStage,
    result: Vec<f32>,
}

#[derive(Debug)]
enum SolveStage {
    /// Take the next component.
    Next,
    Diffuse {
        component: u32,
        sources: Vec<u32>,
        diffusion: Box<Diffusion>,
    },
    /// Turn the heat's direction into the right-hand side.
    Gradient {
        component: u32,
        sources: Vec<u32>,
        heat: Heat,
        b: Vec<f64>,
        /// The first triangle not added yet.
        at: usize,
    },
    Poisson {
        component: u32,
        sources: Vec<u32>,
        solve: Box<Solve>,
    },
    Done,
}

impl HeatSolve {
    pub(super) fn new(
        solver: &GeodesicHeatSolver,
        sources: &[u32],
        max_distance: Option<f32>,
        time_step_scale: f64,
    ) -> Result<HeatSolve> {
        validate_max_distance(max_distance)?;
        if sources.is_empty() {
            return Err(FieldsError::InvalidInput("sources must not be empty"));
        }
        for &s in sources {
            let idx = s as usize;
            if idx >= solver.mesh.vertex_slots || !solver.source_valid[idx] {
                return Err(FieldsError::InvalidInput(
                    "source vertex index is out of range, dead, or non-finite",
                ));
            }
        }
        let mut result = filled(solver.mesh.vertex_slots, f32::INFINITY)?;
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        reserve_exact(&mut pairs, sources.len())?;
        for &s in sources {
            match solver.component_of.get(s as usize) {
                Some(&c) if c != NONE => pairs.push((c, solver.local_of[s as usize])),
                _ => result[s as usize] = 0.0,
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        let mut todo: Vec<(u32, Vec<u32>)> = Vec::new();
        for group in pairs.chunk_by(|a, b| a.0 == b.0).rev() {
            let mut local = Vec::new();
            reserve_exact(&mut local, group.len())?;
            local.extend(group.iter().map(|p| p.1));
            push(&mut todo, (group[0].0, local))?;
        }
        Ok(HeatSolve {
            max_distance,
            time_step_scale,
            todo,
            stage: SolveStage::Next,
            result,
        })
    }

    pub(super) fn phase(&self) -> GeodesicPhase {
        match self.stage {
            SolveStage::Gradient { .. } => GeodesicPhase::Gradient,
            SolveStage::Poisson { .. } => GeodesicPhase::Poisson,
            _ => GeodesicPhase::Diffuse,
        }
    }

    /// Runs the next part of about `work` elements; how many it went over,
    /// or `None` when the distances are done.
    pub(super) fn step(
        &mut self,
        solver: &GeodesicHeatSolver,
        work: usize,
    ) -> Result<Option<usize>> {
        let time_step = self.time_step_scale * solver.length * solver.length;
        let spent;
        self.stage = match std::mem::replace(&mut self.stage, SolveStage::Done) {
            SolveStage::Next => match self.todo.pop() {
                None => return Ok(None),
                Some((component, sources)) => {
                    let n = solver.components[component as usize].vertices.len();
                    spent = n;
                    let diffusion = Box::new(Diffusion::new(n, &sources)?);
                    SolveStage::Diffuse {
                        component,
                        sources,
                        diffusion,
                    }
                }
            },
            SolveStage::Diffuse {
                component,
                sources,
                mut diffusion,
            } => {
                let c = &solver.components[component as usize];
                let surface = Surface {
                    matrix: c.amg.fine(),
                    mass: &c.mass,
                    positions: &c.positions,
                    time_step,
                };
                match diffusion.step(&surface, work)? {
                    Some(done) => {
                        spent = done;
                        SolveStage::Diffuse {
                            component,
                            sources,
                            diffusion,
                        }
                    }
                    None => {
                        spent = c.vertices.len();
                        SolveStage::Gradient {
                            component,
                            sources,
                            heat: diffusion.finish(),
                            b: filled(c.vertices.len(), 0.0f64)?,
                            at: 0,
                        }
                    }
                }
            }
            SolveStage::Gradient {
                component,
                sources,
                heat,
                mut b,
                at,
            } => {
                let c = &solver.components[component as usize];
                let count = c.triangles.len() + c.flat.len();
                let end = count.min(at + GRADIENT_STEP);
                add_divergence(c, &heat, at..end, &mut b)?;
                spent = 2 * (end - at) + 1;
                if end < count {
                    SolveStage::Gradient {
                        component,
                        sources,
                        heat,
                        b,
                        at: end,
                    }
                } else {
                    let solve = Box::new(begin_poisson(c, heat, b, time_step)?);
                    SolveStage::Poisson {
                        component,
                        sources,
                        solve,
                    }
                }
            }
            SolveStage::Poisson {
                component,
                sources,
                mut solve,
            } => {
                let c = &solver.components[component as usize];
                if solve.step(&c.amg)? {
                    spent = c.vertices.len();
                    self.write(c, &sources, solve.finish());
                    SolveStage::Next
                } else {
                    spent = 8 * c.vertices.len();
                    SolveStage::Poisson {
                        component,
                        sources,
                        solve,
                    }
                }
            }
            SolveStage::Done => return Ok(None),
        };
        Ok(Some(spent.max(1)))
    }

    /// Writes a component's distances, from its potential.
    fn write(&mut self, c: &Component, sources: &[u32], mut phi: Vec<f64>) {
        let nearest = sources
            .iter()
            .map(|&s| phi[s as usize])
            .fold(f64::INFINITY, f64::min);
        if nearest.is_finite() {
            for v in &mut phi {
                *v = (*v - nearest).max(0.0);
            }
        }
        for &s in sources {
            phi[s as usize] = 0.0;
        }
        for (&slot, &value) in c.vertices.iter().zip(&phi) {
            if slot == NONE {
                continue;
            }
            let d = value as f32;
            self.result[slot as usize] = match self.max_distance {
                Some(limit) if d > limit => f32::INFINITY,
                _ => d,
            };
        }
    }

    pub(super) fn finish(self) -> Vec<f32> {
        self.result
    }
}

/// Adds to `b` what the unit field away from the heat flows out of some of a
/// component's triangles, the flat ones numbered after the others.
fn add_divergence(c: &Component, heat: &Heat, range: Range<usize>, b: &mut [f64]) -> Result<()> {
    let own = c.triangles.len();
    let plain = &c.triangles[range.start.min(own)..range.end.min(own)];
    let flat = &c.flat[range.start.max(own) - own..range.end.max(own) - own];
    allocation_allowed()?;
    let parts = par::map_chunks(plain, CHUNK, 2 * CHUNK, OOM, |_, chunk| {
        let mut out: Vec<[f64; 3]> = Vec::new();
        reserve_exact(&mut out, chunk.len())?;
        for tri in chunk {
            let v = tri.map(|v| v as usize);
            out.push(divergence(v.map(|v| c.positions[v]), heat.corners(v)));
        }
        Ok(out)
    })?;
    let mut add = |tri: &[u32; 3], flux: &[f64; 3]| {
        for k in 0..3 {
            b[tri[k] as usize] += flux[k];
        }
    };
    for (tri, flux) in plain.iter().zip(parts.iter().flatten()) {
        add(tri, flux);
    }
    allocation_allowed()?;
    let parts = par::map_chunks(flat, CHUNK, 2 * CHUNK, OOM, |_, chunk| {
        let mut out: Vec<[f64; 3]> = Vec::new();
        reserve_exact(&mut out, chunk.len())?;
        for f in chunk {
            let u = heat.corners(f.corners.map(|v| v as usize));
            out.push(flat_divergence(f.layout(), u));
        }
        Ok(out)
    })?;
    for (f, flux) in flat.iter().zip(parts.iter().flatten()) {
        add(&f.corners, flux);
    }
    Ok(())
}

/// The Poisson problem of the divergence `b` of a component's heat: the
/// potential whose gradient is nearest to the unit field away from the heat.
fn begin_poisson(c: &Component, heat: Heat, mut b: Vec<f64>, time_step: f64) -> Result<Solve> {
    let n = c.vertices.len();
    let mean = sum_range(n, PAR_MIN_ROWS, |i| b[i])? / n as f64;
    for v in &mut b {
        *v -= mean;
    }

    // Varadhan's formula, for a first guess: -√t ln(heat) is near the distance.
    let mut guess = filled(n, 0.0f64)?;
    let length = -time_step.sqrt();
    par::for_each_chunk_mut(&mut guess, CHUNK, PAR_MIN_ROWS, |k, part| {
        for (i, g) in part.iter_mut().enumerate() {
            let d = length * heat.ln(k * CHUNK + i);
            *g = if d.is_finite() { d } else { 0.0 };
        }
    });
    drop(heat);
    Solve::new(&c.amg, guess, &b, TOLERANCE * TOLERANCE * c.area)
}

/// What the unit field away from the heat `u` flows out of a triangle around
/// each corner.
fn divergence(corners: [[f32; 3]; 3], u: [f64; 3]) -> [f64; 3] {
    let [a, b, c] = corners.map(widen);
    let normal = triangle_normal(corners);
    let area2 = norm(normal);
    let inv = 1.0 / area2;
    let n = scale(normal, inv);
    let grad = [sub(c, b), sub(a, c), sub(b, a)].map(|e| scale(cross(n, e), inv));
    let g = add(
        add(scale(grad[0], u[0]), scale(grad[1], u[1])),
        scale(grad[2], u[2]),
    );
    let len = norm(g);
    if !(len.is_finite() && len > 0.0) {
        return [0.0; 3];
    }
    let x = scale(g, -0.5 * area2 / len);
    grad.map(|d| dot(d, x))
}

/// [`divergence`] of a triangle laid out flat.
fn flat_divergence(layout: Layout, u: [f64; 3]) -> [f64; 3] {
    let Layout { length, x, y } = layout;
    let gx = (u[1] - u[0]) / length;
    let gy = (u[2] - u[0] - gx * x) / y;
    let len = hypot(gx, gy);
    if !(len.is_finite() && len > 0.0) {
        return [0.0; 3];
    }
    let (gx, gy) = (-0.5 * gx / len, -0.5 * gy / len);
    let second = y * gx - x * gy;
    let third = length * gy;
    [-second - third, second, third]
}

#[cfg(test)]
mod tests {
    use super::super::intrinsic::Check;
    use super::*;
    use crate::grid::fail_allocations_after;
    use crate::test_util::{icosphere, mesh, terrain};

    /// A capped cylinder along Z whose caps are fans around one vertex.
    fn fan_cylinder(segments: u32, rings: u32, height: f32) -> Mesh {
        let mut p = Vec::new();
        let mut t = Vec::new();
        for k in 0..=rings {
            for s in 0..segments {
                let a = f64::from(s) / f64::from(segments) * std::f64::consts::TAU;
                let z = height * (k as f32 / rings as f32);
                p.push([a.cos() as f32, a.sin() as f32, z]);
            }
        }
        for k in 0..rings {
            for s in 0..segments {
                let (a, b) = (k * segments + s, k * segments + (s + 1) % segments);
                let (c, d) = (a + segments, b + segments);
                t.push([a, b, d]);
                t.push([a, d, c]);
            }
        }
        let (low, high) = (p.len() as u32, p.len() as u32 + 1);
        p.push([0.0, 0.0, 0.0]);
        p.push([0.0, 0.0, height]);
        let top = rings * segments;
        for s in 0..segments {
            let next = (s + 1) % segments;
            t.push([low, next, s]);
            t.push([high, top + s, top + next]);
        }
        mesh(&p, &t)
    }

    /// A flat grid of `nx` by `ny` unit cells, `x` scaled.
    fn strip(nx: u32, ny: u32, x: f32) -> Mesh {
        let mut p = Vec::new();
        let mut t = Vec::new();
        for j in 0..=ny {
            for i in 0..=nx {
                p.push([i as f32 * x, j as f32, 0.0]);
            }
        }
        for j in 0..ny {
            for i in 0..nx {
                let a = j * (nx + 1) + i;
                t.push([a, a + 1, a + nx + 2]);
                t.push([a, a + nx + 2, a + nx + 1]);
            }
        }
        mesh(&p, &t)
    }

    /// `m` drawn out along Y.
    fn stretched(m: &Mesh, by: f32) -> Mesh {
        let p: Vec<_> = m
            .positions()
            .iter()
            .map(|v| [v[0], v[1] * by, v[2]])
            .collect();
        mesh(&p, m.triangles())
    }

    /// The solver of a mesh's own triangles, whatever they are like.
    fn as_it_is(m: &Mesh, time_step_scale: f64) -> GeodesicHeatSolver {
        let mut build = SolverBuild::new(m, time_step_scale).unwrap();
        build.recut = false;
        while build.step(m).unwrap().is_some() {}
        build.finish()
    }

    /// The heat of a component by a dense Cholesky solve, which is accurate
    /// at every vertex: all its sums are of positive terms.
    fn dense_heat(c: &Component, time_step: f64, sources: &[u32]) -> Vec<f64> {
        let n = c.vertices.len();
        let a = c.amg.fine();
        // The lower triangle, by columns: `l[j * n + i]`.
        let mut l = vec![0.0f64; n * n];
        for i in 0..n {
            l[i * n + i] = c.mass[i] + time_step * a.diag[i];
            for (j, v) in a.off.row(i) {
                l[j as usize * n + i] = time_step * v;
            }
        }
        for j in 0..n {
            let (before, rest) = l.split_at_mut(j * n);
            let column = &mut rest[..n];
            for done in before.chunks(n) {
                let f = done[j];
                if f != 0.0 {
                    for i in j..n {
                        column[i] -= done[i] * f;
                    }
                }
            }
            let d = column[j].sqrt();
            for v in &mut column[j..] {
                *v /= d;
            }
        }
        let mut x = vec![0.0; n];
        for &s in sources {
            x[s as usize] = 1.0;
        }
        for j in 0..n {
            x[j] /= l[j * n + j];
            for i in j + 1..n {
                x[i] -= l[j * n + i] * x[j];
            }
        }
        for j in (0..n).rev() {
            for i in j + 1..n {
                x[j] -= l[j * n + i] * x[i];
            }
            x[j] /= l[j * n + j];
        }
        x
    }

    /// The heat of a component, and how many passes it took.
    fn diffuse(solver: &GeodesicHeatSolver, component: usize, sources: &[u32]) -> (Heat, usize) {
        let c = &solver.components[component];
        let surface = Surface {
            matrix: c.amg.fine(),
            mass: &c.mass,
            positions: &c.positions,
            time_step: solver.time_step_scale * solver.length * solver.length,
        };
        let mut d = Diffusion::new(c.vertices.len(), sources).unwrap();
        while d.step(&surface, 100).unwrap().is_some() {}
        let passes = d.passes();
        (d.finish(), passes)
    }

    #[test]
    fn heat_is_the_solution_at_every_vertex() {
        // (Heat crosses the stretched triangles much faster than their size
        // suggests: it takes more than one pass.)
        for (name, m, scale) in [
            ("terrain", terrain(18), 1.0),
            ("small step", terrain(14), 0.2),
            ("large step", terrain(14), 40.0),
            ("sphere", icosphere(), 1.0),
            ("fans", fan_cylinder(40, 8, 1.0), 1.0),
            ("wide fans", fan_cylinder(80, 4, 0.3), 0.02),
            ("stretched", stretched(&icosphere(), 20.0), 0.1),
            ("stretched further", stretched(&icosphere(), 100.0), 1e-3),
            ("flat cells", strip(60, 6, 0.1), 1.0),
            ("long strip", strip(150, 1, 1.0), 1.0),
        ] {
            let solver = as_it_is(&m, scale);
            let t = scale * solver.length * solver.length;
            let n = solver.components[0].vertices.len() as u32;
            for sources in [vec![0], vec![n / 2], vec![0, n / 3, n - 1]] {
                let (heat, passes) = diffuse(&solver, 0, &sources);
                assert_eq!(
                    passes > 1,
                    name.starts_with("stretched"),
                    "{name}: {passes}"
                );
                let exact = dense_heat(&solver.components[0], t, &sources);
                let mut far = 0.0f64;
                for (v, want) in exact.iter().enumerate() {
                    let got = heat.ln(v);
                    assert!(
                        (got - want.ln()).abs() < 5e-3,
                        "{name} {sources:?} vertex {v}: e^{got} instead of e^{}",
                        want.ln()
                    );
                    far = far.max(-want.ln());
                }
                if name == "long strip" && sources == [0] {
                    assert!(far > 100.0, "heat reaches e^-{far} only");
                }
            }
        }
    }

    #[test]
    fn multigrid_takes_about_as_many_iterations_on_any_mesh() {
        for (name, m) in [
            ("small", terrain(40)),
            ("medium", terrain(90)),
            ("large", terrain(200)),
            ("flat cells", strip(160, 160, 0.05)),
            ("fans", fan_cylinder(128, 128, 2.0)),
        ] {
            let solver = GeodesicHeatSolver::build(&m, 1.0).unwrap();
            let mut solve = HeatSolve::new(&solver, &[0], None, 1.0).unwrap();
            let mut iterations = 0;
            loop {
                iterations += usize::from(solve.phase() == GeodesicPhase::Poisson);
                if solve.step(&solver, usize::MAX).unwrap().is_none() {
                    break;
                }
            }
            assert!(
                (2..30).contains(&iterations),
                "{name}: {iterations} iterations"
            );
        }
    }

    #[test]
    fn operators_are_what_they_should_be() {
        // Two parts, numbered across each other, with a dead triangle.
        let (a, b) = (terrain(20), icosphere());
        let mut p = Vec::new();
        let mut t = Vec::new();
        let na = a.vertex_slots() as u32;
        for &v in a.positions() {
            p.push(v);
            p.push([v[0], v[1], v[2] + 50.0]); // unused
        }
        for tri in a.triangles() {
            t.push(tri.map(|v| 2 * v));
        }
        for &v in b.positions() {
            p.push([v[0] + 100.0, v[1], v[2]]);
        }
        for tri in b.triangles() {
            t.push(tri.map(|v| 2 * na + v));
        }
        t.push([0, 0, 2]); // degenerate
        let m = mesh(&p, &t);
        let solver = as_it_is(&m, 1.0);
        assert_eq!(solver.components.len(), 2);
        let mut seen = 0;
        for (k, c) in solver.components.iter().enumerate() {
            let n = c.vertices.len();
            seen += n;
            let total: f64 = c.mass.iter().sum();
            assert!((total - c.area).abs() < 1e-9 * c.area);
            for (local, &v) in c.vertices.iter().enumerate() {
                assert_eq!(solver.component_of[v as usize], k as u32);
                assert_eq!(solver.local_of[v as usize], local as u32);
                assert_eq!(c.positions[local], m.positions()[v as usize]);
            }
            let l = c.amg.fine();
            for i in 0..n {
                let row: Vec<_> = l.off.row(i).collect();
                assert!(row.windows(2).all(|w| w[0].0 < w[1].0));
                let sum: f64 = row.iter().map(|e| e.1).sum();
                assert!((sum + l.diag[i]).abs() < 1e-12 * l.diag[i]);
                for (j, v) in row {
                    assert!(v < 0.0);
                    // Symmetric to the bit.
                    let back = l.off.row(j as usize).find(|e| e.0 == i as u32);
                    assert_eq!(back.map(|e| e.1.to_bits()), Some(v.to_bits()));
                }
            }
            for tri in &c.triangles {
                assert!(tri.iter().all(|&v| (v as usize) < n));
            }
        }
        assert_eq!(seen, a.vertex_slots() + b.vertex_slots());
        assert_eq!(
            solver
                .components
                .iter()
                .map(|c| c.triangles.len())
                .sum::<usize>(),
            a.triangle_count() + b.triangle_count()
        );
        assert!(solver.byte_size() > 0);
    }

    /// The triangulation the operators of `m` are assembled on, if it is not
    /// the mesh's own, checked on the way.
    fn recut(m: &Mesh) -> Option<(Check, f64)> {
        let mut build = SolverBuild::new(m, 1.0).unwrap();
        while build.step(m).unwrap().is_some() {
            match &mut build.stage {
                BuildStage::Flip { intrinsic, .. } => {
                    intrinsic.check();
                }
                BuildStage::Refine(intrinsic) => {
                    assert_eq!(intrinsic.check().negative, 0);
                    while !intrinsic.refine().unwrap() {
                        intrinsic.check();
                    }
                    return Some((intrinsic.check(), build.solver.length));
                }
                _ => {}
            }
        }
        None
    }

    fn area_of(m: &Mesh) -> f64 {
        m.triangles()
            .iter()
            .map(|t| 0.5 * norm(triangle_normal(t.map(|v| m.positions()[v as usize]))))
            .sum()
    }

    /// Pages that share one edge, each a stretched strip.
    fn book(pages: u32) -> Mesh {
        let mut p: Vec<[f32; 3]> = (0..=8).map(|k| [0.0, 0.0, k as f32 * 4.0]).collect();
        let mut t = Vec::new();
        for page in 0..pages {
            let a = page as f32 / pages as f32 * std::f32::consts::TAU;
            let base = p.len() as u32 - 9;
            let at = |x: u32, z: u32| if x == 0 { z } else { base + 9 * x + z };
            for x in 1..=6 {
                for z in 0..=8 {
                    let r = x as f32 * 0.3;
                    let z = z as f32 * 4.0 + r * 5.0;
                    p.push([a.cos() * r, a.sin() * r, z]);
                }
            }
            for x in 0..6 {
                for z in 0..8 {
                    let (a, b, c, d) = (at(x, z), at(x + 1, z), at(x + 1, z + 1), at(x, z + 1));
                    t.push([a, b, c]);
                    t.push([a, c, d]);
                }
            }
        }
        mesh(&p, &t)
    }

    #[test]
    fn a_good_triangulation_is_left_alone() {
        let m = strip(60, 60, 1.0);
        let p = m.positions().iter();
        let p: Vec<_> = p.map(|v| [v[0] - 0.4 * v[1], v[1], v[2]]).collect();
        let slanted = mesh(&p, m.triangles());
        assert!(recut(&slanted).is_none());
        // Rectangles are cut either way: rounding does not make a difference.
        for x in [1.0, 0.7, 3.3] {
            for shift in [0.0, 1000.0] {
                let m = strip(90, 90, x);
                let p: Vec<_> = m
                    .positions()
                    .iter()
                    .map(|v| [v[0] * 0.1 + shift, v[1] * 0.1 - shift, v[2]])
                    .collect();
                assert!(recut(&mesh(&p, m.triangles())).is_none(), "{x} {shift}");
            }
        }
        // The same operators, to the bit, as without looking.
        let a = GeodesicHeatSolver::build(&slanted, 1.0).unwrap();
        let b = as_it_is(&slanted, 1.0);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let (a, b) = (&a.components[0].amg.fine(), &b.components[0].amg.fine());
        assert_eq!(bits(&a.off.val), bits(&b.off.val));
        assert_eq!(bits(&a.diag), bits(&b.diag));
    }

    #[test]
    fn a_bad_triangulation_is_cut_again() {
        // Across each other, in one mesh.
        let two = |a: &Mesh, b: &Mesh| {
            let mut p = a.positions().to_vec();
            let mut t = a.triangles().to_vec();
            let base = p.len() as u32;
            p.extend(b.positions().iter().map(|v| [v[0] + 40.0, v[1], v[2]]));
            t.extend(b.triangles().iter().map(|tri| tri.map(|v| v + base)));
            mesh(&p, &t)
        };
        // Every other triangle the other way around.
        let mixed = |m: &Mesh| {
            let t = m.triangles().iter().enumerate();
            let t: Vec<_> = t
                .map(|(k, &[a, b, c])| if k % 2 == 0 { [a, b, c] } else { [a, c, b] })
                .collect();
            mesh(m.positions(), &t)
        };
        let sheared = |m: &Mesh| {
            let p = m.positions().iter();
            let p: Vec<_> = p.map(|v| [v[0] + 6.0 * v[1], v[1], v[2]]).collect();
            mesh(&p, m.triangles())
        };
        for (name, m, kept, grows) in [
            ("stretched", stretched(&icosphere(), 20.0), false, true),
            (
                "stretched further",
                stretched(&icosphere(), 500.0),
                false,
                true,
            ),
            ("sheared", sheared(&strip(12, 12, 1.0)), true, true),
            ("cylinder", fan_cylinder(64, 8, 2.0), false, true),
            ("coarse cylinder", fan_cylinder(9, 1, 2.0), false, true),
            ("cells", stretched(&strip(30, 30, 1.0), 9.0), true, true),
            ("pages", book(3), true, true),
            ("two pages", book(2), true, true),
            ("many pages", book(11), true, true),
            ("mixed", mixed(&stretched(&icosphere(), 5.0)), false, true),
            (
                "two parts",
                two(&stretched(&icosphere(), 3.0), &fan_cylinder(20, 3, 1.0)),
                false,
                true,
            ),
        ] {
            let (check, length) = recut(&m).expect(name);
            let area = area_of(&m);
            assert!(
                (check.area - area).abs() < 1e-5 * area,
                "{name}: {} of {area}",
                check.area
            );
            assert_eq!(check.kept > 0, kept, "{name}: {check:?}");
            assert_eq!(check.vertices > m.vertex_count(), grows, "{name}");
            assert!(check.flat > 0, "{name}");
            if !check.full {
                assert_eq!(check.negative, 0, "{name}: {check:?}");
                assert!(
                    check.radius < 1.2 * length || kept,
                    "{name}: {check:?}, {length}"
                );
            }
            // The operators on it.
            let solver = GeodesicHeatSolver::build(&m, 1.0).unwrap();
            let mut total = 0.0;
            for c in &solver.components {
                total += c.area;
                let mass: f64 = c.mass.iter().sum();
                assert!((mass - c.area).abs() < 1e-9 * c.area, "{name}");
                let l = c.amg.fine();
                for i in 0..c.vertices.len() {
                    for (j, v) in l.off.row(i) {
                        assert!(v < 0.0 && j as usize != i, "{name}");
                        let back = l.off.row(j as usize).find(|e| e.0 == i as u32);
                        assert_eq!(back.map(|e| e.1.to_bits()), Some(v.to_bits()), "{name}");
                    }
                }
                let known = c.vertices.iter().enumerate().filter(|(_, v)| **v != NONE);
                for (local, &v) in known {
                    assert_eq!(solver.local_of[v as usize], local as u32, "{name}");
                    assert_eq!(c.positions[local], m.positions()[v as usize], "{name}");
                }
            }
            assert!((total - area).abs() < 1e-5 * area, "{name}");
            let known: usize = solver
                .components
                .iter()
                .map(|c| c.vertices.iter().filter(|&&v| v != NONE).count())
                .sum();
            assert_eq!(known, m.vertex_count(), "{name}");
        }
    }

    #[test]
    fn a_part_grows_only_so_much() {
        // A large plane among many small spheres: all it lacks is in the plane.
        let s = icosphere();
        let mut p: Vec<[f32; 3]> = vec![
            [-1e4, -1e4, -9.0],
            [1e4, -1e4, -9.0],
            [1e4, 1e4, -9.0],
            [-1e4, 1e4, -9.0],
        ];
        let mut t = vec![[0, 1, 2], [0, 2, 3]];
        for k in 0..30 {
            let base = p.len() as u32;
            let x = k as f32 * 3.0;
            p.extend(s.positions().iter().map(|v| [v[0] + x, v[1], v[2]]));
            t.extend(s.triangles().iter().map(|tri| tri.map(|v| v + base)));
        }
        let m = mesh(&p, &t);
        let (check, _) = recut(&m).unwrap();
        assert!(check.full);
        let most = 4.0 * RESOLUTION + 1.0;
        assert!(check.vertices - m.vertex_count() < most as usize);
    }

    #[test]
    fn running_out_of_memory_while_cutting_is_an_error() {
        let parts = [
            stretched(&icosphere(), 6.0),
            strip(9, 4, 8.0),
            fan_cylinder(24, 2, 1.0),
            book(3),
        ];
        for m in &parts {
            let mut at = 0;
            loop {
                let mut build = SolverBuild::new(m, 1.0).unwrap();
                while !matches!(build.stage, BuildStage::Survey(_)) {
                    build.step(m).unwrap();
                }
                fail_allocations_after(Some(at));
                let done = loop {
                    if let Err(e) = build.step(m) {
                        break Err(e);
                    }
                    if matches!(build.stage, BuildStage::Assemble { survey: None, .. }) {
                        break Ok(());
                    }
                };
                fail_allocations_after(None);
                match done {
                    Ok(()) => break,
                    Err(e) => assert_eq!(e, FieldsError::OutOfMemory),
                }
                at += 1;
            }
            assert!(at > 40, "only {at} allocations");
        }
    }

    #[test]
    fn running_out_of_memory_anywhere_is_an_error() {
        // Large enough for multigrid levels, and in two parts.
        let (a, b) = (terrain(15), icosphere());
        let mut p = a.positions().to_vec();
        let mut t = a.triangles().to_vec();
        let base = p.len() as u32;
        p.extend(b.positions().iter().map(|v| [v[0] + 100.0, v[1], v[2]]));
        t.extend(b.triangles().iter().map(|tri| tri.map(|v| v + base)));
        let m = mesh(&p, &t);
        // (Not cut again, which has a test of its own: it would be much larger.)
        let build = || {
            let mut build = SolverBuild::new(&m, 1.0)?;
            build.recut = false;
            while build.step(&m)?.is_some() {}
            Ok::<_, FieldsError>(build.finish())
        };
        let mut want = Vec::new();
        build().unwrap().solve(&[0, base], None, &mut want).unwrap();

        let mut at = 0;
        loop {
            let mut out = vec![7.5f32];
            fail_allocations_after(Some(at));
            let built = build();
            let solved = match &built {
                Ok(s) => s.solve(&[0, base], None, &mut out),
                Err(e) => Err(e.clone()),
            };
            fail_allocations_after(None);
            match solved {
                Ok(()) => {
                    assert_eq!(out, want);
                    break;
                }
                Err(e) => {
                    assert_eq!(e, FieldsError::OutOfMemory);
                    assert_eq!(out, [7.5]);
                }
            }
            at += 1;
        }
        assert!(at > 100, "only {at} allocations");
    }
}
