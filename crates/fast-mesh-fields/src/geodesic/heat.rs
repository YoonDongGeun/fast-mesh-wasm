//! The heat method's operators and solves; see [`GeodesicHeatSolver`].

use std::ops::Range;

use fast_mesh_store::{Mesh, par};

use super::amg::{Amg, AmgBuild, Matrix, PAR_MIN_ROWS, Rows, Solve, sum_range};
use super::diffuse::{Diffusion, Surface};
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
/// Elements per chunk of the assembly's loops.
const CHUNK: usize = 8192;
/// Triangles per step of the divergence.
const DIVERGENCE_STEP: usize = 1 << 16;
/// No component.
const NONE: u32 = u32::MAX;

const OOM: fn() -> FieldsError = || FieldsError::OutOfMemory;

/// One connected part of the mesh, in its own vertex numbering, which
/// follows a space-filling curve: neighbours on the surface are mostly
/// neighbours in memory.
#[derive(Debug, Clone)]
struct Component {
    /// The mesh's vertex per vertex.
    vertices: Vec<u32>,
    positions: Vec<[f32; 3]>,
    /// The triangles with an area, ordered by their first vertex.
    triangles: Vec<[u32; 3]>,
    /// Lumped mass: a third of the area around each vertex.
    mass: Vec<f64>,
    area: f64,
    /// The clamped cotangent Laplacian and its multigrid levels.
    amg: Amg,
}

impl Component {
    fn byte_size(&self) -> usize {
        self.vertices.capacity() * size_of::<u32>()
            + (self.positions.capacity() + self.triangles.capacity()) * size_of::<[u32; 3]>()
            + self.mass.capacity() * size_of::<f64>()
            + self.amg.byte_size()
    }
}

/// Precomputed solver for Crane et al. (2013) *Geodesics in Heat* on a fixed
/// mesh and diffusion time step.
///
/// Building the solver assembles the lumped mass matrix, the clamped cotangent
/// Laplacian and its multigrid levels per connected component, so subsequent
/// [`GeodesicHeatSolver::solve`] calls for different source sets avoid
/// rebuilding the operators.
#[derive(Debug, Clone)]
pub struct GeodesicHeatSolver {
    mesh: MeshState,
    source_valid: Vec<bool>,
    component_of: Vec<u32>,
    local_of: Vec<u32>,
    components: Vec<Component>,
    mean_edge: f64,
    time_step_scale: f64,
}

impl GeodesicHeatSolver {
    /// Builds a reusable heat-method solver for `mesh` using diffusion time step
    /// `t = time_step_scale * h_mean^2`, where `h_mean` is the mean edge length
    /// across unique edges of live non-degenerate triangles.
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
    /// Assemble the Laplacian of every part.
    Assemble(Numbered),
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
struct Numbered {
    vertices: Vec<u32>,
    starts: Vec<u32>,
    positions: Vec<[f32; 3]>,
    /// Ordered by their smallest vertex.
    triangles: Vec<[u32; 3]>,
}

/// A connected part before its multigrid levels are built.
#[derive(Debug)]
struct Part {
    vertices: Vec<u32>,
    positions: Vec<[f32; 3]>,
    triangles: Vec<[u32; 3]>,
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
                mean_edge: 0.0,
                time_step_scale,
            },
            stage: BuildStage::Triangles,
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
                let triangles = renumber_triangles(triangles, vertices.len(), renumber)?;
                let mut positions: Vec<[f32; 3]> = Vec::new();
                reserve_exact(&mut positions, vertices.len())?;
                positions.extend(vertices.iter().map(|&v| mesh.positions()[v as usize]));
                BuildStage::Assemble(Numbered {
                    vertices,
                    starts,
                    positions,
                    triangles,
                })
            }
            BuildStage::Assemble(numbered) => {
                work = 4 * numbered.triangles.len();
                let (laplacian, mean_edge) = assemble(&numbered, &s.component_of)?;
                s.mean_edge = mean_edge;
                BuildStage::Split {
                    numbered,
                    laplacian,
                }
            }
            BuildStage::Split {
                numbered,
                laplacian,
            } => {
                work = numbered.triangles.len();
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
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
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
        for k in 0..3 {
            lo[k] = lo[k].min(positions[v][k]);
            hi[k] = hi[k].max(positions[v][k]);
        }
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

/// A point's place along a Morton curve through the box, 21 bits per axis.
fn morton(p: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> u64 {
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

/// The triangles with their corners renumbered, ordered by their smallest
/// corner, then as before.
fn renumber_triangles(
    mut triangles: Vec<[u32; 3]>,
    vertices: usize,
    renumber: impl Fn(u32) -> u32,
) -> Result<Vec<[u32; 3]>> {
    for tri in &mut triangles {
        *tri = tri.map(&renumber);
    }
    let first = |t: &[u32; 3]| t[0].min(t[1]).min(t[2]) as usize;
    let mut at = filled(vertices + 1, 0u32)?;
    for t in &triangles {
        at[first(t) + 1] += 1;
    }
    for v in 0..vertices {
        at[v + 1] += at[v];
    }
    let mut out = filled(triangles.len(), [0u32; 3])?;
    for t in &triangles {
        let slot = &mut at[first(t)];
        out[*slot as usize] = *t;
        *slot += 1;
    }
    Ok(out)
}

/// Half the cotangent at each corner of a triangle, and its area.
fn half_cotangents(corners: [[f32; 3]; 3]) -> ([f64; 3], f64) {
    let [a, b, c] = corners.map(widen);
    let area2 = norm(triangle_normal(corners));
    let inv = 1.0 / area2;
    (
        [
            0.5 * dot(sub(b, a), sub(c, a)) * inv,
            0.5 * dot(sub(c, b), sub(a, b)) * inv,
            0.5 * dot(sub(a, c), sub(b, c)) * inv,
        ],
        0.5 * area2,
    )
}

/// The triangles around each vertex: where each vertex's start, and the
/// triangles, ascending.
fn incidence(vertices: usize, triangles: &[[u32; 3]]) -> Result<(Vec<u32>, Vec<u32>)> {
    let mut start = filled(vertices + 1, 0u32)?;
    for &v in triangles.iter().flatten() {
        start[v as usize + 1] += 1;
    }
    for v in 0..vertices {
        start[v + 1] += start[v];
    }
    let mut at = filled(vertices, 0u32)?;
    at.copy_from_slice(&start[..vertices]);
    let mut around = filled(3 * triangles.len(), 0u32)?;
    for (t, tri) in triangles.iter().enumerate() {
        for &v in tri {
            let slot = &mut at[v as usize];
            around[*slot as usize] = t as u32;
            *slot += 1;
        }
    }
    Ok((start, around))
}

/// Off-diagonal Laplacian rows of the vertices of a range, columns local to
/// each vertex's part, with the lengths of their edges to later vertices.
#[derive(Default)]
struct RowsPart {
    rows: Rows,
    length: f64,
    edges: usize,
}

/// The off-diagonal entries of the clamped cotangent Laplacian, columns
/// numbered within each part, and the mean edge length.
fn assemble(mesh: &Numbered, component_of: &[u32]) -> Result<(Rows, f64)> {
    let n = mesh.vertices.len();
    let triangles = &mesh.triangles;
    let (start, around) = incidence(n, triangles)?;
    let mut cots = filled(triangles.len(), [0.0f64; 3])?;
    par::for_each_chunk_mut(&mut cots, CHUNK, 2 * CHUNK, |c, part| {
        for (out, tri) in part.iter_mut().zip(&triangles[c * CHUNK..]) {
            *out = half_cotangents(tri.map(|v| mesh.positions[v as usize])).0;
        }
    });
    allocation_allowed()?;
    let parts = par::map_range(n, CHUNK, 2 * CHUNK, OOM, |range| {
        let mut part = RowsPart::default();
        reserve_exact(&mut part.rows.ptr, range.len())?;
        // Column, triangle, weight: the triangle keeps the order fixed.
        let mut entries: Vec<(u32, u32, f64)> = Vec::new();
        for v in range {
            entries.clear();
            let incident = &around[start[v] as usize..start[v + 1] as usize];
            for &t in incident {
                let (tri, cot) = (triangles[t as usize], cots[t as usize]);
                let k = tri.iter().position(|&c| c as usize == v).unwrap_or(0);
                let (a, b) = ((k + 1) % 3, (k + 2) % 3);
                push(&mut entries, (tri[a], t, cot[b]))?;
                push(&mut entries, (tri[b], t, cot[a]))?;
            }
            entries.sort_unstable_by_key(|e| (e.0, e.1));
            let base = mesh.starts[component_of[mesh.vertices[v] as usize] as usize];
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
        let (mass, area) = masses(&mesh.positions, &mesh.triangles)?;
        parts.push(Part {
            vertices: mesh.vertices,
            positions: mesh.positions,
            triangles: mesh.triangles,
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
    let mut t = 0;
    for k in 0..count {
        let range = mesh.starts[k] as usize..mesh.starts[k + 1] as usize;
        let from = t;
        while t < mesh.triangles.len() && (mesh.triangles[t][0] as usize) < range.end {
            t += 1;
        }
        let mut triangles = copied(&mesh.triangles[from..t])?;
        for v in triangles.iter_mut().flatten() {
            *v -= range.start as u32;
        }
        let positions = copied(&mesh.positions[range.clone()])?;
        let (mass, area) = masses(&positions, &triangles)?;
        let entries = laplacian.ptr[range.start] as usize..laplacian.ptr[range.end] as usize;
        let mut ptr = copied(&laplacian.ptr[range.start..=range.end])?;
        for p in &mut ptr {
            *p -= entries.start as u32;
        }
        parts.push(Part {
            vertices: copied(&mesh.vertices[range])?,
            positions,
            triangles,
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
fn masses(positions: &[[f32; 3]], triangles: &[[u32; 3]]) -> Result<(Vec<f64>, f64)> {
    let mut mass = filled(positions.len(), 0.0f64)?;
    let mut total = 0.0;
    for tri in triangles {
        let area = 0.5 * norm(triangle_normal(tri.map(|v| positions[v as usize])));
        total += area;
        for &v in tri {
            mass[v as usize] += area / 3.0;
        }
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
        let time_step = self.time_step_scale * solver.mean_edge * solver.mean_edge;
        let spent;
        self.stage = match std::mem::replace(&mut self.stage, SolveStage::Done) {
            SolveStage::Next => match self.todo.pop() {
                None => return Ok(None),
                Some((component, sources)) => {
                    let n = solver.components[component as usize].vertices.len();
                    spent = n;
                    let mut own = Vec::new();
                    reserve_exact(&mut own, sources.len())?;
                    own.extend_from_slice(&sources);
                    SolveStage::Diffuse {
                        component,
                        sources,
                        diffusion: Box::new(Diffusion::new(n, own)?),
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
                        spent = 4 * c.vertices.len();
                        let solve = Box::new(begin_poisson(c, *diffusion, time_step)?);
                        SolveStage::Poisson {
                            component,
                            sources,
                            solve,
                        }
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

/// The Poisson problem of a component's heat: the potential whose gradient
/// is nearest to the unit field that points away from the heat.
fn begin_poisson(c: &Component, diffusion: Diffusion, time_step: f64) -> Result<Solve> {
    let n = c.vertices.len();
    let heat = diffusion.finish();
    let mut b = filled(n, 0.0f64)?;
    let mut at = 0;
    while at < c.triangles.len() {
        let end = c.triangles.len().min(at + DIVERGENCE_STEP);
        allocation_allowed()?;
        let parts = par::map_range(end - at, CHUNK, 2 * CHUNK, OOM, |r: Range<usize>| {
            let mut out: Vec<[f64; 3]> = Vec::new();
            reserve_exact(&mut out, r.len())?;
            for tri in &c.triangles[at + r.start..at + r.end] {
                let v = tri.map(|v| v as usize);
                out.push(divergence(v.map(|v| c.positions[v]), heat.corners(v)));
            }
            Ok(out)
        })?;
        for (tri, flux) in c.triangles[at..end].iter().zip(parts.iter().flatten()) {
            for k in 0..3 {
                b[tri[k] as usize] += flux[k];
            }
        }
        at = end;
    }
    let mean = sum_range(n, |i| b[i])? / n as f64;
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
