//! Intrinsic geodesic distance fields on triangle meshes.
//!
//! Supports two complementary methods:
//! - [`GeodesicMethod::Dijkstra`]: exact graph shortest paths over triangle
//!   edges with Euclidean edge weights.
//! - [`GeodesicMethod::HeatMethod`]: Crane, Weischedel & Wardetzky (2013),
//!   *Geodesics in Heat*, solving a backward-Euler heat diffusion step on the
//!   clamped cotangent Laplacian followed by unit-gradient normalization and a
//!   Poisson distance reconstruction.
//!
//! For repeated queries on the same mesh with varying source sets, build
//! [`GeodesicHeatSolver`] once and call [`GeodesicHeatSolver::solve`]. To
//! yield (or stop) within a query, run a [`GeodesicQuery`] in steps.

mod amg;
mod diffuse;
mod heat;

use std::collections::BinaryHeap;
use std::sync::Arc;

use fast_mesh_store::Mesh;

pub use heat::GeodesicHeatSolver;

use crate::geom::{norm, sub, widen};
use crate::grid::{filled, push, reserve_exact};
use crate::{FieldsError, Result};
use diffuse::{HeapItem, heap_push};
use heat::{HeatSolve, SolverBuild};

/// Algorithm used by [`geodesic_distances`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeodesicMethod {
    /// Multi-source Dijkstra along mesh edges weighted by Euclidean edge length.
    Dijkstra,
    /// Crane et al. (2013) *Geodesics in Heat* via the clamped cotangent
    /// Laplacian: the diffusion solved outwards from the sources in bands,
    /// the distances by conjugate gradients with a multigrid preconditioner.
    HeatMethod,
}

/// Configuration for [`geodesic_distances`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodesicOptions {
    /// Distance computation method. Default: [`GeodesicMethod::HeatMethod`].
    pub method: GeodesicMethod,
    /// Optional distance cutoff; vertices farther than this bound (or on
    /// unreachable components) receive `f32::INFINITY`.
    pub max_distance: Option<f32>,
    /// Multiplier `m` for the heat diffusion time step `t = m * h_mean^2`.
    /// Must be finite and strictly positive. Default: `1.0`.
    pub time_step_scale: f64,
}

impl Default for GeodesicOptions {
    fn default() -> Self {
        Self {
            method: GeodesicMethod::HeatMethod,
            max_distance: None,
            time_step_scale: 1.0,
        }
    }
}

/// What a [`GeodesicQuery`] is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeodesicPhase {
    /// Dijkstra: building the edge graph and walking it.
    Graph,
    /// Heat method: assembling the operators of the mesh.
    Assemble,
    /// Heat method: building the Laplacian's multigrid levels.
    Factorize,
    /// Heat method: diffusing heat from the sources.
    Diffuse,
    /// Heat method: solving for the distances.
    Poisson,
    /// The distances are ready.
    Done,
}

/// The mesh state a solver or query belongs to. (An undo brings the revision
/// back but keeps slots it added, so the slot counts are part of it.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MeshState {
    uid: u64,
    epoch: u32,
    revision: u64,
    vertex_slots: usize,
    triangle_slots: usize,
}

impl MeshState {
    fn of(mesh: &Mesh) -> MeshState {
        MeshState {
            uid: mesh.uid(),
            epoch: mesh.epoch(),
            revision: mesh.revision(),
            vertex_slots: mesh.vertex_slots(),
            triangle_slots: mesh.triangle_slots(),
        }
    }
}

/// Computes single- or multi-source geodesic distances from `sources` to every
/// vertex slot of `mesh`, writing `mesh.vertex_slots()` values into `out`.
///
/// Unreached vertices (disconnected components without a source, dead or unused
/// vertex slots, and values exceeding `options.max_distance`) receive
/// `f32::INFINITY`.
///
/// With the `parallel` feature, large meshes assemble and solve on rayon's
/// pool, with bit-identical results across thread counts.
///
/// # Errors
/// - [`FieldsError::InvalidInput`] if `sources` is empty, any source index is
///   out of bounds or dead, `options.time_step_scale` is non-positive/non-finite,
///   `options.max_distance` is negative or non-finite, or a live vertex position
///   referenced by live triangles or `sources` is non-finite.
/// - [`FieldsError::OutOfMemory`] if any internal or output allocation fails.
///
/// On error `out` is left unchanged.
pub fn geodesic_distances(
    mesh: &Mesh,
    sources: &[u32],
    options: &GeodesicOptions,
    out: &mut Vec<f32>,
) -> Result<()> {
    let mut query = GeodesicQuery::new(mesh, sources, options)?;
    while !query.step(mesh, usize::MAX)? {}
    write_out(query.distances().unwrap_or_default(), out)
}

/// [`geodesic_distances`] cut into steps, so that a caller can yield between
/// them, or stop: dropping the query ends it.
///
/// A step ends at a pass over the mesh, at a band of the diffusion or at an
/// iteration of a solve. How the query is cut changes nothing: the distances
/// are the same, bit for bit, as from [`geodesic_distances`] (which runs one
/// of these to the end). A query is tied to the mesh state it started from
/// ([`GeodesicQuery::is_for`]); stepping it on another state fails.
#[derive(Debug)]
pub struct GeodesicQuery {
    mesh: MeshState,
    sources: Vec<u32>,
    options: GeodesicOptions,
    stage: QueryStage,
    /// The heat method's operators, once built or given.
    solver: Option<Arc<GeodesicHeatSolver>>,
}

#[derive(Debug)]
enum QueryStage {
    Graph(Box<Graph>),
    Build(Box<SolverBuild>),
    Solve(Box<HeatSolve>),
    Done(Vec<f32>),
    Failed,
}

impl GeodesicQuery {
    /// A query of `mesh` with nothing done yet.
    ///
    /// # Errors
    /// As for [`geodesic_distances`], for what is wrong with the arguments.
    pub fn new(mesh: &Mesh, sources: &[u32], options: &GeodesicOptions) -> Result<GeodesicQuery> {
        Self::start(mesh, sources, options, None)
    }

    /// Like [`GeodesicQuery::new`], reusing the operators of `solver` when
    /// it was built for `mesh` as it is now ([`GeodesicHeatSolver::is_for`]).
    /// The time step is the query's: the operators do not depend on it.
    ///
    /// # Errors
    /// As for [`GeodesicQuery::new`].
    pub fn with_solver(
        mesh: &Mesh,
        sources: &[u32],
        options: &GeodesicOptions,
        solver: Arc<GeodesicHeatSolver>,
    ) -> Result<GeodesicQuery> {
        Self::start(mesh, sources, options, Some(solver))
    }

    fn start(
        mesh: &Mesh,
        sources: &[u32],
        options: &GeodesicOptions,
        solver: Option<Arc<GeodesicHeatSolver>>,
    ) -> Result<GeodesicQuery> {
        validate_options(options.time_step_scale, options.max_distance)?;
        validate_sources_against_mesh(mesh, sources)?;
        let mut own = Vec::new();
        reserve_exact(&mut own, sources.len())?;
        own.extend_from_slice(sources);
        let solver = solver.filter(|s| s.is_for(mesh));
        let stage =
            match (options.method, &solver) {
                (GeodesicMethod::Dijkstra, _) => QueryStage::Graph(Box::default()),
                (GeodesicMethod::HeatMethod, None) => {
                    QueryStage::Build(Box::new(SolverBuild::new(mesh, options.time_step_scale)?))
                }
                (GeodesicMethod::HeatMethod, Some(s)) => QueryStage::Solve(Box::new(
                    HeatSolve::new(s, sources, options.max_distance, options.time_step_scale)?,
                )),
            };
        Ok(GeodesicQuery {
            mesh: MeshState::of(mesh),
            sources: own,
            options: *options,
            stage,
            solver,
        })
    }

    /// Whether the query started from `mesh` as it is now.
    #[must_use]
    pub fn is_for(&self, mesh: &Mesh) -> bool {
        self.mesh == MeshState::of(mesh)
    }

    /// What the next step does.
    #[must_use]
    pub fn phase(&self) -> GeodesicPhase {
        match &self.stage {
            QueryStage::Graph(_) => GeodesicPhase::Graph,
            QueryStage::Build(build) => build.phase(),
            QueryStage::Solve(solve) => solve.phase(),
            QueryStage::Done(_) | QueryStage::Failed => GeodesicPhase::Done,
        }
    }

    /// Runs the query over about `work` more elements (at least one pass,
    /// band or iteration, unless it is done); returns whether the distances
    /// are ready.
    ///
    /// # Errors
    /// [`FieldsError::InvalidInput`] when `mesh` is not the state the query
    /// started from ([`GeodesicQuery::is_for`]), or for a non-finite position
    /// of a live triangle; [`FieldsError::OutOfMemory`] when an array cannot
    /// be allocated. The query is unusable after an error.
    pub fn step(&mut self, mesh: &Mesh, work: usize) -> Result<bool> {
        if !self.is_for(mesh) {
            return Err(FieldsError::InvalidInput(
                "the geodesic query started from another mesh state",
            ));
        }
        let mut left = work.max(1);
        while left > 0 {
            let spent = self.advance(mesh, left)?;
            if matches!(self.stage, QueryStage::Done(_) | QueryStage::Failed) {
                break;
            }
            left = left.saturating_sub(spent);
        }
        Ok(matches!(self.stage, QueryStage::Done(_)))
    }

    /// Runs the next pass; about how many elements it went over.
    fn advance(&mut self, mesh: &Mesh, work: usize) -> Result<usize> {
        let mut spent = 1;
        self.stage = match std::mem::replace(&mut self.stage, QueryStage::Failed) {
            QueryStage::Graph(mut graph) => {
                match graph.step(mesh, &self.sources, self.options.max_distance, work)? {
                    Some(done) => {
                        spent = done;
                        QueryStage::Graph(graph)
                    }
                    None => QueryStage::Done(graph.distances(mesh, self.options.max_distance)?),
                }
            }
            QueryStage::Build(mut build) => match build.step(mesh)? {
                Some(done) => {
                    spent = done;
                    QueryStage::Build(build)
                }
                None => {
                    let solver = self.solver.insert(Arc::new(build.finish()));
                    QueryStage::Solve(Box::new(HeatSolve::new(
                        solver,
                        &self.sources,
                        self.options.max_distance,
                        self.options.time_step_scale,
                    )?))
                }
            },
            QueryStage::Solve(mut solve) => {
                let Some(solver) = &self.solver else {
                    return Ok(spent);
                };
                match solve.step(solver, work)? {
                    Some(done) => {
                        spent = done;
                        QueryStage::Solve(solve)
                    }
                    None => QueryStage::Done(solve.finish()),
                }
            }
            over @ (QueryStage::Done(_) | QueryStage::Failed) => over,
        };
        Ok(spent)
    }

    /// The distances, once ready: one per vertex slot.
    #[must_use]
    pub fn distances(&self) -> Option<&[f32]> {
        match &self.stage {
            QueryStage::Done(d) => Some(d),
            _ => None,
        }
    }

    /// The distances, once ready, without copying them.
    #[must_use]
    pub fn into_distances(self) -> Option<Vec<f32>> {
        match self.stage {
            QueryStage::Done(d) => Some(d),
            _ => None,
        }
    }

    /// The heat method's operators, for later queries of the same mesh;
    /// `None` for the other method, and while they are being built.
    #[must_use]
    pub fn solver(&self) -> Option<&Arc<GeodesicHeatSolver>> {
        self.solver.as_ref()
    }
}

/// Copies `result` into `out`, which stays as it was when memory runs out.
fn write_out(result: &[f32], out: &mut Vec<f32>) -> Result<()> {
    if out.capacity() < result.len() {
        reserve_exact(out, result.len() - out.len())?;
    }
    out.clear();
    out.extend_from_slice(result);
    Ok(())
}

fn validate_options(time_step_scale: f64, max_distance: Option<f32>) -> Result<()> {
    if !time_step_scale.is_finite() || time_step_scale <= 0.0 {
        return Err(FieldsError::InvalidInput(
            "time_step_scale must be finite and strictly positive",
        ));
    }
    validate_max_distance(max_distance)
}

fn validate_max_distance(max_distance: Option<f32>) -> Result<()> {
    if let Some(d) = max_distance
        && (!d.is_finite() || d < 0.0)
    {
        return Err(FieldsError::InvalidInput(
            "max_distance must be finite and non-negative",
        ));
    }
    Ok(())
}

fn validate_sources_against_mesh(mesh: &Mesh, sources: &[u32]) -> Result<()> {
    if sources.is_empty() {
        return Err(FieldsError::InvalidInput("sources must not be empty"));
    }
    let slots = mesh.vertex_slots();
    let pos = mesh.positions();
    for &s in sources {
        let idx = s as usize;
        if idx >= slots || !mesh.is_vertex_alive(s) {
            return Err(FieldsError::InvalidInput(
                "source vertex index is out of range or dead",
            ));
        }
        if !pos[idx].iter().all(|c| c.is_finite()) {
            return Err(FieldsError::InvalidInput(
                "source vertex position must be finite",
            ));
        }
    }
    Ok(())
}

/// Dijkstra's walk over the mesh edges.
#[derive(Debug, Default)]
struct Graph {
    /// Where each vertex's edges start in `edges`, once they are built.
    offsets: Vec<u32>,
    /// Neighbour and length.
    edges: Vec<(u32, f64)>,
    dist: Vec<f64>,
    heap: BinaryHeap<HeapItem>,
}

impl Graph {
    /// Builds the edges, or settles about `work` more vertices; how much
    /// that was, or `None` when the walk is over.
    fn step(
        &mut self,
        mesh: &Mesh,
        sources: &[u32],
        max_distance: Option<f32>,
        work: usize,
    ) -> Result<Option<usize>> {
        if self.offsets.is_empty() {
            self.build(mesh, sources)?;
            return Ok(Some(mesh.triangle_slots().max(1)));
        }
        let max_d = max_distance.map(f64::from).unwrap_or(f64::INFINITY);
        let mut settled = 0usize;
        while let Some(HeapItem { dist: d, vertex: u }) = self.heap.pop() {
            if d > max_d {
                return Ok(None);
            }
            if d > self.dist[u as usize] {
                continue;
            }
            let start = self.offsets[u as usize] as usize;
            let end = self.offsets[u as usize + 1] as usize;
            for &(v, w) in &self.edges[start..end] {
                let nd = d + w;
                if nd <= max_d && nd < self.dist[v as usize] {
                    self.dist[v as usize] = nd;
                    heap_push(
                        &mut self.heap,
                        HeapItem {
                            dist: nd,
                            vertex: v,
                        },
                    )?;
                }
            }
            settled += 1;
            if settled >= work {
                return Ok(Some(settled));
            }
        }
        Ok(None)
    }

    fn build(&mut self, mesh: &Mesh, sources: &[u32]) -> Result<()> {
        let slots = mesh.vertex_slots();
        let pos = mesh.positions();

        let edge_cap = mesh.triangle_count().saturating_mul(6);
        let mut raw_edges: Vec<(u32, u32)> = Vec::new();
        reserve_exact(&mut raw_edges, edge_cap)?;

        for t in 0..mesh.triangle_slots() as u32 {
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };
            let corners = tri.map(|v| pos[v as usize]);
            if !corners.iter().flatten().all(|c| c.is_finite()) {
                return Err(FieldsError::InvalidInput(
                    "mesh vertex position must be finite",
                ));
            }
            for k in 0..3 {
                let a = tri[k];
                let b = tri[(k + 1) % 3];
                if a != b {
                    push(&mut raw_edges, (a, b))?;
                    push(&mut raw_edges, (b, a))?;
                }
            }
        }

        raw_edges.sort_unstable();
        raw_edges.dedup();

        let mut offsets = filled(slots + 1, 0u32)?;
        for &(u, _) in &raw_edges {
            offsets[u as usize + 1] += 1;
        }
        for i in 0..slots {
            offsets[i + 1] += offsets[i];
        }

        reserve_exact(&mut self.edges, raw_edges.len())?;
        for &(u, v) in &raw_edges {
            let len = norm(sub(widen(pos[v as usize]), widen(pos[u as usize])));
            self.edges.push((v, len));
        }
        drop(raw_edges);

        self.dist = filled(slots, f64::INFINITY)?;
        for &s in sources {
            if self.dist[s as usize] > 0.0 {
                self.dist[s as usize] = 0.0;
                heap_push(
                    &mut self.heap,
                    HeapItem {
                        dist: 0.0,
                        vertex: s,
                    },
                )?;
            }
        }
        self.offsets = offsets;
        Ok(())
    }

    fn distances(&self, mesh: &Mesh, max_distance: Option<f32>) -> Result<Vec<f32>> {
        let mut out = filled(mesh.vertex_slots(), f32::INFINITY)?;
        for (dst, &d) in out.iter_mut().zip(&self.dist) {
            if d.is_finite() {
                let val = d as f32;
                *dst = match max_distance {
                    Some(limit) if val > limit => f32::INFINITY,
                    _ => val,
                };
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::dot;
    use crate::grid::fail_allocations_after;
    use crate::test_util::*;

    #[test]
    fn dijkstra_cube_exact_edge_distances() {
        let m = cube();
        let mut out = Vec::new();
        let opts = GeodesicOptions {
            method: GeodesicMethod::Dijkstra,
            ..Default::default()
        };
        geodesic_distances(&m, &[0], &opts, &mut out).unwrap();
        assert_eq!(out.len(), 8);
        assert_eq!(out[0], 0.0);
        // Cube vertices in test_util::cube() lie on [-1, 1]^3 (edge = 2, face diag = 2*sqrt(2)).
        assert!((out[1] - 2.0).abs() < 1e-5, "edge neighbor got {}", out[1]);
        assert!(out[6] > 2.0 && out.iter().all(|&d| d.is_finite()));
    }

    #[test]
    fn heat_method_single_and_multi_source_on_icosphere() {
        let m = icosphere();
        let solver = GeodesicHeatSolver::build(&m, 1.0).unwrap();
        let mut d_single = Vec::new();
        solver.solve(&[0], None, &mut d_single).unwrap();

        let opts = GeodesicOptions::default();
        let mut d_oneshot = Vec::new();
        geodesic_distances(&m, &[0], &opts, &mut d_oneshot).unwrap();
        assert_eq!(d_single, d_oneshot);
        assert_eq!(d_single[0], 0.0);
        for (i, &d) in d_single.iter().enumerate() {
            assert!(d.is_finite() && d >= 0.0, "vertex {i} invalid d={d}");
        }

        // Farther geodesic vertex has larger angle on the unit sphere.
        let p0 = widen(m.positions()[0]);
        let far = d_single
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let expected_pi = dot(p0, widen(m.positions()[far])).clamp(-1.0, 1.0).acos();
        assert!(
            (f64::from(d_single[far]) - expected_pi).abs() / expected_pi < 0.05,
            "got {} expected {expected_pi}",
            d_single[far]
        );

        // Multi-source reuse.
        let opp = m
            .positions()
            .iter()
            .enumerate()
            .min_by(|a, b| dot(p0, widen(*a.1)).total_cmp(&dot(p0, widen(*b.1))))
            .unwrap()
            .0 as u32;
        let mut d_multi = Vec::new();
        solver.solve(&[0, opp], None, &mut d_multi).unwrap();
        assert_eq!(d_multi[0], 0.0);
        assert_eq!(d_multi[opp as usize], 0.0);
        assert!(d_multi[far] < d_single[far]);
    }

    #[test]
    fn disconnected_component_and_max_distance_cutoff() {
        let mut verts = cube_buffers().0;
        let mut tris = cube_buffers().1;
        let base = verts.len() as u32;
        for v in cube_buffers().0 {
            verts.push([v[0] + 10.0, v[1], v[2]]);
        }
        for t in cube_buffers().1 {
            tris.push([t[0] + base, t[1] + base, t[2] + base]);
        }
        let m = mesh(&verts, &tris);

        for method in [GeodesicMethod::Dijkstra, GeodesicMethod::HeatMethod] {
            let opts = GeodesicOptions {
                method,
                ..Default::default()
            };
            let mut out = Vec::new();
            geodesic_distances(&m, &[0], &opts, &mut out).unwrap();
            for (v, &d) in out.iter().enumerate().take(8) {
                assert!(d.is_finite(), "reached component missing {v}");
            }
            for &d in out.iter().take(16).skip(8) {
                assert!(d.is_infinite(), "unreached component got {d}");
            }

            let capped = GeodesicOptions {
                method,
                max_distance: Some(1.5),
                time_step_scale: 1.0,
            };
            geodesic_distances(&m, &[0], &capped, &mut out).unwrap();
            assert_eq!(out[0], 0.0);
            assert!(out[1].is_infinite());
        }
    }

    #[test]
    fn invalid_inputs_preserve_output() {
        let m = tetrahedron();
        let sentinel = vec![42.0f32, 99.0];
        let mut out = sentinel.clone();

        assert!(matches!(
            geodesic_distances(&m, &[], &GeodesicOptions::default(), &mut out),
            Err(FieldsError::InvalidInput(_))
        ));
        assert_eq!(out, sentinel);

        assert!(matches!(
            geodesic_distances(&m, &[999], &GeodesicOptions::default(), &mut out),
            Err(FieldsError::InvalidInput(_))
        ));
        assert_eq!(out, sentinel);

        let bad_scale = GeodesicOptions {
            time_step_scale: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            geodesic_distances(&m, &[0], &bad_scale, &mut out),
            Err(FieldsError::InvalidInput(_))
        ));
        assert_eq!(out, sentinel);

        let bad_cap = GeodesicOptions {
            max_distance: Some(-1.0),
            ..Default::default()
        };
        assert!(matches!(
            geodesic_distances(&m, &[0], &bad_cap, &mut out),
            Err(FieldsError::InvalidInput(_))
        ));
        assert_eq!(out, sentinel);
    }

    #[test]
    fn oom_fault_injection_is_clean() {
        let m = icosphere();
        for method in [GeodesicMethod::Dijkstra, GeodesicMethod::HeatMethod] {
            let opts = GeodesicOptions {
                method,
                ..Default::default()
            };
            for fail_at in 0..25 {
                let mut out = vec![7.5f32];
                fail_allocations_after(Some(fail_at));
                let res = geodesic_distances(&m, &[0], &opts, &mut out);
                fail_allocations_after(None);
                match res {
                    Ok(()) => break,
                    Err(FieldsError::OutOfMemory) => assert_eq!(out, vec![7.5f32]),
                    Err(other) => panic!("unexpected error: {other:?}"),
                }
            }
        }
    }
}
