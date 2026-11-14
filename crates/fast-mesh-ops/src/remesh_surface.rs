//! Isotropic and curvature-adaptive surface remeshing (Botsch-Kobbelt).
//!
//! [`remesh_surface`] applies iterative edge splits, link-safe collapses,
//! valence-reducing edge flips, tangential vertex relaxation, and optional BVH
//! surface reprojection through one atomic [`Transaction`]. Unlike region
//! [`remesh`](crate::remesh::remesh), it remeshes the full surface cleanly and
//! drops [`UV_CHANNEL`](fast_mesh_store::UV_CHANNEL) undoably when present.

use std::cmp::Ordering;

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_fields::curvatures;
use fast_mesh_query::{ClosestHit, ClosestOptions, closest_point_many};
use fast_mesh_store::{Mesh, Transaction, TriangleId, UV_CHANNEL, VertexId};

use crate::remesh::{COLLAPSE_FACTOR, SPLIT_FACTOR};
use crate::{OpsError, Result, filled, push, reserve};

/// Maximum allowed iteration cap to keep surface remeshing bounded.
const MAX_SURFACE_REMESH_ITERATIONS: usize = 64;

/// Sizing strategy for [`remesh_surface`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurfaceRemeshMode {
    /// Uniform target edge length across the entire surface.
    #[default]
    Isotropic,
    /// Curvature-adaptive edge length derived from discrete mean curvature,
    /// clamped into `[min_edge_length, max_edge_length]`.
    Adaptive,
}

/// Options for [`remesh_surface`] and [`remesh_surface_with_mask`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceRemeshOptions {
    /// Isotropic (uniform) or curvature-adaptive edge sizing. Default `Isotropic`.
    pub mode: SurfaceRemeshMode,
    /// Nominal target edge length in mesh units (must be positive and finite).
    pub target_edge_length: f32,
    /// Lower edge-length bound in `Adaptive` mode. Defaults to `0.35 * target_edge_length`
    /// when `None`.
    pub min_edge_length: Option<f32>,
    /// Upper edge-length bound in `Adaptive` mode. Defaults to `2.0 * target_edge_length`
    /// when `None`.
    pub max_edge_length: Option<f32>,
    /// Number of split/collapse/flip/relax/reproject passes (capped at 64).
    /// Default `5`.
    pub iterations: usize,
    /// Tangential relaxation step strength in `[0.0, 1.0]`. Default `0.5`.
    pub smooth_strength: f32,
    /// Reproject moved interior vertices onto the input reference surface via BVH.
    /// Default `true`.
    pub project_to_surface: bool,
    /// Keep open boundary vertices fixed during collapse, relaxation, and projection.
    /// Default `true`.
    pub lock_boundaries: bool,
    /// Detect sharp dihedral edges above [`SurfaceRemeshOptions::crease_angle_deg`]
    /// and preserve sharp creases and corners. Default `true`.
    pub lock_creases: bool,
    /// Dihedral angle threshold in degrees (`[0.0, 180.0]`) for feature creases.
    /// Default `30.0`.
    pub crease_angle_deg: f32,
}

impl Default for SurfaceRemeshOptions {
    fn default() -> Self {
        Self {
            mode: SurfaceRemeshMode::Isotropic,
            target_edge_length: 0.1,
            min_edge_length: None,
            max_edge_length: None,
            iterations: 5,
            smooth_strength: 0.5,
            project_to_surface: true,
            lock_boundaries: true,
            lock_creases: true,
            crease_angle_deg: 30.0,
        }
    }
}

impl SurfaceRemeshOptions {
    /// Returns a builder for [`SurfaceRemeshOptions`].
    pub fn builder() -> SurfaceRemeshOptionsBuilder {
        SurfaceRemeshOptionsBuilder::default()
    }
}

/// Builder for [`SurfaceRemeshOptions`].
#[derive(Debug, Clone, Default)]
pub struct SurfaceRemeshOptionsBuilder {
    options: SurfaceRemeshOptions,
}

impl SurfaceRemeshOptionsBuilder {
    /// Sets isotropic or curvature-adaptive sizing mode.
    pub fn mode(mut self, mode: SurfaceRemeshMode) -> Self {
        self.options.mode = mode;
        self
    }

    /// Sets the target edge length in mesh units.
    pub fn target_edge_length(mut self, target_edge_length: f32) -> Self {
        self.options.target_edge_length = target_edge_length;
        self
    }

    /// Sets the minimum adaptive edge length.
    pub fn min_edge_length(mut self, min_edge_length: f32) -> Self {
        self.options.min_edge_length = Some(min_edge_length);
        self
    }

    /// Sets the maximum adaptive edge length.
    pub fn max_edge_length(mut self, max_edge_length: f32) -> Self {
        self.options.max_edge_length = Some(max_edge_length);
        self
    }

    /// Sets the number of remeshing iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.options.iterations = iterations;
        self
    }

    /// Sets tangential smoothing strength in `[0.0, 1.0]`.
    pub fn smooth_strength(mut self, smooth_strength: f32) -> Self {
        self.options.smooth_strength = smooth_strength;
        self
    }

    /// Controls whether moved interior vertices reproject onto the reference surface.
    pub fn project_to_surface(mut self, project_to_surface: bool) -> Self {
        self.options.project_to_surface = project_to_surface;
        self
    }

    /// Controls whether boundary loops stay locked.
    pub fn lock_boundaries(mut self, lock_boundaries: bool) -> Self {
        self.options.lock_boundaries = lock_boundaries;
        self
    }

    /// Controls whether sharp dihedral creases stay locked.
    pub fn lock_creases(mut self, lock_creases: bool) -> Self {
        self.options.lock_creases = lock_creases;
        self
    }

    /// Sets the dihedral crease threshold angle in degrees (`[0.0, 180.0]`).
    pub fn crease_angle_deg(mut self, crease_angle_deg: f32) -> Self {
        self.options.crease_angle_deg = crease_angle_deg;
        self
    }

    /// Builds [`SurfaceRemeshOptions`].
    pub fn build(self) -> SurfaceRemeshOptions {
        self.options
    }
}

/// Summary report produced by [`remesh_surface`] and [`remesh_surface_with_mask`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SurfaceRemeshReport {
    /// Alive triangle count before remeshing.
    pub initial_triangles: usize,
    /// Alive triangle count after remeshing.
    pub final_triangles: usize,
    /// Total edge splits performed across all iterations.
    pub splits: usize,
    /// Total edge collapses performed across all iterations.
    pub collapses: usize,
    /// Total valence-reducing edge flips performed across all iterations.
    pub flips: usize,
    /// Number of remeshing rounds executed.
    pub iterations_run: usize,
    /// `true` when [`UV_CHANNEL`](fast_mesh_store::UV_CHANNEL) existed and was dropped.
    pub uvs_dropped: bool,
}

#[derive(Clone, Copy)]
struct WorkTri {
    orig_id: Option<TriangleId>,
    corners: [u32; 3],
    alive: bool,
}

#[inline]
fn f64v(p: [f32; 3]) -> [f64; 3] {
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

#[inline]
fn key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

#[inline]
fn carries(corners: &[u32; 3], a: u32, b: u32) -> bool {
    corners.contains(&a) && corners.contains(&b)
}

#[inline]
fn runs_directed(corners: &[u32; 3], a: u32, b: u32) -> bool {
    (corners[0] == a && corners[1] == b)
        || (corners[1] == a && corners[2] == b)
        || (corners[2] == a && corners[0] == b)
}

#[inline]
fn apex_of(corners: &[u32; 3], a: u32, b: u32) -> u32 {
    if corners[0] != a && corners[0] != b {
        corners[0]
    } else if corners[1] != a && corners[1] != b {
        corners[1]
    } else {
        corners[2]
    }
}

#[inline]
fn sorted_triple(mut c: [u32; 3]) -> [u32; 3] {
    if c[0] > c[1] {
        c.swap(0, 1);
    }
    if c[1] > c[2] {
        c.swap(1, 2);
    }
    if c[0] > c[1] {
        c.swap(0, 1);
    }
    c
}

#[inline]
fn tri_normal(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> [f64; 3] {
    cross3(sub3(p1, p0), sub3(p2, p0))
}

#[inline]
fn unit_normal_tri(pos: &[[f32; 3]], c: [u32; 3]) -> Option<[f64; 3]> {
    let n = tri_normal(
        f64v(pos[c[0] as usize]),
        f64v(pos[c[1] as usize]),
        f64v(pos[c[2] as usize]),
    );
    let len = norm3(n);
    if len.is_finite() && len > 1e-18 {
        Some([n[0] / len, n[1] / len, n[2] / len])
    } else {
        None
    }
}

fn validate_options(
    mesh: &Mesh,
    options: &SurfaceRemeshOptions,
    locked_vertices: &[VertexId],
) -> Result<(f64, f64)> {
    if !options.target_edge_length.is_finite() || options.target_edge_length <= 0.0 {
        return Err(OpsError::InvalidInput(
            "surface remesh target_edge_length must be positive and finite",
        ));
    }
    if let Some(m) = options.min_edge_length
        && (!m.is_finite() || m <= 0.0)
    {
        return Err(OpsError::InvalidInput(
            "surface remesh min_edge_length must be positive and finite",
        ));
    }
    if let Some(m) = options.max_edge_length
        && (!m.is_finite() || m <= 0.0)
    {
        return Err(OpsError::InvalidInput(
            "surface remesh max_edge_length must be positive and finite",
        ));
    }
    let target = f64::from(options.target_edge_length);
    let max_edge = options
        .max_edge_length
        .map(f64::from)
        .unwrap_or_else(|| (target * 2.0).max(options.min_edge_length.map_or(0.0, f64::from)));
    let min_edge = options
        .min_edge_length
        .map(f64::from)
        .unwrap_or_else(|| (target * 0.35).min(max_edge));
    if min_edge > max_edge {
        return Err(OpsError::InvalidInput(
            "surface remesh min_edge_length must not exceed max_edge_length",
        ));
    }
    if !options.smooth_strength.is_finite() || !(0.0..=1.0).contains(&options.smooth_strength) {
        return Err(OpsError::InvalidInput(
            "surface remesh smooth_strength must be finite and in [0, 1]",
        ));
    }
    if !options.crease_angle_deg.is_finite() || !(0.0..=180.0).contains(&options.crease_angle_deg) {
        return Err(OpsError::InvalidInput(
            "surface remesh crease_angle_deg must be finite and in [0, 180]",
        ));
    }
    let slots = mesh.vertex_slots();
    for &v in locked_vertices {
        if (v as usize) >= slots || !mesh.is_vertex_alive(v) {
            return Err(OpsError::InvalidInput(
                "locked vertex id does not refer to an alive vertex",
            ));
        }
    }
    Ok((min_edge, max_edge))
}

fn fill_owners(
    a: u32,
    b: u32,
    v2t: &[Vec<u32>],
    tris: &[WorkTri],
    owners: &mut Vec<usize>,
) -> Result<()> {
    owners.clear();
    reserve(owners, 4)?;
    for &ti in &v2t[a as usize] {
        let t = &tris[ti as usize];
        if t.alive && carries(&t.corners, a, b) {
            push(owners, ti as usize)?;
        }
    }
    owners.sort_unstable();
    owners.dedup();
    Ok(())
}

fn fill_neighbors(v: u32, v2t: &[Vec<u32>], tris: &[WorkTri], out: &mut Vec<u32>) -> Result<()> {
    out.clear();
    reserve(out, 16)?;
    for &ti in &v2t[v as usize] {
        let t = &tris[ti as usize];
        if !t.alive || !t.corners.contains(&v) {
            continue;
        }
        for &w in &t.corners {
            if w != v {
                push(out, w)?;
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(())
}

struct RemeshState {
    pos: Vec<[f32; 3]>,
    tris: Vec<WorkTri>,
    v2t: Vec<Vec<u32>>,
    vlive: Vec<bool>,
    was_orig_vert: Vec<bool>,
    target_size: Vec<f64>,
    masked: Vec<bool>,
    is_boundary: Vec<bool>,
    is_crease: Vec<bool>,
    is_corner: Vec<bool>,
    crease_adj: Vec<Vec<u32>>,
    cos_crease: f64,
    lock_creases: bool,
    lock_boundaries: bool,
    l_min: f64,
    l_max: f64,
}

impl RemeshState {
    fn tag_features(&mut self) -> Result<()> {
        let slots = self.pos.len();
        self.is_boundary.clear();
        self.is_boundary.resize(slots, false);
        self.is_crease.clear();
        self.is_crease.resize(slots, false);
        self.is_corner.clear();
        self.is_corner.resize(slots, false);
        for adj in &mut self.crease_adj {
            adj.clear();
        }
        if self.crease_adj.len() < slots {
            let needed = slots - self.crease_adj.len();
            reserve(&mut self.crease_adj, needed)?;
            self.crease_adj.resize_with(slots, Vec::new);
        }

        let mut edge_tris: Vec<((u32, u32), u32)> = Vec::new();
        let live_tris = self.tris.iter().filter(|t| t.alive).count();
        reserve(&mut edge_tris, live_tris * 3)?;
        for (ti, t) in self.tris.iter().enumerate() {
            if !t.alive {
                continue;
            }
            for k in 0..3 {
                push(
                    &mut edge_tris,
                    (key(t.corners[k], t.corners[(k + 1) % 3]), ti as u32),
                )?;
            }
        }
        edge_tris.sort_unstable();

        let mut start = 0;
        while start < edge_tris.len() {
            let e = edge_tris[start].0;
            let mut end = start + 1;
            while end < edge_tris.len() && edge_tris[end].0 == e {
                end += 1;
            }
            let cnt = end - start;
            if cnt == 1 {
                self.is_boundary[e.0 as usize] = true;
                self.is_boundary[e.1 as usize] = true;
            } else if self.lock_creases {
                let mut sharp = cnt > 2;
                if cnt == 2
                    && let (Some(n0), Some(n1)) = (
                        unit_normal_tri(&self.pos, self.tris[edge_tris[start].1 as usize].corners),
                        unit_normal_tri(
                            &self.pos,
                            self.tris[edge_tris[start + 1].1 as usize].corners,
                        ),
                    )
                {
                    sharp = dot3(n0, n1).clamp(-1.0, 1.0) < self.cos_crease - 1e-12;
                }
                if sharp {
                    push(&mut self.crease_adj[e.0 as usize], e.1)?;
                    push(&mut self.crease_adj[e.1 as usize], e.0)?;
                }
            }
            start = end;
        }

        if self.lock_creases {
            for v in 0..slots {
                if !self.vlive[v] {
                    continue;
                }
                self.crease_adj[v].sort_unstable();
                self.crease_adj[v].dedup();
                let deg = self.crease_adj[v].len();
                if deg == 1 || deg >= 3 {
                    self.is_corner[v] = true;
                    self.is_crease[v] = true;
                } else if deg == 2 {
                    self.is_crease[v] = true;
                    let pv = f64v(self.pos[v]);
                    let d0 = sub3(f64v(self.pos[self.crease_adj[v][0] as usize]), pv);
                    let d1 = sub3(f64v(self.pos[self.crease_adj[v][1] as usize]), pv);
                    let l0 = norm3(d0);
                    let l1 = norm3(d1);
                    if l0 <= 1e-12 || l1 <= 1e-12 || dot3(d0, d1) / (l0 * l1) > -0.5 {
                        self.is_corner[v] = true;
                    }
                }
            }
        }
        Ok(())
    }

    fn edge_is_crease_owners(&self, owners: &[usize]) -> bool {
        if !self.lock_creases || owners.len() != 2 {
            return owners.len() > 2;
        }
        let (Some(n0), Some(n1)) = (
            unit_normal_tri(&self.pos, self.tris[owners[0]].corners),
            unit_normal_tri(&self.pos, self.tris[owners[1]].corners),
        ) else {
            return true;
        };
        dot3(n0, n1).clamp(-1.0, 1.0) < self.cos_crease - 1e-12
    }

    fn push_vertex(&mut self, p: [f32; 3], size: f64, boundary: bool, crease: bool) -> Result<u32> {
        let id = self.pos.len() as u32;
        push(&mut self.pos, p)?;
        push(&mut self.vlive, true)?;
        push(&mut self.was_orig_vert, false)?;
        push(&mut self.target_size, size.clamp(self.l_min, self.l_max))?;
        push(&mut self.masked, false)?;
        push(&mut self.is_boundary, boundary)?;
        push(&mut self.is_crease, crease)?;
        push(&mut self.is_corner, false)?;
        push(&mut self.v2t, Vec::new())?;
        push(&mut self.crease_adj, Vec::new())?;
        Ok(id)
    }

    fn collect_sorted_edges(&self) -> Result<Vec<(f64, u32, u32)>> {
        let mut raw: Vec<(u32, u32)> = Vec::new();
        let live_tris = self.tris.iter().filter(|t| t.alive).count();
        reserve(&mut raw, live_tris * 3)?;
        for t in &self.tris {
            if !t.alive {
                continue;
            }
            for k in 0..3 {
                push(&mut raw, key(t.corners[k], t.corners[(k + 1) % 3]))?;
            }
        }
        raw.sort_unstable();
        raw.dedup();
        let mut out: Vec<(f64, u32, u32)> = Vec::new();
        reserve(&mut out, raw.len())?;
        for (a, b) in raw {
            let d = norm3(sub3(f64v(self.pos[a as usize]), f64v(self.pos[b as usize])));
            push(&mut out, (d, a, b))?;
        }
        Ok(out)
    }

    fn split_phase(&mut self) -> Result<usize> {
        let mut edges = self.collect_sorted_edges()?;
        // Longest edges first, deterministic tiebreak on vertex ids.
        edges.sort_unstable_by(|x, y| {
            y.0.partial_cmp(&x.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| x.1.cmp(&y.1))
                .then_with(|| x.2.cmp(&y.2))
        });

        let mut splits = 0usize;
        let mut owners: Vec<usize> = Vec::new();
        let mut new_tris: Vec<[u32; 3]> = Vec::new();
        reserve(&mut new_tris, 2)?;

        for (_, a, b) in edges {
            if !self.vlive[a as usize] || !self.vlive[b as usize] {
                continue;
            }
            let target = 0.5 * (self.target_size[a as usize] + self.target_size[b as usize]);
            let pa = f64v(self.pos[a as usize]);
            let pb = f64v(self.pos[b as usize]);
            let len = norm3(sub3(pb, pa));
            if len <= SPLIT_FACTOR * target {
                continue;
            }
            fill_owners(a, b, &self.v2t, &self.tris, &mut owners)?;
            if owners.is_empty() || owners.len() > 2 {
                continue;
            }

            let mid64 = scale3(add3(pa, pb), 0.5);
            let mid32 = [mid64[0] as f32, mid64[1] as f32, mid64[2] as f32];
            if !mid32.iter().all(|c| c.is_finite()) {
                continue;
            }
            let edge_crease = self.edge_is_crease_owners(&owners);
            let edge_boundary = owners.len() == 1;
            let m = self.push_vertex(mid32, target, edge_boundary, edge_crease)?;

            for &ti in &owners {
                let c = self.tris[ti].corners;
                let apex = apex_of(&c, a, b);
                let (keep_tri, add_tri) = if runs_directed(&c, a, b) {
                    ([a, m, apex], [m, b, apex])
                } else if runs_directed(&c, b, a) {
                    ([m, a, apex], [b, m, apex])
                } else {
                    continue;
                };
                self.tris[ti].corners = keep_tri;
                let new_idx = self.tris.len() as u32;
                push(
                    &mut self.tris,
                    WorkTri {
                        orig_id: None,
                        corners: add_tri,
                        alive: true,
                    },
                )?;
                self.v2t[b as usize].retain(|&id| id != ti as u32);
                push(&mut self.v2t[b as usize], new_idx)?;
                self.v2t[b as usize].sort_unstable();
                self.v2t[b as usize].dedup();

                push(&mut self.v2t[apex as usize], new_idx)?;
                self.v2t[apex as usize].sort_unstable();
                self.v2t[apex as usize].dedup();

                push(&mut self.v2t[m as usize], ti as u32)?;
                push(&mut self.v2t[m as usize], new_idx)?;
            }
            self.v2t[m as usize].sort_unstable();
            self.v2t[m as usize].dedup();
            splits += 1;
        }
        Ok(splits)
    }

    fn collapse_phase(&mut self) -> Result<usize> {
        let mut edges = self.collect_sorted_edges()?;
        // Shortest edges first, deterministic tiebreak on vertex ids.
        edges.sort_unstable_by(|x, y| {
            x.0.partial_cmp(&y.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| x.1.cmp(&y.1))
                .then_with(|| x.2.cmp(&y.2))
        });

        let mut collapses = 0usize;
        let mut owners: Vec<usize> = Vec::new();
        let mut na: Vec<u32> = Vec::new();
        let mut nb: Vec<u32> = Vec::new();
        let mut apexes: Vec<u32> = Vec::new();
        let mut common: Vec<u32> = Vec::new();
        let mut fan: Vec<usize> = Vec::new();
        let mut tri_keys: Vec<[u32; 3]> = Vec::new();
        let mut v2t_drop: Vec<u32> = Vec::new();

        for (_, a, b) in edges {
            if !self.vlive[a as usize] || !self.vlive[b as usize] {
                continue;
            }
            if self.masked[a as usize] && self.masked[b as usize] {
                continue;
            }
            if self.lock_boundaries
                && (self.is_boundary[a as usize] || self.is_boundary[b as usize])
            {
                continue;
            }
            let target = 0.5 * (self.target_size[a as usize] + self.target_size[b as usize]);
            let pa = f64v(self.pos[a as usize]);
            let pb = f64v(self.pos[b as usize]);
            let len = norm3(sub3(pb, pa));
            if len >= COLLAPSE_FACTOR * target {
                continue;
            }

            fill_owners(a, b, &self.v2t, &self.tris, &mut owners)?;
            if owners.is_empty() || owners.len() > 2 {
                continue;
            }
            if owners.len() == 2 {
                let c0 = self.tris[owners[0]].corners;
                let c1 = self.tris[owners[1]].corners;
                let consistent = (runs_directed(&c0, a, b) && runs_directed(&c1, b, a))
                    || (runs_directed(&c0, b, a) && runs_directed(&c1, a, b));
                if !consistent {
                    continue;
                }
            }

            let is_crease_edge = self.edge_is_crease_owners(&owners);
            if self.lock_creases {
                if self.is_corner[a as usize] && self.is_corner[b as usize] {
                    continue;
                }
                if (self.is_crease[a as usize] || self.is_corner[a as usize])
                    && (self.is_crease[b as usize] || self.is_corner[b as usize])
                    && !is_crease_edge
                {
                    continue;
                }
            }

            // Link condition: common neighbors must match owner apexes cleanly.
            fill_neighbors(a, &self.v2t, &self.tris, &mut na)?;
            fill_neighbors(b, &self.v2t, &self.tris, &mut nb)?;
            apexes.clear();
            reserve(&mut apexes, owners.len())?;
            for &o in &owners {
                push(&mut apexes, apex_of(&self.tris[o].corners, a, b))?;
            }
            apexes.sort_unstable();
            apexes.dedup();
            if apexes.len() != owners.len() {
                continue;
            }
            common.clear();
            reserve(&mut common, na.len().min(nb.len()))?;
            for &w in &na {
                if w != a && w != b && nb.binary_search(&w).is_ok() {
                    push(&mut common, w)?;
                }
            }
            if common != apexes {
                continue;
            }

            // Determine placement and surviving vertex id cleanly.
            let (keep, drop) = if a < b { (a, b) } else { (b, a) };
            let pnew64 = if self.masked[a as usize] {
                pa
            } else if self.masked[b as usize] {
                pb
            } else if !self.lock_boundaries
                && (self.is_boundary[a as usize] ^ self.is_boundary[b as usize])
            {
                if self.is_boundary[a as usize] { pa } else { pb }
            } else if self.lock_creases {
                if self.is_corner[a as usize]
                    || ((self.is_crease[a as usize]) && !self.is_crease[b as usize])
                {
                    pa
                } else if self.is_corner[b as usize]
                    || ((self.is_crease[b as usize]) && !self.is_crease[a as usize])
                {
                    pb
                } else {
                    scale3(add3(pa, pb), 0.5)
                }
            } else {
                scale3(add3(pa, pb), 0.5)
            };
            let pnew32 = [pnew64[0] as f32, pnew64[1] as f32, pnew64[2] as f32];
            if !pnew32.iter().all(|c| c.is_finite()) {
                continue;
            }
            let keep_pos = f64v(pnew32);

            fan.clear();
            reserve(&mut fan, na.len() + nb.len())?;
            for &v in &[a, b] {
                for &ti in &self.v2t[v as usize] {
                    let idx = ti as usize;
                    if !self.tris[idx].alive || owners.contains(&idx) {
                        continue;
                    }
                    push(&mut fan, idx)?;
                }
            }
            fan.sort_unstable();
            fan.dedup();
            if fan.is_empty() {
                continue;
            }

            tri_keys.clear();
            reserve(&mut tri_keys, fan.len())?;
            let mut invalid = false;
            for &idx in &fan {
                let old = self.tris[idx].corners;
                let mapped = old.map(|v| if v == drop { keep } else { v });
                push(&mut tri_keys, sorted_triple(mapped))?;
                let before = tri_normal(
                    f64v(self.pos[old[0] as usize]),
                    f64v(self.pos[old[1] as usize]),
                    f64v(self.pos[old[2] as usize]),
                );
                if dot3(before, before) <= 1e-24 {
                    invalid = true;
                    break;
                }
                let after = tri_normal(
                    if old[0] == keep || old[0] == drop {
                        keep_pos
                    } else {
                        f64v(self.pos[old[0] as usize])
                    },
                    if old[1] == keep || old[1] == drop {
                        keep_pos
                    } else {
                        f64v(self.pos[old[1] as usize])
                    },
                    if old[2] == keep || old[2] == drop {
                        keep_pos
                    } else {
                        f64v(self.pos[old[2] as usize])
                    },
                );
                if dot3(after, after) <= 1e-24 || dot3(after, before) <= 0.0 {
                    invalid = true;
                    break;
                }
            }
            if !invalid && tri_keys.len() > 1 {
                tri_keys.sort_unstable();
                if tri_keys.windows(2).any(|w| w[0] == w[1]) {
                    invalid = true;
                }
            }
            if invalid {
                continue;
            }

            // Preserve target size of fixed endpoint when locked; else average.
            if self.masked[keep as usize] || (self.lock_creases && self.is_corner[keep as usize]) {
                // keep existing target_size
            } else if self.masked[drop as usize]
                || (self.lock_creases && self.is_corner[drop as usize])
            {
                self.target_size[keep as usize] = self.target_size[drop as usize];
            } else {
                self.target_size[keep as usize] =
                    0.5 * (self.target_size[keep as usize] + self.target_size[drop as usize]);
            }

            self.pos[keep as usize] = pnew32;
            self.masked[keep as usize] |= self.masked[drop as usize];
            self.is_boundary[keep as usize] |= self.is_boundary[drop as usize];
            self.is_crease[keep as usize] |= self.is_crease[drop as usize];
            self.is_corner[keep as usize] |= self.is_corner[drop as usize];

            for &o in &owners {
                self.tris[o].alive = false;
            }
            self.v2t[keep as usize].retain(|&i| self.tris[i as usize].alive);
            v2t_drop.clear();
            reserve(&mut v2t_drop, self.v2t[drop as usize].len())?;
            v2t_drop.extend_from_slice(&self.v2t[drop as usize]);
            self.v2t[drop as usize].clear();
            reserve(&mut self.v2t[keep as usize], v2t_drop.len())?;
            for &idx in &v2t_drop {
                let t = &mut self.tris[idx as usize];
                if !t.alive {
                    continue;
                }
                t.corners = t.corners.map(|v| if v == drop { keep } else { v });
                push(&mut self.v2t[keep as usize], idx)?;
            }
            self.v2t[keep as usize].sort_unstable();
            self.v2t[keep as usize].dedup();
            self.vlive[drop as usize] = false;
            collapses += 1;
        }
        Ok(collapses)
    }

    fn flip_phase(&mut self) -> Result<usize> {
        let mut raw: Vec<(u32, u32)> = Vec::new();
        let live_tris = self.tris.iter().filter(|t| t.alive).count();
        reserve(&mut raw, live_tris * 3)?;
        for t in &self.tris {
            if !t.alive {
                continue;
            }
            for k in 0..3 {
                push(&mut raw, key(t.corners[k], t.corners[(k + 1) % 3]))?;
            }
        }
        raw.sort_unstable();
        raw.dedup();

        let mut flips = 0usize;
        let mut owners: Vec<usize> = Vec::new();
        let mut n_scratch: Vec<u32> = Vec::new();

        for (a, b) in raw {
            if !self.vlive[a as usize] || !self.vlive[b as usize] {
                continue;
            }
            fill_owners(a, b, &self.v2t, &self.tris, &mut owners)?;
            if owners.len() != 2 {
                continue;
            }
            if self.edge_is_crease_owners(&owners) {
                continue;
            }
            let (mut t0, mut t1) = (owners[0], owners[1]);
            let c0 = self.tris[t0].corners;
            let c1 = self.tris[t1].corners;
            if runs_directed(&c0, b, a) && runs_directed(&c1, a, b) {
                core::mem::swap(&mut t0, &mut t1);
            }
            let c0 = self.tris[t0].corners;
            let c1 = self.tris[t1].corners;
            if !runs_directed(&c0, a, b) || !runs_directed(&c1, b, a) {
                continue;
            }
            let c = apex_of(&c0, a, b);
            let d = apex_of(&c1, a, b);
            if c == d {
                continue;
            }
            // Reject if opposite edge (c, d) already exists cleanly.
            let cd_exists = self.v2t[c as usize].iter().any(|&ti| {
                self.tris[ti as usize].alive && self.tris[ti as usize].corners.contains(&d)
            });
            if cd_exists {
                continue;
            }

            let target_val = |v: u32| -> isize { if self.is_boundary[v as usize] { 4 } else { 6 } };
            let mut val = [0isize; 4];
            for (slot, &v) in [a, b, c, d].iter().enumerate() {
                fill_neighbors(v, &self.v2t, &self.tris, &mut n_scratch)?;
                val[slot] = n_scratch.len() as isize;
            }
            if val[0] <= 3 || val[1] <= 3 {
                continue;
            }
            let dev_before = (val[0] - target_val(a)).abs()
                + (val[1] - target_val(b)).abs()
                + (val[2] - target_val(c)).abs()
                + (val[3] - target_val(d)).abs();
            let dev_after = ((val[0] - 1) - target_val(a)).abs()
                + ((val[1] - 1) - target_val(b)).abs()
                + ((val[2] + 1) - target_val(c)).abs()
                + ((val[3] + 1) - target_val(d)).abs();
            if dev_after >= dev_before {
                continue;
            }

            let (Some(n0), Some(n1)) = (
                unit_normal_tri(&self.pos, c0),
                unit_normal_tri(&self.pos, c1),
            ) else {
                continue;
            };
            // Avoid flipping across sharp ridges even when creases are unlocked.
            if dot3(n0, n1) < -0.25 {
                continue;
            }
            let f0 = [c, a, d];
            let f1 = [d, b, c];
            let (Some(m0), Some(m1)) = (
                unit_normal_tri(&self.pos, f0),
                unit_normal_tri(&self.pos, f1),
            ) else {
                continue;
            };
            if dot3(m0, n0) <= 0.0
                || dot3(m0, n1) <= 0.0
                || dot3(m1, n0) <= 0.0
                || dot3(m1, n1) <= 0.0
            {
                continue;
            }

            self.tris[t0].corners = f0;
            self.tris[t1].corners = f1;
            self.v2t[a as usize].retain(|&i| i != t1 as u32);
            self.v2t[b as usize].retain(|&i| i != t0 as u32);
            push(&mut self.v2t[c as usize], t1 as u32)?;
            self.v2t[c as usize].sort_unstable();
            self.v2t[c as usize].dedup();
            push(&mut self.v2t[d as usize], t0 as u32)?;
            self.v2t[d as usize].sort_unstable();
            self.v2t[d as usize].dedup();
            flips += 1;
        }
        Ok(flips)
    }

    fn smooth_phase(&mut self, lambda: f64) -> Result<()> {
        if lambda <= 0.0 {
            return Ok(());
        }
        let slots = self.pos.len();
        let mut ring: Vec<u32> = Vec::new();
        for v in 0..slots {
            if !self.vlive[v] || self.masked[v] {
                continue;
            }
            if self.lock_boundaries && self.is_boundary[v] {
                continue;
            }
            if self.lock_creases && self.is_corner[v] {
                continue;
            }
            fill_neighbors(v as u32, &self.v2t, &self.tris, &mut ring)?;
            if ring.is_empty() {
                continue;
            }
            let pv = f64v(self.pos[v]);
            let mut centroid = [0.0; 3];
            for &w in &ring {
                centroid = add3(centroid, f64v(self.pos[w as usize]));
            }
            centroid = scale3(centroid, 1.0 / ring.len() as f64);
            let disp = sub3(centroid, pv);

            let delta = if self.lock_creases && self.is_crease[v] && self.crease_adj[v].len() == 2 {
                let c0 = f64v(self.pos[self.crease_adj[v][0] as usize]);
                let c1 = f64v(self.pos[self.crease_adj[v][1] as usize]);
                let tangent = sub3(c1, c0);
                let tlen = norm3(tangent);
                if tlen <= 1e-12 {
                    continue;
                }
                let t = scale3(tangent, 1.0 / tlen);
                scale3(t, lambda * dot3(disp, t))
            } else {
                let mut nsum = [0.0; 3];
                for &ti in &self.v2t[v] {
                    let tri = &self.tris[ti as usize];
                    if !tri.alive {
                        continue;
                    }
                    let n = tri_normal(
                        f64v(self.pos[tri.corners[0] as usize]),
                        f64v(self.pos[tri.corners[1] as usize]),
                        f64v(self.pos[tri.corners[2] as usize]),
                    );
                    nsum = add3(nsum, n);
                }
                let nlen = norm3(nsum);
                if nlen <= 1e-18 {
                    continue;
                }
                let nv = scale3(nsum, 1.0 / nlen);
                let tang = sub3(disp, scale3(nv, dot3(disp, nv)));
                scale3(tang, lambda)
            };

            let cand64 = add3(pv, delta);
            let cand32 = [cand64[0] as f32, cand64[1] as f32, cand64[2] as f32];
            if !cand32.iter().all(|c| c.is_finite()) {
                continue;
            }
            if self.move_keeps_fan_valid(v as u32, cand32) {
                self.pos[v] = cand32;
            }
        }
        Ok(())
    }

    fn move_keeps_fan_valid(&self, v: u32, to: [f32; 3]) -> bool {
        let to64 = f64v(to);
        for &ti in &self.v2t[v as usize] {
            let tri = &self.tris[ti as usize];
            if !tri.alive {
                continue;
            }
            let c = tri.corners;
            let before = tri_normal(
                f64v(self.pos[c[0] as usize]),
                f64v(self.pos[c[1] as usize]),
                f64v(self.pos[c[2] as usize]),
            );
            if dot3(before, before) <= 1e-24 {
                continue;
            }
            let after = tri_normal(
                if c[0] == v {
                    to64
                } else {
                    f64v(self.pos[c[0] as usize])
                },
                if c[1] == v {
                    to64
                } else {
                    f64v(self.pos[c[1] as usize])
                },
                if c[2] == v {
                    to64
                } else {
                    f64v(self.pos[c[2] as usize])
                },
            );
            if dot3(after, after) <= 1e-24 || dot3(after, before) <= 0.0 {
                return false;
            }
        }
        true
    }

    fn reproject_phase(
        &mut self,
        ref_mesh: &Mesh,
        ref_bvh: &Bvh,
        ref_sizes: Option<&[f64]>,
    ) -> Result<()> {
        let slots = self.pos.len();
        let mut q_vids: Vec<u32> = Vec::new();
        let mut q_pts: Vec<[f32; 3]> = Vec::new();
        reserve(&mut q_vids, slots)?;
        reserve(&mut q_pts, slots)?;
        for v in 0..slots {
            if !self.vlive[v]
                || self.masked[v]
                || self.is_boundary[v]
                || (self.lock_creases && (self.is_crease[v] || self.is_corner[v]))
            {
                continue;
            }
            push(&mut q_vids, v as u32)?;
            push(&mut q_pts, self.pos[v])?;
        }
        if q_vids.is_empty() {
            return Ok(());
        }
        let mut hits: Vec<Option<ClosestHit>> = Vec::new();
        reserve(&mut hits, q_vids.len())?;
        hits.resize(q_vids.len(), None);
        closest_point_many(
            ref_mesh,
            ref_bvh,
            &q_pts,
            &ClosestOptions::default(),
            &mut hits,
        )?;

        for (idx, hit) in hits.into_iter().enumerate() {
            let Some(hit) = hit else {
                continue;
            };
            let v = q_vids[idx];
            if hit.point.iter().all(|c| c.is_finite()) && self.move_keeps_fan_valid(v, hit.point) {
                self.pos[v as usize] = hit.point;
            }
            if let Some(sizes) = ref_sizes
                && let Some(c) = ref_mesh.triangle(hit.triangle)
            {
                let b = hit.barycentric;
                let s = f64::from(b[0]) * sizes[c[0] as usize]
                    + f64::from(b[1]) * sizes[c[1] as usize]
                    + f64::from(b[2]) * sizes[c[2] as usize];
                if s.is_finite() {
                    self.target_size[v as usize] = s.clamp(self.l_min, self.l_max);
                }
            }
        }
        Ok(())
    }
}

fn build_reference_snapshot(mesh: &Mesh) -> Result<(Mesh, Vec<u32>, Vec<[VertexId; 3]>)> {
    let slots = mesh.vertex_slots();
    let mut orig_to_compact = filled(slots, u32::MAX)?;
    let mut compact_positions: Vec<[f32; 3]> = Vec::new();
    reserve(&mut compact_positions, mesh.vertex_count())?;
    let positions = mesh.positions();
    for (v, &p) in positions.iter().enumerate() {
        if mesh.is_vertex_alive(v as VertexId) {
            orig_to_compact[v] = compact_positions.len() as u32;
            push(&mut compact_positions, p)?;
        }
    }
    let mut compact_tris: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut compact_tris, mesh.triangle_count())?;
    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some(c) = mesh.triangle(t) else {
            continue;
        };
        let a = orig_to_compact[c[0] as usize];
        let b = orig_to_compact[c[1] as usize];
        let d = orig_to_compact[c[2] as usize];
        if a != u32::MAX && b != u32::MAX && d != u32::MAX && a != b && b != d && d != a {
            push(&mut compact_tris, [a, b, d])?;
        }
    }
    let mut ref_mesh = Mesh::new();
    {
        let mut rtx = ref_mesh.begin_edit();
        rtx.add_vertices(&compact_positions)?;
        rtx.add_triangles(&compact_tris)?;
        rtx.commit()?;
    }
    Ok((ref_mesh, orig_to_compact, compact_tris))
}

fn compute_adaptive_sizes(
    ref_mesh: &Mesh,
    target_length: f64,
    l_min: f64,
    l_max: f64,
) -> Result<Vec<f64>> {
    let n = ref_mesh.vertex_slots();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut curv = filled(n, 0.0f32)?;
    curvatures(ref_mesh, &mut curv)?;
    let mut mag: Vec<Option<f64>> = filled(n, None)?;
    for (i, &h) in curv.iter().enumerate() {
        if h.is_finite() {
            mag[i] = Some(f64::from(h.abs()));
        }
    }
    // Collect neighbor lists to fill boundary NaNs and gently smooth field.
    let mut adj: Vec<Vec<u32>> = Vec::new();
    reserve(&mut adj, n)?;
    adj.resize_with(n, Vec::new);
    for tri in ref_mesh.triangles() {
        for k in 0..3 {
            let u = tri[k];
            let v = tri[(k + 1) % 3];
            push(&mut adj[u as usize], v)?;
            push(&mut adj[v as usize], u)?;
        }
    }
    for _ in 0..2 {
        let prev = mag.clone();
        for v in 0..n {
            if prev[v].is_none() {
                let mut sum = 0.0;
                let mut cnt = 0usize;
                for &w in &adj[v] {
                    if let Some(val) = prev[w as usize] {
                        sum += val;
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    mag[v] = Some(sum / cnt as f64);
                }
            }
        }
    }
    let mut resolved = filled(n, 0.0f64)?;
    for v in 0..n {
        resolved[v] = mag[v].unwrap_or(0.0);
    }
    // 1-ring smoothing pass to damp noisy discrete curvature spikes.
    let mut smoothed = filled(n, 0.0f64)?;
    for v in 0..n {
        if adj[v].is_empty() {
            smoothed[v] = resolved[v];
        } else {
            let avg: f64 =
                adj[v].iter().map(|&w| resolved[w as usize]).sum::<f64>() / adj[v].len() as f64;
            smoothed[v] = 0.5 * resolved[v] + 0.5 * avg;
        }
    }

    let mut sorted = smoothed.clone();
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    let k_min = sorted[0];
    let k_max = sorted[sorted.len() - 1];
    let fallback = target_length.clamp(l_min, l_max);
    if !k_min.is_finite()
        || !k_max.is_finite()
        || (k_max - k_min) <= 1e-6 * k_max.max(1.0)
        || (l_max - l_min) <= 1e-12
    {
        return filled(n, fallback);
    }
    let k_med = sorted[sorted.len() / 2].max(0.05 * k_max).max(1e-9);
    let mut sizes = filled(n, fallback)?;
    for (v, &k) in smoothed.iter().enumerate() {
        let ratio = k_med / k.max(0.01 * k_med);
        sizes[v] = (target_length * ratio.sqrt()).clamp(l_min, l_max);
    }
    Ok(sizes)
}

/// Remeshes the entire surface of a mesh inside `tx` using Botsch-Kobbelt
/// isotropic or curvature-adaptive iterative remeshing.
///
/// Drops [`UV_CHANNEL`](fast_mesh_store::UV_CHANNEL) undoably if present and
/// at least one iteration runs.
///
/// # Errors
/// [`OpsError::InvalidInput`] for non-positive/non-finite sizing or invalid angles/strength;
/// [`OpsError::Store`] or [`OpsError::Bvh`] / [`OpsError::Query`] / [`OpsError::Fields`] on failure.
pub fn remesh_surface(
    tx: &mut Transaction<'_>,
    options: &SurfaceRemeshOptions,
) -> Result<SurfaceRemeshReport> {
    remesh_surface_with_mask(tx, options, &[])
}

/// Like [`remesh_surface`], while locking all vertex ids in `locked_vertices`
/// from moving, smoothing, projecting, or being removed.
///
/// # Errors
/// Same as [`remesh_surface`], plus [`OpsError::InvalidInput`] if any vertex id
/// in `locked_vertices` is dead or out of range.
pub fn remesh_surface_with_mask(
    tx: &mut Transaction<'_>,
    options: &SurfaceRemeshOptions,
    locked_vertices: &[VertexId],
) -> Result<SurfaceRemeshReport> {
    let mesh = tx.mesh();
    let (l_min, l_max) = validate_options(mesh, options, locked_vertices)?;

    let mut initial_tris: Vec<WorkTri> = Vec::new();
    reserve(&mut initial_tris, mesh.triangle_slots())?;
    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some(corners) = mesh.triangle(t) else {
            continue;
        };
        if corners[0] == corners[1] || corners[1] == corners[2] || corners[2] == corners[0] {
            continue;
        }
        push(
            &mut initial_tris,
            WorkTri {
                orig_id: Some(t),
                corners,
                alive: true,
            },
        )?;
    }
    let initial_triangles = initial_tris.len();
    let iters = options.iterations.min(MAX_SURFACE_REMESH_ITERATIONS);
    if initial_triangles == 0 || iters == 0 {
        return Ok(SurfaceRemeshReport {
            initial_triangles,
            final_triangles: initial_triangles,
            splits: 0,
            collapses: 0,
            flips: 0,
            iterations_run: 0,
            uvs_dropped: false,
        });
    }

    let had_uvs = mesh.uvs().is_some();
    let orig_base = mesh.positions();
    let orig_slots = orig_base.len();
    let mut pos: Vec<[f32; 3]> = Vec::new();
    reserve(&mut pos, orig_slots)?;
    pos.extend_from_slice(orig_base);
    let mut vlive = filled(orig_slots, false)?;
    for (v, alive) in vlive.iter_mut().enumerate() {
        *alive = mesh.is_vertex_alive(v as VertexId);
    }
    let was_orig_vert = vlive.clone();
    let mut masked = filled(orig_slots, false)?;
    for &v in locked_vertices {
        masked[v as usize] = true;
    }

    struct RefBundle {
        mesh: Mesh,
        bvh: Bvh,
        orig_to_ref: Vec<u32>,
        sizes: Option<Vec<f64>>,
    }

    let need_ref = options.project_to_surface || options.mode == SurfaceRemeshMode::Adaptive;
    let ref_bundle = if need_ref {
        let (mesh, orig_to_ref, _) = build_reference_snapshot(mesh)?;
        let bvh = Bvh::build(&mesh, &BuildOptions::default())?;
        let sizes = if options.mode == SurfaceRemeshMode::Adaptive {
            Some(compute_adaptive_sizes(
                &mesh,
                f64::from(options.target_edge_length),
                l_min,
                l_max,
            )?)
        } else {
            None
        };
        Some(RefBundle {
            mesh,
            bvh,
            orig_to_ref,
            sizes,
        })
    } else {
        None
    };

    let init_size = f64::from(options.target_edge_length).clamp(l_min, l_max);
    let mut target_size = filled(orig_slots, init_size)?;
    if let Some(ref_surf) = &ref_bundle
        && let Some(sizes) = &ref_surf.sizes
    {
        for (v, &compact) in ref_surf.orig_to_ref.iter().enumerate() {
            if compact != u32::MAX {
                target_size[v] = sizes[compact as usize];
            }
        }
    }

    let mut v2t: Vec<Vec<u32>> = Vec::new();
    reserve(&mut v2t, orig_slots)?;
    v2t.resize_with(orig_slots, Vec::new);
    for (ti, tri) in initial_tris.iter().enumerate() {
        for &v in &tri.corners {
            push(&mut v2t[v as usize], ti as u32)?;
        }
    }
    for list in &mut v2t {
        list.sort_unstable();
        list.dedup();
    }

    let mut state = RemeshState {
        pos,
        tris: initial_tris,
        v2t,
        vlive,
        was_orig_vert,
        target_size,
        masked,
        is_boundary: Vec::new(),
        is_crease: Vec::new(),
        is_corner: Vec::new(),
        crease_adj: Vec::new(),
        cos_crease: f64::from(options.crease_angle_deg).to_radians().cos(),
        lock_creases: options.lock_creases,
        lock_boundaries: options.lock_boundaries,
        l_min,
        l_max,
    };

    let mut splits = 0usize;
    let mut collapses = 0usize;
    let mut flips = 0usize;
    let lambda = f64::from(options.smooth_strength);

    for _ in 0..iters {
        state.tag_features()?;
        splits += state.split_phase()?;
        state.tag_features()?;
        collapses += state.collapse_phase()?;
        state.tag_features()?;
        flips += state.flip_phase()?;
        state.tag_features()?;
        state.smooth_phase(lambda)?;
        if options.project_to_surface
            && let Some(ref_surf) = &ref_bundle
        {
            state.reproject_phase(&ref_surf.mesh, &ref_surf.bvh, ref_surf.sizes.as_deref())?;
        }
    }

    // Plan and pre-reserve all transaction mutations before writing to `tx`.
    let mut set_p_ids: Vec<VertexId> = Vec::new();
    let mut set_p_pos: Vec<[f32; 3]> = Vec::new();
    let mut del_v_ids: Vec<VertexId> = Vec::new();
    let mut new_v_pos: Vec<[f32; 3]> = Vec::new();
    let mut v_to_final = filled(state.pos.len(), VertexId::MAX)?;

    for v in 0..orig_slots {
        if state.was_orig_vert[v] {
            if state.vlive[v] {
                v_to_final[v] = v as VertexId;
                if state.pos[v] != orig_base[v] {
                    push(&mut set_p_ids, v as VertexId)?;
                    push(&mut set_p_pos, state.pos[v])?;
                }
            } else {
                push(&mut del_v_ids, v as VertexId)?;
            }
        }
    }
    let mut new_v_slots: Vec<usize> = Vec::new();
    for v in orig_slots..state.pos.len() {
        if state.vlive[v] {
            push(&mut new_v_slots, v)?;
            push(&mut new_v_pos, state.pos[v])?;
        }
    }

    let mut set_t_ids: Vec<TriangleId> = Vec::new();
    let mut set_t_corners: Vec<[VertexId; 3]> = Vec::new();
    let mut del_t_ids: Vec<TriangleId> = Vec::new();
    let mut add_t_corners: Vec<[VertexId; 3]> = Vec::new();
    let mut add_count = 0usize;

    for t in &state.tris {
        if let Some(orig_id) = t.orig_id {
            if !t.alive {
                push(&mut del_t_ids, orig_id)?;
            } else if t.corners.iter().all(|&v| (v as usize) < orig_slots) {
                let current = mesh
                    .triangle(orig_id)
                    .expect("remesh_surface: working triangle must exist");
                if current != t.corners {
                    push(&mut set_t_ids, orig_id)?;
                    push(&mut set_t_corners, t.corners)?;
                }
            } else {
                // Original triangle now points at newly allocated vertices: replace
                // cleanly after adding those vertices.
                push(&mut del_t_ids, orig_id)?;
                add_count += 1;
            }
        } else if t.alive {
            add_count += 1;
        }
    }
    reserve(&mut add_t_corners, add_count)?;
    let final_triangles = state.tris.iter().filter(|t| t.alive).count();

    if had_uvs {
        tx.drop_channel(UV_CHANNEL)?;
    }
    let new_vids = tx.add_vertices(&new_v_pos)?;
    for (&slot, &vid) in new_v_slots.iter().zip(&new_vids) {
        v_to_final[slot] = vid;
    }

    for t in &state.tris {
        if !t.alive {
            continue;
        }
        let needs_add = match t.orig_id {
            None => true,
            Some(_) => t.corners.iter().any(|&v| (v as usize) >= orig_slots),
        };
        if needs_add {
            let mapped = t.corners.map(|v| v_to_final[v as usize]);
            add_t_corners.push(mapped);
        }
    }

    if !set_p_ids.is_empty() {
        tx.set_positions(&set_p_ids, &set_p_pos)?;
    }
    if !set_t_ids.is_empty() {
        tx.set_triangles(&set_t_ids, &set_t_corners)?;
    }
    if !del_t_ids.is_empty() {
        tx.remove_triangles(&del_t_ids)?;
    }
    if !add_t_corners.is_empty() {
        let _ = tx.add_triangles(&add_t_corners)?;
    }
    if !del_v_ids.is_empty() {
        tx.remove_vertices(&del_v_ids)?;
    }

    Ok(SurfaceRemeshReport {
        initial_triangles,
        final_triangles,
        splits,
        collapses,
        flips,
        iterations_run: iters,
        uvs_dropped: had_uvs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AnalyzeOptions;
    use crate::test_util::{box_mesh, edit, grid, sweep, uv_sphere};

    fn triangle_count(m: &Mesh) -> usize {
        (0..m.triangle_slots() as u32)
            .filter(|&t| m.triangle(t).is_some())
            .count()
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let mut m = box_mesh([0.0; 3], [1.0; 3]);
        let bad = [
            SurfaceRemeshOptions::builder()
                .target_edge_length(0.0)
                .build(),
            SurfaceRemeshOptions::builder()
                .target_edge_length(-1.0)
                .build(),
            SurfaceRemeshOptions::builder()
                .target_edge_length(f32::NAN)
                .build(),
            SurfaceRemeshOptions::builder()
                .target_edge_length(0.2)
                .min_edge_length(0.4)
                .max_edge_length(0.3)
                .build(),
            SurfaceRemeshOptions::builder().smooth_strength(1.5).build(),
            SurfaceRemeshOptions::builder()
                .crease_angle_deg(-1.0)
                .build(),
        ];
        for o in &bad {
            let mut tx = m.begin_edit();
            assert!(matches!(
                remesh_surface(&mut tx, o),
                Err(OpsError::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn isotropic_refines_coarse_box_and_keeps_sharp_bounds() {
        let mut m = box_mesh([0.0; 3], [2.0; 3]);
        let opts = SurfaceRemeshOptions::builder()
            .mode(SurfaceRemeshMode::Isotropic)
            .target_edge_length(0.5)
            .iterations(4)
            .build();
        let (rep, empty) = edit(&mut m, |tx| remesh_surface(tx, &opts));
        assert!(!empty);
        assert!(rep.splits > 0);
        assert_eq!(rep.iterations_run, 4);
        assert!(triangle_count(&m) > 12);

        let a = crate::analyze(&m, &AnalyzeOptions::default()).unwrap();
        assert_eq!(a.non_manifold_edges, 0);
        assert!(a.non_manifold_vertices.is_empty());
        assert_eq!(a.orientation_conflicts, 0);
        assert!(a.loops.is_empty());

        for v in 0..m.vertex_slots() as VertexId {
            if m.is_vertex_alive(v) {
                let p = m.position(v).unwrap();
                for k in 0..3 {
                    assert!(p[k] >= -1e-4 && p[k] <= 2.0 + 1e-4, "out of bounds: {p:?}");
                }
            }
        }
    }

    #[test]
    fn isotropic_open_grid_locks_boundaries() {
        let mut m = grid(4);
        let boundary_before: Vec<[f32; 3]> = crate::boundary_loops(&m)
            .unwrap()
            .into_iter()
            .flat_map(|l| l.vertices.into_iter().map(|v| m.position(v).unwrap()))
            .collect();
        let opts = SurfaceRemeshOptions::builder()
            .target_edge_length(0.5)
            .iterations(3)
            .lock_boundaries(true)
            .build();
        let (rep, _) = edit(&mut m, |tx| remesh_surface(tx, &opts));
        assert!(rep.splits > 0);
        let a = crate::analyze(&m, &AnalyzeOptions::default()).unwrap();
        assert_eq!(a.non_manifold_edges, 0);
        assert_eq!(a.loops.len(), 1);
        for p in boundary_before {
            assert!(
                (0..m.vertex_slots() as VertexId)
                    .filter(|&v| m.is_vertex_alive(v))
                    .any(|v| {
                        let q = m.position(v).unwrap();
                        (p[0] - q[0]).abs() < 1e-5
                            && (p[1] - q[1]).abs() < 1e-5
                            && (p[2] - q[2]).abs() < 1e-5
                    }),
                "locked corner missing: {p:?}"
            );
        }
    }

    #[test]
    fn adaptive_surface_remesh_succeeds_and_stays_manifold() {
        let mut m = uv_sphere(12, 24, 1.0);
        // Stretch into an ellipsoid so curvature varies significantly.
        let ids: Vec<VertexId> = (0..m.vertex_slots() as VertexId).collect();
        let stretched: Vec<[f32; 3]> = m
            .positions()
            .iter()
            .map(|p| [p[0] * 2.5, p[1], p[2] * 0.5])
            .collect();
        {
            let mut tx = m.begin_edit();
            tx.set_positions(&ids, &stretched).unwrap();
            tx.commit().unwrap();
        }
        let opts = SurfaceRemeshOptions::builder()
            .mode(SurfaceRemeshMode::Adaptive)
            .target_edge_length(0.25)
            .min_edge_length(0.1)
            .max_edge_length(0.45)
            .iterations(4)
            .lock_creases(false)
            .build();
        let (rep, empty) = edit(&mut m, |tx| remesh_surface(tx, &opts));
        assert!(!empty);
        assert_eq!(rep.iterations_run, 4);
        assert!(rep.splits + rep.collapses + rep.flips > 0);
        let a = crate::analyze(&m, &AnalyzeOptions::default()).unwrap();
        assert_eq!(a.non_manifold_edges, 0);
        assert!(a.non_manifold_vertices.is_empty());
        assert_eq!(a.orientation_conflicts, 0);
    }

    #[test]
    fn surface_remesh_sweep_keeps_atomicity() {
        let n = sweep(box_mesh([0.0; 3], [1.0; 3]), |tx| {
            remesh_surface(
                tx,
                &SurfaceRemeshOptions::builder()
                    .target_edge_length(0.5)
                    .iterations(2)
                    .build(),
            )?;
            Ok(())
        });
        assert!(n > 0);
    }
}
