//! Region remeshing: matching a brush region's density to a target edge
//! length.
//!
//! [`remesh`] runs rounds of midpoint splits (Task 1; collapse and flip
//! passes land in Task 2) over the edges under a [`Region`], until a round
//! changes nothing or `max_rounds` runs out, and commits through one
//! [`Transaction`](fast_mesh_store::Transaction). New vertices sit at edge
//! midpoints; existing positions are never modified.

use fast_mesh_store::{Mesh, Transaction, TriangleId, VertexId};

use crate::{OpsError, Result, push, reserve};

/// Edges longer than this multiple of the target length are split.
pub const SPLIT_FACTOR: f64 = 4.0 / 3.0;
/// Edges shorter than this multiple of the target length collapse (Task 2).
pub const COLLAPSE_FACTOR: f64 = 4.0 / 5.0;
/// Default cap on remesh rounds.
pub const REMESH_MAX_ROUNDS: u32 = 10;

/// A spherical region of influence. An edge takes part when its midpoint
/// lies in the strict interior; the boundary itself is inert.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Region {
    /// Center of the region in mesh units.
    pub center: [f32; 3],
    /// Radius of influence in mesh units. Must be positive and finite.
    pub radius: f32,
}

impl Default for Region {
    fn default() -> Self {
        Region {
            center: [0.0; 3],
            radius: 1.0,
        }
    }
}

/// Options for [`remesh`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RemeshOptions {
    /// The region whose edges are remeshed.
    pub region: Region,
    /// Target edge length in mesh units. `None` resolves to the mesh-wide
    /// mean live-edge length; non-finite or non-positive values fall back
    /// to the mesh mean as well.
    pub target_length: Option<f32>,
    /// Maximum number of split/collapse/flip rounds.
    pub max_rounds: u32,
}

impl Default for RemeshOptions {
    fn default() -> Self {
        RemeshOptions {
            region: Region::default(),
            target_length: None,
            max_rounds: REMESH_MAX_ROUNDS,
        }
    }
}

impl RemeshOptions {
    /// A builder for [`RemeshOptions`]: the default region until set.
    pub fn builder() -> RemeshOptionsBuilder {
        RemeshOptionsBuilder::default()
    }
}

/// Builder for [`RemeshOptions`].
#[derive(Debug, Clone, Default)]
pub struct RemeshOptionsBuilder {
    options: RemeshOptions,
}

impl RemeshOptionsBuilder {
    /// Center of the region in mesh units.
    pub fn center(mut self, center: [f32; 3]) -> Self {
        self.options.region.center = center;
        self
    }

    /// Radius of influence in mesh units.
    pub fn radius(mut self, radius: f32) -> Self {
        self.options.region.radius = radius;
        self
    }

    /// Target edge length in mesh units.
    pub fn target_length(mut self, length: f32) -> Self {
        self.options.target_length = Some(length);
        self
    }

    /// Maximum number of rounds.
    pub fn max_rounds(mut self, rounds: u32) -> Self {
        self.options.max_rounds = rounds;
        self
    }

    /// The options as set. Region validity is checked by [`remesh`].
    pub fn build(self) -> RemeshOptions {
        self.options
    }
}

/// Outcome of [`remesh`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemeshReport {
    /// Edge splits performed.
    pub splits: usize,
    /// Edge collapses performed (Task 2).
    pub collapses: usize,
    /// Edge flips performed (Task 2).
    pub flips: usize,
    /// Rounds that changed the mesh.
    pub rounds: usize,
}

impl RemeshReport {
    fn quiet() -> Self {
        RemeshReport {
            splits: 0,
            collapses: 0,
            flips: 0,
            rounds: 0,
        }
    }
}

/// Remeshes the edges inside `options.region` toward the target length.
///
/// Each round splits the edges whose midpoint lies in the region and whose
/// length exceeds the target by [`SPLIT_FACTOR`]. Splitting an edge inserts
/// one vertex at its midpoint, shared by the one or two triangles on either
/// side, so neighbors stay stitched. The whole plan is computed on working
/// copies first and committed through `tx` at the end, which keeps the call
/// atomic under sweep: either every round lands or nothing does.
pub fn remesh(tx: &mut Transaction, options: &RemeshOptions) -> Result<RemeshReport> {
    let region = &options.region;
    if !region.radius.is_finite() || region.radius <= 0.0 {
        return Err(OpsError::InvalidInput(
            "remesh region radius must be positive and finite",
        ));
    }
    if region.center.iter().any(|c| !c.is_finite()) {
        return Err(OpsError::InvalidInput(
            "remesh region center must be finite",
        ));
    }
    let mesh = tx.mesh();
    if !has_member(mesh, region) {
        return Ok(RemeshReport::quiet());
    }
    let sigma = match options.target_length {
        Some(t) if t.is_finite() && t > 0.0 => f64::from(t),
        _ => match global_mean_edge_length(mesh)? {
            Some(mean) => mean,
            None => return Ok(RemeshReport::quiet()),
        },
    };
    let base = mesh.positions();
    let mut positions: Vec<[f32; 3]> = Vec::new();
    reserve(&mut positions, base.len())?;
    positions.extend_from_slice(base);
    let slots = positions.len();
    let mut tris: Vec<WorkTri> = Vec::new();
    reserve(&mut tris, mesh.triangle_slots())?;
    for t in 0..mesh.triangle_slots() as u32 {
        if let Some(corners) = mesh.triangle(t) {
            push(
                &mut tris,
                WorkTri {
                    id: Some(t),
                    corners,
                },
            )?;
        }
    }
    let mut added: Vec<NewVert> = Vec::new();
    let mut report = RemeshReport::quiet();
    for _ in 0..options.max_rounds {
        let splits = split_round(&mut positions, &mut tris, &mut added, region, sigma)?;
        // Task 2 plugs collapse and flip passes in here; the loop breaks
        // once a whole round changes nothing.
        if splits == 0 {
            break;
        }
        report.splits += splits;
        report.rounds += 1;
    }
    if report.splits + report.collapses + report.flips == 0 {
        return Ok(report);
    }
    commit(tx, slots, &tris, &added)?;
    Ok(report)
}

/// A working triangle: `id` is `None` for triangles created by a split.
struct WorkTri {
    id: Option<TriangleId>,
    corners: [u32; 3],
}

/// A midpoint vertex waiting for commit. `parents` weights the new vertex
/// over live base vertices so channels interpolate; positions are exact.
struct NewVert {
    pos: [f32; 3],
    parents: Vec<(VertexId, f32)>,
}

fn in_region(p: &[f32; 3], region: &Region) -> bool {
    let dx = f64::from(p[0]) - f64::from(region.center[0]);
    let dy = f64::from(p[1]) - f64::from(region.center[1]);
    let dz = f64::from(p[2]) - f64::from(region.center[2]);
    let t = dx.hypot(dy).hypot(dz) / f64::from(region.radius);
    t.is_finite() && t < 1.0
}

fn has_member(mesh: &Mesh, region: &Region) -> bool {
    mesh.positions().iter().any(|p| in_region(p, region))
}

fn dist(a: &[f32; 3], b: &[f32; 3]) -> f64 {
    let dx = f64::from(a[0]) - f64::from(b[0]);
    let dy = f64::from(a[1]) - f64::from(b[1]);
    let dz = f64::from(a[2]) - f64::from(b[2]);
    dx.hypot(dy).hypot(dz)
}

fn midpoint(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
        (a[2] + b[2]) / 2.0,
    ]
}

fn global_mean_edge_length(mesh: &Mesh) -> Result<Option<f64>> {
    let mut edges: Vec<(u32, u32)> = Vec::new();
    reserve(&mut edges, 3 * mesh.triangle_slots())?;
    for t in 0..mesh.triangle_slots() as u32 {
        let corners = match mesh.triangle(t) {
            Some(c) => c,
            None => continue,
        };
        if corners[0] == corners[1] || corners[1] == corners[2] || corners[2] == corners[0] {
            continue;
        }
        for k in 0..3 {
            let (a, b) = (corners[k], corners[(k + 1) % 3]);
            push(&mut edges, (a.min(b), a.max(b)))?;
        }
    }
    edges.sort_unstable();
    edges.dedup();
    if edges.is_empty() {
        return Ok(None);
    }
    let positions = mesh.positions();
    let mut sum = 0.0;
    for &(a, b) in &edges {
        sum += dist(&positions[a as usize], &positions[b as usize]);
    }
    let mean = sum / edges.len() as f64;
    Ok((mean.is_finite() && mean > 0.0).then_some(mean))
}

/// Splits every overlong in-region edge once. Returns the split count.
/// Each unique edge is visited once per round; an edge is skipped when a
/// split made earlier in the same round already consumed it.
fn split_round(
    positions: &mut Vec<[f32; 3]>,
    tris: &mut Vec<WorkTri>,
    added: &mut Vec<NewVert>,
    region: &Region,
    sigma: f64,
) -> Result<usize> {
    // `positions` and `added` grow in lockstep, one entry per split, so
    // their length gap is the base slot count for the whole round.
    let slots = positions.len() - added.len();
    let mut pairs: Vec<((u32, u32), usize)> = Vec::new();
    reserve(&mut pairs, 3 * tris.len())?;
    for (i, tri) in tris.iter().enumerate() {
        let c = tri.corners;
        if c[0] == c[1] || c[1] == c[2] || c[2] == c[0] {
            continue;
        }
        for k in 0..3 {
            let (a, b) = (c[k], c[(k + 1) % 3]);
            push(&mut pairs, ((a.min(b), a.max(b)), i))?;
        }
    }
    pairs.sort_unstable();
    let mut splits = 0;
    let mut start = 0;
    while start < pairs.len() {
        let mut end = start + 1;
        while end < pairs.len() && pairs[end].0 == pairs[start].0 {
            end += 1;
        }
        let (a, b) = pairs[start].0;
        let mid = midpoint(&positions[a as usize], &positions[b as usize]);
        if in_region(&mid, region)
            && dist(&positions[a as usize], &positions[b as usize]) > sigma * SPLIT_FACTOR
            && pairs[start..end]
                .iter()
                .all(|&(_, i)| carries(&tris[i].corners, a, b))
        {
            let m = positions.len() as u32;
            let parents = blend_parents(a, b, slots, added)?;
            push(positions, mid)?;
            push(added, NewVert { pos: mid, parents })?;
            for &(_, i) in &pairs[start..end] {
                let corners = tris[i].corners;
                let apex = if corners[0] != a && corners[0] != b {
                    corners[0]
                } else if corners[1] != a && corners[1] != b {
                    corners[1]
                } else {
                    corners[2]
                };
                tris[i].corners = [a, m, apex];
                push(
                    tris,
                    WorkTri {
                        id: None,
                        corners: [m, b, apex],
                    },
                )?;
            }
            splits += 1;
        }
        start = end;
    }
    Ok(splits)
}

fn carries(corners: &[u32; 3], a: u32, b: u32) -> bool {
    corners.contains(&a) && corners.contains(&b)
}

/// Weights a midpoint over live base vertices. Endpoints created by earlier
/// rounds contribute their own weights halved, merged by vertex id, so the
/// weights always describe the exact midpoint over the base mesh.
fn blend_parents(a: u32, b: u32, slots: usize, added: &[NewVert]) -> Result<Vec<(VertexId, f32)>> {
    let mut entries: Vec<(u32, f32)> = Vec::new();
    for v in [a, b] {
        if (v as usize) < slots {
            push(&mut entries, (v, 0.5))?;
        } else {
            for &(id, w) in &added[(v as usize) - slots].parents {
                push(&mut entries, (id, 0.5 * w))?;
            }
        }
    }
    entries.sort_unstable_by_key(|x| x.0);
    let mut parents: Vec<(VertexId, f32)> = Vec::new();
    reserve(&mut parents, entries.len())?;
    for (id, w) in entries {
        match parents.last_mut() {
            Some(last) if last.0 == id => last.1 += w,
            _ => push(&mut parents, (id, w))?,
        }
    }
    Ok(parents)
}

fn commit(tx: &mut Transaction, slots: usize, tris: &[WorkTri], added: &[NewVert]) -> Result<()> {
    // Everything is reserved before the first write: past this point only
    // infallible pushes into reserved capacity remain, so any allocation
    // failure still leaves an empty patch.
    let mut n_old = 0;
    let mut n_new = 0;
    for tri in tris {
        if tri.id.is_some() {
            n_old += 1;
        } else {
            n_new += 1;
        }
    }
    let mut ids: Vec<VertexId> = Vec::new();
    reserve(&mut ids, added.len())?;
    let mut set_ids: Vec<TriangleId> = Vec::new();
    reserve(&mut set_ids, n_old)?;
    let mut set_tris: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut set_tris, n_old)?;
    let mut new_tris: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut new_tris, n_new)?;
    for nv in added {
        ids.push(tx.add_vertex_with_parents(nv.pos, &nv.parents)?);
    }
    let resolve = |v: u32| -> VertexId {
        let i = v as usize;
        if i < slots { v } else { ids[i - slots] }
    };
    let mesh = tx.mesh();
    for tri in tris {
        let current = [
            resolve(tri.corners[0]),
            resolve(tri.corners[1]),
            resolve(tri.corners[2]),
        ];
        match tri.id {
            Some(id) => {
                let original = mesh.triangle(id).expect("remesh: working id must exist");
                if current != original {
                    set_ids.push(id);
                    set_tris.push(current);
                }
            }
            None => new_tris.push(current),
        }
    }
    tx.set_triangles(&set_ids, &set_tris)?;
    tx.add_triangles(&new_tris)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::{Mesh, Transaction, VertexId};

    use super::*;
    use crate::OpsError;
    use crate::test_util::{edit, mesh_of, parts, sweep};
    use crate::topology::{EdgeMap, point, sub};

    fn quad_4x1() -> Mesh {
        mesh_of(
            &[0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            &[[0, 1, 2], [0, 2, 3]],
        )
    }

    fn edge_lengths(mesh: &Mesh) -> Vec<f64> {
        let edges = EdgeMap::build(mesh).unwrap();
        edges
            .edges()
            .map(|((a, b), _)| {
                let d = sub(point(mesh, a), point(mesh, b));
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
            .collect()
    }

    fn all_region() -> RemeshOptions {
        RemeshOptions::builder()
            .center([2.0, 0.5, 0.0])
            .radius(10.0)
            .target_length(1.0)
            .build()
    }

    #[test]
    fn split_refines_sparse_quad_to_target() {
        let mut mesh = quad_4x1();
        let before = parts(&mesh);
        let (report, _) = edit(&mut mesh, |tx| remesh(tx, &all_region()));
        assert!(report.splits > 0, "expected splits, got {report:?}");
        for len in edge_lengths(&mesh) {
            assert!(len <= 4.0 / 3.0 + 1e-4, "long edge left: {len}");
        }
        let (pos_after, tris_after) = parts(&mesh);
        assert_eq!(&pos_after[..12], &before.0[..12]);
        assert!(tris_after.len() > before.1.len());
    }

    #[test]
    fn split_empty_region_and_coarse_target() {
        let mut mesh = quad_4x1();
        let before = parts(&mesh);
        let far = RemeshOptions::builder()
            .center([100.0, 0.0, 0.0])
            .radius(1.0)
            .target_length(1.0)
            .build();
        let (report, empty) = edit(&mut mesh, |tx| remesh(tx, &far));
        assert_eq!((report.splits, report.collapses, report.flips), (0, 0, 0));
        assert!(empty);
        assert_eq!(parts(&mesh), before);
        let mut mesh = quad_4x1();
        let coarse = RemeshOptions::builder()
            .center([2.0, 0.5, 0.0])
            .radius(10.0)
            .target_length(100.0)
            .build();
        let (report, _) = edit(&mut mesh, |tx| remesh(tx, &coarse));
        assert_eq!(report.splits, 0);
    }

    #[test]
    fn split_rejects_invalid_region() {
        let mut mesh = quad_4x1();
        let zero = RemeshOptions::builder()
            .center([2.0, 0.5, 0.0])
            .radius(0.0)
            .target_length(1.0)
            .build();
        {
            let mut tx = mesh.begin_edit();
            assert_eq!(
                remesh(&mut tx, &zero),
                Err(OpsError::InvalidInput(
                    "remesh region radius must be positive and finite"
                ))
            );
        }
        let nan = RemeshOptions::builder()
            .center([f32::NAN, 0.0, 0.0])
            .radius(10.0)
            .target_length(1.0)
            .build();
        {
            let mut tx = mesh.begin_edit();
            assert_eq!(
                remesh(&mut tx, &nan),
                Err(OpsError::InvalidInput(
                    "remesh region center must be finite"
                ))
            );
        }
    }

    #[test]
    fn split_keeps_boundary_vertices_live() {
        let mut mesh = quad_4x1();
        edit(&mut mesh, |tx| remesh(tx, &all_region()));
        for v in 0..4u32 {
            assert!(mesh.is_vertex_alive(v as VertexId));
        }
    }

    #[test]
    fn split_sweep_keeps_atomicity() {
        let n = sweep(quad_4x1(), |tx: &mut Transaction| {
            remesh(tx, &all_region())?;
            Ok(())
        });
        assert!(n > 0);
    }
}
