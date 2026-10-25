//! Region remeshing: matching a brush region's density to a target edge
//! length.
//!
//! [`remesh`] runs rounds of midpoint splits, midpoint collapses, and
//! Delaunay-style flips over the edges under a [`Region`], until a round
//! changes nothing or `max_rounds` runs out, and commits through one
//! [`Transaction`](fast_mesh_store::Transaction). New vertices sit at edge
//! midpoints; existing positions are never modified.

use std::cmp::Ordering;

use fast_mesh_store::{Mesh, Transaction, TriangleId, VertexId};

use crate::{OpsError, Result, push, reserve};

/// Edges longer than this multiple of the target length are split.
pub const SPLIT_FACTOR: f64 = 4.0 / 3.0;
/// Edges shorter than this multiple of the target length collapse.
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RemeshReport {
    /// Edge splits performed.
    pub splits: usize,
    /// Edge collapses performed.
    pub collapses: usize,
    /// Edge flips performed.
    pub flips: usize,
    /// Rounds that changed the mesh.
    pub rounds: usize,
}

/// Remeshes the edges inside `options.region` toward the target length.
///
/// Each round splits the edges whose midpoint lies in the region and whose
/// length exceeds the target by [`SPLIT_FACTOR`], collapses interior edges
/// shorter than the target by [`COLLAPSE_FACTOR`], and flips interior edges
/// whose quad gains a larger minimum angle. Splitting an edge inserts one
/// vertex at its midpoint, shared by the triangles on either side, so
/// neighbors stay stitched; collapsing an edge merges its endpoints into one
/// midpoint vertex. Boundary edges never collapse, and any pass that would
/// fold the surface is skipped. The whole plan is computed on working copies
/// first and committed through `tx` at the end, which keeps the call atomic
/// under sweep: either every round lands or nothing does.
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
        return Ok(RemeshReport::default());
    }
    let sigma = match options.target_length {
        Some(t) if t.is_finite() && t > 0.0 => f64::from(t),
        _ => match global_mean_edge_length(mesh)? {
            Some(mean) => mean,
            None => return Ok(RemeshReport::default()),
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
                    alive: true,
                },
            )?;
        }
    }
    let mut vlive: Vec<bool> = Vec::new();
    reserve(&mut vlive, slots)?;
    vlive.extend(std::iter::repeat_n(true, slots));
    let mut added: Vec<NewVert> = Vec::new();
    let mut report = RemeshReport::default();
    for _ in 0..options.max_rounds {
        let fresh = positions.len();
        let splits = split_round(
            &mut positions,
            &mut vlive,
            &mut tris,
            &mut added,
            region,
            sigma,
        )?;
        let collapses = collapse_round(
            &mut positions,
            &mut vlive,
            &mut tris,
            &mut added,
            region,
            sigma,
            fresh,
        )?;
        let flips = flip_round(&positions, &vlive, &mut tris, region, sigma)?;
        if splits + collapses + flips == 0 {
            break;
        }
        report.splits += splits;
        report.collapses += collapses;
        report.flips += flips;
        report.rounds += 1;
    }
    if report.splits + report.collapses + report.flips == 0 {
        return Ok(report);
    }
    commit(tx, slots, &vlive, &tris, &added)?;
    Ok(report)
}

/// A working triangle: `id` is `None` for triangles created by a split.
struct WorkTri {
    id: Option<TriangleId>,
    corners: [u32; 3],
    alive: bool,
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

/// Whether any live edge midpoint lies in the region. The passes test
/// midpoints, not vertices, so a vertex scan would wrongly early-out a
/// region crossed by a long edge but holding no vertices.
fn has_member(mesh: &Mesh, region: &Region) -> bool {
    let positions = mesh.positions();
    for t in 0..mesh.triangle_slots() as u32 {
        let corners = match mesh.triangle(t) {
            Some(c) => c,
            None => continue,
        };
        if degenerate(&corners) {
            continue;
        }
        for k in 0..3 {
            let mid = midpoint(
                &positions[corners[k] as usize],
                &positions[corners[(k + 1) % 3] as usize],
            );
            if in_region(&mid, region) {
                return true;
            }
        }
    }
    false
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
        if degenerate(&corners) {
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
    vlive: &mut Vec<bool>,
    tris: &mut Vec<WorkTri>,
    added: &mut Vec<NewVert>,
    region: &Region,
    sigma: f64,
) -> Result<usize> {
    // `positions` and `added` grow in lockstep, one entry per split, so
    // their length gap is the base slot count for the whole round.
    let slots = positions.len() - added.len();
    let pairs = snapshot_pairs(tris, vlive)?;
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
            push(vlive, true)?;
            push(added, NewVert { pos: mid, parents })?;
            for &(_, i) in &pairs[start..end] {
                let corners = tris[i].corners;
                let apex = apex_of(&corners, a, b);
                // The children follow the parent's wound order around the
                // edge; fixed-order children would flip every parent that
                // winds b -> a.
                let ab = (corners[0] == a && corners[1] == b)
                    || (corners[1] == a && corners[2] == b)
                    || (corners[2] == a && corners[0] == b);
                let (first, second) = if ab {
                    ([a, m, apex], [m, b, apex])
                } else {
                    ([b, m, apex], [m, a, apex])
                };
                tris[i].corners = first;
                push(
                    tris,
                    WorkTri {
                        id: None,
                        corners: second,
                        alive: true,
                    },
                )?;
            }
            splits += 1;
        }
        start = end;
    }
    Ok(splits)
}

/// Collapses every short in-region interior edge once, merging its endpoints
/// into one midpoint vertex. Edges created earlier in the same round are
/// left alone so the pass cannot undo the split pass behind it. Returns the
/// collapse count.
fn collapse_round(
    positions: &mut Vec<[f32; 3]>,
    vlive: &mut Vec<bool>,
    tris: &mut [WorkTri],
    added: &mut Vec<NewVert>,
    region: &Region,
    sigma: f64,
    fresh: usize,
) -> Result<usize> {
    let slots = positions.len() - added.len();
    let pairs = snapshot_pairs(tris, vlive)?;
    // Vertices touching the boundary (or a non-manifold edge) are locked:
    // collapsing past them would reshape the boundary itself.
    let mut locked: Vec<bool> = Vec::new();
    reserve(&mut locked, vlive.len())?;
    locked.extend(std::iter::repeat_n(false, vlive.len()));
    let mut run = 0;
    while run < pairs.len() {
        let mut end = run + 1;
        while end < pairs.len() && pairs[end].0 == pairs[run].0 {
            end += 1;
        }
        if end - run != 2 {
            locked[pairs[run].0.0 as usize] = true;
            locked[pairs[run].0.1 as usize] = true;
        }
        run = end;
    }
    let mut collapses = 0;
    let mut start = 0;
    while start < pairs.len() {
        let mut end = start + 1;
        while end < pairs.len() && pairs[end].0 == pairs[start].0 {
            end += 1;
        }
        let (a, b) = pairs[start].0;
        if end - start == 2
            && (a as usize) < fresh
            && (b as usize) < fresh
            && vlive[a as usize]
            && vlive[b as usize]
            && !locked[a as usize]
            && !locked[b as usize]
        {
            let mid = midpoint(&positions[a as usize], &positions[b as usize]);
            let short = matches!(
                dist(&positions[a as usize], &positions[b as usize])
                    .partial_cmp(&(sigma * COLLAPSE_FACTOR)),
                Some(Ordering::Less)
            );
            if in_region(&mid, region)
                && short
                && let Some(plan) = validate_collapse(positions, vlive, tris, a, b)?
            {
                let m = positions.len() as u32;
                let parents = blend_parents(a, b, slots, added)?;
                push(positions, plan.mid)?;
                push(vlive, true)?;
                push(
                    added,
                    NewVert {
                        pos: plan.mid,
                        parents,
                    },
                )?;
                tris[plan.owners[0]].alive = false;
                tris[plan.owners[1]].alive = false;
                for i in plan.fan {
                    tris[i].corners = tris[i]
                        .corners
                        .map(|v| if v == a || v == b { m } else { v });
                }
                vlive[a as usize] = false;
                vlive[b as usize] = false;
                collapses += 1;
            }
        }
        start = end;
    }
    Ok(collapses)
}

/// A validated collapse: the two owners to remove and the fan to repoint.
struct CollapsePlan {
    owners: [usize; 2],
    fan: Vec<usize>,
    mid: [f32; 3],
}

/// Validates collapsing `(a, b)` against the current working state: exactly
/// two owners, the link condition (no duplicate triangles after the merge),
/// and no folded fan triangle.
fn validate_collapse(
    positions: &[[f32; 3]],
    vlive: &[bool],
    tris: &[WorkTri],
    a: u32,
    b: u32,
) -> Result<Option<CollapsePlan>> {
    let mut owners: Vec<usize> = Vec::new();
    let mut fan: Vec<usize> = Vec::new();
    let mut na: Vec<u32> = Vec::new();
    let mut nb: Vec<u32> = Vec::new();
    for (i, tri) in tris.iter().enumerate() {
        if !tri.alive {
            continue;
        }
        let c = tri.corners;
        if degenerate(&c) || !vlive[c[0] as usize] || !vlive[c[1] as usize] || !vlive[c[2] as usize]
        {
            // Skipped triangles are never repointed, so the collapse is
            // only safe when none of them touches either endpoint.
            if c.contains(&a) || c.contains(&b) {
                return Ok(None);
            }
            continue;
        }
        let has_a = c.contains(&a);
        let has_b = c.contains(&b);
        if has_a && has_b {
            push(&mut owners, i)?;
        } else if has_a || has_b {
            push(&mut fan, i)?;
            let side = if has_a { &mut na } else { &mut nb };
            for &v in &c {
                if v != a && v != b {
                    push(side, v)?;
                }
            }
        }
    }
    if owners.len() != 2 {
        return Ok(None);
    }
    let c = apex_of(&tris[owners[0]].corners, a, b);
    let d = apex_of(&tris[owners[1]].corners, a, b);
    push(&mut na, c)?;
    push(&mut na, d)?;
    push(&mut nb, c)?;
    push(&mut nb, d)?;
    na.sort_unstable();
    na.dedup();
    nb.sort_unstable();
    nb.dedup();
    let mut common: Vec<u32> = Vec::new();
    for &v in &na {
        if nb.binary_search(&v).is_ok() {
            push(&mut common, v)?;
        }
    }
    if common.as_slice() != [c.min(d), c.max(d)] {
        return Ok(None);
    }
    let mid = midpoint(&positions[a as usize], &positions[b as usize]);
    let m = f64v(&mid);
    for &i in &fan {
        let old = tris[i].corners;
        let new = old.map(|v| {
            if v == a || v == b {
                m
            } else {
                f64v(&positions[v as usize])
            }
        });
        let before = tri_normal(&old, positions);
        let after = cross3(sub3(new[1], new[0]), sub3(new[2], new[0]));
        if !matches!(
            dot3(after, before).partial_cmp(&0.0),
            Some(Ordering::Greater)
        ) {
            return Ok(None);
        }
    }
    Ok(Some(CollapsePlan {
        owners: [owners[0], owners[1]],
        fan,
        mid,
    }))
}

/// Flips every interior edge whose quad gains a larger minimum angle,
/// provided the replacement edge stays within the split band so flips never
/// re-arm the split pass. Returns the flip count.
fn flip_round(
    positions: &[[f32; 3]],
    vlive: &[bool],
    tris: &mut [WorkTri],
    region: &Region,
    sigma: f64,
) -> Result<usize> {
    let pairs = snapshot_pairs(tris, vlive)?;
    let mut unique: Vec<(u32, u32)> = Vec::new();
    reserve(&mut unique, pairs.len())?;
    unique.extend(pairs.iter().map(|&(e, _)| e));
    unique.dedup();
    let mut created: Vec<(u32, u32)> = Vec::new();
    let mut flips = 0;
    let mut start = 0;
    while start < pairs.len() {
        let mut end = start + 1;
        while end < pairs.len() && pairs[end].0 == pairs[start].0 {
            end += 1;
        }
        if end - start == 2 {
            let (a, b) = pairs[start].0;
            let mid = midpoint(&positions[a as usize], &positions[b as usize]);
            let ctx = FlipCtx {
                positions,
                tris: &*tris,
                vlive,
                unique: &unique,
                created: &created,
                sigma,
            };
            if in_region(&mid, region)
                && let Some(plan) = validate_flip(&ctx, a, b, pairs[start].1, pairs[start + 1].1)
            {
                tris[plan.owners[0]].corners = plan.new[0];
                tris[plan.owners[1]].corners = plan.new[1];
                push(&mut created, plan.created)?;
                flips += 1;
            }
        }
        start = end;
    }
    Ok(flips)
}

/// Shared inputs for [`validate_flip`].
struct FlipCtx<'a> {
    positions: &'a [[f32; 3]],
    tris: &'a [WorkTri],
    vlive: &'a [bool],
    unique: &'a [(u32, u32)],
    created: &'a [(u32, u32)],
    sigma: f64,
}

/// A validated flip: the two owners and their replacement corners.
struct FlipPlan {
    owners: [usize; 2],
    new: [[u32; 3]; 2],
    created: (u32, u32),
}

/// Validates flipping `(a, b)` with owners `o1`/`o2` against the current
/// working state: both owners intact, distinct apexes, no duplicate edge,
/// a shorter-or-equal replacement, a strictly larger minimum angle, and no
/// folded triangle.
fn validate_flip(ctx: &FlipCtx, a: u32, b: u32, o1: usize, o2: usize) -> Option<FlipPlan> {
    for &o in &[o1, o2] {
        let t = &ctx.tris[o];
        if !t.alive || degenerate(&t.corners) || !carries(&t.corners, a, b) {
            return None;
        }
        if !ctx.vlive[t.corners[0] as usize]
            || !ctx.vlive[t.corners[1] as usize]
            || !ctx.vlive[t.corners[2] as usize]
        {
            return None;
        }
    }
    let old1 = ctx.tris[o1].corners;
    let old2 = ctx.tris[o2].corners;
    let c = apex_of(&old1, a, b);
    let d = apex_of(&old2, a, b);
    if c == d {
        return None;
    }
    let e = (c.min(d), c.max(d));
    if ctx.unique.binary_search(&e).is_ok() || ctx.created.contains(&e) {
        return None;
    }
    let replacement = dist(&ctx.positions[c as usize], &ctx.positions[d as usize]);
    if !matches!(
        replacement.partial_cmp(&(ctx.sigma * SPLIT_FACTOR)),
        Some(Ordering::Less | Ordering::Equal)
    ) {
        return None;
    }
    let new1 = [c, d, a];
    let new2 = [c, d, b];
    // Area conservation: a valid flip retriangulates the same quad, so the
    // total area only shifts at second order in curvature, while nested or
    // overlapping new triangles inflate it by the double-covered part. 1%
    // separates the two with wide margins either way; sub-1% grazing
    // overlaps are the accepted residual (winding stays consistent).
    let area_old = doubled_area(&old1, ctx.positions) + doubled_area(&old2, ctx.positions);
    let area_new = doubled_area(&new1, ctx.positions) + doubled_area(&new2, ctx.positions);
    if !matches!(
        area_new.partial_cmp(&(area_old * 1.01)),
        Some(Ordering::Less | Ordering::Equal)
    ) {
        return None;
    }
    let old_max = match (
        max_cos_angle(&old1, ctx.positions),
        max_cos_angle(&old2, ctx.positions),
    ) {
        (Some(x), Some(y)) => x.max(y),
        _ => return None,
    };
    let new_max = match (
        max_cos_angle(&new1, ctx.positions),
        max_cos_angle(&new2, ctx.positions),
    ) {
        (Some(x), Some(y)) => x.max(y),
        _ => return None,
    };
    if !matches!(new_max.partial_cmp(&old_max), Some(Ordering::Less)) {
        return None;
    }
    let n1 = orient_like(&new1, &old1, ctx.positions)?;
    let n2 = orient_like(&new2, &old2, ctx.positions)?;
    Some(FlipPlan {
        owners: [o1, o2],
        new: [n1, n2],
        created: e,
    })
}

/// The `new` corners wound to face the same way as `old`, or `None` when
/// neither winding faces it (a fold either way).
fn orient_like(new: &[u32; 3], old: &[u32; 3], positions: &[[f32; 3]]) -> Option<[u32; 3]> {
    let dot = dot3(tri_normal(new, positions), tri_normal(old, positions));
    if dot > 0.0 {
        Some(*new)
    } else if dot < 0.0 {
        Some([new[0], new[2], new[1]])
    } else {
        None
    }
}

/// Every live triangle's edges with their owner index, sorted by edge.
fn snapshot_pairs(tris: &[WorkTri], vlive: &[bool]) -> Result<Vec<((u32, u32), usize)>> {
    let mut pairs: Vec<((u32, u32), usize)> = Vec::new();
    reserve(&mut pairs, 3 * tris.len())?;
    for (i, tri) in tris.iter().enumerate() {
        if !tri.alive || degenerate(&tri.corners) {
            continue;
        }
        let c = tri.corners;
        if !vlive[c[0] as usize] || !vlive[c[1] as usize] || !vlive[c[2] as usize] {
            continue;
        }
        for k in 0..3 {
            let (a, b) = (c[k], c[(k + 1) % 3]);
            push(&mut pairs, ((a.min(b), a.max(b)), i))?;
        }
    }
    pairs.sort_unstable();
    Ok(pairs)
}

fn degenerate(corners: &[u32; 3]) -> bool {
    corners[0] == corners[1] || corners[1] == corners[2] || corners[2] == corners[0]
}

fn carries(corners: &[u32; 3], a: u32, b: u32) -> bool {
    corners.contains(&a) && corners.contains(&b)
}

/// The corner of a triangle carrying `(a, b)` that is neither endpoint.
fn apex_of(corners: &[u32; 3], a: u32, b: u32) -> u32 {
    if corners[0] != a && corners[0] != b {
        corners[0]
    } else if corners[1] != a && corners[1] != b {
        corners[1]
    } else {
        corners[2]
    }
}

fn f64v(p: &[f32; 3]) -> [f64; 3] {
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn tri_normal(corners: &[u32; 3], positions: &[[f32; 3]]) -> [f64; 3] {
    let p0 = f64v(&positions[corners[0] as usize]);
    let p1 = f64v(&positions[corners[1] as usize]);
    let p2 = f64v(&positions[corners[2] as usize]);
    cross3(sub3(p1, p0), sub3(p2, p0))
}

/// Twice the area of a triangle (the length of its cross product).
fn doubled_area(corners: &[u32; 3], positions: &[[f32; 3]]) -> f64 {
    let n = tri_normal(corners, positions);
    dot3(n, n).sqrt()
}

/// The largest corner cosine of a triangle (its smallest angle, as a
/// cosine), or `None` when an edge has zero length.
fn max_cos_angle(corners: &[u32; 3], positions: &[[f32; 3]]) -> Option<f64> {
    let p = [
        f64v(&positions[corners[0] as usize]),
        f64v(&positions[corners[1] as usize]),
        f64v(&positions[corners[2] as usize]),
    ];
    let mut max = f64::NEG_INFINITY;
    for k in 0..3 {
        let u = sub3(p[k], p[(k + 1) % 3]);
        let v = sub3(p[k], p[(k + 2) % 3]);
        let (nu, nv) = (dot3(u, u).sqrt(), dot3(v, v).sqrt());
        if nu == 0.0 || nv == 0.0 {
            return None;
        }
        max = max.max(dot3(u, v) / (nu * nv));
    }
    Some(max)
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

fn commit(
    tx: &mut Transaction,
    slots: usize,
    vlive: &[bool],
    tris: &[WorkTri],
    added: &[NewVert],
) -> Result<()> {
    // Everything is reserved before the first write: past this point only
    // infallible pushes into reserved capacity remain, so any allocation
    // failure still leaves an empty patch.
    let mut n_old = 0;
    let mut n_new = 0;
    let mut n_dead_tri = 0;
    for tri in tris {
        match tri.id {
            Some(_) if !tri.alive => n_dead_tri += 1,
            Some(_) => n_old += 1,
            None if tri.alive => n_new += 1,
            None => {}
        }
    }
    let n_dead_vert = vlive.iter().take(slots).filter(|&&live| !live).count();
    let mut ids: Vec<VertexId> = Vec::new();
    reserve(&mut ids, added.len())?;
    let mut set_ids: Vec<TriangleId> = Vec::new();
    reserve(&mut set_ids, n_old)?;
    let mut set_tris: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut set_tris, n_old)?;
    let mut new_tris: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut new_tris, n_new)?;
    let mut del_tris: Vec<TriangleId> = Vec::new();
    reserve(&mut del_tris, n_dead_tri)?;
    let mut del_verts: Vec<VertexId> = Vec::new();
    reserve(&mut del_verts, n_dead_vert)?;
    for (k, nv) in added.iter().enumerate() {
        if vlive[slots + k] {
            ids.push(tx.add_vertex_with_parents(nv.pos, &nv.parents)?);
        } else {
            // Collapsed before commit; no live triangle references it.
            ids.push(VertexId::MAX);
        }
    }
    let resolve = |v: u32| -> VertexId {
        debug_assert!(vlive[v as usize]);
        let i = v as usize;
        if i < slots { v } else { ids[i - slots] }
    };
    let mesh = tx.mesh();
    for tri in tris {
        match tri.id {
            Some(id) if !tri.alive => del_tris.push(id),
            Some(id) => {
                let current = [
                    resolve(tri.corners[0]),
                    resolve(tri.corners[1]),
                    resolve(tri.corners[2]),
                ];
                let original = mesh.triangle(id).expect("remesh: working id must exist");
                if current != original {
                    set_ids.push(id);
                    set_tris.push(current);
                }
            }
            None if tri.alive => new_tris.push([
                resolve(tri.corners[0]),
                resolve(tri.corners[1]),
                resolve(tri.corners[2]),
            ]),
            None => {}
        }
    }
    for v in 0..slots as u32 {
        if !vlive[v as usize] {
            del_verts.push(v);
        }
    }
    tx.set_triangles(&set_ids, &set_tris)?;
    tx.add_triangles(&new_tris)?;
    tx.remove_triangles(&del_tris)?;
    tx.remove_vertices(&del_verts)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::{Mesh, Transaction, VertexId};

    use super::*;
    use crate::OpsError;
    use crate::test_util::{edit, mesh_of, parts, sweep, with_triangles};
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

    /// Every live triangle faces +z with positive area (planar fixtures).
    fn assert_faces_up(mesh: &Mesh) {
        let (pos, tris) = parts(mesh);
        assert!(!tris.is_empty());
        for t in &tris {
            let x = |v: u32| f64::from(pos[3 * v as usize]);
            let y = |v: u32| f64::from(pos[3 * v as usize + 1]);
            let z = (x(t[1]) - x(t[0])) * (y(t[2]) - y(t[0]))
                - (y(t[1]) - y(t[0])) * (x(t[2]) - x(t[0]));
            assert!(z > 0.0, "flipped or degenerate: {t:?} z={z}");
        }
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
        assert_faces_up(&mesh);
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
        assert_eq!(
            (report.splits, report.collapses, report.flips, report.rounds),
            (0, 0, 0, 0)
        );
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

    fn grid_4x4() -> Mesh {
        let mut pos = Vec::new();
        for j in 0..4 {
            for i in 0..4 {
                pos.extend([i as f32 * 0.1, j as f32 * 0.1, 0.0]);
            }
        }
        let mut tris = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                let a = j * 4 + i;
                tris.push([a, a + 1, a + 5]);
                tris.push([a, a + 5, a + 4]);
            }
        }
        mesh_of(&pos, &tris)
    }

    fn kite() -> Mesh {
        // Convex kite, apexes on opposite sides of the shared edge (0, 1):
        // (0, 1, 2) is skinny, so flipping to (2, 3) improves the angles.
        mesh_of(
            &[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.15, 0.0, 1.0, -2.0, 0.0],
            &[[0, 1, 2], [1, 0, 3]],
        )
    }

    #[test]
    fn collapse_coarsens_dense_interior() {
        let mut mesh = grid_4x4();
        let before = parts(&mesh);
        let options = RemeshOptions::builder()
            .center([0.15, 0.15, 0.0])
            .radius(10.0)
            .target_length(1.0)
            .build();
        let (report, _) = edit(&mut mesh, |tx| remesh(tx, &options));
        assert_eq!(report.splits, 0);
        assert!(report.collapses > 0, "expected collapses, got {report:?}");
        let (_, tris_after) = parts(&mesh);
        assert!(tris_after.len() < before.1.len());
        for t in &tris_after {
            assert!(
                t[0] != t[1] && t[1] != t[2] && t[2] != t[0],
                "degenerate: {t:?}"
            );
        }
        for v in [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15] {
            assert!(mesh.is_vertex_alive(v as VertexId));
        }
        assert_faces_up(&mesh);
    }

    #[test]
    fn collapse_skips_all_boundary() {
        let mut mesh = mesh_of(&[0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 3.0, 0.0], &[[0, 1, 2]]);
        let before = parts(&mesh);
        let options = RemeshOptions::builder()
            .center([1.0, 1.0, 0.0])
            .radius(10.0)
            .target_length(10.0)
            .build();
        let (report, empty) = edit(&mut mesh, |tx| remesh(tx, &options));
        assert_eq!(
            (report.splits, report.collapses, report.flips, report.rounds),
            (0, 0, 0, 0)
        );
        assert!(empty);
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn flip_improves_skinny_kite() {
        let mut mesh = kite();
        let options = RemeshOptions::builder()
            .center([1.0, 0.5, 0.0])
            .radius(10.0)
            .target_length(10.0)
            .build();
        let (report, _) = edit(&mut mesh, |tx| remesh(tx, &options));
        assert_eq!(
            (report.splits, report.collapses, report.flips, report.rounds),
            (0, 0, 1, 1)
        );
        let (_, tris) = parts(&mesh);
        assert!(
            tris.iter().any(|t| t.contains(&2) && t.contains(&3)),
            "flipped diagonal missing: {tris:?}"
        );
        assert_faces_up(&mesh);
    }

    #[test]
    fn remesh_respects_max_rounds() {
        let mut mesh = quad_4x1();
        let before = parts(&mesh);
        let options = RemeshOptions::builder()
            .center([2.0, 0.5, 0.0])
            .radius(10.0)
            .target_length(1.0)
            .max_rounds(0)
            .build();
        let (report, empty) = edit(&mut mesh, |tx| remesh(tx, &options));
        assert_eq!(
            (report.splits, report.collapses, report.flips, report.rounds),
            (0, 0, 0, 0)
        );
        assert!(empty);
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn collapse_and_flip_keep_atomicity() {
        let grid_options = RemeshOptions::builder()
            .center([0.15, 0.15, 0.0])
            .radius(10.0)
            .target_length(1.0)
            .build();
        let n = sweep(grid_4x4(), |tx: &mut Transaction| {
            remesh(tx, &grid_options)?;
            Ok(())
        });
        assert!(n > 0);
        let kite_options = RemeshOptions::builder()
            .center([1.0, 0.5, 0.0])
            .radius(10.0)
            .target_length(10.0)
            .build();
        let n = sweep(kite(), |tx: &mut Transaction| {
            remesh(tx, &kite_options)?;
            Ok(())
        });
        assert!(n > 0);
    }

    #[test]
    fn collapse_skips_edges_touching_degenerate_triangles() {
        // Degenerate (5, 5, 9) touches interior vertex 5 but not edge (5, 6):
        // that collapse is skipped, the rest of the grid still coarsens, and
        // the degenerate triangle is left untouched.
        let mut mesh = with_triangles(&grid_4x4(), &[[5, 5, 9]]);
        let options = RemeshOptions::builder()
            .center([0.15, 0.15, 0.0])
            .radius(10.0)
            .target_length(1.0)
            .build();
        let (report, _) = edit(&mut mesh, |tx| remesh(tx, &options));
        assert!(report.collapses > 0, "expected collapses, got {report:?}");
        let (_, tris) = parts(&mesh);
        assert!(tris.contains(&[5, 5, 9]));
        for t in &tris {
            for &v in t {
                assert!(mesh.is_vertex_alive(v));
            }
        }
    }

    #[test]
    fn flip_rejects_reflex_quad() {
        // Reflex at c=(0, 1): the replacement diagonal (c, d) would leave
        // the quad, nesting one new triangle inside the other.
        let mut mesh = mesh_of(
            &[1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
            &[[0, 1, 2], [1, 0, 3]],
        );
        let before = parts(&mesh);
        let options = RemeshOptions::builder()
            .center([1.0, 0.0, 0.0])
            .radius(10.0)
            .target_length(10.0)
            .build();
        let (report, empty) = edit(&mut mesh, |tx| remesh(tx, &options));
        assert_eq!(
            (report.splits, report.collapses, report.flips, report.rounds),
            (0, 0, 0, 0)
        );
        assert!(empty);
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn split_finds_edge_crossing_vertex_free_region() {
        // The region around (5, 0) holds no vertices, but the long base
        // edge crosses it: its midpoint is in, so it splits.
        let mut mesh = mesh_of(
            &[0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.1, 0.0],
            &[[0, 1, 2]],
        );
        let options = RemeshOptions::builder()
            .center([5.0, 0.0, 0.0])
            .radius(1.0)
            .target_length(1.0)
            .build();
        let (report, _) = edit(&mut mesh, |tx| remesh(tx, &options));
        assert!(report.splits > 0, "expected splits, got {report:?}");
    }

    #[test]
    fn split_defaults_target_to_mesh_mean() {
        // Mean edge length is (4 + 4 + 1 + 1 + sqrt(17)) / 5 ~ 2.82; the
        // two long sides and the diagonal exceed 4/3 of it exactly once.
        let mut mesh = quad_4x1();
        let options = RemeshOptions::builder()
            .center([2.0, 0.5, 0.0])
            .radius(10.0)
            .build();
        let (report, _) = edit(&mut mesh, |tx| remesh(tx, &options));
        assert_eq!(
            (report.splits, report.collapses, report.flips, report.rounds),
            (3, 0, 1, 2)
        );
        for len in edge_lengths(&mesh) {
            assert!(len <= 2.8247 * 4.0 / 3.0 + 1e-4, "long edge left: {len}");
        }
    }
}
