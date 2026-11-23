//! Feature-preserving denoising: bilateral filtering of the face normals,
//! then moving the vertices to fit the filtered normals.
//!
//! [`denoise`] runs [`DenoiseOptions::iterations`] rounds. Each round
//!
//! 1. takes every triangle's normal and area from the current positions;
//! 2. filters the normals of the triangles in the region,
//!    [`NORMAL_PASSES`] times: a triangle's new normal is the mean of the
//!    normals of the triangles sharing a vertex with it (itself included,
//!    a triangle counted once per vertex it shares), weighted by their
//!    area and by how close their normal is to its own. The weight falls
//!    smoothly to 0 at [`DenoiseOptions::feature_angle_deg`], so triangles
//!    across a sharper edge never pull on each other and the edge stays
//!    sharp. The result is blended with the unfiltered normal by
//!    [`DenoiseOptions::strength`];
//! 3. moves each free vertex [`VERTEX_PASSES`] times towards the planes of
//!    its triangles: through each triangle's centroid, along its filtered
//!    normal. A vertex is free when every triangle using it is in the
//!    region, so nothing outside the region moves.
//!
//! The vertex step only moves along normals, so the surface does not
//! shrink tangentially, and vertices on a crease settle onto the line where
//! the planes of its two sides meet. A last check puts back the vertices of
//! any triangle the moves turned over (it faces against its filtered
//! normal, and did not before), so they add no fold.
//!
//! Positions are read as `f32`, worked on in `f64` and written back
//! rounded, as one [`Transaction`] edit of positions only: the triangles,
//! and the UV channel, are kept. Each step is computed per triangle or per
//! vertex from the previous step's values, in parallel with the `parallel`
//! feature, so the result is bit-identical with and without it.

mod quality;

use fast_mesh_store::{Transaction, TriangleId, VertexId, par};

use crate::remesh::{dot3, sub3};
use crate::topology::VertexFans;
use crate::{OpsError, Result, allocation_allowed, filled, push, reserve};
pub(crate) use quality::{
    CHUNK, NONE, PAR_MIN, angle, concat, corners_of, in_mask, min_angle, oom, points, region_mask,
    stats, twins, unit_normal,
};
pub use quality::{SurfaceStats, surface_stats};

/// Normal filtering passes per round.
pub const NORMAL_PASSES: usize = 4;
/// Vertex update passes per round.
pub const VERTEX_PASSES: usize = 10;
/// The most rounds [`denoise`] accepts.
pub const MAX_DENOISE_ITERATIONS: u32 = 100;

/// Options for [`denoise`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenoiseOptions {
    /// Rounds of normal filtering and vertex fitting, at most
    /// [`MAX_DENOISE_ITERATIONS`]. Default 5. 0 changes nothing.
    pub iterations: u32,
    /// How far each round moves the normals towards their filtered value,
    /// in `[0, 1]`: 0 changes nothing, 1 takes the filtered normals as
    /// they are. Default 0.5.
    pub strength: f32,
    /// The angle between two triangles' normals, in degrees in `(0, 180]`,
    /// from which they no longer smooth each other: edges sharper than it
    /// are kept as features. Default 45.
    pub feature_angle_deg: f32,
}

impl Default for DenoiseOptions {
    fn default() -> Self {
        Self {
            iterations: 5,
            strength: 0.5,
            feature_angle_deg: 45.0,
        }
    }
}

/// Outcome of [`denoise`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DenoiseReport {
    /// Vertices whose position changed.
    pub moved: usize,
    /// The largest distance a vertex moved, in mesh units.
    pub max_displacement: f64,
    /// Rounds run.
    pub iterations: u32,
    /// Vertices put back where they were because a triangle using them
    /// would have turned over.
    pub reverted: usize,
    /// The region's [`SurfaceStats`] before.
    pub before: SurfaceStats,
    /// The region's [`SurfaceStats`] after.
    pub after: SurfaceStats,
}

/// Smooths noise out of the surface in `tx` while keeping its sharp
/// features (see the [module docs](self)), on the triangles in `region`
/// (triangle ids; `None` for the whole mesh).
///
/// # Errors
/// [`OpsError::InvalidInput`] for more than [`MAX_DENOISE_ITERATIONS`]
/// rounds, a strength outside `[0, 1]`, a feature angle outside
/// `(0, 180]`, or a dead or out-of-range triangle in `region`;
/// [`OpsError::Store`] when out of memory. Nothing is written on error.
pub fn denoise(
    tx: &mut Transaction<'_>,
    options: &DenoiseOptions,
    region: Option<&[TriangleId]>,
) -> Result<DenoiseReport> {
    validate(options)?;
    let mesh = tx.mesh();
    let mask = region_mask(mesh, region)?;
    let mask = mask.as_deref();
    let corners = corners_of(mesh)?;
    let fans = VertexFans::build(mesh)?;
    let twin = twins(&corners, &fans)?;
    let base = mesh.positions();
    let before = stats(base, &corners, &twin, mask)?;
    let mut report = DenoiseReport {
        before,
        after: before,
        ..DenoiseReport::default()
    };
    if options.iterations == 0 || options.strength == 0.0 {
        return Ok(report);
    }

    let slots = base.len();
    let mut free = filled(slots, false)?;
    let mut any_free = false;
    for (v, f) in free.iter_mut().enumerate() {
        let fan = fans.fan(v as VertexId);
        *f = !fan.is_empty() && fan.iter().all(|&t| in_mask(mask, t as usize));
        any_free |= *f;
    }
    if !any_free {
        return Ok(report);
    }

    let tris = corners.len();
    let mut pos = filled(slots, [0.0f64; 3])?;
    for (p, b) in pos.iter_mut().zip(base) {
        *p = b.map(f64::from);
    }
    let mut next = filled(slots, [0.0f64; 3])?;
    let mut normals = filled(tris, [0.0f64; 3])?;
    let mut areas = filled(tris, 0.0f64)?;
    let mut work = filled(tris, [0.0f64; 3])?;
    let mut filtered = filled(tris, [0.0f64; 3])?;
    let mut centroids = filled(tris, [0.0f64; 3])?;
    let ctx = Ctx {
        corners: &corners,
        fans: &fans,
        mask,
        free: &free,
        cos_feature: f64::from(options.feature_angle_deg).to_radians().cos(),
        strength: f64::from(options.strength),
    };

    for _ in 0..options.iterations {
        ctx.face_normals(&pos, &mut normals, &mut areas);
        for pass in 0..NORMAL_PASSES {
            let src = if pass == 0 { &normals } else { &work };
            ctx.filter(src, &areas, &mut filtered);
            std::mem::swap(&mut work, &mut filtered);
        }
        ctx.blend(&normals, &mut work);
        for _ in 0..VERTEX_PASSES {
            ctx.centroids(&pos, &mut centroids);
            ctx.fit(&pos, &work, &centroids, &mut next);
            std::mem::swap(&mut pos, &mut next);
        }
    }
    report.iterations = options.iterations;

    // Round to the stored precision, then undo any fold on the rounded
    // positions themselves.
    let mut out = filled(slots, [0.0f32; 3])?;
    for (v, o) in out.iter_mut().enumerate() {
        *o = if free[v] {
            pos[v].map(|x| x as f32)
        } else {
            base[v]
        };
    }
    report.reverted = unfold(&ctx, base, &work, &mut out)?;

    let mut ids: Vec<VertexId> = Vec::new();
    let mut moved: Vec<[f32; 3]> = Vec::new();
    let changed = (0..slots).filter(|&v| free[v] && out[v] != base[v]).count();
    reserve(&mut ids, changed)?;
    reserve(&mut moved, changed)?;
    for v in 0..slots {
        if free[v] && out[v] != base[v] {
            ids.push(v as VertexId);
            moved.push(out[v]);
            let d = sub3(out[v].map(f64::from), base[v].map(f64::from));
            report.max_displacement = report.max_displacement.max(dot3(d, d).sqrt());
        }
    }
    report.moved = ids.len();
    report.after = stats(&out, &corners, &twin, mask)?;
    if !ids.is_empty() {
        tx.set_positions(&ids, &moved)?;
    }
    Ok(report)
}

fn validate(options: &DenoiseOptions) -> Result<()> {
    if options.iterations > MAX_DENOISE_ITERATIONS {
        return Err(OpsError::InvalidInput(
            "denoise iterations must be at most 100",
        ));
    }
    if !(0.0..=1.0).contains(&options.strength) {
        return Err(OpsError::InvalidInput("denoise strength must be in [0, 1]"));
    }
    let a = options.feature_angle_deg;
    if !(a > 0.0 && a <= 180.0) {
        return Err(OpsError::InvalidInput(
            "denoise feature angle must be in (0, 180] degrees",
        ));
    }
    Ok(())
}

/// What every step reads.
struct Ctx<'a> {
    corners: &'a [[u32; 3]],
    fans: &'a VertexFans,
    mask: Option<&'a [bool]>,
    free: &'a [bool],
    cos_feature: f64,
    strength: f64,
}

impl Ctx<'_> {
    /// Unit normals (zero for a triangle without one) and areas.
    fn face_normals(&self, pos: &[[f64; 3]], normals: &mut [[f64; 3]], areas: &mut [f64]) {
        par::for_each_chunk_mut(normals, CHUNK, PAR_MIN, |i, chunk| {
            for (j, n) in chunk.iter_mut().enumerate() {
                let c = self.corners[i * CHUNK + j];
                *n = if c[0] == NONE {
                    [0.0; 3]
                } else {
                    unit_normal(c.map(|v| pos[v as usize])).unwrap_or([0.0; 3])
                };
            }
        });
        par::for_each_chunk_mut(areas, CHUNK, PAR_MIN, |i, chunk| {
            for (j, a) in chunk.iter_mut().enumerate() {
                let c = self.corners[i * CHUNK + j];
                *a = if c[0] == NONE {
                    0.0
                } else {
                    let p = c.map(|v| pos[v as usize]);
                    let n = crate::remesh::cross3(sub3(p[1], p[0]), sub3(p[2], p[0]));
                    0.5 * dot3(n, n).sqrt()
                };
            }
        });
    }

    /// One bilateral pass from `src` into `out`; triangles outside the
    /// region keep their normal.
    fn filter(&self, src: &[[f64; 3]], areas: &[f64], out: &mut [[f64; 3]]) {
        let cos_f = self.cos_feature;
        let span = 1.0 - cos_f;
        par::for_each_chunk_mut(out, CHUNK, PAR_MIN, |i, chunk| {
            for (j, o) in chunk.iter_mut().enumerate() {
                let t = i * CHUNK + j;
                let n = src[t];
                if self.corners[t][0] == NONE || n == [0.0; 3] || !in_mask(self.mask, t) {
                    *o = n;
                    continue;
                }
                let mut sum = [0.0f64; 3];
                for v in self.corners[t] {
                    for &g in self.fans.fan(v) {
                        let m = src[g as usize];
                        let d = dot3(n, m);
                        if d <= cos_f {
                            continue;
                        }
                        let x = (d - cos_f) / span;
                        let w = areas[g as usize] * x * x;
                        sum = [sum[0] + w * m[0], sum[1] + w * m[1], sum[2] + w * m[2]];
                    }
                }
                *o = normalized(sum).unwrap_or(n);
            }
        });
    }

    /// `filtered <- normalize((1 - strength) * normals + strength * filtered)`
    /// in the region.
    fn blend(&self, normals: &[[f64; 3]], filtered: &mut [[f64; 3]]) {
        let s = self.strength;
        par::for_each_chunk_mut(filtered, CHUNK, PAR_MIN, |i, chunk| {
            for (j, f) in chunk.iter_mut().enumerate() {
                let n = normals[i * CHUNK + j];
                if s >= 1.0 || n == [0.0; 3] {
                    continue;
                }
                let mixed = [0, 1, 2].map(|k| (1.0 - s) * n[k] + s * f[k]);
                *f = normalized(mixed).unwrap_or(n);
            }
        });
    }

    fn centroids(&self, pos: &[[f64; 3]], out: &mut [[f64; 3]]) {
        par::for_each_chunk_mut(out, CHUNK, PAR_MIN, |i, chunk| {
            for (j, o) in chunk.iter_mut().enumerate() {
                let c = self.corners[i * CHUNK + j];
                if c[0] == NONE {
                    continue;
                }
                let [a, b, d] = c.map(|v| pos[v as usize]);
                *o = [0, 1, 2].map(|k| (a[k] + b[k] + d[k]) / 3.0);
            }
        });
    }

    /// One vertex pass: each free vertex moves by the mean of its offsets
    /// to its triangles' planes, along their normals.
    fn fit(
        &self,
        pos: &[[f64; 3]],
        normals: &[[f64; 3]],
        centroids: &[[f64; 3]],
        out: &mut [[f64; 3]],
    ) {
        par::for_each_chunk_mut(out, CHUNK, PAR_MIN, |i, chunk| {
            for (j, o) in chunk.iter_mut().enumerate() {
                let v = i * CHUNK + j;
                let x = pos[v];
                *o = x;
                if !self.free[v] {
                    continue;
                }
                let mut sum = [0.0f64; 3];
                let mut count = 0u32;
                for &t in self.fans.fan(v as VertexId) {
                    let m = normals[t as usize];
                    if m == [0.0; 3] {
                        continue;
                    }
                    let h = dot3(m, sub3(centroids[t as usize], x));
                    sum = [sum[0] + h * m[0], sum[1] + h * m[1], sum[2] + h * m[2]];
                    count += 1;
                }
                if count > 0 {
                    let inv = 1.0 / f64::from(count);
                    *o = [
                        x[0] + sum[0] * inv,
                        x[1] + sum[1] * inv,
                        x[2] + sum[2] * inv,
                    ];
                }
            }
        });
    }
}

/// Puts back the vertices of every triangle that `out` turned over, until
/// none is left: one that faces against its filtered normal in `target`
/// (or has no area left) and did not at `base`. Triangles the noise itself
/// had turned over are not put back. Returns how many vertices were.
fn unfold(
    ctx: &Ctx<'_>,
    base: &[[f32; 3]],
    target: &[[f64; 3]],
    out: &mut [[f32; 3]],
) -> Result<usize> {
    let mut reverted = 0;
    loop {
        allocation_allowed()?;
        let parts = par::map_range(ctx.corners.len(), CHUNK, PAR_MIN, oom, |r| {
            let mut folded: Vec<u32> = Vec::new();
            for t in r {
                let c = ctx.corners[t];
                if c[0] == NONE || !c.iter().any(|&v| out[v as usize] != base[v as usize]) {
                    continue;
                }
                let faces = |p: &[[f32; 3]]| {
                    unit_normal(points(p, c)).is_some_and(|n| dot3(n, target[t]) > 0.0)
                };
                if faces(base) && !faces(out) {
                    push(&mut folded, t as u32)?;
                }
            }
            Ok(folded)
        })?;
        let folded = concat(parts)?;
        if folded.is_empty() {
            return Ok(reverted);
        }
        for t in folded {
            for v in ctx.corners[t as usize] {
                let v = v as usize;
                if out[v] != base[v] {
                    out[v] = base[v];
                    reverted += 1;
                }
            }
        }
    }
}

#[inline]
fn normalized(v: [f64; 3]) -> Option<[f64; 3]> {
    let len = dot3(v, v).sqrt();
    (len.is_finite() && len > 0.0).then(|| v.map(|x| x / len))
}

#[cfg(test)]
mod tests;
