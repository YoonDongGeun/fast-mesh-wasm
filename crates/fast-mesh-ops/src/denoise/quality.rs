//! What [`denoise`](super::denoise) and
//! [`delaunay_flips`](crate::delaunay::delaunay_flips) share: region masks,
//! half-edge twins and the [`SurfaceStats`] their reports carry.
//!
//! Triangles are addressed by slot: a dead slot, or one with a repeated
//! vertex id, has the corners `[NONE; 3]` and takes part in nothing.
//! Half-edge `3 * t + k` runs from corner `k` of triangle `t` to corner
//! `(k + 1) % 3`.

use std::f64::consts::PI;

use fast_mesh_store::{Mesh, StoreError, TriangleId, par};

use crate::remesh::{carries, cross3, dot3, f64v, runs_directed, sub3};
use crate::topology::{VertexFans, proper_triangle};
use crate::{OpsError, Result, allocation_allowed, filled, reserve};

/// No triangle or half-edge.
pub(crate) const NONE: u32 = u32::MAX;
/// Triangles, half-edges or vertices per parallel chunk. Fixed, so results
/// do not depend on the thread count.
pub(crate) const CHUNK: usize = 4096;
/// Work below this many items stays on the calling thread.
pub(crate) const PAR_MIN: usize = 16 * 1024;

pub(crate) fn oom() -> OpsError {
    OpsError::Store(StoreError::OutOfMemory)
}

/// Readouts of a surface's smoothness and triangle shape, in degrees.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SurfaceStats {
    /// Triangles measured (live, with three distinct vertices, in the
    /// region when one is given).
    pub triangles: usize,
    /// The smallest corner angle of those triangles; 0 when there are none.
    pub min_angle_deg: f64,
    /// The mean angle between the normals of two measured triangles that
    /// share an edge (consistently wound, used by no other triangle), over
    /// every such edge; 0 when there are none. Scan noise raises it; flat
    /// faces bring it to 0, apart from the edges along real features.
    pub mean_normal_deviation_deg: f64,
}

/// [`SurfaceStats`] of `mesh`, or of its triangles in `region` (triangle
/// ids; `None` for the whole mesh).
///
/// # Errors
/// [`OpsError::InvalidInput`] when `region` names a dead or out-of-range
/// triangle; [`OpsError::Store`] when out of memory.
pub fn surface_stats(mesh: &Mesh, region: Option<&[TriangleId]>) -> Result<SurfaceStats> {
    let mask = region_mask(mesh, region)?;
    let corners = corners_of(mesh)?;
    let fans = VertexFans::build(mesh)?;
    let twins = twins(&corners, &fans)?;
    stats(mesh.positions(), &corners, &twins, mask.as_deref())
}

/// `region` as one flag per triangle slot, `None` for the whole mesh.
pub(crate) fn region_mask(mesh: &Mesh, region: Option<&[TriangleId]>) -> Result<Option<Vec<bool>>> {
    let Some(region) = region else {
        return Ok(None);
    };
    let mut mask = filled(mesh.triangle_slots(), false)?;
    for &t in region {
        if !mesh.is_triangle_alive(t) {
            return Err(OpsError::InvalidInput(
                "triangle id is dead or out of range",
            ));
        }
        mask[t as usize] = true;
    }
    Ok(Some(mask))
}

/// Whether triangle slot `t` is in the region.
#[inline]
pub(crate) fn in_mask(mask: Option<&[bool]>, t: usize) -> bool {
    mask.is_none_or(|m| m[t])
}

/// The corners of every triangle slot, `[NONE; 3]` for dead or improper ones.
pub(crate) fn corners_of(mesh: &Mesh) -> Result<Vec<[u32; 3]>> {
    let slots = mesh.triangle_slots();
    if 3 * slots >= NONE as usize {
        return Err(OpsError::InvalidInput("too many triangles for half-edges"));
    }
    let mut corners = Vec::new();
    reserve(&mut corners, slots)?;
    for t in 0..slots as TriangleId {
        corners.push(proper_triangle(mesh, t).unwrap_or([NONE; 3]));
    }
    Ok(corners)
}

/// The twin of every half-edge: the half-edge of the one other triangle on
/// its edge, which must run the other way. `NONE` on a boundary edge, an
/// edge of three or more triangles, or one both triangles wind the same way.
pub(crate) fn twins(corners: &[[u32; 3]], fans: &VertexFans) -> Result<Vec<u32>> {
    let mut twin = filled(3 * corners.len(), NONE)?;
    par::for_each_chunk_mut(&mut twin, 3 * CHUNK, 3 * PAR_MIN, |i, chunk| {
        let base = i * CHUNK;
        for (j, slot) in chunk.chunks_mut(3).enumerate() {
            let t = base + j;
            let c = corners[t];
            if c[0] == NONE {
                continue;
            }
            for (k, out) in slot.iter_mut().enumerate() {
                *out = twin_of(corners, fans, t, c[k], c[(k + 1) % 3]);
            }
        }
    });
    Ok(twin)
}

/// The twin of half-edge `a -> b` of triangle `t` (see [`twins`]).
fn twin_of(corners: &[[u32; 3]], fans: &VertexFans, t: usize, a: u32, b: u32) -> u32 {
    let mut found = NONE;
    for &u in fans.fan(a) {
        let u = u as usize;
        if u == t || !carries(&corners[u], a, b) {
            continue;
        }
        if found != NONE || !runs_directed(&corners[u], b, a) {
            return NONE;
        }
        let c = corners[u];
        let kk = (0..3).find(|&k| c[k] == b).expect("carries b") as u32;
        found = 3 * u as u32 + kk;
    }
    found
}

/// Triangle `t`'s corner positions in `f64`.
#[inline]
pub(crate) fn points(positions: &[[f32; 3]], c: [u32; 3]) -> [[f64; 3]; 3] {
    c.map(|v| f64v(&positions[v as usize]))
}

/// The unit normal of `p`, or `None` for a triangle too thin to have one:
/// twice its area at most `1e-12` of its longest edge squared.
#[inline]
pub(crate) fn unit_normal(p: [[f64; 3]; 3]) -> Option<[f64; 3]> {
    let n = cross3(sub3(p[1], p[0]), sub3(p[2], p[0]));
    let len = dot3(n, n).sqrt();
    let longest = (0..3)
        .map(|k| {
            let e = sub3(p[(k + 1) % 3], p[k]);
            dot3(e, e)
        })
        .fold(0.0, f64::max);
    (len.is_finite() && len > 1e-12 * longest).then(|| n.map(|x| x / len))
}

/// The angle between `u` and `v` in radians, 0 when either is zero.
#[inline]
pub(crate) fn angle(u: [f64; 3], v: [f64; 3]) -> f64 {
    let c = cross3(u, v);
    dot3(c, c).sqrt().atan2(dot3(u, v))
}

/// The corner angles of `p` in radians, at corners 0, 1 and 2.
#[inline]
pub(crate) fn corner_angles(p: [[f64; 3]; 3]) -> [f64; 3] {
    [0, 1, 2].map(|k| angle(sub3(p[(k + 1) % 3], p[k]), sub3(p[(k + 2) % 3], p[k])))
}

/// The smallest corner angle of `p` in radians.
#[inline]
pub(crate) fn min_angle(p: [[f64; 3]; 3]) -> f64 {
    let [a, b, c] = corner_angles(p);
    a.min(b).min(c)
}

/// Per chunk of triangles: measured triangles, smallest angle, summed
/// normal deviation and measured edges.
type Partial = (usize, f64, f64, usize);

/// [`SurfaceStats`] of `corners` at `positions`, over the triangles in
/// `mask`. Chunk sums are added in chunk order, so the result is the same
/// for any thread count.
pub(crate) fn stats(
    positions: &[[f32; 3]],
    corners: &[[u32; 3]],
    twins: &[u32],
    mask: Option<&[bool]>,
) -> Result<SurfaceStats> {
    allocation_allowed()?;
    let parts: Vec<Partial> = par::map_range(corners.len(), CHUNK, PAR_MIN, oom, |r| {
        let mut part: Partial = (0, f64::INFINITY, 0.0, 0);
        for t in r {
            let c = corners[t];
            if c[0] == NONE || !in_mask(mask, t) {
                continue;
            }
            let p = points(positions, c);
            part.0 += 1;
            part.1 = part.1.min(min_angle(p));
            let Some(n) = unit_normal(p) else {
                continue;
            };
            for k in 0..3 {
                let he = 3 * t + k;
                let tw = twins[he] as usize;
                // Each edge once, from its lower half-edge.
                if twins[he] == NONE || tw < he || !in_mask(mask, tw / 3) {
                    continue;
                }
                if let Some(m) = unit_normal(points(positions, corners[tw / 3])) {
                    part.2 += angle(n, m);
                    part.3 += 1;
                }
            }
        }
        Ok(part)
    })?;
    let mut total: Partial = (0, f64::INFINITY, 0.0, 0);
    for p in parts {
        total.0 += p.0;
        total.1 = total.1.min(p.1);
        total.2 += p.2;
        total.3 += p.3;
    }
    let deg = 180.0 / PI;
    Ok(SurfaceStats {
        triangles: total.0,
        min_angle_deg: if total.0 == 0 { 0.0 } else { total.1 * deg },
        mean_normal_deviation_deg: if total.3 == 0 {
            0.0
        } else {
            total.2 / total.3 as f64 * deg
        },
    })
}

/// Concatenates per-chunk lists in chunk order.
pub(crate) fn concat(parts: Vec<Vec<u32>>) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    reserve(&mut out, parts.iter().map(Vec::len).sum())?;
    for p in parts {
        out.extend_from_slice(&p);
    }
    Ok(out)
}
