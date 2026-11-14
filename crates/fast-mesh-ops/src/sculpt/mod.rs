//! Brush sculpting domain: stroke state channels, surface filters, mirror
//! symmetry, and localized region-parity brush kernels.

mod crease;
mod filter;
mod grab;
mod inflate;
mod plane;
mod smooth;
mod stroke;
#[cfg(test)]
mod tests;
mod twist;

use fast_mesh_store::{Incidence, Mesh, Transaction, TriangleId, VertexId};

use crate::topology::proper_triangle;
use crate::{OpsError, Result, push};

pub use crate::stretch::{
    DEFAULT_MAX_SPLITS, DEFAULT_SHARP_ANGLE, DEFAULT_STRETCH_LIMIT, REST_SIZE_CHANNEL,
    StretchOptions, StretchReport, capture_rest_sizes, ensure_rest_sizes, split_stretched,
};
pub use crease::{CreaseOptions, CreaseOptionsBuilder, CreaseReport, crease, crease_region};
pub use filter::{SurfaceFilter, Symmetry};
pub use grab::{
    GRAB_MAX_STRETCH, GrabMethod, GrabOptions, GrabOptionsBuilder, GrabReport, grab, grab_region,
    grab_region_with_stroke, grab_widest_reach,
};
pub use inflate::{
    InflateMethod, InflateOptions, InflateOptionsBuilder, InflateReport, inflate, inflate_region,
    inflate_region_with_stroke,
};
pub use plane::{
    ClayOptions, ClayOptionsBuilder, ClayProfile, ClayReport, FlattenMode, FlattenOptions,
    FlattenOptionsBuilder, FlattenReport, clay, clay_region, flatten, flatten_region,
};
pub use smooth::{
    HC_ALPHA, HC_BETA, SmoothMethod, SmoothOptions, SmoothOptionsBuilder, SmoothReport,
    TAUBIN_LAMBDA, TAUBIN_MU, smooth, smooth_region,
};
pub use stroke::{
    STROKE_NORMAL_CHANNEL, STROKE_POS_CHANNEL, StrokeSlice, capture_stroke_state,
    ensure_stroke_state,
};
pub use twist::{TwistMethod, TwistOptions, TwistOptionsBuilder, TwistReport, twist, twist_region};

/// How a brush's effect fades from its center (`t = 0`) to its rim (`t = 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Falloff {
    /// Linear decay: `1 - t`.
    Linear,
    /// Smoothstep decay: `s = 1 - t`, then `s * s * (3 - 2 * s)`.
    Smooth,
}

/// A sculpt brush: a sphere of influence with a falloff and a strength.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Brush {
    /// Center of the brush in local coordinates.
    pub center: [f32; 3],
    /// Radius of influence in local units; vertices at or past it are
    /// untouched. Must be positive and finite.
    pub radius: f32,
    /// How the effect fades with distance from the center.
    pub falloff: Falloff,
    /// Displacement scale in local units; negative pulls instead of pushing.
    /// Must be finite.
    pub strength: f32,
}

impl Default for Brush {
    fn default() -> Self {
        Brush {
            center: [0.0; 3],
            radius: 1.0,
            falloff: Falloff::Linear,
            strength: 1.0,
        }
    }
}

/// The falloff weight at normalized distance `t` (`0` at the center, `1` at the
/// rim). Callers keep `t` below `1`; at and past the rim the brush has no
/// effect.
pub(crate) fn falloff_value(falloff: Falloff, t: f32) -> f32 {
    match falloff {
        Falloff::Linear => 1.0 - t,
        Falloff::Smooth => {
            let s = 1.0 - t;
            s * s * (3.0 - 2.0 * s)
        }
    }
}

/// Normalized brush distance `dist / radius` when strictly inside `[0, 1)`.
pub(crate) fn normalized_dist(center: [f32; 3], radius: f32, p: &[f32; 3]) -> Option<f32> {
    let dist = center
        .iter()
        .zip(p)
        .map(|(c, p)| (p - c).powi(2))
        .sum::<f32>()
        .sqrt();
    let t = dist / radius;
    (t.is_finite() && t < 1.0).then_some(t)
}

/// The `strength * falloff` weight of a vertex at `p`, or `None` at or past
/// the rim.
#[cfg(test)]
pub(crate) fn brush_weight(brush: &Brush, p: &[f32; 3]) -> Option<f32> {
    normalized_dist(brush.center, brush.radius, p)
        .map(|t| brush.strength * falloff_value(brush.falloff, t))
}

/// Whether any live vertex (and triangle-bearing when `require_triangles` is
/// set) lies inside at least one active symmetry pass of `brush`. Performs zero
/// heap allocations.
pub(crate) fn has_any_region_member(
    mesh: &Mesh,
    brush: &Brush,
    symmetry: Symmetry,
    require_triangles: bool,
) -> bool {
    let (masks, mask_count) = symmetry.active_masks();
    mesh.positions().iter().enumerate().any(|(v, p)| {
        let v = v as VertexId;
        if !mesh.is_vertex_alive(v) || (require_triangles && mesh.corner_count(v) == 0) {
            return false;
        }
        masks[..mask_count].iter().any(|&bits| {
            normalized_dist(filter::reflect_vec(bits, brush.center), brush.radius, p).is_some()
        })
    })
}

/// The mesh's cached incidence index, which `*_region` functions read fans from.
pub(crate) fn region_index(mesh: &Mesh) -> Result<&Incidence> {
    mesh.incidence().ok_or(OpsError::InvalidInput(
        "region deforms need Mesh::ensure_incidence and no topology edit in the transaction",
    ))
}

/// Writes `moved_to` over `ids`, once, if any vertex moved.
pub(crate) fn write_moves(
    tx: &mut Transaction,
    ids: &[VertexId],
    moved_to: &[[f32; 3]],
) -> Result<()> {
    if !ids.is_empty() {
        tx.set_positions(ids, moved_to)?;
    }
    Ok(())
}

/// Area-weighted smooth normal of `v` from its ascending incident `triangles`,
/// matching [`fast_mesh::compute_vertex_normals`] bit for bit.
pub(crate) fn vertex_normal_from_triangles(mesh: &Mesh, triangles: &[TriangleId]) -> [f32; 3] {
    let positions = mesh.positions();
    let mut n = [0.0f32; 3];
    for &t in triangles {
        if let Some([a, b, c]) = mesh.triangle(t) {
            let (pa, pb, pc) = (
                positions[a as usize],
                positions[b as usize],
                positions[c as usize],
            );
            let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            n[0] += e1[1] * e2[2] - e1[2] * e2[1];
            n[1] += e1[2] * e2[0] - e1[0] * e2[2];
            n[2] += e1[0] * e2[1] - e1[1] * e2[0];
        }
    }
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 0.0 {
        n.iter_mut().for_each(|x| *x /= len);
    }
    n
}

/// Clears `v` and fallibly ensures `v.capacity() >= len` (`v.len() == 0` right
/// after `clear()`, so `reserve(v, len)` requests exact capacity `0 + len`).
pub(crate) fn clear_and_reserve<T>(v: &mut Vec<T>, len: usize) -> Result<usize> {
    v.clear();
    if v.capacity() < len {
        crate::reserve(v, len)?;
    }
    debug_assert!(v.capacity() >= len);
    Ok(v.capacity())
}

/// Collects the ascending deduplicated 1-ring neighbors of `v` from proper
/// triangles in `triangles`.
pub(crate) fn collect_proper_neighbors(
    mesh: &Mesh,
    v: VertexId,
    triangles: &[TriangleId],
    out: &mut Vec<VertexId>,
) -> Result<()> {
    out.clear();
    for &t in triangles {
        if let Some(tri) = proper_triangle(mesh, t) {
            for u in tri {
                if u != v {
                    push(out, u)?;
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(())
}
