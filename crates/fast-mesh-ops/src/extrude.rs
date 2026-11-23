//! Extrusion: pushing a face selection out with side walls, and running an
//! open mesh's boundary down to a flat, capped base.
//!
//! - [`extrude_faces`] moves the selected triangles by a distance, along
//!   each region's average normal, each vertex's average normal or one
//!   direction ([`ExtrudeDirection`]), and builds a wall of two triangles on
//!   every edge of the selection's boundary, so that a closed mesh stays
//!   closed and consistently wound. A selection with holes gets walls around
//!   every hole too, and several regions (even ones touching at a vertex)
//!   each get their own walls. The selected triangles keep their ids: they
//!   are the moved caps.
//! - [`extrude_base`] runs boundary loops ([`BaseLoops`]) along a direction
//!   to a plane beyond them ([`BaseLevel`]) and closes each with a flat cap,
//!   refined to the density of its walls, turning an open surface into a
//!   solid. Nested outlines on the plane become holes in the cap. The solid
//!   is turned outward when the surface faced the other way.
//!
//! Both plan everything before writing and check the result for
//! self-intersections first. Every pair that involves a new or moved
//! triangle is looked at: triangles that touch or cross and share no vertex,
//! by the exact test of [`fast_mesh_query::self_intersecting_pairs`], and
//! triangles sharing an edge that fold flat onto each other (within about
//! 1e-3 radians), such as a wall pushed straight back along the face next to
//! it. Their triangles are reported (`self_intersections`) and the result is
//! written all the same. Pairs the operation leaves alone are not listed,
//! and triangles that share just one vertex are not tested against each
//! other. The check looks only at the triangles near the changed ones, so
//! past one scan of the mesh its cost follows the size of the change.
//!
//! UVs: new vertices take their parent vertex's values (the wall's rim
//! repeats them), and new cap vertices interpolate them; see
//! `docs/uv-channel.md`.

mod base;
mod check;
mod faces;
#[cfg(test)]
mod tests;

pub use base::{BaseLevel, BaseLoops, ExtrudeBaseOptions, ExtrudeBaseReport, extrude_base};
pub use faces::{ExtrudeDirection, ExtrudeFacesOptions, ExtrudeFacesReport, extrude_faces};

/// The identity linear map, row-major.
pub(crate) const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// `m * v` for a row-major `m`.
pub(crate) fn apply(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// The cofactor matrix of `m`, `det(m) * m^-T`: it maps the cross product of
/// two vectors to the cross product of their images under `m`, so it carries
/// a triangle's area vector (and its winding) through `m`.
pub(crate) fn cofactor(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c =
        |r0: usize, r1: usize, c0: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    [
        [c(1, 2, 1, 2), -c(1, 2, 0, 2), c(1, 2, 0, 1)],
        [-c(0, 2, 1, 2), c(0, 2, 0, 2), -c(0, 2, 0, 1)],
        [c(0, 1, 1, 2), -c(0, 1, 0, 2), c(0, 1, 0, 1)],
    ]
}

/// The inverse of `m`, or `None` when it is singular or not finite.
pub(crate) fn inverse(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let cof = cofactor(m);
    let det = m[0][0] * cof[0][0] + m[0][1] * cof[0][1] + m[0][2] * cof[0][2];
    if !det.is_finite() || det == 0.0 {
        return None;
    }
    // m^-1 = cof^T / det.
    let inv: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| cof[j][i] / det));
    inv.iter().flatten().all(|x| x.is_finite()).then_some(inv)
}

/// `v` scaled to unit length, or `None` when it has no finite direction.
pub(crate) fn unit(v: [f64; 3]) -> Option<[f64; 3]> {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (len.is_finite() && len > 0.0).then(|| v.map(|x| x / len))
}

/// A local position as stored: `f32`, every coordinate finite.
pub(crate) fn to_f32(p: [f64; 3]) -> Option<[f32; 3]> {
    let q = p.map(|x| x as f32);
    q.iter().all(|x| x.is_finite()).then_some(q)
}
