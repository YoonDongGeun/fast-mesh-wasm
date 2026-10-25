//! Brush deformation: moving vertices along their normals under a falloff.
//!
//! [`inflate`] pushes the vertices under a [`Brush`] along their smooth
//! normals ([`compute_vertex_normals`](fast_mesh::compute_vertex_normals)),
//! scaled by the brush [`strength`](Brush::strength) and the
//! [`falloff`](Brush::falloff) weight of their distance from the brush
//! center. A negative strength pulls instead of pushing.

use fast_mesh::compute_vertex_normals;
use fast_mesh_store::{Transaction, VertexId};

use crate::{OpsError, Result, push, reserve};

/// How a brush's effect fades from its center (`t = 0`) to its rim (`t = 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Falloff {
    /// Linear decay: `1 - t`.
    Linear,
    /// Smoothstep decay: `s = 1 - t`, then `s * s * (3 - 2 * s)`.
    Smooth,
}

/// The falloff weight at normalized distance `t` (0 at the center, 1 at the
/// rim). Callers keep `t` below 1; at and past the rim the brush has no
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

/// A deform brush: a sphere of influence with a falloff and a strength.
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

/// Options for [`inflate`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct InflateOptions {
    /// The brush to apply.
    pub brush: Brush,
}

impl InflateOptions {
    /// A builder for [`InflateOptions`]: the default brush until set.
    pub fn builder() -> InflateOptionsBuilder {
        InflateOptionsBuilder::default()
    }
}

/// A builder for [`InflateOptions`], see [`InflateOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct InflateOptionsBuilder {
    options: InflateOptions,
}

impl InflateOptionsBuilder {
    /// Center of the brush in local coordinates.
    pub fn center(mut self, center: [f32; 3]) -> Self {
        self.options.brush.center = center;
        self
    }

    /// Radius of influence in local units.
    pub fn radius(mut self, radius: f32) -> Self {
        self.options.brush.radius = radius;
        self
    }

    /// How the effect fades with distance from the center.
    pub fn falloff(mut self, falloff: Falloff) -> Self {
        self.options.brush.falloff = falloff;
        self
    }

    /// Displacement scale in local units; negative pulls instead of pushing.
    pub fn strength(mut self, strength: f32) -> Self {
        self.options.brush.strength = strength;
        self
    }

    /// The options as set.
    pub fn build(self) -> InflateOptions {
        self.options
    }
}

/// What [`inflate`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct InflateReport {
    /// How many vertices moved (displacement above 0).
    pub moved: usize,
    /// The largest displacement applied, in local units; 0 when nothing
    /// moved.
    pub max_displacement: f32,
}

/// Pushes the vertices under the brush along their smooth normals.
///
/// - Region: every live vertex closer than `radius` to `center` (brute-force
///   distance check), each displaced by `strength * falloff * normal`, where
///   `falloff` is the [`Falloff`] weight of its normalized distance and
///   `normal` its smooth normal.
/// - The move is written once, in a single `set_positions` call over the
///   vertices whose displacement is above 0; an empty region writes nothing.
///   History is untouched.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when the radius is not positive, or any brush
///   number (center, radius, strength) is not finite.
/// - [`OpsError::Store`] when memory runs out.
///
/// Nothing is written in any of these cases.
pub fn inflate(tx: &mut Transaction, options: &InflateOptions) -> Result<InflateReport> {
    let brush = &options.brush;
    if !brush.radius.is_finite() || brush.radius <= 0.0 {
        return Err(OpsError::InvalidInput(
            "inflate brush radius must be positive and finite",
        ));
    }
    if brush.center.iter().any(|c| !c.is_finite()) {
        return Err(OpsError::InvalidInput(
            "inflate brush center must be finite",
        ));
    }
    if !brush.strength.is_finite() {
        return Err(OpsError::InvalidInput(
            "inflate brush strength must be finite",
        ));
    }
    let mesh = tx.mesh();
    let mut report = InflateReport::default();
    if mesh.vertex_slots() == 0 {
        return Ok(report);
    }
    // Slot-aligned flat positions and the alive triangles, the layout
    // `compute_vertex_normals` takes; normals come back slot-aligned too.
    let positions = mesh.positions();
    let mut flat: Vec<f32> = Vec::new();
    reserve(&mut flat, positions.len() * 3)?;
    flat.extend(positions.iter().flatten().copied());
    let mut indices: Vec<u32> = Vec::new();
    reserve(&mut indices, mesh.triangle_slots() * 3)?;
    for t in 0..mesh.triangle_slots() as u32 {
        if let Some(tri) = mesh.triangle(t) {
            indices.extend(tri);
        }
    }
    let normals = compute_vertex_normals(&flat, &indices)
        .map_err(|_| OpsError::InvalidInput("mesh buffers are not a valid mesh"))?;
    let mut ids: Vec<VertexId> = Vec::new();
    let mut moved_to: Vec<[f32; 3]> = Vec::new();
    for (v, p) in positions.iter().enumerate() {
        let v = v as VertexId;
        if !mesh.is_vertex_alive(v) {
            continue;
        }
        let dist = brush
            .center
            .iter()
            .zip(p)
            .map(|(c, p)| (p - c).powi(2))
            .sum::<f32>()
            .sqrt();
        let t = dist / brush.radius;
        if !t.is_finite() || t >= 1.0 {
            continue;
        }
        let w = brush.strength * falloff_value(brush.falloff, t);
        let n = [
            normals[3 * v as usize],
            normals[3 * v as usize + 1],
            normals[3 * v as usize + 2],
        ];
        let displaced = [p[0] + w * n[0], p[1] + w * n[1], p[2] + w * n[2]];
        let length = n.iter().map(|n| (w * n).powi(2)).sum::<f32>().sqrt();
        if !length.is_finite() || length <= 0.0 {
            continue;
        }
        push(&mut ids, v)?;
        push(&mut moved_to, displaced)?;
        report.max_displacement = report.max_displacement.max(length);
    }
    report.moved = ids.len();
    if !ids.is_empty() {
        tx.set_positions(&ids, &moved_to)?;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use fast_mesh::compute_vertex_normals;
    use fast_mesh_store::Mesh;

    use super::*;
    use crate::OpsError;
    use crate::test_util::{edit, mesh_of, parts};

    /// A 3x3 flat grid in `z = 0`, spacing 1, centered at the origin; vertex
    /// `(x, y)` has id `y * 3 + x`, wound counter-clockwise from `+z`.
    fn centered_grid() -> Mesh {
        let mut positions = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                positions.extend([x as f32 - 1.0, y as f32 - 1.0, 0.0]);
            }
        }
        let mut triangles = Vec::new();
        for y in 0..2u32 {
            for x in 0..2u32 {
                let a = y * 3 + x;
                triangles.extend([[a, a + 1, a + 4], [a, a + 4, a + 3]]);
            }
        }
        mesh_of(&positions, &triangles)
    }

    /// The smooth normal of the center vertex (id 4), straight from
    /// `compute_vertex_normals`, whatever its sign convention.
    fn center_normal(mesh: &Mesh) -> [f32; 3] {
        let (positions, triangles) = parts(mesh);
        let indices: Vec<u32> = triangles.iter().flatten().copied().collect();
        let normals = compute_vertex_normals(&positions, &indices).unwrap();
        [normals[4 * 3], normals[4 * 3 + 1], normals[4 * 3 + 2]]
    }

    fn options(center: [f32; 3], radius: f32, falloff: Falloff, strength: f32) -> InflateOptions {
        InflateOptions::builder()
            .center(center)
            .radius(radius)
            .falloff(falloff)
            .strength(strength)
            .build()
    }

    /// Displacement length of vertex `id` between two flat position buffers.
    fn displacement(old: &[f32], new: &[f32], id: usize) -> f32 {
        (0..3)
            .map(|k| (new[id * 3 + k] - old[id * 3 + k]).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn falloff_exact_values() {
        assert!((falloff_value(Falloff::Linear, 0.5) - 0.5).abs() < 1e-6);
        assert!((falloff_value(Falloff::Smooth, 0.0) - 1.0).abs() < 1e-6);
        assert!((falloff_value(Falloff::Smooth, 1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn falloff_boundary_values() {
        assert!((falloff_value(Falloff::Linear, 0.0) - 1.0).abs() < 1e-6);
        assert!((falloff_value(Falloff::Linear, 1.0) - 0.0).abs() < 1e-6);
        assert!((falloff_value(Falloff::Smooth, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn inflate_moves_center_along_normal() {
        let mut mesh = centered_grid();
        let old = parts(&mesh).0;
        let n = center_normal(&mesh);
        let options = options([0.0; 3], 10.0, Falloff::Linear, 2.0);
        let (report, empty) = edit(&mut mesh, |tx| inflate(tx, &options));
        assert!(!empty);
        assert_eq!(report.moved, 9);
        assert!((report.max_displacement - 2.0).abs() < 1e-4);
        let new = parts(&mesh).0;
        for k in 0..3 {
            assert!(
                (new[4 * 3 + k] - (old[4 * 3 + k] + 2.0 * n[k])).abs() < 1e-4,
                "center follows strength * normal"
            );
        }
        assert!(displacement(&old, &new, 0) < displacement(&old, &new, 4));
    }

    #[test]
    fn inflate_negative_strength_deflates() {
        let mut mesh = centered_grid();
        let old = parts(&mesh).0;
        let n = center_normal(&mesh);
        let options = options([0.0; 3], 10.0, Falloff::Linear, -2.0);
        let (report, empty) = edit(&mut mesh, |tx| inflate(tx, &options));
        assert!(!empty);
        assert_eq!(report.moved, 9);
        assert!((report.max_displacement - 2.0).abs() < 1e-4);
        let new = parts(&mesh).0;
        for k in 0..3 {
            assert!(
                (new[4 * 3 + k] - (old[4 * 3 + k] - 2.0 * n[k])).abs() < 1e-4,
                "negative strength pulls along the normal"
            );
        }
    }

    #[test]
    fn inflate_empty_region_moves_nothing() {
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = options([100.0, 0.0, 0.0], 1.0, Falloff::Linear, 2.0);
        let mut tx = mesh.begin_edit();
        let report = inflate(&mut tx, &options).unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn inflate_zero_radius_rejected() {
        let mut mesh = centered_grid();
        let options = options([0.0; 3], 0.0, Falloff::Linear, 2.0);
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            inflate(&mut tx, &options),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(tx.commit().unwrap().is_empty());
    }

    #[test]
    fn inflate_negative_radius_rejected() {
        let mut mesh = centered_grid();
        let options = options([0.0; 3], -1.0, Falloff::Linear, 2.0);
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            inflate(&mut tx, &options),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(tx.commit().unwrap().is_empty());
    }

    #[test]
    fn inflate_nan_strength_rejected() {
        let mut mesh = centered_grid();
        let options = options([0.0; 3], 10.0, Falloff::Linear, f32::NAN);
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            inflate(&mut tx, &options),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(tx.commit().unwrap().is_empty());
    }

    #[test]
    fn inflate_nonfinite_center_rejected() {
        let mut mesh = centered_grid();
        let options = options([f32::INFINITY, 0.0, 0.0], 10.0, Falloff::Linear, 2.0);
        let mut tx = mesh.begin_edit();
        assert!(matches!(
            inflate(&mut tx, &options),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(tx.commit().unwrap().is_empty());
    }
}
