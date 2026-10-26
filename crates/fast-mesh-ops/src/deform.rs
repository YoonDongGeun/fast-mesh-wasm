//! Brush deformation: moving vertices along their normals under a falloff.
//!
//! [`inflate`] pushes the vertices under a [`Brush`] along their smooth
//! normals ([`compute_vertex_normals`](fast_mesh::compute_vertex_normals)),
//! scaled by the brush [`strength`](Brush::strength) and the
//! [`falloff`](Brush::falloff) weight of their distance from the brush
//! center. A negative strength pulls instead of pushing.

use fast_mesh::compute_vertex_normals;
use fast_mesh_store::{Mesh, Transaction, VertexId};

use crate::topology::VertexFans;
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

/// Whether any live vertex is under the brush: closer than `radius` to
/// `center`, the same membership test the region loops use.
fn has_region_member(mesh: &Mesh, brush: &Brush) -> bool {
    mesh.positions().iter().enumerate().any(|(v, p)| {
        if !mesh.is_vertex_alive(v as VertexId) {
            return false;
        }
        let dist = brush
            .center
            .iter()
            .zip(p)
            .map(|(c, p)| (p - c).powi(2))
            .sum::<f32>()
            .sqrt();
        let t = dist / brush.radius;
        t.is_finite() && t < 1.0
    })
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
    // Members first: with no live vertex under the brush the call is a
    // guaranteed no-op, so skip the whole-mesh temporaries below.
    if !has_region_member(mesh, brush) {
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

/// The shrinking Taubin step: each pass moves a vertex `w * λ` of the way
/// from its position toward its neighbors' centroid.
pub const TAUBIN_LAMBDA: f32 = 0.33;

/// The inflating Taubin step: each pass moves a vertex `w * μ` of the way
/// from its position toward its neighbors' centroid, which with the negative
/// `μ` pushes it back out past where the λ pass left it.
pub const TAUBIN_MU: f32 = -0.34;

/// How [`smooth`] relaxes: volume-preserving Taubin λ/μ cycles, or plain
/// Laplacian passes that melt toward the centroid (sculpt smoothing).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SmoothMethod {
    /// Taubin λ/μ cycles (the default): shrink then inflate back, keeping
    /// volume while removing noise. Converges slowly on low frequencies.
    #[default]
    Taubin,
    /// Plain Laplacian passes: each pass moves the full weight toward the
    /// centroid. Melts bumps fast, with some shrinkage.
    Laplacian,
}

/// Options for [`smooth`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SmoothOptions {
    /// The brush to apply.
    pub brush: Brush,
    /// How many cycles to run; 0 moves nothing.
    pub cycles: u32,
    /// How each cycle relaxes.
    pub method: SmoothMethod,
}

impl SmoothOptions {
    /// A builder for [`SmoothOptions`]: the default brush until set, 0 cycles.
    pub fn builder() -> SmoothOptionsBuilder {
        SmoothOptionsBuilder::default()
    }
}

/// A builder for [`SmoothOptions`], see [`SmoothOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct SmoothOptionsBuilder {
    options: SmoothOptions,
}

impl SmoothOptionsBuilder {
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

    /// Displacement scale; the per-vertex weight is `strength * falloff`.
    pub fn strength(mut self, strength: f32) -> Self {
        self.options.brush.strength = strength;
        self
    }

    /// How many cycles to run.
    pub fn cycles(mut self, cycles: u32) -> Self {
        self.options.cycles = cycles;
        self
    }

    /// How each cycle relaxes (Taubin λ/μ by default).
    pub fn method(mut self, method: SmoothMethod) -> Self {
        self.options.method = method;
        self
    }

    /// The options as set.
    pub fn build(self) -> SmoothOptions {
        self.options
    }
}

/// What [`smooth`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SmoothReport {
    /// How many vertices moved (displacement above 0).
    pub moved: usize,
    /// The largest displacement applied, in local units; 0 when nothing
    /// moved.
    pub max_displacement: f32,
}

/// Smooths the vertices under the brush, Taubin λ/μ or plain Laplacian.
///
/// - Region: every live vertex closer than `radius` to `center` (brute-force
///   distance check). Vertices outside it never move but still serve as
///   neighbor sources.
/// - Each of the `cycles` runs one pass per method constant (λ then μ for
///   [`SmoothMethod::Taubin` ([`TAUBIN_LAMBDA`], [`TAUBIN_MU`])), a single
///   full pass for [`SmoothMethod::Laplacian`]); every pass moves each region
///   vertex by `w * k * (centroid - p)`, where `w` is `strength * falloff`
///   at its normalized distance, `k` the pass constant, and `centroid` the
///   mean of its neighbor vertices (the deduped vertices of its fan
///   triangles, excluding itself). Both the moved positions and the neighbor
///   sources come from a snapshot taken before the pass.
/// - The move is written once, in a single `set_positions` call over the
///   vertices whose total displacement is above 0; an empty region writes
///   nothing. History is untouched.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when the radius is not positive, or any brush
///   number (center, radius, strength) is not finite.
/// - [`OpsError::Store`] when memory runs out.
///
/// Nothing is written in any of these cases.
pub fn smooth(tx: &mut Transaction, options: &SmoothOptions) -> Result<SmoothReport> {
    let brush = &options.brush;
    if !brush.radius.is_finite() || brush.radius <= 0.0 {
        return Err(OpsError::InvalidInput(
            "smooth brush radius must be positive and finite",
        ));
    }
    if brush.center.iter().any(|c| !c.is_finite()) {
        return Err(OpsError::InvalidInput("smooth brush center must be finite"));
    }
    if !brush.strength.is_finite() {
        return Err(OpsError::InvalidInput(
            "smooth brush strength must be finite",
        ));
    }
    let report = SmoothReport::default();
    if options.cycles == 0 {
        return Ok(report);
    }
    let mesh = tx.mesh();
    if mesh.vertex_slots() == 0 {
        return Ok(report);
    }
    // Members first: with no live vertex under the brush the call is a
    // guaranteed no-op, so skip the whole-mesh temporaries below.
    if !has_region_member(mesh, brush) {
        return Ok(report);
    }
    let fans = VertexFans::build(mesh)?;
    let positions = mesh.positions();
    // The region members with their `strength * falloff` weights.
    let mut members: Vec<(VertexId, f32)> = Vec::new();
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
        push(
            &mut members,
            (v, brush.strength * falloff_value(brush.falloff, t)),
        )?;
    }
    // Working positions, slot-aligned; only region members ever change.
    let mut current: Vec<[f32; 3]> = Vec::new();
    reserve(&mut current, positions.len())?;
    current.extend_from_slice(positions);
    let mut before: Vec<[f32; 3]> = Vec::new();
    let mut adjacent: Vec<VertexId> = Vec::new();
    let passes: &[f32] = match options.method {
        SmoothMethod::Taubin => &[TAUBIN_LAMBDA, TAUBIN_MU],
        SmoothMethod::Laplacian => &[1.0],
    };
    for _ in 0..options.cycles {
        for &k in passes {
            before.clear();
            let missing = current.len().saturating_sub(before.capacity());
            reserve(&mut before, missing)?;
            before.extend_from_slice(&current);
            for &(v, w) in &members {
                adjacent.clear();
                for &t in fans.fan(v) {
                    if let Some(tri) = mesh.triangle(t) {
                        for u in tri {
                            if u != v {
                                push(&mut adjacent, u)?;
                            }
                        }
                    }
                }
                if adjacent.is_empty() {
                    continue;
                }
                adjacent.sort_unstable();
                adjacent.dedup();
                let mut centroid = [0.0f32; 3];
                for &u in &adjacent {
                    for axis in 0..3 {
                        centroid[axis] += before[u as usize][axis];
                    }
                }
                for c in &mut centroid {
                    *c /= adjacent.len() as f32;
                }
                let p = before[v as usize];
                for axis in 0..3 {
                    current[v as usize][axis] = p[axis] + w * k * (centroid[axis] - p[axis]);
                }
            }
        }
    }
    let mut report = SmoothReport::default();
    let mut ids: Vec<VertexId> = Vec::new();
    let mut moved_to: Vec<[f32; 3]> = Vec::new();
    for &(v, _) in &members {
        let length = (0..3)
            .map(|k| (current[v as usize][k] - positions[v as usize][k]).powi(2))
            .sum::<f32>()
            .sqrt();
        if !length.is_finite() || length <= 0.0 {
            continue;
        }
        push(&mut ids, v)?;
        push(&mut moved_to, current[v as usize])?;
        report.max_displacement = report.max_displacement.max(length);
    }
    report.moved = ids.len();
    if !ids.is_empty() {
        tx.set_positions(&ids, &moved_to)?;
    }
    Ok(report)
}

/// Options for [`grab`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GrabOptions {
    /// The brush to apply.
    pub brush: Brush,
    /// The drag vector in local coordinates: each region vertex moves
    /// `strength * falloff * direction`. Must be finite; a zero vector moves
    /// nothing.
    pub direction: [f32; 3],
}

impl GrabOptions {
    /// A builder for [`GrabOptions`]: the default brush until set, no drag.
    pub fn builder() -> GrabOptionsBuilder {
        GrabOptionsBuilder::default()
    }
}

/// A builder for [`GrabOptions`], see [`GrabOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct GrabOptionsBuilder {
    options: GrabOptions,
}

impl GrabOptionsBuilder {
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

    /// Displacement scale; the per-vertex offset is `strength * falloff *
    /// direction`.
    pub fn strength(mut self, strength: f32) -> Self {
        self.options.brush.strength = strength;
        self
    }

    /// The drag vector in local coordinates.
    pub fn direction(mut self, direction: [f32; 3]) -> Self {
        self.options.direction = direction;
        self
    }

    /// The options as set.
    pub fn build(self) -> GrabOptions {
        self.options
    }
}

/// What [`grab`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GrabReport {
    /// How many vertices moved (displacement above 0).
    pub moved: usize,
    /// The largest displacement applied, in local units; 0 when nothing
    /// moved.
    pub max_displacement: f32,
}

/// Drags the vertices under the brush along one vector.
///
/// - Region: every live vertex closer than `radius` to `center` (brute-force
///   distance check), each displaced by `strength * falloff * direction`,
///   where `falloff` is the [`Falloff`] weight of its normalized distance.
///   Unlike [`inflate`], the direction is the caller's, not the normals'.
/// - The move is written once, in a single `set_positions` call over the
///   vertices whose displacement is above 0; an empty region, or a zero
///   `direction`, writes nothing. History is untouched.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when the radius is not positive, or any brush
///   number (center, radius, strength) or the direction is not finite.
/// - [`OpsError::Store`] when memory runs out.
///
/// Nothing is written in any of these cases.
pub fn grab(tx: &mut Transaction, options: &GrabOptions) -> Result<GrabReport> {
    let brush = &options.brush;
    if !brush.radius.is_finite() || brush.radius <= 0.0 {
        return Err(OpsError::InvalidInput(
            "grab brush radius must be positive and finite",
        ));
    }
    if brush.center.iter().any(|c| !c.is_finite()) {
        return Err(OpsError::InvalidInput("grab brush center must be finite"));
    }
    if !brush.strength.is_finite() {
        return Err(OpsError::InvalidInput("grab brush strength must be finite"));
    }
    if options.direction.iter().any(|c| !c.is_finite()) {
        return Err(OpsError::InvalidInput("grab direction must be finite"));
    }
    let mesh = tx.mesh();
    let mut report = GrabReport::default();
    if mesh.vertex_slots() == 0 {
        return Ok(report);
    }
    let positions = mesh.positions();
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
        let displaced = [
            p[0] + w * options.direction[0],
            p[1] + w * options.direction[1],
            p[2] + w * options.direction[2],
        ];
        let length = options
            .direction
            .iter()
            .map(|d| (w * d).powi(2))
            .sum::<f32>()
            .sqrt();
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
    use crate::fail_allocations_after;
    use crate::test_util::{edit, mesh_of, parts, sweep};

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

    fn smooth_options(
        center: [f32; 3],
        radius: f32,
        falloff: Falloff,
        strength: f32,
        cycles: u32,
    ) -> SmoothOptions {
        SmoothOptions::builder()
            .center(center)
            .radius(radius)
            .falloff(falloff)
            .strength(strength)
            .cycles(cycles)
            .build()
    }

    #[test]
    fn taubin_pyramid_exact_after_one_cycle() {
        assert_eq!(TAUBIN_LAMBDA, 0.33);
        assert_eq!(TAUBIN_MU, -0.34);
        // Open pyramid: base A(0,0,0) B(1,0,0) C(1,1,0) D(0,1,0), apex
        // E(0.5,0.5,1); 4 side triangles. `VertexFans::build` only needs
        // proper triangles, so the open base needs no closing: E's fan is
        // the 4 side triangles either way.
        let positions = [
            0.0, 0.0, 0.0, // A
            1.0, 0.0, 0.0, // B
            1.0, 1.0, 0.0, // C
            0.0, 1.0, 0.0, // D
            0.5, 0.5, 1.0, // E
        ];
        let triangles = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]];
        let mut mesh = mesh_of(&positions, &triangles);
        let options = smooth_options([0.5, 0.5, 1.0], 0.1, Falloff::Linear, 1.0, 1);
        let (report, empty) = edit(&mut mesh, |tx| smooth(tx, &options));
        assert!(!empty);
        assert_eq!(report.moved, 1);
        let (flat, _) = parts(&mesh);
        // E's neighbors (A-D) stay fixed outside the region, centroid
        // (0.5,0.5,0): λ pass z = 1 + 0.33 * (0 - 1) = 0.67, then μ pass
        // z = 0.67 - 0.34 * (0 - 0.67) = 0.8978.
        assert!((flat[4 * 3 + 2] - 0.8978).abs() < 1e-4);
        assert_eq!(flat[4 * 3], 0.5);
        assert_eq!(flat[4 * 3 + 1], 0.5);
        for v in 0..4 {
            assert_eq!(&flat[v * 3..v * 3 + 3], &positions[v * 3..v * 3 + 3]);
        }
        assert!((report.max_displacement - (1.0 - 0.8978)).abs() < 1e-4);
    }

    #[test]
    fn laplacian_pyramid_melts_to_centroid() {
        assert_eq!(
            SmoothOptions::builder().build().method,
            SmoothMethod::Taubin
        );
        // Same open pyramid as the Taubin test: apex E(0.5,0.5,1) over base
        // centroid (0.5,0.5,0).
        let positions = [
            0.0, 0.0, 0.0, // A
            1.0, 0.0, 0.0, // B
            1.0, 1.0, 0.0, // C
            0.0, 1.0, 0.0, // D
            0.5, 0.5, 1.0, // E
        ];
        let triangles = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]];
        let mut mesh = mesh_of(&positions, &triangles);
        let options = SmoothOptions::builder()
            .center([0.5, 0.5, 1.0])
            .radius(0.1)
            .falloff(Falloff::Linear)
            .strength(1.0)
            .cycles(1)
            .method(SmoothMethod::Laplacian)
            .build();
        let (report, empty) = edit(&mut mesh, |tx| smooth(tx, &options));
        assert!(!empty);
        assert_eq!(report.moved, 1);
        let (flat, _) = parts(&mesh);
        // One full pass at weight 1: E lands exactly on its centroid.
        assert_eq!(&flat[4 * 3..4 * 3 + 3], &[0.5, 0.5, 0.0]);
        assert!((report.max_displacement - 1.0).abs() < 1e-6);
        // Half strength, two passes: 1 -> 0.5 -> 0.25.
        let mut mesh = mesh_of(&positions, &triangles);
        let options = SmoothOptions::builder()
            .center([0.5, 0.5, 1.0])
            .radius(0.1)
            .falloff(Falloff::Linear)
            .strength(0.5)
            .cycles(2)
            .method(SmoothMethod::Laplacian)
            .build();
        edit(&mut mesh, |tx| smooth(tx, &options).map(drop));
        let (flat, _) = parts(&mesh);
        assert!((flat[4 * 3 + 2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn smooth_zero_cycles_moves_nothing() {
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = smooth_options([0.0; 3], 10.0, Falloff::Linear, 1.0, 0);
        let mut tx = mesh.begin_edit();
        let report = smooth(&mut tx, &options).unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn smooth_empty_region_moves_nothing() {
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = smooth_options([100.0, 0.0, 0.0], 1.0, Falloff::Linear, 1.0, 1);
        let mut tx = mesh.begin_edit();
        let report = smooth(&mut tx, &options).unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn smooth_invalid_brush_rejected() {
        let mut mesh = centered_grid();
        let cases = [
            smooth_options([0.0; 3], 0.0, Falloff::Linear, 1.0, 1),
            smooth_options([0.0; 3], -1.0, Falloff::Linear, 1.0, 1),
            smooth_options([0.0; 3], f32::NAN, Falloff::Linear, 1.0, 1),
            smooth_options([0.0; 3], 10.0, Falloff::Linear, f32::NAN, 1),
            smooth_options([f32::INFINITY, 0.0, 0.0], 10.0, Falloff::Linear, 1.0, 1),
        ];
        for options in cases {
            let mut tx = mesh.begin_edit();
            assert!(matches!(
                smooth(&mut tx, &options),
                Err(OpsError::InvalidInput(_))
            ));
            assert!(tx.commit().unwrap().is_empty());
        }
    }

    #[test]
    fn smooth_nonmanifold_fan_moves_toward_centroid() {
        // Two triangles sharing only vertex E(0), with disjoint other
        // vertices: E's neighbors are all four of them.
        let positions = [
            0.0, 0.0, 1.0, // E
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        ];
        let triangles = [[0, 1, 2], [0, 3, 4]];
        let mut mesh = mesh_of(&positions, &triangles);
        let options = smooth_options([0.0, 0.0, 1.0], 0.1, Falloff::Linear, 1.0, 1);
        let (report, _) = edit(&mut mesh, |tx| smooth(tx, &options));
        assert_eq!(report.moved, 1);
        let (flat, _) = parts(&mesh);
        let dist = |p: &[f32]| ((p[0] - 0.5).powi(2) + (p[1] - 0.5).powi(2) + p[2].powi(2)).sqrt();
        assert!(dist(&flat[0..3]) < dist(&positions[0..3]));
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

    #[test]
    fn inflate_empty_region_skips_temporaries() {
        // No region member: no fallible reservation may run at all.
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = options([100.0, 0.0, 0.0], 1.0, Falloff::Linear, 2.0);
        let mut tx = mesh.begin_edit();
        fail_allocations_after(Some(0));
        let report = inflate(&mut tx, &options);
        fail_allocations_after(None);
        let report = report.unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn smooth_empty_region_skips_temporaries() {
        // No region member: no fallible reservation may run at all.
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = smooth_options([100.0, 0.0, 0.0], 1.0, Falloff::Linear, 1.0, 1);
        let mut tx = mesh.begin_edit();
        fail_allocations_after(Some(0));
        let report = smooth(&mut tx, &options);
        fail_allocations_after(None);
        let report = report.unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn inflate_out_of_memory_is_reported_and_writes_nothing() {
        let options = options([0.0; 3], 10.0, Falloff::Linear, 2.0);
        assert!(sweep(centered_grid(), |tx| inflate(tx, &options).map(drop)) > 5);
    }

    #[test]
    fn smooth_out_of_memory_is_reported_and_writes_nothing() {
        let options = smooth_options([0.0; 3], 10.0, Falloff::Linear, 1.0, 1);
        assert!(sweep(centered_grid(), |tx| smooth(tx, &options).map(drop)) > 5);
    }

    fn grab_options(
        center: [f32; 3],
        radius: f32,
        falloff: Falloff,
        strength: f32,
        direction: [f32; 3],
    ) -> GrabOptions {
        GrabOptions::builder()
            .center(center)
            .radius(radius)
            .falloff(falloff)
            .strength(strength)
            .direction(direction)
            .build()
    }

    #[test]
    fn grab_moves_region_along_direction() {
        let mut mesh = centered_grid();
        let old = parts(&mesh).0;
        let options = grab_options([0.0; 3], 10.0, Falloff::Linear, 1.0, [2.0, 0.0, 0.0]);
        let (report, empty) = edit(&mut mesh, |tx| grab(tx, &options));
        assert!(!empty);
        assert_eq!(report.moved, 9);
        assert!((report.max_displacement - 2.0).abs() < 1e-4);
        let new = parts(&mesh).0;
        for k in 0..3 {
            let want = [2.0, 0.0, 0.0][k];
            assert!(
                (new[4 * 3 + k] - (old[4 * 3 + k] + want)).abs() < 1e-4,
                "center follows strength * direction"
            );
        }
        assert!(displacement(&old, &new, 0) < displacement(&old, &new, 4));
    }

    #[test]
    fn grab_strength_scales_the_drag() {
        let mut mesh = centered_grid();
        let old = parts(&mesh).0;
        let options = grab_options([0.0; 3], 10.0, Falloff::Linear, 0.5, [2.0, 0.0, 0.0]);
        let (report, _) = edit(&mut mesh, |tx| grab(tx, &options));
        assert_eq!(report.moved, 9);
        let new = parts(&mesh).0;
        assert!((new[4 * 3] - (old[4 * 3] + 1.0)).abs() < 1e-4);
    }

    #[test]
    fn grab_zero_direction_moves_nothing() {
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = grab_options([0.0; 3], 10.0, Falloff::Linear, 1.0, [0.0; 3]);
        let mut tx = mesh.begin_edit();
        let report = grab(&mut tx, &options).unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn grab_empty_region_moves_nothing() {
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = grab_options(
            [100.0, 0.0, 0.0],
            1.0,
            Falloff::Linear,
            1.0,
            [2.0, 0.0, 0.0],
        );
        let mut tx = mesh.begin_edit();
        let report = grab(&mut tx, &options).unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn grab_invalid_brush_rejected() {
        let mut mesh = centered_grid();
        let cases = [
            grab_options([0.0; 3], 0.0, Falloff::Linear, 1.0, [1.0, 0.0, 0.0]),
            grab_options([0.0; 3], -1.0, Falloff::Linear, 1.0, [1.0, 0.0, 0.0]),
            grab_options([0.0; 3], f32::NAN, Falloff::Linear, 1.0, [1.0, 0.0, 0.0]),
            grab_options([0.0; 3], 10.0, Falloff::Linear, f32::NAN, [1.0, 0.0, 0.0]),
            grab_options(
                [f32::INFINITY, 0.0, 0.0],
                10.0,
                Falloff::Linear,
                1.0,
                [1.0, 0.0, 0.0],
            ),
            grab_options([0.0; 3], 10.0, Falloff::Linear, 1.0, [f32::NAN, 0.0, 0.0]),
            grab_options(
                [0.0; 3],
                10.0,
                Falloff::Linear,
                1.0,
                [0.0, f32::INFINITY, 0.0],
            ),
        ];
        for options in cases {
            let mut tx = mesh.begin_edit();
            assert!(matches!(
                grab(&mut tx, &options),
                Err(OpsError::InvalidInput(_))
            ));
            assert!(tx.commit().unwrap().is_empty());
        }
    }

    #[test]
    fn grab_empty_region_skips_temporaries() {
        // No region member: no fallible reservation may run at all.
        let mut mesh = centered_grid();
        let before = parts(&mesh);
        let options = grab_options(
            [100.0, 0.0, 0.0],
            1.0,
            Falloff::Linear,
            1.0,
            [2.0, 0.0, 0.0],
        );
        let mut tx = mesh.begin_edit();
        fail_allocations_after(Some(0));
        let report = grab(&mut tx, &options);
        fail_allocations_after(None);
        let report = report.unwrap();
        assert_eq!(report.moved, 0);
        assert_eq!(report.max_displacement, 0.0);
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn grab_out_of_memory_is_reported_and_writes_nothing() {
        let options = grab_options([0.0; 3], 10.0, Falloff::Linear, 1.0, [2.0, 0.0, 0.0]);
        assert!(sweep(centered_grid(), |tx| grab(tx, &options).map(drop)) > 5);
    }
}
