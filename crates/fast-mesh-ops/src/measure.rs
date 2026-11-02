//! Scalar measurements of a mesh: surface area and enclosed volume.
//!
//! [`measure`] reports the total triangle area of `mesh` (always defined)
//! and, when the mesh is watertight, its enclosed volume. Watertight here
//! reuses [`analyze`](crate::analyze): no boundary loop or open chain and
//! no non-manifold edge. An open or non-manifold mesh still gets its area;
//! only `volume` is `None`, with [`VolumeReason`] saying why.
//!
//! Volume follows the tolerant convention of this crate: it is the absolute
//! value of the signed divergence-theorem sum, `|sum dot(p, cross(q,
//! r)) / 6|`, so the sign never depends on whether the mesh is wound
//! inward or outward. A mesh with inconsistent winding
//! (`orientation_conflicts > 0`, for example from flipped triangles) still
//! reports a volume, but the opposing triangles partly cancel, so the
//! number is unreliable; the conflict count flags that. This favours
//! returning a number with a warning over failing on slightly broken
//! inputs; callers that need strictness should require
//! `orientation_conflicts == 0` themselves. Duplicate triangles (the same
//! vertex set twice) count twice, like any other triangle.
//!
//! Positions are read as the stored `f32` local coordinates and all
//! geometry is computed in `f64`, like everywhere in this crate. Area and
//! volume are translation invariant, so the origin only matters to
//! [`measure_placed`], which maps each mesh point (origin plus local
//! position) through a caller closure, typically a placement, before
//! accumulating: the report is then in world units and exact under any
//! positive scale, including non-uniform ones.

use fast_mesh_store::{Mesh, VertexId};

use crate::analyze::{Analysis, AnalyzeOptions, analyze};
use crate::{Result, allocation_allowed};

/// Why [`MeasureReport::volume`] is `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeReason {
    /// The mesh has a boundary: at least one loop or open chain.
    Open,
    /// The mesh has at least one non-manifold edge (reported first when
    /// the mesh is both open and non-manifold).
    NonManifold,
}

/// Scalar measurements of a mesh, see [`measure`].
#[derive(Debug, Clone, PartialEq)]
pub struct MeasureReport {
    /// The total triangle area.
    pub area: f64,
    /// The signed divergence-theorem sum, `sum dot(p, cross(q, r)) / 6`.
    /// Its sign follows the winding (negative for inward-wound closed
    /// meshes); meaningless when the mesh is not watertight, since the sum
    /// then depends on where the missing caps would be.
    pub signed_volume: f64,
    /// The enclosed volume, `|signed_volume|`, when the mesh is watertight
    /// ([`closed`](MeasureReport::closed)); `None` otherwise, with
    /// [`volume_reason`](MeasureReport::volume_reason) saying why.
    pub volume: Option<f64>,
    /// No boundary loop or open chain and no non-manifold edge.
    pub closed: bool,
    /// Why `volume` is `None`; `None` exactly when `closed`.
    pub volume_reason: Option<VolumeReason>,
    /// The boundary loops and open chains, as [`analyze`] reports them.
    pub boundary_loops: usize,
    /// How many edges have three or more triangles.
    pub non_manifold_edges: usize,
    /// How many interior edges are used in the same direction by both
    /// their triangles (flipped winding nearby makes `volume` unreliable).
    pub orientation_conflicts: usize,
}

/// The watertight verdict and defect counts behind a [`MeasureReport`].
struct Verdict {
    closed: bool,
    volume_reason: Option<VolumeReason>,
    boundary_loops: usize,
    non_manifold_edges: usize,
    orientation_conflicts: usize,
}

/// The verdict from an [`Analysis`]: watertight exactly when there is no
/// boundary and no non-manifold edge.
fn verdict(a: &Analysis) -> Verdict {
    let closed = a.loops.is_empty() && a.non_manifold_edges == 0;
    let volume_reason = if closed {
        None
    } else if a.non_manifold_edges > 0 {
        Some(VolumeReason::NonManifold)
    } else {
        Some(VolumeReason::Open)
    };
    Verdict {
        closed,
        volume_reason,
        boundary_loops: a.loops.len(),
        non_manifold_edges: a.non_manifold_edges,
        orientation_conflicts: a.orientation_conflicts,
    }
}

/// The area and signed volume of the alive triangles, each mapped through
/// `place` first. Triangles with a repeated vertex id contribute nothing.
fn accumulate(
    mesh: &Mesh,
    origin: [f64; 3],
    mut place: impl FnMut([f64; 3]) -> [f64; 3],
) -> Result<(f64, f64)> {
    allocation_allowed()?;
    let positions = mesh.positions();
    let mut point = |v: VertexId| {
        let p = positions[v as usize];
        place([
            origin[0] + f64::from(p[0]),
            origin[1] + f64::from(p[1]),
            origin[2] + f64::from(p[2]),
        ])
    };
    let (mut area, mut signed) = (0.0, 0.0);
    for t in 0..mesh.triangle_slots() as u32 {
        let Some([a, b, c]) = mesh.triangle(t) else {
            continue;
        };
        if a == b || b == c || a == c {
            continue;
        }
        let (p, q, r) = (point(a), point(b), point(c));
        let u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let v = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        area += 0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        // Divergence theorem against the origin: translation invariant
        // for closed meshes, like the area.
        let cross = [
            q[1] * r[2] - q[2] * r[1],
            q[2] * r[0] - q[0] * r[2],
            q[0] * r[1] - q[1] * r[0],
        ];
        signed += (p[0] * cross[0] + p[1] * cross[1] + p[2] * cross[2]) / 6.0;
    }
    Ok((area, signed))
}

/// Builds a [`MeasureReport`] from accumulated sums and a verdict.
fn report(area: f64, signed_volume: f64, v: Verdict) -> MeasureReport {
    MeasureReport {
        area,
        signed_volume,
        volume: v.closed.then(|| signed_volume.abs()),
        closed: v.closed,
        volume_reason: v.volume_reason,
        boundary_loops: v.boundary_loops,
        non_manifold_edges: v.non_manifold_edges,
        orientation_conflicts: v.orientation_conflicts,
    }
}

/// Measures `mesh` in its local units: total area always, enclosed volume
/// when watertight. See the [module docs](self) for the tolerant volume
/// convention.
///
/// # Errors
/// [`OpsError::Store`] with [`StoreError::OutOfMemory`] when a temporary
/// cannot grow.
///
/// [`OpsError::Store`]: crate::OpsError::Store
/// [`StoreError::OutOfMemory`]: fast_mesh_store::StoreError::OutOfMemory
pub fn measure(mesh: &Mesh) -> Result<MeasureReport> {
    let analysis = analyze(mesh, &AnalyzeOptions::default())?;
    let (area, signed_volume) = accumulate(mesh, [0.0; 3], |p| p)?;
    Ok(report(area, signed_volume, verdict(&analysis)))
}

/// Measures `mesh` with each mesh point (the origin plus a local position)
/// mapped through `place` first, so with a placement the report is in
/// world units, exact under any positive scale. Flags and the watertight
/// verdict are placement independent and match [`measure`].
///
/// # Errors
/// As [`measure`].
pub fn measure_placed(
    mesh: &Mesh,
    origin: [f64; 3],
    place: impl FnMut([f64; 3]) -> [f64; 3],
) -> Result<MeasureReport> {
    let analysis = analyze(mesh, &AnalyzeOptions::default())?;
    let (area, signed_volume) = accumulate(mesh, origin, place)?;
    Ok(report(area, signed_volume, verdict(&analysis)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::*;

    #[test]
    fn unit_cube_volume_and_area() {
        let mesh = box_mesh([0.0; 3], [1.0; 3]);
        let m = measure(&mesh).unwrap();
        assert!((m.area - 6.0).abs() < 1e-12, "area {}", m.area);
        assert!((m.volume.unwrap() - 1.0).abs() < 1e-12, "{m:?}");
        assert!(m.signed_volume.abs() - 1.0 < 1e-12);
        assert!(m.closed && m.volume_reason.is_none());
        assert_eq!(
            (
                m.boundary_loops,
                m.non_manifold_edges,
                m.orientation_conflicts
            ),
            (0, 0, 0)
        );
    }

    #[test]
    fn tetrahedron_volume_and_area() {
        // (0,0,0), (1,0,0), (0,1,0), (0,0,1), outward wound: volume 1/6,
        // area three halves plus the sqrt(2)-sided face.
        let mesh = mesh_of(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]],
        );
        let m = measure(&mesh).unwrap();
        assert!(
            (m.area - (1.5 + 3f64.sqrt() / 2.0)).abs() < 1e-12,
            "area {}",
            m.area
        );
        assert!((m.volume.unwrap() - 1.0 / 6.0).abs() < 1e-12, "{m:?}");
        assert!(m.closed && m.volume_reason.is_none());
        assert_eq!(m.orientation_conflicts, 0);
    }

    #[test]
    fn open_grid_has_area_but_no_volume() {
        let mesh = grid(4);
        let m = measure(&mesh).unwrap();
        assert!((m.area - 16.0).abs() < 1e-9, "area {}", m.area);
        assert_eq!(m.volume, None);
        assert_eq!(m.volume_reason, Some(VolumeReason::Open));
        assert!(!m.closed);
        assert_eq!(m.boundary_loops, 1);
    }

    #[test]
    fn non_manifold_fin_reports_that_reason() {
        let m = measure(&fin()).unwrap();
        assert_eq!(m.volume, None);
        assert_eq!(m.volume_reason, Some(VolumeReason::NonManifold));
        assert!(!m.closed);
        assert!(m.non_manifold_edges > 0);
    }

    #[test]
    fn flipped_triangles_keep_a_flagged_volume() {
        // Tolerant: a volume is still reported, with the conflict count
        // warning that opposing triangles cancelled.
        let sphere = uv_sphere(8, 12, 1.0);
        let whole = measure(&sphere).unwrap().volume.unwrap();
        let flipped = flip(&sphere, &[0, 1, 2]);
        let m = measure(&flipped).unwrap();
        assert!(m.closed && m.volume.is_some());
        assert!(m.orientation_conflicts > 0);
        assert!(
            (m.volume.unwrap() - whole).abs() > 1e-6,
            "cancellation should move the volume"
        );
    }

    #[test]
    fn placed_cube_scales_area_and_volume() {
        // A uniform scale of 2 about the mesh point: area x4, volume x8.
        let mesh = box_mesh([0.0; 3], [1.0; 3]);
        let m = measure_placed(&mesh, [0.0; 3], |p| [2.0 * p[0], 2.0 * p[1], 2.0 * p[2]]).unwrap();
        assert!((m.area - 24.0).abs() < 1e-9, "area {}", m.area);
        assert!((m.volume.unwrap() - 8.0).abs() < 1e-9, "{m:?}");
        // Non-uniform scales are exact too: 2 x 1 x 1 doubles both.
        let m = measure_placed(&mesh, [0.0; 3], |p| [2.0 * p[0], p[1], p[2]]).unwrap();
        assert!((m.area - 10.0).abs() < 1e-9, "area {}", m.area);
        assert!((m.volume.unwrap() - 2.0).abs() < 1e-9, "{m:?}");
        // A translation alone changes nothing.
        let m = measure_placed(&mesh, [0.0; 3], |p| [p[0] + 5.0, p[1] - 3.0, p[2] + 1.0]).unwrap();
        assert!((m.area - 6.0).abs() < 1e-12);
        assert!((m.volume.unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn placed_matches_local_flags() {
        // Flags and the verdict do not depend on the mapping.
        let mesh = damaged_sphere();
        let a = measure(&mesh).unwrap();
        let b = measure_placed(&mesh, [1.0, 2.0, 3.0], |p| [p[0] + 4.0, p[1], p[2]]).unwrap();
        assert_eq!(
            (
                a.closed,
                a.volume_reason,
                a.boundary_loops,
                a.non_manifold_edges,
                a.orientation_conflicts
            ),
            (
                b.closed,
                b.volume_reason,
                b.boundary_loops,
                b.non_manifold_edges,
                b.orientation_conflicts
            )
        );
        assert!((a.area - b.area).abs() < 1e-9);
    }
}
