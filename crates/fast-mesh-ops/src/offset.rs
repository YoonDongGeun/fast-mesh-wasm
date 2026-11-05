//! Offsetting a surface along its normals and solidifying it into a shell.
//!
//! [`offset`] moves every live vertex along its smooth vertex normal
//! ([`compute_vertex_normals`](fast_mesh::compute_vertex_normals)), by
//! `distance` in local units. A positive distance follows the normals, which
//! is outward on an outward-wound closed mesh; a negative distance sinks the
//! surface inward. A zero distance writes nothing.
//!
//! [`shell`] solidifies the mesh: it keeps the input surface, adds a copy
//! moved by `thickness` against the smooth normals with reversed winding, and
//! stitches every boundary rim to its copy with a flat wall band, so an open
//! mesh comes out closed. A closed mesh has no rims to stitch; it comes out
//! as two nested closed components. Triangles with a repeated vertex id are
//! not copied: they have no surface to solidify.
//!
//! Both operations read their whole plan from the untouched mesh and only
//! then write, so an error leaves the mesh as it was. After writing, both
//! run [`remove_degenerate`](crate::remove_degenerate) with the default
//! options over the result and report what it removed: sharp concave offsets
//! pinch triangles down to nothing, and those are dropped here. A fold that
//! keeps full area (a flipped but stretched triangle) is not degenerate and
//! is left alone; [`analyze`](crate::analyze) with a BVH reports such
//! self-intersections.

use fast_mesh::compute_vertex_normals;
use fast_mesh_store::{Mesh, Transaction, UV_CHANNEL, VertexId};

use crate::topology::{EdgeMap, boundary_loops, proper_triangle};
use crate::{DegenerateOptions, OpsError, Result, filled, push, remove_degenerate, reserve};

/// Options for [`offset`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OffsetOptions {
    /// How far to move the surface along its smooth normals, in local units.
    /// Positive follows the normals (outward on an outward-wound closed
    /// mesh); negative sinks inward. Must be finite; zero writes nothing.
    pub distance: f32,
}

impl OffsetOptions {
    /// A builder for [`OffsetOptions`]: a zero distance until set.
    pub fn builder() -> OffsetOptionsBuilder {
        OffsetOptionsBuilder::default()
    }
}

/// A builder for [`OffsetOptions`], see [`OffsetOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct OffsetOptionsBuilder {
    options: OffsetOptions,
}

impl OffsetOptionsBuilder {
    /// How far to move the surface along its smooth normals, in local units.
    pub fn distance(mut self, distance: f32) -> Self {
        self.options.distance = distance;
        self
    }

    /// The options as set.
    pub fn build(self) -> OffsetOptions {
        self.options
    }
}

/// What [`offset`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OffsetReport {
    /// How many vertices moved (displacement above 0).
    pub moved: usize,
    /// The largest displacement applied, in local units; 0 when nothing
    /// moved.
    pub max_displacement: f32,
    /// Degenerate triangles the cleanup pass removed.
    pub degenerate_removed: usize,
    /// Vertices the cleanup pass removed.
    pub vertices_removed: usize,
}

/// Moves every live vertex along its smooth normal by `distance`.
///
/// - Region: every live vertex, displaced by `distance * normal` where
///   `normal` is its smooth normal. Vertices with a zero normal (used by no
///   non-degenerate triangle) or a non-finite displacement stay put.
/// - The move is written once, in a single `set_positions` call over the
///   vertices whose displacement is above 0; an empty move writes nothing.
/// - Cleanup: [`remove_degenerate`](crate::remove_degenerate) with the
///   default options runs over the result and drops the triangles the offset
///   pinched down to nothing (and the vertices left unused), reported as
///   `degenerate_removed` and `vertices_removed`. History is untouched.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when the distance is not finite, or the mesh
///   buffers are not a valid mesh.
/// - [`OpsError::Store`] when memory runs out.
///
/// Nothing is written in any of these cases.
pub fn offset(tx: &mut Transaction, options: &OffsetOptions) -> Result<OffsetReport> {
    let distance = options.distance;
    if !distance.is_finite() {
        return Err(OpsError::InvalidInput("offset distance must be finite"));
    }
    let mut report = OffsetReport::default();
    if distance == 0.0 {
        return Ok(report);
    }
    let mesh = tx.mesh();
    if mesh.triangle_count() == 0 {
        return Ok(report);
    }
    let normals = smooth_normals(mesh)?;
    let positions = mesh.positions();
    let mut ids: Vec<VertexId> = Vec::new();
    let mut moved_to: Vec<[f32; 3]> = Vec::new();
    for (v, p) in positions.iter().enumerate() {
        let v = v as VertexId;
        if !mesh.is_vertex_alive(v) {
            continue;
        }
        let n = [
            normals[3 * v as usize],
            normals[3 * v as usize + 1],
            normals[3 * v as usize + 2],
        ];
        let displaced = [
            p[0] + distance * n[0],
            p[1] + distance * n[1],
            p[2] + distance * n[2],
        ];
        let length = n.iter().map(|n| (distance * n).powi(2)).sum::<f32>().sqrt();
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
    let cleanup = remove_degenerate(tx, &DegenerateOptions::default())?;
    report.degenerate_removed = cleanup.removed_triangles;
    report.vertices_removed = cleanup.removed_vertices;
    Ok(report)
}

/// Options for [`shell`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellOptions {
    /// Wall thickness in local units: the copy sits this far against the
    /// smooth normals. Must be positive and finite.
    pub thickness: f32,
}

impl Default for ShellOptions {
    fn default() -> Self {
        ShellOptions { thickness: 1.0 }
    }
}

impl ShellOptions {
    /// A builder for [`ShellOptions`]: a unit thickness until set.
    pub fn builder() -> ShellOptionsBuilder {
        ShellOptionsBuilder::default()
    }
}

/// A builder for [`ShellOptions`], see [`ShellOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct ShellOptionsBuilder {
    options: ShellOptions,
}

impl ShellOptionsBuilder {
    /// Wall thickness in local units.
    pub fn thickness(mut self, thickness: f32) -> Self {
        self.options.thickness = thickness;
        self
    }

    /// The options as set.
    pub fn build(self) -> ShellOptions {
        self.options
    }
}

/// What [`shell`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShellReport {
    /// Copy vertices added.
    pub vertices_added: usize,
    /// Copy and wall triangles added.
    pub triangles_added: usize,
    /// Boundary rims stitched to their copy. A closed mesh has none.
    pub loops_closed: usize,
    /// Degenerate triangles the cleanup pass removed.
    pub degenerate_removed: usize,
    /// Vertices the cleanup pass removed.
    pub vertices_removed: usize,
}

/// A corner of a planned triangle: a kept live vertex or a copy vertex by
/// its index in the planned copy positions.
#[derive(Debug, Clone, Copy)]
enum Corner {
    Kept(VertexId),
    Copy(usize),
}

/// Solidifies the mesh into a printable shell of wall `thickness`.
///
/// - The input surface stays. Every live vertex gains a copy moved by
///   `thickness` against its smooth normal; every proper triangle gains a
///   reversed copy over the copied vertices, so the copy faces outward from
///   the wall interior.
/// - Every boundary edge (an edge of exactly one proper triangle) gains a
///   flat wall quad over its copy: for the edge the outer triangle uses as
///   `x -> y`, the quad is `(y, x, x')` and `(y, x', y')`, which agrees with
///   both sheets along their shared edges. Rims through non-manifold
///   vertices or orientation seams are stitched edge by edge on a
///   best-effort basis and may stay open.
/// - Writes happen in order: copy vertices, then copy and wall triangles.
///   Then [`remove_degenerate`](crate::remove_degenerate) with the default
///   options drops collapsed triangles (a wall thinner than the local
///   curvature pinches its copy to nothing) and unused vertices, reported as
///   `degenerate_removed` and `vertices_removed`. History is untouched.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when the thickness is not positive and
///   finite, when a vertex position or a copy position is not finite, or
///   when the mesh buffers are not a valid mesh.
/// - [`OpsError::Store`] when memory runs out.
///
/// Nothing is written in any of these cases.
pub fn shell(tx: &mut Transaction, options: &ShellOptions) -> Result<ShellReport> {
    let thickness = options.thickness;
    if !thickness.is_finite() || thickness <= 0.0 {
        return Err(OpsError::InvalidInput(
            "shell thickness must be positive and finite",
        ));
    }
    let mut report = ShellReport::default();
    let mesh = tx.mesh();
    if mesh.triangle_count() == 0 {
        return Ok(report);
    }
    let normals = smooth_normals(mesh)?;
    let positions = mesh.positions();
    // The copy of every live vertex, in ascending id order, with each live
    // vertex's index in it. Planned from the untouched mesh, before writing.
    let mut copied: Vec<[f32; 3]> = Vec::new();
    // On a mesh with UVs, each copy keeps its original's.
    let source_uvs = mesh.uvs();
    let mut copied_uvs: Vec<f32> = Vec::new();
    let mut at: Vec<Option<usize>> = filled(mesh.vertex_slots(), None)?;
    for (v, p) in positions.iter().enumerate() {
        let v = v as VertexId;
        if !mesh.is_vertex_alive(v) {
            continue;
        }
        if !p.iter().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput(
                "shell needs finite vertex positions",
            ));
        }
        let n = [
            normals[3 * v as usize],
            normals[3 * v as usize + 1],
            normals[3 * v as usize + 2],
        ];
        let q = [
            p[0] - thickness * n[0],
            p[1] - thickness * n[1],
            p[2] - thickness * n[2],
        ];
        if !q.iter().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput("shell thickness leaves f32 range"));
        }
        push(&mut copied, q)?;
        if let Some(uv) = source_uvs {
            reserve(&mut copied_uvs, 2)?;
            copied_uvs.extend_from_slice(&uv[2 * v as usize..2 * v as usize + 2]);
        }
        at[v as usize] = Some(copied.len() - 1);
    }
    // The reversed copies and the wall quads, in ascending triangle and edge
    // order. Triangles with a repeated vertex id are not copied.
    let mut planned: Vec<[Corner; 3]> = Vec::new();
    for t in 0..mesh.triangle_slots() as u32 {
        if let Some([a, b, c]) = proper_triangle(mesh, t) {
            let [ia, ib, ic] = [a, b, c].map(|v| at[v as usize].expect("live corner"));
            push(
                &mut planned,
                [Corner::Copy(ic), Corner::Copy(ib), Corner::Copy(ia)],
            )?;
        }
    }
    let edges = EdgeMap::build(mesh)?;
    for ((u, v), tris) in edges.edges() {
        if tris.len() != 1 {
            continue;
        }
        let [a, b, c] = mesh.triangle(tris[0]).expect("mapped triangle is alive");
        let (x, y) = [(a, b), (b, c), (c, a)]
            .into_iter()
            .find(|&(p, q)| (p == u && q == v) || (p == v && q == u))
            .expect("mapped edge uses its triangle");
        let (ix, iy) = (
            at[x as usize].expect("live corner"),
            at[y as usize].expect("live corner"),
        );
        push(
            &mut planned,
            [Corner::Kept(y), Corner::Kept(x), Corner::Copy(ix)],
        )?;
        push(
            &mut planned,
            [Corner::Kept(y), Corner::Copy(ix), Corner::Copy(iy)],
        )?;
    }
    report.loops_closed = boundary_loops(mesh)?.len();
    let has_uvs = source_uvs.is_some();
    let fresh = tx.add_vertices(&copied)?;
    if has_uvs {
        tx.write_channel(UV_CHANNEL, &fresh, &copied_uvs)?;
    }
    let resolve = |c: Corner| match c {
        Corner::Kept(v) => v,
        Corner::Copy(i) => fresh[i],
    };
    let mut added: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut added, planned.len())?;
    for tri in planned {
        added.push(tri.map(resolve));
    }
    tx.add_triangles(&added)?;
    report.vertices_added = fresh.len();
    report.triangles_added = added.len();
    let cleanup = remove_degenerate(tx, &DegenerateOptions::default())?;
    report.degenerate_removed = cleanup.removed_triangles;
    report.vertices_removed = cleanup.removed_vertices;
    Ok(report)
}

/// The smooth vertex normals of `mesh`, slot-aligned, as
/// [`inflate`](crate::deform::inflate) computes them.
fn smooth_normals(mesh: &Mesh) -> Result<Vec<f32>> {
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
    compute_vertex_normals(&flat, &indices)
        .map_err(|_| OpsError::InvalidInput("mesh buffers are not a valid mesh"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{box_mesh, edit, grid, mesh_of, parts};

    /// The distance of every live vertex from `c`.
    fn distances(mesh: &Mesh, c: [f32; 3]) -> Vec<f32> {
        let mut out = Vec::new();
        for (v, p) in mesh.positions().iter().enumerate() {
            if !mesh.is_vertex_alive(v as VertexId) {
                continue;
            }
            out.push(
                ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt(),
            );
        }
        out
    }

    #[test]
    fn offset_moves_along_normals_by_distance() {
        let mut mesh = grid(2);
        let old = parts(&mesh).0;
        let options = OffsetOptions::builder().distance(0.5).build();
        let (report, empty) = edit(&mut mesh, |tx| offset(tx, &options));
        assert!(!empty);
        // A flat grid wound from +z: every normal is +z, so every vertex
        // moves exactly 0.5 up.
        assert_eq!(report.moved, 9);
        assert!((report.max_displacement - 0.5).abs() < 1e-6);
        assert_eq!(report.degenerate_removed, 0);
        assert_eq!(report.vertices_removed, 0);
        let new = parts(&mesh).0;
        for v in 0..9 {
            assert!((new[v * 3] - old[v * 3]).abs() < 1e-6);
            assert!((new[v * 3 + 1] - old[v * 3 + 1]).abs() < 1e-6);
            assert!((new[v * 3 + 2] - old[v * 3 + 2] - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn offset_sign_follows_winding() {
        // A closed, outward-wound box: a positive offset moves every corner
        // exactly `distance` along its own (unit) smooth normal, which points
        // outward, so every corner leaves the centroid behind; a negative
        // distance sinks the corners back inward. (Box corners are shared by
        // an uneven triangulation, so their normals are not exact diagonals:
        // only the length and the outward sense are asserted.)
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        let c = [0.5; 3];
        let before = distances(&mesh, c);
        let (report, _) = edit(&mut mesh, |tx| {
            offset(tx, &OffsetOptions::builder().distance(0.25).build())
        });
        assert_eq!(report.moved, 8);
        assert!((report.max_displacement - 0.25).abs() < 1e-6);
        let grown = distances(&mesh, c);
        for (a, b) in before.iter().zip(&grown) {
            assert!(*b > *a, "{a} -> {b}");
            assert!((b - a) <= 0.25 + 1e-6, "{a} -> {b}");
        }
        // Every corner moved exactly 0.25: smooth normals are unit.
        let prev = parts(&box_mesh([0.0; 3], [1.0; 3])).0;
        let next = parts(&mesh).0;
        for v in 0..8 {
            let d = ((next[v * 3] - prev[v * 3]).powi(2)
                + (next[v * 3 + 1] - prev[v * 3 + 1]).powi(2)
                + (next[v * 3 + 2] - prev[v * 3 + 2]).powi(2))
            .sqrt();
            assert!((d - 0.25).abs() < 1e-5, "{d}");
        }
        // And a negative distance sinks the surface back past its start.
        edit(&mut mesh, |tx| {
            offset(tx, &OffsetOptions::builder().distance(-0.5).build()).map(drop)
        });
        let sunk = distances(&mesh, c);
        for (a, b) in before.iter().zip(&sunk) {
            assert!(*b < *a, "{a} -> {b}");
        }
    }

    #[test]
    fn offset_zero_distance_and_empty_mesh_are_noops() {
        let mut mesh = grid(2);
        let before = parts(&mesh);
        let mut tx = mesh.begin_edit();
        let report = offset(&mut tx, &OffsetOptions::builder().distance(0.0).build()).unwrap();
        assert_eq!(report, OffsetReport::default());
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
        // No triangles: nothing to move along, and no cleanup either.
        let mut mesh = Mesh::new();
        let mut tx = mesh.begin_edit();
        let report = offset(&mut tx, &OffsetOptions::builder().distance(1.0).build()).unwrap();
        assert_eq!(report, OffsetReport::default());
        assert!(tx.commit().unwrap().is_empty());
    }

    #[test]
    fn offset_rejects_nonfinite_distance() {
        for distance in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut mesh = grid(2);
            let before = parts(&mesh);
            let mut tx = mesh.begin_edit();
            assert!(matches!(
                offset(
                    &mut tx,
                    &OffsetOptions::builder().distance(distance).build()
                ),
                Err(OpsError::InvalidInput(_))
            ));
            assert!(tx.commit().unwrap().is_empty());
            assert_eq!(parts(&mesh), before);
        }
    }

    #[test]
    fn offset_collapse_without_pinch_stays_valid() {
        // A regular tetrahedron offset inward by its circumradius moves every
        // vertex onto the centroid. The four triangles shrink to specks, but
        // they stay well-shaped, so the cleanup keeps them: only exact
        // pinches and pre-existing degenerates are removed, never valid
        // triangles, however small. (Near-coincident vertices are not
        // merged either; welding stays an explicit operation.)
        let mut mesh = mesh_of(
            &[1., 1., 1., 1., -1., -1., -1., 1., -1., -1., -1., 1.],
            &[[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
        );
        assert_eq!(mesh.vertex_count(), 4);
        let c = [0.0; 3];
        let radius = distances(&mesh, c)[0];
        let (report, _) = edit(&mut mesh, |tx| {
            offset(tx, &OffsetOptions::builder().distance(-radius).build())
        });
        assert_eq!(report.moved, 4);
        assert!((report.max_displacement - radius).abs() < 1e-4);
        assert_eq!(report.degenerate_removed, 0);
        assert_eq!(mesh.triangle_count(), 4);
        for d in distances(&mesh, c) {
            assert!(d < 1e-6, "{d}");
        }
    }

    #[test]
    fn offset_cleans_preexisting_degenerate_triangles() {
        // A grid with an extra collapsed triangle: the offset moves the grid
        // and the cleanup drops the collapsed triangle, reporting it.
        let mut mesh = grid(2);
        let (flat, mut tris) = parts(&mesh);
        tris.push([0, 0, 1]);
        mesh = mesh_of(&flat, &tris);
        assert_eq!(mesh.triangle_count(), 9);
        let options = OffsetOptions::builder().distance(0.5).build();
        let (report, _) = edit(&mut mesh, |tx| offset(tx, &options));
        assert_eq!(report.moved, 9);
        assert_eq!(report.degenerate_removed, 1);
        assert_eq!(mesh.triangle_count(), 8);
    }

    #[test]
    fn shell_stitches_an_open_triangle() {
        // One triangle in z = 0, wound from +z: the copy lands at z = -t and
        // three wall quads stitch the rim.
        let mut mesh = mesh_of(&[0., 0., 0., 1., 0., 0., 0., 1., 0.], &[[0, 1, 2]]);
        let (report, empty) = edit(&mut mesh, |tx| {
            shell(tx, &ShellOptions::builder().thickness(0.5).build())
        });
        assert!(!empty);
        assert_eq!(
            report,
            ShellReport {
                vertices_added: 3,
                triangles_added: 1 + 6,
                loops_closed: 1,
                degenerate_removed: 0,
                vertices_removed: 0,
            }
        );
        assert_eq!(mesh.vertex_count(), 6);
        assert_eq!(mesh.triangle_count(), 8);
        // The outer sheet stayed, the copy sits exactly one thickness below.
        let (flat, _) = parts(&mesh);
        assert_eq!(&flat[0..9], &[0., 0., 0., 1., 0., 0., 0., 1., 0.][..]);
        assert_eq!(
            &flat[9..18],
            &[0., 0., -0.5, 1., 0., -0.5, 0., 1., -0.5][..]
        );
        // Closed and consistently wound: no boundary, no conflicts.
        let analysis = crate::analyze(&mesh, &crate::AnalyzeOptions::default()).unwrap();
        assert!(analysis.loops.is_empty());
        assert_eq!(analysis.orientation_conflicts, 0);
        assert!(analysis.components.iter().all(|c| c.closed));
    }

    #[test]
    fn shell_nests_a_closed_cube() {
        // A closed box has no rims: the reversed copy nests inside it.
        let mut mesh = box_mesh([0.0; 3], [2.0; 3]);
        let (report, _) = edit(&mut mesh, |tx| {
            shell(tx, &ShellOptions::builder().thickness(0.25).build())
        });
        assert_eq!(report.vertices_added, 8);
        assert_eq!(report.triangles_added, 12);
        assert_eq!(report.loops_closed, 0);
        assert_eq!(report.degenerate_removed, 0);
        assert_eq!(mesh.triangle_count(), 24);
        let analysis = crate::analyze(&mesh, &crate::AnalyzeOptions::default()).unwrap();
        assert!(analysis.loops.is_empty());
        assert_eq!(analysis.orientation_conflicts, 0);
        assert_eq!(analysis.components.len(), 2);
        assert!(analysis.components.iter().all(|c| c.closed));
        // The outer sheet stayed exactly where it was.
        let (flat, _) = parts(&mesh);
        assert_eq!(flat.len(), 16 * 3);
        let outer: Vec<f32> = (0..8)
            .flat_map(|i| (0..3).map(move |k| if i >> k & 1 == 1 { 2.0 } else { 0.0 }))
            .collect();
        assert_eq!(&flat[0..24], &outer[..]);
        // The inner copy sits one thickness inside: every copy vertex is
        // exactly `thickness` from its original (unit smooth normals) and
        // strictly inside the box (the thickness is smaller than its
        // half-size, so nothing overshoots the far side).
        for v in 0..8 {
            let (o, q) = (&flat[v * 3..v * 3 + 3], &flat[(v + 8) * 3..(v + 8) * 3 + 3]);
            let d = ((q[0] - o[0]).powi(2) + (q[1] - o[1]).powi(2) + (q[2] - o[2]).powi(2)).sqrt();
            assert!((d - 0.25).abs() < 1e-5, "{d}");
            for k in 0..3 {
                assert!(q[k] > 0.0 && q[k] < 2.0, "{q:?}");
            }
        }
    }

    #[test]
    fn shell_rejects_bad_thickness() {
        for thickness in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut mesh = grid(2);
            let before = parts(&mesh);
            let mut tx = mesh.begin_edit();
            assert!(matches!(
                shell(
                    &mut tx,
                    &ShellOptions::builder().thickness(thickness).build()
                ),
                Err(OpsError::InvalidInput(_))
            ));
            assert!(tx.commit().unwrap().is_empty());
            assert_eq!(parts(&mesh), before);
        }
    }

    #[test]
    fn shell_empty_mesh_is_a_noop() {
        let mut mesh = Mesh::new();
        let mut tx = mesh.begin_edit();
        let report = shell(&mut tx, &ShellOptions::builder().thickness(0.5).build()).unwrap();
        assert_eq!(report, ShellReport::default());
        assert!(tx.commit().unwrap().is_empty());
    }

    #[test]
    fn shell_first_failure_writes_nothing() {
        // The plan is read before the first write: failing the first
        // reservation leaves the mesh untouched.
        let mut mesh = grid(2);
        let before = parts(&mesh);
        let mut tx = mesh.begin_edit();
        crate::fail_allocations_after(Some(0));
        let r = shell(&mut tx, &ShellOptions::builder().thickness(0.5).build());
        crate::fail_allocations_after(None);
        assert_eq!(
            r,
            Err(OpsError::Store(fast_mesh_store::StoreError::OutOfMemory))
        );
        assert!(tx.commit().unwrap().is_empty());
        assert_eq!(parts(&mesh), before);
    }
}
