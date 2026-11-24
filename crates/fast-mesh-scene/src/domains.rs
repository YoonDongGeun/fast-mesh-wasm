//! Domain-separated extension traits for [`Scene`].
//!
//! Each trait groups cohesive scene-level operations into one of seven domains:
//! - [`SceneRepairOps`]: mesh healing, joining, and cleanup.
//! - [`SceneSculptOps`]: brush dabs, stroke state, and stretch splitting.
//! - [`SceneRemeshOps`]: region remeshing, quadric simplification, and Loop subdivision.
//! - [`SceneBooleanOps`]: CSG booleans, plane cutting, offsetting, and shelling.
//! - [`SceneDeformOps`]: vertex-protected sculpting deformation.
//! - [`SceneInspectOps`]: defect analysis, measurement, world queries, fields, and sections.
//! - [`SceneAlignOps`]: rigid ICP registration.
//!
//! All methods delegate cleanly to [`Scene`] without duplicating core logic.

use fast_mesh_bvh::Aabb;
use fast_mesh_ops::boolean::{
    DrainHoleOptions, DrainHoleReport, DrainHoleSpec, HollowOptions, HollowReport, OffsetOptions,
    OffsetReport, PlaceMeshParams, PlaceMeshReport, ShellOptions, ShellReport,
};
use fast_mesh_ops::deform::{FfdBasis, FfdLattice, FreeformOp, FreeformReport};
use fast_mesh_ops::inspect::{
    Analysis, AnalyzeOptions, BoundaryLoop, ContourSlice, FitPrimitiveOptions, FitPrimitiveReport,
    FittedPrimitive, MeasureReport, PointCloud, PrimitiveKind, SurfaceReconstructOptions,
};
use fast_mesh_ops::remesh::{
    RemeshOptions, RemeshReport, SimplifyOptions, SimplifyReport, SubdivideOptions, SubdivideReport,
};
use fast_mesh_ops::repair::{
    AdjacencyKind, AutoRepairOptions, AutoRepairReport, BridgeStats, ComponentOptions,
    DegenerateOptions, DegenerateReport, FaceEditReport, FillHolesOptions, FillOptions, FillReport,
    FillStats, ManifoldOptions, ManifoldReport, OrientReport, PartialBridgeOptions, RemoveReport,
    ScreenSelectOptions, ScreenSelectionShape, SelfIntersectOptions, SelfIntersectReport,
    SliverOptions, SliverReport, WeldReport,
};
use fast_mesh_ops::sculpt::{DyntopoOptions, DyntopoPatch, StretchOptions, StretchReport};
use fast_mesh_store::{TriangleId, VertexId};

use crate::{
    BooleanOp, BooleanOptions, BooleanReport, Contact, ContactFilter, CsgNode, CurvatureField,
    CutOptions, CutReport, Dab, DabReport, DeviationOptions, DeviationStats, DistanceField,
    FieldId, GeodesicOptions, JoinOptions, JoinReport, LandmarkAlignment, LandmarkPair, MeshId,
    OverhangOptions, OverhangReport, Placement, QualityMetric, QualityReport, RegisterOptions,
    RegisterReport, RegistrationOptions, RegistrationReport, Result, Scene, SceneClosestHit,
    SceneClosestOptions, SceneMeshDistance, SceneRayHit, SceneRayOptions, SectionDesc, SectionId,
    SurfacePathPoint, SymmetryPlaneReport, ThicknessField, TrimCurveOptions, TrimReport, Vec3,
};

/// Repair domain operations on [`Scene`].
pub trait SceneRepairOps {
    /// Joins mesh `b` into mesh `a` across two boundary loops, then removes `b`.
    fn join(&mut self, a: MeshId, b: MeshId, options: &JoinOptions) -> Result<JoinReport>;

    /// Runs the automated repair pipeline on mesh `id`.
    fn auto_repair(&mut self, id: MeshId, options: &AutoRepairOptions) -> Result<AutoRepairReport>;

    /// Welds coincident vertices on mesh `id` within `tolerance`.
    fn weld(&mut self, id: MeshId, tolerance: f32) -> Result<WeldReport>;

    /// Makes triangle winding consistent and outward-facing on mesh `id`.
    fn orient(&mut self, id: MeshId) -> Result<OrientReport>;

    /// Fills matching boundary loops on mesh `id`.
    fn fill_holes(&mut self, id: MeshId, options: &FillOptions) -> Result<FillReport>;

    /// Fills a single boundary loop on mesh `id` with configurable geometric continuity.
    fn fill_hole_continuity(
        &mut self,
        id: MeshId,
        loop_index: usize,
        options: &FillHolesOptions,
    ) -> Result<FillStats>;

    /// Fills all non-rim boundary holes on mesh `id` with configurable geometric continuity
    /// and island merging.
    fn fill_all_holes_continuity(
        &mut self,
        id: MeshId,
        options: &FillHolesOptions,
    ) -> Result<FillStats>;

    /// Bridges two partial boundary arcs (specified either by explicit vertex arcs or
    /// index ranges in `options`) on mesh `id`.
    fn bridge_partial_arcs(
        &mut self,
        id: MeshId,
        loop_a: &[VertexId],
        loop_b: &[VertexId],
        options: &PartialBridgeOptions,
    ) -> Result<BridgeStats>;

    /// Repairs topological non-manifold defects on mesh `id`.
    fn make_manifold(&mut self, id: MeshId, options: &ManifoldOptions) -> Result<ManifoldReport>;

    /// Collapses or flips sliver triangles on mesh `id`.
    fn cleanup_slivers(&mut self, id: MeshId, options: &SliverOptions) -> Result<SliverReport>;

    /// Resolves self-intersecting triangle pairs on mesh `id`.
    fn resolve_self_intersections(
        &mut self,
        id: MeshId,
        options: &SelfIntersectOptions,
    ) -> Result<SelfIntersectReport>;

    /// Removes degenerate and duplicate triangles on mesh `id`.
    fn remove_degenerate(
        &mut self,
        id: MeshId,
        options: &DegenerateOptions,
    ) -> Result<DegenerateReport>;

    /// Removes disconnected components smaller than the configured threshold on mesh `id`.
    fn remove_small_components(
        &mut self,
        id: MeshId,
        options: &ComponentOptions,
    ) -> Result<RemoveReport>;

    /// Selects triangles inside/touching a 2D screen-space rectangle or lasso polygon.
    fn select_faces_screen(
        &mut self,
        id: MeshId,
        shape: &ScreenSelectionShape,
        world_opts: &ScreenSelectOptions,
    ) -> Result<Vec<u32>>;

    /// Selects triangles intersecting a world sphere, optionally filtered by a world normal cone.
    fn select_faces_in_sphere_world(
        &mut self,
        id: MeshId,
        world_center: Vec3,
        world_radius: f64,
        world_normal_cone: Option<(Vec3, f64)>,
    ) -> Result<Vec<u32>>;

    /// Flood-selects triangles edge-connected to `seed_tri` within `max_dihedral_rad`.
    fn select_by_crease_angle(
        &self,
        id: MeshId,
        seed_tri: TriangleId,
        max_dihedral_rad: f64,
    ) -> Result<Vec<u32>>;

    /// Selects the entire edge-connected shell containing `seed_tri`.
    fn select_connected_shell(&self, id: MeshId, seed_tri: TriangleId) -> Result<Vec<u32>>;

    /// Selects edge-connected face islands matching triangle count and area limits.
    fn select_small_face_islands(
        &self,
        id: MeshId,
        max_face_count: usize,
        max_area: Option<f64>,
    ) -> Result<Vec<u32>>;

    /// Grows `selected_tris` outward by `rings` across `adjacency`.
    fn grow_face_selection(
        &self,
        id: MeshId,
        selected_tris: &[u32],
        rings: usize,
        adjacency: AdjacencyKind,
    ) -> Result<Vec<u32>>;

    /// Shrinks `selected_tris` inward by `rings` across `adjacency`.
    fn shrink_face_selection(
        &self,
        id: MeshId,
        selected_tris: &[u32],
        rings: usize,
        adjacency: AdjacencyKind,
    ) -> Result<Vec<u32>>;

    /// Fills interior pockets of unselected faces up to `max_hole_tris` triangles.
    fn fill_selection_holes(
        &self,
        id: MeshId,
        selected_tris: &[u32],
        max_hole_tris: usize,
    ) -> Result<Vec<u32>>;

    /// Smooths the border of `selected_tris` by majority voting for `iterations` passes.
    fn smooth_selection_boundary(
        &self,
        id: MeshId,
        selected_tris: &[u32],
        iterations: usize,
    ) -> Result<Vec<u32>>;

    /// Converts `selected_tris` into a dense per-vertex-slot weight mask.
    fn face_selection_to_vertex_mask(&self, id: MeshId, selected_tris: &[u32]) -> Result<Vec<f32>>;

    /// Converts a dense per-vertex-slot weight mask into sorted triangle indices.
    fn vertex_mask_to_face_selection(
        &self,
        id: MeshId,
        mask: &[f32],
        threshold: f32,
        require_all_vertices: bool,
    ) -> Result<Vec<u32>>;

    /// Deletes `selected_tris` and newly orphaned vertices from `id`.
    fn delete_selected_faces(
        &mut self,
        id: MeshId,
        selected_tris: &[u32],
    ) -> Result<FaceEditReport>;

    /// Extracts `selected_tris` into a new scene mesh preserving exact world coordinates.
    fn extract_selected_faces(&mut self, id: MeshId, selected_tris: &[u32]) -> Result<MeshId>;

    /// Splits `selected_tris` out of `id` into a new scene mesh preserving world coordinates.
    fn split_selected_faces(
        &mut self,
        id: MeshId,
        selected_tris: &[u32],
    ) -> Result<(MeshId, FaceEditReport)>;
}

impl SceneRepairOps for Scene {
    fn join(&mut self, a: MeshId, b: MeshId, options: &JoinOptions) -> Result<JoinReport> {
        Scene::join(self, a, b, options)
    }

    fn auto_repair(&mut self, id: MeshId, options: &AutoRepairOptions) -> Result<AutoRepairReport> {
        self.edit(id, |tx| crate::ops::repair::auto_repair(tx, options))
    }

    fn weld(&mut self, id: MeshId, tolerance: f32) -> Result<WeldReport> {
        self.edit(id, |tx| crate::ops::repair::weld(tx, tolerance))
    }

    fn orient(&mut self, id: MeshId) -> Result<OrientReport> {
        self.edit(id, crate::ops::repair::orient)
    }

    fn fill_holes(&mut self, id: MeshId, options: &FillOptions) -> Result<FillReport> {
        self.edit(id, |tx| crate::ops::repair::fill_holes(tx, options))
    }

    fn fill_hole_continuity(
        &mut self,
        id: MeshId,
        loop_index: usize,
        options: &FillHolesOptions,
    ) -> Result<FillStats> {
        self.edit(id, |tx| {
            crate::ops::repair::fill_hole(tx, loop_index, options)
        })
    }

    fn fill_all_holes_continuity(
        &mut self,
        id: MeshId,
        options: &FillHolesOptions,
    ) -> Result<FillStats> {
        self.edit(id, |tx| {
            let per_hole = crate::ops::repair::fill_all_holes(tx, options)?;
            let mut iter = per_hole.into_iter();
            match iter.next() {
                None => Ok::<_, crate::ops::OpsError>(FillStats {
                    loop_index: 0,
                    filled_vertices: 0,
                    filled_triangles: 0,
                    perimeter: 0.0,
                    max_dihedral_angle_deg: 0.0,
                    continuity: options.continuity,
                    island_loops_merged: 0,
                    converged: true,
                }),
                Some(mut combined) => {
                    combined.continuity = options.continuity;
                    for stat in iter {
                        combined.filled_vertices += stat.filled_vertices;
                        combined.filled_triangles += stat.filled_triangles;
                        combined.perimeter += stat.perimeter;
                        combined.max_dihedral_angle_deg = combined
                            .max_dihedral_angle_deg
                            .max(stat.max_dihedral_angle_deg);
                        combined.island_loops_merged += stat.island_loops_merged;
                        combined.converged &= stat.converged;
                    }
                    Ok(combined)
                }
            }
        })
    }

    fn bridge_partial_arcs(
        &mut self,
        id: MeshId,
        loop_a: &[VertexId],
        loop_b: &[VertexId],
        options: &PartialBridgeOptions,
    ) -> Result<BridgeStats> {
        self.edit(id, |tx| {
            if loop_a.is_empty() && loop_b.is_empty() {
                return crate::ops::repair::bridge_partial_arcs(tx, options);
            }
            let loops = crate::ops::boundary_loops(tx.mesh())?;
            let mut resolved = *options;
            if !loop_a.is_empty() {
                let (idx, range) = resolve_partial_bridge_arc(&loops, loop_a)?;
                resolved.loop_a_index = idx;
                resolved.arc_a_range = range;
            }
            if !loop_b.is_empty() {
                let (idx, range) = resolve_partial_bridge_arc(&loops, loop_b)?;
                resolved.loop_b_index = idx;
                resolved.arc_b_range = range;
            }
            crate::ops::repair::bridge_partial_arcs(tx, &resolved)
        })
    }

    fn make_manifold(&mut self, id: MeshId, options: &ManifoldOptions) -> Result<ManifoldReport> {
        self.edit(id, |tx| crate::ops::repair::make_manifold(tx, options))
    }

    fn cleanup_slivers(&mut self, id: MeshId, options: &SliverOptions) -> Result<SliverReport> {
        self.edit(id, |tx| crate::ops::repair::cleanup_slivers(tx, options))
    }

    fn resolve_self_intersections(
        &mut self,
        id: MeshId,
        options: &SelfIntersectOptions,
    ) -> Result<SelfIntersectReport> {
        self.edit(id, |tx| {
            crate::ops::repair::resolve_self_intersections(tx, None, options)
        })
    }

    fn remove_degenerate(
        &mut self,
        id: MeshId,
        options: &DegenerateOptions,
    ) -> Result<DegenerateReport> {
        self.edit(id, |tx| crate::ops::repair::remove_degenerate(tx, options))
    }

    fn remove_small_components(
        &mut self,
        id: MeshId,
        options: &ComponentOptions,
    ) -> Result<RemoveReport> {
        self.edit(id, |tx| {
            crate::ops::repair::remove_small_components(tx, options)
        })
    }

    fn select_faces_screen(
        &mut self,
        id: MeshId,
        shape: &ScreenSelectionShape,
        world_opts: &ScreenSelectOptions,
    ) -> Result<Vec<u32>> {
        Scene::select_faces_screen(self, id, shape, world_opts)
    }

    fn select_faces_in_sphere_world(
        &mut self,
        id: MeshId,
        world_center: Vec3,
        world_radius: f64,
        world_normal_cone: Option<(Vec3, f64)>,
    ) -> Result<Vec<u32>> {
        Scene::select_faces_in_sphere_world(self, id, world_center, world_radius, world_normal_cone)
    }

    fn select_by_crease_angle(
        &self,
        id: MeshId,
        seed_tri: TriangleId,
        max_dihedral_rad: f64,
    ) -> Result<Vec<u32>> {
        Scene::select_by_crease_angle(self, id, seed_tri, max_dihedral_rad)
    }

    fn select_connected_shell(&self, id: MeshId, seed_tri: TriangleId) -> Result<Vec<u32>> {
        Scene::select_connected_shell(self, id, seed_tri)
    }

    fn select_small_face_islands(
        &self,
        id: MeshId,
        max_face_count: usize,
        max_area: Option<f64>,
    ) -> Result<Vec<u32>> {
        Scene::select_small_face_islands(self, id, max_face_count, max_area)
    }

    fn grow_face_selection(
        &self,
        id: MeshId,
        selected_tris: &[u32],
        rings: usize,
        adjacency: AdjacencyKind,
    ) -> Result<Vec<u32>> {
        Scene::grow_face_selection(self, id, selected_tris, rings, adjacency)
    }

    fn shrink_face_selection(
        &self,
        id: MeshId,
        selected_tris: &[u32],
        rings: usize,
        adjacency: AdjacencyKind,
    ) -> Result<Vec<u32>> {
        Scene::shrink_face_selection(self, id, selected_tris, rings, adjacency)
    }

    fn fill_selection_holes(
        &self,
        id: MeshId,
        selected_tris: &[u32],
        max_hole_tris: usize,
    ) -> Result<Vec<u32>> {
        Scene::fill_selection_holes(self, id, selected_tris, max_hole_tris)
    }

    fn smooth_selection_boundary(
        &self,
        id: MeshId,
        selected_tris: &[u32],
        iterations: usize,
    ) -> Result<Vec<u32>> {
        Scene::smooth_selection_boundary(self, id, selected_tris, iterations)
    }

    fn face_selection_to_vertex_mask(&self, id: MeshId, selected_tris: &[u32]) -> Result<Vec<f32>> {
        Scene::face_selection_to_vertex_mask(self, id, selected_tris)
    }

    fn vertex_mask_to_face_selection(
        &self,
        id: MeshId,
        mask: &[f32],
        threshold: f32,
        require_all_vertices: bool,
    ) -> Result<Vec<u32>> {
        Scene::vertex_mask_to_face_selection(self, id, mask, threshold, require_all_vertices)
    }

    fn delete_selected_faces(
        &mut self,
        id: MeshId,
        selected_tris: &[u32],
    ) -> Result<FaceEditReport> {
        Scene::delete_selected_faces(self, id, selected_tris)
    }

    fn extract_selected_faces(&mut self, id: MeshId, selected_tris: &[u32]) -> Result<MeshId> {
        Scene::extract_selected_faces(self, id, selected_tris)
    }

    fn split_selected_faces(
        &mut self,
        id: MeshId,
        selected_tris: &[u32],
    ) -> Result<(MeshId, FaceEditReport)> {
        Scene::split_selected_faces(self, id, selected_tris)
    }
}

/// Sculpt domain operations on [`Scene`].
pub trait SceneSculptOps {
    /// Applies a sculpting brush dab to mesh `id`.
    fn dab(&mut self, id: MeshId, dab: &Dab) -> Result<DabReport>;

    /// Arms a stroke-start position and normal snapshot on mesh `id`.
    fn capture_stroke_state(&mut self, id: MeshId, only_if_missing: bool) -> Result<()>;

    /// Records vertex rest edge sizes on mesh `id` for stretch splitting.
    fn capture_rest_sizes(&mut self, id: MeshId, only_if_missing: bool) -> Result<()>;

    /// Adaptively splits stretched edges on mesh `id`.
    fn split_stretched(&mut self, id: MeshId, options: &StretchOptions) -> Result<StretchReport>;

    /// Queries vertex ids inside a local sphere on mesh `id`.
    fn vertices_in_sphere(
        &mut self,
        id: MeshId,
        center: [f32; 3],
        radius: f32,
    ) -> Result<Vec<VertexId>>;

    /// Runs localized dynamic-topology refinement inside a sphere on mesh `id`.
    fn apply_dyntopo(
        &mut self,
        id: MeshId,
        center: [f32; 3],
        radius: f32,
        options: &DyntopoOptions,
    ) -> Result<DyntopoPatch>;

    /// Runs localized dynamic-topology refinement while locking `pinned` vertices.
    fn apply_dyntopo_protected(
        &mut self,
        id: MeshId,
        center: [f32; 3],
        radius: f32,
        options: &DyntopoOptions,
        pinned: &[VertexId],
    ) -> Result<DyntopoPatch>;

    /// Runs localized dynamic-topology refinement followed by a sculpt dab on mesh `id`.
    fn dyntopo_dab(
        &mut self,
        id: MeshId,
        dab: &Dab,
        options: &DyntopoOptions,
    ) -> Result<(DyntopoPatch, DabReport)>;

    /// Runs localized dynamic-topology refinement followed by a sculpt dab with vertex
    /// protection weights and a geodesic falloff band on mesh `id`.
    fn dyntopo_dab_protected_band(
        &mut self,
        id: MeshId,
        dab: &Dab,
        options: &DyntopoOptions,
        vertices: &[VertexId],
        weights: &[f32],
        band: f32,
    ) -> Result<(DyntopoPatch, DabReport)>;
}

impl SceneSculptOps for Scene {
    fn dab(&mut self, id: MeshId, dab: &Dab) -> Result<DabReport> {
        Scene::dab(self, id, dab)
    }

    fn capture_stroke_state(&mut self, id: MeshId, only_if_missing: bool) -> Result<()> {
        Scene::capture_stroke_state(self, id, only_if_missing)
    }

    fn capture_rest_sizes(&mut self, id: MeshId, only_if_missing: bool) -> Result<()> {
        Scene::capture_rest_sizes(self, id, only_if_missing)
    }

    fn split_stretched(&mut self, id: MeshId, options: &StretchOptions) -> Result<StretchReport> {
        Scene::split_stretched(self, id, options)
    }

    fn vertices_in_sphere(
        &mut self,
        id: MeshId,
        center: [f32; 3],
        radius: f32,
    ) -> Result<Vec<VertexId>> {
        Scene::vertices_in_sphere(self, id, center, radius)
    }

    fn apply_dyntopo(
        &mut self,
        id: MeshId,
        center: [f32; 3],
        radius: f32,
        options: &DyntopoOptions,
    ) -> Result<DyntopoPatch> {
        Scene::apply_dyntopo(self, id, center, radius, options)
    }

    fn apply_dyntopo_protected(
        &mut self,
        id: MeshId,
        center: [f32; 3],
        radius: f32,
        options: &DyntopoOptions,
        pinned: &[VertexId],
    ) -> Result<DyntopoPatch> {
        Scene::apply_dyntopo_protected(self, id, center, radius, options, pinned)
    }

    fn dyntopo_dab(
        &mut self,
        id: MeshId,
        dab: &Dab,
        options: &DyntopoOptions,
    ) -> Result<(DyntopoPatch, DabReport)> {
        Scene::dyntopo_dab(self, id, dab, options)
    }

    fn dyntopo_dab_protected_band(
        &mut self,
        id: MeshId,
        dab: &Dab,
        options: &DyntopoOptions,
        vertices: &[VertexId],
        weights: &[f32],
        band: f32,
    ) -> Result<(DyntopoPatch, DabReport)> {
        Scene::dyntopo_dab_protected_band(self, id, dab, options, vertices, weights, band)
    }
}

fn resolve_partial_bridge_arc(
    loops: &[crate::ops::BoundaryLoop],
    arc: &[VertexId],
) -> std::result::Result<(usize, (usize, usize)), crate::ops::OpsError> {
    if arc.len() < 2 {
        return Err(crate::ops::OpsError::InvalidInput(
            "bridge arc requires at least two vertices",
        ));
    }
    for (loop_idx, b_loop) in loops.iter().enumerate() {
        let lv = &b_loop.vertices;
        let n = lv.len();
        if n < 2 || arc.len() > n {
            continue;
        }
        let Some(start) = lv.iter().position(|&v| v == arc[0]) else {
            continue;
        };
        if (0..arc.len()).all(|k| lv[(start + k) % n] == arc[k]) {
            let end = (start + arc.len() - 1) % n;
            return Ok((loop_idx, (start, end)));
        }
        if (0..arc.len()).all(|k| lv[(start + n - k) % n] == arc[k]) {
            let end_fwd = (start + n - (arc.len() - 1)) % n;
            return Ok((loop_idx, (end_fwd, start)));
        }
        if arc.len() == 2 && arc[0] != arc[1] && lv.contains(&arc[1]) {
            let end = lv.iter().position(|&v| v == arc[1]).expect("checked above");
            return Ok((loop_idx, (start, end)));
        }
    }
    Err(crate::ops::OpsError::InvalidInput(
        "bridge arc vertices do not form a contiguous boundary arc",
    ))
}

/// Remesh domain operations on [`Scene`].
pub trait SceneRemeshOps {
    /// Remeshes a spherical region of mesh `id` toward a target edge length.
    fn remesh(&mut self, id: MeshId, options: &RemeshOptions) -> Result<RemeshReport>;

    /// Simplifies mesh `id` via quadric-error edge collapse.
    fn simplify(&mut self, id: MeshId, options: &SimplifyOptions) -> Result<SimplifyReport>;

    /// Subdivides mesh `id` with Loop subdivision.
    fn subdivide(&mut self, id: MeshId, options: &SubdivideOptions) -> Result<SubdivideReport>;
}

impl SceneRemeshOps for Scene {
    fn remesh(&mut self, id: MeshId, options: &RemeshOptions) -> Result<RemeshReport> {
        self.edit(id, |tx| crate::ops::remesh::remesh(tx, options))
    }

    fn simplify(&mut self, id: MeshId, options: &SimplifyOptions) -> Result<SimplifyReport> {
        self.edit(id, |tx| crate::ops::remesh::simplify(tx, options))
    }

    fn subdivide(&mut self, id: MeshId, options: &SubdivideOptions) -> Result<SubdivideReport> {
        self.edit(id, |tx| crate::ops::remesh::subdivide(tx, options))
    }
}

/// Boolean and solid modeling domain operations on [`Scene`].
pub trait SceneBooleanOps {
    /// Computes a world-space CSG boolean (`op`) between `a` and `b`, updating `a`.
    fn boolean(&mut self, a: MeshId, b: MeshId, op: BooleanOp) -> Result<BooleanReport>;

    /// Computes a world-space CSG boolean with configurable solver options.
    fn boolean_with_options(
        &mut self,
        a: MeshId,
        b: MeshId,
        options: &BooleanOptions,
    ) -> Result<BooleanReport>;

    /// Evaluates an N-ary CSG boolean tree over `meshes` and adds the resulting mesh.
    fn boolean_batch(
        &mut self,
        meshes: &[MeshId],
        tree: &CsgNode,
        options: &BooleanOptions,
    ) -> Result<MeshId>;

    /// Cuts mesh `id` with a world plane and adds the removed side to the scene.
    fn cut_by_plane(&mut self, id: MeshId, options: &CutOptions) -> Result<CutReport>;

    /// Conformatively imprints a world polyline on `id` and optionally trims the inside or
    /// outside of the component it lies on.
    fn trim_by_curve(&mut self, id: MeshId, options: &TrimCurveOptions) -> Result<TrimReport>;

    /// Offsets mesh `id` along its normals or via volumetric SDF reconstruction.
    fn offset(&mut self, id: MeshId, options: &OffsetOptions) -> Result<OffsetReport>;

    /// Solidifies mesh `id` into a walled shell of the given thickness.
    fn shell(&mut self, id: MeshId, options: &ShellOptions) -> Result<ShellReport>;

    /// Hollows solid mesh `id` by carving an inner cavity.
    fn hollow_mesh(&mut self, id: MeshId, options: &HollowOptions) -> Result<HollowReport>;

    /// Punches a cylindrical/tapered drain hole into `id` in world coordinates.
    fn punch_drain_hole_world(
        &mut self,
        id: MeshId,
        world_spec: &DrainHoleSpec,
        options: &DrainHoleOptions,
    ) -> Result<(DrainHoleReport, Vec<MeshId>)>;

    /// Places `stamp` onto `target` at a world surface point and normal.
    fn place_mesh_world(
        &mut self,
        target: MeshId,
        stamp: MeshId,
        hit_point_world: Vec3,
        hit_normal_world: Vec3,
        params: &PlaceMeshParams,
    ) -> Result<(PlaceMeshReport, Option<MeshId>)>;
}

impl SceneBooleanOps for Scene {
    fn boolean(&mut self, a: MeshId, b: MeshId, op: BooleanOp) -> Result<BooleanReport> {
        Scene::boolean(self, a, b, op)
    }

    fn boolean_with_options(
        &mut self,
        a: MeshId,
        b: MeshId,
        options: &BooleanOptions,
    ) -> Result<BooleanReport> {
        Scene::boolean_with_options(self, a, b, options)
    }

    fn boolean_batch(
        &mut self,
        meshes: &[MeshId],
        tree: &CsgNode,
        options: &BooleanOptions,
    ) -> Result<MeshId> {
        Scene::boolean_batch(self, meshes, tree, options)
    }

    fn cut_by_plane(&mut self, id: MeshId, options: &CutOptions) -> Result<CutReport> {
        Scene::cut_by_plane(self, id, options)
    }

    fn trim_by_curve(&mut self, id: MeshId, options: &TrimCurveOptions) -> Result<TrimReport> {
        Scene::trim_by_curve(self, id, options)
    }

    fn offset(&mut self, id: MeshId, options: &OffsetOptions) -> Result<OffsetReport> {
        self.edit(id, |tx| crate::ops::boolean::offset(tx, options))
    }

    fn shell(&mut self, id: MeshId, options: &ShellOptions) -> Result<ShellReport> {
        self.edit(id, |tx| crate::ops::boolean::shell(tx, options))
    }

    fn hollow_mesh(&mut self, id: MeshId, options: &HollowOptions) -> Result<HollowReport> {
        Scene::hollow_mesh(self, id, options)
    }

    fn punch_drain_hole_world(
        &mut self,
        id: MeshId,
        world_spec: &DrainHoleSpec,
        options: &DrainHoleOptions,
    ) -> Result<(DrainHoleReport, Vec<MeshId>)> {
        Scene::punch_drain_hole_world(self, id, world_spec, options)
    }

    fn place_mesh_world(
        &mut self,
        target: MeshId,
        stamp: MeshId,
        hit_point_world: Vec3,
        hit_normal_world: Vec3,
        params: &PlaceMeshParams,
    ) -> Result<(PlaceMeshReport, Option<MeshId>)> {
        Scene::place_mesh_world(
            self,
            target,
            stamp,
            hit_point_world,
            hit_normal_world,
            params,
        )
    }
}

/// Deform domain operations (with vertex protection) on [`Scene`].
pub trait SceneDeformOps {
    /// Applies a brush dab with per-vertex protection weights.
    fn dab_protected(
        &mut self,
        id: MeshId,
        dab: &Dab,
        vertices: &[VertexId],
        weights: &[f32],
    ) -> Result<DabReport>;

    /// Applies a brush dab with per-vertex protection and a geodesic soft-falloff band.
    fn dab_protected_band(
        &mut self,
        id: MeshId,
        dab: &Dab,
        vertices: &[VertexId],
        weights: &[f32],
        band: f32,
    ) -> Result<DabReport>;

    /// Builds an identity-offset FFD control lattice enclosing `id`.
    fn build_ffd_lattice(
        &self,
        id: MeshId,
        dims: [usize; 3],
        basis: FfdBasis,
        padding: f64,
    ) -> Result<FfdLattice>;

    /// Applies an atomic freeform deformation (FFD, ARAP, Kelvinlet, or XPBD) on `id`.
    fn apply_freeform(&mut self, id: MeshId, op: &FreeformOp) -> Result<FreeformReport>;

    /// Applies a freeform deformation on `id` with optional per-vertex protection weights.
    fn apply_freeform_protected(
        &mut self,
        id: MeshId,
        op: &FreeformOp,
        protected_vertices: &[u32],
        protection_weights: Option<&[f32]>,
    ) -> Result<FreeformReport>;
}

impl SceneDeformOps for Scene {
    fn dab_protected(
        &mut self,
        id: MeshId,
        dab: &Dab,
        vertices: &[VertexId],
        weights: &[f32],
    ) -> Result<DabReport> {
        Scene::dab_protected(self, id, dab, vertices, weights)
    }

    fn dab_protected_band(
        &mut self,
        id: MeshId,
        dab: &Dab,
        vertices: &[VertexId],
        weights: &[f32],
        band: f32,
    ) -> Result<DabReport> {
        Scene::dab_protected_band(self, id, dab, vertices, weights, band)
    }

    fn build_ffd_lattice(
        &self,
        id: MeshId,
        dims: [usize; 3],
        basis: FfdBasis,
        padding: f64,
    ) -> Result<FfdLattice> {
        Scene::build_ffd_lattice(self, id, dims, basis, padding)
    }

    fn apply_freeform(&mut self, id: MeshId, op: &FreeformOp) -> Result<FreeformReport> {
        Scene::apply_freeform(self, id, op)
    }

    fn apply_freeform_protected(
        &mut self,
        id: MeshId,
        op: &FreeformOp,
        protected_vertices: &[u32],
        protection_weights: Option<&[f32]>,
    ) -> Result<FreeformReport> {
        Scene::apply_freeform_protected(self, id, op, protected_vertices, protection_weights)
    }
}

/// Inspection, query, field, and section domain operations on [`Scene`].
pub trait SceneInspectOps {
    /// Runs read-only defect analysis on mesh `id`.
    fn analyze(&self, id: MeshId, options: &AnalyzeOptions<'_>) -> Result<Analysis>;

    /// Computes surface area and watertight volume measurements on mesh `id`.
    fn measure(&self, id: MeshId) -> Result<MeasureReport>;

    /// Extracts boundary loops and open chains from mesh `id`.
    fn boundary_loops(&self, id: MeshId) -> Result<Vec<BoundaryLoop>>;

    /// Casts a world-space ray across the scene.
    fn raycast(
        &mut self,
        origin: [f64; 3],
        dir: [f64; 3],
        opts: &SceneRayOptions,
    ) -> Result<Option<SceneRayHit>>;

    /// Finds the closest surface point across the scene to world point `p`.
    fn closest_point(
        &mut self,
        p: [f64; 3],
        opts: &SceneClosestOptions,
    ) -> Result<Option<SceneClosestHit>>;

    /// Finds the closest surface point on mesh `id` to world point `p`.
    fn closest_point_on(
        &mut self,
        id: MeshId,
        p: [f64; 3],
        opts: &SceneClosestOptions,
    ) -> Result<Option<SceneClosestHit>>;

    /// Checks exact triangle intersection between meshes `a` and `b`.
    fn intersects(&mut self, a: MeshId, b: MeshId) -> Result<bool>;

    /// Computes the minimum world distance between meshes `a` and `b` within `max_distance`.
    fn mesh_distance(
        &mut self,
        a: MeshId,
        b: MeshId,
        max_distance: f64,
    ) -> Result<Option<SceneMeshDistance>>;

    /// Collects mesh pair contacts within `margin`.
    fn find_contacts(
        &mut self,
        margin: f64,
        filter: &ContactFilter,
        out: &mut Vec<Contact>,
    ) -> Result<usize>;

    /// Registers a distance field on the scene.
    fn add_distance_field(&mut self, desc: DistanceField) -> Result<FieldId>;

    /// Registers a wall-thickness field on the scene.
    fn add_thickness_field(&mut self, desc: ThicknessField) -> Result<FieldId>;

    /// Registers a curvature field on the scene.
    fn add_curvature_field(&mut self, desc: CurvatureField) -> Result<FieldId>;

    /// Registers a cross-section cut plane on the scene.
    fn add_section(&mut self, desc: SectionDesc) -> Result<SectionId>;

    /// Returns the world-space axis-aligned bounding box of mesh `id`.
    fn bounds(&self, id: MeshId) -> Result<Option<Aabb>>;

    /// Computes printability overhang angles relative to a world build direction.
    fn overhangs(
        &self,
        id: MeshId,
        options: &OverhangOptions,
        vertex_angles_deg: &mut Vec<f32>,
    ) -> Result<OverhangReport>;

    /// Evaluates vertex-to-surface deviation from `source` to `target` in world units.
    fn mesh_deviation(
        &mut self,
        source: MeshId,
        target: MeshId,
        options: &DeviationOptions,
        out: &mut Vec<f32>,
    ) -> Result<DeviationStats>;

    /// Evaluates triangle shape quality in world coordinates.
    fn triangle_quality(
        &self,
        id: MeshId,
        metric: QualityMetric,
        sliver_threshold: f32,
        vertex_quality: &mut Vec<f32>,
    ) -> Result<QualityReport>;

    /// Computes geodesic surface distances from `sources` in world length units.
    fn geodesic_distances(
        &self,
        id: MeshId,
        sources: &[u32],
        options: &GeodesicOptions,
        out: &mut Vec<f32>,
    ) -> Result<()>;

    /// Traces the intrinsic shortest path from `sources` to `target_vertex`.
    fn shortest_surface_path(
        &self,
        id: MeshId,
        distances: &[f32],
        sources: &[u32],
        target_vertex: u32,
        out: &mut Vec<SurfacePathPoint>,
    ) -> Result<()>;

    /// Fits an analytical geometric primitive to `id` (or `subset_tris`) in world space.
    fn fit_primitive_world(
        &self,
        id: MeshId,
        subset_tris: Option<&[u32]>,
        kind: Option<PrimitiveKind>,
        options: &FitPrimitiveOptions,
    ) -> Result<FitPrimitiveReport>;

    /// Rigidly aligns `id`'s world placement so `primitive` aligns to `(target_axis, target_origin)`.
    fn align_to_fitted_primitive(
        &mut self,
        id: MeshId,
        primitive: &FittedPrimitive,
        target_axis: Vec3,
        target_origin: Vec3,
    ) -> Result<Placement>;

    /// Slices closed and open polyline contours through `id` on a world plane.
    fn slice_contours_world(&self, id: MeshId, origin: Vec3, normal: Vec3) -> Result<ContourSlice>;

    /// Slices `count` parallel contour layers separated by world `spacing`.
    fn slice_contour_stack_world(
        &self,
        id: MeshId,
        origin: Vec3,
        normal: Vec3,
        spacing: f64,
        count: usize,
    ) -> Result<Vec<ContourSlice>>;

    /// Reconstructs a surface mesh from a point cloud and adds it to the scene.
    fn reconstruct_point_cloud(
        &mut self,
        cloud: &PointCloud,
        options: &SurfaceReconstructOptions,
    ) -> Result<MeshId>;
}

impl SceneInspectOps for Scene {
    fn analyze(&self, id: MeshId, options: &AnalyzeOptions<'_>) -> Result<Analysis> {
        Ok(crate::ops::inspect::analyze(self.mesh(id)?, options)?)
    }

    fn measure(&self, id: MeshId) -> Result<MeasureReport> {
        Ok(crate::ops::inspect::measure(self.mesh(id)?)?)
    }

    fn boundary_loops(&self, id: MeshId) -> Result<Vec<BoundaryLoop>> {
        Ok(crate::ops::inspect::boundary_loops(self.mesh(id)?)?)
    }

    fn raycast(
        &mut self,
        origin: [f64; 3],
        dir: [f64; 3],
        opts: &SceneRayOptions,
    ) -> Result<Option<SceneRayHit>> {
        Scene::raycast(self, origin, dir, opts)
    }

    fn closest_point(
        &mut self,
        p: [f64; 3],
        opts: &SceneClosestOptions,
    ) -> Result<Option<SceneClosestHit>> {
        Scene::closest_point(self, p, opts)
    }

    fn closest_point_on(
        &mut self,
        id: MeshId,
        p: [f64; 3],
        opts: &SceneClosestOptions,
    ) -> Result<Option<SceneClosestHit>> {
        Scene::closest_point_on(self, id, p, opts)
    }

    fn intersects(&mut self, a: MeshId, b: MeshId) -> Result<bool> {
        Scene::intersects(self, a, b)
    }

    fn mesh_distance(
        &mut self,
        a: MeshId,
        b: MeshId,
        max_distance: f64,
    ) -> Result<Option<SceneMeshDistance>> {
        Scene::mesh_distance(self, a, b, max_distance)
    }

    fn find_contacts(
        &mut self,
        margin: f64,
        filter: &ContactFilter,
        out: &mut Vec<Contact>,
    ) -> Result<usize> {
        Scene::find_contacts(self, margin, filter, out)
    }

    fn add_distance_field(&mut self, desc: DistanceField) -> Result<FieldId> {
        Scene::add_distance_field(self, desc)
    }

    fn add_thickness_field(&mut self, desc: ThicknessField) -> Result<FieldId> {
        Scene::add_thickness_field(self, desc)
    }

    fn add_curvature_field(&mut self, desc: CurvatureField) -> Result<FieldId> {
        Scene::add_curvature_field(self, desc)
    }

    fn add_section(&mut self, desc: SectionDesc) -> Result<SectionId> {
        Scene::add_section(self, desc)
    }

    fn bounds(&self, id: MeshId) -> Result<Option<Aabb>> {
        Scene::bounds(self, id)
    }

    fn overhangs(
        &self,
        id: MeshId,
        options: &OverhangOptions,
        vertex_angles_deg: &mut Vec<f32>,
    ) -> Result<OverhangReport> {
        Scene::overhangs(self, id, options, vertex_angles_deg)
    }

    fn mesh_deviation(
        &mut self,
        source: MeshId,
        target: MeshId,
        options: &DeviationOptions,
        out: &mut Vec<f32>,
    ) -> Result<DeviationStats> {
        Scene::mesh_deviation(self, source, target, options, out)
    }

    fn triangle_quality(
        &self,
        id: MeshId,
        metric: QualityMetric,
        sliver_threshold: f32,
        vertex_quality: &mut Vec<f32>,
    ) -> Result<QualityReport> {
        Scene::triangle_quality(self, id, metric, sliver_threshold, vertex_quality)
    }

    fn geodesic_distances(
        &self,
        id: MeshId,
        sources: &[u32],
        options: &GeodesicOptions,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        Scene::geodesic_distances(self, id, sources, options, out)
    }

    fn shortest_surface_path(
        &self,
        id: MeshId,
        distances: &[f32],
        sources: &[u32],
        target_vertex: u32,
        out: &mut Vec<SurfacePathPoint>,
    ) -> Result<()> {
        Scene::shortest_surface_path(self, id, distances, sources, target_vertex, out)
    }

    fn fit_primitive_world(
        &self,
        id: MeshId,
        subset_tris: Option<&[u32]>,
        kind: Option<PrimitiveKind>,
        options: &FitPrimitiveOptions,
    ) -> Result<FitPrimitiveReport> {
        Scene::fit_primitive_world(self, id, subset_tris, kind, options)
    }

    fn align_to_fitted_primitive(
        &mut self,
        id: MeshId,
        primitive: &FittedPrimitive,
        target_axis: Vec3,
        target_origin: Vec3,
    ) -> Result<Placement> {
        Scene::align_to_fitted_primitive(self, id, primitive, target_axis, target_origin)
    }

    fn slice_contours_world(&self, id: MeshId, origin: Vec3, normal: Vec3) -> Result<ContourSlice> {
        Scene::slice_contours_world(self, id, origin, normal)
    }

    fn slice_contour_stack_world(
        &self,
        id: MeshId,
        origin: Vec3,
        normal: Vec3,
        spacing: f64,
        count: usize,
    ) -> Result<Vec<ContourSlice>> {
        Scene::slice_contour_stack_world(self, id, origin, normal, spacing, count)
    }

    fn reconstruct_point_cloud(
        &mut self,
        cloud: &PointCloud,
        options: &SurfaceReconstructOptions,
    ) -> Result<MeshId> {
        Scene::reconstruct_point_cloud(self, cloud, options)
    }
}

/// Alignment domain operations on [`Scene`].
pub trait SceneAlignOps {
    /// Rigidly registers `moving` onto `fixed` via trimmed point-to-plane ICP.
    fn register(
        &mut self,
        moving: MeshId,
        fixed: MeshId,
        options: &RegisterOptions,
    ) -> Result<RegisterReport>;

    /// Aligns `moving` onto `fixed` from landmark correspondences.
    fn align_by_landmarks(
        &mut self,
        moving: MeshId,
        fixed: MeshId,
        pairs: &[LandmarkPair],
        allow_scaling: bool,
    ) -> Result<LandmarkAlignment>;

    /// Runs multi-stage global + fine registration of `moving` onto `fixed`.
    fn register_multistage(
        &mut self,
        moving: MeshId,
        fixed: MeshId,
        options: &RegistrationOptions,
    ) -> Result<RegistrationReport>;

    /// Detects the dominant reflection symmetry plane of mesh `id`.
    fn detect_symmetry_plane(&mut self, id: MeshId) -> Result<SymmetryPlaneReport>;
}

impl SceneAlignOps for Scene {
    fn register(
        &mut self,
        moving: MeshId,
        fixed: MeshId,
        options: &RegisterOptions,
    ) -> Result<RegisterReport> {
        Scene::register(self, moving, fixed, options)
    }

    fn align_by_landmarks(
        &mut self,
        moving: MeshId,
        fixed: MeshId,
        pairs: &[LandmarkPair],
        allow_scaling: bool,
    ) -> Result<LandmarkAlignment> {
        Scene::align_by_landmarks(self, moving, fixed, pairs, allow_scaling)
    }

    fn register_multistage(
        &mut self,
        moving: MeshId,
        fixed: MeshId,
        options: &RegistrationOptions,
    ) -> Result<RegistrationReport> {
        Scene::register_multistage(self, moving, fixed, options)
    }

    fn detect_symmetry_plane(&mut self, id: MeshId) -> Result<SymmetryPlaneReport> {
        Scene::detect_symmetry_plane(self, id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::repair::AutoRepairPreset;
    use crate::ops::sculpt::{Brush, Falloff, InflateOptions};
    use crate::{MeshOptions, Plane, SceneOptions};
    use fast_mesh_store::Mesh;

    fn unit_box_mesh() -> Mesh {
        let positions = [
            0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
        ];
        let indices = [
            0, 2, 1, 0, 3, 2, // -z
            4, 5, 6, 4, 6, 7, // +z
            0, 1, 5, 0, 5, 4, // -y
            2, 3, 7, 2, 7, 6, // +y
            0, 4, 7, 0, 7, 3, // -x
            1, 2, 6, 1, 6, 5, // +x
        ];
        Mesh::from_buffers(&positions, &indices).unwrap()
    }

    #[test]
    fn scene_domain_extension_traits_delegate_cleanly() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene
            .add_mesh(unit_box_mesh(), MeshOptions::default())
            .unwrap();

        let report = SceneInspectOps::measure(&scene, id).unwrap();
        assert!(report.closed);
        assert!((report.volume.unwrap() - 1.0).abs() < 1e-6);

        let sub = SceneRemeshOps::subdivide(
            &mut scene,
            id,
            &SubdivideOptions::builder().levels(1).build(),
        )
        .unwrap();
        assert!(sub.triangles_added > 12);

        let sim = SceneRemeshOps::simplify(
            &mut scene,
            id,
            &SimplifyOptions::builder().target_triangles(24).build(),
        )
        .unwrap();
        assert!(sim.collapsed > 0);

        let cut = SceneBooleanOps::cut_by_plane(
            &mut scene,
            id,
            &CutOptions {
                plane: Plane::new([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap(),
                cap: true,
            },
        )
        .unwrap();
        assert!(cut.kept_triangles > 0 && cut.removed_triangles > 0);

        let alive_v = (0..scene.mesh(id).unwrap().vertex_slots() as VertexId)
            .find(|&v| scene.mesh(id).unwrap().position(v).is_some())
            .unwrap();
        let dab = Dab::Inflate(InflateOptions {
            brush: Brush {
                center: [0.5, 0.5, 0.5],
                radius: 2.0,
                falloff: Falloff::Smooth,
                strength: 0.05,
            },
            ..Default::default()
        });
        let d_rep =
            SceneDeformOps::dab_protected(&mut scene, id, &dab, &[alive_v], &[0.0]).unwrap();
        assert!(d_rep.moved > 0);

        let repair_rep = SceneRepairOps::auto_repair(
            &mut scene,
            id,
            &AutoRepairOptions::for_preset(AutoRepairPreset::General),
        )
        .unwrap();
        assert_eq!(repair_rep.weld.merged_vertices, 0);

        let other = scene
            .add_mesh(unit_box_mesh(), MeshOptions::default())
            .unwrap();
        let align_rep =
            SceneAlignOps::register(&mut scene, id, other, &RegisterOptions::default()).unwrap();
        assert!(align_rep.samples > 0);

        let sym = SceneAlignOps::detect_symmetry_plane(&mut scene, other).unwrap();
        assert!(sym.symmetry_error_rms.is_finite());

        let mut angles = Vec::new();
        let ov =
            SceneInspectOps::overhangs(&scene, other, &OverhangOptions::default(), &mut angles)
                .unwrap();
        assert!(ov.overhang_face_count > 0);

        let mut qual = Vec::new();
        let qr = SceneInspectOps::triangle_quality(
            &scene,
            other,
            QualityMetric::RadiusRatio,
            0.1,
            &mut qual,
        )
        .unwrap();
        assert!(qr.live_triangle_count > 0);

        let mut dist = Vec::new();
        SceneInspectOps::geodesic_distances(
            &scene,
            other,
            &[0],
            &GeodesicOptions::default(),
            &mut dist,
        )
        .unwrap();
        assert_eq!(dist[0], 0.0);

        let mut path = Vec::new();
        SceneInspectOps::shortest_surface_path(&scene, other, &dist, &[0], 1, &mut path).unwrap();
        assert!(path.len() >= 2);

        let mut dev = Vec::new();
        let ds = SceneInspectOps::mesh_deviation(
            &mut scene,
            other,
            other,
            &DeviationOptions::default(),
            &mut dev,
        )
        .unwrap();
        assert!(ds.sample_count > 0);
    }

    #[test]
    fn scene_domain_dyntopo_and_continuity_repair_delegate_cleanly() {
        use fast_mesh_ops::repair::FillContinuity;
        use fast_mesh_ops::sculpt::DyntopoMode;

        // Open-box mesh with 1 missing quad face (2 triangles -> 4-vertex hole loop)
        let mut positions = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
        ];
        // Build a 4x4 grid missing the center cell [1..2, 1..2] -> outer rim + center hole loop
        positions.clear();
        for y in 0..4u32 {
            for x in 0..4u32 {
                positions.extend_from_slice(&[x as f32, y as f32, 0.0]);
            }
        }
        let mut indices = Vec::new();
        for y in 0..3u32 {
            for x in 0..3u32 {
                if x == 1 && y == 1 {
                    continue;
                }
                let v0 = y * 4 + x;
                let v1 = v0 + 1;
                let v2 = v0 + 5;
                let v3 = v0 + 4;
                indices.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
            }
        }
        let hole_mesh = Mesh::from_buffers(&positions, &indices).unwrap();
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(hole_mesh, MeshOptions::default()).unwrap();

        // Find the inner 4-vertex hole loop index via boundary_loops
        let loops = fast_mesh_ops::boundary_loops(scene.mesh(id).unwrap()).unwrap();
        let inner_idx = loops.iter().position(|l| l.vertices.len() == 4).unwrap();
        let inner_stats = SceneRepairOps::fill_hole_continuity(
            &mut scene,
            id,
            inner_idx,
            &FillHolesOptions::builder()
                .continuity(FillContinuity::CurvatureG2)
                .refine_patch(true)
                .build(),
        )
        .unwrap();
        assert!(inner_stats.filled_triangles >= 2);
        assert_eq!(inner_stats.continuity, FillContinuity::CurvatureG2);

        scene.undo().unwrap();
        let all_stats = SceneRepairOps::fill_all_holes_continuity(
            &mut scene,
            id,
            &FillHolesOptions::builder()
                .continuity(FillContinuity::TangentG1)
                .refine_patch(false)
                .build(),
        )
        .unwrap();
        assert_eq!(all_stats.filled_triangles, 2);

        scene.undo().unwrap();
        let inner_verts = loops[inner_idx].vertices.clone();
        let bridge_stats = SceneRepairOps::bridge_partial_arcs(
            &mut scene,
            id,
            &inner_verts[0..2],
            &inner_verts[2..4],
            &PartialBridgeOptions::default(),
        )
        .unwrap();
        assert!(bridge_stats.triangles_added > 0);

        let dyntopo_opts = DyntopoOptions {
            mode: DyntopoMode::SplitOnly,
            detail_size: 0.4,
            ..Default::default()
        };
        let patch =
            SceneSculptOps::apply_dyntopo(&mut scene, id, [1.5, 1.5, 0.0], 1.2, &dyntopo_opts)
                .unwrap();
        assert!(patch.splits > 0);

        let alive_v = (0..scene.mesh(id).unwrap().vertex_slots() as VertexId)
            .find(|&v| scene.mesh(id).unwrap().position(v).is_some())
            .unwrap();
        let _ = SceneSculptOps::apply_dyntopo_protected(
            &mut scene,
            id,
            [1.5, 1.5, 0.0],
            0.8,
            &dyntopo_opts,
            &[alive_v],
        )
        .unwrap();

        let dab = Dab::Inflate(InflateOptions {
            brush: Brush {
                center: [1.5, 1.5, 0.0],
                radius: 1.0,
                falloff: Falloff::Smooth,
                strength: 0.1,
            },
            ..Default::default()
        });
        let (_, dab_rep) =
            SceneSculptOps::dyntopo_dab(&mut scene, id, &dab, &dyntopo_opts).unwrap();
        assert!(dab_rep.moved > 0);
        let (_, band_rep) = SceneSculptOps::dyntopo_dab_protected_band(
            &mut scene,
            id,
            &dab,
            &dyntopo_opts,
            &[alive_v],
            &[0.0],
            0.25,
        )
        .unwrap();
        assert!(band_rep.moved > 0);
    }
}
