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
use fast_mesh_ops::boolean::{OffsetOptions, OffsetReport, ShellOptions, ShellReport};
use fast_mesh_ops::inspect::{Analysis, AnalyzeOptions, BoundaryLoop, MeasureReport};
use fast_mesh_ops::remesh::{
    RemeshOptions, RemeshReport, SimplifyOptions, SimplifyReport, SubdivideOptions, SubdivideReport,
};
use fast_mesh_ops::repair::{
    AutoRepairOptions, AutoRepairReport, ComponentOptions, DegenerateOptions, DegenerateReport,
    FillOptions, FillReport, ManifoldOptions, ManifoldReport, OrientReport, RemoveReport,
    SelfIntersectOptions, SelfIntersectReport, SliverOptions, SliverReport, WeldReport,
};
use fast_mesh_ops::sculpt::{StretchOptions, StretchReport};
use fast_mesh_store::VertexId;

use crate::{
    BooleanOp, BooleanOptions, BooleanReport, Contact, ContactFilter, CsgNode, CurvatureField,
    CutOptions, CutReport, Dab, DabReport, DeviationOptions, DeviationStats, DistanceField,
    FieldId, GeodesicOptions, JoinOptions, JoinReport, LandmarkAlignment, LandmarkPair, MeshId,
    OverhangOptions, OverhangReport, QualityMetric, QualityReport, RegisterOptions, RegisterReport,
    RegistrationOptions, RegistrationReport, Result, Scene, SceneClosestHit, SceneClosestOptions,
    SceneMeshDistance, SceneRayHit, SceneRayOptions, SectionDesc, SectionId, SurfacePathPoint,
    SymmetryPlaneReport, ThicknessField, TrimCurveOptions, TrimReport,
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

    /// Conformatively imprints a world polyline on `id` and optionally trims inside/outside.
    fn trim_by_curve(&mut self, id: MeshId, options: &TrimCurveOptions) -> Result<TrimReport>;

    /// Offsets mesh `id` along its normals or via volumetric SDF reconstruction.
    fn offset(&mut self, id: MeshId, options: &OffsetOptions) -> Result<OffsetReport>;

    /// Solidifies mesh `id` into a walled shell of the given thickness.
    fn shell(&mut self, id: MeshId, options: &ShellOptions) -> Result<ShellReport>;
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
}
