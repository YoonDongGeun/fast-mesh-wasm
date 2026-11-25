//! Broadphase, pairwise contacts, and configurable inter-object collision responses.

use std::collections::{BTreeMap, BTreeSet};

use fast_mesh_bvh::{Aabb, BuildOptions, Bvh};
use fast_mesh_fields::{
    DEFAULT_MAX_VOXELS, GridSpec, SignData, SignMode, fh_signed_grid, signed_distances,
};
use fast_mesh_ops::boolean::BooleanMethod;
use fast_mesh_query::{
    ClosestHit, ClosestOptions, Rigid, closest_point_many, self_intersecting_pairs,
};
use fast_mesh_store::{Mesh, StoreError, TriangleId, VertexId};

use crate::baked::{baked_mesh, baked_pose, world_base};
use crate::placement::{
    interpolate, is_relative_rigid, local_to_world, relative_affine, relative_rigid, world_to_local,
};
use crate::query::{Aabb64, MARGIN, cached_world_box, check_bound, pair_pad, round_up, try_vec};
use crate::scene::Entry;
use crate::{BooleanOptions, MeshId, Placement, Result, Scene, SceneError};

/// Two meshes within the margin of [`Scene::find_contacts`].
#[derive(Debug, Clone, PartialEq)]
pub struct Contact {
    /// The mesh with the lower index.
    pub a: MeshId,
    /// The mesh with the higher index.
    pub b: MeshId,
    /// Whether the two meshes intersect or touch, decided exactly.
    pub intersecting: bool,
    /// The distance between the two surfaces; 0 when `intersecting`.
    pub distance: f64,
    /// The triangles of `a` within the margin of `b`, ascending.
    pub triangles_a: Vec<TriangleId>,
    /// The triangles of `b` within the margin of `a`, ascending.
    pub triangles_b: Vec<TriangleId>,
}

/// Which mesh pairs [`Scene::find_contacts`] looks at.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ContactFilter {
    /// Only these pairs, in any order and either way round; `None` means
    /// every pair of meshes.
    pub pairs: Option<Vec<(MeshId, MeshId)>>,
}

impl Scene {
    /// Appends a [`Contact`] for every mesh pair (of `filter`, or of the whole
    /// scene) whose surfaces are within `margin` of each other as placed in
    /// the world, sorted by `(a, b)`, and returns how many were appended.
    /// Elements already in `out` stay in front.
    ///
    /// The broadphase keeps the pairs whose world boxes (BVH root boxes
    /// placed by the rigid transforms) are within `margin`, padded so that it
    /// never drops a pair Phase 3b would report. For each, in A's space:
    /// `intersecting` comes from [`fast_mesh_query::intersects`]; `distance`
    /// is 0 when intersecting, else from [`fast_mesh_query::mesh_distance`]
    /// with `margin` rounded up to `f32`, and a pair whose distance exceeds
    /// the exact `margin` is not reported; the triangle lists come from
    /// [`fast_mesh_query::triangles_near`] with the rounded-up margin (see
    /// [`Scene::triangles_near`]).
    ///
    /// With `filter.pairs`, each pair may be given either way round and more
    /// than once; only the listed meshes' BVHs are brought up to date.
    /// Without, every Lazy BVH is (§15.4).
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] for a negative or NaN `margin` or a pair
    /// naming one mesh twice; [`SceneError::UnknownMesh`] for a stale id in
    /// the filter; then a BVH or memory error from a Lazy update,
    /// [`SceneError::StaleBvh`] for the lowest stale Manual mesh involved, or
    /// a memory error while collecting. On any error `out` keeps its length.
    pub fn find_contacts(
        &mut self,
        margin: f64,
        filter: &ContactFilter,
        out: &mut Vec<Contact>,
    ) -> Result<usize> {
        self.drop_field_unit();
        let start = out.len();
        let found = self.find_contacts_into(margin, filter, out);
        if found.is_err() {
            out.truncate(start);
        }
        found
    }

    fn find_contacts_into(
        &mut self,
        margin: f64,
        filter: &ContactFilter,
        out: &mut Vec<Contact>,
    ) -> Result<usize> {
        check_bound(margin, MARGIN)?;
        // The meshes involved, ascending, and the pairs as indices into them.
        let (ids, pairs) = match &filter.pairs {
            Some(list) => {
                let mut pairs = try_vec(list.len())?;
                for &(a, b) in list {
                    self.entry(a)?;
                    self.entry(b)?;
                    if a == b {
                        return Err(SceneError::InvalidInput(
                            "a contact pair needs two different meshes",
                        ));
                    }
                    pairs.push((a.min(b), a.max(b)));
                }
                pairs.sort_unstable();
                pairs.dedup();
                let mut ids = try_vec(2 * pairs.len())?;
                for &(a, b) in &pairs {
                    ids.push(a);
                    ids.push(b);
                }
                ids.sort_unstable();
                ids.dedup();
                self.prepare(Some(&ids))?;
                (ids, Some(pairs))
            }
            None => {
                self.prepare(None)?;
                let mut ids = try_vec(self.entries.iter().count())?;
                ids.extend(self.ids());
                (ids, None)
            }
        };
        let mut boxes = try_vec(ids.len())?;
        for &id in &ids {
            boxes.push(cached_world_box(self.entry_mut(id)?).0);
        }
        let start = out.len();
        match pairs {
            Some(pairs) => {
                for (a, b) in pairs {
                    // Both are in `ids`, which is sorted.
                    let i = ids.binary_search(&a).map_err(|_| SceneError::Corrupted)?;
                    let j = ids.binary_search(&b).map_err(|_| SceneError::Corrupted)?;
                    self.contact((a, &boxes[i]), (b, &boxes[j]), margin, out)?;
                }
            }
            None => {
                for i in 0..ids.len() {
                    for j in i + 1..ids.len() {
                        self.contact((ids[i], &boxes[i]), (ids[j], &boxes[j]), margin, out)?;
                    }
                }
            }
        }
        Ok(out.len() - start)
    }

    /// Runs the broadphase test and, if it passes, the narrow phase for one
    /// pair (`a` below `b`), appending its contact if within `margin`.
    fn contact(
        &self,
        (a, wa): (MeshId, &Aabb64),
        (b, wb): (MeshId, &Aabb64),
        margin: f64,
        out: &mut Vec<Contact>,
    ) -> Result<()> {
        if wa.is_empty() || wb.is_empty() {
            return Ok(());
        }
        let (ea, eb) = (self.entry(a)?, self.entry(b)?);
        if wa.distance_to_box(wb) > margin + pair_pad(ea, wa, eb, wb) {
            return Ok(());
        }
        #[cfg(test)]
        crate::scene::fault::before_contact()?;
        if crate::placement::is_relative_rigid(&ea.placement, &eb.placement) {
            let xf = relative_rigid(
                &ea.placement,
                ea.mesh.origin(),
                &eb.placement,
                eb.mesh.origin(),
            );
            return self.contact_rigid(a, b, &xf, margin, out);
        }
        self.contact_scaled(a, b, margin, out)
    }

    /// The narrow phase for one pair with the relative transform `xf`: the
    /// rigid path of [`Scene::contact`].
    fn contact_rigid(
        &self,
        a: MeshId,
        b: MeshId,
        xf: &Rigid,
        margin: f64,
        out: &mut Vec<Contact>,
    ) -> Result<()> {
        let (ea, eb) = (self.entry(a)?, self.entry(b)?);
        let (ma, ba, mb, bb) = (&ea.mesh, &ea.bvh, &eb.mesh, &eb.bvh);
        let m32 = round_up(margin);
        let intersecting = fast_mesh_query::intersects(ma, ba, mb, bb, xf)?;
        let distance = if intersecting {
            0.0
        } else {
            match fast_mesh_query::mesh_distance(ma, ba, mb, bb, xf, m32)? {
                Some(d) if f64::from(d.distance) <= margin => f64::from(d.distance),
                _ => return Ok(()),
            }
        };
        let (mut triangles_a, mut triangles_b) = (Vec::new(), Vec::new());
        fast_mesh_query::triangles_near(
            ma,
            ba,
            mb,
            bb,
            xf,
            m32,
            &mut triangles_a,
            &mut triangles_b,
        )?;
        out.try_reserve(1)
            .map_err(|_| SceneError::Store(StoreError::OutOfMemory))?;
        out.push(Contact {
            a,
            b,
            intersecting,
            distance,
            triangles_a,
            triangles_b,
        });
        Ok(())
    }

    /// The narrow phase for one scaled pair: on a copy of B baked into A's
    /// local space (or of both into the world when A is non-uniformly
    /// scaled; see [`Scene::bake_pair_inner`]). Frame distances map to the
    /// world times the frame's factor.
    fn contact_scaled(
        &self,
        a: MeshId,
        b: MeshId,
        margin: f64,
        out: &mut Vec<Contact>,
    ) -> Result<()> {
        let (ta, tb, s) = self.bake_pair_inner(a, b)?;
        let ea = self.entry(a)?;
        let (ma, ba) = match &ta {
            None => (&ea.mesh, &ea.bvh),
            Some((m, bvh)) => (m, bvh),
        };
        let (mb, bb) = (&tb.0, &tb.1);
        let xf = Rigid::IDENTITY;
        let m32 = round_up(margin / s);
        let intersecting = fast_mesh_query::intersects(ma, ba, mb, bb, &xf)?;
        let distance = if intersecting {
            0.0
        } else {
            match fast_mesh_query::mesh_distance(ma, ba, mb, bb, &xf, m32)? {
                Some(d) if s * f64::from(d.distance) <= margin => s * f64::from(d.distance),
                _ => return Ok(()),
            }
        };
        let (mut triangles_a, mut triangles_b) = (Vec::new(), Vec::new());
        fast_mesh_query::triangles_near(
            ma,
            ba,
            mb,
            bb,
            &xf,
            m32,
            &mut triangles_a,
            &mut triangles_b,
        )?;
        out.try_reserve(1)
            .map_err(|_| SceneError::Store(StoreError::OutOfMemory))?;
        out.push(Contact {
            a,
            b,
            intersecting,
            distance,
            triangles_a,
            triangles_b,
        });
        Ok(())
    }
}

/// Canonical collision response mode applied between a pair of meshes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CollisionMode {
    /// Moves freely through other meshes while still reporting contacts and clearance.
    #[default]
    PassThrough,
    /// Prevents interpenetration along the sweep and optionally slides tangentially.
    Rigid,
    /// Subtracts the cutter volume from the target mesh using world-space CSG booleans.
    Cut,
    /// Evaluates signed penetration and deforms compliant mesh vertices smoothly.
    SoftDeform,
}

/// Signed-distance query backend used by [`CollisionMode::SoftDeform`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CollisionDistanceBackend {
    /// Exact BVH closest-point queries combined with pseudonormal / winding sign classification.
    #[default]
    BvhSdf,
    /// Felzenszwalb-Huttenlocher exact 3D Euclidean distance transform over a voxel grid.
    FhEdt,
}

/// When [`CollisionMode::Cut`] executes boolean subtraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CutTrigger {
    /// Defers CSG subtraction until `is_release == true` or `commit_collision_interaction`.
    #[default]
    OnRelease,
    /// Executes CSG subtraction on every intersecting step immediately.
    LiveStep,
}

/// Local deformation solver used by [`CollisionMode::SoftDeform`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SoftDeformSolver {
    /// Surface projection combined with smooth neighborhood Laplacian fairing.
    #[default]
    SdfLaplacian,
    /// Regularized elastostatic Kelvinlet radial push with surface non-penetration clamping.
    Kelvinlet,
    /// Extended Position Based Dynamics over edge-length, bending, and contact constraints.
    Xpbd,
}

/// Whether deformed vertices remember their original rest shape across steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MaterialMemory {
    /// Stores lazy rest positions and recovers smoothly once the indenter separates.
    #[default]
    Elastic,
    /// Retains indentations permanently in an undoable edit step.
    Plastic,
}

/// Directional role assignment for asymmetric collision responses (`Cut` and `SoftDeform`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CollisionRole {
    /// Treats the pair symmetrically (`id` cuts stationary in `Cut`; mutual compliance in `SoftDeform`).
    #[default]
    Symmetric,
    /// The moving mesh acts as the rigid cutter/indenter against stationary targets.
    MovingIsCutter,
    /// The moving mesh is the target/compliant body carved or indented by stationary partners.
    MovingIsTarget,
}

/// Returns the canonical unordered pair `(min, max)` for two mesh ids.
pub fn ordered_pair(a: MeshId, b: MeshId) -> (MeshId, MeshId) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Configuration for collision detection and response between two meshes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollisionRule {
    /// Collision response mode.
    pub mode: CollisionMode,
    /// Cutter/indenter vs target/compliant assignment.
    pub role: CollisionRole,
    /// Distance evaluation backend for soft deformation.
    pub distance_backend: CollisionDistanceBackend,
    /// CSG boolean evaluation strategy for [`CollisionMode::Cut`].
    pub cut_method: BooleanMethod,
    /// Whether cuts run continuously (`LiveStep`) or only on release (`OnRelease`).
    pub cut_trigger: CutTrigger,
    /// Solver strategy for [`CollisionMode::SoftDeform`].
    pub soft_solver: SoftDeformSolver,
    /// Material recovery behavior (`Elastic` vs `Plastic`).
    pub material_memory: MaterialMemory,
    /// Compliance parameter controlling solver softness (`>= 0.0`).
    pub compliance: f32,
    /// Stiffness weight controlling resistance and Laplacian damping (`>= 0.0`).
    pub stiffness: f32,
    /// Per-step elastic recovery fraction in `[0.0, 1.0]` when cleared of the indenter.
    pub recovery_rate: f32,
    /// World contact margin outside the surface treated as active contact (`>= 0.0`).
    pub contact_margin: f64,
    /// Enables self-intersection detection and separation damping during soft deformation.
    pub self_collision_guard: bool,
    /// Enables tangential sliding along contact planes after blocking in [`CollisionMode::Rigid`].
    pub rigid_slide: bool,
}

impl Default for CollisionRule {
    fn default() -> Self {
        Self {
            mode: CollisionMode::PassThrough,
            role: CollisionRole::Symmetric,
            distance_backend: CollisionDistanceBackend::BvhSdf,
            cut_method: BooleanMethod::Manifold,
            cut_trigger: CutTrigger::OnRelease,
            soft_solver: SoftDeformSolver::SdfLaplacian,
            material_memory: MaterialMemory::Elastic,
            compliance: 0.05,
            stiffness: 0.5,
            recovery_rate: 1.0,
            contact_margin: 0.0,
            self_collision_guard: true,
            rigid_slide: true,
        }
    }
}

impl CollisionRule {
    pub(crate) fn validate(&self) -> Result<()> {
        if !self.compliance.is_finite() || self.compliance < 0.0 {
            return Err(SceneError::InvalidInput(
                "collision rule compliance must be finite and non-negative",
            ));
        }
        if !self.stiffness.is_finite() || self.stiffness < 0.0 {
            return Err(SceneError::InvalidInput(
                "collision rule stiffness must be finite and non-negative",
            ));
        }
        if !self.recovery_rate.is_finite() || self.recovery_rate < 0.0 || self.recovery_rate > 1.0 {
            return Err(SceneError::InvalidInput(
                "collision rule recovery_rate must be in [0.0, 1.0]",
            ));
        }
        if !self.contact_margin.is_finite() || self.contact_margin < 0.0 {
            return Err(SceneError::InvalidInput(
                "collision rule contact_margin must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

/// Per-partner interaction outcome recorded during a collision response step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollisionPairReport {
    /// The stationary partner mesh evaluated against the moved mesh.
    pub other: MeshId,
    /// The effective collision mode resolved for this pair.
    pub mode: CollisionMode,
    /// The role configured for this pair.
    pub role: CollisionRole,
    /// Whether the pair intersects or touches at the end of the step.
    pub intersecting: bool,
    /// Minimum world surface distance between the pair at the end of the step (`0.0` if intersecting).
    pub min_distance: f64,
    /// Whether a rigid sweep stopped short of the requested target placement against this partner.
    pub contact_blocked: bool,
    /// Whether tangential sliding moved the object along this partner's contact surface.
    pub slidden: bool,
    /// Whether a boolean cut executed for this pair during the step.
    pub cut_executed: bool,
    /// Whether this pair is an active non-release preview awaiting release/commit.
    pub preview_only: bool,
    /// Number of vertices displaced outward by soft deformation for this pair.
    pub soft_deformed_vertices: usize,
    /// Number of vertices recovered toward rest positions for this pair.
    pub elastic_recovered_vertices: usize,
    /// Whether the self-collision guard prevented or separated a self-intersection on this pair.
    pub self_collision_guard_triggered: bool,
}

/// Summary of one [`Scene::move_with_collision`] or [`Scene::commit_collision_interaction`] step.
#[derive(Debug, Clone, PartialEq)]
pub struct CollisionStepReport {
    /// Final applied placement of the moved mesh.
    pub final_placement: Placement,
    /// Reports for active, intersecting, within-margin, or responding partner pairs.
    pub active_pairs: Vec<CollisionPairReport>,
    /// Minimum world clearance between the moved mesh and all other live meshes (`f64::INFINITY` if none).
    pub min_clearance: f64,
    /// Total number of boolean cuts executed during this step.
    pub cuts_executed: usize,
    /// Total count of compliant vertex deformations across all pairs in this step.
    pub soft_vertices_deformed: usize,
    /// Total count of elastic recovery vertex updates across all pairs in this step.
    pub elastic_recovery_count: usize,
    /// Whether the self-intersection guard triggered for any deformed mesh in this step.
    pub self_intersection_guard_triggered: bool,
}

impl Default for CollisionStepReport {
    fn default() -> Self {
        Self {
            final_placement: Placement::IDENTITY,
            active_pairs: Vec::new(),
            min_clearance: f64::INFINITY,
            cuts_executed: 0,
            soft_vertices_deformed: 0,
            elastic_recovery_count: 0,
            self_intersection_guard_triggered: false,
        }
    }
}

/// Cached rest positions for a mesh under [`MaterialMemory::Elastic`].
#[derive(Debug, Clone)]
pub(crate) struct ElasticRestState {
    pub(crate) rest_positions: Vec<[f32; 3]>,
    pub(crate) expected_revision: u64,
}

/// Scene-owned registry of default/per-pair collision rules and elastic rest shapes.
#[derive(Debug, Clone, Default)]
pub(crate) struct CollisionState {
    pub(crate) default_rule: CollisionRule,
    pub(crate) pair_rules: BTreeMap<(MeshId, MeshId), CollisionRule>,
    pub(crate) rest_positions: BTreeMap<MeshId, ElasticRestState>,
}

impl CollisionState {
    pub(crate) fn rule(&self, a: MeshId, b: MeshId) -> CollisionRule {
        self.pair_rules
            .get(&ordered_pair(a, b))
            .copied()
            .unwrap_or(self.default_rule)
    }

    pub(crate) fn drop_mesh(&mut self, id: MeshId) {
        self.pair_rules.retain(|&(a, b), _| a != id && b != id);
        self.rest_positions.remove(&id);
    }

    /// Takes the pair rules of `id` out, with the other mesh of each.
    pub(crate) fn take_rules_of(&mut self, id: MeshId) -> Vec<(MeshId, CollisionRule)> {
        let mut taken = Vec::new();
        self.pair_rules.retain(|&(a, b), rule| {
            let keep = a != id && b != id;
            if !keep {
                taken.push((if a == id { b } else { a }, *rule));
            }
            keep
        });
        taken
    }

    pub(crate) fn drop_rest_state(&mut self, id: MeshId) {
        self.rest_positions.remove(&id);
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct RigidPairOutcome {
    blocked: bool,
    slidden: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct SoftDeformOutcome {
    deformed_vertices: usize,
    recovered_vertices: usize,
    guard_triggered: bool,
}

#[derive(Debug, Clone, Copy)]
struct VertexContact {
    penetrating: bool,
    signed_dist_world: f64,
    outward_world: [f64; 3],
    target_world: [f64; 3],
}

impl Default for VertexContact {
    fn default() -> Self {
        Self {
            penetrating: false,
            signed_dist_world: 1.0,
            outward_world: [0.0, 1.0, 0.0],
            target_world: [0.0; 3],
        }
    }
}

struct SoftDeformMeshData {
    topology: MeshTopology,
    rest_world: Option<Vec<[f64; 3]>>,
}

struct MeshTopology {
    alive_ids: Vec<VertexId>,
    pos_0: Vec<[f64; 3]>,
    neighbors: Vec<Vec<(usize, f64)>>,
    edges: Vec<(usize, usize)>,
    bends: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Copy)]
enum SdfEvaluator<'a> {
    Bvh {
        mesh: &'a Mesh,
        bvh: &'a Bvh,
        sign: &'a SignData,
        placement: &'a Placement,
        origin: [f64; 3],
        world_box: Aabb64,
    },
    FhEdt {
        grid: &'a fast_mesh_fields::ScalarGrid,
        placement: &'a Placement,
        origin: [f64; 3],
        center_local: [f64; 3],
        /// The placement's, which is uniform.
        scale: f64,
    },
}

fn vec_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec_scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn vec_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec_norm2(a: [f64; 3]) -> f64 {
    vec_dot(a, a)
}

fn vec_norm(a: [f64; 3]) -> f64 {
    vec_norm2(a).sqrt()
}

fn vec_normalize(a: [f64; 3], fallback: [f64; 3]) -> [f64; 3] {
    let n = vec_norm(a);
    if n > 1e-12 && n.is_finite() {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        fallback
    }
}

fn placement_world_box(entry: &Entry, placement: &Placement) -> Aabb64 {
    match entry.bvh.root_bounds().filter(|b| !b.is_empty()) {
        Some(b) => box_to_world(&b, placement, entry.mesh.origin()),
        None => Aabb64::EMPTY,
    }
}

/// The world box of a box in a mesh's stored local coordinates.
fn box_to_world(b: &Aabb, placement: &Placement, origin: [f64; 3]) -> Aabb64 {
    let mut out = Aabb64::EMPTY;
    for corner in 0..8 {
        let p = std::array::from_fn(|k| {
            let end = if corner >> k & 1 == 1 { b.max } else { b.min };
            f64::from(end[k])
        });
        let w = local_to_world(placement, origin, p);
        for (k, &c) in w.iter().enumerate() {
            out.min[k] = out.min[k].min(c);
            out.max[k] = out.max[k].max(c);
        }
    }
    out
}

fn pad_box(b: &Aabb64, pad: f64) -> Aabb64 {
    if b.is_empty() {
        return *b;
    }
    Aabb64 {
        min: [b.min[0] - pad, b.min[1] - pad, b.min[2] - pad],
        max: [b.max[0] + pad, b.max[1] + pad, b.max[2] + pad],
    }
}

fn pair_intersects_with_placements(
    ea: &Entry,
    pa: &Placement,
    eb: &Entry,
    pb: &Placement,
) -> Result<bool> {
    if ea.mesh.triangle_count() == 0 || eb.mesh.triangle_count() == 0 {
        return Ok(false);
    }
    if is_relative_rigid(pa, pb) {
        let xf = relative_rigid(pa, ea.mesh.origin(), pb, eb.mesh.origin());
        return Ok(fast_mesh_query::intersects(
            &ea.mesh, &ea.bvh, &eb.mesh, &eb.bvh, &xf,
        )?);
    }
    let wa = placement_world_box(ea, pa);
    let wb = placement_world_box(eb, pb);
    if wa.is_empty() || wb.is_empty() {
        return Ok(false);
    }
    let pad = 1e-3 * (1.0 + vec_norm(vec_sub(wa.max, wa.min)) + vec_norm(vec_sub(wb.max, wb.min)));
    if wa.distance_to_box(&wb) > pad {
        return Ok(false);
    }
    if let Some(sa) = pa.uniform_scale() {
        let _ = sa;
        let affine = relative_affine(pa, ea.mesh.origin(), pb, eb.mesh.origin());
        let (tb, bb) = baked_mesh(
            &eb.mesh,
            |l| affine.apply(l.map(f64::from)),
            eb.bvh.options(),
        )?;
        return Ok(fast_mesh_query::intersects(
            &ea.mesh,
            &ea.bvh,
            &tb,
            &bb,
            &Rigid::IDENTITY,
        )?);
    }
    let base = world_base(&ea.mesh, pa);
    let (ta, ba) = baked_pose(ea, pa, base)?;
    let (tb, bb) = baked_pose(eb, pb, base)?;
    Ok(fast_mesh_query::intersects(
        &ta,
        &ba,
        &tb,
        &bb,
        &Rigid::IDENTITY,
    )?)
}

fn pair_intersects_symmetric(
    ea: &Entry,
    pa: &Placement,
    eb: &Entry,
    pb: &Placement,
) -> Result<bool> {
    Ok(pair_intersects_with_placements(ea, pa, eb, pb)?
        || pair_intersects_with_placements(eb, pb, ea, pa)?)
}

struct PairSurfaceDistance {
    distance: f64,
    point_a: [f64; 3],
    point_b: [f64; 3],
    triangle_b: TriangleId,
}

fn pair_distance_with_placements(
    ea: &Entry,
    pa: &Placement,
    eb: &Entry,
    pb: &Placement,
) -> Result<Option<PairSurfaceDistance>> {
    let wa = placement_world_box(ea, pa);
    let wb = placement_world_box(eb, pb);
    if wa.is_empty() || wb.is_empty() {
        return Ok(None);
    }
    if is_relative_rigid(pa, pb) {
        let xf = relative_rigid(pa, ea.mesh.origin(), pb, eb.mesh.origin());
        let found = fast_mesh_query::mesh_distance(
            &ea.mesh,
            &ea.bvh,
            &eb.mesh,
            &eb.bvh,
            &xf,
            f32::INFINITY,
        )?;
        return Ok(found.map(|d| PairSurfaceDistance {
            distance: f64::from(d.distance),
            point_a: local_to_world(pa, ea.mesh.origin(), d.point_a.map(f64::from)),
            point_b: local_to_world(pa, ea.mesh.origin(), d.point_b.map(f64::from)),
            triangle_b: d.triangle_b,
        }));
    }
    if let Some(sa) = pa.uniform_scale() {
        let affine = relative_affine(pa, ea.mesh.origin(), pb, eb.mesh.origin());
        let (tb, bb) = baked_mesh(
            &eb.mesh,
            |l| affine.apply(l.map(f64::from)),
            eb.bvh.options(),
        )?;
        let found = fast_mesh_query::mesh_distance(
            &ea.mesh,
            &ea.bvh,
            &tb,
            &bb,
            &Rigid::IDENTITY,
            f32::INFINITY,
        )?;
        return Ok(found.map(|d| PairSurfaceDistance {
            distance: sa * f64::from(d.distance),
            point_a: local_to_world(pa, ea.mesh.origin(), d.point_a.map(f64::from)),
            point_b: local_to_world(pa, ea.mesh.origin(), d.point_b.map(f64::from)),
            triangle_b: d.triangle_b,
        }));
    }
    let base = world_base(&ea.mesh, pa);
    let (ta, ba) = baked_pose(ea, pa, base)?;
    let (tb, bb) = baked_pose(eb, pb, base)?;
    let found =
        fast_mesh_query::mesh_distance(&ta, &ba, &tb, &bb, &Rigid::IDENTITY, f32::INFINITY)?;
    Ok(found.map(|d| PairSurfaceDistance {
        distance: f64::from(d.distance),
        point_a: vec_add(base, d.point_a.map(f64::from)),
        point_b: vec_add(base, d.point_b.map(f64::from)),
        triangle_b: d.triangle_b,
    }))
}

fn world_triangle_normal(entry: &Entry, placement: &Placement, tri: TriangleId) -> [f64; 3] {
    let Some([v0, v1, v2]) = entry.mesh.triangle(tri) else {
        return [0.0, 1.0, 0.0];
    };
    let pos = entry.mesh.positions();
    let p0 = pos[v0 as usize].map(f64::from);
    let p1 = pos[v1 as usize].map(f64::from);
    let p2 = pos[v2 as usize].map(f64::from);
    let n_local = vec_cross(vec_sub(p1, p0), vec_sub(p2, p0));
    if vec_norm(n_local) <= 1e-12 {
        return [0.0, 1.0, 0.0];
    }
    placement.normal_to_world(vec_normalize(n_local, [0.0, 1.0, 0.0]))
}

impl Scene {
    /// Replaces the default collision rule used for pairs without an explicit override.
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] if any numeric parameter in `rule` is invalid or out of range.
    pub fn set_default_collision_rule(&mut self, rule: CollisionRule) -> Result<()> {
        self.drop_field_unit();
        rule.validate()?;
        self.collision.default_rule = rule;
        Ok(())
    }

    /// Returns the scene's default collision rule.
    pub fn default_collision_rule(&self) -> &CollisionRule {
        &self.collision.default_rule
    }

    /// Assigns an explicit collision rule to the canonical unordered pair [`ordered_pair(a, b)`].
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] if either id is stale, or [`SceneError::InvalidInput`] if `a == b`
    /// or `rule` fails validation.
    pub fn set_collision_rule(&mut self, a: MeshId, b: MeshId, rule: CollisionRule) -> Result<()> {
        self.drop_field_unit();
        self.entry(a)?;
        self.entry(b)?;
        if a == b {
            return Err(SceneError::InvalidInput(
                "collision rule requires two distinct meshes",
            ));
        }
        rule.validate()?;
        self.collision.pair_rules.insert(ordered_pair(a, b), rule);
        Ok(())
    }

    /// Removes any per-pair collision rule between `a` and `b`, reverting to the default rule.
    /// Returns `true` if an explicit rule had been stored.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] if either id is stale, or [`SceneError::InvalidInput`] if `a == b`.
    pub fn clear_collision_rule(&mut self, a: MeshId, b: MeshId) -> Result<bool> {
        self.drop_field_unit();
        self.entry(a)?;
        self.entry(b)?;
        if a == b {
            return Err(SceneError::InvalidInput(
                "collision rule requires two distinct meshes",
            ));
        }
        Ok(self
            .collision
            .pair_rules
            .remove(&ordered_pair(a, b))
            .is_some())
    }

    /// Resolves the effective collision rule between `a` and `b` (per-pair override or scene default).
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] if either id is stale, or [`SceneError::InvalidInput`] if `a == b`.
    pub fn collision_rule(&self, a: MeshId, b: MeshId) -> Result<CollisionRule> {
        self.entry(a)?;
        self.entry(b)?;
        if a == b {
            return Err(SceneError::InvalidInput(
                "collision rule requires two distinct meshes",
            ));
        }
        Ok(self.collision.rule(a, b))
    }

    /// Returns `true` if `moving` is a live mesh and either the scene's default collision rule
    /// or any explicit pair rule involving `moving` and another live mesh has a mode other than
    /// [`CollisionMode::PassThrough`].
    pub fn has_active_collision_response(&self, moving: MeshId) -> bool {
        if self.entry(moving).is_err() {
            return false;
        }
        if self.collision.default_rule.mode != CollisionMode::PassThrough {
            return true;
        }
        self.collision.pair_rules.iter().any(|(&(a, b), rule)| {
            rule.mode != CollisionMode::PassThrough
                && ((a == moving && self.entry(b).is_ok())
                    || (b == moving && self.entry(a).is_ok()))
        })
    }

    /// Moves `id` toward `target_placement` and applies configured collision responses against all
    /// other live meshes in one undoable step.
    ///
    /// Set `is_release` to `true` on pointer/drag release so [`CutTrigger::OnRelease`] cuts execute.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`], [`SceneError::InvalidInput`], [`SceneError::StaleBvh`], or any
    /// underlying store/query/field/boolean error. On failure the scene stays unchanged.
    pub fn move_with_collision(
        &mut self,
        id: MeshId,
        target_placement: Placement,
        is_release: bool,
    ) -> Result<CollisionStepReport> {
        self.drop_field_unit();
        target_placement.validate()?;
        self.entry(id)?;
        self.prepare(None)?;

        let was_open = self.history.is_open();
        self.begin_step();
        let res = self.move_with_collision_inner(id, target_placement, is_release);
        match res {
            Ok(report) => {
                self.end_step()?;
                Ok(report)
            }
            Err(err) => {
                if !was_open {
                    let _ = self.cancel_step();
                } else {
                    let _ = self.end_step();
                }
                Err(err)
            }
        }
    }

    /// Commits any pending release-triggered collision response (such as [`CutTrigger::OnRelease`])
    /// for mesh `id` at its current placement.
    ///
    /// # Errors
    /// Same as [`Scene::move_with_collision`].
    pub fn commit_collision_interaction(&mut self, id: MeshId) -> Result<CollisionStepReport> {
        self.drop_field_unit();
        let current = self.entry(id)?.placement;
        self.move_with_collision(id, current, true)
    }

    fn move_with_collision_inner(
        &mut self,
        id: MeshId,
        target_placement: Placement,
        is_release: bool,
    ) -> Result<CollisionStepReport> {
        let start_placement = self.entry(id)?.placement;
        let others: Vec<MeshId> = self.ids().filter(|&other| other != id).collect();
        let rules: Vec<(MeshId, CollisionRule)> = others
            .iter()
            .map(|&other| (other, self.collision.rule(id, other)))
            .collect();

        let rigid_partners: Vec<(MeshId, CollisionRule)> = rules
            .iter()
            .copied()
            .filter(|&(_, rule)| rule.mode == CollisionMode::Rigid)
            .collect();

        let (final_placement, rigid_outcomes) = if rigid_partners.is_empty() {
            (target_placement, BTreeMap::new())
        } else {
            self.solve_rigid_sweep(id, start_placement, target_placement, &rigid_partners)?
        };

        self.set_placement(id, final_placement)?;

        let mut report = CollisionStepReport {
            final_placement,
            active_pairs: Vec::new(),
            min_clearance: f64::INFINITY,
            cuts_executed: 0,
            soft_vertices_deformed: 0,
            elastic_recovery_count: 0,
            self_intersection_guard_triggered: false,
        };

        for (other, rule) in rules {
            match rule.mode {
                CollisionMode::PassThrough => {
                    let (intersecting, dist) = self.measure_pair_clearance(id, other)?;
                    if dist < report.min_clearance {
                        report.min_clearance = dist;
                    }
                    if intersecting || dist <= rule.contact_margin {
                        report.active_pairs.push(CollisionPairReport {
                            other,
                            mode: rule.mode,
                            role: rule.role,
                            intersecting,
                            min_distance: dist,
                            contact_blocked: false,
                            slidden: false,
                            cut_executed: false,
                            preview_only: false,
                            soft_deformed_vertices: 0,
                            elastic_recovered_vertices: 0,
                            self_collision_guard_triggered: false,
                        });
                    }
                }
                CollisionMode::Rigid => {
                    let outcome = rigid_outcomes.get(&other).copied().unwrap_or_default();
                    let (intersecting, dist) = self.measure_pair_clearance(id, other)?;
                    if dist < report.min_clearance {
                        report.min_clearance = dist;
                    }
                    if intersecting
                        || dist <= rule.contact_margin
                        || outcome.blocked
                        || outcome.slidden
                    {
                        report.active_pairs.push(CollisionPairReport {
                            other,
                            mode: rule.mode,
                            role: rule.role,
                            intersecting,
                            min_distance: dist,
                            contact_blocked: outcome.blocked,
                            slidden: outcome.slidden,
                            cut_executed: false,
                            preview_only: false,
                            soft_deformed_vertices: 0,
                            elastic_recovered_vertices: 0,
                            self_collision_guard_triggered: false,
                        });
                    }
                }
                CollisionMode::Cut => {
                    let (target, cutter) = match rule.role {
                        CollisionRole::Symmetric | CollisionRole::MovingIsCutter => (other, id),
                        CollisionRole::MovingIsTarget => (id, other),
                    };
                    let pre_overlap = self.intersects(target, cutter)?
                        || self.meshes_volumetrically_overlap(target, cutter)?;
                    let should_cut =
                        pre_overlap && (rule.cut_trigger == CutTrigger::LiveStep || is_release);
                    let mut cut_executed = false;
                    let preview_only =
                        pre_overlap && rule.cut_trigger == CutTrigger::OnRelease && !is_release;
                    if should_cut {
                        let opts = BooleanOptions {
                            op: crate::ops::BooleanOp::Subtract,
                            method: rule.cut_method,
                            ..BooleanOptions::default()
                        };
                        self.boolean_with_options(target, cutter, &opts)?;
                        self.collision.drop_rest_state(target);
                        cut_executed = true;
                        report.cuts_executed += 1;
                    }
                    let (tri_intersecting, dist) = self.measure_pair_clearance(id, other)?;
                    let intersecting = tri_intersecting || (preview_only && pre_overlap);
                    if dist < report.min_clearance {
                        report.min_clearance = dist;
                    }
                    if cut_executed || intersecting || preview_only || dist <= rule.contact_margin {
                        report.active_pairs.push(CollisionPairReport {
                            other,
                            mode: rule.mode,
                            role: rule.role,
                            intersecting,
                            min_distance: dist,
                            contact_blocked: false,
                            slidden: false,
                            cut_executed,
                            preview_only,
                            soft_deformed_vertices: 0,
                            elastic_recovered_vertices: 0,
                            self_collision_guard_triggered: false,
                        });
                    }
                }
                CollisionMode::SoftDeform => {
                    let soft = self.apply_soft_deform_pair(id, other, &rule)?;
                    report.soft_vertices_deformed += soft.deformed_vertices;
                    report.elastic_recovery_count += soft.recovered_vertices;
                    if soft.guard_triggered {
                        report.self_intersection_guard_triggered = true;
                    }
                    let (intersecting, dist) = self.measure_pair_clearance(id, other)?;
                    if dist < report.min_clearance {
                        report.min_clearance = dist;
                    }
                    if intersecting
                        || dist <= rule.contact_margin
                        || soft.deformed_vertices > 0
                        || soft.recovered_vertices > 0
                        || soft.guard_triggered
                    {
                        report.active_pairs.push(CollisionPairReport {
                            other,
                            mode: rule.mode,
                            role: rule.role,
                            intersecting,
                            min_distance: dist,
                            contact_blocked: false,
                            slidden: false,
                            cut_executed: false,
                            preview_only: false,
                            soft_deformed_vertices: soft.deformed_vertices,
                            elastic_recovered_vertices: soft.recovered_vertices,
                            self_collision_guard_triggered: soft.guard_triggered,
                        });
                    }
                }
            }
        }

        Ok(report)
    }

    fn measure_pair_clearance(&mut self, a: MeshId, b: MeshId) -> Result<(bool, f64)> {
        if self.intersects(a, b)? {
            return Ok((true, 0.0));
        }
        let dist = match self.mesh_distance(a, b, f64::INFINITY)? {
            Some(d) => d.distance,
            None => f64::INFINITY,
        };
        Ok((false, dist))
    }

    fn meshes_volumetrically_overlap(&mut self, a: MeshId, b: MeshId) -> Result<bool> {
        let (wa, wb) = {
            let ea = self.entry(a)?;
            let eb = self.entry(b)?;
            (
                placement_world_box(ea, &ea.placement),
                placement_world_box(eb, &eb.placement),
            )
        };
        if wa.is_empty() || wb.is_empty() || wa.distance_to_box(&wb) > 0.0 {
            return Ok(false);
        }
        Ok(self.first_vertex_inside(a, b)? || self.first_vertex_inside(b, a)?)
    }

    fn first_vertex_inside(&mut self, outer: MeshId, inner: MeshId) -> Result<bool> {
        let sample_world = {
            let ei = self.entry(inner)?;
            let mut found = None;
            for v in 0..ei.mesh.vertex_slots() as u32 {
                if ei.mesh.is_vertex_alive(v) {
                    let p = ei.mesh.positions()[v as usize].map(f64::from);
                    found = Some(local_to_world(&ei.placement, ei.mesh.origin(), p));
                    break;
                }
            }
            let Some(w) = found else {
                return Ok(false);
            };
            w
        };
        let (pl, origin, has_tris) = {
            let eo = self.entry(outer)?;
            (eo.placement, eo.mesh.origin(), eo.mesh.triangle_count() > 0)
        };
        if !has_tris {
            return Ok(false);
        }
        let local64 = world_to_local(&pl, origin, sample_world);
        let sample_local = [local64[0] as f32, local64[1] as f32, local64[2] as f32];
        if !sample_local.iter().all(|c| c.is_finite()) {
            return Ok(false);
        }
        let eo = self.entry_mut(outer)?;
        let sign = SignData::build(&mut eo.mesh, &eo.bvh, SignMode::Auto)?;
        let mut out = [0.0f32; 1];
        signed_distances(
            &eo.mesh,
            &eo.bvh,
            &sign,
            &[sample_local],
            f32::INFINITY,
            &mut out,
        )?;
        Ok(out[0] < 0.0)
    }

    fn rigid_pose_intersects_any(
        &self,
        id: MeshId,
        pose: &Placement,
        partners: &[(MeshId, CollisionRule)],
    ) -> Result<Option<MeshId>> {
        let moved_entry = self.entry(id)?;
        for &(partner, _) in partners {
            let pe = self.entry(partner)?;
            if pair_intersects_symmetric(moved_entry, pose, pe, &pe.placement)? {
                return Ok(Some(partner));
            }
        }
        Ok(None)
    }

    fn rigid_pose_intersects_scene(
        &mut self,
        id: MeshId,
        pose: &Placement,
        partners: &[(MeshId, CollisionRule)],
    ) -> Result<bool> {
        let saved = self.entry(id)?.placement;
        self.entries
            .get_mut(id)
            .ok_or(SceneError::UnknownMesh)?
            .placement = *pose;
        let mut hit = false;
        let mut first_err: Option<SceneError> = None;
        for &(partner, _) in partners {
            match self.intersects(id, partner).and_then(|a| {
                if a {
                    Ok(true)
                } else {
                    self.intersects(partner, id)
                }
            }) {
                Ok(true) => {
                    hit = true;
                    break;
                }
                Ok(false) => {}
                Err(err) => {
                    first_err = Some(err);
                    break;
                }
            }
        }
        if let Some(entry) = self.entries.get_mut(id) {
            entry.placement = saved;
        }
        if let Some(err) = first_err {
            return Err(err);
        }
        Ok(hit)
    }

    fn min_rigid_partner_distance(
        &self,
        id: MeshId,
        pose: &Placement,
        partners: &[(MeshId, CollisionRule)],
    ) -> Result<f64> {
        let ea = self.entry(id)?;
        let mut min_dist = f64::INFINITY;
        for &(partner, _) in partners {
            let eb = self.entry(partner)?;
            if let Some(sd) = pair_distance_with_placements(ea, pose, eb, &eb.placement)?
                && sd.distance < min_dist
            {
                min_dist = sd.distance;
            }
        }
        Ok(min_dist)
    }

    fn ensure_rigid_clearance(
        &mut self,
        id: MeshId,
        pose: Placement,
        partners: &[(MeshId, CollisionRule)],
        fallback_normal: Option<[f64; 3]>,
        fallback_motion: [f64; 3],
        skin: f64,
    ) -> Result<Option<Placement>> {
        let is_clear = |scene: &mut Scene, cand: &Placement| -> Result<(bool, f64)> {
            if scene
                .rigid_pose_intersects_any(id, cand, partners)?
                .is_some()
                || scene.rigid_pose_intersects_scene(id, cand, partners)?
            {
                return Ok((false, 0.0));
            }
            let d = scene.min_rigid_partner_distance(id, cand, partners)?;
            Ok((true, d))
        };

        let (clear0, dist0) = is_clear(self, &pose)?;
        if clear0 && dist0 >= 0.25 * skin {
            return Ok(Some(pose));
        }

        let push_normal = self
            .rigid_contact_normal(id, &pose, partners, fallback_motion)?
            .map(|(_, n)| n)
            .or(fallback_normal);

        let mut best_clear: Option<Placement> = if clear0 { Some(pose) } else { None };
        if let Some(n) = push_normal {
            for &mult in &[1.0, 2.0, 4.0] {
                let cand = Placement {
                    translation: vec_add(pose.translation, vec_scale(n, mult * skin)),
                    ..pose
                };
                let (clear, dist) = is_clear(self, &cand)?;
                if clear {
                    if dist >= 0.25 * skin {
                        return Ok(Some(cand));
                    }
                    if best_clear.is_none() {
                        best_clear = Some(cand);
                    }
                }
            }
        }
        Ok(best_clear)
    }

    fn solve_rigid_sweep(
        &mut self,
        id: MeshId,
        start: Placement,
        target: Placement,
        partners: &[(MeshId, CollisionRule)],
    ) -> Result<(Placement, BTreeMap<MeshId, RigidPairOutcome>)> {
        let mut outcomes: BTreeMap<MeshId, RigidPairOutcome> = BTreeMap::new();
        let (moving_diag, min_extent) = {
            let moved_entry = self.entry(id)?;
            let moving_box = placement_world_box(moved_entry, &start);
            let moving_diag = if moving_box.is_empty() {
                1.0
            } else {
                vec_norm(vec_sub(moving_box.max, moving_box.min)).max(1e-3)
            };
            let mut min_extent = moving_diag;
            for &(partner, _) in partners {
                let pe = self.entry(partner)?;
                let pb = placement_world_box(pe, &pe.placement);
                if !pb.is_empty() {
                    let d = vec_norm(vec_sub(pb.max, pb.min)).max(1e-3);
                    if d < min_extent {
                        min_extent = d;
                    }
                }
            }
            (moving_diag, min_extent)
        };
        let disp = vec_norm(vec_sub(target.translation, start.translation));
        let steps = ((disp / (0.25 * min_extent)).ceil() as usize).clamp(1, 32);
        let skin = 1e-4 * moving_diag.min(min_extent).max(1.0);

        if let Some(hit_start) = self.rigid_pose_intersects_any(id, &start, partners)? {
            if self
                .rigid_pose_intersects_any(id, &target, partners)?
                .is_none()
                && !self.rigid_pose_intersects_scene(id, &target, partners)?
            {
                return Ok((target, outcomes));
            }
            outcomes.entry(hit_start).or_default().blocked = true;
            return Ok((start, outcomes));
        }
        if self.rigid_pose_intersects_scene(id, &start, partners)? {
            if self
                .rigid_pose_intersects_any(id, &target, partners)?
                .is_none()
                && !self.rigid_pose_intersects_scene(id, &target, partners)?
            {
                return Ok((target, outcomes));
            }
            if let Some(&(first_partner, _)) = partners.first() {
                outcomes.entry(first_partner).or_default().blocked = true;
            }
            return Ok((start, outcomes));
        }

        let mut bracket: Option<(f64, f64)> = None;
        let mut prev_t = 0.0;
        for step in 1..=steps {
            let t = if step == steps {
                1.0
            } else {
                step as f64 / steps as f64
            };
            let cand = interpolate(&start, &target, t);
            if self
                .rigid_pose_intersects_any(id, &cand, partners)?
                .is_some()
            {
                bracket = Some((prev_t, t));
                break;
            }
            prev_t = t;
        }

        if bracket.is_none() && self.rigid_pose_intersects_scene(id, &target, partners)? {
            bracket = Some((prev_t, 1.0));
        }

        let Some((mut lo, mut hi)) = bracket else {
            return Ok((target, outcomes));
        };

        for _ in 0..34 {
            let mid = 0.5 * (lo + hi);
            let cand = interpolate(&start, &target, mid);
            if self
                .rigid_pose_intersects_any(id, &cand, partners)?
                .is_some()
            {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        let contact_pose = interpolate(&start, &target, lo);
        let probe_t = hi.min((lo + 1e-4).min(1.0));
        let probe_pose = interpolate(&start, &target, probe_t);
        let mut primary_blocker: Option<(MeshId, CollisionRule)> = None;
        {
            let moved_entry = self.entry(id)?;
            for &(partner, rule) in partners {
                let pe = self.entry(partner)?;
                if pair_intersects_symmetric(moved_entry, &probe_pose, pe, &pe.placement)?
                    || pair_intersects_symmetric(moved_entry, &target, pe, &pe.placement)?
                {
                    outcomes.entry(partner).or_default().blocked = true;
                    if primary_blocker.is_none() {
                        primary_blocker = Some((partner, rule));
                    }
                }
            }
        }

        let sweep_vec = vec_sub(target.translation, start.translation);
        let contact_normal = self.rigid_contact_normal(id, &contact_pose, partners, sweep_vec)?;
        let back_t = if disp > 1e-12 {
            let sweep_dir = vec_scale(sweep_vec, 1.0 / disp);
            let cos_angle = contact_normal
                .map(|(_, n)| (-vec_dot(sweep_dir, n)).clamp(0.1, 1.0))
                .unwrap_or(1.0);
            ((skin / cos_angle) / disp).min(lo)
        } else {
            0.0
        };
        let backed = interpolate(&start, &target, (lo - back_t).max(0.0));
        let needs_normal_fix = self
            .rigid_pose_intersects_any(id, &backed, partners)?
            .is_some()
            || self.rigid_pose_intersects_scene(id, &backed, partners)?
            || ((lo > back_t || disp <= 1e-12)
                && self.min_rigid_partner_distance(id, &backed, partners)? < 0.25 * skin);
        let mut current = if needs_normal_fix {
            if let Some(safe) = self.ensure_rigid_clearance(
                id,
                backed,
                partners,
                contact_normal.map(|(_, n)| n),
                sweep_vec,
                skin,
            )? {
                safe
            } else if !self.rigid_pose_intersects_scene(id, &contact_pose, partners)? {
                contact_pose
            } else {
                start
            }
        } else {
            backed
        };
        if self.rigid_pose_intersects_scene(id, &current, partners)? {
            current = start;
        }
        let pre_slide_safe = current;

        if let Some((_, blocker_rule)) = primary_blocker
            && blocker_rule.rigid_slide
        {
            let mut rem_t = vec_sub(target.translation, current.translation);
            for _ in 0..3 {
                if vec_norm(rem_t) <= 1e-11 {
                    break;
                }
                let Some((blocker_id, normal)) =
                    self.rigid_contact_normal(id, &current, partners, rem_t)?
                else {
                    break;
                };
                if !partners
                    .iter()
                    .find(|&&(p, _)| p == blocker_id)
                    .is_some_and(|&(_, r)| r.rigid_slide)
                {
                    break;
                }
                let into = vec_dot(rem_t, normal);
                if into >= 0.0 {
                    break;
                }
                let tang = vec_sub(rem_t, vec_scale(normal, into));
                let tang_len = vec_norm(tang);
                if tang_len <= 1e-11 {
                    break;
                }
                let slide_base = current;
                let slide_goal = Placement {
                    translation: vec_add(slide_base.translation, tang),
                    ..slide_base
                };

                let base_too_close = self.min_rigid_partner_distance(id, &slide_base, partners)?
                    < 0.5 * skin
                    || self
                        .rigid_pose_intersects_any(
                            id,
                            &interpolate(&slide_base, &slide_goal, 1e-3),
                            partners,
                        )?
                        .is_some();
                let origin_pose = if base_too_close {
                    let lifted = Placement {
                        translation: vec_add(slide_base.translation, vec_scale(normal, skin)),
                        ..slide_base
                    };
                    if self
                        .rigid_pose_intersects_any(id, &lifted, partners)?
                        .is_none()
                        && !self.rigid_pose_intersects_scene(id, &lifted, partners)?
                    {
                        lifted
                    } else {
                        slide_base
                    }
                } else {
                    slide_base
                };

                let shifted_goal = Placement {
                    translation: vec_add(origin_pose.translation, tang),
                    ..origin_pose
                };
                let mut s_lo = 0.0;
                let mut s_hi = 1.0;
                let s_hit = if self
                    .rigid_pose_intersects_any(id, &shifted_goal, partners)?
                    .is_none()
                    && !self.rigid_pose_intersects_scene(id, &shifted_goal, partners)?
                {
                    s_lo = 1.0;
                    false
                } else {
                    for _ in 0..28 {
                        let mid = 0.5 * (s_lo + s_hi);
                        if self
                            .rigid_pose_intersects_any(
                                id,
                                &interpolate(&origin_pose, &shifted_goal, mid),
                                partners,
                            )?
                            .is_some()
                        {
                            s_hi = mid;
                        } else {
                            s_lo = mid;
                        }
                    }
                    let back_s = (skin / tang_len).min(0.5 * s_lo);
                    s_lo = (s_lo - back_s).max(0.0);
                    true
                };

                let cand = interpolate(&origin_pose, &shifted_goal, s_lo);
                let Some(slid_pose) = self.ensure_rigid_clearance(
                    id,
                    cand,
                    partners,
                    Some(normal),
                    vec_scale(tang, -1.0),
                    skin,
                )?
                else {
                    break;
                };

                if vec_norm(vec_sub(slid_pose.translation, current.translation)) > 1e-10 {
                    let o = outcomes.entry(blocker_id).or_default();
                    o.blocked = true;
                    o.slidden = true;
                    current = slid_pose;
                } else {
                    break;
                }
                rem_t = vec_scale(tang, 1.0 - s_lo);
                if !s_hit || s_lo >= 1.0 - 1e-9 || s_lo <= 1e-6 {
                    break;
                }
            }
        }

        if self.rigid_pose_intersects_scene(id, &current, partners)? {
            current = if !self.rigid_pose_intersects_scene(id, &pre_slide_safe, partners)? {
                for o in outcomes.values_mut() {
                    o.slidden = false;
                }
                pre_slide_safe
            } else {
                for o in outcomes.values_mut() {
                    o.slidden = false;
                }
                start
            };
        }

        Ok((current, outcomes))
    }

    fn rigid_contact_normal(
        &self,
        id: MeshId,
        pose: &Placement,
        partners: &[(MeshId, CollisionRule)],
        motion: [f64; 3],
    ) -> Result<Option<(MeshId, [f64; 3])>> {
        let ea = self.entry(id)?;
        let mut best: Option<(f64, MeshId, [f64; 3])> = None;
        for &(partner, _) in partners {
            let eb = self.entry(partner)?;
            let Some(sd) = pair_distance_with_placements(ea, pose, eb, &eb.placement)? else {
                continue;
            };
            let better = match best {
                None => true,
                Some((bd, _, _)) => sd.distance < bd,
            };
            if better {
                let mut n = if sd.distance > 1e-10 {
                    let mut n_sep = vec_normalize(vec_sub(sd.point_a, sd.point_b), [0.0, 1.0, 0.0]);
                    if vec_dot(n_sep, motion) > 0.0 {
                        n_sep = vec_scale(n_sep, -1.0);
                    }
                    // When separation is small (`f32` subtraction `point_a - point_b` can suffer
                    // cancellation), prefer the exact `f64` obstacle face normal when aligned.
                    if sd.distance < 1e-2 {
                        let mut tn = world_triangle_normal(eb, &eb.placement, sd.triangle_b);
                        if vec_dot(tn, n_sep) < 0.0 {
                            tn = vec_scale(tn, -1.0);
                        }
                        if vec_dot(tn, n_sep) > 0.995 {
                            tn
                        } else {
                            n_sep
                        }
                    } else {
                        n_sep
                    }
                } else {
                    world_triangle_normal(eb, &eb.placement, sd.triangle_b)
                };
                if vec_dot(n, motion) > 0.0 {
                    n = vec_scale(n, -1.0);
                }
                best = Some((sd.distance, partner, n));
            }
        }
        Ok(best.map(|(_, pid, n)| (pid, n)))
    }

    fn apply_soft_deform_pair(
        &mut self,
        id: MeshId,
        other: MeshId,
        rule: &CollisionRule,
    ) -> Result<SoftDeformOutcome> {
        match rule.role {
            CollisionRole::MovingIsCutter => {
                self.deform_compliant_against_indenter(other, id, rule, 1.0)
            }
            CollisionRole::MovingIsTarget => {
                self.deform_compliant_against_indenter(id, other, rule, 1.0)
            }
            CollisionRole::Symmetric => {
                let c_other = self.count_penetrating_vertices(other, id, rule)?;
                let c_id = self.count_penetrating_vertices(id, other, rule)?;
                let (frac_id, frac_other) = if c_id > 0 && c_other > 0 {
                    (0.5, 0.5)
                } else {
                    (1.0, 1.0)
                };
                let o1 = self.deform_compliant_against_indenter(other, id, rule, frac_other)?;
                let o2 = self.deform_compliant_against_indenter(id, other, rule, frac_id)?;
                Ok(SoftDeformOutcome {
                    deformed_vertices: o1.deformed_vertices + o2.deformed_vertices,
                    recovered_vertices: o1.recovered_vertices + o2.recovered_vertices,
                    guard_triggered: o1.guard_triggered || o2.guard_triggered,
                })
            }
        }
    }

    fn count_penetrating_vertices(
        &mut self,
        compliant: MeshId,
        indenter: MeshId,
        rule: &CollisionRule,
    ) -> Result<usize> {
        let contacts = self.evaluate_vertex_contacts(compliant, indenter, rule, 1.0)?;
        Ok(contacts.iter().filter(|c| c.penetrating).count())
    }

    fn deform_compliant_against_indenter(
        &mut self,
        compliant: MeshId,
        indenter: MeshId,
        rule: &CollisionRule,
        push_fraction: f64,
    ) -> Result<SoftDeformOutcome> {
        let contacts = self.evaluate_vertex_contacts(compliant, indenter, rule, push_fraction)?;
        let penetrating_count = contacts.iter().filter(|c| c.penetrating).count();

        let (pc, mc_origin, rev, v_slots) = {
            let ec = self.entry(compliant)?;
            (
                ec.placement,
                ec.mesh.origin(),
                ec.mesh.revision(),
                ec.mesh.vertex_slots(),
            )
        };

        if rule.material_memory == MaterialMemory::Plastic {
            self.collision.drop_rest_state(compliant);
        }

        let has_valid_rest = self
            .collision
            .rest_positions
            .get(&compliant)
            .is_some_and(|st| st.rest_positions.len() == v_slots && st.expected_revision == rev);

        if rule.material_memory == MaterialMemory::Elastic
            && penetrating_count > 0
            && !has_valid_rest
        {
            let ec = self.entry(compliant)?;
            self.collision.rest_positions.insert(
                compliant,
                ElasticRestState {
                    rest_positions: ec.mesh.positions().to_vec(),
                    expected_revision: rev,
                },
            );
        }

        let rest_local_opt: Option<Vec<[f32; 3]>> =
            if rule.material_memory == MaterialMemory::Elastic {
                self.collision
                    .rest_positions
                    .get(&compliant)
                    .filter(|st| st.rest_positions.len() == v_slots && st.expected_revision == rev)
                    .map(|st| st.rest_positions.clone())
            } else {
                None
            };

        if penetrating_count == 0 && rest_local_opt.is_none() {
            return Ok(SoftDeformOutcome::default());
        }

        let mesh_data = {
            let ec = self.entry(compliant)?;
            build_mesh_data(&ec.mesh, &pc, mc_origin, rest_local_opt.as_deref())
        };
        if mesh_data.topology.alive_ids.is_empty() {
            return Ok(SoftDeformOutcome::default());
        }

        let n_alive = mesh_data.topology.alive_ids.len();
        let mut pos_work = mesh_data.topology.pos_0.clone();
        let mut recovered_vertices = 0usize;
        let mut exact_rest_snap = vec![false; n_alive];
        let recover_alpha = f64::from(rule.recovery_rate.clamp(0.0, 1.0));

        if let Some(rest_world) = &mesh_data.rest_world
            && recover_alpha > 0.0
        {
            let rest_contacts = self.evaluate_world_points_against_indenter(
                indenter,
                rule,
                rest_world,
                push_fraction,
            )?;
            for i in 0..n_alive {
                if !contacts[i].penetrating && !rest_contacts[i].penetrating {
                    let delta = vec_sub(rest_world[i], pos_work[i]);
                    if vec_norm(delta) > 1e-7 {
                        if recover_alpha >= 1.0 - 1e-7 {
                            pos_work[i] = rest_world[i];
                            exact_rest_snap[i] = true;
                        } else {
                            pos_work[i] = vec_add(pos_work[i], vec_scale(delta, recover_alpha));
                        }
                        recovered_vertices += 1;
                    }
                }
            }
        }

        if penetrating_count > 0 {
            match rule.soft_solver {
                SoftDeformSolver::SdfLaplacian => {
                    solve_sdf_laplacian(&mesh_data, &contacts, rule, &mut pos_work);
                }
                SoftDeformSolver::Kelvinlet => {
                    solve_kelvinlet(&mesh_data, &contacts, rule, &mut pos_work);
                }
                SoftDeformSolver::Xpbd => {
                    solve_xpbd(&mesh_data, &contacts, rule, &mut pos_work);
                }
            }
            for i in 0..n_alive {
                if exact_rest_snap[i]
                    && vec_norm(vec_sub(pos_work[i], mesh_data.topology.pos_0[i])) > 1e-7
                    && mesh_data
                        .rest_world
                        .as_ref()
                        .is_some_and(|rw| vec_norm(vec_sub(pos_work[i], rw[i])) > 1e-6)
                {
                    exact_rest_snap[i] = false;
                }
            }
        }

        let mut guard_triggered = false;
        if penetrating_count > 0 && rule.self_collision_guard {
            let ec = self.entry(compliant)?;
            guard_triggered = apply_self_collision_guard(
                &ec.mesh,
                &pc,
                mc_origin,
                &mesh_data.topology.alive_ids,
                &mesh_data.topology.pos_0,
                &mut pos_work,
            )?;
        }

        let mut changed_ids: Vec<VertexId> = Vec::new();
        let mut changed_positions: Vec<[f32; 3]> = Vec::new();
        let mut deformed_vertices = 0usize;

        {
            let ec = self.entry(compliant)?;
            for i in 0..n_alive {
                let vid = mesh_data.topology.alive_ids[i];
                let old_local = ec.mesh.positions()[vid as usize];
                let new_local = if exact_rest_snap[i]
                    && let Some(rl) = &rest_local_opt
                {
                    rl[vid as usize]
                } else {
                    let wl = world_to_local(&pc, mc_origin, pos_work[i]);
                    [wl[0] as f32, wl[1] as f32, wl[2] as f32]
                };
                let d_local = [
                    f64::from(new_local[0]) - f64::from(old_local[0]),
                    f64::from(new_local[1]) - f64::from(old_local[1]),
                    f64::from(new_local[2]) - f64::from(old_local[2]),
                ];
                // A world length, as every other here.
                let d = if pc.is_scaled() {
                    pc.apply_direction(d_local)
                } else {
                    d_local
                };
                if vec_norm(d) > 1e-6 {
                    changed_ids.push(vid);
                    changed_positions.push(new_local);
                    if contacts[i].penetrating || (penetrating_count > 0 && !exact_rest_snap[i]) {
                        deformed_vertices += 1;
                    }
                }
            }
        }

        if !changed_ids.is_empty() {
            self.edit(compliant, |tx| {
                tx.set_positions(&changed_ids, &changed_positions)
            })?;
        }

        if rule.material_memory == MaterialMemory::Elastic {
            let new_rev = self.entry(compliant)?.mesh.revision();
            if let Some(st) = self.collision.rest_positions.get_mut(&compliant) {
                st.expected_revision = new_rev;
            }
        }

        Ok(SoftDeformOutcome {
            deformed_vertices,
            recovered_vertices,
            guard_triggered,
        })
    }

    fn evaluate_vertex_contacts(
        &mut self,
        compliant: MeshId,
        indenter: MeshId,
        rule: &CollisionRule,
        push_fraction: f64,
    ) -> Result<Vec<VertexContact>> {
        let ec = self.entry(compliant)?;
        let mut world_pts: Vec<[f64; 3]> = Vec::new();
        for v in 0..ec.mesh.vertex_slots() as u32 {
            if !ec.mesh.is_vertex_alive(v) {
                continue;
            }
            let p = ec.mesh.positions()[v as usize].map(f64::from);
            world_pts.push(local_to_world(&ec.placement, ec.mesh.origin(), p));
        }
        self.evaluate_world_points_against_indenter(indenter, rule, &world_pts, push_fraction)
    }

    fn evaluate_world_points_against_indenter(
        &mut self,
        indenter: MeshId,
        rule: &CollisionRule,
        world_pts: &[[f64; 3]],
        push_fraction: f64,
    ) -> Result<Vec<VertexContact>> {
        if world_pts.is_empty() {
            return Ok(Vec::new());
        }
        let pad_margin = rule.contact_margin.max(0.0);

        // A non-uniform scale distorts distances, and which point is nearest
        // with them: the indenter is then a copy of it as placed, in the world
        // around its own origin.
        let baked = {
            let ei = self.entry(indenter)?;
            match ei.placement.uniform_scale() {
                Some(_) => None,
                None => {
                    let base = world_base(&ei.mesh, &ei.placement);
                    Some(baked_pose(ei, &ei.placement, base)?)
                }
            }
        };
        match baked {
            Some((mut mesh, bvh)) => {
                let origin = mesh.origin();
                let at = (&Placement::IDENTITY, origin);
                evaluate_indenter(
                    &mut mesh,
                    &bvh,
                    at,
                    rule,
                    world_pts,
                    push_fraction,
                    pad_margin,
                )
            }
            None => {
                let ei = self.entry_mut(indenter)?;
                let (placement, origin) = (ei.placement, ei.mesh.origin());
                let at = (&placement, origin);
                let (mesh, bvh) = (&mut ei.mesh, &ei.bvh);
                evaluate_indenter(mesh, bvh, at, rule, world_pts, push_fraction, pad_margin)
            }
        }
    }
}

/// Where each world point is to an indenter of a uniform scale: `mesh` as
/// placed by `at`, its placement and origin.
fn evaluate_indenter(
    mesh: &mut Mesh,
    bvh: &Bvh,
    at: (&Placement, [f64; 3]),
    rule: &CollisionRule,
    world_pts: &[[f64; 3]],
    push_fraction: f64,
    pad_margin: f64,
) -> Result<Vec<VertexContact>> {
    let (placement, origin) = at;
    let Some(ib_local) = bvh.root_bounds().filter(|b| !b.is_empty()) else {
        return Ok(vec![VertexContact::default(); world_pts.len()]);
    };
    match rule.distance_backend {
        CollisionDistanceBackend::BvhSdf => {
            let i_world_box = box_to_world(&ib_local, placement, origin);
            let sign = SignData::build(mesh, bvh, SignMode::Auto)?;
            let eval = SdfEvaluator::Bvh {
                mesh,
                bvh,
                sign: &sign,
                placement,
                origin,
                world_box: pad_box(&i_world_box, pad_margin + 1e-4),
            };
            evaluate_sdf_bvh(eval, world_pts, pad_margin, push_fraction)
        }
        CollisionDistanceBackend::FhEdt => {
            let ext = [
                f64::from(ib_local.max[0] - ib_local.min[0]),
                f64::from(ib_local.max[1] - ib_local.min[1]),
                f64::from(ib_local.max[2] - ib_local.min[2]),
            ];
            let max_ext = ext[0].max(ext[1]).max(ext[2]).max(1e-3);
            let voxel_size = ((max_ext / 32.0) as f32).max(1e-4);
            let scale = placement.max_scale();
            let local_margin = (pad_margin / scale.max(1e-6)) as f32 + 2.0 * voxel_size;
            let spec = GridSpec::around_bounds(
                ib_local.min,
                ib_local.max,
                voxel_size,
                local_margin,
                4,
                DEFAULT_MAX_VOXELS,
            )?;
            let grid = fh_signed_grid(mesh, spec, DEFAULT_MAX_VOXELS)?;
            let center_local = [
                0.5 * f64::from(ib_local.min[0] + ib_local.max[0]),
                0.5 * f64::from(ib_local.min[1] + ib_local.max[1]),
                0.5 * f64::from(ib_local.min[2] + ib_local.max[2]),
            ];
            let eval = SdfEvaluator::FhEdt {
                grid: &grid,
                placement,
                origin,
                center_local,
                scale,
            };
            evaluate_sdf_fhedt(eval, world_pts, pad_margin, push_fraction)
        }
    }
}

fn evaluate_sdf_bvh(
    eval: SdfEvaluator<'_>,
    world_pts: &[[f64; 3]],
    margin: f64,
    push_fraction: f64,
) -> Result<Vec<VertexContact>> {
    let SdfEvaluator::Bvh {
        mesh,
        bvh,
        sign,
        placement,
        origin,
        world_box,
    } = eval
    else {
        return Ok(vec![VertexContact::default(); world_pts.len()]);
    };

    let mut out = vec![VertexContact::default(); world_pts.len()];
    let mut cand_indices: Vec<usize> = Vec::new();
    let mut cand_local: Vec<[f32; 3]> = Vec::new();

    for (idx, &pw) in world_pts.iter().enumerate() {
        let in_box = pw[0] >= world_box.min[0]
            && pw[0] <= world_box.max[0]
            && pw[1] >= world_box.min[1]
            && pw[1] <= world_box.max[1]
            && pw[2] >= world_box.min[2]
            && pw[2] <= world_box.max[2];
        if in_box {
            let pl = world_to_local(placement, origin, pw);
            cand_indices.push(idx);
            cand_local.push([pl[0] as f32, pl[1] as f32, pl[2] as f32]);
        }
    }

    if cand_local.is_empty() {
        return Ok(out);
    }

    let mut dists = try_vec::<f32>(cand_local.len())?;
    dists.resize(cand_local.len(), 0.0);
    signed_distances(mesh, bvh, sign, &cand_local, f32::INFINITY, &mut dists)?;

    let mut hits: Vec<Option<ClosestHit>> = try_vec(cand_local.len())?;
    hits.resize(cand_local.len(), None);
    closest_point_many(
        mesh,
        bvh,
        &cand_local,
        &ClosestOptions::default(),
        &mut hits,
    )?;

    let dummy_entry_normal = |tri: TriangleId| -> [f64; 3] {
        let Some([v0, v1, v2]) = mesh.triangle(tri) else {
            return [0.0, 1.0, 0.0];
        };
        let pos = mesh.positions();
        let p0 = pos[v0 as usize].map(f64::from);
        let p1 = pos[v1 as usize].map(f64::from);
        let p2 = pos[v2 as usize].map(f64::from);
        let n_local = vec_cross(vec_sub(p1, p0), vec_sub(p2, p0));
        if vec_norm(n_local) <= 1e-12 {
            [0.0, 1.0, 0.0]
        } else {
            placement.normal_to_world(vec_normalize(n_local, [0.0, 1.0, 0.0]))
        }
    };

    for k in 0..cand_local.len() {
        let idx = cand_indices[k];
        let pw = world_pts[idx];
        let sd_local = f64::from(dists[k]);
        let Some(hit) = hits[k] else {
            continue;
        };
        let hw = local_to_world(placement, origin, hit.point.map(f64::from));
        let gap = vec_norm(vec_sub(hw, pw));
        if sd_local < 0.0 {
            let n_world = if gap > 1e-8 {
                vec_normalize(vec_sub(hw, pw), dummy_entry_normal(hit.triangle))
            } else {
                dummy_entry_normal(hit.triangle)
            };
            let full_target = vec_add(hw, vec_scale(n_world, margin));
            let target = vec_add(pw, vec_scale(vec_sub(full_target, pw), push_fraction));
            out[idx] = VertexContact {
                penetrating: true,
                signed_dist_world: -gap,
                outward_world: n_world,
                target_world: target,
            };
        } else if gap < margin {
            let n_world = if gap > 1e-8 {
                vec_normalize(vec_sub(pw, hw), dummy_entry_normal(hit.triangle))
            } else {
                dummy_entry_normal(hit.triangle)
            };
            let full_target = vec_add(hw, vec_scale(n_world, margin));
            let target = vec_add(pw, vec_scale(vec_sub(full_target, pw), push_fraction));
            out[idx] = VertexContact {
                penetrating: true,
                signed_dist_world: gap,
                outward_world: n_world,
                target_world: target,
            };
        } else {
            out[idx].signed_dist_world = gap;
        }
    }

    Ok(out)
}

fn evaluate_sdf_fhedt(
    eval: SdfEvaluator<'_>,
    world_pts: &[[f64; 3]],
    margin: f64,
    push_fraction: f64,
) -> Result<Vec<VertexContact>> {
    let SdfEvaluator::FhEdt {
        grid,
        placement,
        origin,
        center_local,
        scale,
    } = eval
    else {
        return Ok(vec![VertexContact::default(); world_pts.len()]);
    };

    let mut out = vec![VertexContact::default(); world_pts.len()];
    let margin_local = margin / scale.max(1e-6);

    for (idx, &pw) in world_pts.iter().enumerate() {
        let p_loc64 = world_to_local(placement, origin, pw);
        let p_loc = [p_loc64[0] as f32, p_loc64[1] as f32, p_loc64[2] as f32];
        let Some((sd, grad)) = grid.sample_with_gradient(p_loc) else {
            continue;
        };
        let sd_l = f64::from(sd);
        if sd_l < margin_local {
            let mut cur_local = p_loc64;
            let mut n_local = vec_normalize(
                grad.map(f64::from),
                vec_normalize(vec_sub(cur_local, center_local), [0.0, 1.0, 0.0]),
            );
            // Two-step gradient push reaches the zero-isosurface + margin accurately.
            for _ in 0..2 {
                let sample_p = [
                    cur_local[0] as f32,
                    cur_local[1] as f32,
                    cur_local[2] as f32,
                ];
                let Some((d_step, g_step)) = grid.sample_with_gradient(sample_p) else {
                    break;
                };
                if f64::from(d_step) >= margin_local {
                    break;
                }
                let dir = vec_normalize(g_step.map(f64::from), n_local);
                n_local = dir;
                let push = (margin_local - f64::from(d_step)).max(0.0);
                cur_local = vec_add(cur_local, vec_scale(dir, push));
            }
            let surf_world = local_to_world(placement, origin, cur_local);
            let n_world = placement.normal_to_world(n_local);
            let target = vec_add(pw, vec_scale(vec_sub(surf_world, pw), push_fraction));
            out[idx] = VertexContact {
                penetrating: true,
                signed_dist_world: sd_l * scale,
                outward_world: n_world,
                target_world: target,
            };
        } else {
            out[idx].signed_dist_world = sd_l * scale;
        }
    }

    Ok(out)
}

fn build_mesh_data(
    mesh: &Mesh,
    placement: &Placement,
    origin: [f64; 3],
    rest_local: Option<&[[f32; 3]]>,
) -> SoftDeformMeshData {
    let slots = mesh.vertex_slots();
    let mut slot_to_alive = vec![usize::MAX; slots];
    let mut alive_ids: Vec<VertexId> = Vec::new();
    let mut pos_0: Vec<[f64; 3]> = Vec::new();
    let mut rest_world_vec: Option<Vec<[f64; 3]>> = rest_local.map(|_| Vec::new());

    for s in 0..slots {
        let vid = s as VertexId;
        if !mesh.is_vertex_alive(vid) {
            continue;
        }
        slot_to_alive[s] = alive_ids.len();
        alive_ids.push(vid);
        let pw = local_to_world(placement, origin, mesh.positions()[s].map(f64::from));
        pos_0.push(pw);
        if let (Some(rl), Some(rw)) = (rest_local, rest_world_vec.as_mut()) {
            rw.push(local_to_world(placement, origin, rl[s].map(f64::from)));
        }
    }

    let n = alive_ids.len();
    let mut neighbor_map: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); n];
    let mut edge_opposites: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();

    for t in 0..mesh.triangle_slots() as u32 {
        let Some([v0, v1, v2]) = mesh.triangle(t) else {
            continue;
        };
        let i0 = slot_to_alive[v0 as usize];
        let i1 = slot_to_alive[v1 as usize];
        let i2 = slot_to_alive[v2 as usize];
        if i0 == usize::MAX || i1 == usize::MAX || i2 == usize::MAX {
            continue;
        }
        let tri = [i0, i1, i2];
        for c in 0..3 {
            let a = tri[c];
            let b = tri[(c + 1) % 3];
            let opp = tri[(c + 2) % 3];
            let pa = pos_0[a];
            let pb = pos_0[b];
            let po = pos_0[opp];
            let e1 = vec_sub(pa, po);
            let e2 = vec_sub(pb, po);
            let cross_norm = vec_norm(vec_cross(e1, e2));
            let cot = if cross_norm > 1e-12 {
                (vec_dot(e1, e2) / cross_norm).max(0.05)
            } else {
                0.5
            };
            *neighbor_map[a].entry(b).or_insert(0.0) += 0.5 * cot;
            *neighbor_map[b].entry(a).or_insert(0.0) += 0.5 * cot;
            let key = if a < b { (a, b) } else { (b, a) };
            edge_opposites.entry(key).or_default().push(opp);
        }
    }

    let mut neighbors: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
    for map in neighbor_map {
        let sum: f64 = map.values().sum();
        let len = map.len();
        if sum > 1e-12 {
            neighbors.push(map.into_iter().map(|(k, w)| (k, w / sum)).collect());
        } else if len > 0 {
            let u = 1.0 / len as f64;
            neighbors.push(map.into_keys().map(|k| (k, u)).collect());
        } else {
            neighbors.push(Vec::new());
        }
    }

    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(edge_opposites.len());
    let mut bends: Vec<(usize, usize)> = Vec::new();
    for ((a, b), opps) in edge_opposites {
        edges.push((a, b));
        if opps.len() >= 2 && opps[0] != opps[1] {
            bends.push((opps[0], opps[1]));
        }
    }

    SoftDeformMeshData {
        topology: MeshTopology {
            alive_ids,
            pos_0,
            neighbors,
            edges,
            bends,
        },
        rest_world: rest_world_vec,
    }
}

fn solve_sdf_laplacian(
    data: &SoftDeformMeshData,
    contacts: &[VertexContact],
    rule: &CollisionRule,
    pos: &mut [[f64; 3]],
) {
    let n = pos.len();
    let base = pos.to_vec();
    let mut in_roi = vec![false; n];
    for (i, c) in contacts.iter().enumerate() {
        if c.penetrating {
            pos[i] = c.target_world;
            in_roi[i] = true;
        }
    }

    for _ in 0..2 {
        let prev = in_roi.clone();
        for (i, &was_in) in prev.iter().enumerate() {
            if was_in {
                for &(nb, _) in &data.topology.neighbors[i] {
                    in_roi[nb] = true;
                }
            }
        }
    }

    let alpha = (0.25 + 0.55 / (1.0 + f64::from(rule.stiffness))).clamp(0.1, 0.85);
    let iters = (4.0 + 12.0 * f64::from(rule.compliance).min(2.0)).round() as usize;
    let mut disp: Vec<[f64; 3]> = (0..n).map(|i| vec_sub(pos[i], base[i])).collect();

    for _ in 0..iters {
        let prev = disp.clone();
        for i in 0..n {
            if !in_roi[i] || data.topology.neighbors[i].is_empty() {
                continue;
            }
            let mut avg = [0.0; 3];
            for &(nb, w) in &data.topology.neighbors[i] {
                avg[0] += w * prev[nb][0];
                avg[1] += w * prev[nb][1];
                avg[2] += w * prev[nb][2];
            }
            let blended = vec_add(vec_scale(prev[i], 1.0 - alpha), vec_scale(avg, alpha));
            if contacts[i].penetrating {
                let candidate = vec_add(base[i], blended);
                let signed = vec_dot(
                    vec_sub(candidate, contacts[i].target_world),
                    contacts[i].outward_world,
                );
                if signed >= 0.0 {
                    disp[i] = blended;
                } else {
                    disp[i] = vec_sub(
                        vec_sub(candidate, vec_scale(contacts[i].outward_world, signed)),
                        base[i],
                    );
                }
            } else {
                disp[i] = blended;
            }
        }
    }

    for i in 0..n {
        if in_roi[i] {
            pos[i] = vec_add(base[i], disp[i]);
        }
    }
}

fn solve_kelvinlet(
    data: &SoftDeformMeshData,
    contacts: &[VertexContact],
    rule: &CollisionRule,
    pos: &mut [[f64; 3]],
) {
    let n = pos.len();
    let base = pos.to_vec();
    let mut total_w = 0.0;
    let mut center = [0.0; 3];
    let mut net_push = [0.0; 3];
    let mut max_push = 0.0f64;

    for i in 0..n {
        if !contacts[i].penetrating {
            continue;
        }
        let d = vec_sub(contacts[i].target_world, base[i]);
        let d_len = vec_norm(d);
        let w = d_len.max(1e-6);
        center = vec_add(center, vec_scale(base[i], w));
        net_push = vec_add(net_push, d);
        total_w += w;
        if d_len > max_push {
            max_push = d_len;
        }
    }
    if total_w <= 1e-12 {
        return;
    }
    center = vec_scale(center, 1.0 / total_w);

    let mut mean_edge = 0.0;
    for &(a, b) in &data.topology.edges {
        mean_edge += vec_norm(vec_sub(base[a], base[b]));
    }
    let edge_scale = if data.topology.edges.is_empty() {
        0.1
    } else {
        (mean_edge / data.topology.edges.len() as f64).max(1e-3)
    };

    let mut spread = 0.0f64;
    for i in 0..n {
        if contacts[i].penetrating {
            spread = spread.max(vec_norm(vec_sub(base[i], center)));
        }
    }
    let eps = (spread + edge_scale)
        * (0.65 + 1.2 * f64::from(rule.compliance).min(2.0) / (f64::from(rule.stiffness) + 0.25));
    let eps = eps.max(edge_scale);

    // Regularized incompressible Kelvinlet vector kernel (nu = 0.35).
    let nu = 0.35;
    let c_ratio = 1.0 / (4.0 * (1.0 - nu));
    let g0 = (1.5 - c_ratio) / eps;
    let force_dir = vec_normalize(net_push, [0.0, 1.0, 0.0]);
    let f = vec_scale(force_dir, max_push);

    for i in 0..n {
        let r = vec_sub(base[i], center);
        let r2 = vec_norm2(r);
        let r_eps = (r2 + eps * eps).sqrt();
        let inv_re = 1.0 / r_eps;
        let inv_re3 = inv_re * inv_re * inv_re;
        let iso = ((1.0 - c_ratio) * inv_re + 0.5 * eps * eps * inv_re3) / g0;
        let radial = (c_ratio * inv_re3 / g0) * vec_dot(f, r);
        let u = vec_add(vec_scale(f, iso), vec_scale(r, radial));
        let cand = vec_add(base[i], u);
        if contacts[i].penetrating {
            let signed = vec_dot(
                vec_sub(cand, contacts[i].target_world),
                contacts[i].outward_world,
            );
            pos[i] = if signed < 0.0 {
                vec_sub(cand, vec_scale(contacts[i].outward_world, signed))
            } else {
                cand
            };
        } else {
            pos[i] = cand;
        }
    }
}

fn solve_xpbd(
    data: &SoftDeformMeshData,
    contacts: &[VertexContact],
    rule: &CollisionRule,
    pos: &mut [[f64; 3]],
) {
    let n = pos.len();
    let rest_ref = data.rest_world.as_deref().unwrap_or(&data.topology.pos_0);
    let mut in_roi = vec![false; n];
    for (i, c) in contacts.iter().enumerate() {
        if c.penetrating {
            pos[i] = c.target_world;
            in_roi[i] = true;
        }
    }
    for _ in 0..3 {
        let prev = in_roi.clone();
        for (i, &was_in) in prev.iter().enumerate() {
            if was_in {
                for &(nb, _) in &data.topology.neighbors[i] {
                    in_roi[nb] = true;
                }
            }
        }
    }

    let inv_mass: Vec<f64> = (0..n).map(|i| if in_roi[i] { 1.0 } else { 0.0 }).collect();
    let alpha_edge =
        f64::from(rule.compliance.max(1e-6)) * (0.25 / (f64::from(rule.stiffness) + 0.1));
    let alpha_bend = alpha_edge * 4.0;
    let rest_edge_lens: Vec<f64> = data
        .topology
        .edges
        .iter()
        .map(|&(a, b)| vec_norm(vec_sub(rest_ref[a], rest_ref[b])))
        .collect();
    let rest_bend_lens: Vec<f64> = data
        .topology
        .bends
        .iter()
        .map(|&(a, b)| vec_norm(vec_sub(rest_ref[a], rest_ref[b])))
        .collect();

    let mut lambdas_edge = vec![0.0; data.topology.edges.len()];
    let mut lambdas_bend = vec![0.0; data.topology.bends.len()];

    for _ in 0..8 {
        for (ei, &(a, b)) in data.topology.edges.iter().enumerate() {
            let wa = inv_mass[a];
            let wb = inv_mass[b];
            let wsum = wa + wb + alpha_edge;
            if wa + wb == 0.0 || wsum <= 1e-12 {
                continue;
            }
            let diff = vec_sub(pos[a], pos[b]);
            let len = vec_norm(diff);
            if len <= 1e-12 {
                continue;
            }
            let c = len - rest_edge_lens[ei];
            let dl = (-c - alpha_edge * lambdas_edge[ei]) / wsum;
            lambdas_edge[ei] += dl;
            let n_ab = vec_scale(diff, 1.0 / len);
            if wa > 0.0 {
                pos[a] = vec_add(pos[a], vec_scale(n_ab, wa * dl));
            }
            if wb > 0.0 {
                pos[b] = vec_sub(pos[b], vec_scale(n_ab, wb * dl));
            }
        }

        for (bi, &(a, b)) in data.topology.bends.iter().enumerate() {
            let wa = inv_mass[a];
            let wb = inv_mass[b];
            let wsum = wa + wb + alpha_bend;
            if wa + wb == 0.0 || wsum <= 1e-12 {
                continue;
            }
            let diff = vec_sub(pos[a], pos[b]);
            let len = vec_norm(diff);
            if len <= 1e-12 {
                continue;
            }
            let c = len - rest_bend_lens[bi];
            let dl = (-c - alpha_bend * lambdas_bend[bi]) / wsum;
            lambdas_bend[bi] += dl;
            let n_ab = vec_scale(diff, 1.0 / len);
            if wa > 0.0 {
                pos[a] = vec_add(pos[a], vec_scale(n_ab, wa * dl));
            }
            if wb > 0.0 {
                pos[b] = vec_sub(pos[b], vec_scale(n_ab, wb * dl));
            }
        }

        for i in 0..n {
            if contacts[i].penetrating {
                let signed = vec_dot(
                    vec_sub(pos[i], contacts[i].target_world),
                    contacts[i].outward_world,
                );
                if signed < 0.0 {
                    pos[i] = vec_sub(pos[i], vec_scale(contacts[i].outward_world, signed));
                }
            }
        }
    }
}

fn apply_self_collision_guard(
    mesh: &Mesh,
    placement: &Placement,
    origin: [f64; 3],
    alive_ids: &[VertexId],
    pos_0: &[[f64; 3]],
    pos_work: &mut [[f64; 3]],
) -> Result<bool> {
    let build_opts = BuildOptions::default();
    let (mut cand, _) = baked_mesh(mesh, |p| p.map(f64::from), &build_opts)?;
    let local_new: Vec<[f32; 3]> = pos_work
        .iter()
        .map(|&pw| {
            let l = world_to_local(placement, origin, pw);
            [l[0] as f32, l[1] as f32, l[2] as f32]
        })
        .collect();
    {
        let mut tx = cand.begin_edit();
        tx.set_positions(alive_ids, &local_new)?;
        tx.commit()?;
    }
    let bvh = Bvh::build(&cand, &build_opts)?;
    let mut pairs: Vec<(TriangleId, TriangleId)> = Vec::new();
    self_intersecting_pairs(&cand, &bvh, &mut pairs)?;

    let mut slot_to_alive = vec![usize::MAX; mesh.vertex_slots()];
    for (i, &vid) in alive_ids.iter().enumerate() {
        slot_to_alive[vid as usize] = i;
    }

    let collect_folded = |cur: &[[f64; 3]]| -> BTreeSet<usize> {
        let mut out = BTreeSet::new();
        for t in 0..mesh.triangle_slots() as u32 {
            let Some([v0, v1, v2]) = mesh.triangle(t) else {
                continue;
            };
            let i0 = slot_to_alive[v0 as usize];
            let i1 = slot_to_alive[v1 as usize];
            let i2 = slot_to_alive[v2 as usize];
            if i0 == usize::MAX || i1 == usize::MAX || i2 == usize::MAX {
                continue;
            }
            let n0 = vec_cross(vec_sub(pos_0[i1], pos_0[i0]), vec_sub(pos_0[i2], pos_0[i0]));
            let n1 = vec_cross(vec_sub(cur[i1], cur[i0]), vec_sub(cur[i2], cur[i0]));
            let l0 = vec_norm(n0);
            let l1 = vec_norm(n1);
            if l0 > 1e-12 && l1 > 1e-12 && vec_dot(n0, n1) / (l0 * l1) < -0.25 {
                out.insert(i0);
                out.insert(i1);
                out.insert(i2);
            }
        }
        out
    };

    let folded = collect_folded(pos_work);
    if pairs.is_empty() && folded.is_empty() {
        return Ok(false);
    }

    let add_flagged = |pairs: &[(TriangleId, TriangleId)],
                       folded: BTreeSet<usize>,
                       flagged: &mut BTreeSet<usize>| {
        flagged.extend(folded);
        for &(ta, tb) in pairs {
            for tri_id in [ta, tb] {
                if let Some(vs) = mesh.triangle(tri_id) {
                    for v in vs {
                        let i = slot_to_alive[v as usize];
                        if i != usize::MAX {
                            flagged.insert(i);
                        }
                    }
                }
            }
        }
    };

    let mut flagged = BTreeSet::new();
    add_flagged(&pairs, folded, &mut flagged);

    let saved = pos_work.to_vec();
    let recheck = |cand: &mut Mesh,
                   cur: &[[f64; 3]],
                   out_pairs: &mut Vec<(TriangleId, TriangleId)>|
     -> Result<BTreeSet<usize>> {
        let local_all: Vec<[f32; 3]> = cur
            .iter()
            .map(|&pw| {
                let l = world_to_local(placement, origin, pw);
                [l[0] as f32, l[1] as f32, l[2] as f32]
            })
            .collect();
        {
            let mut tx = cand.begin_edit();
            tx.set_positions(alive_ids, &local_all)?;
            tx.commit()?;
        }
        let test_bvh = Bvh::build(cand, &build_opts)?;
        out_pairs.clear();
        self_intersecting_pairs(cand, &test_bvh, out_pairs)?;
        Ok(collect_folded(cur))
    };

    let mut cleared = false;
    for factor in [0.5, 0.25, 0.125, 0.0625, 0.0] {
        for &i in &flagged {
            pos_work[i] = vec_add(pos_0[i], vec_scale(vec_sub(saved[i], pos_0[i]), factor));
        }
        let now_folded = recheck(&mut cand, pos_work, &mut pairs)?;
        if pairs.is_empty() && now_folded.is_empty() {
            cleared = true;
            break;
        }
        add_flagged(&pairs, now_folded, &mut flagged);
    }

    if !cleared {
        for factor in [0.5, 0.25, 0.125, 0.0625, 0.0] {
            for i in 0..pos_work.len() {
                pos_work[i] = vec_add(pos_0[i], vec_scale(vec_sub(saved[i], pos_0[i]), factor));
            }
            let now_folded = recheck(&mut cand, pos_work, &mut pairs)?;
            if pairs.is_empty() && now_folded.is_empty() {
                break;
            }
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::relative_rigid;
    use crate::query::tests::{RX90, RZ90, cube, place};
    use crate::query::world_box;
    use crate::{MeshOptions, Placement, SceneOptions, UpdateMode};

    /// Every pair `a < b` by calling Phase 3b directly.
    fn brute(scene: &Scene, margin: f64, pairs: &[(MeshId, MeshId)]) -> Vec<Contact> {
        let m32 = margin as f32;
        assert_eq!(f64::from(m32), margin, "test margins are exact in f32");
        let mut out = Vec::new();
        for &(a, b) in pairs {
            let (ea, eb) = (scene.entries.get(a).unwrap(), scene.entries.get(b).unwrap());
            let xf = relative_rigid(
                &ea.placement,
                ea.mesh.origin(),
                &eb.placement,
                eb.mesh.origin(),
            );
            let (ma, ba, mb, bb) = (&ea.mesh, &ea.bvh, &eb.mesh, &eb.bvh);
            let intersecting = fast_mesh_query::intersects(ma, ba, mb, bb, &xf).unwrap();
            let distance = if intersecting {
                0.0
            } else {
                match fast_mesh_query::mesh_distance(ma, ba, mb, bb, &xf, m32).unwrap() {
                    Some(d) => f64::from(d.distance),
                    None => continue,
                }
            };
            let (mut ta, mut tb) = (Vec::new(), Vec::new());
            fast_mesh_query::triangles_near(ma, ba, mb, bb, &xf, m32, &mut ta, &mut tb).unwrap();
            out.push(Contact {
                a,
                b,
                intersecting,
                distance,
                triangles_a: ta,
                triangles_b: tb,
            });
        }
        out
    }

    fn all_pairs(scene: &Scene) -> Vec<(MeshId, MeshId)> {
        let ids: Vec<MeshId> = scene.ids().collect();
        let mut pairs = Vec::new();
        for (i, &a) in ids.iter().enumerate() {
            for &b in &ids[i + 1..] {
                pairs.push((a, b));
            }
        }
        pairs
    }

    /// A rotation by 45 degrees about z.
    fn rz45() -> [[f64; 3]; 3] {
        let (s, c) = 45f64.to_radians().sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    fn scene() -> (Scene, Vec<MeshId>) {
        let mut scene = Scene::new(SceneOptions::default());
        let identity = Placement::IDENTITY.rotation;
        let ids = vec![
            // [0, 2]^3.
            scene
                .add_mesh(cube([0.0; 3], 2.0), MeshOptions::default())
                .unwrap(),
            // x in [1.5, 2.5]: crosses the first.
            scene
                .add_mesh(cube([0.0; 3], 1.0), place(RZ90, [2.5, 0.5, 0.5]))
                .unwrap(),
            // y in [2.25, 3.25]: 0.25 from the first.
            scene
                .add_mesh(cube([0.0; 3], 1.0), place(RX90, [0.5, 3.25, 1.0]))
                .unwrap(),
            // Far from everything.
            scene
                .add_mesh(cube([0.0; 3], 1.0), place(identity, [10.0, 10.0, 10.0]))
                .unwrap(),
            // Turned 45 degrees, off the first's edge x = 0, y = 2: its box
            // is about 0.41 away, its surface about 0.91.
            scene
                .add_mesh(
                    cube([-0.5, -0.5, 0.0], 1.0),
                    place(rz45(), [-1.0, 3.0, 0.5]),
                )
                .unwrap(),
            // 0.6 beyond the first along +z: outside a 0.5 margin.
            scene
                .add_mesh(cube([0.0; 3], 1.0), place(identity, [0.5, 0.5, 2.6]))
                .unwrap(),
        ];
        (scene, ids)
    }

    #[test]
    fn find_contacts_matches_all_pairs() {
        let (mut scene, ids) = scene();
        let pairs = all_pairs(&scene);
        for margin in [0.0, 0.25, 0.5, 1.0, 4.0, f64::INFINITY] {
            let want = if margin.is_finite() {
                brute(&scene, margin, &pairs)
            } else {
                brute(&scene, f64::from(f32::MAX), &pairs)
            };
            let mut out = Vec::new();
            let n = scene
                .find_contacts(margin, &ContactFilter::default(), &mut out)
                .unwrap();
            assert_eq!(n, out.len());
            assert_eq!(out, want, "margin {margin}");
            if !margin.is_finite() {
                assert_eq!(out.len(), pairs.len());
            }
        }
        let mut out = Vec::new();
        scene
            .find_contacts(0.5, &ContactFilter::default(), &mut out)
            .unwrap();
        let found: Vec<_> = out
            .iter()
            .map(|c| (c.a, c.b, c.intersecting, c.distance))
            .collect();
        // The turned cube passes the broadphase but not the narrow phase.
        let (e0, e4) = (
            scene.entries.get(ids[0]).unwrap(),
            scene.entries.get(ids[4]).unwrap(),
        );
        let box_gap = world_box(e0).distance_to_box(&world_box(e4));
        assert!((0.4..0.45).contains(&box_gap), "{box_gap}");
        assert!(out.iter().all(|c| (c.a, c.b) != (ids[0], ids[4])));
        let d = scene.mesh_distance(ids[0], ids[4], 1.0).unwrap().unwrap();
        assert!((0.9..0.95).contains(&d.distance), "{}", d.distance);
        assert!(found.contains(&(ids[0], ids[1], true, 0.0)));
        assert!(found.contains(&(ids[0], ids[2], false, 0.25)));
        assert!(out.iter().all(|c| c.a != ids[3] && c.b != ids[3]));
        assert!(out.iter().all(|c| c.b != ids[5] || c.a != ids[0]));
        assert!(out.windows(2).all(|w| (w[0].a, w[0].b) < (w[1].a, w[1].b)));
    }

    #[test]
    fn find_contacts_with_a_pair_filter() {
        let (mut scene, ids) = scene();
        let filter = ContactFilter {
            pairs: Some(vec![
                (ids[2], ids[0]),
                (ids[0], ids[2]),
                (ids[3], ids[1]),
                (ids[1], ids[0]),
            ]),
        };
        let mut out = vec![];
        let n = scene.find_contacts(0.5, &filter, &mut out).unwrap();
        let want = brute(&scene, 0.5, &[(ids[0], ids[1]), (ids[0], ids[2])]);
        assert_eq!(n, 2);
        assert_eq!(out, want);
        // Existing elements stay in front.
        let n = scene.find_contacts(0.5, &filter, &mut out).unwrap();
        assert_eq!(n, 2);
        assert_eq!(out[..2], out[2..]);
    }

    #[test]
    fn find_contacts_staleness_and_errors() {
        let (mut scene, ids) = scene();
        let keep = || {
            vec![Contact {
                a: ids[0],
                b: ids[1],
                intersecting: false,
                distance: 9.0,
                triangles_a: vec![],
                triangles_b: vec![],
            }]
        };
        // A stale Manual mesh fails a search over all pairs, and one naming
        // it; `out` is left as it was.
        scene.set_mode(ids[3], UpdateMode::Manual).unwrap();
        scene
            .edit(ids[3], |tx| tx.set_positions(&[0], &[[0.0, 0.0, -0.5]]))
            .unwrap();
        let lazy = ids[4];
        scene.set_mode(lazy, UpdateMode::Lazy).unwrap();
        scene
            .edit(lazy, |tx| tx.set_positions(&[0], &[[-0.5, -0.5, -0.25]]))
            .unwrap();
        let mut out = keep();
        let all = ContactFilter::default();
        assert_eq!(
            scene.find_contacts(0.5, &all, &mut out),
            Err(SceneError::StaleBvh(ids[3]))
        );
        assert_eq!(out, keep());
        let with = ContactFilter {
            pairs: Some(vec![(ids[0], ids[3])]),
        };
        assert_eq!(
            scene.find_contacts(0.5, &with, &mut out),
            Err(SceneError::StaleBvh(ids[3]))
        );
        assert_eq!(out, keep());
        // The search over all pairs still updated the Lazy mesh first.
        assert_eq!(scene.is_current(lazy), Ok(true));
        scene
            .edit(lazy, |tx| tx.set_positions(&[0], &[[-0.5, -0.5, -0.3]]))
            .unwrap();
        // A filter that leaves it out runs, and updates only what it names.
        scene
            .edit(ids[2], |tx| tx.set_positions(&[0], &[[0.0, 0.0, -0.1]]))
            .unwrap();
        scene.set_mode(ids[2], UpdateMode::Lazy).unwrap();
        scene
            .edit(ids[2], |tx| tx.set_positions(&[0], &[[0.0, 0.0, -0.2]]))
            .unwrap();
        let without = ContactFilter {
            pairs: Some(vec![(ids[0], ids[2])]),
        };
        assert_eq!(scene.find_contacts(0.5, &without, &mut out), Ok(1));
        assert_eq!(scene.is_current(ids[2]), Ok(true));
        assert_eq!(scene.is_current(lazy), Ok(false));
        out.truncate(1);
        // Bad inputs.
        let same = ContactFilter {
            pairs: Some(vec![(ids[0], ids[0])]),
        };
        assert!(matches!(
            scene.find_contacts(0.5, &same, &mut out),
            Err(SceneError::InvalidInput(_))
        ));
        for margin in [-1.0, f64::NAN] {
            assert!(matches!(
                scene.find_contacts(margin, &without, &mut out),
                Err(SceneError::InvalidInput(_))
            ));
        }
        let gone = ids[5];
        scene.remove_mesh(gone).unwrap();
        let unknown = ContactFilter {
            pairs: Some(vec![(ids[0], ids[2]), (gone, ids[0])]),
        };
        assert_eq!(
            scene.find_contacts(0.5, &unknown, &mut out),
            Err(SceneError::UnknownMesh)
        );
        assert_eq!(out, keep());
    }

    #[test]
    fn find_contacts_truncates_after_a_midway_failure() {
        let (mut scene, ids) = scene();
        let keep = || {
            vec![Contact {
                a: ids[0],
                b: ids[1],
                intersecting: false,
                distance: 9.0,
                triangles_a: vec![],
                triangles_b: vec![],
            }]
        };
        // The second pair to reach the narrow phase fails, after the first
        // was appended: the error leaves `out` as it was.
        let mut out = keep();
        crate::scene::fault::fail_contact_on(2);
        assert_eq!(
            scene.find_contacts(0.5, &ContactFilter::default(), &mut out),
            Err(SceneError::Store(StoreError::OutOfMemory))
        );
        assert_eq!(out, keep());
        // The hook fires once: the search works again afterwards.
        let n = scene
            .find_contacts(0.5, &ContactFilter::default(), &mut out)
            .unwrap();
        assert!(n >= 2, "{n}");
        assert_eq!(out.len(), 1 + n);
    }

    #[test]
    fn collision_registry_cleans_up_on_remove_and_compact() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(cube([0.0; 3], 1.0), MeshOptions::default())
            .unwrap();
        let b = scene
            .add_mesh(cube([2.0, 0.0, 0.0], 1.0), MeshOptions::default())
            .unwrap();
        let c = scene
            .add_mesh(cube([4.0, 0.0, 0.0], 1.0), MeshOptions::default())
            .unwrap();

        let rule = CollisionRule {
            mode: CollisionMode::Rigid,
            ..CollisionRule::default()
        };
        scene.set_collision_rule(b, a, rule).unwrap();
        scene.set_collision_rule(a, c, rule).unwrap();
        assert_eq!(scene.collision.pair_rules.len(), 2);
        assert_eq!(ordered_pair(b, a), ordered_pair(a, b));

        scene.remove_mesh(a).unwrap();
        assert!(scene.collision.pair_rules.is_empty());
        assert!(scene.collision.rest_positions.is_empty());

        let invalid = CollisionRule {
            recovery_rate: 1.5,
            ..CollisionRule::default()
        };
        assert!(matches!(
            scene.set_default_collision_rule(invalid),
            Err(SceneError::InvalidInput(_))
        ));
    }
}
