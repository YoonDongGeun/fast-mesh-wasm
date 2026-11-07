//! Distance fields between meshes of the scene, and the colours derived
//! from them (spec §6, §11).
//!
//! A field's values live in a derived store channel `field:<name>` on mesh
//! A (`on`), one `f32` per vertex slot: the signed distance to the nearest
//! of its target meshes (`to`) as placed in the world, `+∞` beyond
//! `max_distance`, NaN for dead slots and for alive vertices no alive
//! triangle uses. "Nearest" is the smallest signed value: the nearest
//! surface outside every target, or the deepest penetration. The field also
//! keeps, per vertex slot, the index in `to` of the target that value came
//! from (ties to the lowest index; none for `+∞` and NaN). With one target,
//! the values are the distances to that mesh.
//!
//! **What an update recomputes** (§11.6). Each field has a change
//! accumulator on A and on every target (a [`FieldHook`], fed like a
//! subscriber's report). Everything is recomputed when any says "everything"
//! (`add_mesh`, `compact`, a `Corrupted` step), when `max_distance` is
//! infinite, or on the first update. Otherwise a target is *moved* when A's
//! placement changed, its own placement changed, or it was edited, and:
//!
//! - for each moved target B: the vertices of A's triangles within
//!   `pad(max_distance)` of B (`max_distance` plus a relative `1e-6` and
//!   `8·f32::EPSILON` times the coordinate magnitude of the pair), plus the
//!   vertices whose finite value came from B (so they can return to far, or
//!   go to another target);
//! - after an edit of A: its added, moved and removed vertices, plus, when
//!   its triangles changed, the vertices whose "used by a triangle" state
//!   flipped.
//!
//! When the near search for a target would list more than
//! [`NEAR_PAIRS_PER_SLOT`] pairs of triangles per vertex slot of A, it is
//! abandoned and every vertex is recomputed instead: at a wide
//! `max_distance` each triangle of A is near hundreds of B's, and listing the
//! pairs costs far more than the recompute. Recomputing a vertex outside the
//! near set gives it the value it already holds, so the values, colours and
//! reports are the same.
//!
//! A vertex's value depends only on its own position and on the targets.
//! Outside a moved target's new near set, that target's value is `+∞`, which
//! never wins; so a vertex whose value came from a target that did not move
//! keeps it (a moved target that tied with it had a higher index, or it would
//! have won). Rebuilding a target's sign data after an edit can flip signs
//! only of its finite values, which its near set covers. This relies on undo
//! reporting revived slots as added (the store's `ChangeSet::merge` reports a
//! reused slot as moved).
//!
//! **Removing a target mesh** removes it from the targets of every field
//! that has others (such a field is recomputed in full on its next update),
//! and removes the fields left without targets.
//!
//! **Several fields in one update** are prepared one after the other (BVHs,
//! sign data), then computed without writing, in parallel across fields with
//! the `parallel` feature when there is enough work, then applied in field
//! order. Values, colours and reports are the same, bit for bit, as when
//! each field is updated alone in turn.
//!
//! **Stepped updates** ([`Scene::field_step`]) run the same parts in steps:
//! the due Eager fields' BVHs are prepared when the unit opens; then, one
//! field after the other, the targets' sign data are built
//! ([`SignBuild`], a bounded number of elements per step), the near
//! searches run ([`NearSearch`], whole parts per step), the candidates are
//! picked and their values computed a few at a time; the fields are
//! applied together after the last. No value, colour, report or hook is
//! written before that, so dropping the unit leaves none behind (the BVHs
//! the unit brought up to date and the sign data it finished stay so, as
//! after a query; a sign data build it left unfinished is kept with its
//! field and resumed by a later unit while its mesh and BVH are
//! unchanged). Each part gives the same bits in steps as in one call, so a
//! stepped update equals a one-call one.

use fast_mesh_bvh::Bvh;
use fast_mesh_fields::{
    ColorRule, FieldsError, SignBuild, SignData, SignMode, SignStatus, Target, color_of,
    curvatures, signed_distances, signed_distances_multi, thicknesses,
};
use fast_mesh_query::{NearSearch, Rigid};
use fast_mesh_store::{ChangeSet, ChannelDesc, Mesh, StoreError, TriangleId, VertexId, par};

use crate::baked::baked_mesh;

use crate::query::round_up;
use crate::scene::{Entry, FIELD_PREFIX};
use crate::{FieldId, MeshId, Result, Scene, SceneError, UpdateMode};

/// A signed distance field to register with
/// [`Scene::add_distance_field`](crate::Scene::add_distance_field).
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceField {
    /// The field's name, unique among the fields on mesh `on`.
    pub name: String,
    /// The mesh whose vertices carry the values (mesh A).
    pub on: MeshId,
    /// The meshes the distances are measured to: at least one, all
    /// different, and not `on`. A vertex's value is the smallest over them,
    /// ties going to the lowest index (see
    /// [`Scene::field_nearest`](crate::Scene::field_nearest)).
    pub to: Vec<MeshId>,
    /// The largest distance computed (inclusive); vertices farther away hold
    /// `+∞`. Zero or greater, may be infinite.
    pub max_distance: f64,
    /// How the sign is decided on each mesh of `to`.
    pub sign: SignMode,
}

impl DistanceField {
    /// A field on `on` measured to the single mesh `to`.
    pub fn new(
        name: impl Into<String>,
        on: MeshId,
        to: MeshId,
        max_distance: f64,
        sign: SignMode,
    ) -> DistanceField {
        DistanceField {
            name: name.into(),
            on,
            to: vec![to],
            max_distance,
            sign,
        }
    }
}

/// A per-vertex curvature field to register with
/// [`Scene::add_curvature_field`].
///
/// The value at a vertex is its signed mean curvature (the cotangent Laplace
/// estimate of [`fast_mesh_fields::curvatures`]): positive on domes (convex
/// as seen from the normal side, about `1 / r` on a sphere of radius `r`
/// with outward normals), negative in bowls and dents, zero on flat spots;
/// a saddle takes the sign of its dominant bend. Units are 1/world-unit:
/// the placement's scale is applied (a mesh scaled by `s` reads `H / s`),
/// so scaled meshes read scaled. Dead slots and alive vertices no alive
/// triangle uses read NaN. Like a thickness field, the values live in a
/// derived store channel `field:<name>` on `on` and colour through
/// [`Scene::set_coloring`] unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct CurvatureField {
    /// The field's name, unique among the fields on `on`.
    pub name: String,
    /// The mesh whose vertices carry the values.
    pub on: MeshId,
}

/// A per-vertex wall-thickness field to register with
/// [`Scene::add_thickness_field`].
///
/// The value at a vertex is the distance to the opposite surface along the
/// vertex's smooth normal (the nearest ray hit in either normal direction),
/// in world units: the placement's scale is applied, so scaled meshes read
/// scaled. Hits beyond `max_distance` read `+∞`; vertices whose rays escape
/// in both directions (open sheets, the rims of open meshes), dead slots and
/// alive vertices no alive triangle uses read NaN. Like a distance field,
/// the values live in a derived store channel `field:<name>` on `on` and
/// colour through [`Scene::set_coloring`] unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct ThicknessField {
    /// The field's name, unique among the fields on `on`.
    pub name: String,
    /// The mesh whose vertices carry the values.
    pub on: MeshId,
    /// The largest thickness kept (inclusive); thicker vertices hold `+∞`.
    /// Zero or greater, may be infinite.
    pub max_distance: f64,
}

/// The state of a field, from
/// [`Scene::field_status`](crate::Scene::field_status).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldStatus {
    /// True when the values match the meshes and placements as they are now.
    pub current: bool,
    /// What the latest sign data builds found out about the meshes of `to`,
    /// together: `closed` and `consistent_orientation` when every target is,
    /// the counts and the signed volume summed over the targets, and the
    /// first target's `mode` (every target is built in the requested mode,
    /// so they share it unless `Auto` resolved differently). See
    /// [`Scene::field_target_status`](crate::Scene::field_target_status) for
    /// each target.
    pub sign: SignStatus,
}

/// How a mesh is coloured from one of its fields, for
/// [`Scene::set_coloring`](crate::Scene::set_coloring).
#[derive(Debug, Clone, PartialEq)]
pub struct Coloring {
    /// The field, which must be on the coloured mesh.
    pub field: FieldId,
    /// How values become colours.
    pub rule: ColorRule,
}

/// The near-set margin for `max_distance` `x` (§11.6), for meshes of
/// coordinate magnitude `m` (see [`Scene::pair_magnitude`]): slightly
/// larger, so that the pair test in A's space never drops a vertex the
/// `f32` kernel in B's space keeps.
fn pad(x: f64, m: f64) -> f64 {
    x * (1.0 + 1e-6) + 8.0 * f64::from(f32::EPSILON) * m
}

/// The largest absolute coordinate of a box's corners.
fn box_magnitude(min: [f32; 3], max: [f32; 3]) -> f64 {
    min.iter()
        .chain(&max)
        .map(|c| f64::from(c.abs()))
        .fold(0.0, f64::max)
}

fn oom() -> SceneError {
    SceneError::Store(StoreError::OutOfMemory)
}

/// Work, summed over the fields due in one update (see [`FieldJob::work`]),
/// from which those fields are computed in parallel, one field per task,
/// with the `parallel` feature. Each field's kernel call is also parallel on
/// its own above its own threshold.
///
/// Measured on the i5-13600K with six terrains in a row, each overlapping
/// the next, a field for every ordered pair, dragging the middle one (ten
/// fields due per pose): at 968 triangles per mesh (work about 10k) parallel
/// is 1.3 times slower, at 1,352 (about 14k) 1.7 times faster, and at 30k
/// 2.4 times faster.
const PAR_MIN_FIELD_WORK: usize = 12 * 1024;

/// Near pairs of triangles per vertex slot of A beyond which a field's near
/// search is abandoned for recomputing every vertex (see the module docs).
///
/// Measured on the i5-13600K with two 1M-triangle terrains, B flipped over
/// A: a near pair costs about 70 ns to find, and recomputing every vertex
/// about 450 ms (0.9 µs a slot) at `max_distance` 8, where the search finds
/// hundreds of pairs per slot and took 14 s. At one pair per slot, an
/// abandoned search wastes about 8% of the recompute that replaces it. The
/// contact drag (`max_distance` 0.08) and the many-mesh drags (0.5 between
/// 30k-triangle terrains) stay under the limit and keep the near search;
/// a quarter of a pair per slot already sends the latter to the slower
/// recompute.
const NEAR_PAIRS_PER_SLOT: usize = 1;

/// Sign data elements (triangle, vertex or corner slots of a pass of the
/// targets' adjacency, incidence or sign data) one point of a
/// [`FieldBudget::Points`] budget stands for, in a step that builds sign
/// data.
///
/// Measured on the i5-13600K, single-threaded, 1M triangles: a pseudonormal
/// sign data build with its adjacency and incidence is about 7M elements in
/// 220 ms (about 30 ns each), and a wide value about 1 µs, so 32 elements
/// take about as long as a point.
const SIGN_ELEMS_PER_POINT: usize = 32;

/// Sign data elements of a build that one moved vertex of a refit stands for,
/// in a step (see [`SIGN_ELEMS_PER_POINT`]): measured, a refit costs a few µs
/// per moved vertex (its triangles' vertex and edge normals) where an element
/// of a build costs about 30 ns. A larger edit builds in steps instead.
const SIGN_ELEMS_PER_REFIT_VERTEX: usize = 4;

/// Triangle pairs a near search tests per point of a
/// [`FieldBudget::Points`] budget, in a step that searches: a test costs
/// about 70 ns where most pairs are near (see [`NEAR_PAIRS_PER_SLOT`]), so
/// 16 take about as long as a point. A step searches whole parts of the
/// search, so it may test more.
const NEAR_TESTS_PER_POINT: usize = 16;

/// The share of [`Scene::set_field_interaction`]'s budget that goes to a
/// sweep over the vertices outside the near set, a run of consecutive slots per
/// update, so that the colours away from the contact keep following the motion.
/// Consecutive slots are mostly neighbours on the surface: a run reads as a
/// band passing over the mesh, where scattered vertices read as speckle.
const LIVE_SWEEP_SHARE: usize = 8;

/// The share of [`Scene::set_field_interaction`]'s budget for the vertices an
/// edit moved, and again for those around an edit of a target.
const LIVE_EDIT_SHARE: usize = 4;

/// The first band of the interaction tier's near vertices: this fraction of
/// the diagonal of A's box.
const LIVE_FIRST_MARGIN: f64 = 1.0 / 256.0;

/// The interaction tier's band grows by this factor after an update under
/// half of the budget (over it, it is the budget's last vertex).
const LIVE_GROW: f64 = 1.25;

/// A due field whose meshes are prepared: what its update recomputes.
struct FieldJob {
    /// The field's index.
    i: usize,
    /// Every vertex is recomputed.
    full: bool,
    /// An update of the interaction tier (see [`Scene::assemble_live_pick`]):
    /// the other vertices keep their values.
    live: bool,
    /// In such an update, the field's `LiveState::travel` after this update.
    live_travel: f64,
    /// Per target: it moved (see the module docs).
    moved: Vec<bool>,
    /// A's triangles changed since the last update.
    topology: bool,
    /// A's merged changes since the last update (`None`: everything).
    changes: Option<ChangeSet>,
    /// Per target: the margin of its near set, when it is needed.
    near_margins: Vec<Option<f64>>,
    /// Per target: from its local space into A's, as the pair queries take
    /// it (unused when `scaled`).
    b_to_a: Vec<Rigid>,
    /// The most near pairs the near search may list (see
    /// [`NEAR_PAIRS_PER_SLOT`]), decided here on the caller's thread.
    near_limit: usize,
    /// An estimate of the update's cost, in elements: every vertex slot of
    /// A times the targets when all are recomputed; otherwise the known
    /// candidates times the targets, plus A's triangle count for each near
    /// search (which usually costs the most) whose target's root box is
    /// within the margin of A's.
    work: usize,
    /// A's near triangles from a shared pair query with the reciprocal
    /// field (single-target, same `max_distance`, both needing a near set):
    /// used instead of running a second query.
    shared_near: Option<Vec<TriangleId>>,
    /// A scaled placement is involved (A's or a target's): candidates are
    /// all vertices and [`Scene::compute_values_scaled`] measures them
    /// through the placements (uniform scales by arithmetic, non-uniform
    /// targets on baked copies in `temp_targets`).
    scaled: bool,
    /// Frame distances times this are world distances: A's uniform scale
    /// when the frame is A's local space, 1 when it is the world.
    frame_scale: f64,
    /// Per target: the baked copy in the frame with its BVH and sign data,
    /// for non-uniformly scaled targets (`None`: the real target is used).
    temp_targets: Vec<Option<TempTarget>>,
}

/// A non-uniformly scaled target baked into the update's frame (see
/// [`FieldJob::temp_targets`]).
struct TempTarget {
    mesh: Mesh,
    bvh: Bvh,
    sign: SignData,
}

impl FieldJob {
    /// Some target moved.
    fn geometric(&self) -> bool {
        self.moved.contains(&true)
    }
}

/// The new state of a field, computed without writing anything.
struct FieldPlan {
    /// In the interaction tier: the band of the next update.
    margin: f64,
    /// The vertices recomputed, ascending.
    candidates: Vec<VertexId>,
    /// The new value of every candidate, in candidate order.
    values: Vec<f32>,
    /// The target each new value came from, in candidate order.
    nearest: Vec<u32>,
    /// The candidates whose value bits changed, ascending.
    changed: Vec<VertexId>,
    /// The new "alive and used by a triangle" flags, when recomputed.
    referenced: Option<Vec<bool>>,
    /// The vertices with finite values, ascending.
    finite: Vec<VertexId>,
    /// Points the kernel ran on.
    #[cfg_attr(not(test), allow(dead_code))]
    recomputed: usize,
    /// The near search passed its limit, and every vertex was recomputed.
    #[cfg_attr(not(test), allow(dead_code))]
    aborted: bool,
}

/// The vertices a field's update recomputes, chosen without writing
/// anything (see the module docs).
#[derive(Default)]
struct Pick {
    /// The vertices to recompute, ascending.
    candidates: Vec<VertexId>,
    /// Every vertex is recomputed.
    all: bool,
    /// The near search passed its limit.
    aborted: bool,
    /// In the interaction tier: the band of the next update.
    margin: f64,
}

/// What is left of one step's budget (see
/// [`Scene::field_step`](crate::Scene::field_step)).
struct Step {
    /// The step's budget, in points.
    points: usize,
    /// Values left to compute.
    left: usize,
    /// A sign data build, a near search or a walk over every vertex slot
    /// ran: the step does nothing else.
    worked: bool,
}

impl Step {
    /// Claims the step for one long piece of work: `false` when the step
    /// already did any work.
    fn begin_heavy(&mut self) -> bool {
        if self.unbounded() {
            return true;
        }
        if self.worked || self.left < self.points {
            return false;
        }
        self.worked = true;
        true
    }

    /// An unbounded budget (`usize::MAX` points): the step runs the unit to
    /// the end, long pieces of work included.
    fn unbounded(&self) -> bool {
        self.points == usize::MAX
    }
}

/// How much one [`Scene::field_step`](crate::Scene::field_step) computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FieldBudget {
    /// Up to this many candidate vertices (at least one), counted across
    /// the unit's fields in field order.
    Points(usize),
}

/// Where the field step unit stands after
/// [`Scene::field_step`](crate::Scene::field_step).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StepState {
    /// No Eager field was due, so no unit was opened.
    Idle,
    /// A unit is open, with candidates left to compute.
    Pending,
    /// This step computed the unit's last candidates and applied it.
    Done,
}

/// Where a field of an open step unit stands. A field goes through every
/// stage in order; those with nothing to do pass at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// Building the sign data of target `k` and the later ones that are not
    /// current.
    Sign(usize),
    /// Searching A's triangles near target `k` and the later moved ones.
    Near(usize),
    /// Choosing the candidates from the near triangles.
    Pick,
    /// Computing the candidates' values.
    Values,
    /// Computing what changes with them (the field's plan).
    Finish,
}

/// A field of an open step unit.
struct UnitField {
    job: FieldJob,
    stage: Stage,
    /// The near search under way, in [`Stage::Near`].
    search: Option<NearSearch>,
    /// A's triangles near the moved targets searched so far.
    near: Vec<TriangleId>,
    /// Every vertex is recomputed (a full update, or a near search passed
    /// its limit).
    all: bool,
    /// A near search passed its limit.
    aborted: bool,
    /// The candidates, from [`Stage::Pick`] on; taken by `plan` once every
    /// value is computed.
    pick: Pick,
    /// The values of the candidates computed so far, in candidate order.
    values: Vec<f32>,
    /// The targets those values came from.
    nearest: Vec<u32>,
    /// Points the kernel ran on so far.
    recomputed: usize,
    /// The field's new state, once every value is computed; stays `None`
    /// for a field whose computation failed, which is then not applied.
    plan: Option<FieldPlan>,
}

/// The fields of a step unit (see
/// [`Scene::field_step`](crate::Scene::field_step)), stepped as one unit:
/// the Eager fields due at one state of the scene, or one field. It holds
/// field indices and computed values only, so any change to the scene must
/// drop it first.
pub(crate) struct FieldUnit {
    /// The one field the unit is for, when opened by
    /// [`Scene::field_step_field`](crate::Scene::field_step_field).
    only: Option<FieldId>,
    fields: Vec<UnitField>,
    /// The first field not finished yet.
    next: usize,
    /// Every field is finished and the unit applies on the next step (a
    /// large unit applies in a step of its own).
    finished: bool,
    /// The error of the first field, in field order, that failed (when the
    /// unit opened or since), with that field's index.
    error: Option<(usize, SceneError)>,
}

impl FieldUnit {
    /// Records the error of field `i`, keeping the first in field order.
    fn fail(&mut self, i: usize, e: SceneError) {
        if self.error.as_ref().is_none_or(|(j, _)| i < *j) {
            self.error = Some((i, e));
        }
    }
}

/// What happened to one mesh of a field since the field's last update.
#[derive(Debug, Clone)]
pub(crate) struct FieldHook {
    pub(crate) field: FieldId,
    /// Merged changes; `None` means "everything".
    pub(crate) changes: Option<ChangeSet>,
    pub(crate) placement_changed: bool,
    /// On a target: edits were reset from `changes` while the target was baked
    /// (see `FieldJob::temp_targets`), which its sign data never saw. They
    /// cannot be refitted from `changes` then: the next update builds them.
    pub(crate) sign_missed_edits: bool,
}

impl FieldHook {
    fn new(field: FieldId) -> FieldHook {
        FieldHook {
            field,
            changes: Some(ChangeSet::default()),
            placement_changed: false,
            sign_missed_edits: false,
        }
    }

    /// The changes to refit the target's sign data from, when they hold every
    /// edit since those were last current.
    fn refit_changes(&self) -> Option<&ChangeSet> {
        self.changes.as_ref().filter(|_| !self.sign_missed_edits)
    }

    fn is_clean(&self) -> bool {
        !self.placement_changed && self.changes.as_ref().is_some_and(ChangeSet::is_empty)
    }
}

/// A mesh's colouring with its colours, one per vertex slot.
pub(crate) struct ColorState {
    coloring: Coloring,
    /// The field's channel on the mesh.
    channel: String,
    colors: Vec<[u8; 4]>,
}

impl ColorState {
    /// Recolours the vertices in `changed` (ascending) from `values`, and
    /// colours slots added since the last call. Returns the recoloured ids,
    /// ascending without duplicates.
    fn update(&mut self, values: &[f32], changed: &[VertexId]) -> Vec<VertexId> {
        let rule = &self.coloring.rule;
        let old_len = self.colors.len();
        let mut dirty = Vec::with_capacity(changed.len());
        for &v in changed {
            if (v as usize) < old_len {
                self.colors[v as usize] = color_of(rule, values[v as usize]);
                dirty.push(v);
            }
        }
        for (v, &x) in values.iter().enumerate().skip(old_len) {
            self.colors.push(color_of(rule, x));
            dirty.push(v as VertexId);
        }
        dirty
    }
}

/// A registered distance field.
pub(crate) struct Field {
    pub(crate) id: FieldId,
    pub(crate) desc: DistanceField,
    /// `field:<name>`, the channel on mesh `on`.
    channel: String,
    /// Each target's sign data, rebuilt when not current; `None` until first
    /// built (see [`Scene::add_distance_field_deferred`]).
    sign: Vec<Option<SignData>>,
    /// Per target: a sign data build a step unit left unfinished, resumed
    /// while its mesh and BVH are unchanged.
    sign_builds: Vec<Option<SignBuild>>,
    /// Applied at least once: [`Scene::withdraw_field`] no longer applies.
    computed: bool,
    /// Per vertex slot of A as of the last update: the index of the target
    /// its value came from, `u32::MAX` for none. Shorter than the slots
    /// when slots were added since; remapped with A by a compaction.
    pub(crate) nearest: Vec<u32>,
    /// Not computed yet: the next update computes every vertex.
    fresh: bool,
    /// The vertices with finite values, ascending.
    finite: Vec<VertexId>,
    /// Per vertex slot of A at the last update: alive and used by a triangle.
    referenced: Vec<bool>,
    /// What the interaction tier keeps (see [`Scene::set_field_interaction`]).
    live: LiveState,
}

/// A field's state in the interaction tier.
#[derive(Default)]
struct LiveState {
    /// An update of the tier left values out of date: the end of the
    /// interaction recomputes every vertex.
    stale: bool,
    /// The band of the near vertices, in distance units; 0 until the first
    /// update.
    margin: f64,
    /// The next vertex slot of the sweep (see [`LIVE_SWEEP_SHARE`]).
    sweep: u32,
    /// How far, at most, any vertex of A has moved relative to the targets
    /// over the tier's updates (the sum of their moves).
    travel: f64,
    /// Per vertex slot of A: `travel` when its value was last computed.
    stamps: Vec<f32>,
    /// Per target: the `b_to_a` of the last update.
    poses: Vec<Rigid>,
}

impl Field {
    /// What the latest sign data build of target `k` found out, or the
    /// placeholder of [`Scene::field_target_status`] before the first.
    fn sign_status(&self, k: usize) -> SignStatus {
        self.sign[k].as_ref().map_or(
            SignStatus {
                mode: match self.desc.sign {
                    SignMode::Winding => SignMode::Winding,
                    SignMode::Pseudonormal | SignMode::Auto => SignMode::Pseudonormal,
                },
                closed: false,
                consistent_orientation: false,
                non_manifold_vertices: 0,
                degenerate_triangles: 0,
                signed_volume: 0.0,
            },
            SignData::status,
        )
    }
}

/// A registered thickness field: its values are recomputed in full on every
/// update (no targets, no sign data, no near sets), through
/// [`Scene::update_thick_due`].
pub(crate) struct ThickField {
    pub(crate) id: FieldId,
    pub(crate) desc: ThicknessField,
    /// `field:<name>`, the channel on mesh `on`.
    channel: String,
    /// Applied at least once: [`Scene::withdraw_field`] no longer applies.
    computed: bool,
    /// Not computed yet: the next update recomputes every vertex.
    fresh: bool,
}

/// A registered curvature field: its values are recomputed in full on every
/// update (no targets, no sign data, no near sets), through
/// [`Scene::update_curv_due`].
pub(crate) struct CurvField {
    pub(crate) id: FieldId,
    pub(crate) desc: CurvatureField,
    /// `field:<name>`, the channel on mesh `on`.
    channel: String,
    /// Applied at least once: [`Scene::withdraw_field`] no longer applies.
    computed: bool,
    /// Not computed yet: the next update recomputes every vertex.
    fresh: bool,
}

/// Which registry a field id lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldRef {
    Distance(usize),
    Thickness(usize),
    Curvature(usize),
}

impl Entry {
    fn hook(&self, id: FieldId) -> Option<&FieldHook> {
        self.hooks.iter().find(|h| h.field == id)
    }

    /// Empties the field's hook. With `baked` (a target its sign data were not
    /// used for in this update), it remembers any edits those data missed.
    fn reset_hook(&mut self, id: FieldId, baked: bool) {
        if let Some(h) = self.hooks.iter_mut().find(|h| h.field == id) {
            let missed =
                baked && (h.sign_missed_edits || h.changes.as_ref().is_none_or(|c| !c.is_empty()));
            *h = FieldHook::new(id);
            h.sign_missed_edits = missed;
        }
    }

    /// Colours vertex slots added since the colours were last sized, and the
    /// reused slots among `added` (the store resets their field values to
    /// NaN), from the field's channel (NaN for slots the field has not
    /// computed), and reports them: the colours always cover every vertex
    /// slot, and the channel never changes at an unreported vertex.
    pub(crate) fn grow_colors(&mut self, added: &[VertexId]) {
        let slots = self.mesh.vertex_slots();
        let Some(cs) = &mut self.coloring else {
            return;
        };
        let old = cs.colors.len();
        let rule = &cs.coloring.rule;
        let values = self.mesh.channel(&cs.channel).unwrap_or(&[]);
        let color = |v: usize| values.get(v).map_or(rule.nan, |&x| color_of(rule, x));
        let mut dirty: Vec<VertexId> = added
            .iter()
            .copied()
            .filter(|&v| (v as usize) < old)
            .collect();
        for &v in &dirty {
            cs.colors[v as usize] = color(v as usize);
        }
        if slots > old {
            cs.colors.extend((old..slots).map(color));
            dirty.extend(old as VertexId..slots as VertexId);
        }
        if dirty.is_empty() {
            return;
        }
        dirty.sort_unstable();
        dirty.dedup();
        self.report_colors(Some(&dirty));
    }

    /// Recolours every vertex from the current field values (after a new
    /// colouring or a compaction). Missing values colour as NaN.
    pub(crate) fn recolor_all(&mut self) {
        if let Some(cs) = &mut self.coloring {
            let rule = &cs.coloring.rule;
            cs.colors = match self.mesh.channel(&cs.channel) {
                Some(values) => values.iter().map(|&x| color_of(rule, x)).collect(),
                None => vec![rule.nan; self.mesh.vertex_slots()],
            };
        }
    }
}

impl Scene {
    /// Registers a signed distance field on mesh `desc.on`, measured to the
    /// meshes `desc.to`, and returns its id. Its values go into a derived
    /// channel `field:<name>` on `desc.on`, readable with
    /// [`Scene::field_values`] or through [`Scene::mesh`]; they are not
    /// undoable. With several targets, a vertex's value is the smallest over
    /// them, and [`Scene::field_nearest`] tells which target it came from.
    ///
    /// The field follows the [`UpdateMode`] of `desc.on`: an Eager field is
    /// computed right away and after every edit, placement change, undo,
    /// redo or cancel touching `desc.on` or a target; a Lazy one by
    /// [`Scene::flush`] or [`Scene::update_fields`]; a Manual one only by
    /// [`Scene::update_fields`]. Until then its values are NaN.
    ///
    /// The targets' BVHs are brought up to date under the query rules (a
    /// Lazy one is updated) and their sign data built here, so that
    /// [`Scene::field_status`] can report their topology at once.
    ///
    /// A position-only edit of a target mesh refits its pseudonormal sign
    /// data around the dirty vertices; any other edit rebuilds it over the
    /// whole mesh (O(n), about 150 ms at 1M triangles, more after a topology
    /// edit, which drops the cached adjacency). Placement changes and edits
    /// of `desc.on` do not rebuild it.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`]; [`SceneError::InvalidInput`] when `to` is
    /// empty, holds `on` or a mesh twice, `max_distance` is negative or NaN,
    /// or `on` already has a field of that name; [`SceneError::StaleBvh`]
    /// when a target is a stale Manual mesh; a BVH, field or memory error.
    /// The scene is unchanged on error.
    pub fn add_distance_field(&mut self, desc: DistanceField) -> Result<FieldId> {
        self.drop_field_unit();
        let (id, eager) = self.register_field(desc, true)?;
        if eager {
            let _ = self.update_field(self.fields.len() - 1);
        }
        Ok(id)
    }

    /// Registers a field as [`Scene::add_distance_field`] does, but computes
    /// nothing yet: no sign data are built and no BVH is brought up to date,
    /// whatever the mode of `desc.on`. The field is due until an update
    /// computes it: [`Scene::field_step`] (an Eager field),
    /// [`Scene::field_step_field`], [`Scene::update_fields`],
    /// [`Scene::flush`] (Eager and Lazy) or, unless deferred, the next
    /// change to one of its meshes (Eager). Until then its values are NaN
    /// and [`Scene::field_status`] reports placeholder sign data (see
    /// [`Scene::field_target_status`]). Computed in steps, its values,
    /// colours and reports are the same, bit for bit, as from
    /// [`Scene::add_distance_field`].
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`]; [`SceneError::InvalidInput`] as for
    /// [`Scene::add_distance_field`]; a memory error. The scene is unchanged
    /// on error. Errors of the BVHs and sign data come from the update that
    /// computes the field.
    pub fn add_distance_field_deferred(&mut self, desc: DistanceField) -> Result<FieldId> {
        self.drop_field_unit();
        Ok(self.register_field(desc, false)?.0)
    }

    /// Removes a field that has never been computed, as if it had never
    /// been added: unlike [`Scene::remove_field`] nothing is reported.
    /// Returns `false`, doing nothing, when the field was computed (by any
    /// update) or colours its mesh; [`Scene::remove_field`] then applies.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field.
    pub fn withdraw_field(&mut self, id: FieldId) -> Result<bool> {
        self.drop_field_unit();
        match self.field_ref(id)? {
            FieldRef::Distance(i) => {
                let f = &self.fields[i];
                let colours = self
                    .entry(f.desc.on)
                    .ok()
                    .and_then(|e| e.coloring.as_ref())
                    .is_some_and(|c| c.coloring.field == id);
                if f.computed || colours {
                    return Ok(false);
                }
                let f = self.fields.remove(i);
                for &m in std::iter::once(&f.desc.on).chain(&f.desc.to) {
                    if let Some(e) = self.entries.get_mut(m) {
                        e.hooks.retain(|h| h.field != id);
                    }
                }
                if let Some(ea) = self.entries.get_mut(f.desc.on) {
                    let _ = ea.mesh.remove_channel(&f.channel);
                }
                Ok(true)
            }
            FieldRef::Thickness(t) => {
                let f = &self.thick[t];
                let colours = self
                    .entry(f.desc.on)
                    .ok()
                    .and_then(|e| e.coloring.as_ref())
                    .is_some_and(|c| c.coloring.field == id);
                if f.computed || colours {
                    return Ok(false);
                }
                let f = self.thick.remove(t);
                if let Some(ea) = self.entries.get_mut(f.desc.on) {
                    ea.hooks.retain(|h| h.field != id);
                    let _ = ea.mesh.remove_channel(&f.channel);
                }
                Ok(true)
            }
            FieldRef::Curvature(c) => {
                let f = &self.curv[c];
                let colours = self
                    .entry(f.desc.on)
                    .ok()
                    .and_then(|e| e.coloring.as_ref())
                    .is_some_and(|c| c.coloring.field == id);
                if f.computed || colours {
                    return Ok(false);
                }
                let f = self.curv.remove(c);
                if let Some(ea) = self.entries.get_mut(f.desc.on) {
                    ea.hooks.retain(|h| h.field != id);
                    let _ = ea.mesh.remove_channel(&f.channel);
                }
                Ok(true)
            }
        }
    }

    /// Validates `desc` and registers its field (its channel, hooks and,
    /// with `sign`, its targets' sign data, bringing their BVHs up to date
    /// first). Returns its id and whether its mesh is Eager. All or nothing.
    fn register_field(&mut self, desc: DistanceField, sign: bool) -> Result<(FieldId, bool)> {
        self.entry(desc.on)?;
        for &to in &desc.to {
            self.entry(to)?;
        }
        if desc.to.is_empty() {
            return Err(SceneError::InvalidInput("a field needs a target mesh"));
        }
        if desc.to.contains(&desc.on) {
            return Err(SceneError::InvalidInput(
                "a field needs two different meshes",
            ));
        }
        if (1..desc.to.len()).any(|k| desc.to[..k].contains(&desc.to[k])) {
            return Err(SceneError::InvalidInput(
                "a field's target meshes must all differ",
            ));
        }
        if desc.max_distance.is_nan() || desc.max_distance < 0.0 {
            return Err(SceneError::InvalidInput(
                "max_distance must be zero or greater, and not NaN",
            ));
        }
        if self
            .fields
            .iter()
            .any(|f| f.desc.on == desc.on && f.desc.name == desc.name)
            || self
                .thick
                .iter()
                .any(|f| f.desc.on == desc.on && f.desc.name == desc.name)
            || self
                .curv
                .iter()
                .any(|f| f.desc.on == desc.on && f.desc.name == desc.name)
        {
            return Err(SceneError::InvalidInput(
                "the mesh already has a field of that name",
            ));
        }
        let next = self
            .next_field
            .checked_add(1)
            .ok_or(SceneError::InvalidInput("too many fields"))?;
        let mut signs = Vec::new();
        signs.try_reserve_exact(desc.to.len()).map_err(|_| oom())?;
        let mut sign_builds = Vec::new();
        sign_builds
            .try_reserve_exact(desc.to.len())
            .map_err(|_| oom())?;
        if sign {
            self.prepare(Some(&desc.to))?;
        }
        for &to in &desc.to {
            let eb = self.entries.get_mut(to).ok_or(SceneError::UnknownMesh)?;
            signs.push(if sign {
                Some(SignData::build(&mut eb.mesh, &eb.bvh, desc.sign)?)
            } else {
                None
            });
            sign_builds.push(None);
            eb.hooks.try_reserve(1).map_err(|_| oom())?;
        }
        self.fields.try_reserve(1).map_err(|_| oom())?;
        let channel = format!("{FIELD_PREFIX}{}", desc.name);
        let ea = self
            .entries
            .get_mut(desc.on)
            .ok_or(SceneError::UnknownMesh)?;
        ea.hooks.try_reserve(1).map_err(|_| oom())?;
        ea.mesh
            .add_channel(ChannelDesc::derived(channel.clone(), 1))?;
        // Nothing can fail from here on.
        let id = FieldId(self.next_field);
        self.next_field = next;
        ea.hooks.push(FieldHook::new(id));
        let eager = ea.mode == UpdateMode::Eager;
        for &to in &desc.to {
            if let Some(eb) = self.entries.get_mut(to) {
                eb.hooks.push(FieldHook::new(id));
            }
        }
        self.fields.push(Field {
            id,
            desc,
            channel,
            sign: signs,
            sign_builds,
            computed: false,
            nearest: Vec::new(),
            fresh: true,
            finite: Vec::new(),
            referenced: Vec::new(),
            live: LiveState::default(),
        });
        Ok((id, eager))
    }

    /// Registers a per-vertex wall-thickness field on mesh `desc.on`, and
    /// returns its id. Its values go into a derived channel `field:<name>` on
    /// `desc.on`, readable with [`Scene::field_values`] or through
    /// [`Scene::mesh`]; they are not undoable.
    ///
    /// The field follows the [`UpdateMode`] of `desc.on` like a distance
    /// field: an Eager one is computed right away and after every edit,
    /// placement change, undo, redo or cancel touching `desc.on`; a Lazy one
    /// by [`Scene::flush`] or [`Scene::update_fields`]; a Manual one only by
    /// [`Scene::update_fields`]. Until then its values are NaN. Every update
    /// recomputes every vertex in full (a thickness has no targets to narrow
    /// it down to).
    ///
    /// A non-uniformly scaled placement is baked into a world-frame copy for
    /// the update, so the values are world distances; a uniform scale
    /// multiplies the local values, and a rigid placement changes nothing.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`]; [`SceneError::InvalidInput`] when
    /// `max_distance` is negative or NaN, or `on` already has a field of that
    /// name; [`SceneError::StaleBvh`] when the mesh is a stale Manual mesh; a
    /// BVH, field or memory error. The scene is unchanged on error.
    pub fn add_thickness_field(&mut self, desc: ThicknessField) -> Result<FieldId> {
        self.drop_field_unit();
        let (id, eager) = self.register_thickness(desc)?;
        if eager {
            let _ = self.update_thick_field(self.thick.len() - 1);
        }
        Ok(id)
    }

    /// [`Scene::add_thickness_field`] computing nothing yet: the field is due
    /// until an update computes it ([`Scene::field_step`] (an Eager field),
    /// [`Scene::field_step_field`], [`Scene::update_fields`] or
    /// [`Scene::flush`], unless deferred, the next change to its mesh (an
    /// Eager field)). Until then its values are NaN.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`]; [`SceneError::InvalidInput`] as for
    /// [`Scene::add_thickness_field`]; a memory error. The scene is unchanged
    /// on error. Errors of the BVH come from the update that computes the
    /// field.
    pub fn add_thickness_field_deferred(&mut self, desc: ThicknessField) -> Result<FieldId> {
        self.drop_field_unit();
        Ok(self.register_thickness(desc)?.0)
    }

    /// Validates `desc` and registers its thickness field (its channel and
    /// hook). Returns its id and whether its mesh is Eager. All or nothing.
    fn register_thickness(&mut self, desc: ThicknessField) -> Result<(FieldId, bool)> {
        self.entry(desc.on)?;
        if desc.max_distance.is_nan() || desc.max_distance < 0.0 {
            return Err(SceneError::InvalidInput(
                "max_distance must be zero or greater, and not NaN",
            ));
        }
        if self
            .fields
            .iter()
            .any(|f| f.desc.on == desc.on && f.desc.name == desc.name)
            || self
                .thick
                .iter()
                .any(|f| f.desc.on == desc.on && f.desc.name == desc.name)
            || self
                .curv
                .iter()
                .any(|f| f.desc.on == desc.on && f.desc.name == desc.name)
        {
            return Err(SceneError::InvalidInput(
                "the mesh already has a field of that name",
            ));
        }
        let next = self
            .next_field
            .checked_add(1)
            .ok_or(SceneError::InvalidInput("too many fields"))?;
        let channel = format!("{FIELD_PREFIX}{}", desc.name);
        let ea = self
            .entries
            .get_mut(desc.on)
            .ok_or(SceneError::UnknownMesh)?;
        ea.hooks.try_reserve(1).map_err(|_| oom())?;
        ea.mesh
            .add_channel(ChannelDesc::derived(channel.clone(), 1))?;
        // Nothing can fail from here on.
        let id = FieldId(self.next_field);
        self.next_field = next;
        ea.hooks.push(FieldHook::new(id));
        let eager = ea.mode == UpdateMode::Eager;
        self.thick.try_reserve(1).map_err(|_| oom())?;
        self.thick.push(ThickField {
            id,
            desc,
            channel,
            computed: false,
            fresh: true,
        });
        Ok((id, eager))
    }

    /// Registers a per-vertex curvature field on mesh `desc.on`, and returns
    /// its id. Its values go into a derived channel `field:<name>` on
    /// `desc.on`, readable with [`Scene::field_values`] or through
    /// [`Scene::mesh`]; they are not undoable.
    ///
    /// The field follows the [`UpdateMode`] of `desc.on` like a distance
    /// field: an Eager one is computed right away and after every edit,
    /// placement change, undo, redo or cancel touching `desc.on`; a Lazy one
    /// by [`Scene::flush`] or [`Scene::update_fields`]; a Manual one only by
    /// [`Scene::update_fields`]. Until then its values are NaN. Every update
    /// recomputes every vertex in full (a curvature has no targets to narrow
    /// it down to).
    ///
    /// A non-uniformly scaled placement is baked into a world-frame copy for
    /// the update, so the values are world curvatures; a uniform scale `s`
    /// divides the local values by `s`, and a rigid placement changes
    /// nothing.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`]; [`SceneError::InvalidInput`] when `on`
    /// already has a field of that name; [`SceneError::StaleBvh`] when the
    /// mesh is a stale Manual mesh; a BVH, field or memory error. The scene
    /// is unchanged on error.
    pub fn add_curvature_field(&mut self, desc: CurvatureField) -> Result<FieldId> {
        self.drop_field_unit();
        let (id, eager) = self.register_curvature(desc)?;
        if eager {
            let _ = self.update_curv_field(self.curv.len() - 1);
        }
        Ok(id)
    }

    /// [`Scene::add_curvature_field`] computing nothing yet: the field is due
    /// until an update computes it ([`Scene::field_step`] (an Eager field),
    /// [`Scene::field_step_field`], [`Scene::update_fields`] or
    /// [`Scene::flush`], unless deferred, the next change to its mesh (an
    /// Eager field)). Until then its values are NaN.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`]; [`SceneError::InvalidInput`] as for
    /// [`Scene::add_curvature_field`]; a memory error. The scene is unchanged
    /// on error. Errors of the BVH come from the update that computes the
    /// field.
    pub fn add_curvature_field_deferred(&mut self, desc: CurvatureField) -> Result<FieldId> {
        self.drop_field_unit();
        Ok(self.register_curvature(desc)?.0)
    }

    /// Validates `desc` and registers its curvature field (its channel and
    /// hook). Returns its id and whether its mesh is Eager. All or nothing.
    fn register_curvature(&mut self, desc: CurvatureField) -> Result<(FieldId, bool)> {
        self.entry(desc.on)?;
        if self
            .fields
            .iter()
            .any(|f| f.desc.on == desc.on && f.desc.name == desc.name)
            || self
                .thick
                .iter()
                .any(|f| f.desc.on == desc.on && f.desc.name == desc.name)
            || self
                .curv
                .iter()
                .any(|f| f.desc.on == desc.on && f.desc.name == desc.name)
        {
            return Err(SceneError::InvalidInput(
                "the mesh already has a field of that name",
            ));
        }
        let next = self
            .next_field
            .checked_add(1)
            .ok_or(SceneError::InvalidInput("too many fields"))?;
        let channel = format!("{FIELD_PREFIX}{}", desc.name);
        let ea = self
            .entries
            .get_mut(desc.on)
            .ok_or(SceneError::UnknownMesh)?;
        ea.hooks.try_reserve(1).map_err(|_| oom())?;
        ea.mesh
            .add_channel(ChannelDesc::derived(channel.clone(), 1))?;
        // Nothing can fail from here on.
        let id = FieldId(self.next_field);
        self.next_field = next;
        ea.hooks.push(FieldHook::new(id));
        let eager = ea.mode == UpdateMode::Eager;
        self.curv.try_reserve(1).map_err(|_| oom())?;
        self.curv.push(CurvField {
            id,
            desc,
            channel,
            computed: false,
            fresh: true,
        });
        Ok((id, eager))
    }

    /// Removes a field: its channel leaves its mesh, a colouring by it is
    /// removed (reported as `colors_dirty: None`), and the removal is
    /// reported in the mesh's [`MeshChanges::fields_changed`](crate::MeshChanges::fields_changed).
    /// The id is rejected from then on.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field.
    pub fn remove_field(&mut self, id: FieldId) -> Result<()> {
        self.drop_field_unit();
        match self.field_ref(id)? {
            FieldRef::Distance(i) => self.detach_field(i),
            FieldRef::Thickness(t) => self.detach_thick(t),
            FieldRef::Curvature(c) => self.detach_curv(c),
        }
        Ok(())
    }

    /// Brings every field that is not current up to date, whatever the mode
    /// of its mesh, bringing the BVHs it needs up to date under the query
    /// rules (Lazy ones are updated). Renderers call this (or
    /// [`Scene::flush`]), then read any number of fields and colours.
    ///
    /// # Errors
    /// The first error, for example [`SceneError::StaleBvh`] when a BVH a
    /// field needs is a stale Manual one: B's always, A's when the field
    /// must look for A's triangles near B (after a placement change or an
    /// edit of B). Such a field makes every call (and every
    /// [`Scene::flush`]) return `StaleBvh` until that BVH is brought up to
    /// date with [`Scene::update`]. Every field is still attempted; those
    /// that failed stay stale and keep their pending changes.
    pub fn update_fields(&mut self) -> Result<()> {
        self.drop_field_unit();
        self.run_fields(|_| true, None)
    }

    /// The field's values, one per vertex slot of its mesh, as of its last
    /// update (see [`Scene::field_status`]): signed distances, `+∞` for far,
    /// NaN for dead or unused vertices. A thickness field holds wall
    /// thicknesses instead (`+∞` past `max_distance`, NaN where the rays
    /// escape), and a curvature field signed mean curvatures in 1/world-unit
    /// (NaN for dead or unused vertices). Never updates.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field.
    pub fn field_values(&self, id: FieldId) -> Result<&[f32]> {
        let (on, channel) = match self.field_ref(id)? {
            FieldRef::Distance(i) => {
                let f = &self.fields[i];
                (f.desc.on, f.channel.as_str())
            }
            FieldRef::Thickness(t) => {
                let f = &self.thick[t];
                (f.desc.on, f.channel.as_str())
            }
            FieldRef::Curvature(c) => {
                let f = &self.curv[c];
                (f.desc.on, f.channel.as_str())
            }
        };
        self.entry(on)?
            .mesh
            .channel(channel)
            .ok_or(SceneError::Corrupted)
    }

    /// Per vertex slot of the field's mesh, as of its last update: the index
    /// in [`Scene::field_targets`] of the target its value came from (the
    /// lowest on a tie), or `u32::MAX` for none (a `+∞` or NaN value, or a
    /// slot not computed yet). It can be shorter than the mesh's slots
    /// (slots added since the last update): those slots have none.
    /// Thickness and curvature fields have no targets and report none.
    /// Never updates.
    ///
    /// After a target mesh is removed, and until the field's next update
    /// (at once for an Eager field, unless deferred), the vertices whose
    /// value came from it have none here while their value stays finite, and
    /// the later indices are shifted down to the remaining targets.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field.
    pub fn field_nearest(&self, id: FieldId) -> Result<&[u32]> {
        match self.field_ref(id)? {
            FieldRef::Distance(i) => Ok(&self.fields[i].nearest),
            FieldRef::Thickness(_) => Ok(&[]),
            FieldRef::Curvature(_) => Ok(&[]),
        }
    }

    /// The field's target meshes, in index order. Removing a target mesh
    /// removes it from here, shifting the later ones down. Thickness and
    /// curvature fields have no targets.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field.
    pub fn field_targets(&self, id: FieldId) -> Result<&[MeshId]> {
        match self.field_ref(id)? {
            FieldRef::Distance(i) => Ok(&self.fields[i].desc.to),
            FieldRef::Thickness(_) => Ok(&[]),
            FieldRef::Curvature(_) => Ok(&[]),
        }
    }

    /// What the latest sign data build found out about each of the field's
    /// targets, in index order. Thickness and curvature fields have no
    /// targets. Never updates.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field.
    /// A target whose sign data were never built (a field added with
    /// [`Scene::add_distance_field_deferred`], not computed yet) reports a
    /// placeholder: the requested mode (`Auto` as
    /// [`SignMode::Pseudonormal`]), not closed, not consistently oriented,
    /// and no counts.
    pub fn field_target_status(&self, id: FieldId) -> Result<Vec<SignStatus>> {
        match self.field_ref(id)? {
            FieldRef::Distance(i) => {
                let f = &self.fields[i];
                Ok((0..f.sign.len()).map(|k| f.sign_status(k)).collect())
            }
            FieldRef::Thickness(_) => Ok(Vec::new()),
            FieldRef::Curvature(_) => Ok(Vec::new()),
        }
    }

    /// Whether the field is current, and what the latest sign data builds
    /// found out about its target meshes, together (see
    /// [`FieldStatus::sign`]). Thickness and curvature fields have no sign
    /// data and report the placeholder (neither closed nor consistently
    /// oriented). Never updates.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field.
    pub fn field_status(&self, id: FieldId) -> Result<FieldStatus> {
        match self.field_ref(id)? {
            FieldRef::Distance(i) => {
                let f = &self.fields[i];
                let mut sign = f.sign_status(0);
                for k in 1..f.sign.len() {
                    let s = f.sign_status(k);
                    sign.closed &= s.closed;
                    sign.consistent_orientation &= s.consistent_orientation;
                    sign.non_manifold_vertices += s.non_manifold_vertices;
                    sign.degenerate_triangles += s.degenerate_triangles;
                    sign.signed_volume += s.signed_volume;
                }
                Ok(FieldStatus {
                    current: self.field_current(i),
                    sign,
                })
            }
            FieldRef::Thickness(t) => Ok(FieldStatus {
                current: self.thick_current(t),
                sign: SignStatus {
                    mode: SignMode::Pseudonormal,
                    closed: false,
                    consistent_orientation: false,
                    non_manifold_vertices: 0,
                    degenerate_triangles: 0,
                    signed_volume: 0.0,
                },
            }),
            FieldRef::Curvature(c) => Ok(FieldStatus {
                current: self.curv_current(c),
                sign: SignStatus {
                    mode: SignMode::Pseudonormal,
                    closed: false,
                    consistent_orientation: false,
                    non_manifold_vertices: 0,
                    degenerate_triangles: 0,
                    signed_volume: 0.0,
                },
            }),
        }
    }

    /// Colours `mesh` from one of its fields, or stops colouring it with
    /// `None`. Every vertex is recoloured from the field's current values,
    /// and reported as `colors_dirty: None`. Field updates then recolour the
    /// vertices whose values changed and report them. Colours are not
    /// undoable.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`]; [`SceneError::UnknownField`];
    /// [`SceneError::InvalidInput`] when the field is not on `mesh` or the
    /// rule is invalid (see [`ColorRule::validate`]). The scene is unchanged
    /// on error.
    pub fn set_coloring(&mut self, mesh: MeshId, coloring: Option<Coloring>) -> Result<()> {
        self.drop_field_unit();
        self.entry(mesh)?;
        let state = match coloring {
            None => None,
            Some(c) => {
                let (on, channel) = match self.field_ref(c.field)? {
                    FieldRef::Distance(i) => {
                        let f = &self.fields[i];
                        (f.desc.on, f.channel.clone())
                    }
                    FieldRef::Thickness(t) => {
                        let f = &self.thick[t];
                        (f.desc.on, f.channel.clone())
                    }
                    FieldRef::Curvature(c) => {
                        let f = &self.curv[c];
                        (f.desc.on, f.channel.clone())
                    }
                };
                if on != mesh {
                    return Err(SceneError::InvalidInput("the field is not on this mesh"));
                }
                c.rule.validate().map_err(|e| match e {
                    FieldsError::InvalidInput(m) => SceneError::InvalidInput(m),
                    e => SceneError::Fields(e),
                })?;
                Some(ColorState {
                    coloring: c,
                    channel,
                    colors: Vec::new(),
                })
            }
        };
        let e = self.entries.get_mut(mesh).ok_or(SceneError::UnknownMesh)?;
        let had = e.coloring.is_some();
        e.coloring = state;
        e.recolor_all();
        if had || e.coloring.is_some() {
            e.report_colors(None);
        }
        Ok(())
    }

    /// The mesh's colours, or `None` when the mesh has no colouring. Never
    /// updates.
    ///
    /// The slice always holds exactly one RGBA8 per vertex slot
    /// (`colors(mesh).len() == mesh.vertex_slots()`), so it can be indexed
    /// by vertex id at any time. The colours follow the field's values as of
    /// its last update; slots added since are coloured from the field's
    /// channel (NaN, so [`ColorRule::nan`]) until the field is updated.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id.
    pub fn colors(&self, mesh: MeshId) -> Result<Option<&[[u8; 4]]>> {
        Ok(self
            .entry(mesh)?
            .coloring
            .as_ref()
            .map(|c| c.colors.as_slice()))
    }

    /// The values the mesh's colours are made from, or `None` when the mesh
    /// has no colouring. Never updates.
    ///
    /// A view of the colouring field's channel, not a copy: the same slice as
    /// [`Scene::field_values`] of that field, one `f32` per vertex slot. It
    /// changes only at vertices reported in `colors_dirty`, so a renderer can
    /// mirror it with the colours' dirty ranges and map values to colours per
    /// fragment ([`color_of`] at a vertex gives that vertex's colour).
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id.
    pub fn color_values(&self, mesh: MeshId) -> Result<Option<&[f32]>> {
        let e = self.entry(mesh)?;
        Ok(e.coloring
            .as_ref()
            .map(|c| e.mesh.channel(&c.channel).unwrap_or(&[])))
    }

    /// The mesh's colouring, or `None` when the mesh has no colouring. Never
    /// updates.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id.
    pub fn coloring(&self, mesh: MeshId) -> Result<Option<&Coloring>> {
        Ok(self.entry(mesh)?.coloring.as_ref().map(|c| &c.coloring))
    }

    /// The ids of the fields on `mesh`, in the order they were added. Never
    /// updates.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id.
    pub fn fields_on(&self, mesh: MeshId) -> Result<Vec<FieldId>> {
        self.entry(mesh)?;
        Ok(self
            .fields
            .iter()
            .filter(|f| f.desc.on == mesh)
            .map(|f| f.id)
            .chain(
                self.thick
                    .iter()
                    .filter(|f| f.desc.on == mesh)
                    .map(|f| f.id),
            )
            .chain(self.curv.iter().filter(|f| f.desc.on == mesh).map(|f| f.id))
            .collect())
    }

    /// Changes a field's `max_distance` and sign mode, keeping its id, its
    /// name, its meshes and any colouring by it. The field is recomputed in
    /// full on its next update (at once for an Eager field); until then its
    /// values are stale and [`Scene::field_status`] reports it as not
    /// current, with placeholder sign data when the sign mode changed. A
    /// thickness field has no sign mode (`sign` is ignored for it), and a
    /// curvature field has neither (`max_distance` is ignored for it too:
    /// the call still marks it for recompute).
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field;
    /// [`SceneError::InvalidInput`] when `max_distance` is negative or NaN.
    /// The scene is unchanged on error.
    pub fn reconfigure_field(
        &mut self,
        id: FieldId,
        max_distance: f64,
        sign: SignMode,
    ) -> Result<()> {
        self.drop_field_unit();
        if max_distance.is_nan() || max_distance < 0.0 {
            return Err(SceneError::InvalidInput(
                "max_distance must be zero or greater, and not NaN",
            ));
        }
        match self.field_ref(id)? {
            FieldRef::Distance(i) => {
                {
                    let f = &mut self.fields[i];
                    if f.desc.max_distance == max_distance && f.desc.sign == sign {
                        return Ok(());
                    }
                    f.desc.max_distance = max_distance;
                    if f.desc.sign != sign {
                        f.desc.sign = sign;
                        for s in f.sign.iter_mut() {
                            *s = None;
                        }
                        for b in f.sign_builds.iter_mut() {
                            *b = None;
                        }
                    }
                    f.fresh = true;
                }
                let on = self.fields[i].desc.on;
                let eager = self
                    .entry(on)
                    .map(|e| e.mode == UpdateMode::Eager)
                    .unwrap_or(false);
                if eager {
                    let _ = self.update_field(i);
                }
                Ok(())
            }
            FieldRef::Thickness(t) => {
                {
                    let f = &mut self.thick[t];
                    if f.desc.max_distance == max_distance {
                        return Ok(());
                    }
                    f.desc.max_distance = max_distance;
                    f.fresh = true;
                }
                let on = self.thick[t].desc.on;
                let eager = self
                    .entry(on)
                    .map(|e| e.mode == UpdateMode::Eager)
                    .unwrap_or(false);
                if eager {
                    let _ = self.update_thick_field(t);
                }
                Ok(())
            }
            FieldRef::Curvature(c) => {
                {
                    let f = &mut self.curv[c];
                    f.fresh = true;
                }
                let on = self.curv[c].desc.on;
                let eager = self
                    .entry(on)
                    .map(|e| e.mode == UpdateMode::Eager)
                    .unwrap_or(false);
                if eager {
                    let _ = self.update_curv_field(c);
                }
                Ok(())
            }
        }
    }

    /// The id of the registered field with this [`FieldId::raw`] value, or
    /// `None` when there is none (never added, or removed).
    pub fn field_id(&self, raw: u32) -> Option<FieldId> {
        self.fields
            .iter()
            .map(|f| f.id)
            .chain(self.thick.iter().map(|f| f.id))
            .chain(self.curv.iter().map(|f| f.id))
            .find(|id| id.0 == raw)
    }

    fn field_index(&self, id: FieldId) -> Result<usize> {
        self.fields
            .iter()
            .position(|f| f.id == id)
            .ok_or(SceneError::UnknownField)
    }

    /// Which registry `id` lives in.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field.
    fn field_ref(&self, id: FieldId) -> Result<FieldRef> {
        if let Some(i) = self.fields.iter().position(|f| f.id == id) {
            return Ok(FieldRef::Distance(i));
        }
        if let Some(t) = self.thick.iter().position(|f| f.id == id) {
            return Ok(FieldRef::Thickness(t));
        }
        if let Some(c) = self.curv.iter().position(|f| f.id == id) {
            return Ok(FieldRef::Curvature(c));
        }
        Err(SceneError::UnknownField)
    }

    /// The index of thickness field `id`.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field, or a
    /// field of another kind.
    fn thick_index(&self, id: FieldId) -> Result<usize> {
        self.thick
            .iter()
            .position(|f| f.id == id)
            .ok_or(SceneError::UnknownField)
    }

    /// The index of curvature field `id`.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field, or a
    /// field of another kind.
    fn curv_index(&self, id: FieldId) -> Result<usize> {
        self.curv
            .iter()
            .position(|f| f.id == id)
            .ok_or(SceneError::UnknownField)
    }

    /// Computed at least once, and nothing changed on its mesh or a target
    /// since.
    fn field_current(&self, i: usize) -> bool {
        let f = &self.fields[i];
        let clean = |m: MeshId| {
            self.entry(m)
                .ok()
                .and_then(|e| e.hook(f.id))
                .is_some_and(FieldHook::is_clean)
        };
        let settled = !f.live.stale || self.field_interaction.is_some();
        settled && !f.fresh && clean(f.desc.on) && f.desc.to.iter().all(|&m| clean(m))
    }

    /// Computed at least once, and nothing changed on its mesh since.
    fn thick_current(&self, t: usize) -> bool {
        let f = &self.thick[t];
        let clean = self
            .entry(f.desc.on)
            .ok()
            .and_then(|e| e.hook(f.id))
            .is_some_and(FieldHook::is_clean);
        !f.fresh && clean
    }

    /// Computed at least once, and nothing changed on its mesh since.
    fn curv_current(&self, c: usize) -> bool {
        let f = &self.curv[c];
        let clean = self
            .entry(f.desc.on)
            .ok()
            .and_then(|e| e.hook(f.id))
            .is_some_and(FieldHook::is_clean);
        !f.fresh && clean
    }

    /// Updates the fields that are not current, whose mesh's mode passes
    /// `want`, and (with `touched`) that involve one of those meshes.
    /// Returns the first error; every field is attempted.
    pub(crate) fn run_fields(
        &mut self,
        want: impl Fn(UpdateMode) -> bool,
        touched: Option<&[MeshId]>,
    ) -> Result<()> {
        let due = self.due_fields(&want, touched)?;
        let r = self.update_due(&due);
        // Thickness and curvature fields update separately (they hold no
        // shared unit), but every due field is still attempted.
        let due_thick = self.due_thick_fields(&want, touched)?;
        let r2 = self.update_thick_due(&due_thick);
        let due_curv = self.due_curv_fields(&want, touched)?;
        let r3 = self.update_curv_due(&due_curv);
        r.and(r2).and(r3)
    }

    /// The indices of the thickness fields that are not current, whose
    /// mesh's mode passes `want`, and (with `touched`) that are on one of
    /// those meshes, ascending.
    fn due_thick_fields(
        &self,
        want: impl Fn(UpdateMode) -> bool,
        touched: Option<&[MeshId]>,
    ) -> Result<Vec<usize>> {
        let mut due = Vec::new();
        for t in 0..self.thick.len() {
            let on = self.thick[t].desc.on;
            if touched.is_some_and(|touched| !touched.contains(&on)) {
                continue;
            }
            let Ok(ea) = self.entry(on) else { continue };
            if !want(ea.mode) || self.thick_current(t) {
                continue;
            }
            due.try_reserve(1).map_err(|_| oom())?;
            due.push(t);
        }
        Ok(due)
    }

    /// The indices of the curvature fields that are not current, whose
    /// mesh's mode passes `want`, and (with `touched`) that are on one of
    /// those meshes, ascending.
    fn due_curv_fields(
        &self,
        want: impl Fn(UpdateMode) -> bool,
        touched: Option<&[MeshId]>,
    ) -> Result<Vec<usize>> {
        let mut due = Vec::new();
        for c in 0..self.curv.len() {
            let on = self.curv[c].desc.on;
            if touched.is_some_and(|touched| !touched.contains(&on)) {
                continue;
            }
            let Ok(ea) = self.entry(on) else { continue };
            if !want(ea.mode) || self.curv_current(c) {
                continue;
            }
            due.try_reserve(1).map_err(|_| oom())?;
            due.push(c);
        }
        Ok(due)
    }

    /// Brings curvature field `c` up to date: its BVH under the query rules,
    /// then every vertex recomputed in full. All or nothing: on error
    /// nothing is written and the pending changes stay.
    fn update_curv_field(&mut self, c: usize) -> Result<()> {
        self.update_curv_due(&[c])
    }

    /// Brings the curvature fields at indices `due` (ascending) up to date,
    /// one after the other. A field that fails is left as it was. Returns
    /// the first error; every field is attempted.
    fn update_curv_due(&mut self, due: &[usize]) -> Result<()> {
        let mut first: Result<()> = Ok(());
        for &c in due {
            if let Err(e) = self.compute_curv_field(c)
                && first.is_ok()
            {
                first = Err(e);
            }
        }
        first
    }

    /// Prepares curvature field `c`, computes every vertex through the
    /// placement, and applies the values. Fails, writing nothing, when the
    /// BVH cannot be brought up to date, the kernel fails, or memory runs
    /// out.
    fn compute_curv_field(&mut self, c: usize) -> Result<()> {
        use fast_mesh_bvh::BuildOptions;
        use fast_mesh_store::Mesh;

        let (on, channel) = {
            let f = &self.curv[c];
            (f.desc.on, f.channel.clone())
        };
        // The mesh's own BVH is what the field's staleness tracks; the kernel
        // reads the mesh itself.
        self.prepare(Some(&[on]))?;
        let ea = self.entry(on)?;
        let (placement, origin) = (ea.placement, ea.mesh.origin());
        let slots = ea.mesh.vertex_slots();
        // Where the kernel runs, and what turns its local curvatures into
        // world ones: a rigid placement changes nothing, a uniform scale `s`
        // divides (curvature is 1/length), and anything else runs on a baked
        // world-frame copy.
        let mut scale = 1.0f32;
        let mut baked: Option<(Mesh, Bvh)> = None;
        if placement.is_scaled() {
            if let Some(s) = placement.uniform_scale() {
                scale = s as f32;
            } else {
                baked = Some(baked_mesh(
                    &ea.mesh,
                    |p| {
                        placement.apply([
                            origin[0] + f64::from(p[0]),
                            origin[1] + f64::from(p[1]),
                            origin[2] + f64::from(p[2]),
                        ])
                    },
                    &BuildOptions::default(),
                )?);
            }
        }
        let mut values = Vec::new();
        values.try_reserve_exact(slots).map_err(|_| oom())?;
        values.resize(slots, f32::NAN);
        match &baked {
            Some((mesh, _)) => {
                curvatures(mesh, &mut values)?;
            }
            None => {
                let eb = self.entry(on)?;
                curvatures(&eb.mesh, &mut values)?;
                if scale != 1.0 {
                    let s = f64::from(scale);
                    for v in &mut values {
                        if v.is_finite() {
                            *v = (*v as f64 / s) as f32;
                        }
                    }
                }
            }
        }
        drop(baked);
        // Commit: compare with the channel, then write, recolour and report.
        let id = self.curv[c].id;
        let ea = self.entries.get_mut(on).ok_or(SceneError::UnknownMesh)?;
        let old = ea.mesh.channel(&channel).ok_or(SceneError::Corrupted)?;
        let changed: Vec<VertexId> = (0..slots as VertexId)
            .filter(|&v| {
                old.get(v as usize).map(|o| o.to_bits()) != Some(values[v as usize].to_bits())
            })
            .collect();
        ea.mesh.derived_channel_mut(&channel)?[..slots].copy_from_slice(&values);
        ea.reset_hook(id, false);
        if !changed.is_empty() {
            ea.report_field(id);
        }
        if let Some(cs) = &mut ea.coloring
            && cs.coloring.field == id
        {
            let vals = ea.mesh.channel(&cs.channel).unwrap_or(&[]);
            let dirty = cs.update(vals, &changed);
            ea.report_colors(Some(&dirty));
        }
        let f = &mut self.curv[c];
        f.fresh = false;
        f.computed = true;
        Ok(())
    }

    /// Brings thickness field `t` up to date: its BVH under the query rules,
    /// then every vertex recomputed in full. All or nothing: on error
    /// nothing is written and the pending changes stay.
    fn update_thick_field(&mut self, t: usize) -> Result<()> {
        self.update_thick_due(&[t])
    }

    /// Brings the thickness fields at indices `due` (ascending) up to date,
    /// one after the other. A field that fails is left as it was. Returns
    /// the first error; every field is attempted.
    fn update_thick_due(&mut self, due: &[usize]) -> Result<()> {
        let mut first: Result<()> = Ok(());
        for &t in due {
            if let Err(e) = self.compute_thick_field(t)
                && first.is_ok()
            {
                first = Err(e);
            }
        }
        first
    }

    /// Prepares thickness field `t` (its BVH), computes every vertex through
    /// the placement, and applies the values. Fails, writing nothing, when
    /// the BVH cannot be brought up to date, the kernel fails, or memory
    /// runs out.
    fn compute_thick_field(&mut self, t: usize) -> Result<()> {
        use fast_mesh_bvh::BuildOptions;
        use fast_mesh_store::Mesh;

        let (on, max_distance, channel) = {
            let f = &self.thick[t];
            (f.desc.on, f.desc.max_distance, f.channel.clone())
        };
        // The mesh's own BVH is what the kernel reads.
        self.prepare(Some(&[on]))?;
        let ea = self.entry(on)?;
        let (placement, origin) = (ea.placement, ea.mesh.origin());
        let slots = ea.mesh.vertex_slots();
        // Where the kernel runs, and what turns its local distances into
        // world ones: a rigid placement changes nothing, a uniform scale
        // multiplies, and anything else runs on a baked world-frame copy.
        let mut scale = 1.0f32;
        let mut baked: Option<(Mesh, Bvh)> = None;
        if placement.is_scaled() {
            if let Some(s) = placement.uniform_scale() {
                scale = s as f32;
            } else {
                baked = Some(baked_mesh(
                    &ea.mesh,
                    |p| {
                        placement.apply([
                            origin[0] + f64::from(p[0]),
                            origin[1] + f64::from(p[1]),
                            origin[2] + f64::from(p[2]),
                        ])
                    },
                    &BuildOptions::default(),
                )?);
            }
        }
        let mut values = Vec::new();
        values.try_reserve_exact(slots).map_err(|_| oom())?;
        values.resize(slots, f32::NAN);
        match &baked {
            Some((mesh, bvh)) => {
                thicknesses(mesh, bvh, max_distance as f32, &mut values)?;
            }
            None => {
                let eb = self.entry(on)?;
                let bound = (max_distance / f64::from(scale)) as f32;
                thicknesses(&eb.mesh, &eb.bvh, bound, &mut values)?;
                if scale != 1.0 {
                    let s = f64::from(scale);
                    for v in &mut values {
                        if v.is_finite() {
                            *v = (*v as f64 * s) as f32;
                        }
                    }
                }
            }
        }
        drop(baked);
        // Commit: compare with the channel, then write, recolour and report.
        let id = self.thick[t].id;
        let ea = self.entries.get_mut(on).ok_or(SceneError::UnknownMesh)?;
        let old = ea.mesh.channel(&channel).ok_or(SceneError::Corrupted)?;
        let changed: Vec<VertexId> = (0..slots as VertexId)
            .filter(|&v| {
                old.get(v as usize).map(|o| o.to_bits()) != Some(values[v as usize].to_bits())
            })
            .collect();
        ea.mesh.derived_channel_mut(&channel)?[..slots].copy_from_slice(&values);
        ea.reset_hook(id, false);
        if !changed.is_empty() {
            ea.report_field(id);
        }
        if let Some(cs) = &mut ea.coloring
            && cs.coloring.field == id
        {
            let vals = ea.mesh.channel(&cs.channel).unwrap_or(&[]);
            let dirty = cs.update(vals, &changed);
            ea.report_colors(Some(&dirty));
        }
        let f = &mut self.thick[t];
        f.fresh = false;
        f.computed = true;
        Ok(())
    }

    /// The indices of the fields that are not current, whose mesh's mode
    /// passes `want`, and (with `touched`) that involve one of those
    /// meshes, ascending.
    fn due_fields(
        &self,
        want: impl Fn(UpdateMode) -> bool,
        touched: Option<&[MeshId]>,
    ) -> Result<Vec<usize>> {
        let mut due = Vec::new();
        for i in 0..self.fields.len() {
            let (on, to) = (self.fields[i].desc.on, &self.fields[i].desc.to);
            if touched.is_some_and(|t| !t.contains(&on) && !to.iter().any(|m| t.contains(m))) {
                continue;
            }
            let Ok(ea) = self.entry(on) else { continue };
            if !want(ea.mode) || self.field_current(i) {
                continue;
            }
            due.try_reserve(1).map_err(|_| oom())?;
            due.push(i);
        }
        Ok(due)
    }

    /// Updates the Eager fields involving `touched` (all with `None`).
    /// Failures are swallowed: the field stays stale (§11.2). With
    /// [`Scene::set_defer_fields`] on, does nothing: the fields stay due.
    pub(crate) fn run_eager_fields(&mut self, touched: Option<&[MeshId]>) {
        if self.defer_fields {
            return;
        }
        let _ = self.run_fields(|m| m == UpdateMode::Eager, touched);
    }

    /// Defers Eager field updates: with `on`, the edits, placement changes,
    /// undos, redos, cancels and [`Scene::update`] calls that would update
    /// the Eager fields involving a mesh leave them due instead, for
    /// [`Scene::field_step`] (or [`Scene::update_fields`] or
    /// [`Scene::flush`]) to bring up to date. Registering a field on an
    /// Eager mesh still computes it at once. Leaves an open step unit open.
    ///
    /// Turning it off does not bring fields that are already due up to
    /// date: they wait for the next change involving their meshes, or for
    /// [`Scene::field_step`], [`Scene::update_fields`] or [`Scene::flush`].
    pub fn set_defer_fields(&mut self, on: bool) {
        self.defer_fields = on;
    }

    /// Starts (`Some(points)`) or ends (`None`) an interaction: a gesture
    /// that changes a placement or a mesh many times a second, during which
    /// a distance field too large to recompute per update stays live near
    /// the contact and is made exact when the gesture ends.
    ///
    /// During an interaction, a field with more finite values than `points`
    /// updates at most `points` vertices closest to the targets (no pair
    /// search: a distance changes no faster than the motion, so its last value
    /// and how far the meshes have moved since say which can be close), the
    /// vertices around an edit, and a sweep of an eighth of `points` further
    /// vertices. The others keep their values, so that nothing turns far or
    /// loses its colour; those values are out of date by at most how far the
    /// meshes moved since they were computed (for an edit of a target, by how
    /// far it moved its surface, away from the edit's box), and
    /// [`Scene::field_status`] does not report it. The band of the closest
    /// vertices follows the work from the counts alone, so results are the
    /// same with `parallel` on or off, for any thread count. A field of at
    /// most `points` finite values updates exactly as usual.
    ///
    /// Ending the interaction recomputes every vertex of the fields it left
    /// out of date, at once for Eager meshes unless fields are deferred (see
    /// [`Scene::set_defer_fields`] and [`Scene::field_step`]); the values,
    /// colours and reports are then those of an exact update. Starting one
    /// again before that keeps them on the tier, and a gesture that changed
    /// nothing leaves nothing to recompute. Drops an open step unit when
    /// the setting changes.
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] when `points` is `Some(0)`. The scene
    /// is unchanged on error.
    pub fn set_field_interaction(&mut self, points: Option<usize>) -> Result<()> {
        if points == Some(0) {
            return Err(SceneError::InvalidInput(
                "an interaction's budget must be at least one point",
            ));
        }
        if self.field_interaction == points {
            return Ok(());
        }
        self.drop_field_unit();
        self.field_interaction = points;
        if points.is_none() {
            self.run_eager_fields(None);
        }
        Ok(())
    }

    /// The budget of the interaction under way (see
    /// [`Scene::set_field_interaction`]), if any.
    pub fn field_interaction(&self) -> Option<usize> {
        self.field_interaction
    }

    /// Computes part of the Eager fields that are due, so that a long
    /// update can be spread over several calls (a Worker yields between
    /// them).
    ///
    /// The first call opens a unit: it takes every due field on an Eager
    /// mesh (as an Eager update would) and brings the BVHs each reads up to
    /// date. Every call, the first included, then advances the unit's
    /// fields in field order, writing nothing: for each, the build of its
    /// targets' sign data when they are not current (about `budget` times
    /// 32 elements of the build per call), its near search (whole parts,
    /// about `budget` times 16 triangle pairs per call), the choice of the
    /// vertices it recomputes (see the module docs), and the values of up
    /// to `budget` of those vertices per call. A call that builds sign data
    /// or searches does only that. The call that finishes the last field
    /// applies every field of the unit together and returns
    /// [`StepState::Done`]: values, colours and reports are then the same,
    /// bit for bit, as from [`Scene::update_fields`] at the state the unit
    /// opened at.
    ///
    /// An open unit, including one opened by [`Scene::field_step_field`],
    /// is continued. Any other `&mut self` call on the scene drops it first
    /// (see [`Scene::abandon_field_step`]), except
    /// [`Scene::abandon_field_step`] and [`Scene::set_defer_fields`].
    ///
    /// A unit leaves a trace when it is then abandoned: the BVHs it brought
    /// up to date stay refitted, and the sign data it finished building
    /// stay with their fields, as after any query that needs them. A sign
    /// data build it left unfinished is kept with its field, and a later
    /// unit resumes it while its mesh and BVH are unchanged. No value,
    /// colour, report or pending change depends on whether that happened.
    ///
    /// # Errors
    /// Errors are isolated per field, as in [`Scene::update_fields`]: a
    /// field that fails, when the unit opens (for example
    /// [`SceneError::StaleBvh`] for a stale Manual mesh) or while it is
    /// advanced, is dropped from the unit and stays stale and due, as after
    /// a failed Eager update. The other fields go on, and the call that
    /// applies them returns the first error in field order instead of
    /// [`StepState::Done`]. When no due field can be opened, the first
    /// error is returned and no unit is opened.
    pub fn field_step(&mut self, budget: FieldBudget) -> Result<StepState> {
        self.step_unit(None, budget)
    }

    /// [`Scene::field_step`] for field `id` alone, whatever its mesh's
    /// mode: opens a unit over it when it is not current and no unit for it
    /// is open (dropping another open unit), or continues its open unit.
    /// Returns [`StepState::Idle`] when the field is current. The call that
    /// finishes it applies it, with the same values, colours and reports as
    /// [`Scene::update_fields`] would give it. A thickness or curvature field
    /// holds no incremental state: stepping it computes it in full at once.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field; then
    /// as for [`Scene::field_step`].
    pub fn field_step_field(&mut self, id: FieldId, budget: FieldBudget) -> Result<StepState> {
        self.field_ref(id)?;
        self.step_unit(Some(id), budget)
    }

    /// Steps the open unit, or opens one over the due Eager fields
    /// (`only`: `None`) or over the field `only` (see [`Scene::field_step`]
    /// and [`Scene::field_step_field`]). A thickness or curvature field
    /// stepped alone computes in full at once; with `None`, the due Eager
    /// thickness and curvature fields compute in full alongside the distance
    /// unit (their values, colours and reports are the same as from
    /// [`Scene::update_fields`], applied when the unit applies).
    fn step_unit(&mut self, only: Option<FieldId>, budget: FieldBudget) -> Result<StepState> {
        let FieldBudget::Points(points) = budget;
        // A thickness or curvature field holds no incremental state: compute
        // it in full.
        if let Some(id) = only
            && let Ok(t) = self.thick_index(id)
        {
            self.drop_field_unit();
            if self.thick_current(t) {
                return Ok(StepState::Idle);
            }
            self.update_thick_field(t)?;
            return Ok(StepState::Done);
        }
        if let Some(id) = only
            && let Ok(c) = self.curv_index(id)
        {
            self.drop_field_unit();
            if self.curv_current(c) {
                return Ok(StepState::Idle);
            }
            self.update_curv_field(c)?;
            return Ok(StepState::Done);
        }
        // With no single field, due Eager thickness and curvature fields
        // compute in full first, then the distance unit advances as usual.
        let mut thick_worked = false;
        let mut thick_error: Result<()> = Ok(());
        if only.is_none() {
            match self.due_thick_fields(|m| m == UpdateMode::Eager, None) {
                Ok(due) => {
                    thick_worked = !due.is_empty();
                    thick_error = self.update_thick_due(&due);
                }
                Err(e) => thick_error = Err(e),
            }
            match self.due_curv_fields(|m| m == UpdateMode::Eager, None) {
                Ok(due) => {
                    thick_worked = thick_worked || !due.is_empty();
                    thick_error = thick_error.and(self.update_curv_due(&due));
                }
                Err(e) => thick_error = thick_error.and(Err(e)),
            }
        }
        let open = self
            .field_unit
            .take()
            .filter(|u| only.is_none() || u.only == only);
        let mut unit = match open {
            Some(unit) => unit,
            None => match self.open_field_unit(only)? {
                Some(unit) => unit,
                None => {
                    // No distance unit: the step did what the thickness and
                    // curvature fields needed, if anything.
                    thick_error?;
                    return Ok(if thick_worked {
                        StepState::Done
                    } else {
                        StepState::Idle
                    });
                }
            },
        };
        if !unit.finished {
            self.advance_field_unit(&mut unit, points.max(1));
            if unit.next < unit.fields.len() {
                self.field_unit = Some(unit);
                return Ok(StepState::Pending);
            }
            // Applying writes every candidate's value and colour: when there
            // are more than a step's worth, in a step of its own.
            let written: usize = unit
                .fields
                .iter()
                .filter_map(|f| f.plan.as_ref())
                .map(|p| p.candidates.len())
                .sum();
            if points != usize::MAX && written > points {
                unit.finished = true;
                self.field_unit = Some(unit);
                return Ok(StepState::Pending);
            }
        }
        for f in unit.fields {
            // A field that failed has no plan.
            if let Some(plan) = f.plan {
                let i = f.job.i;
                // Fails only when the field's channel or mesh is missing.
                if let Err(e) = self.apply_field(f.job, plan) {
                    unit.error = unit.error.filter(|(j, _)| *j < i).or(Some((i, e)));
                }
            }
        }
        // Thickness and curvature values are already applied; their error
        // joins the unit's.
        if thick_error.is_err() && unit.error.is_none() {
            return thick_error.map(|()| StepState::Done);
        }
        match unit.error {
            Some((_, e)) => Err(e),
            None => Ok(StepState::Done),
        }
    }

    /// Drops the open step unit, if any: nothing it computed is written or
    /// reported, and its fields stay due. Does nothing when no unit is
    /// open. What the unit brought up to date (BVHs, finished sign data)
    /// stays so, and an unfinished sign data build is kept; see
    /// [`Scene::field_step`].
    pub fn abandon_field_step(&mut self) {
        self.drop_field_unit();
    }

    /// Whether a step unit is open (see [`Scene::field_step`]).
    pub fn field_step_pending(&self) -> bool {
        self.field_unit.is_some()
    }

    /// The meshes the fields of the open step unit are on and measured to,
    /// each once, in field order; empty when no unit is open. A change to
    /// any of them makes the unit stale.
    pub fn field_step_meshes(&self) -> Vec<MeshId> {
        let mut meshes = Vec::new();
        for f in self.field_unit.iter().flat_map(|u| &u.fields) {
            let desc = &self.fields[f.job.i].desc;
            for &m in std::iter::once(&desc.on).chain(&desc.to) {
                if !meshes.contains(&m) {
                    meshes.push(m);
                }
            }
        }
        meshes
    }

    /// Drops the open step unit, if any. Every public `&mut self` method of
    /// the scene but the step calls starts with this, as a unit holds field
    /// indices and values computed from the scene as it was. A new public
    /// `&mut self` method must call it first too, and join the list in the
    /// test `mutating_calls_abandon_open_unit`.
    pub(crate) fn drop_field_unit(&mut self) {
        self.field_unit = None;
        self.drop_stale_sign_builds();
    }

    /// Drops the unfinished sign data builds kept with the fields that are
    /// no longer for their mesh and BVH as they are (an edit since, say):
    /// they would only start over, and hold arrays the size of the mesh.
    fn drop_stale_sign_builds(&mut self) {
        let entries = &self.entries;
        for f in &mut self.fields {
            for (k, slot) in f.sign_builds.iter_mut().enumerate() {
                let stale = slot.as_ref().is_some_and(|b| {
                    entries
                        .get(f.desc.to[k])
                        .is_none_or(|e| !b.is_for(&e.mesh, &e.bvh))
                });
                if stale {
                    *slot = None;
                }
            }
        }
    }

    /// The candidates of the open step unit left to compute, counting only
    /// fields whose candidates are chosen.
    #[cfg(test)]
    pub(crate) fn field_step_left(&self) -> usize {
        self.field_unit.as_ref().map_or(0, |u| {
            u.fields[u.next..]
                .iter()
                .map(|f| f.pick.candidates.len() - f.values.len())
                .sum()
        })
    }

    /// Unfinished sign data builds kept with the fields.
    #[cfg(test)]
    fn kept_sign_builds(&self) -> usize {
        self.fields
            .iter()
            .flat_map(|f| &f.sign_builds)
            .filter(|b| b.is_some())
            .count()
    }

    /// The stage of the open step unit's first unfinished field.
    #[cfg(test)]
    fn field_step_stage(&self) -> Option<Stage> {
        let u = self.field_unit.as_ref()?;
        u.fields.get(u.next).map(|f| f.stage)
    }

    /// Opens a step unit over the due Eager fields, or over the field
    /// `only` when it is not current: prepares each (BVHs, what it
    /// recomputes), leaving out (with their errors) those that fail. `None`
    /// when no field is due; the first error when none opens.
    fn open_field_unit(&mut self, only: Option<FieldId>) -> Result<Option<FieldUnit>> {
        let due = match only {
            None => self.due_fields(|m| m == UpdateMode::Eager, None)?,
            Some(id) => {
                let i = self.field_index(id)?;
                if self.field_current(i) {
                    Vec::new()
                } else {
                    vec![i]
                }
            }
        };
        if due.is_empty() {
            return Ok(None);
        }
        let mut unit = FieldUnit {
            only,
            fields: Vec::new(),
            next: 0,
            finished: false,
            error: None,
        };
        unit.fields
            .try_reserve_exact(due.len())
            .map_err(|_| oom())?;
        for &i in &due {
            match self.prepare_job(i) {
                Ok(job) => {
                    let all = job.full;
                    unit.fields.push(UnitField {
                        job,
                        stage: Stage::Sign(0),
                        search: None,
                        near: Vec::new(),
                        all,
                        aborted: false,
                        pick: Pick::default(),
                        values: Vec::new(),
                        nearest: Vec::new(),
                        recomputed: 0,
                        plan: None,
                    });
                }
                Err(e) => unit.fail(i, e),
            }
        }
        if unit.fields.is_empty() {
            return Err(unit.error.map_or(SceneError::Corrupted, |(_, e)| e));
        }
        Ok(Some(unit))
    }

    /// Advances `unit`'s fields in field order with a budget of `points`
    /// (see [`Scene::field_step`]), giving each field whose values are all
    /// computed its plan. A field that fails is dropped from the unit, its
    /// error recorded: it gets no plan, and its work left costs nothing.
    fn advance_field_unit(&mut self, unit: &mut FieldUnit, points: usize) {
        let mut step = Step {
            points,
            left: points,
            worked: false,
        };
        while let Some(f) = unit.fields.get_mut(unit.next) {
            let i = f.job.i;
            match self.advance_field(f, &mut step) {
                Ok(false) => return,
                Ok(true) => {}
                Err(e) => {
                    f.search = None;
                    unit.fail(i, e);
                }
            }
            unit.next += 1;
        }
    }

    /// Advances one field of a unit through its stages within `step`.
    /// Returns `true` once the field has its plan, `false` when the step is
    /// spent first.
    fn advance_field(&mut self, f: &mut UnitField, step: &mut Step) -> Result<bool> {
        let i = f.job.i;
        let targets = self.fields[i].desc.to.len();
        loop {
            match f.stage {
                Stage::Sign(k) if k == targets => f.stage = Stage::Near(0),
                Stage::Sign(k) => {
                    if self.sign_current(i, k)? {
                        f.stage = Stage::Sign(k + 1);
                        continue;
                    }
                    if !step.begin_heavy() {
                        return Ok(false);
                    }
                    let elems = step.points.saturating_mul(SIGN_ELEMS_PER_POINT);
                    if self.step_sign(i, k, elems)? {
                        f.stage = Stage::Sign(k + 1);
                    }
                    if !step.unbounded() {
                        return Ok(false);
                    }
                }
                Stage::Near(k) if k == targets || f.all => f.stage = Stage::Pick,
                Stage::Near(k) => {
                    let Some(margin) = f.job.near_margins[k] else {
                        f.stage = Stage::Near(k + 1);
                        continue;
                    };
                    if !step.begin_heavy() {
                        return Ok(false);
                    }
                    let (on, to) = (self.fields[i].desc.on, self.fields[i].desc.to[k]);
                    let (ea, eb) = (self.entry(on)?, self.entry(to)?);
                    let search = match &mut f.search {
                        Some(search) => search,
                        empty => empty.insert(NearSearch::new(
                            &ea.mesh,
                            &ea.bvh,
                            &eb.mesh,
                            &eb.bvh,
                            &f.job.b_to_a[k],
                            round_up(margin),
                            f.job.near_limit,
                        )?),
                    };
                    let tests = step.points.saturating_mul(NEAR_TESTS_PER_POINT);
                    if search.step(&ea.mesh, &ea.bvh, &eb.mesh, &eb.bvh, tests)? {
                        let done = f.search.take().ok_or(SceneError::Corrupted)?;
                        let within = done.finish(&mut f.near)?.ok_or(SceneError::Corrupted)?;
                        f.aborted = !within;
                        f.all = f.aborted;
                        f.stage = Stage::Near(k + 1);
                    }
                    if !step.unbounded() {
                        return Ok(false);
                    }
                }
                Stage::Pick => {
                    // Choosing every vertex, or scanning for flipped ones,
                    // walks every vertex slot: a step of its own.
                    if (f.all || f.job.topology || f.job.live) && !step.begin_heavy() {
                        return Ok(false);
                    }
                    let near = std::mem::take(&mut f.near);
                    f.pick = self.assemble_pick(&f.job, &near, f.all, f.aborted)?;
                    let n = f.pick.candidates.len();
                    f.values.try_reserve_exact(n).map_err(|_| oom())?;
                    f.nearest.try_reserve_exact(n).map_err(|_| oom())?;
                    f.stage = Stage::Values;
                }
                Stage::Values => {
                    let (done, total) = (f.values.len(), f.pick.candidates.len());
                    if done < total {
                        if (step.worked && !step.unbounded()) || step.left == 0 {
                            return Ok(false);
                        }
                        let end = done + step.left.min(total - done);
                        let run = &f.pick.candidates[done..end];
                        f.recomputed +=
                            self.compute_values(&f.job, run, &mut f.values, &mut f.nearest)?;
                        step.left -= end - done;
                        if end < total {
                            return Ok(false);
                        }
                    }
                    f.stage = Stage::Finish;
                }
                Stage::Finish => {
                    // A full plan walks every vertex slot: a step of its own.
                    if f.all && !step.begin_heavy() {
                        return Ok(false);
                    }
                    let pick = std::mem::take(&mut f.pick);
                    let values = std::mem::take(&mut f.values);
                    let nearest = std::mem::take(&mut f.nearest);
                    f.plan =
                        Some(self.finish_field(&f.job, pick, values, nearest, f.recomputed)?);
                    return Ok(true);
                }
            }
        }
    }

    /// Whether field `i`'s sign data of target `k` are current. A baked
    /// target (see [`Scene::target_uses_temp`]) needs none: its real sign
    /// data are unused.
    fn sign_current(&self, i: usize, k: usize) -> Result<bool> {
        if self.target_uses_temp(i, k) {
            return Ok(true);
        }
        let f = &self.fields[i];
        let eb = self.entry(f.desc.to[k])?;
        Ok(f.sign[k]
            .as_ref()
            .is_some_and(|s| s.is_current(&eb.mesh, &eb.bvh)))
    }

    /// Runs the build of field `i`'s sign data of target `k` over up to
    /// `elems` more elements, resuming the field's unfinished build when it
    /// is for the mesh and BVH as they are, starting one otherwise. Stores
    /// the sign data when done, and returns whether it is.
    fn step_sign(&mut self, i: usize, k: usize, elems: usize) -> Result<bool> {
        let f = &mut self.fields[i];
        let eb = self
            .entries
            .get_mut(f.desc.to[k])
            .ok_or(SceneError::UnknownMesh)?;
        let slot = &mut f.sign_builds[k];
        let resuming = slot.as_ref().is_some_and(|b| b.is_for(&eb.mesh, &eb.bvh));
        // A small position-only edit refits around the dirty vertices, in
        // this one step, with the bits of a build (see `SignData::try_refit`).
        // Past about what a step of a build takes, it builds in steps.
        if !resuming
            && let (Some(sign), Some(changes)) = (
                &mut f.sign[k],
                eb.hook(f.id).and_then(FieldHook::refit_changes),
            )
            && changes.vertices_moved.len() <= elems / SIGN_ELEMS_PER_REFIT_VERTEX
            && sign.try_refit(&eb.mesh, &eb.bvh, changes)?
        {
            *slot = None;
            return Ok(true);
        }
        let build = match slot {
            Some(b) if b.is_for(&eb.mesh, &eb.bvh) => b,
            _ => slot.insert(SignBuild::new(&eb.mesh, &eb.bvh, f.desc.sign)?),
        };
        #[cfg(test)]
        counter::add_sign_step();
        match build.step(&mut eb.mesh, &eb.bvh, elems) {
            Ok(false) => Ok(false),
            Ok(true) => {
                f.sign[k] = slot.take().and_then(SignBuild::finish);
                Ok(true)
            }
            Err(e) => {
                *slot = None;
                Err(e.into())
            }
        }
    }

    /// Removes the fields on `mesh` or measured to it alone, and removes
    /// `mesh` from the targets of the fields that have others; those are
    /// recomputed in full on their next update. Returns the meshes those
    /// fields are on, each once.
    pub(crate) fn drop_fields_of(&mut self, mesh: MeshId) -> Vec<MeshId> {
        let mut kept_on = Vec::new();
        let mut i = 0;
        while i < self.fields.len() {
            let f = &mut self.fields[i];
            let target = f.desc.to.iter().position(|&m| m == mesh);
            if f.desc.on == mesh || (target.is_some() && f.desc.to.len() == 1) {
                self.detach_field(i);
                continue;
            }
            if let Some(k) = target {
                f.desc.to.remove(k);
                f.sign.remove(k);
                f.sign_builds.remove(k);
                f.fresh = true;
                // Until then, the indices follow the remaining targets.
                let k = k as u32;
                for n in &mut f.nearest {
                    if *n == k {
                        *n = u32::MAX;
                    } else if *n != u32::MAX && *n > k {
                        *n -= 1;
                    }
                }
                let (id, on) = (f.id, f.desc.on);
                if let Some(e) = self.entries.get_mut(mesh) {
                    e.hooks.retain(|h| h.field != id);
                }
                if !kept_on.contains(&on) {
                    kept_on.push(on);
                }
            }
            i += 1;
        }
        // A thickness or curvature field lives on its mesh alone: removing
        // the mesh removes the field.
        let mut t = 0;
        while t < self.thick.len() {
            if self.thick[t].desc.on == mesh {
                self.detach_thick(t);
            } else {
                t += 1;
            }
        }
        let mut c = 0;
        while c < self.curv.len() {
            if self.curv[c].desc.on == mesh {
                self.detach_curv(c);
            } else {
                c += 1;
            }
        }
        kept_on
    }

    fn detach_field(&mut self, i: usize) {
        let f = self.fields.remove(i);
        for &to in &f.desc.to {
            if let Some(eb) = self.entries.get_mut(to) {
                eb.hooks.retain(|h| h.field != f.id);
            }
        }
        if let Some(ea) = self.entries.get_mut(f.desc.on) {
            ea.hooks.retain(|h| h.field != f.id);
            let _ = ea.mesh.remove_channel(&f.channel);
            if ea
                .coloring
                .as_ref()
                .is_some_and(|c| c.coloring.field == f.id)
            {
                ea.coloring = None;
                ea.report_colors(None);
            }
            ea.report_field(f.id);
        }
    }

    /// Removes thickness field `t`: as [`Scene::detach_field`] without
    /// targets.
    fn detach_thick(&mut self, t: usize) {
        let f = self.thick.remove(t);
        if let Some(ea) = self.entries.get_mut(f.desc.on) {
            ea.hooks.retain(|h| h.field != f.id);
            let _ = ea.mesh.remove_channel(&f.channel);
            if ea
                .coloring
                .as_ref()
                .is_some_and(|c| c.coloring.field == f.id)
            {
                ea.coloring = None;
                ea.report_colors(None);
            }
            ea.report_field(f.id);
        }
    }

    /// Removes curvature field `c`: as [`Scene::detach_field`] without
    /// targets.
    fn detach_curv(&mut self, c: usize) {
        let f = self.curv.remove(c);
        if let Some(ea) = self.entries.get_mut(f.desc.on) {
            ea.hooks.retain(|h| h.field != f.id);
            let _ = ea.mesh.remove_channel(&f.channel);
            if ea
                .coloring
                .as_ref()
                .is_some_and(|c| c.coloring.field == f.id)
            {
                ea.coloring = None;
                ea.report_colors(None);
            }
            ea.report_field(f.id);
        }
    }

    /// The largest absolute coordinate of A's root box, of B's root box moved
    /// into A's space (the frame of the pair test), as in Phase 3b's pair
    /// pruning, and of B's root box in B's own space (the kernel rounds in
    /// B's local `f32` frame): the magnitude the rounding of either side
    /// scales with. 0 for a mesh without triangles.
    fn pair_magnitude(&self, on: MeshId, to: MeshId) -> Result<f64> {
        let xf = self.relative_transform(on, to)?;
        let root = |m: MeshId| -> Result<Option<_>> {
            Ok(self.entry(m)?.bvh.root_bounds().filter(|b| !b.is_empty()))
        };
        let mut m = root(on)?.map_or(0.0, |b| box_magnitude(b.min, b.max));
        if let Some(b) = root(to)? {
            m = m.max(box_magnitude(b.min, b.max));
            for k in 0..8 {
                let corner =
                    std::array::from_fn(|d| if k >> d & 1 == 0 { b.min[d] } else { b.max[d] });
                let p = xf.apply(corner);
                m = p.iter().fold(m, |m, c| m.max(c.abs()));
            }
        }
        Ok(m)
    }

    /// Whether B's root box, moved into A's space by `b_to_a`, comes within
    /// `margin` of A's root box: when it does not, the near search ends at
    /// the root pair. False when either mesh has no triangles. Only for the
    /// work estimate, so the moved box is the bounds of its moved corners.
    fn roots_within(&self, on: MeshId, to: MeshId, b_to_a: &Rigid, margin: f64) -> Result<bool> {
        let root = |m: MeshId| -> Result<Option<_>> {
            Ok(self.entry(m)?.bvh.root_bounds().filter(|b| !b.is_empty()))
        };
        let (Some(a), Some(b)) = (root(on)?, root(to)?) else {
            return Ok(false);
        };
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for k in 0..8 {
            let corner = std::array::from_fn(|d| if k >> d & 1 == 0 { b.min[d] } else { b.max[d] });
            let p = b_to_a.apply(corner);
            for d in 0..3 {
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
        }
        let gap2: f64 = (0..3)
            .map(|d| {
                let g = (f64::from(a.min[d]) - hi[d])
                    .max(lo[d] - f64::from(a.max[d]))
                    .max(0.0);
                g * g
            })
            .sum();
        Ok(gap2.sqrt() <= margin)
    }

    /// Brings field `i` up to date (see the module docs for what is
    /// recomputed). All or nothing: on error nothing is written and the
    /// pending changes stay.
    fn update_field(&mut self, i: usize) -> Result<()> {
        self.update_due(&[i])
    }

    /// Brings the fields at indices `due` (ascending) up to date, in three
    /// phases: each is prepared in turn ([`Scene::prepare_field`], which
    /// needs the meshes mutably), then all are computed without writing
    /// (in parallel across fields when the work is large enough), then each
    /// is applied in field order, so that values, colours and reports come
    /// out as if the fields were updated one after the other. A field that
    /// fails at any phase is left as it was. Returns the first error in field
    /// order; every field is attempted.
    ///
    /// Preparing a field before the earlier ones are applied changes
    /// nothing: preparing writes no field state and reports nothing, and a
    /// field's update reads only its own channel, sign data and hooks.
    fn update_due(&mut self, due: &[usize]) -> Result<()> {
        let mut outcomes: Vec<Result<()>> = Vec::new();
        outcomes.try_reserve_exact(due.len()).map_err(|_| oom())?;
        let mut jobs: Vec<(usize, FieldJob)> = Vec::new();
        jobs.try_reserve_exact(due.len()).map_err(|_| oom())?;
        for (k, &i) in due.iter().enumerate() {
            match self.prepare_field(i) {
                Ok(job) => {
                    jobs.push((k, job));
                    outcomes.push(Ok(()));
                }
                Err(e) => outcomes.push(Err(e)),
            }
        }
        self.share_pair_queries(&mut jobs)?;
        match self.compute_fields(&jobs) {
            Ok(plans) => {
                for ((k, job), plan) in jobs.into_iter().zip(plans) {
                    outcomes[k] = plan.and_then(|plan| self.apply_field(job, plan));
                }
            }
            Err(e) => {
                for (k, _) in jobs {
                    outcomes[k] = Err(e.clone());
                }
            }
        }
        outcomes.into_iter().find(Result::is_err).unwrap_or(Ok(()))
    }

    /// Runs one pair query for each reciprocal pair of single-target fields
    /// among `jobs` that both need a near set and share the same
    /// `max_distance`, storing each side's near triangles in its job so that
    /// [`Scene::pick_candidates`] runs no second query. With colouring on
    /// both sides, A→B and B→A would otherwise each run `triangles_near`
    /// and throw away the other half; one query roughly halves the
    /// two-sided drag cost.
    fn share_pair_queries(&self, jobs: &mut [(usize, FieldJob)]) -> Result<()> {
        use std::collections::HashMap;
        let mut groups: HashMap<(MeshId, MeshId), Vec<usize>> = HashMap::new();
        for (j, (_, job)) in jobs.iter().enumerate() {
            if job.full || job.shared_near.is_some() {
                continue;
            }
            let f = &self.fields[job.i];
            if f.desc.to.len() != 1 || job.near_margins.len() != 1 {
                continue;
            }
            if job.near_margins[0].is_none() {
                continue;
            }
            let (a, b) = (f.desc.on, f.desc.to[0]);
            let key = if a < b { (a, b) } else { (b, a) };
            groups.entry(key).or_default().push(j);
        }
        for js in groups.into_values() {
            let mut used = vec![false; js.len()];
            for (x, &jx) in js.iter().enumerate() {
                if used[x] {
                    continue;
                }
                let fx = &self.fields[jobs[jx].1.i];
                for (y, &jy) in js.iter().enumerate().skip(x + 1) {
                    if used[y] {
                        continue;
                    }
                    let fy = &self.fields[jobs[jy].1.i];
                    if fx.desc.on != fy.desc.to[0] || fx.desc.to[0] != fy.desc.on {
                        continue;
                    }
                    if fx.desc.max_distance != fy.desc.max_distance {
                        continue;
                    }
                    let (Some(m1), Some(m2)) =
                        (jobs[jx].1.near_margins[0], jobs[jy].1.near_margins[0])
                    else {
                        continue;
                    };
                    let margin = m1.max(m2);
                    let ea = self.entry(fx.desc.on)?;
                    let eb = self.entry(fx.desc.to[0])?;
                    let mut near_a = Vec::new();
                    let mut near_b = Vec::new();
                    fast_mesh_query::triangles_near(
                        &ea.mesh,
                        &ea.bvh,
                        &eb.mesh,
                        &eb.bvh,
                        &jobs[jx].1.b_to_a[0],
                        round_up(margin),
                        &mut near_a,
                        &mut near_b,
                    )?;
                    jobs[jx].1.shared_near = Some(near_a);
                    jobs[jy].1.shared_near = Some(near_b);
                    #[cfg(test)]
                    counter::add_shared();
                    used[x] = true;
                    used[y] = true;
                    break;
                }
            }
        }
        Ok(())
    }

    /// Brings what field `i` reads up to date, the part of an update that
    /// needs the meshes mutably: its BVHs (the targets' always, A's when a
    /// near set is needed) and the targets' sign data. Then decides what the
    /// update recomputes (see the module docs). Writes no field state.
    fn prepare_field(&mut self, i: usize) -> Result<FieldJob> {
        let job = self.prepare_job(i)?;
        let mode = self.fields[i].desc.sign;
        for k in 0..self.fields[i].desc.to.len() {
            // A non-uniformly scaled target is baked into the frame (see
            // `FieldJob::temp_targets`): its real sign data are unused.
            if job.scaled && job.temp_targets[k].is_some() {
                continue;
            }
            let f = &mut self.fields[i];
            let eb = self
                .entries
                .get_mut(f.desc.to[k])
                .ok_or(SceneError::UnknownMesh)?;
            if f.sign[k]
                .as_ref()
                .is_some_and(|s| s.is_current(&eb.mesh, &eb.bvh))
            {
                f.sign_builds[k] = None;
                continue;
            }
            // A position-only edit refits around the dirty vertices; anything
            // else rebuilds over the whole mesh.
            let id = f.id;
            let refitted = match (
                &mut f.sign[k],
                eb.hook(id).and_then(FieldHook::refit_changes),
            ) {
                (Some(s), Some(changes)) => s.try_refit(&eb.mesh, &eb.bvh, changes)?,
                _ => false,
            };
            if !refitted {
                let f = &mut self.fields[i];
                let eb = self
                    .entries
                    .get_mut(f.desc.to[k])
                    .ok_or(SceneError::UnknownMesh)?;
                f.sign[k] = Some(SignData::build(&mut eb.mesh, &eb.bvh, mode)?);
            }
            // Current now: an unfinished build of it is of no use.
            self.fields[i].sign_builds[k] = None;
        }
        Ok(job)
    }

    /// The near margin of field `i`'s next interaction update: what its last
    /// left, or at first a fraction of the diagonal of A's box.
    fn live_margin(&self, i: usize) -> Result<f64> {
        let margin = self.fields[i].live.margin;
        if margin > 0.0 {
            return Ok(margin);
        }
        let ea = self.entry(self.fields[i].desc.on)?;
        let diagonal = ea
            .bvh
            .root_bounds()
            .filter(|b| !b.is_empty())
            .map_or(0.0, |b| {
                let d = [0, 1, 2].map(|c| f64::from(b.max[c]) - f64::from(b.min[c]));
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            });
        Ok(LIVE_FIRST_MARGIN * diagonal)
    }

    /// Field `i`'s `LiveState::travel` once its targets are at `b_to_a`: a
    /// point of A has moved relative to a target by at most the farthest move
    /// of a corner of A's box (a rigid move is affine, and the box holds A).
    fn live_travel(&self, i: usize, b_to_a: &[Rigid]) -> Result<f64> {
        let f = &self.fields[i];
        let mut step = 0.0f64;
        if f.live.poses.len() == b_to_a.len()
            && let Some(b) = self
                .entry(f.desc.on)?
                .bvh
                .root_bounds()
                .filter(|b| !b.is_empty())
        {
            for (old, new) in f.live.poses.iter().zip(b_to_a) {
                let (old, new) = (old.inverse(), new.inverse());
                for k in 0..8 {
                    let corner =
                        std::array::from_fn(|d| if k >> d & 1 == 0 { b.min[d] } else { b.max[d] });
                    let (p, q) = (old.apply(corner), new.apply(corner));
                    let d: f64 = (0..3).map(|c| (p[c] - q[c]) * (p[c] - q[c])).sum();
                    step = step.max(d.sqrt());
                }
            }
        }
        Ok(f.live.travel + step)
    }

    /// [`Scene::prepare_field`] but the sign data: the BVHs, then what the
    /// update recomputes. A step unit builds the sign data in its steps.
    fn prepare_job(&mut self, i: usize) -> Result<FieldJob> {
        let (id, on) = (self.fields[i].id, self.fields[i].desc.on);
        let mut to = Vec::new();
        to.try_reserve_exact(self.fields[i].desc.to.len() + 1)
            .map_err(|_| oom())?;
        to.extend_from_slice(&self.fields[i].desc.to);
        let n = to.len();
        let md = self.fields[i].desc.max_distance;
        let hook = |m: MeshId| -> Result<FieldHook> {
            self.entry(m)?
                .hook(id)
                .cloned()
                .ok_or(SceneError::Corrupted)
        };
        let hook_a = hook(on)?;
        if self.is_scaled_field(on, &to)? {
            // A scaled placement is involved: the rigid maps below do not
            // exist, so the update recomputes every vertex through the
            // placements (see `FieldJob::scaled`). Only the targets' BVHs
            // are needed (A's takes no part); their real sign data are
            // built in `prepare_field` (or the step unit) except for the
            // non-uniform targets baked here.
            self.prepare(Some(&to))?;
            return self.prepare_scaled_job(i, on, to, hook_a);
        }
        let mut whole = self.fields[i].fresh || hook_a.changes.is_none();
        let mut moved = Vec::new();
        moved.try_reserve_exact(n).map_err(|_| oom())?;
        for &m in &to {
            let hook_b = hook(m)?;
            whole |= hook_b.changes.is_none();
            moved.push(
                hook_a.placement_changed
                    || hook_b.placement_changed
                    || hook_b.changes.as_ref().is_some_and(|c| !c.is_empty()),
            );
        }
        // The interaction tier takes what is not due in full for itself.
        let live = !whole
            && self
                .field_interaction
                .is_some_and(|points| self.fields[i].finite.len() > points);
        let full = !live && (whole || md.is_infinite() || self.fields[i].live.stale);
        // The targets' BVHs are always needed; A's only for a near set.
        let need_near = !full && !live && moved.contains(&true);
        // The tier queries A's BVH around an edit of a target.
        let edited = live
            && to.iter().any(|&m| {
                self.entry(m)
                    .ok()
                    .and_then(|e| e.hook(self.fields[i].id))
                    .and_then(|h| h.changes.as_ref())
                    .is_some_and(|c| !c.is_empty())
            });
        if need_near || edited {
            to.push(on);
        }
        self.prepare(Some(&to))?;
        to.truncate(n);
        let mut near_margins = Vec::new();
        near_margins.try_reserve_exact(n).map_err(|_| oom())?;
        for (k, &m) in to.iter().enumerate() {
            near_margins.push(if need_near && moved[k] {
                Some(pad(md, self.pair_magnitude(on, m)?))
            } else {
                None
            });
        }
        let mut b_to_a = Vec::new();
        b_to_a.try_reserve_exact(n).map_err(|_| oom())?;
        for &m in &to {
            b_to_a.push(self.relative_transform(on, m)?);
        }
        let live_travel = if live {
            self.live_travel(i, &b_to_a)?
        } else {
            0.0
        };
        let ma = &self.entry(on)?.mesh;
        let changes = hook_a.changes;
        let topology = changes.as_ref().is_some_and(ChangeSet::topology_changed);
        let work = if full {
            ma.vertex_slots().saturating_mul(n)
        } else {
            let listed = changes.as_ref().map_or(0, |c| {
                c.vertices_added.len() + c.vertices_moved.len() + c.vertices_removed.len()
            });
            let known = if live {
                self.field_interaction.unwrap_or(0)
            } else if moved.contains(&true) {
                self.fields[i].finite.len()
            } else {
                0
            };
            let mut near = 0;
            for (k, &m) in to.iter().enumerate() {
                if let Some(margin) = near_margins[k]
                    && self.roots_within(on, m, &b_to_a[k], margin)?
                {
                    near += ma.triangle_count();
                }
            }
            let flips = if topology { ma.vertex_slots() } else { 0 };
            (listed + known).saturating_mul(n) + near + flips
        };
        Ok(FieldJob {
            i,
            full,
            moved,
            topology,
            changes,
            near_margins,
            b_to_a,
            near_limit: near_pair_limit(ma.vertex_slots()),
            work,
            shared_near: None,
            live,
            live_travel,
            scaled: false,
            frame_scale: 1.0,
            temp_targets: Vec::new(),
        })
    }

    /// Whether the field's update runs scaled: A's placement or any target's
    /// is scaled.
    fn is_scaled_field(&self, on: MeshId, to: &[MeshId]) -> Result<bool> {
        if self.entry(on)?.placement.is_scaled() {
            return Ok(true);
        }
        for &m in to {
            if self.entry(m)?.placement.is_scaled() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether the update of field `i` bakes target `k` (a non-uniformly
    /// scaled target, whose real sign data are unused).
    fn target_uses_temp(&self, i: usize, k: usize) -> bool {
        self.entry(self.fields[i].desc.to[k])
            .is_ok_and(|e| e.placement.is_scaled() && e.placement.uniform_scale().is_none())
    }

    /// [`Scene::prepare_job`] for a scaled field: every vertex is
    /// recomputed through the placements, on baked copies of the
    /// non-uniformly scaled targets. `hook_a` is A's merged changes.
    fn prepare_scaled_job(
        &mut self,
        i: usize,
        on: MeshId,
        to: Vec<MeshId>,
        hook_a: FieldHook,
    ) -> Result<FieldJob> {
        let n = to.len();
        let pa = self.entry(on)?.placement;
        // The frame the update measures in: A's local space when A is rigid
        // or uniformly scaled (frame distances times `frame_scale` are world
        // distances), else the world. Non-uniform targets are baked into it.
        let in_a_frame = pa.uniform_scale().is_some();
        let frame_scale = pa.uniform_scale().unwrap_or(1.0);
        let mut temp_targets: Vec<Option<TempTarget>> = Vec::new();
        temp_targets.try_reserve_exact(n).map_err(|_| oom())?;
        for &m in &to {
            let (pb, ob, options) = {
                let eb = self.entry(m)?;
                (eb.placement, eb.mesh.origin(), *eb.bvh.options())
            };
            if pb.is_scaled() && pb.uniform_scale().is_none() {
                let map = |l: [f32; 3]| {
                    let full = [
                        ob[0] + f64::from(l[0]),
                        ob[1] + f64::from(l[1]),
                        ob[2] + f64::from(l[2]),
                    ];
                    if in_a_frame {
                        pa.inverse_apply(pb.apply(full))
                    } else {
                        pb.apply(full)
                    }
                };
                let eb = self.entry(m)?;
                let (mut mesh, bvh) = baked_mesh(&eb.mesh, map, &options)?;
                let mode = self.fields[i].desc.sign;
                let sign = SignData::build(&mut mesh, &bvh, mode)?;
                temp_targets.push(Some(TempTarget { mesh, bvh, sign }));
            } else {
                temp_targets.push(None);
            }
        }
        let ma = &self.entry(on)?.mesh;
        let changes = hook_a.changes;
        let topology = changes.as_ref().is_some_and(ChangeSet::topology_changed);
        let slots = ma.vertex_slots();
        let mut moved = Vec::new();
        moved.try_reserve_exact(n).map_err(|_| oom())?;
        moved.resize(n, true);
        Ok(FieldJob {
            i,
            full: true,
            moved,
            topology,
            changes,
            near_margins: Vec::new(),
            b_to_a: Vec::new(),
            near_limit: near_pair_limit(slots),
            // Every vertex against every target, plus one baked copy each.
            work: slots.saturating_mul(n),
            shared_near: None,
            live: false,
            live_travel: 0.0,
            scaled: true,
            frame_scale,
            temp_targets,
        })
    }

    /// Computes the prepared fields without writing anything: one result
    /// per job, in job order. With the `parallel` feature, when the jobs'
    /// work adds up to [`PAR_MIN_FIELD_WORK`] or more, the fields are
    /// computed in parallel, one job per task.
    ///
    /// # Errors
    /// `OutOfMemory` when the per-job results cannot be allocated; every
    /// field then stays as it was.
    fn compute_fields(&self, jobs: &[(usize, FieldJob)]) -> Result<Vec<Result<FieldPlan>>> {
        let threshold = field_threshold(jobs.iter().map(|(_, job)| job));
        // Chunks of one job; a field's error is its own result.
        par::map_chunks(jobs, 1, threshold, oom, |_, chunk| {
            Ok(self.compute_field(&chunk[0].1))
        })
    }

    /// Computes the new values of a prepared field and what changes with
    /// them, reading the meshes only: the vertices to recompute (searching
    /// A's triangles near B when needed), their distances, and the field's
    /// new bookkeeping. A step unit runs the same three parts, the middle
    /// one in pieces.
    fn compute_field(&self, job: &FieldJob) -> Result<FieldPlan> {
        let pick = self.pick_candidates(job)?;
        let mut values = Vec::with_capacity(pick.candidates.len());
        let mut nearest = Vec::with_capacity(pick.candidates.len());
        let recomputed = self.compute_values(job, &pick.candidates, &mut values, &mut nearest)?;
        self.finish_field(job, pick, values, nearest, recomputed)
    }

    /// The vertices a prepared field's update recomputes, reading the
    /// meshes only.
    fn pick_candidates(&self, job: &FieldJob) -> Result<Pick> {
        if job.scaled {
            // Scaled updates recompute every vertex through the placements
            // (see `compute_values_scaled`), with no near search.
            return self.assemble_pick(job, &[], true, false);
        }
        let f = &self.fields[job.i];
        let ea = self.entry(f.desc.on)?;
        let ma = &ea.mesh;

        // A shared pair query with the reciprocal field already found A's
        // near triangles: no search here.
        if let Some(shared) = &job.shared_near {
            return self.assemble_pick(job, shared, job.full, false);
        }
        // A's triangles near each moved target, as `Scene::triangles_near`
        // (with the BVHs `prepare_field` brought up to date), unless one
        // search finds too many pairs: then every vertex is recomputed.
        let mut near = Vec::new();
        let mut all = job.full;
        let mut aborted = false;
        for (k, margin) in job.near_margins.iter().enumerate() {
            let Some(margin) = margin.filter(|_| !all) else {
                continue;
            };
            let eb = self.entry(f.desc.to[k])?;
            aborted = !fast_mesh_query::triangles_near_limited(
                ma,
                &ea.bvh,
                &eb.mesh,
                &eb.bvh,
                &job.b_to_a[k],
                round_up(margin),
                job.near_limit,
                &mut near,
            )?;
            all = aborted;
        }
        self.assemble_pick(job, &near, all, aborted)
    }

    /// The vertices a prepared field's update recomputes, given A's
    /// triangles `near` the moved targets, whether `all` are, and whether a
    /// near search passed its limit. Reads the meshes only.
    fn assemble_pick(
        &self,
        job: &FieldJob,
        near: &[TriangleId],
        all: bool,
        aborted: bool,
    ) -> Result<Pick> {
        if job.live {
            return self.assemble_live_pick(job);
        }
        let f = &self.fields[job.i];
        let ma = &self.entry(f.desc.on)?.mesh;
        let slots = ma.vertex_slots();
        let defined = |v: VertexId| ma.is_vertex_alive(v) && ma.corner_count(v) > 0;

        // The vertices to recompute, ascending.
        let candidates: Vec<VertexId> = if all {
            (0..slots as VertexId).collect()
        } else {
            let mut c = Vec::new();
            for &t in near {
                if let Some(tri) = ma.triangle(t) {
                    c.extend_from_slice(&tri);
                }
            }
            if job.geometric() {
                // The finite values that came from a moved target.
                let from_moved = |v: VertexId| match f.nearest.get(v as usize) {
                    Some(&k) => job.moved.get(k as usize).copied().unwrap_or(true),
                    None => true,
                };
                c.extend(f.finite.iter().copied().filter(|&v| from_moved(v)));
            }
            if let Some(ch) = &job.changes {
                c.extend_from_slice(&ch.vertices_added);
                c.extend_from_slice(&ch.vertices_moved);
                c.extend_from_slice(&ch.vertices_removed);
            }
            if job.topology {
                for v in 0..slots as VertexId {
                    let was = f.referenced.get(v as usize).copied().unwrap_or(false);
                    if was != defined(v) {
                        c.push(v);
                    }
                }
            }
            c.sort_unstable();
            c.dedup();
            c
        };
        Ok(Pick {
            candidates,
            all,
            aborted,
            margin: 0.0,
        })
    }

    /// The vertices an update of the interaction tier recomputes, from the
    /// values alone (no pair search: one pass over the slots, whose cost does
    /// not depend on how many triangles are close):
    ///
    /// - the near vertices. A distance changes no faster than the motion, so
    ///   a vertex can now be within the band only when its value, less how
    ///   far the meshes moved since it was computed, is. Over the budget, the
    ///   smallest such first, and the next band is the last one taken. So
    ///   the contact stays exact within the band, and a vertex farther away
    ///   comes up as the motion adds up;
    /// - what a target's edit may have changed the most: A's triangles in the
    ///   box of its moved vertices (widened by the band), spread over them up
    ///   to [`LIVE_EDIT_SHARE`] of the budget;
    /// - A's own added and removed vertices, those whose "used by a triangle"
    ///   state flipped, and those it moved, the closest to the targets first,
    ///   up to the same share (the others differ by no more than they moved);
    /// - the sweep (see [`LIVE_SWEEP_SHARE`]).
    fn assemble_live_pick(&self, job: &FieldJob) -> Result<Pick> {
        let f = &self.fields[job.i];
        let ea = self.entry(f.desc.on)?;
        let ma = &ea.mesh;
        let slots = ma.vertex_slots();
        let defined = |v: VertexId| ma.is_vertex_alive(v) && ma.corner_count(v) > 0;
        let values = ma.channel(&f.channel).ok_or(SceneError::Corrupted)?;
        let points = self.field_interaction.unwrap_or(1);
        let margin = self.live_margin(job.i)?;
        let md = f.desc.max_distance;
        let stamps = &f.live.stamps;
        // Less than the value by how far it may have moved; far values count
        // as `max_distance`, the nearest they can be.
        let reach = |v: usize| -> f64 {
            let moved = stamps
                .get(v)
                .map_or(0.0, |&t| job.live_travel - f64::from(t));
            f64::from(values[v].abs()).min(md) - moved
        };
        let mut c: Vec<VertexId> = (0..values.len().min(slots))
            .filter(|&v| !values[v].is_nan() && reach(v) <= margin)
            .map(|v| v as VertexId)
            .collect();
        let mut next = margin;
        if c.len() > points {
            let key = |&v: &VertexId| (reach(v as usize), v);
            let (_, last, _) = c.select_nth_unstable_by(points - 1, |x, y| {
                let (a, b) = (key(x), key(y));
                a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
            });
            let band = reach(*last as usize);
            c.truncate(points);
            // Overdue vertices take the budget: the band itself holds.
            if band > 0.0 {
                next = band;
            }
        } else if c.len() < points / 2 {
            next = margin * LIVE_GROW;
        }
        let edits = (points / LIVE_EDIT_SHARE).max(1);
        let pad = margin as f32;
        let mut boxed = Vec::new();
        for (k, &m) in f.desc.to.iter().enumerate() {
            let eb = self.entry(m)?;
            let Some(ch) = eb.hook(f.id).and_then(|h| h.changes.as_ref()) else {
                continue;
            };
            let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
            for &v in ch.vertices_moved.iter().chain(&ch.vertices_added) {
                if let Some(p) = eb.mesh.position(v) {
                    let q = job.b_to_a[k].apply(p);
                    for d in 0..3 {
                        lo[d] = lo[d].min(q[d] as f32 - pad);
                        hi[d] = hi[d].max(q[d] as f32 + pad);
                    }
                }
            }
            if lo[0] <= hi[0] {
                boxed.clear();
                ea.bvh.query_aabb(ma, lo, hi, &mut boxed);
                boxed.sort_unstable();
                let every = boxed.len().div_ceil(edits).max(1);
                let tris = boxed.iter().step_by(every).filter_map(|&t| ma.triangle(t));
                c.extend(tris.flatten());
            }
        }
        if let Some(ch) = &job.changes {
            c.extend_from_slice(&ch.vertices_added);
            c.extend_from_slice(&ch.vertices_removed);
            // A brush moves many vertices a little, and a vertex far from
            // the targets costs many times one near them: those closest to
            // the contact, within the budget.
            let mut moved = ch.vertices_moved.clone();
            if moved.len() > edits {
                let key = |&v: &VertexId| (values.get(v as usize).map_or(0.0, |x| x.abs()), v);
                moved.select_nth_unstable_by(edits - 1, |x, y| {
                    let (a, b) = (key(x), key(y));
                    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
                });
                moved.truncate(edits);
            }
            c.extend_from_slice(&moved);
        }
        if job.topology {
            for v in 0..slots as VertexId {
                let was = f.referenced.get(v as usize).copied().unwrap_or(false);
                if was != defined(v) {
                    c.push(v);
                }
            }
        }
        let (start, count) = self.live_sweep(job.i, slots);
        c.extend((0..count).map(|k| ((start as usize + k) % slots) as VertexId));
        c.sort_unstable();
        c.dedup();
        Ok(Pick {
            candidates: c,
            all: false,
            aborted: false,
            margin: next,
        })
    }

    /// The sweep of field `i` over `slots` vertex slots, this update: its
    /// first slot and how many it visits (see [`LIVE_SWEEP_SHARE`]).
    fn live_sweep(&self, i: usize, slots: usize) -> (u32, usize) {
        if slots == 0 {
            return (0, 0);
        }
        let count = self
            .field_interaction
            .map_or(0, |points| (points / LIVE_SWEEP_SHARE).max(1))
            .min(slots);
        (self.fields[i].live.sweep % slots as u32, count)
    }

    /// Appends the new values of `candidates`, a run of a prepared field's
    /// candidates, to `values`, and the targets they came from to
    /// `nearest`, reading the meshes only. Every value depends only on its
    /// own vertex, so computing the candidates in runs gives the same bits
    /// as all at once. Returns the points the kernel ran on.
    fn compute_values(
        &self,
        job: &FieldJob,
        candidates: &[VertexId],
        values: &mut Vec<f32>,
        nearest: &mut Vec<u32>,
    ) -> Result<usize> {
        if job.scaled {
            return self.compute_values_scaled(job, candidates, values, nearest);
        }
        let f = &self.fields[job.i];
        let ma = &self.entry(f.desc.on)?.mesh;
        let defined = |v: VertexId| ma.is_vertex_alive(v) && ma.corner_count(v) > 0;
        #[cfg(test)]
        if counter::fails(f.id) {
            return Err(oom());
        }
        let mut targets = Vec::new();
        targets
            .try_reserve_exact(f.desc.to.len())
            .map_err(|_| oom())?;
        for (k, &m) in f.desc.to.iter().enumerate() {
            let eb = self.entry(m)?;
            targets.push(Target {
                mesh: &eb.mesh,
                bvh: &eb.bvh,
                sign: f.sign[k].as_ref().ok_or(SceneError::Corrupted)?,
                // A's points go into each target's local space in f64, then
                // are rounded to f32 (§5.2).
                to_local: job.b_to_a[k].inverse(),
            });
        }

        let mut ids = Vec::new();
        let mut points = Vec::new();
        for &v in candidates {
            if let Some(p) = ma.position(v).filter(|_| defined(v)) {
                ids.push(v);
                points.push(p);
            }
        }
        let mut out = vec![0.0f32; points.len()];
        let mut from = vec![0u32; points.len()];
        let md = f.desc.max_distance;
        signed_distances_multi(&targets, &points, md, &mut out, &mut from)?;

        let mut k = 0;
        for &v in candidates {
            let (x, n) = if ids.get(k) == Some(&v) {
                k += 1;
                (out[k - 1], from[k - 1])
            } else {
                (f32::NAN, u32::MAX)
            };
            values.push(x);
            nearest.push(n);
        }
        Ok(points.len())
    }

    /// Appends the new values of `candidates` like [`Scene::compute_values`]
    /// for a scaled field: every candidate is measured through the
    /// placements, one target batch at a time.
    ///
    /// The frame the candidates are measured in is A's local space when A
    /// is rigid or uniformly scaled (frame distances times
    /// [`FieldJob::frame_scale`] are world distances), else the world. Each
    /// target is measured in its own local space, where the kernel runs:
    /// - a rigid target: the frame point goes to the world and back into the
    ///   target's local space; the kernel's bound and values are world ones;
    /// - a uniformly scaled target: the same, through the target's inverse;
    ///   the kernel runs with the world bound divided by the scale and its
    ///   values are multiplied back (a uniform scale multiplies every
    ///   distance, and signs are preserved by a positive scale);
    /// - a non-uniformly scaled target: its baked copy in
    ///   [`FieldJob::temp_targets`] already lives in the frame, so the frame
    ///   points go in directly and the values are times `frame_scale`.
    ///
    /// The per-vertex value is the minimum over the targets, ties going to
    /// the lowest target index, with magnitudes past `max_distance`
    /// (compared in `f64`) reading `+∞` and no target — the same bits as
    /// [`signed_distances_multi`] over the same local points.
    fn compute_values_scaled(
        &self,
        job: &FieldJob,
        candidates: &[VertexId],
        values: &mut Vec<f32>,
        nearest: &mut Vec<u32>,
    ) -> Result<usize> {
        let f = &self.fields[job.i];
        #[cfg(test)]
        if counter::fails(f.id) {
            return Err(oom());
        }
        if f.desc.to.len() >= u32::MAX as usize {
            return Err(SceneError::InvalidInput("too many targets"));
        }
        let ea = self.entry(f.desc.on)?;
        let (pa, oa) = (ea.placement, ea.mesh.origin());
        let in_a_frame = pa.uniform_scale().is_some();
        let md = f.desc.max_distance;
        // The candidates' frame points in f64 (`None` for dead or unused
        // vertices, which read NaN), and the positions of the defined ones.
        let mut frame: Vec<Option<[f64; 3]>> = Vec::new();
        frame
            .try_reserve_exact(candidates.len())
            .map_err(|_| oom())?;
        for &v in candidates {
            frame.push(
                ea.mesh
                    .position(v)
                    .filter(|_| ea.mesh.is_vertex_alive(v) && ea.mesh.corner_count(v) > 0)
                    .map(|p| {
                        if in_a_frame {
                            [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
                        } else {
                            pa.apply([
                                oa[0] + f64::from(p[0]),
                                oa[1] + f64::from(p[1]),
                                oa[2] + f64::from(p[2]),
                            ])
                        }
                    }),
            );
        }
        let mut at: Vec<usize> = Vec::new();
        at.try_reserve_exact(candidates.len()).map_err(|_| oom())?;
        let mut fpts: Vec<[f64; 3]> = Vec::new();
        fpts.try_reserve_exact(candidates.len())
            .map_err(|_| oom())?;
        for (ci, q) in frame.iter().enumerate() {
            if let Some(q) = q {
                at.push(ci);
                fpts.push(*q);
            }
        }
        let mut best: Vec<(f32, u32)> = Vec::new();
        best.try_reserve_exact(candidates.len())
            .map_err(|_| oom())?;
        best.resize(candidates.len(), (f32::INFINITY, u32::MAX));
        for (k, &m) in f.desc.to.iter().enumerate() {
            // The measuring space, its kernel bound (in its own units) and
            // what its values are multiplied by to read world distances.
            let eb = self.entry(m)?;
            let (mesh, bvh, sign, bound, to_world) = match &job.temp_targets[k] {
                Some(t) => (
                    &t.mesh,
                    &t.bvh,
                    &t.sign,
                    round_up(md / job.frame_scale),
                    job.frame_scale,
                ),
                None => {
                    let s = eb.placement.uniform_scale().unwrap_or(1.0);
                    (
                        &eb.mesh,
                        &eb.bvh,
                        f.sign[k].as_ref().ok_or(SceneError::Corrupted)?,
                        round_up(md / s),
                        s,
                    )
                }
            };
            let (pb, ob) = (eb.placement, eb.mesh.origin());
            let temp = job.temp_targets[k].is_some();
            let mut points: Vec<[f32; 3]> = Vec::new();
            points.try_reserve_exact(fpts.len()).map_err(|_| oom())?;
            for &q in &fpts {
                // Into the target's measuring space: a baked copy lives in
                // the frame already, so its points go in directly; a real
                // target's go to the world first (A's placement), then back
                // into its stored local space (the placement inverse, minus
                // its origin).
                let l = if temp {
                    q
                } else {
                    let w = if in_a_frame {
                        pa.apply([oa[0] + q[0], oa[1] + q[1], oa[2] + q[2]])
                    } else {
                        q
                    };
                    let m = pb.inverse_apply(w);
                    [m[0] - ob[0], m[1] - ob[1], m[2] - ob[2]]
                };
                points.push([l[0] as f32, l[1] as f32, l[2] as f32]);
            }
            let mut out: Vec<f32> = Vec::new();
            out.try_reserve_exact(points.len()).map_err(|_| oom())?;
            out.resize(points.len(), 0.0);
            signed_distances(mesh, bvh, sign, &points, bound, &mut out)?;
            for (j, &ci) in at.iter().enumerate() {
                let w = to_world * f64::from(out[j]);
                let x = if w.abs() > md {
                    f32::INFINITY
                } else {
                    w as f32
                };
                // Strictly smaller: a tie stays with the lower index.
                if x < best[ci].0 {
                    best[ci] = (x, k as u32);
                }
            }
        }
        values.try_reserve(candidates.len()).map_err(|_| oom())?;
        nearest.try_reserve(candidates.len()).map_err(|_| oom())?;
        for (ci, q) in frame.iter().enumerate() {
            let (x, n) = if q.is_some() {
                best[ci]
            } else {
                (f32::NAN, u32::MAX)
            };
            values.push(x);
            nearest.push(n);
        }
        Ok(fpts.len())
    }

    /// The new state of a prepared field from its candidates, their new
    /// `values` and the targets those came from (in candidate order): what
    /// changed, and the field's new bookkeeping. Reads the meshes only.
    fn finish_field(
        &self,
        job: &FieldJob,
        pick: Pick,
        values: Vec<f32>,
        nearest: Vec<u32>,
        recomputed: usize,
    ) -> Result<FieldPlan> {
        let f = &self.fields[job.i];
        let ma = &self.entry(f.desc.on)?.mesh;
        let slots = ma.vertex_slots();
        let defined = |v: VertexId| ma.is_vertex_alive(v) && ma.corner_count(v) > 0;
        let Pick {
            candidates,
            all,
            aborted,
            margin,
        } = pick;
        if values.len() != candidates.len() || nearest.len() != candidates.len() {
            return Err(SceneError::Corrupted);
        }

        let old = ma.channel(&f.channel).ok_or(SceneError::Corrupted)?;
        let changed: Vec<VertexId> = candidates
            .iter()
            .zip(&values)
            .filter(|&(&v, x)| old.get(v as usize).map(|o| o.to_bits()) != Some(x.to_bits()))
            .map(|(&v, _)| v)
            .collect();
        let referenced = (job.full || job.topology || f.referenced.len() != slots)
            .then(|| (0..slots as VertexId).map(defined).collect::<Vec<bool>>());
        let finite: Vec<VertexId> = {
            let now = candidates
                .iter()
                .zip(&values)
                .filter(|(_, x)| x.is_finite())
                .map(|(&v, _)| v);
            if job.live {
                // Unused until the update that ends the tier, which recomputes it.
                Vec::new()
            } else if all {
                now.collect()
            } else {
                let mut kept: Vec<VertexId> = f
                    .finite
                    .iter()
                    .copied()
                    .filter(|v| candidates.binary_search(v).is_err())
                    .chain(now)
                    .collect();
                kept.sort_unstable();
                kept
            }
        };
        Ok(FieldPlan {
            candidates,
            values,
            nearest,
            changed,
            referenced,
            finite,
            recomputed,
            aborted,
            margin,
        })
    }

    /// Writes a computed field: its values, its colours, its reports and its
    /// bookkeeping. Fails, writing nothing, only when the field's channel or
    /// mesh is missing, or memory runs out.
    fn apply_field(&mut self, job: FieldJob, plan: FieldPlan) -> Result<()> {
        let i = job.i;
        if job.live {
            let slots = self.entry(self.fields[i].desc.on)?.mesh.vertex_slots();
            let (start, swept) = self.live_sweep(i, slots);
            let f = &mut self.fields[i];
            if slots > 0 {
                f.live.sweep = ((start as usize + swept) % slots) as u32;
            }
            f.live.margin = plan.margin;
            f.live.travel = job.live_travel;
            let stamp = job.live_travel as f32;
            f.live.stamps.resize(slots, stamp);
            for &v in &plan.candidates {
                f.live.stamps[v as usize] = stamp;
            }
        }
        let (id, on) = (self.fields[i].id, self.fields[i].desc.on);
        let f = &mut self.fields[i];
        let ea = self.entries.get_mut(on).ok_or(SceneError::UnknownMesh)?;
        let slots = ea.mesh.vertex_slots();
        f.nearest
            .try_reserve(slots.saturating_sub(f.nearest.len()))
            .map_err(|_| oom())?;
        let channel = ea.mesh.derived_channel_mut(&f.channel)?;
        // Commit: nothing below fails.
        #[cfg(test)]
        {
            counter::add(plan.recomputed);
            counter::add_aborted(plan.aborted);
            counter::add_applied(id, plan.recomputed);
        }
        for (&v, &x) in plan.candidates.iter().zip(&plan.values) {
            channel[v as usize] = x;
        }
        f.nearest.resize(slots, u32::MAX);
        for (&v, &n) in plan.candidates.iter().zip(&plan.nearest) {
            f.nearest[v as usize] = n;
        }
        ea.reset_hook(id, false);
        if !plan.changed.is_empty() {
            ea.report_field(id);
        }
        if let Some(cs) = &mut ea.coloring
            && cs.coloring.field == id
        {
            let values = ea.mesh.channel(&cs.channel).unwrap_or(&[]);
            let dirty = cs.update(values, &plan.changed);
            ea.report_colors(Some(&dirty));
        }
        for (k, &m) in f.desc.to.iter().enumerate() {
            if let Some(eb) = self.entries.get_mut(m) {
                let baked = job.temp_targets.get(k).is_some_and(Option::is_some);
                eb.reset_hook(id, baked);
            }
        }
        f.fresh = false;
        f.computed = true;
        if job.live {
            f.live.stale = true;
        } else {
            f.live.stale = false;
            f.finite = plan.finite;
            f.live.travel = 0.0;
            f.live.stamps = Vec::new();
        }
        f.live.poses = job.b_to_a;
        if let Some(r) = plan.referenced {
            f.referenced = r;
        }
        Ok(())
    }
}

/// The [`par::map_chunks`] threshold for computing `jobs` one field per
/// task: 0 (in parallel) with the `parallel` feature when there are several
/// and their work adds up to [`PAR_MIN_FIELD_WORK`] or more, else
/// `usize::MAX`.
fn field_threshold<'a>(jobs: impl Iterator<Item = &'a FieldJob>) -> usize {
    let (n, work) = jobs.fold((0, 0), |(n, w), job| (n + 1, w + job.work));
    let parallel =
        work >= PAR_MIN_FIELD_WORK && n > 1 && par::enabled() && !par::force_sequential();
    #[cfg(test)]
    if parallel {
        counter::add_parallel();
    }
    if parallel { 0 } else { usize::MAX }
}

/// The most near pairs a field's near search may list for a mesh A of
/// `slots` vertex slots (see [`NEAR_PAIRS_PER_SLOT`]). Called on the thread
/// that runs the update, as the test override is per thread.
fn near_pair_limit(slots: usize) -> usize {
    #[cfg(test)]
    if let Some(limit) = counter::near_limit() {
        return limit;
    }
    slots.saturating_mul(NEAR_PAIRS_PER_SLOT)
}

/// Test-only counts of the vertices field updates recompute, and of the
/// batches of fields computed in parallel.
#[cfg(test)]
pub(crate) mod counter {
    use std::cell::Cell;

    // All are counted on the thread that runs the update (when fields are
    // applied), never on the pool's threads.
    thread_local! {
        static RECOMPUTED: Cell<usize> = const { Cell::new(0) };
        static PARALLEL: Cell<usize> = const { Cell::new(0) };
        static NEAR_LIMIT: Cell<Option<usize>> = const { Cell::new(None) };
        static ABORTED: Cell<usize> = const { Cell::new(0) };
        static FAILING: Cell<Option<super::FieldId>> = const { Cell::new(None) };
        static SIGN_STEPS: Cell<usize> = const { Cell::new(0) };
        static SHARED: Cell<usize> = const { Cell::new(0) };
    }

    /// Counts one shared pair query between reciprocal fields.
    pub(crate) fn add_shared() {
        SHARED.with(|c| c.set(c.get() + 1));
    }

    /// Shared pair queries on this thread since the last call.
    pub(crate) fn take_shared() -> usize {
        SHARED.with(|c| c.replace(0))
    }

    /// Counts one step of a stepped sign data build.
    pub(crate) fn add_sign_step() {
        SIGN_STEPS.with(|c| c.set(c.get() + 1));
    }

    /// Steps of stepped sign data builds on this thread since the last call.
    pub(crate) fn take_sign_steps() -> usize {
        SIGN_STEPS.with(|c| c.replace(0))
    }

    /// Makes computing the values of `field` fail (out of memory) on this
    /// thread; `None`: nothing fails.
    pub(crate) fn set_failing(field: Option<super::FieldId>) {
        FAILING.with(|c| c.set(field));
    }

    pub(crate) fn fails(field: super::FieldId) -> bool {
        FAILING.with(Cell::get) == Some(field)
    }

    thread_local! {
        static APPLIED: std::cell::RefCell<Vec<(super::FieldId, usize)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Records that `field` was applied (its update written), having
    /// recomputed `points` vertices.
    pub(crate) fn add_applied(field: super::FieldId, points: usize) {
        APPLIED.with(|c| c.borrow_mut().push((field, points)));
    }

    /// The fields applied on this thread since the last call, in order.
    pub(crate) fn take_applied() -> Vec<super::FieldId> {
        take_applied_counts().into_iter().map(|(f, _)| f).collect()
    }

    /// As [`take_applied`], with the vertices each recomputed.
    pub(crate) fn take_applied_counts() -> Vec<(super::FieldId, usize)> {
        APPLIED.with(|c| std::mem::take(&mut *c.borrow_mut()))
    }

    /// Counts one near search abandoned past its limit.
    pub(crate) fn add_aborted(aborted: bool) {
        ABORTED.with(|c| c.set(c.get() + usize::from(aborted)));
    }

    /// Near searches abandoned on this thread since the last call.
    pub(crate) fn take_aborted() -> usize {
        ABORTED.with(|c| c.replace(0))
    }

    /// Overrides the near-pair limit of field updates on this thread
    /// (`None`: the production limit).
    pub(crate) fn set_near_limit(limit: Option<usize>) {
        NEAR_LIMIT.with(|c| c.set(limit));
    }

    pub(crate) fn near_limit() -> Option<usize> {
        NEAR_LIMIT.with(Cell::get)
    }

    /// Counts `n` recomputed vertices.
    pub(crate) fn add(n: usize) {
        RECOMPUTED.with(|c| c.set(c.get() + n));
    }

    /// Vertices recomputed on this thread since the last call.
    pub(crate) fn take() -> usize {
        RECOMPUTED.with(|c| c.replace(0))
    }

    /// Counts one batch of fields computed in parallel.
    pub(crate) fn add_parallel() {
        PARALLEL.with(|c| c.set(c.get() + 1));
    }

    /// Batches of fields computed in parallel on this thread since the last
    /// call.
    pub(crate) fn take_parallel() -> usize {
        PARALLEL.with(|c| c.replace(0))
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_fields::{Colormap, OutOfRange, SignData, color_of, signed_distances};
    use fast_mesh_store::{Mesh, VertexId};

    use super::*;
    use crate::query::tests::cube;
    use crate::{MeshOptions, Placement, Scene, SceneError, SceneOptions, UpdateMode};

    /// A deterministic linear congruential generator.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next()
        }

        fn below(&mut self, n: usize) -> usize {
            ((self.next() * n as f64) as usize).min(n - 1)
        }
    }

    /// A flat `n` x `n` grid in the plane `z = 0`, `h` apart.
    fn sheet(n: u32, h: f32) -> Mesh {
        let mut pos = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                pos.extend_from_slice(&[x as f32 * h, y as f32 * h, 0.0]);
            }
        }
        let mut idx = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                let b = a + 1;
                let c = a + n + 1;
                idx.extend_from_slice(&[a, b, c, b, c + 1, c]);
            }
        }
        Mesh::from_buffers(&pos, &idx).unwrap()
    }

    fn with_mode(mode: UpdateMode) -> MeshOptions {
        MeshOptions {
            mode,
            ..MeshOptions::default()
        }
    }

    fn at(x: f64, y: f64, z: f64) -> Placement {
        Placement {
            translation: [x, y, z],
            ..Placement::IDENTITY
        }
    }

    /// A rotation by `deg` degrees about z, then a translation.
    fn turned(deg: f64, t: [f64; 3]) -> Placement {
        let (s, c) = deg.to_radians().sin_cos();
        Placement {
            rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            translation: t,
            ..Placement::IDENTITY
        }
    }

    fn field(name: &str, on: MeshId, to: MeshId, max_distance: f64) -> DistanceField {
        DistanceField {
            name: name.to_owned(),
            on,
            to: vec![to],
            max_distance,
            sign: SignMode::Auto,
        }
    }

    fn thick(name: &str, on: MeshId, max_distance: f64) -> ThicknessField {
        ThicknessField {
            name: name.to_owned(),
            on,
            max_distance,
        }
    }

    fn curv(name: &str, on: MeshId) -> CurvatureField {
        CurvatureField {
            name: name.to_owned(),
            on,
        }
    }

    /// A closed plate, 1 x 1 x 0.5 (`n` cells a side), facing out.
    fn plate(n: u32) -> Mesh {
        let h = 0.25;
        let w = n + 1;
        let mut pos = Vec::new();
        for z in [h, -h] {
            for y in 0..=n {
                for x in 0..=n {
                    pos.extend_from_slice(&[x as f32 / n as f32, y as f32 / n as f32, z]);
                }
            }
        }
        let t = |x: u32, y: u32| y * w + x;
        let b = |x: u32, y: u32| w * w + y * w + x;
        let mut idx = Vec::new();
        for y in 0..n {
            for x in 0..n {
                idx.extend_from_slice(&[
                    t(x, y),
                    t(x + 1, y),
                    t(x + 1, y + 1),
                    t(x, y),
                    t(x + 1, y + 1),
                    t(x, y + 1),
                    b(x, y),
                    b(x + 1, y + 1),
                    b(x + 1, y),
                    b(x, y),
                    b(x, y + 1),
                    b(x + 1, y + 1),
                ]);
            }
        }
        for x in 0..n {
            idx.extend_from_slice(&[
                t(x, 0),
                b(x, 0),
                b(x + 1, 0),
                t(x, 0),
                b(x + 1, 0),
                t(x + 1, 0),
                t(x, n),
                t(x + 1, n),
                b(x + 1, n),
                t(x, n),
                b(x + 1, n),
                b(x, n),
                t(0, x),
                t(0, x + 1),
                b(0, x + 1),
                t(0, x),
                b(0, x + 1),
                b(0, x),
                t(n, x),
                b(n, x),
                b(n, x + 1),
                t(n, x),
                b(n, x + 1),
                t(n, x + 1),
            ]);
        }
        Mesh::from_buffers(&pos, &idx).unwrap()
    }

    /// The unit sphere from a twice-subdivided octahedron, facing out: 128
    /// triangles, every vertex about `1 / 1` of curvature.
    fn ball() -> Mesh {
        let mut pos: Vec<[f64; 3]> = vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let mut tris: Vec<[u32; 3]> = vec![
            [4, 0, 2],
            [4, 2, 1],
            [4, 1, 3],
            [4, 3, 0],
            [5, 2, 0],
            [5, 1, 2],
            [5, 3, 1],
            [5, 0, 3],
        ];
        for _ in 0..2 {
            let mut mid = std::collections::HashMap::new();
            let mut midpoint = |a: u32, b: u32, pos: &mut Vec<[f64; 3]>| {
                *mid.entry((a.min(b), a.max(b))).or_insert_with(|| {
                    let (pa, pb) = (pos[a as usize], pos[b as usize]);
                    pos.push(std::array::from_fn(|k| (pa[k] + pb[k]) / 2.0));
                    (pos.len() - 1) as u32
                })
            };
            let mut next = Vec::new();
            for [a, b, c] in tris {
                let ab = midpoint(a, b, &mut pos);
                let bc = midpoint(b, c, &mut pos);
                let ca = midpoint(c, a, &mut pos);
                next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
            }
            tris = next;
        }
        let flat: Vec<f32> = pos
            .iter()
            .flat_map(|p| {
                let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                [p[0] / n, p[1] / n, p[2] / n].map(|c| c as f32)
            })
            .collect();
        let idx: Vec<u32> = tris.iter().flat_map(|t| *t).collect();
        Mesh::from_buffers(&flat, &idx).unwrap()
    }

    /// A saddle `z = (2 x^2 - y^2) / s` over `[-1, 1]^2`, `n` cells a side,
    /// facing up (normals `+z`): mean curvature about `-1 / s` at the middle.
    fn saddle(n: u32, s: f32) -> Mesh {
        let mut pos = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                let (fx, fy) = (
                    x as f32 / n as f32 * 2.0 - 1.0,
                    y as f32 / n as f32 * 2.0 - 1.0,
                );
                pos.extend_from_slice(&[fx, fy, (2.0 * fx * fx - fy * fy) / s]);
            }
        }
        let mut idx = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                idx.extend_from_slice(&[a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        Mesh::from_buffers(&pos, &idx).unwrap()
    }

    /// A thickness colour rule over `range`, clamping both ends.
    fn thick_rule(lo: f32, hi: f32) -> ColorRule {
        ColorRule {
            map: Colormap::Turbo,
            range: (lo, hi),
            below: OutOfRange::Clamp,
            above: OutOfRange::Clamp,
            center: None,
            base: [200, 200, 200, 255],
            nan: [128, 128, 128, 255],
        }
    }

    /// A unit sheet (A) cut by a cube (B), 0.4 wide, centred on it.
    fn sheet_and_cube(a_mode: UpdateMode) -> (Scene, MeshId, MeshId) {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(sheet(10, 0.1), with_mode(a_mode)).unwrap();
        let b = scene
            .add_mesh(cube([0.3, 0.3, -0.2], 0.4), MeshOptions::default())
            .unwrap();
        (scene, a, b)
    }

    /// The field's values computed from scratch: every vertex of A, mapped
    /// to the world and into B's space with the scene's point accessors (not
    /// the production transform), with fresh sign data for B. B's own BVH is
    /// used, which must be current: the fast winding number depends on the
    /// tree's shape, while the closest distance does not.
    fn reference(scene: &mut Scene, id: FieldId) -> Vec<f32> {
        let f = scene.fields.iter().find(|f| f.id == id).unwrap();
        let (on, to, md, mode) = (f.desc.on, f.desc.to[0], f.desc.max_distance, f.desc.sign);
        let ma = scene.mesh(on).unwrap();
        let slots = ma.vertex_slots();
        let defined: Vec<bool> = (0..slots as VertexId)
            .map(|v| ma.is_vertex_alive(v) && ma.corner_count(v) > 0)
            .collect();
        let points: Vec<[f32; 3]> = (0..slots as VertexId)
            .filter(|&v| defined[v as usize])
            .map(|v| {
                let p = ma.position(v).unwrap().map(f64::from);
                let w = scene.local_to_world(on, p).unwrap();
                scene.world_to_local(to, w).unwrap().map(|c| c as f32)
            })
            .collect();
        let eb = scene.entries.get_mut(to).unwrap();
        assert!(eb.is_current(), "the reference needs B's BVH current");
        let sign = SignData::build(&mut eb.mesh, &eb.bvh, mode).unwrap();
        let mut out = vec![0.0; points.len()];
        let bound = crate::query::round_up(md);
        signed_distances(&eb.mesh, &eb.bvh, &sign, &points, bound, &mut out).unwrap();
        let mut values = out.into_iter().map(|v| {
            if f64::from(v.abs()) > md {
                f32::INFINITY
            } else {
                v
            }
        });
        defined
            .iter()
            .map(|&d| if d { values.next().unwrap() } else { f32::NAN })
            .collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Surface tolerance for the reference, whose points take another
    /// rounding path (§11.9: relative to the scenes' unit size).
    const TOL: f32 = 1e-5;

    /// Equal bits; or both finite within `TOL`; or one far (`+∞`) and the
    /// other within `TOL` of `max_distance`. Both infinite must be equal, so
    /// that `+∞` never matches `-∞`.
    fn close(g: f32, w: f32, md: f64) -> bool {
        if g.to_bits() == w.to_bits() {
            return true;
        }
        if g.is_nan() || w.is_nan() {
            return false;
        }
        match (g.is_finite(), w.is_finite()) {
            (true, true) => (g - w).abs() <= TOL,
            (false, false) => g == w,
            _ => {
                let (finite, infinite) = if g.is_finite() { (g, w) } else { (w, g) };
                infinite == f32::INFINITY && f64::from(finite.abs()) >= md - f64::from(TOL)
            }
        }
    }

    fn assert_matches(scene: &mut Scene, id: FieldId) {
        let want = reference(scene, id);
        let md = scene
            .fields
            .iter()
            .find(|f| f.id == id)
            .unwrap()
            .desc
            .max_distance;
        let got = scene.field_values(id).unwrap().to_vec();
        assert_eq!(got.len(), want.len());
        for (v, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(close(*g, *w, md), "vertex {v}: got {g}, want {w}");
        }
        assert!(scene.field_status(id).unwrap().current);
    }

    fn assert_colors(scene: &Scene, mesh: MeshId, rule: &ColorRule, id: FieldId) {
        let values = scene.field_values(id).unwrap();
        let colors = scene.colors(mesh).unwrap().unwrap();
        assert_eq!(colors.len(), values.len());
        for (v, (c, &x)) in colors.iter().zip(values).enumerate() {
            assert_eq!(*c, color_of(rule, x), "vertex {v}, value {x}");
        }
    }

    fn contact_rule(d: f32) -> ColorRule {
        ColorRule {
            map: Colormap::Turbo,
            range: (-d, 0.0),
            below: OutOfRange::Clamp,
            above: OutOfRange::Base,
            center: None,
            base: [200, 200, 200, 255],
            nan: [255, 0, 255, 255],
        }
    }

    fn distance_rule(d: f32, g: f32) -> ColorRule {
        ColorRule {
            map: Colormap::Coolwarm,
            range: (-d, g),
            below: OutOfRange::Clamp,
            above: OutOfRange::Base,
            center: Some(0.0),
            base: [200, 200, 200, 255],
            nan: [255, 0, 255, 255],
        }
    }

    fn finite_count(scene: &Scene, id: FieldId) -> usize {
        scene
            .field_values(id)
            .unwrap()
            .iter()
            .filter(|v| v.is_finite())
            .count()
    }

    fn colors_dirty(scene: &mut Scene, mesh: MeshId) -> Option<Vec<VertexId>> {
        scene
            .take_changes()
            .into_iter()
            .find(|c| c.id == mesh)
            .map_or(Some(Vec::new()), |c| c.colors_dirty)
    }

    fn alive_vertex(scene: &Scene, m: MeshId, rng: &mut Lcg) -> Option<VertexId> {
        let mesh = scene.mesh(m).unwrap();
        let alive: Vec<VertexId> = (0..mesh.vertex_slots() as VertexId)
            .filter(|&v| mesh.is_vertex_alive(v))
            .collect();
        (!alive.is_empty()).then(|| alive[rng.below(alive.len())])
    }

    fn alive_triangle(scene: &Scene, m: MeshId, rng: &mut Lcg) -> Option<u32> {
        let mesh = scene.mesh(m).unwrap();
        let alive: Vec<u32> = (0..mesh.triangle_slots() as u32)
            .filter(|&t| mesh.triangle(t).is_some())
            .collect();
        (!alive.is_empty()).then(|| alive[rng.below(alive.len())])
    }

    fn random_mode(rng: &mut Lcg) -> UpdateMode {
        [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual][rng.below(3)]
    }

    fn random_sign(rng: &mut Lcg) -> SignMode {
        [SignMode::Pseudonormal, SignMode::Winding, SignMode::Auto][rng.below(3)]
    }

    fn random_max_distance(rng: &mut Lcg) -> f64 {
        if rng.below(5) == 0 {
            f64::INFINITY
        } else {
            rng.range(0.03, 0.25)
        }
    }

    /// One random operation of the walk on meshes `a` (sheet) and `b`
    /// (cube). Operations that do not apply (nothing to undo, no such
    /// vertex) are skipped.
    fn random_op(scene: &mut Scene, a: MeshId, b: MeshId, rng: &mut Lcg) {
        let m = if rng.below(2) == 0 { a } else { b };
        match rng.below(16) {
            0 | 1 => {
                let p = turned(
                    rng.range(-25.0, 25.0),
                    [
                        rng.range(-0.1, 0.1),
                        rng.range(-0.1, 0.1),
                        rng.range(-0.1, 0.1),
                    ],
                );
                scene.set_placement(m, p).unwrap();
            }
            2 => {
                if let Some(v) = alive_vertex(scene, a, rng) {
                    let z = rng.range(-0.2, 0.2) as f32;
                    scene
                        .edit(a, |tx| {
                            let p = tx.mesh().position(v).unwrap();
                            tx.set_positions(&[v], &[[p[0], p[1], z]])
                        })
                        .unwrap();
                }
            }
            3 => {
                if let Some(v) = alive_vertex(scene, b, rng) {
                    let s = rng.range(0.95, 1.05) as f32;
                    scene
                        .edit(b, |tx| {
                            let p = tx.mesh().position(v).unwrap();
                            tx.set_positions(&[v], &[p.map(|c| c * s)])
                        })
                        .unwrap();
                }
            }
            4 => {
                if let Some(t) = alive_triangle(scene, m, rng) {
                    scene.edit(m, |tx| tx.remove_triangles(&[t])).unwrap();
                }
            }
            5 => {
                // A new vertex on a new triangle with two alive vertices.
                let (Some(p), Some(q)) = (alive_vertex(scene, a, rng), alive_vertex(scene, a, rng))
                else {
                    return;
                };
                if p == q {
                    return;
                }
                let x = [rng.range(0.0, 1.0) as f32, rng.range(0.0, 1.0) as f32, 0.02];
                scene
                    .edit(a, |tx| {
                        let v = tx.add_vertices(&[x])?;
                        tx.add_triangles(&[[v[0], p, q]])
                    })
                    .unwrap();
            }
            6 => {
                // Removes an alive vertex no triangle uses, if any.
                let mesh = scene.mesh(m).unwrap();
                let unused = (0..mesh.vertex_slots() as VertexId)
                    .find(|&v| mesh.is_vertex_alive(v) && mesh.corner_count(v) == 0);
                if let Some(v) = unused {
                    scene.edit(m, |tx| tx.remove_vertices(&[v])).unwrap();
                }
            }
            7 => {
                // Splits a triangle at its centroid (a topology edit that
                // keeps a closed mesh closed).
                if let Some(t) = alive_triangle(scene, m, rng) {
                    scene
                        .edit(m, |tx| {
                            let [i, j, k] = tx.mesh().triangle(t).unwrap();
                            let ps = [i, j, k].map(|v| tx.mesh().position(v).unwrap());
                            let c = std::array::from_fn(|d| (ps[0][d] + ps[1][d] + ps[2][d]) / 3.0);
                            let v = tx.add_vertices(&[c])?[0];
                            tx.remove_triangles(&[t])?;
                            tx.add_triangles(&[[i, j, v], [j, k, v], [k, i, v]])
                        })
                        .unwrap();
                }
            }
            8 | 9 => {
                scene.undo().unwrap();
            }
            10 => {
                scene.redo().unwrap();
            }
            11 => {
                // A drag in one step.
                scene.begin_step();
                for k in 1..=3 {
                    let d = 0.01 * k as f64;
                    scene
                        .set_placement(m, turned(d * 50.0, [d, -d, d]))
                        .unwrap();
                }
                scene.end_step().unwrap();
            }
            12 => {
                // A drag cancelled half-way.
                scene.begin_step();
                scene.set_placement(m, at(0.03, 0.02, -0.04)).unwrap();
                if let Some(v) = alive_vertex(scene, a, rng) {
                    scene
                        .edit(a, |tx| {
                            let p = tx.mesh().position(v).unwrap();
                            tx.set_positions(&[v], &[[p[0], p[1], 0.1]])
                        })
                        .unwrap();
                }
                scene.cancel_step().unwrap();
            }
            13 => {
                scene.compact(m).unwrap();
            }
            14 => {
                scene.set_mode(m, random_mode(rng)).unwrap();
            }
            _ => {
                // Add a vertex, then remove it in the same batch (or let undo
                // take back an add): exercises slot growth without a net
                // change.
                let x = [rng.range(0.0, 1.0) as f32, rng.range(0.0, 1.0) as f32, 0.0];
                let Some(p) = alive_vertex(scene, a, rng) else {
                    return;
                };
                let Some(q) = alive_vertex(scene, a, rng).filter(|&q| q != p) else {
                    return;
                };
                scene
                    .edit(a, |tx| {
                        let v = tx.add_vertices(&[x])?;
                        tx.add_triangles(&[[v[0], p, q]])
                    })
                    .unwrap();
                scene.undo().unwrap();
            }
        }
    }

    /// `update_fields`, bringing stale Manual BVHs up to date when it asks.
    fn update_all(scene: &mut Scene) {
        for _ in 0..8 {
            match scene.update_fields() {
                Ok(()) => return,
                Err(SceneError::StaleBvh(id)) => scene.update(id).unwrap(),
                Err(e) => panic!("update_fields failed: {e}"),
            }
        }
        panic!("update_fields kept failing");
    }

    fn check_colors_reported(
        scene: &Scene,
        mesh: MeshId,
        before: &[[u8; 4]],
        dirty: Option<&Vec<VertexId>>,
        at: &str,
    ) {
        let now = scene.colors(mesh).unwrap().unwrap();
        assert_eq!(now.len(), scene.mesh(mesh).unwrap().vertex_slots(), "{at}");
        if let Some(dirty) = dirty {
            for (v, c) in now.iter().enumerate() {
                if before.get(v) != Some(c) {
                    assert!(
                        dirty.binary_search(&(v as VertexId)).is_ok(),
                        "{at}: vertex {v} not reported"
                    );
                }
            }
        }
    }

    #[test]
    fn field_matches_from_scratch() {
        let mut rng = Lcg(42);
        for round in 0..24 {
            let mut scene = Scene::new(SceneOptions::default());
            let a_mode = [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual][round % 3];
            let a = scene.add_mesh(sheet(10, 0.1), with_mode(a_mode)).unwrap();
            let b = scene
                .add_mesh(
                    cube([0.3, 0.3, -0.2], 0.4),
                    with_mode(random_mode(&mut rng)),
                )
                .unwrap();
            scene.update(b).unwrap();
            let fa = scene
                .add_distance_field(DistanceField {
                    sign: random_sign(&mut rng),
                    ..field("d", a, b, random_max_distance(&mut rng))
                })
                .unwrap();
            let fb = scene
                .add_distance_field(DistanceField {
                    sign: random_sign(&mut rng),
                    ..field("d", b, a, random_max_distance(&mut rng))
                })
                .unwrap();
            let (rule_a, rule_b) = (distance_rule(0.2, 0.1), contact_rule(0.1));
            for (m, f, rule) in [(a, fa, &rule_a), (b, fb, &rule_b)] {
                let rule = rule.clone();
                scene
                    .set_coloring(m, Some(Coloring { field: f, rule }))
                    .unwrap();
            }
            update_all(&mut scene);
            scene.take_changes();
            for step in 0..40 {
                let at = format!("round {round} step {step}");
                let before_a = scene.colors(a).unwrap().unwrap().to_vec();
                let before_b = scene.colors(b).unwrap().unwrap().to_vec();
                let values_a = bits_of(scene.color_values(a).unwrap().unwrap());
                let values_b = bits_of(scene.color_values(b).unwrap().unwrap());
                // Eager meshes see one operation at a time; others a batch.
                let ops = if a_mode == UpdateMode::Eager {
                    1
                } else {
                    1 + rng.below(5)
                };
                for _ in 0..ops {
                    random_op(&mut scene, a, b, &mut rng);
                    // The colour buffers always cover every vertex slot.
                    for m in [a, b] {
                        let n = scene.mesh(m).unwrap().vertex_slots();
                        assert_eq!(scene.colors(m).unwrap().unwrap().len(), n, "{at}");
                    }
                }
                update_all(&mut scene);
                for m in [a, b] {
                    // The reference reads B's BVH, which update_fields made
                    // current only if a field needed it.
                    scene.update(m).unwrap();
                }
                assert_matches(&mut scene, fa);
                assert_matches(&mut scene, fb);
                assert_colors(&scene, a, &rule_a, fa);
                assert_colors(&scene, b, &rule_b, fb);
                let ch = scene.take_changes();
                let dirty = |m: MeshId| {
                    ch.iter()
                        .find(|c| c.id == m)
                        .map_or(Some(Vec::new()), |c| c.colors_dirty.clone())
                };
                check_colors_reported(&scene, a, &before_a, dirty(a).as_ref(), &at);
                check_colors_reported(&scene, b, &before_b, dirty(b).as_ref(), &at);
                for (m, f, before) in [(a, fa, &values_a), (b, fb, &values_b)] {
                    let now = scene.color_values(m).unwrap().unwrap();
                    assert_eq!(
                        bits_of(now),
                        bits_of(scene.field_values(f).unwrap()),
                        "{at}"
                    );
                    check_values_reported(before, now, dirty(m).as_ref(), &at);
                }
            }
        }
    }

    /// An add undone before a Lazy field updates leaves nothing for the
    /// field to do, but the colours must still cover the new slot.
    #[test]
    fn colors_cover_slots_after_cancelled_add() {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Lazy);
        let f = scene.add_distance_field(field("d", a, b, 0.1)).unwrap();
        let rule = contact_rule(0.1);
        let coloring = Coloring {
            field: f,
            rule: rule.clone(),
        };
        scene.set_coloring(a, Some(coloring)).unwrap();
        scene.update_fields().unwrap();
        scene.take_changes();
        scene
            .edit(a, |tx| {
                let v = tx.add_vertices(&[[0.5, 0.5, 0.0]])?;
                tx.add_triangles(&[[v[0], 0, 1]])
            })
            .unwrap();
        assert_eq!(scene.colors(a).unwrap().unwrap().len(), 122);
        assert_eq!(scene.colors(a).unwrap().unwrap()[121], rule.nan);
        scene.undo().unwrap();
        scene.update_fields().unwrap();
        assert!(scene.field_status(f).unwrap().current);
        assert_eq!(scene.colors(a).unwrap().unwrap().len(), 122);
        assert_colors(&scene, a, &rule, f);
        assert_eq!(colors_dirty(&mut scene, a), Some(vec![121]));
    }

    fn bits_of(values: &[f32]) -> Vec<u32> {
        values.iter().map(|x| x.to_bits()).collect()
    }

    /// The vertices whose value bits differ between `before` and `now`
    /// (slots beyond `before` included) are all in `dirty` (`None`: all).
    fn check_values_reported(before: &[u32], now: &[f32], dirty: Option<&Vec<VertexId>>, at: &str) {
        if let Some(dirty) = dirty {
            for (v, x) in now.iter().enumerate() {
                if before.get(v) != Some(&x.to_bits()) {
                    assert!(
                        dirty.binary_search(&(v as VertexId)).is_ok(),
                        "{at}: value of vertex {v} not reported"
                    );
                }
            }
        }
    }

    /// The values behind the colours are the colouring field's channel, and
    /// change only where the colours are reported to change.
    #[test]
    fn color_values_view_the_coloring_field() {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        let f = scene.add_distance_field(field("d", a, b, 0.1)).unwrap();
        assert_eq!(scene.color_values(a), Ok(None));
        let rule = contact_rule(0.1);
        scene
            .set_coloring(
                a,
                Some(Coloring {
                    field: f,
                    rule: rule.clone(),
                }),
            )
            .unwrap();
        let view = |s: &Scene| bits_of(s.color_values(a).unwrap().unwrap());
        assert_eq!(view(&scene), bits_of(scene.field_values(f).unwrap()));
        assert_eq!(colors_dirty(&mut scene, a), None);
        // A drag: the values that changed are reported with the colours.
        let before = view(&scene);
        scene.set_placement(b, at(0.0, 0.0, 0.03)).unwrap();
        assert_eq!(view(&scene), bits_of(scene.field_values(f).unwrap()));
        assert_ne!(view(&scene), before);
        let dirty = colors_dirty(&mut scene, a);
        check_values_reported(
            &before,
            scene.color_values(a).unwrap().unwrap(),
            dirty.as_ref(),
            "drag",
        );
        // Colouring off: no values.
        scene.set_coloring(a, None).unwrap();
        assert_eq!(scene.color_values(a), Ok(None));

        // A reused slot: the store resets its value to NaN, and the colours
        // report it. A Lazy field leaves a value in the slot until it updates.
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Lazy);
        let f = scene.add_distance_field(field("d", a, b, 0.1)).unwrap();
        scene
            .set_coloring(
                a,
                Some(Coloring {
                    field: f,
                    rule: rule.clone(),
                }),
            )
            .unwrap();
        scene
            .edit(a, |tx| {
                let v = tx.add_vertices(&[[0.5, 0.5, 0.0]])?;
                tx.add_triangles(&[[v[0], 0, 1]])
            })
            .unwrap();
        scene.update_fields().unwrap();
        scene.undo().unwrap();
        // An edit drops the redo step, which releases the slot.
        scene
            .edit(a, |tx| {
                let p = tx.mesh().position(0).unwrap();
                tx.set_positions(&[0], &[p])
            })
            .unwrap();
        scene.take_changes();
        let before = view(&scene);
        assert!(!f32::from_bits(before[121]).is_nan());
        scene
            .edit(a, |tx| tx.add_vertices(&[[5.0, 5.0, 5.0]]))
            .unwrap();
        assert!(
            scene.mesh(a).unwrap().is_vertex_alive(121),
            "the released slot is reused"
        );
        assert_eq!(scene.mesh(a).unwrap().vertex_slots(), 122);
        assert_eq!(view(&scene), bits_of(scene.field_values(f).unwrap()));
        let dirty = colors_dirty(&mut scene, a);
        check_values_reported(
            &before,
            scene.color_values(a).unwrap().unwrap(),
            dirty.as_ref(),
            "reuse",
        );
        // The colour of the reused slot follows its value (NaN) too.
        assert_eq!(scene.colors(a).unwrap().unwrap()[121], rule.nan);
    }

    /// The same for a field measured to several targets: the values behind
    /// the colours are its channel, a move of either target reports the
    /// values it changes, and a reused slot's value (NaN) and colour are
    /// reported, with no nearest target.
    #[test]
    fn color_values_view_a_multi_target_field() {
        let rule = contact_rule(0.1);
        let colour = |scene: &mut Scene, a: MeshId, f: FieldId| {
            scene
                .set_coloring(
                    a,
                    Some(Coloring {
                        field: f,
                        rule: rule.clone(),
                    }),
                )
                .unwrap();
        };
        let view = |s: &Scene, a: MeshId| bits_of(s.color_values(a).unwrap().unwrap());

        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        let c = scene
            .add_mesh(cube([0.7, 0.7, -0.15], 0.2), MeshOptions::default())
            .unwrap();
        let f = scene
            .add_distance_field(DistanceField {
                to: vec![b, c],
                ..field("d", a, b, 0.1)
            })
            .unwrap();
        colour(&mut scene, a, f);
        assert_eq!(view(&scene, a), bits_of(scene.field_values(f).unwrap()));
        let nearest = scene.field_nearest(f).unwrap();
        assert!(
            nearest.contains(&0) && nearest.contains(&1),
            "both targets win vertices"
        );
        assert_eq!(colors_dirty(&mut scene, a), None);
        assert_colors(&scene, a, &rule, f);
        // Moving either target: the values that changed are reported with the colours.
        for (m, p) in [(b, at(0.0, 0.0, 0.03)), (c, at(-0.02, 0.0, 0.02))] {
            let before = view(&scene, a);
            scene.set_placement(m, p).unwrap();
            assert_eq!(view(&scene, a), bits_of(scene.field_values(f).unwrap()));
            assert_ne!(view(&scene, a), before);
            let dirty = colors_dirty(&mut scene, a);
            check_values_reported(
                &before,
                scene.color_values(a).unwrap().unwrap(),
                dirty.as_ref(),
                "move",
            );
            assert_colors(&scene, a, &rule, f);
        }
        assert_eq!(shown(&scene, f), min_of_singles(&mut scene, f));

        // A reused slot, as for one target (see above).
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Lazy);
        let c = scene
            .add_mesh(cube([0.7, 0.7, -0.15], 0.2), MeshOptions::default())
            .unwrap();
        let f = scene
            .add_distance_field(DistanceField {
                to: vec![b, c],
                ..field("d", a, b, 0.1)
            })
            .unwrap();
        colour(&mut scene, a, f);
        scene
            .edit(a, |tx| {
                let v = tx.add_vertices(&[[0.5, 0.5, 0.0]])?;
                tx.add_triangles(&[[v[0], 0, 1]])
            })
            .unwrap();
        scene.update_fields().unwrap();
        scene.undo().unwrap();
        scene
            .edit(a, |tx| {
                let p = tx.mesh().position(0).unwrap();
                tx.set_positions(&[0], &[p])
            })
            .unwrap();
        scene.take_changes();
        let before = view(&scene, a);
        assert!(!f32::from_bits(before[121]).is_nan());
        scene
            .edit(a, |tx| tx.add_vertices(&[[5.0, 5.0, 5.0]]))
            .unwrap();
        assert!(
            scene.mesh(a).unwrap().is_vertex_alive(121),
            "the released slot is reused"
        );
        assert_eq!(view(&scene, a), bits_of(scene.field_values(f).unwrap()));
        let dirty = colors_dirty(&mut scene, a);
        check_values_reported(
            &before,
            scene.color_values(a).unwrap().unwrap(),
            dirty.as_ref(),
            "reuse",
        );
        assert_eq!(scene.colors(a).unwrap().unwrap()[121], rule.nan);
        assert_eq!(shown(&scene, f).1[121], u32::MAX);
    }

    #[test]
    fn small_drag_recomputes_only_near_vertices() {
        let mut scene = Scene::new(SceneOptions::default());
        // 61 x 61 = 3721 vertices.
        let a = scene
            .add_mesh(sheet(60, 1.0 / 60.0), MeshOptions::default())
            .unwrap();
        let b = scene
            .add_mesh(cube([0.1, 0.1, -0.05], 0.1), MeshOptions::default())
            .unwrap();
        let f = scene.add_distance_field(field("d", a, b, 0.02)).unwrap();
        scene.update_fields().unwrap();
        assert_matches(&mut scene, f);
        assert!(finite_count(&scene, f) > 0);
        counter::take();
        scene.set_placement(b, at(0.002, 0.001, 0.0)).unwrap();
        scene.update_fields().unwrap();
        let recomputed = counter::take();
        assert!(recomputed > 0);
        assert!(recomputed < 300, "recomputed {recomputed} of 3721");
        assert_matches(&mut scene, f);
        // An infinite max_distance computes every vertex.
        let g = scene
            .add_distance_field(field("all", a, b, f64::INFINITY))
            .unwrap();
        scene.update_fields().unwrap();
        counter::take();
        scene.set_placement(b, at(0.003, 0.001, 0.0)).unwrap();
        scene.update_fields().unwrap();
        assert!(counter::take() >= 3721);
        assert_matches(&mut scene, g);
        assert!(scene.field_values(g).unwrap().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn reciprocal_fields_share_one_pair_query() {
        // Two-sided colouring with the same max_distance: one shared pair
        // query per pose, same values as separate queries.
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(sheet(20, 0.05), MeshOptions::default())
            .unwrap();
        let b = scene
            .add_mesh(cube([0.2, 0.2, -0.05], 0.2), MeshOptions::default())
            .unwrap();
        let fa = scene.add_distance_field(field("d", a, b, 0.1)).unwrap();
        let fb = scene.add_distance_field(field("d", b, a, 0.1)).unwrap();
        scene.update_fields().unwrap();
        scene.take_changes();
        counter::take_shared();
        scene.set_placement(b, at(0.01, 0.005, 0.0)).unwrap();
        scene.update_fields().unwrap();
        assert_eq!(counter::take_shared(), 1);
        assert_matches(&mut scene, fa);
        assert_matches(&mut scene, fb);
    }

    /// Six sheets of `n` x `n` cells in `mode`, stacked `gap` apart, with a
    /// field for every ordered pair (30, all within reach), each mesh
    /// coloured by its field to the next mesh by index.
    fn six_sheets(n: u32, gap: f64, mode: UpdateMode) -> (Scene, Vec<MeshId>) {
        let mut scene = Scene::new(SceneOptions::default());
        let ids: Vec<MeshId> = (0..6)
            .map(|k| {
                let opts = MeshOptions {
                    mode,
                    placement: at(0.0, 0.0, k as f64 * gap),
                    ..MeshOptions::default()
                };
                scene.add_mesh(sheet(n, 1.0 / n as f32), opts).unwrap()
            })
            .collect();
        for (i, &a) in ids.iter().enumerate() {
            for (j, &b) in ids.iter().enumerate() {
                if i != j {
                    let id = scene
                        .add_distance_field(field(&format!("to{j}"), a, b, 8.0 * gap))
                        .unwrap();
                    if j == (i + 1) % 6 {
                        let rule = distance_rule(8.0 * gap as f32, 8.0 * gap as f32);
                        let coloring = Coloring { field: id, rule };
                        scene.set_coloring(a, Some(coloring)).unwrap();
                    }
                }
            }
        }
        scene.update_fields().unwrap();
        (scene, ids)
    }

    /// Per pose of a drag of the first mesh: the reports, every field's
    /// values and every mesh's colours.
    type Drag = Vec<(Vec<crate::MeshChanges>, Vec<Vec<u32>>, Vec<Vec<[u8; 4]>>)>;

    fn drag(n: u32, mode: UpdateMode) -> Drag {
        let gap = 0.01;
        let (mut scene, ids) = six_sheets(n, gap, mode);
        scene.take_changes();
        let poses = [
            turned(0.5, [0.003, 0.001, 0.002]),
            turned(-1.0, [0.01, -0.004, 0.025]),
            turned(2.0, [0.0, 0.002, 0.051]),
            at(0.0, 0.0, 0.0),
        ];
        let mut out = Vec::new();
        for p in poses {
            scene.set_placement(ids[0], p).unwrap();
            scene.update_fields().unwrap();
            let changes = scene.take_changes();
            let ids_f: Vec<FieldId> = scene.fields.iter().map(|f| f.id).collect();
            let values = ids_f
                .iter()
                .map(|&f| bits(scene.field_values(f).unwrap()))
                .collect();
            let colors = ids
                .iter()
                .map(|&m| scene.colors(m).unwrap().unwrap().to_vec())
                .collect();
            out.push((changes, values, colors));
        }
        out
    }

    #[test]
    fn parallel_field_updates_report_in_order() {
        use fast_mesh_store::par;
        // The forced-sequential run is the code a feature-off build runs.
        for mode in [UpdateMode::Eager, UpdateMode::Lazy] {
            par::set_force_sequential(true);
            let reference = drag(24, mode);
            par::set_force_sequential(false);
            assert!(
                reference.iter().all(|(c, _, _)| c.len() == 6),
                "{mode:?}: every mesh reports"
            );
            if !par::enabled() {
                // Every drag below would be this one again.
                continue;
            }
            for k in [1, 2, 8] {
                let got = par::with_threads(k, || drag(24, mode));
                assert!(got == reference, "{mode:?}: {k} threads");
            }
            counter::take_parallel();
            assert!(drag(24, mode) == reference, "{mode:?}: global pool");
            assert!(counter::take_parallel() > 0, "{mode:?}: ran sequentially");
        }
    }

    #[test]
    fn small_inputs_stay_sequential() {
        use fast_mesh_store::par;
        let before = par::parallel_entries();
        counter::take_parallel();
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        let f = scene.add_distance_field(field("d", a, b, 0.1)).unwrap();
        scene.add_distance_field(field("d", b, a, 0.1)).unwrap();
        for k in 0..4 {
            scene
                .set_placement(b, at(0.01 * k as f64, 0.0, 0.0))
                .unwrap();
        }
        scene.update_fields().unwrap();
        assert_matches(&mut scene, f);
        assert_eq!(par::parallel_entries(), before);
        assert_eq!(counter::take_parallel(), 0);
    }

    #[test]
    fn near_work_counts_only_meshes_in_reach() {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Lazy);
        scene.add_distance_field(field("d", a, b, 0.1)).unwrap();
        scene.update_fields().unwrap();
        let tris = scene.mesh(a).unwrap().triangle_count();
        // The sheet has fewer vertices than triangles, so only the near
        // search can bring the estimate up to its triangle count.
        assert!(scene.mesh(a).unwrap().vertex_count() < tris);
        for (x, in_reach) in [(0.01, true), (100.0, false)] {
            scene.set_placement(b, at(x, 0.0, 0.0)).unwrap();
            let job = scene.prepare_field(0).unwrap();
            assert!(job.near_margins[0].is_some());
            assert_eq!(job.work >= tris, in_reach, "x = {x}");
        }
    }

    #[test]
    fn leaving_contact_restores_far() {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        let f = scene.add_distance_field(field("d", a, b, 0.1)).unwrap();
        assert_matches(&mut scene, f);
        let touching = scene.field_values(f).unwrap().to_vec();
        assert!(touching.iter().any(|v| *v < 0.0));
        assert!(touching.iter().any(|v| *v > 0.0 && v.is_finite()));
        scene.set_placement(b, at(10.0, 0.0, 0.0)).unwrap();
        assert_matches(&mut scene, f);
        assert!(
            scene
                .field_values(f)
                .unwrap()
                .iter()
                .all(|v| *v == f32::INFINITY)
        );
        scene.set_placement(b, Placement::IDENTITY).unwrap();
        assert_matches(&mut scene, f);
        assert_eq!(bits(scene.field_values(f).unwrap()), bits(&touching));
    }

    #[test]
    fn undo_keeps_field_consistent() {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        let f = scene.add_distance_field(field("d", a, b, 0.15)).unwrap();
        let g = scene.add_distance_field(field("d", b, a, 0.15)).unwrap();
        let initial = bits(scene.field_values(f).unwrap());
        // Removing a triangle unreferences vertex 0 (only in triangle 0).
        scene.edit(a, |tx| tx.remove_triangles(&[0])).unwrap();
        assert!(scene.field_values(f).unwrap()[0].is_nan());
        scene.edit(a, |tx| tx.remove_vertices(&[0])).unwrap();
        scene
            .edit(a, |tx| tx.set_positions(&[45], &[[0.5, 0.4, 0.3]]))
            .unwrap();
        scene
            .edit(a, |tx| {
                let v = tx.add_vertices(&[[0.45, 0.45, -0.05]])?;
                tx.add_triangles(&[[v[0], 44, 45]])
            })
            .unwrap();
        scene.set_placement(b, at(0.05, 0.0, 0.02)).unwrap();
        for _ in 0..5 {
            assert_matches(&mut scene, f);
            assert_matches(&mut scene, g);
            scene.undo().unwrap();
        }
        assert_matches(&mut scene, f);
        assert_matches(&mut scene, g);
        // The undone vertex leaves a dead slot behind, which holds NaN.
        let back = scene.field_values(f).unwrap();
        assert_eq!(bits(&back[..initial.len()]), initial);
        assert!(back[initial.len()..].iter().all(|v| v.is_nan()));
        for _ in 0..5 {
            scene.redo().unwrap();
            assert_matches(&mut scene, f);
            assert_matches(&mut scene, g);
        }
        // Cancelling an open step restores the values too.
        scene.begin_step();
        scene.set_placement(b, at(0.0, 0.0, 0.5)).unwrap();
        scene.cancel_step().unwrap();
        assert_matches(&mut scene, f);
    }

    #[test]
    fn manual_bvh_blocks_field_update() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(sheet(10, 0.1), MeshOptions::default())
            .unwrap();
        let b = scene
            .add_mesh(cube([0.3, 0.3, -0.2], 0.4), with_mode(UpdateMode::Manual))
            .unwrap();
        let f = scene.add_distance_field(field("d", a, b, 0.1)).unwrap();
        assert_matches(&mut scene, f);
        scene
            .edit(b, |tx| tx.set_positions(&[7], &[[0.8, 0.8, 0.3]]))
            .unwrap();
        // The Eager field could not update: it is left stale.
        assert!(!scene.field_status(f).unwrap().current);
        assert_eq!(scene.update_fields(), Err(SceneError::StaleBvh(b)));
        assert_eq!(scene.flush(), Err(SceneError::StaleBvh(b)));
        assert!(!scene.field_status(f).unwrap().current);
        scene.update(b).unwrap();
        scene.update_fields().unwrap();
        assert_matches(&mut scene, f);
        // Registering a field on a stale Manual target fails.
        scene
            .edit(b, |tx| tx.set_positions(&[7], &[[0.7, 0.7, 0.2]]))
            .unwrap();
        assert_eq!(
            scene.add_distance_field(field("e", a, b, 0.1)),
            Err(SceneError::StaleBvh(b))
        );
    }

    #[test]
    fn eager_vs_lazy_field_scheduling() {
        for mode in [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual] {
            let (mut scene, a, b) = sheet_and_cube(mode);
            let f = scene.add_distance_field(field("d", a, b, 0.1)).unwrap();
            assert_eq!(
                scene.field_status(f).unwrap().current,
                mode == UpdateMode::Eager
            );
            scene.update_fields().unwrap();
            scene.set_placement(b, at(0.05, 0.0, 0.0)).unwrap();
            assert_eq!(
                scene.field_status(f).unwrap().current,
                mode == UpdateMode::Eager
            );
            scene.flush().unwrap();
            assert_eq!(
                scene.field_status(f).unwrap().current,
                mode != UpdateMode::Manual
            );
            scene.update_fields().unwrap();
            assert_matches(&mut scene, f);
            // Edits of A schedule the same way.
            scene
                .edit(a, |tx| tx.set_positions(&[45], &[[0.5, 0.4, 0.05]]))
                .unwrap();
            assert_eq!(
                scene.field_status(f).unwrap().current,
                mode == UpdateMode::Eager
            );
            if mode == UpdateMode::Manual {
                scene.update(a).unwrap();
            }
            // An unrelated mesh leaves the field alone.
            let c = scene
                .add_mesh(sheet(1, 1.0), MeshOptions::default())
                .unwrap();
            scene.update_fields().unwrap();
            scene.set_placement(c, at(3.0, 0.0, 0.0)).unwrap();
            assert!(scene.field_status(f).unwrap().current);
            assert_matches(&mut scene, f);
        }
    }

    #[test]
    fn colors_follow_values_and_report_dirty() {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        let f = scene.add_distance_field(field("d", a, b, 0.1)).unwrap();
        let s = scene.subscribe();
        scene.take_changes();
        assert_eq!(scene.colors(a), Ok(None));
        let rule = contact_rule(0.1);
        scene
            .set_coloring(
                a,
                Some(Coloring {
                    field: f,
                    rule: rule.clone(),
                }),
            )
            .unwrap();
        assert_colors(&scene, a, &rule, f);
        assert_eq!(colors_dirty(&mut scene, a), None);
        let before = scene.field_values(f).unwrap().to_vec();
        scene.set_placement(b, at(0.0, 0.0, 0.03)).unwrap();
        assert_colors(&scene, a, &rule, f);
        let after = scene.field_values(f).unwrap().to_vec();
        let changed: Vec<VertexId> = (0..after.len() as VertexId)
            .filter(|&v| before[v as usize].to_bits() != after[v as usize].to_bits())
            .collect();
        assert!(!changed.is_empty());
        let ch = scene.take_changes();
        let mine = ch.iter().find(|c| c.id == a).unwrap();
        assert_eq!(mine.colors_dirty.as_deref(), Some(&changed[..]));
        assert_eq!(mine.fields_changed, [f]);
        let theirs = scene.take_changes_for(s).unwrap();
        let theirs = theirs.iter().find(|c| c.id == a).unwrap();
        assert_eq!(theirs.colors_dirty, None, "the colouring was set since");
        assert_eq!(theirs.fields_changed, [f]);
        // Nothing changed, nothing reported.
        scene.update_fields().unwrap();
        assert!(scene.take_changes().is_empty());
        // Changing the colouring recolours everything.
        let other = distance_rule(0.1, 0.05);
        scene
            .set_coloring(
                a,
                Some(Coloring {
                    field: f,
                    rule: other.clone(),
                }),
            )
            .unwrap();
        assert_colors(&scene, a, &other, f);
        assert_eq!(colors_dirty(&mut scene, a), None);
        // Bad colourings are rejected and change nothing.
        let mut bad = other.clone();
        bad.range = (1.0, -1.0);
        assert!(matches!(
            scene.set_coloring(
                a,
                Some(Coloring {
                    field: f,
                    rule: bad
                })
            ),
            Err(SceneError::InvalidInput(_))
        ));
        assert!(matches!(
            scene.set_coloring(
                b,
                Some(Coloring {
                    field: f,
                    rule: other.clone()
                })
            ),
            Err(SceneError::InvalidInput(_))
        ));
        assert_colors(&scene, a, &other, f);
        assert!(scene.take_changes().is_empty());
        scene.set_coloring(a, None).unwrap();
        assert_eq!(scene.colors(a), Ok(None));
        assert_eq!(colors_dirty(&mut scene, a), None);
    }

    #[test]
    fn late_subscriber_gets_full_colour_report() {
        // A subscriber added after a colouring exists gets `colors_dirty:
        // None` for the coloured mesh, so it knows to read every colour.
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        let f = scene.add_distance_field(field("d", a, b, 0.1)).unwrap();
        let rule = contact_rule(0.1);
        scene
            .set_coloring(
                a,
                Some(Coloring {
                    field: f,
                    rule: rule.clone(),
                }),
            )
            .unwrap();
        scene.take_changes();
        let s = scene.subscribe();
        let changes = scene.take_changes_for(s).unwrap();
        let mine = changes.iter().find(|c| c.id == a).unwrap();
        assert_eq!(mine.colors_dirty, None);
        // An uncoloured mesh still starts clean.
        assert!(changes.iter().find(|c| c.id == b).is_none());
    }

    #[test]
    fn coloring_getter_and_field_list() {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        assert_eq!(scene.coloring(a).unwrap(), None);
        assert_eq!(scene.fields_on(a).unwrap(), vec![]);
        let f1 = scene.add_distance_field(field("d1", a, b, 0.1)).unwrap();
        let f2 = scene.add_distance_field(field("d2", a, b, 0.2)).unwrap();
        assert_eq!(scene.fields_on(a).unwrap(), vec![f1, f2]);
        assert_eq!(scene.fields_on(b).unwrap(), vec![]);
        let rule = contact_rule(0.1);
        let coloring = Coloring {
            field: f1,
            rule: rule.clone(),
        };
        scene.set_coloring(a, Some(coloring.clone())).unwrap();
        assert_eq!(scene.coloring(a).unwrap(), Some(&coloring));
        scene.set_coloring(a, None).unwrap();
        assert_eq!(scene.coloring(a).unwrap(), None);
        scene.remove_field(f1).unwrap();
        assert_eq!(scene.fields_on(a).unwrap(), vec![f2]);
    }

    #[test]
    fn reconfigure_field_keeps_id_and_colouring() {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        let f = scene.add_distance_field(field("d", a, b, 0.1)).unwrap();
        let rule = contact_rule(0.1);
        scene
            .set_coloring(
                a,
                Some(Coloring {
                    field: f,
                    rule: rule.clone(),
                }),
            )
            .unwrap();
        scene.take_changes();
        let before = scene.field_values(f).unwrap().to_vec();
        // Widening max_distance keeps the id and the colouring, and brings
        // more vertices into range.
        scene.reconfigure_field(f, 0.5, SignMode::Auto).unwrap();
        assert_eq!(scene.fields_on(a).unwrap(), vec![f]);
        assert!(scene.coloring(a).unwrap().is_some());
        // Eager: recomputed at once.
        assert!(scene.field_status(f).unwrap().current);
        let after = scene.field_values(f).unwrap().to_vec();
        assert_ne!(before, after);
        let finite_before = before.iter().filter(|v| v.is_finite()).count();
        let finite_after = after.iter().filter(|v| v.is_finite()).count();
        assert!(finite_after >= finite_before);
        // Invalid inputs are rejected and change nothing.
        assert!(matches!(
            scene.reconfigure_field(f, f64::NAN, SignMode::Auto),
            Err(SceneError::InvalidInput(_))
        ));
        assert!(scene.field_status(f).unwrap().current);
        // Same values: a no-op.
        let values = scene.field_values(f).unwrap().to_vec();
        scene.reconfigure_field(f, 0.5, SignMode::Auto).unwrap();
        assert!(scene.field_status(f).unwrap().current);
        assert_eq!(scene.field_values(f).unwrap(), &values[..]);
    }

    #[test]
    fn one_sided_and_two_sided_coloring() {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        let d = 0.1;
        let fa = scene.add_distance_field(field("d", a, b, 0.2)).unwrap();
        let rule = contact_rule(d as f32);
        scene
            .set_coloring(
                a,
                Some(Coloring {
                    field: fa,
                    rule: rule.clone(),
                }),
            )
            .unwrap();
        assert_eq!(scene.colors(b), Ok(None));
        assert_colors(&scene, a, &rule, fa);
        // Inside vertices take the colormap, outside ones the base colour.
        let values = scene.field_values(fa).unwrap().to_vec();
        let colors = scene.colors(a).unwrap().unwrap().to_vec();
        for (v, c) in values.iter().zip(&colors) {
            if *v > 0.0 {
                assert_eq!(*c, rule.base);
            }
        }
        assert!(values.iter().any(|v| *v < 0.0));
        // Two-sided: B gets its own field to A and its own colouring.
        let fb = scene.add_distance_field(field("d", b, a, 0.2)).unwrap();
        let other = distance_rule(0.1, 0.1);
        scene
            .set_coloring(
                b,
                Some(Coloring {
                    field: fb,
                    rule: other.clone(),
                }),
            )
            .unwrap();
        scene.set_placement(b, at(0.02, 0.0, 0.05)).unwrap();
        assert_colors(&scene, a, &rule, fa);
        assert_colors(&scene, b, &other, fb);
        assert_matches(&mut scene, fa);
        assert_matches(&mut scene, fb);
    }

    #[test]
    fn compact_recomputes_everything() {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Lazy);
        let f = scene.add_distance_field(field("d", a, b, 0.1)).unwrap();
        let rule = contact_rule(0.1);
        scene
            .set_coloring(
                a,
                Some(Coloring {
                    field: f,
                    rule: rule.clone(),
                }),
            )
            .unwrap();
        scene.edit(a, |tx| tx.remove_triangles(&[0])).unwrap();
        scene.edit(a, |tx| tx.remove_vertices(&[0])).unwrap();
        scene.update_fields().unwrap();
        scene.take_changes();
        scene.compact(a).unwrap();
        let ch = scene.take_changes();
        let mine = ch.iter().find(|c| c.id == a).unwrap();
        assert!(mine.changes.is_none());
        assert_eq!(mine.colors_dirty, None);
        // The colours follow the remapped values right away.
        assert_eq!(scene.mesh(a).unwrap().vertex_slots(), 120);
        assert_colors(&scene, a, &rule, f);
        assert!(!scene.field_status(f).unwrap().current);
        counter::take();
        scene.update_fields().unwrap();
        assert_eq!(counter::take(), 120);
        assert_matches(&mut scene, f);
        assert_colors(&scene, a, &rule, f);
        // Compacting B recomputes too.
        scene.compact(b).unwrap();
        scene.update_fields().unwrap();
        assert_eq!(counter::take(), 120);
        assert_matches(&mut scene, f);
    }

    #[test]
    fn removing_a_mesh_removes_its_fields() {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        let fa = scene.add_distance_field(field("d", a, b, 0.1)).unwrap();
        let fb = scene.add_distance_field(field("d", b, a, 0.1)).unwrap();
        scene
            .set_coloring(
                a,
                Some(Coloring {
                    field: fa,
                    rule: contact_rule(0.1),
                }),
            )
            .unwrap();
        scene.take_changes();
        scene.remove_mesh(b).unwrap();
        assert_eq!(scene.field_values(fa), Err(SceneError::UnknownField));
        assert_eq!(scene.field_values(fb), Err(SceneError::UnknownField));
        assert_eq!(scene.field_status(fa), Err(SceneError::UnknownField));
        assert!(scene.mesh(a).unwrap().channel("field:d").is_none());
        assert_eq!(scene.colors(a), Ok(None));
        let ch = scene.take_changes();
        let mine = ch.iter().find(|c| c.id == a).unwrap();
        assert_eq!(mine.fields_changed, [fa]);
        assert_eq!(mine.colors_dirty, None);
        assert_eq!(scene.remove_field(fa), Err(SceneError::UnknownField));
        // A field removed by hand goes the same way; the name is free again.
        let c = scene
            .add_mesh(cube([0.3, 0.3, -0.2], 0.4), MeshOptions::default())
            .unwrap();
        let g = scene.add_distance_field(field("d", a, c, 0.1)).unwrap();
        assert_eq!(scene.field_id(g.raw()), Some(g));
        assert_eq!(scene.field_id(fa.raw()), None);
        scene.remove_field(g).unwrap();
        assert_eq!(scene.field_id(g.raw()), None);
        assert_eq!(scene.field_values(g), Err(SceneError::UnknownField));
        assert!(scene.mesh(a).unwrap().channel("field:d").is_none());
        scene.add_distance_field(field("d", a, c, 0.1)).unwrap();
        // Editing goes on without the removed fields.
        scene
            .edit(a, |tx| tx.set_positions(&[45], &[[0.5, 0.4, 0.05]]))
            .unwrap();
        scene.update_fields().unwrap();
    }

    /// Everything a renderer sees of two meshes after an update: every
    /// field's value bits, both meshes' colours, and the drained reports.
    type Seen = (
        Vec<Vec<u32>>,
        Vec<Option<Vec<[u8; 4]>>>,
        Vec<crate::MeshChanges>,
    );

    fn seen(scene: &mut Scene, meshes: [MeshId; 2]) -> Seen {
        let ids: Vec<FieldId> = scene.fields.iter().map(|f| f.id).collect();
        let values = ids
            .iter()
            .map(|&f| bits(scene.field_values(f).unwrap()))
            .collect();
        let colors = meshes
            .iter()
            .map(|&m| scene.colors(m).unwrap().map(<[_]>::to_vec))
            .collect();
        (values, colors, scene.take_changes())
    }

    /// A sheet (A) and a cube (B) with a field each way, both coloured,
    /// with random modes, signs and `max_distance`s (infinite ones too).
    fn twin_scene(rng: &mut Lcg) -> (Scene, MeshId, MeshId) {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(sheet(10, 0.1), with_mode(random_mode(rng)))
            .unwrap();
        let b = scene
            .add_mesh(cube([0.3, 0.3, -0.2], 0.4), with_mode(random_mode(rng)))
            .unwrap();
        for (on, to, rule) in [(a, b, distance_rule(0.2, 0.1)), (b, a, contact_rule(0.1))] {
            let id = scene
                .add_distance_field(DistanceField {
                    sign: random_sign(rng),
                    ..field("d", on, to, random_max_distance(rng))
                })
                .unwrap();
            scene
                .set_coloring(on, Some(Coloring { field: id, rule }))
                .unwrap();
        }
        (scene, a, b)
    }

    /// Runs the same random walk on two copies of a random scene, the first
    /// with the near-pair limit `on`, the second with `off`, and asserts
    /// after every update that both show the same values, colours and
    /// reports, bit for bit. Returns, per round, the near searches each copy
    /// abandoned, and whether the round has a field with a finite
    /// `max_distance` (only those search).
    fn assert_twins_agree(
        seed: u64,
        rounds: usize,
        on: usize,
        off: usize,
    ) -> Vec<([usize; 2], bool)> {
        let mut rng = Lcg(seed);
        let mut stats = Vec::new();
        for round in 0..rounds {
            let state = rng.0;
            let mut aborted = [0; 2];
            let mut twins = [(on, Lcg(state)), (off, Lcg(state))].map(|(limit, mut r)| {
                counter::set_near_limit(Some(limit));
                let (scene, a, b) = twin_scene(&mut r);
                (limit, r, scene, a, b)
            });
            rng.0 = twins[0].1.0;
            let finite = twins[0]
                .2
                .fields
                .iter()
                .any(|f| f.desc.max_distance.is_finite());
            counter::take_aborted();
            for step in 0..30 {
                let ops = 1 + rng.below(4);
                let mut out = Vec::new();
                for (k, (limit, r, scene, a, b)) in twins.iter_mut().enumerate() {
                    counter::set_near_limit(Some(*limit));
                    for _ in 0..ops {
                        random_op(scene, *a, *b, r);
                    }
                    update_all(scene);
                    out.push(seen(scene, [*a, *b]));
                    aborted[k] += counter::take_aborted();
                }
                assert_eq!(out[0], out[1], "round {round} step {step}");
            }
            stats.push((aborted, finite));
        }
        counter::set_near_limit(None);
        stats
    }

    #[test]
    fn near_limit_matches_near_path() {
        // Recomputing every vertex once there is any near pair, against
        // always listing them.
        let stats = assert_twins_agree(0xb0b, 12, 0, usize::MAX);
        // The recompute path ran in every round that searches, and never
        // without a limit.
        let ran: Vec<bool> = stats.iter().map(|&([on, _], _)| on > 0).collect();
        let searching: Vec<bool> = stats.iter().map(|&(_, finite)| finite).collect();
        assert_eq!(ran, searching);
        let off: Vec<usize> = stats.iter().map(|&([_, off], _)| off).collect();
        assert_eq!(off, vec![0; stats.len()]);
        assert!(
            searching.iter().filter(|&&f| f).count() >= 10,
            "{searching:?}"
        );
    }

    #[test]
    fn wide_reach_recomputes_every_vertex() {
        let mut scene = Scene::new(SceneOptions::default());
        // 61 x 61 = 3721 vertices, all within reach of the cube.
        let a = scene
            .add_mesh(sheet(60, 1.0 / 60.0), MeshOptions::default())
            .unwrap();
        let b = scene
            .add_mesh(cube([0.4, 0.4, -0.05], 0.1), MeshOptions::default())
            .unwrap();
        let f = scene.add_distance_field(field("d", a, b, 2.0)).unwrap();
        scene.update_fields().unwrap();
        counter::take();
        scene.set_placement(b, at(0.002, 0.001, 0.0)).unwrap();
        scene.update_fields().unwrap();
        assert_eq!(counter::take(), 3721);
        assert_matches(&mut scene, f);
    }

    #[test]
    fn invalid_field_descs_rejected() {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        let bad = |e| matches!(e, Err(SceneError::InvalidInput(_)));
        assert!(bad(scene.add_distance_field(field("d", a, a, 0.1))));
        assert!(bad(scene.add_distance_field(field("d", a, b, -1.0))));
        assert!(bad(scene.add_distance_field(field("d", a, b, f64::NAN))));
        scene.add_distance_field(field("d", a, b, 0.1)).unwrap();
        assert!(bad(scene.add_distance_field(field("d", a, b, 0.2))));
        // The same name on another mesh is fine.
        scene.add_distance_field(field("d", b, a, 0.1)).unwrap();
        let gone = scene
            .add_mesh(sheet(1, 1.0), MeshOptions::default())
            .unwrap();
        scene.remove_mesh(gone).unwrap();
        assert_eq!(
            scene.add_distance_field(field("e", a, gone, 0.1)),
            Err(SceneError::UnknownMesh)
        );
        assert_eq!(
            scene.add_distance_field(field("e", gone, a, 0.1)),
            Err(SceneError::UnknownMesh)
        );
        assert_eq!(scene.fields.len(), 2);
        // The reserved channels are not reachable through the scene.
        assert!(matches!(
            scene.derived_channel_mut(a, "field:d"),
            Err(SceneError::InvalidInput(_))
        ));
    }

    /// Steps units with `Points(budget)` until none is left, bringing stale
    /// Manual BVHs up to date when a step asks, as [`update_all`] does.
    /// Returns the steps that left a unit pending.
    fn step_all(scene: &mut Scene, budget: usize) -> usize {
        let mut pending = 0;
        for _ in 0..1_000_000 {
            match scene.field_step(FieldBudget::Points(budget)) {
                Ok(StepState::Idle) => {
                    assert!(!scene.field_step_pending());
                    return pending;
                }
                Ok(StepState::Pending) => {
                    assert!(scene.field_step_pending());
                    pending += 1;
                }
                Ok(StepState::Done) => assert!(!scene.field_step_pending()),
                Err(SceneError::StaleBvh(id)) => {
                    assert!(!scene.field_step_pending());
                    scene.update(id).unwrap();
                }
                Err(e) => panic!("field_step failed: {e}"),
            }
        }
        panic!("field_step never went idle");
    }

    /// Opens a unit and steps it one candidate at a time, up to `points`
    /// candidates but leaving at least one to compute. Returns the
    /// candidates computed.
    fn step_partway(scene: &mut Scene, points: usize) -> usize {
        let state = scene.field_step(FieldBudget::Points(1)).unwrap();
        assert_eq!(state, StepState::Pending);
        let mut done = 1;
        while done < points && scene.field_step_left() > 1 {
            let state = scene.field_step(FieldBudget::Points(1)).unwrap();
            assert_eq!(state, StepState::Pending);
            done += 1;
        }
        assert!(scene.field_step_pending());
        done
    }

    #[test]
    fn stepped_matches_one_shot() {
        // Four copies of each random scene, all deferring their Eager
        // fields: three are stepped (then `update_fields` brings the Lazy
        // and Manual fields along), the last only calls `update_fields`.
        let budgets = [1, 37, usize::MAX];
        let mut rng = Lcg(0x57e9);
        let mut spread = [0usize; 3];
        for round in 0..10 {
            let state = rng.0;
            let mut twins: Vec<_> = (0..4)
                .map(|_| {
                    let mut r = Lcg(state);
                    let (mut scene, a, b) = twin_scene(&mut r);
                    scene.set_defer_fields(true);
                    scene.set_mode(a, UpdateMode::Eager).unwrap();
                    scene.set_mode(b, UpdateMode::Eager).unwrap();
                    update_all(&mut scene);
                    scene.take_changes();
                    (r, scene, a, b)
                })
                .collect();
            rng.0 = twins[0].0.0;
            for step in 0..30 {
                let ops = 1 + rng.below(4);
                let mut out = Vec::new();
                for (k, (r, scene, a, b)) in twins.iter_mut().enumerate() {
                    for _ in 0..ops {
                        random_op(scene, *a, *b, r);
                    }
                    if let Some(&budget) = budgets.get(k) {
                        spread[k] += step_all(scene, budget);
                    }
                    update_all(scene);
                    out.push(seen(scene, [*a, *b]));
                }
                for k in 0..3 {
                    assert_eq!(
                        out[k], out[3],
                        "round {round} step {step} budget {}",
                        budgets[k]
                    );
                }
            }
        }
        // Small budgets left units open across steps; the largest never.
        assert!(spread[0] > spread[1] && spread[1] > 0, "{spread:?}");
        assert_eq!(spread[2], 0);
    }

    /// [`step_all`], abandoning the unit at up to two random step
    /// boundaries (one in five) and stepping on, which opens a new one.
    /// Returns the units abandoned.
    fn step_all_abandoning(scene: &mut Scene, budget: usize, rng: &mut Lcg) -> usize {
        let mut abandoned = 0;
        for _ in 0..1_000_000 {
            match scene.field_step(FieldBudget::Points(budget)) {
                Ok(StepState::Idle) => return abandoned,
                Ok(StepState::Pending) => {
                    if abandoned < 2 && rng.below(5) == 0 {
                        scene.abandon_field_step();
                        assert!(!scene.field_step_pending());
                        abandoned += 1;
                    }
                }
                Ok(StepState::Done) => {}
                Err(SceneError::StaleBvh(id)) => scene.update(id).unwrap(),
                Err(e) => panic!("field_step failed: {e}"),
            }
        }
        panic!("field_step never went idle");
    }

    #[test]
    fn stepped_with_abandons_matches_one_shot() {
        // As `stepped_matches_one_shot`, with units abandoned at random step
        // boundaries in the random walk and the stepping carried on.
        let budgets = [1, 37];
        let mut rng = Lcg(0xab4d);
        let mut abandoned = [0usize; 2];
        for round in 0..8 {
            let state = rng.0;
            let mut twins: Vec<_> = (0..3)
                .map(|k| {
                    let mut r = Lcg(state);
                    let (mut scene, a, b) = twin_scene(&mut r);
                    scene.set_defer_fields(true);
                    scene.set_mode(a, UpdateMode::Eager).unwrap();
                    scene.set_mode(b, UpdateMode::Eager).unwrap();
                    update_all(&mut scene);
                    scene.take_changes();
                    // Its own generator for when to abandon, so that the
                    // random walks stay the same.
                    (r, Lcg(state ^ (k as u64 + 1)), scene, a, b)
                })
                .collect();
            rng.0 = twins[0].0.0;
            for step in 0..30 {
                let ops = 1 + rng.below(4);
                let mut out = Vec::new();
                for (k, (r, when, scene, a, b)) in twins.iter_mut().enumerate() {
                    for _ in 0..ops {
                        random_op(scene, *a, *b, r);
                    }
                    if let Some(&budget) = budgets.get(k) {
                        abandoned[k] += step_all_abandoning(scene, budget, when);
                    }
                    update_all(scene);
                    out.push(seen(scene, [*a, *b]));
                }
                for k in 0..2 {
                    assert_eq!(
                        out[k], out[2],
                        "round {round} step {step} budget {}",
                        budgets[k]
                    );
                }
            }
        }
        assert!(abandoned.iter().all(|&n| n >= 20), "{abandoned:?}");
    }

    #[test]
    fn step_error_stays_with_its_field() {
        let (mut scene, a, b, [fa, fb]) = deferred_pair(0.15);
        let (mut twin, ..) = deferred_pair(0.15);
        for s in [&mut scene, &mut twin] {
            s.set_placement(b, turned(6.0, [0.03, 0.02, 0.01])).unwrap();
            s.take_changes();
        }
        let before = bits(scene.field_values(fa).unwrap());
        // Computing A's field fails at every step; B's goes on.
        counter::set_failing(Some(fa));
        let mut states = Vec::new();
        let last = loop {
            match scene.field_step(FieldBudget::Points(7)) {
                Ok(StepState::Pending) => states.push(StepState::Pending),
                other => break other,
            }
        };
        counter::set_failing(None);
        assert!(!states.is_empty());
        assert_eq!(last, Err(SceneError::Store(StoreError::OutOfMemory)));
        assert!(!scene.field_step_pending());
        // A's field is left as it was, stale and due; B's field applied.
        assert_eq!(bits(scene.field_values(fa).unwrap()), before);
        assert!(!scene.field_status(fa).unwrap().current);
        assert!(scene.field_status(fb).unwrap().current);
        twin.update_fields().unwrap();
        assert_eq!(
            bits(scene.field_values(fb).unwrap()),
            bits(twin.field_values(fb).unwrap())
        );
        let ch = scene.take_changes();
        assert!(ch.iter().all(|c| c.id != a), "A's field reported");
        // The next unit brings A's field along.
        assert!(step_all(&mut scene, 7) > 0);
        twin.take_changes();
        assert_eq!(seen_values(&scene, [a, b]), seen_values(&twin, [a, b]));
        // With B's field failing in a unit computed in one call, that call
        // applies A's field and returns the error.
        for s in [&mut scene, &mut twin] {
            s.set_placement(b, turned(-4.0, [0.01, 0.0, 0.02])).unwrap();
        }
        counter::set_failing(Some(fb));
        assert_eq!(
            scene.field_step(FieldBudget::Points(usize::MAX)),
            Err(SceneError::Store(StoreError::OutOfMemory))
        );
        counter::set_failing(None);
        assert!(scene.field_status(fa).unwrap().current);
        assert!(!scene.field_status(fb).unwrap().current);
        step_all(&mut scene, usize::MAX);
        twin.update_fields().unwrap();
        assert_eq!(seen(&mut scene, [a, b]).0, seen(&mut twin, [a, b]).0);
    }

    #[test]
    fn stepping_runs_parallel_paths() {
        use fast_mesh_store::par;
        // Two sheets of 6,561 vertices each, every vertex recomputed (an
        // infinite max_distance): enough for runs of 2,048 to compute in
        // parallel chunks.
        let make = || {
            let mut scene = Scene::new(SceneOptions::default());
            let a = scene
                .add_mesh(sheet(80, 1.0 / 80.0), MeshOptions::default())
                .unwrap();
            let opts = MeshOptions {
                placement: at(0.0, 0.0, 0.05),
                ..MeshOptions::default()
            };
            let b = scene.add_mesh(sheet(80, 1.0 / 80.0), opts).unwrap();
            for (on, to) in [(a, b), (b, a)] {
                let id = scene
                    .add_distance_field(field("d", on, to, f64::INFINITY))
                    .unwrap();
                let rule = distance_rule(0.1, 0.1);
                scene
                    .set_coloring(on, Some(Coloring { field: id, rule }))
                    .unwrap();
            }
            scene.set_defer_fields(true);
            scene
                .set_placement(b, turned(3.0, [0.01, 0.0, 0.04]))
                .unwrap();
            scene.take_changes();
            (scene, a, b)
        };
        let (mut scene, a, b) = make();
        let (mut twin, ..) = make();
        counter::take_parallel();
        let entries = par::parallel_entries();
        assert_eq!(
            scene.field_step(FieldBudget::Points(1)),
            Ok(StepState::Pending)
        );
        // The first step chooses A's candidates (every vertex slot, a step
        // of its own); fields are stepped one after the other, never as a
        // parallel batch.
        assert_eq!(counter::take_parallel(), 0);
        let opened = par::parallel_entries();
        assert_eq!(opened, entries);
        assert_eq!(scene.field_step_left(), 6561);
        assert_eq!(scene.field_step_stage(), Some(Stage::Values));
        assert_eq!(
            scene.field_step(FieldBudget::Points(2048)),
            Ok(StepState::Pending)
        );
        assert_eq!(scene.field_step_left(), 6561 - 2048);
        let chunked = par::parallel_entries() - opened;
        if par::enabled() {
            assert!(chunked > 0, "values computed sequentially");
        } else {
            assert_eq!(chunked, 0);
        }
        assert!(step_all(&mut scene, 2048) > 0);
        twin.update_fields().unwrap();
        assert_eq!(seen(&mut scene, [a, b]), seen(&mut twin, [a, b]));
    }

    /// A sheet A and a cube B with a field each way, both coloured, Eager
    /// meshes deferring their fields, everything current and drained.
    fn deferred_pair(max_distance: f64) -> (Scene, MeshId, MeshId, [FieldId; 2]) {
        deferred_pair_signed(max_distance, SignMode::Auto)
    }

    /// [`deferred_pair`], with the fields' sign mode. Winding sign data have no
    /// refit, so that an edit of a target always builds them in steps.
    fn deferred_pair_signed(
        max_distance: f64,
        sign: SignMode,
    ) -> (Scene, MeshId, MeshId, [FieldId; 2]) {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        let signed = |mut d: DistanceField| {
            d.sign = sign;
            d
        };
        let fa = scene
            .add_distance_field(signed(field("d", a, b, max_distance)))
            .unwrap();
        let fb = scene
            .add_distance_field(signed(field("d", b, a, max_distance)))
            .unwrap();
        for (m, f, rule) in [(a, fa, distance_rule(0.2, 0.1)), (b, fb, contact_rule(0.1))] {
            scene
                .set_coloring(m, Some(Coloring { field: f, rule }))
                .unwrap();
        }
        scene.set_defer_fields(true);
        scene.take_changes();
        (scene, a, b, [fa, fb])
    }

    #[test]
    fn abandon_leaves_no_trace() {
        // 121 sheet vertices, then 8 cube vertices: partway into the first
        // field, at its end, and into the second.
        for md in [0.15, f64::INFINITY] {
            for points in [1, 60, 121, 125] {
                let at_ = format!("max_distance {md}, {points} points");
                let (mut scene, a, b, ids) = deferred_pair(md);
                let (mut twin, ..) = deferred_pair(md);
                for s in [&mut scene, &mut twin] {
                    s.set_placement(b, turned(10.0, [0.05, -0.02, 0.03]))
                        .unwrap();
                    s.take_changes();
                }
                let before = seen(&mut scene, [a, b]);
                step_partway(&mut scene, points);
                assert_eq!(scene.field_step_meshes(), vec![a, b], "{at_}");
                // Reading does not abandon.
                assert_eq!(
                    seen_values(&scene, [a, b]),
                    (before.0.clone(), before.1.clone())
                );
                assert!(scene.field_step_pending(), "{at_}");
                scene.abandon_field_step();
                assert!(!scene.field_step_pending(), "{at_}");
                assert!(scene.field_step_meshes().is_empty(), "{at_}");
                // Abandoning when idle does nothing.
                scene.abandon_field_step();
                assert_eq!(seen(&mut scene, [a, b]), before, "{at_}");
                for id in ids {
                    assert!(!scene.field_status(id).unwrap().current, "{at_}");
                }
                scene.update_fields().unwrap();
                twin.update_fields().unwrap();
                assert_eq!(seen(&mut scene, [a, b]), seen(&mut twin, [a, b]), "{at_}");
            }
        }
    }

    /// The values and colours of [`Seen`].
    type Shown = (Vec<Vec<u32>>, Vec<Option<Vec<[u8; 4]>>>);

    /// The values and colours of [`seen`], without draining the reports.
    fn seen_values(scene: &Scene, meshes: [MeshId; 2]) -> Shown {
        let values = scene
            .fields
            .iter()
            .map(|f| bits(scene.field_values(f.id).unwrap()))
            .collect();
        let colors = meshes
            .iter()
            .map(|&m| scene.colors(m).unwrap().map(<[_]>::to_vec))
            .collect();
        (values, colors)
    }

    #[test]
    fn abandon_then_edit_matches_one_shot() {
        let edits: [fn(&mut Scene, MeshId, MeshId); 3] = [
            |s, a, _| {
                s.edit(a, |tx| tx.set_positions(&[45], &[[0.5, 0.4, -0.05]]))
                    .unwrap();
            },
            |s, _, b| {
                s.edit(b, |tx| tx.set_positions(&[7], &[[0.75, 0.75, 0.25]]))
                    .unwrap();
            },
            |s, a, _| {
                s.edit(a, |tx| tx.remove_triangles(&[0, 1, 2])).unwrap();
            },
        ];
        for md in [0.15, f64::INFINITY] {
            for (e, edit) in edits.iter().enumerate() {
                // Abandoned by hand, or by the edit itself.
                for by_hand in [true, false] {
                    let at_ = format!("max_distance {md}, edit {e}, by hand {by_hand}");
                    let (mut scene, a, b, _) = deferred_pair(md);
                    let (mut twin, ..) = deferred_pair(md);
                    let p1 = turned(-8.0, [0.04, 0.03, 0.02]);
                    scene.set_placement(b, p1).unwrap();
                    step_partway(&mut scene, 70);
                    if by_hand {
                        scene.abandon_field_step();
                    }
                    edit(&mut scene, a, b);
                    assert!(!scene.field_step_pending(), "{at_}");
                    step_all(&mut scene, 37);
                    twin.set_placement(b, p1).unwrap();
                    edit(&mut twin, a, b);
                    twin.update_fields().unwrap();
                    assert_eq!(seen(&mut scene, [a, b]), seen(&mut twin, [a, b]), "{at_}");
                    // Nothing of P1 alone: the values are those of a fresh
                    // computation at the final state.
                    for m in [a, b] {
                        scene.update(m).unwrap();
                    }
                    for f in scene.fields.iter().map(|f| f.id).collect::<Vec<_>>() {
                        assert_matches(&mut scene, f);
                    }
                }
            }
        }
    }

    #[test]
    fn thickness_plate_reads_depth() {
        // The interior face vertices of a closed plate read its depth.
        let mut scene = Scene::new(SceneOptions::default());
        let m = scene.add_mesh(plate(4), MeshOptions::default()).unwrap();
        let id = scene
            .add_thickness_field(thick("t", m, f64::INFINITY))
            .unwrap();
        assert!(scene.field_status(id).unwrap().current);
        let values = scene.field_values(id).unwrap();
        assert_eq!(values.len(), scene.mesh(m).unwrap().vertex_slots());
        let w = 5;
        for y in 1..4 {
            for x in 1..4 {
                for &v in &[y * w + x, w * w + y * w + x] {
                    assert!((values[v as usize] - 0.5).abs() < 1e-3, "{values:?}");
                }
            }
        }
        // No targets: nearest and target lists are empty, status is the
        // placeholder, and the field lists on its mesh.
        assert_eq!(scene.field_nearest(id).unwrap(), &[]);
        assert_eq!(scene.field_targets(id).unwrap(), &[]);
        assert!(scene.field_target_status(id).unwrap().is_empty());
        assert_eq!(scene.fields_on(m).unwrap(), vec![id]);
        // Colouring by it works like any field.
        let rule = thick_rule(0.0, 1.0);
        scene
            .set_coloring(
                m,
                Some(Coloring {
                    field: id,
                    rule: rule.clone(),
                }),
            )
            .unwrap();
        assert_colors(&scene, m, &rule, id);
    }

    #[test]
    fn thickness_open_sheet_reads_nan() {
        // An open sheet has no opposite surface: every vertex reads NaN.
        let mut scene = Scene::new(SceneOptions::default());
        let m = scene
            .add_mesh(sheet(4, 0.5), MeshOptions::default())
            .unwrap();
        let id = scene
            .add_thickness_field(thick("t", m, f64::INFINITY))
            .unwrap();
        let values = scene.field_values(id).unwrap();
        assert!(!values.is_empty());
        assert!(values.iter().all(|v| v.is_nan()), "{values:?}");
        // NaN colours show the rule's nan colour.
        let rule = thick_rule(0.0, 1.0);
        scene
            .set_coloring(
                m,
                Some(Coloring {
                    field: id,
                    rule: rule.clone(),
                }),
            )
            .unwrap();
        assert_colors(&scene, m, &rule, id);
        for c in scene.colors(m).unwrap().unwrap() {
            assert_eq!(*c, rule.nan);
        }
    }

    #[test]
    fn thickness_max_distance_reads_far() {
        let mut scene = Scene::new(SceneOptions::default());
        let m = scene.add_mesh(plate(2), MeshOptions::default()).unwrap();
        let id = scene.add_thickness_field(thick("t", m, 0.1)).unwrap();
        assert!(
            scene
                .field_values(id)
                .unwrap()
                .iter()
                .all(|&v| v == f32::INFINITY)
        );
        // Reconfiguring widens the bound (the sign is ignored: thickness
        // has none).
        scene
            .reconfigure_field(id, f64::INFINITY, SignMode::Winding)
            .unwrap();
        let values = scene.field_values(id).unwrap();
        assert!(values.iter().all(|v| v.is_finite()), "{values:?}");
        // And back down again.
        scene
            .reconfigure_field(id, 0.1, SignMode::Pseudonormal)
            .unwrap();
        assert!(
            scene
                .field_values(id)
                .unwrap()
                .iter()
                .all(|&v| v == f32::INFINITY)
        );
        assert!(scene.field_status(id).unwrap().current);
    }

    #[test]
    fn thickness_validation() {
        let mut scene = Scene::new(SceneOptions::default());
        let m = scene.add_mesh(plate(1), MeshOptions::default()).unwrap();
        scene.add_thickness_field(thick("t", m, 1.0)).unwrap();
        assert!(matches!(
            scene.add_thickness_field(thick("t", m, 1.0)),
            Err(SceneError::InvalidInput(_))
        ));
        assert!(matches!(
            scene.add_distance_field(field("t", m, m, 1.0)),
            Err(SceneError::InvalidInput(_))
        ));
        assert!(matches!(
            scene.add_thickness_field(ThicknessField {
                name: "u".into(),
                on: m,
                max_distance: -1.0,
            }),
            Err(SceneError::InvalidInput(_))
        ));
        assert!(matches!(
            scene.add_thickness_field(ThicknessField {
                name: "u".into(),
                on: m,
                max_distance: f64::NAN,
            }),
            Err(SceneError::InvalidInput(_))
        ));
        // Withdrawing a computed field, or one its mesh colours by, refuses.
        let id = scene.field_id(0).unwrap();
        let rule = thick_rule(0.0, 1.0);
        scene
            .set_coloring(m, Some(Coloring { field: id, rule }))
            .unwrap();
        assert!(!scene.withdraw_field(id).unwrap());
        scene.set_coloring(m, None).unwrap();
        assert!(!scene.withdraw_field(id).unwrap());
        scene.remove_field(id).unwrap();
        assert!(matches!(
            scene.field_values(id),
            Err(SceneError::UnknownField)
        ));
        // A deferred field withdraws before it computes.
        let late = scene
            .add_thickness_field_deferred(thick("late", m, 1.0))
            .unwrap();
        assert!(scene.withdraw_field(late).unwrap());
        assert!(matches!(
            scene.field_values(late),
            Err(SceneError::UnknownField)
        ));
    }

    #[test]
    fn thickness_follows_edits_and_placements() {
        let mut scene = Scene::new(SceneOptions::default());
        let m = scene.add_mesh(plate(2), MeshOptions::default()).unwrap();
        let id = scene
            .add_thickness_field(thick("t", m, f64::INFINITY))
            .unwrap();
        // The interior face vertices read the depth, 0.5.
        let interior = [4, 9 + 4];
        for &v in &interior {
            assert!((scene.field_values(id).unwrap()[v as usize] - 0.5).abs() < 1e-3);
        }
        // A uniform scale multiplies the world thicknesses.
        scene
            .set_placement(
                m,
                Placement {
                    scale: [2.0, 2.0, 2.0],
                    ..Placement::IDENTITY
                },
            )
            .unwrap();
        for &v in &interior {
            assert!((scene.field_values(id).unwrap()[v as usize] - 1.0).abs() < 1e-3);
        }
        // A non-uniform scale bakes the mesh: doubling z doubles the depth
        // under the untilted interior normals.
        scene
            .set_placement(
                m,
                Placement {
                    scale: [1.0, 1.0, 2.0],
                    ..Placement::IDENTITY
                },
            )
            .unwrap();
        for &v in &interior {
            assert!((scene.field_values(id).unwrap()[v as usize] - 1.0).abs() < 1e-3);
        }
        scene.set_placement(m, Placement::IDENTITY).unwrap();
        // Raising a top vertex reshapes the plate and its thicknesses.
        let before = scene.field_values(id).unwrap().to_vec();
        scene
            .edit(m, |tx| tx.set_positions(&[1], &[[1.0 / 2.0, 0.0, 1.0]]))
            .unwrap();
        assert!(scene.field_status(id).unwrap().current);
        let after = scene.field_values(id).unwrap().to_vec();
        assert_ne!(before, after);
        assert!(after.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn thickness_steps_in_full() {
        let mut scene = Scene::new(SceneOptions::default());
        let m = scene.add_mesh(plate(2), MeshOptions::default()).unwrap();
        let id = scene
            .add_thickness_field_deferred(thick("t", m, f64::INFINITY))
            .unwrap();
        assert!(!scene.field_status(id).unwrap().current);
        assert_eq!(
            scene.field_step_field(id, FieldBudget::Points(1)),
            Ok(StepState::Done)
        );
        assert!(scene.field_status(id).unwrap().current);
        assert!(
            scene
                .field_values(id)
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
        // Stepping a current field idles.
        assert_eq!(
            scene.field_step_field(id, FieldBudget::Points(1)),
            Ok(StepState::Idle)
        );
        // A Lazy field computes on flush.
        let mut lazy = Scene::new(SceneOptions::default());
        let lm = lazy
            .add_mesh(
                plate(2),
                MeshOptions {
                    mode: UpdateMode::Lazy,
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        let lid = lazy
            .add_thickness_field(thick("t", lm, f64::INFINITY))
            .unwrap();
        assert!(!lazy.field_status(lid).unwrap().current);
        lazy.flush().unwrap();
        assert!(lazy.field_status(lid).unwrap().current);
        assert!(
            lazy.field_values(lid)
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn curvature_sphere_reads_one_over_r() {
        // Every vertex of the unit ball reads about 1, positive (a dome from
        // outside).
        let mut scene = Scene::new(SceneOptions::default());
        let m = scene.add_mesh(ball(), MeshOptions::default()).unwrap();
        let id = scene.add_curvature_field(curv("k", m)).unwrap();
        assert!(scene.field_status(id).unwrap().current);
        let values = scene.field_values(id).unwrap();
        assert_eq!(values.len(), scene.mesh(m).unwrap().vertex_slots());
        for &v in values.iter() {
            assert!(v.is_finite(), "{values:?}");
            assert!(v > 0.0, "{values:?}");
        }
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        assert!((mean - 1.0).abs() < 0.3, "{mean}");
        // No targets: nearest and target lists are empty, status is the
        // placeholder, and the field lists on its mesh.
        assert_eq!(scene.field_nearest(id).unwrap(), &[]);
        assert_eq!(scene.field_targets(id).unwrap(), &[]);
        assert!(scene.field_target_status(id).unwrap().is_empty());
        assert_eq!(scene.fields_on(m).unwrap(), vec![id]);
        // Colouring by it works like any field.
        let rule = thick_rule(-1.0, 2.0);
        scene
            .set_coloring(
                m,
                Some(Coloring {
                    field: id,
                    rule: rule.clone(),
                }),
            )
            .unwrap();
        assert_colors(&scene, m, &rule, id);
    }

    #[test]
    fn curvature_plane_reads_zero_and_saddle_negative() {
        let mut scene = Scene::new(SceneOptions::default());
        // A flat sheet reads 0 everywhere, boundary fan included.
        let flat = scene
            .add_mesh(sheet(8, 0.25), MeshOptions::default())
            .unwrap();
        let fid = scene.add_curvature_field(curv("k", flat)).unwrap();
        for &v in scene.field_values(fid).unwrap().iter() {
            assert!(v.is_finite());
            assert!(v.abs() < 1e-5, "{v}");
        }
        // A saddle reads the sign of its dominant bend: about -1/s at the
        // middle, finite at the one-sided rim.
        let m = scene
            .add_mesh(saddle(32, 4.0), MeshOptions::default())
            .unwrap();
        let id = scene.add_curvature_field(curv("k", m)).unwrap();
        let values = scene.field_values(id).unwrap();
        assert!(values.iter().all(|v| v.is_finite()), "{values:?}");
        let c = (16 * 33 + 16) as usize;
        assert!((values[c] + 0.25).abs() < 0.15, "{:?}", values[c]);
        assert!(values.iter().any(|&v| v < -0.1), "{values:?}");
        // NaN colours show the rule's nan colour on dead slots (none here,
        // but the rule still applies).
        let rule = thick_rule(-1.0, 1.0);
        scene
            .set_coloring(
                m,
                Some(Coloring {
                    field: id,
                    rule: rule.clone(),
                }),
            )
            .unwrap();
        assert_colors(&scene, m, &rule, id);
    }

    #[test]
    fn curvature_follows_scales() {
        let mut scene = Scene::new(SceneOptions::default());
        let m = scene.add_mesh(ball(), MeshOptions::default()).unwrap();
        let id = scene.add_curvature_field(curv("k", m)).unwrap();
        let local = scene.field_values(id).unwrap().to_vec();
        // A uniform scale of 2 halves every curvature (1/world-unit).
        scene
            .set_placement(
                m,
                Placement {
                    scale: [2.0, 2.0, 2.0],
                    ..Placement::IDENTITY
                },
            )
            .unwrap();
        let scaled = scene.field_values(id).unwrap().to_vec();
        assert_eq!(scaled.len(), local.len());
        for (&x, &y) in scaled.iter().zip(&local) {
            assert!((x - y / 2.0).abs() < 1e-4, "{x} vs {y}");
        }
        assert!(scaled.iter().all(|v| v.is_finite()));
        // A non-uniform scale bakes a world-frame copy: still finite and
        // current (an ellipsoid is flatter at its equator than a sphere).
        scene
            .set_placement(
                m,
                Placement {
                    scale: [1.0, 1.0, 2.0],
                    ..Placement::IDENTITY
                },
            )
            .unwrap();
        assert!(scene.field_status(id).unwrap().current);
        let baked = scene.field_values(id).unwrap().to_vec();
        assert!(baked.iter().all(|v| v.is_finite()), "{baked:?}");
        assert_ne!(baked, scaled);
        // Back to rigid: the local values return.
        scene.set_placement(m, Placement::IDENTITY).unwrap();
        assert_eq!(scene.field_values(id).unwrap(), &local);
    }

    #[test]
    fn curvature_validation() {
        let mut scene = Scene::new(SceneOptions::default());
        let m = scene.add_mesh(ball(), MeshOptions::default()).unwrap();
        scene.add_thickness_field(thick("t", m, 1.0)).unwrap();
        scene.add_curvature_field(curv("k", m)).unwrap();
        // Names clash across all three registries.
        assert!(matches!(
            scene.add_curvature_field(curv("t", m)),
            Err(SceneError::InvalidInput(_))
        ));
        assert!(matches!(
            scene.add_curvature_field(curv("k", m)),
            Err(SceneError::InvalidInput(_))
        ));
        assert!(matches!(
            scene.add_thickness_field(thick("k", m, 1.0)),
            Err(SceneError::InvalidInput(_))
        ));
        // Reconfiguring marks the field for recompute (both settings are
        // ignored: a curvature has neither).
        let id = scene.field_id(1).unwrap();
        scene.reconfigure_field(id, 0.5, SignMode::Winding).unwrap();
        assert!(scene.field_status(id).unwrap().current);
        assert!(
            scene
                .field_values(id)
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
        // Removing the mesh removes the field.
        scene.remove_mesh(m).unwrap();
        assert!(matches!(
            scene.field_values(id),
            Err(SceneError::UnknownField)
        ));
        // A deferred field withdraws before it computes.
        let m2 = scene.add_mesh(ball(), MeshOptions::default()).unwrap();
        let late = scene
            .add_curvature_field_deferred(curv("late", m2))
            .unwrap();
        assert!(scene.withdraw_field(late).unwrap());
        assert!(matches!(
            scene.field_values(late),
            Err(SceneError::UnknownField)
        ));
    }

    #[test]
    fn curvature_steps_in_full() {
        let mut scene = Scene::new(SceneOptions::default());
        let m = scene.add_mesh(ball(), MeshOptions::default()).unwrap();
        let id = scene.add_curvature_field_deferred(curv("k", m)).unwrap();
        assert!(!scene.field_status(id).unwrap().current);
        assert_eq!(
            scene.field_step_field(id, FieldBudget::Points(1)),
            Ok(StepState::Done)
        );
        assert!(scene.field_status(id).unwrap().current);
        assert!(
            scene
                .field_values(id)
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
        // Stepping a current field idles.
        assert_eq!(
            scene.field_step_field(id, FieldBudget::Points(1)),
            Ok(StepState::Idle)
        );
        // A Lazy field computes on flush.
        let mut lazy = Scene::new(SceneOptions::default());
        let lm = lazy
            .add_mesh(
                ball(),
                MeshOptions {
                    mode: UpdateMode::Lazy,
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        let lid = lazy.add_curvature_field(curv("k", lm)).unwrap();
        assert!(!lazy.field_status(lid).unwrap().current);
        lazy.flush().unwrap();
        assert!(lazy.field_status(lid).unwrap().current);
        assert!(
            lazy.field_values(lid)
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn mutation_abandons_open_unit() {
        let (mut scene, a, b, _) = deferred_pair(0.15);
        let (mut twin, ..) = deferred_pair(0.15);
        let far = MeshOptions {
            placement: at(5.0, 0.0, 0.0),
            ..MeshOptions::default()
        };
        let c = scene.add_mesh(sheet(2, 0.5), far).unwrap();
        twin.add_mesh(sheet(2, 0.5), far).unwrap();
        // An unrelated placement abandons the unit at P1; the unit stepped
        // afterwards computes P1 with the new placement.
        let p1 = turned(5.0, [0.02, 0.01, 0.0]);
        scene.set_placement(b, p1).unwrap();
        step_partway(&mut scene, 30);
        scene.set_placement(c, at(6.0, 0.0, 0.0)).unwrap();
        assert!(!scene.field_step_pending());
        step_all(&mut scene, 37);
        twin.set_placement(b, p1).unwrap();
        twin.set_placement(c, at(6.0, 0.0, 0.0)).unwrap();
        twin.update_fields().unwrap();
        assert_eq!(seen(&mut scene, [a, b]), seen(&mut twin, [a, b]));
        // A newer pose of B abandons the unit at P2; the next steps compute
        // P3.
        let (p2, p3) = (at(0.1, 0.0, 0.05), at(-0.05, 0.02, -0.03));
        scene.set_placement(b, p2).unwrap();
        step_partway(&mut scene, 100);
        scene.set_placement(b, p3).unwrap();
        assert!(!scene.field_step_pending());
        step_all(&mut scene, 37);
        twin.set_placement(b, p2).unwrap();
        twin.set_placement(b, p3).unwrap();
        twin.update_fields().unwrap();
        assert_eq!(seen(&mut scene, [a, b]), seen(&mut twin, [a, b]));
    }

    /// What [`mutating_calls_abandon_open_unit`] needs of its scene.
    struct Ctx {
        a: MeshId,
        b: MeshId,
        c: MeshId,
        field: FieldId,
        section: crate::SectionId,
        subscriber: crate::SubscriberId,
    }

    /// A scene whose field unit is open: a sheet A and a cube B with a
    /// field A to B, another sheet C far away with a channel `x` and a
    /// section, and a subscriber.
    fn open_unit() -> (Scene, Ctx) {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        let field = scene
            .add_distance_field(field("d", a, b, f64::INFINITY))
            .unwrap();
        let opts = MeshOptions {
            placement: at(5.0, 0.0, 0.0),
            ..MeshOptions::default()
        };
        let c = scene.add_mesh(sheet(2, 0.5), opts).unwrap();
        scene.add_channel(c, ChannelDesc::derived("x", 1)).unwrap();
        let plane = crate::Plane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        let section = scene
            .add_section(crate::SectionDesc {
                plane,
                meshes: vec![c],
                fill: false,
            })
            .unwrap();
        let subscriber = scene.subscribe();
        scene.set_defer_fields(true);
        scene.set_placement(b, at(0.01, 0.0, 0.0)).unwrap();
        step_partway(&mut scene, 1);
        let ctx = Ctx {
            a,
            b,
            c,
            field,
            section,
            subscriber,
        };
        (scene, ctx)
    }

    #[test]
    fn mutating_calls_abandon_open_unit() {
        use crate::{ContactFilter, Plane, SceneClosestOptions, SceneRayOptions, SectionDesc};
        type Call = fn(&mut Scene, &Ctx);
        // Every public `&mut self` method of `Scene` but the three step
        // calls.
        let calls: [(&str, Call); 43] = [
            ("add_mesh", |s, _| {
                s.add_mesh(sheet(1, 1.0), MeshOptions::default()).unwrap();
            }),
            ("add_mesh_built", |s, _| {
                let build = crate::MeshBuild::new(sheet(1, 1.0), MeshOptions::default()).unwrap();
                s.add_mesh_built(build).unwrap();
            }),
            ("remove_mesh", |s, x| {
                s.remove_mesh(x.c).unwrap();
            }),
            ("edit", |s, x| {
                s.edit(x.c, |tx| tx.set_positions(&[0], &[[0.0, 0.0, 0.1]]))
                    .unwrap();
            }),
            ("begin_step", |s, _| s.begin_step()),
            ("end_step", |s, _| {
                let _ = s.end_step();
            }),
            ("cancel_step", |s, _| {
                let _ = s.cancel_step();
            }),
            ("undo", |s, _| {
                s.undo().unwrap();
            }),
            ("redo", |s, _| {
                s.redo().unwrap();
            }),
            ("clear_history", |s, _| s.clear_history().unwrap()),
            ("set_mode", |s, x| {
                s.set_mode(x.c, UpdateMode::Lazy).unwrap()
            }),
            ("update", |s, x| s.update(x.c).unwrap()),
            ("flush", |s, _| s.flush().unwrap()),
            ("take_changes", |s, _| {
                s.take_changes();
            }),
            ("subscribe", |s, _| {
                s.subscribe();
            }),
            ("unsubscribe", |s, x| s.unsubscribe(x.subscriber).unwrap()),
            ("take_changes_for", |s, x| {
                s.take_changes_for(x.subscriber).unwrap();
            }),
            ("add_channel", |s, x| {
                s.add_channel(x.c, ChannelDesc::derived("y", 1)).unwrap();
            }),
            ("remove_channel", |s, x| {
                s.remove_channel(x.c, "x").unwrap();
            }),
            ("derived_channel_mut", |s, x| {
                s.derived_channel_mut(x.c, "x").unwrap();
            }),
            ("compact", |s, x| {
                s.compact(x.c).unwrap();
            }),
            ("set_placement", |s, x| {
                s.set_placement(x.c, at(6.0, 0.0, 0.0)).unwrap();
            }),
            ("add_distance_field", |s, x| {
                s.add_distance_field(field("e", x.c, x.b, 0.1)).unwrap();
            }),
            ("add_distance_field_deferred", |s, x| {
                s.add_distance_field_deferred(field("e", x.c, x.b, 0.1))
                    .unwrap();
            }),
            ("add_thickness_field", |s, x| {
                s.add_thickness_field(thick("e", x.c, f64::INFINITY))
                    .unwrap();
            }),
            ("add_thickness_field_deferred", |s, x| {
                s.add_thickness_field_deferred(thick("e", x.c, f64::INFINITY))
                    .unwrap();
            }),
            ("withdraw_field", |s, x| {
                assert!(!s.withdraw_field(x.field).unwrap());
            }),
            ("remove_field", |s, x| s.remove_field(x.field).unwrap()),
            ("update_fields", |s, _| s.update_fields().unwrap()),
            ("set_coloring", |s, x| s.set_coloring(x.c, None).unwrap()),
            ("set_normals", |s, x| s.set_normals(x.c, true).unwrap()),
            ("raycast", |s, _| {
                let opts = SceneRayOptions::default();
                s.raycast([0.5, 0.5, 1.0], [0.0, 0.0, -1.0], &opts).unwrap();
            }),
            ("closest_point", |s, _| {
                let opts = SceneClosestOptions::default();
                s.closest_point([0.5, 0.5, 1.0], &opts).unwrap();
            }),
            ("closest_point_on", |s, x| {
                let opts = SceneClosestOptions::default();
                s.closest_point_on(x.c, [0.5, 0.5, 1.0], &opts).unwrap();
            }),
            ("intersects", |s, x| {
                s.intersects(x.c, x.b).unwrap();
            }),
            ("mesh_distance", |s, x| {
                s.mesh_distance(x.c, x.b, 0.1).unwrap();
            }),
            ("triangles_near", |s, x| {
                let (mut ta, mut tb) = (Vec::new(), Vec::new());
                s.triangles_near(x.c, x.b, 0.1, &mut ta, &mut tb).unwrap();
            }),
            ("find_contacts", |s, _| {
                s.find_contacts(0.1, &ContactFilter::default(), &mut Vec::new())
                    .unwrap();
            }),
            ("add_section", |s, x| {
                let plane = Plane::new([0.0, 0.0, 0.1], [0.0, 0.0, 1.0]).unwrap();
                let desc = SectionDesc {
                    plane,
                    meshes: vec![x.c],
                    fill: true,
                };
                s.add_section(desc).unwrap();
            }),
            ("remove_section", |s, x| {
                s.remove_section(x.section).unwrap()
            }),
            ("set_section_plane", |s, x| {
                let plane = Plane::new([0.0, 0.0, 0.2], [0.0, 0.0, 1.0]).unwrap();
                s.set_section_plane(x.section, plane).unwrap();
            }),
            ("set_section_meshes", |s, x| {
                s.set_section_meshes(x.section, vec![x.a, x.c]).unwrap();
            }),
            ("set_section_fill", |s, x| {
                s.set_section_fill(x.section, true).unwrap();
            }),
        ];
        for (name, call) in calls {
            let (mut scene, ctx) = open_unit();
            assert!(scene.field_step_pending(), "{name}");
            call(&mut scene, &ctx);
            assert!(!scene.field_step_pending(), "{name} left the unit open");
        }
        // `update_sections` too (39 in all).
        let (mut scene, _) = open_unit();
        scene.update_sections().unwrap();
        assert!(!scene.field_step_pending(), "update_sections");
        // Reads and deferral leave the unit open.
        let (mut scene, x) = open_unit();
        scene.mesh(x.a).unwrap();
        scene.placement(x.b).unwrap();
        scene.is_current(x.a).unwrap();
        scene.field_values(x.field).unwrap();
        scene.field_status(x.field).unwrap();
        scene.colors(x.a).unwrap();
        assert_eq!(scene.field_step_meshes(), vec![x.a, x.b]);
        scene.set_defer_fields(false);
        scene.set_defer_fields(true);
        assert!(scene.field_step_pending(), "a read or deferral abandoned");
        assert_eq!(step_all(&mut scene, usize::MAX), 0);
        assert!(scene.field_status(x.field).unwrap().current);
    }

    #[test]
    fn defer_leaves_fields_due() {
        let (mut scene, a, b, ids) = deferred_pair(0.15);
        assert_eq!(
            scene.field_step(FieldBudget::Points(1)),
            Ok(StepState::Idle)
        );
        counter::take();
        scene.set_placement(b, at(0.02, 0.0, 0.01)).unwrap();
        scene
            .edit(a, |tx| tx.set_positions(&[45], &[[0.5, 0.4, 0.05]]))
            .unwrap();
        assert_eq!(counter::take(), 0);
        for id in ids {
            assert!(!scene.field_status(id).unwrap().current);
        }
        assert!(!scene.field_step_pending());
        // Stepping brings both fields along in one unit, applied together.
        let before = seen_values(&scene, [a, b]);
        assert_eq!(
            scene.field_step(FieldBudget::Points(5)),
            Ok(StepState::Pending)
        );
        assert_eq!(seen_values(&scene, [a, b]), before);
        assert_eq!(counter::take(), 0);
        assert!(step_all(&mut scene, usize::MAX) == 0);
        assert!(counter::take() > 0);
        for id in ids {
            assert_matches(&mut scene, id);
        }
        // Without deferral the fields follow at once.
        scene.set_defer_fields(false);
        scene.set_placement(b, at(0.03, 0.0, 0.01)).unwrap();
        assert!(counter::take() > 0);
        for id in ids {
            assert!(scene.field_status(id).unwrap().current);
        }
        // Lazy fields are not stepped.
        scene.set_mode(a, UpdateMode::Lazy).unwrap();
        scene.set_mode(b, UpdateMode::Lazy).unwrap();
        scene.set_placement(b, at(0.04, 0.0, 0.01)).unwrap();
        assert_eq!(
            scene.field_step(FieldBudget::Points(1)),
            Ok(StepState::Idle)
        );
        assert!(!scene.field_status(ids[0]).unwrap().current);
    }

    /// A contact group (spec §8): a sheet A, cubes B and C cutting it and
    /// each other, and a cube D cutting A only, with the pairs A-B, B-C,
    /// A-C and A-D. Each mesh has one field to its partners, registered in
    /// the order A, B, C, D.
    fn group_scene(mode: UpdateMode) -> (Scene, [MeshId; 4], [FieldId; 4]) {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(sheet(10, 0.1), with_mode(mode)).unwrap();
        let b = scene
            .add_mesh(cube([0.1, 0.1, -0.2], 0.3), with_mode(mode))
            .unwrap();
        let c = scene
            .add_mesh(cube([0.3, 0.2, -0.1], 0.3), with_mode(mode))
            .unwrap();
        let d = scene
            .add_mesh(cube([0.75, 0.7, -0.1], 0.2), with_mode(mode))
            .unwrap();
        let mut add = |on: MeshId, to: Vec<MeshId>| {
            scene
                .add_distance_field(DistanceField {
                    to,
                    ..field("g", on, on, 0.12)
                })
                .unwrap()
        };
        let fields = [
            add(a, vec![b, c, d]),
            add(b, vec![a, c]),
            add(c, vec![a, b]),
            add(d, vec![a]),
        ];
        (scene, [a, b, c, d], fields)
    }

    /// The fields applied by `f`, in order.
    fn applied(scene: &mut Scene, f: impl FnOnce(&mut Scene)) -> Vec<FieldId> {
        counter::take_applied();
        f(scene);
        counter::take_applied()
    }

    #[test]
    fn group_recomputes_only_related_fields() {
        // Near searches never abandoned, so that a move recomputes only near
        // vertices.
        counter::set_near_limit(Some(usize::MAX));
        let (mut scene, [a, b, c, d], [fa, fb, fc, fd]) = group_scene(UpdateMode::Eager);
        // Every field touches something.
        for f in [fa, fb, fc, fd] {
            assert!(finite_count(&scene, f) > 0);
        }
        let a_slots = scene.mesh(a).unwrap().vertex_slots();
        counter::take();
        // D moves: only A's field (measured to D) and D's own.
        let got = applied(&mut scene, |s| {
            s.set_placement(d, at(0.01, 0.0, 0.0)).unwrap()
        });
        assert_eq!(got, [fa, fd]);
        // A's vertices near D, and D's: far fewer than A has.
        let recomputed = counter::take();
        assert!(recomputed > 0 && recomputed < a_slots, "{recomputed}");
        // A moves: every field.
        let got = applied(&mut scene, |s| {
            s.set_placement(a, at(0.0, 0.01, 0.0)).unwrap()
        });
        assert_eq!(got, [fa, fb, fc, fd]);
        // B moves, or is edited: A's, B's and C's, never D's.
        let got = applied(&mut scene, |s| {
            s.set_placement(b, at(0.0, 0.0, 0.01)).unwrap()
        });
        assert_eq!(got, [fa, fb, fc]);
        let got = applied(&mut scene, |s| {
            s.edit(b, |tx| tx.set_positions(&[0], &[[0.12, 0.1, -0.2]]))
                .unwrap();
        });
        assert_eq!(got, [fa, fb, fc]);
        // C moves: A's, B's and C's.
        let got = applied(&mut scene, |s| {
            s.set_placement(c, at(0.01, 0.0, 0.0)).unwrap()
        });
        assert_eq!(got, [fa, fb, fc]);
        // An edit of D: A's and D's.
        let got = applied(&mut scene, |s| {
            s.edit(d, |tx| tx.set_positions(&[0], &[[0.76, 0.7, -0.1]]))
                .unwrap();
        });
        assert_eq!(got, [fa, fd]);
        // Deferred, the same fields are due, and stepping applies them.
        scene.set_defer_fields(true);
        scene.set_placement(d, at(0.02, 0.0, 0.0)).unwrap();
        let current: Vec<bool> = [fa, fb, fc, fd]
            .iter()
            .map(|&f| scene.field_status(f).unwrap().current)
            .collect();
        assert_eq!(current, [false, true, true, false]);
        assert_eq!(
            scene.field_step(FieldBudget::Points(1)),
            Ok(StepState::Pending)
        );
        assert_eq!(scene.field_step_meshes(), [a, b, c, d,]);
        let got = applied(&mut scene, |s| {
            step_all(s, usize::MAX);
        });
        assert_eq!(got, [fa, fd]);
        counter::set_near_limit(None);
    }

    /// The values a field's mesh would get from its targets one at a time:
    /// the minimum of single-target fields, ties to the lowest index.
    fn min_of_singles(scene: &mut Scene, id: FieldId) -> (Vec<u32>, Vec<u32>) {
        let desc = scene
            .fields
            .iter()
            .find(|f| f.id == id)
            .unwrap()
            .desc
            .clone();
        let singles: Vec<FieldId> = desc
            .to
            .iter()
            .enumerate()
            .map(|(k, &m)| {
                scene
                    .add_distance_field(DistanceField {
                        name: format!("single{k}"),
                        to: vec![m],
                        ..desc.clone()
                    })
                    .unwrap()
            })
            .collect();
        update_all(scene);
        let slots = scene.mesh(desc.on).unwrap().vertex_slots();
        let (mut values, mut nearest) = (Vec::new(), Vec::new());
        for v in 0..slots {
            let parts: Vec<f32> = singles
                .iter()
                .map(|&f| scene.field_values(f).unwrap()[v])
                .collect();
            let (mut best, mut at) = (parts[0], u32::MAX);
            if !best.is_nan() {
                best = f32::INFINITY;
                for (k, &x) in parts.iter().enumerate() {
                    if x < best {
                        (best, at) = (x, k as u32);
                    }
                }
            }
            values.push(best.to_bits());
            nearest.push(at);
        }
        for f in singles {
            scene.remove_field(f).unwrap();
        }
        (values, nearest)
    }

    /// The field's value bits and nearest targets (`u32::MAX` past the end
    /// of the nearest list).
    fn shown(scene: &Scene, id: FieldId) -> (Vec<u32>, Vec<u32>) {
        let values = bits(scene.field_values(id).unwrap());
        let nearest = scene.field_nearest(id).unwrap();
        let nearest = (0..values.len())
            .map(|v| nearest.get(v).copied().unwrap_or(u32::MAX))
            .collect();
        (values, nearest)
    }

    #[test]
    fn group_fields_match_single_targets() {
        let (mut scene, meshes, fields) = group_scene(UpdateMode::Eager);
        let mut rng = Lcg(0x6709);
        for step in 0..30 {
            let m = meshes[rng.below(4)];
            let t = [
                rng.range(-0.05, 0.05),
                rng.range(-0.05, 0.05),
                rng.range(-0.05, 0.05),
            ];
            scene
                .set_placement(m, turned(rng.range(-10.0, 10.0), t))
                .unwrap();
            for &f in &fields {
                let got = shown(&scene, f);
                assert_eq!(got, min_of_singles(&mut scene, f), "step {step}");
            }
        }
        // Every target was nearest somewhere in A's field.
        let (_, nearest) = shown(&scene, fields[0]);
        for k in 0..3 {
            assert!(nearest.contains(&k), "target {k}");
        }
    }

    #[test]
    fn group_near_limit_matches_near_path() {
        // The same walk with every near search abandoned (limit 0) and with no
        // limit: the same values, nearest targets and reports.
        let mut runs = Vec::new();
        for limit in [Some(0), Some(usize::MAX)] {
            counter::set_near_limit(limit);
            counter::take_aborted();
            let (mut scene, meshes, fields) = group_scene(UpdateMode::Eager);
            let mut rng = Lcg(0x11);
            let mut out = Vec::new();
            for _ in 0..20 {
                let m = meshes[rng.below(4)];
                let t = [rng.range(-0.05, 0.05), rng.range(-0.05, 0.05), 0.0];
                scene
                    .set_placement(m, turned(rng.range(-5.0, 5.0), t))
                    .unwrap();
                out.push((fields.map(|f| shown(&scene, f)), scene.take_changes()));
            }
            runs.push((out, counter::take_aborted()));
        }
        counter::set_near_limit(None);
        assert!(runs[0].0 == runs[1].0);
        assert!(runs[0].1 > 0);
        assert_eq!(runs[1].1, 0);
    }

    /// Everything a renderer sees of the group: every field's values and
    /// nearest targets, and the drained reports.
    type SeenGroup = (Vec<(Vec<u32>, Vec<u32>)>, Vec<crate::MeshChanges>);

    fn seen_group(scene: &mut Scene, fields: &[FieldId; 4]) -> SeenGroup {
        let shown = fields.iter().map(|&f| shown(scene, f)).collect();
        (shown, scene.take_changes())
    }

    #[test]
    fn group_stepped_matches_one_shot() {
        // Four copies of the group: three stepped with different budgets
        // (one abandoning units at random), the last updated in one shot.
        let budgets = [1, 23, usize::MAX];
        let mut twins: Vec<_> = (0..4)
            .map(|_| {
                let (mut scene, meshes, fields) = group_scene(UpdateMode::Eager);
                scene.set_defer_fields(true);
                scene.take_changes();
                (scene, meshes, fields)
            })
            .collect();
        let mut rng = Lcg(0x5e9);
        let mut abandons = Lcg(3);
        let mut spread = [0usize; 3];
        for step in 0..25 {
            let m = rng.below(4);
            let p = turned(
                rng.range(-8.0, 8.0),
                [
                    rng.range(-0.05, 0.05),
                    rng.range(-0.05, 0.05),
                    rng.range(-0.03, 0.03),
                ],
            );
            let edit = rng.below(3) == 0;
            let z = rng.range(-0.1, 0.1) as f32;
            let mut out = Vec::new();
            for (k, (scene, meshes, fields)) in twins.iter_mut().enumerate() {
                scene.set_placement(meshes[m], p).unwrap();
                if edit {
                    scene
                        .edit(meshes[0], |tx| tx.set_positions(&[12], &[[0.2, 0.1, z]]))
                        .unwrap();
                }
                match k {
                    0 => spread[0] += step_all(scene, budgets[0]),
                    1 => spread[1] += step_all_abandoning(scene, budgets[1], &mut abandons),
                    2 => spread[2] += step_all(scene, budgets[2]),
                    _ => update_all(scene),
                }
                out.push(seen_group(scene, fields));
            }
            for k in 0..3 {
                assert!(out[k] == out[3], "step {step}, twin {k}");
            }
        }
        assert!(spread[0] > 0 && spread[1] > 0, "{spread:?}");
        assert_eq!(spread[2], 0);
    }

    #[test]
    fn removing_a_target_keeps_the_field() {
        let (mut scene, [a, b, c, d], [fa, fb, fc, fd]) = group_scene(UpdateMode::Eager);
        scene
            .set_coloring(
                a,
                Some(Coloring {
                    field: fa,
                    rule: contact_rule(0.05),
                }),
            )
            .unwrap();
        scene.take_changes();
        scene.remove_mesh(c).unwrap();
        // C's own field goes; A's and B's lose C and are recomputed at once.
        assert_eq!(scene.field_values(fc), Err(SceneError::UnknownField));
        assert_eq!(scene.field_targets(fa), Ok(&[b, d][..]));
        assert_eq!(scene.field_targets(fb), Ok(&[a][..]));
        assert_eq!(scene.field_target_status(fa).unwrap().len(), 2);
        for f in [fa, fb, fd] {
            assert!(scene.field_status(f).unwrap().current);
        }
        let got = shown(&scene, fa);
        assert_eq!(got, min_of_singles(&mut scene, fa));
        assert!(got.1.iter().all(|&n| n < 2 || n == u32::MAX));
        assert!(got.1.contains(&1));
        assert_colors(&scene, a, &contact_rule(0.05), fa);
        let ch = scene.take_changes();
        assert!(
            ch.iter()
                .find(|c| c.id == a)
                .unwrap()
                .fields_changed
                .contains(&fa)
        );
        // Deferred, the field waits for a step, then recomputes in full.
        scene.set_defer_fields(true);
        scene.remove_mesh(d).unwrap();
        assert_eq!(scene.field_targets(fa), Ok(&[b][..]));
        assert!(!scene.field_status(fa).unwrap().current);
        assert!(
            scene
                .field_nearest(fa)
                .unwrap()
                .iter()
                .all(|&n| n == 0 || n == u32::MAX)
        );
        counter::take();
        step_all(&mut scene, usize::MAX);
        assert_eq!(counter::take(), scene.mesh(a).unwrap().vertex_count());
        assert_eq!(shown(&scene, fa), min_of_singles(&mut scene, fa));
        // The last target goes: so does the field, and its colouring.
        scene.remove_mesh(b).unwrap();
        assert_eq!(scene.field_values(fa), Err(SceneError::UnknownField));
        assert_eq!(scene.field_values(fb), Err(SceneError::UnknownField));
        assert_eq!(scene.colors(a), Ok(None));
        assert!(scene.fields.is_empty());
    }

    #[test]
    fn compact_remaps_nearest() {
        let (mut scene, [a, ..], [fa, ..]) = group_scene(UpdateMode::Eager);
        // A vertex near the cubes loses its triangles and goes.
        scene
            .edit(a, |tx| {
                let tris: Vec<u32> = (0..tx.mesh().triangle_slots() as u32)
                    .filter(|&t| tx.mesh().triangle(t).is_some_and(|tri| tri.contains(&0)))
                    .collect();
                tx.remove_triangles(&tris)?;
                tx.remove_vertices(&[0])
            })
            .unwrap();
        let before = scene.field_nearest(fa).unwrap().to_vec();
        assert!(before.iter().any(|&n| n != u32::MAX));
        scene.set_defer_fields(true);
        let remap = scene.compact(a).unwrap();
        let after = scene.field_nearest(fa).unwrap();
        assert_eq!(after.len(), scene.mesh(a).unwrap().vertex_slots());
        for (v, &n) in before.iter().enumerate() {
            if let Some(w) = remap.vertex(v as VertexId) {
                assert_eq!(after[w as usize], n, "vertex {v}");
            }
        }
        step_all(&mut scene, usize::MAX);
        assert_eq!(shown(&scene, fa), min_of_singles(&mut scene, fa));
    }

    #[test]
    fn invalid_target_lists_rejected() {
        let (mut scene, [a, b, c, _], _) = group_scene(UpdateMode::Eager);
        let bad = |e| matches!(e, Err(SceneError::InvalidInput(_)));
        let desc = |to: Vec<MeshId>| DistanceField {
            to,
            ..field("x", a, b, 0.1)
        };
        assert!(bad(scene.add_distance_field(desc(vec![]))));
        assert!(bad(scene.add_distance_field(desc(vec![b, c, b]))));
        assert!(bad(scene.add_distance_field(desc(vec![b, a]))));
        assert_eq!(scene.fields.len(), 4);
        let id = scene.add_distance_field(desc(vec![c, b])).unwrap();
        assert_eq!(scene.field_targets(id), Ok(&[c, b][..]));
        // The single-target constructor is sugar for a list of one.
        let one = DistanceField::new("y", a, b, 0.1, SignMode::Auto);
        assert_eq!(one.to, [b]);
        scene.add_distance_field(one).unwrap();
    }

    #[test]
    fn moving_one_target_spares_vertices_another_won() {
        // B is a sheet with a field to a small cube A through one corner and
        // a large cube C just under the rest. Moving A recomputes B's
        // vertices near A, or won by A, never the many that C won.
        counter::set_near_limit(Some(usize::MAX));
        let mut scene = Scene::new(SceneOptions::default());
        let b = scene
            .add_mesh(sheet(10, 0.1), MeshOptions::default())
            .unwrap();
        let a = scene
            .add_mesh(cube([0.05, 0.05, -0.1], 0.2), MeshOptions::default())
            .unwrap();
        let c = scene
            .add_mesh(cube([0.3, 0.3, -0.74], 0.7), MeshOptions::default())
            .unwrap();
        let fb = scene
            .add_distance_field(DistanceField {
                to: vec![a, c],
                ..field("d", b, b, 0.05)
            })
            .unwrap();
        let (_, nearest) = shown(&scene, fb);
        let by = |k: u32| nearest.iter().filter(|&&n| n == k).count();
        let (won_by_a, won_by_c) = (by(0), by(1));
        assert!(won_by_a > 0 && won_by_c > 0);
        counter::take_applied_counts();
        scene.set_placement(a, at(0.01, 0.0, 0.0)).unwrap();
        let counts = counter::take_applied_counts();
        assert_eq!(counts.len(), 1);
        let (f, recomputed) = counts[0];
        assert_eq!(f, fb);
        let slots = scene.mesh(b).unwrap().vertex_slots();
        assert!(recomputed > 0, "nothing recomputed");
        assert!(
            recomputed + won_by_c <= slots,
            "{recomputed} + {won_by_c} > {slots}"
        );
        // The values C won are untouched and still right.
        assert_eq!(shown(&scene, fb), min_of_singles(&mut scene, fb));
        counter::set_near_limit(None);
    }

    #[test]
    fn status_aggregates_targets() {
        let (scene, [a, ..], [_, fb, fc, _]) = group_scene(UpdateMode::Eager);
        // B's field is to the open sheet A and the closed cube C.
        let each = scene.field_target_status(fb).unwrap();
        assert_eq!(each.len(), 2);
        assert!(!each[0].closed && each[1].closed);
        let all = scene.field_status(fb).unwrap().sign;
        assert!(!all.closed);
        assert_eq!(
            all.consistent_orientation,
            each.iter().all(|s| s.consistent_orientation)
        );
        assert_eq!(
            all.non_manifold_vertices,
            each.iter().map(|s| s.non_manifold_vertices).sum::<usize>()
        );
        assert_eq!(
            all.degenerate_triangles,
            each.iter().map(|s| s.degenerate_triangles).sum::<usize>()
        );
        assert_eq!(all.mode, each[0].mode);
        // C's field to [A, B]: the same, A first.
        assert!(!scene.field_status(fc).unwrap().sign.closed);
        let _ = a;
    }

    #[test]
    fn group_walk_with_limits_steps_and_removal() {
        // The same random walk (placements of every mesh, edits of A, and
        // C removed half-way) under near limits 0 and none and step budgets
        // 1, 17 and all, against one-shot updates: the same values, nearest
        // targets and reports; and the minimum of single fields at the end.
        for seed in 0..4u64 {
            let mut runs = Vec::new();
            for limit in [Some(0), None] {
                for budget in [Some(1), Some(17), None] {
                    counter::set_near_limit(limit);
                    let (mut scene, meshes, fields) = group_scene(UpdateMode::Eager);
                    scene.set_defer_fields(budget.is_some());
                    scene.take_changes();
                    let mut rng = Lcg(seed);
                    let mut out = Vec::new();
                    for step in 0..16 {
                        if step == 8 {
                            scene.remove_mesh(meshes[2]).unwrap();
                        }
                        let k = [0, 1, 3, 0][rng.below(4)];
                        let t = [
                            rng.range(-0.04, 0.04),
                            rng.range(-0.04, 0.04),
                            rng.range(-0.03, 0.03),
                        ];
                        scene
                            .set_placement(meshes[k], turned(rng.range(-6.0, 6.0), t))
                            .unwrap();
                        if rng.below(3) == 0 {
                            let z = rng.range(-0.1, 0.1) as f32;
                            scene
                                .edit(meshes[0], |tx| tx.set_positions(&[23], &[[0.3, 0.2, z]]))
                                .unwrap();
                        }
                        if let Some(budget) = budget {
                            step_all(&mut scene, budget);
                        }
                        let live: Vec<FieldId> = fields
                            .iter()
                            .copied()
                            .filter(|&f| scene.field_targets(f).is_ok())
                            .collect();
                        let shownall: Vec<_> = live.iter().map(|&f| shown(&scene, f)).collect();
                        out.push((live, shownall, scene.take_changes()));
                    }
                    counter::set_near_limit(None);
                    for f in fields {
                        if scene.field_targets(f).is_ok() {
                            assert_eq!(
                                shown(&scene, f),
                                min_of_singles(&mut scene, f),
                                "seed {seed}"
                            );
                        }
                    }
                    runs.push(out);
                }
            }
            for (k, run) in runs.iter().enumerate().skip(1) {
                assert!(*run == runs[0], "seed {seed}, run {k}");
            }
        }
    }

    /// A sheet A and a cube B, both Eager and deferring their fields, with
    /// nothing cached on them (no adjacency, no incidence) and no field.
    fn bare_pair(rng: &mut Lcg) -> (Scene, MeshId, MeshId) {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(sheet(10, 0.1), MeshOptions::default())
            .unwrap();
        let b = scene
            .add_mesh(cube([0.3, 0.3, -0.2], 0.4), MeshOptions::default())
            .unwrap();
        scene.set_defer_fields(true);
        let p = turned(rng.range(-20.0, 20.0), [0.0, 0.0, rng.range(-0.1, 0.1)]);
        scene.set_placement(b, p).unwrap();
        scene.take_changes();
        (scene, a, b)
    }

    /// Steps field `id` alone to the end with `Points(budget)`, abandoning
    /// at one in `abandon_one_in` step boundaries (0: never), and returns
    /// the steps that left it pending.
    fn step_field_all(
        scene: &mut Scene,
        id: FieldId,
        budget: usize,
        abandon_one_in: usize,
        rng: &mut Lcg,
    ) -> usize {
        let (mut pending, mut abandoned) = (0, 0);
        for _ in 0..1_000_000 {
            match scene
                .field_step_field(id, FieldBudget::Points(budget))
                .unwrap()
            {
                StepState::Idle => return pending,
                StepState::Pending => {
                    pending += 1;
                    // At most three times: an abandoned unit starts over.
                    if abandon_one_in > 0 && abandoned < 3 && rng.below(abandon_one_in) == 0 {
                        scene.abandon_field_step();
                        abandoned += 1;
                    }
                }
                StepState::Done => {}
            }
        }
        panic!("field_step_field never went idle");
    }

    #[test]
    fn deferred_add_matches_eager_add() {
        // The same fields added at once and added deferred, then stepped
        // alone with small budgets (abandoned at random boundaries too):
        // the same values, colours and reports, bit for bit.
        let mut rng = Lcg(0xadd5);
        let mut spread = 0;
        for round in 0..12 {
            let state = rng.0;
            let md = random_max_distance(&mut rng);
            let sign = random_sign(&mut rng);
            let budget = [1, 13, 500, usize::MAX][round % 4];
            let abandon = if round % 3 == 0 { 4 } else { 0 };
            let mut out = Vec::new();
            for deferred in [false, true] {
                let mut r = Lcg(state);
                let (mut scene, a, b) = bare_pair(&mut r);
                let desc = DistanceField {
                    sign,
                    ..field("d", a, b, md)
                };
                let id = if deferred {
                    let id = scene.add_distance_field_deferred(desc).unwrap();
                    assert!(!scene.field_status(id).unwrap().current);
                    assert!(
                        scene
                            .take_changes()
                            .iter()
                            .all(|c| c.fields_changed.is_empty())
                    );
                    spread += step_field_all(&mut scene, id, budget, abandon, &mut r);
                    id
                } else {
                    scene.add_distance_field(desc).unwrap()
                };
                assert!(scene.field_status(id).unwrap().current, "round {round}");
                scene
                    .set_coloring(
                        a,
                        Some(Coloring {
                            field: id,
                            rule: distance_rule(0.2, 0.1),
                        }),
                    )
                    .unwrap();
                let status = scene.field_target_status(id).unwrap();
                let nearest = scene.field_nearest(id).unwrap().to_vec();
                out.push((seen(&mut scene, [a, b]), status, nearest));
            }
            assert_eq!(out[0], out[1], "round {round} budget {budget}");
        }
        assert!(spread > 100, "{spread}");
    }

    #[test]
    fn deferred_add_of_several_targets_matches() {
        // A field measured to three meshes, added deferred and stepped one
        // point at a time, against the field added at once.
        let make = || {
            let (mut scene, [a, b, c, d], fields) = group_scene(UpdateMode::Eager);
            for f in fields {
                scene.remove_field(f).unwrap();
            }
            scene.set_defer_fields(true);
            scene.take_changes();
            (scene, [a, b, c, d])
        };
        let (mut eager, [a, b, c, d]) = make();
        let (mut stepped, _) = make();
        let desc = DistanceField {
            name: "multi".to_owned(),
            on: a,
            to: vec![b, c, d],
            max_distance: 0.3,
            sign: SignMode::Auto,
        };
        let fe = eager.add_distance_field(desc.clone()).unwrap();
        let fs = stepped.add_distance_field_deferred(desc).unwrap();
        let before = stepped.field_target_status(fs).unwrap();
        assert!(
            before
                .iter()
                .all(|s| !s.closed && s.mode == SignMode::Pseudonormal)
        );
        assert!(step_field_all(&mut stepped, fs, 1, 0, &mut Lcg(1)) > 50);
        assert_eq!(
            bits(eager.field_values(fe).unwrap()),
            bits(stepped.field_values(fs).unwrap())
        );
        assert_eq!(
            eager.field_nearest(fe).unwrap(),
            stepped.field_nearest(fs).unwrap()
        );
        assert_eq!(
            eager.field_target_status(fe).unwrap(),
            stepped.field_target_status(fs).unwrap()
        );
        assert_eq!(eager.take_changes(), stepped.take_changes());
    }

    #[test]
    fn sign_data_builds_in_steps_and_resumes() {
        // After an edit of B, the field on A to B rebuilds B's sign data in
        // the unit's steps; a unit abandoned partway leaves the build to be
        // resumed, so the steps add up to those of an unbroken build.
        let (mut scene, a, b, [fa, fb]) = deferred_pair_signed(0.2, SignMode::Winding);
        let (mut twin, ..) = deferred_pair_signed(0.2, SignMode::Winding);
        let (mut probe, ..) = deferred_pair_signed(0.2, SignMode::Winding);
        let edit = |s: &mut Scene| {
            s.edit(b, |tx| tx.set_positions(&[0], &[[0.31, 0.29, -0.21]]))
                .unwrap();
        };
        edit(&mut scene);
        edit(&mut twin);
        edit(&mut probe);
        counter::take_sign_steps();
        assert!(step_all(&mut probe, 1) > 0);
        let unbroken = counter::take_sign_steps();
        assert!(unbroken >= 4, "{unbroken}");
        // Partway, then abandoned, then a placement change of A (sign data
        // of B stay valid), then on to the end.
        assert_eq!(
            scene.field_step(FieldBudget::Points(1)),
            Ok(StepState::Pending)
        );
        assert_eq!(scene.field_step_stage(), Some(Stage::Sign(0)));
        for _ in 0..unbroken / 2 {
            assert_eq!(
                scene.field_step(FieldBudget::Points(1)),
                Ok(StepState::Pending)
            );
        }
        scene.abandon_field_step();
        scene.set_placement(a, at(0.0, 0.0, 0.01)).unwrap();
        twin.set_placement(a, at(0.0, 0.0, 0.01)).unwrap();
        step_all(&mut scene, 1);
        let total = counter::take_sign_steps();
        twin.update_fields().unwrap();
        assert!(
            total >= unbroken && total <= unbroken + 2,
            "{total} against {unbroken}"
        );
        assert_eq!(seen(&mut scene, [a, b]), seen(&mut twin, [a, b]));
        // Partway, then an edit of B: the build starts over, and the result
        // still matches.
        edit(&mut scene);
        edit(&mut twin);
        for _ in 0..5 {
            assert_eq!(
                scene.field_step(FieldBudget::Points(1)),
                Ok(StepState::Pending)
            );
        }
        let moved = |s: &mut Scene| {
            s.edit(b, |tx| tx.set_positions(&[1], &[[0.7, 0.31, -0.2]]))
                .unwrap();
        };
        moved(&mut scene);
        moved(&mut twin);
        step_all(&mut scene, 3);
        twin.update_fields().unwrap();
        assert_eq!(seen(&mut scene, [a, b]), seen(&mut twin, [a, b]));
        assert!(scene.field_status(fa).unwrap().current);
        assert!(scene.field_status(fb).unwrap().current);
    }

    #[test]
    fn near_search_runs_in_steps() {
        // A contact field on a large sheet: after a move, the near search
        // spreads over several one-point steps, and the result matches.
        let make = || {
            let mut scene = Scene::new(SceneOptions::default());
            let a = scene
                .add_mesh(sheet(120, 1.0 / 120.0), MeshOptions::default())
                .unwrap();
            let opts = MeshOptions {
                placement: at(0.0, 0.0, 0.01),
                ..MeshOptions::default()
            };
            let b = scene.add_mesh(sheet(120, 1.0 / 120.0), opts).unwrap();
            let f = scene.add_distance_field(field("d", a, b, 0.02)).unwrap();
            scene.set_defer_fields(true);
            scene
                .set_placement(b, turned(1.0, [0.0, 0.0, 0.012]))
                .unwrap();
            scene.take_changes();
            (scene, a, b, f)
        };
        counter::set_near_limit(Some(usize::MAX));
        let (mut scene, a, b, _) = make();
        let (mut twin, ..) = make();
        let mut near_steps = 0;
        for _ in 0..1_000_000 {
            if scene.field_step_stage() == Some(Stage::Near(0)) {
                near_steps += 1;
            }
            if scene.field_step(FieldBudget::Points(1)).unwrap() != StepState::Pending {
                break;
            }
        }
        twin.update_fields().unwrap();
        counter::set_near_limit(None);
        assert!(near_steps > 3, "{near_steps}");
        assert_eq!(seen(&mut scene, [a, b]), seen(&mut twin, [a, b]));
    }

    #[test]
    fn withdraw_field_only_before_it_is_computed() {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        scene.take_changes();
        let f = scene
            .add_distance_field_deferred(field("d", a, b, 0.1))
            .unwrap();
        assert_eq!(scene.withdraw_field(f), Ok(true));
        assert_eq!(scene.field_values(f), Err(SceneError::UnknownField));
        assert!(
            scene.take_changes().is_empty(),
            "a withdrawn field reported"
        );
        assert_eq!(scene.withdraw_field(f), Err(SceneError::UnknownField));
        // The name is free again; once computed, withdrawing does nothing.
        let f = scene
            .add_distance_field_deferred(field("d", a, b, 0.1))
            .unwrap();
        scene
            .field_step_field(f, FieldBudget::Points(usize::MAX))
            .unwrap();
        assert_eq!(scene.withdraw_field(f), Ok(false));
        assert!(scene.field_status(f).unwrap().current);
        // A colouring by a field keeps it too.
        let g = scene
            .add_distance_field_deferred(field("g", a, b, 0.1))
            .unwrap();
        let rule = contact_rule(0.1);
        scene
            .set_coloring(a, Some(Coloring { field: g, rule }))
            .unwrap();
        assert_eq!(scene.withdraw_field(g), Ok(false));
        // An Eager add computes at once: nothing to withdraw.
        let h = scene.add_distance_field(field("h", a, b, 0.1)).unwrap();
        assert_eq!(scene.withdraw_field(h), Ok(false));
    }

    #[test]
    fn field_step_field_steps_one_field() {
        let (mut scene, a, b, [fa, fb]) = deferred_pair(0.2);
        scene
            .set_placement(b, turned(5.0, [0.02, 0.0, 0.0]))
            .unwrap();
        scene.take_changes();
        // A unit over all due fields, then one over B's field alone.
        assert_eq!(
            scene.field_step(FieldBudget::Points(1)),
            Ok(StepState::Pending)
        );
        assert_eq!(
            scene.field_step_field(fb, FieldBudget::Points(1)),
            Ok(StepState::Pending)
        );
        assert_eq!(scene.field_step_meshes(), vec![b, a]);
        // `field_step` continues it.
        assert_eq!(
            scene.field_step(FieldBudget::Points(1)),
            Ok(StepState::Pending)
        );
        assert_eq!(scene.field_step_meshes(), vec![b, a]);
        while scene.field_step_field(fb, FieldBudget::Points(7)).unwrap() == StepState::Pending {}
        assert!(scene.field_status(fb).unwrap().current);
        assert!(!scene.field_status(fa).unwrap().current);
        assert_eq!(
            scene.field_step_field(fb, FieldBudget::Points(7)),
            Ok(StepState::Idle)
        );
        let changed: Vec<MeshId> = scene.take_changes().iter().map(|c| c.id).collect();
        assert!(!changed.contains(&a) || scene.colors(a).is_ok());
        assert_eq!(
            scene.field_step_field(FieldId(999), FieldBudget::Points(1)),
            Err(SceneError::UnknownField)
        );
    }

    #[test]
    fn sign_build_survives_edit_and_undo_of_a_manual_target() {
        // A Manual target B: undo brings its revision back, its BVH is not
        // updated (it reads current again) and its slots stay grown. A
        // partial sign build of B must start over, not resume.
        let make = || {
            let mut scene = Scene::new(SceneOptions::default());
            let a = scene
                .add_mesh(sheet(10, 0.1), MeshOptions::default())
                .unwrap();
            let b = scene
                .add_mesh(sheet(30, 1.0 / 30.0), with_mode(UpdateMode::Manual))
                .unwrap();
            scene.set_placement(b, at(0.0, 0.0, 0.05)).unwrap();
            let f = scene
                .add_distance_field_deferred(field("d", a, b, 0.2))
                .unwrap();
            scene.take_changes();
            (scene, a, b, f)
        };
        let (mut scene, a, b, f) = make();
        let (mut twin, ..) = make();
        for _ in 0..3 {
            assert_eq!(
                scene.field_step_field(f, FieldBudget::Points(1)),
                Ok(StepState::Pending)
            );
        }
        assert_eq!(scene.field_step_stage(), Some(Stage::Sign(0)));
        let grow = |s: &mut Scene| {
            s.edit(b, |tx| {
                let vs = tx.add_vertices(&[[5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [5.0, 6.0, 0.0]])?;
                tx.add_triangles(&[[vs[0], vs[1], vs[2]]])
            })
            .unwrap();
            s.undo().unwrap();
        };
        grow(&mut scene);
        grow(&mut twin);
        while scene.field_step_field(f, FieldBudget::Points(1)).unwrap() != StepState::Idle {}
        twin.update_fields().unwrap();
        assert_eq!(seen(&mut scene, [a, b]).0, seen(&mut twin, [a, b]).0);
        assert!(scene.field_status(f).unwrap().current);
        let g = twin.field_id(f.raw()).unwrap();
        assert_eq!(scene.field_target_status(f), twin.field_target_status(g));
    }

    #[test]
    fn stale_partial_sign_builds_are_dropped() {
        // Review M3: an unfinished sign build kept with its field goes once
        // its mesh changes (at the next call that changes the scene), and
        // when the field is updated in one call or removed.
        let (mut scene, _, b, [fa, _]) = deferred_pair_signed(0.2, SignMode::Winding);
        let edit = |s: &mut Scene, z: f32| {
            s.edit(b, |tx| tx.set_positions(&[0], &[[0.31, 0.29, z]]))
                .unwrap();
        };
        let partial = |s: &mut Scene| {
            for _ in 0..2 {
                assert_eq!(s.field_step(FieldBudget::Points(1)), Ok(StepState::Pending));
            }
            s.abandon_field_step();
        };
        edit(&mut scene, -0.21);
        partial(&mut scene);
        assert_eq!(scene.kept_sign_builds(), 1);
        // The target changes: stale, dropped by the next scene change.
        edit(&mut scene, -0.22);
        scene.set_defer_fields(true);
        scene.take_changes();
        scene.abandon_field_step();
        scene.set_placement(b, at(0.0, 0.0, 0.0)).unwrap();
        assert_eq!(scene.kept_sign_builds(), 0);
        // Updated in one call: dropped.
        partial(&mut scene);
        assert_eq!(scene.kept_sign_builds(), 1);
        scene.update_fields().unwrap();
        assert_eq!(scene.kept_sign_builds(), 0);
        // Removed with its field.
        edit(&mut scene, -0.23);
        partial(&mut scene);
        assert_eq!(scene.kept_sign_builds(), 1);
        scene.remove_field(fa).unwrap();
        assert_eq!(scene.kept_sign_builds(), 0);
    }

    #[test]
    fn stepped_unit_bit_identical_across_pools() {
        // Review M7: the same stepped units (a field build from scratch,
        // then a recompute after a move and one after an edit of the
        // target, abandoned once partway) on a one-thread and an
        // eight-thread pool, against the one-call update: bit for bit.
        use fast_mesh_store::par;
        let run = |threads: Option<usize>| {
            let go = || {
                let mut scene = Scene::new(SceneOptions::default());
                let a = scene
                    .add_mesh(sheet(90, 1.0 / 90.0), MeshOptions::default())
                    .unwrap();
                let opts = MeshOptions {
                    placement: at(0.0, 0.0, 0.04),
                    ..MeshOptions::default()
                };
                let b = scene.add_mesh(sheet(90, 1.0 / 90.0), opts).unwrap();
                scene.set_defer_fields(true);
                let f = scene
                    .add_distance_field_deferred(field("d", a, b, 0.1))
                    .unwrap();
                let one_call = threads.is_none();
                let settle = |s: &mut Scene| {
                    if one_call {
                        s.update_fields().unwrap();
                    } else {
                        for k in 0.. {
                            match s.field_step(FieldBudget::Points(97)).unwrap() {
                                StepState::Pending if k == 3 => s.abandon_field_step(),
                                StepState::Pending | StepState::Done => {}
                                _ => break,
                            }
                        }
                    }
                };
                if one_call {
                    scene.update_fields().unwrap();
                } else {
                    while scene.field_step_field(f, FieldBudget::Points(97)).unwrap()
                        != StepState::Idle
                    {}
                }
                let mut out = vec![seen(&mut scene, [a, b])];
                scene
                    .set_placement(b, turned(2.0, [0.01, 0.0, 0.035]))
                    .unwrap();
                settle(&mut scene);
                out.push(seen(&mut scene, [a, b]));
                scene
                    .edit(b, |tx| tx.set_positions(&[100], &[[0.5, 0.5, -0.01]]))
                    .unwrap();
                settle(&mut scene);
                out.push(seen(&mut scene, [a, b]));
                out
            };
            match threads {
                Some(k) => par::with_threads(k, go),
                None => {
                    par::set_force_sequential(true);
                    let out = go();
                    par::set_force_sequential(false);
                    out
                }
            }
        };
        let want = run(None);
        for k in [1, 8] {
            let got = run(Some(k));
            // Values and colours; reports may group differently around the
            // abandon, the values may not.
            for (g, w) in got.iter().zip(&want) {
                assert_eq!((&g.0, &g.1), (&w.0, &w.1), "{k} threads");
            }
        }
    }

    /// A box `[min, min + size]`, the triangulation of [`cube`].
    fn bx(min: [f32; 3], size: [f32; 3]) -> Mesh {
        let mut pos = Vec::new();
        for i in 0..8u32 {
            let b = |bit: u32, s: f32| if i & bit != 0 { s } else { 0.0 };
            pos.extend_from_slice(&[
                min[0] + b(1, size[0]),
                min[1] + b(2, size[1]),
                min[2] + b(4, size[2]),
            ]);
        }
        let idx = [
            0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4,
            6, 1, 3, 5, 3, 7, 5,
        ];
        Mesh::from_buffers(&pos, &idx).unwrap()
    }

    fn scaled_placement(translation: [f64; 3], scale: [f64; 3]) -> MeshOptions {
        MeshOptions {
            placement: Placement {
                rotation: Placement::IDENTITY.rotation,
                translation,
                scale,
            },
            ..MeshOptions::default()
        }
    }

    /// A scene with A (a unit cube at `[3, 0, 0]`, Eager) and B (a unit
    /// cube), B placed by `place_b`.
    fn scaled_pair(place_b: MeshOptions) -> (Scene, MeshId, MeshId) {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(cube([3.0, 0.0, 0.0], 1.0), MeshOptions::default())
            .unwrap();
        let b = scene.add_mesh(cube([0.0; 3], 1.0), place_b).unwrap();
        (scene, a, b)
    }

    fn scaled_values(scene: &mut Scene, on: MeshId, to: &[MeshId]) -> (Vec<f32>, Vec<u32>) {
        let desc = DistanceField {
            name: "d".to_owned(),
            on,
            to: to.to_vec(),
            max_distance: f64::INFINITY,
            sign: SignMode::Auto,
        };
        let id = scene.add_distance_field(desc).unwrap();
        (
            scene.field_values(id).unwrap().to_vec(),
            scene.field_nearest(id).unwrap().to_vec(),
        )
    }

    #[test]
    fn scaled_targets_match_rigid_twins() {
        // B scaled (uniform and non-uniform) measures like the rigid box of
        // the same world shape: A is the unit cube at x in [3, 4].
        for scale in [[2.0, 2.0, 2.0], [2.0, 1.0, 0.5], [0.5, 2.0, 1.5]] {
            let (mut scaled, a, b) = scaled_pair(scaled_placement([0.0; 3], scale));
            let (mut rigid, ra, rb) = scaled_pair(MeshOptions::default());
            // The rigid twin of B's world shape: [0, sx] x [0, sy] x [0, sz].
            let twin = rigid
                .add_mesh(
                    bx([0.0; 3], scale.map(|s| s as f32)),
                    MeshOptions::default(),
                )
                .unwrap();
            rigid.remove_mesh(rb).unwrap();
            let (got, got_near) = scaled_values(&mut scaled, a, &[b]);
            let (want, want_near) = scaled_values(&mut rigid, ra, &[twin]);
            assert_eq!(got.len(), want.len());
            assert_eq!(got_near, want_near, "scale {scale:?}");
            for (v, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    close(*g, *w, f64::INFINITY),
                    "scale {scale:?} vertex {v}: got {g}, want {w}"
                );
            }
            // Every value is finite and positive: A is outside B.
            assert!(got.iter().all(|&x| x > 0.0), "scale {scale:?}: {got:?}");
            // Vertices on A's x = 3 face with y, z inside B's read the x gap.
            let ma = scaled.mesh(a).unwrap();
            for (v, &x) in got.iter().enumerate() {
                let p = ma.position(v as VertexId).unwrap();
                let w = scaled.local_to_world(a, p.map(f64::from)).unwrap();
                // Inside B's y, z span (with room): the value is the x gap.
                if (w[0] - 3.0).abs() < 1e-9 && w[2] < scale[2] - 0.25 {
                    assert!((x - (3.0 - scale[0]) as f32).abs() < 1e-4, "v{v}: {x}");
                }
            }
        }
    }

    #[test]
    fn scaled_a_matches_rigid_twin() {
        // A scaled, B rigid: A's vertices carry the world distances. A's
        // world shape is [1.5, 3.5] x [0, 1]^2; the rigid twin has it
        // directly, with the same triangulation, so slot-wise.
        let scale = [2.0, 1.0, 1.0];
        let mut scaled = Scene::new(SceneOptions::default());
        let a = scaled
            .add_mesh(
                cube([0.0; 3], 1.0),
                scaled_placement([1.5, 0.0, 0.0], scale),
            )
            .unwrap();
        let b = scaled
            .add_mesh(cube([0.0; 3], 1.0), MeshOptions::default())
            .unwrap();
        let mut rigid = Scene::new(SceneOptions::default());
        let ra = rigid
            .add_mesh(bx([1.5, 0.0, 0.0], [2.0, 1.0, 1.0]), MeshOptions::default())
            .unwrap();
        let rb = rigid
            .add_mesh(cube([0.0; 3], 1.0), MeshOptions::default())
            .unwrap();
        let (got, got_near) = scaled_values(&mut scaled, a, &[b]);
        let (want, want_near) = scaled_values(&mut rigid, ra, &[rb]);
        assert_eq!(got_near, want_near);
        for (v, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                close(*g, *w, f64::INFINITY),
                "vertex {v}: got {g}, want {w}"
            );
        }
        // The x = 1.5 face reads the 0.5 gap to B.
        let ma = scaled.mesh(a).unwrap();
        for (v, &x) in got.iter().enumerate() {
            let p = ma.position(v as VertexId).unwrap();
            let w = scaled.local_to_world(a, p.map(f64::from)).unwrap();
            if (w[0] - 1.5).abs() < 1e-9 {
                assert!((x - 0.5).abs() < 1e-4, "v{v}: {x}");
            }
        }
    }

    #[test]
    fn scaled_multi_target_and_stepped_updates_match() {
        // Two targets, one scaled: the minimum (ties to the lowest index)
        // matches the rigid twins'.
        let (mut scaled, a, b) = scaled_pair(scaled_placement([0.0; 3], [2.0, 1.0, 1.0]));
        let c = scaled
            .add_mesh(cube([3.0, 2.0, 0.0], 1.0), MeshOptions::default())
            .unwrap();
        let (mut rigid, ra, _) = scaled_pair(MeshOptions::default());
        let twin = rigid
            .add_mesh(bx([0.0; 3], [2.0, 1.0, 1.0]), MeshOptions::default())
            .unwrap();
        let rc = rigid
            .add_mesh(cube([3.0, 2.0, 0.0], 1.0), MeshOptions::default())
            .unwrap();
        let (got, got_near) = scaled_values(&mut scaled, a, &[b, c]);
        let (want, want_near) = scaled_values(&mut rigid, ra, &[twin, rc]);
        assert_eq!(got_near, want_near);
        for (v, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                close(*g, *w, f64::INFINITY),
                "vertex {v}: got {g}, want {w}"
            );
        }
        // Stepped: the field is registered, a placement makes it due while
        // deferred, then it is stepped to Done with the eager twin's values.
        let (mut eager, ea, eb) = scaled_pair(scaled_placement([0.0; 3], [2.0, 1.0, 0.5]));
        let (mut stepped, sa, sb) = scaled_pair(scaled_placement([0.0; 3], [2.0, 1.0, 0.5]));
        for scene in [&mut eager, &mut stepped] {
            scene.set_defer_fields(true);
        }
        let desc = DistanceField {
            name: "d".to_owned(),
            on: sa,
            to: vec![sb],
            max_distance: f64::INFINITY,
            sign: SignMode::Auto,
        };
        // Registering computes at once even while deferred; the placement
        // then leaves the field due for the steps.
        let id = stepped.add_distance_field(desc).unwrap();
        assert!(stepped.field_status(id).unwrap().current);
        let nudge = Placement {
            translation: [0.01, 0.0, 0.0],
            ..stepped.placement(sb).unwrap()
        };
        stepped.set_placement(sb, nudge).unwrap();
        assert!(!stepped.field_status(id).unwrap().current);
        loop {
            match stepped.field_step(FieldBudget::Points(3)).unwrap() {
                StepState::Done => break,
                StepState::Pending => {}
                StepState::Idle => panic!("a due field idles"),
            }
        }
        eager.set_placement(eb, nudge).unwrap();
        let want = scaled_values(&mut eager, ea, &[eb]).0;
        assert_eq!(bits(stepped.field_values(id).unwrap()), bits(&want));
    }

    // ---- Stepped sign data refit, and the interaction tier (issue #173) ----

    #[test]
    fn stepped_position_edit_refits_sign_data_like_a_rebuild() {
        // A position-only edit of B: the stepped update refits B's sign data
        // in one step instead of a whole build, with the same bits as an
        // Eager update (which refits) and as a new field (a whole build).
        let (mut scene, a, b, ids) = deferred_pair(f64::INFINITY);
        let (mut twin, ..) = deferred_pair(f64::INFINITY);
        twin.set_defer_fields(false);
        let edit = |s: &mut Scene, y: f32| {
            s.edit(b, |tx| tx.set_positions(&[0], &[[0.31, y, -0.21]]))
                .unwrap();
        };
        for y in [0.29, 0.27, 0.33] {
            edit(&mut scene, y);
            edit(&mut twin, y);
            counter::take_sign_steps();
            step_all(&mut scene, 1);
            assert_eq!(counter::take_sign_steps(), 0, "a refit, not a build");
            assert_eq!(seen(&mut scene, [a, b]), seen(&mut twin, [a, b]));
        }
        scene.set_defer_fields(false);
        let again = scene
            .add_distance_field(field("again", a, b, f64::INFINITY))
            .unwrap();
        assert_eq!(
            bits(scene.field_values(again).unwrap()),
            bits(scene.field_values(ids[0]).unwrap())
        );
        // A topology edit cannot refit: it builds, in steps.
        scene.set_defer_fields(true);
        scene.edit(b, |tx| tx.remove_triangles(&[0])).unwrap();
        counter::take_sign_steps();
        step_all(&mut scene, 1);
        assert!(counter::take_sign_steps() > 0);
    }

    /// A pair of balls, A at the origin and B overlapping it, a field each
    /// way at `md`, both coloured.
    fn interacting_pair(md: f64) -> (Scene, MeshId, MeshId, [FieldId; 2]) {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(ball(), MeshOptions::default()).unwrap();
        let b = scene.add_mesh(ball(), MeshOptions::default()).unwrap();
        scene.set_placement(b, at(1.2, 0.0, 0.0)).unwrap();
        let fa = scene.add_distance_field(field("d", a, b, md)).unwrap();
        let fb = scene.add_distance_field(field("d", b, a, md)).unwrap();
        for (m, f) in [(a, fa), (b, fb)] {
            let rule = distance_rule(0.5, 0.5);
            scene
                .set_coloring(m, Some(Coloring { field: f, rule }))
                .unwrap();
        }
        scene.take_changes();
        (scene, a, b, [fa, fb])
    }

    #[test]
    fn interaction_recomputes_a_budget_then_settles_to_the_exact_field() {
        // Stepped and Eager updates, at an infinite bound and a wide one.
        for md in [f64::INFINITY, 10.0] {
            for stepped in [false, true] {
                let (mut scene, a, b, [fa, fb]) = interacting_pair(md);
                let slots = scene.field_values(fa).unwrap().len();
                if stepped {
                    scene.set_defer_fields(true);
                }
                scene.set_field_interaction(Some(slots / 8)).unwrap();
                assert_eq!(scene.field_interaction(), Some(slots / 8));
                for i in 1..=12 {
                    counter::take();
                    scene
                        .set_placement(b, at(1.2 + 0.01 * f64::from(i), 0.0, 0.0))
                        .unwrap();
                    if stepped {
                        step_all(&mut scene, 16);
                    }
                    let recomputed = counter::take();
                    // Two fields: each its near vertices and its sweep, well
                    // under the whole pair once the margin has settled.
                    if i > 6 {
                        assert!(recomputed < 2 * slots, "{recomputed} of {}", 2 * slots);
                    }
                    // Every vertex keeps a value: no far holes meanwhile.
                    for f in [fa, fb] {
                        assert!(scene.field_values(f).unwrap().iter().all(|x| x.is_finite()));
                    }
                }
                // The pointer stops: the exact field, bit for bit a new one.
                scene.set_field_interaction(None).unwrap();
                assert_eq!(scene.field_interaction(), None);
                if stepped {
                    assert!(!scene.field_status(fa).unwrap().current);
                    step_all(&mut scene, 16);
                    scene.set_defer_fields(false);
                }
                let exact_a = scene.add_distance_field(field("exact", a, b, md)).unwrap();
                let exact_b = scene.add_distance_field(field("exact", b, a, md)).unwrap();
                assert_eq!(
                    bits(scene.field_values(fa).unwrap()),
                    bits(scene.field_values(exact_a).unwrap())
                );
                assert_eq!(
                    bits(scene.field_values(fb).unwrap()),
                    bits(scene.field_values(exact_b).unwrap())
                );
                // And against an independent full computation.
                assert_matches(&mut scene, fa);
                assert_matches(&mut scene, fb);
                let rule = distance_rule(0.5, 0.5);
                assert_colors(&scene, a, &rule, fa);
                assert_colors(&scene, b, &rule, fb);
            }
        }
    }

    #[test]
    fn interaction_without_a_change_costs_nothing_when_it_ends() {
        let (mut scene, _, b, [fa, fb]) = interacting_pair(f64::INFINITY);
        scene.set_field_interaction(Some(4)).unwrap();
        scene.set_field_interaction(None).unwrap();
        assert!(scene.field_status(fa).unwrap().current);
        assert!(scene.field_status(fb).unwrap().current);
        // A field of at most the budget stays exact during it.
        scene.set_field_interaction(Some(1 << 20)).unwrap();
        scene.set_placement(b, at(1.25, 0.0, 0.0)).unwrap();
        assert_matches(&mut scene, fa);
        scene.set_field_interaction(None).unwrap();
        assert!(scene.field_status(fa).unwrap().current);
    }

    #[test]
    fn interaction_covers_an_edit_of_either_mesh_and_settles() {
        let (mut scene, a, b, [fa, fb]) = interacting_pair(f64::INFINITY);
        scene.set_field_interaction(Some(16)).unwrap();
        for (m, s) in [(a, 1.01f32), (b, 1.02), (a, 1.03)] {
            scene
                .edit(m, |tx| tx.set_positions(&[0], &[[s, 0.0, 0.0]]))
                .unwrap();
        }
        // During it, the sign data of the edited target were refitted.
        scene.set_field_interaction(None).unwrap();
        assert_matches(&mut scene, fa);
        assert_matches(&mut scene, fb);
    }

    /// A dab near the top of a unit ball.
    fn ball_dab(strength: f32) -> crate::Dab {
        crate::Dab::Inflate(fast_mesh_ops::InflateOptions {
            brush: fast_mesh_ops::Brush {
                center: [0.0, 0.0, 1.0],
                radius: 0.8,
                falloff: fast_mesh_ops::Falloff::Smooth,
                strength,
            },
        })
    }

    #[test]
    fn a_dab_marks_fields_due_and_a_stepped_update_refits() {
        // Scene::dab of a target and of the mesh a field is on, stepped as the
        // Worker steps it: due at once, refitted, with the bits of a new field.
        let (mut scene, a, b, [fa, fb]) = interacting_pair(f64::INFINITY);
        scene.set_defer_fields(true);
        for (m, s) in [(b, 0.02), (a, 0.03), (b, -0.01)] {
            assert!(scene.dab(m, &ball_dab(s)).unwrap().moved > 0);
            assert!(!scene.field_status(fa).unwrap().current);
            assert!(!scene.field_status(fb).unwrap().current);
            counter::take_sign_steps();
            step_all(&mut scene, 16);
            assert_eq!(counter::take_sign_steps(), 0, "a refit, not a build");
            assert!(scene.field_status(fa).unwrap().current);
            assert!(scene.field_status(fb).unwrap().current);
        }
        scene.set_defer_fields(false);
        let exact_a = scene
            .add_distance_field(field("exact", a, b, f64::INFINITY))
            .unwrap();
        let exact_b = scene
            .add_distance_field(field("exact", b, a, f64::INFINITY))
            .unwrap();
        assert_eq!(
            bits(scene.field_values(fa).unwrap()),
            bits(scene.field_values(exact_a).unwrap())
        );
        assert_eq!(
            bits(scene.field_values(fb).unwrap()),
            bits(scene.field_values(exact_b).unwrap())
        );
    }

    #[test]
    fn dabs_during_an_interaction_settle_to_the_exact_field() {
        for stepped in [false, true] {
            let (mut scene, a, b, [fa, fb]) = interacting_pair(f64::INFINITY);
            let before = scene.field_values(fa).unwrap().to_vec();
            scene.set_defer_fields(stepped);
            scene.set_field_interaction(Some(4)).unwrap();
            for (m, s) in [(a, 0.02), (b, 0.03), (a, 0.02)] {
                scene.dab(m, &ball_dab(s)).unwrap();
                if stepped {
                    step_all(&mut scene, 16);
                }
            }
            assert_ne!(bits(scene.field_values(fa).unwrap()), bits(&before));
            scene.set_field_interaction(None).unwrap();
            if stepped {
                step_all(&mut scene, 16);
                scene.set_defer_fields(false);
            }
            assert_matches(&mut scene, fa);
            assert_matches(&mut scene, fb);
        }
    }

    #[test]
    fn interaction_rejects_a_zero_budget() {
        let (mut scene, ..) = interacting_pair(f64::INFINITY);
        assert!(matches!(
            scene.set_field_interaction(Some(0)),
            Err(SceneError::InvalidInput(_))
        ));
        assert_eq!(scene.field_interaction(), None);
    }

    #[test]
    fn interaction_is_bit_identical_for_any_thread_count() {
        let run = |threads: Option<usize>| {
            let go = || {
                let (mut scene, a, b, [fa, _]) = interacting_pair(f64::INFINITY);
                let slots = scene.field_values(fa).unwrap().len();
                scene.set_defer_fields(true);
                scene.set_field_interaction(Some(slots / 8)).unwrap();
                let mut out = Vec::new();
                for i in 1..=8 {
                    scene
                        .set_placement(b, at(1.2 + 0.015 * f64::from(i), 0.0, 0.0))
                        .unwrap();
                    step_all(&mut scene, 64);
                    out.push(seen_values(&scene, [a, b]));
                }
                scene.set_field_interaction(None).unwrap();
                step_all(&mut scene, 64);
                out.push(seen_values(&scene, [a, b]));
                out
            };
            match threads {
                Some(k) => par::with_threads(k, go),
                None => {
                    par::set_force_sequential(true);
                    let out = go();
                    par::set_force_sequential(false);
                    out
                }
            }
        };
        let want = run(None);
        for k in [1, 8] {
            assert_eq!(run(Some(k)), want, "{k} threads");
        }
    }

    /// Whether the fields of a pair hold the bits of a new field, computed from
    /// scratch: the sign data they used then matched a full build.
    fn assert_like_a_new_field(scene: &mut Scene, id: FieldId, step: &str) {
        let (on, to, md, sign) = {
            let f = scene.fields.iter().find(|f| f.id == id).unwrap();
            (
                f.desc.on,
                f.desc.to.clone(),
                f.desc.max_distance,
                f.desc.sign,
            )
        };
        let defer = scene.defer_fields;
        scene.set_defer_fields(false);
        let fresh = scene
            .add_distance_field(DistanceField {
                name: "fresh".to_owned(),
                on,
                to,
                max_distance: md,
                sign,
            })
            .unwrap();
        let got = bits(scene.field_values(id).unwrap());
        let want = bits(scene.field_values(fresh).unwrap());
        scene.remove_field(fresh).unwrap();
        scene.set_defer_fields(defer);
        let off = got.iter().zip(&want).filter(|(g, w)| g != w).count();
        assert_eq!(off, 0, "{step}: {off} of {} values differ", got.len());
    }

    /// Mirrors every vertex of `m` in z: the surface stays, its normals turn
    /// over, so that sign data that miss it read every sign wrong.
    fn mirror(scene: &mut Scene, m: MeshId) {
        let mesh = &scene.entry(m).unwrap().mesh;
        let ids: Vec<VertexId> = (0..mesh.vertex_slots() as VertexId).collect();
        let at: Vec<[f32; 3]> = ids
            .iter()
            .map(|&v| {
                let p = mesh.position(v).unwrap();
                [p[0], p[1], -p[2]]
            })
            .collect();
        scene.edit(m, |tx| tx.set_positions(&ids, &at)).unwrap();
    }

    /// Runs what is due, stepped or at once.
    fn bring_up_to_date(scene: &mut Scene, stepped: bool) {
        if stepped {
            step_all(scene, 16);
        } else {
            scene.update_fields().unwrap();
        }
    }

    #[test]
    fn refit_after_a_baked_target_is_a_rebuild() {
        // While a target is non-uniformly scaled its real sign data are unused
        // and its edits go unseen by them: back to rigid, they must not pass
        // for current after a refit that sees no moved vertices.
        for stepped in [false, true] {
            let (mut scene, _, b, [fa, _]) = interacting_pair(f64::INFINITY);
            scene.set_defer_fields(stepped);
            let rigid = at(1.2, 0.0, 0.0);
            let scaled = Placement {
                scale: [2.0, 1.0, 1.0],
                ..rigid
            };
            scene.set_placement(b, scaled).unwrap();
            bring_up_to_date(&mut scene, stepped);
            mirror(&mut scene, b);
            bring_up_to_date(&mut scene, stepped);
            scene.set_placement(b, rigid).unwrap();
            bring_up_to_date(&mut scene, stepped);
            assert_like_a_new_field(&mut scene, fa, "rigid again");
            // And it goes on refitting afterwards.
            scene
                .edit(b, |tx| tx.set_positions(&[0], &[[1.1, 0.0, 0.0]]))
                .unwrap();
            bring_up_to_date(&mut scene, stepped);
            assert_like_a_new_field(&mut scene, fa, "an edit after it");
        }
    }

    #[test]
    fn sequences_of_edits_match_a_new_field() {
        // The differential test through the scene: edits, dabs, undo, redo,
        // rigid and scaled placements, interactions, abandoned units. After each
        // update the fields hold the bits of a new field.
        let mut rng = Lcg(0x5eed);
        for stepped in [false, true] {
            for round in 0..24 {
                let (mut scene, a, b, ids) = interacting_pair(f64::INFINITY);
                if round % 2 == 1 {
                    // Slivers in the target: a vertex on its neighbour.
                    let p = scene.entry(b).unwrap().mesh.position(7).unwrap();
                    scene.edit(b, |tx| tx.set_positions(&[6], &[p])).unwrap();
                }
                scene.set_defer_fields(stepped);
                let meshes = [a, b];
                let mut interacting = false;
                for op in 0..16 {
                    let r = (rng.next() * 1e9) as usize;
                    let m = meshes[(r >> 4) % 2];
                    let slots = scene.field_values(ids[0]).unwrap().len() as u32;
                    let v = (rng.next() * f64::from(slots)) as u32;
                    let what = r % 9;
                    let s = 0.8 + rng.next() as f32 / 2.0;
                    match what {
                        0..=2 => {
                            let _ = scene
                                .edit(m, |tx| tx.set_positions(&[v], &[[s, s - 1.0, 1.1 - s]]));
                        }
                        3 => {
                            scene.dab(m, &ball_dab(0.02)).unwrap();
                        }
                        8 => mirror(&mut scene, m),
                        4 => {
                            if scene.can_undo() {
                                scene.undo().unwrap();
                            }
                        }
                        5 => {
                            let scale = if r & 256 == 0 {
                                [1.0, 1.0, 1.0]
                            } else {
                                [1.5, 1.0, 1.0]
                            };
                            scene
                                .set_placement(
                                    b,
                                    Placement {
                                        scale,
                                        ..at(1.2, 0.0, f64::from(s) - 1.0)
                                    },
                                )
                                .unwrap();
                        }
                        6 => {
                            interacting = !interacting;
                            let points = interacting.then_some(4);
                            scene.set_field_interaction(points).unwrap();
                        }
                        _ => {
                            if stepped {
                                // A unit dropped partway.
                                let _ = scene.field_step(FieldBudget::Points(2));
                                scene.abandon_field_step();
                            }
                        }
                    }
                    bring_up_to_date(&mut scene, stepped);
                    if !interacting {
                        let step = format!("stepped {stepped}, round {round}, op {op} ({what})");
                        assert_like_a_new_field(&mut scene, ids[0], &step);
                        assert_like_a_new_field(&mut scene, ids[1], &step);
                    }
                }
            }
        }
    }
}
