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
//! **Stepped updates** ([`Scene::field_step`]) run the same parts: the due
//! Eager fields are prepared and their candidates picked when the unit
//! opens, the candidates' values are computed a few at a time over later
//! calls, and the fields are applied together after the last. No value,
//! colour, report or hook is written before that, so dropping the unit
//! leaves none behind (the BVHs and sign data opening it brought up to date
//! stay so, as after a query).

use fast_mesh_fields::{
    ColorRule, FieldsError, SignData, SignMode, SignStatus, Target, color_of,
    signed_distances_multi,
};
use fast_mesh_query::Rigid;
use fast_mesh_store::{ChangeSet, ChannelDesc, StoreError, VertexId, par};

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

/// The state of a field, from
/// [`Scene::field_status`](crate::Scene::field_status).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldStatus {
    /// True when the values match the meshes and placements as they are now.
    pub current: bool,
    /// What the latest sign data build found out about the first mesh of
    /// `to` (see [`Scene::field_target_status`](crate::Scene::field_target_status)
    /// for every target).
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

/// A due field whose meshes are prepared: what its update recomputes.
struct FieldJob {
    /// The field's index.
    i: usize,
    /// Every vertex is recomputed.
    full: bool,
    /// Per target: it moved (see the module docs).
    moved: Vec<bool>,
    /// A's triangles changed since the last update.
    topology: bool,
    /// A's merged changes since the last update (`None`: everything).
    changes: Option<ChangeSet>,
    /// Per target: the margin of its near set, when it is needed.
    near_margins: Vec<Option<f64>>,
    /// Per target: from its local space into A's, as the pair queries take
    /// it.
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
}

impl FieldJob {
    /// Some target moved.
    fn geometric(&self) -> bool {
        self.moved.contains(&true)
    }
}

/// The new state of a field, computed without writing anything.
struct FieldPlan {
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

/// A field of an open step unit.
struct UnitField {
    job: FieldJob,
    /// Taken by `plan` once every value is computed.
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

/// The Eager fields due at one state of the scene, stepped as one unit
/// (see [`Scene::field_step`](crate::Scene::field_step)). It holds field
/// indices and computed values only, so any change to the scene must drop
/// it first.
pub(crate) struct FieldUnit {
    fields: Vec<UnitField>,
    /// The first field with values left to compute.
    next: usize,
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
}

impl FieldHook {
    fn new(field: FieldId) -> FieldHook {
        FieldHook {
            field,
            changes: Some(ChangeSet::default()),
            placement_changed: false,
        }
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
    /// Each target's sign data, rebuilt when not current.
    sign: Vec<SignData>,
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
}

impl Entry {
    fn hook(&self, id: FieldId) -> Option<&FieldHook> {
        self.hooks.iter().find(|h| h.field == id)
    }

    fn reset_hook(&mut self, id: FieldId) {
        if let Some(h) = self.hooks.iter_mut().find(|h| h.field == id) {
            *h = FieldHook::new(id);
        }
    }

    /// Colours vertex slots added since the colours were last sized, from
    /// the field's channel (NaN for slots the field has not computed), and
    /// reports them, so that the colours always cover every vertex slot.
    pub(crate) fn grow_colors(&mut self) {
        let slots = self.mesh.vertex_slots();
        let Some(cs) = &mut self.coloring else {
            return;
        };
        let old = cs.colors.len();
        if slots <= old {
            return;
        }
        let rule = &cs.coloring.rule;
        let values = self.mesh.channel(&cs.channel).unwrap_or(&[]);
        cs.colors
            .extend((old..slots).map(|v| values.get(v).map_or(rule.nan, |&x| color_of(rule, x))));
        let added: Vec<VertexId> = (old as VertexId..slots as VertexId).collect();
        self.report_colors(Some(&added));
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
    /// Every edit of a target mesh rebuilds its sign data over the whole
    /// mesh (O(n), about 150 ms at 1M triangles in pseudonormal mode, more
    /// after a topology edit, which drops the cached adjacency); an
    /// incremental refit is planned for Phase 5b. Placement changes and
    /// edits of `desc.on` do not rebuild it.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`]; [`SceneError::InvalidInput`] when `to` is
    /// empty, holds `on` or a mesh twice, `max_distance` is negative or NaN,
    /// or `on` already has a field of that name; [`SceneError::StaleBvh`]
    /// when a target is a stale Manual mesh; a BVH, field or memory error.
    /// The scene is unchanged on error.
    pub fn add_distance_field(&mut self, desc: DistanceField) -> Result<FieldId> {
        self.drop_field_unit();
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
        {
            return Err(SceneError::InvalidInput(
                "the mesh already has a field of that name",
            ));
        }
        let next = self
            .next_field
            .checked_add(1)
            .ok_or(SceneError::InvalidInput("too many fields"))?;
        self.prepare(Some(&desc.to))?;
        let mut sign = Vec::new();
        sign.try_reserve_exact(desc.to.len()).map_err(|_| oom())?;
        for &to in &desc.to {
            let eb = self.entries.get_mut(to).ok_or(SceneError::UnknownMesh)?;
            sign.push(SignData::build(&mut eb.mesh, &eb.bvh, desc.sign)?);
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
            sign,
            nearest: Vec::new(),
            fresh: true,
            finite: Vec::new(),
            referenced: Vec::new(),
        });
        if eager {
            let _ = self.update_field(self.fields.len() - 1);
        }
        Ok(id)
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
        let i = self.field_index(id)?;
        self.detach_field(i);
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
    /// NaN for dead or unused vertices. Never updates.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field.
    pub fn field_values(&self, id: FieldId) -> Result<&[f32]> {
        let f = &self.fields[self.field_index(id)?];
        self.entry(f.desc.on)?
            .mesh
            .channel(&f.channel)
            .ok_or(SceneError::Corrupted)
    }

    /// Per vertex slot of the field's mesh, as of its last update: the index
    /// in [`Scene::field_targets`] of the target its value came from (the
    /// lowest on a tie), or `u32::MAX` for none (a `+∞` or NaN value, or a
    /// slot not computed yet). It can be shorter than the mesh's slots
    /// (slots added since the last update): those slots have none. Never
    /// updates.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field.
    pub fn field_nearest(&self, id: FieldId) -> Result<&[u32]> {
        Ok(&self.fields[self.field_index(id)?].nearest)
    }

    /// The field's target meshes, in index order. Removing a target mesh
    /// removes it from here, shifting the later ones down.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field.
    pub fn field_targets(&self, id: FieldId) -> Result<&[MeshId]> {
        Ok(&self.fields[self.field_index(id)?].desc.to)
    }

    /// What the latest sign data build found out about each of the field's
    /// targets, in index order. Never updates.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field.
    pub fn field_target_status(&self, id: FieldId) -> Result<Vec<SignStatus>> {
        let f = &self.fields[self.field_index(id)?];
        Ok(f.sign.iter().map(SignData::status).collect())
    }

    /// Whether the field is current, and what the latest sign data build
    /// found out about its target mesh. Never updates.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field.
    pub fn field_status(&self, id: FieldId) -> Result<FieldStatus> {
        let i = self.field_index(id)?;
        Ok(FieldStatus {
            current: self.field_current(i),
            sign: self.fields[i].sign[0].status(),
        })
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
                let f = &self.fields[self.field_index(c.field)?];
                if f.desc.on != mesh {
                    return Err(SceneError::InvalidInput("the field is not on this mesh"));
                }
                c.rule.validate().map_err(|e| match e {
                    FieldsError::InvalidInput(m) => SceneError::InvalidInput(m),
                    e => SceneError::Fields(e),
                })?;
                Some(ColorState {
                    coloring: c,
                    channel: f.channel.clone(),
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

    /// The id of the registered field with this [`FieldId::raw`] value, or
    /// `None` when there is none (never added, or removed).
    pub fn field_id(&self, raw: u32) -> Option<FieldId> {
        self.fields.iter().map(|f| f.id).find(|id| id.0 == raw)
    }

    fn field_index(&self, id: FieldId) -> Result<usize> {
        self.fields
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
        !f.fresh && clean(f.desc.on) && f.desc.to.iter().all(|&m| clean(m))
    }

    /// Updates the fields that are not current, whose mesh's mode passes
    /// `want`, and (with `touched`) that involve one of those meshes.
    /// Returns the first error; every field is attempted.
    pub(crate) fn run_fields(
        &mut self,
        want: impl Fn(UpdateMode) -> bool,
        touched: Option<&[MeshId]>,
    ) -> Result<()> {
        let due = self.due_fields(want, touched)?;
        self.update_due(&due)
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

    /// Computes part of the Eager fields that are due, so that a long
    /// update can be spread over several calls (a Worker yields between
    /// them).
    ///
    /// The first call opens a unit: it takes every due field on an Eager
    /// mesh (as an Eager update would), brings what each reads up to date
    /// (BVHs, sign data) and chooses the vertices each recomputes, including
    /// the near search (see the module docs). That part is not divided.
    /// Every call, the first included, then computes the values of up to
    /// `budget` of those vertices, across the unit's fields in field order,
    /// writing nothing. The call that computes the last ones applies every
    /// field of the unit together and returns [`StepState::Done`]: values,
    /// colours and reports are then the same, bit for bit, as from
    /// [`Scene::update_fields`] at the state the unit opened at.
    ///
    /// Any other `&mut self` call on the scene drops an open unit first
    /// (see [`Scene::abandon_field_step`]), except
    /// [`Scene::abandon_field_step`] and [`Scene::set_defer_fields`].
    ///
    /// Opening a unit is the one part that leaves a trace when the unit is
    /// then abandoned: the BVHs it brought up to date stay refitted and the
    /// sign data it rebuilt stays rebuilt, as after any query that needs
    /// them. No value, colour, report or pending change depends on whether
    /// that happened.
    ///
    /// # Errors
    /// Errors are isolated per field, as in [`Scene::update_fields`]: a
    /// field that fails, when the unit opens (for example
    /// [`SceneError::StaleBvh`] for a stale Manual mesh) or while its values
    /// are computed, is dropped from the unit and stays stale and due, as
    /// after a failed Eager update. The other fields go on, and the call
    /// that applies them returns the first error in field order instead of
    /// [`StepState::Done`]. When no due field can be opened, the first
    /// error is returned and no unit is opened.
    pub fn field_step(&mut self, budget: FieldBudget) -> Result<StepState> {
        let FieldBudget::Points(points) = budget;
        let mut unit = match self.field_unit.take() {
            Some(unit) => unit,
            None => match self.open_field_unit()? {
                Some(unit) => unit,
                None => return Ok(StepState::Idle),
            },
        };
        self.advance_field_unit(&mut unit, points.max(1));
        if unit.next < unit.fields.len() {
            self.field_unit = Some(unit);
            return Ok(StepState::Pending);
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
        match unit.error {
            Some((_, e)) => Err(e),
            None => Ok(StepState::Done),
        }
    }

    /// Drops the open step unit, if any: nothing it computed is written or
    /// reported, and its fields stay due. Does nothing when no unit is
    /// open. What opening the unit brought up to date (BVHs, sign data)
    /// stays so; see [`Scene::field_step`].
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
    }

    /// The candidates of the open step unit left to compute.
    #[cfg(test)]
    pub(crate) fn field_step_left(&self) -> usize {
        self.field_unit.as_ref().map_or(0, |u| {
            u.fields[u.next..]
                .iter()
                .map(|f| f.pick.candidates.len() - f.values.len())
                .sum()
        })
    }

    /// Opens a step unit over the due Eager fields: prepares each and picks
    /// its candidates, leaving out (with their errors) those that fail.
    /// `None` when no field is due; the first error when none opens.
    fn open_field_unit(&mut self) -> Result<Option<FieldUnit>> {
        let due = self.due_fields(|m| m == UpdateMode::Eager, None)?;
        if due.is_empty() {
            return Ok(None);
        }
        let mut unit = FieldUnit {
            fields: Vec::new(),
            next: 0,
            error: None,
        };
        let mut jobs = Vec::new();
        jobs.try_reserve_exact(due.len()).map_err(|_| oom())?;
        for &i in &due {
            match self.prepare_field(i) {
                Ok(job) => jobs.push(job),
                Err(e) => unit.fail(i, e),
            }
        }
        // Picking is the near search, which can be long: in parallel across
        // fields as in `compute_fields`. A field's error is its own result.
        let threshold = field_threshold(jobs.iter());
        let picks = par::map_chunks(&jobs, 1, threshold, oom, |_, chunk| {
            Ok(self.pick_candidates(&chunk[0]))
        })?;
        unit.fields
            .try_reserve_exact(jobs.len())
            .map_err(|_| oom())?;
        for (job, pick) in jobs.into_iter().zip(picks) {
            let pick = match pick {
                Ok(pick) => pick,
                Err(e) => {
                    unit.fail(job.i, e);
                    continue;
                }
            };
            let (mut values, mut nearest) = (Vec::new(), Vec::new());
            values
                .try_reserve_exact(pick.candidates.len())
                .map_err(|_| oom())?;
            nearest
                .try_reserve_exact(pick.candidates.len())
                .map_err(|_| oom())?;
            unit.fields.push(UnitField {
                job,
                pick,
                values,
                nearest,
                recomputed: 0,
                plan: None,
            });
        }
        if unit.fields.is_empty() {
            return Err(unit.error.map_or(SceneError::Corrupted, |(_, e)| e));
        }
        Ok(Some(unit))
    }

    /// Computes the values of up to `left` more candidates of `unit`, in
    /// field order, and the plan of each field whose values are all
    /// computed (a field without candidates costs none of `left`). A field
    /// that fails is dropped from the unit, its error recorded: it gets no
    /// plan, and its candidates left cost nothing.
    fn advance_field_unit(&self, unit: &mut FieldUnit, mut left: usize) {
        while let Some(f) = unit.fields.get_mut(unit.next) {
            let (done, total) = (f.values.len(), f.pick.candidates.len());
            let mut computed = Ok(());
            if done < total {
                if left == 0 {
                    break;
                }
                let end = done + left.min(total - done);
                let run = &f.pick.candidates[done..end];
                computed = self
                    .compute_values(&f.job, run, &mut f.values, &mut f.nearest)
                    .map(|n| f.recomputed += n);
                left -= end - done;
                if computed.is_ok() && end < total {
                    break;
                }
            }
            let pick = std::mem::take(&mut f.pick);
            let values = std::mem::take(&mut f.values);
            let nearest = std::mem::take(&mut f.nearest);
            let plan = computed
                .and_then(|()| self.finish_field(&f.job, pick, values, nearest, f.recomputed));
            let i = f.job.i;
            match plan {
                Ok(plan) => f.plan = Some(plan),
                Err(e) => unit.fail(i, e),
            }
            unit.next += 1;
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

    /// Brings what field `i` reads up to date, the part of an update that
    /// needs the meshes mutably: its BVHs (the targets' always, A's when a
    /// near set is needed) and the targets' sign data. Then decides what the
    /// update recomputes (see the module docs). Writes no field state.
    fn prepare_field(&mut self, i: usize) -> Result<FieldJob> {
        let (id, on) = (self.fields[i].id, self.fields[i].desc.on);
        let mut to = Vec::new();
        to.try_reserve_exact(self.fields[i].desc.to.len() + 1)
            .map_err(|_| oom())?;
        to.extend_from_slice(&self.fields[i].desc.to);
        let n = to.len();
        let (md, mode) = (self.fields[i].desc.max_distance, self.fields[i].desc.sign);
        let hook = |m: MeshId| -> Result<FieldHook> {
            self.entry(m)?
                .hook(id)
                .cloned()
                .ok_or(SceneError::Corrupted)
        };
        let hook_a = hook(on)?;
        let mut full = self.fields[i].fresh || md.is_infinite() || hook_a.changes.is_none();
        let mut moved = Vec::new();
        moved.try_reserve_exact(n).map_err(|_| oom())?;
        for &m in &to {
            let hook_b = hook(m)?;
            full |= hook_b.changes.is_none();
            moved.push(
                hook_a.placement_changed
                    || hook_b.placement_changed
                    || hook_b.changes.as_ref().is_some_and(|c| !c.is_empty()),
            );
        }
        // The targets' BVHs are always needed; A's only for a near set.
        let need_near = !full && moved.contains(&true);
        if need_near {
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
        for (k, &m) in to.iter().enumerate() {
            let eb = self.entries.get_mut(m).ok_or(SceneError::UnknownMesh)?;
            let f = &mut self.fields[i];
            if !f.sign[k].is_current(&eb.mesh, &eb.bvh) {
                f.sign[k] = SignData::build(&mut eb.mesh, &eb.bvh, mode)?;
            }
        }
        let mut b_to_a = Vec::new();
        b_to_a.try_reserve_exact(n).map_err(|_| oom())?;
        for &m in &to {
            b_to_a.push(self.relative_transform(on, m)?);
        }
        let ma = &self.entry(on)?.mesh;
        let changes = hook_a.changes;
        let topology = changes.as_ref().is_some_and(ChangeSet::topology_changed);
        let work = if full {
            ma.vertex_slots().saturating_mul(n)
        } else {
            let listed = changes.as_ref().map_or(0, |c| {
                c.vertices_added.len() + c.vertices_moved.len() + c.vertices_removed.len()
            });
            let known = if moved.contains(&true) {
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
        let f = &self.fields[job.i];
        let ea = self.entry(f.desc.on)?;
        let ma = &ea.mesh;
        let slots = ma.vertex_slots();
        let defined = |v: VertexId| ma.is_vertex_alive(v) && ma.corner_count(v) > 0;

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

        // The vertices to recompute, ascending.
        let candidates: Vec<VertexId> = if all {
            (0..slots as VertexId).collect()
        } else {
            let mut c = Vec::new();
            for &t in &near {
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
        })
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
                sign: &f.sign[k],
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
            if all {
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
        })
    }

    /// Writes a computed field: its values, its colours, its reports and its
    /// bookkeeping. Fails, writing nothing, only when the field's channel or
    /// mesh is missing, or memory runs out.
    fn apply_field(&mut self, job: FieldJob, plan: FieldPlan) -> Result<()> {
        let i = job.i;
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
            counter::add_applied(id);
        }
        for (&v, &x) in plan.candidates.iter().zip(&plan.values) {
            channel[v as usize] = x;
        }
        f.nearest.resize(slots, u32::MAX);
        for (&v, &n) in plan.candidates.iter().zip(&plan.nearest) {
            f.nearest[v as usize] = n;
        }
        ea.reset_hook(id);
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
        for &m in &f.desc.to {
            if let Some(eb) = self.entries.get_mut(m) {
                eb.reset_hook(id);
            }
        }
        f.fresh = false;
        f.finite = plan.finite;
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
        static APPLIED: std::cell::RefCell<Vec<super::FieldId>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Records that `field` was applied (its update written).
    pub(crate) fn add_applied(field: super::FieldId) {
        APPLIED.with(|c| c.borrow_mut().push(field));
    }

    /// The fields applied on this thread since the last call, in order.
    pub(crate) fn take_applied() -> Vec<super::FieldId> {
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
        // infinite max_distance): 13,122 candidates, enough to pick in
        // parallel across the two fields and for runs of 2,048 to compute
        // in parallel chunks.
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
        let picked = counter::take_parallel();
        let opened = par::parallel_entries();
        assert_eq!(scene.field_step_left(), 13_121);
        assert_eq!(
            scene.field_step(FieldBudget::Points(2048)),
            Ok(StepState::Pending)
        );
        let chunked = par::parallel_entries() - opened;
        if par::enabled() {
            assert_eq!(picked, 1, "candidates picked sequentially");
            assert!(opened > entries);
            assert!(chunked > 0, "values computed sequentially");
        } else {
            assert_eq!((picked, chunked), (0, 0));
        }
        assert!(step_all(&mut scene, 2048) > 0);
        twin.update_fields().unwrap();
        assert_eq!(seen(&mut scene, [a, b]), seen(&mut twin, [a, b]));
    }

    /// A sheet A and a cube B with a field each way, both coloured, Eager
    /// meshes deferring their fields, everything current and drained.
    fn deferred_pair(max_distance: f64) -> (Scene, MeshId, MeshId, [FieldId; 2]) {
        let (mut scene, a, b) = sheet_and_cube(UpdateMode::Eager);
        let fa = scene
            .add_distance_field(field("d", a, b, max_distance))
            .unwrap();
        let fb = scene
            .add_distance_field(field("d", b, a, max_distance))
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
        let calls: [(&str, Call); 38] = [
            ("add_mesh", |s, _| {
                s.add_mesh(sheet(1, 1.0), MeshOptions::default()).unwrap();
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
}
