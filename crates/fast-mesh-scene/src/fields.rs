//! Distance fields between meshes of the scene, and the colours derived
//! from them (spec §6, §11).
//!
//! A field's values live in a derived store channel `field:<name>` on mesh
//! A (`on`), one `f32` per vertex slot: the signed distance to mesh B
//! (`to`) as placed in the world, `+∞` beyond `max_distance`, NaN for dead
//! slots and for alive vertices no alive triangle uses.
//!
//! **What an update recomputes** (§11.6). Each field has a change
//! accumulator on A and on B (a [`FieldHook`], fed like a subscriber's
//! report). Everything is recomputed when either says "everything"
//! (`add_mesh`, `compact`, a `Corrupted` step), when `max_distance` is
//! infinite, or on the first update. Otherwise:
//!
//! - after a placement change of either mesh or an edit of B: the vertices
//!   of A's triangles within `pad(max_distance)` of B (`max_distance`
//!   plus a relative `1e-6` and `8·f32::EPSILON` times the coordinate
//!   magnitude of the pair), plus the vertices that
//!   held finite values (so they can return to far);
//! - after an edit of A: its added, moved and removed vertices, plus, when
//!   its triangles changed, the vertices whose "used by a triangle" state
//!   flipped.
//!
//! A vertex's value depends only on its own position and on B, so nothing
//! else can change. Rebuilding B's sign data after an edit of B can flip
//! signs only of finite values, which the near set covers. This relies on
//! undo reporting revived slots as added (the store's `ChangeSet::merge`
//! reports a reused slot as moved).

use fast_mesh_fields::{
    ColorRule, FieldsError, SignData, SignMode, SignStatus, color_of, signed_distances,
};
use fast_mesh_store::{ChangeSet, ChannelDesc, StoreError, VertexId};

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
    /// The mesh the distances are measured to (mesh B).
    pub to: MeshId,
    /// The largest distance computed (inclusive); vertices farther away hold
    /// `+∞`. Zero or greater, may be infinite.
    pub max_distance: f64,
    /// How the sign is decided on mesh `to`.
    pub sign: SignMode,
}

/// The state of a field, from
/// [`Scene::field_status`](crate::Scene::field_status).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldStatus {
    /// True when the values match the meshes and placements as they are now.
    pub current: bool,
    /// What the latest sign data build found out about mesh `to`.
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
    /// B's sign data, rebuilt when not current.
    sign: SignData,
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
    /// Registers a signed distance field on mesh `desc.on`, measured to mesh
    /// `desc.to`, and returns its id. Its values go into a derived channel
    /// `field:<name>` on `desc.on`, readable with [`Scene::field_values`] or
    /// through [`Scene::mesh`]; they are not undoable.
    ///
    /// The field follows the [`UpdateMode`] of `desc.on`: an Eager field is
    /// computed right away and after every edit, placement change, undo,
    /// redo or cancel touching either mesh; a Lazy one by [`Scene::flush`]
    /// or [`Scene::update_fields`]; a Manual one only by
    /// [`Scene::update_fields`]. Until then its values are NaN.
    ///
    /// B's BVH is brought up to date under the query rules (a Lazy one is
    /// updated) and its sign data built here, so that
    /// [`Scene::field_status`] can report B's topology at once.
    ///
    /// Every edit of the target mesh `desc.to` rebuilds its sign data over
    /// the whole mesh (O(n), about 150 ms at 1M triangles in pseudonormal
    /// mode, more after a topology edit, which drops the cached adjacency);
    /// an incremental refit is planned for Phase 5b. Placement changes and
    /// edits of `desc.on` do not rebuild it.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`]; [`SceneError::InvalidInput`] when `on`
    /// equals `to`, `max_distance` is negative or NaN, or `on` already has a
    /// field of that name; [`SceneError::StaleBvh`] when `to` is a stale
    /// Manual mesh; a BVH, field or memory error. The scene is unchanged on
    /// error.
    pub fn add_distance_field(&mut self, desc: DistanceField) -> Result<FieldId> {
        self.entry(desc.on)?;
        self.entry(desc.to)?;
        if desc.on == desc.to {
            return Err(SceneError::InvalidInput(
                "a field needs two different meshes",
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
        self.prepare(Some(&[desc.to]))?;
        let eb = self
            .entries
            .get_mut(desc.to)
            .ok_or(SceneError::UnknownMesh)?;
        let sign = SignData::build(&mut eb.mesh, &eb.bvh, desc.sign)?;
        eb.hooks.try_reserve(1).map_err(|_| oom())?;
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
        if let Some(eb) = self.entries.get_mut(desc.to) {
            eb.hooks.push(FieldHook::new(id));
        }
        self.fields.push(Field {
            id,
            desc,
            channel,
            sign,
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

    /// Whether the field is current, and what the latest sign data build
    /// found out about its target mesh. Never updates.
    ///
    /// # Errors
    /// [`SceneError::UnknownField`] for an unknown or removed field.
    pub fn field_status(&self, id: FieldId) -> Result<FieldStatus> {
        let i = self.field_index(id)?;
        Ok(FieldStatus {
            current: self.field_current(i),
            sign: self.fields[i].sign.status(),
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

    fn field_index(&self, id: FieldId) -> Result<usize> {
        self.fields
            .iter()
            .position(|f| f.id == id)
            .ok_or(SceneError::UnknownField)
    }

    /// Computed at least once, and nothing changed on either mesh since.
    fn field_current(&self, i: usize) -> bool {
        let f = &self.fields[i];
        let clean = |m: MeshId| {
            self.entry(m)
                .ok()
                .and_then(|e| e.hook(f.id))
                .is_some_and(FieldHook::is_clean)
        };
        !f.fresh && clean(f.desc.on) && clean(f.desc.to)
    }

    /// Updates the fields that are not current, whose mesh's mode passes
    /// `want`, and (with `touched`) that involve one of those meshes.
    /// Returns the first error; every field is attempted.
    pub(crate) fn run_fields(
        &mut self,
        want: impl Fn(UpdateMode) -> bool,
        touched: Option<&[MeshId]>,
    ) -> Result<()> {
        let mut first = Ok(());
        for i in 0..self.fields.len() {
            let (on, to) = (self.fields[i].desc.on, self.fields[i].desc.to);
            if touched.is_some_and(|t| !t.contains(&on) && !t.contains(&to)) {
                continue;
            }
            let Ok(ea) = self.entry(on) else { continue };
            if !want(ea.mode) || self.field_current(i) {
                continue;
            }
            let r = self.update_field(i);
            if first.is_ok() {
                first = r;
            }
        }
        first
    }

    /// Updates the Eager fields involving `touched` (all with `None`).
    /// Failures are swallowed: the field stays stale (§11.2).
    pub(crate) fn run_eager_fields(&mut self, touched: Option<&[MeshId]>) {
        let _ = self.run_fields(|m| m == UpdateMode::Eager, touched);
    }

    /// Removes the fields on `mesh` or measured to it.
    pub(crate) fn drop_fields_of(&mut self, mesh: MeshId) {
        let mut i = 0;
        while i < self.fields.len() {
            if self.fields[i].desc.on == mesh || self.fields[i].desc.to == mesh {
                self.detach_field(i);
            } else {
                i += 1;
            }
        }
    }

    fn detach_field(&mut self, i: usize) {
        let f = self.fields.remove(i);
        if let Some(eb) = self.entries.get_mut(f.desc.to) {
            eb.hooks.retain(|h| h.field != f.id);
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

    /// Brings field `i` up to date (see the module docs for what is
    /// recomputed). All or nothing: on error nothing is written and the
    /// pending changes stay.
    fn update_field(&mut self, i: usize) -> Result<()> {
        let (id, on, to) = (
            self.fields[i].id,
            self.fields[i].desc.on,
            self.fields[i].desc.to,
        );
        let (md, mode) = (self.fields[i].desc.max_distance, self.fields[i].desc.sign);
        let hook_a = self
            .entry(on)?
            .hook(id)
            .cloned()
            .ok_or(SceneError::Corrupted)?;
        let hook_b = self
            .entry(to)?
            .hook(id)
            .cloned()
            .ok_or(SceneError::Corrupted)?;
        let full = self.fields[i].fresh
            || md.is_infinite()
            || hook_a.changes.is_none()
            || hook_b.changes.is_none();
        let geometric = hook_a.placement_changed
            || hook_b.placement_changed
            || hook_b.changes.as_ref().is_some_and(|c| !c.is_empty());
        // B's BVH is always needed; A's only for the near set.
        let need_near = !full && geometric;
        let both = [on, to];
        self.prepare(Some(if need_near { &both } else { &both[1..] }))?;
        let mut near = Vec::new();
        if need_near {
            let m = self.pair_magnitude(on, to)?;
            let mut near_b = Vec::new();
            self.triangles_near(on, to, pad(md, m), &mut near, &mut near_b)?;
        }
        {
            let eb = self.entries.get_mut(to).ok_or(SceneError::UnknownMesh)?;
            let f = &mut self.fields[i];
            if !f.sign.is_current(&eb.mesh, &eb.bvh) {
                f.sign = SignData::build(&mut eb.mesh, &eb.bvh, mode)?;
            }
        }
        let xf = self.relative_transform(on, to)?.inverse();
        let (ea, eb) = (self.entry(on)?, self.entry(to)?);
        let f = &self.fields[i];
        let ma = &ea.mesh;
        let slots = ma.vertex_slots();
        let defined = |v: VertexId| ma.is_vertex_alive(v) && ma.corner_count(v) > 0;
        let topology = hook_a
            .changes
            .as_ref()
            .is_some_and(ChangeSet::topology_changed);

        // The vertices to recompute, ascending.
        let candidates: Vec<VertexId> = if full {
            (0..slots as VertexId).collect()
        } else {
            let mut c = Vec::new();
            for &t in &near {
                if let Some(tri) = ma.triangle(t) {
                    c.extend_from_slice(&tri);
                }
            }
            if geometric {
                c.extend_from_slice(&f.finite);
            }
            if let Some(ch) = &hook_a.changes {
                c.extend_from_slice(&ch.vertices_added);
                c.extend_from_slice(&ch.vertices_moved);
                c.extend_from_slice(&ch.vertices_removed);
            }
            if topology {
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

        // A's points in B's local space, in f64 then rounded to f32 (§5.2).
        let mut ids = Vec::new();
        let mut points = Vec::new();
        for &v in &candidates {
            if let Some(p) = ma.position(v).filter(|_| defined(v)) {
                ids.push(v);
                points.push(xf.apply(p).map(|c| c as f32));
            }
        }
        let mut out = vec![0.0f32; points.len()];
        signed_distances(&eb.mesh, &eb.bvh, &f.sign, &points, round_up(md), &mut out)?;
        #[cfg(test)]
        counter::add(points.len());

        let old = ma.channel(&f.channel).ok_or(SceneError::Corrupted)?;
        let mut values = Vec::with_capacity(candidates.len());
        let mut changed = Vec::new();
        let mut k = 0;
        for &v in &candidates {
            let x = if ids.get(k) == Some(&v) {
                let d = out[k];
                k += 1;
                // The kernel ran with the bound rounded up to f32.
                if f64::from(d.abs()) > md {
                    f32::INFINITY
                } else {
                    d
                }
            } else {
                f32::NAN
            };
            if old.get(v as usize).map(|o| o.to_bits()) != Some(x.to_bits()) {
                changed.push(v);
            }
            values.push(x);
        }
        let referenced = (full || topology || f.referenced.len() != slots)
            .then(|| (0..slots as VertexId).map(defined).collect::<Vec<bool>>());
        let finite: Vec<VertexId> = {
            let now = candidates
                .iter()
                .zip(&values)
                .filter(|(_, x)| x.is_finite())
                .map(|(&v, _)| v);
            if full {
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

        // Commit: nothing below fails.
        let ea = self.entries.get_mut(on).ok_or(SceneError::UnknownMesh)?;
        let channel = ea.mesh.derived_channel_mut(&self.fields[i].channel)?;
        for (&v, &x) in candidates.iter().zip(&values) {
            channel[v as usize] = x;
        }
        ea.reset_hook(id);
        if !changed.is_empty() {
            ea.report_field(id);
        }
        if let Some(cs) = &mut ea.coloring
            && cs.coloring.field == id
        {
            let values = ea.mesh.channel(&cs.channel).unwrap_or(&[]);
            let dirty = cs.update(values, &changed);
            ea.report_colors(Some(&dirty));
        }
        if let Some(eb) = self.entries.get_mut(to) {
            eb.reset_hook(id);
        }
        let f = &mut self.fields[i];
        f.fresh = false;
        f.finite = finite;
        if let Some(r) = referenced {
            f.referenced = r;
        }
        Ok(())
    }
}

/// Test-only count of the vertices field updates recompute.
#[cfg(test)]
pub(crate) mod counter {
    use std::cell::Cell;

    thread_local! {
        static RECOMPUTED: Cell<usize> = const { Cell::new(0) };
    }

    /// Counts `n` recomputed vertices.
    pub(crate) fn add(n: usize) {
        RECOMPUTED.with(|c| c.set(c.get() + n));
    }

    /// Vertices recomputed on this thread since the last call.
    pub(crate) fn take() -> usize {
        RECOMPUTED.with(|c| c.replace(0))
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
            to,
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
        let (on, to, md, mode) = (f.desc.on, f.desc.to, f.desc.max_distance, f.desc.sign);
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
        scene.remove_field(g).unwrap();
        assert_eq!(scene.field_values(g), Err(SceneError::UnknownField));
        assert!(scene.mesh(a).unwrap().channel("field:d").is_none());
        scene.add_distance_field(field("d", a, c, 0.1)).unwrap();
        // Editing goes on without the removed fields.
        scene
            .edit(a, |tx| tx.set_positions(&[45], &[[0.5, 0.4, 0.05]]))
            .unwrap();
        scene.update_fields().unwrap();
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
}
