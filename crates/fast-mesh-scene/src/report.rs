//! Change reports: what happened to each mesh since a subscriber last took
//! its changes.

use fast_mesh_store::{ChangeSet, VertexId};

use crate::MeshId;

/// A consumer of change reports, from [`Scene::subscribe`](crate::Scene::subscribe).
///
/// Every subscriber has its own pending reports, merged the same way, and
/// drains them with
/// [`Scene::take_changes_for`](crate::Scene::take_changes_for). Ids are never
/// reused within a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubscriberId(pub(crate) u64);

impl SubscriberId {
    /// The default subscriber, behind
    /// [`Scene::take_changes`](crate::Scene::take_changes). It always exists.
    pub(crate) const DEFAULT: SubscriberId = SubscriberId(0);
}

/// A field registered on a mesh of the scene.
///
/// Opaque: field ids are handed out by the scene and only compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FieldId(pub(crate) u32);

/// What happened to one mesh since the subscriber's last
/// [`Scene::take_changes_for`](crate::Scene::take_changes_for) (or
/// [`Scene::take_changes`](crate::Scene::take_changes) for the default
/// subscriber).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshChanges {
    /// The mesh. For a removed mesh this id no longer matches.
    pub id: MeshId,
    /// The merged changes of every commit, undo and redo since the last
    /// call. `None` means "treat everything as changed": the mesh was added
    /// or compacted, or an undo or redo touching it failed with
    /// [`SceneError::Corrupted`](crate::SceneError::Corrupted).
    pub changes: Option<ChangeSet>,
    /// True if the placement changed at any point since the last call, even
    /// if it was changed back.
    pub placement_changed: bool,
    /// The mesh was added since the last call.
    pub added: bool,
    /// The mesh was removed since the last call.
    pub removed: bool,
    /// The vertices whose colours changed since the last call, ascending
    /// without duplicates. `Some` of an empty list when no colour changed;
    /// `None` means "every colour may have changed" (redraw them all).
    pub colors_dirty: Option<Vec<VertexId>>,
    /// The fields on this mesh whose values changed since the last call,
    /// each once.
    pub fields_changed: Vec<FieldId>,
    /// The channels written through
    /// [`Scene::derived_channel_mut`](crate::Scene::derived_channel_mut),
    /// added with [`Scene::add_channel`](crate::Scene::add_channel) or
    /// removed with [`Scene::remove_channel`](crate::Scene::remove_channel)
    /// since the last call, each name once, in the order first reported. A
    /// `derived_channel_mut` call counts as a write whether or not the slice
    /// is then written.
    pub channels_written: Vec<String>,
}

/// The pending report of a live mesh, for one subscriber.
#[derive(Clone)]
pub(crate) struct Report {
    /// Merged changes; `None` means "everything".
    pub(crate) changes: Option<ChangeSet>,
    pub(crate) placement_changed: bool,
    pub(crate) added: bool,
    /// Recoloured vertices; `None` means "all".
    pub(crate) colors_dirty: Option<Vec<VertexId>>,
    pub(crate) fields_changed: Vec<FieldId>,
    pub(crate) channels_written: Vec<String>,
}

impl Report {
    /// The report of a mesh that was just added: everything is new,
    /// colours included.
    pub(crate) fn added() -> Report {
        Report {
            changes: None,
            added: true,
            colors_dirty: None,
            ..Report::clean()
        }
    }

    /// A report with nothing in it.
    pub(crate) fn clean() -> Report {
        Report {
            changes: Some(ChangeSet::default()),
            placement_changed: false,
            added: false,
            colors_dirty: Some(Vec::new()),
            fields_changed: Vec::new(),
            channels_written: Vec::new(),
        }
    }

    /// True when there is nothing to report.
    pub(crate) fn is_clean(&self) -> bool {
        !self.added
            && !self.placement_changed
            && self.changes.as_ref().is_some_and(ChangeSet::is_empty)
            && self.colors_dirty.as_ref().is_some_and(Vec::is_empty)
            && self.fields_changed.is_empty()
            && self.channels_written.is_empty()
    }

    /// Folds the changes of a later commit into the report. A report that
    /// already says "everything" stays that way.
    ///
    /// `ChangeSet::merge` cannot fail: its allocations abort the process on
    /// exhaustion rather than return an error, so this never falls back to
    /// `None`. That only happens on `add_mesh`, `compact` and a `Corrupted`
    /// undo or redo.
    pub(crate) fn merge(&mut self, later: &ChangeSet) {
        if let Some(c) = &mut self.changes {
            c.merge(later);
        }
    }

    /// Notes a write to channel `name`, once per name.
    pub(crate) fn channel_written(&mut self, name: &str) {
        if !self.channels_written.iter().any(|n| n == name) {
            self.channels_written.push(name.to_owned());
        }
    }

    /// Notes a change of field `id`'s values, once per field.
    pub(crate) fn field_changed(&mut self, id: FieldId) {
        if !self.fields_changed.contains(&id) {
            self.fields_changed.push(id);
        }
    }

    /// Notes recoloured vertices (`ids` ascending without duplicates), or
    /// every vertex for `None`.
    pub(crate) fn colors_changed(&mut self, ids: Option<&[VertexId]>) {
        match (ids, &mut self.colors_dirty) {
            (_, None) => {}
            (None, dirty) => *dirty = None,
            (Some(ids), Some(dirty)) => {
                if dirty.is_empty() {
                    dirty.extend_from_slice(ids);
                } else if !ids.is_empty() {
                    dirty.extend_from_slice(ids);
                    dirty.sort_unstable();
                    dirty.dedup();
                }
            }
        }
    }

    /// Turns the report into its public form and resets it to clean.
    pub(crate) fn take(&mut self, id: MeshId, removed: bool) -> MeshChanges {
        let r = std::mem::replace(self, Report::clean());
        MeshChanges {
            id,
            changes: r.changes,
            placement_changed: r.placement_changed,
            added: r.added,
            removed,
            colors_dirty: r.colors_dirty,
            fields_changed: r.fields_changed,
            channels_written: r.channels_written,
        }
    }
}
