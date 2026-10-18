//! Change reports: what happened to each mesh since the last `take_changes`.

use fast_mesh_store::ChangeSet;

use crate::MeshId;

/// What happened to one mesh since the last
/// [`Scene::take_changes`](crate::Scene::take_changes).
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
}

/// The pending report of a live mesh.
pub(crate) struct Report {
    /// Merged changes; `None` means "everything".
    pub(crate) changes: Option<ChangeSet>,
    pub(crate) placement_changed: bool,
    pub(crate) added: bool,
}

impl Report {
    /// The report of a mesh that was just added.
    pub(crate) fn added() -> Report {
        Report {
            changes: None,
            placement_changed: false,
            added: true,
        }
    }

    /// A report with nothing in it.
    pub(crate) fn clean() -> Report {
        Report {
            changes: Some(ChangeSet::default()),
            placement_changed: false,
            added: false,
        }
    }

    /// True when there is nothing to report.
    pub(crate) fn is_clean(&self) -> bool {
        !self.added
            && !self.placement_changed
            && self.changes.as_ref().is_some_and(ChangeSet::is_empty)
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

    /// Turns the report into its public form and resets it to clean.
    pub(crate) fn take(&mut self, id: MeshId, removed: bool) -> MeshChanges {
        let r = std::mem::replace(self, Report::clean());
        MeshChanges {
            id,
            changes: r.changes,
            placement_changed: r.placement_changed,
            added: r.added,
            removed,
        }
    }
}
