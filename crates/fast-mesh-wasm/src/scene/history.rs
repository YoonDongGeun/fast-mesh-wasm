//! `WasmScene` bindings for the history's memory budget
//! (`Scene::set_history_budget`), the report of steps larger than it, and
//! entries of the application's own (`Scene::record_entry`).
//!
//! Byte counts cross the boundary as `f64`, exact far beyond any memory.

use fast_mesh_scene::{EntryEvent, HistoryEntry, HistorySide, MeshId, PermanentStep};
use wasm_bindgen::prelude::*;

use super::WasmScene;
use crate::error::Error;

#[wasm_bindgen]
impl WasmScene {
    /// The most bytes the history may hold.
    pub fn history_budget(&self) -> f64 {
        self.scene.history_budget() as f64
    }

    /// Changes the most bytes the history may hold, and meets a smaller
    /// budget at once. `bytes` is rounded down and clamped to the address
    /// space (infinity is the largest budget). It must be 0 or more.
    pub fn set_history_budget(&mut self, bytes: f64) -> Result<(), Error> {
        if bytes.is_nan() || bytes < 0.0 {
            return Err(Error::invalid("the history budget must be 0 or more"));
        }
        // The cast saturates.
        Ok(self.scene.set_history_budget(bytes as usize)?)
    }

    /// The bytes the history holds.
    pub fn history_bytes(&self) -> f64 {
        self.scene.history_bytes() as f64
    }

    /// About the bytes an undoable removal of the mesh would hold.
    pub fn removal_bytes(&self, index: u32, generation: u32) -> Result<f64, Error> {
        Ok(self.scene.removal_bytes(self.id(index, generation)?)? as f64)
    }

    /// Whether `take_permanent` or `take_entry_events` has something to
    /// return.
    pub fn has_history_events(&self) -> bool {
        self.scene.has_permanent() || self.scene.has_entry_events()
    }

    /// Records the application's entry `id` in the history, counting `bytes`
    /// (clamped) against its budget. Within an open step it merges with the
    /// change before it when that is an entry with the same `key`; 0 is none.
    pub fn record_entry(&mut self, id: u32, bytes: f64, key: u32) -> Result<(), Error> {
        Ok(self.scene.record_entry(HistoryEntry {
            id: u64::from(id),
            // The cast saturates, and takes NaN as 0.
            bytes: bytes as u64,
            merge: (key != 0).then_some(u64::from(key)),
        })?)
    }

    /// Takes what happened to the application's entries since the last call,
    /// in order, as `[what, id]` each: `what` is 0 for undone, 1 for redone
    /// and 2 for dropped (see `Scene::take_entry_events`).
    pub fn take_entry_events(&mut self) -> Vec<u32> {
        // Ids come from `record_entry`, so they fit.
        let pair = |e| match e {
            EntryEvent::Undone(id) => [0, id as u32],
            EntryEvent::Redone(id) => [1, id as u32],
            EntryEvent::Dropped(id) => [2, id as u32],
        };
        let events = self.scene.take_entry_events();
        events.into_iter().flat_map(pair).collect()
    }

    /// Takes the steps the history could not keep since the last call (see
    /// `Scene::take_permanent`), as `[side, bytes, meshes, gone]` each: `side`
    /// is 0 when the step cannot be undone and 1 when it cannot be redone,
    /// and the two mesh lists are `n, (index, generation) × n`.
    pub fn take_permanent(&mut self) -> Vec<f64> {
        let mut out = Vec::new();
        for step in self.scene.take_permanent() {
            let PermanentStep {
                lost,
                meshes,
                gone,
                bytes,
            } = step;
            out.push(match lost {
                HistorySide::Undo => 0.0,
                HistorySide::Redo => 1.0,
            });
            out.push(bytes as f64);
            push_ids(&mut out, &meshes);
            push_ids(&mut out, &gone);
        }
        out
    }
}

fn push_ids(out: &mut Vec<f64>, ids: &[MeshId]) {
    out.push(ids.len() as f64);
    for id in ids {
        out.extend([f64::from(id.index()), f64::from(id.generation())]);
    }
}

#[cfg(test)]
mod tests {
    use crate::error::codes;
    use crate::scene::WasmScene;

    const QUAD: [f64; 12] = [0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.];

    fn add(s: &mut WasmScene) -> (u32, u32) {
        let id = s
            .add_mesh(&QUAD, &[0, 1, 2, 0, 2, 3], None, 0, true)
            .unwrap();
        (id[0], id[1])
    }

    #[test]
    fn budget_is_set_met_and_checked() {
        let mut s = WasmScene::new();
        assert_eq!(s.history_budget(), 268_435_456.0);
        let (i, g) = add(&mut s);
        let bytes = s.removal_bytes(i, g).unwrap();
        s.remove_mesh_undoable(i, g).unwrap();
        assert!((s.history_bytes() - bytes).abs() < 256.0);
        assert_eq!(s.removal_bytes(i, g).unwrap_err().code, codes::UNKNOWN_MESH);
        s.set_history_budget(s.history_bytes() + 0.75).unwrap();
        assert!(s.is_restorable(i, g));
        s.set_history_budget(0.0).unwrap();
        assert!(!s.is_restorable(i, g) && s.history_bytes() == 0.0);
        for bad in [-1.0, f64::NAN, f64::NEG_INFINITY] {
            assert_eq!(
                s.set_history_budget(bad).unwrap_err().code,
                codes::INVALID_ARGUMENT
            );
        }
        s.set_history_budget(f64::INFINITY).unwrap();
        assert_eq!(s.history_budget(), usize::MAX as f64);
    }

    #[test]
    fn permanent_steps_are_encoded() {
        let mut s = WasmScene::new();
        assert!(!s.has_history_events() && s.take_permanent().is_empty());
        s.set_history_budget(16.0).unwrap();
        let (a, ag) = add(&mut s);
        let (b, bg) = add(&mut s);
        s.begin_step();
        s.set_placement(
            a,
            ag,
            &[1., 0., 0., 0., 1., 0., 0., 0., 1.],
            &[1., 0., 0.],
            None,
        )
        .unwrap();
        s.remove_mesh_undoable(b, bg).unwrap();
        s.end_step().unwrap();
        assert!(s.has_history_events());
        let got = s.take_permanent();
        assert!(got[1] > 16.0);
        let id = |i: u32, g: u32| [f64::from(i), f64::from(g)];
        let want = [
            &[0.0, got[1], 2.0][..],
            &id(a, ag),
            &id(b, bg),
            &[1.0],
            &id(b, bg),
        ]
        .concat();
        assert_eq!(got, want);
        // An add too large to be redone.
        let n = s
            .add_mesh_undoable(&QUAD, &[0, 1, 2, 0, 2, 3], None, None, 0, true)
            .unwrap();
        s.undo().unwrap();
        let got = s.take_permanent();
        assert_eq!(got[0], 1.0);
        assert_eq!(
            got[2..],
            [1.0, n[0].into(), n[1].into(), 1.0, n[0].into(), n[1].into()]
        );
        assert!(!s.has_history_events());
    }

    #[test]
    fn entries_and_their_events_cross() {
        let mut s = WasmScene::new();
        s.record_entry(1, 100.0, 0).unwrap();
        let one = s.history_bytes();
        assert!(one > 100.0 && one < 300.0);
        assert!(s.can_undo() && !s.has_history_events());
        s.begin_step();
        for id in 2..6 {
            s.record_entry(id, 0.0, 9).unwrap();
        }
        s.record_entry(6, f64::NAN, 0).unwrap();
        s.record_entry(7, -1.0, 0).unwrap();
        s.end_step().unwrap();
        assert!(s.has_history_events());
        assert_eq!(s.take_entry_events(), [2, 3, 2, 4]);
        assert!(!s.has_history_events() && s.take_entry_events().is_empty());
        s.undo().unwrap();
        assert_eq!(s.take_entry_events(), [0, 7, 0, 6, 0, 2]);
        s.redo().unwrap();
        assert_eq!(s.take_entry_events(), [1, 5, 1, 6, 1, 7]);
        s.record_entry(u32::MAX, f64::INFINITY, u32::MAX).unwrap();
        assert!(!s.can_undo());
        assert_eq!(
            s.take_entry_events(),
            [2, 1, 2, 2, 2, 5, 2, 6, 2, 7, 2, u32::MAX]
        );
        assert_eq!(s.take_permanent()[2..], [0.0, 0.0]);
    }
}
