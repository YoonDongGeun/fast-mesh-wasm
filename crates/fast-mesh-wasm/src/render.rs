//! Render-side state of each mesh and the per-frame change records.
//!
//! Plain Rust, no wasm-bindgen: the binding layer owns one [`RenderState`]
//! per mesh, feeds it every [`MeshChanges`] it takes from the scene, and
//! ships the resulting [`Record`]s to JavaScript as one `Uint32Array`
//! built by [`encode`].
//!
//! # Record format
//!
//! Every value is a `u32`. The whole message is
//! `[record_count, record...]`, and each record is, in order:
//!
//! 1. `mesh_index, mesh_generation, flags`, where `flags` is the sum of:
//!    - `1` added
//!    - `2` removed
//!    - `4` placement changed
//!    - `8` capacity changed (vertex or triangle slot count differs from
//!      the previous record of the mesh); every buffer is then "all", since
//!      the GPU buffers must be recreated and filled again
//!    - `16` positions all
//!    - `32` normals all
//!    - `64` colours all
//!    - `128` indices all
//! 2. four range lists, for positions, normals, colours and indices, each
//!    `n, start0, count0, ..., start(n-1), count(n-1)`. Ranges are in
//!    elements (vertex slots, or triangle slots for indices), ascending and
//!    disjoint. `n` is 0 when the buffer is "all" (its flag is set) or
//!    unchanged.
//! 3. `fields_n, field ids...`
//! 4. `sections_n, section ids...`
//!
//! The TypeScript decoder mirrors this layout.

use fast_mesh_scene::MeshChanges;
use fast_mesh_store::{Mesh, StoreError};

/// Dirty slots that are at most this many slots apart share one range.
pub(crate) const RANGE_GAP: u32 = 8;

pub(crate) const FLAG_ADDED: u32 = 1;
pub(crate) const FLAG_REMOVED: u32 = 2;
pub(crate) const FLAG_PLACEMENT: u32 = 4;
pub(crate) const FLAG_CAPACITY: u32 = 8;
pub(crate) const FLAG_POSITIONS_ALL: u32 = 16;
pub(crate) const FLAG_NORMALS_ALL: u32 = 32;
pub(crate) const FLAG_COLORS_ALL: u32 = 64;
pub(crate) const FLAG_INDICES_ALL: u32 = 128;

/// The buffer sizes last reported for one mesh. The render buffers
/// themselves live in the scene: positions and triangles in the store
/// (slot-aligned, dead triangle slots read `0, 0, 0`), normals and colours
/// in the scene's per-mesh state.
pub(crate) struct RenderState {
    vertex_slots: usize,
    triangle_slots: usize,
    /// Length of the normals buffer at the last report.
    normals_len: usize,
    /// A previous [`RenderState::apply`] failed after its changes were
    /// taken, so the next one reports everything.
    stale: bool,
}

/// The slots of each render buffer that changed in one report: ascending
/// without duplicates, or `None` for "all".
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Dirty {
    pub positions: Option<Vec<u32>>,
    pub normals: Option<Vec<u32>>,
    pub colors: Option<Vec<u32>>,
    /// Triangle slots.
    pub indices: Option<Vec<u32>>,
    /// The vertex or triangle slot count changed since the previous report.
    /// Every list is then `None`.
    pub capacity_changed: bool,
}

impl Dirty {
    /// Everything changed, slot counts included: the fallback when the
    /// dirty sets cannot be built.
    pub(crate) fn all() -> Dirty {
        Dirty {
            positions: None,
            normals: None,
            colors: None,
            indices: None,
            capacity_changed: true,
        }
    }

    /// Nothing changed (the dirty sets of a removed mesh).
    pub(crate) fn clean() -> Dirty {
        Dirty {
            positions: Some(Vec::new()),
            normals: Some(Vec::new()),
            colors: Some(Vec::new()),
            indices: Some(Vec::new()),
            capacity_changed: false,
        }
    }
}

/// One mesh's entry in a change message; see the module docs for its
/// encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Record {
    pub mesh_index: u32,
    pub mesh_generation: u32,
    pub added: bool,
    pub removed: bool,
    pub placement_changed: bool,
    pub dirty: Dirty,
    pub fields: Vec<u32>,
    pub sections: Vec<u32>,
}

fn oom() -> StoreError {
    StoreError::OutOfMemory
}

/// Buffers over this many slots report "all" past half dirty (see
/// [`slots`]); smaller ones sort cheaply and keep exact ranges.
const DENSE_MIN_SLOTS: usize = 8;

/// The ids of `lists` below `limit`, ascending without duplicates, or
/// `None` for "all" when past half the slots are dirty: a dense frame
/// uploads everything anyway, and skipping the sort avoids the `O(n log n)`
/// work. Tiny buffers are exempt (see [`DENSE_MIN_SLOTS`]).
///
/// Only in-range ids count toward the gate; out-of-range ids are dropped.
/// The lists are expected unique, as `ChangeSet` lists are, so a
/// duplicate-heavy input may still report "all" (a safe superset).
fn slots(lists: &[&[u32]], limit: usize) -> Result<Option<Vec<u32>>, StoreError> {
    let in_range: usize = lists
        .iter()
        .flat_map(|l| l.iter())
        .filter(|s| (**s as usize) < limit)
        .count();
    if limit > DENSE_MIN_SLOTS && in_range > limit / 2 {
        return Ok(None);
    }
    let mut out = Vec::new();
    let raw: usize = lists.iter().map(|l| l.len()).sum();
    out.try_reserve_exact(raw).map_err(|_| oom())?;
    for l in lists {
        out.extend(l.iter().copied().filter(|&s| (s as usize) < limit));
    }
    out.sort_unstable();
    out.dedup();
    Ok(Some(out))
}

impl RenderState {
    /// The render state of `mesh`, just added: its first report says
    /// "all" anyway, so only the slot counts matter.
    pub(crate) fn new(mesh: &Mesh) -> RenderState {
        RenderState {
            vertex_slots: mesh.vertex_slots(),
            triangle_slots: mesh.triangle_slots(),
            normals_len: 0,
            stale: false,
        }
    }

    /// The report of `mesh` when the scene had no change for it but its
    /// slot counts still differ from the last report (slots added, then
    /// undone before the changes were taken): every buffer "all", with the
    /// capacity change. `None` when the counts match.
    pub(crate) fn unreported_growth(&mut self, mesh: &Mesh, normals_len: usize) -> Option<Dirty> {
        let (vs, ts) = (mesh.vertex_slots(), mesh.triangle_slots());
        if vs == self.vertex_slots && ts == self.triangle_slots {
            return None;
        }
        self.vertex_slots = vs;
        self.triangle_slots = ts;
        self.normals_len = normals_len;
        self.stale = false;
        Some(Dirty::all())
    }

    /// Turns `changes`, the scene's report for `mesh`, into the dirty slots
    /// of every render buffer, and remembers the buffer sizes.
    /// `normals_len` is the current length (in vertices) of the mesh's
    /// normals buffer, 0 when normals are off; it may differ from the vertex
    /// slot count after the scene ran out of memory.
    ///
    /// Positions are dirty for moved, added and removed vertices, indices
    /// for added, removed and rewritten triangles, normals and colours as
    /// the report says. Every set is clamped to its buffer's current length.
    /// A `None` change set makes positions and indices "all"; normals are
    /// also "all" when their buffer length changed. A slot count change
    /// makes every buffer "all".
    ///
    /// # Errors
    /// [`StoreError::OutOfMemory`]. The state is then unchanged, apart from
    /// remembering that the next call must report everything, since the
    /// scene has already handed out these changes.
    pub(crate) fn apply(
        &mut self,
        mesh: &Mesh,
        changes: &MeshChanges,
        normals_len: usize,
    ) -> Result<Dirty, StoreError> {
        let r = self.try_apply(mesh, changes, normals_len);
        self.stale = r.is_err();
        r
    }

    fn try_apply(
        &mut self,
        mesh: &Mesh,
        changes: &MeshChanges,
        normals_len: usize,
    ) -> Result<Dirty, StoreError> {
        let (vs, ts) = (mesh.vertex_slots(), mesh.triangle_slots());
        let capacity_changed = vs != self.vertex_slots || ts != self.triangle_slots;
        if capacity_changed {
            // New GPU buffers: the adapter uploads everything anyway.
            self.vertex_slots = vs;
            self.triangle_slots = ts;
            self.normals_len = normals_len;
            return Ok(Dirty::all());
        }
        let set = changes.changes.as_ref().filter(|_| !self.stale);
        // Every allocation comes first, so a failure changes nothing.
        let positions = match set {
            None => None,
            Some(c) => slots(
                &[&c.vertices_moved, &c.vertices_added, &c.vertices_removed],
                vs,
            )?,
        };
        let indices = match set {
            None => None,
            Some(c) => slots(
                &[
                    &c.triangles_added,
                    &c.triangles_removed,
                    &c.triangles_changed,
                ],
                ts,
            )?,
        };
        let normals = match &changes.normals_dirty {
            Some(n) if !self.stale && normals_len == self.normals_len => slots(&[n], normals_len)?,
            _ => None,
        };
        let colors = match &changes.colors_dirty {
            Some(c) if !self.stale => slots(&[c], vs)?,
            _ => None,
        };
        self.normals_len = normals_len;
        Ok(Dirty {
            positions,
            normals,
            colors,
            indices,
            capacity_changed: false,
        })
    }
}

/// Merges ascending slot ids into `(start, count)` ranges. Slots with at
/// most `gap` missing slots between them share a range, so a range may
/// cover a few clean slots.
pub(crate) fn merge_ranges(sorted_slots: &[u32], gap: u32) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for &s in sorted_slots {
        match out.last_mut() {
            // In u64, so slots near `u32::MAX` cannot overflow; a repeated
            // slot merges too.
            Some((start, count))
                if u64::from(s) <= u64::from(*start) + u64::from(*count) + u64::from(gap) =>
            {
                *count = (*count).max(s - *start + 1);
            }
            _ => out.push((s, 1)),
        }
    }
    out
}

/// Writes `records` into `out` (cleared first) in the format of the module
/// docs.
pub(crate) fn encode(records: &[Record], out: &mut Vec<u32>) {
    out.clear();
    out.push(records.len() as u32);
    for r in records {
        let d = &r.dirty;
        let flag = |on: bool, bit: u32| if on { bit } else { 0 };
        // A capacity change makes every buffer "all".
        let all = |set: &Option<Vec<u32>>| d.capacity_changed || set.is_none();
        let flags = flag(r.added, FLAG_ADDED)
            | flag(r.removed, FLAG_REMOVED)
            | flag(r.placement_changed, FLAG_PLACEMENT)
            | flag(d.capacity_changed, FLAG_CAPACITY)
            | flag(all(&d.positions), FLAG_POSITIONS_ALL)
            | flag(all(&d.normals), FLAG_NORMALS_ALL)
            | flag(all(&d.colors), FLAG_COLORS_ALL)
            | flag(all(&d.indices), FLAG_INDICES_ALL);
        out.extend_from_slice(&[r.mesh_index, r.mesh_generation, flags]);
        for set in [&d.positions, &d.normals, &d.colors, &d.indices] {
            // A capacity change sends no ranges: every buffer is "all".
            let ranges = set
                .as_deref()
                .filter(|_| !d.capacity_changed)
                .map(|s| merge_ranges(s, RANGE_GAP))
                .unwrap_or_default();
            out.push(ranges.len() as u32);
            for (start, count) in ranges {
                out.extend_from_slice(&[start, count]);
            }
        }
        for ids in [&r.fields, &r.sections] {
            out.push(ids.len() as u32);
            out.extend_from_slice(ids);
        }
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_scene::{MeshChanges, MeshId, MeshOptions, Placement, Scene, SceneOptions};
    use fast_mesh_store::Mesh;

    use super::*;

    /// Two triangles over four vertices.
    fn quad() -> Mesh {
        let pos = [0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.];
        Mesh::from_buffers(&pos, &[0, 1, 2, 0, 2, 3]).unwrap()
    }

    /// A scene holding `quad()`, its render state, and the initial report
    /// already applied.
    fn setup() -> (Scene, MeshId, RenderState) {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(quad(), MeshOptions::default()).unwrap();
        let mut state = RenderState::new(scene.mesh(id).unwrap());
        let added = take(&mut scene, id);
        state.apply(scene.mesh(id).unwrap(), &added, 0).unwrap();
        (scene, id, state)
    }

    fn take(scene: &mut Scene, id: MeshId) -> MeshChanges {
        let mut all = scene.take_changes();
        let i = all.iter().position(|c| c.id == id).expect("a report");
        all.swap_remove(i)
    }

    fn step(scene: &mut Scene, id: MeshId, state: &mut RenderState) -> Dirty {
        let ch = take(scene, id);
        state.apply(scene.mesh(id).unwrap(), &ch, 0).unwrap()
    }

    #[test]
    fn dense_slots_report_all() {
        // Past half dirty on a buffer over the gate: "all".
        let every: Vec<u32> = (0..100).collect();
        assert_eq!(slots(&[&every], 100), Ok(None));
        // Sparse stays exact, whatever the buffer size...
        assert_eq!(slots(&[&[3, 1]], 100), Ok(Some(vec![1, 3])));
        // ...and tiny buffers keep exact ranges even past half.
        assert_eq!(slots(&[&[0, 1, 3, 9]], 2), Ok(Some(vec![0, 1])));
    }

    #[test]
    fn dense_gate_ignores_out_of_range_ids() {
        // Out-of-range ids are dropped, so they must not trip the gate: 10
        // in-range ids on a 100-slot buffer stay exact however many strays
        // pad the input.
        let padded: Vec<u32> = (0..10).chain(1000..1100).collect();
        assert_eq!(slots(&[&padded], 100), Ok(Some((0..10).collect())));
    }

    #[test]
    fn merge_ranges_gap_rule() {
        assert_eq!(merge_ranges(&[1, 2, 3, 12, 13], 8), vec![(1, 13)]);
        assert_eq!(merge_ranges(&[1, 2, 3, 20], 8), vec![(1, 3), (20, 1)]);
        assert_eq!(merge_ranges(&[], 8), vec![]);
        assert_eq!(merge_ranges(&[0, 9, 19], 8), vec![(0, 10), (19, 1)]);
        assert_eq!(merge_ranges(&[4, 6], 0), vec![(4, 1), (6, 1)]);
        assert_eq!(merge_ranges(&[4, 5], 0), vec![(4, 2)]);
        assert_eq!(merge_ranges(&[4, 4, 5], 8), vec![(4, 2)]);
        assert_eq!(
            merge_ranges(&[u32::MAX - 1, u32::MAX], 8),
            vec![(u32::MAX - 1, 2)]
        );
    }

    /// The render index buffer: the store's slot-aligned triangles.
    fn indices(scene: &Scene, id: MeshId) -> &[u32] {
        scene.mesh(id).unwrap().triangles().as_flattened()
    }

    #[test]
    fn added_mesh_reports_everything() {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(quad(), MeshOptions::default()).unwrap();
        let mut state = RenderState::new(scene.mesh(id).unwrap());
        let ch = take(&mut scene, id);
        let d = state.apply(scene.mesh(id).unwrap(), &ch, 0).unwrap();
        assert_eq!(d.positions, None);
        assert_eq!(d.indices, None);
        assert_eq!(d.normals, None);
        assert_eq!(d.colors, None);
        assert!(!d.capacity_changed);
        assert_eq!(indices(&scene, id), &[0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn indices_zero_for_removed_triangles() {
        let (mut scene, id, mut state) = setup();
        scene.edit(id, |tx| tx.remove_triangles(&[0])).unwrap();
        let d = step(&mut scene, id, &mut state);
        assert_eq!(indices(&scene, id), &[0, 0, 0, 0, 2, 3]);
        assert_eq!(d.indices, Some(vec![0]));
        assert_eq!(d.positions, Some(vec![]));
        assert!(!d.capacity_changed);
    }

    #[test]
    fn indices_follow_triangles_changed() {
        let (mut scene, id, mut state) = setup();
        scene
            .edit(id, |tx| tx.set_triangles(&[1], &[[3, 2, 1]]))
            .unwrap();
        let d = step(&mut scene, id, &mut state);
        assert_eq!(indices(&scene, id), &[0, 1, 2, 3, 2, 1]);
        assert_eq!(d.indices, Some(vec![1]));
        // Undo puts the old triangle back.
        scene.undo().unwrap();
        let d = step(&mut scene, id, &mut state);
        assert_eq!(indices(&scene, id), &[0, 1, 2, 0, 2, 3]);
        assert_eq!(d.indices, Some(vec![1]));
    }

    #[test]
    fn capacity_changed_on_slot_count_change() {
        let (mut scene, id, mut state) = setup();
        scene
            .edit(id, |tx| {
                let v = tx.add_vertices(&[[2., 0., 0.]])?;
                tx.add_triangles(&[[1, v[0], 2]])
            })
            .unwrap();
        let d = step(&mut scene, id, &mut state);
        // Growth means new GPU buffers: every buffer is "all".
        assert_eq!(d, Dirty::all());
        assert_eq!(indices(&scene, id), &[0, 1, 2, 0, 2, 3, 1, 4, 2]);

        // Moving and removing keep the slot counts.
        scene
            .edit(id, |tx| tx.set_positions(&[0], &[[0., 0., 1.]]))
            .unwrap();
        let d = step(&mut scene, id, &mut state);
        assert!(!d.capacity_changed);
        assert_eq!(d.positions, Some(vec![0]));
        assert_eq!(d.indices, Some(vec![]));
        scene.edit(id, |tx| tx.remove_triangles(&[2])).unwrap();
        let d = step(&mut scene, id, &mut state);
        assert!(!d.capacity_changed);
        assert_eq!(&indices(&scene, id)[6..], &[0, 0, 0]);
    }

    #[test]
    fn placement_only_change_has_no_position_ranges() {
        let (mut scene, id, mut state) = setup();
        let moved = Placement {
            translation: [5., 0., 0.],
            ..Placement::IDENTITY
        };
        scene.set_placement(id, moved).unwrap();
        let ch = take(&mut scene, id);
        assert!(ch.placement_changed);
        let dirty = state.apply(scene.mesh(id).unwrap(), &ch, 0).unwrap();
        assert_eq!(dirty, Dirty::clean());
        let mut out = Vec::new();
        let record = Record {
            mesh_index: 0,
            mesh_generation: 0,
            added: false,
            removed: false,
            placement_changed: true,
            dirty,
            fields: vec![],
            sections: vec![],
        };
        encode(&[record], &mut out);
        assert_eq!(out, vec![1, 0, 0, FLAG_PLACEMENT, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn dirty_sets_are_clamped_to_buffer_lengths() {
        let (mut scene, id, mut state) = setup();
        // Normals shorter than the vertex slots (as after a failed growth);
        // the first report at a new length says "all".
        let moved = Placement {
            translation: [1., 0., 0.],
            ..Placement::IDENTITY
        };
        scene.set_placement(id, moved).unwrap();
        let mut ch = take(&mut scene, id);
        ch.normals_dirty = Some(vec![]);
        let d = state.apply(scene.mesh(id).unwrap(), &ch, 2).unwrap();
        assert_eq!(d.normals, None);
        ch.normals_dirty = Some(vec![0, 1, 3, 9]);
        ch.colors_dirty = Some(vec![2, 3, 4]);
        let d = state.apply(scene.mesh(id).unwrap(), &ch, 2).unwrap();
        assert_eq!(d.normals, Some(vec![0, 1]));
        assert_eq!(d.colors, Some(vec![2, 3]));
        // Longer normals: still only the reported slots, within the buffer.
        let d = state.apply(scene.mesh(id).unwrap(), &ch, 6).unwrap();
        assert_eq!(d.normals, None);
        let d = state.apply(scene.mesh(id).unwrap(), &ch, 6).unwrap();
        assert_eq!(d.normals, Some(vec![0, 1, 3]));
    }

    #[test]
    fn compaction_reports_all_and_capacity() {
        let (mut scene, id, mut state) = setup();
        scene.edit(id, |tx| tx.remove_triangles(&[0])).unwrap();
        step(&mut scene, id, &mut state);
        scene.compact(id).unwrap();
        let d = step(&mut scene, id, &mut state);
        assert_eq!(d.positions, None);
        assert_eq!(d.indices, None);
        assert!(d.capacity_changed);
        assert_eq!(indices(&scene, id), &[0, 2, 3]);
    }

    /// Parses a message back into records, independently of `encode`.
    fn decode(msg: &[u32]) -> Vec<Record> {
        let mut it = msg.iter().copied();
        let mut next = || it.next().expect("truncated message");
        let n = next();
        let mut records = Vec::new();
        for _ in 0..n {
            let (mesh_index, mesh_generation, flags) = (next(), next(), next());
            let mut lists: Vec<Option<Vec<u32>>> = Vec::new();
            for bit in [
                FLAG_POSITIONS_ALL,
                FLAG_NORMALS_ALL,
                FLAG_COLORS_ALL,
                FLAG_INDICES_ALL,
            ] {
                let ranges = next();
                let mut slots = Vec::new();
                for _ in 0..ranges {
                    let (start, count) = (next(), next());
                    slots.extend(start..start + count);
                }
                if flags & bit != 0 {
                    assert!(slots.is_empty(), "\"all\" with ranges");
                    lists.push(None);
                } else {
                    lists.push(Some(slots));
                }
            }
            let mut ids = || {
                let k = next();
                (0..k).map(|_| next()).collect::<Vec<u32>>()
            };
            let (fields, sections) = (ids(), ids());
            let mut lists = lists.into_iter();
            records.push(Record {
                mesh_index,
                mesh_generation,
                added: flags & FLAG_ADDED != 0,
                removed: flags & FLAG_REMOVED != 0,
                placement_changed: flags & FLAG_PLACEMENT != 0,
                dirty: Dirty {
                    positions: lists.next().unwrap(),
                    normals: lists.next().unwrap(),
                    colors: lists.next().unwrap(),
                    indices: lists.next().unwrap(),
                    capacity_changed: flags & FLAG_CAPACITY != 0,
                },
                fields,
                sections,
            });
        }
        assert!(it.next().is_none(), "trailing data");
        records
    }

    #[test]
    fn encode_round_trip() {
        // Contiguous sets, so that decoding the merged ranges gives them back.
        let records = vec![
            Record {
                mesh_index: 3,
                mesh_generation: 7,
                added: true,
                removed: false,
                placement_changed: true,
                dirty: Dirty::all(),
                fields: vec![11, 12],
                sections: vec![],
            },
            Record {
                mesh_index: 0,
                mesh_generation: 1,
                added: false,
                removed: false,
                placement_changed: false,
                dirty: Dirty {
                    positions: Some(vec![1, 2, 3, 40]),
                    normals: Some(vec![5]),
                    colors: None,
                    indices: Some(vec![0, 1, 100, 101, 102]),
                    capacity_changed: false,
                },
                fields: vec![],
                sections: vec![2, 9, 4],
            },
            Record {
                mesh_index: 1,
                mesh_generation: 0,
                added: false,
                removed: true,
                placement_changed: false,
                dirty: Dirty::clean(),
                fields: vec![],
                sections: vec![],
            },
        ];
        let mut out = vec![99, 99];
        encode(&records, &mut out);
        assert_eq!(decode(&out), records);
        // Gaps of at most 8 slots become one range.
        let mut gappy = records[1].clone();
        gappy.dirty.positions = Some(vec![1, 5]);
        encode(std::slice::from_ref(&gappy), &mut out);
        assert_eq!(decode(&out)[0].dirty.positions, Some(vec![1, 2, 3, 4, 5]));
        // A capacity change makes every buffer "all", whatever the lists say.
        let mut grown = records[1].clone();
        grown.dirty.capacity_changed = true;
        encode(std::slice::from_ref(&grown), &mut out);
        assert_eq!(decode(&out)[0].dirty, Dirty::all());
        encode(&[], &mut out);
        assert_eq!(out, vec![0]);
    }
}
