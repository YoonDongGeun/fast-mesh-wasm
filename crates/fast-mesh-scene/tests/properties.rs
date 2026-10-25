//! Model-based property test: after any sequence of edits, placements, steps,
//! undo, redo, mode changes, flushes, removals and compactions, a `Scene` must
//! equal a plain model, and its world queries must equal Phase 3 queries
//! called directly with hand-computed local transforms.
//!
//! The model is plain data. Per mesh it holds the expected positions and
//! triangles by slot (`None` for a dead slot), the placement, the update mode
//! and whether the BVH should be current. It also holds a step stack that
//! mirrors the Scene's undo and redo: every entry keeps the whole mesh data
//! from before and after the edit, so undoing means putting that data back.
//! Mesh data is compared by alive slots.
//!
//! The history budget is either unlimited or small enough that steps are
//! evicted within a short sequence. To know which steps go, the model must
//! know each entry's size, and the size of a patch depends on which slots the
//! store hands out, which depends on which slots eviction released. So each
//! model mesh also keeps a shadow `Mesh` that gets the same edits through its
//! own transactions; the model keeps the shadow's patches, applies, reverts
//! and releases them as its own history rules say, and so knows each patch's
//! size. The shadow's free slots must equal the scene mesh's after every
//! action, which checks slot release on eviction, redo drops and removal.
//!
//! After most actions `take_changes` is checked against the data the model
//! had at the previous call: added and removed ids exactly, every slot whose
//! content changed among the moved or changed ones, `changes: None` only after
//! a compaction (the test injects no faults, so there is no `Corrupted`), and
//! `placement_changed` exactly when some placement change ran, including one
//! that was undone. The call is skipped on some actions, so that reports
//! folding across several actions are checked too.
//!
//! Each edit does one thing to one mesh, so that "the patch is not empty" and
//! "the model's data changed" are the same thing; steps group several edits.

use fast_mesh_bvh::{BuildOptions, Bvh, Quality};
use fast_mesh_query::{
    ClosestOptions, Cull, Ray, RayOptions, Rigid, closest_point, intersects, mesh_distance, raycast,
};
use fast_mesh_scene::{
    MeshChanges, MeshId, MeshOptions, Placement, Scene, SceneClosestOptions, SceneError,
    SceneOptions, SceneRayOptions, UpdateMode,
};
use fast_mesh_store::{Mesh, Patch, PatchSide, Transaction};
use proptest::prelude::*;

/// Bytes a placement entry counts against the history budget.
const PLACEMENT_BYTES: usize = 2 * std::mem::size_of::<Placement>();

/// The case count: `PROPTEST_CASES` when set, else `default`. `with_cases`
/// alone would ignore the environment variable.
fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// A deterministic linear congruential generator.
struct Lcg(u64);

impl Lcg {
    /// Uniform in `[0, 1)`.
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f64 / (1u64 << 24) as f64
    }

    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next() * (hi - lo)
    }

    fn below(&mut self, n: usize) -> usize {
        ((self.next() * n as f64) as usize).min(n - 1)
    }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum EditOp {
    Move(u32, [i8; 3]),
    AddVertex(u32, u32, u8),
    AddTriangle([u32; 3]),
    RemoveTriangle(u32),
    SetTriangle(u32, [u32; 3]),
    RemoveVertex(u32),
}

#[derive(Debug, Clone)]
enum Action {
    Edit(u8, EditOp),
    SetPlacement(u8, PlacementSeed),
    /// Several placements of one mesh in a row, as a drag makes them: in an
    /// open step they merge into one entry.
    Drag(u8, PlacementSeed, u8),
    /// Sets a mesh's current placement again: recorded and reported as
    /// nothing.
    SetSamePlacement(u8),
    Begin,
    End,
    Cancel,
    Undo,
    Redo,
    SetMode(u8, UpdateMode),
    Update(u8),
    Flush,
    Remove(u8),
    Compact(u8),
    ClearHistory,
    Query(u64),
}

/// Inputs of a random placement: a kind of rotation, a quaternion and a jitter.
type PlacementSeed = (u8, [i8; 4], [i8; 3]);

fn edit_op() -> impl Strategy<Value = EditOp> {
    prop_oneof![
        4 => (any::<u32>(), any::<[i8; 3]>()).prop_map(|(v, d)| EditOp::Move(v, d)),
        1 => (any::<u32>(), any::<u32>(), any::<u8>())
            .prop_map(|(a, b, w)| EditOp::AddVertex(a, b, w)),
        2 => any::<[u32; 3]>().prop_map(EditOp::AddTriangle),
        2 => any::<u32>().prop_map(EditOp::RemoveTriangle),
        1 => (any::<u32>(), any::<[u32; 3]>()).prop_map(|(t, v)| EditOp::SetTriangle(t, v)),
        1 => any::<u32>().prop_map(EditOp::RemoveVertex),
    ]
}

/// The history budget: unlimited, or small enough that a few edits or
/// placement entries (a placement entry counts 192 bytes) fill it, so that
/// short sequences evict steps and open steps outgrow it.
fn budget() -> impl Strategy<Value = usize> {
    prop_oneof![Just(usize::MAX), 100usize..800]
}

fn mode() -> impl Strategy<Value = UpdateMode> {
    prop_oneof![
        Just(UpdateMode::Eager),
        Just(UpdateMode::Lazy),
        Just(UpdateMode::Manual),
    ]
}

fn placement_seed() -> impl Strategy<Value = PlacementSeed> {
    (any::<u8>(), any::<[i8; 4]>(), any::<[i8; 3]>())
}

fn action() -> impl Strategy<Value = Action> {
    prop_oneof![
        8 => (any::<u8>(), edit_op()).prop_map(|(m, op)| Action::Edit(m, op)),
        3 => (any::<u8>(), placement_seed()).prop_map(|(m, p)| Action::SetPlacement(m, p)),
        2 => (any::<u8>(), placement_seed(), 2u8..6).prop_map(|(m, p, n)| Action::Drag(m, p, n)),
        1 => any::<u8>().prop_map(Action::SetSamePlacement),
        2 => Just(Action::Begin),
        2 => Just(Action::End),
        1 => Just(Action::Cancel),
        3 => Just(Action::Undo),
        2 => Just(Action::Redo),
        2 => (any::<u8>(), mode()).prop_map(|(m, mode)| Action::SetMode(m, mode)),
        1 => any::<u8>().prop_map(Action::Update),
        1 => Just(Action::Flush),
        1 => any::<u8>().prop_map(Action::Remove),
        1 => any::<u8>().prop_map(Action::Compact),
        1 => Just(Action::ClearHistory),
        2 => any::<u64>().prop_map(Action::Query),
    ]
}

fn build_options() -> impl Strategy<Value = BuildOptions> {
    (
        prop_oneof![Just(Quality::Sah), Just(Quality::Fast)],
        1u32..=8,
    )
        .prop_map(|(quality, max_leaf_size)| BuildOptions {
            quality,
            max_leaf_size,
            ..Default::default()
        })
}

// ---------------------------------------------------------------------------
// Meshes, placements and the model
// ---------------------------------------------------------------------------

/// A wavy n×n grid (2n² triangles) at `offset`. An offset of zero builds an
/// `f32` mesh; any other builds from `f64` positions, so the mesh gets an
/// origin and keeps small local coordinates.
fn grid(n: u32, offset: [f64; 3]) -> Mesh {
    let mut positions = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            positions.extend([
                f64::from(x) + offset[0],
                f64::from(y) + offset[1],
                f64::from((x * 7 + y * 3) % 5) * 0.3 + offset[2],
            ]);
        }
    }
    let mut indices = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
        }
    }
    if offset == [0.0; 3] {
        let flat: Vec<f32> = positions.iter().map(|&c| c as f32).collect();
        Mesh::from_buffers(&flat, &indices).unwrap()
    } else {
        Mesh::from_buffers_f64(&positions, &indices).unwrap()
    }
}

/// The rotation of a placement kind: the identity, a quarter turn about z, a
/// flip about x (all exact), or a random rotation from the quaternion.
fn rotation(kind: u8, q: [i8; 4]) -> [[f64; 3]; 3] {
    match kind % 4 {
        0 => Placement::IDENTITY.rotation,
        1 => [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        2 => [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        _ => {
            let mut q = q.map(|v| f64::from(v) + 0.5);
            let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
            q = q.map(|v| v / norm);
            let [w, x, y, z] = q;
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ]
        }
    }
}

/// A placement that puts the point `anchor` (the middle of the mesh, in the
/// mesh's own coordinates) near the middle of the world region the queries
/// look at, so that the meshes touch, overlap or sit slightly apart.
fn placement(seed: PlacementSeed, anchor: [f64; 3]) -> Placement {
    let (kind, q, j) = seed;
    let rotation = rotation(kind, q);
    let target = [
        5.0 + f64::from(j[0]) / 32.0,
        5.0 + f64::from(j[1]) / 32.0,
        1.0 + f64::from(j[2]) / 32.0,
    ];
    let translation = std::array::from_fn(|i| {
        target[i] - (0..3).map(|k| rotation[i][k] * anchor[k]).sum::<f64>()
    });
    Placement {
        rotation,
        translation,
    }
}

/// Positions and triangles by slot; `None` is a dead (or missing) slot.
#[derive(Debug, Clone, PartialEq, Default)]
struct Data {
    positions: Vec<Option<[f32; 3]>>,
    triangles: Vec<Option<[u32; 3]>>,
}

fn slot<T: Copy>(v: &[Option<T>], i: usize) -> Option<T> {
    v.get(i).copied().flatten()
}

fn set_slot<T>(v: &mut Vec<Option<T>>, i: usize, value: Option<T>) {
    if v.len() <= i {
        v.resize_with(i + 1, || None);
    }
    v[i] = value;
}

impl Data {
    fn of(mesh: &Mesh) -> Data {
        let mut d = Data::default();
        for v in 0..mesh.vertex_slots() as u32 {
            if mesh.is_vertex_alive(v) {
                set_slot(&mut d.positions, v as usize, mesh.position(v));
            }
        }
        for t in 0..mesh.triangle_slots() as u32 {
            if mesh.is_triangle_alive(t) {
                set_slot(&mut d.triangles, t as usize, mesh.triangle(t));
            }
        }
        d
    }

    /// Compares by alive slots; `Err` describes the first difference.
    fn check(&self, mesh: &Mesh) -> Result<(), String> {
        let (vs, ts) = (mesh.vertex_slots(), mesh.triangle_slots());
        for v in 0..vs.max(self.positions.len()) {
            let got = (v < vs && mesh.is_vertex_alive(v as u32))
                .then(|| mesh.position(v as u32))
                .flatten();
            if got != slot(&self.positions, v) {
                return Err(format!(
                    "vertex slot {v}: mesh {got:?}, model {:?}",
                    slot(&self.positions, v)
                ));
            }
        }
        for t in 0..ts.max(self.triangles.len()) {
            let got = (t < ts && mesh.is_triangle_alive(t as u32))
                .then(|| mesh.triangle(t as u32))
                .flatten();
            if got != slot(&self.triangles, t) {
                return Err(format!(
                    "triangle slot {t}: mesh {got:?}, model {:?}",
                    slot(&self.triangles, t)
                ));
            }
        }
        Ok(())
    }

    fn vertices(&self) -> Vec<u32> {
        alive_slots(&self.positions)
    }

    fn triangle_ids(&self) -> Vec<u32> {
        alive_slots(&self.triangles)
    }
}

fn alive_slots<T>(v: &[Option<T>]) -> Vec<u32> {
    (0..v.len() as u32)
        .filter(|&i| v[i as usize].is_some())
        .collect()
}

/// What one edit did to the data.
#[derive(Debug, PartialEq)]
enum Delta {
    Move(u32, [f32; 3]),
    AddVertex(u32, [f32; 3]),
    AddTriangle(u32, [u32; 3]),
    RemoveTriangle(u32),
    SetTriangle(u32, [u32; 3]),
    RemoveVertex(u32),
}

impl Delta {
    fn apply(&self, d: &mut Data) {
        match *self {
            Delta::Move(v, p) | Delta::AddVertex(v, p) => {
                set_slot(&mut d.positions, v as usize, Some(p));
            }
            Delta::AddTriangle(t, tri) | Delta::SetTriangle(t, tri) => {
                set_slot(&mut d.triangles, t as usize, Some(tri));
            }
            Delta::RemoveTriangle(t) => set_slot(&mut d.triangles, t as usize, None),
            Delta::RemoveVertex(v) => set_slot(&mut d.positions, v as usize, None),
        }
    }
}

/// How the data at the last `take_changes` (`snap`) differs from `now`, by
/// slot: ids that became alive, ids that died, and ids alive in both whose
/// content differs. All sorted.
#[derive(Debug, Default, PartialEq)]
struct Diff {
    vertices_added: Vec<u32>,
    vertices_removed: Vec<u32>,
    vertices_moved: Vec<u32>,
    triangles_added: Vec<u32>,
    triangles_removed: Vec<u32>,
    triangles_changed: Vec<u32>,
}

/// Per slot of two by-slot lists: (became alive, died, alive in both with
/// different content).
fn diff_slots<T: Copy + PartialEq>(
    snap: &[Option<T>],
    now: &[Option<T>],
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (mut added, mut removed, mut changed) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..snap.len().max(now.len()) {
        match (slot(snap, i), slot(now, i)) {
            (None, Some(_)) => added.push(i as u32),
            (Some(_), None) => removed.push(i as u32),
            (Some(a), Some(b)) if a != b => changed.push(i as u32),
            _ => {}
        }
    }
    (added, removed, changed)
}

impl Diff {
    fn of(snap: &Data, now: &Data) -> Diff {
        let (vertices_added, vertices_removed, vertices_moved) =
            diff_slots(&snap.positions, &now.positions);
        let (triangles_added, triangles_removed, triangles_changed) =
            diff_slots(&snap.triangles, &now.triangles);
        Diff {
            vertices_added,
            vertices_removed,
            vertices_moved,
            triangles_added,
            triangles_removed,
            triangles_changed,
        }
    }

    fn is_empty(&self) -> bool {
        *self == Diff::default()
    }
}

fn pick(ids: &[u32], seed: u32) -> Option<u32> {
    (!ids.is_empty()).then(|| ids[seed as usize % ids.len()])
}

/// Runs one edit op in the transaction, choosing its targets among the alive
/// slots of the model. `None` when there is nothing to do (the transaction
/// stays empty).
fn run(
    tx: &mut Transaction<'_>,
    data: &Data,
    op: &EditOp,
) -> fast_mesh_store::Result<Option<Delta>> {
    let vs = data.vertices();
    let ts = data.triangle_ids();
    Ok(match *op {
        EditOp::Move(s, d) => {
            let Some(v) = pick(&vs, s) else {
                return Ok(None);
            };
            // A zero move would change nothing and record nothing.
            let d = if d == [0; 3] { [1, 0, 0] } else { d };
            let p = slot(&data.positions, v as usize).unwrap();
            let q = [0, 1, 2].map(|k| p[k] + f32::from(d[k]) * 0.05);
            tx.set_positions(&[v], &[q])?;
            Some(Delta::Move(v, q))
        }
        EditOp::AddVertex(a, b, w) => {
            let (Some(a), Some(b)) = (pick(&vs, a), pick(&vs, b)) else {
                return Ok(None);
            };
            let pa = slot(&data.positions, a as usize).unwrap();
            let pb = slot(&data.positions, b as usize).unwrap();
            let t = f32::from(w) / 255.0;
            let p = [0, 1, 2].map(|k| pa[k] * (1.0 - t) + pb[k] * t);
            let v = tx.add_vertex_with_parents(p, &[(a, 1.0 - t), (b, t)])?;
            Some(Delta::AddVertex(v, p))
        }
        EditOp::AddTriangle(s) => {
            let (Some(a), Some(b), Some(c)) = (pick(&vs, s[0]), pick(&vs, s[1]), pick(&vs, s[2]))
            else {
                return Ok(None);
            };
            let t = tx.add_triangles(&[[a, b, c]])?[0];
            Some(Delta::AddTriangle(t, [a, b, c]))
        }
        EditOp::RemoveTriangle(s) => {
            let Some(t) = pick(&ts, s) else {
                return Ok(None);
            };
            tx.remove_triangles(&[t])?;
            Some(Delta::RemoveTriangle(t))
        }
        EditOp::SetTriangle(t, s) => {
            let (Some(t), Some(a), Some(b), Some(c)) = (
                pick(&ts, t),
                pick(&vs, s[0]),
                pick(&vs, s[1]),
                pick(&vs, s[2]),
            ) else {
                return Ok(None);
            };
            // Setting the same corners would change nothing and record nothing.
            if slot(&data.triangles, t as usize) == Some([a, b, c]) {
                return Ok(None);
            }
            tx.set_triangles(&[t], &[[a, b, c]])?;
            Some(Delta::SetTriangle(t, [a, b, c]))
        }
        EditOp::RemoveVertex(s) => {
            // Only a vertex no alive triangle uses can go.
            let used: std::collections::HashSet<u32> =
                data.triangles.iter().flatten().flatten().copied().collect();
            let unused: Vec<u32> = vs.into_iter().filter(|v| !used.contains(v)).collect();
            let Some(v) = pick(&unused, s) else {
                return Ok(None);
            };
            tx.remove_vertices(&[v])?;
            Some(Delta::RemoveVertex(v))
        }
    })
}

/// One entry of a model step.
enum Entry {
    Edit {
        mesh: usize,
        before: (Data, u64),
        after: (Data, u64),
        /// The same edit's patch on the mesh's shadow.
        patch: Box<Patch>,
    },
    Place {
        mesh: usize,
        old: Placement,
        new: Placement,
    },
}

impl Entry {
    fn mesh(&self) -> usize {
        match self {
            Entry::Edit { mesh, .. } | Entry::Place { mesh, .. } => *mesh,
        }
    }

    /// What the entry counts against the history budget.
    fn bytes(&self) -> usize {
        match self {
            Entry::Edit { patch, .. } => patch.byte_size(),
            Entry::Place { .. } => PLACEMENT_BYTES,
        }
    }
}

type Step = Vec<Entry>;

struct ModelMesh {
    id: MeshId,
    live: bool,
    data: Data,
    /// A mesh that gets the same edits as the scene's, through its own
    /// transactions; its patches live in the model's steps.
    shadow: Mesh,
    placement: Placement,
    mode: UpdateMode,
    /// The data and placement at the last `take_changes`.
    snap: Data,
    snap_placement: Placement,
    /// A placement change ran since the last `take_changes`.
    touched: bool,
    /// Compacted since the last `take_changes`: reported as `None`.
    everything: bool,
    /// Removed, and the removal already reported.
    gone: bool,
    /// The mesh's revision, as the store numbers them: an edit gets a new
    /// one, and undo and redo bring back the revision the patch started or
    /// ended at.
    rev: u64,
    /// The revision the BVH was last brought up to.
    bvh_rev: u64,
    /// The middle of the mesh in its own coordinates (origin included).
    anchor: [f64; 3],
}

impl ModelMesh {
    /// Whether the BVH matches the mesh.
    fn current(&self) -> bool {
        self.rev == self.bvh_rev
    }

    /// The BVH is brought up to date.
    fn refresh(&mut self) {
        self.bvh_rev = self.rev;
    }
}

/// The expected history: undo steps (oldest first), redo steps (next to redo
/// last) and the open step, with the Scene's rules for steps, placement runs
/// and the budget written out plainly.
struct Model {
    /// The last revision handed out.
    last_rev: u64,
    meshes: Vec<ModelMesh>,
    undo: Vec<Step>,
    redo: Vec<Step>,
    open: Step,
    depth: u32,
    budget: usize,
}

impl Model {
    fn live(&self) -> Vec<usize> {
        (0..self.meshes.len())
            .filter(|&i| self.meshes[i].live)
            .collect()
    }

    fn choose(&self, sel: u8) -> Option<usize> {
        let live = self.live();
        (!live.is_empty()).then(|| live[sel as usize % live.len()])
    }

    /// Bytes of every step held.
    fn bytes(&self) -> usize {
        self.undo
            .iter()
            .chain(&self.redo)
            .chain(std::iter::once(&self.open))
            .flatten()
            .map(Entry::bytes)
            .sum()
    }

    /// A discarded step gives back the slots its patches reserve.
    fn release(&mut self, step: &[Entry], side: PatchSide) {
        for e in step {
            if let Entry::Edit { mesh, patch, .. } = e
                && self.meshes[*mesh].live
            {
                patch.release(&mut self.meshes[*mesh].shadow, side);
            }
        }
    }

    fn drop_redo(&mut self) {
        for step in std::mem::take(&mut self.redo) {
            self.release(&step, PatchSide::Redoable);
        }
    }

    /// Drops the oldest undo steps until the budget is met.
    fn evict(&mut self) {
        while self.bytes() > self.budget && !self.undo.is_empty() {
            let step = self.undo.remove(0);
            self.release(&step, PatchSide::Undoable);
        }
    }

    /// Drops the oldest redo steps until the budget is met.
    fn evict_redo(&mut self) {
        while self.bytes() > self.budget && !self.redo.is_empty() {
            let step = self.redo.remove(0);
            self.release(&step, PatchSide::Redoable);
        }
    }

    /// Records an entry: a placement entry right after one of the same mesh
    /// in the open step merges into it; anything else drops the redo stack,
    /// then closes the step unless one is open, or else meets the budget.
    fn record(&mut self, e: Entry) {
        if self.depth > 0
            && let Entry::Place { mesh, new, .. } = &e
            && let Some(Entry::Place {
                mesh: last,
                new: last_new,
                ..
            }) = self.open.last_mut()
            && last == mesh
        {
            *last_new = *new;
            return;
        }
        self.drop_redo();
        self.open.push(e);
        if self.depth == 0 {
            self.close();
        } else {
            self.evict();
        }
    }

    /// Closes the open step, whatever its depth; an empty one is dropped.
    fn close(&mut self) {
        self.depth = 0;
        if !self.open.is_empty() {
            let step = std::mem::take(&mut self.open);
            self.undo.push(step);
            self.evict();
        }
    }

    /// Undoes the open step if it has entries, else the latest closed one.
    /// An undone open step meets the budget only afterwards, on the redo
    /// stack. False when there is nothing to undo.
    fn undo(&mut self) -> bool {
        if !self.open.is_empty() {
            let step = std::mem::take(&mut self.open);
            self.depth = 0;
            self.run_step(&step, true);
            self.redo.push(step);
            self.evict();
            self.evict_redo();
            return true;
        }
        self.depth = 0;
        let Some(step) = self.undo.pop() else {
            return false;
        };
        self.run_step(&step, true);
        self.redo.push(step);
        true
    }

    /// Redoes the latest undone step, closing the open step first.
    fn redo(&mut self) -> bool {
        let Some(step) = self.redo.pop() else {
            self.close();
            return false;
        };
        self.depth = 0;
        self.run_step(&step, false);
        self.undo.push(step);
        true
    }

    /// Reverts and discards the open step.
    fn cancel(&mut self) {
        let step = std::mem::take(&mut self.open);
        self.depth = 0;
        self.run_step(&step, true);
        self.release(&step, PatchSide::Redoable);
    }

    /// Closes the open step, then drops every step.
    fn clear(&mut self) {
        self.close();
        for step in std::mem::take(&mut self.undo) {
            self.release(&step, PatchSide::Undoable);
        }
        self.drop_redo();
    }

    /// A revision that no mesh has had.
    fn fresh_rev(&mut self) -> u64 {
        self.last_rev += 1;
        self.last_rev
    }

    /// Puts a mesh's data (and revision) back or forward; an Eager BVH
    /// follows right away, the others are current only if the revision is
    /// the one they were last updated to.
    fn set_data(&mut self, mesh: usize, (data, rev): &(Data, u64)) {
        let m = &mut self.meshes[mesh];
        m.data = data.clone();
        m.rev = *rev;
        if m.mode == UpdateMode::Eager {
            m.refresh();
        }
    }

    /// Reverts (`undo`) or reapplies a step's entries, in reverse or in order.
    fn run_step(&mut self, step: &[Entry], undo: bool) {
        let order: Vec<&Entry> = if undo {
            step.iter().rev().collect()
        } else {
            step.iter().collect()
        };
        for e in order {
            match e {
                Entry::Edit {
                    mesh,
                    before,
                    after,
                    patch,
                } => {
                    let shadow = &mut self.meshes[*mesh].shadow;
                    if undo {
                        patch.revert(shadow).unwrap();
                    } else {
                        patch.apply(shadow).unwrap();
                    }
                    self.set_data(*mesh, if undo { before } else { after });
                }
                Entry::Place { mesh, old, new } => {
                    let m = &mut self.meshes[*mesh];
                    m.placement = if undo { *old } else { *new };
                    m.touched = true;
                }
            }
        }
    }

    /// Drops the entries of `mesh` (all of them, or only the edits) from every
    /// step; steps left empty go, except the open one. Dropping all of them
    /// (removal) releases what their patches reserve on the shadow, in the
    /// Scene's order: undo steps, redo steps, then the open step.
    fn forget(&mut self, mesh: usize, edits_only: bool) {
        let mut undo = std::mem::take(&mut self.undo);
        let mut redo = std::mem::take(&mut self.redo);
        let mut open = std::mem::take(&mut self.open);
        let shadow = &mut self.meshes[mesh].shadow;
        let mut drop_from = |step: &mut Step, side: PatchSide| {
            step.retain(|e| {
                if e.mesh() != mesh {
                    return true;
                }
                if edits_only {
                    return matches!(e, Entry::Place { .. });
                }
                if let Entry::Edit { patch, .. } = e {
                    patch.release(shadow, side);
                }
                false
            });
        };
        for step in &mut undo {
            drop_from(step, PatchSide::Undoable);
        }
        for step in &mut redo {
            drop_from(step, PatchSide::Redoable);
        }
        drop_from(&mut open, PatchSide::Undoable);
        undo.retain(|s| !s.is_empty());
        redo.retain(|s| !s.is_empty());
        (self.undo, self.redo, self.open) = (undo, redo, open);
    }
}

// ---------------------------------------------------------------------------
// Direct Phase 3 oracles, with hand-computed local transforms
// ---------------------------------------------------------------------------

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// `Rᵀ (q - t) - o`: a world point in the mesh's local space.
fn to_local(p: &Placement, origin: [f64; 3], q: [f64; 3]) -> [f64; 3] {
    let d = [
        q[0] - p.translation[0],
        q[1] - p.translation[1],
        q[2] - p.translation[2],
    ];
    let r = &p.rotation;
    std::array::from_fn(|i| r[0][i] * d[0] + r[1][i] * d[1] + r[2][i] * d[2] - origin[i])
}

/// `Rᵀ d`.
fn rotate_back(p: &Placement, d: [f64; 3]) -> [f64; 3] {
    let r = &p.rotation;
    std::array::from_fn(|i| r[0][i] * d[0] + r[1][i] * d[1] + r[2][i] * d[2])
}

/// `R (o + l) + t`: a mesh-local point in the world.
fn to_world(p: &Placement, origin: [f64; 3], l: [f32; 3]) -> [f64; 3] {
    let r = &p.rotation;
    let w = [0, 1, 2].map(|k| origin[k] + f64::from(l[k]));
    std::array::from_fn(|i| r[i][0] * w[0] + r[i][1] * w[1] + r[i][2] * w[2] + p.translation[i])
}

/// The transform from B's local space to A's, with `R = R_Aᵀ R_B` brought back
/// to an exact rotation by Gram-Schmidt on its rows.
fn relative(a: (&Placement, [f64; 3]), b: (&Placement, [f64; 3])) -> Rigid {
    let ((pa, oa), (pb, ob)) = (a, b);
    let (ra, rb) = (&pa.rotation, &pb.rotation);
    let raw: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| ra[0][i] * rb[0][j] + ra[1][i] * rb[1][j] + ra[2][i] * rb[2][j])
    });
    let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let unit = |x: [f64; 3]| {
        let n = dot(x, x).sqrt();
        [x[0] / n, x[1] / n, x[2] / n]
    };
    let r0 = unit(raw[0]);
    let d = dot(raw[1], r0);
    let r1 = unit([
        raw[1][0] - d * r0[0],
        raw[1][1] - d * r0[1],
        raw[1][2] - d * r0[2],
    ]);
    let r2 = [
        r0[1] * r1[2] - r0[2] * r1[1],
        r0[2] * r1[0] - r0[0] * r1[2],
        r0[0] * r1[1] - r0[1] * r1[0],
    ];
    // B's origin in the world, relative to A's translation, then into A's frame.
    let wb: [f64; 3] = std::array::from_fn(|i| {
        let r = &pb.rotation;
        r[i][0] * ob[0] + r[i][1] * ob[1] + r[i][2] * ob[2] + pb.translation[i]
    });
    let w = [
        wb[0] - pa.translation[0],
        wb[1] - pa.translation[1],
        wb[2] - pa.translation[2],
    ];
    let translation =
        std::array::from_fn(|i| ra[0][i] * w[0] + ra[1][i] * w[1] + ra[2][i] * w[2] - oa[i]);
    Rigid {
        rotation: [r0, r1, r2],
        translation,
    }
}

// ---------------------------------------------------------------------------
// Brute-force world-space oracle, independent of the relative transform
// ---------------------------------------------------------------------------

/// How far the brute-force world distance may be from the scene's answer:
/// `f32` rounding and the re-orthonormalised relative rotation move it by
/// about 1e-5 at most, while a wrong transform misses by whole mesh sizes.
const BRUTE_TOL: f64 = 1e-3;

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `R (o + p) + t`: the spec's placement formula, written out plainly. This
/// is the only transform the brute-force check uses: no relative transform,
/// no Gram-Schmidt.
fn world_point(pl: &Placement, origin: [f64; 3], p: [f32; 3]) -> [f64; 3] {
    let q = [
        origin[0] + f64::from(p[0]),
        origin[1] + f64::from(p[1]),
        origin[2] + f64::from(p[2]),
    ];
    std::array::from_fn(|i| {
        pl.translation[i]
            + pl.rotation[i]
                .iter()
                .zip(&q)
                .map(|(r, q)| r * q)
                .sum::<f64>()
    })
}

/// Distance from `p` to the segment `ab` (either end when `a == b`).
fn point_segment(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let denom = dot(ab, ab);
    if denom == 0.0 {
        return norm(sub(p, a));
    }
    let t = (dot(ab, sub(p, a)) / denom).clamp(0.0, 1.0);
    norm(sub(
        p,
        [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]],
    ))
}

/// Distance from `p` to the triangle `abc`: the plane distance when the
/// projection falls inside, else the nearest edge. Degenerate triangles
/// fall through to their edges.
fn point_triangle(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let n = cross(ab, ac);
    let edge = || {
        point_segment(p, a, b)
            .min(point_segment(p, b, c))
            .min(point_segment(p, c, a))
    };
    if dot(n, n) == 0.0 {
        return edge();
    }
    // Barycentric coordinates of the projection onto the plane.
    let ap = sub(p, a);
    let (d00, d01, d11) = (dot(ab, ab), dot(ab, ac), dot(ac, ac));
    let (d20, d21) = (dot(ap, ab), dot(ap, ac));
    let denom = d00 * d11 - d01 * d01;
    if denom == 0.0 {
        return edge();
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    if v >= 0.0 && w >= 0.0 && v + w <= 1.0 {
        dot(ap, n).abs() / norm(n)
    } else {
        edge()
    }
}

/// Distance between the segments `p1q1` and `p2q2` (Real-Time Collision
/// Detection 5.1.9, with exact-zero guards instead of an epsilon).
fn seg_seg(p1: [f64; 3], q1: [f64; 3], p2: [f64; 3], q2: [f64; 3]) -> f64 {
    let d1 = sub(q1, p1);
    let d2 = sub(q2, p2);
    let r = sub(p1, p2);
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);
    if a == 0.0 && e == 0.0 {
        return norm(r);
    }
    let (mut s, mut t);
    if a == 0.0 {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if e == 0.0 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            s = if denom == 0.0 {
                0.0
            } else {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            };
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }
    let c1 = [p1[0] + d1[0] * s, p1[1] + d1[1] * s, p1[2] + d1[2] * s];
    let c2 = [p2[0] + d2[0] * t, p2[1] + d2[1] * t, p2[2] + d2[2] * t];
    norm(sub(c1, c2))
}

/// Whether the segment `pq` meets the triangle `abc` (Möller–Trumbore with
/// a small slop, so near-touching counts as touching).
fn seg_tri_hit(p: [f64; 3], q: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> bool {
    const SLOP: f64 = 1e-9;
    let d = sub(q, p);
    if d == [0.0; 3] {
        return false;
    }
    let ab = sub(b, a);
    let ac = sub(c, a);
    let h = cross(d, ac);
    let det = dot(ab, h);
    if det == 0.0 {
        return false;
    }
    let inv = 1.0 / det;
    let s = sub(p, a);
    let u = dot(s, h) * inv;
    if u < -SLOP || u > 1.0 + SLOP {
        return false;
    }
    let qv = cross(s, ab);
    let v = dot(d, qv) * inv;
    if v < -SLOP || u + v > 1.0 + SLOP {
        return false;
    }
    let t = dot(ac, qv) * inv;
    (-SLOP..=1.0 + SLOP).contains(&t)
}

/// Brute-force distance between two meshes in world space, from the model's
/// plain data, or `None` when either mesh has no alive triangle: every
/// vertex against every triangle of the other mesh, every edge against every
/// edge, and 0 when an edge pierces a triangle.
fn brute_distance(
    pa: &Placement,
    oa: [f64; 3],
    da: &Data,
    pb: &Placement,
    ob: [f64; 3],
    db: &Data,
) -> Option<f64> {
    let world = |pl: &Placement, o: [f64; 3], d: &Data| {
        d.positions
            .iter()
            .map(|p| p.map(|p| world_point(pl, o, p)))
            .collect::<Vec<_>>()
    };
    let (wa, wb) = (world(pa, oa, da), world(pb, ob, db));
    let corners = |w: &[Option<[f64; 3]>], t: &[u32; 3]| {
        [
            w[t[0] as usize].unwrap(),
            w[t[1] as usize].unwrap(),
            w[t[2] as usize].unwrap(),
        ]
    };
    let edges = |c: &[[f64; 3]; 3]| [(c[0], c[1]), (c[1], c[2]), (c[2], c[0])];
    let (mut any, mut best) = (false, f64::INFINITY);
    for xa in da.triangles.iter().flatten() {
        for xb in db.triangles.iter().flatten() {
            any = true;
            let (cx, cy) = (corners(&wa, xa), corners(&wb, xb));
            for &(p, q) in &edges(&cx) {
                if seg_tri_hit(p, q, cy[0], cy[1], cy[2]) {
                    return Some(0.0);
                }
            }
            for &(p, q) in &edges(&cy) {
                if seg_tri_hit(p, q, cx[0], cx[1], cx[2]) {
                    return Some(0.0);
                }
            }
            for &v in &cx {
                best = best.min(point_triangle(v, cy[0], cy[1], cy[2]));
            }
            for &v in &cy {
                best = best.min(point_triangle(v, cx[0], cx[1], cx[2]));
            }
            for &(p, q) in &edges(&cx) {
                for &(r, s) in &edges(&cy) {
                    best = best.min(seg_seg(p, q, r, s));
                }
            }
        }
    }
    any.then_some(best)
}

/// The unbounded `mesh_distance` answer against [`brute_distance`]:
/// `want_hit` is the exact Phase 3 intersection answer, and `got` the
/// scene's distance with its world points.
fn check_brute(
    a: &ModelMesh,
    oa: [f64; 3],
    b: &ModelMesh,
    ob: [f64; 3],
    want_hit: bool,
    got: Option<(f64, [f64; 3], [f64; 3])>,
) -> Result<(), TestCaseError> {
    let brute = brute_distance(&a.placement, oa, &a.data, &b.placement, ob, &b.data);
    match (got, brute) {
        (None, None) => Ok(()),
        (Some((d, pta, ptb)), Some(x)) => {
            if want_hit {
                prop_assert!(x <= BRUTE_TOL, "intersecting pair, brute {x}");
            } else {
                prop_assert!((d - x).abs() <= BRUTE_TOL, "mesh_distance {d} vs brute {x}");
            }
            let pair = norm(sub(pta, ptb));
            prop_assert!(
                (pair - x).abs() <= BRUTE_TOL,
                "reported points {pair} apart vs brute {x}"
            );
            Ok(())
        }
        _ => Err(TestCaseError::fail(format!(
            "mesh_distance {got:?} vs brute {brute:?}"
        ))),
    }
}

/// A live mesh of the scene with a BVH of its own, built fresh. Phase 3
/// queries are exact, so they answer the same on any valid BVH.
struct Fresh {
    id: MeshId,
    bvh: Bvh,
}

fn fresh(scene: &Scene) -> Vec<Fresh> {
    scene
        .ids()
        .map(|id| Fresh {
            id,
            bvh: Bvh::build(scene.mesh(id).unwrap(), &BuildOptions::default()).unwrap(),
        })
        .collect()
}

/// A triangle of `mesh` as a mesh of its own, with its BVH.
fn single(mesh: &Mesh, t: u32) -> (Mesh, Bvh) {
    let flat: Vec<f32> = mesh
        .triangle(t)
        .unwrap()
        .iter()
        .flat_map(|&v| mesh.position(v).unwrap())
        .collect();
    let m = Mesh::from_buffers(&flat, &[0, 1, 2]).unwrap();
    let b = Bvh::build(&m, &BuildOptions::default()).unwrap();
    (m, b)
}

/// The world queries, over the meshes the model says are live.
fn check_queries(scene: &mut Scene, model: &mut Model, seed: u64) -> Result<(), TestCaseError> {
    let mut rng = Lcg(seed);
    let live = model.live();
    let ids: Vec<MeshId> = live.iter().map(|&i| model.meshes[i].id).collect();
    prop_assert_eq!(scene.ids().collect::<Vec<_>>(), ids.clone());
    let stale =
        |m: &Model, i: usize| m.meshes[i].mode == UpdateMode::Manual && !m.meshes[i].current();
    let any_stale = live.iter().any(|&i| stale(model, i));
    let fresh = fresh(scene);
    let placements: Vec<Placement> = ids.iter().map(|&id| scene.placement(id).unwrap()).collect();
    let origin_of = |s: &Scene, id| s.mesh(id).unwrap().origin();

    // A world ray, sometimes aimed at a vertex so that it grazes edges.
    let origin = [
        rng.range(-4.0, 12.0),
        rng.range(-4.0, 12.0),
        rng.range(-4.0, 6.0),
    ];
    let mut dir = [
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
    ];
    if !live.is_empty() && rng.next() < 0.6 {
        let k = rng.below(live.len());
        let m = &model.meshes[live[k]];
        if let Some(v) = pick(&m.data.vertices(), (rng.next() * 1e6) as u32) {
            let p = slot(&m.data.positions, v as usize).unwrap();
            let w = to_world(&placements[k], origin_of(scene, ids[k]), p);
            dir = [w[0] - origin[0], w[1] - origin[1], w[2] - origin[2]];
        }
    }
    if dir == [0.0; 3] {
        dir = [0.0, 0.0, -1.0];
    }
    let n = norm(dir);
    let unit = dir.map(|c| c / n);

    for (min_t, max_t) in [(0.0, f64::INFINITY), (0.5, 6.0)] {
        let mut want: Option<(MeshId, u32, f64, [f64; 3])> = None;
        for (k, f) in fresh.iter().enumerate() {
            let mesh = scene.mesh(f.id).unwrap();
            let o = mesh.origin();
            let lo = to_local(&placements[k], o, origin);
            let dl = rotate_back(&placements[k], unit);
            let ray = Ray {
                origin: lo.map(|c| c as f32),
                dir: dl.map(|c| c as f32),
            };
            let opts = RayOptions {
                min_t: min_t as f32,
                max_t: max_t as f32,
                cull: Cull::None,
            };
            if let Some(h) = raycast(mesh, &f.bvh, &ray, &opts).unwrap() {
                let t = f64::from(h.t);
                if want.is_none_or(|w| t < w.2) {
                    want = Some((f.id, h.triangle, t, to_world(&placements[k], o, h.point)));
                }
            }
        }
        let opts = SceneRayOptions {
            min_t,
            max_t,
            cull: Cull::None,
        };
        let got = scene.raycast(origin, dir, &opts);
        if any_stale {
            prop_assert!(matches!(got, Err(SceneError::StaleBvh(_))), "{got:?}");
        } else {
            let got = got.unwrap().map(|h| (h.mesh, h.hit.triangle, h.t, h.point));
            prop_assert_eq!(got, want, "ray {:?} {:?}", origin, dir);
        }
    }

    // A world point: nearest over the scene, and on each mesh alone.
    let p = [
        rng.range(-4.0, 12.0),
        rng.range(-4.0, 12.0),
        rng.range(-4.0, 6.0),
    ];
    for bound in [f64::INFINITY, 2.0] {
        let mut want_all: Option<(MeshId, u32, f32, [f64; 3])> = None;
        let mut want_on = Vec::new();
        for (k, f) in fresh.iter().enumerate() {
            let mesh = scene.mesh(f.id).unwrap();
            let o = mesh.origin();
            let lp = to_local(&placements[k], o, p).map(|c| c as f32);
            let opts = ClosestOptions {
                max_distance: bound as f32,
            };
            let hit = closest_point(mesh, &f.bvh, lp, &opts).unwrap();
            let hit = hit.map(|h| {
                (
                    f.id,
                    h.triangle,
                    h.distance,
                    to_world(&placements[k], o, h.point),
                )
            });
            if let Some(h) = hit
                && want_all.is_none_or(|w| h.2 < w.2)
            {
                want_all = Some(h);
            }
            want_on.push(hit);
        }
        let opts = SceneClosestOptions {
            max_distance: bound,
        };
        let got = scene.closest_point(p, &opts);
        if any_stale {
            prop_assert!(matches!(got, Err(SceneError::StaleBvh(_))), "{got:?}");
        } else {
            let got = got
                .unwrap()
                .map(|h| (h.mesh, h.hit.triangle, h.hit.distance, h.point));
            prop_assert_eq!(got, want_all, "point {:?} bound {}", p, bound);
        }
        // Each mesh alone: only its own BVH has to be current.
        for (k, &id) in ids.iter().enumerate() {
            let got = scene.closest_point_on(id, p, &opts);
            if stale(model, live[k]) {
                prop_assert!(matches!(got, Err(SceneError::StaleBvh(_))), "{got:?}");
            } else {
                let got = got
                    .unwrap()
                    .map(|h| (h.mesh, h.hit.triangle, h.hit.distance, h.point));
                prop_assert_eq!(got, want_on[k], "closest_point_on {:?}", id);
            }
        }
    }
    // Every Lazy BVH was updated by the queries over all meshes (the Manual
    // ones, which may have failed them, were left alone).
    for &i in &live {
        if model.meshes[i].mode != UpdateMode::Manual {
            model.meshes[i].refresh();
        }
    }

    // A pair of meshes.
    if live.len() >= 2 {
        let a = rng.below(live.len());
        let mut b = rng.below(live.len() - 1);
        if b >= a {
            b += 1;
        }
        let (ia, ib) = (live[a], live[b]);
        let (ida, idb) = (ids[a], ids[b]);
        let pair_stale = stale(model, ia) || stale(model, ib);
        // The scene's answers first (they take `&mut`), then the oracles.
        let got_hit = scene.intersects(ida, idb);
        let bounds = [f64::INFINITY, 1.5];
        let got_dist = bounds.map(|bound| scene.mesh_distance(ida, idb, bound));
        let (ma, mb) = (scene.mesh(ida).unwrap(), scene.mesh(idb).unwrap());
        let (oa, ob) = (ma.origin(), mb.origin());
        let xf = relative((&placements[a], oa), (&placements[b], ob));
        let (ba, bb) = (&fresh[a].bvh, &fresh[b].bvh);

        let want_hit = intersects(ma, ba, mb, bb, &xf).unwrap();
        if pair_stale {
            prop_assert!(
                matches!(got_hit, Err(SceneError::StaleBvh(_))),
                "{got_hit:?}"
            );
        } else {
            prop_assert_eq!(got_hit.unwrap(), want_hit);
        }
        for (bound, got) in bounds.into_iter().zip(got_dist) {
            let want = mesh_distance(ma, ba, mb, bb, &xf, bound as f32).unwrap();
            if pair_stale {
                prop_assert!(matches!(got, Err(SceneError::StaleBvh(_))), "{got:?}");
                continue;
            }
            let got = got.unwrap();
            prop_assert_eq!(got.map(|g| g.distance), want.map(|w| f64::from(w.distance)));
            if bound.is_infinite() {
                check_brute(
                    &model.meshes[ia],
                    oa,
                    &model.meshes[ib],
                    ob,
                    want_hit,
                    got.as_ref().map(|g| (g.distance, g.point_a, g.point_b)),
                )?;
            }
            if let (Some(g), Some(w)) = (got, want) {
                // Ties between triangle pairs may go either way on different
                // trees; the pair named must be alive and its own distance.
                prop_assert!(ma.is_triangle_alive(g.triangle_a));
                prop_assert!(mb.is_triangle_alive(g.triangle_b));
                if w.distance > 0.0 {
                    let (sa, sba) = single(ma, g.triangle_a);
                    let (sb, sbb) = single(mb, g.triangle_b);
                    let one = mesh_distance(&sa, &sba, &sb, &sbb, &xf, f32::INFINITY)
                        .unwrap()
                        .unwrap();
                    prop_assert_eq!(f64::from(one.distance), g.distance);
                    prop_assert_eq!(g.point_a, to_world(&placements[a], oa, one.point_a));
                    prop_assert_eq!(g.point_b, to_world(&placements[a], oa, one.point_b));
                }
            }
        }
        for i in [ia, ib] {
            if model.meshes[i].mode != UpdateMode::Manual {
                model.meshes[i].refresh();
            }
        }
    }
    Ok(())
}

/// After every action: the Eager meshes' BVHs answer like fresh builds.
/// Only queries that touch Eager meshes alone are used (`closest_point_on`,
/// and `mesh_distance` between two Eager meshes), so no Lazy BVH is updated
/// and the staleness the model tracks is left as it is.
fn check_eager(scene: &mut Scene, model: &Model, seed: u64) -> Result<(), TestCaseError> {
    let mut rng = Lcg(seed);
    let eager: Vec<usize> = model
        .live()
        .into_iter()
        .filter(|&i| model.meshes[i].mode == UpdateMode::Eager)
        .collect();
    let p = [
        rng.range(-4.0, 12.0),
        rng.range(-4.0, 12.0),
        rng.range(-4.0, 6.0),
    ];
    let mut built = Vec::new();
    for &i in &eager {
        let id = model.meshes[i].id;
        let mesh = scene.mesh(id).unwrap();
        let bvh = Bvh::build(mesh, &BuildOptions::default()).unwrap();
        let pl = model.meshes[i].placement;
        let o = mesh.origin();
        let lp = to_local(&pl, o, p).map(|c| c as f32);
        let want = closest_point(mesh, &bvh, lp, &ClosestOptions::default())
            .unwrap()
            .map(|h| (h.triangle, h.distance, to_world(&pl, o, h.point)));
        built.push((i, bvh));
        let got = scene
            .closest_point_on(id, p, &SceneClosestOptions::default())
            .unwrap()
            .map(|h| (h.hit.triangle, h.hit.distance, h.point));
        prop_assert_eq!(got, want, "closest_point_on mesh {}", i);
    }
    for x in 0..built.len() {
        for y in x + 1..built.len() {
            let (ia, ib) = (built[x].0, built[y].0);
            let (ida, idb) = (model.meshes[ia].id, model.meshes[ib].id);
            let (ma, mb) = (scene.mesh(ida).unwrap(), scene.mesh(idb).unwrap());
            let xf = relative(
                (&model.meshes[ia].placement, ma.origin()),
                (&model.meshes[ib].placement, mb.origin()),
            );
            let want = mesh_distance(ma, &built[x].1, mb, &built[y].1, &xf, f32::INFINITY).unwrap();
            let got = scene.mesh_distance(ida, idb, f64::INFINITY).unwrap();
            prop_assert_eq!(
                got.map(|g| g.distance),
                want.map(|w| f64::from(w.distance)),
                "mesh_distance {} {}",
                ia,
                ib
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

/// Everything the scene says about itself must equal the model.
fn check_state(scene: &Scene, model: &Model) -> Result<(), TestCaseError> {
    let live = model.live();
    prop_assert_eq!(
        scene.ids().collect::<Vec<_>>(),
        live.iter().map(|&i| model.meshes[i].id).collect::<Vec<_>>()
    );
    for &i in &live {
        let m = &model.meshes[i];
        let mesh = scene.mesh(m.id).unwrap();
        if let Err(e) = m.data.check(mesh) {
            return Err(TestCaseError::fail(format!("mesh {i}: {e}")));
        }
        prop_assert_eq!(scene.placement(m.id).unwrap(), m.placement, "mesh {}", i);
        prop_assert_eq!(scene.is_current(m.id).unwrap(), m.current(), "mesh {}", i);
        // Slots are released exactly when the model's history lets go.
        prop_assert_eq!(
            mesh.free_vertex_slots(),
            m.shadow.free_vertex_slots(),
            "free vertex slots of mesh {}",
            i
        );
        prop_assert_eq!(
            mesh.free_triangle_slots(),
            m.shadow.free_triangle_slots(),
            "free triangle slots of mesh {}",
            i
        );
        if m.mode == UpdateMode::Eager {
            prop_assert!(m.current(), "mesh {} is Eager but the model says stale", i);
        }
    }
    prop_assert_eq!(
        scene.can_undo(),
        !model.undo.is_empty() || !model.open.is_empty()
    );
    prop_assert_eq!(scene.can_redo(), !model.redo.is_empty());
    for m in model.meshes.iter().filter(|m| !m.live) {
        prop_assert!(matches!(scene.mesh(m.id), Err(SceneError::UnknownMesh)));
    }
    Ok(())
}

/// One report against the model's data since the last call.
fn check_report(m: &ModelMesh, i: usize, c: &MeshChanges) -> Result<(), TestCaseError> {
    prop_assert!(!c.added, "mesh {} reported as added", i);
    prop_assert_eq!(c.removed, !m.live, "removed flag of mesh {}", i);
    // Exactly when a placement change ran, even one undone since.
    prop_assert_eq!(
        c.placement_changed,
        m.touched,
        "placement_changed of mesh {}",
        i
    );
    if m.placement != m.snap_placement {
        prop_assert!(c.placement_changed, "mesh {} moved unreported", i);
    }
    let Some(cs) = &c.changes else {
        // No faults are injected, so no step ends `Corrupted`, and only
        // initial adds (checked apart) and compaction report everything.
        prop_assert!(m.everything, "mesh {} reported `None` uncompacted", i);
        return Ok(());
    };
    prop_assert!(!m.everything, "compacted mesh {} reported with changes", i);
    let d = Diff::of(&m.snap, &m.data);
    prop_assert_eq!(&cs.vertices_added, &d.vertices_added, "mesh {}", i);
    prop_assert_eq!(&cs.vertices_removed, &d.vertices_removed, "mesh {}", i);
    prop_assert_eq!(&cs.triangles_added, &d.triangles_added, "mesh {}", i);
    prop_assert_eq!(&cs.triangles_removed, &d.triangles_removed, "mesh {}", i);
    // Moved and changed may hold slots written and restored again, but every
    // slot whose content differs is among them. A moved vertex is dirty for
    // every channel, so the channel lists need not name it.
    for v in &d.vertices_moved {
        prop_assert!(cs.vertices_moved.contains(v), "mesh {} vertex {}", i, v);
    }
    for t in &d.triangles_changed {
        prop_assert!(cs.triangles_changed.contains(t), "mesh {} tri {}", i, t);
    }
    // And they only hold slots alive at both calls.
    for &v in &cs.vertices_moved {
        let alive = slot(&m.snap.positions, v as usize).is_some()
            && slot(&m.data.positions, v as usize).is_some();
        prop_assert!(alive, "mesh {} vertex {} moved but not alive", i, v);
    }
    for &t in &cs.triangles_changed {
        let alive = slot(&m.snap.triangles, t as usize).is_some()
            && slot(&m.data.triangles, t as usize).is_some();
        prop_assert!(alive, "mesh {} triangle {} changed but not alive", i, t);
    }
    Ok(())
}

/// `take_changes` against the model, then the model's snapshots move on.
fn check_reports(scene: &mut Scene, model: &mut Model) -> Result<(), TestCaseError> {
    let got = scene.take_changes();
    prop_assert!(got.windows(2).all(|w| w[0].id < w[1].id), "{:?}", got);
    let mut seen = vec![false; model.meshes.len()];
    for c in &got {
        let i = model.meshes.iter().position(|m| m.id == c.id);
        let Some(i) = i.filter(|&i| !model.meshes[i].gone) else {
            return Err(TestCaseError::fail(format!("unexpected report {c:?}")));
        };
        seen[i] = true;
        check_report(&model.meshes[i], i, c)?;
    }
    for (i, m) in model.meshes.iter().enumerate() {
        if seen[i] || m.gone {
            continue;
        }
        // Left out: nothing to report.
        prop_assert!(m.live, "removal of mesh {} not reported", i);
        prop_assert!(!m.touched, "placement change of mesh {} not reported", i);
        prop_assert!(!m.everything, "compaction of mesh {} not reported", i);
        let d = Diff::of(&m.snap, &m.data);
        prop_assert!(d.is_empty(), "mesh {} changed unreported: {:?}", i, d);
    }
    for m in &mut model.meshes {
        m.snap = m.data.clone();
        m.snap_placement = m.placement;
        m.touched = false;
        m.everything = false;
        m.gone = !m.live;
    }
    Ok(())
}

/// Sets a placement on the scene and the model; an unchanged one records and
/// reports nothing.
fn set_placement(scene: &mut Scene, model: &mut Model, i: usize, new: Placement) {
    let old = model.meshes[i].placement;
    scene.set_placement(model.meshes[i].id, new).unwrap();
    if new != old {
        model.meshes[i].placement = new;
        model.meshes[i].touched = true;
        model.record(Entry::Place { mesh: i, old, new });
    }
}

fn apply(scene: &mut Scene, model: &mut Model, action: &Action) -> Result<(), TestCaseError> {
    match action {
        Action::Edit(sel, op) => {
            let Some(i) = model.choose(*sel) else {
                return Ok(());
            };
            let id = model.meshes[i].id;
            let before = (model.meshes[i].data.clone(), model.meshes[i].rev);
            let delta = scene.edit(id, |tx| run(tx, &before.0, op)).unwrap();
            // The same edit on the shadow hands out the same slots.
            let mut tx = model.meshes[i].shadow.begin_edit();
            let shadow_delta = run(&mut tx, &before.0, op).unwrap();
            let patch = tx.commit().unwrap();
            prop_assert_eq!(&delta, &shadow_delta);
            if let Some(delta) = delta {
                let mut data = before.0.clone();
                delta.apply(&mut data);
                let after = (data, model.fresh_rev());
                model.set_data(i, &after);
                model.record(Entry::Edit {
                    mesh: i,
                    before,
                    after,
                    patch: Box::new(patch),
                });
            }
        }
        Action::SetPlacement(sel, seed) => {
            let Some(i) = model.choose(*sel) else {
                return Ok(());
            };
            set_placement(scene, model, i, placement(*seed, model.meshes[i].anchor));
        }
        Action::SetSamePlacement(sel) => {
            let Some(i) = model.choose(*sel) else {
                return Ok(());
            };
            set_placement(scene, model, i, model.meshes[i].placement);
        }
        Action::Drag(sel, (kind, q, j), frames) => {
            let Some(i) = model.choose(*sel) else {
                return Ok(());
            };
            // Each frame slides a little further along x; the last frame of
            // some drags lands back on the first one's placement, and of
            // others back on the placement from before the drag, which
            // coalesces into an entry that does nothing.
            let before = model.meshes[i].placement;
            for f in 0..*frames as i8 {
                let last = f + 1 == *frames as i8;
                let p = if last && kind % 4 == 2 {
                    before
                } else {
                    let jx = if last && kind % 2 == 0 {
                        j[0]
                    } else {
                        j[0].wrapping_add(f.wrapping_mul(3))
                    };
                    placement((*kind, *q, [jx, j[1], j[2]]), model.meshes[i].anchor)
                };
                set_placement(scene, model, i, p);
            }
        }
        Action::Begin => {
            scene.begin_step();
            model.depth += 1;
        }
        Action::End => {
            let r = scene.end_step();
            if model.depth == 0 {
                prop_assert!(matches!(r, Err(SceneError::StepNotOpen)), "{r:?}");
            } else {
                prop_assert!(r.is_ok(), "{r:?}");
                model.depth -= 1;
                if model.depth == 0 {
                    model.close();
                }
            }
        }
        Action::Cancel => {
            let r = scene.cancel_step();
            if model.depth == 0 {
                prop_assert!(matches!(r, Err(SceneError::StepNotOpen)), "{r:?}");
            } else {
                prop_assert!(r.is_ok(), "{r:?}");
                model.cancel();
            }
        }
        Action::Undo => {
            // With eviction on, this is where a wrong step count shows: the
            // scene undoes when the model has nothing left, or the other way.
            let got = scene.undo().unwrap();
            prop_assert_eq!(got, model.undo());
        }
        Action::Redo => {
            let got = scene.redo().unwrap();
            prop_assert_eq!(got, model.redo());
        }
        Action::ClearHistory => {
            scene.clear_history().unwrap();
            model.clear();
        }
        Action::SetMode(sel, mode) => {
            let Some(i) = model.choose(*sel) else {
                return Ok(());
            };
            scene.set_mode(model.meshes[i].id, *mode).unwrap();
            model.meshes[i].mode = *mode;
            if *mode == UpdateMode::Eager {
                model.meshes[i].refresh();
            }
        }
        Action::Update(sel) => {
            let Some(i) = model.choose(*sel) else {
                return Ok(());
            };
            scene.update(model.meshes[i].id).unwrap();
            model.meshes[i].refresh();
        }
        Action::Flush => {
            scene.flush().unwrap();
            for m in model.meshes.iter_mut().filter(|m| m.live) {
                if m.mode != UpdateMode::Manual {
                    m.refresh();
                }
            }
            // After a flush every non-Manual BVH is current.
            for m in model.meshes.iter().filter(|m| m.live) {
                if m.mode != UpdateMode::Manual {
                    prop_assert!(scene.is_current(m.id).unwrap());
                }
            }
        }
        Action::Remove(sel) => {
            let Some(i) = model.choose(*sel) else {
                return Ok(());
            };
            let mesh = scene.remove_mesh(model.meshes[i].id).unwrap();
            if let Err(e) = model.meshes[i].data.check(&mesh) {
                return Err(TestCaseError::fail(format!("removed mesh {i}: {e}")));
            }
            model.forget(i, false);
            model.meshes[i].live = false;
            // The removed mesh comes back with its history's slots released.
            let shadow = &model.meshes[i].shadow;
            prop_assert_eq!(mesh.free_vertex_slots(), shadow.free_vertex_slots());
            prop_assert_eq!(mesh.free_triangle_slots(), shadow.free_triangle_slots());
        }
        Action::Compact(sel) => {
            let Some(i) = model.choose(*sel) else {
                return Ok(());
            };
            let remap = scene.compact(model.meshes[i].id).unwrap();
            let shadow_remap = model.meshes[i].shadow.compact().unwrap();
            prop_assert_eq!(&remap.vertices, &shadow_remap.vertices);
            prop_assert_eq!(&remap.triangles, &shadow_remap.triangles);
            model.meshes[i].everything = true;
            let old = model.meshes[i].data.clone();
            let mut data = Data::default();
            for (v, p) in old.positions.iter().enumerate() {
                if let Some(p) = p {
                    let n = remap.vertices[v];
                    prop_assert_ne!(n, fast_mesh_store::Remap::REMOVED);
                    set_slot(&mut data.positions, n as usize, Some(*p));
                }
            }
            for (t, tri) in old.triangles.iter().enumerate() {
                if let Some(tri) = tri {
                    let n = remap.triangles[t];
                    prop_assert_ne!(n, fast_mesh_store::Remap::REMOVED);
                    let tri = tri.map(|v| remap.vertices[v as usize]);
                    set_slot(&mut data.triangles, n as usize, Some(tri));
                }
            }
            model.meshes[i].data = data;
            // Compaction is a new revision, and the BVH is rebuilt whatever
            // the mode.
            model.meshes[i].rev = model.fresh_rev();
            model.meshes[i].refresh();
            model.forget(i, true);
        }
        Action::Query(seed) => check_queries(scene, model, *seed)?,
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases(64)))]

    #[test]
    fn scene_matches_model(
        count in 2usize..=3,
        builds in prop::collection::vec(build_options(), 3),
        modes in prop::collection::vec(mode(), 3),
        starts in prop::collection::vec(placement_seed(), 3),
        seed in any::<u64>(),
        budget in budget(),
        actions in prop::collection::vec(action(), 1..40),
    ) {
        let mut scene = Scene::new(SceneOptions { history_budget: budget });
        // The second mesh is built far from zero, so that it has an origin;
        // the others have none.
        let specs: [(u32, [f64; 3]); 3] = [
            (3, [0.0; 3]),
            (4, [1000.5, -300.25, 40.0]),
            (3, [0.0; 3]),
        ];
        let mut model = Model {
            last_rev: 0,
            meshes: Vec::new(),
            undo: Vec::new(),
            redo: Vec::new(),
            open: Vec::new(),
            depth: 0,
            budget,
        };
        for k in 0..count {
            let (n, offset) = specs[k];
            let mesh = grid(n, offset);
            let anchor = [
                offset[0] + f64::from(n) / 2.0,
                offset[1] + f64::from(n) / 2.0,
                offset[2] + 0.6,
            ];
            let data = Data::of(&mesh);
            let p = placement(starts[k], anchor);
            let id = scene
                .add_mesh(
                    mesh,
                    MeshOptions {
                        build: builds[k],
                        mode: modes[k],
                        placement: p,
                        ..Default::default()
                    },
                )
                .unwrap();
            model.meshes.push(ModelMesh {
                id,
                live: true,
                snap: data.clone(),
                data,
                shadow: grid(n, offset),
                placement: p,
                mode: modes[k],
                snap_placement: p,
                touched: false,
                everything: false,
                gone: false,
                rev: 0,
                bvh_rev: 0,
                anchor,
            });
        }
        // Freshly added meshes are reported as added and fully changed.
        let first = scene.take_changes();
        prop_assert_eq!(first.len(), count);
        for (c, m) in first.iter().zip(&model.meshes) {
            prop_assert_eq!(c.id, m.id);
            prop_assert!(c.added && !c.removed && !c.placement_changed);
            prop_assert!(c.changes.is_none());
        }
        check_state(&scene, &model)?;
        check_queries(&mut scene, &mut model, seed)?;
        for (i, a) in actions.iter().enumerate() {
            // `check_state` asserts that an Eager BVH is current, and
            // `check_eager` that it answers like a fresh build.
            apply(&mut scene, &mut model, a)?;
            check_state(&scene, &model)?;
            // Skipped on some actions, so that reports folding across
            // several actions are checked too.
            if (i + seed as usize) % 3 != 2 {
                check_reports(&mut scene, &mut model)?;
            }
            check_eager(&mut scene, &model, seed.wrapping_add(i as u64))?;
        }
        scene.flush().unwrap();
        for m in model.meshes.iter_mut().filter(|m| m.live) {
            if m.mode != UpdateMode::Manual {
                m.refresh();
            }
        }
        check_state(&scene, &model)?;
        check_queries(&mut scene, &mut model, seed ^ 0x5eed)?;
        check_state(&scene, &model)?;
        check_reports(&mut scene, &mut model)?;
    }
}
