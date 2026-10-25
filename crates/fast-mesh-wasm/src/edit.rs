//! Geometry edit op lists, repair op lists and the analysis report, as the
//! flat buffers [`crate::scene::WasmScene`] exchanges with TypeScript.
//!
//! Plain Rust, no wasm-bindgen. Each list is decoded and checked in full
//! before the scene is touched; the scene then runs it inside one
//! [`Scene::edit`](fast_mesh_scene::Scene::edit), so it is one undo step and
//! a failing op rolls back the ops before it.
//!
//! # Edit op list
//!
//! Two buffers: `ops` (`u32`) and `values` (`f64`). `ops` is a sequence of
//! ops, each starting with its code:
//!
//! - `0` set positions: an id list of vertices; takes `3 · n` values.
//! - `1` add vertices: `n`; takes `3 · n` values. Produces `n` vertex ids.
//! - `2` add triangles: an id list of vertices, `n` a multiple of 3.
//!   Produces `n / 3` triangle ids.
//! - `3` remove triangles: an id list of triangles.
//! - `4` remove vertices: an id list of vertices.
//!
//! An id list is `n, r, id × n, (slot, op, i) × r`: `n` ids, of which the
//! `r` refs replace the id at `slot` (ascending) with id `i` of the result
//! of the earlier op `op`, which must produce ids of the kind the list
//! holds. `values` holds world positions (`xyz` each) in op order and must
//! be used up exactly. The result is `[n, id × n]` per op, empty (`n = 0`)
//! for ops that add nothing.
//!
//! # Repair op list
//!
//! Two buffers: `ops` (`f64`) and `loops` (`u32`). Each op is its code and
//! its options, NaN meaning "unset" (a bound) or "default"; flags are 0 or 1:
//!
//! - `0` weld: `tolerance`.
//! - `1` remove degenerate: `area_epsilon, remove_unused_vertices`.
//! - `2` remove small components: `min_triangles, min_area, keep_largest`.
//! - `3` orient.
//! - `4` fill holes: `max_edges, max_perimeter, max_area, loop_count,
//!   allow_unclean, fair, dp_limit, max_dihedral, max_refine_vertices`;
//!   `loop_count` (NaN for no id list) loop ids are taken from `loops`.
//! - `5` inflate: `cx, cy, cz, radius, falloff (0 linear, 1 smooth),
//!   strength`.
//! - `6` smooth: `cx, cy, cz, radius, falloff, strength, cycles`.
//! - `7` remesh: `cx, cy, cz, radius, target_length (NaN for the mesh mean),
//!   max_rounds (NaN for the default)`.
//! - `8` bridge loops: `max_edges (NaN for the default)`; exactly two loop
//!   ids are taken from `loops`.
//!
//! A loop id is four `u32`s: `min_vertex, len, hash low, hash high`. The
//! reports, concatenated in op order (`f64`):
//!
//! - weld: `merged_vertices, removed_triangles`.
//! - remove degenerate: `removed_triangles, removed_vertices`.
//! - remove small components: `removed_components, removed_triangles,
//!   removed_vertices`.
//! - orient: `flipped_triangles, patches, non_orientable_patches,
//!   flipped_outward_components`.
//! - fill holes: `filled_n, (loop id, triangles_added, vertices_added,
//!   converged, weights (0 none, 1 cotangent, 2 uniform), refine_capped)
//!   × filled_n, skipped_n, (loop id, reason (0 bounds, 1 unclean)) ×
//!   skipped_n, failed_n, (loop id, reason (0 projection folds, 1 no valid
//!   triangulation, 2 solver)) × failed_n`.
//! - inflate: `moved, max_displacement`.
//! - smooth: `moved, max_displacement`.
//! - remesh: `splits, collapses, flips, rounds`.
//! - bridge loops: `triangles_added`.
//!
//! # Analysis
//!
//! One `f64` array, see [`encode_analysis`].

use fast_mesh_ops::bridge::BridgeOptions;
use fast_mesh_ops::deform::{Brush, Falloff, InflateOptions, SmoothOptions};
use fast_mesh_ops::remesh::{REMESH_MAX_ROUNDS, Region, RemeshOptions};
use fast_mesh_ops::{
    Analysis, ComponentOptions, DegenerateOptions, FailReason, FairWeights, FillOptions,
    FillReport, LoopId, OpsError, SkipReason,
};
use fast_mesh_store::{StoreError, Transaction};

use crate::error::{Error, codes};

fn oom() -> Error {
    StoreError::OutOfMemory.into()
}

/// An empty vector with room for `n` elements.
fn reserved<T>(n: usize) -> Result<Vec<T>, StoreError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| StoreError::OutOfMemory)?;
    Ok(v)
}

/// Appends `values` with a fallible reservation.
fn put(out: &mut Vec<f64>, values: &[f64]) -> Result<(), StoreError> {
    out.try_reserve(values.len())
        .map_err(|_| StoreError::OutOfMemory)?;
    out.extend_from_slice(values);
    Ok(())
}

/// Appends `ids` as `f64`s with a fallible reservation.
fn put_ids(out: &mut Vec<f64>, ids: &[u32]) -> Result<(), StoreError> {
    out.try_reserve(ids.len())
        .map_err(|_| StoreError::OutOfMemory)?;
    out.extend(ids.iter().map(|&i| f64::from(i)));
    Ok(())
}

/// A count for a flat `f64` result (lossless: far below 2^53).
fn count(n: usize) -> f64 {
    n as f64
}

fn malformed(what: &str) -> Error {
    Error::invalid(&format!("malformed {what} op list"))
}

fn bad_index(text: &str) -> Error {
    Error {
        code: codes::INVALID_INDEX,
        text: text.to_owned(),
    }
}

/// Reads a buffer front to back.
struct Cursor<'a, T> {
    data: &'a [T],
    what: &'static str,
}

impl<'a, T: Copy> Cursor<'a, T> {
    fn new(data: &'a [T], what: &'static str) -> Self {
        Cursor { data, what }
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn take(&mut self, n: usize) -> Result<&'a [T], Error> {
        if n > self.data.len() {
            return Err(malformed(self.what));
        }
        let (head, rest) = self.data.split_at(n);
        self.data = rest;
        Ok(head)
    }

    fn next(&mut self) -> Result<T, Error> {
        Ok(self.take(1)?[0])
    }
}

/// What an id list holds or an op produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Element {
    Vertex,
    Triangle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    SetPositions,
    AddVertices,
    AddTriangles,
    RemoveTriangles,
    RemoveVertices,
}

impl Kind {
    fn from_code(code: u32) -> Result<Kind, Error> {
        Ok(match code {
            0 => Kind::SetPositions,
            1 => Kind::AddVertices,
            2 => Kind::AddTriangles,
            3 => Kind::RemoveTriangles,
            4 => Kind::RemoveVertices,
            _ => return Err(Error::invalid("unknown edit op")),
        })
    }

    /// What the op's id list holds; `None` when it has none.
    fn takes(self) -> Option<Element> {
        match self {
            Kind::SetPositions | Kind::AddTriangles | Kind::RemoveVertices => Some(Element::Vertex),
            Kind::RemoveTriangles => Some(Element::Triangle),
            Kind::AddVertices => None,
        }
    }

    /// What the op's result holds; `None` when it adds nothing.
    fn produces(self) -> Option<Element> {
        match self {
            Kind::AddVertices => Some(Element::Vertex),
            Kind::AddTriangles => Some(Element::Triangle),
            _ => None,
        }
    }
}

/// One decoded edit op, checked against the ops before it.
#[derive(Debug)]
pub(crate) struct EditOp<'a> {
    kind: Kind,
    ids: &'a [u32],
    /// `(slot, op, i)`: the id at `slot` is id `i` of op `op`'s result.
    refs: &'a [[u32; 3]],
    /// Local positions, for set positions and add vertices.
    positions: Vec<[f32; 3]>,
    /// How many ids the op produces.
    produced: usize,
}

/// Decodes an edit op list (see the module docs), mapping world positions
/// to local `f32` with `to_local`. Every ref is checked here, so a bad ref
/// fails before anything is written.
///
/// # Errors
/// `INVALID_ARGUMENT` for a malformed list, an unknown op, a triangle list
/// whose length is not a multiple of 3, or a position that is not finite in
/// local `f32`; `INVALID_INDEX` for a ref to an op that is not earlier, that
/// produces the other kind of id, or past its result.
pub(crate) fn decode_edit<'a>(
    ops: &'a [u32],
    values: &[f64],
    to_local: impl Fn([f64; 3]) -> [f64; 3],
) -> Result<Vec<EditOp<'a>>, Error> {
    const WHAT: &str = "edit";
    let mut words = Cursor::new(ops, WHAT);
    let mut vals = Cursor::new(values, WHAT);
    let mut out: Vec<EditOp> = Vec::new();
    while !words.is_empty() {
        let kind = Kind::from_code(words.next()?)?;
        let n = words.next()? as usize;
        let (ids, refs): (&[u32], &[[u32; 3]]) = if kind.takes().is_some() {
            let r = words.next()? as usize;
            let ids = words.take(n)?;
            let flat = words.take(r.checked_mul(3).ok_or_else(|| malformed(WHAT))?)?;
            (ids, flat.as_chunks::<3>().0)
        } else {
            (&[], &[])
        };
        let mut positions = Vec::new();
        if matches!(kind, Kind::SetPositions | Kind::AddVertices) {
            let flat = vals.take(n.checked_mul(3).ok_or_else(|| malformed(WHAT))?)?;
            positions.try_reserve_exact(n).map_err(|_| oom())?;
            for &[x, y, z] in flat.as_chunks::<3>().0 {
                let p = to_local([x, y, z]).map(|c| c as f32);
                if !p.iter().all(|c| c.is_finite()) {
                    return Err(Error::invalid(
                        "positions must be finite and fit f32 relative to the mesh origin",
                    ));
                }
                positions.push(p);
            }
        }
        let produced = match kind {
            Kind::AddVertices => n,
            Kind::AddTriangles if !n.is_multiple_of(3) => {
                return Err(Error::invalid("triangles need 3 vertex ids each"));
            }
            Kind::AddTriangles => n / 3,
            _ => 0,
        };
        let mut previous: Option<u32> = None;
        for &[slot, op, i] in refs {
            if slot as usize >= n || previous.is_some_and(|p| slot <= p) {
                return Err(malformed(WHAT));
            }
            previous = Some(slot);
            let target = out
                .get(op as usize)
                .ok_or_else(|| bad_index("a ref must name an earlier op"))?;
            if target.kind.produces() != kind.takes() {
                return Err(bad_index(
                    "a ref names an op that adds the other kind of id",
                ));
            }
            if i as usize >= target.produced {
                return Err(bad_index("a ref is past the end of its op's result"));
            }
        }
        out.try_reserve(1).map_err(|_| oom())?;
        out.push(EditOp {
            kind,
            ids,
            refs,
            positions,
            produced,
        });
    }
    if !vals.is_empty() {
        return Err(malformed(WHAT));
    }
    Ok(out)
}

/// Runs decoded edit ops in `tx` and returns each op's new ids.
///
/// # Errors
/// The store's error for the first op that fails, for example
/// [`StoreError::InvalidVertex`] for an id that is not alive; the caller's
/// transaction is then rolled back.
pub(crate) fn run_edit(tx: &mut Transaction, ops: &[EditOp]) -> Result<Vec<Vec<u32>>, StoreError> {
    let mut results: Vec<Vec<u32>> = reserved(ops.len())?;
    for op in ops {
        let mut ids = reserved(op.ids.len())?;
        ids.extend_from_slice(op.ids);
        // Checked by `decode_edit`: earlier op, right kind, in range.
        for &[slot, src, i] in op.refs {
            ids[slot as usize] = results[src as usize][i as usize];
        }
        let added = match op.kind {
            Kind::SetPositions => {
                tx.set_positions(&ids, &op.positions)?;
                Vec::new()
            }
            Kind::AddVertices => tx.add_vertices(&op.positions)?,
            Kind::AddTriangles => tx.add_triangles(ids.as_chunks::<3>().0)?,
            Kind::RemoveTriangles => {
                tx.remove_triangles(&ids)?;
                Vec::new()
            }
            Kind::RemoveVertices => {
                tx.remove_vertices(&ids)?;
                Vec::new()
            }
        };
        results.push(added);
    }
    Ok(results)
}

/// The result of [`run_edit`] as `[n, id × n]` per op.
pub(crate) fn encode_added(added: &[Vec<u32>]) -> Result<Vec<u32>, Error> {
    let len = added.iter().map(|a| 1 + a.len()).sum();
    let mut out = reserved(len)?;
    for a in added {
        out.push(a.len() as u32);
        out.extend_from_slice(a);
    }
    Ok(out)
}

/// One decoded repair op.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RepairOp {
    Weld(f32),
    Degenerate(DegenerateOptions),
    Components(ComponentOptions),
    Orient,
    Fill(FillOptions),
    Inflate(InflateOptions),
    Smooth(SmoothOptions),
    Remesh(RemeshOptions),
    Bridge(BridgeOptions),
}

/// A flag: 0 or 1.
fn flag(v: f64) -> Result<bool, Error> {
    if v == 0.0 {
        Ok(false)
    } else if v == 1.0 {
        Ok(true)
    } else {
        Err(Error::invalid("flags must be 0 or 1"))
    }
}

/// A brush falloff: 0 linear, 1 smooth.
fn falloff(v: f64) -> Result<Falloff, Error> {
    if v == 0.0 {
        Ok(Falloff::Linear)
    } else if v == 1.0 {
        Ok(Falloff::Smooth)
    } else {
        Err(Error::invalid("falloff must be 0 (linear) or 1 (smooth)"))
    }
}

/// A count, or `None` for NaN.
fn count_opt(v: f64, what: &str) -> Result<Option<usize>, Error> {
    if v.is_nan() {
        return Ok(None);
    }
    if !(v >= 0.0 && v.is_finite() && v.fract() == 0.0) {
        return Err(Error::invalid(&format!(
            "{what} must be a non-negative integer"
        )));
    }
    // Saturates above usize::MAX, which no mesh reaches anyway.
    Ok(Some(v as usize))
}

/// A bound, or `None` for NaN; the operation checks its range.
fn bound(v: f64) -> Option<f64> {
    (!v.is_nan()).then_some(v)
}

fn loop_id(w: [u32; 4]) -> LoopId {
    LoopId {
        min_vertex: w[0],
        len: w[1],
        hash: u64::from(w[2]) | (u64::from(w[3]) << 32),
    }
}

/// Decodes a repair op list (see the module docs). The operations check
/// their own options when they run.
///
/// # Errors
/// `INVALID_ARGUMENT` for a malformed list, an unknown op, a flag that is
/// not 0 or 1, or a count that is not a non-negative integer.
pub(crate) fn decode_repair(ops: &[f64], loops: &[u32]) -> Result<Vec<RepairOp>, Error> {
    const WHAT: &str = "repair";
    let mut words = Cursor::new(ops, WHAT);
    let mut ids = Cursor::new(loops, WHAT);
    let mut out = Vec::new();
    while !words.is_empty() {
        let code = words.next()?;
        let op = if code == 0.0 {
            RepairOp::Weld(words.next()? as f32)
        } else if code == 1.0 {
            let [eps, unused] = words.take(2)?.try_into().map_err(|_| malformed(WHAT))?;
            let mut o = DegenerateOptions::default();
            if let Some(eps) = bound(eps) {
                o.area_epsilon = eps;
            }
            o.remove_unused_vertices = flag(unused)?;
            RepairOp::Degenerate(o)
        } else if code == 2.0 {
            let [min_t, min_a, keep] = words.take(3)?.try_into().map_err(|_| malformed(WHAT))?;
            RepairOp::Components(ComponentOptions {
                min_triangles: count_opt(min_t, "minTriangles")?,
                min_area: bound(min_a),
                keep_largest: flag(keep)?,
            })
        } else if code == 3.0 {
            RepairOp::Orient
        } else if code == 4.0 {
            let [max_e, max_p, max_a, n, unclean, fair, dp, max_dh, max_rv]: [f64; 9] =
                words.take(9)?.try_into().map_err(|_| malformed(WHAT))?;
            let loops = match count_opt(n, "loop count")? {
                None => None,
                Some(n) => {
                    let flat = ids.take(n.checked_mul(4).ok_or_else(|| malformed(WHAT))?)?;
                    let mut v = reserved(n)?;
                    v.extend(flat.as_chunks::<4>().0.iter().map(|&w| loop_id(w)));
                    Some(v)
                }
            };
            let mut o = FillOptions {
                max_edges: count_opt(max_e, "maxEdges")?,
                max_perimeter: bound(max_p),
                max_area: bound(max_a),
                loops,
                allow_unclean: flag(unclean)?,
                fair: flag(fair)?,
                max_dihedral: bound(max_dh),
                max_refine_vertices: count_opt(max_rv, "maxRefineVertices")?,
                ..FillOptions::default()
            };
            if let Some(dp) = count_opt(dp, "dpLimit")? {
                o.dp_limit = dp;
            }
            RepairOp::Fill(o)
        } else if code == 5.0 {
            let [cx, cy, cz, radius, fo, strength]: [f64; 6] =
                words.take(6)?.try_into().map_err(|_| malformed(WHAT))?;
            RepairOp::Inflate(InflateOptions {
                brush: Brush {
                    center: [cx as f32, cy as f32, cz as f32],
                    radius: radius as f32,
                    falloff: falloff(fo)?,
                    strength: strength as f32,
                },
            })
        } else if code == 6.0 {
            let [cx, cy, cz, radius, fo, strength, cycles]: [f64; 7] =
                words.take(7)?.try_into().map_err(|_| malformed(WHAT))?;
            // Before `count_opt`: `v as usize` saturates at `usize::MAX`,
            // which on wasm32 equals `u32::MAX`, so without this check any
            // larger value would decode to billions of cycles.
            if !cycles.is_nan() && cycles > f64::from(u32::MAX) {
                return Err(Error::invalid("cycles is too large"));
            }
            let cycles = count_opt(cycles, "cycles")?.unwrap_or_default();
            RepairOp::Smooth(SmoothOptions {
                brush: Brush {
                    center: [cx as f32, cy as f32, cz as f32],
                    radius: radius as f32,
                    falloff: falloff(fo)?,
                    strength: strength as f32,
                },
                cycles: u32::try_from(cycles).map_err(|_| Error::invalid("cycles is too large"))?,
            })
        } else if code == 7.0 {
            let [cx, cy, cz, radius, target, rounds]: [f64; 6] =
                words.take(6)?.try_into().map_err(|_| malformed(WHAT))?;
            // As with smooth's `cycles`: without this check any value past
            // `u32::MAX` would decode to billions of rounds on wasm32.
            if !rounds.is_nan() && rounds > f64::from(u32::MAX) {
                return Err(Error::invalid("maxRounds is too large"));
            }
            let rounds = count_opt(rounds, "maxRounds")?
                .map(|r| u32::try_from(r).map_err(|_| Error::invalid("maxRounds is too large")));
            RepairOp::Remesh(RemeshOptions {
                region: Region {
                    center: [cx as f32, cy as f32, cz as f32],
                    radius: radius as f32,
                },
                target_length: bound(target).map(|t| t as f32),
                max_rounds: match rounds {
                    None => REMESH_MAX_ROUNDS,
                    Some(r) => r?,
                },
            })
        } else if code == 8.0 {
            let [max_e]: [f64; 1] = words.take(1)?.try_into().map_err(|_| malformed(WHAT))?;
            let flat = ids.take(8)?;
            let pair = flat.as_chunks::<4>().0;
            RepairOp::Bridge(BridgeOptions {
                loops: (loop_id(pair[0]), loop_id(pair[1])),
                max_edges: count_opt(max_e, "maxEdges")?,
            })
        } else {
            return Err(Error::invalid("unknown repair op"));
        };
        out.try_reserve(1).map_err(|_| oom())?;
        out.push(op);
    }
    if !ids.is_empty() {
        return Err(malformed(WHAT));
    }
    Ok(out)
}

fn put_loop_id(out: &mut Vec<f64>, id: LoopId) -> Result<(), StoreError> {
    put(
        out,
        &[
            f64::from(id.min_vertex),
            f64::from(id.len),
            f64::from(id.hash as u32),
            f64::from((id.hash >> 32) as u32),
        ],
    )
}

fn put_fill(out: &mut Vec<f64>, r: &FillReport) -> Result<(), StoreError> {
    put(out, &[count(r.filled.len())])?;
    for f in &r.filled {
        put_loop_id(out, f.id)?;
        let weights = match f.weights {
            None => 0.0,
            Some(FairWeights::Cotangent) => 1.0,
            Some(FairWeights::Uniform) => 2.0,
        };
        put(
            out,
            &[
                count(f.triangles_added),
                count(f.vertices_added),
                f64::from(u8::from(f.converged)),
                weights,
                f64::from(u8::from(f.refine_capped)),
            ],
        )?;
    }
    put(out, &[count(r.skipped.len())])?;
    for &(id, reason) in &r.skipped {
        put_loop_id(out, id)?;
        let reason = match reason {
            SkipReason::Bounds => 0.0,
            SkipReason::Unclean => 1.0,
        };
        put(out, &[reason])?;
    }
    put(out, &[count(r.failed.len())])?;
    for &(id, reason) in &r.failed {
        put_loop_id(out, id)?;
        let reason = match reason {
            FailReason::ProjectionFolds => 0.0,
            FailReason::NoValidTriangulation => 1.0,
            FailReason::Solver => 2.0,
        };
        put(out, &[reason])?;
    }
    Ok(())
}

/// Runs decoded repair ops in `tx`, in order, and returns their reports
/// (see the module docs).
///
/// # Errors
/// The first operation's error; the caller's transaction is then rolled
/// back.
pub(crate) fn run_repair(tx: &mut Transaction, ops: &[RepairOp]) -> Result<Vec<f64>, OpsError> {
    let mut out = Vec::new();
    for op in ops {
        match op {
            RepairOp::Weld(tolerance) => {
                let r = fast_mesh_ops::weld(tx, *tolerance)?;
                put(
                    &mut out,
                    &[count(r.merged_vertices), count(r.removed_triangles)],
                )?;
            }
            RepairOp::Degenerate(o) => {
                let r = fast_mesh_ops::remove_degenerate(tx, o)?;
                put(
                    &mut out,
                    &[count(r.removed_triangles), count(r.removed_vertices)],
                )?;
            }
            RepairOp::Components(o) => {
                let r = fast_mesh_ops::remove_small_components(tx, o)?;
                put(
                    &mut out,
                    &[
                        count(r.removed_components),
                        count(r.removed_triangles),
                        count(r.removed_vertices),
                    ],
                )?;
            }
            RepairOp::Orient => {
                let r = fast_mesh_ops::orient(tx)?;
                put(
                    &mut out,
                    &[
                        count(r.flipped_triangles),
                        count(r.patches),
                        count(r.non_orientable_patches),
                        count(r.flipped_outward_components),
                    ],
                )?;
            }
            RepairOp::Fill(o) => {
                let r = fast_mesh_ops::fill_holes(tx, o)?;
                put_fill(&mut out, &r)?;
            }
            RepairOp::Inflate(o) => {
                let r = fast_mesh_ops::deform::inflate(tx, o)?;
                put(&mut out, &[count(r.moved), f64::from(r.max_displacement)])?;
            }
            RepairOp::Smooth(o) => {
                let r = fast_mesh_ops::deform::smooth(tx, o)?;
                put(&mut out, &[count(r.moved), f64::from(r.max_displacement)])?;
            }
            RepairOp::Remesh(o) => {
                let r = fast_mesh_ops::remesh::remesh(tx, o)?;
                put(
                    &mut out,
                    &[
                        count(r.splits),
                        count(r.collapses),
                        count(r.flips),
                        count(r.rounds),
                    ],
                )?;
            }
            RepairOp::Bridge(o) => {
                let r = fast_mesh_ops::bridge::bridge_loops(tx, o)?;
                put(&mut out, &[count(r.triangles_added)])?;
            }
        }
    }
    Ok(out)
}

/// An [`Analysis`] as one `f64` array:
///
/// 1. `loop_n`, then per loop: its id (4 values, see the module docs),
///    `vertex_n, vertices..., perimeter, area, closed, clean`.
/// 2. `coincident_groups, listed_n`, then per listed group `n, ids...`.
/// 3. `degenerate, listed_n, ids...`; then `duplicate, listed_n, ids...`.
/// 4. `non_manifold_edges, listed_n, (a, b) × listed_n`.
/// 5. `n, non-manifold vertices...`; `n, pinch vertices...`.
/// 6. `component_n`, then per component `triangle_n, triangles..., area,
///    closed`.
/// 7. `orientation_conflicts, self_intersects` (-1 not checked, 0, 1).
pub(crate) fn encode_analysis(a: &Analysis) -> Result<Vec<f64>, StoreError> {
    let mut out = Vec::new();
    put(&mut out, &[count(a.loops.len())])?;
    for l in &a.loops {
        put_loop_id(&mut out, l.id)?;
        put(&mut out, &[count(l.vertices.len())])?;
        put_ids(&mut out, &l.vertices)?;
        put(
            &mut out,
            &[
                l.perimeter,
                l.area,
                f64::from(u8::from(l.closed)),
                f64::from(u8::from(l.clean)),
            ],
        )?;
    }
    put(
        &mut out,
        &[count(a.coincident_groups), count(a.coincident.len())],
    )?;
    for g in &a.coincident {
        put(&mut out, &[count(g.len())])?;
        put_ids(&mut out, g)?;
    }
    for (n, list) in [
        (a.degenerate, &a.degenerate_list),
        (a.duplicate, &a.duplicate_list),
    ] {
        put(&mut out, &[count(n), count(list.len())])?;
        put_ids(&mut out, list)?;
    }
    put(
        &mut out,
        &[
            count(a.non_manifold_edges),
            count(a.non_manifold_edge_list.len()),
        ],
    )?;
    for &(x, y) in &a.non_manifold_edge_list {
        put_ids(&mut out, &[x, y])?;
    }
    for list in [&a.non_manifold_vertices, &a.pinch_vertices] {
        put(&mut out, &[count(list.len())])?;
        put_ids(&mut out, list)?;
    }
    put(&mut out, &[count(a.components.len())])?;
    for c in &a.components {
        put(&mut out, &[count(c.triangles.len())])?;
        put_ids(&mut out, &c.triangles)?;
        put(&mut out, &[c.area, f64::from(u8::from(c.closed))])?;
    }
    let self_intersects = match a.self_intersects {
        None => -1.0,
        Some(b) => f64::from(u8::from(b)),
    };
    put(&mut out, &[count(a.orientation_conflicts), self_intersects])?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id_list(ids: &[u32], refs: &[[u32; 3]]) -> Vec<u32> {
        let mut v = vec![ids.len() as u32, refs.len() as u32];
        v.extend_from_slice(ids);
        v.extend(refs.iter().flatten());
        v
    }

    fn shift(p: [f64; 3]) -> [f64; 3] {
        [p[0] - 1.0, p[1] - 2.0, p[2] - 3.0]
    }

    #[test]
    fn decode_edit_ops_and_refs() {
        // Add two vertices, a triangle using both by ref, remove triangle 0.
        let mut ops = vec![1, 2];
        ops.push(2);
        ops.extend(id_list(&[0, 0, 0], &[[1, 0, 0], [2, 0, 1]]));
        ops.push(3);
        ops.extend(id_list(&[0], &[]));
        let values = [1., 2., 3., 4., 5., 6.];
        let decoded = decode_edit(&ops, &values, shift).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].positions, vec![[0., 0., 0.], [3., 3., 3.]]);
        assert_eq!(decoded[0].produced, 2);
        assert_eq!(decoded[1].produced, 1);
        assert_eq!(decoded[1].refs, &[[1, 0, 0], [2, 0, 1]]);
        assert_eq!(decoded[2].ids, &[0]);
        assert!(decode_edit(&[], &[], shift).unwrap().is_empty());
    }

    #[test]
    fn decode_edit_rejects_bad_lists() {
        let code = |ops: &[u32], values: &[f64]| decode_edit(ops, values, shift).unwrap_err().code;
        let tri = |refs: &[[u32; 3]]| {
            let mut v = vec![1, 1];
            v.push(2);
            v.extend(id_list(&[0, 1, 2], refs));
            v
        };
        let p = [0.; 3];
        // Refs: later op, other kind, past the result.
        assert_eq!(code(&tri(&[[0, 1, 0]]), &p), codes::INVALID_INDEX);
        assert_eq!(code(&tri(&[[0, 0, 1]]), &p), codes::INVALID_INDEX);
        let mut v = tri(&[]);
        v.push(4);
        v.extend(id_list(&[0], &[[0, 1, 0]]));
        assert_eq!(code(&v, &p), codes::INVALID_INDEX);
        // Malformed: slot out of range or repeated, short buffers, values left over.
        assert_eq!(code(&tri(&[[3, 0, 0]]), &p), codes::INVALID_ARGUMENT);
        assert_eq!(
            code(&tri(&[[1, 0, 0], [1, 0, 0]]), &p),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(code(&[1, 1], &[]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[1, 1], &[0.; 4]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[3, 2, 0, 1], &[]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[3, 0, u32::MAX], &[]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[9], &[]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[2, 2, 0, 0, 1], &[]), codes::INVALID_ARGUMENT);
        // Positions that overflow f32 locally.
        assert_eq!(code(&[1, 1], &[1e300, 0., 0.]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[1, 1], &[f64::NAN, 0., 0.]), codes::INVALID_ARGUMENT);
    }

    #[test]
    fn decode_repair_rejects_cycles_edges() {
        let smooth = |cycles: f64| decode_repair(&[6., 0., 0., 0., 1., 0., 1., cycles], &[]);
        let code = |cycles: f64| smooth(cycles).unwrap_err().code;
        assert_eq!(code(-1.), codes::INVALID_ARGUMENT);
        assert_eq!(code(2f64.powi(32)), codes::INVALID_ARGUMENT);
        // The u32 boundary itself still decodes.
        assert_eq!(
            smooth(f64::from(u32::MAX)).unwrap(),
            vec![RepairOp::Smooth(SmoothOptions {
                brush: Brush {
                    center: [0.0; 3],
                    radius: 1.0,
                    falloff: Falloff::Linear,
                    strength: 1.0,
                },
                cycles: u32::MAX,
            })]
        );
    }

    #[test]
    fn decode_repair_ops() {
        let nan = f64::NAN;
        let hash: u64 = 0x1234_5678_9abc_def0;
        let ops = [
            0., 0.5, //
            1., nan, 1., //
            2., 10., nan, 0., //
            3., //
            4., nan, 2., nan, 1., 0., 1., nan, 0.5, 100.,
        ];
        let loops = [7, 12, hash as u32, (hash >> 32) as u32];
        let d = decode_repair(&ops, &loops).unwrap();
        assert_eq!(d[0], RepairOp::Weld(0.5));
        assert_eq!(d[1], RepairOp::Degenerate(DegenerateOptions::default()));
        assert_eq!(
            d[2],
            RepairOp::Components(ComponentOptions {
                min_triangles: Some(10),
                min_area: None,
                keep_largest: false,
            })
        );
        assert_eq!(d[3], RepairOp::Orient);
        let RepairOp::Fill(f) = &d[4] else {
            panic!("not a fill")
        };
        assert_eq!(f.max_perimeter, Some(2.));
        assert_eq!(f.max_edges, None);
        assert_eq!(
            f.loops,
            Some(vec![LoopId {
                min_vertex: 7,
                len: 12,
                hash
            }])
        );
        assert!(f.fair && !f.allow_unclean);
        assert_eq!(f.dp_limit, FillOptions::default().dp_limit);
        assert_eq!(f.max_dihedral, Some(0.5));
        assert_eq!(f.max_refine_vertices, Some(100));

        let code = |ops: &[f64], loops: &[u32]| decode_repair(ops, loops).unwrap_err().code;
        assert_eq!(code(&[5.], &[]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[1., nan, 2.], &[]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[2., 1.5, nan, 1.], &[]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[2., -1., nan, 1.], &[]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[0.], &[]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[3.], &[1]), codes::INVALID_ARGUMENT);
        assert_eq!(
            code(&[4., nan, nan, nan, 1., 0., 1., nan], &[1, 2, 3]),
            codes::INVALID_ARGUMENT
        );
    }

    #[test]
    fn decode_repair_remesh() {
        use fast_mesh_ops::remesh::{REMESH_MAX_ROUNDS, Region, RemeshOptions};

        let nan = f64::NAN;
        let d = decode_repair(&[7., 1., 2., 3., 4., 0.5, nan], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Remesh(RemeshOptions {
                region: Region {
                    center: [1.0, 2.0, 3.0],
                    radius: 4.0,
                },
                target_length: Some(0.5),
                max_rounds: REMESH_MAX_ROUNDS,
            })]
        );
        let d = decode_repair(&[7., 0., 0., 0., 1., nan, 3.], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Remesh(RemeshOptions {
                region: Region::default(),
                target_length: None,
                max_rounds: 3,
            })]
        );
        // The u32 boundary itself still decodes.
        let d = decode_repair(&[7., 0., 0., 0., 1., nan, f64::from(u32::MAX)], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Remesh(RemeshOptions {
                region: Region::default(),
                target_length: None,
                max_rounds: u32::MAX,
            })]
        );
        let code = |ops: &[f64]| decode_repair(ops, &[]).unwrap_err().code;
        assert_eq!(code(&[7.]), codes::INVALID_ARGUMENT);
        assert_eq!(
            code(&[7., 0., 0., 0., 1., nan, -1.]),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(&[7., 0., 0., 0., 1., nan, 1.5]),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(&[7., 0., 0., 0., 1., nan, 2f64.powi(32)]),
            codes::INVALID_ARGUMENT
        );
    }

    #[test]
    fn decode_repair_bridge() {
        use fast_mesh_ops::LoopId;
        use fast_mesh_ops::bridge::BridgeOptions;

        let nan = f64::NAN;
        let ids = |a: [u32; 4], b: [u32; 4]| [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]];
        let d = decode_repair(&[8., 64.], &ids([0, 4, 11, 22], [4, 4, 33, 44])).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Bridge(BridgeOptions {
                loops: (
                    LoopId {
                        min_vertex: 0,
                        len: 4,
                        hash: 11 | (22 << 32),
                    },
                    LoopId {
                        min_vertex: 4,
                        len: 4,
                        hash: 33 | (44 << 32),
                    },
                ),
                max_edges: Some(64),
            })]
        );
        let d = decode_repair(&[8., nan], &ids([0, 4, 11, 22], [4, 4, 33, 44])).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Bridge(BridgeOptions {
                loops: (
                    LoopId {
                        min_vertex: 0,
                        len: 4,
                        hash: 11 | (22 << 32),
                    },
                    LoopId {
                        min_vertex: 4,
                        len: 4,
                        hash: 33 | (44 << 32),
                    },
                ),
                max_edges: None,
            })]
        );
        let code = |ops: &[f64], ids: &[u32]| decode_repair(ops, ids).unwrap_err().code;
        let full = ids([0, 4, 11, 22], [4, 4, 33, 44]);
        assert_eq!(code(&[8.], &full), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[8., nan], &[0, 4, 11]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[8., -1.], &full), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[8., 1.5], &full), codes::INVALID_ARGUMENT);
    }
}
