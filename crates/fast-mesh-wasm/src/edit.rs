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
//! - `5` set UVs: an id list of vertices; takes `2 · n` values.
//!
//! An id list is `n, r, id × n, (slot, op, i) × r`: `n` ids, of which the
//! `r` refs replace the id at `slot` (ascending) with id `i` of the result
//! of the earlier op `op`, which must produce ids of the kind the list
//! holds. `values` holds world positions (`xyz` each) and UVs (`uv` each) in
//! op order and must be used up exactly. The result is `[n, id × n]` per op, empty (`n = 0`)
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
//!   strength, method (NaN/0 vertexNormal, 1 averageNormal, 2 strokeNormal)`.
//! - `6` smooth: `cx, cy, cz, radius, falloff, strength, cycles,
//!   method (NaN/0 taubin, 1 laplacian, 2 hcLaplacian, 3 cotangent)`.
//! - `7` remesh: `cx, cy, cz, radius, target_length (NaN for the mesh mean),
//!   max_rounds (NaN for the default), split_only (0 or 1)`.
//! - `16` split stretched: `cx, cy, cz, radius, limit (NaN for the default),
//!   max_splits (NaN for the default), smooth (0 or 1), sharp_angle in degrees
//!   (NaN for the default)`; see
//!   [`fast_mesh_ops::stretch`].
//! - `17` capture rest sizes: `only_if_missing (0 or 1)`. Must be the only
//!   operation of its list, because it is not an edit.
//! - `18` capture stroke state: `only_if_missing (0 or 1)`. Must be the only
//!   operation of its list, because it is not an edit.
//! - `19` flatten: `cx, cy, cz, radius, falloff, strength, mode (0 both,
//!   1 peaks, 2 valleys), has_origin (0/1), ox, oy, oz, has_normal (0/1),
//!   nx, ny, nz`.
//! - `20` crease: `cx, cy, cz, radius, falloff, strength, pinch (NaN for 0.5)`.
//! - `21` clay: `cx, cy, cz, radius, falloff, strength, profile (0 round,
//!   1 strips), has_tangent (0/1), tx, ty, tz`.
//! - `22` twist: `cx, cy, cz, radius, falloff, strength, method (0 vortex,
//!   1 kelvinlet), has_axis (0/1), ax, ay, az`.
//! - `8` bridge loops: `max_edges (NaN for the default)`; exactly two loop
//!   ids are taken from `loops`.
//! - `9` grab: `cx, cy, cz, radius, falloff (0 linear, 1 smooth), strength,
//!   dx, dy, dz, method (NaN/0 sphere, 1 geodesic, 2 kelvinlet),
//!   poisson (NaN for 0.4), from_stroke (0/1), adaptive_reach (0/1)`.
//! - `10` relax: `cx, cy, cz, radius, falloff, strength, cycles` (plain
//!   Laplacian passes).
//! - `11` offset: `distance, algorithm (0 normal, 1 fhEdt, 2 bvhSdf; NaN for
//!   the default), voxel_size (NaN for automatic), max_voxels (NaN for the
//!   default)`.
//! - `12` shell: `thickness`.
//! - `13` simplify: `target_triangles, preserve_boundary (0 or 1)`.
//! - `14` subdivide: `method (0 loop, NaN for the default), levels (NaN
//!   for 1)`. Each level splits every triangle into four by Loop's scheme;
//!   see [`fast_mesh_ops::subdivide`].
//! - `15` auto repair: `weld_tolerance, min_triangles, max_edges (Infinity
//!   for any size), fair`, NaN meaning the default of
//!   [`AutoRepairOptions`]. Its report is, for each stage in the order it
//!   ran, the code of the op that stage is, then that op's report.
//! - `23` make manifold: `remove_duplicates, prune_fins,
//!   split_non_manifold_edges, split_pinch_vertices` (0/1 or NaN for true).
//! - `24` resolve self intersections: `mode (0 none, 1/NaN deleteAndFill,
//!   2 cutAndStitch, 3 voxelHeal), ring_expansion, voxel_size, max_voxels,
//!   max_hole_edges (NaN/Infinity for None), fair, stitch_tolerance`.
//! - `25` cleanup slivers: `max_aspect_ratio, min_angle_deg, max_angle_deg,
//!   max_iterations`.
//! - `26` preset auto repair: `preset (0/NaN general, 1 scan3d, 2 print3d,
//!   3 cadImport), self_intersection_mode (NaN preset default, 0..=3),
//!   weld_tolerance, min_triangles, max_edges (NaN preset default, Infinity
//!   unbounded, or integer), fair, voxel_size, max_voxels`.
//!
//! - `32` protect: `n, band (0.0 when unset), id × n, weight × n`, wrapping a
//!   brush op (`5, 6, 9, 10, 19, 20, 21, 22`), composable in either order with
//!   `33`. Runs under [`with_protection_band`](fast_mesh_ops::protect::with_protection_band)
//!   and reports as usual; `32` adds no report of its own.
//! - `33` brush modifiers: `sym_mask (0..=7 bitmask: x=1, y=2, z=4),
//!   filter_kind (0 none, 1 peaksOnly, 2 valleysOnly, 3 normalAngle), p0, p1,
//!   has_vec (0/1), vx, vy, vz`, wrapping a brush op (`5, 6, 9, 10, 19, 20,
//!   21, 22`) and composable in either order with `32`.
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
//!   × filled_n, skipped_n, (loop id, reason (0 bounds, 1 unclean, 2 UV seam)) ×
//!   skipped_n, failed_n, (loop id, reason (0 projection folds, 1 no valid
//!   triangulation, 2 solver)) × failed_n`.
//! - inflate: `moved, max_displacement`.
//! - smooth: `moved, max_displacement`.
//! - remesh: `splits, collapses, flips, rounds`.
//! - bridge loops: `triangles_added`.
//! - grab: `moved, max_displacement`.
//! - relax: `moved, max_displacement`.
//! - flatten: `moved, max_displacement`.
//! - crease: `moved, max_displacement`.
//! - clay: `moved, max_displacement`.
//! - twist: `moved, max_displacement`.
//! - offset: `moved, max_displacement, degenerate_removed, vertices_removed,
//!   vertices_added, triangles_added, triangles_removed, voxel_size,
//!   uvs_dropped`.
//! - shell: `vertices_added, triangles_added, loops_closed, degenerate_removed,
//!   vertices_removed`.
//! - simplify: `collapsed, triangles_removed, vertices_removed`.
//! - subdivide: `vertices_added, triangles_added`.
//! - split stretched: `splits, remaining, (new vertex, a, b) x splits`.
//! - make manifold: `removed_duplicate_triangles, pruned_fin_triangles,
//!   split_non_manifold_edges, split_pinch_vertices, vertices_added,
//!   removed_vertices`.
//! - resolve self intersections: `initial_pairs, remaining_pairs,
//!   removed_triangles, added_triangles, mode_used (0..=3), uvs_dropped`.
//! - cleanup slivers: `collapsed_needles, flipped_caps, removed_degenerate,
//!   removed_vertices`.
//! - capture rest sizes: nothing.
//! - capture stroke state: nothing.
//!
//! # Analysis
//!
//! One `f64` array, see [`encode_analysis`].

use fast_mesh_ops::bridge::BridgeOptions;
use fast_mesh_ops::protect::with_protection_band;
use fast_mesh_ops::remesh::{REMESH_MAX_ROUNDS, Region, RemeshOptions};
use fast_mesh_ops::sculpt::{
    Brush, ClayOptions, ClayProfile, CreaseOptions, Falloff, FlattenMode, FlattenOptions,
    GrabMethod, GrabOptions, InflateMethod, InflateOptions, SmoothMethod, SmoothOptions,
    SurfaceFilter, Symmetry, TwistMethod, TwistOptions, clay, crease, flatten, grab, inflate,
    smooth, twist,
};
use fast_mesh_ops::simplify::SimplifyOptions;
use fast_mesh_ops::stretch::{StretchOptions, StretchReport};
use fast_mesh_ops::subdivide::{SubdivideMethod, SubdivideOptions};
use fast_mesh_ops::{
    Analysis, AutoRepairOptions, AutoRepairPreset, AutoRepairStage, ComponentOptions,
    DegenerateOptions, FailReason, FairWeights, FillOptions, FillReport, LoopId, ManifoldOptions,
    OffsetAlgorithm, OffsetOptions, OpsError, SelfIntersectOptions, SelfIntersectionMode,
    ShellOptions, SkipReason, SliverOptions,
};
use fast_mesh_store::{StoreError, Transaction, UV_CHANNEL};

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
    SetUvs,
}

impl Kind {
    fn from_code(code: u32) -> Result<Kind, Error> {
        Ok(match code {
            0 => Kind::SetPositions,
            1 => Kind::AddVertices,
            2 => Kind::AddTriangles,
            3 => Kind::RemoveTriangles,
            4 => Kind::RemoveVertices,
            5 => Kind::SetUvs,
            _ => return Err(Error::invalid("unknown edit op")),
        })
    }

    /// What the op's id list holds; `None` when it has none.
    fn takes(self) -> Option<Element> {
        match self {
            Kind::SetPositions | Kind::AddTriangles | Kind::RemoveVertices | Kind::SetUvs => {
                Some(Element::Vertex)
            }
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
    /// Flat `[u, v, ...]`, for set UVs.
    uvs: Vec<f32>,
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
        let mut uvs = Vec::new();
        if kind == Kind::SetUvs {
            let flat = vals.take(n.checked_mul(2).ok_or_else(|| malformed(WHAT))?)?;
            uvs.try_reserve_exact(flat.len()).map_err(|_| oom())?;
            uvs.extend(flat.iter().map(|&x| x as f32));
            if !uvs.iter().all(|x| x.is_finite()) {
                return Err(Error::invalid("uvs must be finite"));
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
            uvs,
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
            Kind::SetUvs => {
                tx.write_channel(UV_CHANNEL, &ids, &op.uvs)?;
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
    Grab(GrabOptions),
    Flatten(FlattenOptions),
    Crease(CreaseOptions),
    Clay(ClayOptions),
    Twist(TwistOptions),
    Offset(OffsetOptions),
    Shell(f32),
    Simplify(SimplifyOptions),
    Subdivide(SubdivideOptions),
    Auto(AutoRepairOptions),
    MakeManifold(ManifoldOptions),
    ResolveSelfIntersections(SelfIntersectOptions),
    CleanupSlivers(SliverOptions),
    AutoPreset(AutoRepairOptions),
    SplitStretched(StretchOptions),
    /// Not an edit: `WasmScene::repair` runs it outside the transaction.
    CaptureRest {
        only_if_missing: bool,
    },
    /// Not an edit: `WasmScene::repair` runs it outside the transaction.
    CaptureStroke {
        only_if_missing: bool,
    },
    /// A brush op whose moves are scaled per vertex.
    Protected {
        vertices: Vec<u32>,
        weights: Vec<f32>,
        band: f32,
        op: Box<RepairOp>,
    },
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

fn finite_vec3(raw: [f64; 3], what: &str) -> Result<[f32; 3], Error> {
    let v = [raw[0] as f32, raw[1] as f32, raw[2] as f32];
    if v.iter().any(|x| !x.is_finite()) {
        return Err(Error::invalid(&format!("{what} must be finite")));
    }
    Ok(v)
}

fn nonzero_vec3(raw: [f64; 3], what: &str) -> Result<[f32; 3], Error> {
    let v = finite_vec3(raw, what)?;
    let len = (f64::from(v[0]).powi(2) + f64::from(v[1]).powi(2) + f64::from(v[2]).powi(2)).sqrt();
    if !(len > 0.0 && len.is_finite()) {
        return Err(Error::invalid(&format!("{what} must be non-zero")));
    }
    Ok(v)
}

/// A count, or `None` for NaN.
pub(crate) fn count_opt(v: f64, what: &str) -> Result<Option<usize>, Error> {
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

fn opt_flag(v: f64, default: bool) -> Result<bool, Error> {
    if v.is_nan() { Ok(default) } else { flag(v) }
}

fn opt_voxel_size(voxel_size: f64) -> Result<Option<f32>, Error> {
    if voxel_size.is_nan() {
        Ok(None)
    } else if voxel_size.is_finite() && voxel_size > 0.0 {
        let h = voxel_size as f32;
        if !h.is_finite() || h <= 0.0 {
            return Err(Error::invalid("voxelSize must be positive and finite"));
        }
        Ok(Some(h))
    } else {
        Err(Error::invalid("voxelSize must be positive and finite"))
    }
}

fn opt_max_voxels(max_voxels: f64, default: usize) -> Result<usize, Error> {
    // Reject values past JS `Number.MAX_SAFE_INTEGER` (2^53 - 1); on wasm32
    // `count_opt`'s `v as usize` saturates safely at `usize::MAX`.
    if !max_voxels.is_nan() && max_voxels > 9_007_199_254_740_991.0 {
        return Err(Error::invalid("maxVoxels is too large"));
    }
    match count_opt(max_voxels, "maxVoxels")? {
        None => Ok(default),
        Some(n) if n < 64 => Err(Error::invalid("maxVoxels must be at least 64")),
        Some(n) => Ok(n),
    }
}

fn opt_u32_count(v: f64, what: &str, default: u32) -> Result<u32, Error> {
    if !v.is_nan() && v > f64::from(u32::MAX) {
        return Err(Error::invalid(&format!("{what} is too large")));
    }
    Ok(count_opt(v, what)?.unwrap_or(default as usize) as u32)
}

fn decode_si_mode(v: f64, allow_nan_unset: bool) -> Result<Option<SelfIntersectionMode>, Error> {
    if v.is_nan() {
        return Ok(if allow_nan_unset {
            None
        } else {
            Some(SelfIntersectionMode::DeleteAndFill)
        });
    }
    if v == 0.0 {
        Ok(Some(SelfIntersectionMode::None))
    } else if v == 1.0 {
        Ok(Some(SelfIntersectionMode::DeleteAndFill))
    } else if v == 2.0 {
        Ok(Some(SelfIntersectionMode::CutAndStitch))
    } else if v == 3.0 {
        Ok(Some(SelfIntersectionMode::VoxelHeal))
    } else {
        Err(Error::invalid(
            "selfIntersectionMode must be 0 (none), 1 (deleteAndFill), 2 (cutAndStitch) or 3 (voxelHeal)",
        ))
    }
}

fn si_mode_code(mode: SelfIntersectionMode) -> f64 {
    match mode {
        SelfIntersectionMode::None => 0.0,
        SelfIntersectionMode::DeleteAndFill => 1.0,
        SelfIntersectionMode::CutAndStitch => 2.0,
        SelfIntersectionMode::VoxelHeal => 3.0,
    }
}

pub(crate) fn loop_id(w: [u32; 4]) -> LoopId {
    LoopId {
        min_vertex: w[0],
        len: w[1],
        hash: u64::from(w[2]) | (u64::from(w[3]) << 32),
    }
}

/// Decodes a protection's `n, band, id × n, weight × n`; the range of a weight
/// and the liveness of a vertex are checked when it runs.
fn decode_protect(words: &mut Cursor<'_, f64>) -> Result<(Vec<u32>, Vec<f32>, f32), Error> {
    let n = count_opt(words.next()?, "protect count")?.ok_or_else(|| malformed("repair"))?;
    let band_raw = words.next()?;
    let band = if band_raw.is_nan() {
        0.0
    } else {
        let b = band_raw as f32;
        if !b.is_finite() || b < 0.0 {
            return Err(Error::invalid(
                "protect band must be finite and non-negative",
            ));
        }
        b
    };
    let ids = words.take(n)?;
    let raw_weights = words.take(n)?;
    let mut vertices = reserved(n)?;
    for &id in ids {
        if !(id >= 0.0 && id <= f64::from(u32::MAX) && id.fract() == 0.0) {
            return Err(Error::invalid("protect vertices must be u32 integers"));
        }
        vertices.push(id as u32);
    }
    let mut weights = reserved(n)?;
    weights.extend(raw_weights.iter().map(|&w| w as f32));
    Ok((vertices, weights, band))
}

/// Decodes brush modifiers `33`:
/// `sym_mask (0..=7), filter_kind (0..=3), p0, p1, has_vec (0/1), vx, vy, vz`.
fn decode_brush_modifiers(words: &mut Cursor<'_, f64>) -> Result<(Symmetry, SurfaceFilter), Error> {
    let [sym_mask, filter_kind, p0, p1, has_vec, vx, vy, vz]: [f64; 8] =
        words.take(8)?.try_into().map_err(|_| malformed("repair"))?;
    let mask = count_opt(sym_mask, "symmetry mask")?.ok_or_else(|| malformed("repair"))?;
    if mask > 7 {
        return Err(Error::invalid("symmetry mask must be in 0..=7"));
    }
    let symmetry = Symmetry {
        x: (mask & 1) != 0,
        y: (mask & 2) != 0,
        z: (mask & 4) != 0,
    };
    let has_vec = flag(has_vec)?;
    let filter = if filter_kind == 0.0 {
        if has_vec {
            return Err(malformed("repair"));
        }
        SurfaceFilter::None
    } else if filter_kind == 1.0 || filter_kind == 2.0 {
        if has_vec {
            return Err(malformed("repair"));
        }
        let threshold = if p0.is_nan() { 0.0 } else { p0 as f32 };
        if !(0.0..=1.0).contains(&threshold) {
            return Err(Error::invalid(
                "surface filter threshold must be finite and in [0, 1]",
            ));
        }
        let contrast = if p1.is_nan() { 0.5 } else { p1 as f32 };
        if !(0.0..=1.0).contains(&contrast) {
            return Err(Error::invalid(
                "surface filter contrast must be finite and in [0, 1]",
            ));
        }
        if filter_kind == 1.0 {
            SurfaceFilter::PeaksOnly {
                threshold,
                contrast,
            }
        } else {
            SurfaceFilter::ValleysOnly {
                threshold,
                contrast,
            }
        }
    } else if filter_kind == 3.0 {
        let max_angle_deg = if p0.is_nan() { 45.0 } else { p0 as f32 };
        if !(max_angle_deg > 0.0 && max_angle_deg <= 180.0) {
            return Err(Error::invalid(
                "surface filter maxAngleDeg must be finite and in (0, 180]",
            ));
        }
        let normal = if has_vec {
            Some(nonzero_vec3([vx, vy, vz], "surface filter normal")?)
        } else {
            None
        };
        SurfaceFilter::NormalAngle {
            normal,
            max_angle_deg,
        }
    } else {
        return Err(Error::invalid("surface filter kind must be 0..=3"));
    };
    Ok((symmetry, filter))
}

fn is_brush_op(op: &RepairOp) -> bool {
    matches!(
        op,
        RepairOp::Inflate(_)
            | RepairOp::Smooth(_)
            | RepairOp::Grab(_)
            | RepairOp::Flatten(_)
            | RepairOp::Crease(_)
            | RepairOp::Clay(_)
            | RepairOp::Twist(_)
    )
}

fn apply_brush_modifiers(
    op: &mut RepairOp,
    symmetry: Symmetry,
    filter: SurfaceFilter,
) -> Result<(), Error> {
    match op {
        RepairOp::Inflate(o) => {
            o.symmetry = symmetry;
            o.filter = filter;
        }
        RepairOp::Smooth(o) => {
            o.symmetry = symmetry;
            o.filter = filter;
        }
        RepairOp::Grab(o) => {
            o.symmetry = symmetry;
            o.filter = filter;
        }
        RepairOp::Flatten(o) => {
            o.symmetry = symmetry;
            o.filter = filter;
        }
        RepairOp::Crease(o) => {
            o.symmetry = symmetry;
            o.filter = filter;
        }
        RepairOp::Clay(o) => {
            o.symmetry = symmetry;
            o.filter = filter;
        }
        RepairOp::Twist(o) => {
            o.symmetry = symmetry;
            o.filter = filter;
        }
        _ => {
            return Err(Error::invalid(
                "brushModifiers applies only to brush operations",
            ));
        }
    }
    Ok(())
}

fn decode_cycles(cycles: f64) -> Result<u32, Error> {
    // Before `count_opt`: `v as usize` saturates at `usize::MAX`,
    // which on wasm32 equals `u32::MAX`, so without this check any
    // larger value would decode to billions of cycles.
    if !cycles.is_nan() && cycles > f64::from(u32::MAX) {
        return Err(Error::invalid("cycles is too large"));
    }
    let cycles = count_opt(cycles, "cycles")?.unwrap_or_default();
    u32::try_from(cycles).map_err(|_| Error::invalid("cycles is too large"))
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
    // Prefix wrappers (`32` protect and `33` brushModifiers), composable in either order.
    let mut protect: Option<(Vec<u32>, Vec<f32>, f32)> = None;
    let mut modifiers: Option<(Symmetry, SurfaceFilter)> = None;
    while !words.is_empty() {
        let code = words.next()?;
        if code == 32.0 {
            if protect.is_some() {
                return Err(malformed(WHAT));
            }
            protect = Some(decode_protect(&mut words)?);
            continue;
        }
        if code == 33.0 {
            if modifiers.is_some() {
                return Err(malformed(WHAT));
            }
            modifiers = Some(decode_brush_modifiers(&mut words)?);
            continue;
        }
        let mut op = if code == 0.0 {
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
            let [cx, cy, cz, radius, fo, strength, method]: [f64; 7] =
                words.take(7)?.try_into().map_err(|_| malformed(WHAT))?;
            let method = if method.is_nan() || method == 0.0 {
                InflateMethod::VertexNormal
            } else if method == 1.0 {
                InflateMethod::AverageNormal
            } else if method == 2.0 {
                InflateMethod::StrokeNormal
            } else {
                return Err(Error::invalid("inflate method must be 0..=2"));
            };
            RepairOp::Inflate(InflateOptions {
                brush: Brush {
                    center: [cx as f32, cy as f32, cz as f32],
                    radius: radius as f32,
                    falloff: falloff(fo)?,
                    strength: strength as f32,
                },
                method,
                ..Default::default()
            })
        } else if code == 6.0 {
            let [cx, cy, cz, radius, fo, strength, cycles, method]: [f64; 8] =
                words.take(8)?.try_into().map_err(|_| malformed(WHAT))?;
            let method = if method.is_nan() || method == 0.0 {
                SmoothMethod::Taubin
            } else if method == 1.0 {
                SmoothMethod::Laplacian
            } else if method == 2.0 {
                SmoothMethod::HcLaplacian
            } else if method == 3.0 {
                SmoothMethod::Cotangent
            } else {
                return Err(Error::invalid("smooth method must be 0..=3"));
            };
            RepairOp::Smooth(SmoothOptions {
                brush: Brush {
                    center: [cx as f32, cy as f32, cz as f32],
                    radius: radius as f32,
                    falloff: falloff(fo)?,
                    strength: strength as f32,
                },
                cycles: decode_cycles(cycles)?,
                method,
                ..Default::default()
            })
        } else if code == 10.0 {
            let [cx, cy, cz, radius, fo, strength, cycles]: [f64; 7] =
                words.take(7)?.try_into().map_err(|_| malformed(WHAT))?;
            RepairOp::Smooth(SmoothOptions {
                brush: Brush {
                    center: [cx as f32, cy as f32, cz as f32],
                    radius: radius as f32,
                    falloff: falloff(fo)?,
                    strength: strength as f32,
                },
                cycles: decode_cycles(cycles)?,
                method: SmoothMethod::Laplacian,
                ..Default::default()
            })
        } else if code == 7.0 {
            let [cx, cy, cz, radius, target, rounds, split]: [f64; 7] =
                words.take(7)?.try_into().map_err(|_| malformed(WHAT))?;
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
                split_only: flag(split)?,
            })
        } else if code == 8.0 {
            let [max_e]: [f64; 1] = words.take(1)?.try_into().map_err(|_| malformed(WHAT))?;
            let flat = ids.take(8)?;
            let pair = flat.as_chunks::<4>().0;
            RepairOp::Bridge(BridgeOptions {
                loops: (loop_id(pair[0]), loop_id(pair[1])),
                max_edges: count_opt(max_e, "maxEdges")?,
            })
        } else if code == 9.0 {
            let [
                cx,
                cy,
                cz,
                radius,
                fo,
                strength,
                dx,
                dy,
                dz,
                method,
                poisson,
                from_stroke,
                adaptive_reach,
            ]: [f64; 13] = words.take(13)?.try_into().map_err(|_| malformed(WHAT))?;
            let method = if method.is_nan() || method == 0.0 {
                GrabMethod::Sphere
            } else if method == 1.0 {
                GrabMethod::Geodesic
            } else if method == 2.0 {
                GrabMethod::Kelvinlet
            } else {
                return Err(Error::invalid("grab method must be 0..=2"));
            };
            let poisson = if poisson.is_nan() {
                0.4
            } else {
                let p = poisson as f32;
                if !(0.0..=0.5).contains(&p) {
                    return Err(Error::invalid(
                        "grab poisson must be finite and in [0, 0.5]",
                    ));
                }
                p
            };
            RepairOp::Grab(GrabOptions {
                brush: Brush {
                    center: [cx as f32, cy as f32, cz as f32],
                    radius: radius as f32,
                    falloff: falloff(fo)?,
                    strength: strength as f32,
                },
                direction: [dx as f32, dy as f32, dz as f32],
                method,
                poisson,
                from_stroke: flag(from_stroke)?,
                adaptive_reach: flag(adaptive_reach)?,
                ..Default::default()
            })
        } else if code == 11.0 {
            let [distance, alg, voxel_size, max_voxels]: [f64; 4] =
                words.take(4)?.try_into().map_err(|_| malformed(WHAT))?;
            let voxel_size = opt_voxel_size(voxel_size)?;
            let max_voxels = opt_max_voxels(max_voxels, OffsetOptions::default().max_voxels)?;
            let algorithm = if alg.is_nan() || alg == 0.0 {
                OffsetAlgorithm::Normal
            } else if alg == 1.0 {
                OffsetAlgorithm::FhEdt { voxel_size }
            } else if alg == 2.0 {
                OffsetAlgorithm::BvhSdf { voxel_size }
            } else {
                return Err(Error::invalid(
                    "algorithm must be 0 (normal), 1 (fhEdt) or 2 (bvhSdf)",
                ));
            };
            RepairOp::Offset(OffsetOptions {
                distance: distance as f32,
                algorithm,
                max_voxels,
            })
        } else if code == 12.0 {
            RepairOp::Shell(words.next()? as f32)
        } else if code == 13.0 {
            let [target, preserve]: [f64; 2] =
                words.take(2)?.try_into().map_err(|_| malformed(WHAT))?;
            let target = count_opt(target, "targetTriangles")?
                .ok_or_else(|| Error::invalid("targetTriangles is required"))?;
            if target == 0 {
                return Err(Error::invalid("targetTriangles must be at least 1"));
            }
            RepairOp::Simplify(SimplifyOptions {
                target_triangles: target,
                preserve_boundary: flag(preserve)?,
            })
        } else if code == 14.0 {
            let [method, levels]: [f64; 2] =
                words.take(2)?.try_into().map_err(|_| malformed(WHAT))?;
            let method = if method.is_nan() || method == 0.0 {
                SubdivideMethod::Loop
            } else {
                return Err(Error::invalid("method must be 0 (loop)"));
            };
            // Before `count_opt`: `v as usize` saturates at `usize::MAX`,
            // which on wasm32 equals `u32::MAX`, so without this check any
            // larger value would decode to billions of levels.
            if !levels.is_nan() && levels > f64::from(u32::MAX) {
                return Err(Error::invalid("levels is too large"));
            }
            let levels = count_opt(levels, "levels")?.unwrap_or(1);
            let levels =
                u32::try_from(levels).map_err(|_| Error::invalid("levels is too large"))?;
            if levels == 0 {
                return Err(Error::invalid("levels must be at least 1"));
            }
            RepairOp::Subdivide(SubdivideOptions { method, levels })
        } else if code == 15.0 {
            let [weld, min_t, max_e, fair]: [f64; 4] =
                words.take(4)?.try_into().map_err(|_| malformed(WHAT))?;
            let mut o = AutoRepairOptions::default();
            if let Some(weld) = bound(weld) {
                o.weld_tolerance = weld as f32;
            }
            if let Some(min_t) = count_opt(min_t, "minTriangles")? {
                o.min_triangles = min_t;
            }
            if max_e == f64::INFINITY {
                o.max_edges = None;
            } else if let Some(max_e) = count_opt(max_e, "maxEdges")? {
                o.max_edges = Some(max_e);
            }
            if !fair.is_nan() {
                o.fair = flag(fair)?;
            }
            RepairOp::Auto(o)
        } else if code == 16.0 {
            let [cx, cy, cz, radius, limit, max_splits, smooth, sharp_angle]: [f64; 8] =
                words.take(8)?.try_into().map_err(|_| malformed(WHAT))?;
            let mut o = StretchOptions {
                region: Region {
                    center: [cx as f32, cy as f32, cz as f32],
                    radius: radius as f32,
                },
                smooth: flag(smooth)?,
                ..StretchOptions::default()
            };
            if let Some(limit) = bound(limit) {
                o.limit = limit as f32;
            }
            if let Some(max_splits) = count_opt(max_splits, "maxSplits")? {
                o.max_splits = u32::try_from(max_splits).unwrap_or(u32::MAX);
            }
            if !sharp_angle.is_nan() {
                // Out of range (an infinity too) fails in `split_stretched`.
                o.sharp_angle = sharp_angle as f32;
            }
            RepairOp::SplitStretched(o)
        } else if code == 17.0 {
            let [only_if_missing]: [f64; 1] =
                words.take(1)?.try_into().map_err(|_| malformed(WHAT))?;
            RepairOp::CaptureRest {
                only_if_missing: flag(only_if_missing)?,
            }
        } else if code == 18.0 {
            let [only_if_missing]: [f64; 1] =
                words.take(1)?.try_into().map_err(|_| malformed(WHAT))?;
            RepairOp::CaptureStroke {
                only_if_missing: flag(only_if_missing)?,
            }
        } else if code == 19.0 {
            let [
                cx,
                cy,
                cz,
                radius,
                fo,
                strength,
                mode,
                has_origin,
                ox,
                oy,
                oz,
                has_normal,
                nx,
                ny,
                nz,
            ]: [f64; 15] = words.take(15)?.try_into().map_err(|_| malformed(WHAT))?;
            let mode = if mode.is_nan() || mode == 0.0 {
                FlattenMode::Both
            } else if mode == 1.0 {
                FlattenMode::Peaks
            } else if mode == 2.0 {
                FlattenMode::Valleys
            } else {
                return Err(Error::invalid("flatten mode must be 0..=2"));
            };
            let plane_origin = if flag(has_origin)? {
                Some(finite_vec3([ox, oy, oz], "flatten planeOrigin")?)
            } else {
                None
            };
            let plane_normal = if flag(has_normal)? {
                Some(nonzero_vec3([nx, ny, nz], "flatten planeNormal")?)
            } else {
                None
            };
            RepairOp::Flatten(FlattenOptions {
                brush: Brush {
                    center: [cx as f32, cy as f32, cz as f32],
                    radius: radius as f32,
                    falloff: falloff(fo)?,
                    strength: strength as f32,
                },
                mode,
                plane_origin,
                plane_normal,
                ..Default::default()
            })
        } else if code == 20.0 {
            let [cx, cy, cz, radius, fo, strength, pinch]: [f64; 7] =
                words.take(7)?.try_into().map_err(|_| malformed(WHAT))?;
            let pinch = if pinch.is_nan() {
                0.5
            } else {
                let p = pinch as f32;
                if !(0.0..=1.0).contains(&p) {
                    return Err(Error::invalid("crease pinch must be finite and in [0, 1]"));
                }
                p
            };
            RepairOp::Crease(CreaseOptions {
                brush: Brush {
                    center: [cx as f32, cy as f32, cz as f32],
                    radius: radius as f32,
                    falloff: falloff(fo)?,
                    strength: strength as f32,
                },
                pinch,
                ..Default::default()
            })
        } else if code == 21.0 {
            let [
                cx,
                cy,
                cz,
                radius,
                fo,
                strength,
                profile,
                has_tangent,
                tx,
                ty,
                tz,
            ]: [f64; 11] = words.take(11)?.try_into().map_err(|_| malformed(WHAT))?;
            let profile = if profile.is_nan() || profile == 0.0 {
                ClayProfile::Round
            } else if profile == 1.0 {
                ClayProfile::Strips
            } else {
                return Err(Error::invalid("clay profile must be 0..=1"));
            };
            let tangent = if flag(has_tangent)? {
                Some(nonzero_vec3([tx, ty, tz], "clay tangent")?)
            } else {
                None
            };
            RepairOp::Clay(ClayOptions {
                brush: Brush {
                    center: [cx as f32, cy as f32, cz as f32],
                    radius: radius as f32,
                    falloff: falloff(fo)?,
                    strength: strength as f32,
                },
                profile,
                tangent,
                ..Default::default()
            })
        } else if code == 22.0 {
            let [
                cx,
                cy,
                cz,
                radius,
                fo,
                strength,
                method,
                has_axis,
                ax,
                ay,
                az,
            ]: [f64; 11] = words.take(11)?.try_into().map_err(|_| malformed(WHAT))?;
            let method = if method.is_nan() || method == 0.0 {
                TwistMethod::Vortex
            } else if method == 1.0 {
                TwistMethod::Kelvinlet
            } else {
                return Err(Error::invalid("twist method must be 0..=1"));
            };
            let axis = if flag(has_axis)? {
                Some(nonzero_vec3([ax, ay, az], "twist axis")?)
            } else {
                None
            };
            RepairOp::Twist(TwistOptions {
                brush: Brush {
                    center: [cx as f32, cy as f32, cz as f32],
                    radius: radius as f32,
                    falloff: falloff(fo)?,
                    strength: strength as f32,
                },
                axis,
                method,
                ..Default::default()
            })
        } else if code == 23.0 {
            let [rm_dup, prune_fins, split_edges, split_pinch]: [f64; 4] =
                words.take(4)?.try_into().map_err(|_| malformed(WHAT))?;
            let defaults = ManifoldOptions::default();
            RepairOp::MakeManifold(ManifoldOptions {
                remove_duplicates: opt_flag(rm_dup, defaults.remove_duplicates)?,
                prune_fins: opt_flag(prune_fins, defaults.prune_fins)?,
                split_non_manifold_edges: opt_flag(split_edges, defaults.split_non_manifold_edges)?,
                split_pinch_vertices: opt_flag(split_pinch, defaults.split_pinch_vertices)?,
            })
        } else if code == 24.0 {
            let [
                mode,
                ring,
                voxel_size,
                max_voxels,
                max_hole_edges,
                fair,
                stitch,
            ]: [f64; 7] = words.take(7)?.try_into().map_err(|_| malformed(WHAT))?;
            let defaults = SelfIntersectOptions::default();
            let mode = decode_si_mode(mode, false)?.unwrap_or(defaults.mode);
            let ring_expansion = opt_u32_count(ring, "ringExpansion", defaults.ring_expansion)?;
            let voxel_size = opt_voxel_size(voxel_size)?;
            let max_voxels = opt_max_voxels(max_voxels, defaults.max_voxels)?;
            let max_hole_edges = if max_hole_edges == f64::INFINITY || max_hole_edges.is_nan() {
                None
            } else {
                count_opt(max_hole_edges, "maxHoleEdges")?
            };
            let fair = opt_flag(fair, defaults.fair)?;
            let stitch_tolerance = if stitch.is_nan() {
                defaults.stitch_tolerance
            } else {
                let t = stitch as f32;
                if !t.is_finite() || t < 0.0 {
                    return Err(Error::invalid(
                        "stitchTolerance must be finite and non-negative",
                    ));
                }
                t
            };
            RepairOp::ResolveSelfIntersections(SelfIntersectOptions {
                mode,
                ring_expansion,
                voxel_size,
                max_voxels,
                max_hole_edges,
                fair,
                stitch_tolerance,
            })
        } else if code == 25.0 {
            let [max_aspect, min_angle, max_angle, max_iters]: [f64; 4] =
                words.take(4)?.try_into().map_err(|_| malformed(WHAT))?;
            let defaults = SliverOptions::default();
            let max_aspect_ratio = if max_aspect.is_nan() {
                defaults.max_aspect_ratio
            } else {
                let a = max_aspect as f32;
                if !a.is_finite() || a < 1.0 {
                    return Err(Error::invalid(
                        "maxAspectRatio must be finite and at least 1",
                    ));
                }
                a
            };
            let min_angle_deg = if min_angle.is_nan() {
                defaults.min_angle_deg
            } else {
                let a = min_angle as f32;
                if !a.is_finite() || !(0.0..=60.0).contains(&a) {
                    return Err(Error::invalid("minAngleDeg must be finite and in [0, 60]"));
                }
                a
            };
            let max_angle_deg = if max_angle.is_nan() {
                defaults.max_angle_deg
            } else {
                let a = max_angle as f32;
                if !a.is_finite() || !(60.0..=180.0).contains(&a) {
                    return Err(Error::invalid(
                        "maxAngleDeg must be finite and in [60, 180]",
                    ));
                }
                a
            };
            if min_angle_deg > max_angle_deg {
                return Err(Error::invalid("minAngleDeg must not exceed maxAngleDeg"));
            }
            let max_iterations =
                opt_u32_count(max_iters, "maxIterations", defaults.max_iterations)?;
            RepairOp::CleanupSlivers(SliverOptions {
                max_aspect_ratio,
                min_angle_deg,
                max_angle_deg,
                max_iterations,
            })
        } else if code == 26.0 {
            let [
                preset,
                si_mode,
                weld,
                min_t,
                max_e,
                fair,
                voxel_size,
                max_voxels,
            ]: [f64; 8] = words.take(8)?.try_into().map_err(|_| malformed(WHAT))?;
            let preset = if preset.is_nan() || preset == 0.0 {
                AutoRepairPreset::General
            } else if preset == 1.0 {
                AutoRepairPreset::Scan3D
            } else if preset == 2.0 {
                AutoRepairPreset::Print3D
            } else if preset == 3.0 {
                AutoRepairPreset::CadImport
            } else {
                return Err(Error::invalid(
                    "preset must be 0 (general), 1 (scan3d), 2 (print3d) or 3 (cadImport)",
                ));
            };
            let mut b = AutoRepairOptions::builder().preset(preset);
            if let Some(m) = decode_si_mode(si_mode, true)? {
                b = b.self_intersection_mode(Some(m));
            }
            if let Some(weld) = bound(weld) {
                b = b.weld_tolerance(weld as f32);
            }
            if let Some(min_t) = count_opt(min_t, "minTriangles")? {
                b = b.min_triangles(min_t);
            }
            if max_e == f64::INFINITY {
                b = b.max_edges(None);
            } else if let Some(max_e) = count_opt(max_e, "maxEdges")? {
                b = b.max_edges(Some(max_e));
            }
            if !fair.is_nan() {
                b = b.fair(flag(fair)?);
            }
            if let Some(h) = opt_voxel_size(voxel_size)? {
                b = b.voxel_size(Some(h));
            }
            if !max_voxels.is_nan() {
                let default_mv = AutoRepairOptions::default().max_voxels;
                b = b.max_voxels(opt_max_voxels(max_voxels, default_mv)?);
            }
            RepairOp::AutoPreset(b.build())
        } else {
            return Err(Error::invalid("unknown repair op"));
        };
        if let Some((symmetry, filter)) = modifiers.take() {
            apply_brush_modifiers(&mut op, symmetry, filter)?;
        }
        out.try_reserve(1).map_err(|_| oom())?;
        out.push(match protect.take() {
            None => op,
            Some((vertices, weights, band)) => {
                if !is_brush_op(&op) {
                    return Err(Error::invalid("protect applies only to brush operations"));
                }
                RepairOp::Protected {
                    vertices,
                    weights,
                    band,
                    op: Box::new(op),
                }
            }
        });
    }
    if protect.is_some() || modifiers.is_some() {
        return Err(malformed(WHAT));
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
            SkipReason::UvSeam => 2.0,
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

fn auto_stage_code(stage: AutoRepairStage<'_>) -> f64 {
    match stage {
        AutoRepairStage::Weld(_) => 0.0,
        AutoRepairStage::Degenerate(_) => 1.0,
        AutoRepairStage::Components(_) => 2.0,
        AutoRepairStage::Orient(_) => 3.0,
        AutoRepairStage::Fill(_) => 4.0,
        AutoRepairStage::Manifold(_) => 23.0,
        AutoRepairStage::SelfIntersections(_) => 24.0,
        AutoRepairStage::Slivers(_) => 25.0,
    }
}

/// Appends the report of one of the ops an auto repair is made of (see the
/// module docs).
fn put_stage(out: &mut Vec<f64>, stage: AutoRepairStage) -> Result<(), StoreError> {
    match stage {
        AutoRepairStage::Weld(r) => {
            put(out, &[count(r.merged_vertices), count(r.removed_triangles)])
        }
        AutoRepairStage::Degenerate(r) => put(
            out,
            &[count(r.removed_triangles), count(r.removed_vertices)],
        ),
        AutoRepairStage::Components(r) => put(
            out,
            &[
                count(r.removed_components),
                count(r.removed_triangles),
                count(r.removed_vertices),
            ],
        ),
        AutoRepairStage::Orient(r) => put(
            out,
            &[
                count(r.flipped_triangles),
                count(r.patches),
                count(r.non_orientable_patches),
                count(r.flipped_outward_components),
            ],
        ),
        AutoRepairStage::Fill(r) => put_fill(out, r),
        AutoRepairStage::Manifold(r) => put(
            out,
            &[
                count(r.removed_duplicate_triangles),
                count(r.pruned_fin_triangles),
                count(r.split_non_manifold_edges),
                count(r.split_pinch_vertices),
                count(r.vertices_added),
                count(r.removed_vertices),
            ],
        ),
        AutoRepairStage::Slivers(r) => put(
            out,
            &[
                count(r.collapsed_needles),
                count(r.flipped_caps),
                count(r.removed_degenerate),
                count(r.removed_vertices),
            ],
        ),
        AutoRepairStage::SelfIntersections(r) => put(
            out,
            &[
                count(r.initial_pairs),
                count(r.remaining_pairs),
                count(r.removed_triangles),
                count(r.added_triangles),
                si_mode_code(r.mode_used),
                f64::from(u8::from(r.uvs_dropped)),
            ],
        ),
    }
}

pub(crate) fn put_stretch(out: &mut Vec<f64>, r: &StretchReport) -> Result<(), StoreError> {
    let words = r
        .vertices
        .len()
        .checked_mul(3)
        .and_then(|n| n.checked_add(2))
        .ok_or(StoreError::OutOfMemory)?;
    out.try_reserve(words)
        .map_err(|_| StoreError::OutOfMemory)?;
    out.extend_from_slice(&[count(r.splits), count(r.remaining)]);
    for [id, a, b] in &r.vertices {
        out.extend_from_slice(&[f64::from(*id), f64::from(*a), f64::from(*b)]);
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
                put_stage(&mut out, AutoRepairStage::Weld(&r))?;
            }
            RepairOp::Degenerate(o) => {
                let r = fast_mesh_ops::remove_degenerate(tx, o)?;
                put_stage(&mut out, AutoRepairStage::Degenerate(&r))?;
            }
            RepairOp::Components(o) => {
                let r = fast_mesh_ops::remove_small_components(tx, o)?;
                put_stage(&mut out, AutoRepairStage::Components(&r))?;
            }
            RepairOp::Orient => {
                let r = fast_mesh_ops::orient(tx)?;
                put_stage(&mut out, AutoRepairStage::Orient(&r))?;
            }
            RepairOp::Fill(o) => {
                let r = fast_mesh_ops::fill_holes(tx, o)?;
                put_stage(&mut out, AutoRepairStage::Fill(&r))?;
            }
            RepairOp::Auto(o) => {
                let r = fast_mesh_ops::auto_repair(tx, o)?;
                for stage in r.stages() {
                    put(&mut out, &[auto_stage_code(stage)])?;
                    put_stage(&mut out, stage)?;
                }
            }
            RepairOp::AutoPreset(o) => {
                let r = fast_mesh_ops::auto_repair(tx, o)?;
                for stage in r.executed_stages(o)? {
                    put(&mut out, &[auto_stage_code(stage)])?;
                    put_stage(&mut out, stage)?;
                }
            }
            RepairOp::MakeManifold(o) => {
                let r = fast_mesh_ops::make_manifold(tx, o)?;
                put_stage(&mut out, AutoRepairStage::Manifold(&r))?;
            }
            RepairOp::ResolveSelfIntersections(o) => {
                let r = fast_mesh_ops::resolve_self_intersections(tx, None, o)?;
                put_stage(&mut out, AutoRepairStage::SelfIntersections(&r))?;
            }
            RepairOp::CleanupSlivers(o) => {
                let r = fast_mesh_ops::cleanup_slivers(tx, o)?;
                put_stage(&mut out, AutoRepairStage::Slivers(&r))?;
            }
            RepairOp::Protected {
                vertices,
                weights,
                band,
                op,
            } => {
                let report = with_protection_band(tx, None, vertices, weights, *band, |tx| {
                    run_repair(tx, std::slice::from_ref(op))
                })?;
                put(&mut out, &report)?;
            }
            RepairOp::Inflate(o) => {
                let r = inflate(tx, o)?;
                put(&mut out, &[count(r.moved), f64::from(r.max_displacement)])?;
            }
            RepairOp::Smooth(o) => {
                let r = smooth(tx, o)?;
                put(&mut out, &[count(r.moved), f64::from(r.max_displacement)])?;
            }
            RepairOp::SplitStretched(o) => {
                let r = fast_mesh_ops::stretch::split_stretched(tx, o)?;
                put_stretch(&mut out, &r)?;
            }
            RepairOp::CaptureRest { .. } | RepairOp::CaptureStroke { .. } => {
                return Err(OpsError::InvalidInput(
                    "capture op must be the only operation in its list",
                ));
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
            RepairOp::Grab(o) => {
                let r = grab(tx, o)?;
                put(&mut out, &[count(r.moved), f64::from(r.max_displacement)])?;
            }
            RepairOp::Flatten(o) => {
                let r = flatten(tx, o)?;
                put(&mut out, &[count(r.moved), f64::from(r.max_displacement)])?;
            }
            RepairOp::Crease(o) => {
                let r = crease(tx, o)?;
                put(&mut out, &[count(r.moved), f64::from(r.max_displacement)])?;
            }
            RepairOp::Clay(o) => {
                let r = clay(tx, o)?;
                put(&mut out, &[count(r.moved), f64::from(r.max_displacement)])?;
            }
            RepairOp::Twist(o) => {
                let r = twist(tx, o)?;
                put(&mut out, &[count(r.moved), f64::from(r.max_displacement)])?;
            }
            RepairOp::Offset(o) => {
                let r = fast_mesh_ops::offset(tx, o)?;
                put(
                    &mut out,
                    &[
                        count(r.moved),
                        f64::from(r.max_displacement),
                        count(r.degenerate_removed),
                        count(r.vertices_removed),
                        count(r.vertices_added),
                        count(r.triangles_added),
                        count(r.triangles_removed),
                        f64::from(r.voxel_size),
                        f64::from(u8::from(r.uvs_dropped)),
                    ],
                )?;
            }
            RepairOp::Shell(thickness) => {
                let r = fast_mesh_ops::shell(
                    tx,
                    &ShellOptions {
                        thickness: *thickness,
                    },
                )?;
                put(
                    &mut out,
                    &[
                        count(r.vertices_added),
                        count(r.triangles_added),
                        count(r.loops_closed),
                        count(r.degenerate_removed),
                        count(r.vertices_removed),
                    ],
                )?;
            }
            RepairOp::Simplify(o) => {
                let r = fast_mesh_ops::simplify::simplify(tx, o)?;
                put(
                    &mut out,
                    &[
                        count(r.collapsed),
                        count(r.triangles_removed),
                        count(r.vertices_removed),
                    ],
                )?;
            }
            RepairOp::Subdivide(o) => {
                let r = fast_mesh_ops::subdivide::subdivide(tx, o)?;
                put(
                    &mut out,
                    &[count(r.vertices_added), count(r.triangles_added)],
                )?;
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
        let smooth =
            |cycles: f64| decode_repair(&[6., 0., 0., 0., 1., 0., 1., cycles, f64::NAN], &[]);
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
                method: SmoothMethod::Taubin,
                ..Default::default()
            })]
        );
    }

    #[test]
    fn decode_repair_relax() {
        use fast_mesh_ops::deform::{Brush, Falloff, SmoothMethod, SmoothOptions};

        let d = decode_repair(&[10., 1., 2., 3., 4., 1., 0.5, 2.], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Smooth(SmoothOptions {
                brush: Brush {
                    center: [1.0, 2.0, 3.0],
                    radius: 4.0,
                    falloff: Falloff::Smooth,
                    strength: 0.5,
                },
                cycles: 2,
                method: SmoothMethod::Laplacian,
                ..Default::default()
            })]
        );
        // Code 6 keeps Taubin when method is NaN or 0, and supports 1..=3.
        for method in [f64::NAN, 0.] {
            let d = decode_repair(&[6., 1., 2., 3., 4., 1., 0.5, 2., method], &[]).unwrap();
            assert!(matches!(
                d.as_slice(),
                [RepairOp::Smooth(o)] if o.method == SmoothMethod::Taubin
            ));
        }
        let code = |ops: &[f64]| decode_repair(ops, &[]).unwrap_err().code;
        assert_eq!(code(&[10.]), codes::INVALID_ARGUMENT);
        assert_eq!(
            code(&[10., 0., 0., 0., 1., 0., 1.]),
            codes::INVALID_ARGUMENT
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
    fn decode_repair_split_stretched() {
        use fast_mesh_ops::remesh::Region;
        use fast_mesh_ops::stretch::StretchOptions;
        let nan = f64::NAN;
        let region = Region {
            center: [1.0, 2.0, 3.0],
            radius: 4.0,
        };
        assert_eq!(
            decode_repair(&[16., 1., 2., 3., 4., nan, nan, 1., nan], &[]).unwrap(),
            vec![RepairOp::SplitStretched(StretchOptions {
                region,
                ..StretchOptions::default()
            })]
        );
        assert_eq!(
            decode_repair(&[16., 1., 2., 3., 4., 2., 7., 0., 20.], &[]).unwrap(),
            vec![RepairOp::SplitStretched(StretchOptions {
                region,
                limit: 2.0,
                max_splits: 7,
                smooth: false,
                sharp_angle: 20.0,
            })]
        );
        assert_eq!(
            decode_repair(&[17., 1.], &[]).unwrap(),
            vec![RepairOp::CaptureRest {
                only_if_missing: true
            }]
        );
        let code = |ops: &[f64]| decode_repair(ops, &[]).unwrap_err().code;
        assert_eq!(
            code(&[16., 1., 2., 3., 4., nan, nan, 2.]),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(&[16., 1., 2., 3., 4., nan, -1., 1.]),
            codes::INVALID_ARGUMENT
        );
    }

    #[test]
    fn decode_repair_remesh() {
        use fast_mesh_ops::remesh::{REMESH_MAX_ROUNDS, Region, RemeshOptions};

        let nan = f64::NAN;
        let d = decode_repair(&[7., 1., 2., 3., 4., 0.5, nan, 0.], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Remesh(RemeshOptions {
                region: Region {
                    center: [1.0, 2.0, 3.0],
                    radius: 4.0,
                },
                target_length: Some(0.5),
                max_rounds: REMESH_MAX_ROUNDS,
                split_only: false,
            })]
        );
        let d = decode_repair(&[7., 0., 0., 0., 1., nan, 3., 1.], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Remesh(RemeshOptions {
                region: Region::default(),
                target_length: None,
                max_rounds: 3,
                split_only: true,
            })]
        );
        // The u32 boundary itself still decodes.
        let d = decode_repair(&[7., 0., 0., 0., 1., nan, f64::from(u32::MAX), 0.], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Remesh(RemeshOptions {
                region: Region::default(),
                target_length: None,
                max_rounds: u32::MAX,
                split_only: false,
            })]
        );
        let code = |ops: &[f64]| decode_repair(ops, &[]).unwrap_err().code;
        assert_eq!(code(&[7.]), codes::INVALID_ARGUMENT);
        assert_eq!(
            code(&[7., 0., 0., 0., 1., nan, -1., 0.]),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(&[7., 0., 0., 0., 1., nan, 1.5, 0.]),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(&[7., 0., 0., 0., 1., nan, 2f64.powi(32), 0.]),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(&[7., 0., 0., 0., 1., nan, 2., 0.5]),
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

    #[test]
    fn decode_repair_protect() {
        let brush = [5., 1., 2., 3., 4., 1., 0.5, f64::NAN];
        let mut list = vec![32., 2., 0.75, 7., 9., 0., 0.25];
        list.extend(brush);
        let d = decode_repair(&list, &[]).unwrap();
        let [
            RepairOp::Protected {
                vertices,
                weights,
                band,
                op,
            },
        ] = d.as_slice()
        else {
            panic!("expected one protected op: {d:?}");
        };
        assert_eq!(vertices, &[7, 9]);
        assert_eq!(weights, &[0.0, 0.25]);
        assert_eq!(*band, 0.75);
        assert!(matches!(**op, RepairOp::Inflate(_)));

        // Nothing to wrap, two in a row, negative band, or an op that is not a brush.
        assert!(decode_repair(&[32., 0., 0.], &[]).is_err());
        assert!(
            decode_repair(
                &[32., 0., 0., 32., 0., 0., 5., 0., 0., 0., 1., 0., 1., 0.],
                &[]
            )
            .is_err()
        );
        assert!(decode_repair(&[32., 0., -0.1, 5., 0., 0., 0., 1., 0., 1., 0.], &[]).is_err());
        assert!(decode_repair(&[32., 0., 0., 3.], &[]).is_err());
        // Cut short, and an id that is not a u32.
        assert!(decode_repair(&[32., 2., 0., 7.], &[]).is_err());
        assert!(decode_repair(&[32., 1., 0., 0.5, 0.], &[]).is_err());
        assert!(decode_repair(&[32., 1., 0., -1., 0.], &[]).is_err());
    }

    #[test]
    fn decode_repair_grab() {
        use fast_mesh_ops::sculpt::{Brush, Falloff, GrabMethod, GrabOptions};

        let d = decode_repair(
            &[
                9.,
                1.,
                2.,
                3.,
                4.,
                1.,
                0.5,
                5.,
                6.,
                7.,
                f64::NAN,
                f64::NAN,
                0.,
                0.,
            ],
            &[],
        )
        .unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Grab(GrabOptions {
                brush: Brush {
                    center: [1.0, 2.0, 3.0],
                    radius: 4.0,
                    falloff: Falloff::Smooth,
                    strength: 0.5,
                },
                direction: [5.0, 6.0, 7.0],
                ..Default::default()
            })]
        );
        let d2 = decode_repair(
            &[9., 1., 2., 3., 4., 0., 0.5, 1., 0., 0., 2., 0.25, 1., 1.],
            &[],
        )
        .unwrap();
        assert_eq!(
            d2,
            vec![RepairOp::Grab(GrabOptions {
                brush: Brush {
                    center: [1.0, 2.0, 3.0],
                    radius: 4.0,
                    falloff: Falloff::Linear,
                    strength: 0.5,
                },
                direction: [1.0, 0.0, 0.0],
                method: GrabMethod::Kelvinlet,
                poisson: 0.25,
                from_stroke: true,
                adaptive_reach: true,
                ..Default::default()
            })]
        );
        let code = |ops: &[f64]| decode_repair(ops, &[]).unwrap_err().code;
        assert_eq!(code(&[9.]), codes::INVALID_ARGUMENT);
        assert_eq!(
            code(&[9., 0., 0., 0., 1., 1., 1., 2.]),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(&[9., 0., 0., 0., 1., 2., 1., 0., 0., 0., 0., 0.4, 0., 0.]),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(&[9., 0., 0., 0., 1., 1., 1., 1., 0., 0., 2., 0.9, 0., 0.]),
            codes::INVALID_ARGUMENT
        );
    }

    #[test]
    fn decode_repair_sculpt_methods_and_modifiers() {
        use fast_mesh_ops::sculpt::{
            ClayOptions, ClayProfile, CreaseOptions, FlattenMode, FlattenOptions, InflateMethod,
            SmoothMethod, SurfaceFilter, Symmetry, TwistMethod, TwistOptions,
        };

        let nan = f64::NAN;
        // Inflate methods 0..=2 and smooth methods 0..=3.
        let d = decode_repair(
            &[
                5., 0., 0., 0., 1., 1., 0.2, 2., //
                6., 0., 0., 0., 1., 1., 0.5, 3., 3., //
                18., 1., //
                19., 0., 0., 0., 2., 1., 0.8, 1., 1., 1., 2., 3., 0., 0., 0., 0., //
                20., 0., 0., 0., 1., 1., 0.4, 0.7, //
                21., 0., 0., 0., 1., 0., 0.5, 1., 1., 1., 0., 0., //
                22., 0., 0., 0., 1., 1., 1.2, 1., 1., 0., 0., 1.,
            ],
            &[],
        )
        .unwrap();
        assert!(matches!(
            d[0],
            RepairOp::Inflate(o) if o.method == InflateMethod::StrokeNormal
        ));
        assert!(matches!(
            d[1],
            RepairOp::Smooth(o) if o.method == SmoothMethod::Cotangent && o.cycles == 3
        ));
        assert_eq!(
            d[2],
            RepairOp::CaptureStroke {
                only_if_missing: true
            }
        );
        assert!(matches!(
            d[3],
            RepairOp::Flatten(FlattenOptions {
                mode: FlattenMode::Peaks,
                plane_origin: Some([1.0, 2.0, 3.0]),
                plane_normal: None,
                ..
            })
        ));
        assert!(matches!(
            d[4],
            RepairOp::Crease(CreaseOptions { pinch, .. }) if (pinch - 0.7).abs() < 1e-6
        ));
        assert!(matches!(
            d[5],
            RepairOp::Clay(ClayOptions {
                profile: ClayProfile::Strips,
                tangent: Some([1.0, 0.0, 0.0]),
                ..
            })
        ));
        assert!(matches!(
            d[6],
            RepairOp::Twist(TwistOptions {
                method: TwistMethod::Kelvinlet,
                axis: Some([0.0, 0.0, 1.0]),
                ..
            })
        ));

        // 32 (protect) and 33 (brushModifiers) compose in BOTH wire orders.
        let p = [32., 1., 0.5, 4., 0.];
        let m = [33., 3., 3., 30., 0., 1., 0., 0., 1.];
        let b = [5., 0., 0., 0., 1., 1., 0.5, 1.];
        let mut order_a = Vec::new();
        order_a.extend(p);
        order_a.extend(m);
        order_a.extend(b);
        let mut order_b = Vec::new();
        order_b.extend(m);
        order_b.extend(p);
        order_b.extend(b);
        let da = decode_repair(&order_a, &[]).unwrap();
        let db = decode_repair(&order_b, &[]).unwrap();
        assert_eq!(da, db);
        let [
            RepairOp::Protected {
                vertices, band, op, ..
            },
        ] = da.as_slice()
        else {
            panic!("expected protected op");
        };
        assert_eq!(vertices, &[4]);
        assert_eq!(*band, 0.5);
        let RepairOp::Inflate(inf) = **op else {
            panic!("expected inflate");
        };
        assert_eq!(inf.method, InflateMethod::AverageNormal);
        assert_eq!(
            inf.symmetry,
            Symmetry {
                x: true,
                y: true,
                z: false
            }
        );
        assert_eq!(
            inf.filter,
            SurfaceFilter::NormalAngle {
                normal: Some([0.0, 0.0, 1.0]),
                max_angle_deg: 30.0,
            }
        );

        // Invalid maxAngleDeg (0) and non-brush wrapped by 33 fail up front.
        let code = |ops: &[f64]| decode_repair(ops, &[]).unwrap_err().code;
        assert_eq!(
            code(&[
                33., 0., 3., 0., 0., 0., 0., 0., 0., 5., 0., 0., 0., 1., 1., 0.5, nan
            ]),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(&[33., 1., 0., 0., 0., 0., 0., 0., 0., 3.]),
            codes::INVALID_ARGUMENT
        );
    }

    #[test]
    fn decode_repair_offset_shell() {
        let nan = f64::NAN;
        let d = decode_repair(&[11., 0.5, nan, nan, nan], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Offset(OffsetOptions {
                distance: 0.5,
                algorithm: OffsetAlgorithm::Normal,
                max_voxels: OffsetOptions::default().max_voxels,
            })]
        );
        let d = decode_repair(&[11., -2., 0., 0.25, 1024.], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Offset(OffsetOptions {
                distance: -2.0,
                algorithm: OffsetAlgorithm::Normal,
                max_voxels: 1024,
            })]
        );
        let d = decode_repair(&[11., 0.25, 1., 0.125, 4096.], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Offset(OffsetOptions {
                distance: 0.25,
                algorithm: OffsetAlgorithm::FhEdt {
                    voxel_size: Some(0.125)
                },
                max_voxels: 4096,
            })]
        );
        let d = decode_repair(&[11., 0.0, 2., nan, nan], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Offset(OffsetOptions {
                distance: 0.0,
                algorithm: OffsetAlgorithm::BvhSdf { voxel_size: None },
                max_voxels: OffsetOptions::default().max_voxels,
            })]
        );
        let d = decode_repair(&[12., 0.25], &[]).unwrap();
        assert_eq!(d, vec![RepairOp::Shell(0.25)]);

        let code = |ops: &[f64]| decode_repair(ops, &[]).unwrap_err().code;
        assert_eq!(code(&[11.]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[11., 0.5]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[11., 0.5, nan, nan]), codes::INVALID_ARGUMENT);
        assert_eq!(
            code(&[11., 0.5, nan, nan, nan, 0.]),
            codes::INVALID_ARGUMENT
        );
        // Invalid algorithm codes.
        assert_eq!(code(&[11., 0.5, 3., nan, nan]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[11., 0.5, -1., nan, nan]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[11., 0.5, 1.5, nan, nan]), codes::INVALID_ARGUMENT);
        assert_eq!(
            code(&[11., 0.5, f64::INFINITY, nan, nan]),
            codes::INVALID_ARGUMENT
        );
        // Invalid voxel_size values.
        assert_eq!(code(&[11., 0.5, 1., 0.0, nan]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[11., 0.5, 1., -1.0, nan]), codes::INVALID_ARGUMENT);
        assert_eq!(
            code(&[11., 0.5, 1., f64::INFINITY, nan]),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(code(&[11., 0.5, 1., 1e-46, nan]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[11., 0.5, 1., 1e300, nan]), codes::INVALID_ARGUMENT);
        // Invalid max_voxels values.
        assert_eq!(code(&[11., 0.5, 1., nan, 16.0]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[11., 0.5, 1., nan, 63.0]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[11., 0.5, 1., nan, -1.0]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[11., 0.5, 1., nan, 64.5]), codes::INVALID_ARGUMENT);
        assert_eq!(
            code(&[11., 0.5, 1., nan, f64::INFINITY]),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(code(&[11., 0.5, 1., nan, 1e20]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[12.]), codes::INVALID_ARGUMENT);
    }

    #[test]
    fn run_repair_offset_reports_nine_words() {
        use fast_mesh_store::Mesh;

        let unit_cube = |with_uvs: bool| {
            let positions = [
                0.0f32, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                1.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, //
                1.0, 0.0, 1.0, //
                1.0, 1.0, 1.0, //
                0.0, 1.0, 1.0,
            ];
            let indices = [
                0u32, 2, 1, 0, 3, 2, //
                4, 5, 6, 4, 6, 7, //
                0, 1, 5, 0, 5, 4, //
                2, 3, 7, 2, 7, 6, //
                1, 2, 6, 1, 6, 5, //
                3, 0, 4, 3, 4, 7,
            ];
            let mut m = Mesh::from_buffers(&positions, &indices).unwrap();
            if with_uvs {
                m.add_uvs(&[0.0f32; 16]).unwrap();
            }
            m
        };

        let nan = f64::NAN;
        // Normal offset: 9 words, zero volumetric counts, uvs_dropped = 0.
        let mut m = unit_cube(true);
        let ops = decode_repair(&[11., 0.25, nan, nan, nan], &[]).unwrap();
        let mut tx = m.begin_edit();
        let out = run_repair(&mut tx, &ops).unwrap();
        tx.commit().unwrap();
        assert_eq!(out.len(), 9);
        assert_eq!(out[0], 8.0); // moved
        assert!((out[1] - 0.25).abs() < 1e-5); // max_displacement
        assert_eq!(out[2], 0.0); // degenerate_removed
        assert_eq!(out[3], 0.0); // vertices_removed
        assert_eq!(out[4], 0.0); // vertices_added
        assert_eq!(out[5], 0.0); // triangles_added
        assert_eq!(out[6], 0.0); // triangles_removed
        assert_eq!(out[7], 0.0); // voxel_size
        assert_eq!(out[8], 0.0); // uvs_dropped
        assert!(m.uvs().is_some());

        // FhEdt offset on plain cube: 9 words, positive volumetric counts, uvs_dropped = 0.
        let mut m = unit_cube(false);
        let ops = decode_repair(&[11., 0.1, 1., 0.2, 64_000.], &[]).unwrap();
        let mut tx = m.begin_edit();
        let out = run_repair(&mut tx, &ops).unwrap();
        tx.commit().unwrap();
        assert_eq!(out.len(), 9);
        assert!(out[4] > 0.0); // vertices_added
        assert!(out[5] > 0.0); // triangles_added
        assert_eq!(out[6], 12.0); // triangles_removed
        assert!((out[7] - 0.2).abs() < 1e-6); // voxel_size
        assert_eq!(out[8], 0.0); // uvs_dropped

        // BvhSdf offset on textured cube: drops UV_CHANNEL and reports uvs_dropped = 1.
        let mut m = unit_cube(true);
        let ops = decode_repair(&[11., 0.1, 2., 0.2, 64_000.], &[]).unwrap();
        let mut tx = m.begin_edit();
        let out = run_repair(&mut tx, &ops).unwrap();
        tx.commit().unwrap();
        assert_eq!(out.len(), 9);
        assert!(out[4] > 0.0); // vertices_added
        assert!(out[5] > 0.0); // triangles_added
        assert_eq!(out[6], 12.0); // triangles_removed
        assert!((out[7] - 0.2).abs() < 1e-6); // voxel_size
        assert_eq!(out[8], 1.0); // uvs_dropped
        assert!(m.uvs().is_none());
    }

    #[test]
    fn decode_repair_simplify() {
        let d = decode_repair(&[13., 200_000., 1.], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Simplify(SimplifyOptions {
                target_triangles: 200_000,
                preserve_boundary: true,
            })]
        );
        let d = decode_repair(&[13., 10., 0.], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Simplify(SimplifyOptions {
                target_triangles: 10,
                preserve_boundary: false,
            })]
        );
        let code = |ops: &[f64]| decode_repair(ops, &[]).unwrap_err().code;
        assert_eq!(code(&[13.]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[13., 0., 1.]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[13., f64::NAN, 1.]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[13., 1.5, 1.]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[13., 10., 2.]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[13., 10., 1., 0.]), codes::INVALID_ARGUMENT);
    }

    #[test]
    fn decode_repair_auto() {
        let nan = f64::NAN;
        let d = decode_repair(&[15., nan, nan, nan, nan], &[]).unwrap();
        assert_eq!(d, vec![RepairOp::Auto(AutoRepairOptions::default())]);
        let d = decode_repair(&[15., 0.5, 3., f64::INFINITY, 0.], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Auto(AutoRepairOptions {
                weld_tolerance: 0.5,
                min_triangles: 3,
                max_edges: None,
                fair: false,
                ..AutoRepairOptions::default()
            })]
        );
        let d = decode_repair(&[15., nan, nan, 8., 1.], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Auto(AutoRepairOptions {
                max_edges: Some(8),
                ..AutoRepairOptions::default()
            })]
        );
        let code = |ops: &[f64]| decode_repair(ops, &[]).unwrap_err().code;
        assert_eq!(code(&[15., nan, nan, nan]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[15., nan, 1.5, nan, nan]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[15., nan, nan, -1., nan]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[15., nan, nan, 1.5, nan]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[15., nan, nan, nan, 2.]), codes::INVALID_ARGUMENT);
    }

    #[test]
    fn decode_repair_subdivide() {
        let d = decode_repair(&[14., 0., 2.], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Subdivide(SubdivideOptions {
                method: SubdivideMethod::Loop,
                levels: 2,
            })]
        );
        // NaN means the defaults: Loop, one level.
        let d = decode_repair(&[14., f64::NAN, f64::NAN], &[]).unwrap();
        assert_eq!(
            d,
            vec![RepairOp::Subdivide(SubdivideOptions {
                method: SubdivideMethod::Loop,
                levels: 1,
            })]
        );
        let code = |ops: &[f64]| decode_repair(ops, &[]).unwrap_err().code;
        assert_eq!(code(&[14.]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[14., 0.]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[14., 1., 1.]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[14., 0., 0.]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[14., 0., 1.5]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[14., 0., -1.]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[14., 0., 2f64.powi(32)]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[14., 0., 1., 0.]), codes::INVALID_ARGUMENT);
    }

    #[test]
    fn decode_repair_healing_ops_and_presets() {
        let nan = f64::NAN;
        assert_eq!(
            decode_repair(&[23., nan, nan, nan, nan], &[]).unwrap(),
            vec![RepairOp::MakeManifold(ManifoldOptions::default())]
        );
        assert_eq!(
            decode_repair(&[23., 1., 0., 1., 0.], &[]).unwrap(),
            vec![RepairOp::MakeManifold(ManifoldOptions {
                remove_duplicates: true,
                prune_fins: false,
                split_non_manifold_edges: true,
                split_pinch_vertices: false,
            })]
        );

        assert_eq!(
            decode_repair(&[24., nan, nan, nan, nan, nan, nan, nan], &[]).unwrap(),
            vec![RepairOp::ResolveSelfIntersections(
                SelfIntersectOptions::default()
            )]
        );
        assert_eq!(
            decode_repair(&[24., 3., 2., 0.25, 1024., 16., 0., 1e-4], &[]).unwrap(),
            vec![RepairOp::ResolveSelfIntersections(SelfIntersectOptions {
                mode: SelfIntersectionMode::VoxelHeal,
                ring_expansion: 2,
                voxel_size: Some(0.25),
                max_voxels: 1024,
                max_hole_edges: Some(16),
                fair: false,
                stitch_tolerance: 1e-4,
            })]
        );

        assert_eq!(
            decode_repair(&[25., nan, nan, nan, nan], &[]).unwrap(),
            vec![RepairOp::CleanupSlivers(SliverOptions::default())]
        );
        assert_eq!(
            decode_repair(&[25., 30., 5., 170., 4.], &[]).unwrap(),
            vec![RepairOp::CleanupSlivers(SliverOptions {
                max_aspect_ratio: 30.0,
                min_angle_deg: 5.0,
                max_angle_deg: 170.0,
                max_iterations: 4,
            })]
        );

        assert_eq!(
            decode_repair(&[26., 2., nan, nan, nan, nan, nan, nan, nan], &[]).unwrap(),
            vec![RepairOp::AutoPreset(AutoRepairOptions::for_preset(
                AutoRepairPreset::Print3D
            ))]
        );
        assert_eq!(
            decode_repair(&[26., 3., 2., 1e-6, 0., 12., 0., nan, nan], &[]).unwrap(),
            vec![RepairOp::AutoPreset(
                AutoRepairOptions::builder()
                    .preset(AutoRepairPreset::CadImport)
                    .self_intersection_mode(Some(SelfIntersectionMode::CutAndStitch))
                    .weld_tolerance(1e-6)
                    .min_triangles(0)
                    .max_edges(Some(12))
                    .fair(false)
                    .build()
            )]
        );

        let code = |ops: &[f64]| decode_repair(ops, &[]).unwrap_err().code;
        assert_eq!(code(&[23., 2., 1., 1., 1.]), codes::INVALID_ARGUMENT);
        assert_eq!(
            code(&[24., 4., 1., nan, nan, nan, 1., 0.]),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(&[24., 1., 1., 0., nan, nan, 1., 0.]),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(&[24., 1., 1., nan, 32., nan, 1., 0.]),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(&[24., 1., 1., nan, nan, nan, 1., -1.]),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(code(&[25., 0.5, 2., 176., 3.]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[25., 20., 70., 176., 3.]), codes::INVALID_ARGUMENT);
        assert_eq!(code(&[25., 20., 2., 50., 3.]), codes::INVALID_ARGUMENT);
        assert_eq!(
            code(&[26., 9., nan, nan, nan, nan, nan, nan, nan]),
            codes::INVALID_ARGUMENT
        );
    }
}
