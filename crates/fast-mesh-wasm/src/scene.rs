//! The `WasmScene` class: a [`Scene`] behind wasm-bindgen.
//!
//! Each method converts its arguments, calls the scene and maps errors to
//! codes ([`crate::error`]). Mesh ids cross the boundary as
//! `(index, generation)` pairs and are checked with [`Scene::mesh_id`], so
//! a stale or made-up pair is `UNKNOWN_MESH`.
//!
//! Render buffers are read in place: the `*_ptr` / `*_len` methods give an
//! address in wasm memory and a length in elements of the TypeScript array
//! type, and the TypeScript layer builds typed-array views from them. A
//! view is valid until the next call that may allocate.
//!
//! Field and section ids cross as their raw `u32` and are checked with
//! [`Scene::field_id`] and [`Scene::section_id`]. Query and section results
//! are flat `f64` arrays (ids, counts and flags included, all exact in
//! `f64`) whose layout each method documents; the TypeScript layer builds
//! objects from them. A list of meshes is flat `[index, generation, ...]`.

use std::collections::BTreeMap;

use fast_mesh_ops::repair::{FillContinuity, FillHolesOptions, FillStats, PartialBridgeOptions};
use fast_mesh_ops::sculpt::{DyntopoMode, DyntopoOptions, DyntopoPatch};
use fast_mesh_ops::{AnalyzeOptions, IcpOptions, IcpStatus, VolumeReason, analyze};
use fast_mesh_query::Cull;
use fast_mesh_scene::{
    BooleanMethod, BooleanOp, BooleanOptions, CoarseAlignMethod, ColorRule, Coloring, Colormap,
    ContactFilter, CsgNode, CurvatureField, CutOptions, Dab, DeviationOptions, DistanceField,
    FieldBudget, FieldId, GeodesicJob, IcpObjective, JoinOptions, LandmarkPair, MeshBuild, MeshId,
    MeshOptions, NotApplied, OutOfRange, OverhangOptions, Placement, Plane, QualityMetric,
    RegisterOptions, RegistrationOptions, Scene, SceneClosestOptions, SceneError, SceneOptions,
    SceneRayOptions, SceneRepairOps as _, SectionDesc, SectionId, SectionStatus, SignMode,
    StepState, ThicknessField, TrimCurveOptions, UpdateMode, ops::BooleanOp as OpsBooleanOp,
};
use fast_mesh_store::{Mesh, MeshBuilder};
use wasm_bindgen::prelude::*;

use crate::edit::{
    RepairOp, count_opt, decode_edit, decode_repair, encode_added, encode_analysis, loop_id,
    opt_positive_f32, put_stretch, run_edit, run_repair,
};
use crate::error::Error;
use crate::render::{Dirty, Record, RenderState, encode, uvs_reported};

mod denoise;
mod extrude;
mod geodesic;
mod height_map;
mod history;
mod hull;
mod lattice;
mod wrap;

/// A scene of editable meshes with render buffers and change records.
#[wasm_bindgen]
pub struct WasmScene {
    scene: Scene,
    /// The render state of every live mesh.
    render: BTreeMap<MeshId, RenderState>,
    /// Meshes being added in steps, by build id.
    adds: BTreeMap<u32, PendingAdd>,
    /// The id the next `begin_add_mesh_upload` hands out.
    next_add: u32,
    /// Geodesic queries running in steps, by query id.
    geodesics: BTreeMap<u32, GeodesicJob>,
    /// The id the next `begin_geodesic_distances` hands out.
    next_geodesic: u32,
}

/// A mesh being added in steps: its inputs still uploading into wasm
/// memory ([`MeshBuilder`]) and its store still building, or its store
/// built and its BVH (and vertex normals) building.
enum PendingAdd {
    /// The store uploads and builds in steps; `options` and `normals`
    /// start the BVH build once it is done, with `uvs` (empty for none) on
    /// the built mesh.
    Store {
        build: Box<MeshBuilder>,
        options: MeshOptions,
        normals: bool,
        uvs: Vec<f32>,
    },
    /// The store is built; the BVH builds in steps: the build, its update
    /// mode and whether it gets vertex normals.
    Bvh(Box<MeshBuild>, UpdateMode, bool),
}

/// Starts the BVH build of a built store, with `options` and `normals`,
/// as [`WasmScene::begin_add_mesh_upload`] promised it.
fn start_build(
    build: Box<MeshBuilder>,
    options: MeshOptions,
    normals: bool,
    uvs: &[f32],
) -> Result<(Box<MeshBuild>, UpdateMode, bool), Error> {
    let mut mesh = build.finish().ok_or(SceneError::Corrupted)?;
    if !uvs.is_empty() {
        mesh.add_uvs(uvs)?;
    }
    let mode = options.mode;
    let mut build = MeshBuild::new(mesh, options)?;
    // Computed in the build's steps; `added` then finds them on.
    build.set_normals(normals);
    Ok((Box::new(build), mode, normals))
}

/// A mesh from world positions, stored relative to `origin` (or the
/// bounding-box centre).
fn mesh_from(positions: &[f64], indices: &[u32], origin: Option<Vec<f64>>) -> Result<Mesh, Error> {
    Ok(match origin {
        None => Mesh::from_buffers_f64(positions, indices)?,
        Some(o) => {
            let o: [f64; 3] = o
                .as_slice()
                .try_into()
                .map_err(|_| Error::invalid("origin must have 3 values"))?;
            Mesh::from_buffers_f64_with_origin(positions, indices, o)?
        }
    })
}

/// The update mode of its binding code: 0 Eager, 1 Lazy, 2 Manual.
fn parse_mode(mode: u8) -> Result<UpdateMode, Error> {
    match mode {
        0 => Ok(UpdateMode::Eager),
        1 => Ok(UpdateMode::Lazy),
        2 => Ok(UpdateMode::Manual),
        _ => Err(Error::invalid(
            "mode must be 0 (eager), 1 (lazy) or 2 (manual)",
        )),
    }
}

/// The address of a slice in wasm memory (lossless on wasm32).
fn addr<T>(s: &[T]) -> u32 {
    s.as_ptr().addr() as u32
}

/// A count or length for JavaScript (lossless on wasm32).
fn len(n: usize) -> u32 {
    n as u32
}

/// A count for a flat `f64` result (lossless: far below 2^53).
fn count(n: usize) -> f64 {
    n as f64
}

fn vec3(v: &[f64], what: &str) -> Result<[f64; 3], Error> {
    v.try_into().map_err(|_| Error::invalid(what))
}

/// A world plane from `[origin xyz, normal xyz]`, with its default frame.
fn parse_plane(v: &[f64]) -> Result<Plane, Error> {
    let [ox, oy, oz, nx, ny, nz]: [f64; 6] = v
        .try_into()
        .map_err(|_| Error::invalid("plane must have 6 values (origin, normal)"))?;
    Plane::new([ox, oy, oz], [nx, ny, nz]).map_err(|e| SceneError::Query(e).into())
}

/// The sign mode of its binding code: 0 Auto, 1 Pseudonormal, 2 Winding.
fn parse_sign(sign: u8) -> Result<SignMode, Error> {
    match sign {
        0 => Ok(SignMode::Auto),
        1 => Ok(SignMode::Pseudonormal),
        2 => Ok(SignMode::Winding),
        _ => Err(Error::invalid(
            "sign must be 0 (auto), 1 (pseudonormal) or 2 (winding)",
        )),
    }
}

/// The binding code of a sign mode, as [`parse_sign`] reads it.
fn sign_code(sign: SignMode) -> u32 {
    match sign {
        SignMode::Auto => 0,
        SignMode::Pseudonormal => 1,
        SignMode::Winding => 2,
    }
}

/// An out-of-range rule from its code: 0 Clamp, 1 Base.
fn parse_out_of_range(code: f32) -> Result<OutOfRange, Error> {
    if code == 0.0 {
        Ok(OutOfRange::Clamp)
    } else if code == 1.0 {
        Ok(OutOfRange::Base)
    } else {
        Err(Error::invalid(
            "below and above must be 0 (clamp) or 1 (base)",
        ))
    }
}

/// An RGBA8 colour from 4 values that must be integers in `0..=255`.
fn parse_rgba(v: &[f32]) -> Result<[u8; 4], Error> {
    let mut out = [0; 4];
    for (o, &c) in out.iter_mut().zip(v) {
        if !(0.0..=255.0).contains(&c) || c.fract() != 0.0 {
            return Err(Error::invalid(
                "colour channels must be integers in 0..=255",
            ));
        }
        *o = c as u8;
    }
    Ok(out)
}

/// A colour rule from `map` (0 Turbo, 1 Viridis, 2 Coolwarm, 3 Grayscale,
/// 4 custom stops) and `rule`: `[lo, hi, center (NaN for none), below,
/// above, base rgba, nan rgba]`, then for custom stops `(t, r, g, b, a)`
/// each. The rule's own checks run in [`Scene::set_coloring`].
pub(crate) fn parse_rule(map: u8, rule: &[f32]) -> Result<ColorRule, Error> {
    const HEAD: usize = 13;
    let (stops_flat, rest) = rule.get(HEAD..).unwrap_or_default().as_chunks::<5>();
    if rule.len() < HEAD || !rest.is_empty() || (map != 4 && !stops_flat.is_empty()) {
        return Err(Error::invalid("malformed colour rule"));
    }
    let map = match map {
        0 => Colormap::Turbo,
        1 => Colormap::Viridis,
        2 => Colormap::Coolwarm,
        3 => Colormap::Grayscale,
        4 => {
            let mut stops = Vec::new();
            stops
                .try_reserve(stops_flat.len())
                .map_err(|_| Error::from(fast_mesh_store::StoreError::OutOfMemory))?;
            for s in stops_flat {
                stops.push((s[0], parse_rgba(&s[1..])?));
            }
            Colormap::Custom(stops)
        }
        _ => return Err(Error::invalid("unknown colormap")),
    };
    Ok(ColorRule {
        map,
        range: (rule[0], rule[1]),
        center: (!rule[2].is_nan()).then_some(rule[2]),
        below: parse_out_of_range(rule[3])?,
        above: parse_out_of_range(rule[4])?,
        base: parse_rgba(&rule[5..9])?,
        nan: parse_rgba(&rule[9..13])?,
    })
}

impl Default for WasmScene {
    fn default() -> Self {
        WasmScene::new()
    }
}

impl WasmScene {
    /// After a mesh `id` is added: its vertex normals (with `normals`)
    /// and its render state; returns `[index, generation]`. When the
    /// normals cannot be turned on, the mesh is removed again.
    fn added(&mut self, id: MeshId, normals: bool) -> Result<Vec<u32>, Error> {
        if normals && let Err(e) = self.scene.set_normals(id, true) {
            // Added and removed before any report: the scene reports nothing.
            let _ = self.scene.remove_mesh(id);
            return Err(e.into());
        }
        let state = RenderState::new(self.scene.mesh(id)?);
        self.render.insert(id, state);
        Ok(vec![id.index(), id.generation()])
    }

    /// The live mesh id behind `(index, generation)`.
    fn id(&self, index: u32, generation: u32) -> Result<MeshId, Error> {
        self.scene
            .mesh_id(index, generation)
            .ok_or_else(|| SceneError::UnknownMesh.into())
    }

    fn mesh(&self, index: u32, generation: u32) -> Result<&Mesh, Error> {
        Ok(self.scene.mesh(self.id(index, generation)?)?)
    }

    /// The live mesh ids of a flat `[index, generation, ...]` list.
    fn ids(&self, flat: &[u32]) -> Result<Vec<MeshId>, Error> {
        let (pairs, rest) = flat.as_chunks::<2>();
        if !rest.is_empty() {
            return Err(Error::invalid("a mesh list needs index, generation pairs"));
        }
        let mut out = Vec::new();
        out.try_reserve(pairs.len())
            .map_err(|_| Error::from(fast_mesh_store::StoreError::OutOfMemory))?;
        for &[index, generation] in pairs {
            out.push(self.id(index, generation)?);
        }
        Ok(out)
    }

    fn field(&self, raw: u32) -> Result<FieldId, Error> {
        self.scene
            .field_id(raw)
            .ok_or_else(|| SceneError::UnknownField.into())
    }

    fn section_id(&self, raw: u32) -> Result<SectionId, Error> {
        self.scene
            .section_id(raw)
            .ok_or_else(|| SceneError::UnknownSection.into())
    }
}

#[wasm_bindgen]
impl WasmScene {
    /// An empty scene with the default history budget.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmScene {
        WasmScene {
            scene: Scene::new(SceneOptions::default()),
            render: BTreeMap::new(),
            adds: BTreeMap::new(),
            next_add: 0,
            geodesics: BTreeMap::new(),
            next_geodesic: 0,
        }
    }

    // ---- Meshes ----

    /// Adds a mesh from world positions and returns `[index, generation]`.
    /// Positions are stored as `p - origin` in `f32`; without `origin` it
    /// is the bounding-box centre. `mode` is 0 (Eager), 1 (Lazy) or 2
    /// (Manual); `normals` turns vertex normals on. The placement starts as
    /// the identity. A Lazy or Manual mesh builds its BVH on first use
    /// instead of up front. Nothing is added on error.
    pub fn add_mesh(
        &mut self,
        positions: &[f64],
        indices: &[u32],
        origin: Option<Vec<f64>>,
        mode: u8,
        normals: bool,
    ) -> Result<Vec<u32>, Error> {
        let mode = parse_mode(mode)?;
        let mesh = mesh_from(positions, indices, origin)?;
        let options = MeshOptions {
            mode,
            defer_bvh: mode != UpdateMode::Eager,
            ..MeshOptions::default()
        };
        let id = self.scene.add_mesh(mesh, options)?;
        self.added(id, normals)
    }

    /// [`WasmScene::add_mesh`] with UVs: 2 finite `f32`s per vertex
    /// (`Mesh::add_uvs`). Nothing is added on error.
    pub fn add_mesh_with_uvs(
        &mut self,
        positions: &[f64],
        indices: &[u32],
        uvs: &[f32],
        origin: Option<Vec<f64>>,
        mode: u8,
        normals: bool,
    ) -> Result<Vec<u32>, Error> {
        let mode = parse_mode(mode)?;
        let mut mesh = mesh_from(positions, indices, origin)?;
        mesh.add_uvs(uvs)?;
        let options = MeshOptions {
            mode,
            defer_bvh: mode != UpdateMode::Eager,
            ..MeshOptions::default()
        };
        let id = self.scene.add_mesh(mesh, options)?;
        self.added(id, normals)
    }

    /// [`WasmScene::add_mesh`], or [`WasmScene::add_mesh_with_uvs`] with
    /// `uvs`, as an undo step (recorded in the open step, or as a step of its
    /// own): undo removes the mesh again, and redo brings it back under the
    /// same index and generation. Nothing is added on error.
    pub fn add_mesh_undoable(
        &mut self,
        positions: &[f64],
        indices: &[u32],
        uvs: Option<Vec<f32>>,
        origin: Option<Vec<f64>>,
        mode: u8,
        normals: bool,
    ) -> Result<Vec<u32>, Error> {
        let mode = parse_mode(mode)?;
        let mut mesh = mesh_from(positions, indices, origin)?;
        if let Some(uvs) = uvs {
            mesh.add_uvs(&uvs)?;
        }
        let options = MeshOptions {
            mode,
            defer_bvh: mode != UpdateMode::Eager,
            undoable: true,
            ..MeshOptions::default()
        };
        let id = self.scene.add_mesh(mesh, options)?;
        self.added(id, normals)
    }

    /// [`WasmScene::add_mesh`] in steps: reserves a mesh of `vertices`
    /// vertices and `triangles` triangles outside the scene and returns the
    /// build's id. Its inputs upload in chunks
    /// ([`WasmScene::add_mesh_push_positions`] and
    /// [`WasmScene::add_mesh_push_indices`]), then its store, BVH and
    /// normals build in steps ([`WasmScene::add_mesh_step`]), for
    /// [`WasmScene::finish_add_mesh`]. The scene is unchanged until the
    /// mesh is added.
    pub fn begin_add_mesh_upload(
        &mut self,
        vertices: u32,
        triangles: u32,
        origin: Option<Vec<f64>>,
        mode: u8,
        normals: bool,
    ) -> Result<u32, Error> {
        let mode = parse_mode(mode)?;
        let origin: Option<[f64; 3]> = match origin {
            None => None,
            Some(o) => Some(
                o.as_slice()
                    .try_into()
                    .map_err(|_| Error::invalid("origin must have 3 values"))?,
            ),
        };
        let build = MeshBuilder::new(vertices as usize, triangles as usize, origin)?;
        let options = MeshOptions {
            mode,
            ..MeshOptions::default()
        };
        let id = self.next_add;
        self.next_add = self.next_add.wrapping_add(1);
        self.adds.insert(
            id,
            PendingAdd::Store {
                build: Box::new(build),
                options,
                normals,
                uvs: Vec::new(),
            },
        );
        Ok(id)
    }

    /// Appends a chunk of flat `[u, v, ...]` UVs to a mesh being added. The
    /// chunks must add up to 2 per vertex; the build checks them when its
    /// store is done and fails like [`WasmScene::add_mesh_with_uvs`].
    pub fn add_mesh_push_uvs(&mut self, build: u32, chunk: &[f32]) -> Result<(), Error> {
        match self
            .adds
            .get_mut(&build)
            .ok_or_else(|| Error::invalid("no mesh is being added with this id"))?
        {
            PendingAdd::Store { uvs, .. } => {
                uvs.try_reserve(chunk.len())
                    .map_err(|_| fast_mesh_store::StoreError::OutOfMemory)?;
                uvs.extend_from_slice(chunk);
                Ok(())
            }
            PendingAdd::Bvh(..) => Err(Error::invalid("the mesh upload is finished")),
        }
    }

    /// Appends a chunk of flat `[x, y, z, ...]` world positions to a mesh
    /// being added; the chunks must add up to its declared vertices.
    pub fn add_mesh_push_positions(&mut self, build: u32, chunk: &[f64]) -> Result<(), Error> {
        match self
            .adds
            .get_mut(&build)
            .ok_or_else(|| Error::invalid("no mesh is being added with this id"))?
        {
            PendingAdd::Store { build, .. } => Ok(build.push_positions(chunk)?),
            PendingAdd::Bvh(..) => Err(Error::invalid("the mesh upload is finished")),
        }
    }

    /// Appends a chunk of flat `[a, b, c, ...]` indices to a mesh being
    /// added; the chunks must add up to its declared triangles.
    pub fn add_mesh_push_indices(&mut self, build: u32, chunk: &[u32]) -> Result<(), Error> {
        match self
            .adds
            .get_mut(&build)
            .ok_or_else(|| Error::invalid("no mesh is being added with this id"))?
        {
            PendingAdd::Store { build, .. } => Ok(build.push_indices(chunk)?),
            PendingAdd::Bvh(..) => Err(Error::invalid("the mesh upload is finished")),
        }
    }

    /// Builds about `budget` more units of a mesh being added: first its
    /// store (vertices or triangles a unit), then its BVH (see
    /// `MeshBuild::step`); `true` once its BVH is built. The upload must be
    /// complete first. A failed build is dropped.
    pub fn add_mesh_step(&mut self, build: u32, budget: u32) -> Result<bool, Error> {
        let entry = self
            .adds
            .remove(&build)
            .ok_or_else(|| Error::invalid("no mesh is being added with this id"))?;
        match entry {
            PendingAdd::Store {
                build: mut store,
                options,
                normals,
                uvs,
            } => {
                if store.step(budget as usize)? {
                    let (b, mode, normals) = start_build(store, options, normals, &uvs)?;
                    self.adds.insert(build, PendingAdd::Bvh(b, mode, normals));
                } else {
                    self.adds.insert(
                        build,
                        PendingAdd::Store {
                            build: store,
                            options,
                            normals,
                            uvs,
                        },
                    );
                }
                Ok(false)
            }
            PendingAdd::Bvh(mut b, mode, normals) => {
                let done = b.step(budget as usize)?;
                self.adds.insert(build, PendingAdd::Bvh(b, mode, normals));
                Ok(done)
            }
        }
    }

    /// The render buffers of a mesh being added in steps, as
    /// `[positions_ptr, positions_len, normals_ptr, normals_len,
    /// indices_ptr, indices_len]` (as the `*_ptr` / `*_len` methods give
    /// them once it is added; normals 0, 0 until built or when off). The
    /// added mesh keeps these addresses and contents until it changes, so
    /// they can be copied out before it lands. Valid until the next call
    /// that may allocate.
    pub fn add_mesh_buffers(&self, build: u32) -> Result<Vec<u32>, Error> {
        let b = match self
            .adds
            .get(&build)
            .ok_or_else(|| Error::invalid("no mesh is being added with this id"))?
        {
            PendingAdd::Store { .. } => {
                return Err(Error::invalid("the mesh is still being built"));
            }
            PendingAdd::Bvh(b, ..) => b,
        };
        let m = b.mesh();
        let (np, nl) = b.normals().map_or((0, 0), |n| (addr(n), len(n.len() * 3)));
        Ok(vec![
            addr(m.positions()),
            len(m.positions().len() * 3),
            np,
            nl,
            addr(m.triangles()),
            len(m.triangles().len() * 3),
        ])
    }

    /// Adds a mesh built in steps, finishing what is left of its store,
    /// BVH and normals, and returns `[index, generation]`, as
    /// [`WasmScene::add_mesh`] would.
    pub fn finish_add_mesh(&mut self, build: u32) -> Result<Vec<u32>, Error> {
        let entry = self
            .adds
            .remove(&build)
            .ok_or_else(|| Error::invalid("no mesh is being added with this id"))?;
        let (b, _, normals) = match entry {
            PendingAdd::Store {
                build: mut store,
                options,
                normals,
                uvs,
            } => {
                while !store.step(usize::MAX)? {}
                start_build(store, options, normals, &uvs)?
            }
            PendingAdd::Bvh(b, mode, normals) => (b, mode, normals),
        };
        let id = self.scene.add_mesh_built(*b)?;
        self.added(id, normals)
    }

    /// The mesh's content stamp (`Scene::content_stamp`): it goes up
    /// whenever the mesh's positions, triangles or normals change.
    pub fn content_stamp(&self, index: u32, generation: u32) -> Result<f64, Error> {
        // Exact in `f64` far beyond any count of changes.
        Ok(self.scene.content_stamp(self.id(index, generation)?)? as f64)
    }

    /// Drops a mesh being added in steps (nothing was added).
    pub fn drop_add_mesh(&mut self, build: u32) {
        self.adds.remove(&build);
    }

    /// Removes a mesh as an undo step (recorded in the open step, or as a
    /// step of its own): undo brings it back under the same index and
    /// generation, which give `UNKNOWN_MESH` until then. See
    /// `Scene::remove_mesh_undoable` for what comes back with it.
    pub fn remove_mesh_undoable(&mut self, index: u32, generation: u32) -> Result<(), Error> {
        let id = self.id(index, generation)?;
        // Its render state goes with the removal record (or stays, when the
        // removal is undone before that).
        Ok(self.scene.remove_mesh_undoable(id)?)
    }

    /// Whether the scene has this mesh (false for a removed one).
    pub fn has_mesh(&self, index: u32, generation: u32) -> bool {
        self.scene.mesh_id(index, generation).is_some()
    }

    /// Whether this removed mesh can still be brought back by undo or redo.
    pub fn is_restorable(&self, index: u32, generation: u32) -> bool {
        self.scene.restorable_id(index, generation).is_some()
    }

    /// Removes a mesh for good; its id never matches again.
    pub fn remove_mesh(&mut self, index: u32, generation: u32) -> Result<(), Error> {
        let id = self.id(index, generation)?;
        self.scene.remove_mesh(id)?;
        self.render.remove(&id);
        Ok(())
    }

    /// Sets the placement from a row-major rotation (9 values), a
    /// translation (3 values) and a scale (3 values, each positive; `None`
    /// for `[1, 1, 1]`). The rotation must be a proper rotation.
    pub fn set_placement(
        &mut self,
        index: u32,
        generation: u32,
        rotation: &[f64],
        translation: &[f64],
        scale: Option<Vec<f64>>,
    ) -> Result<(), Error> {
        let id = self.id(index, generation)?;
        let r: [f64; 9] = rotation
            .try_into()
            .map_err(|_| Error::invalid("rotation must have 9 values"))?;
        let translation: [f64; 3] = translation
            .try_into()
            .map_err(|_| Error::invalid("translation must have 3 values"))?;
        let scale: [f64; 3] = match scale {
            None => [1.0; 3],
            Some(s) => s
                .as_slice()
                .try_into()
                .map_err(|_| Error::invalid("scale must have 3 values"))?,
        };
        let placement = Placement {
            rotation: [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]],
            translation,
            scale,
        };
        Ok(self.scene.set_placement(id, placement)?)
    }

    /// The placement: the row-major rotation (9 values), then the
    /// translation (3 values), then the scale (3 values).
    pub fn placement(&self, index: u32, generation: u32) -> Result<Vec<f64>, Error> {
        let p = self.scene.placement(self.id(index, generation)?)?;
        let mut out = p.rotation.as_flattened().to_vec();
        out.extend_from_slice(&p.translation);
        out.extend_from_slice(&p.scale);
        Ok(out)
    }

    /// The world matrix of the local render coordinates,
    /// `placement · translate(origin)`, column-major (16 values): the
    /// rotation columns are scaled, and the last column is
    /// `rotation · (scale · origin) + translation`.
    pub fn matrix4(&self, index: u32, generation: u32) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let p = self.scene.placement(id)?;
        let o = self.scene.mesh(id)?.origin();
        let (r, s) = (p.rotation, p.scale);
        // The last column is R·(S·o) + t.
        let w = p.apply(o);
        Ok(vec![
            s[0] * r[0][0],
            s[0] * r[1][0],
            s[0] * r[2][0],
            0.0, //
            s[1] * r[0][1],
            s[1] * r[1][1],
            s[1] * r[2][1],
            0.0, //
            s[2] * r[0][2],
            s[2] * r[1][2],
            s[2] * r[2][2],
            0.0, //
            w[0],
            w[1],
            w[2],
            1.0,
        ])
    }

    /// Sets when the mesh's BVH, normals, fields and sections follow its
    /// edits: 0 (Eager), 1 (Lazy) or 2 (Manual).
    pub fn set_mode(&mut self, index: u32, generation: u32, mode: u8) -> Result<(), Error> {
        let id = self.id(index, generation)?;
        let mode = parse_mode(mode)?;
        self.scene.set_mode(id, mode)?;
        Ok(())
    }

    /// When the mesh's BVH, normals, fields and sections follow its edits:
    /// 0 (Eager), 1 (Lazy) or 2 (Manual).
    pub fn mode(&self, index: u32, generation: u32) -> Result<u8, Error> {
        Ok(match self.scene.mode(self.id(index, generation)?)? {
            UpdateMode::Eager => 0,
            UpdateMode::Lazy => 1,
            UpdateMode::Manual => 2,
        })
    }

    /// The box around the vertices the live triangles use, in the mesh's
    /// stored local coordinates: `[min xyz, max xyz]`. Empty without live
    /// triangles.
    pub fn bounds(&self, index: u32, generation: u32) -> Result<Vec<f64>, Error> {
        Ok(match self.scene.bounds(self.id(index, generation)?)? {
            Some(b) => vec![
                f64::from(b.min[0]),
                f64::from(b.min[1]),
                f64::from(b.min[2]),
                f64::from(b.max[0]),
                f64::from(b.max[1]),
                f64::from(b.max[2]),
            ],
            None => Vec::new(),
        })
    }

    /// Brings the mesh's BVH, normals, fields and sections up to date now.
    pub fn update(&mut self, index: u32, generation: u32) -> Result<(), Error> {
        let id = self.id(index, generation)?;
        Ok(self.scene.update(id)?)
    }

    /// Brings every non-Manual mesh up to date.
    pub fn flush(&mut self) -> Result<(), Error> {
        Ok(self.scene.flush()?)
    }

    /// Vertex slots, alive or dead: the length of the vertex buffers.
    pub fn vertex_slots(&self, index: u32, generation: u32) -> Result<u32, Error> {
        Ok(len(self.mesh(index, generation)?.vertex_slots()))
    }

    /// Triangle slots, alive or dead: the length of the index buffer.
    pub fn triangle_slots(&self, index: u32, generation: u32) -> Result<u32, Error> {
        Ok(len(self.mesh(index, generation)?.triangle_slots()))
    }

    /// Alive vertices.
    pub fn vertex_count(&self, index: u32, generation: u32) -> Result<u32, Error> {
        Ok(len(self.mesh(index, generation)?.vertex_count()))
    }

    /// Alive triangles.
    pub fn triangle_count(&self, index: u32, generation: u32) -> Result<u32, Error> {
        Ok(len(self.mesh(index, generation)?.triangle_count()))
    }

    // ---- Views ----

    /// Address of the local positions (`f32`, 3 per vertex slot).
    pub fn positions_ptr(&self, index: u32, generation: u32) -> Result<u32, Error> {
        Ok(addr(self.mesh(index, generation)?.positions()))
    }

    /// Length of the positions in `f32`s.
    pub fn positions_len(&self, index: u32, generation: u32) -> Result<u32, Error> {
        Ok(len(self.mesh(index, generation)?.positions().len() * 3))
    }

    /// Address of the normals (`f32`, 3 per vertex slot), 0 when they are
    /// off.
    pub fn normals_ptr(&self, index: u32, generation: u32) -> Result<u32, Error> {
        let n = self.scene.normals(self.id(index, generation)?)?;
        Ok(n.map_or(0, addr))
    }

    /// Length of the normals in `f32`s, 0 when they are off. After the
    /// scene ran out of memory it may differ from 3 · vertex slots until
    /// the next successful update.
    pub fn normals_len(&self, index: u32, generation: u32) -> Result<u32, Error> {
        let n = self.scene.normals(self.id(index, generation)?)?;
        Ok(n.map_or(0, |n| len(n.len() * 3)))
    }

    /// Address of the UVs (`f32`, 2 per vertex slot), 0 for a mesh without
    /// them.
    pub fn uvs_ptr(&self, index: u32, generation: u32) -> Result<u32, Error> {
        Ok(self.mesh(index, generation)?.uvs().map_or(0, addr))
    }

    /// Length of the UVs in `f32`s, 0 for a mesh without them.
    pub fn uvs_len(&self, index: u32, generation: u32) -> Result<u32, Error> {
        Ok(self.mesh(index, generation)?.uvs().map_or(0, <[_]>::len) as u32)
    }

    /// Address of the colours (RGBA8, 4 bytes per vertex slot), 0 when the
    /// mesh has no colouring.
    pub fn colors_ptr(&self, index: u32, generation: u32) -> Result<u32, Error> {
        let c = self.scene.colors(self.id(index, generation)?)?;
        Ok(c.map_or(0, addr))
    }

    /// Length of the colours in bytes, 0 when the mesh has no colouring.
    pub fn colors_len(&self, index: u32, generation: u32) -> Result<u32, Error> {
        let c = self.scene.colors(self.id(index, generation)?)?;
        Ok(c.map_or(0, |c| len(c.len() * 4)))
    }

    /// Address of the values the colours are made from (`f32`, 1 per vertex
    /// slot: the colouring field's channel), 0 when the mesh has no
    /// colouring. They change only at the colours' dirty slots.
    pub fn values_ptr(&self, index: u32, generation: u32) -> Result<u32, Error> {
        let v = self.scene.color_values(self.id(index, generation)?)?;
        Ok(v.map_or(0, addr))
    }

    /// Length of the colour values in `f32`s, 0 when the mesh has no
    /// colouring.
    pub fn values_len(&self, index: u32, generation: u32) -> Result<u32, Error> {
        let v = self.scene.color_values(self.id(index, generation)?)?;
        Ok(v.map_or(0, |v| len(v.len())))
    }

    /// Address of the index buffer (`u32`, 3 per triangle slot; a removed
    /// triangle reads `0, 0, 0`).
    pub fn indices_ptr(&self, index: u32, generation: u32) -> Result<u32, Error> {
        Ok(addr(self.mesh(index, generation)?.triangles()))
    }

    /// Length of the index buffer in `u32`s.
    pub fn indices_len(&self, index: u32, generation: u32) -> Result<u32, Error> {
        Ok(len(self.mesh(index, generation)?.triangles().len() * 3))
    }

    // ---- Changes ----

    /// Everything that changed since the last call, as one message in the
    /// format of [`crate::render`]: `[record_count, record...]`.
    pub fn take_changes(&mut self) -> Vec<u32> {
        let changes = self.scene.take_changes();
        let mut records = Vec::with_capacity(changes.len());
        for c in changes {
            let dirty = if c.removed {
                self.render.remove(&c.id);
                Dirty::clean()
            } else {
                match self.scene.mesh(c.id) {
                    Ok(mesh) => {
                        // A restored mesh starts over, like a new one.
                        let state = self
                            .render
                            .entry(c.id)
                            .or_insert_with(|| RenderState::new(mesh));
                        let normals = self.scene.normals(c.id).ok().flatten();
                        let normals_len = normals.map_or(0, <[_]>::len);
                        // Out of memory: report everything rather than lose
                        // the other meshes' changes.
                        state
                            .apply(mesh, &c, normals_len)
                            .unwrap_or_else(|_| Dirty::all())
                    }
                    Err(_) => Dirty::all(),
                }
            };
            let uvs = self.scene.mesh(c.id).is_ok_and(|m| uvs_reported(m, &c));
            records.push(Record {
                mesh_index: c.id.index(),
                mesh_generation: c.id.generation(),
                added: c.added,
                removed: c.removed,
                restorable: c.removed && self.scene.is_restorable(c.id),
                placement_changed: c.placement_changed,
                dirty,
                uvs_reported: uvs,
                fields: c.fields_changed.iter().map(|f| f.raw()).collect(),
                sections: c.sections_changed.iter().map(|s| s.raw()).collect(),
            });
        }
        // Slots added and undone between two takes leave no change behind,
        // but the buffers grew: report the capacity change anyway.
        let mut reported: Vec<(u32, u32)> = records
            .iter()
            .map(|r| (r.mesh_index, r.mesh_generation))
            .collect();
        reported.sort_unstable();
        for (&id, state) in &mut self.render {
            if reported
                .binary_search(&(id.index(), id.generation()))
                .is_ok()
            {
                continue;
            }
            let Ok(mesh) = self.scene.mesh(id) else {
                continue;
            };
            let normals = self.scene.normals(id).ok().flatten();
            let normals_len = normals.map_or(0, <[_]>::len);
            if let Some(dirty) = state.unreported_growth(mesh, normals_len) {
                records.push(Record {
                    mesh_index: id.index(),
                    mesh_generation: id.generation(),
                    added: false,
                    removed: false,
                    restorable: false,
                    placement_changed: false,
                    uvs_reported: mesh.uvs().is_some(),
                    dirty,
                    fields: Vec::new(),
                    sections: Vec::new(),
                });
            }
        }
        let mut out = Vec::new();
        encode(&records, &mut out);
        out
    }

    // ---- History ----

    /// Undoes the latest step; false when there is nothing to undo.
    pub fn undo(&mut self) -> Result<bool, Error> {
        Ok(self.scene.undo()?)
    }

    /// Redoes the latest undone step; false when there is nothing to redo.
    pub fn redo(&mut self) -> Result<bool, Error> {
        Ok(self.scene.redo()?)
    }

    /// True if `undo` would undo something.
    pub fn can_undo(&self) -> bool {
        self.scene.can_undo()
    }

    /// True if `redo` would redo something.
    pub fn can_redo(&self) -> bool {
        self.scene.can_redo()
    }

    /// Drops every undo and redo step.
    pub fn clear_history(&mut self) -> Result<(), Error> {
        Ok(self.scene.clear_history()?)
    }

    /// Opens a history step, or one more nesting level of the open one.
    pub fn begin_step(&mut self) {
        self.scene.begin_step();
    }

    /// Ends one nesting level of the open step.
    pub fn end_step(&mut self) -> Result<(), Error> {
        Ok(self.scene.end_step()?)
    }

    /// Reverts and discards the open step.
    pub fn cancel_step(&mut self) -> Result<(), Error> {
        Ok(self.scene.cancel_step()?)
    }

    // ---- Geometry edits and repair ----

    /// Runs an edit op list on a mesh as one transaction: one undo step, and
    /// nothing changes when an op fails. `ops` and `values` are laid out as
    /// in [`crate::edit`]: `ops` is a `u32` sequence of op codes (0 set
    /// positions, 1 add vertices, 2 add triangles, 3 remove triangles, 4
    /// remove vertices, 5 set UVs) with their id lists, whose refs name ids added by
    /// earlier ops of the list; `values` holds the world positions, mapped
    /// to local with the placement inverse and the origin in `f64`, then
    /// stored as `f32`, and the UVs of op 5. Returns `[n, id × n]` per op: the new vertex or
    /// triangle ids, none for the other ops. The list is decoded and every
    /// ref checked before the mesh is touched.
    pub fn edit(
        &mut self,
        index: u32,
        generation: u32,
        ops: &[u32],
        values: &[f64],
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        let scene = &self.scene;
        // The id is live, so the mapping cannot fail.
        let decoded = decode_edit(ops, values, |q| {
            scene.world_to_local(id, q).unwrap_or([f64::NAN; 3])
        })?;
        let added = self.scene.edit(id, |tx| run_edit(tx, &decoded))?;
        encode_added(&added)
    }

    /// Removes the mesh's dead slots and renumbers the rest. Its next change
    /// record reports every buffer as "all", with the capacity change; its
    /// edits leave the history (placements stay).
    pub fn compact(&mut self, index: u32, generation: u32) -> Result<(), Error> {
        let id = self.id(index, generation)?;
        self.scene.compact(id)?;
        Ok(())
    }

    /// A read-only report of the mesh's defects, laid out as in
    /// [`crate::edit::encode_analysis`]. `weld_tolerance` (local units, before
    /// the placement's scale) groups coincident vertices; lists with a count
    /// keep their first `max_listed` entries. `self_intersection` checks
    /// for self-intersection with the mesh's BVH, which is brought up to
    /// date first unless the mesh is Manual (then `STALE_BVH` when stale).
    pub fn analyze(
        &mut self,
        index: u32,
        generation: u32,
        weld_tolerance: f32,
        self_intersection: bool,
        max_listed: u32,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let manual = self.scene.mode(id)? == UpdateMode::Manual;
        if self_intersection && !manual && !self.scene.is_current(id)? {
            self.scene.update(id)?;
        }
        let bvh = if self_intersection {
            Some(self.scene.bvh(id)?)
        } else {
            None
        };
        let options = AnalyzeOptions {
            weld_tolerance,
            self_intersection: bvh,
            max_listed: max_listed as usize,
        };
        let a = analyze(self.scene.mesh(id)?, &options).map_err(SceneError::from)?;
        Ok(encode_analysis(&a)?)
    }

    /// Scalar measurements of a mesh in world coordinates (its placement
    /// applied, so scaled meshes measure scaled): `[area, volume (NaN when
    /// the mesh is not watertight), closed (0/1), volume reason (0 closed,
    /// 1 open, 2 non-manifold), boundary loops, non-manifold edges,
    /// orientation conflicts]`. Area is always defined; volume is `None`
    /// with its reason unless the mesh has no boundary and no
    /// non-manifold edge. The volume is the absolute signed sum, so
    /// inconsistent winding only flags `orientation conflicts` without
    /// failing.
    pub fn measure(&self, index: u32, generation: u32) -> Result<Vec<f64>, Error> {
        let m = self.scene.measure_placed(self.id(index, generation)?)?;
        let reason = match m.volume_reason {
            None => 0.0,
            Some(VolumeReason::Open) => 1.0,
            Some(VolumeReason::NonManifold) => 2.0,
        };
        Ok(vec![
            m.area,
            m.volume.unwrap_or(f64::NAN),
            f64::from(u8::from(m.closed)),
            reason,
            count(m.boundary_loops),
            count(m.non_manifold_edges),
            count(m.orientation_conflicts),
        ])
    }

    /// The mesh's origin: its stored positions are relative to it, and
    /// `setPositions` takes positions with it added back.
    pub fn origin(&self, index: u32, generation: u32) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        Ok(self.scene.mesh(id)?.origin().to_vec())
    }

    /// Runs repair operations on a mesh, in order, in one transaction: one
    /// undo step, and nothing changes when an operation fails. `ops` (`f64`)
    /// and `loops` (`u32`, loop ids for fill holes) are laid out as in
    /// [`crate::edit`]; returns the reports concatenated in op order.
    /// Lengths and areas in the options are in local units, before the
    /// placement's scale.
    pub fn repair(
        &mut self,
        index: u32,
        generation: u32,
        ops: &[f64],
        loops: &[u32],
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let decoded = decode_repair(ops, loops)?;
        // One brush op alone runs as a dab, protected or not, its cost
        // following the brush region; in a list, the BVH cannot follow the ops
        // inside the one transaction, so they scan.
        if let Some((dab, vertices, weights, band)) = extract_dab(&decoded) {
            let r = self
                .scene
                .dab_protected_band(id, &dab, vertices, weights, band)?;
            return Ok(vec![count(r.moved), f64::from(r.max_displacement)]);
        }
        if let [RepairOp::CaptureRest { only_if_missing }] = decoded.as_slice() {
            self.scene.capture_rest_sizes(id, *only_if_missing)?;
            return Ok(Vec::new());
        }
        if decoded
            .iter()
            .any(|op| matches!(op, RepairOp::CaptureRest { .. }))
        {
            // Not an edit, so it cannot join a transaction.
            return Err(Error::invalid(
                "captureRestSizes must be the only operation in its list",
            ));
        }
        if let [RepairOp::CaptureStroke { only_if_missing }] = decoded.as_slice() {
            self.scene.capture_stroke_state(id, *only_if_missing)?;
            return Ok(Vec::new());
        }
        if decoded
            .iter()
            .any(|op| matches!(op, RepairOp::CaptureStroke { .. }))
        {
            // Not an edit, so it cannot join a transaction.
            return Err(Error::invalid(
                "captureStrokeState must be the only operation in its list",
            ));
        }
        if let [RepairOp::SplitStretched(o)] = decoded.as_slice() {
            self.scene.ensure_incidence(id)?;
            let r = self.scene.split_stretched(id, o)?;
            let mut out = Vec::new();
            put_stretch(&mut out, &r)?;
            return Ok(out);
        }
        if decoded
            .iter()
            .any(|op| matches!(op, RepairOp::SplitStretched(_)))
        {
            self.scene.ensure_incidence(id)?;
        }
        Ok(self.scene.edit(id, |tx| run_repair(tx, &decoded))?)
    }

    /// Live triangle-used vertex ids of a mesh inside the closed local-space
    /// sphere `(cx, cy, cz, radius)`, in ascending order. Brings a Lazy BVH up
    /// to date first; refuses a stale Manual BVH with `STALE_BVH`.
    pub fn vertices_in_sphere(
        &mut self,
        index: u32,
        generation: u32,
        cx: f64,
        cy: f64,
        cz: f64,
        radius: f64,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        Ok(self
            .scene
            .vertices_in_sphere(id, [cx as f32, cy as f32, cz as f32], radius as f32)?)
    }

    /// Joins mesh B into mesh A: copies B's geometry into A, bridges the two
    /// loops and removes B, as one undo step (undo restores A and brings B
    /// back); see [`Scene::join`]. `loops` holds the two loop
    /// ids (`min_vertex, len, hash low, hash high` each, A's first) and
    /// `max_edges` is NaN for the default. Returns `[vertices_added,
    /// triangles_added, bridge_triangles_added]`.
    pub fn join(
        &mut self,
        a_index: u32,
        a_generation: u32,
        b_index: u32,
        b_generation: u32,
        loops: &[u32],
        max_edges: f64,
    ) -> Result<Vec<f64>, Error> {
        let a = self.id(a_index, a_generation)?;
        let b = self.id(b_index, b_generation)?;
        let (pair, rest) = loops.as_chunks::<4>();
        if !rest.is_empty() || pair.len() != 2 {
            return Err(Error::invalid("join needs two loop ids"));
        }
        let options = JoinOptions {
            loop_a: loop_id(pair[0]),
            loop_b: loop_id(pair[1]),
            max_edges: count_opt(max_edges, "maxEdges")?,
        };
        // B's render state goes with its removal record, as for
        // `remove_mesh_undoable`.
        let r = self.scene.join(a, b, &options)?;
        Ok(vec![
            r.vertices_added as f64,
            r.triangles_added as f64,
            r.bridge_triangles_added as f64,
        ])
    }

    /// Aligns the moving mesh to the fixed one by ICP from their current
    /// placements; see [`Scene::register`]. The numeric options are NaN for
    /// their defaults. Returns `[rotation (9, row-major), translation (3),
    /// scale (3), rmse, iterations, status, applied, samples, inliers,
    /// conditioning, not applied, max rmse, then per undetermined motion
    /// translation (3), rotation (3), about (3) and its other rmse]`, with
    /// status 0 converged, 1 out of iterations, 2 underconstrained, 3 too few
    /// correspondences, 4 diverged; and not applied 0 applied, 1 not requested,
    /// 2 not converged, 3 rmse above the limit, 4 too few inliers.
    #[allow(clippy::too_many_arguments)]
    pub fn register(
        &mut self,
        moving_index: u32,
        moving_generation: u32,
        fixed_index: u32,
        fixed_generation: u32,
        max_iterations: f64,
        tolerance: f64,
        overlap: f64,
        max_distance: f64,
        max_samples: f64,
        apply: bool,
        max_rmse: f64,
    ) -> Result<Vec<f64>, Error> {
        let moving = self.id(moving_index, moving_generation)?;
        let fixed = self.id(fixed_index, fixed_generation)?;
        let defaults = RegisterOptions::default();
        let or = |v: f64, default: f64| if v.is_nan() { default } else { v };
        let options = RegisterOptions {
            icp: IcpOptions {
                max_iterations: crate::edit::count_opt(max_iterations, "maxIterations")?
                    .unwrap_or(defaults.icp.max_iterations),
                tolerance: or(tolerance, defaults.icp.tolerance),
                overlap: or(overlap, defaults.icp.overlap),
                // Rounded to the query's f32; a cutoff needs no more.
                max_distance: or(max_distance, f64::INFINITY) as f32,
            },
            max_samples: crate::edit::count_opt(max_samples, "maxSamples")?
                .unwrap_or(defaults.max_samples),
            apply,
            max_rmse: (!max_rmse.is_nan()).then_some(max_rmse),
        };
        let r = self.scene.register(moving, fixed, &options)?;
        let status = icp_status_code(r.status);
        let not_applied = match r.not_applied {
            None => 0.0,
            Some(NotApplied::NotRequested) => 1.0,
            Some(NotApplied::NotConverged) => 2.0,
            Some(NotApplied::RmseAboveLimit) => 3.0,
            Some(NotApplied::TooFewInliers) => 4.0,
        };
        let mut out = Vec::with_capacity(24 + 10 * r.undetermined.len());
        out.extend(r.placement.rotation.iter().flatten());
        out.extend(r.placement.translation);
        out.extend(r.placement.scale);
        out.extend([
            r.rmse,
            count(r.iterations),
            status,
            f64::from(u8::from(r.applied)),
            count(r.samples),
            count(r.inliers),
            r.conditioning,
            not_applied,
            r.max_rmse,
        ]);
        for m in &r.undetermined {
            out.extend(m.translation);
            out.extend(m.rotation);
            out.extend(m.about);
            out.push(m.alternative_rmse);
        }
        Ok(out)
    }

    /// Cuts a mesh with a world plane `[origin xyz, normal xyz]`: the `+`
    /// side stays on the mesh and the `-` side is added as a new mesh with
    /// the same placement. Returns `[kept index, kept generation, added
    /// index, added generation, kept triangles, removed triangles, cap
    /// triangles]`. With `cap` (the usual case) both openings are closed
    /// with flat, unfaired patches, across UV seams on a mesh with UVs;
    /// without it both parts keep the open rim.
    /// The cut and the add are one undo step: undo restores the mesh and
    /// removes the added one.
    pub fn cut_by_plane(
        &mut self,
        index: u32,
        generation: u32,
        plane: &[f64],
        cap: bool,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        let options = CutOptions {
            plane: parse_plane(plane)?,
            cap,
        };
        let r = self.scene.cut_by_plane(id, &options)?;
        let added = self.added(r.added, false)?;
        Ok(vec![
            index,
            generation,
            added[0],
            added[1],
            len(r.kept_triangles),
            len(r.removed_triangles),
            len(r.cap_triangles),
        ])
    }

    /// Combines mesh B into mesh A in the world and returns `[index,
    /// generation, kept triangles, triangles added]` of A (the result
    /// replaces A with A's placement; B is unchanged). `op` is 0 (union),
    /// 1 (subtract) or 2 (intersect). The boolean undoes in one step;
    /// combining a mesh with itself is a no-op for union and intersect,
    /// and clears it for subtract. Only positions and triangles cross
    /// over; the new vertices take channel defaults.
    pub fn boolean(
        &mut self,
        a_index: u32,
        a_generation: u32,
        b_index: u32,
        b_generation: u32,
        op: u8,
    ) -> Result<Vec<u32>, Error> {
        let a = self.id(a_index, a_generation)?;
        let b = self.id(b_index, b_generation)?;
        let op = match op {
            0 => BooleanOp::Union,
            1 => BooleanOp::Subtract,
            2 => BooleanOp::Intersect,
            _ => {
                return Err(Error::invalid(
                    "op must be 0 (union), 1 (subtract) or 2 (intersect)",
                ));
            }
        };
        let r = self.scene.boolean(a, b, op)?;
        Ok(vec![
            a_index,
            a_generation,
            len(r.kept_triangles),
            len(r.triangles_added),
        ])
    }

    /// Combines mesh B into mesh A in the world using full boolean options:
    /// `op` (0 union, 1 subtract, 2 intersect), `method` (0 manifold,
    /// 1 bvhSdf, 2 fhEdt), `coplanarTolerance` (`NaN` -> 1e-5), `snapSlivers`,
    /// `voxelPitch` (`NaN` -> auto), `maxTriangles` (`NaN` -> unbounded).
    /// Returns `[a_index, a_generation, kept_triangles, triangles_added]`.
    #[allow(clippy::too_many_arguments)]
    pub fn boolean_with_options(
        &mut self,
        a_index: u32,
        a_generation: u32,
        b_index: u32,
        b_generation: u32,
        op: u8,
        method: u8,
        coplanar_tolerance: f64,
        snap_slivers: bool,
        voxel_pitch: f64,
        max_triangles: f64,
    ) -> Result<Vec<u32>, Error> {
        let a = self.id(a_index, a_generation)?;
        let b = self.id(b_index, b_generation)?;
        let options = parse_boolean_options(
            op,
            method,
            coplanar_tolerance,
            snap_slivers,
            voxel_pitch,
            max_triangles,
        )?;
        let r = self.scene.boolean_with_options(a, b, &options)?;
        Ok(vec![
            a_index,
            a_generation,
            len(r.kept_triangles),
            len(r.triangles_added),
        ])
    }

    /// Evaluates an N-ary CSG boolean tree over `meshes` (`[idx0, gen0, ...]`)
    /// using prefix-encoded `tree` (`[0, leafIdx]` for Leaf, `[1, op, ...left, ...right]`
    /// for Op) and inserts the resulting mesh at identity placement, its
    /// origin the centre of the bounds of the meshes used, as one undo step
    /// (undo removes it). Returns `[added_index, added_generation]`.
    #[allow(clippy::too_many_arguments)]
    pub fn boolean_batch(
        &mut self,
        meshes: &[u32],
        tree: &[f64],
        op: u8,
        method: u8,
        coplanar_tolerance: f64,
        snap_slivers: bool,
        voxel_pitch: f64,
        max_triangles: f64,
    ) -> Result<Vec<u32>, Error> {
        let ids = self.ids(meshes)?;
        let options = parse_boolean_options(
            op,
            method,
            coplanar_tolerance,
            snap_slivers,
            voxel_pitch,
            max_triangles,
        )?;
        let mut cursor = 0usize;
        let node = decode_csg_node(tree, &mut cursor, 0)?;
        if cursor != tree.len() {
            return Err(Error::invalid("csg tree contains trailing data"));
        }
        let added = self.scene.boolean_batch(&ids, &node, &options)?;
        self.added(added, false)
    }

    /// Conformatively imprints a world-space polyline (`points`: flat `[x, y, z, ...]`)
    /// onto mesh `(index, generation)` and optionally trims the enclosed interior
    /// (`closed && delete_inside`) or exterior side of the component it lies on.
    /// Returns `[index, generation, triangles_added, triangles_removed, imprint_triangles,
    /// cut_triangles, removed_area, vertices_added, seam_edges]`.
    pub fn trim_by_curve(
        &mut self,
        index: u32,
        generation: u32,
        points: &[f64],
        closed: bool,
        delete_inside: bool,
        snap_tolerance: f64,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        if !points.len().is_multiple_of(3) {
            return Err(Error::invalid("points length must be a multiple of 3"));
        }
        let parsed_points: Vec<[f64; 3]> = points.as_chunks::<3>().0.to_vec();
        let snap = if snap_tolerance.is_nan() {
            0.0
        } else {
            snap_tolerance
        };
        let r = self.scene.trim_by_curve(
            id,
            &TrimCurveOptions {
                points: parsed_points,
                closed,
                delete_inside,
                snap_tolerance: snap,
            },
        )?;
        Ok(vec![
            f64::from(index),
            f64::from(generation),
            r.triangles_added as f64,
            r.triangles_removed as f64,
            r.imprint_triangles as f64,
            r.cut_triangles as f64,
            r.removed_area,
            r.vertices_added as f64,
            r.seam_edges as f64,
        ])
    }

    /// Aligns `moving` onto `fixed` from 3D landmark pairs (`source_points`,
    /// `target_points`, each flat `[x, y, z, ...]`). Updates `moving`'s placement
    /// and returns `[rotation(9), translation(3), scale(3), alignment_scale, rmse]`.
    #[allow(clippy::too_many_arguments)]
    pub fn align_by_landmarks(
        &mut self,
        moving_index: u32,
        moving_generation: u32,
        fixed_index: u32,
        fixed_generation: u32,
        source_points: &[f64],
        target_points: &[f64],
        allow_scaling: bool,
    ) -> Result<Vec<f64>, Error> {
        let moving = self.id(moving_index, moving_generation)?;
        let fixed = self.id(fixed_index, fixed_generation)?;
        if !source_points.len().is_multiple_of(3) || !target_points.len().is_multiple_of(3) {
            return Err(Error::invalid(
                "landmark coordinate arrays must be multiples of 3",
            ));
        }
        if source_points.len() != target_points.len() {
            return Err(Error::invalid(
                "source and target landmark counts must match",
            ));
        }
        let pairs: Vec<LandmarkPair> = source_points
            .as_chunks::<3>()
            .0
            .iter()
            .zip(target_points.as_chunks::<3>().0.iter())
            .map(|(&source, &target)| LandmarkPair { source, target })
            .collect();
        let alignment = self
            .scene
            .align_by_landmarks(moving, fixed, &pairs, allow_scaling)?;
        let placement = self.scene.placement(moving)?;
        let mut out = Vec::with_capacity(17);
        out.extend(placement.rotation.iter().flatten());
        out.extend(placement.translation);
        out.extend(placement.scale);
        out.push(alignment.scale);
        out.push(alignment.rmse);
        Ok(out)
    }

    /// Multi-stage coarse + fine registration of `moving` onto `fixed`.
    /// `coarse_method`: 0 none, 1 pcaCentroid, 2 orientedBoundingBox.
    /// `objective`: 0 pointToPoint, 1 pointToPlane, 2 symmetricNormal.
    /// Returns `[rotation(9), translation(3), scale(3), transform_col_major(16),
    /// final_rmse, iterations, status, converged]`.
    #[allow(clippy::too_many_arguments)]
    pub fn register_multistage(
        &mut self,
        moving_index: u32,
        moving_generation: u32,
        fixed_index: u32,
        fixed_generation: u32,
        coarse_method: u8,
        objective: u8,
        max_iterations: f64,
        tolerance: f64,
        overlap: f64,
        max_distance: f64,
        target_rmse: f64,
        sample_count: f64,
        anderson_window: f64,
    ) -> Result<Vec<f64>, Error> {
        let moving = self.id(moving_index, moving_generation)?;
        let fixed = self.id(fixed_index, fixed_generation)?;
        let coarse_method = match coarse_method {
            0 => CoarseAlignMethod::None,
            1 => CoarseAlignMethod::PcaCentroid,
            2 => CoarseAlignMethod::OrientedBoundingBox,
            _ => {
                return Err(Error::invalid(
                    "coarseMethod must be 0 (none), 1 (pcaCentroid) or 2 (orientedBoundingBox)",
                ));
            }
        };
        let objective = match objective {
            0 => IcpObjective::PointToPoint,
            1 => IcpObjective::PointToPlane,
            2 => IcpObjective::SymmetricNormal,
            _ => {
                return Err(Error::invalid(
                    "objective must be 0 (pointToPoint), 1 (pointToPlane) or 2 (symmetricNormal)",
                ));
            }
        };
        let defaults = RegistrationOptions::default();
        let or = |v: f64, default: f64| if v.is_nan() { default } else { v };
        let options = RegistrationOptions {
            coarse_method,
            objective,
            max_iterations: crate::edit::count_opt(max_iterations, "maxIterations")?
                .unwrap_or(defaults.max_iterations),
            tolerance: or(tolerance, defaults.tolerance),
            overlap: or(overlap, defaults.overlap),
            max_distance: if max_distance.is_nan() {
                None
            } else {
                Some(max_distance)
            },
            target_rmse: or(target_rmse, defaults.target_rmse),
            sample_count: crate::edit::count_opt(sample_count, "sampleCount")?
                .unwrap_or(defaults.sample_count),
            anderson_window: crate::edit::count_opt(anderson_window, "andersonWindow")?
                .unwrap_or(defaults.anderson_window),
        };
        let r = self.scene.register_multistage(moving, fixed, &options)?;
        let placement = self.scene.placement(moving)?;
        let mut out = Vec::with_capacity(35);
        out.extend(placement.rotation.iter().flatten());
        out.extend(placement.translation);
        out.extend(placement.scale);
        out.extend(r.transform);
        out.push(r.final_rmse);
        out.push(count(r.iterations));
        out.push(icp_status_code(r.status));
        out.push(f64::from(u8::from(r.converged)));
        Ok(out)
    }

    /// Detects the dominant reflection symmetry plane of `(index, generation)`
    /// in world coordinates.
    /// Returns `[normal_xyz(3), point_xyz(3), symmetry_error_rms, symmetry_score]`.
    pub fn detect_symmetry_plane(
        &mut self,
        index: u32,
        generation: u32,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let r = self.scene.detect_symmetry_plane(id)?;
        Ok(vec![
            r.plane_normal[0],
            r.plane_normal[1],
            r.plane_normal[2],
            r.plane_point[0],
            r.plane_point[1],
            r.plane_point[2],
            r.symmetry_error_rms,
            r.symmetry_score,
        ])
    }

    /// Printability overhang inspection in world coordinates.
    /// Returns `[total_support_area, overhang_face_count, max_overhang_angle_deg, ...vertex_angles_deg]`.
    pub fn overhangs(
        &self,
        index: u32,
        generation: u32,
        build_direction: &[f64],
        critical_angle_deg: f64,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let dir = vec3(build_direction, "buildDirection must have 3 values")?;
        let defaults = OverhangOptions::default();
        let angle = if critical_angle_deg.is_nan() {
            defaults.critical_angle_deg
        } else {
            critical_angle_deg as f32
        };
        let opts = OverhangOptions {
            build_direction: [dir[0] as f32, dir[1] as f32, dir[2] as f32],
            critical_angle_deg: angle,
        };
        let mut angles = Vec::new();
        let report = self.scene.overhangs(id, &opts, &mut angles)?;
        let mut out = Vec::with_capacity(3 + angles.len());
        out.push(report.total_support_area);
        out.push(count(report.overhang_face_count));
        out.push(f64::from(report.max_overhang_angle_deg));
        out.extend(angles.into_iter().map(f64::from));
        Ok(out)
    }

    /// Triangle quality evaluation in world coordinates.
    /// `metric`: 0 radiusRatio, 1 minAngleDeg, 2 aspectRatio.
    /// Returns `[min, max, mean, sliver_triangle_count, live_triangle_count, ...vertex_quality]`.
    pub fn triangle_quality(
        &self,
        index: u32,
        generation: u32,
        metric: u8,
        sliver_threshold: f64,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let metric = match metric {
            0 => QualityMetric::RadiusRatio,
            1 => QualityMetric::MinAngleDeg,
            2 => QualityMetric::AspectRatio,
            _ => {
                return Err(Error::invalid(
                    "metric must be 0 (radiusRatio), 1 (minAngleDeg) or 2 (aspectRatio)",
                ));
            }
        };
        let mut vertex_q = Vec::new();
        let report =
            self.scene
                .triangle_quality(id, metric, sliver_threshold as f32, &mut vertex_q)?;
        let mut out = Vec::with_capacity(5 + vertex_q.len());
        out.push(f64::from(report.min));
        out.push(f64::from(report.max));
        out.push(f64::from(report.mean));
        out.push(count(report.sliver_triangle_count));
        out.push(count(report.live_triangle_count));
        out.extend(vertex_q.into_iter().map(f64::from));
        Ok(out)
    }

    /// Vertex-to-surface Hausdorff/signed deviation from `source` to `target` in world units.
    /// Returns `[min, max, mean, rms, p95, sample_count, ...vertex_deviations]`.
    pub fn mesh_deviation(
        &mut self,
        source_index: u32,
        source_generation: u32,
        target_index: u32,
        target_generation: u32,
        signed: bool,
        max_distance: f64,
    ) -> Result<Vec<f64>, Error> {
        let source = self.id(source_index, source_generation)?;
        let target = self.id(target_index, target_generation)?;
        let opts = DeviationOptions {
            signed,
            max_distance: if max_distance.is_nan() {
                f32::INFINITY
            } else {
                max_distance as f32
            },
        };
        let mut values = Vec::new();
        let stats = self
            .scene
            .mesh_deviation(source, target, &opts, &mut values)?;
        let mut out = Vec::with_capacity(6 + values.len());
        out.push(f64::from(stats.min));
        out.push(f64::from(stats.max));
        out.push(f64::from(stats.mean));
        out.push(f64::from(stats.rms));
        out.push(f64::from(stats.p95));
        out.push(count(stats.sample_count));
        out.extend(values.into_iter().map(f64::from));
        Ok(out)
    }

    /// Traces the intrinsic shortest surface path from `sources` to `target_vertex`
    /// using vertex `distances` (aligned with `mesh.vertex_slots()`).
    /// Returns `[px, py, pz, triangle, u, v, w, ...]` (7 f64 values per point).
    pub fn shortest_surface_path(
        &self,
        index: u32,
        generation: u32,
        distances: &[f32],
        sources: &[u32],
        target_vertex: u32,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let mut path = Vec::new();
        self.scene
            .shortest_surface_path(id, distances, sources, target_vertex, &mut path)?;
        let mut out = Vec::with_capacity(path.len() * 7);
        for pt in path {
            out.extend(pt.position);
            out.push(f64::from(pt.triangle));
            out.push(f64::from(pt.barycentric[0]));
            out.push(f64::from(pt.barycentric[1]));
            out.push(f64::from(pt.barycentric[2]));
        }
        Ok(out)
    }

    // ---- Wave 5: Hole continuity (#210) & Dynamic topology (#212) ----

    /// Fills boundary loop `loop_index` with selectable continuity (`0` flat,
    /// `1` tangent G1, `2` curvature G2). `max_hole_perimeter`,
    /// `max_hole_vertices`, and `smooth_iterations` accept `NaN` for defaults.
    /// Returns `[loopIndex, filledVertices, filledTriangles, perimeter, maxDihedralAngleDeg, continuityCode, islandLoopsMerged, converged]`.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_hole_continuity(
        &mut self,
        index: u32,
        generation: u32,
        loop_index: u32,
        continuity: u8,
        max_hole_perimeter: f64,
        max_hole_vertices: f64,
        refine_patch: bool,
        smooth_iterations: f64,
        _detect_islands: bool,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let options = parse_fill_holes_continuity_options(
            continuity,
            max_hole_perimeter,
            max_hole_vertices,
            refine_patch,
            smooth_iterations,
        )?;
        let stats = self
            .scene
            .fill_hole_continuity(id, loop_index as usize, &options)?;
        Ok(encode_fill_stats(&stats))
    }

    /// Fills all eligible boundary holes with selectable continuity (`0` flat,
    /// `1` tangent G1, `2` curvature G2).
    /// Returns aggregated `[loopIndex, filledVertices, filledTriangles, perimeter, maxDihedralAngleDeg, continuityCode, islandLoopsMerged, converged]`.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_all_holes_continuity(
        &mut self,
        index: u32,
        generation: u32,
        continuity: u8,
        max_hole_perimeter: f64,
        max_hole_vertices: f64,
        refine_patch: bool,
        smooth_iterations: f64,
        _detect_islands: bool,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let options = parse_fill_holes_continuity_options(
            continuity,
            max_hole_perimeter,
            max_hole_vertices,
            refine_patch,
            smooth_iterations,
        )?;
        let stats = self.scene.fill_all_holes_continuity(id, &options)?;
        Ok(encode_fill_stats(&stats))
    }

    /// Bridges two boundary arcs (resolved from `loop_a`/`loop_b` vertex lists
    /// when non-empty, or by `loop_*_index` + `arc_*_start`/`arc_*_end`).
    /// `subdivisions` accepts `NaN` for automatic matching.
    /// Returns `[trianglesAdded, verticesAdded, resultingLoops, arcAVertices, arcBVertices]`.
    #[allow(clippy::too_many_arguments)]
    pub fn bridge_partial_arcs(
        &mut self,
        index: u32,
        generation: u32,
        loop_a: &[u32],
        loop_b: &[u32],
        loop_a_index: u32,
        arc_a_start: u32,
        arc_a_end: u32,
        loop_b_index: u32,
        arc_b_start: u32,
        arc_b_end: u32,
        twist_offset: i32,
        subdivisions: f64,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        let options = PartialBridgeOptions {
            loop_a_index: loop_a_index as usize,
            arc_a_range: (arc_a_start as usize, arc_a_end as usize),
            loop_b_index: loop_b_index as usize,
            arc_b_range: (arc_b_start as usize, arc_b_end as usize),
            twist_offset: twist_offset as isize,
            subdivisions: count_opt(subdivisions, "subdivisions")?,
        };
        let stats = self
            .scene
            .bridge_partial_arcs(id, loop_a, loop_b, &options)?;
        Ok(vec![
            len(stats.triangles_added),
            len(stats.vertices_added),
            len(stats.resulting_loops),
            len(stats.arc_a_vertices),
            len(stats.arc_b_vertices),
        ])
    }

    /// Applies localized dynamic topology remeshing inside sphere `(center, radius)`.
    /// `mode`: 0 split-collapse, 1 split-only, 2 brush-relative, 3 constant-detail.
    /// Returns `[splits, collapses, flips, smoothed, newCount, removedCount, ...newVertices, ...removedVertices]`.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_dyntopo(
        &mut self,
        index: u32,
        generation: u32,
        center: &[f64],
        radius: f64,
        mode: u8,
        detail_size: f64,
        min_detail_ratio: f64,
        max_detail_ratio: f64,
        equalize_valence: bool,
        smooth_tangent: bool,
        pinned: &[u32],
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let c = vec3(center, "center must have 3 values")?;
        if !c.iter().all(|v| v.is_finite()) || !radius.is_finite() {
            return Err(Error::invalid("center and radius must be finite"));
        }
        let options = parse_dyntopo_options(
            mode,
            detail_size,
            min_detail_ratio,
            max_detail_ratio,
            equalize_valence,
            smooth_tangent,
        )?;
        let patch = self.scene.apply_dyntopo_protected(
            id,
            [c[0] as f32, c[1] as f32, c[2] as f32],
            radius as f32,
            &options,
            pinned,
        )?;
        let mut out =
            Vec::with_capacity(6 + patch.new_vertices.len() + patch.removed_vertices.len());
        encode_dyntopo_patch(&mut out, &patch);
        Ok(out)
    }

    /// Applies localized dynamic topology remeshing followed by a brush dab
    /// encoded in `ops` (single brush op with optional `protect` and
    /// `brushModifiers` prefixes).
    /// Returns `[moved, maxDisplacement, splits, collapses, flips, smoothed, newCount, removedCount, ...newVertices, ...removedVertices]`.
    #[allow(clippy::too_many_arguments)]
    pub fn dyntopo_dab(
        &mut self,
        index: u32,
        generation: u32,
        ops: &[f64],
        mode: u8,
        detail_size: f64,
        min_detail_ratio: f64,
        max_detail_ratio: f64,
        equalize_valence: bool,
        smooth_tangent: bool,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let options = parse_dyntopo_options(
            mode,
            detail_size,
            min_detail_ratio,
            max_detail_ratio,
            equalize_valence,
            smooth_tangent,
        )?;
        let decoded = decode_repair(ops, &[])?;
        let Some((dab, vertices, weights, band)) = extract_dab(&decoded) else {
            return Err(Error::invalid(
                "dyntopoDab expects a single brush operation",
            ));
        };
        let (patch, report) = self
            .scene
            .dyntopo_dab_protected_band(id, &dab, &options, vertices, weights, band)?;
        let mut out =
            Vec::with_capacity(8 + patch.new_vertices.len() + patch.removed_vertices.len());
        out.push(count(report.moved));
        out.push(f64::from(report.max_displacement));
        encode_dyntopo_patch(&mut out, &patch);
        Ok(out)
    }

    // ---- Queries (world space) ----

    /// The nearest hit of a world ray over every mesh with
    /// `min_distance <= t <= max_distance`; `cull` is 0 (none), 1 (back
    /// faces) or 2 (front faces). Empty on a miss, else `[index,
    /// generation, triangle, point xyz, distance, barycentric uvw]`.
    pub fn raycast(
        &mut self,
        origin: &[f64],
        direction: &[f64],
        min_distance: f64,
        max_distance: f64,
        cull: u8,
    ) -> Result<Vec<f64>, Error> {
        let origin = vec3(origin, "origin must have 3 values")?;
        let dir = vec3(direction, "direction must have 3 values")?;
        let cull = match cull {
            0 => Cull::None,
            1 => Cull::Back,
            2 => Cull::Front,
            _ => {
                return Err(Error::invalid(
                    "cull must be 0 (none), 1 (back) or 2 (front)",
                ));
            }
        };
        let opts = SceneRayOptions {
            min_t: min_distance,
            max_t: max_distance,
            cull,
        };
        Ok(match self.scene.raycast(origin, dir, &opts)? {
            None => Vec::new(),
            Some(h) => {
                let mut out = vec![
                    f64::from(h.mesh.index()),
                    f64::from(h.mesh.generation()),
                    f64::from(h.hit.triangle),
                ];
                out.extend_from_slice(&h.point);
                out.push(h.t);
                out.extend(h.hit.barycentric.map(f64::from));
                out
            }
        })
    }

    /// The point of the scene nearest to world point `p` within
    /// `max_distance`. Empty when none, else `[index, generation, triangle,
    /// point xyz, distance]`.
    pub fn closest_point(&mut self, p: &[f64], max_distance: f64) -> Result<Vec<f64>, Error> {
        let p = vec3(p, "point must have 3 values")?;
        let opts = SceneClosestOptions { max_distance };
        let hit = self.scene.closest_point(p, &opts)?;
        Ok(hit.map_or_else(Vec::new, closest_record))
    }

    /// [`WasmScene::closest_point`] on one mesh.
    pub fn closest_point_on(
        &mut self,
        index: u32,
        generation: u32,
        p: &[f64],
        max_distance: f64,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let p = vec3(p, "point must have 3 values")?;
        let opts = SceneClosestOptions { max_distance };
        let hit = self.scene.closest_point_on(id, p, &opts)?;
        Ok(hit.map_or_else(Vec::new, closest_record))
    }

    /// The distance between two meshes' surfaces as placed. Empty when it
    /// exceeds `max_distance` or a mesh has no triangle, else `[distance,
    /// point on a xyz, point on b xyz, triangle of a, triangle of b]`.
    pub fn mesh_distance(
        &mut self,
        a_index: u32,
        a_generation: u32,
        b_index: u32,
        b_generation: u32,
        max_distance: f64,
    ) -> Result<Vec<f64>, Error> {
        let a = self.id(a_index, a_generation)?;
        let b = self.id(b_index, b_generation)?;
        Ok(match self.scene.mesh_distance(a, b, max_distance)? {
            None => Vec::new(),
            Some(d) => {
                let mut out = vec![d.distance];
                out.extend_from_slice(&d.point_a);
                out.extend_from_slice(&d.point_b);
                out.push(f64::from(d.triangle_a));
                out.push(f64::from(d.triangle_b));
                out
            }
        })
    }

    /// Whether two meshes intersect or touch as placed.
    pub fn intersects(
        &mut self,
        a_index: u32,
        a_generation: u32,
        b_index: u32,
        b_generation: u32,
    ) -> Result<bool, Error> {
        let a = self.id(a_index, a_generation)?;
        let b = self.id(b_index, b_generation)?;
        Ok(self.scene.intersects(a, b)?)
    }

    /// The alive triangles of the mesh that intersect or touch another of
    /// its triangles sharing no vertex id, ascending (see
    /// [`Scene::self_intersecting_triangles`]). The mesh's BVH is brought
    /// up to date first unless the mesh is Manual (then `STALE_BVH` when
    /// stale).
    pub fn self_intersecting_triangles(
        &mut self,
        index: u32,
        generation: u32,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        let mut out = Vec::new();
        self.scene.self_intersecting_triangles(id, &mut out)?;
        Ok(out)
    }

    /// The mesh pairs within `margin` of each other, of `pairs` (flat
    /// `[a index, a generation, b index, b generation, ...]`) or of the
    /// whole scene. `[count, contact...]`, each contact `[a index, a
    /// generation, b index, b generation, intersecting (0/1), distance,
    /// triangle count of a, of b, triangles of a..., triangles of b...]`.
    pub fn find_contacts(
        &mut self,
        margin: f64,
        pairs: Option<Vec<u32>>,
    ) -> Result<Vec<f64>, Error> {
        let pairs = match pairs {
            None => None,
            Some(flat) => {
                if !flat.len().is_multiple_of(4) {
                    return Err(Error::invalid("contact pairs need two meshes each"));
                }
                let ids = self.ids(&flat)?;
                Some(
                    ids.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&[a, b]| (a, b))
                        .collect(),
                )
            }
        };
        let mut found = Vec::new();
        self.scene
            .find_contacts(margin, &ContactFilter { pairs }, &mut found)?;
        let mut out = vec![count(found.len())];
        for c in &found {
            out.extend([
                f64::from(c.a.index()),
                f64::from(c.a.generation()),
                f64::from(c.b.index()),
                f64::from(c.b.generation()),
                f64::from(u8::from(c.intersecting)),
                c.distance,
                count(c.triangles_a.len()),
                count(c.triangles_b.len()),
            ]);
            out.extend(c.triangles_a.iter().map(|&t| f64::from(t)));
            out.extend(c.triangles_b.iter().map(|&t| f64::from(t)));
        }
        Ok(out)
    }

    // ---- Fields and colours ----

    /// Adds a signed distance field on mesh `on` to one or more target
    /// meshes, given as flat pairs `[on index, on generation, to index, to
    /// generation, ...]`, and returns its raw id. `sign` is 0 (auto), 1
    /// (pseudonormal) or 2 (winding).
    pub fn add_distance_field(
        &mut self,
        name: String,
        on_to: &[u32],
        max_distance: f64,
        sign: u8,
    ) -> Result<u32, Error> {
        let mut ids = self.ids(on_to)?;
        if ids.len() < 2 {
            return Err(Error::invalid("a field needs two meshes"));
        }
        let on = ids.remove(0);
        let desc = DistanceField {
            name,
            on,
            to: ids,
            max_distance,
            sign: parse_sign(sign)?,
        };
        Ok(self.scene.add_distance_field(desc)?.raw())
    }

    /// [`WasmScene::add_distance_field`] computing nothing yet: the field is
    /// due until stepped or updated; see
    /// [`Scene::add_distance_field_deferred`].
    pub fn add_distance_field_deferred(
        &mut self,
        name: String,
        on_to: &[u32],
        max_distance: f64,
        sign: u8,
    ) -> Result<u32, Error> {
        let mut ids = self.ids(on_to)?;
        if ids.len() < 2 {
            return Err(Error::invalid("a field needs two meshes"));
        }
        let on = ids.remove(0);
        let desc = DistanceField {
            name,
            on,
            to: ids,
            max_distance,
            sign: parse_sign(sign)?,
        };
        Ok(self.scene.add_distance_field_deferred(desc)?.raw())
    }

    /// Adds a per-vertex wall-thickness field on the mesh given as a flat
    /// pair `[index, generation]`, and returns its raw id: the distance to
    /// the opposite surface along each vertex's smooth normal, in world
    /// units. Open sheets and rims read NaN; hits beyond `max_distance`
    /// read `Infinity`.
    pub fn add_thickness_field(
        &mut self,
        name: String,
        on: &[u32],
        max_distance: f64,
    ) -> Result<u32, Error> {
        let ids = self.ids(on)?;
        if ids.len() != 1 {
            return Err(Error::invalid("a thickness field needs one mesh"));
        }
        let desc = ThicknessField {
            name,
            on: ids[0],
            max_distance,
        };
        Ok(self.scene.add_thickness_field(desc)?.raw())
    }

    /// [`WasmScene::add_thickness_field`] computing nothing yet: the field
    /// is due until stepped or updated; see
    /// [`Scene::add_thickness_field_deferred`].
    pub fn add_thickness_field_deferred(
        &mut self,
        name: String,
        on: &[u32],
        max_distance: f64,
    ) -> Result<u32, Error> {
        let ids = self.ids(on)?;
        if ids.len() != 1 {
            return Err(Error::invalid("a thickness field needs one mesh"));
        }
        let desc = ThicknessField {
            name,
            on: ids[0],
            max_distance,
        };
        Ok(self.scene.add_thickness_field_deferred(desc)?.raw())
    }

    /// Adds a per-vertex curvature field on the mesh given as a flat pair
    /// `[index, generation]`, and returns its raw id: the signed mean
    /// curvature at each vertex, in 1/world-units (positive on domes, about
    /// `1 / r` on a sphere of radius `r`; negative in bowls; zero on flats).
    /// Dead slots and unused vertices read NaN.
    pub fn add_curvature_field(&mut self, name: String, on: &[u32]) -> Result<u32, Error> {
        let ids = self.ids(on)?;
        if ids.len() != 1 {
            return Err(Error::invalid("a curvature field needs one mesh"));
        }
        let desc = CurvatureField { name, on: ids[0] };
        Ok(self.scene.add_curvature_field(desc)?.raw())
    }

    /// [`WasmScene::add_curvature_field`] computing nothing yet: the field
    /// is due until stepped or updated; see
    /// [`Scene::add_curvature_field_deferred`].
    pub fn add_curvature_field_deferred(&mut self, name: String, on: &[u32]) -> Result<u32, Error> {
        let ids = self.ids(on)?;
        if ids.len() != 1 {
            return Err(Error::invalid("a curvature field needs one mesh"));
        }
        let desc = CurvatureField { name, on: ids[0] };
        Ok(self.scene.add_curvature_field_deferred(desc)?.raw())
    }

    /// Removes a field never computed, reporting nothing; `false` (nothing
    /// done) when it was computed or colours its mesh. See
    /// [`Scene::withdraw_field`].
    pub fn withdraw_field(&mut self, field: u32) -> Result<bool, Error> {
        Ok(self.scene.withdraw_field(self.field(field)?)?)
    }

    /// What the sign data found out about each target, in index order: five
    /// numbers per target, `[mode, closed, consistent orientation,
    /// non-manifold vertices, degenerate triangles]` as in
    /// [`WasmScene::field_status`].
    pub fn field_target_status(&self, field: u32) -> Result<Vec<u32>, Error> {
        let all = self.scene.field_target_status(self.field(field)?)?;
        Ok(all
            .iter()
            .flat_map(|s| {
                [
                    sign_code(s.mode),
                    u32::from(s.closed),
                    u32::from(s.consistent_orientation),
                    len(s.non_manifold_vertices),
                    len(s.degenerate_triangles),
                ]
            })
            .collect())
    }

    /// The field's target meshes as flat `[index, generation, ...]` pairs,
    /// in target index order (see [`Scene::field_targets`]).
    pub fn field_targets(&self, field: u32) -> Result<Vec<u32>, Error> {
        Ok(self
            .scene
            .field_targets(self.field(field)?)?
            .iter()
            .flat_map(|m| [m.index(), m.generation()])
            .collect())
    }

    /// A copy of the field's nearest-target indices, one per value (see
    /// [`Scene::field_nearest`]): `u32::MAX` for none, including slots
    /// added since the field's last update.
    pub fn field_nearest(&self, field: u32) -> Result<Vec<u32>, Error> {
        let id = self.field(field)?;
        let slots = self.scene.field_values(id)?.len();
        let nearest = self.scene.field_nearest(id)?;
        let mut out = Vec::new();
        out.try_reserve_exact(slots)
            .map_err(|_| Error::from(fast_mesh_store::StoreError::OutOfMemory))?;
        out.extend((0..slots).map(|v| nearest.get(v).copied().unwrap_or(u32::MAX)));
        Ok(out)
    }

    /// Removes a field (and a colouring by it).
    pub fn remove_field(&mut self, field: u32) -> Result<(), Error> {
        Ok(self.scene.remove_field(self.field(field)?)?)
    }

    /// Address of the field's values (`f32`, one per vertex slot of its
    /// mesh).
    pub fn field_values_ptr(&self, field: u32) -> Result<u32, Error> {
        Ok(addr(self.scene.field_values(self.field(field)?)?))
    }

    /// Length of the field's values in `f32`s.
    pub fn field_values_len(&self, field: u32) -> Result<u32, Error> {
        Ok(len(self.scene.field_values(self.field(field)?)?.len()))
    }

    /// `[current (0/1), sign mode (1 pseudonormal, 2 winding), closed,
    /// consistent orientation, non-manifold vertices, degenerate
    /// triangles]`, the sign part over every target together (see
    /// [`fast_mesh_scene::FieldStatus::sign`]).
    pub fn field_status(&self, field: u32) -> Result<Vec<u32>, Error> {
        let s = self.scene.field_status(self.field(field)?)?;
        let sign = s.sign;
        Ok(vec![
            u32::from(s.current),
            sign_code(sign.mode),
            u32::from(sign.closed),
            u32::from(sign.consistent_orientation),
            len(sign.non_manifold_vertices),
            len(sign.degenerate_triangles),
        ])
    }

    /// Colours a mesh from one of its fields; see [`parse_rule`] for `map`
    /// and `rule`. A display setting: not recorded in history.
    pub fn set_coloring(
        &mut self,
        index: u32,
        generation: u32,
        field: u32,
        map: u8,
        rule: &[f32],
    ) -> Result<(), Error> {
        let id = self.id(index, generation)?;
        let coloring = Coloring {
            field: self.field(field)?,
            rule: parse_rule(map, rule)?,
        };
        Ok(self.scene.set_coloring(id, Some(coloring))?)
    }

    /// Stops colouring a mesh. Not recorded in history.
    pub fn clear_coloring(&mut self, index: u32, generation: u32) -> Result<(), Error> {
        let id = self.id(index, generation)?;
        Ok(self.scene.set_coloring(id, None)?)
    }

    /// Starts (a budget of `points`) or ends (`undefined`) an interaction; see
    /// [`Scene::set_field_interaction`].
    pub fn set_field_interaction(&mut self, points: Option<u32>) -> Result<(), Error> {
        Ok(self
            .scene
            .set_field_interaction(points.map(|p| p as usize))?)
    }

    // ---- Stepped field updates ----

    /// Defers Eager field updates to [`WasmScene::field_step`]; see
    /// [`Scene::set_defer_fields`].
    pub fn set_defer_fields(&mut self, on: bool) {
        self.scene.set_defer_fields(on);
    }

    /// Computes up to `points` candidate vertices of the due Eager fields,
    /// opening a unit first if none is open; see [`Scene::field_step`].
    /// Returns 0 (idle: nothing was due), 1 (pending) or 2 (done: the unit
    /// was applied). An error also ends the unit: the fields that could be
    /// applied were.
    pub fn field_step(&mut self, points: u32) -> Result<u8, Error> {
        let budget = FieldBudget::Points(points as usize);
        Ok(match self.scene.field_step(budget)? {
            StepState::Idle => 0,
            StepState::Pending => 1,
            StepState::Done => 2,
            // A state added later: stop stepping (an open unit is dropped
            // by the next call that changes the scene).
            _ => 2,
        })
    }

    /// [`WasmScene::field_step`] for one field alone; see
    /// [`Scene::field_step_field`]. Returns 0 (idle: the field is current),
    /// 1 (pending) or 2 (done).
    pub fn field_step_field(&mut self, field: u32, points: u32) -> Result<u8, Error> {
        let budget = FieldBudget::Points(points as usize);
        Ok(
            match self.scene.field_step_field(self.field(field)?, budget)? {
                StepState::Idle => 0,
                StepState::Pending => 1,
                StepState::Done => 2,
                _ => 2,
            },
        )
    }

    /// Drops the open step unit, if any; see [`Scene::abandon_field_step`].
    pub fn abandon_field_step(&mut self) {
        self.scene.abandon_field_step();
    }

    /// The meshes of the open step unit's fields as flat `[index,
    /// generation, ...]` pairs, each once; empty when no unit is open.
    pub fn field_step_meshes(&self) -> Vec<u32> {
        self.scene
            .field_step_meshes()
            .into_iter()
            .flat_map(|m| [m.index(), m.generation()])
            .collect()
    }

    // ---- Sections ----

    /// Adds a section cutting `meshes` (flat pairs) with the world plane
    /// `[origin xyz, normal xyz]`, and returns its raw id.
    pub fn add_section(&mut self, plane: &[f64], meshes: &[u32], fill: bool) -> Result<u32, Error> {
        let desc = SectionDesc {
            plane: parse_plane(plane)?,
            meshes: self.ids(meshes)?,
            fill,
        };
        Ok(self.scene.add_section(desc)?.raw())
    }

    /// Removes a section.
    pub fn remove_section(&mut self, section: u32) -> Result<(), Error> {
        Ok(self.scene.remove_section(self.section_id(section)?)?)
    }

    /// Moves a section's plane, `[origin xyz, normal xyz]`.
    pub fn set_section_plane(&mut self, section: u32, plane: &[f64]) -> Result<(), Error> {
        let id = self.section_id(section)?;
        Ok(self.scene.set_section_plane(id, parse_plane(plane)?)?)
    }

    /// A section's plane as last set, `[origin xyz, normal xyz]`, the normal
    /// normalised.
    pub fn section_desc_plane(&self, section: u32) -> Result<Vec<f64>, Error> {
        let plane = self.scene.section_desc(self.section_id(section)?)?.plane;
        Ok(plane.origin().into_iter().chain(plane.normal()).collect())
    }

    /// Changes the meshes a section cuts (flat pairs).
    pub fn set_section_meshes(&mut self, section: u32, meshes: &[u32]) -> Result<(), Error> {
        let id = self.section_id(section)?;
        Ok(self.scene.set_section_meshes(id, self.ids(meshes)?)?)
    }

    /// Turns a section's fill on or off.
    pub fn set_section_fill(&mut self, section: u32, fill: bool) -> Result<(), Error> {
        let id = self.section_id(section)?;
        Ok(self.scene.set_section_fill(id, fill)?)
    }

    /// Computes every stale section entry.
    pub fn update_sections(&mut self) -> Result<(), Error> {
        Ok(self.scene.update_sections()?)
    }

    /// A section's result for a mesh, in world coordinates: `[status (0
    /// current, 1 stale), length, area, contour count, contour..., has fill
    /// (0/1), fill]`. Each contour is `[closed (0/1), point count, xyz...]`;
    /// the fill is `[point count, xyz..., triangle count, abc...]`.
    pub fn section(&self, section: u32, index: u32, generation: u32) -> Result<Vec<f64>, Error> {
        let id = self.section_id(section)?;
        let mesh = self.id(index, generation)?;
        let r = self.scene.section(id, mesh)?;
        let status = match self.scene.section_status(id, mesh)? {
            SectionStatus::Current => 0.0,
            SectionStatus::Stale => 1.0,
        };
        let mut out = vec![status, r.length, r.area, count(r.contours.len())];
        for c in &r.contours {
            out.push(f64::from(u8::from(c.closed)));
            out.push(count(c.points.len()));
            out.extend(c.points.iter().flatten());
        }
        // The fill is in the frame of the plane it was computed with.
        match (&r.fill, self.scene.section_plane(id, mesh)?) {
            (Some(fill), Some(plane)) => {
                out.push(1.0);
                out.push(count(fill.points.len()));
                out.extend(fill.points.iter().flat_map(|&q| plane.from_uv(q)));
                out.push(count(fill.triangles.len()));
                out.extend(fill.triangles.iter().flatten().map(|&i| f64::from(i)));
            }
            _ => out.push(0.0),
        }
        Ok(out)
    }

    /// The closest pair between two meshes' contours in a section. Empty
    /// when there is none, else `[distance, point on a xyz, point on b
    /// xyz]`.
    pub fn section_gap(
        &self,
        section: u32,
        a_index: u32,
        a_generation: u32,
        b_index: u32,
        b_generation: u32,
    ) -> Result<Vec<f64>, Error> {
        let id = self.section_id(section)?;
        let a = self.id(a_index, a_generation)?;
        let b = self.id(b_index, b_generation)?;
        Ok(match self.scene.section_gap(id, a, b)? {
            None => Vec::new(),
            Some(g) => {
                let mut out = vec![g.distance];
                out.extend_from_slice(&g.a);
                out.extend_from_slice(&g.b);
                out
            }
        })
    }

    // -----------------------------------------------------------------------
    // Wave 6 — #233 Face Selection
    // -----------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub fn select_faces_screen(
        &mut self,
        index: u32,
        generation: u32,
        shape_kind: u8,
        shape_data: &[f64],
        view_proj16: &[f64],
        camera_origin3: &[f64],
        camera_forward3: &[f64],
        is_orthographic: bool,
        visible_only: bool,
        reject_backfaces: bool,
        include_touching: bool,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        let shape = crate::wave6::parse_screen_shape(shape_kind, shape_data)?;
        let (camera, opts) = crate::wave6::parse_screen_options(
            view_proj16,
            camera_origin3,
            camera_forward3,
            is_orthographic,
            visible_only,
            reject_backfaces,
            include_touching,
        )?;
        Ok(self
            .scene
            .select_faces_screen_from(id, &shape, &camera, &opts)?)
    }

    pub fn select_faces_in_sphere(
        &mut self,
        index: u32,
        generation: u32,
        center3: &[f64],
        radius: f64,
        normal3: Option<Vec<f64>>,
        cone_angle_rad: f64,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        let center = crate::wave6::vec3_f64(center3, "center")?;
        let cone = match normal3 {
            None => None,
            Some(dir) => {
                if !cone_angle_rad.is_finite() {
                    return Err(Error::invalid("coneAngleRad must be finite"));
                }
                Some((crate::wave6::vec3_f64(&dir, "normal")?, cone_angle_rad))
            }
        };
        Ok(self
            .scene
            .select_faces_in_sphere_world(id, center, radius, cone)?)
    }

    pub fn select_by_crease_angle(
        &self,
        index: u32,
        generation: u32,
        seed_tri: u32,
        max_dihedral_rad: f64,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        Ok(self
            .scene
            .select_by_crease_angle(id, seed_tri, max_dihedral_rad)?)
    }

    pub fn select_connected_shell(
        &self,
        index: u32,
        generation: u32,
        seed_tri: u32,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        Ok(self.scene.select_connected_shell(id, seed_tri)?)
    }

    pub fn select_small_face_islands(
        &self,
        index: u32,
        generation: u32,
        max_face_count: f64,
        max_area: f64,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        let max_faces = crate::wave6::req_count(max_face_count, "maxFaceCount")?;
        let max_area = if max_area.is_nan() {
            None
        } else {
            Some(max_area)
        };
        Ok(self
            .scene
            .select_small_face_islands(id, max_faces, max_area)?)
    }

    pub fn grow_face_selection(
        &self,
        index: u32,
        generation: u32,
        selected_tris: &[u32],
        rings: f64,
        adjacency: u8,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        let rings = crate::wave6::req_count(rings, "rings")?;
        let adj = crate::wave6::parse_adjacency(adjacency)?;
        Ok(self
            .scene
            .grow_face_selection(id, selected_tris, rings, adj)?)
    }

    pub fn shrink_face_selection(
        &self,
        index: u32,
        generation: u32,
        selected_tris: &[u32],
        rings: f64,
        adjacency: u8,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        let rings = crate::wave6::req_count(rings, "rings")?;
        let adj = crate::wave6::parse_adjacency(adjacency)?;
        Ok(self
            .scene
            .shrink_face_selection(id, selected_tris, rings, adj)?)
    }

    pub fn fill_selection_holes(
        &self,
        index: u32,
        generation: u32,
        selected_tris: &[u32],
        max_hole_tris: f64,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        let max_holes = crate::wave6::req_count(max_hole_tris, "maxHoleTris")?;
        Ok(self
            .scene
            .fill_selection_holes(id, selected_tris, max_holes)?)
    }

    pub fn smooth_selection_boundary(
        &self,
        index: u32,
        generation: u32,
        selected_tris: &[u32],
        iterations: f64,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        let iters = crate::wave6::req_count(iterations, "iterations")?;
        Ok(self
            .scene
            .smooth_selection_boundary(id, selected_tris, iters)?)
    }

    pub fn face_selection_to_vertex_mask(
        &self,
        index: u32,
        generation: u32,
        selected_tris: &[u32],
    ) -> Result<Vec<f32>, Error> {
        let id = self.id(index, generation)?;
        Ok(self
            .scene
            .face_selection_to_vertex_mask(id, selected_tris)?)
    }

    pub fn vertex_mask_to_face_selection(
        &self,
        index: u32,
        generation: u32,
        mask: &[f32],
        threshold: f32,
        require_all_vertices: bool,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        Ok(self
            .scene
            .vertex_mask_to_face_selection(id, mask, threshold, require_all_vertices)?)
    }

    pub fn delete_selected_faces(
        &mut self,
        index: u32,
        generation: u32,
        selected_tris: &[u32],
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let report = self.scene.delete_selected_faces(id, selected_tris)?;
        Ok(crate::wave6::encode_face_edit_report(&report))
    }

    /// Copies `selected_tris` into a new mesh, one undo step (undo removes
    /// it). Returns `[index, generation]`.
    pub fn extract_selected_faces(
        &mut self,
        index: u32,
        generation: u32,
        selected_tris: &[u32],
        normals: bool,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        let new_id = self.scene.extract_selected_faces(id, selected_tris)?;
        self.added(new_id, normals)
    }

    /// Moves `selected_tris` into a new mesh, one undo step (undo restores the
    /// mesh and removes the new one). Returns `[index, generation,
    /// removedTriangles, removedVertices]`.
    pub fn split_selected_faces(
        &mut self,
        index: u32,
        generation: u32,
        selected_tris: &[u32],
        normals: bool,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        let (new_id, report) = self.scene.split_selected_faces(id, selected_tris)?;
        let mut out = self.added(new_id, normals)?;
        out.push(report.removed_triangles as u32);
        out.push(report.removed_vertices as u32);
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Wave 6 — #193 Free-Form Deformation
    // -----------------------------------------------------------------------

    pub fn build_ffd_lattice(
        &self,
        index: u32,
        generation: u32,
        dims: &[u32],
        basis: u8,
        padding: f64,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let [nx, ny, nz]: [u32; 3] = dims
            .try_into()
            .map_err(|_| Error::invalid("dims must be [nx, ny, nz]"))?;
        let basis = crate::wave6::parse_ffd_basis(basis)?;
        let lattice = self.scene.build_ffd_lattice(
            id,
            [nx as usize, ny as usize, nz as usize],
            basis,
            padding,
        )?;
        crate::wave6::encode_ffd_lattice(&lattice)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_ffd(
        &mut self,
        index: u32,
        generation: u32,
        dims: &[u32],
        basis: u8,
        frame15: &[f64],
        offsets: &[f64],
        outside_mode: u8,
        region_center: Option<Vec<f64>>,
        region_radius: f64,
        vertices: Option<Vec<u32>>,
        protected_verts: Option<Vec<u32>>,
        protection_weights: Option<Vec<f32>>,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        crate::wave6::check_live_mesh(&self.scene, id)?;
        let op = crate::wave6::decode_ffd_op(
            dims,
            basis,
            frame15,
            offsets,
            outside_mode,
            region_center,
            region_radius,
            vertices,
        )?;
        let rep = crate::wave6::apply_freeform_with_protection(
            &mut self.scene,
            id,
            &op,
            protected_verts,
            protection_weights,
        )?;
        Ok(crate::wave6::encode_freeform_report(&rep))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_arap(
        &mut self,
        index: u32,
        generation: u32,
        anchor_verts: &[u32],
        anchor_pos: &[f64],
        anchor_has_pos: &[u8],
        handle_verts: &[u32],
        handle_targets: &[f64],
        weights: u8,
        solver: u8,
        iterations: f64,
        tolerance: f64,
        region_center: Option<Vec<f64>>,
        region_radius: f64,
        vertices: Option<Vec<u32>>,
        protected_verts: Option<Vec<u32>>,
        protection_weights: Option<Vec<f32>>,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        crate::wave6::check_live_mesh(&self.scene, id)?;
        let op = crate::wave6::decode_arap_op(
            anchor_verts,
            anchor_pos,
            anchor_has_pos,
            handle_verts,
            handle_targets,
            weights,
            solver,
            iterations,
            tolerance,
            region_center,
            region_radius,
            vertices,
        )?;
        let rep = crate::wave6::apply_freeform_with_protection(
            &mut self.scene,
            id,
            &op,
            protected_verts,
            protection_weights,
        )?;
        Ok(crate::wave6::encode_freeform_report(&rep))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_kelvinlet(
        &mut self,
        index: u32,
        generation: u32,
        center3: &[f64],
        load_kind: u8,
        load_data: &[f64],
        epsilon: f64,
        nu: f64,
        mu: f64,
        scale_mode: u8,
        scale_alpha: f64,
        sym_x: bool,
        sym_y: bool,
        sym_z: bool,
        region_center: Option<Vec<f64>>,
        region_radius: f64,
        vertices: Option<Vec<u32>>,
        protected_verts: Option<Vec<u32>>,
        protection_weights: Option<Vec<f32>>,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        crate::wave6::check_live_mesh(&self.scene, id)?;
        let op = crate::wave6::decode_kelvinlet_op(
            center3,
            load_kind,
            load_data,
            epsilon,
            nu,
            mu,
            scale_mode,
            scale_alpha,
            sym_x,
            sym_y,
            sym_z,
            region_center,
            region_radius,
            vertices,
        )?;
        let rep = crate::wave6::apply_freeform_with_protection(
            &mut self.scene,
            id,
            &op,
            protected_verts,
            protection_weights,
        )?;
        Ok(crate::wave6::encode_freeform_report(&rep))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_xpbd(
        &mut self,
        index: u32,
        generation: u32,
        substeps: f64,
        iterations: f64,
        dt: f64,
        stretch_compliance: f64,
        bend_compliance: f64,
        volume_compliance: f64,
        damping: f64,
        gravity3: &[f64],
        pin_verts: &[u32],
        pin_targets: &[f64],
        handle_verts: &[u32],
        handle_targets: &[f64],
        handle_compliances: &[f64],
        self_collision: bool,
        collision_thickness: f64,
        protected_verts: Option<Vec<u32>>,
        protection_weights: Option<Vec<f32>>,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        crate::wave6::check_live_mesh(&self.scene, id)?;
        let op = crate::wave6::decode_xpbd_op(
            substeps,
            iterations,
            dt,
            stretch_compliance,
            bend_compliance,
            volume_compliance,
            damping,
            gravity3,
            pin_verts,
            pin_targets,
            handle_verts,
            handle_targets,
            handle_compliances,
            self_collision,
            collision_thickness,
        )?;
        let rep = crate::wave6::apply_freeform_with_protection(
            &mut self.scene,
            id,
            &op,
            protected_verts,
            protection_weights,
        )?;
        Ok(crate::wave6::encode_freeform_report(&rep))
    }

    // -----------------------------------------------------------------------
    // Wave 6 — #194 Inter-Object Collision
    // -----------------------------------------------------------------------

    pub fn set_default_collision_rule(&mut self, rule_flat16: &[f64]) -> Result<(), Error> {
        let rule = crate::wave6::decode_collision_rule(rule_flat16)?;
        Ok(self.scene.set_default_collision_rule(rule)?)
    }

    pub fn default_collision_rule(&self) -> Vec<f64> {
        crate::wave6::encode_collision_rule(self.scene.default_collision_rule())
    }

    pub fn set_collision_rule(
        &mut self,
        a_index: u32,
        a_generation: u32,
        b_index: u32,
        b_generation: u32,
        rule_flat16: &[f64],
    ) -> Result<(), Error> {
        let a = self.id(a_index, a_generation)?;
        let b = self.id(b_index, b_generation)?;
        let rule = crate::wave6::decode_collision_rule(rule_flat16)?;
        Ok(self.scene.set_collision_rule(a, b, rule)?)
    }

    pub fn clear_collision_rule(
        &mut self,
        a_index: u32,
        a_generation: u32,
        b_index: u32,
        b_generation: u32,
    ) -> Result<bool, Error> {
        let a = self.id(a_index, a_generation)?;
        let b = self.id(b_index, b_generation)?;
        Ok(self.scene.clear_collision_rule(a, b)?)
    }

    pub fn collision_rule(
        &self,
        a_index: u32,
        a_generation: u32,
        b_index: u32,
        b_generation: u32,
    ) -> Result<Vec<f64>, Error> {
        let a = self.id(a_index, a_generation)?;
        let b = self.id(b_index, b_generation)?;
        let rule = self.scene.collision_rule(a, b)?;
        Ok(crate::wave6::encode_collision_rule(&rule))
    }

    pub fn has_active_collision_response(&self, index: u32, generation: u32) -> bool {
        self.id(index, generation)
            .is_ok_and(|id| self.scene.has_active_collision_response(id))
    }

    pub fn move_with_collision(
        &mut self,
        index: u32,
        generation: u32,
        rotation: &[f64],
        translation: &[f64],
        scale: Option<Vec<f64>>,
        is_release: bool,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let target = crate::wave6::parse_placement(rotation, translation, scale)?;
        let report = self.scene.move_with_collision(id, target, is_release)?;
        crate::wave6::encode_collision_step_report(&self.scene, id, &report)
    }

    pub fn commit_collision_interaction(
        &mut self,
        index: u32,
        generation: u32,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let report = self.scene.commit_collision_interaction(id)?;
        crate::wave6::encode_collision_step_report(&self.scene, id, &report)
    }

    // -----------------------------------------------------------------------
    // Wave 6 — #234 Primitive Fit, Contours & Point Cloud
    // -----------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub fn fit_primitive(
        &self,
        index: u32,
        generation: u32,
        subset_tris: Option<Vec<u32>>,
        kind: i32,
        inlier_threshold: f64,
        ransac_iterations: f64,
        lm_iterations: f64,
        seed: f64,
        min_inlier_ratio: f64,
        normal_weight: f64,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let kind = crate::wave6::parse_primitive_kind(kind)?;
        let opts = crate::wave6::parse_fit_options(
            inlier_threshold,
            ransac_iterations,
            lm_iterations,
            seed,
            min_inlier_ratio,
            normal_weight,
        )?;
        let rep = self
            .scene
            .fit_primitive_world(id, subset_tris.as_deref(), kind, &opts)?;
        crate::wave6::encode_fit_report(&rep)
    }

    pub fn align_to_fitted_primitive(
        &mut self,
        index: u32,
        generation: u32,
        prim9: &[f64],
        target_axis3: &[f64],
        target_origin3: &[f64],
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let prim = crate::wave6::decode_fitted_primitive(prim9)?;
        let axis = crate::wave6::vec3_f64(target_axis3, "targetAxis")?;
        let origin = crate::wave6::vec3_f64(target_origin3, "targetOrigin")?;
        let placement = self
            .scene
            .align_to_fitted_primitive(id, &prim, axis, origin)?;
        Ok(crate::wave6::encode_placement(&placement))
    }

    pub fn slice_contours(
        &self,
        index: u32,
        generation: u32,
        origin3: &[f64],
        normal3: &[f64],
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let origin = crate::wave6::vec3_f64(origin3, "origin")?;
        let normal = crate::wave6::vec3_f64(normal3, "normal")?;
        let slice = self.scene.slice_contours_world(id, origin, normal)?;
        crate::wave6::encode_contour_slices(&[slice])
    }

    pub fn slice_contour_stack(
        &self,
        index: u32,
        generation: u32,
        origin3: &[f64],
        normal3: &[f64],
        spacing: f64,
        count: f64,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let origin = crate::wave6::vec3_f64(origin3, "origin")?;
        let normal = crate::wave6::vec3_f64(normal3, "normal")?;
        let n = crate::wave6::req_count(count, "count")?;
        let slices = self
            .scene
            .slice_contour_stack_world(id, origin, normal, spacing, n)?;
        crate::wave6::encode_contour_slices(&slices)
    }

    /// Reconstructs a surface from a packed cloud as a new mesh, one undo step
    /// (undo removes it). The cloud's positions are relative to `origin` (a
    /// world point, zero by default), which is the mesh's origin. Returns
    /// `[index, generation]`.
    #[allow(clippy::too_many_arguments)]
    pub fn reconstruct_point_cloud(
        &mut self,
        packed_cloud: &[f32],
        method: u8,
        voxel_size: f64,
        ball_radius: f64,
        k_neighbors: f64,
        normals: bool,
        origin: Option<Vec<f64>>,
    ) -> Result<Vec<u32>, Error> {
        let cloud = crate::wave6::decode_point_cloud_f32(packed_cloud)?;
        let opts =
            crate::wave6::parse_reconstruct_options(method, voxel_size, ball_radius, k_neighbors)?;
        let origin = match origin {
            Some(o) => crate::wave6::vec3_f64(&o, "origin")?,
            None => [0.0; 3],
        };
        let id = self
            .scene
            .reconstruct_point_cloud_at(&cloud, origin, &opts)?;
        self.added(id, normals)
    }

    // -----------------------------------------------------------------------
    // Wave 6 — #236 Hollow, Drain Holes & Place Mesh
    // -----------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub fn hollow_mesh(
        &mut self,
        index: u32,
        generation: u32,
        wall_thickness: f64,
        backend: u8,
        voxel_size: f64,
        max_voxels: f64,
        max_cavity_triangles: f64,
        min_cavity_volume: f64,
        modulation_verts: Option<Vec<u32>>,
        modulation_weights: Option<Vec<f32>>,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let opts = crate::wave6::parse_hollow_options(
            wall_thickness,
            backend,
            voxel_size,
            max_voxels,
            max_cavity_triangles,
            min_cavity_volume,
            modulation_verts,
            modulation_weights,
        )?;
        let rep = self.scene.hollow_mesh(id, &opts)?;
        Ok(crate::wave6::encode_hollow_report(&rep))
    }

    /// Punches a drain hole, adding its plug as a new mesh when asked to; the
    /// hole and the plug are one undo step (undo fills the hole and removes
    /// the plug).
    #[allow(clippy::too_many_arguments)]
    pub fn punch_drain_hole(
        &mut self,
        index: u32,
        generation: u32,
        surface_point3: &[f64],
        surface_normal3: &[f64],
        diameter: f64,
        depth: f64,
        outer_chamfer: f64,
        taper_angle_deg: f64,
        segments: u32,
        boolean_method: u8,
        fallback_bvh_sdf: bool,
        generate_plug: bool,
        plug_clearance: f64,
        plug_normals: bool,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let (point, spec, opts) = crate::wave6::parse_drain_hole(
            surface_point3,
            surface_normal3,
            diameter,
            depth,
            outer_chamfer,
            taper_angle_deg,
            segments,
            boolean_method,
            fallback_bvh_sdf,
            generate_plug,
            plug_clearance,
        )?;
        let (rep, plugs) = self.scene.punch_drain_hole_at(id, point, &spec, &opts)?;
        let mut plug_ids = Vec::with_capacity(plugs.len() * 2);
        for plug_id in plugs {
            let pair = self.added(plug_id, plug_normals)?;
            plug_ids.extend_from_slice(&pair);
        }
        crate::wave6::encode_drain_report(&rep, &plug_ids)
    }

    /// Places a stamp on the target, one undo step: kept separate, it is added
    /// as a new mesh, which undo removes. `stamp_anchor` is the point of the
    /// stamp, in its own coordinates, that goes to the hit point; without it
    /// see `Scene::place_mesh_world`.
    #[allow(clippy::too_many_arguments)]
    pub fn place_mesh(
        &mut self,
        target_index: u32,
        target_generation: u32,
        stamp_index: u32,
        stamp_generation: u32,
        hit_point3: &[f64],
        hit_normal3: &[f64],
        up_axis: u8,
        twist_angle_rad: f64,
        uniform_scale: f64,
        embed_depth: f64,
        mode: u8,
        conform_to_surface: bool,
        boolean_method: u8,
        normals: bool,
        stamp_anchor: Option<Vec<f64>>,
    ) -> Result<Vec<f64>, Error> {
        let target = self.id(target_index, target_generation)?;
        let stamp = self.id(stamp_index, stamp_generation)?;
        let (hp, hn, params) = crate::wave6::parse_place_params(
            hit_point3,
            hit_normal3,
            up_axis,
            twist_angle_rad,
            uniform_scale,
            embed_depth,
            mode,
            conform_to_surface,
            boolean_method,
        )?;
        let (rep, added) = match stamp_anchor {
            Some(a) => {
                let anchor = crate::wave6::vec3_f64(&a, "stampAnchor")?;
                self.scene
                    .place_mesh_anchored(target, stamp, anchor, hp, hn, &params)?
            }
            None => self
                .scene
                .place_mesh_world(target, stamp, hp, hn, &params)?,
        };
        let added_pair = match added {
            Some(new_id) => {
                let pair = self.added(new_id, normals)?;
                Some([pair[0], pair[1]])
            }
            None => None,
        };
        Ok(crate::wave6::encode_place_report(&rep, added_pair))
    }
}

/// `[index, generation, triangle, point xyz, distance]` of a closest hit.
fn closest_record(h: fast_mesh_scene::SceneClosestHit) -> Vec<f64> {
    let mut out = vec![
        f64::from(h.mesh.index()),
        f64::from(h.mesh.generation()),
        f64::from(h.hit.triangle),
    ];
    out.extend_from_slice(&h.point);
    out.push(h.distance);
    out
}

fn icp_status_code(status: IcpStatus) -> f64 {
    match status {
        IcpStatus::Converged => 0.0,
        IcpStatus::MaxIterations => 1.0,
        IcpStatus::Underconstrained => 2.0,
        IcpStatus::TooFewCorrespondences => 3.0,
        IcpStatus::Diverged => 4.0,
    }
}

fn parse_ops_boolean_op(op: u8) -> Result<OpsBooleanOp, Error> {
    match op {
        0 => Ok(OpsBooleanOp::Union),
        1 => Ok(OpsBooleanOp::Subtract),
        2 => Ok(OpsBooleanOp::Intersect),
        _ => Err(Error::invalid(
            "op must be 0 (union), 1 (subtract) or 2 (intersect)",
        )),
    }
}

fn parse_boolean_options(
    op: u8,
    method: u8,
    coplanar_tolerance: f64,
    snap_slivers: bool,
    voxel_pitch: f64,
    max_triangles: f64,
) -> Result<BooleanOptions, Error> {
    let op = parse_ops_boolean_op(op)?;
    let method = match method {
        0 => BooleanMethod::Manifold,
        1 => BooleanMethod::BvhSdf,
        2 => BooleanMethod::FhEdt,
        _ => {
            return Err(Error::invalid(
                "method must be 0 (manifold), 1 (bvhSdf) or 2 (fhEdt)",
            ));
        }
    };
    let defaults = BooleanOptions::default();
    let coplanar_tolerance = if coplanar_tolerance.is_nan() {
        defaults.coplanar_tolerance
    } else {
        coplanar_tolerance
    };
    let voxel_pitch = if voxel_pitch.is_nan() {
        None
    } else if !voxel_pitch.is_finite() || voxel_pitch <= 0.0 {
        return Err(Error::invalid("voxelPitch must be finite and positive"));
    } else {
        Some(voxel_pitch as f32)
    };
    let max_triangles = match crate::edit::count_opt(max_triangles, "maxTriangles")? {
        Some(n) => {
            Some(u32::try_from(n).map_err(|_| Error::invalid("maxTriangles exceeds u32 range"))?)
        }
        None => None,
    };
    Ok(BooleanOptions {
        op,
        method,
        coplanar_tolerance,
        snap_slivers,
        voxel_pitch,
        max_triangles,
    })
}

fn decode_csg_node(words: &[f64], cursor: &mut usize, depth: usize) -> Result<CsgNode, Error> {
    if depth > 256 {
        return Err(Error::invalid("csg tree nesting depth exceeds limit"));
    }
    let tag = *words
        .get(*cursor)
        .ok_or_else(|| Error::invalid("truncated csg tree"))?;
    *cursor += 1;
    if tag == 0.0 {
        let leaf_idx = *words
            .get(*cursor)
            .ok_or_else(|| Error::invalid("truncated csg leaf index"))?;
        *cursor += 1;
        let idx = crate::edit::count_opt(leaf_idx, "leafIndex")?
            .ok_or_else(|| Error::invalid("csg leaf index must not be NaN"))?;
        Ok(CsgNode::Leaf(idx))
    } else if tag == 1.0 {
        let op_word = *words
            .get(*cursor)
            .ok_or_else(|| Error::invalid("truncated csg op code"))?;
        *cursor += 1;
        if !op_word.is_finite() || op_word.fract() != 0.0 || !(0.0..=2.0).contains(&op_word) {
            return Err(Error::invalid(
                "csg op must be 0 (union), 1 (subtract) or 2 (intersect)",
            ));
        }
        let op = parse_ops_boolean_op(op_word as u8)?;
        let left = Box::new(decode_csg_node(words, cursor, depth + 1)?);
        let right = Box::new(decode_csg_node(words, cursor, depth + 1)?);
        Ok(CsgNode::Op { op, left, right })
    } else {
        Err(Error::invalid("csg node tag must be 0 (leaf) or 1 (op)"))
    }
}

fn dab_of(op: &RepairOp) -> Option<Dab> {
    match op {
        RepairOp::Inflate(o) => Some(Dab::Inflate(*o)),
        RepairOp::Smooth(o) => Some(Dab::Smooth(*o)),
        RepairOp::Grab(o) => Some(Dab::Grab(*o)),
        RepairOp::Flatten(o) => Some(Dab::Flatten(*o)),
        RepairOp::Crease(o) => Some(Dab::Crease(*o)),
        RepairOp::Clay(o) => Some(Dab::Clay(*o)),
        RepairOp::Twist(o) => Some(Dab::Twist(*o)),
        _ => None,
    }
}

fn extract_dab(decoded: &[RepairOp]) -> Option<(Dab, &[u32], &[f32], f32)> {
    match decoded {
        [
            RepairOp::Protected {
                vertices,
                weights,
                band,
                op,
            },
        ] => dab_of(op).map(|d| (d, vertices.as_slice(), weights.as_slice(), *band)),
        [op] => dab_of(op).map(|d| (d, &[][..], &[][..], 0.0)),
        _ => None,
    }
}

fn parse_fill_continuity(code: u8) -> Result<FillContinuity, Error> {
    match code {
        0 => Ok(FillContinuity::Flat),
        1 => Ok(FillContinuity::TangentG1),
        2 => Ok(FillContinuity::CurvatureG2),
        _ => Err(Error::invalid(
            "continuity must be 0 (flat), 1 (tangent) or 2 (curvature)",
        )),
    }
}

fn fill_continuity_code(continuity: FillContinuity) -> f64 {
    match continuity {
        FillContinuity::Flat => 0.0,
        FillContinuity::TangentG1 => 1.0,
        FillContinuity::CurvatureG2 => 2.0,
    }
}

fn parse_fill_holes_continuity_options(
    continuity: u8,
    max_hole_perimeter: f64,
    max_hole_vertices: f64,
    refine_patch: bool,
    smooth_iterations: f64,
) -> Result<FillHolesOptions, Error> {
    let defaults = FillHolesOptions::default();
    Ok(FillHolesOptions {
        continuity: parse_fill_continuity(continuity)?,
        max_hole_perimeter: opt_positive_f32(max_hole_perimeter, "maxHolePerimeter")?,
        max_hole_vertices: count_opt(max_hole_vertices, "maxHoleVertices")?,
        refine_patch,
        smooth_iterations: count_opt(smooth_iterations, "smoothIterations")?
            .unwrap_or(defaults.smooth_iterations),
    })
}

fn encode_fill_stats(stats: &FillStats) -> Vec<f64> {
    vec![
        count(stats.loop_index),
        count(stats.filled_vertices),
        count(stats.filled_triangles),
        stats.perimeter,
        stats.max_dihedral_angle_deg,
        fill_continuity_code(stats.continuity),
        count(stats.island_loops_merged),
        if stats.converged { 1.0 } else { 0.0 },
    ]
}

fn opt_finite_f32(v: f64, default: f32, what: &str) -> Result<f32, Error> {
    if v.is_nan() {
        Ok(default)
    } else if v.is_finite() {
        let out = v as f32;
        if !out.is_finite() {
            return Err(Error::invalid(&format!("{what} out of f32 range")));
        }
        Ok(out)
    } else {
        Err(Error::invalid(&format!("{what} must be finite")))
    }
}

fn parse_dyntopo_options(
    mode: u8,
    detail_size: f64,
    min_detail_ratio: f64,
    max_detail_ratio: f64,
    equalize_valence: bool,
    smooth_tangent: bool,
) -> Result<DyntopoOptions, Error> {
    let defaults = DyntopoOptions::default();
    let mode = match mode {
        0 => DyntopoMode::SplitCollapse,
        1 => DyntopoMode::SplitOnly,
        2 => DyntopoMode::BrushRelative,
        3 => DyntopoMode::ConstantDetail,
        _ => {
            return Err(Error::invalid(
                "mode must be 0 (split-collapse), 1 (split-only), 2 (brush-relative) or 3 (constant-detail)",
            ));
        }
    };
    Ok(DyntopoOptions {
        mode,
        detail_size: opt_finite_f32(detail_size, defaults.detail_size, "detailSize")?,
        min_detail_ratio: opt_finite_f32(
            min_detail_ratio,
            defaults.min_detail_ratio,
            "minDetailRatio",
        )?,
        max_detail_ratio: opt_finite_f32(
            max_detail_ratio,
            defaults.max_detail_ratio,
            "maxDetailRatio",
        )?,
        equalize_valence,
        smooth_tangent,
    })
}

fn encode_dyntopo_patch(out: &mut Vec<f64>, patch: &DyntopoPatch) {
    out.push(count(patch.splits));
    out.push(count(patch.collapses));
    out.push(count(patch.flips));
    out.push(count(patch.smoothed));
    out.push(count(patch.new_vertices.len()));
    out.push(count(patch.removed_vertices.len()));
    out.extend(patch.new_vertices.iter().map(|&v| f64::from(v)));
    out.extend(patch.removed_vertices.iter().map(|&v| f64::from(v)));
}

#[cfg(test)]
mod tests {
    use fast_mesh_scene::SceneError;

    use crate::render::{
        FLAG_ADDED, FLAG_CAPACITY, FLAG_COLORS_ALL, FLAG_INDICES_ALL, FLAG_NORMALS_ALL,
        FLAG_PLACEMENT, FLAG_POSITIONS_ALL, FLAG_REMOVED, FLAG_RESTORABLE, FLAG_UVS, FLAG_UVS_ALL,
    };

    use super::*;
    use crate::error::codes;

    const QUAD: [f64; 12] = [0., 0., 0., 2., 0., 0., 2., 2., 0., 0., 2., 0.];
    const QUAD_TRIS: [u32; 6] = [0, 1, 2, 0, 2, 3];
    const IDENTITY: [f64; 9] = [1., 0., 0., 0., 1., 0., 0., 0., 1.];

    fn add(s: &mut WasmScene, normals: bool) -> (u32, u32) {
        let id = s.add_mesh(&QUAD, &QUAD_TRIS, None, 0, normals).unwrap();
        (id[0], id[1])
    }

    fn code<T: std::fmt::Debug>(r: Result<T, Error>) -> &'static str {
        r.unwrap_err().code
    }

    #[test]
    fn add_place_matrix4() {
        let mut s = WasmScene::new();
        // The default origin is the bounding-box centre.
        let (i, g) = add(&mut s, false);
        let m = s.matrix4(i, g).unwrap();
        let mut expected = [0.0; 16];
        for k in [0, 5, 10, 15] {
            expected[k] = 1.0;
        }
        expected[12..15].copy_from_slice(&[1., 1., 0.]);
        assert_eq!(m, expected);
        // Quarter turn about z, then a translation: R·(p) + t with
        // p = o + local, so the last column is R·o + t.
        let rot = [0., -1., 0., 1., 0., 0., 0., 0., 1.];
        s.set_placement(i, g, &rot, &[10., 20., 30.], None).unwrap();
        let p = s.placement(i, g).unwrap();
        assert_eq!(&p[..9], &rot);
        assert_eq!(&p[9..12], &[10., 20., 30.]);
        assert_eq!(&p[12..], &[1., 1., 1.]);
        let m = s.matrix4(i, g).unwrap();
        // Column-major: column j is m[4j..4j+4].
        assert_eq!(&m[0..4], &[0., 1., 0., 0.]);
        assert_eq!(&m[4..8], &[-1., 0., 0., 0.]);
        assert_eq!(&m[8..12], &[0., 0., 1., 0.]);
        assert_eq!(&m[12..16], &[9., 21., 30., 1.]);
        // A scale multiplies the rotation columns, and the last column is
        // R·(S·o) + t: the origin [1, 1, 0] maps to [9, 21, 30] here, since
        // R·(S·o) = Rz90·(2, 1, 0) = (-1, 2, 0).
        s.set_placement(i, g, &rot, &[10., 20., 30.], Some(vec![2., 1., 3.]))
            .unwrap();
        let p = s.placement(i, g).unwrap();
        assert_eq!(&p[12..], &[2., 1., 3.]);
        let m = s.matrix4(i, g).unwrap();
        assert_eq!(&m[0..4], &[0., 2., 0., 0.]);
        assert_eq!(&m[4..8], &[-1., 0., 0., 0.]);
        assert_eq!(&m[8..12], &[0., 0., 3., 0.]);
        assert_eq!(&m[12..16], &[9., 22., 30., 1.]);
        // An explicit origin is used as given.
        let id = s
            .add_mesh(&QUAD, &QUAD_TRIS, Some(vec![0., 0., 0.]), 1, true)
            .unwrap();
        let m = s.matrix4(id[0], id[1]).unwrap();
        assert_eq!(&m[12..16], &[0., 0., 0., 1.]);
        let mesh = s.scene.mesh(s.id(id[0], id[1]).unwrap()).unwrap();
        assert_eq!(mesh.positions()[2], [2., 2., 0.]);
    }

    #[test]
    fn views_ptr_len_match_slots() {
        let mut s = WasmScene::new();
        let (i, g) = add(&mut s, true);
        assert_eq!(s.vertex_slots(i, g).unwrap(), 4);
        assert_eq!(s.triangle_slots(i, g).unwrap(), 2);
        assert_eq!(s.positions_len(i, g).unwrap(), 12);
        assert_eq!(s.normals_len(i, g).unwrap(), 12);
        assert_eq!(s.indices_len(i, g).unwrap(), 6);
        // Colouring is off.
        assert_eq!(s.colors_ptr(i, g).unwrap(), 0);
        assert_eq!(s.colors_len(i, g).unwrap(), 0);
        let id = s.id(i, g).unwrap();
        let mesh = s.scene.mesh(id).unwrap();
        assert_eq!(
            s.positions_ptr(i, g).unwrap(),
            mesh.positions().as_ptr() as u32
        );
        assert_eq!(
            s.indices_ptr(i, g).unwrap(),
            mesh.triangles().as_ptr() as u32
        );
        let normals = s.scene.normals(id).unwrap().unwrap();
        assert_eq!(s.normals_ptr(i, g).unwrap(), normals.as_ptr() as u32);

        // Normals off: no view.
        let (j, h) = add(&mut s, false);
        assert_eq!(s.normals_ptr(j, h).unwrap(), 0);
        assert_eq!(s.normals_len(j, h).unwrap(), 0);

        // Growth and removal keep the lengths slot-aligned.
        s.scene
            .edit(id, |tx| {
                let v = tx.add_vertices(&[[3., 0., 0.]])?;
                tx.add_triangles(&[[1, v[0], 2]])?;
                tx.remove_triangles(&[0])
            })
            .unwrap();
        assert_eq!(s.positions_len(i, g).unwrap(), 15);
        assert_eq!(s.normals_len(i, g).unwrap(), 15);
        assert_eq!(s.indices_len(i, g).unwrap(), 9);
        assert_eq!(s.vertex_count(i, g).unwrap(), 5);
        assert_eq!(s.triangle_count(i, g).unwrap(), 2);
    }

    #[test]
    fn cut_by_plane_binding() {
        let mut s = WasmScene::new();
        // A closed unit cube, outward wound.
        let positions: [f64; 24] = [
            0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., //
            0., 0., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1.,
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, //
            0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, //
            1, 2, 6, 1, 6, 5, 3, 0, 4, 3, 4, 7,
        ];
        let id = s.add_mesh(&positions, &indices, None, 0, false).unwrap();
        let (i, g) = (id[0], id[1]);
        let r = s
            .cut_by_plane(i, g, &[0.5, 0., 0., 1., 0., 0.], true)
            .unwrap();
        assert_eq!(r.len(), 7);
        assert_eq!(&r[..2], &[i, g]);
        assert_eq!(s.triangle_count(i, g).unwrap(), r[4]);
        assert_eq!(s.triangle_count(r[2], r[3]).unwrap(), r[5]);
        assert!(r[6] > 0);
        // The added mesh renders: it has a render state and the placement.
        assert_eq!(s.placement(r[2], r[3]).unwrap(), s.placement(i, g).unwrap());
        assert_eq!(s.bounds(r[2], r[3]).unwrap().len(), 6);
        // A plane past the cube misses it; nothing changes.
        assert_eq!(
            code(s.cut_by_plane(i, g, &[5., 0., 0., 1., 0., 0.], true)),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(s.triangle_count(i, g).unwrap(), r[4]);
        // The cut undoes in one step.
        assert!(s.undo().unwrap());
        assert_eq!(s.triangle_count(i, g).unwrap(), 12);
    }

    #[test]
    fn register_binding_lays_out_the_report() {
        let mut s = WasmScene::new();
        // A closed box with three different sides: all six motions constrained.
        let positions: [f64; 24] = [
            0., 0., 0., 3., 0., 0., 3., 2., 0., 0., 2., 0., //
            0., 0., 1., 3., 0., 1., 3., 2., 1., 0., 2., 1.,
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, //
            0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, //
            1, 2, 6, 1, 6, 5, 3, 0, 4, 3, 4, 7,
        ];
        let f = s.add_mesh(&positions, &indices, None, 0, false).unwrap();
        let m = s.add_mesh(&positions, &indices, None, 0, false).unwrap();
        s.set_placement(m[0], m[1], &IDENTITY, &[0.05, -0.03, 0.02], None)
            .unwrap();
        let nan = f64::NAN;
        let r = s
            .register(m[0], m[1], f[0], f[1], nan, nan, nan, nan, nan, true, nan)
            .unwrap();
        assert_eq!(r.len(), 24);
        // Back on the box: identity rotation, zero translation, unit scale.
        for (got, want) in r[..9].iter().zip(&IDENTITY) {
            assert!((got - want).abs() < 1e-6, "{r:?}");
        }
        assert!(r[9..12].iter().all(|v| v.abs() < 1e-6), "{r:?}");
        assert_eq!(&r[12..15], &[1.0, 1.0, 1.0]);
        assert!(r[15] < 1e-6, "{r:?}");
        assert!(r[16] >= 1.0 && r[16] <= 50.0, "{r:?}");
        assert_eq!(&r[17..21], &[0.0, 1.0, 20.0, 20.0], "{r:?}");
        assert!(r[21] > 1e-6 && r[21] <= 1.0, "{r:?}");
        assert_eq!(r[22], 0.0, "{r:?}");
        assert!(r[23] > 0.0, "{r:?}");
        // A bad count is rejected here; a bad range by the solver.
        let bad = s.register(m[0], m[1], f[0], f[1], -1.0, nan, nan, nan, nan, true, nan);
        assert_eq!(code(bad), codes::INVALID_ARGUMENT);
        let bad = s.register(m[0], m[1], f[0], f[1], nan, nan, 2.0, nan, nan, true, nan);
        assert_eq!(code(bad), codes::INVALID_ARGUMENT);
        let bad = s.register(m[0], m[1], m[0], m[1], nan, nan, nan, nan, nan, true, nan);
        assert_eq!(code(bad), codes::INVALID_ARGUMENT);
    }

    #[test]
    fn measure_binding_reports_world_space() {
        let mut s = WasmScene::new();
        // A closed unit cube, outward wound.
        let positions: [f64; 24] = [
            0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., //
            0., 0., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1.,
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, //
            0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, //
            1, 2, 6, 1, 6, 5, 3, 0, 4, 3, 4, 7,
        ];
        let id = s.add_mesh(&positions, &indices, None, 0, false).unwrap();
        let (i, g) = (id[0], id[1]);
        let m = s.measure(i, g).unwrap();
        assert!((m[0] - 6.0).abs() < 1e-9, "{m:?}");
        assert!((m[1] - 1.0).abs() < 1e-9, "{m:?}");
        assert_eq!(&m[2..], &[1.0, 0.0, 0.0, 0.0, 0.0]);
        // A uniform scale of 2: area x4, volume x8.
        s.set_placement(i, g, &IDENTITY, &[1., -2., 3.], Some(vec![2., 2., 2.]))
            .unwrap();
        let m = s.measure(i, g).unwrap();
        assert!((m[0] - 24.0).abs() < 1e-9, "{m:?}");
        assert!((m[1] - 8.0).abs() < 1e-9, "{m:?}");
        // An open 2x2 quad: area without volume, reason open.
        let (j, h) = add(&mut s, false);
        let m = s.measure(j, h).unwrap();
        assert!((m[0] - 4.0).abs() < 1e-9, "{m:?}");
        assert!(m[1].is_nan(), "{m:?}");
        assert_eq!(&m[2..], &[0.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(code(s.measure(i + 9, g)), codes::UNKNOWN_MESH);
    }

    #[test]
    fn error_codes_map() {
        let mut s = WasmScene::new();
        let (i, g) = add(&mut s, true);
        // Reachable through the bindings.
        assert_eq!(code(s.remove_mesh(i, g + 1)), codes::UNKNOWN_MESH);
        assert_eq!(code(s.matrix4(i + 7, 0)), codes::UNKNOWN_MESH);
        assert_eq!(
            code(s.set_placement(i, g, &[2.; 9], &[0.; 3], None)),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.set_placement(i, g, &IDENTITY, &[0.; 2], None)),
            codes::INVALID_ARGUMENT
        );
        // Scales must be 3 finite positive values.
        assert_eq!(
            code(s.set_placement(i, g, &IDENTITY, &[0.; 3], Some(vec![1., 0., 1.]))),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.set_placement(i, g, &IDENTITY, &[0.; 3], Some(vec![1., 1.]))),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(code(s.set_mode(i, g, 3)), codes::INVALID_ARGUMENT);
        assert_eq!(
            code(s.add_mesh(&QUAD, &QUAD_TRIS, None, 9, true)),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.add_mesh(&QUAD, &QUAD_TRIS, Some(vec![0.; 2]), 0, true)),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.add_mesh(&QUAD[..4], &QUAD_TRIS, None, 0, true)),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.add_mesh(&QUAD, &[0, 1, 9], None, 0, true)),
            codes::INVALID_INDEX
        );
        assert_eq!(code(s.end_step()), codes::STEP_NOT_OPEN);
        assert_eq!(code(s.cancel_step()), codes::STEP_NOT_OPEN);
        // A failed add leaves no mesh behind.
        assert_eq!(s.scene.ids().count(), 1);

        // Every scene error has its code.
        use fast_mesh_bvh::BvhError;
        use fast_mesh_fields::FieldsError;
        use fast_mesh_ops::OpsError;
        use fast_mesh_query::QueryError;
        use fast_mesh_store::StoreError;
        let oom = StoreError::OutOfMemory;
        let table = [
            (SceneError::UnknownMesh, codes::UNKNOWN_MESH),
            (SceneError::UnknownField, codes::UNKNOWN_FIELD),
            (SceneError::UnknownSection, codes::UNKNOWN_SECTION),
            (SceneError::InvalidInput("x"), codes::INVALID_ARGUMENT),
            (SceneError::StaleBvh(s.id(i, g).unwrap()), codes::STALE_BVH),
            (SceneError::StepNotOpen, codes::STEP_NOT_OPEN),
            (SceneError::Corrupted, codes::CORRUPTED),
            (SceneError::Store(oom.clone()), codes::OUT_OF_MEMORY),
            (SceneError::Bvh(BvhError::OutOfMemory), codes::OUT_OF_MEMORY),
            (
                SceneError::Bvh(BvhError::Store(oom.clone())),
                codes::OUT_OF_MEMORY,
            ),
            (
                SceneError::Query(QueryError::OutOfMemory),
                codes::OUT_OF_MEMORY,
            ),
            (
                SceneError::Fields(FieldsError::OutOfMemory),
                codes::OUT_OF_MEMORY,
            ),
            (
                SceneError::Fields(FieldsError::Query(QueryError::OutOfMemory)),
                codes::OUT_OF_MEMORY,
            ),
            (
                SceneError::Ops(OpsError::Store(oom.clone())),
                codes::OUT_OF_MEMORY,
            ),
            (SceneError::Ops(OpsError::StaleLoops), codes::STALE_LOOPS),
            (SceneError::Query(QueryError::StaleBvh), codes::STALE_BVH),
            (
                SceneError::Store(StoreError::InvalidVertex(3)),
                codes::INVALID_INDEX,
            ),
            (
                SceneError::Store(StoreError::InvalidTriangle(3)),
                codes::INVALID_INDEX,
            ),
            (
                SceneError::Store(StoreError::PatchMismatch),
                codes::INVALID_ARGUMENT,
            ),
            (
                SceneError::Ops(OpsError::InvalidInput("x")),
                codes::INVALID_ARGUMENT,
            ),
        ];
        for (err, expected) in table {
            let e = Error::from(err.clone());
            assert_eq!(e.code, expected, "{err:?}");
            assert_eq!(e.message(), format!("{expected}: {err}"));
        }
    }

    #[test]
    fn undoable_adds_and_removals_report_and_restore() {
        let mut s = WasmScene::new();
        let (other, _) = add(&mut s, false);
        let uvs = vec![0., 0., 1., 0., 1., 1., 0., 1.];
        let id = s
            .add_mesh_undoable(&QUAD, &QUAD_TRIS, Some(uvs.clone()), None, 0, true)
            .unwrap();
        let (i, g) = (id[0], id[1]);
        assert!(s.can_undo());
        s.take_changes();
        let all = FLAG_POSITIONS_ALL | FLAG_NORMALS_ALL | FLAG_COLORS_ALL | FLAG_INDICES_ALL;
        let uv = FLAG_UVS | FLAG_UVS_ALL;
        let gone = vec![1, i, g, FLAG_REMOVED | FLAG_RESTORABLE, 0, 0, 0, 0, 0, 0];
        let back = vec![1, i, g, FLAG_ADDED | all | uv, 0, 0, 0, 0, 0, 0, 0];

        // Undo of the add, redo of the add, removal, undo of the removal.
        assert!(s.undo().unwrap());
        assert_eq!(s.take_changes(), gone);
        assert!(s.is_restorable(i, g) && !s.is_restorable(other, 0));
        assert!(!s.has_mesh(i, g) && s.has_mesh(other, 0));
        assert_eq!(code(s.mesh(i, g).map(|_| ())), codes::UNKNOWN_MESH);
        assert_eq!(code(s.remove_mesh_undoable(i, g)), codes::UNKNOWN_MESH);
        assert!(s.redo().unwrap());
        assert_eq!(s.take_changes(), back);
        assert!(!s.is_restorable(i, g));
        s.remove_mesh_undoable(i, g).unwrap();
        assert_eq!(s.take_changes(), gone);
        assert!(s.undo().unwrap());
        assert_eq!(s.take_changes(), back);
        assert_eq!(s.mesh(i, g).unwrap().uvs(), Some(&uvs[..]));
        // Its render state is back too: later edits report exact ranges.
        let mid = s.id(i, g).unwrap();
        s.scene
            .edit(mid, |tx| tx.set_positions(&[2], &[[0.5, 0.5, 0.]]))
            .unwrap();
        let msg = s.take_changes();
        assert_eq!(&msg[..6], &[1, i, g, FLAG_UVS, 1, 2]);

        // Removed and back between two calls: changed in full, never gone.
        s.remove_mesh_undoable(i, g).unwrap();
        assert!(s.undo().unwrap());
        assert_eq!(
            s.take_changes(),
            vec![1, i, g, FLAG_PLACEMENT | all | uv, 0, 0, 0, 0, 0, 0, 0]
        );

        // The history drops it: the removal is final.
        s.remove_mesh_undoable(i, g).unwrap();
        s.clear_history().unwrap();
        assert!(!s.is_restorable(i, g));
        assert_eq!(
            s.take_changes(),
            vec![1, i, g, FLAG_REMOVED, 0, 0, 0, 0, 0, 0]
        );

        // A bad add records nothing.
        let bad = s.add_mesh_undoable(&QUAD, &[0, 1, 9], None, None, 0, true);
        assert_eq!(code(bad), codes::INVALID_INDEX);
        let bad = s.add_mesh_undoable(&QUAD, &QUAD_TRIS, Some(vec![0.]), None, 0, true);
        assert_eq!(code(bad), codes::INVALID_ARGUMENT);
        assert!(!s.can_undo() && s.take_changes() == [0]);
    }

    #[test]
    fn slots_added_and_undone_report_capacity() {
        let mut s = WasmScene::new();
        let (i, g) = add(&mut s, false);
        let (j, h) = add(&mut s, false);
        s.take_changes();
        // Added then undone before the next take: the edit cancels out, but
        // the slots stay, so the buffers must still be recreated.
        let id = s.id(i, g).unwrap();
        s.scene
            .edit(id, |tx| {
                let v = tx.add_vertices(&[[3., 0., 0.]])?;
                tx.add_triangles(&[[1, v[0], 2]])
            })
            .unwrap();
        assert!(s.undo().unwrap());
        assert_eq!(s.vertex_slots(i, g).unwrap(), 5);
        let all = FLAG_POSITIONS_ALL | FLAG_NORMALS_ALL | FLAG_COLORS_ALL | FLAG_INDICES_ALL;
        assert_eq!(
            s.take_changes(),
            vec![1, i, g, FLAG_CAPACITY | all, 0, 0, 0, 0, 0, 0]
        );
        // Reported once; the untouched mesh never is.
        assert_eq!(s.take_changes(), vec![0]);
        assert_eq!(s.vertex_slots(j, h).unwrap(), 4);
    }

    #[test]
    fn take_changes_after_edit_via_scene() {
        let mut s = WasmScene::new();
        let (i, g) = add(&mut s, true);
        // The add: every buffer "all".
        let all = FLAG_POSITIONS_ALL | FLAG_NORMALS_ALL | FLAG_COLORS_ALL | FLAG_INDICES_ALL;
        assert_eq!(
            s.take_changes(),
            vec![1, i, g, FLAG_ADDED | all, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(s.take_changes(), vec![0]);

        // Move vertex 2 through the scene directly.
        let id = s.id(i, g).unwrap();
        s.scene
            .edit(id, |tx| tx.set_positions(&[2], &[[1., 1., 1.]]))
            .unwrap();
        // Positions: vertex 2. Normals: every vertex of both triangles
        // using it. Indices: none.
        assert_eq!(
            s.take_changes(),
            vec![1, i, g, 0, 1, 2, 1, 1, 0, 4, 0, 0, 0, 0]
        );

        // Growth reports the capacity change with every buffer "all".
        s.scene
            .edit(id, |tx| {
                let v = tx.add_vertices(&[[3., 0., 0.]])?;
                tx.add_triangles(&[[1, v[0], 2]])
            })
            .unwrap();
        assert_eq!(
            s.take_changes(),
            vec![1, i, g, FLAG_CAPACITY | all, 0, 0, 0, 0, 0, 0]
        );

        // A placement change only sets its flag.
        s.set_placement(i, g, &IDENTITY, &[1., 0., 0.], None)
            .unwrap();
        assert_eq!(
            s.take_changes(),
            vec![1, i, g, FLAG_PLACEMENT, 0, 0, 0, 0, 0, 0]
        );

        // History through the bindings.
        assert!(s.can_undo());
        assert!(s.undo().unwrap());
        assert!(s.can_redo());
        s.take_changes();
        s.begin_step();
        s.set_placement(i, g, &IDENTITY, &[2., 0., 0.], None)
            .unwrap();
        s.set_placement(i, g, &IDENTITY, &[3., 0., 0.], None)
            .unwrap();
        s.end_step().unwrap();
        assert!(!s.can_redo());
        assert!(s.undo().unwrap());
        assert_eq!(s.placement(i, g).unwrap()[9], 0.);
        s.clear_history().unwrap();
        assert!(!s.can_undo() && !s.can_redo());
        assert!(!s.undo().unwrap());

        // Removal: one record with the removed flag, nothing dirty.
        s.take_changes();
        s.remove_mesh(i, g).unwrap();
        assert_eq!(
            s.take_changes(),
            vec![1, i, g, FLAG_REMOVED, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(code(s.remove_mesh(i, g)), codes::UNKNOWN_MESH);
    }

    #[test]
    fn mode_and_bounds_getters() {
        let mut s = WasmScene::new();
        let (i, g) = add(&mut s, true);
        assert_eq!(s.mode(i, g).unwrap(), 0);
        s.set_mode(i, g, 1).unwrap();
        assert_eq!(s.mode(i, g).unwrap(), 1);
        // QUAD spans 0..=2 in x and y; stored relative to its centre.
        assert_eq!(s.bounds(i, g).unwrap(), vec![-1., -1., 0., 1., 1., 0.]);
        assert_eq!(code(s.mode(i + 9, g)), codes::UNKNOWN_MESH);
        assert_eq!(code(s.bounds(i + 9, g)), codes::UNKNOWN_MESH);
    }

    #[test]
    fn lazy_and_manual_add_mesh_defer_the_bvh() {
        let mut s = WasmScene::new();
        let lazy = s.add_mesh(&QUAD, &QUAD_TRIS, None, 1, true).unwrap();
        let (li, lg) = (lazy[0], lazy[1]);
        assert!(!s.scene.is_current(s.id(li, lg).unwrap()).unwrap());
        // First use builds it: analyzing with self-intersection updates Lazy.
        s.analyze(li, lg, 0.0, true, 10).unwrap();
        assert!(s.scene.is_current(s.id(li, lg).unwrap()).unwrap());
        // Manual waits for `update`: analyze refuses until then.
        let manual = s.add_mesh(&QUAD, &QUAD_TRIS, None, 2, false).unwrap();
        let (mi, mg) = (manual[0], manual[1]);
        assert_eq!(code(s.analyze(mi, mg, 0.0, true, 10)), codes::STALE_BVH);
        s.update(mi, mg).unwrap();
        s.analyze(mi, mg, 0.0, true, 10).unwrap();
    }

    #[test]
    fn modes_update_and_flush() {
        let mut s = WasmScene::new();
        let (i, g) = add(&mut s, true);
        s.set_mode(i, g, 2).unwrap();
        let id = s.id(i, g).unwrap();
        s.scene
            .edit(id, |tx| tx.set_positions(&[2], &[[1., 1., 1.]]))
            .unwrap();
        assert!(!s.scene.is_current(id).unwrap());
        s.flush().unwrap();
        assert!(!s.scene.is_current(id).unwrap());
        s.update(i, g).unwrap();
        assert!(s.scene.is_current(id).unwrap());
        assert_eq!(parse_mode(1).unwrap(), UpdateMode::Lazy);
    }

    /// An octahedron of "radius" `r` centred at `c`: closed and consistently
    /// oriented, with its equator square in the plane `z = c.z`.
    fn octahedron(s: &mut WasmScene, c: [f64; 3], r: f64) -> (u32, u32) {
        let mut pos = Vec::new();
        for d in [
            [r, 0., 0.],
            [0., r, 0.],
            [-r, 0., 0.],
            [0., -r, 0.],
            [0., 0., r],
            [0., 0., -r],
        ] {
            pos.extend((0..3).map(|k| c[k] + d[k]));
        }
        let tris = [
            0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4, 1, 0, 5, 2, 1, 5, 3, 2, 5, 0, 3, 5,
        ];
        let id = s.add_mesh(&pos, &tris, None, 0, true).unwrap();
        (id[0], id[1])
    }

    #[test]
    fn queries_in_world_space() {
        let mut s = WasmScene::new();
        let (ai, ag) = octahedron(&mut s, [0., 0., 0.], 1.);
        let (bi, bg) = octahedron(&mut s, [3., 0., 0.], 1.);
        let inf = f64::INFINITY;
        // The ray along +x hits A's vertex at x = -1.
        let h = s
            .raycast(&[-5., 0., 0.], &[2., 0., 0.], 0., inf, 0)
            .unwrap();
        assert_eq!(&h[..2], &[f64::from(ai), f64::from(ag)]);
        assert_eq!(h.len(), 10);
        assert!((h[6] - 4.).abs() < 1e-6 && (h[3] + 1.).abs() < 1e-6);
        assert!((h[7] + h[8] + h[9] - 1.).abs() < 1e-6);
        assert_eq!(
            s.raycast(&[-5., 0., 0.], &[1., 0., 0.], 0., 3., 0).unwrap(),
            []
        );
        // Placements move meshes in the world.
        s.set_placement(bi, bg, &IDENTITY, &[0., 0., 10.], None)
            .unwrap();
        let h = s
            .raycast(&[3., 0., 20.], &[0., 0., -1.], 0., inf, 1)
            .unwrap();
        assert_eq!(&h[..2], &[f64::from(bi), f64::from(bg)]);
        assert!((h[5] - 11.).abs() < 1e-6);
        s.set_placement(bi, bg, &IDENTITY, &[0., 0., 0.], None)
            .unwrap();

        let c = s.closest_point(&[1.5, 0., 0.], inf).unwrap();
        assert_eq!(&c[..2], &[f64::from(ai), f64::from(ag)]);
        assert!((c[6] - 0.5).abs() < 1e-6);
        let c = s.closest_point_on(bi, bg, &[1.5, 0., 0.], inf).unwrap();
        assert_eq!(c[0], f64::from(bi));
        assert_eq!(s.closest_point(&[1.5, 0., 0.], 0.1).unwrap(), []);

        let d = s.mesh_distance(ai, ag, bi, bg, inf).unwrap();
        assert_eq!(d.len(), 9);
        assert!((d[0] - 1.).abs() < 1e-6 && (d[1] - 1.).abs() < 1e-6 && (d[4] - 2.).abs() < 1e-6);
        assert_eq!(s.mesh_distance(ai, ag, bi, bg, 0.5).unwrap(), []);
        assert!(!s.intersects(ai, ag, bi, bg).unwrap());

        assert_eq!(s.find_contacts(0.5, None).unwrap(), [0.]);
        let msg = s.find_contacts(1.5, None).unwrap();
        assert_eq!(msg[0], 1.);
        let head = [ai, ag, bi, bg].map(f64::from);
        assert_eq!(&msg[1..5], &head);
        assert_eq!(msg[5], 0.);
        let (na, nb) = (msg[7] as usize, msg[8] as usize);
        assert!(na > 0 && nb > 0 && msg.len() == 9 + na + nb);
        let pairs = Some(vec![bi, bg, ai, ag]);
        assert_eq!(s.find_contacts(1.5, pairs).unwrap(), msg);

        // Malformed input.
        assert_eq!(
            code(s.raycast(&[0.; 2], &[1., 0., 0.], 0., inf, 0)),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.raycast(&[0.; 3], &[1., 0., 0.], 0., inf, 3)),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.find_contacts(1., Some(vec![ai, ag, bi]))),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.find_contacts(1., Some(vec![ai, ag, bi, bg + 1]))),
            codes::UNKNOWN_MESH
        );
    }

    #[test]
    fn thickness_field_bindings_report_world_space() {
        let mut s = WasmScene::new();
        // The unit cube's corners read in [2, 2 sqrt(3)] locally.
        let (i, g) = octahedron(&mut s, [0., 0., 0.], 1.);
        let f = s
            .add_thickness_field("t".into(), &[i, g], f64::INFINITY)
            .unwrap();
        let values = s.scene.field_values(s.field(f).unwrap()).unwrap().to_vec();
        assert_eq!(values.len(), 6);
        // Every vertex reads the diameter, 2: each normal ray crosses the
        // octahedron to the opposite vertex.
        for &v in &values {
            assert!((v - 2.0).abs() < 1e-3, "{values:?}");
        }
        assert_eq!(s.field_targets(f).unwrap(), Vec::<u32>::new());
        assert_eq!(s.field_target_status(f).unwrap(), Vec::<u32>::new());
        // A uniform scale of 2 doubles every thickness.
        s.set_placement(i, g, &IDENTITY, &[0., 0., 0.], Some(vec![2., 2., 2.]))
            .unwrap();
        let scaled = s.scene.field_values(s.field(f).unwrap()).unwrap().to_vec();
        for (&x, &y) in scaled.iter().zip(&values) {
            assert!((x - 2.0 * y).abs() < 1e-4, "{scaled:?} vs {values:?}");
        }
        // An open quad reads NaN everywhere.
        let (j, h) = add(&mut s, false);
        let of = s
            .add_thickness_field("t".into(), &[j, h], f64::INFINITY)
            .unwrap();
        let open = s.scene.field_values(s.field(of).unwrap()).unwrap().to_vec();
        assert!(open.iter().all(|v| v.is_nan()), "{open:?}");
        // Malformed input.
        assert_eq!(
            code(s.add_thickness_field("e".into(), &[], 1.)),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.add_thickness_field("e".into(), &[i, g, j], 1.)),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.add_thickness_field("e".into(), &[i + 9, g], 1.)),
            codes::UNKNOWN_MESH
        );
        assert_eq!(
            code(s.add_thickness_field("e".into(), &[i, g], -1.)),
            codes::INVALID_ARGUMENT
        );
    }

    #[test]
    fn curvature_field_bindings_report_world_space() {
        let mut s = WasmScene::new();
        // A closed convex octahedron reads positive everywhere.
        let (i, g) = octahedron(&mut s, [0., 0., 0.], 1.);
        let f = s.add_curvature_field("k".into(), &[i, g]).unwrap();
        let values = s.scene.field_values(s.field(f).unwrap()).unwrap().to_vec();
        assert_eq!(values.len(), 6);
        for &v in &values {
            assert!(v.is_finite(), "{values:?}");
            assert!(v > 0.0, "{values:?}");
        }
        assert_eq!(s.field_targets(f).unwrap(), Vec::<u32>::new());
        assert_eq!(s.field_target_status(f).unwrap(), Vec::<u32>::new());
        // A uniform scale of 2 halves every curvature.
        s.set_placement(i, g, &IDENTITY, &[0., 0., 0.], Some(vec![2., 2., 2.]))
            .unwrap();
        let scaled = s.scene.field_values(s.field(f).unwrap()).unwrap().to_vec();
        for (&x, &y) in scaled.iter().zip(&values) {
            assert!((x - y / 2.0).abs() < 1e-5, "{scaled:?} vs {values:?}");
        }
        // A flat open quad reads zero everywhere (curvature is defined on
        // open sheets, unlike thickness).
        let (j, h) = add(&mut s, false);
        let of = s.add_curvature_field("k".into(), &[j, h]).unwrap();
        let open = s.scene.field_values(s.field(of).unwrap()).unwrap().to_vec();
        assert!(!open.is_empty());
        for &v in &open {
            assert!(v.abs() < 1e-6, "{open:?}");
        }
        // Malformed input.
        assert_eq!(
            code(s.add_curvature_field("e".into(), &[])),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.add_curvature_field("e".into(), &[i, g, j])),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.add_curvature_field("e".into(), &[i + 9, g])),
            codes::UNKNOWN_MESH
        );
    }

    #[test]
    fn multi_target_field_bindings() {
        let mut s = WasmScene::new();
        let (ai, ag) = octahedron(&mut s, [0., 0., 0.], 1.);
        let (bi, bg) = octahedron(&mut s, [3., 0., 0.], 1.);
        let (ci, cg) = octahedron(&mut s, [-2.5, 0., 0.], 1.);
        let f = s
            .add_distance_field("d".into(), &[ai, ag, bi, bg, ci, cg], 5., 0)
            .unwrap();
        assert_eq!(s.field_targets(f).unwrap(), vec![bi, bg, ci, cg]);
        // Two closed, consistent octahedra: one status each, the same overall.
        assert_eq!(
            s.field_target_status(f).unwrap(),
            vec![1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
        );
        assert_eq!(s.field_status(f).unwrap(), vec![1, 1, 1, 1, 0, 0]);
        // A's vertex at +x (index 0) is nearest B, the one at -x nearest C.
        let values = s.scene.field_values(s.field(f).unwrap()).unwrap();
        assert!((values[0] - 1.).abs() < 1e-6);
        assert!((values[2] - 0.5).abs() < 1e-6);
        let nearest = s.field_nearest(f).unwrap();
        assert_eq!(nearest.len(), 6);
        assert_eq!((nearest[0], nearest[2]), (0, 1));
        // Removing C leaves B.
        s.remove_mesh(ci, cg).unwrap();
        assert_eq!(s.field_targets(f).unwrap(), vec![bi, bg]);
        let nearest = s.field_nearest(f).unwrap();
        assert_eq!((nearest[0], nearest[2]), (0, 0));
        // One mesh alone, or an odd list, is rejected.
        assert_eq!(
            code(s.add_distance_field("e".into(), &[ai, ag], 1., 0)),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.add_distance_field("e".into(), &[ai, ag, bi], 1., 0)),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.add_distance_field("e".into(), &[ai, ag, bi, bg, bi, bg], 1., 0)),
            codes::INVALID_ARGUMENT
        );
    }

    #[test]
    fn fields_and_coloring_off_history() {
        let mut s = WasmScene::new();
        let (ai, ag) = octahedron(&mut s, [0., 0., 0.], 1.);
        let (bi, bg) = octahedron(&mut s, [3., 0., 0.], 1.);
        let f = s
            .add_distance_field("d".into(), &[ai, ag, bi, bg], 5., 0)
            .unwrap();
        assert_eq!(s.field_status(f).unwrap(), vec![1, 1, 1, 1, 0, 0]);
        assert_eq!(s.field_values_len(f).unwrap(), 6);
        let values = s.scene.field_values(s.field(f).unwrap()).unwrap();
        assert_eq!(s.field_values_ptr(f).unwrap(), values.as_ptr() as u32);
        assert!((values[0] - 1.).abs() < 1e-6);

        s.set_placement(bi, bg, &IDENTITY, &[1., 0., 0.], None)
            .unwrap();
        s.undo().unwrap();
        s.take_changes();
        let rule = |extra: &[f32]| {
            let mut r = vec![0., 4., f32::NAN, 0., 1., 9., 9., 9., 255., 0., 0., 0., 255.];
            r.extend_from_slice(extra);
            r
        };
        assert_eq!(s.values_ptr(ai, ag).unwrap(), 0);
        assert_eq!(s.values_len(ai, ag).unwrap(), 0);
        s.set_coloring(ai, ag, f, 0, &rule(&[])).unwrap();
        assert_eq!(s.colors_len(ai, ag).unwrap(), 24);
        // The values behind the colours: a view of the field's channel.
        let values = s.scene.field_values(s.field(f).unwrap()).unwrap();
        assert_eq!(s.values_ptr(ai, ag).unwrap(), values.as_ptr() as u32);
        assert_eq!(s.values_len(ai, ag).unwrap(), 6);
        assert!(s.take_changes()[3] & FLAG_COLORS_ALL != 0);
        s.clear_coloring(ai, ag).unwrap();
        assert_eq!(s.colors_ptr(ai, ag).unwrap(), 0);
        assert_eq!(s.values_ptr(ai, ag).unwrap(), 0);
        // Display settings leave history alone: the redo step survives.
        assert!(s.can_redo());
        // Custom stops, and every malformed rule.
        let stops = [0., 255., 0., 0., 255., 1., 0., 0., 255., 255.];
        s.set_coloring(ai, ag, f, 4, &rule(&stops)).unwrap();
        let bad = [
            (0, rule(&stops)),
            (4, rule(&stops[..7])),
            (5, rule(&[])),
            (0, rule(&[])[..12].to_vec()),
            (0, {
                let mut r = rule(&[]);
                r[3] = 2.;
                r
            }),
            (0, {
                let mut r = rule(&[]);
                r[5] = 256.;
                r
            }),
            (0, {
                let mut r = rule(&[]);
                r[1] = -1.;
                r
            }),
        ];
        for (map, r) in bad {
            assert_eq!(
                code(s.set_coloring(ai, ag, f, map, &r)),
                codes::INVALID_ARGUMENT,
                "{map} {r:?}"
            );
        }
        assert_eq!(
            code(s.add_distance_field("e".into(), &[ai, ag], 1., 0)),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.add_distance_field("e".into(), &[ai, ag, bi, bg], 1., 3)),
            codes::INVALID_ARGUMENT
        );
        s.remove_field(f).unwrap();
        assert_eq!(s.colors_ptr(ai, ag).unwrap(), 0);
        assert_eq!(code(s.field_values_ptr(f)), codes::UNKNOWN_FIELD);
        assert_eq!(code(s.field_status(f)), codes::UNKNOWN_FIELD);
        assert_eq!(code(s.remove_field(f)), codes::UNKNOWN_FIELD);
        assert_eq!(
            code(s.set_coloring(ai, ag, f, 0, &rule(&[]))),
            codes::UNKNOWN_FIELD
        );
    }

    #[test]
    fn field_step_bindings() {
        let mut s = WasmScene::new();
        let (ai, ag) = octahedron(&mut s, [0., 0., 0.], 1.);
        let (bi, bg) = octahedron(&mut s, [3., 0., 0.], 1.);
        let f = s
            .add_distance_field("d".into(), &[ai, ag, bi, bg], 5., 0)
            .unwrap();
        let before = s.scene.field_values(s.field(f).unwrap()).unwrap().to_vec();
        s.set_defer_fields(true);
        assert_eq!(s.field_step(1).unwrap(), 0);
        assert!(s.field_step_meshes().is_empty());

        s.set_placement(bi, bg, &IDENTITY, &[1., 0., 0.], None)
            .unwrap();
        // Deferred: the values wait for the steps.
        assert_eq!(s.scene.field_values(s.field(f).unwrap()).unwrap(), before);
        assert_eq!(s.field_step(1).unwrap(), 1);
        assert_eq!(s.field_step_meshes(), vec![ai, ag, bi, bg]);
        s.abandon_field_step();
        assert!(s.field_step_meshes().is_empty());
        assert_eq!(s.scene.field_values(s.field(f).unwrap()).unwrap(), before);

        let mut steps = 0;
        loop {
            steps += 1;
            match s.field_step(1).unwrap() {
                1 => continue,
                state => {
                    assert_eq!(state, 2);
                    break;
                }
            }
        }
        // Six vertices, one per step, after the near search's own step.
        assert!(steps >= 6, "{steps} steps");
        let values = s.scene.field_values(s.field(f).unwrap()).unwrap();
        assert!((values[0] - 2.).abs() < 1e-6, "{values:?}");
        assert_eq!(s.field_step(u32::MAX).unwrap(), 0);
    }

    #[test]
    fn stepped_field_add_and_mesh_add_bindings() {
        let mut s = WasmScene::new();
        let mut t = WasmScene::new();
        let (ai, ag) = octahedron(&mut s, [0., 0., 0.], 1.);
        let (bi, bg) = octahedron(&mut s, [3., 0., 0.], 1.);
        octahedron(&mut t, [0., 0., 0.], 1.);
        octahedron(&mut t, [3., 0., 0.], 1.);
        // A field added deferred, stepped alone: the values of an eager add.
        let f = s
            .add_distance_field_deferred("d".into(), &[ai, ag, bi, bg], 5., 0)
            .unwrap();
        let g = t
            .add_distance_field("d".into(), &[ai, ag, bi, bg], 5., 0)
            .unwrap();
        while s.field_step_field(f, 1).unwrap() == 1 {}
        assert_eq!(s.field_step_field(f, 1).unwrap(), 0);
        let bits = |w: &WasmScene, id: u32| -> Vec<u32> {
            let v = w.scene.field_values(w.field(id).unwrap()).unwrap();
            v.iter().map(|x| x.to_bits()).collect()
        };
        assert_eq!(bits(&s, f), bits(&t, g));
        assert!(!s.withdraw_field(f).unwrap());
        let h = s
            .add_distance_field_deferred("e".into(), &[ai, ag, bi, bg], 5., 0)
            .unwrap();
        assert!(s.withdraw_field(h).unwrap());
        assert!(s.field_status(h).is_err());
        // A mesh added in steps: the same id and buffers as an add.
        let positions = [0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.5];
        let indices = [0, 1, 2, 1, 3, 2];
        let build = s.begin_add_mesh_upload(4, 2, None, 0, true).unwrap();
        s.add_mesh_push_positions(build, &positions).unwrap();
        s.add_mesh_push_indices(build, &indices).unwrap();
        while !s.add_mesh_step(build, 1).unwrap() {}
        let id = s.finish_add_mesh(build).unwrap();
        let want = t.add_mesh(&positions, &indices, None, 0, true).unwrap();
        assert_eq!(id, want);
        let mesh = |w: &WasmScene| {
            let m = w.scene.mesh(w.id(id[0], id[1]).unwrap()).unwrap();
            (m.positions().to_vec(), m.triangles().to_vec())
        };
        assert_eq!(mesh(&s), mesh(&t));
        assert!(s.add_mesh_step(build, 1).is_err());
        // While built, the build's render buffers are those of the added
        // mesh: the same addresses (the vectors move into the scene) and
        // lengths, normals included once they are built.
        let build = s.begin_add_mesh_upload(4, 2, None, 0, true).unwrap();
        assert!(s.add_mesh_buffers(build).is_err());
        s.add_mesh_push_positions(build, &positions).unwrap();
        s.add_mesh_push_indices(build, &indices).unwrap();
        // Until the store is built there are no buffers; then the mesh's,
        // without normals until those build.
        while s.add_mesh_buffers(build).is_err() {
            assert!(!s.add_mesh_step(build, 1).unwrap());
        }
        let before = s.add_mesh_buffers(build).unwrap();
        assert_eq!(before[2..4], [0, 0], "no normals yet");
        while !s.add_mesh_step(build, 1).unwrap() {}
        let bufs = s.add_mesh_buffers(build).unwrap();
        let id = s.finish_add_mesh(build).unwrap();
        let (i, g) = (id[0], id[1]);
        assert_eq!(
            bufs,
            vec![
                s.positions_ptr(i, g).unwrap(),
                s.positions_len(i, g).unwrap(),
                s.normals_ptr(i, g).unwrap(),
                s.normals_len(i, g).unwrap(),
                s.indices_ptr(i, g).unwrap(),
                s.indices_len(i, g).unwrap(),
            ]
        );
        assert_eq!(bufs[3], 12);
        assert!(s.add_mesh_buffers(build).is_err());
        // The content stamp goes up with a compaction of the mesh.
        let before = s.content_stamp(i, g).unwrap();
        s.compact(i, g).unwrap();
        assert!(s.content_stamp(i, g).unwrap() > before);
        let other = s.begin_add_mesh_upload(4, 2, None, 0, false).unwrap();
        s.drop_add_mesh(other);
        assert!(s.finish_add_mesh(other).is_err());
    }

    /// An edit id list without refs.
    fn ids(list: &[u32]) -> Vec<u32> {
        let mut v = vec![list.len() as u32, 0];
        v.extend_from_slice(list);
        v
    }

    #[test]
    fn edit_ops_atomic_with_refs() {
        let mut s = WasmScene::new();
        let (i, g) = add(&mut s, true);
        // Quarter turn about z and a shift; the origin is (1, 1, 0).
        let rot = [0., -1., 0., 1., 0., 0., 0., 0., 1.];
        s.set_placement(i, g, &rot, &[10., 20., 30.], None).unwrap();
        s.clear_history().unwrap();
        s.take_changes();
        // Add a vertex at world t (local -origin), a triangle on it by ref,
        // move vertex 2 to world (5, 25, 31), remove triangle 1.
        let mut ops = vec![1, 1];
        // Ids [1, _, 2]; slot 1 is id 0 of op 0.
        ops.extend([2, 3, 1, 1, 0, 2, 1, 0, 0]);
        ops.push(0);
        ops.extend(ids(&[2]));
        ops.push(3);
        ops.extend(ids(&[1]));
        let values = [10., 20., 30., 5., 25., 31.];
        let added = s.edit(i, g, &ops, &values).unwrap();
        assert_eq!(added, vec![1, 4, 1, 2, 0, 0]);
        let id = s.id(i, g).unwrap();
        let mesh = s.scene.mesh(id).unwrap();
        assert_eq!(mesh.positions()[4], [-1., -1., 0.]);
        assert_eq!(mesh.positions()[2], [4., 4., 1.]);
        assert_eq!(mesh.triangle(2), Some([1, 4, 2]));
        assert_eq!(mesh.triangle(1), None);
        assert!(s.can_undo());
        assert!(s.undo().unwrap());
        assert!(!s.can_undo());
        assert_eq!(s.vertex_count(i, g).unwrap(), 4);
        s.take_changes();

        // A failing op after one that wrote: nothing is left, no step.
        let mut bad = vec![1, 1, 3];
        bad.extend(ids(&[9]));
        assert_eq!(
            code(s.edit(i, g, &bad, &[0., 0., 0.])),
            codes::INVALID_INDEX
        );
        // A bad ref is caught before the scene is touched.
        let bad_ref = [1, 1, 4, 1, 1, 0, 0, 0, 1];
        assert_eq!(
            code(s.edit(i, g, &bad_ref, &[0., 0., 0.])),
            codes::INVALID_INDEX
        );
        assert_eq!(s.vertex_slots(i, g).unwrap(), 5);
        assert_eq!(s.vertex_count(i, g).unwrap(), 4);
        // Nothing was recorded: the redo step survives.
        assert!(!s.can_undo() && s.can_redo());
        assert_eq!(s.take_changes(), vec![0]);
        assert_eq!(code(s.edit(i, g + 1, &[], &[])), codes::UNKNOWN_MESH);
    }

    #[test]
    fn compact_reports_all_and_capacity() {
        let mut s = WasmScene::new();
        let (i, g) = add(&mut s, true);
        let mut ops = vec![3];
        ops.extend(ids(&[1]));
        ops.push(4);
        ops.extend(ids(&[3]));
        s.edit(i, g, &ops, &[]).unwrap();
        s.take_changes();
        s.compact(i, g).unwrap();
        let all = FLAG_POSITIONS_ALL | FLAG_NORMALS_ALL | FLAG_COLORS_ALL | FLAG_INDICES_ALL;
        assert_eq!(
            s.take_changes(),
            vec![1, i, g, FLAG_CAPACITY | all, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(s.vertex_slots(i, g).unwrap(), 3);
        assert!(!s.can_undo());
        assert_eq!(code(s.compact(i, g + 1)), codes::UNKNOWN_MESH);
    }

    /// The open upper half of a UV sphere, outward, with the fans of two
    /// vertices removed (two holes) and two triangles flipped.
    fn damaged_hemisphere(s: &mut WasmScene, mode: u8) -> (u32, u32) {
        let (stacks, slices) = (8u32, 32u32);
        let mut pos = vec![0., 0., 1.];
        for i in 1..=stacks {
            let theta = std::f64::consts::FRAC_PI_2 * f64::from(i) / f64::from(stacks);
            for j in 0..slices {
                let phi = std::f64::consts::TAU * f64::from(j) / f64::from(slices);
                pos.extend([
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ]);
            }
        }
        let ring = |i: u32, j: u32| 1 + (i - 1) * slices + j % slices;
        let mut tris = Vec::new();
        for j in 0..slices {
            tris.push([0, ring(1, j), ring(1, j + 1)]);
            for i in 1..stacks {
                let (a, b, c, d) = (
                    ring(i, j),
                    ring(i + 1, j),
                    ring(i + 1, j + 1),
                    ring(i, j + 1),
                );
                tris.extend([[a, b, c], [a, c, d]]);
            }
        }
        let holes = [ring(3, 4), ring(5, 20)];
        tris.retain(|t| !t.iter().any(|v| holes.contains(v)));
        for k in [40, 200] {
            tris[k].swap(1, 2);
        }
        let id = s
            .add_mesh(&pos, tris.as_flattened(), None, mode, true)
            .unwrap();
        (id[0], id[1])
    }

    #[test]
    fn repair_pipeline_one_undo_step() {
        let mut s = WasmScene::new();
        let (i, g) = damaged_hemisphere(&mut s, 0);
        let a = s.analyze(i, g, 0., false, 1000).unwrap();
        assert_eq!(a[0], 3.);
        // The last two values: 6 orientation conflicts, not checked.
        assert_eq!(&a[a.len() - 2..], &[6., -1.]);
        let triangles = s.triangle_count(i, g).unwrap();
        let nan = f64::NAN;
        let ops = [3., 4., nan, 2., nan, nan, 0., 1., nan, nan, nan];
        let r = s.repair(i, g, &ops, &[]).unwrap();
        // Orient flipped the two triangles; two loops filled, one skipped.
        assert_eq!(r[0], 2.);
        assert_eq!(r[4], 2.);
        let a = s.analyze(i, g, 0., true, 5).unwrap();
        assert_eq!(a[0], 1.);
        assert_eq!(&a[a.len() - 2..], &[0., 0.]);
        assert!(s.undo().unwrap());
        assert!(!s.can_undo());
        assert_eq!(s.triangle_count(i, g).unwrap(), triangles);

        // A failing op rolls the whole list back.
        let bad = [3., 4., nan, nan, nan, nan, 0., 1., nan, nan, nan];
        assert_eq!(code(s.repair(i, g, &bad, &[])), codes::INVALID_ARGUMENT);
        assert!(!s.can_undo());
        assert_eq!(
            code(s.analyze(i, g, -1., false, 10)),
            codes::INVALID_ARGUMENT
        );
    }

    #[test]
    fn auto_repair_one_undo_step() {
        let mut s = WasmScene::new();
        let (i, g) = damaged_hemisphere(&mut s, 0);
        let triangles = s.triangle_count(i, g).unwrap();
        let nan = f64::NAN;
        let r = s.repair(i, g, &[15., nan, nan, nan, nan], &[]).unwrap();
        // Each stage is the code of its op, then its report, in run order:
        // weld, degenerate, orient (two flipped), components, fill.
        assert_eq!(r[0], 0.);
        assert_eq!(r[3], 1.);
        assert_eq!(r[6], 3.);
        assert_eq!(r[7], 2.);
        assert_eq!(r[11], 2.);
        assert_eq!(r[15], 4.);
        assert!(r[16] > 0.);
        assert_eq!(s.analyze(i, g, 0., false, 10).unwrap()[0], 0.);
        assert!(s.undo().unwrap());
        assert!(!s.can_undo());
        assert_eq!(s.triangle_count(i, g).unwrap(), triangles);

        // A bad tolerance changes nothing and records nothing.
        let bad = [15., -1., nan, nan, nan];
        assert_eq!(code(s.repair(i, g, &bad, &[])), codes::INVALID_ARGUMENT);
        assert!(!s.can_undo());
        assert_eq!(s.triangle_count(i, g).unwrap(), triangles);
    }

    #[test]
    fn stale_loops_error_code() {
        let mut s = WasmScene::new();
        let (i, g) = damaged_hemisphere(&mut s, 2);
        s.repair(i, g, &[3.], &[]).unwrap();
        let a = s.analyze(i, g, 0., false, 10).unwrap();
        // The first loop's id, as the fill op takes it.
        let id: Vec<u32> = a[1..5].iter().map(|&v| v as u32).collect();
        let nan = f64::NAN;
        let fill = [4., nan, nan, nan, 1., 0., 0., nan, nan, nan];
        s.repair(i, g, &fill, &id).unwrap();
        s.clear_history().unwrap();
        assert_eq!(code(s.repair(i, g, &fill, &id)), codes::STALE_LOOPS);
        assert!(!s.can_undo());
        // Self-intersection needs the BVH, stale under Manual.
        assert_eq!(code(s.analyze(i, g, 0., true, 10)), codes::STALE_BVH);
        s.update(i, g).unwrap();
        let a = s.analyze(i, g, 0., true, 10).unwrap();
        assert_eq!(a[a.len() - 1], 0.);
        // A Lazy mesh is brought up to date first.
        s.set_mode(i, g, 1).unwrap();
        let mut ops = vec![0];
        ops.extend(ids(&[0]));
        s.edit(i, g, &ops, &[0., 0., 1.01]).unwrap();
        assert!(!s.scene.is_current(s.id(i, g).unwrap()).unwrap());
        s.analyze(i, g, 0., true, 10).unwrap();
    }

    #[test]
    fn section_results_in_world_space() {
        let mut s = WasmScene::new();
        let (ai, ag) = octahedron(&mut s, [0., 0., 0.], 1.);
        let (bi, bg) = octahedron(&mut s, [3., 0., 0.], 1.);
        let z0 = [0., 0., 0., 0., 0., 1.];
        let sec = s.add_section(&z0, &[ai, ag, bi, bg], true).unwrap();
        let r = s.section(sec, bi, bg).unwrap();
        // Current; the equator square: length 4·√2, area 2.
        assert_eq!(r[0], 0.);
        assert!((r[1] - 4. * 2f64.sqrt()).abs() < 1e-6);
        assert!((r[2].abs() - 2.).abs() < 1e-6);
        assert_eq!(&r[3..5], &[1., 1.]);
        let n = r[5] as usize;
        assert_eq!(n, 4);
        let pts = &r[6..6 + 3 * n];
        assert!(pts.chunks(3).all(|p| p[2] == 0. && (p[0] - 3.).abs() <= 1.));
        let fill = &r[6 + 3 * n..];
        assert_eq!(fill[0], 1.);
        let fp = fill[1] as usize;
        let ft = fill[2 + 3 * fp] as usize;
        assert_eq!(fill.len(), 3 + 3 * fp + 3 * ft);
        // Fill points are mapped to the world: around x = 3, at z = 0.
        assert!(
            fill[2..2 + 3 * fp]
                .chunks(3)
                .all(|p| p[2] == 0. && p[0] >= 2.)
        );

        let g = s.section_gap(sec, ai, ag, bi, bg).unwrap();
        assert!((g[0] - 1.).abs() < 1e-6 && (g[1] - 1.).abs() < 1e-6 && (g[4] - 2.).abs() < 1e-6);

        s.set_section_plane(sec, &[0., 0., 5., 0., 0., 2.]).unwrap();
        assert_eq!(
            s.section_desc_plane(sec).unwrap(),
            vec![0., 0., 5., 0., 0., 1.]
        );
        s.set_section_fill(sec, false).unwrap();
        s.update_sections().unwrap();
        assert_eq!(s.section(sec, ai, ag).unwrap(), vec![0., 0., 0., 0., 0.]);
        assert_eq!(s.section_gap(sec, ai, ag, bi, bg).unwrap(), []);
        s.set_section_meshes(sec, &[ai, ag]).unwrap();
        assert_eq!(code(s.section(sec, bi, bg)), codes::INVALID_ARGUMENT);
        assert_eq!(
            code(s.set_section_plane(sec, &[0.; 6])),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.add_section(&z0[..5], &[ai, ag], false)),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.add_section(&z0, &[ai], false)),
            codes::INVALID_ARGUMENT
        );
        s.remove_section(sec).unwrap();
        assert_eq!(code(s.section(sec, ai, ag)), codes::UNKNOWN_SECTION);
        assert_eq!(code(s.set_section_fill(sec, true)), codes::UNKNOWN_SECTION);
        assert_eq!(code(s.remove_section(sec)), codes::UNKNOWN_SECTION);
        assert_eq!(code(s.section_desc_plane(sec)), codes::UNKNOWN_SECTION);
        assert_eq!(
            code(s.section_gap(sec, ai, ag, bi, bg)),
            codes::UNKNOWN_SECTION
        );
    }

    #[test]
    fn join_merges_source_and_reports_counts() {
        use fast_mesh_ops::LoopId;
        use fast_mesh_ops::boundary_loops;

        // Quad A at z=0 facing -z, quad B at z=1 facing +z.
        let mut s = WasmScene::new();
        let a = s
            .add_mesh(
                &[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.],
                &[0, 2, 1, 0, 3, 2],
                None,
                0,
                false,
            )
            .unwrap();
        let b = s
            .add_mesh(
                &[0., 0., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1.],
                &[0, 1, 2, 0, 2, 3],
                None,
                0,
                false,
            )
            .unwrap();
        let (ai, ag) = (a[0], a[1]);
        let (bi, bg) = (b[0], b[1]);
        let flat = |id: LoopId| {
            [
                id.min_vertex,
                id.len,
                id.hash as u32,
                (id.hash >> 32) as u32,
            ]
        };
        let loop_id = |s: &WasmScene, i: u32, g: u32| {
            boundary_loops(s.scene.mesh(s.id(i, g).unwrap()).unwrap()).unwrap()[0].id
        };
        let la = flat(loop_id(&s, ai, ag));
        let lb = flat(loop_id(&s, bi, bg));
        let loops = [la[0], la[1], la[2], la[3], lb[0], lb[1], lb[2], lb[3]];

        assert_eq!(
            code(s.join(ai, ag, bi, bg, &loops[..4], f64::NAN)),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.join(ai, ag, bi, bg, &loops, -1.0)),
            codes::INVALID_ARGUMENT
        );
        let fake = [loops[0], loops[1], loops[2], loops[3], 99, 4, 7, 0];
        assert_eq!(
            code(s.join(ai, ag, bi, bg, &fake, f64::NAN)),
            codes::STALE_LOOPS
        );

        assert_eq!(
            s.join(ai, ag, bi, bg, &loops, f64::NAN).unwrap(),
            vec![4.0, 2.0, 8.0]
        );
        assert_eq!(s.triangle_count(ai, ag).unwrap(), 2 + 2 + 8);
        assert_eq!(code(s.triangle_count(bi, bg)), codes::UNKNOWN_MESH);
        // The join is one step: undo brings B back, reported like an add (#271).
        assert!(s.is_restorable(bi, bg));
        s.take_changes();
        assert!(s.undo().unwrap());
        assert!(!s.can_undo());
        assert_eq!(s.triangle_count(ai, ag).unwrap(), 2);
        assert_eq!(s.triangle_count(bi, bg).unwrap(), 2);
        let added = FLAG_ADDED | FLAG_POSITIONS_ALL | FLAG_INDICES_ALL;
        let back = s.take_changes();
        assert_eq!(back[0], 2);
        let b_at = (1..back.len())
            .find(|&k| back[k..k + 2] == [bi, bg])
            .unwrap();
        assert_eq!(back[b_at + 2] & added, added);
        // Redo takes it out again, and says that undo can bring it back.
        assert!(s.redo().unwrap());
        assert_eq!(code(s.triangle_count(bi, bg)), codes::UNKNOWN_MESH);
        let gone = s.take_changes();
        let b_at = (1..gone.len())
            .find(|&k| gone[k..k + 2] == [bi, bg])
            .unwrap();
        assert_eq!(gone[b_at + 2], FLAG_REMOVED | FLAG_RESTORABLE);
    }

    #[test]
    fn sculpt_stroke_survives_split_stretched_and_vertices_in_sphere() {
        let mut s = WasmScene::new();
        // 5x5 flat grid on [-2, 2]^2 (z = 0), origin = [0, 0, 0].
        let mut pos = Vec::new();
        let mut tris = Vec::new();
        let n = 4u32;
        for y in 0..=n {
            for x in 0..=n {
                pos.extend([f64::from(x) - 2.0, f64::from(y) - 2.0, 0.0]);
            }
        }
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                tris.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        let id = s.add_mesh(&pos, &tris, None, 0, true).unwrap();
        let (i, g) = (id[0], id[1]);

        // vertices_in_sphere around center (0, 0, 0) with radius 0.5 finds only center vertex 12.
        assert_eq!(
            s.vertices_in_sphere(i, g, 0., 0., 0., 0.5).unwrap(),
            vec![12]
        );
        assert_eq!(
            code(s.vertices_in_sphere(i, g + 1, 0., 0., 0., 1.)),
            codes::UNKNOWN_MESH
        );
        assert_eq!(
            code(s.vertices_in_sphere(i, g, 0., 0., 0., -0.1)),
            codes::INVALID_ARGUMENT
        );

        // captureStrokeState cannot mix in a multi-op list.
        assert_eq!(
            code(s.repair(i, g, &[3., 18., 0.], &[])),
            codes::INVALID_ARGUMENT
        );

        // Capture rest sizes + stroke state, grab from stroke, splitStretched, then grab from stroke again!
        assert_eq!(s.repair(i, g, &[17., 0.], &[]).unwrap(), Vec::<f64>::new());
        assert_eq!(s.repair(i, g, &[18., 0.], &[]).unwrap(), Vec::<f64>::new());
        let nan = f64::NAN;
        let grab1 = [9., 0., 0., 0., 1.5, 1., 1., 0., 0., 2., 0., nan, 1., 1.];
        let r1 = s.repair(i, g, &grab1, &[]).unwrap();
        assert!(r1[0] > 0.0);
        // Single splitStretched op routes through Scene::split_stretched so stroke state survives.
        let split = [16., 0., 0., 0., 10., nan, nan, 1., nan];
        let sr = s.repair(i, g, &split, &[]).unwrap();
        assert!(sr[0] > 0.0, "expected splits: {sr:?}");
        // Subsequent from_stroke grab evaluates relative to the original z=0 rest plane (not cumulative!).
        let grab2 = [9., 0., 0., 0., 1.5, 1., 1., 0., 0., 2.5, 0., nan, 1., 1.];
        let r2 = s.repair(i, g, &grab2, &[]).unwrap();
        assert!(r2[0] > 0.0);
        let mesh = s.scene.mesh(s.id(i, g).unwrap()).unwrap();
        assert!((mesh.positions()[12][2] - 2.5).abs() < 1e-4);
    }

    #[test]
    fn wave4_boolean_inspect_curve_and_align_bindings_roundtrip() {
        let mut s = WasmScene::new();
        // Unit cube centered at origin (-1..1).
        let cube_pos = [
            -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
            1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
        ];
        let cube_idx = [
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6,
            5, 3, 0, 4, 3, 4, 7,
        ];
        let a = s.add_mesh(&cube_pos, &cube_idx, None, 0, false).unwrap();
        let b = s.add_mesh(&cube_pos, &cube_idx, None, 0, false).unwrap();
        s.set_placement(
            b[0],
            b[1],
            &[1., 0., 0., 0., 1., 0., 0., 0., 1.],
            &[0.5, 0., 0.],
            None,
        )
        .unwrap();

        // Symmetry & overhangs & quality & geodesic & path & deviation.
        let sym = s.detect_symmetry_plane(a[0], a[1]).unwrap();
        assert_eq!(sym.len(), 8);
        assert!(sym[7] > 0.99);

        let ov = s.overhangs(a[0], a[1], &[0., 0., 1.], 45.0).unwrap();
        assert_eq!(ov[1], 2.0);
        assert_eq!(ov.len(), 3 + 8);

        let qual = s.triangle_quality(a[0], a[1], 1, 15.0).unwrap();
        assert_eq!(qual[4], 12.0);
        assert_eq!(qual.len(), 5 + 8);

        let geo = s
            .geodesic_distances(a[0], a[1], &[0], 0, f64::NAN, f64::NAN)
            .unwrap();
        assert_eq!(geo[0], 0.0);
        let path = s.shortest_surface_path(a[0], a[1], &geo, &[0], 6).unwrap();
        assert!(path.len().is_multiple_of(7));
        assert!(path.len() >= 14);

        let dev = s
            .mesh_deviation(a[0], a[1], b[0], b[1], true, f64::NAN)
            .unwrap();
        assert_eq!(dev[5], 8.0);
        assert_eq!(dev.len(), 6 + 8);

        // Landmarks + multistage registration.
        let lm = s
            .align_by_landmarks(
                a[0],
                a[1],
                b[0],
                b[1],
                &[0., 0., 0., 1., 0., 0., 0., 1., 0.],
                &[0.5, 0., 0., 1.5, 0., 0., 0.5, 1., 0.],
                false,
            )
            .unwrap();
        assert_eq!(lm.len(), 17);
        assert!(lm[16] < 1e-6);

        let ms = s
            .register_multistage(
                a[0],
                a[1],
                b[0],
                b[1],
                1,
                2,
                30.,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                200.,
                f64::NAN,
            )
            .unwrap();
        assert_eq!(ms.len(), 35);
        assert!(ms[31] < 0.05);

        // Boolean with options + booleanBatch.
        let batch = s
            .boolean_batch(
                &[a[0], a[1], b[0], b[1]],
                &[1., 0., 0., 0., 0., 1.],
                0,
                1,
                f64::NAN,
                true,
                0.25,
                f64::NAN,
            )
            .unwrap();
        assert_eq!(batch.len(), 2);

        let bw = s
            .boolean_with_options(
                a[0],
                a[1],
                b[0],
                b[1],
                0,
                0,
                f64::NAN,
                true,
                f64::NAN,
                f64::NAN,
            )
            .unwrap();
        assert_eq!(bw.len(), 4);
        assert!(bw[2] > 0);

        // Curve trim on a 2x2 planar quad mesh.
        let sheet = s
            .add_mesh(
                &[0., 0., 0., 2., 0., 0., 2., 2., 0., 0., 2., 0.],
                &[0, 1, 2, 0, 2, 3],
                None,
                0,
                false,
            )
            .unwrap();
        let tr = s
            .trim_by_curve(
                sheet[0],
                sheet[1],
                &[0.5, 0.5, 0., 1.5, 0.5, 0., 1.0, 1.5, 0.],
                true,
                true,
                0.,
            )
            .unwrap();
        assert_eq!(tr.len(), 9);
        assert_eq!(tr[..2], [f64::from(sheet[0]), f64::from(sheet[1])]);
        // Both triangles lost surface; the loop encloses half a unit.
        assert_eq!(tr[3..6], [2., 0., 2.]);
        assert!((tr[6] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn wave5_continuity_fill_bridge_and_dyntopo_roundtrip() {
        let mut s = WasmScene::new();

        // 4x4 planar grid missing the center quad -> outer rim + 4-vertex inner hole.
        let mut positions = Vec::new();
        for y in 0..4u32 {
            for x in 0..4u32 {
                positions.extend_from_slice(&[x as f64, y as f64, 0.0]);
            }
        }
        let mut indices = Vec::new();
        for y in 0..3u32 {
            for x in 0..3u32 {
                if x == 1 && y == 1 {
                    continue;
                }
                let v0 = y * 4 + x;
                let v1 = v0 + 1;
                let v2 = v0 + 5;
                let v3 = v0 + 4;
                indices.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
            }
        }
        let grid = s.add_mesh(&positions, &indices, None, 0, false).unwrap();
        let gid = s.id(grid[0], grid[1]).unwrap();
        let loops = fast_mesh_ops::boundary_loops(s.scene.mesh(gid).unwrap()).unwrap();
        let inner_idx = loops.iter().position(|l| l.vertices.len() == 4).unwrap() as u32;
        let inner_verts = loops[inner_idx as usize].vertices.clone();
        let initial_tris = s.scene.mesh(gid).unwrap().triangle_count();
        let fill_stats = s
            .fill_hole_continuity(
                grid[0],
                grid[1],
                inner_idx,
                2,
                f64::NAN,
                f64::NAN,
                true,
                1.0,
                true,
            )
            .unwrap();
        assert_eq!(fill_stats.len(), 8);
        assert!(fill_stats[2] > 0.0);
        assert_eq!(fill_stats[5], 2.0);
        assert!(s.scene.mesh(gid).unwrap().triangle_count() > initial_tris);

        s.undo().unwrap();
        assert_eq!(s.scene.mesh(gid).unwrap().triangle_count(), initial_tris);

        let fill_all = s
            .fill_all_holes_continuity(grid[0], grid[1], 1, f64::NAN, f64::NAN, false, 0.0, true)
            .unwrap();
        assert_eq!(fill_all.len(), 8);
        assert_eq!(fill_all[5], 1.0);
        assert!(fill_all[2] >= 2.0);

        s.undo().unwrap();
        let bridged = s
            .bridge_partial_arcs(
                grid[0],
                grid[1],
                &inner_verts[0..2],
                &inner_verts[2..4],
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                f64::NAN,
            )
            .unwrap();
        assert_eq!(bridged.len(), 5);
        assert!(bridged[0] > 0);

        // Localized dyntopo + dyntopo_dab + stroke undo coalescing.
        let sheet = s
            .add_mesh(
                &[0., 0., 0., 4., 0., 0., 4., 4., 0., 0., 4., 0.],
                &[0, 1, 2, 0, 2, 3],
                None,
                0,
                false,
            )
            .unwrap();
        let patch = s
            .apply_dyntopo(
                sheet[0],
                sheet[1],
                &[2.0, 2.0, 0.0],
                4.0,
                1,
                0.75,
                0.35,
                1.2,
                true,
                false,
                &[],
            )
            .unwrap();
        assert!(patch.len() >= 6);
        assert!(patch[0] > 0.0);

        s.begin_step();
        let dab_ops = [5.0, 2.0, 2.0, 0.0, 2.0, 1.0, 0.25, 0.0];
        let dab_res = s
            .dyntopo_dab(
                sheet[0], sheet[1], &dab_ops, 2, 0.25, 0.4, 1.33, true, false,
            )
            .unwrap();
        assert!(dab_res.len() >= 8);
        assert_eq!(dab_res[6] as usize + dab_res[7] as usize + 8, dab_res.len());
        s.end_step().unwrap();
        s.undo().unwrap();
    }
}
