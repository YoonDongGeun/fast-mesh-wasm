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

use fast_mesh_ops::{AnalyzeOptions, analyze};
use fast_mesh_query::Cull;
use fast_mesh_scene::{
    ColorRule, Coloring, Colormap, ContactFilter, DistanceField, FieldBudget, FieldId, MeshBuild,
    MeshId, MeshOptions, OutOfRange, Placement, Plane, Scene, SceneClosestOptions, SceneError,
    SceneOptions, SceneRayOptions, SectionDesc, SectionId, SectionStatus, SignMode, StepState,
    UpdateMode,
};
use fast_mesh_store::{Mesh, MeshBuilder};
use wasm_bindgen::prelude::*;

use crate::edit::{
    decode_edit, decode_repair, encode_added, encode_analysis, run_edit, run_repair,
};
use crate::error::Error;
use crate::render::{Dirty, Record, RenderState, encode};

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
}

/// A mesh being added in steps: its inputs still uploading into wasm
/// memory ([`MeshBuilder`]) and its store still building, or its store
/// built and its BVH (and vertex normals) building.
enum PendingAdd {
    /// The store uploads and builds in steps; `options` and `normals`
    /// start the BVH build once it is done.
    Store {
        build: Box<MeshBuilder>,
        options: MeshOptions,
        normals: bool,
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
) -> Result<(Box<MeshBuild>, UpdateMode, bool), Error> {
    let mesh = build.finish().ok_or(SceneError::Corrupted)?;
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
            },
        );
        Ok(id)
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
            } => {
                if store.step(budget as usize)? {
                    let (b, mode, normals) = start_build(store, options, normals)?;
                    self.adds.insert(build, PendingAdd::Bvh(b, mode, normals));
                } else {
                    self.adds.insert(
                        build,
                        PendingAdd::Store {
                            build: store,
                            options,
                            normals,
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
            } => {
                while !store.step(usize::MAX)? {}
                start_build(store, options, normals)?
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

    /// Removes a mesh; its id never matches again.
    pub fn remove_mesh(&mut self, index: u32, generation: u32) -> Result<(), Error> {
        let id = self.id(index, generation)?;
        self.scene.remove_mesh(id)?;
        self.render.remove(&id);
        Ok(())
    }

    /// Sets the placement from a row-major rotation (9 values) and a
    /// translation (3 values). The rotation must be a proper rotation.
    pub fn set_placement(
        &mut self,
        index: u32,
        generation: u32,
        rotation: &[f64],
        translation: &[f64],
    ) -> Result<(), Error> {
        let id = self.id(index, generation)?;
        let r: [f64; 9] = rotation
            .try_into()
            .map_err(|_| Error::invalid("rotation must have 9 values"))?;
        let translation: [f64; 3] = translation
            .try_into()
            .map_err(|_| Error::invalid("translation must have 3 values"))?;
        let placement = Placement {
            rotation: [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]],
            translation,
        };
        Ok(self.scene.set_placement(id, placement)?)
    }

    /// The placement: the row-major rotation (9 values), then the
    /// translation (3 values).
    pub fn placement(&self, index: u32, generation: u32) -> Result<Vec<f64>, Error> {
        let p = self.scene.placement(self.id(index, generation)?)?;
        let mut out = p.rotation.as_flattened().to_vec();
        out.extend_from_slice(&p.translation);
        Ok(out)
    }

    /// The world matrix of the local render coordinates,
    /// `placement · translate(origin)`, column-major (16 values).
    pub fn matrix4(&self, index: u32, generation: u32) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let p = self.scene.placement(id)?;
        let o = self.scene.mesh(id)?.origin();
        let r = p.rotation;
        // The last column is R·o + t.
        let w = p.apply(o);
        Ok(vec![
            r[0][0], r[1][0], r[2][0], 0.0, //
            r[0][1], r[1][1], r[2][1], 0.0, //
            r[0][2], r[1][2], r[2][2], 0.0, //
            w[0], w[1], w[2], 1.0,
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
                match (self.render.get_mut(&c.id), self.scene.mesh(c.id)) {
                    (Some(state), Ok(mesh)) => {
                        let normals = self.scene.normals(c.id).ok().flatten();
                        let normals_len = normals.map_or(0, <[_]>::len);
                        // Out of memory: report everything rather than lose
                        // the other meshes' changes.
                        state
                            .apply(mesh, &c, normals_len)
                            .unwrap_or_else(|_| Dirty::all())
                    }
                    _ => Dirty::all(),
                }
            };
            records.push(Record {
                mesh_index: c.id.index(),
                mesh_generation: c.id.generation(),
                added: c.added,
                removed: c.removed,
                placement_changed: c.placement_changed,
                dirty,
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
                    placement_changed: false,
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
    /// remove vertices) with their id lists, whose refs name ids added by
    /// earlier ops of the list; `values` holds the world positions, mapped
    /// to local with the placement inverse and the origin in `f64`, then
    /// stored as `f32`. Returns `[n, id × n]` per op: the new vertex or
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
    /// [`crate::edit::encode_analysis`]. `weld_tolerance` (local units,
    /// equal to world units) groups coincident vertices; lists with a count
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

    /// Runs repair operations on a mesh, in order, in one transaction: one
    /// undo step, and nothing changes when an operation fails. `ops` (`f64`)
    /// and `loops` (`u32`, loop ids for fill holes) are laid out as in
    /// [`crate::edit`]; returns the reports concatenated in op order.
    /// Lengths and areas in the options are in local units, equal to world
    /// units.
    pub fn repair(
        &mut self,
        index: u32,
        generation: u32,
        ops: &[f64],
        loops: &[u32],
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let decoded = decode_repair(ops, loops)?;
        Ok(self.scene.edit(id, |tx| run_repair(tx, &decoded))?)
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

#[cfg(test)]
mod tests {
    use fast_mesh_scene::SceneError;

    use crate::render::{
        FLAG_ADDED, FLAG_CAPACITY, FLAG_COLORS_ALL, FLAG_INDICES_ALL, FLAG_NORMALS_ALL,
        FLAG_PLACEMENT, FLAG_POSITIONS_ALL, FLAG_REMOVED,
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
        s.set_placement(i, g, &rot, &[10., 20., 30.]).unwrap();
        let p = s.placement(i, g).unwrap();
        assert_eq!(&p[..9], &rot);
        assert_eq!(&p[9..], &[10., 20., 30.]);
        let m = s.matrix4(i, g).unwrap();
        // Column-major: column j is m[4j..4j+4].
        assert_eq!(&m[0..4], &[0., 1., 0., 0.]);
        assert_eq!(&m[4..8], &[-1., 0., 0., 0.]);
        assert_eq!(&m[8..12], &[0., 0., 1., 0.]);
        assert_eq!(&m[12..16], &[9., 21., 30., 1.]);
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
    fn error_codes_map() {
        let mut s = WasmScene::new();
        let (i, g) = add(&mut s, true);
        // Reachable through the bindings.
        assert_eq!(code(s.remove_mesh(i, g + 1)), codes::UNKNOWN_MESH);
        assert_eq!(code(s.matrix4(i + 7, 0)), codes::UNKNOWN_MESH);
        assert_eq!(
            code(s.set_placement(i, g, &[2.; 9], &[0.; 3])),
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            code(s.set_placement(i, g, &IDENTITY, &[0.; 2])),
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
        s.set_placement(i, g, &IDENTITY, &[1., 0., 0.]).unwrap();
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
        s.set_placement(i, g, &IDENTITY, &[2., 0., 0.]).unwrap();
        s.set_placement(i, g, &IDENTITY, &[3., 0., 0.]).unwrap();
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
        s.set_placement(bi, bg, &IDENTITY, &[0., 0., 10.]).unwrap();
        let h = s
            .raycast(&[3., 0., 20.], &[0., 0., -1.], 0., inf, 1)
            .unwrap();
        assert_eq!(&h[..2], &[f64::from(bi), f64::from(bg)]);
        assert!((h[5] - 11.).abs() < 1e-6);
        s.set_placement(bi, bg, &IDENTITY, &[0., 0., 0.]).unwrap();

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

        s.set_placement(bi, bg, &IDENTITY, &[1., 0., 0.]).unwrap();
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

        s.set_placement(bi, bg, &IDENTITY, &[1., 0., 0.]).unwrap();
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
        s.set_placement(i, g, &rot, &[10., 20., 30.]).unwrap();
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
}
