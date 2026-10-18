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

use std::collections::BTreeMap;

use fast_mesh_scene::{
    MeshId, MeshOptions, Placement, Scene, SceneError, SceneOptions, UpdateMode,
};
use fast_mesh_store::Mesh;
use wasm_bindgen::prelude::*;

use crate::error::Error;
use crate::render::{Dirty, Record, RenderState, encode};

/// A scene of editable meshes with render buffers and change records.
#[wasm_bindgen]
pub struct WasmScene {
    scene: Scene,
    /// The render state of every live mesh.
    render: BTreeMap<MeshId, RenderState>,
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

impl Default for WasmScene {
    fn default() -> Self {
        WasmScene::new()
    }
}

impl WasmScene {
    /// The live mesh id behind `(index, generation)`.
    fn id(&self, index: u32, generation: u32) -> Result<MeshId, Error> {
        self.scene
            .mesh_id(index, generation)
            .ok_or_else(|| SceneError::UnknownMesh.into())
    }

    fn mesh(&self, index: u32, generation: u32) -> Result<&Mesh, Error> {
        Ok(self.scene.mesh(self.id(index, generation)?)?)
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
        }
    }

    // ---- Meshes ----

    /// Adds a mesh from world positions and returns `[index, generation]`.
    /// Positions are stored as `p - origin` in `f32`; without `origin` it
    /// is the bounding-box centre. `mode` is 0 (Eager), 1 (Lazy) or 2
    /// (Manual); `normals` turns vertex normals on. The placement starts as
    /// the identity. Nothing is added on error.
    pub fn add_mesh(
        &mut self,
        positions: &[f64],
        indices: &[u32],
        origin: Option<Vec<f64>>,
        mode: u8,
        normals: bool,
    ) -> Result<Vec<u32>, Error> {
        let mode = parse_mode(mode)?;
        let mesh = match origin {
            None => Mesh::from_buffers_f64(positions, indices)?,
            Some(o) => {
                let o: [f64; 3] = o
                    .as_slice()
                    .try_into()
                    .map_err(|_| Error::invalid("origin must have 3 values"))?;
                Mesh::from_buffers_f64_with_origin(positions, indices, o)?
            }
        };
        let options = MeshOptions {
            mode,
            ..MeshOptions::default()
        };
        let id = self.scene.add_mesh(mesh, options)?;
        if normals && let Err(e) = self.scene.set_normals(id, true) {
            // Added and removed before any report: the scene reports nothing.
            let _ = self.scene.remove_mesh(id);
            return Err(e.into());
        }
        let state = RenderState::new(self.scene.mesh(id)?);
        self.render.insert(id, state);
        Ok(vec![id.index(), id.generation()])
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
        Ok(self.scene.set_mode(id, parse_mode(mode)?)?)
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

        // Growth reports the capacity change; the normals buffer grew, so
        // it is "all".
        s.scene
            .edit(id, |tx| {
                let v = tx.add_vertices(&[[3., 0., 0.]])?;
                tx.add_triangles(&[[1, v[0], 2]])
            })
            .unwrap();
        let msg = s.take_changes();
        assert_eq!(&msg[..4], &[1, i, g, FLAG_CAPACITY | FLAG_NORMALS_ALL]);
        // Positions: vertex 4; normals: none listed; colours: none;
        // indices: triangle 2.
        assert_eq!(&msg[4..], &[1, 4, 1, 0, 0, 1, 2, 1, 0, 0]);

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
}
