//! `WasmScene` bindings for extrusions (`Scene::extrude_faces`,
//! `Scene::extrude_base`).

use fast_mesh_ops::{BaseLoops, ExtrudeDirection};
use fast_mesh_scene::{BasePlane, ExtrudeBaseOptions, ExtrudeFacesOptions};
use fast_mesh_store::StoreError;
use wasm_bindgen::prelude::*;

use super::{WasmScene, count, len, vec3};
use crate::edit::loop_id;
use crate::error::Error;

/// Reserves room for `additional` more elements of a result.
fn room<T>(out: &mut Vec<T>, additional: usize) -> Result<(), Error> {
    out.try_reserve_exact(additional)
        .map_err(|_| StoreError::OutOfMemory.into())
}

#[wasm_bindgen]
impl WasmScene {
    /// Extrudes the selected `triangles` of a mesh by a world `distance` and
    /// walls the selection's boundary, as one undo step. `mode` is 0 for each
    /// region's average normal, 1 for each vertex's, and 2 for the world
    /// `direction` (3 values, else empty). `check` looks for
    /// self-intersections. Returns `[regions, capTriangles, movedVertices,
    /// addedVertices, checked, wallCount, ...walls, ...selfIntersections]`.
    #[allow(clippy::too_many_arguments)]
    pub fn extrude_faces(
        &mut self,
        index: u32,
        generation: u32,
        triangles: &[u32],
        distance: f64,
        mode: u8,
        direction: &[f64],
        check: bool,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        let options = ExtrudeFacesOptions {
            distance,
            direction: match mode {
                0 => ExtrudeDirection::RegionNormal,
                1 => ExtrudeDirection::VertexNormals,
                2 => ExtrudeDirection::Fixed(vec3(direction, "direction must have 3 values")?),
                _ => {
                    return Err(Error::invalid(
                        "mode must be 0 (region), 1 (vertex) or 2 (direction)",
                    ));
                }
            },
            check_self_intersections: check,
        };
        let r = self.scene.extrude_faces(id, triangles, &options)?;
        let crossing = r.self_intersections.as_deref().unwrap_or_default();
        let mut out = Vec::new();
        room(&mut out, 6 + r.wall_triangles.len() + crossing.len())?;
        out.extend([
            len(r.regions),
            len(r.cap_triangles),
            len(r.moved_vertices),
            len(r.added_vertices),
            u32::from(r.self_intersections.is_some()),
            len(r.wall_triangles.len()),
        ]);
        out.extend_from_slice(&r.wall_triangles);
        out.extend_from_slice(crossing);
        Ok(out)
    }

    /// Extrudes boundary loops of a mesh along the world `direction` to a
    /// flat, capped base perpendicular to it, as one undo step. `plane` is 0
    /// for a base at the height `value` (along the opposite of `direction`)
    /// and 1 for one `value` past the mesh's furthest vertex. `which` is 0
    /// for the longest loop, 1 for all and 2 for the `loops` (4 words per
    /// loop id). `refine` refines the caps and `check` looks for
    /// self-intersections. Returns `[height, baseVertices, capVertices,
    /// wallTriangles, capTriangles, capped, flipped, checked, loopCount,
    /// ...(minVertex, len, hash low, hash high) per loop,
    /// ...selfIntersections]`.
    #[allow(clippy::too_many_arguments)]
    pub fn extrude_base(
        &mut self,
        index: u32,
        generation: u32,
        direction: &[f64],
        plane: u8,
        value: f64,
        which: u8,
        loops: &[u32],
        refine: bool,
        check: bool,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let loops = match which {
            0 => BaseLoops::Largest,
            1 => BaseLoops::All,
            2 => {
                let (words, rest) = loops.as_chunks::<4>();
                if !rest.is_empty() {
                    return Err(Error::invalid("a loop id has 4 words"));
                }
                let mut ids = Vec::new();
                room(&mut ids, words.len())?;
                ids.extend(words.iter().map(|&w| loop_id(w)));
                BaseLoops::Ids(ids)
            }
            _ => {
                return Err(Error::invalid(
                    "which must be 0 (largest), 1 (all) or 2 (ids)",
                ));
            }
        };
        let options = ExtrudeBaseOptions {
            direction: vec3(direction, "direction must have 3 values")?,
            plane: match plane {
                0 => BasePlane::Height(value),
                1 => BasePlane::Below(value),
                _ => return Err(Error::invalid("plane must be 0 (height) or 1 (below)")),
            },
            loops,
            refine,
            check_self_intersections: check,
        };
        let r = self.scene.extrude_base(id, &options)?;
        let base = &r.base;
        let crossing = base.self_intersections.as_deref().unwrap_or_default();
        let mut out = Vec::new();
        room(&mut out, 9 + 4 * base.loops.len() + crossing.len())?;
        out.extend([
            r.height,
            count(base.base_vertices),
            count(base.cap_vertices),
            count(base.wall_triangles),
            count(base.cap_triangles),
            f64::from(u8::from(base.capped)),
            f64::from(u8::from(base.flipped)),
            f64::from(u8::from(base.self_intersections.is_some())),
            count(base.loops.len()),
        ]);
        out.extend(base.loops.iter().flat_map(|l| {
            [
                f64::from(l.min_vertex),
                f64::from(l.len),
                f64::from(l.hash as u32),
                f64::from((l.hash >> 32) as u32),
            ]
        }));
        out.extend(crossing.iter().map(|&t| f64::from(t)));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::codes;

    /// A unit cube; triangles 10 and 11 are its +z face.
    const CUBE: [f64; 24] = [
        0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1.,
        1.,
    ];
    const CUBE_TRIS: [u32; 36] = [
        0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0, 2, 3, 0, 3, 1,
        4, 5, 7, 4, 7, 6,
    ];
    /// A square sheet facing +z.
    const SHEET: [f64; 12] = [0., 0., 0., 4., 0., 0., 4., 4., 0., 0., 4., 0.];
    const SHEET_TRIS: [u32; 6] = [0, 1, 2, 0, 2, 3];

    #[test]
    fn faces_return_their_report() {
        let mut s = WasmScene::new();
        let m = s.add_mesh(&CUBE, &CUBE_TRIS, None, 0, false).unwrap();
        let (i, g) = (m[0], m[1]);
        let r = s.extrude_faces(i, g, &[10, 11], 0.5, 0, &[], true).unwrap();
        assert_eq!(r[..6], [1, 2, 0, 4, 1, 8]);
        assert_eq!(r[6..], [12, 13, 14, 15, 16, 17, 18, 19]);
        assert_eq!(s.triangle_slots(i, g).unwrap(), 20);
        assert!(s.undo().unwrap());
        assert_eq!(s.mesh(i, g).unwrap().triangle_count(), 12);

        // Through the bottom: crossings follow the walls.
        let r = s
            .extrude_faces(i, g, &[10, 11], 3.0, 2, &[0.0, 0.0, -1.0], true)
            .unwrap();
        assert_eq!(r[5], 8);
        assert!(r.len() > 14);
        assert!(s.undo().unwrap());
        let r = s
            .extrude_faces(i, g, &[10, 11], -3.0, 1, &[], false)
            .unwrap();
        assert_eq!((r[4], r.len()), (0, 14));

        let invalid = |r: Result<Vec<u32>, Error>| r.unwrap_err().code;
        let bad = [
            s.extrude_faces(i, g, &[10], 0.5, 3, &[], true),
            s.extrude_faces(i, g, &[10], 0.5, 2, &[0.0, 1.0], true),
            s.extrude_faces(i, g, &[10], 0.0, 0, &[], true),
            s.extrude_faces(i, g, &[], 0.5, 0, &[], true),
        ];
        for r in bad {
            assert_eq!(invalid(r), codes::INVALID_ARGUMENT);
        }
        assert_eq!(
            invalid(s.extrude_faces(9, 0, &[10], 0.5, 0, &[], true)),
            codes::UNKNOWN_MESH
        );
    }

    #[test]
    fn bases_return_their_report() {
        let mut s = WasmScene::new();
        let m = s.add_mesh(&SHEET, &SHEET_TRIS, None, 0, false).unwrap();
        let (i, g) = (m[0], m[1]);
        let down = [0.0, 0.0, -1.0];
        let r = s
            .extrude_base(i, g, &down, 1, 2.0, 0, &[], false, true)
            .unwrap();
        assert_eq!(r[..9], [-2.0, 4.0, 0.0, 8.0, 2.0, 0.0, 0.0, 1.0, 1.0]);
        assert_eq!(r.len(), 13);
        assert_eq!(s.mesh(i, g).unwrap().triangle_count(), 12);
        assert!(s.undo().unwrap());
        // The loop id goes back in.
        let words: Vec<u32> = r[9..13].iter().map(|&w| w as u32).collect();
        let r = s
            .extrude_base(i, g, &down, 0, -1.0, 2, &words, false, false)
            .unwrap();
        assert_eq!((r[0], r[7], r[8]), (-1.0, 0.0, 1.0));
        assert!(s.undo().unwrap());
        let up = s
            .extrude_base(i, g, &[0.0, 0.0, 1.0], 1, 1.0, 1, &[], false, true)
            .unwrap();
        assert_eq!((up[0], up[6]), (-1.0, 1.0));
        assert!(s.undo().unwrap());

        let mut stale = words.clone();
        stale[2] ^= 1;
        let code = |r: Result<Vec<f64>, Error>| r.unwrap_err().code;
        let bad = s.extrude_base(i, g, &down, 1, 1.0, 2, &stale, true, true);
        assert_eq!(code(bad), codes::STALE_LOOPS);
        let bad = [
            s.extrude_base(i, g, &down, 1, 1.0, 2, &words[..3], true, true),
            s.extrude_base(i, g, &down, 2, 1.0, 0, &[], true, true),
            s.extrude_base(i, g, &down, 1, 1.0, 3, &[], true, true),
            s.extrude_base(i, g, &down[..2], 1, 1.0, 0, &[], true, true),
            s.extrude_base(i, g, &[0.0; 3], 1, 1.0, 0, &[], true, true),
            s.extrude_base(i, g, &down, 1, 0.0, 0, &[], true, true),
        ];
        for r in bad {
            assert_eq!(code(r), codes::INVALID_ARGUMENT);
        }
        assert!(!s.can_undo());
    }
}
