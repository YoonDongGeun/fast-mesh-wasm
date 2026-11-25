//! `WasmScene` bindings for convex hulls (`Scene::convex_hull`,
//! `Scene::convex_hull_points`).

use fast_mesh_scene::ConvexHullReport;
use wasm_bindgen::prelude::*;

use super::{WasmScene, len};
use crate::error::Error;

#[wasm_bindgen]
impl WasmScene {
    /// Adds the convex hull of the meshes `meshes` (flat `[index,
    /// generation, ...]`) as a new closed, outward-wound mesh, built from
    /// the vertices their live triangles use, as one undo step (undo removes
    /// it). With one shared placement the new mesh gets it, else it
    /// is built in the world at identity. `normals` turns its smooth vertex
    /// normals on. Returns `[index, generation, inputPoints, vertices,
    /// triangles]`.
    pub fn convex_hull(&mut self, meshes: &[u32], normals: bool) -> Result<Vec<u32>, Error> {
        let ids = self.ids(meshes)?;
        let report = self.scene.convex_hull(&ids)?;
        self.hull_added(report, normals)
    }

    /// Adds the convex hull of world `points` (flat `[x, y, z, ...]`) as a
    /// new closed, outward-wound mesh at identity, one undo step, as for
    /// `convex_hull`. Returns `[index, generation, inputPoints, vertices,
    /// triangles]`.
    pub fn convex_hull_points(&mut self, points: &[f64], normals: bool) -> Result<Vec<u32>, Error> {
        let (points, rest) = points.as_chunks::<3>();
        if !rest.is_empty() {
            return Err(Error::invalid("points length must be a multiple of 3"));
        }
        let report = self.scene.convex_hull_points(points)?;
        self.hull_added(report, normals)
    }
}

impl WasmScene {
    fn hull_added(&mut self, r: ConvexHullReport, normals: bool) -> Result<Vec<u32>, Error> {
        let mut out = self.added(r.added, normals)?;
        out.extend([len(r.input_points), len(r.vertices), len(r.triangles)]);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::codes;

    /// A unit cube's corners, and its 12 triangles (winding irrelevant here).
    const CUBE: [f64; 24] = [
        0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1.,
        1.,
    ];
    const CUBE_TRIS: [u32; 36] = [
        0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6,
        1, 3, 5, 3, 7, 5,
    ];

    #[test]
    fn hulls_of_meshes_and_points() {
        let mut s = WasmScene::new();
        let a = s.add_mesh(&CUBE, &CUBE_TRIS, None, 0, false).unwrap();
        let moved: Vec<f64> = CUBE
            .iter()
            .enumerate()
            .map(|(i, &c)| if i % 3 == 0 { c + 3.0 } else { c })
            .collect();
        let b = s.add_mesh(&moved, &CUBE_TRIS, None, 0, false).unwrap();
        s.take_changes();

        let one = s.convex_hull(&a, true).unwrap();
        assert_eq!(&one[2..], &[8, 8, 12]);
        assert_eq!(s.triangle_slots(one[0], one[1]).unwrap(), 12);
        let both = s.convex_hull(&[a[0], a[1], b[0], b[1]], false).unwrap();
        assert_eq!(&both[2..], &[16, 8, 12]);
        // One undo step removes the last hull (#269).
        assert!(s.undo().unwrap());
        assert!(s.mesh(both[0], both[1]).is_err());
        assert!(s.is_restorable(both[0], both[1]));
        assert!(s.redo().unwrap());
        let mesh = s.mesh(both[0], both[1]).unwrap();
        assert_eq!(mesh.triangle_count(), 12);

        let pts = s.convex_hull_points(&CUBE, false).unwrap();
        assert_eq!(&pts[2..], &[8, 8, 12]);
        assert_eq!(
            s.convex_hull_points(&CUBE[..23], false).unwrap_err().code,
            codes::INVALID_ARGUMENT
        );
        let flat = [0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.];
        let err = s.convex_hull_points(&flat, false).unwrap_err();
        assert_eq!(err.code, codes::INVALID_ARGUMENT);
        assert!(err.text.contains("coplanar"), "{}", err.text);
        assert_eq!(
            s.convex_hull(&[], false).unwrap_err().code,
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            s.convex_hull(&[99, 0], false).unwrap_err().code,
            codes::UNKNOWN_MESH
        );
    }
}
