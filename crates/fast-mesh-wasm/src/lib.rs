//! JavaScript bindings for `fast-mesh`.
//!
//! Doc comments on exported items become JSDoc in the generated `.d.ts`,
//! so they are what users see in editor autocomplete. Keep them user-facing.

use wasm_bindgen::prelude::*;

fn to_js_error(err: fast_mesh::MeshError) -> JsError {
    JsError::new(&err.to_string())
}

/// Computes smooth, area-weighted per-vertex normals.
///
/// @param positions - Vertex positions `[x0, y0, z0, x1, ...]`. Length must be a multiple of 3.
/// @param indices - Triangle indices `[a0, b0, c0, a1, ...]`. Length must be a multiple of 3.
/// @returns Normals with the same length as `positions`. Unused vertices get `[0, 0, 0]`.
/// @throws If a buffer length is not a multiple of 3 or an index is out of range.
#[wasm_bindgen(js_name = computeVertexNormals)]
pub fn compute_vertex_normals(positions: &[f32], indices: &[u32]) -> Result<Vec<f32>, JsError> {
    fast_mesh::compute_vertex_normals(positions, indices).map_err(to_js_error)
}

/// Computes the axis-aligned bounding box of a mesh.
///
/// The returned object owns WebAssembly memory. Call `free()` when done,
/// or declare it with `using` so it is released automatically.
///
/// @param positions - Vertex positions `[x0, y0, z0, x1, ...]`. Length must be a multiple of 3.
/// @throws If `positions` is empty or its length is not a multiple of 3.
#[wasm_bindgen(js_name = computeBounds)]
pub fn compute_bounds(positions: &[f32]) -> Result<Bounds, JsError> {
    fast_mesh::compute_bounds(positions)
        .map(|aabb| Bounds { aabb })
        .map_err(to_js_error)
}

/// Axis-aligned bounding box returned by {@link computeBounds}.
#[wasm_bindgen]
pub struct Bounds {
    aabb: fast_mesh::Aabb,
}

#[wasm_bindgen]
impl Bounds {
    /// Smallest `[x, y, z]` over all vertices.
    #[wasm_bindgen(getter)]
    pub fn min(&self) -> Vec<f32> {
        self.aabb.min.to_vec()
    }

    /// Largest `[x, y, z]` over all vertices.
    #[wasm_bindgen(getter)]
    pub fn max(&self) -> Vec<f32> {
        self.aabb.max.to_vec()
    }

    /// Box center `[x, y, z]`.
    #[wasm_bindgen(getter)]
    pub fn center(&self) -> Vec<f32> {
        (0..3)
            .map(|i| (self.aabb.min[i] + self.aabb.max[i]) * 0.5)
            .collect()
    }

    /// Box extent `[width, height, depth]`.
    #[wasm_bindgen(getter)]
    pub fn size(&self) -> Vec<f32> {
        (0..3)
            .map(|i| self.aabb.max[i] - self.aabb.min[i])
            .collect()
    }
}
