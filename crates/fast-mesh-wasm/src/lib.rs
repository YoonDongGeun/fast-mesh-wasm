//! JavaScript bindings for `fast-mesh`.
//!
//! These functions are the raw WebAssembly exports. The npm package wraps them in a
//! TypeScript layer (`packages/fast-mesh-wasm/src/api.ts`) that converts inputs,
//! shapes results into plain objects and documents the public API.

use wasm_bindgen::prelude::*;

fn to_js_error(err: fast_mesh::MeshError) -> JsError {
    JsError::new(&err.to_string())
}

/// Smooth, area-weighted per-vertex normals; same length as `positions`.
#[wasm_bindgen(js_name = computeVertexNormals)]
pub fn compute_vertex_normals(positions: &[f32], indices: &[u32]) -> Result<Vec<f32>, JsError> {
    fast_mesh::compute_vertex_normals(positions, indices).map_err(to_js_error)
}

/// Axis-aligned bounding box as six floats: `[minX, minY, minZ, maxX, maxY, maxZ]`.
///
/// Returned as plain numbers so JavaScript never holds WebAssembly memory for it.
#[wasm_bindgen(js_name = computeBounds)]
pub fn compute_bounds(positions: &[f32]) -> Result<Vec<f32>, JsError> {
    let aabb = fast_mesh::compute_bounds(positions).map_err(to_js_error)?;
    Ok([aabb.min, aabb.max].concat())
}
