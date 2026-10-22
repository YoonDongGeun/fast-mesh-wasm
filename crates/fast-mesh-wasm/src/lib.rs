//! JavaScript bindings for `fast-mesh`.
//!
//! These functions are the raw WebAssembly exports. The npm package wraps them in a
//! TypeScript layer (`packages/fast-mesh-wasm/src/api.ts`) that converts inputs,
//! shapes results into plain objects and documents the public API.

use wasm_bindgen::prelude::*;

mod edit;
mod error;
#[cfg(all(feature = "threads", target_arch = "wasm32"))]
mod pool;
mod render;
mod scene;

/// `initThreadPool(n)` (threaded artifact only): starts rayon's pool on `n`
/// Web Workers that share this module and its memory. Await it once, right
/// after `init()`, before any other call.
#[cfg(all(feature = "threads", target_arch = "wasm32"))]
pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg(all(feature = "threads", target_arch = "wasm32"))]
pub use pool::watch_pool_crashes;

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

/// Panics, so the TypeScript tests can check that a trap poisons the
/// instance. Only in builds with the `test-panic` feature
/// (`FAST_MESH_TEST_BUILD=1`), never in releases.
#[cfg(feature = "test-panic")]
#[wasm_bindgen(js_name = __test_panic)]
pub fn test_panic() {
    panic!("test panic");
}

/// Panics on a thread of rayon's pool, inside a parallel loop, so the browser
/// check can confirm the Worker scene poisons (`watchPoolCrashes`). Threaded
/// test builds only (`node scripts/build-wasm.mjs --threads --test`).
#[cfg(all(feature = "test-panic", feature = "threads", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = __test_panic_pool)]
pub fn test_panic_pool() {
    use rayon::prelude::*;
    (0..1024u32).into_par_iter().for_each(|i| {
        if i == 1000 {
            panic!("test panic on a pool thread");
        }
    });
}

/// Traps on a thread of rayon's pool (`unreachable`, no panic), inside a
/// parallel loop: a crash the panic hook never sees. Threaded test builds only.
#[cfg(all(feature = "test-panic", feature = "threads", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = __test_trap_pool)]
pub fn test_trap_pool() {
    use rayon::prelude::*;
    (0..1024u32).into_par_iter().for_each(|i| {
        if i == 1000 {
            core::arch::wasm32::unreachable();
        }
    });
}
