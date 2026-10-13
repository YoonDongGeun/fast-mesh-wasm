// Node.js and Bun entry. The .wasm file is read from disk and instantiated
// synchronously on import, so `init()` is only kept for API parity with the browser.
import { readFileSync } from "node:fs";
import { initSync } from "./wasm/fast_mesh_wasm.js";

initSync({ module: readFileSync(new URL("./wasm/fast_mesh_wasm_bg.wasm", import.meta.url)) });

/**
 * Resolves immediately: in Node.js and Bun the module is ready as soon as it is imported.
 * Exists so the same code runs in the browser, where `await init()` is required.
 */
export function init(): Promise<void> {
  return Promise.resolve();
}

export { Bounds, computeBounds, computeVertexNormals } from "./wasm/fast_mesh_wasm.js";
