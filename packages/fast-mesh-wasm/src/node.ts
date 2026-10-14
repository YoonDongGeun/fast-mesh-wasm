// Node.js and Bun entry. The .wasm file is read from disk and instantiated
// synchronously on import, so `init()` is only kept for API parity with the browser.
// This module has a side effect on purpose; package.json lists it under "sideEffects"
// so bundlers never drop it.
import { readFileSync } from "node:fs";
import { type InitInput, markInitialized } from "./api.js";
import { initSync } from "./wasm/fast_mesh_wasm.js";

initSync({ module: readFileSync(new URL("./wasm/fast_mesh_wasm_bg.wasm", import.meta.url)) });
markInitialized();

/**
 * Resolves immediately: in Node.js and Bun the module is ready as soon as it is imported.
 * Accepts the same argument as the browser `init()` so the same code runs everywhere;
 * the argument is ignored.
 */
export function init(_input?: InitInput | Promise<InitInput>): Promise<void> {
  return Promise.resolve();
}

export {
  type Bounds,
  computeBounds,
  computeVertexNormals,
  type Indices,
  type InitInput,
  type Positions,
  type Vec3,
} from "./api.js";
