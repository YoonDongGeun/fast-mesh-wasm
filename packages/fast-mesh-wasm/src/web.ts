// Browser and bundler entry (Vite, webpack, Next.js client, Deno, ...).
// The glue locates the .wasm with `new URL("fast_mesh_wasm_bg.wasm", import.meta.url)`,
// which bundlers recognize and copy as an asset.
import { type InitInput, markInitialized } from "./api.js";
import wasmInit from "./wasm/fast_mesh_wasm.js";

let ready: Promise<void> | undefined;

/**
 * Loads the WebAssembly module. Await this once before calling any other function.
 * Calling it again returns the same promise.
 *
 * @param input - Optional custom source for the `.wasm` file (URL, Response, bytes or module).
 *   Leave empty to load the file shipped with this package.
 */
export function init(input?: InitInput | Promise<InitInput>): Promise<void> {
  ready ??= wasmInit(input === undefined ? undefined : { module_or_path: input }).then(
    markInitialized,
  );
  return ready;
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
