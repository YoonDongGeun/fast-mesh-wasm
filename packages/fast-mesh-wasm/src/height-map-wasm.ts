// `generateHeightfield`: the engine's relief generator. A standalone function
// for the reasons `generatePrimitive` is one (`primitives-wasm.ts`).
import { call, ensureInitialized } from "./core.js";
import { type CheckedHeightfield, checkHeightfield, encodeHeightfield, type Heightfield } from "./height-map.js";
import { decodePrimitiveMesh, type PrimitiveMesh } from "./primitives.js";
import { activeWasmScene } from "./scene.js";

/**
 * Generates the relief of a grid of heights: a closed, welded, outward-wound
 * solid from a flat base up to the heights, with a vertex per sample at
 * exactly its height. `x` runs along the rows, `y` against them (the first
 * row is at the top, as in an image) and `z` is the height; it is centred on
 * the origin in `x` and `y`, height 0 is `z = 0` and the base is `z = -base`.
 * Each cell is split along its flatter diagonal.
 *
 * `scene.addHeightfield` adds the same relief to a scene without its buffers
 * leaving the engine, and on a `WorkerScene`, `scene.generateHeightfield(field)`
 * runs this generator in its Worker, so the main thread needs no wasm.
 *
 * @example
 * ```ts
 * const { positions, indices } = generateHeightfield({ heights, width: 256, height: 256, pixelSize: 0.1, base: 2 });
 * ```
 * @throws `INVALID_ARGUMENT` for fewer than 2 samples on a side, a wrong
 * number of heights, or a height that is infinite or not above `-base`;
 * `OUT_OF_MEMORY` when the buffers cannot be allocated.
 */
export function generateHeightfield(field: Heightfield): PrimitiveMesh {
  ensureInitialized("generateHeightfield()");
  return generateCheckedHeightfield(checkHeightfield(field));
}

/** Internal (the Worker engine): {@link generateHeightfield} of a checked grid. */
export function generateCheckedHeightfield(f: CheckedHeightfield): PrimitiveMesh {
  const options = encodeHeightfield(f);
  return decodePrimitiveMesh(
    call(() => activeWasmScene().generateHeightfield(f.heights, f.width, f.height, options)),
  );
}
