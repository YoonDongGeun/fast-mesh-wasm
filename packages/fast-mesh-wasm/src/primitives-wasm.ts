// `generatePrimitive`: the engine's primitive generators. A standalone
// function, not a Scene op: a primitive is a pure function of its
// parameters (positions and indices, no scene, no history), so it runs
// wherever the module is initialised: the main thread after `init()`, any
// Worker of the caller's, or the scene's own Worker through
// `WorkerScene.generatePrimitive`. It reaches the binding through the
// `WasmScene` class scenes are made from (`activeWasmScene`), which in the
// scene's Worker is the threaded glue's when that runs.
import { call, ensureInitialized, FastMeshError } from "./core.js";
import {
  decodePrimitiveMesh,
  type EncodedPrimitive,
  encodePrimitive,
  type PrimitiveMesh,
  type PrimitiveShape,
} from "./primitives.js";
import { activeWasmScene } from "./scene.js";

/**
 * Generates a primitive shape: a box, cylinder, cone, UV sphere,
 * icosphere, torus, arrow, n-gon prism or plane, centred on the origin,
 * welded, closed (except the plane) and wound outward; see
 * {@link PrimitiveShape} for each shape's options and defaults.
 *
 * Add it to a scene with `scene.addMesh(mesh.positions, mesh.indices)`
 * (translate the positions or set a placement to move it), or draw it as a
 * `BufferGeometry`. On a `WorkerScene`, `scene.generatePrimitive(shape)`
 * runs the same generator in its Worker, so the main thread needs no wasm.
 *
 * @example
 * ```ts
 * const { positions, indices } = generatePrimitive({ kind: "cylinder", radius: 2, height: 5, segments: 64 });
 * ```
 * @throws `INVALID_ARGUMENT` for an unknown kind or a parameter out of
 * range; `OUT_OF_MEMORY` when the buffers cannot be allocated.
 */
export function generatePrimitive(shape: PrimitiveShape): PrimitiveMesh {
  ensureInitialized("generatePrimitive()");
  return generateEncodedPrimitive(encodePrimitive(shape));
}

/** Internal (the Worker engine): {@link generatePrimitive} of an encoded shape. */
export function generateEncodedPrimitive(shape: EncodedPrimitive): PrimitiveMesh {
  if (!Array.isArray(shape?.params)) {
    throw new FastMeshError("INVALID_ARGUMENT", "an encoded primitive needs a kind and parameters");
  }
  const params = Float64Array.from(shape.params);
  return decodePrimitiveMesh(call(() => activeWasmScene().generatePrimitive(shape.kind, params)));
}
