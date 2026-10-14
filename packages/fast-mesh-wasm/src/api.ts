// Public API shared by the browser and Node entries. Wraps the raw wasm-bindgen
// exports: converts inputs, shapes results into plain values and guards against
// calls made before the WebAssembly module is ready.
import * as wasm from "./wasm/fast_mesh_wasm.js";

export type { InitInput } from "./wasm/fast_mesh_wasm.js";

/** A 3D vector `[x, y, z]`. */
export type Vec3 = [x: number, y: number, z: number];

/**
 * Flat vertex positions `[x0, y0, z0, x1, y1, z1, ...]`.
 * A `Float32Array` is used as is; any other array is converted once.
 */
export type Positions = Float32Array | ArrayLike<number>;

/**
 * Flat triangle indices `[a0, b0, c0, a1, ...]`.
 * A `Uint32Array` is used as is; `Uint16Array` (common in three.js) and plain arrays are converted once.
 */
export type Indices = Uint32Array | Uint16Array | ArrayLike<number>;

/** Axis-aligned bounding box. */
export interface Bounds {
  /** Smallest coordinate on each axis. */
  min: Vec3;
  /** Largest coordinate on each axis. */
  max: Vec3;
  /** Box center. */
  center: Vec3;
  /** Box extent: `[width, height, depth]`. */
  size: Vec3;
}

let initialized = false;

/** Called by the entry point once the WebAssembly module is instantiated. */
export function markInitialized(): void {
  initialized = true;
}

function ensureInitialized(fn: string): void {
  if (!initialized) {
    throw new Error(`fast-mesh-wasm is not initialized: call \`await init()\` before ${fn}().`);
  }
}

function toFloat32(values: Positions): Float32Array {
  return values instanceof Float32Array ? values : Float32Array.from(values);
}

function toUint32(values: Indices): Uint32Array {
  return values instanceof Uint32Array ? values : Uint32Array.from(values);
}

/**
 * Computes smooth, area-weighted per-vertex normals.
 *
 * @param positions - Vertex positions. Length must be a multiple of 3.
 * @param indices - Triangle indices. Length must be a multiple of 3.
 * @returns Normals with the same length as `positions`. Vertices not used by any triangle get `[0, 0, 0]`.
 * @throws If a length is not a multiple of 3 or an index is out of range.
 */
export function computeVertexNormals(positions: Positions, indices: Indices): Float32Array {
  ensureInitialized("computeVertexNormals");
  return wasm.computeVertexNormals(toFloat32(positions), toUint32(indices));
}

/**
 * Computes the axis-aligned bounding box of a mesh.
 *
 * @param positions - Vertex positions. Length must be a non-zero multiple of 3.
 * @throws If `positions` is empty or its length is not a multiple of 3.
 */
export function computeBounds(positions: Positions): Bounds {
  ensureInitialized("computeBounds");
  const b = wasm.computeBounds(toFloat32(positions));
  const min: Vec3 = [b[0], b[1], b[2]];
  const max: Vec3 = [b[3], b[4], b[5]];
  return {
    min,
    max,
    center: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2],
    size: [max[0] - min[0], max[1] - min[1], max[2] - min[2]],
  };
}
