// Scene handles and the placement rotation helper. Free of WebAssembly, so
// the main-thread Worker client (`fast-mesh-wasm/worker`) uses them without
// loading the wasm glue.
import type { Mat3, Quat } from "./api.js";
import { FastMeshError } from "./core.js";
import type { Scene } from "./scene.js";

/**
 * A mesh in a {@link Scene}. Opaque: pass it to the scene's methods. The same
 * mesh always gives the same handle object, so handles compare with `===`.
 */
export class MeshHandle {
  declare private readonly brand: "MeshHandle";
}

/**
 * A field in a {@link Scene}. Opaque and deduplicated like {@link MeshHandle}.
 * The scene keeps every field handle it gave out for its own lifetime, even
 * after the field is removed, so a removal in a change record names the same
 * object you hold.
 */
export class FieldHandle {
  declare private readonly brand: "FieldHandle";
}

/**
 * A section in a {@link Scene}. Opaque and deduplicated like {@link MeshHandle},
 * and kept for the scene's lifetime like {@link FieldHandle}.
 */
export class SectionHandle {
  declare private readonly brand: "SectionHandle";
}

/** A quaternion as a row-major rotation matrix, normalised first, in f64. */
function quatToMat3(q: Quat): number[] {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!(n > 0) || !Number.isFinite(n)) {
    throw new FastMeshError("INVALID_ARGUMENT", "quaternion must be non-zero and finite");
  }
  const x = q[0] / n;
  const y = q[1] / n;
  const z = q[2] / n;
  const w = q[3] / n;
  return [
    1 - 2 * (y * y + z * z),
    2 * (x * y - z * w),
    2 * (x * z + y * w),
    2 * (x * y + z * w),
    1 - 2 * (x * x + z * z),
    2 * (y * z - x * w),
    2 * (x * z - y * w),
    2 * (y * z + x * w),
    1 - 2 * (x * x + y * y),
  ];
}

/**
 * A placement rotation as 9 numbers, row-major (a quaternion converted in
 * f64). Internal: shared with the Worker client, not part of the public API.
 */
export function rotationMatrix(rotation: Mat3 | Quat): number[] {
  if (rotation.length === 4) {
    return quatToMat3(rotation as Quat);
  }
  if (rotation.length === 9) {
    return Array.from(rotation);
  }
  throw new FastMeshError("INVALID_ARGUMENT", "rotation must be a 3×3 matrix (9 numbers) or a quaternion (4 numbers)");
}
