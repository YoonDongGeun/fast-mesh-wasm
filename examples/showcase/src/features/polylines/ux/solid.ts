// A mesh a polyline tool made, as the engine stores one (#268).
import type { PrimitiveMesh } from "fast-mesh-wasm";
import { bounds, type Vec3 } from "../../../io/mesh-data";

/** A mesh to add: positions about `center` (world millimetres), and triangles wound outward. */
export interface SolidMesh extends PrimitiveMesh {
  center: Vec3;
}

/** World positions and triangles about the centre of their box. */
export function centredSolid(world: Float64Array, indices: Uint32Array): SolidMesh {
  const { center } = bounds(world);
  const positions = new Float32Array(world.length);
  for (let i = 0; i < world.length; i++) {
    positions[i] = world[i] - center[i % 3];
  }
  return { positions, indices, center };
}
