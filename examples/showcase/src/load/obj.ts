// Wavefront OBJ: delegates to `fast-mesh-wasm/io` (merged groups,
// position-only vertex indexing so showcase `MeshData` keeps original order).
import { importOBJ } from "fast-mesh-wasm/io";
import { checkMesh, type MeshData } from "./mesh-data";

/** @throws on a malformed vertex or face, an index out of range, or no faces. */
export function parseObj(buffer: ArrayBuffer): MeshData {
  const [mesh] = importOBJ(buffer, { mergeGroups: true, positionsOnly: true });
  return checkMesh({ positions: mesh.positions, indices: mesh.indices }, "OBJ");
}
