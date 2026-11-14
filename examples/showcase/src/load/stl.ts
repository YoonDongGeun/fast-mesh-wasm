// STL, binary and ASCII. Delegates to `fast-mesh-wasm/io` with exact-duplicate
// corner welding (near-duplicates are left to the Weld repair).
import { importSTL } from "fast-mesh-wasm/io";
import { checkMesh, type MeshData } from "./mesh-data";

/**
 * Parses an STL file. Binary when its size matches the triangle count in its
 * header (even if the header text starts with "solid", as some exporters
 * write), else ASCII when it starts with "solid".
 *
 * @throws when the file is neither or holds no faces.
 */
export function parseStl(buffer: ArrayBuffer): MeshData {
  const mesh = importSTL(buffer, { deduplicateVertices: true });
  return checkMesh({ positions: mesh.positions, indices: mesh.indices }, "STL");
}
