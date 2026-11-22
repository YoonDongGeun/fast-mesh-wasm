// PLY, ASCII and binary (little- or big-endian): delegates to `fast-mesh-wasm/io`
// while validating vertex/face counts and preserving existing error contexts.
import { importPLY } from "fast-mesh-wasm/io";
import { checkMesh, type MeshData } from "./mesh-data";

/** @throws on a malformed file, a face index out of range, or no faces. */
export function parsePly(buffer: ArrayBuffer): MeshData {
  try {
    const mesh = importPLY(buffer);
    return checkMesh({ positions: mesh.positions, indices: mesh.indices }, "PLY");
  } catch (error) {
    const head = new TextDecoder().decode(
      new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 256)),
    );
    if (/(?:^|[\r\n])format\s+binary_big_endian\b/.test(head)) {
      const msg = error instanceof Error ? error.message : String(error);
      throw new Error(`PLY (big-endian): ${msg}`);
    }
    throw error;
  }
}
