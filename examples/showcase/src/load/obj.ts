// Wavefront OBJ: `v` positions and `f` faces (any polygon, fan-triangulated;
// `i`, `i/t`, `i//n` and `i/t/n` corners; negative indices count back from
// the latest vertex). Everything else (normals, texture coordinates, groups,
// materials) is ignored.
import { checkMesh, type MeshData } from "./mesh-data";

/** @throws on a malformed vertex or face, an index out of range, or no faces. */
export function parseObj(buffer: ArrayBuffer): MeshData {
  const src = new TextDecoder().decode(buffer);
  const positions: number[] = [];
  const indices: number[] = [];
  const corner: number[] = [];
  let line = 0;
  for (let start = 0; start < src.length; ) {
    let end = src.indexOf("\n", start);
    if (end === -1) {
      end = src.length;
    }
    line++;
    const text = src.slice(start, end).trim();
    start = end + 1;
    if (text.startsWith("v ") || text.startsWith("v\t")) {
      const p = text.split(/\s+/);
      const x = Number(p[1]);
      const y = Number(p[2]);
      const z = Number(p[3]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        throw new Error(`OBJ line ${line}: bad vertex "${text}"`);
      }
      positions.push(x, y, z);
    } else if (text.startsWith("f ") || text.startsWith("f\t")) {
      const p = text.split(/\s+/);
      corner.length = 0;
      const vertices = positions.length / 3;
      for (let k = 1; k < p.length; k++) {
        const raw = Number.parseInt(p[k], 10);
        if (!Number.isInteger(raw) || raw === 0) {
          throw new Error(`OBJ line ${line}: bad face index "${p[k]}"`);
        }
        corner.push(raw < 0 ? vertices + raw : raw - 1);
      }
      if (corner.length < 3) {
        throw new Error(`OBJ line ${line}: a face needs 3 corners`);
      }
      for (let k = 1; k + 1 < corner.length; k++) {
        indices.push(corner[0], corner[k], corner[k + 1]);
      }
    }
  }
  if (indices.length === 0) {
    throw new Error("OBJ: no faces");
  }
  for (const i of indices) {
    if (i < 0) {
      throw new Error(`OBJ: a negative face index points before the first vertex`);
    }
  }
  return checkMesh({ positions: Float64Array.from(positions), indices: Uint32Array.from(indices) }, "OBJ");
}
