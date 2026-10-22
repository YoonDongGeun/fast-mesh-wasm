// STL, binary and ASCII. STL is triangle soup: corners with bit-identical
// positions are merged into one vertex (near-duplicates are left to the
// Weld repair).
import { checkMesh, type MeshData, weldCorners } from "./mesh-data";

/**
 * Parses an STL file. Binary when its size matches the triangle count in its
 * header (even if the header text starts with "solid", as some exporters
 * write), else ASCII when it starts with "solid".
 *
 * @throws when the file is neither.
 */
export function parseStl(buffer: ArrayBuffer): MeshData {
  if (buffer.byteLength >= 84) {
    const count = new DataView(buffer).getUint32(80, true);
    if (count > 0 && buffer.byteLength === 84 + 50 * count) {
      return checkMesh(parseBinary(buffer, count), "STL");
    }
  }
  const head = new TextDecoder().decode(new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 256)));
  if (/^\s*solid/.test(head)) {
    return checkMesh(parseAscii(new TextDecoder().decode(buffer)), "STL");
  }
  if (buffer.byteLength >= 84) {
    const count = new DataView(buffer).getUint32(80, true);
    if (count > 0) {
      throw new Error(
        `binary STL: the header says ${count} triangles (${84 + 50 * count} bytes), but the file has ${buffer.byteLength} bytes`,
      );
    }
  }
  throw new Error("not an STL file (neither binary nor ASCII STL)");
}

function parseBinary(buffer: ArrayBuffer, count: number): MeshData {
  const view = new DataView(buffer);
  const corners = new Float64Array(9 * count);
  for (let t = 0; t < count; t++) {
    // 50 bytes per triangle: normal (12), three corners (36), attribute (2).
    const at = 84 + 50 * t + 12;
    for (let k = 0; k < 9; k++) {
      corners[9 * t + k] = view.getFloat32(at + 4 * k, true);
    }
  }
  return weldCorners(corners);
}

function parseAscii(src: string): MeshData {
  const values: number[] = [];
  const vertex = /\bvertex\s+(\S+)\s+(\S+)\s+(\S+)/g;
  for (let m = vertex.exec(src); m !== null; m = vertex.exec(src)) {
    values.push(Number(m[1]), Number(m[2]), Number(m[3]));
  }
  if (values.length === 0 || values.length % 9 !== 0) {
    throw new Error(`ASCII STL: ${values.length / 3} vertices, not 3 per facet`);
  }
  return weldCorners(Float64Array.from(values));
}
