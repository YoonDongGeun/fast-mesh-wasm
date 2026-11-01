// The export path: `exportMeshBytes` output re-parses with the showcase's
// own file parsers (STL, OBJ, PLY), and the download names carry the right
// extension. (Needs the package `dist` built, like every showcase test that
// touches `fast-mesh-wasm`: CI builds before testing.)
import { describe, expect, it } from "vitest";
import type { SceneLike } from "fast-mesh-wasm";
import { MeshHandle } from "fast-mesh-wasm";
import { exportFileName, exportMeshBytes, type MeshExportFormat } from "../src/load/export";
import type { MeshData } from "../src/load/mesh-data";
import { parseObj } from "../src/load/obj";
import { parsePly } from "../src/load/ply";
import { parseStl } from "../src/load/stl";

const QUAD = Float32Array.from([0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 9, 9, 9]);
const QUAD_TRIS = Uint32Array.from([0, 1, 2, 0, 2, 3, 0, 0, 0]);

/** One quad in memory, as the store's scene would serve it. */
class FakeScene implements SceneLike {
  readonly mesh = new MeshHandle();
  views() {
    return {
      positions: () => QUAD,
      normals: () => null,
      colors: () => null,
      indices: () => QUAD_TRIS,
    };
  }
  matrix4() {
    return Float64Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  }
  placement() {
    return { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0] as [number, number, number] };
  }
  section(): never {
    throw new Error("no sections");
  }
  sectionPlane(): never {
    throw new Error("no sections");
  }
  onChanges() {
    return () => {};
  }
  commitFrame() {
    return [];
  }
  drag(): never {
    throw new Error("no drags");
  }
}

/** Triangles as sorted corner strings, so vertex order does not matter. */
function corners(m: MeshData): Set<string> {
  const out = new Set<string>();
  for (let t = 0; t < m.indices.length; t += 3) {
    const tri = [0, 1, 2]
      .map((k) => {
        const v = m.indices[t + k];
        return `${m.positions[3 * v]},${m.positions[3 * v + 1]},${m.positions[3 * v + 2]}`;
      })
      .sort()
      .join("|");
    out.add(tri);
  }
  return out;
}

const EXPECTED = new Set(["0,0,0|2,0,0|2,2,0", "0,0,0|0,2,0|2,2,0"]);

/** A standalone `ArrayBuffer` copy, as the parsers take. */
function toBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer as ArrayBuffer;
}

describe("mesh export", () => {
  it.each([["stl"], ["obj"], ["ply"]] as [MeshExportFormat][])("exports %s that re-parses", (format) => {
    const scene = new FakeScene();
    const bytes = exportMeshBytes(scene, scene.mesh, format, "quad");
    const buffer = toBuffer(bytes);
    const parsed = format === "stl" ? parseStl(buffer) : format === "obj" ? parseObj(buffer) : parsePly(buffer);
    expect(corners(parsed)).toEqual(EXPECTED);
  });

  it("names downloads by format", () => {
    expect(exportFileName("my mesh", "stl")).toBe("my mesh.stl");
    expect(exportFileName("my mesh", "glb")).toBe("my mesh.glb");
    expect(exportFileName("my mesh", "gltf")).toBe("my mesh.gltf");
  });
});
