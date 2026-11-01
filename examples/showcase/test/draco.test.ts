// The showcase's Draco load path: `.drc` files and `KHR_draco_mesh_compression`
// glTF assets decode through the draco3d decoder in the loader Worker, onto
// the same `MeshData` (positions/indices) every other parser produces. The
// `.drc` bytes below are a real 75-byte encoder-made quad (two triangles on
// the XY plane, corners (0,0,0), (2,0,0), (2,2,0), (0,2,0)), regenerated with
// draco3d's encoder if they ever need replacing. (Needs the package `dist`
// built, like every showcase test that touches `fast-mesh-wasm`: CI builds
// before testing.)
import { describe, expect, it } from "vitest";
import { MESH_EXTENSIONS, parseMeshFile, parseMeshFiles, parseMeshFilesAsync } from "../src/load/parse";

// A quad: corners (0,0,0), (2,0,0), (2,2,0), (0,2,0), faces (0,1,2) and (0,2,3).
const QUAD_DRC_B64 =
  "RFJBQ08CAgEBAAAABAIAAgAAAR//AREB/wAAAQAJAwAAAgEBAQADAwEwARADACiCmAAAAAAA/z8AAAAAAAAAAAAAAAAAAAAAAEAO";

function drcBytes(): ArrayBuffer {
  const g = globalThis as { Buffer?: { from(s: string, e: string): Uint8Array } };
  const bytes =
    g.Buffer !== undefined ? g.Buffer.from(QUAD_DRC_B64, "base64") : Uint8Array.from(atob(QUAD_DRC_B64), (c) => c.charCodeAt(0));
  // An exact-length copy: a Node Buffer may view a shared pool (and even
  // `slice` stays a view), while the decoder reads the whole backing store.
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return copy.buffer as ArrayBuffer;
}

function b64(bytes: Uint8Array): string {
  const g = globalThis as { Buffer?: { from(b: Uint8Array): { toString(e: string): string } } };
  if (g.Buffer !== undefined) {
    return g.Buffer.from(bytes).toString("base64");
  }
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(text);
}

/** Decoded triangles as rounded corner strings, so quantization noise does not matter. */
function triangleSet(mesh: { positions: ArrayLike<number>; indices: ArrayLike<number> }): Set<string> {
  const tris = new Set<string>();
  for (let t = 0; t < mesh.indices.length; t += 3) {
    const corners = [0, 1, 2].map((k) => {
      const v = mesh.indices[t + k];
      return [0, 1, 2].map((c) => mesh.positions[3 * v + c].toFixed(2)).join(",");
    });
    corners.sort();
    tris.add(corners.join("|"));
  }
  return tris;
}

const EXPECTED = new Set([
  "0.00,0.00,0.00|2.00,0.00,0.00|2.00,2.00,0.00",
  "0.00,0.00,0.00|0.00,2.00,0.00|2.00,2.00,0.00",
]);

/** A glTF asset whose primitive is the Draco quad, in the GLB body chunk. */
function dracoGlb(): ArrayBuffer {
  const drc = new Uint8Array(drcBytes());
  const doc = {
    asset: { version: "2.0" },
    scenes: [{ nodes: [0] }],
    nodes: [{ name: "zipped", mesh: 0 }],
    meshes: [
      {
        primitives: [
          {
            attributes: { POSITION: 0 },
            extensions: { KHR_draco_mesh_compression: { bufferView: 0, attributes: { POSITION: 0 } } },
          },
        ],
      },
    ],
    accessors: [{ bufferView: 0, componentType: 5126, count: 0, type: "VEC3" }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: drc.length }],
    buffers: [{ byteLength: drc.length }],
  };
  const json = new TextEncoder().encode(JSON.stringify(doc));
  const jsonPadded = (json.length + 3) & ~3;
  const binPadded = (drc.length + 3) & ~3;
  const out = new Uint8Array(12 + 8 + jsonPadded + 8 + binPadded);
  const view = new DataView(out.buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, out.length, true);
  view.setUint32(12, jsonPadded, true);
  view.setUint32(16, 0x4e4f534a, true);
  out.set(json, 20);
  out.fill(0x20, 20 + json.length, 20 + jsonPadded);
  const at = 20 + jsonPadded;
  view.setUint32(at, binPadded, true);
  view.setUint32(at + 4, 0x004e4942, true);
  out.set(drc, at + 8);
  return out.buffer as ArrayBuffer;
}

function dracoGltf(): ArrayBuffer {
  const drc = new Uint8Array(drcBytes());
  const doc = {
    asset: { version: "2.0" },
    scenes: [{ nodes: [0] }],
    nodes: [{ name: "zipped", mesh: 0 }],
    meshes: [
      {
        primitives: [
          {
            attributes: { POSITION: 0 },
            extensions: { KHR_draco_mesh_compression: { bufferView: 0, attributes: { POSITION: 0 } } },
          },
        ],
      },
    ],
    accessors: [{ bufferView: 0, componentType: 5126, count: 0, type: "VEC3" }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: drc.length }],
    buffers: [{ byteLength: drc.length, uri: `data:application/octet-stream;base64,${b64(drc)}` }],
  };
  return new TextEncoder().encode(JSON.stringify(doc)).buffer as ArrayBuffer;
}

describe("draco load path", () => {
  it("lists .drc as a mesh extension, decoded asynchronously", () => {
    expect(MESH_EXTENSIONS).toContain(".drc");
    expect(() => parseMeshFile("quad.drc", new ArrayBuffer(8))).toThrow(/parseMeshFilesAsync/);
  });

  it("decodes a .drc file to the quad, under its file name", async () => {
    const [mesh] = await parseMeshFilesAsync("quad.DRC", drcBytes());
    expect(mesh.name).toBe("quad.DRC");
    expect(mesh.data.indices.length).toBe(6);
    expect(mesh.data.positions.length).toBe(4 * 3);
    expect(triangleSet(mesh.data)).toEqual(EXPECTED);
  });

  it("decodes a Draco .glb primitive, under its file name", async () => {
    const [mesh] = await parseMeshFilesAsync("zipped.glb", dracoGlb());
    expect(mesh.name).toBe("zipped.glb");
    expect(triangleSet(mesh.data)).toEqual(EXPECTED);
  });

  it("decodes a Draco .gltf primitive, under its file name", async () => {
    const [mesh] = await parseMeshFilesAsync("zipped.gltf", dracoGltf());
    expect(mesh.name).toBe("zipped.gltf");
    expect(triangleSet(mesh.data)).toEqual(EXPECTED);
  });

  it("keeps the sync entries Draco-free", () => {
    expect(() => parseMeshFiles("zipped.glb", dracoGlb())).toThrow(/KHR_draco_mesh_compression/);
    expect(() => parseMeshFile("quad.drc", drcBytes())).toThrow(/parseMeshFilesAsync/);
  });

  it("rejects what is not Draco", async () => {
    await expect(parseMeshFilesAsync("quad.drc", new TextEncoder().encode("hello").buffer as ArrayBuffer)).rejects.toThrow(
      /Draco/,
    );
  });
});
