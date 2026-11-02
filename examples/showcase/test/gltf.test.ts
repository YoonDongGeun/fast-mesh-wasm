// The showcase's glTF/GLB load path: extension dispatch, single- versus
// multi-mesh naming, and placement baking through to `MeshData` (its
// positions/indices feed `store.addMesh` in the loader Worker). (Needs the
// package `dist` built, like every showcase test that touches
// `fast-mesh-wasm`: CI builds before testing.)
import { describe, expect, it } from "vitest";
import { parseMeshFiles } from "../src/load/parse";

function f32(values: number[]): Uint8Array {
  const out = new Uint8Array(4 * values.length);
  const view = new DataView(out.buffer);
  values.forEach((v, i) => view.setFloat32(4 * i, v, true));
  return out;
}

function u16(values: number[]): Uint8Array {
  const out = new Uint8Array(2 * values.length);
  const view = new DataView(out.buffer);
  values.forEach((v, i) => view.setUint16(2 * i, v, true));
  return out;
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

/** One triangle (f32 positions, u16 indices) as the body of a glTF asset. */
function triangleBin(): Uint8Array {
  const out = new Uint8Array(36 + 6);
  out.set(f32([0, 0, 0, 1, 0, 0, 0, 1, 0]), 0);
  out.set(u16([0, 1, 2]), 36);
  return out;
}

function triangleDoc(bin: Uint8Array, nodes: object[]): object {
  return {
    asset: { version: "2.0" },
    scenes: [{ nodes: nodes.map((_, i) => i) }],
    nodes,
    meshes: [{ name: "tri", primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: "VEC3" },
      { bufferView: 1, componentType: 5123, count: 3, type: "SCALAR" },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36 },
      { buffer: 0, byteOffset: 36, byteLength: 6 },
    ],
    buffers: [{ byteLength: bin.length, uri: `data:application/octet-stream;base64,${b64(bin)}` }],
  };
}

function jsonBytes(doc: object): ArrayBuffer {
  return new TextEncoder().encode(JSON.stringify(doc)).buffer as ArrayBuffer;
}

/** A minimal binary `.glb`: a JSON chunk and the body chunk. */
function glbBytes(doc: object, bin: Uint8Array): ArrayBuffer {
  const json = new TextEncoder().encode(JSON.stringify(doc));
  const jsonPadded = (json.length + 3) & ~3;
  const binPadded = (bin.length + 3) & ~3;
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
  out.set(bin, at + 8);
  return out.buffer as ArrayBuffer;
}

describe("parseMeshFiles gltf", () => {
  it("parses a single-mesh .glb under its file name", () => {
    const bin = triangleBin();
    const [mesh] = parseMeshFiles("tri.glb", glbBytes(triangleDoc(bin, [{ mesh: 0 }]), bin));
    expect(mesh.name).toBe("tri.glb");
    expect(Array.from(mesh.data.positions)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    expect(Array.from(mesh.data.indices)).toEqual([0, 1, 2]);
  });

  it("parses a single-mesh .gltf under its file name", () => {
    const bin = triangleBin();
    const [mesh] = parseMeshFiles("tri.gltf", jsonBytes(triangleDoc(bin, [{ mesh: 0 }])));
    expect(mesh.name).toBe("tri.gltf");
    expect(Array.from(mesh.data.positions)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  });

  it("names each mesh of a multi-node asset after the file stem and node, placements baked", () => {
    const bin = triangleBin();
    const meshes = parseMeshFiles(
      "assembly.glb",
      glbBytes(
        triangleDoc(bin, [
          { name: "left", mesh: 0, translation: [10, 0, 0] },
          { name: "right", mesh: 0, translation: [-10, 0, 0] },
        ]),
        bin,
      ),
    );
    expect(meshes.map((m) => m.name)).toEqual(["assembly · left", "assembly · right"]);
    expect(Array.from(meshes[0].data.positions)).toEqual([10, 0, 0, 11, 0, 0, 10, 1, 0]);
    expect(Array.from(meshes[1].data.positions)).toEqual([-10, 0, 0, -9, 0, 0, -10, 1, 0]);
  });

  it("rejects unknown extensions and keeps the single-mesh entry single-mesh", () => {
    expect(() => parseMeshFiles("mesh.xyz", new ArrayBuffer(8))).toThrow(/\.xyz/);
    expect(() => parseMeshFiles("mesh.glb", new ArrayBuffer(0))).toThrow(/glTF/);
  });
});
