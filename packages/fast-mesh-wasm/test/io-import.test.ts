// The glTF importer over a hand-written fake SceneLike: GLB and JSON
// round-trips of the exporters, node placement baking, vertex colors, fan
// expansion, name deduplication and the documented rejections.
import { describe, expect, it } from "vitest";
import { MeshHandle } from "../src/handles.js";
import { exportGLTF, importGLTF } from "../src/io/index.js";
import type { SceneLike } from "../src/scene-like.js";

/** Two triangles on the XY plane facing +Z, with a dead slot of each kind. */
const QUAD = Float32Array.from([0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 9, 9, 9]);
const QUAD_TRIS = Uint32Array.from([0, 1, 2, 0, 2, 3, 0, 0, 0]);
const QUAD_NORMALS = Float32Array.from([
  0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
]);
const QUAD_COLORS = Uint8Array.from([
  255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255,
]);
const IDENTITY = Float64Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

/** Minimal in-memory SceneLike: positions, indices, optional normals/colors, fixed matrix. */
class FakeScene implements SceneLike {
  readonly mesh = new MeshHandle();
  constructor(
    readonly positions: Float32Array = QUAD,
    readonly indices: Uint32Array = QUAD_TRIS,
    readonly normals: Float32Array | null = QUAD_NORMALS,
    readonly colors: Uint8Array | null = QUAD_COLORS,
    readonly matrix: Float64Array = IDENTITY,
  ) {}
  views() {
    return {
      positions: () => this.positions,
      normals: () => this.normals,
      colors: () => this.colors,
      indices: () => this.indices,
    };
  }
  matrix4() {
    return this.matrix;
  }
  placement() {
    return { rotation: [1, 0, 0, 0, 1, 0, 0, 1, 0], translation: [0, 0, 0] as [number, number, number] };
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

/** Live triangles as sorted corner strings, so tests ignore vertex order. */
function triangleSet(positions: ArrayLike<number>, indices: ArrayLike<number>): Set<string> {
  const tris = new Set<string>();
  for (let t = 0; t < indices.length; t += 3) {
    const corners = [0, 1, 2].map((k) => {
      const v = indices[t + k];
      return `${positions[3 * v]},${positions[3 * v + 1]},${positions[3 * v + 2]}`;
    });
    corners.sort();
    tris.add(corners.join("|"));
  }
  return tris;
}

const EXPECTED = new Set(["0,0,0|2,0,0|2,2,0", "0,0,0|0,2,0|2,2,0"]);

function toBase64(bytes: Uint8Array): string {
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

function f32(values: number[]): Uint8Array {
  const out = new Uint8Array(4 * values.length);
  const view = new DataView(out.buffer);
  values.forEach((v, i) => view.setFloat32(4 * i, v, true));
  return out;
}

/** A `.gltf` JSON asset whose buffer is embedded as a data URI. */
function jsonAsset(doc: object): Uint8Array {
  return new TextEncoder().encode(JSON.stringify(doc));
}

function uriOf(bin: Uint8Array): string {
  return `data:application/octet-stream;base64,${toBase64(bin)}`;
}

/** A binary `.glb` of the JSON document and its body chunk. */
function glb(doc: object, bin: Uint8Array): Uint8Array {
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
  return out;
}

describe("io import gltf", () => {
  it("glb_round_trip_with_colors", () => {
    const scene = new FakeScene();
    const [part] = importGLTF(exportGLTF(scene, scene.mesh));
    expect(part.name).toBe("mesh");
    expect(triangleSet(part.positions, part.indices)).toEqual(EXPECTED);
    // The dead vertex slot is gone; its color went with it.
    expect(part.positions.length).toBe(4 * 3);
    expect(Array.from(part.colors ?? [])).toEqual(Array.from(QUAD_COLORS.subarray(0, 16)));
    expect(Array.from(part.placement)).toEqual(Array.from(IDENTITY));
  });

  it("gltf_json_round_trip_without_normals_or_colors", () => {
    const scene = new FakeScene(QUAD, QUAD_TRIS, null, null);
    const parts = importGLTF(exportGLTF(scene, scene.mesh, { binary: false }));
    expect(parts.length).toBe(1);
    expect(triangleSet(parts[0].positions, parts[0].indices)).toEqual(EXPECTED);
    expect(parts[0].colors).toBeNull();
  });

  it("bakes_node_placements", () => {
    const bin = new Uint8Array(36 + 3);
    bin.set(f32([0, 0, 0, 1, 0, 0, 0, 1, 0]), 0);
    bin.set([0, 1, 2], 36);
    const doc = {
      asset: { version: "2.0" },
      scenes: [{ nodes: [0, 1] }],
      nodes: [
        { name: "moved", mesh: 0, translation: [10, 0, 0] },
        { name: "spun", mesh: 0, rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2] },
      ],
      meshes: [{ name: "tri", primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: "VEC3" },
        { bufferView: 1, componentType: 5121, count: 3, type: "SCALAR" },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 36 },
        { buffer: 0, byteOffset: 36, byteLength: 3 },
      ],
      buffers: [{ byteLength: bin.length, uri: uriOf(bin) }],
    };
    const parts = importGLTF(jsonAsset(doc));
    expect(parts.map((p) => p.name)).toEqual(["moved", "spun"]);
    // The translation bakes into world positions; so does the +90° Z spin.
    expect(Array.from(parts[0].positions)).toEqual([10, 0, 0, 11, 0, 0, 10, 1, 0]);
    for (const [got, want] of [
      [parts[1].positions[0], 0],
      [parts[1].positions[1], 0],
      [parts[1].positions[3], 0],
      [parts[1].positions[4], 1],
      [parts[1].positions[6], -1],
      [parts[1].positions[7], 0],
    ] as const) {
      expect(got).toBeCloseTo(want, 6);
    }
    expect(Array.from(parts[0].placement.subarray(12, 15))).toEqual([10, 0, 0]);
    // A float quaternion normalizes, so the spin reads back up to an epsilon.
    for (const [got, want] of [
      [parts[1].placement[0], 0],
      [parts[1].placement[1], 1],
      [parts[1].placement[4], -1],
      [parts[1].placement[5], 0],
    ] as const) {
      expect(got).toBeCloseTo(want, 6);
    }
  });

  it("reads_float_colors_and_expands_fans_with_mesh_name_fallback", () => {
    const bin = new Uint8Array(4 * 12 + 4 * 12);
    bin.set(f32([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]), 0);
    bin.set(f32([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]), 48);
    const doc = {
      asset: { version: "2.0" },
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ name: "fan", primitives: [{ attributes: { POSITION: 0, COLOR_0: 1 }, mode: 6 }] }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 4, type: "VEC3" },
        { bufferView: 1, componentType: 5126, count: 4, type: "VEC3" },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 48 },
        { buffer: 0, byteOffset: 48, byteLength: 48 },
      ],
      buffers: [{ byteLength: bin.length, uri: uriOf(bin) }],
    };
    const [part] = importGLTF(glb(doc, bin));
    // The unnamed node falls back to the mesh name; the VEC3 colors gain an opaque alpha.
    expect(part.name).toBe("fan");
    expect(part.indices.length).toBe(6);
    expect(triangleSet(part.positions, part.indices)).toEqual(new Set(["0,0,0|1,0,0|1,1,0", "0,0,0|0,1,0|1,1,0"]));
    expect(Array.from(part.colors ?? [])).toEqual([
      255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    ]);
  });

  it("deduplicates_part_names", () => {
    const bin = new Uint8Array(36 + 3);
    bin.set(f32([0, 0, 0, 1, 0, 0, 0, 1, 0]), 0);
    bin.set([0, 1, 2], 36);
    const doc = {
      asset: { version: "2.0" },
      scenes: [{ nodes: [0, 1] }],
      nodes: [
        { name: "dup", mesh: 0 },
        { name: "dup", mesh: 0 },
      ],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: "VEC3" },
        { bufferView: 1, componentType: 5121, count: 3, type: "SCALAR" },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 36 },
        { buffer: 0, byteOffset: 36, byteLength: 3 },
      ],
      buffers: [{ byteLength: bin.length, uri: uriOf(bin) }],
    };
    expect(importGLTF(jsonAsset(doc)).map((p) => p.name)).toEqual(["dup", "dup (2)"]);
  });

  it("rejects_draco_primitives", () => {
    const doc = {
      asset: { version: "2.0" },
      scenes: [{ nodes: [0] }],
      nodes: [{ name: "zipped", mesh: 0 }],
      meshes: [
        {
          primitives: [
            {
              attributes: { POSITION: 0 },
              extensions: { KHR_draco_mesh_compression: { bufferView: 0 } },
            },
          ],
        },
      ],
      accessors: [{ bufferView: 0, componentType: 5126, count: 0, type: "VEC3" }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 0 }],
      buffers: [{ byteLength: 0, uri: uriOf(new Uint8Array(0)) }],
    };
    expect(() => importGLTF(jsonAsset(doc))).toThrow(/KHR_draco_mesh_compression/);
  });

  it("rejects_external_buffers", () => {
    const doc = {
      asset: { version: "2.0" },
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 1, type: "VEC3" }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 12 }],
      buffers: [{ byteLength: 12, uri: "model.bin" }],
    };
    expect(() => importGLTF(jsonAsset(doc))).toThrow(/external/);
  });

  it("rejects_files_without_triangle_meshes", () => {
    const points = {
      asset: { version: "2.0" },
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, mode: 0 }] }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 1, type: "VEC3" }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 12 }],
      buffers: [{ byteLength: 12, uri: uriOf(f32([0, 0, 0])) }],
    };
    expect(() => importGLTF(jsonAsset(points))).toThrow(/mode 0/);
    const empty = { asset: { version: "2.0" }, scenes: [{ nodes: [] }] };
    expect(() => importGLTF(jsonAsset(empty))).toThrow(/no triangle meshes/);
  });

  it("rejects_non_gltf_bytes", () => {
    expect(() => importGLTF(new Uint8Array([1, 2, 3, 4]))).toThrow(/glTF/);
  });
});
