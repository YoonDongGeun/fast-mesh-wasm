// TEXCOORD_0 through the glTF importers and exporter: plain, external-buffer
// and Draco paths, each a round trip through a scene, seams included.
import { createDecoderModule, createEncoderModule } from "draco3d";
import { describe, expect, it } from "vitest";
import {
  decodeDraco,
  encodeDraco,
  exportGLTF,
  exportOBJ,
  type GltfExportOptions,
  importGLTF,
  importGLTFAsync,
} from "../src/io/index.js";
import { Scene } from "../src/node.js";

/** Two quads with a UV seam along their shared edge: slots 4 and 5 repeat 1 and 2. */
const SEAM = {
  positions: Float64Array.from([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 1, 0]),
  indices: Uint32Array.from([0, 1, 2, 0, 2, 3, 4, 6, 7, 4, 7, 5]),
  uvs: Float32Array.from([0, 0, 0.5, 0, 0.5, 1, 0, 1, 0.75, 0, 0.75, 1, 1, 0, 1, 1]),
};

function seamScene(s: Scene) {
  const m = s.addMesh(SEAM.positions, SEAM.indices, { uvs: SEAM.uvs, normals: false });
  return m;
}

/** A vertex's position and UV, the order-free identity of a seamed vertex. */
function corners(p: ArrayLike<number>, uvs: ArrayLike<number>, digits: number): string[] {
  const out: string[] = [];
  for (let v = 0; v < p.length / 3; v++) {
    const f = (x: number) => x.toFixed(digits);
    out.push(`${f(p[3 * v])} ${f(p[3 * v + 1])} ${f(p[3 * v + 2])} : ${f(uvs[2 * v])} ${f(uvs[2 * v + 1])}`);
  }
  return out.sort();
}

function jsonOf(bytes: Uint8Array): any {
  return JSON.parse(new TextDecoder().decode(bytes));
}

describe("exportGLTF and importGLTF", () => {
  it("round-trips UVs exactly, seams intact", () => {
    const s = new Scene();
    const m = seamScene(s);
    const [back] = importGLTF(exportGLTF(s, m));
    expect(corners(back.positions, back.uvs as Float32Array, 6)).toEqual(corners(SEAM.positions, SEAM.uvs, 6));
    // The second trip, through the scene again, is the same.
    const again = s.addMesh(back.positions, back.indices, { uvs: back.uvs ?? undefined });
    const [twice] = importGLTF(exportGLTF(s, again));
    expect(corners(twice.positions, twice.uvs as Float32Array, 6)).toEqual(corners(SEAM.positions, SEAM.uvs, 6));
    expect(s.repair(again, { op: "weld", tolerance: 0.001 }).mergedVertices).toBe(0);
    s.free();
  });

  it("writes a float VEC2 TEXCOORD_0 accessor in JSON too", () => {
    const s = new Scene();
    const doc = jsonOf(exportGLTF(s, seamScene(s), { binary: false }));
    const primitive = doc.meshes[0].primitives[0];
    const accessor = doc.accessors[primitive.attributes.TEXCOORD_0];
    expect(accessor).toMatchObject({ componentType: 5126, type: "VEC2", count: 8 });
    const [back] = importGLTF(new TextEncoder().encode(JSON.stringify(doc)));
    expect(corners(back.positions, back.uvs as Float32Array, 6)).toEqual(corners(SEAM.positions, SEAM.uvs, 6));
    s.free();
  });

  it("sits with normals and aligned buffer views", () => {
    const s = new Scene();
    const m = s.addMesh(SEAM.positions, SEAM.indices, { uvs: SEAM.uvs });
    const doc = jsonOf(exportGLTF(s, m, { binary: false }));
    const primitive = doc.meshes[0].primitives[0];
    expect(Object.keys(primitive.attributes)).toEqual(["POSITION", "NORMAL", "TEXCOORD_0"]);
    for (const view of doc.bufferViews) {
      expect(view.byteOffset % 4).toBe(0);
    }
    s.free();
  });

  it("writes none for a mesh without UVs, and imports null", () => {
    const s = new Scene();
    const m = s.addMesh(SEAM.positions, SEAM.indices);
    const doc = jsonOf(exportGLTF(s, m, { binary: false }));
    expect(doc.meshes[0].primitives[0].attributes.TEXCOORD_0).toBeUndefined();
    expect(importGLTF(exportGLTF(s, m))[0].uvs).toBeNull();
    s.free();
  });

  it("compacts UVs with the vertices after an edit", () => {
    const s = new Scene();
    const m = seamScene(s);
    // Drop the left quad: slots 0..3 go.
    s.edit(m, [
      { op: "removeTriangles", triangles: [0, 1] },
      { op: "removeVertices", vertices: [0, 1, 2, 3] },
    ]);
    const [back] = importGLTF(exportGLTF(s, m, { applyPlacement: false }));
    expect(back.positions.length / 3).toBe(4);
    const world = back.positions.map((x, i) => x + [1.0, 0.5, 0][i % 3]);
    expect(corners(world, back.uvs as Float32Array, 3)).toEqual(
      corners(SEAM.positions.subarray(12), SEAM.uvs.subarray(8), 3),
    );
    s.free();
  });

  it("a dropped channel exports without UVs", () => {
    const s = new Scene();
    const box = {
      positions: Float64Array.from(
        [0, 1].flatMap((z) => [0, 0, z, 1, 0, z, 1, 1, z, 0, 1, z]),
      ),
      indices: Uint32Array.from([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 3, 0, 4, 3, 4, 7]),
    };
    const a = s.addMesh(box.positions, box.indices, { uvs: new Float32Array(16) });
    const b = s.addMesh(box.positions.map((x, i) => (i % 3 === 0 ? x + 0.5 : x)), box.indices);
    s.boolean(a, b, "union");
    expect(importGLTF(exportGLTF(s, a))[0].uvs).toBeNull();
    s.undo();
    expect(importGLTF(exportGLTF(s, a))[0].uvs?.length).toBe(16);
    s.free();
  });

  it("does not put UVs in OBJ", () => {
    const s = new Scene();
    const text = new TextDecoder().decode(exportOBJ(s, seamScene(s)));
    expect(text).not.toContain("vt ");
    s.free();
  });
});

/** A one-triangle glTF document over the given bin bytes. */
function asset(bin: Uint8Array, uv: Record<string, unknown>) {
  const b64 = Buffer.from(bin).toString("base64");
  const doc = {
    asset: { version: "2.0" },
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 } }] }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3" }, uv],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36 },
      { buffer: 0, byteOffset: 36, byteLength: bin.length - 36 },
    ],
    buffers: [{ byteLength: bin.length, uri: `data:application/octet-stream;base64,${b64}` }],
  };
  return new TextEncoder().encode(JSON.stringify(doc));
}

function binWith(uvBytes: ArrayBufferView): Uint8Array {
  const p = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const out = new Uint8Array(36 + uvBytes.byteLength);
  out.set(new Uint8Array(p.buffer), 0);
  out.set(new Uint8Array(uvBytes.buffer, uvBytes.byteOffset, uvBytes.byteLength), 36);
  return out;
}

describe("importGLTF TEXCOORD_0 forms", () => {
  it("reads normalized u8 and u16", () => {
    const u8 = asset(binWith(Uint8Array.from([0, 255, 51, 102, 255, 0, 0, 0])), {
      bufferView: 1,
      componentType: 5121,
      normalized: true,
      count: 3,
      type: "VEC2",
    });
    expect(Array.from(importGLTF(u8)[0].uvs as Float32Array)).toEqual(
      [0, 1, 0.2, 0.4, 1, 0].map((x) => Math.fround(x)),
    );
    const u16 = asset(binWith(Uint16Array.from([0, 65535, 65535, 0, 0, 0])), {
      bufferView: 1,
      componentType: 5123,
      normalized: true,
      count: 3,
      type: "VEC2",
    });
    expect(Array.from(importGLTF(u16)[0].uvs as Float32Array)).toEqual([0, 1, 1, 0, 0, 0]);
  });

  it("honours a stride", () => {
    const interleaved = new Float32Array([0.1, 0.2, 9, 0.3, 0.4, 9, 0.5, 0.6, 9]);
    const bytes = asset(binWith(interleaved), { bufferView: 1, componentType: 5126, count: 3, type: "VEC2" });
    const doc = JSON.parse(new TextDecoder().decode(bytes));
    doc.bufferViews[1].byteStride = 12;
    const back = importGLTF(new TextEncoder().encode(JSON.stringify(doc)))[0].uvs as Float32Array;
    expect(Array.from(back)).toEqual([0.1, 0.2, 0.3, 0.4, 0.5, 0.6].map((x) => Math.fround(x)));
  });

  it("reads an accessor without a buffer view as zeros", () => {
    const bytes = asset(binWith(new Uint8Array(4)), { componentType: 5126, count: 3, type: "VEC2" });
    expect(Array.from(importGLTF(bytes)[0].uvs as Float32Array)).toEqual([0, 0, 0, 0, 0, 0]);
  });

  it("throws on malformed UVs", () => {
    const uvs = new Float32Array(6);
    const accessor = { bufferView: 1, componentType: 5126, count: 3, type: "VEC2" };
    expect(() => importGLTF(asset(binWith(uvs), { ...accessor, count: 2 }))).toThrow(/2 UVs for 3 vertices/);
    expect(() => importGLTF(asset(binWith(uvs), { ...accessor, type: "VEC3", count: 1 }))).toThrow(/TEXCOORD_0/);
    uvs[3] = Number.NaN;
    expect(() => importGLTF(asset(binWith(uvs), accessor))).toThrow(/non-finite UV at vertex 1/);
  });
});

/** `asset` with extra top-level document fields, like `extensionsRequired`. */
function assetWith(bin: Uint8Array, uv: Record<string, unknown>, extra: Record<string, unknown>) {
  const doc = jsonOf(asset(bin, uv));
  return new TextEncoder().encode(JSON.stringify({ ...doc, ...extra }));
}

const QUANTIZED = { extensionsUsed: ["KHR_mesh_quantization"], extensionsRequired: ["KHR_mesh_quantization"] };
const TRIANGLE = [0, 0, 0, 1, 0, 0, 0, 1, 0];

describe("importGLTF unsupported TEXCOORD_0", () => {
  const sparse = { count: 1, indices: { bufferView: 1, componentType: 5123 }, values: { bufferView: 1 } };
  const unsupported: [string, Record<string, unknown>, ArrayBufferView, RegExp][] = [
    ["normalized i8", { bufferView: 1, componentType: 5120, normalized: true }, new Int8Array(8), /KHR_mesh_quantization/],
    ["normalized i16", { bufferView: 1, componentType: 5122, normalized: true }, new Int16Array(6), /KHR_mesh_quantization/],
    ["raw i16", { bufferView: 1, componentType: 5122 }, new Int16Array(6), /KHR_mesh_quantization/],
    ["raw u16", { bufferView: 1, componentType: 5123 }, new Uint16Array(6), /KHR_mesh_quantization/],
    ["raw u8", { bufferView: 1, componentType: 5121 }, new Uint8Array(8), /KHR_mesh_quantization/],
    ["u32", { bufferView: 1, componentType: 5125 }, new Uint32Array(6), /component type 5125/],
    ["sparse float", { bufferView: 1, componentType: 5126, sparse }, new Float32Array(6), /sparse/],
    ["sparse, no buffer view", { componentType: 5126, sparse }, new Float32Array(6), /sparse/],
    ["sparse normalized u16", { bufferView: 1, componentType: 5123, normalized: true, sparse }, new Uint16Array(6), /sparse/],
  ];
  for (const [label, fields, bytes, reason] of unsupported) {
    it(`imports the geometry of ${label} without UVs and says why`, async () => {
      const file = asset(binWith(bytes), { count: 3, type: "VEC2", ...fields });
      for (const [mesh] of [importGLTF(file), await importGLTFAsync(file)]) {
        expect(Array.from(mesh.positions)).toEqual(TRIANGLE);
        expect(Array.from(mesh.indices)).toEqual([0, 1, 2]);
        expect(mesh.uvs).toBeNull();
        expect(mesh.uvsSkipped).toMatch(reason);
      }
    });
  }

  it("skips a sparse accessor even with the quantization extension", () => {
    const uv = { bufferView: 1, componentType: 5126, count: 3, type: "VEC2", sparse };
    const [mesh] = importGLTF(assetWith(binWith(new Float32Array(6)), uv, QUANTIZED));
    expect(Array.from(mesh.positions)).toEqual(TRIANGLE);
    expect(mesh.uvs).toBeNull();
    expect(mesh.uvsSkipped).toMatch(/sparse/);
  });

  it("skips only the primitive whose UVs are unsupported", () => {
    const uv = { bufferView: 1, componentType: 5126, count: 3, type: "VEC2" };
    const doc = jsonOf(asset(binWith(new Float32Array([0.25, 0.5, 0.75, 1, 0, 0])), uv));
    doc.accessors.push({ bufferView: 1, componentType: 5122, normalized: true, count: 3, type: "VEC2" });
    doc.meshes[0].primitives.push({ attributes: { POSITION: 0, TEXCOORD_0: 2 } });
    const [good, bad] = importGLTF(new TextEncoder().encode(JSON.stringify(doc)));
    expect(Array.from(good.uvs as Float32Array)).toEqual([0.25, 0.5, 0.75, 1, 0, 0]);
    expect(good.uvsSkipped).toBeUndefined();
    expect(Array.from(bad.positions)).toEqual(TRIANGLE);
    expect(bad.uvs).toBeNull();
    expect(bad.uvsSkipped).toMatch(/KHR_mesh_quantization/);
  });

  it("leaves normal UV files unaffected: nothing skipped", () => {
    const forms: [Record<string, unknown>, ArrayBufferView][] = [
      [{ componentType: 5126 }, new Float32Array(6)],
      [{ componentType: 5121, normalized: true }, new Uint8Array(8)],
      [{ componentType: 5123, normalized: true }, new Uint16Array(6)],
    ];
    for (const [fields, bytes] of forms) {
      const [mesh] = importGLTF(asset(binWith(bytes), { bufferView: 1, count: 3, type: "VEC2", ...fields }));
      expect(mesh.uvs).not.toBeNull();
      expect(mesh.uvsSkipped).toBeUndefined();
    }
  });

  it("a file without TEXCOORD_0 reports nothing either", () => {
    const uv = { bufferView: 1, componentType: 5126, count: 3, type: "VEC2" };
    const doc = jsonOf(asset(binWith(new Float32Array(6)), uv));
    delete doc.meshes[0].primitives[0].attributes.TEXCOORD_0;
    const [mesh] = importGLTF(new TextEncoder().encode(JSON.stringify(doc)));
    expect(mesh.uvs).toBeNull();
    expect(mesh.uvsSkipped).toBeUndefined();
  });
});

// The expected values are the glTF 2.0 accessor equations worked by hand:
// f = c / 255, c / 65535, max(c / 127, -1) and max(c / 32767, -1), or f = c
// without `normalized`.
describe("importGLTF TEXCOORD_0 with KHR_mesh_quantization", () => {
  const read = (fields: Record<string, unknown>, bytes: ArrayBufferView, extra: Record<string, unknown> = QUANTIZED) =>
    importGLTF(assetWith(binWith(bytes), { bufferView: 1, count: 3, type: "VEC2", ...fields }, extra))[0].uvs as Float32Array;

  it("keeps non-normalized u8 and u16 as stored", () => {
    expect(Array.from(read({ componentType: 5121 }, Uint8Array.from([0, 255, 51, 102, 200, 1, 0, 0])))).toEqual([0, 255, 51, 102, 200, 1]);
    expect(Array.from(read({ componentType: 5123 }, Uint16Array.from([0, 65535, 40000, 2, 7, 32768])))).toEqual([0, 65535, 40000, 2, 7, 32768]);
  });

  it("keeps non-normalized i8 and i16 as stored", () => {
    expect(Array.from(read({ componentType: 5120 }, Int8Array.from([-128, 127, 0, -1, 5, 6, 0, 0])))).toEqual([-128, 127, 0, -1, 5, 6]);
    expect(Array.from(read({ componentType: 5122 }, Int16Array.from([-32768, 32767, 0, -1, 5, 6])))).toEqual([-32768, 32767, 0, -1, 5, 6]);
  });

  it("maps normalized i8 and i16 to [-1, 1]", () => {
    const i8 = read({ componentType: 5120, normalized: true }, Int8Array.from([-128, -127, 127, 0, 64, -64, 0, 0]));
    [-1, -1, 1, 0, 0.50393701, -0.50393701].forEach((x, i) => expect(i8[i]).toBeCloseTo(x, 6));
    const i16 = read({ componentType: 5122, normalized: true }, Int16Array.from([-32768, -32767, 32767, 0, 16384, -16384]));
    [-1, -1, 1, 0, 0.50001526, -0.50001526].forEach((x, i) => expect(i16[i]).toBeCloseTo(x, 6));
  });

  it("still normalizes u8 and u16 that ask for it", () => {
    const u16 = read({ componentType: 5123, normalized: true }, Uint16Array.from([0, 65535, 13107, 26214, 0, 0]));
    [0, 1, 0.2, 0.4, 0, 0].forEach((x, i) => expect(u16[i]).toBeCloseTo(x, 6));
  });

  it("accepts the extension named in extensionsUsed alone", () => {
    const uvs = read({ componentType: 5123 }, Uint16Array.from([1, 2, 3, 4, 5, 6]), { extensionsUsed: ["KHR_mesh_quantization"] });
    expect(Array.from(uvs)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("does not read those forms under another extension", () => {
    const uv = { bufferView: 1, componentType: 5123, count: 3, type: "VEC2" };
    const [mesh] = importGLTF(assetWith(binWith(new Uint16Array(6)), uv, { extensionsUsed: ["KHR_texture_transform"] }));
    expect(mesh.uvs).toBeNull();
  });

  it("keeps a stride for the integer forms", () => {
    const uv = { bufferView: 1, componentType: 5123, count: 3, type: "VEC2" };
    const doc = jsonOf(assetWith(binWith(Uint16Array.from([1, 2, 99, 99, 3, 4, 99, 99, 5, 6])), uv, QUANTIZED));
    doc.bufferViews[1].byteStride = 8;
    const uvs = importGLTF(new TextEncoder().encode(JSON.stringify(doc)))[0].uvs as Float32Array;
    expect(Array.from(uvs)).toEqual([1, 2, 3, 4, 5, 6]);
  });
});

describe("Draco quantizationBits.texCoord", () => {
  const encode = (quantizationBits: Record<string, number>) =>
    encodeDraco(SEAM.positions, SEAM.indices, { uvs: SEAM.uvs, quantizationBits });

  it("is an alias of uv", async () => {
    expect(await encode({ texCoord: 3 })).toEqual(await encode({ uv: 3 }));
    expect(await encode({ texCoord: 3 })).not.toEqual(await encode({}));
  });

  it("gives way to uv", async () => {
    expect(await encode({ texCoord: 3, uv: 9 })).toEqual(await encode({ uv: 9 }));
  });

  it("is still validated", async () => {
    await expect(encode({ texCoord: 0 })).rejects.toThrow(/texCoord quantization bits/);
    await expect(encode({ texCoord: 0, uv: 9 })).rejects.toThrow(/texCoord quantization bits/);
  });
});

describe("importGLTFAsync external buffers", () => {
  it("keeps UVs read through resolveBuffer", async () => {
    const s = new Scene();
    const glb = exportGLTF(s, seamScene(s), { binary: false });
    const doc = jsonOf(glb);
    const data = doc.buffers[0].uri as string;
    const bin = Buffer.from(data.slice(data.indexOf(",") + 1), "base64");
    doc.buffers[0].uri = "model.bin";
    const [back] = await importGLTFAsync(new TextEncoder().encode(JSON.stringify(doc)), {
      resolveBuffer: (uri) => {
        expect(uri).toBe("model.bin");
        return new Uint8Array(bin);
      },
    });
    expect(corners(back.positions, back.uvs as Float32Array, 6)).toEqual(corners(SEAM.positions, SEAM.uvs, 6));
    s.free();
  });
});

describe("Draco", () => {
  const encoder = createEncoderModule({});
  const decoder = createDecoderModule({});
  const options = async (extra?: Partial<GltfExportOptions>) => ({
    draco: true as const,
    dracoOptions: { encoderModule: await encoder, ...extra?.dracoOptions },
    ...extra,
  });

  it("round-trips a glTF's UVs within the quantization, seams intact", async () => {
    const s = new Scene();
    const m = seamScene(s);
    const bytes = await exportGLTF(s, m, await options());
    const doc = jsonOf(await exportGLTF(s, m, { ...(await options()), binary: false }));
    expect(doc.meshes[0].primitives[0].extensions.KHR_draco_mesh_compression.attributes.TEXCOORD_0).toBeTypeOf("number");
    const [back] = await importGLTFAsync(bytes, { decoderModule: await decoder });
    expect(corners(back.positions, back.uvs as Float32Array, 3)).toEqual(corners(SEAM.positions, SEAM.uvs, 3));
    // The seam survives a weld of what came back.
    const again = s.addMesh(back.positions, back.indices, { uvs: back.uvs ?? undefined });
    expect(s.repair(again, { op: "weld", tolerance: 0.001 }).mergedVertices).toBe(0);
    s.free();
  });

  it("keeps UVs with a preserved face order and other quantization", async () => {
    const s = new Scene();
    const m = seamScene(s);
    const bytes = await exportGLTF(s, m, {
      draco: true,
      dracoOptions: { encoderModule: await encoder, preserveFaceOrder: true, quantizationBits: { uv: 12 } },
    });
    const [back] = await importGLTFAsync(bytes, { decoderModule: await decoder });
    expect(corners(back.positions, back.uvs as Float32Array, 2)).toEqual(corners(SEAM.positions, SEAM.uvs, 2));
    s.free();
  });

  it("a Draco export without UVs imports null", async () => {
    const s = new Scene();
    const m = s.addMesh(SEAM.positions, SEAM.indices);
    const bytes = await exportGLTF(s, m, await options());
    expect((await importGLTFAsync(bytes, { decoderModule: await decoder }))[0].uvs).toBeNull();
    s.free();
  });

  it("standalone encodeDraco and decodeDraco carry UVs", async () => {
    const positions = Float32Array.from(SEAM.positions);
    const bytes = await encodeDraco(positions, SEAM.indices, {
      encoderModule: await encoder,
      uvs: SEAM.uvs,
    });
    const back = await decodeDraco(bytes, { decoderModule: await decoder });
    expect(corners(back.positions, back.uvs as Float32Array, 3)).toEqual(corners(SEAM.positions, SEAM.uvs, 3));
    const plain = await encodeDraco(positions, SEAM.indices, { encoderModule: await encoder });
    expect((await decodeDraco(plain, { decoderModule: await decoder })).uvs).toBeNull();
    await expect(
      encodeDraco(positions, SEAM.indices, { encoderModule: await encoder, uvs: new Float32Array(3) }),
    ).rejects.toThrow(/uvs need 2 numbers/);
    await expect(
      encodeDraco(positions, SEAM.indices, { encoderModule: await encoder, uvs: new Float32Array(16).fill(Number.NaN) }),
    ).rejects.toThrow(/uv 0 is not finite/);
  });

  it("skips a sparse plain TEXCOORD_0 beside a Draco primitive", async () => {
    const s = new Scene();
    const m = s.addMesh(SEAM.positions, SEAM.indices, { normals: false });
    const doc = jsonOf(await exportGLTF(s, m, { ...(await options()), binary: false }));
    s.free();
    doc.meshes[0].primitives[0].attributes.TEXCOORD_0 = doc.accessors.length;
    const sparse = { count: 1, indices: { bufferView: 0, componentType: 5123 }, values: { bufferView: 0 } };
    doc.accessors.push({ bufferView: 0, componentType: 5126, count: 8, type: "VEC2", sparse });
    const bytes = new TextEncoder().encode(JSON.stringify(doc));
    const [mesh] = await importGLTFAsync(bytes, { decoderModule: await decoder });
    expect(mesh.indices.length).toBe(SEAM.indices.length);
    expect(mesh.uvs).toBeNull();
    expect(mesh.uvsSkipped).toMatch(/sparse/);
  });

  it("a bad TEXCOORD_0 id is an error", async () => {
    const s = new Scene();
    const doc = jsonOf(await exportGLTF(s, seamScene(s), { ...(await options()), binary: false }));
    const ext = doc.meshes[0].primitives[0].extensions.KHR_draco_mesh_compression;
    ext.attributes.TEXCOORD_0 = -1;
    await expect(
      importGLTFAsync(new TextEncoder().encode(JSON.stringify(doc)), { decoderModule: await decoder }),
    ).rejects.toThrow(/TEXCOORD_0 attribute id/);
    ext.attributes.TEXCOORD_0 = 99;
    await expect(
      importGLTFAsync(new TextEncoder().encode(JSON.stringify(doc)), { decoderModule: await decoder }),
    ).rejects.toThrow(/UV attribute has 0 components/);
    s.free();
  });
});
