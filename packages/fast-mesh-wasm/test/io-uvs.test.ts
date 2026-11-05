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
