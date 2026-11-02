// Draco decoding: standalone `.drc` round-trips and the
// `KHR_draco_mesh_compression` glTF extension. The bitstreams are synthesized
// at test time with draco3d's own encoder, so no binary fixtures are checked
// in; the decoder under test is the same WASM module the app ships.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createDecoderModule, createEncoderModule } from "draco3d";
import type { DecoderModule, EncoderModule } from "draco3d";
import { decodeDraco, importGLTF, importGLTFAsync } from "../src/io/index.js";

/** Two triangles on the XY plane facing +Z. */
const QUAD = [0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0];
const QUAD_FACES = [0, 1, 2, 0, 2, 3];

/** The encoder's mesh builder as draco3d@1.5 exposes it (`@types/draco3d@1.4` predates the `*ToMesh` names). */
interface DracoMeshBuilder {
  AddFacesToMesh(mesh: object, numFaces: number, faces: Uint32Array): void;
  AddFloatAttributeToMesh(mesh: object, attribute: number, count: number, size: number, values: Float32Array): void;
}

/** Encodes the quad into Draco bytes (14-bit POSITION quantization). */
async function encodeQuad(encoderModule: EncoderModule): Promise<Uint8Array> {
  const mesh = new encoderModule.Mesh();
  const builder = new encoderModule.MeshBuilder() as unknown as DracoMeshBuilder;
  try {
    builder.AddFacesToMesh(mesh, 2, Uint32Array.from(QUAD_FACES));
    builder.AddFloatAttributeToMesh(mesh, encoderModule.POSITION, 4, 3, Float32Array.from(QUAD));
    const encoder = new encoderModule.Encoder();
    try {
      encoder.SetSpeedOptions(5, 5);
      encoder.SetAttributeQuantization(encoderModule.POSITION, 14);
      const out = new encoderModule.DracoInt8Array();
      try {
        const length = encoder.EncodeMeshToDracoBuffer(mesh, out);
        expect(length).toBeGreaterThan(0);
        const bytes = new Uint8Array(length);
        for (let i = 0; i < length; i++) {
          bytes[i] = out.GetValue(i);
        }
        return bytes;
      } finally {
        encoderModule.destroy(out);
      }
    } finally {
      encoderModule.destroy(encoder);
    }
  } finally {
    encoderModule.destroy(builder);
    encoderModule.destroy(mesh);
  }
}

/** Decoded triangles as rounded corner strings, so vertex order and quantization noise do not matter. */
function triangleSet(positions: ArrayLike<number>, indices: ArrayLike<number>): Set<string> {
  const tris = new Set<string>();
  for (let t = 0; t < indices.length; t += 3) {
    const corners = [0, 1, 2].map((k) => {
      const v = indices[t + k];
      const rounded = [0, 1, 2].map((c) => positions[3 * v + c].toFixed(2));
      return rounded.join(",");
    });
    corners.sort();
    tris.add(corners.join("|"));
  }
  return tris;
}

const EXPECTED = new Set(["0.00,0.00,0.00|2.00,0.00,0.00|2.00,2.00,0.00", "0.00,0.00,0.00|0.00,2.00,0.00|2.00,2.00,0.00"]);

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

/** A glTF asset whose primitive is Draco-compressed: the extension carries the bitstream. */
function dracoDoc(drc: Uint8Array, translate: [number, number, number] = [0, 0, 0]): object {
  return {
    asset: { version: "2.0" },
    scenes: [{ nodes: [0] }],
    nodes: [{ name: "zipped", translation: translate, mesh: 0 }],
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
    buffers: [{ byteLength: drc.length, uri: `data:application/octet-stream;base64,${toBase64(drc)}` }],
  };
}

function jsonBytes(doc: object): Uint8Array {
  return new TextEncoder().encode(JSON.stringify(doc));
}

/** One uncompressed triangle (f32 positions, u16 indices) as a glTF asset. */
function plainDoc(): object {
  const bin = new Uint8Array(36 + 6);
  bin.set(
    (() => {
      const out = new Uint8Array(36);
      const view = new DataView(out.buffer);
      [0, 0, 0, 1, 0, 0, 0, 1, 0].forEach((v, i) => view.setFloat32(4 * i, v, true));
      return out;
    })(),
    0,
  );
  bin.set([0, 0, 1, 0, 2, 0], 36);
  return {
    asset: { version: "2.0" },
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: "VEC3" },
      { bufferView: 1, componentType: 5123, count: 3, type: "SCALAR" },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36 },
      { buffer: 0, byteOffset: 36, byteLength: 6 },
    ],
    buffers: [{ byteLength: bin.length, uri: `data:application/octet-stream;base64,${toBase64(bin)}` }],
  };
}

describe("io draco", () => {
  it("decodes_a_drc_round_trip", async () => {
    const encoderModule = await createEncoderModule({});
    const drc = await encodeQuad(encoderModule);
    const { positions, indices } = await decodeDraco(drc);
    expect(positions).toBeInstanceOf(Float64Array);
    expect(indices).toBeInstanceOf(Uint32Array);
    expect(indices.length).toBe(6);
    expect(triangleSet(positions, indices)).toEqual(EXPECTED);
  });

  it("reuses_a_supplied_decoder_module", async () => {
    const encoderModule = await createEncoderModule({});
    const drc = await encodeQuad(encoderModule);
    const decoderModule: DecoderModule = await createDecoderModule({});
    const { positions, indices } = await decodeDraco(drc, { decoderModule });
    expect(triangleSet(positions, indices)).toEqual(EXPECTED);
  });

  it("seeds_the_decoder_from_wasm_bytes", async () => {
    const encoderModule = await createEncoderModule({});
    const drc = await encodeQuad(encoderModule);
    const wasmBinary = readFileSync(new URL("../node_modules/draco3d/draco_decoder.wasm", import.meta.url));
    const { positions, indices } = await decodeDraco(drc, { wasmBinary });
    expect(triangleSet(positions, indices)).toEqual(EXPECTED);
  });

  it("rejects_non_draco_bytes", async () => {
    await expect(decodeDraco(new Uint8Array([1, 2, 3, 4]))).rejects.toThrow(/Draco/);
    await expect(decodeDraco(new TextEncoder().encode("hello"))).rejects.toThrow(/Draco/);
  });

  it("rejects_a_foreign_decoder_module", async () => {
    await expect(decodeDraco(new Uint8Array([0]), { decoderModule: {} })).rejects.toThrow(/decoder module/);
  });

  it("decodes_khr_draco_mesh_compression_inside_gltf", async () => {
    const encoderModule = await createEncoderModule({});
    const drc = await encodeQuad(encoderModule);
    const [part] = await importGLTFAsync(jsonBytes(dracoDoc(drc, [5, 0, 0])));
    expect(part.name).toBe("zipped");
    // The node placement bakes into world positions.
    expect(triangleSet(part.positions, part.indices)).toEqual(
      new Set([
        "5.00,0.00,0.00|7.00,0.00,0.00|7.00,2.00,0.00",
        "5.00,0.00,0.00|5.00,2.00,0.00|7.00,2.00,0.00",
      ]),
    );
    expect(part.positions[0]).toBeCloseTo(5, 2);
    expect(part.colors).toBeNull();
    expect(Array.from(part.placement)).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1]);
    // The sync importer still throws on the same asset: it cannot await the decoder.
    expect(() => importGLTF(jsonBytes(dracoDoc(drc)))).toThrow(/KHR_draco_mesh_compression/);
  });

  it("reads_uncompressed_gltf_without_a_decoder", async () => {
    // An invalid module would throw on use: success proves the decoder never initializes.
    const [part] = await importGLTFAsync(jsonBytes(plainDoc()), { decoderModule: {} });
    expect(part.indices.length).toBe(3);
    expect(part.positions.length).toBe(9);
  });

  it("rejects_bad_khr_extensions", async () => {
    const encoderModule = await createEncoderModule({});
    const drc = await encodeQuad(encoderModule);
    const docOf = (extension: unknown, mode?: number): Uint8Array => {
      const doc = dracoDoc(drc) as {
        meshes: { primitives: { extensions?: unknown; mode?: number }[] }[];
      };
      doc.meshes[0].primitives[0].extensions = { KHR_draco_mesh_compression: extension };
      if (mode !== undefined) {
        doc.meshes[0].primitives[0].mode = mode;
      }
      return jsonBytes(doc);
    };
    await expect(importGLTFAsync(docOf({}))).rejects.toThrow(/bad KHR_draco_mesh_compression/);
    await expect(
      importGLTFAsync(docOf({ bufferView: 0, attributes: { POSITION: 0 } }, 5)),
    ).rejects.toThrow(/TRIANGLES/);
    await expect(importGLTFAsync(docOf({ bufferView: 7, attributes: {} }))).rejects.toThrow(/buffer view 7/);
  });
});
