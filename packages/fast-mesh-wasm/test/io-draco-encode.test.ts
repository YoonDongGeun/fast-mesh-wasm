// Draco encoding: standalone `.drc` bytes (`encodeDraco`) and compressed glTF
// (`exportGLTF` with `draco: true`). Round-trips decode through this
// package's own `decodeDraco` / `importGLTFAsync`, so the write path stays in
// agreement with the read path; per-attribute checks (normals, colors) go
// through draco3d's own decoder, like the fixtures in `io-draco.test.ts`.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createDecoderModule, createEncoderModule } from "draco3d";
import type { DecoderModule, EncoderModule } from "draco3d";
import { MeshHandle } from "../src/handles.js";
import { decodeDraco, encodeDraco, exportGLTF, importGLTFAsync } from "../src/io/index.js";
import type { SceneLike } from "../src/scene-like.js";

/** Two triangles on the XY plane facing +Z. */
const QUAD = [0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0];
const QUAD_FACES = [0, 1, 2, 0, 2, 3];
const QUAD_NORMALS = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1];
const QUAD_COLORS = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255];

/** Six faces in a row: the edgebreaker traversal reorders them, sequential keeps them. */
const STRIP = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0, 0, 2, 1, 0, 3, 0, 0, 3, 1, 0];
const STRIP_FACES = [0, 1, 2, 0, 2, 3, 1, 4, 5, 1, 5, 2, 4, 6, 7, 4, 7, 5];

/** The export quad, with a dead slot of each kind (like `io-export.test.ts`). */
const EXPORT_POSITIONS = Float32Array.from([0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 9, 9, 9]);
const EXPORT_TRIS = Uint32Array.from([0, 1, 2, 0, 2, 3, 0, 0, 0]);
const EXPORT_NORMALS = Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
const EXPORT_COLORS = Uint8Array.from([
  255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255,
]);
const IDENTITY = Float64Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

/** Minimal in-memory SceneLike: positions, indices, optional normals/colors, fixed matrix. */
class FakeScene implements SceneLike {
  readonly mesh = new MeshHandle();
  constructor(
    readonly positions: Float32Array = EXPORT_POSITIONS,
    readonly indices: Uint32Array = EXPORT_TRIS,
    readonly normals: Float32Array | null = EXPORT_NORMALS,
    readonly colors: Uint8Array | null = EXPORT_COLORS,
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

/** Decodes Draco bytes with draco3d's own decoder (attributes beyond POSITION). */
function decodeWithDraco(bytes: Uint8Array, decoderModule: DecoderModule) {
  const decoder = new decoderModule.Decoder();
  const buffer = new decoderModule.DecoderBuffer();
  buffer.Init(new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), bytes.length);
  const mesh = new decoderModule.Mesh();
  const status = decoder.DecodeBufferToMesh(buffer, mesh);
  expect(status.ok()).toBe(true);
  return { decoder, buffer, mesh };
}

describe("io draco encode", () => {
  it("encodes_a_drc_round_trip", async () => {
    const drc = await encodeDraco(QUAD, QUAD_FACES);
    expect(drc).toBeInstanceOf(Uint8Array);
    expect(new TextDecoder().decode(drc.subarray(0, 5))).toBe("DRACO");
    const { positions, indices } = await decodeDraco(drc);
    expect(positions).toBeInstanceOf(Float64Array);
    expect(indices).toBeInstanceOf(Uint32Array);
    expect(indices.length).toBe(6);
    expect(triangleSet(positions, indices)).toEqual(EXPECTED);
  });

  it("accepts_typed_arrays_and_full_options", async () => {
    // Float64 positions narrow to f32; u16 indices widen; every knob set.
    for (const compressionLevel of [0, 5, 10]) {
      const drc = await encodeDraco(Float64Array.from(QUAD), Uint16Array.from(QUAD_FACES), {
        quantizationBits: { position: 10, normal: 8, color: 8, texCoord: 10 },
        compressionLevel,
      });
      const { positions, indices } = await decodeDraco(drc);
      expect(triangleSet(positions, indices)).toEqual(EXPECTED);
    }
  });

  it("round_trips_normals_and_colors", async () => {
    const drc = await encodeDraco(QUAD, QUAD_FACES, { normals: QUAD_NORMALS, colors: QUAD_COLORS });
    const decoderModule: DecoderModule = await createDecoderModule({});
    const { decoder, buffer, mesh } = decodeWithDraco(drc, decoderModule);
    try {
      expect(mesh.num_points()).toBe(4);
      // Normals survive quantization (10 bits by default); the vertex order
      // need not, so compare rounded triples as a set.
      const normalId = decoder.GetAttributeId(mesh, decoderModule.NORMAL);
      expect(normalId).toBeGreaterThanOrEqual(0);
      const normalAttr = decoder.GetAttribute(mesh, normalId);
      expect(normalAttr.num_components()).toBe(3);
      const normals = new decoderModule.DracoFloat32Array();
      try {
        decoder.GetAttributeFloatForAllPoints(mesh, normalAttr, normals);
        const seen = new Set<string>();
        for (let v = 0; v < mesh.num_points(); v++) {
          seen.add(
            [normals.GetValue(3 * v), normals.GetValue(3 * v + 1), normals.GetValue(3 * v + 2)]
              .map((x) => x.toFixed(2))
              .join(","),
          );
        }
        expect(seen).toEqual(new Set(["0.00,0.00,1.00"]));
      } finally {
        decoderModule.destroy(normals);
      }
      // Colors are lossless u8: the decoded quads match exactly, as a set.
      const colorId = decoder.GetAttributeId(mesh, decoderModule.COLOR);
      expect(colorId).toBeGreaterThanOrEqual(0);
      const colorAttr = decoder.GetAttribute(mesh, colorId);
      expect(colorAttr.num_components()).toBe(4);
      const colors = new decoderModule.DracoUInt8Array();
      try {
        decoder.GetAttributeUInt8ForAllPoints(mesh, colorAttr, colors);
        const seen = new Set<string>();
        for (let v = 0; v < mesh.num_points(); v++) {
          seen.add(
            [colors.GetValue(4 * v), colors.GetValue(4 * v + 1), colors.GetValue(4 * v + 2), colors.GetValue(4 * v + 3)].join(
              ",",
            ),
          );
        }
        expect(seen).toEqual(new Set(["255,0,0,255", "0,255,0,255", "0,0,255,255", "255,255,255,255"]));
      } finally {
        decoderModule.destroy(colors);
      }
    } finally {
      decoderModule.destroy(mesh);
      decoderModule.destroy(buffer);
      decoderModule.destroy(decoder);
    }
  });

  it("keeps_exact_face_order_when_asked", async () => {
    // Sequential encoding preserves the input order, vertex for vertex.
    const drc = await encodeDraco(STRIP, STRIP_FACES, { preserveFaceOrder: true });
    const { positions, indices } = await decodeDraco(drc);
    expect(Array.from(indices)).toEqual(STRIP_FACES);
    for (let i = 0; i < STRIP.length; i++) {
      expect(positions[i]).toBeCloseTo(STRIP[i], 2);
    }
    // The default (edgebreaker) only promises the same triangles.
    const fast = await encodeDraco(STRIP, STRIP_FACES);
    const back = await decodeDraco(fast);
    expect(triangleSet(back.positions, back.indices)).toEqual(triangleSet(STRIP, STRIP_FACES));
  });

  it("rejects_bad_inputs", async () => {
    await expect(encodeDraco([], [])).rejects.toThrow(/no vertices/);
    await expect(encodeDraco([0, 0], [0, 0, 0])).rejects.toThrow(/positions/);
    await expect(encodeDraco(QUAD, [])).rejects.toThrow(/no triangles/);
    await expect(encodeDraco(QUAD, [0, 1])).rejects.toThrow(/indices/);
    await expect(encodeDraco(QUAD, [0, 1, 99])).rejects.toThrow(/missing vertex 99/);
    await expect(encodeDraco(QUAD, [0, 1, 0.5])).rejects.toThrow(/missing vertex/);
    await expect(encodeDraco([Number.NaN, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 2])).rejects.toThrow(/finite/);
    await expect(encodeDraco(QUAD, QUAD_FACES, { normals: [0, 0, 0] })).rejects.toThrow(/normals/);
    await expect(encodeDraco(QUAD, QUAD_FACES, { colors: [1, 2, 3] })).rejects.toThrow(/colors/);
    await expect(encodeDraco(QUAD, QUAD_FACES, { colors: [...QUAD_COLORS.slice(0, 15), 300] })).rejects.toThrow(/colors/);
    await expect(encodeDraco(QUAD, QUAD_FACES, { quantizationBits: { position: 0 } })).rejects.toThrow(/quantization/);
    await expect(encodeDraco(QUAD, QUAD_FACES, { quantizationBits: { position: 31 } })).rejects.toThrow(/quantization/);
    await expect(encodeDraco(QUAD, QUAD_FACES, { quantizationBits: { normal: 2.5 } })).rejects.toThrow(/quantization/);
    await expect(encodeDraco(QUAD, QUAD_FACES, { compressionLevel: -1 })).rejects.toThrow(/compression level/);
    await expect(encodeDraco(QUAD, QUAD_FACES, { compressionLevel: 11 })).rejects.toThrow(/compression level/);
  });

  it("reuses_a_supplied_encoder_module", async () => {
    const encoderModule: EncoderModule = await createEncoderModule({});
    const drc = await encodeDraco(QUAD, QUAD_FACES, { encoderModule });
    const { positions, indices } = await decodeDraco(drc);
    expect(triangleSet(positions, indices)).toEqual(EXPECTED);
  });

  it("seeds_the_encoder_from_wasm_bytes", async () => {
    const wasmBinary = readFileSync(new URL("../node_modules/draco3d/draco_encoder.wasm", import.meta.url));
    const drc = await encodeDraco(QUAD, QUAD_FACES, { wasmBinary });
    const { positions, indices } = await decodeDraco(drc);
    expect(triangleSet(positions, indices)).toEqual(EXPECTED);
  });

  it("rejects_a_foreign_encoder_module", async () => {
    await expect(encodeDraco(QUAD, QUAD_FACES, { encoderModule: {} })).rejects.toThrow(/encoder module/);
  });

  it("exports_gltf_with_khr_draco_mesh_compression", async () => {
    const scene = new FakeScene();
    const pending = exportGLTF(scene, scene.mesh, { draco: true });
    expect(pending).toBeInstanceOf(Promise);
    const bytes = await pending;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    expect(view.getUint32(0, true)).toBe(0x46546c67);
    const jsonLength = view.getUint32(12, true);
    const json = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + jsonLength)));
    expect(json.extensionsUsed).toEqual(["KHR_draco_mesh_compression"]);
    expect(json.extensionsRequired).toEqual(["KHR_draco_mesh_compression"]);
    const primitive = json.meshes[0].primitives[0];
    expect(primitive.mode).toBe(4);
    expect(primitive.attributes).toEqual({ POSITION: 0, NORMAL: 1, COLOR_0: 2 });
    expect(primitive.indices).toBe(3);
    const extension = primitive.extensions["KHR_draco_mesh_compression"];
    expect(extension.bufferView).toBe(0);
    expect(extension.attributes).toEqual({ POSITION: 0, NORMAL: 1, COLOR_0: 2 });
    // Compressed accessors describe the decoded shape; the bytes live in the extension.
    expect(json.accessors.length).toBe(4);
    for (const accessor of json.accessors) {
      expect(accessor.bufferView).toBeUndefined();
    }
    expect(json.accessors[0].min).toEqual([0, 0, 0]);
    expect(json.accessors[0].max).toEqual([2, 2, 0]);
    expect(json.accessors[2]).toMatchObject({ componentType: 5121, type: "VEC4", normalized: true });
    expect(json.accessors[3]).toMatchObject({ componentType: 5125, count: 6, type: "SCALAR" });
    expect(json.bufferViews.length).toBe(1);
    expect(json.bufferViews[0].target).toBeUndefined();
    // The single buffer view is the .drc bitstream (dead slots compacted away).
    const binAt = 20 + jsonLength;
    const drc = bytes.subarray(binAt + 8, binAt + 8 + json.bufferViews[0].byteLength);
    expect(new TextDecoder().decode(drc.subarray(0, 5))).toBe("DRACO");
    const { positions, indices } = await decodeDraco(drc);
    expect(indices.length).toBe(6);
    expect(triangleSet(positions, indices)).toEqual(EXPECTED);
  });

  it("exports_draco_gltf_json_with_data_uri", async () => {
    const scene = new FakeScene();
    const json = JSON.parse(new TextDecoder().decode(await exportGLTF(scene, scene.mesh, { draco: true, binary: false })));
    expect(json.extensionsUsed).toEqual(["KHR_draco_mesh_compression"]);
    expect(json.meshes[0].primitives[0].extensions["KHR_draco_mesh_compression"].bufferView).toBe(0);
    const uri: string = json.buffers[0].uri;
    expect(uri.startsWith("data:application/octet-stream;base64,")).toBe(true);
    const drc = Buffer.from(uri.slice(uri.indexOf(",") + 1), "base64");
    const { positions, indices } = await decodeDraco(drc);
    expect(triangleSet(positions, indices)).toEqual(EXPECTED);
  });

  it("draco_gltf_round_trips_through_importGLTFAsync", async () => {
    const scene = new FakeScene();
    const bytes = await exportGLTF(scene, scene.mesh, { draco: true });
    const [part] = await importGLTFAsync(bytes);
    expect(part.name).toBe("mesh");
    expect(triangleSet(part.positions, part.indices)).toEqual(EXPECTED);
    // Compressed attributes beyond POSITION stay unmapped on the read path.
    expect(part.colors).toBeNull();
  });

  it("draco_gltf_without_normals_or_colors_maps_position_only", async () => {
    const scene = new FakeScene(EXPORT_POSITIONS, EXPORT_TRIS, null, null);
    const bytes = await exportGLTF(scene, scene.mesh, { draco: true });
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const jsonLength = view.getUint32(12, true);
    const json = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + jsonLength)));
    const primitive = json.meshes[0].primitives[0];
    expect(primitive.attributes).toEqual({ POSITION: 0 });
    expect(primitive.extensions["KHR_draco_mesh_compression"].attributes).toEqual({ POSITION: 0 });
    expect(json.accessors.length).toBe(2);
  });

  it("sync_exportGLTF_stays_synchronous_without_draco", () => {
    const scene = new FakeScene();
    expect(exportGLTF(scene, scene.mesh)).toBeInstanceOf(Uint8Array);
    expect(exportGLTF(scene, scene.mesh, { draco: false })).toBeInstanceOf(Uint8Array);
  });
});
