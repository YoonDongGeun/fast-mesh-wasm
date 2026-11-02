// glTF external `.bin` buffers: `importGLTFAsync` loads them through a
// caller-supplied `resolveBuffer` callback (fetch / FileReader / fs), while
// the sync importer throws a helpful pointer at the async form. The assets
// are synthesized in-test: no binary fixtures.
import { describe, expect, it } from "vitest";
import { importGLTF, importGLTFAsync } from "../src/io/index.js";

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

/** One triangle (f32 positions, u16 indices) as raw buffer bytes. */
function triangleBin(): Uint8Array {
  const out = new Uint8Array(36 + 6);
  out.set(f32([0, 0, 0, 1, 0, 0, 0, 1, 0]), 0);
  out.set(u16([0, 1, 2]), 36);
  return out;
}

/** A `.gltf` JSON asset whose single buffer lives at the given URI. */
function docWithUris(uris: (string | null)[]): Uint8Array {
  const bin = triangleBin();
  const views =
    uris.length === 1
      ? [
          { buffer: 0, byteOffset: 0, byteLength: 36 },
          { buffer: 0, byteOffset: 36, byteLength: 6 },
        ]
      : [
          { buffer: 0, byteOffset: 0, byteLength: 36 },
          { buffer: 1, byteOffset: 0, byteLength: 6 },
        ];
  const doc = {
    asset: { version: "2.0" },
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ name: "tri", primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: "VEC3" },
      { bufferView: 1, componentType: 5123, count: 3, type: "SCALAR" },
    ],
    bufferViews: views,
    buffers: uris.map((uri, i) => ({
      byteLength: uris.length === 1 ? bin.length : i === 0 ? 36 : 6,
      ...(uri === null ? {} : { uri }),
    })),
  };
  return new TextEncoder().encode(JSON.stringify(doc));
}

describe("io external buffers", () => {
  it("loads_an_external_bin_through_the_resolver", async () => {
    const bin = triangleBin();
    const [part] = await importGLTFAsync(docWithUris(["triangle.bin"]), {
      resolveBuffer: (uri) => {
        expect(uri).toBe("triangle.bin");
        return bin;
      },
    });
    expect(part.name).toBe("tri");
    expect(Array.from(part.positions)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    expect(Array.from(part.indices)).toEqual([0, 1, 2]);
  });

  it("awaits_an_async_resolver_and_mixes_embedded_and_external_buffers", async () => {
    const bin = triangleBin();
    const embedded = `data:application/octet-stream;base64,${toBase64(bin.subarray(36))}`;
    const seen: string[] = [];
    const [part] = await importGLTFAsync(docWithUris(["positions.bin", embedded]), {
      resolveBuffer: async (uri) => {
        seen.push(uri);
        // Only the external buffer consults the resolver, and an ArrayBuffer
        // resolves as well as a Uint8Array.
        return bin.subarray(0, 36).slice().buffer as ArrayBuffer;
      },
    });
    expect(seen).toEqual(["positions.bin"]);
    expect(Array.from(part.positions)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    expect(Array.from(part.indices)).toEqual([0, 1, 2]);
  });

  it("rejects_external_buffers_without_a_resolver", async () => {
    await expect(importGLTFAsync(docWithUris(["triangle.bin"]))).rejects.toThrow(/resolveBuffer/);
  });

  it("wraps_a_throwing_resolver_with_buffer_index_and_uri", async () => {
    await expect(
      importGLTFAsync(docWithUris(["triangle.bin"]), {
        resolveBuffer: () => {
          throw new Error("404");
        },
      }),
    ).rejects.toThrow(/buffer 0.*triangle\.bin.*404/);
  });

  it("rejects_a_resolver_returning_no_bytes", async () => {
    await expect(
      importGLTFAsync(docWithUris(["triangle.bin"]), {
        resolveBuffer: () => undefined as unknown as Uint8Array,
      }),
    ).rejects.toThrow(/buffer 0.*triangle\.bin.*no bytes/);
    await expect(
      importGLTFAsync(docWithUris(["triangle.bin"]), {
        resolveBuffer: () => "nope" as unknown as Uint8Array,
      }),
    ).rejects.toThrow(/no bytes/);
  });

  it("the_sync_importer_points_at_the_async_form", () => {
    expect(() => importGLTF(docWithUris(["triangle.bin"]))).toThrow(/external/);
    expect(() => importGLTF(docWithUris(["triangle.bin"]))).toThrow(/importGLTFAsync/);
  });
});
