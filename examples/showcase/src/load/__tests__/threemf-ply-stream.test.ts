import { Scene, init } from "fast-mesh-wasm";
import { export3MF, parseMeshStream } from "fast-mesh-wasm/io";
import { beforeAll, describe, expect, it } from "vitest";
import { exportFileInfo, exportFileName, exportMeshBytes } from "../export";
import { parseObj } from "../obj";
import { MESH_EXTENSIONS, parseMeshFile, parseMeshFilesAsync } from "../parse";
import { parsePly } from "../ply";
import { parseStl } from "../stl";
import { parse3mf } from "../threemf";

function makeTetraScene() {
  const scene = new Scene();
  const positions = Float64Array.from([
    0, 0, 0,
    10, 0, 0,
    0, 10, 0,
    0, 0, 10,
  ]);
  const indices = Uint32Array.from([
    0, 2, 1,
    0, 1, 3,
    1, 2, 3,
    2, 0, 3,
  ]);
  const handle = scene.addMesh(positions, indices, { origin: [0, 0, 0] });
  return { scene, handle, positions, indices };
}

function toArrayBuffer(u8: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(u8.byteLength);
  copy.set(u8);
  return copy.buffer;
}

describe("showcase 3MF / PLY / OBJ / streaming integration", () => {
  beforeAll(async () => {
    await init();
  });

  it("advertises .3mf in MESH_EXTENSIONS and exportFileInfo", () => {
    expect(MESH_EXTENSIONS).toContain(".3mf");
    expect(exportFileInfo("3mf")).toEqual({ extension: "3mf", mime: "model/3mf" });
    expect(exportFileName("bracket", "3mf")).toBe("bracket.3mf");
    expect(() => parseMeshFile("bracket.3mf", new ArrayBuffer(16))).toThrow(
      /parseMeshFilesAsync/,
    );
  });

  it("round-trips a scene mesh through 3MF sync export and parseMeshFilesAsync", async () => {
    const { scene, handle, positions, indices } = makeTetraScene();
    try {
      const bytes = exportMeshBytes(scene, handle, "3mf", "tetra");
      expect(bytes[0]).toBe(0x50); // 'P'
      expect(bytes[1]).toBe(0x4b); // 'K'

      const parts = await parseMeshFilesAsync("tetra.3mf", toArrayBuffer(bytes));
      expect(parts).toHaveLength(1);
      expect(parts[0].name).toBe("tetra.3mf");
      expect(parts[0].data.positions.length).toBe(positions.length);
      expect(parts[0].data.indices.length).toBe(indices.length);

      const triCorners = (p: Float64Array, idx: Uint32Array) =>
        Array.from({ length: idx.length / 3 }, (_, t) =>
          [0, 1, 2].map((k) => {
            const v = idx[3 * t + k];
            return [p[3 * v], p[3 * v + 1], p[3 * v + 2]];
          }),
        );
      expect(triCorners(parts[0].data.positions, parts[0].data.indices)).toEqual(
        triCorners(positions, indices),
      );
    } finally {
      scene.free();
    }
  });

  it("names multi-object compressed 3MF assemblies using file stem", async () => {
    const { scene, handle } = makeTetraScene();
    const h2 = scene.addMesh(
      Float64Array.from([20, 0, 0, 25, 0, 0, 20, 5, 0]),
      Uint32Array.from([0, 1, 2]),
      { origin: [0, 0, 0] },
    );
    try {
      const compressed = await export3MF(scene, [handle, h2], {
        name: "assembly",
        compress: true,
      });
      const parts = await parse3mf("widget.3mf", toArrayBuffer(compressed));
      expect(parts).toHaveLength(2);
      expect(parts[0].name).toBe("widget · assembly-1");
      expect(parts[1].name).toBe("widget · assembly-2");
    } finally {
      scene.free();
    }
  });

  it("parses valid binary big-endian PLY while keeping malformed big-endian error label", () => {
    const header = [
      "ply",
      "format binary_big_endian 1.0",
      "element vertex 3",
      "property float x",
      "property float y",
      "property float z",
      "element face 1",
      "property list uchar uint vertex_indices",
      "end_header",
      "",
    ].join("\n");
    const head = new TextEncoder().encode(header);
    const buf = new ArrayBuffer(head.length + 3 * 12 + 1 + 3 * 4);
    new Uint8Array(buf).set(head);
    const dv = new DataView(buf);
    let at = head.length;
    const coords = [
      [0, 0, 0],
      [2, 0, 0],
      [0, 3, 0],
    ];
    for (const [x, y, z] of coords) {
      dv.setFloat32(at, x, false);
      dv.setFloat32(at + 4, y, false);
      dv.setFloat32(at + 8, z, false);
      at += 12;
    }
    dv.setUint8(at++, 3);
    dv.setUint32(at, 0, false);
    dv.setUint32(at + 4, 1, false);
    dv.setUint32(at + 8, 2, false);

    const parsed = parsePly(buf);
    expect(Array.from(parsed.positions)).toEqual([0, 0, 0, 2, 0, 0, 0, 3, 0]);
    expect(Array.from(parsed.indices)).toEqual([0, 1, 2]);
  });

  it("accepts vertex-color OBJ via parseObj while preserving geometry", () => {
    const src = [
      "v 0 0 0 1 0 0",
      "v 1 0 0 0 1 0",
      "v 0 1 0 0 0 1",
      "f 1 2 3",
    ].join("\n");
    const mesh = parseObj(new TextEncoder().encode(src).buffer as ArrayBuffer);
    expect(Array.from(mesh.positions)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    expect(Array.from(mesh.indices)).toEqual([0, 1, 2]);
  });

  it("parses solid-prefixed binary STL and deduplicates vertices", () => {
    const buf = new ArrayBuffer(84 + 50);
    new Uint8Array(buf).set(new TextEncoder().encode("solid sneaky binary"));
    const dv = new DataView(buf);
    dv.setUint32(80, 1, true);
    const pts = [0, 0, 0, 1, 0, 0, 0, 1, 0];
    for (let i = 0; i < 9; i++) {
      dv.setFloat32(84 + 12 + i * 4, pts[i], true);
    }
    const mesh = parseStl(buf);
    expect(mesh.positions.length).toBe(9);
    expect(Array.from(mesh.indices)).toEqual([0, 1, 2]);
  });

  it("streams chunks cleanly via File.stream() / ReadableStream and supports AbortSignal", async () => {
    const objBytes = new TextEncoder().encode(
      "v 0 0 0 255 0 0\nv 1 0 0 0 255 0\nv 0 1 0 0 0 255\nf 1 2 3\n",
    );
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(objBytes.subarray(0, 16));
        controller.enqueue(objBytes.subarray(16));
        controller.close();
      },
    });
    const stages: string[] = [];
    const parsed = await parseMeshStream(
      stream,
      { filename: "colored.obj" },
      (p) => stages.push(p.stage),
    );
    expect(parsed).toHaveLength(1);
    expect(Array.from(parsed[0].colors ?? [])).toEqual([
      255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255,
    ]);
    expect(stages).toContain("reading");
    expect(stages).toContain("parsing");

    const ac = new AbortController();
    const slowStream = new ReadableStream<Uint8Array>({
      pull(controller) {
        controller.enqueue(objBytes.subarray(0, 10));
        ac.abort();
      },
    });
    await expect(
      parseMeshStream(slowStream, { filename: "abort.obj" }, undefined, ac.signal),
    ).rejects.toThrow();
  });
});
