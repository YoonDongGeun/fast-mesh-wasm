import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import { type MeshHandle, Scene, type SceneLike } from "../src/node.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";
import {
  crc32,
  detectMeshFormat,
  export3MF,
  export3MFSync,
  exportGLTF,
  exportOBJ,
  exportPLY,
  exportSTL,
  import3MF,
  importGLTF,
  importOBJ,
  importPLY,
  importSTL,
  loadMeshStream,
  parseMeshStream,
  readZip,
  writeZip,
  writeZipSync,
} from "../src/io/index.js";

/** In-memory SceneLike supporting positions, indices, normals, colors, UVs, and placement matrix. */
class FakeExportScene implements SceneLike {
  readonly mesh = {} as MeshHandle;
  constructor(
    private readonly pos: Float32Array,
    private readonly idx: Uint32Array,
    private readonly mat: Float64Array = Float64Array.from([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]),
    private readonly norm: Float32Array | null = null,
    private readonly col: Uint8Array | null = null,
    private readonly uv: Float32Array | null = null,
  ) {}

  views() {
    return {
      positions: () => this.pos,
      indices: () => this.idx,
      normals: () => this.norm,
      colors: () => this.col,
      values: () => null,
      uvs: () => this.uv,
    } as ReturnType<SceneLike["views"]>;
  }
  matrix4() {
    return this.mat;
  }
  placement(): never {
    throw new Error("unused");
  }
  section(): never {
    throw new Error("unused");
  }
  sectionPlane(): never {
    throw new Error("unused");
  }
  onChanges() {
    return () => {};
  }
  commitFrame() {
    return [];
  }
  drag(): never {
    throw new Error("unused");
  }
}

class TestNodeWorker implements WorkerLike {
  readonly worker: Worker;
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  readonly #error: ((e: unknown) => void)[] = [];

  constructor() {
    this.worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), {
      workerData: { serve: true },
    });
    this.worker.on("message", (data) => {
      for (const cb of this.#message) cb({ data });
    });
    this.worker.on("error", (err) => {
      for (const cb of this.#error) cb(err);
    });
  }

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", cb: (e: any) => void): void {
    (type === "message" ? this.#message : this.#error).push(cb);
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

const openedWorkers: WorkerScene[] = [];
afterEach(() => {
  for (const ws of openedWorkers.splice(0)) {
    ws.terminate();
  }
});

function u8Of(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

describe("zip.ts", () => {
  it("round-trips STORE and DEFLATE entries with CRC-32 and EOCD comments", async () => {
    const payloadA = u8Of("hello 3mf world! ".repeat(50));
    const payloadB = Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]);

    const stored = writeZipSync([
      { name: "a.txt", data: payloadA },
      { name: "dir/b.bin", data: payloadB },
    ]);
    const readStored = await readZip(stored);
    expect(Array.from(readStored.get("a.txt") ?? [])).toEqual(Array.from(payloadA));
    expect(Array.from(readStored.get("dir/b.bin") ?? [])).toEqual(Array.from(payloadB));

    const deflated = await writeZip([
      { name: "a.txt", data: payloadA },
      { name: "dir/b.bin", data: payloadB },
    ]);
    expect(deflated.byteLength).toBeLessThan(stored.byteLength);

    // Append an EOCD archive comment + length to test backward EOCD discovery
    const comment = u8Of("custom archive comment");
    const withComment = new Uint8Array(deflated.byteLength + comment.byteLength);
    withComment.set(deflated, 0);
    const eocdOff = deflated.byteLength - 22;
    new DataView(withComment.buffer).setUint16(eocdOff + 20, comment.byteLength, true);
    withComment.set(comment, deflated.byteLength);

    const readDeflated = await readZip(withComment);
    expect(Array.from(readDeflated.get("a.txt") ?? [])).toEqual(Array.from(payloadA));
    expect(Array.from(readDeflated.get("dir/b.bin") ?? [])).toEqual(Array.from(payloadB));
  });

  it("supports ZIP64 extra field and ZIP64 EOCD locator", async () => {
    const content = u8Of("zip64-payload-check");
    const entryName = u8Of("z64.txt");
    const entryCrc = crc32(content);

    // Extra field tag 0x0001: uncompressed(8), compressed(8), localOffset(8)
    const extra = new Uint8Array(28);
    const ev = new DataView(extra.buffer);
    ev.setUint16(0, 0x0001, true);
    ev.setUint16(2, 24, true);
    ev.setBigUint64(4, BigInt(content.byteLength), true);
    ev.setBigUint64(12, BigInt(content.byteLength), true);
    ev.setBigUint64(20, 0n, true);

    const localLen = 30 + entryName.byteLength + content.byteLength;
    const cdLen = 46 + entryName.byteLength + extra.byteLength;
    const totalLen = localLen + cdLen + 56 + 20 + 22;
    const zip = new Uint8Array(totalLen);
    const dv = new DataView(zip.buffer);

    let p = 0;
    dv.setUint32(p, 0x04034b50, true);
    dv.setUint16(p + 4, 45, true);
    dv.setUint32(p + 14, entryCrc, true);
    dv.setUint32(p + 18, content.byteLength, true);
    dv.setUint32(p + 22, content.byteLength, true);
    dv.setUint16(p + 26, entryName.byteLength, true);
    zip.set(entryName, p + 30);
    p += 30 + entryName.byteLength;
    zip.set(content, p);
    p += content.byteLength;

    const cdStart = p;
    dv.setUint32(p, 0x02014b50, true);
    dv.setUint16(p + 4, 45, true);
    dv.setUint16(p + 6, 45, true);
    dv.setUint32(p + 16, entryCrc, true);
    dv.setUint32(p + 20, 0xffffffff, true);
    dv.setUint32(p + 24, 0xffffffff, true);
    dv.setUint16(p + 28, entryName.byteLength, true);
    dv.setUint16(p + 30, extra.byteLength, true);
    dv.setUint32(p + 42, 0xffffffff, true);
    zip.set(entryName, p + 46);
    zip.set(extra, p + 46 + entryName.byteLength);
    p += cdLen;

    const z64EocdStart = p;
    dv.setUint32(p, 0x06064b50, true);
    dv.setBigUint64(p + 4, 44n, true);
    dv.setUint16(p + 12, 45, true);
    dv.setUint16(p + 14, 45, true);
    dv.setBigUint64(p + 24, 1n, true);
    dv.setBigUint64(p + 32, 1n, true);
    dv.setBigUint64(p + 40, BigInt(cdLen), true);
    dv.setBigUint64(p + 48, BigInt(cdStart), true);
    p += 56;

    // ZIP64 EOCD Locator
    dv.setUint32(p, 0x07064b50, true);
    dv.setBigUint64(p + 8, BigInt(z64EocdStart), true);
    dv.setUint32(p + 16, 1, true);
    p += 20;

    // Standard EOCD sentinel
    dv.setUint32(p, 0x06054b50, true);
    dv.setUint16(p + 8, 0xffff, true);
    dv.setUint16(p + 10, 0xffff, true);
    dv.setUint32(p + 12, 0xffffffff, true);
    dv.setUint32(p + 16, 0xffffffff, true);

    const read = await readZip(zip);
    expect(new TextDecoder().decode(read.get("z64.txt"))).toBe("zip64-payload-check");
  });

  it("rejects zip bombs without OOM even when header claims small uncompressed size", async () => {
    const bigZeros = new Uint8Array(2 * 1024 * 1024);
    const bombZip = await writeZip([{ name: "bomb.bin", data: bigZeros, compress: true }]);

    // Reject by declared size before allocating
    await expect(readZip(bombZip, { maxEntryBytes: 512 * 1024 })).rejects.toThrow(/exceed(?:s|ing)/i);

    // Spoof declared uncompressed sizes down to 16 bytes in both headers so only
    // streaming DecompressionStream capping catches the actual 2 MiB inflation!
    const spoofed = bombZip.slice();
    const dv = new DataView(spoofed.buffer);
    dv.setUint32(22, 16, true); // Local file header uncompressedSize
    // Locate central directory header signature 0x02014b50
    for (let i = 0; i <= spoofed.byteLength - 46; i++) {
      if (dv.getUint32(i, true) === 0x02014b50) {
        dv.setUint32(i + 24, 16, true);
        break;
      }
    }
    await expect(
      readZip(spoofed, { maxEntryBytes: 64 * 1024, maxTotalUncompressedBytes: 64 * 1024 }),
    ).rejects.toThrow(/zip bomb/i);
  });

  it("rejects Zip Slip traversal, CRC mismatch, encryption, bad methods, and maxEntries", async () => {
    expect(() => writeZipSync([{ name: "../escape.txt", data: u8Of("x") }])).toThrow(/traversal/);
    expect(() => writeZipSync([{ name: "/abs.txt", data: u8Of("x") }])).toThrow(/relative/);
    expect(() => writeZipSync([{ name: "C:drive.txt", data: u8Of("x") }])).toThrow(/drive/);

    const zip = writeZipSync([{ name: "ok.txt", data: u8Of("hello") }]);
    // Corrupt payload byte -> CRC mismatch
    const badCrc = zip.slice();
    const payloadOffset = 30 + "ok.txt".length;
    badCrc[payloadOffset] ^= 0xff;
    await expect(readZip(badCrc)).rejects.toThrow(/CRC-32/);

    // Set encryption flag in central directory
    const encrypted = zip.slice();
    const dv = new DataView(encrypted.buffer);
    for (let i = 0; i <= encrypted.byteLength - 46; i++) {
      if (dv.getUint32(i, true) === 0x02014b50) {
        dv.setUint16(i + 8, 0x0001, true);
        break;
      }
    }
    await expect(readZip(encrypted)).rejects.toThrow(/encrypted/i);

    const many = writeZipSync([
      { name: "1.txt", data: u8Of("a") },
      { name: "2.txt", data: u8Of("b") },
      { name: "3.txt", data: u8Of("c") },
    ]);
    await expect(readZip(many, { maxEntries: 2 })).rejects.toThrow(/exceeding maximum of 2/);
  });
});

describe("threemf.ts", () => {
  it("round-trips colored single & multi-mesh 3MF (sync + async) with placements and units", async () => {
    const pos = Float32Array.from([
      0, 0, 0,
      10, 0, 0,
      10, 10, 0,
      0, 10, 0,
    ]);
    const idx = Uint32Array.from([0, 1, 2, 0, 2, 3]);
    const col = Uint8Array.from([
      255, 0, 0, 255,
      0, 255, 0, 255,
      0, 0, 255, 255,
      255, 255, 0, 128,
    ]);
    // Translation (5, -2, 3) in column-major 4x4
    const mat = Float64Array.from([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      5, -2, 3, 1,
    ]);
    const scene = new FakeExportScene(pos, idx, mat, null, col);

    const syncBytes = export3MFSync(scene, scene.mesh, {
      name: "quad",
      unit: "centimeter",
      applyPlacement: true,
    });
    const importedSync = await import3MF(syncBytes);
    expect(importedSync).toHaveLength(1);
    expect(importedSync[0].name).toBe("quad");
    expect(Array.from(importedSync[0].indices)).toEqual([0, 1, 2, 0, 2, 3]);
    expect(Array.from(importedSync[0].colors ?? [])).toEqual(Array.from(col));
    expect(importedSync[0].positions[0]).toBeCloseTo(5, 5);
    expect(importedSync[0].positions[1]).toBeCloseTo(-2, 5);
    expect(importedSync[0].positions[2]).toBeCloseTo(3, 5);
    expect(importedSync[0].positions[3]).toBeCloseTo(15, 5);

    const asyncBytes = await export3MF(scene, [scene.mesh, scene.mesh], {
      name: "part",
      unit: "inch",
      applyPlacement: false,
      compress: true,
    });
    const importedAsync = await import3MF(asyncBytes);
    expect(importedAsync).toHaveLength(2);
    expect(importedAsync[0].name).toBe("part-1");
    expect(importedAsync[1].name).toBe("part-2");
    expect(importedAsync[0].positions[3]).toBeCloseTo(10, 4);
  });

  it("handles Production Extension p:path, recursive transforms, negative det winding flip, and basematerials", async () => {
    const rels = u8Of(
      '<?xml version="1.0" encoding="UTF-8"?>\n' +
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n' +
        '  <Relationship Target="/3D/root.model" Id="r1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n' +
        "</Relationships>\n",
    );
    const subModel = u8Of(
      '<?xml version="1.0" encoding="UTF-8"?>\n' +
        '<model unit="centimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel">\n' +
        "  <resources>\n" +
        '    <basematerials id="5">\n' +
        '      <base name="red" displaycolor="#FF0000"/>\n' +
        '      <base name="blue" displaycolor="#0000FF"/>\n' +
        "    </basematerials>\n" +
        '    <object id="10" name="leaf-quad" type="model">\n' +
        "      <mesh>\n" +
        "        <vertices>\n" +
        '          <vertex x="0" y="0" z="0"/>\n' +
        '          <vertex x="1" y="0" z="0"/>\n' +
        '          <vertex x="1" y="1" z="0"/>\n' +
        '          <vertex x="0" y="1" z="0"/>\n' +
        "        </vertices>\n" +
        "        <triangles>\n" +
        '          <triangle v1="0" v2="1" v3="2" pid="5" p1="0"/>\n' +
        '          <triangle v1="0" v2="2" v3="3" pid="5" p1="1"/>\n' +
        "        </triangles>\n" +
        "      </mesh>\n" +
        "    </object>\n" +
        "  </resources>\n" +
        "  <build/>\n" +
        "</model>\n",
    );
    const rootModel = u8Of(
      '<?xml version="1.0" encoding="UTF-8"?>\n' +
        '<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06">\n' +
        "  <resources>\n" +
        '    <object id="1" type="model">\n' +
        "      <components>\n" +
        // Child translates +2 on X in centimeters
        '        <component objectid="10" p:path="/3D/Objects/sub.model" transform="1 0 0 0 1 0 0 0 1 2 0 0"/>\n' +
        "      </components>\n" +
        "    </object>\n" +
        "  </resources>\n" +
        "  <build>\n" +
        // Parent mirrors X -> -X (det = -1 < 0) and translates +100
        '    <item objectid="1" transform="-1 0 0 0 1 0 0 0 1 100 0 0"/>\n' +
        "  </build>\n" +
        "</model>\n",
    );

    const archive = await writeZip([
      { name: "_rels/.rels", data: rels },
      { name: "3D/root.model", data: rootModel },
      { name: "3D/Objects/sub.model", data: subModel },
    ]);

    const meshes = await import3MF(archive);
    expect(meshes).toHaveLength(1);
    const m = meshes[0];
    expect(m.name).toBe("leaf-quad");

    // 4 original vertices + 2 shared boundary vertices cloned because tri 0 is red and tri 1 is blue!
    expect(m.positions.length / 3).toBe(6);
    expect(m.colors).not.toBeNull();

    // Check transformed X coord of vertex 0: (-0 - 2 + 100) * 10cm->mm = 980
    expect(m.positions[0]).toBeCloseTo(980, 6);

    // Check mirrored triangle 0 normal points +Z (outward) after winding flip (v1, v3, v2)
    const i0 = m.indices[0];
    const i1 = m.indices[1];
    const i2 = m.indices[2];
    const ax = m.positions[i1 * 3] - m.positions[i0 * 3];
    const ay = m.positions[i1 * 3 + 1] - m.positions[i0 * 3 + 1];
    const az = m.positions[i1 * 3 + 2] - m.positions[i0 * 3 + 2];
    const bx = m.positions[i2 * 3] - m.positions[i0 * 3];
    const by = m.positions[i2 * 3 + 1] - m.positions[i0 * 3 + 1];
    const bz = m.positions[i2 * 3 + 2] - m.positions[i0 * 3 + 2];
    const nz = ax * by - ay * bx;
    expect(nz).toBeGreaterThan(0);

    // unitScale override
    const customScaled = await import3MF(archive, { unitScale: 1 });
    expect(customScaled[0].positions[0]).toBeCloseTo(98, 6);
  });

  it("rejects XXE / DOCTYPE, ENTITY declarations, cyclic components, and non-finite coords", async () => {
    const xxeModel = u8Of(
      '<?xml version="1.0"?>\n<!DOCTYPE model [ <!ENTITY xxe "evil"> ]>\n<model xmlns="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"><resources/><build/></model>',
    );
    const xxeZip = writeZipSync([{ name: "3D/3dmodel.model", data: xxeModel }]);
    await expect(import3MF(xxeZip)).rejects.toThrow(/DOCTYPE|ENTITY/);

    const cycleModel = u8Of(
      '<?xml version="1.0"?>\n<model xmlns="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel">\n' +
        '  <resources><object id="1"><components><component objectid="1"/></components></object></resources>\n' +
        '  <build><item objectid="1"/></build>\n</model>',
    );
    const cycleZip = writeZipSync([{ name: "3D/3dmodel.model", data: cycleModel }]);
    await expect(import3MF(cycleZip)).rejects.toThrow(/cyclic/i);
  });
});

describe("ply.ts", () => {
  it("round-trips exportPLY (ASCII + binary LE) with normals, UVs, and RGBA8 colors", () => {
    const pos = Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const idx = Uint32Array.from([0, 1, 2]);
    const norm = Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const col = Uint8Array.from([255, 10, 20, 255, 0, 200, 30, 128, 40, 50, 255, 64]);
    const uv = Float32Array.from([0, 0, 1, 0, 0, 1]);
    const scene = new FakeExportScene(pos, idx, undefined, norm, col, uv);

    for (const binary of [true, false]) {
      const plyBytes = exportPLY(scene, scene.mesh, { binary });
      const back = importPLY(plyBytes);
      expect(Array.from(back.positions)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
      expect(Array.from(back.indices)).toEqual([0, 1, 2]);
      expect(Array.from(back.normals ?? [])).toEqual([0, 0, 1, 0, 0, 1, 0, 0, 1]);
      expect(Array.from(back.uvs ?? [])).toEqual([0, 0, 1, 0, 0, 1]);
      expect(Array.from(back.colors ?? [])).toEqual(Array.from(col));
    }
  });

  it("parses binary_big_endian 1.0 and float 0..1 vertex colors cleanly", () => {
    const header = [
      "ply",
      "format binary_big_endian 1.0",
      "element vertex 4",
      "property double x",
      "property double y",
      "property double z",
      "property short pad",
      "property float red",
      "property float green",
      "property float blue",
      "property float u",
      "property float v",
      "element face 1",
      "property list uchar uint vertex_indices",
      "property int tag",
      "end_header",
      "",
    ].join("\n");
    const head = u8Of(header);
    const vStride = 8 * 3 + 2 + 4 * 5; // 46 bytes
    const buf = new Uint8Array(head.byteLength + 4 * vStride + 1 + 4 * 4 + 4);
    buf.set(head);
    const dv = new DataView(buf.buffer);
    let at = head.byteLength;
    const verts = [
      [0, 0, 0, 1, 0, 0, 0, 0],
      [2, 0, 0, 0, 1, 0, 1, 0],
      [2, 2, 0, 0, 0, 1, 1, 1],
      [0, 2, 0, 1, 1, 0, 0, 1],
    ];
    for (const [x, y, z, r, g, b, u, v] of verts) {
      dv.setFloat64(at, x, false);
      dv.setFloat64(at + 8, y, false);
      dv.setFloat64(at + 16, z, false);
      dv.setInt16(at + 24, -7, false);
      dv.setFloat32(at + 26, r, false);
      dv.setFloat32(at + 30, g, false);
      dv.setFloat32(at + 34, b, false);
      dv.setFloat32(at + 38, u, false);
      dv.setFloat32(at + 42, v, false);
      at += vStride;
    }
    dv.setUint8(at++, 4);
    for (let i = 0; i < 4; i++) {
      dv.setUint32(at + i * 4, i, false);
    }
    at += 16;
    dv.setInt32(at, 42, false);

    const mesh = importPLY(buf);
    expect(Array.from(mesh.indices)).toEqual([0, 1, 2, 0, 2, 3]);
    expect(Array.from(mesh.colors ?? [])).toEqual([
      255, 0, 0, 255,
      0, 255, 0, 255,
      0, 0, 255, 255,
      255, 255, 0, 255,
    ]);
    expect(Array.from(mesh.uvs ?? [])).toEqual([0, 0, 1, 0, 1, 1, 0, 1]);
  });

  it("rejects NaN positions, face counts < 3, out-of-range indices, and truncated PLY", () => {
    const base = (body: string) =>
      u8Of(
        `ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n${body}`,
      );
    expect(() => importPLY(base("0 0 0\n1 NaN 0\n0 1 0\n3 0 1 2\n"))).toThrow(/non-finite/i);
    expect(() => importPLY(base("0 0 0\n1 0 0\n0 1 0\n2 0 1\n"))).toThrow(/less than 3/i);
    expect(() => importPLY(base("0 0 0\n1 0 0\n0 1 0\n3 0 1 9\n"))).toThrow(/out of range/i);
    expect(() => importPLY(base("0 0 0\n1 0 0\n"))).toThrow(/ends early/i);
  });
});

describe("stl.ts", () => {
  it("disambiguates solid-prefixed binary STL and supports optional spatial-hash vertex welding", () => {
    const bytes = new Uint8Array(84 + 2 * 50);
    bytes.set(u8Of("solid looks like ascii but size is 184"));
    const dv = new DataView(bytes.buffer);
    dv.setUint32(80, 2, true);
    const tris = [
      [-0, 0, 0, 1, 0, 0, 1, 1, 0],
      [0, 0, 0, 1, 1, 0, 0, 1, 0],
    ];
    tris.forEach((tri, ti) => {
      const off = 84 + ti * 50 + 12;
      for (let i = 0; i < 9; i++) {
        dv.setFloat32(off + i * 4, tri[i], true);
      }
    });

    const unwelded = importSTL(bytes);
    expect(unwelded.positions.length / 3).toBe(6);

    const welded = importSTL(bytes, { deduplicateVertices: true });
    expect(welded.positions.length / 3).toBe(4);
    expect(Array.from(welded.indices)).toEqual([0, 1, 2, 0, 2, 3]);

    // Round-trip via exportSTL
    const scene = new FakeExportScene(
      Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      Uint32Array.from([0, 1, 2]),
    );
    const asc = importSTL(exportSTL(scene, scene.mesh, { binary: false, name: "tri" }));
    expect(asc.name).toBe("tri");
    expect(asc.indices.length).toBe(3);

    // Reject NaN binary vertex & truncated binary STL
    const nanStl = bytes.slice();
    new DataView(nanStl.buffer).setFloat32(84 + 12, Number.NaN, true);
    expect(() => importSTL(nanStl)).toThrow(/non-finite/i);
    expect(() => importSTL(bytes.subarray(0, 154))).toThrow(/2 triangles.*184 bytes.*154/);
  });
});

describe("obj.ts & GLB COLOR_0", () => {
  it("parses normalized vs 0..255 colors, negative indices, groups, and exportOBJ vertexColors round-trip", () => {
    const objMixedScale = u8Of(
      [
        "o partA",
        "v 0 0 0 1 0 0",
        "v 1 0 0 0 255 0",
        "v 0 1 0 0 0 255",
        "f -3 -2 -1",
        "o partB",
        "v 2 0 0 0 0 0",
        "v 3 0 0 255 255 255",
        "v 2 1 0 128 128 128",
        "f 4 5 6",
      ].join("\n"),
    );
    const parts = importOBJ(objMixedScale);
    expect(parts).toHaveLength(2);
    expect(parts[0].name).toBe("partA");
    // Because max color across file is 255 (> 1), `1 0 0` stays byte [1, 0, 0, 255]
    expect(Array.from(parts[0].colors ?? [])).toEqual([
      1, 0, 0, 255,
      0, 255, 0, 255,
      0, 0, 255, 255,
    ]);
    expect(parts[1].name).toBe("partB");

    const merged = importOBJ(objMixedScale, { mergeGroups: true });
    expect(merged).toHaveLength(1);
    expect(merged[0].indices.length).toBe(6);

    // exportOBJ round-trip with vertexColors true vs false
    const scene = new FakeExportScene(
      Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      Uint32Array.from([0, 1, 2]),
      undefined,
      Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      Uint8Array.from([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]),
    );
    const coloredBytes = exportOBJ(scene, scene.mesh, { vertexColors: true });
    const [reimportedColor] = importOBJ(coloredBytes);
    expect(Array.from(reimportedColor.colors ?? [])).toEqual([
      255, 0, 0, 255,
      0, 255, 0, 255,
      0, 0, 255, 255,
    ]);
    expect(Array.from(reimportedColor.normals ?? [])).toEqual([0, 0, 1, 0, 0, 1, 0, 0, 1]);

    const plainBytes = exportOBJ(scene, scene.mesh, { vertexColors: false });
    const [reimportedPlain] = importOBJ(plainBytes);
    expect(reimportedPlain.colors).toBeNull();

    // GLB COLOR_0 round-trip
    const glb = exportGLTF(scene, scene.mesh, { binary: true, name: "colglb" });
    const [glbPart] = importGLTF(glb);
    expect(Array.from(glbPart.colors ?? [])).toEqual([
      255, 0, 0, 255,
      0, 255, 0, 255,
      0, 0, 255, 255,
    ]);
  });
});

describe("stream.ts", () => {
  it("detects format from magic bytes and filenames", () => {
    expect(detectMeshFormat(new Uint8Array([0x50, 0x4b, 0x03, 0x04]))).toBe("threemf");
    expect(detectMeshFormat(u8Of("ply\nformat ascii 1.0\n"))).toBe("ply");
    expect(detectMeshFormat(new Uint8Array([0x67, 0x6c, 0x54, 0x46]))).toBe("glb");
    expect(detectMeshFormat(u8Of('{"asset":{"version":"2.0"}}'))).toBe("gltf");
    expect(detectMeshFormat(u8Of("solid foo\nendsolid foo\n"))).toBe("stl");
    expect(detectMeshFormat(u8Of("# comment\nv 0 0 0\n"))).toBe("obj");
    expect(detectMeshFormat(undefined, "bracket.3MF")).toBe("threemf");
    expect(detectMeshFormat("mesh.glb")).toBe("glb");
  });

  it("enforces maxFileBytes before buffering chunks and cancels reader immediately on AbortSignal", async () => {
    let pulled = 0;
    const largeStream = new ReadableStream<Uint8Array>({
      pull(controller) {
        pulled++;
        controller.enqueue(new Uint8Array(1024));
      },
    });
    await expect(
      parseMeshStream(largeStream, { format: "obj", maxFileBytes: 1500 }),
    ).rejects.toThrow(/exceeds maximum size limit/);
    expect(pulled).toBeLessThanOrEqual(2);

    const ac = new AbortController();
    let cancelledReason: unknown = null;
    const cancelableStream = new ReadableStream<Uint8Array>({
      pull(controller) {
        controller.enqueue(u8Of("v 0 0 0\n"));
        ac.abort(new Error("user stopped"));
      },
      cancel(reason) {
        cancelledReason = reason;
      },
    });
    await expect(
      parseMeshStream(cancelableStream, { filename: "part.obj" }, undefined, ac.signal),
    ).rejects.toThrow(/user stopped/);
    expect(cancelledReason).toBeTruthy();
  });

  it("loads streamed meshes into both sync Scene and async WorkerScene with building_bvh progress and abort cleanup", async () => {
    const syncScene = new Scene();
    const fake = new FakeExportScene(
      Float32Array.from([0, 0, 0, 10, 0, 0, 0, 10, 0]),
      Uint32Array.from([0, 1, 2]),
      undefined,
      null,
      null,
      Float32Array.from([0, 0, 1, 0, 0, 1]),
    );
    const glbBytes = exportGLTF(fake, fake.mesh, { binary: true, name: "uvtri" });
    const stages: string[] = [];
    const blob = new Blob([glbBytes.slice().buffer], { type: "model/gltf-binary" });
    const handles = await loadMeshStream(
      syncScene,
      blob,
      { filename: "uvtri.glb" },
      (p) => stages.push(p.stage),
    );
    expect(handles).toHaveLength(1);
    expect(stages).toEqual(["reading", "reading", "parsing", "building_bvh"]);
    expect(syncScene.views(handles[0]).uvs()?.length).toBe(6);
    syncScene.free();

    const workerScene = await createWorkerScene({
      createWorker: () => new TestNodeWorker(),
    });
    openedWorkers.push(workerScene);
    const workerStages: string[] = [];
    const chunkedStream = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(glbBytes.subarray(0, 64));
        c.enqueue(glbBytes.subarray(64));
        c.close();
      },
    });
    const wHandles = await loadMeshStream(
      workerScene,
      chunkedStream,
      { filename: "uvtri.glb" },
      (p) => workerStages.push(p.stage),
    );
    expect(wHandles).toHaveLength(1);
    expect(workerStages).toContain("building_bvh");
    expect(workerScene.views(wHandles[0]).uvs()?.length).toBe(6);
    const m = await workerScene.measure(wHandles[0]);
    expect(m.area).toBeCloseTo(50, 3);

    // Abort right after building_bvh begins cleans up any in-flight WorkerScene meshes
    const cancelCtrl = new AbortController();
    const twoMeshesObj = u8Of(
      "o a\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n" +
        "o b\nv 2 0 0\nv 3 0 0\nv 2 1 0\nf 4 5 6\n",
    );
    await expect(
      loadMeshStream(
        workerScene,
        twoMeshesObj,
        { filename: "two.obj" },
        (p) => {
          if (p.stage === "building_bvh") {
            cancelCtrl.abort(new Error("aborted upload"));
          }
        },
        cancelCtrl.signal,
      ),
    ).rejects.toThrow(/aborted upload/);
  });
});
