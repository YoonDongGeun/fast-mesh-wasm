// The example's mesh file parsers, on small hand-made fixtures: STL (binary
// and ASCII), OBJ (polygons, index forms) and PLY (ASCII and binary
// little-endian), plus the dispatch by file name.
import { describe, expect, it } from "vitest";
import type { MeshData } from "../src/load/mesh-data";
import { parseObj } from "../src/load/obj";
import { parseMeshFile } from "../src/load/parse";
import { parsePly } from "../src/load/ply";
import { parseStl } from "../src/load/stl";

const text = (s: string): ArrayBuffer => new TextEncoder().encode(s).buffer as ArrayBuffer;

/** The triangles of `m` as corner positions, so tests do not depend on vertex order. */
function corners(m: MeshData): number[][][] {
  const out: number[][][] = [];
  for (let t = 0; t < m.indices.length; t += 3) {
    out.push([0, 1, 2].map((k) => Array.from(m.positions.subarray(3 * m.indices[t + k], 3 * m.indices[t + k] + 3))));
  }
  return out;
}

/** A binary STL of `tris` (each three corners), with an optional 80-byte header text. */
function binaryStl(tris: number[][][], header = "binary"): ArrayBuffer {
  const buf = new ArrayBuffer(84 + 50 * tris.length);
  const view = new DataView(buf);
  new Uint8Array(buf, 0, 80).set(new TextEncoder().encode(header).subarray(0, 80));
  view.setUint32(80, tris.length, true);
  tris.forEach((tri, t) => {
    tri.forEach((p, k) => p.forEach((x, c) => view.setFloat32(84 + 50 * t + 12 + 12 * k + 4 * c, x, true)));
  });
  return buf;
}

const QUAD = [
  [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
  ],
  [
    [0, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
  ],
];

describe("parseStl", () => {
  it("reads binary STL and merges bit-identical corners", () => {
    const m = parseStl(binaryStl(QUAD));
    expect(m.indices.length).toBe(6);
    expect(m.positions.length).toBe(4 * 3);
    expect(corners(m)).toEqual(QUAD);
  });

  it("keeps near-duplicate corners apart (welding is the Weld button's job)", () => {
    const tris = [
      [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
      ],
      [
        [1.0001, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
      ],
    ];
    const m = parseStl(binaryStl(tris));
    expect(m.positions.length).toBe(5 * 3);
  });

  it("reads a binary STL whose header starts with 'solid'", () => {
    const m = parseStl(binaryStl(QUAD, "solid but binary"));
    expect(m.indices.length).toBe(6);
  });

  it("reads ASCII STL", () => {
    const src = `solid quad
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 1 1 0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 1 0
      vertex 0 1.0e0 0
    endloop
  endfacet
endsolid quad
`;
    const m = parseStl(text(src));
    expect(m.positions.length).toBe(4 * 3);
    expect(corners(m)).toEqual(QUAD);
  });

  it("rejects what is not STL", () => {
    expect(() => parseStl(text("hello"))).toThrow(/STL/);
    expect(() => parseStl(new ArrayBuffer(90))).toThrow(/STL/);
  });
});

describe("parseObj", () => {
  it("fan-triangulates polygons and reads every index form", () => {
    const src = `# a quad and a pentagon
o thing
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0 0
v 3 0 0
v 3 1 0
vt 0 0
vn 0 0 1
f 1 2 3 4
f 2/1/1 5/1/1 6//1 7 3
g other
f -7 -6 -5
`;
    const m = parseObj(text(src));
    expect(m.positions.length).toBe(7 * 3);
    // Quad -> 2, pentagon -> 3, triangle -> 1.
    expect(Array.from(m.indices)).toEqual([0, 1, 2, 0, 2, 3, 1, 4, 5, 1, 5, 6, 1, 6, 2, 0, 1, 2]);
  });

  it("handles CRLF line ends and a w coordinate", () => {
    const m = parseObj(text("v 0 0 0 1\r\nv 1 0 0 1\r\nv 0 1 0 1\r\nf 1 2 3\r\n"));
    expect(Array.from(m.positions)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    expect(Array.from(m.indices)).toEqual([0, 1, 2]);
  });

  it("rejects out-of-range indices and files without faces", () => {
    expect(() => parseObj(text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n"))).toThrow(/index/);
    expect(() => parseObj(text("v 0 0 0\n"))).toThrow(/no faces/);
  });
});

const PLY_ASCII = `ply
format ascii 1.0
comment a quad, with a colour per vertex and an extra element
element vertex 4
property float x
property float y
property float z
property uchar red
element face 1
property list uchar int vertex_indices
element edge 1
property int vertex1
property int vertex2
end_header
0 0 0 255
1 0 0 255
1 1 0 255
0 1 0 255
4 0 1 2 3
0 1
`;

/** A binary little-endian PLY: 4 vertices (double x, float y z, uchar flag), a triangle and a quad, then an element with a list. */
function binaryPly(format = "binary_little_endian"): ArrayBuffer {
  const header = `ply
format ${format} 1.0
element vertex 4
property double x
property float y
property float z
property uchar flag
element face 2
property list uchar uint vertex_index
element extra 1
property list uchar short values
property ushort tail
end_header
`;
  const head = new TextEncoder().encode(header);
  const size = head.length + 4 * (8 + 4 + 4 + 1) + (1 + 3 * 4) + (1 + 4 * 4) + (1 + 2 * 2 + 2);
  const buf = new ArrayBuffer(size);
  new Uint8Array(buf).set(head);
  const v = new DataView(buf);
  let at = head.length;
  const verts = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
  ];
  for (const [x, y, z] of verts) {
    v.setFloat64(at, x, true);
    v.setFloat32(at + 8, y, true);
    v.setFloat32(at + 12, z, true);
    v.setUint8(at + 16, 7);
    at += 17;
  }
  v.setUint8(at++, 3);
  for (const i of [0, 1, 2]) {
    v.setUint32(at, i, true);
    at += 4;
  }
  v.setUint8(at++, 4);
  for (const i of [0, 1, 2, 3]) {
    v.setUint32(at, i, true);
    at += 4;
  }
  v.setUint8(at++, 2);
  v.setInt16(at, -1, true);
  v.setInt16(at + 2, 5, true);
  v.setUint16(at + 4, 9, true);
  return buf;
}

describe("parsePly", () => {
  it("reads ASCII PLY, fan-triangulating faces and skipping other properties and elements", () => {
    const m = parsePly(text(PLY_ASCII));
    expect(Array.from(m.positions)).toEqual([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
    expect(Array.from(m.indices)).toEqual([0, 1, 2, 0, 2, 3]);
  });

  it("reads binary little-endian PLY with mixed property types", () => {
    const m = parsePly(binaryPly());
    expect(Array.from(m.positions)).toEqual([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
    expect(Array.from(m.indices)).toEqual([0, 1, 2, 0, 1, 2, 0, 2, 3]);
  });

  it("rejects big-endian PLY and non-PLY input", () => {
    expect(() => parsePly(binaryPly("binary_big_endian"))).toThrow(/big-endian/);
    expect(() => parsePly(text("solid x\n"))).toThrow(/PLY/);
  });

  it("rejects a face index past the vertices", () => {
    expect(() => parsePly(text(PLY_ASCII.replace("4 0 1 2 3", "4 0 1 2 9")))).toThrow(/index/);
  });
});

describe("parseMeshFile", () => {
  it("picks the parser by extension, in any case", () => {
    expect(parseMeshFile("quad.STL", binaryStl(QUAD)).indices.length).toBe(6);
    expect(parseMeshFile("quad.ply", text(PLY_ASCII)).indices.length).toBe(6);
    expect(parseMeshFile("t.Obj", text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")).indices.length).toBe(3);
  });

  it("rejects unknown extensions", () => {
    expect(() => parseMeshFile("mesh.glb", new ArrayBuffer(4))).toThrow(/\.glb/);
  });
});
