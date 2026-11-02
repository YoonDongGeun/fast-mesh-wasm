// The mesh exporters over a hand-written fake SceneLike: STL (binary and
// ASCII), OBJ, PLY (binary and ASCII) and glTF (GLB and JSON), including
// dead-slot compaction, missing normals/colors, and placement handling.
import { describe, expect, it } from "vitest";
import { MeshHandle } from "../src/handles.js";
import {
  exportGLTF,
  exportOBJ,
  exportPLY,
  exportSTL,
} from "../src/io/index.js";
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

describe("io export", () => {
  it("stl_binary_round_trip", () => {
    const scene = new FakeScene();
    const bytes = exportSTL(scene, scene.mesh);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    expect(view.getUint32(80, true)).toBe(2);
    expect(bytes.length).toBe(84 + 50 * 2);
    // Facet normals face +Z (per-vertex normals agree).
    expect(view.getFloat32(84, true)).toBeCloseTo(0, 5);
    expect(view.getFloat32(84 + 8, true)).toBeCloseTo(1, 5);
    const positions: number[] = [];
    const indices: number[] = [];
    for (let t = 0; t < 2; t++) {
      for (let k = 0; k < 3; k++) {
        const at = 84 + 50 * t + 12 + 12 * k;
        positions.push(view.getFloat32(at, true), view.getFloat32(at + 4, true), view.getFloat32(at + 8, true));
        indices.push(3 * t + k);
      }
    }
    expect(triangleSet(positions, indices)).toEqual(EXPECTED);
  });

  it("stl_binary_without_normals_uses_facet_normals", () => {
    const scene = new FakeScene(QUAD, QUAD_TRIS, null, null);
    const bytes = exportSTL(scene, scene.mesh);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    expect(view.getUint32(80, true)).toBe(2);
    expect(view.getFloat32(84 + 8, true)).toBeCloseTo(1, 5);
  });

  it("stl_ascii_names_solids", () => {
    const scene = new FakeScene();
    const text = new TextDecoder().decode(exportSTL(scene, scene.mesh, { binary: false, name: "quad" }));
    expect(text.startsWith("solid quad\n")).toBe(true);
    expect(text.endsWith("endsolid quad\n")).toBe(true);
    expect(text.match(/facet normal/g)?.length).toBe(2);
  });

  it("obj_round_trip_with_normals", () => {
    const scene = new FakeScene();
    const text = new TextDecoder().decode(exportOBJ(scene, scene.mesh));
    const verts = text.split("\n").filter((l) => l.startsWith("v "));
    const normals = text.split("\n").filter((l) => l.startsWith("vn "));
    const faces = text.split("\n").filter((l) => l.startsWith("f "));
    // Dead vertex slot 4 is gone; colors ride as `r g b` suffixes.
    expect(verts.length).toBe(4);
    expect(normals.length).toBe(4);
    expect(faces.length).toBe(2);
    expect(faces[0]).toMatch(/^f \d+\/\/\d+ \d+\/\/\d+ \d+\/\/\d+$/);
    expect(verts[0].split(" ").length).toBe(7);
    const positions: number[] = [];
    for (const v of verts) {
      const [, x, y, z] = v.split(" ");
      positions.push(Number(x), Number(y), Number(z));
    }
    const indices: number[] = [];
    for (const f of faces) {
      for (const corner of f.split(" ").slice(1)) {
        indices.push(Number(corner.split("//")[0]) - 1);
      }
    }
    expect(triangleSet(positions, indices)).toEqual(EXPECTED);
  });

  it("obj_without_normals_or_colors", () => {
    const scene = new FakeScene(QUAD, QUAD_TRIS, null, null);
    const text = new TextDecoder().decode(exportOBJ(scene, scene.mesh));
    expect(text.includes("vn ")).toBe(false);
    expect(text.split("\n").filter((l) => l.startsWith("f "))[0]).toMatch(/^f \d+ \d+ \d+$/);
  });

  it("ply_binary_round_trip", () => {
    const scene = new FakeScene();
    const bytes = exportPLY(scene, scene.mesh);
    const head = new TextDecoder().decode(bytes.subarray(0, 400));
    expect(head).toContain("format binary_little_endian 1.0");
    expect(head).toContain("element vertex 4");
    expect(head).toContain("property float nx");
    expect(head).toContain("property uchar red");
    expect(head).toContain("element face 2");
    const headerEnd = head.indexOf("end_header\n") + "end_header\n".length;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    // 4 vertices of (3 + 3 floats, 4 color bytes) then 2 faces of 13 bytes.
    expect(bytes.length).toBe(headerEnd + 4 * (24 + 4) + 2 * 13);
    let at = headerEnd;
    const positions: number[] = [];
    const colors: number[] = [];
    for (let v = 0; v < 4; v++) {
      positions.push(view.getFloat32(at, true), view.getFloat32(at + 4, true), view.getFloat32(at + 8, true));
      colors.push(bytes[at + 24], bytes[at + 25], bytes[at + 26], bytes[at + 27]);
      at += 28;
    }
    expect(colors.slice(0, 4)).toEqual([255, 0, 0, 255]);
    const indices: number[] = [];
    for (let t = 0; t < 2; t++) {
      expect(view.getUint8(at)).toBe(3);
      indices.push(view.getUint32(at + 1, true), view.getUint32(at + 5, true), view.getUint32(at + 9, true));
      at += 13;
    }
    expect(triangleSet(positions, indices)).toEqual(EXPECTED);
  });

  it("ply_ascii_round_trip", () => {
    const scene = new FakeScene();
    const text = new TextDecoder().decode(exportPLY(scene, scene.mesh, { binary: false }));
    expect(text).toContain("format ascii 1.0");
    const lines = text.split("\n");
    const start = lines.indexOf("end_header") + 1;
    // 4 vertices with 3 + 3 + 4 fields, then 2 `3 a b c` faces.
    expect(lines[start].split(" ").length).toBe(10);
    expect(lines[start + 4]).toMatch(/^3 \d+ \d+ \d+$/);
    expect(lines[start + 5]).toMatch(/^3 \d+ \d+ \d+$/);
  });

  it("gltf_glb_round_trip", () => {
    const scene = new FakeScene();
    const bytes = exportGLTF(scene, scene.mesh);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    expect(view.getUint32(0, true)).toBe(0x46546c67);
    expect(view.getUint32(4, true)).toBe(2);
    expect(view.getUint32(8, true)).toBe(bytes.length);
    const jsonLength = view.getUint32(12, true);
    expect(view.getUint32(16, true)).toBe(0x4e4f534a);
    const json = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + jsonLength)));
    expect(json.asset.version).toBe("2.0");
    expect(json.meshes.length).toBe(1);
    expect(json.meshes[0].primitives[0].attributes["COLOR_0"]).toBeDefined();
    expect(json.meshes[0].primitives[0].attributes["NORMAL"]).toBeDefined();
    expect(json.accessors[0].min).toEqual([0, 0, 0]);
    expect(json.accessors[0].max).toEqual([2, 2, 0]);
    const binAt = 20 + jsonLength;
    expect(view.getUint32(binAt + 4, true)).toBe(0x004e4942);
    // Positions are the first buffer view: 4 vertices, 12 floats.
    const binStart = binAt + 8 + json.bufferViews[0].byteOffset;
    const positions = new Float32Array(bytes.buffer, bytes.byteOffset + binStart, 12);
    const indices = new Uint32Array(
      bytes.buffer,
      bytes.byteOffset + binAt + 8 + json.bufferViews[json.bufferViews.length - 1].byteOffset,
      6,
    );
    expect(triangleSet(Array.from(positions), Array.from(indices))).toEqual(EXPECTED);
  });

  it("gltf_json_embeds_data_uri", () => {
    const scene = new FakeScene(QUAD, QUAD_TRIS, null, null);
    const json = JSON.parse(new TextDecoder().decode(exportGLTF(scene, scene.mesh, { binary: false })));
    expect(json.buffers[0].uri.startsWith("data:application/octet-stream;base64,")).toBe(true);
    expect(json.meshes[0].primitives[0].attributes["COLOR_0"]).toBeUndefined();
    expect(json.meshes[0].primitives[0].attributes["NORMAL"]).toBeUndefined();
  });

  it("apply_placement_defaults_to_world", () => {
    // Translate by (10, 0, 0): the default export moves, `false` keeps local.
    const matrix = Float64Array.from(IDENTITY);
    matrix[12] = 10;
    const scene = new FakeScene(QUAD, QUAD_TRIS, QUAD_NORMALS, null, matrix);
    const world = exportSTL(scene, scene.mesh);
    const local = exportSTL(scene, scene.mesh, { applyPlacement: false });
    const x = (b: Uint8Array, t: number, k: number) =>
      new DataView(b.buffer, b.byteOffset, b.byteLength).getFloat32(84 + 50 * t + 12 + 12 * k, true);
    expect(x(world, 0, 0)).toBeCloseTo(10, 5);
    expect(x(local, 0, 0)).toBeCloseTo(0, 5);
  });

  it("skips_dead_slots", () => {
    const positions = Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 7, 7, 7]);
    const indices = Uint32Array.from([0, 1, 2, 0, 0, 0]);
    const scene = new FakeScene(positions, indices, null, null);
    const text = new TextDecoder().decode(exportOBJ(scene, scene.mesh));
    expect(text.split("\n").filter((l) => l.startsWith("v ")).length).toBe(3);
    expect(text.split("\n").filter((l) => l.startsWith("f ")).length).toBe(1);
    expect(text).not.toContain("7 7 7");
  });
});
