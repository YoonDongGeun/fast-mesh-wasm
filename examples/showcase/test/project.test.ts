// The project save/reopen format (issue #126): the base64 geometry codec is
// bit-exact, a built doc survives JSON and the tolerant loader, and the
// loader drops bad entries instead of failing the file.
import { describe, expect, it } from "vitest";
import {
  buildProjectDoc,
  decodeF64,
  decodeU32,
  encodeF64,
  encodeU32,
  parseProjectDoc,
  PROJECT_VERSION,
  type ProjectDoc,
} from "../src/load/project";

const POSITIONS = Float64Array.from([0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, -1.5, 1e100, -1e-100]);
const INDICES = Uint32Array.from([0, 1, 2, 0, 2, 3]);

function doc(): ProjectDoc {
  return buildProjectDoc({
    meshes: [
      {
        name: "A",
        color: 0xd9735f,
        opacity: 0.5,
        visible: false,
        positions: POSITIONS,
        indices: INDICES,
        rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        translation: [1, 2, 3],
        scale: [1, 2, 3],
        coloring: { targets: [1], maxDistance: null, map: "viridis", range: [0, 5] },
      },
      {
        name: "B",
        color: 0x6ea8d9,
        opacity: 1,
        visible: true,
        positions: Float64Array.from([9, 9, 9, 10, 9, 9, 10, 10, 9]),
        indices: Uint32Array.from([0, 1, 2]),
        rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1],
        translation: [0, 0, 0],
        coloring: null,
      },
    ],
    sections: [
      { plane: { origin: [0, 0, 0], normal: [0, 0, 1] }, meshes: [0, 1], fill: true },
    ],
    selection: 1,
    wireframe: true,
    edgeSoftness: 2,
  });
}

describe("project codec", () => {
  it("round-trips f64 positions bit-exactly, including extremes", () => {
    expect(decodeF64(encodeF64(POSITIONS), "positions")).toEqual(POSITIONS);
  });

  it("round-trips u32 indices exactly", () => {
    expect(decodeU32(encodeU32(INDICES), "indices")).toEqual(INDICES);
  });

  it("accepts plain arrays as well as typed arrays", () => {
    expect(decodeF64(encodeF64([1.5, -2.25, 0]), "p")).toEqual(Float64Array.from([1.5, -2.25, 0]));
    expect(decodeU32(encodeU32([0, 1, 2]), "i")).toEqual(Uint32Array.from([0, 1, 2]));
  });

  it("rejects broken base64 and partial words", () => {
    expect(() => decodeF64("!!!", "positions")).toThrow(/base64/);
    expect(() => decodeF64(encodeU32([1, 2, 3]), "positions")).toThrow(/float64/);
    expect(() => decodeU32("YQ==", "indices")).toThrow(/uint32/);
  });
});

describe("project save/reopen", () => {
  it("round-trips a session through JSON", () => {
    const parsed = parseProjectDoc(JSON.stringify(doc()));
    expect(parsed.version).toBe(PROJECT_VERSION);
    expect(parsed.meshes).toHaveLength(2);
    const [a, b] = parsed.meshes;
    expect(a.name).toBe("A");
    expect(a.color).toBe(0xd9735f);
    expect(a.opacity).toBe(0.5);
    expect(a.visible).toBe(false);
    expect(decodeF64(a.positions, "p")).toEqual(POSITIONS);
    expect(decodeU32(a.indices, "i")).toEqual(INDICES);
    expect(a.placement).toEqual({ rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [1, 2, 3], scale: [1, 2, 3] });
    expect(a.coloring).toMatchObject({ targets: [1], maxDistance: null, map: "viridis", range: [0, 5] });
    expect(b.coloring).toBeNull();
    expect(parsed.sections).toEqual([{ plane: { origin: [0, 0, 0], normal: [0, 0, 1] }, meshes: [0, 1], fill: true }]);
    expect(parsed.selection).toBe(1);
    expect(parsed.wireframe).toBe(true);
    expect(parsed.edgeSoftness).toBe(2);
  });

  it("is forward-compatible: a newer version with unknown fields still loads", () => {
    const raw = { ...doc(), version: PROJECT_VERSION + 5, future: { foo: [1, 2] }, meshes: doc().meshes.map((m) => ({ ...m, futureMesh: true })) };
    const parsed = parseProjectDoc(JSON.stringify(raw));
    expect(parsed.version).toBe(PROJECT_VERSION + 5);
    expect(parsed.meshes).toHaveLength(2);
  });

  it("defaults missing optionals instead of failing", () => {
    const parsed = parseProjectDoc(JSON.stringify({ version: 1, meshes: [{ positions: encodeF64([0, 0, 0]), indices: encodeU32([0, 0, 0]) }] }));
    expect(parsed.meshes[0].name).toBe("Mesh 1");
    expect(parsed.meshes[0].placement.rotation).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(parsed.sections).toEqual([]);
    expect(parsed.selection).toBeNull();
    expect(parsed.wireframe).toBe(false);
    expect(parsed.edgeSoftness).toBe(0);
  });

  it("drops bad colorings and sections but keeps their meshes", () => {
    const base = doc();
    base.meshes[0].coloring = { targets: [7], maxDistance: 3, map: "viridis", range: [0, 1] };
    base.meshes[1].coloring = { targets: [0], maxDistance: -3, map: "viridis", range: [0, 1] };
    base.sections.push({ plane: { origin: [0, 0, 0], normal: [0, 0, 1] }, meshes: [0, 9], fill: false });
    const parsed = parseProjectDoc(JSON.stringify(base));
    expect(parsed.meshes[0].coloring).toBeNull();
    expect(parsed.meshes[1].coloring).toBeNull();
    expect(parsed.sections).toHaveLength(2);
    expect(parsed.sections[1].meshes).toEqual([0]);
  });

  it("rejects files with no readable meshes", () => {
    expect(() => parseProjectDoc("{}")).toThrow(/no version|not a showcase project/);
    expect(() => parseProjectDoc("nope")).toThrow(/not JSON/);
    expect(() => parseProjectDoc(JSON.stringify({ version: 1, meshes: [] }))).toThrow(/no readable meshes/);
    expect(() => parseProjectDoc(JSON.stringify({ version: 1, meshes: [{ name: "empty" }] }))).toThrow(/no readable meshes/);
  });
});
