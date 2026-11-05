// The project save/reopen format (issue #126): the base64 geometry codec is
// bit-exact, a built doc survives JSON and the tolerant loader, and the
// loader drops bad entries instead of failing the file.
import type { SceneStore } from "fast-mesh-wasm/store";
import { describe, expect, it } from "vitest";
import { bounds } from "../src/load/mesh-data";
import { importScaleOf, scalePositions } from "../src/panels/units";
import { AppState, type Item } from "../src/scene/app-state";
import {
  buildProjectDoc,
  collectProject,
  decodeF64,
  decodeU32,
  encodeF64,
  encodeU32,
  parseProjectDoc,
  PROJECT_VERSION,
  type ProjectDoc,
  restoreProject,
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

describe("project units (issue #145)", () => {
  /** A store stub: enough of `SceneStore` to add, save and replace meshes. */
  function fakeStore() {
    const meshes = new Map<object, { positions: Float64Array; indices: Uint32Array }>();
    const display = { coloring: null, opacity: 1, visible: true };
    return {
      meshes,
      store: {
        addMesh(positions: Float64Array, indices: Uint32Array) {
          const handle = {};
          meshes.set(handle, { positions, indices });
          return handle;
        },
        removeMesh: (h: object) => void meshes.delete(h),
        removeSection() {},
        addSection() {},
        select() {},
        setDisplay() {},
        setPlacement() {},
        getSnapshot: () => ({
          meshes: new Map([...meshes.keys()].map((h) => [h, { desired: display }])),
          sections: new Map(),
          selection: null,
        }),
        scene: {
          views: (h: object) => ({
            positions: () => meshes.get(h)!.positions,
            indices: () => meshes.get(h)!.indices,
          }),
          placement: () => ({ rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0] }),
        },
      } as unknown as SceneStore,
    };
  }

  /** A 1-inch-edge triangle as a file would hold it, imported as inches. */
  function importInches(app: AppState, store: SceneStore): Item {
    const positions = scalePositions(Float64Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]), importScaleOf("in").factor);
    return app.addMesh(store, {
      name: "part",
      positions,
      indices: Uint32Array.from([0, 1, 2]),
      ...bounds(positions),
      volumeId: null,
      preset: null,
      importScale: importScaleOf("in"),
    });
  }

  it("round-trips the display units and the import scale", () => {
    const built = buildProjectDoc({
      meshes: [
        {
          name: "A",
          color: 1,
          opacity: 1,
          visible: true,
          positions: POSITIONS,
          indices: INDICES,
          rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
          translation: [0, 0, 0],
          coloring: null,
          importScale: { factor: 25.4, units: "in" },
        },
      ],
      sections: [],
      selection: null,
      wireframe: false,
      edgeSoftness: 0,
      units: "in",
    });
    const parsed = parseProjectDoc(JSON.stringify(built));
    expect(parsed.units).toBe("in");
    expect(parsed.meshes[0].importScale).toEqual({ factor: 25.4, units: "in" });
    expect(parsed.version).toBe(PROJECT_VERSION);
  });

  it("an older file without these fields loads unchanged", () => {
    const older = JSON.parse(JSON.stringify(doc())) as Record<string, unknown>;
    expect("units" in older).toBe(false);
    const parsed = parseProjectDoc(JSON.stringify(older));
    expect(parsed).toEqual(doc());
    expect(parsed.units).toBeUndefined();
    expect(parsed.meshes[0].importScale).toBeUndefined();
    expect(decodeF64(parsed.meshes[0].positions, "p")).toEqual(POSITIONS);
  });

  it("restoring an older file keeps its geometry, and shows the global default", () => {
    const { store, meshes } = fakeStore();
    const app = new AppState();
    app.setUnits("in");
    restoreProject(store, app, parseProjectDoc(JSON.stringify(doc())));
    const s = app.getSnapshot();
    expect(s.units).toBe("in");
    expect(s.items[0].importScale).toEqual({ factor: 1, units: "mm" });
    expect([...meshes.values()][0].positions).toEqual(POSITIONS);
  });

  it("tolerates bad values: a bad unit or factor is as if absent", () => {
    const bad = JSON.parse(JSON.stringify(doc())) as { units: unknown; meshes: { importScale: unknown }[] };
    bad.units = "furlong";
    bad.meshes[0].importScale = { factor: -2, units: "in" };
    bad.meshes[1].importScale = { factor: 3, units: "furlong" };
    const parsed = parseProjectDoc(JSON.stringify(bad));
    expect(parsed.units).toBeUndefined();
    expect(parsed.meshes[0].importScale).toBeUndefined();
    expect(parsed.meshes[1].importScale).toEqual({ factor: 3, units: null });
  });

  it("a mesh is not scaled twice on reload: repeated round trips keep the geometry bit-exact", () => {
    const first = fakeStore();
    const app = new AppState();
    const item = importInches(app, first.store);
    const stored = Float64Array.from(first.meshes.get(item.mesh as unknown as object)!.positions);
    // Written out: the 1 in edge is 25.4 in the world, once.
    expect([...stored]).toEqual([0, 0, 0, 25.4, 0, 0, 0, 25.4, 0]);

    let json = JSON.stringify(collectProject(first.store, app));
    for (let trip = 0; trip < 3; trip++) {
      const next = fakeStore();
      const reopened = new AppState();
      restoreProject(next.store, reopened, parseProjectDoc(json));
      const [restored] = reopened.getSnapshot().items;
      expect(next.meshes.get(restored.mesh as unknown as object)!.positions).toEqual(stored);
      // The provenance rides along; the size follows the stored geometry.
      expect(restored.importScale).toEqual({ factor: 25.4, units: "in" });
      expect(restored.radius).toBeCloseTo((25.4 * Math.SQRT2) / 2, 12);
      const again = JSON.stringify(collectProject(next.store, reopened));
      expect(again).toBe(json);
      json = again;
    }
  });

  it("the display unit changes no stored value in the file", () => {
    const { store } = fakeStore();
    const app = new AppState();
    importInches(app, store);
    const inMm = collectProject(store, app);
    app.setUnits("in");
    const inIn = collectProject(store, app);
    expect(inMm.units).toBe("mm");
    expect(inIn.units).toBe("in");
    expect(inIn.meshes).toEqual(inMm.meshes);
    expect(inIn.edgeSoftness).toBe(inMm.edgeSoftness);
  });

  it("opening a project in inches leaves the global default alone", () => {
    const { store } = fakeStore();
    const saved = new AppState();
    importInches(saved, store);
    saved.setUnits("in");
    const json = JSON.stringify(collectProject(store, saved));
    const other = fakeStore();
    const app = new AppState();
    restoreProject(other.store, app, parseProjectDoc(json));
    expect(app.getSnapshot().units).toBe("in");
    expect(app.getSnapshot().defaultUnits).toBe("mm");
  });
});
