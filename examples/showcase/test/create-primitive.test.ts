// The Create dialog's behaviour (#259): shape specs and defaults, the engine
// shape they map to, triangle estimates against the real generator, the
// checks, naming, where generation runs, and adding as one undo step.
import { generatePrimitive, type PrimitiveShape, Scene, type SceneLike } from "fast-mesh-wasm";
import { createSceneStore } from "fast-mesh-wasm/store";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppState } from "../src/app/state/app-state";
import { createPrimitiveCommands } from "../src/features/home/ux/create-commands";
import { createDialogOf } from "../src/features/home/ux/create-dialog-state";
import {
  addPrimitive,
  CREATE_SHAPES,
  defaultBaseSize,
  defaultParams,
  estimateTriangles,
  generateShape,
  MAX_CREATE_TRIANGLES,
  niceSize,
  paramsProblem,
  placePositions,
  toShape,
  uniqueName,
} from "../src/features/home/ux/create-primitive";

const cleanups: (() => void)[] = [];
afterEach(() => {
  for (const c of cleanups.splice(0).reverse()) c();
});

function open() {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  app.follow(store);
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return { scene, store, app };
}

describe("Create shapes", () => {
  it("offers every engine shape, each with valid defaults that estimate their triangles exactly", () => {
    expect(CREATE_SHAPES.map((s) => s.kind)).toEqual([
      "box",
      "cylinder",
      "cone",
      "uvSphere",
      "icosphere",
      "torus",
      "arrow",
      "prism",
      "plane",
    ]);
    for (const s of CREATE_SHAPES) {
      for (const size of [0.5, 2, 50]) {
        const params = defaultParams(s.kind, size);
        expect(paramsProblem(s.kind, params), s.kind).toBeNull();
        const mesh = generatePrimitive(toShape(s.kind, params));
        expect(estimateTriangles(s.kind, params), s.kind).toBe(mesh.indices.length / 3);
      }
    }
  });

  it("estimates resolutions as the engine builds them", () => {
    const cases: [string, Record<string, number>][] = [
      ["cylinder", { radius: 1, height: 2, segments: 7, rings: 3, capRings: 2 }],
      ["cone", { radius: 0, topRadius: 1, height: 2, segments: 9, rings: 4, capRings: 3 }],
      ["cone", { radius: 1, topRadius: 0.5, height: 2, segments: 5, rings: 1, capRings: 2 }],
      ["uvSphere", { radius: 1, segments: 11, rings: 5 }],
      ["icosphere", { radius: 1, subdivisions: 4 }],
      ["torus", { majorRadius: 2, minorRadius: 0.5, segments: 13, tubeSegments: 7 }],
      ["arrow", { length: 2, shaftRadius: 0.1, headRadius: 0.3, headLength: 0.5, segments: 17 }],
      ["prism", { sides: 5, radius: 1, height: 1 }],
      ["box", { sizeX: 1, sizeY: 2, sizeZ: 3, segX: 2, segY: 3, segZ: 4 }],
      ["plane", { width: 2, depth: 3, segX: 4, segZ: 5 }],
    ];
    for (const [kind, params] of cases) {
      const k = kind as (typeof CREATE_SHAPES)[number]["kind"];
      expect(paramsProblem(k, params), kind).toBeNull();
      expect(estimateTriangles(k, params), kind).toBe(generatePrimitive(toShape(k, params)).indices.length / 3);
    }
  });

  it("maps per-axis sizes onto the engine's box and plane", () => {
    expect(toShape("box", { sizeX: 1, sizeY: 2, sizeZ: 3, segX: 4, segY: 5, segZ: 6 })).toEqual({
      kind: "box",
      size: [1, 2, 3],
      segments: [4, 5, 6],
    });
    const plane = toShape("plane", { width: 4, depth: 2, segX: 3, segZ: 1 });
    expect(plane).toEqual({ kind: "plane", size: [4, 2], segments: [3, 1] });
    // Width along X, depth along Z, facing +Y.
    const p = generatePrimitive(plane).positions;
    const xs = p.filter((_, i) => i % 3 === 0);
    const zs = p.filter((_, i) => i % 3 === 2);
    expect([Math.min(...xs), Math.max(...xs), Math.min(...zs), Math.max(...zs)]).toEqual([-2, 2, -1, 1]);
  });

  it("says what is wrong with parameters before the engine is asked", () => {
    const torus = defaultParams("torus", 2);
    expect(paramsProblem("torus", { ...torus, minorRadius: torus.majorRadius })).toMatch(/tube radius/i);
    const arrow = defaultParams("arrow", 2);
    expect(paramsProblem("arrow", { ...arrow, headRadius: arrow.shaftRadius })).toMatch(/head radius/i);
    expect(paramsProblem("arrow", { ...arrow, headLength: arrow.length })).toMatch(/head length/i);
    expect(paramsProblem("cone", { ...defaultParams("cone", 2), radius: 0, topRadius: 0 })).toMatch(/cone/i);
    expect(paramsProblem("cone", { ...defaultParams("cone", 2), radius: 0, topRadius: 1 })).toBeNull();
    expect(paramsProblem("box", { ...defaultParams("box", 2), sizeX: 0 })).toMatch(/Width/);
    expect(paramsProblem("cylinder", { ...defaultParams("cylinder", 2), segments: 2 })).toMatch(/Sides/);
    expect(paramsProblem("cylinder", { ...defaultParams("cylinder", 2), segments: 12.5 })).toMatch(/whole/);
    expect(paramsProblem("uvSphere", { ...defaultParams("uvSphere", 2), radius: Number.NaN })).toMatch(/number/);
    const huge = { ...defaultParams("plane", 2), segX: 4096, segZ: 4096 };
    expect(estimateTriangles("plane", huge)).toBeGreaterThan(MAX_CREATE_TRIANGLES);
    expect(paramsProblem("plane", huge)).toMatch(/at most/);
  });

  it("sizes defaults to the scene and names shapes uniquely", () => {
    expect(niceSize(1)).toBe(1);
    expect(niceSize(1.6)).toBe(2);
    expect(niceSize(4)).toBe(5);
    expect(niceSize(8)).toBe(10);
    expect(niceSize(0.03)).toBeCloseTo(0.02, 12);
    expect(defaultBaseSize([])).toBe(2);
    expect(defaultBaseSize([{ radius: 1 }, { radius: 12 }, { radius: 20 }])).toBe(20);
    expect(defaultParams("cylinder", 20)).toMatchObject({ radius: 10, height: 20, segments: 48 });
    expect(uniqueName("Box", [])).toBe("Box");
    expect(uniqueName("Box", ["Box", "Box 2", "Torus"])).toBe("Box 3");
  });

  it("bakes the rotation (Euler XYZ degrees) and the centre into world positions", () => {
    const p = Float32Array.from([1, 0, 0, 0, 1, 0]);
    const w = placePositions(p, [10, 20, 30], [0, 0, 90]);
    expect(Array.from(w).map((v) => Math.round(v * 1e9) / 1e9)).toEqual([10, 21, 30, 9, 20, 30]);
    expect(Array.from(placePositions(p, [0, 0, 0], [0, 0, 0]))).toEqual([1, 0, 0, 0, 1, 0]);
  });
});

describe("Create runs where the engine runs", () => {
  it("uses the Worker scene's generator, and main-thread wasm for the synchronous scene", async () => {
    const shape: PrimitiveShape = { kind: "prism", sides: 4 };
    const fromWorker = { positions: new Float32Array(3), indices: new Uint32Array(3) };
    const worker = { status: "ready", generatePrimitive: vi.fn(async () => fromWorker) } as unknown as SceneLike;
    await expect(generateShape(worker, shape)).resolves.toBe(fromWorker);
    expect((worker as unknown as { generatePrimitive: ReturnType<typeof vi.fn> }).generatePrimitive).toHaveBeenCalledWith(
      shape,
    );
    const { scene } = open();
    const local = await generateShape(scene, shape);
    expect(local.indices).toEqual(generatePrimitive(shape).indices);
  });
});

describe("addPrimitive", () => {
  it("adds the shape as one undo step: undo removes it, redo brings it back", async () => {
    const { scene, store, app } = open();
    const mesh = generatePrimitive({ kind: "box", size: [2, 4, 6] });
    expect(scene.canUndo()).toBe(false);
    const { item, done } = addPrimitive(store, app, mesh, { name: "Box", centre: [10, 0, -5], rotationDeg: [0, 90, 0] });
    await done;
    expect(app.getSnapshot().items).toEqual([item]);
    expect(item).toMatchObject({ name: "Box", triangles: 12, center: [10, 0, -5] });
    expect(item.radius).toBeCloseTo(Math.hypot(2, 4, 6) / 2, 5);
    // The mesh origin is the centre; the rotation is baked in, the placement stays identity.
    expect(scene.placement(item.mesh).rotation).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const m = scene.inspect.measure(item.mesh);
    expect(m.closed).toBe(true);
    expect(m.volume).toBeCloseTo(48, 3);
    // 90° about Y turns X (width 2) onto Z: the world box spans z ± 1 around -5 and x ± 3 around 10.
    const local = scene.views(item.mesh).positions();
    const m4 = scene.matrix4(item.mesh);
    const xs: number[] = [];
    const zs: number[] = [];
    for (let i = 0; i < local.length; i += 3) {
      xs.push(m4[0] * local[i] + m4[4] * local[i + 1] + m4[8] * local[i + 2] + m4[12]);
      zs.push(m4[2] * local[i] + m4[6] * local[i + 1] + m4[10] * local[i + 2] + m4[14]);
    }
    expect([Math.min(...xs), Math.max(...xs)].map((v) => Math.round(v * 1e4) / 1e4)).toEqual([7, 13]);
    expect([Math.min(...zs), Math.max(...zs)].map((v) => Math.round(v * 1e4) / 1e4)).toEqual([-6, -4]);

    expect(scene.canUndo()).toBe(true);
    store.undo();
    expect(scene.canUndo()).toBe(false);
    // The mesh goes, and its row with it (#269).
    expect(scene.has(item.mesh)).toBe(false);
    expect(app.getSnapshot().items).toEqual([]);
    store.redo();
    expect(scene.inspect.measure(item.mesh).volume).toBeCloseTo(48, 3);
    expect(app.getSnapshot().items).toEqual([item]);
  });

  it("adds nothing when the scene refuses the mesh", () => {
    const { scene, store, app } = open();
    const bad = { positions: Float32Array.from([0, 0, 0]), indices: Uint32Array.from([0, 1, 2]) };
    expect(() => addPrimitive(store, app, bad, { name: "Bad", centre: [0, 0, 0], rotationDeg: [0, 0, 0] })).toThrow();
    expect(app.getSnapshot().items).toEqual([]);
    expect(store.getSnapshot().meshes.size).toBe(0);
    expect(scene.canUndo()).toBe(false);
  });
});

describe("Create dialog state and commands", () => {
  it("opens afresh, keeps a preview only while open, and clears it on close", () => {
    const app = new AppState();
    const dialog = createDialogOf(app);
    expect(createDialogOf(app)).toBe(dialog);
    expect(createDialogOf(new AppState())).not.toBe(dialog);
    const mesh = generatePrimitive({ kind: "box" });
    dialog.setPreview({ mesh, centre: [0, 0, 0], rotationDeg: [0, 0, 0] });
    expect(dialog.getSnapshot().preview).toBeNull();
    dialog.open("torus");
    expect(dialog.getSnapshot()).toMatchObject({ open: true, kind: "torus", session: 1 });
    dialog.setPreview({ mesh, centre: [1, 2, 3], rotationDeg: [0, 0, 0] });
    expect(dialog.getSnapshot().preview?.centre).toEqual([1, 2, 3]);
    dialog.close();
    expect(dialog.getSnapshot()).toMatchObject({ open: false, preview: null });
    dialog.open();
    expect(dialog.getSnapshot()).toMatchObject({ open: true, kind: "torus", session: 2 });
  });

  it("puts a Create entry and one per shape in the command palette", () => {
    const app = new AppState();
    const cmds = createPrimitiveCommands(app);
    expect(cmds.map((c) => c.id)).toEqual(["action:create-primitive", ...CREATE_SHAPES.map((s) => `create:${s.kind}`)]);
    cmds.find((c) => c.id === "create:icosphere")!.run();
    expect(createDialogOf(app).getSnapshot()).toMatchObject({ open: true, kind: "icosphere" });
    expect(cmds.find((c) => c.id === "create:plane")!.keywords).toMatch(/grid/);
  });
});
