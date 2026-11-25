// Height map export (#267) on a real engine: the views, the checks, sampling
// in bands with progress, cancelling and changes meanwhile, the two files, and
// the round trip through them: meshes, to a 16-bit PNG and its JSON, to a
// relief, back to the same image.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { generatePrimitive, init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { AppState } from "../src/app/state/app-state";
import { runAddRelief } from "../src/engine/workbench-ops";
import {
  BAND_PIXELS,
  CancelledError,
  DEFAULT_HEIGHT_MAP,
  describeFrame,
  exportHeightMap,
  frameOptions,
  frameRange,
  HEIGHT_MAP_VIEWS,
  heightMapFileNames,
  heightMapProblem,
  sampleHeightMap,
  viewAxes,
} from "../src/features/home/ux/height-map-export";
import { reliefGrid, reliefOfFrame } from "../src/features/mesh-edit/ux/relief";
import { eulerDegToMat3 } from "../src/features/transform/ux/transform-math";
import { imageGrid, parseHeightMapFile, pngHeights } from "../src/io/height-image";
import { decodePng } from "../src/io/png";

beforeAll(() =>
  init(readFileSync(resolve(__dirname, "../../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"))),
);

const cleanups: (() => void)[] = [];
afterEach(() => {
  for (const c of cleanups.splice(0).reverse()) {
    c();
  }
});

function open(): { scene: Scene; store: SceneStore; app: AppState } {
  const scene = new Scene();
  const store = createSceneStore(scene);
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return { scene, store, app: new AppState() };
}

/** The raw scene's, without a store: what the export runs on. */
const asScene = (s: Scene) => s as never as Parameters<typeof sampleHeightMap>[0];

describe("height map settings", () => {
  it("looks from the standard views and the camera, heights towards the viewer", () => {
    expect(HEIGHT_MAP_VIEWS.map(([id]) => id)).toEqual(["top", "bottom", "front", "back", "left", "right", "camera"]);
    expect(viewAxes("top", null)).toEqual({ direction: [0, -1, 0], up: [0, 0, -1] });
    expect(viewAxes("front", null)).toEqual({ direction: [0, 0, -1], up: [0, 1, 0] });
    expect(viewAxes("left", null)).toEqual({ direction: [1, 0, 0], up: [0, 1, 0] });
    expect(viewAxes("camera", null)).toBeNull();
    // Columns right, up, back: looking down -x with z up.
    const elements = [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 5, 6, 7, 1];
    expect(viewAxes("camera", { matrixWorld: { elements } })).toEqual({ direction: [-1, -0, -0], up: [0, 0, 1] });
  });

  it("checks the resolution and the range", () => {
    expect(heightMapProblem(DEFAULT_HEIGHT_MAP)).toBeNull();
    for (const resolution of [1, 8193, 10.5, Number.NaN]) {
      expect(heightMapProblem({ ...DEFAULT_HEIGHT_MAP, resolution })).toMatch(/resolution/);
    }
    expect(heightMapProblem({ ...DEFAULT_HEIGHT_MAP, range: [0, 1] })).toBeNull();
    for (const range of [
      [1, 1],
      [2, 1],
      [Number.NaN, 1],
    ] as const) {
      expect(heightMapProblem({ ...DEFAULT_HEIGHT_MAP, range })).toMatch(/lowest height/);
    }
  });

  it("names the files and describes a frame in the display unit", () => {
    expect(heightMapFileNames("a/b:c")).toEqual({ png: "a_b_c-height-map.png", json: "a_b_c-height-map.json" });
    expect(heightMapFileNames("  ").png).toBe("mesh-height-map.png");
    const frame = {
      width: 1024,
      height: 512,
      origin: [0, 3, 0] as [number, number, number],
      right: [1, 0, 0] as [number, number, number],
      up: [0, 0, -1] as [number, number, number],
      normal: [0, 1, 0] as [number, number, number],
      pixelSize: 0.254,
      range: 25.4,
    };
    expect(describeFrame(frame, "in")).toBe("1024 × 512 px · 0.01 in per pixel · heights 0 to 1 in");
    expect(frameRange(frame)).toEqual([3, 28.4]);
  });
});

describe("sampling and the files", () => {
  it("makes a 16-bit PNG and a JSON that say what the mesh is", async () => {
    const { scene } = open();
    const box = generatePrimitive({ kind: "box", size: [4, 1, 2] });
    const mesh = scene.addMesh(box.positions, box.indices);
    scene.setPlacement(mesh, { rotation: [0, 0, 0, 1], translation: [10, 5, -3] });
    const axes = viewAxes("top", null)!;
    const progress: number[] = [];
    const out = await exportHeightMap(
      asScene(scene),
      [{ mesh, name: "Block" }],
      frameOptions({ ...DEFAULT_HEIGHT_MAP, resolution: 64 }, axes),
      { onProgress: (p) => progress.push(p) },
    );
    expect(out.png.name).toBe("Block-height-map.png");
    expect(progress.at(-1)).toBe(1);
    const png = await decodePng(out.png.bytes);
    expect([png.width, png.height, png.channels, png.depth]).toEqual([64, 32, 1, 16]);
    // Every pixel sees the flat top: white.
    expect(png.samples.every((c) => c === 65535)).toBe(true);
    expect(out.seen).toBe(64 * 32);
    const doc = JSON.parse(out.json.text) as Record<string, unknown>;
    expect(doc).toMatchObject({
      type: "height-map",
      image: "Block-height-map.png",
      width: 64,
      height: 32,
      units: "mm",
      pixelSize: 1 / 16,
      heightRange: 1,
      heightStep: 1 / 65535,
      right: [1, 0, 0],
      up: [0, 0, -1],
      normal: [0, 1, 0],
      meshes: ["Block"],
    });
    // The top left pixel's centre at height 0: the box's bottom, at its -x, -z corner.
    const origin = doc.origin as number[];
    [8 + 1 / 32, 4.5, -4 + 1 / 32].forEach((c, k) => expect(origin[k]).toBeCloseTo(c, 9));
  });

  it("samples large images in bands that hold exactly what one call would", async () => {
    const { scene } = open();
    const ball = generatePrimitive({ kind: "icosphere", radius: 3, subdivisions: 5 });
    const mesh = scene.addMesh(ball.positions, ball.indices);
    const resolution = 1500;
    expect(resolution * resolution).toBeGreaterThan(2 * BAND_PIXELS);
    const progress: number[] = [];
    const { frame, heights } = await sampleHeightMap(
      asScene(scene),
      [mesh],
      { direction: [0.2, -1, 0.3], resolution, margin: 0.1 },
      { onProgress: (p) => progress.push(p) },
    );
    expect(progress.length).toBeGreaterThan(2);
    expect([...progress].sort((a, b) => a - b)).toEqual(progress);
    const whole = scene.heightMap(mesh, frame).heights;
    expect(Buffer.from(heights.buffer).equals(Buffer.from(whole.buffer, whole.byteOffset, whole.byteLength))).toBe(true);
  });

  it("stops when cancelled", async () => {
    const { scene } = open();
    const ball = generatePrimitive({ kind: "icosphere", radius: 3, subdivisions: 3 });
    const mesh = scene.addMesh(ball.positions, ball.indices);
    const control = new AbortController();
    const run = sampleHeightMap(
      asScene(scene),
      [mesh],
      { resolution: 1500 },
      { signal: control.signal, onProgress: () => control.abort() },
    );
    await expect(run).rejects.toBeInstanceOf(CancelledError);
  });

  it("refuses a seam when a shown mesh moves or is edited, not for other meshes or earlier changes", async () => {
    const { scene } = open();
    const ball = generatePrimitive({ kind: "icosphere", radius: 3, subdivisions: 3 });
    const shown = scene.addMesh(ball.positions, ball.indices);
    const other = scene.addMesh(ball.positions, ball.indices);
    const at = (x: number) => ({ rotation: [0, 0, 0, 1] as const, translation: [x, 0, 0] as [number, number, number] });
    const meanwhile = (change: () => void) => {
      let done = false;
      return sampleHeightMap(
        asScene(scene),
        [shown],
        { resolution: 1500 },
        {
          onProgress: () => {
            if (!done) {
              done = true;
              change();
            }
          },
        },
      );
    };
    await expect(meanwhile(() => scene.setPlacement(shown, at(1)))).rejects.toThrow(/changed while/);
    await expect(
      meanwhile(() => void scene.edit(shown, [{ op: "setPositions", vertices: [0], positions: [0, 0, 9] }])),
    ).rejects.toThrow(/changed while/);
    await expect(meanwhile(() => scene.setPlacement(other, at(5)))).resolves.toBeDefined();
    // A change before it starts is in the frame: no seam.
    scene.setPlacement(shown, at(2));
    await expect(sampleHeightMap(asScene(scene), [shown], { resolution: 1500 })).resolves.toBeDefined();
  });
});

describe("round trip", () => {
  it("comes back from the files as the surface the map saw, within one code", async () => {
    const { scene, store, app } = open();
    const torus = generatePrimitive({ kind: "torus", majorRadius: 8, minorRadius: 3, segments: 160, tubeSegments: 80 });
    const a = scene.addMesh(torus.positions, torus.indices);
    scene.setPlacement(a, { rotation: [0.2, 0.1, 0, Math.sqrt(1 - 0.05)], translation: [40, -7, 12] });
    const cone = generatePrimitive({ kind: "cone", radius: 4, height: 9, segments: 90, rings: 8 });
    const b = scene.addMesh(cone.positions, cone.indices);
    scene.setPlacement(b, { rotation: [0, 0, 0, 1], translation: [40, -4, 12] });

    for (const view of ["top", "front", "left"] as const) {
      const axes = viewAxes(view, null)!;
      const options = { ...frameOptions({ ...DEFAULT_HEIGHT_MAP, resolution: 200 }, axes), margin: 0.5 };
      const list = [
        { mesh: a, name: "torus" },
        { mesh: b, name: "cone" },
      ];
      const out = await exportHeightMap(asScene(scene), list, options);
      const first = await decodePng(out.png.bytes);

      // As the Image to Relief dialog reads the two files.
      const image = pngHeights(first);
      const frame = parseHeightMapFile(out.json.text)!;
      const fitted = reliefOfFrame(frame, image, 1)!;
      const { w, h, pixelSize } = reliefGrid(fitted.params, image);
      expect([w, h]).toEqual([frame.width, frame.height]);
      const heights = imageGrid(image, w, h, { scale: fitted.params.maxHeight, invert: false });
      const { item, triangles } = await runAddRelief({
        store,
        app,
        name: `relief ${view}`,
        field: { heights, width: w, height: h, pixelSize, base: fitted.params.base },
        placement: { rotation: eulerDegToMat3(fitted.place.rotationDeg), translation: fitted.place.centre },
      });
      expect(item.triangles).toBe(triangles);
      const measured = scene.inspect.measure(item.mesh);
      expect([measured.closed, measured.boundaryLoops, measured.nonManifoldEdges]).toEqual([true, 0, 0]);

      // The same image of the relief alone, in the same frame.
      const again = scene.heightMap(item.mesh, frame).heights;
      const step = frame.range / 65535;
      const codes = first.samples;
      let worst = 0;
      for (let row = 1; row < h - 1; row++) {
        for (let col = 1; col < w - 1; col++) {
          const i = row * w + col;
          worst = Math.max(worst, Math.abs(again[i] - codes[i] * step));
        }
      }
      expect(worst).toBeLessThanOrEqual(step);

      // One undo step takes it away again.
      store.undo();
      expect(scene.has(item.mesh)).toBe(false);
    }
  });
});
