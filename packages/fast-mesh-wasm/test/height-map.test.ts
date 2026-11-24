// Height maps and reliefs (#267) on Scene and WorkerScene: the frame, the
// heights of simple shapes, bands of rows, adding a relief as one undo step,
// the round trip through 16-bit codes, and the errors.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import {
  FastMeshError,
  generateHeightfield,
  generatePrimitive,
  type HeightMap,
  type HeightMapFrame,
  heightMapPlacement,
  type MeshHandle,
  Scene,
} from "../src/node.js";
import { decodeHeightMapOptions, encodeHeightMapOptions } from "../src/height-map.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

class NodeWorker implements WorkerLike {
  readonly worker: Worker;
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  readonly #error: ((e: unknown) => void)[] = [];

  constructor() {
    this.worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), {
      workerData: { serve: true },
    });
    this.worker.on("message", (data) => {
      for (const l of [...this.#message]) l({ data });
    });
    this.worker.on("error", (err) => {
      const ev = { message: err instanceof Error ? err.message : String(err) };
      for (const l of [...this.#error]) l(ev);
    });
  }

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", listener: (e: never) => void): void {
    (type === "message" ? this.#message : this.#error).push(listener as never);
  }

  removeEventListener(type: "message" | "error", listener: (e: never) => void): void {
    const list = type === "message" ? this.#message : this.#error;
    const i = list.indexOf(listener as never);
    if (i >= 0) list.splice(i, 1);
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

/** Live triangles in the mesh's render view (dead slots read 0, 0, 0). */
function liveTriangles(s: Scene | WorkerScene, mesh: MeshHandle): number {
  const idx = s.views(mesh).indices();
  let n = 0;
  for (let k = 0; k < idx.length; k += 3) {
    if (idx[k] !== 0 || idx[k + 1] !== 0 || idx[k + 2] !== 0) n++;
  }
  return n;
}

function relief(w: number, h: number): number {
  return 2 * (w - 1) * (h - 1) + 3 * (2 * (w + h) - 4) - 2;
}

/** Every edge crossed once per direction: closed, consistently wound. */
function expectClosed(indices: Uint32Array, vertices: number): void {
  const directed = new Set<number>();
  for (let t = 0; t < indices.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      const key = indices[t + k] * vertices + indices[t + ((k + 1) % 3)];
      expect(directed.has(key)).toBe(false);
      directed.add(key);
    }
  }
  for (const key of directed) {
    const a = Math.floor(key / vertices);
    const b = key % vertices;
    expect(directed.has(b * vertices + a)).toBe(true);
  }
}

/** The world point of a pixel. */
function pointOf(f: HeightMapFrame, col: number, row: number, h: number): number[] {
  return [0, 1, 2].map(
    (k) => f.origin[k] + col * f.pixelSize * f.right[k] - row * f.pixelSize * f.up[k] + h * f.normal[k],
  );
}

async function rejects(fn: () => unknown, code: string): Promise<void> {
  let error: unknown;
  try {
    await fn();
  } catch (e) {
    error = e;
  }
  expect(error).toBeInstanceOf(FastMeshError);
  expect((error as FastMeshError).code).toBe(code);
}

describe("height maps on Scene & WorkerScene", () => {
  const cleanups: (() => void)[] = [];

  afterEach(() => {
    for (const c of cleanups.splice(0)) c();
  });

  const open: Record<string, () => Promise<Scene | WorkerScene>> = {
    Scene: async () => {
      const scene = new Scene();
      cleanups.push(() => scene.free());
      return scene;
    },
    WorkerScene: async () => {
      const worker = await createWorkerScene({ createWorker: () => new NodeWorker() });
      cleanups.push(() => worker.terminate());
      return worker;
    },
  };

  /** Sends the frame and waits for every result so far. */
  async function settle(s: Scene | WorkerScene): Promise<void> {
    if (s instanceof Scene) {
      s.commitFrame();
      return;
    }
    const barrier = s.raycast([0, 0, 1e6], [0, 0, 1]);
    s.commitFrame();
    await barrier;
  }

  for (const [kind, make] of Object.entries(open)) {
    it(`maps a box and a sphere to their analytic heights (${kind})`, async () => {
      const s = await make();
      const box = generatePrimitive({ kind: "box", size: [4, 2, 1] });
      const b = s.addMesh(box.positions, box.indices);
      const ball = generatePrimitive({ kind: "icosphere", radius: 1, subdivisions: 6 });
      const m = s.addMesh(ball.positions, ball.indices);
      s.setPlacement(m, { rotation: [0, 0, 0, 1], translation: [0, 0, 5], scale: [2, 2, 2] });
      await settle(s);

      const frame = await s.heightMapFrame(b, { resolution: 64 });
      expect([frame.width, frame.height, frame.range]).toEqual([64, 32, 1]);
      expect(frame.right).toEqual([1, 0, 0]);
      expect(frame.up).toEqual([0, 1, 0]);
      expect(frame.normal).toEqual([0, 0, 1]);
      expect(frame.pixelSize).toBe(1 / 16);
      expect(frame.origin).toEqual([-2 + 1 / 32, 1 - 1 / 32, -0.5]);
      const flat = await s.heightMap(b, frame);
      expect(flat.rows).toEqual([0, 32]);
      expect(flat.heights).toBeInstanceOf(Float32Array);
      expect(flat.heights.length).toBe(64 * 32);
      expect(flat.heights.every((h) => Math.abs(h - 1) < 1e-6)).toBe(true);

      // A sphere of radius 2 over the box, its frame fitted on the way.
      const map = await s.heightMap([b, m], { resolution: 128, margin: 0.5, direction: [0, 0, -3] });
      const f = map.frame;
      expect(f.range).toBeCloseTo(7.5, 3);
      for (let row = 0; row < f.height; row++) {
        for (let col = 0; col < f.width; col++) {
          const [x, y] = pointOf(f, col, row, 0);
          const r = Math.hypot(x, y);
          const h = map.heights[row * f.width + col];
          if (r < 1.9) {
            expect(Math.abs(h - (5.5 + Math.sqrt(4 - r * r)))).toBeLessThan(3e-3);
          } else if (r > 2) {
            const on = Math.abs(x) < 2 && Math.abs(y) < 1;
            expect(on ? Math.abs(h - 1) < 1e-6 : Number.isNaN(h)).toBe(true);
          }
        }
      }

      // Bands hold exactly the whole.
      const bands: number[] = [];
      for (let first = 0; first < f.height; first += 37) {
        const band: HeightMap = await s.heightMap([b, m], f, [first, Math.min(first + 37, f.height)]);
        expect(band.frame).toEqual(f);
        bands.push(...band.heights);
      }
      expect(Float32Array.from(bands)).toEqual(map.heights);
    });

    it(`adds reliefs as one undo step, and maps back to what was mapped (${kind})`, async () => {
      const s = await make();
      const torus = generatePrimitive({ kind: "torus", majorRadius: 2, minorRadius: 0.7, segments: 96, tubeSegments: 48 });
      const t = s.addMesh(torus.positions, torus.indices);
      s.setPlacement(t, { rotation: [0.3, 0, 0, Math.sqrt(1 - 0.09)], translation: [3, 1, -2] });
      await settle(s);
      const { frame, heights } = await s.heightMap(t, { direction: [0.2, -1, 0.1], resolution: 96, margin: 0.2 });
      expect(heights.some(Number.isNaN)).toBe(true);

      // As a 16-bit image holds it.
      const step = frame.range / 65535;
      const codes = Uint16Array.from(heights, (h) => (Number.isNaN(h) ? 0 : Math.round(h / step)));
      const stored = Float32Array.from(codes, (c) => c * step);
      const field = { heights: stored, width: frame.width, height: frame.height, pixelSize: frame.pixelSize, base: 0.5 };
      const added = await s.addHeightfield(field, { placement: heightMapPlacement(frame) });
      await settle(s);
      expect(stored.length).toBe(frame.width * frame.height);
      expect(added.triangles).toBe(relief(frame.width, frame.height));
      expect(liveTriangles(s, added.mesh)).toBe(added.triangles);
      expect(s.views(added.mesh).normals()).not.toBeNull();
      const back = await s.heightMap(added.mesh, frame);
      for (let row = 1; row < frame.height - 1; row++) {
        for (let col = 1; col < frame.width - 1; col++) {
          const i = row * frame.width + col;
          expect(Math.abs(back.heights[i] - stored[i])).toBeLessThan(0.25 * step);
          expect(Math.round(back.heights[i] / step)).toBe(codes[i]);
        }
      }

      s.undo();
      await settle(s);
      expect([s.has(added.mesh), s.isRestorable(added.mesh)]).toEqual([false, true]);
      s.redo();
      await settle(s);
      expect(liveTriangles(s, added.mesh)).toBe(added.triangles);

      // At the identity without options; a plain array of heights.
      const flat = await s.addHeightfield({ heights: [1, 2, 3, 4, 5, 6], width: 3, height: 2 }, { normals: false });
      await settle(s);
      expect([flat.vertices, flat.triangles]).toEqual([12, relief(3, 2)]);
      expect(s.placement(flat.mesh).translation).toEqual([0, 0, 0]);
      expect(s.views(flat.mesh).normals()).toBeNull();
    });

    it(`refuses bad input (${kind})`, async () => {
      const s = await make();
      const box = generatePrimitive({ kind: "box" });
      const b = s.addMesh(box.positions, box.indices);
      const gone = s.addMesh(box.positions, box.indices);
      await settle(s);
      s.removeMesh(gone);
      await settle(s);
      const bad = "INVALID_ARGUMENT";
      await rejects(() => s.heightMapFrame([]), bad);
      await rejects(() => s.heightMapFrame(gone), "UNKNOWN_MESH");
      await rejects(() => s.heightMap(gone), "UNKNOWN_MESH");
      await rejects(() => s.heightMapFrame(b, { direction: [0, 0, 0] }), bad);
      await rejects(() => s.heightMapFrame(b, { direction: [0, 0] as never }), bad);
      await rejects(() => s.heightMapFrame(b, { up: [0, 0, Number.NaN] }), bad);
      await rejects(() => s.heightMapFrame(b, { up: [0, 0, 1] }), bad);
      await rejects(() => s.heightMapFrame(b, { resolution: 1.5 }), bad);
      await rejects(() => s.heightMapFrame(b, { resolution: 100_000 }), bad);
      await rejects(() => s.heightMapFrame(b, { pixelSize: 0 }), bad);
      await rejects(() => s.heightMapFrame(b, { margin: -1 }), bad);
      await rejects(() => s.heightMapFrame(b, { range: [1, 0] }), bad);
      await rejects(() => s.heightMapFrame(b, { range: [1] as never }), bad);
      const frame = await s.heightMapFrame(b, { resolution: 8 });
      await rejects(() => s.heightMap(b, frame, [0, 9]), bad);
      await rejects(() => s.heightMap(b, frame, [3, 2]), bad);
      await rejects(() => s.heightMap(b, { ...frame, up: frame.right }), bad);
      await rejects(() => s.heightMap(b, { ...frame, width: 0 }), bad);
      await rejects(() => s.addHeightfield({ heights: [1, 2, 3], width: 2, height: 2 }), bad);
      await rejects(() => s.addHeightfield({ heights: [1, 2], width: 1, height: 2 }), bad);
      await rejects(() => s.addHeightfield({ heights: [1, 2, 3, -1], width: 2, height: 2 }), bad);
      await rejects(() => s.addHeightfield({ heights: [1, 2, 3, 4], width: 2, height: 2, base: 0 }), bad);
      await rejects(
        () =>
          s.addHeightfield(
            { heights: [1, 2, 3, 4], width: 2, height: 2 },
            { placement: { rotation: [2, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0] } },
          ),
        bad,
      );
      // Nothing was added, and the scene still works.
      expect((await s.heightMap(b, frame)).heights.every((h) => Math.abs(h - 1) < 1e-6)).toBe(true);
    });
  }

  it("generates the relief it adds, on the main thread and in the Worker", async () => {
    const w = 33;
    const h = 20;
    const heights = Float32Array.from({ length: w * h }, (_, i) => 1 + Math.sin(i * 0.3));
    heights[40] = Number.NaN;
    const field = { heights, width: w, height: h, pixelSize: [0.5, 0.25] as const, base: 2 };
    const mesh = generateHeightfield(field);
    expect(mesh.indices.length / 3).toBe(relief(w, h));
    expect(mesh.positions.length / 3).toBe(w * h + 2 * (w + h) - 4);
    expectClosed(mesh.indices, mesh.positions.length / 3);
    for (let i = 0; i < w * h; i++) {
      expect(mesh.positions[3 * i]).toBe(0.5 * ((i % w) - 16));
      expect(mesh.positions[3 * i + 1]).toBe(0.25 * (9.5 - Math.floor(i / w)));
      expect(mesh.positions[3 * i + 2]).toBe(i === 40 ? 0 : heights[i]);
    }
    expect(mesh.positions.at(-1)).toBe(-2);
    const worker = (await open.WorkerScene()) as WorkerScene;
    expect(await worker.generateHeightfield(field)).toEqual(mesh);
    // The caller keeps its heights.
    expect(heights.length).toBe(w * h);
    await worker.addHeightfield(field);
    expect(heights.length).toBe(w * h);
    await rejects(() => generateHeightfield({ ...field, width: 32 }), "INVALID_ARGUMENT");
    await rejects(() => generateHeightfield({ ...field, pixelSize: [1, 0] }), "INVALID_ARGUMENT");
    await rejects(() => worker.generateHeightfield({ ...field, base: -1 }), "INVALID_ARGUMENT");
  });

  it("encodes options the same way back", () => {
    const options = {
      direction: [1, 2, 3] as [number, number, number],
      up: [0, 0, 1] as [number, number, number],
      resolution: 12,
      pixelSize: 0.5,
      margin: 0.25,
      range: [-1, 4] as [number, number],
    };
    expect(decodeHeightMapOptions(encodeHeightMapOptions(options))).toEqual(options);
    const defaults = { direction: [0, 0, -1], resolution: 1024, margin: 0 };
    expect(decodeHeightMapOptions(encodeHeightMapOptions())).toEqual(defaults);
  });
});
