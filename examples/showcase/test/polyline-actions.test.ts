// The polyline tools on the engine (#268), on the sync `Scene` and on a
// Worker: keeping contours and paths, the edits and their undo in the scene's
// history, tubes and extrusions as undoable meshes, the SVG and DXF files,
// and projects.
import { Worker } from "node:worker_threads";
import { Scene } from "fast-mesh-wasm";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "fast-mesh-wasm/worker";
import { afterEach, describe, expect, it } from "vitest";
import { AppState, type Item } from "../src/app/state/app-state";
import { sliceCurves } from "../src/features/inspect/ux/use-contour-slices";
import { clickedSelection, cssColor, parseColor } from "../src/features/polylines/ux/polyline-list";
import {
  activePolylines,
  describeKept,
  extrudePolylines,
  keepCurves,
  keepPath,
  keepSection,
  keepSectionText,
  nextColor,
  offsetPolylines,
  planarProblemText,
  removePolylines,
  resamplePolylines,
  simplifyPolylines,
  tubePolylines,
} from "../src/features/polylines/ux/polyline-actions";
import { polylinesFile } from "../src/features/polylines/ux/polyline-export";
import { curveLength, makeCurve, pointCount, POLYLINE_PALETTE } from "../src/features/polylines/ux/polyline";
import { defaultLengths, describeEdit } from "../src/features/polylines/ux/use-polyline-ops";
import { blobbySphere } from "../src/io/demo";
import { collectProject, parseProjectDoc, restoreProject } from "../src/io/project";

const cleanups: (() => void | Promise<void>)[] = [];
afterEach(async () => {
  while (cleanups.length > 0) {
    await cleanups.pop()?.();
  }
});

/** The SDK's Node worker-thread stand-in for the browser engine Worker. */
class NodeWorker implements WorkerLike {
  readonly worker = new Worker(new URL("../../../packages/fast-mesh-wasm/test/harness/node-worker.mjs", import.meta.url), {
    workerData: { serve: true },
  });

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", cb: (e: never) => void): void {
    this.worker.on(type, (data: unknown) => cb((type === "message" ? { data } : data) as never));
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

interface Harness {
  scene: Scene | WorkerScene;
  store: SceneStore;
  app: AppState;
  settle: () => Promise<void>;
}

async function local(): Promise<Harness> {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(app.follow(store));
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return {
    scene,
    store,
    app,
    settle: async () => {
      store.commitFrame();
    },
  };
}

async function worker(): Promise<Harness> {
  const scene = await createWorkerScene({ createWorker: () => new NodeWorker() });
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(app.follow(store));
  cleanups.push(() => {
    store.dispose();
    scene.terminate();
  });
  return {
    scene,
    store,
    app,
    settle: async () => {
      for (let k = 0; k < 2; k++) {
        const barrier = scene.raycast([0, 0, 1e6], [0, 0, 1]);
        store.commitFrame();
        await barrier;
        store.commitFrame();
      }
    },
  };
}

function addBall(h: Harness, name = "Ball", center: [number, number, number] = [0, 0, 0]): Item {
  const d = blobbySphere(center, 1, 48, 32, 0);
  return h.app.addMesh(h.store, { name, ...d, center, radius: 1, volumeId: null, preset: null });
}

const SQUARE = { points: [0, 0, 0, 10, 0, 0, 10, 10, 0, 0, 10, 0], closed: true };
const LINE = { points: [0, 0, 5, 4, 0, 5, 4, 3, 5], closed: false };

const names = (h: Harness) => h.app.getSnapshot().polylines.map((p) => p.name);
const items = (h: Harness) => h.app.getSnapshot().items.map((i) => i.name);
const active = (h: Harness) => activePolylines(h.app.getSnapshot());

describe("keeping curves", () => {
  it("names, colours and selects what it keeps, as one undo step", async () => {
    const h = await local();
    const [a] = keepCurves(h.app, h.store, [SQUARE], "Cut");
    expect(a).toMatchObject({ name: "Cut", closed: true, visible: true, color: POLYLINE_PALETTE[0] });
    expect(active(h)).toEqual([a]);
    const more = keepCurves(h.app, h.store, [SQUARE, LINE, { points: [1, 1, 1], closed: false }], "Cut");
    expect(more.map((p) => p.name)).toEqual(["Cut 1", "Cut 2"]);
    expect(more.map((p) => p.color)).toEqual([POLYLINE_PALETTE[1], POLYLINE_PALETTE[1]]);
    expect(h.app.getSnapshot().polylineSelection).toEqual(more.map((p) => p.id));
    expect(describeKept(more)).toBe('Kept 2 polylines, "Cut 1" to "Cut 2".');
    expect(describeKept([a])).toBe('Kept as "Cut".');
    h.store.undo();
    expect(names(h)).toEqual(["Cut"]);
    expect(h.app.getSnapshot().polylineSelection).toEqual([]);
    h.store.undo();
    expect(names(h)).toEqual([]);
    expect(h.store.getSnapshot().history.canUndo).toBe(false);
    h.store.redo();
    h.store.redo();
    expect(names(h)).toEqual(["Cut", "Cut 1", "Cut 2"]);
    // The markers that stand for the list's states are no meshes of the list.
    expect(items(h)).toEqual([]);
  });

  it("refuses nothing and too much, and changes nothing then", async () => {
    const h = await local();
    expect(() => keepCurves(h.app, h.store, [{ points: [0, 0, 0], closed: true }], "Cut")).toThrow("no curve");
    expect(() => keepCurves(h.app, h.store, Array.from({ length: 1001 }, () => LINE), "Cut")).toThrow("at most 1,000");
    expect(() => keepPath(h.app, h.store, null, "Ball")).toThrow("Draw a path");
    expect(names(h)).toEqual([]);
    expect(h.store.getSnapshot().history.canUndo).toBe(false);
  });

  it("gives each Keep the colour least used", () => {
    expect(nextColor([])).toBe(POLYLINE_PALETTE[0]);
    expect(nextColor([{ color: POLYLINE_PALETTE[0] }, { color: POLYLINE_PALETTE[1] }])).toBe(POLYLINE_PALETTE[2]);
    expect(nextColor(POLYLINE_PALETTE.map((color) => ({ color })))).toBe(POLYLINE_PALETTE[0]);
  });

  it("keeps the engine's section contours, of slices and of the section plane", async () => {
    const h = await local();
    const ball = addBall(h);
    addBall(h, "Other", [5, 0, 0]);
    const slices = (h.scene as Scene).inspect.sliceContourStack(ball.mesh, [0, 0, -0.5], [0, 0, 1], 0.5, 3);
    const kept = keepCurves(h.app, h.store, sliceCurves(slices), "Ball contour");
    expect(kept.map((p) => p.name)).toEqual(["Ball contour 1", "Ball contour 2", "Ball contour 3"]);
    kept.forEach((p, i) => {
      expect(p.closed).toBe(true);
      expect(curveLength(p)).toBeCloseTo(slices[i].totalPerimeter, 3);
    });
    const handle = h.store.addSection({
      plane: { origin: [0, 0, 0.2], normal: [0, 0, 1] },
      meshes: h.app.getSnapshot().items.map((i) => i.mesh),
      fill: false,
    });
    h.store.commitFrame();
    const section = { handle, plane: { origin: [0, 0, 0.2] as [number, number, number], normal: [0, 0, 1] as [number, number, number] } };
    const cut = keepSection(h.app, h.store, h.scene, section);
    expect(cut.map((p) => p.name)).toEqual(["Section 1", "Section 2"]);
    expect(cut.every((p) => p.closed && Math.abs(p.points[2] - 0.2) < 1e-6)).toBe(true);
    // The plane stays, and so does what was kept, when the meshes change or go.
    h.app.removeMesh(h.store, ball);
    expect(names(h)).toHaveLength(5);
    expect(keepSectionText(h.app, h.store, h.scene, { ...section, handle: {} as never })).toMatchObject({ error: true });
  });
});

describe("editing polylines", () => {
  it("simplifies and resamples in place, undone and redone in order with other edits", async () => {
    const h = await local();
    const ring = Array.from({ length: 400 }, (_, i) => {
      const a = (2 * Math.PI * i) / 400;
      return [10 * Math.cos(a), 10 * Math.sin(a), 1];
    }).flat();
    const [p] = keepCurves(h.app, h.store, [{ points: ring, closed: true }], "Ring");
    const s = simplifyPolylines(h.app, h.store, active(h), 0.05);
    expect(s).toMatchObject({ before: 400, unchanged: false });
    expect(s.after).toBeLessThan(60);
    const ball = addBall(h);
    h.app.removeMesh(h.store, ball);
    const r = resamplePolylines(h.app, h.store, active(h), { spacing: 1, smoothing: 0 });
    expect(r.after).toBe(63);
    expect(h.app.getSnapshot().polylines).toHaveLength(1);
    expect(h.app.getSnapshot().polylines[0]).toMatchObject({ id: p.id, name: "Ring" });
    expect(describeEdit("Resample", r)).toBe(`Resample: ${s.after} points → 63.`);
    h.store.undo();
    expect(pointCount(h.app.getSnapshot().polylines[0])).toBe(s.after);
    expect(items(h)).toEqual([]);
    h.store.undo(); // the ball's removal
    expect(items(h)).toEqual(["Ball"]);
    expect(pointCount(h.app.getSnapshot().polylines[0])).toBe(s.after);
    h.store.undo();
    expect(h.app.getSnapshot().polylines[0].points).toBe(p.points);
    h.store.redo();
    h.store.redo();
    h.store.redo();
    expect(pointCount(h.app.getSnapshot().polylines[0])).toBe(63);
    expect(h.app.getSnapshot().polylineSelection).toEqual([p.id]);
  });

  it("makes no undo step of an edit that changes nothing", async () => {
    const h = await local();
    keepCurves(h.app, h.store, [SQUARE], "Square");
    const c = simplifyPolylines(h.app, h.store, active(h), 0.01);
    expect(c.unchanged).toBe(true);
    expect(describeEdit("Simplify", c)).toBe("Simplify: nothing to change.");
    h.store.undo();
    expect(names(h)).toEqual([]);
    expect(() => simplifyPolylines(h.app, h.store, active(h), 0)).toThrow("above zero");
  });

  it("removes as one undo step, and leaves display settings out of undo", async () => {
    const h = await local();
    const kept = keepCurves(h.app, h.store, [SQUARE, LINE, SQUARE], "Cut");
    h.app.setPolylineDisplay([kept[0].id], { name: "Outline", color: 0x123456, visible: false });
    removePolylines(h.app, h.store, [kept[1].id, kept[2].id]);
    expect(names(h)).toEqual(["Outline"]);
    expect(h.app.getSnapshot().polylineSelection).toEqual([kept[0].id]);
    h.app.setPolylineDisplay([kept[0].id], { name: "Edge" });
    h.store.undo();
    expect(h.app.getSnapshot().polylines.map((p) => [p.name, p.visible])).toEqual([
      ["Edge", false],
      ["Cut 2", true],
      ["Cut 3", true],
    ]);
    expect(h.app.getSnapshot().polylines[0].color).toBe(0x123456);
    // Nothing to remove: no step.
    h.store.redo();
    removePolylines(h.app, h.store, [kept[1].id]);
    h.store.undo();
    expect(names(h)).toHaveLength(3);
  });

  it("is part of an open step, and goes with it when the step is cancelled", async () => {
    const h = await local();
    keepCurves(h.app, h.store, [SQUARE], "Square");
    h.store.beginStep();
    keepCurves(h.app, h.store, [LINE], "Line");
    addBall(h);
    expect(names(h)).toEqual(["Square", "Line"]);
    h.store.cancelStep();
    h.app.syncMeshes(h.store);
    expect(names(h)).toEqual(["Square"]);
    h.store.undo();
    expect(names(h)).toEqual([]);
  });

  it("keeps no state the history cannot bring back", async () => {
    const h = await local();
    const before = h.store.getSnapshot().meshes.size;
    for (let k = 0; k < 20; k++) {
      keepCurves(h.app, h.store, [LINE], "Line");
    }
    // One marker, however many edits.
    expect(h.store.getSnapshot().meshes.size).toBe(before + 1);
    for (let k = 0; k < 20; k++) {
      h.store.undo();
    }
    expect(names(h)).toEqual([]);
    keepCurves(h.app, h.store, [SQUARE], "Square"); // drops the redo steps
    h.store.redo();
    expect(names(h)).toEqual(["Square"]);
  });

  it("offsets into new polylines of the same colour, and says why it cannot", async () => {
    const h = await local();
    const [square] = keepCurves(h.app, h.store, [SQUARE], "Square");
    const [out] = offsetPolylines(h.app, h.store, active(h), 1, "sharp", "mm");
    expect(out).toMatchObject({ name: "Square offset", color: square.color, closed: true });
    expect(curveLength(out)).toBeCloseTo(48, 9);
    expect(active(h)).toEqual([out]);
    h.app.selectPolylines([square.id]);
    expect(offsetPolylines(h.app, h.store, active(h), -1, "round", "mm")[0].name).toBe("Square offset 2");
    h.app.selectPolylines([square.id]);
    expect(() => offsetPolylines(h.app, h.store, active(h), -6, "round", "in")).toThrow("Nothing is left at 0.236 in");
    expect(() => offsetPolylines(h.app, h.store, active(h), 0, "round", "mm")).toThrow("not be zero");
    expect(names(h)).toEqual(["Square", "Square offset", "Square offset 2"]);
    h.store.undo();
    h.store.undo();
    expect(names(h)).toEqual(["Square"]);
    expect(planarProblemText({ reason: "bent", deviation: 25.4 }, "Path", "in")).toBe(
      '"Path" is not planar: it strays 1 in from its plane.',
    );
    expect(planarProblemText({ reason: "points", points: 60_000 }, "Path", "mm")).toContain("60,000 points");
  });
});

describe.each([
  ["local", local],
  ["worker", worker],
] as const)("meshes from polylines (%s)", (_mode, make) => {
  it("adds closed tubes, one undo step for all of them", async () => {
    const h = await make();
    const ball = addBall(h);
    await h.settle();
    const kept = keepCurves(h.app, h.store, [SQUARE, LINE], "Cut");
    const r = tubePolylines(h.app, h.store, kept, 0.5, 12);
    await h.settle();
    await r.done;
    expect(r.items.map((i) => i.name)).toEqual(["Cut 1 tube", "Cut 2 tube"]);
    expect(r.triangles).toBe(2 * 12 * 4 + 2 * 12 * 2 + 2 * 12);
    expect(r.folds).toBe(0);
    for (const item of r.items) {
      const a = await h.scene.analyze(item.mesh);
      expect(a.loops).toHaveLength(0);
      expect(a.nonManifoldEdges).toBe(0);
      expect(a.orientationConflicts).toBe(0);
    }
    expect(h.store.getSnapshot().selection).toBe(r.items[1].mesh);
    // The tubes stand where the curves are.
    expect(r.items[0].center.map((x) => Math.round(x * 1e3) / 1e3)).toEqual([5, 5, 0]);
    h.store.undo();
    await h.settle();
    expect(items(h)).toEqual(["Ball"]);
    expect(names(h)).toEqual(["Cut 1", "Cut 2"]);
    h.store.redo();
    await h.settle();
    expect(items(h)).toEqual(["Ball", "Cut 1 tube", "Cut 2 tube"]);
    void ball;
  });

  it("extrudes closed planar curves, holes included, as one undo step", async () => {
    const h = await make();
    const hole = { points: [3, 3, 0, 7, 3, 0, 7, 7, 0, 3, 7, 0], closed: true };
    const high = { points: SQUARE.points.map((v, i) => (i % 3 === 2 ? v + 20 : v)), closed: true };
    const kept = keepCurves(h.app, h.store, [SQUARE, hole, high], "Cut");
    const r = extrudePolylines(h.app, h.store, kept, 2, "one", "mm");
    await h.settle();
    await r.done;
    expect(r.items.map((i) => i.name)).toEqual(["Cut 1 extrusion", "Cut 3 extrusion"]);
    const volumes: (number | null)[] = [];
    for (const item of r.items) {
      const a = await h.scene.analyze(item.mesh);
      expect(a.loops).toHaveLength(0);
      expect(a.nonManifoldEdges).toBe(0);
      volumes.push((await h.scene.inspect.measure(item.mesh)).volume);
    }
    expect(volumes[0]).toBeCloseTo(2 * (100 - 16), 3);
    expect(volumes[1]).toBeCloseTo(200, 3);
    h.store.undo();
    await h.settle();
    expect(items(h)).toEqual([]);
    h.store.redo();
    await h.settle();
    expect(items(h)).toHaveLength(2);
  });

  it("follows undo and redo of the list itself", async () => {
    const h = await make();
    keepCurves(h.app, h.store, [SQUARE], "Square");
    keepCurves(h.app, h.store, [LINE], "Line");
    const ball = addBall(h);
    h.app.removeMesh(h.store, ball);
    await h.settle();
    removePolylines(h.app, h.store, [h.app.getSnapshot().polylines[0].id]);
    await h.settle();
    expect(names(h)).toEqual(["Line"]);
    for (const want of [["Square", "Line"], ["Square", "Line"], ["Square"], []]) {
      h.store.undo();
      await h.settle();
      expect(names(h)).toEqual(want);
    }
    for (const want of [["Square"], ["Square", "Line"], ["Square", "Line"], ["Line"]]) {
      h.store.redo();
      await h.settle();
      expect(names(h)).toEqual(want);
    }
    expect(items(h)).toEqual([]);
  });
});

describe("the engine's restart", () => {
  it("leaves the polylines, and edits after it are undoable", async () => {
    const h = await worker();
    addBall(h);
    keepCurves(h.app, h.store, [SQUARE], "Square");
    removePolylines(h.app, h.store, [h.app.getSnapshot().polylines[0].id]);
    keepCurves(h.app, h.store, [LINE], "Line");
    await h.settle();
    await h.store.restart();
    await h.settle();
    expect(names(h)).toEqual(["Line"]);
    keepCurves(h.app, h.store, [SQUARE], "Square");
    await h.settle();
    expect(names(h)).toEqual(["Line", "Square"]);
    h.store.undo();
    await h.settle();
    expect(names(h)).toEqual(["Line"]);
    h.store.redo();
    await h.settle();
    expect(names(h)).toEqual(["Line", "Square"]);
    expect(items(h)).toEqual(["Ball"]);
  });
});

describe("what cannot be a mesh", () => {
  it("is refused before anything changes", async () => {
    const h = await local();
    const bent = { points: [0, 0, 0, 5, 0, 0, 5, 5, 2, 0, 5, 0], closed: true };
    const kept = keepCurves(h.app, h.store, [SQUARE, LINE, bent], "Cut");
    expect(() => extrudePolylines(h.app, h.store, kept, 2, "one", "mm")).toThrow('"Cut 2" is open');
    expect(() => extrudePolylines(h.app, h.store, [kept[0], kept[2]], 2, "one", "mm")).toThrow('"Cut 3" is not planar');
    expect(() => extrudePolylines(h.app, h.store, [kept[0]], 0, "one", "mm")).toThrow("not be zero");
    expect(() => tubePolylines(h.app, h.store, kept, 0, 12)).toThrow("radius");
    expect(() => tubePolylines(h.app, h.store, kept, 1, 1000)).toThrow("sides");
    expect(items(h)).toEqual([]);
    h.store.undo();
    expect(names(h)).toEqual([]);
  });
});

describe("selection", () => {
  it("takes one at a time, or more with shift; the only polyline needs none", async () => {
    expect(clickedSelection([], 3, false)).toEqual([3]);
    expect(clickedSelection([3], 3, false)).toEqual([]);
    expect(clickedSelection([3, 4], 3, false)).toEqual([3]);
    expect(clickedSelection([3], 4, true)).toEqual([3, 4]);
    expect(clickedSelection([3, 4], 3, true)).toEqual([4]);
    const h = await local();
    const [a] = keepCurves(h.app, h.store, [SQUARE], "Square");
    h.app.selectPolylines([]);
    expect(active(h)).toEqual([a]);
    keepCurves(h.app, h.store, [LINE], "Line");
    h.app.selectPolylines([]);
    expect(active(h)).toEqual([]);
  });

  it("reads colours as a colour input gives them", () => {
    expect(cssColor(0x00ff12)).toBe("#00ff12");
    expect(parseColor("#00ff12")).toBe(0x00ff12);
    expect(parseColor("red")).toBeNull();
  });
});

describe("defaults", () => {
  it("are round shares of the curves' size", () => {
    expect(defaultLengths(100, 314)).toEqual({ tolerance: 0.1, spacing: 2, smoothing: 0, offset: 2, radius: 1, depth: 10 });
  });
});

describe("files", () => {
  const square = makeCurve(SQUARE.points, true)!;
  const tilted = makeCurve([0, 0, 0, 10, 0, 0, 10, 0, 10, 0, 0, 10], true)!;

  it("draw planar curves in their own plane, true to size", () => {
    const svg = polylinesFile([tilted], "svg", "fit");
    expect(svg.flattened).toBe(0);
    expect(svg.text.match(/<polygon/g)).toHaveLength(1);
    const pts = [...svg.text.matchAll(/(-?[\d.]+),(-?[\d.]+)/g)].map((m) => [Number(m[1]), Number(m[2])]);
    expect(Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])).toBeCloseTo(10, 4);
    expect(Math.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])).toBeCloseTo(10, 4);
    const dxf = polylinesFile([square, makeCurve(LINE.points, false)!], "dxf", "z");
    expect(dxf.text.match(/LWPOLYLINE/g)).toHaveLength(2);
    expect(dxf.text).toContain("\n70\n1\n");
    expect(dxf.text).toContain("\n70\n0\n");
    expect(dxf.text).toContain("\n10\n10\n20\n10\n");
  });

  it("flatten the rest, and say by how much", () => {
    const top = polylinesFile([tilted], "svg", "z");
    expect(top.flattened).toBeCloseTo(10, 9);
    const apart = polylinesFile([square, makeCurve(SQUARE.points.map((v, i) => (i % 3 === 2 ? 7 : v)), true)!], "dxf", "z");
    expect(apart.flattened).toBeCloseTo(7, 9);
    expect(polylinesFile([square], "svg", "z").flattened).toBe(0);
    expect(polylinesFile([], "svg", "fit").text).toContain("<svg");
  });
});

describe("projects", () => {
  it("keep polylines bit for bit, and replace the ones there", async () => {
    const h = await local();
    addBall(h);
    const kept = keepCurves(h.app, h.store, [{ points: SQUARE.points.map((v) => v / 3), closed: true }, LINE], "Cut");
    h.app.setPolylineDisplay([kept[1].id], { name: "Edge", color: 0x00aa55, visible: false });
    const doc = parseProjectDoc(JSON.stringify(collectProject(h.store, h.app)));
    expect(doc.polylines).toHaveLength(2);
    const saved = h.app.getSnapshot().polylines;
    const g = await local();
    keepCurves(g.app, g.store, [LINE], "Old");
    restoreProject(g.store, g.app, doc);
    const now = g.app.getSnapshot().polylines;
    expect(now.map((p) => [p.name, p.color, p.visible, p.closed])).toEqual(saved.map((p) => [p.name, p.color, p.visible, p.closed]));
    expect([...now[0].points]).toEqual([...saved[0].points]);
    expect(g.store.getSnapshot().history.canUndo).toBe(false);
    // The marker of the old list is gone; new edits are undoable from here.
    expect(g.store.getSnapshot().meshes.size).toBe(1);
    removePolylines(g.app, g.store, [now[0].id]);
    g.store.undo();
    expect(names(g)).toEqual(["Cut 1", "Edge"]);
  });

  it("open without polylines, with unreadable ones, or with nothing else", async () => {
    const h = await local();
    addBall(h);
    const plain = JSON.parse(JSON.stringify(collectProject(h.store, h.app))) as Record<string, unknown>;
    expect("polylines" in plain).toBe(false);
    expect(parseProjectDoc(JSON.stringify(plain)).polylines).toBeUndefined();
    const some = parseProjectDoc(
      JSON.stringify({
        ...plain,
        polylines: [{ points: "!!" }, null, { points: 3 }, { name: "One", points: "AAAAAAAAAAA=" }],
      }),
    );
    expect(some.polylines).toBeUndefined();
    const only = await local();
    keepCurves(only.app, only.store, [SQUARE], "Square");
    const doc = parseProjectDoc(JSON.stringify(collectProject(only.store, only.app)));
    expect(doc.meshes).toEqual([]);
    restoreProject(h.store, h.app, doc);
    expect(items(h)).toEqual([]);
    expect(names(h)).toEqual(["Square"]);
  });
});
