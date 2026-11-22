// Real Wasm Scene + Showcase engine/store integration tests for #210 & #212.
import { Scene, type DyntopoMode, type SculptBrushOp } from "fast-mesh-wasm";
import { createSceneStore } from "fast-mesh-wasm/store";
import * as THREE from "three";
import { afterEach, describe, expect, it } from "vitest";
import { SETTINGS_VERSION, parseSettings } from "../src/shared/ux/settings";
import { AppState } from "../src/app/state/app-state";
import { makeBrushCursor, setBrushDyntopoIndicator } from "../src/features/sculpt/ui/brush-cursor";
import { MaskLiveTracker } from "../src/features/sculpt/ux/mask-protect";
import {
  applyDyntopoRoi,
  buildDyntopoOptions,
  dyntopoCursorRatio,
  sculptBrush,
  SculptStroke,
} from "../src/features/sculpt/ux/sculpt";

function openCylinder(slices = 12, radius = 1) {
  const positions: number[] = [];
  for (let r = 0; r < 2; r++) {
    const z = r === 0 ? -0.5 : 0.5;
    for (let j = 0; j < slices; j++) {
      const a = (2 * Math.PI * j) / slices;
      positions.push(radius * Math.cos(a), radius * Math.sin(a), z);
    }
  }
  const indices: number[] = [];
  for (let j = 0; j < slices; j++) {
    const a = j;
    const b = (j + 1) % slices;
    const c = slices + b;
    const d = slices + a;
    indices.push(a, b, c, a, c, d);
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

function twoRings(slices = 10) {
  const positions: number[] = [];
  const indices: number[] = [];
  for (const zPair of [
    [-1.2, -0.6],
    [0.6, 1.2],
  ]) {
    const off = positions.length / 3;
    for (const z of zPair) {
      for (let j = 0; j < slices; j++) {
        const a = (2 * Math.PI * j) / slices;
        positions.push(Math.cos(a), Math.sin(a), z);
      }
    }
    for (let j = 0; j < slices; j++) {
      const a = off + j;
      const b = off + ((j + 1) % slices);
      const c = off + slices + ((j + 1) % slices);
      const d = off + slices + j;
      indices.push(a, b, c, a, c, d);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

function flatGrid(nx = 8, ny = 8, size = 2) {
  const positions: number[] = [];
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      positions.push((x / (nx - 1) - 0.5) * size, (y / (ny - 1) - 0.5) * size, 0);
    }
  }
  const indices: number[] = [];
  for (let y = 0; y < ny - 1; y++) {
    for (let x = 0; x < nx - 1; x++) {
      const a = y * nx + x;
      indices.push(a, a + 1, a + nx + 1, a, a + nx + 1, a + nx);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

describe("Showcase Wave 5 Wasm Scene integration (#210 & #212)", () => {
  const cleanups: (() => void)[] = [];

  afterEach(() => {
    for (const fn of cleanups.splice(0).reverse()) fn();
  });

  function setup() {
    const scene = new Scene();
    const store = createSceneStore(scene);
    const app = new AppState();
    cleanups.push(() => {
      store.dispose();
    });
    return { scene, store, app };
  }

  it("persists dyntopo settings, scales mode options cleanly, and updates cursor detail indicator", () => {
    const parsed = parseSettings({
      version: SETTINGS_VERSION,
      sculptDyntopo: true,
      sculptDyntopoMode: "constant-detail",
      sculptDyntopoDetail: 0.15,
    });
    expect(parsed.sculptDyntopo).toBe(true);
    expect(parsed.sculptDyntopoMode).toBe("constant-detail");
    expect(parsed.sculptDyntopoDetail).toBeCloseTo(0.15);

    const modes: DyntopoMode[] = ["brush-relative", "constant-detail", "split-collapse", "split-only"];
    for (const mode of modes) {
      const opts = buildDyntopoOptions(mode, 0.2, 10);
      expect(opts.mode).toBe(mode);
      if (mode === "brush-relative") {
        expect(opts.detailSize).toBeCloseTo(0.2);
      } else {
        expect(opts.detailSize).toBeCloseTo(2.0);
      }
      expect(opts.smoothTangent).toBe(true);
    }

    const cursor = makeBrushCursor(0x3b82f6);
    const ring = cursor.getObjectByName("dyntopo-detail-ring") as THREE.Mesh;
    expect(ring.visible).toBe(false);

    const ratio = dyntopoCursorRatio("brush-relative", 0.3, 1, 0.5);
    expect(ratio).toBeCloseTo(0.3);
    setBrushDyntopoIndicator(cursor, ratio);
    expect(ring.visible).toBe(true);
    expect(ring.scale.x).toBeCloseTo(0.3);

    setBrushDyntopoIndicator(cursor, null);
    expect(ring.visible).toBe(false);
  });

  it("runs a real SculptStroke with dyntopoDab inside beginStep/endStep with single undo & mask tracker sync", async () => {
    const { scene, store, app } = setup();
    const grid = flatGrid(8, 8, 2);
    const item = app.addMesh(store, {
      name: "grid",
      positions: grid.positions,
      indices: grid.indices,
      center: [0, 0, 0],
      radius: 1.4,
      volumeId: null,
      preset: null,
    });
    const tracker = new MaskLiveTracker(scene, store);
    tracker.attach();
    cleanups.push(() => tracker.dispose());

    const beforeTris = scene.inspect.triangleQuality(item.mesh).liveTriangleCount;

    const opts = buildDyntopoOptions("constant-detail", 0.08, item.radius);
    const stroke = new SculptStroke(
      () => sculptBrush("inflate", item.mesh, 0.45, 0.35, "smooth"),
      {
        beginStep: () => store.beginStep(),
        endStep: () => store.endStep(),
        cancelStep: () => store.cancelStep(),
      },
      async (ops) => {
        for (const op of ops) {
          await scene.dyntopoDab(item.mesh, op as SculptBrushOp, opts);
        }
      },
    );

    stroke.dab([0, 0, 0], true);
    await stroke.flush();
    stroke.dab([0.1, 0, 0], true);
    await stroke.flush();
    stroke.end();
    await stroke.done;

    const afterTris = scene.inspect.triangleQuality(item.mesh).liveTriangleCount;
    expect(afterTris).toBeGreaterThan(beforeTris);
    expect(store.getSnapshot().history.canUndo).toBe(true);

    store.undo();
    expect(scene.inspect.triangleQuality(item.mesh).liveTriangleCount).toBe(beforeTris);

    // Also exercise one-click applyDyntopoRoi helper
    await applyDyntopoRoi({
      scene,
      store,
      target: item,
      brushPick: { mesh: item.mesh, point: [0, 0, 0] },
      mask: null,
      mode: "split-only",
      detail: 0.1,
      brushRadius: 0.5,
    });
    expect(scene.inspect.triangleQuality(item.mesh).liveTriangleCount).toBeGreaterThan(beforeTris);
    store.undo();
    expect(scene.inspect.triangleQuality(item.mesh).liveTriangleCount).toBe(beforeTris);
  });

  it("fills open holes with G0/G1/G2 continuity and bridges partial arcs cleanly with SceneStore undo/redo", () => {
    const { scene, store, app } = setup();
    const cyl = openCylinder(12);
    const item = app.addMesh(store, {
      name: "cyl",
      positions: cyl.positions,
      indices: cyl.indices,
      center: [0, 0, 0],
      radius: 1.2,
      volumeId: null,
      preset: null,
    });
    const before = scene.inspect.triangleQuality(item.mesh).liveTriangleCount;

    for (const continuity of ["flat", "tangent", "curvature"] as const) {
      store.beginStep();
      const stats = scene.fillAllHolesContinuity(item.mesh, {
        continuity,
        refinePatch: true,
        detectIslands: true,
      });
      store.endStep();
      expect(stats.continuity).toBe(continuity);
      expect(stats.filledTriangles).toBeGreaterThan(0);
      expect(scene.inspect.triangleQuality(item.mesh).liveTriangleCount).toBe(before + stats.filledTriangles);

      store.undo();
      expect(scene.inspect.triangleQuality(item.mesh).liveTriangleCount).toBe(before);
    }

    const rings = twoRings(10);
    const rItem = app.addMesh(store, {
      name: "rings",
      positions: rings.positions,
      indices: rings.indices,
      center: [0, 0, 0],
      radius: 1.6,
      volumeId: null,
      preset: null,
    });
    const beforeRings = scene.inspect.triangleQuality(rItem.mesh).liveTriangleCount;
    const analysis = scene.analyze(rItem.mesh);
    const pos = scene.views(rItem.mesh).positions();
    const loopA = analysis.loops.find((l) => Math.abs(pos[l.vertices[0]! * 3 + 2]! + 0.6) < 0.05)!;
    const loopB = analysis.loops.find((l) => Math.abs(pos[l.vertices[0]! * 3 + 2]! - 0.6) < 0.05)!;

    store.beginStep();
    const bStats = scene.bridgePartialArcs(rItem.mesh, loopA.vertices, loopB.vertices, {
      arcARange: [0, 3],
      arcBRange: [0, 3],
      twistOffset: 0,
      subdivisions: 2,
    });
    store.endStep();
    expect(bStats.trianglesAdded).toBeGreaterThan(0);
    expect(scene.inspect.triangleQuality(rItem.mesh).liveTriangleCount).toBe(beforeRings + bStats.trianglesAdded);

    store.undo();
    expect(scene.inspect.triangleQuality(rItem.mesh).liveTriangleCount).toBe(beforeRings);
  });
});
