// @vitest-environment jsdom
// Real-WASM integration and UI tests for Issue #193 (Free-Form Deformation & Self-Collision).
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import type { FreeformProtection } from "fast-mesh-wasm";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import * as THREE from "three";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { DeformPanel, type DeformPanelProps } from "../src/features/sculpt/ui/DeformPanel";
import { AppState } from "../src/app/state/app-state";
import { DeformHandlesLayer, type DeformHandlesLayerProps } from "../src/features/sculpt/ui/DeformHandlesLayer";
import { buildFreeformProtectionFromMask, buildProtectionOption } from "../src/features/sculpt/ux/deform-protection";
import {
  applyDisplacementToLattice,
  buildLatticeEdgeSegments,
  computeLatticeControlPoints,
  ffdControlPointIndex,
  moveFfdControlPoint,
  resetLatticeOffsets,
} from "../src/features/sculpt/ux/ffd-lattice";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function resolveWasmBytes(): Uint8Array {
  const candidates = [
    path.resolve(process.cwd(), "../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"),
    path.resolve(process.cwd(), "packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"),
  ];
  for (const p of candidates) {
    if (existsSync(p)) {
      return readFileSync(p);
    }
  }
  throw new Error(`Could not locate fast_mesh_wasm_bg.wasm in: ${candidates.join(", ")}`);
}

function createSubdividedGrid(cols = 4, rows = 4, width = 2, height = 2): {
  positions: Float64Array;
  indices: Uint32Array;
} {
  const vxCount = cols + 1;
  const vyCount = rows + 1;
  const positions = new Float64Array(vxCount * vyCount * 3);
  let p = 0;
  for (let iy = 0; iy < vyCount; iy++) {
    const y = -height / 2 + (iy / rows) * height;
    for (let ix = 0; ix < vxCount; ix++) {
      const x = -width / 2 + (ix / cols) * width;
      positions[p++] = x;
      positions[p++] = y;
      positions[p++] = 0;
    }
  }
  const indices = new Uint32Array(cols * rows * 6);
  let k = 0;
  for (let iy = 0; iy < rows; iy++) {
    for (let ix = 0; ix < cols; ix++) {
      const v00 = iy * vxCount + ix;
      const v10 = v00 + 1;
      const v01 = v00 + vxCount;
      const v11 = v01 + 1;
      indices[k++] = v00;
      indices[k++] = v10;
      indices[k++] = v11;
      indices[k++] = v00;
      indices[k++] = v11;
      indices[k++] = v01;
    }
  }
  return { positions, indices };
}

function createDeformSetup() {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  const grid = createSubdividedGrid(4, 4, 2, 2);
  const item = app.addMesh(store, {
    name: "TestGrid",
    positions: grid.positions,
    indices: grid.indices,
    center: [0, 0, 0],
    radius: Math.SQRT2,
    volumeId: null,
    preset: null,
  });
  store.select(item.mesh);
  return { scene, store, app, item };
}

function readPos(store: SceneStore, mesh: ReturnType<Scene["addMesh"]>, vertex: number): [number, number, number] {
  const pos = store.scene.views(mesh).positions();
  const b = vertex * 3;
  return [pos[b], pos[b + 1], pos[b + 2]];
}

async function clickAndFlush(btn: HTMLElement) {
  await act(async () => {
    btn.click();
    await new Promise((resolve) => setTimeout(resolve, 25));
  });
}

describe("Issue #193 Deform UI & Viewport Helpers", () => {
  const cleanups: Array<() => void> = [];

  beforeAll(async () => {
    await init(resolveWasmBytes());
  });

  afterEach(() => {
    while (cleanups.length > 0) {
      cleanups.pop()!();
    }
  });

  function track(scene: Scene, store: SceneStore, root?: Root, container?: HTMLElement) {
    cleanups.push(() => {
      if (root) {
        act(() => {
          root.unmount();
        });
      }
      container?.remove();
      (store as { dispose?: () => void }).dispose?.();
      (scene as { free?: () => void }).free?.();
    });
  }

  it("exports expected component prop contracts and pure wireframe/displacement/protection helpers", () => {
    const _panelCheck: DeformPanelProps | null = null;
    const _layerCheck: DeformHandlesLayerProps | null = null;
    expect(_panelCheck).toBeNull();
    expect(_layerCheck).toBeNull();
    expect(typeof DeformPanel).toBe("function");
    expect(typeof DeformHandlesLayer).toBe("function");

    const dims: [number, number, number] = [3, 4, 2];
    const total = dims[0] * dims[1] * dims[2];
    const lattice = {
      dims,
      basis: "bernstein" as const,
      frame: {
        center: [0, 0, 0] as [number, number, number],
        axes: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ] as [[number, number, number], [number, number, number], [number, number, number]],
        halfExtents: [1, 2, 0.5] as [number, number, number],
      },
      offsets: new Float64Array(total * 3),
    };

    const restPoints = computeLatticeControlPoints(lattice);
    expect(restPoints.length).toBe(total);
    expect(restPoints[ffdControlPointIndex(dims, 0, 0, 0)]).toEqual([-1, -2, -0.5]);
    expect(restPoints[ffdControlPointIndex(dims, 2, 3, 1)]).toEqual([1, 2, 0.5]);

    const expectedSegCount =
      (dims[0] - 1) * dims[1] * dims[2] +
      dims[0] * (dims[1] - 1) * dims[2] +
      dims[0] * dims[1] * (dims[2] - 1);
    const restSegs = buildLatticeEdgeSegments(lattice);
    expect(restSegs).toBeInstanceOf(Float32Array);
    expect(restSegs.length).toBe(expectedSegCount * 6);
    expect(Array.from(restSegs).every(Number.isFinite)).toBe(true);

    const moved = moveFfdControlPoint(lattice, 2, 3, 1, [0, 0.75, 0], "delta");
    expect(lattice.offsets[ffdControlPointIndex(dims, 2, 3, 1) * 3 + 1]).toBe(0);
    expect(moved.offsets[ffdControlPointIndex(dims, 2, 3, 1) * 3 + 1]).toBeCloseTo(0.75, 6);

    const movedAgain = applyDisplacementToLattice(moved, [2, 3, 1], [0.25, 0.5, 0], "delta");
    expect(movedAgain.offsets[ffdControlPointIndex(dims, 2, 3, 1) * 3 + 0]).toBeCloseTo(0.25, 6);
    expect(movedAgain.offsets[ffdControlPointIndex(dims, 2, 3, 1) * 3 + 1]).toBeCloseTo(1.25, 6);

    const movedSegs = buildLatticeEdgeSegments(movedAgain);
    expect(movedSegs.length).toBe(restSegs.length);
    const segChanged = movedSegs.some((val, idx) => Math.abs(val - restSegs[idx]) > 1e-4);
    expect(segChanged).toBe(true);

    const zeroed = resetLatticeOffsets(movedAgain);
    expect(Array.from(zeroed.offsets).every((v) => v === 0)).toBe(true);

    // Verify FreeformProtection critical semantics: weight = 0 frozen, 1 = full move.
    const fakeMesh = {} as ReturnType<Scene["addMesh"]>;
    const otherMesh = {} as ReturnType<Scene["addMesh"]>;
    const mask = { mesh: fakeMesh, vertices: Uint32Array.from([2, 5, 9]), slots: 25 };
    expect(buildFreeformProtectionFromMask(mask, otherMesh)).toBeUndefined();
    const maskProtect = buildFreeformProtectionFromMask(mask, fakeMesh);
    expect(maskProtect).toBeDefined();
    expect(Array.from(maskProtect!.vertices)).toEqual([2, 5, 9]);
    expect(Array.from(maskProtect!.weights!)).toEqual([0, 0, 0]);

    const combined = buildProtectionOption({
      mask,
      activeMesh: fakeMesh,
      protectMask: true,
      frozenVertices: [5, 11],
      weights: [1.0, 0.25],
    });
    expect(Array.from(combined!.vertices)).toEqual([2, 5, 9, 11]);
    // Vertex 5 overlapped mask (0 = frozen) and explicit (1 = full move); frozen 0 takes precedence.
    expect(Array.from(combined!.weights!)).toEqual([0, 0, 0, 0.25]);
  });

  it("builds FFD lattice, displaces control points, and applies FFD on real Scene/Store", async () => {
    const { scene, store, item } = createDeformSetup();
    track(scene, store);

    const topRightVertex = 24;
    const beforeTop = readPos(store, item.mesh, topRightVertex);
    const beforeBottom = readPos(store, item.mesh, 0);

    const domain = store.scene.deform!;
    let lattice = await domain.buildFfdLattice(item.mesh, [3, 3, 2], "bernstein", 0);
    expect(lattice.dims).toEqual([3, 3, 2]);

    // Move the upper plane j = 2 upward in +Y across all (i, k).
    for (let i = 0; i < 3; i++) {
      for (let k = 0; k < 2; k++) {
        lattice = moveFfdControlPoint(lattice, i, 2, k, [0, 0.6, 0], "delta");
      }
    }

    const report = await domain.applyFfd(item.mesh, { lattice, outsideMode: "clamp" });
    store.commitFrame();

    const afterTop = readPos(store, item.mesh, topRightVertex);
    const afterBottom = readPos(store, item.mesh, 0);
    expect(report.verticesMoved).toBeGreaterThan(0);
    expect(report.maxDisplacement).toBeGreaterThan(0.3);
    expect(afterTop[1]).toBeGreaterThan(beforeTop[1] + 0.3);
    expect(afterBottom[1]).toBeCloseTo(beforeBottom[1], 5);
  });

  it("applies ARAP honoring FreeformProtection weights (0 = frozen, 1 = full move)", async () => {
    const { scene, store, item } = createDeformSetup();
    track(scene, store);

    const anchorVertex = 0;
    const frozenVertex = 20;
    const movableVertex = 22;
    const handleVertex = 24;

    const beforeFrozen = readPos(store, item.mesh, frozenVertex);
    const beforeMovable = readPos(store, item.mesh, movableVertex);
    const beforeHandle = readPos(store, item.mesh, handleVertex);

    const protect: FreeformProtection = {
      vertices: Uint32Array.from([frozenVertex, movableVertex]),
      weights: Float32Array.from([0, 1]),
    };

    const report = await store.scene.deform!.applyArap(item.mesh, {
      anchors: [{ vertex: anchorVertex }],
      handles: [{ vertex: handleVertex, target: [beforeHandle[0], beforeHandle[1] + 0.5, beforeHandle[2]] }],
      weights: "cotangent",
      iterations: 20,
      protect,
    });
    store.commitFrame();

    const afterFrozen = readPos(store, item.mesh, frozenVertex);
    const afterMovable = readPos(store, item.mesh, movableVertex);
    const afterHandle = readPos(store, item.mesh, handleVertex);

    expect(report.verticesMoved).toBeGreaterThan(0);
    expect(
      Math.hypot(afterFrozen[0] - beforeFrozen[0], afterFrozen[1] - beforeFrozen[1], afterFrozen[2] - beforeFrozen[2]),
    ).toBeLessThan(1e-5);
    expect(
      Math.hypot(
        afterMovable[0] - beforeMovable[0],
        afterMovable[1] - beforeMovable[1],
        afterMovable[2] - beforeMovable[2],
      ),
    ).toBeGreaterThan(1e-3);
    expect(afterHandle[1]).toBeGreaterThan(beforeHandle[1] + 0.25);
  });

  it("applies Kelvinlet and XPBD (with selfCollision toggle) cleanly and commits frames", async () => {
    const { scene, store, item } = createDeformSetup();
    track(scene, store);

    const centerVertex = 12;
    const beforeCenter = readPos(store, item.mesh, centerVertex);
    const kReport = await store.scene.deform!.applyKelvinlet(item.mesh, {
      center: [0, 0, 0],
      load: { kind: "grab", force: [0, 0.4, 0] },
      epsilon: 0.7,
      nu: 0.35,
      scale: "bi",
    });
    store.commitFrame();

    const afterKelvin = readPos(store, item.mesh, centerVertex);
    expect(kReport.verticesMoved).toBeGreaterThan(0);
    expect(afterKelvin[1]).toBeGreaterThan(beforeCenter[1]);

    // Pin top row (vertices 20..24), run XPBD with selfCollision enabled then disabled.
    const pins = [20, 21, 22, 23, 24].map((v) => ({
      vertex: v,
      target: readPos(store, item.mesh, v),
    }));

    const xReportOn = await store.scene.deform!.applyXpbd(item.mesh, {
      substeps: 4,
      iterations: 6,
      stretchCompliance: 0.0,
      bendCompliance: 0.02,
      damping: 0.02,
      gravity: [0, -0.5, 0],
      pins,
      selfCollision: true,
      collisionThickness: 0.05,
    });
    store.commitFrame();
    expect(xReportOn.verticesMoved).toBeGreaterThan(0);
    const afterXpbdOn = readPos(store, item.mesh, 2);
    expect(afterXpbdOn[1]).toBeLessThan(afterKelvin[1]);

    const xReportOff = await store.scene.deform!.applyXpbd(item.mesh, {
      substeps: 2,
      iterations: 4,
      gravity: [0, -0.3, 0],
      pins,
      selfCollision: false,
    });
    store.commitFrame();
    expect(xReportOff.verticesMoved).toBeGreaterThan(0);
    expect(xReportOff.selfCollisionContactsResolved).toBe(0);
  });

  it("drives DeformPanel DOM interactions across FFD, ARAP, Kelvinlets, XPBD, and mask protection", async () => {
    const { scene, store, app, item } = createDeformSetup();
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    track(scene, store, root, container);

    act(() => {
      root.render(<DeformPanel app={app} store={store} />);
    });

    const getBtn = (label: string): HTMLButtonElement => {
      const btn = Array.from(container.querySelectorAll("button")).find(
        (b) => b.textContent?.trim() === label,
      );
      if (!btn) throw new Error(`Button "${label}" not found`);
      return btn;
    };

    // Toggle viewport tools safely.
    act(() => getBtn("FFD Cage").click());
    expect(app.getSnapshot().wave6.deform.activeTool).toBe("ffdCage");
    act(() => getBtn("FFD Cage").click());
    expect(app.getSnapshot().wave6.deform.activeTool).toBeNull();
    act(() => getBtn("Kelvinlet Drag").click());
    expect(app.getSnapshot().wave6.deform.activeTool).toBe("kelvinletDrag");
    act(() => getBtn("Off").click());
    expect(app.getSnapshot().wave6.deform.activeTool).toBeNull();

    // Freeze vertex 0 via SculptMask (weight = 0 = frozen).
    app.setSculptMask({
      mesh: item.mesh,
      vertices: Uint32Array.from([0]),
      slots: 25,
    });
    const beforeMasked0 = readPos(store, item.mesh, 0);
    const beforeTopCenter22 = readPos(store, item.mesh, 22);

    // Build FFD lattice, move default CP (1, 2, 1), and apply FFD via panel UI.
    await clickAndFlush(getBtn("Build Lattice"));
    expect(app.getSnapshot().wave6.deform.ffdLattice).not.toBeNull();

    act(() => getBtn("Move Control Point").click());
    const movedLattice = app.getSnapshot().wave6.deform.ffdLattice!;
    const cpIdx = ffdControlPointIndex(movedLattice.dims, 1, 2, 1);
    expect(movedLattice.offsets[cpIdx * 3 + 1]).toBeCloseTo(0.25, 6);

    await clickAndFlush(getBtn("Apply FFD"));
    const reportEl = container.querySelector('[data-testid="deform-report"]');
    expect(reportEl?.textContent).toContain("FFD:");
    const afterMasked0 = readPos(store, item.mesh, 0);
    const afterTopCenter22 = readPos(store, item.mesh, 22);
    expect(new THREE.Vector3(...afterMasked0).distanceTo(new THREE.Vector3(...beforeMasked0))).toBeLessThan(1e-6);
    expect(afterTopCenter22[1]).toBeGreaterThan(beforeTopCenter22[1] + 0.04);

    // Switch to Kelvinlets tab & apply.
    act(() => getBtn("Kelvinlets").click());
    expect(app.getSnapshot().wave6.deform.algorithm).toBe("kelvinlet");
    act(() => getBtn("Center").click());
    await clickAndFlush(getBtn("Apply Kelvinlet"));
    expect(app.getSnapshot().wave6.deform.lastReport).toContain("Kelvinlet:");

    // Switch to XPBD tab, toggle self-collision checkbox, and apply.
    act(() => getBtn("XPBD").click());
    expect(app.getSnapshot().wave6.deform.algorithm).toBe("xpbd");
    const selfColInput = container.querySelector('input[aria-label="Self-collision"]') as HTMLInputElement;
    expect(selfColInput.checked).toBe(true);
    act(() => {
      selfColInput.click();
    });
    expect(app.getSnapshot().wave6.deform.xpbdSelfCollision).toBe(false);
    act(() => {
      selfColInput.click();
    });
    expect(app.getSnapshot().wave6.deform.xpbdSelfCollision).toBe(true);

    await clickAndFlush(getBtn("Apply XPBD"));
    expect(app.getSnapshot().wave6.deform.lastReport).toContain("XPBD:");

    // Switch to ARAP tab, pin base, add handle, and apply.
    act(() => getBtn("ARAP").click());
    expect(app.getSnapshot().wave6.deform.algorithm).toBe("arap");
    act(() => getBtn("Pin Base").click());
    expect(app.getSnapshot().wave6.deform.arapAnchors.length).toBeGreaterThan(0);
    act(() => getBtn("Add Handle (+Delta)").click());
    expect(app.getSnapshot().wave6.deform.arapHandles.length).toBeGreaterThan(0);
    await clickAndFlush(getBtn("Apply ARAP"));
    expect(app.getSnapshot().wave6.deform.lastReport).toContain("ARAP:");
  });
});
