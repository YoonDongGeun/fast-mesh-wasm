// @vitest-environment jsdom
// UI & DOM interaction tests for #210 (G0/G1/G2 hole fill & partial arc bridge)
// and #212 (Dynamic Topology sculpt toolbox, inspector, and ROI controls).
import type { Analysis, LoopId, MeshHandle, Scene } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore } from "fast-mesh-wasm/store";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RepairPanel } from "../src/panels/RepairPanel";
import { SculptInspectorCard } from "../src/panels/Ribbon";
import { AppState } from "../src/scene/app-state";
import { selectMesh } from "../src/scene/selection";
import { SculptToolbox } from "../src/viewport/Toolbox";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const cleanups: (() => Promise<void> | void)[] = [];

afterEach(async () => {
  vi.restoreAllMocks();
  for (const fn of cleanups.splice(0).reverse()) {
    await fn();
  }
});

function setNativeValue(el: HTMLInputElement | HTMLSelectElement, value: string): void {
  const proto = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
  if (setter) {
    setter.call(el, value);
  } else {
    el.value = value;
  }
  el.dispatchEvent(new Event(el instanceof HTMLSelectElement ? "change" : "input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

function makeFakeAnalysis(): Analysis {
  return {
    loops: [
      {
        id: { minVertex: 0, len: 6 } as unknown as LoopId,
        vertices: new Uint32Array([0, 1, 2, 3, 4, 5]),
        perimeter: 6,
        area: 1,
        closed: true,
        clean: true,
      },
      {
        id: { minVertex: 10, len: 8 } as unknown as LoopId,
        vertices: new Uint32Array([10, 11, 12, 13, 14, 15, 16, 17]),
        perimeter: 8,
        area: 2,
        closed: true,
        clean: true,
      },
    ],
    components: [{ triangles: new Uint32Array([0, 1, 2, 3]), closed: false, clean: true }],
    orientationConflicts: 0,
    nonManifoldEdges: 0,
    nonManifoldVertices: new Uint32Array(),
    pinchVertices: new Uint32Array(),
    degenerate: 0,
    duplicate: 0,
    nonManifoldEdgeList: new Uint32Array(),
    degenerateTriangles: new Uint32Array(),
    duplicateTriangles: new Uint32Array(),
    selfIntersects: null,
  } as unknown as Analysis;
}

function createMockScene() {
  let nextId = 0;
  let undoDepth = 0;
  let stepDepth = 0;
  const fillHoleContinuity = vi.fn((..._args: unknown[]) => {
    if (stepDepth === 0) undoDepth++;
    return {
      loopIndex: 0,
      filledVertices: 1,
      filledTriangles: 6,
      perimeter: 6,
      maxDihedralAngleDeg: 1.5,
      continuity: "tangent" as const,
      islandLoopsMerged: 0,
      converged: true,
    };
  });
  const fillAllHolesContinuity = vi.fn((..._args: unknown[]) => {
    if (stepDepth === 0) undoDepth++;
    return {
      loopIndex: 0,
      filledVertices: 3,
      filledTriangles: 14,
      perimeter: 14,
      maxDihedralAngleDeg: 0.8,
      continuity: "curvature" as const,
      islandLoopsMerged: 1,
      converged: true,
    };
  });
  const bridgePartialArcs = vi.fn((..._args: unknown[]) => {
    if (stepDepth === 0) undoDepth++;
    return {
      trianglesAdded: 8,
      verticesAdded: 2,
      resultingLoops: 2,
      arcAVertices: 4,
      arcBVertices: 4,
    };
  });
  const applyDyntopo = vi.fn(() => {
    if (stepDepth === 0) undoDepth++;
    return {
      splits: 4,
      collapses: 1,
      flips: 2,
      smoothed: 4,
      newVertices: new Uint32Array([20, 21]),
      removedVertices: new Uint32Array(),
    };
  });
  const beginStep = vi.fn(() => {
    stepDepth++;
  });
  const endStep = vi.fn(() => {
    if (stepDepth > 0) {
      stepDepth--;
      if (stepDepth === 0) undoDepth++;
    }
  });
  const cancelStep = vi.fn(() => {
    stepDepth = 0;
  });

  const scene = {
    addMesh: () => ({ __mesh: ++nextId }) as unknown as MeshHandle,
    removeMesh: () => {},
    onChanges: () => () => {},
    canUndo: () => undoDepth > 0,
    canRedo: () => false,
    undo: vi.fn(() => {
      if (undoDepth > 0) undoDepth--;
    }),
    redo: vi.fn(),
    analyze: vi.fn(() => makeFakeAnalysis()),
    views: () => ({
      positions: () => new Float64Array(60),
      indices: () => new Uint32Array([0, 1, 2]),
      normals: () => new Float32Array(60),
    }),
    fillHoleContinuity,
    fillAllHolesContinuity,
    bridgePartialArcs,
    applyDyntopo,
    beginStep,
    endStep,
    cancelStep,
  } as unknown as Scene;

  return {
    scene,
    spies: {
      fillHoleContinuity,
      fillAllHolesContinuity,
      bridgePartialArcs,
      applyDyntopo,
      beginStep,
      endStep,
    },
  };
}

function mount(element: React.ReactNode): { host: HTMLDivElement; root: Root } {
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  cleanups.push(async () => {
    await act(async () => {
      root.unmount();
    });
    host.remove();
  });
  act(() => {
    root.render(element);
  });
  return { host, root };
}

describe("Showcase Wave 5 UI controls (#210 & #212)", () => {
  it("renders Dyntopo controls in SculptToolbox & SculptInspectorCard and wires ROI application", async () => {
    const { scene, spies } = createMockScene();
    const store = createSceneStore(scene);
    cleanups.push(() => store.dispose());
    const app = new AppState();
    const item = app.addMesh(store, {
      name: "sphere",
      positions: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
      center: [0, 0, 0],
      radius: 2,
      volumeId: null,
      preset: null,
    });
    selectMesh(app, (m) => store.select(m), null, item.mesh, false);

    const { host } = mount(
      createElement(
        SceneProvider,
        { store },
        createElement(
          "div",
          null,
          createElement(SculptToolbox, { app, selection: item.mesh }),
          createElement(SculptInspectorCard, { app, selection: item.mesh }),
        ),
      ),
    );

    const toggle = host.querySelector<HTMLButtonElement>('[data-action="sculpt-dyntopo"]')!;
    expect(toggle).not.toBeNull();
    expect(toggle.getAttribute("aria-pressed")).toBe("false");

    await act(async () => {
      toggle.click();
    });
    expect(toggle.getAttribute("aria-pressed")).toBe("true");
    expect(app.getSnapshot().sculptDyntopo).toBe(true);

    const modeSelect = host.querySelector<HTMLSelectElement>('[data-action="sculpt-dyntopo-mode"]')!;
    expect(Array.from(modeSelect.options).map((o) => o.textContent)).toEqual([
      "Brush Relative",
      "Constant Detail",
      "Split & Collapse",
      "Subdivide Only",
    ]);

    await act(async () => {
      setNativeValue(modeSelect, "constant-detail");
    });
    expect(app.getSnapshot().sculptDyntopoMode).toBe("constant-detail");

    const inspectorMode = host.querySelector<HTMLSelectElement>('[data-action="sculpt-inspector-dyntopo-mode"]')!;
    expect(inspectorMode.value).toBe("constant-detail");

    const applyBtn = host.querySelector<HTMLButtonElement>('[data-action="sculpt-dyntopo-apply"]')!;
    await act(async () => {
      applyBtn.click();
      await Promise.resolve();
    });
    expect(spies.applyDyntopo).toHaveBeenCalledTimes(1);
    expect(spies.beginStep).toHaveBeenCalledTimes(1);
    expect(spies.endStep).toHaveBeenCalledTimes(1);

    const inspectorApplyBtn = host.querySelector<HTMLButtonElement>('[data-action="sculpt-inspector-dyntopo-apply"]')!;
    await act(async () => {
      inspectorApplyBtn.click();
      await Promise.resolve();
    });
    expect(spies.applyDyntopo).toHaveBeenCalledTimes(2);
  });

  it("renders G0/G1/G2 continuity fill and partial arc bridge controls in RepairPanel with undo coalescing", async () => {
    const { scene, spies } = createMockScene();
    const store = createSceneStore(scene);
    cleanups.push(() => store.dispose());
    const app = new AppState();
    const item = app.addMesh(store, {
      name: "defective",
      positions: new Float64Array(60),
      indices: new Uint32Array([0, 1, 2]),
      center: [0, 0, 0],
      radius: 1,
      volumeId: null,
      preset: null,
    });
    selectMesh(app, (m) => store.select(m), null, item.mesh, false);

    const { host } = mount(
      createElement(
        SceneProvider,
        { store },
        createElement(RepairPanel, { app, scene }),
      ),
    );
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    const continuitySelect = host.querySelector<HTMLSelectElement>('[data-continuity-mode="true"]')!;
    expect(continuitySelect).not.toBeNull();
    expect(Array.from(continuitySelect.options).map((o) => o.textContent)).toEqual([
      "Flat (G0)",
      "Tangent (G1)",
      "Curvature (G2)",
    ]);

    const refineBox = host.querySelector<HTMLInputElement>('[data-continuity-refine="true"]')!;
    const islandsBox = host.querySelector<HTMLInputElement>('[data-continuity-islands="true"]')!;
    expect(refineBox.checked).toBe(true);
    expect(islandsBox.checked).toBe(true);

    // Fill all holes with G2 Curvature continuity
    await act(async () => {
      host.querySelector<HTMLButtonElement>('[data-repair-op="fillContinuity"]')!.click();
      await Promise.resolve();
    });
    expect(spies.fillAllHolesContinuity).toHaveBeenCalledWith(item.mesh, {
      continuity: "curvature",
      refinePatch: true,
      detectIslands: true,
      maxHoleVertices: 64,
    });
    expect(spies.beginStep).toHaveBeenCalledTimes(1);
    expect(spies.endStep).toHaveBeenCalledTimes(1);

    // Switch to single loop (#1 => loopIndex 1 in raw analysis since holes sorted largest first!) + Tangent (G1)
    const targetSelect = host.querySelector<HTMLSelectElement>('[data-continuity-target="true"]')!;
    await act(async () => {
      setNativeValue(continuitySelect, "tangent");
      setNativeValue(targetSelect, "1");
      islandsBox.click();
    });
    await act(async () => {
      host.querySelector<HTMLButtonElement>('[data-repair-op="fillContinuity"]')!.click();
      await Promise.resolve();
    });
    expect(spies.fillHoleContinuity).toHaveBeenCalledWith(item.mesh, 1, {
      continuity: "tangent",
      refinePatch: true,
      detectIslands: false,
    });

    // Existing fillHoles button remains preserved for legacy callers & E2E scripts
    expect(host.querySelector('[data-repair-op="fillHoles"]')).not.toBeNull();
    expect(host.querySelector('[data-repair-op="bridgeLoops"]')).not.toBeNull();

    // Partial arc bridge
    const pA = host.querySelector<HTMLSelectElement>('[data-partial-bridge-loop="a"]')!;
    const pB = host.querySelector<HTMLSelectElement>('[data-partial-bridge-loop="b"]')!;
    await act(async () => {
      setNativeValue(pA, "0");
      setNativeValue(pB, "1");
      setNativeValue(host.querySelector<HTMLInputElement>('[data-partial-arc="a-end"]')!, "4");
      setNativeValue(host.querySelector<HTMLInputElement>('[data-partial-arc="b-end"]')!, "5");
      setNativeValue(host.querySelector<HTMLInputElement>('[data-partial-subdiv="true"]')!, "2");
    });
    await act(async () => {
      host.querySelector<HTMLButtonElement>('[data-repair-op="bridgePartialArcs"]')!.click();
      await Promise.resolve();
    });
    expect(spies.bridgePartialArcs).toHaveBeenCalledTimes(1);
    const [, loopAVerts, loopBVerts, partialOpts] = spies.bridgePartialArcs.mock.calls[0]!;
    expect(loopAVerts).toBeInstanceOf(Uint32Array);
    expect(loopBVerts).toBeInstanceOf(Uint32Array);
    expect(partialOpts).toEqual({
      loopAIndex: 1,
      arcARange: [0, 4],
      loopBIndex: 0,
      arcBRange: [0, 5],
      twistOffset: 0,
      subdivisions: 2,
    });
  });
});
