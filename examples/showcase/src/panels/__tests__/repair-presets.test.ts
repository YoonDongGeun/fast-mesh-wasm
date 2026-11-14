// @vitest-environment jsdom
import type {
  Analysis,
  AutoRepairReport,
  AutoRepairRequest,
  ManifoldReport,
  MeshHandle,
  RepairOp,
  RepairReport,
  Scene,
  SelfIntersectReport,
  SliverReport,
} from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppState } from "../../scene/app-state";
import {
  buildPresetAutoRepairRequest,
  describeReport,
  formatPresetAutoRepairReport,
  summarize,
} from "../repair-summary";
import { RepairPanel } from "../RepairPanel";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function emptyAnalysis(patch: Partial<Analysis> = {}): Analysis {
  return {
    loops: [],
    coincidentGroups: 0,
    coincident: [],
    degenerate: 0,
    degenerateList: new Uint32Array(),
    duplicate: 0,
    duplicateList: new Uint32Array(),
    nonManifoldEdges: 0,
    nonManifoldEdgeList: [],
    nonManifoldVertices: new Uint32Array(),
    pinchVertices: new Uint32Array(),
    components: [{ triangles: new Uint32Array(12), area: 6, closed: true }],
    orientationConflicts: 0,
    selfIntersects: null,
    ...patch,
  };
}

const cleanups: (() => Promise<void> | void)[] = [];

afterEach(async () => {
  vi.restoreAllMocks();
  for (const fn of cleanups.splice(0).reverse()) {
    await fn();
  }
});

async function mountRepairPanel(options?: {
  onAnalyze?: (opts?: { selfIntersection?: boolean }) => Analysis;
  onRepair?: (mesh: MeshHandle, op: RepairOp | AutoRepairRequest) => RepairReport | AutoRepairReport;
}): Promise<{
  container: HTMLDivElement;
  app: AppState;
  store: SceneStore;
  mesh: MeshHandle;
  repairSpy: ReturnType<typeof vi.spyOn>;
  analyzeSpy: ReturnType<typeof vi.fn>;
}> {
  let nextId = 0;
  const analyzeSpy = vi.fn((_mesh: MeshHandle, opts?: { selfIntersection?: boolean }) =>
    options?.onAnalyze ? options.onAnalyze(opts) : emptyAnalysis({ selfIntersects: opts?.selfIntersection ? false : null }),
  );
  const fakeScene = {
    addMesh: () => ({ __mesh: ++nextId }) as unknown as MeshHandle,
    removeMesh: () => {},
    onChanges: () => () => {},
    canUndo: () => false,
    canRedo: () => false,
    analyze: analyzeSpy,
    repair: vi.fn((mesh: MeshHandle, op: RepairOp | AutoRepairRequest) => {
      if (options?.onRepair) {
        return options.onRepair(mesh, op);
      }
      if ("kind" in op && op.kind === "auto") {
        const rep: AutoRepairReport = {
          weld: { mergedVertices: 2, removedTriangles: 0 },
          removeDegenerate: { removedTriangles: 1, removedVertices: 0 },
          orient: { flippedTriangles: 0, patches: 1, nonOrientablePatches: 0, flippedOutwardComponents: 0 },
          removeSmallComponents: { removedComponents: 0, removedTriangles: 0, removedVertices: 0 },
          fillHoles: { filled: [], skipped: [], failed: [] },
          makeManifold: {
            removedDuplicateTriangles: 1,
            prunedFinTriangles: 1,
            splitNonManifoldEdges: 0,
            splitPinchVertices: 1,
            verticesAdded: 1,
            removedVertices: 0,
          },
          resolveSelfIntersections: {
            initialPairs: 4,
            remainingPairs: 0,
            removedTriangles: 6,
            addedTriangles: 6,
            modeUsed: op.selfIntersectionMode ?? "deleteAndFill",
            uvsDropped: op.selfIntersectionMode === "voxelHeal",
          },
        };
        return rep;
      }
      if ("op" in op && op.op === "makeManifold") {
        const r: ManifoldReport = {
          removedDuplicateTriangles: 1,
          prunedFinTriangles: 2,
          splitNonManifoldEdges: 1,
          splitPinchVertices: 1,
          verticesAdded: 2,
          removedVertices: 1,
        };
        return r;
      }
      if ("op" in op && op.op === "cleanupSlivers") {
        const r: SliverReport = {
          collapsedNeedles: 3,
          flippedCaps: 2,
          removedDegenerate: 1,
          removedVertices: 3,
        };
        return r;
      }
      if ("op" in op && op.op === "resolveSelfIntersections") {
        const r: SelfIntersectReport = {
          initialPairs: 5,
          remainingPairs: 0,
          removedTriangles: 8,
          addedTriangles: 12,
          modeUsed: op.mode ?? "deleteAndFill",
          uvsDropped: op.mode === "voxelHeal",
        };
        return r;
      }
      throw new Error(`unexpected op: ${JSON.stringify(op)}`);
    }),
  };

  const scene = fakeScene as unknown as Scene;
  const store = createSceneStore(scene);
  const app = new AppState();
  const item = app.addMesh(store, {
    name: "PresetMesh",
    positions: Float64Array.from([0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10]),
    indices: Uint32Array.from([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]),
    center: [2.5, 2.5, 2.5],
    radius: 10,
    volumeId: null,
    preset: null,
  });
  store.select(item.mesh);

  const repairSpy = vi.spyOn(store, "repair");
  const container = document.createElement("div");
  document.body.appendChild(container);
  let root: Root;
  await act(async () => {
    root = createRoot(container);
    root.render(createElement(SceneProvider, { store }, createElement(RepairPanel, { app, scene })));
  });

  cleanups.push(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    app.removeMesh(store, item);
    store.dispose();
  });

  return { container, app, store, mesh: item.mesh, repairSpy, analyzeSpy };
}

async function setInputValue(input: HTMLInputElement, value: string): Promise<void> {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
  await act(async () => {
    setter?.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

async function setSelectValue(select: HTMLSelectElement, value: string): Promise<void> {
  const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")?.set;
  await act(async () => {
    setter?.call(select, value);
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

async function clickButton(button: HTMLButtonElement): Promise<void> {
  await act(async () => {
    button.click();
    await Promise.resolve();
  });
}

describe("repair healing summaries & panel presets", () => {
  it("formats manifold, sliver, self-intersection, and preset reports cleanly", () => {
    expect(
      describeReport("makeManifold", {
        removedDuplicateTriangles: 1,
        prunedFinTriangles: 2,
        splitNonManifoldEdges: 1,
        splitPinchVertices: 3,
        verticesAdded: 4,
        removedVertices: 1,
      }),
    ).toBe("Make manifold: removed 1 duplicate, pruned 2 fins, split 1 non-manifold edge and 3 pinch vertices");

    expect(
      describeReport("cleanupSlivers", {
        collapsedNeedles: 2,
        flippedCaps: 1,
        removedDegenerate: 0,
        removedVertices: 2,
      }),
    ).toBe("Cleanup slivers: collapsed 2 needles, flipped 1 cap, removed 0 degenerate triangles");

    expect(
      describeReport("resolveSelfIntersections", {
        initialPairs: 4,
        remainingPairs: 0,
        removedTriangles: 10,
        addedTriangles: 16,
        modeUsed: "voxelHeal",
        uvsDropped: true,
      }),
    ).toBe("Self-intersections (voxelHeal): 4 → 0 pairs (-10 triangles, +16 triangles) (UVs dropped)");

    const cadReq = buildPresetAutoRepairRequest({
      preset: "cadImport",
      weldTolerance: 1e-6,
      minTriangles: 500,
      maxEdges: 0,
      fair: false,
    });
    expect(cadReq.maxEdges).toBeUndefined();

    const printReq = buildPresetAutoRepairRequest({
      preset: "print3d",
      selfIntersectionMode: "cutAndStitch",
      weldTolerance: 1e-5,
      minTriangles: 500,
      maxEdges: 0,
      fair: true,
    });
    expect(printReq.maxEdges).toBe(Number.POSITIVE_INFINITY);
    expect(printReq.selfIntersectionMode).toBe("cutAndStitch");

    const before = summarize(emptyAnalysis({ nonManifoldEdges: 2, pinchVertices: new Uint32Array([0]) }));
    const after = summarize(emptyAnalysis());
    const formatted = formatPresetAutoRepairReport(
      "print3d",
      {
        weld: { mergedVertices: 0, removedTriangles: 0 },
        removeDegenerate: { removedTriangles: 0, removedVertices: 0 },
        orient: { flippedTriangles: 0, patches: 1, nonOrientablePatches: 0, flippedOutwardComponents: 0 },
        removeSmallComponents: { removedComponents: 0, removedTriangles: 0, removedVertices: 0 },
        fillHoles: { filled: [], skipped: [], failed: [] },
        makeManifold: {
          removedDuplicateTriangles: 0,
          prunedFinTriangles: 1,
          splitNonManifoldEdges: 1,
          splitPinchVertices: 1,
          verticesAdded: 2,
          removedVertices: 0,
        },
        resolveSelfIntersections: {
          initialPairs: 3,
          remainingPairs: 0,
          removedTriangles: 6,
          addedTriangles: 6,
          modeUsed: "deleteAndFill",
          uvsDropped: false,
        },
      },
      before,
      after,
      true,
      false,
    );
    expect(formatted.line).toContain("Auto repair (3D Print, one undo step)");
    expect(formatted.line).toContain("2 non-manifold edges → none");
    expect(formatted.line).toContain("self-intersections yes → none");
    expect(formatted.stageLines.some((l) => l.startsWith("Self-intersections (deleteAndFill)"))).toBe(true);
  });

  it("wires manual healing buttons and auto repair preset selector through RepairPanel", async () => {
    const { container, mesh, repairSpy, analyzeSpy } = await mountRepairPanel({
      onAnalyze: (opts) => emptyAnalysis({ selfIntersects: opts?.selfIntersection ? true : null }),
    });

    const manifoldBtn = container.querySelector<HTMLButtonElement>('[data-repair-op="makeManifold"]')!;
    const sliverBtn = container.querySelector<HTMLButtonElement>('[data-repair-op="cleanupSlivers"]')!;
    const siBtn = container.querySelector<HTMLButtonElement>('[data-repair-op="resolveSelfIntersections"]')!;
    const siMode = container.querySelector<HTMLSelectElement>("[data-si-mode]")!;

    await clickButton(manifoldBtn);
    expect(repairSpy).toHaveBeenLastCalledWith(mesh, { op: "makeManifold" });
    expect(container.querySelector("[data-repair-report]")?.textContent).toContain("Make manifold: removed 1 duplicate");

    await clickButton(sliverBtn);
    expect(repairSpy).toHaveBeenLastCalledWith(mesh, { op: "cleanupSlivers" });
    expect(container.querySelector("[data-repair-report]")?.textContent).toContain("Cleanup slivers: collapsed 3 needles");

    await setSelectValue(siMode, "voxelHeal");
    const voxelInput = container.querySelector<HTMLInputElement>("[data-si-voxel]")!;
    await setInputValue(voxelInput, "0.2");
    await clickButton(siBtn);
    expect(repairSpy).toHaveBeenLastCalledWith(mesh, {
      op: "resolveSelfIntersections",
      mode: "voxelHeal",
      voxelSize: 0.2,
    });
    expect(container.querySelector("[data-repair-report]")?.textContent).toContain("(UVs dropped)");

    // Select 3D Print preset + Cut & Stitch override and click Auto repair.
    const presetSelect = container.querySelector<HTMLSelectElement>("[data-repair-preset]")!;
    const overrideSelect = container.querySelector<HTMLSelectElement>("[data-repair-si-mode]")!;
    const autoBtn = container.querySelector<HTMLButtonElement>('[data-repair-op="autoRepair"]')!;
    await setSelectValue(presetSelect, "print3d");
    await setSelectValue(overrideSelect, "cutAndStitch");
    await clickButton(autoBtn);

    expect(repairSpy).toHaveBeenLastCalledWith(mesh, {
      kind: "auto",
      preset: "print3d",
      selfIntersectionMode: "cutAndStitch",
      weldTolerance: 1e-5,
      minTriangles: 500,
      maxEdges: 64,
      fair: true,
      voxelSize: 0.2,
    });
    expect(container.querySelector("[data-repair-report]")?.textContent).toContain("Auto repair (3D Print, one undo step)");

    // Toggling Check self-intersections passes selfIntersection: true to analyze.
    const checkBox = container.querySelector<HTMLInputElement>("[data-check-self-intersections]")!;
    await act(async () => {
      checkBox.click();
      await Promise.resolve();
    });
    expect(analyzeSpy.mock.calls.some((c) => c[1]?.selfIntersection === true)).toBe(true);
  });
});
