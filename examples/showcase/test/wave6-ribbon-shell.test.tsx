// @vitest-environment jsdom
// Wave 6 Showcase shell wiring verification:
// - 8 WORKBENCH_TABS preserved + extended InspectorSlot routing (`select`, `deform`, `points`)
// - Tier 2 Ribbon Wave 6 quick tools & pickMode disarming
// - CommandPalette Wave 6 tool commands
// - Clean jsdom rendering of all Wave 6 panels & FaceSelectionMarquee without eager Wasm loading
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore } from "fast-mesh-wasm/store";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CollisionPanel } from "../src/features/collision/ui/CollisionPanel";
import { buildCommandPaletteItems } from "../src/shell/ui/CommandPalette";
import { DeformPanel } from "../src/features/sculpt/ui/DeformPanel";
import { InspectPanel } from "../src/features/inspect/ui/InspectPanel";
import { ModelingPanel } from "../src/features/mesh-edit/ui/ModelingPanel";
import { PointsPanel } from "../src/features/points/ui/PointsPanel";
import { tabMatchesSlot, WORKBENCH_TABS, WorkbenchRibbon } from "../src/shell/ui/Ribbon";
import { SelectPanel } from "../src/features/select/ui/SelectPanel";
import { AppState } from "../src/app/state/app-state";
import {
  FaceSelectionMarquee,
  setFaceSelectionMarquee,
} from "../src/features/select/ui/FaceSelectionLayer";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const cleanups: (() => Promise<void> | void)[] = [];

afterEach(async () => {
  setFaceSelectionMarquee(null);
  vi.restoreAllMocks();
  for (const fn of cleanups.splice(0).reverse()) {
    await fn();
  }
});

function createMockScene(): Scene {
  let nextId = 0;
  return {
    addMesh: () => ({ __mesh: ++nextId }) as unknown as MeshHandle,
    removeMesh: () => {},
    onChanges: () => () => {},
    canUndo: () => false,
    canRedo: () => false,
    views: () => ({
      positions: () => new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      indices: () => new Uint32Array([0, 1, 2]),
      normals: () => null,
      colors: () => null,
    }),
  } as unknown as Scene;
}

describe("Wave 6 Ribbon, Shell Slot Routing, CommandPalette & Lazy Wasm Safety", () => {
  it("preserves the 8 WORKBENCH_TABS IDs and routes select/deform/points slots cleanly", () => {
    expect(WORKBENCH_TABS.map((t) => t.id)).toEqual([
      "home",
      "select",
      "repair",
      "edit",
      "sculpt",
      "align",
      "volume",
      "inspect",
    ]);

    expect(tabMatchesSlot("select", "transform")).toBe(true);
    expect(tabMatchesSlot("select", "select")).toBe(true);
    expect(tabMatchesSlot("select", "deform")).toBe(false);

    expect(tabMatchesSlot("sculpt", "sculpt")).toBe(true);
    expect(tabMatchesSlot("sculpt", "deform")).toBe(true);
    expect(tabMatchesSlot("sculpt", "select")).toBe(false);

    expect(tabMatchesSlot("align", "points")).toBe(true);
    expect(tabMatchesSlot("align", "align")).toBe(true);
    expect(tabMatchesSlot("align", "collision")).toBe(true);
    expect(tabMatchesSlot("align", "inspect")).toBe(false);
  });

  it("mounts all Wave 6 panels and FaceSelectionMarquee in jsdom without Wasm init and handles Ribbon + CommandPalette tools", async () => {
    const mockScene = createMockScene();
    const store = createSceneStore(mockScene);
    const app = new AppState();
    const item = app.addMesh(store, {
      name: "ShellPart",
      positions: Float64Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      indices: Uint32Array.from([0, 1, 2]),
      center: [0, 0, 0],
      radius: 1,
      volumeId: null,
      preset: null,
    });
    store.select(item.mesh);

    const container = document.createElement("div");
    document.body.appendChild(container);
    const onSetMode = vi.fn();
    let root: Root;

    await act(async () => {
      root = createRoot(container);
      root.render(
        createElement(
          SceneProvider,
          { store },
          createElement(
            "div",
            null,
            createElement(WorkbenchRibbon, {
              app,
              mode: "translate",
              canUndo: false,
              canRedo: false,
              wireframe: false,
              selection: item.mesh,
              hasItems: true,
              hasVolume: false,
              onSetMode,
              onResetView: () => {},
              onFocusSelection: () => {},
              onClearSelection: () => {},
              onUndo: () => {},
              onRedo: () => {},
              onOpenHelp: () => {},
            }),
            createElement(FaceSelectionMarquee),
            createElement(SelectPanel, { app, store }),
            createElement(DeformPanel, { app, store }),
            createElement(PointsPanel, { app, store }),
            createElement(ModelingPanel, { app }),
            createElement(CollisionPanel, { app, scene: mockScene }),
            createElement(InspectPanel, { app, scene: mockScene }),
          ),
        ),
      );
    });

    cleanups.push(async () => {
      await act(async () => root.unmount());
      container.remove();
      app.removeMesh(store, item);
      store.dispose();
    });

    // Switch to Select tab -> Tier 2 face selection quick tools
    const selectTab = container.querySelector<HTMLButtonElement>('[data-ribbon-tab="select"]')!;
    await act(async () => {
      selectTab.click();
    });
    expect(app.getSnapshot().activeTab).toBe("select");

    const lassoBtn = container.querySelector<HTMLButtonElement>('[data-ribbon-select-tool="lasso"]')!;
    expect(lassoBtn).not.toBeNull();
    await act(async () => {
      lassoBtn.click();
    });
    expect(app.getSnapshot().wave6.select.activeTool).toBe("lasso");

    // Marquee SVG renders when marquee state is active
    await act(async () => {
      setFaceSelectionMarquee({
        kind: "box",
        x0: 10,
        y0: 20,
        x1: 120,
        y1: 90,
        width: 400,
        height: 300,
      });
    });
    expect(container.querySelector('[data-select-marquee-overlay="true"]')).not.toBeNull();

    // Gizmo Scale button disarms active Wave 6 tools cleanly
    const scaleBtn = container.querySelector<HTMLButtonElement>('[data-action="mode-scale"]')!;
    await act(async () => {
      scaleBtn.click();
    });
    expect(onSetMode).toHaveBeenCalledWith("scale");
    expect(app.getSnapshot().wave6.select.activeTool).toBeNull();

    // Switch to Sculpt & Deform -> Deform algorithm quick buttons appear alongside SculptToolbox
    const sculptTab = container.querySelector<HTMLButtonElement>('[data-ribbon-tab="sculpt"]')!;
    await act(async () => {
      sculptTab.click();
    });
    const arapBtn = container.querySelector<HTMLButtonElement>('[data-ribbon-deform-algo="arap"]')!;
    expect(arapBtn).not.toBeNull();
    await act(async () => {
      arapBtn.click();
    });
    expect(app.getSnapshot().wave6.deform.algorithm).toBe("arap");

    // Edit, Align, and Inspect Tier-2 Wave 6 quick focus buttons exist
    await act(async () => {
      container.querySelector<HTMLButtonElement>('[data-ribbon-tab="edit"]')!.click();
    });
    expect(container.querySelector('[data-ribbon-focus="hollow"]')).not.toBeNull();
    expect(container.querySelector('[data-ribbon-focus="drain"]')).not.toBeNull();
    expect(container.querySelector('[data-ribbon-focus="place"]')).not.toBeNull();

    await act(async () => {
      container.querySelector<HTMLButtonElement>('[data-ribbon-tab="align"]')!.click();
    });
    expect(container.querySelector('[data-ribbon-focus="points"]')).not.toBeNull();
    expect(container.querySelector('[data-ribbon-focus="collision-rules"]')).not.toBeNull();

    await act(async () => {
      container.querySelector<HTMLButtonElement>('[data-ribbon-tab="inspect"]')!.click();
    });
    expect(container.querySelector('[data-ribbon-focus="fit"]')).not.toBeNull();
    expect(container.querySelector('[data-ribbon-focus="contours"]')).not.toBeNull();

    // CommandPalette Wave 6 items
    const cmds = buildCommandPaletteItems({
      app,
      items: app.getSnapshot().items,
      selection: item.mesh,
      canUndo: false,
      canRedo: false,
      ortho: false,
      units: "mm",
      onSelectItem: () => {},
      onSetMode,
      onResetView: () => {},
      onUndo: () => {},
      onRedo: () => {},
      onOpenHelp: () => {},
    });
    const drainCmd = cmds.find((c) => c.id === "tool:print-drain-hole");
    expect(drainCmd).toBeDefined();
    drainCmd!.run();
    expect(app.getSnapshot().activeTab).toBe("edit");
    expect(app.getSnapshot().wave6.printPrep.activeTool).toBe("drainHole");
  });
});
