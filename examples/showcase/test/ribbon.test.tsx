// @vitest-environment jsdom
// Workspace 2-tier ribbon, single active right inspector slot matching,
// and Sculpt & Deform mounted DOM preservation tests (#228).
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore } from "fast-mesh-wasm/store";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  InspectorHeader,
  SculptInspectorCard,
  tabMatchesSlot,
  WORKBENCH_TABS,
  WorkbenchRibbon,
} from "../src/panels/Ribbon";
import { AppState } from "../src/scene/app-state";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const cleanups: (() => Promise<void> | void)[] = [];

afterEach(async () => {
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
  } as unknown as Scene;
}

describe("WorkbenchRibbon & Inspector Slot Routing (#228)", () => {
  it("maps each workspace tab to a focused active inspector slot without stacking all domains", () => {
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

    expect(tabMatchesSlot("repair", "repair")).toBe(true);
    expect(tabMatchesSlot("repair", "modeling")).toBe(false);
    expect(tabMatchesSlot("repair", "volume")).toBe(false);

    expect(tabMatchesSlot("sculpt", "sculpt")).toBe(true);
    expect(tabMatchesSlot("sculpt", "repair")).toBe(false);

    expect(tabMatchesSlot("align", "align")).toBe(true);
    expect(tabMatchesSlot("align", "collision")).toBe(true);
    expect(tabMatchesSlot("align", "inspect")).toBe(false);

    expect(tabMatchesSlot("volume", "volume")).toBe(true);
    expect(tabMatchesSlot("inspect", "inspect")).toBe(true);
  });

  it("renders Tier 1 .toolbar and Tier 2 .ribbon-shelf cleanly while keeping SculptToolbox DOM hooks live", async () => {
    const store = createSceneStore(createMockScene());
    const app = new AppState();
    const item = app.addMesh(store, {
      name: "SculptSphere",
      positions: Float64Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      indices: Uint32Array.from([0, 1, 2]),
      center: [0, 0, 0],
      radius: 1,
      volumeId: null,
      preset: null,
    });

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
              selection: null,
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
            createElement(InspectorHeader, { app, activeTab: app.getSnapshot().activeTab }),
            createElement(SculptInspectorCard, { app, selection: item.mesh }),
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

    const toolbar = container.querySelector(".toolbar");
    const shelf = container.querySelector(".ribbon-shelf");
    expect(toolbar).not.toBeNull();
    expect(shelf).not.toBeNull();
    // Shelf is a distinct Tier 2 sibling below .toolbar, never nested inside .toolbar.
    expect(toolbar?.contains(shelf)).toBe(false);

    // Essential quick-access selectors and all 8 tabs exist in .toolbar.
    expect(toolbar?.querySelector('[data-action="undo"]')).not.toBeNull();
    expect(toolbar?.querySelector('[data-action="redo"]')).not.toBeNull();
    expect(toolbar?.querySelector('[data-action="view-reset"]')).not.toBeNull();
    expect(toolbar?.querySelector('[data-action="mode-scale"]')).not.toBeNull();
    expect(toolbar?.querySelector('[data-action="command-palette"]')).not.toBeNull();
    expect(toolbar?.querySelectorAll('[data-ribbon-tab]').length).toBe(8);

    // Sculpt controls remain live in DOM even when activeTab is "home",
    // allowing E2E selectors and hover-to-sculpt arming without selection.
    const radiusInput = container.querySelector('label.field[title="Brush radius"] input');
    const inflateBtn = container.querySelector<HTMLButtonElement>('[data-action="sculpt-tool-inflate"]');
    expect(radiusInput).not.toBeNull();
    expect(inflateBtn).not.toBeNull();
    expect(inflateBtn?.disabled).toBe(false);
    expect(container.querySelector(".ribbon-sculpt-wrap")?.classList.contains("ribbon-hidden")).toBe(true);

    // Clicking a ribbon tab activates Tier 2 contextual controls.
    const sculptTabBtn = container.querySelector<HTMLButtonElement>('[data-ribbon-tab="sculpt"]')!;
    await act(async () => {
      sculptTabBtn.click();
    });
    expect(app.getSnapshot().activeTab).toBe("sculpt");
    expect(container.querySelector(".ribbon-sculpt-wrap")?.classList.contains("ribbon-hidden")).toBe(false);

    // Clicking Move/Rotate/Scale in .toolbar disarms sculpt and activates "select".
    await act(async () => {
      inflateBtn?.click();
    });
    expect(app.getSnapshot().sculpt?.tool).toBe("inflate");

    const scaleBtn = toolbar?.querySelector<HTMLButtonElement>('[data-action="mode-scale"]')!;
    await act(async () => {
      scaleBtn.click();
    });
    expect(onSetMode).toHaveBeenCalledWith("scale");
    expect(app.getSnapshot().sculpt).toBeNull();
    expect(app.getSnapshot().activeTab).toBe("select");
  });
});
