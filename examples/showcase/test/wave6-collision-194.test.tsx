// @vitest-environment jsdom
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("fast-mesh-wasm", async () => {
  return await import("../../../packages/fast-mesh-wasm/dist/web.js");
});

import { init, Scene } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { CollisionPanel } from "../src/panels/CollisionPanel";
import { GroupsBar } from "../src/panels/GroupsBar";
import { AppState, type Item } from "../src/scene/app-state";
import { roundedBox } from "../src/scene/demo";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function setSelectValue(el: HTMLSelectElement, value: string): void {
  const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")?.set;
  if (setter) {
    setter.call(el, value);
  } else {
    el.value = value;
  }
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

function setInputValue(el: HTMLInputElement, value: string): void {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
  if (setter) {
    setter.call(el, value);
  } else {
    el.value = value;
  }
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

describe("Wave 6 Inter-Object Collision UI & Group Controls (#194)", () => {
  const cleanups: (() => void)[] = [];

  beforeAll(async () => {
    const candidates = [
      resolve(process.cwd(), "../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"),
      resolve(process.cwd(), "packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"),
    ];
    const wasmPath = candidates.find((p) => existsSync(p));
    if (!wasmPath) {
      throw new Error(`Missing prebuilt Wasm binary in: ${candidates.join(", ")}`);
    }
    await init(readFileSync(wasmPath));
  });

  afterEach(() => {
    for (const fn of cleanups.splice(0).reverse()) {
      fn();
    }
  });

  function setupScene() {
    const scene = new Scene();
    const store = createSceneStore(scene);
    const app = new AppState();

    const meshAData = roundedBox([0, 0, 0], [1, 1, 1], 0.08, 4);
    const meshBData = roundedBox([0, 0, 0], [1, 1, 1], 0.08, 4);

    const itemA = app.addMesh(store, {
      name: "box-a",
      positions: meshAData.positions,
      indices: meshAData.indices,
      center: [0, 0, 0],
      radius: 1.75,
      volumeId: null,
      preset: null,
    });
    const itemB = app.addMesh(store, {
      name: "box-b",
      positions: meshBData.positions,
      indices: meshBData.indices,
      center: [0, 0, 0],
      radius: 1.75,
      volumeId: null,
      preset: null,
    });

    scene.setPlacement(itemB.mesh, {
      rotation: [0, 0, 0, 1],
      translation: [3, 0, 0],
    });

    cleanups.push(() => {
      store.dispose();
    });

    return { scene, store, app, itemA, itemB };
  }

  function mountCollisionPanel(app: AppState, store: SceneStore, scene: Scene) {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root: Root = createRoot(host);
    act(() => {
      root.render(
        <SceneProvider store={store}>
          <CollisionPanel app={app} scene={scene} />
        </SceneProvider>,
      );
    });
    cleanups.push(() => {
      act(() => {
        root.unmount();
      });
      host.remove();
    });
    return host;
  }

  function mountGroupsBar(app: AppState, store: SceneStore) {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root: Root = createRoot(host);
    act(() => {
      root.render(
        <SceneProvider store={store}>
          <GroupsBar app={app} />
        </SceneProvider>,
      );
    });
    cleanups.push(() => {
      act(() => {
        root.unmount();
      });
      host.remove();
    });
    return host;
  }

  async function pickPair(host: HTMLElement, a: Item, b: Item) {
    const pickA = host.querySelector<HTMLSelectElement>('[data-pick="a"]');
    const pickB = host.querySelector<HTMLSelectElement>('[data-pick="b"]');
    expect(pickA).not.toBeNull();
    expect(pickB).not.toBeNull();
    await act(async () => {
      setSelectValue(pickA!, String(a.id));
      setSelectValue(pickB!, String(b.id));
      await new Promise((r) => setTimeout(r, 0));
    });
  }

  it("preserves existing CollisionPanel and GroupsBar selectors and defaults to passThrough", async () => {
    const { scene, store, app, itemA, itemB } = setupScene();
    const defaultRuleSpy = vi.spyOn(scene, "setDefaultCollisionRule");

    const collisionHost = mountCollisionPanel(app, store, scene);
    const groupsHost = mountGroupsBar(app, store);

    expect(app.getSnapshot().wave6.collision.mode).toBe("passThrough");
    expect(scene.defaultCollisionRule().mode).toBe("passThrough");
    expect(defaultRuleSpy).not.toHaveBeenCalled();

    expect(collisionHost.querySelector('[data-mode="pair"]')).not.toBeNull();
    expect(collisionHost.querySelector('[data-mode="group"]')).not.toBeNull();
    expect(collisionHost.querySelector('[data-control="depth"]')).not.toBeNull();
    expect(collisionHost.querySelector('[data-control="gap"]')).not.toBeNull();
    expect(collisionHost.querySelector('[data-control="tolerance"]')).not.toBeNull();
    expect(collisionHost.querySelector('[data-control="softness"]')).not.toBeNull();
    expect(collisionHost.querySelector('[data-control="fast"]')).not.toBeNull();

    await pickPair(collisionHost, itemA, itemB);
    expect(collisionHost.querySelector("[data-readouts]")).not.toBeNull();

    const groupModeBtn = collisionHost.querySelector<HTMLButtonElement>('[data-mode="group"]');
    await act(async () => {
      groupModeBtn!.click();
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(collisionHost.querySelector('[data-control="all-pairs"]')).not.toBeNull();

    act(() => {
      app.toggleCheck(itemA.mesh, true);
      app.toggleCheck(itemB.mesh, true);
    });
    const groupBtn = groupsHost.querySelector<HTMLButtonElement>('[data-action="group-checked"]');
    expect(groupBtn).not.toBeNull();
    expect(groupBtn!.disabled).toBe(false);
    await act(async () => {
      groupBtn!.click();
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(groupsHost.querySelector('[data-group-color="Group 1"]')).not.toBeNull();
  });

  it("pushes rigid rule to store.scene and clamps drag penetration between two boxes", async () => {
    const { scene, store, app, itemA, itemB } = setupScene();
    const host = mountCollisionPanel(app, store, scene);
    await pickPair(host, itemA, itemB);

    const modeSelect = host.querySelector<HTMLSelectElement>("[data-collision-default-mode]")!;
    const marginInput = host.querySelector<HTMLInputElement>("[data-collision-margin]")!;
    const slideCheck = host.querySelector<HTMLInputElement>("[data-collision-rigid-slide]")!;

    act(() => {
      setSelectValue(modeSelect, "rigid");
      setInputValue(marginInput, "0.1");
      if (slideCheck.checked) {
        slideCheck.click();
      }
    });

    expect(app.getSnapshot().wave6.collision.mode).toBe("rigid");
    expect(app.getSnapshot().wave6.collision.rigidSlide).toBe(false);
    const activeRule = scene.defaultCollisionRule();
    expect(activeRule.mode).toBe("rigid");
    expect(activeRule.contactMargin).toBeCloseTo(0.1, 4);
    expect(activeRule.rigidSlide).toBe(false);

    const session = scene.drag(itemB.mesh);
    session.move({ rotation: [0, 0, 0, 1], translation: [1.0, 0, 0] });
    session.end();

    const placedX = scene.placement(itemB.mesh).translation[0];
    expect(placedX).toBeGreaterThan(1.9);
  });

  it("maps cutBackend to cutMethod, manages pair overrides, and executes boolean cut on drag release", async () => {
    const { scene, store, app, itemA, itemB } = setupScene();
    const host = mountCollisionPanel(app, store, scene);
    await pickPair(host, itemA, itemB);

    const defaultModeSelect = host.querySelector<HTMLSelectElement>("[data-collision-default-mode]")!;
    const distanceBackendSelect = host.querySelector<HTMLSelectElement>("[data-collision-distance-backend]")!;
    const cutBackendSelect = host.querySelector<HTMLSelectElement>("[data-collision-cut-backend]")!;
    const cutTriggerSelect = host.querySelector<HTMLSelectElement>("[data-collision-cut-trigger]")!;
    const roleSelect = host.querySelector<HTMLSelectElement>("[data-collision-role]")!;
    const pairModeSelect = host.querySelector<HTMLSelectElement>("[data-collision-pair-mode]")!;
    const pairClearBtn = host.querySelector<HTMLButtonElement>("[data-collision-pair-clear]")!;

    act(() => {
      setSelectValue(defaultModeSelect, "cut");
      setSelectValue(distanceBackendSelect, "fhEdt");
      setSelectValue(cutBackendSelect, "bvhSdf");
      setSelectValue(cutTriggerSelect, "liveStep");
      setSelectValue(roleSelect, "movingIsCutter");
    });

    expect(app.getSnapshot().wave6.collision.cutBackend).toBe("bvhSdf");
    expect(scene.defaultCollisionRule()).toMatchObject({
      mode: "cut",
      distanceBackend: "fhEdt",
      cutMethod: "bvhSdf",
      cutTrigger: "liveStep",
      role: "movingIsCutter",
    });

    act(() => {
      setSelectValue(cutTriggerSelect, "onRelease");
    });

    act(() => {
      setSelectValue(pairModeSelect, "passThrough");
    });
    expect(scene.collisionRule(itemA.mesh, itemB.mesh).mode).toBe("passThrough");
    expect(pairClearBtn.disabled).toBe(false);

    act(() => {
      pairClearBtn.click();
    });
    expect(scene.collisionRule(itemA.mesh, itemB.mesh).mode).toBe("cut");

    act(() => {
      setSelectValue(cutBackendSelect, "manifold");
    });
    expect(scene.defaultCollisionRule().cutMethod).toBe("manifold");

    const volumeBefore = scene.inspect.measure(itemA.mesh).volume!;
    expect(volumeBefore).toBeGreaterThan(6);

    const session = scene.drag(itemB.mesh);
    session.move({ rotation: [0, 0, 0, 1], translation: [1.2, 0, 0] });
    session.end();

    const volumeAfter = scene.inspect.measure(itemA.mesh).volume!;
    expect(volumeAfter).toBeLessThan(volumeBefore);
  });

  it("pushes softDeform options to store.scene, deforms on drag, and updates step report on nudge", async () => {
    const { scene, store, app, itemA, itemB } = setupScene();
    const host = mountCollisionPanel(app, store, scene);
    await pickPair(host, itemA, itemB);

    const defaultModeSelect = host.querySelector<HTMLSelectElement>("[data-collision-default-mode]")!;
    const softSolverSelect = host.querySelector<HTMLSelectElement>("[data-collision-soft-solver]")!;
    const memorySelect = host.querySelector<HTMLSelectElement>("[data-collision-memory]")!;
    const complianceInput = host.querySelector<HTMLInputElement>("[data-collision-compliance]")!;
    const stiffnessInput = host.querySelector<HTMLInputElement>("[data-collision-stiffness]")!;
    const recoveryInput = host.querySelector<HTMLInputElement>("[data-collision-recovery]")!;
    const selfGuardCheck = host.querySelector<HTMLInputElement>("[data-collision-self-guard]")!;

    act(() => {
      setSelectValue(defaultModeSelect, "softDeform");
      setSelectValue(softSolverSelect, "kelvinlet");
      setSelectValue(memorySelect, "plastic");
      setInputValue(complianceInput, "0.25");
      setInputValue(stiffnessInput, "0.8");
      setInputValue(recoveryInput, "0.4");
      selfGuardCheck.click();
    });

    const rule = scene.defaultCollisionRule();
    expect(rule.mode).toBe("softDeform");
    expect(rule.softSolver).toBe("kelvinlet");
    expect(rule.materialMemory).toBe("plastic");
    expect(rule.compliance).toBeCloseTo(0.25, 4);
    expect(rule.stiffness).toBeCloseTo(0.8, 4);
    expect(rule.recoveryRate).toBeCloseTo(0.4, 4);
    expect(rule.selfCollisionGuard).toBe(false);

    act(() => {
      setSelectValue(softSolverSelect, "sdfLaplacian");
      selfGuardCheck.click();
    });

    const beforeA = Float32Array.from(scene.views(itemA.mesh).positions());
    const session = scene.drag(itemB.mesh);
    session.move({ rotation: [0, 0, 0, 1], translation: [1.5, 0, 0] });
    session.end();

    const afterA = scene.views(itemA.mesh).positions();
    let maxDelta = 0;
    for (let i = 0; i < beforeA.length; i++) {
      maxDelta = Math.max(maxDelta, Math.abs(afterA[i] - beforeA[i]));
    }
    expect(maxDelta).toBeGreaterThan(1e-3);

    const nudgeBtn = host.querySelector<HTMLButtonElement>("[data-collision-nudge]")!;
    await act(async () => {
      nudgeBtn.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    const reportEl = host.querySelector("[data-collision-report]");
    expect(reportEl).not.toBeNull();
    expect(reportEl!.textContent).toContain("active pair");
    expect(app.getSnapshot().wave6.collision.lastStepSummary).toContain("active pair");
  });

  it("exposes per-group collision response badge and quick override selector in GroupsBar", () => {
    const { scene, store, app, itemA, itemB } = setupScene();
    const groupsHost = mountGroupsBar(app, store);

    act(() => {
      app.toggleCheck(itemA.mesh, true);
      app.toggleCheck(itemB.mesh, true);
    });
    const groupBtn = groupsHost.querySelector<HTMLButtonElement>('[data-action="group-checked"]')!;
    act(() => {
      groupBtn.click();
    });

    const badge = groupsHost.querySelector('[data-group-collision-badge="Group 1"]')!;
    const select = groupsHost.querySelector<HTMLSelectElement>('[data-group-collision="Group 1"]')!;
    expect(badge.textContent).toBe("passThrough");
    expect(select.value).toBe("default");
    expect(select.disabled).toBe(false);

    act(() => {
      setSelectValue(select, "rigid");
    });

    expect(badge.textContent).toBe("rigid");
    expect(scene.collisionRule(itemA.mesh, itemB.mesh).mode).toBe("rigid");
    expect(scene.defaultCollisionRule().mode).toBe("passThrough");

    act(() => {
      setSelectValue(select, "cut");
    });
    expect(badge.textContent).toBe("cut");
    expect(scene.collisionRule(itemA.mesh, itemB.mesh).mode).toBe("cut");

    act(() => {
      setSelectValue(select, "default");
    });
    expect(badge.textContent).toBe("passThrough");
    expect(scene.collisionRule(itemA.mesh, itemB.mesh).mode).toBe("passThrough");

    act(() => {
      setSelectValue(select, "softDeform");
    });
    expect(scene.collisionRule(itemA.mesh, itemB.mesh).mode).toBe("softDeform");

    const ungroupBtn = Array.from(groupsHost.querySelectorAll("button")).find(
      (b) => b.textContent?.trim() === "Ungroup",
    )!;
    act(() => {
      ungroupBtn.click();
    });
    expect(scene.collisionRule(itemA.mesh, itemB.mesh).mode).toBe("passThrough");
  });
});
