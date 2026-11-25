// @vitest-environment jsdom
// The Modeling panel's #254 sections wired to a real wasm Scene: the
// multi-mesh boolean lists the shift-clicked meshes and runs on three of
// them (undo restores them), and the isotropic remesh shows the triangle
// count and mean edge, defaults its target to that mean, and reports the
// counts before and after.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { MeshHandle, SceneLike } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { AppState, type Item } from "../src/app/state/app-state";
import { ModelingPanel } from "../src/features/mesh-edit/ui/ModelingPanel";
import { STATS_DELAY_MS } from "../src/features/mesh-edit/ux/use-isotropic-remesh";
import { selectMesh } from "../src/features/select/ux/selection";
import { WorkbenchRibbon } from "../src/shell/ui/Ribbon";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(() =>
  init(readFileSync(resolve(__dirname, "../../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"))),
);

const cleanups: (() => void)[] = [];

afterEach(() => {
  vi.useRealTimers();
  while (cleanups.length > 0) {
    cleanups.pop()?.();
  }
});

function addBox(store: SceneStore, app: AppState, name: string, c: readonly [number, number, number]): Item {
  const [cx, cy, cz] = c;
  const [x0, x1, y0, y1, z0, z1] = [cx - 1, cx + 1, cy - 1, cy + 1, cz - 1, cz + 1];
  return app.addMesh(store, {
    name,
    positions: Float64Array.from([x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1]),
    indices: Uint32Array.from([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5]),
    center: [cx, cy, cz],
    radius: Math.sqrt(3),
    volumeId: null,
    preset: null,
  });
}

function liveTriangles(scene: SceneLike, mesh: MeshHandle): number {
  const idx = scene.views(mesh).indices();
  let n = 0;
  for (let k = 0; k + 2 < idx.length; k += 3) {
    if (idx[k] !== 0 || idx[k + 1] !== 0 || idx[k + 2] !== 0) {
      n++;
    }
  }
  return n;
}

function setup() {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  app.follow(store);
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  cleanups.push(() => {
    act(() => root.unmount());
    container.remove();
    store.dispose();
    scene.free();
  });
  const mount = () =>
    act(() => {
      root.render(createElement(SceneProvider, { store }, createElement(ModelingPanel, { app })));
    });
  return { scene, store, app, container, mount };
}

async function click(el: Element | null): Promise<void> {
  expect(el).not.toBeNull();
  await act(async () => {
    (el as HTMLElement).click();
    for (let k = 0; k < 10; k++) {
      await Promise.resolve();
    }
  });
}

async function setChecked(el: HTMLInputElement | null, checked: boolean): Promise<void> {
  expect(el).not.toBeNull();
  if (el!.checked !== checked) {
    await click(el);
  }
}

async function typeInto(el: HTMLInputElement | null, value: string): Promise<void> {
  expect(el).not.toBeNull();
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set?.call(el, value);
    el!.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

describe("Modeling panel: multi-mesh boolean", () => {
  it("lists the shift-clicked meshes and runs on three of them, one undo restoring them", async () => {
    const { scene, store, app, container, mount } = setup();
    const a = addBox(store, app, "A", [0, 0, 0]);
    const b = addBox(store, app, "B", [1, 0, 0]);
    const c = addBox(store, app, "C", [0, 1, 0]);
    selectMesh(app, (m) => store.select(m), null, a.mesh, false);
    mount();

    const run = () => container.querySelector<HTMLButtonElement>("[data-batch-boolean-run]");
    // One mesh selected: not enough to run.
    expect(run()?.disabled).toBe(true);

    act(() => {
      selectMesh(app, (m) => store.select(m), store.getSnapshot().selection, b.mesh, true);
      selectMesh(app, (m) => store.select(m), store.getSnapshot().selection, c.mesh, true);
    });
    const listed = Array.from(container.querySelectorAll("[data-batch-boolean-inputs] li")).map((li) => li.textContent);
    expect(listed).toEqual(["A (primary)", "B", "C"]);
    expect(run()?.disabled).toBe(false);
    expect(run()?.textContent).toBe("Run on 3 meshes");

    // Primary minus the others, the inputs removed.
    await click(container.querySelector('[data-batch-boolean-op="subtract"]'));
    expect(container.querySelector('[data-batch-boolean-op="subtract"]')?.getAttribute("aria-pressed")).toBe("true");
    await setChecked(container.querySelector<HTMLInputElement>("[data-batch-boolean-keep]"), false);
    await click(run());

    const report = container.querySelector("[data-batch-boolean-report]");
    expect(report?.classList.contains("error")).toBe(false);
    expect(report?.textContent).toMatch(/^Multi-mesh boolean: Difference of 3 meshes → A: \d+ triangles/);
    expect(app.getSnapshot().items).toEqual([a]);
    expect(store.getSnapshot().selection).toBe(a.mesh);

    act(() => store.undo());
    expect(app.getSnapshot().items).toEqual([a, b, c]);
    expect([a, b, c].map((i) => liveTriangles(scene, i.mesh))).toEqual([12, 12, 12]);
  });

  it("keeps the inputs by default and adds the result as a new mesh", async () => {
    const { store, app, container, mount } = setup();
    const inputs = [addBox(store, app, "A", [0, 0, 0]), addBox(store, app, "B", [1, 0, 0]), addBox(store, app, "C", [0, 1, 0])];
    inputs.forEach((it, i) => selectMesh(app, (m) => store.select(m), store.getSnapshot().selection, it.mesh, i > 0));
    mount();

    expect(container.querySelector<HTMLInputElement>("[data-batch-boolean-keep]")?.checked).toBe(true);
    expect(container.querySelector<HTMLSelectElement>("[data-batch-boolean-method]")?.value).toBe("manifold");
    await click(container.querySelector("[data-batch-boolean-run]"));
    expect(app.getSnapshot().items.map((i) => i.name)).toEqual(["A", "B", "C", "A union +2"]);
    expect(container.querySelector("[data-batch-boolean-report]")?.textContent).toContain("inputs kept");
  });
});

describe("Mesh Edit ribbon", () => {
  it("brings each #254 section into view", async () => {
    const { store, app } = setup();
    const a = addBox(store, app, "A", [0, 0, 0]);
    store.select(a.mesh);
    const ribbon = document.createElement("div");
    document.body.appendChild(ribbon);
    const root = createRoot(ribbon);
    cleanups.push(() => {
      act(() => root.unmount());
      ribbon.remove();
    });
    act(() => {
      root.render(
        createElement(
          SceneProvider,
          { store },
          createElement(WorkbenchRibbon, {
            app,
            mode: "translate",
            canUndo: false,
            canRedo: false,
            wireframe: false,
            selection: a.mesh,
            hasItems: true,
            hasVolume: false,
            onSetMode: () => {},
            onResetView: () => {},
            onFocusSelection: () => {},
            onClearSelection: () => {},
            onUndo: () => {},
            onRedo: () => {},
            onOpenHelp: () => {},
          }),
          createElement(ModelingPanel, { app }),
        ),
      );
    });
    await click(ribbon.querySelector('[data-ribbon-tab="edit"]'));
    for (const [focus, section] of [
      ["batch-boolean", "[data-batch-boolean-section]"],
      ["isotropic-remesh", "[data-isotropic-section]"],
    ] as const) {
      const target = ribbon.querySelector<HTMLElement>(section);
      expect(target).not.toBeNull();
      const seen = vi.fn();
      target!.scrollIntoView = seen;
      await click(ribbon.querySelector(`[data-ribbon-focus="${focus}"]`));
      expect(seen).toHaveBeenCalled();
    }
  });
});

describe("Modeling panel: isotropic remesh", () => {
  it("defaults the target to the mean edge length and reports the triangle counts", async () => {
    vi.useFakeTimers();
    const { scene, store, app, container, mount } = setup();
    const a = addBox(store, app, "A", [0, 0, 0]);
    store.select(a.mesh);
    mount();

    // A 2-unit cube: 12 triangles, edges 2 and the face diagonals 2√2.
    const mean = (8 * 2 + 4 * 2 * Math.SQRT2) / 12;
    const stats = container.querySelector("[data-isotropic-stats]")?.textContent ?? "";
    expect(stats).toContain("A: 12 triangles");
    const edge = container.querySelector<HTMLInputElement>("[data-isotropic-edge]");
    expect(Number(edge?.value)).toBeCloseTo(Number(mean.toPrecision(3)), 10);
    expect(container.querySelector("[data-isotropic-edge-mean]")?.getAttribute("aria-pressed")).toBe("true");

    await typeInto(edge, "0.5");
    expect(container.querySelector("[data-isotropic-edge-mean]")?.getAttribute("aria-pressed")).toBe("false");
    // About 24 / (√3/4 · 0.25) triangles.
    expect(container.querySelector("[data-isotropic-stats]")?.textContent).toContain("about 222 triangles");
    await typeInto(container.querySelector<HTMLInputElement>("[data-isotropic-iterations]"), "3");
    await setChecked(container.querySelector<HTMLInputElement>("[data-isotropic-creases]"), false);
    expect(container.querySelector("[data-isotropic-crease-angle]")).toBeNull();

    const repair = vi.spyOn(store, "repair");
    await click(container.querySelector("[data-isotropic-run]"));
    expect(repair).toHaveBeenCalledWith(a.mesh, {
      op: "remeshSurface",
      targetEdgeLength: 0.5,
      iterations: 3,
      lockBoundaries: true,
      lockCreases: false,
      mode: "isotropic",
    });
    const report = container.querySelector("[data-isotropic-report]");
    expect(report?.classList.contains("error")).toBe(false);
    const after = liveTriangles(scene, a.mesh);
    expect(after).toBeGreaterThan(12);
    expect(report?.textContent).toContain(`12 → ${after.toLocaleString("en")} triangles`);
    expect(report?.textContent).toContain("in 3 passes");

    // The statistics follow the remeshed mesh once it rests, and its undo.
    await act(async () => {
      store.commitFrame();
      vi.advanceTimersByTime(STATS_DELAY_MS + 1);
    });
    expect(container.querySelector("[data-isotropic-stats]")?.textContent).toContain(`A: ${after.toLocaleString("en")} triangles`);
    await act(async () => {
      store.undo();
      store.commitFrame();
      vi.advanceTimersByTime(STATS_DELAY_MS + 1);
    });
    expect(container.querySelector("[data-isotropic-stats]")?.textContent).toContain("A: 12 triangles");
  });
});
