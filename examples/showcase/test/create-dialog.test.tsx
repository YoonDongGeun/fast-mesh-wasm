// @vitest-environment jsdom
// The Create dialog's wiring (#259): the Home shelf button opens it, shape
// buttons and fields drive the live preview, lengths follow the display
// unit, bad values disable Create, and Create adds the shape as one undo
// step, selects it and closes the dialog. Real wasm Scene and SceneStore.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { AppState } from "../src/app/state/app-state";
import { CreatePrimitiveDialog } from "../src/features/home/ui/CreatePrimitiveDialog";
import { buildCreatePreview } from "../src/features/home/ui/CreatePreviewLayer";
import { createDialogOf } from "../src/features/home/ux/create-dialog-state";
import { WorkbenchRibbon } from "../src/shell/ui/Ribbon";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// The synchronous scene generates on the main thread through the lazy
// module; under jsdom it is the web build initialised below (the bare
// package name would resolve to the Node build, which reads its .wasm by a
// file: URL jsdom does not give it).
vi.mock("../src/engine/wave6-wasm-lazy", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/engine/wave6-wasm-lazy")>();
  const web = await import("../../../packages/fast-mesh-wasm/dist/web.js");
  return { ...actual, ensureWave6Wasm: async () => web };
});

beforeAll(() =>
  init(readFileSync(resolve(__dirname, "../../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"))),
);

const cleanups: (() => Promise<void> | void)[] = [];
afterEach(async () => {
  for (const c of cleanups.splice(0).reverse()) {
    await c();
  }
});

async function mount(): Promise<{ app: AppState; store: SceneStore; scene: Scene; container: HTMLElement }> {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  const container = document.createElement("div");
  document.body.appendChild(container);
  let root: Root;
  await act(async () => {
    root = createRoot(container);
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
          selection: null,
          hasItems: false,
          hasVolume: false,
          onSetMode: () => {},
          onResetView: () => {},
          onFocusSelection: () => {},
          onClearSelection: () => {},
          onUndo: () => {},
          onRedo: () => {},
          onOpenHelp: () => {},
        }),
        createElement(CreatePrimitiveDialog, { app }),
      ),
    );
  });
  cleanups.push(async () => {
    await act(async () => root.unmount());
    container.remove();
    store.dispose();
    scene.free();
  });
  return { app, store, scene, container };
}

/** Types `value` into a controlled input, as a user would. */
function type(input: HTMLInputElement, value: string): void {
  const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
  set.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

const q = <T extends Element>(root: ParentNode, sel: string) => root.querySelector<T>(sel)!;

async function previewed(app: AppState): Promise<number> {
  let triangles = 0;
  await vi.waitFor(
    () => {
      const p = createDialogOf(app).getSnapshot().preview;
      expect(p, document.querySelector("[data-create-status]")?.textContent ?? "").not.toBeNull();
      triangles = p!.mesh.indices.length / 3;
    },
    { timeout: 3000 },
  );
  return triangles;
}

describe("Create dialog", () => {
  it("opens from the Home shelf, previews live, follows the unit and creates one undoable mesh", async () => {
    const { app, scene, store, container } = await mount();
    expect(container.querySelector("[data-panel=create-primitive]")).toBeNull();
    await act(async () => q<HTMLButtonElement>(container, '[data-action="create-primitive"]').click());
    const dialog = q<HTMLElement>(container, "[data-panel=create-primitive]");
    expect(dialog.querySelectorAll("[data-create-shape]")).toHaveLength(9);
    // A box by default, 2 mm in an empty scene, previewed as 12 triangles.
    expect(q<HTMLButtonElement>(dialog, '[data-create-shape="box"]').getAttribute("aria-pressed")).toBe("true");
    expect(q<HTMLInputElement>(dialog, '[data-create-param="sizeX"]').value).toBe("2");
    expect(await previewed(app)).toBe(12);

    // A cylinder, its radius typed in inches.
    await act(async () => q<HTMLButtonElement>(dialog, '[data-create-shape="cylinder"]').click());
    await act(async () => app.setUnits("in"));
    const radius = q<HTMLInputElement>(dialog, '[data-create-param="radius"]');
    expect(Number(radius.value)).toBeCloseTo(1 / 25.4, 9);
    expect(dialog.textContent).toContain("Radius (in)");
    await act(async () => type(radius, "0.5"));
    await act(async () => type(q<HTMLInputElement>(dialog, '[data-create-param="segments"]'), "16"));
    await vi.waitFor(() => expect(createDialogOf(app).getSnapshot().preview?.mesh.indices.length).toBe(3 * 4 * 16), {
      timeout: 3000,
    });
    expect(q<HTMLElement>(dialog, "[data-create-status]").textContent).toContain("64 triangles");
    expect(q<HTMLInputElement>(dialog, "[data-create-name]").value).toBe("Cylinder");

    // Centre and rotation move the preview without regenerating it.
    const mesh = createDialogOf(app).getSnapshot().preview!.mesh;
    await act(async () => type(q<HTMLInputElement>(dialog, '[data-create-centre="x"]'), "1"));
    await act(async () => type(q<HTMLInputElement>(dialog, '[data-create-rotation="z"]'), "90"));
    const preview = createDialogOf(app).getSnapshot().preview!;
    expect(preview.mesh).toBe(mesh);
    expect(preview.centre[0]).toBeCloseTo(25.4, 9);
    expect(preview.rotationDeg).toEqual([0, 0, 90]);

    await act(async () => q<HTMLButtonElement>(dialog, '[data-action="create-primitive-confirm"]').click());
    await vi.waitFor(() => expect(app.getSnapshot().items).toHaveLength(1), { timeout: 3000 });
    await vi.waitFor(() => expect(container.querySelector("[data-panel=create-primitive]")).toBeNull());
    const item = app.getSnapshot().items[0];
    expect(item).toMatchObject({ name: "Cylinder", triangles: 64 });
    expect(item.center[0]).toBeCloseTo(25.4, 9);
    expect(store.getSnapshot().selection).toBe(item.mesh);
    expect(createDialogOf(app).getSnapshot().preview).toBeNull();
    const m = scene.inspect.measure(item.mesh);
    expect(m.closed).toBe(true);
    // 12.7 mm radius, 2 mm high, 16 sides.
    expect(m.volume).toBeCloseTo(8 * 12.7 ** 2 * Math.sin(Math.PI / 8) * 2, 1);
    // One undo step.
    expect(scene.canUndo()).toBe(true);
    await act(async () => store.undo());
    expect(scene.canUndo()).toBe(false);
    expect(scene.inspect.measure(item.mesh).area).toBe(0);
  });

  it("names repeats uniquely, refuses bad values and closes on Escape", async () => {
    const { app, container } = await mount();
    await act(async () => createDialogOf(app).open("torus"));
    let dialog = q<HTMLElement>(container, "[data-panel=create-primitive]");
    await previewed(app);
    await act(async () => q<HTMLButtonElement>(dialog, '[data-action="create-primitive-confirm"]').click());
    await vi.waitFor(() => expect(app.getSnapshot().items).toHaveLength(1), { timeout: 3000 });

    await act(async () => createDialogOf(app).open("torus"));
    dialog = q<HTMLElement>(container, "[data-panel=create-primitive]");
    expect(q<HTMLInputElement>(dialog, "[data-create-name]").value).toBe("Torus 2");
    // A tube as thick as its ring: refused, no preview, Create disabled.
    await act(async () => type(q<HTMLInputElement>(dialog, '[data-create-param="minorRadius"]'), "100"));
    expect(q<HTMLElement>(dialog, "[data-create-status]").textContent).toMatch(/tube radius/i);
    expect(q<HTMLButtonElement>(dialog, '[data-action="create-primitive-confirm"]').disabled).toBe(true);
    expect(createDialogOf(app).getSnapshot().preview).toBeNull();
    // Reset values brings the defaults back.
    await act(async () => q<HTMLButtonElement>(dialog, '[data-action="create-primitive-reset"]').click());
    expect(q<HTMLButtonElement>(dialog, '[data-action="create-primitive-confirm"]').disabled).toBe(false);

    await act(async () => {
      q<HTMLInputElement>(dialog, "[data-create-name]").dispatchEvent(
        new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
      );
    });
    expect(container.querySelector("[data-panel=create-primitive]")).toBeNull();
    expect(createDialogOf(app).getSnapshot()).toMatchObject({ open: false, preview: null });
    expect(app.getSnapshot().items).toHaveLength(1);
  });

  it("builds a preview that takes no clicks, with a wireframe on small shapes only", () => {
    const small = buildCreatePreview({
      positions: Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      indices: Uint32Array.from([0, 1, 2]),
    });
    expect(small.root.children).toHaveLength(2);
    const hits: unknown[] = [];
    small.root.children[0].raycast({} as never, hits as never);
    expect(hits).toEqual([]);
    small.dispose();
    const big = buildCreatePreview({ positions: new Float32Array(9), indices: new Uint32Array(3 * 200_001) });
    expect(big.root.children).toHaveLength(1);
    big.dispose();
  });
});
