// @vitest-environment jsdom
// The Text tool's wiring (#260): the Home and Mesh Edit shelves open the
// Text dialog, which loads the font, previews the text live, follows the
// display unit, and creates it as one undoable, closed mesh; Emboss and
// Engrave arm Place Mesh with the text; the dialog and the Create dialog
// replace each other; and the Surface Stamp section edits the same text.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, createElement, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { AppState } from "../src/app/state/app-state";
import { CreatePrimitiveDialog } from "../src/features/home/ui/CreatePrimitiveDialog";
import { createDialogOf } from "../src/features/home/ux/create-dialog-state";
import { ModelingPanel } from "../src/features/mesh-edit/ui/ModelingPanel";
import { TextDialog } from "../src/features/mesh-edit/ui/TextDialog";
import { textDialogOf } from "../src/features/mesh-edit/ux/text-dialog-state";
import { defaultTextName } from "../src/features/mesh-edit/ux/use-text-tool";
import { WorkbenchRibbon } from "../src/shell/ui/Ribbon";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(() =>
  init(readFileSync(resolve(__dirname, "../../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"))),
);

const cleanups: (() => Promise<void> | void)[] = [];
afterEach(async () => {
  for (const c of cleanups.splice(0).reverse()) {
    await c();
  }
});

async function mount(
  children: (app: AppState) => ReactElement[],
  setup?: (app: AppState, store: SceneStore) => void,
): Promise<{ app: AppState; store: SceneStore; scene: Scene; container: HTMLElement }> {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  setup?.(app, store);
  const container = document.createElement("div");
  document.body.appendChild(container);
  let root: Root;
  await act(async () => {
    root = createRoot(container);
    root.render(createElement(SceneProvider, { store }, ...children(app)));
  });
  cleanups.push(async () => {
    await act(async () => root.unmount());
    container.remove();
    store.dispose();
    scene.free();
  });
  return { app, store, scene, container };
}

const ribbon = (app: AppState, activeTab?: "edit") => {
  if (activeTab !== undefined) {
    app.setActiveTab(activeTab);
  }
  return createElement(WorkbenchRibbon, {
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
  });
};

/** Types `value` into a controlled input or text box, as a user would. */
function type(input: HTMLInputElement | HTMLTextAreaElement, value: string): void {
  const proto = input instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, "value")!.set!.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

const q = <T extends Element>(root: ParentNode, sel: string) => root.querySelector<T>(sel)!;

/** The preview's triangle count once it shows `text`'s latest build. */
async function previewed(app: AppState, dialog: HTMLElement): Promise<number> {
  let triangles = 0;
  await vi.waitFor(
    () => {
      const status = q<HTMLElement>(dialog, "[data-text-status]").textContent ?? "";
      expect(status).toMatch(/triangles/);
      expect(status).not.toMatch(/updating/);
      triangles = textDialogOf(app).getSnapshot().preview!.mesh.indices.length / 3;
    },
    { timeout: 5000 },
  );
  return triangles;
}

describe("Text dialog", () => {
  it("opens from the Home shelf, previews live, follows the unit and creates one undoable mesh", async () => {
    const { app, scene, store, container } = await mount((app) => [ribbon(app), createElement(TextDialog, { app, key: "t" })]);
    expect(container.querySelector("[data-panel=text-tool]")).toBeNull();
    await act(async () => q<HTMLButtonElement>(container, '[data-action="create-text"]').click());
    const dialog = q<HTMLElement>(container, "[data-panel=text-tool]");
    const first = await previewed(app, dialog);
    expect(first).toBeGreaterThan(0);
    expect(q<HTMLInputElement>(dialog, "[data-text-name]").value).toBe("Text");

    await act(async () => type(q<HTMLTextAreaElement>(dialog, '[data-text-param="text"]'), "AB8"));
    await act(async () => app.setUnits("in"));
    expect(dialog.textContent).toContain("Letter height (in)");
    const size = q<HTMLInputElement>(dialog, '[data-text-param="size"]');
    expect(Number(size.value)).toBeCloseTo(5 / 25.4, 9);
    await act(async () => type(size, "0.5"));
    const triangles = await previewed(app, dialog);
    expect(app.getSnapshot().wave6.printPrep.placeParams.text).toMatchObject({ text: "AB8" });
    expect(app.getSnapshot().wave6.printPrep.placeParams.text.size).toBeCloseTo(12.7, 9);
    expect(q<HTMLInputElement>(dialog, "[data-text-name]").value).toBe("AB8");
    expect(q<HTMLElement>(dialog, "[data-text-status]").textContent).toContain(`${triangles.toLocaleString("en")} triangles`);

    // Centre and rotation move the preview without rebuilding it.
    const mesh = textDialogOf(app).getSnapshot().preview!.mesh;
    await act(async () => type(q<HTMLInputElement>(dialog, '[data-text-centre="x"]'), "1"));
    await act(async () => type(q<HTMLInputElement>(dialog, '[data-text-rotation="x"]'), "-90"));
    const preview = textDialogOf(app).getSnapshot().preview!;
    expect(preview.mesh).toBe(mesh);
    expect(preview.centre[0]).toBeCloseTo(25.4, 9);
    expect(preview.rotationDeg).toEqual([-90, 0, 0]);

    await act(async () => q<HTMLButtonElement>(dialog, '[data-action="text-create"]').click());
    await vi.waitFor(() => expect(app.getSnapshot().items).toHaveLength(1), { timeout: 3000 });
    await vi.waitFor(() => expect(container.querySelector("[data-panel=text-tool]")).toBeNull());
    const item = app.getSnapshot().items[0];
    expect(item).toMatchObject({ name: "AB8", triangles });
    expect(store.getSnapshot().selection).toBe(item.mesh);
    expect(textDialogOf(app).getSnapshot().preview).toBeNull();
    const m = scene.inspect.measure(item.mesh);
    expect(m.closed).toBe(true);
    expect(m.orientationConflicts).toBe(0);
    expect(m.volume).toBeGreaterThan(0);
    // One undo step: undo removes the mesh and its row.
    expect(scene.canUndo()).toBe(true);
    await act(async () => store.undo());
    expect(scene.canUndo()).toBe(false);
    expect(scene.has(item.mesh)).toBe(false);
  });

  it("arms Place Mesh with the text to emboss or engrave it, from the Mesh Edit shelf", async () => {
    const { app, container } = await mount((app) => [ribbon(app, "edit"), createElement(TextDialog, { app, key: "t" })]);
    await act(async () => q<HTMLButtonElement>(container, '[data-action="text-tool"]').click());
    let dialog = q<HTMLElement>(container, "[data-panel=text-tool]");
    await previewed(app, dialog);
    await act(async () => type(q<HTMLTextAreaElement>(dialog, '[data-text-param="text"]'), "Hi"));
    await act(async () => q<HTMLButtonElement>(dialog, '[data-action="text-emboss"]').click());
    expect(container.querySelector("[data-panel=text-tool]")).toBeNull();
    let prep = app.getSnapshot().wave6.printPrep;
    expect(prep.activeTool).toBe("placeStamp");
    expect(prep.placeParams).toMatchObject({ stampSource: "text", mode: "embedUnion", conformToSurface: true, text: { text: "Hi" } });
    expect(app.getSnapshot().activeTab).toBe("edit");

    await act(async () => textDialogOf(app).open());
    dialog = q<HTMLElement>(container, "[data-panel=text-tool]");
    // The dialog opens on the same text.
    expect(q<HTMLTextAreaElement>(dialog, '[data-text-param="text"]').value).toBe("Hi");
    await previewed(app, dialog);
    await act(async () => q<HTMLButtonElement>(dialog, '[data-action="text-engrave"]').click());
    prep = app.getSnapshot().wave6.printPrep;
    expect(prep.placeParams.mode).toBe("engraveSubtract");
    expect(prep.activeTool).toBe("placeStamp");
  });

  it("refuses bad values, closes on Escape, and gives way to the Create dialog", async () => {
    const { app, container } = await mount((app) => [
      createElement(TextDialog, { app, key: "t" }),
      createElement(CreatePrimitiveDialog, { app, key: "c" }),
    ]);
    await act(async () => textDialogOf(app).open());
    let dialog = q<HTMLElement>(container, "[data-panel=text-tool]");
    await previewed(app, dialog);
    await act(async () => type(q<HTMLInputElement>(dialog, '[data-text-param="depth"]'), "0"));
    expect(q<HTMLElement>(dialog, "[data-text-status]").textContent).toMatch(/depth must be above zero/i);
    expect(q<HTMLButtonElement>(dialog, '[data-action="text-create"]').disabled).toBe(true);
    expect(q<HTMLButtonElement>(dialog, '[data-action="text-emboss"]').disabled).toBe(true);
    expect(textDialogOf(app).getSnapshot().preview).toBeNull();
    await act(async () => type(q<HTMLInputElement>(dialog, '[data-text-param="depth"]'), "2"));

    await act(async () => createDialogOf(app).open());
    expect(container.querySelector("[data-panel=text-tool]")).toBeNull();
    expect(container.querySelector("[data-panel=create-primitive]")).not.toBeNull();
    await act(async () => textDialogOf(app).open());
    expect(container.querySelector("[data-panel=create-primitive]")).toBeNull();
    dialog = q<HTMLElement>(container, "[data-panel=text-tool]");
    await act(async () => {
      q<HTMLInputElement>(dialog, "[data-text-name]").dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(container.querySelector("[data-panel=text-tool]")).toBeNull();
    expect(textDialogOf(app).getSnapshot()).toMatchObject({ open: false, preview: null });
    expect(app.getSnapshot().items).toHaveLength(0);
  });

  it("names a text by its first line", () => {
    expect(defaultTextName("  \n  Part 42 \nrev B")).toBe("Part 42");
    expect(defaultTextName(" ")).toBe("Text");
    expect(defaultTextName("x".repeat(40))).toBe(`${"x".repeat(23)}…`);
  });
});

describe("Surface Stamp with a text", () => {
  it("shows the text fields when the stamp is Text, editing the dialog's text", async () => {
    const { app, container } = await mount(
      (app) => [createElement(ModelingPanel, { app, key: "m" })],
      (app, store) => {
        app.addMesh(store, {
          name: "Block",
          positions: Float64Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
          indices: Uint32Array.from([0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 3, 2]),
          center: [0, 0, 0],
          radius: 1,
          volumeId: null,
          preset: null,
        });
      },
    );
    const section = q<HTMLElement>(container, "[data-modeling-place-section]");
    expect(section.querySelector("[data-modeling-place-text]")).toBeNull();
    const source = q<HTMLSelectElement>(section, "[data-modeling-place-source]");
    expect(Array.from(source.options).map((o) => o.value)).toContain("text");
    await act(async () => {
      source.value = "text";
      source.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(app.getSnapshot().wave6.printPrep.placeParams.stampSource).toBe("text");
    const fields = q<HTMLElement>(section, "[data-modeling-place-text]");
    expect(q<HTMLElement>(fields, "[data-modeling-place-text-note]").textContent).toBe(
      "Raises the text 1 mm out of the surface.",
    );
    await act(async () => type(q<HTMLTextAreaElement>(fields, "[data-modeling-place-text-value]"), "No. 7"));
    await act(async () => type(q<HTMLInputElement>(fields, "[data-modeling-place-text-depth]"), "3"));
    expect(app.getSnapshot().wave6.printPrep.placeParams.text).toMatchObject({ text: "No. 7", depth: 3 });
    expect(q<HTMLElement>(fields, "[data-modeling-place-text-note]").textContent).toBe(
      "Raises the text 2.5 mm out of the surface.",
    );
    const mode = q<HTMLSelectElement>(section, "[data-modeling-place-mode]");
    await act(async () => {
      mode.value = "engraveSubtract";
      mode.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(q<HTMLElement>(fields, "[data-modeling-place-text-note]").textContent).toBe(
      "Cuts the text 0.5 mm into the surface.",
    );
    await act(async () => q<HTMLButtonElement>(fields, "[data-modeling-place-text-dialog]").click());
    expect(textDialogOf(app).getSnapshot().open).toBe(true);
  });
});
