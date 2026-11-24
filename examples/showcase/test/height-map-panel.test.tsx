// @vitest-environment jsdom
// The height map tools' wiring (#267) on a real engine: the export panel's
// section (settings, the display unit, its two downloads, its errors), the
// Image to Relief dialog (opened from the Mesh Edit shelf or by a dropped
// image, its settings and live preview, a height map's JSON, one undo step)
// and the command palette's entries.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Blob as NodeBlob, File as NodeFile } from "node:buffer";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, createElement, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { generatePrimitive, init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { AppState } from "../src/app/state/app-state";
import { CreatePrimitiveDialog } from "../src/features/home/ui/CreatePrimitiveDialog";
import { ExportPanel } from "../src/features/home/ui/ExportPanel";
import { createDialogOf } from "../src/features/home/ux/create-dialog-state";
import { ReliefDialog } from "../src/features/mesh-edit/ui/ReliefDialog";
import { TextDialog } from "../src/features/mesh-edit/ui/TextDialog";
import { reliefDialogOf } from "../src/features/mesh-edit/ux/relief-dialog-state";
import { reliefCommands } from "../src/features/mesh-edit/ux/relief-commands";
import { reliefTriangles } from "../src/features/mesh-edit/ux/relief";
import { textDialogOf } from "../src/features/mesh-edit/ux/text-dialog-state";
import { useReliefDrop } from "../src/features/mesh-edit/ux/use-relief-drop";
import { MAX_CODE, heightMapFile } from "../src/io/height-image";
import { decodePng, encodePng16 } from "../src/io/png";
import { WorkbenchRibbon } from "../src/shell/ui/Ribbon";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// The lazily loaded module is the web build initialised below (the Node entry
// reads its wasm from disk, which jsdom's URLs do not name).
vi.mock("../src/engine/wave6-wasm-lazy", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/engine/wave6-wasm-lazy")>();
  return { ...actual, ensureWave6Wasm: () => import("../../../packages/fast-mesh-wasm/dist/web.js") };
});

beforeAll(() =>
  init(readFileSync(resolve(__dirname, "../../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"))),
);

const cleanups: (() => Promise<void> | void)[] = [];
afterEach(async () => {
  for (const c of cleanups.splice(0).reverse()) {
    await c();
  }
  vi.restoreAllMocks();
});

async function mount(
  children: (app: AppState, scene: Scene) => ReactElement[],
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
    root.render(createElement(SceneProvider, { store }, ...children(app, scene)));
  });
  cleanups.push(async () => {
    await act(async () => root.unmount());
    container.remove();
    store.dispose();
    scene.free();
  });
  return { app, store, scene, container };
}

const q = <T extends Element>(root: ParentNode, sel: string): T => {
  const el = root.querySelector<T>(sel);
  if (el === null) {
    throw new Error(`no ${sel}`);
  }
  return el;
};

/** Types `value` into a controlled input, as a user would. */
function type(input: HTMLInputElement, value: string): void {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

function choose(select: HTMLSelectElement, value: string): void {
  Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")!.set!.call(select, value);
  select.dispatchEvent(new Event("change", { bubbles: true }));
}

const box = (size: [number, number, number]) => generatePrimitive({ kind: "box", size });

/** Lists a box in the app. */
function addBox(app: AppState, store: SceneStore, name: string, size: [number, number, number]) {
  const b = box(size);
  const mesh = store.addMesh(b.positions, b.indices);
  return app.listMesh(mesh, { name, triangles: b.indices.length / 3, center: [0, 0, 0], radius: 1 });
}

/** A `File` with what the code reads of it (jsdom's own lacks `arrayBuffer`). */
function file(parts: (string | Uint8Array)[], name: string, mime: string): File {
  return new NodeFile(parts as never, name, { type: mime }) as unknown as File;
}

/** A grey PNG whose brightness is `value(col, row)` in `[0, 1]`. */
async function greyPng(w: number, h: number, value: (col: number, row: number) => number): Promise<Uint8Array> {
  const codes = Uint16Array.from({ length: w * h }, (_, i) => Math.round(value(i % w, Math.floor(i / w)) * MAX_CODE));
  return encodePng16(codes, w, h);
}

/** Captures the files the page downloads. */
function captureDownloads(): { name: string; blob: NodeBlob }[] {
  const files: { name: string; blob: NodeBlob }[] = [];
  const blobs = new Map<string, NodeBlob>();
  let n = 0;
  const url = URL as unknown as { createObjectURL?: unknown; revokeObjectURL?: unknown };
  const before = [url.createObjectURL, url.revokeObjectURL];
  url.createObjectURL = (b: NodeBlob) => {
    const id = `blob:${++n}`;
    blobs.set(id, b);
    return id;
  };
  url.revokeObjectURL = () => {};
  cleanups.push(() => {
    [url.createObjectURL, url.revokeObjectURL] = before;
  });
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    files.push({ name: this.download, blob: blobs.get(this.href)! });
  });
  return files;
}

/** The bytes of a blob made in the page's realm. */
async function bytesOf(blob: NodeBlob): Promise<Uint8Array> {
  if (typeof blob.arrayBuffer === "function") {
    return new Uint8Array(await blob.arrayBuffer());
  }
  return new Promise((done, fail) => {
    const reader = new FileReader();
    reader.onload = () => done(new Uint8Array(reader.result as ArrayBuffer));
    reader.onerror = () => fail(reader.error);
    reader.readAsArrayBuffer(blob as unknown as Blob);
  });
}

describe("export panel", () => {
  const panel = (app: AppState, scene: Scene) => [createElement(ExportPanel, { app, scene: scene as never, key: "e" })];

  it("exports the picked mesh as a 16-bit PNG and its JSON", async () => {
    const files = captureDownloads();
    const { container } = await mount(panel, (app, store) => void addBox(app, store, "Block", [4, 1, 2]));
    const resolution = q<HTMLInputElement>(container, '[data-height-map="resolution"]');
    expect(resolution.value).toBe("1024");
    await act(async () => type(resolution, "32"));
    await act(async () => q<HTMLButtonElement>(container, '[data-action="export-height-map"]').click());
    await vi.waitFor(() => expect(files).toHaveLength(2), { timeout: 5000 });
    expect(files.map((f) => f.name)).toEqual(["Block-height-map.png", "Block-height-map.json"]);
    const png = await decodePng(await bytesOf(files[0].blob));
    expect([png.width, png.height, png.depth]).toEqual([32, 16, 16]);
    const doc = JSON.parse(Buffer.from(await bytesOf(files[1].blob)).toString("utf8")) as Record<string, unknown>;
    expect(doc).toMatchObject({ image: "Block-height-map.png", pixelSize: 0.125, heightRange: 1, meshes: ["Block"] });
    await vi.waitFor(() =>
      expect(q(container, "[data-height-map-status]").textContent).toBe(
        "32 × 16 px · 0.125 mm per pixel · heights 0 to 1 mm",
      ),
    );
    expect(q<HTMLButtonElement>(container, '[data-action="export-height-map"]').disabled).toBe(false);
  });

  it("starts a typed range at the meshes' own, in the display unit, and clips to it", async () => {
    const files = captureDownloads();
    const { app, container, scene, store } = await mount(panel, (app, store) => {
      const item = addBox(app, store, "Block", [4, 2, 2]);
      store.setPlacement(item.mesh, { rotation: [0, 0, 0, 1], translation: [0, 10, 0] });
    });
    store.commitFrame();
    scene.commitFrame();
    await act(async () => type(q<HTMLInputElement>(container, '[data-height-map="resolution"]'), "16"));
    await act(async () => q<HTMLInputElement>(container, '[data-height-map="custom-range"]').click());
    await vi.waitFor(() => expect(q<HTMLInputElement>(container, '[data-height-map="low"]').value).toBe("9"));
    expect(q<HTMLInputElement>(container, '[data-height-map="high"]').value).toBe("11");
    await act(async () => app.setUnits("in"));
    expect(container.textContent).toContain("lowest (in)");
    await act(async () => type(q<HTMLInputElement>(container, '[data-height-map="high"]'), String(19 / 25.4)));
    // Lowest 9, highest 19 mm: the top at 11 is a fifth of the way up.
    await act(async () => q<HTMLButtonElement>(container, '[data-action="export-height-map"]').click());
    await vi.waitFor(() => expect(files).toHaveLength(2), { timeout: 5000 });
    const png = await decodePng(await bytesOf(files[0].blob));
    expect(png.samples[0]).toBe(Math.round(0.2 * MAX_CODE));
    // The lowest above the highest: refused, with the reason.
    await act(async () => type(q<HTMLInputElement>(container, '[data-height-map="low"]'), "5"));
    expect(q(container, "[data-height-map-status]").textContent).toMatch(/lowest height must be below/);
    expect(q<HTMLButtonElement>(container, '[data-action="export-height-map"]').disabled).toBe(true);
    // Another view starts over from the meshes' own range.
    await act(async () => choose(q<HTMLSelectElement>(container, '[data-height-map="view"]'), "left"));
    expect(q<HTMLInputElement>(container, '[data-height-map="custom-range"]').checked).toBe(false);
  });

  it("shows every visible mesh when asked, and says when there is nothing to see or look from", async () => {
    const files = captureDownloads();
    const { app, container, store, scene } = await mount(panel, (app, store) => {
      addBox(app, store, "A", [2, 2, 2]);
      const b = addBox(app, store, "B", [2, 2, 2]);
      store.setPlacement(b.mesh, { rotation: [0, 0, 0, 1], translation: [6, 0, 0] });
      const hidden = addBox(app, store, "C", [2, 2, 2]);
      store.setPlacement(hidden.mesh, { rotation: [0, 0, 0, 1], translation: [0, 0, 50] });
      store.setDisplay(hidden.mesh, { visible: false });
    });
    store.commitFrame();
    scene.commitFrame();
    await act(async () => type(q<HTMLInputElement>(container, '[data-height-map="resolution"]'), "8"));
    await act(async () => choose(q<HTMLSelectElement>(container, '[data-height-map="source"]'), "visible"));
    await act(async () => q<HTMLButtonElement>(container, '[data-action="export-height-map"]').click());
    await vi.waitFor(() => expect(files).toHaveLength(2), { timeout: 5000 });
    expect(files[0].name).toBe("meshes-height-map.png");
    const doc = JSON.parse(Buffer.from(await bytesOf(files[1].blob)).toString("utf8")) as { meshes: string[]; width: number };
    // 8 mm across for two boxes 2 mm wide and 4 mm apart; the hidden one is out.
    expect(doc.meshes).toEqual(["A", "B"]);
    expect([doc.width, (await decodePng(await bytesOf(files[0].blob))).height]).toEqual([8, 2]);

    await act(async () => choose(q<HTMLSelectElement>(container, '[data-height-map="view"]'), "camera"));
    await act(async () => q<HTMLButtonElement>(container, '[data-action="export-height-map"]').click());
    await vi.waitFor(() => expect(q(container, "[data-height-map-status]").textContent).toMatch(/no camera/));
    expect(files).toHaveLength(2);
    expect(app.getSnapshot().items).toHaveLength(3);
  });
});

describe("image to relief dialog", () => {
  const dialogs = (app: AppState) => [
    createElement(ReliefDialog, { app, key: "r" }),
    createElement(TextDialog, { app, key: "t" }),
    createElement(CreatePrimitiveDialog, { app, key: "c" }),
  ];
  const ribbon = (app: AppState) => {
    app.setActiveTab("edit");
    return createElement(WorkbenchRibbon, {
      key: "rib",
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
  const panelOf = (c: HTMLElement) => q<HTMLElement>(c, "[data-panel=image-to-relief]");
  const status = (c: HTMLElement) => q(c, "[data-relief-status]").textContent ?? "";
  const param = (c: HTMLElement, key: string) => q<HTMLInputElement>(c, `[data-relief-param="${key}"]`);

  /** Hands the dialog files as its file input would. */
  async function pick(container: HTMLElement, files: File[]): Promise<void> {
    const input = q<HTMLInputElement>(container, "[data-relief-file]");
    Object.defineProperty(input, "files", { value: files, configurable: true });
    await act(async () => {
      input.dispatchEvent(new Event("change", { bubbles: true }));
    });
  }

  /** Waits until `check` passes, letting React apply what arrives meanwhile. */
  async function until(check: () => void): Promise<void> {
    const deadline = Date.now() + 8000;
    for (;;) {
      await act(async () => {
        await new Promise((r) => setTimeout(r, 25));
      });
      try {
        check();
        return;
      } catch (e) {
        if (Date.now() > deadline) {
          throw e;
        }
      }
    }
  }

  /** Waits for the settings and the preview of the image. */
  async function previewed(app: AppState, container: HTMLElement): Promise<number> {
    await until(() => {
      expect(status(container)).toMatch(/triangles/);
      expect(status(container)).not.toMatch(/updating/);
      expect(reliefDialogOf(app).getSnapshot().preview).not.toBeNull();
    });
    return reliefDialogOf(app).getSnapshot().preview!.mesh.indices.length / 3;
  }

  it("opens from the shelf, previews live in the display unit, and creates one undoable closed mesh", async () => {
    const { app, scene, store, container } = await mount((app) => [ribbon(app), ...dialogs(app)]);
    await act(async () => q<HTMLButtonElement>(container, '[data-action="image-to-relief"]').click());
    const dialog = panelOf(container);
    expect(status(dialog)).toMatch(/Choose an image/);
    expect(q<HTMLButtonElement>(dialog, '[data-action="relief-create"]').disabled).toBe(true);

    const png = await greyPng(21, 11, (c) => c / 20);
    await pick(container, [file([png], "ramp.png", "image/png")]);
    const first = await previewed(app, container);
    expect(first).toBe(reliefTriangles(21, 11));
    expect(q(dialog, "[data-relief-image]").textContent).toBe("ramp.png · 21 × 11 px · 16-bit");
    expect(q<HTMLInputElement>(dialog, "[data-relief-name]").value).toBe("ramp");

    await act(async () => app.setUnits("in"));
    expect(dialog.textContent).toContain("Width (in)");
    await act(async () => type(param(dialog, "width"), "2"));
    await act(async () => type(param(dialog, "maxHeight"), "0.5"));
    await act(async () => type(param(dialog, "base"), "0.1"));
    await act(async () => type(q<HTMLInputElement>(dialog, '[data-relief-centre="y"]'), "1"));
    await act(async () => type(q<HTMLInputElement>(dialog, '[data-relief-rotation="x"]'), "-90"));
    await previewed(app, container);
    expect(status(dialog)).toContain("21 × 11 samples · 2 in × 1 in × 0.6 in");
    const preview = reliefDialogOf(app).getSnapshot().preview!;
    expect(preview.centre[1]).toBeCloseTo(25.4, 9);
    expect(preview.rotationDeg).toEqual([-90, 0, 0]);

    await act(async () => q<HTMLButtonElement>(dialog, '[data-action="relief-create"]').click());
    await until(() => expect(app.getSnapshot().items).toHaveLength(1));
    await until(() => expect(container.querySelector("[data-panel=image-to-relief]")).toBeNull());
    const item = app.getSnapshot().items[0];
    expect(item).toMatchObject({ name: "ramp", triangles: reliefTriangles(21, 11) });
    expect(store.getSnapshot().selection).toBe(item.mesh);
    const m = scene.inspect.measure(item.mesh);
    expect([m.closed, m.boundaryLoops, m.nonManifoldEdges, m.orientationConflicts]).toEqual([true, 0, 0, 0]);
    // 50.8 by 25.4 mm, a ramp to 12.7 mm on a 2.54 mm base.
    expect(m.volume).toBeCloseTo(50.8 * 25.4 * (12.7 / 2 + 2.54), 0);
    // Turned to face up, at the centre, the ramp runs along +x and rises.
    const map = scene.heightMap(item.mesh, { direction: [0, -1, 0], resolution: 21 });
    const row = Math.floor(map.frame.height / 2) * map.frame.width;
    const at = (col: number) => map.frame.origin[1] + map.heights[row + col];
    expect(at(19)).toBeGreaterThan(at(1) + 10);
    expect(at(10)).toBeCloseTo(25.4 + 6.35, 0);

    expect(scene.canUndo()).toBe(true);
    await act(async () => store.undo());
    expect(scene.canUndo()).toBe(false);
    expect(scene.has(item.mesh)).toBe(false);
  });

  it("takes the size, height and place from a height map's JSON, and says when it is not one", async () => {
    const { app, container } = await mount(dialogs);
    await act(async () => reliefDialogOf(app).open());
    const png = await greyPng(21, 11, (c) => c / 20);
    const frame = {
      width: 21,
      height: 11,
      origin: [5, 0, 5] as [number, number, number],
      right: [1, 0, 0] as [number, number, number],
      up: [0, 0, -1] as [number, number, number],
      normal: [0, 1, 0] as [number, number, number],
      pixelSize: 2,
      range: 6,
    };
    const json = file([JSON.stringify(heightMapFile(frame, "ramp.png", ["x"]))], "ramp.json", "application/json");
    await pick(container, [file([png], "ramp.png", "image/png"), json]);
    await previewed(app, container);
    const dialog = panelOf(container);
    expect(q(dialog, "[data-relief-note]").textContent).toMatch(/from the height map/);
    expect(param(dialog, "width").value).toBe("40");
    expect(param(dialog, "maxHeight").value).toBe("6");
    expect(q<HTMLInputElement>(dialog, '[data-relief-centre="x"]').value).toBe("25");
    expect(Number(q<HTMLInputElement>(dialog, '[data-relief-centre="z"]').value)).toBe(15);
    expect(reliefDialogOf(app).getSnapshot().preview!.centre).toEqual([25, 0, 15]);

    // Another image: the JSON does not fit it. No JSON at all: no note.
    const other = await greyPng(7, 7, () => 0.5);
    await pick(container, [file([other], "small.png", "image/png"), json]);
    await until(() => expect(q(dialog, "[data-relief-note]").textContent).toMatch(/for another image \(21 × 11 px\)/));
    await pick(container, [
      file([other], "small.png", "image/png"),
      file(["{}"], "notes.json", "application/json"),
    ]);
    await until(() => expect(q(dialog, "[data-relief-note]").textContent).toMatch(/notes\.json is not a height map/));
    await pick(container, [file([other], "small.png", "image/png")]);
    await until(() => expect(container.querySelector("[data-relief-note]")).toBeNull());
  });

  it("reports unreadable files and bad values, keeps its place in the stack of dialogs, and closes on Escape", async () => {
    const { app, container } = await mount(dialogs);
    await act(async () => reliefDialogOf(app).open());
    const dialog = panelOf(container);
    await pick(container, [file(["mesh"], "part.stl", "")]);
    expect(status(dialog)).toMatch(/Choose an image/);
    await pick(container, [file(["nope"], "nope.png", "image/png")]);
    await until(() => expect(status(dialog)).toMatch(/not an image/));
    expect(q<HTMLButtonElement>(dialog, '[data-action="relief-create"]').disabled).toBe(true);

    await pick(container, [file([await greyPng(9, 9, (c, r) => (c + r) / 16)], "grey.png", "image/png")]);
    await previewed(app, container);
    await act(async () => type(param(dialog, "base"), "0"));
    expect(status(dialog)).toMatch(/base thickness must be above zero/i);
    expect(reliefDialogOf(app).getSnapshot().preview).toBeNull();
    await act(async () => type(param(dialog, "base"), "1"));
    await act(async () => type(param(dialog, "samples"), "10"));
    expect(status(dialog)).toMatch(/whole number from 2 to 9/);

    // The dialogs share a place: opening any closes the others.
    await act(async () => createDialogOf(app).open());
    expect(reliefDialogOf(app).getSnapshot().open).toBe(false);
    await act(async () => reliefDialogOf(app).open());
    expect(createDialogOf(app).getSnapshot().open).toBe(false);
    await act(async () => textDialogOf(app).open());
    expect(reliefDialogOf(app).getSnapshot().open).toBe(false);
    await act(async () => reliefDialogOf(app).open());
    expect(textDialogOf(app).getSnapshot().open).toBe(false);

    await act(async () => {
      panelOf(container).dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(reliefDialogOf(app).getSnapshot()).toMatchObject({ open: false, files: null, preview: null });
    expect(app.getSnapshot().items).toHaveLength(0);
  });

  it("opens on a dropped image, takes another while open, and leaves other files to load", async () => {
    const loaded: File[][] = [];
    const load = (files: Iterable<File>) => void loaded.push([...files]);
    let drop: (files: Iterable<File>) => void = () => {};
    function Drop(props: { app: AppState }) {
      drop = useReliefDrop(props.app, load);
      return null;
    }
    const { app, container } = await mount((app) => [...dialogs(app), createElement(Drop, { app, key: "d" })]);
    const stl = file(["x"], "a.stl", "");
    await act(async () => drop([stl]));
    expect(loaded).toEqual([[stl]]);
    expect(reliefDialogOf(app).getSnapshot().open).toBe(false);

    const first = file([await greyPng(9, 5, (c) => c / 8)], "first.png", "image/png");
    await act(async () => drop([first, stl]));
    expect(loaded).toEqual([[stl], [stl]]);
    await previewed(app, container);
    expect(q(container, "[data-relief-image]").textContent).toContain("first.png · 9 × 5 px");
    const { session } = reliefDialogOf(app).getSnapshot();
    const second = file([await greyPng(5, 9, (_, r) => r / 8)], "second.png", "image/png");
    await act(async () => drop([second]));
    await until(() => expect(q(container, "[data-relief-image]").textContent).toContain("second.png · 5 × 9 px"));
    expect(reliefDialogOf(app).getSnapshot().session).toBe(session);
    expect(loaded).toHaveLength(2);
  });
});

describe("command palette", () => {
  it("opens the dialog and shows the export section", async () => {
    const { app } = await mount(() => []);
    const commands = reliefCommands(app);
    expect(commands.map((c) => c.id)).toEqual(["action:image-to-relief", "action:export-height-map"]);
    commands[0].run();
    expect(reliefDialogOf(app).getSnapshot().open).toBe(true);
    app.setActiveTab("inspect");
    commands[1].run();
    expect(app.getSnapshot().activeTab).toBe("home");
  });
});
