// @vitest-environment jsdom
// The mesh list's Remove button is undoable (#269): undo brings the row back
// with its name, colour, material and place, redo removes it again.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { AppState, type Item } from "../src/app/state/app-state";
import { MeshList } from "../src/features/home/ui/MeshList";
import { selectMesh } from "../src/features/select/ux/selection";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(() =>
  init(readFileSync(resolve(__dirname, "../../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"))),
);

const cleanups: (() => void)[] = [];
afterEach(() => {
  while (cleanups.length > 0) {
    cleanups.pop()?.();
  }
});

function addTetra(store: SceneStore, app: AppState, name: string, x: number): Item {
  return app.addMesh(store, {
    name,
    positions: Float64Array.from([x, 0, 0, x + 1, 0, 0, x, 1, 0, x, 0, 1]),
    indices: Uint32Array.from([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]),
    center: [x + 0.5, 0.5, 0.5],
    radius: 1,
    volumeId: null,
    preset: null,
  });
}

const rows = (root: ParentNode) => [...root.querySelectorAll("li[data-mesh]")].map((li) => li.getAttribute("data-mesh"));

describe("the mesh list's Remove button", () => {
  it("removes a mesh undoably: undo brings the row back with its name, colour, material and place", () => {
    const scene = new Scene();
    const store = createSceneStore(scene);
    const app = new AppState();
    cleanups.push(app.follow(store), () => {
      store.dispose();
      scene.free();
    });
    const [a, b, c] = ["A", "B", "C"].map((n, k) => addTetra(store, app, n, 3 * k));
    app.setMeshShadingOverride(b.mesh, "flat");
    const color = b.material.color.getHex();
    store.setDisplay(b.mesh, { opacity: 0.5 });
    store.setPlacement(b.mesh, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 2, 0] });
    store.commitFrame();
    const disposed = vi.fn();
    b.material.addEventListener("dispose", disposed);
    const list = document.createElement("div");
    document.body.appendChild(list);
    const root = createRoot(list);
    cleanups.push(() => {
      act(() => root.unmount());
      list.remove();
    });
    const none = () => {};
    const props = { onLoad: none, onCancelLoad: none, onLoadPhantom: none, onLoadSample: none, onLoadDemo: none, onFocus: none };
    act(() => root.render(createElement(SceneProvider, { store }, createElement(MeshList, { app, ...props }))));
    expect(rows(list)).toEqual(["A", "B", "C"]);
    act(() => selectMesh(app, (m) => store.select(m), null, b.mesh, false));

    const listed = app.getSnapshot().items[1];
    expect(listed.mesh).toBe(b.mesh);
    act(() => list.querySelector<HTMLButtonElement>('li[data-mesh="B"] button.danger')!.click());
    expect(rows(list)).toEqual(["A", "C"]);
    expect(store.getSnapshot().selection).toBeNull();
    expect(app.getSnapshot().multi).toEqual([]);
    expect(disposed).toHaveBeenCalledTimes(1);
    act(() => {
      store.commitFrame();
    });

    act(() => store.undo());
    expect(rows(list)).toEqual(["A", "B", "C"]);
    const back = app.getSnapshot().items[1];
    expect(back).toBe(listed);
    expect(back.id).toBe(b.id);
    expect(back.material.color.getHex()).toBe(color);
    expect(back.shadingOverride).toBe("flat");
    expect(store.getSnapshot().meshes.get(b.mesh)!.display.opacity).toBe(0.5);
    expect(store.scene.placement(b.mesh).translation).toEqual([0, 2, 0]);
    expect(scene.triangleCount(b.mesh)).toBe(4);
    expect(store.getSnapshot().selection).toBe(b.mesh);
    expect(app.getSnapshot().multi).toEqual([b.mesh]);

    act(() => store.redo());
    expect(rows(list)).toEqual(["A", "C"]);
    act(() => store.undo());
    expect(rows(list)).toEqual(["A", "B", "C"]);
    expect([a, c].every((i) => app.getSnapshot().items.includes(i))).toBe(true);
    // What came before the removal still undoes.
    act(() => store.undo());
    expect(store.scene.placement(b.mesh).translation).toEqual([0, 0, 0]);
  });
});
