// @vitest-environment jsdom
// The status bar's notice that an action is permanent (#281): it comes with
// the mesh list's Remove button, and goes when dismissed or after a while.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore } from "fast-mesh-wasm/store";
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import type { Engine } from "../src/app/App";
import { AppState } from "../src/app/state/app-state";
import { HISTORY_NOTICE_MS } from "../src/app/ux/history-notice";
import { MeshList } from "../src/features/home/ui/MeshList";
import { StatusBar } from "../src/shell/ui/StatusBar";

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

describe("the permanent action notice", () => {
  it("shows in the status bar, and goes when dismissed or after a while", () => {
    vi.useFakeTimers();
    cleanups.push(() => vi.useRealTimers());
    const scene = new Scene({ historyBudget: 64 });
    const store = createSceneStore(scene);
    const app = new AppState();
    cleanups.push(app.follow(store), () => {
      store.dispose();
      scene.free();
    });
    for (const name of ["A", "B"]) {
      app.addMesh(store, {
        name,
        positions: Float64Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
        indices: Uint32Array.from([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]),
        center: [0.5, 0.5, 0.5],
        radius: 1,
        volumeId: null,
        preset: null,
      });
    }
    const engine = { mode: "local", scene, threads: () => 1, isolated: false } as unknown as Engine;
    const root = document.createElement("div");
    document.body.appendChild(root);
    const react = createRoot(root);
    cleanups.push(() => {
      act(() => react.unmount());
      root.remove();
    });
    const none = () => {};
    const props = { onLoad: none, onCancelLoad: none, onLoadPhantom: none, onLoadSample: none, onLoadDemo: none, onFocus: none };
    act(() =>
      react.render(
        createElement(
          SceneProvider,
          { store },
          createElement(MeshList, { app, ...props }),
          createElement(StatusBar, { engine, app }),
        ),
      ),
    );
    const shown = () => root.querySelector("[data-history-notice]");
    const remove = (name: string) =>
      act(() => root.querySelector<HTMLButtonElement>(`li[data-mesh="${name}"] button.danger`)!.click());
    expect(shown()).toBeNull();
    remove("A");
    expect(shown()?.textContent).toContain('Removing "A" cannot be undone');
    // In the bar's live region, so it is announced.
    expect(shown()?.closest('[role="status"]')).not.toBeNull();
    act(() => root.querySelector<HTMLButtonElement>('[data-action="dismiss-history-notice"]')!.click());
    expect(shown()).toBeNull();

    remove("B");
    expect(shown()?.textContent).toContain('Removing "B" cannot be undone');
    act(() => vi.advanceTimersByTime(HISTORY_NOTICE_MS - 1));
    expect(shown()).not.toBeNull();
    act(() => vi.advanceTimersByTime(1));
    expect(shown()).toBeNull();
    expect(store.getSnapshot().history.canUndo).toBe(false);
  });
});
