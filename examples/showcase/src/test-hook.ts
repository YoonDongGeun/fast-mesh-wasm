// `window.__showcase`: what scripts/showcase-check.mjs reads in headless
// Chrome. Read-only views of the app; the check drives it through the UI
// (the file input, mouse events on the canvas, the toolbar buttons).
import type { SceneStore } from "fast-mesh-wasm/store";
import * as THREE from "three";
import type { Engine } from "./App";
import type { AppState } from "./scene/app-state";
import { view } from "./viewport/view";

export function installTestHook(store: SceneStore, engine: Engine, app: AppState): void {
  const find = (name: string) => {
    const item = app.getSnapshot().items.find((i) => i.name === name);
    if (item === undefined) {
      throw new Error(`no mesh named ${name}`);
    }
    return item;
  };
  (globalThis as { __showcase?: unknown }).__showcase = {
    store,
    mode: engine.mode,
    threads: () => engine.threads(),
    isolated: engine.isolated,
    /** The listed meshes: name, triangles as loaded, and whether `<FastMesh>` has drawn geometry for it. */
    items: () =>
      app.getSnapshot().items.map((i) => ({
        name: i.name,
        triangles: i.triangles,
        drawn: (view.bindings.get(i.mesh)?.object.geometry.getIndex()?.count ?? 0) > 0,
      })),
    jobs: () => app.getSnapshot().jobs.map((j) => ({ ...j })),
    status: () => {
      const s = store.getSnapshot();
      return { status: s.status, error: s.error?.message, ...s.history };
    },
    selected: () => {
      const sel = store.getSnapshot().selection;
      return app.getSnapshot().items.find((i) => i.mesh === sel)?.name ?? null;
    },
    /** The mesh's placement translation. */
    translation: (name: string) => [...store.scene.placement(find(name).mesh).translation],
    dragging: () => view.dragging,
    /** Where the mesh's origin (its centre when loaded, where the gizmo sits) is on the page, in CSS pixels. */
    screenPoint: (name: string) => {
      if (view.camera === null || view.canvas === null) {
        return null;
      }
      const m = store.scene.matrix4(find(name).mesh);
      const p = new THREE.Vector3(m[12], m[13], m[14]).project(view.camera);
      const r = view.canvas.getBoundingClientRect();
      return { x: r.left + ((p.x + 1) / 2) * r.width, y: r.top + ((1 - p.y) / 2) * r.height };
    },
  };
}
