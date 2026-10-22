// `window.__showcase`: what scripts/showcase-check.mjs reads in headless
// Chrome. Read-only views of the app; the check drives it through the UI
// (the file input, mouse events on the canvas, the toolbar buttons).
import type { SceneStore } from "fast-mesh-wasm/store";
import * as THREE from "three";
import type { Engine } from "./App";
import { inspect } from "./panels/inspect";
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
    /**
     * The gizmo: `null` when there is none, else the mesh it is attached to
     * and the handle under the pointer (`axis`, `null` until it is hovered).
     */
    gizmo: () => {
      const g = view.gizmo;
      if (g === null) {
        return null;
      }
      const item = app.getSnapshot().items.find((i) => view.bindings.get(i.mesh)?.object === g.object);
      return { mesh: item?.name ?? null, axis: g.axis };
    },
    /**
     * The collision panel: the pair (names), each mesh's applied colouring
     * (its target's name, and the range) and status, and the readouts with
     * how often they refreshed (and while the gizmo dragged).
     */
    collision: () => {
      const { a, b } = app.getSnapshot().collision;
      const nameOf = (m: unknown) => app.getSnapshot().items.find((i) => i.mesh === m)?.name ?? null;
      const shown = (m: typeof a) => {
        const st = m === null ? undefined : store.getSnapshot().meshes.get(m);
        const c = st?.display.coloring ?? null;
        return st === undefined ? null : { status: st.status, to: c === null ? null : nameOf(c.distanceTo), range: c?.range ?? null };
      };
      return {
        a: nameOf(a),
        b: nameOf(b),
        coloring: Object.fromEntries(app.getSnapshot().items.map((i) => [i.name, shown(i.mesh)])),
        readouts: inspect.collision.readouts,
        updates: inspect.collision.updates,
        updatesWhileDragging: inspect.collision.updatesWhileDragging,
        settled: inspect.collision.settled,
        warnings: inspect.collision.warnings,
      };
    },
    /** The repair panel: the latest analysis (mesh name and summary), the latest report, the op running. */
    repair: () => ({ ...inspect.repair }),
    /** RGBA8 colours of the mesh's vertices that are drawn: how many are reddish (penetration), white, or blue-green (gap). */
    colorCounts: (name: string) => {
      const v = store.scene.views(find(name).mesh);
      const colors = v.colors();
      const idx = v.indices();
      const used = new Set<number>();
      for (const i of idx) {
        used.add(i);
      }
      const out = { red: 0, white: 0, gap: 0, other: 0 };
      if (colors === null) {
        return null;
      }
      for (const i of used) {
        const [r, g, b] = [colors[4 * i], colors[4 * i + 1], colors[4 * i + 2]];
        if (r > 240 && g > 240 && b > 240) {
          out.white++;
        } else if (r > g + 60 && r > b + 60) {
          out.red++;
        } else if (g > r + 30 || b > r + 30) {
          out.gap++;
        } else {
          out.other++;
        }
      }
      return out;
    },
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
