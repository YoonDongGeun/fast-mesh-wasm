// The Modeling panel's Convex Hull action: the hull of the selected meshes
// (the shift-click multi-selection, or the active mesh) as a new mesh, one
// undo step, with a one-line report. The engine call is
// `engine/workbench-ops.ts`'s `runConvexHull`.
import type { MeshHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { runConvexHull } from "../../../engine/workbench-ops";

/**
 * The meshes a hull is built from: the multi-selection while it starts at the
 * primary (the live ones, in click order), else the active mesh alone.
 */
export function hullSources(
  items: readonly Item[],
  multi: readonly MeshHandle[],
  active: Item | null,
): Item[] {
  if (active !== null && multi.length > 1 && multi[0] === active.mesh) {
    const picked = multi.flatMap((m) => items.filter((i) => i.mesh === m));
    if (picked.length > 0) {
      return picked;
    }
  }
  return active === null ? [] : [active];
}

/** The new mesh's name: after its one source, or the number of sources. */
export function hullMeshName(sources: readonly Item[]): string {
  return sources.length === 1 ? `${sources[0].name} hull` : `Hull of ${sources.length} meshes`;
}

/** The report line for a finished hull. */
export function describeHull(r: { name: string; inputPoints: number; vertices: number; triangles: number }): string {
  return (
    `${r.name}: ${r.vertices.toLocaleString()} vertices, ${r.triangles.toLocaleString()} triangles ` +
    `from ${r.inputPoints.toLocaleString()} points`
  );
}

/**
 * The hull sources and the action, reported through the panel's shared
 * `runAction` (busy state and report line).
 */
export function useConvexHull(args: {
  app: AppState;
  store: SceneStore;
  items: readonly Item[];
  activeItem: Item | null;
  runAction: (label: string, fn: () => Promise<string>) => Promise<void> | void;
}) {
  const { app, store, items, activeItem, runAction } = args;
  const multi = useApp(app, (s) => s.multi);
  const sources = hullSources(items, multi, activeItem);

  const runHull = () => {
    if (sources.length === 0) return;
    const name = hullMeshName(sources);
    void runAction("Convex hull", async () => {
      const r = await runConvexHull({ store, app, name, sources });
      return describeHull({ ...r, name: r.item.name });
    });
  };

  return { hullSources: sources, runHull };
}
