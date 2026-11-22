// The deform tools' shared state (#193): the `wave6.deform` slice, the sculpt
// mask that protects, the mesh they work on (the selection, else the first
// item), and the edits that write the slice back.
import type { MeshHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useState, useSyncExternalStore } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import type { Wave6DeformState } from "../../../app/state/wave6-state";

/** Resolves the active mesh from SceneStore selection (falling back to the first loaded item). */
export function resolveActiveDeformMesh(store: SceneStore, items: readonly Item[]): MeshHandle | null {
  const selection = store.getSnapshot().selection;
  if (selection !== null) {
    return selection;
  }
  return items.length > 0 ? items[0].mesh : null;
}

/** Merges `patch` into `wave6.deform`. */
export function patchDeform(app: AppState, patch: Partial<Wave6DeformState>): void {
  app.updateWave6((prev) => ({
    ...prev,
    deform: { ...prev.deform, ...patch },
  }));
}

/** Merges what `update` derives from the current `wave6.deform` into it. */
export function updateDeform(
  app: AppState,
  update: (prev: Wave6DeformState) => Partial<Wave6DeformState>,
): void {
  app.updateWave6((prev) => ({
    ...prev,
    deform: { ...prev.deform, ...update(prev.deform) },
  }));
}

/** The `wave6.deform` slice, the items, the sculpt mask, and the active mesh: the selection, else the first item. */
export function useDeformState(app: AppState, store: SceneStore) {
  const deform = useApp(app, (s) => s.wave6.deform);
  const items = useApp(app, (s) => s.items);
  const sculptMask = useApp(app, (s) => s.sculptMask);

  const selection = useSyncExternalStore(
    store.subscribe,
    () => store.getSnapshot().selection,
    () => store.getSnapshot().selection,
  );
  const activeMesh: MeshHandle | null = selection ?? items[0]?.mesh ?? null;
  return { deform, items, sculptMask, activeMesh };
}

/** Counts the store's changes, so what reads live vertex positions or matrices reads them again. */
export function useSceneRevision(store: SceneStore): number {
  const [sceneRev, setSceneRev] = useState(0);

  useEffect(() => {
    return store.subscribe(() => setSceneRev((n) => n + 1));
  }, [store]);
  return sceneRev;
}
