// Face selection's scene and app calls (#233): which mesh the tools act
// on, where a selection is stored (`wave6.select.selectedByMesh`, per item
// id), raycasts that skip hidden meshes, and the screen-shape selection
// through `store.scene.repair`.
import type { MeshHandle, RayHit, ScreenSelectOptions, ScreenSelectionShape, Vec3 } from "fast-mesh-wasm";
import { meshState, type SceneStore } from "fast-mesh-wasm/store";
import type { AppState, Item } from "../../../app/state/app-state";
import type { Wave6SelectState } from "../../../app/state/wave6-state";
import { combineFaceSelection, EMPTY_U32 } from "./face-sets";

const RAYCAST_HIDDEN_SKIP_LIMIT = 8;

function isItemVisible(store: SceneStore, mesh: MeshHandle): boolean {
  const st = meshState(store.getSnapshot(), mesh);
  return st ? st.desired.visible : true;
}

/** Resolves the active item (primary selection when present, else first visible item). */
export function resolveActiveSelectItem(app: AppState, store: SceneStore): Item | null {
  const items = app.getSnapshot().items;
  const primary = store.getSnapshot().selection;
  if (primary !== null) {
    const matched = items.find((it) => it.mesh === primary);
    if (matched) {
      return matched;
    }
  }
  return items.find((it) => isItemVisible(store, it.mesh)) ?? items[0] ?? null;
}

/** Commits a new triangle selection array for `itemId` into `wave6.select.selectedByMesh`. */
export function writeMeshFaceSelection(app: AppState, itemId: number, selected: Uint32Array): void {
  app.updateWave6((prev) => ({
    ...prev,
    select: {
      ...prev.select,
      selectedByMesh: {
        ...prev.select.selectedByMesh,
        [itemId]: selected,
      },
    },
  }));
}

/** Raycasts visible scene meshes through `store.scene.inspect.raycast`, skipping hidden meshes. */
export async function raycastVisibleItem(
  store: SceneStore,
  items: readonly Item[],
  origin: Readonly<Vec3>,
  direction: Readonly<Vec3>,
  backfaceCull: boolean,
): Promise<{ item: Item; hit: RayHit } | null> {
  const inspect = store.scene.inspect;
  if (!inspect) {
    return null;
  }
  const byMesh = new Map<MeshHandle, Item>();
  for (const item of items) {
    if (isItemVisible(store, item.mesh)) {
      byMesh.set(item.mesh, item);
    }
  }
  if (byMesh.size === 0) {
    return null;
  }

  let minDistance = 0;
  for (let step = 0; step < RAYCAST_HIDDEN_SKIP_LIMIT; step++) {
    const hit = await Promise.resolve(
      inspect.raycast(origin, direction, {
        minDistance,
        cull: backfaceCull ? "back" : "none",
      }),
    );
    if (!hit) {
      return null;
    }
    const item = byMesh.get(hit.mesh);
    if (item) {
      return { item, hit };
    }
    minDistance = hit.distance + 1e-4;
  }
  return null;
}

/** Runs `repair.selectFacesScreen`, combines with `mode`, and stores into `app.wave6.select`. */
export async function applyScreenFaceSelection(params: {
  app: AppState;
  store: SceneStore;
  item: Item;
  shape: ScreenSelectionShape;
  options: ScreenSelectOptions;
  combineMode: Wave6SelectState["combineMode"];
}): Promise<Uint32Array> {
  const repair = params.store.scene.repair;
  if (!repair) {
    return EMPTY_U32;
  }
  const incoming = await Promise.resolve(
    repair.selectFacesScreen(params.item.mesh, params.shape, params.options),
  );
  const current = params.app.getSnapshot().wave6.select.selectedByMesh[params.item.id] ?? EMPTY_U32;
  const combined = combineFaceSelection(current, incoming, params.combineMode);
  writeMeshFaceSelection(params.app, params.item.id, combined);
  return combined;
}
